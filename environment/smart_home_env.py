from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.devices import DEVICE_TYPE_CONTINUOUS, DEVICE_TYPE_EMPTY, DEVICE_TYPE_SHIFTABLE
from environment.pricing import get_price_category_from_value
from environment.scenario import DailyScenario, build_daily_scenario
from environment.slots import SlotManager

CATEGORY_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """
    Ödül fonksiyonundaki ağırlıklar.
    cost: maliyet cezasının ağırlığı
    comfort: konfor ödülünün ağırlığı
    task: görev tamamlama ödülünün ağırlığı
    """
    cost: float = 0.42
    comfort: float = 0.38
    task: float = 0.20


class SmartHomeEnv(gym.Env):
    """
    Akıllı ev enerji yönetimi simülasyon ortamı.
    1 episode = 1 gün = 24 saat.
    Her saat ajan karar verir, ortam güncellenir, ödül hesaplanır.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        slot_manager: SlotManager,
        *,
        temp_min: float = 20.0,
        temp_max: float = 24.0,
        scenario: DailyScenario | None = None,
        reward_weights: RewardWeights | None = None,
        episode_hours: int = 24,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.slot_manager = slot_manager
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.episode_hours = episode_hours
        self.reward_weights = reward_weights or RewardWeights()

        self._rng = np.random.default_rng(seed)
        self._fixed_scenario = scenario

        # Durum değişkenleri — reset() içinde sıfırlanır
        self.current_hour = 0
        self.current_price = 0.0
        self.previous_price = 0.0
        self.outdoor_temp = 15.0
        self.indoor_temp = 20.0
        self.previous_indoor_temp = 20.0

        self.total_cost = 0.0
        self.comfort_violations = 0
        self.deadline_violations = 0
        self.invalid_action_count = 0
        self.hvac_switch_count = 0

        self._step_hvac_switches = 0
        self._step_invalid_actions = 0

        self.scenario: DailyScenario | None = None

        # MAX_SLOTS yerine slot_manager'dan al — tutarlılık için kritik
        n_slots = self.slot_manager.max_slots

        self.action_space = spaces.MultiBinary(n_slots)

        general_dim = 12
        slot_dim = 9
        obs_dim = general_dim + n_slots * slot_dim
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def _is_user_awake(self) -> bool:
        assert self.scenario is not None
        return self.scenario.awake_start <= self.current_hour < self.scenario.sleep_start

    def _is_night(self) -> bool:
        return self.current_hour >= 18 or self.current_hour < 7

    def _lighting_need(self) -> float:
        assert self.scenario is not None
        return float(self.scenario.lighting_need_profile[self.current_hour % 24])

    def _occupancy(self) -> float:
        assert self.scenario is not None
        return float(self.scenario.occupancy_profile[self.current_hour % 24])

    def _is_hvac_device(self, device) -> bool:
        return device.device_type == DEVICE_TYPE_CONTINUOUS and device.comfort_sensitive

    def _is_lighting_device(self, device) -> bool:
        return device.device_type == DEVICE_TYPE_CONTINUOUS and device.lighting_sensitive

    def _get_outdoor_temp_for_hour(self, hour: int) -> float:
        assert self.scenario is not None
        angle = 2 * np.pi * ((hour - 8) / 24.0)
        temp = self.scenario.daily_temp_base + self.scenario.daily_temp_amplitude * np.sin(angle)
        return float(np.clip(temp, -10.0, 45.0))

    def _hours_left(self) -> int:
        return max(self.episode_hours - self.current_hour - 1, 0)

    def set_scenario(self, scenario: DailyScenario) -> None:
        """Dışarıdan sabit senaryo set et (paired evaluation için kullanılır)"""
        self._fixed_scenario = scenario

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        scenario = self._fixed_scenario if self._fixed_scenario is not None else build_daily_scenario(self._rng)
        self.scenario = scenario

        self.current_hour = 0
        self.total_cost = 0.0
        self.comfort_violations = 0
        self.deadline_violations = 0
        self.invalid_action_count = 0
        self.hvac_switch_count = 0
        self._step_hvac_switches = 0
        self._step_invalid_actions = 0

        self.outdoor_temp = self._get_outdoor_temp_for_hour(self.current_hour)
        self.indoor_temp = float(np.clip(scenario.initial_indoor_temp, -10.0, 45.0))
        self.previous_indoor_temp = self.indoor_temp

        self.current_price = float(scenario.price_profile[self.current_hour])
        self.previous_price = self.current_price

        self.slot_manager.reset_all()

        return self._get_obs(), self._build_info(last_step_cost=0.0)

    def _get_obs(self) -> np.ndarray:
        assert self.scenario is not None

        active_power = sum(d.get_power() for d in self.slot_manager.slots)
        price_delta = self.current_price - self.previous_price
        temp_delta = self.indoor_temp - self.previous_indoor_temp

        general = [
            # Terminal state'de (current_hour=24) 1.0'ı aşmasın — obs_space sınırı
            np.clip(self.current_hour / max(self.episode_hours - 1, 1), 0.0, 1.0),
            np.clip(self.current_price / 5.0, 0.0, 1.0),
            np.clip(self.previous_price / 5.0, 0.0, 1.0),
            np.clip((price_delta + 1.0) / 2.0, 0.0, 1.0),
            np.clip((self.outdoor_temp + 10.0) / 55.0, 0.0, 1.0),
            np.clip((self.indoor_temp + 10.0) / 55.0, 0.0, 1.0),
            np.clip((temp_delta + 3.0) / 6.0, 0.0, 1.0),
            float(self._is_user_awake()),
            float(self._is_night()),
            self._occupancy(),
            self._lighting_need(),
            np.clip(active_power / 15.0, 0.0, 1.0),
        ]

        slot_features: list[float] = []
        for device in self.slot_manager.slots:
            deadline_remaining = 0.0
            deadline_urgent = 0.0

            if device.deadline is not None:
                remaining = max(device.deadline - self.current_hour, 0)
                deadline_remaining = np.clip(remaining / 23.0, 0.0, 1.0)
                slack = remaining - max(device.duration or 0, 1)
                # Reward ile tutarlı: reward slack<=2 için ceza veriyor
                deadline_urgent = 1.0 if slack <= 2 else 0.0

            slot_features.extend(
                [
                    float(device.is_active),
                    float(device.is_completed),
                    device.device_type / 2.0,
                    CATEGORY_TO_INDEX[device.category] / 4.0 if device.category is not None else 0.0,
                    np.clip(device.power_kw / 10.0, 0.0, 1.0),
                    np.clip(device.remaining_hours / 24.0, 0.0, 1.0),
                    deadline_remaining,
                    deadline_urgent,
                    float(device.usage_profile.user_priority),
                ]
            )

        return np.array(general + slot_features, dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)

        self._step_hvac_switches = 0
        self._step_invalid_actions = 0

        self.previous_indoor_temp = self.indoor_temp
        self.previous_price = self.current_price

        # 1) AKSİYONLARI UYGULA
        for i, device in enumerate(self.slot_manager.slots):
            desired = int(action[i])

            if device.device_type == DEVICE_TYPE_EMPTY:
                if desired == 1:
                    self._step_invalid_actions += 1
                    self.invalid_action_count += 1
                continue

            if device.device_type == DEVICE_TYPE_SHIFTABLE:
                if desired == 1:
                    started = device.start()
                    if not started:
                        self._step_invalid_actions += 1
                        self.invalid_action_count += 1
                continue

            if device.device_type == DEVICE_TYPE_CONTINUOUS:
                was_active = device.is_active
                if desired == 1:
                    device.turn_on()
                else:
                    device.turn_off()
                if self._is_hvac_device(device) and was_active != device.is_active:
                    self._step_hvac_switches += 1
                    self.hvac_switch_count += 1

        # 2) DEADLINE KONTROLÜ — cihazlar ilerlemeden önce yap
        self._update_deadline_state()

        # 3) ISIL DURUMU GÜNCELLE — bu saatin aktif cihazlarına göre (step_all öncesi)
        self._update_thermal_state()

        # 4) ÖDÜLÜ HESAPLA — step_all öncesi, o saatin gerçek maliyetini yakalar.
        # Son çalışma saatinde cihaz henüz aktif → maliyet doğru hesaplanır.
        reward, step_cost, comfort_ok = self._calculate_reward()

        # 5) İSTATİSTİKLERİ GÜNCELLE
        self.total_cost += step_cost
        if not comfort_ok:
            self.comfort_violations += 1

        # 6) SİMÜLASYONU İLERLET — cihazlar bir saat ilerler (tamamlananlar pasif olur)
        self.slot_manager.step_all()

        # 7) SONRAKİ SAATE GEÇ
        # Raporlama için bu step'in saat ve fiyatını kaydet (sonra güncelleniyor)
        processed_hour = self.current_hour
        processed_price = self.current_price

        self.current_hour += 1
        terminated = self.current_hour >= self.episode_hours

        if not terminated:
            self.outdoor_temp = self._get_outdoor_temp_for_hour(self.current_hour)
            assert self.scenario is not None
            self.current_price = float(self.scenario.price_profile[self.current_hour])

        info = self._build_info(
            last_step_cost=step_cost,
            step_hour=processed_hour,
            step_price=processed_price,
        )
        # Terminal obs da gerçek state — SB3 value bootstrap için kullanır
        observation = self._get_obs()
        return observation, reward, terminated, False, info

    def _update_thermal_state(self) -> None:
        hvac_on = any(self._is_hvac_device(d) and d.is_active for d in self.slot_manager.slots)
        target = (self.temp_min + self.temp_max) / 2.0

        if hvac_on:
            thermal_power = sum(
                d.power_kw for d in self.slot_manager.slots
                if self._is_hvac_device(d) and d.is_active
            )
            hvac_gain = min(0.18 + 0.06 * thermal_power, 0.38)
            self.indoor_temp += (target - self.indoor_temp) * hvac_gain
        else:
            occupancy_heat = 0.15 * self._occupancy()
            drift_rate = 0.05 if self._is_night() else 0.08
            self.indoor_temp += (self.outdoor_temp - self.indoor_temp) * drift_rate + occupancy_heat

        self.indoor_temp += float(self._rng.normal(0.0, 0.05))
        self.indoor_temp = float(np.clip(self.indoor_temp, -10.0, 45.0))

    def _update_deadline_state(self) -> None:
        """
        Deadline kaçırıldı mı?
        Cihazlar step ilerlemeden ÖNCE çağrılır.
        "Deadline geçti mi" değil, "artık zamanında bitirmek imkansız mı" diye bakar.

        Kritik: cihaz zaten çalışıyorsa sadece remaining_hours kadar pencere gerekir,
        duration değil. Örnek: duration=2, deadline=22, saat 20'de başlarsa
        remaining_hours=2 → deadline=22, 2 saat içinde biter → miss sayılmaz.
        """
        for device in self.slot_manager.slots:
            if device.device_type != DEVICE_TYPE_SHIFTABLE:
                continue
            if device.deadline is None:
                continue
            if device.is_completed or device.deadline_missed:
                continue

            remaining_to_deadline = device.deadline - self.current_hour

            if device.is_active:
                # Cihaz çalışıyor: sadece kalan süresi kadar pencere gerekir
                required_window = max(device.remaining_hours, 1)
            else:
                # Cihaz başlamamış: tam duration kadar pencere gerekir
                required_window = max(device.duration or 0, 1)

            if remaining_to_deadline < required_window:
                device.deadline_missed = True
                self.deadline_violations += 1

    def _calculate_reward(self) -> tuple[float, float, bool]:
        # 1) MALİYET
        total_power = sum(d.get_power() for d in self.slot_manager.slots)
        cost = total_power * self.current_price
        cost_reward = -cost

        # 2) STEP BAZLI CEZALAR
        invalid_penalty = -0.20 * self._step_invalid_actions
        switch_penalty = -0.04 * self._step_hvac_switches

        # 3) KONFOR
        target = (self.temp_min + self.temp_max) / 2.0

        if self.temp_min <= self.indoor_temp <= self.temp_max:
            deviation = abs(self.indoor_temp - target)
            temp_score = 1.2 - min(deviation / 2.5, 0.8)
            temp_ok = True
        else:
            deviation = min(abs(self.indoor_temp - target), 8.0)
            temp_score = -(1.8 + deviation / 1.5)
            temp_ok = False

        light_active = any(self._is_lighting_device(d) and d.is_active for d in self.slot_manager.slots)
        lighting_need = self._lighting_need()

        if lighting_need >= 0.55 and not light_active:
            light_score = -1.4 * lighting_need
            light_ok = False
        elif lighting_need < 0.20 and light_active:
            light_score = -0.30
            light_ok = True
        else:
            light_score = 0.25 if light_active and lighting_need >= 0.55 else 0.05
            light_ok = True

        comfort_ok = temp_ok and light_ok
        comfort_reward = temp_score + light_score

        # 4) GÖREV ÖDÜLÜ
        task_reward = 0.0
        for device in self.slot_manager.slots:
            if device.device_type != DEVICE_TYPE_SHIFTABLE:
                continue

            priority = max(device.usage_profile.user_priority, 0.1)

            if device.is_completed:
                task_reward += 0.25 * priority
                continue

            if device.deadline is not None:
                remaining_to_deadline = device.deadline - self.current_hour
                required_window = max(device.duration or 0, 1)
                slack = remaining_to_deadline - required_window

                if not device.is_active and slack <= 2:
                    task_reward -= 0.25 * priority
                if not device.is_active and slack <= 0:
                    task_reward -= 0.9 * priority
                if device.deadline_missed:
                    task_reward -= 2.0 * priority

            in_preferred_window = (
                device.usage_profile.preferred_start_hour is not None
                and device.usage_profile.preferred_end_hour is not None
                and device.usage_profile.preferred_start_hour <= self.current_hour < device.usage_profile.preferred_end_hour
            )
            if in_preferred_window and device.is_active:
                task_reward += 0.08 * priority

        # 5) TOPLAM REWARD
        total_reward = (
            self.reward_weights.cost * cost_reward
            + self.reward_weights.comfort * comfort_reward
            + self.reward_weights.task * task_reward
            + invalid_penalty
            + switch_penalty
        )

        return total_reward, cost, comfort_ok

    def _build_info(
        self,
        *,
        last_step_cost: float,
        step_hour: int | None = None,
        step_price: float | None = None,
    ) -> dict:
        # step() işlenen saati geçer; reset() geçmez (current_hour=0 doğru)
        reported_hour = step_hour if step_hour is not None else self.current_hour
        reported_price = step_price if step_price is not None else self.current_price
        return {
            "hour": reported_hour,
            "current_price": reported_price,
            "price_category": get_price_category_from_value(reported_price),
            "last_step_cost": last_step_cost,
            "total_cost": self.total_cost,
            "comfort_violations": self.comfort_violations,
            "deadline_violations": self.deadline_violations,
            "indoor_temp": self.indoor_temp,
            "outdoor_temp": self.outdoor_temp,
            "invalid_action_count": self.invalid_action_count,
            "step_invalid_actions": self._step_invalid_actions,
            "hvac_switch_count": self.hvac_switch_count,
            "step_hvac_switches": self._step_hvac_switches,
            # reported_hour üzerinden doğrudan profil oku — self.current_hour değil.
            # _occupancy() ve _lighting_need() self.current_hour kullanır; step()
            # sonrasında current_hour artırıldığından bir sonraki saatin değeri dönerdi.
            "occupancy": float(self.scenario.occupancy_profile[reported_hour % 24]) if self.scenario is not None and reported_hour < self.episode_hours else 0.0,
            "lighting_need": float(self.scenario.lighting_need_profile[reported_hour % 24]) if self.scenario is not None and reported_hour < self.episode_hours else 0.0,
        }