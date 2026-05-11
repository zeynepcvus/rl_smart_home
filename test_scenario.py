from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.scenario import build_daily_scenario
from environment.slots import SlotManager
from environment.devices import (
    create_device_from_preset,
    create_custom_device,
    DEVICE_TYPE_SHIFTABLE,
)
from environment.smart_home_env import SmartHomeEnv, RewardWeights

MODEL_DIR = "models_balanced_dynamic_v6"


def build_scenario():
    rng = np.random.default_rng(42)
    return build_daily_scenario(rng)


def make_env(slot_manager: SlotManager, scenario):
    return SmartHomeEnv(
        slot_manager=slot_manager,
        temp_min=20.0,
        temp_max=24.0,
        scenario=scenario,
        reward_weights=RewardWeights(cost=0.42, comfort=0.38, task=0.20),
    )


def run_scenario(name: str, slot_manager: SlotManager, scenario, model: PPO) -> None:
    print(f"\n{'=' * 60}")
    print(f"SENARYO: {name}")
    for i, d in enumerate(slot_manager.slots):
        if d.device_type > 0:  # DEVICE_TYPE_EMPTY = 0
            print(f"  Slot {i}: {d.name}")
    print(f"{'=' * 60}")

    raw_env = make_env(slot_manager, scenario)
    dummy = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(f"{MODEL_DIR}/vec_normalize.pkl", dummy)
    vec_env.training = False
    vec_env.norm_reward = False
    model.set_env(vec_env)

    obs = vec_env.reset()
    inner_env = vec_env.venv.envs[0]

    done = False
    step_info: dict = {}
    while not done:
        hour = inner_env.current_hour
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, info = vec_env.step(action)
        done = bool(dones[0])
        step_info = info[0]

        active = [d.name for d in inner_env.slot_manager.slots if d.is_active]
        print(
            f"Saat {hour:02d} | "
            f"Aksiyon: {list(action[0])} | "
            f"Aktif: {active if active else ['-']} | "
            f"İç Sıcaklık: {step_info['indoor_temp']:.1f}°C | "
            f"Fiyat: {step_info['current_price']:.3f} TL/kWh | "
            f"Maliyet: {step_info['last_step_cost']:.4f} TL"
        )

    print("\n--- Özet ---")
    print(f"Toplam maliyet:    {step_info['total_cost']:.2f} TL")
    print(f"Konfor ihlali:     {step_info['comfort_violations']}")
    print(f"Deadline ihlali:   {step_info['deadline_violations']}")
    print(f"Geçersiz aksiyon:  {step_info['invalid_action_count']}")


def main() -> None:
    scenario = build_scenario()

    # Modeli bir kez yükle, her senaryo için set_env ile güncelle
    sm_init = SlotManager()
    sm_init.add_device(create_device_from_preset("HVAC"))
    sm_init.add_device(create_device_from_preset("Lighting"))
    env_init = make_env(sm_init, scenario)
    dummy_init = DummyVecEnv([lambda: env_init])
    vec_init = VecNormalize.load(f"{MODEL_DIR}/vec_normalize.pkl", dummy_init)
    vec_init.training = False
    vec_init.norm_reward = False
    model = PPO.load(f"{MODEL_DIR}/best_model/best_model", env=vec_init)

    # SENARYO 1: Standart 3 cihaz
    sm1 = SlotManager()
    sm1.add_device(create_device_from_preset("HVAC"))
    sm1.add_device(create_device_from_preset("Lighting"))
    sm1.add_device(create_device_from_preset("Washing Machine"))
    run_scenario("Standart 3 Cihaz (HVAC + Lighting + Washing Machine)", sm1, scenario, model)

    # SENARYO 2: Custom shiftable — Fırın
    sm2 = SlotManager()
    sm2.add_device(create_device_from_preset("HVAC"))
    sm2.add_device(create_device_from_preset("Lighting"))
    sm2.add_device(create_custom_device(
        "Fırın",
        device_type=DEVICE_TYPE_SHIFTABLE,
        power_kw=2.0,
        duration=1,
        deadline=20,
        user_priority=0.7,
    ))
    run_scenario("Custom Shiftable — Fırın (2.0 kW, deadline=20)", sm2, scenario, model)

    # SENARYO 3: 4 cihaz
    sm3 = SlotManager()
    sm3.add_device(create_device_from_preset("HVAC"))
    sm3.add_device(create_device_from_preset("Lighting"))
    sm3.add_device(create_device_from_preset("Washing Machine"))
    sm3.add_device(create_device_from_preset("Dishwasher"))
    run_scenario("4 Cihaz (HVAC + Lighting + Washing Machine + Dishwasher)", sm3, scenario, model)

    # SENARYO 4: Custom — Bulaşık Makinesi
    sm4 = SlotManager()
    sm4.add_device(create_device_from_preset("HVAC"))
    sm4.add_device(create_device_from_preset("Lighting"))
    sm4.add_device(create_custom_device(
        "Bulaşık Makinesi",
        device_type=DEVICE_TYPE_SHIFTABLE,
        power_kw=1.8,
        duration=2,
        deadline=22,
        user_priority=0.6,
    ))
    run_scenario("Custom Shiftable — Bulaşık Makinesi (1.8 kW, deadline=22)", sm4, scenario, model)


if __name__ == "__main__":
    main()
