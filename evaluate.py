from __future__ import annotations

import random

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.rule_based_agent import RuleBasedAgent
from environment.devices import (
    create_device_from_preset,
    create_custom_device,
    DEVICE_TYPE_SHIFTABLE,
    ENERGY_CATEGORIES,
)
from environment.scenario import DailyScenario, build_daily_scenario
from environment.slots import SlotManager
from environment.smart_home_env import RewardWeights, SmartHomeEnv


def make_env_with_scenario(scenario: DailyScenario):
    """
    Verilen senaryo için değerlendirme ortamı oluştur.
    Her iki ajan da aynı cihazlarla ve aynı senaryoyla test edilir.
    """
    slot_manager = SlotManager()
    slot_manager.add_device(create_device_from_preset("HVAC"))
    slot_manager.add_device(create_device_from_preset("Lighting"))  # önce Lighting
    slot_manager.add_device(create_device_from_preset("Washing Machine"))  # sonra Washing

    return SmartHomeEnv(
        slot_manager=slot_manager,
        temp_min=20.0,
        temp_max=24.0,
        scenario=scenario,
        reward_weights=RewardWeights(cost=0.20, comfort=0.60, task=0.20),
        dynamic_device_training=False,
    )


def make_env_dynamic(scenario: DailyScenario, seed: int = 0):
    rng = random.Random(seed)

    slot_manager = SlotManager()
    slot_manager.add_device(create_device_from_preset("HVAC"))
    slot_manager.add_device(create_device_from_preset("Lighting"))

    SHIFTABLE_PRESETS = [
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Water Heater",
        "EV Charger",
    ]
    used_presets: list[str] = []

    for slot_index in range(2, 5):
        roll = rng.random()
        if roll < 0.40:
            pass  # empty
        elif roll < 0.90:
            available = [p for p in SHIFTABLE_PRESETS if p not in used_presets]
            if available:
                chosen = rng.choice(available)
                used_presets.append(chosen)
                slot_manager.add_device(create_device_from_preset(chosen))
        else:
            category = rng.choice(["A", "B", "C", "D"])
            duration = rng.randint(1, 4)
            deadline = rng.randint(12, 23)
            priority = round(rng.uniform(0.3, 0.9), 2)
            slot_manager.add_device(create_custom_device(
                name=f"CustomDevice_{slot_index}",
                device_type=DEVICE_TYPE_SHIFTABLE,
                power_kw=ENERGY_CATEGORIES[category],
                duration=duration,
                deadline=deadline,
                user_priority=priority,
            ))

    return SmartHomeEnv(
        slot_manager=slot_manager,
        temp_min=20.0,
        temp_max=24.0,
        scenario=scenario,
        reward_weights=RewardWeights(cost=0.42, comfort=0.38, task=0.20),
    )


def build_shared_scenarios(n_episodes: int, seed: int = 2026) -> list[DailyScenario]:
    """
    Her iki ajan için ortak senaryolar oluştur.
    Aynı seed kullanıldığı için RL ve kural tabanlı ajan
    tamamen aynı günlük koşullarla karşılaşır — adil karşılaştırma.
    """
    rng = np.random.default_rng(seed)
    return [build_daily_scenario(rng) for _ in range(n_episodes)]


def evaluate_rl_agent(scenarios: list[DailyScenario]):
    """
    Eğitilmiş PPO ajanını verilen senaryolar üzerinde değerlendir.
    Her senaryoda bir episode çalıştırılır, metrikler toplanır.
    """
    print("\n=== RL AGENT EVALUATION ===")

    # Modeli yükle — normalizasyon istatistikleri de yükleniyor
    dummy_env = DummyVecEnv([lambda: make_env_with_scenario(scenarios[0])])
    env = VecNormalize.load("models_balanced_dynamic_v6/vec_normalize.pkl", dummy_env)
    env.training = False       # normalizasyon istatistikleri güncellenmeyecek
    env.norm_reward = False    # reward normalize edilmeyecek (gerçek değerleri görmek için)

    model = PPO.load("models_balanced_dynamic_v6/best_model/best_model", env=env)

    # Her episode için toplanacak metrikler
    metrics = {
        "cost": [],
        "comfort_violations": [],
        "deadline_violations": [],
        "reward": [],
        "hvac_switch_count": [],
        "invalid_action_count": [],
    }

    for scenario in scenarios:
        # Her senaryoyu ortama set et ve başlat
        env.envs[0].set_scenario(scenario)
        obs = env.reset()
        total_reward = 0.0
        done = False
        info = None

        while not done:
            # Deterministik tahmin — en yüksek olasılıklı aksiyonu seç
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])

        # Episode sonu metriklerini kaydet
        last_info = info[0]
        metrics["cost"].append(last_info["total_cost"])
        metrics["comfort_violations"].append(last_info["comfort_violations"])
        metrics["deadline_violations"].append(last_info["deadline_violations"])
        metrics["hvac_switch_count"].append(last_info["hvac_switch_count"])
        metrics["invalid_action_count"].append(last_info["invalid_action_count"])
        metrics["reward"].append(total_reward)

    # Sonuçları yazdır
    print(f"Mean reward:              {np.mean(metrics['reward']):.3f} ± {np.std(metrics['reward']):.3f}")
    print(f"Mean cost:                {np.mean(metrics['cost']):.2f} TL ± {np.std(metrics['cost']):.2f}")
    print(f"Mean comfort violations:  {np.mean(metrics['comfort_violations']):.2f}")
    print(f"Mean deadline violations: {np.mean(metrics['deadline_violations']):.2f}")
    print(f"Mean HVAC switches:       {np.mean(metrics['hvac_switch_count']):.2f}")
    print(f"Mean invalid actions:     {np.mean(metrics['invalid_action_count']):.2f}")
    return metrics


def evaluate_rule_based_agent(scenarios: list[DailyScenario]):
    """
    Kural tabanlı ajanı verilen senaryolar üzerinde değerlendir.
    RL ajanıyla aynı senaryolar kullanılır — paired evaluation.
    """
    print("\n=== RULE-BASED AGENT EVALUATION ===")
    agent = RuleBasedAgent()

    # Her episode için toplanacak metrikler
    metrics = {
        "cost": [],
        "comfort_violations": [],
        "deadline_violations": [],
        "reward": [],
        "hvac_switch_count": [],
        "invalid_action_count": [],
    }

    for scenario in scenarios:
        # Her senaryo için yeni ortam oluştur
        env = make_env_with_scenario(scenario)
        _, _ = env.reset()
        total_reward = 0.0

        while True:
            # Kural tabanlı ajan ortamdan gerekli bilgileri alarak karar verir
            action = agent.select_action(
                slot_manager=env.slot_manager,
                current_hour=env.current_hour,
                current_price=env.current_price,
                indoor_temp=env.indoor_temp,
                temp_min=env.temp_min,
                temp_max=env.temp_max,
                user_awake=env._is_user_awake(),
                lighting_need=env._lighting_need(),
            )
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        # Episode sonu metriklerini kaydet
        metrics["cost"].append(info["total_cost"])
        metrics["comfort_violations"].append(info["comfort_violations"])
        metrics["deadline_violations"].append(info["deadline_violations"])
        metrics["hvac_switch_count"].append(info["hvac_switch_count"])
        metrics["invalid_action_count"].append(info["invalid_action_count"])
        metrics["reward"].append(total_reward)

    # Sonuçları yazdır
    print(f"Mean reward:              {np.mean(metrics['reward']):.3f} ± {np.std(metrics['reward']):.3f}")
    print(f"Mean cost:                {np.mean(metrics['cost']):.2f} TL ± {np.std(metrics['cost']):.2f}")
    print(f"Mean comfort violations:  {np.mean(metrics['comfort_violations']):.2f}")
    print(f"Mean deadline violations: {np.mean(metrics['deadline_violations']):.2f}")
    print(f"Mean HVAC switches:       {np.mean(metrics['hvac_switch_count']):.2f}")
    print(f"Mean invalid actions:     {np.mean(metrics['invalid_action_count']):.2f}")
    return metrics


def evaluate_rl_agent_dynamic(scenarios: list[DailyScenario]):
    print("\n=== RL AGENT EVALUATION (DYNAMIC) ===")

    dummy_env = DummyVecEnv([lambda: make_env_dynamic(scenarios[0], seed=0)])
    vec_env = VecNormalize.load(
        "models_balanced_dynamic_v6/vec_normalize.pkl", dummy_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(
        "models_balanced_dynamic_v6/best_model/best_model", env=vec_env)

    metrics = {
        "cost": [], "comfort_violations": [], "deadline_violations": [],
        "reward": [], "hvac_switch_count": [], "invalid_action_count": [],
    }

    for i, scenario in enumerate(scenarios):
        raw_env = make_env_dynamic(scenario, seed=i)
        new_dummy = DummyVecEnv([lambda: raw_env])
        new_vec = VecNormalize.load(
            "models_balanced_dynamic_v6/vec_normalize.pkl", new_dummy)
        new_vec.training = False
        new_vec.norm_reward = False
        model.set_env(new_vec)

        obs = new_vec.reset()
        total_reward = 0.0
        done = False
        info = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = new_vec.step(action)
            total_reward += float(reward[0])

        last_info = info[0]
        metrics["cost"].append(last_info["total_cost"])
        metrics["comfort_violations"].append(last_info["comfort_violations"])
        metrics["deadline_violations"].append(last_info["deadline_violations"])
        metrics["hvac_switch_count"].append(last_info["hvac_switch_count"])
        metrics["invalid_action_count"].append(last_info["invalid_action_count"])
        metrics["reward"].append(total_reward)

    print(f"Mean reward:              {np.mean(metrics['reward']):.3f} ± {np.std(metrics['reward']):.3f}")
    print(f"Mean cost:                {np.mean(metrics['cost']):.2f} TL ± {np.std(metrics['cost']):.2f}")
    print(f"Mean comfort violations:  {np.mean(metrics['comfort_violations']):.2f}")
    print(f"Mean deadline violations: {np.mean(metrics['deadline_violations']):.2f}")
    print(f"Mean HVAC switches:       {np.mean(metrics['hvac_switch_count']):.2f}")
    print(f"Mean invalid actions:     {np.mean(metrics['invalid_action_count']):.2f}")
    return metrics


def evaluate_rule_based_agent_dynamic(scenarios: list[DailyScenario]):
    print("\n=== RULE-BASED AGENT EVALUATION (DYNAMIC) ===")
    agent = RuleBasedAgent()

    metrics = {
        "cost": [], "comfort_violations": [], "deadline_violations": [],
        "reward": [], "hvac_switch_count": [], "invalid_action_count": [],
    }

    for i, scenario in enumerate(scenarios):
        env = make_env_dynamic(scenario, seed=i)
        _, _ = env.reset()
        total_reward = 0.0

        while True:
            action = agent.select_action(
                slot_manager=env.slot_manager,
                current_hour=env.current_hour,
                current_price=env.current_price,
                indoor_temp=env.indoor_temp,
                temp_min=env.temp_min,
                temp_max=env.temp_max,
                user_awake=env._is_user_awake(),
                lighting_need=env._lighting_need(),
            )
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        metrics["cost"].append(info["total_cost"])
        metrics["comfort_violations"].append(info["comfort_violations"])
        metrics["deadline_violations"].append(info["deadline_violations"])
        metrics["hvac_switch_count"].append(info["hvac_switch_count"])
        metrics["invalid_action_count"].append(info["invalid_action_count"])
        metrics["reward"].append(total_reward)

    print(f"Mean reward:              {np.mean(metrics['reward']):.3f} ± {np.std(metrics['reward']):.3f}")
    print(f"Mean cost:                {np.mean(metrics['cost']):.2f} TL ± {np.std(metrics['cost']):.2f}")
    print(f"Mean comfort violations:  {np.mean(metrics['comfort_violations']):.2f}")
    print(f"Mean deadline violations: {np.mean(metrics['deadline_violations']):.2f}")
    print(f"Mean HVAC switches:       {np.mean(metrics['hvac_switch_count']):.2f}")
    print(f"Mean invalid actions:     {np.mean(metrics['invalid_action_count']):.2f}")
    return metrics


def compare(n_episodes: int = 100):
    scenarios = build_shared_scenarios(n_episodes=n_episodes, seed=2026)

    print("\n--- FIXED EVALUATION ---")
    rl_fixed = evaluate_rl_agent(scenarios)
    rb_fixed = evaluate_rule_based_agent(scenarios)

    print("\n--- DYNAMIC EVALUATION ---")
    rl_dynamic = evaluate_rl_agent_dynamic(scenarios)
    rb_dynamic = evaluate_rule_based_agent_dynamic(scenarios)

    print("\n" + "=" * 72)
    print("FIXED EVALUATION RESULTS")
    print("=" * 72)
    print(f"{'Metric':<28} {'RL Agent':>14} {'Rule-Based':>16}")
    print("-" * 72)
    print(f"{'Cost (TL)':<28} {np.mean(rl_fixed['cost']):>14.2f} {np.mean(rb_fixed['cost']):>16.2f}")
    print(f"{'Comfort violations':<28} {np.mean(rl_fixed['comfort_violations']):>14.2f} {np.mean(rb_fixed['comfort_violations']):>16.2f}")
    print(f"{'Deadline violations':<28} {np.mean(rl_fixed['deadline_violations']):>14.2f} {np.mean(rb_fixed['deadline_violations']):>16.2f}")
    print(f"{'HVAC switches':<28} {np.mean(rl_fixed['hvac_switch_count']):>14.2f} {np.mean(rb_fixed['hvac_switch_count']):>16.2f}")
    print(f"{'Invalid actions':<28} {np.mean(rl_fixed['invalid_action_count']):>14.2f} {np.mean(rb_fixed['invalid_action_count']):>16.2f}")
    print("=" * 72)
    paired_cost_gain_fixed = (np.mean(rb_fixed["cost"]) - np.mean(rl_fixed["cost"])) / max(np.mean(rb_fixed["cost"]), 1e-6) * 100.0
    print(f"Fixed cost improvement: {paired_cost_gain_fixed:.2f}%")

    print("\n" + "=" * 72)
    print("DYNAMIC EVALUATION RESULTS")
    print("=" * 72)
    print(f"{'Metric':<28} {'RL Agent':>14} {'Rule-Based':>16}")
    print("-" * 72)
    print(f"{'Cost (TL)':<28} {np.mean(rl_dynamic['cost']):>14.2f} {np.mean(rb_dynamic['cost']):>16.2f}")
    print(f"{'Comfort violations':<28} {np.mean(rl_dynamic['comfort_violations']):>14.2f} {np.mean(rb_dynamic['comfort_violations']):>16.2f}")
    print(f"{'Deadline violations':<28} {np.mean(rl_dynamic['deadline_violations']):>14.2f} {np.mean(rb_dynamic['deadline_violations']):>16.2f}")
    print(f"{'HVAC switches':<28} {np.mean(rl_dynamic['hvac_switch_count']):>14.2f} {np.mean(rb_dynamic['hvac_switch_count']):>16.2f}")
    print(f"{'Invalid actions':<28} {np.mean(rl_dynamic['invalid_action_count']):>14.2f} {np.mean(rb_dynamic['invalid_action_count']):>16.2f}")
    print("=" * 72)
    paired_cost_gain_dynamic = (np.mean(rb_dynamic["cost"]) - np.mean(rl_dynamic["cost"])) / max(np.mean(rb_dynamic["cost"]), 1e-6) * 100.0
    print(f"Dynamic cost improvement: {paired_cost_gain_dynamic:.2f}%")


if __name__ == "__main__":
    compare()