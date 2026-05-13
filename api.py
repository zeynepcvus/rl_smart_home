# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import replace

from environment.slots import SlotManager
from environment.devices import create_device_from_preset, create_custom_device, DEVICE_TYPE_SHIFTABLE
from environment.scenario import build_daily_scenario
from environment.smart_home_env import SmartHomeEnv, RewardWeights
from agents.rule_based_agent import RuleBasedAgent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATHS = {
    "cost":     "models_cost_dynamic_v1",
    "balanced": "models_balanced_dynamic_v6",
    "comfort":  "models_comfort_dynamic_v1",
}

REWARD_WEIGHTS = {
    "cost":     RewardWeights(cost=0.65, comfort=0.15, task=0.20),
    "balanced": RewardWeights(cost=0.42, comfort=0.38, task=0.20),
    "comfort":  RewardWeights(cost=0.20, comfort=0.60, task=0.20),
}


class DeviceInput(BaseModel):
    name: str
    preset: bool = True
    power_kw: float | None = None
    duration: int | None = None
    deadline: int | None = None


class RunRequest(BaseModel):
    mode: str = "balanced"
    devices: list[DeviceInput]
    user_home: bool = True
    temp_min: float = 20.0
    temp_max: float = 24.0
    awake_start: int = 7
    sleep_start: int = 23


def build_user_profiles(awake_start: int, sleep_start: int, user_home: bool):
    occupancy_profile = np.zeros(24, dtype=np.float32)
    lighting_need_profile = np.zeros(24, dtype=np.float32)
    if user_home:
        for hour in range(24):
            occupancy = 1.0 if awake_start <= hour < sleep_start else 0.0
            occupancy_profile[hour] = occupancy
            is_dark = hour >= 18 or hour < 7
            lighting_need_profile[hour] = 1.0 if is_dark and occupancy > 0 else 0.0
    return occupancy_profile, lighting_need_profile


def build_slot_manager(req: RunRequest) -> SlotManager:
    slot_manager = SlotManager()
    slot_manager.add_device(create_device_from_preset("HVAC"))
    slot_manager.add_device(create_device_from_preset("Lighting"))
    for d in req.devices:
        if d.name in ("HVAC", "Lighting"):
            continue
        if d.preset:
            device = create_device_from_preset(d.name)
            if (d.deadline is not None
                    and device.device_type == DEVICE_TYPE_SHIFTABLE
                    and 0 <= d.deadline <= 24):
                device.deadline = d.deadline
            slot_manager.add_device(device)
        else:
            slot_manager.add_device(create_custom_device(
                name=d.name,
                device_type=DEVICE_TYPE_SHIFTABLE,
                power_kw=d.power_kw,
                duration=d.duration,
                deadline=d.deadline,
            ))
    return slot_manager


def run_rl(slot_manager: SlotManager, scenario, mode: str, temp_min: float = 20.0, temp_max: float = 24.0) -> dict:
    env = SmartHomeEnv(
        slot_manager=slot_manager,
        temp_min=temp_min,
        temp_max=temp_max,
        scenario=scenario,
        reward_weights=REWARD_WEIGHTS[mode],
    )

    model_dir = MODEL_PATHS[mode]
    dummy = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(f"{model_dir}/vec_normalize.pkl", dummy)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(f"{model_dir}/best_model/best_model.zip", env=vec_env)

    obs = vec_env.reset()
    hours = []
    done = False

    while not done:
        hour = env.current_hour
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, info = vec_env.step(action)
        done = bool(dones[0])
        hours.append({
            "hour": hour,
            "active_devices": [d.name for d in env.slot_manager.slots if d.is_active],
            "indoor_temp": round(info[0]["indoor_temp"], 1),
            "outdoor_temp": round(info[0]["outdoor_temp"], 1),
            "price": round(info[0]["current_price"], 3),
            "price_category": info[0]["price_category"],
            "step_cost": round(info[0]["last_step_cost"], 4),
        })

    last = info[0]
    return {
        "summary": {
            "total_cost": round(last["total_cost"], 2),
            "comfort_violations": last["comfort_violations"],
            "deadline_violations": last["deadline_violations"],
            "hvac_switches": last["hvac_switch_count"],
            "invalid_actions": last["invalid_action_count"],
        },
        "hours": hours,
    }


def run_rule_based(slot_manager: SlotManager, scenario, mode: str, temp_min: float = 20.0, temp_max: float = 24.0) -> dict:
    env = SmartHomeEnv(
        slot_manager=slot_manager,
        temp_min=temp_min,
        temp_max=temp_max,
        scenario=scenario,
        reward_weights=REWARD_WEIGHTS[mode],
    )
    env.reset()
    agent = RuleBasedAgent()
    hours = []

    while True:
        hour = env.current_hour
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
        _, _, terminated, truncated, info = env.step(action)

        hours.append({
            "hour": hour,
            "active_devices": [d.name for d in env.slot_manager.slots if d.is_active],
            "indoor_temp": round(info["indoor_temp"], 1),
            "outdoor_temp": round(info["outdoor_temp"], 1),
            "price": round(info["current_price"], 3),
            "price_category": info["price_category"],
            "step_cost": round(info["last_step_cost"], 4),
        })

        if terminated or truncated:
            break

    return {
        "summary": {
            "total_cost": round(info["total_cost"], 2),
            "comfort_violations": info["comfort_violations"],
            "deadline_violations": info["deadline_violations"],
            "hvac_switches": info["hvac_switch_count"],
            "invalid_actions": info["invalid_action_count"],
        },
        "hours": hours,
    }


@app.post("/run")
def run_simulation(req: RunRequest):
    rng = np.random.default_rng(42)
    base_scenario = build_daily_scenario(rng)
    occupancy_profile, lighting_need_profile = build_user_profiles(
        req.awake_start, req.sleep_start, req.user_home
    )
    scenario = replace(
        base_scenario,
        awake_start=req.awake_start,
        sleep_start=req.sleep_start,
        user_home=req.user_home,
        occupancy_profile=occupancy_profile,
        lighting_need_profile=lighting_need_profile,
    )

    sm_rl = build_slot_manager(req)
    rl_result = run_rl(sm_rl, scenario, req.mode, req.temp_min, req.temp_max)

    sm_rb = build_slot_manager(req)
    rb_result = run_rule_based(sm_rb, scenario, req.mode, req.temp_min, req.temp_max)

    return {
        "rl": rl_result,
        "rule_based": rb_result,
    }
