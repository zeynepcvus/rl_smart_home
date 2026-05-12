# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.slots import SlotManager
from environment.devices import create_device_from_preset, create_custom_device, DEVICE_TYPE_SHIFTABLE
from environment.scenario import build_daily_scenario
from environment.smart_home_env import SmartHomeEnv, RewardWeights
from dataclasses import replace

from fastapi.middleware.cors import CORSMiddleware  # ← ekle

app = FastAPI()

# ← bunu ekle
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
    preset: bool = True          # True ise preset, False ise custom
    power_kw: float | None = None
    duration: int | None = None
    deadline: int | None = None


class RunRequest(BaseModel):
    mode: str = "balanced"       # cost / balanced / comfort
    devices: list[DeviceInput]
    user_home: bool = True


@app.post("/run")
def run_simulation(req: RunRequest):
    slot_manager = SlotManager()

    # Cihazları ekle
    slot_manager.add_device(create_device_from_preset("HVAC"))
    slot_manager.add_device(create_device_from_preset("Lighting"))

    for d in req.devices:
        if d.preset:
            slot_manager.add_device(create_device_from_preset(d.name))
        else:
            slot_manager.add_device(create_custom_device(
                name=d.name,
                device_type=DEVICE_TYPE_SHIFTABLE,
                power_kw=d.power_kw,
                duration=d.duration,
                deadline=d.deadline,
            ))

    rng = np.random.default_rng(42)
    scenario = replace(build_daily_scenario(rng), user_home=req.user_home)

    env = SmartHomeEnv(
        slot_manager=slot_manager,
        temp_min=20.0,
        temp_max=24.0,
        scenario=scenario,
        reward_weights=REWARD_WEIGHTS[req.mode],
    )

    model_dir = MODEL_PATHS[req.mode]
    dummy = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(f"{model_dir}/vec_normalize.pkl", dummy)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(f"{model_dir}/best_model/best_model", env=vec_env)

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
            "price": round(info[0]["current_price"], 3),
            "step_cost": round(info[0]["last_step_cost"], 4),
        })

    last = info[0]
    return {
        "summary": {
            "total_cost": round(last["total_cost"], 2),
            "comfort_violations": last["comfort_violations"],
            "deadline_violations": last["deadline_violations"],
        },
        "hours": hours,
    }