from __future__ import annotations

import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.devices import create_device_from_preset
from environment.slots import SlotManager
from environment.smart_home_env import RewardWeights, SmartHomeEnv

# Model ve log klasörleri
MODELS_DIR = "models"
LOGS_DIR = "logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def make_env(seed: int | None = None):
    def _factory():
        slot_manager = SlotManager()
        slot_manager.add_device(create_device_from_preset("HVAC"))
        slot_manager.add_device(create_device_from_preset("Washing Machine"))
        slot_manager.add_device(create_device_from_preset("Lighting"))

        return SmartHomeEnv(
            slot_manager=slot_manager,
            temp_min=20.0,
            temp_max=24.0,
            reward_weights=RewardWeights(
                cost=0.42,
                comfort=0.38,
                task=0.20,
            ),
            seed=seed,
        )

    return _factory


def train() -> None:
    # Eğitim ortamı
    train_env = DummyVecEnv([make_env(seed=42)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # Değerlendirme ortamı
    eval_env = DummyVecEnv([make_env(seed=43)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # Eval env normalizasyon istatistiklerini güncelleme — sadece train_env günceller
    eval_env.training = False
    eval_env.norm_reward = False

    # EvalCallback, VecNormalize tespit edince obs_rms'yi her eval öncesi
    # otomatik senkronize eder — ayrıca parametre gerekmez.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODELS_DIR}/best_model",
        log_path=LOGS_DIR,
        eval_freq=5_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=f"{MODELS_DIR}/checkpoints",
        name_prefix="ppo_smart_home",
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=8e-5,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOGS_DIR,
    )

    print("Training started...")
    model.learn(
        total_timesteps=700_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(f"{MODELS_DIR}/final_model")
    train_env.save(f"{MODELS_DIR}/vec_normalize.pkl")
    print("Training completed. Model and normalization statistics saved.")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()