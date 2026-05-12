from __future__ import annotations

from environment.devices import DEVICE_TYPE_SHIFTABLE
from environment.slots import SlotManager
from environment.smart_home_env import RewardWeights


MODE_WEIGHTS = {
    "cost":     RewardWeights(cost=0.65, comfort=0.15, task=0.20),
    "balanced": RewardWeights(cost=0.42, comfort=0.38, task=0.20),
    "comfort":  RewardWeights(cost=0.20, comfort=0.60, task=0.20),
}


def apply_daily_setup(
    slot_manager: SlotManager,
    *,
    user_home: bool,
    active_devices: dict[str, bool],
    deadlines: dict[str, int],
) -> None:
    for device in slot_manager.slots:
        if device.device_type != DEVICE_TYPE_SHIFTABLE:
            continue

        is_active_today = active_devices.get(device.name, False)
        device.active_today = is_active_today

        if is_active_today and device.name in deadlines:
            new_deadline = deadlines[device.name]
            if 0 <= new_deadline <= 24:
                device.deadline = new_deadline


def get_reward_weights(mode: str) -> RewardWeights:
    return MODE_WEIGHTS.get(mode, MODE_WEIGHTS["balanced"])
