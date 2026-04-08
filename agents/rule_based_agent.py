from __future__ import annotations

from environment.devices import DEVICE_TYPE_CONTINUOUS, DEVICE_TYPE_EMPTY, DEVICE_TYPE_SHIFTABLE
from environment.pricing import get_price_category_from_value
from environment.slots import SlotManager


class RuleBasedAgent:
    def select_action(
        self,
        slot_manager: SlotManager,
        *,
        current_hour: int,
        current_price: float,
        indoor_temp: float,
        temp_min: float,
        temp_max: float,
        user_awake: bool,
        lighting_need: float,
    ) -> list[int]:
        actions: list[int] = []
        price_category = get_price_category_from_value(current_price)
        target = (temp_min + temp_max) / 2.0

        for device in slot_manager.slots:
            if device.device_type == DEVICE_TYPE_EMPTY:
                actions.append(0)
                continue

            if device.device_type == DEVICE_TYPE_SHIFTABLE:
                if device.is_completed or device.is_active:
                    actions.append(0)
                    continue

                remaining_to_deadline = 99 if device.deadline is None else device.deadline - current_hour
                must_start_now = remaining_to_deadline <= max(device.duration or 0, 1)
                urgent_window = remaining_to_deadline <= max((device.duration or 0) + 2, 3)

                in_preferred_window = False
                if (
                    device.usage_profile.preferred_start_hour is not None
                    and device.usage_profile.preferred_end_hour is not None
                ):
                    in_preferred_window = (
                        device.usage_profile.preferred_start_hour
                        <= current_hour
                        < device.usage_profile.preferred_end_hour
                    )

                if must_start_now:
                    actions.append(1)
                elif price_category == "cheap":
                    actions.append(1)
                elif price_category == "normal" and urgent_window:
                    actions.append(1)
                elif in_preferred_window and price_category != "expensive":
                    actions.append(1)
                else:
                    actions.append(0)
                continue

            if device.device_type == DEVICE_TYPE_CONTINUOUS:
                if device.comfort_sensitive:
                    if indoor_temp < temp_min:
                        actions.append(1)
                    elif indoor_temp > temp_max:
                        actions.append(1)
                    elif price_category == "cheap" and abs(indoor_temp - target) > 0.6:
                        actions.append(1)
                    else:
                        actions.append(0)
                elif device.lighting_sensitive:
                    actions.append(1 if user_awake and lighting_need >= 0.5 else 0)
                else:
                    actions.append(0)
                continue

            actions.append(0)

        return actions
