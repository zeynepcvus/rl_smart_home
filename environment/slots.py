from __future__ import annotations

from environment.devices import Device, DEVICE_TYPE_EMPTY

MAX_SLOTS = 5


class SlotManager:
    def __init__(self, max_slots: int = MAX_SLOTS) -> None:
        self.max_slots = max_slots
        self.slots: list[Device] = [Device.empty_slot() for _ in range(self.max_slots)]

    def add_device(self, device: Device) -> bool:
        for i in range(self.max_slots):
            if self.slots[i].device_type == DEVICE_TYPE_EMPTY:
                self.slots[i] = device
                return True
        return False

    def remove_device(self, index: int) -> bool:
        if 0 <= index < self.max_slots:
            self.slots[index] = Device.empty_slot()
            return True
        return False

    def get_non_empty_devices(self) -> list[Device]:
        return [d for d in self.slots if d.device_type != DEVICE_TYPE_EMPTY]

    def get_active_running_devices(self) -> list[Device]:
        return [d for d in self.slots if d.is_active]

    def reset_all(self) -> None:
        for device in self.slots:
            device.reset()

    def step_all(self) -> None:
        for device in self.slots:
            device.step()

    def get_slot_count(self) -> int:
        return sum(1 for d in self.slots if d.device_type != DEVICE_TYPE_EMPTY)

    def save_profile(self, filepath: str) -> None:
        import json
        from environment.devices import DEVICE_TYPE_EMPTY

        devices_data = []
        for device in self.slots:
            if device.device_type == DEVICE_TYPE_EMPTY:
                devices_data.append({"device_type": 0})
                continue
            devices_data.append({
                "name": device.name,
                "category": device.category,
                "device_type": device.device_type,
                "duration": device.duration,
                "deadline": device.deadline,
                "user_priority": device.usage_profile.user_priority,
                "comfort_sensitive": device.usage_profile.comfort_sensitive,
                "lighting_sensitive": device.usage_profile.lighting_sensitive,
                "preferred_start_hour": device.usage_profile.preferred_start_hour,
                "preferred_end_hour": device.usage_profile.preferred_end_hour,
            })

        profile = {
            "max_slots": self.max_slots,
            "devices": devices_data,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_profile(cls, filepath: str) -> "SlotManager":
        import json
        from environment.devices import (
            Device, DeviceUsageProfile,
            DEVICE_TYPE_EMPTY,
        )

        with open(filepath, "r", encoding="utf-8") as f:
            profile = json.load(f)

        manager = cls(max_slots=profile["max_slots"])

        for i, d in enumerate(profile["devices"]):
            if d["device_type"] == DEVICE_TYPE_EMPTY:
                continue
            usage_profile = DeviceUsageProfile(
                user_priority=d.get("user_priority", 0.5),
                comfort_sensitive=d.get("comfort_sensitive", False),
                lighting_sensitive=d.get("lighting_sensitive", False),
                preferred_start_hour=d.get("preferred_start_hour"),
                preferred_end_hour=d.get("preferred_end_hour"),
            )
            device = Device(
                name=d["name"],
                category=d["category"],
                device_type=d["device_type"],
                duration=d.get("duration"),
                deadline=d.get("deadline"),
                usage_profile=usage_profile,
            )
            manager.slots[i] = device

        return manager
