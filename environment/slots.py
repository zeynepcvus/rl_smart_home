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
