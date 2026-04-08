from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Enerji kategorileri ve yaklaşık güç tüketimleri (kW)
ENERGY_CATEGORIES = {
    "A": 0.3,
    "B": 0.8,
    "C": 1.5,
    "D": 2.5,
    "E": 7.0,
}

# Cihaz tipleri
DEVICE_TYPE_EMPTY = 0
DEVICE_TYPE_SHIFTABLE = 1
DEVICE_TYPE_CONTINUOUS = 2

VALID_DEVICE_TYPES = {DEVICE_TYPE_EMPTY, DEVICE_TYPE_SHIFTABLE, DEVICE_TYPE_CONTINUOUS}


@dataclass(slots=True)
class DeviceUsageProfile:
    """
    Cihazın kullanıcı tarafındaki kullanım tercihleri.

    preferred_start_hour / preferred_end_hour:
        Kullanıcının cihazın daha çok çalışmasını istediği aralık
    user_priority:
        0-1 arası önem seviyesi
    comfort_sensitive:
        HVAC gibi sıcaklık konforunu etkileyen cihaz mı?
    lighting_sensitive:
        Aydınlatma gibi ışık ihtiyacını etkileyen cihaz mı?
    """
    preferred_start_hour: Optional[int] = None
    preferred_end_hour: Optional[int] = None
    user_priority: float = 0.5
    comfort_sensitive: bool = False
    lighting_sensitive: bool = False

    def __post_init__(self) -> None:
        self.user_priority = float(min(max(self.user_priority, 0.0), 1.0))


@dataclass(slots=True)
class Device:
    """
    Genel cihaz sınıfı.
    Kullanıcı ister preset cihaz ekler, ister custom cihaz ekler;
    sonunda hepsi Device nesnesi olur.
    """
    name: str
    category: Optional[str]
    device_type: int
    duration: Optional[int] = None
    deadline: Optional[int] = None
    usage_profile: DeviceUsageProfile = field(default_factory=DeviceUsageProfile)
    power_kw: float = field(init=False)

    # Durum bilgileri
    is_active: bool = False
    is_completed: bool = False
    deadline_missed: bool = False   # yeni: deadline gerçekten kaçırıldı mı?
    remaining_hours: int = 0
    previous_active_state: bool = False
    switch_count: int = 0
    activation_count: int = 0

    def __post_init__(self) -> None:
        if self.device_type not in VALID_DEVICE_TYPES:
            raise ValueError(f"Invalid device type: {self.device_type}")

        # Boş slot ise diğer alanlar anlamsız
        if self.device_type == DEVICE_TYPE_EMPTY:
            self.category = None
            self.duration = None
            self.deadline = None
            self.power_kw = 0.0
            return

        if self.category not in ENERGY_CATEGORIES:
            raise ValueError(f"Invalid category: {self.category}")

        # Ertelenebilir cihaz için süre ve deadline gerekli
        if self.device_type == DEVICE_TYPE_SHIFTABLE:
            if self.duration is None or self.duration <= 0:
                raise ValueError("Shiftable devices require a positive duration")
            if self.deadline is None or not 0 <= self.deadline <= 24:
                raise ValueError("Shiftable devices require a deadline in [0, 24]")

        # Sürekli cihazlarda duration anlamsız
        if self.device_type == DEVICE_TYPE_CONTINUOUS:
            self.duration = None

        self.power_kw = float(ENERGY_CATEGORIES[self.category])

    @property
    def comfort_sensitive(self) -> bool:
        return self.usage_profile.comfort_sensitive

    @property
    def lighting_sensitive(self) -> bool:
        return self.usage_profile.lighting_sensitive

    @classmethod
    def empty_slot(cls) -> "Device":
        return cls(name="EMPTY", category=None, device_type=DEVICE_TYPE_EMPTY)

    def get_power(self) -> float:
        """Şu an ne kadar güç tüketiyor?"""
        if self.device_type == DEVICE_TYPE_EMPTY:
            return 0.0
        return self.power_kw if self.is_active else 0.0

    def start(self) -> bool:
        """
        Ertelenebilir cihazı başlat.
        Başladıysa True, başlayamadıysa False döner.
        """
        if (
            self.device_type == DEVICE_TYPE_SHIFTABLE
            and not self.is_completed
            and not self.is_active
        ):
            self.previous_active_state = self.is_active
            self.is_active = True
            self.remaining_hours = int(self.duration or 0)
            self.activation_count += 1
            return True
        return False

    def turn_on(self) -> bool:
        """Sürekli cihazı aç"""
        if self.device_type == DEVICE_TYPE_CONTINUOUS:
            self.previous_active_state = self.is_active
            if not self.is_active:
                self.switch_count += 1
            self.is_active = True
            return True
        return False

    def turn_off(self) -> bool:
        """Sürekli cihazı kapat"""
        if self.device_type == DEVICE_TYPE_CONTINUOUS:
            self.previous_active_state = self.is_active
            if self.is_active:
                self.switch_count += 1
            self.is_active = False
            return True
        return False

    def step(self) -> None:
        """
        1 saat ilerlet.
        Shiftable cihaz çalışıyorsa remaining_hours azaltılır.
        """
        self.previous_active_state = self.is_active

        if self.device_type == DEVICE_TYPE_SHIFTABLE and self.is_active:
            self.remaining_hours -= 1
            if self.remaining_hours <= 0:
                self.is_active = False
                self.is_completed = True
                self.remaining_hours = 0

    def reset(self) -> None:
        """Yeni episode başında cihazı sıfırla"""
        self.is_active = False
        self.is_completed = False
        self.deadline_missed = False   # yeni: her yeni günde sıfırlanmalı
        self.remaining_hours = 0
        self.previous_active_state = False
        self.switch_count = 0
        self.activation_count = 0


# Hazır cihaz şablonları
DEVICE_PRESETS = {
    "Washing Machine": {
        "category": "C",
        "device_type": DEVICE_TYPE_SHIFTABLE,
        "duration": 2,
        "deadline": 22,
        "usage_profile": DeviceUsageProfile(user_priority=0.7),
    },
    "Dishwasher": {
        "category": "C",
        "device_type": DEVICE_TYPE_SHIFTABLE,
        "duration": 2,
        "deadline": 23,
        "usage_profile": DeviceUsageProfile(user_priority=0.6),
    },
    "HVAC": {
        "category": "D",
        "device_type": DEVICE_TYPE_CONTINUOUS,
        "duration": None,
        "deadline": None,
        "usage_profile": DeviceUsageProfile(user_priority=0.9, comfort_sensitive=True),
    },
    "Lighting": {
        "category": "A",
        "device_type": DEVICE_TYPE_CONTINUOUS,
        "duration": None,
        "deadline": None,
        "usage_profile": DeviceUsageProfile(user_priority=0.8, lighting_sensitive=True),
    },
    "EV Charger": {
        "category": "E",
        "device_type": DEVICE_TYPE_SHIFTABLE,
        "duration": 4,
        "deadline": 22,
        "usage_profile": DeviceUsageProfile(user_priority=0.85),
    },
    "Tumble Dryer": {
        "category": "D",
        "device_type": DEVICE_TYPE_SHIFTABLE,
        "duration": 2,
        "deadline": 22,
        "usage_profile": DeviceUsageProfile(user_priority=0.65),
    },
    "Water Heater": {
        "category": "C",
        "device_type": DEVICE_TYPE_SHIFTABLE,
        "duration": 1,
        "deadline": 22,
        "usage_profile": DeviceUsageProfile(user_priority=0.75),
    },
}


def power_to_category(power_kw: float) -> str:
    """
    Kullanıcının verdiği kW değerini en yakın enerji kategorisine çevir.
    """
    if power_kw <= 0:
        raise ValueError("Power must be positive")
    if power_kw > 10:
        raise ValueError("Power cannot exceed 10 kW for residential devices")

    return min(
        ENERGY_CATEGORIES.items(),
        key=lambda item: abs(power_kw - item[1]),
    )[0]


def create_device_from_preset(preset_name: str) -> Device:
    """
    Hazır şablondan cihaz oluştur.
    """
    if preset_name not in DEVICE_PRESETS:
        raise ValueError(f"Unknown device: {preset_name}")

    preset = DEVICE_PRESETS[preset_name]
    return Device(
        name=preset_name,
        category=preset["category"],
        device_type=preset["device_type"],
        duration=preset["duration"],
        deadline=preset["deadline"],
        usage_profile=preset["usage_profile"],
    )


def create_custom_device(
    name: str,
    device_type: int,
    power_kw: float,
    duration: Optional[int] = None,
    deadline: Optional[int] = None,
    *,
    comfort_sensitive: bool = False,
    lighting_sensitive: bool = False,
    user_priority: float = 0.5,
    preferred_start_hour: Optional[int] = None,
    preferred_end_hour: Optional[int] = None,
) -> Device:
    """
    Kullanıcının özel cihazını oluştur.
    """
    category = power_to_category(power_kw)
    usage_profile = DeviceUsageProfile(
        preferred_start_hour=preferred_start_hour,
        preferred_end_hour=preferred_end_hour,
        user_priority=user_priority,
        comfort_sensitive=comfort_sensitive,
        lighting_sensitive=lighting_sensitive,
    )
    return Device(
        name=name,
        category=category,
        device_type=device_type,
        duration=duration,
        deadline=deadline,
        usage_profile=usage_profile,
    )