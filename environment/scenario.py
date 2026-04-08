from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment.pricing import PriceConfig, build_daily_price_profile


@dataclass(frozen=True, slots=True)
class DailyScenario:
    """
    Bir günlük simülasyon senaryosunu temsil eder.
    Bu dataclass sayesinde aynı gün hem RL ajanına hem kural tabanlı ajana
    verilebilir — paired evaluation için kritik.

    Alanlar:
        daily_temp_base: günün ortalama dış sıcaklığı (°C)
        daily_temp_amplitude: gün içi sıcaklık salınımı (°C)
        initial_indoor_temp: gün başındaki iç sıcaklık (°C)
        awake_start: kullanıcının uyanma saati
        sleep_start: kullanıcının uyuma saati
        occupancy_profile: her saat için evde bulunma oranı (0-1, 24 değer)
        lighting_need_profile: her saat için aydınlatma ihtiyacı (0-1, 24 değer)
        price_profile: her saat için elektrik fiyatı (TL/kWh, 24 değer)
    """
    daily_temp_base: float
    daily_temp_amplitude: float
    initial_indoor_temp: float
    awake_start: int
    sleep_start: int
    occupancy_profile: np.ndarray
    lighting_need_profile: np.ndarray
    price_profile: np.ndarray


def build_daily_scenario(
    rng: np.random.Generator,
    *,
    outdoor_temp_min: float = 10.0,          # minimum günlük baz sıcaklık
    outdoor_temp_max: float = 30.0,          # maksimum günlük baz sıcaklık
    amplitude_min: float = 3.0,              # minimum sıcaklık salınımı
    amplitude_max: float = 8.0,              # maksimum sıcaklık salınımı
    awake_start_range: tuple[int, int] = (6, 9),    # uyanma saati aralığı
    sleep_start_range: tuple[int, int] = (22, 24),  # uyuma saati aralığı
    price_config: PriceConfig | None = None,         # fiyat profili yapılandırması
) -> DailyScenario:
    """
    Rastgele bir günlük senaryo oluştur.

    Her episode için farklı bir gün simüle edilir:
    - Farklı sıcaklık profilleri (yaz/kış)
    - Farklı kullanıcı uyku/uyanıklık saatleri
    - Farklı occupancy ve aydınlatma ihtiyaçları
    - Gürültülü fiyat profili (TEDAŞ tarifesine dayalı)

    Bu çeşitlilik ajanın genelleme yapmasını sağlar —
    sadece tek bir senaryo ezberlemez.
    """

    # Günlük sıcaklık profili parametreleri
    daily_temp_base = float(rng.uniform(outdoor_temp_min, outdoor_temp_max))
    daily_temp_amplitude = float(rng.uniform(amplitude_min, amplitude_max))

    # Kullanıcının uyku/uyanıklık saatlerini rastgele belirle
    awake_start = int(rng.integers(awake_start_range[0], awake_start_range[1] + 1))
    sleep_start = int(rng.integers(sleep_start_range[0], sleep_start_range[1] + 1))
    # Mantıksız değerleri düzelt: uyuma saati uyanma saatinden önce olamaz
    if sleep_start <= awake_start:
        sleep_start = awake_start + 14

    # Her saat için occupancy ve aydınlatma profilleri
    occupancy_profile = np.zeros(24, dtype=np.float32)
    lighting_need_profile = np.zeros(24, dtype=np.float32)

    for hour in range(24):
        # Kullanıcı uyanık mı?
        awake = 1.0 if awake_start <= hour < min(sleep_start, 24) else 0.0

        # Küçük rastgele gürültü — gerçek hayatta her gün biraz farklı
        occupancy_noise = float(rng.uniform(-0.15, 0.15))
        lighting_noise = float(rng.uniform(-0.1, 0.1))

        # Occupancy: uyanıksa 1, uyuyorsa 0 + gürültü
        occupancy_profile[hour] = float(np.clip(awake + occupancy_noise, 0.0, 1.0))

        # Aydınlatma ihtiyacı: karanlıksa ve evde birileri varsa ışık gerekir
        is_dark = 1.0 if hour >= 18 or hour < 7 else 0.0
        lighting_need_profile[hour] = float(
            np.clip(is_dark * occupancy_profile[hour] + lighting_noise, 0.0, 1.0)
        )

    # Gün başındaki (saat 00:00) dış sıcaklığı hesapla
    # Sinüzoidal profil — saat 14 civarında en sıcak
    angle = 2 * np.pi * ((0 - 8) / 24.0)
    outdoor_at_hour_zero = daily_temp_base + daily_temp_amplitude * np.sin(angle)

    # İç sıcaklık dış sıcaklığa yakın başlasın (±2°C)
    initial_indoor_temp = float(rng.uniform(outdoor_at_hour_zero - 2.0, outdoor_at_hour_zero + 2.0))

    # Fiyat profili: gün başında bir kez oluşturulur ve sabit kalır
    # Bu sayede observation ve reward aynı fiyatı kullanır (tutarlılık)
    price_profile = build_daily_price_profile(rng, config=price_config)

    return DailyScenario(
        daily_temp_base=daily_temp_base,
        daily_temp_amplitude=daily_temp_amplitude,
        initial_indoor_temp=initial_indoor_temp,
        awake_start=awake_start,
        sleep_start=sleep_start,
        occupancy_profile=occupancy_profile,
        lighting_need_profile=lighting_need_profile,
        price_profile=price_profile,
    )