import numpy as np
import pandas as pd


def generate_tou_tariff():
    """Returns dynamic hourly pricing ($/kWh) for a 24-hour cycle.

    Off-peak: 00:00 - 06:00 ($0.10)
    Mid-peak: 06:00 - 16:00, 21:00 - 24:00 ($0.18)
    Peak: 16:00 - 21:00 ($0.38)
    """
    tariff = np.zeros(24)
    for h in range(24):
        if 0 <= h < 6:
            tariff[h] = 0.10
        elif (6 <= h < 16) or (21 <= h < 24):
            tariff[h] = 0.18
        else:  # 16:00 - 21:00 Peak
            tariff[h] = 0.38
    return tariff


def generate_synthetic_day_data(days=30, seed=42):
    """Generates synthetic hourly historical data for model training."""
    np.random.seed(seed)
    hours = days * 24

    date_range = pd.date_range(
        start="2026-01-01", periods=hours, freq="h"
    )

    # Ambient Temperature (°C) with diurnal cycle
    time_of_day = date_range.hour.values
    temp = (
        24
        + 6 * np.sin((time_of_day - 9) * np.pi / 12)
        + np.random.normal(0, 1.5, hours)
    )

    # Occupancy Pattern (Probabilistic: higher during morning/evening)
    occupancy_prob = np.where(
        (time_of_day >= 7) & (time_of_day <= 9),
        0.8,
        np.where(
            (time_of_day >= 17) & (time_of_day <= 23),
            0.9,
            np.where((time_of_day >= 0) & (time_of_day <= 6), 0.95, 0.2),
        ),
    )
    occupancy = (np.random.rand(hours) < occupancy_prob).astype(int)

    # Baseline Non-Shiftable Load (kW)
    baseline_load = (
        0.5
        + 0.3 * occupancy
        + 0.05 * (temp - 20).clip(min=0)
        + np.random.normal(0, 0.05, hours)
    )
    baseline_load = np.clip(baseline_load, 0.2, 5.0)

    # Hourly TOU Tariff
    tou_rates = np.tile(generate_tou_tariff(), days)

    df = pd.DataFrame(
        {
            "timestamp": date_range,
            "hour": time_of_day,
            "day_of_week": date_range.dayofweek,
            "temperature_c": np.round(temp, 2),
            "occupancy": occupancy,
            "tou_rate_usd": tou_rates,
            "baseline_load_kw": np.round(baseline_load, 3),
        }
    )

    return df


if __name__ == "__main__":
    data = generate_synthetic_day_data(days=60)
    print("Dataset Summary:")
    print(data.head(12))
    print("\nData Schema Info:")
    print(data.info())