from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "generated" / "campus_microgrid_historical.csv"


def _hourly_tariff(hour: int) -> float:
    if 0 <= hour < 6:
        return 4.2
    if 6 <= hour < 10:
        return 6.8
    if 10 <= hour < 17:
        return 5.4
    if 17 <= hour < 21:
        return 9.5
    return 6.8


def generate_microgrid_data(days: int = 365, output_path: str | Path | None = None) -> pd.DataFrame:
    """Generate a realistic hourly microgrid dataset for campus energy flows."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2025-01-01 00:00:00")
    ts = pd.date_range(start=start, periods=days * 24, freq="h")
    df = pd.DataFrame({"timestamp": ts})
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.day_of_year
    df["day_of_week"] = df["timestamp"].dt.day_of_week

    daylight = np.maximum(0.0, np.sin((df["hour"] - 6) / 12 * np.pi))
    seasonal = 0.7 + 0.4 * np.sin((df["day_of_year"] - 80) / 365 * 2 * np.pi)
    cloud_factor = 0.7 + 0.35 * rng.normal(size=len(df))
    solar = 500 * np.clip(daylight, 0, 1) * np.clip(seasonal, 0.2, 1.4) * np.clip(cloud_factor, 0.45, 1.35)
    solar = np.clip(solar + rng.normal(0, 22, size=len(df)), 0, 520)
    df["solar_kw"] = solar.round(2)

    wind_base = 0.38 + 0.62 * ((np.sin((df["hour"] + 7) / 24 * 2 * np.pi) + 1) / 2)
    seasonal_wind = 0.8 + 0.35 * np.sin((df["day_of_year"] + 40) / 365 * 2 * np.pi)
    wind = 300 * np.clip(wind_base * seasonal_wind, 0.1, 1.5)
    wind = np.clip(wind + rng.normal(0, 18, size=len(df)), 0, 380)
    df["wind_kw"] = wind.round(2)

    demand_base = 210 + 55 * np.sin((df["hour"] - 10) / 24 * 2 * np.pi)
    weekday_factor = np.where(df["day_of_week"].isin([5, 6]), 35, 0)
    daytime_academic = np.where((df["hour"] >= 9) & (df["hour"] <= 17), 120, 0)
    evening_peak = np.where((df["hour"] >= 17) & (df["hour"] <= 22), 175, 0)
    hostel_peak = np.where((df["hour"] >= 21) | (df["hour"] <= 5), 80, 0)
    seasonal_demand = 1 + 0.18 * np.sin((df["day_of_year"] - 150) / 365 * 2 * np.pi)
    demand = (demand_base + weekday_factor + daytime_academic + evening_peak + hostel_peak) * seasonal_demand
    demand = np.clip(demand + rng.normal(0, 24, size=len(df)), 180, 900)
    df["demand_kw"] = demand.round(2)

    df["temperature_c"] = (18 + 12 * np.sin((df["day_of_year"] / 365) * 2 * np.pi - 1.5) + 5 * np.cos(df["hour"] / 24 * 2 * np.pi)).round(2)
    df["tariff_inr_per_kwh"] = df["hour"].map(_hourly_tariff)

    output = output_path if output_path is not None else DEFAULT_OUTPUT
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


if __name__ == "__main__":
    generate_microgrid_data()
    print(f"Generated data to {DEFAULT_OUTPUT}")
