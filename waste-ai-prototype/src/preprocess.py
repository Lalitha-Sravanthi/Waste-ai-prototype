from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce", utc=True)

    bool_map = {"true": True, "false": False, "1": True, "0": False}
    for col in ("overflow_alert",):
        cleaned[col] = (
            cleaned[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(bool_map)
            .fillna(False)
        )

    numeric_cols = [
        "fill_level_percent",
        "battery_level_percent",
        "route_efficiency",
        "travel_time",
        "urban_density_factor",
        "waste_generation_index",
        "bin_capacity_kg",
        "bin_fill_percent",
        "weight_kg",
        "gas_level",
        "temperature",
        "rainfall",
        "lag_1",
        "lag_2",
        "rolling_mean_7",
        "next_day_waste_kg",
        "peak_flag",
    ]

    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned["date"] = cleaned["timestamp"].dt.date
    cleaned["hour"] = cleaned["timestamp"].dt.hour
    cleaned["day_name"] = cleaned["timestamp"].dt.day_name()
    cleaned["week_number"] = cleaned["timestamp"].dt.isocalendar().week.astype("Int64")

    cleaned["peak_flag"] = cleaned["peak_flag"].fillna(0).astype(int)
    cleaned["truck_available"] = (
        pd.to_numeric(cleaned["truck_available"], errors="coerce").fillna(0).astype(int)
    )

    cleaned = cleaned.dropna(subset=["timestamp", "fill_level_percent", "weight_kg"])
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def load_and_clean_data(path: str | Path) -> pd.DataFrame:
    return clean_dataset(load_raw_dataset(path))
