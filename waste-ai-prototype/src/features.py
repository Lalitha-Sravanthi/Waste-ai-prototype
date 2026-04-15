from __future__ import annotations

import numpy as np
import pandas as pd


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    department_map = {
        "organic": "Cafeteria",
        "hazardous": "Quality Lab",
        "recycling": "Production",
        "general": "Administration",
        "other": "Storage",
    }

    source_map = {
        "organic": "Food Waste Area",
        "hazardous": "Chemical Handling Area",
        "recycling": "Packaging and Production",
        "general": "Office Block",
        "other": "Warehouse Utility Area",
    }

    enriched["department"] = enriched["bin_type"].map(department_map).fillna("General Operations")
    enriched["waste_source"] = enriched["bin_type"].map(source_map).fillna("Common Collection Point")

    enriched["shift"] = np.select(
        [
            enriched["hour"].between(6, 13, inclusive="both"),
            enriched["hour"].between(14, 21, inclusive="both"),
        ],
        ["Morning", "Evening"],
        default="Night",
    )

    shift_volume_map = {"Morning": 1.15, "Evening": 1.0, "Night": 0.8}
    enriched["production_volume_index"] = (
        enriched["waste_generation_index"] * enriched["shift"].map(shift_volume_map)
    ).round(2)

    enriched["estimated_pickup_cost"] = (
        (enriched["travel_time"] * 1.8)
        + (enriched["weight_kg"] * 1.4)
        + np.where(enriched["bin_type"].eq("hazardous"), 35, 0)
    ).round(2)

    enriched["pickup_required"] = (
        (enriched["fill_level_percent"] >= 80) | (enriched["overflow_alert"])
    ).astype(int)

    enriched["manager_approval_required"] = (
        enriched["bin_type"].eq("hazardous")
        | (enriched["estimated_pickup_cost"] > 120)
    ).astype(int)

    enriched["manager_approval_status"] = np.where(
        enriched["manager_approval_required"].eq(1),
        "Pending",
        "Not Needed",
    )

    enriched["overflow_risk"] = np.select(
        [
            enriched["fill_level_percent"] >= 90,
            enriched["fill_level_percent"] >= 70,
        ],
        ["High", "Medium"],
        default="Low",
    )

    enriched["vehicle_assignment_status"] = np.select(
        [
            (enriched["pickup_required"] == 1) & (enriched["truck_available"] == 1),
            (enriched["pickup_required"] == 1) & (enriched["truck_available"] == 0),
        ],
        ["Vehicle Ready", "Awaiting Vehicle"],
        default="No Dispatch Needed",
    )

    enriched["cost_saving_if_skipped"] = np.where(
        enriched["pickup_required"].eq(0),
        enriched["estimated_pickup_cost"],
        0,
    ).round(2)

    daily_waste = enriched.groupby("date")["weight_kg"].transform("sum")
    waste_threshold = daily_waste.quantile(0.75)
    enriched["derived_peak_day"] = (daily_waste >= waste_threshold).astype(int)

    return enriched


def get_model_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "fill_level_percent",
            "travel_time",
            "route_efficiency",
            "battery_level_percent",
            "temperature",
            "rainfall",
            "gas_level",
            "production_volume_index",
            "truck_available",
            "lag_1",
            "lag_2",
            "rolling_mean_7",
            "urban_density_factor",
            "waste_generation_index",
        ]
    ]
