from __future__ import annotations

from src.config import APPROVAL_COST_THRESHOLD, PICKUP_THRESHOLD, URGENT_THRESHOLD


def get_pickup_decision(row) -> str:
    if row["fill_level_percent"] >= URGENT_THRESHOLD or bool(row["overflow_alert"]):
        return "Dispatch Immediately"
    if row["fill_level_percent"] >= PICKUP_THRESHOLD:
        if int(row["truck_available"]) == 1:
            return "Schedule Pickup"
        return "Pickup Required but No Vehicle"
    return "No Pickup Needed"


def get_approval_message(row) -> str:
    if row["bin_type"] == "hazardous":
        return "Manager approval required for hazardous waste."
    if row["estimated_pickup_cost"] > APPROVAL_COST_THRESHOLD:
        return "Manager approval required due to high transport cost."
    return "No approval needed."
