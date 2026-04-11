from __future__ import annotations

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MODEL_PATH, PROCESSED_DATASET_PATH, RAW_DATASET_PATH
from src.features import add_business_features, get_model_features
from src.preprocess import load_and_clean_data
from src.rules import get_approval_message, get_pickup_decision
from src.utils import ensure_directories


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    if PROCESSED_DATASET_PATH.exists():
        df = pd.read_csv(PROCESSED_DATASET_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    df = load_and_clean_data(RAW_DATASET_PATH)
    return add_business_features(df)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    model = load_model()
    enriched["predicted_peak_day"] = model.predict(get_model_features(enriched)).astype(int)
    enriched["pickup_decision"] = enriched.apply(get_pickup_decision, axis=1)
    enriched["approval_message"] = enriched.apply(get_approval_message, axis=1)
    return enriched


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Operational Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Bins", int(df["bin_id"].nunique()))
    col2.metric("Pickups Needed", int(df["pickup_required"].sum()))
    col3.metric("Pending Approvals", int(df["manager_approval_required"].sum()))
    col4.metric("Potential Savings", f"₹{df['cost_saving_if_skipped'].sum():,.0f}")


def render_charts(df: pd.DataFrame) -> None:
    left, right = st.columns(2)

    with left:
        st.subheader("Fill Level by Waste Type")
        fig = px.box(
            df,
            x="bin_type",
            y="fill_level_percent",
            color="bin_type",
            points="outliers",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Daily Waste Trend")
        daily = df.groupby("date", as_index=False)["weight_kg"].sum()
        fig = px.line(
            daily,
            x="date",
            y="weight_kg",
            markers=True,
            title="Daily waste generated (kg)",
        )
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Pickup Decisions")
        decision_counts = df["pickup_decision"].value_counts().reset_index()
        decision_counts.columns = ["pickup_decision", "count"]
        fig = px.pie(decision_counts, names="pickup_decision", values="count", hole=0.45)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Overflow Risk Heatmap")
        risk_table = (
            df.groupby(["department", "overflow_risk"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        fig = px.bar(
            risk_table,
            x="department",
            y="count",
            color="overflow_risk",
            barmode="group",
        )
        st.plotly_chart(fig, width="stretch")


def render_tables(df: pd.DataFrame) -> None:
    st.subheader("Manager Approval Alerts")
    approvals = df[df["manager_approval_required"] == 1][
        [
            "bin_id",
            "department",
            "bin_type",
            "estimated_pickup_cost",
            "manager_approval_status",
            "approval_message",
        ]
    ].sort_values("estimated_pickup_cost", ascending=False)
    st.dataframe(approvals.head(20), width="stretch")

    st.subheader("Live Pickup Recommendation Table")
    st.dataframe(
        df[
            [
                "timestamp",
                "bin_id",
                "department",
                "fill_level_percent",
                "overflow_risk",
                "vehicle_assignment_status",
                "pickup_decision",
                "predicted_peak_day",
            ]
        ].sort_values("timestamp", ascending=False).head(50),
        width="stretch",
    )


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    departments = ["All"] + sorted(df["department"].dropna().unique().tolist())
    waste_types = ["All"] + sorted(df["bin_type"].dropna().unique().tolist())

    selected_department = st.sidebar.selectbox("Department", departments)
    selected_waste_type = st.sidebar.selectbox("Waste Type", waste_types)
    fill_range = st.sidebar.slider("Fill Level Range", 0, 100, (0, 100))

    filtered = df.copy()
    if selected_department != "All":
        filtered = filtered[filtered["department"] == selected_department]
    if selected_waste_type != "All":
        filtered = filtered[filtered["bin_type"] == selected_waste_type]

    filtered = filtered[
        filtered["fill_level_percent"].between(fill_range[0], fill_range[1], inclusive="both")
    ]
    return filtered


def main() -> None:
    ensure_directories()
    st.set_page_config(page_title="AI Waste Analytics Prototype", layout="wide")
    st.title("AI-Assisted Predictive Waste Analytics Prototype")
    st.caption(
        "Prototype dashboard for smart pickup decisions, approval alerts, and peak waste prediction."
    )

    if not RAW_DATASET_PATH.exists():
        st.error(
            "Raw dataset not found. Run `python scripts/prepare_data.py` first to copy your source data."
        )
        st.stop()

    if not MODEL_PATH.exists():
        st.warning(
            "Model file not found. Run `python -m src.train_model` first to train the prototype model."
        )
        st.stop()

    df = load_dashboard_data()
    df = add_predictions(df)
    filtered_df = render_sidebar(df)

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    render_overview(filtered_df)
    render_charts(filtered_df)
    render_tables(filtered_df)


if __name__ == "__main__":
    main()
