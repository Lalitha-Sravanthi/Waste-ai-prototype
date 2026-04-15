from __future__ import annotations

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.config import MODEL_PATH, PROCESSED_DATASET_PATH, RAW_DATASET_PATH
from src.features import add_business_features, get_model_features
from src.preprocess import load_and_clean_data
from src.utils import ensure_directories


def train_and_save_model() -> None:
    ensure_directories()

    df = load_and_clean_data(RAW_DATASET_PATH)
    df = add_business_features(df)
    df.to_csv(PROCESSED_DATASET_PATH, index=False)

    # Use the source dataset's peak label for a more stable target.
    # The custom derived_peak_day label turns one entire day into class 1,
    # which creates an unrealistic and unstable forecasting task.
    target = "peak_flag"
    X = get_model_features(df)
    y = df[target]
    unique_dates = sorted(df["date"].dropna().unique())
    if len(unique_dates) < 2:
        raise ValueError("Need at least 2 unique dates for time-based evaluation.")

    split_index = max(1, int(len(unique_dates) * 0.8))
    if split_index >= len(unique_dates):
        split_index = len(unique_dates) - 1

    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]

    train_mask = df["date"].isin(train_dates)
    test_mask = df["date"].isin(test_dates)

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    print(f"Training on dates: {train_dates[0]} to {train_dates[-1]}")
    print(f"Testing on dates: {test_dates[0]} to {test_dates[-1]}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved processed dataset to: {PROCESSED_DATASET_PATH}")


if __name__ == "__main__":
    train_and_save_model()
