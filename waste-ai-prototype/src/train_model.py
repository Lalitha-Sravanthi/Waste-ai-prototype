from __future__ import annotations

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.config import MODEL_PATH, PROCESSED_DATASET_PATH, RAW_DATASET_PATH
from src.features import add_business_features, get_model_features
from src.preprocess import load_and_clean_data
from src.utils import ensure_directories


def train_and_save_model() -> None:
    ensure_directories()

    df = load_and_clean_data(RAW_DATASET_PATH)
    df = add_business_features(df)
    df.to_csv(PROCESSED_DATASET_PATH, index=False)

    target = "derived_peak_day"
    X = get_model_features(df)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
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
