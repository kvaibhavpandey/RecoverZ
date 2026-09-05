from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "recovery_training.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "total_amount",
    "promo_amount",
    "shipment_fee",
    "net_amount",

    "transaction_hour",
    "transaction_day_of_week",
    "transaction_month",
    "is_weekend",

    "previous_transaction_count",
    "previous_success_count",
    "previous_failure_count",
    "previous_success_rate",
    "days_since_previous_transaction",

    "session_event_count",
    "unique_event_types",
    "search_count",
    "booking_count",
    "promo_page_count",
    "add_promo_count",
    "session_duration_seconds",
    "purchase_intent_score",

    "year",
]


CATEGORICAL_FEATURES = [
    "payment_method",
    "gender",
    "device_type",
    "device_version",
    "home_country",
    "traffic_source",
    "product_gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "usage",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading recovery dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["created_at"]
    )

    # ONLY failed transactions.
    df = df[
        df["payment_success"] == 0
    ].copy()

    # Target must exist.
    df = df[
        df["recovery_proxy"].notna()
    ].copy()

    df["recovery_proxy"] = (
        df["recovery_proxy"]
        .astype(int)
    )

    df = df.sort_values(
        "created_at"
    ).reset_index(
        drop=True
    )

    print(
        f"Failed transactions used: "
        f"{len(df):,}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        df["recovery_proxy"]
        .value_counts()
    )

    return df


# ============================================================
# TIME SPLIT
# ============================================================

def time_split(df):

    n = len(df)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    print("\nTime split:")

    print(
        f"Train:      {len(train):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"Test:       {len(test):,}"
    )

    print(
        "\nDate ranges:"
    )

    print(
        "Train:",
        train["created_at"].min(),
        "→",
        train["created_at"].max()
    )

    print(
        "Validation:",
        validation["created_at"].min(),
        "→",
        validation["created_at"].max()
    )

    print(
        "Test:",
        test["created_at"].min(),
        "→",
        test["created_at"].max()
    )

    return (
        train,
        validation,
        test
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),

            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
        ],
        remainder="drop"
    )

    return preprocessor


# ============================================================
# METRICS
# ============================================================

def evaluate_model(
    name,
    model,
    X,
    y
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y,
            predictions
        ),

        "precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y,
            probabilities
        ),

        "brier_score": brier_score_loss(
            y,
            probabilities
        ),
    }

    print(
        f"\n{name}"
    )

    for key, value in metrics.items():

        print(
            f"{key:15s}: "
            f"{value:.4f}"
        )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    train, validation, test = (
        time_split(df)
    )

    X_train = train[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    ]

    y_train = train[
        "recovery_proxy"
    ]

    X_validation = validation[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    ]

    y_validation = validation[
        "recovery_proxy"
    ]

    X_test = test[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    ]

    y_test = test[
        "recovery_proxy"
    ]

    preprocessor = create_preprocessor()

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ),

    }

    results = {}

    trained_models = {}

    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for name, estimator in models.items():

        print(
            "\n" + "=" * 70
        )

        print(
            f"TRAINING: {name}"
        )

        print(
            "=" * 70
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    estimator
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        validation_metrics = evaluate_model(
            name + " - Validation",
            pipeline,
            X_validation,
            y_validation
        )

        results[name] = {
            "validation": validation_metrics
        }

        trained_models[name] = pipeline

    # --------------------------------------------------------
    # Select best model by PR-AUC
    # --------------------------------------------------------

    best_name = max(
        results,
        key=lambda name:
        results[name]["validation"]["pr_auc"]
    )

    best_pipeline = trained_models[
        best_name
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        f"BEST MODEL: {best_name}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Evaluate best model on untouched test set
    # --------------------------------------------------------

    test_metrics = evaluate_model(
        best_name + " - TEST",
        best_pipeline,
        X_test,
        y_test
    )

    results[
        best_name
    ]["test"] = test_metrics

    # --------------------------------------------------------
    # Calibrate the selected model
    #
    # Important:
    # calibration is performed using the validation period,
    # not the final test period.
    # --------------------------------------------------------

    print(
        "\nCalibrating best model..."
    )

    calibrated_model = CalibratedClassifierCV(
    FrozenEstimator(best_pipeline),
    method="sigmoid"
    )

    calibrated_model.fit(
        X_validation,
        y_validation
    )

    calibrated_metrics = evaluate_model(
        best_name + " - CALIBRATED TEST",
        calibrated_model,
        X_test,
        y_test
    )

    results[
        best_name
    ]["calibrated_test"] = calibrated_metrics

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "recovery_model.joblib"
    )

    joblib.dump(
        calibrated_model,
        model_path
    )

    print(
        f"\nSaved model:\n{model_path}"
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "model_name": best_name,
        "target": "recovery_proxy",
        "recovery_window": "14 days",
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "test_metrics": test_metrics,
        "calibrated_test_metrics": calibrated_metrics,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }

    metadata_path = (
        MODEL_DIR
        / "recovery_model_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        f"Saved metadata:\n"
        f"{metadata_path}"
    )


if __name__ == "__main__":
    main()