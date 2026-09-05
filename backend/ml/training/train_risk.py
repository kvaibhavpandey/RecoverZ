from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "payment_risk"
    / "transactions_train.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = MODEL_DIR / "risk_model.joblib"
METADATA_FILE = MODEL_DIR / "risk_model_metadata.json"


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "account_age_days",
    "avg_monthly_spend",
    "merchant_risk_score",
    "transaction_amount",
    "ip_risk_score",
    "txn_count_1h",
    "txn_count_24h",
    "failed_txn_count_24h",
    "geo_distance_from_last_txn",
    "amount_deviation_from_user_mean",
    "is_international",
]

CATEGORICAL_FEATURES = [
    "credit_score_band",
    "kyc_level",
    "payment_channel",
    "device_type",
]

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

TARGET = "is_fraud"


# ============================================================
# METRICS
# ============================================================

def evaluate_model(
    name,
    model,
    X,
    y
):

    probabilities = model.predict_proba(X)[:, 1]

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

    print(
        f"accuracy     : {metrics['accuracy']:.4f}"
    )

    print(
        f"precision    : {metrics['precision']:.4f}"
    )

    print(
        f"recall       : {metrics['recall']:.4f}"
    )

    print(
        f"f1           : {metrics['f1']:.4f}"
    )

    print(
        f"roc_auc      : {metrics['roc_auc']:.4f}"
    )

    print(
        f"pr_auc       : {metrics['pr_auc']:.4f}"
    )

    print(
        f"brier_score  : {metrics['brier_score']:.4f}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RECOVERZ RISK MODEL TRAINING")
    print("=" * 70)

    print(
        f"\nLoading:\n{DATA_FILE}"
    )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"\nRows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    required_columns = (
        FEATURES
        + [
            TARGET,
            "transaction_time",
        ]
    )

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + str(missing)
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            TARGET,
            "transaction_time",
        ]
    )

    # --------------------------------------------------------
    # Parse transaction time
    # --------------------------------------------------------

    df["transaction_time"] = pd.to_datetime(
        df["transaction_time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "transaction_time"
        ]
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "transaction_time"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Show fraud distribution
    # --------------------------------------------------------

    print(
        "\nFraud distribution:"
    )

    print(
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    fraud_rate = df[TARGET].mean()

    print(
        f"\nFraud rate: {fraud_rate:.4%}"
    )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    n = len(df)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    train_df = df.iloc[
        :train_end
    ]

    validation_df = df.iloc[
        train_end:validation_end
    ]

    test_df = df.iloc[
        validation_end:
    ]

    print("\n" + "=" * 70)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 70)

    print(
        f"Train      : {len(train_df):,}"
    )

    print(
        f"Validation : {len(validation_df):,}"
    )

    print(
        f"Test       : {len(test_df):,}"
    )

    print(
        f"\nTrain dates:"
        f" {train_df['transaction_time'].min()}"
        f" → {train_df['transaction_time'].max()}"
    )

    print(
        f"Validation dates:"
        f" {validation_df['transaction_time'].min()}"
        f" → {validation_df['transaction_time'].max()}"
    )

    print(
        f"Test dates:"
        f" {test_df['transaction_time'].min()}"
        f" → {test_df['transaction_time'].max()}"
    )

    # ========================================================
    # DATA
    # ========================================================

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ].astype(int)

    X_validation = validation_df[
        FEATURES
    ]

    y_validation = validation_df[
        TARGET
    ].astype(int)

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        TARGET
    ].astype(int)

    # ========================================================
    # PREPROCESSING
    # ========================================================

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
        )
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
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
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
            )
        ]
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=250,
                max_depth=14,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ),
    }

    trained_models = {}

    validation_results = {}

    # ========================================================
    # TRAIN
    # ========================================================

    for name, estimator in models.items():

        print("\n" + "=" * 70)

        print(
            f"TRAINING: {name}"
        )

        print("=" * 70)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    estimator
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            name + " - VALIDATION",
            pipeline,
            X_validation,
            y_validation
        )

        trained_models[name] = pipeline

        validation_results[name] = metrics

    # ========================================================
    # SELECT BEST MODEL
    #
    # PR-AUC is used because fraud is normally imbalanced.
    # ========================================================

    best_name = max(
        validation_results,
        key=lambda name:
            validation_results[name]["pr_auc"]
    )

    best_pipeline = trained_models[
        best_name
    ]

    print("\n" + "=" * 70)

    print(
        f"BEST RISK MODEL: {best_name}"
    )

    print("=" * 70)

    # ========================================================
    # TEST
    # ========================================================

    test_metrics = evaluate_model(
        best_name + " - TEST",
        best_pipeline,
        X_test,
        y_test
    )

    # ========================================================
    # CALIBRATION
    # ========================================================

    print(
        "\nCalibrating risk model..."
    )

    calibrated_model = (
        CalibratedClassifierCV(
            FrozenEstimator(
                best_pipeline
            ),
            method="sigmoid"
        )
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

    # ========================================================
    # SAVE
    # ========================================================

    import joblib

    joblib.dump(
        calibrated_model,
        MODEL_FILE
    )

    metadata = {

        "model_name": best_name,

        "target": TARGET,

        "features": FEATURES,

        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "excluded_features": [
            "transaction_id",
            "customer_id",
            "merchant_id",
            "post_auth_risk_score"
        ],

        "split": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
            "method": "chronological"
        },

        "validation_metrics":
            validation_results[best_name],

        "test_metrics":
            test_metrics,

        "calibrated_test_metrics":
            calibrated_metrics,

        "fraud_rate":
            float(fraud_rate),

        "rows":
            int(len(df)),

        "calibration":
            "sigmoid",

        "random_state":
            42
    }

    with open(
        METADATA_FILE,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print("\n" + "=" * 70)

    print(
        "RISK MODEL SAVED"
    )

    print("=" * 70)

    print(
        f"\nModel:\n{MODEL_FILE}"
    )

    print(
        f"\nMetadata:\n{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()