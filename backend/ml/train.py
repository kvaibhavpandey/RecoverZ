from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .generate_data import generate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = DATA / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CSV = DATA / "raw" / "transactions.csv"
MODEL = MODEL_DIR / "recovery_rf.joblib"
METRICS = MODEL_DIR / "classification_metrics.json"

FEATURES = [
    "amount",
    "payment_method",
    "failure_code",
    "attempt_count",
    "previous_success_rate",
    "customer_lifetime_value",
    "customer_age_days",
    "days_since_last_successful_payment",
    "transaction_hour",
    "previous_failed_attempts",
    "customer_activity_score",
]

CAT = ["payment_method", "failure_code"]
NUM = [x for x in FEATURES if x not in CAT]

def train():
    if not CSV.exists():
        generate(4000)

    df = pd.read_csv(CSV)
    X = df[FEATURES]
    y = df["recovered"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", "passthrough", NUM),
    ])

    clf = RandomForestClassifier(
        n_estimators=220,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipe = Pipeline([("preprocess", pre), ("model", clf)])
    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
        "rows": int(len(df)),
    }

    joblib.dump(pipe, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    print(train())
