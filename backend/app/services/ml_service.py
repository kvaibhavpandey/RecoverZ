from pathlib import Path
import joblib
import pandas as pd
from ..core.config import MODEL_DIR
from ml.train import train, FEATURES

MODEL_PATH = MODEL_DIR / "recovery_rf.joblib"

def get_model():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)

def score_payment(payment: dict) -> float:
    model = get_model()
    row = pd.DataFrame([payment])[FEATURES]
    return float(model.predict_proba(row)[0, 1])
