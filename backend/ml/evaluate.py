from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .generate_data import generate
from .train import train, FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CSV = DATA / "raw" / "transactions.csv"
MODEL = DATA / "models" / "recovery_rf.joblib"
METRICS = DATA / "models" / "classification_metrics.json"
OUT = DATA / "models" / "evaluation.json"

# These are the same hard safety constraints used by RecoverZ's policy layer.
MAX_AUTONOMOUS_ATTEMPTS = 2
MAX_UNATTENDED_AMOUNT = 10000
MIN_RECOVERY_PROBABILITY = 0.30


def evaluate():
    if not CSV.exists():
        generate(4000)
    if not MODEL.exists():
        train()

    df = pd.read_csv(CSV)
    _, holdout = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["recovered"],
    )

    model = joblib.load(MODEL)
    holdout = holdout.copy()
    holdout["recovery_probability"] = model.predict_proba(
        holdout[FEATURES]
    )[:, 1]
    holdout["expected_value"] = (
        holdout["amount"] * holdout["recovery_probability"]
    )

    # Fair comparison universe: both strategies are evaluated on the same
    # payments that are safe to attempt autonomously under the hard policy.
    # The difference is that Naive Retry attempts every safe payment, while
    # RecoverZ adds an ML probability gate before intervening.
    safe_universe = (
        (holdout["attempt_count"] < MAX_AUTONOMOUS_ATTEMPTS)
        & (holdout["amount"] <= MAX_UNATTENDED_AMOUNT)
    )

    naive = safe_universe
    recoverz = safe_universe & (
        holdout["recovery_probability"] >= MIN_RECOVERY_PROBABILITY
    )

    def metrics(mask):
        interventions = int(mask.sum())
        successes = int((mask & (holdout["recovered"] == 1)).sum())
        false_interventions = int((mask & (holdout["recovered"] == 0)).sum())
        recovered_value = float(
            holdout.loc[mask & (holdout["recovered"] == 1), "amount"].sum()
        )
        return {
            "interventions": interventions,
            "successful_recoveries": successes,
            "recovered_value": round(recovered_value, 2),
            "recovery_rate": round(successes / interventions, 4) if interventions else 0,
            "false_intervention_rate": round(
                false_interventions / interventions, 4
            ) if interventions else 0,
            "recovered_value_per_intervention": round(
                recovered_value / interventions, 2
            ) if interventions else 0,
        }

    naive_m = metrics(naive)
    recoverz_m = metrics(recoverz)

    revenue_retained_vs_naive = (
        recoverz_m["recovered_value"] / naive_m["recovered_value"]
        if naive_m["recovered_value"]
        else 0
    )
    intervention_reduction = (
        1 - recoverz_m["interventions"] / naive_m["interventions"]
        if naive_m["interventions"]
        else 0
    )
    false_intervention_reduction = (
        1
        - recoverz_m["false_intervention_rate"]
        / naive_m["false_intervention_rate"]
        if naive_m["false_intervention_rate"]
        else 0
    )

    model_metrics = {}
    if METRICS.exists():
        model_metrics = json.loads(METRICS.read_text())

    result = {
        "holdout_cases": int(len(holdout)),
        "revenue_at_risk": round(float(holdout["amount"].sum()), 2),
        "safe_autonomous_universe": int(safe_universe.sum()),
        "no_intervention_recovered": 0.0,
        "naive_recovered": naive_m["recovered_value"],
        "recoverz_recovered": recoverz_m["recovered_value"],
        "naive": naive_m,
        "recoverz": recoverz_m,
        "revenue_retained_vs_naive": round(revenue_retained_vs_naive, 4),
        "intervention_reduction_vs_naive": round(intervention_reduction, 4),
        "false_intervention_reduction_vs_naive": round(false_intervention_reduction, 4),
        # Backward-compatible fields used by the existing UI.
        "recovery_rate": recoverz_m["recovery_rate"],
        "eligible_cases": recoverz_m["interventions"],
        "successful_recoveries": recoverz_m["successful_recoveries"],
        "false_intervention_rate": recoverz_m["false_intervention_rate"],
        "recovery_uplift_vs_no_intervention": None,
        "model_metrics": model_metrics,
        "policy": {
            "max_autonomous_attempts": MAX_AUTONOMOUS_ATTEMPTS,
            "max_unattended_amount": MAX_UNATTENDED_AMOUNT,
            "min_recovery_probability": MIN_RECOVERY_PROBABILITY,
        },
        "note": (
            "Synthetic holdout benchmark; not live merchant GMV. "
            "Naive Retry and RecoverZ use the same hard safety universe; "
            "RecoverZ adds an ML probability gate."
        ),
    }

    OUT.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
