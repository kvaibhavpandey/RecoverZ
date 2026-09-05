from ..core.config import (
    MAX_AUTONOMOUS_RETRIES,
    AUTO_ACTION_AMOUNT_CAP,
    MIN_RECOVERY_PROBABILITY,
    ALLOWED_ACTIONS,
)

def apply_policy(payment: dict, recommended_action: str, probability: float):
    reasons = []

    if recommended_action not in ALLOWED_ACTIONS:
        return "STOP", ["Action is not allowlisted."]

    if payment["attempt_count"] >= MAX_AUTONOMOUS_RETRIES:
        return "ESCALATE", [f"Autonomous retry limit of {MAX_AUTONOMOUS_RETRIES} reached."]

    if payment["amount"] > AUTO_ACTION_AMOUNT_CAP:
        return "ESCALATE", [f"Amount exceeds unattended action cap of ₹{AUTO_ACTION_AMOUNT_CAP:,.0f}."]

    if probability < MIN_RECOVERY_PROBABILITY:
        return "STOP", [f"Recovery probability is below {MIN_RECOVERY_PROBABILITY:.0%}."]

    reasons.append("Within retry limit.")
    reasons.append("Within unattended amount cap.")
    reasons.append("Recovery probability clears policy floor.")
    reasons.append("Recommended action is allowlisted.")

    if recommended_action in {"ESCALATE", "STOP"}:
        return recommended_action, reasons

    return recommended_action, reasons
