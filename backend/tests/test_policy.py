import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.policy_service import apply_policy

def base():
    return {
        "amount": 2499,
        "attempt_count": 0,
    }

def test_high_value_escalates():
    p = base()
    p["amount"] = 18000
    final, _ = apply_policy(p, "RETRY", 0.9)
    assert final == "ESCALATE"

def test_repeated_attempts_escalate():
    p = base()
    p["attempt_count"] = 4
    final, _ = apply_policy(p, "RETRY", 0.9)
    assert final == "ESCALATE"

def test_low_probability_stops():
    final, _ = apply_policy(base(), "RETRY", 0.18)
    assert final == "STOP"
