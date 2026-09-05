import uuid
from ..core.database import get_db
from .audit_service import audit
from ..core.config import RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET
import requests
import time

def create_case(payment_id: str):
    with get_db() as conn:
        payment = conn.execute(
            "SELECT * FROM payments WHERE payment_id=?", (payment_id,)
        ).fetchone()
        if not payment:
            raise ValueError("Payment not found")

        existing = conn.execute(
            "SELECT * FROM recovery_cases WHERE payment_id=?", (payment_id,)
        ).fetchone()
        if existing:
            return dict(existing)

        case_id = f"REC-{uuid.uuid4().hex[:8].upper()}"
        conn.execute(
            "INSERT INTO recovery_cases(case_id,payment_id,status) VALUES(?,?,?)",
            (case_id, payment_id, "OPEN"),
        )
    audit(case_id, "CASE_OPENED", "Recovery case opened.", {"payment_id": payment_id})
    return get_case(case_id)

def get_case(case_id: str):
    with get_db() as conn:
        row = conn.execute("""
            SELECT c.*, p.customer_id, p.amount, p.currency, p.payment_method,
                   p.failure_code, p.attempt_count, p.previous_success_rate,
                   p.customer_lifetime_value, p.customer_age_days,
                   p.days_since_last_successful_payment,
                   p.transaction_hour, p.previous_failed_attempts,
                   p.customer_activity_score, p.recovered, p.source
            FROM recovery_cases c
            JOIN payments p ON p.payment_id=c.payment_id
            WHERE c.case_id=?
        """, (case_id,)).fetchone()
    return dict(row) if row else None

def update_analysis(case_id, probability, expected_value, diagnosis,
                    recommended_action, confidence, policy_result, final_action, reasons):
    with get_db() as conn:
        conn.execute("""
            UPDATE recovery_cases
            SET recovery_probability=?, expected_recovery_value=?, diagnosis=?,
                recommended_action=?, confidence=?, policy_result=?,
                final_action=?, status='ANALYZED', updated_at=CURRENT_TIMESTAMP
            WHERE case_id=?
        """, (
            probability, expected_value, diagnosis, recommended_action,
            confidence, policy_result, final_action, case_id
        ))
    audit(case_id, "ANALYZED", "Recovery analysis completed.", {
        "probability": probability,
        "expected_value": expected_value,
        "recommended_action": recommended_action,
        "final_action": final_action,
        "policy_reasons": reasons,
    })

def execute_case(case_id: str):
    case = get_case(case_id)
    if not case:
        raise ValueError("Case not found")
    if not case["final_action"]:
        raise ValueError("Analyze the case before executing.")

    action = case["final_action"]
    idem = f"rz-{case_id}-{action}"

    with get_db() as conn:
        existing = conn.execute(
            "SELECT * FROM recovery_actions WHERE idempotency_key=?", (idem,)
        ).fetchone()
        if existing:
            return {
                "case_id": case_id,
                "final_action": action,
                "status": "REPLAYED",
                "outcome": existing["result"],
                "recovered_amount": existing["recovered_amount"],
                "replayed": True,
                "payment_link": None,
            }

    if action in {"STOP", "ESCALATE"}:
        outcome = "STOPPED" if action == "STOP" else "ESCALATED"
        recovered = 0.0
        link = None
    elif RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET and action in {"RECOVERY_LINK", "ALTERNATIVE_PAYMENT"}:
        link = _create_payment_link(case, idem)
        outcome = "RECOVERY_LINK_CREATED"
        recovered = 0.0
    else:
        # Simulation adapter. Uses the synthetic ground-truth label.
        link = None
        recovered = float(case["amount"]) if case["recovered"] else 0.0
        outcome = "RECOVERED" if recovered > 0 else "FAILED"

    with get_db() as conn:
        conn.execute("""
            INSERT INTO recovery_actions(case_id,action,idempotency_key,result,recovered_amount)
            VALUES(?,?,?,?,?)
        """, (case_id, action, idem, outcome, recovered))
        conn.execute("""
            UPDATE recovery_cases
            SET status='EXECUTED', outcome=?, recovered_amount=?, updated_at=CURRENT_TIMESTAMP
            WHERE case_id=?
        """, (outcome, recovered, case_id))

    audit(case_id, "EXECUTED", f"Final action {action} executed.", {
        "outcome": outcome,
        "recovered_amount": recovered,
        "idempotency_key": idem,
    })
    return {
        "case_id": case_id,
        "final_action": action,
        "status": "EXECUTED",
        "outcome": outcome,
        "recovered_amount": recovered,
        "replayed": False,
        "payment_link": link,
    }

def _create_payment_link(case, idem):
    payload = {
        "amount": int(round(case["amount"] * 100)),
        "currency": case["currency"],
        "accept_partial": False,
        "reference_id": idem[:40],
        "description": f"RecoverZ recovery for {case['payment_id']}",
        "customer": {
            "name": case["customer_id"],
        },
        "reminder_enable": False,
    }
    response = requests.post(
        "https://api.razorpay.com/v1/payment_links",
        auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
        json=payload,
        timeout=12,
    )
    response.raise_for_status()
    return response.json().get("short_url")
