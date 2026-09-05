import json
from fastapi import APIRouter, HTTPException, Request
from ..services.ml_context_service import build_ml_context
from ..core.database import get_db
from ..services.recovery_service import create_case, get_case, update_analysis, execute_case
from ..services.ml_service import score_payment
from ..services.diagnosis_service import diagnose
from ..services.policy_service import apply_policy
from ..services.audit_service import list_audit
from ..core.config import RAZORPAY_WEBHOOK_SECRET
import hashlib
import hmac

router = APIRouter(prefix="/api")

def rows(query, params=()):
    with get_db() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]

@router.get("/health")
def health():
    return {"status": "healthy", "service": "RecoverZ"}

@router.get("/dashboard")
def dashboard():
    with get_db() as conn:
        row = conn.execute("""
            SELECT
              COUNT(*) cases,
              COALESCE(SUM(amount),0) revenue_at_risk,
              COALESCE(SUM(CASE WHEN recovered=1 THEN amount ELSE 0 END),0) labelled_recoverable
            FROM payments
        """).fetchone()
        recovered = conn.execute(
            "SELECT COALESCE(SUM(recovered_amount),0) x FROM recovery_cases"
        ).fetchone()["x"]
        executed = conn.execute(
            "SELECT COUNT(*) x FROM recovery_actions"
        ).fetchone()["x"]
        successful = conn.execute(
             """
             SELECT COUNT(*) x
             FROM recovery_cases
             WHERE status='RECOVERED' OR outcome='RECOVERED'
             """
        ).fetchone()["x"]
        escalated = conn.execute(
            "SELECT COUNT(*) x FROM recovery_cases WHERE status='ESCALATED'"
        ).fetchone()["x"]

    return {
        "cases": row["cases"],
        "revenue_at_risk": round(row["revenue_at_risk"], 2),
        "revenue_recovered": round(recovered, 2),
        "recovery_rate": round(successful / executed, 4) if executed else 0,
        "executed_actions": executed,
        "successful_recoveries": successful,
        "escalations": escalated,
    }

@router.get("/payments")
def payments(limit: int = 100):
    return rows("""
        SELECT p.*, c.case_id, c.recovery_probability, c.expected_recovery_value,
               c.recommended_action, c.final_action, c.status, c.outcome, c.recovered_amount
        FROM payments p
        LEFT JOIN recovery_cases c ON c.payment_id=p.payment_id
        ORDER BY p.created_at DESC LIMIT ?
    """, (min(max(limit, 1), 500),))

@router.get("/payments/{payment_id}")
def payment(payment_id: str):
    with get_db() as conn:
        row = conn.execute("""
            SELECT p.*, c.case_id, c.recovery_probability, c.expected_recovery_value,
                   c.diagnosis, c.recommended_action, c.confidence,
                   c.policy_result, c.final_action, c.status, c.outcome,
                   c.recovered_amount
            FROM payments p
            LEFT JOIN recovery_cases c ON c.payment_id=p.payment_id
            WHERE p.payment_id=?
        """, (payment_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Payment not found")
    result = dict(row)
    if result.get("case_id"):
        result["audit"] = list_audit(result["case_id"])
    return result

@router.get("/recovery-cases")
def recovery_cases():
    return rows("""
        SELECT c.*, p.customer_id, p.amount, p.payment_method, p.failure_code,
               p.attempt_count, p.previous_success_rate
        FROM recovery_cases c
        JOIN payments p ON p.payment_id=c.payment_id
        ORDER BY c.created_at DESC
    """)

@router.get("/recovery-cases/{case_id}")
def recovery_case(case_id: str):
    result = get_case(case_id)
    if not result:
        raise HTTPException(404, "Recovery case not found")
    result["audit"] = list_audit(case_id)
    return result

@router.post("/recovery-cases/{case_id}/analyze")
def analyze_case(case_id: str):
    case = get_case(case_id)
    if not case:
        raise HTTPException(404, "Recovery case not found")

    payment = dict(case)

    payment = build_ml_context(payment)

    try:
        probability = score_payment(payment)
        ai = diagnose(payment, probability)
        final_action, reasons = apply_policy(
            payment, ai["recommended_action"], probability
        )
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}")

    expected = float(payment["amount"]) * probability
    update_analysis(
        case_id, probability, expected, ai["diagnosis"],
        ai["recommended_action"], ai["confidence"],
        "APPROVED" if final_action == ai["recommended_action"] else "BLOCKED",
        final_action, reasons
    )
    return {
        "case_id": case_id,
        "payment_id": payment["payment_id"],
        "recovery_probability": probability,
        "expected_recovery_value": expected,
        "diagnosis": ai["diagnosis"],
        "recommended_action": ai["recommended_action"],
        "confidence": ai["confidence"],
        "policy_result": "APPROVED" if final_action == ai["recommended_action"] else "BLOCKED",
        "final_action": final_action,
        "reasons": reasons,
        "ai_provider": ai.get("provider"),
    }

@router.post("/recovery-cases/{case_id}/execute")
def execute(case_id: str):
    try:
        return execute_case(case_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Execution failed safely: {exc}")

@router.get("/analytics")
def analytics():
    with get_db() as conn:
        by_failure = [
            dict(r) for r in conn.execute("""
                SELECT failure_code,
                       COUNT(*) cases,
                       COALESCE(SUM(recovered_amount),0) recovered
                FROM recovery_cases c
                JOIN payments p ON p.payment_id=c.payment_id
                GROUP BY failure_code
                ORDER BY recovered DESC
            """).fetchall()
        ]
        by_method = [
            dict(r) for r in conn.execute("""
                SELECT payment_method,
                       COUNT(*) cases,
                       COALESCE(SUM(recovered_amount),0) recovered
                FROM recovery_cases c
                JOIN payments p ON p.payment_id=c.payment_id
                GROUP BY payment_method
                ORDER BY recovered DESC
            """).fetchall()
        ]
    evaluation_path = __import__("pathlib").Path(__file__).resolve().parents[2] / "data/models/evaluation.json"
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
    return {"evaluation": evaluation, "by_failure": by_failure, "by_method": by_method}

@router.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request):
    raw = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    # 1. Verify Razorpay webhook signature
    if RAZORPAY_WEBHOOK_SECRET:
        expected = hmac.new(
            RAZORPAY_WEBHOOK_SECRET.encode(),
            raw,
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(expected, signature):
            raise HTTPException(400, "Invalid webhook signature")

    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON payload")

    event_id = request.headers.get("x-razorpay-event-id", "")
    event_type = payload.get("event", "unknown")

    if not event_id:
        event_id = hashlib.sha256(raw).hexdigest()

    with get_db() as conn:

        # 2. Idempotency: ignore duplicate Razorpay events
        existing = conn.execute(
            "SELECT event_id FROM webhook_events WHERE event_id=?",
            (event_id,),
        ).fetchone()

        if existing:
            return {
                "ok": True,
                "replayed": True,
                "event": event_type,
            }

        # 3. Store webhook event
        conn.execute(
            """
            INSERT INTO webhook_events(
                event_id,
                event_type,
                payload_json
            )
            VALUES(?,?,?)
            """,
            (
                event_id,
                event_type,
                raw.decode("utf-8"),
            ),
        )

        # 4. Handle successful Payment Link payment
        if event_type == "payment_link.paid":

            payment_link_entity = (
                payload
                .get("payload", {})
                .get("payment_link", {})
                .get("entity", {})
            )

            reference_id = payment_link_entity.get("reference_id")
            paid_amount = payment_link_entity.get("amount_paid")

            if reference_id and reference_id.startswith("rz-"):

                # Reference format:
                # rz-<CASE_ID>-<ACTION>
                parts = reference_id.split("-")

                if len(parts) >= 3:
                    case_id = "-".join(parts[1:2])

                    # RecoverZ case IDs are REC-XXXXXXXX.
                    # Extract the REC-XXXXXXXX portion safely.
                    if len(parts) >= 3 and parts[1].startswith("REC"):
                        case_id = f"{parts[1]}-{parts[2]}"
                    else:
                        case_id = None

                    if case_id:
                        case = conn.execute(
                            """
                            SELECT case_id, payment_id, status
                            FROM recovery_cases
                            WHERE case_id=?
                            """,
                            (case_id,),
                        ).fetchone()

                        if case:
                            # Razorpay amounts are in paise.
                            recovered_amount = (
                                paid_amount / 100
                                if paid_amount is not None
                                else 0
                            )

                            # 5. Mark recovery as successfully recovered
                            conn.execute(
                                """
                                UPDATE recovery_cases
                                SET status='RECOVERED',
                                    outcome='PAYMENT_CONFIRMED',
                                    recovered_amount=?,
                                    updated_at=CURRENT_TIMESTAMP
                                WHERE case_id=?
                                """,
                                (
                                    recovered_amount,
                                    case_id,
                                ),
                            )

                            # 6. Mark original payment as recovered
                            conn.execute(
                                """
                                UPDATE payments
                                SET recovered=1
                                WHERE payment_id=?
                                """,
                                (case["payment_id"],),
                            )

                            # 7. Add audit trail
                            conn.execute(
                                """
                                INSERT INTO audit_events(
                                    case_id,
                                    event_type,
                                    message,
                                    metadata_json
                                )
                                VALUES(?,?,?,?)
                                """,
                                (
                                    case_id,
                                    "RECOVERY_CONFIRMED",
                                    "Razorpay Payment Link payment confirmed",
                                    json.dumps({
                                        "event_id": event_id,
                                        "event_type": event_type,
                                        "reference_id": reference_id,
                                        "recovered_amount": recovered_amount,
                                    }),
                                ),
                            )

    return {
        "ok": True,
        "replayed": False,
        "event": event_type,
    }
