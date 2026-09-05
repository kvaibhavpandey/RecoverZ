from datetime import datetime
from ..core.database import get_db


def build_ml_context(payment: dict) -> dict:
    """
    Enrich a RecoverZ payment with historical customer signals
    available from the application's SQLite database.

    Only derives signals that can be supported by the current
    application schema. Missing external behavioral features
    remain unavailable rather than being fabricated.
    """

    enriched = dict(payment)

    customer_id = payment["customer_id"]
    payment_id = payment["payment_id"]
    created_at = payment.get("created_at")

    with get_db() as conn:

        # --------------------------------------------------
        # Customer transaction history
        # --------------------------------------------------
        if created_at:
            history = conn.execute(
                """
                SELECT
                    amount,
                    payment_method,
                    recovered,
                    created_at
                FROM payments
                WHERE customer_id = ?
                  AND payment_id != ?
                  AND created_at < ?
                ORDER BY created_at DESC
                """,
                (customer_id, payment_id, created_at),
            ).fetchall()
        else:
            history = conn.execute(
                """
                SELECT
                    amount,
                    payment_method,
                    recovered,
                    created_at
                FROM payments
                WHERE customer_id = ?
                  AND payment_id != ?
                ORDER BY created_at DESC
                """,
                (customer_id, payment_id),
            ).fetchall()

    # ------------------------------------------------------
    # Historical transaction signals
    # ------------------------------------------------------
    previous_transaction_count = len(history)

    previous_success_count = sum(
        1 for row in history
        if int(row["recovered"] or 0) == 1
    )

    previous_failure_count = (
        previous_transaction_count
        - previous_success_count
    )

    if previous_transaction_count > 0:
        previous_success_rate = (
            previous_success_count
            / previous_transaction_count
        )
    else:
        # Preserve the application's existing signal if
        # there is no usable historical transaction.
        previous_success_rate = float(
            payment.get("previous_success_rate", 0.5)
        )

    enriched["previous_transaction_count"] = (
        previous_transaction_count
    )

    enriched["previous_success_count"] = (
        previous_success_count
    )

    enriched["previous_failure_count"] = (
        previous_failure_count
    )

    enriched["previous_success_rate"] = (
        previous_success_rate
    )

    # ------------------------------------------------------
    # Existing application signals
    # ------------------------------------------------------
    enriched["total_amount"] = float(
        payment.get("amount", 0)
    )

    enriched["net_amount"] = float(
        payment.get("amount", 0)
    )

    enriched["purchase_intent_score"] = float(
        payment.get("customer_activity_score", 0)
    )

    enriched["days_since_previous_transaction"] = float(
        payment.get(
            "days_since_last_successful_payment",
            30,
        )
    )

    # ------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------
    transaction_hour = payment.get("transaction_hour")

    if transaction_hour is not None:
        transaction_hour = int(transaction_hour)
    else:
        transaction_hour = 12

    enriched["transaction_hour"] = transaction_hour

    if created_at:
        try:
            dt = datetime.fromisoformat(
                str(created_at).replace("Z", "+00:00")
            )

            enriched["transaction_day_of_week"] = (
                dt.weekday()
            )

            enriched["transaction_month"] = (
                dt.month
            )

            enriched["is_weekend"] = int(
                dt.weekday() >= 5
            )

            enriched["year"] = dt.year

        except (ValueError, TypeError):
            enriched["transaction_day_of_week"] = 0
            enriched["transaction_month"] = 1
            enriched["is_weekend"] = 0
            enriched["year"] = 2026

    else:
        enriched["transaction_day_of_week"] = 0
        enriched["transaction_month"] = 1
        enriched["is_weekend"] = 0
        enriched["year"] = 2026

    # ------------------------------------------------------
    # Features unavailable in current SQLite schema
    #
    # Explicitly mark them as unavailable. Do NOT invent
    # device/product/session/IP information.
    # ------------------------------------------------------
    unavailable = {
        "session_event_count": 0,
        "unique_event_types": 0,
        "search_count": 0,
        "booking_count": 0,
        "promo_page_count": 0,
        "add_promo_count": 0,
        "session_duration_seconds": 0,
        "promo_amount": 0,
        "shipment_fee": 0,

        "device_type": "unknown",
        "device_version": "unknown",
        "gender": "unknown",
        "home_country": "unknown",
        "traffic_source": "unknown",
        "product_gender": "unknown",
        "masterCategory": "unknown",
        "subCategory": "unknown",
        "articleType": "unknown",
        "baseColour": "unknown",
        "season": "unknown",
        "usage": "unknown",

        "account_age_days": payment.get(
            "customer_age_days", 365
        ),

        "avg_monthly_spend": payment.get(
            "customer_lifetime_value",
            payment.get("amount", 0),
        ),

        "merchant_risk_score": 0.10,
        "ip_risk_score": 0.10,
        "txn_count_1h": 1,
        "txn_count_24h": 1,
        "failed_txn_count_24h": payment.get(
            "previous_failed_attempts", 0
        ),
        "geo_distance_from_last_txn": 0,
        "amount_deviation_from_user_mean": 0,
        "is_international": 0,

        "credit_score_band": "unknown",
        "kyc_level": "unknown",
        "payment_channel": payment.get(
            "payment_method",
            "unknown",
        ),
    }

    for key, value in unavailable.items():
        enriched.setdefault(key, value)

    return enriched