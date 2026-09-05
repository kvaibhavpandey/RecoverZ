from pathlib import Path
import pandas as pd
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.core.database import init_db, get_db
from ml.generate_data import generate
from ml.train import train
from ml.evaluate import evaluate
from app.services.recovery_service import create_case

def seed():
    init_db()
    generate(4000)
    train()
    evaluate()

    df = pd.read_csv(BACKEND / "data/raw/transactions.csv")
    # Replace first five rows with deterministic demo cases.
    demos = [
        ("pay_demo_a", "cust_demo_a", 2499, "upi", "network_error", 0, 0.91, 1),
        ("pay_demo_b", "cust_demo_b", 18000, "card", "bank_decline", 4, 0.82, 1),
        ("pay_demo_c", "cust_demo_c", 3500, "card", "bank_decline", 4, 0.22, 0),
        ("pay_demo_d", "cust_demo_d", 24000, "netbanking", "timeout", 0, 0.88, 1),
        ("pay_demo_e", "cust_demo_e", 4200, "card", "authentication_failure", 0, 0.76, 1),
    ]

    with get_db() as conn:
        conn.execute("DELETE FROM recovery_actions")
        conn.execute("DELETE FROM audit_events")
        conn.execute("DELETE FROM recovery_cases")
        conn.execute("DELETE FROM payments")
        for i, row in enumerate(demos):
            pid, cid, amount, method, failure, attempts, success_rate, recovered = row
            conn.execute("""
                INSERT INTO payments(
                    payment_id,customer_id,amount,currency,payment_method,failure_code,
                    attempt_count,previous_success_rate,customer_lifetime_value,
                    customer_age_days,days_since_last_successful_payment,
                    transaction_hour,previous_failed_attempts,customer_activity_score,
                    recovered,source
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                pid, cid, amount, "INR", method, failure, attempts, success_rate,
                amount * 8, 400, 5, 14, attempts, 0.85, recovered, "demo"
            ))

        # Add the remaining generated rows.
        for _, r in df.iloc[5:].iterrows():
            conn.execute("""
                INSERT INTO payments(
                    payment_id,customer_id,amount,currency,payment_method,failure_code,
                    attempt_count,previous_success_rate,customer_lifetime_value,
                    customer_age_days,days_since_last_successful_payment,
                    transaction_hour,previous_failed_attempts,customer_activity_score,
                    recovered,source
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, tuple(r[
                ["payment_id","customer_id","amount","currency","payment_method","failure_code",
                 "attempt_count","previous_success_rate","customer_lifetime_value",
                 "customer_age_days","days_since_last_successful_payment","transaction_hour",
                 "previous_failed_attempts","customer_activity_score","recovered","source"]
            ]))

    for pid, *_ in demos:
        create_case(pid)

    print("RecoverZ seeded: 4,000+ payments and 5 demo cases.")

if __name__ == "__main__":
    seed()
