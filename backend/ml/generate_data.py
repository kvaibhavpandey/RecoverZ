from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT = RAW / "transactions.csv"

RNG = np.random.default_rng(42)

def generate(n=4000):
    methods = np.array(["upi", "card", "netbanking", "wallet"])
    failures = np.array([
        "network_error",
        "timeout",
        "bank_decline",
        "insufficient_funds",
        "authentication_failure",
        "unknown_failure",
    ])

    amount = RNG.lognormal(mean=7.2, sigma=0.75, size=n).clip(199, 50000).round(0)
    payment_method = RNG.choice(methods, n, p=[0.48, 0.34, 0.10, 0.08])
    failure_code = RNG.choice(
        failures, n, p=[0.24, 0.16, 0.22, 0.14, 0.14, 0.10]
    )
    attempt_count = RNG.poisson(0.65, n).clip(0, 5)
    previous_success_rate = RNG.beta(7, 2, n)
    customer_lifetime_value = (
        amount * RNG.uniform(2, 15, n)
    ).round(0)
    customer_age_days = RNG.integers(10, 1200, n)
    days_since_last_successful_payment = RNG.integers(0, 120, n)
    transaction_hour = RNG.integers(0, 24, n)
    previous_failed_attempts = RNG.poisson(0.8, n).clip(0, 6)
    customer_activity_score = RNG.beta(5, 2, n)

    transient = np.isin(failure_code, ["network_error", "timeout"]).astype(float)
    bank_decline = (failure_code == "bank_decline").astype(float)
    auth_fail = (failure_code == "authentication_failure").astype(float)

    # Correlated synthetic ground truth:
    # strong history + transient failure + fewer attempts -> more recoverable.
    logit = (
        -0.8
        + 2.0 * previous_success_rate
        + 1.0 * customer_activity_score
        + 0.9 * transient
        - 0.75 * bank_decline
        - 0.9 * auth_fail
        - 0.65 * attempt_count
        - 0.35 * previous_failed_attempts
        - 0.000018 * amount
        - 0.003 * days_since_last_successful_payment
    )

    p = 1 / (1 + np.exp(-logit))
    recovered = RNG.binomial(1, p)

    df = pd.DataFrame({
        "payment_id": [f"pay_{i:05d}" for i in range(n)],
        "customer_id": [f"cust_{RNG.integers(1000, 9999)}" for _ in range(n)],
        "amount": amount.astype(int),
        "currency": "INR",
        "payment_method": payment_method,
        "failure_code": failure_code,
        "attempt_count": attempt_count.astype(int),
        "previous_success_rate": previous_success_rate.round(4),
        "customer_lifetime_value": customer_lifetime_value.astype(int),
        "customer_age_days": customer_age_days.astype(int),
        "days_since_last_successful_payment": days_since_last_successful_payment.astype(int),
        "transaction_hour": transaction_hour.astype(int),
        "previous_failed_attempts": previous_failed_attempts.astype(int),
        "customer_activity_score": customer_activity_score.round(4),
        "recovered": recovered.astype(int),
        "source": "synthetic",
    })
    df.to_csv(OUT, index=False)
    return OUT

if __name__ == "__main__":
    print(generate())
