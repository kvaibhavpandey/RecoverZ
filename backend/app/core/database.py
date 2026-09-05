import sqlite3
from contextlib import contextmanager
from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS payments (
    payment_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'INR',
    payment_method TEXT NOT NULL,
    failure_code TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    previous_success_rate REAL NOT NULL,
    customer_lifetime_value REAL NOT NULL,
    customer_age_days INTEGER NOT NULL,
    days_since_last_successful_payment INTEGER NOT NULL,
    transaction_hour INTEGER NOT NULL,
    previous_failed_attempts INTEGER NOT NULL,
    customer_activity_score REAL NOT NULL,
    recovered INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'synthetic',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recovery_cases (
    case_id TEXT PRIMARY KEY,
    payment_id TEXT NOT NULL UNIQUE,
    recovery_probability REAL,
    expected_recovery_value REAL,
    diagnosis TEXT,
    recommended_action TEXT,
    confidence REAL,
    policy_result TEXT,
    final_action TEXT,
    status TEXT NOT NULL DEFAULT 'OPEN',
    outcome TEXT,
    recovered_amount REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(payment_id) REFERENCES payments(payment_id)
);

CREATE TABLE IF NOT EXISTS recovery_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    action TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    result TEXT NOT NULL,
    recovered_amount REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(case_id) REFERENCES recovery_cases(case_id)
);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(case_id) REFERENCES recovery_cases(case_id)
);

CREATE TABLE IF NOT EXISTS webhook_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.executescript(SCHEMA)
