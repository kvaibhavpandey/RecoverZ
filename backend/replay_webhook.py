import sqlite3
import urllib.request

db = sqlite3.connect("data/recoverz.db")

row = db.execute("""
    SELECT event_id, payload_json
    FROM webhook_events
    WHERE event_type = 'payment_link.paid'
    ORDER BY created_at DESC
    LIMIT 1
""").fetchone()

db.close()

if not row:
    print("NO PAYMENT_LINK.PAID EVENT FOUND")
    raise SystemExit

event_id, payload = row

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/webhooks/razorpay",
    data=payload.encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "X-Razorpay-Signature": "",
        "x-razorpay-event-id": "replay-" + event_id,
    },
    method="POST",
)

try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode())
except Exception as e:
    print(e)
