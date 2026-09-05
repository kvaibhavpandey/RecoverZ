import json
from ..core.database import get_db

def audit(case_id: str, event_type: str, message: str, metadata=None):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO audit_events(case_id,event_type,message,metadata_json) VALUES(?,?,?,?)",
            (case_id, event_type, message, json.dumps(metadata or {})),
        )

def list_audit(case_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_events WHERE case_id=? ORDER BY id",
            (case_id,),
        ).fetchall()
    return [dict(r) for r in rows]
