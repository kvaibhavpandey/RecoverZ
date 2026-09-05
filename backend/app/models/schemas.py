from typing import Optional, Literal
from pydantic import BaseModel, Field

Action = Literal[
    "RETRY",
    "RECOVERY_LINK",
    "ALTERNATIVE_PAYMENT",
    "ESCALATE",
    "STOP",
]

class AnalyzeResponse(BaseModel):
    case_id: str
    payment_id: str
    recovery_probability: float
    expected_recovery_value: float
    diagnosis: str
    recommended_action: Action
    confidence: float
    policy_result: str
    final_action: Action
    reasons: list[str]

class ExecuteResponse(BaseModel):
    case_id: str
    final_action: Action
    status: str
    outcome: str
    recovered_amount: float
    replayed: bool = False
    payment_link: Optional[str] = None

class WebhookPayload(BaseModel):
    event: str
