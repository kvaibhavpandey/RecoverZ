import json
import requests
from ..core.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, ALLOWED_ACTIONS
)

def _fallback(payment: dict, probability: float) -> dict:
    failure = payment["failure_code"]
    attempts = payment["attempt_count"]
    history = payment["previous_success_rate"]

    if attempts >= 2:
        action = "ESCALATE"
        diagnosis = "Repeated payment failure; another autonomous attempt is not appropriate."
    elif failure in {"network_error", "timeout"} and probability >= 0.55:
        action = "RETRY"
        diagnosis = "The failure appears transient and the customer has a strong payment history."
    elif failure == "authentication_failure":
        action = "ALTERNATIVE_PAYMENT"
        diagnosis = "Authentication failure makes an alternative payment path more appropriate."
    elif probability < 0.30:
        action = "STOP"
        diagnosis = "The estimated recovery likelihood is too low to justify another intervention."
    else:
        action = "RECOVERY_LINK"
        diagnosis = "A fresh recovery path is preferable to repeatedly retrying the same attempt."

    confidence = min(0.98, max(0.55, probability if action != "STOP" else 1 - probability))
    return {
        "diagnosis": diagnosis,
        "recommended_action": action,
        "reason": f"Failure={failure}; attempts={attempts}; historical success={history:.0%}.",
        "confidence": confidence,
        "provider": "deterministic-fallback",
    }

def diagnose(payment: dict, probability: float) -> dict:
    if not (LLM_API_KEY and LLM_BASE_URL and LLM_MODEL):
        return _fallback(payment, probability)

    prompt = f"""
You are a payment operations analyst. Diagnose this failed payment.
Return ONLY valid JSON with:
diagnosis, recommended_action, reason, confidence.
recommended_action must be one of:
RETRY, RECOVERY_LINK, ALTERNATIVE_PAYMENT, ESCALATE, STOP.
Do not invent payment APIs or financial actions.

Payment:
{json.dumps(payment, indent=2)}
Recovery probability: {probability:.4f}
""".strip()

    try:
        response = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=12,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        action = data.get("recommended_action")
        if action not in ALLOWED_ACTIONS:
            raise ValueError("LLM returned an unsupported action")
        return {
            "diagnosis": str(data.get("diagnosis", "No diagnosis")),
            "recommended_action": action,
            "reason": str(data.get("reason", "")),
            "confidence": float(data.get("confidence", probability)),
            "provider": "llm",
        }
    except Exception:
        fallback = _fallback(payment, probability)
        fallback["provider"] = "deterministic-fallback-after-llm-error"
        return fallback
