import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "recoverz.db"

for directory in (DATA_DIR, MODEL_DIR, RAW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "")

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

MAX_AUTONOMOUS_RETRIES = 2
AUTO_ACTION_AMOUNT_CAP = 10000
MIN_RECOVERY_PROBABILITY = 0.30
OUTREACH_COOLDOWN_HOURS = 6

ALLOWED_ACTIONS = {
    "RETRY",
    "RECOVERY_LINK",
    "ALTERNATIVE_PAYMENT",
    "ESCALATE",
    "STOP",
}
