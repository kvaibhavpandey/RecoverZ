RecoverZ — AI Revenue Recovery Agent

Recover failed payments into verified revenue — with AI for decisions, deterministic policy for safety, and Razorpay for execution.

RecoverZ is an AI-powered revenue recovery system built for Razorpay Buildathon 2026 — Track 03: AI Revenue Recovery.

It detects failed payments, diagnoses failure context, estimates recovery probability, selects a bounded intervention, applies deterministic safety policy, executes an allowed recovery action, verifies the outcome through Razorpay webhooks, and records an auditable recovery trail.

Core loop

DETECT → DIAGNOSE → SCORE → DECIDE → POLICY → EXECUTE → VERIFY → MEASURE

Trust boundary: LLM recommends. Policy authorizes. Executor acts.

Key capabilities

Failed-payment detection and customer/payment context enrichment

Calibrated ML recovery-probability scoring

Expected recovery value = amount × recovery probability

AI/deterministic diagnosis with allowlisted actions

Deterministic policy engine

Razorpay Test Mode Payment Link execution

Razorpay webhook HMAC-SHA256 verification

Webhook idempotency/deduplication

Verified recovery status and amount

Audit trail

Dashboard and evaluation benchmark

Allowlisted actions

RETRY · RECOVERY_LINK · ALTERNATIVE_PAYMENT · ESCALATE · STOP

Current safety policy

Control

Limit

Maximum autonomous attempts

2

Maximum unattended amount

₹10,000

Minimum recovery probability

30%

Architecture

flowchart LR
    A[Failed Payment] --> B[Context Builder]
    B --> C[ML Recovery Scoring]
    B --> D[Risk Scoring]
    C --> E[AI Diagnosis]
    D --> E
    E --> F[Deterministic Policy]
    F --> G{Allowed Action}
    G --> H[Recovery Executor]
    G --> I[Escalate / Stop]
    H --> J[Razorpay Test Mode]
    J --> K[Customer Payment]
    K --> L[Razorpay Webhook]
    L --> M[Signature Verification]
    M --> N[Case + Payment Update]
    N --> O[Audit Trail]
    N --> P[Dashboard]

Razorpay integration

RecoverZ has been verified end-to-end in Razorpay Test Mode:

RecoverZ case
  ↓
ALTERNATIVE_PAYMENT
  ↓
Razorpay Payment Link
  ↓
Test payment completed
  ↓
payment.captured / order.paid / payment_link.paid
  ↓
RecoverZ webhook
  ↓
HMAC verification + deduplication
  ↓
RECOVERED
  ↓
Recovered amount recorded
  ↓
Audit event

Verified case:

Field

Result

Case

REC-2263A612

Payment

pay_fresh_ae9a1ef1

Amount

₹4,200

Action

ALTERNATIVE_PAYMENT

Policy

APPROVED

Final status

RECOVERED

Outcome

PAYMENT_CONFIRMED

Recovered amount

₹4,200

The audit sequence is:

ANALYZED → EXECUTED → RECOVERY_CONFIRMED

No live money is used in the demo.

Evaluation

RecoverZ includes a synthetic holdout benchmark against a safety-matched naive retry baseline.

Metric

RecoverZ

Holdout cases

800

RecoverZ interventions

620

Successful recoveries

430

Recovery rate

69.35%

False intervention rate

30.65%

Recovered value / intervention

₹1,187.51

Interventions vs naive

−8.82%

False interventions vs naive

−7.79%

Revenue retained vs naive

96.15%

ROC-AUC

0.7521

Precision

0.7885

Recall

0.7395

F1

0.7632

Honest interpretation

RecoverZ does not claim higher absolute recovered revenue than the naive baseline in this benchmark.

The measured advantage is selectivity:

RecoverZ achieves a higher recovery rate and higher recovered value per intervention while reducing autonomous interventions by 8.82% and false interventions by 7.79%, retaining 96.15% of naive recovered revenue.

Disclosure: This is a synthetic holdout benchmark, not live merchant GMV or production recovery performance.

Tech stack

Backend: Python, FastAPI, SQLite, scikit-learn, joblib, Pydantic, python-dotenv

Frontend: React, Vite, Recharts, Lucide

Payments: Razorpay Payment Links, Razorpay Webhooks, HMAC-SHA256 verification

Project structure

RecoverZ/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   ├── routes/
│   │   └── services/
│   ├── ml/
│   │   ├── models/
│   │   ├── generate_data.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── data/
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.*
├── docs/
└── README.md

Local setup

Backend

cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

Backend: http://127.0.0.1:8000

Swagger: http://127.0.0.1:8000/docs

Frontend

cd frontend
npm install
npm run dev

Frontend: http://localhost:5173

Environment variables

Use a local .env file:

RAZORPAY_KEY_ID=your_test_key_id
RAZORPAY_KEY_SECRET=your_test_key_secret
RAZORPAY_WEBHOOK_SECRET=your_webhook_secret

Never commit real credentials, API keys, webhook secrets, .env files, or sensitive databases.

Testing

cd backend
pytest -q

Verified result:

4 passed

Evaluation:

python -m ml.evaluate

API

Endpoint

Purpose

GET /api/health

Health check

GET /api/dashboard

Dashboard metrics

GET /api/payments

Payment queue

GET /api/payments/{payment_id}

Payment details

GET /api/recovery-cases

Recovery case queue

GET /api/recovery-cases/{case_id}

Case details

POST /api/recovery-cases/{case_id}/analyze

Analyze and decide

POST /api/recovery-cases/{case_id}/execute

Execute approved action

GET /api/analytics

Recovery analytics

POST /api/webhooks/razorpay

Razorpay webhook receiver

Security and reliability

Deterministic policy enforcement

Allowlisted financial actions

Autonomous attempt limit

Unattended amount cap

Recovery-probability floor

HMAC webhook verification

Webhook event deduplication

Idempotent execution

Audit trail

STOP / ESCALATE paths

Known-case validation before webhook-driven recovery confirmation

No unrestricted LLM access to payment execution

Production hardening roadmap

For production deployment:

PostgreSQL

Secrets manager / KMS

Queue-based execution

Distributed idempotency

Dead-letter webhook handling

Merchant-specific policy configuration

Real-time risk signals

Device/IP/geo enrichment

Observability and alerting

RBAC

Encryption

Model drift monitoring

Human approval for high-value recoveries

Load and integration testing

Demo story

Failed payment
      ↓
RecoverZ analyzes context
      ↓
Recovery probability + expected value
      ↓
AI recommendation
      ↓
Deterministic policy
      ↓
Razorpay Payment Link
      ↓
Test payment
      ↓
Razorpay webhook
      ↓
Verified recovery
      ↓
Audit trail

Product principle

RecoverZ is not “an LLM that retries payments.”

It is:

A bounded revenue recovery agent combining predictive ML, AI diagnosis, deterministic financial policy, payment execution, webhook verification, and auditability.

Razorpay Buildathon 2026

Track 03 — AI Revenue Recovery

Built to demonstrate an end-to-end revenue recovery workflow from failed payment detection through verified recovery.