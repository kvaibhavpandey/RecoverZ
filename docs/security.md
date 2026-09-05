# RecoverZ Security Notes

- Razorpay secrets are backend-only environment variables.
- LLM outputs are restricted to an allowlist.
- Policy is deterministic and authoritative.
- High-value and repeated-attempt cases can escalate.
- Recovery execution uses a unique idempotency key.
- Webhook signatures are checked against the raw request body.
- Webhook event IDs are deduplicated.
- Simulation and Razorpay Test Mode outcomes are explicitly separated.
- The prototype should never be used with live credentials.
