# RecoverZ Architecture

```text
React merchant dashboard
          |
          v
       FastAPI
          |
   +------+------+
   |             |
SQLite       Recovery Engine
                 |
        +--------+--------+
        |        |        |
     Features   ML      Diagnosis
                 |        |
                 +--------+
                      |
                 Next action
                      |
                 Policy Engine
                      |
              +-------+-------+
              |               |
           Execute          Stop/Escalate
              |
       Simulation / Razorpay
              |
          Verification
              |
          Audit + Metrics
```

The critical trust boundary is:

`LLM recommendation -> deterministic policy -> executor`

The LLM cannot bypass policy or call payment infrastructure directly.
