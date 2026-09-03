# Recoup — Round 6 Findings (Production Hardening, subset)

Round 6's request (40 parts: chaos engineering, model registry, shadow
mode, circuit breakers, rate limiting, load testing, security hardening,
observability) is a full SRE/production-ML-platform build — realistically
weeks of work, not a single round. Following the same pattern as rounds
2-5, this round implemented and REALLY TESTED the highest-leverage,
most safety-critical, most tractable subset, and lists everything else
as explicitly not attempted.

## What was actually built and verified this round

### 1. Idempotency (Part 4/5) — built, tested live, works

`ml-service/decision_store.py` now maps `event_id -> decision_id` and
caches the full `/decide` response. Sending the same `event_id` three
times in a row returns the **exact same `decision_id`** on calls 2 and 3,
flagged `idempotent_replay: true`, without re-running model inference or
creating duplicate audit entries. Feedback is separately idempotent: a
`decision_id` that already received feedback returns
`{"updated": false, "already_processed": true}` on a repeat call rather
than updating the bandit a second time.

**Verified live** (not just code-reviewed): `chaos/chaos_test.py` tests
sent the same event 2x and the same feedback 2x against the running
service — both PASS.

**Stated limitation**: this is tested sequentially, not under real
concurrency (Part 22's "100 concurrent requests for the same event_id"
was NOT tested — the in-memory dict has no lock, so a true race between
two simultaneous requests for a brand-new event_id could still both miss
the cache and produce two decision_ids. A production version needs an
atomic check-and-set, e.g. a DB unique constraint on event_id).

### 2. Fail-closed decision states (Part 3) — built, tested live, works

`/decide` now returns an explicit `decision_state` field:
`AUTO_APPROVED | SAFE_FALLBACK | HUMAN_REVIEW | SYSTEM_ERROR` — a single
unambiguous field instead of three booleans a caller had to combine
correctly themselves. The entire decision pipeline is wrapped in a
try/except that returns `SYSTEM_ERROR` + `requires_human_review: true`
on any internal exception, rather than a bare 500 with no safe-fallback
signal.

**Verified live**: negative amount, zero amount, an extreme
₹50,000,000 amount, an unrecognized merchant category, and an
unrecognized decline code were all sent to the live service — none
crashed it, and the amount-based edge cases correctly resolved to
`SAFE_FALLBACK`.

### 3. Chaos test suite (Part 2) — built, run, 15/15 PASS

`chaos/chaos_test.py` runs 15 real failure-injection scenarios against
the live service (not mocked): duplicate event, duplicate feedback,
malformed/missing fields, negative/zero/extreme amount, non-numeric
amount (type error), unknown merchant category, unknown decline code,
extreme retry count, feedback against an unknown `decision_id`, a reward-
injection attempt via the feedback endpoint, high-value-transaction
safety, service liveness, and an empty decline_code (a legitimate case
for checkout-abandoned events). **All 15 passed on this run** — genuinely
verified, not asserted.

### 4. Reward integrity (Part 28) — was already true, now explicitly tested

`FeedbackIn`'s schema has no `reward` field at all — reward is always
derived server-side from `arm + recovered + amount` via the shared cost
table. This was already the case before round 6 (round-2 fix), but
round 6 added an explicit test attempting to inject
`{"reward": 999999}` into a feedback call — confirmed the extra field is
silently ignored by the schema and has zero effect.

## What round 6's 40-part request explicitly asked for that was NOT built

| Requested | Status | Why not |
|---|---|---|
| Concurrent/parallel duplicate-request race testing | NOT TESTED | Idempotency is tested sequentially only; the in-memory store has no lock for true concurrent first-writers |
| Database/ML-service/policy-service outage simulation, circuit breakers | NOT BUILT | No circuit breaker pattern implemented; the service either responds or the client's own request times out |
| Rate limiting | NOT BUILT | — |
| Load testing (10/50/100/500 RPS, p50/p95/p99) | NOT BUILT | No load-testing harness exists in this repo |
| Model registry with checksums/schema versioning/rejection | NOT BUILT | Models are loaded from disk with no version metadata or compatibility check |
| Model promotion pipeline (shadow → validate → promote), rollback | NOT BUILT | The bandit still updates in-place; no shadow-mode comparison exists |
| Out-of-order event handling at the temporal-feature level | NOT BUILT | `data/generate_data.py`'s incremental history dict assumes chronological arrival; the live service doesn't re-derive temporal features per-request from a live event stream at all (temporal features are client-supplied in `EventIn`, not computed server-side from history) — **this is a real, previously-undocumented architectural gap**: the live `/decide` API trusts the caller to supply correct `prior_failure_count` etc., it does not maintain that state itself |
| Stale-state TTL / staleness detection | NOT BUILT | — |
| Structured logging with request_id, hashed merchant/customer IDs | NOT BUILT | `audit.py` logs raw `event.dict()`, including merchant_category and other fields, unhashed — a real gap for a "no raw sensitive data in logs" requirement, though this build's fields are already synthetic/non-PII |
| Metrics/alerting (Prometheus-style counters, alert rules) | NOT BUILT | — |
| Full 40-scenario chaos matrix | PARTIAL (15/40) | The 25 not covered require infrastructure (DB, circuit breakers, load gen) this build doesn't have |

## Honest score against Part 34's production-readiness rubric

| Category | /Max | Score | Why |
|---|---|---|---|
| Reliability | /15 | 6 | Fail-closed states + idempotency are real; no circuit breakers, no timeout handling on the FastAPI side itself |
| Fault tolerance | /10 | 5 | 15 chaos scenarios pass; no outage/circuit-breaker testing |
| Observability | /10 | 2 | Audit log exists; no structured request IDs, no metrics, no alerting |
| Security | /10 | 4 | Reward injection correctly blocked; no rate limiting, no auth on any endpoint, PII not hashed in logs |
| Data quality | /10 | 3 | Pydantic schema validation catches malformed types; no formal data contract doc, no missingness/range monitoring pipeline |
| Model governance | /10 | 1 | No model registry, no version metadata, no promotion gate |
| Scalability | /10 | 0 | Not load-tested at all; genuinely unknown |
| Reproducibility | /10 | 6 | `evaluation/run_seed_robustness.sh` + fixed seeds give real reproducibility for the ML pipeline; no single `reproduce.py` covering the full stack including the live service |
| Auditability | /10 | 6 | decision_id + full context/result caching is real, in-memory only (not restored across restarts) |
| Safe fallback | /5 | 4 | `SAFE_FALLBACK`/`SYSTEM_ERROR` states are real and tested |
| **Total** | **/100** | **37** | Genuinely low relative to the /90 target stated in the request — this build is a research prototype with some real hardening, not a production system, and the score says so honestly |

## Combined overall project score (carrying forward round 5's 59/100 ML score)

The request also asked for an "overall project score." These are two
different rubrics (ML rigor vs production readiness) measuring different
things; reporting them separately rather than averaging them into one
number that would obscure which dimension is strong and which is weak:

- **ML/research rigor: 59/100** (round 5)
- **Production readiness: 37/100** (this round)

## The honest recommendation

This is a **hackathon submission for a 6-or-12-month internship
program**, not a system going to production Monday. The highest-value
use of any remaining time is almost certainly **not** a seventh round of
infrastructure hardening — it's making sure the pitch clearly states
what this repository actually demonstrates (a genuinely rigorous,
honestly-evaluated ML decision system) versus what it does not claim
(production infrastructure, which round 6 now has honest, tested
evidence for exactly how much exists: idempotency and fail-closed
states, and no more). Continuing to chase the full 40-part production
checklist has the same diminishing-returns shape round 5 already
flagged for the ML-rigor track.
