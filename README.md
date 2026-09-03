# Recoup — AI Revenue Recovery

**Razorpay AI Buildathon 2026 · Track: AI Revenue Recovery**

## 1. Problem

A failed payment isn't the end of a transaction — it's a decision
point most systems skip. They detect the failure and stop. Recoup
decides **what to do about it**.

## 2. Why this matters

Recoup doesn't ask "should I retry?" It asks: **which of five recovery
interventions creates the highest incremental net-recovery value for
this specific failed payment, and when should I *not* act at all?**

## 3. What Recoup does

Given a failed payment, Recoup estimates the *incremental* recovery
value of each possible intervention (not raw probability — the
difference vs. doing nothing), weighs it against real operational cost,
checks its own confidence, and enforces hard safety limits no model
score can override. Every decision produces a full, replayable audit
trace.

## 4. Architecture

```
PAYMENT EVENT
  -> SERVER-SIDE TEMPORAL STATE   (client cannot lie about history)
  -> RECOVERY MODELS + UPLIFT     (T-learner, calibrated)
  -> UNCERTAINTY                  (bootstrap ensemble -> confidence tier)
  -> DRIFT CHECK                  (multi-signal PSI)
  -> ECONOMIC VALUE                (uplift x amount - cost, per arm)
  -> SAFETY POLICY GATE           (hard caps, fail-closed)
  -> BOUNDED ACTION               (simulated -- no real money moves)
  -> AUDIT TRAIL                   (replayable by decision_id)
```

`LinUCB` (contextual bandit) is an informational/online-learning
component — exploration measurably helps, but it is not the primary
decision driver. `DR` (cross-fitted doubly-robust estimation) is a
validated research/evaluation module, not the production estimator
(Section 8). The **T-learner is Recoup's final, primary estimator.**

## 5. ML methodology

GradientBoosting per-arm recovery models (T-learner), isotonic
calibration, a 5-model bootstrap uncertainty ensemble, and PSI-based
drift detection across 4 signals. Leakage-safe temporal features are
computed server-side from strict event history (4 automated leakage
tests, all pass) — never trusted from the client.

## 6. Economic decision engine

`net_value(a) = uplift_a(X) * amount - cost(a)`, with discount leakage
netted separately. Every evaluation script in this repo shares one
reward/regret definition (`evaluation/protocol.py`) — this fixed a real
contradiction where two scripts used to disagree on the same number.
The engine's arithmetic is unit-tested in isolation (10/10 tests pass,
`ml-service/test_economic_value_engine.py`).

## 7. Safety

Fail-closed decision states (`AUTO_APPROVED | SAFE_FALLBACK |
HUMAN_REVIEW | SYSTEM_ERROR`), atomic idempotency (verified under real
concurrent load — the same test suite caught and fixed a real race
condition), server-side temporal state (live-verified: a client
claiming 999 fake prior failures has zero effect), and reward integrity
(no client-supplied reward field exists). 15/15 chaos scenarios pass
against the live service.

## 8. Final measured results (primary synthetic benchmark, seed 42)

**Primary synthetic benchmark (seed 42):** for 541 failed payments with
₹5.58L at risk, Recoup produced ₹1.36L net recovered value versus
₹1.15L for Always Retry — an incremental ₹20.95K (**+18.2%**).

**Robustness: across 3 seeds, lift ranged from -0.5% to +49.1%. We
therefore do not present +18.2% as a production performance guarantee.**
Full multi-seed detail: `docs/archive/round3_findings.md`.

| Strategy | Net recovered (seed 42) | Lift (seed 42) |
|---|---|---|
| Always retry | ₹1,14,900 | — |
| ML-only (cost-blind) | ₹1,31,189 | +14.2% |
| **Recoup** | **₹1,35,853** | **+18.2%** |

Full batch detail: `docs/final_business_results.md`.

**Cross-fitted DR was implemented and validated as a research/evaluation
module**, then tested against the T-learner across 3 seeds — DR won on
2 but lost on the third. **DR is REJECTED as the primary policy because
its multi-seed economic improvement was not sufficiently stable** —
this is not a claim that DR failed or is broken, only that it did not
clear this project's own adoption bar for *production* use
(`docs/final_policy_selection.md`). **The T-learner remains the final
primary estimator/policy.** This reversal of an earlier single-seed
"adopt DR" verdict is itself part of this project's evidence — the same
evaluation discipline that produced the headline number also caught and
corrected its own over-eager conclusion.

**Where the money is still lost**: mean regret vs. the oracle-optimal
action is ₹514/event — larger than Recoup's own mean reward. **72.3% of
measured regret is associated with uplift-estimation error** (a
diagnostic association from a heuristic forensic pass, not a controlled
causal-attribution experiment — see `docs/final_action_ranking_analysis.md`
for the method and its stated limits) — not raw probability error, and
not the safety layer (only 3.8% of regret is associated with policy
overrides).

## 9. Robustness (read before repeating any single number)

- **The +18.2% lift is not stable across seeds.** 3-seed range: -0.5%
  to +49.1%. Honest aggregate: +22.3% ± 20.5%. `docs/archive/round3_findings.md`.
- **DR's apparent win did not survive multi-seed testing** (Section 8).
- Calibration: isotonic reduced Brier 0.2524→0.2016, ECE 0.2041→0.0604.
- Concurrency: 10/50/100 simultaneous requests → exactly one decision,
  live-verified; the same test found and fixed a real `/feedback` race.
- 15/15 chaos scenarios pass (duplicate events, malformed input, reward
  injection attempts, high-value safety, more).
- **NOT built**: OOD test suite, adversarial/model-mismatched simulator,
  10-seed robustness, model registry/promotion/rollback/shadow mode,
  authentication, rate limiting, load testing. Full list:
  `docs/limitations.md`.

Production-readiness self-score: **42/100** against a 90/100 target —
a research prototype with real, tested hardening, not a production
system.

## 10. Demo

**Start with money, not architecture:**

1. Show the batch view: ₹5.58L revenue at risk across 541 failed
   payments (Section 8).
2. Drill into one failed payment → show the candidate interventions →
   incremental net value per intervention → uncertainty tier → safety
   policy check → the selected action → bounded (simulated) execution
   → the audit trail entry.
3. Trigger a high-value transaction → watch it require human review
   regardless of model confidence.
4. Trigger a duplicate event → watch idempotency return the identical
   decision, not a second one.
5. Run `chaos/chaos_test.py` scenarios live.

**Only after the above**, walk through the ML architecture, the DR
experiment and why it was rejected, and the regret-decomposition
finding as the honest "here's what we'd fix next" moment.

Full script and 38 prepared judge questions: `docs/pitch_and_judge_qa.md`.

## 11. Limitations

One authoritative list: `docs/limitations.md`. Headlines: synthetic
data only, 3-seed (not 10-seed) robustness, DR rejected on evidence,
no production auth/governance/load-testing, causal claims caveated by
a measured overlap weakness.

Repository trust/integrity audit (secrets, PII, oracle-leakage
boundaries, third-party licenses, results-consistency check):
`docs/final_repo_audit.md`, `docs/security_audit.md`,
`docs/data_leakage_audit.md`, `docs/third_party_and_licenses.md`,
`docs/github_tracked_files_audit.md`, `docs/repository_artifact_policy.md`.
**Git-history secret scan requires commands run against the actual
GitHub repository, not verifiable from this delivered code archive —
see `docs/security_audit.md` for the exact commands.**

## 12. Reproduction

```bash
# Data + training
cd data && python3 generate_data.py && cd ../ml-service
pip3 install -r requirements.txt && python3 train.py

# Correctness tests
python3 test_no_leakage.py
python3 test_economic_value_engine.py

# Start the service (keep running)
uvicorn app:app --reload --port 8000
```

In a second terminal:
```bash
cd backend && npm install && npm start   # :4000

cd ../evaluation
python3 evaluate.py                       # headline comparison
python3 final_business_benchmark.py       # batch-level ₹ table (Section 8)
python3 dr_cross_fitting.py               # DR vs T-learner, single seed
python3 judge_attack.py                   # every diagnostic, PASS/WARN/FAIL

cd ../ml-service
python3 test_server_side_state.py
python3 test_concurrency.py
cd ../chaos && python3 chaos_test.py
```

Open `frontend/index.html` directly — no build step. Full historical
record of every round of analysis (evaluation-protocol fixes, regret
decomposition, DR validation, production hardening) is archived in
`docs/archive/`.

## Author

Kunal Kumar — AI/ML Engineer
GitHub: [github.com/kunalkumar-aiml](https://github.com/kunalkumar-aiml)
LinkedIn: [linkedin.com/in/kunal-kumar-382b25336](https://linkedin.com/in/kunal-kumar-382b25336)

Recoup — AI Revenue Recovery, built for the Razorpay AI Buildathon 2026.
