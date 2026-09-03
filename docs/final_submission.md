# Recoup — Final Submission Summary

## What Recoup does

Recoup is a decision engine for payment revenue recovery. Given a
failed payment, it estimates the *incremental* recovery value of each
possible intervention, weighs that against each intervention's real
cost, checks its own confidence before acting, and enforces hard safety
limits no model score can override.

## Why it matters

Most systems that touch this problem stop at "predict whether this
payment will fail." Recoup answers the harder, more valuable question:
**given a failure already happened, which specific action creates the
most real economic value, and is the system sure enough to act alone?**

## What is novel

Not any single technique. The novelty is treating *evaluation* with the
same rigor as the system itself: one shared reward/regret definition
used by every script, causal claims stated with their identification
assumptions and a measured overlap diagnostic, a forensic regret
decomposition that traced 72.3% of lost value to a specific cause, and —
critically — a doubly-robust estimator that was implemented, validated,
initially adopted on a single seed, and then **correctly rejected** once
multi-seed testing showed it wasn't consistently better. That
self-correction is itself part of the evidence.

## What was measured (single seed unless noted — see limitations)

- Batch business result: for 541 synthetic failed payments (₹5,58,339 at
  risk), Recoup net-recovered ₹1,35,853 vs ₹1,14,900 (always-retry) and
  ₹1,31,189 (ML-only) — +18.2% / +3.6% lift (`docs/final_business_results.md`).
- 3-seed robustness: lift range -0.5% to +49.1% (mean ± std: 22.3% ±
  20.5%) — reported honestly as unstable.
- Regret decomposition: 90.2% of lost value from wrong-arm selection;
  a deeper forensic pass narrowed this to 72.3% from uplift-estimation
  error specifically (not the safety layer — only 3.8%).
- Cross-fitted DR: implemented, validated against synthetic ground
  truth, won on 2 of 3 real-data seeds but lost on the third — formally
  **rejected as primary policy**, retained as a research module.
- Calibration: isotonic reduced Brier 0.2524→0.2016, ECE 0.2041→0.0604.
- Concurrency: 10/50/100 simultaneous requests → exactly one decision;
  the same test caught and fixed a real `/feedback` race condition.
- Chaos testing: 15/15 scenarios pass against the live service.

## What actually worked

Leakage-safe temporal features (4 automated tests, all pass), a policy
gate that fails closed on any internal error, server-side temporal
state that a client cannot lie to (live-verified), idempotent
decision-making under real concurrent load, and a unified evaluation
protocol that eliminated a real contradiction between two scripts.

## What did not work / was not built

A candidate simpler no_action model was tested and correctly rejected
(fit the truth 170% worse, collapsed ranking accuracy). Cross-fitted DR
was built, validated, and ultimately rejected as the primary policy on
multi-seed evidence (see above — this is a completed, evidence-backed
negative result, not an unfinished item). Not attempted at all, given
the September 4 deadline: out-of-distribution testing, an adversarial
(model-family-mismatched) simulator, 10-seed robustness, model
governance (registry/promotion/rollback/shadow mode), authentication,
rate limiting, and load testing.

## How it is safe

Hard policy caps, confidence-tier gating, drift-aware gating, and a
fail-closed decision pipeline where any internal error routes to
mandatory human review.

## How it handles uncertainty

A 5-model bootstrap ensemble per intervention produces a confidence tier
that gates automation.

## How it recovers money

By comparing the incremental value of acting versus not acting against
each action's real cost — including discount leakage — rather than
simply predicting who is likely to pay anyway.

## Limitations (state these in the pitch, don't wait to be asked)

Full authoritative list: `docs/limitations.md`. Headlines: synthetic
data only, the headline lift is not stable across seeds, DR was rejected
on evidence not deferred by default, no production infrastructure, and
regret remains large relative to reward with the dominant cause
diagnosed but not yet fixed.
