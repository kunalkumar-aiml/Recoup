# Recoup — Limitations (Authoritative, Consolidated)

Stated plainly, up front, not discovered under questioning.

## 1. Synthetic data only
Every model, every number, every evaluation in this repository runs
against a synthetic data generator (`data/generate_data.py`) and a
synthetic counterfactual oracle. No real Razorpay transaction data was
used anywhere.

## 2. No real Razorpay traffic
The system has never processed a real payment event. All "recovered
revenue" figures (including `docs/final_business_results.md`) are
simulation outputs, clearly labeled as such.

## 3. 3-seed, not 10-seed, robustness
The headline lift (+18.2% single-seed) was tested across 3 seeds, not
the 5-10 that would give real statistical confidence. The 3-seed result
already shows wide variance (-0.5% to +49.1%, `docs/round3_findings.md`)
— more seeds would very likely sharpen this uncertainty, not resolve it
to a clean number.

## 4. DR rejected as primary policy due to instability
Cross-fitted doubly-robust (AIPW) estimation was implemented, validated
against synthetic ground truth, and initially showed a promising
single-seed result. A 3-seed validation found it won on 2 seeds and lost
outright on the third (`docs/final_policy_selection.md`) — DR is
retained as a validated research/evaluation module, not the production
estimator.

## 5. Weak overlap for the no_action control arm
73.8% of test contexts fall below common support for `no_action`
(`evaluation/overlap_diagnostics.py`). Every uplift number in this
system is measured against this baseline — its weak support is the
single biggest caveat on any causal claim this project makes.

## 6. Action ranking remains imperfect
Top-1 optimal-action accuracy is ~21%, Top-3 ~73% (Spearman rank
correlation 0.25). A forensic analysis found 72.3% of total regret
is associated with uplift-estimation error specifically
(`docs/final_action_ranking_analysis.md`) — diagnosed with increasing
specificity across several rounds of analysis, not yet fully resolved.

## 7. No real-world online A/B test
Everything here is offline evaluation (oracle-scored, or IPS/SNIPS/DR
off-policy estimates). No online experiment against real traffic has
been run or could be run without production access.

## 8. Production authentication/governance gaps
No authentication or authorization on any endpoint. No model registry,
version/checksum validation, promotion gate, shadow mode, or rollback
mechanism. Honest production-readiness self-score: 42/100 against a
90/100 target (`docs/round7_findings.md`).

## 9. No real traffic load benchmark
Latency and throughput at realistic transaction volumes are genuinely
untested — not "assumed fine," simply unmeasured. A basic concurrency
test (10/50/100 simultaneous threads) confirmed idempotency holds under
that load, which is a correctness property, not a performance
benchmark.

## 10. Causal identification assumptions
The T-learner's uplift estimate is causal only under three assumptions
(consistency, positivity/overlap, conditional exchangeability), stated
explicitly in `docs/causal_identification.md`. Consistency and
exchangeability are true by construction in this synthetic environment
(so not independently verified the way they'd need to be on real data);
positivity/overlap is measurably violated (item 5, above) and reported,
not assumed.

## Cross-cutting note

This document consolidates limitations previously scattered across
`docs/round3_findings.md` through `docs/round7_findings.md`,
`docs/model_cards.md`, and `docs/causal_identification.md`. Those
documents remain the detailed record; this is the one-page version for
a judge who wants the complete list without reading seven historical
documents.
