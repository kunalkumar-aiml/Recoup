# Final Policy Selection — DR vs T-learner, Multi-Seed Validation

## What this document does

V12's DR adoption decision ("ADOPT", +11.4% reward, -5.7% regret) was
**single-seed only** — exactly the trap round 3 already demonstrated is
easy to fall into (the original T-learner-vs-baseline headline swung
from -0.5% to +49.1% across just 3 seeds). This document runs the same
skepticism against V12's own DR result before treating it as final.

## Method

`evaluation/run_dr_multiseed_validation.sh` regenerates data, retrains
T-learner AND DR, and re-runs `evaluation/dr_cross_fitting.py`'s full
comparison for each of several seeds. **Scope note, stated honestly**:
5 seeds were planned; the full run exceeded this environment's execution
time limit partway through a 4th seed. **3 complete seeds (42, 101,
2026) are reported** — a real scope reduction under time pressure, not
a silently shrunk claim.

## Result

| Seed | T-learner reward | DR reward | Reward delta | Regret delta | Top-1 delta |
|---|---|---|---|---|---|
| 42 | ₹256.12 | ₹285.33 | **+11.4%** | +5.7% | -1.0pp |
| 101 | ₹249.76 | ₹287.52 | **+15.1%** | +7.0% | +7.5pp |
| 2026 | ₹298.60 | ₹287.38 | **-3.8%** | -2.1% | +3.2pp |

**Aggregate**: reward delta mean **+7.6% ± 8.2%** (std nearly as large
as the mean); regret delta mean +3.5% ± 4.0%; DR beat T-learner on
reward in **2 of 3 seeds**.

## Does DR pass V13's own adoption rule?

The rule (Step 10) requires, across multiple seeds: (1) consistently
better or non-inferior economic value, (2) regret not materially
worsened, (3) no safety regression, **(4) performance not driven by one
seed**, (5) overlap instability doesn't make the policy unreliable.

**Condition 4 is not met.** Seed 2026 shows DR *losing* to the current
T-learner on both reward (-3.8%) and regret (-2.1%) — this is not "one
outlier confirming robustness," it's one of three tested seeds actively
favoring the alternative conclusion. The standard deviation of the
reward delta (8.2 points) is nearly as large as its mean (7.6 points),
meaning the true effect could plausibly be anywhere from roughly zero to
double the single-seed V12 headline — the same shape of uncertainty
round 3 found for the original T-learner result.

## Final decision

**REJECT DR as the primary production policy. RETAIN DR as a validated
research/evaluation module.**

This is the same practical outcome V12 already chose (T-learner stays
in production) — but for a stronger, now evidence-based reason. V12's
stated reason was deadline risk ("don't swap production days before
submission"); **this document's reason is that the multi-seed evidence
itself doesn't clear the bar**, independent of deadline considerations.
Both reasons point the same direction, which is itself a reassuring
consistency check.

## Why did DR improve economic value in 2/3 seeds despite worse synthetic-adjacent MAE on the real no_action fit?

DR's no_action oracle-diagnostic MAE (0.1886) is worse than the
T-learner's (0.1084) in every seed tested (consistent with V12's finding
for seed 42). The plausible explanation, unchanged from V12: DR's
propensity-weighted correction improves the other five (better-supported)
arms enough to help in aggregate on some seeds, but this benefit is not
consistent enough across seeds to call it a settled win — exactly the
kind of nuance a single-seed report would have missed entirely.

## Overlap (unchanged, restated)

73.8% of contexts remain below common support for the `no_action` arm,
across all seeds tested (`evaluation/overlap_diagnostics.py`). This is
very plausibly *why* DR's win is seed-unstable: DR's correction term is
only as reliable as the propensity model's estimate in exactly the
region where that estimate is least trustworthy.

## Limitations of this validation itself

- 3 seeds, not the requested 5-10 — a real scope reduction, stated
  above, not hidden.
- No paired bootstrap confidence interval was computed (Step 5) — the
  aggregate mean/std reported above is a simpler, less rigorous summary
  than a proper paired bootstrap CI would give; flagged as unfinished,
  not substituted silently.
- High-value-segment-specific DR vs T-learner comparison (Step 8) was
  not run separately in this validation round.

## THE ML STOP LINE

Per this document's own instructions: **the current T-learner remains
Recoup's production estimator. DR is retained, documented, and
validated as a research/evaluation layer — not deployed.** No further
model architecture experiments follow this document. Remaining time
goes to README, demo, pitch, judge Q&A, and submission packaging.

## Judge Q&A additions

**Why T-learner in production, not DR, given DR showed a reward gain?**
Multi-seed validation (this document) found DR's apparent win doesn't
hold up consistently — it lost on one of three tested seeds, with a
standard deviation nearly as large as the mean effect. Both this
finding and V12's independent deadline-risk reasoning point to the same
conclusion: keep T-learner in production.

**Why did you even build DR if you're not using it?**
To test a specific, evidence-backed hypothesis (Section 6,
`docs/final_action_ranking_analysis.md`: 72.3% of regret is associated with
uplift/no_action-baseline error) with the estimator theoretically
best-suited to correct it. The honest negative-to-mixed result is
itself the valuable output — it tells us the no_action overlap problem
is not simply solved by a better estimator alone, which is useful
information for anyone continuing this work.

**Is this rigor itself part of your pitch?**
Yes, explicitly. A team that ships a flattering single-seed number is
easier to attack than one that ran the multi-seed check on its own
result and reported the answer honestly, including when that answer
walks back an earlier "ADOPT" decision.
