# Recoup — Causal Identification Assumptions

The T-learner uplift estimate `tau_a(X) = mu_a(X) - mu_no_action(X)` is
a valid estimate of the conditional average treatment effect `E[Y(a) -
Y(0) | X]` **only if three identification assumptions hold**. We state
them explicitly rather than assume them silently, and report exactly
where they are weak in this build.

## 1. Consistency
The observed outcome for units that received arm `a` equals their
potential outcome under `a`. **True by construction** in our synthetic
environment — the logged `recovered` value for an event IS that event's
potential outcome under whichever arm was actually chosen
(`data/generate_data.py`). In a real deployment this needs its own
justification (e.g. no interference between customers).

## 2. Positivity / overlap
Every arm must have nonzero probability of being chosen for every
context region we predict on. **Measurably violated here.**
`evaluation/overlap_diagnostics.py` reports, per arm, the minimum
estimated propensity and effective sample size on the test split. The
control arm (`no_action`) is the weakest: **73.8% of test-split contexts
fall below the 0.05 common-support threshold for `no_action`** on this
run. This means the uplift baseline `mu_no_action(X)` is extrapolated,
not directly supported by logged data, for the large majority of
contexts — the single biggest caveat on any causal claim this project
makes, and we are not hiding it.

## 3. Conditional ignorability (no unmeasured confounding given X)
Treatment assignment depends only on observed features. **True by
construction** in the simulator (`assign_historical_intervention` in
`data/generate_data.py` reads only observed merchant/customer/decline_code
fields), but this is the assumption real Razorpay data would need
independent justification for — historical human agents may have used
signals (a phone call, a support ticket's tone) that never made it into
the logged feature set.

## What this means in practice

Given assumption 2 is measurably weak, we do **not** call the T-learner
"a genuine causal estimator" unconditionally anywhere in this codebase —
see `ml-service/uplift.py`'s docstring for the corrected framing. It is
a T-learner conditional-outcome-difference estimate, causal under the
stated assumptions, with the region-by-region reliability of assumption
2 reported by `overlap_diagnostics.py` rather than assumed uniform.

## What we did NOT build (stated honestly)

A cross-fitted doubly-robust (AIPW) estimator, which would be more
robust to violations of assumption 2 by combining the outcome model with
an explicit propensity correction, was **not built** in this round due
to time constraints — this is a real gap, not a hidden one. The correct
next step, if pursued: fit propensity and outcome nuisance models with
K-fold cross-fitting, then compute `tau_AIPW(X) = mu_1(X) - mu_0(X) +
(A/e(X))(Y-mu_1(X)) - ((1-A)/(1-e(X)))(Y-mu_0(X))`, and compare its
implied policy value against the T-learner's on the same test split.
