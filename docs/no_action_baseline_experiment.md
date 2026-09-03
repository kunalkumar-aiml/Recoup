# No-Action Baseline Experiment

## The hypothesis being tested

`docs/final_action_ranking_analysis.md` found 72.3% of regret is associated with
uplift-estimation error, and hypothesized this points at
`mu_no_action(X)` specifically — the control-arm baseline every uplift
number is measured against — which is fit with the same
`GradientBoostingClassifier(n_estimators=100, max_depth=3)` as every
other arm despite having only ~80 training examples, a plausible
overfitting risk on the smallest-sample arm.

**Candidate fix tested**: replace the no_action model with a much
simpler, more regularized `LogisticRegression(penalty="l2", C=0.5)` —
far fewer effective parameters, better suited to ~60-80 training rows
than a 100-tree ensemble.

## Result: DID WE FIND THE REAL SOURCE OF WRONG-ARM REGRET?

**NO** (for this specific candidate fix).

## Experiment D — oracle diagnostic (never used for training)

| Model | MAE vs true oracle P(no_action) | Bias |
|---|---|---|
| Current (GBM depth=3) | 0.1084 | +0.09 |
| Improved (Logistic L2) | 0.2928 | **+0.25** |

**The "improved" model fit the ground truth 170% worse**, not better —
a large positive bias (systematically overestimates no_action's
recovery probability by 0.25 on average). The regularization hypothesis
was wrong in the direction tested: logistic regression's linear
decision boundary apparently fits this particular ~80-row, likely
non-linear relationship worse than the GBM does, despite having fewer
parameters. Fewer parameters is not automatically better-suited to
small data when the underlying relationship isn't well-approximated by
a linear model.

## Experiment E/F — full-pipeline policy evaluation

| | Current | Improved | Delta |
|---|---|---|---|
| Mean net reward | ₹256.12 | ₹269.59 | +5.3% |
| Mean regret | ₹510.79 | ₹497.32 | -2.6% |
| Top-1 accuracy | 20.2% | 14.0% | **-6.2pp** |
| Top-3 accuracy | 66.8% | 43.6% | **-23.2pp** |

**A naive read of just the reward/regret numbers would suggest
adoption** (+5.3% reward). This is exactly the trap the experiment's
own decision rule exists to catch: **Top-1 and Top-3 ranking accuracy
both collapsed** at the same time. A model that fits the ground truth
worse (Experiment D) and ranks actions worse (Top-3 down 23 points)
should not be adopted on the strength of a single-seed reward number
that went up — that combination is a strong signal of noise from a
differently-biased model landing on different decisions on this
specific 500-event sample, not genuine improvement.

**Decision: REJECTED. Keep the current T-learner unchanged.**

## The honest methodological lesson

This negative result is itself valuable: **checking only aggregate
reward/regret would have led to adopting a strictly worse model.**
Experiment D's oracle diagnostic (comparing against true probabilities,
never used in training) and the ranking-accuracy metrics were what
caught this — a reminder of exactly why this project's evaluation
protocol insists on multiple, cross-checking metrics rather than a
single headline number, and why the earlier forensic analysis was
careful to call its own 72.3% finding a "diagnostic heuristic," not
causal proof.

## What this does and doesn't tell us

- It does **not** refute the underlying hypothesis that
  `mu_no_action(X)` is under-supported and contributes to uplift error
  — the oracle diagnostic's MAE=0.108 for even the current (best-tested)
  model is still a substantial absolute error, and the overlap problem
  (73.8% of contexts below common support for `no_action`) is unchanged
  and still measured.
- It **does** rule out "simply use a smaller/more regularized model" as
  the fix — that specific, cheap candidate was tested honestly and
  failed.
- The correctly-targeted fix remains what rounds 4-7 and the prior
  forensic analysis already identified: a **cross-fitted doubly-robust
  estimator**, which corrects outcome-model bias via propensity
  weighting rather than by changing the outcome model's functional
  form — a structurally different approach than what was tested here,
  and still not built given the September 4 deadline.

## Experiment C (cross-fitted DR)

**NOT IMPLEMENTED** — out of scope given the deadline, per this
experiment's own explicit escape clause ("only if practical within the
deadline"). Stated plainly rather than attempted partially and
mis-scoped under time pressure.

## Multi-seed

**NOT RE-RUN** for this experiment — single-seed result only,
consistent with the project's already-documented 3-seed (not 10-seed)
scope limitation. A result this close to noise (as this experiment
itself demonstrates) would benefit from multi-seed confirmation before
any future candidate fix is adopted — noted as a requirement for next
time, not assumed away.

## Final recommendation for the pitch

Say: **"We tested our own leading hypothesis for the dominant regret
source, found the specific cheap fix we tried didn't hold up under
closer inspection, and correctly rejected it rather than shipping a
misleading reward number — this is the same rigor we applied throughout
the evaluation, and it's why the current system, not the untested
alternative, is what's in the submitted repository."**

Do not say the no_action baseline problem is "fixed." It is diagnosed,
one candidate fix was honestly tested and rejected, and the
correctly-targeted remaining fix (cross-fitted DR) is still open work.
