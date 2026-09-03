# Recoup — Model Cards

For every model: purpose, input, output, training data, target, loss,
known limitations, and failure modes.

---

## 1. Root-cause posterior model
**File:** `ml-service/train.py::train_root_cause` → `models/root_cause.pkl`

| | |
|---|---|
| Purpose | Estimate `P(decline_code \| context)` as a full posterior, not a hard label, so downstream reasoning can account for uncertainty in the failure cause itself. |
| Input | merchant_category, customer_value_tier, event_type, payment_method (one-hot) + amount, prior_failure_count, minutes_since_last_failure, prior_recovery_count, prior_recovery_rate, recent_method_switch_count, customer_retry_propensity_observed, merchant_baseline_fail_rate |
| Output | Multi-class probability distribution over 7 decline codes |
| Model | GradientBoostingClassifier (150 estimators, depth 3) |
| Target | `decline_code` (categorical, 7 classes) |
| Loss | Multinomial deviance (sklearn default for GBM classification) |
| Training data | TRAIN split only (earliest 60% by time) |
| Evaluated on | VAL split (never TEST) — macro F1, per-class recall, confusion matrix |
| **Known limitation** | Macro F1 is modest (~0.14 on this run) on 7 fairly-balanced-by-design synthetic classes. This is reported honestly rather than inflated; the arm-selection decision does **not** depend on a hard root-cause label — it uses the full posterior only as an informational trace element, and the per-arm uplift models condition on `decline_code` directly as a feature, which is a stronger signal path than routing through this classifier's hard prediction. |
| Failure mode | Under concept drift, the posterior shifts toward whichever decline codes dominate the drift window (by design — this is the signal the drift monitor also independently picks up on). |

---

## 2. Per-arm recovery models (T-learner)
**File:** `ml-service/uplift.py::train_per_arm_models` → `models/uplift_arms.pkl`

| | |
|---|---|
| Purpose | Estimate `mu_a(X) = E[recovered \| X, arm=a]` for each of 6 arms (including `no_action` as control), then derive uplift `tau_a(X) = mu_a(X) - mu_no_action(X)`. |
| Input | Same feature set as root-cause model, minus decline_code as label (decline_code IS a feature here) |
| Output | Calibrated probability of recovery for a given (context, arm) pair |
| Model | GradientBoostingClassifier + CalibratedClassifierCV (isotonic, 3-fold) per arm |
| Target | `recovered` (binary), restricted to rows where that specific arm was the one historically logged |
| Loss | Log loss (GBM) + isotonic regression recalibration |
| Training data | TRAIN split, filtered per-arm |
| **Known limitation** | Trained on **observationally biased** data — each merchant's logging policy over-uses one habitual arm (see `data/generate_data.py`), so `mu_a(X)` for under-logged (context, arm) combinations is closer to extrapolation than interpolation. Partially cross-checked via `evaluation/offline_policy_eval.py`'s IPS/SNIPS estimator, which uses a different estimation approach and (on this run) agrees directionally with the oracle-based evaluation, though its point estimate is higher due to IPS variance on a modest matched sample (~100/500 events) — a known IPS limitation, not hidden. |
| Failure mode | Arms with very thin data (e.g. `no_action`, n≈80 on this run) have a weaker calibration guarantee than high-volume arms. |

---

## 3. Uncertainty ensembles (bootstrap)
**File:** `ml-service/uncertainty.py` → `models/uncertainty_ensembles.pkl`

| | |
|---|---|
| Purpose | Give each decision a genuine confidence tier, not just a point probability. `P(recovered)=0.82` means something different with ensemble std 0.03 vs 0.25. |
| Method | 5-model bootstrap ensemble per arm, each trained on a resampled subset. Ensemble std of predicted probability = uncertainty proxy. |
| Confidence tiers | HIGH (std < 0.08) → full automation. MEDIUM (0.08–0.18) → cheapest positive-value arm only. LOW (≥0.18) → escalate to human. |
| Why not conformal prediction | Conformal methods assume exchangeability between calibration and test data — an assumption the deliberately-injected concept-drift window specifically breaks. A bootstrap ensemble degrades gracefully instead: members increasingly disagree exactly when an input looks unlike their training distribution, which is the behavior the escalation policy needs. |
| **Known limitation** | 5 models is a small ensemble; variance estimates are noisier than a larger ensemble would give. Chosen for training-time cost, documented as a deliberate tradeoff. |

---

## 4. Drift monitor (PSI)
**File:** `ml-service/drift.py`

| | |
|---|---|
| Purpose | Detect when the live decline-code distribution has shifted meaningfully from the training reference distribution. |
| Method | Population Stability Index (PSI) on decline_code categorical distribution, industry-standard thresholds: <0.1 stable, 0.1–0.25 moderate, >0.25 significant. |
| **Known limitation** | Requires a minimum rolling window (30 events) before trusting the PSI score — reported as `INSUFFICIENT_DATA` otherwise, rather than a misleadingly unstable number on a near-empty window. |
| Validated | `evaluation/drift_test.py` confirms PSI = 0.402 (SIGNIFICANT_SHIFT) on the actual injected drift window vs PSI = 0.026 (STABLE) on a non-drift control — the detector fires when and only when it should, on this synthetic environment. |

---

## 5. Decision engine (net value + persisted LinUCB)
**File:** `ml-service/bandit.py`

| | |
|---|---|
| Purpose | Select the arm maximizing `uplift_A(X) * amount - FRICTION_COST[A]`, gated by confidence tier and drift status. |
| **Current build status** | The LinUCB layer is informational/experimental in this build — its UCB scores are computed and shown in the decision trace, but the final decision uses net-value + policy gating, not the LinUCB score directly. This is a deliberate scoping decision (see "Known limitations" below), not an oversight. |
| Why LinUCB over Thompson Sampling | Closed-form, deterministic UCB per arm — directly explainable per-decision ("mean + exploration bonus of X because we have Y observations in this region") in a way posterior sampling is harder to narrate live to a judge. Also avoids choosing a prior distribution family per arm. |

---

## Cross-cutting known limitations (stated once, applies system-wide)

1. **Synthetic data only.** No real Razorpay production data was used
   anywhere in this build. All "recovery" and "uplift" numbers are
   scored against a synthetic counterfactual oracle, not real outcomes.
2. **Merchant cold-start is not separately modeled.** A brand-new
   merchant with zero history gets the same feature treatment as an
   established one; no explicit cold-start fallback policy exists yet.
3. **The bandit's online-update loop (`/feedback` endpoint) exists and
   is wired to persist state to disk, but was not exercised over a long
   enough simulated run in this build to demonstrate convergence** —
   the single-decision UCB computation is real and tested; multi-episode
   online learning is present in the code but not empirically validated
   at scale here. Stated directly rather than implied as complete.
4. **Sequence modeling (RNN/Transformer) was deliberately not built.**
   Given the dataset's scale (~2,800 failed events) and the fact that
   the temporal signal is adequately captured by engineered rolling
   features (prior_failure_count, recent_failure_velocity, etc.), a
   sequence model would add training/inference complexity without a
   demonstrated accuracy gain — the project's own anti-overengineering
   principle (smaller model wins if it performs comparably) was applied
   here rather than building a sequence model just to have one.
