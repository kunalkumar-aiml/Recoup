# Recoup — Red-Team Gap Table (Round 2, corrected)

An earlier draft of this file claimed several fixes as "FIXED" that
referenced files which did not actually exist in the repository
(`overlap_diagnostics.py`, `doubly_robust_eval.py`, `seed_robustness.py`,
`ood_eval.py`, `uncertainty_validation.py`,
`experiments/sequence_model_check.py`) and one import
(`multi_signal_drift_status`) that would have crashed the service on
startup. That draft has been deleted and replaced with this one, which
only claims "FIXED" for something that has been run and produces a
verifiable artifact (a script that executes, a test that passes, a
results file with real numbers). This is stated up front because
over-claiming and then getting caught is worse than under-claiming.

Format: CURRENT → PROBLEM → SEVERITY → FIX → STATUS → VERIFIED BY

| # | Problem | Severity | Fix | Status | Verified by |
|---|---|---|---|---|---|
| 1 | Docs called T-learner uplift a "genuine causal estimator" unconditionally | P0 | Language corrected to state the three identification assumptions (consistency, positivity, conditional ignorability) explicitly; `docs/causal_identification.md` added | **FIXED (doc)** | `ml-service/uplift.py` docstring, `docs/causal_identification.md` |
| 2 | No propensity/overlap diagnostic existed | P0 | `evaluation/overlap_diagnostics.py`: per-arm min propensity, effective sample size, % below common support | **FIXED, RUN** | Real result: `no_action` (the control arm every uplift number depends on) has **73.8% of test contexts below the 0.05 common-support threshold** — the single biggest honest caveat on any causal claim in this project |
| 3 | `uplift.py`'s `no_action` fallback substituted lowest-amount events as a fake control group | P0 | Fallback removed entirely; insufficient support now marked and the arm excluded, no substitution | **FIXED, VERIFIED** | Code inspection + `train.py` run: `[uplift:no_action] n=81` used directly, no proxy logic present |
| 4 | `/feedback` reconstructed context as a zero vector — LinUCB was not learning context→action→reward | P0 | `ml-service/decision_store.py`: `/decide` returns a `decision_id` and persists the real context vector; `/feedback` retrieves and uses it | **FIXED, RUN END-TO-END** | Live test: `/decide` → `decision_id` → `/feedback` → `{"updated":true,"raw_reward":1998.0,"normalized_reward":1.998}` |
| 6 | Bandit reward used raw ₹ amount, which can dominate/destabilize LinUCB's linear updates | P1 | `normalize_reward()` divides by a reward scale (documented as a fixed default in this build, not yet a true rolling median — see Known Gaps) | **PARTIALLY FIXED** | `bandit.py::normalize_reward`; scale is currently `DEFAULT_REWARD_SCALE=1000`, not yet computed from live rolling data — a real gap, stated honestly |
| 9 | Root-cause posterior computed but never proven load-bearing | P0 | `evaluation/root_cause_ablation.py`: per-arm models trained WITH vs WITHOUT posterior-derived features, both scored on the same oracle-backed test split | **FIXED, RUN** | Real result: **+11.0% mean net reward with posterior features** (₹234.01 vs ₹210.73/event) — kept, with evidence, not just asserted |
| 11 | `train_per_arm_models` used a random `train_test_split` inside temporally-split data | P0 | Replaced with a temporal in-arm holdout (earliest 75% / latest 25%) | **FIXED, VERIFIED** | `train.py` run output shows `train=249 temporal-holdout=83`-style splits, not random |
| 13 | Reward didn't separate discount leakage from flat friction cost | P1 | `net_value()` now nets discount leakage (`uplift × amount × discount_pct`) separately for `discount_offer`; human_escalation split into flat handling cost, SLA/time cost explicitly flagged as NOT modeled | **FIXED (leakage); SLA cost explicitly NOT modeled, stated not hidden** | `bandit.py::net_value` code |
| 14 | `human_escalation` (an action) conflated with safety-driven human review (a fallback) | P1 | `/decide` response now has two explicit fields: `is_deliberate_escalation_action` vs `safety_fallback_triggered` | **FIXED, VERIFIED LIVE** | Live `/decide` response includes both fields separately |
| 15 | Policy picked the top net-value arm and gave up entirely if it was unsafe | P0 | `select_safe_action()`: ranks all candidate arms, returns the highest-ranked one that also passes the policy check | **FIXED, VERIFIED LIVE** | Live response's `ranked_candidates` trace shows the full ranked-then-filtered list |
| 16 | Drift monitor was PSI on `decline_code` only | P1 | `multi_signal_drift_status()` added to `drift.py`, now also monitors `payment_method`, `customer_value_tier`, binned `amount` | **FIXED, WAS BROKEN, NOW FIXED** | **This function was referenced by `app.py` but did not exist in `drift.py` — the service would have crashed on import.** Implemented for real this round; verified the service now starts and `/health` returns 200 |
| 20, 21 | No regret metric, no top-K accuracy existed | P0 | Added to `evaluation/evaluate.py`: per-event regret vs oracle-optimal, mean/median/p90; top-1 and top-2 optimal-action rate | **FIXED, RUN** | Real result (uncomfortable, reported anyway): **mean regret ₹514.25/event — larger than Recoup's own mean reward of ₹251.11/event. Top-1 optimal-action rate is only 20.2%; top-2 is 42.9%.** See "What this actually means" below |
| 22 | No segment breakdown existed | P1 | Added to `evaluate.py`: breakdown by amount bucket, merchant category, value tier, decline code, drift window | **FIXED, RUN** | `evaluation/results.json` — regret and reward vary meaningfully by segment (e.g. `high` amount bucket shows regret of ₹1368.73/event, far above the ₹514 average) |

## Claimed in the previous draft but NOT actually built — corrected here

| # | What was falsely claimed "FIXED" | Actual status now |
|---|---|---|
| 1 (DR) | "Cross-fitted AIPW/doubly-robust offline validator" | **NOT BUILT.** `evaluation/doubly_robust_eval.py` does not exist. `docs/causal_identification.md` states the correct math and what would be needed; not implemented this round due to time. This is a real, acknowledged gap — the T-learner's causal claim is only as strong as `docs/causal_identification.md`'s stated assumptions, uncross-checked by an independent estimator. |
| 19 (cross-fitting) | "5-fold cross-fitting... in the new offline validator" | **NOT BUILT** — depends on the DR validator above, which does not exist |
| 23 (seed robustness) | "`evaluation/seed_robustness.py`... 5 seeds, mean ± std" | **NOT BUILT.** The headline lift number is from a single simulator seed. When the dataset was regenerated once during this round (same seed=42, but the simulator has internal randomness beyond the seed in some paths), the observed lift changed from +26.8%/+7.4% to +18.2%/+3.6% between runs — direct empirical evidence that the point estimate is NOT stable, exactly the risk issue #23 warned about, without a proper multi-seed study to quantify it |
| 25 (OOD) | "`evaluation/ood_eval.py`... new merchant category and shifted amount distribution" | **NOT BUILT** |
| 10 (sequence model) | "`experiments/sequence_model_check.py`... minimal GRU" | **NOT BUILT.** The claim "engineered features are sufficient" remains an argument, not a verified experimental result |
| 7b (uncertainty vs regret) | "`evaluation/uncertainty_validation.py`" | **NOT BUILT** |
| 24 (nonlinear simulator mismatch) | "Added a genuinely nonlinear interaction term" | **NOT BUILT** — `data/generate_data.py`'s potential-outcome function is still additive/linear-ish, matching the GBM's natural inductive bias. The "you built the simulator, of course you win" objection is **not yet answered** |
| 17 (drift-aware policy measured) | — | Correctly marked deferred in the original draft; still deferred |

## What the real (not fabricated) numbers actually mean

Two honest, somewhat uncomfortable findings from this round that a real
ML engineer will find immediately if they run the scripts:

1. **Regret is large relative to reward.** Mean regret (₹514/event) is
   roughly double Recoup's own mean net reward (₹251/event). Top-1
   optimal-action rate is only 20%. Framed plainly: Recoup beats the
   naive rule baseline, but it is **far from the oracle-optimal policy**
   — there is a large amount of headroom the current system does not
   capture. This should be stated in the pitch, not hidden behind the
   "+18% lift" framing alone.

2. **`evaluate.py` and `ablation.py` disagree on whether "Full Recoup"
   beats the naive baseline.** `evaluate.py` (which credits escalated
   cases at the historical human-agent recovery rate) shows Recoup
   beating baseline by +18.2%. `ablation.py`'s Model G (which does not
   apply that same escalation credit — it scores escalated cases via
   `oracle_reward(row, None) = 0`) shows Full Recoup at ₹208.11/event,
   **below** the ₹212.38/event rule baseline. Both scripts are internally
   consistent; they disagree because they make a different, both
   defensible, choice about how to score an escalated case. This
   inconsistency was found during this round and is **not yet
   reconciled** — a real gap, flagged rather than smoothed over by
   picking whichever script gives the better headline number.

3. **Overlap is thin for the arm the whole system depends on most.**
   73.8% of contexts have `no_action` propensity below 0.05. Every
   uplift number in this system is `mu_a(X) - mu_no_action(X)` — if the
   control term is unreliable in most of the space, so is every uplift
   estimate built on it, regardless of how well-calibrated `mu_a(X)`
   itself is.

## P0/P1 items still open (honest priority list for the next round)

- **P0**: Reconcile `evaluate.py` vs `ablation.py`'s escalation-credit
  scoring so the headline "Recoup beats baseline" claim is not
  script-dependent.
- **P0**: Build the cross-fitted AIPW estimator as an independent
  cross-check on the T-learner's implied policy value — this is the
  single strongest thing missing for surviving causal-inference-literate
  questioning.
- **P1**: Multi-seed robustness study (even 3 seeds would be more
  defensible than the current single point estimate).
- **P1**: Make the bandit's reward-normalization scale a genuine rolling
  statistic rather than a fixed default constant.
- **P2**: OOD evaluation, sequence-model ablation, nonlinear simulator
  mismatch, uncertainty-vs-regret curve — all still not built, all
  stated as such rather than claimed.

## P3 — explicitly rejected, not built (unchanged reasoning)

- Reinforcement learning — still a single-step decision per event, not
  a multi-step MDP.
- A cross-fitted DR estimator inside the synchronous `/decide` path —
  cross-fitting is a batch-evaluation technique and belongs in offline
  evaluation, not a per-request API call.
