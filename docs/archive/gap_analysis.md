# Recoup — Gap Analysis & Exact Changes

From a working hackathon-grade prototype (single classifier per arm, no
causal reasoning, no offline evaluation, no drift handling) to a
research-grade decision system. Format: FILE → PROBLEM → CHANGE → WHY →
EXPECTED RESULT.

---

### `data/generate_data.py`
**Problem:** Original generator produced independent rows with random
treatment assignment and no counterfactual ground truth — supervised
`P(recovered|X,A)` learning on that data is unbiased by construction,
which is the opposite of a realistic production environment and gives
uplift modeling nothing to correct for.
**Change:** Rewrote with (1) a non-random logging policy
(`assign_historical_intervention`) biased per-merchant toward a habitual
default arm, (2) a full counterfactual oracle written to a separate file
(`oracle_potential_outcomes.csv`) covering every arm per event, (3)
incremental, strictly-prior temporal features computed via a per-customer
history dict updated only after each row is written.
**Why:** Selection bias in logged data is *the* reason a T-learner (or
any causal estimator) is defensible rather than decorative — without it,
a plain classifier would already be an unbiased estimate of treatment
effect and the causal layer would be theater.
**Expected result:** `ml-service/test_no_leakage.py` passes all 4 checks;
`evaluation/offline_policy_eval.py` has genuine propensity variation to
estimate against.

---

### `ml-service/features.py`
**Problem:** No explicit, enforced list of forbidden (leaky) columns.
**Change:** Added `FORBIDDEN_COLS` set and an assertion in
`build_feature_frame` that fails loudly if a leaky column is ever added
to the feature spec.
**Why:** Makes leakage prevention a runtime-checked invariant, not a
comment that can silently go stale as the code evolves.
**Expected result:** Any future PR that accidentally adds `recovered` or
`chosen_intervention` to `CATEGORICAL_COLS`/`NUMERIC_COLS` fails
immediately at import time, not at evaluation time.

---

### `ml-service/temporal_split.py` *(new file)*
**Problem:** No file previously enforced a temporal train/val/test split
— a random split on sequential data leaks population-level drift signal
backward in time.
**Change:** New module: sort by timestamp, split 60/20/20 by time, not
row index.
**Why:** The concept-drift window needs to fall mostly in val/test for
the drift experiments to mean anything; a random split would smear it
evenly across all three and hide the effect entirely.
**Expected result:** `evaluation/drift_test.py` shows a genuinely
elevated PSI in the drift window vs a stable control.

---

### `ml-service/uplift.py` *(new file, replaces a single shared classifier)*
**Problem:** Original system predicted `P(recovered|X,A)` with one model
per arm but never asked "would this have recovered anyway without the
action?" — the exact conflation a judge will call out immediately.
**Change:** T-learner: one calibrated model per arm including a
`no_action` control; `compute_uplift` derives `tau_A(X) = mu_A(X) -
mu_no_action(X)`.
**Why:** Decision-relevant quantity is the *incremental* effect of
acting, not raw recovery probability — a customer who was always going
to pay shouldn't be credited to whichever arm happened to be logged.
**Expected result:** `expected_net_value` in `/decide` responses is
computed from uplift, not raw probability — visible directly in the API
response and in `evaluation/ablation.py`'s Model E vs Model B comparison.

---

### `ml-service/uncertainty.py` *(new file)*
**Problem:** A recovery probability of 0.82 was treated as equally
trustworthy everywhere, regardless of how much data supported it.
**Change:** 5-model bootstrap ensemble per arm; ensemble std → 3-tier
confidence (HIGH/MEDIUM/LOW) feeding the policy gate.
**Why:** Answers "what happens when the model is wrong?" with a
mechanism, not a promise — thin-data contexts get routed to conservative
action or human escalation automatically.
**Expected result:** Failure Lab scenario 5 (OOD customer) and scenario
8 (thin history) both demonstrate the tier downgrading action
automaticity.

---

### `ml-service/drift.py` *(new file)*
**Problem:** No drift detection existed; the injected drift window in
the data had no monitoring counterpart.
**Change:** PSI-based drift detector on decline_code distribution,
industry-standard thresholds, wired into `/decide`'s live rolling
window and into the policy gate.
**Why:** "What happens under distribution shift?" needs a real,
citable answer, not "the model would probably still work."
**Expected result:** `evaluation/drift_test.py` shows PSI = 0.402
(SIGNIFICANT_SHIFT) on the drift window vs 0.026 (STABLE) on control.

---

### `ml-service/bandit.py`
**Problem:** LinUCB existed but was disconnected from the actual
decision path — it computed scores nobody used.
**Change:** Decision path now explicitly: `net_value()` (uplift-based,
cost-aware) → `select_action()` (confidence + drift gated) → policy
check. LinUCB is persisted (`models/linucb_state.pkl`), scored, and
returned in the trace as an informational/experimental signal, with the
scoping decision to not let it drive the final action stated directly in
`docs/model_cards.md` rather than silently shipped as if it were doing
more than it is.
**Why:** Better to be honest about what's driving the decision than to
claim a bandit-driven system where the bandit isn't actually load-bearing
yet.
**Expected result:** Judge asking "does the bandit actually decide
anything?" gets a true answer either way, and the trace shows exactly
which signal did.

---

### `evaluation/evaluate.py`, `evaluation/ablation.py`, `evaluation/offline_policy_eval.py`, `evaluation/drift_test.py` *(new files)*
**Problem:** Original evaluation scored strategies against the single
logged (biased) outcome — literally the observational-bias problem the
uplift model exists to correct, reintroduced at evaluation time.
**Change:** All evaluation now scores against the counterfactual oracle
(ground truth potential outcomes) instead, PLUS an independent IPS/SNIPS
estimator on the logged data as a cross-check that doesn't require the
oracle at all.
**Why:** "How did you evaluate a policy without deploying it?" needs an
answer that works even when you don't control a magic oracle — IPS/SNIPS
is that answer; the oracle is used here only to sanity-check the
estimator, exactly as a real team without ground truth couldn't.
**Expected result:** On this run, oracle-scored Recoup beats baseline by
+26.8% and ML-only by +7.4%; IPS/SNIPS agrees directionally but with a
higher point estimate — a documented, expected consequence of IPS
variance on a modest matched sample, not a discrepancy to hide.

---

### `ml-service/test_no_leakage.py` *(new file)*
**Problem:** No executable proof of the leakage claims — only prose.
**Change:** Four assertions any reviewer can run themselves.
**Why:** "How do you prevent temporal leakage?" is much stronger
answered with `python3 test_no_leakage.py` → 4/4 PASS than with a
paragraph.
**Expected result:** Reproducible in front of a judge in under 5
seconds.

---

### `failure-lab/scenarios.py`
**Problem:** Original 7 scenarios exercised only the policy engine's
hard caps, not the new uncertainty/drift/uplift layers.
**Change:** Expanded to 12 scenarios covering duplicate events, missing
features, unseen merchant categories, drift-like contexts, OOD
customers, conflicting signals, high-value thresholds, thin-history low
confidence, policy-restricted actions, retry-cap exhaustion, nudge-cap
exhaustion, and a documented (not directly exercised) API-failure case.
**Why:** Matches Part 22's explicit scenario list; each new layer
(uncertainty, drift) needs its own stress test, not just the original
policy caps.
**Expected result:** All 12 run without crashing; each produces a
`decision_reason` that correctly explains why the system acted, deferred,
or escalated.

---

### `frontend/index.html`
**Problem:** UI showed only the final decision and raw per-arm
probabilities — none of the new reasoning layers were visible.
**Change:** Decision panel now renders root-cause posterior, uplift
table (probability + uplift + net value per arm side by side), and
confidence/drift badges.
**Why:** Part 21 requires the decision trace be visible in the UI, not
just returned by the API.
**Expected result:** A judge can watch a live decision and see every
intermediate quantity that produced it.

---

### `docs/leakage.md`, `docs/model_cards.md` *(new files)*
**Problem:** No standalone documentation a judge could read without
reverse-engineering the code.
**Change:** Explicit leakage rules + executable tests; per-model cards
with purpose/input/output/target/loss/limitations/failure modes for all
5 trained components.
**Why:** Parts 18 and 25's explicit deliverables — and the fastest way
to survive 10–15 minutes of interrogation is to have already written
down the honest limitations before being asked.

---

## Deliberately NOT built (Part 23 compliance)

- **Sequence/Transformer model** — engineered temporal aggregates were
  not shown to underperform on this dataset scale; building a sequence
  model without a demonstrated gain would be complexity for its own
  sake. Documented as a comparison not run rather than silently skipped.
- **X-learner / causal forest / doubly-robust estimator** — T-learner is
  the smallest model that is still a genuine causal estimator here;
  documented reasoning in `ml-service/uplift.py`'s docstring.
- **Conformal prediction** — breaks under the deliberately-injected
  drift's violated exchangeability assumption; bootstrap ensemble
  degrades more gracefully for this use case.
- **Reinforcement learning** — the problem is single-step (act once per
  failed payment, observe once), not a multi-step MDP; a contextual
  bandit is the correctly-scoped tool, not full RL.
- **LLM as the money decision** — used only for explanation-text
  generation, never for choosing an arm; stated explicitly, repeatedly,
  because it is the single most common thing judges assume by default.
