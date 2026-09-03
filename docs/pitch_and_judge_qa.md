# Recoup — 5-Minute Pitch & Judge Q&A

## Positioning (say this early, verbatim or close to it)

"Recoup is a constrained AI decision engine that estimates the
incremental economic value of recovery interventions under uncertainty,
selects the safest high-value action, evaluates its counterfactual
impact offline, and continuously adapts from observed outcomes."

## 5-Minute pitch

**0:00–0:20 — Problem.** "Every competing project here will predict
whether a payment fails. That's the easy 20%. We built the hard 80%:
given a failure already happened, which of five interventions has the
highest *causal* recovered value, net of its own cost — and does the
model even trust itself enough to act alone?"

**0:20–0:50 — Why naive ML fails here.** Our own logged data is
observationally biased — merchants habitually over-use one intervention.
A plain classifier trained on that data conflates "this action worked"
with "this customer was always going to pay." Show the
`docs/leakage.md` observational-bias paragraph on screen.

**0:50–1:20 — Our insight.** T-learner uplift: `tau_A(X) = mu_A(X) -
mu_no_action(X)`. Net value = uplift × amount − friction cost. This is
the quantity that should drive the decision, not raw recovery
probability.

**1:20–2:30 — Live demo.** Submit a failed-payment event on the
Decision Screen. Walk through the full trace live: root-cause posterior
→ per-arm recovery probabilities → uplift table → confidence tier →
drift status → chosen action → policy gate result.

**2:30–3:20 — Failure Lab.** Run 2–3 of the 12 scenarios live: an OOD
customer triggers LOW confidence and escalates; an oversized discount
request gets rejected by the policy engine regardless of what the model
wants.

**3:20–4:00 — Evaluation, honestly.** Show `evaluation/results.json`
(oracle-scored: +26.8% over baseline, +7.4% over ML-only on this run)
AND `evaluation/offline_policy_eval.py`'s IPS/SNIPS estimate side by
side — explain *why* they don't match exactly (IPS variance on a
modest matched sample) rather than only showing the flattering number.

**4:00–4:30 — Ablation.** One slide: Model A (rule) → G (full Recoup),
mean net reward per layer. Prove complexity earns its keep, or show
honestly where it doesn't (e.g. this run's Model D/G policy-gating step
trades some raw reward for safety — that's a deliberate trade, not a
regression).

**4:30–5:00 — Why Razorpay.** "This is the same shape as reconciliation,
chargebacks, settlement forecasting: detect, estimate causal value under
cost, decide safely, prove it offline before ever deploying. We built
the smallest complete version of that loop, and we can tell you exactly
what we didn't build and why."

---

## 30 Judge Questions — Prepared Answers

**1. Why not just use XGBoost?**
We do — GradientBoostingClassifier is our base learner throughout. The
depth is in the *decision layer* (uplift, uncertainty, drift, policy)
built on top, not in swapping algorithms.

**2. Where is the causal component?**
`ml-service/uplift.py` — T-learner, per-arm calibrated models, uplift =
treatment-arm probability minus no-action control probability. Chosen
over S-learner (dilutes treatment signal), X-learner/causal forest
(need more data/better overlap than we have), doubly-robust (harder to
sanity-check live).

**3. How do you know your intervention caused recovery?**
We don't claim certainty — we claim an unbiased *estimate* under the
T-learner's assumptions, cross-checked two ways: against the synthetic
oracle (ground truth in this environment) and via IPS/SNIPS on the
logged data (works even without an oracle, as a real deployment would
need).

**4. How did you evaluate an unseen policy offline?**
Inverse Propensity Scoring / Self-Normalized IPS
(`evaluation/offline_policy_eval.py`), using a propensity model fit on
the logged (biased) assignment data — the standard approach when you
don't have the logging system's true propensities.

**5. How do you prevent temporal leakage?**
Three independent mechanisms: generation-time incremental history
(features literally cannot see the future because they're written
before it exists), a temporal (not random) train/val/test split, and 4
automated tests in `test_no_leakage.py` — run them live if asked.

**6. What happens under distribution shift?**
PSI-based drift monitor. Verified: PSI = 0.402 (SIGNIFICANT_SHIFT) on
the actual injected drift window vs 0.026 (STABLE) on a control window.
On significant drift, the policy gate forces escalation regardless of
model confidence.

**7. Why contextual bandits?**
For the eventual online-adaptation loop (`/feedback` endpoint,
persisted LinUCB state) — the arm-selection problem is a genuine
single-step contextual bandit (act once, observe once), not a
multi-step MDP.

**8. Why not reinforcement learning?**
Because there's no multi-step state transition to exploit — each
failed-payment event is a single decision with a single reward. Full
RL would be solving a harder problem than the one we have, which is
exactly the kind of complexity-for-its-own-sake this project explicitly
avoided.

**9. Why not an LLM?**
LLM output isn't calibrated in the probabilistic sense the money
decision needs. Used only for explanation-text generation, never on the
decision path — stated explicitly, checked by not appearing anywhere in
`bandit.py` or `uplift.py`.

**10. What is your business metric?**
Net reward per event: recovered value (if the oracle/observed outcome
says that arm recovers) minus that arm's friction cost. Reported at the
population level (mean/total) and per-layer (ablation).

**11. What happens if your model is wrong?**
Three independent safety nets: calibration (so probabilities mean what
they say), uncertainty tiers (LOW confidence → mandatory escalation),
and hard policy caps that no model score can override (discount %,
retry count, daily nudges, human-approval amount threshold).

**12. Which component actually improved performance?**
`evaluation/ablation.py`, Model A through G, same held-out test split,
same oracle scoring. On this run: B (ML-only) beats A (rule) by
~18%; C (expected value) adds a further small gain; D (policy gating)
trades some raw reward for safety (expected — it's supposed to be
conservative); E (uplift) recovers most of that trade while keeping the
causal framing.

**13. Why T-learner and not X-learner/causal forest/doubly-robust?**
Documented in `ml-service/uplift.py`'s docstring: T-learner is the
smallest estimator that is still genuinely causal, appropriate for our
propensity overlap and data volume (~2,800 failed events). The more
complex estimators earn their keep with worse overlap or more data than
we have — using them here would be complexity without a demonstrated
benefit.

**14. How did you generate your data?**
Latent per-customer/per-merchant variables (price sensitivity, recovery
profile, retry propensity) drive both the true causal outcome AND a
biased historical logging policy — deliberately non-random treatment
assignment, because that's what makes the causal layer necessary rather
than decorative. Full detail: `data/generate_data.py` docstring.

**15. Is this actually novel?**
The individual techniques (GBM, isotonic calibration, T-learner, PSI,
IPS) are all standard. The novelty is the complete, honestly-evaluated
chain — most student teams stop at prediction; we went to causal
estimation, offline policy evaluation, and drift-gated safety, with
executable proof for each claim.

**16. What stops another team from copying this in a weekend?**
Not any single file — the discipline of oracle-separated counterfactual
evaluation, leakage tests that actually run, and an ablation study that
doesn't cherry-pick its own comparison. That combination, done honestly,
takes longer than a weekend to get right.

**17. What is your false-positive/false-negative cost?**
Encoded directly: friction cost per arm (`bandit.py::FRICTION_COST`)
and the human-approval amount threshold in `policy.py`. A wrong
"cheap" action costs little; a wrong "expensive" one is gated behind
human review before it can execute.

**18. Can the system be abused?**
Hard caps (discount %, retry count, daily nudges) are enforced in
`policy.py` independent of any model score — no prediction, however
confident, can override them.

**19. What happens if the ML service is unreachable?**
Backend (`server.js`) returns a clear 502, not a silent failure —
verified as Failure Lab scenario 12. A production version would fail
closed (escalate) rather than fail open.

**20. What is your latency?**
Not formally load-tested in this build — a genuine limitation, stated
directly rather than claiming a number we didn't measure.

**21. What happens at 10 million transactions?**
Not validated at that scale; the architecture (stateless FastAPI
service, per-arm models small enough to hold in memory) has no obvious
structural blocker, but this is an honest "untested," not a claim.

**22. What is your model's calibration?**
Reported directly: raw Brier 0.2524 → calibrated 0.2016; raw ECE 0.2041
→ calibrated 0.0604 on the highest-volume arm this run
(`ml-service/train.py`'s calibration report). Isotonic regression,
chosen for its non-parametric flexibility given class imbalance.

**23. Where is human approval required?**
LOW confidence tier, SIGNIFICANT drift, amount above the human-approval
threshold, or any policy-cap violation — all documented in
`policy.py`/`bandit.py` and demonstrated live in the Failure Lab.

**24. Why isotonic calibration and not Platt scaling?**
Isotonic is non-parametric and handles the non-sigmoid miscalibration
shape common under class imbalance better than Platt's fixed logistic
form, at the cost of needing slightly more calibration data — a
reasonable trade at our sample sizes.

**25. Why PSI over KL/Jensen-Shannon for drift?**
PSI has industry-standard, citable thresholds (credit-risk/fintech
monitoring convention) we can defend rather than invent; KL is
asymmetric and harder to threshold meaningfully; Jensen-Shannon lacks
an equivalent domain convention.

**26. Why bootstrap ensembles over conformal prediction for
uncertainty?**
Conformal prediction assumes exchangeability between calibration and
test data — an assumption our deliberately-injected drift window
specifically violates. A bootstrap ensemble degrades more gracefully:
members disagree more exactly when inputs look unlike training data.

**27. How do you handle a brand-new merchant with no history?**
Not separately modeled in this build — stated as a known limitation in
`docs/model_cards.md` rather than glossed over. The natural extension is
a hierarchical/backoff estimator that shrinks toward a category-level
prior for low-data merchants.

**28. Does the bandit actually drive the decision?**
Honestly: not yet. LinUCB scores are computed, persisted, and shown in
the trace, but the current decision uses uplift-based net value plus
policy gating. This scoping decision is documented directly in
`docs/model_cards.md` rather than overclaimed.

**29. What's the single biggest weakness in this system right now?**
Selection bias in the training data means per-arm models for
under-logged (context, arm) combinations are extrapolating, not
interpolating — partially checked via IPS/SNIPS, but not eliminated.

**30. If you had one more week, what would you build next?**
Merchant hierarchical backoff for cold-start, a real online-learning
loop exercised over many simulated episodes (the code exists; the
empirical convergence check doesn't yet), and load-testing latency at
scale.

## Doubly-Robust (DR) / Cross-Fitting Q&A (updated after multi-seed validation)

**31. Why T-learner as the production default, and not DR?**
Both were implemented and honestly compared. DR showed a real
single-seed improvement (+11.4% reward, -5.7% regret) and was initially
formally "ADOPT"-eligible by this project's own decision rule
(`docs/dr_cross_fitting_final.md`). A follow-up multi-seed validation
(`docs/final_policy_selection.md`) overturned that: across 3 seeds, DR
won on 2 but lost outright on the third (-3.8% reward, worse regret,
Top-3 down 8.6pp), with an aggregate reward-delta standard deviation
(8.2) nearly as large as its mean (7.6) — not distinguishable from
noise at this sample size. **DR is formally rejected as the primary
policy on the evidence itself, not merely deferred for deadline
caution.**

**32. What is cross-fitting, concretely, in this codebase?**
`ml-service/dr_estimator.py`: propensity (P(A=a|X)) and outcome
(E[Y|X,A=a]) models are fit on K-1 folds (K=3) and predict ONLY the
held-out fold, rotated across folds — so no observation's own outcome
ever influences the nuisance-model prediction used to score it.

**33. What is propensity here, and how is it estimated?**
P(A=a|X): a multinomial logistic regression trained on the logged
(observationally biased) treatment assignment data, reused from
`evaluation/overlap_diagnostics.py`.

**34. What is overlap, and what happens when it's poor?**
Overlap means every arm has a realistic (non-near-zero) probability of
being assigned for a given context. It's measurably poor for the
`no_action` control arm (73.8% of contexts below common support,
`evaluation/overlap_diagnostics.py`) — propensity clipping (floor 0.05)
prevents the AIPW correction term from exploding, but does not fix the
underlying thin support; DR's own no_action fit is actually *worse* in
isolation than the T-learner's (Section 7, `docs/dr_cross_fitting_final.md`)
— stated plainly, not hidden behind the aggregate win.

**35. What are DR's causal assumptions?**
Consistency, positivity/overlap, and conditional exchangeability — all
three stated explicitly in `ml-service/dr_estimator.py`'s docstring,
with overlap explicitly flagged as violated, not assumed.

**36. Why isn't a synthetic simulator proof of real-world causal validity?**
The simulator's logging-policy bias, outcome-generating mechanism, and
"ground truth" are all specified by the same team that built the
estimator being tested — the synthetic validation (Section 4,
`docs/dr_cross_fitting_final.md`) proves the *implementation* correctly
recovers a known effect under confounding, not that the same estimator
will perform identically on real, unknown-ground-truth data.

**37. How does DR change the actual economic decision, concretely?**
It doesn't, in the submitted build — DR exists as an offline evaluation
script (`evaluation/dr_cross_fitting.py`), not wired into
`ml-service/app.py`'s live decision path. The T-learner remains the
production estimator.

**38. When would you NOT trust this DR result?**
When treating a single-seed, single-run number as stable (exactly the
mistake round 3 already demonstrated is easy to make); when extrapolating
the no_action arm's DR estimate specifically, given its own oracle fit is
worse than the T-learner's there; and when treating "formally meets the
adoption decision rule" as equivalent to "safe to ship days before a
deadline" — these are different bars, and this project deliberately
did not conflate them.
