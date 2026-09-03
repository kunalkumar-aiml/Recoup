# Recoup — Architecture

```
RAW PAYMENT EVENTS (data/generate_data.py)
    |  observational bias in historical logging (not random) +
    |  counterfactual oracle written separately for eval-only use
    v
LEAKAGE-SAFE TEMPORAL FEATURES (computed incrementally at generation
    time from strictly-prior per-customer history; see docs/leakage.md)
    v
ROOT-CAUSE POSTERIOR              ml-service/train.py::train_root_cause
    P(decline_code | context), multi-class, not a hard label
    v
PER-ARM RECOVERY MODELS (T-learner)     ml-service/uplift.py
    mu_a(X) = E[recovered | X, arm=a], calibrated (isotonic)
    v
UPLIFT                              tau_a(X) = mu_a(X) - mu_no_action(X)
    v
UNCERTAINTY                         ml-service/uncertainty.py
    5-model bootstrap ensemble std -> HIGH/MEDIUM/LOW confidence tier
    v
DRIFT CHECK                         ml-service/drift.py
    PSI on rolling decline_code window vs training reference
    v
EXPECTED NET VALUE                  ml-service/bandit.py::net_value
    uplift_A(X) * amount - FRICTION_COST[A], per arm
    v
BANDIT SCORE (informational)        ml-service/bandit.py::LinUCB
    persisted UCB per arm; shown in trace, not yet decision-driving
    v
POLICY / CONSTRAINT GATE            ml-service/policy.py
    discount cap, retry cap, daily nudge cap, human-approval threshold,
    confidence-tier gate, drift-status gate
    v
ACTION (simulated only — no real money moves)
    v
AUDIT TRAIL                         ml-service/audit.py -> audit_log.jsonl
```

## Services

- **ml-service/** — Python FastAPI microservice. Owns every model,
  the uplift/uncertainty/drift computations, the decision logic, the
  policy gate, and audit logging. Single endpoint that matters:
  `POST /decide`, which returns the full trace above in one response.
- **backend/** — Node/Express thin orchestration layer between the
  frontend and the ML service; also serves evaluation/ablation results.
- **frontend/** — Single-file React (via CDN, no build step) dashboard:
  scoreboard, decision screen showing the full trace, audit timeline.
- **data/** — synthetic generator producing `events.csv` (training data,
  one observed outcome per event) and `oracle_potential_outcomes.csv`
  (counterfactual ground truth, eval-only, never joined into features).
- **evaluation/** — `evaluate.py` (oracle-scored 3-way comparison),
  `ablation.py` (Model A→G layer contribution), `offline_policy_eval.py`
  (IPS/SNIPS), `drift_test.py` (PSI validation on the injected drift
  window).
- **failure-lab/** — 12 scripted edge-case scenarios run against the
  live ml-service.
- **audit/** — append-only JSONL audit log.

## Why this shape

The central bet: revenue recovery is a decision problem under cost and
uncertainty, not a classification problem. Competing projects will stop
at "predict payment failure." Recoup goes further — given a failure has
happened, which intervention has the highest *causal* recovered value
net of cost, and does the model trust itself enough to act alone?

LLM calls are not on the decision path anywhere in this system — used
only to turn a decision trace into human-readable explanation text. This
is a stated, deliberate choice (see `docs/pitch_and_judge_qa.md`, Q9),
not an oversight.

See `docs/gap_analysis.md` for exactly what changed and why, file by
file, and `docs/model_cards.md` for what every trained component does,
how it was evaluated, and where it's known to be weak.
