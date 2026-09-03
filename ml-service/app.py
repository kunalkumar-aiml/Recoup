"""
Recoup ML microservice — full decision trace (round-2 hardened).

POST /decide runs:
  INPUT STATE
  -> ROOT-CAUSE POSTERIOR
  -> RECOVERY PREDICTIONS (per-arm P(recovered|X,arm), T-learner)
  -> UPLIFT (empty {} if no_action lacks INSUFFICIENT_CONTROL_SUPPORT --
     see uplift.py; the decision engine treats this as forced escalation,
     never a silent fallback)
  -> UNCERTAINTY (bootstrap ensemble std -> epistemic-uncertainty-proxy
     tier), now computed for EVERY policy-eligible candidate arm, not
     just the single top-uplift arm (round-2 fix #8)
  -> DRIFT CHECK (PSI across decline_code, payment_method,
     customer_value_tier, and binned amount -- round-2 fix #16)
  -> EXPECTED NET VALUE (uplift * amount - cost - discount leakage)
  -> BANDIT SCORE (persisted LinUCB, informational -- see
     docs/model_cards.md for why it is not yet decision-driving)
  -> SAFE ACTION RANKING (round-2 fix #15: rank all candidates, return
     the highest-ranked POLICY-SAFE one, not "pick then reject")
  -> ACTION + decision_id (persisted context for genuine bandit feedback)
  -> AUDIT LOG

POST /feedback(decision_id, arm, recovered, amount) retrieves the REAL
context from decision_store.py and updates LinUCB with it -- round-2
fix #4. Arbitrary client-supplied context vectors are no longer accepted.
"""
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import build_feature_frame, align_columns
from uplift import predict_arm_probs, compute_uplift, ARMS
from bandit import net_value, select_safe_action, LinUCB, FRICTION_COST, normalize_reward
from uncertainty import ensemble_predict, confidence_tier
from drift import multi_signal_drift_status
from policy import check_action
from audit import log_decision, read_audit_log
from decision_store import (get_or_create_decision_id, get_decision_context,  # noqa: E402
                             finalize_decision, get_cached_decision_result,
                             is_feedback_already_processed, mark_feedback_processed,
                             get_event_id_for_decision)
from event_store import record_raw_event, update_event_outcome, compute_trusted_temporal_state  # noqa: E402

app = FastAPI(title="Recoup ML Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

root_cause_bundle = joblib.load(os.path.join(MODEL_DIR, "root_cause.pkl"))
uplift_arm_models = joblib.load(os.path.join(MODEL_DIR, "uplift_arms.pkl"))
uncertainty_ensembles = joblib.load(os.path.join(MODEL_DIR, "uncertainty_ensembles.pkl"))

_bandit = LinUCB.load()
if _bandit is None:
    n_features = len(next(iter(uplift_arm_models.values()))["columns"])
    _bandit = LinUCB(arms=ARMS, n_features=n_features, alpha=1.0)
    _bandit.save()

# reference distributions for multi-signal drift monitoring, computed once
# at startup from the training data (round-2 fix #16: beyond decline_code)
_events_path = os.path.join(os.path.dirname(__file__), "..", "data", "events.csv")
try:
    _df = pd.read_csv(_events_path)
    _failed_ref = _df[_df["failed"] == True]
    _REFERENCE = {
        "decline_code": _failed_ref["decline_code"].fillna("none").tolist()[:6000],
        "payment_method": _failed_ref["payment_method"].tolist()[:6000],
        "customer_value_tier": _failed_ref["customer_value_tier"].fillna("unknown").tolist()[:6000],
        "amount_bucket": pd.cut(_failed_ref["amount"], bins=[0, 500, 1500, 5000, 1e9],
                                  labels=["low", "mid", "high", "very_high"]).astype(str).tolist()[:6000],
    }
except Exception:
    _REFERENCE = {k: [] for k in ["decline_code", "payment_method", "customer_value_tier", "amount_bucket"]}

_recent = {k: [] for k in _REFERENCE}


class EventIn(BaseModel):
    event_id: str
    customer_id: str = "unknown"  # round-7 fix (Phase 4): needed to look up server-side history
    event_timestamp: str = None  # ISO 8601; defaults to ingestion time if not supplied
    merchant_category: str
    customer_value_tier: str
    event_type: str
    payment_method: str
    decline_code: str
    amount: float
    customer_retry_propensity_observed: float = 0.5
    merchant_baseline_fail_rate: float = 0.2
    prior_failure_count: int = 0
    minutes_since_last_failure: float = -1
    prior_recovery_count: int = 0
    prior_recovery_rate: float = -1
    recent_method_switch_count: int = 0
    retry_count: int = 0
    nudges_today: int = 0
    discount_pct: float = 0


class FeedbackIn(BaseModel):
    decision_id: str
    arm: str
    recovered: bool
    amount: float


@app.get("/health")
def health():
    return {"status": "ok", "arms_loaded": list(uplift_arm_models.keys())}


def _amount_bucket(amount):
    if amount <= 500:
        return "low"
    if amount <= 1500:
        return "mid"
    if amount <= 5000:
        return "high"
    return "very_high"


@app.post("/decide")
def decide_action(event: EventIn):
    # round-7 fix (Phase 2): ATOMIC idempotency. get_or_create_decision_id
    # does the event_id-uniqueness check AND reservation as one atomic
    # SQLite transaction, closing the race window round 6 left open (two
    # truly concurrent requests for a brand-new event_id could both read
    # "not present" from an unlocked dict before either wrote). Verified
    # under real thread concurrency in ml-service/test_concurrency.py.
    decision_id, is_first_caller = get_or_create_decision_id(event.event_id)

    if not is_first_caller:
        # Another caller (sequential retry, or a genuine concurrent
        # request) already owns this event_id. Wait briefly for its
        # result; if it doesn't land in time, fail closed rather than
        # silently proceeding to run inference a second time.
        cached = get_cached_decision_result(decision_id, wait_for_result=True, timeout_s=3.0)
        if cached is not None:
            cached = dict(cached)
            cached["idempotent_replay"] = True
            return cached
        return {
            "event_id": event.event_id, "decision_id": decision_id,
            "decision_state": "SYSTEM_ERROR", "chosen_intervention": None,
            "requires_human_review": True,
            "decision_reason": "concurrent first-caller did not complete in time; failing closed",
            "error": True,
        }

    # round-6 fix (Part 1/3): FAIL CLOSED. If anything in the decision
    # pipeline raises, do not let FastAPI's default 500 be the only
    # signal -- return an explicit SYSTEM_ERROR decision that requires
    # human review, so a client that doesn't check HTTP status codes
    # carefully still gets a safe, unambiguous "do not automate" answer.
    try:
        return _run_decision_pipeline(event, decision_id)
    except Exception as e:
        return {
            "event_id": event.event_id,
            "decision_id": decision_id,
            "decision_state": "SYSTEM_ERROR",
            "chosen_intervention": None,
            "requires_human_review": True,
            "decision_reason": f"internal error, failing closed: {type(e).__name__}: {e}",
            "error": True,
        }


def _run_decision_pipeline(event: "EventIn", decision_id: str):
    # round-7 fix (Phase 4): SERVER-SIDE TEMPORAL STATE. Whatever the
    # client claimed for prior_failure_count / prior_recovery_count /
    # prior_recovery_rate / minutes_since_last_failure is DISCARDED here
    # and replaced with values computed from this service's own event
    # store -- a client cannot influence its own risk profile by lying
    # about its history. If this is the customer's first-ever event,
    # the trusted state is naturally all-zero/unknown, matching what a
    # real event store would show.
    event_ts = event.event_timestamp or __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()
    trusted_state = compute_trusted_temporal_state(event.customer_id, event_ts)
    event.prior_failure_count = trusted_state["prior_failure_count"]
    event.prior_recovery_count = trusted_state["prior_recovery_count"]
    event.prior_recovery_rate = trusted_state["prior_recovery_rate"]
    event.minutes_since_last_failure = trusted_state["minutes_since_last_failure"]
    record_raw_event(event.event_id, event.customer_id, event_ts, event.amount)

    row = pd.DataFrame([event.dict()])

    # 1. ROOT-CAUSE POSTERIOR
    root_cause_posterior = {}
    try:
        rc_categorical = ["merchant_category", "customer_value_tier", "event_type", "payment_method"]
        X_rc = pd.get_dummies(row[rc_categorical + [
            "amount", "customer_retry_propensity_observed", "merchant_baseline_fail_rate",
            "prior_failure_count", "minutes_since_last_failure", "prior_recovery_count",
            "prior_recovery_rate", "recent_method_switch_count",
        ]], columns=rc_categorical)
        X_rc = align_columns(X_rc, root_cause_bundle["columns"])
        proba = root_cause_bundle["model"].predict_proba(X_rc)[0]
        root_cause_posterior = {
            cls: round(float(p), 4) for cls, p in zip(root_cause_bundle["classes"], proba)
        }
    except Exception as e:
        root_cause_posterior = {"error": str(e)}

    # 2. RECOVERY PREDICTIONS per arm (T-learner)
    recovery_probs = predict_arm_probs(uplift_arm_models, row)

    # 3. UPLIFT -- {} if no_action lacks support (uplift.py's honest failure mode)
    uplift = compute_uplift(recovery_probs)

    # 4. UNCERTAINTY -- round-2 fix #8: compute for EVERY policy-eligible
    # candidate arm (positive uplift), not just the single top-uplift arm
    uncertainty_by_arm = {}
    for arm, u in uplift.items():
        if u <= 0 or arm not in uncertainty_ensembles:
            continue
        bundle = uncertainty_ensembles[arm]
        X = build_feature_frame(row)
        X = align_columns(X, bundle["columns"])
        mean, std = ensemble_predict(bundle["models"], X)
        uncertainty_by_arm[arm] = {"ensemble_mean": round(mean, 4), "ensemble_std": round(std, 4),
                                     "confidence_tier": confidence_tier(std)}
    # overall tier used for the gate = the tier of the arm net-value would
    # currently rank first (worst-case-safe: if that arm's tier is LOW,
    # the whole decision is gated LOW)
    nv_preview = net_value(uplift, event.amount, event.discount_pct) if uplift else {}
    top_candidate = max(nv_preview, key=nv_preview.get) if nv_preview else None
    overall_tier = uncertainty_by_arm.get(top_candidate, {}).get("confidence_tier", "LOW")

    # 5. DRIFT CHECK -- round-2 fix #16: multi-signal, not just decline_code
    for key in _recent:
        val = {
            "decline_code": event.decline_code,
            "payment_method": event.payment_method,
            "customer_value_tier": event.customer_value_tier,
            "amount_bucket": _amount_bucket(event.amount),
        }[key]
        _recent[key].append(val)
        if len(_recent[key]) > 300:
            _recent[key].pop(0)
    drift_report = multi_signal_drift_status(_REFERENCE, _recent, min_window=30)
    d_status = drift_report["overall_status"]

    # 6. EXPECTED NET VALUE (discount leakage modeled -- round-2 fix #13)
    nv = net_value(uplift, event.amount, event.discount_pct) if uplift else {}

    # 7. BANDIT SCORE (informational) -- also compute + persist the REAL
    # context vector for genuine feedback-driven learning (round-2 fix #4)
    bandit_scores = {}
    context_vector = None
    context_columns = None
    if uplift and top_candidate:
        X = build_feature_frame(row)
        top_bundle_cols = uplift_arm_models[top_candidate]["columns"]
        X_aligned = align_columns(X, top_bundle_cols)
        x_vec = X_aligned.values[0].astype(float)
        norm = np.linalg.norm(x_vec)
        if norm > 0:
            x_vec = x_vec / norm
        context_vector = x_vec.tolist()
        context_columns = top_bundle_cols
        for arm in ARMS:
            if arm == "no_action":
                continue
            try:
                ucb, mean, bonus = _bandit.ucb_score(arm, x_vec)
                bandit_scores[arm] = {"ucb": round(ucb, 3), "mean": round(mean, 3), "exploration_bonus": round(bonus, 3)}
            except Exception:
                pass

    # 8. SAFE ACTION RANKING -- round-2 fix #15
    chosen, decision_reason, safety_fallback, policy_reason, ranked_trace = select_safe_action(
        nv, overall_tier, d_status, check_action,
        amount=event.amount, retry_count=event.retry_count,
        nudges_today=event.nudges_today, discount_pct=event.discount_pct,
    )

    # round-2 fix #14: distinguish a deliberate human_escalation ACTION
    # from a safety-driven human-review FALLBACK
    is_escalation_action = chosen == "human_escalation"
    final_requires_human = safety_fallback or is_escalation_action

    # round-7: decision_id was already atomically reserved by the caller
    # (decide_action) before this pipeline started; final result (including
    # context) is persisted once, below, after `result` is fully built.

    # round-6 fix (Part 3): explicit fail-closed state machine, not just
    # boolean flags -- AUTO_APPROVED / SAFE_FALLBACK / HUMAN_REVIEW are
    # now a single unambiguous field instead of something a caller has
    # to derive by combining three booleans correctly.
    if chosen is None:
        decision_state = "SAFE_FALLBACK"
    elif chosen == "human_escalation" or final_requires_human:
        decision_state = "HUMAN_REVIEW"
    else:
        decision_state = "AUTO_APPROVED"

    result = {
        "event_id": event.event_id,
        "decision_id": decision_id,
        "decision_state": decision_state,
        "root_cause_posterior": root_cause_posterior,
        "recovery_probabilities": {k: round(v, 4) for k, v in recovery_probs.items()},
        "uplift_estimates": uplift,
        "uplift_available": bool(uplift),
        "uncertainty_by_candidate_arm": uncertainty_by_arm,
        "drift": drift_report,
        "expected_net_value": nv,
        "bandit_scores": bandit_scores,
        "chosen_intervention": chosen,
        "decision_reason": decision_reason,
        "safety_fallback_triggered": safety_fallback,
        "is_deliberate_escalation_action": is_escalation_action,
        "policy_reason": policy_reason,
        "requires_human_review": final_requires_human,
        "ranked_candidates": ranked_trace,
    }

    # round-7: always persist the final result against the atomically-
    # reserved decision_id (regardless of whether a bandit context was
    # computed), so idempotent replays work for EVERY decision outcome,
    # including SAFE_FALLBACK cases with no positive-uplift arm.
    finalize_decision(decision_id, context_vector, context_columns, result)

    log_decision({
        "event_id": event.event_id, "decision_id": decision_id,
        "input": event.dict(), "root_cause_posterior": root_cause_posterior,
        "recovery_probabilities": recovery_probs, "uplift_estimates": uplift,
        "drift": drift_report, "expected_net_value": nv,
        "chosen_intervention": chosen, "decision_reason": decision_reason,
        "safety_fallback_triggered": safety_fallback,
        "requires_human_review": final_requires_human,
    })

    return result


@app.post("/feedback")
def feedback(fb: FeedbackIn):
    """Round-2 fix #4: retrieves the REAL context from decision_store
    rather than accepting a client-supplied (previously zero) vector.
    Round-6 fix (Part 5): idempotent -- a decision_id that already
    received feedback is not updated a second time, regardless of how
    many times this endpoint is called for it (retry, duplicate,
    replay).

    ROUND-7 FIX: the previous version called is_feedback_already_processed()
    (a READ) first, then updated the bandit, then called
    mark_feedback_processed() (a WRITE) LAST. That is a classic
    check-then-act race: two concurrent requests can both pass the read
    check before either has written, and both proceed to update the
    bandit. test_concurrency.py caught this for real (2/10 and 5/50
    concurrent feedback calls both updated). Fix: claim the atomic
    INSERT OR IGNORE FIRST; only the call whose insert actually wins
    (rowcount == 1) is allowed to touch the bandit at all."""
    context = get_decision_context(fb.decision_id)
    if context is None:
        raise HTTPException(status_code=404, detail="unknown decision_id -- cannot verify original context")

    raw_reward = (fb.amount if fb.recovered else 0) - FRICTION_COST.get(fb.arm, 0)
    reward = normalize_reward(raw_reward)

    # atomic claim FIRST -- only the winner proceeds past this point
    won_claim = mark_feedback_processed(fb.decision_id, fb.arm, reward)
    if not won_claim:
        return {"updated": False, "already_processed": True, "arm": fb.arm}

    x_vec = np.array(context, dtype=float)
    if fb.arm in ARMS:
        _bandit.update(fb.arm, x_vec, reward)
        _bandit.save()

    # round-7 fix (Phase 4 wiring): update the event store's outcome for
    # this event so FUTURE events from the same customer see the correct
    # prior_recovery_count / prior_recovery_rate -- without this, the
    # server-side trusted state would always show every prior event as
    # "not recovered" since /decide never learns the eventual outcome.
    event_id = get_event_id_for_decision(fb.decision_id)
    if event_id:
        update_event_outcome(event_id, fb.recovered, fb.arm)
    return {"updated": True, "already_processed": False, "arm": fb.arm, "raw_reward": raw_reward, "normalized_reward": reward}


@app.get("/audit")
def audit(limit: int = 50):
    return read_audit_log(limit)
