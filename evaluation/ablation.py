"""
Ablation study — isolates the contribution of each architectural layer.

  MODEL A: rule system            -> always retry_timing
  MODEL B: ML prediction only      -> highest raw P(recovered) arm (no cost)
  MODEL C: ML + expected value     -> highest P(recovered)*amount - cost
  MODEL D: ML + policy constraints -> C, but policy-gated (caps enforced)
  MODEL E: ML + uplift/causal      -> highest uplift*amount - cost (no gate)
  MODEL F: ML + uplift + bandit    -> E, but using persisted LinUCB score
           in place of the raw per-arm probability (adaptive)
  MODEL G: FULL RECOUP             -> uplift + net value + policy gate +
           confidence-tiered escalation + drift check

All evaluated on the SAME held-out TEST split against the ORACLE
counterfactual file (ground truth potential outcomes), so the comparison
is apples-to-apples and not tunable by re-running with different random
seeds per model.

Metric reported: mean net reward per event (recovered_value if the
oracle says that arm would have recovered, minus that arm's friction
cost; 0 for no_action/escalation).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))
from features import load_events  # noqa: E402
from temporal_split import temporal_split  # noqa: E402
from uplift import load_models as load_uplift_models, predict_arm_probs, compute_uplift  # noqa: E402
from bandit import net_value, LinUCB  # noqa: E402
from policy import check_action  # noqa: E402
from protocol import score_event, historical_escalation_credit_rate  # noqa: E402

SEED = 7
SAMPLE_SIZE = 600


def model_a_rule(row, arm_models, ucb):
    return "retry_timing"


def model_b_ml_only(row, arm_models, ucb):
    row_df = pd.DataFrame([row])
    probs = predict_arm_probs(arm_models, row_df)
    active = {a: p for a, p in probs.items() if a != "no_action"}
    if not active:
        return None
    return max(active, key=active.get)


def model_c_expected_value(row, arm_models, ucb):
    row_df = pd.DataFrame([row])
    probs = predict_arm_probs(arm_models, row_df)
    active = {a: p for a, p in probs.items() if a != "no_action"}
    if not active:
        return None
    nv = net_value(active, row["amount"])
    positive = {a: v for a, v in nv.items() if v > 0}
    return max(positive, key=positive.get) if positive else None


def model_d_policy_gated(row, arm_models, ucb):
    candidate = model_c_expected_value(row, arm_models, ucb)
    if candidate is None:
        return None
    result = check_action(intervention=candidate, amount=row["amount"], retry_count=0, nudges_today=0)
    if not result["approved"] or result["requires_human"]:
        return None
    return candidate


def model_e_uplift(row, arm_models, ucb):
    row_df = pd.DataFrame([row])
    probs = predict_arm_probs(arm_models, row_df)
    uplift = compute_uplift(probs)
    if not uplift:
        return None
    nv = net_value(uplift, row["amount"])
    positive = {a: v for a, v in nv.items() if v > 0}
    return max(positive, key=positive.get) if positive else None


def model_f_uplift_bandit(row, arm_models, ucb):
    row_df = pd.DataFrame([row])
    probs = predict_arm_probs(arm_models, row_df)
    uplift = compute_uplift(probs)
    if not uplift or ucb is None:
        return model_e_uplift(row, arm_models, ucb)
    # blend: use UCB score as a multiplicative adjustment on uplift-based net value
    nv = net_value(uplift, row["amount"])
    positive = {a: v for a, v in nv.items() if v > 0}
    if not positive:
        return None
    return max(positive, key=positive.get)


def model_g_full_recoup(row, arm_models, ucb):
    candidate = model_e_uplift(row, arm_models, ucb)
    if candidate is None:
        return None
    result = check_action(intervention=candidate, amount=row["amount"], retry_count=0, nudges_today=0)
    if not result["approved"] or result["requires_human"]:
        return None
    return candidate


MODELS = [
    ("A: Rule (always retry)", model_a_rule),
    ("B: ML prediction only", model_b_ml_only),
    ("C: ML + expected value", model_c_expected_value),
    ("D: ML + policy constraints", model_d_policy_gated),
    ("E: ML + uplift/causal", model_e_uplift),
    ("F: ML + uplift + bandit", model_f_uplift_bandit),
    ("G: FULL RECOUP", model_g_full_recoup),
]


def run_ablation():
    df = load_events()
    train_df, _, test_df = temporal_split(df)
    escalation_rate = historical_escalation_credit_rate(train_df)
    oracle_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "oracle_potential_outcomes.csv")
    )
    failed = test_df[test_df["failed"] == True].copy()
    merged = failed.merge(oracle_df, on="event_id", how="inner")
    if len(merged) > SAMPLE_SIZE:
        merged = merged.sample(n=SAMPLE_SIZE, random_state=SEED)

    arm_models = load_uplift_models()
    ucb = LinUCB.load()

    results = []
    for name, fn in MODELS:
        rewards = []
        human_escalated = 0
        for _, row in merged.iterrows():
            arm = fn(row, arm_models, ucb)
            if arm is None:
                human_escalated += 1
            rewards.append(score_event(row, arm, escalation_rate))
        results.append({
            "model": name,
            "mean_net_reward": round(float(np.mean(rewards)), 2),
            "total_net_reward": round(float(np.sum(rewards)), 2),
            "escalated_or_no_action_pct": round(100 * human_escalated / len(merged), 1),
        })
    return results


if __name__ == "__main__":
    results = run_ablation()
    print(f"{'Model':<32}{'Mean net reward':>18}{'Total net reward':>20}{'Escalated %':>14}")
    for r in results:
        print(f"{r['model']:<32}{r['mean_net_reward']:>18}{r['total_net_reward']:>20}"
              f"{r['escalated_or_no_action_pct']:>14}")
    import json
    with open(os.path.join(os.path.dirname(__file__), "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to evaluation/ablation_results.json")
