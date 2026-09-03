"""
Flat policy vs Two-Stage policy (round-3, Phase 7/8).

Both scored via the SAME evaluation/protocol.py functions used by
evaluate.py and ablation.py -- no separate scoring logic here.

FLAT:      uplift = mu_a(X) - mu_no_action(X) -> net value -> policy gate
           (Model G from ablation.py)
TWO-STAGE: Stage 1 P(intervene|X) -> Stage 2 argmax mu_a(X)*amount-cost
           among actionable arms only (ml-service/two_stage_policy.py)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))
from features import load_events  # noqa: E402
from temporal_split import temporal_split  # noqa: E402
from uplift import load_models, predict_arm_probs, compute_uplift  # noqa: E402
from bandit import net_value  # noqa: E402
from policy import check_action  # noqa: E402
from two_stage_policy import load_stage1, two_stage_decide  # noqa: E402
from protocol import score_event, regret, historical_escalation_credit_rate, COST_TABLE  # noqa: E402

SAMPLE_SIZE = 600


def flat_policy_decide(arm_models, row_df, amount):
    probs = predict_arm_probs(arm_models, row_df)
    uplift = compute_uplift(probs)
    if not uplift:
        return None
    nv = net_value(uplift, amount)
    positive = {a: v for a, v in nv.items() if v > 0}
    if not positive:
        return None
    candidate = max(positive, key=positive.get)
    result = check_action(intervention=candidate, amount=amount, retry_count=0, nudges_today=0)
    if not result["approved"] or result["requires_human"]:
        return None
    return candidate


def run():
    df = load_events()
    train_df, _, test_df = temporal_split(df)
    escalation_rate = historical_escalation_credit_rate(train_df)

    oracle_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "oracle_potential_outcomes.csv"))
    failed = test_df[test_df["failed"] == True].copy()
    merged = failed.merge(oracle_df, on="event_id", how="inner")
    if len(merged) > SAMPLE_SIZE:
        merged = merged.sample(n=SAMPLE_SIZE, random_state=7)

    arm_models = load_models()
    stage1_bundle = load_stage1()
    if stage1_bundle is None:
        print("Stage 1 model not found -- run: python3 ml-service/two_stage_policy.py first")
        return

    flat_rewards, two_stage_rewards = [], []
    flat_regrets, two_stage_regrets = [], []
    flat_escalated, two_stage_no_action = 0, 0

    for _, row in merged.iterrows():
        row_df = pd.DataFrame([row])
        amount = row["amount"]

        flat_arm = flat_policy_decide(arm_models, row_df, amount)
        if flat_arm is None:
            flat_escalated += 1
        flat_r = score_event(row, flat_arm, escalation_rate)
        flat_rewards.append(flat_r)
        flat_regrets.append(regret(row, flat_arm, escalation_rate))

        ts_arm, trace = two_stage_decide(stage1_bundle, arm_models, row_df, amount, COST_TABLE)
        if ts_arm is None:
            two_stage_no_action += 1
        ts_r = score_event(row, ts_arm, escalation_rate)
        two_stage_rewards.append(ts_r)
        two_stage_regrets.append(regret(row, ts_arm, escalation_rate))

    n = len(merged)
    result = {
        "sample_size": n,
        "flat_policy": {
            "mean_net_reward": round(float(np.mean(flat_rewards)), 2),
            "mean_regret": round(float(np.mean(flat_regrets)), 2),
            "escalated_pct": round(100 * flat_escalated / n, 1),
        },
        "two_stage_policy": {
            "mean_net_reward": round(float(np.mean(two_stage_rewards)), 2),
            "mean_regret": round(float(np.mean(two_stage_regrets)), 2),
            "stage1_no_action_pct": round(100 * two_stage_no_action / n, 1),
        },
        "lift_two_stage_over_flat_pct": round(
            100 * (np.mean(two_stage_rewards) - np.mean(flat_rewards)) / max(abs(np.mean(flat_rewards)), 1), 1
        ),
        "regret_reduction_pct": round(
            100 * (np.mean(flat_regrets) - np.mean(two_stage_regrets)) / max(abs(np.mean(flat_regrets)), 1), 1
        ),
    }
    return result


if __name__ == "__main__":
    import json
    result = run()
    print(json.dumps(result, indent=2))
    with open(os.path.join(os.path.dirname(__file__), "two_stage_vs_flat_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to evaluation/two_stage_vs_flat_results.json")
