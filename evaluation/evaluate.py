"""
Three-way evaluation on the held-out TEST split (temporal_split.py),
scored against the ORACLE counterfactual file, using the ONE canonical
reward/regret protocol in evaluation/protocol.py (round-3 fix, Phase 1).

  BASELINE  - always retry_timing
  ML-ONLY   - highest raw P(recovered|X,arm), no cost/uplift awareness
  RECOUP    - uplift-based net value + policy gate (Model G in ablation.py,
              now guaranteed to score identically here and there because
              both call protocol.score_event())
"""
import json
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
from protocol import (  # noqa: E402
    score_event, regret, oracle_optimal_action, oracle_ranked_actions,
    amount_bucket, historical_escalation_credit_rate,
)

SAMPLE_SIZE = 800


def run_evaluation():
    df = load_events()
    train_df, _, test_df = temporal_split(df)
    escalation_rate = historical_escalation_credit_rate(train_df)

    oracle_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "oracle_potential_outcomes.csv")
    )
    failed = test_df[test_df["failed"] == True].copy()
    merged = failed.merge(oracle_df, on="event_id", how="inner")
    if len(merged) > SAMPLE_SIZE:
        merged = merged.sample(n=SAMPLE_SIZE, random_state=7)

    arm_models = load_models()

    baseline_reward, ml_only_reward, recoup_reward = 0.0, 0.0, 0.0
    recoup_escalated = 0
    n = 0
    regrets, top1_hits, top2_hits = [], 0, 0
    segment_rows = []

    for _, row in merged.iterrows():
        n += 1
        row_df = pd.DataFrame([row])

        # BASELINE: always retry_timing (never escalates)
        baseline_reward += score_event(row, "retry_timing", escalation_rate)

        # ML-ONLY: highest raw recovery probability, ignoring uplift/cost
        probs = predict_arm_probs(arm_models, row_df)
        active_probs = {a: p for a, p in probs.items() if a != "no_action"}
        ml_only_arm = max(active_probs, key=active_probs.get) if active_probs else None
        ml_only_reward += score_event(row, ml_only_arm, escalation_rate)

        # RECOUP: uplift -> net value -> policy gate (Model G logic,
        # identical to ablation.py's model_g_full_recoup)
        uplift = compute_uplift(probs)
        candidate = None
        if uplift:
            nv = net_value(uplift, row["amount"])
            positive = {a: v for a, v in nv.items() if v > 0}
            if positive:
                candidate = max(positive, key=positive.get)
        recoup_chosen = None
        if candidate:
            policy_result = check_action(intervention=candidate, amount=row["amount"], retry_count=0, nudges_today=0)
            if policy_result["approved"] and not policy_result["requires_human"]:
                recoup_chosen = candidate
        if recoup_chosen is None:
            recoup_escalated += 1
        event_reward = score_event(row, recoup_chosen, escalation_rate)
        recoup_reward += event_reward

        # regret + top-K vs the SAME oracle used everywhere else
        event_regret = regret(row, recoup_chosen, escalation_rate)
        regrets.append(event_regret)
        oracle_arm, _ = oracle_optimal_action(row)
        ranked = oracle_ranked_actions(row)
        if recoup_chosen == oracle_arm:
            top1_hits += 1
        if recoup_chosen in ranked[:2]:
            top2_hits += 1

        segment_rows.append({
            "amount_bucket": amount_bucket(row["amount"]),
            "merchant_category": row.get("merchant_category"),
            "customer_value_tier": row.get("customer_value_tier"),
            "decline_code": row.get("decline_code"),
            "drift_window": bool(row.get("drift_window")),
            "recoup_reward": event_reward,
            "baseline_reward": score_event(row, "retry_timing", escalation_rate),
            "regret": event_regret,
        })

    regrets = np.array(regrets)
    seg_df = pd.DataFrame(segment_rows)

    def segment_summary(col):
        if col not in seg_df or seg_df[col].isna().all():
            return {}
        g = seg_df.groupby(col).agg(
            mean_recoup_reward=("recoup_reward", "mean"),
            mean_baseline_reward=("baseline_reward", "mean"),
            mean_regret=("regret", "mean"),
            n=("recoup_reward", "count"),
        ).round(2)
        return g.to_dict(orient="index")

    results = {
        "sample_size": n,
        "escalation_credit_rate_used": round(escalation_rate, 4),
        "baseline": {"strategy": "always retry_timing", "total_net_reward": round(baseline_reward, 2),
                     "mean_net_reward": round(baseline_reward / n, 2)},
        "ml_only": {"strategy": "highest raw P(recovered), no cost/uplift", "total_net_reward": round(ml_only_reward, 2),
                    "mean_net_reward": round(ml_only_reward / n, 2)},
        "recoup": {"strategy": "uplift + net value + policy gate", "total_net_reward": round(recoup_reward, 2),
                   "mean_net_reward": round(recoup_reward / n, 2), "escalated_or_no_action_count": recoup_escalated},
        "lift_over_baseline_pct": round(100 * (recoup_reward - baseline_reward) / max(abs(baseline_reward), 1), 1),
        "lift_over_ml_only_pct": round(100 * (recoup_reward - ml_only_reward) / max(abs(ml_only_reward), 1), 1),
        "regret_vs_oracle_optimal": {
            "mean": round(float(np.mean(regrets)), 2),
            "median": round(float(np.median(regrets)), 2),
            "p90": round(float(np.percentile(regrets, 90)), 2),
        },
        "top_k_accuracy": {
            "top1_optimal_action_rate": round(top1_hits / n, 4),
            "top2_contains_optimal_rate": round(top2_hits / n, 4),
        },
        "segment_breakdown": {
            "by_amount_bucket": segment_summary("amount_bucket"),
            "by_merchant_category": segment_summary("merchant_category"),
            "by_customer_value_tier": segment_summary("customer_value_tier"),
            "by_decline_code": segment_summary("decline_code"),
            "by_drift_window": segment_summary("drift_window"),
        },
        "note": "Every number above is computed via evaluation/protocol.py's score_event()/regret() — "
                "the same functions ablation.py uses for its 'G: FULL RECOUP' row. The two scripts can "
                "no longer disagree on this number by construction, not by coincidence.",
    }
    return results


if __name__ == "__main__":
    results = run_evaluation()
    print(json.dumps(results, indent=2))
    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to evaluation/results.json")
