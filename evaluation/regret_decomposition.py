"""
Regret decomposition (round-4, lightweight version).

The full 12-category decomposition requested (outcome-prediction error,
treatment-effect error, calibration error, uncertainty error, overlap,
ranking error, cost-estimation error, cold-start, high-value, OOD,
simulator stochasticity) would require counterfactual instrumentation
that does not exist yet in this codebase -- NOT built this round,
stated plainly, not faked.

What IS tractable with what already exists: split every event's regret
into three buckets using data we already have (the oracle file + the
system's actual decision):

  1. ESCALATED     - Recoup chose no action / escalated (safety-driven,
                      not a wrong-arm choice)
  2. WRONG_ARM     - Recoup chose an actionable arm, but not the
                      oracle-optimal one
  3. OPTIMAL       - Recoup chose the oracle-optimal arm

This tells us the SHAPE of the regret problem (is it mostly caused by
being too conservative/escalating, or by picking the wrong arm when it
does act) even without the full source-attribution the request asked
for. A genuinely useful, honest partial answer -- not a substitute for
the full decomposition, which remains a documented gap.
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
from protocol import score_event, regret, oracle_optimal_action, historical_escalation_credit_rate, amount_bucket  # noqa: E402

SAMPLE_SIZE = 600


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
    buckets = {"ESCALATED": [], "WRONG_ARM": [], "OPTIMAL": []}
    by_amount_bucket = {}

    for _, row in merged.iterrows():
        row_df = pd.DataFrame([row])
        probs = predict_arm_probs(arm_models, row_df)
        uplift = compute_uplift(probs)
        candidate = None
        if uplift:
            nv = net_value(uplift, row["amount"])
            positive = {a: v for a, v in nv.items() if v > 0}
            if positive:
                candidate = max(positive, key=positive.get)
        chosen = None
        if candidate:
            policy_result = check_action(intervention=candidate, amount=row["amount"], retry_count=0, nudges_today=0)
            if policy_result["approved"] and not policy_result["requires_human"]:
                chosen = candidate

        event_regret = regret(row, chosen, escalation_rate)
        oracle_arm, _ = oracle_optimal_action(row)

        if chosen is None:
            bucket = "ESCALATED"
        elif chosen == oracle_arm:
            bucket = "OPTIMAL"
        else:
            bucket = "WRONG_ARM"
        buckets[bucket].append(event_regret)

        ab = amount_bucket(row["amount"])
        by_amount_bucket.setdefault(ab, []).append(event_regret)

    total_regret = sum(sum(v) for v in buckets.values())
    result = {
        "total_regret": round(total_regret, 2),
        "by_decision_type": {
            k: {
                "n": len(v),
                "pct_of_events": round(100 * len(v) / len(merged), 1),
                "total_regret_contribution": round(sum(v), 2),
                "pct_of_total_regret": round(100 * sum(v) / max(total_regret, 1), 1),
                "mean_regret": round(float(np.mean(v)), 2) if v else 0.0,
            }
            for k, v in buckets.items()
        },
        "by_amount_bucket_mean_regret": {
            k: round(float(np.mean(v)), 2) for k, v in by_amount_bucket.items()
        },
        "note": "Partial decomposition (3 buckets: escalated / wrong-arm / optimal), NOT the full "
                "12-source attribution requested. See docs/round4_findings.md for what remains undone.",
    }
    return result


if __name__ == "__main__":
    import json
    result = run()
    print(json.dumps(result, indent=2))
    with open(os.path.join(os.path.dirname(__file__), "regret_decomposition_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to evaluation/regret_decomposition_results.json")
