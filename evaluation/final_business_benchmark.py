"""
Final batch-level business benchmark.

Answers the Razorpay Revenue Recovery track's actual question directly:
"For a batch of N failed payments with Rs X at risk, what did each
strategy actually recover?" -- using the same evaluation/protocol.py
functions as every other script in this repo (no new scoring logic).

SYNTHETIC SIMULATION. NOT REAL RAZORPAY DATA. Every number below is
computed against this repo's synthetic counterfactual oracle
(data/oracle_potential_outcomes.csv), not observed production outcomes.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))
from features import load_events  # noqa: E402
from temporal_split import temporal_split  # noqa: E402
from uplift import load_models, predict_arm_probs, compute_uplift  # noqa: E402
from bandit import net_value  # noqa: E402
from policy import check_action  # noqa: E402
from protocol import score_event, historical_escalation_credit_rate, COST_TABLE  # noqa: E402

SAMPLE_SIZE = 800


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
    n = len(merged)
    total_at_risk = float(merged["amount"].sum())

    strategies = {"always_retry": {}, "ml_only": {}, "recoup": {}}
    for key in strategies:
        strategies[key] = {
            "gross_recovered": 0.0, "intervention_cost": 0.0, "net_recovered": 0.0,
            "n_automated": 0, "n_human_review": 0, "n_no_action": 0, "n_unsafe_blocked": 0,
        }

    for _, row in merged.iterrows():
        row_df = pd.DataFrame([row])
        amount = row["amount"]

        # ---- Always retry ----
        col = "potential_recovered_retry_timing"
        recovered = bool(row[col]) if col in row and not pd.isna(row[col]) else False
        gross = amount if recovered else 0.0
        cost = COST_TABLE.get("retry_timing", 0)
        strategies["always_retry"]["gross_recovered"] += gross
        strategies["always_retry"]["intervention_cost"] += cost
        strategies["always_retry"]["net_recovered"] += gross - cost
        strategies["always_retry"]["n_automated"] += 1

        # ---- ML-only (highest raw probability, no cost/uplift awareness) ----
        probs = predict_arm_probs(arm_models, row_df)
        active_probs = {a: p for a, p in probs.items() if a != "no_action"}
        ml_arm = max(active_probs, key=active_probs.get) if active_probs else None
        col = f"potential_recovered_{ml_arm}" if ml_arm else None
        recovered = bool(row[col]) if col and col in row and not pd.isna(row[col]) else False
        gross = amount if recovered else 0.0
        cost = COST_TABLE.get(ml_arm, 0) if ml_arm else 0
        strategies["ml_only"]["gross_recovered"] += gross
        strategies["ml_only"]["intervention_cost"] += cost
        strategies["ml_only"]["net_recovered"] += gross - cost
        strategies["ml_only"]["n_automated"] += 1

        # ---- Recoup (uplift + net value + policy gate) ----
        uplift = compute_uplift(probs)
        candidate = None
        if uplift:
            nv = net_value(uplift, amount)
            positive = {a: v for a, v in nv.items() if v > 0}
            if positive:
                candidate = max(positive, key=positive.get)
        chosen = None
        blocked = False
        if candidate:
            pr = check_action(intervention=candidate, amount=amount, retry_count=0, nudges_today=0)
            if pr["approved"] and not pr["requires_human"]:
                chosen = candidate
            elif not pr["approved"]:
                blocked = True

        if chosen:
            col = f"potential_recovered_{chosen}"
            recovered = bool(row[col]) if col in row and not pd.isna(row[col]) else False
            gross = amount if recovered else 0.0
            cost = COST_TABLE.get(chosen, 0)
            strategies["recoup"]["gross_recovered"] += gross
            strategies["recoup"]["intervention_cost"] += cost
            strategies["recoup"]["net_recovered"] += gross - cost
            strategies["recoup"]["n_automated"] += 1
        else:
            # escalated / no economically justified action -- scored via
            # the same escalation-credit rule as every other script
            credit_gross = escalation_rate * amount
            cost = COST_TABLE.get("human_escalation", 30)
            strategies["recoup"]["gross_recovered"] += credit_gross
            strategies["recoup"]["intervention_cost"] += cost
            strategies["recoup"]["net_recovered"] += credit_gross - cost
            strategies["recoup"]["n_human_review"] += 1
            if candidate is None:
                strategies["recoup"]["n_no_action"] += 1
            if blocked:
                strategies["recoup"]["n_unsafe_blocked"] += 1

    result = {
        "SIMULATION_TYPE": "SYNTHETIC -- NOT REAL RAZORPAY DATA",
        "batch_size_n": n,
        "total_revenue_at_risk": round(total_at_risk, 2),
        "strategies": {},
    }
    for key, s in strategies.items():
        result["strategies"][key] = {
            "gross_recovered": round(s["gross_recovered"], 2),
            "intervention_cost": round(s["intervention_cost"], 2),
            "net_recovered": round(s["net_recovered"], 2),
            "recovery_rate_pct": round(100 * s["gross_recovered"] / total_at_risk, 2),
            "automation_pct": round(100 * s["n_automated"] / n, 1),
            "human_review_pct": round(100 * s["n_human_review"] / n, 1),
            "escalated_or_no_action_count": s["n_no_action"],
            "unsafe_action_blocked_count": s["n_unsafe_blocked"],
        }

    baseline_net = result["strategies"]["always_retry"]["net_recovered"]
    recoup_net = result["strategies"]["recoup"]["net_recovered"]
    ml_net = result["strategies"]["ml_only"]["net_recovered"]
    result["incremental_net_recovered_vs_always_retry"] = round(recoup_net - baseline_net, 2)
    result["incremental_net_recovered_vs_ml_only"] = round(recoup_net - ml_net, 2)
    result["lift_vs_always_retry_pct"] = round(100 * (recoup_net - baseline_net) / abs(baseline_net), 1)
    result["lift_vs_ml_only_pct"] = round(100 * (recoup_net - ml_net) / abs(ml_net), 1)
    result["headline_sentence"] = (
        f"For a synthetic batch of {n} failed payments totaling Rs {total_at_risk:,.0f} at risk, "
        f"Recoup produced Rs {recoup_net:,.0f} net recovered value under this repo's evaluation "
        f"protocol, vs Rs {baseline_net:,.0f} for always-retry and Rs {ml_net:,.0f} for a cost-blind "
        f"ML-only strategy (single seed -- see docs/final_policy_selection.md for multi-seed variance)."
    )
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    with open(os.path.join(os.path.dirname(__file__), "final_business_benchmark.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to evaluation/final_business_benchmark.json")
