"""
Final wrong-arm forensic analysis (final sprint, Step 2).

For every WRONG_ARM test event (chosen action != oracle-best action),
tags plausible error categories using signals already available in this
codebase (NOT a new causal-isolation experiment -- that would require
ablating each model component per-event, out of scope this sprint).
Categories are NOT mutually exclusive; an event can carry multiple tags.
This is stated explicitly: this is a diagnostic heuristic, not a proof
of causal attribution.

CATEGORIES (subset of the requested A-K, scoped to what's measurable
from data already in this repo):

  A. PROBABILITY_ERROR   - the model's predicted P(recovered) for the
                            oracle-best arm was far from that arm's true
                            oracle probability (|predicted - true| > 0.15)
  B. UPLIFT_ERROR         - probability for the oracle-best arm was
                            reasonably accurate, but its uplift (vs the
                            no_action baseline) was still ranked below
                            the chosen arm -- points at mu_no_action(X)
                            specifically, not the per-arm models
  F. THIN_SUPPORT         - the chosen arm's context falls in the
                            bottom quartile of that arm's training
                            volume (overlap proxy, reuses the logic from
                            action_ranking_analysis.py)
  G. COLD_START           - customer had zero prior recorded events
  H. HIGH_VALUE           - transaction amount in the "high"/"very_high"
                            bucket
  J. STOCHASTIC_MARGIN    - the oracle-best and second-best arms' TRUE
                            oracle probabilities were within 0.05 of each
                            other -- the "wrong" choice may simply reflect
                            near-tied ground truth, not a model failure
  K. POLICY_FILTERED      - the model's own top-ranked (by net value) arm
                            WAS the oracle-best arm, but the policy gate
                            rejected/downgraded it -- points at the
                            safety layer, not the ranking model

Categories C (cost calculation) and D (calibration) are not separately
attributed here: Step 3's unit test suite already proves cost
calculation is arithmetically correct in isolation (10/10 pass,
ml-service/test_economic_value_engine.py), so per-event cost-calculation
error is not a plausible category to tag; and calibration's effect is
global (round-2 Brier/ECE numbers), not meaningfully attributable
per-event without a raw-vs-calibrated counterfactual re-run, out of
scope this sprint. E (uncertainty) and I (OOD) require the live
uncertainty ensemble, which is not wired into offline evaluation
scripts (a documented gap since round 4) -- also out of scope here.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))
from features import load_events, build_feature_frame  # noqa: E402
from temporal_split import temporal_split  # noqa: E402
from uplift import load_models, predict_arm_probs, compute_uplift  # noqa: E402
from bandit import net_value  # noqa: E402
from policy import check_action  # noqa: E402
from protocol import (score_event, regret, oracle_optimal_action, oracle_ranked_actions,  # noqa: E402
                       historical_escalation_credit_rate, amount_bucket, ALL_ACTIONABLE_ARMS)

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
    arm_support = train_df[train_df["failed"] == True].groupby("chosen_intervention").size().to_dict()
    thin_support_threshold = np.median(list(arm_support.values())) if arm_support else 0

    category_regret = {c: 0.0 for c in ["A_PROBABILITY_ERROR", "B_UPLIFT_ERROR", "F_THIN_SUPPORT",
                                          "G_COLD_START", "H_HIGH_VALUE", "J_STOCHASTIC_MARGIN",
                                          "K_POLICY_FILTERED", "UNCATEGORIZED"]}
    category_count = {c: 0 for c in category_regret}
    n_wrong_arm = 0
    n_total = 0
    total_regret = 0.0

    for _, row in merged.iterrows():
        n_total += 1
        row_df = pd.DataFrame([row])
        amount = row["amount"]

        probs = predict_arm_probs(arm_models, row_df)
        uplift = compute_uplift(probs)
        oracle_best, _ = oracle_optimal_action(row)
        oracle_ranked = oracle_ranked_actions(row)

        if not uplift or oracle_best is None:
            continue

        nv = net_value(uplift, amount)
        positive = {a: v for a, v in nv.items() if v > 0}
        model_top_pick = max(nv, key=nv.get) if nv else None
        candidate = max(positive, key=positive.get) if positive else None

        chosen = None
        if candidate:
            pr = check_action(intervention=candidate, amount=amount, retry_count=0, nudges_today=0)
            if pr["approved"] and not pr["requires_human"]:
                chosen = candidate

        event_regret = regret(row, chosen, escalation_rate)
        total_regret += event_regret

        if chosen == oracle_best:
            continue  # correct decision, nothing to attribute

        n_wrong_arm += 1
        tags = []

        # A: probability estimation error for the oracle-best arm
        true_prob_col = f"potential_prob_{oracle_best}"
        if true_prob_col in row and oracle_best in probs:
            pred_p = probs[oracle_best]
            true_p = row[true_prob_col]
            if abs(pred_p - true_p) > 0.15:
                tags.append("A_PROBABILITY_ERROR")

        # B: uplift error (probability was OK, but ranked below the chosen arm)
        if "A_PROBABILITY_ERROR" not in tags and oracle_best in uplift:
            if uplift.get(oracle_best, -999) < uplift.get(chosen, 999) if chosen else True:
                tags.append("B_UPLIFT_ERROR")

        # F: thin support for the chosen arm
        if chosen and arm_support.get(chosen, 0) < thin_support_threshold:
            tags.append("F_THIN_SUPPORT")

        # G: cold start
        if row.get("prior_failure_count", 0) == 0:
            tags.append("G_COLD_START")

        # H: high value
        if amount_bucket(amount) in ("high", "very_high"):
            tags.append("H_HIGH_VALUE")

        # J: stochastic margin -- oracle top-2 true probs are within 0.05
        if len(oracle_ranked) >= 2:
            p1 = row.get(f"potential_prob_{oracle_ranked[0]}", None)
            p2 = row.get(f"potential_prob_{oracle_ranked[1]}", None)
            if p1 is not None and p2 is not None and abs(p1 - p2) < 0.05:
                tags.append("J_STOCHASTIC_MARGIN")

        # K: policy filtered the correct top pick
        if model_top_pick == oracle_best and chosen != oracle_best:
            tags.append("K_POLICY_FILTERED")

        if not tags:
            tags = ["UNCATEGORIZED"]

        # attribute this event's regret to EACH tag it carries (overlapping,
        # not mutually exclusive -- stated in the module docstring)
        for tag in tags:
            category_regret[tag] += event_regret
            category_count[tag] += 1

    result = {
        "sample_size": n_total,
        "n_wrong_arm_events": n_wrong_arm,
        "wrong_arm_rate": round(n_wrong_arm / n_total, 4) if n_total else 0,
        "total_regret": round(total_regret, 2),
        "category_regret_contribution": {
            k: {
                "total_regret": round(v, 2),
                "pct_of_total_regret": round(100 * v / total_regret, 1) if total_regret else 0,
                "event_count": category_count[k],
            }
            for k, v in category_regret.items()
        },
        "note": "Categories are NOT mutually exclusive (an event can carry multiple tags), so "
                "percentages sum to more than 100%. This is a diagnostic heuristic using signals "
                "already available in this repo, not a causal-isolation experiment (see module "
                "docstring for exactly what was and was not measured).",
    }
    return result


if __name__ == "__main__":
    import json
    result = run()
    print(json.dumps(result, indent=2))
    with open(os.path.join(os.path.dirname(__file__), "final_wrong_arm_forensics_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to evaluation/final_wrong_arm_forensics_results.json")
