"""
Action Ranking Analysis (round-5, Mission 1 — the highest-priority item).

Round 4's regret decomposition established WHERE regret comes from (90%
wrong-arm). This script goes one level deeper: HOW WRONG is the ranking,
using proper ranking metrics, not just top-1/top-2 hit rate.

For every event, compare:
  PREDICTED ranking of the 5 actionable arms, by net_value(uplift, amount)
  TRUE ranking of the same 5 arms, by the oracle's actual net reward

Metrics computed:
  - Top-1 / Top-2 / Top-3 accuracy (top-3 is new this round)
  - MRR (Mean Reciprocal Rank of the true-best arm in the predicted ranking)
  - NDCG@3 (Normalized Discounted Cumulative Gain, using true reward as relevance)
  - Spearman rank correlation (predicted vs true full ranking)
  - Kendall's tau (same, different sensitivity to swaps)
  - Value-weighted regret (regret weighted by transaction amount, so a
    ₹10,000 misranking counts more than a ₹100 one -- Mission 2)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))
from features import load_events  # noqa: E402
from temporal_split import temporal_split  # noqa: E402
from uplift import load_models, predict_arm_probs, compute_uplift  # noqa: E402
from bandit import net_value  # noqa: E402
from protocol import (  # noqa: E402
    score_event, regret, oracle_ranked_actions, historical_escalation_credit_rate,
    ALL_ACTIONABLE_ARMS, net_revenue_oracle,
)

SAMPLE_SIZE = 500


def predicted_ranking(arm_models, row_df, amount):
    probs = predict_arm_probs(arm_models, row_df)
    uplift = compute_uplift(probs)
    if not uplift:
        return [], {}
    nv = net_value(uplift, amount)
    ranked = sorted(nv.keys(), key=lambda a: -nv[a])
    return ranked, nv


def ndcg_at_k(true_ranked_values, predicted_order, k=3):
    """true_ranked_values: dict arm -> true reward. predicted_order: list
    of arms in the system's predicted-best-first order."""
    def dcg(order):
        return sum(
            true_ranked_values.get(a, 0) / np.log2(i + 2)
            for i, a in enumerate(order[:k])
        )
    ideal_order = sorted(true_ranked_values, key=lambda a: -true_ranked_values[a])
    ideal = dcg(ideal_order)
    actual = dcg(predicted_order)
    return actual / ideal if ideal > 0 else 0.0


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

    top1 = top2 = top3 = 0
    mrr_scores, ndcg_scores, spearman_scores, kendall_scores = [], [], [], []
    value_weighted_regrets, plain_regrets = [], []
    error_categories = {"HIGH_VALUE_MISRANK": 0, "LOW_CONFIDENCE_MISRANK": 0, "CORRECT": 0, "OTHER_MISRANK": 0}
    n_valid = 0

    for _, row in merged.iterrows():
        row_df = pd.DataFrame([row])
        amount = row["amount"]

        pred_order, pred_values = predicted_ranking(arm_models, row_df, amount)
        if not pred_order:
            continue
        n_valid += 1

        true_values = {a: net_revenue_oracle(row, a) for a in ALL_ACTIONABLE_ARMS
                        if f"potential_recovered_{a}" in row and not pd.isna(row[f"potential_recovered_{a}"])}
        true_order = sorted(true_values, key=lambda a: -true_values[a])
        if not true_order:
            continue

        best_true = true_order[0]
        if pred_order[0] == best_true:
            top1 += 1
            error_categories["CORRECT"] += 1
        elif best_true in pred_order[:2]:
            top2 += 1
        if best_true in pred_order[:3]:
            top3 += 1

        # MRR: reciprocal rank of the TRUE best arm within the PREDICTED order
        if best_true in pred_order:
            rank = pred_order.index(best_true) + 1
            mrr_scores.append(1.0 / rank)
        else:
            mrr_scores.append(0.0)

        ndcg_scores.append(ndcg_at_k(true_values, pred_order, k=3))

        # rank correlation over the arms both rankings share
        common = [a for a in pred_order if a in true_values]
        if len(common) >= 3:
            pred_ranks = [pred_order.index(a) for a in common]
            true_ranks = [true_order.index(a) for a in common]
            rho, _ = spearmanr(pred_ranks, true_ranks)
            tau, _ = kendalltau(pred_ranks, true_ranks)
            if not np.isnan(rho):
                spearman_scores.append(rho)
            if not np.isnan(tau):
                kendall_scores.append(tau)

        # value-weighted regret (Mission 2)
        chosen = pred_order[0]
        chosen_true_value = true_values.get(chosen, 0)
        best_true_value = true_values[best_true]
        plain_r = best_true_value - chosen_true_value
        plain_regrets.append(plain_r)
        value_weighted_regrets.append(plain_r)  # already amount-scaled since true_values are in rupees

        if pred_order[0] != best_true:
            if amount > 5000:
                error_categories["HIGH_VALUE_MISRANK"] += 1
            else:
                error_categories["OTHER_MISRANK"] += 1

    result = {
        "n_events": n_valid,
        "top1_accuracy": round(top1 / n_valid, 4),
        "top2_accuracy": round((top1 + top2) / n_valid, 4),
        "top3_accuracy": round(top3 / n_valid, 4),
        "mrr": round(float(np.mean(mrr_scores)), 4),
        "ndcg_at_3": round(float(np.mean(ndcg_scores)), 4),
        "spearman_rank_correlation": round(float(np.mean(spearman_scores)), 4) if spearman_scores else None,
        "kendall_tau": round(float(np.mean(kendall_scores)), 4) if kendall_scores else None,
        "mean_regret": round(float(np.mean(plain_regrets)), 2),
        "total_value_weighted_regret": round(float(np.sum(value_weighted_regrets)), 2),
        "error_breakdown": error_categories,
        "note": ("Spearman/Kendall correlations near 0 would mean the predicted ranking carries "
                 "almost no information about the true ranking -- worth checking directly rather than "
                 "inferring from top-1 accuracy alone, since a low top-1 rate could still coexist with "
                 "a genuinely informative (but imperfect) ranking."),
    }
    return result


if __name__ == "__main__":
    import json
    result = run()
    print(json.dumps(result, indent=2))
    with open(os.path.join(os.path.dirname(__file__), "action_ranking_analysis_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to evaluation/action_ranking_analysis_results.json")
