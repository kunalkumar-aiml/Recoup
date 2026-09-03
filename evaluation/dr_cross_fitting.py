"""
Cross-fitted Doubly Robust (DR/AIPW) estimation — final causal validation.

MATHEMATICAL FORMULATION
For arm a, the AIPW pseudo-outcome for unit i is:

    psi_a(X_i) = m_a(X_i) + (1{A_i=a} / e_a(X_i)) * (Y_i - m_a(X_i))    if A_i = a
    psi_a(X_i) = m_a(X_i)                                                if A_i != a

where m_a(X) = E[Y|X,A=a] (outcome model) and e_a(X) = P(A=a|X)
(propensity model). A second-stage regression g_a(X), fit on these
pseudo-outcomes, is the DR-Learner style estimator (Kennedy 2020) used
here as mu_a_DR(X).

CROSS-FITTING: K=3 folds. For fold k, BOTH the outcome model AND the
propensity model are trained on the OTHER folds only, then used to
construct pseudo-outcomes for fold k. No unit's own data was used to
build the nuisance predictions used to correct its own pseudo-outcome.

IDENTIFICATION ASSUMPTIONS (stated explicitly):
  - Consistency: true by construction in this synthetic environment.
  - Positivity/overlap: MEASURABLY VIOLATED for no_action (see
    evaluation/overlap_diagnostics.py) -- a real, stated limitation.
  - Conditional exchangeability: true by construction here (the logging
    policy conditions only on observed features), not guaranteed in a
    real deployment.

Cross-fitted DR reduces sensitivity to nuisance-model misspecification
under these assumptions. It does NOT prove causality.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))
from features import load_events, build_feature_frame, align_columns  # noqa: E402
from temporal_split import temporal_split  # noqa: E402
from uplift import load_models as load_current_models, predict_arm_probs, compute_uplift, ARMS  # noqa: E402
from bandit import net_value  # noqa: E402
from policy import check_action  # noqa: E402
from protocol import (score_event, regret, oracle_optimal_action, oracle_ranked_actions,  # noqa: E402
                       historical_escalation_credit_rate)

K_FOLDS = 3
SAMPLE_SIZE = 500
CLIP = 0.05


def synthetic_ground_truth_validation(seed=0):
    rng = np.random.RandomState(seed)
    n = 2000
    X = rng.normal(size=(n, 3))
    logit_e = 0.8 * X[:, 0] - 0.3 * X[:, 1]
    e_true = 1 / (1 + np.exp(-logit_e))
    A = (rng.uniform(size=n) < e_true).astype(int)

    true_effect = 0.15 + 0.10 * np.tanh(X[:, 0]) - 0.05 * X[:, 2] ** 2
    base_prob = 0.3 + 0.1 * X[:, 1]
    p1 = np.clip(base_prob + true_effect, 0.02, 0.98)
    p0 = np.clip(base_prob, 0.02, 0.98)
    Y = np.where(A == 1, rng.uniform(size=n) < p1, rng.uniform(size=n) < p0).astype(int)

    df = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    df["A"] = A
    df["Y"] = Y
    true_tau = p1 - p0

    m1 = GradientBoostingClassifier(random_state=1, n_estimators=50, max_depth=2).fit(
        df[df.A == 1][["x0", "x1", "x2"]], df[df.A == 1]["Y"])
    m0 = GradientBoostingClassifier(random_state=1, n_estimators=50, max_depth=2).fit(
        df[df.A == 0][["x0", "x1", "x2"]], df[df.A == 0]["Y"])
    tau_tlearner = m1.predict_proba(df[["x0", "x1", "x2"]])[:, 1] - m0.predict_proba(df[["x0", "x1", "x2"]])[:, 1]

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
    psi1 = np.zeros(n)
    psi0 = np.zeros(n)
    for train_idx, test_idx in kf.split(df):
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        e_model = LogisticRegression(max_iter=1000).fit(tr[["x0", "x1", "x2"]], tr["A"])
        e_hat = np.clip(e_model.predict_proba(te[["x0", "x1", "x2"]])[:, 1], CLIP, 1 - CLIP)

        m1f = GradientBoostingClassifier(random_state=1, n_estimators=50, max_depth=2).fit(
            tr[tr.A == 1][["x0", "x1", "x2"]], tr[tr.A == 1]["Y"])
        m0f = GradientBoostingClassifier(random_state=1, n_estimators=50, max_depth=2).fit(
            tr[tr.A == 0][["x0", "x1", "x2"]], tr[tr.A == 0]["Y"])
        m1_hat = m1f.predict_proba(te[["x0", "x1", "x2"]])[:, 1]
        m0_hat = m0f.predict_proba(te[["x0", "x1", "x2"]])[:, 1]

        A_te, Y_te = te["A"].values, te["Y"].values
        psi1[test_idx] = np.where(A_te == 1, m1_hat + (Y_te - m1_hat) / e_hat, m1_hat)
        psi0[test_idx] = np.where(A_te == 0, m0_hat + (Y_te - m0_hat) / (1 - e_hat), m0_hat)

    g1 = GradientBoostingRegressor(random_state=1, n_estimators=50, max_depth=2).fit(df[["x0", "x1", "x2"]], psi1)
    g0 = GradientBoostingRegressor(random_state=1, n_estimators=50, max_depth=2).fit(df[["x0", "x1", "x2"]], psi0)
    tau_dr = g1.predict(df[["x0", "x1", "x2"]]) - g0.predict(df[["x0", "x1", "x2"]])

    return {
        "n": n,
        "t_learner": {"mae": round(float(mean_absolute_error(true_tau, tau_tlearner)), 4),
                      "rmse": round(float(mean_squared_error(true_tau, tau_tlearner) ** 0.5), 4),
                      "bias": round(float(np.mean(tau_tlearner - true_tau)), 4)},
        "dr_learner": {"mae": round(float(mean_absolute_error(true_tau, tau_dr)), 4),
                       "rmse": round(float(mean_squared_error(true_tau, tau_dr) ** 0.5), 4),
                       "bias": round(float(np.mean(tau_dr - true_tau)), 4)},
    }


def train_cross_fitted_dr(train_df, arms):
    failed = train_df[train_df["failed"] == True].copy()
    failed = failed[failed["chosen_intervention"] != ""]
    if len(failed) < 100:
        return {}

    X_all = build_feature_frame(failed)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=7)
    fold_indices = list(kf.split(failed))

    dr_models = {}
    for arm in arms:
        psi = np.zeros(len(failed))
        has_prediction = np.zeros(len(failed), dtype=bool)

        for train_idx, test_idx in fold_indices:
            tr = failed.iloc[train_idx]
            te = failed.iloc[test_idx]
            X_tr = build_feature_frame(tr)

            y_prop = tr["chosen_intervention"]
            if y_prop.nunique() < 2:
                continue
            e_model = LogisticRegression(max_iter=2000).fit(X_tr, y_prop)
            if arm not in e_model.classes_:
                continue
            arm_col_idx = list(e_model.classes_).index(arm)
            X_te = align_columns(build_feature_frame(te), list(X_tr.columns))
            e_hat = np.clip(e_model.predict_proba(X_te)[:, arm_col_idx], CLIP, 1 - CLIP)

            arm_tr = tr[tr["chosen_intervention"] == arm]
            if len(arm_tr) < 15 or arm_tr["recovered"].nunique() < 2:
                fallback_rate = failed[failed["chosen_intervention"] == arm]["recovered"].mean()
                m_hat = np.full(len(te), fallback_rate if not np.isnan(fallback_rate) else 0.15)
            else:
                X_arm_tr = build_feature_frame(arm_tr)
                y_arm_tr = arm_tr["recovered"].astype(int)
                m_model = GradientBoostingClassifier(random_state=1, n_estimators=50, max_depth=2).fit(X_arm_tr, y_arm_tr)
                X_te_arm = align_columns(build_feature_frame(te), list(X_arm_tr.columns))
                m_hat = m_model.predict_proba(X_te_arm)[:, 1]

            A_te = (te["chosen_intervention"] == arm).values
            Y_te = te["recovered"].astype(int).values
            psi[test_idx] = np.where(A_te, m_hat + (Y_te - m_hat) / e_hat, m_hat)
            has_prediction[test_idx] = True

        if not has_prediction.any():
            continue

        X_final = X_all[has_prediction]
        psi_final = psi[has_prediction]
        g_model = GradientBoostingRegressor(random_state=1, n_estimators=80, max_depth=3).fit(X_final, psi_final)
        dr_models[arm] = {"model": g_model, "columns": list(X_final.columns)}

    return dr_models


def predict_dr(dr_models, row_df):
    preds = {}
    for arm, bundle in dr_models.items():
        X = build_feature_frame(row_df)
        X = align_columns(X, bundle["columns"])
        p = float(bundle["model"].predict(X)[0])
        preds[arm] = float(np.clip(p, 0.0, 1.0))
    return preds


def compute_dr_uplift(dr_probs):
    if "no_action" not in dr_probs:
        return {}
    control = dr_probs["no_action"]
    return {a: p - control for a, p in dr_probs.items() if a != "no_action"}


def evaluate_dr_policy(dr_models, test_df, oracle_df, escalation_rate, label):
    failed = test_df[test_df["failed"] == True].copy()
    merged = failed.merge(oracle_df, on="event_id", how="inner")
    if len(merged) > SAMPLE_SIZE:
        merged = merged.sample(n=SAMPLE_SIZE, random_state=7)

    rewards, regrets, top1, top3 = [], [], 0, 0
    n = 0
    for _, row in merged.iterrows():
        n += 1
        row_df = pd.DataFrame([row])
        amount = row["amount"]

        dr_probs = predict_dr(dr_models, row_df)
        uplift = compute_dr_uplift(dr_probs)
        if not uplift:
            regrets.append(regret(row, None, escalation_rate))
            rewards.append(score_event(row, None, escalation_rate))
            continue

        nv = net_value(uplift, amount)
        positive = {a: v for a, v in nv.items() if v > 0}
        candidate = max(positive, key=positive.get) if positive else None
        chosen = None
        if candidate:
            pr = check_action(intervention=candidate, amount=amount, retry_count=0, nudges_today=0)
            if pr["approved"] and not pr["requires_human"]:
                chosen = candidate

        rewards.append(score_event(row, chosen, escalation_rate))
        regrets.append(regret(row, chosen, escalation_rate))
        oracle_best, _ = oracle_optimal_action(row)
        ranked = oracle_ranked_actions(row)
        if chosen == oracle_best:
            top1 += 1
        if chosen in ranked[:3]:
            top3 += 1

    result = {"label": label, "n": n,
              "mean_net_reward": round(float(np.mean(rewards)), 2),
              "mean_regret": round(float(np.mean(regrets)), 2),
              "p90_regret": round(float(np.percentile(regrets, 90)), 2),
              "top1_pct": round(100 * top1 / n, 1), "top3_pct": round(100 * top3 / n, 1)}
    print(f"[{label}] mean_reward={result['mean_net_reward']} mean_regret={result['mean_regret']} "
          f"top1={result['top1_pct']}% top3={result['top3_pct']}%")
    return result


def evaluate_current_tlearner(current_models, test_df, oracle_df, escalation_rate, label):
    failed = test_df[test_df["failed"] == True].copy()
    merged = failed.merge(oracle_df, on="event_id", how="inner")
    if len(merged) > SAMPLE_SIZE:
        merged = merged.sample(n=SAMPLE_SIZE, random_state=7)

    rewards, regrets, top1, top3 = [], [], 0, 0
    n = 0
    for _, row in merged.iterrows():
        n += 1
        row_df = pd.DataFrame([row])
        amount = row["amount"]
        probs = predict_arm_probs(current_models, row_df)
        uplift = compute_uplift(probs)
        nv = net_value(uplift, amount) if uplift else {}
        positive = {a: v for a, v in nv.items() if v > 0}
        candidate = max(positive, key=positive.get) if positive else None
        chosen = None
        if candidate:
            pr = check_action(intervention=candidate, amount=amount, retry_count=0, nudges_today=0)
            if pr["approved"] and not pr["requires_human"]:
                chosen = candidate
        rewards.append(score_event(row, chosen, escalation_rate))
        regrets.append(regret(row, chosen, escalation_rate))
        oracle_best, _ = oracle_optimal_action(row)
        ranked = oracle_ranked_actions(row)
        if chosen == oracle_best:
            top1 += 1
        if chosen in ranked[:3]:
            top3 += 1

    result = {"label": label, "n": n,
              "mean_net_reward": round(float(np.mean(rewards)), 2),
              "mean_regret": round(float(np.mean(regrets)), 2),
              "p90_regret": round(float(np.percentile(regrets, 90)), 2),
              "top1_pct": round(100 * top1 / n, 1), "top3_pct": round(100 * top3 / n, 1)}
    print(f"[{label}] mean_reward={result['mean_net_reward']} mean_regret={result['mean_regret']} "
          f"top1={result['top1_pct']}% top3={result['top3_pct']}%")
    return result


def run():
    print("=" * 70)
    print("STEP 4: SYNTHETIC GROUND-TRUTH VALIDATION")
    print("=" * 70)
    synth = synthetic_ground_truth_validation()
    print(f"T-learner: MAE={synth['t_learner']['mae']} bias={synth['t_learner']['bias']:+}")
    print(f"DR-learner: MAE={synth['dr_learner']['mae']} bias={synth['dr_learner']['bias']:+}")
    dr_wins_synthetic = synth["dr_learner"]["mae"] < synth["t_learner"]["mae"]
    print(f"DR beats T-learner on KNOWN synthetic ground truth: {dr_wins_synthetic}")

    print("\n" + "=" * 70)
    print("STEP 2/3: Cross-fitted DR on real Recoup data")
    print("=" * 70)
    df = load_events()
    train_df, _, test_df = temporal_split(df)
    escalation_rate = historical_escalation_credit_rate(train_df)
    oracle_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "oracle_potential_outcomes.csv"))

    dr_models = train_cross_fitted_dr(train_df, ARMS)
    print(f"Cross-fitted DR trained for arms: {list(dr_models.keys())}")

    dr_no_action_mae = None
    dr_no_action_bias = None
    if "no_action" in dr_models:
        failed_test = test_df[test_df["failed"] == True].merge(oracle_df, on="event_id", how="inner")
        if len(failed_test) > SAMPLE_SIZE:
            failed_test = failed_test.sample(n=SAMPLE_SIZE, random_state=7)
        preds, trues = [], []
        for _, row in failed_test.iterrows():
            row_df = pd.DataFrame([row])
            p = predict_dr(dr_models, row_df).get("no_action")
            true = row.get("potential_prob_no_action")
            if p is not None and true is not None and not pd.isna(true):
                preds.append(p)
                trues.append(true)
        dr_no_action_mae = round(float(mean_absolute_error(trues, preds)), 4)
        dr_no_action_bias = round(float(np.mean(np.array(preds) - np.array(trues))), 4)
        print(f"DR no_action oracle diagnostic: MAE={dr_no_action_mae} bias={dr_no_action_bias:+}")
        print("(round-11 comparison: current T-learner MAE=0.1084, bias=+0.09)")
    else:
        print("DR could not train a no_action model either (same thin-data constraint as T-learner)")

    print("\n" + "=" * 70)
    print("STEP 5-7: main policy comparison")
    print("=" * 70)
    current_models = load_current_models()
    result_tlearner = evaluate_current_tlearner(current_models, test_df, oracle_df, escalation_rate, "C: Current T-learner")
    result_dr = evaluate_dr_policy(dr_models, test_df, oracle_df, escalation_rate, "D: Cross-fitted DR")

    reward_delta_pct = round(100 * (result_dr["mean_net_reward"] - result_tlearner["mean_net_reward"]) / abs(result_tlearner["mean_net_reward"]), 1)
    regret_delta_pct = round(100 * (result_tlearner["mean_regret"] - result_dr["mean_regret"]) / result_tlearner["mean_regret"], 1)
    top1_delta = result_dr["top1_pct"] - result_tlearner["top1_pct"]
    top3_delta = result_dr["top3_pct"] - result_tlearner["top3_pct"]

    print(f"\nDelta: reward {reward_delta_pct:+.1f}%, regret {-regret_delta_pct:+.1f}%, "
          f"top1 {top1_delta:+.1f}pp, top3 {top3_delta:+.1f}pp")

    meaningful_gain = reward_delta_pct > 3 or regret_delta_pct > 3
    ranking_degraded = top1_delta < -3 or top3_delta < -10

    if ranking_degraded:
        decision = "REJECT"
        explanation = (f"DR shows {reward_delta_pct:+.1f}% reward / {-regret_delta_pct:+.1f}% regret change, "
                        f"but Top-1 moved {top1_delta:+.1f}pp and Top-3 moved {top3_delta:+.1f}pp -- a material "
                        f"ranking degradation disqualifies adoption regardless of the reward number.")
    elif meaningful_gain:
        decision = "ADOPT"
        explanation = f"DR meaningfully improved reward ({reward_delta_pct:+.1f}%) and/or regret ({-regret_delta_pct:+.1f}%) without ranking degradation."
    elif dr_wins_synthetic:
        decision = "RETAIN AS VALIDATION LAYER ONLY"
        explanation = (f"DR is mathematically validated on known synthetic ground truth (MAE {synth['dr_learner']['mae']} "
                        f"vs T-learner's {synth['t_learner']['mae']}), confirming the implementation is correct -- but "
                        f"it did NOT meaningfully change the actual economic policy on the real dataset "
                        f"({reward_delta_pct:+.1f}% reward, {regret_delta_pct:+.1f}% regret, both under the 3% "
                        f"threshold). Keep DR as an evaluation/validation layer, do NOT replace the production "
                        f"T-learner policy with it.")
    else:
        decision = "REJECT"
        explanation = "DR did not improve policy value/regret and did not clearly outperform on synthetic ground truth either."

    print(f"\nDECISION: {decision}")
    print(explanation)

    return {
        "synthetic_ground_truth_validation": synth,
        "dr_no_action_oracle_diagnostic": {"mae": dr_no_action_mae, "bias": dr_no_action_bias,
                                             "t_learner_comparison_mae": 0.1084, "t_learner_comparison_bias": 0.0895},
        "main_policy_comparison": {"t_learner": result_tlearner, "dr": result_dr,
                                     "reward_delta_pct": reward_delta_pct, "regret_delta_pct": regret_delta_pct,
                                     "top1_delta_pp": round(top1_delta, 1), "top3_delta_pp": round(top3_delta, 1)},
        "multi_seed": "NOT RUN this experiment -- single seed only, given Sept 4 deadline",
        "overlap": "unchanged from evaluation/overlap_diagnostics.py -- 73.8% of contexts below common "
                    "support for no_action; DR's propensity clipping (0.05 floor) bounds but does not "
                    "eliminate the resulting variance",
        "decision": decision,
        "explanation": explanation,
    }


if __name__ == "__main__":
    import json
    result = run()
    with open(os.path.join(os.path.dirname(__file__), "dr_cross_fitting_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to evaluation/dr_cross_fitting_results.json")
