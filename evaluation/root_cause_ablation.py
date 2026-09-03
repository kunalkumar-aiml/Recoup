"""
Root-cause posterior ablation (red-team round 2, issue #9).

The root-cause model predicts P(decline_code | context). But decline_code
is already a directly-observed field on every failed event -- Razorpay's
gateway gives it to us. So: does feeding the *posterior* (soft,
uncertain) as an engineered input to the per-arm uplift models improve
policy value versus just using the *raw observed* decline_code (which
the per-arm models already do via one-hot encoding in features.py)?

WITHOUT: per-arm models trained on the standard feature set (decline_code
         one-hot included directly, as they already are today)
WITH:    per-arm models trained on the standard feature set PLUS the
         root-cause posterior's max-class probability and entropy as two
         extra numeric features (a cheap, realistic way to inject
         "uncertainty about the cause" without leaking anything)

Both evaluated on the same TEST split against the oracle, same metric
(mean net reward) used everywhere else in this repo.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))
from features import load_events, build_feature_frame, align_columns  # noqa: E402
from temporal_split import temporal_split  # noqa: E402
from uplift import ARMS, MIN_ARM_SUPPORT, _rows_for_arm, _temporal_holdout_split  # noqa: E402
from bandit import net_value, FRICTION_COST  # noqa: E402
from policy import check_action  # noqa: E402

SAMPLE_SIZE = 500


def train_root_cause_for_ablation(train_df):
    failed = train_df[train_df["failed"] == True].copy()
    cat_cols = ["merchant_category", "customer_value_tier", "event_type", "payment_method"]
    num_cols = ["amount", "customer_retry_propensity_observed", "merchant_baseline_fail_rate",
                "prior_failure_count", "minutes_since_last_failure", "prior_recovery_count",
                "prior_recovery_rate", "recent_method_switch_count"]
    X = pd.get_dummies(failed[cat_cols + num_cols], columns=cat_cols)
    y = failed["decline_code"]
    clf = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3)
    clf.fit(X, y)
    return clf, list(X.columns), list(clf.classes_)


def add_posterior_features(df, rc_model, rc_columns, rc_classes):
    cat_cols = ["merchant_category", "customer_value_tier", "event_type", "payment_method"]
    num_cols = ["amount", "customer_retry_propensity_observed", "merchant_baseline_fail_rate",
                "prior_failure_count", "minutes_since_last_failure", "prior_recovery_count",
                "prior_recovery_rate", "recent_method_switch_count"]
    X = pd.get_dummies(df[cat_cols + num_cols], columns=cat_cols)
    for col in rc_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[rc_columns]
    proba = rc_model.predict_proba(X)
    df = df.copy()
    df["root_cause_max_prob"] = proba.max(axis=1)
    df["root_cause_entropy"] = entropy(proba.T)
    return df


def train_arm_models_with_extra_features(train_df, extra_cols):
    """Same logic as uplift.py::train_per_arm_models but optionally
    includes extra numeric columns (the root-cause posterior features)."""
    models = {}
    for arm in ARMS:
        subset = _rows_for_arm(train_df, arm)
        if len(subset) < MIN_ARM_SUPPORT or subset["recovered"].nunique() < 2:
            continue
        arm_train, arm_holdout = _temporal_holdout_split(subset)
        if len(arm_holdout) < 10 or arm_train["recovered"].nunique() < 2:
            continue
        X_train = build_feature_frame(arm_train)
        for c in extra_cols:
            X_train[c] = arm_train[c].values
        y_train = arm_train["recovered"].astype(int)
        base = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3)
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        model.fit(X_train, y_train)
        models[arm] = {"model": model, "columns": list(X_train.columns)}
    return models


def predict_with_extra(models, row_df, extra_cols):
    probs = {}
    for arm, bundle in models.items():
        X = build_feature_frame(row_df)
        for c in extra_cols:
            X[c] = row_df[c].values
        X = align_columns(X, bundle["columns"])
        probs[arm] = float(bundle["model"].predict_proba(X)[0][1])
    return probs


def oracle_reward(row, arm):
    if arm is None:
        return 0.0
    col = f"potential_recovered_{arm}"
    if col not in row or pd.isna(row[col]):
        return 0.0
    return (row["amount"] if row[col] else 0) - FRICTION_COST.get(arm, 0)


def score_policy(test_merged, models, extra_cols):
    total = 0.0
    n = 0
    for _, row in test_merged.iterrows():
        n += 1
        row_df = pd.DataFrame([row])
        probs = predict_with_extra(models, row_df, extra_cols)
        if "no_action" not in probs:
            continue
        control = probs["no_action"]
        uplift = {a: p - control for a, p in probs.items() if a != "no_action"}
        nv = net_value(uplift, row["amount"])
        positive = {a: v for a, v in nv.items() if v > 0}
        if not positive:
            continue
        candidate = max(positive, key=positive.get)
        policy_result = check_action(intervention=candidate, amount=row["amount"], retry_count=0, nudges_today=0)
        if policy_result["approved"] and not policy_result["requires_human"]:
            total += oracle_reward(row, candidate)
    return total / n if n else 0.0, n


def run():
    df = load_events()
    train_df, _, test_df = temporal_split(df)

    oracle_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "oracle_potential_outcomes.csv"))
    failed_test = test_df[test_df["failed"] == True].copy()
    merged = failed_test.merge(oracle_df, on="event_id", how="inner")
    if len(merged) > SAMPLE_SIZE:
        merged = merged.sample(n=SAMPLE_SIZE, random_state=7)

    print("Training root-cause model for the posterior features...")
    rc_model, rc_columns, rc_classes = train_root_cause_for_ablation(train_df)

    print("WITHOUT root-cause posterior: training arm models on standard features only...")
    models_without = train_arm_models_with_extra_features(train_df, extra_cols=[])
    reward_without, n_without = score_policy(merged, models_without, extra_cols=[])

    print("WITH root-cause posterior: adding max-prob + entropy features...")
    train_df_aug = add_posterior_features(train_df, rc_model, rc_columns, rc_classes)
    merged_aug = add_posterior_features(merged, rc_model, rc_columns, rc_classes)
    models_with = train_arm_models_with_extra_features(train_df_aug, extra_cols=["root_cause_max_prob", "root_cause_entropy"])
    reward_with, n_with = score_policy(merged_aug, models_with, extra_cols=["root_cause_max_prob", "root_cause_entropy"])

    delta = reward_with - reward_without
    delta_pct = 100 * delta / max(abs(reward_without), 1)

    print("\n" + "=" * 70)
    print("ROOT-CAUSE POSTERIOR ABLATION")
    print("=" * 70)
    print(f"WITHOUT root-cause posterior features: mean net reward = {reward_without:.2f}  (n={n_without})")
    print(f"WITH    root-cause posterior features: mean net reward = {reward_with:.2f}  (n={n_with})")
    print(f"Delta: {delta:+.2f} ({delta_pct:+.1f}%)")
    if abs(delta_pct) < 3:
        verdict = ("NO MEANINGFUL IMPROVEMENT. The root-cause posterior does not earn its "
                    "place as a decision-relevant feature -- decline_code is already directly "
                    "observed and one-hot encoded in the base feature set, so a soft posterior "
                    "over the SAME variable adds negligible new information. Recommendation: "
                    "keep the root-cause model ONLY as an informational/diagnostic trace element "
                    "in the API response (Option C from the red-team request, partially), NOT as "
                    "a claimed decision-improving input.")
    elif delta_pct > 0:
        verdict = "Modest improvement -- root-cause posterior features may carry a small amount of usable signal beyond the raw decline_code."
    else:
        verdict = "Root-cause posterior features HURT performance -- do not add them."
    print(f"\nVerdict: {verdict}")

    import json
    result = {
        "reward_without_posterior": round(reward_without, 2),
        "reward_with_posterior": round(reward_with, 2),
        "delta": round(delta, 2),
        "delta_pct": round(delta_pct, 1),
        "verdict": verdict,
    }
    with open(os.path.join(os.path.dirname(__file__), "root_cause_ablation_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    run()
