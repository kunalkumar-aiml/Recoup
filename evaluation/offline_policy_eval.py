"""
Offline Policy Evaluation (OPE).

THE QUESTION: "How do you evaluate a new policy without deploying it?"

We have LOGGED bandit-style data: for each failed event, we know which
arm was historically chosen (by a non-random, biased assignment
process -- see generate_data.py's assign_historical_intervention) and
whether it recovered. We want to estimate how a DIFFERENT policy (e.g.
Recoup's uplift+net-value policy) would have performed, without ever
actually deploying it.

METHOD: Inverse Propensity Scoring (IPS) and Self-Normalized IPS (SNIPS).

    IPS estimate of policy pi's value:
        V_IPS(pi) = (1/N) * sum_i [ I(pi(X_i) == A_i) / p(A_i | X_i) ] * R_i

    where A_i is the historically-logged arm, R_i the observed reward,
    and p(A_i | X_i) the PROPENSITY -- the probability the logging
    policy would have chosen A_i given X_i. We estimate this propensity
    with a multinomial logistic model fit on (X, chosen_intervention)
    from the logged data itself (a standard, defensible approach when
    the true logging propensities aren't directly available).

    SNIPS divides by the sum of the importance weights instead of N,
    which trades a small amount of bias for a large reduction in
    variance -- standard practice, and what we report as the headline
    number, with plain IPS shown alongside for comparison.

WHY THIS MATTERS: it lets us sanity-check the T-learner/bandit policy's
apparent superiority using a DIFFERENT estimation approach than "run it
against the simulator" -- if IPS/SNIPS on real logged data disagrees
sharply with the simulator-based evaluation, that is itself a useful
red flag about overfitting to the simulator's own structure.

WE ALSO cross-check both against the ORACLE counterfactual file
(oracle_potential_outcomes.csv) -- which, because it's synthetic, lets us
compute the *actual* ground-truth policy value directly (no estimator
needed) as a validation of whether IPS/SNIPS are in the right ballpark.
In a real deployment we would NOT have this oracle; here we use it only
to sanity-check the estimator, never to train or make decisions.

CAVEAT stated explicitly: IPS/SNIPS variance grows when propensities are
small (rare historical arm choices in some contexts) -- we clip
propensities at a floor of 0.02 to bound the importance weights, a
standard mitigation, documented rather than hidden.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))
from features import load_events, build_feature_frame, align_columns  # noqa: E402
from temporal_split import temporal_split  # noqa: E402
from uplift import load_models as load_uplift_models, predict_arm_probs, compute_uplift, ARMS  # noqa: E402
from bandit import net_value, FRICTION_COST  # noqa: E402

PROPENSITY_FLOOR = 0.02


def fit_propensity_model(train_df, feature_cols_built):
    """Multinomial logistic regression estimating p(arm | X) from the
    logged (observationally-biased) assignment in the training data."""
    failed = train_df[train_df["failed"] == True].copy()
    X = build_feature_frame(failed)
    y = failed["chosen_intervention"]
    model = LogisticRegression(max_iter=3000)
    model.fit(X, y)
    return model, list(X.columns), list(model.classes_)


def estimate_propensity(model, columns, classes, row_df, arm):
    X = build_feature_frame(row_df)
    X = align_columns(X, columns)
    probs = model.predict_proba(X)[0]
    if arm not in classes:
        return PROPENSITY_FLOOR
    idx = classes.index(arm)
    return max(probs[idx], PROPENSITY_FLOOR)


def evaluate_policy_ips(test_df, arm_models, propensity_model, propensity_cols, propensity_classes,
                          policy_fn, sample_size=500, seed=7):
    """policy_fn(row) -> chosen_arm (or None for no-action/escalate)."""
    rng = np.random.RandomState(seed)
    failed = test_df[test_df["failed"] == True].copy()
    if len(failed) > sample_size:
        failed = failed.sample(n=sample_size, random_state=seed)

    ips_terms = []
    weights = []
    for _, row in failed.iterrows():
        row_df = pd.DataFrame([row])
        policy_arm = policy_fn(row_df, arm_models)
        logged_arm = row["chosen_intervention"]
        reward = (row["recovered_value"] or 0) - FRICTION_COST.get(logged_arm, 0)

        if policy_arm == logged_arm:
            propensity = estimate_propensity(
                propensity_model, propensity_cols, propensity_classes, row_df, logged_arm
            )
            weight = 1.0 / propensity
            ips_terms.append(weight * reward)
            weights.append(weight)

    n = len(failed)
    ips_value = float(np.sum(ips_terms) / n) if n > 0 else 0.0
    snips_value = float(np.sum(ips_terms) / np.sum(weights)) if weights else 0.0
    return {
        "sample_size": n,
        "matched_logged_arm_count": len(ips_terms),
        "ips_estimate": round(ips_value, 2),
        "snips_estimate": round(snips_value, 2),
    }


def oracle_policy_value(test_df, oracle_df, policy_fn, arm_models, sample_size=500, seed=7):
    """Ground-truth policy value using the ORACLE file -- validation only,
    never used by the policy itself."""
    failed = test_df[test_df["failed"] == True].copy()
    if len(failed) > sample_size:
        failed = failed.sample(n=sample_size, random_state=seed)
    merged = failed.merge(oracle_df, on="event_id", how="inner")

    total_reward = 0.0
    n = 0
    for _, row in merged.iterrows():
        row_df = pd.DataFrame([row])
        policy_arm = policy_fn(row_df, arm_models)
        n += 1
        if policy_arm is None:
            continue
        col = f"potential_recovered_{policy_arm}"
        if col not in row:
            continue
        recovered = row[col]
        reward = (row["amount"] if recovered else 0) - FRICTION_COST.get(policy_arm, 0)
        total_reward += reward
    return {"sample_size": n, "oracle_policy_value": round(total_reward / n, 2) if n else 0.0}


# ---- policies to compare ----

def policy_rule_always_retry(row_df, arm_models):
    return "retry_timing"


def policy_uplift_best(row_df, arm_models):
    probs = predict_arm_probs(arm_models, row_df)
    uplift = compute_uplift(probs)
    if not uplift:
        return None
    amount = float(row_df.iloc[0]["amount"])
    nv = net_value(uplift, amount)
    positive = {a: v for a, v in nv.items() if v > 0}
    if not positive:
        return None
    return max(positive, key=positive.get)


if __name__ == "__main__":
    df = load_events()
    _, _, test_df = temporal_split(df)
    oracle_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "oracle_potential_outcomes.csv"))

    train_df, _, _ = temporal_split(df)
    arm_models = load_uplift_models()
    propensity_model, propensity_cols, propensity_classes = fit_propensity_model(train_df, None)

    print("=" * 70)
    print("OFFLINE POLICY EVALUATION (IPS / SNIPS on logged test data)")
    print("=" * 70)
    for name, policy_fn in [
        ("Rule: always retry_timing", policy_rule_always_retry),
        ("Recoup: uplift + net-value", policy_uplift_best),
    ]:
        result = evaluate_policy_ips(
            test_df, arm_models, propensity_model, propensity_cols, propensity_classes, policy_fn
        )
        print(f"\n{name}")
        print(result)

    print("\n" + "=" * 70)
    print("ORACLE VALIDATION (ground truth from counterfactual simulator, sanity-check only)")
    print("=" * 70)
    for name, policy_fn in [
        ("Rule: always retry_timing", policy_rule_always_retry),
        ("Recoup: uplift + net-value", policy_uplift_best),
    ]:
        result = oracle_policy_value(test_df, oracle_df, policy_fn, arm_models)
        print(f"\n{name}")
        print(result)
