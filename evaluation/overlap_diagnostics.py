"""
Propensity / Overlap Diagnostics (red-team round 2, issue #2).

For the T-learner's implied causal claim to be trustworthy in a region
of context-space, the logging policy must have given every arm a
realistic chance of being chosen there (positivity/overlap). This script
estimates P(A=a | X) per arm via a multinomial logistic regression fit
on the logged (biased) assignment data, then reports, per arm:

  - minimum estimated propensity across the test sample
  - effective sample size (Kish's ESS: (sum w)^2 / sum(w^2), w = 1/propensity)
  - % of the test sample below a common-support threshold (0.05)

Where an arm's overlap is thin, the uplift estimate for that arm is
extrapolation, not interpolation -- and should be treated with
correspondingly less confidence. This does NOT fix confounding; it only
tells us WHERE the T-learner's assumptions are weakest.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))
from features import load_events, build_feature_frame  # noqa: E402
from temporal_split import temporal_split  # noqa: E402

COMMON_SUPPORT_THRESHOLD = 0.05


def fit_propensity_model(train_df):
    failed = train_df[train_df["failed"] == True].copy()
    failed = failed[failed["chosen_intervention"] != ""]
    X = build_feature_frame(failed)
    y = failed["chosen_intervention"]
    model = LogisticRegression(max_iter=3000)
    model.fit(X, y)
    return model, list(X.columns), list(model.classes_)


def run():
    df = load_events()
    train_df, _, test_df = temporal_split(df)
    model, columns, classes = fit_propensity_model(train_df)

    test_failed = test_df[test_df["failed"] == True].copy()
    test_failed = test_failed[test_failed["chosen_intervention"] != ""]
    X_test = build_feature_frame(test_failed)
    for col in columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[columns]

    probs = model.predict_proba(X_test)  # shape (n, n_classes)

    print("=" * 78)
    print("PROPENSITY / OVERLAP DIAGNOSTICS  (test split, n =", len(test_failed), ")")
    print("=" * 78)
    print(f"{'Arm':<22}{'Min P(A|X)':>12}{'Effective N':>14}{'% below 0.05':>16}")

    report = {}
    for i, arm in enumerate(classes):
        arm_probs = probs[:, i]
        arm_probs_clipped = np.clip(arm_probs, 1e-4, 1.0)
        weights = 1.0 / arm_probs_clipped
        ess = (weights.sum() ** 2) / (weights ** 2).sum()
        pct_below = float((arm_probs < COMMON_SUPPORT_THRESHOLD).mean() * 100)
        report[arm] = {
            "min_propensity": round(float(arm_probs.min()), 4),
            "effective_sample_size": round(float(ess), 1),
            "pct_below_common_support": round(pct_below, 1),
        }
        print(f"{arm:<22}{report[arm]['min_propensity']:>12}{report[arm]['effective_sample_size']:>14}"
              f"{report[arm]['pct_below_common_support']:>15}%")

    print("\nInterpretation: an arm with a low minimum propensity and a large "
          "% below common support means the T-learner's uplift estimate for "
          "that arm, in the thin-support region, is closer to extrapolation "
          "than a directly-supported causal estimate. This is exactly what "
          "we'd expect given generate_data.py's deliberately merchant-biased "
          "logging policy -- reported here rather than assumed away.")

    import json
    out_path = os.path.join(os.path.dirname(__file__), "overlap_diagnostics_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out_path}")
    return report


if __name__ == "__main__":
    run()
