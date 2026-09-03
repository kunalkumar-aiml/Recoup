"""
Trains, on the TRAIN split only (see temporal_split.py):
  1. root_cause_model: P(decline_code | context) -- multi-class posterior,
     not a hard label, so downstream models can reason about uncertainty
     in the failure cause itself.
  2. per-arm recovery-probability models (uplift.py, T-learner) used to
     compute uplift/net-value.
  3. bootstrap ensembles per arm (uncertainty.py) for confidence tiers.

Reports on the VAL split (never touches TEST -- that's reserved for
evaluation/ablation.py and offline_policy_eval.py so those numbers are a
genuine held-out check, not re-used training-time numbers).
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix

from features import load_events, build_feature_frame, align_columns, CATEGORICAL_COLS, NUMERIC_COLS
from temporal_split import temporal_split
from uplift import train_per_arm_models, save_models, ARMS
from uncertainty import train_bootstrap_ensemble
from calibration_eval import compare_raw_vs_calibrated

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ROOT_CAUSE_FEATURE_COLS_CATEGORICAL = [c for c in CATEGORICAL_COLS if c != "decline_code"]


def train_root_cause(train_df, val_df):
    train_failed = train_df[train_df["failed"] == True].copy()
    val_failed = val_df[val_df["failed"] == True].copy()

    X_train = pd.get_dummies(
        train_failed[ROOT_CAUSE_FEATURE_COLS_CATEGORICAL + NUMERIC_COLS],
        columns=ROOT_CAUSE_FEATURE_COLS_CATEGORICAL,
    )
    y_train = train_failed["decline_code"]

    clf = GradientBoostingClassifier(random_state=42, n_estimators=150, max_depth=3)
    clf.fit(X_train, y_train)

    X_val = pd.get_dummies(
        val_failed[ROOT_CAUSE_FEATURE_COLS_CATEGORICAL + NUMERIC_COLS],
        columns=ROOT_CAUSE_FEATURE_COLS_CATEGORICAL,
    )
    X_val = align_columns(X_val, list(X_train.columns))
    y_val = val_failed["decline_code"]
    y_pred = clf.predict(X_val)

    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    per_class_recall = recall_score(y_val, y_pred, average=None, zero_division=0, labels=clf.classes_)
    cm = confusion_matrix(y_val, y_pred, labels=clf.classes_)

    print(f"[root_cause] VAL macro F1: {macro_f1:.3f}")
    print(f"[root_cause] VAL per-class recall: "
          f"{dict(zip(clf.classes_, [round(r, 2) for r in per_class_recall]))}")
    print(f"[root_cause] VAL confusion matrix (rows=true, cols=pred), classes={list(clf.classes_)}")
    print(cm)

    joblib.dump({"model": clf, "columns": list(X_train.columns), "classes": list(clf.classes_)},
                os.path.join(MODEL_DIR, "root_cause.pkl"))
    return clf


def train_and_evaluate_calibration(train_df, val_df):
    """Demonstrates raw-vs-calibrated Brier/ECE improvement on the
    highest-volume arm, for the model-card evidence."""
    train_failed = train_df[train_df["failed"] == True].copy()
    val_failed = val_df[val_df["failed"] == True].copy()

    arm_counts = train_failed["chosen_intervention"].value_counts()
    if arm_counts.empty:
        print("[calibration] no arms with data, skipping")
        return None
    top_arm = arm_counts.idxmax()

    train_subset = train_failed[train_failed["chosen_intervention"] == top_arm]
    val_subset = val_failed[val_failed["chosen_intervention"] == top_arm]
    if len(val_subset) < 20 or train_subset["recovered"].nunique() < 2:
        print(f"[calibration] insufficient data for arm '{top_arm}', skipping")
        return None

    X_train = build_feature_frame(train_subset)
    y_train = train_subset["recovered"].astype(int)
    X_val = build_feature_frame(val_subset)
    X_val = align_columns(X_val, list(X_train.columns))
    y_val = val_subset["recovered"].astype(int)

    raw_model = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3)
    raw_model.fit(X_train, y_train)
    raw_probs = raw_model.predict_proba(X_val)[:, 1]

    from sklearn.calibration import CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(
        GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3),
        method="isotonic", cv=3,
    )
    calibrated_model.fit(X_train, y_train)
    calibrated_probs = calibrated_model.predict_proba(X_val)[:, 1]

    report = compare_raw_vs_calibrated(y_val.values, raw_probs, calibrated_probs)
    report["arm"] = top_arm
    print(f"[calibration:{top_arm}] raw Brier={report['raw']['brier_score']} "
          f"vs calibrated Brier={report['calibrated']['brier_score']} "
          f"(lower is better)")
    print(f"[calibration:{top_arm}] raw ECE={report['raw']['ece']} "
          f"vs calibrated ECE={report['calibrated']['ece']} (lower is better)")

    with open(os.path.join(MODEL_DIR, "calibration_report.json"), "w") as f:
        import json
        json.dump(report, f, indent=2)
    return report


def train_uncertainty_ensembles(train_df):
    train_failed = train_df[train_df["failed"] == True].copy()
    ensembles = {}
    for arm in ARMS:
        subset = train_failed[train_failed["chosen_intervention"] == arm]
        if len(subset) < 40 or subset["recovered"].nunique() < 2:
            continue
        X = build_feature_frame(subset)
        y = subset["recovered"].astype(int)
        models = train_bootstrap_ensemble(X, y)
        ensembles[arm] = {"models": models, "columns": list(X.columns)}
        print(f"[uncertainty:{arm}] trained {len(models)}-model bootstrap ensemble")
    joblib.dump(ensembles, os.path.join(MODEL_DIR, "uncertainty_ensembles.pkl"))
    return ensembles


if __name__ == "__main__":
    df = load_events()
    train_df, val_df, test_df = temporal_split(df)
    print(f"Loaded {len(df)} events -> train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Failed events: train={train_df['failed'].sum()} val={val_df['failed'].sum()} "
          f"test={test_df['failed'].sum()}")

    train_root_cause(train_df, val_df)

    arm_models = train_per_arm_models(train_df)
    save_models(arm_models)

    train_and_evaluate_calibration(train_df, val_df)

    train_uncertainty_ensembles(train_df)

    print("\nTraining complete. Models saved to ml-service/models/")
    print("NOTE: test_df was never touched during training -- reserved for "
          "evaluation/ablation.py and evaluation/offline_policy_eval.py")
