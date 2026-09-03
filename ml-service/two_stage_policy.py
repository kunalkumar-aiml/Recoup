"""
Two-Stage Policy (round-3 fix, Phase 7/8).

WHY: round-2's overlap diagnostics found that the flat 6-arm T-learner's
control arm (no_action) has only ~80 logged examples system-wide, and
73.8% of test contexts fall below the 0.05 common-support threshold for
it -- every uplift estimate in the flat policy depends on this thin arm.

FIX: split the decision into two stages with much better-supported
training data at each stage:

  STAGE 1 -- "should we intervene at all?"
    Binary classifier: intervened (any of the 5 actionable arms) vs
    no_action. The positive class here pools ALL actionable-arm events
    (~1,500+ rows in this build) against the same ~80 no_action rows --
    the classifier itself doesn't get MORE no_action data, but the
    overlap diagnostic that matters here is P(intervened=1 | X), which
    is a much better-supported quantity than P(A=one specific arm | X)
    was for each of 6 arms individually, because we're no longer asking
    the logging policy "how often did you pick THIS SPECIFIC arm in
    THIS SPECIFIC context" -- just "did you act at all".

  STAGE 2 -- "given we should intervene, which arm?"
    Conditional on Stage 1 saying "intervene", rank the 5 actionable
    arms by mu_a(X) * amount - cost(a) using the EXISTING per-arm
    models from uplift.py (no change needed there) -- but critically,
    this comparison never needs mu_no_action(X) at all, so the thin
    no_action arm's poor overlap no longer contaminates the arm-choice
    decision. It only still matters for Stage 1's binary question.

This does NOT eliminate the overlap problem (Stage 1 still needs
no_action data to estimate P(intervene|X) well in regions where the
logging policy almost never left an event unactioned) -- it ISOLATES the
problem to one binary decision instead of letting it corrupt all 5
arm-vs-no_action uplift comparisons individually. Reported honestly:
this is a mitigation, not a fix.
"""
import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
from features import build_feature_frame, align_columns  # noqa: E402
from uplift import ARMS, load_models as load_uplift_models  # noqa: E402
from protocol import COST_TABLE  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
STAGE1_MODEL_PATH = os.path.join(MODEL_DIR, "two_stage_stage1.pkl")

ACTIONABLE = ["retry_timing", "alt_method_nudge", "discount_offer",
              "human_escalation", "hinglish_voice_nudge"]


def train_stage1(train_df: pd.DataFrame):
    failed = train_df[train_df["failed"] == True].copy()
    failed = failed[failed["chosen_intervention"] != ""]
    failed["intervened"] = (failed["chosen_intervention"] != "no_action").astype(int)
    X = build_feature_frame(failed)
    y = failed["intervened"]
    if y.nunique() < 2:
        return None, None
    base = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X, y)
    return model, list(X.columns)


def save_stage1(model, columns):
    joblib.dump({"model": model, "columns": columns}, STAGE1_MODEL_PATH)


def load_stage1():
    if not os.path.exists(STAGE1_MODEL_PATH):
        return None
    return joblib.load(STAGE1_MODEL_PATH)


def stage1_should_intervene(bundle, row_df, threshold=0.5):
    X = build_feature_frame(row_df)
    X = align_columns(X, bundle["columns"])
    p_intervene = float(bundle["model"].predict_proba(X)[0][1])
    return p_intervene >= threshold, p_intervene


def stage2_pick_arm(uplift_arm_models, row_df, amount, cost_table, discount_pct=0.0):
    """Ranks actionable arms by mu_a(X) * amount - cost, using the raw
    per-arm probability directly (NOT uplift vs no_action) -- this is the
    key move that avoids the thin no_action overlap contaminating the
    arm-choice decision."""
    best_arm, best_value = None, -1e18
    for arm in ACTIONABLE:
        if arm not in uplift_arm_models:
            continue
        bundle = uplift_arm_models[arm]
        X = build_feature_frame(row_df)
        X = align_columns(X, bundle["columns"])
        mu = float(bundle["model"].predict_proba(X)[0][1])
        gross = mu * amount
        leakage = gross * (discount_pct / 100.0) if arm == "discount_offer" else 0.0
        value = gross - leakage - cost_table.get(arm, 0)
        if value > best_value:
            best_value = value
            best_arm = arm
    return best_arm, best_value


def two_stage_decide(stage1_bundle, uplift_arm_models, row_df, amount, cost_table,
                      threshold=0.5, discount_pct=0.0):
    should_act, p_intervene = stage1_should_intervene(stage1_bundle, row_df, threshold)
    if not should_act:
        return None, {"stage1_p_intervene": round(p_intervene, 4), "stage": "stage1_no_action"}
    arm, value = stage2_pick_arm(uplift_arm_models, row_df, amount, cost_table, discount_pct)
    return arm, {"stage1_p_intervene": round(p_intervene, 4), "stage2_value": round(value, 2), "stage": "stage2_arm_choice"}


if __name__ == "__main__":
    from features import load_events
    from temporal_split import temporal_split

    df = load_events()
    train_df, _, _ = temporal_split(df)
    model, columns = train_stage1(train_df)
    if model is None:
        print("Stage 1 could not be trained (insufficient class variation).")
    else:
        save_stage1(model, columns)
        print(f"Stage 1 trained and saved. Features: {len(columns)}")
        sample = train_df[train_df["failed"] == True].sample(1, random_state=1)
        should_act, p = stage1_should_intervene({"model": model, "columns": columns}, sample)
        print(f"Example: P(intervene|X) = {p:.3f} -> {'INTERVENE' if should_act else 'NO ACTION'}")
