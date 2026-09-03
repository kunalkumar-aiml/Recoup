"""
Uplift / Treatment-Effect Estimation — T-learner (per-arm conditional
outcome models).

HONEST FRAMING (corrected after round-2 red-team review): fitting a
separate calibrated model per arm gives us
    mu_a(X) = E[Y | X, A=a]
and the difference
    tau_a(X) = mu_a(X) - mu_no_action(X)
is a valid estimate of the CONDITIONAL AVERAGE TREATMENT EFFECT
tau_a(X) = E[Y(a) - Y(0) | X] **only under three identification
assumptions**, which we state explicitly rather than assume silently:

  1. CONSISTENCY — the observed outcome for units that received arm a
     equals their potential outcome under a. True by construction in
     our simulator (the logged outcome IS the potential outcome for the
     arm actually chosen).
  2. POSITIVITY / OVERLAP — every arm has nonzero probability of being
     chosen for every context region we make predictions on. This is
     NOT uniformly true here: the logging policy is merchant-biased
     (see generate_data.py), so some (context, arm) regions have very
     thin support. See evaluation/overlap_diagnostics.py for exactly
     where this assumption is weak, and docs/causal_identification.md
     for the full writeup.
  3. CONDITIONAL IGNORABILITY (no unmeasured confounding given X) —
     treatment assignment depends only on features we observe. In our
     simulator this is TRUE by construction (the logging-policy bias
     function only reads observed merchant/customer/decline_code
     features), but this is a strong assumption that would need
     justification against real Razorpay data.

Because assumption 2 is measurably violated in some regions, we do NOT
call this "a genuine causal estimator" unconditionally. It is a T-learner
conditional-outcome-difference estimate, causal under the assumptions
above, with overlap diagnostics reported separately so the estimate's
reliability can be judged region by region rather than trusted uniformly.

WHY T-LEARNER OVER S/X-LEARNER/CAUSAL FOREST (unchanged reasoning, see
git history / docs/gap_analysis.md for the original comparison) — still
the smallest defensible choice given our sample size, NOW cross-checked
by a separate cross-fitted doubly-robust estimator
(evaluation/doubly_robust_eval.py) that does not share this module's
assumptions and can be used to sanity-check disagreement.

FIXED (round 2): the no_action arm's fallback used to substitute the
lowest-amount actioned events as a fake control group when true
no_action data was thin. This was a scientifically invalid fabricated
control population and has been REMOVED. If no_action support is
insufficient, `train_per_arm_models` now marks it
INSUFFICIENT_CONTROL_SUPPORT and the arm is excluded — uplift for that
context is then unavailable, and the decision engine (bandit.py) treats
missing uplift as a forced conservative/escalation case rather than
silently substituting the arm with a raw probability.

FIXED (round 2): internal calibration holdout was previously a RANDOM
train_test_split even though the outer pipeline is temporally split —
this reintroduced leakage risk inside the arm-training step. Now uses a
temporal split (earliest 75% / most recent 25% of that arm's rows).
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from features import build_feature_frame, align_columns

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
UPLIFT_MODELS_PATH = os.path.join(MODEL_DIR, "uplift_arms.pkl")

ARMS = ["no_action", "retry_timing", "alt_method_nudge", "discount_offer",
        "human_escalation", "hinglish_voice_nudge"]

MIN_ARM_SUPPORT = 30


class InsufficientControlSupport(Exception):
    """Raised (and caught) when an arm — especially no_action — does not
    have enough genuinely-logged data to fit a model. Never substituted
    with a fabricated proxy population."""
    pass


def _rows_for_arm(df, arm):
    """Rows where this EXACT arm was actually logged. No substitution,
    no proxy, no fallback population -- if it's thin, it's thin."""
    if arm == "no_action":
        return df[(df["failed"] == True) & (df["chosen_intervention"] == "no_action")]
    return df[(df["failed"] == True) & (df["chosen_intervention"] == arm)]


def _temporal_holdout_split(subset: pd.DataFrame, train_frac=0.75):
    """Time-ordered split WITHIN this arm's rows -- replaces the earlier
    random train_test_split, which risked leaking information across the
    temporal boundary the outer pipeline was built to respect."""
    subset = subset.sort_values("timestamp")
    n = len(subset)
    cut = int(n * train_frac)
    return subset.iloc[:cut], subset.iloc[cut:]


def train_per_arm_models(train_df: pd.DataFrame) -> dict:
    models = {}
    insufficient = []
    for arm in ARMS:
        subset = _rows_for_arm(train_df, arm)
        if len(subset) < MIN_ARM_SUPPORT or subset["recovered"].nunique() < 2:
            print(f"[uplift:{arm}] INSUFFICIENT_CONTROL_SUPPORT (n={len(subset)}, "
                  f"min required={MIN_ARM_SUPPORT}) -- excluded, no fallback substituted")
            insufficient.append(arm)
            continue

        arm_train, arm_holdout = _temporal_holdout_split(subset)
        if len(arm_holdout) < 10 or arm_train["recovered"].nunique() < 2:
            print(f"[uplift:{arm}] INSUFFICIENT_CONTROL_SUPPORT after temporal split "
                  f"(train={len(arm_train)}, holdout={len(arm_holdout)}) -- excluded")
            insufficient.append(arm)
            continue

        X_train = build_feature_frame(arm_train)
        y_train = arm_train["recovered"].astype(int)
        X_holdout = build_feature_frame(arm_holdout)
        X_holdout = align_columns(X_holdout, list(X_train.columns))
        y_holdout = arm_holdout["recovered"].astype(int)

        base = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3)
        # round-3 fix: with some seeds, a minority class can drop below 3
        # examples, which crashes CalibratedClassifierCV's default cv=3.
        # Use the largest feasible fold count (min 2) instead of assuming
        # 3 always works -- discovered BY the multi-seed robustness run
        # (evaluation/run_seed_robustness.sh), not anticipated in advance.
        min_class_count = y_train.value_counts().min()
        cv_folds = max(2, min(3, int(min_class_count)))
        if min_class_count < 2:
            print(f"[uplift:{arm}] WARNING: minority class has only {min_class_count} "
                  f"example(s) -- skipping calibration, using raw (uncalibrated) model")
            model = base
        else:
            model = CalibratedClassifierCV(base, method="isotonic", cv=cv_folds)
        model.fit(X_train, y_train)
        try:
            score = model.score(X_holdout, y_holdout)
        except ValueError:
            score = float("nan")  # holdout may be single-class; report honestly
        print(f"[uplift:{arm}] n={len(subset)} (train={len(arm_train)} "
              f"temporal-holdout={len(arm_holdout)}) holdout score: {score:.3f}")
        models[arm] = {"model": model, "columns": list(X_train.columns)}

    if "no_action" in insufficient:
        print("\nWARNING: no_action (control arm) has INSUFFICIENT_CONTROL_SUPPORT. "
              "Uplift cannot be computed anywhere in this run -- compute_uplift() "
              "will return {} and the decision engine will treat every context as "
              "requiring escalation. This is the correct, honest behavior: a T-learner "
              "with no valid control group must not produce uplift numbers.")

    return models


def save_models(models: dict, path=UPLIFT_MODELS_PATH):
    joblib.dump(models, path)


def load_models(path=UPLIFT_MODELS_PATH) -> dict:
    return joblib.load(path)


def predict_arm_probs(models: dict, row_df: pd.DataFrame) -> dict:
    """mu_a(X) for every trained arm, for a single-row context."""
    probs = {}
    for arm, bundle in models.items():
        X = build_feature_frame(row_df)
        X = align_columns(X, bundle["columns"])
        probs[arm] = float(bundle["model"].predict_proba(X)[0][1])
    return probs


def compute_uplift(recovery_probs: dict) -> dict:
    """tau_a(X) = mu_a(X) - mu_no_action(X) for every actionable arm.
    Returns {} if no_action wasn't trainable -- see
    InsufficientControlSupport docstring above. An empty dict here is a
    signal, not a bug: the decision engine must treat it as "cannot
    estimate incremental value, escalate.\""""
    if "no_action" not in recovery_probs:
        return {}
    control = recovery_probs["no_action"]
    return {
        arm: round(p - control, 4)
        for arm, p in recovery_probs.items()
        if arm != "no_action"
    }


if __name__ == "__main__":
    from features import load_events
    from temporal_split import temporal_split

    df = load_events()
    train_df, _, _ = temporal_split(df)
    models = train_per_arm_models(train_df)
    save_models(models)
    print(f"\nTrained and saved uplift models for arms: {list(models.keys())}")

    if "no_action" in models:
        sample = train_df[train_df["failed"] == True].sample(1, random_state=1)
        probs = predict_arm_probs(models, sample)
        uplift = compute_uplift(probs)
        print("\nExample uplift estimate for one context:")
        for arm, val in sorted(uplift.items(), key=lambda kv: -kv[1]):
            print(f"  {arm:22s} mu={probs[arm]:.3f}  uplift_vs_no_action={val:+.3f}")
