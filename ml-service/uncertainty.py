"""
Uncertainty estimation — bootstrap ensemble disagreement.

APPROACH CHOSEN: train a small bootstrap ensemble (5 GradientBoosting
models, each on a resampled subset of the arm's training data) instead
of conformal prediction or full Bayesian posterior estimation.

WHY: conformal prediction needs a held-out calibration set with
exchangeability assumptions that are shakier here given the deliberate
concept-drift window (drift specifically breaks the exchangeability
assumption conformal methods rely on). A bootstrap ensemble is simpler,
directly gives us a spread of predictions we can use as a variance
proxy, and degrades gracefully under drift (the ensemble members
increasingly disagree exactly when the input looks unlike what any of
them were trained on) -- which is the behavior we want for the
escalation policy below.

CONFIDENCE TIERS (used by the policy engine):
  HIGH   ensemble std < 0.08  -> eligible for full automation
  MEDIUM 0.08 <= std < 0.18   -> conservative action only (cheapest arm)
  LOW    std >= 0.18          -> escalate to human, do not automate

This directly answers "P(recovered)=0.82 doesn't mean confident" --
0.82 with ensemble std 0.03 is a genuinely different situation than
0.82 with ensemble std 0.25, and only the policy engine sees the
difference if we compute it explicitly, which is what this module does.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample

HIGH_CONFIDENCE_STD = 0.08
MEDIUM_CONFIDENCE_STD = 0.18


def train_bootstrap_ensemble(X, y, n_models=5, random_state=42):
    models = []
    for i in range(n_models):
        X_boot, y_boot = resample(X, y, random_state=random_state + i)
        if y_boot.nunique() < 2:
            continue
        m = GradientBoostingClassifier(random_state=random_state + i, n_estimators=80, max_depth=3)
        m.fit(X_boot, y_boot)
        models.append(m)
    return models


def ensemble_predict(models, X_row):
    if not models:
        return 0.5, 1.0  # maximal uncertainty if no ensemble available
    preds = [m.predict_proba(X_row)[0][1] for m in models]
    return float(np.mean(preds)), float(np.std(preds))


def confidence_tier(std: float) -> str:
    if std < HIGH_CONFIDENCE_STD:
        return "HIGH"
    if std < MEDIUM_CONFIDENCE_STD:
        return "MEDIUM"
    return "LOW"


def selective_prediction_report(y_true, mean_preds, stds, threshold_std=HIGH_CONFIDENCE_STD):
    """Coverage vs risk: if we only auto-act when std < threshold_std, what
    fraction of cases do we cover, and what's the error rate on the ones
    we DO act on (vs the ones we'd have escalated)?"""
    y_true = np.asarray(y_true)
    mean_preds = np.asarray(mean_preds)
    stds = np.asarray(stds)
    covered = stds < threshold_std
    coverage = float(covered.mean())
    if covered.sum() == 0:
        return {"coverage": 0.0, "covered_error_rate": None, "escalated_error_rate": None}
    covered_pred_label = (mean_preds[covered] > 0.5).astype(int)
    covered_error = float(np.mean(covered_pred_label != y_true[covered]))
    escalated = ~covered
    result = {"coverage": round(coverage, 3), "covered_error_rate": round(covered_error, 3)}
    if escalated.sum() > 0:
        esc_pred_label = (mean_preds[escalated] > 0.5).astype(int)
        result["escalated_error_rate"] = round(float(np.mean(esc_pred_label != y_true[escalated])), 3)
    else:
        result["escalated_error_rate"] = None
    return result
