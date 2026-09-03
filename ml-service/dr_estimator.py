"""
Cross-fitted Doubly-Robust (AIPW) treatment effect estimator.

MATH (multi-arm AIPW, per arm a vs control arm "no_action"):

  e_a(X) = P(A=a | X)                    propensity model
  m_a(X) = E[Y | X, A=a]                 outcome model, per arm

  AIPW pseudo-outcome for arm a, evaluated at a unit with observed
  action A and outcome Y:

    psi_a(X, A, Y) = m_a(X) + (1[A=a] / e_a(X)) * (Y - m_a(X))   if A == a
                     m_a(X)                                       otherwise
                     (equivalently: psi_a = m_a(X) + 1[A=a]/e_a(X) * (Y - m_a(X)))

  tau_a(X) = E[psi_a(X)] - E[psi_control(X)]

CROSS-FITTING (K folds, default K=3 given small per-arm sample sizes):
  For each fold k: fit e_a and m_a on the OTHER K-1 folds, predict on
  fold k, construct the pseudo-outcome on fold k using out-of-fold
  predictions only. This avoids the nuisance models overfitting into
  the treatment-effect estimate -- a unit's own outcome never
  influences the nuisance predictions used to score it.

WHY THIS SHOULD BE MORE ROBUST THAN THE T-LEARNER (round-2 through
round-8's stated hypothesis): the T-learner's control-arm estimate
mu_no_action(X) is a single model with no correction if it's
mis-specified in low-support regions. AIPW's propensity-weighted
correction term explicitly down-weights/corrects for exactly the kind
of low-overlap region round 2's diagnostics found for no_action --
IF the propensity model itself is reasonably accurate. This is a
testable claim, not an assumption -- see the synthetic validation below
and evaluation/dr_cross_fitting_experiment.py for the real-data test.

IDENTIFICATION ASSUMPTIONS (stated explicitly, not implied):
  - Consistency: observed outcome under the logged action equals the
    potential outcome under that action.
  - Conditional exchangeability / no unmeasured confounding: treatment
    assignment depends only on X, the observed features.
  - Positivity/overlap: every arm has nonzero propensity for every X in
    the population of interest -- KNOWN TO BE VIOLATED for no_action in
    this codebase (73.8% of contexts below common support,
    evaluation/overlap_diagnostics.py). AIPW is more robust to
    misspecification of EITHER nuisance model, not to overlap violations
    themselves -- if propensity for an arm is near zero, its IPW term
    still explodes.

This module implements the estimator only. Validation (synthetic +
real-data) lives in evaluation/dr_cross_fitting_experiment.py.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

PROPENSITY_CLIP = 0.05  # default; sensitivity tested separately


def cross_fitted_aipw(
    X: np.ndarray, A: np.ndarray, Y: np.ndarray, arms: list,
    n_folds: int = 3, propensity_clip: float = PROPENSITY_CLIP, seed: int = 42,
):
    """
    X: (n, d) feature matrix
    A: (n,) array of arm labels (strings, values in `arms`)
    Y: (n,) array of binary outcomes
    arms: list of all arm labels, arms[0] is treated as the control arm

    Returns: dict {arm: array of shape (n,) of out-of-fold pseudo-outcomes psi_a}
    """
    n = len(Y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    psi = {a: np.full(n, np.nan) for a in arms}

    for train_idx, held_idx in kf.split(X):
        X_train, X_held = X[train_idx], X[held_idx]
        A_train, A_held = A[train_idx], A[held_idx]
        Y_train, Y_held = Y[train_idx], Y[held_idx]

        # propensity model: multinomial logistic regression on the
        # training fold only
        if len(set(A_train)) < 2:
            continue
        prop_model = LogisticRegression(max_iter=2000)
        prop_model.fit(X_train, A_train)
        prop_classes = list(prop_model.classes_)
        e_held = prop_model.predict_proba(X_held)  # (n_held, n_classes)

        for arm in arms:
            if arm not in prop_classes:
                continue
            arm_idx_in_model = prop_classes.index(arm)
            e_a_held = np.clip(e_held[:, arm_idx_in_model], propensity_clip, 1 - propensity_clip)

            # outcome model m_a(X): fit on training-fold rows where A==arm only
            arm_mask_train = A_train == arm
            if arm_mask_train.sum() < 10 or len(set(Y_train[arm_mask_train])) < 2:
                # insufficient data to fit an outcome model for this arm
                # on this fold -- fall back to the arm's mean outcome
                # (a degenerate but honest outcome model, not a crash)
                fallback_rate = Y_train[arm_mask_train].mean() if arm_mask_train.sum() > 0 else Y_train.mean()
                m_a_held = np.full(len(X_held), fallback_rate)
            else:
                outcome_model = GradientBoostingClassifier(random_state=seed, n_estimators=50, max_depth=2)
                outcome_model.fit(X_train[arm_mask_train], Y_train[arm_mask_train])
                m_a_held = outcome_model.predict_proba(X_held)[:, 1]

            indicator = (A_held == arm).astype(float)
            residual = np.where(A_held == arm, Y_held - m_a_held, 0.0)
            psi_a_held = m_a_held + (indicator / e_a_held) * residual
            psi[arm][held_idx] = psi_a_held

    return psi


def estimate_tau(psi: dict, control_arm: str):
    """tau_a(X) per unit = psi_a - psi_control, averaged where both exist."""
    control = psi[control_arm]
    tau = {}
    for arm, p in psi.items():
        if arm == control_arm:
            continue
        tau[arm] = p - control
    return tau


if __name__ == "__main__":
    # ---- SYNTHETIC GROUND-TRUTH SELF-VALIDATION (Step 4) ----
    # A dataset with KNOWN true treatment effects, generated independently
    # of any model family used for estimation, so the estimator's own
    # correctness can be checked before trusting it on real data.
    rng = np.random.RandomState(0)
    n = 3000
    d = 4
    X = rng.uniform(-1, 1, size=(n, d))

    # true (nonlinear) propensity: control vs treatment, WITH confounding
    # (propensity depends on X, and so does the outcome -- classic
    # observational-bias setup)
    true_logit = 1.5 * X[:, 0] - 1.0 * X[:, 1]
    p_treat = 1 / (1 + np.exp(-true_logit))
    A = np.where(rng.uniform(size=n) < p_treat, "treatment", "control")

    # true outcome model: nonlinear, with a KNOWN, CONSTANT treatment
    # effect of +0.15 (so the estimator's job is to recover 0.15)
    true_control_prob = 1 / (1 + np.exp(-(0.8 * X[:, 0] + 0.5 * X[:, 2] ** 2 - 0.3)))
    TRUE_TAU = 0.15
    true_treat_prob = np.clip(true_control_prob + TRUE_TAU, 0.01, 0.99)
    p_outcome = np.where(A == "treatment", true_treat_prob, true_control_prob)
    Y = (rng.uniform(size=n) < p_outcome).astype(int)

    print("=" * 70)
    print("SYNTHETIC GROUND-TRUTH VALIDATION")
    print(f"True constant treatment effect: {TRUE_TAU}")
    print("=" * 70)

    # T-learner baseline for comparison
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    control_mask = A == "control"
    treat_mask = A == "treatment"
    m0 = GBC(random_state=0, n_estimators=50, max_depth=2).fit(X[control_mask], Y[control_mask])
    m1 = GBC(random_state=0, n_estimators=50, max_depth=2).fit(X[treat_mask], Y[treat_mask])
    tlearner_tau = m1.predict_proba(X)[:, 1] - m0.predict_proba(X)[:, 1]

    psi = cross_fitted_aipw(X, A, Y, arms=["control", "treatment"], n_folds=3)
    tau_dict = estimate_tau(psi, control_arm="control")
    dr_tau = tau_dict["treatment"]
    valid = ~np.isnan(dr_tau)

    t_mae = np.mean(np.abs(tlearner_tau - TRUE_TAU))
    t_bias = np.mean(tlearner_tau - TRUE_TAU)
    dr_mae = np.mean(np.abs(dr_tau[valid] - TRUE_TAU))
    dr_bias = np.mean(dr_tau[valid] - TRUE_TAU)

    print(f"T-learner: mean estimated tau={tlearner_tau.mean():.4f}  MAE={t_mae:.4f}  bias={t_bias:+.4f}")
    print(f"DR (AIPW): mean estimated tau={dr_tau[valid].mean():.4f}  MAE={dr_mae:.4f}  bias={dr_bias:+.4f}")
    print(f"\nBoth should be reasonably close to the true effect ({TRUE_TAU}) -- "
          f"this validates the cross-fitting implementation is not obviously broken "
          f"before trusting it on the real (unknown ground truth in production) data.")
