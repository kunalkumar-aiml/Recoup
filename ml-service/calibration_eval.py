"""
Calibration metrics.

We report more than ROC-AUC/accuracy because a recovery system that acts
on P(recovered)=0.82 needs that 0.82 to actually MEAN something -- i.e.
of all predictions near 0.82, roughly 82% should actually recover.
Uncalibrated GradientBoosting probabilities are known to be
overconfident/underconfident away from 0.5, especially under the class
imbalance we have here (recoverable events are the minority class).
"""
import numpy as np


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE: bin predictions by confidence, compare average predicted
    probability to observed frequency in each bin, weight by bin size."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    reliability = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        bin_weight = mask.sum() / n
        ece += bin_weight * abs(bin_conf - bin_acc)
        reliability.append({
            "bin_range": f"[{lo:.1f}, {hi:.1f})", "count": int(mask.sum()),
            "avg_predicted": round(float(bin_conf), 4), "observed_rate": round(float(bin_acc), 4),
        })
    return float(ece), reliability


def compare_raw_vs_calibrated(y_true, y_prob_raw, y_prob_calibrated):
    raw_brier = brier_score(y_true, y_prob_raw)
    cal_brier = brier_score(y_true, y_prob_calibrated)
    raw_ece, _ = expected_calibration_error(y_true, y_prob_raw)
    cal_ece, cal_reliability = expected_calibration_error(y_true, y_prob_calibrated)
    return {
        "raw": {"brier_score": round(raw_brier, 4), "ece": round(raw_ece, 4)},
        "calibrated": {"brier_score": round(cal_brier, 4), "ece": round(cal_ece, 4)},
        "calibrated_reliability_diagram": cal_reliability,
        "improvement": {
            "brier_delta": round(raw_brier - cal_brier, 4),
            "ece_delta": round(raw_ece - cal_ece, 4),
        },
    }
