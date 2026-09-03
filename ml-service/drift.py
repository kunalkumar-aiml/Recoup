"""
Concept drift detection — Population Stability Index (PSI).

WHY PSI over KL/Jensen-Shannon: PSI is the standard, defensible metric
used in credit-risk / fintech model monitoring specifically because it
has widely-accepted, industry-standard interpretation thresholds
(< 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift) that
we can cite rather than invent. KL divergence is asymmetric and harder
to threshold meaningfully; Jensen-Shannon is a reasonable alternative
but has no equivalent standard threshold convention in this domain.

PSI formula for a categorical feature (e.g. decline_code) comparing a
reference distribution R to a current window distribution C:

    PSI = sum_i (C_i - R_i) * ln(C_i / R_i)

summed over each category i, with a small epsilon added to avoid log(0).

WHEN DRIFT IS DETECTED (PSI > 0.25 on decline_code distribution):
  1. reduce automated action confidence (treat all predictions as one
     confidence tier lower than their raw ensemble std would suggest)
  2. increase human escalation rate (policy engine consults this flag)
  3. log a monitoring event to the audit trail
  4. block the most aggressive/expensive interventions
     (discount_offer, human_escalation stays allowed since it's the
     SAFE fallback, but hinglish_voice_nudge / discount_offer are paused)
"""
import math
from collections import Counter

PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25
EPSILON = 1e-6


def distribution(values):
    counts = Counter(values)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def psi(reference_values, current_values):
    ref_dist = distribution(reference_values)
    cur_dist = distribution(current_values)
    categories = set(ref_dist.keys()) | set(cur_dist.keys())
    score = 0.0
    breakdown = []
    for cat in categories:
        r = ref_dist.get(cat, 0) + EPSILON
        c = cur_dist.get(cat, 0) + EPSILON
        contribution = (c - r) * math.log(c / r)
        score += contribution
        breakdown.append({
            "category": cat, "reference_pct": round(r, 4),
            "current_pct": round(c, 4), "contribution": round(contribution, 4),
        })
    return round(score, 4), breakdown


def drift_status(psi_score: float) -> str:
    if psi_score < PSI_MODERATE_THRESHOLD:
        return "STABLE"
    if psi_score < PSI_SIGNIFICANT_THRESHOLD:
        return "MODERATE_SHIFT"
    return "SIGNIFICANT_SHIFT"


def multi_signal_drift_status(reference: dict, recent: dict, min_window: int = 30) -> dict:
    """Round-2 fix #16: PSI on decline_code alone misses drift that shows
    up in payment_method mix, customer value-tier mix, or the amount
    distribution while decline_code holds steady. This computes PSI
    independently for each signal in `reference`/`recent` (dicts of
    {signal_name: [values]}) and reports an overall status = the worst
    (highest-severity) individual signal's status, so any single drifting
    signal is enough to trigger the gate -- not just decline_code.

    Kept deliberately small (4 signals: decline_code, payment_method,
    customer_value_tier, a binned amount) rather than monitoring every
    feature, per the project's own anti-overengineering principle.
    """
    per_signal = {}
    worst_rank = 0
    rank = {"INSUFFICIENT_DATA": 0, "STABLE": 1, "MODERATE_SHIFT": 2, "SIGNIFICANT_SHIFT": 3}
    worst_status = "INSUFFICIENT_DATA"

    for signal, cur_values in recent.items():
        ref_values = reference.get(signal, [])
        if len(cur_values) < min_window or not ref_values:
            per_signal[signal] = {"psi": None, "status": "INSUFFICIENT_DATA"}
            continue
        score, _ = psi(ref_values, cur_values)
        status = drift_status(score)
        per_signal[signal] = {"psi": score, "status": status}
        if rank[status] > worst_rank:
            worst_rank = rank[status]
            worst_status = status

    return {"overall_status": worst_status, "per_signal": per_signal}
