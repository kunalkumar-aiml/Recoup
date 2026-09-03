"""
Feature engineering — leakage-safe.

WHAT IS AVAILABLE AT DECISION TIME (and therefore usable as a feature):
  - everything about the current event: merchant, category, amount,
    payment method, decline_code, event_type, drift_window flag
  - everything about the customer/merchant's PAST behavior strictly
    before this event's timestamp (prior_failure_count, prior_recovery_rate,
    minutes_since_last_failure, recent_method_switch_count)
  - observed proxies of latent behavior (customer_retry_propensity_observed)

WHAT IS **NOT** AVAILABLE and therefore NEVER a feature:
  - recovered / recovered_value (this event's own label — the target)
  - chosen_intervention (this is the treatment/action, modeled separately
    per-arm — including it as a raw feature would leak the action into a
    "predict everything" model and break the per-arm uplift structure)
  - any potential_outcome_* column from oracle_potential_outcomes.csv
    (that file is COUNTERFACTUAL GROUND TRUTH and must never be joined
    into training — see data/generate_data.py docstring and
    docs/leakage_test.md)
  - future events for the same customer/merchant (the temporal features
    above are computed only from history strictly BEFORE the event
    timestamp at generation time — see generate_data.py's use of
    customer_history, which is appended to AFTER the row is written)

See docs/leakage_test.md for the explicit leakage unit tests that check
these invariants.
"""
import pandas as pd

CATEGORICAL_COLS = [
    "merchant_category", "customer_value_tier", "event_type",
    "payment_method", "decline_code",
]

NUMERIC_COLS = [
    "amount",
    "customer_retry_propensity_observed",
    "merchant_baseline_fail_rate",
    "prior_failure_count",
    "minutes_since_last_failure",
    "prior_recovery_count",
    "prior_recovery_rate",
    "recent_method_switch_count",
]

# columns that must NEVER appear in a feature frame
FORBIDDEN_COLS = {
    "recovered", "recovered_value", "chosen_intervention",
    "event_id", "timestamp", "customer_id", "merchant_id", "failed",
}


def load_events(path="../data/events.csv"):
    df = pd.read_csv(path)
    df["customer_value_tier"] = df["customer_value_tier"].fillna("unknown")
    df["decline_code"] = df["decline_code"].fillna("none")
    df["minutes_since_last_failure"] = df["minutes_since_last_failure"].fillna(-1)
    df["prior_recovery_rate"] = df["prior_recovery_rate"].fillna(-1)
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals, keep numerics. Raises if a forbidden
    (leaky) column is present in the input to catch mistakes early."""
    leaky = FORBIDDEN_COLS.intersection(set(CATEGORICAL_COLS + NUMERIC_COLS))
    assert not leaky, f"Leaky columns in feature spec: {leaky}"
    X = pd.get_dummies(df[CATEGORICAL_COLS + NUMERIC_COLS], columns=CATEGORICAL_COLS)
    return X


def align_columns(X: pd.DataFrame, reference_columns):
    for col in reference_columns:
        if col not in X.columns:
            X[col] = 0
    return X[reference_columns]
