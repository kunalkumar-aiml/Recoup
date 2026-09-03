"""
Explicit leakage tests. Run: python3 test_no_leakage.py

These are the tests we'd want an ML engineer to be able to run themselves
rather than take our word for.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))

from features import load_events, build_feature_frame, FORBIDDEN_COLS, CATEGORICAL_COLS, NUMERIC_COLS  # noqa: E402


def test_forbidden_columns_not_in_feature_spec():
    used = set(CATEGORICAL_COLS + NUMERIC_COLS)
    leaked = used.intersection(FORBIDDEN_COLS)
    assert not leaked, f"FAIL: forbidden columns present in feature spec: {leaked}"
    print("PASS: no forbidden columns in feature spec (target, action, ids all excluded)")


def test_feature_frame_excludes_target_and_action():
    df = load_events()
    X = build_feature_frame(df.head(50))
    for col in X.columns:
        assert "recovered" not in col.lower(), f"FAIL: target leaked into feature '{col}'"
        assert "chosen_intervention" not in col.lower(), f"FAIL: action leaked into feature '{col}'"
    print("PASS: built feature frame contains no target or action columns")


def test_temporal_features_are_causally_prior():
    """prior_failure_count / prior_recovery_count for the FIRST event of any
    given customer must be 0 -- there is no history before the first event.
    If any customer's first-ever event already shows prior_failure_count > 0,
    that's a smoking gun for future information leaking backward."""
    df = load_events()
    df_sorted = df.sort_values("timestamp")
    first_events = df_sorted.groupby("customer_id").first()
    violations = first_events[first_events["prior_failure_count"] > 0]
    assert len(violations) == 0, (
        f"FAIL: {len(violations)} customers have prior_failure_count > 0 on "
        f"their first-ever logged event -- temporal leakage detected"
    )
    print("PASS: temporal features are 0 at each customer's first event (no backward leakage)")


def test_oracle_file_not_importable_by_training_code():
    """train.py must never read oracle_potential_outcomes.csv. This is a
    static check: grep the training module source for the filename."""
    train_path = os.path.join(os.path.dirname(__file__), "..", "ml-service", "train.py")
    with open(train_path) as f:
        source = f.read()
    assert "oracle_potential_outcomes" not in source, (
        "FAIL: train.py references the oracle file -- counterfactual ground "
        "truth must never be used for training"
    )
    print("PASS: train.py never references oracle_potential_outcomes.csv")


if __name__ == "__main__":
    test_forbidden_columns_not_in_feature_spec()
    test_feature_frame_excludes_target_and_action()
    test_temporal_features_are_causally_prior()
    test_oracle_file_not_importable_by_training_code()
    print("\nAll leakage tests passed.")
