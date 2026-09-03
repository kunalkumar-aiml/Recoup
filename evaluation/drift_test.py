"""
Drift test — compares the TRAIN split's decline_code distribution
(reference) against the drift-window portion of the data (production
shift) and reports PSI + the resulting drift status.
"""
import os
import sys
import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))
from features import load_events  # noqa: E402
from temporal_split import temporal_split  # noqa: E402
from drift import psi, drift_status  # noqa: E402


def run():
    df = load_events()
    train_df, val_df, test_df = temporal_split(df)

    reference = train_df[train_df["failed"] == True]["decline_code"].tolist()
    drift_window = df[(df["failed"] == True) & (df["drift_window"] == True)]["decline_code"].tolist()
    non_drift_val_test = pd.concat([val_df, test_df])
    non_drift_window = non_drift_val_test[
        (non_drift_val_test["failed"] == True) & (non_drift_val_test["drift_window"] == False)
    ]["decline_code"].tolist()

    print("=" * 70)
    print("DRIFT TEST: reference (train) vs simulated production shift (drift window)")
    print("=" * 70)
    score, breakdown = psi(reference, drift_window)
    status = drift_status(score)
    print(f"PSI = {score}  ->  status = {status}")
    for b in sorted(breakdown, key=lambda x: -abs(x["contribution"])):
        print(f"  {b['category']:<25} ref={b['reference_pct']:.3f} cur={b['current_pct']:.3f} "
              f"contribution={b['contribution']:.4f}")

    print("\n" + "=" * 70)
    print("CONTROL: reference (train) vs non-drift val/test window (should be STABLE)")
    print("=" * 70)
    score2, _ = psi(reference, non_drift_window)
    status2 = drift_status(score2)
    print(f"PSI = {score2}  ->  status = {status2}")

    print(f"\nInterpretation: drift-window PSI ({score}, {status}) should be meaningfully "
          f"higher than the non-drift control ({score2}, {status2}) if drift detection is "
          f"working correctly.")


if __name__ == "__main__":
    run()
