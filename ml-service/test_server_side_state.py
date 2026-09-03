"""
Server-side temporal state test (round-7, Phase 4).

PROVES the exact claim round 7 requires evidence for: "CLIENT LIES ABOUT
HISTORY -> MODEL STILL USES SERVER-SIDE STATE." Requires the live
ml-service running on :8000.
"""
import requests

BASE_URL = "http://localhost:8000"


def get_audit_input(event_id, limit=20):
    data = requests.get(f"{BASE_URL}/audit", params={"limit": limit}, timeout=5).json()
    for rec in data:
        if rec.get("event_id") == event_id:
            return rec.get("input", {})
    return None


def test_client_lies_about_history_ignored():
    customer_id = "CUST-TEST-STATE"
    base = {
        "merchant_category": "ecommerce", "customer_value_tier": "high", "event_type": "payment_attempt",
        "payment_method": "card", "decline_code": "network_error", "amount": 2000,
    }

    # event 1: genuinely the customer's first event -- trusted state should be all-zero/unknown
    r1 = requests.post(f"{BASE_URL}/decide", json={
        **base, "event_id": "state-e2e-1", "customer_id": customer_id,
        "event_timestamp": "2026-07-01T09:00:00+00:00",
        "prior_failure_count": 50, "prior_recovery_count": 50, "prior_recovery_rate": 1.0,  # LIE
    }, timeout=5).json()
    input1 = get_audit_input("state-e2e-1")
    assert input1["prior_failure_count"] == 0, f"expected 0 (first event), got {input1['prior_failure_count']}"
    print("PASS: first-ever event for this customer shows prior_failure_count=0 despite client claiming 50")

    # record the outcome so event 2 can see it
    did1 = r1.get("decision_id")
    requests.post(f"{BASE_URL}/feedback", json={
        "decision_id": did1, "arm": "retry_timing", "recovered": True, "amount": 2000
    }, timeout=5)

    # event 2: same customer, 2 hours later. Client claims wildly different
    # (also false) history. Server should report exactly 1 prior failure,
    # 1 prior recovery, ~120 minutes since last failure -- from its OWN store.
    r2 = requests.post(f"{BASE_URL}/decide", json={
        **base, "event_id": "state-e2e-2", "customer_id": customer_id,
        "event_timestamp": "2026-07-01T11:00:00+00:00",
        "prior_failure_count": 999, "prior_recovery_count": 0, "prior_recovery_rate": 0.0,  # LIE
        "minutes_since_last_failure": 1,  # LIE
    }, timeout=5).json()
    input2 = get_audit_input("state-e2e-2")
    assert input2["prior_failure_count"] == 1, f"expected 1, got {input2['prior_failure_count']}"
    assert input2["prior_recovery_count"] == 1, f"expected 1, got {input2['prior_recovery_count']}"
    assert abs(input2["minutes_since_last_failure"] - 120) < 1, f"expected ~120, got {input2['minutes_since_last_failure']}"
    print("PASS: second event shows server-computed prior_failure_count=1, prior_recovery_count=1, "
          "~120 minutes since last failure -- despite client claiming 999 failures, 0 recoveries, 1 minute")

    return True


if __name__ == "__main__":
    try:
        ok = test_client_lies_about_history_ignored()
        print("\nALL SERVER-SIDE STATE TESTS PASSED" if ok else "\nFAILED")
    except AssertionError as e:
        print(f"\nFAILED: {e}")
    except requests.exceptions.ConnectionError:
        print("ml-service not running. Start it: cd ml-service && uvicorn app:app --reload --port 8000")
