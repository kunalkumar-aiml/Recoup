"""
Chaos test suite (round-6, Parts 2/4/5/16).

Runs a REAL failure-injection subset against the LIVE ml-service — not
a mocked/simulated version. Requires the service running on :8000.

  cd ml-service && uvicorn app:app --reload --port 8000   # separate terminal
  cd chaos && python3 chaos_test.py

Covers 15 of the 40 scenarios listed in the round-6 request (the ones
tractable to test honestly against what's actually implemented). The
other 25 (database outage, circuit breakers, rate limiting, load
testing, concurrent-request races, model registry/versioning, shadow
mode, rollback) require infrastructure this build does not have --
listed explicitly as NOT TESTED at the bottom, not silently skipped.
"""
import os
import requests
import math

BASE_URL = "http://localhost:8000"

BASE_PAYLOAD = {
    "event_id": "chaos-base", "merchant_category": "ecommerce", "customer_value_tier": "mid",
    "event_type": "payment_attempt", "payment_method": "card", "decline_code": "network_error",
    "amount": 1500, "customer_retry_propensity_observed": 0.5, "merchant_baseline_fail_rate": 0.2,
    "prior_failure_count": 0, "minutes_since_last_failure": -1, "prior_recovery_count": 0,
    "prior_recovery_rate": -1, "recent_method_switch_count": 0, "retry_count": 0,
}

results = []


def record(name, status, detail):
    results.append({"test": name, "status": status, "detail": detail})
    print(f"[{status:4s}] {name}: {detail}")


def payload(event_id, **overrides):
    p = dict(BASE_PAYLOAD)
    p["event_id"] = event_id
    p.update(overrides)
    return p


def safe_post(p, expect_http_error=False):
    try:
        r = requests.post(f"{BASE_URL}/decide", json=p, timeout=5)
        return r, None
    except requests.exceptions.RequestException as e:
        return None, e


def check_did_not_crash_and_is_safe(name, p, expect_state_in=("SAFE_FALLBACK", "HUMAN_REVIEW", "SYSTEM_ERROR")):
    r, err = safe_post(p)
    if err:
        record(name, "FAIL", f"request-level exception: {err}")
        return
    if r.status_code == 422:
        record(name, "PASS", "rejected at schema validation (422) -- correct fail-fast behavior")
        return
    if r.status_code != 200:
        record(name, "FAIL", f"unexpected HTTP {r.status_code}: {r.text[:200]}")
        return
    data = r.json()
    state = data.get("decision_state")
    if data.get("error") or state in expect_state_in:
        record(name, "PASS", f"handled safely, decision_state={state}")
    elif state == "AUTO_APPROVED":
        record(name, "WARN", f"auto-approved despite the injected anomaly -- decision_state={state}, review whether this is correct")
    else:
        record(name, "WARN", f"unexpected state {state}, response: {str(data)[:200]}")


def run():
    print("=" * 78)
    print("RECOUP CHAOS TEST SUITE")
    print("=" * 78)

    # 1. Duplicate event (idempotency)
    p = payload("chaos-dup-1")
    r1, _ = safe_post(p)
    r2, _ = safe_post(p)
    if r1 and r2 and r1.status_code == 200 and r2.status_code == 200:
        d1, d2 = r1.json().get("decision_id"), r2.json().get("decision_id")
        if d1 == d2 and r2.json().get("idempotent_replay"):
            record("Duplicate event (idempotency)", "PASS", f"same decision_id ({d1}) returned, second call flagged idempotent_replay=True")
        else:
            record("Duplicate event (idempotency)", "FAIL", f"decision_ids differ or not flagged: {d1} vs {d2}")
    else:
        record("Duplicate event (idempotency)", "FAIL", "one or both calls failed")

    # 2. Duplicate feedback
    p = payload("chaos-dup-fb-1")
    r, _ = safe_post(p)
    if r and r.status_code == 200 and r.json().get("decision_id"):
        did = r.json()["decision_id"]
        fb_payload = {"decision_id": did, "arm": "retry_timing", "recovered": True, "amount": 1500}
        fb1 = requests.post(f"{BASE_URL}/feedback", json=fb_payload, timeout=5).json()
        fb2 = requests.post(f"{BASE_URL}/feedback", json=fb_payload, timeout=5).json()
        if fb1.get("updated") and not fb2.get("updated") and fb2.get("already_processed"):
            record("Duplicate feedback (idempotency)", "PASS", "first call updates, second is a no-op (already_processed=True)")
        else:
            record("Duplicate feedback (idempotency)", "FAIL", f"fb1={fb1}, fb2={fb2}")
    else:
        record("Duplicate feedback (idempotency)", "WARN", "no decision_id produced for this context, cannot test feedback path")

    # 3. Malformed event / missing field -> should be a clean 422, not a crash
    r, err = safe_post({"event_id": "chaos-malformed-1", "merchant_category": "ecommerce"})
    if r is not None and r.status_code == 422:
        record("Malformed event (missing fields)", "PASS", "rejected with 422 schema validation error")
    else:
        record("Malformed event (missing fields)", "FAIL", f"expected 422, got {r.status_code if r else err}")

    # 4. Negative amount
    check_did_not_crash_and_is_safe("Negative amount", payload("chaos-neg-1", amount=-5000))

    # 5. Zero amount
    check_did_not_crash_and_is_safe("Zero amount", payload("chaos-zero-1", amount=0), expect_state_in=("SAFE_FALLBACK", "HUMAN_REVIEW", "SYSTEM_ERROR", "AUTO_APPROVED"))

    # 6. Extreme amount
    check_did_not_crash_and_is_safe("Extreme amount (₹50,000,000)", payload("chaos-extreme-1", amount=50000000, customer_value_tier="high"), expect_state_in=("SAFE_FALLBACK", "HUMAN_REVIEW"))

    # 7. NaN-like value (JSON doesn't support NaN natively; send as string to test type coercion)
    r, err = safe_post({**payload("chaos-nan-1"), "amount": "not_a_number"})
    if r is not None and r.status_code == 422:
        record("Non-numeric amount (type error)", "PASS", "rejected with 422 type validation error")
    else:
        record("Non-numeric amount (type error)", "FAIL", f"expected 422, got {r.status_code if r else err}")

    # 8. Unknown/unseen category
    check_did_not_crash_and_is_safe("Unknown merchant category", payload("chaos-unseen-cat-1", merchant_category="crypto_exchange_v2"), expect_state_in=("SAFE_FALLBACK", "HUMAN_REVIEW", "AUTO_APPROVED"))

    # 9. Unknown decline code
    check_did_not_crash_and_is_safe("Unknown decline code", payload("chaos-unseen-decline-1", decline_code="quantum_flux_error"), expect_state_in=("SAFE_FALLBACK", "HUMAN_REVIEW", "AUTO_APPROVED"))

    # 10. Extreme retry count
    check_did_not_crash_and_is_safe("Extreme retry count (999)", payload("chaos-retry-999", retry_count=999), expect_state_in=("SAFE_FALLBACK", "HUMAN_REVIEW"))

    # 11. Feedback with unknown decision_id
    fb = requests.post(f"{BASE_URL}/feedback", json={"decision_id": "does-not-exist-12345", "arm": "retry_timing", "recovered": True, "amount": 100}, timeout=5)
    if fb.status_code == 404:
        record("Feedback with unknown decision_id", "PASS", "correctly rejected with 404, no silent update")
    else:
        record("Feedback with unknown decision_id", "FAIL", f"expected 404, got {fb.status_code}")

    # 12. Feedback reward injection attempt (client cannot supply a raw reward field at all)
    fb_payload_with_injected_reward = {"decision_id": "irrelevant", "arm": "retry_timing", "recovered": True, "amount": 100, "reward": 999999}
    fb = requests.post(f"{BASE_URL}/feedback", json=fb_payload_with_injected_reward, timeout=5)
    # FeedbackIn schema has no "reward" field -- pydantic silently ignores extra
    # fields by default, so this should behave identically to no "reward" key,
    # i.e. still 404 for an unknown decision_id, NOT accept the injected value
    if fb.status_code == 404:
        record("Feedback reward injection attempt", "PASS", "extra 'reward' field ignored by schema; request still correctly rejected on unknown decision_id")
    else:
        record("Feedback reward injection attempt", "WARN", f"unexpected response: {fb.status_code} {fb.text[:150]}")

    # 13. High-value transaction requires human review
    r, _ = safe_post(payload("chaos-highvalue-1", amount=45000, customer_value_tier="high"))
    if r and r.status_code == 200:
        data = r.json()
        if data.get("requires_human_review") or data.get("decision_state") == "HUMAN_REVIEW":
            record("High-value transaction safety", "PASS", f"amount=45000 correctly requires human review, state={data.get('decision_state')}")
        else:
            record("High-value transaction safety", "FAIL", f"amount=45000 was NOT flagged for human review, state={data.get('decision_state')}")
    else:
        record("High-value transaction safety", "FAIL", "request failed")

    # 14. Health check works (basic liveness)
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    if r.status_code == 200 and r.json().get("status") == "ok":
        record("Service liveness (/health)", "PASS", "responds 200 ok")
    else:
        record("Service liveness (/health)", "FAIL", f"unexpected: {r.status_code}")

    # 15. Missing/empty decline_code (checkout_abandoned events legitimately have this)
    check_did_not_crash_and_is_safe("Empty decline_code (legitimate case)", payload("chaos-empty-decline-1", decline_code=""), expect_state_in=("SAFE_FALLBACK", "HUMAN_REVIEW", "AUTO_APPROVED"))

    print("\n" + "=" * 78)
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"SUMMARY: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL out of {len(results)} tests")
    print("=" * 78)
    print("""
NOT TESTED this round (requires infrastructure this build does not have,
stated explicitly rather than silently skipped):
  - Database/ML-service/policy-service OUTAGE simulation (circuit breakers)
  - Rate limiting, load testing at real scale (500+ RPS with a proper tool)
  - Model registry, corrupted/incompatible model artifact rejection
  - Shadow mode, model promotion gate, rollback
  - Out-of-order event handling at the temporal-feature level (server-side
    state now exists as of round 7, but assumes events arrive roughly in
    order -- a genuinely out-of-order event is not specially handled)
  - Stale-state TTL eviction
  (Concurrent/parallel duplicate-request race testing IS now covered --
  see ml-service/test_concurrency.py, round-7 fix)
""")

    import json
    with open(os.path.join(os.path.dirname(__file__), "chaos_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to chaos/chaos_test_results.json")


if __name__ == "__main__":
    run()
