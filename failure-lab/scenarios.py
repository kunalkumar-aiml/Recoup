"""
Failure Lab — 12 scripted scenarios stressing Recoup's full decision
pipeline (root-cause posterior -> uplift -> uncertainty -> drift check ->
net value -> bandit -> policy gate). Run against the live ml-service:

    cd ml-service && uvicorn app:app --reload --port 8000   # separate terminal
    cd failure-lab && python3 scenarios.py

For each scenario: DETECT -> DECIDE -> SAFE RESPONSE -> AUDIT.
"""
import requests

BASE_URL = "http://localhost:8000"

BASE_PAYLOAD = {
    "merchant_category": "ecommerce", "customer_value_tier": "mid",
    "event_type": "payment_attempt", "payment_method": "card",
    "decline_code": "network_error", "amount": 1500,
    "customer_retry_propensity_observed": 0.5, "merchant_baseline_fail_rate": 0.2,
    "prior_failure_count": 0, "minutes_since_last_failure": -1,
    "prior_recovery_count": 0, "prior_recovery_rate": -1,
    "recent_method_switch_count": 0, "retry_count": 0,
    "nudges_today": 0, "discount_pct": 0,
}


def payload(event_id, **overrides):
    p = dict(BASE_PAYLOAD)
    p["event_id"] = event_id
    p.update(overrides)
    return p


SCENARIOS = [
    ("1. Duplicate event replay", payload("dup-1", decline_code="network_error"),
     "Same event_id sent twice; audit log should show two entries -- flagged for dedup review in a full deployment."),

    ("2. Missing / unseen customer tier", payload("missing-1", customer_value_tier="unknown"),
     "Should still produce a full decision trace without crashing; lower confidence expected."),

    ("3. Unseen/new merchant category", payload("unseen-merchant-1", merchant_category="crypto_exchange"),
     "Category not in training data -- one-hot encoding degrades to all-zero for that column; system should not crash, confidence should be lower."),

    ("4. Concept drift context (issuer-outage-like)", payload("drift-1", decline_code="network_error", merchant_baseline_fail_rate=0.35),
     "High network_error rate resembling the injected drift window; drift monitor (rolling PSI) should trend toward MODERATE/SIGNIFICANT as more such events arrive."),

    ("5. OOD customer (implausible combination)", payload("ood-1", customer_retry_propensity_observed=0.99, prior_failure_count=50, prior_recovery_rate=0.02),
     "Extreme, rarely-seen feature combination -- bootstrap ensemble disagreement should be high -> LOW confidence -> escalate."),

    ("6. Conflicting signals (high value + high risk)", payload("conflict-1", customer_value_tier="high", decline_code="risk_flagged", amount=8000),
     "High-value customer but risk-flagged decline; net value may look attractive but amount likely exceeds human-approval threshold anyway."),

    ("7. High-value transaction (policy threshold)", payload("highvalue-1", amount=45000, customer_value_tier="high"),
     "Amount exceeds HUMAN_APPROVAL_AMOUNT_THRESHOLD -> must require human review regardless of model confidence."),

    ("8. Low-confidence prediction (thin history)", payload("lowconf-1", prior_failure_count=0, prior_recovery_rate=-1, customer_retry_propensity_observed=0.5),
     "No prior history for this customer -- ensemble std should be relatively high -> MEDIUM/LOW tier, conservative or escalated action."),

    ("9. Policy-restricted action (oversized discount request)", payload("discount-abuse-1", decline_code="insufficient_funds", customer_value_tier="low", discount_pct=40),
     "If discount_offer is the top net-value arm, policy engine must reject it (exceeds 15% cap) even though the model likes it."),

    ("10. Repeated failed retries (retry cap)", payload("retry-exhausted-1", decline_code="insufficient_funds", retry_count=3, prior_failure_count=3),
     "retry_count already at cap -- if retry_timing is selected, policy engine must reject it and require human review."),

    ("11. Daily nudge cap already reached", payload("nudge-cap-1", decline_code="issuer_declined", nudges_today=2),
     "If a nudge-type arm (alt_method_nudge / hinglish_voice_nudge) is chosen, policy engine must reject it (daily cap reached)."),

    ("12. API/service failure (simulated)", None,
     "Backend calling an unreachable ml-service should surface a clear 502, not a silent failure or a crash -- verified separately via backend/server.js's error handling, not exercised here."),
]


def run():
    for name, p, expectation in SCENARIOS:
        print("=" * 70)
        print(name)
        print(f"Expectation: {expectation}")
        if p is None:
            print("(Scenario 12 is verified via the backend's /api/decide error handling, not a direct ml-service call.)")
            continue
        try:
            r = requests.post(f"{BASE_URL}/decide", json=p, timeout=5)
            data = r.json()
            print(f"Status: {r.status_code}")
            print(f"  chosen_intervention: {data.get('chosen_intervention')}")
            print(f"  decision_reason: {data.get('decision_reason')}")
            print(f"  confidence_tier: {data.get('uncertainty', {}).get('confidence_tier')}")
            print(f"  drift_status: {data.get('drift', {}).get('status')}")
            print(f"  policy_approved: {data.get('policy_approved')}  requires_human_review: {data.get('requires_human_review')}")
        except requests.exceptions.ConnectionError:
            print("!! ml-service not running. Start it with:")
            print("   cd ml-service && uvicorn app:app --reload --port 8000")
            return

    # Send scenario 1 a second time to demonstrate duplicate detection in the audit log
    requests.post(f"{BASE_URL}/decide", json=payload("dup-1", decline_code="network_error"), timeout=5)

    print("=" * 70)
    print("Failure Lab complete. Inspect ml-service/audit/audit_log.jsonl or GET /audit for the full trace.")


if __name__ == "__main__":
    run()
