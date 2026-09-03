"""
Recoup synthetic data generator.

Produces two files with a strict separation of concerns:

  events.csv
    The TRAINING data. One row per event, with exactly ONE observed
    outcome (the result of whichever arm the historical logging policy
    actually chose). All "prior_*" / "minutes_since_last_failure" /
    "recent_method_switch_count" columns are computed INCREMENTALLY as
    we iterate through customers in time order -- each row's temporal
    features are written using only the history dict's state BEFORE
    that customer's history is updated with the current event. This is
    what features.py's docstring and
    ml-service/test_no_leakage.py::test_temporal_features_are_causally_prior
    check: the first-ever event for any customer must show all
    prior_* == 0, because at generation time we literally have not
    updated that customer's history dict yet.

  oracle_potential_outcomes.csv
    The COUNTERFACTUAL ORACLE. For every failed event we also compute
    the potential outcome under EVERY possible arm (including
    "no_action"), using the same latent generative process. This file
    is used ONLY by evaluation/*.py for offline counterfactual scoring
    of policies -- never joined into training features. See
    ml-service/test_no_leakage.py::test_oracle_file_not_importable_by_training_code.

OBSERVATIONAL BIAS: the historical logging policy (assign_historical_intervention)
is NOT random -- each merchant has a habitual "default" arm it over-uses,
and decline-code / customer-tier nudge the choice further. This means
naive supervised P(recovered | X, A) learning on the logged data alone
conflates "this action worked" with "this context always recovers" --
which is the concrete reason the uplift/T-learner layer (ml-service/uplift.py)
exists rather than a single classifier.
"""
import csv
import random
from datetime import datetime, timedelta

ARMS = ["no_action", "retry_timing", "alt_method_nudge", "discount_offer",
        "human_escalation", "hinglish_voice_nudge"]
ACTIONABLE_ARMS = [a for a in ARMS if a != "no_action"]

N_MERCHANTS = 40
N_CUSTOMERS = 2000
N_EVENTS = 12000
DRIFT_START_FRAC = 0.55
DRIFT_END_FRAC = 0.70

PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]
DECLINE_CODES = [
    "insufficient_funds", "issuer_declined", "expired_card",
    "risk_flagged", "network_error", "invalid_otp", "limit_exceeded",
]
MERCHANT_CATEGORIES = ["ecommerce", "subscription", "b2b_invoice", "food_delivery", "edtech"]


def make_merchants(rng):
    merchants = []
    for i in range(N_MERCHANTS):
        merchants.append({
            "merchant_id": f"M{i:03d}",
            "category": rng.choice(MERCHANT_CATEGORIES),
            "baseline_fail_rate": round(rng.uniform(0.05, 0.35), 3),
            "logging_policy_bias": rng.choice(ACTIONABLE_ARMS),
            "recovery_profile": round(rng.uniform(0.7, 1.3), 2),
        })
    return merchants


def make_customers(rng):
    customers = []
    for i in range(N_CUSTOMERS):
        customers.append({
            "customer_id": f"C{i:04d}",
            "retry_propensity_true": round(rng.uniform(0.1, 0.9), 3),
            "value_tier": rng.choices(["low", "mid", "high"], weights=[0.6, 0.3, 0.1])[0],
            "repeat_high_value": rng.random() < 0.05,
            "price_sensitivity": round(rng.uniform(0, 1), 3),
            "intervention_fatigue": 0.0,
        })
    return customers


def pick_decline_code(rng, drift_active):
    if drift_active:
        return rng.choices(DECLINE_CODES, weights=[10, 35, 5, 8, 30, 6, 6])[0]
    return rng.choices(DECLINE_CODES, weights=[20, 15, 10, 8, 8, 10, 10])[0]


def potential_outcome_prob(rng, customer, merchant, arm, decline_code):
    """Latent causal ground truth: TRUE P(recovered | X, do(arm))."""
    if arm == "no_action":
        base = 0.06 + customer["retry_propensity_true"] * 0.05
        return max(0.01, min(0.95, base + rng.gauss(0, 0.02)))

    base = 0.15
    if decline_code == "insufficient_funds" and arm == "retry_timing":
        base += 0.05
    if decline_code in ("network_error", "issuer_declined") and arm == "alt_method_nudge":
        base += 0.15
    if customer["value_tier"] == "high" and arm == "human_escalation":
        base += 0.25
    if customer["value_tier"] == "low" and arm == "discount_offer":
        base += 0.10 * (1 + customer["price_sensitivity"])
    if arm == "hinglish_voice_nudge" and customer["retry_propensity_true"] > 0.6:
        base += 0.12
    base += customer["retry_propensity_true"] * 0.2
    base -= merchant["baseline_fail_rate"] * 0.1
    base *= merchant["recovery_profile"]
    if arm in ("alt_method_nudge", "hinglish_voice_nudge"):
        base -= customer["intervention_fatigue"] * 0.15
    return max(0.01, min(0.95, base + rng.gauss(0, 0.05)))


def assign_historical_intervention(rng, customer, merchant, decline_code):
    """The (biased, non-random) historical logging policy. Returns the
    chosen arm. Occasionally logs 'no_action' (event was never actually
    worked)."""
    weights = {a: 1.0 for a in ACTIONABLE_ARMS}
    weights[merchant["logging_policy_bias"]] += 3.0
    if decline_code in ("network_error", "issuer_declined"):
        weights["alt_method_nudge"] += 1.5
    if customer["value_tier"] == "high":
        weights["human_escalation"] += 1.0
    # small chance the event was simply never actioned historically
    if rng.random() < 0.06:
        return "no_action"
    total = sum(weights.values())
    r = rng.random() * total
    cum = 0.0
    for a in ACTIONABLE_ARMS:
        cum += weights[a]
        if r <= cum:
            return a
    return ACTIONABLE_ARMS[-1]


def generate(seed=42, n_events=N_EVENTS):
    rng = random.Random(seed)
    merchants = make_merchants(rng)
    customers = make_customers(rng)
    merchant_by_id = {m["merchant_id"]: m for m in merchants}
    customer_by_id = {c["customer_id"]: c for c in customers}

    # per-customer incremental history -- read BEFORE this event is
    # generated, updated AFTER. This is the mechanism that guarantees
    # prior_* features cannot see the current or any future event.
    history = {
        c["customer_id"]: {
            "prior_failures": 0, "prior_recoveries": 0,
            "last_failure_time": None, "methods_seen": [],
        }
        for c in customers
    }

    start_time = datetime(2026, 6, 1)
    # pre-generate a time-ordered stream of (customer, timestamp) draws so
    # that iterating in this order == iterating in true chronological
    # order per customer, which is what makes the incremental history
    # dict leakage-safe without a separate sort-then-recompute pass.
    draws = []
    for i in range(n_events):
        customer = rng.choice(customers)
        merchant = rng.choice(merchants)
        ts = start_time + timedelta(minutes=rng.randint(0, 60 * 24 * 60))
        draws.append((ts, customer["customer_id"], merchant["merchant_id"]))
    draws.sort(key=lambda d: d[0])

    observed_rows = []
    oracle_rows = []

    for idx, (ts, cust_id, merch_id) in enumerate(draws):
        frac = idx / n_events
        drift_active = DRIFT_START_FRAC <= frac <= DRIFT_END_FRAC

        customer = customer_by_id[cust_id]
        merchant = merchant_by_id[merch_id]
        h = history[cust_id]

        amount = round(rng.lognormvariate(6.5, 1.0), 2)
        method = rng.choice(PAYMENT_METHODS)
        missing_field = rng.random() < 0.03

        fails = rng.random() < (merchant["baseline_fail_rate"] + (0.15 if drift_active else 0))
        decline_code = pick_decline_code(rng, drift_active) if fails else ""

        event_type = (
            "invoice_overdue" if merchant["category"] == "b2b_invoice" and rng.random() < 0.3
            else ("checkout_abandoned" if rng.random() < 0.3 else "payment_attempt")
        )

        # ---- temporal features, read from history BEFORE update ----
        prior_failures = h["prior_failures"]
        prior_recoveries = h["prior_recoveries"]
        prior_recovery_rate = (prior_recoveries / prior_failures) if prior_failures > 0 else -1
        minutes_since_last_failure = (
            (ts - h["last_failure_time"]).total_seconds() / 60 if h["last_failure_time"] else -1
        )
        recent_method_switch_count = sum(
            1 for j in range(1, len(h["methods_seen"]))
            if h["methods_seen"][j] != h["methods_seen"][j - 1]
        )
        # observed proxy of the latent retry propensity -- noisy, not the
        # exact latent value, standing in for what a real system would
        # estimate from behavior rather than know directly
        retry_propensity_observed = max(0.0, min(1.0, customer["retry_propensity_true"] + rng.gauss(0, 0.05)))

        event_id = f"E{idx:06d}"
        row = {
            "event_id": event_id,
            "timestamp": ts.isoformat(),
            "merchant_id": merchant["merchant_id"],
            "merchant_category": merchant["category"],
            "customer_id": customer["customer_id"],
            "customer_value_tier": "" if missing_field else customer["value_tier"],
            "customer_retry_propensity_observed": round(retry_propensity_observed, 4),
            "merchant_baseline_fail_rate": merchant["baseline_fail_rate"],
            "repeat_high_value_flag": customer["repeat_high_value"],
            "event_type": event_type,
            "amount": amount,
            "payment_method": method,
            "failed": fails,
            "decline_code": decline_code,
            "prior_failure_count": prior_failures,
            "prior_recovery_count": prior_recoveries,
            "prior_recovery_rate": round(prior_recovery_rate, 4),
            "minutes_since_last_failure": round(minutes_since_last_failure, 2),
            "recent_method_switch_count": recent_method_switch_count,
            "drift_window": drift_active,
        }

        if fails:
            oracle_row = {"event_id": event_id}
            for arm in ARMS:
                p = potential_outcome_prob(rng, customer, merchant, arm, decline_code)
                recovered = rng.random() < p
                oracle_row[f"potential_prob_{arm}"] = round(p, 4)
                oracle_row[f"potential_recovered_{arm}"] = int(recovered)
            oracle_rows.append(oracle_row)

            chosen = assign_historical_intervention(rng, customer, merchant, decline_code)
            observed_recovered = bool(oracle_row[f"potential_recovered_{chosen}"])
            observed_value = amount if observed_recovered else 0.0

            row.update({
                "chosen_intervention": chosen,
                "recovered": observed_recovered,
                "recovered_value": observed_value,
            })

            if chosen in ("alt_method_nudge", "hinglish_voice_nudge"):
                customer["intervention_fatigue"] = min(1.0, customer["intervention_fatigue"] + 0.2)
            else:
                customer["intervention_fatigue"] = max(0.0, customer["intervention_fatigue"] - 0.05)

            # ---- update history AFTER this event is fully written ----
            h["prior_failures"] += 1
            if observed_recovered:
                h["prior_recoveries"] += 1
            h["last_failure_time"] = ts
        else:
            row.update({"chosen_intervention": "", "recovered": False, "recovered_value": 0.0})

        h["methods_seen"].append(method)
        if len(h["methods_seen"]) > 10:
            h["methods_seen"].pop(0)

        observed_rows.append(row)

    return observed_rows, oracle_rows


def write_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    import sys
    seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    observed, oracle = generate(seed=seed_arg)
    write_csv(observed, "events.csv")
    write_csv(oracle, "oracle_potential_outcomes.csv")
    print(
        "\nNOTE: oracle_potential_outcomes.csv holds hidden counterfactual "
        "outcomes for ALL arms. It is read only by evaluation/*.py for "
        "offline policy scoring -- ml-service/train.py never imports it "
        "(enforced by ml-service/test_no_leakage.py)."
    )
