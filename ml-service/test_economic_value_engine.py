"""
Economic value engine unit tests (final sprint, Step 3).

Directly tests net_value(a) = uplift(a) x amount - cost(a) in isolation,
with known inputs and hand-computed expected outputs -- not just
observed behind an end-to-end pipeline. This is what "audit the
economic engine" means concretely: pin down the arithmetic itself.
"""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from bandit import net_value, FRICTION_COST


def test_zero_uplift_gives_negative_cost_only():
    """An arm with zero uplift should show net value = -cost exactly
    (no recovered revenue, but the operational cost is still paid)."""
    result = net_value({"retry_timing": 0.0}, amount=1000)
    expected = -FRICTION_COST["retry_timing"]
    assert result["retry_timing"] == expected, f"expected {expected}, got {result['retry_timing']}"
    print(f"PASS: zero uplift -> net_value = -cost ({expected})")


def test_no_action_has_zero_cost_and_zero_gross():
    """no_action should net to exactly (uplift * amount) since its cost
    is 0 by definition -- there is no operational cost to doing nothing."""
    result = net_value({"no_action": 0.1}, amount=1000)
    expected = round(0.1 * 1000 - FRICTION_COST["no_action"], 2)
    assert result["no_action"] == expected == 100.0, f"expected 100.0, got {result['no_action']}"
    print(f"PASS: no_action net value = uplift*amount exactly (cost=0): {result['no_action']}")


def test_discount_leakage_deducted_only_from_recovered_portion():
    """A 20% discount on a ₹1000 payment with 0.5 uplift should net:
    gross = 0.5*1000 = 500
    cost = FRICTION_COST['discount_offer']
    leakage = 0.5*1000*0.20 = 100  (leakage applies to recovered value only)
    net = 500 - cost - 100
    """
    result = net_value({"discount_offer": 0.5}, amount=1000, discount_pct=20)
    cost = FRICTION_COST["discount_offer"]
    expected = round(500 - cost - 100, 2)
    assert result["discount_offer"] == expected, f"expected {expected}, got {result['discount_offer']}"
    print(f"PASS: discount leakage correctly computed on recovered portion only: {result['discount_offer']}")


def test_discount_offer_without_discount_pct_has_no_leakage():
    """If discount_pct is not supplied (0), discount_offer should behave
    like any other arm -- gross minus flat cost, no leakage term."""
    result = net_value({"discount_offer": 0.4}, amount=1000, discount_pct=0)
    cost = FRICTION_COST["discount_offer"]
    expected = round(0.4 * 1000 - cost, 2)
    assert result["discount_offer"] == expected, f"expected {expected}, got {result['discount_offer']}"
    print(f"PASS: discount_offer with discount_pct=0 has zero leakage: {result['discount_offer']}")


def test_human_escalation_cost_deducted():
    result = net_value({"human_escalation": 0.3}, amount=2000)
    expected = round(0.3 * 2000 - FRICTION_COST["human_escalation"], 2)
    assert result["human_escalation"] == expected, f"expected {expected}, got {result['human_escalation']}"
    print(f"PASS: human_escalation flat handling cost correctly deducted: {result['human_escalation']}")


def test_amount_scaling_is_linear():
    """Doubling the amount should double the gross component exactly,
    while cost stays fixed (cost is a flat operational fee, not a
    percentage of amount) -- net value should NOT simply double."""
    small = net_value({"retry_timing": 0.3}, amount=1000)["retry_timing"]
    large = net_value({"retry_timing": 0.3}, amount=2000)["retry_timing"]
    cost = FRICTION_COST["retry_timing"]
    expected_small = round(0.3 * 1000 - cost, 2)
    expected_large = round(0.3 * 2000 - cost, 2)
    assert small == expected_small and large == expected_large, f"got small={small}, large={large}"
    # explicitly confirm it does NOT double (since cost is flat, not scaled)
    assert large != 2 * small, "net value should not scale linearly with amount because cost is flat, not proportional"
    print(f"PASS: amount scaling correct -- gross scales linearly, cost stays flat "
          f"(small={small}, large={large}, NOT {2*small})")


def test_negative_uplift_gives_negative_net_value():
    """A negative uplift (this action is estimated to make things WORSE
    than doing nothing) should net to something more negative than just
    -cost -- proving negative uplift correctly drags the value down
    rather than being clamped to zero."""
    result = net_value({"retry_timing": -0.2}, amount=1000)
    expected = round(-0.2 * 1000 - FRICTION_COST["retry_timing"], 2)
    assert result["retry_timing"] == expected, f"expected {expected}, got {result['retry_timing']}"
    assert result["retry_timing"] < -FRICTION_COST["retry_timing"], "negative uplift must make net value worse than -cost alone"
    print(f"PASS: negative uplift correctly produces net value below -cost: {result['retry_timing']}")


def test_nan_uplift_is_rejected_or_propagates_visibly():
    """A NaN uplift must not silently become a valid-looking number --
    either the function should raise, or the result must itself be NaN
    (so a caller's own NaN-check catches it), never a plausible-looking
    finite value that hides the corruption."""
    result = net_value({"retry_timing": float("nan")}, amount=1000)
    val = result["retry_timing"]
    assert math.isnan(val), f"expected NaN to propagate visibly, got {val} instead"
    print(f"PASS: NaN uplift correctly propagates as NaN in net_value (not silently coerced to a plausible number)")


def test_zero_amount_gives_cost_only():
    result = net_value({"retry_timing": 0.5}, amount=0)
    expected = -FRICTION_COST["retry_timing"]
    assert result["retry_timing"] == expected, f"expected {expected}, got {result['retry_timing']}"
    print(f"PASS: zero amount -> net value = -cost exactly (no gross recovery possible): {result['retry_timing']}")


def test_multiple_arms_ranked_correctly_by_net_value():
    """Sanity check that ranking by net_value (not raw uplift) can flip
    the order vs ranking by raw uplift alone, when costs differ enough --
    this is the entire justification for a cost-aware policy over a
    naive highest-uplift policy."""
    uplifts = {"retry_timing": 0.10, "human_escalation": 0.12}
    result = net_value(uplifts, amount=200)
    # human_escalation has higher uplift (0.12 > 0.10) but a much higher
    # flat cost (30 vs 2), so on a SMALL transaction, retry_timing should
    # win on net value despite lower raw uplift
    assert result["retry_timing"] > result["human_escalation"], (
        f"expected retry_timing ({result['retry_timing']}) to beat human_escalation "
        f"({result['human_escalation']}) on a small transaction despite lower raw uplift"
    )
    print(f"PASS: cost-aware ranking correctly flips vs raw-uplift ranking on a small transaction "
          f"(retry_timing={result['retry_timing']} > human_escalation={result['human_escalation']})")


TESTS = [
    test_zero_uplift_gives_negative_cost_only,
    test_no_action_has_zero_cost_and_zero_gross,
    test_discount_leakage_deducted_only_from_recovered_portion,
    test_discount_offer_without_discount_pct_has_no_leakage,
    test_human_escalation_cost_deducted,
    test_amount_scaling_is_linear,
    test_negative_uplift_gives_negative_net_value,
    test_nan_uplift_is_rejected_or_propagates_visibly,
    test_zero_amount_gives_cost_only,
    test_multiple_arms_ranked_correctly_by_net_value,
]

if __name__ == "__main__":
    failures = []
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL: {t.__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)}/{len(TESTS)} economic engine tests FAILED")
    else:
        print(f"ALL {len(TESTS)}/{len(TESTS)} economic engine tests PASSED")
