"""
Policy / Constraint Engine.

Every proposed action from the bandit passes through here before
"execution". This is the safety layer: hard caps that no model score
can override. Nothing here moves real money — everything downstream
of an approved action is simulation only.
"""

MAX_DISCOUNT_PCT = 15          # never offer more than 15% off
MAX_RETRIES_PER_EVENT = 3
MAX_NUDGES_PER_CUSTOMER_PER_DAY = 2
HUMAN_APPROVAL_AMOUNT_THRESHOLD = 5000  # ₹ — above this, always require human sign-off


class PolicyViolation(Exception):
    pass


def check_action(intervention: str, amount: float, retry_count: int,
                  nudges_today: int, discount_pct: float = 0) -> dict:
    """
    Returns a dict: {"approved": bool, "reason": str, "requires_human": bool}
    Raises nothing — a rejected action is a normal, expected outcome, not
    an error. The decision engine must handle "approved: False" gracefully
    (fall back to next-best action or escalate).
    """
    if intervention == "discount_offer" and discount_pct > MAX_DISCOUNT_PCT:
        return {"approved": False, "reason": f"discount {discount_pct}% exceeds cap {MAX_DISCOUNT_PCT}%",
                "requires_human": False}

    if intervention == "retry_timing" and retry_count >= MAX_RETRIES_PER_EVENT:
        return {"approved": False, "reason": "retry cap reached", "requires_human": True}

    if nudges_today >= MAX_NUDGES_PER_CUSTOMER_PER_DAY and intervention in (
        "alt_method_nudge", "hinglish_voice_nudge"
    ):
        return {"approved": False, "reason": "daily nudge cap reached for this customer",
                "requires_human": False}

    if amount >= HUMAN_APPROVAL_AMOUNT_THRESHOLD:
        return {"approved": True, "reason": "approved, but above human-approval threshold",
                "requires_human": True}

    return {"approved": True, "reason": "within policy bounds", "requires_human": False}
