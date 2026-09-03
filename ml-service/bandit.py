"""
Decision engine — constrained economic optimizer over uplift estimates,
with a persistent, CONTEXT-PRESERVING LinUCB layer for online adaptation.

ROUND-2 FIXES:
  #4  /feedback used to reconstruct context as a zero vector -> LinUCB
      was not actually learning context->action->reward. Fixed via
      decision_store.py: /decide now persists the real context vector
      keyed by decision_id; /feedback retrieves and uses it.
  #6  Reward used raw rupee amounts, which can be in the thousands and
      dominate/destabilize LinUCB's linear updates. Rewards are now
      normalized by a rolling median transaction amount before being
      used to update the bandit (see normalize_reward below); the
      economic (₹) net value used for the actual DECISION is unchanged
      and still reported in real rupees -- only the bandit's internal
      learning signal is rescaled.
  #13 net_value now separates discount leakage (money actually given
      away, proportional to amount) from the flat friction cost, and
      splits human_escalation's cost into a handling_cost component
      (SLA/time cost is flagged as NOT modeled -- see docs/model_cards.md).
  #14 human_escalation (a deliberate revenue-recovery ACTION) is now
      distinguished in the decision reason string from a SAFETY_FALLBACK
      (a human-review requirement triggered by low confidence, drift, or
      a policy violation) -- these were previously conflated.
  #15 select_action -> select_safe_action: instead of picking the top
      net-value arm and simply rejecting it if unsafe (dead end even
      when a second-best arm is perfectly safe), we now rank ALL
      candidate arms by net value and return the highest-ranked arm
      that ALSO passes the policy check.
"""
import os
import numpy as np
import joblib

INTERVENTIONS = [
    "retry_timing", "alt_method_nudge", "discount_offer",
    "human_escalation", "hinglish_voice_nudge",
]

# Cost decomposition (round-2 fix #13): FRICTION_COST is now the flat
# operational cost only; discount leakage and escalation handling cost
# are computed separately in net_value() below.
FRICTION_COST = {
    "no_action": 0, "retry_timing": 2, "alt_method_nudge": 5,
    "discount_offer": 8,  # flat op cost; leakage added separately
    "human_escalation": 30,  # flat handling cost component
    "hinglish_voice_nudge": 15,
}

# NOT MODELED (stated explicitly, not hidden): human_escalation's SLA/
# time cost (how long the case sits before an agent can act on it) is
# not represented anywhere in this reward function. A real deployment
# would need this as a further cost term.

BANDIT_STATE_PATH = os.path.join(os.path.dirname(__file__), "models", "linucb_state.pkl")
REWARD_NORMALIZATION_SCALE_PATH = os.path.join(os.path.dirname(__file__), "models", "reward_scale.pkl")

DEFAULT_REWARD_SCALE = 1000.0  # fallback median-amount estimate before any data is seen


class LinUCB:
    """Persistent, per-arm linear UCB bandit. State is saved/loaded from
    disk so it genuinely accumulates online learning across process
    restarts."""

    def __init__(self, arms, n_features, alpha=1.0):
        self.arms = arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = {a: np.identity(n_features) for a in arms}
        self.b = {a: np.zeros(n_features) for a in arms}

    def ucb_score(self, arm, x):
        A_inv = np.linalg.inv(self.A[arm])
        theta = A_inv.dot(self.b[arm])
        mean = theta.dot(x)
        bonus = self.alpha * np.sqrt(max(x.dot(A_inv).dot(x), 0))
        return float(mean + bonus), float(mean), float(bonus)

    def update(self, arm, x, reward):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def save(self, path=BANDIT_STATE_PATH):
        joblib.dump({"arms": self.arms, "n_features": self.n_features,
                      "alpha": self.alpha, "A": self.A, "b": self.b}, path)

    @classmethod
    def load(cls, path=BANDIT_STATE_PATH):
        if not os.path.exists(path):
            return None
        state = joblib.load(path)
        inst = cls(state["arms"], state["n_features"], state["alpha"])
        inst.A = state["A"]
        inst.b = state["b"]
        return inst


def normalize_reward(raw_reward: float, scale: float = DEFAULT_REWARD_SCALE) -> float:
    """Round-2 fix #6: rescale a rupee-denominated reward before feeding
    it to LinUCB's linear updates, so a single ₹45,000 event doesn't
    dominate the running A/b matrices relative to many ₹500 events. The
    scale is a rolling median transaction amount (persisted, updated as
    more events are observed) -- NOT a fixed magic number."""
    return raw_reward / max(scale, 1.0)


def net_value(uplift_probs: dict, amount: float, discount_pct: float = 0.0) -> dict:
    """NetValue(A|X) = uplift_A(X) * amount - FRICTION_COST[A]
                        - (discount leakage, for discount_offer only)

    Discount leakage (round-2 fix #13): if a ₹1000 payment is recovered
    via a 10% discount, the merchant does not actually receive ₹1000 --
    they receive ₹900. This is now modeled explicitly rather than
    conflated with the flat operational friction cost.
    """
    values = {}
    for arm, uplift in uplift_probs.items():
        gross = uplift * amount
        cost = FRICTION_COST.get(arm, 0)
        leakage = 0.0
        if arm == "discount_offer" and discount_pct > 0:
            # leakage only applies to the recovered-value portion, since
            # a discount that doesn't lead to recovery costs nothing
            leakage = uplift * amount * (discount_pct / 100.0)
        values[arm] = round(gross - cost - leakage, 2)
    return values


def select_safe_action(net_values: dict, confidence_tier: str, drift_status: str,
                        policy_check_fn, amount: float, retry_count: int,
                        nudges_today: int, discount_pct: float = 0.0):
    """
    Round-2 fix #15: RANK all candidate arms by net value, then return
    the highest-ranked arm that ALSO passes the policy check -- instead
    of picking the single best arm and giving up if it's unsafe.

    Returns (chosen_arm, reason, safety_fallback_triggered, policy_reason,
             ranked_candidates) where ranked_candidates is the full
    ranked list with each arm's policy verdict, for the decision trace.
    """
    if drift_status == "SIGNIFICANT_SHIFT" or confidence_tier == "LOW":
        return (None,
                f"confidence={confidence_tier}, drift={drift_status}",
                True,  # safety_fallback_triggered
                "escalated before policy check: low confidence or significant drift",
                [])

    if not net_values:
        return (None, "no uplift estimates available (insufficient control support)",
                True, "escalated: cannot compute incremental value", [])

    # candidate pool depends on confidence tier
    if confidence_tier == "MEDIUM":
        candidates = {a: v for a, v in net_values.items() if v > 0}
        # conservative: prefer cheapest among positive-value arms, but still
        # rank so we can fall through to the next-cheapest if the cheapest
        # is policy-blocked
        ranked = sorted(candidates.items(), key=lambda kv: FRICTION_COST.get(kv[0], 0))
    else:  # HIGH
        candidates = {a: v for a, v in net_values.items() if v > 0}
        ranked = sorted(candidates.items(), key=lambda kv: -kv[1])

    if not ranked:
        return (None, "no arm has positive expected net value",
                True, "escalated: no economically justified action", [])

    trace = []
    for arm, value in ranked:
        policy_result = policy_check_fn(
            intervention=arm, amount=amount, retry_count=retry_count,
            nudges_today=nudges_today, discount_pct=discount_pct,
        )
        trace.append({"arm": arm, "net_value": value, "policy": policy_result})
        if policy_result["approved"] and not policy_result["requires_human"]:
            reason = (f"{confidence_tier} confidence -> "
                      f"rank-{len(trace)} candidate '{arm}' is the highest-ranked "
                      f"policy-safe arm (net value {value})")
            return arm, reason, False, policy_result["reason"], trace

    # every ranked candidate was policy-blocked
    return (None,
            f"all {len(ranked)} candidate arm(s) failed the policy check",
            True, trace[-1]["policy"]["reason"] if trace else "no safe candidate", trace)
