"""
Evaluation Protocol (round-3 fix, Phase 1) — the ONE canonical definition
of reward, cost, regret, and oracle action. Every evaluation script
(evaluate.py, ablation.py, offline_policy_eval.py) imports FROM HERE and
computes nothing independently. This exists specifically because round-2
had a real contradiction: evaluate.py reported Full Recoup at ₹251.11/event
while ablation.py reported the same "Full Recoup" at ₹208.11/event, purely
because the two scripts scored an escalated case differently. That
contradiction is now structurally impossible, because there is exactly
one function that scores an event, and every script calls it.

MATHEMATICAL DEFINITION

For event x with true amount `amount` and chosen action a:

  gross_recovered(x, a) = amount   if the event recovers under a, else 0
  discount_leakage(x, a) = gross_recovered(x, a) * (discount_pct / 100)
                            if a == "discount_offer", else 0
  intervention_cost(a)   = flat operational cost, see COST_TABLE below
  net_revenue(x, a)      = gross_recovered(x, a)
                            - discount_leakage(x, a)
                            - intervention_cost(a)

If the chosen action is None (escalated to a human, or no economically
justified action exists): the event is scored at the EMPIRICAL HISTORICAL
human-agent recovery rate (not ₹0 — a human-worked case is not lost
revenue, and not the oracle-best value either — it is scored at what a
human agent has actually achieved historically, which is the honest,
auditable middle ground). This is the ESCALATION_CREDIT_RULE and it is
now the ONLY place this logic is implemented.

REGRET

  oracle_action(x)  = argmax_a net_revenue(x, a)  over ALL actions
                       (using the ORACLE's true potential outcome, never
                        the model's estimate)
  regret(x, a)      = net_revenue(x, oracle_action(x)) - net_revenue(x, a)

  (with the same ESCALATION_CREDIT_RULE applied if a is None)

COST TABLE (round-2 Phase 13, unchanged, restated here as the single
source of truth)
"""
import os
import pandas as pd

COST_TABLE = {
    "no_action": 0,
    "retry_timing": 2,
    "alt_method_nudge": 5,
    "discount_offer": 8,       # flat op cost; leakage computed separately
    "human_escalation": 30,    # flat handling cost; SLA/time cost NOT modeled (stated, not hidden)
    "hinglish_voice_nudge": 15,
}

ALL_ACTIONABLE_ARMS = ["retry_timing", "alt_method_nudge", "discount_offer",
                        "human_escalation", "hinglish_voice_nudge"]


def historical_escalation_credit_rate(train_df: pd.DataFrame) -> float:
    """The ONE place this number is computed. Empirical recovery rate for
    human_escalation in the logged (training) data — used as the reward
    credit for any event where the final decision is 'no action taken /
    escalated', since a human-worked case is not the same as a lost one."""
    failed = train_df[train_df["failed"] == True]
    human = failed[failed["chosen_intervention"] == "human_escalation"]
    if len(human) == 0:
        return 0.0
    return float(human["recovered"].mean())


def net_revenue_oracle(row, action, discount_pct: float = 0.0) -> float:
    """The ONE reward function every script must call. Uses the ORACLE's
    ground-truth potential outcome for `action` (never the model's own
    prediction) — this is what makes offline evaluation honest."""
    if action is None:
        return None  # caller must apply escalation_credit_reward instead
    col = f"potential_recovered_{action}"
    if col not in row or pd.isna(row[col]):
        return 0.0
    recovered = bool(row[col])
    gross = row["amount"] if recovered else 0.0
    leakage = gross * (discount_pct / 100.0) if action == "discount_offer" else 0.0
    cost = COST_TABLE.get(action, 0)
    return gross - leakage - cost


def escalation_credit_reward(row, escalation_rate: float) -> float:
    """The ONE place an escalated/no-action event gets scored."""
    return escalation_rate * row["amount"] - COST_TABLE.get("human_escalation", 30)


def score_event(row, action, escalation_rate: float, discount_pct: float = 0.0) -> float:
    """The single entry point every evaluation script must call to score
    one event under one chosen action. Handles the None (escalated) case
    consistently -- this function existing is what makes the evaluate.py
    vs ablation.py contradiction structurally impossible going forward."""
    if action is None:
        return escalation_credit_reward(row, escalation_rate)
    return net_revenue_oracle(row, action, discount_pct)


def oracle_optimal_action(row, arms=None):
    """The ONE definition of the oracle-best action, over ALL actionable
    arms (never including no_action as a competitor for regret purposes --
    regret asks 'given we should act, did we pick the best act', matching
    how score_event's None-branch is scored separately via escalation
    credit rather than compared arm-for-arm against no_action's oracle
    value, which would conflate two different questions)."""
    arms = arms or ALL_ACTIONABLE_ARMS
    best_arm, best_reward = None, -1e18
    for arm in arms:
        col = f"potential_recovered_{arm}"
        if col not in row or pd.isna(row[col]):
            continue
        reward = net_revenue_oracle(row, arm)
        if reward > best_reward:
            best_reward = reward
            best_arm = arm
    return best_arm, best_reward


def oracle_ranked_actions(row, arms=None):
    arms = arms or ALL_ACTIONABLE_ARMS
    scored = []
    for arm in arms:
        col = f"potential_recovered_{arm}"
        if col not in row or pd.isna(row[col]):
            continue
        scored.append((arm, net_revenue_oracle(row, arm)))
    return [a for a, _ in sorted(scored, key=lambda kv: -kv[1])]


def regret(row, chosen_action, escalation_rate: float, discount_pct: float = 0.0) -> float:
    """regret = oracle-optimal reward for this event minus the reward the
    system's chosen action actually achieved (with the same
    escalation-credit rule applied on both sides where relevant)."""
    _, oracle_reward = oracle_optimal_action(row)
    achieved = score_event(row, chosen_action, escalation_rate, discount_pct)
    return oracle_reward - achieved


def amount_bucket(amount: float) -> str:
    if amount <= 500:
        return "low"
    if amount <= 1500:
        return "mid"
    if amount <= 5000:
        return "high"
    return "very_high"
