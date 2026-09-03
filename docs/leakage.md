# Recoup — Leakage Prevention

**The question a judge will ask: "What information is available at decision time, and how do you know nothing from the future leaked in?"**

## What IS available at decision time (and therefore a legal feature)

- Everything about the current event itself: merchant category, customer
  value tier, event type, payment method, decline code, amount.
- Everything about the customer/merchant's history **strictly before**
  this event's timestamp: `prior_failure_count`, `prior_recovery_count`,
  `prior_recovery_rate`, `minutes_since_last_failure`,
  `recent_method_switch_count`.
- Observed proxies of latent behavior, e.g.
  `customer_retry_propensity_observed` — a noisy estimate of behavior,
  not the true latent value the simulator uses to generate outcomes.

## What is NEVER a feature

- `recovered` / `recovered_value` — this event's own label (the target).
- `chosen_intervention` — the treatment/action itself. This is modeled
  separately per-arm (T-learner); including it as a raw input feature in
  a single shared model would let the model use the action as a proxy
  for the outcome rather than learning the outcome's actual drivers.
- Any `potential_outcome_*` / `potential_recovered_*` column from
  `data/oracle_potential_outcomes.csv`. That file is counterfactual
  ground truth for **offline evaluation only** and must never be joined
  into training features — doing so would hand the model the answer.

## How leakage is mechanically prevented (not just promised)

1. **Generation-time causality.** `data/generate_data.py` iterates
   events in true chronological order and maintains a per-customer
   history dict. Every temporal feature for event *i* is read from that
   dict **before** it is updated with event *i*'s own outcome. There is
   no possibility of a later pass "looking backward" incorrectly,
   because the features are written once, at generation time, from
   state that provably does not yet include the current event.

2. **Temporal train/val/test split**, not a random split
   (`ml-service/temporal_split.py`). A random 80/20 split on sequential
   data lets a model implicitly see population-level drift signals from
   events that happened chronologically after its test set — a common,
   subtle leakage source. We split by time: earliest 60% train, next
   20% validation, most recent 20% test. The concept-drift window is
   deliberately placed late in the timeline so it appears mostly in
   val/test, letting us honestly evaluate behavior on drift the model
   never trained on.

3. **Automated leakage tests** (`ml-service/test_no_leakage.py`), run
   independently of the training pipeline:
   - `test_forbidden_columns_not_in_feature_spec` — statically checks
     the feature column list against a forbidden-column set.
   - `test_feature_frame_excludes_target_and_action` — builds an actual
     feature frame and checks no column name contains "recovered" or
     "chosen_intervention".
   - `test_temporal_features_are_causally_prior` — asserts every
     customer's **first-ever** logged event has `prior_failure_count ==
     0`. If any customer's first event shows nonzero prior history,
     that's direct evidence of backward leakage.
   - `test_oracle_file_not_importable_by_training_code` — greps
     `ml-service/train.py`'s source for the oracle filename and fails
     if it's referenced anywhere.

Run them yourself: `cd ml-service && python3 test_no_leakage.py`

## Known limitation, stated plainly

`merchant_baseline_fail_rate` is passed as a static per-merchant feature
rather than computed incrementally like the customer-level features. In
a real deployment this would also need to be an expanding, time-ordered
estimate (a brand-new merchant shouldn't have a fully-formed "baseline"
on day one) — deferred here because merchant cold-start is a distinct
problem (see `docs/model_cards.md`, "Known limitations") and conflating
it with the customer-level leakage fix would have muddied both.
