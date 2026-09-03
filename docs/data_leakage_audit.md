# Data Leakage / PII Audit

## PII scan

Grepped all data files (`data/events.csv`, `data/oracle_potential_outcomes.csv`)
and all code for real names, email addresses, phone numbers, card
numbers, and CVV patterns. **None found.** Customer and merchant
identifiers are synthetic, sequentially generated (`C0001`, `M001`,
etc.) by `data/generate_data.py` — not derived from or resembling any
real individual or entity.

All data in this repository is synthetic, generated fresh by
`data/generate_data.py` (seeded, reproducible). No real Razorpay
transaction data, merchant data, or customer data was used anywhere in
this project.

## Oracle / test leakage audit

This is the most safety-critical check for an ML competition submission,
and this project already has automated tests for it
(`ml-service/test_no_leakage.py`, 4 checks, all pass) — this section
restates what those tests verify and adds the phase-boundary picture the
request asked for explicitly.

### Phase boundaries

| Phase | Allowed inputs | Forbidden inputs |
|---|---|---|
| **TRAIN** | `data/events.csv`, temporal-split TRAIN rows only | `data/oracle_potential_outcomes.csv` (never read by any training script); TEST-split rows; `recovered`/`chosen_intervention` as a feature (only as the label being predicted) |
| **VALIDATION** (in-arm temporal holdout inside `uplift.py`) | TRAIN-split rows only, further split by time within the arm | Oracle file; TEST-split rows |
| **TEST** | `data/events.csv` TEST-split rows | Oracle file — TEST-split feature values are used for prediction, but the true recovered outcome under untested arms is never used to pick features or tune thresholds |
| **ORACLE DIAGNOSTICS** (`evaluation/*.py` — `evaluate.py`, `ablation.py`, `regret_decomposition.py`, `dr_cross_fitting.py`, etc.) | `data/oracle_potential_outcomes.csv`, joined to TEST-split events **only to score a policy's decisions after the fact** | Any training script; any threshold-selection code path |

### Verified mechanically, not just asserted

`ml-service/test_no_leakage.py` runs four checks on every invocation:
1. No forbidden column (target, action, oracle columns) appears in the
   feature specification.
2. The actual built feature frame (not just the spec) excludes target
   and action columns.
3. Temporal features are exactly zero for each customer's first-ever
   event (a positive value would prove a feature saw the future).
4. `ml-service/train.py`'s source code is greped for the oracle
   filename and fails if found anywhere.

### One deliberate, documented exception

`evaluation/dr_cross_fitting.py` and other diagnostic scripts under
`evaluation/` **do** read `data/oracle_potential_outcomes.csv` — this is
correct and intentional: these scripts score what a policy *would have
recovered* under the oracle's known ground truth, entirely after the
policy's decision logic has already been fixed by training. No oracle
value is ever fed into a training call, a hyperparameter search, or a
threshold-selection routine. The distinction is enforced by which
Python files import the oracle CSV — `ml-service/train.py` and
`ml-service/uplift.py` never do; every file in `evaluation/` that does
is a scoring/diagnostic script, not a training script.

## Post-hoc seed selection

Every multi-seed result in this repository reports the seeds actually
run and their raw values (see `docs/round3_findings.md`,
`evaluation/dr_multiseed_results.json`) — including seeds where the
system performed worse, not just the best one. No seed was excluded
from a reported aggregate after seeing its result.

## Conclusion

No PII or real transaction data exists anywhere in this repository. The
train/validation/test/oracle-diagnostic boundary is enforced both by
convention (evaluation scripts vs. training scripts) and by an
automated, repeatedly-run test suite — not by policy alone.
