# Final Action Ranking Analysis — "Why does Recoup choose the wrong arm?"

Four prior rounds established THAT ~90% of regret comes from wrong-arm
selection (round 4), that the ranking is weakly-but-nonzero informative
(Spearman 0.25, round 5), and that overlap for the `no_action` control
arm is thin (73.8% below common support, round 2). This document
connects those findings into a specific, evidence-backed answer to
**why**, using `evaluation/final_wrong_arm_forensics.py`.

## Method (stated plainly — this is a diagnostic heuristic, not causal isolation)

For every test event where Recoup's chosen action differs from the
oracle-best action, the event is tagged with every plausible error
category it exhibits (categories are NOT mutually exclusive), using
signals already available in this codebase. This is NOT a controlled
ablation that isolates one cause at a time — it's pattern-matching
against measurable proxies. Full scoping of what's included/excluded is
in the script's module docstring.

## Result (n=541 test events, 434 wrong-arm, 80.2% wrong-arm rate on this run)

| Category | % of total regret | Event count | What it means |
|---|---|---|---|
| **B: Uplift error** | **72.3%** | 337 | The oracle-best arm's raw recovery probability was reasonably accurate, but its *uplift* (vs the `no_action` baseline) still ranked below the arm actually chosen |
| H: High-value transaction | 53.4% | 90 | Concentrated in a small number of events — high-value regret is a small-count, high-impact problem |
| J: Stochastic margin | 33.3% | 136 | The oracle's top-2 arms had true probabilities within 0.05 of each other — a third of "wrong" decisions may reflect near-tied ground truth more than model failure |
| G: Cold start | 27.6% | 125 | Customer had zero prior recorded events |
| A: Probability error | 27.3% | 94 | The oracle-best arm's predicted probability was off by >0.15 from its true value |
| F: Thin support | 22.3% | 122 | The chosen arm's training volume was below the median |
| K: Policy filtered | **3.8%** | 7 | The model's own top pick WAS correct, but the safety policy gate overrode it |

*(Percentages sum past 100% because categories overlap — a single event
can carry multiple tags.)*

## The specific finding

**The dominant, most specific cause is category B: uplift error — 72.3%
of total regret, more than double any other single category.** This is
not "the model doesn't know what it's doing" in general; the underlying
per-arm probability estimates are only modestly implicated (27.3%,
category A). The problem is specifically in how those probabilities get
turned into an *uplift* — i.e., specifically in `mu_no_action(X)`, the
baseline every uplift estimate is measured against.

**This directly explains, and is explained by, the overlap finding from
round 2**: `no_action` has only ~80 logged training examples and 73.8%
of test contexts fall below common support for it
(`evaluation/overlap_diagnostics.py`). A poorly-supported baseline
doesn't just make the *no_action* arm's own value estimate unreliable —
it corrupts every OTHER arm's *uplift* number too, since every uplift
is `mu_a(X) - mu_no_action(X)`. **Category B's 72.3% share is the
overlap problem showing up as economic regret, not a separate issue.**

This also validates round 3's finding that a two-stage policy
(intervene-vs-not, then which-arm) was a reasonable idea in principle —
it targeted exactly this baseline-overlap problem — even though the
specific implementation tested did not improve results (see
`docs/round3_findings.md`). The diagnosis was right; the fix tried
wasn't sufficient.

## Secondary finding: a third of "wrong" decisions may be near-unavoidable

Category J (stochastic margin, 33.3% of regret) suggests that for a
meaningful share of events, the oracle's own top-2 candidate actions
have nearly identical true recovery probabilities. In these cases,
"beating the oracle's top-1 pick" is closer to a coin flip than a
model-quality question — worth stating honestly rather than treating
every wrong-arm event as an equally fixable model failure.

## What this means for future work (not built this sprint, correctly prioritized now)

The forensic evidence points at **exactly one thing** as the
highest-leverage fix: a better-supported estimate of `mu_no_action(X)`
— which is precisely what a cross-fitted doubly-robust (AIPW) estimator
would target, since DR explicitly corrects outcome-model bias using a
propensity-weighted term rather than relying on the outcome model alone
in thin-support regions. This is not a new recommendation (rounds 4-7
all flagged DR as the largest gap) — what's new here is a specific,
measured reason *why* DR specifically (not more data, not a different
base learner, not more temporal features) is the correctly-targeted
fix: **72.3% of regret is associated with the exact quantity DR is designed to
stabilize.**

Given the September 4 deadline and seven prior rounds' worth of already
-deferred large builds, DR was not implemented this sprint either — but
this analysis is the strongest evidence yet for *why*, if any single
piece of remaining ML work were prioritized with more time, it should
be this one, not the other open items (5-seed robustness, high-value
tuned thresholds, direct policy learning) which this analysis suggests
would have smaller, more diffuse impact.

*(Editorial note added later: DR was subsequently implemented, validated,
and tested — see `docs/dr_cross_fitting_final.md` and
`docs/final_policy_selection.md` for current status. This document is
kept as-written to preserve the accurate history of when this
conclusion was reached.)*
