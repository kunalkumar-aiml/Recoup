# Final Business Results (Synthetic Simulation — Not Real Razorpay Data)

**For a synthetic batch of 541 failed payments totaling ₹5,58,339 at
risk**, under this repo's evaluation protocol (`evaluation/protocol.py`,
`evaluation/final_business_benchmark.py`):

| Strategy | Gross recovered | Cost | **Net recovered** | Recovery rate | Automation | Human review |
|---|---|---|---|---|---|---|
| Always retry (baseline) | ₹1,15,982 | ₹1,082 | **₹1,14,900** | 20.8% | 100% | 0% |
| ML-only (no cost/uplift awareness) | ₹1,36,400 | ₹5,211 | **₹1,31,189** | 24.4% | 100% | 0% |
| **Recoup** | ₹1,41,181 | ₹5,328 | **₹1,35,853** | 25.3% | 94.3% | 5.7% |

**Incremental net recovered value: +₹20,953 vs always-retry (+18.2%),
+₹4,664 vs ML-only (+3.6%).** 20 of 541 events (3.7%) were escalated to
human review or received no economically-justified automated action;
zero unsafe actions were blocked this run (the policy gate simply never
saw a request to approve one this time — see the Failure Lab / chaos
suite for cases where it does).

## This is a single seed. Read this before quoting the percentage.

This table matches the single-seed (seed 42) headline reported
throughout this repo. **A 3-seed robustness study found this lift is
not stable** — ranging from -0.5% to +49.1% across seeds
(`docs/round3_findings.md`). The honest aggregate is **+22.3% ± 20.5%**
(n=3). Quote the range or the methodology, not the single percentage,
under expert questioning.

## What "recovery rate" and "automation" mean here, precisely

- **Recovery rate** = gross recovered value ÷ total revenue at risk for
  that strategy. It rises with more aggressive automated action, which
  is exactly why it's reported alongside *net* recovered value (which
  nets out cost) — a strategy that recovers more gross revenue by
  spending more on intervention cost isn't automatically better.
- **Automation** = % of events where a bounded action executed without
  human review. **Human review** = % escalated due to low confidence,
  drift, or a policy gate rejection. These sum with a small
  "no-economically-justified-action" bucket to 100%.
- **Escalated/no-action events are credited at the empirical historical
  human-agent recovery rate** (not ₹0) — an escalated case is still
  worked by a person in reality; scoring it as pure loss would unfairly
  penalize conservative, safety-driven behavior. This is the same
  convention used everywhere else in this repo's evaluation
  (`evaluation/protocol.py::historical_escalation_credit_rate`).

## Reproduce this exact table

```bash
cd data && python3 generate_data.py && cd ../ml-service && python3 train.py
cd ../evaluation && python3 final_business_benchmark.py
```

Output: `evaluation/final_business_benchmark.json`.

## What this table is NOT

- Not a claim about real Razorpay transaction volumes, real merchant
  revenue, or real recovery outcomes. Every number is computed against
  this repo's synthetic counterfactual oracle
  (`data/oracle_potential_outcomes.csv`).
- Not evidence of causal real-world impact — see
  `docs/causal_identification.md` for the identification assumptions
  underlying even the synthetic causal claims.
- Not a production SLA or throughput claim — see `docs/limitations.md`
  for what was and wasn't load-tested.
