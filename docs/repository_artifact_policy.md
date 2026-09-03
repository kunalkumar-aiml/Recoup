# Repository Artifact Policy

## Source of truth
Source code, configuration, and generation scripts. Specifically:
`data/generate_data.py`, everything under `ml-service/` except
`ml-service/models/`, everything under `evaluation/`, `backend/`,
`frontend/`, `chaos/`, `docs/`, `README.md`, `.gitignore`,
`requirements.txt`, `package.json`.

## Generated (not committed, gitignored)
- `data/events.csv`, `data/oracle_potential_outcomes.csv` — produced by
  `data/generate_data.py`.
- `ml-service/models/*.pkl`, `*.json` (calibration report) — produced
  by `ml-service/train.py`.
- `ml-service/models/*.db*`, `decision_store.jsonl`, `feedback_log.jsonl`
  — produced at runtime by the live service.
- `evaluation/*_results.json` — produced by each evaluation script.
- `chaos/chaos_test_results.json` — produced by `chaos/chaos_test.py`.
- `audit/audit_log.jsonl` — produced at runtime by the live service.

## Regenerable — confirmed, not assumed
Every file in the "generated" list above was deleted and successfully
regenerated from a simulated clean checkout during this audit (see
`docs/github_tracked_files_audit.md`, "Clean-checkout reproduction").
None of these files are required to be present for someone cloning the
repository to reproduce the project — they are produced by running the
documented commands in `README.md` Section 12.

## Why generated artifacts are not committed
1. **Reproducibility as the actual proof.** A judge who can regenerate
   the models and data from source has stronger evidence the project
   works than one who is asked to trust a committed binary.
2. **Repository size and diff hygiene.** Binary model files and
   multi-thousand-row CSVs produce unreviewable diffs on every retrain.
3. **No secret/PII risk from committed generated data**, since none is
   committed in the first place (see `docs/data_leakage_audit.md`).

## Exception policy
If any generated artifact were ever committed for a specific,
documented reason (none currently are), that reason would be stated
here explicitly rather than left implicit. Currently: no exceptions.
