# GitHub Tracked-Files Audit

## What I can and cannot verify from this environment

**I do not have access to your actual GitHub repository or your local
`.git` directory.** I only have the working-tree contents of this
delivered archive. `git ls-files` and `git log --all` must be run by
you, against your real repository, to get ground truth. What follows
is (a) what I verified from the working tree + `.gitignore` rules, and
(b) the exact commands for what only you can check.

## What I verified: .gitignore pattern coverage vs. actual generated files

I generated a fresh copy of the repository's data/models from a
simulated clean checkout (all gitignored files deleted, then
regenerated via the documented reproduction steps) and confirmed every
currently-generated file matches an existing `.gitignore` pattern:

| File | Gitignore pattern | Covered? |
|---|---|---|
| `data/events.csv` | `data/events.csv` | Yes |
| `data/oracle_potential_outcomes.csv` | `data/oracle_potential_outcomes.csv` | Yes |
| `ml-service/models/*.pkl` (5 files) | `ml-service/models/*.pkl` | Yes |
| `ml-service/models/*.db*` | `ml-service/models/*.db`, `*.db-*` | Yes |
| `ml-service/models/decision_store.jsonl` | explicit pattern | Yes |
| `ml-service/models/feedback_log.jsonl` | explicit pattern | Yes |
| `evaluation/*_results.json` | `evaluation/*_results.json` | Yes |
| `chaos/chaos_test_results.json` | explicit pattern | Yes |
| `audit/audit_log.jsonl` | explicit pattern | Yes |
| `node_modules/`, `__pycache__/` | standard patterns | Yes |

**Historical evidence from this project's actual push transcripts**
(visible earlier in this conversation): every `git add . && git commit`
output shown across all prior rounds listed only source files (`.py`,
`.js`, `.md`, `.json` config, `.html`) — never a `.pkl`, `.csv` data
file, or `.db` file. This is consistent with `.gitignore` having worked
correctly on every push, but it is evidence from what you showed me in
chat, not something I independently re-verified against the live repo
in this session.

## What only you can verify — exact commands

Run these in your actual repository (`~/Desktop/Recoup`) and check the
output against the classifications below:

```bash
cd ~/Desktop/Recoup

# 1. What's actually tracked in the current commit
git ls-files > /tmp/tracked_files.txt
cat /tmp/tracked_files.txt

# 2. Explicitly check for the risky patterns
git ls-files | grep -E '\.pkl$|\.db$|\.db-|events\.csv$|oracle_potential_outcomes\.csv$|_results\.json$|audit_log|decision_store\.jsonl|feedback_log\.jsonl|\.env|credential|\.pem$|\.key$'
```

**Expected result**: the second command should print **nothing**. If it
prints any filename, that file IS tracked in GitHub despite
`.gitignore`, most likely because it was `git add`-ed before the
relevant `.gitignore` line existed, or added with `git add -f`. If that
happens:
```bash
git rm --cached <filename>
git commit -m "Remove generated file from tracking"
git push
```

## Oracle exposure decision

**Decision: keep `data/oracle_potential_outcomes.csv` gitignored /
untracked.** It is fully and deterministically regenerable from
`data/generate_data.py` (verified in this audit — see clean-checkout
reproduction below), so there is no reproducibility cost to not
publishing it, and not publishing it keeps the repository smaller and
avoids any appearance of the ground-truth file being something a judge
needs to inspect directly rather than regenerate. README already
documents the regeneration step as the first line of the reproduction
sequence.

## Model artifact exposure decision

**Decision: keep `ml-service/models/*.pkl` gitignored / untracked.**
`ml-service/train.py` deterministically reproduces them (verified
below). A public ML-competition repository is stronger when it proves
"the source code trains this from scratch" rather than shipping opaque
binaries a judge has to trust without re-running anything.

## Clean-checkout reproduction — actually run, not assumed

Simulated a clean checkout in this environment (deleted every
gitignored generated file, then followed the README's exact
reproduction commands in order) and confirmed:

1. `python3 generate_data.py` → regenerates `events.csv` and
   `oracle_potential_outcomes.csv` from scratch.
2. `python3 train.py` → regenerates all 5 `.pkl` models, calibration
   report, uncertainty ensembles.
3. `python3 test_no_leakage.py` → 4/4 PASS.
4. `python3 test_economic_value_engine.py` → 10/10 PASS.
5. `python3 evaluate.py` → produces the same headline numbers
   documented in `README.md` (mean regret ₹514.25, n=541, etc.) — an
   exact match, not just "similar."

**This confirms the README's reproduction sequence works end-to-end
from nothing.** Not verified in this pass: the frontend/backend Node
services and the live-service tests (`test_server_side_state.py`,
`test_concurrency.py`, `chaos/chaos_test.py`) require `npm install` and
a running `uvicorn` process — both were verified working in earlier
rounds of this project's development but were not re-run fresh in this
specific audit pass, given the "no functional changes" scope of this
round. State this as "reproducible under the documented local
environment," not "fully reproducible in all respects," per this
audit's own instruction not to overclaim.
