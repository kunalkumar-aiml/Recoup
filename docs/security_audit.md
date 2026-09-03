# Security Audit

## Scope
Working tree of this repository, as it exists in this delivered
archive. See "Git history" note below for what could NOT be checked
from this environment.

## Method
- Pattern grep across all `.py`, `.js`, `.md`, `.json`, `.txt`, `.env`
  files for: API key/secret key/password assignment patterns, AWS
  access key format (`AKIA...`), PEM/private-key headers, OpenAI-style
  keys (`sk-...`), Slack tokens (`xox...`), GitHub tokens (`ghp_...`),
  MongoDB connection strings with embedded credentials, and generic
  `Bearer <token>` patterns.
- Search for `.env`, `*credential*`, `*.pem`, `*.key` files anywhere in
  the tree.
- Email-address and phone-number pattern scan across code, docs, and
  data files.

## Findings

**No secrets, credentials, API keys, tokens, or private keys were found
in the working tree.** Zero matches across all patterns above.

**No `.env`, credential, or private-key files exist anywhere in the
repository.**

**No email addresses or phone numbers were found** in code, docs, or
the synthetic data files.

Every network endpoint reference in the code (`localhost:8000`,
`localhost:4000`, `127.0.0.1`) is a local development URL, not a
credential or a real production endpoint.

## Git history — not checkable from this environment, flagged honestly

This audit was run against the working-tree contents of this delivered
archive, which has no `.git` directory in this environment. **It could
not scan the git history of the actual GitHub repository
(`github.com/kunalkumar-aiml/Recoup`)** for secrets that may have been
committed and later removed in an earlier commit still reachable by SHA.

Because each round of this project's development involved a fresh
`git init` + `git push --force` from the local machine (not incremental
commits building on prior history), GitHub may retain now-unreferenced
commit objects from earlier force-pushes for a period of time, even
though they no longer appear in the normal commit history view.

**Run these exact commands in your actual repository** to complete this
audit — I cannot run them from this environment:

```bash
cd ~/Desktop/Recoup

# 1. See the full reachable commit history
git log --all --oneline

# 2. List every object ever reachable in this repo (not just current HEAD)
git rev-list --all --objects | head -50

# 3. Best option — install and run gitleaks (a real secret scanner)
brew install gitleaks   # macOS, if not already installed
gitleaks detect --source . --log-opts="--all"

# 4. If gitleaks isn't available, a manual pattern grep across all history
git log --all -p | grep -iE "api[_-]?key|secret[_-]?key|password\s*=|AKIA[0-9A-Z]{16}|-----BEGIN (RSA|PRIVATE|OPENSSH)|sk-[a-zA-Z0-9]{20,}|xox[baprs]-|ghp_[a-zA-Z0-9]{30,}"
```

**Expected result**: `gitleaks detect` should report "no leaks found."
The manual grep (command 4) should print nothing. If either finds a
real credential:
1. Do not paste it anywhere, including back to me.
2. Note only the file name and commit hash.
3. Rotate/revoke that credential immediately (assume it's compromised
   the moment it touched a public repo, even briefly).
4. Use `git filter-repo` or BFG Repo-Cleaner to remove it from history,
   then force-push, then verify again with `gitleaks`.

Every working-tree scan in this document (and in
`docs/final_repo_audit.md`) found nothing — but the working tree is not
the same thing as full git history, and this section exists specifically
so that distinction doesn't get glossed over.

## Conclusion

Working tree: **clean, verified**. Git history: **not verified from
this environment — run the commands above against the real repository
before treating this security audit as complete.**
