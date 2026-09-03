"""Append-only audit log. Every decision the system makes is recorded here,
including inputs, model scores, policy check result, and outcome."""
import json
import os
import time

AUDIT_PATH = os.path.join(os.path.dirname(__file__), "..", "audit", "audit_log.jsonl")
os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)


def log_decision(entry: dict):
    entry["logged_at"] = time.time()
    with open(AUDIT_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_audit_log(limit=100):
    if not os.path.exists(AUDIT_PATH):
        return []
    with open(AUDIT_PATH) as f:
        lines = f.readlines()[-limit:]
    return [json.loads(line) for line in lines]
