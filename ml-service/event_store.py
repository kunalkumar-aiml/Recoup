"""
Event Store — round-7 fix (Phase 4): server-side temporal state.

PROBLEM FIXED: round 6's own findings doc flagged this honestly --
prior_failure_count, prior_recovery_count, prior_recovery_rate, and
minutes_since_last_failure were CLIENT-SUPPLIED fields on EventIn. A
client (buggy or malicious) could claim any history it wanted and the
model would use it uncritically. That is not a real behavioral-state
system, it's just trusting the caller's word.

FIX: a SQLite-backed store of raw events keyed by customer_id. At
/decide time, the server computes prior_failure_count etc. ITSELF from
events it has actually seen for that customer_id with a timestamp
strictly before the current event -- and overrides whatever the client
sent for those fields, regardless of the claimed value. The client can
still lie about them in the request; it just has no effect.

SQLite gives us atomic, persistent idempotency for free via a UNIQUE
constraint on event_id (round-7 fix, Phase 2) -- a concurrent duplicate
INSERT fails at the database level rather than racing an in-memory
dict.
"""
import os
import sqlite3
import time
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "models", "recoup_state.db")


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")  # allows concurrent readers during a writer
    return conn


def init_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_events (
            event_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            event_timestamp TEXT NOT NULL,
            ingestion_timestamp REAL NOT NULL,
            amount REAL NOT NULL,
            recovered INTEGER,
            chosen_intervention TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS event_id_decision_map (
            event_id TEXT PRIMARY KEY,
            decision_id TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_feedback (
            decision_id TEXT PRIMARY KEY,
            arm TEXT NOT NULL,
            reward REAL NOT NULL,
            processed_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


def try_claim_event_id(event_id: str, decision_id: str) -> bool:
    """Round-7 atomic idempotency (Phase 2). Returns True if THIS call
    successfully claimed event_id (i.e. it's genuinely new -- proceed
    with a fresh decision). Returns False if event_id was already
    claimed (by this call or a concurrent one) -- the caller must look
    up and return the EXISTING decision instead. The PRIMARY KEY
    constraint on event_id makes this atomic even under concurrent
    requests: only one INSERT can ever succeed for a given event_id,
    enforced by SQLite itself, not by application-level locking."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO event_id_decision_map (event_id, decision_id) VALUES (?, ?)",
            (event_id, decision_id),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_decision_id_for_event(event_id: str):
    conn = _connect()
    row = conn.execute(
        "SELECT decision_id FROM event_id_decision_map WHERE event_id = ?", (event_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def record_raw_event(event_id: str, customer_id: str, event_timestamp: str, amount: float):
    conn = _connect()
    conn.execute(
        "INSERT OR IGNORE INTO raw_events (event_id, customer_id, event_timestamp, ingestion_timestamp, amount) "
        "VALUES (?, ?, ?, ?, ?)",
        (event_id, customer_id, event_timestamp, time.time(), amount),
    )
    conn.commit()
    conn.close()


def update_event_outcome(event_id: str, recovered: bool, chosen_intervention: str):
    conn = _connect()
    conn.execute(
        "UPDATE raw_events SET recovered = ?, chosen_intervention = ? WHERE event_id = ?",
        (int(recovered), chosen_intervention, event_id),
    )
    conn.commit()
    conn.close()


def compute_trusted_temporal_state(customer_id: str, before_timestamp: str):
    """The server-computed, trustworthy version of prior_failure_count /
    prior_recovery_count / prior_recovery_rate / minutes_since_last_failure
    -- derived ONLY from events this store has actually recorded for this
    customer_id with event_timestamp strictly before `before_timestamp`.
    Ignores anything the client claimed."""
    conn = _connect()
    rows = conn.execute(
        "SELECT event_timestamp, recovered FROM raw_events "
        "WHERE customer_id = ? AND event_timestamp < ? ORDER BY event_timestamp",
        (customer_id, before_timestamp),
    ).fetchall()
    conn.close()

    prior_failure_count = len(rows)
    prior_recovery_count = sum(1 for _, recovered in rows if recovered == 1)
    prior_recovery_rate = (prior_recovery_count / prior_failure_count) if prior_failure_count > 0 else -1

    minutes_since_last_failure = -1
    if rows:
        last_ts_str = rows[-1][0]
        try:
            last_ts = datetime.fromisoformat(last_ts_str)
            cur_ts = datetime.fromisoformat(before_timestamp)
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            if cur_ts.tzinfo is None:
                cur_ts = cur_ts.replace(tzinfo=timezone.utc)
            minutes_since_last_failure = (cur_ts - last_ts).total_seconds() / 60
        except Exception:
            minutes_since_last_failure = -1

    return {
        "prior_failure_count": prior_failure_count,
        "prior_recovery_count": prior_recovery_count,
        "prior_recovery_rate": round(prior_recovery_rate, 4),
        "minutes_since_last_failure": round(minutes_since_last_failure, 2),
    }


def try_claim_feedback(decision_id: str, arm: str, reward: float) -> bool:
    """Round-7 atomic feedback idempotency, same PRIMARY KEY mechanism as
    try_claim_event_id."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO processed_feedback (decision_id, arm, reward, processed_at) VALUES (?, ?, ?, ?)",
            (decision_id, arm, reward, time.time()),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()
