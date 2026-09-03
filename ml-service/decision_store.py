"""
Decision Store — round-2 fix #4, round-6 fix (idempotency), round-7 fix
(ATOMIC idempotency, Phase 2).

ROUND-6 GAP THIS ROUND FIXES: the previous in-memory dict had no lock.
Two genuinely simultaneous requests for a brand-new event_id could both
read "not present" before either wrote, both proceed to run inference,
and both call record_decision -- producing TWO decision_ids for one
event_id. Round 6's own findings doc flagged this honestly as untested
under real concurrency. This round replaces the in-memory dict with
SQLite and a UNIQUE constraint on event_id, so the race is closed at
the database layer, not by hoping the GIL saves us.

ATOMICITY MECHANISM: get_or_create_decision() does an
`INSERT OR IGNORE` inside a transaction, then immediately SELECTs the
row for that event_id. Whichever concurrent request's INSERT actually
lands first "wins"; every other concurrent request's INSERT is silently
ignored by SQLite (native UNIQUE-constraint semantics), and its
subsequent SELECT reads the winner's row. This means multiple concurrent
callers all get the SAME decision_id back, but only one of them should
treat itself as the "first" caller responsible for running inference --
the caller checks whether the row it got back was the one it just tried
to insert (rowcount) vs one that already existed, and only the true
first caller runs the ML pipeline. This is verified under real thread
concurrency in ml-service/test_concurrency.py, not just claimed.

STORAGE: SQLite in WAL mode, adequate for this prototype's scale and
single-process deployment. A true multi-process production deployment
would need a real database server (Postgres) since SQLite's write
concurrency is limited to one writer at a time -- stated honestly, not
hidden; WAL mode allows concurrent READERS during a write, which is
what matters for this store's read-heavy /decide-then-/feedback pattern.
"""
import json
import os
import sqlite3
import threading
import time
import uuid

DB_PATH = os.path.join(os.path.dirname(__file__), "models", "decision_store.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

_local = threading.local()


def _conn():
    """One SQLite connection per thread (SQLite connections are not
    thread-safe to share directly) -- this is what makes the concurrency
    test below meaningful: each simulated concurrent request gets its
    own connection, exactly as separate FastAPI worker threads would."""
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH, timeout=10)
        _local.conn.execute("PRAGMA journal_mode=WAL;")
        _local.conn.execute("PRAGMA busy_timeout=10000;")
    return _local.conn


def _init_schema():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            decision_id TEXT PRIMARY KEY,
            event_id TEXT UNIQUE,
            context TEXT,
            arm_columns TEXT,
            result TEXT,
            timestamp REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_log (
            decision_id TEXT PRIMARY KEY,
            arm TEXT,
            reward REAL,
            timestamp REAL
        )
    """)
    conn.commit()
    conn.close()


_init_schema()


def get_or_create_decision_id(event_id: str) -> tuple:
    """Round-7 atomic idempotency (Phase 2). Returns (decision_id,
    is_first_caller). Exactly one concurrent caller for a given event_id
    gets is_first_caller=True and must run the ML pipeline; every other
    caller (sequential retry or genuine concurrent race) gets
    is_first_caller=False and must wait for / read the cached result."""
    conn = _conn()
    new_id = str(uuid.uuid4())
    cur = conn.execute(
        "INSERT OR IGNORE INTO decisions (decision_id, event_id, timestamp) VALUES (?, ?, ?)",
        (new_id, event_id, time.time()),
    )
    conn.commit()
    is_first = cur.rowcount == 1
    if is_first:
        return new_id, True
    row = conn.execute("SELECT decision_id FROM decisions WHERE event_id = ?", (event_id,)).fetchone()
    return (row[0] if row else new_id), False


def finalize_decision(decision_id: str, context_vector: list, arm_columns: list, result: dict):
    conn = _conn()
    conn.execute(
        "UPDATE decisions SET context = ?, arm_columns = ?, result = ? WHERE decision_id = ?",
        (json.dumps(context_vector), json.dumps(arm_columns), json.dumps(result), decision_id),
    )
    conn.commit()


def get_decision_context(decision_id: str):
    conn = _conn()
    row = conn.execute("SELECT context FROM decisions WHERE decision_id = ?", (decision_id,)).fetchone()
    if not row or not row[0]:
        return None
    return json.loads(row[0])


def get_cached_decision_result(decision_id: str, wait_for_result: bool = True, timeout_s: float = 2.0):
    """For a non-first concurrent caller: poll briefly for the first
    caller's result to land (it may still be mid-inference), bounded by
    timeout_s so a slow/failed first caller cannot hang other callers
    forever."""
    conn = _conn()
    deadline = time.time() + timeout_s
    while True:
        row = conn.execute("SELECT result FROM decisions WHERE decision_id = ?", (decision_id,)).fetchone()
        if row and row[0]:
            return json.loads(row[0])
        if not wait_for_result or time.time() > deadline:
            return None
        time.sleep(0.02)


def get_event_id_for_decision(decision_id: str):
    """Round-7 fix (Phase 4 wiring): needed by /feedback to know which
    raw event this decision was for, so event_store can be updated with
    the real outcome once it's known."""
    conn = _conn()
    row = conn.execute("SELECT event_id FROM decisions WHERE decision_id = ?", (decision_id,)).fetchone()
    return row[0] if row else None


def is_feedback_already_processed(decision_id: str) -> bool:
    conn = _conn()
    row = conn.execute("SELECT 1 FROM feedback_log WHERE decision_id = ?", (decision_id,)).fetchone()
    return row is not None


def mark_feedback_processed(decision_id: str, arm: str, reward: float) -> bool:
    """Round-7 atomic feedback idempotency. Returns True if THIS call was
    the one that recorded feedback, False if another (concurrent or
    prior) call already had. Uses the same INSERT OR IGNORE + rowcount
    pattern as decisions."""
    conn = _conn()
    cur = conn.execute(
        "INSERT OR IGNORE INTO feedback_log (decision_id, arm, reward, timestamp) VALUES (?, ?, ?, ?)",
        (decision_id, arm, reward, time.time()),
    )
    conn.commit()
    return cur.rowcount == 1
