"""
audit_log.py
============
Step 6 of the AI-Governed Multi-Agent Platform.

Provides a dedicated, tamper-EVIDENT SQLite audit log for pipeline runs.

DESIGN PRINCIPLES:
  - Zero LLM calls. Fully deterministic, rule-based database logger.
  - Storage is completely independent of LangGraph's checkpointer (uses audit_log.db,
    never pipeline_state.db).
  - Schema captures structured, timestamped events per pipeline stage:
      run_id, seq, timestamp, event_type, event_source, summary, details_json
  - Distinguishes event_source: 'automated' vs 'human_reviewer'.

TAMPER EVIDENCE — WHAT THE HASH CHAIN DOES AND DOES NOT GIVE YOU
-----------------------------------------------------------------
Every entry stores `entry_hash`, the SHA-256 of its own canonical content
*together with* the `entry_hash` of the preceding entry for the same run. Each
run is therefore a linked chain anchored at GENESIS_HASH.

  entry_hash[n] = sha256(canonical(entry[n]) + entry_hash[n-1])

DETECTED — verify_audit_chain() will report these:
  - Any edit to a logged field (summary, details, timestamp, event_source...).
    Recomputing the hash no longer matches the stored one.
  - Deletion of an entry from the middle of a run. The next entry's prev_hash
    no longer matches its predecessor.
  - Insertion of a forged entry, or reordering of entries. `seq` is part of the
    hashed content, so positions cannot be shuffled.

NOT DETECTED — be honest about this in any report:
  - Truncation. An attacker who deletes a *trailing* run of entries leaves a
    shorter but internally consistent chain. Nothing inside the database can
    prove entries once existed beyond its own head.
  - Deletion of an entire run.
  - Wholesale recomputation. Anyone who can write to the file can rebuild a
    fully valid chain over falsified content, because the chain is unsigned.

  Closing those requires an anchor OUTSIDE this file: periodically publishing
  the head hash somewhere append-only (a signed log, another host, a timestamping
  service), or signing each entry with a key the database host does not hold.
  That is the natural next step and is deliberately not claimed here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

DEFAULT_AUDIT_DB = "audit_log.db"

# Anchor for the first entry of every run's chain.
GENESIS_HASH = "0" * 64

# Marker written into prev_hash/entry_hash for rows that predate hash chaining
# (i.e. rows already present in an audit_log.db created by an older version).
LEGACY_MARKER = "legacy"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _canonical_content(
    run_id: str,
    seq: int,
    timestamp: str,
    event_type: str,
    event_source: str,
    summary: str,
    details_json: str,
    prev_hash: str,
) -> bytes:
    """
    Serialise an entry to a stable byte string for hashing.

    sort_keys and fixed separators make the encoding canonical: the same logical
    entry always produces identical bytes, so a hash mismatch means the content
    genuinely changed rather than that a dict happened to iterate differently.
    """
    return json.dumps(
        {
            "run_id": run_id,
            "seq": seq,
            "timestamp": timestamp,
            "event_type": event_type,
            "event_source": event_source,
            "summary": summary,
            "details_json": details_json,
            "prev_hash": prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_entry_hash(
    run_id: str,
    seq: int,
    timestamp: str,
    event_type: str,
    event_source: str,
    summary: str,
    details_json: str,
    prev_hash: str,
) -> str:
    """SHA-256 of an entry's canonical content, chained to prev_hash."""
    return hashlib.sha256(
        _canonical_content(
            run_id, seq, timestamp, event_type, event_source,
            summary, details_json, prev_hash,
        )
    ).hexdigest()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def init_audit_db(db_path: str = DEFAULT_AUDIT_DB) -> None:
    """
    Create the audit_entries table if absent, and migrate an older table that
    lacks the chain columns by adding them and marking existing rows as legacy.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_source TEXT NOT NULL,
                summary TEXT NOT NULL,
                details_json TEXT NOT NULL,
                seq INTEGER NOT NULL DEFAULT 0,
                prev_hash TEXT NOT NULL DEFAULT '',
                entry_hash TEXT NOT NULL DEFAULT ''
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_run_id ON audit_entries(run_id)
        """)

        # Migrate a pre-chain table created by an earlier version.
        existing = {row[1] for row in cursor.execute("PRAGMA table_info(audit_entries)")}
        for column, ddl in (
            ("seq", "ALTER TABLE audit_entries ADD COLUMN seq INTEGER NOT NULL DEFAULT 0"),
            ("prev_hash", f"ALTER TABLE audit_entries ADD COLUMN prev_hash TEXT NOT NULL DEFAULT '{LEGACY_MARKER}'"),
            ("entry_hash", f"ALTER TABLE audit_entries ADD COLUMN entry_hash TEXT NOT NULL DEFAULT '{LEGACY_MARKER}'"),
        ):
            if column not in existing:
                cursor.execute(ddl)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def log_audit_event(
    run_id: str,
    event_type: str,
    event_source: str,
    summary: str,
    details: dict,
    db_path: str = DEFAULT_AUDIT_DB,
) -> str:
    """
    Append a structured audit event, chained to the previous entry for this run.

    Parameters
    ----------
    run_id : str
        Unique session / thread_id identifier for the pipeline run.
    event_type : str
        Stage identifier (e.g. 'planner_run', 'data_agent_run', 'training_run',
        'fairness_run', 'human_decision', 'final_outcome').
    event_source : {"automated", "human_reviewer"}
    summary : str
        Short human-readable event description.
    details : dict
        Structured details, serialised as JSON text.
    db_path : str

    Returns
    -------
    str
        The entry_hash of the newly appended entry (the run's new chain head).
    """
    init_audit_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    details_str = json.dumps(details, default=str)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        cursor = conn.cursor()
        # BEGIN IMMEDIATE takes the write lock up front, so the read of the
        # current chain head and the insert that extends it cannot interleave
        # with another writer and fork the chain.
        cursor.execute("BEGIN IMMEDIATE")

        row = cursor.execute(
            """
            SELECT seq, entry_hash FROM audit_entries
            WHERE run_id = ?
            ORDER BY seq DESC, id DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()

        if row is None:
            seq = 0
            prev_hash = GENESIS_HASH
        else:
            seq = int(row[0]) + 1
            prev_hash = row[1] or LEGACY_MARKER

        entry_hash = compute_entry_hash(
            run_id=run_id,
            seq=seq,
            timestamp=now_iso,
            event_type=event_type,
            event_source=event_source,
            summary=summary,
            details_json=details_str,
            prev_hash=prev_hash,
        )

        cursor.execute(
            """
            INSERT INTO audit_entries
                (run_id, timestamp, event_type, event_source, summary,
                 details_json, seq, prev_hash, entry_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, now_iso, event_type, event_source, summary,
             details_str, seq, prev_hash, entry_hash),
        )
        conn.commit()
        return entry_hash
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def _fetch_rows(run_id: str, db_path: str) -> list[tuple]:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        return conn.execute(
            """
            SELECT id, run_id, timestamp, event_type, event_source, summary,
                   details_json, seq, prev_hash, entry_hash
            FROM audit_entries
            WHERE run_id = ?
            ORDER BY seq ASC, id ASC
            """,
            (run_id,),
        ).fetchall()
    finally:
        conn.close()


def get_audit_trail(run_id: str, db_path: str = DEFAULT_AUDIT_DB) -> list[dict[str, Any]]:
    """
    Retrieve all audit log entries for a given run_id, ordered chronologically.

    Returns
    -------
    list[dict]:
        List of event entries with deserialized 'details' dict.
    """
    init_audit_db(db_path)
    entries = []
    for row in _fetch_rows(run_id, db_path):
        (entry_id, r_id, ts, ev_type, ev_source, summary,
         det_json, seq, prev_hash, entry_hash) = row
        try:
            parsed_details = json.loads(det_json)
        except Exception:
            parsed_details = {"raw": det_json}

        entries.append({
            "id": entry_id,
            "run_id": r_id,
            "seq": seq,
            "timestamp": ts,
            "event_type": ev_type,
            "event_source": ev_source,
            "summary": summary,
            "details": parsed_details,
            "prev_hash": prev_hash,
            "entry_hash": entry_hash,
        })

    return entries


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_audit_chain(
    run_id: str,
    db_path: str = DEFAULT_AUDIT_DB,
) -> dict[str, Any]:
    """
    Recompute every entry hash for a run and check the chain linkage.

    Returns
    -------
    dict:
        verified        : bool  — True only if every chained entry is intact
        entries_total   : int
        entries_checked : int   — chained entries actually verified
        legacy_entries  : int   — rows predating hash chaining, unverifiable
        head_hash       : str | None — chain head, for external anchoring
        broken_at       : dict | None — first failure: {id, seq, reason}
        detail          : str   — human-readable one-liner for the UI
    """
    init_audit_db(db_path)
    rows = _fetch_rows(run_id, db_path)

    if not rows:
        return {
            "verified": True,
            "entries_total": 0,
            "entries_checked": 0,
            "legacy_entries": 0,
            "head_hash": None,
            "broken_at": None,
            "detail": "No audit entries recorded for this run.",
        }

    expected_prev = GENESIS_HASH
    checked = 0
    legacy = 0
    broken: Optional[dict] = None
    head_hash = None

    for row in rows:
        (entry_id, r_id, ts, ev_type, ev_source, summary,
         det_json, seq, prev_hash, entry_hash) = row

        if entry_hash == LEGACY_MARKER or not entry_hash:
            # Written before chaining existed. Cannot be verified, and cannot be
            # used as an anchor for what follows.
            legacy += 1
            expected_prev = LEGACY_MARKER
            continue

        if prev_hash != expected_prev:
            broken = {
                "id": entry_id,
                "seq": seq,
                "reason": (
                    f"chain link broken — entry declares prev_hash "
                    f"{prev_hash[:12]}... but the preceding entry hashes to "
                    f"{str(expected_prev)[:12]}... (an entry was removed, "
                    f"reordered, or inserted)"
                ),
            }
            break

        recomputed = compute_entry_hash(
            run_id=r_id,
            seq=seq,
            timestamp=ts,
            event_type=ev_type,
            event_source=ev_source,
            summary=summary,
            details_json=det_json,
            prev_hash=prev_hash,
        )
        if recomputed != entry_hash:
            broken = {
                "id": entry_id,
                "seq": seq,
                "reason": (
                    "content modified — the stored hash does not match a hash "
                    "recomputed from this entry's own fields"
                ),
            }
            break

        checked += 1
        expected_prev = entry_hash
        head_hash = entry_hash

    verified = broken is None

    if broken is not None:
        detail = f"TAMPERING DETECTED at entry #{broken['seq']}: {broken['reason']}"
    elif legacy and checked:
        detail = (
            f"Chain verified across {checked} entries "
            f"({legacy} earlier entry/entries predate hash chaining and cannot "
            f"be verified)."
        )
    elif legacy:
        detail = (
            f"{legacy} entry/entries predate hash chaining and cannot be verified."
        )
    else:
        detail = f"Chain verified: {checked} entries intact and correctly linked."

    return {
        "verified": verified,
        "entries_total": len(rows),
        "entries_checked": checked,
        "legacy_entries": legacy,
        "head_hash": head_hash,
        "broken_at": broken,
        "detail": detail,
    }
