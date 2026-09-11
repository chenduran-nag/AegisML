"""
Tests for the tamper-evident audit log (audit_log.py).

These cover the claim the README makes — that the audit trail is immutable — by
actually tampering with the database and asserting the chain notices.
"""

from __future__ import annotations

import sqlite3

import pytest

from audit_log import (
    GENESIS_HASH,
    LEGACY_MARKER,
    get_audit_trail,
    init_audit_db,
    log_audit_event,
    verify_audit_chain,
)


def _log_three(db: str, run_id: str = "run-test") -> None:
    log_audit_event(run_id, "planner_run", "automated", "Planner produced a plan",
                    {"models": ["RandomForest"]}, db_path=db)
    log_audit_event(run_id, "training_run", "automated", "Trained 2 models",
                    {"auc": 0.81}, db_path=db)
    log_audit_event(run_id, "human_decision", "human_reviewer", "Reviewer approved",
                    {"decision": "approve"}, db_path=db)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_untouched_chain_verifies(audit_db):
    _log_three(audit_db)
    result = verify_audit_chain("run-test", db_path=audit_db)

    assert result["verified"] is True
    assert result["entries_checked"] == 3
    assert result["legacy_entries"] == 0
    assert result["broken_at"] is None
    assert result["head_hash"] is not None


def test_first_entry_anchors_to_genesis(audit_db):
    _log_three(audit_db)
    entries = get_audit_trail("run-test", db_path=audit_db)

    assert entries[0]["prev_hash"] == GENESIS_HASH
    assert [e["seq"] for e in entries] == [0, 1, 2]
    # Each entry links to the one before it.
    for prev, curr in zip(entries, entries[1:]):
        assert curr["prev_hash"] == prev["entry_hash"]


def test_runs_are_independently_chained(audit_db):
    _log_three(audit_db, run_id="run-a")
    _log_three(audit_db, run_id="run-b")

    assert verify_audit_chain("run-a", db_path=audit_db)["verified"] is True
    assert verify_audit_chain("run-b", db_path=audit_db)["verified"] is True
    assert verify_audit_chain("run-a", db_path=audit_db)["entries_checked"] == 3


def test_empty_run_is_vacuously_verified(audit_db):
    init_audit_db(audit_db)
    result = verify_audit_chain("no-such-run", db_path=audit_db)
    assert result["verified"] is True
    assert result["entries_total"] == 0


# ---------------------------------------------------------------------------
# Tampering
# ---------------------------------------------------------------------------


def test_editing_a_summary_is_detected(audit_db):
    """The headline demo: rewrite a reviewer's decision, chain goes red."""
    _log_three(audit_db)

    conn = sqlite3.connect(audit_db)
    conn.execute(
        "UPDATE audit_entries SET summary = ? WHERE event_type = 'human_decision'",
        ("Reviewer rejected the model",),
    )
    conn.commit()
    conn.close()

    result = verify_audit_chain("run-test", db_path=audit_db)
    assert result["verified"] is False
    assert result["broken_at"]["seq"] == 2
    assert "content modified" in result["broken_at"]["reason"]


def test_editing_details_is_detected(audit_db):
    _log_three(audit_db)

    conn = sqlite3.connect(audit_db)
    conn.execute(
        "UPDATE audit_entries SET details_json = ? WHERE event_type = 'training_run'",
        ('{"auc": 0.99}',),
    )
    conn.commit()
    conn.close()

    result = verify_audit_chain("run-test", db_path=audit_db)
    assert result["verified"] is False
    assert result["broken_at"]["seq"] == 1


def test_backdating_a_timestamp_is_detected(audit_db):
    _log_three(audit_db)

    conn = sqlite3.connect(audit_db)
    conn.execute(
        "UPDATE audit_entries SET timestamp = ? WHERE seq = 0",
        ("1999-01-01T00:00:00+00:00",),
    )
    conn.commit()
    conn.close()

    assert verify_audit_chain("run-test", db_path=audit_db)["verified"] is False


def test_deleting_a_middle_entry_is_detected(audit_db):
    _log_three(audit_db)

    conn = sqlite3.connect(audit_db)
    conn.execute("DELETE FROM audit_entries WHERE seq = 1")
    conn.commit()
    conn.close()

    result = verify_audit_chain("run-test", db_path=audit_db)
    assert result["verified"] is False
    assert "chain link broken" in result["broken_at"]["reason"]


def test_swapping_event_source_is_detected(audit_db):
    """Relabelling an automated event as a human sign-off must not go unnoticed."""
    _log_three(audit_db)

    conn = sqlite3.connect(audit_db)
    conn.execute(
        "UPDATE audit_entries SET event_source = 'human_reviewer' WHERE seq = 0"
    )
    conn.commit()
    conn.close()

    assert verify_audit_chain("run-test", db_path=audit_db)["verified"] is False


# ---------------------------------------------------------------------------
# Documented limitation — kept as a test so nobody claims otherwise later
# ---------------------------------------------------------------------------


def test_truncation_is_NOT_detected(audit_db):
    """
    Deleting trailing entries leaves a shorter but internally consistent chain.

    This is a real, documented limitation: nothing inside the file can prove
    entries once existed past its own head. Closing it needs an external anchor.
    This test exists so the limitation stays visible rather than being quietly
    assumed away.
    """
    _log_three(audit_db)

    conn = sqlite3.connect(audit_db)
    conn.execute("DELETE FROM audit_entries WHERE seq = 2")
    conn.commit()
    conn.close()

    result = verify_audit_chain("run-test", db_path=audit_db)
    assert result["verified"] is True          # <-- not a bug; a known gap
    assert result["entries_checked"] == 2


# ---------------------------------------------------------------------------
# Migration from a pre-chain database
# ---------------------------------------------------------------------------


def test_legacy_rows_are_reported_not_silently_trusted(tmp_path):
    """Rows written by the old schema must be flagged, never counted as verified."""
    db = str(tmp_path / "legacy.db")

    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE audit_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_source TEXT NOT NULL,
            summary TEXT NOT NULL,
            details_json TEXT NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO audit_entries (run_id, timestamp, event_type, event_source,"
        " summary, details_json) VALUES (?, ?, ?, ?, ?, ?)",
        ("run-old", "2026-01-01T00:00:00+00:00", "planner_run", "automated",
         "Old entry", "{}"),
    )
    conn.commit()
    conn.close()

    # init_audit_db must migrate the table rather than crash on the new columns.
    init_audit_db(db)
    result = verify_audit_chain("run-old", db_path=db)

    assert result["legacy_entries"] == 1
    assert result["entries_checked"] == 0
    assert "predate hash chaining" in result["detail"]


def test_new_entries_append_cleanly_after_migration(tmp_path):
    db = str(tmp_path / "mixed.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE audit_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL, timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL, event_source TEXT NOT NULL,
            summary TEXT NOT NULL, details_json TEXT NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO audit_entries (run_id, timestamp, event_type, event_source,"
        " summary, details_json) VALUES ('run-x','t','planner_run','automated','old','{}')"
    )
    conn.commit()
    conn.close()

    log_audit_event("run-x", "training_run", "automated", "new entry", {}, db_path=db)
    trail = get_audit_trail("run-x", db_path=db)

    assert len(trail) == 2
    assert trail[0]["entry_hash"] == LEGACY_MARKER
    assert trail[1]["entry_hash"] != LEGACY_MARKER
