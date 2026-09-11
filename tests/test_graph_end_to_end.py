"""
End-to-end graph tests with the LLM planner mocked out.

Covers the governance gate (interrupt -> resume), the reroute loops, and the
model-serialisation fix — none of which need a Groq key to exercise.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")

import pipeline_graph
from graph_state import df_to_bytes
from langgraph.types import Command


@pytest.fixture
def graph(tmp_path, monkeypatch, fake_plan):
    """A graph wired to temporary databases and a static plan."""
    audit_db = str(tmp_path / "audit.db")

    monkeypatch.setattr(
        pipeline_graph, "plan_pipeline",
        lambda **kwargs: dict(fake_plan),
    )
    # log_audit_event was imported by value, so its default db_path is already
    # bound — redirect the name the module actually calls.
    real_log = pipeline_graph.log_audit_event
    monkeypatch.setattr(
        pipeline_graph, "log_audit_event",
        lambda **kw: real_log(db_path=audit_db, **kw),
    )
    monkeypatch.setattr(
        pipeline_graph, "SAVED_MODELS_DIR", str(tmp_path / "saved_models")
    )

    g = pipeline_graph.build_graph(db_path=str(tmp_path / "state.db"))
    return g, audit_db, str(tmp_path / "saved_models")


def _initial_state(toy_df):
    return {
        "df_bytes": df_to_bytes(toy_df),
        "target_column": "income",
        "task_type": "classification",
        "business_objective": "",
        "retry_count": 0,
        "unresolved_quality_issue": False,
        "last_failure_reason": None,
        "human_decision": None,
        "human_feedback": None,
        "rejection_reroute_count": 0,
        "unresolved_human_rejection": False,
        "rejected_models": [],
    }


# ---------------------------------------------------------------------------
# The governance gate
# ---------------------------------------------------------------------------


def test_pipeline_pauses_at_the_governance_gate(graph, toy_df):
    g, _, _ = graph
    config = {"configurable": {"thread_id": "t-pause"}}
    g.invoke(_initial_state(toy_df), config=config)

    snapshot = g.get_state(config)
    assert "human_approval_node" in snapshot.next, "graph should be paused for review"

    payload = snapshot.tasks[0].interrupts[0].value
    assert set(payload["allowed_decisions"]) == {
        "approve", "reject_data_quality", "reject_model_or_fairness"
    }
    assert payload["selected_model_name"] in ("LogisticRegression", "RandomForest")
    assert "fairness_evaluated" in payload


def test_split_is_carried_through_state(graph, toy_df):
    g, _, _ = graph
    config = {"configurable": {"thread_id": "t-split"}}
    g.invoke(_initial_state(toy_df), config=config)

    values = g.get_state(config).values
    split = values["split_index"]
    assert set(split) == {"train", "test"}
    assert set(split["train"]).isdisjoint(split["test"])

    # And it must NOT have leaked into the audit-log-bound result dict.
    assert "train_index" not in values["data_agent_result"]
    assert "test_index" not in values["data_agent_result"]


# ---------------------------------------------------------------------------
# Approval actually writes a model to disk
# ---------------------------------------------------------------------------


def test_approval_serialises_the_model_to_disk(graph, toy_df):
    g, _, models_dir = graph
    config = {"configurable": {"thread_id": "t-approve"}}
    g.invoke(_initial_state(toy_df), config=config)
    g.invoke(Command(resume={"decision": "approve", "human_feedback": ""}), config=config)

    values = g.get_state(config).values
    saved = values.get("model_saved_path")

    assert saved is not None, "an approved run must produce a real artifact path"
    assert os.path.isfile(saved), f"{saved} does not exist on disk"
    assert os.path.getsize(saved) > 0
    assert values.get("model_save_error") is None

    # And it must be loadable back into a usable estimator.
    import joblib
    model = joblib.load(saved)
    assert hasattr(model, "predict")


def test_rejected_run_writes_no_artifact(graph, toy_df):
    """Only approved models become artifacts."""
    g, _, models_dir = graph
    config = {"configurable": {"thread_id": "t-reject"}}
    g.invoke(_initial_state(toy_df), config=config)
    g.invoke(
        Command(resume={"decision": "reject_model_or_fairness", "human_feedback": ""}),
        config=config,
    )

    values = g.get_state(config).values
    assert values.get("model_saved_path") is None
    if os.path.isdir(models_dir):
        assert os.listdir(models_dir) == []


# ---------------------------------------------------------------------------
# Reroute loops
# ---------------------------------------------------------------------------


def test_model_rejection_excludes_the_model_and_repauses(graph, toy_df):
    g, _, _ = graph
    config = {"configurable": {"thread_id": "t-loop3"}}
    g.invoke(_initial_state(toy_df), config=config)

    first_choice = g.get_state(config).values["training_result"]["selected_model_name"]
    g.invoke(
        Command(resume={"decision": "reject_model_or_fairness", "human_feedback": ""}),
        config=config,
    )

    values = g.get_state(config).values
    assert first_choice in values["rejected_models"]
    assert values["rejection_reroute_count"] == 1
    assert values["training_result"]["selected_model_name"] != first_choice
    assert "human_approval_node" in g.get_state(config).next, "should pause again"


def test_data_quality_rejection_reroutes_to_planner(graph, toy_df):
    g, _, _ = graph
    config = {"configurable": {"thread_id": "t-loop2"}}
    g.invoke(_initial_state(toy_df), config=config)
    g.invoke(
        Command(resume={
            "decision": "reject_data_quality",
            "human_feedback": "Drop the notes column entirely.",
        }),
        config=config,
    )

    values = g.get_state(config).values
    assert values["rejection_reroute_count"] == 1
    assert values["last_failure_reason"]["source"] == "human_reviewer"
    assert values["human_feedback"] == "Drop the notes column entirely."


# ---------------------------------------------------------------------------
# Audit trail of a full run
# ---------------------------------------------------------------------------


def test_full_run_produces_a_verified_audit_chain(graph, toy_df):
    from audit_log import get_audit_trail, verify_audit_chain

    g, audit_db, _ = graph
    config = {"configurable": {"thread_id": "t-audit"}}
    g.invoke(_initial_state(toy_df), config=config)
    g.invoke(Command(resume={"decision": "approve", "human_feedback": ""}), config=config)

    trail = get_audit_trail("t-audit", db_path=audit_db)
    event_types = [e["event_type"] for e in trail]

    for expected in ("planner_run", "data_agent_run", "training_run",
                     "fairness_run", "human_decision", "final_outcome"):
        assert expected in event_types, f"missing {expected} in {event_types}"

    assert any(e["event_source"] == "human_reviewer" for e in trail)

    chain = verify_audit_chain("t-audit", db_path=audit_db)
    assert chain["verified"] is True
    assert chain["entries_checked"] == len(trail)

    final = [e for e in trail if e["event_type"] == "final_outcome"][-1]
    assert final["details"]["model_saved_path"] is not None
    assert final["details"]["status"] == "APPROVED"
