"""
End-to-end graph tests with the LLM planner mocked out.

Covers the governance gate (interrupt -> resume), the reroute loops, model
serialisation, and compliance artifact generation — none of which need a Groq key.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")

import pipeline_graph
from graph_state import df_to_bytes
from langgraph.types import Command


@pytest.fixture
def graph(tmp_path, monkeypatch, fake_plan):
    """
    A graph wired to temporary databases and directories, with a static plan.

    `AUDIT_DB_PATH` and the two output directories are module-level constants read
    at call time, so repointing them here keeps every run fully inside tmp_path.
    """
    def _fake_plan_pipeline(**kwargs):
        # Populate meta_out the way the real planner does, so artifact provenance
        # has something to record.
        meta = kwargs.get("meta_out")
        if meta is not None:
            meta.update({
                "model_id": "stub/test-model",
                "system_prompt_sha256": "a" * 64,
                "user_prompt_sha256": "b" * 64,
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                "total_tokens": 15},
                "attempts": 1,
            })
        return dict(fake_plan)

    env = SimpleNamespace(
        audit_db=str(tmp_path / "audit.db"),
        models_dir=str(tmp_path / "saved_models"),
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    monkeypatch.setattr(pipeline_graph, "plan_pipeline", _fake_plan_pipeline)
    monkeypatch.setattr(pipeline_graph, "AUDIT_DB_PATH", env.audit_db)
    monkeypatch.setattr(pipeline_graph, "SAVED_MODELS_DIR", env.models_dir)
    monkeypatch.setattr(pipeline_graph, "ARTIFACTS_DIR", env.artifacts_dir)

    env.g = pipeline_graph.build_graph(db_path=str(tmp_path / "state.db"))
    return env


def _initial_state(toy_df):
    return {
        "df_bytes": df_to_bytes(toy_df),
        "target_column": "income",
        "task_type": "classification",
        "business_objective": "Maximise recall on the high-income class",
        "dataset_sha256": "c" * 64,
        "retry_count": 0,
        "unresolved_quality_issue": False,
        "last_failure_reason": None,
        "human_decision": None,
        "human_feedback": None,
        "rejection_reroute_count": 0,
        "unresolved_human_rejection": False,
        "rejected_models": [],
    }


def _run_to_gate(env, toy_df, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    env.g.invoke(_initial_state(toy_df), config=config)
    return config


# ---------------------------------------------------------------------------
# The governance gate
# ---------------------------------------------------------------------------


def test_pipeline_pauses_at_the_governance_gate(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-pause")
    snapshot = graph.g.get_state(config)
    assert "human_approval_node" in snapshot.next, "graph should be paused for review"

    payload = snapshot.tasks[0].interrupts[0].value
    assert set(payload["allowed_decisions"]) == {
        "approve", "reject_data_quality", "reject_model_or_fairness"
    }
    assert payload["selected_model_name"] in ("LogisticRegression", "RandomForest")
    assert "fairness_evaluated" in payload


def test_split_is_carried_through_state(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-split")
    values = graph.g.get_state(config).values

    split = values["split_index"]
    assert set(split) == {"train", "test"}
    assert set(split["train"]).isdisjoint(split["test"])

    # And it must NOT have leaked into the audit-log-bound result dict.
    assert "train_index" not in values["data_agent_result"]
    assert "test_index" not in values["data_agent_result"]


def test_planner_provenance_is_recorded(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-meta")
    meta = graph.g.get_state(config).values["planner_meta"]

    assert meta["model_id"] == "stub/test-model"
    assert len(meta["system_prompt_sha256"]) == 64
    assert meta["token_usage"]["total_tokens"] == 15
    # Provenance must stay out of the plan itself.
    assert "model_id" not in graph.g.get_state(config).values["plan"]


# ---------------------------------------------------------------------------
# Approval writes a model AND its paperwork
# ---------------------------------------------------------------------------


def test_approval_serialises_the_model_to_disk(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-approve")
    graph.g.invoke(Command(resume={"decision": "approve", "human_feedback": ""}),
                   config=config)

    values = graph.g.get_state(config).values
    saved = values.get("model_saved_path")

    assert saved is not None, "an approved run must produce a real artifact path"
    assert os.path.isfile(saved), f"{saved} does not exist on disk"
    assert values.get("model_save_error") is None

    import joblib
    assert hasattr(joblib.load(saved), "predict")


def test_rejected_run_writes_no_model_and_no_artifacts(graph, toy_df):
    """Only approved models become artifacts — paperwork included."""
    config = _run_to_gate(graph, toy_df, "t-reject")
    graph.g.invoke(
        Command(resume={"decision": "reject_model_or_fairness", "human_feedback": ""}),
        config=config,
    )

    values = graph.g.get_state(config).values
    assert values.get("model_saved_path") is None
    assert values.get("artifacts_manifest") is None
    if os.path.isdir(graph.models_dir):
        assert os.listdir(graph.models_dir) == []
    assert not os.path.isdir(os.path.join(graph.artifacts_dir, "t-reject"))


# ---------------------------------------------------------------------------
# Compliance artifacts
# ---------------------------------------------------------------------------


def test_approval_generates_the_artifact_set(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-artifacts")
    graph.g.invoke(Command(resume={"decision": "approve", "human_feedback": "looks fine"}),
                   config=config)

    manifest = graph.g.get_state(config).values["artifacts_manifest"]
    assert manifest is not None
    assert manifest["errors"] == []
    assert set(manifest["files"]) == {
        "model_card.md", "model_card.json", "aibom.json",
        "technical_documentation.md",
    }
    for name, meta in manifest["files"].items():
        assert os.path.isfile(meta["path"]), f"{name} missing on disk"
        assert len(meta["sha256"]) == 64
        assert meta["size_bytes"] > 0


def test_artifact_digests_are_logged_and_verify(graph, toy_df):
    from compliance_artifacts import ARTIFACT_EVENT_TYPE, verify_artifacts
    from audit_log import get_audit_trail

    config = _run_to_gate(graph, toy_df, "t-verify")
    graph.g.invoke(Command(resume={"decision": "approve", "human_feedback": ""}),
                   config=config)

    trail = get_audit_trail("t-verify", db_path=graph.audit_db)
    events = [e for e in trail if e["event_type"] == ARTIFACT_EVENT_TYPE]
    assert len(events) == 1
    assert set(events[0]["details"]["files"]) == {
        "model_card.md", "model_card.json", "aibom.json",
        "technical_documentation.md",
    }

    result = verify_artifacts("t-verify", audit_db_path=graph.audit_db)
    assert result["verified"] is True
    assert result["chain"]["verified"] is True
    assert all(f["status"] == "OK" for f in result["files"])


def test_editing_an_artifact_is_detected(graph, toy_df):
    """The paperwork inherits the audit log's tamper evidence."""
    from compliance_artifacts import verify_artifacts

    config = _run_to_gate(graph, toy_df, "t-tamper")
    graph.g.invoke(Command(resume={"decision": "approve", "human_feedback": ""}),
                   config=config)

    card = os.path.join(graph.artifacts_dir, "t-tamper", "model_card.md")
    with open(card, "a", encoding="utf-8") as fh:
        fh.write("\n\nFairness verdict: PASSED (definitely)\n")

    result = verify_artifacts("t-tamper", audit_db_path=graph.audit_db)
    assert result["verified"] is False
    assert any(f["status"] == "MODIFIED" and f["file"] == "model_card.md"
               for f in result["files"])
    assert "FAILED" in result["detail"]


def test_deleting_an_artifact_is_detected(graph, toy_df):
    from compliance_artifacts import verify_artifacts

    config = _run_to_gate(graph, toy_df, "t-del")
    graph.g.invoke(Command(resume={"decision": "approve", "human_feedback": ""}),
                   config=config)

    os.remove(os.path.join(graph.artifacts_dir, "t-del", "aibom.json"))
    result = verify_artifacts("t-del", audit_db_path=graph.audit_db)

    assert result["verified"] is False
    assert any(f["status"] == "MISSING" for f in result["files"])


def test_artifacts_record_the_real_run_content(graph, toy_df):
    """Spot-check that the paperwork describes this run rather than placeholders."""
    config = _run_to_gate(graph, toy_df, "t-content")
    graph.g.invoke(
        Command(resume={"decision": "approve",
                        "human_feedback": "approved with reservations"}),
        config=config,
    )
    values = graph.g.get_state(config).values
    out = os.path.join(graph.artifacts_dir, "t-content")

    card = json.load(open(os.path.join(out, "model_card.json"), encoding="utf-8"))
    assert card["model_details"]["selected_model"] == \
        values["training_result"]["selected_model_name"]
    assert card["model_details"]["artifact_path"] == values["model_saved_path"]
    assert card["training_data"]["test_rows"] == len(values["split_index"]["test"])
    assert card["human_governance"]["final_decision"] == "approve"
    assert card["human_governance"]["decision_history"], "decisions not reconstructed"
    assert card["intended_use"]["stated_objective"].startswith("Maximise recall")

    aibom = json.load(open(os.path.join(out, "aibom.json"), encoding="utf-8"))
    assert aibom["dataset"]["sha256"] == "c" * 64
    assert aibom["llm"]["model_id"] == "stub/test-model"
    assert aibom["model"]["sha256"], "model file was not hashed"
    assert aibom["governance"]["audit_chain_verified"] is True
    assert aibom["runtime"]["libraries"]["pandas"] != "not installed"

    md = open(os.path.join(out, "model_card.md"), encoding="utf-8").read()
    assert values["training_result"]["selected_model_name"] in md

    techdoc = open(os.path.join(out, "technical_documentation.md"),
                   encoding="utf-8").read()
    assert "NOT A CONFORMITY ASSESSMENT" in techdoc
    for heading in ("## 1. General description",
                    "## 2. Detailed description",
                    "## 9. Post-market monitoring plan"):
        assert heading in techdoc


# ---------------------------------------------------------------------------
# Reroute loops
# ---------------------------------------------------------------------------


def test_model_rejection_excludes_the_model_and_repauses(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-loop3")
    first_choice = graph.g.get_state(config).values["training_result"]["selected_model_name"]

    graph.g.invoke(
        Command(resume={"decision": "reject_model_or_fairness", "human_feedback": ""}),
        config=config,
    )

    values = graph.g.get_state(config).values
    assert first_choice in values["rejected_models"]
    assert values["rejection_reroute_count"] == 1
    assert values["training_result"]["selected_model_name"] != first_choice
    assert "human_approval_node" in graph.g.get_state(config).next


def test_data_quality_rejection_reroutes_to_planner(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-loop2")
    graph.g.invoke(
        Command(resume={
            "decision": "reject_data_quality",
            "human_feedback": "Drop the notes column entirely.",
        }),
        config=config,
    )

    values = graph.g.get_state(config).values
    assert values["rejection_reroute_count"] == 1
    assert values["last_failure_reason"]["source"] == "human_reviewer"
    assert values["human_feedback"] == "Drop the notes column entirely."


def test_rejection_cap_terminates_without_approval_or_artifacts(graph, toy_df):
    """
    Three rejections hit MAX_HUMAN_REROUTES and end the run.

    The run then reports status "completed" with human_decision set to a
    *rejection* — which is exactly the state the dashboard used to render as
    "Model Formally Approved & Saved to Disk". Nothing about a capped run may
    look like an approval: no model on disk, no compliance artifacts.
    """
    config = _run_to_gate(graph, toy_df, "t-cap")

    for _ in range(3):
        graph.g.invoke(
            Command(resume={"decision": "reject_model_or_fairness",
                            "human_feedback": "not acceptable"}),
            config=config,
        )

    values = graph.g.get_state(config).values
    assert graph.g.get_state(config).next == (), "run should have ended"
    assert values["unresolved_human_rejection"] is True
    assert values["rejection_reroute_count"] == 2
    assert values["human_decision"] != "approve"

    assert values.get("model_saved_path") is None
    assert values.get("artifacts_manifest") is None
    assert not os.path.isdir(os.path.join(graph.artifacts_dir, "t-cap"))


# ---------------------------------------------------------------------------
# Audit trail of a full run
# ---------------------------------------------------------------------------


def test_full_run_produces_a_verified_audit_chain(graph, toy_df):
    from audit_log import get_audit_trail, verify_audit_chain

    config = _run_to_gate(graph, toy_df, "t-audit")
    graph.g.invoke(Command(resume={"decision": "approve", "human_feedback": ""}),
                   config=config)

    trail = get_audit_trail("t-audit", db_path=graph.audit_db)
    event_types = [e["event_type"] for e in trail]

    for expected in ("planner_run", "data_agent_run", "training_run",
                     "fairness_run", "human_decision", "final_outcome",
                     "compliance_artifacts_generated"):
        assert expected in event_types, f"missing {expected} in {event_types}"

    assert any(e["event_source"] == "human_reviewer" for e in trail)

    chain = verify_audit_chain("t-audit", db_path=graph.audit_db)
    assert chain["verified"] is True
    assert chain["entries_checked"] == len(trail)

    final = [e for e in trail if e["event_type"] == "final_outcome"][-1]
    assert final["details"]["model_saved_path"] is not None
    assert final["details"]["status"] == "APPROVED"
