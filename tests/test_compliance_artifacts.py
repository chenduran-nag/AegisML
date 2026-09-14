"""
Unit tests for compliance_artifacts.py.

These exercise the generators directly on hand-built state dicts, with no graph
and no LangGraph import, so they also pin down the degraded paths: partial state,
an unevaluated fairness result, a run whose model was never saved.
"""

from __future__ import annotations

import json
import os

import pytest

from compliance_artifacts import (
    NOT_RECORDED,
    build_aibom,
    build_model_card,
    build_technical_documentation,
    fairness_verdict,
    generate_artifacts,
    render_model_card_md,
    sha256_file,
    verify_artifacts,
)


@pytest.fixture
def approved_state():
    """A realistic post-approval state slice."""
    return {
        "target_column": "income",
        "task_type": "classification",
        "business_objective": "Maximise recall on the high-income class",
        "dataset_sha256": "d" * 64,
        "plan": {
            "data_quality_concerns": ["High null rate in 'notes': 25.0% missing"],
            "recommended_preprocessing_steps": ["Impute hours using median"],
            "recommended_models": ["LogisticRegression", "RandomForest"],
            "sensitive_attribute_candidates": ["sex", "race"],
            "reasoning": "Tree ensembles suit mixed-type tabular data.",
        },
        "planner_meta": {
            "model_id": "openai/gpt-oss-20b",
            "system_prompt_sha256": "e" * 64,
            "user_prompt_sha256": "f" * 64,
            "token_usage": {"total_tokens": 1234},
        },
        "eda_report": {"summary": {"total_rows": 400, "total_columns": 7,
                                   "missing_pct": 4.29}},
        "data_agent_result": {
            "quality_check_passed": True,
            "quality_report": {"train_rows": 320, "test_rows": 80,
                               "rows_dropped": 0, "columns_dropped": [],
                               "class_balance_ratio": 1.88},
            "actions_taken": ["Imputed 'hours' with train-split median 40.89"],
        },
        "training_result": {
            "selected_model_name": "RandomForest",
            "selected_model_metrics": {"accuracy": 0.71, "f1": 0.42, "auc_roc": 0.66},
            "leaderboard": [
                {"model_name": "RandomForest", "metrics": {"auc_roc": 0.66},
                 "trained_successfully": True},
                {"model_name": "FakeBoost", "metrics": {},
                 "trained_successfully": False, "skip_reason": "no registry entry"},
            ],
            "shap_summary": [{"feature": "age", "importance": 0.21}],
        },
        "fairness_result": {
            "overall_fairness_passed": False,
            "fairness_evaluated": True,
            "evaluated_rows": 80,
            "fairness_report": [{
                "attribute": "sex", "disparate_impact": 0.30,
                "demographic_parity_difference": 0.07, "violation": True,
                "group_details": {"group_a": "Male", "group_b": "Female"},
            }],
            "attributes_skipped": ["age (continuous numeric feature)"],
        },
        "human_decision": "approve",
        "human_feedback": "Accepted for coursework only.",
        "retry_count": 0,
        "rejection_reroute_count": 1,
        "unresolved_quality_issue": False,
        "unresolved_human_rejection": False,
        "model_saved_path": None,
    }


# ---------------------------------------------------------------------------
# fairness_verdict — invariant 4
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("passed,expected", [
    (True, "PASSED"),
    (False, "VIOLATION DETECTED"),
    (None, "NOT EVALUATED"),
])
def test_fairness_verdict_has_three_states(passed, expected):
    assert fairness_verdict({"overall_fairness_passed": passed}) == expected


def test_fairness_verdict_of_missing_result_is_not_a_pass():
    assert fairness_verdict(None) == "NOT EVALUATED"
    assert fairness_verdict({}) == "NOT EVALUATED"


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


def test_model_card_reports_recorded_values(approved_state):
    card = build_model_card(approved_state, "run-1")

    assert card["model_details"]["selected_model"] == "RandomForest"
    assert card["training_data"]["train_rows"] == 320
    assert card["training_data"]["test_rows"] == 80
    assert card["evaluation"]["metrics"]["auc_roc"] == 0.66
    assert card["fairness"]["verdict"] == "VIOLATION DETECTED"
    assert card["human_governance"]["human_reroutes"] == 1
    assert card["explainability"]["top_features"][0]["feature"] == "age"


def test_model_card_reconstructs_decision_history(approved_state):
    trail = [
        {"event_type": "human_decision", "timestamp": "2026-09-14T10:00:00+00:00",
         "details": {"human_decision": "reject_model_or_fairness",
                     "human_feedback": "try another model"}},
        {"event_type": "training_run", "timestamp": "x", "details": {}},
        {"event_type": "human_decision", "timestamp": "2026-09-14T10:05:00+00:00",
         "details": {"human_decision": "approve", "human_feedback": ""}},
    ]
    card = build_model_card(approved_state, "run-1", audit_trail=trail)
    history = card["human_governance"]["decision_history"]

    assert [d["decision"] for d in history] == [
        "reject_model_or_fairness", "approve"
    ], "a card showing only the approval would hide the earlier rejection"


def test_unevaluated_fairness_becomes_an_explicit_limitation(approved_state):
    approved_state["fairness_result"] = {
        "overall_fairness_passed": None, "fairness_evaluated": False,
        "fairness_report": [], "attributes_skipped": ["all (regression task)"],
    }
    card = build_model_card(approved_state, "run-1")

    assert card["fairness"]["verdict"] == "NOT EVALUATED"
    assert any("NOT evaluated" in lim for lim in card["limitations"])

    md = render_model_card_md(card)
    assert "NOT EVALUATED" in md
    assert "PASSED" not in md.split("## 5. Fairness")[1].split("## 6.")[0]


def test_capped_quality_run_is_flagged_in_limitations(approved_state):
    approved_state["unresolved_quality_issue"] = True
    card = build_model_card(approved_state, "run-1")
    assert any("retry cap" in lim for lim in card["limitations"])


def test_model_card_survives_empty_state():
    """Artifacts must still generate for a partial run, marking gaps explicitly."""
    card = build_model_card({}, "run-empty")
    assert card["model_details"]["selected_model"] == NOT_RECORDED
    assert card["fairness"]["verdict"] == "NOT EVALUATED"
    assert render_model_card_md(card)


def test_rendered_card_is_markdown_with_expected_sections(approved_state):
    md = render_model_card_md(build_model_card(approved_state, "run-1"))
    for heading in ("# Model Card", "## 1. Model details", "## 3. Training data",
                    "## 5. Fairness", "## 7. Human governance", "## 8. Limitations"):
        assert heading in md


# ---------------------------------------------------------------------------
# AIBOM
# ---------------------------------------------------------------------------


def test_aibom_records_provenance(approved_state):
    aibom = build_aibom(approved_state, "run-1",
                        chain={"head_hash": "9" * 64, "verified": True})

    assert aibom["dataset"]["sha256"] == "d" * 64
    assert aibom["llm"]["model_id"] == "openai/gpt-oss-20b"
    assert aibom["llm"]["token_usage"]["total_tokens"] == 1234
    assert "planning only" in aibom["llm"]["role"]
    assert aibom["governance"]["audit_chain_head"] == "9" * 64
    assert aibom["runtime"]["libraries"]["pandas"] != "not installed"
    assert aibom["runtime"]["python"]


def test_aibom_hashes_the_model_file(approved_state, tmp_path):
    model = tmp_path / "m.joblib"
    model.write_bytes(b"not really a model")
    approved_state["model_saved_path"] = str(model)

    aibom = build_aibom(approved_state, "run-1")
    assert aibom["model"]["sha256"] == sha256_file(str(model))
    assert aibom["model"]["size_bytes"] == 18


def test_aibom_tolerates_a_missing_model_file(approved_state):
    approved_state["model_saved_path"] = "does/not/exist.joblib"
    aibom = build_aibom(approved_state, "run-1")
    assert aibom["model"]["sha256"] is None


# ---------------------------------------------------------------------------
# Annex IV draft
# ---------------------------------------------------------------------------


def test_technical_documentation_is_labelled_as_a_draft(approved_state):
    card = build_model_card(approved_state, "run-1")
    aibom = build_aibom(approved_state, "run-1")
    doc = build_technical_documentation(approved_state, card, aibom)

    assert "NOT A CONFORMITY ASSESSMENT" in doc
    # Match single words: the surrounding sentence is line-wrapped inside a
    # markdown blockquote, so multi-word phrases can straddle a "\n> " boundary.
    assert "notified body" in doc.lower()
    assert "no conformity assessment has been carried out" in doc.lower()


def test_technical_documentation_covers_all_nine_annex_iv_sections(approved_state):
    card = build_model_card(approved_state, "run-1")
    aibom = build_aibom(approved_state, "run-1")
    doc = build_technical_documentation(approved_state, card, aibom)

    for n, fragment in [
        (1, "General description of the AI system"),
        (2, "Detailed description of elements and development process"),
        (3, "Monitoring, functioning and control"),
        (4, "Appropriateness of performance metrics"),
        (5, "Risk management system"),
        (6, "Relevant changes through the lifecycle"),
        (7, "Harmonised standards applied"),
        (8, "EU declaration of conformity"),
        (9, "Post-market monitoring plan"),
    ]:
        assert f"## {n}. {fragment}" in doc, f"Annex IV section {n} missing"


def test_technical_documentation_states_the_known_gaps(approved_state):
    card = build_model_card(approved_state, "run-1")
    aibom = build_aibom(approved_state, "run-1")
    doc = build_technical_documentation(approved_state, card, aibom)

    assert "OUT OF SCOPE" in doc          # declaration of conformity
    assert "NOT IMPLEMENTED" in doc       # post-market monitoring
    assert "no post-deployment monitoring" in doc.lower()
    assert "reviewer identity is not yet authenticated" in doc


# ---------------------------------------------------------------------------
# Generation and verification without the graph
# ---------------------------------------------------------------------------


def test_generate_artifacts_writes_matching_digests(approved_state, tmp_path, audit_db):
    manifest = generate_artifacts(
        approved_state, "run-gen",
        out_root=str(tmp_path / "artifacts"), audit_db_path=audit_db,
    )

    assert manifest["errors"] == []
    assert manifest["directory"].endswith(os.path.join("artifacts", "run-gen"))

    for name, meta in manifest["files"].items():
        assert os.path.isfile(meta["path"])
        # The recorded digest must match the bytes on disk, since verification
        # re-hashes the file rather than the in-memory text.
        assert sha256_file(meta["path"]) == meta["sha256"], name

    card = json.load(open(
        os.path.join(manifest["directory"], "model_card.json"), encoding="utf-8"))
    assert card["run_id"] == "run-gen"


def test_verify_artifacts_returns_none_when_nothing_was_generated(audit_db):
    result = verify_artifacts("never-ran", audit_db_path=audit_db)
    assert result["verified"] is None
    assert "No compliance artifacts" in result["detail"]
