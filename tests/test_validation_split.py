"""
Tests for the validation split.

Every gate decision — leaderboard ranking, the fairness audit, a reviewer's
rejection or mitigation — looks at the rows it is evaluated on. Those must be the
validation rows. The test rows stay untouched until the approved model is scored on
them once, in audit_log_node, and that final evaluation is what gets reported.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from data_agent import TEST_SIZE, VALIDATION_SIZE, run_data_agent
from training_agent import evaluate_model, run_training_agent


def test_the_split_has_three_disjoint_parts_of_the_expected_sizes(toy_df, fake_plan):
    cleaned = run_data_agent(toy_df, fake_plan, "income", "classification")
    total = len(cleaned["cleaned_df"])
    train, validation, test = (set(cleaned["train_index"]), set(cleaned["validation_index"]),
                               set(cleaned["test_index"]))

    assert not (train & validation) and not (train & test) and not (validation & test)
    assert len(train) + len(validation) + len(test) == total
    assert len(validation) / total == pytest.approx((1 - TEST_SIZE) * VALIDATION_SIZE, abs=0.02)

    report = cleaned["quality_report"]
    assert (report["train_rows"], report["validation_rows"], report["test_rows"]) == \
        (len(train), len(validation), len(test))


def test_evaluate_model_matches_the_training_agents_own_scoring(toy_df, fake_plan):
    cleaned = run_data_agent(toy_df, fake_plan, "income", "classification")
    trained = run_training_agent(
        cleaned_df=cleaned["cleaned_df"], target_column="income", task_type="classification",
        recommended_models=["LogisticRegression"],
        train_index=cleaned["train_index"], test_index=cleaned["validation_index"],
        eval_label="validation",
    )
    rescored = evaluate_model(trained["_fitted_model"], cleaned["cleaned_df"], "income",
                              "classification", cleaned["validation_index"])

    assert rescored == trained["selected_model_metrics"]
    assert any("validation rows" in a for a in trained["actions_taken"])


# ---------------------------------------------------------------------------
# Through the graph
# ---------------------------------------------------------------------------

pytest.importorskip("langgraph", reason="langgraph not installed")

from langgraph.types import Command  # noqa: E402

from tests.test_graph_end_to_end import _run_to_gate, graph  # noqa: E402,F401


def _approve(env, config):
    env.g.invoke(Command(resume={"decision": "approve", "human_feedback": ""}), config=config)


def test_the_gate_is_decided_on_validation_rows_and_test_rows_stay_untouched(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-val-gate")
    snapshot = graph.g.get_state(config)
    values = snapshot.values
    payload = snapshot.tasks[0].interrupts[0].value

    assert values["fairness_result"]["evaluated_rows"] == len(values["split_index"]["validation"])
    assert any("validation rows" in a for a in payload.get("data_agent_actions", []) +
               values["training_result"]["actions_taken"])
    assert not values.get("final_evaluation"), "test rows must not be scored before approval"


def test_approval_scores_the_approved_model_once_on_the_test_rows(graph, toy_df):
    from audit_log import get_audit_trail

    config = _run_to_gate(graph, toy_df, "t-val-final")
    _approve(graph, config)
    values = graph.g.get_state(config).values

    final = values["final_evaluation"]
    assert final["split"] == "test"
    assert final["rows"] == len(values["split_index"]["test"])
    assert final["fairness"]["evaluated_rows"] == len(values["split_index"]["test"])
    assert final["metrics"]["auc_roc"] is not None

    trail = get_audit_trail("t-val-final", db_path=graph.audit_db)
    types = [e["event_type"] for e in trail]
    (event,) = [e for e in trail if e["event_type"] == "final_test_evaluation"]
    assert types.index("human_decision") < types.index("final_test_evaluation")
    assert event["details"]["metrics"] == final["metrics"]
    outcome = [e for e in trail if e["event_type"] == "final_outcome"][-1]
    assert outcome["details"]["final_test_metrics"] == final["metrics"]


def test_a_rejected_run_never_scores_the_test_rows(graph, toy_df):
    from audit_log import get_audit_trail

    config = _run_to_gate(graph, toy_df, "t-val-reject")
    graph.g.invoke(Command(resume={"decision": "reject_model_or_fairness", "human_feedback": ""}),
                   config=config)
    trail = get_audit_trail("t-val-reject", db_path=graph.audit_db)
    assert "final_test_evaluation" not in [e["event_type"] for e in trail]
    assert not graph.g.get_state(config).values.get("final_evaluation")


def test_the_model_card_reports_the_final_test_evaluation(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-val-card")
    _approve(graph, config)
    values = graph.g.get_state(config).values
    run_dir = os.path.join(graph.artifacts_dir, "t-val-card")

    with open(os.path.join(run_dir, "model_card.json"), encoding="utf-8") as fh:
        card = json.load(fh)
    assert card["evaluation"]["final_test"]["metrics"] == values["final_evaluation"]["metrics"]
    assert card["evaluation"]["metrics"] == values["training_result"]["selected_model_metrics"]
    assert card["training_data"]["validation_rows"] == len(values["split_index"]["validation"])

    with open(os.path.join(run_dir, "model_card.md"), encoding="utf-8") as fh:
        markdown = fh.read()
    assert "Final evaluation on untouched test rows" in markdown


def test_the_evaluation_harness_reports_test_numbers_for_approved_runs():
    from experiments import run_governance_eval as ev

    values = {
        "human_decision": "approve",
        "training_result": {"selected_model_name": "LR",
                            "selected_model_metrics": {"auc_roc": 0.9, "accuracy": 0.8}},
        "fairness_result": {"fairness_evaluated": True, "overall_fairness_passed": True,
                            "fairness_report": [{"attribute": "sex", "protected": True,
                                                 "counts_toward_verdict": True,
                                                 "disparate_impact": 0.95, "violation": False}]},
        "final_evaluation": {
            "metrics": {"auc_roc": 0.85, "accuracy": 0.78},
            "fairness": {"fairness_evaluated": True, "overall_fairness_passed": False,
                         "fairness_report": [{"attribute": "sex", "protected": True,
                                              "counts_toward_verdict": True,
                                              "disparate_impact": 0.6, "violation": True}]},
        },
    }
    row = ev._final_metrics(values, gates=1)
    assert (row["auc_roc"], row["gate_auc_roc"], row["evaluated_on"]) == (0.85, 0.9, "test")
    assert row["overall_fairness_passed"] is False
    assert row["min_disparate_impact"] == 0.6
