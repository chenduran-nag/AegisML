"""
Tests for reweighing mitigation (mitigation.py) and the reject_and_mitigate decision.

The first half checks the arithmetic on frames small enough to work out by hand.
The second half drives the real graph: mitigation must count against the rejection
cap, fit its weights on train rows only, keep per-row data out of the audit log,
and show the reviewer before against after.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from data_agent import run_data_agent
from fairness_agent import run_fairness_agent
from mitigation import cell_weights, choose_attribute, group_labels, row_weights
from training_agent import run_training_agent


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def _skewed():
    """Group A: 45 positive / 15 negative. Group B: 5 positive / 35 negative."""
    groups = pd.Series(["A"] * 60 + ["B"] * 40)
    labels = pd.Series([1] * 45 + [0] * 15 + [1] * 5 + [0] * 35)
    return groups, labels


def test_weights_follow_the_reweighing_formula():
    cells = {(c["group"], c["label"]): c for c in cell_weights(*_skewed())}

    # P(A)=0.6, P(y=1)=0.5, P(A, y=1)=0.45
    assert cells[("A", 1)]["weight"] == pytest.approx(0.6 * 0.5 / 0.45, abs=1e-6)
    assert cells[("B", 1)]["weight"] == pytest.approx(0.4 * 0.5 / 0.05, abs=1e-6)
    assert cells[("B", 0)]["n"] == 35


def test_weighted_positive_rate_is_the_same_in_every_group():
    groups, labels = _skewed()
    lookup = {(c["group"], c["label"]): c["weight"] for c in cell_weights(groups, labels)}
    weights = pd.Series([lookup[(g, y)] for g, y in zip(groups, labels)])

    for group in ("A", "B"):
        mask = groups == group
        rate = (weights[mask] * labels[mask]).sum() / weights[mask].sum()
        assert rate == pytest.approx(labels.mean(), abs=1e-5)


def test_weights_are_fitted_on_train_rows_only():
    raw = pd.DataFrame({"sex": ["F", "M"] * 50})
    y = pd.Series([0, 1] * 50)
    train = list(range(80))

    _, cells = row_weights(raw, ["sex"], train, y)
    changed_test = y.copy()
    changed_test.iloc[80:] = 1 - changed_test.iloc[80:]
    _, cells_again = row_weights(raw, ["sex"], train, changed_test)

    assert cells == cells_again
    assert sum(c["n"] for c in cells) == 80


def test_age_is_grouped_in_the_audit_bands():
    raw = pd.DataFrame({"age": np.arange(18, 88)})
    assert set(group_labels(raw, ["age"])) == {"<25", "25-59", "60+"}


def test_a_second_attribute_is_mitigated_as_an_intersection():
    raw = pd.DataFrame({"sex": ["F", "M", "F", "M"], "race": ["X", "X", "Y", "Y"]})
    assert list(group_labels(raw, ["sex", "race"])) == ["F x X", "M x X", "F x Y", "M x Y"]


def test_an_attribute_that_cannot_be_grouped_is_refused_with_a_reason():
    raw = pd.DataFrame({"hours": np.linspace(0, 1, 50)})
    with pytest.raises(ValueError, match="not a column"):
        group_labels(raw, ["sex"])
    with pytest.raises(ValueError, match="only age has a banding rule"):
        group_labels(raw, ["hours"])


def _entry(attribute, di, protected, violation=True):
    return {"attribute": attribute, "disparate_impact": di, "protected": protected,
            "violation": violation}


def test_the_worst_protected_violation_is_chosen_first():
    fairness = {"fairness_report": [
        _entry("job", 0.1, False), _entry("sex", 0.5, True), _entry("race", 0.3, True),
        _entry("age", 0.05, True, violation=False),
    ]}
    assert choose_attribute(fairness, []) == "race"
    assert choose_attribute(fairness, ["race"]) == "sex"
    # `job` is advisory: reweighing it cannot change the verdict, so it is never chosen.
    assert choose_attribute(fairness, ["race", "sex"]) is None


# ---------------------------------------------------------------------------
# Training with weights, and whether it helps
# ---------------------------------------------------------------------------


def test_training_rejects_weights_that_do_not_cover_the_train_rows(toy_df, fake_plan):
    cleaned = run_data_agent(toy_df, fake_plan, "income", "classification")
    partial = pd.Series(1.0, index=pd.Index(cleaned["train_index"][:10]))
    with pytest.raises(ValueError, match="no weight"):
        run_training_agent(
            cleaned_df=cleaned["cleaned_df"], target_column="income",
            task_type="classification", recommended_models=["LogisticRegression"],
            train_index=cleaned["train_index"], test_index=cleaned["test_index"],
            sample_weight=partial,
        )


def _sex_disparate_impact(toy_df, fake_plan, reweigh: bool):
    cleaned = run_data_agent(toy_df, fake_plan, "income", "classification")
    frame = cleaned["cleaned_df"]
    weights = None
    if reweigh:
        weights, _ = row_weights(toy_df, ["sex"], cleaned["train_index"], frame["income"])
    trained = run_training_agent(
        cleaned_df=frame, target_column="income", task_type="classification",
        recommended_models=["LogisticRegression"],
        train_index=cleaned["train_index"], test_index=cleaned["test_index"],
        sample_weight=weights,
    )
    fairness = run_fairness_agent(
        cleaned_df=frame, fitted_model=trained["_fitted_model"], target_column="income",
        sensitive_attribute_candidates=["sex"], task_type="classification",
        eval_index=cleaned["test_index"], raw_frame=toy_df,
    )
    entry = next(r for r in fairness["fairness_report"] if r["attribute"] == "sex")
    return entry["disparate_impact"], trained


def test_reweighing_on_sex_raises_its_disparate_impact(toy_df, fake_plan):
    """toy_df's target depends on sex, so the unweighted model is skewed by sex."""
    before, _ = _sex_disparate_impact(toy_df, fake_plan, reweigh=False)
    after, trained = _sex_disparate_impact(toy_df, fake_plan, reweigh=True)

    assert after > before, f"reweighing did not raise DI on sex: {before} -> {after}"
    assert any("bias-mitigation sample weights" in a for a in trained["actions_taken"])


# ---------------------------------------------------------------------------
# The reject_and_mitigate decision through the real graph
# ---------------------------------------------------------------------------

pytest.importorskip("langgraph", reason="langgraph not installed")

from langgraph.types import Command  # noqa: E402

from tests.test_graph_end_to_end import _run_to_gate, graph  # noqa: E402,F401


def _payload(env, config):
    return env.g.get_state(config).tasks[0].interrupts[0].value


def test_mitigation_reweights_retrains_and_repauses(graph, toy_df):
    from audit_log import get_audit_trail

    config = _run_to_gate(graph, toy_df, "t-mitigate")
    first = _payload(graph, config)
    expected = choose_attribute({"fairness_report": first["fairness_report"]}, [])
    assert expected is not None, "toy_df should produce a fairness violation to mitigate"

    graph.g.invoke(Command(resume={"decision": "reject_and_mitigate", "human_feedback": ""}),
                   config=config)

    snapshot = graph.g.get_state(config)
    assert "human_approval_node" in snapshot.next
    values = snapshot.values
    assert values["rejection_reroute_count"] == 1
    assert values["rejected_models"] == [], "mitigation retrains; it does not exclude a model"
    assert values["mitigation"]["attributes"] == [expected]

    second = _payload(graph, config)
    summary = second["mitigation"]
    assert summary["before"]["model"] == first["selected_model_name"]
    assert summary["before"]["auc_roc"] == first["selected_model_metrics"]["auc_roc"]
    assert summary["after"]["model"] == second["selected_model_name"]
    assert summary["applications"][0]["status"] == "applied"

    trail = get_audit_trail("t-mitigate", db_path=graph.audit_db)
    (event,) = [e for e in trail if e["event_type"] == "mitigation_applied"]
    details = event["details"]
    # Aggregates only (invariant 8): cells cover every train row, no per-row keys.
    assert sum(c["n"] for c in details["cells"]) == len(values["split_index"]["train"])
    assert not {"train", "test", "sample_weight", "weights"} & set(details)
    retrain = [e for e in trail if e["event_type"] == "training_run"][-1]
    assert any("bias-mitigation sample weights" in a
               for a in retrain["details"].get("actions_taken", []))


def test_mitigation_counts_against_the_rejection_cap(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-mitigate-cap")
    for _ in range(3):
        if "human_approval_node" not in graph.g.get_state(config).next:
            break
        graph.g.invoke(Command(resume={"decision": "reject_and_mitigate", "human_feedback": ""}),
                       config=config)

    values = graph.g.get_state(config).values
    assert values["unresolved_human_rejection"] is True
    assert values["rejection_reroute_count"] == 2
    assert not values.get("model_saved_path")


def test_an_approved_mitigated_model_says_so_in_its_model_card(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-mitigate-card")
    graph.g.invoke(Command(resume={"decision": "reject_and_mitigate", "human_feedback": ""}),
                   config=config)
    graph.g.invoke(Command(resume={"decision": "approve", "human_feedback": ""}), config=config)

    run_dir = os.path.join(graph.artifacts_dir, "t-mitigate-card")
    with open(os.path.join(run_dir, "model_card.json"), encoding="utf-8") as fh:
        card = json.load(fh)
    record = card["training_data"]["bias_mitigation"]
    assert record["method"] == "reweighing"
    assert record["attributes"]

    with open(os.path.join(run_dir, "model_card.md"), encoding="utf-8") as fh:
        markdown = fh.read()
    assert "| Bias mitigation | reweighing on " in markdown
    assert any("trained with reweighing" in lim for lim in card["limitations"])
