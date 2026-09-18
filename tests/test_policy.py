"""
Tests for policy-as-code (policy.py, policy.yaml) and its enforcement in the graph.

A policy is only useful if (1) a bad file fails loudly instead of silently falling
back, (2) changing a value in YAML changes behaviour without a code edit, and (3)
every run records which policy governed it.
"""

from __future__ import annotations

import os
import textwrap

import numpy as np
import pandas as pd
import pytest

import data_agent
import fairness_agent
from policy import (
    PolicyError,
    default_policy,
    load_policy,
    policy_sha256,
    validate_policy,
    with_overrides,
)


def _write(tmp_path, text):
    path = tmp_path / "policy.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def test_the_committed_policy_file_equals_the_code_defaults():
    import pipeline_graph

    loaded = load_policy()
    assert loaded["policy"] == default_policy()
    assert loaded["version"] == loaded["policy"]["version"]
    assert loaded["sha256"] == policy_sha256(loaded["policy"])
    governance = default_policy()["governance"]
    assert governance["max_retries"] == pipeline_graph.MAX_RETRIES
    assert governance["max_human_reroutes"] == pipeline_graph.MAX_HUMAN_REROUTES
    assert governance["block_approval_when_fairness_not_evaluated"] is True
    assert governance["require_dual_signoff_for_violating_approval"] is True
    assert governance["dual_signoff_can_override_approval_block"] is False


@pytest.mark.parametrize("text,match", [
    ('version: "1"\nfairness:\n  min_grup_size: 5\n', "unknown key"),
    ('version: "1"\nfairness:\n  min_group_size: 0\n', "min_group_size"),
    ('version: "1"\nfairness:\n  min_group_size: 2.5\n', "integer"),
    ('version: "1"\ngovernance:\n  block_approval_when_fairness_not_evaluated: "yes"\n', "bool"),
    ('fairness:\n  min_group_size: 5\n', "version"),
    ('version: "1"\ntraining:\n  allowed_models: [SVM]\n', "unknown model"),
    ('version: "1"\ntraining:\n  allowed_models: []\n', "non-empty"),
    ('version: "1"\ndata: [1, 2]\n', "mapping"),
    ('version: "1"\nnot_a_section: {}\n', "unknown top-level"),
    ('version: "1"\ndata:\n  column_high_null_warning_threshold: 0.9\n', "cannot exceed"),
    ('version: "1"\nfairness: [unclosed\n', "not valid YAML"),
])
def test_an_invalid_policy_fails_loudly(tmp_path, text, match):
    with pytest.raises(PolicyError, match=match):
        load_policy(_write(tmp_path, text))


def test_a_missing_policy_file_fails_loudly(tmp_path):
    with pytest.raises(PolicyError, match="cannot read"):
        load_policy(tmp_path / "nope.yaml")


def test_the_hash_ignores_formatting_but_not_values(tmp_path):
    a = load_policy(_write(tmp_path, 'version: "1"  # a comment\nfairness:\n  min_group_size: 30\n'))
    b = load_policy(_write(tmp_path, 'version: "1"\n\nfairness: {min_group_size: 30}\n'))
    c = load_policy(_write(tmp_path, 'version: "1"\nfairness:\n  min_group_size: 31\n'))
    assert a["sha256"] == b["sha256"]
    assert a["sha256"] != c["sha256"]


def test_model_names_are_normalised_to_registry_spelling():
    policy = validate_policy({"version": "1", "training": {"allowed_models": ["xgboost", "Random Forest"]}})
    assert policy["training"]["allowed_models"] == ["XGBoost", "RandomForest"]


def test_overrides_are_validated_too():
    with pytest.raises(PolicyError):
        with_overrides(default_policy(), {"governance": {"max_retries": -1}})


# ---------------------------------------------------------------------------
# Values change behaviour without code edits
# ---------------------------------------------------------------------------


def test_the_one_hot_limit_comes_from_the_thresholds(toy_df, fake_plan):
    """toy_df's occupation has 12 categories: frequency-encoded by default."""
    default = data_agent.run_data_agent(toy_df, fake_plan, "income", "classification")
    raised = data_agent.run_data_agent(toy_df, fake_plan, "income", "classification",
                                       thresholds={"ohe_cardinality_limit": 20})
    assert "occupation" in default["cleaned_df"].columns
    assert any(c.startswith("occupation_") for c in raised["cleaned_df"].columns)


def test_split_sizes_come_from_the_thresholds(toy_df, fake_plan):
    res = data_agent.run_data_agent(toy_df, fake_plan, "income", "classification",
                                    thresholds={"test_size": 0.30, "validation_size": 0.25})
    total = len(res["cleaned_df"])
    assert len(res["test_index"]) / total == pytest.approx(0.30, abs=0.02)
    assert len(res["validation_index"]) / total == pytest.approx(0.70 * 0.25, abs=0.02)


class _Stub:
    def __init__(self, preds):
        self.preds = preds

    def predict(self, X):
        return self.preds.loc[X.index].to_numpy()


def test_the_disparate_impact_threshold_comes_from_the_arguments():
    index = pd.RangeIndex(200)
    cleaned = pd.DataFrame({"x": np.zeros(200), "y": np.resize([0, 1], 200)}, index=index)
    raw = pd.DataFrame({"sex": ["M"] * 100 + ["F"] * 100, "y": cleaned["y"]}, index=index)
    preds = pd.Series([1] * 50 + [0] * 50 + [1] * 30 + [0] * 70, index=index)  # DI 0.6, DPD 0.2

    def run(**kwargs):
        return fairness_agent.run_fairness_agent(
            cleaned_df=cleaned, fitted_model=_Stub(preds), target_column="y",
            sensitive_attribute_candidates=["sex"], task_type="classification",
            raw_frame=raw, **kwargs)

    assert run()["overall_fairness_passed"] is False
    lenient = run(disparate_impact_threshold=0.5, parity_difference_threshold=0.25)
    assert lenient["overall_fairness_passed"] is True
    assert lenient["thresholds"]["disparate_impact_min"] == 0.5


# ---------------------------------------------------------------------------
# Through the graph
# ---------------------------------------------------------------------------

pytest.importorskip("langgraph", reason="langgraph not installed")

from langgraph.types import Command  # noqa: E402

from tests.test_graph_end_to_end import _initial_state, graph  # noqa: E402,F401
from tests.conftest import approve, decide


def _state(df, policy=None, **extra):
    state = {**_initial_state(df), **extra}
    if policy is not None:
        state.update(policy=policy, policy_version=policy["version"],
                     policy_sha256=policy_sha256(policy))
    return state


def _run(env, state, thread):
    config = {"configurable": {"thread_id": thread}}
    env.g.invoke(state, config=config)
    return config


def _decide(env, config, decision):
    """One decision, one reviewer — enough unless the run asks for a second sign-off."""
    decide(env.g, config, decision)


def test_the_governing_policy_is_recorded_in_the_audit_trail(graph, toy_df):
    from audit_log import get_audit_trail

    policy = with_overrides(default_policy(), {"fairness": {"min_group_size": 25}})
    config = _run(graph, _state(toy_df, policy), "t-policy-audit")
    approve(graph.g, config)

    trail = get_audit_trail("t-policy-audit", db_path=graph.audit_db)
    (applied,) = [e for e in trail if e["event_type"] == "policy_applied"]
    assert applied["details"]["policy_sha256"] == policy_sha256(policy)
    assert trail[0]["event_type"] == "policy_applied", "recorded before anything acts on it"
    final = [e for e in trail if e["event_type"] == "final_outcome"][-1]
    assert final["details"]["policy_sha256"] == policy_sha256(policy)
    fairness = graph.g.get_state(config).values["fairness_result"]
    assert fairness["thresholds"]["min_group_size"] == 25


def test_the_rejection_cap_comes_from_the_policy(graph, toy_df):
    policy = with_overrides(default_policy(), {"governance": {"max_human_reroutes": 1}})
    config = _run(graph, _state(toy_df, policy), "t-policy-cap")
    _decide(graph, config, "reject_model_or_fairness")
    assert "human_approval_node" in graph.g.get_state(config).next
    _decide(graph, config, "reject_model_or_fairness")

    values = graph.g.get_state(config).values
    assert values["unresolved_human_rejection"] is True
    assert values["rejection_reroute_count"] == 1


def test_allowed_models_restrict_training(graph, toy_df):
    policy = with_overrides(default_policy(), {"training": {"allowed_models": ["RandomForest"]}})
    config = _run(graph, _state(toy_df, policy), "t-policy-models")
    values = graph.g.get_state(config).values
    assert [m["model_name"] for m in values["training_result"]["leaderboard"]] == ["RandomForest"]


def _unauditable(toy_df):
    """No protected column at all: the fairness verdict cannot be anything but None."""
    return toy_df.drop(columns=["sex", "race", "age"])


def test_approving_an_unevaluated_model_is_blocked_by_default(graph, toy_df):
    from audit_log import get_audit_trail

    config = _run(graph, _state(_unauditable(toy_df)), "t-policy-block")
    payload = graph.g.get_state(config).tasks[0].interrupts[0].value
    assert payload["approval_blocked_reason"]
    assert "approve" not in payload["allowed_decisions"]

    _decide(graph, config, "approve")
    values = graph.g.get_state(config).values
    assert values["unresolved_approval_blocked"] is True
    assert not values.get("model_saved_path")
    assert not os.path.isdir(os.path.join(graph.artifacts_dir, "t-policy-block"))
    final = [e for e in get_audit_trail("t-policy-block", db_path=graph.audit_db)
             if e["event_type"] == "final_outcome"][-1]
    assert final["details"]["status"] == "APPROVAL_BLOCKED_BY_POLICY"


def test_the_block_can_be_switched_off_in_policy(graph, toy_df):
    policy = with_overrides(default_policy(),
                            {"governance": {"block_approval_when_fairness_not_evaluated": False}})
    config = _run(graph, _state(_unauditable(toy_df), policy), "t-policy-unblocked")
    payload = graph.g.get_state(config).tasks[0].interrupts[0].value
    assert payload["approval_blocked_reason"] is None and "approve" in payload["allowed_decisions"]
    _decide(graph, config, "approve")
    assert graph.g.get_state(config).values["model_saved_path"]


def test_regression_is_exempt_from_the_block(graph, toy_regression_df):
    state = _state(toy_regression_df, target_column="earnings", task_type="regression")
    config = _run(graph, state, "t-policy-regression")
    payload = graph.g.get_state(config).tasks[0].interrupts[0].value
    assert payload["approval_blocked_reason"] is None
    _decide(graph, config, "approve")
    assert graph.g.get_state(config).values["model_saved_path"]
