"""
Tests that the Fairness Agent never reports a pass for something it did not
measure.

A governance dashboard that shows a green badge for an unevaluated property is
worse than one that shows nothing: it manufactures false assurance. These tests
pin that behaviour down.
"""

from __future__ import annotations

import pytest

from data_agent import run_data_agent
from fairness_agent import run_fairness_agent
from training_agent import run_training_agent


def test_regression_reports_not_evaluated_rather_than_passed(toy_regression_df, fake_plan):
    """
    Disparate Impact is undefined for a continuous target.

    The old behaviour returned overall_fairness_passed=True here, which rendered
    as a green "fairness passed" badge on a check that never ran.
    """
    res = run_fairness_agent(
        cleaned_df=toy_regression_df,
        fitted_model=None,
        target_column="earnings",
        sensitive_attribute_candidates=["sex", "race"],
        task_type="regression",
    )

    assert res["overall_fairness_passed"] is None
    assert res["overall_fairness_passed"] is not True
    assert res["fairness_evaluated"] is False
    assert "NOT EVALUATED" in " | ".join(res["actions_taken"])


def test_no_resolvable_attribute_reports_not_evaluated(toy_df, fake_plan):
    """If every candidate is skipped, nothing was measured — so not a pass."""
    cleaned = run_data_agent(toy_df, fake_plan, "income", "classification")
    train = run_training_agent(
        cleaned_df=cleaned["cleaned_df"],
        target_column="income",
        task_type="classification",
        recommended_models=["RandomForest"],
        train_index=cleaned["train_index"],
        test_index=cleaned["test_index"],
    )

    res = run_fairness_agent(
        cleaned_df=cleaned["cleaned_df"],
        fitted_model=train["_fitted_model"],
        target_column="income",
        sensitive_attribute_candidates=["nonexistent_column"],
        task_type="classification",
        eval_index=cleaned["test_index"],
    )

    assert res["fairness_report"] == []
    assert res["overall_fairness_passed"] is None
    assert res["fairness_evaluated"] is False
    assert res["attributes_skipped"]


def test_a_real_evaluation_still_returns_a_boolean(toy_df, fake_plan):
    """The None sentinel must not leak into runs that genuinely were evaluated."""
    cleaned = run_data_agent(toy_df, fake_plan, "income", "classification")
    train = run_training_agent(
        cleaned_df=cleaned["cleaned_df"],
        target_column="income",
        task_type="classification",
        recommended_models=["RandomForest"],
        train_index=cleaned["train_index"],
        test_index=cleaned["test_index"],
    )
    res = run_fairness_agent(
        cleaned_df=cleaned["cleaned_df"],
        fitted_model=train["_fitted_model"],
        target_column="income",
        sensitive_attribute_candidates=["sex", "race"],
        task_type="classification",
        eval_index=cleaned["test_index"],
    )

    assert isinstance(res["overall_fairness_passed"], bool)
    assert res["fairness_evaluated"] is True
    for entry in res["fairness_report"]:
        assert 0.0 <= entry["disparate_impact"] <= 1.0 or entry["disparate_impact"] >= 0
        assert "violation" in entry
