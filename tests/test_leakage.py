"""
Tests for the train/test leakage boundary.

The Data Agent draws the split before fitting anything, and every learned
parameter — imputation fill values, frequency maps, scaler mean/std — comes from
the train rows only. These tests assert that property directly rather than
trusting the comment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_agent import TEST_SIZE, run_data_agent
from fairness_agent import run_fairness_agent
from training_agent import run_training_agent


@pytest.fixture
def cleaned(toy_df, fake_plan):
    return run_data_agent(
        df=toy_df, plan=fake_plan, target_column="income", task_type="classification"
    )


# ---------------------------------------------------------------------------
# The split itself
# ---------------------------------------------------------------------------


def test_data_agent_returns_a_split(cleaned):
    assert {"train_index", "validation_index", "test_index"} <= set(cleaned)
    train, validation, test = (cleaned["train_index"], cleaned["validation_index"],
                               cleaned["test_index"])

    assert len(train) > 0 and len(validation) > 0 and len(test) > 0
    assert set(train).isdisjoint(test), "train and test rows must not overlap"
    assert set(train).isdisjoint(validation) and set(validation).isdisjoint(test)

    total = len(cleaned["cleaned_df"])
    assert len(train) + len(validation) + len(test) == total
    assert abs(len(test) / total - TEST_SIZE) < 0.02


def test_split_sizes_are_recorded_in_the_quality_report(cleaned):
    report = cleaned["quality_report"]
    assert report["train_rows"] == len(cleaned["train_index"])
    assert report["test_rows"] == len(cleaned["test_index"])


def test_split_indices_are_json_safe(cleaned):
    """They travel through LangGraph state, so they must be plain Python ints."""
    import json
    json.dumps({"train": cleaned["train_index"], "test": cleaned["test_index"]})


# ---------------------------------------------------------------------------
# The actual leakage property
# ---------------------------------------------------------------------------


def test_scaler_is_fitted_on_train_rows_only(cleaned):
    """
    This is the load-bearing assertion.

    StandardScaler subtracts the mean and divides by the std of whatever it was
    fitted on (ddof=0). So if it was fitted on the train rows, those rows — and
    only those — have mean 0 and std 1 afterwards. If it had been fitted on the
    full frame (the old behaviour), the FULL frame would show mean 0 instead.
    """
    df = cleaned["cleaned_df"]
    train_idx = pd.Index(cleaned["train_index"])

    for col in ("age", "hours"):
        train_vals = df.loc[train_idx, col]
        assert abs(train_vals.mean()) < 1e-9, f"{col}: train mean should be 0"
        assert abs(train_vals.std(ddof=0) - 1.0) < 1e-9, f"{col}: train std should be 1"

    # And the full frame should NOT be centred — if it were, the scaler had seen
    # the held-out rows at fit time.
    full_means = [abs(df[c].mean()) for c in ("age", "hours")]
    assert max(full_means) > 1e-6, (
        "full-frame mean is 0, which means the scaler was fitted on all rows"
    )


def test_imputation_uses_the_train_median(toy_df, fake_plan):
    """`hours` has nulls; the fill value must be the train median, not the global one."""
    res = run_data_agent(
        df=toy_df, plan=fake_plan, target_column="income", task_type="classification"
    )
    actions = " | ".join(res["actions_taken"])
    assert "train-split median" in actions
    assert "hours" in actions


def test_no_nulls_survive_cleaning(cleaned):
    assert cleaned["cleaned_df"].isnull().sum().sum() == 0


def test_frequency_encoding_is_fitted_on_train(cleaned):
    actions = " | ".join(cleaned["actions_taken"])
    assert "train-split frequencies" in actions
    # occupation has 12 categories (>= OHE limit of 10) so it is frequency encoded
    assert cleaned["cleaned_df"]["occupation"].dtype == float


def test_split_is_deterministic(toy_df, fake_plan):
    a = run_data_agent(toy_df, fake_plan, "income", "classification")
    b = run_data_agent(toy_df, fake_plan, "income", "classification")
    assert a["train_index"] == b["train_index"]


# ---------------------------------------------------------------------------
# Training agent honours the split it is handed
# ---------------------------------------------------------------------------


def test_training_agent_reuses_the_data_agent_split(cleaned):
    res = run_training_agent(
        cleaned_df=cleaned["cleaned_df"],
        target_column="income",
        task_type="classification",
        recommended_models=["LogisticRegression", "RandomForest"],
        train_index=cleaned["train_index"],
        test_index=cleaned["test_index"],
    )
    actions = " | ".join(res["actions_taken"])
    assert "reused the Data Agent's split" in actions
    assert f"{len(cleaned['train_index']):,} train" in actions
    assert res["selected_model_name"] in ("LogisticRegression", "RandomForest")


def test_training_agent_warns_when_no_split_supplied(cleaned):
    """Standalone use still works, but must say the metrics are optimistic."""
    res = run_training_agent(
        cleaned_df=cleaned["cleaned_df"],
        target_column="income",
        task_type="classification",
        recommended_models=["LogisticRegression"],
    )
    actions = " | ".join(res["actions_taken"])
    assert "WARNING: no split supplied" in actions
    assert "optimistic" in actions


def test_training_agent_rejects_an_out_of_sync_split(cleaned):
    with pytest.raises(ValueError, match="out of sync"):
        run_training_agent(
            cleaned_df=cleaned["cleaned_df"],
            target_column="income",
            task_type="classification",
            recommended_models=["LogisticRegression"],
            train_index=cleaned["train_index"],
            test_index=[999_999, 1_000_000],
        )


# ---------------------------------------------------------------------------
# Fairness evaluates on held-out rows
# ---------------------------------------------------------------------------


def test_fairness_evaluates_on_the_test_split(cleaned):
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

    assert res["evaluated_rows"] == len(cleaned["test_index"])
    assert res["fairness_evaluated"] is True
    assert "held-out test split" in " | ".join(res["actions_taken"])
    # `sex` was one-hot encoded by the Data Agent and must be reconstructed.
    assert any(r["attribute"] == "sex" for r in res["fairness_report"])


def test_fairness_warns_when_evaluating_on_training_rows(cleaned):
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
        sensitive_attribute_candidates=["sex"],
        task_type="classification",
    )
    assert "WARNING: no held-out split" in " | ".join(res["actions_taken"])
