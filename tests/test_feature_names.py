"""
One-hot encoding copies category values into column names, and XGBoost rejects
feature names containing "[", "]" or "<".

Found by the governance evaluation: German Credit has category values such as
"<0" and "0<=X<200", so XGBoost failed on every fit there and dropped silently out
of the leaderboard — and once a reviewer had rejected the models that did train,
nothing trainable was left.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_agent import run_data_agent
from fairness_agent import run_fairness_agent
from training_agent import run_training_agent


@pytest.fixture
def credit_like_df() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 300
    return pd.DataFrame({
        "checking_status": rng.choice(["<0", "0<=X<200", ">=200", "no checking"], size=n),
        "savings_status": rng.choice(["<100", "100<=X<500", "[legacy]", "none"], size=n),
        "duration": rng.integers(6, 60, size=n),
        "credit_amount": rng.integers(500, 15000, size=n).astype(float),
        "sex": rng.choice(["male", "female"], size=n),
        "class": rng.choice(["good", "bad"], size=n, p=[0.7, 0.3]),
    })


def _clean(df):
    return run_data_agent(df, {"recommended_preprocessing_steps": []}, "class", "classification")


def test_one_hot_names_are_made_xgboost_safe(credit_like_df):
    res = _clean(credit_like_df)
    cols = list(res["cleaned_df"].columns)

    assert not [c for c in cols if any(ch in c for ch in "[]<")]
    assert "checking_status_lt0" in cols
    assert "checking_status_0leXlt200" in cols
    assert "savings_status_(legacy)" in cols
    assert any("Renamed" in a and "XGBoost" in a for a in res["actions_taken"])


def test_safe_values_and_original_columns_are_left_alone(credit_like_df):
    cols = list(_clean(credit_like_df)["cleaned_df"].columns)
    assert "checking_status_>=200" in cols, "'>' is allowed and must not be rewritten"
    assert "duration" in cols and "credit_amount" in cols


def test_renaming_never_merges_two_categories(credit_like_df):
    df = credit_like_df.copy()
    df["checking_status"] = np.random.default_rng(1).choice(["<0", "lt0", "other"], size=len(df))
    cols = list(_clean(df)["cleaned_df"].columns)

    assert len(cols) == len(set(cols))
    assert "checking_status_lt0" in cols and "checking_status_lt0_2" in cols


def test_xgboost_trains_on_credit_style_categories(credit_like_df):
    cleaned = _clean(credit_like_df)
    result = run_training_agent(
        cleaned_df=cleaned["cleaned_df"], target_column="class",
        task_type="classification", recommended_models=["XGBoost"],
        train_index=cleaned["train_index"], test_index=cleaned["test_index"],
    )
    entry = result["leaderboard"][0]
    assert entry["trained_successfully"], entry.get("skip_reason")
    assert result["selected_model_name"] == "XGBoost"


def test_fairness_still_reconstructs_renamed_groups(credit_like_df):
    """The attribute prefix survives the rename, so its groups are still found."""
    cleaned = _clean(credit_like_df)
    trained = run_training_agent(
        cleaned_df=cleaned["cleaned_df"], target_column="class",
        task_type="classification", recommended_models=["XGBoost"],
        train_index=cleaned["train_index"], test_index=cleaned["test_index"],
    )
    fairness = run_fairness_agent(
        cleaned_df=cleaned["cleaned_df"], fitted_model=trained["_fitted_model"],
        target_column="class", sensitive_attribute_candidates=["checking_status"],
        task_type="classification", eval_index=cleaned["test_index"],
        # A 60-row test split: lower the group minimum so the prefix rule is tested.
        min_group_size=5,
    )
    assert [r["attribute"] for r in fairness["fairness_report"]] == ["checking_status"]
