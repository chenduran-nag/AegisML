"""
Shared fixtures for the AegisML test suite.

Everything here is synthetic and offline. The original test_*.py scripts in the
repository root download the UCI Adult dataset and call the live Groq API, which
means they cannot run in CI, on a plane, or without an API key. These fixtures
reproduce the same *shapes* — nulls, class imbalance, high-cardinality
categoricals, a sensitive attribute — deterministically and in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# 1,000 rows gives a 200-row test split: large enough that the main groups clear the
# Fairness Agent's minimum group size (30), small enough that a minority group does not.
N_ROWS = 1000


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """
    A small Adult-Income-shaped classification dataset.

    Deliberately includes:
      - `age`, `hours`   : numeric features (scaled by the Data Agent)
      - `sex`, `race`    : low-cardinality categoricals -> one-hot encoded,
                           and usable as fairness subgroups
      - `occupation`     : 12 categories -> frequency encoded
      - `notes`          : ~25% null -> imputed, below the 35% "unhandled" bar
      - `income`         : imbalanced binary target (~3:1)
    """
    rng = np.random.default_rng(20260909)

    sex = rng.choice(["Male", "Female"], size=N_ROWS, p=[0.62, 0.38])
    race = rng.choice(["White", "Black", "Asian"], size=N_ROWS, p=[0.7, 0.2, 0.1])
    age = rng.integers(18, 75, size=N_ROWS).astype(float)
    hours = rng.normal(40, 9, size=N_ROWS)
    occupation = rng.choice([f"occ_{i}" for i in range(12)], size=N_ROWS)

    # Target correlates with age/hours AND with sex, so the fairness agent has a
    # genuine disparity to find rather than noise.
    logit = 0.05 * (age - 45) + 0.04 * (hours - 40) + 0.9 * (sex == "Male")
    prob = 1 / (1 + np.exp(-logit))
    income = np.where(rng.random(N_ROWS) < prob * 0.55, ">50K", "<=50K")

    df = pd.DataFrame({
        "age": age,
        "hours": hours,
        "sex": sex,
        "race": race,
        "occupation": occupation,
        "notes": rng.choice(["a", "b", "c"], size=N_ROWS),
        "income": income,
    })

    # Scattered nulls. `notes` sits at 25% — high enough to exercise imputation,
    # below the Data Agent's 35% "unhandled high null" bar so the quality gate
    # passes and the run reaches the governance gate.
    df.loc[rng.choice(N_ROWS, N_ROWS // 20, replace=False), "hours"] = np.nan
    df.loc[rng.choice(N_ROWS, N_ROWS // 4, replace=False), "notes"] = np.nan
    return df


@pytest.fixture
def toy_regression_df(toy_df: pd.DataFrame) -> pd.DataFrame:
    """Same frame with a continuous target, for the regression paths."""
    df = toy_df.copy()
    df["earnings"] = df["age"] * 900 + df["hours"].fillna(40) * 250
    return df.drop(columns=["income"])


@pytest.fixture
def fake_plan() -> dict:
    """
    A plan of the shape plan_pipeline() returns, without calling Groq.

    The Data Agent only ever keyword-matches these strings, so a static plan
    exercises exactly the same code path as a live LLM response.
    """
    return {
        "data_quality_concerns": ["High null rate in 'notes': 40.0% missing"],
        "recommended_preprocessing_steps": [
            "Impute hours using median imputation",
            "Frequency encode occupation",
        ],
        "recommended_models": ["LogisticRegression", "RandomForest"],
        "sensitive_attribute_candidates": ["sex", "race"],
        "reasoning": "Static plan used by the offline test suite.",
    }


@pytest.fixture
def audit_db(tmp_path) -> str:
    """An isolated audit database path, torn down with the tmp_path fixture."""
    return str(tmp_path / "audit_test.db")
