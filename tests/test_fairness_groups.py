"""
Tests for how the Fairness Agent forms and compares groups.

Each rule here fixes a defect the governance evaluation exposed on real benchmarks:
a 4-person group setting Adult's disparate impact to 0.0; age never audited; COMPAS
sex and Adult occupation skipped because scaling and encoding had destroyed their
groups; and protected attributes audited only if the planner remembered to name them.

The model is a stub that returns fixed predictions, so every expected number can be
worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairness_agent import MIN_GROUP_SIZE, MISSING_GROUP, run_fairness_agent


class StubModel:
    """Returns predetermined predictions for whichever rows it is asked about."""

    def __init__(self, predictions: pd.Series):
        self.predictions = predictions

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predictions.loc[X.index].to_numpy()


def _frames(group_spec: dict, attribute: str = "sex"):
    """
    Build (cleaned, raw, predictions) from {group: (n, positives_predicted)}.

    The cleaned frame carries only a target and a filler feature, so group membership
    can come from nowhere but the raw frame.
    """
    labels, preds = [], []
    for name, (n, positives) in group_spec.items():
        labels += [name] * n
        preds += [1] * positives + [0] * (n - positives)
    index = pd.RangeIndex(len(labels))
    y = pd.Series(np.resize([0, 1], len(labels)), index=index)
    cleaned = pd.DataFrame({"x": np.arange(len(labels), dtype=float), "y": y}, index=index)
    raw = pd.DataFrame({attribute: labels, "y": y}, index=index)
    return cleaned, raw, pd.Series(preds, index=index)


def _audit(cleaned, raw, predictions, candidates, **kwargs):
    return run_fairness_agent(
        cleaned_df=cleaned, fitted_model=StubModel(predictions), target_column="y",
        sensitive_attribute_candidates=candidates, task_type="classification",
        raw_frame=raw, **kwargs,
    )


def _entry(result, attribute):
    return next(r for r in result["fairness_report"] if r["attribute"] == attribute)


# ---------------------------------------------------------------------------
# Minimum group size
# ---------------------------------------------------------------------------


def test_a_tiny_group_no_longer_sets_disparate_impact_to_zero():
    """The Adult case: a 4-row group with no positive predictions."""
    cleaned, raw, preds = _frames({"A": (100, 50), "B": (100, 45), "C": (4, 0)},
                                  attribute="marital")
    entry = _entry(_audit(cleaned, raw, preds, ["marital"]), "marital")

    assert entry["disparate_impact"] == pytest.approx(0.9)
    assert entry["violation"] is False
    assert entry["excluded_groups"] == {"C": 4}
    assert set(entry["groups"]) == {"A", "B"}


def test_the_same_data_without_a_minimum_reproduces_the_old_artifact():
    cleaned, raw, preds = _frames({"A": (100, 50), "B": (100, 45), "C": (4, 0)},
                                  attribute="marital")
    entry = _entry(_audit(cleaned, raw, preds, ["marital"], min_group_size=1), "marital")
    assert entry["disparate_impact"] == 0.0
    assert entry["violation"] is True


def test_fewer_than_two_comparable_groups_is_skipped_with_sizes():
    cleaned, raw, preds = _frames({"A": (100, 50), "B": (10, 5), "C": (5, 1)},
                                  attribute="marital")
    result = _audit(cleaned, raw, preds, ["marital"])

    assert result["fairness_report"] == []
    assert result["overall_fairness_passed"] is None
    (reason,) = result["attributes_skipped"]
    assert f"at least {MIN_GROUP_SIZE}" in reason and "A=100" in reason


def test_group_sizes_are_reported_for_the_groups_compared():
    cleaned, raw, preds = _frames({"Male": (120, 60), "Female": (80, 20)})
    gd = _entry(_audit(cleaned, raw, preds, ["sex"]), "sex")["group_details"]
    assert (gd["group_a"], gd["group_a_n"]) == ("Male", 120)
    assert (gd["group_b"], gd["group_b_n"]) == ("Female", 80)


# ---------------------------------------------------------------------------
# Raw values as the source of group membership
# ---------------------------------------------------------------------------


def test_a_binary_attribute_scaled_in_the_cleaned_data_is_audited_from_raw_values():
    """The COMPAS case: 0/1 sex becomes a standardised float in the cleaned frame."""
    cleaned, raw, preds = _frames({0: (100, 60), 1: (100, 30)})
    cleaned["sex"] = (raw["sex"] - raw["sex"].mean()) / raw["sex"].std()

    without_raw = run_fairness_agent(
        cleaned_df=cleaned, fitted_model=StubModel(preds), target_column="y",
        sensitive_attribute_candidates=["sex"], task_type="classification")
    assert without_raw["fairness_report"] == []
    assert "numeric in the cleaned data" in without_raw["attributes_skipped"][0]

    entry = _entry(_audit(cleaned, raw, preds, ["sex"]), "sex")
    assert entry["grouping"] == "raw values"
    assert set(entry["groups"]) == {"0", "1"}


def test_a_frequency_encoded_categorical_is_audited_from_raw_values():
    """The Adult occupation case: 12 categories, frequency-encoded to a float."""
    spec = {f"occ_{i}": (40, 10 + i) for i in range(12)}
    cleaned, raw, preds = _frames(spec, attribute="occupation")
    cleaned["occupation"] = raw["occupation"].map(
        raw["occupation"].value_counts(normalize=True)).astype(float)

    entry = _entry(_audit(cleaned, raw, preds, ["occupation"]), "occupation")
    assert len(entry["groups"]) == 12


def test_missing_raw_values_form_their_own_group():
    cleaned, raw, preds = _frames({"A": (60, 30), "B": (60, 20), "missing": (40, 5)},
                                  attribute="workclass")
    raw.loc[raw["workclass"] == "missing", "workclass"] = np.nan
    entry = _entry(_audit(cleaned, raw, preds, ["workclass"]), "workclass")

    assert MISSING_GROUP in entry["groups"]
    assert entry["groups"][MISSING_GROUP]["n"] == 40


def test_too_many_categories_are_skipped_with_that_reason():
    spec = {f"country_{i}": (35, 10) for i in range(25)}
    cleaned, raw, preds = _frames(spec, attribute="region")
    result = _audit(cleaned, raw, preds, ["region"])
    assert "too many categories" in result["attributes_skipped"][0]


def test_falls_back_to_one_hot_reconstruction_without_raw_values():
    cleaned, raw, preds = _frames({"Male": (100, 60), "Female": (100, 30)})
    cleaned["sex_Male"] = raw["sex"] == "Male"
    cleaned["sex_Female"] = raw["sex"] == "Female"
    result = run_fairness_agent(
        cleaned_df=cleaned, fitted_model=StubModel(preds), target_column="y",
        sensitive_attribute_candidates=["sex"], task_type="classification")
    assert _entry(result, "sex")["grouping"] == "reconstructed from one-hot columns"


# ---------------------------------------------------------------------------
# Age banding
# ---------------------------------------------------------------------------


def test_continuous_age_is_audited_in_bands():
    rng = np.random.default_rng(0)
    n = 600
    ages = pd.Series(rng.integers(18, 80, size=n))
    index = pd.RangeIndex(n)
    cleaned = pd.DataFrame({"x": np.zeros(n), "y": np.resize([0, 1], n)}, index=index)
    raw = pd.DataFrame({"age": ages, "y": cleaned["y"]}, index=index)
    preds = pd.Series((ages >= 40).astype(int), index=index)

    entry = _entry(_audit(cleaned, raw, preds, ["age"]), "age")

    assert entry["grouping"].startswith("raw values, banded")
    assert set(entry["groups"]) == {"<25", "25-59", "60+"}
    assert entry["groups"]["<25"]["positive_rate"] == 0.0


def test_a_categorical_age_column_is_used_as_is_not_banded():
    cleaned, raw, preds = _frames({"Less than 25": (60, 30), "25-45": (60, 20)},
                                  attribute="age_cat")
    assert _entry(_audit(cleaned, raw, preds, ["age_cat"]), "age_cat")["grouping"] == "raw values"


def test_other_continuous_attributes_are_skipped_with_an_accurate_reason():
    n = 300
    index = pd.RangeIndex(n)
    cleaned = pd.DataFrame({"x": np.zeros(n), "y": np.resize([0, 1], n)}, index=index)
    raw = pd.DataFrame({"hours_per_week": np.linspace(1, 99, n), "y": cleaned["y"]}, index=index)
    result = _audit(cleaned, raw, pd.Series(np.zeros(n, dtype=int), index=index),
                    ["hours_per_week"])
    assert "only age has a banding rule" in result["attributes_skipped"][0]


# ---------------------------------------------------------------------------
# Protected attributes are audited regardless of the planner
# ---------------------------------------------------------------------------


def test_a_protected_attribute_the_planner_omitted_is_audited_anyway():
    cleaned, raw, preds = _frames({"Male": (100, 60), "Female": (100, 30)})
    result = _audit(cleaned, raw, preds, candidates=[])

    entry = _entry(result, "sex")
    assert entry["source"] == "auto"
    assert entry["protected"] is True
    assert any("not proposed by the planner" in a for a in result["actions_taken"])


def test_planner_proposed_attributes_are_not_duplicated_and_keep_their_source():
    cleaned, raw, preds = _frames({"Male": (100, 60), "Female": (100, 30)})
    raw["job"] = np.resize(["a", "b"], len(raw))
    result = _audit(cleaned, raw, preds, candidates=["sex", "job"])

    assert [r["attribute"] for r in result["fairness_report"]] == ["sex", "job"]
    assert _entry(result, "sex")["source"] == "planner"
    assert _entry(result, "job")["protected"] is False


# ---------------------------------------------------------------------------
# Error-rate metrics: reported, never part of the verdict
# ---------------------------------------------------------------------------


def _error_rate_frames():
    """
    Equal positive-prediction rates (0.5) in both groups, very different error rates:
      group A: TPR 0.9, FPR 0.1      group B: TPR 0.5, FPR 0.5
    """
    rows = []
    for group, tp, fp in (("A", 90, 10), ("B", 50, 50)):
        rows += [(group, 1, 1)] * tp + [(group, 1, 0)] * (100 - tp)
        rows += [(group, 0, 1)] * fp + [(group, 0, 0)] * (100 - fp)
    frame = pd.DataFrame(rows, columns=["sex", "y", "pred"])
    cleaned = pd.DataFrame({"x": np.zeros(len(frame)), "y": frame["y"]})
    raw = frame[["sex", "y"]]
    return cleaned, raw, frame["pred"]


def test_error_rate_gaps_are_reported():
    cleaned, raw, preds = _error_rate_frames()
    entry = _entry(_audit(cleaned, raw, preds, ["sex"]), "sex")

    assert entry["groups"]["A"]["tpr"] == pytest.approx(0.9)
    assert entry["groups"]["B"]["fpr"] == pytest.approx(0.5)
    assert entry["equal_opportunity_difference"] == pytest.approx(0.4)
    assert entry["equalized_odds_difference"] == pytest.approx(0.4)


def test_error_rate_gaps_do_not_change_the_verdict():
    cleaned, raw, preds = _error_rate_frames()
    result = _audit(cleaned, raw, preds, ["sex"])

    entry = _entry(result, "sex")
    assert entry["disparate_impact"] == pytest.approx(1.0)
    assert entry["violation"] is False
    assert result["overall_fairness_passed"] is True


def test_error_rate_metrics_are_none_for_a_non_binary_target():
    cleaned, raw, preds = _frames({"Male": (90, 45), "Female": (90, 30)})
    cleaned["y"] = np.resize([0, 1, 2], len(cleaned))
    raw["y"] = cleaned["y"]
    entry = _entry(_audit(cleaned, raw, preds, ["sex"]), "sex")

    assert entry["equal_opportunity_difference"] is None
    assert entry["equalized_odds_difference"] is None


def test_raw_frame_must_share_the_cleaned_index():
    cleaned, raw, preds = _frames({"Male": (100, 60), "Female": (100, 30)})
    with pytest.raises(ValueError, match="share index labels"):
        _audit(cleaned, raw.iloc[:50], preds, ["sex"])
