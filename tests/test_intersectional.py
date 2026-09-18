"""
Tests for intersectional subgroups (Step 3b item 4).

A model can treat every group of each protected attribute alike and still treat
some combinations very differently. Pairs of protected attributes are therefore
audited together. Their results are REPORTED ONLY: they never change the verdict,
the same rule as the equal-opportunity gap. Whether they should count is a policy
decision (Step 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairness_agent import MIN_GROUP_SIZE, run_fairness_agent


class StubModel:
    def __init__(self, predictions: pd.Series):
        self.predictions = predictions

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predictions.loc[X.index].to_numpy()


def _cells(spec: dict):
    """{(sex, race): (n, positives)} -> (cleaned, raw, predictions)."""
    sex, race, preds = [], [], []
    for (s, r), (n, positives) in spec.items():
        sex += [s] * n
        race += [r] * n
        preds += [1] * positives + [0] * (n - positives)
    index = pd.RangeIndex(len(sex))
    y = pd.Series(np.resize([0, 1], len(sex)), index=index)
    cleaned = pd.DataFrame({"x": np.zeros(len(sex)), "y": y}, index=index)
    raw = pd.DataFrame({"sex": sex, "race": race, "y": y}, index=index)
    return cleaned, raw, pd.Series(preds, index=index)


def _audit(cleaned, raw, preds, candidates=("sex", "race"), **kwargs):
    return run_fairness_agent(
        cleaned_df=cleaned, fitted_model=StubModel(preds), target_column="y",
        sensitive_attribute_candidates=list(candidates), task_type="classification",
        raw_frame=raw, **kwargs,
    )


# Each attribute alone is perfectly balanced (every group 50% positive), but the
# combinations are not: 80% for (M, X) and (F, Y), 20% for (M, Y) and (F, X).
CROSSED = {("M", "X"): (60, 48), ("M", "Y"): (60, 12),
           ("F", "X"): (60, 12), ("F", "Y"): (60, 48)}


def test_a_disparity_hidden_inside_fair_marginals_is_reported():
    result = _audit(*_cells(CROSSED))

    assert _single(result, "sex")["disparate_impact"] == pytest.approx(1.0)
    assert _single(result, "race")["disparate_impact"] == pytest.approx(1.0)

    (entry,) = result["intersectional_report"]
    assert entry["attributes"] == ["sex", "race"]
    assert entry["attribute"] == "sex × race"
    assert entry["disparate_impact"] == pytest.approx(0.25)
    assert entry["violation"] is True
    assert entry["counts_toward_verdict"] is False
    assert set(entry["groups"]) == {"M × X", "M × Y", "F × X", "F × Y"}


def test_intersectional_results_never_change_the_verdict():
    result = _audit(*_cells(CROSSED))
    assert result["overall_fairness_passed"] is True
    assert all(r["attribute"] != "sex × race" for r in result["fairness_report"])
    assert any("reported only" in a for a in result["actions_taken"])


def _single(result, attribute):
    return next(r for r in result["fairness_report"] if r["attribute"] == attribute)


def test_small_combinations_are_excluded_and_too_few_are_skipped():
    # Every single group clears 30 rows (race Y has 40), but two combinations do not.
    spec = {("M", "X"): (100, 50), ("M", "Y"): (20, 10),
            ("F", "X"): (100, 50), ("F", "Y"): (20, 10)}
    result = _audit(*_cells(spec), min_group_size=MIN_GROUP_SIZE)
    (entry,) = result["intersectional_report"]
    assert entry["excluded_groups"] == {"M × Y": 20, "F × Y": 20}

    # Single groups: M 125, F 50, X 125, Y 50. Only (M, X) reaches 30 as a combination.
    lopsided = {("M", "X"): (100, 50), ("M", "Y"): (25, 12),
                ("F", "X"): (25, 12), ("F", "Y"): (25, 12)}
    result = _audit(*_cells(lopsided), min_group_size=5)
    result_strict = _audit(*_cells(lopsided), min_group_size=MIN_GROUP_SIZE)
    assert result["intersectional_report"]
    assert result_strict["intersectional_report"] == []
    assert any("sex × race" in s for s in result_strict["intersections_skipped"])


def test_unprotected_attributes_are_not_intersected():
    cleaned, raw, preds = _cells(CROSSED)
    raw["job"] = np.resize(["a", "b"], len(raw))
    result = _audit(cleaned, raw, preds, candidates=("sex", "job"))
    # race is protected and auto-audited; job is advisory and never combined.
    assert [e["attribute"] for e in result["intersectional_report"]] == ["sex × race"]


def test_one_protected_attribute_has_no_intersections():
    cleaned, raw, preds = _cells(CROSSED)
    raw = raw.drop(columns=["race"])
    result = _audit(cleaned, raw, preds, candidates=("sex",))
    assert result["intersectional_report"] == []


# ---------------------------------------------------------------------------
# Through the graph
# ---------------------------------------------------------------------------

pytest.importorskip("langgraph", reason="langgraph not installed")

from tests.test_graph_end_to_end import _run_to_gate, graph  # noqa: E402,F401


def test_the_gate_payload_carries_intersectional_results(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-intersect")
    payload = graph.g.get_state(config).tasks[0].interrupts[0].value
    assert "intersectional_report" in payload
    for entry in payload["intersectional_report"]:
        assert entry["counts_toward_verdict"] is False
        assert all(n >= MIN_GROUP_SIZE for n in (g["n"] for g in entry["groups"].values()))
