"""
Tests for eda_insights.py and for how its findings are consumed.

The load-bearing property is routing: structural findings are acted on
deterministically, judgement-light findings go to the planner, and suspected
leakage and proxy variables are held for the human reviewer and must never reach
the LLM.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data_agent import run_data_agent
from eda_insights import (
    build_eda_linkage,
    compact_findings_for_planner,
    derive_eda_findings,
    findings_for_route,
    is_protected_attribute,
    mentions_column,
)
from fairness_agent import run_fairness_agent
from training_agent import run_training_agent


@pytest.fixture
def eda_df() -> pd.DataFrame:
    """
    A frame engineered to trigger every finding type exactly where expected.

      customer_id    per-row identifier                -> identifier_column
      constant_col   one value                          -> constant_column
      capital_gain   90% zeros                          -> zero_inflated
      hours          spike at 40 plus a wide spread     -> heavy_outliers
      education      deterministic re-encoding of
                     education_num                      -> redundant_features
      approved_flag  copy of the target                 -> target_leakage_suspect
      relationship   largely determined by sex          -> proxy_variable (high)
      gender_code    same information as sex, but both
                     are protected                      -> NOT a proxy finding
      measurement    unique floats                      -> NOT an identifier
    """
    rng = np.random.default_rng(7)
    n = 600
    sex = rng.choice(["Male", "Female"], size=n, p=[0.6, 0.4])
    relationship = np.where(
        sex == "Male",
        rng.choice(["Husband", "Own-child", "Unmarried"], size=n, p=[0.8, 0.1, 0.1]),
        rng.choice(["Wife", "Own-child", "Unmarried"], size=n, p=[0.7, 0.15, 0.15]),
    )
    education_num = rng.integers(1, 17, size=n)
    income = np.where(rng.random(n) < 0.3, ">50K", "<=50K")

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "constant_col": "same",
        "age": rng.integers(18, 70, size=n).astype(float),
        "sex": sex,
        "gender_code": np.where(sex == "Male", "M", "F"),
        "race": rng.choice(["White", "Black", "Asian"], size=n, p=[0.7, 0.2, 0.1]),
        "relationship": relationship,
        "hours": np.where(rng.random(n) < 0.45, 40.0, rng.normal(40, 15, size=n)),
        "capital_gain": np.where(rng.random(n) < 0.9, 0.0,
                                 rng.integers(1000, 20000, size=n)).astype(float),
        "education_num": education_num,
        "education": [f"level_{v}" for v in education_num],
        "measurement": rng.normal(size=n),
        "approved_flag": np.where(income == ">50K", "yes", "no"),
        "income": income,
    })


@pytest.fixture
def findings(eda_df):
    return derive_eda_findings(eda_df, "income", "classification")


def _by_type(findings, ftype):
    return [f for f in findings if f["type"] == ftype]


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("age", True), ("age_group", True), ("sex", True), ("gender_code", True),
    ("race", True), ("ethnicity", True), ("native-country", True),
    ("sexual_orientation", True), ("disability_status", True),
    ("wage", False), ("percentage", False), ("language", False), ("agency", False),
    ("relationship", False), ("occupation", False),
])
def test_protected_attributes_match_on_tokens_not_substrings(name, expected):
    assert is_protected_attribute(name) is expected


@pytest.mark.parametrize("text,column,expected", [
    ("Drop column age now", "age", True),
    ("percentage of wage earners", "age", False),
    ("Winsorize hours-per-week at the 99th percentile", "hours-per-week", True),
    ("capital-gain is sparse", "cap", False),
])
def test_mentions_column_respects_word_boundaries(text, column, expected):
    assert mentions_column(text, column) is expected


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def test_every_finding_has_a_valid_route_and_is_json_safe(findings):
    assert findings, "fixture should trigger findings"
    for f in findings:
        assert f["route"] in ("data_agent", "planner", "reviewer")
        assert f["severity"] in ("high", "medium", "info")
    json.dumps(findings)


def test_findings_are_ordered_most_severe_first(findings):
    order = {"high": 0, "medium": 1, "info": 2}
    ranks = [order[f["severity"]] for f in findings]
    assert ranks == sorted(ranks)


def test_identifier_and_constant_columns_route_to_the_data_agent(findings):
    ids = {f["columns"][0]: f for f in _by_type(findings, "identifier_column")}
    assert "customer_id" in ids and ids["customer_id"]["route"] == "data_agent"

    consts = [f["columns"][0] for f in _by_type(findings, "constant_column")]
    assert "constant_col" in consts


def test_unique_floats_are_not_mistaken_for_identifiers(findings):
    ids = [f["columns"][0] for f in _by_type(findings, "identifier_column")]
    assert "measurement" not in ids


def test_zero_inflated_column_is_not_reported_as_outlier_heavy(findings):
    zero = [f["columns"][0] for f in _by_type(findings, "zero_inflated")]
    outliers = [f["columns"][0] for f in _by_type(findings, "heavy_outliers")]
    assert "capital_gain" in zero
    assert "capital_gain" not in outliers, (
        "an IQR of zero makes every non-zero value an 'outlier' — reporting it as "
        "outlier-heavy would invite destructive clipping"
    )


def test_peaked_column_is_reported_as_outlier_heavy(findings):
    outliers = {f["columns"][0]: f for f in _by_type(findings, "heavy_outliers")}
    assert "hours" in outliers
    assert outliers["hours"]["route"] == "planner"


def test_redundant_pair_is_found(findings):
    pairs = [set(f["columns"]) for f in _by_type(findings, "redundant_features")]
    assert {"education", "education_num"} in pairs


def test_target_leakage_is_held_for_the_reviewer(findings):
    leaks = {f["columns"][0]: f for f in _by_type(findings, "target_leakage_suspect")}
    assert "approved_flag" in leaks
    assert leaks["approved_flag"]["route"] == "reviewer"
    assert leaks["approved_flag"]["severity"] == "high"


def test_proxy_variable_is_found_and_held_for_the_reviewer(findings):
    proxies = {f["id"]: f for f in _by_type(findings, "proxy_variable")}
    finding = proxies.get("proxy_variable:relationship->sex")
    assert finding is not None
    assert finding["route"] == "reviewer"
    assert finding["severity"] == "high"
    assert finding["metric"]["name"] == "cramers_v"


def test_two_protected_attributes_are_not_reported_as_proxies_of_each_other(findings):
    for f in _by_type(findings, "proxy_variable"):
        assert not (is_protected_attribute(f["columns"][0])
                    and is_protected_attribute(f["columns"][1])), f["id"]


def test_structural_columns_are_excluded_from_association_findings(findings):
    for f in findings:
        if f["type"] not in ("identifier_column", "constant_column"):
            assert "customer_id" not in f["columns"]
            assert "constant_col" not in f["columns"]


def test_skewed_regression_target_is_flagged():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=500),
                       "price": rng.lognormal(mean=0, sigma=1.5, size=500)})
    found = derive_eda_findings(df, "price", "regression")
    assert _by_type(found, "skewed_target")


def test_empty_frame_yields_no_findings():
    assert derive_eda_findings(pd.DataFrame(), "y", "classification") == []


# ---------------------------------------------------------------------------
# Planner: sees planner-routed findings, never reviewer-routed ones
# ---------------------------------------------------------------------------


def test_compact_planner_view_excludes_reviewer_and_data_agent_findings(findings):
    planner_view = compact_findings_for_planner(findings)
    types = {f["type"] for f in planner_view}

    assert types, "planner should receive the judgement-light findings"
    assert "target_leakage_suspect" not in types
    assert "proxy_variable" not in types
    assert "identifier_column" not in types
    assert "constant_column" not in types


def test_planner_prompt_carries_planner_findings_but_not_proxies(monkeypatch, eda_df, findings):
    """
    The safety property end to end: capture the exact prompts sent to the LLM and
    assert that no proxy or leakage finding is in them.
    """
    import planner_agent

    captured = {}

    def _fake_call(system_prompt, user_prompt, model, client):
        captured["system"], captured["user"] = system_prompt, user_prompt
        return json.dumps({
            "data_quality_concerns": ["capital_gain is zero-inflated"],
            "recommended_preprocessing_steps": [],
            "recommended_models": ["RandomForest"],
            "sensitive_attribute_candidates": ["sex"],
            "reasoning": "stub",
        }), {}

    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")
    monkeypatch.setattr(planner_agent, "_call_groq", _fake_call)

    planner_agent.plan_pipeline(eda_df, "income", "classification", eda_findings=findings)

    assert "EDA FINDINGS" in captured["system"]
    assert "zero_inflated" in captured["user"]
    assert "proxy_variable" not in captured["user"]
    assert "target_leakage_suspect" not in captured["user"]
    assert "relationship' and protected attribute" not in captured["user"]


def test_prompt_is_unchanged_when_there_are_no_planner_findings(monkeypatch, eda_df):
    import planner_agent

    captured = {}

    def _fake_call(system_prompt, user_prompt, model, client):
        captured["system"] = system_prompt
        return json.dumps({
            "data_quality_concerns": [], "recommended_preprocessing_steps": [],
            "recommended_models": ["RandomForest"],
            "sensitive_attribute_candidates": [], "reasoning": "stub",
        }), {}

    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")
    monkeypatch.setattr(planner_agent, "_call_groq", _fake_call)

    planner_agent.plan_pipeline(eda_df, "income", "classification", eda_findings=[])
    assert "EDA FINDINGS" not in captured["system"]


# ---------------------------------------------------------------------------
# Data Agent: acts on data_agent findings and explicit, safe instructions only
# ---------------------------------------------------------------------------


def test_data_agent_drops_structural_findings_and_records_why(eda_df, findings):
    res = run_data_agent(eda_df, {"recommended_preprocessing_steps": []},
                         "income", "classification", eda_findings=findings)

    cols = res["cleaned_df"].columns
    assert "customer_id" not in cols
    assert "constant_col" not in cols

    acted = {a["finding_id"] for a in res["eda_actions"]}
    assert "identifier_column:customer_id" in acted
    assert "constant_column:constant_col" in acted
    assert any("EDA finding identifier_column" in a for a in res["actions_taken"])


def test_data_agent_never_acts_on_reviewer_findings(eda_df, findings):
    """Leakage and proxy columns stay in unless a human directs otherwise."""
    res = run_data_agent(eda_df, {"recommended_preprocessing_steps": []},
                         "income", "classification", eda_findings=findings)
    dropped = res["quality_report"]["columns_dropped"]
    cols = set(res["cleaned_df"].columns)

    # Both are low-cardinality categoricals, so they survive as one-hot columns
    # (approved_flag_yes, relationship_Husband, ...) rather than under their raw names.
    assert "approved_flag" not in dropped
    assert "relationship" not in dropped
    assert any(c.startswith("approved_flag") for c in cols)
    assert any(c.startswith("relationship") for c in cols)


def test_data_agent_without_findings_behaves_as_before(eda_df):
    res = run_data_agent(eda_df, {"recommended_preprocessing_steps": []},
                         "income", "classification")
    assert res["eda_actions"] == []
    assert res["winsorized_columns"] == []


def test_winsorizing_uses_train_split_percentiles(eda_df):
    plan = {"recommended_preprocessing_steps": [
        "Winsorize hours at the 1st and 99th percentiles"]}
    res = run_data_agent(eda_df, plan, "income", "classification")

    assert res["winsorized_columns"] == ["hours"]
    lower, upper = eda_df.loc[pd.Index(res["train_index"]), "hours"].quantile([0.01, 0.99])
    action = next(a for a in res["actions_taken"] if a.startswith("Winsorized 'hours'"))
    assert f"{lower:.4g}" in action and f"{upper:.4g}" in action


def test_winsorizing_a_degenerate_column_is_skipped(eda_df):
    df = eda_df.copy()
    df["all_zero"] = 0.0
    plan = {"recommended_preprocessing_steps": ["Clip outliers in all_zero"]}
    res = run_data_agent(df, plan, "income", "classification")

    assert "all_zero" not in res["winsorized_columns"]
    assert any("Skipped winsorizing 'all_zero'" in a for a in res["actions_taken"])


def test_clip_instruction_that_mentions_dropping_does_not_drop(eda_df):
    plan = {"recommended_preprocessing_steps": [
        "Clip extreme values in hours rather than dropping them"]}
    res = run_data_agent(eda_df, plan, "income", "classification")

    assert "hours" in res["cleaned_df"].columns
    assert "hours" not in res["quality_report"]["columns_dropped"]
    assert "hours" in res["winsorized_columns"]


ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]


@pytest.mark.parametrize("step,expected_drop,expected_winsorize", [
    # Written by the live planner (openai/gpt-oss-20b) on UCI Adult. The old
    # substring matcher dropped BOTH columns here.
    ("Drop 'education-num' and keep 'education' to avoid redundancy",
     ["education-num"], []),
    # The example form the EDA prompt rule itself suggests. Must not drop both.
    ("Drop column education-num (redundant with education)",
     ["education-num"], []),
    # Verbatim from a live pipeline run: quoted names before the parenthesis.
    ("Drop 'education-num' (redundant with 'education')",
     ["education-num"], []),
    # Live planner. The old matcher winsorized hours-per-week despite "keep as is".
    ("Keep 'hours-per-week' as is; consider winsorizing at the 1st/99th "
     "percentiles if extreme values are errors",
     [], []),
    # Live planner, from a human reviewer's directive. Must keep working.
    ("Drop the 'fnlwgt' column as it is a survey sampling weight and must not "
     "be used as a predictor",
     ["fnlwgt"], []),
    # Advice, not an instruction.
    ("Consider dropping fnlwgt", [], []),
    ("Winsorize capital-gain at the 1st and 99th percentiles", [], ["capital-gain"]),
    ("Clip extreme values in hours-per-week rather than dropping them",
     [], ["hours-per-week"]),
    # A drop verb after the terminator must not trigger a drop.
    ("Impute hours-per-week with the median rather than dropping rows", [], []),
    # Independent clauses are read independently.
    ("Drop column fnlwgt; consider winsorizing capital-gain", ["fnlwgt"], []),
])
def test_plan_step_parsing_on_real_planner_phrasings(step, expected_drop, expected_winsorize):
    from data_agent import _parse_plan_steps

    hints = _parse_plan_steps([step], ADULT_COLUMNS)
    assert hints["explicit_drop_columns"] == expected_drop
    assert hints["winsorize_columns"] == expected_winsorize


def test_columns_named_in_masks_longer_names_first():
    from eda_insights import columns_named_in

    assert columns_named_in("drop 'education-num' and", ADULT_COLUMNS) == ["education-num"]
    assert columns_named_in("education and education-num", ADULT_COLUMNS) == [
        "education", "education-num"]
    assert columns_named_in("percentage of wage earners", ADULT_COLUMNS) == []


# ---------------------------------------------------------------------------
# Fairness: proxy warnings are informational
# ---------------------------------------------------------------------------


def test_proxy_warnings_do_not_change_the_fairness_verdict(eda_df, findings):
    cleaned = run_data_agent(eda_df, {"recommended_preprocessing_steps": []},
                             "income", "classification", eda_findings=findings)
    train = run_training_agent(
        cleaned_df=cleaned["cleaned_df"], target_column="income",
        task_type="classification", recommended_models=["RandomForest"],
        train_index=cleaned["train_index"], test_index=cleaned["test_index"],
    )
    common = dict(
        cleaned_df=cleaned["cleaned_df"], fitted_model=train["_fitted_model"],
        target_column="income", sensitive_attribute_candidates=["sex", "race"],
        task_type="classification", eval_index=cleaned["test_index"],
    )

    without = run_fairness_agent(**common)
    with_proxies = run_fairness_agent(
        **common, proxy_findings=findings_for_route(findings, "reviewer"))

    assert with_proxies["overall_fairness_passed"] == without["overall_fairness_passed"]
    assert with_proxies["fairness_report"] == without["fairness_report"]

    warning = next(w for w in with_proxies["proxy_warnings"]
                   if w["id"] == "proxy_variable:relationship->sex")
    assert warning["protected_attribute_audited"] is True
    assert any("PROXY WARNING" in a for a in with_proxies["actions_taken"])


def test_regression_run_still_surfaces_proxy_warnings(toy_regression_df):
    proxy = [{
        "id": "proxy_variable:occupation->sex", "type": "proxy_variable",
        "severity": "medium", "columns": ["occupation", "sex"],
        "metric": {"name": "cramers_v", "value": 0.43}, "route": "reviewer",
    }]
    res = run_fairness_agent(toy_regression_df, None, "earnings", ["race"],
                             "regression", proxy_findings=proxy)

    assert res["overall_fairness_passed"] is None
    assert res["proxy_warnings"][0]["protected_attribute_audited"] is False


# ---------------------------------------------------------------------------
# Linkage
# ---------------------------------------------------------------------------


def test_linkage_reports_how_each_finding_was_used(eda_df, findings):
    plan = {"data_quality_concerns": ["capital_gain is zero-inflated"],
            "recommended_preprocessing_steps": []}
    data_res = run_data_agent(eda_df, plan, "income", "classification",
                              eda_findings=findings)
    fairness = {"proxy_warnings": [{"id": "proxy_variable:relationship->sex"}]}

    linked = {f["id"]: f for f in build_eda_linkage(findings, plan, data_res, fairness)}

    def statuses(fid):
        return {o["status"] for o in linked[fid]["outcomes"]}

    assert "applied" in statuses("identifier_column:customer_id")
    assert {"sent", "mentioned"} <= statuses("zero_inflated:capital_gain")

    redundant = next(fid for fid in linked if fid.startswith("redundant_features:"))
    assert "not_reflected" in statuses(redundant)

    assert statuses("target_leakage_suspect:approved_flag") == {"flagged"}
    assert statuses("proxy_variable:relationship->sex") == {"flagged", "attached"}


def test_linkage_before_downstream_stages_run_is_pending(findings):
    linked = build_eda_linkage(findings)
    structural = next(f for f in linked if f["route"] == "data_agent")
    assert structural["outcomes"][0]["status"] == "pending"
