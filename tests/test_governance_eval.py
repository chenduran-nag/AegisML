"""
Tests for the governance evaluation: the planner record/replay cache, the per-run
split seed, the scripted reviewer, and the harness end to end.

All offline. The planner is either stubbed or served from a temporary cache.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")

import pipeline_graph
import planner_agent
from data_agent import SPLIT_RANDOM_STATE, run_data_agent
from experiments import run_governance_eval as ev

VALID_PLAN = {
    "data_quality_concerns": [],
    "recommended_preprocessing_steps": [],
    "recommended_models": ["LogisticRegression", "RandomForest"],
    "sensitive_attribute_candidates": ["sex"],
    "reasoning": "stub",
}


@pytest.fixture(autouse=True)
def _planner_cache_always_reset():
    """The cache is process-global; never let one test leak it into another."""
    yield
    planner_agent.configure_planner_cache("off")


@pytest.fixture
def fake_groq(monkeypatch):
    """Replace the network call. `responses` can queue specific raw outputs."""
    calls: list[tuple] = []
    responses: list[str] = []

    def _fake(system_prompt, user_prompt, model, client):
        calls.append((system_prompt, user_prompt, model))
        content = responses.pop(0) if responses else json.dumps(VALID_PLAN)
        return content, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    monkeypatch.setattr(planner_agent, "_call_groq", _fake)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")
    return calls, responses


def _plan(df, **kwargs):
    meta: dict = {}
    plan = planner_agent.plan_pipeline(df=df, target_column="income",
                                       task_type="classification", meta_out=meta, **kwargs)
    return plan, meta


# ---------------------------------------------------------------------------
# Planner cache
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic_and_prompt_sensitive():
    key = planner_agent.planner_cache_key("m", "sys", "user")
    assert key == planner_agent.planner_cache_key("m", "sys", "user")
    assert key != planner_agent.planner_cache_key("m", "sys", "user!")
    assert key != planner_agent.planner_cache_key("other-model", "sys", "user")


def test_record_mode_stores_on_miss_then_serves_hits(tmp_path, toy_df, fake_groq):
    calls, _ = fake_groq
    planner_agent.configure_planner_cache("record", str(tmp_path))

    first, meta1 = _plan(toy_df)
    second, meta2 = _plan(toy_df)

    assert meta1["cache"] == "miss" and meta2["cache"] == "hit"
    assert len(calls) == 1, "the second identical prompt must not reach the LLM"
    assert first == second
    entries = list(tmp_path.glob("*.json"))
    assert len(entries) == 1
    stored = json.loads(entries[0].read_text(encoding="utf-8"))
    assert stored["usage"]["total_tokens"] == 150
    assert "raw rows" not in stored["user_prompt"]


def test_replay_needs_no_api_key_and_never_calls_the_llm(tmp_path, toy_df, fake_groq, monkeypatch):
    planner_agent.configure_planner_cache("record", str(tmp_path))
    recorded, _ = _plan(toy_df)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("replay mode called the LLM")

    monkeypatch.setattr(planner_agent, "_call_groq", _must_not_be_called)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Stop the loader from filling the key back in from a developer's real .env.
    monkeypatch.setattr(planner_agent, "_ensure_groq_api_key", lambda: None)
    planner_agent.configure_planner_cache("replay", str(tmp_path))

    replayed, meta = _plan(toy_df)
    assert replayed == recorded
    assert meta["cache"] == "hit"


def test_replay_miss_raises_immediately_and_is_not_retried(tmp_path, toy_df, monkeypatch):
    attempts = []
    monkeypatch.setattr(planner_agent, "_call_groq",
                        lambda *a, **k: attempts.append(1) or ("{}", {}))
    monkeypatch.setattr(planner_agent, "_ensure_groq_api_key", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    planner_agent.configure_planner_cache("replay", str(tmp_path))

    with pytest.raises(planner_agent.PlannerCacheMiss):
        _plan(toy_df)
    assert attempts == []


def test_a_malformed_generation_is_never_cached(tmp_path, toy_df, fake_groq):
    calls, responses = fake_groq
    responses.extend(["this is not json", json.dumps(VALID_PLAN)])
    planner_agent.configure_planner_cache("record", str(tmp_path))

    plan, meta = _plan(toy_df)

    assert plan["recommended_models"] == VALID_PLAN["recommended_models"]
    assert meta["attempts"] == 2
    entries = list(tmp_path.glob("*.json"))
    assert len(entries) == 1
    assert json.loads(json.loads(entries[0].read_text(encoding="utf-8"))["content"]) == VALID_PLAN


def test_cache_off_is_the_default_and_writes_nothing(tmp_path, toy_df, fake_groq):
    _, meta = _plan(toy_df)
    assert meta["cache"] == "off"
    assert list(tmp_path.iterdir()) == []


def test_configure_rejects_bad_input():
    with pytest.raises(ValueError):
        planner_agent.configure_planner_cache("sometimes", "x")
    with pytest.raises(ValueError):
        planner_agent.configure_planner_cache("record", None)


# ---------------------------------------------------------------------------
# Split seed
# ---------------------------------------------------------------------------


def test_split_seed_changes_the_partition_and_is_recorded(toy_df, fake_plan):
    a = run_data_agent(toy_df, fake_plan, "income", "classification", split_seed=7)
    b = run_data_agent(toy_df, fake_plan, "income", "classification", split_seed=8)
    default = run_data_agent(toy_df, fake_plan, "income", "classification")

    assert a["train_index"] != b["train_index"]
    assert a["quality_report"]["split_seed"] == 7
    assert default["quality_report"]["split_seed"] == SPLIT_RANDOM_STATE
    assert any("seed 7" in action for action in a["actions_taken"])


# ---------------------------------------------------------------------------
# Scripted reviewer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm,passed,reroutes,expected", [
    ("A", False, 0, "approve"),
    ("B", False, 0, "approve"),
    ("C", False, 0, "reject_model_or_fairness"),
    ("C", False, 1, "reject_model_or_fairness"),
    ("C", False, 2, "approve"),            # reroutes exhausted
    ("C", True, 0, "approve"),
    ("C", None, 0, "approve"),             # not evaluated: nothing to reject on
])
def test_scripted_decision(arm, passed, reroutes, expected):
    decision = ev.scripted_decision(ev.ARMS[arm], {"overall_fairness_passed": passed},
                                    reroutes, max_reroutes=2)
    assert decision["decision"] == expected
    assert decision["reason"]


# ---------------------------------------------------------------------------
# Harness end to end (planner stubbed)
# ---------------------------------------------------------------------------


TOY_SPEC = ev.DatasetSpec(key="toy", label="Toy", openml_id=0, target="income",
                          note="synthetic")


@pytest.fixture
def stub_planner(monkeypatch, fake_plan):
    def _stub(**kwargs):
        meta = kwargs.get("meta_out")
        if meta is not None:
            meta.update({"model_id": "stub/model", "cache": "off",
                         "token_usage": {"total_tokens": 10}})
        return dict(fake_plan)

    monkeypatch.setattr(pipeline_graph, "plan_pipeline", _stub)
    return _stub


def test_prepare_frame_round_trips_like_an_upload(toy_df):
    loaded = ev.prepare_frame(TOY_SPEC, toy_df)
    assert loaded.rows == len(toy_df)
    assert len(loaded.csv_sha256) == 64
    assert ev.prepare_frame(TOY_SPEC, toy_df).csv_sha256 == loaded.csv_sha256


def test_compas_dummies_collapse_into_single_columns():
    raw = pd.DataFrame({
        "race_African-American": [1, 0, 0], "race_Caucasian": [0, 1, 0],
        "age_cat_Lessthan25": [1, 0, 0], "age_cat_25-45": [0, 1, 0],
        "age_cat_Greaterthan45": [0, 0, 1],
        "c_charge_degree_F": [1, 0, 1], "c_charge_degree_M": [0, 1, 0],
        "two_year_recid": [1, 0, 1],
    })
    out = ev.collapse_compas_dummies(raw)
    assert list(out["race"]) == ["African-American", "Caucasian", "Other"]
    assert list(out["age_cat"]) == ["Less than 25", "25-45", "Greater than 45"]
    assert list(out["c_charge_degree"]) == ["F", "M", "F"]
    assert not any(c.startswith("race_") for c in out.columns)
    assert list(out["two_year_recid"]) == [1, 0, 1], "the target must be untouched"


def test_arm_c_run_records_every_gate_and_ends_approved(toy_df, stub_planner):
    original_retries = pipeline_graph.MAX_RETRIES
    original_audit = pipeline_graph.AUDIT_DB_PATH
    loaded = ev.prepare_frame(TOY_SPEC, toy_df)

    row, trajectory = ev.run_single(loaded, ev.ARMS["C"], seed=19)

    assert row["status"] == "approved", row.get("error")
    assert row["gates"] == row["reroutes"] + 1
    assert row["split_seed_recorded"] == 19
    gate_indices = sorted({t["gate_index"] for t in trajectory})
    assert gate_indices == list(range(row["gates"]))
    last_gate = [t for t in trajectory if t["gate_index"] == row["gates"] - 1]
    assert all(t["decision"] == "approve" for t in last_gate)

    assert pipeline_graph.MAX_RETRIES == original_retries
    assert pipeline_graph.AUDIT_DB_PATH == original_audit
    assert pipeline_graph.plan_pipeline is stub_planner


def test_arm_a_disables_retry_where_arm_b_uses_it(toy_df, stub_planner):
    """A frame that fails the quality gate on every attempt separates the arms."""
    df = toy_df.copy()
    rng = np.random.default_rng(3)
    df.loc[rng.choice(len(df), int(len(df) * 0.4), replace=False), "income"] = np.nan
    loaded = ev.prepare_frame(TOY_SPEC, df)

    a_row, a_traj = ev.run_single(loaded, ev.ARMS["A"], seed=42)
    b_row, _ = ev.run_single(loaded, ev.ARMS["B"], seed=42)

    assert a_row["status"] == b_row["status"] == "terminated_quality_cap"
    assert a_row["retries"] == 0 and a_row["planner_calls"] == 1
    assert b_row["retries"] == pipeline_graph.MAX_RETRIES
    assert b_row["planner_calls"] == pipeline_graph.MAX_RETRIES + 1
    assert a_row["auc_roc"] is None, "no model was trained, so there is no metric"
    assert a_traj == []


def test_unevaluated_fairness_is_never_recorded_as_zero(toy_df, monkeypatch, fake_plan):
    def _stub(**kwargs):
        return {**fake_plan, "sensitive_attribute_candidates": ["no_such_column"]}

    monkeypatch.setattr(pipeline_graph, "plan_pipeline", _stub)
    row, trajectory = ev.run_single(ev.prepare_frame(TOY_SPEC, toy_df),
                                    ev.ARMS["C"], seed=42)

    assert row["status"] == "approved"
    assert row["fairness_evaluated"] is False
    assert row["overall_fairness_passed"] is None
    assert row["min_disparate_impact"] is None
    assert row["n_violations"] is None
    assert row["gates"] == 1
    assert trajectory[0]["decision_reason"].startswith("fairness not evaluated")


# ---------------------------------------------------------------------------
# Aggregation and figure
# ---------------------------------------------------------------------------


def _synthetic_runs():
    runs = []
    for seed, (auc, di, viol) in zip((1, 2, 3), ((0.80, 0.50, 2), (0.82, 0.60, 1), (0.81, 0.55, 2))):
        for arm in ("A", "C"):
            runs.append({"dataset": "d1", "arm": arm, "seed": seed, "status": "approved",
                         "auc_roc": auc, "accuracy": 0.7, "fairness_evaluated": True,
                         "overall_fairness_passed": False, "min_disparate_impact": di,
                         "n_violations": viol, "reroutes": 0 if arm == "A" else 2,
                         "retries": 0, "wall_clock_s": 1.0, "final_model": "RF"})
        runs.append({"dataset": "d2", "arm": "A", "seed": seed, "status": "approved",
                     "auc_roc": 0.7, "accuracy": 0.7, "fairness_evaluated": False,
                     "overall_fairness_passed": None, "min_disparate_impact": None,
                     "n_violations": None, "reroutes": 0, "retries": 0,
                     "wall_clock_s": 1.0, "final_model": "RF"})
    return runs


def test_mean_std_excludes_missing_values():
    assert ev.mean_std([None, "", float("nan")]) is None
    mean, std, n = ev.mean_std([1.0, None, 3.0])
    assert (mean, n) == (2.0, 2)
    assert ev.fmt_mean_std(None) == "n/a"


def test_summarise_never_reports_unevaluated_fairness_as_a_number():
    markdown, rows = ev.summarise(_synthetic_runs(), trajectory=[])
    d2 = next(r for r in rows if r["dataset"] == "d2")
    assert d2["fairness_evaluated"] == 0
    assert d2["min_di_mean"] is None
    assert "n/a" in markdown


def test_reroute_effects_compare_first_and_approved_gates():
    trajectory = [
        {"dataset": "d", "arm": "C", "seed": 1, "gate_index": 0, "model": "XGB",
         "auc_roc": 0.9, "attribute": "sex", "disparate_impact": 0.3, "violation": True},
        {"dataset": "d", "arm": "C", "seed": 1, "gate_index": 0, "model": "XGB",
         "auc_roc": 0.9, "attribute": "race", "disparate_impact": 0.7, "violation": True},
        {"dataset": "d", "arm": "C", "seed": 1, "gate_index": 1, "model": "LR",
         "auc_roc": 0.85, "attribute": "sex", "disparate_impact": 0.4, "violation": True},
        {"dataset": "d", "arm": "C", "seed": 1, "gate_index": 1, "model": "LR",
         "auc_roc": 0.85, "attribute": "race", "disparate_impact": 0.85, "violation": False},
    ]
    (effect,) = ev.reroute_effects(trajectory)
    assert effect["rerouted"] is True
    assert (effect["violations_first"], effect["violations_final"]) == (2, 1)
    assert (effect["min_di_first"], effect["min_di_final"]) == (0.3, 0.4)
    assert (effect["first_model"], effect["final_model"]) == ("XGB", "LR")


def test_charts_render_both_themes_and_skip_unmeasured_runs(tmp_path):
    pytest.importorskip("matplotlib")
    result = ev.render_charts(_synthetic_runs(), tmp_path)

    assert (tmp_path / "fairness_vs_auc.png").is_file()
    assert (tmp_path / "fairness_vs_auc_dark.png").is_file()
    assert result["plotted"] == {"d1": 6, "d2": 0}


@pytest.mark.parametrize("values,expected", [
    ({"unresolved_training_failure": True, "training_result": {"error": "x"}},
     "terminated_training_failure"),
    ({"human_decision": "approve", "training_result": {"error": "x"}},
     "approved_without_model"),
    ({"human_decision": "approve", "training_result": {"selected_model_name": "RF"}},
     "approved"),
    ({"unresolved_quality_issue": True}, "terminated_quality_cap"),
])
def test_run_status_is_never_approved_without_a_model(values, expected):
    assert ev._final_metrics(values, gates=0)["status"] == expected


def _approved(dataset, arm, seed, auc, di):
    return {"dataset": dataset, "arm": arm, "seed": seed, "status": "approved",
            "auc_roc": auc, "min_disparate_impact": di}


def test_coincident_arm_means_share_one_label(tmp_path):
    pytest.importorskip("matplotlib")
    runs = [_approved("d1", arm, seed, 0.8, 0.5) for arm in ("A", "B") for seed in (1, 2)]
    runs += [_approved("d1", "C", seed, 0.7, 0.9) for seed in (1, 2)]
    result = ev.render_charts(runs, tmp_path)
    assert sorted(result["labels"]["d1"]) == ["A, B mean", "C mean"]


def test_panels_say_why_an_arm_is_not_plotted(tmp_path):
    pytest.importorskip("matplotlib")
    runs = [_approved("d1", "A", 1, 0.8, 0.5),
            {"dataset": "d1", "arm": "C", "seed": 1,
             "status": "terminated_training_failure", "auc_roc": None,
             "min_disparate_impact": None}]
    result = ev.render_charts(runs, tmp_path)
    assert result["notes"]["d1"] == ["C: 1 terminated training failure"]


def _synthetic_trajectory():
    return [
        {"dataset": "d1", "arm": "C", "seed": 1, "gate_index": 0, "model": "XGB",
         "auc_roc": 0.9, "accuracy": 0.8, "overall_fairness_passed": False,
         "attribute": "sex", "disparate_impact": 0.3,
         "demographic_parity_difference": 0.25, "violation": True,
         "decision": "reject_model_or_fairness", "decision_reason": "violation"},
        {"dataset": "d1", "arm": "C", "seed": 1, "gate_index": 1, "model": "LR",
         "auc_roc": 0.85, "accuracy": 0.78, "overall_fairness_passed": False,
         "attribute": "sex", "disparate_impact": 0.85,
         "demographic_parity_difference": 0.12, "violation": True,
         "decision": "approve", "decision_reason": "exhausted"},
        {"dataset": "d2", "arm": "A", "seed": 1, "gate_index": 0, "model": "RF",
         "auc_roc": 0.7, "accuracy": 0.7, "overall_fairness_passed": None,
         "attribute": None, "disparate_impact": None,
         "demographic_parity_difference": None, "violation": None,
         "decision": "approve", "decision_reason": "first gate"},
    ]


def test_parity_difference_is_reported_alongside_disparate_impact():
    runs = [{**r, "max_parity_difference": 0.2} for r in _synthetic_runs()]
    markdown, rows = ev.summarise(runs, _synthetic_trajectory())
    assert "Max parity difference" in markdown
    assert "Max parity difference: first → approved" in markdown
    d1c = next(r for r in rows if r["dataset"] == "d1" and r["arm"] == "C")
    assert d1c["max_dpd_mean"] == pytest.approx(0.2)


def test_results_round_trip_through_csv_without_changing_the_report(tmp_path):
    runs, trajectory = _synthetic_runs(), _synthetic_trajectory()
    ev._write_csv(tmp_path / "runs.csv", ev.RUN_FIELDS, runs)
    ev._write_csv(tmp_path / "fairness_trajectory.csv", ev.TRAJECTORY_FIELDS, trajectory)

    loaded_runs, loaded_trajectory, _ = ev.load_results(tmp_path)

    assert ev.summarise(loaded_runs, loaded_trajectory)[0] == ev.summarise(runs, trajectory)[0]
    d2 = next(r for r in loaded_runs if r["dataset"] == "d2")
    assert d2["fairness_evaluated"] is False
    assert d2["min_disparate_impact"] is None, "a blank cell must come back as None, not 0"
    assert loaded_trajectory[0]["violation"] is True
    assert loaded_trajectory[2]["attribute"] is None


def test_summarise_only_rebuilds_the_report_without_running_pipelines(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    ev._write_csv(tmp_path / "runs.csv", ev.RUN_FIELDS, _synthetic_runs())
    ev._write_csv(tmp_path / "fairness_trajectory.csv", ev.TRAJECTORY_FIELDS,
                  _synthetic_trajectory())
    (tmp_path / "manifest.json").write_text(json.dumps({
        "mode": "replay", "generated_at": "2026-09-14T00:00:00+00:00",
        "arms": {"A": {}, "C": {}}, "datasets": {}, "seeds": [1, 2, 3],
    }), encoding="utf-8")

    def _no_pipelines(*args, **kwargs):
        raise AssertionError("--summarise-only must not run a pipeline")

    monkeypatch.setattr(ev, "run_single", _no_pipelines)
    monkeypatch.setattr(ev, "load_dataset", _no_pipelines)

    assert ev.main(["--summarise-only", "--out", str(tmp_path)]) == 0
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "rebuilt from the recorded CSVs" in summary
    assert (tmp_path / "fairness_vs_auc.png").is_file()


@pytest.mark.parametrize("desired", [
    [0.50, 0.52],            # two means almost on top of each other
    [0.81],                  # a mean on the threshold line
    [0.74, 0.79, 0.83],      # a cluster straddling the threshold
    [0.02, 1.04],            # outside the drawable range
])
def test_label_placement_keeps_labels_apart_and_off_the_threshold(desired):
    ys = ev.place_labels(desired)
    low, high = ev.LABEL_BLOCKED_BAND
    assert all(not (low < y < high) for y in ys)
    assert all(ev.LABEL_Y_RANGE[0] <= y <= ev.LABEL_Y_RANGE[1] for y in ys)
    ordered = sorted(ys)
    assert all(b - a >= ev.MIN_LABEL_GAP - 1e-9 for a, b in zip(ordered, ordered[1:]))


def test_rendered_mean_labels_respect_the_placement_rules(tmp_path):
    pytest.importorskip("matplotlib")
    runs = [_approved("d1", "A", s, 0.80, 0.50) for s in (1, 2)]
    runs += [_approved("d1", "C", s, 0.70, 0.52) for s in (1, 2)]
    runs += [_approved("d2", "A", s, 0.80, 0.81) for s in (1, 2)]
    result = ev.render_charts(runs, tmp_path)

    d1 = sorted(y for _, y in result["label_positions"]["d1"])
    assert d1[1] - d1[0] >= ev.MIN_LABEL_GAP - 1e-9
    ((_, d2_y),) = result["label_positions"]["d2"]
    assert not (ev.LABEL_BLOCKED_BAND[0] < d2_y < ev.LABEL_BLOCKED_BAND[1])


def test_quick_mode_never_writes_into_committed_results():
    args = ev.build_parser().parse_args(["--quick"])
    assert ev.resolve_output_dir(args) == ev.QUICK_OUT_DIR
    explicit = ev.build_parser().parse_args(["--quick", "--out", "somewhere"])
    assert str(ev.resolve_output_dir(explicit)) == "somewhere"
