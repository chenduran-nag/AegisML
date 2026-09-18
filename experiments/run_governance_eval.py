"""
run_governance_eval.py
======================
Quantitative evaluation of the AegisML governance loops.

QUESTION:
  Do the governance loops change outcomes? In particular: when the approval gate
  rejects a model for a fairness violation and routes the run back to training
  (Loop 3), is the model finally approved any fairer — and what does it cost in
  accuracy?

DESIGN:
  Every run drives the real compiled graph (pipeline_graph.build_graph) end to end:
  EDA, planner, Data Agent, Training, Fairness and the governance gate. Only the
  human reviewer is replaced, by a scripted policy (scripted_decision). Four arms:

    A  governance off       No automatic data-quality retry; approve at the first gate.
    B  auto-retry           Automatic retry at the default cap; approve at the first gate.
    C  fairness reviewer    Automatic retry; at each gate, reject the model while a
                            fairness violation remains and reroutes remain, else approve.
    D  mitigation reviewer  As C, but the rejection is reject_and_mitigate: the same
                            candidates are retrained with reweighing (mitigation.py).

WHAT VARIES ACROSS SEEDS, AND WHAT DOES NOT:
  - The train/test split seed varies (PipelineState["split_seed"]).
  - Model seeds are fixed (training_agent.RANDOM_STATE), so seed-to-seed spread
    reflects the data partition, not model initialisation.
  - The planner is held fixed. Each distinct prompt goes to the LLM once, in
    --mode record, and that response is replayed for every seed and arm. Differences
    between arms are therefore attributable to the governance loops rather than to
    LLM sampling variance, which is a separate question this harness does not answer.

REPRODUCIBILITY:
  python experiments/run_governance_eval.py            (replay is the default)
  regenerates every table and figure from experiments/planner_cache/ without a Groq
  key and without contacting the LLM. Datasets are fetched from OpenML on first use
  and cached in experiments/.data_cache/. Every run records the SHA-256 of the exact
  CSV bytes fed to the pipeline, so a re-download can be checked against the results.

DATASET LOADING:
  Each dataset is serialised to CSV and parsed back before it reaches the pipeline —
  exactly what the dashboard does with an uploaded file — so the evaluation exercises
  the same dtype handling as a real run. See the COMPAS note in DATASETS for the one
  loader transform.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import logging
import math
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
from langgraph.types import Command  # noqa: E402

import fairness_agent  # noqa: E402
import pipeline_graph  # noqa: E402
import planner_agent  # noqa: E402
from policy import load_policy, policy_sha256, with_overrides  # noqa: E402
from graph_state import df_to_bytes  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_OUT_DIR = EXPERIMENTS_DIR / "results"
QUICK_OUT_DIR = EXPERIMENTS_DIR / "results_quick"
DEFAULT_CACHE_DIR = EXPERIMENTS_DIR / "planner_cache"
DEFAULT_DATA_HOME = EXPERIMENTS_DIR / ".data_cache"

# 42 is the application's default split seed, so the first seed of every arm
# reproduces the partition a dashboard run would use.
DEFAULT_SEEDS = (42, 7, 19, 73, 128)
QUICK_SEEDS = 2
QUICK_MAX_ROWS = 5000

# Safety valve only. The rejection cap ends every run well before this.
MAX_GATES_PER_RUN = 10

RUN_FIELDS = [
    "dataset", "arm", "seed", "status", "error", "gates", "final_model",
    "accuracy", "f1", "auc_roc", "gate_auc_roc", "evaluated_on",
    "fairness_evaluated", "overall_fairness_passed", "attributes_evaluated",
    "n_violations", "n_advisory_violations", "min_disparate_impact", "max_parity_difference",
    "max_equal_opportunity_difference", "protected_attributes_evaluated",
    "protected_attributes_unaudited", "mitigated_attributes",
    "approvers", "n_signoffs",
    "retries", "reroutes", "eda_findings", "columns_dropped",
    "train_rows", "validation_rows", "test_rows", "split_seed_recorded",
    "planner_calls", "planner_cache_hits", "planner_tokens_recorded", "planner_model",
    "dataset_rows", "dataset_sha256", "wall_clock_s",
]

TRAJECTORY_FIELDS = [
    "dataset", "arm", "seed", "gate_index", "model", "auc_roc", "accuracy",
    "overall_fairness_passed", "attribute", "disparate_impact",
    "demographic_parity_difference", "equal_opportunity_difference", "protected",
    "excluded_groups", "violation", "decision", "decision_reason",
]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    openml_id: int
    target: str
    note: str
    transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    task_type: str = "classification"


def _flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str), errors="coerce").fillna(0).astype(float) == 1.0


def collapse_compas_dummies(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Undo OpenML's pre-applied one-hot encoding of COMPAS categoricals.

    OpenML ships race, age category and charge degree as indicator columns
    (race_African-American, race_Caucasian, ...). Fed in that form, the Data Agent
    would one-hot encode each indicator a second time, and the Fairness Agent's
    prefix reconstruction of "race" would then see four overlapping dummy columns
    and invent meaningless groups. Collapsing each group back to one categorical
    column lets the pipeline treat COMPAS like every other dataset.

    Deterministic, row-by-row, and uses no target information. A row with no race
    flag set becomes "Other": this OpenML version keeps indicators only for
    African-American and Caucasian defendants.
    """
    out = frame.copy()
    groups = {
        "race": {"race_African-American": "African-American",
                 "race_Caucasian": "Caucasian"},
        "age_cat": {"age_cat_Lessthan25": "Less than 25",
                    "age_cat_25-45": "25-45",
                    "age_cat_Greaterthan45": "Greater than 45"},
        "c_charge_degree": {"c_charge_degree_F": "F",
                            "c_charge_degree_M": "M"},
    }
    for new_column, mapping in groups.items():
        present = [c for c in mapping if c in out.columns]
        if not present:
            continue
        value = pd.Series("Other", index=out.index, dtype=object)
        for column in present:
            value = value.mask(_flag(out[column]), mapping[column])
        out = out.drop(columns=present)
        out[new_column] = value
    return out


DATASETS: dict[str, DatasetSpec] = {
    "adult": DatasetSpec(
        key="adult", label="UCI Adult Income", openml_id=1590, target="class",
        note=("48,842 rows (OpenML v2 combines the original train and test files). "
              "Positive class '>50K' is the favourable outcome."),
    ),
    "credit_g": DatasetSpec(
        key="credit_g", label="German Credit", openml_id=31, target="class",
        note=("1,000 rows. Positive class 'good' is the favourable outcome. Age is "
              "audited in bands, but a 200-row test split often leaves fewer than two "
              "bands with 30 rows, so age goes unaudited on some seeds. "
              "`personal_status` combines sex and marital status and is not recognised "
              "as protected, because protected attributes are detected by column name."),
    ),
    "bank_marketing": DatasetSpec(
        key="bank_marketing", label="Bank Marketing", openml_id=46910,
        target="SubscribeTermDeposit",
        note=("OpenML v10 (46910), the only version with named columns: v1 (1461) and "
              "v9 (45065) use anonymised V1..V16. 45,211 rows. Positive class 'yes' "
              "(subscribed)."),
    ),
    "compas": DatasetSpec(
        key="compas", label="COMPAS (two-year recidivism)", openml_id=42192,
        target="two_year_recid", transform=collapse_compas_dummies,
        note=("OpenML v3 (42192). Race, age category and charge degree arrive "
              "pre-one-hot-encoded and are collapsed back into single columns by the "
              "loader. Positive class 1 is the ADVERSE outcome (predicted "
              "recidivism): disparate impact still flags imbalance between groups, "
              "but a higher rate is worse, not better."),
    ),
}


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    frame: pd.DataFrame
    csv_sha256: str

    @property
    def rows(self) -> int:
        return int(len(self.frame))


def prepare_frame(spec: DatasetSpec, frame: pd.DataFrame,
                  max_rows: Optional[int] = None) -> LoadedDataset:
    """Transform, optionally subsample, then round-trip through CSV like an upload."""
    frame = frame.copy()
    if spec.transform is not None:
        frame = spec.transform(frame)
    if max_rows and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=0)
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    parsed = pd.read_csv(io.BytesIO(csv_bytes))
    if spec.target not in parsed.columns:
        raise ValueError(f"{spec.key}: target column {spec.target!r} not found "
                         f"in {list(parsed.columns)}")
    return LoadedDataset(spec=spec, frame=parsed,
                         csv_sha256=hashlib.sha256(csv_bytes).hexdigest())


def load_dataset(spec: DatasetSpec, data_home: Path,
                 max_rows: Optional[int] = None) -> LoadedDataset:
    from sklearn.datasets import fetch_openml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bunch = fetch_openml(data_id=spec.openml_id, as_frame=True,
                             data_home=str(data_home), parser="auto")
    return prepare_frame(spec, bunch.frame, max_rows=max_rows)


# ---------------------------------------------------------------------------
# Arms and the scripted reviewer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    description: str
    max_retries: int
    reviewer: str   # "approve_first" | "reject_violations" | "mitigate_violations"


ARMS: dict[str, Arm] = {
    "A": Arm("A", "governance off",
             "No automatic data-quality retry; approve at the first gate.",
             max_retries=0, reviewer="approve_first"),
    "B": Arm("B", "auto-retry",
             "Automatic data-quality retry at the default cap; approve at the first gate.",
             max_retries=pipeline_graph.MAX_RETRIES, reviewer="approve_first"),
    "C": Arm("C", "fairness reviewer",
             "Automatic retry; reject the model while a fairness violation remains "
             "and reroutes remain, otherwise approve.",
             max_retries=pipeline_graph.MAX_RETRIES, reviewer="reject_violations"),
    "D": Arm("D", "mitigation reviewer",
             "Automatic retry; while a fairness violation remains and reroutes remain, "
             "reject and mitigate (reweighing), otherwise approve.",
             max_retries=pipeline_graph.MAX_RETRIES, reviewer="mitigate_violations"),
}


# Arms whose reviewer rejects at the gate, and what the rejection does. Each gets its
# own first-gate vs approved-model table.
REROUTE_ARMS = {"C": "rerouting to the next model", "D": "mitigation"}

# The scripted reviewer's identities. Two of them, because approving a model with a
# fairness violation takes two different reviewers under the committed policy. The
# second one exists to satisfy that rule mechanically — which is exactly the point
# being measured: the arms differ in what the reviewer DOES, never in who signs.
EVAL_REVIEWERS = (
    {"reviewer_id": "eval.reviewer.1", "reviewer_role": "ml_engineer",
     "reviewer_authenticated": True},
    {"reviewer_id": "eval.reviewer.2", "reviewer_role": "compliance_officer",
     "reviewer_authenticated": True},
)


def scripted_decision(arm: Arm, payload: dict, reroutes_used: int,
                      max_reroutes: int) -> dict:
    """
    The scripted reviewer. Pure: it sees only what a human sees at the gate.

    Note the "not evaluated" branch. When fairness was never measured there is no
    evidence to reject on, so the reviewer approves — and the run is reported as
    unevaluated, never as fair.
    """
    def approve(reason):
        return {"decision": "approve", "human_feedback": "", "reason": reason}

    if arm.reviewer == "approve_first":
        return approve("arm approves at the first gate")

    passed = payload.get("overall_fairness_passed")
    if passed is None:
        if payload.get("fairness_report"):
            return approve("no violation, but a protected attribute was not audited")
        return approve("fairness not evaluated: no evidence to reject on")
    if passed:
        return approve("no fairness violation")
    if reroutes_used >= max_reroutes:
        return approve("violation remains but reroutes are exhausted")
    if arm.reviewer == "mitigate_violations":
        return {
            "decision": "reject_and_mitigate",
            "human_feedback": "",
            "reason": "fairness violation with reroutes remaining: mitigate",
        }
    return {
        "decision": "reject_model_or_fairness",
        "human_feedback": "Fairness violation at the governance gate.",
        "reason": "fairness violation with reroutes remaining",
    }


# ---------------------------------------------------------------------------
# Running one (dataset, arm, seed)
# ---------------------------------------------------------------------------


def _arm_policy_state(arm: Arm) -> dict:
    """State keys for the committed policy.yaml with the arm's retry cap applied."""
    policy = with_overrides(load_policy()["policy"],
                            {"governance": {"max_retries": arm.max_retries}})
    return {"policy": policy, "policy_version": policy["version"],
            "policy_sha256": policy_sha256(policy)}


@contextlib.contextmanager
def _isolated_graph(workdir: Path, max_retries: int, planner_calls: list):
    """
    Build a graph whose databases and outputs live in `workdir`, with the arm's retry
    cap, and count planner calls. Module state is restored afterwards, even on error,
    so one run can never leak its configuration into the next.
    """
    names = ("AUDIT_DB_PATH", "SAVED_MODELS_DIR", "ARTIFACTS_DIR",
             "MAX_RETRIES", "plan_pipeline")
    saved = {name: getattr(pipeline_graph, name) for name in names}
    real_plan = saved["plan_pipeline"]

    def counting_plan(**kwargs):
        plan = real_plan(**kwargs)
        planner_calls.append(dict(kwargs.get("meta_out") or {}))
        return plan

    graph = None
    try:
        pipeline_graph.AUDIT_DB_PATH = str(workdir / "audit.db")
        pipeline_graph.SAVED_MODELS_DIR = str(workdir / "saved_models")
        pipeline_graph.ARTIFACTS_DIR = str(workdir / "artifacts")
        pipeline_graph.MAX_RETRIES = max_retries
        pipeline_graph.plan_pipeline = counting_plan
        graph = pipeline_graph.build_graph(db_path=str(workdir / "state.db"))
        yield graph
    finally:
        for name, value in saved.items():
            setattr(pipeline_graph, name, value)
        conn = getattr(getattr(graph, "checkpointer", None), "conn", None)
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


def _min(values):
    values = [v for v in values if v is not None]
    return min(values) if values else None


def _max(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _gate_rows(dataset: str, arm: Arm, seed: int, gate_index: int,
               payload: dict, decision: dict) -> list[dict]:
    metrics = payload.get("selected_model_metrics") or {}
    base = {
        "dataset": dataset, "arm": arm.key, "seed": seed, "gate_index": gate_index,
        "model": payload.get("selected_model_name"),
        "auc_roc": metrics.get("auc_roc"), "accuracy": metrics.get("accuracy"),
        "overall_fairness_passed": payload.get("overall_fairness_passed"),
        "decision": decision["decision"], "decision_reason": decision["reason"],
    }
    report = payload.get("fairness_report") or []
    if not report:
        return [{**base, "attribute": None, "disparate_impact": None,
                 "demographic_parity_difference": None,
                 "equal_opportunity_difference": None, "protected": None,
                 "excluded_groups": None, "violation": None}]
    return [{**base, "attribute": r.get("attribute"),
             "disparate_impact": r.get("disparate_impact"),
             "demographic_parity_difference": r.get("demographic_parity_difference"),
             "equal_opportunity_difference": r.get("equal_opportunity_difference"),
             "protected": r.get("protected"),
             "excluded_groups": ";".join(f"{k}={v}" for k, v in
                                         (r.get("excluded_groups") or {}).items()) or None,
             "violation": r.get("violation")} for r in report]


def _final_metrics(values: dict, gates: int) -> dict:
    training = values.get("training_result") or {}
    quality = (values.get("data_agent_result") or {}).get("quality_report") or {}
    gate_metrics = training.get("selected_model_metrics") or {}
    # An approved run is reported on the untouched test rows (final_evaluation); every
    # gate decision used the validation rows. Runs that never reached approval keep
    # their gate numbers.
    final = values.get("final_evaluation") or {}
    fairness = final.get("fairness") or values.get("fairness_result") or {}
    metrics = final.get("metrics") or gate_metrics
    report = fairness.get("fairness_report") or []
    verdict_rows = [r for r in report
                    if r.get("counts_toward_verdict", r.get("protected")) is not False]

    if values.get("unresolved_quality_issue"):
        status = "terminated_quality_cap"
    elif values.get("unresolved_training_failure"):
        status = "terminated_training_failure"
    elif values.get("unresolved_approval_blocked"):
        status = "terminated_approval_blocked"
    elif values.get("unresolved_human_rejection"):
        status = "terminated_rejection_cap"
    elif values.get("human_decision") == "approve" and not training.get("selected_model_name"):
        # Defensive: the graph no longer allows this, but a run must never be
        # counted as approved when no model exists.
        status = "approved_without_model"
    elif values.get("human_decision") == "approve":
        status = "approved"
    else:
        status = "ended_without_decision"

    approvers = [record.get("reviewer_id")
                 for record in (values.get("reviewer_decisions") or [])
                 if record.get("decision") == "approve"]

    return {
        "status": status,
        "gates": gates,
        # Who signed, so a results file can show that no approval had a single author.
        "approvers": ";".join(a or "unnamed" for a in approvers),
        "n_signoffs": len(approvers),
        "final_model": training.get("selected_model_name"),
        "accuracy": metrics.get("accuracy"),
        "f1": metrics.get("f1"),
        "auc_roc": metrics.get("auc_roc"),
        "gate_auc_roc": gate_metrics.get("auc_roc"),
        "evaluated_on": "test" if final.get("metrics") else "gate",
        "fairness_evaluated": bool(fairness.get("fairness_evaluated")),
        "overall_fairness_passed": fairness.get("overall_fairness_passed"),
        "attributes_evaluated": ";".join(str(r.get("attribute")) for r in report),
        # Fairness columns follow the verdict: protected attributes only. Advisory
        # (unprotected) violations are counted separately. None, not 0, when nothing
        # was measured (invariant 4).
        "n_violations": (sum(1 for r in verdict_rows if r.get("violation"))
                         if verdict_rows else None),
        "n_advisory_violations": (sum(1 for r in report if r.get("violation")
                                      and r not in verdict_rows) if report else None),
        "min_disparate_impact": _min(r.get("disparate_impact") for r in verdict_rows),
        "max_parity_difference": _max(
            r.get("demographic_parity_difference") for r in verdict_rows),
        "max_equal_opportunity_difference": _max(
            r.get("equal_opportunity_difference") for r in verdict_rows),
        "protected_attributes_evaluated": (sum(1 for r in report if r.get("protected"))
                                           if report else None),
        "protected_attributes_unaudited": ";".join(
            str(p.get("attribute")) for p in fairness.get("protected_attributes_unaudited") or []),
        "mitigated_attributes": ";".join((values.get("mitigation") or {}).get("attributes") or []),
        "retries": values.get("retry_count", 0),
        "reroutes": values.get("rejection_reroute_count", 0),
        "eda_findings": len(values.get("eda_findings") or []),
        "columns_dropped": ";".join(quality.get("columns_dropped") or []),
        "train_rows": quality.get("train_rows"),
        "validation_rows": quality.get("validation_rows"),
        "test_rows": quality.get("test_rows"),
        "split_seed_recorded": quality.get("split_seed"),
    }


def run_single(dataset: LoadedDataset, arm: Arm, seed: int,
               verbose: bool = False) -> tuple[dict, list[dict]]:
    """Run one (dataset, arm, seed) through the full graph with the scripted reviewer."""
    planner_calls: list[dict] = []
    trajectory: list[dict] = []
    row: dict = {
        "dataset": dataset.spec.key, "arm": arm.key, "seed": seed, "error": None,
        "dataset_rows": dataset.rows, "dataset_sha256": dataset.csv_sha256,
    }
    started = time.perf_counter()
    workdir = Path(tempfile.mkdtemp(prefix=f"aegis-eval-{dataset.spec.key}-"))
    sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())

    try:
        with sink, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _isolated_graph(workdir, arm.max_retries, planner_calls) as graph:
                config = {"configurable": {
                    "thread_id": f"{dataset.spec.key}-{arm.key}-{seed}"}}
                graph.invoke({
                    "df_bytes": df_to_bytes(dataset.frame),
                    "dataset_sha256": dataset.csv_sha256,
                    "target_column": dataset.spec.target,
                    "task_type": dataset.spec.task_type,
                    # Deliberately empty: an objective such as "be fair" would change
                    # the plan and confound the comparison between arms.
                    "business_objective": "",
                    "split_seed": seed,
                    "retry_count": 0,
                    "unresolved_quality_issue": False,
                    "last_failure_reason": None,
                    "human_decision": None,
                    "human_feedback": None,
                    "rejection_reroute_count": 0,
                    "unresolved_human_rejection": False,
                    "rejected_models": [],
                    # The committed policy, with this arm's retry cap: every run records
                    # the exact policy it ran under.
                    **_arm_policy_state(arm),
                }, config=config)

                gates = 0
                second_signoffs = 0
                while True:
                    snapshot = graph.get_state(config)
                    if "human_approval_node" not in snapshot.next:
                        break
                    payload = snapshot.tasks[0].interrupts[0].value

                    # A re-opened gate awaiting the second sign-off is not a new review:
                    # the model and the evidence are unchanged, so it is not counted as
                    # a gate and adds no trajectory row.
                    if payload.get("awaiting_second_approval"):
                        if second_signoffs >= MAX_GATES_PER_RUN:
                            raise RuntimeError("exceeded second sign-offs")
                        second_signoffs += 1
                        graph.invoke(Command(resume={
                            "decision": "approve",
                            "human_feedback": "Second sign-off (scripted reviewer).",
                            **EVAL_REVIEWERS[1],
                        }), config=config)
                        continue

                    if gates >= MAX_GATES_PER_RUN:
                        raise RuntimeError(f"exceeded {MAX_GATES_PER_RUN} gates")
                    decision = scripted_decision(
                        arm, payload,
                        snapshot.values.get("rejection_reroute_count", 0),
                        pipeline_graph.MAX_HUMAN_REROUTES,
                    )
                    trajectory.extend(_gate_rows(dataset.spec.key, arm, seed,
                                                 gates, payload, decision))
                    gates += 1
                    graph.invoke(Command(resume={
                        "decision": decision["decision"],
                        "human_feedback": decision["human_feedback"],
                        **EVAL_REVIEWERS[0],
                    }), config=config)

                row.update(_final_metrics(graph.get_state(config).values, gates))
    except planner_agent.PlannerCacheMiss:
        raise
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    row["wall_clock_s"] = round(time.perf_counter() - started, 2)
    row["planner_calls"] = len(planner_calls)
    row["planner_cache_hits"] = sum(1 for m in planner_calls if m.get("cache") == "hit")
    row["planner_tokens_recorded"] = sum(
        int((m.get("token_usage") or {}).get("total_tokens") or 0) for m in planner_calls)
    row["planner_model"] = next(
        (m["model_id"] for m in planner_calls if m.get("model_id")), None)
    return row, trajectory


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _numbers(values) -> list[float]:
    out = []
    for v in values:
        if v is None or v == "":
            continue
        f = float(v)
        if not math.isnan(f):
            out.append(f)
    return out


def mean_std(values) -> Optional[tuple[float, float, int]]:
    nums = _numbers(values)
    if not nums:
        return None
    return (statistics.fmean(nums),
            statistics.stdev(nums) if len(nums) > 1 else 0.0,
            len(nums))


def fmt_mean_std(stat, digits: int = 3) -> str:
    if stat is None:
        return "n/a"
    mean, std, _ = stat
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _gate_stats(rows: list[dict]) -> dict:
    # Protected attributes only, matching the verdict. A row without a protected flag
    # (results recorded before the flag existed) counts; only an explicit False is advisory.
    evaluated = [r for r in rows if r.get("attribute") is not None
                 and r.get("protected") not in (False, "False")]
    return {
        "violations": (sum(1 for r in evaluated if r.get("violation"))
                       if evaluated else None),
        "min_di": _min(r.get("disparate_impact") for r in evaluated),
        "max_dpd": _max(r.get("demographic_parity_difference") for r in evaluated),
        "auc": rows[0].get("auc_roc") if rows else None,
        "model": rows[0].get("model") if rows else None,
    }


def reroute_effects(trajectory: list[dict]) -> list[dict]:
    """First gate vs approval gate, per run, for runs that visited the gate."""
    by_run: dict[tuple, dict[int, list[dict]]] = {}
    for t in trajectory:
        key = (t["dataset"], t["arm"], t["seed"])
        by_run.setdefault(key, {}).setdefault(int(t["gate_index"]), []).append(t)
    effects = []
    for (dataset, arm, seed), gates in by_run.items():
        first, last = gates[min(gates)], gates[max(gates)]
        f, l = _gate_stats(first), _gate_stats(last)
        effects.append({
            "dataset": dataset, "arm": arm, "seed": seed,
            "gates": len(gates), "rerouted": len(gates) > 1,
            "first_model": f["model"], "final_model": l["model"],
            "violations_first": f["violations"], "violations_final": l["violations"],
            "min_di_first": f["min_di"], "min_di_final": l["min_di"],
            "max_dpd_first": f["max_dpd"], "max_dpd_final": l["max_dpd"],
            "auc_first": f["auc"], "auc_final": l["auc"],
        })
    return effects


def summarise(runs: list[dict], trajectory: list[dict],
              arms: dict[str, Arm] = ARMS,
              dataset_meta: Optional[dict] = None) -> tuple[str, list[dict]]:
    """Return (markdown, summary_rows). Numbers only; no interpretation."""
    dataset_meta = dataset_meta or {}
    datasets = list(dict.fromkeys(r["dataset"] for r in runs))
    arm_keys = [k for k in arms if any(r["arm"] == k for r in runs)]
    effects = reroute_effects(trajectory)
    summary_rows: list[dict] = []
    md: list[str] = []

    for ds in datasets:
        meta = dataset_meta.get(ds, {})
        md.append(f"### {meta.get('label', ds)}\n")
        if meta:
            md.append(f"{meta.get('rows', '?'):,} rows · target `{meta.get('target')}` · "
                      f"OpenML {meta.get('openml_id')} · CSV SHA-256 "
                      f"`{str(meta.get('csv_sha256', ''))[:12]}…`\n")
            md.append(f"{meta.get('note', '')}\n")
        md.append("| Arm | Approved | AUC | Accuracy | Fairness evaluated | "
                  "Approved with a violation | Min disparate impact | Max parity difference | "
                  "Max equal-opportunity difference | Protected attributes audited | "
                  "Violated attributes | Reroutes | Retries | Seconds / run |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

        for arm_key in arm_keys:
            ar = [r for r in runs if r["dataset"] == ds and r["arm"] == arm_key]
            if not ar:
                continue
            approved = [r for r in ar if r.get("status") == "approved"]
            evaluated = [r for r in approved if r.get("fairness_evaluated") in (True, "True")]
            violating = [r for r in evaluated
                         if r.get("overall_fairness_passed") in (False, "False")]
            # Audited, no violation, but a protected attribute went unmeasured: no pass.
            partial = [r for r in evaluated if r.get("overall_fairness_passed") in (None, "")]
            errors = sum(1 for r in ar if r.get("status") == "error")
            stats = {
                "auc": mean_std(r.get("auc_roc") for r in approved),
                "accuracy": mean_std(r.get("accuracy") for r in approved),
                "min_di": mean_std(r.get("min_disparate_impact") for r in evaluated),
                "max_dpd": mean_std(r.get("max_parity_difference") for r in evaluated),
                "max_eod": mean_std(r.get("max_equal_opportunity_difference") for r in evaluated),
                "protected": mean_std(r.get("protected_attributes_evaluated") for r in evaluated),
                "violations": mean_std(r.get("n_violations") for r in evaluated),
                "reroutes": mean_std(r.get("reroutes") for r in ar),
                "retries": mean_std(r.get("retries") for r in ar),
                "seconds": mean_std(r.get("wall_clock_s") for r in ar),
            }
            summary_rows.append({
                "dataset": ds, "arm": arm_key, "runs": len(ar), "errors": errors,
                "approved": len(approved), "fairness_evaluated": len(evaluated),
                "fairness_partial": len(partial),
                "approved_with_violation": len(violating),
                **{f"{k}_mean": (v[0] if v else None) for k, v in stats.items()},
                **{f"{k}_std": (v[1] if v else None) for k, v in stats.items()},
            })
            approved_cell = f"{len(approved)}/{len(ar)}" + (f" ({errors} error)" if errors else "")
            md.append(
                f"| **{arm_key}** {arms[arm_key].label} | {approved_cell} | "
                f"{fmt_mean_std(stats['auc'])} | {fmt_mean_std(stats['accuracy'])} | "
                f"{len(evaluated)}/{len(approved)}"
                f"{f' ({len(partial)} partial)' if partial else ''} | "
                f"{len(violating)}/{len(evaluated) if evaluated else 0} | "
                f"{fmt_mean_std(stats['min_di'])} | {fmt_mean_std(stats['max_dpd'])} | "
                f"{fmt_mean_std(stats['max_eod'])} | {fmt_mean_std(stats['protected'], 1)} | "
                f"{fmt_mean_std(stats['violations'], 2)} | "
                f"{fmt_mean_std(stats['reroutes'], 2)} | {fmt_mean_std(stats['retries'], 2)} | "
                f"{fmt_mean_std(stats['seconds'], 1)} |"
            )

        a_runs = sorted((r["seed"], r.get("final_model"), r.get("auc_roc"),
                         r.get("min_disparate_impact"))
                        for r in runs if r["dataset"] == ds and r["arm"] == "A")
        b_runs = sorted((r["seed"], r.get("final_model"), r.get("auc_roc"),
                         r.get("min_disparate_impact"))
                        for r in runs if r["dataset"] == ds and r["arm"] == "B")
        if a_runs and a_runs == b_runs:
            md.append("\nArms A and B produced identical results on every seed: the "
                      "automatic data-quality retry never engaged on this dataset.")
        md.append("")

    for reroute_arm in REROUTE_ARMS:
        c_effects = [e for e in effects if e["arm"] == reroute_arm]
        if not c_effects:
            continue
        md.append(f"### Effect of {REROUTE_ARMS[reroute_arm]} (arm {reroute_arm}): first gate vs approved model\n")
        md.append("| Dataset | Runs rerouted | Violated attributes: first → approved | "
                  "Min disparate impact: first → approved | "
                  "Max parity difference: first → approved | AUC: first → approved | "
                  "Fewer / same / more violations |")
        md.append("|---|---|---|---|---|---|---|")
        for ds in datasets:
            es = [e for e in c_effects if e["dataset"] == ds]
            if not es:
                continue
            rerouted = [e for e in es if e["rerouted"]]
            comparable = [e for e in rerouted
                          if e["violations_first"] is not None
                          and e["violations_final"] is not None]
            fewer = sum(1 for e in comparable if e["violations_final"] < e["violations_first"])
            same = sum(1 for e in comparable if e["violations_final"] == e["violations_first"])
            more = sum(1 for e in comparable if e["violations_final"] > e["violations_first"])

            def arrow(first_key, final_key, digits):
                return (f"{fmt_mean_std(mean_std(e[first_key] for e in rerouted), digits)} → "
                        f"{fmt_mean_std(mean_std(e[final_key] for e in rerouted), digits)}"
                        if rerouted else "no reroutes")

            md.append(
                f"| {dataset_meta.get(ds, {}).get('label', ds)} | {len(rerouted)}/{len(es)} | "
                f"{arrow('violations_first', 'violations_final', 2)} | "
                f"{arrow('min_di_first', 'min_di_final', 3)} | "
                f"{arrow('max_dpd_first', 'max_dpd_final', 3)} | "
                f"{arrow('auc_first', 'auc_final', 3)} | "
                f"{fewer} / {same} / {more} |"
            )
        total_rerouted = sum(1 for e in c_effects if e["rerouted"])
        comparable = [e for e in c_effects if e["rerouted"]
                      and e["violations_first"] is not None
                      and e["violations_final"] is not None]
        md.append(
            f"\nAcross all datasets arm {reroute_arm} rerouted {total_rerouted} of {len(c_effects)} "
            f"runs. Of the {len(comparable)} rerouted runs with fairness measured at both "
            f"gates, the approved model had fewer violated attributes in "
            f"{sum(1 for e in comparable if e['violations_final'] < e['violations_first'])}, "
            f"the same number in "
            f"{sum(1 for e in comparable if e['violations_final'] == e['violations_first'])}, "
            f"and more in "
            f"{sum(1 for e in comparable if e['violations_final'] > e['violations_first'])}."
        )
    return "\n".join(md), summary_rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# First three slots of the reference categorical palette: validated all-pairs in both
# modes (the cap for scatter). Colour follows the arm, never its rank.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": {"A": "#2a78d6", "B": "#2a78d6", "C": "#1baf7a", "D": "#eb6834"},
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": {"A": "#3987e5", "B": "#3987e5", "C": "#199e70", "D": "#d95926"},
    },
}
# Four arms on a scatter, but only three hues: the reference palette validates just
# its first three slots when every pair must be distinguishable. A and B share blue
# because they are the same reviewer (approve at the first gate) and usually coincide
# exactly; shape and fill tell them apart. B is a hollow square, so when it sits on
# A's filled circle both stay visible.
MARKERS = {"A": "o", "B": "s", "C": "^", "D": "D"}
HOLLOW = {"B"}
# (run point, arm mean) areas. Drawn in arm order, so B's smaller square lands on
# top of A's larger circle.
SIZES = {"A": (95, 270), "B": (40, 120), "C": (60, 175), "D": (55, 160)}


def _marker_colours(theme: dict, arm_key: str) -> tuple[str, str]:
    """(face, edge): filled markers get a surface ring, hollow ones a coloured outline."""
    colour = theme["series"][arm_key]
    return (theme["surface"], colour) if arm_key in HOLLOW else (colour, theme["surface"])

# Mean labels sit in a column to the right of every point and connect to their mean
# with a hairline leader. Placing them beside the marker overprinted neighbouring
# seed points. They keep at least MIN_LABEL_GAP apart on the DI axis and stay out of
# LABEL_BLOCKED_BAND, which holds the 0.80 threshold line and its caption.
MIN_LABEL_GAP = 0.055
LABEL_BLOCKED_BAND = (0.765, 0.865)
LABEL_Y_RANGE = (0.04, 1.02)


def place_labels(desired: list[float], gap: float = MIN_LABEL_GAP,
                 blocked: tuple[float, float] = LABEL_BLOCKED_BAND,
                 y_range: tuple[float, float] = LABEL_Y_RANGE) -> list[float]:
    """
    Vertical positions for labels, as close to `desired` as the rules allow.

    Deterministic: labels are placed bottom-up in order of desired position. Each is
    snapped out of the blocked band to its nearer edge, then pushed up until it
    clears the label below it (and out of the band again, if the push lands in it).
    """
    def unblock(y: float) -> float:
        low, high = blocked
        if low < y < high:
            return low if (y - low) <= (high - y) else high
        return y

    placed: dict[int, float] = {}
    previous = None
    for i in sorted(range(len(desired)), key=lambda k: desired[k]):
        y = unblock(min(max(desired[i], y_range[0]), y_range[1]))
        if previous is not None and y < previous + gap:
            y = previous + gap
            if blocked[0] < y < blocked[1]:
                y = blocked[1]
        placed[i] = min(y, y_range[1])
        previous = placed[i]
    return [placed[i] for i in range(len(desired))]
DI_THRESHOLD = 0.80


def _exclusion_summary(arm_runs: list[dict]) -> str:
    reasons: Counter = Counter()
    for r in arm_runs:
        status = r.get("status") or "unknown"
        reasons["no fairness measurement" if status == "approved"
                else status.replace("_", " ")] += 1
    return ", ".join(f"{count} {reason}" for reason, count in reasons.items())


def render_charts(runs: list[dict], out_dir: Path, arms: dict[str, Arm] = ARMS,
                  dataset_meta: Optional[dict] = None) -> dict:
    """
    Small multiples, one panel per dataset: AUC (x) against minimum disparate
    impact across audited attributes (y), one point per approved run, plus the arm
    mean. Rendered once per theme; the CSVs are the table view.

    Honesty rules the figure follows:
      - A run without a fairness measurement is never plotted. Drawing it at 0
        would claim a measurement that was not made (invariant 4).
      - An arm with nothing plotted in a panel is named in that panel, with the
        reason, rather than silently absent.
      - Arms whose means coincide share one label, and are drawn at different
        sizes so neither hides the other.

    Returns {"paths", "plotted", "labels", "notes"} for testing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    dataset_meta = dataset_meta or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(dict.fromkeys(r["dataset"] for r in runs))
    arm_keys = [k for k in arms if any(r["arm"] == k for r in runs)]
    plotted: dict[str, int] = {}
    labels: dict[str, list[str]] = {}
    label_positions: dict[str, list[tuple[str, float]]] = {}
    notes: dict[str, list[str]] = {}
    paths: dict[str, str] = {}

    for theme_name, theme in THEMES.items():
        rc = {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "figure.facecolor": theme["surface"], "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "text.color": theme["ink"], "axes.labelcolor": theme["ink2"],
            "xtick.color": theme["muted"], "ytick.color": theme["muted"],
            "axes.edgecolor": theme["axis"], "axes.linewidth": 1.0,
        }
        with matplotlib.rc_context(rc):
            n = max(1, len(datasets))
            cols = 2 if n > 1 else 1
            rows = math.ceil(n / cols)
            header_in = 1.2
            height = 4.4 * rows + header_in
            fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, height), squeeze=False)

            for i, ds in enumerate(datasets):
                ax = axes[i // cols][i % cols]
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.grid(axis="y", color=theme["grid"], linewidth=1.0, linestyle="-")
                ax.set_axisbelow(True)
                ax.set_ylim(0, 1.05)
                ax.set_title(dataset_meta.get(ds, {}).get("label", ds),
                             loc="left", fontsize=11, color=theme["ink"], pad=10)

                ds_runs = [r for r in runs if r["dataset"] == ds]
                usable = [r for r in ds_runs
                          if r.get("status") == "approved"
                          and r.get("auc_roc") not in (None, "")
                          and r.get("min_disparate_impact") not in (None, "")]
                plotted[ds] = len(usable)
                labels[ds] = []
                label_positions[ds] = []
                # Every excluded run is accounted for, including an arm that is only
                # partly plotted. Arms with the same exclusions share one note.
                by_text: dict[str, list[str]] = {}
                for arm_key in arm_keys:
                    arm_runs = [r for r in ds_runs if r["arm"] == arm_key]
                    excluded = [r for r in arm_runs if r not in usable]
                    if not excluded:
                        continue
                    text = _exclusion_summary(excluded)
                    if len(excluded) < len(arm_runs):
                        text += f" (of {len(arm_runs)})"
                    by_text.setdefault(text, []).append(arm_key)
                missing = [f"{', '.join(keys)}: {text}" for text, keys in by_text.items()]
                notes[ds] = missing

                if not usable:
                    ax.set_xticks([])
                    ax.text(0.5, 0.58, "No approved run with a fairness measurement",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=10, color=theme["ink2"])
                    if missing:
                        ax.text(0.5, 0.48, "\n".join(missing), transform=ax.transAxes,
                                ha="center", va="top", fontsize=8.5, color=theme["muted"])
                    continue

                ax.axhline(DI_THRESHOLD, color=theme["muted"], linewidth=1.0, zorder=1)
                aucs = [float(r["auc_roc"]) for r in usable]
                span = max(aucs) - min(aucs)
                pad = max(span * 0.2, 0.008)
                # Extra room on the right is the label column.
                ax.set_xlim(min(aucs) - pad, max(aucs) + pad * 4.5)
                label_x = max(aucs) + pad * 1.4
                ax.text(ax.get_xlim()[1], DI_THRESHOLD + 0.015, "DI 0.80 threshold",
                        ha="right", va="bottom", fontsize=8, color=theme["muted"])
                ax.set_xlabel("AUC (untouched test rows)", fontsize=9)
                ax.set_ylabel("Minimum disparate impact", fontsize=9)

                means: dict[str, tuple[float, float]] = {}
                for order, arm_key in enumerate(arm_keys):
                    pts = [r for r in usable if r["arm"] == arm_key]
                    if not pts:
                        continue
                    xs = [float(r["auc_roc"]) for r in pts]
                    ys = [float(r["min_disparate_impact"]) for r in pts]
                    run_size, mean_size = SIZES.get(arm_key, (50, 150))
                    face, edge = _marker_colours(theme, arm_key)
                    ax.scatter(xs, ys, s=run_size, marker=MARKERS[arm_key],
                               facecolors=face, edgecolors=edge, linewidths=1.5,
                               zorder=3 + order)
                    means[arm_key] = (statistics.fmean(xs), statistics.fmean(ys))
                    ax.scatter([means[arm_key][0]], [means[arm_key][1]], s=mean_size,
                               marker=MARKERS[arm_key], facecolors=face, edgecolors=edge,
                               linewidths=2, zorder=10 + order)

                x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
                groups: list[dict] = []
                for arm_key, (mx, my) in means.items():
                    for group in groups:
                        gx, gy = group["xy"]
                        if abs(mx - gx) <= 0.01 * x_span and abs(my - gy) <= 0.01:
                            group["arms"].append(arm_key)
                            break
                    else:
                        groups.append({"xy": (mx, my), "arms": [arm_key]})
                label_ys = place_labels([g["xy"][1] for g in groups])
                for group, label_y in zip(groups, label_ys):
                    text = ", ".join(group["arms"]) + " mean"
                    mx, my = group["xy"]
                    # Leader drawn beneath the markers, so points sit on top of it.
                    ax.plot([mx, label_x], [my, label_y], color=theme["muted"],
                            linewidth=0.8, solid_capstyle="round", zorder=2)
                    ax.text(label_x + pad * 0.15, label_y, text, ha="left", va="center",
                            fontsize=8, color=theme["ink2"], zorder=30)
                    labels[ds].append(text)
                    label_positions[ds].append((text, label_y))

                if missing:
                    ax.text(0.0, -0.21, "Not plotted — " + "; ".join(missing),
                            transform=ax.transAxes, ha="left", va="top",
                            fontsize=7.5, color=theme["muted"])

            for j in range(len(datasets), rows * cols):
                axes[j // cols][j % cols].set_visible(False)

            fig.text(0.012, 1 - 0.14 / height,
                     "Minimum disparate impact against AUC, by governance arm",
                     ha="left", va="top", fontsize=13, color=theme["ink"])
            fig.text(0.012, 1 - 0.48 / height,
                     "Small points: one approved run per split seed. Large points: arm "
                     "mean. Only approved runs with a fairness measurement are plotted; "
                     "exclusions are noted under each panel.",
                     ha="left", va="top", fontsize=8.5, color=theme["ink2"])
            handles = [Line2D([0], [0], marker=MARKERS[k], linestyle="",
                              markersize=8, markerfacecolor=_marker_colours(theme, k)[0],
                              markeredgecolor=_marker_colours(theme, k)[1],
                              markeredgewidth=1.5,
                              label=f"{k}  {arms[k].label}") for k in arm_keys]
            fig.legend(handles=handles, loc="upper left", ncol=len(handles),
                       frameon=False, fontsize=9, labelcolor=theme["ink2"],
                       bbox_to_anchor=(0.006, 1 - 0.74 / height), borderaxespad=0.0)
            fig.tight_layout(rect=(0, 0, 1, 1 - header_in / height))

            suffix = "" if theme_name == "light" else "_dark"
            path = out_dir / f"fairness_vs_auc{suffix}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths[theme_name] = str(path)

    return {"paths": paths, "plotted": plotted, "labels": labels, "notes": notes,
            "label_positions": label_positions}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _csv_value(v):
    return "" if v is None else v


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fields})


def _git_commit() -> Optional[str]:
    with contextlib.suppress(Exception):
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    return None


def _git_dirty() -> Optional[bool]:
    """
    True if tracked files other than the committed results differ from HEAD.

    git_commit alone is misleading when results come from uncommitted code: the
    manifest would name a commit that does not contain the code that produced it.
    The results directory is excluded: this check runs after the harness has written
    runs.csv and the trajectory, and counting its own outputs made every run look
    dirty.
    """
    with contextlib.suppress(Exception):
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--",
                              ".", ":(exclude)experiments/results"],
                             cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return bool(out.stdout.strip())
    return None


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """
    A --quick smoke test must never overwrite the committed results. Unless an
    output directory was given explicitly, quick runs go to experiments/results_quick.
    """
    if args.quick and Path(args.out).resolve() == DEFAULT_OUT_DIR.resolve():
        return QUICK_OUT_DIR
    return Path(args.out)


_INT_FIELDS = {
    "seed", "gates", "retries", "reroutes", "eda_findings", "train_rows", "test_rows",
    "split_seed_recorded", "planner_calls", "planner_cache_hits",
    "planner_tokens_recorded", "dataset_rows", "gate_index", "n_violations",
    "protected_attributes_evaluated",
}
_FLOAT_FIELDS = {
    "accuracy", "f1", "auc_roc", "min_disparate_impact", "max_parity_difference",
    "wall_clock_s", "disparate_impact", "demographic_parity_difference",
    "max_equal_opportunity_difference", "equal_opportunity_difference",
}
_BOOL_FIELDS = {"fairness_evaluated", "overall_fairness_passed", "violation", "protected"}


def _coerce(field: str, value):
    """Restore the type a CSV cell had before it was written. Blank means None."""
    if value is None or value == "":
        return None
    if field in _BOOL_FIELDS:
        return {"True": True, "False": False}.get(value)
    if field in _INT_FIELDS:
        return int(float(value))
    if field in _FLOAT_FIELDS:
        return float(value)
    return value


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return [{k: _coerce(k, v) for k, v in row.items()} for row in csv.DictReader(fh)]


def load_results(out_dir: Path) -> tuple[list[dict], list[dict], dict]:
    """Read back runs.csv, fairness_trajectory.csv and manifest.json from a results dir."""
    out_dir = Path(out_dir)
    runs = _read_csv(out_dir / "runs.csv")
    trajectory_path = out_dir / "fairness_trajectory.csv"
    trajectory = _read_csv(trajectory_path) if trajectory_path.is_file() else []
    manifest_path = out_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else {})
    return runs, trajectory, manifest


def write_report(out_dir: Path, runs: list[dict], trajectory: list[dict],
                 manifest: dict, regenerated: bool = False) -> dict:
    """
    Write summary.csv, summary.md and the figure from recorded runs.

    Separate from running the pipelines, so presentation can be rebuilt from the
    saved CSVs (--summarise-only) in seconds instead of re-running every pipeline.
    """
    out_dir = Path(out_dir)
    # Notes are explanatory prose from DATASETS, not measurements: take the current
    # wording so a correction reaches --summarise-only. Rows and hashes stay as recorded.
    dataset_meta = {ds: {**meta, "note": DATASETS[ds].note} if ds in DATASETS else meta
                    for ds, meta in (manifest.get("datasets") or {}).items()}
    arm_keys = ([k for k in (manifest.get("arms") or {}) if k in ARMS]
                or [k for k in ARMS if any(r["arm"] == k for r in runs)])
    arms = {k: ARMS[k] for k in arm_keys}

    markdown, summary_rows = summarise(runs, trajectory, arms, dataset_meta)
    summary_fields = list(summary_rows[0].keys()) if summary_rows else ["dataset"]
    _write_csv(out_dir / "summary.csv", summary_fields, summary_rows)
    charts = render_charts(runs, out_dir, arms, dataset_meta)

    seeds = manifest.get("seeds") or sorted({r["seed"] for r in runs})
    planner_models = manifest.get("planner_models") or []
    arm_rows = "\n".join(f"| **{k}** | {ARMS[k].label} | {ARMS[k].description} |"
                         for k in arm_keys)
    regenerated_note = (
        f"\n> Tables and figure rebuilt from the recorded CSVs at "
        f"{datetime.now(timezone.utc).isoformat()}."
        if regenerated else "")
    quick_note = " · **quick smoke test, not results**" if manifest.get("quick") else ""

    header = f"""# Governance evaluation — results

> Generated by `experiments/run_governance_eval.py` ({manifest.get('mode', '?')} mode) at
> {manifest.get('generated_at', '?')}. Do not edit by hand; re-run the script.{regenerated_note}

## Setup

| Arm | Name | Scripted reviewer |
|---|---|---|
{arm_rows}

- **Seeds:** {len(seeds)} train/test split seeds {list(seeds)}. Model seeds are fixed, so
  the spread reflects the data partition only.
- **Planner:** {', '.join(planner_models) or 'n/a'}, held fixed — each distinct prompt
  was answered once and replayed for every seed and arm
  ({manifest.get('planner_cache_hits', '?')} of {manifest.get('planner_calls', '?')} planner
  calls served from `experiments/planner_cache/`).
- **Rejection cap:** {manifest.get('max_human_reroutes', '?')} human reroutes per run.
- **Gate decisions use validation rows; results are reported on test rows.** The Data
  Agent splits train / validation / test (64 / 16 / 20). The leaderboard, the fairness
  audit and the scripted reviewer's decisions (including every reroute and mitigation)
  use the validation rows; the per-gate trajectory records those. The approved model is
  then scored once on the untouched test rows, and every AUC, accuracy and fairness
  column in the tables below comes from that final evaluation.
- **The verdict covers protected attributes only**, and so do the fairness columns:
  **min disparate impact** is the lowest DI across the protected attributes audited for
  that run, **max parity difference** the largest demographic parity difference, and
  **violated attributes** the protected attributes in violation. Attributes the planner
  proposes that are not protected (occupation, job, education) are audited but advisory;
  they appear in `fairness_trajectory.csv` and `n_advisory_violations`. An attribute is
  violated when DI < 0.80 **or** parity difference > 0.10, so a run can clear the DI
  threshold and still carry violations. Runs where fairness was not evaluated are
  excluded from every fairness column, never counted as fair.
- **Groups** come from raw uploaded values; age is banded (<25, 25-59, 60+); groups
  with fewer than {manifest.get('min_group_size', 30)} evaluation rows are excluded from
  the comparison; protected attributes present in the data are audited even when the
  planner does not propose them. **Max equal-opportunity difference** is the largest
  true-positive-rate gap between groups; it is reported, but it does not affect the verdict.
- Runs: {manifest.get('runs', len(runs))} · errors: {manifest.get('errors', '?')} · wall
  clock: {manifest.get('wall_clock_s', '?')} s{quick_note}.

## Results by dataset

"""
    figure = """
## Figure

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fairness_vs_auc_dark.png">
  <img alt="Minimum disparate impact against AUC for each governance arm, one panel per dataset" src="fairness_vs_auc.png">
</picture>

Table view: `runs.csv` (one row per run), `fairness_trajectory.csv` (every attribute at
every gate), `summary.csv` (the tables above).
"""
    (out_dir / "summary.md").write_text(header + markdown + "\n" + figure, encoding="utf-8")
    return charts


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("replay", "record"), default="replay",
                    help="replay (default) never calls the LLM; record calls it on a cache miss")
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help=f"comma-separated subset of: {', '.join(DATASETS)}")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset of A,B,C,D")
    ap.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS),
                    help=f"number of split seeds, taken from {DEFAULT_SEEDS}")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="subsample each dataset to at most this many rows")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test: {QUICK_SEEDS} seeds, at most {QUICK_MAX_ROWS} rows; "
                         f"writes to experiments/results_quick")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--data-home", default=str(DEFAULT_DATA_HOME))
    ap.add_argument("--verbose", action="store_true", help="show pipeline output")
    ap.add_argument("--summarise-only", action="store_true",
                    help="rebuild summary.md, summary.csv and the figure from an existing "
                         "results directory, without running any pipeline")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = resolve_output_dir(args)

    if args.summarise_only:
        runs, trajectory, manifest = load_results(out_dir)
        charts = write_report(out_dir, runs, trajectory, manifest, regenerated=True)
        print(f"Rebuilt {out_dir / 'summary.md'} from {len(runs)} recorded run(s); "
              f"figure points plotted per dataset: {charts['plotted']}")
        return 0
    dataset_keys = [d.strip() for d in args.datasets.split(",") if d.strip()]
    arm_keys = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    unknown = [d for d in dataset_keys if d not in DATASETS] + \
              [a for a in arm_keys if a not in ARMS]
    if unknown:
        print(f"Unknown dataset/arm: {unknown}", file=sys.stderr)
        return 2

    n_seeds = QUICK_SEEDS if args.quick else args.seeds
    seeds = list(DEFAULT_SEEDS[:max(1, min(n_seeds, len(DEFAULT_SEEDS)))])
    max_rows = QUICK_MAX_ROWS if (args.quick and not args.max_rows) else args.max_rows
    out_dir.mkdir(parents=True, exist_ok=True)

    planner_agent.configure_planner_cache(mode=args.mode, cache_dir=args.cache_dir)
    runs: list[dict] = []
    trajectory: list[dict] = []
    dataset_meta: dict[str, dict] = {}
    total = len(dataset_keys) * len(arm_keys) * len(seeds)
    done = 0
    started = time.perf_counter()

    try:
        for key in dataset_keys:
            spec = DATASETS[key]
            print(f"\nLoading {spec.label} (OpenML {spec.openml_id})...", flush=True)
            dataset = load_dataset(spec, Path(args.data_home), max_rows=max_rows)
            dataset_meta[key] = {"label": spec.label, "openml_id": spec.openml_id,
                                 "target": spec.target, "rows": dataset.rows,
                                 "csv_sha256": dataset.csv_sha256, "note": spec.note}
            print(f"  {dataset.rows:,} rows, CSV SHA-256 {dataset.csv_sha256[:12]}", flush=True)

            for arm_key in arm_keys:
                for seed in seeds:
                    row, traj = run_single(dataset, ARMS[arm_key], seed,
                                           verbose=args.verbose)
                    runs.append(row)
                    trajectory.extend(traj)
                    done += 1
                    auc = row.get("auc_roc")
                    min_di = row.get("min_disparate_impact")
                    print(f"  [{done}/{total}] {key} arm {arm_key} seed {seed}: "
                          f"{row.get('status')}"
                          f" | AUC {auc if auc is not None else 'n/a'}"
                          f" | min DI {min_di if min_di is not None else 'n/a'}"
                          f" | gates {row.get('gates', 0)}"
                          f" | planner {row.get('planner_calls')} call(s), "
                          f"{row.get('planner_cache_hits')} cached"
                          f" | {row.get('wall_clock_s')}s"
                          + (f" | ERROR {row.get('error')}" if row.get("error") else ""),
                          flush=True)
                    # Written after every run so a long evaluation keeps partial results.
                    _write_csv(out_dir / "runs.csv", RUN_FIELDS, runs)
                    _write_csv(out_dir / "fairness_trajectory.csv",
                               TRAJECTORY_FIELDS, trajectory)
    except planner_agent.PlannerCacheMiss as exc:
        print(f"\nREPLAY CACHE MISS: {exc}", file=sys.stderr)
        return 3
    finally:
        planner_agent.configure_planner_cache("off")

    planner_models = sorted({r["planner_model"] for r in runs if r.get("planner_model")})
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "quick": bool(args.quick),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seeds": seeds,
        "max_rows": max_rows,
        "arms": {k: {"label": ARMS[k].label, "description": ARMS[k].description,
                     "max_retries": ARMS[k].max_retries} for k in arm_keys},
        "max_human_reroutes": pipeline_graph.MAX_HUMAN_REROUTES,
        "min_group_size": fairness_agent.MIN_GROUP_SIZE,
        # Each arm runs under policy.yaml with its own max_retries; this is the base file.
        "policy": {k: load_policy()[k] for k in ("version", "sha256", "source")},
        "datasets": dataset_meta,
        "planner_models": planner_models,
        "planner_calls": sum(r.get("planner_calls") or 0 for r in runs),
        "planner_cache_hits": sum(r.get("planner_cache_hits") or 0 for r in runs),
        "runs": len(runs),
        "errors": sum(1 for r in runs if r.get("status") == "error"),
        "wall_clock_s": round(time.perf_counter() - started, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    charts = write_report(out_dir, runs, trajectory, manifest)

    print(f"\nWrote {out_dir / 'summary.md'}")
    print(f"Figure points plotted per dataset: {charts['plotted']}")
    print(f"Planner: {manifest['planner_calls']} call(s), "
          f"{manifest['planner_cache_hits']} served from cache")
    return 1 if manifest["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
