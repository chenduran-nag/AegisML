"""
pipeline_graph.py
=================
LangGraph StateGraph wiring Planner, Data, Training, Fairness, and Human Approval.

Graph structure:
    START
      │
      ▼
  data_analysis_node    ← profiles raw data, derives routed EDA findings
      │
      ▼
  planner_node          ← calls plan_pipeline() (LLM)
      │
      ▼
  data_agent_node       ← calls run_data_agent() (deterministic)
      │
      ▼
  [route_after_data_agent]  ← conditional edge
      │
      ├─ quality OK  ► training_node ── no model trained ──► mark_training_failure ──► END
      │                       │
      │                       ▼
      │                 fairness_node
      │                       │
      │                       ▼
      │                 human_approval_node  ← calls interrupt() (pauses here)
      │                       │
      │                       ▼
      │                 [route_after_human_approval]  ← conditional edge
      │                       │
      │                       ├─ "approve", fairness passed  ►►►►► audit_log_node ► END
      │                       ├─ "approve", violation, 1st   ►►►►► record_first_approval
      │                       │                                    ► back to the gate
      │                       ├─ "approve", same reviewer /
      │                       │   no reviewer id             ►►►►► reject_signoff ► the gate
      │                       ├─ "approve", 2nd reviewer     ►►►►► audit_log_node ► END
      │                       ├─ "reject_data_quality"       ►►►►► planner_node
      │                       ├─ "reject_and_mitigate"       ►►►►► mitigation_node
      │                       └─ "reject_model_or_fairness"  ►►►►► training_node
      │
      ├─ quality FAIL, retry_count < MAX_RETRIES ──► increment_retry node ──► planner_node
      │
      └─ quality FAIL, retry_count >= MAX_RETRIES ─► mark_cap_failure node ──► END

CHECKPOINTER RATIONALE (SqliteSaver):
  Switched from MemorySaver to SqliteSaver (persisted local SQLite file).
  Human approval pauses may last minutes, hours, or days. MemorySaver stores checkpoints
  in-memory, so if the process restarts during a human pause, the state is lost.
  SqliteSaver guarantees thread_id state persistence across process and server restarts.
  All state data (pickle DataFrames, joblib models, dicts) remain fully SqliteSaver compatible.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from graph_state import (
    PipelineState,
    df_to_bytes,
    bytes_to_df,
    model_to_bytes,
    bytes_to_model,
)
from planner_agent import plan_pipeline
from data_agent import run_data_agent
from training_agent import evaluate_model, run_training_agent
from fairness_agent import run_fairness_agent
from policy import default_policy, policy_value
from mitigation import (
    INTERSECTION_SEPARATOR,
    METHOD as MITIGATION_METHOD,
    choose_attribute as choose_mitigation_attribute,
    row_weights,
    snapshot as mitigation_snapshot,
    summary as mitigation_summary,
)
from audit_log import DEFAULT_AUDIT_DB, log_audit_event
from compliance_artifacts import ARTIFACT_EVENT_TYPE, generate_artifacts
from data_analysis_agent import analyze_raw_dataset
from eda_insights import build_eda_linkage, derive_eda_findings, findings_for_route

MAX_RETRIES = 2          # maximum planner→data_agent auto-retries
MAX_HUMAN_REROUTES = 2   # maximum human rejection reroutes before capping

# The policy the code enforces when a run carries none in state (tests, direct calls).
# A run started by the server or the evaluation carries its own validated policy.
DEFAULT_POLICY = default_policy()


def _policy_value(state: PipelineState, section: str, key: str, fallback):
    """A limit from the run's policy, or `fallback` (read at call time) without one."""
    return policy_value(state.get("policy"), section, key, fallback)


def _policy_section(state: PipelineState, section: str) -> dict:
    return dict((state.get("policy") or {}).get(section) or {})


def _fairness_limits(state: PipelineState) -> dict:
    """Keyword arguments for run_fairness_agent from the run's policy (none without one)."""
    fairness = _policy_section(state, "fairness")
    mapping = {"min_group_size": "min_group_size",
               "disparate_impact_threshold": "disparate_impact_threshold",
               "demographic_parity_difference_threshold": "parity_difference_threshold"}
    return {arg: fairness[key] for key, arg in mapping.items() if key in fairness}


def _fairness_not_measured_reason(state: PipelineState) -> str | None:
    """
    Why this run's fairness verdict does not support an approval, or None.

    A classification model whose fairness verdict is None — NOT EVALUATED, or NOT
    FULLY EVALUATED — cannot be approved while block_approval_when_fairness_not_evaluated
    is set (the default). The evaluation showed why: on German Credit no protected
    attribute could be audited at the gate, and reviewers approved blind. Regression
    has no fairness definition and is exempt.
    """
    block = _policy_value(state, "governance", "block_approval_when_fairness_not_evaluated",
                          DEFAULT_POLICY["governance"]["block_approval_when_fairness_not_evaluated"])
    if not block or state.get("task_type") != "classification":
        return None
    fairness = state.get("fairness_result") or {}
    if fairness.get("overall_fairness_passed") is not None:
        return None
    partial = fairness.get("fairness_coverage") == "partial"
    unaudited = [p.get("attribute") for p in fairness.get("protected_attributes_unaudited") or []]
    return (
        f"fairness is {'NOT FULLY EVALUATED' if partial else 'NOT EVALUATED'}"
        + (f" (protected attribute(s) not audited: {', '.join(unaudited)})" if unaudited else
           " (no protected attribute could be audited)")
        + ". The governance policy blocks approving a model whose fairness was not measured. "
          "Reject to try another model or to fix the data, or start a new run that "
          "declares the protected attributes."
    )


def _approval_block_reason(state: PipelineState) -> str | None:
    """
    Why approving this model is refused outright, or None if some approval is possible.

    Unmeasured fairness blocks the approval unless the policy lets two reviewers sign
    it off instead (`dual_signoff_can_override_approval_block`), in which case this
    returns None and `_dual_signoff_reason` takes over: the route is open, but not to
    one reviewer alone.
    """
    reason = _fairness_not_measured_reason(state)
    if reason and _policy_value(state, "governance", "dual_signoff_can_override_approval_block",
                                DEFAULT_POLICY["governance"]["dual_signoff_can_override_approval_block"]):
        return None
    return reason


def _dual_signoff_reason(state: PipelineState) -> str | None:
    """
    Why approving this model takes two different reviewers, or None if one suffices.

    Approving a model the pipeline itself found to discriminate is the decision most
    worth slowing down: the evaluation approved 60 violating models, every one of them
    on a single click. Regression never reaches the first branch — it has no verdict.
    """
    fairness = state.get("fairness_result") or {}
    if fairness.get("overall_fairness_passed") is False and _policy_value(
            state, "governance", "require_dual_signoff_for_violating_approval",
            DEFAULT_POLICY["governance"]["require_dual_signoff_for_violating_approval"]):
        return ("the fairness audit recorded a violation on a protected attribute, so "
                "the policy requires a second, different reviewer to sign off this "
                "approval")
    if _fairness_not_measured_reason(state) and _policy_value(
            state, "governance", "dual_signoff_can_override_approval_block",
            DEFAULT_POLICY["governance"]["dual_signoff_can_override_approval_block"]):
        return ("fairness was not measured for this run, and the policy allows two "
                "different reviewers to sign that off in place of refusing it")
    return None


def _reviewer_from_resume(raw: object) -> dict:
    """
    The identity attached to a resume payload, normalised — never assumed present.

    `reviewer_authenticated` is set by the server when the caller's token matched the
    reviewer roster. Nothing else may set it: a caller that simply claims a role is
    recorded as unverified (see reviewers.py for the threat model).
    """
    if not isinstance(raw, dict):
        return {"reviewer_id": None, "reviewer_role": None, "authenticated": False}

    def text(value):
        cleaned = str(value).strip() if value is not None else ""
        return cleaned or None

    return {
        "reviewer_id": text(raw.get("reviewer_id")),
        "reviewer_role": text(raw.get("reviewer_role")),
        "authenticated": bool(raw.get("reviewer_authenticated")),
    }


def _reviewer_label(reviewer: dict | None) -> str:
    """How a reviewer is named in audit summaries, including when they are not."""
    reviewer = reviewer or {}
    if not reviewer.get("reviewer_id"):
        return "an unidentified reviewer (no reviewer id supplied)"
    return (f"{reviewer['reviewer_id']} ({reviewer.get('reviewer_role') or 'role not stated'}, "
            f"{'authenticated' if reviewer.get('authenticated') else 'UNVERIFIED identity'})")


def _signoff_shortfall(state: PipelineState) -> str | None:
    """
    Why the approval in hand does not complete a dual sign-off, or None if it does.

    Covers both halves of "two people": an approval with no name at all, and the same
    reviewer approving twice.
    """
    reviewer_id = (state.get("current_reviewer") or {}).get("reviewer_id")
    first = state.get("first_approval") or {}
    if not reviewer_id:
        return ("this approval carries no reviewer id, and the policy requires two named "
                "reviewers — submit the decision with a reviewer id")
    if first and reviewer_id == first.get("reviewer_id"):
        return (f"'{reviewer_id}' already gave the first approval; the second sign-off has "
                "to come from a different reviewer")
    return None


def _approvers(state: PipelineState) -> list[dict]:
    """The identities that approved this run, in order. Identity only, no feedback."""
    return [
        {k: record.get(k) for k in
         ("reviewer_id", "reviewer_role", "authenticated", "stage", "timestamp")}
        for record in (state.get("reviewer_decisions") or [])
        if record.get("decision") == "approve"
    ]

# Directory approved models are serialised into by audit_log_node.
SAVED_MODELS_DIR = "saved_models"

# Root directory for generated compliance artifacts (one subdirectory per run).
ARTIFACTS_DIR = "artifacts"

# Audit database this graph writes to. Passed explicitly to every
# log_audit_event() call and to generate_artifacts(), rather than relying on the
# default bound into log_audit_event's signature — the artifact generator has to
# READ the same log back to hash-chain its digests, so "which audit db" must be a
# single knob both paths agree on (and one a test can repoint).
AUDIT_DB_PATH = DEFAULT_AUDIT_DB


def _get_run_id(config: RunnableConfig | None) -> str:
    if config and "configurable" in config:
        return config["configurable"].get("thread_id", "unknown_run")
    return "unknown_run"


# ---------------------------------------------------------------------------
# Node: Data Analysis
# ---------------------------------------------------------------------------


def data_analysis_node(state: PipelineState, config: RunnableConfig) -> dict:
    """
    First node: profile the raw data and derive routed findings.

    This used to run in server.py before graph.invoke(), which meant it was not
    audited, not part of the checkpointed run, and not executed at all when the
    pipeline was driven from a script or a test. As a node it is all three.

    Runs exactly once per run — every loop re-enters at the planner or at
    training, never here — so findings stay stable across reroutes.

    A failure here must not stop the pipeline: exploratory analysis informs later
    stages but is not required by them. A failure is recorded in the audit log
    instead, so its absence is itself visible to a reviewer.
    """
    run_id = _get_run_id(config)
    df = bytes_to_df(state["df_bytes"])
    target_column = state["target_column"]
    task_type = state["task_type"]

    # Record the policy governing this run before anything acts on it.
    if state.get("policy"):
        log_audit_event(
            run_id=run_id,
            db_path=AUDIT_DB_PATH,
            event_type="policy_applied",
            event_source="automated",
            summary=(f"Governance policy {state.get('policy_version')} applied "
                     f"(SHA-256 {str(state.get('policy_sha256'))[:12]}…)"),
            details={"policy_version": state.get("policy_version"),
                     "policy_sha256": state.get("policy_sha256"),
                     "policy": state.get("policy")},
        )

    try:
        eda_report = analyze_raw_dataset(df, target_column)
    except Exception as exc:
        print(f"[data_analysis_node] Profiling failed: {type(exc).__name__}: {exc}")
        eda_report = {"error": f"{type(exc).__name__}: {exc}"}

    findings_error = None
    try:
        findings = derive_eda_findings(
            df, target_column, task_type,
            declared_protected=state.get("declared_protected_attributes"),
        )
    except Exception as exc:
        findings = []
        findings_error = f"{type(exc).__name__}: {exc}"
        print(f"[data_analysis_node] Finding derivation failed: {findings_error}")

    counts = {route: len(findings_for_route(findings, route))
              for route in ("data_agent", "planner", "reviewer")}
    print(
        f"[data_analysis_node] {len(findings)} finding(s): "
        f"data_agent={counts['data_agent']} planner={counts['planner']} "
        f"reviewer={counts['reviewer']}"
    )

    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="data_analysis_run",
        event_source="automated",
        summary=(
            f"Data Analysis profiled {len(df):,} rows x {df.shape[1]} columns and "
            f"derived {len(findings)} finding(s): {counts['data_agent']} for the "
            f"Data Agent, {counts['planner']} for the Planner, {counts['reviewer']} "
            f"held for the reviewer"
            + (f" (finding derivation FAILED: {findings_error})" if findings_error else "")
        ),
        # Aggregate-only: the profiling summary and the findings, never per-row data.
        details={
            "summary": eda_report.get("summary", {}),
            "findings": findings,
            "error": eda_report.get("error") or findings_error,
        },
    )

    return {"eda_report": eda_report, "eda_findings": findings}


# ---------------------------------------------------------------------------
# Node: Planner
# ---------------------------------------------------------------------------


def planner_node(state: PipelineState, config: RunnableConfig) -> dict:
    """
    Deserialises the raw DataFrame from state, calls plan_pipeline(),
    and writes the resulting plan back to state.
    On retries/reroutes, passes last_failure_reason as failure_context.
    """
    run_id = _get_run_id(config)
    df = bytes_to_df(state["df_bytes"])
    retry_count = state.get("retry_count", 0)
    reroute_count = state.get("rejection_reroute_count", 0)
    failure_context = None

    if (retry_count > 0 or reroute_count > 0) and state.get("last_failure_reason") is not None:
        failure_context = state["last_failure_reason"]
        print(
            f"[planner_node] Retry/Reroute pass — passing failure context: "
            f"source={failure_context.get('source', 'data_agent')}"
        )
    else:
        print("[planner_node] First run — calling plan_pipeline()...")

    planner_meta: dict = {}
    plan = plan_pipeline(
        df=df,
        target_column=state["target_column"],
        task_type=state["task_type"],
        failure_context=failure_context,
        business_objective=state.get("business_objective", "") or "",
        human_feedback=state.get("human_feedback", "") or "",
        meta_out=planner_meta,
        eda_findings=state.get("eda_findings") or [],
    )

    models_str = ", ".join(plan.get("recommended_models", []))
    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="planner_run",
        event_source="automated",
        summary=f"Planner Agent generated plan ({len(plan.get('recommended_models', []))} recommended models: {models_str})",
        details=plan,
    )

    # planner_meta records which model produced the plan and a hash of the exact
    # prompt sent. The AI Bill of Materials needs it, and on a reroute it evidences
    # that the revised plan came from a genuinely different prompt.
    return {"plan": plan, "planner_meta": planner_meta}


# ---------------------------------------------------------------------------
# Node: Data Agent
# ---------------------------------------------------------------------------


def data_agent_node(state: PipelineState, config: RunnableConfig) -> dict:
    run_id = _get_run_id(config)
    df = bytes_to_df(state["df_bytes"])
    print("[data_agent_node] Running data cleaning pipeline...")

    result = run_data_agent(
        df=df,
        plan=state["plan"],
        target_column=state["target_column"],
        task_type=state["task_type"],
        eda_findings=state.get("eda_findings") or [],
        split_seed=state.get("split_seed"),
        thresholds=_policy_section(state, "data"),
    )

    cleaned_df = result.pop("cleaned_df")
    cleaned_df_bytes = df_to_bytes(cleaned_df)

    # Lift the split out of `result` before it reaches the audit log: these are
    # per-row index lists (tens of thousands of ints on a real dataset) and would
    # swamp every audit entry. They live in their own state key instead.
    split_index = {
        "train": result.pop("train_index"),
        "validation": result.pop("validation_index", None),
        "test": result.pop("test_index"),
    }

    passed = result["quality_check_passed"]
    report = result["quality_report"]
    print(
        f"[data_agent_node] quality_check_passed={passed} | "
        f"missing_pct={report['missing_pct_after_cleaning']}% | "
        f"rows_dropped={report['rows_dropped']} | "
        f"cols_dropped={report['columns_dropped']}"
    )

    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="data_agent_run",
        event_source="automated",
        summary=f"Data Agent cleaned dataset (quality_passed={passed}, missing_pct={report.get('missing_pct_after_cleaning')}%, rows_dropped={report.get('rows_dropped')})",
        details=result,
    )

    return {
        "data_agent_result": result,
        "cleaned_df_bytes": cleaned_df_bytes,
        "split_index": split_index,
    }


# ---------------------------------------------------------------------------
# Conditional edge: route after data_agent_node
# ---------------------------------------------------------------------------


def route_after_data_agent(state: PipelineState) -> str:
    result = state.get("data_agent_result", {})
    passed = result.get("quality_check_passed", True)
    retry_count = state.get("retry_count", 0)

    if passed:
        print(f"[router] Quality PASSED → training_node (retries used: {retry_count})")
        return "training_node"

    max_retries = _policy_value(state, "governance", "max_retries", MAX_RETRIES)
    if retry_count < max_retries:
        print(
            f"[router] Quality FAILED → retry {retry_count + 1}/{max_retries} "
            "— routing to planner"
        )
        return "planner_node"

    print(
        f"[router] Quality FAILED after {retry_count} retries → "
        "cap reached, routing to END with unresolved_quality_issue=True"
    )
    return "cap_failure"


# ---------------------------------------------------------------------------
# Node: Training Agent
# ---------------------------------------------------------------------------


def _gate_index(split: dict) -> list | None:
    """
    The rows every gate decision is based on: the validation split.

    Leaderboard ranking, the fairness audit and the reviewer's decisions all look at
    these rows, so the test rows stay untouched until audit_log_node scores the
    approved model on them. State recorded before the validation split existed falls
    back to the test rows.
    """
    return split.get("validation") or split.get("test")


def _gate_label(split: dict) -> str:
    return "validation" if split.get("validation") else "test"


def training_node(state: PipelineState, config: RunnableConfig) -> dict:
    run_id = _get_run_id(config)
    cleaned_df = bytes_to_df(state["cleaned_df_bytes"])
    recommended_models = list(state["plan"].get("recommended_models", []))
    rejected = [m.lower() for m in (state.get("rejected_models") or [])]

    if rejected:
        filtered_models = [m for m in recommended_models if m.lower() not in rejected]
        print(
            f"[training_node] Filtering out rejected models {rejected} from recommended {recommended_models} "
            f"→ remaining: {filtered_models}"
        )
        if filtered_models:
            recommended_models = filtered_models
        else:
            print(f"[training_node] All recommended models {recommended_models} were rejected! Fallback to unrejected models...")
            all_known = ["LogisticRegression", "RandomForest", "XGBoost", "GradientBoosting", "Ridge", "Lasso"]
            recommended_models = [m for m in all_known if m.lower() not in rejected]

    # Policy: only allowed models may be trained, whatever the planner recommends.
    allowed = _policy_value(state, "training", "allowed_models", None)
    if allowed:
        def _norm(name):
            return str(name).lower().replace("-", "").replace("_", "").replace(" ", "")
        allowed_keys = {_norm(a) for a in allowed}
        kept = [m for m in recommended_models if _norm(m) in allowed_keys]
        if not kept:
            kept = [a for a in allowed if a.lower() not in rejected]
        if kept != recommended_models:
            print(f"[training_node] Policy allowed_models {allowed}: training {kept} "
                  f"instead of {recommended_models}")
        recommended_models = kept

    print(f"[training_node] Training models: {recommended_models}")

    split = state.get("split_index") or {}

    # After a reject_and_mitigate decision, every later training pass is reweighted.
    # Weights are rebuilt from the logged cells' rule on the train rows only.
    sample_weight = None
    mitigated = (state.get("mitigation") or {}).get("attributes") or []
    if mitigated:
        sample_weight, _ = row_weights(
            bytes_to_df(state["df_bytes"]), mitigated, split.get("train"),
            cleaned_df[state["target_column"]],
        )
        print(f"[training_node] Reweighing on {mitigated}")

    try:
        raw = run_training_agent(
            cleaned_df=cleaned_df,
            target_column=state["target_column"],
            task_type=state["task_type"],
            recommended_models=recommended_models,
            train_index=split.get("train"),
            test_index=_gate_index(split),
            sample_weight=sample_weight,
            eval_label=_gate_label(split),
        )
    except RuntimeError as exc:
        print(f"[training_node] Training failed entirely: {exc}")
        err_res = {"error": str(exc), "leaderboard": []}
        log_audit_event(
            run_id=run_id,
            db_path=AUDIT_DB_PATH,
            event_type="training_run",
            event_source="automated",
            summary=f"Training Agent failed: {exc}",
            details=err_res,
        )
        return {
            "training_result": err_res,
            "selected_model_bytes": None,
        }

    fitted_model = raw.pop("_fitted_model")
    selected_model_bytes = model_to_bytes(fitted_model)

    selected_name = raw["selected_model_name"]
    metrics = raw.get("selected_model_metrics", {})
    print(
        f"[training_node] Selected: '{selected_name}' — "
        f"metrics={metrics}"
    )

    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="training_run",
        event_source="automated",
        summary=f"Training Agent evaluated {len(raw.get('leaderboard', []))} models; selected '{selected_name}' (AUC={metrics.get('auc_roc')})",
        details=raw,
    )

    return {
        "training_result": raw,
        "selected_model_bytes": selected_model_bytes,
    }


# ---------------------------------------------------------------------------
# Conditional edge: route after training_node
# ---------------------------------------------------------------------------


def route_after_training(state: PipelineState) -> str:
    """
    Continue to fairness and the gate only if training produced a model.

    Without this edge, a run whose remaining candidates all failed to train still
    reached the approval gate. A reviewer could "approve" it, and compliance
    artifacts were then generated for a model that did not exist. Found by the
    governance evaluation on German Credit: the reviewer rejected every model that
    trained, leaving only one that could not.
    """
    if state.get("selected_model_bytes") is None:
        print("[router] Training produced no model → ending the run without a gate")
        return "training_failure"
    return "fairness_node"


def _mark_training_failure(state: PipelineState, config: RunnableConfig) -> dict:
    """Terminal node: record that nothing was presented for approval, and why."""
    run_id = _get_run_id(config)
    training_result = state.get("training_result") or {}
    error = training_result.get("error")
    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="final_outcome",
        event_source="automated",
        summary=(
            "Pipeline execution terminated: no model could be trained"
            + (f" ({error})" if error else "")
            + ". Nothing was presented for approval."
        ),
        details={
            "status": "TRAINING_FAILED",
            "error": error,
            "rejected_models": state.get("rejected_models") or [],
            "rejection_reroute_count": state.get("rejection_reroute_count", 0),
        },
    )
    return {"unresolved_training_failure": True}


# ---------------------------------------------------------------------------
# Node: Fairness Agent
# ---------------------------------------------------------------------------


def fairness_node(state: PipelineState, config: RunnableConfig) -> dict:
    run_id = _get_run_id(config)
    cleaned_df = bytes_to_df(state["cleaned_df_bytes"])
    model_bytes = state.get("selected_model_bytes")

    if model_bytes is None:
        print("[fairness_node] No fitted model bytes in state — skipping fairness check")
        err_res = {
            "error": "No fitted model in state",
            "overall_fairness_passed": None,
            "fairness_evaluated": False,
            "fairness_report": [],
            "attributes_skipped": [],
            "actions_taken": ["NOT EVALUATED: no fitted model bytes in state"],
        }
        log_audit_event(
            run_id=run_id,
            db_path=AUDIT_DB_PATH,
            event_type="fairness_run",
            event_source="automated",
            summary="Fairness Agent skipped (no fitted model in state)",
            details=err_res,
        )
        return {"fairness_result": err_res}

    fitted_model = bytes_to_model(model_bytes)
    plan = state.get("plan") or {}
    sensitive_candidates = (
        plan.get("sensitive_attribute_candidates")
        or plan.get("sensitive_attributes")
        or plan.get("protected_attributes")
        or []
    )
    print(f"[fairness_node] Running fairness check on candidates: {sensitive_candidates}")

    split = state.get("split_index") or {}

    result = run_fairness_agent(
        cleaned_df=cleaned_df,
        fitted_model=fitted_model,
        target_column=state["target_column"],
        sensitive_attribute_candidates=sensitive_candidates,
        task_type=state["task_type"],
        eval_index=_gate_index(split),
        proxy_findings=findings_for_route(state.get("eda_findings"), "reviewer"),
        # Group membership is read from the raw upload: scaling and encoding in the
        # cleaned frame destroy it (a 0/1 sex column becomes a float).
        raw_frame=bytes_to_df(state["df_bytes"]),
        declared_protected=state.get("declared_protected_attributes"),
        **_fairness_limits(state),
    )

    passed = result.get("overall_fairness_passed", False)
    n_evaluated = len(result.get("fairness_report", []))
    n_violations = sum(1 for e in result.get("fairness_report", []) if e.get("violation"))
    print(
        f"[fairness_node] overall_fairness_passed={passed} | "
        f"evaluated={n_evaluated} | violations={n_violations}"
    )

    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="fairness_run",
        event_source="automated",
        summary=f"Fairness Agent evaluated {n_evaluated} sensitive attribute(s) (overall_passed={passed}, violations={n_violations})",
        details=result,
    )

    return {"fairness_result": result}


# ---------------------------------------------------------------------------
# Node: Human Approval Gate
# ---------------------------------------------------------------------------


def human_approval_node(state: PipelineState, config: RunnableConfig) -> dict:
    """
    Lightweight gate node whose ENTIRE job is to assemble the review payload and
    call interrupt().
    Does NO heavy computation or deserialisation so re-executing this node
    upon resumption (LangGraph's standard node re-execution behavior) is instant and safe.
    """
    run_id = _get_run_id(config)
    plan = state.get("plan") or {}
    data_res = state.get("data_agent_result") or {}
    train_res = state.get("training_result") or {}
    fair_res = state.get("fairness_result") or {}

    awaiting_second = bool(state.get("awaiting_second_approval"))

    payload = {
        "question": (
            "Second sign-off required: a reviewer other than the first approver must "
            "confirm this approval, or reject it."
            if awaiting_second else
            "Governance Review: Please evaluate pipeline outputs and select a decision."
        ),
        # Approve is withheld when policy blocks it; the reason is shown to the reviewer.
        "allowed_decisions": [
            d for d in ("approve", "reject_data_quality", "reject_model_or_fairness",
                        "reject_and_mitigate")
            if not (d == "approve" and _approval_block_reason(state))
        ],
        "approval_blocked_reason": _approval_block_reason(state),
        # Dual sign-off (policy). The reason is shown before the first approval, so a
        # reviewer knows the click will not finish the run.
        "dual_signoff_required_reason": _dual_signoff_reason(state),
        "awaiting_second_approval": awaiting_second,
        "first_approval": state.get("first_approval"),
        "signoff_error": state.get("signoff_error"),
        "reviewer_decisions": state.get("reviewer_decisions") or [],
        "policy_version": state.get("policy_version"),
        "policy_sha256": state.get("policy_sha256"),
        # Pure function of recorded state, so safe before interrupt() (invariant 6).
        "mitigation": mitigation_summary(state.get("mitigation"), train_res, fair_res),
        "plan_summary": {
            "data_quality_concerns": plan.get("data_quality_concerns", []),
            "recommended_preprocessing_steps": plan.get("recommended_preprocessing_steps", []),
            "recommended_models": plan.get("recommended_models", []),
            "sensitive_attribute_candidates": plan.get("sensitive_attribute_candidates", []),
            "reasoning": plan.get("reasoning", ""),
        },
        "data_agent_actions": data_res.get("actions_taken", []),
        "quality_report": data_res.get("quality_report", {}),
        "selected_model_name": train_res.get("selected_model_name"),
        "selected_model_metrics": train_res.get("selected_model_metrics", {}),
        "leaderboard": train_res.get("leaderboard", []),
        "fairness_report": fair_res.get("fairness_report", []),
        "overall_fairness_passed": fair_res.get("overall_fairness_passed"),
        "fairness_evaluated": fair_res.get("fairness_evaluated", False),
        "fairness_coverage": fair_res.get("fairness_coverage"),
        "protected_attributes_unaudited": fair_res.get("protected_attributes_unaudited", []),
        "advisory_violations": fair_res.get("advisory_violations", []),
        "intersectional_report": fair_res.get("intersectional_report", []),
        "intersections_skipped": fair_res.get("intersections_skipped", []),
        "declared_protected_attributes": state.get("declared_protected_attributes") or [],
        "attributes_skipped": fair_res.get("attributes_skipped", []),
        "unresolved_quality_issue": state.get("unresolved_quality_issue", False),
        # Every EDA finding annotated with what each stage did with it. A pure
        # function of recorded state, so safe to build before interrupt().
        "eda_findings": build_eda_linkage(
            state.get("eda_findings") or [], plan, data_res, fair_res,
        ),
        "proxy_warnings": fair_res.get("proxy_warnings", []),
        "fairness_min_group_size": fair_res.get("min_group_size"),
    }

    print("[human_approval_node] Interrupting execution for Human Approval...")
    raw_decision = interrupt(payload)

    if isinstance(raw_decision, dict):
        decision = raw_decision.get("decision", "approve")
        human_feedback = raw_decision.get("human_feedback", "")
    else:
        decision = str(raw_decision)
        human_feedback = ""

    # Who submitted it. Recorded for every decision, not only approvals: a rejection
    # nobody is named for is as unaccountable as an approval nobody is named for.
    reviewer = _reviewer_from_resume(raw_decision)
    stage = "second" if awaiting_second else "first"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "feedback": human_feedback,
        "stage": stage,
        **reviewer,
    }

    print(f"[human_approval_node] RESUMED! Human decision received = '{decision}', "
          f"feedback = '{human_feedback}', reviewer = {_reviewer_label(reviewer)}")

    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="human_decision",
        event_source="human_reviewer",
        summary=(f"Reviewer {_reviewer_label(reviewer)} submitted decision: '{decision}'"
                 + (f" (Feedback: '{human_feedback}')" if human_feedback else "")),
        details={
            "human_decision": decision,
            "human_feedback": human_feedback,
            "reviewer_id": reviewer["reviewer_id"],
            "reviewer_role": reviewer["reviewer_role"],
            "reviewer_authenticated": reviewer["authenticated"],
            "signoff_stage": stage,
            "rejection_reroute_count": state.get("rejection_reroute_count", 0),
        },
    )

    # The payload the reviewer decided on, kept so a completed run can still show its
    # evaluation tabs. Written after interrupt() returns, so re-execution is harmless.
    return {"human_decision": decision, "human_feedback": human_feedback,
            "current_reviewer": reviewer,
            "reviewer_decisions": [*(state.get("reviewer_decisions") or []), record],
            # Any new decision supersedes the previous sign-off complaint.
            "signoff_error": None,
            "last_review_payload": payload}


# ---------------------------------------------------------------------------
# Node: Audit Log (Final Approval Node)
# ---------------------------------------------------------------------------


def _final_test_evaluation(state: PipelineState, run_id: str) -> dict | None:
    """
    Score the approved model once on the untouched test rows. Approve path only.

    No gate decision looked at these rows, so these are the numbers to report. Returns
    None when there is no separate validation split (the gate already used the test
    rows) or no model. A failure is recorded, not raised: it must not undo an approval.
    """
    split = state.get("split_index") or {}
    if not split.get("validation") or not split.get("test") or not state.get("selected_model_bytes"):
        return None
    plan = state.get("plan") or {}
    try:
        cleaned_df = bytes_to_df(state["cleaned_df_bytes"])
        model = bytes_to_model(state["selected_model_bytes"])
        metrics = evaluate_model(model, cleaned_df, state["target_column"],
                                 state["task_type"], split["test"])
        fairness = run_fairness_agent(
            cleaned_df=cleaned_df,
            fitted_model=model,
            target_column=state["target_column"],
            sensitive_attribute_candidates=plan.get("sensitive_attribute_candidates") or [],
            task_type=state["task_type"],
            eval_index=split["test"],
            raw_frame=bytes_to_df(state["df_bytes"]),
            declared_protected=state.get("declared_protected_attributes"),
            **_fairness_limits(state),
        )
        result = {
            "split": "test",
            "rows": len(split["test"]),
            "metrics": metrics,
            "fairness": {k: fairness.get(k) for k in (
                "overall_fairness_passed", "fairness_evaluated", "fairness_coverage",
                "fairness_report", "advisory_violations", "protected_attributes_unaudited",
                "attributes_skipped", "evaluated_rows", "min_group_size",
                "intersectional_report", "intersections_skipped")},
        }
        summary = (f"Final evaluation of the approved model on {result['rows']:,} untouched "
                   f"test rows: {metrics}; fairness overall_passed="
                   f"{fairness.get('overall_fairness_passed')}")
    except Exception as exc:
        result = {"split": "test", "error": f"{type(exc).__name__}: {exc}"}
        summary = f"Final test evaluation FAILED: {result['error']}"
    print(f"[audit_log_node] {summary}")
    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="final_test_evaluation",
        event_source="automated",
        summary=summary,
        details=result,
    )
    return result


def audit_log_node(state: PipelineState, config: RunnableConfig) -> dict:
    """
    Final node on the approve path. Serialises the approved model to disk and
    logs the final outcome event.

    The model is written here rather than in training_node because only an
    approved model is a deployable artifact. A model that was trained but then
    rejected at the governance gate must not appear on disk as though it had
    passed review.
    """
    run_id = _get_run_id(config)
    training_result = state.get("training_result") or {}
    fairness_result = state.get("fairness_result") or {}
    selected_name = training_result.get("selected_model_name", "unknown")
    fairness_passed = fairness_result.get("overall_fairness_passed")

    # --- Serialise the approved model -------------------------------------
    # selected_model_bytes was produced by graph_state.model_to_bytes(), which
    # is a joblib dump — so the bytes are already a valid .joblib payload and can
    # be written straight out without a deserialise/reserialise round trip.
    saved_path = None
    save_error = None
    model_bytes = state.get("selected_model_bytes")

    if not model_bytes:
        save_error = "No fitted model bytes in state — nothing to serialise"
        print(f"[audit_log_node] WARNING: {save_error}")
    else:
        try:
            os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(selected_name))
            candidate = os.path.join(
                SAVED_MODELS_DIR, f"{run_id}_{safe_name}.joblib"
            )
            with open(candidate, "wb") as fh:
                fh.write(model_bytes)
            saved_path = candidate
            print(
                f"[audit_log_node] Saved approved model to '{saved_path}' "
                f"({len(model_bytes):,} bytes)"
            )
        except Exception as exc:
            save_error = f"{type(exc).__name__}: {exc}"
            saved_path = None
            print(f"[audit_log_node] FAILED to save model: {save_error}")

    final_evaluation = _final_test_evaluation(state, run_id)
    approvers = _approvers(state)

    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="final_outcome",
        event_source="automated",
        summary=(
            "Pipeline execution completed successfully with human approval by "
            + (" and ".join(_reviewer_label(a) for a in approvers) if approvers
               else "an unidentified reviewer")
            + "."
            + (f" Model artifact saved to '{saved_path}'." if saved_path
               else f" Model artifact NOT saved ({save_error}).")
        ),
        details={
            "status": "APPROVED",
            "selected_model": selected_name,
            "approvers": approvers,
            "dual_signoff": len(approvers) > 1,
            "overall_fairness_passed": fairness_passed,
            "fairness_evaluated": fairness_result.get("fairness_evaluated", False),
            "fairness_coverage": fairness_result.get("fairness_coverage"),
            "advisory_violations": fairness_result.get("advisory_violations", []),
            "declared_protected_attributes": state.get("declared_protected_attributes") or [],
            "policy_version": state.get("policy_version"),
            "policy_sha256": state.get("policy_sha256"),
            "protected_attributes_unaudited": [
                p.get("attribute") for p in fairness_result.get("protected_attributes_unaudited", [])
            ],
            "model_saved_path": saved_path,
            "model_save_error": save_error,
            "final_test_metrics": (final_evaluation or {}).get("metrics"),
            "final_test_fairness_passed": ((final_evaluation or {}).get("fairness") or {}).get(
                "overall_fairness_passed"),
            "total_retries_used": state.get("retry_count", 0),
            "total_human_reroutes_used": state.get("rejection_reroute_count", 0),
        },
    )
    print(f"[audit_log_node] Final outcome logged to audit_log.db for run_id='{run_id}'")

    # --- Compliance artifacts ---------------------------------------------
    # Generated AFTER the final_outcome event, so the audit chain head recorded
    # inside the AIBOM covers the whole run. The resulting digests are then logged
    # as their own chained event, and that is what makes the artifacts themselves
    # tamper-evident: editing a model card afterwards leaves its hash disagreeing
    # with the one recorded here. verify_artifacts() performs the comparison.
    artifacts = None
    try:
        artifacts = generate_artifacts(
            state={**state, "model_saved_path": saved_path, "final_evaluation": final_evaluation},
            run_id=run_id,
            out_root=ARTIFACTS_DIR,
            audit_db_path=AUDIT_DB_PATH,
        )
        log_audit_event(
            run_id=run_id,
            db_path=AUDIT_DB_PATH,
            event_type=ARTIFACT_EVENT_TYPE,
            event_source="automated",
            summary=(
                f"Generated {len(artifacts['files'])} compliance artifact(s) in "
                f"'{artifacts['directory']}': {', '.join(sorted(artifacts['files']))}"
                + (f" (errors: {'; '.join(artifacts['errors'])})"
                   if artifacts["errors"] else "")
            ),
            details={
                "directory": artifacts["directory"],
                "files": artifacts["files"],
                "errors": artifacts["errors"],
            },
        )
        print(
            f"[audit_log_node] Compliance artifacts written to "
            f"'{artifacts['directory']}' ({len(artifacts['files'])} files)"
        )
    except Exception as exc:
        # Never fail an approved run because paperwork generation broke: the model
        # and the audit trail are the load-bearing outputs. Record the failure in
        # the log instead, so a missing artifact set is itself auditable.
        print(f"[audit_log_node] Compliance artifact generation FAILED: "
              f"{type(exc).__name__}: {exc}")
        log_audit_event(
            run_id=run_id,
            db_path=AUDIT_DB_PATH,
            event_type=ARTIFACT_EVENT_TYPE,
            event_source="automated",
            summary=f"Compliance artifact generation FAILED: {type(exc).__name__}: {exc}",
            details={"directory": None, "files": {}, "errors": [str(exc)]},
        )

    return {
        "model_saved_path": saved_path,
        "model_save_error": save_error,
        "artifacts_manifest": artifacts,
        "final_evaluation": final_evaluation,
    }


# ---------------------------------------------------------------------------
# Conditional edge: route after human_approval_node
# ---------------------------------------------------------------------------


def route_after_human_approval(state: PipelineState) -> str:
    """
    Conditional routing edge after human_approval_node.
    Returns:
      "audit_log_node"    — human approved ("approve") -> proceeds to audit logging -> END
      "reroute_planner"   — human rejected data quality ("reject_data_quality")
      "reroute_training"  — human rejected model/fairness ("reject_model_or_fairness")
      "human_cap_failure" — human rejected, but rejection cap reached
      "record_first_approval" — approved, but the policy wants a second reviewer
      "signoff_rejected" — approved without a usable second identity; back to the gate
      "approval_blocked" — approve refused outright by policy
    """
    decision = state.get("human_decision")
    reroute_count = state.get("rejection_reroute_count", 0)

    if decision == "approve":
        if _approval_block_reason(state):
            print("[router] Human decision: APPROVE refused by governance policy → "
                  "END with unresolved_approval_blocked=True")
            return "approval_blocked"
        if _dual_signoff_reason(state):
            shortfall = _signoff_shortfall(state)
            if shortfall:
                print(f"[router] Human decision: APPROVE does not complete dual sign-off "
                      f"({shortfall}) → back to the gate")
                return "signoff_rejected"
            if not state.get("first_approval"):
                print("[router] Human decision: APPROVE is the FIRST of two required "
                      "sign-offs → back to the gate for a second reviewer")
                return "record_first_approval"
            print("[router] Human decision: APPROVED by a second, different reviewer → "
                  "audit_log_node")
            return "audit_log_node"
        print(f"[router] Human decision: APPROVED → audit_log_node (human reroutes used: {reroute_count})")
        return "audit_log_node"

    max_reroutes = _policy_value(state, "governance", "max_human_reroutes", MAX_HUMAN_REROUTES)
    if reroute_count >= max_reroutes:
        print(
            f"[router] Human decision: '{decision}', but rejection cap ({max_reroutes}) "
            "reached → routing to END with unresolved_human_rejection=True"
        )
        return "human_cap_failure"

    if decision == "reject_data_quality":
        print(
            f"[router] Human decision: REJECT_DATA_QUALITY → reroute {reroute_count + 1}/{max_reroutes} "
            "to planner_node"
        )
        return "reroute_planner"

    if decision == "reject_model_or_fairness":
        print(
            f"[router] Human decision: REJECT_MODEL_OR_FAIRNESS → reroute {reroute_count + 1}/{max_reroutes} "
            "directly to training_node (skipping planner and data agent)"
        )
        return "reroute_training"

    if decision == "reject_and_mitigate":
        print(
            f"[router] Human decision: REJECT_AND_MITIGATE → reroute {reroute_count + 1}/{max_reroutes} "
            "to mitigation_node, then retraining with reweighing"
        )
        return "reroute_mitigation"

    print(f"[router] Unrecognized human decision '{decision}' → routing to audit_log_node")
    return "audit_log_node"


# ---------------------------------------------------------------------------
# State-mutation helper nodes
# ---------------------------------------------------------------------------


def _increment_retry(state: PipelineState) -> dict:
    """Called when automated Data Agent quality check failed and retries remain."""
    quality_report = state["data_agent_result"]["quality_report"]
    new_count = state.get("retry_count", 0) + 1
    print(f"[increment_retry] retry_count: {new_count - 1} → {new_count}")
    return {
        "last_failure_reason": quality_report,
        "retry_count": new_count,
    }


def _mark_cap_failure(state: PipelineState, config: RunnableConfig) -> dict:
    """Called when automated Data Agent retry cap is reached."""
    run_id = _get_run_id(config)
    print("[mark_cap_failure] Setting unresolved_quality_issue=True")
    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="final_outcome",
        event_source="automated",
        summary="Pipeline execution terminated: Data Agent retry cap reached.",
        details={
            "status": "DATA_QUALITY_CAP_REACHED",
            "retry_count": state.get("retry_count", 0),
        },
    )
    return {"unresolved_quality_issue": True}


def _increment_human_reroute_planner(state: PipelineState) -> dict:
    """Called when human rejects for data quality. Sets failure_context and increments reroute count."""
    new_count = state.get("rejection_reroute_count", 0) + 1
    failure_context = {
        "source": "human_reviewer",
        "reason": "Human auditor rejected data quality outcome during governance review.",
    }
    print(f"[increment_human_reroute_planner] rejection_reroute_count: {new_count - 1} → {new_count}")
    return {
        "last_failure_reason": failure_context,
        "rejection_reroute_count": new_count,
        # The next gate reviews a different pipeline, so a pending sign-off lapses.
        **_clear_pending_signoff(),
    }


def _increment_human_reroute_training(state: PipelineState) -> dict:
    """Called when human rejects model/fairness. Increments reroute count and tracks rejected model."""
    new_count = state.get("rejection_reroute_count", 0) + 1
    selected_model = state.get("training_result", {}).get("selected_model_name")
    rejected = list(state.get("rejected_models") or [])
    if selected_model and selected_model not in rejected:
        rejected.append(selected_model)
    print(
        f"[increment_human_reroute_training] rejection_reroute_count: {new_count - 1} → {new_count} | "
        f"Appended '{selected_model}' to rejected_models: {rejected}"
    )
    return {
        "rejection_reroute_count": new_count,
        "rejected_models": rejected,
        **_clear_pending_signoff(),
    }


def _increment_human_reroute_mitigation(state: PipelineState) -> dict:
    """
    Called when the human chooses reject_and_mitigate. Counts against the same cap
    as the other rejections. The selected model is NOT excluded: the same candidates
    are retrained with weights, so the comparison isolates the mitigation.
    """
    new_count = state.get("rejection_reroute_count", 0) + 1
    print(f"[increment_human_reroute_mitigation] rejection_reroute_count: {new_count - 1} → {new_count}")
    return {"rejection_reroute_count": new_count, **_clear_pending_signoff()}


def mitigation_node(state: PipelineState, config: RunnableConfig) -> dict:
    """
    Choose the attribute to mitigate and compute reweighing cell weights.

    Deterministic, no LLM. Records the numbers the reviewer saw at the gate as
    `before`, so the next gate can show before against after.
    """
    run_id = _get_run_id(config)
    previous = state.get("mitigation") or {}
    attributes = list(previous.get("attributes") or [])
    applications = list(previous.get("applications") or [])
    fair_res = state.get("fairness_result") or {}
    before = mitigation_snapshot(state.get("training_result"), fair_res)

    chosen = choose_mitigation_attribute(fair_res, attributes)
    if chosen is None:
        application = {"attribute": None, "status": "skipped", "before": before,
                       "reason": "no violated attribute that has not already been mitigated"}
    else:
        split = state.get("split_index") or {}
        cleaned_df = bytes_to_df(state["cleaned_df_bytes"])
        try:
            _, cells = row_weights(
                bytes_to_df(state["df_bytes"]), attributes + [chosen], split.get("train"),
                cleaned_df[state["target_column"]],
            )
            attributes.append(chosen)
            application = {"attribute": chosen, "status": "applied", "cells": cells,
                           "before": before}
        except ValueError as exc:
            application = {"attribute": chosen, "status": "skipped", "reason": str(exc),
                           "before": before}
    applications.append(application)

    applied = application["status"] == "applied"
    print(f"[mitigation_node] {application['status']}: {application.get('attribute')} "
          f"| mitigated attributes now {attributes}")
    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="mitigation_applied" if applied else "mitigation_skipped",
        event_source="automated",
        summary=(
            f"Reweighing on {INTERSECTION_SEPARATOR.join(attributes)} "
            f"({len(application['cells'])} group-label cells, train rows only)"
            if applied else f"Mitigation skipped: {application['reason']}"
        ),
        details={"method": MITIGATION_METHOD, "attributes": attributes, **application},
    )
    return {"mitigation": {"method": MITIGATION_METHOD, "attributes": attributes,
                           "applications": applications}}


def _clear_pending_signoff() -> dict:
    """State reset for a reroute: a half-finished approval must not survive it."""
    return {"awaiting_second_approval": False, "first_approval": None, "signoff_error": None}


def _record_first_approval(state: PipelineState, config: RunnableConfig) -> dict:
    """
    Hold a first approval and send the run back to the gate for a second reviewer.

    The run is NOT approved here: no model is written, no artifacts are generated, and
    `human_decision` is cleared so nothing downstream can read a half-finished approval
    as a completed one.
    """
    run_id = _get_run_id(config)
    reviewer = state.get("current_reviewer") or {}
    reason = _dual_signoff_reason(state) or "the governance policy requires dual sign-off"
    record = {
        **reviewer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feedback": state.get("human_feedback") or "",
    }
    print(f"[record_first_approval] First approval from {_reviewer_label(reviewer)}; "
          "awaiting a second, different reviewer")
    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="signoff_first_approval",
        event_source="automated",
        summary=(f"First approval recorded from {_reviewer_label(reviewer)}. The run is "
                 f"NOT approved: a second, different reviewer must sign off, because "
                 f"{reason}."),
        details={
            "first_approval": record,
            "reason": reason,
            "policy_version": state.get("policy_version"),
            "policy_sha256": state.get("policy_sha256"),
        },
    )
    return {"first_approval": record, "awaiting_second_approval": True,
            "human_decision": None}


def _reject_signoff(state: PipelineState, config: RunnableConfig) -> dict:
    """
    An approval that cannot count as a sign-off, logged and returned to the gate.

    Reached when an approval carries no reviewer id, or when the first approver tries
    to supply the second sign-off as well. The attempt is recorded either way: a
    refused self-approval is exactly the event a reviewer of the reviewers wants to see.
    """
    run_id = _get_run_id(config)
    reviewer = state.get("current_reviewer") or {}
    reason = _signoff_shortfall(state) or "the approval did not satisfy dual sign-off"
    print(f"[reject_signoff] {reason} → back to the gate")
    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="signoff_rejected",
        event_source="automated",
        summary=(f"Approval from {_reviewer_label(reviewer)} was not accepted as a "
                 f"sign-off: {reason}."),
        details={
            "reason": reason,
            "attempted_by": reviewer,
            "first_approval": state.get("first_approval"),
            "policy_version": state.get("policy_version"),
        },
    )
    return {"signoff_error": reason, "human_decision": None}


def _mark_approval_blocked(state: PipelineState, config: RunnableConfig) -> dict:
    """
    An approve decision refused by policy ends the run without an approved model.

    The dashboard and API refuse it before it reaches the graph; this is the last line
    of enforcement for any other caller. Nothing is saved and no artifacts are written.
    """
    run_id = _get_run_id(config)
    fairness = state.get("fairness_result") or {}
    reason = _approval_block_reason(state) or "approval blocked by governance policy"
    print(f"[mark_approval_blocked] {reason}")
    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="final_outcome",
        event_source="automated",
        summary=f"Approval refused by governance policy: {reason}",
        details={
            "status": "APPROVAL_BLOCKED_BY_POLICY",
            "reason": reason,
            "selected_model": (state.get("training_result") or {}).get("selected_model_name"),
            "overall_fairness_passed": fairness.get("overall_fairness_passed"),
            "fairness_coverage": fairness.get("fairness_coverage"),
            "policy_version": state.get("policy_version"),
            "policy_sha256": state.get("policy_sha256"),
        },
    )
    return {"unresolved_approval_blocked": True}


def _mark_human_cap_failure(state: PipelineState, config: RunnableConfig) -> dict:
    """Called when human rejection cap is reached."""
    run_id = _get_run_id(config)
    print("[mark_human_cap_failure] Setting unresolved_human_rejection=True")
    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="final_outcome",
        event_source="automated",
        summary="Pipeline execution terminated: Human rejection cap reached.",
        details={
            "status": "HUMAN_REJECTION_CAP_REACHED",
            "rejection_reroute_count": state.get("rejection_reroute_count", 0),
        },
    )
    return {"unresolved_human_rejection": True}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(db_path: str = "pipeline_state.db"):
    """
    Construct and compile the multi-agent governance graph with SqliteSaver and Audit Logging.

    Nodes: planner_node → data_agent_node → training_node → fairness_node → human_approval_node → audit_log_node → END.
    Reroutes:
      - Data Agent auto-retry loop: data_agent_node → planner_node
      - Human reject_data_quality: human_approval_node → planner_node
      - Human reject_model_or_fairness: human_approval_node → training_node (skips data agent)
      - Dual sign-off: human_approval_node → record_first_approval / reject_signoff →
        human_approval_node (the gate re-opens; nothing is written until a second,
        different reviewer approves)

    Checkpointer Rationale:
      Switched from MemorySaver to SqliteSaver (persisted local SQLite file).
      Human approval reviews can take minutes, hours, or days. In-memory checkpoints
      disappear if the process restarts. SqliteSaver guarantees thread_id state
      persistence across process and server restarts.
    """
    builder = StateGraph(PipelineState)

    # Register processing nodes
    builder.add_node("data_analysis_node", data_analysis_node)
    builder.add_node("planner_node", planner_node)
    builder.add_node("data_agent_node", data_agent_node)
    builder.add_node("training_node", training_node)
    builder.add_node("fairness_node", fairness_node)
    builder.add_node("human_approval_node", human_approval_node)
    builder.add_node("audit_log_node", audit_log_node)

    # Register state-mutation helper nodes
    builder.add_node("increment_retry", _increment_retry)
    builder.add_node("mark_cap_failure", _mark_cap_failure)
    builder.add_node("increment_human_reroute_planner", _increment_human_reroute_planner)
    builder.add_node("increment_human_reroute_training", _increment_human_reroute_training)
    builder.add_node("increment_human_reroute_mitigation", _increment_human_reroute_mitigation)
    builder.add_node("mitigation_node", mitigation_node)
    builder.add_node("record_first_approval", _record_first_approval)
    builder.add_node("reject_signoff", _reject_signoff)
    builder.add_node("mark_approval_blocked", _mark_approval_blocked)
    builder.add_node("mark_human_cap_failure", _mark_human_cap_failure)
    builder.add_node("mark_training_failure", _mark_training_failure)

    # Fixed edges
    builder.add_edge(START, "data_analysis_node")
    builder.add_edge("data_analysis_node", "planner_node")
    builder.add_edge("planner_node", "data_agent_node")

    # Conditional routing after data agent
    builder.add_conditional_edges(
        "data_agent_node",
        route_after_data_agent,
        {
            "training_node": "training_node",
            "planner_node": "increment_retry",
            "cap_failure": "mark_cap_failure",
        },
    )
    builder.add_edge("increment_retry", "planner_node")
    builder.add_edge("mark_cap_failure", END)

    # Training → Fairness → Human Approval
    # A run with no trained model ends here: there is nothing to audit or approve.
    builder.add_conditional_edges(
        "training_node",
        route_after_training,
        {"fairness_node": "fairness_node", "training_failure": "mark_training_failure"},
    )
    builder.add_edge("mark_training_failure", END)
    builder.add_edge("fairness_node", "human_approval_node")

    # Conditional routing after human approval
    builder.add_conditional_edges(
        "human_approval_node",
        route_after_human_approval,
        {
            "audit_log_node": "audit_log_node",
            "reroute_planner": "increment_human_reroute_planner",
            "reroute_training": "increment_human_reroute_training",
            "reroute_mitigation": "increment_human_reroute_mitigation",
            "human_cap_failure": "mark_human_cap_failure",
            "approval_blocked": "mark_approval_blocked",
            "record_first_approval": "record_first_approval",
            "signoff_rejected": "reject_signoff",
        },
    )
    builder.add_edge("mark_approval_blocked", END)
    # Dual sign-off: both paths re-open the gate rather than ending the run.
    builder.add_edge("record_first_approval", "human_approval_node")
    builder.add_edge("reject_signoff", "human_approval_node")
    builder.add_edge("increment_human_reroute_mitigation", "mitigation_node")
    builder.add_edge("mitigation_node", "training_node")

    builder.add_edge("audit_log_node", END)
    builder.add_edge("increment_human_reroute_planner", "planner_node")
    builder.add_edge("increment_human_reroute_training", "training_node")
    builder.add_edge("mark_human_cap_failure", END)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)


# Module-level compiled graph
graph = build_graph()
