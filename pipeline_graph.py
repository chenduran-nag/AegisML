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
      │                       ├─ "approve"                   ►►►►► END
      │                       ├─ "reject_data_quality"       ►►►►► planner_node
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
from training_agent import run_training_agent
from fairness_agent import run_fairness_agent
from audit_log import DEFAULT_AUDIT_DB, log_audit_event
from compliance_artifacts import ARTIFACT_EVENT_TYPE, generate_artifacts
from data_analysis_agent import analyze_raw_dataset
from eda_insights import build_eda_linkage, derive_eda_findings, findings_for_route

MAX_RETRIES = 2          # maximum planner→data_agent auto-retries
MAX_HUMAN_REROUTES = 2   # maximum human rejection reroutes before capping

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

    try:
        eda_report = analyze_raw_dataset(df, target_column)
    except Exception as exc:
        print(f"[data_analysis_node] Profiling failed: {type(exc).__name__}: {exc}")
        eda_report = {"error": f"{type(exc).__name__}: {exc}"}

    findings_error = None
    try:
        findings = derive_eda_findings(df, target_column, task_type)
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
    )

    cleaned_df = result.pop("cleaned_df")
    cleaned_df_bytes = df_to_bytes(cleaned_df)

    # Lift the split out of `result` before it reaches the audit log: these are
    # per-row index lists (tens of thousands of ints on a real dataset) and would
    # swamp every audit entry. They live in their own state key instead.
    split_index = {
        "train": result.pop("train_index"),
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

    if retry_count < MAX_RETRIES:
        print(
            f"[router] Quality FAILED → retry {retry_count + 1}/{MAX_RETRIES} "
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

    print(f"[training_node] Training models: {recommended_models}")

    split = state.get("split_index") or {}

    try:
        raw = run_training_agent(
            cleaned_df=cleaned_df,
            target_column=state["target_column"],
            task_type=state["task_type"],
            recommended_models=recommended_models,
            train_index=split.get("train"),
            test_index=split.get("test"),
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
        eval_index=split.get("test"),
        proxy_findings=findings_for_route(state.get("eda_findings"), "reviewer"),
        # Group membership is read from the raw upload: scaling and encoding in the
        # cleaned frame destroy it (a 0/1 sex column becomes a float).
        raw_frame=bytes_to_df(state["df_bytes"]),
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

    payload = {
        "question": "Governance Review: Please evaluate pipeline outputs and select a decision.",
        "allowed_decisions": ["approve", "reject_data_quality", "reject_model_or_fairness"],
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

    print(f"[human_approval_node] RESUMED! Human decision received = '{decision}', feedback = '{human_feedback}'")

    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="human_decision",
        event_source="human_reviewer",
        summary=f"Human reviewer submitted decision: '{decision}'" + (f" (Feedback: '{human_feedback}')" if human_feedback else ""),
        details={
            "human_decision": decision,
            "human_feedback": human_feedback,
            "rejection_reroute_count": state.get("rejection_reroute_count", 0),
        },
    )

    return {"human_decision": decision, "human_feedback": human_feedback}


# ---------------------------------------------------------------------------
# Node: Audit Log (Final Approval Node)
# ---------------------------------------------------------------------------


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

    log_audit_event(
        run_id=run_id,
        db_path=AUDIT_DB_PATH,
        event_type="final_outcome",
        event_source="automated",
        summary=(
            "Pipeline execution completed successfully with human approval."
            + (f" Model artifact saved to '{saved_path}'." if saved_path
               else f" Model artifact NOT saved ({save_error}).")
        ),
        details={
            "status": "APPROVED",
            "selected_model": selected_name,
            "overall_fairness_passed": fairness_passed,
            "fairness_evaluated": fairness_result.get("fairness_evaluated", False),
            "fairness_coverage": fairness_result.get("fairness_coverage"),
            "protected_attributes_unaudited": [
                p.get("attribute") for p in fairness_result.get("protected_attributes_unaudited", [])
            ],
            "model_saved_path": saved_path,
            "model_save_error": save_error,
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
            state={**state, "model_saved_path": saved_path},
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
    """
    decision = state.get("human_decision")
    reroute_count = state.get("rejection_reroute_count", 0)

    if decision == "approve":
        print(f"[router] Human decision: APPROVED → audit_log_node (human reroutes used: {reroute_count})")
        return "audit_log_node"

    if reroute_count >= MAX_HUMAN_REROUTES:
        print(
            f"[router] Human decision: '{decision}', but rejection cap ({MAX_HUMAN_REROUTES}) "
            "reached → routing to END with unresolved_human_rejection=True"
        )
        return "human_cap_failure"

    if decision == "reject_data_quality":
        print(
            f"[router] Human decision: REJECT_DATA_QUALITY → reroute {reroute_count + 1}/{MAX_HUMAN_REROUTES} "
            "to planner_node"
        )
        return "reroute_planner"

    if decision == "reject_model_or_fairness":
        print(
            f"[router] Human decision: REJECT_MODEL_OR_FAIRNESS → reroute {reroute_count + 1}/{MAX_HUMAN_REROUTES} "
            "directly to training_node (skipping planner and data agent)"
        )
        return "reroute_training"

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
    }


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
            "human_cap_failure": "mark_human_cap_failure",
        },
    )

    builder.add_edge("audit_log_node", END)
    builder.add_edge("increment_human_reroute_planner", "planner_node")
    builder.add_edge("increment_human_reroute_training", "training_node")
    builder.add_edge("mark_human_cap_failure", END)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)


# Module-level compiled graph
graph = build_graph()
