"""
graph_state.py
==============
Shared LangGraph state schema for the AI-Governed Multi-Agent Platform.

DESIGN NOTE — why DataFrames are stored as bytes, not directly:
  LangGraph's MemorySaver serialises ALL state values using msgpack, even
  for in-memory operation. pd.DataFrame is not msgpack-serialisable, so
  DataFrames must be converted to bytes before entering the state.

  We use pickle for this conversion because:
    - Pickle preserves all pandas dtypes faithfully (object, float64, bool,
      nullable string, etc.) with no roundtrip conversion issues.
    - Pickle bytes are plain Python bytes objects — fully msgpack-safe.
    - For in-memory, single-process use (MemorySaver), pickle's security
      limitations are irrelevant (we are pickling our own DataFrames).

  Helper functions df_to_bytes() / bytes_to_df() in this module handle the
  conversion so nodes stay readable.

  When we switch to a persistent checkpointer (SQLite/Postgres), we can
  swap pickle for parquet and update only the two helper functions.
"""

from __future__ import annotations

import io
import pickle
from typing import Optional

import joblib
import pandas as pd
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# DataFrame serialisation helpers
# ---------------------------------------------------------------------------


def df_to_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to bytes for safe storage in LangGraph state."""
    return pickle.dumps(df)


def bytes_to_df(b: bytes) -> pd.DataFrame:
    """Deserialise bytes back to a DataFrame."""
    return pickle.loads(b)


def model_to_bytes(model: object) -> bytes:
    """
    Serialise a fitted sklearn/XGBoost model to bytes using joblib.

    joblib is preferred over pickle for model objects because:
      - It handles large numpy arrays efficiently (memory-mapped compression).
      - It is the standard used by sklearn itself in its persistence docs.
      - Bytes output is plain Python bytes — fully msgpack-safe for MemorySaver.
    """
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


def bytes_to_model(b: bytes) -> object:
    """Deserialise a fitted model from bytes produced by model_to_bytes()."""
    buf = io.BytesIO(b)
    return joblib.load(buf)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class PipelineState(TypedDict, total=False):
    """
    Shared state passed between every node in the pipeline graph.

    Fields
    ------
    df_bytes : bytes
        The raw input dataset, pickled. Deserialise with bytes_to_df().
        Never mutated after initial construction.
    target_column : str
        Name of the target/label column.
    task_type : str
        "classification" or "regression".
    plan : Optional[dict]
        Plan returned by plan_pipeline(). Set by planner_node.
    data_agent_result : Optional[dict]
        Return value of run_data_agent() WITH cleaned_df removed
        (cleaned_df is stored separately as cleaned_df_bytes).
        Contains: quality_check_passed, quality_report, actions_taken.
    cleaned_df_bytes : Optional[bytes]
        The cleaned DataFrame from data_agent_node, pickled.
        Deserialise with bytes_to_df(). Used by training_node.
    dataset_sha256 : Optional[str]
        SHA-256 of the raw uploaded file's bytes, computed by the server before
        parsing. Hashed from the upload rather than from df_bytes because pickle
        bytes are not stable across pandas/Python versions, so a pickle digest
        would change without the data changing. Provenance anchor for the AIBOM.
    planner_meta : Optional[dict]
        Provenance of the planner LLM call: resolved model id, SHA-256 of the exact
        prompts sent, and token usage. Kept separate from `plan` so that metadata
        about the call can never be confused with content from the call.
    artifacts_manifest : Optional[dict]
        Result of compliance_artifacts.generate_artifacts(): output directory plus
        a per-file path/digest map. Set only on the approve path.
    split_seed : Optional[int]
        Seed for the train/test split drawn by the Data Agent. None means the
        default (data_agent.SPLIT_RANDOM_STATE). Set per run by the governance
        evaluation to repeat a run over several partitions.
    eda_findings : Optional[list]
        Structured findings from eda_insights.derive_eda_findings(), each carrying a
        `route` ("data_agent", "planner" or "reviewer") that decides which stage may
        act on it. Set once by data_analysis_node.
    eda_report : Optional[dict]
        Exploratory profiling report from analyze_raw_dataset(), computed by the
        server at upload time. Stored in state (not a module-level cache) so it
        survives a process restart alongside the rest of the run.
    split_index : Optional[dict]
        {"train": [...], "test": [...]} — index labels of the train/test split
        drawn by run_data_agent() BEFORE any preprocessing parameter was fitted.
        Held here rather than inside data_agent_result so these long lists never
        reach the audit log. training_node reuses this split verbatim, and
        fairness_node evaluates on the "test" half.
    last_failure_reason : Optional[dict]
        Populated with quality_report when quality_check_passed = False.
        Passed as failure_context to plan_pipeline() on the next retry.
    retry_count : int
        Number of planner→data_agent cycles completed. Starts at 0.
    unresolved_quality_issue : bool
        True when retry_count >= MAX_RETRIES and quality still fails.
        Prevents silent failures — downstream consumers must check this.
    training_result : Optional[dict]
        Return value of run_training_agent() with the fitted model object
        stripped out (leaderboard, selected_model_name, selected_model_metrics,
        shap_summary, actions_taken). All values are plain Python — no special
        serialisation needed.
    selected_model_bytes : Optional[bytes]
        The fitted selected model, serialised with joblib via model_to_bytes().
        Deserialise with bytes_to_model(). Stored separately to keep
        training_result cleanly serialisable (same pattern as cleaned_df_bytes).
    fairness_result : Optional[dict]
        Return value of run_fairness_agent() (fairness_report,
        overall_fairness_passed, attributes_skipped, actions_taken).
        All values are plain Python — no special serialisation needed.
    human_decision : Optional[str]
        Decision returned by Human Approval gate ("approve", "reject_data_quality",
        "reject_model_or_fairness", or None if not yet reached).
    rejection_reroute_count : int
        Number of human-triggered rejection reroutes executed. Starts at 0.
        Independent counter from retry_count (Data Agent retry cap).
    unresolved_human_rejection : bool
        True when rejection_reroute_count >= MAX_HUMAN_REROUTES and human rejects again.
        Prevents infinite human-rejection loops.
    unresolved_training_failure : bool
        True when training produced no model — every remaining candidate failed to
        fit. The run ends without reaching the approval gate.
    model_saved_path : Optional[str]
        Filesystem path the approved model was serialised to by audit_log_node,
        or None if it was never approved or the write failed. This is the real
        path on disk — never construct a display path from other fields.
    model_save_error : Optional[str]
        Reason the serialisation failed, when model_saved_path is None.
    mitigation : Optional[dict]
        Set by mitigation_node after a "reject_and_mitigate" decision:
        {method, attributes, applications: [{attribute, status, reason, cells,
        before}]}. Cells are per-(group, label) weights, never per-row data. While
        attributes is non-empty, training_node trains with reweighing weights.
    declared_protected_attributes : Optional[list[str]]
        Columns the reviewer declared protected at run start. They are audited, count
        toward the fairness verdict, and feed EDA proxy detection, in addition to the
        columns detected by name.
    final_evaluation : Optional[dict]
        Written by audit_log_node on approval only: the approved model scored once on
        the untouched test rows ({split, rows, metrics, fairness}). Everything before
        approval — leaderboard, fairness audit, reviewer decisions — uses the
        validation rows in split_index["validation"].
    """

    df_bytes: bytes
    target_column: str
    task_type: str
    plan: Optional[dict]
    data_agent_result: Optional[dict]
    cleaned_df_bytes: Optional[bytes]
    split_index: Optional[dict]
    eda_report: Optional[dict]
    eda_findings: Optional[list]
    split_seed: Optional[int]
    dataset_sha256: Optional[str]
    planner_meta: Optional[dict]
    artifacts_manifest: Optional[dict]
    last_failure_reason: Optional[dict]
    retry_count: int
    unresolved_quality_issue: bool
    training_result: Optional[dict]
    selected_model_bytes: Optional[bytes]
    fairness_result: Optional[dict]
    human_decision: Optional[str]
    rejection_reroute_count: int
    unresolved_human_rejection: bool
    unresolved_training_failure: bool
    business_objective: Optional[str]
    rejected_models: Optional[list[str]]
    human_feedback: Optional[str]
    model_saved_path: Optional[str]
    model_save_error: Optional[str]
    mitigation: Optional[dict]
    declared_protected_attributes: Optional[list[str]]
    final_evaluation: Optional[dict]
    last_review_payload: Optional[dict]
