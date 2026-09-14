"""
planner_agent.py
================
Step 1 of the AI-Governed Multi-Agent Platform.

Exposes a single public function:
    plan_pipeline(df, target_column, task_type) -> dict

The function computes a compact dataset summary locally (NO raw rows sent
to the LLM), calls the Groq API, and returns a validated JSON plan with:
  - data_quality_concerns
  - recommended_preprocessing_steps
  - recommended_models
  - sensitive_attribute_candidates
  - reasoning

Environment variable required:
    GROQ_API_KEY   — your Groq API key

Optional override:
    GROQ_MODEL     — Groq model ID to use (default: openai/gpt-oss-20b)
                     As of Aug 2026, active production models on Groq include:
                       - openai/gpt-oss-20b        (primary default)
                       - openai/gpt-oss-120b        (higher capacity)
                       - meta-llama/llama-4-scout-17b-16e-instruct
                     Deprecated as of Aug 15 2026 (do NOT use):
                       - llama-3.3-70b-versatile
                       - llama-3.1-8b-instant
"""

import hashlib
import json
import os
import textwrap
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from groq import Groq

# ---------------------------------------------------------------------------
# Environment loader (single source of truth for local execution)
# ---------------------------------------------------------------------------


def _ensure_groq_api_key():
    """Ensure GROQ_API_KEY is populated from local .env if not set in environment."""
    if os.environ.get("GROQ_API_KEY"):
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_ensure_groq_api_key()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_PLAN_KEYS = {
    "data_quality_concerns",
    "recommended_preprocessing_steps",
    "recommended_models",
    "sensitive_attribute_candidates",
    "reasoning",
}

# Active Groq production model as of Aug 2026.
# Override with the GROQ_MODEL env var if needed.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

# Keyword patterns used locally to hint the LLM about sensitive columns.
SENSITIVE_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "nationality", "religion",
    "age", "disability", "marital", "zip", "postal", "income",
    "education", "occupation",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_dataset_summary(
    df: pd.DataFrame,
    target_column: str,
    task_type: Literal["classification", "regression"],
) -> dict:
    """
    Build a compact, LLM-safe summary of the dataset.
    No raw rows are included — only aggregate statistics.

    Critically, data quality issues are pre-computed deterministically here
    (not left to LLM discretion) so they always appear in data_quality_concerns.
    """
    n_rows, n_cols = df.shape

    # Per-column info
    columns_info: list[dict] = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round(null_count / n_rows * 100, 2) if n_rows > 0 else 0.0
        columns_info.append(
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": null_count,
                "null_pct": null_pct,
            }
        )

    summary: dict = {
        "shape": {"rows": n_rows, "columns": n_cols},
        "target_column": target_column,
        "task_type": task_type,
    }

    # Cap column metadata entries to prevent Groq token limit / 413 error on large datasets
    if len(columns_info) > 40:
        target_cols = [c for c in columns_info if c["name"] == target_column]
        sensitive_cols = [c for c in columns_info if any(kw in c["name"].lower() for kw in SENSITIVE_KEYWORDS)]
        high_null_cols = [c for c in columns_info if c["null_pct"] > 5.0]

        priority_names = {c["name"] for c in target_cols + sensitive_cols + high_null_cols}
        remaining = [c for c in columns_info if c["name"] not in priority_names]

        selected_cols = [c for c in columns_info if c["name"] in priority_names] + remaining[:max(0, 40 - len(priority_names))]
        summary["columns"] = selected_cols
        summary["columns_truncated_note"] = f"Total columns: {n_cols}. Showing {len(selected_cols)} representative columns."
    else:
        summary["columns"] = columns_info

    # --- Pre-compute quality flags deterministically (no LLM involved) ---
    # These are passed to the LLM with an explicit instruction to include
    # ALL of them in data_quality_concerns, removing ambiguity.
    quality_flags: list[str] = []

    # Flag columns with > 5% nulls
    for col in df.columns:
        null_pct = df[col].isnull().sum() / n_rows * 100
        if null_pct > 5.0:
            quality_flags.append(
                f"High null rate in '{col}': {null_pct:.1f}% missing"
            )

    # Target-specific stats
    if target_column in df.columns:
        if task_type == "classification":
            vc = df[target_column].value_counts()
            if len(vc) >= 2:
                imbalance_ratio = round(float(vc.iloc[0]) / float(vc.iloc[-1]), 2)
            else:
                imbalance_ratio = 1.0
            summary["target_distribution"] = {
                "value_counts": {str(k): int(v) for k, v in vc.items()},
                "imbalance_ratio_major_minor": imbalance_ratio,
            }
            # Flag class imbalance
            if imbalance_ratio > 1.5:
                quality_flags.append(
                    f"Class imbalance in target '{target_column}': "
                    f"{imbalance_ratio:.2f}:1 ratio (majority:minority)"
                )
        else:  # regression
            desc = df[target_column].describe()
            target_stats = {
                "mean": round(float(desc["mean"]), 4),
                "std": round(float(desc["std"]), 4),
                "min": round(float(desc["min"]), 4),
                "max": round(float(desc["max"]), 4),
            }
            summary["target_stats"] = target_stats
            # Flag severe regression skew: range >> 3*std
            target_range = target_stats["max"] - target_stats["min"]
            if target_stats["std"] > 0 and target_range > 3 * target_stats["std"]:
                quality_flags.append(
                    f"Severe skew in target '{target_column}': "
                    f"range={target_range:.2f} is more than 3x std={target_stats['std']:.2f}"
                )

    # Attach pre-computed flags to the summary
    summary["precomputed_quality_flags"] = quality_flags

    # Local heuristic hint: column names that look like sensitive attributes
    plausible_sensitive = [
        col
        for col in df.columns
        if any(kw in col.lower() for kw in SENSITIVE_KEYWORDS)
    ]
    summary["plausible_sensitive_columns_hint"] = plausible_sensitive

    return summary


def _build_prompts(summary: dict) -> tuple[str, str]:
    """
    Build the (system_prompt, user_prompt) tuple from the dataset summary.
    """
    summary_json = json.dumps(summary, indent=2, default=str)

    system_prompt = textwrap.dedent(
        """\
        You are an expert ML pipeline planner.
        You will be given a compact dataset summary (no raw data rows).
        Analyse it and return a JSON object — nothing else.
        Do NOT include markdown fences, explanations, or any text outside the JSON.

        The JSON object MUST have exactly these top-level keys:
        {
          "data_quality_concerns": [
            "string describing each concern, e.g. 'High null % (42%) in column Age'"
          ],
          "recommended_preprocessing_steps": [
            "actionable step, e.g. 'Impute Age using median imputation'"
          ],
          "recommended_models": [
            "2–3 model names appropriate for the task type and data size"
          ],
          "sensitive_attribute_candidates": [
            "column names that could be sensitive attributes for fairness checking"
          ],
          "reasoning": "a short paragraph (3–5 sentences) explaining your choices"
        }

        CRITICAL RULE — data_quality_concerns:
        The summary includes a field called 'precomputed_quality_flags'.
        These flags were calculated deterministically from the data.
        You MUST include ALL of them verbatim in 'data_quality_concerns'.
        You may also add any further concerns you identify from the summary.
        An empty data_quality_concerns list is ONLY acceptable if
        precomputed_quality_flags is also empty.

        Other rules:
        - sensitive_attribute_candidates: include column names matching patterns
          like gender, sex, race, ethnicity, age, zip, religion, nationality,
          marital, disability, occupation.
        - For classification, pick 2–3 from:
            [LogisticRegression, RandomForest, XGBoost, GradientBoosting, SVM]
        - For regression, pick 2–3 from:
            [Ridge, Lasso, RandomForestRegressor, XGBoost, GradientBoostingRegressor]
          Choose based on data size and feature count.
        - Output ONLY the JSON object. No preamble, no postamble.
        """
    )

    user_prompt = textwrap.dedent(
        f"""\
        Dataset summary:
        {summary_json}

        Return the JSON plan now.
        """
    )

    return system_prompt, user_prompt


def _call_groq(
    system_prompt: str,
    user_prompt: str,
    model: str,
    client: Groq,
) -> tuple[str, dict]:
    """
    Call the Groq chat completion endpoint with JSON mode enabled.

    Returns (raw_content, token_usage). Token usage is captured rather than
    discarded because it is provenance for the AI Bill of Materials and the cost
    column of the governance evaluation — both need to know what the planner
    actually consumed.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},  # structured output — avoids fenced output
        temperature=0.2,  # low temperature for deterministic structured output
        max_tokens=2048,
    )

    usage: dict = {}
    raw_usage = getattr(response, "usage", None)
    if raw_usage is not None:
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(raw_usage, field, None)
            if value is not None:
                usage[field] = int(value)

    return response.choices[0].message.content, usage


def _parse_and_validate(raw: str) -> dict:
    """
    Parse the LLM response into a dict, strip any stray fences defensively,
    then validate that all required keys are present.
    """
    cleaned = raw.strip()

    # Defensive fence-strip (json_object mode usually prevents fences, but just in case)
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        end = -1 if lines[-1].strip() == "```" else len(lines)
        cleaned = "\n".join(lines[1:end])

    plan = json.loads(cleaned)

    missing = REQUIRED_PLAN_KEYS - set(plan.keys())
    if missing:
        raise ValueError(
            f"LLM plan is missing required keys: {missing}. "
            f"Keys present: {set(plan.keys())}"
        )

    # Coerce any accidentally stringified list fields
    for list_field in (
        "data_quality_concerns",
        "recommended_preprocessing_steps",
        "recommended_models",
        "sensitive_attribute_candidates",
    ):
        if isinstance(plan[list_field], str):
            plan[list_field] = [plan[list_field]]

    return plan


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_pipeline(
    df: pd.DataFrame,
    target_column: str,
    task_type: Literal["classification", "regression"],
    failure_context: Optional[dict] = None,
    business_objective: str = "",
    human_feedback: Optional[str] = None,
    meta_out: Optional[dict] = None,
) -> dict:
    """
    Generate a validated, JSON-serialisable pipeline plan for a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to plan for. Raw rows are NOT sent to the LLM — only
        aggregate statistics (shape, dtypes, null counts, target distribution).
    target_column : str
        Name of the target/label column in df.
    task_type : {"classification", "regression"}
        The ML task type.
    failure_context : Optional[dict], optional
        If provided (on a retry from LangGraph), this dict is appended to the
        LLM prompt so the planner can produce a genuinely revised plan.
    business_objective : str, optional
        Optional user-defined business objective or domain constraint (e.g.
        "Maximize recall on high income", "Ensure strict fairness across gender/race").
    meta_out : dict, optional
        If provided, populated in place with call provenance: the resolved
        model id, SHA-256 of the exact prompts sent, and token usage. Passed as an
        out-parameter rather than added to the return value so that the plan dict
        stays exactly what the LLM produced — provenance about the call must not
        be mistakable for content from the call.

    Returns
    -------
    dict
        Keys: data_quality_concerns, recommended_preprocessing_steps,
        recommended_models, sensitive_attribute_candidates, reasoning.
    """
    # Auto-load .env if GROQ_API_KEY is not already in environment
    _ensure_groq_api_key()

    # --- Input validation ---
    if task_type not in ("classification", "regression"):
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got {task_type!r}"
        )
    if target_column not in df.columns:
        raise ValueError(
            f"target_column {target_column!r} not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please add GROQ_API_KEY to your .env file or environment."
        )

    model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    client = Groq(api_key=api_key)

    # --- Build compact summary (no raw data) ---
    summary = _build_dataset_summary(df, target_column, task_type)
    system_prompt, user_prompt = _build_prompts(summary)

    # --- Inject optional business objective ---
    if business_objective and business_objective.strip():
        biz_note = (
            f"\nBusiness objective provided by user: {business_objective.strip()}\n"
            "Please ensure your preprocessing steps, model choices, and governance reasoning align with this objective."
        )
        system_prompt = system_prompt + biz_note

    # --- Inject explicit human feedback / directives ---
    if human_feedback and human_feedback.strip():
        fb_note = (
            f"\nEXPLICIT HUMAN REVIEWER INSTRUCTION: {human_feedback.strip()}\n"
            "The human auditor rejected the previous pipeline proposal and provided the explicit directive above. "
            "You MUST incorporate these instructions into your revised plan and preprocessing steps."
        )
        system_prompt = system_prompt + fb_note

    # --- Inject failure context on retry (backward-compatible: skipped when None) ---
    # IMPORTANT: inject into the SYSTEM prompt as a concise plain-text note —
    # NOT into user_prompt and NOT as a JSON dump. Adding large text or preamble
    # to user_prompt when json_object mode is active causes the model to generate
    # an empty string (Groq error: json_validate_failed, failed_generation: '').
    if failure_context is not None:
        source = failure_context.get("source", "data_agent")
        if source == "human_reviewer":
            reason = failure_context.get("reason", "Human auditor rejected the previous data quality outcome.")
            retry_note = (
                f"\nRETRY NOTE: A human auditor reviewed your previous plan's outcome and REJECTED it for data quality reasons: '{reason}'. "
                "Please revise recommended_preprocessing_steps to address the human auditor's feedback. "
                "Output ONLY valid JSON as usual."
            )
        else:
            rows_pct = failure_context.get("rows_dropped_pct", "?")
            cols = failure_context.get("columns_dropped", [])
            missing = failure_context.get("missing_pct_after_cleaning", "?")
            unhandled = failure_context.get("unhandled_high_null_columns", [])

            details = []
            if unhandled:
                details.append(f"unhandled high-null feature columns: {unhandled}")
            if rows_pct != "?" and float(rows_pct) > 0:
                details.append(f"rows_dropped_pct={rows_pct}%")
            if missing != "?" and float(missing) > 0:
                details.append(f"missing_pct_after_cleaning={missing}%")

            failure_str = "; ".join(details) if details else f"columns_dropped={cols}"
            retry_note = (
                f"\nRETRY NOTE: Your previous plan caused a quality failure ({failure_str}). "
                "Please revise recommended_preprocessing_steps to address these issues "
                "(e.g., explicitly instruct to drop high-null feature columns like 'Drop column <col>'). "
                "Output ONLY valid JSON as usual."
            )
        system_prompt = system_prompt + retry_note

    # --- LLM call with one retry on parse failure ---
    raw: str | None = None
    last_error: Exception | None = None

    for attempt in range(1, 3):  # attempts 1 and 2
        try:
            raw, usage = _call_groq(system_prompt, user_prompt, model, client)
            plan = _parse_and_validate(raw)
            if meta_out is not None:
                meta_out.update({
                    "model_id": model,
                    # Hash the prompts actually sent, after every injection
                    # (business objective, human feedback, retry note), so the
                    # digest identifies this call and not just the template.
                    "system_prompt_sha256": hashlib.sha256(
                        system_prompt.encode("utf-8")).hexdigest(),
                    "user_prompt_sha256": hashlib.sha256(
                        user_prompt.encode("utf-8")).hexdigest(),
                    "token_usage": usage,
                    "attempts": attempt,
                })
            return plan
        except Exception as exc:
            last_error = exc
            if attempt == 1:
                print(
                    f"[planner_agent] Attempt {attempt} error: {exc}. "
                    "Retrying..."
                )

    raise ValueError(
        f"plan_pipeline() failed to obtain a valid JSON plan from the LLM "
        f"after 2 attempts.\n"
        f"  Model used : {model}\n"
        f"  Last error : {last_error}\n"
        f"  Last raw   : {raw!r}"
    )
