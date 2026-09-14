"""
fairness_agent.py
=================
Step 4 of the AI-Governed Multi-Agent Platform.

Exposes a single public function:
    run_fairness_agent(cleaned_df, fitted_model, target_column,
                       sensitive_attribute_candidates, task_type) -> dict

DESIGN PRINCIPLES:
  - Zero LLM calls. Fully deterministic.
  - Computes standard algorithmic fairness metrics (Disparate Impact and
    Demographic Parity Difference) across sensitive attributes.
  - Reconstructs One-Hot Encoded (OHE) categorical attributes created by
    Data Agent by prefix matching (e.g. 'sex_Male', 'sex_Female' -> 'sex').
  - Evaluates multi-group disparities using Max-vs-Min (worst-case ratio).
  - Flags violation if Disparate Impact < 0.80 (80% rule) or
    Demographic Parity Difference > 0.10.

TASK TYPE COVERAGE:
  - Classification: Fully supported.
  - Regression: NOT EVALUATED. Disparate Impact and Demographic Parity Difference
    are defined over a binary positive-prediction rate and have no meaning for a
    continuous output. Regression runs therefore return
    overall_fairness_passed=None with fairness_evaluated=False - deliberately NOT
    True, so that a governance reviewer is never shown a green "fairness passed"
    badge for a property that was never measured.

EVALUATION SET:
  Metrics are computed on the held-out test rows when eval_index is supplied by
  the caller. Measuring subgroup rates on rows the model was fitted on flatters
  the model, because a sufficiently flexible model reproduces the training
  distribution's subgroup rates by construction.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level thresholds
# ---------------------------------------------------------------------------

DISPARATE_IMPACT_THRESHOLD = 0.80        # standard 80% rule (4/5ths rule)
DEMOGRAPHIC_PARITY_DIFF_THRESHOLD = 0.10 # max 10% rate gap allowed


def _proxy_warnings(
    proxy_findings: list[dict] | None,
    candidates: list[str] | None,
    evaluated: list[str] | None,
) -> list[dict]:
    """
    Surface EDA proxy-variable findings alongside the fairness audit.

    Deliberately INFORMATIONAL: a proxy warning never changes
    overall_fairness_passed. The verdict is a measurement of this model's
    outcomes; a proxy is a property of the data that explains why a disparity can
    persist. Folding one into the other would make the verdict mean two things.
    """
    audited = {str(c).strip() for c in (candidates or [])} | set(evaluated or [])
    warnings = []
    for finding in proxy_findings or []:
        cols = finding.get("columns") or []
        if finding.get("type") != "proxy_variable" or len(cols) < 2:
            continue
        warnings.append({
            "id": finding["id"],
            "proxy": cols[0],
            "protected_attribute": cols[1],
            "metric": finding.get("metric"),
            "severity": finding.get("severity"),
            "protected_attribute_audited": cols[1] in audited,
        })
    return warnings


# ---------------------------------------------------------------------------
# Group Reconstruction Helpers
# ---------------------------------------------------------------------------


def _extract_attribute_series(
    df: pd.DataFrame,
    attribute_candidate: str,
    target_column: str,
) -> tuple[pd.Series | None, str | None]:
    """
    Extract or reconstruct a categorical Series for a sensitive attribute candidate.

    Returns (series, error_reason).
    If series is returned, error_reason is None.
    If error_reason is returned, series is None.
    """
    # Case 1: Attribute exists verbatim as a column in df
    if attribute_candidate in df.columns:
        col_data = df[attribute_candidate]

        # If it's a numeric float (continuous / scaled), skip
        # KNOWN LIMITATION: Continuous numeric attributes (e.g., 'age', 'income', 'hours')
        # are currently skipped because fairness metrics (disparate impact / demographic parity)
        # require discrete group boundaries. Automatically bucketing continuous features into
        # ranges (e.g. age: <25, 25–60, >60) is a natural Phase 2 extension.
        if pd.api.types.is_float_dtype(col_data):
            return None, f"'{attribute_candidate}' is a continuous numeric feature (requires explicit bucketing/binarization threshold, e.g. <25/25-60/>60)"

        # If it's object / categorical / int / bool with moderate cardinality, use directly
        nunique = col_data.nunique()
        if nunique < 2:
            return None, f"'{attribute_candidate}' has fewer than 2 unique values ({nunique})"
        if nunique > 20:
            return None, f"'{attribute_candidate}' has too many unique categories ({nunique})"

        return col_data.astype(str), None

    # Case 2: Attribute was One-Hot Encoded by Data Agent into columns like attr_val1, attr_val2...
    # Look for matching dummy columns starting with prefix "<candidate>_" or "<candidate>-"
    prefix_underscore = f"{attribute_candidate}_"
    prefix_dash = f"{attribute_candidate}-"

    matching_cols = [
        c for c in df.columns
        if (c.startswith(prefix_underscore) or c.startswith(prefix_dash))
        and c != target_column
    ]

    if not matching_cols:
        return None, f"'{attribute_candidate}' not found in dataset"

    # Reconstruct single categorical series by taking idxmax across matching dummy columns
    dummy_df = df[matching_cols]

    # Clean group label by removing the prefix and leading/trailing whitespace
    def _clean_group_name(col_name: str) -> str:
        if col_name.startswith(prefix_underscore):
            raw = col_name[len(prefix_underscore):]
        elif col_name.startswith(prefix_dash):
            raw = col_name[len(prefix_dash):]
        else:
            raw = col_name
        return raw.strip()

    col_map = {col: _clean_group_name(col) for col in matching_cols}
    renamed_dummy = dummy_df.rename(columns=col_map)

    # For rows where all dummy columns are 0 (e.g. if one category was dropped or zeroed), mark as 'Other'
    row_sums = renamed_dummy.sum(axis=1)
    reconstructed = renamed_dummy.idxmax(axis=1)
    reconstructed[row_sums == 0] = "Other"

    nunique = reconstructed.nunique()
    if nunique < 2:
        return None, f"Reconstructed '{attribute_candidate}' has fewer than 2 unique groups ({nunique})"

    return reconstructed, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_fairness_agent(
    cleaned_df: pd.DataFrame,
    fitted_model: Any,
    target_column: str,
    sensitive_attribute_candidates: list[str],
    task_type: str,
    eval_index: list | None = None,
    proxy_findings: list | None = None,
) -> dict:
    """
    Evaluate algorithmic fairness across sensitive attribute candidates.
    Zero LLM calls. Fully deterministic.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Fully cleaned/encoded dataset from run_data_agent().
    fitted_model : Any
        Fitted model instance (e.g. XGBClassifier, RandomForestClassifier).
    target_column : str
        Name of the target column in cleaned_df.
    sensitive_attribute_candidates : list[str]
        Attribute names recommended by Planner Agent (e.g. ["age", "sex", "race"]).
    task_type : {"classification", "regression"}
    eval_index : list, optional
        Index labels of the held-out test rows (from run_data_agent()). When
        supplied, metrics are computed on these rows only. When omitted, metrics
        are computed on every row including those the model trained on, and a
        warning is recorded in actions_taken.

    Returns
    -------
    dict:
        fairness_report         : list[dict] - metrics per evaluated attribute
        overall_fairness_passed : bool | None - False if ANY evaluated attribute
                                  violates thresholds; None when fairness was not
                                  evaluated at all (e.g. regression)
        fairness_evaluated      : bool - whether any metric was actually computed
        attributes_skipped      : list[str] - candidate names skipped with reasons
        actions_taken           : list[str] - human-readable log entries
    """
    if task_type == "regression":
        # NOT "passed". Disparate Impact and Demographic Parity Difference are
        # undefined for a continuous target, so nothing was measured. Reporting
        # True here would show a governance reviewer a green badge for a check
        # that never ran.
        return {
            "overall_fairness_passed": None,
            "fairness_evaluated": False,
            "proxy_warnings": _proxy_warnings(
                proxy_findings, sensitive_attribute_candidates, []),
            "fairness_report": [],
            "attributes_skipped": [
                "All attributes (regression task - Disparate Impact and "
                "Demographic Parity Difference are defined only for classification)"
            ],
            "actions_taken": [
                "NOT EVALUATED: regression task. Disparate Impact and Demographic "
                "Parity Difference require a binary positive-prediction rate and "
                "have no definition for a continuous output. No fairness "
                "conclusion can be drawn about this model."
            ],
        }

    if task_type != "classification":
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got {task_type!r}"
        )

    if target_column not in cleaned_df.columns:
        raise ValueError(
            f"target_column '{target_column}' not found in cleaned_df. "
            f"Available: {list(cleaned_df.columns)}"
        )

    actions: list[str] = []

    # Restrict to the held-out rows when the caller supplied them. Subgroup rates
    # measured on training rows understate disparity.
    if eval_index is not None:
        eval_idx = pd.Index(eval_index)
        unknown = len(eval_idx.difference(cleaned_df.index))
        if unknown:
            raise ValueError(
                f"Fairness Agent: {unknown} label(s) in eval_index are absent "
                f"from cleaned_df."
            )
        eval_df = cleaned_df.loc[eval_idx]
        actions.append(
            f"Evaluating fairness on the {len(eval_df):,}-row held-out test split "
            f"(the model was not fitted on these rows)"
        )
    else:
        eval_df = cleaned_df
        actions.append(
            f"WARNING: no held-out split supplied - evaluating fairness on all "
            f"{len(eval_df):,} rows, including rows the model trained on. "
            f"Measured disparity is likely understated."
        )

    # Prepare feature matrix X for prediction (exclude target column)
    X = eval_df.drop(columns=[target_column])

    # Convert boolean columns to int for model prediction compatibility
    bool_cols = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]
    if bool_cols:
        X = X.copy()
        X[bool_cols] = X[bool_cols].astype(int)

    # Generate predictions across the full cleaned dataset
    try:
        predictions = fitted_model.predict(X)
    except Exception as exc:
        raise RuntimeError(
            f"Fairness Agent: model prediction failed on cleaned_df: {exc}"
        ) from exc

    actions.append(
        f"Generated model predictions for {len(eval_df):,} rows using {type(fitted_model).__name__}"
    )

    fairness_report: list[dict] = []
    attributes_skipped: list[str] = []
    seen_candidates: set[str] = set()

    for candidate in sensitive_attribute_candidates:
        cand_clean = candidate.strip()
        if cand_clean in seen_candidates:
            continue
        seen_candidates.add(cand_clean)

        # Extract/reconstruct group Series for this candidate
        group_series, skip_reason = _extract_attribute_series(
            df=eval_df,
            attribute_candidate=cand_clean,
            target_column=target_column,
        )

        if group_series is None:
            skipped_msg = f"{cand_clean} ({skip_reason})"
            attributes_skipped.append(skipped_msg)
            actions.append(f"Skipped candidate '{cand_clean}': {skip_reason}")
            continue

        # Compute positive prediction rate for each group
        df_group = pd.DataFrame({"group": group_series, "pred": predictions})
        group_rates: dict[str, float] = {}
        group_counts: dict[str, int] = {}

        for grp_name, grp_data in df_group.groupby("group"):
            count = len(grp_data)
            if count == 0:
                continue
            pos_rate = float(grp_data["pred"].mean())
            group_rates[str(grp_name)] = round(pos_rate, 4)
            group_counts[str(grp_name)] = count

        if len(group_rates) < 2:
            reason = f"Fewer than 2 valid sub-groups with data (found: {list(group_rates.keys())})"
            attributes_skipped.append(f"{cand_clean} ({reason})")
            actions.append(f"Skipped candidate '{cand_clean}': {reason}")
            continue

        # Identify Group Max (highest positive rate) and Group Min (lowest positive rate)
        sorted_groups = sorted(group_rates.items(), key=lambda x: x[1], reverse=True)
        group_max_name, max_rate = sorted_groups[0]
        group_min_name, min_rate = sorted_groups[-1]

        # Compute Disparate Impact and Demographic Parity Difference
        if max_rate > 0:
            disparate_impact = round(min_rate / max_rate, 4)
        else:
            disparate_impact = 1.0  # if max_rate == 0, both rates are 0 -> perfect parity

        demographic_parity_diff = round(max_rate - min_rate, 4)

        # Violation check
        di_violation = disparate_impact < DISPARATE_IMPACT_THRESHOLD
        dpd_violation = demographic_parity_diff > DEMOGRAPHIC_PARITY_DIFF_THRESHOLD
        violation = di_violation or dpd_violation

        status_icon = "❌ VIOLATION" if violation else "✅ PASSED"
        actions.append(
            f"Evaluated '{cand_clean}' [{status_icon}]: "
            f"Disparate Impact={disparate_impact:.4f} (threshold {DISPARATE_IMPACT_THRESHOLD}), "
            f"Demographic Parity Diff={demographic_parity_diff:.4f} (threshold {DEMOGRAPHIC_PARITY_DIFF_THRESHOLD}) "
            f"[Max group '{group_max_name}': {max_rate:.4f}, Min group '{group_min_name}': {min_rate:.4f}]"
        )

        fairness_report.append({
            "attribute": cand_clean,
            "disparate_impact": disparate_impact,
            "demographic_parity_difference": demographic_parity_diff,
            "violation": violation,
            "group_details": {
                "group_a": group_max_name,
                "group_a_positive_rate": max_rate,
                "group_b": group_min_name,
                "group_b_positive_rate": min_rate,
            },
        })

    # Overall fairness passed if no evaluated attribute has a violation.
    # If every candidate was skipped, nothing was measured - report None rather
    # than True, so an unevaluated run is never displayed as a pass.
    if not fairness_report:
        overall_passed = None
        actions.append(
            "NOT EVALUATED: no sensitive attribute candidate could be resolved to "
            "a usable subgroup column. No fairness conclusion can be drawn."
        )
    else:
        overall_passed = not any(r["violation"] for r in fairness_report)

    proxy_warnings = _proxy_warnings(
        proxy_findings,
        sensitive_attribute_candidates,
        [r["attribute"] for r in fairness_report],
    )
    for warning in proxy_warnings:
        metric = warning.get("metric") or {}
        actions.append(
            f"PROXY WARNING: exploratory analysis found '{warning['proxy']}' "
            f"associated with '{warning['protected_attribute']}' "
            f"({metric.get('name')} {metric.get('value')}). Disparities by "
            f"'{warning['protected_attribute']}' can persist through "
            f"'{warning['proxy']}' even if '{warning['protected_attribute']}' were "
            f"removed. Informational: the verdict above is unchanged."
        )

    return {
        "proxy_warnings": proxy_warnings,
        "fairness_report": fairness_report,
        "overall_fairness_passed": overall_passed,
        "fairness_evaluated": bool(fairness_report),
        "evaluated_rows": int(len(eval_df)),
        "attributes_skipped": attributes_skipped,
        "actions_taken": actions,
    }
