"""
fairness_agent.py
=================
Step 4 of the AI-Governed Multi-Agent Platform.

Exposes a single public function:
    run_fairness_agent(cleaned_df, fitted_model, target_column,
                       sensitive_attribute_candidates, task_type, ...) -> dict

DESIGN PRINCIPLES:
  - Zero LLM calls. Fully deterministic.
  - Verdict: an attribute is violated when Disparate Impact < 0.80 (the
    four-fifths rule) or Demographic Parity Difference > 0.10, comparing the
    groups with the highest and lowest positive-prediction rate.
  - Equal-opportunity and equalized-odds differences are REPORTED alongside the
    verdict but do not change it. What counts as a violation is a policy decision
    (NEXT_STEPS Step 4), not something to change silently inside a metric.

WHICH ROWS AND WHICH GROUPS. Each rule below fixes a defect that the governance
evaluation (experiments/run_governance_eval.py) exposed:

  Evaluation rows.
    The held-out test rows, when eval_index is supplied. Subgroup rates measured on
    training rows flatter the model.

  Group membership comes from the RAW uploaded values when raw_frame is supplied.
    The cleaned frame is a poor record of who belongs to which group. A binary 0/1
    attribute such as COMPAS `sex` is standardised into a float; a categorical with
    ten or more values, such as Adult `occupation`, is frequency-encoded into a
    float. Both used to be skipped as "continuous" and were never audited. Raw
    values fix both, keep readable labels ("<0", not "lt0"), and show missing
    values as their own group rather than as the imputed mode. Without raw_frame,
    groups are reconstructed from the cleaned frame (verbatim column or one-hot
    prefix), as before.

  Age is audited in bands.
    A continuous numeric attribute named like age is grouped into AGE_BANDS. Other
    continuous attributes have no defensible banding rule and are skipped with that
    reason. Age was proposed by the planner on every benchmark and, before this,
    was never once audited.

  Small groups are excluded from the comparison.
    Groups with fewer than MIN_GROUP_SIZE evaluation rows are left out of the
    max-vs-min comparison and listed in excluded_groups. On UCI Adult a group of 4
    people ("Married-AF-spouse") with no positive predictions set the minimum
    disparate impact to 0.0 and made the headline number meaningless.

  Protected attributes are always audited.
    Columns whose names match the protected-attribute keywords in eda_insights are
    added when the planner did not propose them: whether sex is audited must not
    depend on an LLM remembering to mention it. Each report entry records whether
    it came from the planner or was added automatically.

  The verdict covers protected attributes only.
    Protected means named like one (eda_insights.is_protected_attribute) or declared by
    the reviewer at run start (declared_protected). Other attributes the planner
    proposes, such as occupation or job, are still audited and reported, but their
    violations are ADVISORY (counts_toward_verdict False, listed in advisory_violations):
    anti-discrimination rules such as the four-fifths rule are defined over protected
    characteristics. With no protected attribute evaluated, the verdict is None and
    fairness_evaluated is False.

  Intersectional subgroups are reported, not judged.
    Pairs of evaluated protected attributes (sex × race) are audited together with
    the same group rules, because a model can treat every group of each attribute
    alike and still treat some combinations very differently. Results go to
    intersectional_report and never change the verdict — the same rule as the
    error-rate gaps. Whether they should count is a policy decision (Step 4).

  An unaudited protected attribute blocks a pass.
    If a protected attribute present in the data is skipped (for example, its age
    bands are all under MIN_GROUP_SIZE) and no audited attribute is violated, the
    verdict is None (NOT FULLY EVALUATED), not True. On German Credit one split
    "passed" on `job` alone while age went unaudited. A measured violation still
    yields False. fairness_coverage is "complete", "partial" or "none".

TASK TYPE COVERAGE:
  - Classification: supported. Error-rate metrics additionally need a binary target.
  - Regression: NOT EVALUATED. overall_fairness_passed is None, never True, so a
    reviewer is never shown a pass for a property that was never measured.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from eda_insights import is_protected_attribute

# ---------------------------------------------------------------------------
# Thresholds and grouping rules
# ---------------------------------------------------------------------------

DISPARATE_IMPACT_THRESHOLD = 0.80          # four-fifths rule
DEMOGRAPHIC_PARITY_DIFF_THRESHOLD = 0.10   # max 10-point rate gap

MIN_GROUP_SIZE = 30                        # evaluation rows a group needs to be compared
MIN_CLASS_ROWS_FOR_ERROR_RATES = 10        # true positives / negatives needed for TPR / FPR
MAX_GROUPS = 20                            # more categories than this are not compared
MAX_INTERSECTIONS = 10                     # pairs of protected attributes audited together
INTERSECTION_SEPARATOR = " × "
MISSING_GROUP = "(missing)"

# [lower, upper) bounds and the label a reviewer reads.
AGE_BANDS = ((None, 25, "<25"), (25, 60, "25-59"), (60, None, "60+"))
AGE_BAND_DESCRIPTION = "<25, 25-59, 60+"


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t]


def _is_age_attribute(name: str) -> bool:
    return "age" in _name_tokens(name)


def _attribute_present(attribute: str, raw_eval: pd.DataFrame | None,
                       eval_df: pd.DataFrame, target_column: str) -> bool:
    """Whether the attribute exists in the data, verbatim or as one-hot columns."""
    if raw_eval is not None and attribute in raw_eval.columns:
        return True
    if attribute in eval_df.columns:
        return True
    return any((c.startswith(f"{attribute}_") or c.startswith(f"{attribute}-"))
               and c != target_column for c in eval_df.columns)


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
# Group resolution
# ---------------------------------------------------------------------------


def _groups_from_raw(
    attribute: str,
    raw_eval: pd.DataFrame,
) -> tuple[pd.Series | None, str | None, str | None]:
    """Return (groups, grouping description, skip reason) from raw uploaded values."""
    series = raw_eval[attribute]
    non_null = series.dropna()
    if non_null.empty:
        return None, None, f"'{attribute}' has no non-missing values in the evaluation rows"

    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    distinct = int(non_null.nunique())

    if numeric and distinct > MAX_GROUPS:
        if not _is_age_attribute(attribute):
            return None, None, (
                f"'{attribute}' is continuous ({distinct} distinct values); only age "
                f"has a banding rule, so it is not audited"
            )
        values = pd.to_numeric(series, errors="coerce")
        groups = pd.Series(MISSING_GROUP, index=series.index, dtype=object)
        for lower, upper, label in AGE_BANDS:
            mask = values.notna()
            if lower is not None:
                mask &= values >= lower
            if upper is not None:
                mask &= values < upper
            groups[mask] = label
        return groups, f"raw values, banded ({AGE_BAND_DESCRIPTION})", None

    if distinct < 2:
        return None, None, f"'{attribute}' has fewer than 2 distinct values ({distinct})"
    if distinct > MAX_GROUPS:
        return None, None, (
            f"'{attribute}' has too many categories to compare ({distinct} > {MAX_GROUPS})"
        )
    groups = series.map(lambda v: MISSING_GROUP if pd.isna(v) else str(v))
    return groups.astype(object), "raw values", None


def resolve_groups_from_raw(
    attribute: str,
    raw_rows: pd.DataFrame,
) -> tuple[pd.Series | None, str | None, str | None]:
    """
    Public entry to the audit's grouping rules (raw values, age bands, a missing
    group). Mitigation uses it so the attribute it reweights is grouped exactly as
    the attribute that was audited.
    """
    return _groups_from_raw(attribute, raw_rows)


def _groups_from_cleaned(
    attribute: str,
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.Series | None, str | None, str | None]:
    """Fallback without raw values: verbatim cleaned column or one-hot reconstruction."""
    if attribute in df.columns:
        column = df[attribute]
        if pd.api.types.is_float_dtype(column):
            return None, None, (
                f"'{attribute}' is numeric in the cleaned data (it may have been scaled "
                f"or frequency-encoded) and no raw values were supplied to recover its groups"
            )
        distinct = column.nunique()
        if distinct < 2:
            return None, None, f"'{attribute}' has fewer than 2 distinct values ({distinct})"
        if distinct > MAX_GROUPS:
            return None, None, (
                f"'{attribute}' has too many categories to compare ({distinct} > {MAX_GROUPS})"
            )
        return column.astype(str), "cleaned column values", None

    prefix_underscore = f"{attribute}_"
    prefix_dash = f"{attribute}-"
    matching = [
        c for c in df.columns
        if (c.startswith(prefix_underscore) or c.startswith(prefix_dash)) and c != target_column
    ]
    if not matching:
        return None, None, f"'{attribute}' not found in dataset"

    def _label(col: str) -> str:
        for prefix in (prefix_underscore, prefix_dash):
            if col.startswith(prefix):
                return col[len(prefix):].strip()
        return col

    dummies = df[matching].rename(columns={c: _label(c) for c in matching})
    reconstructed = dummies.idxmax(axis=1).astype(object)
    reconstructed[dummies.sum(axis=1) == 0] = "Other"
    if reconstructed.nunique() < 2:
        return None, None, (
            f"reconstructed '{attribute}' has fewer than 2 groups ({reconstructed.nunique()})"
        )
    return reconstructed, "reconstructed from one-hot columns", None


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
    raw_frame: pd.DataFrame | None = None,
    min_group_size: int = MIN_GROUP_SIZE,
    declared_protected: list | None = None,
) -> dict:
    """
    Evaluate algorithmic fairness across sensitive attributes.
    Zero LLM calls. Fully deterministic.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Fully cleaned/encoded dataset from run_data_agent().
    fitted_model : Any
        Fitted model instance.
    target_column : str
        Name of the target column in cleaned_df.
    sensitive_attribute_candidates : list[str]
        Attribute names proposed by the Planner Agent.
    task_type : {"classification", "regression"}
    eval_index : list, optional
        Index labels of the held-out test rows. Without it, metrics are computed on
        every row, including those the model trained on, and a warning is recorded.
    proxy_findings : list, optional
        Reviewer-routed EDA findings, surfaced as proxy warnings.
    raw_frame : pd.DataFrame, optional
        The raw uploaded frame, sharing cleaned_df's index labels. When supplied,
        group membership is read from raw values and protected attributes present in
        it are audited even if the planner did not propose them.
    min_group_size : int
        Groups with fewer evaluation rows than this are excluded from comparison.

    Returns
    -------
    dict:
        fairness_report         : list[dict] - metrics per evaluated attribute
        overall_fairness_passed : bool | None - None when nothing was evaluated
        fairness_evaluated      : bool
        attributes_skipped      : list[str] - attributes not audited, with reasons
        proxy_warnings          : list[dict]
        evaluated_rows          : int
        min_group_size          : int
        actions_taken           : list[str]
    """
    if task_type == "regression":
        # NOT "passed": the metrics are undefined for a continuous target.
        return {
            "overall_fairness_passed": None,
            "fairness_evaluated": False,
            "fairness_coverage": "none",
            "protected_attributes_unaudited": [],
            "advisory_violations": [],
            "declared_protected_attributes": list(declared_protected or []),
            "intersectional_report": [],
            "intersections_skipped": [],
            "proxy_warnings": _proxy_warnings(
                proxy_findings, sensitive_attribute_candidates, []),
            "fairness_report": [],
            "min_group_size": min_group_size,
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

    if eval_index is not None:
        eval_idx = pd.Index(eval_index)
        unknown = len(eval_idx.difference(cleaned_df.index))
        if unknown:
            raise ValueError(
                f"Fairness Agent: {unknown} label(s) in eval_index are absent from cleaned_df."
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

    raw_eval = None
    if raw_frame is not None:
        missing = len(eval_df.index.difference(raw_frame.index))
        if missing:
            raise ValueError(
                f"Fairness Agent: {missing} evaluation row(s) are absent from raw_frame; "
                f"the raw and cleaned frames must share index labels."
            )
        raw_eval = raw_frame.loc[eval_df.index]
        actions.append("Group membership read from raw uploaded values")

    X = eval_df.drop(columns=[target_column])
    bool_cols = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]
    if bool_cols:
        X = X.copy()
        X[bool_cols] = X[bool_cols].astype(int)
    try:
        predictions = fitted_model.predict(X)
    except Exception as exc:
        raise RuntimeError(f"Fairness Agent: model prediction failed: {exc}") from exc
    actions.append(
        f"Generated model predictions for {len(eval_df):,} rows using "
        f"{type(fitted_model).__name__}"
    )

    y_true = eval_df[target_column]
    observed_classes = set(pd.unique(y_true.dropna()))
    binary_target = len(observed_classes) == 2 and observed_classes <= {0, 1}

    declared = [str(c).strip() for c in (declared_protected or []) if str(c).strip()]

    def _protected(name: str) -> bool:
        return is_protected_attribute(name, declared)

    # Planner candidates first, then reviewer-declared attributes, then protected
    # attributes detected by name that neither named.
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate in sensitive_attribute_candidates or []:
        name = str(candidate).strip()
        if name and name not in seen:
            candidates.append((name, "planner"))
            seen.add(name)
    for name in declared:
        if name != target_column and name not in seen:
            candidates.append((name, "declared"))
            seen.add(name)
            actions.append(f"Added '{name}' to the audit: declared protected by the reviewer")
    if raw_eval is not None:
        for column in raw_eval.columns:
            if column != target_column and column not in seen and _protected(column):
                candidates.append((column, "auto"))
                seen.add(column)
                actions.append(
                    f"Added protected attribute '{column}' to the audit: present in the "
                    f"data but not proposed by the planner"
                )

    fairness_report: list[dict] = []
    attributes_skipped: list[str] = []
    protected_unaudited: list[dict] = []
    # Group label per evaluation row for each evaluated protected attribute, kept for
    # the intersectional comparison.
    resolved_groups: dict[str, pd.Series] = {}

    def _skip(attribute: str, reason: str) -> None:
        attributes_skipped.append(f"{attribute} ({reason})")
        actions.append(f"Skipped '{attribute}': {reason}")
        # A planner-named column that does not exist is not a coverage gap: there is
        # no group in the data to be unfair to.
        if _protected(attribute) and _attribute_present(
                attribute, raw_eval, eval_df, target_column):
            protected_unaudited.append({"attribute": attribute, "reason": reason})

    for attribute, source in candidates:
        if raw_eval is not None and attribute in raw_eval.columns:
            groups, grouping, reason = _groups_from_raw(attribute, raw_eval)
        else:
            groups, grouping, reason = _groups_from_cleaned(attribute, eval_df, target_column)

        if groups is None:
            _skip(attribute, reason)
            continue

        frame = pd.DataFrame({
            "group": groups.to_numpy(),
            "pred": predictions,
            "y": y_true.to_numpy(),
        })
        sizes = frame.groupby("group").size().sort_values(ascending=False)
        eligible = sizes[sizes >= min_group_size]
        excluded = {str(k): int(v) for k, v in sizes[sizes < min_group_size].items()}

        if len(eligible) < 2:
            listed = ", ".join(f"{k}={int(v)}" for k, v in sizes.items())
            _skip(attribute, (f"fewer than 2 groups with at least {min_group_size} "
                              f"evaluation rows (group sizes: {listed})"))
            continue

        groups_detail: dict[str, dict] = {}
        for name in eligible.index:
            rows = frame[frame["group"] == name]
            detail = {
                "n": int(len(rows)),
                "positive_rate": round(float(rows["pred"].mean()), 4),
                "tpr": None,
                "fpr": None,
            }
            if binary_target:
                positives = rows[rows["y"] == 1]
                negatives = rows[rows["y"] == 0]
                if len(positives) >= MIN_CLASS_ROWS_FOR_ERROR_RATES:
                    detail["tpr"] = round(float(positives["pred"].mean()), 4)
                if len(negatives) >= MIN_CLASS_ROWS_FOR_ERROR_RATES:
                    detail["fpr"] = round(float(negatives["pred"].mean()), 4)
            groups_detail[str(name)] = detail

        ranked = sorted(groups_detail.items(), key=lambda kv: kv[1]["positive_rate"],
                        reverse=True)
        (max_name, max_detail), (min_name, min_detail) = ranked[0], ranked[-1]
        max_rate, min_rate = max_detail["positive_rate"], min_detail["positive_rate"]
        disparate_impact = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0
        parity_difference = round(max_rate - min_rate, 4)

        def _gap(key: str):
            values = [d[key] for d in groups_detail.values() if d[key] is not None]
            return round(max(values) - min(values), 4) if len(values) >= 2 else None

        equal_opportunity = _gap("tpr") if binary_target else None
        fpr_gap = _gap("fpr") if binary_target else None
        equalized_odds = (round(max(equal_opportunity, fpr_gap), 4)
                          if equal_opportunity is not None and fpr_gap is not None else None)

        violation = (disparate_impact < DISPARATE_IMPACT_THRESHOLD
                     or parity_difference > DEMOGRAPHIC_PARITY_DIFF_THRESHOLD)

        protected = _protected(attribute)
        status = ("VIOLATION" if violation else "passed") if protected else (
            "advisory violation, not a protected attribute" if violation else
            "passed, advisory")
        excluded_note = (f"; excluded groups under {min_group_size} rows: "
                         + ", ".join(f"{k} ({v})" for k, v in excluded.items())
                         if excluded else "")
        eo_note = (f", equal-opportunity difference {equal_opportunity:.4f}"
                   if equal_opportunity is not None else "")
        actions.append(
            f"Evaluated '{attribute}' [{status}] "
            f"({grouping}; {source}): disparate impact {disparate_impact:.4f}, "
            f"parity difference {parity_difference:.4f}{eo_note} - highest "
            f"'{max_name}' {max_rate:.4f} (n={max_detail['n']}), lowest '{min_name}' "
            f"{min_rate:.4f} (n={min_detail['n']}){excluded_note}"
        )

        if protected:
            resolved_groups[attribute] = frame["group"].astype(str)
        fairness_report.append({
            "attribute": attribute,
            "protected": protected,
            # The verdict covers protected attributes only; the rest are advisory.
            "counts_toward_verdict": protected,
            "source": source,
            "grouping": grouping,
            "disparate_impact": disparate_impact,
            "demographic_parity_difference": parity_difference,
            "equal_opportunity_difference": equal_opportunity,
            "equalized_odds_difference": equalized_odds,
            "violation": violation,
            "groups": groups_detail,
            "excluded_groups": excluded,
            "group_details": {
                "group_a": max_name,
                "group_a_positive_rate": max_rate,
                "group_a_n": max_detail["n"],
                "group_b": min_name,
                "group_b_positive_rate": min_rate,
                "group_b_n": min_detail["n"],
            },
        })

    verdict_entries = [r for r in fairness_report if r["counts_toward_verdict"]]
    advisory_violations = [r["attribute"] for r in fairness_report
                           if r["violation"] and not r["counts_toward_verdict"]]
    if not verdict_entries:
        overall_passed, coverage = None, "none"
        actions.append(
            "NOT EVALUATED: no protected attribute could be resolved into at least two "
            "comparable groups. No fairness conclusion can be drawn"
            + (f"; audited attributes that are not protected are advisory only "
               f"({', '.join(r['attribute'] for r in fairness_report)})."
               if fairness_report else ".")
        )
    else:
        coverage = "partial" if protected_unaudited else "complete"
        if any(r["violation"] for r in verdict_entries):
            # A measured violation is a finding whatever else went unmeasured.
            overall_passed = False
        elif protected_unaudited:
            # Not a pass: the protected attributes that were audited cleared, but another
            # protected attribute present in the data was not measured at all.
            overall_passed = None
            actions.append(
                "NOT FULLY EVALUATED: no violation among the audited protected attributes, "
                "but protected attribute(s) present in the data could not be audited: "
                + "; ".join(f"'{p['attribute']}' ({p['reason']})" for p in protected_unaudited)
                + ". No overall pass can be recorded."
            )
        else:
            overall_passed = True
    if advisory_violations:
        actions.append(
            "ADVISORY: violations on attributes that are not protected do not affect the "
            "verdict: " + ", ".join(advisory_violations)
        )

    # Intersectional subgroups: reported only, never part of the verdict.
    intersectional_report: list[dict] = []
    intersections_skipped: list[str] = []
    names = [r["attribute"] for r in verdict_entries]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]][:MAX_INTERSECTIONS]
    for first, second in pairs:
        name = f"{first}{INTERSECTION_SEPARATOR}{second}"
        cross = pd.DataFrame({
            "group": (resolved_groups[first] + INTERSECTION_SEPARATOR
                      + resolved_groups[second]).to_numpy(),
            "pred": predictions,
        })
        sizes = cross.groupby("group").size().sort_values(ascending=False)
        eligible = sizes[sizes >= min_group_size]
        excluded = {str(k): int(v) for k, v in sizes[sizes < min_group_size].items()}
        if len(eligible) < 2:
            listed = ", ".join(f"{k}={int(v)}" for k, v in sizes.items())
            intersections_skipped.append(
                f"{name} (fewer than 2 combinations with at least {min_group_size} "
                f"evaluation rows: {listed})")
            continue
        rates = {
            str(g): {"n": int(sizes[g]),
                     "positive_rate": round(float(cross.loc[cross["group"] == g, "pred"].mean()), 4)}
            for g in eligible.index
        }
        ranked = sorted(rates.items(), key=lambda kv: kv[1]["positive_rate"], reverse=True)
        (high, high_d), (low, low_d) = ranked[0], ranked[-1]
        di = (round(low_d["positive_rate"] / high_d["positive_rate"], 4)
              if high_d["positive_rate"] > 0 else 1.0)
        dpd = round(high_d["positive_rate"] - low_d["positive_rate"], 4)
        gap = di < DISPARATE_IMPACT_THRESHOLD or dpd > DEMOGRAPHIC_PARITY_DIFF_THRESHOLD
        intersectional_report.append({
            "attribute": name,
            "attributes": [first, second],
            "disparate_impact": di,
            "demographic_parity_difference": dpd,
            "violation": gap,
            "counts_toward_verdict": False,
            "groups": rates,
            "excluded_groups": excluded,
            "group_details": {
                "group_a": high, "group_a_positive_rate": high_d["positive_rate"],
                "group_a_n": high_d["n"],
                "group_b": low, "group_b_positive_rate": low_d["positive_rate"],
                "group_b_n": low_d["n"],
            },
        })
        actions.append(
            f"Intersectional '{name}' (reported only, not in the verdict): disparate impact "
            f"{di:.4f}, parity difference {dpd:.4f} - highest '{high}' "
            f"{high_d['positive_rate']:.4f} (n={high_d['n']}), lowest '{low}' "
            f"{low_d['positive_rate']:.4f} (n={low_d['n']})"
        )

    proxy_warnings = _proxy_warnings(
        proxy_findings,
        [name for name, _ in candidates],
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
        # True only when a protected attribute was measured: advisory results alone
        # support no fairness conclusion.
        "fairness_evaluated": bool(verdict_entries),
        "fairness_coverage": coverage,
        "protected_attributes_unaudited": protected_unaudited,
        "advisory_violations": advisory_violations,
        "declared_protected_attributes": declared,
        "intersectional_report": intersectional_report,
        "intersections_skipped": intersections_skipped,
        "evaluated_rows": int(len(eval_df)),
        "min_group_size": min_group_size,
        "attributes_skipped": attributes_skipped,
        "actions_taken": actions,
    }
