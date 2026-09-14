"""
eda_insights.py
===============
Turns exploratory data analysis from a passive dashboard into structured findings
that the rest of the pipeline consumes.

WHY THIS EXISTS:
  analyze_raw_dataset() computes rich profiling — cardinality, outliers,
  correlations — but until this module nothing downstream read it. The planner
  recomputed its own thinner summary, and insights that matter for governance
  (a column that silently encodes a protected attribute, a column that encodes the
  target) were visible on a chart and nowhere else.

FINDING ROUTES — who is allowed to act on what:
  Every finding carries a `route`. It is the same trust model the pipeline already
  applies to data versus the LLM, extended to exploratory analysis.

  data_agent  Structural and unambiguous: per-row identifiers or free text, and
              constant columns. Dropped deterministically before the train/test
              split, in the same category as the existing >50%-null column drop.

  planner     Judgement-light: zero-inflated or outlier-heavy numeric features,
              redundant feature pairs, a heavily skewed regression target. Passed
              to the planner as concerns it must address by name. The Data Agent
              then executes only concrete, safe instructions — dropping a named
              column, or winsorizing a named column at train-fitted percentiles.

  reviewer    Judgement-heavy: suspected target leakage and proxy variables for
              protected attributes. Shown at the governance gate and deliberately
              NEVER sent to the planner. If the LLM saw "relationship is a proxy for
              sex" it might recommend dropping `relationship`, and the Data Agent's
              keyword matcher would execute that — an automated fairness decision
              nobody approved. Removing a proxy costs accuracy and frequently fails
              to remove the disparity anyway ("fairness through unawareness"). If
              the reviewer decides to act, Loop 2 (reject on data quality with a
              written directive) carries that human decision back through the planner.

WHAT THE STATISTICS ARE COMPUTED ON:
  The raw uploaded frame, before the split. That is deliberate and safe here: a
  finding is a descriptive flag, not a fitted parameter. The one transformation a
  finding can lead to that learns from data — winsorization — recomputes its bounds
  on the train rows inside the Data Agent (invariant 3: split before fit).

THRESHOLDS:
  Chosen against the UCI Adult Income data and documented next to each constant.
  They are heuristics, not statistical tests.

HONEST LIMITS:
  - Association is not use. A Cramér's V of 0.65 between `relationship` and `sex`
    shows a model COULD recover sex from relationship, not that this model does.
  - Proxy detection covers only columns whose names match protected-attribute
    keywords. A protected attribute stored under an opaque name is missed.
  - Pairs of protected attributes (e.g. race and native-country) are not reported
    as proxies of each other.
  - Integer identifiers are caught only when named like an id or stored as a
    monotonic row number; a shuffled, unnamed integer key is missed.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Structural (route: data_agent)
IDENTIFIER_UNIQUE_RATIO = 0.95   # text column with >=95% distinct non-null values
MIN_ROWS_FOR_IDENTIFIER = 20     # below this, all-unique values happen by chance

# Distribution (route: planner)
ZERO_INFLATION_FRACTION = 0.50   # Adult: capital-gain 91.7% zeros, capital-loss 95.3%
HEAVY_OUTLIER_FRACTION = 0.05    # share of values outside the 1.5xIQR fences
PEAKED_DISTRIBUTION_FRACTION = 0.20  # Adult: hours-per-week 27.7% "outliers" around a
                                     # spike at 40 — a shape, not a data error
SKEWED_TARGET_ABS_SKEW = 2.0     # regression target

# Association (routes: planner / reviewer)
REDUNDANCY_THRESHOLD = 0.90      # feature-feature; Adult: education vs education-num = 1.0
LEAKAGE_ASSOCIATION_THRESHOLD = 0.95
PROXY_MEDIUM_THRESHOLD = 0.40    # Adult: occupation->sex 0.43, marital-status->sex 0.46
PROXY_HIGH_THRESHOLD = 0.60      # Adult: relationship->sex 0.65

MAX_CATEGORIES_FOR_ASSOCIATION = 100   # skip contingency tables beyond this
MAX_FEATURES_FOR_PAIRWISE = 60         # ~1,800 pairs; beyond that, redundancy is skipped
ASSOCIATION_SAMPLE_ROWS = 20_000       # deterministic subsample for association stats
MIN_ROWS_FOR_ASSOCIATION = 10

MAX_PLANNER_FINDINGS = 12        # keeps the planner prompt compact

# Protected attributes, matched on name tokens rather than substrings: a substring
# test for "age" matches "wage", "percentage" and "language".
PROTECTED_EXACT_TOKENS = {
    "age", "sex", "gender", "race", "religion", "nationality", "native", "country",
}
PROTECTED_TOKEN_STEMS = ("ethnic", "disab", "relig", "national", "sexual")

_IDENTIFIER_NAME_TOKENS = {"id", "uuid", "guid", "key", "index", "idx"}

ROUTES = ("data_agent", "planner", "reviewer")
_SEVERITY_ORDER = {"high": 0, "medium": 1, "info": 2}

_METHOD_LABELS = {
    "cramers_v": "Cramér's V",
    "correlation_ratio": "correlation ratio",
    "pearson_abs": "|Pearson r|",
}


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t]


def is_protected_attribute(column: str) -> bool:
    """True if a column name looks like a protected attribute."""
    for token in _name_tokens(column):
        if token in PROTECTED_EXACT_TOKENS:
            return True
        if any(token.startswith(stem) for stem in PROTECTED_TOKEN_STEMS):
            return True
    return False


def mentions_column(text: str, column: str) -> bool:
    """
    True if `text` names `column` as a whole word.

    A plain substring test is not good enough: "age" occurs inside "percentage"
    and "capital-gain" occurs inside nothing, but "cap" occurs inside
    "capital-gain". Column names may contain hyphens, so only alphanumerics count
    as a word boundary.
    """
    if not text or not column:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(str(column).lower()) + r"(?![a-z0-9])"
    return re.search(pattern, str(text).lower()) is not None


def columns_named_in(text: str, columns: list[str]) -> list[str]:
    """
    Columns named in `text` as whole words, returned in `columns` order.

    Names are tried longest first, and each match is blanked out of the working
    text before shorter names are tried. Because a hyphen counts as a word
    boundary, mentions_column() alone reads "education-num" as also naming
    "education"; masking the longer match first prevents that.
    """
    remaining = str(text or "").lower()
    found: set[str] = set()
    for col in sorted(columns, key=lambda c: len(str(c)), reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(str(col).lower()) + r"(?![a-z0-9])"
        if re.search(pattern, remaining):
            found.add(col)
            remaining = re.sub(pattern, " ", remaining)
    return [c for c in columns if c in found]


# ---------------------------------------------------------------------------
# Association measures
# ---------------------------------------------------------------------------


def _is_categorical(series: pd.Series) -> bool:
    return (not pd.api.types.is_numeric_dtype(series)) or pd.api.types.is_bool_dtype(series)


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    k = min(observed.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else 0.0


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    values = values.astype(float)
    grand_mean = values.mean()
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total == 0:
        return 0.0
    grouped = values.groupby(categories, observed=True)
    ss_between = float((grouped.count() * (grouped.mean() - grand_mean) ** 2).sum())
    return float(np.sqrt(min(ss_between / ss_total, 1.0)))


def association(a: pd.Series, b: pd.Series) -> Optional[tuple[str, float]]:
    """
    Strength of association between two columns, on a 0..1 scale.

    Picks the measure by type: Cramér's V for two categoricals, the correlation
    ratio (eta) for categorical vs numeric, |Pearson r| for two numerics. Returns
    None when the pair is too small, constant, or too high-cardinality to measure.
    """
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    if len(a) < MIN_ROWS_FOR_ASSOCIATION:
        return None

    a_cat, b_cat = _is_categorical(a), _is_categorical(b)
    if a_cat and a.nunique() > MAX_CATEGORIES_FOR_ASSOCIATION:
        return None
    if b_cat and b.nunique() > MAX_CATEGORIES_FOR_ASSOCIATION:
        return None

    if a_cat and b_cat:
        return "cramers_v", _cramers_v(a, b)
    if a_cat:
        return "correlation_ratio", _correlation_ratio(a, b)
    if b_cat:
        return "correlation_ratio", _correlation_ratio(b, a)

    if a.nunique() < 2 or b.nunique() < 2:
        return None
    r = a.astype(float).corr(b.astype(float))
    return ("pearson_abs", abs(float(r))) if pd.notna(r) else None


# ---------------------------------------------------------------------------
# Finding derivation
# ---------------------------------------------------------------------------


def _finding(
    ftype: str,
    severity: str,
    columns: list[str],
    evidence: str,
    recommendation: str,
    route: str,
    metric: Optional[tuple[str, float]] = None,
    key: Optional[str] = None,
) -> dict:
    return {
        "id": f"{ftype}:{key or '|'.join(columns)}",
        "type": ftype,
        "severity": severity,
        "columns": list(columns),
        "metric": ({"name": metric[0], "value": round(float(metric[1]), 4)}
                   if metric else None),
        "evidence": evidence,
        "recommendation": recommendation,
        "route": route,
    }


def _metric_text(metric: tuple[str, float]) -> str:
    return f"{_METHOD_LABELS.get(metric[0], metric[0])} {metric[1]:.3f}"


def derive_eda_findings(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> list[dict]:
    """
    Derive structured, routed findings from the raw dataset.

    Returns a list of finding dicts, most severe first. Every value is plain
    Python and aggregate-only, so findings are safe to checkpoint, audit-log and
    — for the planner route — send to the LLM.
    """
    findings: list[dict] = []
    n_rows = len(df)
    if n_rows == 0:
        return findings

    features = [c for c in df.columns if c != target_column]
    structural: set[str] = set()

    # --- 1. Structural: identifiers and constants (route: data_agent) --------
    for col in features:
        series = df[col]
        non_null = series.dropna()
        nunique = int(non_null.nunique())

        if nunique <= 1:
            structural.add(col)
            findings.append(_finding(
                "constant_column", "medium", [col],
                evidence=f"'{col}' has {nunique} distinct non-null value(s) across {n_rows:,} rows",
                recommendation="Drop: a column with no variation carries no signal.",
                route="data_agent",
            ))
            continue

        if len(non_null) < MIN_ROWS_FOR_IDENTIFIER:
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        ratio = nunique / len(non_null)
        named_like_id = bool(set(_name_tokens(col)) & _IDENTIFIER_NAME_TOKENS)
        is_identifier = (
            (_is_categorical(series) and ratio >= IDENTIFIER_UNIQUE_RATIO)
            or (pd.api.types.is_integer_dtype(series) and ratio == 1.0
                and (named_like_id or non_null.is_monotonic_increasing))
        )
        if is_identifier:
            structural.add(col)
            findings.append(_finding(
                "identifier_column", "high", [col],
                evidence=(f"'{col}' has {ratio:.1%} distinct values "
                          f"({nunique:,} / {len(non_null):,} non-null rows)"),
                recommendation=(
                    "Drop: a per-row identifier or free-text field carries no "
                    "generalisable signal and lets a model memorise individual rows."
                ),
                route="data_agent",
            ))

    # --- 2. Distribution of numeric features (route: planner) ---------------
    for col in features:
        if col in structural:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        values = series.dropna().astype(float)
        if len(values) < MIN_ROWS_FOR_ASSOCIATION:
            continue

        zero_fraction = float((values == 0).mean())
        if zero_fraction >= ZERO_INFLATION_FRACTION:
            findings.append(_finding(
                "zero_inflated", "info", [col],
                evidence=f"{zero_fraction:.1%} of non-null values in '{col}' are exactly 0",
                recommendation=(
                    "Treat as zero-inflated. IQR fences are meaningless here (the "
                    "interquartile range collapses to zero, so every non-zero value "
                    "looks like an outlier) — do not clip to them. Consider a "
                    "binary 'is non-zero' indicator or a log1p transform."
                ),
                route="planner",
            ))
            continue

        q1, q3 = values.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        if iqr <= 0:
            continue
        outlier_fraction = float(
            ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean()
        )
        if outlier_fraction >= HEAVY_OUTLIER_FRACTION:
            if outlier_fraction >= PEAKED_DISTRIBUTION_FRACTION:
                recommendation = (
                    "So many values fall outside the fences that the distribution is "
                    "sharply peaked rather than outlier-laden; clipping to the fences "
                    "would distort genuine signal. Leave as is, or winsorize at the "
                    "1st/99th percentiles only if extreme values are known to be errors."
                )
            else:
                recommendation = (
                    "Consider winsorizing at the train-split 1st/99th percentiles to "
                    "limit the influence of extreme values."
                )
            findings.append(_finding(
                "heavy_outliers", "medium", [col],
                evidence=(f"{outlier_fraction:.1%} of values in '{col}' fall outside "
                          f"the 1.5xIQR fences (IQR={iqr:.4g})"),
                recommendation=recommendation,
                route="planner",
            ))

    # --- 3. Skewed regression target (route: planner) -----------------------
    if (task_type == "regression" and target_column in df.columns
            and pd.api.types.is_numeric_dtype(df[target_column])):
        target_values = df[target_column].dropna().astype(float)
        if len(target_values) >= MIN_ROWS_FOR_ASSOCIATION:
            skew = float(target_values.skew())
            if abs(skew) >= SKEWED_TARGET_ABS_SKEW:
                findings.append(_finding(
                    "skewed_target", "medium", [target_column],
                    evidence=f"target '{target_column}' has skewness {skew:.2f}",
                    recommendation=(
                        "Consider modelling a log1p-transformed target, or report a "
                        "metric robust to a long tail (e.g. MAE) alongside RMSE."
                    ),
                    route="planner",
                ))

    # --- 4. Associations -----------------------------------------------------
    work = (df.sample(n=ASSOCIATION_SAMPLE_ROWS, random_state=0)
            if n_rows > ASSOCIATION_SAMPLE_ROWS else df)
    candidates = [c for c in features if c not in structural]

    # 4a. Target leakage (route: reviewer)
    if target_column in work.columns:
        for col in candidates:
            result = association(work[col], work[target_column])
            if result and result[1] >= LEAKAGE_ASSOCIATION_THRESHOLD:
                findings.append(_finding(
                    "target_leakage_suspect", "high", [col],
                    metric=result,
                    evidence=(f"{_metric_text(result)} between '{col}' and the "
                              f"target '{target_column}'"),
                    recommendation=(
                        "An association this strong usually means the column encodes "
                        "the outcome, for example because it is recorded after the "
                        "fact. Confirm it is available at prediction time before "
                        "approving; if it is not, reject on data quality and direct "
                        "the planner to drop it."
                    ),
                    route="reviewer",
                ))

    # 4b. Proxy variables for protected attributes (route: reviewer)
    protected = [c for c in candidates if is_protected_attribute(c)]
    for attribute in protected:
        for col in candidates:
            if col == attribute or is_protected_attribute(col):
                continue
            result = association(work[col], work[attribute])
            if not result or result[1] < PROXY_MEDIUM_THRESHOLD:
                continue
            severity = "high" if result[1] >= PROXY_HIGH_THRESHOLD else "medium"
            findings.append(_finding(
                "proxy_variable", severity, [col, attribute],
                key=f"{col}->{attribute}",
                metric=result,
                evidence=(f"{_metric_text(result)} between '{col}' and protected "
                          f"attribute '{attribute}'"),
                recommendation=(
                    f"The model can partly reconstruct '{attribute}' from '{col}', so "
                    f"disparities by '{attribute}' can persist even if '{attribute}' "
                    f"itself were removed. Dropping '{col}' is a policy decision with "
                    f"an accuracy cost that often fails to remove the disparity; read "
                    f"the fairness results with this association in mind."
                ),
                route="reviewer",
            ))

    # 4c. Redundant feature pairs (route: planner)
    non_protected = [c for c in candidates if not is_protected_attribute(c)]
    if len(non_protected) <= MAX_FEATURES_FOR_PAIRWISE:
        for i, left in enumerate(non_protected):
            for right in non_protected[i + 1:]:
                result = association(work[left], work[right])
                if result and result[1] >= REDUNDANCY_THRESHOLD:
                    findings.append(_finding(
                        "redundant_features", "info", [left, right],
                        metric=result,
                        evidence=f"{_metric_text(result)} between '{left}' and '{right}'",
                        recommendation=(
                            "The two columns carry largely the same information. "
                            "Consider dropping one, keeping whichever is available and "
                            "interpretable at prediction time."
                        ),
                        route="planner",
                    ))
    else:
        findings.append(_finding(
            "analysis_limited", "info", [],
            key="redundancy",
            evidence=(f"Redundancy check skipped: {len(non_protected)} candidate "
                      f"features exceeds the pairwise limit of {MAX_FEATURES_FOR_PAIRWISE}"),
            recommendation="Redundant feature pairs were not assessed for this dataset.",
            route="reviewer",
        ))

    findings.sort(key=lambda f: _SEVERITY_ORDER.get(f["severity"], 9))
    return findings


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def findings_for_route(findings: list[dict] | None, route: str) -> list[dict]:
    return [f for f in (findings or []) if f.get("route") == route]


def compact_findings_for_planner(findings: list[dict] | None) -> list[dict]:
    """
    The subset of findings the LLM is allowed to see, trimmed for the prompt.

    Only route == "planner". Reviewer-routed findings are excluded on purpose —
    see the module docstring for why a proxy or leakage flag must not reach the
    planner. Data-Agent-routed findings are excluded because they are already
    handled deterministically and need no judgement.
    """
    return [
        {
            "type": f["type"],
            "severity": f["severity"],
            "columns": f["columns"],
            "evidence": f["evidence"],
            "recommendation": f["recommendation"],
        }
        for f in findings_for_route(findings, "planner")
    ][:MAX_PLANNER_FINDINGS]


# ---------------------------------------------------------------------------
# Linkage: how each finding was actually used
# ---------------------------------------------------------------------------


def build_eda_linkage(
    findings: list[dict] | None,
    plan: dict | None = None,
    data_agent_result: dict | None = None,
    fairness_result: dict | None = None,
) -> list[dict]:
    """
    Annotate every finding with what each downstream stage did with it.

    Pure function over recorded state, so it is safe to call inside
    human_approval_node before interrupt() (invariant 6), and produces the same
    answer when the model card is generated later.

    Outcome statuses:
      applied        a deterministic stage acted on it (drop, winsorize)
      not_applied    routed to the Data Agent, but no action was recorded
      sent           passed to the planner
      mentioned      the plan names the finding's column(s) — a textual check,
                     not proof the concern was resolved
      not_reflected  passed to the planner, but the plan never names the column(s)
      flagged        held for the human reviewer
      attached       also surfaced inside the fairness audit
      pending        the consuming stage has not run yet
    """
    plan = plan or {}
    plan_text = " ".join(
        list(plan.get("data_quality_concerns", []) or [])
        + list(plan.get("recommended_preprocessing_steps", []) or [])
    )
    data_res = data_agent_result or {}
    eda_actions = {a.get("finding_id"): a for a in (data_res.get("eda_actions") or [])}
    winsorized = set(data_res.get("winsorized_columns") or [])
    dropped = set((data_res.get("quality_report") or {}).get("columns_dropped") or [])
    attached = {w.get("id") for w in ((fairness_result or {}).get("proxy_warnings") or [])}

    linked = []
    for finding in findings or []:
        outcomes: list[dict] = []
        route = finding.get("route")
        columns = finding.get("columns") or []

        if route == "data_agent":
            action = eda_actions.get(finding["id"])
            if action:
                outcomes.append({"stage": "data_agent", "status": "applied",
                                 "detail": action.get("action", "")})
            elif data_agent_result is None:
                outcomes.append({"stage": "data_agent", "status": "pending",
                                 "detail": "The Data Agent has not run yet."})
            else:
                outcomes.append({"stage": "data_agent", "status": "not_applied",
                                 "detail": "No action recorded; the column may already "
                                           "have been removed by another rule."})

        elif route == "planner":
            outcomes.append({"stage": "planner", "status": "sent",
                             "detail": "Passed to the planner as a concern to address."})
            if plan:
                named = [c for c in columns if mentions_column(plan_text, c)]
                if named:
                    outcomes.append({"stage": "planner", "status": "mentioned",
                                     "detail": f"The plan names {', '.join(named)}."})
                else:
                    outcomes.append({"stage": "planner", "status": "not_reflected",
                                     "detail": "The plan does not name the column(s) "
                                               "this finding concerns."})
            acted_winsor = [c for c in columns if c in winsorized]
            if acted_winsor:
                outcomes.append({"stage": "data_agent", "status": "applied",
                                 "detail": f"Winsorized {', '.join(acted_winsor)} at "
                                           f"train-split percentiles."})
            acted_drop = [c for c in columns if c in dropped]
            if acted_drop:
                outcomes.append({"stage": "data_agent", "status": "applied",
                                 "detail": f"Dropped {', '.join(acted_drop)}."})

        else:
            outcomes.append({"stage": "reviewer", "status": "flagged",
                             "detail": "Held for the human reviewer: not sent to the "
                                       "planner and not acted on automatically."})
            if finding.get("id") in attached:
                outcomes.append({"stage": "fairness", "status": "attached",
                                 "detail": "Attached to the fairness audit as a proxy warning."})

        linked.append({**finding, "outcomes": outcomes})
    return linked
