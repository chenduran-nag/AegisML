"""
data_agent.py
=============
Step 2 of the AI-Governed Multi-Agent Platform.

Exposes a single public function:
    run_data_agent(df, plan, target_column, task_type) -> dict

DESIGN PRINCIPLES:
  - Zero LLM calls. Fully deterministic.
  - All preprocessing logic is in pre-built functions.
  - The plan's free-text steps are parsed ONLY via keyword matching
    to log hints — never to generate or execute code.

PREPROCESSING PIPELINE (applied in order):
  1. Drop feature columns with > 50% missing
  2. Drop rows with null target (labels cannot be imputed)
  3. COMPUTE THE TRAIN/TEST SPLIT  <-- every step below fits on train rows only
  4. Impute remaining nulls:  numeric → train median, categorical → train mode
  5. Label-encode target column (classification only)
  6. Encode categoricals:
       - One-hot encode if unique values < OHE_CARDINALITY_LIMIT (10)
       - Frequency encode using TRAIN-split frequencies otherwise
  7. StandardScale all numeric, non-boolean feature columns (scaler fitted on train)

LEAKAGE BOUNDARY:
  Steps 1-2 are structural (they depend on missingness, not on learned values).
  The split is drawn immediately after them, and every parameter learned from
  data thereafter — fill values, frequency maps, scaler mean/std — is fitted on
  the train rows and merely APPLIED to the test rows. run_data_agent() returns
  train_index / test_index so the Training Agent reuses this exact split instead
  of drawing a second, inconsistent one.

QUALITY CHECK LOGIC:
  quality_check_passed = True only if ALL of:
    1. missing_pct_after_cleaning < 5%        (spec condition)
    2. No column has any unresolved nulls      (spec condition)
    3. rows_dropped_pct < 30%                 (extension: guards against
       null-label data decimation — if >30% of rows have no label,
       the remaining training set is untrustworthy)

  If quality_check_passed = False, the LangGraph wiring (Step 3) will
  route execution back to the Planner Agent for a revised plan.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from eda_insights import columns_named_in

# ---------------------------------------------------------------------------
# Module-level thresholds — easy to tune without hunting through logic
# ---------------------------------------------------------------------------

COLUMN_DROP_NULL_THRESHOLD = 0.50   # drop column if null_pct > 50%
COLUMN_HIGH_NULL_WARNING_THRESHOLD = 0.35 # unhandled if null_pct > 35% and not dropped by plan
ROW_DROP_RATIO_LIMIT = 0.30         # quality fails if > 30% of original rows dropped
OHE_CARDINALITY_LIMIT = 10          # OHE if unique < 10, else frequency encode
NULL_PCT_QUALITY_LIMIT = 5.0        # quality fails if missing_pct >= 5%

# Train/test split — computed HERE, before any statistic is fitted, and reused
# verbatim by the Training Agent so that the split boundary is identical in both
# agents. See _split_indices() for why this must happen in the Data Agent.
TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42

# Winsorization — applied only to columns a plan step names explicitly. Percentile
# bounds rather than 1.5xIQR fences: on UCI Adult, IQR fences would zero out
# capital-gain (IQR = 0, 92% zeros) and flatten hours-per-week (27.7% of values
# outside the fences around a spike at 40). Percentile bounds touch ~2% of rows by
# construction and are recomputed on the train split.
WINSOR_LOWER_QUANTILE = 0.01
WINSOR_UPPER_QUANTILE = 0.99

# Verbs that mark a plan step as a clipping instruction. Bare "cap" is excluded
# because it matches inside "capital-gain".
CLIP_KEYWORDS = ("winsor", "clip", "capping", "cap outlier", "cap extreme")
DROP_KEYWORDS = ("drop", "remove", "exclude")

# Phrases that end the part of a clause a verb applies to. What follows them names
# what to keep, or why: "Drop education-num and keep education", "Drop column
# education-num (redundant with education)". Without the cut, the column the step
# says to keep is dropped along with the one it says to drop.
SCOPE_TERMINATORS = (
    " keep ", " keeping ", " retain", " preserve", "redundant with",
    "in favour of", "in favor of", "instead of", "rather than", "(",
)

# Words that make a clause advice rather than an instruction. The Data Agent
# executes only unconditional steps: "consider winsorizing X if extreme values are
# errors" is a suggestion for a human, not an order to clip X. Conservative by
# design — a benign "if present" also blocks the action, which errs toward keeping
# data rather than deleting it.
HEDGE_MARKERS = (
    "consider", "optionally", "optional", " if ", " may ", " might ", " could ",
    " as is", " as-is", " leave ", "where appropriate",
)

# ---------------------------------------------------------------------------
# Dtype predicates
# ---------------------------------------------------------------------------


def _is_encodable_categorical(series: pd.Series) -> bool:
    """
    True for columns that need encoding before a model can consume them.

    Checking `is_object_dtype` alone is NOT sufficient. Since pandas 2.x the
    string backend may hand back a dedicated string dtype (``str`` /
    ``StringDtype``) rather than ``object``, and pandas 3 makes that the default.
    Under those versions an is_object_dtype-only test matches nothing, so every
    categorical column silently passes through unencoded and the Training Agent
    then fails on raw strings. Cover object, string and categorical explicitly.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


# ---------------------------------------------------------------------------
# Plan parsing (keyword matching only — no code generation)
# ---------------------------------------------------------------------------


def _instruction_scope(clause: str) -> str:
    """The part of a clause before its first scope terminator."""
    cut = min((i for i in (clause.find(t) for t in SCOPE_TERMINATORS) if i != -1),
              default=len(clause))
    return clause[:cut]


def _parse_plan_steps(steps: list[str], columns: list[str]) -> dict:
    """
    Extract processing hints from the plan's free-text preprocessing steps via
    keyword matching. Nothing here generates or executes code.

    Only UNCONDITIONAL instructions become actions. Each step is split into
    clauses on ";" and every clause is read on its own:
      - a hedged clause ("consider", "optionally", "if ...") is ignored;
      - a drop or clip verb counts only if it appears BEFORE any scope terminator,
        and applies only to columns named there;
      - column names are matched longest first, so "education-num" is not also
        read as "education".

    Every rule exists because the live planner wrote a step that the earlier
    substring matcher misread — dropping both columns of a redundant pair, or
    winsorizing a column the step said to keep as is. The exact phrasings are
    pinned in tests/test_eda_insights.py.
    """
    hints = {
        "smote_mentioned": False,
        "class_weight_mentioned": False,
        "frequency_encoding_mentioned": False,
        "explicit_drop_columns": [],
        "winsorize_columns": [],
    }
    for step in steps:
        sl = step.lower()
        if any(kw in sl for kw in ("smote", "oversample", "adasyn", "oversampl")):
            hints["smote_mentioned"] = True
        if any(kw in sl for kw in ("class weight", "class_weight")):
            hints["class_weight_mentioned"] = True
        if "frequency" in sl:
            hints["frequency_encoding_mentioned"] = True

        for clause in sl.split(";"):
            padded = f" {clause.strip()} "
            if any(marker in padded for marker in HEDGE_MARKERS):
                continue
            scope = _instruction_scope(padded)
            # Clip before drop: if a clause somehow names both verbs, take the
            # non-destructive reading.
            if any(kw in scope for kw in CLIP_KEYWORDS):
                hints["winsorize_columns"].extend(columns_named_in(scope, columns))
            elif any(kw in scope for kw in DROP_KEYWORDS):
                hints["explicit_drop_columns"].extend(columns_named_in(scope, columns))
    return hints


# ---------------------------------------------------------------------------
# Pre-built preprocessing functions (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _split_indices(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    actions: list[str],
    random_state: int = SPLIT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute train/test row indices BEFORE any statistic is fitted.

    WHY THIS LIVES IN THE DATA AGENT:
      Imputation fill values, frequency-encoding maps and StandardScaler
      mean/std are all parameters *learned from data*. If they are learned from
      the full dataset and the split happens later (in the Training Agent), then
      information from the held-out rows has already been baked into the
      training features — the model is evaluated on rows that influenced its own
      preprocessing, and every reported metric is optimistically biased.

      Splitting first, fitting every parameter on the train rows only, and
      applying those fitted parameters to all rows removes that leakage. The
      indices are returned so the Training Agent reuses this exact split rather
      than drawing a second, inconsistent one.

    Returns (train_index, test_index) as arrays of DataFrame index labels.
    """
    y = df[target_column]
    stratify = None

    if task_type == "classification":
        vc = y.value_counts(dropna=False)
        if len(vc) >= 2 and int(vc.min()) >= 2:
            stratify = y
        else:
            actions.append(
                "Split: stratification disabled (target has a class with fewer "
                "than 2 rows)"
            )

    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=stratify,
    )

    actions.append(
        f"Train/test split (seed {random_state}) computed before fitting: "
        f"{len(train_idx):,} train / "
        f"{len(test_idx):,} test rows. All imputation, frequency-encoding and "
        f"scaling parameters below are fitted on the train rows ONLY."
    )
    return train_idx, test_idx


def _drop_eda_structural_columns(
    df: pd.DataFrame,
    target_column: str,
    eda_findings: list[dict] | None,
    actions: list[str],
    eda_actions: list[dict],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns that exploratory analysis routed to the Data Agent.

    Only structural findings are routed here (per-row identifiers, constant
    columns). These are decided from missingness-like properties of the data, not
    from any learned parameter, so like the >50%-null rule they run before the
    train/test split. Each drop is recorded against its finding id so the
    governance gate can show exactly which EDA insight caused it.
    """
    dropped: list[str] = []
    for finding in eda_findings or []:
        if finding.get("route") != "data_agent":
            continue
        for col in finding.get("columns") or []:
            if col == target_column or col not in df.columns or col in dropped:
                continue
            action = (f"Dropped column '{col}' (EDA finding {finding['type']}: "
                      f"{finding.get('evidence', '')})")
            actions.append(action)
            eda_actions.append({"finding_id": finding["id"], "column": col,
                                "action": action})
            dropped.append(col)
    return (df.drop(columns=dropped) if dropped else df), dropped


def _winsorize_columns(
    df: pd.DataFrame,
    target_column: str,
    columns: list[str],
    actions: list[str],
    fit_index: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Clip plan-named numeric columns to percentile bounds fitted on the train rows.

    Bounds are LEARNED FROM DATA, so they come from df.loc[fit_index] and are only
    applied to the held-out rows (invariant 3). A column whose train percentiles
    coincide — e.g. one that is zero throughout the train split — is skipped,
    because clipping would collapse it to a constant.
    """
    done: list[str] = []
    for col in dict.fromkeys(columns):
        if col == target_column or col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            actions.append(f"Skipped winsorizing '{col}': not a numeric column")
            continue
        train_values = df.loc[fit_index, col].dropna()
        if train_values.empty:
            actions.append(f"Skipped winsorizing '{col}': no non-null values in the train split")
            continue
        lower, upper = (float(v) for v in train_values.quantile(
            [WINSOR_LOWER_QUANTILE, WINSOR_UPPER_QUANTILE]))
        if not lower < upper:
            actions.append(
                f"Skipped winsorizing '{col}': the train-split "
                f"{WINSOR_LOWER_QUANTILE:.0%} and {WINSOR_UPPER_QUANTILE:.0%} "
                f"percentiles coincide ({lower:.4g}), so clipping would collapse the column"
            )
            continue
        n_clipped = int(((df[col] < lower) | (df[col] > upper)).sum())
        df[col] = df[col].astype(float).clip(lower=lower, upper=upper)
        actions.append(
            f"Winsorized '{col}' to train-split percentiles [{lower:.4g}, {upper:.4g}] "
            f"(plan instruction; {n_clipped:,} value(s) clipped)"
        )
        done.append(col)
    return df, done


def _drop_high_null_columns(
    df: pd.DataFrame,
    target_column: str,
    explicit_drop_columns: list[str],
    actions: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop feature columns where null_pct > COLUMN_DROP_NULL_THRESHOLD, or columns
    explicitly instructed to be dropped in the plan's preprocessing steps.
    """
    dropped: list[str] = []
    for col in list(df.columns):
        if col == target_column:
            continue
        null_pct = df[col].isnull().sum() / len(df)
        if null_pct > COLUMN_DROP_NULL_THRESHOLD or col in explicit_drop_columns:
            reason = (
                f"plan instruction ('drop {col}')"
                if col in explicit_drop_columns
                else f"{null_pct * 100:.1f}% missing > {COLUMN_DROP_NULL_THRESHOLD * 100:.0f}% threshold"
            )
            actions.append(f"Dropped column '{col}' ({reason})")
            dropped.append(col)
    if not dropped:
        actions.append("No columns exceeded drop threshold or specified for dropping — all retained")
    return df.drop(columns=dropped), dropped


def _drop_null_target_rows(
    df: pd.DataFrame,
    target_column: str,
    actions: list[str],
) -> tuple[pd.DataFrame, int]:
    """
    Drop rows where the target is null.
    Labels cannot be imputed — missing labels are untrainable.
    """
    n_before = len(df)
    df = df.dropna(subset=[target_column])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        actions.append(
            f"Dropped {n_dropped:,} rows with null target '{target_column}' "
            f"({n_dropped / n_before * 100:.1f}% of rows)"
        )
    return df, n_dropped


def _impute_columns(
    df: pd.DataFrame,
    target_column: str,
    actions: list[str],
    fit_index: np.ndarray,
) -> pd.DataFrame:
    """
    Fill nulls in all remaining feature columns:
      - Numeric  → median of the TRAIN rows
      - Object   → mode of the TRAIN rows

    Fill values are computed from df.loc[fit_index] only, then applied to every
    row. Computing them from the full frame would leak held-out information into
    the training features.
    """
    for col in df.columns:
        if col == target_column:
            continue
        null_count = int(df[col].isnull().sum())
        if null_count == 0:
            continue

        train_values = df.loc[fit_index, col]

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = train_values.median()
            if pd.isna(fill_val):
                # Guard: every train value for this column is null.
                actions.append(
                    f"WARNING: Cannot impute '{col}' — no non-null values in the "
                    f"train split. Column will have unresolved nulls."
                )
                continue
            df[col] = df[col].fillna(fill_val)
            actions.append(
                f"Imputed '{col}' with train-split median {fill_val:.4g} "
                f"({null_count:,} nulls filled)"
            )
        else:
            mode_series = train_values.mode()
            if mode_series.empty:
                # Guard: column is entirely null below the drop threshold.
                # Shouldn't occur with 50% drop rule, but log explicitly.
                actions.append(
                    f"WARNING: Cannot impute '{col}' — no non-null values in the "
                    f"train split. Column will have unresolved nulls."
                )
                continue
            fill_val = str(mode_series.iloc[0])
            df[col] = df[col].fillna(fill_val)
            actions.append(
                f"Imputed '{col}' with train-split mode '{fill_val}' "
                f"({null_count:,} nulls filled)"
            )
    return df


def _label_encode_target(
    df: pd.DataFrame,
    target_column: str,
    actions: list[str],
) -> pd.DataFrame:
    """
    Convert a categorical target to integer codes (0, 1, …).
    Classes are sorted lexicographically for deterministic mapping.
    Numeric targets are left unchanged.
    """
    if pd.api.types.is_numeric_dtype(df[target_column]):
        return df  # already numeric (regression or previously encoded)

    classes = sorted(df[target_column].dropna().unique())
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    df[target_column] = df[target_column].map(mapping)
    mapping_str = ", ".join(f"'{k}'→{v}" for k, v in mapping.items())
    actions.append(
        f"Label-encoded target '{target_column}': {mapping_str}"
    )
    return df


# XGBoost refuses feature names containing these. One-hot encoding copies category
# VALUES into column names, and real data has values such as "<0" and "0<=X<200"
# (German Credit). Unsanitised, XGBoost failed on every fit for such a dataset and
# dropped silently out of the leaderboard. "<=" is listed before "<" so it maps first.
_FEATURE_NAME_REPLACEMENTS = {"<=": "le", "<": "lt", "[": "(", "]": ")"}
_UNSAFE_FEATURE_NAME_CHARS = "[]<"


def _sanitize_feature_names(
    df: pd.DataFrame,
    columns: list[str],
    actions: list[str],
) -> pd.DataFrame:
    """
    Rename one-hot columns whose names XGBoost cannot accept.

    Only the generated dummy columns are renamed, never an original column, so a
    sensitive attribute keeps the prefix the Fairness Agent uses to reconstruct its
    groups. Renaming stays unique: a clash gets a numeric suffix rather than
    silently merging two different categories into one column.
    """
    taken = set(df.columns)
    renames: dict[str, str] = {}
    for col in columns:
        if not any(ch in col for ch in _UNSAFE_FEATURE_NAME_CHARS):
            continue
        new = col
        for bad, good in _FEATURE_NAME_REPLACEMENTS.items():
            new = new.replace(bad, good)
        base, suffix = new, 2
        while new in taken:
            new = f"{base}_{suffix}"
            suffix += 1
        taken.discard(col)
        taken.add(new)
        renames[col] = new

    if renames:
        df = df.rename(columns=renames)
        shown = ", ".join(f"'{a}' → '{b}'" for a, b in list(renames.items())[:4])
        more = f" and {len(renames) - 4} more" if len(renames) > 4 else ""
        actions.append(
            f"Renamed {len(renames)} one-hot column(s) to remove characters XGBoost "
            f"cannot accept in feature names ('[', ']', '<'): {shown}{more}"
        )
    return df


def _encode_categoricals(
    df: pd.DataFrame,
    target_column: str,
    actions: list[str],
    fit_index: np.ndarray,
) -> pd.DataFrame:
    """
    Encode all object-dtype feature columns:
      - One-hot encoding  if nunique < OHE_CARDINALITY_LIMIT
      - Frequency encoding otherwise (value → relative frequency in [0, 1])

    One-hot encoding is a structural expansion, not a learned statistic, so the
    category set is taken from the full frame — this is deliberate, and keeps the
    train and test rows in the same column space.

    Frequency encoding IS a learned statistic, so the value→frequency map is
    built from the train rows only. Categories that appear exclusively in the
    test rows are unseen at fit time and map to 0.0.
    """
    object_cols = [
        col for col in df.columns
        if col != target_column and _is_encodable_categorical(df[col])
    ]

    ohe_cols = [c for c in object_cols if df[c].nunique() < OHE_CARDINALITY_LIMIT]
    freq_cols = [c for c in object_cols if df[c].nunique() >= OHE_CARDINALITY_LIMIT]

    # Capture nunique BEFORE get_dummies reshapes the df
    ohe_nunique = {col: df[col].nunique() for col in ohe_cols}

    if ohe_cols:
        columns_before = set(df.columns)
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=bool)
        for col in ohe_cols:
            n = ohe_nunique[col]
            actions.append(
                f"One-hot encoded '{col}' "
                f"({n} unique values → {n} new boolean columns)"
            )
        df = _sanitize_feature_names(
            df, [c for c in df.columns if c not in columns_before], actions,
        )

    for col in freq_cols:
        freq_map = df.loc[fit_index, col].value_counts(normalize=True).to_dict()
        mapped = pd.to_numeric(df[col].map(freq_map), errors="coerce")
        unseen = int(mapped.isnull().sum())
        df[col] = mapped.fillna(0.0).astype("float64")
        unseen_note = (
            f"; {unseen:,} row(s) held a category unseen in the train split → 0.0"
            if unseen else ""
        )
        actions.append(
            f"Frequency-encoded '{col}' using train-split frequencies "
            f"({len(freq_map)} unique values → relative frequency [0.0–1.0])"
            f"{unseen_note}"
        )

    return df


def _scale_numeric_features(
    df: pd.DataFrame,
    target_column: str,
    actions: list[str],
    fit_index: np.ndarray,
) -> pd.DataFrame:
    """
    Apply StandardScaler to all numeric, non-boolean feature columns.
    Skips: target column and boolean OHE columns (True/False — no scaling needed).

    The scaler is FITTED on the train rows only and then used to transform every
    row. Fitting on the full frame would put held-out means and variances into
    the training features.
    """
    numeric_feature_cols = [
        col for col in df.columns
        if col != target_column
        and pd.api.types.is_numeric_dtype(df[col])
        and not pd.api.types.is_bool_dtype(df[col])
    ]
    if not numeric_feature_cols:
        actions.append("No numeric feature columns to scale")
        return df

    scaler = StandardScaler()
    scaler.fit(df.loc[fit_index, numeric_feature_cols])
    df[numeric_feature_cols] = scaler.transform(df[numeric_feature_cols])
    actions.append(
        f"Applied StandardScaler (fitted on the train split only) to "
        f"{len(numeric_feature_cols)} numeric feature(s): {numeric_feature_cols}"
    )
    return df


def _compute_quality_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    rows_dropped: int,
    columns_dropped: list[str],
) -> tuple[dict, bool]:
    """
    Compute quality_report and quality_check_passed.

    quality_check_passed = True only when ALL conditions hold:
      1. missing_pct_after_cleaning < NULL_PCT_QUALITY_LIMIT (5%)
      2. No column in cleaned_df has any null values
      3. rows_dropped_pct < ROW_DROP_RATIO_LIMIT * 100 (30%)
      4. No feature column with > 35% missing values remains unhandled
         (if missing > 35% and not dropped by default threshold or plan, quality fails)
    """
    n_original = len(original_df)
    null_cells = int(cleaned_df.isnull().sum().sum())
    total_cells = cleaned_df.size
    missing_pct = (
        round(float(null_cells / total_cells * 100), 4) if total_cells > 0 else 0.0
    )

    unresolved_null_cols = [
        col for col in cleaned_df.columns if cleaned_df[col].isnull().any()
    ]

    # Feature columns with > 35% missing in original data that were not dropped
    unhandled_high_null = []
    for col in original_df.columns:
        if col == target_column or col in columns_dropped:
            continue
        col_null_pct = original_df[col].isnull().sum() / len(original_df)
        if col_null_pct >= COLUMN_HIGH_NULL_WARNING_THRESHOLD:
            unhandled_high_null.append(f"{col} ({col_null_pct * 100:.1f}% missing)")

    # Class balance ratio on the cleaned target (classification only)
    class_balance_ratio = None
    if task_type == "classification" and target_column in cleaned_df.columns:
        vc = cleaned_df[target_column].value_counts()
        if len(vc) >= 2:
            class_balance_ratio = round(float(vc.iloc[0]) / float(vc.iloc[-1]), 4)

    rows_dropped_pct = round(
        rows_dropped / n_original * 100, 2
    ) if n_original > 0 else 0.0

    quality_report = {
        "missing_pct_after_cleaning": missing_pct,
        "class_balance_ratio": class_balance_ratio,
        "rows_dropped": rows_dropped,
        "rows_dropped_pct": rows_dropped_pct,
        "columns_dropped": columns_dropped,
        "unresolved_null_columns": unresolved_null_cols,  # diagnostic
        "unhandled_high_null_columns": unhandled_high_null,
    }

    quality_check_passed = (
        missing_pct < NULL_PCT_QUALITY_LIMIT           # spec condition 1
        and len(unresolved_null_cols) == 0              # spec condition 2
        and rows_dropped_pct < ROW_DROP_RATIO_LIMIT * 100  # condition 3
        and len(unhandled_high_null) == 0               # condition 4
    )

    return quality_report, quality_check_passed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_data_agent(
    df: pd.DataFrame,
    plan: dict,
    target_column: str,
    task_type: str,
    eda_findings: list[dict] | None = None,
    split_seed: int | None = None,
) -> dict:
    """
    Deterministic data cleaning pipeline. Zero LLM calls.

    Returns a dict with cleaned_df, quality_check_passed, quality_report,
    actions_taken, and the train_index / test_index of the split that every
    fitted preprocessing parameter was learned from.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if task_type not in ("classification", "regression"):
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got {task_type!r}"
        )

    df = df.copy()
    original_df = df.copy()
    actions: list[str] = []
    eda_actions: list[dict] = []
    columns_dropped_total: list[str] = []

    # Step 0 — Parse plan hints (keyword matching; no code generation)
    hints = _parse_plan_steps(
        plan.get("recommended_preprocessing_steps", []),
        list(df.columns),
    )

    # Step 0b — Drop columns exploratory analysis routed to the Data Agent
    # (identifiers, constants). Structural, so it precedes the split.
    df, eda_dropped = _drop_eda_structural_columns(
        df, target_column, eda_findings, actions, eda_actions,
    )
    columns_dropped_total.extend(eda_dropped)

    # Step 1 — Drop columns > 50% missing or explicitly dropped by plan
    df, dropped = _drop_high_null_columns(
        df,
        target_column,
        hints["explicit_drop_columns"],
        actions,
    )
    columns_dropped_total.extend(dropped)

    # Step 2 — Drop rows with null target
    df, rows_dropped = _drop_null_target_rows(df, target_column, actions)

    # Step 3 — Draw the train/test split BEFORE fitting anything.
    # Everything below this line learns its parameters from train_idx only.
    # The seed is a run parameter rather than a constant so that an evaluation can
    # repeat a run over several partitions. It is recorded in the quality report,
    # so every result states which partition produced it.
    split_seed = SPLIT_RANDOM_STATE if split_seed is None else int(split_seed)
    train_idx, test_idx = _split_indices(
        df, target_column, task_type, actions, random_state=split_seed,
    )

    # Step 4 — Impute remaining nulls (fill values from the train split)
    df = _impute_columns(df, target_column, actions, fit_index=train_idx)

    # Step 4b — Winsorize columns a plan step explicitly named (train-fitted bounds)
    df, winsorized = _winsorize_columns(
        df, target_column, hints["winsorize_columns"], actions, fit_index=train_idx,
    )

    # Step 5 — Label-encode target (classification only).
    # Deterministic sorted mapping over all observed classes — not a fitted
    # statistic, and both splits must share one mapping, so it uses the full frame.
    if task_type == "classification":
        df = _label_encode_target(df, target_column, actions)

    # Step 6 — Encode categoricals (frequency maps from the train split)
    df = _encode_categoricals(df, target_column, actions, fit_index=train_idx)

    # Step 7 — Scale numeric features (scaler fitted on the train split)
    df = _scale_numeric_features(df, target_column, actions, fit_index=train_idx)

    # Step 8 — Imbalance note (flag only — SMOTE deferred to Training Agent)
    if task_type == "classification":
        vc = original_df[target_column].dropna().value_counts()
        if len(vc) >= 2:
            ratio = float(vc.iloc[0]) / float(vc.iloc[-1])
            if ratio > 1.5 or hints["smote_mentioned"] or hints["class_weight_mentioned"]:
                actions.append(
                    f"NOTE: Class imbalance — {ratio:.2f}:1 ratio "
                    f"(majority:minority). SMOTE/class_weight deferred to "
                    f"Training Agent."
                )

    # Step 9 — Quality check
    quality_report, quality_check_passed = _compute_quality_report(
        original_df=original_df,
        cleaned_df=df,
        target_column=target_column,
        task_type=task_type,
        rows_dropped=rows_dropped,
        columns_dropped=columns_dropped_total,
    )
    quality_report["train_rows"] = int(len(train_idx))
    quality_report["test_rows"] = int(len(test_idx))
    quality_report["split_seed"] = split_seed

    return {
        "cleaned_df": df,
        "quality_check_passed": quality_check_passed,
        "quality_report": quality_report,
        "actions_taken": actions,
        # Which EDA findings this agent acted on, and which columns it winsorized.
        # Read by eda_insights.build_eda_linkage() to show the reviewer how each
        # exploratory insight was used.
        "eda_actions": eda_actions,
        "winsorized_columns": winsorized,
        # Index labels of the split drawn in step 3. Consumed by
        # pipeline_graph.data_agent_node, which lifts them out of this dict and
        # into PipelineState["split_index"] so they never reach the audit log
        # (a 30k-element list would swamp every audit entry).
        "train_index": train_idx.tolist(),
        "test_index": test_idx.tolist(),
    }
