"""
training_agent.py
=================
Step 3 of the AI-Governed Multi-Agent Platform.

Exposes a single public function:
    run_training_agent(cleaned_df, target_column, task_type,
                       recommended_models) -> dict

DESIGN PRINCIPLES:
  - Zero LLM calls. Fully deterministic.
  - Models are selected from a pre-registered registry by name matching
    the Planner's output (case-insensitive, partial match tolerated).
  - Unknown model names are skipped with a logged warning — no crash.
  - Individual model training errors are caught and logged — other models
    continue training unaffected.

PIPELINE (in order):
  1. Train/test split: reuses the split drawn by the Data Agent (passed in as
     train_index / test_index) so that models are evaluated on rows that had no
     influence on any fitted preprocessing parameter. Falls back to an internal
     80/20 stratified split only when called standalone.
  2. Train each recommended model on X_train / y_train
  3. Evaluate each model on X_test / y_test
  4. Build leaderboard sorted by primary metric
  5. Select best model (top of leaderboard)
  6. Compute SHAP summary for selected model:
       TreeExplainer   -> RandomForest, XGBoost, GradientBoosting
       LinearExplainer -> LogisticRegression, Ridge, Lasso
     Sample up to SHAP_SAMPLE_SIZE rows from the test set for speed.
  7. Return results dict

RESOLVED (graph wiring, Step 4): the fitted model is returned under the private
key "_fitted_model"; pipeline_graph.training_node serialises it with
graph_state.model_to_bytes() (joblib) into PipelineState["selected_model_bytes"]
and strips the key before storing the rest of the result.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import shap

# Suppress RuntimeWarnings from sklearn's LBFGS/matmul internals (LogisticRegression
# numerical noise with frequency-encoded features). These are cosmetic — model
# convergence and metrics are unaffected. Cannot use np.errstate because that
# silences overflow signals and causes the optimizer to receive nan/inf, crashing fit.
warnings.filterwarnings("ignore", message=".*divide by zero.*", module="sklearn")
warnings.filterwarnings("ignore", message=".*overflow encountered.*", module="sklearn")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*", module="sklearn")
warnings.filterwarnings("ignore", message=".*divide by zero.*", module="numpy")
warnings.filterwarnings("ignore", message=".*overflow encountered.*", module="numpy")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*", module="numpy")
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
SHAP_SAMPLE_SIZE = 200   # max rows of test set used for SHAP (keeps runtime sane)

# Primary sort metric
PRIMARY_METRIC_CLASSIFICATION = "auc_roc"   # higher is better -> sort descending
PRIMARY_METRIC_REGRESSION = "rmse"          # lower is better  -> sort ascending

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Maps lower-case keyword to (ClassifierTemplate, RegressorTemplate).
# Lookup: find first key that is a substring of the normalised model name.
# Each entry is a TEMPLATE; _resolve_model() clones it so runs are isolated.

_REGISTRY: dict[str, tuple[Any, Any]] = {
    "logisticregression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),  # n_jobs has no effect here; sklearn 1.8+ warns
        None,
    ),
    "logistic": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),  # n_jobs has no effect here; sklearn 1.8+ warns
        None,
    ),
    "randomforest": (
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    ),
    "xgboost": (
        XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        ),
        XGBRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            eval_metric="rmse",
            verbosity=0,
        ),
    ),
    "gradientboosting": (
        GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    ),
    "ridge": (
        None,
        Ridge(alpha=1.0),
    ),
    "lasso": (
        None,
        Lasso(alpha=0.1),
    ),
    # SVM excluded: no TreeExplainer/LinearExplainer support; KernelExplainer
    # is too slow (10-100x) for governance use on this data size.
    "svm": (None, None),
}


def _resolve_model(name: str, task_type: str) -> tuple[Any | None, str]:
    """
    Return (cloned_model_instance, matched_key) or (None, reason_string).
    """
    import sklearn.base

    key_norm = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    matched_key: str | None = None
    for registry_key in _REGISTRY:
        if registry_key in key_norm:
            matched_key = registry_key
            break

    if matched_key is None:
        return None, f"No registry entry matching '{name}'"

    clf_template, reg_template = _REGISTRY[matched_key]

    if task_type == "classification":
        if clf_template is None:
            return None, f"'{name}' has no classification implementation"
        return sklearn.base.clone(clf_template), matched_key
    else:
        if reg_template is None:
            return None, f"'{name}' has no regression implementation"
        return sklearn.base.clone(reg_template), matched_key


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _classification_metrics(model: Any, X_test: np.ndarray,
                             y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)

    n_classes = len(np.unique(y_test))
    avg = "binary" if n_classes == 2 else "weighted"

    auc = None
    if y_prob is not None:
        try:
            auc = round(float(roc_auc_score(y_test, y_prob)), 4)
        except Exception:
            auc = None

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred, average=avg)), 4),
        "auc_roc": auc,
    }


def _regression_metrics(model: Any, X_test: np.ndarray,
                        y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }


def evaluate_model(model: Any, cleaned_df: pd.DataFrame, target_column: str,
                   task_type: str, index: list) -> dict:
    """
    Score an already-fitted model on the given rows exactly as run_training_agent
    scores its evaluation rows: the same target encoding and bool-to-int conversion.

    Used once, after approval, on the untouched test rows (pipeline_graph.audit_log_node).
    """
    X = cleaned_df.drop(columns=[target_column])
    y = cleaned_df[target_column]
    if task_type == "classification":
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
    bool_cols = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]
    if bool_cols:
        X = X.copy()
        X[bool_cols] = X[bool_cols].astype(int)
    rows = pd.Index(index)
    if task_type == "classification":
        return _classification_metrics(model, X.loc[rows], y.loc[rows])
    return _regression_metrics(model, X.loc[rows], y.loc[rows])


# ---------------------------------------------------------------------------
# SHAP explainability summary
# ---------------------------------------------------------------------------

_TREE_TYPES = (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    XGBClassifier, XGBRegressor,
)
_LINEAR_TYPES = (LogisticRegression, Ridge, Lasso)


def _shap_summary(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    actions: list[str],
) -> list[dict]:
    """
    Compute top-5 features by mean absolute SHAP value.

    - TreeExplainer  for tree-based models (fast, exact)
    - LinearExplainer for linear models (fast; values in log-odds for classifiers --
      acceptable for a governance summary, not a user-facing report)
    """
    sample = X_test.sample(
        n=min(SHAP_SAMPLE_SIZE, len(X_test)),
        random_state=RANDOM_STATE,
    )
    feature_names = list(X_test.columns)

    try:
        if isinstance(model, _TREE_TYPES):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]   # binary: use positive-class SHAP
            actions.append(
                f"SHAP: TreeExplainer on {len(sample)} test samples"
            )

        elif isinstance(model, _LINEAR_TYPES):
            background = X_train.mean(axis=0).values.reshape(1, -1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = shap.LinearExplainer(model, background)
            shap_vals = explainer.shap_values(sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            actions.append(
                f"SHAP: LinearExplainer (log-odds space) on {len(sample)} test samples"
            )

        else:
            actions.append(
                f"SHAP: {type(model).__name__} not supported by "
                "TreeExplainer/LinearExplainer — skipping"
            )
            return []

        if isinstance(shap_vals, list):
            mean_abs = np.mean([np.abs(v).mean(axis=0) for v in shap_vals], axis=0)
        else:
            mean_abs = np.abs(shap_vals)
            if mean_abs.ndim == 3:
                mean_abs = mean_abs.mean(axis=-1)
            mean_abs = mean_abs.mean(axis=0)

        indexed = sorted(
            zip(feature_names, [float(x) for x in mean_abs]),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {"feature": feat, "importance": round(imp, 6)}
            for feat, imp in indexed[:5]
        ]

    except Exception as exc:
        actions.append(f"SHAP computation failed ({type(exc).__name__}: {exc}) — skipping")
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_training_agent(
    cleaned_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    recommended_models: list[str],
    train_index: list | None = None,
    test_index: list | None = None,
    sample_weight: pd.Series | None = None,
    eval_label: str = "test",
) -> dict:
    """
    Train, evaluate, and rank models recommended by the Planner Agent.

    sample_weight, when given, is a Series of per-row training weights indexed by
    train-row labels (reweighing mitigation, see mitigation.py). Every registry model
    accepts it in fit(). Evaluation metrics stay unweighted.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Fully cleaned/encoded dataset from run_data_agent().
        Target must be numeric (label-encoded for classification).
    target_column : str
        Name of the target column in cleaned_df.
    task_type : {"classification", "regression"}
    recommended_models : list[str]
        Names from plan["recommended_models"].
    train_index, test_index : list, optional
        Index labels of the split drawn by run_data_agent(). When supplied, this
        exact split is reused, guaranteeing that the rows used to fit imputation
        / encoding / scaling parameters are precisely the rows the model trains
        on. When omitted, an internal split is drawn and a warning is recorded in
        actions_taken, because preprocessing statistics may then have been fitted
        over rows that end up in the test set.

    Returns
    -------
    dict:
        leaderboard           : list[dict]
        selected_model_name   : str
        selected_model_metrics: dict
        shap_summary          : list[dict] (top-5 features)
        actions_taken         : list[str]
    """
    if task_type not in ("classification", "regression"):
        raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type!r}")
    if target_column not in cleaned_df.columns:
        raise ValueError(
            f"target_column '{target_column}' not in cleaned_df. "
            f"Available: {list(cleaned_df.columns)}"
        )

    actions: list[str] = []

    # ------------------------------------------------------------------
    # Step 1 — Train/test split
    # ------------------------------------------------------------------
    X = cleaned_df.drop(columns=[target_column])
    y = cleaned_df[target_column]

    # Convert classification target to 0..N-1 contiguous integers for XGBoost compatibility
    if task_type == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    # Convert bool columns to int for XGBoost compatibility
    bool_cols = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]
    if bool_cols:
        X = X.copy()
        X[bool_cols] = X[bool_cols].astype(int)

    if train_index is not None and test_index is not None:
        train_idx = pd.Index(train_index)
        test_idx = pd.Index(test_index)

        unknown = len(train_idx.difference(X.index)) + len(test_idx.difference(X.index))
        if unknown:
            raise ValueError(
                f"Training Agent: {unknown} index label(s) supplied in "
                f"train_index/test_index are absent from cleaned_df. The Data "
                f"Agent's split and the cleaned frame are out of sync."
            )

        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        actions.append(
            f"Train/{eval_label} split: reused the Data Agent's split "
            f"({len(X_train):,} train / {len(X_test):,} {eval_label} rows; the leaderboard "
            f"ranks models on the {eval_label} rows). Imputation, "
            f"encoding and scaling parameters were fitted on these train rows only."
        )
    else:
        stratify_param = y if task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify_param,
        )
        actions.append(
            f"WARNING: no split supplied by the Data Agent — drew an internal "
            f"{len(X_train):,} train / {len(X_test):,} test split "
            f"(stratified={task_type == 'classification'}). Preprocessing "
            f"statistics may have been fitted over held-out rows; treat these "
            f"metrics as optimistic."
        )

    # ------------------------------------------------------------------
    # Steps 2–3 — Train and evaluate each recommended model
    # ------------------------------------------------------------------
    leaderboard: list[dict] = []
    trained_models: dict[str, Any] = {}

    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        missing = len(X_train.index.difference(sample_weight.index))
        if missing:
            raise ValueError(
                f"Training Agent: sample_weight has no weight for {missing} train row(s)."
            )
        weights = sample_weight.loc[X_train.index]
        fit_kwargs["sample_weight"] = weights.to_numpy()
        actions.append(
            f"Training with bias-mitigation sample weights on {len(weights):,} train rows "
            f"(min {weights.min():.4f}, max {weights.max():.4f}); evaluation is unweighted"
        )

    for model_name in recommended_models:
        model, resolved = _resolve_model(model_name, task_type)

        if model is None:
            actions.append(f"WARN: Skipped '{model_name}': {resolved}")
            leaderboard.append({
                "model_name": model_name,
                "metrics": {},
                "trained_successfully": False,
                "skip_reason": resolved,
            })
            continue

        try:
            model.fit(X_train, y_train, **fit_kwargs)
            metrics = (
                _classification_metrics(model, X_test, y_test)
                if task_type == "classification"
                else _regression_metrics(model, X_test, y_test)
            )
            actions.append(
                f"Trained '{model_name}': "
                + ", ".join(f"{k}={v}" for k, v in metrics.items())
            )
            leaderboard.append({
                "model_name": model_name,
                "metrics": metrics,
                "trained_successfully": True,
            })
            trained_models[model_name] = model

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            actions.append(f"ERROR: Training '{model_name}' failed: {reason}")
            leaderboard.append({
                "model_name": model_name,
                "metrics": {},
                "trained_successfully": False,
                "skip_reason": reason,
            })

    # ------------------------------------------------------------------
    # Step 4 — Sort leaderboard by primary metric
    # ------------------------------------------------------------------
    successful = [e for e in leaderboard if e["trained_successfully"]]

    if not successful:
        raise RuntimeError(
            f"Training Agent: no model trained successfully. "
            f"Attempted: {recommended_models}. See actions_taken."
        )

    if task_type == "classification":
        def _sort_key(e: dict) -> float:
            auc = e["metrics"].get("auc_roc")
            return -(auc if auc is not None else e["metrics"].get("f1", 0.0))
        successful.sort(key=_sort_key)
    else:
        successful.sort(key=lambda e: e["metrics"].get("rmse", float("inf")))

    failed = [e for e in leaderboard if not e["trained_successfully"]]
    leaderboard = successful + failed

    # ------------------------------------------------------------------
    # Step 5 — Select best model
    # ------------------------------------------------------------------
    selected_entry = leaderboard[0]
    selected_name = selected_entry["model_name"]
    selected_model = trained_models[selected_name]
    actions.append(f"Selected model: '{selected_name}' (top of leaderboard by primary metric)")

    # ------------------------------------------------------------------
    # Step 6 — SHAP summary for selected model
    # ------------------------------------------------------------------
    shap_summary = _shap_summary(selected_model, X_train, X_test, actions)
    if shap_summary:
        actions.append(
            "Top SHAP features: "
            + ", ".join(f"{e['feature']} ({e['importance']:.4f})" for e in shap_summary[:3])
        )

    return {
        "leaderboard": leaderboard,
        "selected_model_name": selected_name,
        "selected_model_metrics": selected_entry["metrics"],
        "shap_summary": shap_summary,
        "actions_taken": actions,
        # NOTE: _fitted_model is a private key consumed by pipeline_graph's
        # training_node, which extracts it, serialises it via model_to_bytes(),
        # and strips it before storing the rest in PipelineState["training_result"].
        # This key is intentionally not part of the public API contract.
        "_fitted_model": selected_model,
    }
