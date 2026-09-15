"""
mitigation.py
=============
Bias mitigation by reweighing (Kamiran & Calders, 2012). It runs only when a human
reviewer chooses `reject_and_mitigate` at the governance gate.

DESIGN / WHY
  - Reweighing, not a constrained learner or per-group decision thresholds. It
    changes only the weights of training rows, so the approved model is an ordinary
    estimator: SHAP, the joblib artifact and the model card work unchanged, and the
    deployed model needs no protected attribute at prediction time. Its effect can
    be modest, which is why the gate shows the reviewer before and after.
  - The weight of a (group, label) cell is P(group) * P(label) / P(group, label),
    computed on TRAIN rows only (invariant 3). In the weighted training data the
    label is independent of group membership.
  - Groups are formed with the Fairness Agent's own rules (raw values, age bands, a
    missing-value group), so the reweighted attribute is the audited one.
  - The attribute is chosen deterministically, not by the LLM: the protected
    attribute with the lowest disparate impact among the violations. Advisory
    (unprotected) attributes are never reweighted. A later mitigation adds its
    attribute, and weights are computed on the intersection of all of them.
  - State and the audit log hold per-cell weights, which are aggregates, never
    per-row weights (invariant 8). Row weights are rebuilt from the raw frame at
    training time from those same cells, so what is logged is what was trained on.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fairness_agent import resolve_groups_from_raw

METHOD = "reweighing"
INTERSECTION_SEPARATOR = " x "


def choose_attribute(fairness_result: dict | None, already_mitigated: list[str]) -> str | None:
    """
    The protected attribute to mitigate next, or None if none is left.

    Only attributes that count toward the verdict are candidates. Reweighing an
    advisory attribute cannot change the verdict, and in the evaluation it did not
    help (German Credit `job`).
    """
    report = (fairness_result or {}).get("fairness_report") or []
    pool = [r for r in report
            if r.get("violation") and r.get("counts_toward_verdict", r.get("protected"))
            and r.get("attribute") not in already_mitigated]
    if not pool:
        return None
    return min(pool, key=lambda r: r["disparate_impact"])["attribute"]


def group_labels(raw_rows: pd.DataFrame, attributes: list[str]) -> pd.Series:
    """Group label per row; the intersection when several attributes are given."""
    labels: pd.Series | None = None
    for attribute in attributes:
        if attribute not in raw_rows.columns:
            raise ValueError(
                f"cannot mitigate '{attribute}': it is not a column of the uploaded data")
        groups, _, reason = resolve_groups_from_raw(attribute, raw_rows)
        if groups is None:
            raise ValueError(f"cannot mitigate '{attribute}': {reason}")
        groups = groups.astype(str)
        labels = groups if labels is None else labels + INTERSECTION_SEPARATOR + groups
    if labels is None:
        raise ValueError("no attribute to mitigate")
    return labels


def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def cell_weights(groups: pd.Series, labels: pd.Series) -> list[dict]:
    """Reweighing weight for every (group, label) cell present in these rows."""
    frame = pd.DataFrame({"group": groups.to_numpy(), "label": labels.to_numpy()})
    n = len(frame)
    if n == 0:
        raise ValueError("no training rows to compute weights on")
    p_group = frame["group"].value_counts() / n
    p_label = frame["label"].value_counts() / n
    cells = []
    for (group, label), count in frame.groupby(["group", "label"]).size().items():
        weight = p_group[group] * p_label[label] / (count / n)
        cells.append({"group": str(group), "label": _plain(label), "n": int(count),
                      "weight": round(float(weight), 6)})
    return sorted(cells, key=lambda c: (c["group"], str(c["label"])))


def row_weights(raw_frame: pd.DataFrame, attributes: list[str], train_index: list,
                y: pd.Series) -> tuple[pd.Series, list[dict]]:
    """
    Per-row training weights and the cells they came from.

    Only the train rows are read: the weights are a fitted quantity.
    """
    if train_index is None:
        raise ValueError("no train split in state; reweighing needs the Data Agent's split")
    train_idx = pd.Index(train_index)
    raw_train = raw_frame.loc[train_idx]
    groups = group_labels(raw_train, attributes)
    labels = y.loc[train_idx]
    cells = cell_weights(groups, labels)
    lookup = {(c["group"], str(c["label"])): c["weight"] for c in cells}
    weights = pd.Series(
        [lookup[(g, str(_plain(v)))] for g, v in zip(groups.to_numpy(), labels.to_numpy())],
        index=train_idx, dtype=float,
    )
    return weights, cells


def snapshot(training_result: dict | None, fairness_result: dict | None) -> dict:
    """The numbers a reviewer compares before and after mitigation."""
    metrics = (training_result or {}).get("selected_model_metrics") or {}
    report = (fairness_result or {}).get("fairness_report") or []
    return {
        "model": (training_result or {}).get("selected_model_name"),
        "auc_roc": metrics.get("auc_roc"),
        "accuracy": metrics.get("accuracy"),
        "overall_fairness_passed": (fairness_result or {}).get("overall_fairness_passed"),
        "violated_attributes": sum(1 for r in report if r.get("violation")) if report else None,
        "attributes": {
            r["attribute"]: {
                "disparate_impact": r.get("disparate_impact"),
                "demographic_parity_difference": r.get("demographic_parity_difference"),
                "violation": r.get("violation"),
            } for r in report
        },
    }


def summary(mitigation: dict | None, training_result: dict | None,
            fairness_result: dict | None) -> dict | None:
    """Gate-payload view: what was mitigated, and the latest before against now."""
    if not mitigation or not mitigation.get("applications"):
        return None
    applications = []
    for app in mitigation["applications"]:
        weights = [c["weight"] for c in app.get("cells") or []]
        applications.append({
            "attribute": app.get("attribute"),
            "status": app.get("status"),
            "reason": app.get("reason"),
            "cells": len(weights),
            "weight_min": min(weights) if weights else None,
            "weight_max": max(weights) if weights else None,
        })
    return {
        "method": mitigation.get("method", METHOD),
        "attributes": list(mitigation.get("attributes") or []),
        "applications": applications,
        "before": mitigation["applications"][-1].get("before"),
        "after": snapshot(training_result, fairness_result),
    }
