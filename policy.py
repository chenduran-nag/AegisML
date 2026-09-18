"""
policy.py
=========
Policy-as-code: the thresholds and governance rules that govern a run, loaded from
policy.yaml, validated, versioned and hashed.

DESIGN / WHY
  - One file states every rule a reviewer might ask about ("why was this column
    dropped?", "why could I not approve?"), and every run records which policy
    governed it: in state, in a `policy_applied` audit event, in the final outcome,
    the AIBOM and the model card.
  - Validation is strict. An unknown key, a wrong type or an out-of-range value
    raises PolicyError at load time: a typo such as `min_grup_size` silently falling
    back to a default would change behaviour without anyone noticing.
  - The hash is taken over the validated policy as canonical JSON, not over the file
    bytes, so comments and formatting do not change it but every value does.
  - Defaults equal the agents' module constants, so a run with no policy in state
    (tests, direct calls) behaves exactly as a run under the default policy.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

import data_agent
import fairness_agent

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "policy.yaml"

# Names the training registry implements (training_agent._REGISTRY).
KNOWN_MODELS = ("LogisticRegression", "RandomForest", "XGBoost", "GradientBoosting",
                "Ridge", "Lasso")

# section -> key -> (type, minimum, maximum). None means unbounded.
_SCHEMA: dict[str, dict[str, tuple]] = {
    "data": {
        "column_drop_null_threshold": (float, 0.0, 1.0),
        "column_high_null_warning_threshold": (float, 0.0, 1.0),
        "row_drop_ratio_limit": (float, 0.0, 1.0),
        "ohe_cardinality_limit": (int, 2, 1000),
        "null_pct_quality_limit": (float, 0.0, 100.0),
        "test_size": (float, 0.05, 0.5),
        "validation_size": (float, 0.05, 0.5),
    },
    "fairness": {
        "disparate_impact_threshold": (float, 0.0, 1.0),
        "demographic_parity_difference_threshold": (float, 0.0, 1.0),
        "min_group_size": (int, 1, 100_000),
    },
    "governance": {
        "max_retries": (int, 0, 10),
        "max_human_reroutes": (int, 0, 10),
        "block_approval_when_fairness_not_evaluated": (bool, None, None),
        "require_dual_signoff_for_violating_approval": (bool, None, None),
        "dual_signoff_can_override_approval_block": (bool, None, None),
    },
    "training": {
        "allowed_models": (list, None, None),
    },
}


class PolicyError(ValueError):
    """The policy file is missing, unparseable or invalid."""


def default_policy() -> dict:
    """The policy the code enforces when none is supplied."""
    return {
        # Tracks policy.yaml's version; a test pins the two together.
        "version": "1.1.0",
        "data": {
            "column_drop_null_threshold": data_agent.COLUMN_DROP_NULL_THRESHOLD,
            "column_high_null_warning_threshold": data_agent.COLUMN_HIGH_NULL_WARNING_THRESHOLD,
            "row_drop_ratio_limit": data_agent.ROW_DROP_RATIO_LIMIT,
            "ohe_cardinality_limit": data_agent.OHE_CARDINALITY_LIMIT,
            "null_pct_quality_limit": data_agent.NULL_PCT_QUALITY_LIMIT,
            "test_size": data_agent.TEST_SIZE,
            "validation_size": data_agent.VALIDATION_SIZE,
        },
        "fairness": {
            "disparate_impact_threshold": fairness_agent.DISPARATE_IMPACT_THRESHOLD,
            "demographic_parity_difference_threshold":
                fairness_agent.DEMOGRAPHIC_PARITY_DIFF_THRESHOLD,
            "min_group_size": fairness_agent.MIN_GROUP_SIZE,
        },
        "governance": {
            # Must equal pipeline_graph.MAX_RETRIES / MAX_HUMAN_REROUTES (a test pins
            # this; importing pipeline_graph here would be circular).
            "max_retries": 2,
            "max_human_reroutes": 2,
            "block_approval_when_fairness_not_evaluated": True,
            "require_dual_signoff_for_violating_approval": True,
            "dual_signoff_can_override_approval_block": False,
        },
        "training": {
            "allowed_models": None,
        },
    }


def _normalise_model(name: str) -> str:
    return str(name).lower().replace("-", "").replace("_", "").replace(" ", "")


def _check(path: str, value: Any, kind: type, minimum: Any, maximum: Any) -> Any:
    if kind is bool:
        if not isinstance(value, bool):
            raise PolicyError(f"'{path}' must be a bool (true/false), got {value!r}")
        return value
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise PolicyError(f"'{path}' must be an integer, got {value!r}")
    elif kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PolicyError(f"'{path}' must be a number, got {value!r}")
        value = float(value)
    elif kind is list:
        if value is None:
            return None
        if not isinstance(value, list) or not value:
            raise PolicyError(f"'{path}' must be null or a non-empty list, got {value!r}")
        known = {_normalise_model(m): m for m in KNOWN_MODELS}
        unknown = [m for m in value if _normalise_model(m) not in known]
        if unknown:
            raise PolicyError(f"'{path}' names unknown model(s) {unknown}; "
                              f"known: {list(KNOWN_MODELS)}")
        return [known[_normalise_model(m)] for m in value]
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise PolicyError(f"'{path}' must be between {minimum} and {maximum}, got {value!r}")
    return value


def validate_policy(raw: Any) -> dict:
    """Return the complete, validated policy: defaults overlaid with `raw`."""
    if not isinstance(raw, dict):
        raise PolicyError("the policy must be a YAML mapping")
    unknown = set(raw) - {"version", *_SCHEMA}
    if unknown:
        raise PolicyError(f"unknown top-level key(s): {sorted(unknown)}")
    version = raw.get("version")
    if not isinstance(version, str) or not version.strip():
        raise PolicyError("the policy needs a non-empty string 'version'")

    policy = default_policy()
    policy["version"] = version.strip()
    for section, fields in _SCHEMA.items():
        given = raw.get(section)
        if given is None:
            continue
        if not isinstance(given, dict):
            raise PolicyError(f"'{section}' must be a mapping")
        unknown = set(given) - set(fields)
        if unknown:
            raise PolicyError(f"unknown key(s) in '{section}': {sorted(unknown)}")
        for key, value in given.items():
            policy[section][key] = _check(f"{section}.{key}", value, *fields[key])

    data = policy["data"]
    if data["column_high_null_warning_threshold"] > data["column_drop_null_threshold"]:
        raise PolicyError("'data.column_high_null_warning_threshold' cannot exceed "
                          "'data.column_drop_null_threshold'")
    return policy


def policy_sha256(policy: dict) -> str:
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_policy(path: str | Path | None = None) -> dict:
    """
    Load and validate a policy file. Raises PolicyError on any problem.

    Returns {"policy", "version", "sha256", "source"}.
    """
    source = Path(path) if path is not None else DEFAULT_POLICY_PATH
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyError(f"cannot read policy file '{source}': {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PolicyError(f"policy file '{source}' is not valid YAML: {exc}") from exc
    policy = validate_policy(raw)
    return {"policy": policy, "version": policy["version"],
            "sha256": policy_sha256(policy), "source": str(source)}


def policy_value(policy: dict | None, section: str, key: str, fallback: Any) -> Any:
    """A value from a run's policy, or the fallback when the run carries none."""
    if not policy:
        return fallback
    return (policy.get(section) or {}).get(key, fallback)


def with_overrides(policy: dict, overrides: dict) -> dict:
    """A validated copy of `policy` with {section: {key: value}} overrides applied."""
    merged = copy.deepcopy(policy)
    for section, values in overrides.items():
        merged.setdefault(section, {}).update(values)
    return validate_policy(merged)
