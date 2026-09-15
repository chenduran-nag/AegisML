"""
compliance_artifacts.py
=======================
Step 7 of the AI-Governed Multi-Agent Platform.

Generates the paper trail a governance regime actually asks for, from state the
pipeline has already computed:

  - model_card.md / model_card.json  — Model Card (architecture, intended use,
    metrics, fairness results, limitations, training-data characteristics).
  - technical_documentation.md       — draft documentation organised under the
    EU AI Act Annex IV section headings.
  - aibom.json                       — AI Bill of Materials: dataset hash, model
    hash, library versions, LLM model id and prompt hashes, approver, audit
    chain head.

DESIGN PRINCIPLES:
  - Zero LLM calls. Every field is read from PipelineState or computed locally,
    so the documentation cannot hallucinate a metric the pipeline never produced.
  - Pure functions over a state dict. This module imports no graph code, so it
    can be unit-tested without LangGraph and reused by an offline exporter.
  - Missing state degrades to an explicit "not recorded" marker rather than a
    plausible-looking blank. A governance document that silently omits a field is
    worse than one that says the field is missing.

WHY THE ARTIFACTS ARE HASHED INTO THE AUDIT CHAIN:
  generate_artifacts() returns a SHA-256 for every file it writes, and
  pipeline_graph logs those digests as a 'compliance_artifacts_generated' audit
  event. Because that event is itself hash-chained (see audit_log.py), editing a
  model card after the fact leaves its digest disagreeing with the one recorded
  in a tamper-evident log. verify_artifacts() performs that comparison.

  The limits of audit_log.py's chain apply here too: this detects modification of
  an artifact, not deletion of both the artifact and its log entry. See the
  README section "Audit Chain: What Is and Is Not Guaranteed".

HONEST SCOPE:
  technical_documentation.md follows the Annex IV *headings* so that a reviewer
  can see which obligations the system has evidence for and which it does not.
  It is a draft input to technical documentation, NOT a conformity assessment,
  and it is labelled as such in the document itself. Several Annex IV items
  (notably 8, the EU declaration of conformity) cannot be produced by a tool at
  all and are reported as out of scope.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from typing import Any, Optional

from audit_log import DEFAULT_AUDIT_DB, get_audit_trail, verify_audit_chain
from eda_insights import build_eda_linkage

# Artifacts are written to <ARTIFACT_ROOT>/<run_id>/.
ARTIFACT_ROOT = "artifacts"

NOT_RECORDED = "not recorded"

# Libraries whose versions materially change model behaviour or outputs.
TRACKED_LIBRARIES = (
    "pandas", "numpy", "scikit-learn", "xgboost", "shap", "joblib",
    "langgraph", "groq",
)

ARTIFACT_EVENT_TYPE = "compliance_artifacts_generated"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    """SHA-256 of a file's bytes, streamed so large model files stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(128 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _library_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in TRACKED_LIBRARIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def fairness_verdict(fairness_result: dict | None) -> str:
    """
    Render the fairness outcome as one of three words.

    Mirrors invariant 4: `overall_fairness_passed is None` means no pass could be
    recorded, and must never be presented as one. It is NOT FULLY EVALUATED when
    some attributes were audited but a protected attribute in the data was not.
    """
    fr = fairness_result or {}
    passed = fr.get("overall_fairness_passed")
    if passed is None:
        if fr.get("fairness_report") and fr.get("protected_attributes_unaudited"):
            return "NOT FULLY EVALUATED"
        return "NOT EVALUATED"
    return "PASSED" if passed else "VIOLATION DETECTED"


def _md_cell(value: Any) -> str:
    """
    Render one table cell.

    Pipes are escaped and newlines folded: cell content is arbitrary text — a
    reviewer's feedback, a skip reason, a JSON blob — and an unescaped pipe
    silently adds a column, corrupting the whole rendered table.
    """
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_None recorded._"
    out = ["| " + " | ".join(_md_cell(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_md_cell(c) for c in row) + " |")
    return "\n".join(out)


def _bullets(items: list[Any] | None, empty: str = "_None recorded._") -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


def build_model_card(
    state: dict,
    run_id: str,
    audit_trail: Optional[list[dict]] = None,
) -> dict:
    """
    Assemble a Model Card from pipeline state.

    Parameters
    ----------
    state : dict
        PipelineState values (as returned by graph.get_state(...).values).
    run_id : str
        The pipeline thread_id.
    audit_trail : list[dict], optional
        Entries from get_audit_trail(run_id). Used to reconstruct the full
        sequence of human decisions rather than just the final one — a card that
        shows only the approval hides that two models were rejected first.
    """
    plan = state.get("plan") or {}
    data_res = state.get("data_agent_result") or {}
    train_res = state.get("training_result") or {}
    fair_res = state.get("fairness_result") or {}
    eda = state.get("eda_report") or {}
    eda_summary = eda.get("summary") or {}
    quality = data_res.get("quality_report") or {}

    decisions = []
    for entry in (audit_trail or []):
        if entry.get("event_type") == "human_decision":
            details = entry.get("details") or {}
            decisions.append({
                "timestamp": entry.get("timestamp"),
                "decision": details.get("human_decision"),
                "feedback": details.get("human_feedback") or "",
            })

    # Limitations are stated unconditionally where they are properties of the
    # implementation, and conditionally where they depend on this run.
    limitations = [
        "Age is audited in fixed bands (<25, 25-59, 60+); other continuous "
        "attributes are not audited. Groups with fewer evaluation rows than the "
        "minimum group size are excluded from the comparison and listed separately.",
        "The verdict uses outcome rates only (disparate impact and demographic parity "
        "difference). Equal-opportunity and equalized-odds differences are reported "
        "but do not affect it, and calibration across groups is not measured.",
        "Sensitive attributes remain in the feature matrix; the model may use them "
        "directly as predictors.",
        "The leaderboard reports a single 80/20 hold-out split, not "
        "cross-validation, so metric estimates carry a correspondingly wide "
        "uncertainty that is not quantified here.",
    ]
    if train_res.get("selected_model_name") in ("LogisticRegression", "Ridge", "Lasso"):
        limitations.append(
            "SHAP values for this linear model are in log-odds space, suitable "
            "for ranking features but not as user-facing effect sizes."
        )
    unaudited = [p.get("attribute") for p in fair_res.get("protected_attributes_unaudited") or []]
    if fair_res.get("overall_fairness_passed") is None and not fair_res.get("fairness_report"):
        limitations.append(
            "Fairness was NOT evaluated for this run. No fairness claim of any "
            "kind is supported by this document."
        )
    elif unaudited:
        limitations.append(
            "Protected attribute(s) present in the data could not be audited: "
            + ", ".join(unaudited) + ". No fairness claim about "
            + ("them" if len(unaudited) > 1 else "it")
            + " is supported, and no overall pass was recorded."
        )
    if state.get("unresolved_quality_issue"):
        limitations.append(
            "The data-quality gate never passed; the retry cap was reached. "
            "Training proceeded on data the pipeline itself flagged as deficient."
        )

    return {
        "schema": "aegisml.model_card/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,

        "model_details": {
            "selected_model": train_res.get("selected_model_name", NOT_RECORDED),
            "task_type": state.get("task_type", NOT_RECORDED),
            "target_column": state.get("target_column", NOT_RECORDED),
            "leaderboard": train_res.get("leaderboard", []),
            "artifact_path": state.get("model_saved_path"),
        },

        "intended_use": {
            "stated_objective": state.get("business_objective") or NOT_RECORDED,
            "planner_reasoning": plan.get("reasoning", NOT_RECORDED),
            "out_of_scope": (
                "Not assessed for deployment on any population other than the "
                "uploaded dataset's, and not validated for use as the sole basis "
                "of a decision about an individual."
            ),
        },

        "training_data": {
            "rows_total": eda_summary.get("total_rows", NOT_RECORDED),
            "columns_total": eda_summary.get("total_columns", NOT_RECORDED),
            "missing_cells_pct": eda_summary.get("missing_pct", NOT_RECORDED),
            "train_rows": quality.get("train_rows", NOT_RECORDED),
            "test_rows": quality.get("test_rows", NOT_RECORDED),
            "class_balance_ratio": quality.get("class_balance_ratio"),
            "rows_dropped": quality.get("rows_dropped", NOT_RECORDED),
            "columns_dropped": quality.get("columns_dropped", []),
            "quality_check_passed": data_res.get("quality_check_passed"),
            "identified_concerns": plan.get("data_quality_concerns", []),
            "preprocessing_applied": data_res.get("actions_taken", []),
        },

        "evaluation": {
            "metrics": train_res.get("selected_model_metrics", {}),
            "evaluated_on": (
                f"{quality.get('test_rows', '?')} held-out rows, excluded from "
                f"every fitted preprocessing parameter and from model training"
            ),
        },

        "fairness": {
            "verdict": fairness_verdict(fair_res),
            "evaluated": fair_res.get("fairness_evaluated", False),
            "evaluated_rows": fair_res.get("evaluated_rows"),
            "thresholds": {
                "disparate_impact_min": 0.80,
                "demographic_parity_difference_max": 0.10,
                "min_group_size": fair_res.get("min_group_size"),
            },
            "report": fair_res.get("fairness_report", []),
            "attributes_skipped": fair_res.get("attributes_skipped", []),
            "coverage": fair_res.get("fairness_coverage") or NOT_RECORDED,
            "protected_attributes_unaudited": fair_res.get("protected_attributes_unaudited", []),
            "candidates_proposed": plan.get("sensitive_attribute_candidates", []),
        },

        "data_insights": {
            "findings": build_eda_linkage(
                state.get("eda_findings") or [], plan, data_res, fair_res,
            ),
        },

        "explainability": {
            "method": "SHAP (TreeExplainer / LinearExplainer)",
            "top_features": train_res.get("shap_summary", []),
        },

        "human_governance": {
            "final_decision": state.get("human_decision", NOT_RECORDED),
            "final_feedback": state.get("human_feedback") or "",
            "decision_history": decisions,
            "automated_quality_retries": state.get("retry_count", 0),
            "human_reroutes": state.get("rejection_reroute_count", 0),
            "unresolved_quality_issue": state.get("unresolved_quality_issue", False),
            "unresolved_human_rejection": state.get("unresolved_human_rejection", False),
        },

        "limitations": limitations,
    }


def render_model_card_md(card: dict) -> str:
    md = card["model_details"]
    use = card["intended_use"]
    data = card["training_data"]
    ev = card["evaluation"]
    fair = card["fairness"]
    expl = card["explainability"]
    gov = card["human_governance"]

    leaderboard_rows = [
        [e.get("model_name"),
         "yes" if e.get("trained_successfully") else "no",
         json.dumps(e.get("metrics", {})),
         e.get("skip_reason", "")]
        for e in md.get("leaderboard", [])
    ]
    def _groups_compared(f: dict) -> str:
        gd = f.get("group_details") or {}
        return (f"{gd.get('group_a')} (n={gd.get('group_a_n', '?')}) vs "
                f"{gd.get('group_b')} (n={gd.get('group_b_n', '?')})")

    fairness_rows = [
        [f.get("attribute"),
         {True: "yes", False: "no"}.get(f.get("protected"), NOT_RECORDED),
         f.get("disparate_impact"),
         f.get("demographic_parity_difference"),
         f.get("equal_opportunity_difference"),
         "VIOLATION" if f.get("violation") else "passed",
         _groups_compared(f)]
        for f in fair.get("report", [])
    ]
    shap_rows = [[e.get("feature"), e.get("importance")]
                 for e in expl.get("top_features", [])]
    decision_rows = [[d.get("timestamp"), d.get("decision"), d.get("feedback")]
                     for d in gov.get("decision_history", [])]
    insight_rows = [
        [f.get("severity"), f.get("type"), ", ".join(f.get("columns") or []),
         f.get("evidence"),
         "; ".join(f"{o.get('stage')}: {o.get('status')}" for o in f.get("outcomes", []))]
        for f in (card.get("data_insights") or {}).get("findings", [])
    ]

    return f"""# Model Card — {md['selected_model']}

**Run ID:** `{card['run_id']}`
**Generated:** {card['generated_at']}
**Fairness verdict:** **{fair['verdict']}**

> Generated automatically by AegisML from recorded pipeline state. Every figure
> below was produced by the run it describes; none of it is authored prose.

---

## 1. Model details

| Field | Value |
|---|---|
| Selected model | `{md['selected_model']}` |
| Task type | {md['task_type']} |
| Target column | `{md['target_column']}` |
| Serialised artifact | `{md['artifact_path'] or 'NOT SAVED'}` |

### Candidate leaderboard

{_md_table(["Model", "Trained", "Metrics", "Skip reason"], leaderboard_rows)}

---

## 2. Intended use

**Stated objective:** {use['stated_objective']}

**Planner reasoning:** {use['planner_reasoning']}

**Out of scope:** {use['out_of_scope']}

---

## 3. Training data

| Field | Value |
|---|---|
| Rows / columns | {data['rows_total']} / {data['columns_total']} |
| Missing cells | {data['missing_cells_pct']}% |
| Train / test rows | {data['train_rows']} / {data['test_rows']} |
| Class balance (majority:minority) | {data['class_balance_ratio']} |
| Rows dropped | {data['rows_dropped']} |
| Columns dropped | {data['columns_dropped'] or 'none'} |
| Quality gate passed | {data['quality_check_passed']} |

### Data quality concerns identified

{_bullets(data['identified_concerns'])}

### Preprocessing applied

{_bullets(data['preprocessing_applied'])}

### Findings from exploratory analysis

Each finding is routed to the stage allowed to act on it: structural issues to the
Data Agent, judgement-light ones to the Planner, and suspected target leakage or
proxy variables to the human reviewer only.

{_md_table(["Severity", "Finding", "Column(s)", "Evidence", "How it was used"], insight_rows)}

---

## 4. Evaluation

**Metrics:** `{json.dumps(ev['metrics'])}`

**Evaluated on:** {ev['evaluated_on']}

---

## 5. Fairness

**Verdict: {fair['verdict']}** — thresholds: Disparate Impact >= \
{fair['thresholds']['disparate_impact_min']}, Demographic Parity Difference <= \
{fair['thresholds']['demographic_parity_difference_max']}.

Measured on {fair['evaluated_rows']} held-out rows. Groups with fewer than {fair['thresholds'].get('min_group_size')} rows are excluded from the comparison. Equal-opportunity difference is reported but does not affect the verdict.

{_md_table(["Attribute", "Protected", "Disparate impact", "Parity difference", "Equal opportunity difference", "Status", "Groups compared"], fairness_rows)}

**Protected attributes present but not audited:** {', '.join(p.get('attribute', '') for p in fair.get('protected_attributes_unaudited') or []) or 'none'}

**Candidates proposed by the planner:** {', '.join(fair['candidates_proposed']) or 'none'}

**Skipped (with reason):**

{_bullets(fair['attributes_skipped'])}

---

## 6. Explainability

Method: {expl['method']}

{_md_table(["Feature", "Mean absolute SHAP"], shap_rows)}

---

## 7. Human governance

| Field | Value |
|---|---|
| Final decision | **{gov['final_decision']}** |
| Automated quality retries | {gov['automated_quality_retries']} |
| Human reroutes | {gov['human_reroutes']} |
| Unresolved quality issue | {gov['unresolved_quality_issue']} |
| Unresolved human rejection | {gov['unresolved_human_rejection']} |

### Decision history

{_md_table(["Timestamp", "Decision", "Feedback"], decision_rows)}

---

## 8. Limitations

{_bullets(card['limitations'])}
"""


# ---------------------------------------------------------------------------
# AI Bill of Materials
# ---------------------------------------------------------------------------


def build_aibom(state: dict, run_id: str, chain: Optional[dict] = None) -> dict:
    """
    Inventory every input that determined this model, with content hashes.

    The point of an AIBOM is reproducibility and provenance: given this document,
    a reviewer can tell whether a rerun used the same data, the same library
    versions and the same planner model.
    """
    planner_meta = state.get("planner_meta") or {}
    train_res = state.get("training_result") or {}
    model_path = state.get("model_saved_path")

    model_entry: dict[str, Any] = {
        "name": train_res.get("selected_model_name", NOT_RECORDED),
        "path": model_path,
        "sha256": None,
        "size_bytes": None,
    }
    if model_path and os.path.isfile(model_path):
        model_entry["sha256"] = sha256_file(model_path)
        model_entry["size_bytes"] = os.path.getsize(model_path)

    return {
        "schema": "aegisml.aibom/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,

        "dataset": {
            "sha256": state.get("dataset_sha256", NOT_RECORDED),
            "target_column": state.get("target_column", NOT_RECORDED),
            "task_type": state.get("task_type", NOT_RECORDED),
            "rows": ((state.get("eda_report") or {}).get("summary") or {}).get("total_rows"),
            "columns": ((state.get("eda_report") or {}).get("summary") or {}).get("total_columns"),
        },

        "model": model_entry,

        "llm": {
            "role": "planning only — never used to transform data or generate executed code",
            "model_id": planner_meta.get("model_id", NOT_RECORDED),
            "system_prompt_sha256": planner_meta.get("system_prompt_sha256", NOT_RECORDED),
            "user_prompt_sha256": planner_meta.get("user_prompt_sha256", NOT_RECORDED),
            "token_usage": planner_meta.get("token_usage"),
        },

        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "libraries": _library_versions(),
        },

        "governance": {
            "approver": state.get("reviewer_id", NOT_RECORDED),
            "final_decision": state.get("human_decision", NOT_RECORDED),
            "policy_version": state.get("policy_version", NOT_RECORDED),
            "audit_chain_head": (chain or {}).get("head_hash"),
            "audit_chain_verified": (chain or {}).get("verified"),
        },
    }


# ---------------------------------------------------------------------------
# Annex IV draft technical documentation
# ---------------------------------------------------------------------------


def build_technical_documentation(state: dict, card: dict, aibom: dict) -> str:
    """
    Draft documentation laid out under the EU AI Act Annex IV headings.

    Where the pipeline holds evidence, it is inserted. Where it does not, the
    section says so explicitly — the value of this document for a review is
    partly in showing which obligations are NOT yet covered.
    """
    md = card["model_details"]
    data = card["training_data"]
    fair = card["fairness"]
    gov = card["human_governance"]
    libs = aibom["runtime"]["libraries"]
    insights = (card.get("data_insights") or {}).get("findings", [])
    n_insights = len(insights)
    n_reviewer_insights = sum(1 for f in insights if f.get("route") == "reviewer")

    lib_rows = [[name, ver] for name, ver in sorted(libs.items())]

    return f"""# Draft Technical Documentation — AegisML run `{card['run_id']}`

> **Status: DRAFT — NOT A CONFORMITY ASSESSMENT.**
> Structured under the section headings of Annex IV of Regulation (EU) 2024/1689
> (the AI Act) so that a reviewer can see which obligations this system currently
> holds evidence for. It is generated from recorded pipeline state by
> `compliance_artifacts.py`. No conformity assessment has been carried out, no
> notified body has been involved, and no CE marking or declaration of conformity
> exists for this system. Sections marked OUT OF SCOPE cannot be produced by a
> tool and require an organisational process.

Generated: {card['generated_at']}

---

## 1. General description of the AI system

- **(a) Intended purpose / provider / version.** Purpose: {card['intended_use']['stated_objective']}.
  Provider: not recorded — AegisML is an academic research prototype.
  Model version: run `{card['run_id']}`, selected algorithm `{md['selected_model']}`.
- **(b) Interaction with external hardware or software.** Groq API for planning
  only (model `{aibom['llm']['model_id']}`). No other external dependency at
  inference time.
- **(c) Software versions.** See section 2(c).
- **(d) Form of market placement.** Not placed on any market. Local research
  deployment via a FastAPI server on `localhost`.
- **(e) Hardware requirements.** Commodity CPU. No accelerator required.
- **(f) Photographs / illustrations.** Not applicable.
- **(g) User interface for deployers.** Single-page governance dashboard showing
  data profiling, the LLM plan, the model leaderboard, fairness metrics, and the
  human approval gate.
- **(h) Instructions for use.** See `README.md`.

---

## 2. Detailed description of elements and development process

- **(a) Development methods and third-party tools.** Multi-agent pipeline
  orchestrated with LangGraph. Preprocessing and training are deterministic
  scikit-learn / XGBoost. An LLM produces a *plan* only; its free-text output is
  consumed by keyword matching and never executed. No pre-trained model is
  incorporated into the delivered artifact.
- **(b) Design specifications and algorithms.** Candidate algorithms were proposed
  by the planner and trained in full; selection is by a fixed ranking metric
  (AUC for classification, RMSE for regression). Leaderboard in the model card.
- **(c) System architecture and computational resources.** Python {aibom['runtime']['python']}
  on {aibom['runtime']['platform']}.

{_md_table(["Library", "Version"], lib_rows)}

- **(d) Data requirements and dataset characteristics.** Dataset SHA-256
  `{aibom['dataset']['sha256']}`; {data['rows_total']} rows x {data['columns_total']} columns;
  {data['missing_cells_pct']}% missing cells; split {data['train_rows']} train /
  {data['test_rows']} test. Preprocessing applied is itemised in the model card.
  Exploratory analysis derived {n_insights} structured finding(s), of which
  {n_reviewer_insights} (suspected target leakage or proxy variables for protected
  attributes) were held for the human reviewer rather than acted on automatically;
  each is listed with how it was used in the model card.
  **Provenance, lawful basis and labelling methodology of the uploaded dataset are
  OUT OF SCOPE for this tool** — they are properties of the data supplier.
- **(e) Human oversight measures.** Execution suspends at a governance gate before
  any model is treated as approved. The reviewer may approve, reject on data
  quality (returning to the planner with written directives), or reject the model
  or its fairness (returning to training with that model excluded). This run:
  {gov['human_reroutes']} human reroute(s), {gov['automated_quality_retries']}
  automated retry/retries, final decision **{gov['final_decision']}**.
  **Limitation: reviewer identity is not yet authenticated** — see `NEXT_STEPS.md`
  Step 5. An approval currently records the decision but not a verified approver.
- **(f) Pre-determined changes and continuous compliance.** None. Each run is
  independent; the system performs no online learning.
- **(g) Validation and testing procedures.** Single stratified 80/20 hold-out.
  Every fitted preprocessing parameter (imputation values, frequency maps,
  feature scaler) is fitted on the train rows only; fairness is measured on the
  held-out rows. Metrics: `{json.dumps(card['evaluation']['metrics'])}`.
  **No cross-validation and no separate validation set**, so model selection and
  reporting share one hold-out split.
- **(h) Cybersecurity measures.** None implemented. The API is unauthenticated and
  intended for localhost research use. Uploaded data is processed in memory and
  checkpointed to a local SQLite file. **Not suitable for deployment as-is.**

---

## 3. Monitoring, functioning and control

Every agent execution, metric evaluation and human decision is appended to a
SHA-256 hash-chained audit log (`audit_log.db`). Chain head:
`{aibom['governance']['audit_chain_head']}`; verified at generation time:
{aibom['governance']['audit_chain_verified']}.

Foreseeable failure modes recorded by the system itself: the data-quality gate
failing to the retry cap (`unresolved_quality_issue` = {gov['unresolved_quality_issue']})
and the human-rejection cap being reached
(`unresolved_human_rejection` = {gov['unresolved_human_rejection']}).

**Gap: there is no post-deployment monitoring.** The pipeline ends at approval.

---

## 4. Appropriateness of performance metrics

Classification is ranked by AUC-ROC with accuracy and F1 also recorded; regression
by RMSE with MAE and R². Fairness uses Disparate Impact (four-fifths rule, >= 0.80)
and Demographic Parity Difference (<= 0.10).

**Known inadequacy:** the verdict compares outcome *rates* only. Equal-opportunity and
equalized-odds differences are reported per attribute but do not affect it, and
calibration across groups is not measured, so a "{fair['verdict']}" verdict here is
narrower than the everyday meaning of "fair". Groups with fewer evaluation rows than the
minimum group size are excluded from the comparison.

---

## 5. Risk management system

Partial and automated rather than organisational. Implemented controls: a
deterministic data-quality gate with bounded automatic retries; a fairness audit
against fixed thresholds; a mandatory human approval gate before approval; bounded
human reroute loops; and a tamper-evident audit trail.

**Not implemented:** documented risk analysis, residual-risk evaluation, risk
acceptance criteria, or a named accountable owner. These are process artifacts, not
code, and are OUT OF SCOPE for this tool.

---

## 6. Relevant changes through the lifecycle

Not applicable to a single run. Version identity is the run id plus the hashes in
`aibom.json`. There is no model registry or change-approval workflow.

---

## 7. Harmonised standards applied

None. No harmonised standard under the AI Act has been applied, and no CEN/CENELEC
deliverable has been used. Design was informed by the control objectives of the EU
AI Act and the NIST AI Risk Management Framework, which is not the same as
conformity with either.

---

## 8. EU declaration of conformity

**OUT OF SCOPE.** No declaration of conformity exists. This system has not been
placed on the market and no conformity assessment has been performed.

---

## 9. Post-market monitoring plan

**NOT IMPLEMENTED.** No drift detection, performance-decay alerting or incident
reporting exists. The single largest gap between this prototype and a deployable
high-risk system, and the reason `NEXT_STEPS.md` Step 3 exists.
"""


# ---------------------------------------------------------------------------
# Generation and verification
# ---------------------------------------------------------------------------


def generate_artifacts(
    state: dict,
    run_id: str,
    out_root: str = ARTIFACT_ROOT,
    audit_db_path: str = DEFAULT_AUDIT_DB,
) -> dict:
    """
    Write the artifact set for a run and return a manifest of digests.

    Returns
    -------
    dict
        {"run_id", "directory", "files": {name: {"path", "sha256", "size_bytes"}},
         "errors": [...]}
        `errors` is non-empty if a document failed to render; the rest are still
        written, because a partial paper trail beats none.
    """
    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Verified at generation time so the head hash recorded in the AIBOM is the
    # chain state these artifacts describe.
    chain = verify_audit_chain(run_id, db_path=audit_db_path)
    trail = get_audit_trail(run_id, db_path=audit_db_path)

    errors: list[str] = []
    documents: dict[str, str] = {}

    try:
        card = build_model_card(state, run_id, audit_trail=trail)
        documents["model_card.json"] = json.dumps(card, indent=2, default=str)
        documents["model_card.md"] = render_model_card_md(card)
    except Exception as exc:
        errors.append(f"model_card: {type(exc).__name__}: {exc}")
        card = None

    try:
        aibom = build_aibom(state, run_id, chain=chain)
        documents["aibom.json"] = json.dumps(aibom, indent=2, default=str)
    except Exception as exc:
        errors.append(f"aibom: {type(exc).__name__}: {exc}")
        aibom = None

    if card is not None and aibom is not None:
        try:
            documents["technical_documentation.md"] = build_technical_documentation(
                state, card, aibom
            )
        except Exception as exc:
            errors.append(f"technical_documentation: {type(exc).__name__}: {exc}")

    files: dict[str, dict] = {}
    for name, text in documents.items():
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        files[name] = {
            "path": path,
            "sha256": sha256_text(text),
            "size_bytes": len(text.encode("utf-8")),
        }

    return {
        "run_id": run_id,
        "directory": out_dir,
        "files": files,
        "errors": errors,
    }


def verify_artifacts(
    run_id: str,
    audit_db_path: str = DEFAULT_AUDIT_DB,
) -> dict:
    """
    Re-hash the artifacts on disk and compare against the digests recorded in the
    hash-chained audit log.

    Because the recording event is itself chained, an attacker must break the
    audit chain to make a modified artifact verify — which verify_audit_chain()
    reports separately.

    Returns
    -------
    dict
        verified   : bool | None (None when no artifacts were ever generated)
        files      : list[dict] per-file status
        chain      : the audit chain verdict, since artifact integrity is only as
                     trustworthy as the log holding the expected digests
        detail     : one-line human-readable summary
    """
    trail = get_audit_trail(run_id, db_path=audit_db_path)
    chain = verify_audit_chain(run_id, db_path=audit_db_path)

    events = [e for e in trail if e.get("event_type") == ARTIFACT_EVENT_TYPE]
    if not events:
        return {
            "verified": None,
            "files": [],
            "chain": chain,
            "detail": "No compliance artifacts were generated for this run.",
        }

    recorded = ((events[-1].get("details") or {}).get("files")) or {}
    results = []
    all_ok = True

    for name, meta in recorded.items():
        path = (meta or {}).get("path")
        expected = (meta or {}).get("sha256")
        if not path or not os.path.isfile(path):
            all_ok = False
            results.append({"file": name, "status": "MISSING", "path": path,
                            "expected_sha256": expected, "actual_sha256": None})
            continue
        actual = sha256_file(path)
        ok = actual == expected
        all_ok = all_ok and ok
        results.append({
            "file": name,
            "status": "OK" if ok else "MODIFIED",
            "path": path,
            "expected_sha256": expected,
            "actual_sha256": actual,
        })

    if all_ok and chain.get("verified"):
        detail = f"All {len(results)} artifact(s) match the digests recorded in a verified audit chain."
    elif all_ok:
        detail = (
            f"All {len(results)} artifact(s) match their recorded digests, but the "
            f"audit chain itself is BROKEN, so those digests cannot be trusted."
        )
    else:
        bad = [r["file"] for r in results if r["status"] != "OK"]
        detail = f"Artifact integrity FAILED for: {', '.join(bad)}"

    return {
        "verified": all_ok,
        "files": results,
        "chain": chain,
        "detail": detail,
    }
