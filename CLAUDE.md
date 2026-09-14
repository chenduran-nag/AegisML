# AegisML — Project Context for Claude Code

Governed AutoML for tabular data. A LangGraph pipeline profiles a CSV, has an LLM
*plan* preprocessing, cleans the data deterministically, trains a leaderboard,
audits demographic fairness, then **pauses at a human governance gate** before a
model is approved. Every step goes to a hash-chained SQLite audit log.

The ML is deliberately ordinary. The contribution is the governance layer —
protect it.

- Task plan: `NEXT_STEPS.md`. Background, landscape scan, defect history: `PROJECT_REVIEW.md`.
- Semester 7 academic project, built with a partner. Evaluated in formal reviews.

## Repository and git

- `origin` should be the user's fork: `https://github.com/chenduran-nag/AegisML`
- `upstream` is the partner's original: `https://github.com/ruhannpn/AegisML`
  (add with `git remote add upstream https://github.com/ruhannpn/AegisML.git`)
- **Never push to `upstream`, and never open a PR against it, unless the user asks
  in that session.** It is someone else's repository.
- One branch per feature. Do not commit or push unless asked.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pytest                      # 59 offline tests, no API key or network needed
```

Running the app needs `GROQ_API_KEY` in a `.env` file at the repo root (gitignored,
never commit it), then `python server.py` → http://localhost:8000.

## Architecture map

| File | Role |
|---|---|
| `pipeline_graph.py` | LangGraph `StateGraph`: nodes, 3 feedback loops, `SqliteSaver`, model serialisation in `audit_log_node` |
| `graph_state.py` | `PipelineState` TypedDict; DataFrames stored as pickled bytes, models as joblib bytes |
| `planner_agent.py` | The ONLY LLM call (Groq, JSON mode). Sends aggregate stats only |
| `data_agent.py` | Deterministic cleaning. Draws the train/test split, fits everything on train |
| `training_agent.py` | Model registry, leaderboard, SHAP. Reuses the Data Agent's split |
| `fairness_agent.py` | Disparate Impact + Demographic Parity Difference on held-out rows |
| `audit_log.py` | Append-only, SHA-256 hash-chained audit log + `verify_audit_chain()` |
| `compliance_artifacts.py` | Model card, AIBOM and Annex IV draft; digests chained into the audit log + `verify_artifacts()` |
| `server.py` | FastAPI: `/api/pipeline/{start,resume,status,audit,eda,artifacts}` |
| `static/index.html` | Single-file dashboard (vanilla JS + Chart.js) |
| `app.py` | Superseded Streamlit UI — do not extend |
| `test_*.py` (repo root) | Legacy manual scripts; need live Groq + network. Not collected by pytest |
| `tests/` | The real, offline pytest suite |

Loops: **1** data-quality auto-retry → planner (max 2). **2** human "reject data
quality" → planner with feedback injected into the prompt (max 2). **3** human
"reject model/fairness" → straight to training with the model excluded.

## Invariants — do not break these

1. **The LLM plans; it never touches data.** Plan text is consumed only by keyword
   matching (`_parse_plan_steps`). Never `eval`/`exec` LLM output, never let it
   generate code that runs.
2. **Only aggregates go to the LLM.** No raw rows in any prompt.
3. **Split before fit.** The Data Agent draws the split right after the structural
   drops. Any new learned preprocessing step must fit on `fit_index` (train rows)
   and only *apply* to the rest. Never `reset_index()` on the cleaned frame — the
   split travels through state as index labels (`PipelineState["split_index"]`).
4. **Unmeasured is not passed.** `overall_fairness_passed` is `True`, `False`, or
   `None` (not evaluated). `None` must never render as a pass. The UI has three
   states for this; keep it that way for any new metric.
5. **The audit log is append-only.** Write only through `log_audit_event()`. Never
   `UPDATE`/`DELETE`. Changing `_canonical_content()` breaks verification of every
   existing entry — if it must change, version it.
6. **`human_approval_node` re-executes on resume.** Put no side effects *before*
   `interrupt()` in it. Counters live in their own small nodes for the same reason.
7. **Only approved models reach disk**, written in `audit_log_node`. The API and UI
   show only the path actually written — never construct one.
8. **Keep per-row data out of audit `details`** (see how `split_index` is lifted out
   of the Data Agent result before logging).
9. **Do not overclaim.** The README states what the audit chain does *not* detect
   and that no compliance certification exists. New features get the same honesty.
   The generated Annex IV pack names its own gaps (sections 8 and 9) on purpose —
   do not quietly turn those into claims.
10. **Compliance artifacts are generated, never authored.** Every field in
    `compliance_artifacts.py` is read from recorded state. Never let an LLM write
    a model card, and never emit a plausible blank where a value is missing —
    use the `NOT_RECORDED` marker.

## Conventions

- Module docstrings carry DESIGN / WHY notes; comments explain *why*, not what.
- Thresholds are module-level constants near the top of each agent.
- Agents append human-readable strings to an `actions` list; these surface in the
  UI and the audit log, so write them for a reviewer.
- Existing files use CRLF line endings; git's `autocrlf` normalises them.

## Testing notes and gotchas

- Fixtures in `tests/conftest.py`: `toy_df`, `toy_regression_df`, `fake_plan`, `audit_db`.
- Graph tests repoint four module-level names on `pipeline_graph` — `plan_pipeline`,
  `AUDIT_DB_PATH`, `SAVED_MODELS_DIR`, `ARTIFACTS_DIR` — and build with
  `build_graph(db_path=tmp)`. All are read at call time, so `monkeypatch.setattr`
  is enough; see the `graph` fixture in `tests/test_graph_end_to_end.py`.
- A `plan_pipeline` stub should populate the `meta_out` dict it is handed, or the
  AIBOM has no planner provenance to record.
- Importing `pipeline_graph` builds a module-level graph, creating `pipeline_state.db`
  in the working directory (gitignored).
- **pandas 3** uses a `str` dtype for text columns, not `object`. Use
  `_is_encodable_categorical()` in `data_agent.py`; never test `is_object_dtype` alone.
- **Windows + redirected output:** agent `print`s contain `→`. When stdout is piped or
  redirected on Windows it falls back to cp1252 and raises `UnicodeEncodeError`. Set
  `PYTHONIOENCODING=utf-8` when running scripts through a pipe.
