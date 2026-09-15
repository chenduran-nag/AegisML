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
pytest                      # 251 offline tests, no API key or network needed
python experiments/run_governance_eval.py                  # reproduce the evaluation from the committed cache, no Groq key
python experiments/run_governance_eval.py --summarise-only # rebuild its tables + figure from the saved CSVs
```

Running the app needs `GROQ_API_KEY` in a `.env` file at the repo root (gitignored,
never commit it), then `python server.py` → http://localhost:8000.

## Architecture map

| File | Role |
|---|---|
| `pipeline_graph.py` | LangGraph `StateGraph`: `data_analysis_node` first, 3 feedback loops, `SqliteSaver`, model serialisation in `audit_log_node` |
| `data_analysis_agent.py` | Raw profiling + Chart.js payloads for the EDA page |
| `eda_insights.py` | Turns profiling into routed findings (`data_agent` / `planner` / `reviewer`) + `build_eda_linkage()`; shared column-name matching |
| `graph_state.py` | `PipelineState` TypedDict; DataFrames stored as pickled bytes, models as joblib bytes |
| `planner_agent.py` | The ONLY LLM call (Groq, JSON mode). Sends aggregate stats only |
| `data_agent.py` | Deterministic cleaning. Draws the train/test split, fits everything on train |
| `training_agent.py` | Model registry, leaderboard, SHAP. Reuses the Data Agent's split |
| `fairness_agent.py` | DI + parity difference (the verdict) and TPR/FPR gaps (reported only) on held-out rows. Groups from raw values; age banded; groups under 30 rows excluded |
| `mitigation.py` | Reweighing mitigation for the `reject_and_mitigate` decision: attribute choice, train-only cell weights, before/after summary |
| `policy.py`, `policy.yaml` | Policy-as-code: validated thresholds and governance rules, version + SHA-256 recorded per run |
| `audit_log.py` | Append-only, SHA-256 hash-chained audit log + `verify_audit_chain()` |
| `compliance_artifacts.py` | Model card, AIBOM and Annex IV draft; digests chained into the audit log + `verify_artifacts()` |
| `server.py` | FastAPI: `/api/pipeline/{start,resume,status,audit,eda,artifacts}` |
| `static/index.html` | Single-file dashboard (vanilla JS + Chart.js) |
| `app.py` | Superseded Streamlit UI — do not extend |
| `test_*.py` (repo root) | Legacy manual scripts; need live Groq + network. Not collected by pytest |
| `tests/` | The real, offline pytest suite |
| `experiments/run_governance_eval.py` | Governance evaluation: the real graph with a scripted reviewer, arms A–D (D = mitigation) × datasets × split seeds |
| `experiments/planner_cache/`, `experiments/results/` | **Committed.** Recorded planner responses, and the results they reproduce |

Loops: **1** data-quality auto-retry → planner (max 2). **2** human "reject data
quality" → planner with feedback injected into the prompt (max 2). **3** human
"reject model/fairness" → straight to training with the model excluded. **4** human
"reject and mitigate" → `mitigation_node` → training with reweighing, same candidates.
Loops 2–4 share one rejection cap (`MAX_HUMAN_REROUTES`).

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
   states for this; keep it that way for any new metric. `None` also covers partial
   coverage: no violation, but a protected attribute present in the data could not be
   audited (`fairness_coverage == "partial"`, named in `protected_attributes_unaudited`,
   shown as NOT FULLY EVALUATED).
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
10. **EDA findings are routed, and reviewer findings never reach the LLM.** Suspected
    target leakage and proxy variables (`route == "reviewer"`) are shown at the gate
    only. If the planner saw "X is a proxy for sex" it could write "Drop column X",
    and the Data Agent would execute it — an automated fairness decision nobody
    approved. `compact_findings_for_planner()` enforces this; a test captures the
    real prompt to prove it. A reviewer who wants to act uses Loop 2.
11. **The Data Agent executes only unconditional instructions.** `_parse_plan_steps`
    reads each `;` clause separately, ignores hedged clauses ("consider", "if ..."),
    applies a verb only before a scope terminator ("keep", "redundant with", "("),
    and matches column names longest-first via `columns_named_in()`. Each rule
    fixes a misreading of a real planner step; the phrasings are pinned in tests.
12. **Compliance artifacts are generated, never authored.** Every field in
    `compliance_artifacts.py` is read from recorded state. Never let an LLM write
    a model card, and never emit a plausible blank where a value is missing —
    use the `NOT_RECORDED` marker.
13. **A run with no trained model never reaches the gate.** `route_after_training`
    ends it at `mark_training_failure`. Without that edge, a run whose remaining
    candidates all failed to fit was approved and received a model card for a model
    that did not exist.
14. **Fairness groups come from raw uploaded values, not the cleaned frame.**
    `fairness_node` passes `raw_frame`; scaling and encoding destroy group membership
    (COMPAS 0/1 `sex` became a float, Adult `occupation` a frequency). Groups under
    `MIN_GROUP_SIZE` (30) evaluation rows are excluded and listed, `age` is banded,
    and protected attributes present in the data are audited even if the planner omits
    them. Equal-opportunity and equalized-odds gaps are reported but never change the
    verdict: that rule is a policy decision (Step 4).
    **The verdict covers protected attributes only** (named like one, or declared by the
    reviewer at run start via `declared_protected_attributes`). Other audited attributes
    are advisory (`counts_toward_verdict: False`, `advisory_violations`). With no
    protected attribute evaluated, the verdict is `None` and `fairness_evaluated` is
    False. **Intersectional pairs** of protected attributes go to `intersectional_report`
    and are reported only, never part of the verdict.
15. **Mitigation is reviewer-triggered, deterministic and train-only.** It runs only on a
    human `reject_and_mitigate` decision; the attribute is chosen by `choose_attribute()`,
    never by the LLM. Weights are fitted on train rows with the Fairness Agent's grouping
    (`resolve_groups_from_raw`). State and the audit log hold per-(group, label) cells,
    never per-row weights; `training_node` rebuilds row weights from the raw frame. Once
    applied it stays on for the rest of the run, and the model card must say so. Only
    protected violations are mitigated.
16. **Gate decisions use validation rows; test rows are scored once, after approval.**
    The Data Agent splits train / validation / test (64 / 16 / 20). The leaderboard,
    the fairness audit and every reviewer decision use `_gate_index(split)` (validation).
    Nothing may read the test rows before `audit_log_node`, which scores the approved
    model on them (`_final_test_evaluation`) and logs `final_test_evaluation`. Report
    those numbers; the gate numbers were used to choose.
17. **The run's policy governs it, and is recorded.** The server loads `policy.yaml` once
    (an invalid file stops startup) and puts the validated policy, its version and its
    SHA-256 in state. Nodes read limits through `_policy_value` / `_policy_section` /
    `_fairness_limits`, falling back to module constants only when a run carries no
    policy. Never read a threshold constant directly in a node. A `policy_applied` audit
    event is the first entry of every run that carries a policy. **Approving a
    classification model whose fairness verdict is `None` is refused by default**
    (`block_approval_when_fairness_not_evaluated`): the payload withholds `approve`, the
    API returns 409, and the graph routes a stray approve to `mark_approval_blocked`.
    Regression is exempt.

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
  AIBOM has no planner provenance to record. It also receives `eda_findings=`;
  accept `**kwargs`.
- EDA runs inside the graph (`data_analysis_node`), not in `server.py`. Tests that
  build state by hand do not need to supply `eda_report`.
- To test what the LLM is actually sent without a key, monkeypatch
  `planner_agent._call_groq` and capture its arguments — see
  `test_planner_prompt_carries_planner_findings_but_not_proxies`.
- Importing `pipeline_graph` builds a module-level graph, creating `pipeline_state.db`
  in the working directory (gitignored).
- **pandas 3** uses a `str` dtype for text columns, not `object`. Use
  `_is_encodable_categorical()` in `data_agent.py`; never test `is_object_dtype` alone.
- **Windows + redirected output:** agent `print`s contain `→`. When stdout is piped or
  redirected on Windows it falls back to cp1252 and raises `UnicodeEncodeError`. Set
  `PYTHONIOENCODING=utf-8` when running scripts through a pipe.
- **Windows + `uvicorn --reload` can serve stale code.** Observed here: the reloader
  logged "Reloading...", but every server process still predated the edits, and a
  live run reproduced a parser bug that was already fixed on disk. For any live
  verification, stop the server, start it **without** `--reload`, and confirm the
  process serving port 8000 started after your last edit. A passing test suite does
  not prove the running server has the same code.
- **Planner cache is process-global.** `planner_agent.configure_planner_cache(mode, dir)`
  with `off` (default), `record` or `replay`. Always reset it to `off` afterwards; the
  eval tests do this with an autouse fixture. Replay raises `PlannerCacheMiss` rather
  than silently calling the LLM.
- **Smoke-test the evaluation with `--quick --cache-dir <scratch>`.** Quick runs subsample
  the data, which changes the planner prompts; recording them into the committed
  `experiments/planner_cache/` would pollute it. `--quick` output goes to the
  gitignored `experiments/results_quick/`.
- **Fix evaluation presentation with `--summarise-only`**, which rebuilds the tables and
  figure from `runs.csv` in seconds. Never re-run 60 pipelines for a formatting change.
- **One-hot column names are sanitised for XGBoost** (`[`, `]`, `<` → `(`, `)`, `lt`/`le`).
  Only the generated dummy columns are renamed, so the Fairness Agent's one-hot
  fallback still finds an attribute by its prefix (with a raw frame it reads raw values).
- **`toy_df` is 1,000 rows on purpose.** The gate audits a 160-row validation split
  (200 test rows), about the smallest in which the main groups clear `MIN_GROUP_SIZE`. A test that calls `run_fairness_agent` on a smaller
  frame must pass `min_group_size=` explicitly, or its attributes are skipped. And since
  protected columns are audited automatically, a test that needs "nothing auditable"
  must drop `sex`, `race` and `age` from the frame, not just leave them out of the plan.
