# AegisML — Next Steps

Work plan picked up after commit `cc22712` on `chenduran-nag/AegisML` `main`. Read
`CLAUDE.md` first for context and the invariants; `PROJECT_REVIEW.md` has the
reasoning behind this ordering.

**Suggested first message to Claude Code on a new machine:**

> Read CLAUDE.md and NEXT_STEPS.md. Set up the venv and run the test suite, then
> do Step 0. Work on a new branch, don't push anything, and never push to the
> `upstream` remote.

---

## Where things stand

Done: every Tier 0 repair, the hash-chained audit log, and an offline pytest suite
(35 tests). In detail:

- The train/test split is drawn before any preprocessing parameter is fitted.
- Approved models are actually written to `saved_models/`.
- Fairness is measured on held-out rows. Unevaluated fairness reports `None`, shown
  as NOT EVALUATED.
- Categorical encoding works on pandas 3.
- `requirements.txt` is complete, and the README matches the code.

Not done, in recommended order:

| Step | Item | Why this order |
|---|---|---|
| 0 | Live verification with a real Groq key | Nothing since the fixes has run against the real LLM or in a browser |
| 1 | Compliance artifact generation | The project's main differentiator |
| 2 | Quantitative governance evaluation | Turns the demo into a measurable result |
| 3 | Fairness metrics + mitigation | Closes the loop the thesis promises |
| 4 | Policy-as-code | Cheap, big framing gain |
| 5 | Reviewer identity + dual sign-off | An approval with no approver identity is not an audit trail |
| 6 | Small cleanups | Anytime |

Each step below lists the goal, design, files and acceptance criteria. **Every step
ends with `pytest` green, plus new tests for the new behaviour.**

---

## Step 0 — Live verification (do this first, about 30 min)

The planner was stubbed in every test and the dashboard was never clicked through
after the fixes. Confirm the real thing works before building on it.

1. Put `GROQ_API_KEY=...` in `.env`, then run `python server.py`.
2. Upload a UCI Adult CSV, target `income`, task `classification`. (The loader
   lives in `dataset_utils.load_adult_dataset()`; save its frame with `to_csv`.)
3. Check the following:
   - The run pauses at the gate, the leaderboard shows, and the fairness table lists
     `sex`/`race`.
   - **Approve** shows a real `saved_models/<run_id>_<model>.joblib` path, and that
     file exists.
   - The **Audit page** shows a green "HASH CHAIN VERIFIED" banner.
   - Open `audit_log.db` in `sqlite3`, edit one `summary`, reload the audit page. The
     banner turns red and names the entry.
   - Run a **regression** dataset. The fairness badge reads NOT EVALUATED in amber,
     not green.
   - **Reject model** reroutes to training and pauses again with a different model.
   - **Reject data quality** with feedback text reroutes to the planner, and the
     feedback appears in the next plan.
4. Fix anything that breaks. Likely weak spots are the UI handling of the
   `unresolved_quality_issue` / `unresolved_human_rejection` end states, which have
   never been exercised in the browser.

Optional, useful for the review presentation: a `demo_tamper_evidence.py` script
that runs a stubbed pipeline, verifies the chain, tampers with one row, and
re-verifies. Build it from the fixtures in `tests/test_graph_end_to_end.py` plus
`tests/test_audit_chain.py::test_editing_a_summary_is_detected`.

---

## Step 1 — Compliance artifact generation

**Goal.** On approval, generate a Model Card, a technical-documentation pack
structured on the EU AI Act Annex IV headings, and an AI Bill of Materials. Tie
each one into the audit chain so they are tamper-evident too. Nothing in the
open-source agentic-AutoML space does this, and every input already exists in
`PipelineState`.

**New module `compliance_artifacts.py`** (deterministic, zero LLM calls):

- `build_model_card(state, audit_trail, chain) -> dict`
  - Intended use: from `business_objective`.
  - Model: `training_result.selected_model_name` plus the full leaderboard.
  - Metrics on held-out rows.
  - Top SHAP features.
  - Training data: shape, `quality_report` including `train_rows` / `test_rows`, and
    the preprocessing `actions_taken`.
  - Fairness results. Say NOT EVALUATED explicitly when `overall_fairness_passed is None`.
  - Known limitations: continuous sensitive attributes are skipped, SHAP is
    log-odds for linear models, and so on.
  - Human decision history: decisions, feedback and reroute counts from the audit trail.
- `render_model_card_md(card) -> str`: human-readable version.
- `build_aibom(state, artifact_paths) -> dict`
  - Dataset SHA-256 (see below).
  - Model file SHA-256.
  - Python and library versions via `importlib.metadata`: pandas, scikit-learn,
    xgboost, shap, langgraph, groq.
  - LLM model ID and planner prompt hash.
  - Policy version (after Step 4).
  - Approver (after Step 5).
  - Timestamp and audit-chain head hash.
- `build_technical_documentation(state, card, aibom) -> str`: markdown organised
  under the Annex IV section headings. **Check the headings against the current
  official text before finalising.** The source is
  https://artificialintelligenceact.eu/annex/4/ (and Article 11). Title it *draft
  technical documentation aligned to Annex IV* and state that it is not a
  conformity assessment.

**Wiring.**

- Dataset hash: compute `sha256(contents)` of the raw upload bytes in
  `server.py::start_pipeline`, and store it as a new state key `dataset_sha256`.
  Don't hash the pickled `df_bytes` — pickle bytes are not stable across versions.
- Planner metadata: in `planner_node`, record `{model_id, prompt_sha256}` in a new
  state key `planner_meta`. Don't put it inside `plan`, which is LLM output.
- Call the generators in `audit_log_node` after the model is saved. Write to
  `artifacts/<run_id>/`: `model_card.md`, `model_card.json`, `aibom.json`,
  `technical_documentation.md`. Add `artifacts/` to `.gitignore`.
- Then append an audit event `compliance_artifacts_generated` whose `details` holds
  the SHA-256 of every artifact. That puts the artifacts under the hash chain: edit
  a model card afterwards and its hash no longer matches the logged one. Add
  `verify_artifacts(run_id)`, which re-hashes the files and compares them.
- API: `GET /api/pipeline/artifacts/{thread_id}` returns the list, hashes and
  verification status. UI: an "Artifacts" section on the approved banner that
  renders the model card inline.

**Acceptance.**

- [ ] An approved run produces all four files, and the audit event lists their hashes.
- [ ] A rejected or capped run produces no artifacts.
- [ ] A regression run's model card says fairness was NOT EVALUATED, never "passed".
- [ ] Editing an artifact file makes `verify_artifacts` fail.
- [ ] The AIBOM model hash equals the SHA-256 of the saved `.joblib`.
- [ ] Tests cover all of the above, offline.

---

## Step 2 — Quantitative governance evaluation

**Goal.** Measure whether the governance loops actually change outcomes. Today
the answer is a screenshot. A results table is what makes this defensible in a
review.

**Design.** Put it in `experiments/run_governance_eval.py`, separate from the app.

- Datasets: Adult, German Credit, Bank Marketing, and COMPAS if available. Try
  `sklearn.datasets.fetch_openml`. Adult and `credit-g` exist there; **verify the
  exact names and IDs for the others** before relying on them. Record the dataset
  source and hash in the output.
- Arms, all driven through `build_graph()` with scripted decisions via
  `Command(resume=...)`:
  - **A**: approve at the first gate (governance off).
  - **B**: auto-retry loop only, then approve.
  - **C**: scripted reviewer — `reject_model_or_fairness` while any fairness
    violation remains and reroutes remain, otherwise approve.
  - **D** (after Step 3): the same as C but using `reject_and_mitigate`.
- Seeds: parameterise `SPLIT_RANDOM_STATE` in `data_agent.py`. Today it is a module
  constant; make it overridable, ideally through Step 4's policy. Run 5 seeds per
  dataset per arm and report mean ± std.
- Planner reproducibility and cost: add a **record/replay cache** for
  `plan_pipeline`, keyed by a SHA-256 of the prompt. The first run calls Groq and
  stores the response; later runs replay it. Without this, results aren't
  reproducible and the Groq bill grows with every rerun. Also start capturing
  `response.usage` token counts in `planner_agent._call_groq`; they are currently
  discarded.
- Metrics per run: accuracy, AUC, DI and DPD per attribute, number of violations,
  reroutes used, wall-clock time, planner tokens.
- Output: `experiments/results/*.csv`, a markdown summary table, and one chart
  (fairness-vs-accuracy trade-off per arm).

**Report honestly.** Loop 3 only switches to the next-best model. Whether that
improves fairness is an empirical question, and "it mostly doesn't" is a valid,
publishable finding — it is also the motivation for Step 3.

**Acceptance.**

- [ ] One command regenerates the full table from cache without network access.
- [ ] Per-arm results with mean ± std over seeds.
- [ ] The README or report gains a short "Evaluation" section with the table.

---

## Step 3 — Fairness metrics and mitigation

**Goal.** Move from *detecting* bias to being able to *act* on it, and measure it
properly.

**Metrics** (in `fairness_agent.py`, alongside DI and DPD):

- Equal opportunity difference (TPR gap) and equalized odds (max of TPR and FPR
  gaps). The true labels are available: the target column is in `cleaned_df`, and
  evaluation already runs on `eval_index`.
- **Intersectional subgroups**, e.g. `sex × race`. Enforce a minimum group size
  (for example 30 rows) and report small groups as skipped, not as zero-rate
  groups. Tiny groups make DI meaningless.
- **Bucket continuous attributes** such as `age` (<25 / 25–60 / >60). Gotcha: in
  `cleaned_df`, `age` is already *standardised*, so bucket the **raw** values. Get
  them from `bytes_to_df(state["df_bytes"])` aligned on the shared index labels.
  This works because no index is ever reset.

**Mitigation.** Add a fourth gate decision, `reject_and_mitigate`:

- A new `mitigation_node` goes `fairness_node → human_approval_node`, and each use
  counts against `MAX_HUMAN_REROUTES`.
- Use Fairlearn (add `fairlearn` to requirements). Either:
  - **`ThresholdOptimizer`** (post-processing). Caveat: it needs the sensitive
    feature at predict time, so the saved artifact needs a wrapper and the model
    card must state this deployment constraint.
  - **`ExponentiatedGradient`** (reductions, in-training). No predict-time
    dependency, but slower.
  Pick one and document why.
- Fit the mitigator on train rows only (invariant 3). Evaluate on held-out rows.
- Show before and after DI, DPD and accuracy in the gate payload so the reviewer
  sees the cost of mitigation.
- Update `server.py` (allowed decisions), `route_after_human_approval`, and the UI
  decision panel.

**Acceptance.**

- [ ] New metrics appear in the fairness table, with small groups reported as skipped.
- [ ] `age` is evaluated through buckets instead of being skipped.
- [ ] Mitigation reduces DI violation on the Adult `sex` attribute in a test, or the
  test documents that it did not.
- [ ] Arm D is added to the Step 2 evaluation.

---

## Step 4 — Policy-as-code

**Goal.** Turn "a pipeline with thresholds in it" into "a configurable governance
engine", and record which policy governed each run.

- `policy.yaml` at the repo root, plus `policy.py` with a loader that validates the
  schema. Add `pyyaml` to requirements.
- It moves these constants, keeping the module values as defaults:
  - `data_agent.py`: `COLUMN_DROP_NULL_THRESHOLD`,
    `COLUMN_HIGH_NULL_WARNING_THRESHOLD`, `ROW_DROP_RATIO_LIMIT`,
    `OHE_CARDINALITY_LIMIT`, `NULL_PCT_QUALITY_LIMIT`, `TEST_SIZE`,
    `SPLIT_RANDOM_STATE`.
  - `fairness_agent.py`: `DISPARATE_IMPACT_THRESHOLD`,
    `DEMOGRAPHIC_PARITY_DIFF_THRESHOLD`.
  - `pipeline_graph.py`: `MAX_RETRIES`, `MAX_HUMAN_REROUTES`.
  - An allowed-models list that filters the training registry.
- The policy carries `version` and gets a computed SHA-256. Store both in state,
  include them in the `planner_run` and `final_outcome` audit events, and show them
  in the AIBOM (Step 1).
- Optional: select a policy per run via a form field (e.g. `strict` / `default`).

**Acceptance.** Changing a threshold in YAML changes behaviour without code edits.
Every approved run's audit trail names the policy version and hash. An invalid
policy fails loudly at startup.

---

## Step 5 — Reviewer identity and dual sign-off

**Goal.** Record *who* approved, and require a second approver when a model with
fairness violations is approved anyway.

- `ResumeRequest` gains `reviewer_id` and `reviewer_role`. For the scope of an
  academic project, authenticate with per-reviewer API tokens from a gitignored
  config file, sent in a header. Don't build a full auth system, and state the
  threat model honestly in the README.
- Log the reviewer in the `human_decision` audit event.
- Dual sign-off: if the decision is `approve` and `overall_fairness_passed is
  False`, don't finish. Re-interrupt with "awaiting second approval" and require a
  *different* `reviewer_id`. Make the rule a policy flag (Step 4).
- CORS: with credentials in play, replace `allow_origins=["*"]` with an explicit
  origin list before setting `allow_credentials=True` (see the comment in
  `server.py`).

**Acceptance.** A single reviewer can't approve a fairness-violating model alone.
Both identities appear in the audit trail and the model card.

---

## Step 6 — Small cleanups (anytime)

- [ ] `training_agent.py`: remove `n_jobs=-1` from `LogisticRegression`. scikit-learn
      1.8+ warns that it has no effect.
- [ ] `training_agent.py`: `"svm"` maps to `(None, None)`, so the planner can
      recommend a model that is always skipped. Remove SVM from the planner prompt
      lists in `planner_agent._build_prompts`, or implement it.
- [ ] Target is label-encoded twice (Data Agent, then Training Agent). Harmless;
      tidy up if touching that code anyway.
- [ ] `server.py`: call `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`
      at startup so `→` in log prints can't crash the server when stdout is
      redirected to a file on Windows.
- [ ] Retire `app.py` (the Streamlit UI) — delete it or move it to `legacy/` — and
      drop `streamlit` from requirements.
- [ ] Move the legacy root `test_*.py` scripts into `scripts/manual/` and update the
      README.
- [ ] Add a GitHub Actions workflow that runs `pytest` on push. The suite is fully
      offline, so it needs no secrets.
- [ ] Longer term: audit-chain anchoring. Periodically publish the head hash
      outside the database (a signed log, another host, a timestamping service).
      This closes the truncation gap documented in the README.

---

## Collaboration note

These changes live on the fork (`chenduran-nag/AegisML`), not the partner's repo
(`ruhannpn/AegisML`). When they're ready to share, open a PR from the fork into
upstream — and only after the user asks.
