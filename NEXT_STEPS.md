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

Done: every Tier 0 repair, the hash-chained audit log, compliance artifact
generation, trustworthy fairness metrics, reviewer-triggered mitigation, and an offline
pytest suite (205 tests). In detail:

- The train/test split is drawn before any preprocessing parameter is fitted.
- Approved models are actually written to `saved_models/`.
- Fairness is measured on held-out rows. Unevaluated fairness reports `None`, shown
  as NOT EVALUATED.
- Categorical encoding works on pandas 3.
- Approval generates a model card, an AIBOM and an Annex IV draft, with their
  digests chained into the audit log and re-verified on the dashboard.
- `requirements.txt` is complete, and the README matches the code.

Remaining, in recommended order:

| Step | Item | Why this order |
|---|---|---|
| 0 | Live verification with a real Groq key | **DONE** — 3 bugs found and fixed |
| 1 | Compliance artifact generation | **DONE** — `compliance_artifacts.py`, 24 new tests |
| — | EDA-driven pipeline | **DONE** — findings routed to Data Agent / Planner / reviewer |
| 2 | Quantitative governance evaluation | **DONE** — no arm approved a compliant model; replay-verified |
| 3a | Trustworthy fairness metrics | **DONE** — re-run showed the first run *understated* disparity |
| 3b | Verdict coverage, mitigation, protected-only verdict, validation gate, intersectional | **DONE** — re-run: no approval passes; reweighing beats model switching; small datasets leave the gate blind |
| 4 | Policy-as-code | **DONE** — `policy.yaml`; approving an unevaluated model blocked by default |
| 5 | Reviewer identity + dual sign-off | An approval with no approver identity is not an audit trail |
| 6 | Small cleanups | Anytime |

Each step below lists the goal, design, files and acceptance criteria. **Every step
ends with `pytest` green, plus new tests for the new behaviour.**

---

## Step 0 — Live verification — DONE

Run against the live Groq planner (`openai/gpt-oss-20b`) on a 10,000-row UCI Adult
sample, driven through the real dashboard. XGBoost won with AUC 0.908, and the
Fairness Agent found genuine violations on `sex` (DI 0.260), `race` (DI 0.235) and
`marital-status` (DI 0.000) — all well documented for this dataset.

Verified end to end: target auto-detection, the governance gate, approval writing a
real model plus all four compliance artifacts, the hash chain (7 entries) and
artifact integrity badges, a regression run reporting NOT EVALUATED in amber, the
rejection cap, and the human-feedback loop — a reviewer directive to drop `fnlwgt`
reached the planner, came back as "Drop the 'fnlwgt' column...", and the Data Agent
acted on it (`columns_dropped: ['fnlwgt']`).

**Three bugs found and fixed, all in `static/index.html`:**

1. **Terminated runs rendered as approved.** A run that hit the rejection cap
   reports `status: "completed"`, fell into the approved branch, and displayed
   "Model Formally Approved & Saved to Disk!" — presenting a rejected, terminated
   run as a successful deployment. The worst possible failure for a governance
   dashboard. There is now a separate red "Pipeline Terminated Without Approval"
   banner, the approved banner requires an actual approval, and
   `test_rejection_cap_terminates_without_approval_or_artifacts` pins it down.
2. **Temporal dead zone in `renderPipelineResults`.** `payload` was read before its
   `const` declaration, throwing a `ReferenceError` that silently aborted the
   render on the approve path — the banner appeared but the artifact path and the
   artifacts panel never populated. Declaration hoisted.
3. **Model name lost after completion.** `review_payload` is `null` once a run
   finishes, so the banner showed a placeholder. `selected_model_name` is now
   exposed in `values`.

**Known issues left open (see Steps 3 and 6):**

- Frequency-encoded categoricals are skipped by the Fairness Agent with the
  *inaccurate* reason "is a continuous numeric feature". `occupation` (14
  categories, above the one-hot limit of 10) becomes a float and is silently
  dropped from the fairness audit. The verdict stays honest — it reports skipped,
  not passed — but the reason is wrong and a real sensitive attribute goes
  unaudited. **Fixed in Step 3a** — groups now come from raw values.
- A completed run's evaluation tabs render empty, because `review_payload` is
  `null` once the graph ends. The banners are correct; the detail panes are not.
- The regression KPI header reads "ACCURACY N/A" instead of showing RMSE.

The original checklist follows, for re-running after future changes.

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

## Step 1 — Compliance artifact generation — DONE

Implemented in `compliance_artifacts.py`. On approval a run now writes
`artifacts/<run_id>/` containing `model_card.md`, `model_card.json`, `aibom.json`
and `technical_documentation.md`; their SHA-256 digests are logged as a chained
`compliance_artifacts_generated` audit event, `verify_artifacts()` re-checks them,
and the dashboard shows an integrity badge with an inline viewer. Supporting
changes: `dataset_sha256` hashed from the raw upload, `planner_meta` (model id,
prompt hashes, token usage) captured via a `meta_out` out-parameter on
`plan_pipeline`, and `pipeline_graph.AUDIT_DB_PATH` so the graph and the artifact
generator agree on one audit database.

Still open from this step, deliberately deferred:

- The AIBOM `approver` and `policy_version` fields read `not recorded` until
  Steps 5 and 4 land.
- Artifacts are regenerated per run; there is no cross-run model registry.

The original specification follows, for reference.

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

## EDA-driven pipeline — DONE

Exploratory analysis used to be a dashboard page that nothing downstream read.
It is now `data_analysis_node`, the first node in the graph, and it produces
structured findings (`eda_insights.py`). Each finding carries a **route** that
decides who may act on it:

| Route | Findings | Who acts |
|---|---|---|
| `data_agent` | per-row identifiers / free text, constant columns | Dropped deterministically before the split |
| `planner` | zero-inflated or outlier-heavy features, redundant pairs, skewed target | Sent to the LLM, which must address each by name |
| `reviewer` | suspected target leakage, proxy variables for protected attributes | Shown at the gate only — **never sent to the LLM** |

`build_eda_linkage()` annotates every finding with what each stage actually did
(applied / sent / mentioned / not reflected / flagged / attached). This shows on
the EDA page, in a new **EDA Insights** tab at the gate, as proxy warnings under
the fairness verdict (informational — the verdict is unchanged), and in the model
card.

**On UCI Adult (32,561 rows, 0.74 s):** 9 findings. `relationship` is a high-severity
proxy for `sex` (Cramér's V 0.65); `marital-status` for `age` (0.57) and `sex`
(0.46); `occupation` for `sex` (0.44); `capital-gain` and `capital-loss` are
zero-inflated; `education` and `education-num` are redundant (1.0);
`hours-per-week` is peaked. There were no identifier, constant or leakage false
positives. Live against `openai/gpt-oss-20b`, the planner named all four
planner-routed findings, proposed binary indicators for the zero-inflated columns,
left `hours-per-week` alone, and did not touch any proxy column.

**Found and fixed along the way.** The Data Agent's substring matcher misread two
of those live planner steps. It dropped *both* `education` and `education-num`
from "Drop 'education-num' and keep 'education'", and winsorized `hours-per-week`
from "Keep 'hours-per-week' as is; consider winsorizing ... if ...". It also
misread the example phrasing the new prompt rule itself suggests. Rewritten
(clauses, hedges, scope terminators, longest-first names); the exact strings are
pinned as regression tests.

Re-verified live after a clean server restart: the planner again wrote
"Drop 'education-num' (redundant with 'education')", and this time only
`education-num` was dropped. An earlier live run had silently executed pre-fix
code, because `uvicorn --reload` logged a reload but never replaced its worker — see
the gotcha in `CLAUDE.md`. Passing tests did not reveal this; checking the serving
process's start time did.

**Still open:**

- The planner recommends transforms the deterministic Data Agent cannot execute
  yet — binary "is non-zero" indicators and `log1p`. These findings show as
  *sent / mentioned* but never *applied*. Implementing a small, whitelisted set of
  transforms (fit on train where they learn anything) is the natural extension.
- Proxy detection keys on column **names**. A protected attribute stored under an
  opaque name is missed, and pairs of protected attributes are not reported as
  proxies of each other.
- Integer identifiers are caught only when named like an id or stored as monotonic
  row numbers.
- The hedge rule is deliberately conservative: "Drop X if present" is treated as
  advice and not executed.

---

## Step 2 — Quantitative governance evaluation — DONE

`experiments/run_governance_eval.py` runs **4 datasets × 3 arms × 5 split seeds = 60 runs**
through the real graph with a scripted reviewer. The results are committed in
`experiments/results/`, and the README has the headline table.

**How it works.**

- The planner has a **record/replay cache**. Record mode calls the LLM on a miss and stores
  the response. Replay never calls it and raises `PlannerCacheMiss` on a miss, so a replayed
  result can never silently contain a fresh plan.
- The split seed is a real `PipelineState["split_seed"]` parameter, recorded in the quality
  report.
- Datasets are round-tripped through CSV, exactly like an upload, and each run records the
  SHA-256 of the bytes.
- `--summarise-only` rebuilds the tables and figure from the saved CSVs in seconds.

**Verified reproducible.** A full replay with an invalid `GROQ_API_KEY` served all 60
planner calls from cache, with 0 errors, and matched the recording exactly: 25 columns × 60
runs and 9 columns × 200 gate rows, excluding only wall clock and cache-hit counts.

**Findings.**

1. **No arm approved a fairness-compliant model** — all 60 approved models violated.
2. **Loop 3 is not a fairness intervention.** Across 20 rerouted runs, violations were fewer
   in 0, the same in 18 and more in 2, at an AUC cost on every dataset (Adult −0.049, COMPAS
   −0.051, Bank Marketing −0.032, German Credit −0.019). COMPAS magnitudes improved (min DI
   0.19 → 0.46, max parity difference 0.64 → 0.32) but still violated; Adult's min DI fell
   0.043 → 0.030.
3. **Loop 1 never engaged.** Arms A and B were identical on every run, because benchmark
   data passes the quality gate first time.

**Caveats that change how to read those numbers.** These describe the *first* run. The
Fairness Agent was then corrected and the evaluation re-run — see Step 3a, which supersedes
the numbers above and corrects the first caveat.

- **Adult's minimum DI is a tiny-group artifact.** The minimum group is
  `marital-status = "Married-AF-spouse"`, 4 people in the test split, all predicted
  negative. The real disparity is `sex` (DI ≈ 0.31, groups of 6,490 and 3,279).
- **The audited attributes are the planner's choice**, and several are not protected:
  German Credit audited only `job`; Bank Marketing audited `education` and `marital`.
- **`age` was never audited on any dataset**, because it is continuous. Adult `occupation`
  was skipped with the inaccurate reason "continuous numeric feature": it was
  frequency-encoded.

**Defects found by running it, fixed before the final results.**

- XGBoost rejected one-hot column names containing `[`, `]` or `<` and silently failed on
  every German Credit fit. The names are now sanitised.
- A run whose remaining candidates all failed to train reached the gate, was approved, and
  got a model card. `route_after_training` now ends such a run.
- Found by inspecting the chart: the legend overprinted the subtitle, coincident arm means
  overprinted each other, and labels overprinted points. Mean labels now sit in a
  leader-line column, and an absent arm is explained in its panel.

**Provenance.** Recorded on top of `91544dc` with the evaluation code uncommitted. Its
manifest predates the `git_dirty` flag that now records this.

The original specification follows, for reference.

### Original specification

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

## Step 3a — Trustworthy fairness metrics — DONE

Step 2 showed the Fairness Agent was not fit to report. It was corrected in
`fairness_agent.py` (18 new tests in `tests/test_fairness_groups.py`) and the whole
evaluation re-run from the planner cache.

**What changed.**

- **Groups come from raw uploaded values** (`raw_frame`, aligned on the shared index
  labels), not the cleaned frame. Missing values form a `(missing)` group. The old
  one-hot reconstruction remains as a fallback when no raw frame is supplied.
- **`MIN_GROUP_SIZE = 30`**: smaller groups are excluded from DI and parity difference and
  listed with their sizes. Fewer than two groups left means the attribute is skipped.
- **`age` is banded** <25 / 25–59 / 60+. Other continuous attributes are skipped with the
  accurate reason "only age has a banding rule".
- **Protected attributes are audited even when the planner omits them** (`source: auto`),
  detected by name through `eda_insights.is_protected_attribute`.
- **Equal-opportunity and equalized-odds differences** are reported per attribute. They are
  not part of the verdict.
- The dashboard fairness table shows the new columns, the group sizes compared, excluded
  groups and a "Not audited" list with reasons; the model card and Annex IV draft say the
  same.

**Re-run, verified.** 60 runs, 0 errors, 60/60 planner calls from cache. Arms A and B
chose the same model with the same AUC on all 40 runs, so every change is in the
measurement.

| Dataset | Violated attributes (A) | Min DI (A) | Newly audited |
|---|---|---|---|
| Adult | 3.0 → 5.0 | 0.043 → 0.004 | `age`, `occupation` |
| German Credit | 1.0 → 1.0 | 0.771 → 0.798 | `age` (3 of 5 splits) |
| Bank Marketing | 1.6 → 2.6 | 0.376 → 0.128 | `age` |
| COMPAS | 2.0 → 4.0 | 0.194 → 0.179 | `age`, `sex` |

- **The first run understated disparity.** Bank Marketing's minimum is now `age` on every
  split; COMPAS `sex` (DI ≈ 0.30) had simply never been audited.
- **Correction to the Step 2 write-up.** Step 2 expected a minimum group size to make
  Adult's headline DI meaningful. It removed the 4-person artifact, but the minimum stayed
  near zero for genuine reasons: `occupation = "Priv-house-serv"` (about 50 test rows, no
  positive predictions) on 4 of 5 splits, and `age` under 25 (about 1,700 rows, positive
  rate under 1%) on the fifth.
- **Loop 3 is still not a fairness intervention**: 19 of 20 runs rerouted; fewer violated
  attributes in 2, the same in 14, more in 3.
- **COMPAS `sex` coding is undocumented** on OpenML. Value 1 is 80.5% of rows, consistent
  with the known male share, but that is an inference, so groups are reported as "0"/"1".

**Found by the re-run, open (#26):** 3 of 60 approvals "passed", all German Credit seed 19,
whose audit covered only `job` — age bands were too small to compare there. A skipped
protected attribute does not stop a pass.

---

## Step 3b — Verdict coverage and mitigation

Do these in this order:

1. **Verdict coverage (#26) — DONE.** No violation plus an unaudited protected attribute
   now gives `None`, shown amber as NOT FULLY EVALUATED with the attribute and reason
   named. `None` rather than a fourth state keeps invariant 4's three states; the new
   `fairness_coverage` ("complete" / "partial" / "none") and
   `protected_attributes_unaudited` fields distinguish it from nothing evaluated. A
   measured violation is still `False`. A protected name the planner invents, absent from
   the data, is not a gap. German Credit seed 19's shape is pinned in
   `tests/test_fairness_groups.py`. In the arm-D re-run, seed 19 is NOT FULLY EVALUATED
   under all four arms; nothing else changed for arms A–C.
2. **Protected-only verdict — DONE** (user decision). The verdict covers protected
   attributes only; planner-proposed attributes that are not protected are audited and
   shown as ADVISORY, and never mitigated. With no protected attribute evaluated the
   verdict is NOT EVALUATED.
2b. **Validation split for gate decisions — DONE** (user decision). Train / validation /
   test = 64 / 16 / 20. The leaderboard, fairness audit and every reviewer decision use the
   validation rows; the approved model is scored once on the untouched test rows
   (`final_test_evaluation` audit event, model card section 4, dashboard banner), and the
   evaluation reports those numbers.
3. **Real mitigation (arm D) — DONE.** New gate decision `reject_and_mitigate` →
   `mitigation_node` → retraining with **reweighing** (`mitigation.py`), chosen over Fairlearn
   by the user: no new dependency, and the approved model stays an ordinary estimator, so
   SHAP, the joblib artifact and the model card are unchanged and no protected attribute is
   needed at prediction time. Weights are fitted on train rows with the audit's grouping; the
   attribute is the worst-DI protected violation (else worst-DI violation); a second
   mitigation reweights the intersection; it shares the rejection cap. The gate shows before
   against now, and the model card records it.

   **Evaluated as arm D** (80 runs, 0 errors, 80/80 from cache; arms A–C unchanged):
   - Fewer violated attributes in 6 of 19 rerouted runs (arm C: 2), more in 2 (arm C: 3).
   - AUC cost −0.004 Adult, −0.013 COMPAS, −0.020 Bank Marketing, none on German Credit
     (arm C: −0.018 to −0.051).
   - Reweighed attributes moved a lot (COMPAS `sex` DI 0.25–0.38 → 0.68–0.99; Bank Marketing
     max parity difference 0.20 → 0.07), but every Adult, Bank Marketing and COMPAS run still
     violated. Two reweighings cannot cover four or five violated attributes.
   - The only 2 genuine passes in all 80 approvals: German Credit arm D, seeds 7 and 128.
   - Adult's minimum DI did not move: both reweighings went to protected attributes (`age`,
     then `race`/`sex`), never to the unprotected `occupation` group that sets the minimum.
   - Reweighing one attribute can push disparity onto another (Bank Marketing, 2 of 5 runs).

   **Open from this item:** gate decisions use the held-out rows, so approved-model metrics
   are not an untouched estimate (true of any reviewer); a validation split for gate
   decisions would fix it. Mitigating an unprotected attribute (German Credit seed 42, `job`)
   is allowed and did not help.
4. **Reviewer-declared protected attributes — DONE.** A dashboard field
   (`protected_attributes`, validated against the CSV headers) adds columns that count
   toward the verdict and feed EDA proxy detection.
5. **Intersectional subgroups — DONE, reported only.** Pairs of evaluated protected
   attributes (`sex × race`) with the same group rules, in `intersectional_report`, a
   dashboard table and the model card. They never change the verdict; whether they should
   is a Step 4 policy question.

**Re-run after items 2, 2b and 4 (validation gate, test reporting, protected-only verdict).**
80 runs, 0 errors, 80/80 from cache, every approved run reported on its test rows.

- No approved model passed: 72 violated, 8 not evaluated (German Credit seeds 19 and 42).
- Arm C: 15 rerouted, same violated protected attributes in all 15; test AUC −0.022 to −0.058.
- Arm D: 15 rerouted, fewer in 4 (COMPAS), none worse; test AUC −0.003 to −0.015. COMPAS min
  DI 0.18 → 0.42, max parity difference 0.62 → 0.23.
- **German Credit's gate was blind**: no protected attribute auditable on 160 validation rows
  on any seed, so C and D approved without rejecting; the test rows then showed age
  violations on 3 of 5 seeds.
- Gate AUC overstates test AUC (German Credit 0.805 vs 0.780); for arms A and B, ranking on
  validation rows changed the selected model in 16 of 40 runs.

**Open from this re-run:**

- **Approving a NOT EVALUATED model** should be a policy decision, not a default: a Step 4 flag
  (`allow_approval_when_not_evaluated`) and/or the Step 5 second-approver rule.
- **Small datasets and the validation split.** Consider a policy-set validation size, or
  cross-validated fairness on train rows at the gate, so small groups can be audited.

The original plan follows. Its metric items are done in 3a. Its mitigation was implemented
with reweighing rather than Fairlearn (item 3 above).

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

## Step 4 — Policy-as-code — DONE

`policy.yaml` + `policy.py`. What landed:

- **Validated, versioned, hashed.** Unknown keys, wrong types, out-of-range values,
  unknown model names and a warning threshold above the drop threshold all raise
  `PolicyError`; the server refuses to start on an invalid file. The SHA-256 covers the
  validated values as canonical JSON, so comments and formatting do not change it.
- **What it controls:** Data Agent null/quality limits, one-hot limit, test and
  validation sizes; fairness DI and parity thresholds and minimum group size; retry and
  rejection caps; an allowed-models list; and
  `block_approval_when_fairness_not_evaluated`.
- **Recorded per run:** the policy is in state; a `policy_applied` audit event (the full
  policy) opens the trail; `final_outcome`, the AIBOM (`policy_version`,
  `policy_sha256` — no longer `not recorded`) and the model card name it.
- **Approval block (user decision: on by default).** A classification model whose
  fairness verdict is NOT EVALUATED or NOT FULLY EVALUATED cannot be approved: the gate
  payload withholds `approve` and gives the reason, the dashboard disables it, the API
  returns 409, and the graph ends a stray approve at `mark_approval_blocked`
  (`APPROVAL_BLOCKED_BY_POLICY`). Regression is exempt. This closes #30 as a policy:
  German Credit's blind gate can no longer end in an approval.
- **Evaluation:** every run carries the committed policy with its arm's retry cap; the
  manifest records the base policy's version and hash; blocked runs get the status
  `terminated_approval_blocked`.

**Re-run under the policy** (80 runs, 0 errors, 80/80 from cache, from `b2fd32b`): all 20
German Credit runs now end `terminated_approval_blocked` — the gate verdict was NOT
EVALUATED on every seed, not just the 8 whose test rows were unmeasurable, because the block
acts on the gate's validation-row verdict. The other 60 runs are identical to `d81faa2`
(one AUC differs in the fourth decimal); all 60 approved models violate. Found along the
way: the harness's `git_dirty` flag counted its own result files, so it was true for every
run writing into `experiments/results/` (#31, fixed).

**Still open:** selecting a policy per run (e.g. `strict` / `default`) from the
dashboard; whether error-rate gaps and intersectional gaps should count toward the verdict
(both are still hard-coded as reported-only, deliberately, until someone decides);
a policy key for the verdict scope.

The original specification follows.

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

- [x] `training_agent.py`: remove `n_jobs=-1` from `LogisticRegression`. scikit-learn
      1.8+ warns that it has no effect.
- [ ] `training_agent.py`: `"svm"` maps to `(None, None)`, so the planner can
      recommend a model that is always skipped. Remove SVM from the planner prompt
      lists in `planner_agent._build_prompts`, or implement it.
- [ ] Target is label-encoded twice (Data Agent, then Training Agent). Harmless;
      tidy up if touching that code anyway.
- [x] Fairness Agent: distinguish "frequency-encoded categorical" from "continuous
      numeric" when skipping an attribute. Done in Step 3a — groups now come from raw
      values, so such columns are audited rather than skipped.
- [x] UI: a completed run's evaluation tabs are empty. The gate node now keeps
      `last_review_payload`, returned by the status API once a run completes.
- [x] UI: the KPI header showed "ACCURACY N/A" on regression runs. Cause: it read the
      upload form's task toggle, not the run's metrics. Fixed.
- [x] `server.py`: stdout/stderr reconfigured to UTF-8 with `errors="replace"`.
- [x] Add a GitHub Actions workflow that runs `pytest` on push
      (`.github/workflows/tests.yml`, Python 3.13, no secrets).
- [x] Found along the way: planner lists, Data Agent actions and the leaderboard error
      were inserted into the dashboard unescaped. Escaped.
- [ ] Retire `app.py` (the Streamlit UI) — delete it or move it to `legacy/` — and
      drop `streamlit` from requirements. **Deferred: the partner's code; agree first.**
- [ ] Move the legacy root `test_*.py` scripts into `scripts/manual/` and update the
      README. **Deferred:** they import root modules and open `.env` by relative path,
      so moving them means editing all eight of the partner's scripts; agree first.
- [ ] SVM: the planner can still recommend it, and it is skipped with a reason. Removing
      it from the planner prompt would change every prompt hash and invalidate the
      committed planner cache, so it waits for the next cache re-record.
- [ ] Longer term: audit-chain anchoring. Periodically publish the head hash
      outside the database (a signed log, another host, a timestamping service).
      This closes the truncation gap documented in the README.

---

## Collaboration note

These changes live on the fork (`chenduran-nag/AegisML`), not the partner's repo
(`ruhannpn/AegisML`). When they're ready to share, open a PR from the fork into
upstream — and only after the user asks.
