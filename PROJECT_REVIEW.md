# AegisML — Project Review, Landscape Scan & Roadmap

> Working document. Part 1 inventories what exists today. Part 2 compares AegisML
> against the current research and product landscape. Part 3 is a prioritised list
> of what to build next, with the reasoning for each item.
>
> Reviewed at commit `01c8c7d` (`docs: Add interactive GitHub Mermaid DAG flowchart`).
>
> **Update — branch `fix/tier0-repairs-and-audit-chain`:** all of Tier 0, item 1.1
> (hash-chained audit log) and item 1.2 (compliance artifact generation) are
> implemented, plus two Medium-severity items (#5, #8) that fell out of the same
> refactor. See [Implementation status](#implementation-status). 59 offline tests
> pass. Live verification against the real Groq planner (Step 0 in `NEXT_STEPS.md`)
> is still outstanding and needs an API key.

---

## Implementation status

| Item | Status |
|---|---|
| 0.1 `requirements.txt` completed (10 missing packages added) | Done |
| 0.2 Model actually serialised to disk on approval | Done |
| 0.3 Leakage: split drawn before any parameter is fitted | Done |
| 0.4 Duplicate `build_graph()` removed | Done |
| 0.5 `graph.invoke` moved off the event loop | Done |
| 0.6 README aligned with the code | Done |
| 0.7 Regression fairness reported as NOT EVALUATED | Done |
| 0.8 Offline pytest suite (`tests/`, 35 tests) | Done |
| 1.1 Hash-chained tamper-evident audit log + UI badge | Done |
| #5 Fairness measured on held-out rows | Done (fell out of 0.3) |
| #8 EDA report moved into checkpointed state | Done (fell out of 0.5) |
| #16 **New:** categorical encoding silently no-op on pandas 3 | Done — see below |
| 1.2 Compliance artifact generation | **Done** — model card, AIBOM, Annex IV draft, digests chained |
| 1.3 Policy-as-code | Not started |
| 1.4 Reviewer identity | Not started |
| Tier 2 / Tier 3 | Not started |

### Defect found during implementation

**#16 — Categorical encoding silently did nothing on pandas 3.**
`_encode_categoricals` selected columns with `pd.api.types.is_object_dtype`. Since
pandas 2.x the string backend may return a dedicated `str`/`StringDtype` rather
than `object`, and pandas 3 makes that the default. Under those versions the
selector matched **nothing**: `sex`, `race` and `occupation` passed through
unencoded, and the Training Agent then failed on raw strings with
`no model trained successfully`. The whole pipeline was unrunnable on a current
pandas — it was not caught earlier because the manual test scripts were last run
on pandas 2.x with the legacy string backend. Fixed with an explicit
`_is_encodable_categorical()` predicate covering object, string and categorical
dtypes.

---

## Part 1 — What Is Built Today

### 1.1 One-line summary

A **governed AutoML pipeline**: upload a CSV, and a chain of agents profiles the data,
plans preprocessing with an LLM, cleans it deterministically, trains a model
leaderboard, audits it for demographic fairness, then **pauses for human sign-off**
before the run is considered complete. Every step is written to a SQLite audit trail.

The ML itself is intentionally ordinary. The contribution is the **orchestration and
governance layer** around it.

### 1.2 Component inventory

| File | LOC | LLM? | Status | What it does |
|---|---:|:---:|---|---|
| `pipeline_graph.py` | 641 | no | Complete | LangGraph `StateGraph`, all 6 nodes, 3 loops, `SqliteSaver` checkpointing |
| `graph_state.py` | 147 | no | Complete | `PipelineState` TypedDict + pickle/joblib serialisation helpers |
| `data_analysis_agent.py` | 145 | no | Complete | EDA profiling + Chart.js payload generation |
| `planner_agent.py` | 464 | **yes** | Complete | Groq call, prompt construction, JSON validation, retry-context injection |
| `data_agent.py` | 428 | no | Complete | Deterministic cleaning, encoding, scaling, quality gate |
| `training_agent.py` | 459 | no | Complete | Model registry, leaderboard, SHAP |
| `fairness_agent.py` | 287 | no | Complete | Disparate Impact + Demographic Parity Difference |
| `audit_log.py` | 130 | no | Complete | SQLite event log, separate DB from the checkpointer |
| `server.py` | 233 | no | Complete | FastAPI: start / resume / status / audit / EDA endpoints |
| `static/index.html` | 1173 | no | Complete | Single-page dark dashboard, 4 views, Chart.js |
| `app.py` | 468 | no | **Superseded** | Original Streamlit UI, still functional, now redundant |
| `dataset_utils.py` | 51 | no | Complete | UCI Adult loader used by the test scripts |
| `build_presentation.py` | 352 | no | Complete | Generates the Review 2 `.pptx` |
| `test_*.py` (8 files) | 1771 | mixed | Scripts, not CI | Integration walkthroughs; hit live Groq + network |

### 1.3 The graph

Six nodes plus five state-mutation helper nodes, compiled with a `SqliteSaver`
against `pipeline_state.db`.

```
START -> planner -> data_agent -> [quality?] -> training -> fairness -> GATE -> [decision?] -> audit_log -> END
              ^          |                          ^                            |
              |          |                          |                            |
              |          +-- LOOP 1: auto retry ----+----------------------------+
              |              (max 2, back to planner)                            |
              |                                                                  |
              +-- LOOP 2: human rejects data quality (max 2, back to planner) ----+
                                                                                 |
                  LOOP 3: human rejects model/fairness (back to training) --------+
```

**Node responsibilities**

1. **Data Analysis** — null counts, cardinality, min/max/mean/std, IQR outliers,
   Pearson correlations with `|r| >= 0.20`, target distribution (histogram for
   numeric with >15 uniques, else value counts), Chart.js payloads.
2. **Planner** — the only LLM call in the system. Builds a compact summary
   (never raw rows), calls Groq in `json_object` mode at `temperature=0.2`,
   validates 5 required keys, retries once on parse failure.
3. **Data Agent** — drop columns >50% null, drop null-target rows, median/mode
   impute, label-encode target, one-hot (<10 uniques) or frequency encode,
   `StandardScaler` on numeric non-bool features.
4. **Training** — keyword-matched model registry, 80/20 stratified split at
   `random_state=42`, leaderboard sorted by AUC (classification) or RMSE
   (regression), top-5 SHAP via Tree/Linear explainer.
5. **Fairness** — reconstructs one-hot-encoded attributes back into categorical
   series by prefix matching, computes max-vs-min positive-prediction rates,
   flags violation if DI < 0.80 or DPD > 0.10.
6. **Governance Gate** — calls `interrupt(payload)`; the graph suspends and the
   checkpoint persists to disk until a decision arrives.

### 1.4 Design decisions worth keeping

These are the things that are genuinely well done and should survive any refactor:

- **The LLM never touches data.** It produces a *plan*; all mutation is
  deterministic Python. Free-text plan steps are consumed by keyword matching only
  (`_parse_plan_steps`) — never executed. This is the correct trust boundary for a
  governance tool and is the project's strongest architectural claim.
- **Aggregates-only prompting.** No raw rows leave the machine. Privacy-preserving
  by construction, and it also keeps the prompt inside token limits.
- **Counters live in dedicated nodes.** `increment_retry`,
  `increment_human_reroute_planner/training` are separate graph nodes rather than
  side effects inside agent nodes. Correct, because LangGraph re-executes a node on
  resume — inline increments would double-count.
- **Deliberately thin gate node.** `human_approval_node` does no heavy work, so
  re-execution on resume is instant and safe. The docstring says so explicitly.
- **Two separate databases.** `pipeline_state.db` (checkpointer, framework-owned)
  vs `audit_log.db` (evidence, application-owned). Keeping evidence independent of
  the orchestration framework is right.
- **Serialisation rationale is documented.** `graph_state.py` explains *why*
  DataFrames are pickled and models are joblib'd, and flags the parquet migration
  path. Best-documented file in the repo.
- **Loop 3 skips ahead.** Rejecting a model reroutes straight to training rather
  than replanning — cheap, and preserves the already-validated cleaned data.

### 1.5 Known defects and gaps

Ordered by how much they matter.

All of the High-severity rows and several others are now **fixed** on branch
`fix/tier0-repairs-and-audit-chain`; the Status column records what changed.

| # | Issue | Where | Severity | Status |
|---|---|---|---|---|
| 1 | **Model is never saved to disk.** Nothing writes `model_saved_path`; the API fabricates the string and the UI displays it as a real artifact path. README section 7 claims otherwise. | `server.py:79` | **High** | **Fixed** — written in `audit_log_node`, real path in state |
| 2 | **Train/test leakage.** `StandardScaler.fit_transform` and frequency encoding run on the full dataset before the split. Test metrics are optimistically biased. | `data_agent.py` -> `training_agent.py` | **High** | **Fixed** — split precedes all fitting |
| 3 | **`requirements.txt` is missing ~10 packages** — langgraph, langgraph-checkpoint-sqlite, langchain-core, fastapi, uvicorn, python-multipart, xgboost, shap, joblib, streamlit. README quickstart cannot work on a clean machine. | `requirements.txt` | **High** | **Fixed** |
| 4 | **The audit log is not actually immutable.** It is a plain SQLite table; any `UPDATE`/`DELETE` is untraceable. The word "immutable" appears three times in the README. | `audit_log.py` | **High** (it is the core claim) | **Fixed** — SHA-256 chain + `verify_audit_chain()` |
| 5 | Fairness is evaluated on the full cleaned dataset, including training rows. | `fairness_agent.py` | Medium | **Fixed** — evaluates on `eval_index` |
| 6 | `graph = build_graph()` appears **twice** at the bottom of the file — compiles twice, opens two SQLite connections, leaks the first. | `pipeline_graph.py` (end) | Medium | **Fixed** |
| 7 | `graph.invoke()` runs synchronously inside `async def` handlers — blocks the event loop for the entire training run. | `server.py` | Medium | **Fixed** — `run_in_threadpool` |
| 8 | `_EDA_CACHE` is an in-memory dict. Everything else survives restart; this doesn't. Also unbounded. | `server.py` | Medium | **Fixed** — moved into graph state |
| 9 | Regression runs return `overall_fairness_passed: True` unconditionally, and the dashboard renders a green pass. Misleading for a compliance tool. | `fairness_agent.py` | Medium | **Fixed** — returns `None`, UI shows NOT EVALUATED |
| 10 | README/code drift: badge says `llama-3.3-70b-versatile`, code default is `openai/gpt-oss-20b` (and the docstring lists llama-3.3 as deprecated). README's "<1,800 token guardrail" is actually a 40-column cap. | `README.md` | Low | **Fixed** |
| 11 | `CORSMiddleware` with `allow_origins=["*"]` **and** `allow_credentials=True` — invalid combination, browsers reject it. | `server.py` | Low | **Fixed** — `allow_credentials=False` |
| 12 | Tests are print-based scripts requiring live Groq API + UCI download. Cannot run in CI or offline. | `test_*.py` | Low | **Fixed** — `tests/` suite, 35 offline tests |
| 13 | Target is label-encoded twice (Data Agent, then again in Training Agent). Harmless, but redundant. | `data_agent.py`, `training_agent.py` | Low | Open (harmless) |
| 14 | `SVM` is in the registry as `(None, None)`, so the planner can recommend a model that is always skipped. | `training_agent.py` | Low | Open |
| 15 | No reviewer identity. Any POST to `/api/pipeline/resume` with a valid `thread_id` can approve a model. | `server.py` | Low (today) | Open — see 1.4 |

---

## Part 2 — Landscape Scan

### 2.1 Where the research field actually is

Autonomous data-science agents are a crowded and fast-moving area, and the
published work is **overwhelmingly optimisation-focused** — the metric is Kaggle
medal rate, not accountability.

- **MLE-bench** (OpenAI) is the reference benchmark: 75 Kaggle competitions with
  local grading. The leaderboard moved from **AIDE at ~25.8%** medal rate to
  **MLE-STAR at ~43.9%** with Gemini-2.0-Flash. Newer harnesses — **DSGym**,
  **DSAEval**, **TML-bench** — extend evaluation to broader real-world and
  tabular-specific tasks.
- **LightAutoDS-Tab** (arXiv 2507.13413) is the closest structural analogue to
  AegisML: a multi-agent AutoML system specifically for **tabular** data, with
  separate AutoML and data-processing agents plus a coordination layer.
  **Critically, it has no fairness auditing, no human-in-the-loop approval, and no
  audit logging.** The same is true of AIDE, DS-Agent, AutoMind and ChainBuddy.
- On the governance side, **LanG** (arXiv 2604.05440) is a governance-aware agentic
  platform on LangGraph with human-in-the-loop checkpoints and a policy engine —
  but it targets **security operations**, not the ML lifecycle.

**The gap AegisML sits in is real.** The AutoML agent papers optimise; the
governance platforms govern; almost nobody does governed AutoML for tabular data as
one artifact. That is a defensible framing for the report — but it means the
differentiation has to come from **governance depth**, because on raw AutoML
capability the published agents are far ahead and always will be.

### 2.2 Where the product market is

Commercial AI governance in 2026 (Credo AI, Holistic AI, Arthur, Fiddler) has
converged on a lifecycle model much wider than AegisML's:

- Governance is **continuous, not a single pre-deployment gate**. Arthur's platform
  is built around agent discovery, runtime guardrails and continuous evaluation.
  Holistic AI shipped "Guardian Agents" in 2026 that move from passive monitoring to
  **real-time intervention**.
- Gartner projects **>40% of agentic AI projects will be cancelled by end of 2027**,
  citing inadequate risk controls among the causes — which is, conveniently, exactly
  the thesis AegisML argues.

### 2.3 Regulatory trends that map to concrete features

This is where the most useful roadmap items come from, because each trend maps
directly onto something buildable.

| Trend | What it means | What AegisML could build |
|---|---|---|
| **EU AI Act Article 11** — technical documentation must exist *before* market placement; high-risk obligations phase in from Aug 2026, full application Dec 2027 | A structured, reviewable document per system | Auto-generate an Article 11 documentation pack from `audit_log.db` |
| **Model cards moved from best practice to legal requirement** (2024–2026) | Architecture, intended use, metrics, risks, limitations, training-data characteristics | Auto-generate a model card at approval time — you already compute every field |
| **Data lineage entered audit scope** | Full lifecycle: sources, transformations, access, usage | The `actions_taken` lists are already a transformation log — formalise them |
| **AIBOM (AI Bill of Materials)** — SBOM extended to training-data provenance, model versions, tool integrations | Inventory + dependency + provenance manifest | Emit a signed AIBOM JSON per approved run |
| **NIST AI RMF crosswalks** to EU AI Act / ISO 42001 / OWASP — one evidence set, many regimes | Map controls once, satisfy several frameworks | Tag each audit event with the AI RMF function it evidences (GOVERN/MAP/MEASURE/MANAGE) |
| **OpenTelemetry GenAI semantic conventions** — adopted by Datadog, Arize, LangSmith (still "Development" status as of May 2026) | Standard `gen_ai.*` span attributes for LLM and agent calls | Instrument the six nodes as OTel spans |
| **Continuous monitoring materially improves outcomes** — incidents caught by internal monitoring show 95.8% AI RMF alignment vs 58.1% without | Post-deployment drift and decay detection | Reopen the governance gate on drift |

### 2.4 Fairness tooling AegisML is not yet using

The current agent implements two metrics by hand. The ecosystem is much richer:

- **Fairlearn** and **AIF360** are the mainstream toolkits; AIF360 is sklearn-coupled
  with no PyTorch/GPU path.
- **fairlib** supports **14 debiasing methods** against Fairlearn's four.
- **Aequitas**, **OxonFair**, **FAT Forensics**, **Themis-ml**, Google's
  **What-If Tool** and LinkedIn's **LiFT** occupy adjacent niches.

The important structural point: **AegisML currently only detects bias — it cannot
fix it.** Every one of those libraries offers mitigation. The reroute loop asks the
LLM to write different preprocessing text and hopes the fairness number moves, which
is an indirect and unreliable lever.

### 2.5 Honest assessment of the differentiation

**Genuinely distinctive right now:**

1. The LLM-plans / deterministic-executes trust boundary, enforced architecturally.
2. Three *typed* feedback loops with independent caps and different re-entry points —
   most HITL demos have one generic "retry".
3. Governance evidence stored independently of the orchestration framework.
4. Aggregates-only prompting.

**Not distinctive**, despite the README's framing:

- Multi-agent LangGraph orchestration — this is the standard pattern now.
- SHAP explanations — table stakes since roughly 2019.
- Human-in-the-loop interrupt/resume — LangGraph's flagship documented feature.
- "Enterprise-grade" / "immutable" / "EU AI Act compliant" — currently claims, not
  demonstrated capabilities. Each is a feature waiting to be built, and a reviewer
  who checks will notice. The single highest-leverage move available is to **make
  these claims true**, because the code is already most of the way there.

---

## Part 3 — What Could Be Done

Grouped by tier. Tier 0 is required before anything else is worth showing.

### Tier 0 — Repairs (a few hours, non-negotiable)

| # | Task | Why |
|---|---|---|
| 0.1 | Fix `requirements.txt` | Nobody can reproduce the project without it |
| 0.2 | Actually save the model in `audit_log_node` and write `model_saved_path` into state | The UI currently displays a path to a file that does not exist |
| 0.3 | Move scaling + frequency encoding to after the train/test split | Leakage in a project about trustworthy ML is the worst possible bug to leave in |
| 0.4 | Delete the duplicate `graph = build_graph()` | Double compile, leaked connection |
| 0.5 | Make handlers non-async, or wrap `graph.invoke` in `run_in_threadpool` | Server is unresponsive during training |
| 0.6 | Align README with the code (model name, token-guardrail claim) | Cheap credibility |
| 0.7 | Surface "fairness not evaluated" honestly for regression | A green pass on an unevaluated property is misleading |
| 0.8 | Convert `test_*.py` to pytest with a small committed fixture CSV and a mocked planner | Tests needing a live API key and a network download are tests nobody runs |

### Tier 1 — Make the existing claims true (highest value per hour)

**1.1 Hash-chained tamper-evident audit log** — *strongly recommended*

Add `prev_hash` and `entry_hash` columns; each row hashes its own content plus the
previous row's hash. Add `verify_audit_chain(run_id)` returning the first broken
link. Surface it in the UI: a green "chain verified, 14 entries" badge, plus a demo
where you manually `UPDATE` a row in `sqlite3` and the badge turns red.

Roughly 60 lines of code. It converts "immutable" from an unsupported adjective into
a demonstrable property, and it demos in fifteen seconds. Highest
impact-to-effort ratio in this entire document.

**1.2 Compliance artifact generation** — *the flagship differentiator*

At approval, generate from the audit trail:

- a **Model Card** (intended use, metrics, fairness results, limitations,
  training-data characteristics — every field is already computed);
- an **EU AI Act Article 11 technical documentation pack**;
- an **AIBOM** JSON (dataset hash, row/column counts, library versions, model
  version, LLM model ID and prompt version, approver, timestamp).

Nothing in the open-source agentic-AutoML space does this. It is the concrete answer
to "so what does the audit log actually give me?", and it is mostly templating over
data structures that already exist.

**1.3 Policy-as-code**

Move the hardcoded constants (`DISPARATE_IMPACT_THRESHOLD`, `MAX_RETRIES`,
`NULL_PCT_QUALITY_LIMIT`, allowed model list) into a versioned `policy.yaml`, and
record which policy version governed each run.

This reframes the project from "a pipeline with thresholds in it" to "a configurable
governance engine" — a much stronger claim, for maybe 100 lines.

**1.4 Reviewer identity and sign-off**

Add a reviewer field, log who approved what, and optionally require dual sign-off
when fairness failed but the reviewer approves anyway. An approval record with no
approver identity is not an audit trail.

### Tier 2 — Genuine capability extensions

**2.1 Fairness that can actually fix things**

- Add **equalized odds**, **equal opportunity** and **calibration-by-group** — DI and
  DPD alone are the weakest pair in common use.
- **Intersectional subgroups** (sex x race), where real violations hide and where
  most tooling still falls short.
- **Auto-bucket continuous attributes** (age -> <25 / 25–60 / >60). The code already
  flags this as a known limitation with a TODO.
- **Add mitigation, not just detection**: Fairlearn's `ThresholdOptimizer` or
  reweighing as a fourth reroute option — "Reject and auto-mitigate" — then re-audit
  and show before/after DI. This closes the loop the project currently only gestures
  at.

**2.2 Post-deployment monitoring — closing the lifecycle**

Right now the graph ends at approval, which is exactly the one-time-gate model the
commercial platforms have moved away from. Add a second graph: score new data,
compute PSI/KS drift per feature, detect performance decay, and **reopen the
governance gate** when a threshold trips. Same interrupt mechanism, reused.

This turns a one-shot pipeline into a lifecycle system, and it is the single biggest
conceptual upgrade available.

**2.3 OpenTelemetry instrumentation**

Wrap each node in an OTel span using the GenAI semantic conventions (`gen_ai.*`
attributes on the planner call: model, token counts, latency, cost). Standards-based,
currently fashionable, and it gives you a real latency/cost table for the report
instead of anecdotes.

**2.4 Counterfactual explanations alongside SHAP**

SHAP says which features mattered in aggregate. Regulators increasingly want
**recourse** — "what would this individual have had to change?". DiCE-style
counterfactuals for a selected row would visibly outclass a standard SHAP bar chart.

### Tier 3 — Research-grade, if there is time

**3.1 A quantitative evaluation** — *the biggest gap in the project as an academic artifact*

There is currently **no measurement of whether the governance loops actually work**.
That is the obvious question a reviewer will ask, and right now the answer is a
screenshot.

Proposed experiment: take 5–10 datasets (Adult, COMPAS, German Credit, Bank
Marketing, Diabetes). For each, run three arms:

- **A**: pipeline with loops disabled
- **B**: auto-retry loop only
- **C**: human reroute simulated by a scripted policy

Report DI, DPD, accuracy, and wall-clock/token cost per arm. A table showing
"fairness violations dropped from 7/10 to 2/10 for a 12% accuracy cost" is worth more
than every screenshot in the README combined. This is what turns the project from a
demo into a result.

**3.2 LLM ablation** — run the planner across three Groq models and report plan
quality and JSON-validity rate. Cheap, and directly answers "why this model?".

**3.3 NIST AI RMF tagging** — tag each audit event with the function it evidences
(GOVERN / MAP / MEASURE / MANAGE). Crosswalks then let one evidence set speak to the
EU AI Act, ISO 42001 and NIST simultaneously. Mostly a labelling exercise with a
large framing payoff.

**3.4 Retire `app.py`** — two UIs is confusing. Delete it or move it to `legacy/`.

### 3.5 Recommended sequence

For maximum reviewer impact within bounded effort:

1. **Tier 0** — everything. Non-negotiable, a few hours.
2. **1.1 hash-chained audit log** — highest impact-to-effort in this document.
3. **1.2 compliance artifact generation** — the actual differentiator.
4. **3.1 quantitative evaluation** — what makes it a project rather than a demo.
5. **2.1 fairness mitigation** — closes the loop the thesis promises.
6. Then 1.3, 2.2 and 2.3 as time allows.

Items 2 and 3 are what make the project *unlike* LightAutoDS-Tab, AIDE and DS-Agent.
Item 4 is what makes it defensible under questioning. Chasing AutoML capability
instead — more models, hyperparameter search, feature engineering — means competing
directly with well-funded research groups on their own turf, and is the one direction
that will not pay off.

---

## Sources

- [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/pdf/2410.07095)
- [LightAutoDS-Tab: Multi-AutoML Agentic System for Tabular Data](https://arxiv.org/pdf/2507.13413)
- [AIDE: AI-Driven Exploration in the Space of Code](https://arxiv.org/pdf/2502.13138)
- [DSGym: A Holistic Framework for Evaluating and Training Data Science Agents](https://arxiv.org/pdf/2601.16344)
- [TML-bench: Benchmark for Data Science Agents on Tabular ML Tasks](https://arxiv.org/html/2603.05764)
- [LanG — A Governance-Aware Agentic AI Platform](https://arxiv.org/html/2604.05440v1)
- [EU AI Act Article 11: Technical Documentation](https://artificialintelligenceact.eu/article/11/)
- [EU AI Act 2026 Updates: Compliance Requirements and Business Risks](https://www.legalnodes.com/article/eu-ai-act-2026-updates-compliance-requirements-and-business-risks)
- [AI Model Cards & Data Provenance: What 2026 Compliance Demands](https://www.techaheadcorp.com/blog/ai-model-cards-data-provenance/)
- [From Shadow AI to AI-BOMs: A Proactive AI Governance Framework](https://obot.ai/blog/shadow-ai-to-ai-boms-proactive-ai-governance-framework/)
- [Stanford HAI — Responsible AI, 2026 AI Index Report](https://hai.stanford.edu/ai-index/2026-ai-index-report/responsible-ai)
- [NIST AI RMF Implementation Guide (April 2026)](https://www.openlayer.com/blog/nist-ai-rmf-implementation-guide)
- [OpenTelemetry GenAI Semantic Conventions](https://greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions)
- [Detect & Mitigate AI Bias: 7 Open-Source Tools (2026)](https://www.turingpost.com/p/ai-fairness-tools)
- [OxonFair: A Flexible Toolkit for Algorithmic Fairness](https://arxiv.org/pdf/2407.13710)
- [fairlib: A Unified Framework for Assessing and Improving Classification Fairness](https://arxiv.org/pdf/2205.01876)
- [Top AI Governance Platforms for Agentic AI in 2026](https://www.arthur.ai/column/best-ai-governance-platforms-2026)
