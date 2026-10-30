# Research 640: RT-J — Relational-Foundation-Model Recipe (Walk-Proximity Context, Corpus Pipeline, Test-Time Scaling)

> **Source:** "RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions" — [arXiv:2610.17867](https://arxiv.org/abs/2610.17867) (NeurIPS 2026), Ranjan, Kothapalli, Agarwal, Kanatsoulis, Upendra, Palczewski, Guestrin, Leskovec (Stanford STAR). CC-BY 4.0. Sites: [star-project.stanford.edu/rt-j](https://star-project.stanford.edu/rt-j); code [stanford-star/relational-transformer](https://github.com/stanford-star/relational-transformer); data [stanford-star/the-join](https://huggingface.co/datasets/stanford-star/the-join) + [plurel](https://huggingface.co/datasets/stanford-star/plurel); models [rt-j](https://huggingface.co/stanford-star/rt-j) / [rt-plurel](https://huggingface.co/stanford-star/rt-plurel).
> **Date:** 2026-10-30
> **Status:** Active
> **Related Research:** 364 (TabFM — the single-table ICL predecessor family; RT is its multi-table relational successor), 238 (Muon — same optimizer family RT-J uses)
> **Cross-ref:** riir-neuron-db issue 640 (walk-proximity retrieval mode) · riir-reflex issue 104 (learnability-gate class) · riir-train issue 657 (recipe backlog) · ndb `.research/312` (lexical/facet retrieval reference — sibling surface) · ndb Plan 319 (`experience_graph` — the traversal substrate the walk ranker would join)
> **Classification:** Public

---

## TL;DR

RT-J is an 85M-param relational foundation model (cells-as-tokens; column/feature/neighbor attention over PK/FK links) that matches strong feature-engineering+GBDT pipelines on RelBench with **10–25× fewer labeled in-context examples**. The transferable part is NOT the architecture — it is the **context-construction recipe**, which is almost entirely training-free: a two-stage **random-walk + budgeted-BFS context retriever** that runs on-the-fly with zero precompute and (per the paper's own ablation) *beats embedding-similarity retrieval* at ranking relational context; **query-centered context with implicit label induction** (shared join cells represented once); **sampler-seed context ensembling** (average predictions over independently sampled contexts); and a **corpus pipeline** (LLM proposes tasks inside a registered template taxonomy → deterministic schema checker → learnability filter vs cheap baselines → difficulty tiers). Synthetic-first pretraining (PluRel → THE JOIN continued pretraining) is externally validated curriculum.

**Distilled for the workspace (modelless-first):**
The paper's ablation is evidence for a *modelless* law: **structural graph proximity (walk visit-counts) ranks relational context better than embedding-cosine similarity, with no index and no features** (walk 33.4 nMAE vs cosine-over-learned-embeddings +2.3–3.2 worse at every context size; ties an engineered-feature retriever at 33.5 with zero precompute). Plus a test-time-compute law (ensembling over *retrieval-sampler* seeds, `Var(mean) = σ²/n`) and a corpus-synthesis gate class our synth lane has never measured (learnability-vs-cheap-baselines). The trained-model half decomposes into an 8-item recipe backlog for our training lanes.

---

## 1. Paper Core Findings

**Model.** 85.6M params (12 layers, hidden 512, 8 heads), QK-Norm; every DB cell is a token whose embedding = per-type value encoding + frozen `all-MiniLM-L12-v2` embedding of `"<column> of <table>"` schema text. Each block: column attention → feature (row) attention → neighbor attention (PK/FK links), gated; Huber loss on all targets; one checkpoint serves classification + regression.

**The recipe (what made RT [ICLR 2026] into a few-shot ICL model):**
1. **Two-stage context retrieval.** Stage 1: W=10,000 random walks × K=20 steps from the query row over the PK/FK graph, counting visits to same-table rows (Pixie/PinSAGE-style personalized walk proximity), with **walk-level temporal validity** (a step may only move to a neighbor whose timestamp ≤ the target's — the leak guard is inside the sampler, changing the walk distribution itself, not an output filter) and an optional recency ordering ρ. Stage 2: consume ranked seeds in order, each launching a width-bounded BFS until the total cell budget L (≤8,192) fills. Zero precompute, zero index; the CPU sampler stays ahead of the GPU forward pass; 28.1 ms/example at full (W,K), 14–18 ms at cheaper settings with minor accuracy loss.
2. **Query-centered context.** One window around the query; labeled support examples are *induced* by the expansion (other same-table rows met en route carry observed labels). Shared join cells across candidate examples are represented **once** — the classical per-example-context formulation duplicates them and burns budget.
3. **Test-time compute.** **Context ensembling**: average predictions over n_ens ∈ {1,2,4,8,16} independently *sampled contexts* (randomness in the sampler seeds, not the model): +1.1 AUROC aggregate at the default context, +1.3 tuned; per-task extremes +5.3 / −3.0. **Context tuning**: per-task grid over (L, per-seed budget, BFS width, recency flag) on validation — regression 33.4→32.5 nMAE (8/9 tasks improve), classification single-seed *hurts* on average (72.9→72.4; val→test transfer on only 5/12 tasks — tuning needs a transfer gate).
4. **THE JOIN corpus pipeline** (639 real DBs, 6,023 forecasting + 10,600 autocompletion tasks): collect (5 source classes, per-source license tags, benchmark overlap excluded by source *identity*) → standardize (5 idempotent detector–transformer fixes: boolean coercion via canonical truth-set; numeric coercion with identifier blacklist guarding zip codes/ISBNs; FK inference = name match AND ≥80% value containment; one time column per table; entity-axis subsampling that preserves complete activity histories) → LLM proposes tasks constrained to **25 registered template families** (8 reasoning regimes incl. aggregation, churn, ranking, streak, tail-event) → **deterministic checker** (columns exist, entity col is a real FK, pattern registered, no name collision, thresholds near median, no trivial imbalance; refusals logged with reason) → **XGBoost learnability filter** vs cheap historical baselines (entity-mean, last-value): keep learnable tasks AND hard-but-non-trivial ones, tiered easy/medium/hard → multi-pass proposer targeting under-represented templates.
5. **Synthetic-first curriculum.** PluRel (1,900 synthetic DBs, 86k tasks) then continued pretraining on THE JOIN: synthetic-only beats random init (+1.7 nMAE FT / +0.8 AUROC); real CPT adds more (+0.5/+0.5; in-context +1.5 AUROC over synthetic-only). Forecasting+autocompletion **mixture** beats either family alone (autocompletion-alone: −3 AUROC and degrades over training).
6. **Schema semantics.** Real table/column names carry world knowledge: shuffling names at eval costs up to 2.1 AUROC / 0.8 nMAE and the gap **widens with more examples**.

**Results.** Regression: RT-J beats every label-matched baseline at every context size (2.3–5.0 nMAE); 15 examples beat the best baseline at 225; the strongest pipeline needs ~2,300 (10×) to surpass. Tuned+ensembled: 32.4 nMAE at a few hundred labels vs RDBLearn 31.4 at 10,000 (≈25× fewer labels). KumoRFM-2 (closed) still ahead at 29.1. Classification: leads by up to 3 AUROC when labels are scarce; extended-context baselines (20× labels) pass it at the largest budgets. Pretraining: ~1 day/phase on 8×A100, Muon (2D weights) + AdamW, constant LR, EMA/SWA checkpoint selection (momentum 0.9995, ~2k-step window).

**Key ablation (the one this note is mostly about).** Retrieval comparison at inference, all else fixed: random-walk 33.4 nMAE ≈ RDBLearn-engineered-feature similarity 33.5, both far ahead of uniform random (+4.1 nMAE / +6.1 AUROC behind), RT-embedding-cosine similarity (+2.3–3.2 nMAE behind at every L), and plain BFS (+1.6–1.8 nMAE behind with 2.7×/5.6× *fewer* labels retrieved). Authors: "retriever quality is not strongly correlated with label count; relational context is important too." Masking ablation: masking non-target cells at any rate costs 1.6–2.6 AUROC / 0.6–1.5 nMAE — masked cells corrupt the very context the prediction reads from.

**License note.** THE JOIN's assembly is CC-BY-SA 4.0 but per-source upstream terms carry (research-only, NC, ToS-restricted rows — e.g. Manifold/Polymarket ToS prohibit commercial AI/ML training). Any reuse of the corpus in commercial lanes must honor per-row restrictions; the contamination-exclusion practice (benchmark overlap by source identity) is itself a transferable hygiene law.

---

## 2. Distillation

### 2.1 Path 0 inventory — modelless track (training-free extractions)

Panel run: No-GD advocate returned a 21-item inventory (concurred). The consolidated, coverage-checked set:

| # | Mechanism | Closed-form/derivable? | Coverage in workspace (signal-diff) | Action |
|---|---|---|---|---|
| M1 | **Walk-proximity ranker** — visit-count ranking of same-table rows by W×K random walks; Monte-Carlo estimate of personalized (reset-at-query) walk affinity | Yes — deterministic per-seed sampler, zero training | **Partial cousin, real gap.** ndb `experience_graph` (Plan 319) seeds traversal by *latent cosine* over `task_embedding` then walks edges — embedding-dependent seed, and the paper's ablation is precisely the claim that walk-proximity *beats* embedding-similarity ranking. `GossipRing` pushes state (propagation, not retrieval); mmorpg wander-walk is movement noise. No embedding-free visit-count ranker anywhere | **riir-neuron-db issue 640** (third retrieval mode in `ExperienceGraph` + `KgTripleIndex`/AnyRAG consumers) |
| M2 | **Walk-level temporal guard** — edge admissible iff `ts(neighbor) ≤ ts(query)`, enforced *inside* the sampler so the measure is time-respecting-path proximity | Yes — per-step predicate | Gap. ndb `as_of` reconstructs the mutation log AS-OF (replay axis); no walk admissibility predicate exists | Rides issue 640 (same sampler) |
| M3 | **Budgeted two-stage builder + query-centered dedup** — ranked seeds → width-b BFS under a total cell budget L; shared join cells represented once | Yes — set accounting | Gap. No relational-window builder in any index | Rides issue 640 |
| M4 | **Sampler-seed context ensembling** — average scores over n independently seeded contexts; `Var(mean)=σ²/n` floor; per-task reporting mandatory (paper's own −3.0 AUROC outlier proves pooled-only gates green real losses) | Yes | **Cousin, different purpose.** `katgpt-core::perturbation_ensemble` + reflex `mc_ensemble` perturb *embeddings* to MEASURE uncertainty (sample 0 stays the served answer; the wrapper is "never a silent re-route"). RT-J averages over *retrieval samples* and the ensemble IS the answer (variance exploited for accuracy). Same determinism discipline transports: per-seed BLAKE3-derived sampler seeds, byte-identical re-runs, N=1 = legacy bytes | Recorded; activates in issue 640 as an optional phase once a stochastic retrieval mode exists (reflex has no relational context sampler yet) |
| M5 | **Transfer-gated per-task context tuning** — validation argmax over retrieval hyperparameters, shipped only where a held-out val→test read is positive | Yes — pure selection | Gap (no per-task retrieval-config selection exists; reflex `--cal-select-cap` selects corpus cap on calibration — nearest cousin, different knob) | Recorded as the gate *shape* for issue 640's tuning phase |
| M6 | **Corpus-synthesis standardization + eligibility + deterministic checker + deficit-targeted multi-pass** | Yes — all deterministic | Partial. reflex synth lane has teacher agreement VETO + echo gates + density gates; no eligibility-style structural pre-gate, no template-registry deficit targeting | Template-taxonomy + deficit-pass ideas recorded here; the actionable gate class is M7 (issue filed) |
| M7 | **Learnability filter vs cheap baselines** — score each generated task by margin over entity-mean / last-value (paper: GBDT; the *modelless* transport is the baseline-margin score itself); keep learnable + hard-but-non-trivial; difficulty tiers | Yes — closed-form baselines (mean/persistence), no GD needed | **Untested gate class.** Benches 120/121/122 measured ungated > every gate strength we tried — but those were teacher-agreement/density gates; baseline-solvability margin was never measured. External evidence the class works for a pretraining corpus | **riir-reflex issue 104** (cheap frozen V5 read on the existing sealed artifact; honest null is a live outcome) |
| M8 | **Schema-name shuffle probe** — eval-time consistent name shuffling; the widening-with-context delta certifies names are load-bearing | Yes — fixed permutation seed | Gap as a *probe*; ndb BM25 keys on names and reflex embeds text — neither ships the diagnostic | Recorded; free to attach to any future retrieval/eval gate |

### 2.2 Path 0.5 — training track (recipe backlog; panel advocate concurred)

| # | Recipe item | Our surfaces | Cost class | Action |
|---|---|---|---|---|
| R1 | Context-length sampling at train time (one checkpoint, every context; + retrieval-hyperparameter randomization → test-time tuning/ensembling validity) | specialist heads (instinct/rethink) at head scale = 10–30 GPU-h; 27B CPT = riir-train lane, not a side bench | Small (heads) | **riir-train issue 657** |
| R2 | Target-only masking / completion-only loss (masking non-targets *cost* 1.6–2.6 AUROC — supervision density is negative when it corrupts conditioning) | quest_grammar LoRA + specialist SFT: the completion-vs-full-sequence-loss convention converted from convention to measurement; 1–5 GPU-h/arm, cleanest A/B in the set | Trivial | issue 657 |
| R3 | Synthetic-first curriculum (synthetic pretrain → real CPT; convergence-speed + stays-ahead effects) | reflex sealed synth artifact as phase-1 material; specialist winners currently train real-only; Bonsai distill is synthetic-first in spirit but untiered | <1 4090-day | issue 657 |
| R4 | Task-family mixture (single-family arms degrade over training) | per-suite specialists ARE the single-family arms; one mixture trunk → per-suite specialize | 10–20 GPU-h | issue 657 |
| R5 | Learnability-gated generated corpus | = M7's training-side twin; scorer doubles as R6's GBDT control | CPU-minutes + one V5 read | reflex issue 104 (+ cross-ref from 657) |
| R6 | Label-budget curves (n ∈ {50,200,1000}) for trained-vs-modelless comparisons | instinct `stats.rs` Wilson instrument exists; E0 density probe measures the axis modelless-side | Eval-dominated | issue 657 |
| R7 | Constant LR + EMA/SWA selection (stability claim, ~0 incremental compute) | every training lane; bolt onto next scheduled retrain | ~0 | issue 657 |
| R8 | Schema-name semantics conditioning (weakest transfer — web schema names are world-knowledge-dense; our suite/lint names are thinner) | suite-name conditioning in specialists; healer lint-name retrieval | 1–2 GPU-h eval | recorded in 657 as lowest priority |

### 2.3 Fusion (mandatory subsection)

**Closest cousins across the workspace, and what paper × cousin produces that neither has alone:**
1. **ndb `experience_graph` (Plan 319) × RT-J walk ranker** → a *structural* retrieval mode that needs no embeddings: today's traversal must embed the query to seed; a visit-count ranker ranks by relational proximity alone, works on cold schemas, and (per the paper's ablation) can beat the cosine seed. Fusing the temporal guard (M2) with `as_of` gives time-respecting experience retrieval for non-stationary NPC worlds. *Game reframe (priority #1 surface):* this is the NPC think-brain's memory-selection law — from a stimulus entity, walk the KG/social/encounter graph, rank belief triples by visit-count proximity, staleness stays a separate axis (`decay_confidence`); `limelight`/`goal_salience` gate attention *cost rings*, not *which memories surface*, so no conflict — the two compose.
2. **reflex synth lane × THE JOIN pipeline** → the lane's gates (agreement VETO, echo, density) measure *teacher consistency and distribution*; RT-J adds the missing axis: *is the task learnable-at-all / non-trivially-solvable* (margin over trivial baselines). Also transfers: eligibility pre-gates, template-registry deficit targeting, and the corpus-hygiene provenance law (license tags + contamination exclusion by source identity — grammar-not-answers-adjacent).
3. **`heal_with_pool` (riir-refine) × kinship-walk context** (healer reframe, speculative — recorded, no issue): kernel_opt corpus entries already carry kinship annotations ("compose-never-merge", "distinct-from"); a rule-kinship graph walked from the matched rule could diversify escalation-pool membership beyond score ranking. Needs the rule-graph to exist first; the ndb primitive (M1) would be the shared substrate.
4. **`mc_ensemble`/`perturbation_ensemble` × sampler-seed ensembling (M4)** → same determinism contract (per-seed BLAKE3, N=1 identity), new purpose (accuracy vs uncertainty). If ndb lands a stochastic retrieval mode, the ensemble wrapper pattern ports directly.

---

## 3. Verdict

**Tiers (applied):**

| Tier | Criteria | Verdict |
|---|---|---|
| Super-GOAT | Novel mechanism (no prior art) + new capability class + selling point + force multiplier | **NO — Q1 fails.** Walk-visit-count retrieval is published prior art: Pixie (WWW 2018) and PinSAGE (KDD 2018) are the paper's own cited origins; the KG-RAG class (G-Retriever, Think-on-Graph, SubgraphRAG ICLR 2025, Walk&Retrieve 2025) ships walk-based subgraph retrieval broadly. Claiming mechanism novelty would be a false claim |
| **GOAT** | Provable gain over existing approach | **Modelless track: GOAT-tier fusion, gated on PoC.** The M1 fusion (structural ranker vs cosine seed on our graphs) has a concrete pre-registered bar — the paper's own ablation (beat cosine ranking; tie engineered features) — and a cheap toy-graph gate. **Training track: Gain** (recipe items are real but each lane owns its adoption) |
| Gain | Incremental, useful | Corpus-gate class (M7) and recipe backlog (R1–R8) are Gain — filed as issues |

**Per-track verdicts (no cascade):**
- **Modelless (a+b):** Gain → `riir-neuron-db` issue 640 (M1–M5 fused: walk ranker + temporal guard + budget/dedup + optional ensemble + transfer-gated tuning), `riir-reflex` issue 104 (M7). M6/M8 recorded in this note only.
- **Model-based (c):** Gain → `riir-train` issue 657 (R1/R2/R3/R4/R6/R7 backlog, R5 cross-ref, R8 lowest).

**MOAT gate per domain (§1.6):**
- `katgpt-rs`: the generic primitive (visit-count rank sampler + temporal admissibility + budget accounting) is a base-primitive fit — lands in `katgpt-core` **when ndb/ai proves the gain** (per-stack ledger: feature flag + bench + GOAT; no consumer-first violation).
- `riir-neuron-db`: retrieval-index novelty, in-scope (AnyRAG/experience_graph/KG) — the issue lives there.
- `riir-reflex`: corpus-synthesis gate class — the issue lives there; Benches 120/121/122 stay the controlling internal record (ungated won, until a *learnability* gate is actually measured).
- `riir-train`: recipe backlog — issue; no Plan (no committed training run; adoption is per-lane, owner-gated).
- `riir-ai`: game reframe recorded in §2.3(1); no issue filed — it consumes ndb's primitive first (fusion-priority ladder: game surface benefits via the ndb substrate it already path-deps).

**Selling-point test (why this is GOAT not Super-GOAT):** "Our experience/KG retrieval ranks context by walk-proximity with zero embeddings, zero precompute, and a leak-proof temporal guard" is a real differentiator vs cosine-only indexes — but the mechanism is published, so it strengthens the moat by *fusion and evidence*, not by novelty.

---

## 4. What ships where

| Repo | Artifact | Status |
|---|---|---|
| `riir-neuron-db` | `.issues/640` — walk-proximity retrieval mode (M1–M5) | committed `3792a1d4aec2` |
| `riir-reflex` | `.issues/104` — learnability-gate class for synth corpora (M7) | committed `f7032899cf74` |
| `riir-train` | `.issues/657` — RT-J recipe backlog (R1–R8) | committed `b5ce4c8442a6` (merged into develop at `94860ad3`) |
| `katgpt-rs` | this note; future `katgpt-core` primitive once a consumer lands | this commit |
| `katgpt-rs/.research/364` | one-line cross-ref added (RT-J = the relational successor of the TabFM family) | edited this commit |

## 5. Open questions / risks

- **Honesty about the null:** reflex Benches 120/121/122 (ungated corpora beat every gate) is the controlling internal record; issue 104 must treat a learnability-gate null as the live prior, not assume RT-J transfers. RT-J's filter served a *pretraining* corpus; ours serve ICL retrieval.
- **The walk ranker's regime:** the paper's win is multi-table relational data with heterogeneous link structure. Our KG triples are sparser and typed; the toy-graph PoC must include a sparse-typed arm or the gate proves nothing about our graphs.
- **Latency class:** ndb's online budget is sub-50 µs; the paper's sampler is 14–28 ms/example. The walk ranker is an OFFLINE/batch-mode candidate first (like `TraversalBudget::offline`); an online variant would need W,K orders of magnitude smaller and its own gate.
- **THE JOIN license:** per-source NC/ToS rows make wholesale corpus reuse in commercial lanes unsafe without a per-row license pass.
- **Context ensembling (M4)** has no consumer until a stochastic retrieval mode exists; do not pre-wire it.
