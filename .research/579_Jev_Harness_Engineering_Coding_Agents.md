# Research 579: Jev Harness Engineering for Coding Agents — the Decision-Point Map Around the Decision Layer

> **Source:** "Jev Engineering for Coding Agents — The TypeSafe Founder's Blueprint for Building with Jev" — independently compiled working note, September 2026 (PDF at `riir-reflex/.raw/`, pdftotext-extracted; explicitly "not affiliated with or endorsed by TypeSafe"). Underlying material: design notes by Diogo Almeida (TypeSafe founder) — public via the Latent Space podcast "Jev: System One models for Prod, not God — with Diogo" (latent.space/p/jev), the typesafe.ai blog, and @CompleteSkeptic. Sources section also cites Microsoft fastcontext figures (56.2% of tool-use turns, 46.5% of main-agent tokens on reading/searching in GPT-5.4 trajectories) and Zhang 2025 Recursive Language Models.
> **Date:** 2026-09-22
> **Status:** Done — **Gain (narrow)**: filed riir-reflex `.issues/004_jev_harness_decision_point_map.md` (the six decision points as the Proposal-014 arena consumer map, feeding the Phase-3 corpus). No new katgpt-rs primitive — the decision-model half is saturated and the harness-compression half has published prior art.
> **Related Research:** 562 (TypeSafe SystemOne/Jev product distill — the decision model itself), 573 (CUA-S1 open recipe), 574 (structured reads / vLLM #57250), 576 (laya open generalist arena), 251 §1.5-O5 (security-aware token budgeting — the routing-trust cousin), 146 (RLM/GEPA — the RLM PASS class), 440 (AIDE² — the harness-engineering-as-LLM-process PASS class, incl. the Ning et al. arXiv:2605.18747 §3.5 survey adjudication)
> **Related Plans:** Proposal 014 (Phases 3 flywheel + 4 hosted plane remain — this note is Phase-3 intake), Plan 603 / 606 (reflex landed Phases 1–2), katgpt-rs `decision_wire` (Plan 603 T1.2)
> **Cross-ref (workspace):** riir-clippy (the healer IS the shipped harness instance — span context assembly, intent routing, typed validated edits, consent gates), riir-reflex (the decision layer), riir-neuron-db `ShardIndex` (shared retrieval), riir-game-sdk `crates/riir-games-cluster` `aoi_reveal_bridge` (the game-vocabulary visibility ladder), riir-ai `riir-rag::ast_chunker` (the healer's chunk substrate)
> **Classification:** Public

---

## TL;DR

This synthesis's **decision-model half is already ours**: Jev's typed choice/score/noul-with-probabilities contract was distilled in 562, replicated openly in 573/574, and now SHIPS as `katgpt_core::decision_wire` + the `riir-reflex` engine (Phases 1–2 complete, live at reflex.gist.rs). The **new content is the harness architecture around that layer**: Almeida's "how would you design a coding agent if LLMs had no KV cache?" thought experiment, the six KV-cache symptoms (routing fails on reprocessing cost, tool schemas crowd context, compaction is query-blind, sub-agents are rare, restarts discard good state, batteries cost permanent context), and six **per-turn Jev decision points** (context visibility, cache reuse, routing, tool fit, permissions, security sensitivity). For us the actionable delta is exactly one thing: **that six-point table is the coding-agent consumer map the Proposal-014 arena has been missing** — every row maps 1:1 onto `decision_wire` vocabulary (`choice`/`score`/`noul`) and becomes an arena task family. Filed as **riir-reflex Issue 004**. Everything else is covered-or-adjacent (§2 map); the compression class has published prior art (KVzip, Quest, Code Comp), so no primitive claim is made or needed.

**Source caveat carried:** this is a *synthesis of design notes*, not a peer-reviewed paper — the routing arithmetic (4.15 vs 6.19 at X=0.65/Y=0.12/Z=0.23) and the token-share table are explicitly illustrative estimates; treat as framing, not evidence.

## 1. Source core claims

**The thesis:** coding agents are simple loops; the leverage is in what the loop feeds the model each turn. Jev is not the model that writes code — it is the **decision layer beside it**, answering a predefined question over (goal, context, rules, available actions, previous actions) with a typed answer + probability at high-frequency decision points ("every native feature in this note is, underneath, a Jev question asked at a high-frequency decision point").

**The no-KV-cache thought experiment:** the KV cache is why agents are append-only transcripts (changing anything early invalidates everything after). Designing as if it did not exist yields six symptoms the paper claims current agents inherit unexamined:

| # | Symptom | The paper's fix |
|---|---|---|
| S1 | Routing fails — mixed routes cost MORE than pure frontier (load-on-the-way-down + reprocess-on-the-way-back ≈ 6.19 vs 4.15 pure-Opus units) | Price routing **per context rebuild**, not per token; hand the cheap model a small purpose-built context; merge results back as scored chunks |
| S2 | Tool schemas crowd context (all schemas in the system message always) | Tiered disclosure: snippets (100s of tools, one line each) → schema on demand → docs; harness-as-router (model states intent in plain text, typed calls select the tool + construct args) |
| S3 | Compaction is query-blind (compresses before the question is known) | **Visibility ladder** per chunk per query: hide / short / long / full — query-aware compression; nothing is deleted from state |
| S4 | Sub-agents are rare (passing state in and merging back is hard) | Make context construction cheap + automatic; explicit read/write typing so read-only tasks never contend |
| S5 | Restarts discard good state with bad | Addressable typed state; reload only the still-relevant chunks on demand |
| S6 | Batteries debate (every built-in costs permanent context) | Near-zero-cost built-ins via tiered disclosure dissolves the ease-vs-power tradeoff |

**The six Jev decision points (the note's Table I — the actionable row for us):**

| Decision | Question | Typed answer |
|---|---|---|
| Context visibility | How visible should this chunk be for this query? | `choice`: hide / short / long / full |
| Cache reuse | Reuse the cached prefix or rebuild? | `noul` + probability |
| Routing | Can this subtask leave the frontier model? | `choice` + cost estimate |
| Tool fit | Which tool fits this intent? | ranked `choice`, top-k |
| Permissions | Should this command run? | `choice`: allow / ask / deny |
| Security | Which files will this task touch? | sensitivity `score` |

**Token budget (illustrative):** reading files 30–40% · searching 10–18% · command output 10–20% · system+tools+AGENTS.md 5–12% · **writing code 4–10%** — retrieval dominates ~2/3 of processed tokens (corroborated by fastcontext: 46.5% of main-agent tokens). The paper's conclusion: the largest efficiency gain in a coding agent is smarter retrieval, not a better model.

**Other designs:** programmable permissions (a policy DSL over commands + file-contents inspection, e.g. deny if script contents contain network egress); goal deduplication (register subgoals, never launch work already done or in flight); conditional AGENTS.md (instruction fragments bound to conditions — file patterns, directories, task type — reloaded whenever the condition holds, immune to compaction; distinguished from skills: "do this now" vs "keep this in memory"); security-aware routing (route by file sensitivity — public/standard/restricted/custom — not just difficulty/cost; trust as a third routing axis); background processing (ONE shared retrieval pass — relevant files/symbols/diffs found once — feeding many read-only background tasks: eval generation, ELI5 explainer, progress page, cross-model review); Recursive Language Models (Zhang 2025: agent state as explicit named variables rather than transcript text).

## 2. In-workspace coverage map (vocabulary translation, both layers)

| Paper mechanism | Ships here? | Evidence (signal-diffed, not name-matched) |
|---|---|---|
| Jev decision layer (typed choice/score/noul + probabilities) | **SHIPS — saturated** | Research 562 (distill) + 573 (open recipe) + 574 (read-only structured reads, `structured_read` Bench 816) + 576 (arena); `katgpt_core::decision_wire` (Plan 603 T1.2) + `riir-reflex` serving binary (Phase 1–2 COMPLETE, reflex.gist.rs live). Abstention is our structural edge (Jev cannot; we can) — 562's recorded flaw, unchanged by this note. |
| Per-query chunk scoring → visibility ladder | **Kernel + healer SHIPS; the LADDER-as-decision is the gap** | Kernel: `katgpt-attn::hga_forward::forward_hga` scores chunks per query (`simd_dot_f32(query, summary)`) + entmax top-k selection — the scoring half, minus the 4-level budget output. Healer: `LatentMatcher` ranks spans per query across the fan-out. What does not ship anywhere: the **visibility level** (`choice` over hide/short/long/full, budget-scaled) as a typed decision output rather than a raw score → folded into Issue 004 T2. |
| Cache-reuse noul | **N/A to our hot path by construction; arena-lane relevant** | The modelless decision lane HAS no KV cache — the paper's thought experiment is our default operating mode (assembled context, no accumulated transcript). The decision is real only for LLM lanes (laya lane, hosted plane Phase 4) → Issue 004 T3 scopes it there. |
| Harness-as-tool-router (intent → typed tool selection) | **SHIPS (shape)** | Substrate: `pick_domain` / `pick_domains_top_k` (katgpt-core `variable_rank_domain_expert`); consumer: the healer's `KernelExpertRouter` / `HealDomain` fan-out routes intent → domain, `RerankMode::Structural` = ranked top-k tool fit; the healer constructs typed edits (not free-form tool args) validated by the compile gate. Covered. |
| Programmable permissions DSL | **Adjacent — and the healer ships a stronger form over edits** | Two healer surfaces ARE this pattern: `--fix --repo-posture` applies an edit **iff the target repo's own clippy flagged that exact span** — a programmable permission over edits, expressed as a query over what is allowed, closer to the paper's DSL than any command gate (clippy_lints LANE ONLY — the target-gate mirror; the rust_perf/sec lanes are this tool's own rules the target's clippy says nothing about); and the consent planes (`--mine`/`--stats`/`--unmine`) + the KAT soft gate implement the paper's allow/ask/deny ladder over contribution (deny depleted → ask via the contribute-now y/N offer). No policy DSL over arbitrary command execution exists — we ship no coding-agent CLI — external surface. Covered for edits, recorded for commands. |
| Security-aware routing (sensitivity axis) | **Cousin recorded — and the healer ships the data-egress form** | The healer's `.heal-sync-policy` v3 (deny-by-default, own-repo + source globs, `allow_redacted: false`) routes DATA EGRESS by file sensitivity — the paper's trust axis applied to what code may leave the machine, which for a private-source healer is the load-bearing direction. Plus the recorded cousins: Research 251 §1.5 O5 ("security-aware token budgeting", partially Plan 308 Cognitive Integrity Layer) and the raw-vs-latent sync-boundary law (our sensitivity tiers over what crosses the boundary). We do not multi-vendor-route compute. No new primitive. |
| Shared retrieval for background tasks | **SHIPS structurally** | `riir-neuron-db::ShardIndex` (lock-free papaya, O(1)+KNN) is one shared retrieval surface; the healer's multi-domain fan-out shares one `LatentMatcher` pass. Read-only-never-contends matches our lock-free-read discipline (papaya, `CARGO_TARGET_DIR` isolation law). Covered. |
| Goal deduplication | **SHIPS** | riir-agents' `shared_context_core` Cargo feature — the admission gate in `src/shared_context.rs` + ExperienceGraph avoid-set (Plan 529, DeLM) — dedup of context entries/avoid-respawns is the same shape. Covered. |
| Conditional instructions (AGENTS.md) | **External surface** | Zed's skill/AGENTS.md loading is the host's, not ours. Recorded. |
| Recursive Language Models (Zhang 2025) | **PASS class, already adjudicated** | Research 146 (RLM/GEPA PASS: LLM-orchestration, no modelless substrate). The *philosophy* — explicit typed state over transcript text — is our entire architecture by construction (NeuronShard Pods, freeze/thaw, typed wire contracts, `decision_wire`). No action. |
| Addressable state / restart-without-loss | **SHIPS structurally** | freeze/thaw (`MerkleFrozenEnvelope`) + `NeuronShard` + `LocalKvStore` — reload-relevant-chunks-on-demand is the shard-retrieval pattern. Covered. |
| Routing priced per context rebuild (the arithmetic) | **Recorded, not filed** | Economics framing for reflex hosted-plane Phase 4 and any future multi-lane serving: token price ≠ context price; the return trip dominates. Validates our abstain-first economics (an abstained decision costs one µs matvec, not a re-read). No file. |

## 3. Track classification (Path 0 / three-track)

**No training content exists in the source** — zero optimizer/loss/backprop/RL anywhere in the text; the RLCD training question was 562's territory and its audited discard stands (published recipes RLCR/Rewarding-Doubt are their own distill tasks). Track (a) modelless/architecture only. No riir-train plan; no adversarial panel (the training-touch trigger never fires).

**§4 prior-art record (pinned claim):** "the six-point harness decision map for `decision_wire`/riir-reflex, as the Phase-3 consumer intake" — searched: the source's own provenance (Latent Space podcast + typesafe.ai blog confirm the notes are public), and the query-aware-compression class (KVzip: query-agnostic KV compression with context-dependent eviction; Quest: query-aware sparsity; Code Comp: training-free structural KV compression for agentic coding; SAGE; Perplexity's query-aware span-compression module). **The compression math is mature published prior art** — which is precisely why no primitive is claimed here; the delta we file is the *typed-decision framing* (visibility as a `choice`, cache-reuse as a `noul`) and the *consumer map*, not the compression itself.

## 4. Reframes (step 4, both mandatory)

**Consumer reframe (riir-clippy, priority #2):** the healer already IS a Jev-harness instance — span-level context assembly per query (`AstChunker` (riir-rag substrate) chunks scored via `LatentMatcher`), intent→domain routing (`pick_domain`, katgpt-core substrate, consumed via `KernelExpertRouter`), typed outputs validated by a real oracle (the compile gate — stronger than Jev's schema guarantee), consent/soft gates on the write path. The note VALIDATES the healer's architecture rather than extending it; the one lead (a budget-scaled span-visibility ladder) rides Issue 004's task family as an arena lane the healer could consume later — not a separate healer issue, because the healer's retrieval floors already pin what it assembles.

**Game reframe (investment #1):** every harness mechanism has a game-vocabulary twin, and all of them already ship — visibility ladder = fog-of-war AOI reveal tiers (riir-game-sdk `crates/riir-games-cluster` `aoi_reveal_bridge`, per-viewer reveal sets: "how visible should this chunk be" IS the fog-of-war question); cache-reuse noul = the two-brain re-observation decision (`σ(-λ·(t - t_observed))` confidence decay decides keep-stale-belief vs re-observe — the same noul, µs instead of ms); security-aware routing = the raw-vs-latent sync-boundary law + Glacial-tier write rates (sensitivity tiers over what crosses the boundary). 562's arena table (rows 1–15) already holds the game rows; nothing new files here — which is the expected outcome for a harness paper landing on a stack whose hot path never had the transcript problem.

## 5. Fusion

**The note's one reusable synthesis for us: the decision-point taxonomy as the consumer map.** Proposal 014 has the substrate (decision_wire, calibrated heads, the live arena) and a corpus-engine design (KAT contribution network), but the arena's task families are generalist suites — no coding-agent-shaped consumers. This note hands us that **consumer story in one table**: six per-turn questions, each already expressible in our wire vocabulary, each a synthetic-fixture arena task family, each with the same G1 (calibration-vs-floor) + G2 (µs budgets) gate shape. The families extend the Phase-2 arena and feed the Phase-3 corpus engine with task-shaped rows — they gate neither Phase-3's wire/settle tasks (T3.1–T3.3, blocked on their own cross-repo costs) nor anything else. That is riir-reflex Issue 004 — filed, not built here.

**Fusion verdict:** no Super-GOAT (Q1 fails: the taxonomy is an application map over a mature decision contract + mature compression prior art, not a new mechanism; Q2/Q3 do not arise). The Gain is routing + consumer-intake, exactly the shape §1.55 calls actionable: it feeds recorded deferred surfaces (the arena's task-family axis now; Phase 3's corpus engine once consent wiring lands) with a concrete consumer taxonomy — and the visibility-as-typed-`choice` output genuinely ships nowhere in the workspace (`forward_hga` emits scores, not a budgeted 4-level decision).

## 6. PASS-Redirects (synthesis)

> **PASS-Redirects (provenance):** Latent Space podcast — Diogo Almeida, "Jev: System One models for Prod, not God" (<https://www.latent.space/p/jev>) — the public origin of the design notes this PDF independently compiles; the coding-agent-beyond-the-KV-cache framing is Almeida's, the compilation's six-symptom/decision-point tables are the compiler's. Cite the podcast + blog for the ideas, the PDF only as this note's reading copy.
> **PASS-Redirects (compression class):** KVzip (query-agnostic KV compression with context-dependent eviction), Quest (query-aware sparsity, arXiv:2406.10735), Code Comp (training-free structural KV compression for agentic coding), SAGE (shortlist-aware guided encoding, OpenReview 2026) — query-aware/context-aware compression is a saturated published class at the KV layer; the paper's visibility ladder is the HARNESS-layer framing of it, not new inference math. No primitive filed.
> **PASS-Redirects (harness-engineering class):** Ning et al. [arXiv:2605.18747 "Code as Agent Harness"] §3.5 — already adjudicated PASS in Research 440 (LLM-dependent harness-mutation class; the modelless instance ships as riir-clippy `self_evolve`). This note's harness is the decision/context half of the same territory, and its decisions are the part that DOES distill — hence the narrow Gain rather than the 440 PASS.

## 7. Verdict

**Verdict:** **Gain (narrow).** Files created: riir-reflex `.issues/004_jev_harness_decision_point_map.md` (the six-point map → arena task families). No katgpt-rs primitive, no plan, no guide — the decision layer ships, the harness patterns are covered-or-external, and the one actionable artifact is the consumer map. **Implementation landed same day (2026-09-22, riir-reflex `6759a79`):** the six families live as `src/harness/families.rs` — five modelless synthetic suites + `harness_cache_reuse` LLM-only — with T6 gates measured (G2 per-set p99 59–79 µs tail support 11/1000, G4 alloc-free ×5, canary-first) and the verdict round 3's degenerate-pick finding fixed at the root (Issue 004 T7): the engine now blends the drafter delta with the state-alone centroid cosine, a discrimination floor (distinct picks ≥ 2, distinct vectors ≥ 2) pins it in the gates, and every family reads above chance (0.375–0.500) — banking77 0.040 → 0.446, ag_news 0.258 → 0.510 dataset-side; per-suite deltas (two small regressions included) published in Bench 001 Addendum 5, never gated. Reopen triggers: (a) Issue 004 T3's laya-lane measurement on a weights-present harness run; (b) the hosted plane (Phase 4) ever routes across lanes — the priced-per-rebuild arithmetic becomes binding then; (c) Jev/TypeSafe ships a real harness product that measures the six decision points — compare then.

---

> **PASS-Redirects (2026-11-01):** Shoham, David, Akrabi, Bilgory, Levy,
> Marreed, Zeltyn, Shlomov [arXiv:2610.38620 "Think Before You Act: Decision
> Gate for LLM Agents"] — the runtime-gate half of this note's map, now
> PUBLISHED with ablations: policy → (executable checks + yes/no questions +
> Boolean rules + hand-off checklist) compiled offline, selective trigger
> (write calls + hand-offs only), prefill-only judge (Kev, Qwen3.8-27B
> fine-tune), bounded revision with temporary feedback. Its placebo ablation
> (random flags at the gate's rate reproduced the full gate's gain for one
> agent) is the missing evaluation law for every gate-shaped feature —
> distilled at `../riir-reflex/.research/013_JevAct_Decision_Gate_Agent_Runtime.md`;
> the policy→check compiler this class implies stays adjacent-only here
> (no stack consumer today).
