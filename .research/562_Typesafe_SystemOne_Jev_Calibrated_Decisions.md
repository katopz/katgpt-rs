# Research 562: TypeSafe "System One Models" & Jev — Calibrated Structured Decisions

> **Source:** "Introducing System One Models & Jev" — TypeSafe AI blog, Diogo Almeida, 2026-09-15 — https://typesafe.ai/blog/introducing-system-one-models-and-jev (secondary: HN thread 49717558, The Register 2026-09-16, docs.typesafe.ai)
> **Date:** 2026-09-16
> **Status:** Done — Gain verdict, filed katgpt-rs Issue 810 · addendum 2026-09-16: arena head-to-head (11 W / 2 L / 2 SPLIT) · **reopen trigger FIRED 2026-09-19** (independent replication with numbers + a decision-head recipe) → Research 573 · addendum 2026-10-01: Raschka article read in full (OpenAI Decisions API; external CLM/laya IMDb+Tetris cells; instinct-018 tripwire wiring)
> **Related Research:** 322 (Report-the-Floor UQ rule), 311 (conformal line)
> **Related Plans:** Plan 340 (ConformalIntervalCalibrator, default-on)
> **Cross-ref (riir-ai / riir-chain / riir-neuron-db):** riir-ai `arg_runtime/pipeline.rs` (ActionBridge ABSTAIN), `integrity/injection.rs` (5 affect scalars), `.proofs/RiirAiProof/Hla/Bounded.lean`
> **Classification:** Public
> **Harness follow-up (2026-09-22):** [Research 579](579_Jev_Harness_Engineering_Coding_Agents.md) — the coding-agent-harness synthesis of the same design notes (no-KV-cache thought experiment, six symptoms, six per-turn decision points, visibility ladder, priced-per-rebuild routing); decision-model coverage stands, consumer map filed as riir-reflex Issue 004.
> **Addendum (2026-10-15):** **Jeva** (arXiv:2610.18773, Li & Wang, NUS; HF `LIQIIIII/Jeva-4B`, code Apache-2.0) — the first decision-native System-One model with a full published paper: coverage-factorized null head (no log-K drift, their Prop 2), permutation invariance by construction (0.00% flips, Prop 3), one candidate-conditioned decision block over a cached option K/V memory (27.98× reuse), 78.30 typed_decisions, null AUROC 98.66. §3's audited RLCD discard named its reopen trigger "Jev paper release, or any plan that trains a decision head" — **fired**: the recipe (factorized loss Eq 9, dual-shuffle teacher blending, 13.37M-param head) is now consumable; product-side distill + teacher lane live in the private tier (rethink Research 007, riir-train Issue 632).

---

## TL;DR

TypeSafe AI (ex-OpenAI RLHF founder) announced **Jev**: a model class that gives up string generation entirely and emits **type-safe structured decisions (Bernoulli/Choice/Score) with calibrated probabilities, all in one parallel pass** (70–500 ms, output tokens free), trained with a "RLCD — Reinforcement Learning for Calibrated Decisions" method. Every component has published prior art (non-autoregressive generation, schema-conditioned encoders, calibration literature, constrained decoding); the integration + positioning ("a frontier-intelligence function call") is the product. For our stack this **validates the two-system serving architecture we already ship** (fast sigmoid-gated decision heads + slow deliberation) and exposes one real gap: **our sigmoid decision scalars are boundedness-proven but never calibration-gated**. Filed as katgpt-rs Issue 810 (Platt-style calibrator primitive + gates).

**Distilled for katgpt-rs (modelless, inference-time):**
A decision head is just a readout: state latent → per-schema-slot dot-product → sigmoid → calibrated probability, computed in ONE pass (one matvec — our ternary SIMD lane does this natively). The transferable primitives are (a) **calibration as a first-class output contract** (the probability means what it says, fit from outcomes — a tiny 2-param convex refit, no base-weight mutation: track-b self-adaptive), and (b) the **negative lessons**: a closed schema that cannot abstain forces confident wrong answers (HN's sharpest critique — our `ActionBridge` already abstains below threshold), and per-answer calibration does not compose into per-decision calibration (workflow-level ECE is the metric that matters).

---

## 1. Source Core Claims (with their own caveats)

| Claim | Jev | Blog's own "Nuance" / community audit |
|---|---|---|
| Outputs | Typed `Noul`(Bernoulli)/`Choice`(≤10 native, 255 via 2-stage)/`Score` + calibrated p + confidence | "Can't hallucinate" = schema-guarantee only; still confidently wrong; **cannot abstain** (forced answer) |
| Sampling | All outputs parallel, one pass, non-autoregressive | Architecture undisclosed; HN consensus guess: small encoder + calibrated readout heads (GLiNER2-shaped); a Qwen-2.5-1B "RLCD" clone appeared on HF within hours of launch |
| Training | "RLCD" — RL for calibrated decisions | No paper, no math; **acronym collision** with RLCD = Reinforcement Learning from Contrastive Distillation (Yang et al., ICLR 2024, arXiv:2307.12950) |
| Speed/cost | 70–500 ms; $0.042/MTok in, output free | Baselines include reasoning-mode LLMs; workflow evals authored in-house; reference = mean of GPT-6-Astra + Fable 5.1 (inherits their errors) |
| Eval | "Workflow evals": fixed compute graph, score = probability agreement with big-model ensemble | Nearest published: GKD/on-policy distillation (agreement as *training* loss), lm-eval-harness (fixed harness vs ground truth). The composition appears unpublished — and weaker than a real-tool oracle |

Use cases they name: "smart if-statements" (classify/route/score/branch), map-reduce over big data, real-time control (Doom demo on structured game state, 10 qps), LLM-output verification/guardrails, agent-overseer rubrics.

## 2. Prior-Art Landscape (all components covered)

- Non-autoregressive single-pass outputs: NAT (Gu et al. 2017, arXiv:1711.02281), LLaDA (arXiv:2502.09992), Mercury (Inception Labs), parallel-generation survey arXiv:2508.08712.
- Schema-conditioned single-pass decisions: **GLiNER2** (arXiv:2507.18546) — closest shipping analog; BERT-encoder classifiers; SPENs (arXiv:1511.06350).
- Calibration: Guo et al. 2017 (temperature scaling/ECE, arXiv:1706.04599); Kadavath 2022 (arXiv:2207.05221); verbalized-confidence overconfidence (arXiv:2306.13063).
- RL-for-calibration (the real "RLCD" content): **RLCR** (Damani 2025, arXiv:2507.16806 — proper-scoring-rule RL reward), Rewarding Doubt (arXiv:2503.02623 — log-score reward), reward-calibration RLHF (ICLR 2025, arXiv:2410.09724), CAPO (ACL 2026).
- Guaranteed-valid outputs: Outlines (arXiv:2307.09702), Geng EMNLP 2023 (arXiv:2305.13971), PICARD (arXiv:2109.05093).
- Two-stage high-cardinality score-then-choose: XMC family (AttentionXML, NeurIPS 2019; survey arXiv:2302.05971); retrieve-then-rerank (arXiv:1901.04085).

Verdict: the *class* is mature; TypeSafe's defensible novelty is integration + API + (claimed) training-data flywheel, none of which is extractable math.

## 3. Path-0 Inventory + Three-Track Verdicts

| Paper component | Track | Modelless analog / extraction | Verdict |
|---|---|---|---|
| Parallel structured readout (one pass) | a | **Ships**: `pick_domain`/`VariableRankRouter` argmax-over-dot-projections; one SIMD matvec = the whole "parallel sampler" for our schemas | Covered (§4) |
| Calibrated probabilities per output | a+b | **Gap → Issue 810**: Platt/temperature refit from recorded outcomes; CLR's ECE gate is the one existing calibrator, scoped to its own verifier | **Gain — file** |
| Abstain on low confidence | a | **Ships, and Jev lacks it**: `arg_runtime/pipeline.rs` ABSTAIN + SalienceTriGate CLARIFY | Covered (selling point) |
| 2-stage score-then-choose (≤255) | a | **Ships**: `rerank()`+`RerankMode`, `argtopk_with_scratch`, `pick_domains_top_k` | Covered |
| Workflow evals (reference-agreement) | eval | We are **ahead**: riir-clippy's clippy/rustc/git/lake oracles are real-tool ground truth; Bench-706-style Brier-vs-floor precedent exists | Covered (note the idea for NPC decision-graph evals) |
| RLCD training (RL for calibration) | c | Published recipes exist as separate papers (RLCR, Rewarding Doubt, ICRL); this source carries **zero method math** to distill | Recorded, no plan — audited discard below |

**Audited discard (riir-train plan not filed):** (1) the blog publishes no RLCD objective/recipe — there is nothing to transcribe; the actionable methods are the cited published papers, which are their own distill tasks; (2) serving-envelope: a trained decision head sits outside our 20 Hz hot path, while the modelless 2-param calibrator refit sits inside it (TTPO envelope rule); (3) existing GRPO reward-shaping (`riir-train-engine` bench_251) can absorb a calibration-reward row later **if** a trained head ever enters serving. Reopen trigger: Jev paper release, or any plan that trains a decision head.
*Adversarial-panel note:* the two mandatory briefs were covered by the parallel sub-agents already run (component-agent surfaced the model-based recipes incl. RLCR; substrate-sweep surfaced the modelless gaps + closed-form/convex feasibility). Merged above; no additional agents spawned to avoid duplicating completed work.

## 4. Workspace Substrate Mapping (vocabulary-translation grep, both layers)

- **Decision heads**: katgpt-core `variable_rank_domain_expert.rs` (`pick_domain<N,A>`, `RoutingVerdict`), `cgsp/dual_pool.rs` (`route_select`), riir-ai `arg_runtime/pipeline.rs` (`select_action → (action_idx, sigmoid_confidence)` + ABSTAIN), riir-games-shared `quant_expert_route`, riir-clippy `KernelExpertRouter::build_activity`. No `DecisionSchema` type exists — selection is modelless tables + argmax + sigmoid.
- **Sigmoid confidence**: 5 affect scalars via `project_belief_to_raw_signature` (`sigmoid(dot8(belief_delta, DIR))` — valence/arousal/desperation/calm/fear); belief-decay `sigmoid(-λ·Δtick)`; curiosity/verify-budget gates. **Bounded, not calibrated**: `.proofs/RiirAiProof/Hla/Bounded.lean` proves range (0,1) — a range proof, not distributional calibration.
- **Calibration substrate**: `ConformalIntervalCalibrator` (Plan 340, default-on; CRPS/coverage/Winkler + the Report-the-Floor rule); CLR verifier **ECE gate ≤ 0.10** (measured 0.0087, Bench 284); ECE/Brier/log-loss metrics in Bench 706 (nonergodic belief; Bayesian ω beat the sigmoid floor 119×) and Bench 579. **No generic Platt/temperature-scaling calibrator primitive anywhere** — the gap Issue 810 fills.
- **Two-stage/rerank**: `katgpt-attn-match/rerank.rs` (+`ndcg_at`), `argtopk_with_scratch` (NEON+AVX2), riir-clippy `RerankMode::Structural` (default).
- **Parallel-vs-AR**: speculative decode dominant; D2F masked diffusion (the one parallel all-at-once lane); BoM parallel hypothesis sampling (deliberately not calibrated UQ).
- **Oracles/ensembles**: real-tool oracles (clippy/rustc/git/lake) + known-answer tables; **no reference-model ensemble harness** (we consider that a feature — ground truth beats reference-agreement).

## 5. Game-Context Reframe (riir-ai)

Jev's Doom demo is *structured game state in → typed decisions + confidence out at 10 qps* — literally our NPC tick: state JSON → `pick_domain`-style head → action + sigmoid confidence. The frame transfers cleanly:

1. **"Smart if-statements" = our sigmoid gates.** Zone attention, curiosity triggers, VerifyBudgetGate tiers — all fuzzy decision rules composed into the tick graph. The System One framing is a *selling-point vocabulary* for the `riir-ai/.docs/` moat book (fast system = 20 Hz latent ops; slow system = CLR/KARC/MCTS deliberation — we have both, named).
2. **The can't-abstain critique is a competitive edge for us.** `ActionBridge` abstains below confidence threshold and escalates (CLARIFY); Jev structurally cannot. An NPC decision head that must always answer produces pathological behavior on out-of-distribution state; ours degrades to deliberation instead. Worth one line in the docs book.
3. **Per-scalar calibration ≠ per-decision calibration.** Even if fear=0.8 is calibrated as a *scalar*, "NPC fled" correctness is a composite. Issue 810's gate must therefore measure **decision-level ECE/Brier** (did the chosen action match outcome), not only per-direction scalar ECE — the same composite-calibration critique HN aimed at TypeSafe's workflows.
4. **Calibrated affect scalars unlock downstream consumers**: sleep-cycle consolidation thresholds, KG-triple emission thresholds (latent-similarity gates), and the two-brain confidence decay could all consume calibrated probabilities instead of raw sigmoid outputs — with the sync boundary untouched (raw scalars stay raw; calibration is a local monotone transform, still (0,1)-bounded, still sigmoid-not-softmax).

## 6. Consumer Reframe (riir-clippy, priority #2)

- The healer's selection math (`select_best_candidate` = `W_EVO·evolution + W_RATE·reliability`, ρ hand-pinned — vocab.md §6/Le Critique lineage) has an **already-firing outcome stream** (`EvolveRecorder::record_outcome`). The Issue-810 calibrator is exactly the primitive that would data-fit those hand-pinned constants from outcomes — a follow-on consumer after the base primitive proves G1.
- "Overseer rubric layer" (HN's strongest use-case): maps to our existing `llmexec_guard`/`VerifyBudgetGate` — no new substrate needed.
- Workflow-eval idea: score-bench already beats it (real clippy oracle vs reference-agreement). Record only.

## 7. Fusion

**Paper × Plan 340 conformal floor × Bench-706 Brier-vs-floor × ActionBridge ABSTAIN → the Issue-810 calibration pair ("CalibratedSigmoidGate" here is a prose phantom — shipped as `SigmoidGateCalibrator` + `CalibratedActionBridge`, Bench 808).**
A generic katgpt-core primitive: record `(sigmoid_output, outcome)` pairs per direction; refit temperature+bias (2-param convex fit — track-b, no base-weight mutation, BLAKE3-frozen + freeze/thaw-versioned like any snapshot); emit calibrated p + keep ABSTAIN. Consumers: ActionBridge confidence, affect scalars, CLR verifier (already ECE-gated — becomes the first calibrated consumer), later riir-clippy blend weights. Ranking preserved iff temperature > 0 — gate it. GOAT gate must beat (a) uncalibrated baseline on decision-level ECE/Brier and (b) the Bench-706-style sigmoid floor, per the Report-the-Floor rule (Research 322).

## 8. Verdict

**Tiers:**

| Tier | Criteria | This paper |
|---|---|---|
| Super-GOAT | — | **No**: calibration of classifier outputs is a mature field (Platt 1999, Guo 2017); in-workspace the fusion is novel, class-level it is not. Q1 fails. |
| GOAT | — | Not directly: the source ships no measurable technique we can port (closed model, no math). |
| **Gain** | ✓ | **Actionable calibration gap + validated architecture + negative lessons (abstain, composite calibration) → files.** |
| Pass | — | — |

**One-line reasoning:** a product announcement with every component pre-published and no extractable math, but it prices a real gap in our stack (sigmoid confidence everywhere, calibrated nowhere except CLR) and hands us the eval vocabulary to gate it.

**MOAT gate:** `katgpt-rs` — base-primitive fit (sigmoid mechanics + calibration), force-multiplier ≥ 2 pillars (ActionBridge/NPC decisions, CLR, HLA scalars, later riir-clippy selection) → **open primitive, Issue 810, feature-flag + bench before any promotion**. `riir-ai` — consumer only (affect scalars + ABSTAIN already ship). `riir-train` — recorded row, no plan (audited discard above). `riir-clippy` — follow-on consumer (outcome-fitted blend weights).

**Files:** katgpt-rs `.issues/810_calibrated_sigmoid_gate_poc.md` (poc/proof task). No plan until the PoC proves the gain.

**Reopen triggers:** Jev architecture paper or independent replication with numbers (the hours-later Qwen-2.5-1B-RLCD clone is the one to watch); any plan that trains a decision head (→ RLCR arXiv:2507.16806 as the recipe source); if decision-level calibration proves GOAT, promote primitive and wire riir-clippy's `W_EVO/W_RATE` fit as consumer #2.

---

## Addendum (2026-09-16): Game-arena head-to-head — claim by claim

Question posed by owner: *can we beat it in the game arena?* Verdict: **yes in our arena, structurally; no head-on in their zero-shot-judgment niche, today.** Their unit of work is a network call (70–500 ms); ours is a tick (50 ms / 20 Hz, shared by thousands of agents). Every row below is measured on our side and quoted from their own blog on theirs.

### Scoreboard — every TypeSafe promotion vs shipped substrate

| # | TypeSafe promotes | Their number | Our shipped equivalent (measured) | Verdict |
|---|---|---|---|---|
| 1 | Speed ("40×–200× faster than LLMs") | 70–500 ms/call | 1000-NPC tick = **0.90 ms** (0.90 µs/NPC, Bench 152); functor apply **62.8 ns** (Bench 263) → **10⁵–10⁷× per decision** | **WIN** |
| 2 | Real-time apps ("100 ms speeds, UX-critical") | 70–500 ms exceeds our *entire 50 ms tick* — cannot serve one NPC per tick | plasma-tier ns–µs decisions inside the 400 µs/NPC serial ceiling (Bench 324) | **WIN** |
| 3 | Output tokens FREE | free via parallel sampling | decision = 1 action idx + 5 raw scalars; **bit-deterministic → compatible with quorum replay + anti-cheat** (a cloud model can never sit on the raw-sync path) | **WIN** |
| 4 | Cost: $0.042/MTok in | $7/hr at *10 qps, one bot* (their Doom figure) | $0 marginal, local; state never leaves the process (sovereignty/privacy). 20k q/s crowd ≈ **$14k/hr** by linear extrapolation of their own figure | **WIN** |
| 5 | Parallel sampler (all outputs, one pass) | architecture undisclosed, one pass | one SIMD matvec + argmax = one pass; batched. Parity at 1 query; decisive at 20k q/s aggregate | **SPLIT** (parity × scale-win) |
| 6 | Type-safe, "can't hallucinate" (types) | schema-guarantee, no paper | typed outputs by construction (`RoutingVerdict`, action enums) + `ConstraintPruner` + **14 Lean theorems** proving scalar bounds (`.proofs/RiirAiProof/Hla/Bounded.lean`) — we *prove* what they *assert*; they ship zero FV | **WIN** |
| 7 | Calibrated probabilities (their headline) | RLCD-trained, claimed frontier-grade | bounded ≠ calibrated; only CLR ECE-gated (0.0087 vs ≤0.10 gate). **Issue 810 filed** (Platt refit from outcomes, decision-level ECE gate) | **LOSS today** (closeable, G1–G4 defined) |
| 8 | Consistency (similar in → similar out) | claimed, no determinism guarantee | modelless heads are **bit-deterministic** — replayable, quorum-safe, regression-testable | **WIN** (determinism > consistency) |
| 9 | Doom demo | 1 bot, 10 qps, text state, occlusion-blind ("sees through walls") | 1000-NPC swarm at 20 Hz (Bench 152/263/324) + fog-of-war think-brain (`sigmoid(-λ·Δt)` stale-belief decay) + ABSTAIN → System-2 (CLR/KARC/MCTS) | **WIN** |
| 10 | Wikiracing: cardinality ≤ 255, 2-stage score-then-choose | 10 native, 255 via 2-stage, occasional slowdown | `argtopk_with_scratch` (NEON+AVX2) + `pick_domains_top_k` + `RerankMode::Structural` — same mechanism, ns-tier, no 255 ceiling | **WIN** |
| 11 | Workflows / "smart if-statements" (classify, route, score, branch) | their core use case | sigmoid gates everywhere: zone attention, curiosity, VerifyBudgetGate tiers, ActionBridge **with ABSTAIN** — a capability they structurally lack (forced answer) | **WIN** |
| 12 | "Verify everything" (judge/guardrail LLM outputs) | Jev-as-judge | CLR vote + SalienceTriGate + rubric L1/L2/L3 + `llmexec_guard` + **real-tool oracles** (clippy/rustc/compile/git) | **WIN** (ground truth beats model-judges-model) |
| 13 | Map-reduce over big data | cheap zero-shot scoring at scale | batched ternary SIMD + local serving (Metal decode parity/lead vs llama.cpp; 4090 tg128 **1.131×**, pp2048 **1.046×** row-best) | **SPLIT** (cost/privacy win; zero-shot quality loss) |
| 14 | Workflow evals (agreement with mean of Astra+Fable) | fixed graph, reference-agreement | GOAT gates **execute**: known-answer tables, real oracles, count floors, Report-the-Floor UQ rule — executable methodology > agreement-with-a-model-average | **WIN** |
| 15 | Frontier-distilled zero-shot judgment (the implicit core claim behind RLCD) | "frontier intelligence on System One tasks" | modelless heads know only what we authored or fit from runtime — narrow by design | **LOSS today** (track-c closeable, below) |

**Score: 11 WIN · 2 SPLIT · 2 LOSS.** The wins are structural (latency class, determinism, sovereignty, FV, abstain); the splits are fair; the losses share one root cause — their heads are *trained* and ours are not yet *calibrated*.

### The two losses, and how they close

- **Calibration (#7)** → Issue 810. Our simulator is already the labeled-outcome data engine RLCR-style calibration needs; the 2-param refit is track-b legal. This is weeks, not quarters, and it converts #7 from LOSS to competitive.
- **Zero-shot breadth (#15)** → track-c, if it ever matters: post-train a small decision head on our own game outcomes (riir-train GRPO reward-shaping exists; RLCR arXiv:2507.16806 is the recipe; the hours-later Qwen-2.5-1B "RLCD" HF clone is the feasibility proof). The moat is not the head — it is the outcome data + tick-scale serving, which is exactly what TypeSafe does not have.

### What they cannot enter at any price

The benchmark class "thousands of calibrated decision-makers at 20 Hz with fog-of-war, memory, and quorum-verifiable raw sync" requires: per-decision latency ≪ one tick share (µs), bit-determinism, and zero network round-trip on the sync path. Jev fails all three by construction — 70 ms > the whole 50 ms tick, no determinism guarantee, cloud-only. Their own demos (one Doom bot at 10 qps) are the correct size of their unit of work. **The game arena is not a fight they can show up to; the fight they *can* show up to — zero-shot fuzzy judgment on novel text-state tasks — is the one niche where they win today.

**Reopen fired (2026-09-19):** trycua's CUA-S1 is the "independent replication with numbers" this note's trigger named — open code + data + a supervised (no-RL) recipe reaching the Jev contract, 706k params, 99.7% vs 83.6% for hosted Jev in-domain. Architecture confirmed (options-as-queries attention head from `jevlike`); recipe extracted. Distilled in [Research 573](573_CUA_S1_Open_Jev_Recipe_Specialist_Option_Scorer.md); the healer track-c PoC it prices is riir-clippy Issue 125.**

> **PASS-Redirects (synthesis):** Sebastian Raschka ["Language Models for Text Classification: From Bag-of-Words to Jev" (magazine.sebastianraschka.com/p/classifier-history-and-jev, 2026-09-29)] — independent-press confirmation of this note's reads: his educated architecture guess (small ModernBERT-like encoder + shared scalar head) matches the HN-consensus/GLiNER2-shaped guess and 573's confirmed options-as-queries class; RLCD→RLCR (arXiv:2507.16806) is the community-standard reading; his IMDb head-to-head (Jev Choice 96.47% vs ModernBERT ~95% at similar cost/time) is the economics datapoint behind scoreboard row 13 and the "fine-tuning bar raised" positioning — scoreboard unchanged (11W/2S/2L). **Tripwire carried:** if a gradient-trained specialist's G1 cell ever reads "raw FAILS the floor" (not "refit doesn't beat raw"), the audited discard above re-arms — file the narrow riir-train calibration-aware-loss issue (CE+Brier/proper-scoring-rule, per 576's published recipe) that day.

---

## Addendum (2026-10-01): the sources read in full — the survey, the independent-evidence review, and the market event

The 2026-09-29 PASS-Redirect line above was this note's first contact with the Raschka survey. Read in full, plus the landscape behind it, four fact-blocks now on record that were not. **Sources pinned:** (S1) Raschka survey, 2026-09-29 · (S2) "Jev After Eight Days of Independent Tests" (dev.to/gde, 2026-09-24; full graded report xbill9/gemma4-dev — 14 arXiv preprints, 104 GitHub evals, 33 posts; every figure provenance-graded re-scored/checked/reported) · (S3) OpenAI DevDay 2026 recap (openai.com/index/devday-2026-recap) · (S4) arXiv:2609.28940 (Barbosa, 2026-09-24, pentest-harness decision layers, exploratory) · (S5) SSRN 7495638 + Turing Post + HN — RLCD still unpublished as of 2026-10-01.

### 1. The market event — OpenAI Decisions API (S3)

Jev-like on the OpenAI platform, limited preview at announcement: "focusing Luna's intelligence on a specific set of user-defined questions with finite pre-defined answers... classify content, route requests, or choose an agent's next action." Press reads it as a Jev-killer. **Scoreboard impact: none of the 15 rows move** — the unit of work is still a network call (rows 1/2/4 stand); row 15's zero-shot-judgment LOSS is now priced against Luna-class breadth too. What changes: `choice`/`score`/`noul` is now the market's shared wire shape across TypeSafe AND OpenAI — `decision_wire` (Plan 603 T1.2) called the class right; nominative-care caution rises (Plan 603 caveat 2). Reopen triggers unchanged: RLCD/architecture still unpublished (S5; HN: "close to the chest... we have talked about writing a paper").

### 2. The independent-evidence review (S2) — the scoreboard's external audit

The strongest third-party audit to date. What it settles:

- **Accuracy**: Jev sits level with mid-price LLMs (72.5% vs Kimi K3 74.5 / MiniMax M3 75.5 / DeepSeek V4.1 Flash 76.0 on Janardhan's six-model run) and 6.5–11.5 points behind the frontier; median 11.6 macro-F1 behind best-per-task LLM in the largest pre-registered study (arXiv 2609.24574, 7,977 human-labelled items). Scoreboard rows 13/15 unchanged in direction; the "frontier intelligence" margin language is now externally bounded.
- **banking77 — the first same-suite external cell this stack gets**: eight independent runs put Jev at **0.753–0.840, median 0.809**. Our cells: instinct H2 **0.8540** (serving) — a classifier TRAINED on the suite's own labels (nbsvm v2) — and reflex modelless A0, published **0.8260** / adopted **0.8620** (Bench 064 `bc0a3fb`, ridge@1 + head off; site republish pending). All above the entire independent Jev range — but read the comparison honestly: ours are supervised (label-trained) cells vs Jev's ZERO-SHOT runs, so this instantiates the labels-beat-zero-shot finding below rather than a like-for-like accuracy race. Protocol/item-subset overlap unverified, vendor-adjacent runs included in the eight — external context, never a published cell (the arena rule).
- **Calibration**: Jev's median ECE 0.071 as shipped (Bespoke 13-subset suite) is the best measured — but **once every model gets one fitted temperature, 15 of 19 LLMs beat it**, and no single temperature fits across domains (refit temps 0.65–4.45). API quirks recorded: probabilities rounded to 0.01 (one committed sample: 70.4% of `choice` probabilities exactly 0; `noul` clamped [0.01, 0.98]).
- **The refit-first consensus** — this note's Issue-810 adjudication, now the field's confirmed remedy: fit a temperature on 50–300 of your own labels (Jev civil_comments ECE 0.16–0.21 → <0.025; Luce ships with 100–300 real labels fitted). Reflex `--gate-fit-calibrated` (Platt, default since Issue 056) is this posture, shipped. The per-domain temperature spread (0.65–4.45) is the argument for fitting PER FAMILY — which instinct Issue 018's calibration gate already is.
- **Break modes** (seat-design-relevant): option-name sensitivity (swapping the rubric behind "no"/"yes" flipped 32.5% of answers, AUROC 0.81→0.58 — one C-grade unreplicated study); bare labels mislead (all 40 hard tasks to the cheap model at median confidence 0.96; one-line descriptions fixed 37/40); absent-from-input overconfidence (KoBBQ 0.79 confidence on by-design-wrong answers; fair-die `choice` 83% vs true 17% while `noul` read 19%); batched questions cannot see each other's answers; language shift (Russian XNLI 88.3→77.3, ECE ×3). The stack's defenses already exist and now carry external justification: instinct's C1 canonical-presentation contract + name-first noul law, reflex's byte-identical questions + first-class abstention (Jev structurally cannot abstain — the fair-die read is what a forced answer costs).

### 3. External cells for our comparison lanes + the specialist thesis (S1 + S2)

- Raschka's IMDb/Tetris head-to-heads: **CLM 82.90% IMDb + fails his Tetris generality probe; laya 92.33% + fails Tetris "even worse"; Jev 96.47% (Choice) / 96.20% (Noul) + plays Tetris; fine-tuned ModernBERT ~95%**. Laya identity settled (S1's HF link convaiinnovations/laya + S2's "Laya (Convai), ModernBERT-large 421M" + our pinned checkpoint's 421M params): the external laya IS the family riir-infer-laya pins. S2's laya row is harsher than ours: "the card's own zero-shot 0.362 sits below its 0.461 majority baseline". Consistent with our lane's read — laya is the encoder substrate (G5-parity-checked), not a generalist.
- **Labels beat zero-shot wherever tried** (S2): a 310M Japanese encoder trained on 200 rows beat Jev by 12 points (88.8 vs 76.8); Luce trained on 1,000 labels hit 97.4% vs Jev 62.6% on phishing; fine-tuned models led 2–15 points on five public splits. The instinct/Rethink thesis (A1 specialists seated where they win, A0 modelless where they don't) confirmed from outside — and the shape argument for 018's Lane C (one checkpoint serving all seated cells).
- Vendor/press numbers are never our published cells (the reflex-001 §5 rule) — every row above is external context only.
- Landscape survey (S4): Jev/Laya as decision layers for pentest harnesses; exploratory single-author, n=1 case study, explicitly not statistically significant. One reusable datum: **laya documents training against the Brier score explicitly** (a published-spec fact beside 576's recipe row). No further content for this stack (pentest domain; its "Rave" model is a proposal, not a system).

### 4. The recipe validated + the market-standard construction (S1)

- The Jev-like retrofit recipe is now common knowledge (S1 §"build your own"): ONE output node shared across candidates, candidate-conditioned inputs, softmax across per-candidate scores, backbone+head jointly fine-tuned — structurally the v2 per-option NLEH head (instinct T9/Bench 040) and 573's options-as-queries class. Nothing to adopt (his head is candidate-softmax; the house law stays sigmoid-then-L1).
- S1's honest warning is our measured posture verbatim: a Jev-like API is trivial; working well across diverse tasks (his Tetris probe) is not — exactly why the encoder cells are seated per-suite with frozen reads and record-only losses rather than a generality claim.
- The synthetic-data bet confirmed from the founder (S1 + S2/Latent Space: "100% of our data is synthetic"): the stack's analogs stand (corpus synthesis plan 426 T5 with the agreement veto; the RLCR-shaped calibration reward already in riir-train `svr.rs::r_verify` `λ_cal`).
- The 193.6×/444.6× headline claims have no named comparator; averaged over the eight workflow setups: 97.8×/149.2× (S2's check). Scoreboard rows 1/4 never depended on TypeSafe's numbers — unchanged.

### Tripwire wiring (the Issue-018 fusion) — two cases, kept separate

instinct `riir-instinct/.issues/018_rethink_encoder_lean_goat.md` Lane D now wires the standing rule with the two readings kept DISTINCT (round-1 verdict correction — the first draft merged them):

- **Δ-retention breach** (Lane D's own gate: per-family Δ(Brier)/Δ(ECE) above its pre-declared margin) is a QUANT retention failure, not a head defect — in D1 the heads are the existing F16-trained ones and only the encoder is fake-quantized, so a Δ breach is quantization noise. Response: the per-family temperature refit may run first only if PRE-REGISTERED in the margin declaration (argmax-preserving, Issue-810 track-b; S2's external warrant, refit temps 0.65–4.45 → per-family fitting); unrefit-able ⇒ a D1/D2 calibration FAIL through the EXISTING branch (D1-fail → C on F16 → E distill). No new trigger beside D1's verdict.
- **Absolute-floor failure** (a G1 raw-FAILS-the-floor read against the conformal-naive floor — the 562/576 tripwire verbatim) on ANY gradient-trained head, F16 or quantized, is the calibration-aware-loss trigger, scoped by where the fix can live: for the head Lane C retrains, CE+λ·Brier is C's pre-declared loss option (no separate filing — C retrains it anyway; Raschka's supervised form per 576's PASS-Redirect line — "modest gain, may not generalize"; the RL-family scoring-rule recipes are 576 §3's log/spherical/RPS, the loss form chosen from that family at filing time); for any OTHER seated head, the tripwire applies verbatim — the narrow riir-train calibration-aware-loss issue files THAT DAY for that head's own retrain. The reward-shaped twin already ships in `svr.rs::r_verify` `λ_cal`.

The refit-first posture is the INDEPENDENT consensus (S2); until a raw-FAILS reading exists the CE+λ·Brier option stays undeclared and C trains plain CE. This addendum records the wiring, not a firing.
