---
name: research
description: Research workflow for distilling ML/AI papers into modelless inference primitives, freeze/thaw runtime patterns, latent-space operations, AND model-based training plans across the multi-repo stack. Use when reading arxiv papers, deciding which repo a paper belongs in, creating .research/ notes or .plans/ files, implementing modelless inference primitives, or routing training-vs-inference insights. Enforces the commercial strategy (public engine / private runtime / private chain / private neuron-db / private training / private SDK facade / private product-domain), three-track system (modelless inference + self-adaptive runtime + model-based trained weights — all three exist across the stack, not just riir-train), latent-to-latent preference, and freeze/thaw-over-fine-tuning rule.
---

# Research Workflow — Modelless Inference, Freeze/Thaw, Latent-to-Latent

This repo (`katgpt-rs`) + `riir-ai` (freeze/thaw runtime + adaptive NPCs + game systems) + `riir-chain` (neuro-symbolic chain transport, LatCal) + `riir-neuron-db` (NeuronShard, BLAKE3/Merkle, freeze/thaw envelope) ship **runtime + latent-space operations**. Training-method research lives in `riir-train`. If a paper's value is its training loop → `riir-train` (see §3.5 Path 0.5 — applicable training papers get a Plan, not a lazy redirect). If its value is a latent-space insight, a routing trick, a freeze/thaw pattern, a chain-commitment bridge, a neuron-shard primitive, or a modelless inference primitive → distill here.

## When to use

Reading/fetching/summarizing ML/AI/systems papers · deciding which repo a paper belongs to · creating `.research/` notes or `.plans/` files · implementing modelless inference primitives · designing freeze/thaw cycles, adapter hot-swap, runtime adapter routing · designing latent-to-latent ops (dot-product projections, sigmoid gating, manifold geometry, spectral methods) · designing MMORPG-scale game AI (thousands of concurrent NPCs, 20Hz tick, fog-of-war, emergent social/economic behavior).

Do NOT activate for: pure refactor, bug fixes with no research angle, or ordinary feature work.

## Repos

- `katgpt-rs/` — public MIT engine. Generic modelless inference primitives. **No game/chain/shard IP.**
- `riir-ai/` — private game product. Freeze/thaw runtime, self-learn, game systems. Hosts the `.docs/` moat book.
- `riir-chain/` — private neuro-symbolic chain transport. LatCal, `riir-chaind`, economics, asset lifecycle, `catchup/` (Turso/libSQL, quorum). **Re-exports `riir-neuron-db` under `neuron_db` feature; canonical shard source is `riir-neuron-db/`.**
- `riir-neuron-db/` — private leaf. `NeuronShard` (Pod, zero-copy mmap), `ShardIndex` (lock-free papaya), generic `MerkleTree`/`MerkleProof`, `MerkleFrozenEnvelope`, MAPE-K, Raven/δ-Mem consolidation, AnyRAG gateway, vibe KG triples, spectral init, `ShardCompactor`, dendritic LoRA branch. **No chain dep — usable standalone.**
- `riir-train/` — private training vault. **As of 2026-08-06: actively pursued, not lazily redirected.** Applicable training papers get a Plan in `riir-train/.plans/` per §3.5 Path 0.5. `read_file riir-train/.docs/02_pipelines/training_data_pipeline.md` before any training-paper verdict.
- `riir-game-sdk/` — downstream consumer; rarely a distillation target.
  (`riir-armageddon/` sat here until it was retired 2026-09-02, owner act —
  the directory is gone; do not route to it.)

**Routing rule of thumb:** if it's about *how a shard is structured/committed/frozen/consolidated/retrieved/projected* → `riir-neuron-db`. If it's about *how a shard crosses quorum or bridges to LatCal fixed-point* → `riir-chain`.

## Commercial strategy (inline short version)

| Tier | Where | Role |
|---|---|---|
| **0 — Substrate** | `katgpt-core` (leaf, crates.io) | Pure inference mechanics (SIMD, transformer/weights, `hla`, `dd_tree`, `mcts`, `sampling`, `delta_mem`). Leaf-clean. |
| **1 — Engine** | `katgpt-rs` (public) | Re-exports substrate + BASIC cognitive primitives + toy games (each ships WITH its `.md`). |
| **2 — GOAT + IP** | `riir-ai` / `riir-chain` / `riir-neuron-db` / `riir-train` (private) | GOAT/Super-GOAT tuned versions, `*_runtime` composition layers, IP. |

**What = public. How = private.** Training how → `riir-train`. Runtime how → `riir-ai`. Chain how → `riir-chain`. Shard how → `riir-neuron-db`. When unsure → default private (safe to keep private; never safe to un-leak public). Toy 2D games (`bomber`, `monopoly`, `go`, `fft`) are NOT product IP — public. `*_runtime` suffix = private GOAT composition layer; bare-name = public primitive. FV moat: ~79 Lean 4 theorems across 4 `.proofs/` instances.

## Pre-flight — MANDATORY before any verdict/file

Do all five before creating any file:

1. **`read_file` 5 READMEs + `riir-ai/.docs/README.md` + (for training papers) `riir-train/.docs/02_pipelines/training_data_pipeline.md`** — defines scope boundaries. Skipping = #1 cause of false Super-GOAT claims + false PASS on training papers.
2. **`list_directory` all 5 `.research/` folders** (katgpt-rs, riir-ai, riir-chain, riir-neuron-db, riir-train — create missing ones on first use).
3. **`list_directory` the 4 runtime/chain/db src trees** — module names are vocab. Skipping = #2 cause of false Super-GOATs.
4. **`web_search` for published prior art** on the paper's headline technique (see §4). Skipping = #4 false-novelty failure mode.
5. **`grep` ALL repos for existing training/model-based/self-adaptive code** — the system is three-track, not modelless-only. Skipping = #5 false-PASS mode (canonical failure: arXiv:2511.18538 Code LLM survey was falsely PASSed because the agent only checked riir-train for training code, missing quest_grammar's LoRA training in riir-ai + TernaryDraftModel in riir-clippy + self_evolve). Grep patterns: `*_training.rs|*_train*.rs|LoRA|SFT|GRPO|DPO|ternary|TernaryDraftModel|self_evolve|with_weights|\.bits` across `riir-ai/crates/**/src/`, `riir-clippy/src/`, `riir-neuron-db/src/`. Training pipelines ship OUTSIDE riir-train — the three tracks are: (a) modelless inference, (b) self-adaptive runtime latent updates, (c) model-based trained weights.

**`list_directory` the src trees** (vocab; skipping causes the vocabulary-miss class of false verdicts — mechanisms ship under non-obvious names):
- `riir-ai/crates/riir-engine/src/` (`latent_functor/`, `cgsp_runtime/`, `hla/`, ...), `riir-ai/crates/riir-games/src/`
- `riir-chain/src/` (`encoding/`, `consensus/`, `economics/`, `asset_lifecycle/`, `forensic/`, `catchup/`)
- `riir-neuron-db/src/` (`shard.rs`, `index.rs`, `merkle.rs`, `freeze.rs`, `mape_k.rs`, `consolidation.rs`, `gateway.rs`, `vibe.rs`, `spectral_flatness.rs`, `shard_compactor.rs`)
- `riir-train/crates/riir-train-{gpu,engine}/src/` (`distill_attention.rs`, `loss_grpo.rs`, `loss_dpo.rs`, `delta_filter.rs`)

Do NOT create any file until all five done.

## Primary focus — Fusion-first

**The highest-value Super-GOATs come from fusing 2–3 papers/primitives into a novel combination, not from direct-mapping a single paper.** Always grep `.research/` + `.plans/` for the 2–3 closest cousins before verdict, and ask: *"what does paper × note A × note B produce that none of them alone can?"*

### Fusion priority ladder (owner-set investment order, 2026-09-02)

The repos are NOT equal-weight fusion surfaces. When a paper fuses into several, evaluate and file in THIS order — the higher-priority surface wins plan/issue filing priority for the same primitive:

1. **Game / MMORPG runtime** (`riir-ai` + `riir-game-sdk` + product repos) — the #1 investment.
2. **riir-clippy healer** (code/perf/sec fixing: corpus, retrieval, fix trajectories, score-bench, rustc_errors playbooks, domain router) — the #2 investment and the workspace's densest latent-state consumer per LOC.
3. **Inference-perf league** (beat llama.cpp / oMLX / vLLM — the riir-gpu / riir-ai serving path).
4. **riir-chain** (consensus / LatCal / settlement).
5. **Training / fine-tuning / LoRA** (`riir-train` + in-repo pipelines; recipes and adapters — full pretraining is out of scope).

The ladder orders EFFORT, not existence: lower-priority fusions still get recorded (note + issue), but a higher-priority surface carrying the same mechanism gets the plan first. The standing per-verdict check this ladder encodes: **"did I ask what this does for the healer?"** — canonical failure #5 (below) is what happens when that question is never asked.

Patterns that ship here:
- **Latent-to-latent operations** — dot-product projections, cosine retrieval, sigmoid-gated routing, manifold geometry, spectral methods on activations. Prefer operating on latents over decode→re-encode. Fuse with freeze/thaw to version direction vectors; with self-learn to update them from runtime curiosity.
- **Freeze/thaw patterns** — versioned snapshots, atomic hot-swap, lock-free reads, BLAKE3-checked reload, per-entity personality divergence. Fuse with adapter routing to dispatch by latent similarity; with self-learn to checkpoint emergent personalities.
- **Runtime adapter routing** — selecting between frozen adapters by state/objective/context (Dynamic Pair, Polytope, dMoE — inference-time, zero training). Fuse with freeze/thaw to make the pool versioned + committed; with bandits to learn routing online.
- **Self-learn / adaptive CoT** — runtime curiosity, entropy exploration, collapse detection/recovery, latent prediction SSL, trajectory folding. No backprop. Fuse with MMORPG-scale game AI for thousands of NPCs each with independent curiosity; with freeze/thaw to checkpoint.
- **Modelless inference primitives** — ConstraintPruners, bandits, DDTree, speculative decode, sparse attention, quant-aware inference.
- **MMORPG-scale game AI** — thousands of concurrent NPCs, 20Hz tick, fog-of-war, zone attention, emergent social/economic behavior. Latent ops must batch; raw sync must stay bit-identical.

### Super-GOAT factory modules — grep FIRST

The highest-value latent Super-GOATs cluster in seven module trees. `list_directory` these explicitly even if the paper looks pure-training — vocab mismatch is the #3 cause of false verdicts:

| Module | What ships | Super-GOAT angle |
|---|---|---|
| `katgpt-rs/crates/katgpt-core/src/sense/` | belief-state kernels (`evolve_belief`, `SenseModule::project`, ternary bit-plane projection) | Per-NPC recurrent latent state — runtime substrate for any "hidden state"/"belief"/"activation" paper |
| `riir-ai/crates/riir-engine/src/latent_functor/` | `zone_gating.rs`, `reestimation.rs`, `arithmetic.rs`, `cross_game.rs`, `k_selector.rs`, `quality_gate.rs` | **Game-theory in latent space** — vector-op functors, coherence-driven re-estimation, zone-gated activation. Maps "stage"/"application"/"bypass"/"collapse" papers |
| `riir-ai/crates/riir-engine/src/hla/` | `kernel.rs`, `forward.rs`, `types.rs` — **Higher-order Linear Attention** (Transformer attention-layer replacement). Paper: Zhang et al. 2026. **NOT the per-NPC belief** (different layer, different repo). | Maps "attention layer"/"linear attention"/"recurrent state" papers to Transformer-scale ops |
| `riir-ai/crates/riir-engine/src/cgsp_runtime/` | Curiosity-guided self-play, latent prediction SSL, MCTS collapse bridge | Runtime curiosity/exploration — maps "self-learn"/"entropy-driven"/"collapse recovery" |
| `riir-neuron-db/src/` | `shard.rs` (`NeuronShard` Pod, `style_weights[64]`, dendritic branch), `freeze.rs` (`MerkleFrozenEnvelope`), `consolidation.rs` (Raven/δ-Mem), `gateway.rs` (AnyRAG), `vibe.rs` (KG triples), `merkle.rs`, `mape_k.rs`, `spectral_flatness.rs`, `shard_compactor.rs` | **Frozen latent-state storage + integrity + retrieval** — maps "memory"/"replay buffer"/"experience replay"/"spectral init"/"Merkle commitment"/"snapshot"/"KG triple" |
| `riir-chain/src/encoding/latcal*.rs` + `latcal_fixed.rs` | Lattice Calculus: 2×2 matrix arithmetic obfuscation, fixed-point bridge, spectral fixed-point, DeFi programs | **The sync-boundary bridge** — deterministic, committed, raw-numeric. Maps "fixed-point"/"deterministic commitment"/"raw↔latent bridge"/"arithmetic obfuscation" |
| `katgpt-rs/crates/katgpt-dec/src/` | `operators.rs` (`exterior_derivative` d, `codifferential` δ, `hodge_laplacian` Δ), `hodge.rs` (`hodge_decompose`, `betti_numbers`, `harmonic_projector`), `flow.rs` (`DecFlowField`), `stokes_calculus.rs` (`boundary_flux_mass`, `belief_mass_divergence`, `line_integral`) | **Generalized Stokes' Theorem substrate** — `d∘d=0` by construction. Maps "divergence"/"boundary flux"/"line integral"/"curl"/"Hodge"/"Fokker-Planck"/"mass conservation"/"manifold geometry"/"exterior calculus"/"Stokes". **Curse-of-dim caveat: boundary-vs-volume wins only for d≤3 (maps, belief regions, KG embeddings) — NOT high-dim shards.** |
| `riir-clippy/src/draft/` | span-level retrieval under `latent_retrieval`: `AstChunker` → `ModellessEmbedder` → fan-out `RuleIndex` (riir-rag: Clifford-wedge KNN + BM25) + `RerankMode::Structural` + oracle-labeled eval fixtures (`retrieval_eval.rs`, `eval_fixtures_composite.rs`, `clippy_oracle.rs`) | **A labeled (rule ↔ code-shape) binding corpus with measurable downstream accuracy** — maps any "structured index / retrieval / compositionality / binding" paper; the withheld-pair OOD generalization axis is unmeasured almost everywhere |
| `riir-clippy/src/self_evolve.rs` + `src/traj_store.rs` + `src/elo.rs` | fix-trajectory memory (Warm-tier `.heal/` store, evidence tiers Certified/Heuristic/Withdrawn, Elo + BetaPosterior selection, `EvolveRecorder`) | Maps "experience replay / memory / trajectory store / bandit-evidence" papers — outcomes are labeled and persisted |
| `riir-clippy/src/domains/` + `corpus_gen` + `rustc_errors` playbooks | 580+ rule corpora across six domains; error-code-keyed repair playbooks; per-domain centroids (`KernelExpertRouter::pick_domain`) | Domain/category keys ARE roles — maps "role-conditioned routing / expert routing / structured generation" papers |

**Adapter routing, KV compression, speculative decode = GOAT-tier framings. Latent-to-latent ops on belief/functor/shard/LatCal state = Super-GOAT-tier framings. Attempt Super-GOAT first.** Defaulting to adapter routing when a latent reframe is stronger is the primary failure mode.

**Consumer-surface miss (canonical failure #5 — arXiv:2608.29530 TPR, 2026-09-02):** the paper's mechanism (role-filler binding, `E = W(Σ fᵢ⊗rᵢ)+b`) mapped onto the healer corpus (rule = role, code-shape = filler, oracle ground truth, measurable downstream top-1) MORE directly than onto any game surface — but the fusion hunt enumerated only game kernels: the factory table had no clippy rows, the MOAT table no clippy domain, and the repo-wide grep stopped at page 1 of matches, so the riir-rag/riir-clippy wedge-consumption evidence was never read. The clippy fusion (riir-clippy Issue 62: withheld-pair OOD bench + structured retrieval axis) surfaced only by user challenge — **5th rescue** of the challenge-rescue class after NQF, LOPD, Le Critique, TTPO. Fixes applied: the priority ladder above, the clippy factory rows above, the step-4 consumer reframe, the MOAT clippy row, and the step-1 pagination rule.

## Redirect to riir-train

**Pre-check before ANY redirect:** exhaust §3.5 modelless unblock paths first. A mechanism that *looks* training-only may be modelless-validable because its training-target MATH decomposes into shipped primitives (Flow Sampling is the canonical case: "trains a drift network via backprop" decomposed into dllm interpolant + Latent Field Steering reward-gradient + freeze/thaw replay buffer).

**As of 2026-08-06:** training is actively pursued, not lazily redirected. Applicable training papers (optimizer, loss, schedule, recipe, LoRA/OFT/IA3/QLoRA/DPO/GRPO/SFT/RL) get a **Plan in `riir-train/.plans/`** with recipe + GPU-hours estimate + GOAT gate comparing trained-vs-modelless baseline. Only genuinely out-of-scope training (image-specific DiT we'll never train, medical imaging) gets the one-line redirect with explicit justification.

Runtime GRPO self-play stays in `riir-ai` (self-adaptive track — updates latent state, not weights). Model-based training (LoRA/SFT/GRPO on actual weights) lives in BOTH `riir-train` AND in-repo training pipelines (`quest_grammar/grammar_training.rs` in riir-ai, `TernaryDraftModel` `.bits` files in riir-ai/riir-clippy). Quant-aware **inference** stays here; quant-aware **training** → riir-train. **Never assume a repo is modelless-only** — grep for existing training/model-based code first (pre-flight #5).

## Workflow

### 0. Read & classify

Fetch via `https://r.jina.ai/https://arxiv.org/pdf/{ID}`. Ask: *is the value in the training loop itself (optimizer/loss/schedule/RL), or in the math the training computes (closed-form drift, conditional score, Riemannian correction, steering formula)?* Optimizer/loss/schedule → riir-train. **Math → run §3.5 Path 0 first.**

**Label-anchoring hazard (canonical failure #2 — arXiv:2608.13335 NQF, 2026-08-18):** the paper's own vocabulary ("Neural", "trained by gradient descent", "learning dynamics") is a routing hazard. Training-dynamics papers routinely carry closed-form modelless content — ignition-time formulas, closed-form trajectory families, initial-behavior limits, offline-computable spectra. NQF was routed wholesale to riir-train although its theorems carried a sigmoid-in-time adoption family, a `t* = ln(1/ε)/ζ` patience law, and a fresh-module leading-behavior predictor; surfaced only by user challenge (same rescue class as the LOPD canonical failure). Classification comes AFTER the §3.5 two-question decomposition + adversarial panel — never from the abstract's framing.

**Test-time split hazard (canonical failure #4 — arXiv:2608.27448 TTPO, 2026-08-29):** a "test-time training" paper carries TWO separable claims — the gradient objective (training track) AND the runtime decision layer the objective is built on (vote partition, asymmetric agree/penalize treatment, confidence scoring). TTPO's abstract framing routed the gradient objective to a riir-train plan filed as PRIMARY while the modelless decision layer — the half that fits the serving envelope — was downgraded to "recorded"; rescued by user challenge (4th rescue of this class after NQF, LOPD, Le Critique — user challenge is load-bearing, treat every challenge as a granularity audit). Fix: per-track verdicts (§1.5), discard-reason scrutiny (§3.5), serving-envelope fit (§3.5).

**Three-track system — do NOT classify a repo as "modelless only" without checking pre-flight #5.** The system runs three concurrent tracks: (a) **modelless inference** (primary hot path — no weights, BLAKE3-deterministic), (b) **self-adaptive** (runtime latent updates — `self_evolve`, EMA direction vectors, freeze/thaw cycles), (c) **model-based** (trained weights — ternary `.bits` files, LoRA/QLoRA adapters, Bonsai model comparison). A paper touching ANY track is actionable. The model-based track exists in repos OUTSIDE riir-train: `quest_grammar/grammar_training.rs` + `quest_training.rs` (LoRA rank=16-32, alpha=32-64, QLoRA 4-bit) in riir-ai; `TernaryDraftModel` with `.bits` weight files in riir-ai/riir-clippy; `self_evolve` feature in riir-clippy (shipped, G1-G7 ALL PASS). Before classifying a paper as "training-only, deferred" → grep for existing training code in the target repo (pre-flight #5).

**Substrate ≠ value.** Hardware/NMP/PIM/ASIC papers: the value is usually the *technique* (LUT INT→FP, shared ALU, sideband-tag) stripped of the hardware — grep `simd_*`, `ternary`, `Plasma`, `Q4_K`, `LUT`, `from_bits` for the software analog. Database/systems papers: `riir-neuron-db` IS a database (Pod + ShardIndex + Merkle + MAPE-K + Raven/δ-Mem + vibe KG) — the value is usually the *access pattern*. OS/kernel papers (io_uring, DPDK): substrate is Linux, value is usually the technique (lock-free queue, batching, zero-copy). Pure math/combinatorics: value may be a guaranteed-peak property on a per-entity scalar — see §1 step 4 game-context reframe.

Before PASS on hardware-vocab papers: (1) identify technique stripped of substrate, (2) grep for software-SIMD analog, (3) PASS only if both confirm no analog.

**External code repo (not a paper)?** Skip jina — §0.5 clones it into `.raw/` for full-tree grep.

### 0.5 External-repo source access — clone into `.raw/` (ephemeral, `rm` when done)

Papers distill fine over the wire (`https://r.jina.ai/...`). **External CODE repos do not** — distill verdicts, mining batches, and prior-art verification need full-tree `grep`, byte-accurate quotes, and multi-file reads that URL fetches cannot provide. When a research task needs an external repo's source:

```bash
# Clone (shallow by default; full history ONLY if the task mines git log)
git clone --depth 1 https://github.com/<org>/<repo>.git .raw/<repo>
# Pin provenance BEFORE reading anything into a verdict:
git -C .raw/<repo> rev-parse HEAD   # record this sha + the license in the note
```

Rules (all mandatory):

1. **Location:** `.raw/` at the root of the repo whose `.research/`/`.plans/` will hold the output (e.g. `katgpt-rs/.raw/ds4/`). `.raw` is ephemeral scratch — gitignored in every workspace repo, never committed, never a sibling repo.
2. **Pin before verdict:** record `@ <full-sha>` + license (Apache-2.0/MIT/...) in the research note BEFORE deleting the clone — the sha is the provenance; the clone is throwaway. Every quote later cited must match the pinned sha (precedent: riir-clippy Batches 96–98 distills cite `antirez/ds4 @ 110afdd8…` with quotes "re-verified at mine time").
3. **Read-only.** Do not modify files inside `.raw`. If a patch experiment is needed, copy the file out first; never let experiments dirty the clone mid-verdict.
4. **No build-graph contamination:** never add a Cargo/path dep into `.raw` (BOUNDARY.md + standalone-dep gates), and never include `.raw` in codebase/pre-flight grep scopes — it is NOT shipped substrate, so grepping it inflates hit counts and makes a mining candidate read as "already ships".
5. **Cleanup is part of the task:** when the verdict/note/plan is committed, `rm -rf .raw/<repo>` (or the whole `.raw/`). A finished research task with a live `.raw/` entry is an unfinished task. If a session dies mid-task, treat any stale `.raw/` entry as UNTRUSTED on resume — delete and re-clone at the pinned sha; never verdict against a clone whose commit you did not pin yourself.
6. **Shallow unless history-mining:** `--depth 1` by default; full clone only when the task needs `git log`/diff archaeology (fix-pair mining, regression hunts) — and say so in the note. For very large repos where only a few files are read, `--filter=blob:none` fetches blobs on demand.

### 1. Distill fundamentally — fuse, don't direct-map

Find the transferable primitive (the geometric/spectral/information-theoretic insight that works without the paper's training setup). **Then look for fusion opportunities**: cross-pollinate with existing notes/plans/shipped primitives to synthesize something novel.

**Fusion protocol:**

1. **Grep ALL SEVEN repos, BOTH layers (notes AND code), in parallel via subagents. Do NOT stop after the first repo or layer, do NOT wait for user prompts to grep the next repo. Grep results are PAGINATED — a full first page is NOT a finished sweep: run `offset` until exhausted, or scope the grep per-repo so every repo's hits are visible.** Closest cousin is frequently in the OTHER repo (cross-repo fusion) or in CODE not notes (mechanisms ship without research notes — `evolve_belief` is a per-NPC recurrent belief-state kernel with no `.research/` framing). Grep:
   - All 7 `.research/` + `.plans/` folders (riir-train included — applicable training papers get Plans per §3.5 Path 0.5)
   - `riir-ai/.docs/` (the moat/selling-point book — grep alongside `.research/` so you don't claim novelty over a pillar that ships)
   - All shipped `src/`/`crates/` trees (notes describe intent; code describes what shipped)
   - The seven Super-GOAT factory modules above

2. **Vocabulary translation BEFORE grepping.** Papers and codebase use different words. List the paper's 3–5 key terms; for EACH, brainstorm ≥2 codebase equivalents ("if we shipped this, what would we call it?"). Grep BOTH sets. `read_file` `vocab.md` (sibling file) for 7 standing vocabulary tables + worked examples. **Coefficient-shape mechanisms** (adaptive blend / mixture coefficient / interpolation ratio / ensemble weight): shipped blends hide as hand-tuned constants (`W_EVO = 0.6`, `alpha: f32`, `trust: f32`), never under the word "blend" — use vocab.md §6's grep set and run it across ALL repos, never scoped to the expected home (canonical failure #3).

3. **Latent-space reframing BEFORE verdict.** Re-cast the core mechanism as a latent-to-latent op on the codebase's kernels: (a) per-NPC belief, (b) `latent_functor/` ops, (c) `cgsp_runtime/` curiosity, (d) LatCal fixed-point commitment, (e) `NeuronShard` style_weights/dendritic/`MerkleFrozenEnvelope`/Raven/AnyRAG, (f) DEC Stokes operators (d, δ, `hodge_decompose`, `DecFlowField`). If you only reach adapter routing/KV/spec-decode framing → likely GOAT, missed the Super-GOAT angle.

4. **Game-context AND consumer-context reframing BEFORE verdict** (especially when step 3 returns no hits). Game question: *how does this mechanism manifest as a per-NPC behavior signal / crowd pattern / selling point in MMORPG context?* A guarantee → what per-entity scalar does it bound, can that drive behavior (salience cadence, curiosity trigger, consolidation window)? A combinatorial structure → does it appear in NPC routines, market cycles, quest scheduling? A number-theoretic property → fairness/diversity/coverage on a game signal? A geometric property → NPC phase scheduling, spatial spread, fog-of-war coverage? **Consumer question (priority #2, same evidence bar): does the mechanism manifest on a healer surface — (a) the retrieval index / corpus structure, (b) fix-trajectory memory / selection, (c) the score-bench / eval axis, (d) drafter-pruner selection, (e) rustc_errors playbooks?** One sentence each; a bare "no" without reading the surface is the canonical-failure-#5 shape. **If either reframe is missing, you are NOT ready to PASS.**

5. **Zero hits ≠ novelty.** Most likely you're using the wrong vocab — try a third semantic angle (grep *output behavior* "swap when X" instead of *mechanism name* "tightness monitor") before claiming "no prior art".

6. List the 2–3 closest cousins across all repos. Ask: *what novel combination of paper × A × B produces a capability none has alone?* Write that into the note's §Distillation as a **Fusion** subsection even if unplanned.

7. Verdict by tiers (§1.5). Create research `.md` at the right repo. Naming: `{NNN}_{Short_Title}.md` (next free number, monotonic, never reused). Format: `read_file` `templates.md` (sibling) — canonical example: `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md`.

### 1.5. Novelty gate — is this Super-GOAT?

Score all four:

1. **No prior art?** Grep notes + plans + shipped code across all repos, using BOTH paper vocab AND codebase-vocab alternatives. **Read every grep hit's TL;DR before claiming novelty** — a filename match is a lead, not confirmation. Symmetric rule for the opposite direction: a cousin NAME match is a lead toward coverage, never coverage itself — "already ships" claims require the §3.6 signal-diff check. Mechanisms ship under non-obvious names (DiPOD's "interleave self-distillation when ELBO drifts" ships as "coherence-driven re-estimation scheduler when coherence < tau_reest"). When the candidate selling point touches per-NPC + memory + personality + swap, the `riir-ai/.research/` corpus is saturated — grep it broadly and READ every hit before claiming novelty.
2. **New behavior class?** Not better numbers — a capability no incumbent has. **Requires step 4 game-context reframe to detect.** If you reached Q2 without step 4, go back.
3. **Product selling point?** Can you finish: *"Our NPCs/systems do X that no competitor can"*? If not → Gain.
4. **Force multiplier?** Connects to ≥2 existing pillars/systems. Solo novelty = GOAT, not Super-GOAT.

**All 4 YES → Super-GOAT.** Mandatory outputs (in this same session):
- **Open primitive** → `katgpt-rs` (generic math, no game semantics).
- **Architectural GUIDE** → private selling-point doc. Repo by selling-point domain: `riir-ai/.research/` for game-runtime/belief/functor/self-learn; `riir-chain/.research/` for chain/LatCal/commitment/sync-bridge; `riir-neuron-db/.research/` for shard/freeze/consolidation/AnyRAG/vibe/Merkle. Cross-domain selling points → primary guide in the repo owning the boundary being crossed. Guide includes: TL;DR with commercial value, distilled modelless primitive, connection map, latent↔raw boundary, what's private vs open, validation protocol, P0–P3 priority.
- **Plan(s)** → appropriate `.plans/` folders.

> **No "candidate" escape hatch.** Writing "all 4 YES" / "passes the gate" / "Super-GOAT candidate" anywhere in a note triggers the mandatory outputs in THIS session. If not confident on all 4 → write "fusion idea, novelty TBD" + create `.issues/` entry. "Candidate" is not a deferred-commitment escape — it either triggers the guide now, or downgrades to an issue.

**One verdict per track, not one per paper (TTPO lesson, 2026-08-29).** When a paper yields BOTH a training recipe AND modelless extractions (the Path 0 inventory rows), score this gate SEPARATELY per track. A training-track plan and a modelless-track plan are two claims with independent tiers — a Q1 kill on one does NOT cascade to the other. "Recorded in the note" is NOT a tier (same violation class as the banned "candidate" escape hatch): every Path-0 inventory row ends in exactly one of (a) filed plan (GOAT+), (b) `.issues/` entry (fusion idea / novelty TBD), or (c) an audited discard whose reason cites the mechanism-level §3.6 diff — see the discard-scrutiny rule in §3.5.

**Pin the claim BEFORE searching it (§4 precondition — TTPO lesson, 2026-08-29).** Prior-art search 3 below is only meaningful against a pinned claim. Before running ANY §4 search, write the claim in one sentence: "<mechanism> for <surface/consumer>, consuming <signal>, distinguished from <closest cousin> by <delta>." A novelty check against an unpinned claim is void — class-level prior art (JitRL/OptPO killed "training-free test-time policy optimization" the CLASS, not the asymmetric-partition + calibration-law MECHANISM) will silently kill anything, and the verdict regresses to vibes. If you cannot write the sentence, you have not distilled enough to claim novelty either way.

### 1.55. PASS vs Gain — no middle tier

PASS = no new research/plan files. Gain = files. Scan the paper for actionable improvements before verdict.

- Ships + actionable improvements → **Gain** (create `.issues/` per AGENTS.md "Create issue at .issues for poc, proof, optimization or refactor task, do not create plan", and/or plan behind feature flag).
- Ships + no actionable → **Pass** (verdict + one-line reason + closest shipped cousin, in conversation only; no files).
- Doesn't ship + modelless → **Gain** or higher.
- Doesn't ship + training-only → run §3.5 Path 0.5. Applicable → Plan in `riir-train/.plans/`. Out of scope → one-line redirect with explicit justification.

**PASS-Redirects line (mandatory):** PASS verdicts still update the 1–3 closest shipped cousin `.research/` notes with a one-line reference. Format:
```
> **PASS-Redirects (synthesis):** <Author> [arXiv:XXXX.XXXXX "<Full Title>"] — <one-line reason>.
```
Must include arxiv ID AND full title (so `grep arxiv:ID` AND `grep "Title"` hit). **Prevents paper-number invisibility** — without this, a future session greps the arxiv ID, finds nothing, re-distills from scratch.

**Actionable = Gain.** Actionable: paper data contradicts a current config default; exposes a failure mode with no existing mitigation; unblocks a deferred task. NOT actionable: "validates our design" / "theoretical lens" / "could inform a future config". Unsure → not actionable → Pass.

**Reverse-grep before PASS (mandatory).** Before any PASS, grep the codebase for documented gaps the paper could fill: `.docs/` for `Limitation|deferred|TODO|FIXME|gap|pending`; `.benchmarks/` for `Caveat|deferred|artifact`; `.rs` comments for `TODO|FIXME|deferred|limitation` near the paper's vocab. If ANY hit maps → Gain, not Pass. *Compact heuristic: "Is there any documented limitation this paper could fix?"* — if you can't answer "no, I checked" with evidence, don't PASS.

**Training papers third defense:** both defenses above can return clean and still produce a false PASS if "model-based track" is narrowed to one pipeline or one repo. Before PASS-ing any training paper: (1) `read_file riir-train/.docs/02_pipelines/training_data_pipeline.md` and ask: does a training pipeline for this domain already exist? (RL, distillation, linear attention, trajectory collection). (2) **`grep` ALL repos for existing training/model-based code** (pre-flight #5) — training pipelines ship OUTSIDE riir-train: `quest_grammar/grammar_training.rs` + `quest_training.rs` (LoRA training in riir-ai), `TernaryDraftModel` (ternary weights in riir-ai/riir-clippy), `self_evolve` (self-adaptive in riir-clippy). If ANY existing training/model-based code maps to the paper's domain → Gain, not Pass. Canonical failure (arXiv:2511.18538): Code LLM survey falsely PASSed because agent checked only riir-train (game AI pipeline), missing quest_grammar's code-domain LoRA training + riir-clippy's ternary drafter + Issue 010 (generalization gap that RLVR would fix).

### 1.6. MOAT gate per domain + promote/demote

Tier verdict measures *how strong*. MOAT gate measures *whether it strengthens THIS repo's moat*. Mismatch → reroute.

| Domain | MOAT bar | In scope | Out of scope |
|---|---|---|---|
| `katgpt-rs` | Fundamental/principle/base primitive via fusion; promote/demote tracked per stack | Transformer stack (layers/attn/KV/sampling/sparse/quant-aware **inference**/spec decode/DDTree/MCTS/bandits/pruners); 2D toy games; DEC/Stokes; belief kernel; sigmoid mechanics | Product game wiring → riir-ai; chain → riir-chain; shards → riir-neuron-db; weights → riir-train |
| `riir-ai` | Pillar-level or Super-GOAT (fusion connecting ≥2 pillars, or new pillar candidate) | Adaptive/self-learn NPCs, reasoning pack, MMORPG-scale (20Hz/fog/zone/crowd MCGS), 3D game wiring, freeze/thaw runtime, latent ops on belief/functor/cgsp | Generic transformer → katgpt-rs; chain → riir-chain; shards → riir-neuron-db; training → riir-train |
| `riir-chain` | Pillar-level or sync-boundary bridge novelty | LatCal, quorum/catchup, economics, asset lifecycle/forensic, DeFi, `riir-chaind`, raw↔latent bridge | Generic fixed-point without commitment → katgpt-rs; shards → riir-neuron-db |
| `riir-neuron-db` | Pillar-level or shard/freeze/consolidation novelty | `NeuronShard`, freeze envelope, Raven/δ-Mem, AnyRAG, vibe KG, Merkle, spectral init, compaction, dendritic | Chain commitment of shards → riir-chain; runtime swap → riir-ai |
| `riir-train` | **Active moat.** Training-method implementations + configs + trained weight assets | Adapter training, optimizers, losses, quant-aware **training**, DPO/GRPO/SFT, trained assets | Inference/runtime/latent ops → katgpt-rs or riir-ai |
| `riir-clippy` | **Consumer-first moat (investment #2).** Measured healer-quality gains: retrieval floors, score-bench heal rate, OOD generalization, fixer coverage | Corpus/retrieval structure, trajectory memory, selection math, eval harnesses, healer-consumed primitives (`.research/` notes live here for healer-domain distills) | Generic math stays in katgpt-core (consume, never fork); game-runtime concerns → riir-ai |

9 sloppy-test pillars live in `riir-ai/.docs/03_pillars/README.md` — `read_file` `03_pillars/README.md` + `04_supergoat_candidates/README.md` before any "does this become a pillar?" verdict. Sloppy test: *if it doesn't exist, the system goes structurally sloppy — not slower, broken.*

**`katgpt-rs` per-stack ledger:** every primitive gets a feature flag + benchmark + GOAT gate; the verdict note records which stack slot (attention/KV/sampling/speculative/pruning) + promote-to-default or stay-opt-in. Re-gate on feature touch. Demote the loser when a newer primitive wins the same slot.

### 1.7. Pre-plan cherry-pick audit

If your plan will consume/wire/fuse with a katgpt-rs primitive into riir-* → run the `goat-audit` skill before opening the plan. Catches (a) stalls (default-on in katgpt-rs ≥7 days with zero riir-* consumer), (b) DRY violations (riir-* shipping a local duplicate of substrate — e.g. defining its own `KVCache` instead of consuming `katgpt-transformer`). When NOT to run: purely katgpt-rs-internal; bug fix no cross-repo angle; training-only genuinely out of scope.

### 2. If gain/GOAT, plan it

Plan `.md` in `katgpt-rs/.plans/` (modelless), `riir-ai/.plans/` (runtime/game), `riir-chain/.plans/` (chain/LatCal), `riir-neuron-db/.plans/` (shards), `riir-train/.plans/` (training-efficiency per §3.5 Path 0.5). `## Phase N` sections, `- [ ]` per task (`- [x]` done, `- [-]` deferred). Planning into riir-train IS allowed + actively encouraged for applicable training-efficiency papers. Super-GOAT plans come AFTER the guide — guide is strategy, plan is execution.

**Plan format + GOAT gate rule + UQ "Report the Floor" extension:** `read_file` `templates.md`. Compact: every new technique needs feature flag + benchmark proving gain before promoting to default; demote loser if newer wins same slot. UQ-bearing primitives (distributions/intervals/coverage) MUST benchmark against conformal-naive floor (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`) on CRPS/coverage/Winkler — if they can't beat it, the gate FAILS.

### 3. Implement to unblock

If a plan is blocked by a missing primitive, implement the minimal version. After GOAT check + proof of gain: promote to default if it wins, demote the loser.

### 3.5. Modelless unblock — MANDATORY before any riir-train deferral

Before deferring ANY gate/task/mechanism to riir-train, exhaust modelless paths first.

**Path 0 — training-target decomposition.** Is the value the **training-loop innovation** (new optimizer/loss/curriculum/RL) or the **math it computes** (closed-form drift, conditional score, Riemannian correction, steering formula, regression target)? If the math → decompose into components, grep for modelless analog of EACH. ALL have analogs → MODELLESS-VALIDABLE (no deferral). SOME missing → check paths 1–3 for those.

**Path 0 asks TWO questions per component, not one.** (1) **Coverage** — does an analog already ship? (feeds the §1.5 Q1 novelty gate). (2) **Extraction** — can the component be computed WITHOUT gradient descent: closed-form solution, derived schedule, predictor, bound, ordering law, limit behavior (zero-init / mean-field / leading-order), offline-computable spectrum? A row with no coverage but yes-extraction is the STRONGEST finding class (open-primitive candidate → katgpt-rs) — marking it "no analog" and moving on is the canonical failure #2 miss (the NQF Path-0 table said "no closed-form plateau predictor" and stopped, instead of noticing the paper HAD one to extract).

**Path 0 outputs an INVENTORY, not just a verdict.** Complete the component table even when the funnel lands on Path 0.5 — every row marked "analog exists" or "partial" is a candidate open primitive. Before dismissing any such row as "covered", run the **signal-diff check** (§3.6): read the shipped analog's core formula and state what signal it *consumes* vs the paper component's. Name-level cousin match ≠ coverage — the diff that surfaces a real gap sounds like "relevance vs utility" / "history vs current-query" / "aggregate vs counterfactual", and it is answerable with ONE read of the cousin's code. Canonical failure (2026-08-16, arXiv:2608.13040 LOPD): the modelless δ (counterfactual with-vs-without utility gate) was dismissed as "covered by EvidenceTier / engram gate" — but engram's kernel gate is `σ(dot(q,k)/τ)` (relevance-only; the shipped zero-query test pins gate=0.5 fusion of useless slots as correct behavior) and EvidenceTier is history-only 3-tier, while δ is query-conditional utility at use time. Reverse-grep could NOT catch this class — the gap was pinned as a feature, not a TODO (→ katgpt-rs Issue 656; surfaced only by user challenge, one grep into kernel.rs had sufficed).

**Granularity rule (canonical failure #3 — arXiv:2608.16739 Le Critique, 2026-08-20): diff the CONTAINING expression, not only the named cousin.** The BetaPosterior↔TETHER signal-diff was run and correctly returned “different shapes” (per-candidate posterior vs global mixture weight) — but the formula that CONSUMES BetaPosterior, `W_EVO·evolution + W_RATE·reliability` in riir-clippy `select_best_candidate`, is exactly TETHER's `b(ρ) = (1−ρ)p1 + ρp2` with ρ hand-pinned at 0.4 and never swept, while the realized-outcome feed (`EvolveRecorder::record_outcome`) already fires on every applied fix (→ riir-clippy Issue 033; found by user challenge — third rescue of this class after LOPD and NQF; user challenge is a load-bearing defense layer, treat every challenge as a signal-diff granularity audit). Rule: after diffing cousin C, read ONE level up — the expression combining C with other signals. Hand-tuned constants composing C (`const W_X: f32 = 0.6`) are the paper's shape wearing a different name; the grep is `W_[A-Z]|const [A-Z_]+: f32` on C's caller (one cross-repo grep had sufficed).

**Three-track adversarial panel — MANDATORY whenever the first-pass classification touches training** (routes to riir-train, PASSes a training-adjacent paper, or the abstract carries optimizer/loss/RL/backprop framing). Spawn 2 `spawn_agent`s IN THE SAME parallel batch as the §4 prior-art searches (one spawn round total). Advocacy briefs — neutral merging is the coordinator's job; each agent argues FOR its track and does NOT inherit your classification:

- **No-GD advocate** (constraint #1 tracks a+b: modelless inference + self-adaptive runtime): *"Extract every closed-form, derived quantity, predictor, schedule, bound, ordering law, limit-behavior, or offline-computable spectrum in this paper. For each: can it ship without training? Which repo/module? What GOAT gate? Argue FOR feasibility. Do NOT judge whether it already ships — another pass owns coverage."*
- **Model-based advocate** (constraint #1 track c: trained weights): *"Extract every actionable training-recipe item (optimizer/loss/schedule/init/rank/width/capacity/data). Which existing pipeline does it improve (riir-train Plans, quest_grammar, TernaryDraftModel, edge_lora)? Recipe + GPU-hours + GOAT gate vs modelless baseline. Argue FOR applicability."*

Coordinator merges both into the Path 0 table. **Discarding an advocate finding requires a one-line auditable reason in the note** (extends the §3.6 signal-diff discipline to track routing — the coordinator can still be anchored; this is the backstop). **A discard reason must survive mechanism-level scrutiny (TTPO lesson, 2026-08-29):** class-level prior art ("JitRL ships training-free TPO"), pure cost notes ("2× decode"), and action-level coverage ("the prune action ships in Plan 133") are NOT sufficient kills when the finding includes a calibration law, a product-form signal, or a wiring fusion those references do not carry. Cite the exact uncovered delta, or do not discard. **Brief hygiene — do not leak classifications through the stack description.** Describe repos by WHAT SHIPS (file + type names: “riir-clippy `self_evolve.rs` ships `select_best_candidate` + `EvolveRecorder` outcome recording”), never by your shape conclusion (“BetaPosterior candidate selection” pre-answers the examination the advocate should perform). The Le Critique miss propagated exactly this way: the No-GD advocate inherited the framing and never read the selection formula — its only riir-clippy blend proposal was the retrieval-layer one the framing made salient. Clearly inference-side papers (KV cache, sampling, spec-decode) skip the model-based advocate; clearly pure-hardware papers skip the panel only because the §0 technique-stripping pass already ran.

**Path 0.5 — training-cost-weighted re-evaluation (DEFAULT for training-efficiency papers as of 2026-08-06).** Training efficiency is actively pursued. Applicable training paper → Plan in `riir-train/.plans/` with recipe + GPU-hours estimate + GOAT gate comparing trained-vs-modelless baseline. Only genuinely out-of-scope → one-line redirect with justification.

**Track priority — serving-envelope fit (TTPO lesson, 2026-08-29: "test-time gain is modelless shallow reasoning").** When BOTH tracks produce filed plans, rank them by whether the mechanism's decision layer runs INSIDE the stack's hot path: modelless shallow selection (branch rollouts, EqR/pruner/selector choices, healer spans, 20 Hz game ticks) vs outside it (gradient TTT / per-item training — structurally unaffordable where base competence ≈ 0 or per-fix budget ≫ serving budget; canonical datum: the L4 fixer's ~191 s/fix + 0/60 EM). The envelope-fit track is the PRIMARY plan; the other is SECONDARY and says so in its Status line. GPU-hours affordability (Path 0.5) is necessary but not sufficient — a cheap training run that never reaches a production consumer loses to a modelless gate on the live serving path.

**The model-based track = ALL training pipelines across ALL repos, not just riir-train** (GDN-blog lesson + arXiv:2511.18538 lesson). `read_file riir-train/.docs/02_pipelines/training_data_pipeline.md` before PASS-ing any training paper — RL/distillation/linear-attention/trajectory-collection pipelines all count. **BUT ALSO grep non-riir-train repos for training code** (pre-flight #5): `quest_grammar/grammar_training.rs` + `quest_training.rs` in riir-ai (LoRA training for quest grammar + quest corpus), `TernaryDraftModel` in riir-ai/riir-clippy (ternary-weighted drafter loaded from `.bits` files), `self_evolve` in riir-clippy (self-adaptive latent-first loop). Training is NEVER "deferred indefinitely" — it is actively pursued across the whole stack.

**Systematic backstop:** when ≥3 training-recipe gaps accumulate from PASS-Redirects across repos, batch them into a single `riir-train/.plans/NNN_training_recipe_gap_backlog.md` plan.

**Three modelless unblock paths (check ALL before deferring, AFTER Path 0 fails; run Path 0.5 AFTER all three fail):**

1. **Freeze/thaw snapshot correction** (`MerkleFrozenEnvelope`) — can a corrected snapshot, thawed at inference, fix a systematic bias (doubled signal, position offset, attention asymmetry)?
2. **Raw/lora reader-writer hot-swap** (`LoraPair { reader, writer }`; `LoRAHotSwap`, `dispatch_lora_merge` in riir-ai) — can a **deterministically constructed** (not trained) adapter fix it? Closed-form (scale-by-0.5, zero-out-positions, identity-minus-projection) rather than gradient descent?
3. **Latent-space correction** (dot-product + sigmoid gate, per constraint #2) — project latent onto correction direction, gate the output. Modelless analog of trained adapter.

**Decision protocol:**
```
Paper appears to need training
  → Path 0: value = MATH not training loop? YES → decompose + grep modelless analog of each
      ALL have analogs + a use case → MODELLESS-VALIDABLE. No deferral.
      SOME missing → paths 1–3 for those. ALL fail → Path 0.5.
      (either way: rows marked exists/partial → signal-diff EACH before "covered" — §3.6)
  → Systematic characterizable cause? YES → paths 1→2→3. ALL fail → Path 0.5.
  → Path 0.5: applicable → Plan in riir-train. Out of scope → redirect with WHY.
```
**Documentation requirement** for every riir-train Plan: Path 0 decomposition (which components had analogs, which didn't) · paths 1–3 checked + why each failed · what specifically requires GD no deterministic construction can provide · Path 0.5 affordability (recipe + GPU-hours + GOAT gate) · dual-track contribution (modelless inference primitive + trained weight artifact).

### 3.6. Defend-wrong PoC — MANDATORY before any "already ships" / "parity" verdict

Before claiming a mechanism "already ships", achieves "parity", or "covers" the paper's loop, distinguish three claim types:

| Claim | Proof |
|---|---|
| **Architectural** ("runtime analog exists") | grep + read code (sufficient) |
| **Latency/resource** ("modelless, sub-µs, no GD") | criterion bench |
| **Quality** ("matches/beats paper's numbers") | **head-to-head PoC on controlled toy — architectural reasoning NOT sufficient** |

**Failure mode:** claiming all three with only architectural evidence. Grep proves mechanism exists; it does NOT prove it performs as well as the paper's version.

**PoC mandatory when:** quality-parity verdict ("matches", "competitive with", "covers at parity"); qualitative Super-GOAT/GOAT claim; **any PASS that downgrades on grounds "the runtime analog already ships"** (architectural-only PASS is the #1 false-PASS mode).

**Scope extension (2026-08-16, LOPD lesson): coverage-dismissals inside Gain verdicts need the same defense.** Claiming a paper's secondary mechanism is "covered by cousin X" — while filing the primary mechanism as a training plan — is a verdict, and it must rest on mechanism-level evidence, not a name match. The proportionate defense is the **signal-diff check**: read the cousin's core formula/gate and state what signal it *consumes* (relevance? history? aggregate?) vs the paper component's (utility? current-query? counterfactual?). Full PoC remains mandatory only for quality-parity claims; a dismissal needs one read of the cousin's formula. Two blind spots this closes: (a) every other defense (§3.6 triggers, reverse-grep) fires only on PASS — a Gain verdict's secondary dismissal had NO guard; (b) reverse-grep finds *documented* gaps (`TODO|FIXME|limitation`) — this miss class ships **undocumented**, with the blind spot pinned as correct behavior in the cousin's own test (engram zero-query gate=0.5 fusion; → katgpt-rs Issue 656).

**PoC NOT required when:** pure architectural redirects (no quality claim); training-only genuinely out of scope; latency-only claims (single bench suffices); low-confidence verdicts that explicitly mark quality claim unproven + create `.issues/` follow-up.

**Where the PoC lives:** `riir-ai/crates/riir-poc/` (defend-wrong R&D crate). Three competitors minimum: paper's mechanism (or distilled modelless analog), frozen/no-adaptation baseline, shipped runtime analog. Head-to-head on controlled toy domain, no training, print verdict table. `CARGO_TARGET_DIR=/tmp/...` + clean up.

**PoC defends OR refutes.** If it refutes quality: do NOT silently revise. Record raw numbers in research note §"PoC Addendum". State which axes were confirmed (architectural, latency) vs refuted (quality). Verdict stands on confirmed axes; refuted axis becomes tracked follow-up. PoC stays as permanent regression check.

### 4. Published prior-art search — MANDATORY before any novelty verdict

**Hard gate. Not optional.** Before claiming ANY novelty (Q1), web-search for published prior art on the headline technique. The canonical failure mode: claiming "ternary MoE is novel" when one search for "mixture of ternary experts" would have found published prior art. The miss happens because the agent only read the paper's own references + grepped the codebase, never searching broader literature.

**Mandatory searches (run ALL before any novelty claim):**
1. Headline technique verbatim (`"mixture of ternary experts"`, `"BitNet distillation recipe"`).
2. 2–3 component techniques.
3. The selling-point framing you're about to claim.
4. Recent (2-year) surveys on the topic — they name the competitive landscape.

**If any published paper does what you claim is novel → downgrade Q1 BEFORE writing the verdict.** Cite explicitly. Do NOT write the verdict then discover prior art later.

**Parallelize via subagents:** 2–3 `spawn_agent` in parallel (headline search, component search, codebase grep). Web catches published prior art; codebase grep catches shipped prior art. Both mandatory; neither substitutes.

**Re-run after corrections:** discovering prior art in a later pass is NOT complete until you've (a) updated novelty verdict, (b) re-checked surviving novelty claim, (c) committed the correction. Don't leave notes in overclaimed state.

### 4.5. Optional deeper search

If §4 surfaces rich landscape, use web search for deeper exploration of specific papers/authors/follow-ups. Not mandatory, valuable when prior-art landscape is dense.

## Constraints (non-negotiable)

1. **Three-track system (not modelless-only)** — the stack runs three concurrent tracks: (a) **modelless inference** (primary hot path — no weights, BLAKE3-deterministic, zero-alloc), (b) **self-adaptive** (runtime latent updates — `self_evolve`, EMA direction vectors, freeze/thaw cycles, CommittedFieldBlend — NO base weight mutation), (c) **model-based** (trained weights — ternary `.bits` files via `TernaryDraftModel`, LoRA/QLoRA adapters, Bonsai model comparison, quest_grammar training pipelines). Do NOT characterize any repo as "modelless only" without checking pre-flight #5. Closest to "training" for the modelless track: freeze/thaw cycles, raw/lora hot-swap with **deterministically constructed** adapters (not trained), latent direction-vector updates at runtime. **Before any riir-train deferral, exhaust §3.5. AND check for existing model-based code in the target repo.**
2. **Latent-to-latent preferred** — operate in latent space as long as possible. Decode/project only at boundary. **Sigmoid, never softmax**, for projections onto learned directions. Semantic (emotion/mood/curiosity/style) → latent. Physical (position/HP/wallet) → raw, deterministic, synced.
3. **Freeze/thaw over fine-tuning** — only runtime weight mutation is swapping a frozen snapshot (atomic, versioned, BLAKE3-checked) or applying a deterministically-constructed LoRA overlay (raw/lora hot-swap, no GD). Never mutate weights in-place during inference. Gradient updates (after §3.5) → riir-train.
4. **Self-learn / adaptive CoT welcome** — runtime curiosity, latent prediction, trajectory folding, collapse detection. Update latent state / direction vectors / routing tables, NOT base weights.
5. **7-repo discipline** (the product/distillation set; canonical list in `katgpt-rs/AGENTS.md` §"Repo count") — katgpt-rs (public) → riir-ai → riir-chain → riir-neuron-db → riir-train (all private) + riir-game-sdk (facade) + riir-dapps (dApp layer: game outcome → generic chain settlement, added 2026-08-20). It read "8-repo" and included `riir-armageddon` until 2026-09-03; that repo was retired 2026-09-02 (owner act, directory gone). Training how never leaks to katgpt-rs; chain IP in riir-chain; shard IP in riir-neuron-db; SDK stays facade over `riir-games-shared`.
6. **SOLID, DRY** — per `katgpt-rs/.contexts/optimization.md`. Zero-alloc hot paths. Pre-computed lookup tables. Fixed-size arrays for bounded domains.
7. **Tests/examples** — before/after showing the gain. Latent ops: projection preserves ranking. Freeze/thaw: readers never see torn snapshots.
8. **CPU/GPU/ANE auto-route** — threshold-adaptive. Plasma (µs SIMD) → Hot (sub-ms GPU) → Warm/Cold (ms+ GPU/ANE). L1-fitting latent ops stay SIMD; batched matmul goes GPU.
9. **Plasma → Hot → Warm → Cold → Freeze tiering** — perf on game side (plasma/hot budget), security on chain side (cold/freeze commitment, BLAKE3-hashed, tamper-evident). Latent state crossing sync boundary MUST be raw scalars (valence/arousal/desperation/calm/fear), never the full embedding.

## Latent vs raw space rules (critical for game AI)

- **Physical** (position, velocity, HP, wallet): MUST be raw exact. Deterministic replay, quorum sync, anti-cheat require bit-identical reconstruction.
- **Semantic** (emotion, mood, curiosity, style, habit): SHOULD be latent via dot-product + sigmoid onto learned direction vectors.
- **Social** (encounters, relationships, factions): SHOULD produce KG triples from latent/embedding proximity, not raw coordinate distance.

**Sync boundary:** if data flows through `SyncBlock → ChainConsensus` quorum → Cold tier, it MUST be raw + deterministic. If consumed locally (emotion projection, shard retrieval, consolidation sleep-cycle), it SHOULD be latent. Bridge functions (raw→latent projection, latent→raw scalar clamp) MUST be zero-alloc, gateable, sync-invariant.

**KG triple emission:** semantic encounters → KG triple from latent similarity. Physical events → TxDelta with raw values, NOT KG triple. Never substitute latent embedding for raw position in anti-cheat.

**Spatial cognition (two-brain model):** info brain = real `MapPos` (synced, ground truth). Think brain = per-NPC `SpatialBelief` (zone KG triple + stale last_known_pos, fog-of-war gated, NOT synced). Bridge is one-way: real position → belief update only when within `visible_radius`. Confidence decay: `sigmoid(-λ * (current_tick - last_observed_tick))`. Two brains MUST exist independently — divergence is emergent, not a bug.

## Cross-references (read on demand)

Commercial strategy / moat map: the inline short version above suffices for routing decisions. For Super-GOAT novelty gates or "does this become a pillar?" questions, `read_file` `riir-ai/.docs/README.md` (+ `03_pillars/README.md`, `04_supergoat_candidates/README.md`). Exhaustive moat analysis at `riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md` (commercially sensitive — read only when inline short version insufficient).
