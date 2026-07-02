---
name: research
description: Research workflow for distilling ML/AI papers into modelless inference primitives, freeze/thaw runtime patterns, and latent-space operations across the katgpt-rs / riir-ai / riir-chain / riir-neuron-db / riir-train repo quintet. Use when reading arxiv papers, deciding which repo a paper belongs in, creating .research/ notes or .plans/ files, implementing modelless inference primitives, or routing training-vs-inference insights. Enforces the 5-repo commercial strategy (public engine / private runtime / private chain / private neuron-db / private training), modelless-first constraint, latent-to-latent preference, and freeze/thaw-over-fine-tuning rule.
---

# Research Workflow — Modelless Inference, Freeze/Thaw, Latent-to-Latent

Training-method research lives in `riir-train`. This repo (`katgpt-rs`), `riir-ai` (freeze/thaw runtime + self-learn/adaptive NPCs + game systems), `riir-chain` (neuro-symbolic chain transport, LatCal, chain economics), and `riir-neuron-db` (neuron weight shards, BLAKE3/Merkle commitment, freeze/thaw envelope, consolidation, AnyRAG gateway, vibe KG triples) ship **runtime + latent-space operations**. No LoRA training, no adapter fine-tuning, no optimizer research here. If a paper's value is its training loop → `riir-train/.research`. If its value is a latent-space insight, a routing trick, a freeze/thaw pattern, a chain-commitment bridge, a neuron-shard primitive, or a modelless inference primitive → distill here.

## When to use this skill

Activate when the user (or you) are doing any of:

- Reading / fetching / summarizing an ML, AI, or systems paper (arxiv, PDF, blog).
- Deciding which of the 5 repos a paper or idea belongs in (katgpt-rs / riir-ai / riir-chain / riir-neuron-db / riir-train).
- Creating a new `.research/NNN_*.md` note or `.plans/NNN_*.md` plan.
- Implementing a modelless inference primitive (pruner, bandit, router, speculative decode, KV-cache op, sparse attention, quantization-aware inference).
- Designing freeze/thaw snapshot cycles, adapter hot-swap, or runtime adapter routing.
- Designing latent-to-latent operations (dot-product projection, sigmoid gating, manifold geometry, spectral methods on activations).
- Designing MMORPG-scale game AI (thousands of concurrent NPCs, 20Hz tick, fog-of-war, zone attention, emergent social/economic behavior).

Do NOT activate for: pure refactor tasks, bug fixes with no research angle, or ordinary feature work that doesn't touch the research/plans folders.

## Repos (siblings under the same parent)

- `katgpt-rs/` — public MIT engine. Generic modelless inference primitives. **No game IP, no chain IP, no neuron-shard IP.**
- `riir-ai/` — private game product. Freeze/thaw runtime, self-learn, game systems. Hosts the consolidated `.docs/` book (selling points / moats — see §Read first).
- `riir-chain/` — private neuro-symbolic chain transport. LatCal (Lattice Calculus), `riir-chaind` daemon, chain economics, Solana-parity features, asset lifecycle / forensic fingerprinting, `catchup/` (Turso/libSQL persistence, quorum), `DataTier` / `DATA_TIERS` / `build_tier_root`. **The sync-boundary bridge repo. Re-exports `riir-neuron-db` under its `neuron_db` feature, but the canonical shard source is `riir-neuron-db/`.**
- `riir-neuron-db/` — private leaf crate. `NeuronShard` (`#[repr(C)]` Pod, zero-copy mmap), `ShardIndex` (lock-free `papaya::HashMap`), generic `MerkleTree`/`MerkleProof`, MAPE-K self-healing loop, Raven/δ-Mem consolidation, AnyRAG escalation gateway, vibe KG triple templates + arch agent, `MerkleFrozenEnvelope` (freeze/thaw integrity), spectral initialization, `ShardCompactor`, dendritic LoRA branch view. **No chain dependency — usable standalone.**
- `riir-train/` — private training vault. Adapter training, optimizers, loss functions. Out of scope for this workflow — just note "→ riir-train" and stop.

## Repos (siblings under the same parent)

**Routing rule of thumb (chain vs neuron-db):** if the mechanism is about *how a shard is structured, committed, frozen, consolidated, retrieved, or projected* → `riir-neuron-db`. If it is about *how a shard is committed to a chain block, transported across quorum, or bridged to LatCal fixed-point* → `riir-chain`. The `LatCalWalletExt` trait (typed wallet accessors on `NeuronShard` using `LatCalMatrix`) stays in `riir-chain` because it is the bridge.

Always reference files with project-relative paths (e.g. `katgpt-rs/.research/238_*.md`, `riir-ai/.plans/NNN_*.md`, `riir-chain/.plans/NNN_*.md`, `riir-neuron-db/.plans/001_*.md`). The agent can `read_file` these directly.

## Commercial strategy — inline short version (no external ref)

Five repos. The split is absolute. See §Repos above for the boundary table + per-repo roles. What follows is the decision-routing essence an agent needs at verdict time.

**Tier model (the single most useful structural rule):**

| Tier | Where | Role | Dep profile |
|------|-------|------|-------------|
| **0 — Substrate** | `katgpt-core` (leaf, crates.io) | Pure inference mechanics (SIMD, transformer/weights, `hla`, `dd_tree`, `mcts`, `sampling`, `delta_mem`). The pillars EVERY repo needs for compute. | Minimal, leaf-clean. No `rayon`/`bevy_ecs`/`wasmi`/`plotters`/`metal`. |
| **1 — Engine + cognitive basics** | `katgpt-rs` (root, public) | Adoption funnel — re-exports substrate + ships the BASIC cognitive/reasoning layer + engine primitives + toy game engines/examples (each ships WITH its `.md`). | Heavier (`rayon`, etc.) but deps kept optional where possible. |
| **2 — GOAT versions + composition + IP** | `riir-ai` / `riir-chain` / `riir-neuron-db` / `riir-train` (private) | The gas — GOAT/Super-GOAT tuned versions, `*_runtime` composition layers, game/chain/shard/training IP. | Private; whatever each product needs. |

Two rules fall out: (1) a module moves DOWN to core only if it's pure inference substrate (no heavy deps, no cognitive semantics, needed by every compute consumer). `hla` qualifies; `cce`/`cgsp` do not. (2) a module stays in root (tier 1) if it's a cognitive/reasoning primitive — the BASIC public version. Its GOAT-tuned sibling (the `*_runtime` module) lives in `riir-*`.

**The `*_runtime` suffix convention encodes the WHAT/HOW split at module granularity:** `cce` (public primitive) + `cce_runtime` (private GOAT composition); `cgsp` + `cgsp_runtime`; `arg` + `arg_runtime`. Bare-name module = public WHAT; `*_runtime` module = private HOW.

**WHAT vs HOW routing rule:**
- **What = public. How = private.** Training how → `riir-train`. Runtime how → `riir-ai`. Chain how → `riir-chain`. Shard how → `riir-neuron-db`.
- "NPCs hot-swap personalities at runtime via versioned snapshots" → public (capability).
- "The snapshot uses [specific concurrency primitive] with [specific memory ordering]" → `riir-ai` private (runtime how).
- "We train adapters with [specific method] at [specific config]" → `riir-train` private (training how).
- "Balances are encoded as latent vectors and committed via LatCal fixed-point bridges" → public (concept).
- "The LatCal projection uses [specific learned values]" → `riir-chain` private (implementation).
- When unsure → default to the relevant private repo. It is always safe to keep something private; it is never safe to un-leak something public.

**Benchmark domain exception (toy games ≠ product IP):** Toy 2D rule-system games (`bomber`, `monopoly`, `go`, `fft` ATB arena) used as benchmark domains are NOT game IP — they live in `katgpt-rs` (public) and that is correct. The moat is the runtime that runs *on top* of a game (freeze/thaw composition, archetype wiring, HLA tuning, trained weights, design data), not the rule system. Distinguishing test: *"Could a competitor re-implement this from public rules + generic primitives in a weekend?"* → public. *"Does this encode product-specific tuning, weights, wiring, or design?"* → private. **Anti-pattern:** a public benchmark constant whose comment names a private module path (`must match riir_gpu::...`) IS a leak even if the value itself is benign — cross-boundary coupling constants are forbidden; private consumers import from public (one-way), never the reverse.

**Cognitive/reasoning — the asymmetric moat:** Basic cognitive/reasoning primitives (`cce`, `cgsp`, `clr`, trajectory compaction, claim rubrics) stay PUBLIC in tier 1 (good enough to adopt, demonstrates the capability — the adoption hook). GOAT/Super-GOAT tuned versions stay PRIVATE in tier 2 (the version that actually wins — collapse-recovery thresholds, curiosity tuning, personality-blend freeze/thaw integration). "Good enough to adopt, not good enough to win." A competitor forking `katgpt-rs` gets the basic primitives but not the tuning that makes NPCs actually behave well.

**Why katgpt-rs is public (the "Ferrari, no gas" model):** `katgpt-rs` is the open Ferrari. It's the adoption funnel — developers build on the engine, then need the platform. The engine alone produces no useful competitive game AI output (it's a runtime, not the intelligence). MIT attracts contributors + creates dependency without exposing know-how + no legal friction for enterprise. The gas is inside `riir-ai` (gameplay) and `riir-chain` (ledger).

**Why each private repo is hard to replicate (one-liners):**
- `riir-ai` — freeze/thaw runtime (lock-free snapshot swap, concurrency tuned, reader-invariant now a Lean theorem); latent-to-latent bridge (raw↔latent boundary requires per-game tuning, scalar boundedness proven); self-learn/adaptive NPCs (collapse-detection + exploration tuned against real sessions).
- `riir-chain` — co-located AI+wallet (no incumbent co-locates these in fixed-size structures); LatCal commitment bridge (deterministic fixed-point, round-trip + quorum determinism now Lean theorems); network effects (a forked chain has no players/validators/economy — can't be copied).
- `riir-neuron-db` — `NeuronShard` fixed-layout Pod (every field offset pinned for cross-platform determinism; layout + `merkle_root` init now Lean theorems closing the forgetting-bug class); freeze/thaw envelope (freeze gate contract proven); Raven/δ-Mem consolidation (emergent, session-tuned — not derivable from first principles); AnyRAG escalation policy (the *when to escalate* is the IP); vibe KG triple templates (domain-tuned).
- `riir-train` — 90+ adapter-training method implementations + configs (algorithms are published; the implementations + honest benchmarking took years — most methods deliver stability not quality, but the few that work + trained weight assets are GPU-hours).

**Formal verification as a moat (not ceremony):** The quintet carries ~79 Lean 4 theorems across four `.proofs/` instances (`KatgptProof` public, `RiirChainProof` / `NeuronDbProof` / `RiirAiProof` private). A Lean theorem is the strongest modelless guarantee: zero runtime cost, forever-verified, refactor-immune. A competitor forking any `riir-*` repo gets the code but not the theorems (private proofs stay private) — the theorems encode the *invariant shape* the runtime depends on, so a re-implementation that satisfies the same empirical tests can still violate invariants in a code path the tests don't cover (the `merkle_root` / `can_freeze` / Issue-354 torn-read pattern repeats this lesson). FV is also a bug-finder: scoping the riir-ai freeze/thaw theorem found a real concurrency bug the existing test couldn't catch.

## Read first (grounding) — MANDATORY pre-flight

**Hard rule:** before any distillation, verdict, or file creation, you MUST do **all three** of these in this session:

1. **`read_file` the four READMEs + the riir-ai `.docs/` book** — these define repo purpose, the commercial moat map, and the raw-vs-latent sync boundary the research must respect. Skipping this is the #1 cause of research notes that ignore the actual codebase architecture and the #1 cause of false Super-GOAT claims (claiming novelty over a moat that already ships).
2. **`list_directory` all four `.research/` folders** — these hold the existing distillation corpus you must not duplicate. (Create `riir-chain/.research/` and `riir-neuron-db/.research/` if they do not yet exist and you are about to drop a note there.)
3. **`list_directory` the four runtime/chain/neuron-db crate src trees** — module names are the codebase's own vocabulary; skipping this is the #2 cause of false Super-GOAT claims (vocabulary mismatch).

**Mandatory reads (before any verdict):**
- `katgpt-rs/README.md` (`read_file`) — public engine purpose, architecture, current feature set.
- `riir-ai/README.md` (`read_file`) — private runtime context (freeze/thaw, self-learn, game systems).
- `riir-chain/README.md` (`read_file`) — chain transport, LatCal, economics, feature-flag umbrellas (`chain`, `chain_economics`, `chain_solana_parity`, `chain_catchup`, `chain_asset_*`, `shard_compactor`, `lora_posterior`). Required reading for any LatCal / commitment / sync-bridge research.
- `riir-neuron-db/README.md` (`read_file`) — neuron-shard leaf crate. `NeuronShard` Pod layout, BLAKE3/Merkle commitment, feature gates (`spectral_shard`, `shard_compactor`, `merkle_freeze`, `dendritic_lora`, `state_compression`). Required reading for any shard / freeze-envelope / consolidation / AnyRAG / vibe-KG / Merkle-tree research.
- `riir-ai/.docs/README.md` (`read_file`) — **the consolidated selling-point book.** Organized by capability: `pillars/` (9 proven commercial moats, 4-layer architecture), `supergoat_candidates/` (bets that might become pillars), `reasoning/`, `self_learn_npcs/`, `neuro_symbolic_chain/`, `performance/`, `browser/`, `wasm_validators/`, `showcase/`. **`read_file` the `pillars/README.md` and `supergoat_candidates/README.md` indexes before any novelty gate or Super-GOAT verdict** — claiming novelty over a moat that already ships is the worst false-positive class. When a verdict touches a specific pillar's domain, `read_file` that pillar's doc (e.g. `pillars/reasoning_pack.md`, `pillars/fourier_spatial.md`) before claiming the new primitive multiplies it.
- `katgpt-rs/.research/` (`list_directory`) — public modelless research corpus (do not duplicate).
- `riir-ai/.research/` (`list_directory`) — private runtime/game research corpus (do not duplicate).
- `riir-chain/.research/` (`list_directory` — create the folder on first use) — private chain research corpus.
- `riir-neuron-db/.research/` (`list_directory` — create the folder on first use) — private neuron-shard research corpus.
- `riir-ai/crates/riir-engine/src/` (`list_directory`) — **runtime module tree = codebase vocabulary at the highest level.** Module names (`latent_functor/`, `cgsp_runtime/`, `micro_belief/`, `adapters/`, ...) are how the codebase describes its own mechanisms. Skipping this caused the Research DiPOD miss: `latent_functor/reestimation.rs` ships the exact "drift-triggered self-healing swap" pattern under the name "coherence-driven re-estimation scheduler" — invisible to a paper-vocabulary grep.
- `riir-ai/crates/riir-games/src/` (`list_directory`) — game systems module tree (same rationale).
- `riir-chain/src/` (`list_directory`) — chain module tree: `encoding/` (LatCal), `consensus/`, `economics/`, `asset_lifecycle/`, `forensic/`, `programs/`, `validator/`, `wallet/`, `batch/`, `catchup/`, `deploy/`, `shell/`. The chain-side `LatCalWalletExt`, `DataTier`, `DATA_TIERS`, `build_tier_root`, `build_block_root` live in `riir-chain/src/catchup/merkle.rs`.
- `riir-neuron-db/src/` (`list_directory`) — **shard module tree = neuron-db vocabulary.** Files: `shard.rs` (NeuronShard Pod layout, dendritic branch view), `index.rs` (ShardIndex lock-free papaya), `merkle.rs` (generic MerkleTree/Proof), `freeze.rs` (`MerkleFrozenEnvelope`), `mape_k.rs` (self-healing loop), `consolidation.rs` (Raven/δ-Mem), `gateway.rs` (AnyRAG escalation), `vibe.rs` (KG triple templates + arch agent), `spectral_flatness.rs` (lottery-ticket init), `shard_compactor.rs` (cold-tier compaction), `reconstruction_metrics.rs`.

If you have NOT done all of: `read_file` the 4 READMEs + `riir-ai/.docs/README.md`, `list_directory` all four `.research/` folders, AND `list_directory` the four runtime/chain/neuron-db crate src trees, STOP and do so now. Do not create any file until all of them are done.

Then read for additional context (as relevant to the topic):
- `katgpt-rs/src/` + `katgpt-rs/crates/katgpt-core/src/` — existing modelless primitives (ConstraintPruners, bandits, DDTree, speculative decode).
- `riir-ai/crates/` — runtime IP: `riir-engine`, `riir-games`, `riir-ffi`, `riir-data`, `riir-examples`.
- `riir-chain/crates/` — chain daemon crate: `riir-chaind`. (LatCal, encoding, etc. live under `riir-chain/src/`; shard types come from `riir-neuron-db`.)
- `katgpt-rs/.plans/` + `riir-ai/.plans/` + `riir-chain/.plans/` + `riir-neuron-db/.plans/` — existing plans. **Do NOT list these in pre-flight.** Grep them during fusion search (§Workflow step 1), not as grounding — they describe what we *plan to build*, not what the repos *are*.

## Primary focus (distill HERE in katgpt-rs / riir-ai)

**Fusion-first mindset:** The highest-value Super-GOATs in this codebase come from **fusing 2–3 papers/primitives into a novel combination**, not from direct-mapping a single paper. Always grep `.research/` + `.plans/` for the 2–3 closest cousins before verdict, and ask: "what does paper × note A × note B produce that none of them alone can?" Examples that shipped: Gemini Fourier × LatCal (research 212 → plan 242); EGA × SpectralQuant (research 100 × 039); collapse-aware × bandit × sigmoid-margin (plans 212 × 157 × 061). See §Workflow step 1 for the full fusion protocol.

- **Latent-to-latent operations** — anything that stays in embedding/latent space: dot-product projections, cosine similarity retrieval, sigmoid-gated routing, manifold geometry, spectral methods on activations. Prefer operating on latents over decoding to tokens then re-encoding. **Fusion hook:** combine with freeze/thaw to version latent-direction vectors; combine with self-learn to update direction vectors from runtime curiosity signal.
- **Freeze/thaw patterns** — versioned weight snapshots, atomic hot-swap, lock-free read paths, BLAKE3/commitment-checked adapter reload, per-entity personality divergence via snapshot versioning. **Fusion hook:** combine with runtime adapter routing to dispatch by latent-state similarity; combine with self-learn to snapshot emergent NPC personalities.
- **Runtime adapter routing** — selecting between frozen adapters by state/objective/context (Dynamic Pair, Polytope, dMoE — all inference-time, zero training). **Fusion hook:** combine with freeze/thaw to make the adapter pool itself versioned and BLAKE3-committed; combine with bandits to learn routing policy online.
- **Self-learn / adaptive CoT** — runtime curiosity, entropy-driven exploration, collapse detection/recovery, latent prediction SSL, trajectory folding. No LLM training, no backprop through weights — runtime self-improvement via latent-space updates is welcome. **Fusion hook:** combine with MMORPG-scale game AI to give thousands of NPCs independent curiosity/entropy signals; combine with freeze/thaw to checkpoint learned latent directions.
- **Modelless inference primitives** — ConstraintPruners, bandits, DDTree, speculative decode, sparse attention, quantization-aware inference.
- **MMORPG-scale game AI** — thousands of concurrent NPCs each with independent latent state, real-time latency budgets (20Hz tick, plasma/hot tier), spatial partitioning + fog-of-war, emergent social/economic behavior (factions, trade routes, reputation), zone-level attention routing, crowd-scale curiosity/exploration signals. Latent ops must batch across many entities; raw sync must stay bit-identical for deterministic replay/anti-cheat.

### Super-GOAT factory modules — grep FIRST, explicitly

The highest-value latent-space Super-GOATs cluster in seven module trees. When grepping for fusion cousins and prior art, `list_directory` these explicitly — do NOT rely on keyword grep alone (vocabulary mismatch is the #3 cause of false verdicts):

| Module | What ships | Super-GOAT angle |
|---|---|---|
| `katgpt-rs/crates/katgpt-core/src/sense/` | HLA belief-state kernels, `evolve_hla`, `SenseModule::project`, ternary bit-plane projection | Per-NPC recurrent latent state — the runtime substrate for any "hidden state" / "belief" / "activation" paper |
| `riir-ai/crates/riir-engine/src/latent_functor/` | `zone_gating.rs`, `reestimation.rs`, `arithmetic.rs`, `cross_game.rs`, `k_selector.rs`, `quality_gate.rs` | **Game-theory in latent space** — functors as vector ops, coherence-driven re-estimation, zone-gated activation. Maps any "stage" / "application" / "bypass" / "collapse" paper |
| `riir-ai/crates/riir-engine/src/hla/` | `kernel.rs`, `forward.rs`, `types.rs` — per-NPC 8-dim latent state (valence/arousal/desperation/calm/fear + 3) | The emotional/cognitive latent state — maps any "subspace" / "width" / "channel" paper to per-NPC affect |
| `riir-ai/crates/riir-engine/src/cgsp_runtime/` | Curiosity-guided self-play, latent prediction SSL, MCTS collapse bridge | Runtime curiosity/exploration — maps any "self-learn" / "entropy-driven" / "collapse recovery" paper |
| `riir-neuron-db/src/` | `shard.rs` (NeuronShard Pod, `style_weights[64]`, dendritic branch), `freeze.rs` (`MerkleFrozenEnvelope`), `consolidation.rs` (Raven/δ-Mem), `gateway.rs` (AnyRAG escalation), `vibe.rs` (KG triple arch agent), `merkle.rs` (generic MerkleTree/Proof), `mape_k.rs` (self-healing loop), `spectral_flatness.rs` (lottery-ticket init), `shard_compactor.rs` | **Frozen latent-state storage + integrity + retrieval** — the persistence substrate for any "snapshot" / "integrity envelope" / "memory consolidation" / "external knowledge escalation" / "KG triple emission" paper. Maps any "memory" / "replay buffer" / "experience replay" / "spectral init" / "Merkle commitment" paper. |
| `riir-chain/src/encoding/latcal*.rs` + `latcal_fixed.rs` | Lattice Calculus: 2×2 matrix arithmetic obfuscation, fixed-point bridge, spectral fixed-point, batch determinant validation, DeFi programs | **The sync-boundary bridge** — deterministic, committed, raw-numeric. Maps any "fixed-point" / "deterministic commitment" / "raw↔latent bridge" / "arithmetic obfuscation" paper. LatCal is how latent ops become chain-committed raw values. |
| `katgpt-rs/crates/katgpt-core/src/dec/` | `operators.rs` (d=`exterior_derivative`, δ=`codifferential`, Δ=`hodge_laplacian`), `hodge.rs` (`hodge_decompose` exact/coexact/harmonic, `betti_numbers`, `harmonic_projector`), `flow.rs` (`DecFlowField` exact/coexact/harmonic channels), `terrain_cochains.rs` (Safety/Threat/Occupancy/Destruction typed cochains) — **shipped Plan 251, Research 219** | **The Generalized Stokes' Theorem substrate** — `d∘d=0` enforced by construction (tests verify `curl(grad)=0`, `div(curl)=0`). Maps any "divergence" / "boundary flux" / "line integral" / "curl" / "Hodge decomposition" / "Fokker-Planck" / "mass conservation" / "manifold geometry" / "exterior calculus" / "Stokes theorem" paper. The math ships; thin wrapper primitives (`boundary_flux_mass`, `belief_mass_divergence`, `line_integral`) are Plan 314 (Research 296). **Curse-of-dimensionality caveat: boundary-vs-volume wins only for d ≤ 3 (game maps, HLA regions, KG embeddings) — NOT high-dim shards.** |

**Adapter routing, KV compression, and speculative decode are GOAT-tier framings. Latent-to-latent operations on HLA/functor/neuron-shard/LatCal state are Super-GOAT-tier framings. Attempt the Super-GOAT framing first.** Defaulting to adapter routing when a latent-space reframing is stronger is the primary failure mode this protocol prevents.

## Redirect to riir-train (do NOT distill here)

**MANDATORY pre-check:** before redirecting ANY mechanism to riir-train, exhaust the modelless unblock paths in §3.5 below. A mechanism that *looks* training-only may be modelless-validable via freeze/thaw, raw/lora hot-swap, or latent-space correction. Only redirect if §3.5's decision protocol returns "genuine riir-train dependency".

If a paper is training-only (after §3.5 check) → note "→ riir-train" in one line and stop. Do not create files in this session for it.

**By topic:**
- LoRA / OFT / SPEFT / IA3 / QLoRA / ManifoldE / BAKE / GPart / MSA / Dendritic and all adapter-**training** methods.
- Training optimizers (Muon, Adam variants, symmetry-compatible optimizers).
- Training loss functions, curricula, distillation recipes.
- Quantization-aware **training** (quantization-aware **inference** stays here).
- DPO / GRPO / SFT / RL **training** pipelines (runtime GRPO self-play stays in `riir-ai` — it updates latent state, not weights).
- Anything that requires backpropagation through base weights.

**By user-request phrasing (these mean "→ riir-train"):**
- "Train a LoRA adapter to do X"
- "Fine-tune with method Y"
- "Optimizer Z improves convergence"
- "Distillation recipe from teacher to student"
- "Quantization-aware training" (but "quantization-aware inference" stays here)
- "DPO/GRPO/SFT/RL training pipeline" (but runtime GRPO self-play stays in riir-ai)

## Distillation targets (5-repo strategy)

| Repo | Role | What lands here |
|------|------|-----------------|
| `katgpt-rs` (public, MIT) | Engine — modelless inference framework | Generic primitives: ConstraintPruner traits, bandits, DDTree, speculative decode, sparse attention kernels. **No game IP, no chain IP, no neuron-shard IP.** |
| `riir-ai` (private) | Game product — freeze/thaw runtime, self-learn, game systems | Runtime IP: `LoRAWeightVersion`, `LoRAHotSwap`, `dispatch_lora_merge`, `TrainingProvider` trait, routing, game systems. |
| `riir-chain` (private) | Neuro-symbolic chain transport — LatCal, chaind | Chain IP: LatCal encoding/bridges, split-key ledger, chain economics, Solana-parity features, asset lifecycle / forensic, `riir-chaind` daemon, validator SDK bridges, `catchup/` (Turso/libSQL, quorum), `DataTier` / `DATA_TIERS` / `build_tier_root` / `build_block_root`. **Re-exports `riir-neuron-db` via `neuron_db` feature, but the shard source of truth is `riir-neuron-db/`.** |
| `riir-neuron-db` (private) | Neuron-shard leaf crate — shards, freeze, consolidation, retrieval | Shard IP: `NeuronShard` Pod layout + `style_weights[64]` + dendritic branch, `ShardIndex` lock-free papaya, generic `MerkleTree`/`MerkleProof`, `MerkleFrozenEnvelope`, MAPE-K self-healing, Raven/δ-Mem consolidation, AnyRAG escalation gateway, vibe KG triple templates + arch agent, spectral lottery-ticket init, `ShardCompactor`. **No chain dependency — usable standalone.** |
| `riir-train` (private) | Training research vault | **Only if the paper's value is its training method.** Out of scope for this workflow — just note "→ riir-train" and move on. |

Distill into:
- **Modelless** → `katgpt-rs/.research/` + `katgpt-rs/.plans/` + `katgpt-rs/src/` (or `katgpt-rs/crates/katgpt-core/`)
- **Runtime/game** → `riir-ai/.research/` + `riir-ai/.plans/` + `riir-ai/crates/`
- **Chain / LatCal / sync-bridge / commitment / quorum / catchup** → `riir-chain/.research/` (create if missing) + `riir-chain/.plans/` + `riir-chain/src/` (or `riir-chain/crates/`)
- **Neuron shards / freeze envelope / consolidation / AnyRAG / vibe KG / Merkle tree / spectral init / shard compaction** → `riir-neuron-db/.research/` (create if missing) + `riir-neuron-db/.plans/` + `riir-neuron-db/src/`
- **Training-only** → note the redirect, do not create files in this session

## Workflow

### 0. Read & classify the paper

Fetch via `https://r.jina.ai/https://arxiv.org/pdf/{ID}` (per AGENTS.md). Ask: *is the value in the training loop, or in a latent-space / inference / routing insight?* If training-only → note "→ riir-train", stop.

### 1. Distill fundamentally — fuse, don't just direct-map

Don't direct-map the paper to our code. Find the transferable primitive: the geometric, spectral, or information-theoretic insight that works without the paper's training setup. **Then look for fusion opportunities**: cross-pollinate this paper's insight with existing `.research/` notes, `.plans/`, and shipped primitives to synthesize a *novel* combination. The highest-value Super-GOATs in freeze/thaw runtime and self-learn/adaptive CoT almost always come from **fusing** 2–3 papers, not from a single-paper direct mapping.

**Fusion examples that shipped:**
- Gemini Fourier × LatCal → `katgpt-rs/.research/212_Gemini_Fourier_LatCal_Fusion_Verdict.md` → `katgpt-rs/.plans/242_Fourier_Smoothed_Potential_Fields_LEO.md`
- EGA spectral salience × SpectralQuant eigenbasis KV compression → `katgpt-rs/.research/100_*.md` + `039_*.md`
- Collapse-aware × bandit coverage × sigmoid margin → `katgpt-rs/.plans/212_*`, `157_*`, `061_*`
- G-Zero self-play × Hint-δ bandit × absorb-compress → `katgpt-rs/.plans/049_*` (modelless self-play distillation, 1.16M cycles/sec)

**Fusion protocol:**
1. **MANDATORY — grep ALL FIVE repos in this session, BOTH layers (notes AND code). Do NOT stop after the first repo or the first layer.** Run keyword / paper-title / author / primitive-name grep across:
   - `katgpt-rs/.research/` + `katgpt-rs/.plans/` (intent — what we planned)
   - `riir-ai/.research/` + `riir-ai/.plans/` (intent — runtime/game)
   - `riir-chain/.research/` + `riir-chain/.plans/` (intent — current chain research; `.research/` may need creating on first use)
   - `riir-neuron-db/.research/` + `riir-neuron-db/.plans/` (intent — current shard research; `.research/` may need creating on first use)
   - `riir-ai/.docs/` — the consolidated selling-point book (`pillars/`, `supergoat_candidates/`, `reasoning/`, `self_learn_npcs/`, `neuro_symbolic_chain/`, ...). These are not academic distillation; they are the moat/selling-point framing. Grep them alongside `.research/` so you do not claim novelty over a pillar that already ships.
   - `katgpt-rs/src/` + `katgpt-rs/crates/` (shipped primitives — what actually exists)
   - `riir-ai/crates/` (shipped runtime)
   - `riir-chain/src/` + `riir-chain/crates/` (shipped chain — LatCal, encoding, economics, forensic, catchup, etc.)
   - `riir-neuron-db/src/` (shipped shards — `shard.rs`, `freeze.rs`, `consolidation.rs`, `gateway.rs`, `vibe.rs`, `merkle.rs`, `mape_k.rs`, `spectral_flatness.rs`, `shard_compactor.rs`)
   - **Super-GOAT factory modules** (from §Primary focus) — `list_directory` these explicitly even if the paper looks pure-training: `katgpt-rs/crates/katgpt-core/src/sense/`, `riir-ai/crates/riir-engine/src/latent_functor/`, `riir-ai/crates/riir-engine/src/hla/`, `riir-ai/crates/riir-engine/src/cgsp_runtime/`, `riir-neuron-db/src/` (shards/freeze/consolidation/AnyRAG/vibe/merkle), `riir-chain/src/encoding/latcal*.rs`, `katgpt-rs/crates/katgpt-core/src/dec/` (Stokes/exterior-derivative/Hodge — maps any divergence/boundary/line-integral/Fokker-Planck/manifold-geometry paper)

   (riir-train is deliberately excluded — training methods are out of scope for this workflow.)

   Two layers, five repos. The closest cousin is frequently in the OTHER repo (e.g., a `katgpt-rs` modelless primitive fused with a `riir-chain` LatCal commitment bridge — see Gemini Fourier × LatCal; or a `riir-neuron-db` freeze envelope fused with a `riir-ai` runtime adapter hot-swap) OR in the CODE not the notes. **Notes describe intent; code describes what shipped.** A mechanism can ship without a research note — e.g., HLA's `evolve_hla` (`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs`) is a per-NPC recurrent belief-state kernel with no `.research/` note framing it as such; a notes-only grep misses it and produces a false Super-GOAT claim (verdict then has to be revised down). If you only grep `katgpt-rs/.research/`, you will miss both axes and produce a duplicate, weaker note, or an overclaimed verdict.

2. **MANDATORY — vocabulary translation before grepping.** Papers and our codebase use different words for the same mechanism. Before any grep, list the paper's 3–5 key mechanism terms, then for EACH, brainstorm ≥2 codebase-equivalent terms by asking: "if we shipped this, what would we call it?" Then grep BOTH sets.

   **Standing latent-state vocabulary (ALWAYS include, even for non-latent papers — most architecture/training papers have a latent-space reframing):**
   - "residual stream" / "hidden state" / "activation" → "HLA state", "belief state", "latent subspace", "sense projection"
   - "layer" / "depth" / "stage" → "decision stage", "functor application", "cgsp cycle", "consolidation tick"
   - "width" / "dimension" / "capacity" → "latent subspace", "active projection channel", "sense channel"
   - "carry-forward" / "bypass" / "skip" → "leaky integrator", "dormant subspace", "decay gate", "persistence"
   - "collapse" / "degeneration" / "valley" → "coherence decay", "re-estimation trigger", "staleness"
   - "bottleneck" / "narrowing" → "subspace projection", "channel selection", "zone gating"
   - "fixed-point" / "deterministic" / "committed" → "LatCal", "lattice calculus", "BLAKE3 commitment", "raw scalar bridge"
   - "divergence" / "flux" / "∇·F" / "density change" → "codifferential", "δ", "DEC divergence", "belief_mass_divergence"
   - "curl" / "vorticity" / "∇×F" / "circulation" → "d₁", "DEC curl", "exterior_derivative rank 1→2"
   - "boundary" / "∂M" / "frontier" / "perimeter" → "exterior_derivative", "d", "coboundary operator", "boundary_flux_mass"
   - "line integral" / "trajectory energy" / "path cost" / "geodesic cost" → "line_integral", "edge field sum", "rank-1 cochain path sum"
   - "Stokes theorem" / "divergence theorem" / "Green's theorem" / "Gauss" / "generalized Stokes" → "DEC identity d∘d=0", "curl(grad)=0", "div(curl)=0", "hodge_decompose"
   - "Hodge decomposition" / "exact/coexact/harmonic" / "Helmholtz" / "conservative/solenoidal" → "hodge_decompose", "DecFlowField", "exact_flow/coexact_flow/harmonic_flow"
   - "Fokker-Planck" / "continuity equation" / "mass conservation" / "probability flow" → "belief_mass_divergence", "codifferential on belief cochain"
   - "cell complex" / "mesh" / "simplicial" / "cubical" → "CellComplex", "CochainField", "grid_2d"

   **Standing per-NPC runtime / freeze-thaw / personality vocabulary (ALWAYS include when the paper touches per-entity state, memory, personality, swap, evaluator/judge/critic, or selective erasure/forgetting — the `riir-ai/.research/` guide corpus is SATURATED in this space, so paper-vocabulary-only greps produce false novelty claims; see R320 failure below):**
   - "personality swap" / "personality drift" / "character shift" / "persona change" → "committed personality", "freeze/thaw cadence", "direction vectors versioned via freeze/thaw", `ArchetypeBlendShard`, `KarcShard`
   - "selective forgetting" / "memory erasure on swap" / "dependent record invalidation" / "forget on swap" → "non-interference branches", "branch-local", `BranchBank`, "orthogonal subspace projection"
   - "survives swap" / "invariant to observation gaps" / "replay-deterministic personality" → "sampling invariance", "quorum-verifiable", "bit-identical across nodes", "FAME Proposition 3"
   - "epoch boundary" / "controlled utility evolution" / "checkpoint replacement" / "non-stationary utility" → "freeze/thaw cadence", "consolidation sleep-cycle", `tau_reest`, "re-estimation trigger", "coherence < tau"
   - "co-evolution" / "evaluator replacement" / "moving target" → "personality divergence", "direction vector drift", "emergent personality at crowd scale"
   - "evaluator" / "judge" / "critic" / "verifier" (per-entity) → "claim verifier", "CLR vote", "direction vector projection", "Salience Tri-Gate", `ConstraintPruner`
   - "frozen snapshot" / "frozen artifact" (generic) → NAME THE CONCRETE TYPE: `KarcShard`, `ArchetypeBlendShard`, `BranchBank` snapshot, `ZoneGeometryPod`, `MerkleFrozenEnvelope`, `SleepAnticipationShard`. Never use the generic "frozen snapshot" in a selling-point sentence — grep `riir-neuron-db/src/shard.rs` + `riir-neuron-db/src/freeze.rs` for the concrete subtype.
   - "cache invalidation on swap" / "dependent records" / "criterion consistency" → `DecCache::mark_face_destroyed`, `ZoneGeometryCache::invalidate(zone, new_version)`, `topology_version` bump, `SourceShardHashMismatch`

   Example (DiPOD paper → riir-ai code):
   - "double drift" / "ELBO drift" → "coherence decay", "staleness", "divergence"
   - "self-distillation" → "re-estimation", "re-derive", "recommit"
   - "tight bound" / "adequate estimator" → "coherence > tau", "parallelism quality", "confidence gate"
   - "policy-preserving" → "atomic Arc swap", "readers keep old snapshot"
   - "drop-in regularizer" → "feature flag", "warm-tier scheduler tick"

   Grep ONLY paper vocabulary → misses `latent_functor/reestimation.rs` (which ships DiPOD's exact pattern under the name "coherence-driven re-estimation scheduler"). Grep BOTH sets → hits it on the first pass. **Notes framing can use codebase vocabulary that a paper-vocabulary grep misses on BOTH layers — translate before grepping.**

   Example (Stokes/divergence-theorem paper → katgpt-rs DEC code):
   - "divergence" / "flux" / "density tracking" → "codifferential", "δ", "DEC divergence"
   - "boundary integral" / "CDF via boundary" / "surface flux" → "exterior_derivative", "d", "coboundary", "boundary_flux_mass"
   - "line integral" / "path energy" → "line_integral", "rank-1 cochain sum"
   - "Stokes theorem" / "∫_M dω = ∫_∂M ω" → "DEC identity d∘d=0", "curl_of_gradient_is_zero"
   - "Hodge decomposition" / "exact/coexact/harmonic" → "hodge_decompose", "DecFlowField"
   - "Fokker-Planck" / "continuity equation" → "belief_mass_divergence", "codifferential"

   Grep ONLY paper vocabulary → ZERO hits across all repos (a corpus grep for `stokes|divergence theorem|boundary integral|fokker-planck` returns nothing). Grep BOTH sets → hits `dec/operators.rs` (`codifferential`, `exterior_derivative`), `dec/hodge.rs` (`hodge_decompose`), `dec/flow.rs` (`DecFlowField`). The Generalized Stokes' theorem machinery ships as DEC operators, but no note framed it in Stokes-theorem vocabulary, so a paper-vocabulary grep missed BOTH notes AND code until the standing DEC vocabulary above was added.

3. **MANDATORY — latent-space reframing before verdict.** Before any verdict, re-cast the paper's core mechanism as a latent-to-latent operation on the codebase's latent-state kernels (the seven Super-GOAT factory modules above). Ask explicitly: "How does this mechanism look when operating on (a) HLA's per-NPC latent state, (b) `latent_functor/` operations, (c) `cgsp_runtime/` curiosity signals, (d) LatCal fixed-point commitment (in `riir-chain/src/encoding/`), (e) `NeuronShard` style_weights / dendritic branch / `MerkleFrozenEnvelope` / Raven consolidation / AnyRAG escalation (in `riir-neuron-db/src/`), (f) DEC Stokes-calculus operators (`katgpt-rs/crates/katgpt-core/src/dec/` — `exterior_derivative` d, `codifferential` δ, `hodge_decompose`, `DecFlowField` exact/coexact/harmonic)?" If your fusion idea only touches adapter routing / KV compression / speculative decode without a latent-state reframing, you are likely in GOAT territory and have probably missed the Super-GOAT angle. If you find yourself reaching for an adapter-routing framing, treat it as a symptom that the stronger latent-functor / HLA / neuron-shard / LatCal reframing is still unfound — adapter routing is the fallback, never the primary Super-GOAT framing. The latent reframing is mandatory even for papers that look pure-training/architecture — most have a latent subspace / stage-gating / persistence / memory-consolidation / manifold-geometry angle that lands in HLA/functor/neuron-shard/DEC.

4. **Zero grep hits ≠ novelty.** If your paper-vocabulary grep AND your codebase-vocabulary grep BOTH return zero hits, that is evidence of one of three things, in order of likelihood: (a) you are still using the wrong vocabulary — try a third semantic angle (e.g., grep for the *output behavior* like "swap when X" instead of the *mechanism name* like "tightness monitor"); (b) the mechanism is genuinely not shipped; (c) the mechanism is novel. Do NOT jump to (c). Default to (a): re-grep with at least one more semantic alternative before claiming "no prior art".
5. After finding the transferable primitive of *this* paper, list the 2–3 closest existing notes/plans **across all five repos** and ask: "what novel combination of this paper + note A + note B produces a capability none of them has alone?" Write that combination into the research note's §Distillation as a **Fusion** subsection, even if you don't plan it yet.
6. Verdict by the commercial strategy tiers (see §Cross-references for the strategy doc): **Super-GOAT** > GOAT > Gain > Pass (see §Verdict tiers below). **A fusion that produces a new capability class is a strong Super-GOAT candidate — check the novelty gate (§1.5).**
7. Create research `.md` at the right repo (see table above).

**File naming:** `{NNN}_{Short_Title_with_Underscores}.md` where NNN is the next free number (zero-padded to 3 digits, e.g. `239_`, `240_`). Check the folder first — numbers may be non-contiguous; pick the next free slot.

**Research note format** (see `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md` for a canonical example):

```markdown
# Research NNN: <Title>

> **Source:** <paper title + arxiv link + authors + date>
> **Date:** YYYY-MM-DD
> **Status:** Active | Done | Shelved
> **Related Research:** NNN (short note), ...
> **Related Plans:** NNN (short note), ...
> **Cross-ref (riir-ai / riir-chain / riir-neuron-db):** Research NNN, Plan NNN   ← only if cross-repo (game runtime → riir-ai; chain/LatCal → riir-chain; shards/freeze/consolidation/AnyRAG/vibe → riir-neuron-db)
> **Classification:** Public | Private   ← katgpt-rs notes are always Public

---

## TL;DR

<2-4 sentences: the distilled primitive, why it matters here, what it unblocks>

**Distilled for katgpt-rs (modelless, inference-time):**
<the transferable insight, stripped of training setup>

---

## 1. Paper Core Findings
...
## 2. Distillation
...
## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | Open primitive → katgpt-rs. **Architectural guide → riir-ai/.research/ (game runtime) OR riir-chain/.research/ (chain/LatCal) OR riir-neuron-db/.research/ (shards/freeze/consolidation/AnyRAG/vibe/Merkle)**. Plans → appropriate repo(s) as needed. |
| **GOAT** | Provable gain (latency/quality/security) over existing approach, but not a new class of capability. Promotes to default if it wins. | Plan + implement → appropriate repo. Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | Plan only, behind feature flag. |
| **Pass** | Not relevant to modelless/latent/freeze-thaw/runtime, OR training-only (→ riir-train note, stop). | One-line note. No files created in this session. |

**One-line reasoning required for each verdict.** For Super-GOAT: state the selling point explicitly.

**After the tier verdict, run the MOAT gate per domain (§1.6)** — a tier verdict without a domain-fit check can land a great primitive in the wrong repo and dilute the moat. The tier answers "how strong?"; the MOAT gate answers "does this strengthen THIS repo's moat, and is this the right repo?".
```

### 1.5. Novelty gate — is this Super-GOAT?

Before planning, score novelty. Ask all four:

1. **No prior art?** Grep `.research/` + `.plans/` across all repos AND grep the shipped code (`katgpt-rs/src/`, `katgpt-rs/crates/`, `riir-ai/crates/`, `riir-chain/src/`, `riir-chain/crates/`, `riir-neuron-db/src/`) for the primitive name and mechanism keywords. **You MUST grep BOTH paper vocabulary AND codebase-vocabulary alternatives (see §Workflow fusion protocol step 2 — vocabulary translation).** **Notes describe intent; code describes what shipped.** A mechanism can ship under either of two failure modes:
   - **No notes framing at all** — canonical example: HLA's `evolve_hla` (`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs`) is a per-NPC recurrent belief-state kernel with no `.research/` note framing it as such; missing it has historically caused false Super-GOAT claims.
   - **Notes framing uses different vocabulary than the paper** — canonical example: DiPOD's "interleave self-distillation when ELBO drifts" is shipped as `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` "coherence-driven re-estimation scheduler when coherence < tau_reest". The note DOES frame the mechanism, but using codebase vocabulary, so a paper-vocabulary grep misses it on BOTH notes AND code layers. This is strictly worse than the `evolve_hla` failure: even a diligent notes grep fails. **Vocabulary translation (fusion protocol step 2) is the only defense.**
   If the code already covers the mechanism → not novel, Gain at best. **This three-layer check (notes + code + vocabulary translation) is mandatory — notes-only is the #1 cause of false Super-GOAT claims; paper-vocabulary-only is the #2 cause; skipping the seven Super-GOAT factory modules is the #3 cause.**

   **Grep returns candidates; READING the candidates is mandatory.** A grep hit is a lead to follow, not a prior-art confirmation. When a grep hit's filename or first-line summary touches the candidate's selling-point space (per-NPC, memory, personality, swap, freeze/thaw, evaluator, critic, curiosity, test-time scaling, sleep-time, sub-goal), `read_file` the hit's TL;DR + §1 (selling point) BEFORE claiming novelty. Grepping `riir-ai/.research/`, seeing a filename match, and moving on is the failure mode: the guide frames the mechanism under different vocabulary, so the filename looks unrelated even though the content is exact prior art. **When the candidate selling point touches per-NPC + memory + personality + swap, the `riir-ai/.research/` corpus is saturated — grep it for `Per_NPC|Committed|Cognitive_Branch|Sub_Goal|Karc|CLR|Sleep_Time|Gain_Cost|Personality|Curiosity|Mind_Reading` and READ every hit's TL;DR before claiming novelty. Assume covered until proven otherwise.**
2. **New class of behavior?** Not better numbers, but something no incumbent can do (a new capability, not an optimization).
3. **Product selling point?** Can you finish the sentence: "Our NPCs/systems do X that no competitor can"? If you can't → Gain.
4. **Force multiplier?** Connects to ≥2 existing pillars/systems (check connection map in `.research/`). Solo novelty without integration = GOAT, not Super-GOAT.

**If YES to all 4 → verdict = Super-GOAT.** Mandatory outputs:
1. **Open primitive** → `katgpt-rs` (generic math, no game semantics).
2. **Architectural GUIDE** → the private selling-point doc. **Pick the repo by where the selling point lives**: `riir-ai/.research/NNN_*.md` for game-runtime / HLA / functor / self-learn selling points; `riir-chain/.research/NNN_*.md` for chain / LatCal / commitment / quorum / catchup / sync-bridge selling points (create folder on first use); `riir-neuron-db/.research/NNN_*.md` for shard / freeze envelope / consolidation / AnyRAG / vibe KG / Merkle tree / spectral init / shard compaction selling points (create folder on first use). If the selling point spans multiple repos (e.g., latent ops that cross the chain sync boundary via a shard commitment), create the primary guide in the repo that owns the boundary being crossed, and cross-reference from the others. The guide MUST include:
   - TL;DR with commercial value (the selling point in one sentence)
   - Distilled primitive (how the mechanism works modellessly)
   - Connection map (which existing systems it multiplies)
   - Latent vs raw boundary (what crosses sync, what stays local)
   - What stays private vs open
   - Validation protocol (how to prove it's Super-GOAT, not just hype)
   - Implementation priority table (P0–P3)
3. **Plan(s)** → `katgpt-rs/.plans/` (open) and/or `riir-ai/.plans/` (private runtime) and/or `riir-chain/.plans/` (private chain) and/or `riir-neuron-db/.plans/` (private shards).

**If NO to any → proceed to GOAT/Gain verdict.** Plan only, no guide.

> **Rule:** Super-GOAT ideas are the private IP moat. The open primitive is the adoption hook; the riir-ai/riir-chain/riir-neuron-db guide is the selling point. Never ship the guide publicly. Never skip the guide for a Super-GOAT — that's losing the knowledge.
>
> **No "candidate" escape hatch.** If you write "all 4 YES", "passes the novelty gate", or "Super-GOAT candidate" anywhere in a note (main verdict OR a fusion subsection), the mandatory outputs above apply **in this same session** — open primitive in katgpt-rs, **private guide (riir-ai OR riir-chain OR riir-neuron-db, by selling-point domain) created now**, plans as needed. The guide *contains* the validation protocol (G1–Gn gate), so you create it **before** running the gate, not after. Deferring the guide "until validation passes" inverts the order and silently drops the moat doc — this is the #1 way selling points leak into the public repo.
>
> If you are NOT confident enough to commit all 4 YES right now, **do not write "Super-GOAT candidate"**. Write "fusion idea — novelty TBD, needs Q1–Q4 check before verdict" and create an issue in `.issues/` to track the follow-up. "Candidate" is not a deferred-commitment escape hatch — it either triggers the guide now, or it gets downgraded to an issue.

### 1.6. MOAT gate per domain

The global verdict tiers (Super-GOAT / GOAT / Gain / Pass) measure *how strong* a contribution is. The **MOAT gate per domain** measures *whether a contribution strengthens THIS repo's moat*. A primitive can be a clean GOAT win yet contribute nothing to the moat if it lands outside the repo's pillar scope, or if a stronger latent reframing was missed. **Check the domain MOAT gate at verdict time — a mismatch means reroute to the correct repo.**

| Domain | MOAT contribution bar | In scope | Out of scope (reroute) |
|--------|----------------------|----------|----------------------|
| **`katgpt-rs`** (public engine) | **Paper-derived fundamental / principle / base-foundation primitive** that passes GOAT or Gain via fusion, with **promote/demote tracked per stack**. Aim: research-grade primitives the adoption funnel depends on. Each primitive ships behind a feature flag; the GOAT gate decides promote-to-default vs demote-loser per stack. | Transformer stack (layers, attention, KV cache, sampling, sparse / quant-aware **inference**, speculative decode, DDTree, MCTS, bandits, ConstraintPruners); **2D toy benchmark games** (bomber/go/monopoly/fft-arena) + their generic MCTS/bandit/CCE wiring; DEC/Stokes substrate; HLA kernel; sigmoid mechanics. | Product game wiring (→ riir-ai); chain commitment (→ riir-chain); shard internals (→ riir-neuron-db); trained weights (→ riir-train). |
| **`riir-ai`** (private runtime) | **Pillar-level or Super-GOAT**: fusion-GOAT / fusion-Gain that connects to ≥2 pillars, OR a new pillar candidate (sloppy-test winner). | **Adaptive / self-learn NPCs**, **reasoning pack** (P8), **MMORPG-scale** (20Hz tick, fog-of-war, zone attention, crowd MCGS), **3D game wiring**, freeze/thaw runtime, latent-to-latent ops on HLA/functor/cgsp state. | Generic transformer mechanics (→ katgpt-rs); chain transport (→ riir-chain); shard storage (→ riir-neuron-db); training methods (→ riir-train). |
| **`riir-chain`** (private chain) | **Pillar-level or Super-GOAT**: pillar 3 (riir-chain) amplifier, OR sync-boundary bridge novelty. | LatCal commitment, quorum/catchup, chain economics, asset lifecycle / forensic, DeFi programs, `riir-chaind`, the raw↔latent sync-boundary bridge. | Generic fixed-point math without commitment semantics (→ katgpt-rs); shard internals (→ riir-neuron-db). |
| **`riir-neuron-db`** (private shards) | **Pillar-level or Super-GOAT**: pillar 2 (riir-neuron-db) amplifier, OR shard/freeze/consolidation novelty. | `NeuronShard` layout, freeze/thaw envelope, Raven/δ-Mem consolidation, AnyRAG escalation, vibe KG triples, Merkle integrity, spectral init, shard compaction, dendritic branch. | Chain commitment of shards (→ riir-chain); runtime adapter swap (→ riir-ai). |
| **`riir-train`** (private training) | **Secondary moat**: training-method implementations + configs + trained weight assets (GPU-hours). Out of scope for THIS workflow — note "→ riir-train" and stop. | Adapter training, optimizers, loss functions, quant-aware **training**, DPO/GRPO/SFT pipelines, trained weight assets. | Inference-time / runtime / latent ops (→ katgpt-rs or riir-ai). |

**Pillar reference (riir-* repos):** the 9 sloppy-test winners live in `riir-ai/.docs/pillars/README.md` — **`read_file` `pillars/README.md` + `supergoat_candidates/README.md` before any "does this become a pillar?" MOAT verdict.** The 4-layer architecture (Foundation → AI → Emergent → Delivery, strict downward dependency) and the sloppy test (*if it doesn't exist, the system goes structurally sloppy — not slower, broken*) define what a pillar-level contribution means. The 9 pillars: (1) Egg/Shell + Bridge, (2) riir-neuron-db, (3) riir-chain, (4) Fourier Spatial AI, (5) WASM Validators, (6) NPC Dialog Engine, (7) Frame-Sampling Bridge, (8) Reasoning Pack, (9) Asset Vessel.

**MOAT verdict (per contribution):**
- **Strengthens moat** (in-scope pillar-level / Super-GOAT / fusion-GOAT connecting ≥2 pillars) → promote aggressively; if Super-GOAT, capture the private guide now (§1.5).
- **Neutral GOAT/Gain** (in-scope but not pillar-level) → ship behind feature flag, track promote/demote, do NOT overclaim moat in the note.
- **Out-of-scope** → reroute to the correct repo (5-repo discipline). A great primitive in the wrong repo dilutes the moat — e.g. a generic attention kernel merged into `riir-ai` instead of `katgpt-rs` leaks nothing privately valuable but starves the public adoption funnel.

**`katgpt-rs` promote/demote tracking (per stack):** every primitive that lands in the public engine gets a feature flag + benchmark + GOAT gate, and the verdict note MUST record the per-stack outcome — which transformer stack slot (attention / KV / sampling / speculative / pruning) and whether it promoted to default or stayed opt-in. Re-gate on feature touch. Demote the loser when a newer primitive wins the same slot. This per-stack ledger is the engine's quality contract.

### 2. If gain (or GOAT), plan it

Add plan `.md` to `katgpt-rs/.plans/` (modelless), `riir-ai/.plans/` (runtime/game), and/or `riir-chain/.plans/` (chain / LatCal / neuron_db). Use `## Phase N` sections with `- [ ]` per task (mark `- [x]` when done). **Never** plan into `riir-train` from this workflow.

> Super-GOAT plans should be created AFTER the riir-ai guide. The guide is the strategy; the plan is the execution.

**Plan format** (see `katgpt-rs/.plans/271_attention_matching_compaction.md` for a canonical example):

```markdown
# Plan NNN: <Title>

**Date:** YYYY-MM-DD
**Research:** [katgpt-rs/.research/NNN_*.md](../.research/NNN_*.md)
**Source paper:** [arxiv ID.NNN](https://arxiv.org/abs/ID) — <short cite>
**Target:** `katgpt-rs/src/<module>/` (new module) + Cargo feature `<feature_name>`
**Status:** Active — Phase N <state>

---

## Goal

<one paragraph: what ships, what it enables, GOAT gate>

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** <concrete, verifiable task>
- [ ] **T1.2** ...
```

**GOAT gate rule** (AGENTS.md): every plan that introduces a new technique must have a feature flag and a benchmark proving the gain before promoting to default. Demote the loser if the new technique wins.

**UQ-bearing primitive GOAT gate extension (the "Report the Floor" rule, adopted 2026-06-28 per Research 322):** Any primitive that claims a probability distribution, predictive interval, quantile, coverage guarantee, confidence score, or calibrated uncertainty (collectively: **UQ-bearing**) MUST include a benchmark against the **conformal-naive floor** in its GOAT gate. The floor is `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340 with `m=1`, which degenerates to plain split conformal prediction over 1-step residuals). The primitive must beat the floor on CRPS / empirical coverage / Winkler interval score (whichever applies). If it cannot beat the floor, it is not a UQ primitive — it is noise, and the GOAT gate FAILS.

**Retroactive application (existing UQ-bearing primitives; per-policy retroactive audit COMPLETE 2026-06-30, consolidated in `.benchmarks/010_report_the_floor_consolidated.md`):** BoMSampler (Plan 281), Sleep-Time Query Anticipator (Plan 334), Best-Belief Beta Selector (Plan 336), and KARC+conformal-overlay (Plan 308+340) were grandfathered at their current promotion state and have all passed their floor comparison. Future UQ-bearing primitives MUST include the floor from initial GOAT gate (no grandfathering). The floor shipped in Plan 340 Phase 1 (2026-06-30); the rule is now **enforceable**.

**Why:** Manokhin's companion paper *Report the Floor* (arXiv:2606.09473) proves that a trivial training-free conformal interval is a mandatory baseline — any probabilistic forecaster that can't beat it is not adding value over the floor. Adopting this as policy prevents future UQ claims that are actually just the floor in disguise.

### 3. Implement to unblock

If a plan is blocked by a missing primitive, implement the minimal version. After GOAT check + proof of gain: promote to default if it wins, demote the loser.

### 3.5. Modelless unblock protocol — MANDATORY before any riir-train deferral

**Hard rule:** before deferring ANY GOAT gate, plan task, or mechanism to riir-train ("this needs training"), you MUST exhaust all modelless correction paths first. A gate that *appears* to need training may be passable modellessly via freeze/thaw, raw/lora hot-swap, or latent-space correction. Deferring to riir-train without checking is the failure mode this protocol prevents.

**The three modelless unblock paths (check ALL before deferring):**

1. **Freeze/thaw snapshot correction** (`riir-neuron-db/src/freeze.rs`, `MerkleFrozenEnvelope`) — can a frozen snapshot state, thawed at inference, fix the issue? If the failure is a systematic bias from a runtime construction (e.g., doubled signal, position mismatch, attention pattern asymmetry), a corrected snapshot + thaw may eliminate it without gradient descent.
2. **Raw/lora reader-writer hot-swap** (`LoraPair { reader, writer }`, Plan 025; `LoRAHotSwap`, `dispatch_lora_merge` in riir-ai) — can a **deterministically constructed** (not trained) reader or writer adapter fix the issue? Applying a pre-constructed LoRA overlay is modelless (weight addition, no backprop). The question is: can the correction be derived in closed form (e.g., scale-by-0.5, zero-out-specific-positions, identity-minus-projection) rather than learned via gradient descent?
3. **Latent-space correction** (dot-product projection + sigmoid gate, per constraint #2) — can the bias be corrected by projecting the latent state onto a correction direction and gating the output? This is the modelless analog of a trained adapter: instead of learning the correction, derive it analytically from the failure mode.

**Decision protocol:**

```
Gate/mechanism appears to need training
  → Does the failure have a SYSTEMATIC, characterizable cause (e.g., "signal doubled", "position offset", "attention asymmetry")?
    NO → genuine riir-train dependency. Note "→ riir-train", stop.
    YES → Can freeze/thaw (path 1) fix it? Check the freeze envelope API.
      NO → Can a deterministically constructed reader/writer LoRA (path 2) fix it? Check the LoraPair API.
        NO → Can a latent-space projection/gate (path 3) fix it?
          NO → genuine riir-train dependency. Note "→ riir-train", stop.
          YES → implement the latent correction modellessly. Gate is MODELLESS-VALIDABLE.
        YES → construct the LoRA correction modellessly. Gate is MODELLESS-VALIDABLE.
      YES → freeze the corrected state. Gate is MODELLESS-VALIDABLE.
    
  → MODELLESS-VALIDABLE gates must be implemented and checked BEFORE any riir-train deferral.
  → If all three paths fail, THEN note "→ riir-train" with explicit documentation of WHY each path failed.
```

**Documentation requirement:** every "→ riir-train" deferral MUST include:
- Which of the three modelless paths were checked.
- Why each failed (concrete reason, not "doesn't apply").
- What specifically requires gradient descent that no deterministic construction can provide.

### 3.6. Defend-wrong PoC for parity claims — MANDATORY before any "already ships" / "parity" verdict

**Hard rule:** before claiming in a verdict that a paper's mechanism "already ships" modellessly, achieves "parity" with the paper, or that the runtime analog "covers" the paper's loop, you MUST distinguish three claim types and prove each at the level it requires:

| Claim type | Example | Proof required |
|---|---|---|
| **Architectural** ("the runtime analog exists") | "the plan-execute-adapt-replan loop ships as `ReestimationScheduler`" | grep + read the code (sufficient) |
| **Latency / resource** ("modelless, sub-µs, no GD") | "adaptation overhead is +30 ns" | criterion bench |
| **Quality** ("matches / beats the paper's numbers") | "recovers planning success under shift as well as the paper's loop" | **head-to-head PoC on a controlled toy benchmark — architectural reasoning is NOT sufficient** |

**The failure mode this prevents:** claiming all three with only architectural evidence. Architectural coverage does NOT imply quality parity — the shipped version may have tuning gaps, divergence modes, or trigger thresholds that make it underperform the paper on the paper's own task. A grep proves the mechanism exists; it does not prove the mechanism *works as well as the paper's version*.

**When a PoC is mandatory:**
- Any verdict that asserts quality parity ("matches", "competitive with", "recovers as well as", "covers the paper's loop at parity").
- Any Super-GOAT/GOAT claim where the gain is qualitative ("recovers from distribution shift", "matches paper's success rate").
- **Any PASS verdict that downgrades a paper on the grounds that "the runtime analog already ships"** — the downgrade is only honest if the analog actually performs. A PASS verdict backed only by architectural reasoning is the #1 false-PASS failure mode.

**When a PoC is NOT required:**
- Pure architectural redirects (paper X is a refinement of shipped primitive Y, no quality claim).
- Training-only redirects (→ riir-train, no parity claim).
- Latency-only claims (a single criterion bench suffices, no full PoC).
- Low-confident verdicts that explicitly mark the quality claim as unproven and create a `.issues/` entry to track the PoC follow-up.

**Where the PoC lives:** `riir-ai/crates/riir-poc/` — the "defend-wrong" R&D crate. It exists for exactly this: empirical settlements of disputed primitives before any verdict becomes a feature flag. A PoC has three competitors minimum: the paper's mechanism (or its distilled modelless analog), a frozen/no-adaptation baseline, and the shipped runtime analog. Run them head-to-head on a controlled toy domain (no training), print a verdict table. Use `CARGO_TARGET_DIR=/tmp/...` per the AGENTS.md rule and clean up when done.

**The PoC's job is to defend OR refute.** A PoC that only confirms the verdict is weaker than one that honestly refutes part of it. If the PoC refutes the quality claim:
1. **Do NOT silently revise the verdict to match the PoC.** Record the raw numbers in the research note as a §"PoC Addendum" section.
2. **Honest revision:** explicitly state which claim type was confirmed (architectural, latency) and which was refuted (quality). The verdict stands on the confirmed axes; the refuted axis becomes a tracked follow-up (issue in `.issues/`).
3. **The PoC stays as a permanent regression check** in `riir-poc` — its job was to settle the dispute, and it should keep settling it if the shipped primitive is later tuned.

**Canonical example (Research 360, AdaJEPA, 2026-07-01):** the verdict claimed "parity" between the shipped `ReestimationScheduler` and AdaJEPA's per-MPC-step GD loop, based on architectural coverage alone. The PoC at `riir-ai/crates/riir-poc/benches/adajepa_modelless_goat.rs` confirmed latency parity (~940 ns/replan) and architectural coverage, but **refuted quality parity** — the coherence trigger was too conservative for mild shifts (0 updates at a mild shift), and all adaptation strategies diverged on overshoot shifts. The verdict was honestly revised in a §9 PoC Addendum; the follow-ups are tracked in `riir-ai/.issues/363`. This is the canonical "architectural coverage ≠ quality parity" lesson — grep proved the loop existed, the PoC proved it didn't perform.

### 4. Search if curious

Keyword search arxiv:

```
https://r.jina.ai/https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term={KEYWORD}&terms-0-field=abstract&classification-computer_science=y&classification-mathematics=y&classification-physics_archives=all&classification-statistics=y&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first
```

Good keywords: `latent space routing`, `adapter hot-swap`, `inference-time composition`, `spectral pruning`, `sigmoid gating`, `snapshot consistency`, `lock-free weight swap`.

## Constraints (non-negotiable)

1. **Modelless first** — inference-time only. No LLM training, no backprop through base weights. Closest to "training" allowed: freeze/thaw snapshot cycles, raw/lora reader-writer hot-swap with **deterministically constructed** adapters (not trained), and latent-space direction-vector updates at runtime. **Before deferring any gate/mechanism to riir-train, exhaust §3.5 modelless unblock paths (freeze/thaw, raw/lora, latent correction).**
2. **Latent-to-latent preferred** — operate in embedding/latent space as long as possible. Decode to tokens or project to raw scalars only at the boundary. Use dot-product + **sigmoid** (never softmax) for projections onto learned direction vectors. Semantic domain (emotion, mood, curiosity, style) → latent. Physical domain (position, HP, wallet balance) → raw, deterministic, synced.
3. **Freeze/thaw over fine-tuning** — the only weight mutation allowed at runtime is swapping a frozen snapshot (atomic, versioned, BLAKE3-checked) or applying a deterministically constructed LoRA overlay (raw/lora hot-swap, no gradient descent). Never mutate weights in-place during inference. If a paper needs gradient updates (after exhausting §3.5 modelless paths), redirect to riir-train.
4. **Self-learn / adaptive CoT welcome** — runtime curiosity, latent prediction, trajectory folding, collapse detection. These update latent state / direction vectors / routing tables, NOT base weights.
5. **5-repo discipline** — katgpt-rs (public engine) → riir-ai (private runtime/game) → riir-chain (private chain) → riir-neuron-db (private neuron-shard leaf) → riir-train (private training). Keep the commercial strategy intact. Training know-how never leaks to katgpt-rs; chain IP stays in `riir-chain/`, not `riir-ai/`; neuron-shard IP stays in `riir-neuron-db/`, not `riir-chain/` (chain only re-exports via the `neuron_db` feature).
6. **SOLID, DRY** — per `katgpt-rs/.contexts/optimization.md`. Zero-allocation hot paths. Pre-computed lookup tables. Fixed-size arrays for bounded domains.
7. **Tests/examples** — before/after showing the gain (latency, quality, or security). For latent ops: show the projection preserves ranking. For freeze/thaw: show readers never see torn snapshots.
8. **CPU/GPU/ANE auto-route** — threshold-adaptive dispatch. Plasma (µs, CPU/SIMD) → Hot (sub-ms, GPU) → Warm/Cold (ms+, GPU/ANE). Latent ops that fit in L1 cache stay on SIMD; manifold ops that need batched matmul go to GPU.
9. **Plasma → Hot → Warm → Cold → Freeze tiering** — aim for perf on game side (plasma/hot latency budget) AND security on chain side (cold/freeze commitment, BLAKE3-hashed, tamper-evident). Latent state that crosses the sync boundary MUST be raw scalars (valence/arousal/desperation/calm/fear), never the full embedding vector.

## Latent vs raw space rules (critical for game AI)

Reinforce these when designing game systems or chain state:

- **Physical domain** (position, velocity, HP, wallet balance): MUST remain raw exact values. Deterministic replay, quorum sync, anti-cheat require bit-identical reconstruction.
- **Semantic domain** (emotion, mood, curiosity, style, habit): SHOULD operate in latent space via dot-product + sigmoid onto learned direction vectors.
- **Social domain** (encounters, relationships, factions): SHOULD produce KG triples from proximity in latent/embedding space, not from raw coordinate distance.

**Sync boundary:** if data flows through `SyncBlock → ChainConsensus` quorum commit → Cold tier, it MUST be raw and deterministic. If data is consumed locally (emotion projection, shard retrieval, consolidation sleep-cycle), it SHOULD be latent. Bridge functions (raw→latent projection, latent→raw scalar clamp) MUST be zero-allocation, gateable by feature flag, and not introduce sync dependency.

**KG triple emission:** semantic encounters → KG triple from latent similarity. Physical events → TxDelta with raw values, NOT KG triple. Never substitute latent embedding for raw position in anti-cheat validation.

**Spatial cognition (two-brain model):** info brain = real `MapPos` (synced, ground truth). Think brain = per-NPC `SpatialBelief` (zone-level KG triple + stale last_known_pos, fog-of-war gated, NOT synced). Bridge is one-way: real position → belief update only when within `visible_radius`. Confidence decay: `sigmoid(-λ * (current_tick - last_observed_tick))`. Two brains MUST exist independently — divergence is emergent behavior, not a bug.

## Cross-references (read on demand)

**Commercial strategy / moat map:**
- The inline short version of the commercial strategy lives in §"Commercial strategy — inline short version" above (5-repo roles, tier model, What/How rule, benchmark exception, asymmetric cognitive moat, why-hard-to-replicate, FV moat). No external doc lookup needed for routing decisions.
- `riir-ai/.docs/README.md` (+ `pillars/README.md`, `supergoat_candidates/README.md`) — the **live moat map by capability**. Read these for any Super-GOAT novelty gate or "does this become a pillar?" MOAT-gate question (§1.6). The full internal strategy doc with exhaustive moat analysis lives at `riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md` (commercially sensitive — read only when the inline short version is insufficient).

**Other reference docs:**
- `katgpt-rs/.contexts/optimization.md` — perf rules (zero-alloc, SIMD, rayon, caching)
- `katgpt-rs/.contexts/ibraheemdev-papaya-v0.2.3-examples.md` — papaya lock-free hashmap usage
- `katgpt-rs/.research/004_LoRA_Architecture_Verdict.md` — LoRA / validator terminology
- `katgpt-rs/.research/005_Artifact_Definition.md` — artifact terminology
- `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md` — canonical research-note example
- `katgpt-rs/.plans/271_attention_matching_compaction.md` — canonical plan example
- `riir-chain/AGENTS.md` — repo-local context for the chain repo
- `riir-neuron-db/AGENTS.md` — repo-local context for the neuron-db repo
- `katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md` — DEC operators distillation (the parent note that shipped the Stokes substrate)
- `katgpt-rs/.research/271_MIT_6S184_Diffusion_Flow_Textbook_Vocabulary_Crosswalk.md` — diffusion/flow vocabulary crosswalk (also flags Fokker-Planck as a known gap, closed by Research 296)
- `katgpt-rs/.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md` — Stokes/Divergence/Hodge vocabulary crosswalk + GOAT verdict for the three wrapper primitives
- `katgpt-rs/.plans/251_dec_operators_cell_complex.md` — DEC operators (COMPLETE — ships `d`, `δ`, `Δ`, `hodge_decompose`, `DecFlowField`)
- `katgpt-rs/.plans/314_stokes_calculus_wrappers.md` — Stokes-calculus wrapper primitives (`belief_mass_divergence`, `boundary_flux_mass`, `line_integral`)

## TL;DR

This skill packages the katgpt-rs research workflow: **MANDATORY pre-flight: `read_file` the 4 READMEs (`katgpt-rs/README.md`, `riir-ai/README.md`, `riir-chain/README.md`, `riir-neuron-db/README.md`) + `riir-ai/.docs/README.md` (the consolidated selling-point book — `pillars/` is the live moat map; read `pillars/README.md` + `supergoat_candidates/README.md` before any Super-GOAT novelty gate), AND `list_directory` all four `.research/` folders (`katgpt-rs/.research/`, `riir-ai/.research/`, `riir-chain/.research/`, `riir-neuron-db/.research/` — create the last two on first use) AND the four runtime/chain/neuron-db crate src trees (`riir-ai/crates/riir-engine/src/`, `riir-ai/crates/riir-games/src/`, `riir-chain/src/`, `riir-neuron-db/src/`) before any verdict** → read paper → classify (training? → riir-train, stop) → **distill + fuse** (find the transferable primitive, then **vocabulary-translate** paper terms to ≥2 codebase-equivalent terms each INCLUDING the standing latent-state vocabulary + the standing DEC/Stokes vocabulary, then grep BOTH layers — `.research/`+`.plans/`+`.docs/` for intent AND `src/`+`crates/` for shipped code — across all five repos, using BOTH paper vocabulary AND codebase vocabulary, AND `list_directory` the **seven Super-GOAT factory modules** explicitly: `katgpt-rs/crates/katgpt-core/src/sense/` (HLA), `riir-ai/crates/riir-engine/src/latent_functor/`, `riir-ai/crates/riir-engine/src/hla/`, `riir-ai/crates/riir-engine/src/cgsp_runtime/`, `riir-neuron-db/src/` (shards/freeze/consolidation/AnyRAG/vibe/Merkle), `riir-chain/src/encoding/latcal*.rs` (LatCal), `katgpt-rs/crates/katgpt-core/src/dec/` (Stokes/exterior-derivative/Hodge — maps any divergence/boundary/line-integral/Fokker-Planck/manifold-geometry paper) — for the 2–3 closest cousins to synthesize a novel combination) → **MANDATORY latent-space reframing before verdict** (re-cast the mechanism as a latent-to-latent op on HLA/functor/cgsp/neuron-shard/LatCal/DEC state; adapter routing / KV compression / speculative decode are GOAT-tier fallback framings, NOT the primary) → **novelty gate** (Super-GOAT? → open primitive + private riir-ai or riir-chain or riir-neuron-db guide depending on whether the selling point is game runtime, chain transport, or neuron-shard storage; else GOAT/Gain → plan only) → **MOAT gate per domain (§1.6)** — tier answers "how strong?", MOAT gate answers "does this strengthen THIS repo's moat, and is this the right repo?". `katgpt-rs` MOAT = paper-derived fundamental/principle primitive passing GOAT/Gain via fusion, promote/demote tracked per stack (transformer stack + 2D toy games); `riir-*` MOAT = pillar-level / fusion-GOAT / Super-GOAT (adaptive, reasoning, MMORPG, 3D, chain, shards, training). **Zero grep hits ≠ novelty — try one more semantic angle before claiming "no prior art"** → implement behind feature flag → benchmark → promote GOAT or demote loser. Hard constraints: modelless-first, latent-to-latent with sigmoid (never softmax), freeze/thaw over fine-tuning, 5-repo commercial discipline (public engine / private runtime / private chain / private neuron-db / private training) per the inline commercial strategy short version (§"Commercial strategy — inline short version" — tier model + What/How + benchmark exception + asymmetric cognitive moat, no external doc lookup needed), raw scalars at the sync boundary, **fusion-first mindset** (the best Super-GOATs come from fusing papers across all five repos, not direct-mapping one). **Super-GOAT = private moat; never skip the riir-ai/riir-chain/riir-neuron-db guide. Never grep only katgpt-rs — riir-ai + riir-chain + riir-neuron-db are three-fifths of the corpus. Never grep only notes — code is half the prior art, `.docs/` is the moat framing. Never grep only paper vocabulary — codebase vocabulary is the other half (DiPOD's "self-distillation when ELBO drifts" ships as `latent_functor/reestimation.rs` "coherence-driven re-estimation scheduler when coherence < tau_reest"; missed by paper-vocabulary grep on BOTH notes AND code, even though Research 123 + Plan 303 frame the mechanism under codebase vocabulary). Never default to adapter routing when a latent-functor/HLA/neuron-shard/LatCal reframing is available — that is the adapter-routing-default failure mode. Never grep-and-move-on when a hit touches per-NPC + memory + personality + swap — the `riir-ai/.research/` corpus is saturated there; `read_file` the TL;DR before claiming novelty. The standing vocabulary blocks (latent-state, DEC, per-NPC runtime) and the read-the-hits rule are the prophylactics. **Parity / "already ships" quality claims require a defend-wrong PoC in `riir-ai/crates/riir-poc/` (§3.6) — architectural coverage ≠ quality parity; grep proves the mechanism exists, not that it works as well as the paper. A PASS verdict backed only by architectural reasoning is the #1 false-PASS failure mode; the PoC job is to defend OR refute, and a refutation that produces an honest §PoC Addendum + a tracked `.issues/` follow-up is a successful PoC.**
