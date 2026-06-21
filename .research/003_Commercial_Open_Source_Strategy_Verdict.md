# Commercial Strategy — Engine, Runtime, Chain, Neuron-DB, Training Split

**Date:** 2026-06 (revised 2026-06-22 for `riir-neuron-db` spin-off)
**Status:** Active
**Purpose:** Guide for AI agents to reason about what is public vs internal when creating research, plans, and docs.

> ⚠️ **This doc is PUBLIC** (lives in `katgpt-rs`, MIT licensed). Every line is visible to the world. Describe **capabilities**, never implementation details. When in doubt, cut it.

---

## Revision history

- **2026-06 (initial):** 3-repo split (`katgpt-rs` / `riir-ai` / `riir-train`).
- **2026-06-21:** `riir-chain` spun off from `riir-ai/crates/riir-chain` + `riir-ai/crates/riir-chaind` into its own standalone repo. Chain IP moves from `riir-ai` to `riir-chain`. The engine/runtime/training split rationale is unchanged; only the chain boundary is sharpened. See `riir-chain/.plans/001_chain_spinoff.md` and `riir-chain/.plans/002_chaind_spinoff.md`.
- **2026-06-22:** `riir-neuron-db` spun off from `riir-chain/src/neuron_db/` (plus the generic `MerkleTree` from `riir-chain/src/catchup/merkle.rs`) into its own standalone leaf crate. Neuron-shard IP moves from `riir-chain` to `riir-neuron-db`; `riir-chain` re-exports it under its `neuron_db` feature. The engine/runtime/chain/training split rationale is unchanged; only the shard boundary is sharpened. See `riir-neuron-db/.plans/001_extraction_from_riir_chain.md`.

---

## The Boundary

Five repos. The split is absolute.

| Repo | License | Role |
|------|---------|------|
| `katgpt-rs` | MIT (public) | **Engine** — generic inference framework. Adoption funnel. No game IP, no chain IP, no neuron-shard IP. |
| `riir-ai` | Private (internal) | **Game product** — freeze/thaw runtime, self-learn/adaptive NPCs, latent-space operations, game systems. The ship-focus repo for gameplay. |
| `riir-chain` | Private (internal) | **Neuro-symbolic chain transport** — co-located AI+wallet state, LatCal (Lattice Calculus) encoding, split-key ledger, chain economics, DeFi programs, `riir-chaind` daemon, validator SDK bridges, `catchup/` (Turso/libSQL persistence, quorum), `DataTier` / `DATA_TIERS` / `build_tier_root`. Re-exports `riir-neuron-db` under its `neuron_db` feature. |
| `riir-neuron-db` | Private (internal) | **Neuron-shard leaf crate** — `NeuronShard` weight blob (fixed-layout Pod, zero-copy mmap), `ShardIndex` lock-free lookup, generic `MerkleTree`/`MerkleProof`, `MerkleFrozenEnvelope` (freeze/thaw integrity), MAPE-K self-healing loop, Raven/δ-Mem consolidation, AnyRAG escalation gateway, vibe KG triple templates. No chain dependency — usable standalone. |
| `riir-train` | Private (internal) | **Training research** — adapter training methods, training data, trained weights. Know-how vault. |

**Rule: anything `riir-*` is internal. No exceptions. No per-crate deliberation.**

(`riir-armageddon/` is a sixth repo for arena/game-product domain types only — it is not a distillation target. Read its README for the raw-vs-latent boundary; do not put research or chain code there.)

### Why five repos (not three, not four) — the chain + neuron-db spin-offs

Chain transport (`LatCal`, `chaind`, chain economics, asset lifecycle / forensic fingerprinting, `catchup/`) grew large enough to warrant its own workspace with its own `Cargo.toml`, feature flags, and 60+ GOAT-gated umbrella features (`chain`, `chain_economics`, `chain_solana_parity`, `chain_catchup`, `chain_asset_*`, `shard_compactor`, `lora_posterior`). Keeping it under `riir-ai/crates/` conflated two distinct product surfaces (gameplay runtime vs ledger transport) and made CI feature-guard testing awkward.

Neuron shards (`NeuronShard`, `ShardIndex`, `MerkleFrozenEnvelope`, MAPE-K self-healing, Raven/δ-Mem consolidation, AnyRAG gateway, vibe KG triples, spectral init, `ShardCompactor`, dendritic LoRA branch) then grew large enough *and* turned out to have **zero hard chain dependency** — shards are usable standalone for any consumer that needs frozen weight blobs with integrity envelopes, not just chain validators. Keeping them inside `riir-chain/src/neuron_db/` conflated two distinct concerns (chain transport vs weight storage) and forced every shard consumer to drag in chain types. The leaf extraction lets `riir-ai` depend on `riir-neuron-db` directly for shard access without going through `riir-chain`.

The split keeps the **sync boundary** sharper: chain IP is the only thing that crosses the `SyncBlock → ChainConsensus → Cold tier` commitment path. Neuron shards are *committed by* the chain (BLAKE3-hashed into cold tier) but their *structure, integrity envelope, consolidation, and retrieval* are chain-agnostic. Gameplay runtime never touches the commitment path directly — it bridges to chain via scalar commitments from the latent side, and bridges to neuron-db via shard reads. Separating the three repos makes those boundaries physical, not just conceptual.

**Routing rule of thumb (chain vs neuron-db):** if the mechanism is about *how a shard is structured, committed-at-rest, frozen, consolidated, retrieved, or projected* → `riir-neuron-db`. If it is about *how a shard is committed to a chain block, transported across quorum, or bridged to LatCal fixed-point* → `riir-chain`. The `LatCalWalletExt` trait (typed wallet accessors on `NeuronShard` using `LatCalMatrix`) stays in `riir-chain` because it IS the bridge.

### Why we still need separate private repos (and why LoRA training is NOT the focus)

We benchmarked 90+ adapter-training methods. **Most produced stability proofs (training converges, no NaN) rather than quality proofs (the adapter actually plays measurably better).** The few real, reproducible gains came from **runtime adapter selection (routing)** — not from the training method that produced the adapter.

**The lesson:** the value of an adapted NPC brain is in *which frozen adapter you swap in at runtime*, not in *how that adapter was trained*. Training method choice is a second-order effect; routing and runtime composition are first-order.

So we pivoted:
- **Training research** moved to `riir-train` (still a moat — implementations + configs took years). But it's no longer the headline.
- **`riir-ai` ships**: freeze/thaw runtime (swap frozen adapters at runtime), latent-to-latent operations (compute NPC decisions in embedding space, never decode-then-re-encode), and self-learn/adaptive NPCs (runtime improvement, no offline training round-trip).
- **`riir-chain` ships**: neuro-symbolic chain transport — co-located AI+wallet state, latent-encoded balances, split-key transactions, LatCal commitment bridges, self-healing ledger, chain economics, DeFi programs.
- **`riir-neuron-db` ships**: standalone neuron weight shards with BLAKE3/Merkle integrity envelopes, freeze/thaw snapshots, Raven/δ-Mem consolidation, AnyRAG escalation gateway, vibe KG triple templates. Consumed by both `riir-ai` (game runtime reads shards) and `riir-chain` (chain commits shards to cold tier).
- **`riir-ai`, `riir-chain`, and `riir-neuron-db` never name a training method.** They expose generic training interfaces (traits); `riir-train` implements them. Zero-cost abstraction.

---

## Why katgpt-rs Is Public (MIT)

`katgpt-rs` is the **generic inference framework** — DDTree, ConstraintPruner trait, bandit, pruners infra, speculative decode. These are the hooks that make game devs depend on the engine.

**Why public:**
- It's the adoption funnel. Developers build on the engine, then need the platform.
- The engine alone produces no useful game AI output — it's a runtime, not the intelligence.
- MIT attracts contributors and creates dependency without exposing know-how.
- No legal friction for enterprise adoption.

**"Ferrari, no gas":** `katgpt-rs` is the open Ferrari. Without the private platform, it runs but produces nothing competitive. The gas is inside `riir-ai` (gameplay) and `riir-chain` (ledger).

---

## What riir-ai Can Do (Capabilities — Not How)

`riir-ai` is the game platform. Below is **what it can do**, not how it's built. The how stays internal.

### Freeze/Thaw Runtime (the product value)

NPC personality adapters are **frozen, immutable, version-checked blobs**. The runtime swaps them without downtime.

- **Versioned adapter snapshots** — lock-free reads (readers never block), atomic writes with proven memory-ordering guarantees (readers never see a torn / half-updated snapshot)
- **Runtime hot-swap** — reload NPC personality adapters at runtime with integrity verification, zero downtime, no game pause
- **Per-NPC personality versioning** — each NPC can hold a different adapter snapshot version, so two NPCs of the same type can diverge behaviorally over time
- **Fused adapter inference** — base weights + adapter delta merged in a single kernel pass; no separate forward pass for the adapter
- **Dynamic adapter routing** — select between frozen adapters by game state and objective (100% win rate vs 0% static routing in arena; this is runtime routing, not training)

### Latent-Space Operations (the efficiency multiplier)

NPC cognition operates **latent-to-latent** wherever possible — dot-products in embedding space, never decode-to-token-then-re-encode.

- **Latent-to-latent routing** — NPC decisions (emotion, mood, curiosity, aggression) computed as projections in embedding space
- **Sigmoid-gated scalar projections** — bounded outputs (valence, arousal, desperation, calm, fear) projected from latent state via dot-product + sigmoid; never softmax (preserves signal independence)
- **Raw ↔ latent bridge** — physical domain (position, HP, wallet balance) stays raw and deterministic for sync/replay/anti-cheat; semantic domain (emotion, relationships, habits) operates in latent space; bridge functions are zero-allocation, one-way where possible
- **Manifold retrieval** — proximity-based recall (similarity threshold) instead of coordinate distance; knowledge-graph triples emitted from latent similarity, not raw position checks

**Why latent-to-latent over training:** training has high cost (GPU-hours) and uncertain payoff (see LoRA lesson above). Latent operations are inference-time, composable, deterministic, and run on any backend (CPU SIMD → GPU → ANE) without retraining. The freeze/thaw cycle gives weight-level adaptivity; latent operations give decision-level adaptivity. Together they cover the design space that training was supposed to cover, at a fraction of the cost.

### Self-Learn / Adaptive NPCs (the selling point)

- **Self-play**: learns game strategy without external oracle
- **All-goals learning**: NPCs learn every objective at once, no curriculum hand-holding
- **Open-ended policy gradient**: emergent NPC behavior from runtime exploration
- **Collapse-aware recovery**: detect and recover from mode collapse mid-session
- **Trajectory folding**: large reduction in redundant self-play moves
- **Curiosity-driven exploration**: entropy-driven information gathering without a reward oracle

This is what ships in the game. The capability that turns scripted NPCs into living ones — and it runs at runtime, not during offline training.

### Arena Proofs (Outcomes, Not Methods)

- Adaptive AI: large win-rate improvement vs baseline across multiple game types
- Game-theory-optimal play achieved in turn-based tactics (99% win rate)
- Frame-sampling: 939K decisions/sec
- Dynamic routing: 100% win rate vs 0% static routing (runtime adapter selection)
- Adaptive reasoning: +177% quality on hard queries at ≤50% cost
- Browser NPC inference: sub-µs-per-call brain forward pass (WASM SIMD 5.47-7.27× over scalar)

---

## What riir-chain Can Do (Capabilities — Not How)

`riir-chain` is the neuro-symbolic chain transport. Below is **what it can do**, not how it's built. (Neuron-shard storage itself is provided by `riir-neuron-db`; `riir-chain` re-exports it under the `neuron_db` feature and adds the chain-commitment path on top.)

### Co-Located AI + Wallet State

- **AI state and wallet state co-located** in the same zero-copy data structure
- **Latent-encoded balances** — not plaintext integers, tamper-resistant by inspection
- **Split-key transactions** — neither party holds the combined key in transit
- **Self-healing chain** — detects anomalies and repairs automatically
- **Five-tier memory** with graceful degradation — engine never fails to boot
- **Cross-chain bridge** to Solana
- **Full DeFi economy**: gas, rent, slashing
- **9 GOAT proofs**: roundtrip fidelity, key security, pipeline throughput, tamper rejection

### LatCal — Lattice Calculus (the sync-boundary bridge)

- **Deterministic commitment** — latent-side decisions become raw committed scalars at the chain boundary
- **Fixed-point arithmetic bridges** — no float non-determinism across nodes
- **Batch validation** — determinant-checked matrix arithmetic for high-throughput verification
- **Raw ↔ latent bridge protocol** — what crosses the chain sync is always raw scalar (valence/arousal/desperation/calm/fear + signed deltas), never the full latent embedding vector
- **DeFi programs** — gas/rent/stake primitives composed from the same fixed-point arithmetic

### `riir-chaind` Daemon

- Validator node runtime
- Snapshot sync, congestion control, upgradeable programs
- Asset lifecycle / forensic fingerprinting for tamper-evident on-chain assets

---

## What riir-neuron-db Can Do (Capabilities — Not How)

`riir-neuron-db` is the standalone neuron-shard leaf crate (no chain dependency). Below is **what it can do**, not how it's built.

### NeuronShard — Frozen Weight Blob with Integrity

- **Fixed-layout weight blob** — `#[repr(C)]` Pod struct, zero-copy mmap, cross-platform deterministic byte layout (every field offset pinned)
- **BLAKE3 content commitment** — every shard carries its own hash; tamper-evident by inspection
- **Generic Merkle tree / proof** — BLAKE3 binary Merkle tree (moved here from chain's `catchup/merkle.rs`); reusable for any cold-tier commitment, not just chain blocks
- **Spectral lottery-ticket init** — `new_spectral` constructor produces spectrally-flat initialization (gated `spectral_shard`)

### Freeze/Thaw Integrity Envelope

- **`MerkleFrozenEnvelope`** — self-play freeze/thaw with Merkle-checked integrity on every thaw (gated `merkle_freeze`)
- **Atomic snapshot reload** — readers never see a torn / half-thawed shard
- **Dendritic LoRA branch view** — read-only branch view over `style_weights[64]` for dendritic-style adapter composition (gated `dendritic_lora`)

### Consolidation + Retrieval

- **MAPE-K self-healing loop** — shard integrity monitor + auto-repair loop
- **Raven / δ-Mem consolidation pipeline** — emergent memory compaction across shards (the "sleep cycle" that consolidates fresh shards into cold tier)
- **AnyRAG escalation gateway** — the boundary where local latent retrieval escalates to external LLM/RAG; *when to escalate* is the IP
- **Vibe KG triple templates + arch agent** — turn latent proximity into knowledge-graph triples (semantic encounters → KG triple from similarity)
- **ShardCompactor** — cold-tier compaction (gated `shard_compactor`)

---

## Trained Weight Assets (in riir-train)

- Adapters trained across our game portfolio
- Cross-game universal concept neurons
- Per-zone weight snapshots
- Episode DB (game strategy history, edge cases)

**These live in `riir-train` (internal). `riir-ai`, `riir-chain`, and `riir-neuron-db` consume them via the runtime freeze/thaw path — none of the three ships raw training data; they ship the snapshotted, version-checked adapter the runtime hot-swaps.**

---

## Why riir-ai Is Hard to Replicate

| Pillar | Capability | Why Hard to Copy |
|--------|-----------|-----------------|
| **Freeze/thaw runtime** | Lock-free adapter snapshots, integrity-verified hot-swap, per-NPC personality versioning, fused adapter inference | The runtime is small but every concurrency detail is tuned: memory ordering, zero-copy reads, atomic snapshot swap. A re-implementation will race, stall, or see torn updates. Months of profiling to get right. |
| **Latent-to-latent operations** | Decision-level cognition in embedding space, sigmoid-gated projections, raw↔latent bridge with deterministic sync boundary | The bridge between raw (synced, replay-safe) and latent (efficient, composable) is the hard part. Getting the boundary right — what crosses as raw scalars vs what stays latent — requires domain tuning per game. |
| **Self-learn / adaptive NPCs** | All-goals learning, open-ended policy gradient, collapse-aware recovery, self-play, curiosity-driven exploration — all runtime, no offline training round-trip | Turning scripted NPCs into living ones at runtime is the product. The collapse-detection and exploration signals are tuned against real game sessions. |

## Why riir-chain Is Hard to Replicate

| Pillar | Capability | Why Hard to Copy |
|--------|-----------|-----------------|
| **Chain design** | Co-located AI+wallet, latent-encoded balances, split-key, self-healing, five-tier memory | Novel neuro-symbolic economic design. No incumbent co-locates AI weights with wallet state in zero-copy fixed-size structures. |
| **LatCal commitment bridge** | Deterministic fixed-point arithmetic that turns latent decisions into raw committed scalars | The arithmetic obfuscation + batch determinant validation took extensive validation. Round-trip fidelity and tamper rejection are tuned against real attack patterns. |
| **Network effects** | Live chain with real economic activity | A forked chain has no players, validators, or economy. Can't be copied. |

## Why riir-neuron-db Is Hard to Replicate

| Pillar | Capability | Why Hard to Copy |
|--------|-----------|-----------------|
| **NeuronShard layout + integrity** | Fixed-layout Pod, BLAKE3 commitment, zero-copy mmap, Merkle proof | Every field offset is pinned for cross-platform determinism. A re-implementation with different padding/alignment breaks cold-tier round-trip and anti-tamper checks. Months of layout tuning + a `merkle_root`-forgetting bug class that only shows up under `--all-features` CI caught this. |
| **Freeze/thaw envelope** | `MerkleFrozenEnvelope`, atomic reload, dendritic branch view | The freeze/thaw integrity envelope is the bridge between runtime adapter swap (riir-ai) and cold-tier commitment (riir-chain). Concurrency details + Merkle proof mechanics are tuned; a re-implementation will serve torn shards under load. |
| **Raven / δ-Mem consolidation** | Sleep-cycle compaction across shards, MAPE-K self-healing | The consolidation algorithm is emergent — tuned over many sessions to decide which shards to merge, which to evict, which to repair. Not derivable from first principles. |
| **AnyRAG escalation policy** | Gateway that decides when local latent retrieval escalates to external LLM/RAG | Knowing *when to escalate* (vs. answer from local shards) is the IP. Escalate too often → cost + latency; too rarely → quality loss. Tuned against real query distributions. |
| **Vibe KG triple templates** | Arch agent that turns latent proximity into knowledge-graph triples | The templates that map (entity, latent-similarity-threshold, entity) → KG triple are domain-tuned. Generic triples don't capture the semantic encounter structure games need. |

## Why riir-train Is Hard to Replicate

| Pillar | Capability | Why Hard to Copy |
|--------|-----------|-----------------|
| **Training know-how** | 90+ adapter-training methods, consumer-GPU training, trained weight assets | Algorithms are published (arXiv). The implementations + configs took years of validation. Honest benchmarking showed most methods deliver stability, not quality gains — but the few that work + the trained weight assets are still GPU-hours of data. Secondary moat. |

---

## Decision Rules for AI (When Creating Research / Plans / Docs)

Use these rules to decide what is safe for public `katgpt-rs/.research/` vs what must stay internal.

### Ask: Is this the WHAT or the HOW? And which repo?

| If it's about... | Goes in | Because |
|------------------|---------|---------|
| Inference engine mechanics (DDTree, ConstraintPruner trait, bandit theory, speculative decode) | `katgpt-rs/.research/` (public) | Generic framework — adoption value, no moat risk |
| An arXiv paper survey (what algorithm exists, why it's interesting) | `katgpt-rs/.research/` (public) | Literature review — tells WHAT exists, not HOW we use it |
| A capability description ("riir-ai hot-swaps NPC personalities at runtime", "riir-chain commits latent decisions as raw scalars") | `katgpt-rs/.research/` (public, if needed for context) | Outcome — doesn't reveal the method |
| **Training-method research, plans, benchmarks** | `riir-train` internal | Training know-how vault — separate repo so riir-ai/riir-chain ship clean |
| **Trained weights, training data, training artifacts** | `riir-train` internal (never shipped) | Data assets — GPU-hours to produce |
| Which specific training method produced a given adapter | `riir-train` internal | Naming the technique hands competitors the implementation direction |
| Exact hyperparameters, configs, or fusion recipes | `riir-train` internal | That's the fuel — the HOW that achieves the result |
| GPU kernel source for a specific training method | `riir-train` internal | Kernel implementations are the implementation detail |
| **Freeze/thaw runtime internals** (concurrency protocol, hot-swap watcher, merge kernel) | `riir-ai` internal | Runtime IP — this is the ship-focus game product |
| **Latent-operation internals** (projection directions, bridge function code, sigmoid gate tuning) | `riir-ai` internal | The efficiency multiplier — keep private |
| **Self-learn / adaptive internals** (mixer parameters, collapse detector, exploration signal tuning) | `riir-ai` internal | The selling point — keep private |
| **Chain internals** (LatCal encoding projections, key derivation, healing loop, chain economics parameters, `catchup/` persistence) | `riir-chain` internal | The implementation IS the IP — chain IP lives in `riir-chain`, not `riir-ai` |
| **Chain / LatCal / commitment / sync-bridge / quorum / catchup research and plans** | `riir-chain/.research/` + `riir-chain/.plans/` internal | New chain-flavored notes land in `riir-chain`; historical chain notes (pre-spin-off) remain in `riir-ai/.research/` as-is |
| **Neuron-shard internals** (NeuronShard Pod layout, BLAKE3 commitment, MerkleFrozenEnvelope, MAPE-K loop, Raven/δ-Mem consolidation, AnyRAG gateway, vibe KG triple templates, spectral init, ShardCompactor, dendritic branch) | `riir-neuron-db` internal | Shard IP — chain re-exports via `neuron_db` feature but the canonical source is `riir-neuron-db/`, not `riir-chain/src/neuron_db/` |
| **Neuron-shard / freeze-envelope / consolidation / AnyRAG / vibe-KG / Merkle-tree research and plans** | `riir-neuron-db/.research/` + `riir-neuron-db/.plans/` internal | New shard-flavored notes land in `riir-neuron-db`; historical shard notes (pre-spin-off) remain in `riir-chain/.research/` or `riir-ai/.research/` as-is |
| Game domain configs (character classes, zone behavior, economy rules, quest grammar) | `riir-ai` internal | Game design IP |
| Our benchmark numbers beyond what's already public | `riir-ai` (runtime/game) / `riir-chain` (chain) / `riir-neuron-db` (shards) / `riir-train` (training) | Match the repo to the proof's subject |

### Rule of Thumb

**What = public. How = private. Training how = riir-train. Runtime how = riir-ai. Chain how = riir-chain. Shard how = riir-neuron-db.**

- "NPCs hot-swap personalities at runtime via versioned snapshots" → public (capability)
- "The snapshot uses [specific concurrency primitive] with [specific memory ordering]" → `riir-ai` private (runtime how)
- "We train adapters with [specific method] at [specific config]" → `riir-train` private (training how)
- "Balances are encoded as latent vectors and committed via LatCal fixed-point bridges" → public (concept)
- "The LatCal projection uses [specific learned values] / [specific matrix decomposition]" → `riir-chain` private (implementation)
- "Shards are fixed-layout Pods with BLAKE3 commitment, reloadable atomically" → public (concept)
- "The NeuronShard field layout uses [specific offsets] / [specific Pod alignment]" → `riir-neuron-db` private (implementation)
- "The AnyRAG escalation triggers at [specific score threshold] / [specific cost-latency tradeoff]" → `riir-neuron-db` private (implementation)
- "The projection uses [specific learned values]" (training side) → `riir-train` private (implementation)

### When Unsure

Default to the relevant private repo (`riir-ai` for gameplay, `riir-chain` for chain transport, `riir-neuron-db` for shards / freeze / consolidation / AnyRAG / vibe-KG, `riir-train` for training). It is always safe to keep something private. It is never safe to un-leak something public.

---

## Super-GOAT Capture Protocol

A **Super-GOAT** is a novel mechanism that creates a capability competitors don't have — a private IP moat, not just a benchmark win. Most papers are GOAT/Gain (incremental). Detecting Super-GOAT early and routing it correctly is the difference between building a moat and giving away the store.

### Detection — 4 gates, ALL must pass

| Gate | Question | Fail → |
|------|----------|--------|
| **Novelty** | Grep `.research/` + `.plans/` across all 5 repos AND shipped code in `katgpt-rs/`, `riir-ai/crates/`, `riir-chain/src/`, `riir-chain/crates/`, `riir-neuron-db/src/`, `riir-armageddon/crates/`. Does any existing note or shipped module cover this mechanism? | → Gain |
| **Capability class** | Is this a new *class* of behavior (not just better numbers on an existing capability)? | → GOAT |
| **Selling point** | Can you finish: "Our NPCs/systems/chain do ___ that no competitor can"? | → GOAT |
| **Force multiplier** | Connects to ≥2 existing pillars/systems (freeze/thaw, latent ops, self-learn, chain, neuron-shard/freeze, KG/HLA, AnyRAG/vibe, LatCal)? | → GOAT |

4/4 → Super-GOAT. Any miss → highest matching tier.

### Routing — the two-output rule

Super-GOAT MUST produce **both** outputs. Skipping either is a process failure.

| Output | Location | Purpose |
|--------|----------|--------|
| **Open primitive** | `katgpt-rs/.research/` + `crates/katgpt-core/src/` | Adoption hook — generic math, no game semantics, no chain semantics. The Ferrari part. |
| **Private guide** | `riir-ai/.research/NNN_*.md` (gameplay selling points) OR `riir-chain/.research/NNN_*.md` (chain / LatCal / sync-bridge selling points — create folder on first use) OR `riir-neuron-db/.research/NNN_*.md` (shard / freeze / consolidation / AnyRAG / vibe-KG / Merkle selling points — create folder on first use) | The selling-point doc — commercial value, connection map, validation protocol. The gas. |

**How to pick the guide repo:**
- Game-runtime / HLA / functor / self-learn / NPC behavior selling point → `riir-ai/.research/`
- Chain / LatCal / commitment / quorum / catchup / sync-bridge / DeFi economics selling point → `riir-chain/.research/`
- Neuron-shard / freeze envelope / consolidation / AnyRAG / vibe-KG / Merkle integrity selling point → `riir-neuron-db/.research/`
- Crosses the chain sync boundary (latent → raw commitment) → primary guide in `riir-chain/` (owns the boundary), cross-reference from `riir-ai/` and/or `riir-neuron-db/`
- Crosses the shard→runtime boundary (frozen shard → runtime adapter hot-swap) → primary guide in `riir-neuron-db/` (owns the shard structure), cross-reference from `riir-ai/`

The private guide MUST answer: what's the selling point, what connects to existing systems, what crosses the sync boundary (raw scalars only), what stays private, and how to validate the Super-GOAT claim.

### Why the guide matters

The open primitive is reproducible from the paper — anyone reading arXiv can build it. The **guide** is where the private IP lives: the game-specific, chain-specific, or shard-specific semantics, the integration with HLA/emotions/KG/freeze-thaw/AnyRAG/vibe/LatCal, the cross-system transfer protocol. Without the guide, the knowledge is lost — a future agent won't know why the open primitive matters or how to use it commercially.

**Common failure:** agent reads paper → writes open primitive + plan → marks done → moves on. The private selling-point knowledge was never captured. Next agent sees generic math with no context. Fix: the novelty gate (SKILL.md §1.5) blocks this — Super-GOAT requires the riir-ai, riir-chain, or riir-neuron-db guide before the task is complete.

---

## Related

| Doc | Connection |
|-----|-----------|
| 119 — Worms Armageddon Latent Space Game | Game product concept. Internal (`riir-ai/.research/119`). Moved out of public repo. |
| `riir-chain/AGENTS.md` | Repo-local context for the chain spin-off (workspace layout, `merkle_root` lesson, drift resolution, `develop` branch policy). |
| `riir-chain/.plans/001_chain_spinoff.md` | Chain crate migration record (riir-ai → riir-chain). |
| `riir-chain/.plans/002_chaind_spinoff.md` | Chaind daemon migration record. |
| `riir-neuron-db/AGENTS.md` | Repo-local context for the neuron-db spin-off (what this crate owns vs what stays in `riir-chain`, feature → chain feature mapping, `merkle_root` lesson still applies, `develop` branch policy). |
| `riir-neuron-db/.plans/001_extraction_from_riir_chain.md` | Shard crate migration record (riir-chain → riir-neuron-db). |
