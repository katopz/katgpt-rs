# Commercial Strategy — Engine & Platform Split

**Date:** 2026-06
**Status:** Active
**Purpose:** Guide for AI agents to reason about what is public vs internal when creating research, plans, and docs.

> ⚠️ **This doc is PUBLIC** (lives in `katgpt-rs`, MIT licensed). Every line is visible to the world. Describe **capabilities**, never implementation details. When in doubt, cut it.

---

## The Boundary

Three repos. The split is absolute.

| Repo | License | Role |
|------|---------|------|
| `katgpt-rs` | MIT (public) | Engine — generic inference framework. Adoption funnel. |
| `riir-ai` | Private (internal) | **Game product** — freeze/thaw runtime, self-learn/adaptive NPCs, latent-space operations, neuro-symbolic chain, game systems. The ship-focus repo. |
| `riir-train` | Private (internal) | **Training research** — adapter training methods, training data, trained weights. Know-how vault. |

**Rule: anything `riir-*` is internal. No exceptions. No per-crate deliberation.**

### Why three repos (not two) — and why LoRA training is NOT the focus

We benchmarked 90+ adapter-training methods. **Most produced stability proofs (training converges, no NaN) rather than quality proofs (the adapter actually plays measurably better).** The few real, reproducible gains came from **runtime adapter selection (routing)** — not from the training method that produced the adapter.

**The lesson:** the value of an adapted NPC brain is in *which frozen adapter you swap in at runtime*, not in *how that adapter was trained*. Training method choice is a second-order effect; routing and runtime composition are first-order.

So we pivoted:
- **Training research** moved to `riir-train` (still a moat — implementations + configs took years). But it's no longer the headline.
- **`riir-ai` ships**: freeze/thaw runtime (swap frozen adapters at runtime), latent-to-latent operations (compute NPC decisions in embedding space, never decode-then-re-encode), and self-learn/adaptive NPCs (runtime improvement, no offline training round-trip).
- **`riir-ai` never names a training method.** It exposes a generic training interface (trait); `riir-train` implements it. Zero-cost abstraction.

---

## Why katgpt-rs Is Public (MIT)

`katgpt-rs` is the **generic inference framework** — DDTree, ConstraintPruner trait, bandit, pruners infra, speculative decode. These are the hooks that make game devs depend on the engine.

**Why public:**
- It's the adoption funnel. Developers build on the engine, then need the platform.
- The engine alone produces no useful game AI output — it's a runtime, not the intelligence.
- MIT attracts contributors and creates dependency without exposing know-how.
- No legal friction for enterprise adoption.

**"Ferrari, no gas":** `katgpt-rs` is the open Ferrari. Without the private platform, it runs but produces nothing competitive. The gas is inside `riir-ai`.

---

## What riir-ai Can Do (Capabilities — Not How)

`riir-ai` is the platform. Below is **what it can do**, not how it's built. The how stays internal.

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

### Neuro-Symbolic Chain

- **AI state and wallet state co-located** in the same zero-copy data structure
- **Latent-encoded balances** — not plaintext integers, tamper-resistant by inspection
- **Split-key transactions** — neither party holds the combined key in transit
- **Self-healing chain** — detects anomalies and repairs automatically
- **Five-tier memory** with graceful degradation — engine never fails to boot
- **Cross-chain bridge** to Solana
- **Full DeFi economy**: gas, rent, slashing
- **9 GOAT proofs**: roundtrip fidelity, key security, pipeline throughput, tamper rejection

### Arena Proofs (Outcomes, Not Methods)

- Adaptive AI: large win-rate improvement vs baseline across multiple game types
- Game-theory-optimal play achieved in turn-based tactics (99% win rate)
- Frame-sampling: 939K decisions/sec
- Dynamic routing: 100% win rate vs 0% static routing (runtime adapter selection)
- Adaptive reasoning: +177% quality on hard queries at ≤50% cost
- Browser NPC inference: sub-µs-per-call brain forward pass (WASM SIMD 5.47-7.27× over scalar)

### Trained Weight Assets (in riir-train)

- Adapters trained across our game portfolio
- Cross-game universal concept neurons
- Per-zone weight snapshots
- Episode DB (game strategy history, edge cases)

**These live in `riir-train` (internal). riir-ai consumes them via the runtime freeze/thaw path — it never ships raw training data in-game; it ships the snapshotted, version-checked adapter the runtime hot-swaps.**

---

## Why riir-ai Is Hard to Replicate

| Pillar | Capability | Why Hard to Copy |
|--------|-----------|-----------------|
| **Freeze/thaw runtime** | Lock-free adapter snapshots, integrity-verified hot-swap, per-NPC personality versioning, fused adapter inference | The runtime is small but every concurrency detail is tuned: memory ordering, zero-copy reads, atomic snapshot swap. A re-implementation will race, stall, or see torn updates. Months of profiling to get right. |
| **Latent-to-latent operations** | Decision-level cognition in embedding space, sigmoid-gated projections, raw↔latent bridge with deterministic sync boundary | The bridge between raw (synced, replay-safe) and latent (efficient, composable) is the hard part. Getting the boundary right — what crosses as raw scalars vs what stays latent — requires domain tuning per game. |
| **Self-learn / adaptive NPCs** | All-goals learning, open-ended policy gradient, collapse-aware recovery, self-play, curiosity-driven exploration — all runtime, no offline training round-trip | Turning scripted NPCs into living ones at runtime is the product. The collapse-detection and exploration signals are tuned against real game sessions. |
| **Chain design** | Co-located AI+wallet, latent-encoded balances, split-key, self-healing, five-tier memory | Novel neuro-symbolic economic design. No incumbent co-locates AI weights with wallet state in zero-copy fixed-size structures. |
| **Training know-how (riir-train)** | 90+ adapter-training methods, consumer-GPU training, trained weight assets | Algorithms are published (arXiv). The implementations + configs took years of validation. Honest benchmarking showed most methods deliver stability, not quality gains — but the few that work + the trained weight assets are still GPU-hours of data. Secondary moat. |
| **Network effects** | Live chain with real economic activity | A forked chain has no players, validators, or economy. Can't be copied. |

---

## Decision Rules for AI (When Creating Research / Plans / Docs)

Use these rules to decide what is safe for public `katgpt-rs/.research/` vs what must stay internal.

### Ask: Is this the WHAT or the HOW? And which repo?

| If it's about... | Goes in | Because |
|------------------|---------|---------|
| Inference engine mechanics (DDTree, ConstraintPruner trait, bandit theory, speculative decode) | `katgpt-rs/.research/` (public) | Generic framework — adoption value, no moat risk |
| An arXiv paper survey (what algorithm exists, why it's interesting) | `katgpt-rs/.research/` (public) | Literature review — tells WHAT exists, not HOW we use it |
| A capability description ("riir-ai hot-swaps NPC personalities at runtime") | `katgpt-rs/.research/` (public, if needed for context) | Outcome — doesn't reveal the method |
| **Training-method research, plans, benchmarks** | `riir-train` internal | Training know-how vault — separate repo so riir-ai ships clean |
| **Trained weights, training data, training artifacts** | `riir-train` internal (never shipped) | Data assets — GPU-hours to produce |
| Which specific training method produced a given adapter | `riir-train` internal | Naming the technique hands competitors the implementation direction |
| Exact hyperparameters, configs, or fusion recipes | `riir-train` internal | That's the fuel — the HOW that achieves the result |
| GPU kernel source for a specific training method | `riir-train` internal | Kernel implementations are the implementation detail |
| **Freeze/thaw runtime internals** (concurrency protocol, hot-swap watcher, merge kernel) | `riir-ai` internal | Runtime IP — this is the ship-focus product |
| **Latent-operation internals** (projection directions, bridge function code, sigmoid gate tuning) | `riir-ai` internal | The efficiency multiplier — keep private |
| **Self-learn / adaptive internals** (mixer parameters, collapse detector, exploration signal tuning) | `riir-ai` internal | The selling point — keep private |
| Chain internals (encoding projections, key derivation, data layout, healing loop) | `riir-ai` internal | The implementation IS the IP |
| Game domain configs (character classes, zone behavior, economy rules, quest grammar) | `riir-ai` internal | Game design IP |
| Our benchmark numbers beyond what's already public | `riir-ai` (runtime/game/chain) or `riir-train` (training) | Match the repo to the proof's subject |

### Rule of Thumb

**What = public. How = private. Training how = riir-train. Runtime how = riir-ai.**

- "NPCs hot-swap personalities at runtime via versioned snapshots" → public (capability)
- "The snapshot uses [specific concurrency primitive] with [specific memory ordering]" → `riir-ai` private (runtime how)
- "We train adapters with [specific method] at [specific config]" → `riir-train` private (training how)
- "Balances are encoded as latent vectors" → public (concept)
- "The projection uses [specific learned values]" → `riir-train` private (implementation)

### When Unsure

Default to `riir-ai` internal. It is always safe to keep something private. It is never safe to un-leak something public.

---

## Related

| Doc | Connection |
|-----|-----------|
| 119 — Worms Armageddon Latent Space Game | Game product concept. Internal (`riir-ai/.research/119`). Moved out of public repo. |
