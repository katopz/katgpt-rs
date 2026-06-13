# Commercial Strategy — Engine & Platform Split

**Date:** 2026-06
**Status:** Active
**Purpose:** Guide for AI agents to reason about what is public vs internal when creating research, plans, and docs.

---

## The Boundary

Two repos. The split is absolute.

| Repo | License | Role |
|------|---------|------|
| `katgpt-rs` | MIT (public) | Engine — generic inference framework. Adoption funnel. |
| `riir-ai` | Private (internal) | Platform — all accumulated know-how. The moat. |

**Rule: anything `riir-*` is internal. No exceptions. No per-crate deliberation.**

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

### Game AI Training

- Train game AI on **consumer GPUs** (single Apple M-series or single NVIDIA) — no cluster required
- **30+ training methods**, all GOAT-gated and arena-proven
- **Zero catastrophic forgetting** when training across multiple games sequentially (96.5% cross-game retention)
- **Compose trained adapters** without quality loss (associative, zero inference cost)
- **Dynamic adapter routing** based on game state and objective (100% win rate vs static routing)
- **Sparse attention** reducing KV cache 6× without quality loss
- **Extract human-readable rules** from trained weights at zero inference cost
- **Self-play**: learns game strategy without external oracle (2.4T ops/sec)
- **Trajectory folding**: 78% reduction in redundant training moves

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

- Bomber: adaptive AI +475 vs baseline
- Go: 100% win vs Random 35%
- FFT Tactics: 99% win rate (game theory optimal)
- Frame-sampling: 939K decisions/sec
- Go LoRA: trained adapter (979 ELO) matches MCTS-200 (961 ELO)
- Adaptive reasoning: +177% quality on hard queries at ≤50% cost

### Trained Weight Assets

- LoRA adapters trained across Bomber, Go, FFT, Civ
- Cross-game universal concept neurons
- Per-zone weight snapshots
- Episode DB (game strategy history, edge cases)

---

## Why riir-ai Is Hard to Replicate

| Pillar | Capability | Why Hard to Copy |
|--------|-----------|-----------------|
| **Training know-how** | 30+ GOAT-proven methods, consumer-GPU training, zero forgetting, adapter composition | Algorithms are published (arXiv). But the implementations + configs + fusion strategies + GPU kernels took years of GOAT-gated validation. Re-deriving per-method = months. 30+ = years. |
| **Chain design** | Co-located AI+wallet, latent-encoded balances, split-key, self-healing, five-tier memory | Novel neuro-symbolic economic design. No incumbent co-locates AI weights with wallet state in zero-copy fixed-size structures. |
| **Trained weights + data** | Adapters, concept neurons, episode DB, weight snapshots | Pure data assets — GPU-hours to produce. Flywheel: more games → more cross-game knowledge → better warm-start. |
| **Network effects** | Live chain with real economic activity | A forked chain has no players, validators, or economy. Can't be copied. |

---

## Decision Rules for AI (When Creating Research / Plans / Docs)

Use these rules to decide what is safe for public `katgpt-rs/.research/` vs what must stay in `riir-ai` internal docs.

### Ask: Is this the WHAT or the HOW?

| If it's about... | Goes in | Because |
|------------------|---------|---------|
| Inference engine mechanics (DDTree, ConstraintPruner trait, bandit theory, speculative decode) | `katgpt-rs/.research/` (public) | Generic framework — adoption value, no moat risk |
| An arXiv paper survey (what algorithm exists, why it's interesting) | `katgpt-rs/.research/` (public) | Literature review — tells WHAT exists, not HOW we use it |
| A capability description ("riir-ai can train with zero forgetting") | `katgpt-rs/.research/` (public, if needed for context) | Outcome — doesn't reveal the method |
| Which specific training method we use for a capability | `riir-ai` internal | Naming the technique hands competitors the implementation direction |
| Exact hyperparameters, configs, or fusion recipes | `riir-ai` internal | That's the fuel — the HOW that achieves the result |
| GPU kernel source for a specific training method | `riir-ai` internal | Kernel implementations are the implementation detail that took years |
| Chain internals (encoding projections, key derivation, data layout, healing loop) | `riir-ai` internal | The implementation IS the IP |
| Game domain configs (character classes, zone behavior, economy rules, quest grammar) | `riir-ai` internal | Game design IP |
| Trained weights (`.bin`, concept neurons, snapshots) | `riir-ai` internal (never shipped) | Data assets — GPU-hours to produce |
| Our GOAT proof configs, benchmark numbers beyond what's already public | `riir-ai` internal | Implementation-level detail — the HOW |

### Rule of Thumb

**What = public. How = private.**

- "This system can train across games without forgetting" → public (capability)
- "We achieve this using [specific method] with [specific config]" → private (technique + config)
- "Balances are encoded as latent vectors" → public (concept)
- "The projection uses [specific learned values]" → private (implementation)

### When Unsure

Default to `riir-ai` internal. It is always safe to keep something private. It is never safe to un-leak something public.

---

## Related

| Doc | Connection |
|-----|-----------|
| [230 — Worms Armageddon Latent Space Game](./230_Worms_Armageddon_Latent_Space_Artillery_Game.md) | Game product concept. Internal (`riir-ai`). |
