# Commercial Strategy — Public Routing Rules (trimmed)

**Date:** 2026-06 (revised 2026-06-29 — added Benchmark Domain Exception; revised 2026-06-27 — sensitive content moved to private)
**Status:** Active (public subset)
**Purpose:** Let public-research agents self-govern the public/private boundary without needing the sensitive moat doc.

> ⚠️ **This is the PUBLIC routing-rules subset.** The full strategy doc — moat analysis, "why hard to replicate" detail, capability specifics — moved to **`riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md`** (internal) on 2026-06-27 because it exposes too much commercial detail for a public MIT repo. This trimmed version keeps only what public research needs to route correctly.

---

## The Boundary

Five repos. The split is absolute.

| Repo | License | Role |
|------|---------|------|
| `katgpt-rs` | MIT (public) | **Engine** — generic inference framework. Adoption funnel. No *product* game IP, no chain IP, no neuron-shard IP. Toy benchmark domains (Bomber/Go/Monopoly/FFT-arena) are explicitly fine — see §"Benchmark Domain Exception" below. |
| `riir-ai` | Private (internal) | **Game product** — freeze/thaw runtime, self-learn/adaptive NPCs, latent-space operations, game systems. |
| `riir-chain` | Private (internal) | **Neuro-symbolic chain transport** — co-located AI+wallet state, LatCal encoding, chain economics, `riir-chaind` daemon, `catchup/` persistence. Re-exports `riir-neuron-db` under its `neuron_db` feature. |
| `riir-neuron-db` | Private (internal) | **Neuron-shard leaf crate** — `NeuronShard` weight blob, `ShardIndex`, generic `MerkleTree`/`MerkleProof`, `MerkleFrozenEnvelope`, MAPE-K self-healing, Raven/δ-Mem consolidation, AnyRAG gateway, vibe KG triples. No chain dependency. |
| `riir-train` | Private (internal) | **Training research** — adapter training methods, training data, trained weights. Know-how vault. |

**Rule: anything `riir-*` is internal. No exceptions.**

### Benchmark Domain Exception (toy games ≠ product IP)

**Toy 2D rule-system games used as benchmark domains are NOT game IP.** Their implementations live in `katgpt-rs` (public) and that is correct, not a leak.

The actual game-product moat is the **runtime that runs on top of a game** — freeze/thaw composition, NPC archetype wiring, HLA affect projection, trained LoRA weights, level/economy/quest design. The toy game itself is just a benchmark harness; Bomberman, Go, Monopoly, and a generic ATB battle engine are public-domain rule systems anyone can re-implement in a weekend.

| Category | Example | Lives in |
|---|---|---|
| ✅ Public benchmark domain | `bomber`, `monopoly`, `go`, `fft` (ATB arena) — generic rule systems on `bevy_ecs` + generic MCTS/bandit | `katgpt-rs` (public) |
| ✅ Public benchmark wiring | MCTS over Bomber, CCE over heterogeneous cost tables, generic `GameState` / `game_state` forward-model trait | `katgpt-rs` (public) |
| ❌ Private product runtime | NPC brain wiring for a real product game, archetype blends wired to specific characters, HLA projection tuned for specific NPCs, freeze/thaw composition for a commercial title | `riir-ai` (private) |
| ❌ Private design data | Level/quest/economy tuning, character class balance, zone behavior configs for a commercial product | `riir-ai` (private) |
| ❌ Private trained weights | LoRA adapters trained for a specific product game | `riir-train` (private) |

**The distinguishing test:** *"Could a competitor re-implement this from public rules + generic primitives in a weekend?"* → public. *"Does this encode product-specific tuning, weights, wiring, or design?"* → private.

**Anti-pattern — cross-boundary coupling constants:** A public benchmark domain must not hardcode a constant whose comment says `must match riir_gpu::game::fft_replay::FFT_STATE_VOCAB`. The benchmark's constants are self-contained in the public repo. If a private consumer needs the same value, the private side imports from public (one-way), never the reverse. A cross-reference comment that names a private module path IS a leak, even if the constant value itself is benign.

---

## Repo Structure & Tier Model (public engine only)

The public engine splits across TWO crates:

| Tier | Crate | Role | What lives here |
|------|-------|------|-----------------|
| **0 — Substrate** | `katgpt-core` (leaf, on crates.io) | Pure inference mechanics — the engine block | SIMD, `types`, `transformer`/`weights`, `hla`, `dd_tree`/`spec_types`, `mcts`, `sampling`, `tokenizer`, `delta_mem`. Minimal deps. |
| **1 — Engine + cognitive basics** | `katgpt-rs` (root, public) | The adoption funnel — re-exports substrate + ships the BASIC cognitive/reasoning layer + engine primitives + games/examples | `cce`, `cgsp`, `clr`, `compaction`, `attn_match`, speculative, game engines, examples, benches. |
| **2 — GOAT versions + composition + IP** | `riir-*` (private) | The gas — GOAT/Super-GOAT tuned versions, `*_runtime` composition layers, game/chain/shard IP | See private doc. |

**Two rules:** (1) a module moves DOWN to core only if it's pure inference substrate; (2) cognitive/reasoning primitives stay in root as the BASIC public version, with their GOAT-tuned `*_runtime` siblings in `riir-*`.

---

## Decision Rules for AI (When Creating Research / Plans / Docs)

**Rule of Thumb: What = public. How = private. Training how = riir-train. Runtime how = riir-ai. Chain how = riir-chain. Shard how = riir-neuron-db.**

| If it's about... | Goes in | Because |
|------------------|---------|---------|
| Inference engine mechanics (DDTree, ConstraintPruner trait, bandit, speculative decode) | `katgpt-rs` (public) | Generic framework — adoption value, no moat risk |
| An arXiv paper survey (what algorithm exists) | `katgpt-rs` (public) | Literature review — tells WHAT exists, not HOW we use it |
| A capability description ("NPCs hot-swap personalities at runtime") | `katgpt-rs` (public, for context) | Outcome — doesn't reveal the method |
| **Toy benchmark game engines** (`bomber`, `monopoly`, `go`, `fft` ATB arena) and their generic MCTS/bandit/CCE wiring | `katgpt-rs` (public) | Public-domain rule systems + generic primitives. NOT product IP — see §"Benchmark Domain Exception". The runtime that runs *on top* (freeze/thaw, archetype wiring, trained weights) is what's private. |
| **Training-method research, plans, benchmarks, weights, configs** | `riir-train` internal | Training know-how vault |
| **Freeze/thaw / latent-op / self-learn internals** (the HOW) | `riir-ai` internal | Runtime IP — the ship-focus game product |
| **Chain internals** (LatCal, key derivation, healing loop, economics, `catchup/`) | `riir-chain` internal | The implementation IS the IP |
| **Neuron-shard internals** (Pod layout, BLAKE3, MerkleFrozenEnvelope, consolidation, AnyRAG, vibe KG) | `riir-neuron-db` internal | Shard IP |
| **Product game design configs** (commercial title's character classes, zone behavior, economy rules, quest grammar) | `riir-ai` internal | Product game design IP — NOT the same as toy benchmark domains, which stay public |
| Our benchmark numbers beyond what's already public | match the repo to the proof's subject | |

### When Unsure

Default to the relevant private repo. It is always safe to keep something private. It is never safe to un-leak something public. **For the full moat analysis and "why hard to replicate" detail, see `riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md` (internal).**

---

## Super-GOAT Capture Protocol (routing summary)

A **Super-GOAT** is a novel mechanism that creates a capability competitors don't have. Super-GOAT MUST produce **both** outputs:

| Output | Location | Purpose |
|--------|----------|--------|
| **Open primitive** | `katgpt-rs` + `crates/katgpt-core/src/` | Adoption hook — generic math, no game/chain semantics. The Ferrari part. |
| **Private guide** | `riir-ai/.research/` (gameplay) OR `riir-chain/.research/` (chain/LatCal) OR `riir-neuron-db/.research/` (shard/freeze/AnyRAG/vibe) | The selling-point doc — commercial value, connection map, validation. The gas. |

**How to pick the guide repo:** gameplay/HLA/functor/self-learn → `riir-ai`; chain/LatCal/commitment/quorum → `riir-chain`; shard/freeze/consolidation/AnyRAG/vibe/Merkle → `riir-neuron-db`. Full detection gates + routing detail in the private doc.

---

## Related

| Doc | Connection |
|-----|-----------|
| `riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md` | **Full strategy doc (internal)** — moat analysis, capability details, "why hard to replicate" tables, full Super-GOAT detection gates. |
| `riir-chain/AGENTS.md` | Repo-local context for the chain spin-off. |
| `riir-neuron-db/AGENTS.md` | Repo-local context for the neuron-db spin-off. |
