# Commercial Strategy — Public Routing Rules (trimmed)

**Date:** 2026-06 (revised 2026-08-20 — added `riir-dapps` (the dApp layer) + §"The Second Axis: Layering"; 8-repo count; revised 2026-07-17 — added `riir-game-sdk` + `riir-armageddon` to Boundary table; revised 2026-06-29 — added Benchmark Domain Exception; revised 2026-06-27 — sensitive content moved to private)
**Status:** Active (public subset)
**Purpose:** Let public-research agents self-govern the public/private boundary without needing the sensitive moat doc.

> **Amendment 2026-11-11 — riir-refine is a public crate.** `riir-ai/riir-refine` is PUBLIC (MIT) and publishes **`cargo-refine` to crates.io** (0.5.0 first publish 2026-11-11 — riir-refine issue 221 P3, record in its HISTORY.md `6cdaffed`; published from a generated manifest carrying ZERO private-sibling deps). The "anything `riir-*` is internal" rule below is superseded FOR THAT REPO ONLY (riir-refine issue 192 decision 1 + GAP 11); every other private sibling stays internal and `publish = false`. `katgpt-core` 0.5.0 published the same day (Plan 635: 414 → 278 features under the crates.io cap).

> ⚠️ **This is the PUBLIC routing-rules subset.** The full strategy doc — moat analysis, "why hard to replicate" detail, capability specifics — moved to **`riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md`** (internal) on 2026-06-27 because it exposes too much commercial detail for a public MIT repo. This trimmed version keeps only what public research needs to route correctly.

---

## The Boundary

Eight repos. The split is absolute.

| Repo | License | Role |
|------|---------|------|
| `katgpt-rs` | MIT (public) | **Engine** — generic inference framework. Adoption funnel. No *product* game IP, no chain IP, no neuron-shard IP. Toy benchmark domains (Bomber/Go/Monopoly/FFT-arena) are explicitly fine — see §"Benchmark Domain Exception" below. |
| `riir-ai` | Private (internal) | **Game product** — freeze/thaw runtime, self-learn/adaptive NPCs, latent-space operations, game systems. |
| `riir-chain` | Private (internal) | **Neuro-symbolic chain transport** — co-located AI+wallet state, LatCal encoding, chain economics, `riir-chaind` daemon, `catchup/` persistence. Re-exports `riir-neuron-db` under its `neuron_db` feature. |
| `riir-neuron-db` | Private (internal) | **Neuron-shard leaf crate** — `NeuronShard` weight blob, `ShardIndex`, generic `MerkleTree`/`MerkleProof`, `MerkleFrozenEnvelope`, MAPE-K self-healing, Raven/δ-Mem consolidation, AnyRAG gateway, vibe KG triples. No chain dependency. |
| `riir-train` | Private (internal) | **Training research** — adapter training methods, training data, trained weights. Know-how vault. |
| `riir-game-sdk` | Private (internal) | **Game-vocabulary SDK** — facade over `riir-games-shared` (Layer 0 vocabulary) + `riir-games` systems; dev-tool workspace hosting `crates/riir-viz` + `crates/riir-gm-tool`. Consumed by `mmorpg-remaster`, `riir-mmorpg-examples`. The dev entry point for the game stack — see `riir-game-sdk/AGENTS.md` facade constraint. |
| `riir-dapps` | Private (internal) | **dApp layer** — composes game outcomes into generic chain settlements. Depends on the chain side ONLY; the game layer calls *into* it. Game vocabulary in, `Settlement` out; a game predicate crosses as an opaque 32-byte hash so the ledger never learns "quest". See its `AGENTS.md`. |
| `riir-armageddon` | Private (internal) | **Arena/game-product domain types** — raw-vs-latent boundary for the game-product domain (the arena). Read its README; do not put research, chain code, or training data here. |

**Rule: anything `riir-*` is internal. No exceptions.**

> **Historical note:** the original "5-repo quintet" terminology referred to
> the 5 distillation targets (`katgpt-rs` + `riir-ai` + `riir-chain` +
> `riir-neuron-db` + `riir-train`). `riir-game-sdk` (game vocabulary facade
> + dev-tool workspace) and `riir-armageddon` (arena/game-product domain
> types) were added later as the stack grew consumer-facing tooling and a
> product-domain boundary. The "quintet" framing is retained only in
> historical documents and revision histories.

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

## The Second Axis: Layering (game / dApp / chain)

The table above is the **public/private** axis. It says nothing about *which
private repo* a game concern goes in, and that gap let game rules land inside
the chain's consensus-critical program set (`riir-chain/src/programs/`
quest/bounty/crafting — filed as `riir-chain` Issue 096, 2,294 LOC, one of them
moving no money at all).

The dApp layer that closes it is **`riir-dapps`** (private, created 2026-08-20,
`riir-chain` Issue 096 T1 / `riir-dapps` Plan 001) — the 8th repo. It was made a
separate repo rather than a crate under either end deliberately: under
`riir-chain` the chain repo would again contain "quest", and under
`riir-game-sdk`/`riir-ai` the game repo would depend on the ledger — which is
what `riir-games-civ/src/civ/latcal_wire.rs` already does wrong. A separate repo
makes the dependency direction checkable, and
`riir-dapps/scripts/direction_gate.sh` checks it.

**The rule: a layer is defined by what it must agree on, not by what it stores.**

| Layer | Must agree on | Repo | Examples |
|---|---|---|---|
| **game** | nothing globally — local rules, content, progress | `riir-game-sdk` (vocabulary + systems), `riir-ai` (`riir-games`) | recipe tables, quest objectives, kill-credit, class balance, loot rules |
| **store** | durability, not consensus | `riir-neuron-db` | quest progress, experience graph, local KV |
| **dApp** | how a game outcome becomes a chain instruction | `riir-dapps` (private) | "quest completed → claim escrow", "craft succeeded → mint NFT" |
| **chain** | value, authority, unmanipulable randomness | `riir-chain` | token transfer, escrow, staking, `FairRng` commit-reveal, AOI reveal filter |

### The test — three questions, all must pass

Revised 2026-08-20: the original one-question form ("does it move value or bind
authority") was **insufficient**. It admits FAME as "value" and says nothing
about write rate, which is what actually disqualifies game traffic.

**1. Product.** *Would a customer using the chain for commerce want this in
their dependency?* Ask this first — it needs no benchmark and reaches the right
verdict instantly. An NFT: **yes**, an NFT *is* a token, with its own value,
that trades and settles. A quest, a recipe table, a kill-credit predicate:
**no**.

**2. Value.** *Is it BigInt fungible currency, a token, or an authority
binding?* FAME, XP, items, reputation, karma and quest progress are **game
scalars** — not fungible, not BigInt, not DeFi. A number you can gain and lose
is not money.

**3. Rate.** *Does the write rate fit a Glacial tier (≤ 0.1 Hz)?* This binds
hardest: a BigInt currency transfer at 20 Hz is still wrong. Measured —
`riir-neuron-db` is **1,627× cheaper per write** than the chain (826 ns vs
1,344.5 µs), and one chain transaction at 10⁵ accounts costs **31.6 ms, 63% of
an entire 20 Hz hot tick**. `riir-game-sdk`'s `tick_tier_model.md` already
placed quorum at Glacial and said *"chain heartbeat; never 20Hz"*.

A game system with **no transaction** never belongs in a ledger, however
naturally it fits the account layout. Storability is not jurisdiction. **And
most game systems have no transaction** — a free quest ("kill 10 boars, get
FAME") settles nothing at all, so the paid quest is the exception, not the
model.

The weights are inverted, which is why one substrate cannot serve both:
**game = 80 fast / 20 secure** (`riir-neuron-db`: BLAKE3-committed, keyed row
MACs, Merkle-frozen — fast *and* secure enough), **defi = 20 fast / 80 secure**
(`riir-chain`: quorum, split-key, determinant audits). A quest turn-in does not
need Byzantine agreement between mutually distrusting validators; it needs an
authenticated write.

Note the "quest/economy tuning → `riir-ai`" row in the Decision Rules table
below already implied the layering; it was routed as *IP secrecy* and read by
no one as a *layering* constraint, which is exactly how the drift happened.

### When a game feature genuinely needs settlement: split it

The chain gets the game-agnostic half under a game-agnostic name; the game keeps
the predicate. A "quest reward" is a *multi-claim, deadline-bounded conditional
escrow* plus a claim predicate the chain must not be able to name — ship the
escrow, pass the predicate as an opaque 32-byte hash. A chain primitive whose
doc comment says "kill contract" has already lost the boundary.

Inverse direction, same principle: `FairRng` needs both split-key halves, so an
unmanipulable roll is a genuine chain **service**. The game layer *calls* it; it
does not host it — and hosting a recipe table next to it is not justified by
needing the roll.

### Anti-pattern — a feature gate that hides the impl but not the vocabulary

`riir-chain` gates each game program behind `chain_prog_*`, which looks like it
keeps a commerce customer clean. It does not: `LatCalIx` is **one
`#[repr(C, u8)]` enum, 88 variants, zero `cfg` attributes**
(`grep -c "cfg(feature" src/programs/cpi.rs` → `0`). Enable no program feature
at all and `QuestCreate` / `CraftingRegister` / `ReputationInit` are still in
your wire format. **An opt-in gate on the implementation is not a boundary if
the vocabulary is unconditional.**

*Status: RESOLVED by removal, not gating — riir-chain Issue 096 T4 (bounty,
quest), T3 (crafting), T7 (reputation, 2026-08-21) retired all four game
programs outright; no game variant exists in any build and the retired tags
(45..=48, 54..=57, 58..=61, 68..=71) are pinned undecodable.*

Worse for the fix: the enum has no explicit discriminants and the tag is read
straight out of the layout, so the wire tag is *declaration order* — removing a
game variant renumbers unrelated programs. Pin discriminants before deleting
anything (`riir-chain` Issue 096 T0/T8).

### Anti-pattern — game vocabulary as a chain type name

`ProgramId 13 = CRAFTING`, `BountyProgram`, `QuestCreate`. A wire constant
naming a game mechanic makes rebalancing content a **protocol change**: the
instruction set is versioned, proof-gated, and combinatorially CI'd, so a recipe
tweak pays consensus review costs forever. (Retiring such an ID is itself a
protocol change — retire, never reuse. All four game IDs — quest 17,
bounty 11, crafting 13, reputation 14 — were retired 2026-08-21,
`riir-chain` Issue 096 T3/T4/T7; their tags are dead holes, pinned
undecodable.) The mirror-image anti-pattern is a chain-vocabulary type in
a game crate reaching for `LatCalIx` directly, which
`riir-ai/crates/riir-games-civ/src/civ/latcal_wire.rs` did before 2026-08-21
— it now composes through `riir-dapps` (`riir-dapps` Plan 001 §3.1), the
layer built for exactly that call.

**Naming-only exception:** anti-cheat is *chain*, even when it is named for the
game. `riir-chain`'s `game_trust_flag` / `TrustFlag` is driven by consensus
anomalies and feeds inclusion probability — misnamed, correctly placed. It was
renamed to `chain_trust_flag` on 2026-08-20 (`riir-chain` Issue 096 T5); the
old name remains a deprecated Cargo alias until
`riir-ai/crates/riir-games` (`game_replay_verify`) migrates off it.

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
