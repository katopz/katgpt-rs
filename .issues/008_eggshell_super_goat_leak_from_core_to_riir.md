# Issue 008: Eggshell Super-GOAT IP Leaked Into Public `katgpt-core`

> **Type:** Architecture / IP-boundary violation (Super-GOAT moat leak)
> **Status:** Open
> **Owner:** develop
> **Created:** 2026-06-27
> **Severity:** HIGH — game IP is visible in a public MIT-licensed crate that
> ships to crates.io (`katgpt-core`). This is a commercial moat leak.
> **Cross-repo:** katgpt-rs (source — leak) → riir-ai (destination — private).
> **Origin:** Plan 330 (analytic lattice) + the original Plan 335 (eggshell:
> interest cochains, lattice utility) both landed game-IP code in
> `katgpt-core`, which is the public leaf crate that publishes to crates.io.
> **Rule violated:** "you cant leak egg/shell to core, it's super goat pillar
> and sec riir" (user directive 2026-06-27). Eggshell = the lattice/interest/
> HLA-bridge substrate that constitutes a Super-GOAT commercial pillar. It
> must stay in the private repos (riir-ai), never in public katgpt-rs.

---

## TL;DR

Three surfaces of eggshell (Super-GOAT) game IP currently live in
`katgpt-rs/crates/katgpt-core/` — the public MIT crate that publishes to
crates.io. The worst offender (`dec/terrain_cochains.rs`) ships
**unconditionally** (no feature gate) whenever `dec_operators` is on (which is
the default). This exposes NPC cognitive-navigation IP (fame/notability
interest fields, projectile-threat safety fields, HLA→cochain affective
weights) to anyone reading crates.io. All three surfaces must migrate to
`riir-ai` (private), keeping only the genuinely-generic DEC math in
`katgpt-core`.

This is NOT blocking the current Plan 335 (paired loss diagnostic) work —
paired loss is generic math (subtract + tag-stratify + log-vocab bound), not
eggshell. This issue tracks the separate eggshell cleanup.

---

## Leaked surfaces (in scope)

### 1. `dec/terrain_cochains.rs` — WORST (unconditional leak)

Ships **unconditionally** under `dec_operators` (default-on). No feature gate
on `pub mod terrain_cochains` in `dec/mod.rs`. Five game-IP cochain types:

| Type | Game IP |
|---|---|
| `InterestCohain` | fame/notability interest field (Gaussian-falloff anchors) |
| `SafetyCochain` | projectile-threat safety field |
| `ThreatCochain` | per-edge threat cochain |
| `OccupancyCochain` | per-face occupancy cochain |
| `DestructionCohain` | per-edge destruction-tolerance cochain |

All five are NPC cognitive-navigation domain types. `InterestCohain::from_anchors`
(fame/notability placement) is a selling-point moat primitive. None of this
is generic DEC math — it's game semantics layered on the generic
`CochainField` substrate.

**Re-exported unconditionally** from `dec/mod.rs`:
`pub use terrain_cochains::{DestructionCochain, InterestCohain, OccupancyCochain, SafetyCochain, ThreatCochain};`

### 2. `dec/lattice_utility.rs` — gated but still public-source

Gated behind `lattice_utility = ["dec_operators"]` feature (opt-in). Ships:

| Item | Game IP |
|---|---|
| `HlaToCohainWeights` | HLA affective-state → cochain bridge (curiosity/calm/fear/desperation lanes) |
| `lattice_edge_utility_into` | NPC per-edge traversal utility (curiosity × interest + calm × safety·occupancy − fear × threat + desperation × destruction) |

The function signature itself (interest_lane, safety_lane, occupancy_lane,
threat_lane, destruction_lane, hla_weights) reveals the full NPC cognitive-
navigation model. Even though the feature is opt-in, the **source code** is
visible in the public repo. The IP leak is the source, not the binary.

**Re-exported** from `dec/mod.rs` under the same feature gate:
`pub use lattice_utility::{HlaToCohainWeights, lattice_edge_utility_into};`

### 3. `analytic_lattice/` — cosmetic eggshell coupling (evaluate)

The math (`compose_chain` row-major matmul, `batch_compose_chain` prefix
factoring, `direction_vector_decode` SIMD dot-product, `spectral_audit`) is
**genuinely generic**. The eggshell coupling is cosmetic — comments
referencing "eggshell lanes (k=8)" and the `LatticeVector<N>` const-generic
default mentioned alongside Plan 335's transport lanes.

**Decision needed:** does the generic math stay in core (it's reusable by
non-eggshell consumers) or move to riir-ai (it's only consumed by eggshell
today)? Default recommendation: the math stays (it's generic), but strip
the eggshell references from comments + remove the k=8 eggshell framing from
`LatticeVector`'s doc. The ASOC trait shapes (`PlasmaDraft`, `RederiveOp`)
are generic trait shapes — stay.

---

## Migration plan (high level — NOT executed in this issue)

1. **Move `terrain_cochains.rs` → `riir-ai/crates/riir-engine/src/dec/terrain_cochains.rs`**
   (or a new riir-ai crate/module). It depends only on
   `katgpt-core::dec::types::{CellComplex, CochainField}` (generic), which
   stays in core. Update `dec/mod.rs` to drop the unconditional
   `pub mod terrain_cochains` + `pub use`. Remove the
   `interest_cochain` feature from `katgpt-core/Cargo.toml`.

2. **Move `lattice_utility.rs` → `riir-ai`** alongside the terrain cochains
   (it bridges HLA → terrain cochains; both are game IP). Update `dec/mod.rs`
   to drop `pub mod lattice_utility` + `pub use`. Remove the
   `lattice_utility` feature from `katgpt-core/Cargo.toml`.

3. **Audit `analytic_lattice/`** — strip eggshell references from comments.
   Keep the generic math + trait shapes in core unless a review concludes
   it's eggshell-only infrastructure. Update `LatticeVector` doc to drop the
   "Plan 335's transport lanes (k=8)" framing.

4. **Update Cargo.toml comments** — the "Plan 335 Phase 1 — InterestCohain"
   and "Plan 335 Phase 5 — SIMD lattice-edge utility op" comments are stale
   (Plan 335 was reused for paired loss). After migration these features are
   deleted entirely.

5. **Verify no katgpt-core consumer breaks.** `riir-engine` will consume the
   migrated types via a path-dep on riir-ai (inverted direction is fine —
   riir-ai already depends on katgpt-core, and the terrain types are
   engine-side not core-side).

6. **Run GOAT gate** — `cargo check --all-features` on katgpt-core must stay
   clean (verify no `interest_cochain`/`lattice_utility` references remain
   in core's own code).

---

## Why this matters

`katgpt-core` is the **public** leaf crate (MIT, ships to crates.io per
`release-plz.toml`). The 5-repo commercial strategy (AGENTS.md global rule)
puts the public engine in katgpt-rs and the private runtime/chain/neuron-db
in riir-*. Eggshell (NPC cognitive navigation: fame/notability, threat
avoidance, HLA affective steering) is explicitly a Super-GOAT pillar — it's
the selling-point moat. Having it in a public MIT crate defeats the moat:
anyone can read the source on crates.io and replicate the cognitive model.

The fix is mechanical (move files + update re-exports + delete features) but
cross-repo, so it's tracked here rather than mixed into feature work.

---

## Out of scope

- `dec/operators.rs`, `dec/hodge.rs`, `dec/flow.rs`, `dec/stokes_calculus.rs`,
  `dec/types.rs`, `dec/cache.rs`, `dec/backend.rs` — these are **generic DEC
  math** (exterior derivative, codifferential, Hodge decomposition, Stokes
  wrappers). They stay in core. Only the game-semantic cochain types
  (`terrain_cochains.rs`) and the HLA-bridge utility (`lattice_utility.rs`)
  leak.
- `analytic_lattice/chain.rs`, `batch_chain.rs`, `decoder.rs`, `audit.rs`,
  `asoc.rs` — generic math + generic trait shapes. Stay (subject to the
  comment-cleanup note above).
- Plan 335 (paired loss diagnostic) — NOT eggshell. Generic measurement math.
  Unaffected by this issue.

---

## Acceptance

- [ ] `dec/terrain_cochains.rs` removed from katgpt-core; migrated to riir-ai.
- [ ] `dec/lattice_utility.rs` removed from katgpt-core; migrated to riir-ai.
- [ ] `interest_cochain` feature deleted from `katgpt-core/Cargo.toml`.
- [ ] `lattice_utility` feature deleted from `katgpt-core/Cargo.toml`.
- [ ] `dec/mod.rs` updated: no `pub mod terrain_cochains` / `pub mod lattice_utility`, no matching `pub use`.
- [ ] `analytic_lattice/` comments stripped of eggshell references (or full migration if review decides).
- [ ] `cargo check -p katgpt-core --all-features` clean (no dangling references).
- [ ] `cargo check -p katgpt-core --no-default-features` clean.
- [ ] riir-ai consumes the migrated types; its `cargo check` clean.
- [ ] Commit on `develop` with `refactor:` prefix (cross-repo: katgpt-rs delete + riir-ai add).

---

## TL;DR

Eggshell (NPC cognitive-navigation: fame/notability interest fields, threat
safety fields, HLA→cochain affective bridge) leaked into the public
`katgpt-core` crate (ships to crates.io). Worst offender
(`dec/terrain_cochains.rs`) ships **unconditionally** under default
`dec_operators`. Three surfaces need migration to riir-ai (private):
`terrain_cochains.rs`, `lattice_utility.rs`, and a comment-cleanup pass on
`analytic_lattice/`. NOT blocking Plan 335 (paired loss) — that's generic
math, not eggshell. Tracked here so the Super-GOAT moat cleanup isn't lost.
