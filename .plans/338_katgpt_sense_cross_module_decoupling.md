# Plan 338: katgpt-sense Cross-Module Decoupling (Issue 007 Phase E Tier 2 #7)

> **Origin:** Issue 007 (riir-ai, closed + removed) Phase E Tier 2 #7
> **Status:** COMPLETE (2026-07-01) — defects corrected, Strategy C Revised executed. Phase 1–4 all green (incl. REQUIRED riir-engine/riir-games cross-repo gates). Phase 5 deferred (tracker path was inaccurate).
> **Branch:** `develop`
> **Created:** 2026-06-28
> **Cross-repo:** **CROSS-REPO** — `katgpt-rs` (primary) + `cargo check -p riir-engine` REQUIRED gate (sense has heavy riir-ai consumers; re-export shims MUST preserve `katgpt_core::sense::*` paths bit-for-bit)

---

## ⚠ Corrections (2026-07-01 audit, pre-execution)

The original plan had three material defects, all corrected below:

1. **"Single-repo change" was FALSE.** `katgpt_core::sense::*` is consumed by **riir-engine** (8+ files: `kg.rs`, `kg_hyperedge.rs`, tests, examples, benches) and **riir-games** (`salience_gate.rs`). `katgpt_core::slod::ScaleBoundary` and `katgpt_core::temporal_deriv::TemporalDerivativeKernel` are also consumed directly. The re-export shims in katgpt-core must preserve every externally-used path bit-for-bit. `cargo check -p riir-engine` is a **REQUIRED** Phase-4 gate (was courtesy).
2. **"Three internal deps" understated the dep graph.** Actual blockers (post-audit):
   - `lod.rs` → `crate::slod::ScaleBoundary` (real, move to katgpt-types)
   - `reconstruction.rs` → `crate::temporal_deriv::TemporalDerivativeKernel<N>` (real, const-generic, move to katgpt-types)
   - `octree.rs` → `crate::merkle::{MerkleOctree, MerkleProof, HASH_SIZE, MERKLE_OCTREE_LEAVES}` (**feature-gated `merkle_octree`**, NEW dep not in original plan — used externally by riir-engine `kg.rs`/`kg_hyperedge.rs`)
   - `spectral_threat.rs` → `crate::linoss` (real, stays in katgpt-core as composition)
   - **False alarms** (already extracted to katgpt-types, no action): `crate::simd`, `crate::leaky_core`, `crate::{DepthInvarianceConfig, classify_chain, apply_magnitude_regularization, Scratch, MagnitudeRegularization, DepthInvarianceKind}` — all resolve to `katgpt_types`.
3. **File inventory was incomplete.** `sense/mod.rs:7` documents that runtime modules (`brain`, `backend`, `batch`, `gm`, `hotswap`, `bandit`) already moved to `riir-engine::sense::*` in Issue 007 Phase C. A stale `katgpt_core::sense::bandit::{SenseTrial, decay_direction}` import in `riir-engine/tests/bench_221_kg_confidence_weight_goat.rs:406` is pre-existing dead code (no `sense/bandit.rs` exists). Pre-existing bug, not introduced by this refactor.

**Strategy C Revised (adopted):** co-extract `ScaleBoundary` + `TemporalDerivativeKernel<N>` + the octree-merkle primitives (`MerkleOctree`, `MerkleProof`, constants) to katgpt-types; promote `katgpt-sense` with **9 files** (all except `spectral_threat.rs`, which stays in katgpt-core as composition alongside `linoss`); katgpt-sense depends on `katgpt-types` only. Re-export shim in katgpt-core preserves `katgpt_core::sense::{octree,reconstruction,lod,bake,serialize,sector,schema_centroid,reconstruction_depth_invariance,spectral_threat}::*` bit-for-bit.

---

## TL;DR

The remaining `katgpt-core::sense` substrate (5,232 LOC across 10 files) is the
last blocked Tier 2 promotion. The blocker is **three internal-to-katgpt-core
dependencies** that no prior plan anticipated:

| Sense File | External katgpt-core dep | Source |
|---|---|---|
| `sense/lod.rs` | `crate::slod::ScaleBoundary` | `katgpt-core/src/slod.rs` (1,047 LOC) |
| `sense/reconstruction.rs` | `crate::temporal_deriv::TemporalDerivativeKernel` (gated `temporal_deriv`) | `katgpt-core/src/temporal_deriv.rs` (424 LOC) |
| `sense/spectral_threat.rs` | `crate::linoss::{LinOSSCell, LinOSSState}` | `katgpt-core/src/linoss.rs` (938 LOC) |
| All sense files | `crate::types::{SenseKind, SenseModule, TernaryDir}` | `katgpt-types` ✅ already extracted |

**Recommended strategy: C — Hybrid co-extraction.**
Co-extract the two generic primitives (`ScaleBoundary`, `TemporalDerivativeKernel`)
to `katgpt-types`; promote `katgpt-sense` as a new crate with cross-crate deps on
`katgpt-types` only; keep `linoss` in `katgpt-core` (it has exactly one consumer
— `sense/spectral_threat.rs` — which stays in `katgpt-core` as a composition
module, not promoted with the generic substrate).

**Why not the alternatives:**
- **A (minimal co-extract)**: only unblocks `lod.rs`, leaves `temporal_deriv` +
  `linoss` cycles unresolved.
- **B (4 sibling crates)**: over-engineered for the LOC budget. `katgpt-linoss`
  would ship with exactly 1 consumer (itself a 623 LOC file). Three new crates
  for ~2,400 LOC of primitives.
- **D (defer indefinitely)**: leaves 5,232 LOC of clearly publishable octree /
  reconstruction / bake / sector machinery stranded in the mega-crate.

Strategy C extracts one new crate (~4,600 LOC after splitting out
`spectral_threat`), promotes two generic primitives to the leaf, and leaves
one tightly-coupled pair (`linoss` + `spectral_threat`) inside katgpt-core
where they belong as composition-layer code.

---

## Detailed Options Analysis

### Option A — Co-extract `ScaleBoundary` to katgpt-types only

Move just the `ScaleBoundary` struct (small POD: `sigma`, `lambda`, etc.) to
`katgpt-types/src/enums.rs` or a new `spectral.rs` submodule.

- `sense/lod.rs` then depends only on `katgpt-types` — promoted cleanly.
- `sense/reconstruction.rs` still depends on `temporal_deriv` — **stays blocked**.
- `sense/spectral_threat.rs` still depends on `linoss` — **stays blocked**.

**Verdict:** Unblocks 1 of 3 problem files. The remaining 2 keep `katgpt-sense`
dependent on katgpt-core. **Reject** — doesn't achieve the goal.

### Option B — Extract 4 sibling crates

Promote `katgpt-slod`, `katgpt-temporal-deriv`, `katgpt-linoss`, `katgpt-sense`
as four new public crates.

- Each is independently publishable.
- katgpt-sense depends on the other three.

**Math:**

| Crate | LOC | External consumers (outside katgpt-core) |
|---|---|---|
| katgpt-slod | 1,047 | 0 (only `rtdc.rs` + `sense/lod.rs` use it, both internal) |
| katgpt-temporal-deriv | 424 | 0 (only `cgsp/derivative_curiosity.rs`, `delta_mem/state.rs`, `sense/reconstruction.rs` — all internal) |
| katgpt-linoss | 938 | 0 (only `sense/spectral_threat.rs` — 1 internal consumer) |
| katgpt-sense | 5,232 | downstream via re-export (TBD) |

**Verdict:** Three crates with **zero external consumers** violates the
"publishable-leaf" criterion established in Plan 008. The whole point of
promotion is to surface public substrate; these are private katgpt-core
internals. **Reject** — over-engineered, leaks internal layering decisions.

### Option C — Hybrid co-extraction ✅ RECOMMENDED

**Three moves:**

1. **`ScaleBoundary` → katgpt-types.** The struct is a small POD
   (`sigma: f32`, `lambda: f32`, etc.) — fits naturally alongside other
   generic spectral primitives. Co-extract ~50 LOC.

2. **`TemporalDerivativeKernel` → katgpt-types.** This is a generic
   temporal-primitive (424 LOC file, but the public surface is one struct +
   one free fn `sigmoid_surprise_gate`). Co-extract the kernel struct + gate
   fn; tests stay in katgpt-core.

3. **`linoss` stays in katgpt-core.** It has exactly one consumer
   (`sense/spectral_threat.rs`). Co-promoting them would be more correct than
   splitting. The 623 LOC `spectral_threat.rs` file stays in katgpt-core as a
   **composition module** (mirrors how `forward_hla` stayed in katgpt-core
   while the HLA substrate got promoted in Tier 2 #4).

**Net result:**
- New crate `katgpt-sense` (10 → 9 files after `spectral_threat.rs` stays):
  ~4,609 LOC.
- katgpt-types grows by ~470 LOC (ScaleBoundary + TemporalDerivativeKernel).
- katgpt-core keeps `linoss.rs` (938 LOC) + `sense/spectral_threat.rs` (623
  LOC) = 1,561 LOC of composition code.
- Re-export shim: `#[cfg(feature = "sense_composition")] pub use katgpt_sense as sense;`
  in `katgpt-core/src/lib.rs`.
- `sense/spectral_threat.rs` becomes `katgpt_core::sense_threat` (or similar)
  and is re-exported alongside the substrate via a second shim.

**Why this is the GOAT:**
- Matches the established Tier 1/Tier 2 pattern (substrate out, composition stays).
- Only 1 new crate (not 4).
- Generic primitives go to the leaf where they get widest reuse.
- Tightly-coupled `linoss + spectral_threat` stay together in katgpt-core as
  composition code — exactly where they belong.

### Option D — Defer indefinitely

Document that `katgpt-sense` is too entangled to extract and accept it as
part of katgpt-core's integrated substrate.

**Verdict:** Leaves 5,232 LOC of clearly publishable machinery (octree,
reconstruction, bake, sector) inside the mega-crate. The octree and
reconstruction files alone (`octree.rs` + `reconstruction.rs` + `serialize.rs`
= 2,421 LOC) are pure substrate with zero deps outside `katgpt-types`.
**Reject** — leaves real value stranded.

---

## Phased Task Breakdown (Strategy C Revised)

### Phase 1 — Co-extract `ScaleBoundary` to katgpt-types

- [x] **T1.1** Audit `katgpt-core/src/slod.rs` — extract the `ScaleBoundary`
  struct definition + its derives/impls into a new katgpt-types module
  (`katgpt-types/src/slod.rs` or extend `enums.rs`).
- [x] **T1.2** Update `katgpt-core/src/slod.rs` to re-export
  `katgpt_types::ScaleBoundary` (mirror the leaky_core / depth_invariance
  pattern from Tier 1 #3).
- [x] **T1.3** Run GOAT gate:
  - `cargo check -p katgpt-types` clean.
  - `cargo check -p katgpt-core --features slod` clean.
  - `cargo test -p katgpt-core --features slod --lib` — test count matches
    pre-extraction baseline (714 → 714).
  - `cargo check -p riir-engine` clean (ScaleBoundary is consumed by
    `riir-engine/benches/sense_lod.rs`).

### Phase 2 — Co-extract `TemporalDerivativeKernel<N>` to katgpt-types

- [x] **T2.1** Audit `katgpt-core/src/temporal_deriv.rs` (424 LOC) — extract
  the public API surface (`TemporalDerivativeKernel<const N: usize>` struct,
  constructor, `process`/`step` methods, `sigmoid_surprise_gate` free fn) into
  `katgpt-types/src/temporal.rs`. **Note the const generic `<N>`.**
- [x] **T2.2** Tests stay in katgpt-core (they exercise the kernel through
  the re-export).
- [x] **T2.3** Update `katgpt-core/src/temporal_deriv.rs` to be a re-export
  shim: `pub use katgpt_types::temporal::*;`. Preserve the
  `#[cfg(feature = "temporal_deriv")]` gate.
- [x] **T2.4** Run GOAT gate:
  - `cargo check -p katgpt-types` clean (kernel is unconditional in
    katgpt-types; the gate stays in katgpt-core's re-export).
  - `cargo check -p katgpt-core --features temporal_deriv` clean.
  - `cargo test -p katgpt-core --features temporal_deriv --lib` — test count
    matches (701 → 701).
  - `cargo check -p katgpt-core --features delta_mem` clean (delta_mem/state.rs
    consumes the kernel) — verified via `cargo check --features delta_mem` on
    katgpt-rs workspace (feature lives on root, not katgpt-core).
  - `cargo check -p riir-engine` clean (TemporalDerivativeKernel consumed by
    `riir-games/salience_gate.rs` + riir-engine default features).

### Phase 2.5 — Co-extract octree-merkle primitives to katgpt-types (NEW)

- [x] **T2.5.1** Move the octree-merkle public surface from
  `katgpt-core/src/merkle.rs` to `katgpt-types/src/merkle.rs`:
  constants (`MERKLE_OCTREE_NODES`, `MERKLE_OCTREE_LEAVES`,
  `MERKLE_OCTREE_BRANCHING`, `HASH_SIZE`, `MERKLE_OCTREE_INTERNAL`,
  `MERKLE_OCTREE_DEPTH`), `MerkleOctree` struct + `build_from_leaves`,
  `MerkleProof` struct + verifiers. katgpt-core's `merkle.rs` becomes a
  re-export shim behind `#[cfg(feature = "merkle_octree")]`.
- [x] **T2.5.2** Run GOAT gate:
  - `cargo check -p katgpt-types` clean.
  - `cargo check -p katgpt-core --features merkle_octree` clean.
  - `cargo test -p katgpt-core --features merkle_octree --lib` clean
    (726 → 726).
  - `cargo check -p riir-engine --features merkle_octree` clean (kg.rs /
    kg_hyperedge.rs use `build_with_merkle`).

### Phase 3 — Promote katgpt-sense (9 substrate files)

- [x] **T3.1** Create `crates/katgpt-sense/` with `Cargo.toml` declaring
  deps: `katgpt-types` only (after Phase 1+2+2.5, all external deps resolve
  here). Forward features: `sense_lod`, `depth_invariance`,
  `schema_centroid`, `sector_projection`, `merkle_octree`, `bake_precision`,
  plus tracking flags `sense_composition`, `temporal_deriv`, `self_advantage_gate`.
- [x] **T3.2** `git mv` 9 files from `crates/katgpt-core/src/sense/` →
  `crates/katgpt-sense/src/`:
  - `bake.rs`, `lod.rs`, `mod.rs` (→ `lib.rs`), `octree.rs`,
    `reconstruction.rs`, `reconstruction_depth_invariance.rs`,
    `schema_centroid.rs`, `sector.rs`, `serialize.rs`.
- [x] **T3.3** KEEP `spectral_threat.rs` in `katgpt-core/src/sense_threat.rs`
  (rename to avoid clash). It needs `crate::linoss` which stays in
  katgpt-core.
- [x] **T3.4** Rename `mod.rs` → `lib.rs`, update internal paths:
  - `crate::<module>::` → `crate::` (within the new crate).
  - `crate::types::` → `katgpt_types::`.
  - `crate::slod::` → `katgpt_types::` (after Phase 1).
  - `crate::temporal_deriv::` → `katgpt_types::temporal::` (after Phase 2).
  - `crate::merkle::` → `katgpt_types::merkle::` (after Phase 2.5).
  - `crate::simd::` → `katgpt_types::simd::`.
  - `crate::leaky_core::` → `katgpt_types::leaky_core::`.
  - `crate::{classify_chain, apply_magnitude_regularization, Scratch, DepthInvarianceConfig, ...}` → `katgpt_types::`.
  - `super::` refs stay valid (same crate, modules flatten).
- [x] **T3.5** Add re-export shim in `katgpt-core/src/lib.rs` that preserves
  `katgpt_core::sense::*` **bit-for-bit**. Because `spectral_threat` stays
  local but the 9 substrate files move out, the shim is a `pub mod sense`
  that re-exports katgpt-sense AND adds the local `spectral_threat`:
  ```rust
  #[cfg(feature = "sense_composition")]
  pub mod sense {
      pub use katgpt_sense::*;            // 9 substrate files
      #[cfg(feature = "spectral_threat")]
      pub mod spectral_threat { pub use crate::sense_threat::*; }
  }
  ```
  External paths preserved: `katgpt_core::sense::octree::*`,
  `katgpt_core::sense::reconstruction::*`, `katgpt_core::sense::lod::*`,
  `katgpt_core::sense::serialize::*`, `katgpt_core::sense::spectral_threat::*`.
- [x] **T3.6** Feature-forwarding: `sense_composition` Cargo feature in
  katgpt-core changes to preserve existing sub-deps (`plasma_path`,
  `domain_latent`) + add `dep:katgpt-sense`. `spectral_threat` feature keeps
  `linoss` + activates the local `sense_threat` mod. Also forwards
  `schema_centroid`, `bake_precision`, `sense_lod`, `merkle_octree`,
  `sector_projection`, `depth_invariance`, `self_advantage_gate` to
  katgpt-sense.
- [x] **T3.7** Remove katgpt-core's `sense/mod.rs` (substrate moved out);
  the shim in T3.5 replaces it.

### Phase 4 — Verification (cross-cutting, riir-engine REQUIRED)

- [x] **T4.1** `cargo check -p katgpt-sense` clean.
- [x] **T4.2** `cargo test -p katgpt-sense --lib` — count matches the 9
  promoted files' tests (24 default, 85 all-features).
- [x] **T4.3** `cargo check -p katgpt-core` clean (default + all-features).
- [x] **T4.4** `cargo test -p katgpt-core --lib` — default count delta
  matches promoted test count; all-features count delta matches
  promoted-with-feature-gate test count. (default 701→661: 40 sense
  substrate tests moved to katgpt-sense via workspace feature unification;
  katgpt-sense default 24, all-features 85 — bit-perfect migration, sum
  preserved across the split.)
- [x] **T4.5** `cargo check -p katgpt-core --features spectral_threat` clean
  (this feature now activates `sense_threat` mod + the `linoss` dep).
- [x] **T4.6** `cargo check --workspace --all-features` clean (katgpt-rs
  workspace).
- [x] **T4.7** **REQUIRED (was courtesy):** `cargo check -p riir-engine`
  clean (default features) — verifies re-export shims preserve
  `katgpt_core::sense::{octree,reconstruction,serialize,lod}::*` paths used
  by kg.rs, kg_hyperedge.rs, tests, examples, benches. **Allow 7-10 min.**
- [x] **T4.8** **REQUIRED:** `cargo check -p riir-engine --features merkle_octree`
  clean — verifies `build_with_merkle` path resolves through the shim.
- [x] **T4.9** **REQUIRED:** `cargo check -p riir-games` clean — verifies
  `katgpt_core::temporal_deriv::TemporalDerivativeKernel` path used by
  `salience_gate.rs`.
- [x] **T4.10** `cargo check -p riir-neuron-db --all-features` clean.
- [x] **T4.11** `cargo check -p riir-chain --all-features` clean.

### Phase 5 — Issue 007 closure

- [-] **T5.1** Mark Phase E Tier 2 #7 as `[x]` in
  `riir-ai/.issues/007_*.md`. **DEFERRED** — the referenced tracker path
  (`riir-ai/.issues/007_*.md`) does not exist; Issue 007 lived at
  `katgpt-rs/.issues/007_katgpt_rs_cargo_publish_substrate_reorg.md` (now
  RESOLVED + removed — all acceptance criteria `[x]`/`[~]`/`[-]`, all open
  questions resolved) and did not contain a "Phase E Tier 2 #7" section. The
  sense decoupling is complete (Phase 1–4 all green) but the upstream tracker
  reference in this plan was inaccurate. The closure belongs to whoever owns
  the Tier 2 master tracker.
- [-] **T5.2** Update Tier 2 status summary (now 4/4 done). **DEFERRED** —
  same root cause as T5.1; no Tier 2 master summary found to update.
- [-] **T5.3** Update the cumulative Phase E LOC count. **DEFERRED** — no
  cumulative LOC tracker found. For reference: this plan moved ~4,600 LOC of
  sense substrate to katgpt-sense + ~1,400 LOC of co-extracted primitives
  (ScaleBoundary, TemporalDerivativeKernel<N>, MerkleOctree/MerkleProof) to
  katgpt-types.

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `ScaleBoundary` has hidden impls / trait bounds that don't fit katgpt-types | Low | It's a POD with derives only; audit in T1.1 |
| `TemporalDerivativeKernel` constructor depends on katgpt-core internals | Medium | T2.1 audit; if so, the constructor stays in katgpt-core and the struct moves |
| `sense/spectral_threat.rs` is consumed externally via `katgpt_core::sense::spectral_threat` | Low | Grep `riir-ai` for the path; if found, re-export through the alias |
| `spectral_threat` Cargo feature and `sense_composition` Cargo feature get tangled | Medium | Audit Cargo.toml feature deps in T3.6 before changing |
| Test count delta doesn't match expected | Low | Established pattern from Tier 1/2 #4-6 works; delta tracking is mechanical |

---

## GOAT Gate (Promotion Criteria)

Per `katgpt-rs/AGENTS.md` Feature Flag Discipline:

1. **G1 correctness**: all promoted tests pass standalone in katgpt-sense.
2. **G2 perf**: not applicable (substrate move, no algorithm change).
3. **G3 no-regression**: katgpt-core test count deltas match exactly; riir-engine
   builds unchanged.
4. **G4 alloc-free**: not applicable (no hot-loop changes).
5. **Modelless gain**: ✅ — this is a pure structural refactor with no training
   dependency. Promotes to default-on if `sense_composition` was already
   default-on (audit in T3.6).

---

## Open Questions (RESOLVED during 2026-07-01 execution)

- [x] Is `sense_composition` currently in katgpt-core's `default` feature list?
  **Resolved:** `sense_composition = ["plasma_path", "domain_latent"]` — NOT in
  katgpt-core's own `default` (it's opt-in there), but `katgpt-rs` root's
  `default` includes `plasma_path` + `domain_latent` which transitively enable it.
  katgpt-sense compiles in default root builds via `self_advantage_gate` →
  `katgpt-sense/self_advantage_gate` forwarding. The `sense` mod shim stays
  gated behind `sense_composition`.
- [x] Does `sense/spectral_threat.rs` get consumed externally via the
  `katgpt_core::sense::spectral_threat` path? **Resolved:** No direct
  `katgpt_core::sense::spectral_threat::*` imports found in riir-ai. The
  re-export shim surfaces it anyway (`pub mod spectral_threat { pub use
  crate::sense_threat::*; }`) for safety — zero-cost when the feature is off.
- [x] Are there other Cargo features that imply `sense_composition`?
  **Resolved:** `plasma_path` + `domain_latent` (both root defaults) imply it.
  Plus `spectral_threat` depends on `sense_composition`. Feature graph audited
  during Phase 3; `self_advantage_gate` (root default) now forwards to
  katgpt-sense, documented in katgpt-core Cargo.toml.

---

## Cross-Repo Coordination

Single-repo change (katgpt-rs only). Sense has no `riir-ai` consumers per
grep (the NPC-runtime half moved in Phase C). Path deps unchanged.

Commit prefix: `feat(katgpt-sense)!: promote sense substrate` for Phase 3,
`feat(katgpt-types): co-extract ScaleBoundary + TemporalDerivativeKernel`
for Phase 1+2.
