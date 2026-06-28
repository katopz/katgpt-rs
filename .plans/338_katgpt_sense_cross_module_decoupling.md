# Plan 338: katgpt-sense Cross-Module Decoupling (Issue 007 Phase E Tier 2 #7)

> **Origin:** [Issue 007](../../../riir-ai/.issues/007_katgpt_runtime_ip_exfiltration_from_public_mit_repo.md) Phase E Tier 2 #7
> **Status:** Active — options analysis done, awaiting decision
> **Branch:** `develop`
> **Created:** 2026-06-28
> **Cross-repo:** katgpt-rs only (sense has no riir-ai consumers per grep)

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

## Phased Task Breakdown (Strategy C)

### Phase 1 — Co-extract `ScaleBoundary` to katgpt-types

- [ ] **T1.1** Audit `katgpt-core/src/slod.rs` — extract the `ScaleBoundary`
  struct definition + its derives/impls into a new katgpt-types module
  (`katgpt-types/src/spectral.rs` or extend `enums.rs`).
- [ ] **T1.2** Update `katgpt-core/src/slod.rs` to re-export
  `katgpt_types::ScaleBoundary` (mirror the leaky_core / depth_invariance
  pattern from Tier 1 #3).
- [ ] **T1.3** Run GOAT gate:
  - `cargo check -p katgpt-types` clean.
  - `cargo check -p katgpt-core --features slod` clean.
  - `cargo test -p katgpt-core --features slod --lib` — test count matches
    pre-extraction baseline.
  - `cargo check -p riir-engine` clean (no API surface change).

### Phase 2 — Co-extract `TemporalDerivativeKernel` to katgpt-types

- [ ] **T2.1** Audit `katgpt-core/src/temporal_deriv.rs` (424 LOC) — extract
  the public API surface (`TemporalDerivativeKernel` struct, constructor,
  `process`/`step` methods, `sigmoid_surprise_gate` free fn) into
  `katgpt-types/src/temporal.rs`.
- [ ] **T2.2** Tests stay in katgpt-core (they exercise the kernel through
  the re-export).
- [ ] **T2.3** Update `katgpt-core/src/temporal_deriv.rs` to be a re-export
  shim: `pub use katgpt_types::temporal::*;`. Preserve the
  `#[cfg(feature = "temporal_deriv")]` gate.
- [ ] **T2.4** Run GOAT gate:
  - `cargo check -p katgpt-types --features temporal_deriv` (if feature is
    propagated to katgpt-types — likely yes, since the kernel is now there).
  - `cargo check -p katgpt-core --features temporal_deriv` clean.
  - `cargo test -p katgpt-core --features temporal_deriv --lib` — test count
    matches.
  - `cargo check -p katgpt-core --features delta_mem` clean (delta_mem/state.rs
    consumes the kernel).

### Phase 3 — Promote katgpt-sense (substrate minus spectral_threat)

- [ ] **T3.1** Create `crates/katgpt-sense/` with `Cargo.toml` declaring
  deps: `katgpt-types` only (after Phase 1+2, all external deps resolve here).
- [ ] **T3.2** `git mv` 9 files from `crates/katgpt-core/src/sense/` →
  `crates/katgpt-sense/src/`:
  - `bake.rs`, `lod.rs`, `mod.rs` (→ `lib.rs`), `octree.rs`,
    `reconstruction.rs`, `reconstruction_depth_invariance.rs`,
    `schema_centroid.rs`, `sector.rs`, `serialize.rs`.
- [ ] **T3.3** KEEP `spectral_threat.rs` in `katgpt-core/src/sense_threat.rs`
  (rename to avoid clash). It needs `crate::linoss` which stays in
  katgpt-core.
- [ ] **T3.4** Rename `mod.rs` → `lib.rs`, update internal paths:
  - `crate::<module>::` → `crate::` (within the new crate).
  - `crate::types::` → `katgpt_types::`.
  - `crate::slod::` → `katgpt_types::` (after Phase 1).
  - `crate::temporal_deriv::` → `katgpt_types::temporal::` (after Phase 2).
- [ ] **T3.5** Add re-export shims in `katgpt-core/src/lib.rs`:
  ```rust
  #[cfg(feature = "sense_composition")]
  pub use katgpt_sense as sense;
  ```
  The `spectral_threat` module stays as a separate katgpt-core mod and is
  consumed via `katgpt_core::sense_threat::*` or re-exported through the
  `sense` alias if downstream compat requires.
- [ ] **T3.6** Feature-forwarding: `sense_composition` Cargo feature in
  katgpt-core changes from `[]` to `["dep:katgpt-sense"]`. Default-on / opt-in
  status preserved (audit current default-on list to confirm).
- [ ] **T3.7** Update katgpt-core's `sense/mod.rs` (if it survives) or remove
  it (likely removed — the substrate moved out).

### Phase 4 — Verification (cross-cutting)

- [ ] **T4.1** `cargo check -p katgpt-sense` clean.
- [ ] **T4.2** `cargo test -p katgpt-sense --lib` — count matches the 9
  promoted files' tests.
- [ ] **T4.3** `cargo check -p katgpt-core` clean (default + all-features).
- [ ] **T4.4** `cargo test -p katgpt-core --lib` — default count delta
  matches promoted test count; all-features count delta matches
  promoted-with-feature-gate test count.
- [ ] **T4.5** `cargo check -p katgpt-core --features spectral_threat` clean
  (this feature now activates `sense_threat` mod + the `linoss` dep).
- [ ] **T4.6** `cargo check --workspace --all-features` clean.
- [ ] **T4.7** `cargo check -p riir-engine` clean (verifies re-export shims
  preserve API). **Allow 7-10 min for this step.**
- [ ] **T4.8** `cargo check -p riir-neuron-db --all-features` clean.
- [ ] **T4.9** `cargo check -p riir-chain --all-features` clean.

### Phase 5 — Issue 007 closure

- [ ] **T5.1** Mark Phase E Tier 2 #7 as `[x]` in
  `riir-ai/.issues/007_*.md`.
- [ ] **T5.2** Update Tier 2 status summary (now 4/4 done).
- [ ] **T5.3** Update the cumulative Phase E LOC count.

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

## Open Questions to Resolve Before Phase 1

- [ ] Is `sense_composition` currently in katgpt-core's `default` feature list?
  (Determines whether katgpt-sense gets default-on status.)
- [ ] Does `sense/spectral_threat.rs` get consumed externally via the
  `katgpt_core::sense::spectral_threat` path? If yes, the re-export alias
  needs to surface it.
- [ ] Are there other Cargo features that imply `sense_composition`?
  (Mirrors how `committed_field_blend` implies `personality_composition`.)

---

## Cross-Repo Coordination

Single-repo change (katgpt-rs only). Sense has no `riir-ai` consumers per
grep (the NPC-runtime half moved in Phase C). Path deps unchanged.

Commit prefix: `feat(katgpt-sense)!: promote sense substrate` for Phase 3,
`feat(katgpt-types): co-extract ScaleBoundary + TemporalDerivativeKernel`
for Phase 1+2.
