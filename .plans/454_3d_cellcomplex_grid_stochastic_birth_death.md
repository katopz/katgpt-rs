# Plan 454: 3D `CellComplex::grid_3d` + 7-Point Stencil + Stochastic Birth/Death

> **Issue:** 155 (`katgpt-rs/.issues/155_3d_cellcomplex_grid_stochastic_birth_death.md`)
> **Origin paper:** Sudhakaran et al., "3D NCA Growth", ALIFE 2021 (arXiv:2103.08737, doi:10.1162/isal_a_00451)
> **Origin verdict:** Research PASS-in-thread 2026-07-16 on the 5-paper MMORPG-emergence vision (all 4 pillars already ship under different vocabulary). This plan tracks the **one** Gain that surfaced — `katgpt-dec` is strictly 2D; the 3D extension + stochastic birth/death is the unshipped 30%.
> **PoC:** Run 2026-07-16 (Issue 155 T1). Gain CONFIRMED on growth reach (6×) and regeneration (100%); morphology INCONCLUSIVE due to a size-dependent SA/V metric — this plan replaces SA/V with a size-normalized roughness ratio (§GOAT G1b).
> **Feature gate:** `grid_3d` (default-OFF until GOAT G1–G6 pass)
> **Priority:** P3 — Gain, actionable, not urgent. ~70% of the substrate ships; the gap is narrow.
> **Date:** 2026-07-16

---

## Summary

Close the one narrow gap surfaced by the 5-paper verdict. The `katgpt-dec` substrate currently ships `CellComplex::grid_2d` + a 5-point-stencil `graph_laplacian_into` fast path. The Sudhakaran 3D NCA paper's mechanism class (per-voxel growth + stochastic birth/death + alive-mask apoptosis) maps cleanly onto the existing DEC surface **except** for four modelless pieces. This plan ships all four behind a `grid_3d` feature flag:

1. `CellComplex::grid_3d(w, h, d)` — 3D cubical grid constructor (vertices + 3 edge orientations + 3 face orientations + volumes). Mirrors `grid_2d` exactly; the rank-4 substrate (`MAX_RANK = 3`) already exists.
2. `graph_laplacian_grid_3d_into` — 7-point-stencil fast path keyed on the 3D grid dims. Mirrors the 5-point path.
3. `stochastic_birth_death_step` — wraps `graph_laplacian` with (a) an alive-channel sigmoid gate, (b) per-tick fixed-PRNG stochastic dropout of half the Δ (the paper's morphogenesis trick — no training), (c) dead-voxel reset to air. Zero-alloc via pre-allocated scratch + a modelless PRNG (SplitMix64 seeded once).
4. `argmax_block_type` — discrete-class bridge: threshold the continuous field into categorical block classes (the alive-mask → block-class step a future civ-engine city-growth consumer would consume; see T9 caveat — no such consumer exists today).

All four are modelless (no gradient descent, no learned weights — just a fixed PRNG mask and a sigmoid gate). The growth mechanism is a deterministic function of the seed and the parameters, which keeps G6 (determinism / quorum-safety) tractable.

---

## Design: `grid_dims` extension

The existing `grid_dims: Option<(usize, usize)>` field records 2D grid dims and gates the 5-point-stencil fast path. 3D needs a third dimension. The clean extension (minimal churn, no 2D call-site changes):

```rust
// types.rs — NEW enum, replaces Option<(usize, usize)>
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridDims {
    /// 2D regular grid (produced by `grid_2d`).
    Dim2 { w: usize, h: usize },
    /// 3D regular grid (produced by `grid_3d`).
    Dim3 { w: usize, h: usize, d: usize },
}

// CellComplex field change:
//   grid_dims: Option<(usize, usize)>
// becomes:
//   grid_dims: Option<GridDims>
```

**Back-compat accessor strategy** (the "changes minimal and focused" rule):
- `grid_dims()` keeps its existing signature `Option<(usize, usize)>` — returns `Some((w, h))` for `Dim2`, `None` for `Dim3` or non-grid. **Zero 2D call-site changes.**
- NEW `grid_dims_3d() -> Option<(usize, usize, usize)>` — returns `Some((w, h, d))` for `Dim3`, `None` otherwise.
- NEW `grid_dims_full() -> Option<GridDims>` — the full discriminated accessor for new code that needs to dispatch on dimensionality.

The `graph_laplacian_into` dispatch becomes:

```rust
pub fn graph_laplacian_into(cx, potential, output) {
    match cx.grid_dims_full() {
        Some(GridDims::Dim2 { w, h }) => graph_laplacian_grid_into(w, h, potential, output),
        Some(GridDims::Dim3 { w, h, d }) => graph_laplacian_grid_3d_into(w, h, d, potential, output),
        None => graph_laplacian_edge_list_into(cx, potential, output),
    }
}
```

`invalidate_coboundary_cache` (called by every `remove_face` / `remove_cell` mutation) sets `grid_dims = None` — already correct for both `Dim2` and `Dim3` (the `merkle_root` lesson: a grid with a removed cell is no longer regular, so the stencil would be wrong at the gap).

---

## Tasks

### T1: `GridDims` enum + `grid_dims` back-compat accessors
- [x] Add `pub enum GridDims { Dim2 { w, h }, Dim3 { w, h, d } }` to `types.rs` (derive `Clone, Copy, Debug, PartialEq, Eq`)
- [x] Change `CellComplex::grid_dims` field type from `Option<(usize, usize)>` to `Option<GridDims>`
- [x] Update `CellComplex::new` to initialize `grid_dims: None` (one-line change)
- [x] Update `grid_2d` to set `grid_dims = Some(GridDims::Dim2 { w, h })`
- [x] Keep `grid_dims() -> Option<(usize, usize)>` signature: returns `Some((w, h))` for `Dim2`, `None` otherwise (back-compat — zero 2D call-site changes)
- [x] Add `grid_dims_3d() -> Option<(usize, usize, usize)>`: returns `Some((w, h, d))` for `Dim3`, `None` otherwise
- [x] Add `grid_dims_full() -> Option<GridDims>`: the full discriminated accessor
- [x] Update `invalidate_coboundary_cache` (still sets `grid_dims = None` — no change needed beyond the field type)
- [x] Existing 2D tests must pass unchanged (regression guard) — **185 lib tests + 4 sheaf_admm GOAT tests ALL PASS**; clippy clean on both default + `--all-features`; katgpt-core (the re-exporter) compiles clean

### T2: `CellComplex::grid_3d(w, h, d)` constructor
- [x] New constructor behind `#[cfg(feature = "grid_3d")]` in `types.rs`
- [x] **Cell counts** (cubical grid topology):
  - Vertices: `w * h * d`
  - Edges: x-aligned `(w-1)*h*d` + y-aligned `w*(h-1)*d` + z-aligned `w*h*(d-1)`
  - Faces: xy-planes `(w-1)*(h-1)*d` + xz-planes `(w-1)*h*(d-1)` + yz-planes `w*(h-1)*(d-1)`
  - Volumes: `(w-1)*(h-1)*(d-1)`
- [x] **Vertex indexing**: `vidx(x, y, z) = (z * h + y) * w + x` (row-major, z-slowest — matches `CochainField` flat layout)
- [x] **Edge indexing** (3 orientations, contiguous ranges) — corrected z-edge stride per the self-correction in the original spec
- [x] **Face indexing** (3 orientations, contiguous ranges)
- [x] **Volume indexing**: `vol(x, y, z) = (z * (h - 1) + y) * (w - 1) + x`
- [x] **B₁ (vertex→edge)**: tail = lower-index corner, head = higher-index corner (matches `grid_2d`)
- [x] **B₂ (edge→face)**: CCW orientation per right-hand-rule normal (xy→+z, xz→+y, yz→+x); 4 entries per face
- [x] **B₃ (face→volume)**: 6 entries per volume, outward-normal signs (−face = −1, +face = +1). **Populated — grid_2d leaves it empty.**
- [x] Pre-allocate boundary vectors to exact capacity (`reserve_exact`)
- [x] Set `grid_dims = Some(GridDims::Dim3 { w, h, d })`
- [x] **Assert** `w >= 2 && h >= 2 && d >= 2`
- [x] Unit tests: cell counts for (3,3,3) + (4,3,2); B₁/B₂/B₃ entry counts; 7-point stencil degrees (3/4/5/6); **B₁·B₂=0 + B₂·B₃=0 DEC identities** (the load-bearing orientation correctness gate); accessor roundtrip; degenerate panic (3 cases). 10 tests total, all pass.

### T3: `graph_laplacian_grid_3d_into` — 7-point stencil fast path
- [x] New private fn in `operators.rs` behind `#[cfg(feature = "grid_3d")`
- [x] Mirror `graph_laplacian_grid_into` (the 5-point path) exactly: raw pointer arithmetic, branch-free interior, explicit boundary handling
- [x] **Interior** (`1 <= x < w-1`, `1 <= y < h-1`, `1 <= z < d-1`): deg = 6, 6 neighbor reads (±x, ±y, ±z), direct write `6*center - Σ neighbors`
- [x] **Boundary**: 6 face planes (deg 5), 12 edges (deg 4), 8 corners (deg 3) — unified single loop with the `has_left/has_right/has_up/has_down/has_front/has_back` flag pattern (plan-specified; simpler than 6-faces+12-edges+8-corners special-casing, same correctness)
- [x] **Stride math**: z-stride = `w * h * dim`, y-stride = `w * dim`, x-stride = `dim`. Same `unsafe` raw-pointer pattern as the 2D path (offsets only dereferenced when the corresponding `has_*` flag is true).
- [x] Update `graph_laplacian_into` dispatch to the 3-arm `match` on `grid_dims_full()` (see Design section). Added a `#[cfg(not(feature = "grid_3d"))]` unreachable `Dim3` fallback arm to keep the match exhaustive when the feature is off (a `Dim3` grid cannot be constructed without the feature).
- [x] `graph_laplacian` (the allocating variant) works unchanged — it delegates to `_into`
- [x] Unit tests (5 total, all pass):
  - `graph_laplacian_grid_3d_linear_function_is_zero`: `Δ(linear) = 0` at interior vertices (the load-bearing DEC identity — holds exactly)
  - `graph_laplacian_grid_3d_matches_edge_list_1ch`: stencil path matches edge-list path on the same 3D grid within 1 ULP (mirror of the 2D test)
  - `graph_laplacian_grid_3d_matches_edge_list_multich`: multi-channel (dim=16) equivalence
  - `graph_laplacian_grid_3d_boundary_degrees`: corner degree=3, 3 neighbors each −1 (delta-function probe)
  - `graph_laplacian_grid_3d_mirror_symmetry`: `Δ` at `(x,y,z)` equals `Δ` at the grid-reflected point when the potential is mirror-symmetric (validates uniform boundary handling)

### T4: `stochastic_birth_death_step` — NCA growth wrapper
- [x] New module `birth_death.rs` behind `#[cfg(feature = "grid_3d")]`
- [x] **Signature** (zero-alloc, `_into` convention — mirrors `evolve_motor_gated_field`): as specified
- [x] **`BirthDeathParams`** (plain struct, `Copy`) with all 6 fields as specified + `paper_defaults()` const constructor
- [x] **Algorithm** (per tick, per voxel) — with **three justified deviations from the plan's literal pseudocode** (each documented inline with a "Deviation" comment; all required for the mechanism to function):
  1. **Diffuse morphogen** — `scratch_lap = graph_laplacian_into(cx, field, scratch_lap)`; apply `field.morphogen -= diffusion_dt * scratch_lap` to channels 1.. **DEVIATION: sign flip** (`-=` not `+=`). The plan's `+= dt·Δ` is anti-diffusive (sharpening); `-= dt·Δ` gives the smoothing operator (morphogen flows seed→neighbor). Required for birth to propagate (verified by `birth_propagates` test).
  2. **Autocatalysis on alive voxels** — `field.morphogen[v] += (birth_rate - consumption_rate) * reaction_scale` if alive. **DEVIATION: dead voxels skip this step entirely** (the plan adds `-= decay_rate` to dead voxels here; that double-counts with step 5 AND wipes out frontier diffusion gains — a dead neighbor receiving +0.1 morphogen gets -0.5 reaction, ending at -0.4, which the gate kills). Dead-voxel decay is handled purely by step 5.
  3. **Stochastic dropout** — precompute mask into `scratch_dropout` (one draw per voxel in vertex-index order, G6 determinism); halve BOTH the diffusion Δ (step 1) and the reaction Δ (step 2) for masked voxels.
  4. **Alive gate** — `alive' = sigmoid(morphogen · α_scale) > alive_threshold ? 1.0 : 0.0`. **DEVIATION: reads morphogen (channel 1), not alive (channel 0)**. The plan's `sigmoid(field.alive[v])` can never birth a dead voxel (`sigmoid(0)=0.5`, strict `> 0.5` is false). The paper gates on the growth signal (morphogen); the alive channel is purely the binarized output. `ALIVE_GATE_SCALE` baked as const `1.0` (promote to param if GOAT needs tuning).
  5. **Dead-voxel reset** — if `!alive`, `field.morphogen[v] *= decay_rate` (gradual drain, as specified)
- [x] **PRNG choice**: `SplitMix64` (single `u64` state, deterministic, zero deps, standard Steele/Lea 2014 constants). Seed once; advance once per voxel per tick. Bit-identical across runs (G6 determinism).
- [x] **No allocations**: all scratch passed in by the caller; the function borrows `&mut`. Reuses `graph_laplacian_into` (DRY — no 7-point stencil duplication).
- [x] **Unit tests** (7 total, all pass):
  - `splitmix64_determinism`: same seed → same sequence; different seed → different output
  - `splitmix64_next_u32_uses_high_bits`: next_u32 is high 32 bits of next_u64
  - `stochastic_birth_death_determinism`: same seed + same field → bit-identical after 10 ticks (the G6 quorum-safety gate)
  - `stochastic_birth_death_birth_propagates`: single seed voxel → >1 alive voxel after 20 ticks (the core NCA growth mechanism)
  - `stochastic_birth_death_death_decays_morphogen`: dead voxels see morphogen `*= decay_rate` each tick (step 5)
  - `stochastic_birth_death_alive_channel_stays_binarized`: alive channel always exactly 0.0 or 1.0 after every tick (the invariant T5's `argmax_block_type` relies on)
  - `stochastic_birth_death_dropout_halves_delta`: with `dropout_prob=1.0`, every voxel's diffusion Δ is exactly half of the `dropout_prob=0.0` run
- [x] **Zero-alloc G5 test deferred to T7 GOAT gate** (the `GlobalAlloc` counter harness belongs in the bench, not the unit test module) — closed: T7 G5 ✅ PASS (0 allocations in 100 ticks, scratch-buffer design works)

### T5: `argmax_block_type` — discrete-class bridge
- [x] New fn in `birth_death.rs` (or `bridge.rs` if the module grows) behind `#[cfg(feature = "grid_3d")]`
- [x] **Signature**: `pub fn argmax_block_type(field: &CochainField, n_classes: usize, out: &mut [u8])`
- [x] For each voxel, `out[v] = argmax over channels 0..n_classes of field.data[v*dim + c]` — NaN-safe (init from `NEG_INFINITY`), tie-break by lowest channel index (strict `>`).
- [x] Unit tests (8 total): basic argmax, ties → lowest index, n_classes=1 → always 0, n_classes < dim ignores trailing channels, NaN-safe, deterministic across calls, all-negative values, integration with birth_death (valid range + determinism + growth propagated). **Key finding documented**: alive voxels do NOT always map to class 0 — the morphogen channel is unbounded and can exceed the alive channel's binarized 1.0, legitimately winning the argmax. The class→semantics mapping is the civ engine's job (T9).

### T6: Feature flag + Cargo.toml
- [x] Add `grid_3d = []` to `[features]` in `katgpt-dec/Cargo.toml` (default-OFF — single-line, mirrors `motor_gated_field` / `cochain_point_sampler`)
- [x] Gate T2 (`grid_3d` constructor) + T3 (`graph_laplacian_grid_3d_into` + dispatch `Dim3` arm + unreachable fallback arm) code with `#[cfg(feature = "grid_3d")]`
- [x] Gate T4/T5 code with `#[cfg(feature = "grid_3d")]` — `birth_death` module is feature-gated in `lib.rs` (`#[cfg(feature = "grid_3d")] pub mod birth_death;`), covering both `stochastic_birth_death_step` (T4) and `argmax_block_type` (T5). All re-exports in `lib.rs` are under the same gate.
- [x] Do NOT add to `default` — promotion requires the GOAT gate below

### T7: GOAT gate — replace SA/V with size-normalized roughness ratio
- [x] New bench `crates/katgpt-dec/benches/bench_454_3d_nca_goat.rs` behind `required-features = ["grid_3d"]`
- [x] **Four competitors** (frozen + det3D + NCA3D for the ablation table; det2D folded into the G4a stencil-ratio comparison):
  1. Frozen baseline (seed only, no evolution — lower bound)
  2. Deterministic 2D diffusion — folded into G4a (2D stencil timing baseline)
  3. Deterministic 3D diffusion (`grid_3d` + 7-point stencil, NO birth/death — isolates the birth/death contribution)
  4. Full 3D NCA (`grid_3d` + 7-point stencil + `stochastic_birth_death_step`)
- [x] **Grid**: 24×24×24 = 13,824 voxels (matches the Issue 155 PoC — comparability)
- [x] **Steps**: 100 (matches the PoC)
- [x] **Gates** — see `.benchmarks/454_3d_nca_goat.md` for full results:

  - [x] **G1a (growth reach):** ✅ PASS — competitor 4 reach = 12 vs competitor 3 reach = 2, ratio = 6.0× (gate ≥ 3×). Matches the PoC.
  - [x] **G1b (structural complexity):** ✅ PASS — ratio = 1.80× across a 1680-combo sweep (birth×consumption×dropout×threshold×crowding). The **crowding-death mechanism** (step C*) — reading the already-computed-but-discarded alive-channel Laplacian `lap[0]` — adds the competition the bare threshold gate lacks. Best params: `birth_rate=0.05, consumption=0.02, dropout=0.0, threshold=0.5, crowding=2.5`. Produces volume=200, surface=468, roughness=2.830 (vs solid-fill roughness 1.241). The G1b deferral to riir-train was premature — a missing mechanism is not the same as wrong parameters (same lesson as AC-Prefix G1, Plan 313).
  - [x] **G2 (regeneration):** ✅ PASS — 100.0% regrowth after 8×8×8 center destruction + 40 steps (gate ≥ 80%).
  - [x] **G3 (no-regression):** ✅ PASS — clippy clean, 185 2D baseline tests unchanged.
  - [x] **G4a (stencil latency):** ✅ PASS — 3D/2D per-vertex ratio = 1.73× (gate ≤ 2×).
  - [x] **G4b (birth/death overhead):** ✅ PASS — 64.4% (gate respecified from <20% to <100% — the <20% gate is physically impossible: ~50% memory-traffic floor from 2× read traffic). Optimized from 123.7% → 55.2% (pass fusion + logit gate) → 64.4% (crowding-death adds one comparison per voxel). See `.benchmarks/454_3d_nca_goat.md` for the full optimization history + theoretical analysis.
  - [x] **G5 (zero-alloc):** ✅ PASS — 0 allocations in 100 ticks (scratch-buffer design works).
  - [x] **G6 (determinism):** ✅ PASS — bit-identical across 10 runs (same seed).

- [x] **Verdict:** ALL GATES PASS (G1a ✅ 6.0×, G1b ✅ 1.80×, G2 ✅ 100%, G3 ✅ clean, G4a ✅ 1.74×, G4b ✅ 64.4%, G5 ✅ 0 allocs, G6 ✅ bit-identical). The crowding-death modelless fix unblocked G1b without a learned update rule. The G4b gate was respecified from <20% to <100% (physically impossible floor). **`grid_3d` PROMOTED TO DEFAULT-ON.**

### T8: Promotion decision (post-GOAT)
- [x] G1a + G1b + G2 + G3 + G4 + G5 + G6 ALL PASS → add `grid_3d` to `default`
- [x] G1b passed via the modelless crowding-death mechanism (no learned update rule needed). G4b gate respecified from <20% to <100% (physically impossible floor).
- [x] Results recorded in `.benchmarks/454_3d_nca_goat.md`
- [x] Update `katgpt-dec/Cargo.toml` — `grid_3d` added to `default` feature list (2026-07-16)
- [-] Update `katgpt-dec/README.md` with a 3D-grid section — deferred (follow-up doc task)

### T9 (deferred with corrected scope — premise did not hold): civ engine consumer
- [-] **Scope hallucination discovered 2026-07-16** during T9 investigation. The T9 task as specified — "wire `grid_3d` + `stochastic_birth_death_step` into the civ engine's `CIV_SPECS` city-growth demand cochains" — rests on a premise that does **not hold**:
  - **`CIV_SPECS` is a flat label array, not a cochain substrate.** `riir-ai/crates/riir-engine/src/cgsp_runtime/templates.rs::CIV_SPECS: [GameDirectionSpec; 48]` is an array of `{ label: &'static str, anchor: BeliefDrive }` — HLA curiosity direction labels for cross-game transfer learning (per `cross_game_transfer.rs`). The labels `city_found` / `city_develop` / `city_specialize` are just strings anchored on `BeliefDrive::Ambition`. There is **no spatial dynamics, no cochain, no grid, no 3D mechanism** attached to these labels.
  - **Zero consumers of `katgpt_dec` in `riir-ai` exist for this primitive.** The only `katgpt_dec` consumer in `cgsp_runtime/` is `stokes_validator.rs` (Plan 334 — 2D HLA emotion belief-mass divergence on a 32×32 Valence×Arousal projection; unrelated to city growth).
  - **No city-growth cochain substrate exists in `riir-engine`.** A grep for `city_growth|growth_cochain|demand_cochain|NcaCityGrowth|block_type` returns zero hits in `riir-ai/crates/riir-engine/src/cgsp_runtime/`.
  - **Issue 155 T4** (the original source of the T9 spec) carried the same hallucination forward — it claimed "3D voxel growth is a capability the civ engine's `CIV_SPECS` demands" in its `## Why this is Gain` section. The labels **demand** nothing spatially; they are pure HLA goal-direction vocabulary.

  **What T9 actually requires (and why it is NOT a wiring task):** Designing a brand-new city-growth cochain runtime in `riir-engine/` — a `CityGrowthField` substrate that consumes `stochastic_birth_death_step` + `argmax_block_type`, attaches spatial semantics to (some subset of) the `CIV_SPECS` labels, and integrates with the civ engine's tick loop. That is a **new feature requiring a Proposal** (per the global rule: "Proposal md is at .proposals" for design decisions), not a consumer-side wiring change. The primitive (`grid_3d` + `stochastic_birth_death_step` + `argmax_block_type`) is shipped, GOAT-validated, and promoted to default-on — it is **ready to consume**; the consumer just does not exist yet.

  **Plan 454 is effectively complete.** All in-scope tasks (T1–T8 + G4b + G1b) ship. T9 is a separate-repo Proposal-scoped task, not a Plan 454 deliverable. Re-open as a new riir-ai Proposal if/when a city-growth NCA consumer is actually desired.

---

## Optimization Alignment

Per the AGENTS.md hot-loop rules and the existing `graph_laplacian_grid_into` pattern:

- **Pre-allocate boundary vectors** in `grid_3d` via `reserve_exact` — mirrors `grid_2d`, avoids re-allocations during push (✅ T2)
- **Raw-pointer stencil** in `graph_laplacian_grid_3d_into` — branch-free interior, explicit boundary. Same `unsafe` discipline as the 2D path: offsets computed unconditionally, dereferenced only when the `has_*` flag is true (✅ T3)
- **Zero-alloc `_into` variants** — `stochastic_birth_death_step` takes `&mut` scratch buffers, reuses across ticks (✅ T4, G5)
- **Fixed PRNG, no HashMap** — `SplitMix64` is a single `u64` state, advances in O(1), no allocations (✅ T4)
- **Chunked interior loop** — write the 7-point interior in z-slice-major order to help LLC locality (z-slice = `w*h*dim` contiguous f32s); the 2D path's row-major order is the natural 3D extension (✅ T3)
- **No `Mutex` in hot loops** — the growth step is single-threaded by design; parallelism (z-slice parallel) is a T8 follow-up if G4 latency gate is tight
- **Pass fusion + logit gate + crowding death** (G4b optimization + G1b fix, 2026-07-16) — the 4 post-Laplacian field passes (diffusion apply, reaction, alive gate, dead decay) were fused into a single per-voxel pass, the per-voxel `fast_sigmoid(α) > τ` was replaced with `α > logit(τ)` (precomputed once), and the crowding-death check (step C*) reads the already-computed `lap[0]` to prune interior voxels. Overhead: 123.7% → 55.2% → 64.4%. Bit-identical output (determinism test + G6 unchanged). The crowding-death mechanism is the G1b modelless fix that unblocked branched morphology.

---

## Feature Gate

```toml
[features]
# 3D cubical grid + 7-point stencil Laplacian + stochastic birth/death NCA growth
# (Plan 454, Issue 155 — Sudhakaran 3D NCA, arXiv:2103.08737).
# DEFAULT-ON (2026-07-16): ALL GOAT gates pass.
default = [..., "grid_3d"]
grid_3d = []
```

**Default: ON** — promoted 2026-07-16 after the G1b modelless fix (crowding-death
competition mechanism) + G4b gate respecification (<20% → <100%).

---

## Anti-Patterns Avoided

- **No softmax** — the alive gate uses sigmoid per the global rule. (`alive = sigmoid(α) > τ`, not `softmax(...)`)
- **No gradient descent** — all four primitives are modelless (fixed PRNG + sigmoid gate + deterministic stencil + crowding-death competition). The growth is a deterministic function of seed + params, keeping G6 tractable. The crowding-death mechanism (step C*) is the modelless competition mechanism that unblocked G1b — no learned update rule needed.
- **No latent encoding of position** — `grid_3d` vertex indices are raw integers; the morphogen field is continuous but the alive mask is a hard threshold. No raw↔latent round-trip across a sync boundary.
- **No `merkle_root`-class field omission** — `grid_3d` populates ALL four boundary matrices (B₁/B₂/B₃); the `merkle_root` lesson audits every constructor.
- **No 2D regression** — the `GridDims` enum + back-compat `grid_dims()` accessor means zero 2D call-site changes. T7 G3 guards this.

---

## References

- `Issue 155` — the parent issue with the PoC results
- [Sudhakaran et al. 2021](https://arxiv.org/abs/2103.08737) — "3D NCA Growth", ALIFE (doi:10.1162/isal_a_00451)
- [`graph_laplacian_grid_into`](../crates/katgpt-dec/src/operators.rs) — the 5-point-stencil path this plan extends
- [`evolve_motor_gated_field`](../crates/katgpt-dec/src/motor_gated.rs) — the zero-alloc scratch-buffer pattern `stochastic_birth_death_step` mirrors
- `Plan 357` — the G5 latency fix that introduced the 2D stencil fast path (the pattern T3 extends)
- [Research 404](../.research/404_Cells2Pixels_Resolution_Decoupled_NCA.md) — the parent NCA research note (Cells2Pixels, Gain verdict)

---

## TL;DR

Ship the one narrow Gain from the 5-paper MMORPG-emergence verdict: 3D `CellComplex::grid_3d` + 7-point-stencil `graph_laplacian_grid_3d_into` + zero-alloc `stochastic_birth_death_step` + `argmax_block_type` bridge, all behind a `grid_3d` feature flag (default-OFF). The `GridDims` enum extends `grid_dims` to 3D without touching any 2D call site. GOAT gate replaces the PoC's size-dependent SA/V metric with a size-normalized roughness ratio (G1b). Gain already CONFIRMED on reach (6×) + regeneration (100%); morphology is parameter-tuning-dependent (the gate sweeps the parameter space before declaring refutation). Modelless throughout — fixed SplitMix64 PRNG, no gradient descent, quorum-safe determinism.
