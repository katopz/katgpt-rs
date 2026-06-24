# Issue 005: Stokes Calculus G-C — `line_integral` Cannot Encode Turn Penalties (needs rank-2 wrapper)

**Date:** 2026-06-24
**Status:** **CLOSED — RESOLVED** (Plan 317, 2026-06-24). `circulation_integral` rank-2 wrapper implemented + tested + benchmarked. G-C "≥20% fewer reversals" **still structural FAIL** (confirmed empirically: minimizing circulation INCREASES turns). Primitive is correct and useful; `stokes_calculus` stays opt-in pending G-A (riir-ai Plan 334).
**Origin:** Plan 314 Phase 3 GOAT gate (benchmark `.benchmarks/314_stokes_calculus_goat.md`)
**Severity:** Low (the primitive is correct and useful as a path-cost function; only the "smoothness" framing is wrong)
**Related:** katgpt-rs/.plans/314 (Stokes Calculus Wrappers), katgpt-rs/.plans/317 (Circulation Integral), katgpt-rs/.benchmarks/314_stokes_calculus_goat.md, katgpt-rs/.benchmarks/317_circulation_integral_goat.md

## Problem

The Plan 314 G-C gate target ("≥20% fewer direction reversals via `line_integral`-weighted path reranking") is **structurally unreachable** for `line_integral` on a rank-1 (edge) cochain.

**Root cause:** A rank-1 edge cochain assigns one scalar per edge. `line_integral` sums these scalars along a path. Turn penalties depend on the **angle between consecutive edges** — a path-level property that cannot be expressed as a sum of per-edge scalars. Mathematically:

- `line_integral(path) = Σ_e sign(e, path) · field[e]`
- Turn penalty would require `Σ_{(e_i, e_{i+1}) ∈ path} penalty(angle(e_i, e_{i+1}))` — a **pairwise** edge term.
- A rank-1 cochain has no way to encode pairwise edge interactions.

**Evidence from the G-C benchmark:**
- Smooth path (1 turn) and zigzag path (29 turns) between the same endpoints have DIFFERENT `line_integral` values (2.231 vs 0.359) — but this difference comes from **which edges** they traverse (spatially varying non-exact field), NOT from the turn count.
- On a uniform field, both paths give identical `line_integral` regardless of turn count.
- On an exact (gradient) field, both paths give identical `line_integral` by the fundamental theorem of calculus (path-independence).

## Why this is not a bug

`line_integral` is **correct** — it faithfully computes the discrete line integral of a rank-1 cochain. The 4 Phase-2 unit tests all pass (straight path, reversal antisymmetry, closed-loop-of-exact-field = 0, short path). The issue is that the Plan 314 G-C target was based on a **misclassification** of what rank-1 cochains can express. The plan's risk note ("G-C may fail if manifold_geodesic paths are already near-optimal") understated the problem — the failure is more fundamental than path optimality.

## `line_integral` is still useful

The primitive remains a valid **path-cost function** for:
- Path energy / work computation (Σ per-edge cost)
- Terrain-friction accumulation along a route
- Comparing the cost of two candidate paths (it correctly discriminates on non-exact fields)
- Composing with `manifold_geodesic` output as a post-hoc cost label

It just cannot serve as a **smoothness/reversal regularizer**.

## Proposed fix: rank-2 `circulation_integral` wrapper

The natural Stokes-theorem companion to `line_integral` (rank-1, ∫ over 1-paths) is a **rank-2 circulation integral** (∮ over closed loops, integrating curl over enclosed area). This CAN encode turn penalties because:

- A path with many turns encloses more area (in the sense of the signed area between the path and the straight-line shortcut) than a smooth path.
- The circulation `∫_loop field = ∫_area curl(field)` by Stokes' theorem.
- Turn penalties emerge naturally as the curl integrated over the "detour area."

**Sketch:**
```rust
/// Circulation of a rank-1 edge field around a closed vertex loop.
/// Equals the integral of curl(field) over the enclosed area (Stokes).
/// Non-zero for rotational fields; zero for exact (gradient) fields.
pub fn circulation_integral(cx: &CellComplex, edge_field: &CochainField, closed_loop: &[u32]) -> f32 {
    // = line_integral(cx, edge_field, closed_loop) since the loop is closed.
    // But the INTERPRETATION differs: this measures enclosed curl, not path energy.
    // For turn-smoothness: compare circulation_integral of a candidate path's
    // "closure" (path + straight-line return) — smooth paths enclose less area.
}
```

This is a ~20 LOC wrapper, same complexity class as the existing primitives. It composes with `line_integral` (a closed loop's line_integral IS its circulation).

## Tasks

- [x] Implement `circulation_integral(cx, edge_field, closed_loop) -> f32` in `stokes_calculus.rs` (~15 LOC, delegates to `line_integral` for the closed loop + debug_assert closed).
- [x] Add 3 unit tests: zero-curl field → zero circulation; constant-curl field → circulation = curl × area; reversal antisymmetry (clockwise vs counterclockwise). All 3 PASS.
- [x] Re-run G-C benchmark with `circulation_integral` as the smoothness metric. **RESULT: smooth loop circulation=128 (3 turns) vs zigzag circulation=112 (25 turns)** — minimizing circulation picks MORE turns. G-C FAILS empirically.
- [-] If G-C passes with `circulation_integral` → update Plan 314 G-C target to use the rank-2 wrapper; consider promoting `stokes_calculus` to default-on. **NOT APPLICABLE — G-C fails.** `stokes_calculus` stays opt-in.

## Resolution (Plan 317, 2026-06-24)

**What was done:**
1. Implemented `circulation_integral` as a thin, Stokes-theorem-correct wrapper over `line_integral` (a closed loop's line integral IS its circulation). Debug-asserts the loop is closed.
2. Added 3 unit tests verifying the Stokes identities (zero-curl→0, constant-curl→curl×area with cross-check against `exterior_derivative`, reversal antisymmetry). All PASS.
3. Ran the G-C2 benchmark on a 32×32 grid with a constant-curl field. **Empirical result**: smooth 8×8 rectangle loop has circulation=128 (3 turns) while zigzag sawtooth loop has circulation=112 (25 turns). Minimizing circulation picks the zigzag (MORE turns).

**Why G-C still fails (the honest finding):**

The pre-implementation analysis in Plan 317 predicted that turn count and enclosed area are **independent geometric properties**. The empirical result confirms this:
- A smooth rectangle (4 turns) MAXIMIZES enclosed area for a given bounding box → HIGHER circulation.
- A zigzag (many turns) can cut corners and enclose LESS area → LOWER circulation.
- So minimizing `|circulation_integral|` picks the zigzag — the OPPOSITE of "fewer reversals".

This is not a bug in `circulation_integral` — it's a mathematical fact about what Stokes integrals can express. Turn count is combinatorial; Stokes integrals are geometric. The G-C framing ("fewer reversals via Stokes reranking") was based on a false intuition from Issue 005's original proposal.

**The primitive is still valuable:**
- Stokes-theorem-correct (3 unit tests confirm the identities).
- Natural rank-2 companion to `line_integral`.
- Valid applications: rotational/vortex detection, Stokes-correct area measurement, harmonic field identification.

**Promotion decision:** `stokes_calculus` stays opt-in. G-B (5.36× boundary-flux speedup) is the only gate that passed. G-A already FAILED in riir-ai Plan 334 (9.5× slower, 36% lower F1 than JS-divergence — fixed-grid cost cannot compete at action_dim=8). G-C fails structurally (confirmed empirically with `circulation_integral`). All three GOAT gates now have verdicts: G-A FAIL, G-B PASS, G-C FAIL. The 4 primitives are all correct and available to callers who enable the feature.

### Verification

| Check | Result |
|-------|--------|
| `cargo test -p katgpt-core --features dec_operators --lib dec::stokes_calculus` | **15 passed** (12 existing + 3 new), 0 failed |
| `cargo test -p katgpt-core --features dec_operators --lib dec::` (G3 regression) | **99 passed**, 0 failed |
| `cargo test -p katgpt-core --lib` (full G3) | **509 passed**, 0 failed |
| `cargo check --all-features` | **EXIT 0** (Issue 004 fix holds) |
| G-C2 benchmark (smooth vs zigzag circulation) | smooth=128/3turns vs zigzag=112/25turns → **G-C FAILS** (minimizing circulation increases turns) |

## Verdict

**Keep `stokes_calculus` opt-in** until either:
1. `circulation_integral` is added and G-C passes with it, OR
2. G-A (Fokker-Planck validator, deferred to riir-ai) passes and becomes the headline application.

The three existing primitives (`belief_mass_divergence`, `boundary_flux_mass`, `line_integral`) are all correct and available to callers who want them. The opt-in status reflects that the GOAT gate didn't fully clear, not that the code is broken.
