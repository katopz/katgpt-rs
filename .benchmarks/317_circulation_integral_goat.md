# Benchmark 317: Circulation Integral — Rank-2 Stokes Wrapper (Issue 005)

**Date:** 2026-06-24
**Plan:** 317 (Circulation Integral — Rank-2 Stokes Wrapper)
**Issue:** 005 (Stokes Calculus G-C — `line_integral` Cannot Encode Turn Penalties)
**Features:** `--features dec_operators` (root alias: `stokes_calculus`)
**Commands:**
  - Tests: `cargo test -p katgpt-core --features dec_operators --lib dec::stokes_calculus`
  - Bench:  `cargo bench -p katgpt-core --features dec_operators --bench stokes_calculus_bench -- --warm-up-time 1 --measurement-time 2 --sample-size 10 circulation_integral`

---

## Summary Verdict

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| **G-C2** (circulation_integral turn reduction) | ≥20% fewer reversals via `circulation_integral`-weighted reranking | `circulation_integral` correctly measures enclosed curl (Stokes holds), but **minimizing circulation INCREASES turns** in the test case | ⚠️ **STRUCTURAL FAIL (confirmed empirically)** — see §G-C2 Honest Finding |
| Primitive correctness (Stokes identities) | Zero-curl→0, constant-curl→curl×area, reversal antisymmetry | All 3 unit tests PASS | ✅ **PASS** |

**Promotion decision:** `stokes_calculus` **stays opt-in**. G-B's 5.36× win (from Plan 314) is a genuine standalone gain, but G-A FAILED in riir-ai Plan 334 (9.5× slower, 36% lower F1 — fixed-grid cost cannot compete at action_dim=8), G-C fails (even with `circulation_integral`, confirmed empirically), and `dec_operators` itself is opt-in. The 4 primitives are all correct and available to callers who want them. All three GOAT gates now have verdicts: G-A FAIL, G-B PASS, G-C FAIL.

---

## The New Primitive: `circulation_integral`

```rust
pub fn circulation_integral(cx: &CellComplex, edge_field: &CochainField, closed_loop: &[u32]) -> f32
```

Thin wrapper over `line_integral` (a closed loop's line integral IS its circulation). Debug-asserts the loop is closed (first == last vertex). Returns 0.0 for loops shorter than 3 vertices.

**Mathematical content** (Generalized Stokes' Theorem):
```
∮_loop field = ∬_enclosed_area curl(field) dA = Σ_faces d₁(field)[f]
```

For a constant-curl field (curl = c everywhere), circulation = c × (signed enclosed area). This makes `circulation_integral` a pure area-measuring instrument on such fields.

---

## Unit Tests (Phase 2, all PASS)

All 15 `dec::stokes_calculus` tests pass (12 existing + 3 new):

| Primitive | Tests | Status |
|-----------|-------|--------|
| `belief_mass_divergence` | 4 | ✅ (pre-existing) |
| `boundary_flux_mass` | 4 | ✅ (pre-existing) |
| `line_integral` | 4 | ✅ (pre-existing) |
| `circulation_integral` | 3 (zero-curl→0, constant-curl→curl×area, reversal antisymmetry) | ✅ **NEW** |

### T3.2 detail — Stokes identity cross-check

The constant-curl test constructs a field with curl = +1 on face 0 (only the bottom edge non-zero), then verifies:
- `circulation_integral` around face 0's boundary == `d₁(field)[face 0]` (Stokes identity).
- Both equal 1.0 (curl × unit area).

This cross-checks `circulation_integral` against the shipped `exterior_derivative` operator — confirming the wrapper is Stokes-theorem-correct by construction.

---

## G-C2 Honest Finding: circulation_integral CANNOT Reduce Turns

### The empirical result

On a 32×32 grid with a constant-curl field (rigid rotation, curl = 2):

| Loop | Circulation | Enclosed Area | Turns |
|------|-------------|---------------|-------|
| **Smooth** (8×8 rectangle boundary) | **128.000** | 64 (= 8²) | 3 (4 corners; `count_turns` doesn't wrap) |
| **Zigzag** (sawtooth bottom + smooth top/right/left) | **112.000** | 56 (= 64 − 8 teeth × 1 unit²) | 25 |

**The smooth loop has HIGHER circulation (128 > 112) despite FEWER turns (3 < 25).**

### Why this happens (and why it's not a bug)

`circulation_integral` measures **enclosed curl** (Stokes' theorem). On a constant-curl field, this is proportional to **enclosed area**. The smooth rectangle encloses the FULL 8×8 = 64 area. The zigzag's sawtooth bottom cuts INTO the square, removing 8 unit triangles, so it encloses only 56.

Turn count and enclosed area are **independent geometric properties**:
- A smooth rectangle (4 turns) maximizes enclosed area for a given bounding box.
- A zigzag (many turns) can enclose LESS area by cutting corners.

**Therefore**: minimizing `|circulation_integral|` picks the **zigzag** (less area, MORE turns) — the OPPOSITE of the G-C target "≥20% fewer reversals".

This empirically confirms the pre-implementation analysis in Plan 317: `circulation_integral` is a correct Stokes primitive, but the G-C framing ("fewer reversals via circulation reranking") is based on a false intuition that "smooth = less enclosed area". In reality, smooth loops often enclose MORE area than zigzags within the same bounding box.

### The primitive is still correct and useful

The 3 unit tests confirm Stokes' theorem holds exactly:
- Zero-curl field → zero circulation (FTC). ✓
- Constant-curl field → circulation = curl × area. ✓
- Reversal antisymmetry: clockwise = −counterclockwise. ✓

`circulation_integral` is the natural rank-2 Stokes companion to `line_integral`. Its valid applications:
- **Rotational content detection**: non-zero circulation on a closed loop reveals rotational (non-exact) field structure.
- **Vortex/swirl detection**: high circulation around a loop indicates a vortex center nearby.
- **Stokes-theorem-correct area measurement** on constant-curl fields.
- **Harmonic field identification**: harmonic fields have zero circulation AND zero divergence.

It just cannot serve as a **turn-count reducer** — that requires a combinatorial path-level operator, not a Stokes-theorem integral.

---

## Perf: circulation_integral latency

| Bench | Median | Notes |
|-------|--------|-------|
| `G-C2_smooth_closed_loop_8x8` (32 vertices) | **12.10 µs** | 378 ns/vertex; delegates to `line_integral` |
| `G-C2_zigzag_closed_loop` (52 vertices) | **20.30 µs** | 390 ns/vertex; same O(loop_len × \|B₁\|) as `line_integral` |

Per-vertex cost matches `line_integral`'s ~310 ns/edge from Plan 314's G-C bench — the wrapper overhead is a single function call + debug_assert (negligible). The latency is dominated by `line_integral`'s O(\|B₁\|) edge-lookup scan per step (same future optimization: CSR vertex-pair→edge index, filed as a follow-up to Plan 312).

---

## Promotion Decision

**`stokes_calculus` stays OPT-IN** (not promoted to default-on).

Rationale:
1. **G-C2 fails empirically** — `circulation_integral` cannot reduce turns (smooth loops enclose MORE area than zigzags in the same bounding box).
2. **`dec_operators` itself is opt-in** — promoting `stokes_calculus` would require promoting the entire DEC machinery, a bigger decision beyond Issue 005's scope.
3. **G-A is the real test** — the Fokker-Planck validator (deferred to riir-ai Plan 334) is the headline application. Its result feeds back into the promotion decision.
4. **The 4 primitives are all correct** — opt-in status reflects the GOAT gate's outcome, not a correctness concern. Callers who want Stokes-theorem tools can enable the feature.

**Future promotion path:**
- All three GOAT gates (G-A, G-B, G-C) now have verdicts. Re-evaluation would require a new application/use-case that demonstrates a modelless win not covered by the current gates.
- If a combinatorial turn-penalty primitive is added (e.g., `path_turn_cost` that sums |d₁(field)| at faces adjacent to turns) → the G-C gate could pass. Out of scope for Issue 005.
- If the coboundary index (Issue 006) lands → G-B's win widens from 5.36× toward true O(boundary).

---

## Honest Risk Notes

- ✅ **`circulation_integral` is Stokes-theorem-correct** — 3 unit tests confirm zero-curl→0, constant-curl→curl×area, reversal antisymmetry.
- ⚠️ **G-C "≥20% fewer reversals" is structurally unreachable** via ANY rank-k cochain integral. Turn count is combinatorial; Stokes integrals are geometric. This is a mathematical fact, not a fixable bug. The plan's original framing (Plan 314 G-C) was based on a misclassification of what Stokes integrals can express.
- ✅ **The primitive has standalone value** — rotational/vortex detection, Stokes-correct area measurement, harmonic field identification.
- ❌ **G-A has already FAILED** in riir-ai Plan 334 (9.5× slower, 36% lower F1 than JS-divergence). All three GOAT gates now have verdicts: G-A FAIL, G-B PASS, G-C FAIL. `stokes_calculus` stays opt-in — the winning `boundary_flux_mass` is available to callers who need single-query zone-mass computation.

---

## References

- Issue 005 — Stokes Calculus G-C Turn Penalty
- Plan 317 — Circulation Integral (this plan)
- Plan 314 — Stokes Calculus Wrappers (parent plan, G-B PASS / G-C structural fail)
- Benchmark 314 — Stokes Calculus GOAT (G-B 5.36×, G-C line_integral discriminates but can't encode turns)
- Research 296 — Stokes Calculus DEC Vocabulary Crosswalk
