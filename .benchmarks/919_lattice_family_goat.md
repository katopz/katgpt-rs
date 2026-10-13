# Bench 919 — LatticeMemory family GOAT (Plan 619 T1.7/T1.8)

**Status:** MEASURED — floor NOT PROMOTED at 10⁶ (G5 bar beyond the point's support; the negative + the support×capacity law recorded); all six gates green, verdict below
**Date:** 2026-10-07
**Plan:** [619_lattice_memory_e2lsh_delta_cells.md](../.plans/619_lattice_memory_e2lsh_delta_cells.md) T1.7 + T1.8
**Bench target:** `cargo bench -p katgpt-core --features lattice_memory,engram --bench bench_919_lattice_goat` (G4 cell: add `alloc_tracking`)
**Box:** M3 Max, 16 cores, quiet, AC power — workstation readings; quote with a box-state caveat

## Verdict

**FLOOR: NOT PROMOTED.** G2 ✓ (lattice 3755 ns vs PKM 5730 ns at 10⁶) · G6 ✓ (767 MB < 2 GiB) · **G5 ✗** — the plan's bar (graded recovery at ε = 0.05σ) sits 8× beyond the 10⁶ point's support radius (0.006σ). The class claim itself HOLDS at its scale: within the delivered support the recovery is graded (0.983 → 0.912 → 0.399 → 0.157 across 0.1R → 2R) where Engram cliffs instantly (0.379 → 0.004 at ε = 0.001σ) — the Claim A differentiator is real, it just cannot be exercised at 0.05σ and 10⁶ items simultaneously with d_k = 32.

**The governing law (the bench's product, more than a promote/demote):**

```
N_max ≈ 0.35·d_k·(3.76/R)²      — 63,336 items at d_k=32, R=0.05σ
```

The support radius drives the grid (`L = ⌈3.76/R⌉`, the T1.5 sizer law — the first occupancy-driven cut was REFUTED by this bench's G5 cell: at 10⁶ it chose a 0.005σ support and recall CLIFFED to 0.06 at ε = 0.02σ, exactly the failure the plan's gate exists to catch; the sizer was rewritten to the support law in the same landing), and the delta-rule capacity (0.35·d_k, first-order erosion bound) gates the crowding. Where the consumer's (N, R) fits under the law, the lattice wins its gates; where it doesn't, the sizer REFUSES with the named remedy (`OverCapacity { required_d_k }`).

## Gate results (d=32, d_k=32, d_v=16, auto-radius policy R = min(0.05σ, λ≤6 radius))

| N | support σ | G1 recall@1 (brute 1.0) | G2 lattice ns | G2 pkm ns | G2 engram ns | G6 slab |
|---|---|---|---|---|---|---|
| 10³ | 0.050 | 1.0000 | 357 | 162 | 196 | 11 MB |
| 10⁴ | 0.050 | 0.9885 | 1016 | 502 | 214 | 11 MB |
| 10⁵ | 0.028 | 0.9790 | 2453 | 1572 | 259 | 77 MB |
| 10⁶ | 0.006 | 0.9830 | **3755** | **5730** | 280 | 767 MB |

- **G1** — flat vs N (0.979–1.0 at 10⁶× growth): the growing-state claim holds; the brute-force anchor is 1.0 by construction (same window, exact item at distance 0).
- **G2** — the O(1) claim holds asymptotically (MACs constant) but the constant is cache-shaped: 357→3755 ns tracks the slab (11 MB→767 MB) leaving L2/L3 — still a 1.53× WIN over PKM O(√N) at 10⁶; Engram's hash+memcpy (280 ns) is the floor nobody beats at equal memory. Latency numbers are box-shaped: interleaved best-of-5 rounds, quiet M3 Max.
- **G3** — PASS: newest-value recall after 8 same-cell near-collinear overwrites ✓ both engines; old value eroded to 0.034 (the sin²θ removal law); far-key association retained at 0.273 ≈ kernel scale vs eroded 0.034 — the structural win Engram's single-slot overwrite cannot have.
- **G5** — the Claim A differentiator, gated directly: graded within support (tent, N=10⁵, R=0.019σ: 0.973 → 0.898 → 0.405 → 0.162 → 0.105) vs Engram's hash cliff (0.373 → 0.007 at ε = 0.002σ) and PKM's flat quantization plateau (~0.17 everywhere — coarse cells generalize but never rank). **The fixed 0.05σ floor bar FAILS at 10⁶** — recorded as the negative the plan asks for.
- **G4** — PASS (0 allocs / 2000 write+read+score, `alloc_tracking` release posture). The first two runs measured 4000/2001 allocs — BOTH were the harness's own `to_vec`/precompute inside the measured region; the primitive was clean all along (the unit gate `tests/lattice_memory_gates.rs` pins it independently).
- **G6** — bytes/item tracks `4·d_k·d_v/λ` (11829 B/item at λ=0.17 … 805 B/item at λ=2.8); linear memory is on the table, not hidden under flat latency.

## T1.8 kernel A/B (cos² vs tent, N=10⁵, same support 0.0189σ)

| kernel | near-miss AUC | exact recall | verdict |
|---|---|---|---|
| tent | 0.0210 | **0.9725** | **WINS — stays the default** (`BumpKernel::Tent`) |
| cos² | **0.0223** | 0.9685 | demoted — +6% AUC bought with −0.4pt exact recall, fails the exact-recall floor (≥ tent − 0.1pt) |

cos² stays selectable (`BumpKernel::Cos2`) with its truncation-leak shape pinned in `bump.rs` (far-tap mass ~0.078 at the primary switch, where the tent is exactly 0). The plan's "cos² exists for SGD differentiability, which we don't have" reading is MEASURED, not assumed.

## Recorded negatives / riders

1. **The occupancy-driven sizer (the first T1.5 cut) is REFUTED by G5** — sizing from `items/λ` decouples the bump support from the embedding scale at scale (10⁶ → 0.005σ support → cliff at ε = 0.02σ). Rewritten in the same landing: `L = ⌈3.76/R⌉` (support drives), occupancy floats, `capacity_per_cell(d_k) = 0.35·d_k` gates, `OverCapacity` names the remedy. The histogram gate now predicts from the support-driven λ.
2. **The class's operating envelope is (support × capacity)-bounded** — at d_k=32 the 0.05σ claim caps at ~63k items. The consumer plan (riir-refine Plan 199 traj/insight evidence memory) targets ≪ that; anything bigger needs d_k > 32 (a key-projection widening — NOT implemented here; that is its own gate: the key would stop being the embedding).
3. **Rider:** `examples/karc_deployed_shape_quality` failed the `--no-default-features --all-targets` posture (E0432, pre-existing) — got its `required-features = ["karc_forecaster"]` row in this landing.
4. **Rider (Issue 938, 2026-10-13 — the SECOND measured remedy):** the 3-axis lattice clears the cell this floor died at — at R = 0.05σ, N = 10⁶ the 3D sizer accepts (λ = 2.28, slab 857 MB at d_v = 16), G1 recall 0.9345, G5 graded recovery 0.184 vs Engram 0.004 at ε = 0.05σ — but FAILS G2: the 27-tap read costs 2.90× the 9-tap kernel (cell-matched arm 1808 vs 623 ns) and lands 7379 ns vs PKM's 5171 ns at 10⁶. `lattice_memory_3d` stays opt-in; full record: [`.benchmarks/938_lattice_3d_axis_goat.md`](938_lattice_3d_axis_goat.md). The envelope now has two measured remedies — widen d_k (breaks key = embedding) or widen AXES (recall+memory clear, latency does not); the named re-open lever is a 2-tap kernel (8 taps vs 27 at 3D).
