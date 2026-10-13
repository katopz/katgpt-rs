# Issue 938 — lattice_memory: 3D axis remedy + measured-erosion capacity gate

**Status:** CLOSED 2026-10-13 — T1 (capacity parameterization) LANDED, gates PASS (random keys bracket the aggregate law, promise 0.723 ≥ 0.5 at λ=11; clustered keys saturate — the eroded surface is conservative, the safe direction). T2 (3D axis) LANDED behind `lattice_memory_3d` and **RECORDED NEGATIVE at G2** ([Bench 938](../.benchmarks/938_lattice_3d_axis_goat.md): G1 0.9345 + G5 0.184-vs-0.004 + G6 857 MB all PASS, G2 FAIL — 27 taps cost 2.90× kernel MACs, 7379 vs PKM 5171 ns at 10⁶; stays opt-in, the negative is Bench 919's second measured remedy). T3 deferred (the 2-tap kernel is the named re-open lever). T4 landed (`read_direct` + the exact-round-trip gate). Record: HISTORY.md; removed per the noise rule.

**Source:** μLattice — [github.com/eelbaz/mLattice](https://github.com/eelbaz/mLattice) @ `7545dcf1d5ed920681a8f59159657a231c9186dc` (MIT, Tensorbend Research, Oct 2026, "μLattice: Breaking the Quadratic Memory Barrier in AI"). This is an independent TRAINED reproduction of Percepta Spotlight Memory: the same cos² 3×3 bump and per-cell gated-delta states, MQAR 1.0000 @ 8,192 pairs / T=133K (≈1/16 of Spotlight's claimed scale), and MAD 0.834 with Copy = 0.003.
**Prior art (ours, controlling):** [Plan 619](../.plans/619_lattice_memory_e2lsh_delta_cells.md) (`katgpt-core/src/lattice_memory/`, landed 2026-10-07) · [Bench 919](../.benchmarks/919_lattice_family_goat.md) (floor NOT PROMOTED at 10⁶) · riir-refine Research 233 (Percepta Spotlight distill). The class is already covered, so this issue has no novelty claim. It targets a shipped primitive's recorded negative and its sizing surface.
**Sibling routing:** the model-based track is riir-train Plan 440 (revised 2026-10-13 with mLattice's reference code and its measured training constraints).

## Why

Bench 919 recorded the envelope as **(support × capacity)-bounded**: at
d_k = 32 the 0.05σ near-miss claim caps out at about 63k items, and the only
remedy it recorded is widening d_k. That breaks key = embedding, so it is its
own gate. mLattice surfaces two things our record lacks:

1. **A second remedy: the axis count.** The sizer's law generalizes to
   `N_max ≈ 0.35·d_k·(3.76/R)^AXES`. At d_k = 32 and R = 0.05σ:
   2D ≈ 63k, **3D ≈ 4.76M**. That re-opens the exact failed G5 cell
   (R = 0.05σ, N = 10⁶) without touching d_k. `address.rs:19` hard-codes
   `pub const AXES: usize = 2`. mLattice §4.7 supplies a 3D addressing
   **analysis only**: nobody has measured 3D recall, and its scale-96 cell
   count (6,960,697) contradicts its own arithmetic (195³ = 7,414,875).
   Recompute; never quote it.
2. **Measured density vs the random-key assumption.** `capacity_per_cell(d_k)
   = 0.35·d_k` (`sizer.rs:43`) assumes same-cell keys erode at
   `E[cos²] = 1/d_k` (0.031 at d_k = 32), which is ≈ 11 associations per cell.
   mLattice §5.1 (d_k = 32, learned content keys) measured recall degrading at
   **far lower load**: 0.24 keys/cell → 0.98, 0.49 → 0.96 (plateau), and
   0.22 → 1.0000. With the 3×3 bump that is ~2–4 keys per neighbourhood, not
   11. Correlated keys eroding faster than `1/d_k` is one candidate mechanism.
   A two-point fit gives E[cos²] ≈ 0.083 (our inference, **not a number they
   report**), which would put a `sized()` "within capacity" consumer at
   survival ≈ 0.39. Learned-address dynamics is the competing explanation, and
   it does not transfer to us. So this is a HYPOTHESIZED sizing hazard: if it
   holds, the refusal surface says yes when the physics says no.

## Tasks

### T1 — measured-erosion capacity parameterization (sizing hazard)

- [x] **T1.1** Generalize the sizer: `capacity_per_cell(d_k, e_cos2, target_recall)`
      (derivation: survival `(1 − e_cos2)^λ ≥ target` → `λ ≤ ln(1/target)/e_cos2`,
      halved for the cross-term floor). The pinned defaults `e_cos2 = 1/d_k` and
      `target = ½` reproduce today's `0.35·d_k` bit-for-bit. Keep the 1-arg
      form as a thin wrapper so callers don't churn. — LANDED as
      `capacity_per_cell_eroded` (+ the `capacity_per_cell` wrapper); the pin
      is returned directly (bit-for-bit asserted for d_k 1..128).
- [x] **T1.2** `sized()` already fits a corpus. Add an opt-in estimate of the
      same-cell `E[cos²]` from the fit samples (sample pairs that share a cell
      under the fitted grid), and use it when it exceeds `1/d_k`. It must stay
      zero-alloc in the hot path (fit-time only). — LANDED as `sized_eroded`
      (capped sorted-scan estimator, fit-time only).
- [x] **T1.3** Gate (G1): sweep λ vs recall on OUR fixtures, both random keys
      and correlated keys (a clustered/vocabulary-style corpus). The predicted
      survival curve must bracket the measured one within the stated tolerance.
      **Do not adopt mLattice's 0.083.** It is two points on their protocol
      (learned content keys, 16K synthetic vocab). It is directionally right
      and specific to that protocol. — PASSED with a recorded refinement:
      random keys bracket the aggregate law (promise 0.723 ≥ 0.5 at λ=11);
      clustered keys SATURATE above it (0.30 vs predicted 0.131 at λ=11,
      e=0.184) and track the exact key-recursion algebra ±0.08 — the eroded
      surface is CONSERVATIVE for parallel keys (the safe direction), and the
      0.083 figure stays a hypothesis input in one unit test, never adopted.
- [x] **T1.4** G3 no-regression: Bench 919's six gates are unchanged at
      default parameters (the defaults are bit-identical by construction, and
      the test asserts it). — Re-run in-session: G1 recalls digit-identical
      (1.0000/0.9885/0.9790/0.9830), G2/G5/G3/T1.8 verdicts unchanged,
      N_max 63,336 identical.

### T2 — 3D axis (the un-recorded second remedy to Bench 919's NOT-PROMOTED)

- [x] **T2.1** Make `AXES` a const generic (or a sealed `Axes2`/`Axes3`
      type), not a `pub const`. Bump support becomes 3×3×3 = 27 taps. The
      tent kernel generalizes per axis (it is a product kernel). Behind a
      `lattice_memory_3d` feature (opt-in). — LANDED: `LshAxes<A>` /
      `LatticeConfigG<A>` / `LatticeMemoryG<A>`, aliases keep the 2D surface
      byte-stable (bench 919 re-run: digit-identical G1, identical verdicts),
      `LatticeMemory3D` behind `lattice_memory_3d`.
- [x] **T2.2** Generalize the sizer law to `L = ⌈3.76/R⌉` per axis with
      cells = L^AXES, and recompute the memory budget (G6). At R = 0.05σ,
      L = 76 → 438,976 cells × (d_k·d_v·4 B) must be priced against the 2 GiB
      bar **before** the build. If it doesn't fit, record the negative. —
      PRICED: 1,798,045,696 B = 1.674 GiB at d_v=32 — UNDER the bar (~16%
      headroom), build unblocked; d_v=64 recorded as the over-bar boundary
      row; `grid_side_for` extracted; gated in `lattice_memory_3d_gates`.
- [x] **T2.3** Re-run Bench 919's G5 cell (R = 0.05σ, N = 10⁶, ε sweep
      0.1R → 2R) with 3D. Pass = graded recovery at ε = 0.05σ (the plan's
      original bar). — PASS: 0.184 vs Engram 0.004; the curve is graded
      through the support (0.934 → 0.917 → 0.766 → 0.186 → 0.184 → 0.046).
- [x] **T2.4** G2: 27-tap read vs 9-tap (≈3× MACs) against the PKM baseline
      (5730 ns at 10⁶). A 3D lattice slower than PKM is a recorded negative,
      not a failure to hide. `--release`, interleaved arms
      (`tests/common/ab_timing.rs`), with box state recorded. — RECORDED
      NEGATIVE: 7379 ns vs PKM 5171 ns (0.70×); the cell-matched tap arm
      measured the cause (1808 vs 623 ns = 2.90×, the predicted ≈3× kernel
      scaling). Box state: PROVENANCE AC/load 2.49/powermode=high, release,
      interleaved best-of-5.
- [x] **T2.5** Verdict: promote `lattice_memory_3d` only if G1/G2/G5/G6 all
      pass. Otherwise record the negative in Bench 919's riders as the second
      measured remedy. — G2 fails → NOT promoted; rider #4 added to Bench 919;
      [Bench 938](../.benchmarks/938_lattice_3d_axis_goat.md) is the record.

### T3 — secondary A/B queue (T1.8 discipline: each A/B decides a posture, not the verdict)

- [-] **T3.1** Spread-write (write to all bump taps, weighted) vs our
      primary-cell write. mLattice writes through the bump. — DEFERRED:
      decides a posture, not this verdict; the G2 negative does not hinge on
      it.
- [-] **T3.2** 2-tap bilinear kernel. mLattice's `kernel.py` defaults to the
      blog's cos²(πt/3) with support 1.5. It also ships `cos2_half`, cos²(πt/2)
      with support 1.0, and its comment says that variant is what Percepta's
      released reference figure code uses. A 2×2 read is 4 taps vs 9 (2D) or 8 vs 27
      (3D), which matters most for T2.4. — DEFERRED and NAMED: this is the
      re-open lever for the T2.4 negative (8 taps vs 27 at 3D); an owner
      decision on whether the support-cost trade is worth an A/B.
- [-] **T3.3** Deterministic seed selection via mLattice's write-overlap
      statistic: pick the LSH seed that minimizes same-cell overlap on the
      fit corpus, using `config_seed` candidates only (Issue 809 law preserved). —
      DEFERRED: optimization, not correctness; the sized constructor's
      occupancy histogram already matches its prediction.

### T4 — boundary rider: positional copy needs an unblended read

- [x] **T4.1** Add `read_direct(x, out)`, an unblended primary-cell read of the
      `write_value` lane. The tent's 0.6 exact-hit centre mass is diluted by
      `read_blend`. mLattice found the same thing independently: MAD Copy
      = 0.003 for content-addressed reads, and they added
      `write_cell`/`read_cell` for the same reason. — LANDED on the generic
      core (both arities); returns whether the primary cell was written;
      zero-alloc (Gated in the 3D G4 arm).
- [x] **T4.2** Test: positional copy via `write_value` + `read_direct` is exact
      (bit-identical round-trip), while `read_blend` is graded. This documents
      which read a positional consumer must use. — LANDED:
      `read_direct_is_the_exact_write_value_round_trip` (2D) + the direction
      and dilution assertions + freeze/thaw preservation at both arities.

## GOAT gates

| gate | item | bar |
|---|---|---|
| G1 | T1.3 / T2.3 | predicted-vs-measured survival bracketed; graded recovery at ε = 0.05σ, N = 10⁶ (3D) |
| G2 | T2.4 | 3D read latency vs PKM at 10⁶, `--release`, interleaved |
| G3 | T1.4 | Bench 919 gates unchanged at defaults |
| G4 | T1.2 / T2.1 | zero alloc on read/write hot paths |
| G6 | T2.2 | 3D memory at 10⁶ under 2 GiB, priced before the build |

## Weakest points (named up front)

- T2's capacity lift rests on an analysis plus an algebraic generalization of
  our own law. There are zero 3D recall measurements anywhere, and mLattice
  §4.7 carries an erratum, so its analysis was not fully proofread. A negative
  result is an acceptable product. — OUTCOME: the negative arrived exactly
  where the analysis was silent (latency), while the recall/memory analysis
  held (G1/G5/G6 all pass).
- T1's 0.083 is our two-point inference from a foreign, learned-address
  protocol, and the mechanism is unconfirmed. The gate measures our own
  fixtures, and "the 0.35·d_k law holds for our correlated keys" is an
  acceptable outcome.
