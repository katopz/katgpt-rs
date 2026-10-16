# Bench 938 — lattice_memory 3D axis + measured-erosion capacity (Issue 938 T1/T2)

**Status:** COMPLETE 2026-10-13 — T1 (measured-erosion capacity) GATES PASS; T2 (3D axis) RECORDED NEGATIVE at G2 (G1/G5/G6 PASS, G2 FAIL: the 27-tap read loses to PKM). `lattice_memory_3d` stays OPT-IN; the negative is the second measured remedy in Bench 919's story. T3 (kernel/write A/B queue) DEFERRED.

**Source:** Issue 938 (the μLattice distill) — `.issues/938_lattice_memory_3d_axis_and_measured_capacity_gate.md`. Prior art: [Bench 919](919_lattice_family_goat.md) (2D floor NOT PROMOTED, support×capacity-bounded), Plan 619, riir-refine Research 233.

**Posture:** M3 Max 16 cores, d = d_k = 32, d_v = 16 (bench) / 32 (G6 worst case), queries = 2000, window = 200, interleaved best-of-5 rounds, `--release`.
**Box state (Issue 021 law):** `PROVENANCE: power=AC Power load=2.49 swap=0.00M canary=skipped powermode=2(high)` (riir-reflex `bench_preflight.sh`), RAM 90% free, no concurrent cargo, load avg 2.53/7.63/7.51 (decaying tail from earlier work), 2026-10-13 ~10:30. Latency numbers are workstation readings — quote them with this caveat.

## T2.2 — the G6 pricing, BEFORE the build (the gate that unblocked everything)

| posture | cells | slab | verdict |
|---|---|---|---|
| 3D, R = 0.05σ, d_k = d_v = 32 | 76³ = 438,976 | **1,798,045,696 B = 1.674 GiB** | UNDER the 2 GiB bar (~16% headroom) — build UNBLOCKED |
| 3D, R = 0.05σ, d_k = 32, d_v = 64 | 438,976 | 3,596,091,392 B | OVER — the recorded boundary row (needs a budget conversation or a thinner value) |
| 2D, R = 0.05σ (any d_v) | 5,776 | — | the cell 2D cannot SIZE: λ = 173.1 ≫ 11.2 (the Bench 919 negative, as a constructor verdict) |

Capacity envelope, recomputed and pinned (never quoted from μLattice §4.7, whose scale-96 cell count carries an erratum): `N_max ≈ 0.35·d_k·(3.76/R)^AXES` → 2D **63,336**, 3D **4,762,900** at d_k = 32, R = 0.05σ. Gated in `lattice_memory_3d_gates::g6_memory_is_priced_before_the_build`.

## T2.3/T2.4/T2.5 — the 10⁶ G5 cell re-run at 3D

| gate | bar | measured | verdict |
|---|---|---|---|
| G1 recall@1, exact keys, N = 10⁶ | ≥ 0.90 | **0.9345** | PASS — the cell 2D could not size HOLDS 0.93 in 3D (λ = 2.28, support 0.0495σ) |
| G5 graded recovery @ ε = 0.05σ | lattice > Engram + 0.05 | **0.184 vs 0.004** | PASS — the ε-curve is genuinely graded through the support: 0.934 → 0.917 (0.1R) → 0.766 (0.3R) → 0.186 (1.0R) → 0.184 (0.05σ) → 0.046 (2R); Engram's hash cliff sits at 0.004 everywhere past ε = 0 |
| G2 read latency @ 10⁶ | 3D < PKM | **7379 ns vs PKM 5171 ns (0.70×)** | **FAIL — the recorded negative** |
| G6 slab @ the bench posture | under 2 GiB | 857 MB (d_v = 16) | PASS |
| G4 hot path (2000 write+read+score, TrackingAllocator, release) | 0 allocs | **0 allocs, 0 B** | PASS |

Tap-cost arm (cell-count-matched ~110k cells, d_k = 32, d_v = 16, 10k items, 9-tap 2D [332,332] vs 27-tap 3D [48,48,48]): **623 ns vs 1808 ns = 2.90×** — the kernel cost scales with the tap count as the MAC arithmetic predicts (3× taps), and that is the whole G2 story: 3D buys the capacity remedy with ~2.9× kernel MACs, and at d_v = 16 the read (13.8k MACs at λ = 2.28, most taps unwritten) lands ~2× the 2D read (3656 ns in the same-session 919 run) — past PKM's O(√N) 5171 ns.

**T2.5 VERDICT: RECORDED NEGATIVE** — G1/G5/G6 all pass (the second remedy is REAL at the recall/memory axis), G2 fails (the remedy costs ~2.9× kernel MACs and loses to PKM at 10⁶). Per the issue: `lattice_memory_3d` stays opt-in, the negative is recorded here and as a rider in Bench 919 — the support×capacity envelope now has TWO measured remedies: widen d_k (breaks key = embedding, its own gate) or widen AXES (this bench: recall and memory clear, latency does not). The pointed-at remedy for a future re-open is T3.2's 2-tap kernel (8 taps vs 27 at 3D) — an A/B that decides a posture, not this verdict.

## T1 — the measured-erosion capacity gate (G1, fixture-scale, deterministic)

Implementation (all in `sizer.rs`): `capacity_per_cell_eroded(d_k, e_cos2, target_recall)` — the survival law `λ ≤ ln(1/target)/e`, cross-term-halving calibrated so the pinned defaults (`e = 1/d_k`, `target = ½`) return EXACTLY `0.35·d_k` bit-for-bit (asserted at d_k ∈ {1..128} incl. non-powers-of-two — the T1.4 G3 law holds by construction). `sized_eroded(...)` estimates the same-cell `E[cos²]` from the fit corpus (sorted-scan pair census, capped 24/cell + 200k pairs, fit-time only — the hot path stays 0-alloc) and tightens the refusal surface when the estimate exceeds `1/d_k`.

Gates (`lattice_memory_gates::erosion_survival_brackets_the_predicted_curve`, d_k = 32, grid 8×8, λ ∈ {2, 6, 11, 16, 22, 32}, 24 trials/rung, fresh lattice per measurement, values orthogonalized against v₀ so the v₀-projection IS the erosion signal):

- **Random keys** (ê = 0.032 ≈ 1/d_k): median survival brackets the aggregate law `(1−ē)^(λ−1)` at every rung (±0.12 band; lower bound `p² − 0.12`), and the shipped promise holds: **survival 0.723 at the pinned capacity rung λ = 11 (bar ≥ 0.5)** — `0.35·d_k` is honest on our random keys.
- **Correlated keys** (clustered fixture, ê = 0.184 ≈ 6× the random-key assumption): the aggregate law's prediction (0.131 at λ = 11) sits FAR BELOW the measurement (≈ 0.30) — the removals share a direction, the effective key rotates away from the cluster, and survival **saturates** instead of decaying geometrically. The measured survival tracks the EXACT key-recursion algebra `k_eff ← k_eff − (k_effᵀkw)kw` within ±0.08 at every (trial, rung) — the delta-rule erosion identity, pinned. Direction of the divergence: the aggregate law (and `sized_eroded`'s refusal surface built on it) is **CONSERVATIVE for parallel-key corpora** — it over-refuses, the safe direction (>90% of (trial, rung) cells). The dangerous direction (faster-than-law erosion) did not appear on our fixtures.
- **T1.2 end-to-end** (`eroded_sizer_tightens_on_correlated_keys_only`): the clustered corpus is REFUSED by `sized_eroded` (capacity tightens 2.8 → 0.38) where `sized` accepts; the random corpus gets identical verdicts from both (no behavior change without measured cause).
- **μLattice's 0.083 was never adopted** — the two-point fit appears exactly once, as a hypothesis input in the `capacity_per_cell_eroded` unit test's arithmetic pin (0.35/0.083 ≈ 4.2 associations), labeled as their protocol's number.

## T3 — DEFERRED (`- [-]`)

T3.1 spread-write / T3.2 2-tap kernel / T3.3 seed selection: each is an A/B that decides a posture, not this verdict; T3.2 is the named re-open lever for the G2 negative (8 taps vs 27 at 3D) and stays queued in Issue 938.

## G3 — no-regression (the 2D surface under the const-generic refactor)

Bench 919 re-run in the same session, same box, release: G1 recalls **1.0000 / 0.9885 / 0.9790 / 0.9830** — digit-identical to the recorded table; G2 lattice 3656 ns (record 3755) vs PKM 5844 (record 5730) — box noise around identical verdicts (WIN at 10⁶); G6 767 MB identical; G3 overwrite PASS; T1.8 tent-wins verdict unchanged; N_max 63,336 identical; FLOOR: NOT PROMOTED unchanged. The A = 2 monomorphization is behaviorally unchanged: same draw order, same accumulation order, same commitment byte stream.

## What landed

- `crates/katgpt-core/src/lattice_memory/{mod,address,types,sizer}.rs` — the const-generic core (`LatticeMemoryG<A>`/`LatticeConfigG<A>`/`LshAxes<A>`; aliases `LatticeMemory`/`LatticeConfig` = the A=2 surface), `read_direct`, the eroded sizer, `grid_side_for`.
- `crates/katgpt-core/tests/lattice_memory_3d_gates.rs` + `benches/bench_938_lattice_3d_goat.rs` (+ Cargo.toml feature `lattice_memory_3d = ["lattice_memory"]` + the two required-features rows).
- Feature-gate discipline: `lattice_memory_3d` is opt-in, implies `lattice_memory`; the measured posture is this bench; the alias is never a default.
