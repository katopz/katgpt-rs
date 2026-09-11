# Bench 709: gw_alignment — Gromov–Wasserstein Quotient-Alignment GOAT (Plan 594 / Issue 743)

**Status:** DONE 2026-09-11 — G1–G4 ALL PASS; `gw_alignment` stays OPT-IN (consumer-first promotion rule; the zone-belief consumer PoC is riir-poc follow-up per Issue 743 Downstream)
**Date:** 2026-09-11
**Issue:** `.issues/743_gw_alignment_quotient_primitive.md` · Plan: [`.plans/594_gw_alignment_primitive.md`](../.plans/594_gw_alignment_primitive.md) · Source: riir-ai Research 371 / Issue 912 T4 BUILD decision
**Code:** `crates/katgpt-core/src/gw_alignment/` (mod.rs + solve.rs + tests.rs, feature `gw_alignment`, pure std, zero deps)
**Run:** `cargo test -p katgpt-core --lib gw_alignment --features gw_alignment -- --nocapture` (11 tests; isolated `CARGO_TARGET_DIR=/tmp/gw_743`; CPU-only, no GPU touched)

## G1 correctness — PASS

- **Analytic 2×2** (`analytic_two_by_two_known_coupling`): D_A=[[0,1],[1,0]], D_B=[[0,2],[2,0]] — solver lands the hand-worked optimum **0.5** (uniform coupling: 1.5).
- **Brute-force permutation dominance at n=4** (`g1_brute_force_permutation_dominance_n4`, 8 geometry cases): unrestricted GW ≤ min over all 24 permutation-restricted couplings (permutation loss via the closed identity `⟨T_π, D_A T_π D_B⟩ = (1/n²)·Σ_ij a_ij·b_π(i)π(j)`), tolerance 1e-3.
- **Planted recovery at n=8** (`g1_planted_permutation_recovery_n8`, 4 cases, 2% noise): loss-based gate — planted-basin loss < 0.2% of the uniform-coupling scale (row-argmax recovery is informational only: symmetric optimal couplings legitimately split mass, measured 5/8 argmax hits at loss 0).
- **Loss identity** (`loss_matches_direct_quadratic_form`, n=3×m=4): the solver's sum-exact evaluation matches the direct O(n²m²) quadratic form AND an independent closed-form recomputation of the same coupling, both to 1e-5.
- **Isometric loss ≈ 0** (`isometric_permutation_zero_loss`, n=8): loss < 1e-3 × uniform-coupling scale.

## G2 discriminability — PASS

`g2_planted_vs_shuffled_separation_and_auc` — 16 geometries (n=12, 4-dim) × 8 noise levels (ε = 0 … 0.40), planted = permuted+noised B vs shuffled = same multiset, structure destroyed:

- **Working-noise paired dominance** (ε ∈ [0.025, 0.15], levels 1–4): planted loss < shuffled loss, asserted per level — **measured 16/16/15/15/14 of 16 geoms per level; gate ≥ 13/16**.
- **Per-level AUC (ε ≤ 0.10): gate ≥ 0.93** — measured 1.0000 / 0.9961 / 0.9922 / 0.9609 per level (standalone f64 reference).
- ε ≥ 0.2 tail: dominance degrades toward chance (11/16 → 4/16) — by construction of the sweep; pooled-AUC and tail counts are printed informationally, not gated.
- **Correspondence-break case** (`g2_gw_sees_through_correspondence_break_rsa_cannot`): planted-vs-shuffled GW separation asserted where the flat-vector RSA baseline is structurally blind (the flat multiset is permutation-invariant — recorded honestly: RSA cannot fail on a pure rename, so the gate is the planted/shuffled separation, not an RSA number).

## G3 no-regression — PASS

- Default-feature lib build: module compiles to nothing (`#[cfg(feature = "gw_alignment")]`); default lib tests **1987 passed / 0 failed** — count unchanged from pre-landing (1987 = the 1979 Plan 590 baseline + intervening default-visible additions from sibling plans).
- README feature-count claim synced **585 → 586** total flags; `./scripts/docs_gate.sh` **14/14 PASS** (12.0s wall).

## G4 alloc-free — PASS

`g4_zero_steady_state_alloc`: 16 steady-state solves after warmup — **0 allocations** (TrackingAllocator, `debug_assertions` path). Setup (`GwScratch::new`) and the warmup solve sit outside the measured window.

## Determinism (house contract)

Same inputs ⇒ bit-identical loss and score (`to_bits` equality, double-run + cross-scratch-reuse test). No RNG in the solver; fixed init; fixed GW_ITERS=128; fixed checkpoint schedule.

## Honest notes

- **Method evolution (measured, not tuned away):** (a) pure uniform-start power iteration is structure-blind at the uniform saddle AND glacial from near-uniform (n=8 isometric: 0.84 → 0.36 over 64 iters, still descending) — the greedy second-order init replaces it; (b) an earlier Frank–Wolfe atom + line-search overlay non-monotonically degraded good inits — removed; (c) a 2-opt "polish" on Σ P walked a 0.0016 coupling to 0.145 (Σ P is not the GW objective) — removed. The shipped algorithm: multi-start (greedy-on-P, entropic softmin, uniform+tilt; + brute-force best permutation when square n ≤ 8) → power loop with fixed-schedule checkpointing (best-seen restored) → sum-exact loss.
- **f32 coupling, f64 loss:** the coupling stores f32; the loss evaluates through an f64 `D_A·T` staging (`buf64`) — an f32 round-trip there cost ~1e-3 relative drift, the direct-form identity's whole tolerance.
- **Score β is calibration, not learning:** `GW_SCORE_BETA = 4.0`; consumers re-derive from raw loss if needed.
- **Solver-endpoint vs per-iteration best:** the per-start checkpointing restores the best-seen coupling; non-monotone segments (measured: a 0-loss init degrading to 0.29 over 128 iters on one n=12 pair) motivated fixed-schedule checkpointing over raw endpoints.
- **Scope:** n, m ≤ 64, uniform weights, no fused GW, no TDA (the katgpt-dec TDA lane owns stage-1 orbits per Issue 743).
- GPU-exclusivity: CPU-only eval, no compute workload touched (Unity/Zed exempt per the standing rule; nothing to note).
