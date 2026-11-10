//! Plan 416 GOAT Gate — Region-Conditioned Subspace Field.
//!
//! Runs the G1–G5 GOAT gate for the `region_subspace_steering` feature.
//! Mirrors the convention of `tests/bench_412_subspace_steering_goat.rs`.
//!
//! Run with:
//! ```sh
//! cargo test -p katgpt-core \
//!   --test bench_416_region_subspace_goat -- --test-threads=1
//! ```
//!
//! # Gates
//!
//! - **G1 (parity)**: `steer_local` at degenerate K=1, μ=0, W=I must be
//!   bit-identical to Plan 412's `apply_subspace_steering`. 100 random offsets
//!   × D=8 = 800 element comparisons, 0 mismatches.
//! - **G2 (two-mode steering)**: K=2 field; centroid steering produces distinct
//!   outputs per region; local steering produces distinct outputs per region.
//! - **G4 (latency smoke)**: 100k `steer_local` + `steer_centroid` +
//!   `membership_gates` calls at D=8 K=8 R=2 complete under headroom.
//! - **G5 (determinism)**: commitment + decompose + reconstruct bit-identical
//!   for fixed state + field.

#![allow(clippy::float_cmp)]

use katgpt_core::region_subspace::{
    RegionDecomposition, RegionSubspaceField, compute_field_commitment, reconstruct,
};
use katgpt_core::subspace_steering::SubspaceSteeringField;

// Issue 855: the load-invariant timing treatment. `best_of_us` panics on an
// all-zero reading instead of letting a ceiling be satisfied by a loop the
// optimiser deleted.
#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

/// Build an R×D identity-ish loadings block: axis r has a 1.0 at index r.
fn identity_loadings<const D: usize, const R: usize>() -> [[f32; D]; R] {
    let mut block = [[0f32; D]; R];
    for (r, row) in block.iter_mut().enumerate() {
        row[r] = 1.0;
    }
    block
}

/// Deterministic LCG for reproducible pseudo-random offsets.
struct Lcg {
    state: u32,
}
impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((self.state >> 8) as f32 / 65535.0) * 2.0 - 1.0 // ∈ [-1, 1)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// G1 — Degenerate K=1 parity with Plan 412 (load-bearing gate)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn g1_k1_parity_with_plan_412_100_random_offsets() {
    const D: usize = 8;
    const K: usize = 1;
    const R: usize = D; // R=D so W_1 = I_D

    let centroids = [[0f32; D]; K];
    let loadings = [identity_loadings::<D, R>()];
    let log_pi = [0f32; K];
    let psi_inv = [1f32; D];

    let region_field =
        RegionSubspaceField::<D, K, R>::new_unchecked(centroids, loadings, log_pi, psi_inv);
    let block_412 = identity_loadings::<D, D>();

    let mut rng = Lcg::new(42);
    let mut mismatches = 0usize;

    for _case in 0..100 {
        let mut offset = [0f32; R];
        for o in &mut offset {
            *o = rng.next_f32();
        }

        // Region field: steer_local.
        let mut state_region = [0f32; D];
        region_field.steer_local(&mut state_region, 0, &offset);

        // Plan 412 field: apply.
        let field_412 = SubspaceSteeringField::<D, D>::new_unchecked(block_412, offset);
        let mut state_412 = [0f32; D];
        field_412.apply(&mut state_412);

        for d in 0..D {
            if state_region[d].to_bits() != state_412[d].to_bits() {
                mismatches += 1;
            }
        }
    }

    println!("── G1: K=1 parity with Plan 412 ──");
    println!("   comparisons: 100 offsets × D={D} = {} elements", 100 * D);
    println!("   mismatches:  {mismatches}");
    println!("   threshold:   0");
    assert_eq!(
        mismatches, 0,
        "G1 FAIL: {mismatches} bit mismatches vs Plan 412"
    );
    println!("   G1 ✓ PASS");
}

// ═══════════════════════════════════════════════════════════════════════════════
// G2 — Two-mode steering produces distinct region/local effects
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn g2_two_mode_steering_distinct() {
    const D: usize = 8;
    const K: usize = 4;
    const R: usize = 2;

    // K=4 regions with distinct centroids.
    let mut centroids = [[0f32; D]; K];
    for (k, centroid) in centroids.iter_mut().enumerate() {
        centroid[0] = (k as f32) * 10.0; // centroids along dim 0: 0, 10, 20, 30
    }
    // Each region has identity loadings (same subspace for all — tests centroid distinction).
    let loadings = [identity_loadings::<D, R>(); K];

    let field =
        RegionSubspaceField::<D, K, R>::new_unchecked(centroids, loadings, [0f32; K], [1f32; D]);

    // G2a: centroid steering toward different regions produces distinct outputs.
    let base = [0f32; D];
    let mut steered_states = [[0f32; D]; K];
    for (k, state) in steered_states.iter_mut().enumerate() {
        *state = base;
        field.steer_centroid(state, k, 0.5);
    }
    // Each steered state should differ from every other.
    let mut centroid_distinct = 0usize;
    let mut centroid_pairs = 0usize;
    for i in 0..K {
        for j in (i + 1)..K {
            centroid_pairs += 1;
            let dist: f32 = steered_states[i]
                .iter()
                .zip(steered_states[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            if dist > 1.0 {
                centroid_distinct += 1;
            }
        }
    }
    println!("── G2a: centroid steering distinct per region ──");
    println!("   distinct pairs: {centroid_distinct}/{centroid_pairs}");
    assert_eq!(
        centroid_distinct, centroid_pairs,
        "G2a FAIL: not all centroid-steered states are distinct"
    );

    // G2b: local steering with same offset but different regions produces distinct
    // outputs when the loadings differ. Build a field where each region has a
    // different subspace.
    let mut diff_loadings = [[[0f32; D]; R]; K];
    for k in 0..K {
        // Region k: axes along dims (2k, 2k+1).
        diff_loadings[k][0][2 * k] = 1.0;
        diff_loadings[k][1][(2 * k + 1).min(D - 1)] = 1.0;
    }
    let diff_field = RegionSubspaceField::<D, K, R>::new_unchecked(
        [[0f32; D]; K],
        diff_loadings,
        [0f32; K],
        [1f32; D],
    );
    let offset = [1f32, 1f32];
    let mut local_states = [[0f32; D]; K];
    for (k, state) in local_states.iter_mut().enumerate() {
        *state = base;
        diff_field.steer_local(state, k, &offset);
    }
    let mut local_distinct = 0usize;
    let mut local_pairs = 0usize;
    for i in 0..K {
        for j in (i + 1)..K {
            local_pairs += 1;
            let dist: f32 = local_states[i]
                .iter()
                .zip(local_states[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            if dist > 1.0 {
                local_distinct += 1;
            }
        }
    }
    println!("── G2b: local steering distinct per region subspace ──");
    println!("   distinct pairs: {local_distinct}/{local_pairs}");
    assert_eq!(
        local_distinct, local_pairs,
        "G2b FAIL: not all local-steered states are distinct"
    );

    println!("   G2 ✓ PASS");
}

// ═══════════════════════════════════════════════════════════════════════════════
// G4 — Latency smoke (structural size + 100k calls)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only timing gate (debug ~10x + machine-marginal under load; the struct-size half stays pinned by the release run)"
)]
fn g4_latency_smoke_and_struct_size() {
    const D: usize = 8;
    const K: usize = 8;
    const R: usize = 2;

    let centroids = [[0f32; D]; K];
    let loadings = [identity_loadings::<D, R>(); K];
    let field =
        RegionSubspaceField::<D, K, R>::new_unchecked(centroids, loadings, [0f32; K], [1f32; D]);

    // Structural size check.
    let expected_size = K * D * 4       // centroids
        + K * R * D * 4                 // loadings
        + K * 4                          // log_pi
        + D * 4                          // psi_inv
        + K * R * D * 4                  // projectors
        + 32; // commitment
    let actual_size = std::mem::size_of::<RegionSubspaceField<D, K, R>>();
    println!("── G4a: structural size ──");
    println!("   D={D} K={K} R={R}");
    println!("   size_of:     {actual_size} bytes");
    println!("   expected:    {expected_size} bytes");
    assert_eq!(actual_size, expected_size, "G4a FAIL: struct size mismatch");

    // 100k calls of steer_local + steer_centroid + membership_gates.
    let mut state = [0f32; D];
    let offset = [0.5f32; R];
    // Issue 855: the previous loop ran this body 100 000 times with
    // `let _gates = …` discarding the only returned value and `state` dead
    // after the loop — so rustc deleted the whole timed region and this
    // printed `elapsed: 0ns`, `per-call: 0ns`, satisfying the < 1s budget with
    // maximum margin on work that never ran. `best_of_us` FAILS loudly on an
    // all-zero reading; the gates are summed into a sink and both the sink and
    // the steered state are consumed through `black_box`. The bar below is
    // unchanged, and the total timed iteration count is still 100 000.
    const ROUNDS: usize = 10;
    const ITERS: usize = 10_000;

    let best_us = ab_timing::best_of_us(2, ROUNDS, || {
        let mut sink = 0f32;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            field.steer_local(&mut state, 0, &offset);
            field.steer_centroid(&mut state, 0, 0.1);
            sink += field.membership_gates(&state, 0.0)[0];
        }
        let e = t0.elapsed();
        std::hint::black_box(sink);
        std::hint::black_box(state);
        e
    });
    // Scale the best ROUND back to the 100 000-call budget the bar is written
    // against, so `elapsed` means exactly what it meant before.
    let elapsed = std::time::Duration::from_secs_f64(best_us * 1e-6 * (100_000.0 / ITERS as f64));
    println!("── G4b: 100k steer_local + steer_centroid + membership_gates ──");
    println!("   elapsed:     {elapsed:?}");
    println!("   per-call:    {:?}", elapsed / 100_000);
    // Generous budget: 100k iterations should complete in well under 1s.
    // (At HLA scale D=8 K=8 R=2, each call is ~500 FLOPs — plasma-tier.)
    assert!(
        elapsed.as_millis() < 1000,
        "G4b FAIL: 100k calls took {elapsed:?} (budget 1s)"
    );
    println!("   G4 ✓ PASS");
}

// ═══════════════════════════════════════════════════════════════════════════════
// G5 — Determinism (commitment + decompose + reconstruct bit-identical)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn g5_determinism() {
    const D: usize = 8;
    const K: usize = 4;
    const R: usize = 2;

    let centroids = [[0f32; D], [5f32; D], [10f32; D], [15f32; D]];
    let loadings = [identity_loadings::<D, R>(); K];
    let log_pi = [0f32; K];
    let psi_inv = [1f32; D];

    // G5a: commitment is deterministic for identical parameters.
    let c1 = compute_field_commitment(&centroids, &loadings, &log_pi, &psi_inv);
    let c2 = compute_field_commitment(&centroids, &loadings, &log_pi, &psi_inv);
    println!("── G5a: commitment determinism ──");
    assert_eq!(c1, c2, "G5a FAIL: commitment not deterministic");
    println!("   G5a ✓ PASS");

    // G5b: decompose + reconstruct bit-identical for fixed state + field.
    let field = RegionSubspaceField::<D, K, R>::new_unchecked(centroids, loadings, log_pi, psi_inv);
    let state = [1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32];
    let decomp1: RegionDecomposition<K, R> = field.decompose(&state, 0.0);
    let decomp2: RegionDecomposition<K, R> = field.decompose(&state, 0.0);
    let recon1 = reconstruct(&decomp1, &field);
    let recon2 = reconstruct(&decomp2, &field);

    println!("── G5b: decompose + reconstruct determinism ──");
    // Gates bit-identical.
    for k in 0..K {
        assert_eq!(
            decomp1.gates[k].to_bits(),
            decomp2.gates[k].to_bits(),
            "G5b FAIL: gate[{k}] not deterministic"
        );
    }
    // Local coords bit-identical.
    for k in 0..K {
        for r in 0..R {
            assert_eq!(
                decomp1.local_coords[k][r].to_bits(),
                decomp2.local_coords[k][r].to_bits(),
                "G5b FAIL: local_coords[{k}][{r}] not deterministic"
            );
        }
    }
    // Reconstruction bit-identical.
    for d in 0..D {
        assert_eq!(
            recon1[d].to_bits(),
            recon2[d].to_bits(),
            "G5b FAIL: recon[{d}] not deterministic"
        );
    }
    println!("   G5b ✓ PASS");
    println!("   G5 ✓ PASS");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn print_goat_summary() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 416 — Region-Conditioned Subspace Field GOAT Gate Summary  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  G1 (K=1 parity with Plan 412):  see g1_k1_parity_with_plan_412_100_random_offsets");
    println!("  G2 (two-mode steering):          see g2_two_mode_steering_distinct");
    println!("  G4 (latency + struct size):      see g4_latency_smoke_and_struct_size");
    println!("  G5 (determinism):                see g5_determinism");
    println!();
    println!("  G3 (zero-alloc) is verified by construction: all fields are");
    println!("  fixed-size arrays — no heap allocations possible. (Same argument");
    println!("  as Plan 412 Phase 4 T4.2.)");
}
