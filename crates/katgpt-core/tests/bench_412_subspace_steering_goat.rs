//! Plan 412 Phase 4 — Subspace Steering Field GOAT gate.
//!
//! Validates the k-dim manifold steering primitive across the five gates:
//!
//! | Gate | Target | Metric |
//! |------|--------|--------|
//! | G1 | K=1 parity with Plan 309 | bit-identical `apply` output vs `apply_latent_steering` across 100 random direction+alpha pairs |
//! | G2 | (covered by Phase 2 unit tests) | norm bounds + grid coverage |
//! | G3 | zero-alloc hot path | 0 heap allocations / 1000 steady-state `apply` calls at D=8, K={1,2,4} |
//! | G4 | latency | structural size proof + smoke (criterion bench is separate) |
//! | G5 | determinism | commitment deterministic; `walk_manifold` bit-identical for fixed alpha_grid |

use katgpt_core::latent_steering::{LatentSteeringVector, apply_latent_steering};
use katgpt_core::subspace_steering::{
    SubspaceSteeringField, apply_subspace_steering, compute_block_commitment, walk_manifold,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

// ─── Reproducible LCG (matches Plan 404/405 GOAT gates) ─────────────────────

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    /// Fill a `Vec<f32>` of length `d` with values in `[0, 1)`.
    #[inline]
    fn next_f32_vec(&mut self, d: usize) -> Vec<f32> {
        (0..d)
            .map(|_| {
                let bits = self.next_u64() >> 40; // top 24 bits → mantissa
                (bits as f32) / ((1u64 << 24) as f32)
            })
            .collect()
    }
    /// Normalize a Vec in place to unit norm.
    fn normalize_inplace(v: &mut [f32]) {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }
    /// Build a random unit-norm direction of length `d`.
    fn next_unit_direction(&mut self, d: usize) -> Vec<f32> {
        let mut v = self.next_f32_vec(d);
        // Avoid the zero vector (add 1.0 to each element).
        for x in &mut v {
            *x += 1.0;
        }
        Self::normalize_inplace(&mut v);
        v
    }
    /// Build a K-row orthonormal block via Gram-Schmidt from K random directions.
    fn next_orthonormal_block<const D: usize, const K: usize>(&mut self) -> [[f32; D]; K] {
        let mut block = [[0f32; D]; K];
        for k in 0..K {
            let dir = self.next_unit_direction(D);
            block[k].copy_from_slice(&dir);
            // Gram-Schmidt: subtract projections onto previous rows.
            for j in 0..k {
                let prev = block[j];
                let proj: f32 = block[k].iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                for d in 0..D {
                    block[k][d] -= proj * prev[d];
                }
            }
            // Renormalize.
            let norm = block[k].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut block[k] {
                    *x /= norm;
                }
            }
        }
        block
    }
}

// ─── G1: K=1 parity with Plan 309 (expanded to 100 random pairs) ────────────

/// **G1 (T4.5):** `SubspaceSteeringField<D, 1>` is bit-identical to Plan 309's
/// `apply_latent_steering` across 100 random direction+alpha pairs.
///
/// This is the load-bearing gate — it proves the k-dim generalization subsumes
/// the 1D case. The Phase 1 unit test (`k1_parity_with_plan_309`) checks one
/// pair; this gate checks 100 random pairs.
#[test]
fn g1_k1_parity_with_plan_309_100_pairs() {
    const D: usize = 8;
    let mut rng = Lcg::new(0x5885_7777);

    let mut mismatches = 0usize;
    for _ in 0..100 {
        let direction = rng.next_unit_direction(D);
        // Alpha in [0, 1] — use the full range (not just the <=0.3 hot-path caveat).
        let alpha = (rng.next_u64() as f32 / u64::MAX as f32).clamp(0.0, 1.0);

        // Plan 309 reference.
        let steering = LatentSteeringVector::new_unchecked(direction.clone(), alpha);

        // Plan 412 K=1 field from the same direction + alpha.
        let mut block = [[0f32; D]];
        block[0].copy_from_slice(&direction);
        let field = SubspaceSteeringField::<D, 1>::new(block, [alpha], 1e-5).unwrap();

        // Identical random starting state.
        let mut state_rng = Lcg::new(rng.next_u64());
        let mut state = [0f32; D];
        for x in &mut state {
            *x = (state_rng.next_u64() as f32 / u64::MAX as f32) * 2.0 - 1.0; // [-1, 1]
        }
        let mut s_ref = state;
        let mut s_412 = state;

        apply_latent_steering(&mut s_ref, &steering);
        field.apply(&mut s_412);

        // Bit-identical check.
        for j in 0..D {
            if s_ref[j].to_bits() != s_412[j].to_bits() {
                mismatches += 1;
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "G1 FAIL: {mismatches} element mismatches across 100 random K=1 parity pairs (expected bit-identical)"
    );
}

// ─── G3: zero-alloc hot path ────────────────────────────────────────────────

/// **G3 (T4.2):** `apply_subspace_steering` performs zero heap allocations
/// after warmup at D=8, K={1,2,4} (HLA scale). The field is all fixed-size
/// arrays — allocation is impossible by construction — but we verify with the
/// counting allocator to catch any regression (e.g. someone adding a `Vec`
/// field to the struct).
#[test]
fn g3_apply_zero_alloc_after_warmup() {
    // Build fields at K=1, K=2, K=4.
    const N_CALLS: usize = 1000;

    let mut rng = Lcg::new(2026);

    let block1 = rng.next_orthonormal_block::<8, 1>();
    let field1 = SubspaceSteeringField::<8, 1>::new(block1, [0.3], 1e-5).unwrap();

    let block2 = rng.next_orthonormal_block::<8, 2>();
    let field2 = SubspaceSteeringField::<8, 2>::new(block2, [0.3, 0.5], 1e-5).unwrap();

    let block4 = rng.next_orthonormal_block::<8, 4>();
    let field4 = SubspaceSteeringField::<8, 4>::new(block4, [0.1, 0.2, 0.3, 0.4], 1e-5).unwrap();

    // Warmup.
    let mut state = [0.1f32; 8];
    for _ in 0..50 {
        apply_subspace_steering(&mut state, &field1);
        apply_subspace_steering(&mut state, &field2);
        apply_subspace_steering(&mut state, &field4);
    }
    black_box(state);

    // Measure. NOTE: this test relies on the global counting allocator, so it
    // MUST be run with `--test-threads=1` to avoid parallel-test alloc noise.
    // The other alloc-check tests in this crate (conformal_alloc_check, etc.)
    // have the same constraint.
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..N_CALLS {
        apply_subspace_steering(&mut state, &field1);
        apply_subspace_steering(&mut state, &field2);
        apply_subspace_steering(&mut state, &field4);
    }
    black_box(state);

    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;

    assert_eq!(
        alloc_delta, 0,
        "G3 FAIL: apply_subspace_steering allocated {alloc_delta} times in {N_CALLS} calls × 3 fields (K=1,2,4)"
    );
    assert_eq!(
        dealloc_delta, 0,
        "G3 FAIL: apply_subspace_steering deallocated {dealloc_delta} times in {N_CALLS} calls × 3 fields"
    );
}

// ─── G4: structural size + latency smoke ────────────────────────────────────

/// **G4 (T4.3):** Structural size proof + latency smoke check.
///
/// The field is `K*D*4 + K*4 + 32` bytes (block + alphas + commitment). We
/// verify the size is exactly as expected for D=8, K={1,2,4}, then run a
/// latency smoke (full criterion bench is `benches/subspace_steering_bench.rs`).
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only timing gate (debug ~10x + machine-marginal under load; the precise gate is the criterion bench)"
)]
#[allow(clippy::identity_op)] // K=1 kept literal to mirror the K*D*4 + K*4 + 32 formula.
fn g4_structural_size_and_latency_smoke() {
    // D=8, K=1: 1*8*4 + 1*4 + 32 = 68 bytes.
    assert_eq!(
        std::mem::size_of::<SubspaceSteeringField<8, 1>>(),
        1 * 8 * 4 + 1 * 4 + 32,
        "G4: SubspaceSteeringField<8,1> size mismatch"
    );
    // D=8, K=2: 2*8*4 + 2*4 + 32 = 104 bytes.
    assert_eq!(
        std::mem::size_of::<SubspaceSteeringField<8, 2>>(),
        2 * 8 * 4 + 2 * 4 + 32,
        "G4: SubspaceSteeringField<8,2> size mismatch"
    );
    // D=8, K=4: 4*8*4 + 4*4 + 32 = 176 bytes.
    assert_eq!(
        std::mem::size_of::<SubspaceSteeringField<8, 4>>(),
        4 * 8 * 4 + 4 * 4 + 32,
        "G4: SubspaceSteeringField<8,4> size mismatch"
    );

    // Latency smoke: 100k applies must complete in well under a second at
    // HLA scale (K=4, D=8). This is a smoke check, not a precise benchmark.
    let mut rng = Lcg::new(412);
    let block = rng.next_orthonormal_block::<8, 4>();
    let field = SubspaceSteeringField::<8, 4>::new(block, [0.1, 0.2, 0.3, 0.4], 1e-5).unwrap();
    let mut state = [0.1f32; 8];

    let start = std::time::Instant::now();
    for _ in 0..100_000 {
        apply_subspace_steering(&mut state, &field);
    }
    let elapsed = start.elapsed();
    black_box(state);

    // 100k applies at K=4, D=8. Target per Plan 412: K=4 < 400ns.
    // Total should be < 40ms. We allow 10× headroom (400ms) for the smoke check
    // — the precise gate is the criterion bench.
    // ⚠ PRINTED, not only asserted (Issue 855 T5/T6). This arm's quantity lived
    // solely inside the `assert!` message — visible only on FAILURE — and
    // Issue 855 T3 had to plant a temporary `eprintln!` probe to read it at
    // all. That slice measured 4x the population's VANISHED rate, and two of
    // this repo's own arms were dead the first time they were made to print.
    println!(
        "G4 latency smoke: {:.2} ns/call over 100k applies (total {elapsed:?})",
        elapsed.as_nanos() as f64 / 100_000.0
    );
    assert!(
        elapsed.as_millis() < 400,
        "G4 latency smoke: 100k K=4 applies took {elapsed:?} (target < 400ms = 4µs/call headroom)",
    );
}

// ─── G5: determinism ────────────────────────────────────────────────────────

/// **G5 (T4.4):** `compute_block_commitment` is deterministic (same block +
/// alphas → same BLAKE3 across runs), and `walk_manifold` is bit-identical for
/// a fixed alpha_grid (quorum-safe).
#[test]
fn g5_commitment_and_walk_are_deterministic() {
    const D: usize = 8;
    let mut rng = Lcg::new(0x60A7_0505);
    let block = rng.next_orthonormal_block::<D, 2>();
    let alphas = [0.3f32, 0.7];

    // Commitment determinism.
    let h1 = compute_block_commitment(&block, &alphas);
    let h2 = compute_block_commitment(&block, &alphas);
    assert_eq!(h1, h2, "G5: commitment must be deterministic");
    assert_ne!(h1, [0u8; 32], "G5: commitment must be non-trivial");

    // walk_manifold determinism.
    let state = [0.1f32; D];
    let alpha_grid = [[0.0f32, 0.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]];
    let mut out1 = [[0f32; D]; 4];
    let mut out2 = [[0f32; D]; 4];
    walk_manifold(&state, &block, &alpha_grid, &mut out1);
    walk_manifold(&state, &block, &alpha_grid, &mut out2);
    assert_eq!(
        out1, out2,
        "G5: walk_manifold must be bit-identical for fixed alpha_grid"
    );
}

fn main() {
    let mut all_pass = true;

    let g1 = std::panic::catch_unwind(g1_k1_parity_with_plan_309_100_pairs);
    let g1_pass = g1.is_ok();
    println!(
        "  G1 K=1 parity (100 random pairs, bit-identical):    {}",
        if g1_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    all_pass &= g1_pass;

    let g3 = std::panic::catch_unwind(g3_apply_zero_alloc_after_warmup);
    let g3_pass = g3.is_ok();
    println!(
        "  G3 zero-alloc (1000 calls × K=1,2,4):               {}",
        if g3_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    all_pass &= g3_pass;

    let g4 = std::panic::catch_unwind(g4_structural_size_and_latency_smoke);
    let g4_pass = g4.is_ok();
    println!(
        "  G4 structural size + latency smoke (D=8 K=1,2,4):   {}",
        if g4_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    all_pass &= g4_pass;

    let g5 = std::panic::catch_unwind(g5_commitment_and_walk_are_deterministic);
    let g5_pass = g5.is_ok();
    println!(
        "  G5 determinism (commitment + walk_manifold):        {}",
        if g5_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    all_pass &= g5_pass;

    println!();
    println!(
        "  ─── Plan 412 Phase 4 GOAT verdict: {} ───",
        if all_pass { "ALL PASS ✅" } else { "FAIL ❌" }
    );

    if !all_pass {
        std::process::exit(1);
    }
}
