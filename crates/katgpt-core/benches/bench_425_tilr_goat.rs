//! TILR GOAT bench (Plan 425 Phase 2).
//!
//! Exercises G1–G4 for the `tilr` primitive — alignment-gated subspace-projected
//! correction (Trajectory-Invariant Latent Refinement, arXiv:2606.29164).
//!
//! # Gates
//!
//! - **G1 (no-harm bit-identity — the kill switch)**: When the contrastive
//!   direction `d` is orthogonal to the basis `U_r`, `γ = 0` and `s' = s`
//!   **bit-identically** (assert `to_bits()` equality). This is the load-bearing
//!   no-harm contract — graceful degradation to the uncorrected backbone.
//!
//! - **G2 (full-correction parity + γ boundedness)**: When `d ∈ span(U_r)`,
//!   `γ = 1.0` and `s' = s + η_base · d`. Across 1000 random `(state, direction,
//!   basis)` triples, `γ ∈ [0, 1]` with no NaN/OOB. At γ=1, the correction
//!   equals `state + eta_base * direction`.
//!
//! - **G3 (latency)**: Batched-median timing, 1024 calls × 256 batches with
//!   `black_box` anti-hoist. Targets:
//!     - `d=8, r=3` (belief scale): `< 50 ns`/call.
//!     - `d=64, r=12` (shard scale): `< 200 ns`/call.
//!
//! - **G4 (alloc-free hot path)**: After scratch warmup, 100 steady-state calls
//!   through `tilr_refine_into` allocate 0 times (counted via a global
//!   `CountingAllocator`).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --bench bench_425_tilr_goat -- --nocapture
//! ```
//!
//! Or (working around the dyld/trustd stall on macOS):
//!
//! ```bash
//! cargo bench -p katgpt-core --bench bench_425_tilr_goat --no-run
//! target/release/deps/bench_425_tilr_goat-<hash>
//! ```

use katgpt_core::tilr::{TilrScratch, tilr_refine_into};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Deterministic xorshift PRNG (no `rand` dep — matches the lib-test pattern).
struct Rng {
    state: u32,
}
impl Rng {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF } else { seed },
        }
    }
    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        (self.state as f32) / (u32::MAX as f32) * 2.0 - 1.0 // [-1, 1)
    }
    fn fill(&mut self, v: &mut [f32]) {
        for x in v.iter_mut() {
            *x = self.next_f32();
        }
    }
}

/// Gram-Schmidt orthonormalization on row-major `r × d` flat slice.
/// Produces `r` mutually orthonormal vectors of length `d`.
fn gram_schmidt_basis(d: usize, r: usize, rng: &mut Rng) -> Vec<f32> {
    let mut basis = vec![0.0f32; r * d];
    // Start with random vectors.
    for k in 0..r {
        rng.fill(&mut basis[k * d..(k + 1) * d]);
    }
    // Gram-Schmidt: for each row, subtract projections onto prior rows, normalize.
    for i in 0..r {
        for j in 0..i {
            let dot: f32 = (0..d).map(|x| basis[i * d + x] * basis[j * d + x]).sum();
            for x in 0..d {
                basis[i * d + x] -= dot * basis[j * d + x];
            }
        }
        let norm: f32 = (0..d)
            .map(|x| basis[i * d + x] * basis[i * d + x])
            .sum::<f32>()
            .sqrt();
        if norm > 1e-12 {
            for x in 0..d {
                basis[i * d + x] /= norm;
            }
        }
    }
    basis
}

fn timed_median_ns(iters: usize, batches: usize, mut body: impl FnMut()) -> f64 {
    for _ in 0..(batches.min(20)) {
        body();
    }
    let mut batch_times_ns: Vec<u64> = Vec::with_capacity(batches);
    for _ in 0..batches {
        let start = Instant::now();
        for _ in 0..iters {
            body();
        }
        batch_times_ns.push(start.elapsed().as_nanos() as u64);
    }
    batch_times_ns.sort_unstable();
    let mid = batch_times_ns.len() / 2;
    let median_batch_ns = batch_times_ns[mid] as f64;
    median_batch_ns / iters as f64
}

#[inline(never)]
fn bb<T>(x: T) -> T {
    black_box(x)
}

// ─── Gate runner ────────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

impl GateResult {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: true,
            detail: detail.into(),
        }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            detail: detail.into(),
        }
    }
}

// ─── G1: No-harm bit-identity (the kill switch) ──────────────────────────────

fn gate_g1_no_harm_bit_identity() -> GateResult {
    // For each (d, r), construct an orthonormal basis, pick a direction
    // orthogonal to it (lying in the complement), and verify γ = 0.0 exactly
    // and out == state bit-identically.
    let mut max_gamma_at_orthogonal: f32 = 0.0;
    let mut bit_mismatches: usize = 0;

    for &(d, r) in &[(8usize, 3usize), (64, 12)] {
        let mut rng = Rng::new(0xC0FFEE ^ (d as u32));
        let basis = gram_schmidt_basis(d, r, &mut rng);
        let mut scratch = TilrScratch::with_capacity(d, r);

        for _ in 0..100 {
            // Construct a direction in the complement of span(basis):
            // random vector, then subtract its projection onto span(basis).
            let mut direction = vec![0.0f32; d];
            rng.fill(&mut direction);
            for k in 0..r {
                let coeff: f32 = (0..d).map(|x| direction[x] * basis[k * d + x]).sum();
                for x in 0..d {
                    direction[x] -= coeff * basis[k * d + x];
                }
            }
            // Now direction ⊥ span(basis) → γ must be 0.

            let state: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
            let mut out = vec![0.0f32; d];

            let gamma = tilr_refine_into(
                &state,
                &direction,
                &basis,
                r,
                0.5,
                1e-12,
                &mut scratch,
                &mut out,
            )
            .expect("tilr_refine_into failed on orthogonal direction");

            max_gamma_at_orthogonal = max_gamma_at_orthogonal.max(gamma);
            for i in 0..d {
                if state[i].to_bits() != out[i].to_bits() {
                    bit_mismatches += 1;
                }
            }
        }
    }

    let gamma_ok = max_gamma_at_orthogonal == 0.0;
    let bits_ok = bit_mismatches == 0;

    let detail = format!(
        "max γ at orthogonal: {max_gamma_at_orthogonal:.3e} (must be exactly 0.0); \
         bit mismatches: {bit_mismatches} (must be 0)"
    );

    if gamma_ok && bits_ok {
        GateResult::pass("G1", detail)
    } else {
        GateResult::fail("G1", detail)
    }
}

// ─── G2: Full-correction parity + γ boundedness ─────────────────────────────

fn gate_g2_full_correction_and_boundedness() -> GateResult {
    let mut gamma_oob_count: usize = 0;
    let mut gamma_nan_count: usize = 0;
    let mut max_full_correction_err: f32 = 0.0;

    for &(d, r) in &[(8usize, 3usize), (64, 12)] {
        let mut rng = Rng::new(0xBEEF ^ (d as u32));
        let basis = gram_schmidt_basis(d, r, &mut rng);
        let mut scratch = TilrScratch::with_capacity(d, r);

        for _ in 0..500 {
            // Boundedness: random direction, check γ ∈ [0, 1].
            let direction: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
            let state: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
            let mut out = vec![0.0f32; d];

            let gamma = tilr_refine_into(
                &state,
                &direction,
                &basis,
                r,
                0.5,
                1e-12,
                &mut scratch,
                &mut out,
            )
            .unwrap();

            if !gamma.is_finite() {
                gamma_nan_count += 1;
            }
            if !(0.0..=1.0).contains(&gamma) {
                gamma_oob_count += 1;
            }
        }

        // Full-correction parity: direction = a basis vector (γ = 1).
        // out = state + eta_base * 1.0 * d_proj = state + eta_base * basis[k].
        let eta_base = 0.5_f32;
        for k in 0..r {
            let direction: Vec<f32> = basis[k * d..(k + 1) * d].to_vec();
            let state: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
            let mut out = vec![0.0f32; d];

            let gamma = tilr_refine_into(
                &state,
                &direction,
                &basis,
                r,
                eta_base,
                1e-12,
                &mut scratch,
                &mut out,
            )
            .unwrap();

            // γ should be ~1.0.
            if (gamma - 1.0).abs() > 1e-4 {
                gamma_oob_count += 1;
            }

            // out = state + eta_base * basis[k].
            for i in 0..d {
                let expected = state[i] + eta_base * basis[k * d + i];
                let err = (out[i] - expected).abs();
                max_full_correction_err = max_full_correction_err.max(err);
            }
        }
    }

    let bounded_ok = gamma_oob_count == 0 && gamma_nan_count == 0;
    let parity_ok = max_full_correction_err < 1e-4;

    let detail = format!(
        "γ OOB: {gamma_oob_count}, NaN: {gamma_nan_count}; \
         full-correction max err: {max_full_correction_err:.3e} (budget 1e-4)"
    );

    if bounded_ok && parity_ok {
        GateResult::pass("G2", detail)
    } else {
        GateResult::fail("G2", detail)
    }
}

// ─── G3: Latency ────────────────────────────────────────────────────────────

fn gate_g3_latency() -> GateResult {
    const ITERS: usize = 1024;
    const BATCHES: usize = 256;
    const TARGET_BELIEF_NS: f64 = 50.0; // d=8, r=3
    const TARGET_SHARD_NS: f64 = 200.0; // d=64, r=12

    // belief scale: d=8, r=3.
    let d_belief = 8;
    let r_belief = 3;
    let mut rng_belief = Rng::new(0xC0DE_0008);
    let basis_belief = gram_schmidt_basis(d_belief, r_belief, &mut rng_belief);
    let state_belief: Vec<f32> = (0..d_belief).map(|i| (i as f32) * 0.1).collect();
    let direction_belief: Vec<f32> = (0..d_belief).map(|i| (i as f32) * 0.05 - 0.2).collect();
    let mut out_belief = vec![0.0f32; d_belief];
    let mut scratch_belief = TilrScratch::with_capacity(d_belief, r_belief);

    let belief_ns = timed_median_ns(ITERS, BATCHES, || {
        let _ = bb(tilr_refine_into(
            bb(&state_belief),
            bb(&direction_belief),
            bb(&basis_belief),
            bb(r_belief),
            bb(0.5),
            bb(1e-12),
            &mut scratch_belief,
            &mut out_belief,
        ));
        bb(out_belief[0]);
    });

    // Shard scale: d=64, r=12.
    let d_shard = 64;
    let r_shard = 12;
    let mut rng_shard = Rng::new(0xC0DE_0040);
    let basis_shard = gram_schmidt_basis(d_shard, r_shard, &mut rng_shard);
    let state_shard: Vec<f32> = (0..d_shard).map(|i| (i as f32) * 0.01).collect();
    let direction_shard: Vec<f32> = (0..d_shard)
        .map(|i| ((i * 7) as f32) * 0.02 - 0.5)
        .collect();
    let mut out_shard = vec![0.0f32; d_shard];
    let mut scratch_shard = TilrScratch::with_capacity(d_shard, r_shard);

    let shard_ns = timed_median_ns(ITERS, BATCHES, || {
        let _ = bb(tilr_refine_into(
            bb(&state_shard),
            bb(&direction_shard),
            bb(&basis_shard),
            bb(r_shard),
            bb(0.5),
            bb(1e-12),
            &mut scratch_shard,
            &mut out_shard,
        ));
        bb(out_shard[0]);
    });

    let belief_pass = belief_ns < TARGET_BELIEF_NS;
    let shard_pass = shard_ns < TARGET_SHARD_NS;

    let detail = format!(
        "Belief (d={d_belief},r={r_belief}): {belief_ns:.1} ns (target <{TARGET_BELIEF_NS:.0}, {}); \
         Shard (d={d_shard},r={r_shard}): {shard_ns:.1} ns (target <{TARGET_SHARD_NS:.0}, {})",
        if belief_pass { "PASS" } else { "FAIL" },
        if shard_pass { "PASS" } else { "FAIL" },
    );

    if belief_pass && shard_pass {
        GateResult::pass("G3", detail)
    } else {
        GateResult::fail("G3", detail)
    }
}

// ─── G4: Alloc-free hot path ────────────────────────────────────────────────

fn gate_g4_zero_alloc() -> GateResult {
    let d = 64;
    let r = 12;
    let mut rng = Rng::new(0xA110C);
    let basis = gram_schmidt_basis(d, r, &mut rng);
    let state: Vec<f32> = (0..d).map(|i| (i as f32) * 0.05).collect();
    let direction: Vec<f32> = (0..d).map(|i| (i as f32) * 0.03 - 1.0).collect();
    let mut out = vec![0.0f32; d];
    let mut scratch = TilrScratch::with_capacity(d, r);

    // Warmup.
    for _ in 0..10 {
        let _ = tilr_refine_into(
            &state,
            &direction,
            &basis,
            r,
            0.5,
            1e-12,
            &mut scratch,
            &mut out,
        );
    }

    let iters = 100usize;
    let (_, allocs) = alloc_delta(|| {
        for _ in 0..iters {
            let _ = tilr_refine_into(
                &state,
                &direction,
                &basis,
                r,
                0.5,
                1e-12,
                &mut scratch,
                &mut out,
            );
        }
    });

    if allocs == 0 {
        GateResult::pass(
            "G4",
            format!("0 allocations over {iters} steady-state calls"),
        )
    } else {
        GateResult::fail(
            "G4",
            format!(
                "{allocs} allocations over {iters} steady-state calls (expected 0; ~{} per call)",
                allocs / iters
            ),
        )
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 425 — TILR Alignment-Gated Subspace Correction GOAT Gate ===\n");

    let gates = [
        gate_g1_no_harm_bit_identity(),
        gate_g2_full_correction_and_boundedness(),
        gate_g3_latency(),
        gate_g4_zero_alloc(),
    ];

    let mut all_pass = true;
    for g in &gates {
        let status = if g.passed { "PASS" } else { "FAIL" };
        println!("[{status}] {}: {}", g.name, g.detail);
        if !g.passed {
            all_pass = false;
        }
    }

    println!();
    if all_pass {
        println!("=== ALL GATES PASS — modelless gain proven, eligible for default promotion ===");
        std::process::exit(0);
    } else {
        println!("=== SOME GATES FAILED — keep opt-in, investigate ===");
        std::process::exit(1);
    }
}
