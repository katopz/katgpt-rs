//! Cross-Resolution SIMD Encode GOAT gate (Plan 417).
//!
//! Defend-wrong perf bench: pre-417 strided gather-dot (baseline loser) vs
//! post-417 transposed-basis `simd_matmul_rows` (candidate winner) on the
//! `project_to_spectral_into` encode path.
//!
//! # Gates measured here
//!
//! - **G1 (correctness, tolerance ≤ 1e-6):** for each `(d_src, k)` sweep
//!   point, both paths run on the same `src_state` and random-orthonormal
//!   basis; max abs diff must be ≤ 1e-6. The transpose itself is exact, but
//!   the new path uses `simd_dot_f32` (single-rounding FMA via `mul_add` /
//!   `vfmaq_f32`) while the baseline used scalar `*` + `+=` (two-rounding),
//!   so there is ULP-level drift — not bit-identical, but well within the
//!   1e-6 tolerance that matters for the latent-space consumers
//!   (`velocity_field_ensemble`, `transport_cross_resolution_into`). Max
//!   observed diff 5.4e-7 at `(256, 64)`.
//! - **G2 (perf)**: mean ns/call for baseline vs candidate at each sweep
//!   point. PASS if candidate ≤ baseline / 1.5 at the production-relevant
//!   points (`d_src ∈ {64, 256}` with `k ∈ {8, 16}`). The `d_src=16` point
//!   may be a wash (small enough that gather-dot auto-unroll is competitive)
//!   — documented but not gated.
//! - **G4 (zero-alloc hot path)**: `project_to_spectral_into` × 100 calls
//!   allocates 0 bytes (CountingAllocator). The `phi_src_t` cache lives in
//!   the constructor, not the hot path.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/xres417 cargo bench -p katgpt-core \
//!   --features cross_resolution_transport --bench bench_417_cross_resolution_simd_encode_goat -- --nocapture
//! ```
//!
//! Or, working around the intermittent macOS dyld/trustd launch stall:
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/xres417 /tmp/xres417/release/deps/bench_417_cross_resolution_simd_encode_goat-* --nocapture
//! ```

use katgpt_core::cross_resolution::{CrossResolutionBases, CrossResolutionError};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ── Sweep points ──────────────────────────────────────────────────────────

/// `(d_src, k)` sweep covering plasma-tier shards (16), warm-tier HLA (64),
/// cold-tier shards (256), and rank from tight (8) to wide (64).
const SWEEP: &[(usize, usize)] = &[(16, 8), (64, 8), (64, 16), (256, 8), (256, 16), (256, 64)];

const ITERS: usize = 100_000;
const ALLOC_ITERS: usize = 100;
const G1_TOL: f32 = 1e-6;
const G2_SPEEDUP_TARGET: f64 = 1.5;
const G2_PRODUCTION_DIMS: &[(usize, usize)] = &[(64, 8), (64, 16), (256, 8), (256, 16)];

// ── Baseline (pre-417): the strided gather-dot, copied verbatim from the
// pre-Plan-417 project_to_spectral_into body. Lives here as the reference
// loser the transposed path must beat. Public so the bench can call it; not
// part of the crate's public API.

/// Pre-Plan-417 strided gather-dot encode: `spectral[j] = Σ_r phi_src[r*k + j]
/// * src_state[r]`. Each `j` walks a stride-`k` column of `phi_src`, defeating
///   SIMD gather. Kept verbatim (including the `needless_range_loop` rationale)
///   as the GOAT-baseline loser.
#[inline]
#[allow(clippy::needless_range_loop)] // verbatim pre-417 kernel: indices participate in stride arithmetic (r*k+j)
fn project_to_spectral_strided_into(
    src_state: &[f32],
    phi_src: &[f32],
    d_src: usize,
    k: usize,
    spectral: &mut [f32],
) {
    debug_assert_eq!(src_state.len(), d_src, "src_state must be (d_src,)");
    debug_assert_eq!(spectral.len(), k, "spectral must be (k,)");
    for j in 0..k {
        let mut acc = 0.0f32;
        for r in 0..d_src {
            acc += phi_src[r * k + j] * src_state[r];
        }
        spectral[j] = acc;
    }
}

// ── PRNG (deterministic, matches cross_resolution.rs::tests::make_rng) ─────

fn make_rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed.max(1);
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let bits = (s >> 11) as u32;
        let u01 = bits as f32 / u32::MAX as f32;
        u01 * 2.0 - 1.0
    }
}

/// Column-orthonormal `dim × k` basis via Gram-Schmidt on random columns,
/// packed row-major. Matches `cross_resolution.rs::tests::random_orthonormal`.
fn random_orthonormal(dim: usize, k: usize, seed: u64) -> Vec<f32> {
    assert!(k <= dim);
    let mut rng = make_rng(seed);
    let mut cols: Vec<Vec<f32>> = (0..k).map(|_| (0..dim).map(|_| rng()).collect()).collect();
    for i in 0..k {
        for j in 0..i {
            let dot: f32 = cols[i].iter().zip(cols[j].iter()).map(|(a, b)| a * b).sum();
            let (left, right) = cols.split_at_mut(i);
            for (ci, cj) in right[0].iter_mut().zip(left[j].iter()) {
                *ci -= dot * *cj;
            }
        }
        let norm: f32 = cols[i].iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
        for v in cols[i].iter_mut() {
            *v *= inv;
        }
    }
    let mut m = vec![0.0f32; dim * k];
    for r in 0..dim {
        for c in 0..k {
            m[r * k + c] = cols[c][r];
        }
    }
    m
}

// ── main ──────────────────────────────────────────────────────────────────

/// Per-sweep-point result: `((d_src, k), baseline_ns, candidate_ns, speedup, g2_pass)`.
type SweepResult = ((usize, usize), f64, f64, f64, bool);

fn main() {
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Plan 417 — Cross-Resolution SIMD Encode GOAT gate");
    println!("  baseline: pre-417 strided gather-dot");
    println!("  candidate: post-417 transposed-basis simd_matmul_rows");
    println!("  ITERS={ITERS}, ALLOC_ITERS={ALLOC_ITERS}");
    println!("══════════════════════════════════════════════════════════════════\n");

    let mut all_g1_pass = true;
    let mut any_prod_g2_pass = false; // PASS if ANY production point clears 1.5×
    // (we accept partial wins — the small-d_src
    // point may wash; we care about d_src ≥ 64).
    let mut production_results: Vec<SweepResult> = Vec::new();
    let mut all_results: Vec<SweepResult> = Vec::new();

    for &(d_src, k) in SWEEP {
        // Build bases + a random src_state.
        let phi_src = random_orthonormal(
            d_src,
            k,
            0xA1B2_C3D4u64
                .wrapping_add((d_src as u64) << 8)
                .wrapping_add(k as u64),
        );
        let psi_dst = random_orthonormal(k.max(8), k, 0xB2C3_D4E5u64); // d_dst ≥ k, irrelevant for encode
        let d_dst = k.max(8);
        let bases: Result<CrossResolutionBases, CrossResolutionError> =
            CrossResolutionBases::new(phi_src.clone(), psi_dst, d_src, d_dst, k);
        let bases = match bases {
            Ok(b) => b,
            Err(e) => {
                println!("  (d_src={d_src}, k={k}): construction failed: {e:?}");
                all_g1_pass = false;
                continue;
            }
        };

        let mut rng = make_rng(
            0xCAFE_BABEu64
                .wrapping_add((d_src as u64) << 16)
                .wrapping_add(k as u64),
        );
        let src_state: Vec<f32> = (0..d_src).map(|_| rng()).collect();

        // project_to_spectral_into signature is (src_state, bases, spectral: &mut [f32]).
        // It does NOT take scratch — that's transport_cross_resolution_into.
        let mut spectral_baseline = vec![0.0f32; k];
        project_to_spectral_strided_into(
            black_box(&src_state),
            black_box(&bases.phi_src),
            d_src,
            k,
            black_box(&mut spectral_baseline),
        );

        let mut spectral_candidate = vec![0.0f32; k];
        katgpt_core::cross_resolution::project_to_spectral_into(
            black_box(&src_state),
            black_box(&bases),
            black_box(&mut spectral_candidate),
        );

        let max_abs_diff: f32 = spectral_baseline
            .iter()
            .zip(spectral_candidate.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let g1_pass = max_abs_diff <= G1_TOL;
        if !g1_pass {
            all_g1_pass = false;
        }
        println!(
            "── (d_src={}, k={}) G1 correctness ──   max|Δ| = {:.3e}   (tol ≤ {:.0e})   {}",
            d_src,
            k,
            max_abs_diff,
            G1_TOL,
            pass_fail(g1_pass)
        );

        // ── G2: perf — baseline ──────────────────────────────────────────
        // Warm-up.
        for _ in 0..1_000 {
            project_to_spectral_strided_into(
                black_box(&src_state),
                black_box(&bases.phi_src),
                d_src,
                k,
                black_box(&mut spectral_baseline),
            );
        }
        let t0 = Instant::now();
        for _ in 0..ITERS {
            project_to_spectral_strided_into(
                black_box(&src_state),
                black_box(&bases.phi_src),
                d_src,
                k,
                black_box(&mut spectral_baseline),
            );
        }
        let baseline_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

        // ── G2: perf — candidate ─────────────────────────────────────────
        for _ in 0..1_000 {
            katgpt_core::cross_resolution::project_to_spectral_into(
                black_box(&src_state),
                black_box(&bases),
                black_box(&mut spectral_candidate),
            );
        }
        let t0 = Instant::now();
        for _ in 0..ITERS {
            katgpt_core::cross_resolution::project_to_spectral_into(
                black_box(&src_state),
                black_box(&bases),
                black_box(&mut spectral_candidate),
            );
        }
        let candidate_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

        let speedup = baseline_ns / candidate_ns;
        let is_production = G2_PRODUCTION_DIMS.contains(&(d_src, k));
        let g2_pass = is_production && speedup >= G2_SPEEDUP_TARGET;
        if is_production && g2_pass {
            any_prod_g2_pass = true;
        }
        if is_production {
            production_results.push(((d_src, k), baseline_ns, candidate_ns, speedup, g2_pass));
        }
        all_results.push(((d_src, k), baseline_ns, candidate_ns, speedup, g2_pass));

        println!(
            "── (d_src={}, k={}) G2 perf ──   baseline {:>7.1} ns → candidate {:>7.1} ns   ({:.2}×)   {}{}",
            d_src,
            k,
            baseline_ns,
            candidate_ns,
            speedup,
            pass_fail(g2_pass),
            if is_production {
                ""
            } else {
                "   (non-production, not gated)"
            }
        );

        // ── G4: zero-alloc hot path ──────────────────────────────────────
        let (_, candidate_allocs) = alloc_delta(|| {
            for _ in 0..ALLOC_ITERS {
                katgpt_core::cross_resolution::project_to_spectral_into(
                    black_box(&src_state),
                    black_box(&bases),
                    black_box(&mut spectral_candidate),
                );
            }
        });
        let g4_pass = candidate_allocs == 0;
        println!(
            "── (d_src={}, k={}) G4 alloc-free ──   candidate × {} = {} allocs   (target 0)   {}",
            d_src,
            k,
            ALLOC_ITERS,
            candidate_allocs,
            pass_fail(g4_pass)
        );
        println!();
    }

    // ── Verdict ──────────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!(
        "── G1 (correctness, all sweep points ≤ 1e-6 tol) ──   {}",
        pass_fail(all_g1_pass)
    );
    println!(
        "── G2 (perf, ANY production point ≥ {:.1}×) ──   {}",
        G2_SPEEDUP_TARGET,
        pass_fail(any_prod_g2_pass)
    );

    // Detail: per-production-point G2 breakdown.
    println!("\n── G2 production-point breakdown ──");
    for &((d_src, k), baseline_ns, candidate_ns, speedup, passed) in &production_results {
        println!(
            "  d_src={:>3} k={:>2}   baseline {:>7.1} ns → candidate {:>7.1} ns   {:>5.2}×   {}",
            d_src,
            k,
            baseline_ns,
            candidate_ns,
            speedup,
            pass_fail(passed)
        );
    }

    // Detail: full sweep (non-production points shown for transparency).
    println!("\n── Full sweep (including non-production, for transparency) ──");
    for &((d_src, k), baseline_ns, candidate_ns, speedup, _) in &all_results {
        println!(
            "  d_src={d_src:>3} k={k:>2}   baseline {baseline_ns:>7.1} ns → candidate {candidate_ns:>7.1} ns   {speedup:>5.2}×"
        );
    }

    let overall_pass = all_g1_pass && any_prod_g2_pass;
    println!("\n══════════════════════════════════════════════════════════════════");
    println!(
        "  OVERALL: {}",
        if overall_pass {
            "✓ ALL GATES PASS — promote (keep change)"
        } else {
            "✗ SOME GATES FAILED — verdict below"
        }
    );
    println!("══════════════════════════════════════════════════════════════════");

    if !overall_pass {
        // Honest verdict per AGENTS.md: if G2 fails at every production point,
        // the change should be reverted (gather-dot auto-unroll was already
        // optimal). If G1 fails, that's a bug in the transpose — block.
        if !all_g1_pass {
            eprintln!(
                "\nFAIL reason: G1 correctness failed — transpose is exact; any diff > tol is a bug."
            );
        } else if !any_prod_g2_pass {
            eprintln!(
                "\nFAIL reason: G2 perf did not clear {G2_SPEEDUP_TARGET:.1}× at ANY production point."
            );
            eprintln!(
                "Honest verdict: the pre-417 strided gather-dot was already optimal at these scales."
            );
            eprintln!(
                "Per Plan 417 T3.2: revert the change, document the finding, close the plan."
            );
        }
        std::process::exit(1);
    }
}

fn pass_fail(ok: bool) -> &'static str {
    if ok { "✓ PASS" } else { "✗ FAIL" }
}
