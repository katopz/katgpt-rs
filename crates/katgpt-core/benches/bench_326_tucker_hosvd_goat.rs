//! Tucker / HOSVD GOAT gate bench (Plan 326 Phase 1).
//!
//! Exercises G1–G4 for the `tucker_factorization` primitive.
//!
//! # Gates
//!
//! - **G1 (reconstruction quality — modelless correctness):** A synthetic
//!   rank-`(2,2,2)` tensor (outer product of 3 vectors with only the first 2
//!   entries nonzero) decomposed with ranks `(2,2,2)` must reconstruct with
//!   relative Frobenius error `< 1e-4`. This proves HOSVD recovers the exact
//!   low-rank structure when the rank budget matches the true rank.
//!
//! - **G2 (perf):** `tucker_decompose_into` + `tucker_reconstruct_into` on a
//!   `(8, 8, 8)` tensor with ranks `(4, 4, 4)`, mean latency over 1000 calls
//!   with pre-warmed scratch. **PASS** if mean ≤ 500µs (cold-tier archival
//!   budget; 3 small SVDs + 3 contractions on a 512-element tensor).
//!
//! - **G3 (no-regression):** Full-rank decomposition (`ranks = shape`) of a
//!   `(4, 4, 4)` tensor must reconstruct with max abs error `< 1e-4` — the
//!   decomposition is lossless up to f32 round-off when no truncation occurs.
//!
//! - **G4 (alloc-free hot path):** `tucker_decompose_into` with pre-warmed
//!   `TuckerScratch` + `TuckerResultScratch` allocates 0 times over 100
//!   steady-state calls (counted via a global `CountingAllocator`).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --bench bench_326_tucker_hosvd_goat -- --nocapture
//! ```

use katgpt_core::linalg::{
    TuckerConfig, TuckerResultScratch, TuckerScratch, tucker_decompose_into,
    tucker_reconstruct_into,
};
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── Helpers ────────────────────────────────────────────────────────────────

fn rel_frob_error(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..x.len() {
        let d = x[i] - y[i];
        num += d * d;
        den += x[i] * x[i];
    }
    if den < f32::EPSILON {
        return 0.0;
    }
    (num / den).sqrt()
}

fn max_abs_error(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

// ─── G1: reconstruction quality ─────────────────────────────────────────────

fn g1_reconstruction_quality() -> bool {
    // Build an exact rank-(2,2,2) tensor: X[i,j,k] = a[i]·b[j]·c[k] where only
    // the first 2 entries of a, b, c are nonzero. Every mode-n unfolding has
    // rank ≤ 2, so HOSVD with ranks (2,2,2) must reconstruct X nearly exactly.
    let shape = [4usize, 4, 4];
    let total = 64usize;
    let a = [1.0f32, 0.5, 0.0, 0.0];
    let b = [0.7f32, -0.3, 0.0, 0.0];
    let c = [0.4f32, 0.9, 0.0, 0.0];
    let mut x = vec![0.0f32; total];
    for (i0, &a0) in a.iter().enumerate().take(4) {
        for (i1, &b1) in b.iter().enumerate().take(4) {
            for (i2, &c2) in c.iter().enumerate().take(4) {
                let flat = (i0 * 4 + i1) * 4 + i2;
                x[flat] = a0 * b1 * c2;
            }
        }
    }

    let cfg = TuckerConfig::new(&shape, &[2, 2, 2]).expect("valid config");
    let mut scratch = TuckerScratch::with_capacity(&cfg);
    let mut result = TuckerResultScratch::with_capacity(&cfg);
    tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).expect("decompose");
    let mut recon = vec![0.0f32; total];
    tucker_reconstruct_into(&result, &shape, &mut recon, &mut scratch).expect("reconstruct");

    let err = rel_frob_error(&x, &recon);
    let pass = err < 1e-4;
    println!(
        "G1 (rank-(2,2,2) recovery): rel Frob error = {err:.3e} (target < 1e-4) → {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

// ─── G2: perf ───────────────────────────────────────────────────────────────

fn g2_perf() -> bool {
    let shape = [8usize, 8, 8];
    let total = 512usize;
    let x: Vec<f32> = (0..total)
        .map(|i| {
            let v = (i as f32) * 0.01 - 2.5;
            v.sin() + 0.5 * (2.0 * v).cos()
        })
        .collect();
    let cfg = TuckerConfig::new(&shape, &[4, 4, 4]).expect("valid config");
    let mut scratch = TuckerScratch::with_capacity(&cfg);
    let mut result = TuckerResultScratch::with_capacity(&cfg);
    let mut recon = vec![0.0f32; total];

    // Warmup (first call may touch cold caches / page in scratch buffers).
    for _ in 0..10 {
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).expect("decompose");
        tucker_reconstruct_into(&result, &shape, &mut recon, &mut scratch).expect("reconstruct");
    }

    let n_iters = 1000;
    let start = Instant::now();
    for _ in 0..n_iters {
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).expect("decompose");
        tucker_reconstruct_into(&result, &shape, &mut recon, &mut scratch).expect("reconstruct");
    }
    let elapsed = start.elapsed();
    let mean_us = elapsed.as_secs_f64() * 1e6 / (n_iters as f64);

    // Cold-tier archival budget. The dominant cost is 3 small SVDs + 3 tensor
    // contractions on a 512-element tensor. Target is generous to stay stable
    // across CI machines; the actual perf is typically much lower.
    let target_us = 500.0;
    let pass = mean_us <= target_us;
    println!(
        "G2 (perf, (8,8,8) ranks (4,4,4)): mean = {mean_us:.2}µs over {n_iters} iters (target ≤ {target_us}µs) → {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

// ─── G3: no-regression (full-rank is lossless) ──────────────────────────────

fn g3_no_regression() -> bool {
    let shape = [4usize, 4, 4];
    let total = 64usize;
    let x: Vec<f32> = (0..total)
        .map(|i| ((i as f32) * 0.5 - 15.0).sin())
        .collect();
    // Full rank: ranks = shape. Reconstruction must be near-identity (f32 round-off).
    let cfg = TuckerConfig::new(&shape, &shape).expect("valid config");
    let mut scratch = TuckerScratch::with_capacity(&cfg);
    let mut result = TuckerResultScratch::with_capacity(&cfg);
    tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).expect("decompose");
    let mut recon = vec![0.0f32; total];
    tucker_reconstruct_into(&result, &shape, &mut recon, &mut scratch).expect("reconstruct");

    let err = max_abs_error(&x, &recon);
    let pass = err < 1e-4;
    println!(
        "G3 (full-rank lossless): max abs error = {err:.3e} (target < 1e-4) → {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

// ─── G4: alloc-free hot path ────────────────────────────────────────────────

fn g4_alloc_free() -> bool {
    let shape = [8usize, 8, 8];
    let total = 512usize;
    let x: Vec<f32> = (0..total)
        .map(|i| {
            let v = (i as f32) * 0.01 - 2.5;
            v.sin() + 0.5 * (2.0 * v).cos()
        })
        .collect();
    let cfg = TuckerConfig::new(&shape, &[4, 4, 4]).expect("valid config");
    let mut scratch = TuckerScratch::with_capacity(&cfg);
    let mut result = TuckerResultScratch::with_capacity(&cfg);

    // Warmup: ensure all internal buffers are sized and any one-time setup is done.
    for _ in 0..5 {
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).expect("decompose");
    }

    // Measure: 100 steady-state decompose calls must allocate 0 times.
    let n_calls = 100;
    let (_, allocs) = alloc_delta(|| {
        for _ in 0..n_calls {
            tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).expect("decompose");
        }
    });

    let pass = allocs == 0;
    println!(
        "G4 (alloc-free hot path): {allocs} allocations over {n_calls} steady-state calls (target 0) → {}",
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        // Hint: the most likely culprit is `SvdScratch` / `SvdResultScratch`
        // auto-growing when the presented matrix exceeds `with_capacity`. Verify
        // that `TuckerScratch::with_capacity` sizes them for the worst-case mode.
        println!("  hint: check TuckerScratch::with_capacity SVD sizing vs actual call sizes");
    }
    pass
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 326 Phase 1 — Tucker/HOSVD GOAT gate ===\n");

    let g1 = g1_reconstruction_quality();
    let g2 = g2_perf();
    let g3 = g3_no_regression();
    let g4 = g4_alloc_free();

    println!();
    println!(
        "Verdict: G1={} G2={} G3={} G4={}",
        if g1 { "PASS" } else { "FAIL" },
        if g2 { "PASS" } else { "FAIL" },
        if g3 { "PASS" } else { "FAIL" },
        if g4 { "PASS" } else { "FAIL" }
    );

    let all_pass = g1 && g2 && g3 && g4;
    println!();
    if all_pass {
        println!("ALL GATES PASS — primitive is GOAT-validated.");
        println!("   tucker_factorization was PROMOTED TO DEFAULT-ON");
        println!("   (katgpt-core/Cargo.toml Phase 2, 2026-06). This bench was");
        println!("   the gate evidence; promotion has already landed.");
    } else {
        println!("ONE OR MORE GATES FAILED — note: tucker_factorization was previously");
        println!("promoted to default-on; a regression here would warrant re-audit.");
        std::process::exit(1);
    }
}
