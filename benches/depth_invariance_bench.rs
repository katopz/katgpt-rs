//! Plan 306 Phase 6 — G4 latency benchmark for the depth-invariance
//! diagnostic and the magnitude-regularization fix primitive.
//!
//! # Gates
//!
//! - **T6.1** `classify_chain` latency vs one `LatentDynamicsMLP::forward_into`
//!   at matched `d`. Target: ≤5% of the forward pass time.
//! - **T6.2** `classify_chain_batched` throughput on 1000 chains at d=8, k=16.
//!   Target: ≥10M classifications/sec.
//! - **T6.3** `apply_magnitude_regularization` (RmsNorm + ScalarPinch) overhead
//!   vs unregularized residual write. Target: ≤2%.
//!
//! # Convention
//!
//! `harness = false` + `std::time::Instant` (criterion is not a katgpt-rs
//! dev-dep; see `benches/faithfulness_probe_bench.rs` doc-comment for the
//! established pattern). Run:
//!
//! ```bash
//! cargo bench --bench depth_invariance_bench --no-default-features \
//!     --features depth_invariance --features belief_drafter -p katgpt-rs
//! ```
//!
//! Source paper: arXiv:2605.09992 §3 / §4.4.
//! Research note: `.research/286_Attention_Drift_Depth_Invariance_Diagnostic.md`.

#![cfg(all(feature = "depth_invariance", feature = "belief_drafter"))]

use std::time::{Duration, Instant};

use katgpt_core::{
    DepthInvarianceConfig, MagnitudeRegularization, Scratch, apply_magnitude_regularization,
    classify_chain, classify_chain_batched,
};
use katgpt_rs::speculative::belief_drafter::{LatentDynamicsMLP, MlpForwardScratch};

// ── Helpers ──────────────────────────────────────────────────────────────

/// Deterministic xorshift64* PRNG.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 & 0xFFFF) as f32 / 0x8000 as f32) - 1.0
    }
}

/// Build a flattened `[k+1][d]` chain of pseudo-random values.
fn rand_chain(k_plus_1: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..k_plus_1 * d).map(|_| rng.next_f32()).collect()
}

/// Best-of-N wall-clock microseconds for a closure.
fn bench_us(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut best = Duration::from_secs(60);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed();
        if dt < best {
            best = dt;
        }
    }
    best.as_secs_f64() * 1e6
}

// ── T6.1: classify_chain vs LatentDynamicsMLP::forward_into ──────────────

fn bench_t6_1() {
    println!("## T6.1 — classify_chain vs LatentDynamicsMLP::forward_into\n");
    println!(
        "{:>6} {:>4} {:>14} {:>14} {:>10}",
        "d", "k", "classify_us", "forward_us", "ratio_%"
    );
    println!("{}", "-".repeat(60));

    let cfg = DepthInvarianceConfig::default();
    let d_values: &[usize] = &[8, 64, 256, 1024];
    let k_values: &[usize] = &[4, 16, 64];

    for &d in d_values {
        // Build a random-init MLP at this dimension for the forward baseline.
        let mlp = LatentDynamicsMLP::random_init(d);
        let mut fwd_scratch = MlpForwardScratch::new(d);
        let mut h_t = vec![0.5f32; d];
        let next_emb = vec![0.3f32; d];
        let mut h_out = vec![0.0f32; d];

        // Time one forward pass — best-of-N.
        let fwd_us = bench_us(5, 50, || {
            mlp.forward_into(&h_t, &next_emb, &mut fwd_scratch, &mut h_out);
            std::hint::black_box(&h_out);
            // Rotate inputs so we don't measure an unfair cache-warm path.
            std::mem::swap(&mut h_t, &mut h_out);
        });

        for &k in k_values {
            let k_plus_1 = k + 1;
            let chain = rand_chain(k_plus_1, d, 42 + (k * d) as u64);
            let mut scratch = Scratch::with_capacity(k_plus_1, d);

            let classify_us = bench_us(5, 50, || {
                let diag = classify_chain(&chain, d, &cfg, &mut scratch);
                std::hint::black_box(diag);
            });

            let ratio_pct = 100.0 * classify_us / fwd_us;
            println!(
                "{:>6} {:>4} {:>14.3} {:>14.3} {:>9.2}%",
                d, k, classify_us, fwd_us, ratio_pct
            );
        }
        println!();
    }

    println!(
        "Target: classify_chain ≤5% of forward_into time.\n\
         (Per Plan 306 §T6.1: aspirational on CPU/SIMD; report actuals.)\n"
    );
}

// ── T6.2: classify_chain_batched throughput ──────────────────────────────

fn bench_t6_2() {
    println!("## T6.2 — classify_chain_batched throughput (d=8, k=16, N=1000)\n");

    let d = 8usize;
    let k = 16usize;
    let n = 1000usize;
    let cfg = DepthInvarianceConfig::default();

    // Build N chains.
    let mut chains: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        chains.push(rand_chain(k + 1, d, i as u64 * 131 + 7));
    }
    let chain_refs: Vec<&[f32]> = chains.iter().map(|c| c.as_slice()).collect();

    let mut scratch = Scratch::with_capacity(k + 1, d);
    let mut out = Vec::with_capacity(n);

    let us = bench_us(3, 30, || {
        classify_chain_batched(&chain_refs, d, &cfg, &mut scratch, &mut out);
        std::hint::black_box(&out);
    });

    let throughput_per_sec = (n as f64) / (us * 1e-6);
    println!("  N={n} chains classified in {us:.3} µs");
    println!("  throughput: {throughput_per_sec:.3e} classifications/sec");
    println!(
        "  target: ≥1e7 classifications/sec — {}",
        if throughput_per_sec >= 1.0e7 {
            "PASS"
        } else {
            "below target (documented)"
        }
    );
    println!();
}

// ── T6.3: apply_magnitude_regularization overhead ────────────────────────

fn bench_t6_3() {
    println!("## T6.3 — apply_magnitude_regularization overhead vs raw residual write\n");
    println!(
        "{:>6} {:>16} {:>16} {:>16} {:>10}",
        "d", "raw_write_us", "rmsnorm_us", "scalarpinch_us", "overhead_%"
    );
    println!("{}", "-".repeat(70));

    let d_values: &[usize] = &[8, 64, 256, 1024];

    for &d in d_values {
        let h_a = vec![0.5f32; d];
        let delta = vec![0.05f32; d];
        let mut h_b = vec![0.0f32; d];
        let mut reg_scratch = vec![0.0f32; d];

        // Baseline: the realistic unregularized residual write is
        // `out[i] = h_t[i] + delta[i]` (NOT just a memcpy). The regularization
        // gate compares the regularized version against this.
        let raw_us = bench_us(5, 200, || {
            for i in 0..d {
                h_b[i] = h_a[i] + delta[i];
            }
            std::hint::black_box(&h_b);
        });

        // RmsNorm path: same residual write, then in-place RMS normalization.
        let rmsnorm_us = bench_us(5, 200, || {
            for i in 0..d {
                h_b[i] = h_a[i] + delta[i];
            }
            apply_magnitude_regularization(
                &mut h_b,
                MagnitudeRegularization::RmsNorm,
                &mut reg_scratch,
            );
            std::hint::black_box(&h_b);
        });

        // ScalarPinch path: same residual write, then in-place scalar pinch.
        // max_rms = 0.5 forces a scale step (our random h_a+delta has RMS ≈ 0.55).
        let pinch_us = bench_us(5, 200, || {
            for i in 0..d {
                h_b[i] = h_a[i] + delta[i];
            }
            apply_magnitude_regularization(
                &mut h_b,
                MagnitudeRegularization::ScalarPinch { max_rms: 0.5 },
                &mut reg_scratch,
            );
            std::hint::black_box(&h_b);
        });

        // Overhead = (regularized − raw) / raw. Report the worst of the two modes.
        let overhead_pct =
            100.0 * (rmsnorm_us.max(pinch_us) - raw_us) / raw_us;
        println!(
            "{:>6} {:>16.3} {:>16.3} {:>16.3} {:>9.2}%",
            d, raw_us, rmsnorm_us, pinch_us, overhead_pct
        );
    }

    println!();
    println!(
        "Target: ≤2% overhead vs raw residual write.\n\
         (Per Plan 306 §T6.3: aspirational; the regularization adds O(d) work vs the\n\
         raw O(d) copy, so overhead > 0 is expected. Report actuals.)\n"
    );
}

// ── main ─────────────────────────────────────────────────────────────────

fn main() {
    println!(
        "=== Depth-Invariance Diagnostic Benchmark (Plan 306 Phase 6) ===\n"
    );
    bench_t6_1();
    bench_t6_2();
    bench_t6_3();
}
