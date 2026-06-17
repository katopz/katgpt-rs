//! Sink-Aware Attention dual-policy latency benchmark (Plan 287 Phase 3, T3.5).
//!
//! Compares `apply_dual_policy_gate` with `SinkAwarePolicy::Uniform` vs
//! `SinkAwarePolicy::DualPolicy` at `n ∈ {128, 512}`, `d_h = 64`.
//! Plan target: ≤5% overhead for DualPolicy vs Uniform.
//!
//! Uses `std::time::Instant` (NOT criterion — matches other katgpt-rs benches).
//!
//! Run:
//! ```bash
//! cargo run --release --bench sink_aware_latency_bench --features sink_aware_attn
//! ```

#![cfg(feature = "sink_aware_attn")]

use katgpt_rs::data_probe::sink_classify::{
    SinkAwarePolicy, SinkClassifierConfig, StableRankScratch, apply_dual_policy_gate,
};
use std::time::{Duration, Instant};

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 & 0xFFFF) as f32 / 0x8000 as f32) - 1.0
    }
}

fn rand_matrix(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.next_f32()).collect())
        .collect()
}

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

fn main() {
    println!("=== Sink-Aware Dual-Policy Latency Benchmark (Plan 287 T3.5) ===\n");

    let d = 64usize;
    let n_values: &[usize] = &[128, 512];

    println!(
        "{:>5} {:>14} {:>14} {:>10} {:>10}",
        "n", "uniform_us", "dual_us", "overhead%", "kind"
    );
    println!("{}", "-".repeat(60));

    let cfg = SinkClassifierConfig::default();
    let mut scratch = StableRankScratch::new(d);

    for &n in n_values {
        // Use a rank-1 O so DualPolicy classifies as Broadcast (fast early-exit).
        // This matches the common paper case (Broadcast heads are the
        // fast path; the slow random-O case is covered by Phase 2 bench).
        let v_s: Vec<f32> = (0..d).map(|i| 0.1 * (i as f32).sin() + 0.5).collect();
        let values: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let o: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let mut out_uniform: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
        let mut out_dual: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();

        // Build an attention map with a dominant sink column at pos 0.
        let mut attn: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = vec![0.1 / (n as f32 - 1.0); n];
            row[0] = 0.9; // dominant sink
            // Renormalize row so it sums to ~1.0 (skip for raw mass).
            let _ = i;
            attn.push(row);
        }

        let policy_uniform = SinkAwarePolicy::Uniform;
        let policy_dual = SinkAwarePolicy::DualPolicy(cfg);

        let us_uniform = bench_us(3, 30, || {
            let kind = apply_dual_policy_gate(
                &attn,
                &values,
                &o,
                &policy_uniform,
                0.0,
                &mut scratch,
                &mut out_uniform,
            );
            std::hint::black_box(kind);
        });

        let us_dual = bench_us(3, 30, || {
            let kind = apply_dual_policy_gate(
                &attn,
                &values,
                &o,
                &policy_dual,
                0.0,
                &mut scratch,
                &mut out_dual,
            );
            std::hint::black_box(kind);
        });

        let overhead = if us_uniform > 0.0 {
            100.0 * (us_dual - us_uniform) / us_uniform
        } else {
            0.0
        };

        // Final kind for display.
        let kind = apply_dual_policy_gate(
            &attn,
            &values,
            &o,
            &policy_dual,
            0.0,
            &mut scratch,
            &mut out_dual,
        );

        println!(
            "{:>5} {:>14.3} {:>14.3} {:>9.2}% {:>10?}",
            n, us_uniform, us_dual, overhead, kind
        );
    }
    println!();
    println!("G3 target: overhead ≤5% (DualPolicy vs Uniform).");
}
