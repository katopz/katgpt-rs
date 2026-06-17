//! Sink-Aware forward-path composition latency benchmark (Plan 289 T3.5).
//!
//! Measures the G3 gate for `tiled_attention_parallax_forward_sink_aware`:
//! the overhead of the wrapper itself vs vanilla `tiled_attention_parallax_forward`.
//!
//! # Contracts verified
//!
//! - **Uniform short-circuit overhead (primary G3 gate)**: the wrapper must add
//!   ≤5% latency when `policy = Uniform` vs calling vanilla forward directly.
//!   This is the zero-cost-abstraction contract for callers who construct the
//!   sink-aware scratch but never enable DualPolicy.
//! - **Uniform vs DualPolicy gap (reference)**: confirms the cost difference
//!   between the short-circuit path and the full DualPolicy composition.
//!   Plan 287 G3 established the DualPolicy absolute cost; here we only verify
//!   the wrapper isn't adding surprise overhead beyond the known gate cost.
//!
//! # Methodology
//!
//! `std::time::Instant`, best-of-N (matches other katgpt-rs benches — no
//! criterion dep). n ∈ {64, 128, 256}, d_h = 64, gate_scale = 0.0 (no parallax
//! correction; pure attention forward so the sink-aware path's overhead is the
//! only variable).
//!
//! Run:
//! ```bash
//! cargo run --release --bench sink_aware_forward_bench --features parallax_attn,sink_aware_attn
//! ```

#![cfg(all(feature = "parallax_attn", feature = "sink_aware_attn"))]

use katgpt_core::data_probe::{
    SinkAwarePolicy, SinkClassifierConfig,
};
use katgpt_core::parallax_attn::{
    ParallaxActivation, ParallaxConfig, ParallaxScratch, SinkAwareParallaxScratch,
    tiled_attention_parallax_forward, tiled_attention_parallax_forward_sink_aware,
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

fn rand_buf(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..len).map(|_| rng.next_f32()).collect()
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

fn pct(measured: f64, baseline: f64) -> f64 {
    if baseline > 0.0 {
        100.0 * (measured - baseline) / baseline
    } else {
        0.0
    }
}

fn run_case(n: usize, d: usize) {
    let scale = 1.0 / (d as f32).sqrt();
    let q = rand_buf(n * d, 0xABCDEF01);
    let k = rand_buf(n * d, 0x12345678);
    let v = rand_buf(n * d, 0x9ABCDEF0);

    // gate_scale = 0 → parallax correction skipped, pure attention forward.
    // This isolates the sink-aware wrapper overhead from the parallax math.
    let cfg = ParallaxConfig {
        gate_scale: 0.0,
        zero_init: true,
        activation: ParallaxActivation::Sigmoid,
    };
    let r = vec![0.0f32; d * d];
    let x = vec![0.0f32; d];

    let policy_uniform = SinkAwarePolicy::Uniform;
    let policy_dual = SinkAwarePolicy::DualPolicy(SinkClassifierConfig::default());

    // ── Vanilla forward (baseline) ─────────────────────────────────
    let mut out_vanilla = vec![0.0f32; n * d];
    let mut pscratch = ParallaxScratch::new(n, d);
    let us_vanilla = bench_us(5, 50, || {
        tiled_attention_parallax_forward(
            &q, &k, &v, &mut out_vanilla, n, d, scale, &r, &x, &cfg,
            Some(&mut pscratch),
        );
        std::hint::black_box(out_vanilla[0]);
    });

    // ── Sink-aware forward, Uniform policy (PRIMARY G3 GATE) ───────
    let mut out_uniform = vec![0.0f32; n * d];
    let mut sscratch_uniform = SinkAwareParallaxScratch::new(n, d);
    let mut pscratch_u = ParallaxScratch::new(n, d);
    let us_sa_uniform = bench_us(5, 50, || {
        let kind = tiled_attention_parallax_forward_sink_aware(
            &q, &k, &v, &mut out_uniform, n, d, scale, &r, &x, &cfg,
            &policy_uniform, 0.0, &mut sscratch_uniform, Some(&mut pscratch_u),
        );
        std::hint::black_box((out_uniform[0], kind));
    });

    // ── Sink-aware forward, DualPolicy (per-call) ──────────────────
    let mut out_dual = vec![0.0f32; n * d];
    let mut sscratch_dual = SinkAwareParallaxScratch::new(n, d);
    let mut pscratch_d = ParallaxScratch::new(n, d);
    let us_sa_dual = bench_us(5, 50, || {
        let kind = tiled_attention_parallax_forward_sink_aware(
            &q, &k, &v, &mut out_dual, n, d, scale, &r, &x, &cfg,
            &policy_dual, -2.0, &mut sscratch_dual, Some(&mut pscratch_d),
        );
        std::hint::black_box((out_dual[0], kind));
    });

    // ── Sink-aware forward, DualPolicy (cached, cadence 16) ────────
    let mut out_cached = vec![0.0f32; n * d];
    let mut sscratch_cached = SinkAwareParallaxScratch::new(n, d).with_cache();
    if let Some(c) = sscratch_cached.cached.as_mut() {
        c.audit_every_n = 16;
    }
    let mut pscratch_c = ParallaxScratch::new(n, d);
    // Prime the cache so steady-state runs reflect the cached path.
    {
        let _ = tiled_attention_parallax_forward_sink_aware(
            &q, &k, &v, &mut out_cached, n, d, scale, &r, &x, &cfg,
            &policy_dual, -2.0, &mut sscratch_cached, Some(&mut pscratch_c),
        );
    }
    let us_sa_cached = bench_us(5, 50, || {
        let kind = tiled_attention_parallax_forward_sink_aware(
            &q, &k, &v, &mut out_cached, n, d, scale, &r, &x, &cfg,
            &policy_dual, -2.0, &mut sscratch_cached, Some(&mut pscratch_c),
        );
        std::hint::black_box((out_cached[0], kind));
    });

    println!(
        "{:>5} {:>10.3} {:>10.3} {:>7.1}% {:>10.3} {:>7.1}% {:>10.3} {:>7.1}%",
        n, us_vanilla,
        us_sa_uniform, pct(us_sa_uniform, us_vanilla),
        us_sa_dual, pct(us_sa_dual, us_vanilla),
        us_sa_cached, pct(us_sa_cached, us_vanilla),
    );
}

fn main() {
    println!("=== Sink-Aware Forward Composition Latency Benchmark ===");
    println!("=== (Plan 289 T3.5 — wrapper overhead G3 gate) ===");
    println!();
    println!("parallax gate_scale = 0.0 (pure attention forward, isolates wrapper cost)");
    println!("d_h = 64, activation = Sigmoid");
    println!();
    println!(
        "{:>5} {:>10} {:>10} {:>8} {:>10} {:>8} {:>10} {:>8}",
        "n", "vanilla", "sa(Uniform)", "oh%",
        "sa(Dual)", "oh%", "sa(cached)", "oh%"
    );
    println!("{}", "-".repeat(80));

    let d = 64usize;
    for &n in &[64usize, 128, 256] {
        run_case(n, d);
    }

    println!();
    println!("G3 gate: sa(Uniform) overhead ≤ 5% vs vanilla. MUST PASS.");
    println!("sa(Dual): includes n×n attention retention + classifier + gate;");
    println!("          cost established by Plan 287 G3; cached path mitigates.");
    println!();
    println!("Note: sa(Uniform) calls vanilla forward directly — the only added");
    println!("cost is one `matches!` check on the policy enum. If overhead > 5%,");
    println!("investigate compiler inlining of the wrapper around the forward.");
}
