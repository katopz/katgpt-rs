//! Plan 292 Phase 4 T4.5 G6 — `FutureBehaviorProbe::forecast()` latency bench.
//!
//! Target (Plan 292 T4.5 G6): **`forecast()` < 200ns per call**, matching the
//! `EmotionDirections::project` cousin latency. The probe does strictly more
//! work than a bare projection (RwLock read-clone + `simd_dot_f32` + sigmoid),
//! so this bench reports BOTH so the overhead is visible and the gate verdict
//! is honest.
//!
//! # Why not criterion
//!
//! `criterion` is **not** a `katgpt-rs` dev-dependency (see root `Cargo.toml`
//! `[dev-dependencies]`: only `ratatui`, `crossterm`, `tempfile`). The repo
//! bench convention — established by `benches/faithfulness_probe_bench.rs` and
//! `benches/self_advantage_gate_bench.rs` — is `std::time::Instant` +
//! `std::hint::black_box` + `harness = false` + `fn main()`. This file follows
//! that convention. (Deviation from the task prompt's "criterion" wording is
//! forced by dependency availability — measuring with `Instant` produces a real
//! ns/iter number, which is what the GOAT gate actually needs.)
//!
//! # Run
//!
//! ```text
//! cargo bench --bench fpcg_probe_forecast_bench --features future_probe
//! # or equivalently:
//! cargo run --release --bench fpcg_probe_forecast_bench --features future_probe
//! ```
//!
//! Release build is required for SIMD auto-vectorization to engage
//! (`.contexts/optimization.md`: "SIMD benefits only appear with optimizations").

use std::hint::black_box;
use std::time::Instant;

use katgpt_rs::pruners::emotion_vector::EmotionDirections;
use katgpt_rs::pruners::future_probe::FutureBehaviorProbe;

/// Warm-up iterations to prime CPU caches before the timed loop
/// (`.contexts/optimization.md`: "Warm up before measuring (100+ iterations)").
const WARMUP_ITERS: usize = 10_000;

/// Timed iterations. 1_000_000 × ~100ns ≈ 0.1s per measurement — enough samples
/// for a stable mean without making the bench slow.
const BENCH_ITERS: usize = 1_000_000;

/// d_model sizes to sweep. 64 matches the `bench_emotion_vector_goat.rs`
/// cousin config (n_embd = 16 × n_head = 16 × 4 = 64) for an apples-to-apples
/// comparison at our default scale. 768/1024/2048/4096 cover realistic
/// mid-layer residual widths (BERT-base → 8B-class models) so the gate verdict
/// is reported at the sizes a real probe would actually run at.
const D_MODELS: &[usize] = &[64, 256, 768, 1024, 2048, 4096];

fn main() {
    println!("=== Plan 292 T4.5 G6 — FutureBehaviorProbe::forecast() latency ===");
    println!(
        "Target: < 200 ns/call (matches EmotionDirections). warmup={WARMUP_ITERS}, timed={BENCH_ITERS} iters.\n"
    );

    println!(
        "{:>8} {:>16} {:>16} {:>14} {:>14}",
        "d_model", "forecast ns/it", "project ns/it", "probe_verdict", "ratio"
    );

    let mut any_fail = false;

    for &d_model in D_MODELS {
        // ── Build probe and cousin projection with identical d_model ──
        // Unit direction so the forecast stays in a sane logit range regardless
        // of activation scale; bias = 0.0 so σ(dot) is the readout.
        let direction: Vec<f32> = (0..d_model)
            .map(|i| 1.0 / ((d_model as f32).sqrt()) * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let activation: Vec<f32> = (0..d_model)
            .map(|i| (i as f32) * 0.01 - 0.5)
            .collect();

        let probe = FutureBehaviorProbe::new(direction.clone(), 0.0, 7, "bench");
        // Cousin: one EmotionDirections direction vector of the same dimension.
        // `project` is the canonical detection-side dot product (Plan 162).
        let emotion_dirs = EmotionDirections::new(
            direction.clone(),
            direction.clone(),
            direction.clone(),
            direction.clone(),
        );

        // ── Warm up both paths ──
        let mut sink = 0.0_f32;
        for _ in 0..WARMUP_ITERS {
            sink += black_box(probe.forecast(black_box(&activation)).probability);
        }
        for _ in 0..WARMUP_ITERS {
            sink += black_box(EmotionDirections::project(
                black_box(&activation),
                black_box(&emotion_dirs.valence),
            ));
        }
        // Prevent the compiler from folding `sink` away.
        if black_box(sink.is_nan()) {
            eprintln!("warmup sink nan (impossible for finite inputs)");
        }

        // ── Timed: forecast() ──
        let start = Instant::now();
        let mut acc_probe = 0.0_f32;
        for _ in 0..BENCH_ITERS {
            acc_probe += black_box(probe.forecast(black_box(&activation)).probability);
        }
        let probe_elapsed = start.elapsed();
        let probe_ns_per_it = probe_elapsed.as_secs_f64() * 1e9 / BENCH_ITERS as f64;

        // ── Timed: EmotionDirections::project() (cousin baseline) ──
        let start = Instant::now();
        let mut acc_proj = 0.0_f32;
        for _ in 0..BENCH_ITERS {
            acc_proj += black_box(EmotionDirections::project(
                black_box(&activation),
                black_box(&emotion_dirs.valence),
            ));
        }
        let proj_elapsed = start.elapsed();
        let proj_ns_per_it = proj_elapsed.as_secs_f64() * 1e9 / BENCH_ITERS as f64;

        // Keep accumulators live (sum to a single comparison so the loop
        // bodies can't be DCE'd).
        let _ = acc_probe + acc_proj;

        let verdict = if probe_ns_per_it < 200.0 {
            "PASS ✅"
        } else {
            any_fail = true;
            "FAIL ❌ (>200ns)"
        };
        let ratio = probe_ns_per_it / proj_ns_per_it.max(1e-9);

        println!(
            "{:>8} {:>16.2} {:>16.2} {:>14} {:>13.2}×",
            d_model, probe_ns_per_it, proj_ns_per_it, verdict, ratio
        );
    }

    println!();
    println!(
        "Gate G6 (<200ns for forecast): {}",
        if any_fail {
            "FAIL at one or more sizes — see per-size rows"
        } else {
            "PASS at all swept sizes"
        }
    );
    println!();
    println!("Notes:");
    println!("  - forecast() = RwLock read-clone + simd_dot_f32 + sigmoid (strictly more work than project).");
    println!("  - project()  = chunked dot product only (the detection-side cousin, Plan 162).");
    println!("  - The 200ns gate is the relevant one at the d_model a real probe would run at");
    println!("    (paper uses mid-layer residual; 768–4096 is the realistic band).");
    println!("  - ratio > 1.0 means the prediction-side probe pays a measurable overhead vs");
    println!("    the detection-side projection; the gate tolerance (200ns) absorbs this.");

    // Non-zero exit on failure so CI / `cargo bench` surfaces a regression.
    if any_fail {
        std::process::exit(1);
    }
}
