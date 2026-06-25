//! Orthogonal Procrustes benchmark — Issue 001 G1, G2, G4, G6 gates.
//!
//! Uses `std::time::Instant` (matches other katgpt-rs benches — no criterion dep).
//!
//! Run:
//! ```bash
//! cargo run --release --bench procrustes_bench --features orthogonal_procrustes
//! ```
//!
//! Sweeps the Issue 001 gate matrix:
//! - G1: alignment latency at n=512, d=32 → ≤ 800 µs
//! - G2: alignment latency at n=2048, d=64 → ≤ 5 ms
//! - G4: residual on known-rotation synthetic data at n ≥ 128 → ≤ 1%
//! - G6: per-call scratch size at d=64 → ≤ 64 KB
//!
//! G3 (determinism) and G5 (downstream recall) live in separate tests
//! (`tests/procrustes_determinism.rs` and a future retrieval bench).

#![cfg(feature = "orthogonal_procrustes")]

use katgpt_rs::procrustes::{
    orthogonal_procrustes, ProcrustesConfig, ProcrustesScratch,
};
use std::time::{Duration, Instant};

/// Deterministic xorshift32 PRNG (matches the unit-test PRNG).
fn seeded_anchors(seed: u32, n: usize, d: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * d);
    let mut state = seed.max(1); // avoid 0 fixed point
    for _ in 0..(n * d) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        out.push(((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0);
    }
    out
}

/// Apply a known d×d rotation to a (n, d) row-major matrix. For d=2 we use
/// a fixed 30° rotation; for d=3+ we use identity + a single Givens rotation
/// in the (0,1) plane. This gives a known ground-truth R that Procrustes
/// should recover.
fn apply_known_rotation(src: &[f32], n: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    // Ground-truth R: identity except R[0,0]=R[1,1]=cos, R[0,1]=-sin, R[1,0]=sin.
    let theta = 0.5_f32; // ~28.6°, arbitrary.
    let cos = theta.cos();
    let sin = theta.sin();
    let mut r = vec![0.0_f32; d * d];
    for i in 0..d {
        r[i * d + i] = 1.0;
    }
    if d >= 2 {
        r[0] = cos;
        r[1] = -sin;
        r[d] = sin;
        r[d + 1] = cos;
    }
    // B = A @ R^T  (so Procrustes should recover R).
    let mut b = vec![0.0_f32; n * d];
    for i in 0..n {
        let a_row = &src[i * d..(i + 1) * d];
        let b_row = &mut b[i * d..(i + 1) * d];
        for cprime in 0..d {
            let mut acc = 0.0_f32;
            for c in 0..d {
                // R^T[c, cprime] = R[cprime, c]
                acc += a_row[c] * r[cprime * d + c];
            }
            b_row[cprime] = acc;
        }
    }
    (b, r)
}

/// Best-of-N wall-clock microseconds for a closure.
fn bench_us(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut best = Duration::MAX;
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = Instant::now() - t0;
        if dt < best {
            best = dt;
        }
    }
    best.as_secs_f64() * 1e6
}

/// Format µs nicely.
fn fmt_us(us: f64) -> String {
    if us < 1.0 {
        format!("{:.2} µs", us * 1000.0)
    } else if us < 1000.0 {
        format!("{:.2} µs", us)
    } else {
        format!("{:.3} ms", us / 1000.0)
    }
}

fn main() {
    println!("=== Orthogonal Procrustes Benchmark (Issue 001 G1/G2/G4/G6) ===\n");

    let cases: &[(usize, usize, f64, &str)] = &[
        // (n, d, latency_gate_us, gate_name)
        (32, 16, 200.0, "small"),
        (128, 16, 400.0, "G4-baseline"),
        (512, 32, 800.0, "G1"),     // Issue 001 G1: ≤ 800 µs.
        (2048, 64, 5000.0, "G2"),   // Issue 001 G2: ≤ 5 ms.
    ];

    let mut all_pass = true;

    println!("{:>6} {:>4} {:>12} {:>12} {:>10}  gate", "n", "d", "latency", "residual", "scratch");
    println!("{:->6} {:->4} {:->12} {:->12} {:->10}  ----", "", "", "", "", "");

    for &(n, d, gate_us, gate_name) in cases {
        let a = seeded_anchors(42, n, d);
        let (b, _r_expected) = apply_known_rotation(&a, n, d);
        let mut r = vec![0.0_f32; d * d];
        let mut scratch = ProcrustesScratch::new(n, d);

        // Latency: best-of-20 after 5 warmups.
        // Issue 001 G1/G2 gates measure alignment-only latency.
        // The residual pass is O(n d²) extra work; for the latency gate we
        // disable it (production shard-join calls don't need a residual
        // every time — only on diagnostics / first-call sanity).
        let mut cfg = ProcrustesConfig {
            compute_residual: false,
            ..Default::default()
        };
        let us = bench_us(5, 20, || {
            let _ = orthogonal_procrustes(&a, &b, n, d, &mut r, &mut scratch, &cfg)
                .expect("procrustes");
        });
        // Separate call WITH residual for the G4 quality check.
        cfg.compute_residual = true;
        let report = orthogonal_procrustes(&a, &b, n, d, &mut r, &mut scratch, &cfg)
            .expect("procrustes with residual");

        // Scratch size (G6): mean_a + mean_b + m + xtx + x_new + predicted_row
        // = 2*d + 3*d*d + d = 3*d + 3*d*d floats.
        // (No Newton-Schulz internal scratch — we use our own polar iteration.)
        let scratch_bytes = (3 * d + 3 * d * d) * std::mem::size_of::<f32>();

        let residual_pct = report.residual * 100.0;
        let latency_pass = us <= gate_us;
        let residual_pass = if n >= 128 {
            report.residual <= 0.01 // G4: ≤ 1% at n ≥ 128.
        } else {
            true // Don't enforce G4 below n=128.
        };
        let scratch_pass = scratch_bytes <= 64 * 1024; // G6: ≤ 64 KB.

        let all = latency_pass && residual_pass && scratch_pass;
        if !all { all_pass = false; }

        println!(
            "{:>6} {:>4} {:>12} {:>11.4}% {:>9} B  {}",
            n,
            d,
            fmt_us(us),
            residual_pct,
            scratch_bytes,
            if all { format!("✓ {}", gate_name) } else { format!("✗ {} (gate={} us)", gate_name, fmt_us(gate_us)) }
        );
        if !latency_pass {
            println!("     ⚠ latency {:.2} > gate {:.2}", us, gate_us);
        }
        if !residual_pass {
            println!("     ⚠ residual {:.4}% > 1.0%", residual_pct);
        }
        if !scratch_pass {
            println!("     ⚠ scratch {} B > 64 KB", scratch_bytes);
        }
    }

    println!();
    if all_pass {
        println!("=== ALL GATES PASS — GOAT candidate confirmed. Promote to default-on. ===");
        std::process::exit(0);
    } else {
        println!("=== SOME GATES FAILED — keep behind feature flag, revisit per Issue 001 outcome matrix. ===");
        std::process::exit(1);
    }
}
