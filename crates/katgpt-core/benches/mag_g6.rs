//! MAG G6 — latency GOAT gate (Plan 418 Phase 2 T2.6).
//!
//! Measures the hot-path latency of the core MAG primitives:
//! - `mine_direction` on 500×64: target < 100µs.
//! - `mine_contrast_direction` on 250+250×64: target < 100µs.
//! - `transfer_score` (class-conditional) on 100-sample × 64-dim: target < 10µs.
//! - `reconstruction_error` on 100×64: target < 50µs.
//!
//! Convention: `std::time::Instant` + `harness = false` (no Criterion dev-dep),
//! matching the `bench_377` / `bench_410` convention.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/mag_g6 cargo bench -p katgpt-core \
//!   --features mag_mining --bench mag_g6 -- --nocapture
//! ```

use katgpt_core::mag::{
    DataSet, TransferMetric, mine_contrast_direction, mine_direction, rank_candidates,
    reconstruction_error,
};
use std::hint::black_box;
use std::time::Instant;

const D: usize = 64;

#[inline]
fn gaussian(rng: &mut fastrand::Rng) -> f32 {
    let u1 = rng.f32().max(1e-10);
    let u2 = rng.f32();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

fn main() {
    const WARMUP: usize = 200;

    let mut rng = fastrand::Rng::with_seed(0xA600_0001);

    println!("══════════════════════════════════════════════════════════════════");
    println!("  Plan 418 Phase 2 — MAG G6 latency gate");
    println!("  D = {D}");
    println!("══════════════════════════════════════════════════════════════════\n");

    // ── Fixture 1: 500×64 paired data for mine_direction. ─────────────
    let n_mine = 500;
    let mut with = vec![[0.0_f32; D]; n_mine];
    let mut without = vec![[0.0_f32; D]; n_mine];
    for i in 0..n_mine {
        for j in 0..D {
            without[i][j] = gaussian(&mut rng);
            with[i][j] = without[i][j] + if j == 0 { 2.0 } else { 0.0 };
        }
    }

    // ── Fixture 2: 250+250×64 for mine_contrast_direction. ────────────
    let n_contrast = 250;
    let mut positive = vec![[0.0_f32; D]; n_contrast];
    let mut negative = vec![[0.0_f32; D]; n_contrast];
    for i in 0..n_contrast {
        for j in 0..D {
            positive[i][j] = gaussian(&mut rng) + if j == 0 { -2.0 } else { 0.0 };
            negative[i][j] = gaussian(&mut rng) + if j == 0 { 2.0 } else { 0.0 };
        }
    }

    // ── Fixture 3: 100-sample dataset for transfer_score. ─────────────
    let n_transfer = 100;
    let mut acts = vec![[0.0_f32; D]; n_transfer];
    let labels = {
        let mut l = vec![false; n_transfer];
        for x in l.iter_mut().take(n_transfer / 2) {
            *x = true;
        }
        l
    };
    for i in 0..n_transfer {
        for val in &mut acts[i] {
            *val = gaussian(&mut rng) + if labels[i] { 1.0 } else { -1.0 };
        }
    }
    let ds = DataSet::new(&acts, &labels);

    // ── Fixture 4: 100×64 for reconstruction_error. ──────────────────
    let n_recon = 100;
    let mut with_recon = vec![[0.0_f32; D]; n_recon];
    let mut without_recon = vec![[0.0_f32; D]; n_recon];
    let mut direction = vec![0.0_f32; D];
    direction[0] = 1.0;
    let alpha = 2.0;
    for i in 0..n_recon {
        for j in 0..D {
            without_recon[i][j] = gaussian(&mut rng);
            with_recon[i][j] = without_recon[i][j] + alpha * direction[j];
        }
    }

    // ── Latency measurements ──────────────────────────────────────────
    const ITERS: usize = 5_000;

    // Warmup all paths.
    for _ in 0..WARMUP {
        let _ = black_box(mine_direction(black_box(&with), black_box(&without)));
    }
    for _ in 0..WARMUP {
        let _ = black_box(mine_contrast_direction(
            black_box(&positive),
            black_box(&negative),
        ));
    }
    for _ in 0..WARMUP {
        let ds = DataSet::new(&acts, &labels);
        let _ = black_box(rank_candidates(
            black_box(&[ds, ds, ds, ds, ds, ds]),
            black_box(&ds),
            black_box(&[
                TransferMetric::ClassConditionalCosineBenign,
                TransferMetric::ClassConditionalCosineMalicious,
            ]),
        ));
    }
    for _ in 0..WARMUP {
        let _ = black_box(reconstruction_error(
            black_box(&with_recon),
            black_box(&without_recon),
            black_box(&direction),
            black_box(alpha),
        ));
    }

    // G6a: mine_direction 500×64 — target < 100µs.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let dir = mine_direction(black_box(&with), black_box(&without)).unwrap();
        let _ = black_box(dir);
    }
    let mine_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;
    let mine_us = mine_ns / 1000.0;

    // G6b: mine_contrast_direction 250+250×64 — target < 100µs.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let dir = mine_contrast_direction(black_box(&positive), black_box(&negative)).unwrap();
        let _ = black_box(dir);
    }
    let contrast_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;
    let contrast_us = contrast_ns / 1000.0;

    // G6c: rank_candidates (6 candidates × 2 metrics) on 100-sample × 64-dim —
    // target < 10µs for a single transfer_score call. rank_candidates does
    // 6 × 2 = 12 transfer_score calls, so target < 120µs total.
    let candidates: [DataSet<'_, [f32; D]>; 6] = [ds, ds, ds, ds, ds, ds];
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let entries = rank_candidates(
            black_box(&candidates),
            black_box(&ds),
            black_box(&[
                TransferMetric::ClassConditionalCosineBenign,
                TransferMetric::ClassConditionalCosineMalicious,
            ]),
        )
        .unwrap();
        let _ = black_box(entries);
    }
    let rank_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;
    let rank_us = rank_ns / 1000.0;
    let per_score_us = rank_us / 12.0; // 6 candidates × 2 metrics

    // G6d: reconstruction_error 100×64 — target < 50µs.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let r = reconstruction_error(
            black_box(&with_recon),
            black_box(&without_recon),
            black_box(&direction),
            black_box(alpha),
        )
        .unwrap();
        let _ = black_box(r);
    }
    let recon_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;
    let recon_us = recon_ns / 1000.0;

    // ── Verdict ───────────────────────────────────────────────────────
    let mine_pass = mine_us < 100.0;
    let contrast_pass = contrast_us < 100.0;
    let transfer_pass = per_score_us < 10.0;
    let recon_pass = recon_us < 50.0;

    println!("── G6 latency results ({ITERS} iters) ──");
    println!(
        "  mine_direction  500×64     {:>8.2} µs   (target < 100 µs)   {}",
        mine_us,
        pass_fail(mine_pass)
    );
    println!(
        "  mine_contrast   250+250×64 {:>8.2} µs   (target < 100 µs)   {}",
        contrast_us,
        pass_fail(contrast_pass)
    );
    println!(
        "  transfer_score  100×64     {:>8.3} µs   (target < 10 µs)    {}",
        per_score_us,
        pass_fail(transfer_pass)
    );
    println!(
        "  recon_error     100×64     {:>8.2} µs   (target < 50 µs)    {}",
        recon_us,
        pass_fail(recon_pass)
    );

    println!("\n══════════════════════════════════════════════════════════════════");
    let overall_pass = mine_pass && contrast_pass && transfer_pass && recon_pass;
    println!(
        "  OVERALL: {}",
        if overall_pass {
            "✓ ALL GATES PASS"
        } else {
            "✗ SOME GATES FAILED"
        }
    );
    println!("══════════════════════════════════════════════════════════════════");
    if !overall_pass {
        std::process::exit(1);
    }
}

fn pass_fail(ok: bool) -> &'static str {
    if ok { "✓ PASS" } else { "✗ FAIL" }
}
