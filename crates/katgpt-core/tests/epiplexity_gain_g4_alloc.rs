//! Plan 641 G4 — zero-allocation gate for the epiplexity decision path.
//!
//! `update_counts` → `marginal_gain` → `derivative_gain` →
//! `sigmoid_weights` → `predicted_gap` must allocate 0 bytes after warmup:
//! fixed `[f64; MAX_DOMAINS]` scratch, in-place quickselect, no interior
//! allocation. The fit's `Vec`-scratch class stays inside
//! `GainEstimator::fit` and is OUT of the counted window (fit is the
//! bounded, warm path). Mirrors the `marginal_rewind_alloc_check`
//! convention (separate binary: `#[global_allocator]` is crate-binary-unique).

#![cfg(feature = "estimators")]

use katgpt_core::epiplexity_gain::{GainCell, GainEstimator, MixPanel};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

/// Linear 0→1 ramp over `[a, b)`, 1 after.
fn ramp(t: usize, a: usize, b: usize) -> f64 {
    if t < a {
        0.0
    } else if t >= b {
        1.0
    } else {
        (t - a) as f64 / (b - a) as f64
    }
}

/// The Plan-641 test-1-shaped panel: W=160, K=3 staggered ramps.
fn cross_mix_counts() -> Vec<f32> {
    let (w, k) = (160_usize, 3_usize);
    let mut v = Vec::with_capacity(w * k);
    for t in 0..w {
        v.push((4.0 + 116.0 * ramp(t, 0, 100)) as f32);
        v.push((90.0 * ramp(t, 40, 140)) as f32);
        v.push((2.0 + 38.0 * ramp(t, 10, 159)) as f32);
    }
    v
}

#[test]
fn decision_path_zero_alloc_after_warmup() {
    // Warmup: build the panel + fit (allocations allowed; the Vec-scratch
    // class lives and dies inside fit). Deterministic accuracy series —
    // no RNG needed to warm the internals.
    let counts = cross_mix_counts();
    let gamma = [0.5_f64, 0.3, 0.2];
    let mut acc = Vec::with_capacity(160);
    for t in 0..160 {
        let mut n = 0.0_f64;
        for d in 0..3 {
            n += gamma[d] * (f64::from(counts[t * 3 + d]) + 1.0);
        }
        let gap = 0.02 + n.powf(-0.7);
        acc.push((1.0 - gap) as f32);
    }
    let panel = MixPanel::new(3, 160, &counts);
    let mut est = GainEstimator::fit(&panel, &acc).expect("warmup fit");
    assert!(
        est.cross_fit().is_some(),
        "warmup fit must arm the cross curve"
    );

    // The counter canary — a dead counter makes every alloc gate vacuous.
    assert_counter_is_live();

    let last: &[f32] = &counts[159 * 3..160 * 3];
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let mut sink = 0.0_f64;
    for _ in 0..4096_usize {
        est.update_counts(last).expect("update");
        let gains = est.marginal_gain();
        let derivs = est.derivative_gain();
        let wts = est.sigmoid_weights(0.5);
        let gap = est.predicted_gap(last);
        // The sink reads every path's output so nothing can fold away.
        sink += f64::from(wts[0]) + f64::from(wts[15]);
        sink += gap.unwrap_or(0.0);
        if let GainCell::Gain(v) = gains[2] {
            sink += f64::from(v) * 1e-6;
        }
        if let GainCell::Gain(v) = derivs[1] {
            sink += f64::from(v) * 1e-6;
        }
    }
    let delta = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(
        delta, 0,
        "epiplexity decision path (update_counts/marginal_gain/derivative_gain/\
         sigmoid_weights/predicted_gap) allocated {delta} bytes over 4096 iterations"
    );
    assert!(sink.is_finite(), "sink drifted non-finite: {sink}");
}
