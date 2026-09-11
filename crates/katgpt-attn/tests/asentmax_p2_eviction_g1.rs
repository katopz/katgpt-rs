//! Issue 747 P2 T2.2 — eviction-window G1 gate: **bit-identity**.
//!
//! Prop E.2 (arXiv:2506.16640; "Prop 6" in Research 549): for an
//! ALiBi-biased entmax-1.5 head with raw logits bounded `[z_min, z_max]`,
//! attention at distance `> d_max` is *exactly* zero — so an evicted-vs-full
//! run must be **bit-identical**, not close. This gate asserts, on
//! adversarial rows:
//!
//! - **G1a (support ⊆ window):** every supported index of the FULL row sits
//!   at distance ≤ d_max.
//! - **G1b (exact zeros):** probabilities at evicted indices are exactly
//!   `0.0` (not ε).
//! - **G1c (bit-identity):** `entmax_1p5` over the windowed suffix equals
//!   the full row's kept entries bit-for-bit (`f32::to_bits`).
//!
//! Adversarial constructions: the max logit parked at the FARTHEST distance
//! (worst case for the window), boundary values, tied clusters at the
//! support edge, and seeded random rows.

#![cfg(feature = "asentmax_schedule")]

use katgpt_attn::dash_attn::entmax::entmax_1p5;
use katgpt_attn::dash_attn::eviction_window::{
    alibi_entmax_window_1p5, evicted_kv_fraction, kv_within_window,
};

/// Deterministic splitmix64 → f32 uniform in [0, 1).
fn unit(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Assert the full G1 triple for one biased row.
///
/// `biased[j]` is the score of the key at position j for a query at the row
/// end: distance `d(j) = (n−1) − j`, so eviction drops the row PREFIX
/// (old tokens), keeping the last `d_max + 1` entries.
fn check_row(biased: &[f32], z_min: f32, z_max: f32, slope: f32, label: &str) {
    let n = biased.len();
    let d_max = alibi_entmax_window_1p5(z_min, z_max, slope);
    let (probs_full, _tau) = entmax_1p5(biased);

    let keep_from = n.saturating_sub(d_max.saturating_add(1));
    let (probs_windowed, _) = entmax_1p5(&biased[keep_from..]);

    for j in 0..n {
        let d = (n - 1) - j;
        if d > d_max {
            // G1b: evicted positions are EXACTLY zero in the full run.
            assert!(
                probs_full[j] == 0.0,
                "{label}: evicted index {j} (d={d}, d_max={d_max}) has prob {}",
                probs_full[j]
            );
        } else {
            // G1a (support ⊆ window) is implied by G1b; assert support
            // positively too via the windowed bit-identity below.
            // G1c: bit-identity at kept positions.
            let w = probs_windowed[j - keep_from].to_bits();
            let f = probs_full[j].to_bits();
            assert_eq!(
                w, f,
                "{label}: bit mismatch at kept index {j} (d={d}, d_max={d_max})"
            );
        }
    }
}

/// Adversarial row builders — every raw logit strictly inside [z_min, z_max].
fn row_random(n: usize, z_min: f32, z_max: f32, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| z_min + unit(&mut s) * (z_max - z_min))
        .collect()
}

fn row_max_at_far(n: usize, z_min: f32, z_max: f32, seed: u64) -> Vec<f32> {
    // Worst case for the window: the BEST content at the WORST distance
    // (index 0), the worst content nearest the query.
    let mut z = row_random(n, z_min, z_max, seed);
    z[0] = z_max;
    z[n - 1] = z_min;
    z[n / 2] = z_max; // and one more far-out max
    z
}

fn row_tied_clusters(n: usize, z_min: f32, z_max: f32, seed: u64) -> Vec<f32> {
    // Tied clusters at the support edge — stresses the stable-sort boundary.
    let mut z = row_random(n, z_min, z_max, seed);
    for (i, slot) in z.iter_mut().enumerate().take(16) {
        *slot = z_max - (i as f32 % 4.0) * 1e-4;
    }
    z
}

fn apply_alibi(z: &[f32], slope: f32) -> Vec<f32> {
    let n = z.len();
    (0..n)
        .map(|j| z[j] - slope * ((n - 1 - j) as f32))
        .collect()
}

#[test]
fn p2_g1_bit_identity_sweep() {
    let slopes = [0.02_f32, 0.125, 0.5, 2.0];
    let ranges = [(-0.0_f32, 4.0), (-2.0, 2.0), (-1.0, 12.0)];
    for &slope in &slopes {
        for &(z_min, z_max) in &ranges {
            for &n in &[64_usize, 256, 1024] {
                for seed in 0..4_u64 {
                    for (builder, label) in [
                        (row_random as fn(usize, f32, f32, u64) -> Vec<f32>, "random"),
                        (row_max_at_far as fn(usize, f32, f32, u64) -> Vec<f32>, "max_at_far"),
                        (
                            row_tied_clusters as fn(usize, f32, f32, u64) -> Vec<f32>,
                            "tied_clusters",
                        ),
                    ] {
                        let z = builder(n, z_min, z_max, seed.wrapping_mul(0x9E37) + 1);
                        let biased = apply_alibi(&z, slope);
                        check_row(&biased, z_min, z_max, slope, label);
                    }
                }
            }
        }
    }
}

#[test]
fn p2_g1_window_actually_evicts() {
    // The gate must not pass vacuously: with a steep slope and a wide
    // z-range the window is far smaller than the row and mass is evicted.
    let n = 4_096_usize;
    let (z_min, z_max) = (-1.0_f32, 12.0);
    let slope = 2.0_f32;
    let d_max = alibi_entmax_window_1p5(z_min, z_max, slope);
    assert_eq!(d_max, 8); // ⌊(13+2)/2 + 1⌋ = ⌊8.5⌋
    assert!(d_max + 1 < n, "window must be smaller than the row");
    let frac = evicted_kv_fraction(n, d_max);
    assert!(frac > 0.99, "expected >99% evictable at n={n}, got {frac}");

    // And a full run on an adversarial row: support stays inside the window.
    let z = row_max_at_far(n, z_min, z_max, 42);
    let biased = apply_alibi(&z, slope);
    check_row(&biased, z_min, z_max, slope, "evicts");
}

#[test]
fn p2_g1_retention_predicate_splits_at_the_boundary() {
    let d_max = alibi_entmax_window_1p5(0.0, 4.0, 2.0);
    assert_eq!(d_max, 4);
    assert!(kv_within_window(4, d_max));
    assert!(!kv_within_window(5, d_max));
}

#[test]
fn p2_g1_window_covers_row_keeps_everything() {
    // Window ≥ row: nothing evicted, windowed == full trivially (the
    // saturating keep_from path).
    let n = 16_usize;
    let (z_min, z_max) = (0.0_f32, 4.0);
    let slope = 0.02_f32; // d_max = ⌊300+1⌋ = 301 ≫ n
    let d_max = alibi_entmax_window_1p5(z_min, z_max, slope);
    assert!(d_max >= n);
    let z = row_random(n, z_min, z_max, 7);
    let biased = apply_alibi(&z, slope);
    check_row(&biased, z_min, z_max, slope, "cover");
}
