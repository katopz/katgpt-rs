#![cfg(feature = "estimators")]
//! Plan 638 T2.2 — the G1b subset-conditioning gate (binding amendment 1).
//!
//! **(i) Identity (bit-exact):** `cond_id` over a group G is BIT-IDENTICAL
//! to a restricted average of the once-computed per-point statistics —
//! the gate recomputes k̄_A^G / k̄_B^G from the public scratch counts with
//! the SAME pinned log form and asserts bit equality. This proves the
//! estimator consumes full-population counts, never subset-private ones.
//!
//! **(ii) The estimand gap (disclosed, never asserted small):** the
//! subset-in-isolation estimate (counts recomputed on G alone) DIFFERS —
//! a different estimand (the neighbor sets themselves change), not an
//! error. The gate asserts the difference is present and both readings
//! are finite, and prints the gap.
//!
//! Fixture: two well-separated clusters (128 + 128 rows, dim 8 embedded
//! in 32) — isolation radii collapse inside a cluster, so the gap is
//! large by construction.
//!
//! ⚠ Run at `--release` (O(n²·d) arms; the debug wall proves nothing).

#[path = "fixture_cond_geometry/mod.rs"]
mod fixture;

use fixture::Lcg;
use katgpt_core::data_probe::cond_geometry::{
    ball_counts, cond_id, BallCountScratch, MIN_GROUP_TOKENS,
};

const CLUSTER: usize = 128;
const DIM: usize = 8;
const BIG_D: usize = 32;
const TAU: f32 = 0.9;

fn cluster_population() -> Vec<f32> {
    let n = 2 * CLUSTER;
    let mut pts = vec![0.0f32; n * BIG_D];
    let mut rng = Lcg::new(0x638_0b01);
    for i in 0..n {
        let row = &mut pts[i * BIG_D..(i + 1) * BIG_D];
        for (c, v) in row.iter_mut().enumerate() {
            *v = if c < DIM {
                if i < CLUSTER {
                    // Cluster A at the origin.
                    rng.next_gauss() * 0.5
                } else {
                    // Cluster B at (50, …, 50).
                    50.0 + rng.next_gauss() * 0.5
                }
            } else {
                0.0
            };
        }
    }
    pts
}

#[test]
fn g1b_identity_and_estimand_gap() {
    const {
        assert!(CLUSTER >= MIN_GROUP_TOKENS);
    }
    let pts = cluster_population();
    let n = 2 * CLUSTER;
    let a: Vec<u32> = (0..CLUSTER as u32).collect();
    let all: Vec<u32> = (0..n as u32).collect();

    // ── (i) the identity: cond_id == restricted average, bit-exact ──
    let mut scratch = BallCountScratch::with_capacity(n);
    ball_counts(&pts, n, BIG_D, &all, TAU, &mut scratch).expect("counts");
    let d_full = cond_id(&scratch, &a).expect("conditional estimate");

    let sum_a: u64 = a.iter().map(|&i| u64::from(scratch.k_a[i as usize])).sum();
    let sum_b: u64 = a.iter().map(|&i| u64::from(scratch.k_b[i as usize])).sum();
    let ka_bar = sum_a as f64 / CLUSTER as f64;
    let kb_bar = sum_b as f64 / CLUSTER as f64;
    let d_ref = ((ka_bar.ln() - kb_bar.ln()) / (scratch.tau as f64).ln()) as f32;
    assert_eq!(
        d_full.to_bits(),
        d_ref.to_bits(),
        "cond_id is NOT the restricted average of the once-computed counts"
    );
    assert!(
        ka_bar > 0.0 && kb_bar > 0.0,
        "support disclosure: empty balls"
    );

    // ── (ii) the estimand gap: subset-in-isolation is a DIFFERENT number ──
    // The neighbor sets change in isolation — the gate asserts presence
    // and finiteness, never smallness.
    let mut sub = vec![0.0f32; CLUSTER * BIG_D];
    for i in 0..CLUSTER {
        let src = i * BIG_D..(i + 1) * BIG_D;
        sub[src.clone()].copy_from_slice(&pts[src]);
    }
    let sub_all: Vec<u32> = (0..CLUSTER as u32).collect();
    let mut sub_scratch = BallCountScratch::with_capacity(CLUSTER);
    ball_counts(&sub, CLUSTER, BIG_D, &sub_all, TAU, &mut sub_scratch).expect("subset counts");
    let d_sub = cond_id(&sub_scratch, &sub_all).expect("isolation estimate");

    println!(
        "G1b: d̂(full-population conditioning) = {d_full:.4}, d̂(isolation) = {d_sub:.4}, gap = {:.4}",
        (d_sub - d_full).abs()
    );
    println!(
        "     support: full k̄_A={ka_bar:.2} k̄_B={kb_bar:.2} | isolation k̄_A={:.2} k̄_B={:.2}",
        sub_scratch.k_a.iter().map(|&v| f64::from(v)).sum::<f64>() / CLUSTER as f64,
        sub_scratch.k_b.iter().map(|&v| f64::from(v)).sum::<f64>() / CLUSTER as f64
    );
    assert!(
        d_full.is_finite() && d_sub.is_finite(),
        "estimand gap: non-finite reading"
    );
    assert_ne!(
        d_full.to_bits(),
        d_sub.to_bits(),
        "the estimand gap vanished — the fixture no longer separates the two estimands"
    );
}
