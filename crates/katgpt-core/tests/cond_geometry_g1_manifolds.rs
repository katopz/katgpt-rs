#![cfg(feature = "estimators")]
//! Plan 638 T2.1 — the G1 estimator-accuracy gate: `cond_id` on synthetic
//! manifolds with known dimension.
//!
//! Arms: uniform samples in d-balls (d ∈ {1, 2, 4, 8, 16}) embedded in
//! D = 64, plus the d-sphere arm (the L2N2 bias class, arXiv:2603.10493)
//! reported SEPARATELY — never pooled into a ball-accuracy claim.
//!
//! **Pre-registered bar (Plan 638 T2.1):** `|d̂ − d| ≤ max(1, 0.25·d)` on
//! ≥ 5 of the 6 manifold arms at n = 2000. The sphere arm's expected
//! dimension is d−1 (S³ → 3) and its reading is disclosed on its own
//! line.
//!
//! ⚠ Run at `--release` — each arm is an O(n²·d) pass (the debug-profile
//! wall is ~30× and proves nothing extra).

#[path = "fixture_cond_geometry/mod.rs"]
mod fixture;

use fixture::Lcg;
use katgpt_core::data_probe::cond_geometry::{BallCountScratch, ball_counts, cond_id};

const N: usize = 2000;
const BIG_D: usize = 64;
const TAU: f32 = 0.95;

/// One arm: population → d̂ + the k̄ support (disclosed, never hidden).
fn run_arm(points: &[f32]) -> (f32, f64, f64) {
    let all: Vec<u32> = (0..N as u32).collect();
    let mut scratch = BallCountScratch::with_capacity(N);
    ball_counts(points, N, BIG_D, &all, TAU, &mut scratch).expect("ball counts");
    let d = cond_id(&scratch, &all).expect("cond_id");
    let sum_a: u64 = scratch.k_a.iter().map(|&v| u64::from(v)).sum();
    let sum_b: u64 = scratch.k_b.iter().map(|&v| u64::from(v)).sum();
    (d, sum_a as f64 / N as f64, sum_b as f64 / N as f64)
}

fn bar_for(d: usize) -> f64 {
    f64::max(1.0, 0.25 * d as f64)
}

#[test]
fn g1_manifold_accuracy() {
    let ball_arms: [(usize, u64); 5] = [
        (1, 0x638_0001),
        (2, 0x638_0002),
        (4, 0x638_0004),
        (8, 0x638_0008),
        (16, 0x638_0016),
    ];

    let mut passed = 0usize;
    for (d, seed) in ball_arms {
        let mut pts = vec![0.0f32; N * BIG_D];
        let mut rng = Lcg::new(seed);
        fixture::uniform_in_dball_into(&mut pts, N, d, BIG_D, &mut rng);
        let (d_hat, ka, kb) = run_arm(&pts);
        let bar = bar_for(d);
        let ok = (d_hat as f64 - d as f64).abs() <= bar;
        if ok {
            passed += 1;
        }
        println!(
            "G1 ball d={d:>2}: d̂={d_hat:.3} bar=±{bar:.2} {} k̄_A={ka:.1} k̄_B={kb:.1}",
            if ok { "PASS" } else { "MISS" }
        );
        assert!(d_hat.is_finite(), "ball d={d}: non-finite d̂");
    }

    // The d-sphere arm (L2N2 bias class): S³ reads its LOCAL dimension 3.
    // Reported separately — the number below is never pooled into a
    // ball-accuracy claim; it participates in the pre-registered ≥5/6
    // count only.
    {
        let mut pts = vec![0.0f32; N * BIG_D];
        let mut rng = Lcg::new(0x638_0003);
        fixture::sphere_into(&mut pts, N, 4, BIG_D, &mut rng);
        let (d_hat, ka, kb) = run_arm(&pts);
        let expect = 3.0f64;
        let bar = bar_for(3);
        let ok = (d_hat as f64 - expect).abs() <= bar;
        if ok {
            passed += 1;
        }
        println!(
            "G1 sphere S³ (separate): d̂={d_hat:.3} (expect {expect}) bar=±{bar:.2} {} k̄_A={ka:.1} k̄_B={kb:.1}",
            if ok { "PASS" } else { "MISS" }
        );
        assert!(d_hat.is_finite(), "sphere: non-finite d̂");
    }

    println!("G1 tally: {passed}/6 arms within the pre-registered bar");
    assert!(
        passed >= 5,
        "G1 FAILED: {passed}/6 arms within bar (pre-registered floor 5)"
    );
}
