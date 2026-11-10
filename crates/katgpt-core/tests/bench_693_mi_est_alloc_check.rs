//! Plan 583 G4 — zero-alloc steady state for the `mi_est` evaluate path.
//!
//! Separate single-purpose binary so the counting allocator global picks up
//! only the mi path's allocations (the bench_576/655/680/685 convention —
//! parallel tests in the GOAT binary allocate). The evaluate path is: score
//! joint → draw permutation → score perm → DV report → bound ladder →
//! permutation test (uniform + dCor + stratified) + multi-draw antithetic —
//! all buffers scratch-resident, so the steady state must allocate NOTHING.

#![cfg(feature = "gaussianity_probe")]

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::mi::bounds;
use katgpt_core::mi::dv::dv_report;
use katgpt_core::mi::perm::{PermStat, PermTest, PermVariant};
use katgpt_core::mi::{Critic, MiScratch, PermSource};

fn population(n: usize, d: usize, rho: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = SplitMix(seed);
    let mut x = Vec::with_capacity(n * d);
    let mut y = Vec::with_capacity(n * d);
    for _ in 0..n {
        for _ in 0..d {
            let gx = rng.normal();
            let ge = rng.normal();
            x.push(gx);
            y.push(rho * gx + (1.0 - rho * rho).sqrt() * ge);
        }
    }
    (x, y)
}

struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn normal(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln())
            .sqrt()
            .mul_add((2.0 * std::f64::consts::PI * u2).cos(), 0.0) as f32
    }
}

/// One function so the checks run serially against the shared allocator.
#[test]
fn g4_zero_alloc_steady_state() {
    let n = 4096;
    let d = 16;
    let (x, y) = population(n, d, 0.3, 42);
    // One-time scratch construction (allocates by design).
    let mut scratch = MiScratch::new(n, d, 7);
    let test = PermTest {
        b: 64,
        seed: 11,
        variant: PermVariant::Uniform,
        stat: PermStat::Median,
    };
    let strata: Vec<u32> = (0..n).map(|i| (i / 64) as u32).collect();

    // Warm EVERY path once (first-run buffer growth is by design).
    let _ = test.run(Critic::Dot, &x, &y, n, d, None, &mut scratch);
    let _ = test.run_dcor(&x[..512 * d], &y[..512 * d], 512, d, None, &mut scratch);
    let _ = test.run(Critic::Dot, &x, &y, n, d, Some(&strata), &mut scratch);
    let _ = katgpt_core::mi::dv::dv_bound_perm_average(
        Critic::Dot,
        &x,
        &y,
        n,
        d,
        8,
        true,
        &mut scratch,
    );

    // ── Steady state: core score + bound + uniform perm + dCor ────────────
    let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    for _ in 0..8 {
        scratch.score_joint(Critic::Dot, &x, &y, n, d);
        scratch.next_perm(n);
        scratch.score_perm(Critic::Dot, &x, &y, n, d, PermSource::Current);
        let _ = dv_report(&scratch.joint, &scratch.perm);
        let _ = bounds::bounds_all(&scratch, &[4, 16, 64]);
        let _ = test.run(Critic::Dot, &x, &y, n, d, None, &mut scratch);
        let _ = test.run_dcor(&x[..512 * d], &y[..512 * d], 512, d, None, &mut scratch);
    }
    let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "steady-state evaluate path allocated {} times",
        after - before
    );

    // ── Steady state: stratified + multi-draw antithetic ──────────────────
    let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    for _ in 0..4 {
        let _ = test.run(Critic::Dot, &x, &y, n, d, Some(&strata), &mut scratch);
        let _ = katgpt_core::mi::dv::dv_bound_perm_average(
            Critic::Dot,
            &x,
            &y,
            n,
            d,
            8,
            true,
            &mut scratch,
        );
    }
    let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "stratified/multi-draw path allocated {} times",
        after - before
    );
}
