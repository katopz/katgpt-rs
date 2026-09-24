//! Issue 879 T1/T2 G1 — correctness gates for the MAttr budget-primitive
//! family (`exact_mass_admit` + `log_frontier`, Research 584 /
//! arXiv:2609.25518).
//!
//! Pins the operator's three headline properties (sum-to-k,
//! shift-invariance, nestedness-in-k), the edge semantics (k ≤ 0, k ≥ n,
//! degenerate scores, tiny temperature), and the LogFrontier controller
//! protocol (sign-step both sides of target, both clamp ends, probe
//! cadence, determinism, the disarmed `probe_frac = 0` posture).
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --features exact_mass_admit \
//!     --test exact_mass_admit_g1 -- --nocapture
//! ```

#![cfg(feature = "exact_mass_admit")]

use katgpt_core::exact_mass_admit::{exact_mass_admit, exact_mass_admit_into, MAX_BISECT_ITERS};
use katgpt_core::log_frontier::LogFrontier;

/// Deterministic SplitMix64 (the workspace-bench idiom — no rand dep).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform `f32` in `[0, 1)`.
    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Uniform scores in `[-3, 3)`.
fn gen_scores(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64(seed);
    (0..n).map(|_| rng.next_unit() * 6.0 - 3.0).collect()
}

/// Sum-to-k tolerance: the bisection early-exits at `1e-9·n` mass error and
/// each emitted f32 mask element carries ≤ ~1.2e-7 of cast error.
fn sum_tol(n: usize) -> f64 {
    1e-3 + 2.0e-7 * n as f64
}

fn mask_sum(mask: &[f32]) -> f64 {
    mask.iter().map(|&m| m as f64).sum()
}

#[test]
fn sum_to_k_across_grid() {
    for (n, ks) in [
        (10usize, [1.0f32, 5.0, 9.0]),
        (1_000, [1.0, 500.0, 999.0]),
        (10_000, [1.0, 5_000.0, 9_999.0]),
    ] {
        let scores = gen_scores(n, 0x879_0000 + n as u64);
        let mut mask = vec![0.0f32; n];
        for &t in &[0.1f32, 1.0, 10.0] {
            for &k in &ks {
                let tau = exact_mass_admit_into(&scores, k, t, &mut mask);
                assert!(tau.is_finite(), "N={n} k={k} T={t}: tau must be finite");
                let err = (mask_sum(&mask) - k as f64).abs();
                assert!(
                    err <= sum_tol(n),
                    "N={n} k={k} T={t}: |Σm−k| = {err:.3e} > {}",
                    sum_tol(n)
                );
                for &m in &mask {
                    assert!((0.0..=1.0).contains(&m), "mask out of [0,1]: {m}");
                }
            }
        }
    }
}

#[test]
fn shift_invariance_power_of_two() {
    // Dyadic scores + power-of-two shift + T=1: every bisection intermediate
    // is an exactly-representable dyadic rational, so the shifted run is
    // BIT-identical (same break points, same mass values) and tau shifts by
    // exactly c.
    let mut rng = SplitMix64(0x5A1F7);
    let n = 512usize;
    let scores: Vec<f32> = (0..n)
        .map(|_| ((rng.next_unit() * 64.0) as u32 >> 4) as f32 * 0.25)
        .collect();
    let c = 1024.0f32;
    let shifted: Vec<f32> = scores.iter().map(|&s| s + c).collect();

    let k = 64.0f32;
    let (mut m1, mut m2) = (vec![0.0f32; n], vec![0.0f32; n]);
    let tau1 = exact_mass_admit_into(&scores, k, 1.0, &mut m1);
    let tau2 = exact_mass_admit_into(&shifted, k, 1.0, &mut m2);
    // The returned f32 taus can differ by 1 ulp (f64→f32 cast at magnitudes
    // ~9 vs ~1033 rounds on different grids); the MASK is the exact claim —
    // computed inside from the f64 tau, which shifts bit-exactly.
    assert!(
        (tau2 - (tau1 + c)).abs() <= 1e-3,
        "tau must shift by the score shift ({tau2} vs {tau1}+{c})"
    );
    for (a, b) in m1.iter().zip(&m2) {
        assert_eq!(a.to_bits(), b.to_bits(), "mask must be bit-identical");
    }
}

#[test]
fn nested_in_k_and_tau_monotone() {
    let n = 2_000usize;
    let scores = gen_scores(n, 0x4E57);
    let mut prev = vec![0.0f32; n];
    let mut prev_tau = f32::INFINITY;
    let mut cur = vec![0.0f32; n];
    for k in (50usize..=500).step_by(50) {
        let tau = exact_mass_admit_into(&scores, k as f32, 1.0, &mut cur);
        assert!(
            tau < prev_tau,
            "tau must strictly decrease in k (k={k}, tau {tau} !< {prev_tau})"
        );
        for i in 0..n {
            assert!(
                cur[i] >= prev[i] - 1e-9,
                "mask must be nested in k (k={k}, i={i}: {} < {})",
                cur[i],
                prev[i]
            );
        }
        prev.copy_from_slice(&cur);
        prev_tau = tau;
    }
}

#[test]
fn mass_extremes() {
    let scores = gen_scores(64, 0xE37);
    let mut mask = vec![0.0f32; 64];
    for k in [0.0f32, -3.0] {
        let tau = exact_mass_admit_into(&scores, k, 1.0, &mut mask);
        assert_eq!(tau, f32::INFINITY, "k ≤ 0 → tau = +inf");
        assert!(mask.iter().all(|&m| m == 0.0), "k ≤ 0 → zero mask");
    }
    for k in [64.0f32, 100.0] {
        let tau = exact_mass_admit_into(&scores, k, 1.0, &mut mask);
        assert_eq!(tau, f32::NEG_INFINITY, "k ≥ n → tau = -inf");
        assert!(mask.iter().all(|&m| m == 1.0), "k ≥ n → ones mask");
    }
    // k = n − 1 stays inside the bracket (n·σ(−10) ≈ 2.9e-3 < 1 at n=64).
    let tau = exact_mass_admit_into(&scores, 63.0, 1.0, &mut mask);
    assert!(tau.is_finite());
    assert!(
        (mask_sum(&mask) - 63.0).abs() <= sum_tol(64),
        "k = n−1 must still sum to k, got {}",
        mask_sum(&mask)
    );
}

#[test]
fn degenerate_constant_scores() {
    let n = 100usize;
    let scores = vec![2.5f32; n];
    let mut mask = vec![0.0f32; n];
    for &k in &[1.0f32, 37.5, 99.0] {
        let tau = exact_mass_admit_into(&scores, k, 1.0, &mut mask);
        assert!(tau.is_finite());
        let expect = k / n as f32;
        for &m in &mask {
            assert!(
                (m - expect).abs() <= 1e-5,
                "constant scores → uniform mask k/n ({m} vs {expect})"
            );
        }
        assert!(
            (mask_sum(&mask) - k as f64).abs() <= sum_tol(n),
            "constant scores must still sum to k"
        );
    }
}

#[test]
fn tiny_temperature_near_binary_mask() {
    // T = 1e-3 with scores at spacing ~6e-3: the mask is near-binary, and
    // the operator still pins the mass — the fractional boundary element
    // carries k − ⌊k⌋.
    let n = 1_000usize;
    let scores = gen_scores(n, 0x7149);
    let mut mask = vec![0.0f32; n];
    let k = 333.0f32;
    let tau = exact_mass_admit_into(&scores, k, 1e-3, &mut mask);
    assert!(tau.is_finite());
    assert!(
        (mask_sum(&mask) - k as f64).abs() <= sum_tol(n),
        "tiny T must still sum to k, got {}",
        mask_sum(&mask)
    );
    let binary = mask.iter().filter(|&&m| m > 0.99 || m < 0.01).count();
    assert!(
        binary >= n - 8,
        "T=1e-3 should be near-binary ({binary}/{n} saturated)"
    );
}

#[test]
fn allocating_wrapper_matches_into() {
    let scores = gen_scores(777, 0xA11C);
    let (owned, tau_a) = exact_mass_admit(&scores, 77.0, 2.0);
    let mut into = vec![0.0f32; 777];
    let tau_b = exact_mass_admit_into(&scores, 77.0, 2.0, &mut into);
    assert_eq!(tau_a, tau_b);
    assert_eq!(owned, into);
}

#[test]
fn iteration_cap_is_fifty() {
    assert_eq!(MAX_BISECT_ITERS, 50, "the paper's fixed cap; ≤ with early exits");
}

// ── log_frontier ──────────────────────────────────────────────────────

#[test]
fn log_frontier_sign_step_both_directions() {
    // The tracker starts AT the ceiling, so growth is only observable after
    // one shrink step. target 0.9, lr 0.05, every draw probes.
    let mut lf = LogFrontier::new(1024, 0.90, 0.05, 1.0, 1);
    lf.sample(0.5);
    let ceil_k = lf.observe(0.95); // acc > target → shrink, off the ceiling
    assert!(ceil_k < 1024.0, "shrink must move off the ceiling ({ceil_k})");

    lf.sample(0.5);
    let grow_k = lf.observe(0.85); // acc < target → grow
    assert!(
        grow_k > ceil_k,
        "acc < target must grow k_max ({grow_k} !> {ceil_k})"
    );

    lf.sample(0.5);
    let shrink_k = lf.observe(0.95); // acc > target → shrink
    assert!(
        shrink_k < grow_k,
        "acc > target must shrink k_max ({shrink_k} !< {grow_k})"
    );

    lf.sample(0.5);
    let hold_k = lf.observe(0.90); // acc == target → hold
    assert!(
        (hold_k - shrink_k).abs() < 1e-4,
        "acc == target must hold ({hold_k} vs {shrink_k})"
    );
}

#[test]
fn log_frontier_clamps_both_ends() {
    // Repeated "too accurate" probes drive k_max to the floor and STOP;
    // repeated "not accurate enough" probes drive it to total and STOP.
    let mut low = LogFrontier::new(1024, 0.90, 0.5, 1.0, 8);
    for _ in 0..64 {
        low.sample(0.42); // probe
        low.observe(0.99); // acc > target → shrink
    }
    assert!(
        (low.k_max() - 8.0).abs() < 1e-3,
        "floor clamp: k_max = {} want 8",
        low.k_max()
    );
    let mut high = LogFrontier::new(1024, 0.90, 0.5, 1.0, 8);
    // First move it off the ceiling so growth is observable, then push up.
    for _ in 0..4 {
        high.sample(0.42);
        high.observe(0.99);
    }
    assert!(high.k_max() < 1024.0, "sanity: moved off the ceiling");
    for _ in 0..64 {
        high.sample(0.42);
        high.observe(0.10); // acc < target → grow
    }
    assert!(
        (high.k_max() - 1024.0).abs() < 0.5,
        "ceil clamp: k_max = {} want 1024",
        high.k_max()
    );
}

#[test]
fn log_frontier_probe_cadence() {
    // probe_frac = 0.25 → every 4th draw is a probe AT k_max and arms
    // observe; the other three are log-uniform below it and disarm.
    let mut lf = LogFrontier::new(256, 0.9, 0.1, 0.25, 1);
    let k_max = lf.k_max();
    for step in 1..=8u32 {
        let k = lf.sample(0.5);
        if step % 4 == 0 {
            assert!((k - k_max).abs() < 1e-3, "step {step}: probe must return k_max");
            // Armed: observe must move (acc far below target, +lr).
            let after = lf.observe(0.0);
            assert!(
                after >= k_max - 1e-3,
                "step {step}: probe observe must be able to move (was at ceiling: {after})"
            );
        } else {
            assert!(
                k < k_max,
                "step {step}: non-probe must sample below k_max ({k} vs {k_max})"
            );
            let before = lf.k_max();
            lf.observe(0.0); // disarmed → no-op
            assert_eq!(
                lf.k_max(),
                before,
                "step {step}: non-probe observe must be a no-op"
            );
        }
    }
}

#[test]
fn log_frontier_probe_frac_zero_disarms() {
    let mut lf = LogFrontier::new(1024, 0.9, 0.1, 0.0, 1);
    for _ in 0..32 {
        let _ = lf.sample(0.3);
        lf.observe(0.0);
    }
    assert!(
        (lf.k_max() - 1024.0).abs() < 1e-3,
        "probe_frac=0 must never move the controller"
    );
}

#[test]
fn log_frontier_determinism_same_u_sequence() {
    // sample is a pure function of (state, u01): two trackers fed the same
    // u sequence produce identical k sequences AND identical trajectories.
    let us: Vec<f32> = {
        let mut rng = SplitMix64(0x0DE7);
        (0..256).map(|_| rng.next_unit()).collect()
    };
    let mut a = LogFrontier::new(512, 0.8, 0.07, 0.25, 4);
    let mut b = LogFrontier::new(512, 0.8, 0.07, 0.25, 4);
    for &u in &us {
        let ka = a.sample(u);
        let kb = b.sample(u);
        assert_eq!(ka.to_bits(), kb.to_bits());
        let oa = a.observe(0.75); // just below target: +lr when armed
        let ob = b.observe(0.75);
        assert_eq!(oa.to_bits(), ob.to_bits());
    }
    assert_eq!(a, b, "full tracker state must agree");
}

#[test]
fn log_frontier_sample_range_is_log_uniform_support() {
    let mut lf = LogFrontier::new(10_000, 0.9, 0.1, 0.0, 1);
    let mut rng = SplitMix64(0x5A3E);
    for _ in 0..1000 {
        let k = lf.sample(rng.next_unit());
        assert!((1.0..=10_000.0).contains(&k), "k out of [1, total]: {k}");
    }
}
