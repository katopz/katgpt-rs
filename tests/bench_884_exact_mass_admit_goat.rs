//! Bench 884 — `exact_mass_admit` GOAT gate (Issue 879 T3 / Research 584,
//! arXiv:2609.25518 "Matryoshka attribution").
//!
//! The claim under test is NOT "exact mass is faster" — it is not, by
//! construction (~30-50 sigmoid passes vs one): the gate records the honest
//! **cost-vs-calibration tradeoff** of the operator against the hard-cut
//! baseline family the workspace ships today, at the issue's N ∈ {1e3,
//! 1e5, 1e7}:
//!
//! - **ours** — `katgpt_core::exact_mass_admit::exact_mass_admit_into`
//!   (calibrated mass: `Σm = k` by construction).
//! - **gate** — `katgpt_spectral::manifold_power_iter_router::
//!   gate_sigmoid_topk_into` (per-expert sigmoids + selection-sort hard
//!   cut, UNCALIBRATED mass). Its own doc names the hot N≤64 /
//!   game-scale-≤256 regime and the O(k·N) selection sort — this bench
//!   runs it ONLY at N=1e3 (k=100, ~1e7 compare-swaps/call), its designed
//!   neighborhood, and records why it is absent at larger N.
//! - **hardcut** — `select_nth_unstable` + sigmoid over the top-k (the
//!   generic arbitrary-k hard cut; UNCALIBRATED mass).
//!
//! Gates:
//! - **G1** (spot, in-process): sum-to-k at every N; masks in [0,1]. The
//!   heavy G1 suite lives in katgpt-core's `exact_mass_admit_g1`.
//! - **G2** (record + calibration assert — the no-assumed-win rule): per-arm
//!   µs/call via the shared `best_of_arms` harness (round-robin interleave +
//!   per-arm minimum — the load-invariant form), plus the **calibration
//!   column** `|Σm − k|` per arm — the quantity only `ours` controls, and
//!   the only hard assert. A single GROSS regression ceiling on `ours`
//!   (2.5 s at N=1e7, ~2.5× the measured 1.1 s) catches de-optimization
//!   without pretending to be a measurement on a shared box.
//! - **G3**: feature-off posture — nothing outside the feature gates on
//!   `exact_mass_admit`; katgpt-core's default build is unchanged (the
//!   `#![cfg]` binary would be empty — the Issue-808 green-zero class is
//!   why the Cargo.toml row carries required-features).
//! - **G4**: rides katgpt-core's `exact_mass_admit_g4_alloc`.
//!
//! # Run
//!
//! ```bash
//! cargo test --features exact_mass_admit \
//!   --test bench_884_exact_mass_admit_goat --release -- --nocapture
//! ```

#![cfg(feature = "exact_mass_admit")]

#[path = "common/ab_timing.rs"]
mod ab_timing;

use ab_timing::best_of_arms;
use katgpt_core::exact_mass_admit::exact_mass_admit_into;
use katgpt_core::simd::exact_sigmoid_f64;
use katgpt_spectral::manifold_power_iter_router::gate_sigmoid_topk_into;
use std::hint::black_box;

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
    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn gen_scores(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64(seed);
    (0..n).map(|_| rng.next_unit() * 6.0 - 3.0).collect()
}

fn mask_sum(mask: &[f32]) -> f64 {
    mask.iter().map(|&m| m as f64).sum()
}

fn sum_tol(n: usize) -> f64 {
    1e-3 + 2.0e-7 * n as f64
}

/// The hard-cut baseline: `select_nth_unstable` top-k + sigmoid weights on
/// the winners (uncalibrated mass — Σ is whatever the sigmoids sum to).
/// Continuous random scores make threshold ties measure-zero, so
/// `s >= thresh` admits exactly k.
fn hardcut_sigmoid_into(scores: &[f32], k: usize, out: &mut [f32], scratch: &mut [f32]) {
    scratch.copy_from_slice(scores);
    scratch.select_nth_unstable_by(
        k - 1,
        |a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal),
    );
    let thresh = scratch[k - 1];
    for (o, &s) in out.iter_mut().zip(scores) {
        *o = if s >= thresh {
            exact_sigmoid_f64(s as f64) as f32
        } else {
            0.0
        };
    }
}

/// G1 at scale — calibration must hold at every N the G2 table records.
#[test]
fn g1_calibration_at_scale() {
    for &n in &[1_000usize, 100_000, 10_000_000] {
        let scores = gen_scores(n, 0x884 + n as u64);
        let k = (n / 10) as f32;
        let mut mask = vec![0.0f32; n];
        let tau = exact_mass_admit_into(black_box(&scores), k, 1.0, &mut mask);
        assert!(tau.is_finite(), "N={n}: tau must be finite");
        let err = (mask_sum(&mask) - k as f64).abs();
        assert!(
            err <= sum_tol(n),
            "N={n}: |Σm−k| = {err:.3e} > {} — calibration broken",
            sum_tol(n)
        );
        for &m in &mask {
            assert!((0.0..=1.0).contains(&m), "N={n}: mask out of [0,1]: {m}");
        }
        println!("G1 N={n}: |Σm−k| = {err:.3e} (k = {k})");
    }
}

/// G2 — the cost + calibration table. Relative latency is RECORDED, never
/// asserted (the no-assumed-win rule); the two hard asserts are (a) ours
/// pins the mass at every N and (b) the gross regression ceiling.
#[test]
fn g2_cost_vs_calibration_table() {
    println!("bench_884 G2 — best-of-arms = per-arm MIN over round-robin samples (load-invariant form)");

    // N = 1e3: three arms (gate is in its designed regime here only).
    {
        let n = 1_000usize;
        let k = n / 10;
        let scores = gen_scores(n, 0x1884);
        let mut mask = vec![0.0f32; n];
        let mut gate_scores = vec![0.0f32; n];
        let mut gate_idx = vec![0usize; n];
        let x = [1.0f32]; // d_model = 1: per-expert score = σ(x · s) = σ(s)
        let mut scratch = vec![0.0f32; n];

        // Calibration column (off the clock).
        let mut errs = Vec::new();
        exact_mass_admit_into(&scores, k as f32, 1.0, &mut mask);
        errs.push((mask_sum(&mask) - k as f64).abs());
        let kk = gate_sigmoid_topk_into(
            &x, &scores, n, 1, 1.0, k, &mut gate_scores, &mut gate_idx,
        );
        let gate_mass: f64 = gate_idx[..kk].iter().map(|&i| gate_scores[i] as f64).sum();
        errs.push((gate_mass - k as f64).abs());
        hardcut_sigmoid_into(&scores, k, &mut mask, &mut scratch);
        errs.push((mask_sum(&mask) - k as f64).abs());

        let us = best_of_arms(3, 1, 5, |arm| {
            let t = std::time::Instant::now();
            match arm {
                0 => {
                    exact_mass_admit_into(black_box(&scores), k as f32, 1.0, &mut mask);
                }
                1 => {
                    let kk = gate_sigmoid_topk_into(
                        black_box(&x), black_box(&scores), n, 1, 1.0, k,
                        &mut gate_scores, &mut gate_idx,
                    );
                    black_box(kk);
                }
                _ => {
                    hardcut_sigmoid_into(black_box(&scores), k, &mut mask, &mut scratch);
                }
            }
            black_box(mask[0]);
            t.elapsed()
        });
        let labels = ["ours", "gate", "hardcut"];
        println!("\n┌─────────┬──────────┬────────────────┬──────────────────────┐");
        println!("│    N    │   arm    │  best µs/call  │  |Σm − k| (calib)    │");
        println!("├─────────┼──────────┼────────────────┼──────────────────────┤");
        for (i, label) in labels.iter().enumerate() {
            println!(
                "│ {:>7} │ {:<8} │ {:>14.2} │ {:>20.4e} │",
                n, label, us[i], errs[i]
            );
        }
        assert!(
            errs[0] <= sum_tol(n),
            "G2 N={n}: ours |Σm−k| = {:.4e} — calibration broken",
            errs[0]
        );
    }

    // N ∈ {1e5, 1e7}: ours + hardcut.
    let mut ours_1e7_us = f64::MAX;
    for &n in &[100_000usize, 10_000_000] {
        let k = n / 10;
        let scores = gen_scores(n, 0x2884 + n as u64);
        let mut mask = vec![0.0f32; n];
        let mut scratch = vec![0.0f32; n];

        let mut errs = Vec::new();
        exact_mass_admit_into(&scores, k as f32, 1.0, &mut mask);
        errs.push((mask_sum(&mask) - k as f64).abs());
        hardcut_sigmoid_into(&scores, k, &mut mask, &mut scratch);
        errs.push((mask_sum(&mask) - k as f64).abs());

        // At 1e7 a full 5-sample round-robin over ours would burn ~25 s of
        // exp calls; 2 timed samples still give the per-arm MIN.
        let iters = if n >= 10_000_000 { 2 } else { 5 };
        let us = best_of_arms(2, 1, iters, |arm| {
            let t = std::time::Instant::now();
            match arm {
                0 => {
                    exact_mass_admit_into(black_box(&scores), k as f32, 1.0, &mut mask);
                }
                _ => {
                    hardcut_sigmoid_into(black_box(&scores), k, &mut mask, &mut scratch);
                }
            }
            black_box(mask[0]);
            t.elapsed()
        });
        let labels = ["ours", "hardcut"];
        println!("├─────────┼──────────┼────────────────┼──────────────────────┤");
        for (i, label) in labels.iter().enumerate() {
            println!(
                "│ {:>7} │ {:<8} │ {:>14.2} │ {:>20.4e} │",
                n, label, us[i], errs[i]
            );
        }
        assert!(
            errs[0] <= sum_tol(n),
            "G2 N={n}: ours |Σm−k| = {:.4e} — calibration broken",
            errs[0]
        );
        if n == 10_000_000 {
            ours_1e7_us = us[0];
        }
    }
    println!("└─────────┴──────────┴────────────────┴──────────────────────┘");
    println!("(gate arm at N≥1e5: absent — its own doc bounds the O(k·N) selection sort to game-scale N; absence-of-regime, not a result)");

    // The only hard perf assertion: a GROSS regression ceiling on ours
    // (measured 1.1 s/call at N=1e7 — ~2.2 ns per expit·element across
    // ~32-50 bisection passes; the ceiling sits at ~2.5× the measured cost
    // and catches de-optimization, not box noise — the bench-845 pattern).
    println!("G2: ours @1e7 = {ours_1e7_us:.0} µs/call (gross ceiling 2_500_000 µs)");
    assert!(
        ours_1e7_us < 2_500_000.0,
        "G2 gross regression: exact_mass_admit_into @1e7 = {ours_1e7_us:.0} µs > 2.5 s ceiling"
    );
    println!("G2: PASS — calibration pinned at every N; cost recorded as-is (no relative-perf assert, the no-assumed-win rule)");
}
