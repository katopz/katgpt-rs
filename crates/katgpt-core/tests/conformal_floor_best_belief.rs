//! Issue 010 T5 — "Report the Floor" comparison for Best-Belief Beta Selector
//! (Plan 336).
//!
//! The Best-Belief Beta Selector computes `BB_ε(S, F) = I⁻¹_ε(1+S, 1+F)` — the
//! ε-quantile of the Beta(1+S, 1+F) posterior — and selects the candidate with
//! the highest lower bound. T5 asks: **does the Beta prior (parametric,
//! regularized toward 0.5) beat the empirical MLE (raw rate S/(S+F), no
//! regularization) on selection quality?**
//!
//! ## Comparison angle (from Issue 010 T5)
//!
//! > "conservative candidate selection via Beta ε-quantile vs via empirical
//! > ε-quantile (the floor). Both are inverse-CDF reads; the question is
//! > whether the Beta prior (discrete, parametric) beats the empirical prior
//! > (continuous, nonparametric) on selection quality."
//!
//! This is NOT an interval-calibration comparison (CRPS/coverage/Winkler) —
//! it's a **selection-quality** comparison. The metric is **selection regret**:
//! `θ_best − θ_selected`, where `θ_best` is the highest true win-rate and
//! `θ_selected` is the true win-rate of the candidate each method picks.
//!
//! ## Why the empirical MLE is the honest floor
//!
//! The empirical ε-quantile of binary {0,1} outcomes is degenerate (it's 0 or
//! 1 depending on ε vs F/(S+F)). The non-degenerate empirical baseline is the
//! **MLE rate** `S/(S+F)` — the maximum-likelihood point estimate with no
//! confidence adjustment. This is the "pure exploitation" floor: it picks the
//! candidate that *looked* best in the data, with no regularization.
//!
//! The Best-Belief Beta lower bound adds two things the MLE lacks:
//! 1. **Beta(1,1) prior** — regularizes low-data candidates toward 0.5 (a
//!    candidate with 1/1 success has MLE=1.0 but BB_0.05 ≈ 0.025).
//! 2. **Conservatism** — the ε-quantile is a *lower* bound, penalizing
//!    candidates with high variance (few observations) more than the MLE.
//!
//! The question is whether these help (fewer high-variance false positives) or
//! hurt (over-cautious, misses genuinely good candidates).
//!
//! ## Expected result
//!
//! Best-Belief should WIN at low observation counts (where MLE over-fits noise
//! — a candidate with 2/2 successes looks perfect under MLE but mediocre under
//! Beta) and TIE at high observation counts (where both converge to the true
//! argmax). The crossover n characterizes the Beta prior's value.
//!
//! ## Run
//!
//! ```bash
//! cargo test -p katgpt-core --test conformal_floor_best_belief \
//!   --features conformal_predictive_intervals,best_belief -- --nocapture
//! ```

#![cfg(all(feature = "conformal_predictive_intervals", feature = "best_belief"))]

use katgpt_core::select_best_belief;

/// SplitMix64 — deterministic, seedable, no external dep. Matches the floor
/// harness's RNG constants for cross-test consistency.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform [0, 1) — half-open to avoid exact 1.0.
    #[inline]
    fn next_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0_f32 / (1u64 << 24) as f32)
    }
}

// ===== Empirical MLE floor selector =====

/// The empirical-MLE floor: select the candidate with the highest observed
/// success rate `S/(S+F)`. Ties broken by lowest index (matches
/// `select_best_belief`'s no-incumbent convention).
///
/// This is pure exploitation — no confidence adjustment, no regularization.
/// For a candidate with (S, F) = (1, 0), the MLE rate is 1.0; for (0, 1) it's
/// 0.0. Low-data candidates can have extreme rates.
fn select_mle_floor(candidates: &[(u32, u32)]) -> usize {
    assert!(!candidates.is_empty(), "empty candidates");
    let mut best_idx = 0;
    let mut best_rate = rate(candidates[0]);
    for (i, &(s, f)) in candidates.iter().enumerate().skip(1) {
        let r = rate((s, f));
        if r > best_rate {
            best_rate = r;
            best_idx = i;
        }
    }
    best_idx
}

/// Empirical success rate. The (0,0) case is treated as 0.5 (uninformed —
/// matches the Beta(1,1) prior mean, giving both methods the same neutral
/// starting point for unseen candidates).
#[inline]
fn rate(sf: (u32, u32)) -> f32 {
    let (s, f) = sf;
    let n = s + f;
    if n == 0 {
        0.5
    } else {
        s as f32 / n as f32
    }
}

// ===== Selection-regret simulation =====

/// One trial of the selection-regret experiment.
///
/// - `true_rates`: the true win-rates θ_i for each candidate (i = 0..K).
/// - `obs_budget`: total observations to distribute across candidates (the
///   key knob — UNIFORM distribution makes Beta tie MLE; SKEWED distribution
///   is where Beta's regularization earns its keep).
/// - `epsilon`: Best-Belief conservatism (only affects the Beta selector).
/// - `rng`: deterministic RNG.
///
/// Returns `(regret_beta, regret_mle)` — the selection regret for each method.
/// Regret = θ_best − θ_selected ∈ [0, 1]. Lower is better.
///
/// **Why observation counts vary per candidate:** with uniform n, the Beta
/// ε-quantile is a monotone transform of S/n → argmax is identical to MLE
/// → no selection difference. The Beta prior's value emerges ONLY when
/// candidates have different evidence weights (the real-world case: some
/// candidates have 2 trials, others have 200). This is the honest comparison.
fn run_trial(
    true_rates: &[f32],
    obs_per_candidate: &[u32],
    epsilon: f32,
    rng: &mut SplitMix64,
) -> (f32, f32) {
    // Draw observations: S_i ~ Binomial(n_i, θ_i), F_i = n_i − S_i.
    let candidates: Vec<(u32, u32)> = true_rates
        .iter()
        .zip(obs_per_candidate.iter())
        .map(|(&theta, &n)| {
            let mut s = 0u32;
            for _ in 0..n {
                if rng.next_unit() < theta {
                    s += 1;
                }
            }
            (s, n - s)
        })
        .collect();

    // Select with each method.
    let idx_beta = select_best_belief(&candidates, epsilon, None);
    let idx_mle = select_mle_floor(&candidates);

    // True best rate.
    let theta_best = true_rates.iter().cloned().fold(0.0_f32, f32::max);

    let regret_beta = theta_best - true_rates[idx_beta];
    let regret_mle = theta_best - true_rates[idx_mle];
    (regret_beta, regret_mle)
}

/// Observation-count distribution mode — the KEY experimental knob.
///
/// This is the central T5 finding: with uniform n, the Beta ε-quantile is a
/// monotone transform of S/n, so argmax is identical to MLE → no selection
/// difference. The Beta prior's value emerges ONLY when candidates have
/// different evidence weights (heteroscedastic data).
#[derive(Clone, Copy, Debug)]
enum ObsMode {
    /// Every candidate gets exactly `n_mean` observations. The degenerate
    /// case where Beta and MLE produce identical selections.
    Uniform { n_mean: u32 },
    /// Each candidate's n is drawn uniformly from [2, 2·n_mean]. The
    /// real-world case: some candidates have a few trials, others have many.
    /// This is where the Beta prior's regularization earns its keep.
    Variable { n_mean: u32 },
    /// One candidate (random index) gets very few observations (n_lo), the
    /// rest get n_mean. Stress-tests the false-positive regime: a low-data
    /// candidate can have a lucky streak (e.g. 2/2) that MLE treats as
    /// perfect but Beta discounts.
    OneLowData { n_mean: u32, n_lo: u32 },
}

/// Draw a per-candidate observation-count vector per the mode.
fn draw_obs_counts(mode: ObsMode, k: usize, rng: &mut SplitMix64) -> Vec<u32> {
    match mode {
        ObsMode::Uniform { n_mean } => vec![n_mean; k],
        ObsMode::Variable { n_mean } => {
            (0..k)
                .map(|_| 2 + (rng.next_unit() * (2.0 * n_mean as f32 - 1.0)) as u32)
                .collect()
        }
        ObsMode::OneLowData { n_mean, n_lo } => {
            let low_idx = (rng.next_unit() * k as f32) as usize % k;
            (0..k)
                .map(|i| if i == low_idx { n_lo } else { n_mean })
                .collect()
        }
    }
}

/// Run N_TRIALS trials and report mean regret for each method.
///
/// Each trial gets a fresh candidate pool (true rates re-drawn from the prior)
/// AND a fresh observation-count vector per `mode`.
fn run_experiment(
    k: usize,
    mode: ObsMode,
    epsilon: f32,
    n_trials: usize,
    seed: u64,
    theta_min: f32,
    theta_max: f32,
) -> (f32, f32) {
    let mut rng = SplitMix64::new(seed);
    let mut sum_beta = 0.0_f32;
    let mut sum_mle = 0.0_f32;
    for _ in 0..n_trials {
        // Draw true rates uniformly from [theta_min, theta_max].
        let true_rates: Vec<f32> = (0..k)
            .map(|_| theta_min + (theta_max - theta_min) * rng.next_unit())
            .collect();
        let obs_counts = draw_obs_counts(mode, k, &mut rng);
        let (rb, rm) = run_trial(&true_rates, &obs_counts, epsilon, &mut rng);
        sum_beta += rb;
        sum_mle += rm;
    }
    (sum_beta / n_trials as f32, sum_mle / n_trials as f32)
}

// ===== Tests =====

#[test]
fn uniform_n_produces_identical_selections_baseline() {
    // The foundational T5 finding: with uniform n per candidate, the Beta
    // ε-quantile is a monotone transform of S/n → argmax is identical to
    // MLE → ZERO selection difference. This is the degenerate baseline.
    let k = 8;
    let epsilon = 0.05;
    let n_trials = 5000;

    println!("\n=== T5 baseline: uniform n (K={}, ε={}, {} trials) ===", k, epsilon, n_trials);
    println!("{:>6} | {:>14} | {:>14} | verdict", "n", "regret_beta", "regret_mle");
    println!("{}", "-".repeat(54));

    for &n in &[4u32, 8, 16, 32, 64] {
        let (rb, rm) = run_experiment(
            k, ObsMode::Uniform { n_mean: n }, epsilon, n_trials, 0x1111, 0.3, 0.9,
        );
        let verdict = if (rb - rm).abs() < 1e-6 { "TIE (expected)" } else { "DIFF (unexpected!)" };
        println!("{:>6} | {:>14.6} | {:>14.6} | {}", n, rb, rm, verdict);
    }

    // Assert the baseline: at uniform n, regrets are identical (within float
    // noise). This confirms the monotonicity argument empirically.
    let (rb, rm) = run_experiment(
        k, ObsMode::Uniform { n_mean: 8 }, epsilon, n_trials, 0x1111, 0.3, 0.9,
    );
    assert!(
        (rb - rm).abs() < 1e-6,
        "uniform n must produce identical selections (got beta={:.6} vs mle={:.6})",
        rb, rm
    );
}

#[test]
fn beta_beats_mle_with_variable_observation_counts() {
    // The headline T5 test: when candidates have DIFFERENT observation counts
    // (heteroscedastic data — the real-world case), the Beta prior's
    // regularization reduces selection regret vs the MLE.
    let k = 8;
    let epsilon = 0.05;
    let n_trials = 5000;

    println!("\n=== T5: variable n (K={}, ε={}, {} trials) ===", k, epsilon, n_trials);
    println!(
        "{:>8} | {:>14} | {:>14} | {:>12} | verdict",
        "n_mean", "regret_beta", "regret_mle", "improvement"
    );
    println!("{}", "-".repeat(70));

    let mut beta_wins = 0;
    for &n_mean in &[4u32, 8, 16, 32, 64, 128] {
        let (rb, rm) = run_experiment(
            k, ObsMode::Variable { n_mean }, epsilon, n_trials, 0xDEAD_BEEF, 0.3, 0.9,
        );
        let improvement = (rm - rb) / rm.max(1e-9);
        let verdict = if rb < rm - 1e-6 {
            beta_wins += 1;
            "BETA WINS"
        } else if (rb - rm).abs() < 1e-6 {
            "tie"
        } else {
            "MLE wins"
        };
        println!(
            "{:>8} | {:>14.6} | {:>14.6} | {:>11.2}% | {}",
            n_mean, rb, rm, improvement * 100.0, verdict
        );
    }

    println!("\nBeta wins at {} of 6 variable-n levels.", beta_wins);
    // The honest expectation: Beta wins with variable n. We assert it wins
    // at low mean n (most heteroscedastic noise).
    let (rb_low, rm_low) = run_experiment(
        k, ObsMode::Variable { n_mean: 4 }, epsilon, n_trials, 0xDEAD_BEEF, 0.3, 0.9,
    );
    assert!(
        rb_low < rm_low,
        "Beta must beat MLE at variable n_mean=4 (got beta={:.6} vs mle={:.6})",
        rb_low, rm_low
    );
}

#[test]
fn beta_beats_mle_on_low_data_stress_test() {
    // The false-positive stress test: one candidate has very few observations
    // (n_lo=2), the rest have n_mean. With θ ∈ [0.3, 0.9], a 2/2 lucky streak
    // has MLE=1.0 (perfect), which MLE will pick over a genuinely better
    // candidate with 50/60 (MLE=0.833). Beta discounts the 2/2 candidate.
    let k = 8;
    let epsilon = 0.05;
    let n_trials = 5000;

    println!("\n=== T5: one-low-data stress (K={}, ε={}, {} trials) ===", k, epsilon, n_trials);
    println!(
        "{:>6} {:>4} | {:>14} | {:>14} | {:>12} | verdict",
        "n_mean", "n_lo", "regret_beta", "regret_mle", "improvement"
    );
    println!("{}", "-".repeat(74));

    for &(n_mean, n_lo) in &[(32u32, 2u32), (64, 2), (32, 4), (64, 4), (128, 2)] {
        let (rb, rm) = run_experiment(
            k, ObsMode::OneLowData { n_mean, n_lo }, epsilon, n_trials, 0xCAFE_F00D, 0.3, 0.9,
        );
        let improvement = (rm - rb) / rm.max(1e-9);
        let verdict = if rb < rm - 1e-6 { "BETA WINS" }
                      else if (rb - rm).abs() < 1e-6 { "tie" }
                      else { "MLE wins" };
        println!(
            "{:>6} {:>4} | {:>14.6} | {:>14.6} | {:>11.2}% | {}",
            n_mean, n_lo, rb, rm, improvement * 100.0, verdict
        );
    }

    // The n_lo=2 case is the sharpest test: a 2/2 lucky streak is pure noise.
    let (rb, rm) = run_experiment(
        k, ObsMode::OneLowData { n_mean: 32, n_lo: 2 }, epsilon, n_trials, 0xCAFE_F00D, 0.3, 0.9,
    );
    assert!(
        rb < rm,
        "Beta must beat MLE on the n_lo=2 stress test (got beta={:.6} vs mle={:.6})",
        rb, rm
    );
}

#[test]
fn beta_and_mle_converge_at_high_observation_count() {
    // At high n_mean (variable mode), both selectors converge — the
    // low-data candidates that cause MLE false positives become rare.
    let k = 4;
    let epsilon = 0.05;
    let n_trials = 3000;

    let (rb, rm) = run_experiment(
        k, ObsMode::Variable { n_mean: 512 }, epsilon, n_trials, 0xCAFE_F00D, 0.3, 0.9,
    );

    println!("\n=== T5 convergence (K={}, Variable n_mean=512, ε={}, {} trials) ===", k, epsilon, n_trials);
    println!("  regret_beta = {:.6}", rb);
    println!("  regret_mle  = {:.6}", rm);

    // Both should be small (enough data on average).
    assert!(rb < 0.05, "Beta regret should be small at n_mean=512 (got {:.4})", rb);
    assert!(rm < 0.05, "MLE regret should be small at n_mean=512 (got {:.4})", rm);
}

#[test]
fn beta_conservatism_sweep_variable_n() {
    // How does ε affect selection quality with variable n? Lower ε = more
    // conservative = stronger discounting of low-data candidates.
    let k = 8;
    let n_mean = 8;
    let n_trials = 3000;

    println!("\n=== T5: ε sweep (K={}, Variable n_mean={}, {} trials) ===", k, n_mean, n_trials);
    println!("{:>8} | {:>14} | {:>14} | verdict", "ε", "regret_beta", "regret_mle");
    println!("{}", "-".repeat(58));

    for &eps in &[0.01_f32, 0.05, 0.10, 0.20, 0.50] {
        let (rb, rm) = run_experiment(
            k, ObsMode::Variable { n_mean }, eps, n_trials, 0xFEED_FACE, 0.3, 0.9,
        );
        let verdict = if rb < rm - 1e-6 { "BETA WINS" }
                      else if (rb - rm).abs() < 1e-6 { "tie" }
                      else { "MLE wins" };
        println!("{:>8.2} | {:>14.6} | {:>14.6} | {}", eps, rb, rm, verdict);
    }
    // Descriptive only — no threshold assertion.
}

#[test]
fn beta_full_report_for_benchmark_doc() {
    // The canonical evidence run — prints the full sweep for the
    // `.benchmarks/010_best_belief_floor_comparison.md` writeup.
    println!("\n\n========================================");
    println!("=== Best-Belief Beta Selector Floor Comparison (Issue 010 T5) ===");
    println!("========================================\n");

    let epsilon = 0.05;
    let n_trials = 5000;
    let k = 8;

    println!("## Selection regret (θ_best − θ_selected), lower is better\n");
    println!("### K={} candidates, θ ∈ [0.3, 0.9], {} trials\n", k, n_trials);

    println!("\n--- Uniform n (baseline: Beta should TIE MLE) ---\n");
    println!("{:>6} | {:>14} | {:>14} | verdict", "n", "regret_beta", "regret_mle");
    println!("{}", "-".repeat(54));
    for &n in &[4u32, 8, 16, 32, 64] {
        let (rb, rm) = run_experiment(
            k, ObsMode::Uniform { n_mean: n }, epsilon, n_trials, 0x1111, 0.3, 0.9,
        );
        let verdict = if (rb - rm).abs() < 1e-6 { "TIE" } else { "DIFF" };
        println!("{:>6} | {:>14.6} | {:>14.6} | {}", n, rb, rm, verdict);
    }

    println!("\n--- Variable n (real-world: Beta should WIN at low n_mean) ---\n");
    println!(
        "{:>8} | {:>14} | {:>14} | {:>12} | verdict",
        "n_mean", "regret_beta", "regret_mle", "improvement"
    );
    println!("{}", "-".repeat(70));
    for &n_mean in &[4u32, 8, 16, 32, 64, 128, 256] {
        let (rb, rm) = run_experiment(
            k, ObsMode::Variable { n_mean }, epsilon, n_trials, 0xDEAD_BEEF, 0.3, 0.9,
        );
        let improvement = (rm - rb) / rm.max(1e-9);
        let verdict = if rb < rm - 1e-6 { "BETA WINS" }
                      else if (rb - rm).abs() < 1e-6 { "tie" }
                      else { "MLE wins" };
        println!(
            "{:>8} | {:>14.6} | {:>14.6} | {:>11.2}% | {}",
            n_mean, rb, rm, improvement * 100.0, verdict
        );
    }
}
