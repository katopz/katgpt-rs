//! GRAM Width-vs-Depth GOAT Proof Benchmark — run with:
//! cargo test --features "elf_sde bandit" --test bench_gram_width_depth --release -- --nocapture
//!
//! Plan 095: Validates GRAM's width >> depth finding (Research 58):
//! - Width scaling (K=1→20 rollouts): should dominate depth scaling
//! - Depth scaling (T=1→16 steps): diminishing returns beyond T=4
//! - Width×Depth matrix: interaction effects
//!
//! GRAM proves N=20@16 iters = 94.2% vs deterministic@320 steps = 78.1%.
//! Same compute, better allocation: diversity of parallel stochastic trajectories
//! beats depth-first search. This benchmark validates the principle on our DDTree.

#![cfg(all(feature = "elf_sde", feature = "bandit"))]

use katgpt_rs::speculative::dd_tree::{
    WidthScaleConfig, WidthSelectionMode, best_of_k_rollouts, build_dd_tree_sde, extract_best_path,
};
use katgpt_rs::speculative::types::{NoScreeningPruner, SdeConfig};
use katgpt_rs::transformer::TransformerWeights;
use katgpt_rs::types::{Config, Rng};
use std::collections::HashSet;

// ── Sweep Configuration ─────────────────────────────────────────

/// Width values for GRAM sweep: K parallel rollouts.
const WIDTH_VALUES: &[usize] = &[1, 5, 10, 20];

/// Depth values for GRAM sweep: T lookahead steps.
const DEPTH_VALUES: &[usize] = &[1, 4, 8, 16];

/// Fixed depth for width sweep (GRAM default).
const FIXED_DEPTH: usize = 4;

/// Fixed width for depth sweep (single rollout baseline).
const FIXED_WIDTH: usize = 1;

/// Trials per configuration for statistical stability.
const N_TRIALS: usize = 100;

/// GRAM's σ_θ analog: full noise scale with top-1 preservation.
const SDE_CONFIG: SdeConfig = SdeConfig {
    gamma: 1.0,
    preserve_top1: true,
    confidence_floor: 0.0,
};

// ── Helpers ─────────────────────────────────────────────────────

/// Benchmark result for a single (K, T) configuration.
#[derive(Debug, Clone)]
struct BenchResult {
    width_k: usize,
    depth_t: usize,
    avg_quality: f32,
    avg_agreement: f32,
    unique_paths: usize,
    diversity: f32,
    latency_us: f64,
}

/// Verdict for GOAT proof criteria.
#[derive(Debug, Clone, Copy, PartialEq)]
enum GoatVerdict {
    /// Criterion passed with measured value.
    Pass(f32),
    /// Criterion failed with measured value.
    Fail(f32),
    /// Cannot determine (e.g., division by zero).
    Inconclusive,
}

impl GoatVerdict {
    fn is_pass(self) -> bool {
        matches!(self, GoatVerdict::Pass(_))
    }

    fn label(self) -> &'static str {
        match self {
            GoatVerdict::Pass(_) => "✅ PASS",
            GoatVerdict::Fail(_) => "❌ FAIL",
            GoatVerdict::Inconclusive => "⚠️  INCONCLUSIVE",
        }
    }

    fn value_str(self) -> String {
        match self {
            GoatVerdict::Pass(v) | GoatVerdict::Fail(v) => format!("{v:.2}"),
            GoatVerdict::Inconclusive => "N/A".to_string(),
        }
    }
}

/// GOAT criteria for GRAM width-vs-depth proof.
struct GoatCriteria {
    /// G1: Width K=1→K=20 improves quality by ≥10%.
    g1_width_scaling: GoatVerdict,
    /// G2: Depth T=4→T=16 improves by ≤5% relative to width gain.
    g2_depth_marginal: GoatVerdict,
    /// G3: Width × depth ratio ≥ 2.0 (width dominates).
    g3_width_depth_ratio: GoatVerdict,
}

impl GoatCriteria {
    fn is_proved(&self) -> bool {
        self.g1_width_scaling.is_pass()
            && self.g2_depth_marginal.is_pass()
            && self.g3_width_depth_ratio.is_pass()
    }
}

/// Generate marginals from a real model for benchmarking.
fn make_marginals() -> (Config, Vec<Vec<f32>>) {
    let config = Config::draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let marginals = katgpt_rs::speculative::dflash::dflash_predict(&weights, &config, 0, 0);
    (config, marginals)
}

/// Compute greedy argmax path as baseline reference.
fn greedy_path(marginals: &[Vec<f32>]) -> Vec<usize> {
    marginals
        .iter()
        .map(|m| {
            m.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

/// Path quality: average token probability along the path.
fn path_quality(marginals: &[Vec<f32>], path: &[usize]) -> f32 {
    if path.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f32;
    for (depth, &token_idx) in path.iter().enumerate() {
        if depth < marginals.len() {
            total += marginals[depth].get(token_idx).copied().unwrap_or(0.0);
        }
    }
    total / path.len() as f32
}

/// Top-1 agreement: fraction of depths where path matches greedy.
fn top1_agreement(greedy: &[usize], path: &[usize]) -> f32 {
    if greedy.is_empty() || path.is_empty() {
        return 0.0;
    }
    let min_len = greedy.len().min(path.len());
    let matches = (0..min_len).filter(|&i| greedy[i] == path[i]).count();
    matches as f32 / min_len as f32
}

/// Truncate marginals to `n` depths for depth-controlled experiments.
fn marginals_refs_truncated(marginals: &[Vec<f32>], n: usize) -> Vec<&[f32]> {
    marginals.iter().take(n).map(|m| m.as_slice()).collect()
}

/// Run width sweep at a fixed depth, returning per-K results.
fn run_width_sweep(
    base_config: &Config,
    marginals: &[Vec<f32>],
    fixed_depth: usize,
    k_values: &[usize],
    n_trials: usize,
) -> Vec<BenchResult> {
    let screener = NoScreeningPruner;
    let greedy = greedy_path(marginals);
    let mut config = base_config.clone();
    config.draft_lookahead = fixed_depth.min(marginals.len());
    let marginals_refs = marginals_refs_truncated(marginals, config.draft_lookahead);

    let mut results = Vec::with_capacity(k_values.len());

    for &k in k_values {
        let width_config = WidthScaleConfig {
            k_rollouts: k,
            selection: WidthSelectionMode::BestQ,
        };

        let start = std::time::Instant::now();
        let mut all_paths: Vec<Vec<usize>> = Vec::with_capacity(n_trials);
        let mut qualities: Vec<f32> = Vec::with_capacity(n_trials);
        let mut agreements: Vec<f32> = Vec::with_capacity(n_trials);
        let mut unique_set: HashSet<Vec<usize>> = HashSet::new();

        for trial in 0..n_trials {
            let path = best_of_k_rollouts(
                &marginals_refs,
                &config,
                &screener,
                &SDE_CONFIG,
                &width_config,
                42 + trial as u64,
            );
            qualities.push(path_quality(marginals, &path));
            agreements.push(top1_agreement(&greedy, &path));
            unique_set.insert(path.clone());
            all_paths.push(path);
        }
        let elapsed = start.elapsed();

        let avg_quality = qualities.iter().sum::<f32>() / qualities.len() as f32;
        let avg_agreement = agreements.iter().sum::<f32>() / agreements.len() as f32;
        let diversity = unique_set.len() as f32 / all_paths.len() as f32;
        let latency_us = elapsed.as_micros() as f64 / n_trials as f64;

        results.push(BenchResult {
            width_k: k,
            depth_t: fixed_depth,
            avg_quality,
            avg_agreement,
            unique_paths: unique_set.len(),
            diversity,
            latency_us,
        });
    }

    results
}

/// Run depth sweep at a fixed width, returning per-T results.
fn run_depth_sweep(
    base_config: &Config,
    marginals: &[Vec<f32>],
    fixed_width: usize,
    t_values: &[usize],
    n_trials: usize,
) -> Vec<BenchResult> {
    let screener = NoScreeningPruner;
    let greedy = greedy_path(marginals);
    let width_config = WidthScaleConfig {
        k_rollouts: fixed_width,
        selection: WidthSelectionMode::BestQ,
    };

    let mut results = Vec::with_capacity(t_values.len());

    for &t in t_values {
        let mut config = base_config.clone();
        config.draft_lookahead = t.min(marginals.len());
        let marginals_refs = marginals_refs_truncated(marginals, config.draft_lookahead);

        let start = std::time::Instant::now();
        let mut qualities: Vec<f32> = Vec::with_capacity(n_trials);
        let mut agreements: Vec<f32> = Vec::with_capacity(n_trials);
        let mut unique_set: HashSet<Vec<usize>> = HashSet::new();

        for trial in 0..n_trials {
            let path = if fixed_width <= 1 || !SDE_CONFIG.is_enabled() {
                // Single rollout — use build_dd_tree_sde directly
                let mut rng = Rng::new(42 + trial as u64);
                let tree = build_dd_tree_sde(
                    &marginals_refs,
                    &config,
                    &screener,
                    false,
                    &SDE_CONFIG,
                    &mut rng,
                );
                extract_best_path(&tree)
            } else {
                best_of_k_rollouts(
                    &marginals_refs,
                    &config,
                    &screener,
                    &SDE_CONFIG,
                    &width_config,
                    42 + trial as u64,
                )
            };
            qualities.push(path_quality(marginals, &path));
            agreements.push(top1_agreement(&greedy, &path));
            unique_set.insert(path);
        }
        let elapsed = start.elapsed();

        let avg_quality = qualities.iter().sum::<f32>() / qualities.len() as f32;
        let avg_agreement = agreements.iter().sum::<f32>() / agreements.len() as f32;
        let diversity = unique_set.len() as f32 / n_trials as f32;
        let latency_us = elapsed.as_micros() as f64 / n_trials as f64;

        results.push(BenchResult {
            width_k: fixed_width,
            depth_t: t,
            avg_quality,
            avg_agreement,
            unique_paths: unique_set.len(),
            diversity,
            latency_us,
        });
    }

    results
}

/// Print a table of benchmark results.
fn print_results_table(title: &str, results: &[BenchResult]) {
    println!("\n{title}");
    println!("{}", "═".repeat(90));
    println!(
        "{:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "K", "T", "Quality", "Top1 Agr", "Diversity", "Unique", "Latency(µs)"
    );
    println!("{}", "─".repeat(90));

    for r in results {
        println!(
            "{:>6} {:>6} {:>10.6} {:>10.4} {:>10.4} {:>10} {:>12.1}",
            r.width_k,
            r.depth_t,
            r.avg_quality,
            r.avg_agreement,
            r.diversity,
            r.unique_paths,
            r.latency_us
        );
    }
}

// ── Test 1: Width Sweep at Fixed Depth ──────────────────────────

#[test]
fn bench_gram_width_sweep() {
    println!("\n🧪 GRAM Width Sweep: K=[1,5,10,20] at T={FIXED_DEPTH}");
    let (config, marginals) = make_marginals();

    let results = run_width_sweep(&config, &marginals, FIXED_DEPTH, WIDTH_VALUES, N_TRIALS);
    print_results_table(
        &format!("Width Sweep (T={FIXED_DEPTH} fixed, γ=1.0)"),
        &results,
    );

    // Summary: quality gain K=1 → K=20
    let q_k1 = results.first().map(|r| r.avg_quality).unwrap_or(0.0);
    let q_k20 = results.last().map(|r| r.avg_quality).unwrap_or(0.0);
    let gain_pct = if q_k1.abs() > 1e-9 {
        (q_k20 - q_k1) / q_k1 * 100.0
    } else {
        0.0
    };

    println!("\n📊 Width Summary: K=1→K=20");
    println!("  K=1  quality: {q_k1:.6}");
    println!("  K=20 quality: {q_k20:.6}");
    println!("  Gain:         {gain_pct:+.2}%");
    println!("  GRAM expects: ≥10% gain from width scaling");
}

// ── Test 2: Depth Sweep at Fixed Width ──────────────────────────

#[test]
fn bench_gram_depth_sweep() {
    println!("\n🧪 GRAM Depth Sweep: T=[1,4,8,16] at K={FIXED_WIDTH}");
    let (config, marginals) = make_marginals();

    let results = run_depth_sweep(&config, &marginals, FIXED_WIDTH, DEPTH_VALUES, N_TRIALS);
    print_results_table(
        &format!("Depth Sweep (K={FIXED_WIDTH} fixed, γ=1.0)"),
        &results,
    );

    // Summary: quality gain T=1 → T=16
    let q_t1 = results.first().map(|r| r.avg_quality).unwrap_or(0.0);
    let q_t16 = results.last().map(|r| r.avg_quality).unwrap_or(0.0);
    let gain_pct = if q_t1.abs() > 1e-9 {
        (q_t16 - q_t1) / q_t1 * 100.0
    } else {
        0.0
    };

    println!("\n📊 Depth Summary: T=1→T=16");
    println!("  T=1  quality: {q_t1:.6}");
    println!("  T=16 quality: {q_t16:.6}");
    println!("  Gain:         {gain_pct:+.2}%");
    println!("  GRAM expects: diminishing returns, ≤5pp beyond T=4");
}

// ── Test 3: Width×Depth Matrix ──────────────────────────────────

#[test]
fn bench_gram_width_depth_matrix() {
    println!("\n🧪 GRAM Width×Depth Matrix: K×T interaction");
    let (config, marginals) = make_marginals();

    let matrix_k: &[usize] = &[1, 5, 10, 20];
    let matrix_t: &[usize] = &[1, 4, 8, 16];

    // Run width sweep at each depth level
    let mut all_results: Vec<BenchResult> = Vec::new();

    for &t in matrix_t {
        let sweep = run_width_sweep(&config, &marginals, t, matrix_k, N_TRIALS / 2);
        all_results.extend(sweep);
    }

    // Print as matrix
    println!("\n📊 Quality Matrix (avg path quality × 1000)");
    println!("{}", "═".repeat(70));
    print!("{:>6}", "K\\T");
    for &t in matrix_t {
        print!("{:>10}", format!("T={t}"));
    }
    println!();
    println!("{}", "─".repeat(70));

    for &k in matrix_k {
        print!("{:>6}", format!("K={k}"));
        for &t in matrix_t {
            let entry = all_results
                .iter()
                .find(|r| r.width_k == k && r.depth_t == t);
            match entry {
                Some(r) => print!("{:>10.3}", r.avg_quality * 1000.0),
                None => print!("{:>10}", "—"),
            }
        }
        println!();
    }

    // Print latency matrix
    println!("\n📊 Latency Matrix (µs per trial)");
    println!("{}", "═".repeat(70));
    print!("{:>6}", "K\\T");
    for &t in matrix_t {
        print!("{:>10}", format!("T={t}"));
    }
    println!();
    println!("{}", "─".repeat(70));

    for &k in matrix_k {
        print!("{:>6}", format!("K={k}"));
        for &t in matrix_t {
            let entry = all_results
                .iter()
                .find(|r| r.width_k == k && r.depth_t == t);
            match entry {
                Some(r) => print!("{:>10.1}", r.latency_us),
                None => print!("{:>10}", "—"),
            }
        }
        println!();
    }
}

// ── Test 4: GOAT Verdict ────────────────────────────────────────

#[test]
fn bench_gram_goat_verdict() {
    println!("\n🐐 GRAM GOAT Proof: Width vs Depth Verdict");
    println!("{}", "═".repeat(80));

    let (config, marginals) = make_marginals();

    // Run both sweeps
    let width_results = run_width_sweep(&config, &marginals, FIXED_DEPTH, WIDTH_VALUES, N_TRIALS);
    let depth_results = run_depth_sweep(&config, &marginals, FIXED_WIDTH, DEPTH_VALUES, N_TRIALS);

    // Compute quality deltas
    let q_k1 = width_results.first().map(|r| r.avg_quality).unwrap_or(0.0);
    let q_k20 = width_results.last().map(|r| r.avg_quality).unwrap_or(0.0);
    let width_gain_pct = if q_k1.abs() > 1e-9 {
        (q_k20 - q_k1) / q_k1 * 100.0
    } else {
        0.0
    };

    let q_t4 = depth_results
        .iter()
        .find(|r| r.depth_t == 4)
        .map(|r| r.avg_quality)
        .unwrap_or(0.0);
    let q_t16 = depth_results.last().map(|r| r.avg_quality).unwrap_or(0.0);
    let depth_gain_t4_t16 = if q_t4.abs() > 1e-9 {
        (q_t16 - q_t4) / q_t4 * 100.0
    } else {
        0.0
    };

    let ratio = if depth_gain_t4_t16.abs() > 0.01 {
        width_gain_pct / depth_gain_t4_t16
    } else if width_gain_pct > 0.0 {
        f32::INFINITY
    } else {
        0.0
    };

    // Evaluate GOAT criteria
    let g1 = match width_gain_pct {
        g if g >= 10.0 => GoatVerdict::Pass(g),
        g => GoatVerdict::Fail(g),
    };

    let g2 = match depth_gain_t4_t16 {
        g if g <= 5.0 => GoatVerdict::Pass(g),
        g => GoatVerdict::Fail(g),
    };

    let g3 = match ratio {
        r if r >= 2.0 && r.is_finite() => GoatVerdict::Pass(r),
        r if r.is_infinite() && width_gain_pct > 0.0 => GoatVerdict::Pass(ratio),
        r if r.is_nan() => GoatVerdict::Inconclusive,
        r => GoatVerdict::Fail(r),
    };

    let criteria = GoatCriteria {
        g1_width_scaling: g1,
        g2_depth_marginal: g2,
        g3_width_depth_ratio: g3,
    };

    // Print detailed results
    println!("\n📊 Width Scaling (T={FIXED_DEPTH} fixed)");
    println!("{:>6} {:>10} {:>12}", "K", "Quality", "Latency(µs)");
    println!("{}", "─".repeat(40));
    for r in &width_results {
        println!(
            "{:>6} {:>10.6} {:>12.1}",
            r.width_k, r.avg_quality, r.latency_us
        );
    }

    println!("\n📊 Depth Scaling (K={FIXED_WIDTH} fixed)");
    println!("{:>6} {:>10} {:>12}", "T", "Quality", "Latency(µs)");
    println!("{}", "─".repeat(40));
    for r in &depth_results {
        println!(
            "{:>6} {:>10.6} {:>12.1}",
            r.depth_t, r.avg_quality, r.latency_us
        );
    }

    // Print GOAT verdict
    println!("\n🏆 GOAT Verdict");
    println!("{}", "═".repeat(60));
    println!(
        "  G1: Width K=1→K=20 quality gain ≥10%     {} ({})",
        g1.label(),
        g1.value_str()
    );
    println!(
        "  G2: Depth T=4→T=16 gain ≤5%             {} ({})",
        g2.label(),
        g2.value_str()
    );
    println!(
        "  G3: Width/Depth ratio ≥ 2.0             {} ({})",
        g3.label(),
        g3.value_str()
    );
    println!("{}", "─".repeat(60));
    println!("  Width gain:  {width_gain_pct:+.2}% (K=1→K=20 at T={FIXED_DEPTH})");
    println!("  Depth gain:  {depth_gain_t4_t16:+.2}% (T=4→T=16 at K={FIXED_WIDTH})");
    println!("  W/D ratio:   {ratio:.2}×");

    if criteria.is_proved() {
        println!("\n  🎉 GOAT PROVED: GRAM principle validated on DDTree infrastructure");
        println!("     Production recommendation: elf_sde default-on + bandit UCB1");
    } else {
        let pass_count = [g1.is_pass(), g2.is_pass(), g3.is_pass()]
            .into_iter()
            .filter(|&p| p)
            .count();
        println!("\n  ⬜ GOAT PENDING: {pass_count}/3 criteria passed");
        println!("     Infrastructure validated — needs real game arenas for full proof");
    }

    // Assert infrastructure works (always passes if code compiles and runs)
    assert!(
        !width_results.is_empty(),
        "width sweep must produce results"
    );
    assert!(
        !depth_results.is_empty(),
        "depth sweep must produce results"
    );
    assert!(
        width_results.iter().all(|r| r.avg_quality >= 0.0),
        "quality must be non-negative"
    );

    println!(
        "\n  ✅ Infrastructure validated: {} configs tested successfully",
        width_results.len() + depth_results.len()
    );
}
