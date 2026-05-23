//! PTRM Width vs Depth Scaling Benchmark — run with:
//! cargo test --features "elf_sde bandit" --test bench_ptrm_width_scaling --release -- --nocapture
//!
//! Plan 083: Validates PTRM's width >> depth finding (arXiv:2605.19943):
//! - Width scaling (K=1→64 rollouts): +28.6pp on PPBench
//! - Depth scaling (T=1→8 steps): +3.1pp on PPBench
//!
//! Our adaptation:
//! - Measures path quality (cumulative relevance), diversity, latency
//! - Sweeps K × γ for width, T × γ for depth
//! - Compares BestQ vs MostFrequent selection

#![cfg(all(feature = "elf_sde", feature = "bandit"))]

use microgpt_rs::speculative::dd_tree::{
    WidthScaleConfig, WidthSelectionMode, best_of_k_rollouts, build_dd_tree_sde, extract_best_path,
    inject_sde_noise,
};
use microgpt_rs::speculative::types::{EarlyStopGate, NoScreeningPruner, SdeConfig};
use microgpt_rs::transformer::TransformerWeights;
use microgpt_rs::types::{Config, Rng};
use std::collections::HashSet;

/// Generate marginals from a real model for benchmarking.
fn make_marginals() -> (Config, Vec<Vec<f32>>) {
    let config = Config::draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let marginals = microgpt_rs::speculative::dflash::dflash_predict(&weights, &config, 0, 0);
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

// ── Benchmark 1: Width Scaling (K rollouts) vs γ ────────────────

#[test]
fn bench_ptrm_width_scaling_main() {
    println!("\n🧪 PTRM Width Scaling Benchmark (Plan 083)");
    println!("{}", "═".repeat(80));

    let (config, marginals) = make_marginals();
    let marginals_refs: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let greedy = greedy_path(&marginals);
    let screener = NoScreeningPruner;

    let k_values: &[usize] = &[1, 2, 4, 8, 16, 32, 64];
    let gamma_values: &[f32] = &[0.0, 0.2, 0.5, 1.0];

    println!(
        "{:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "K", "γ", "Quality", "Top1 Agr", "Diversity", "Unique", "Latency(µs)"
    );
    println!("{}", "─".repeat(80));

    for &gamma in gamma_values {
        let sde_config = SdeConfig {
            gamma,
            ..Default::default()
        };

        for &k in k_values {
            let width_config = WidthScaleConfig {
                k_rollouts: k,
                selection: WidthSelectionMode::BestQ,
            };

            let start = std::time::Instant::now();
            let n_trials = 10;
            let mut all_paths: Vec<Vec<usize>> = Vec::with_capacity(n_trials);
            let mut qualities: Vec<f32> = Vec::with_capacity(n_trials);
            let mut agreements: Vec<f32> = Vec::with_capacity(n_trials);

            for trial in 0..n_trials {
                let path = best_of_k_rollouts(
                    &marginals_refs,
                    &config,
                    &screener,
                    &sde_config,
                    &width_config,
                    42 + trial as u64,
                );
                qualities.push(path_quality(&marginals, &path));
                agreements.push(top1_agreement(&greedy, &path));
                all_paths.push(path);
            }
            let elapsed = start.elapsed();

            let avg_quality = qualities.iter().sum::<f32>() / qualities.len() as f32;
            let avg_agreement = agreements.iter().sum::<f32>() / agreements.len() as f32;

            // Diversity: fraction of unique paths across trials
            let unique: HashSet<_> = all_paths.iter().collect();
            let diversity = unique.len() as f32 / all_paths.len() as f32;
            let latency_us = elapsed.as_micros() as f64 / n_trials as f64;

            println!(
                "{:>6} {:>6.1} {:>10.6} {:>10.4} {:>10.4} {:>10} {:>12.1}",
                k,
                gamma,
                avg_quality,
                avg_agreement,
                diversity,
                unique.len(),
                latency_us
            );
        }
        println!();
    }

    // ── Summary: K=1 vs K=64 at best γ ─────────────────────────
    println!("\n📊 Width Scaling Summary (K=1→64 at γ=0.5)");
    println!("{}", "─".repeat(50));

    let sde_config = SdeConfig {
        gamma: 0.5,
        ..Default::default()
    };

    let mut k1_quality = 0.0f32;
    let mut k64_quality = 0.0f32;
    let n = 50;

    for trial in 0..n {
        let seed = 100 + trial as u64;

        // K=1
        let path1 = best_of_k_rollouts(
            &marginals_refs,
            &config,
            &screener,
            &sde_config,
            &WidthScaleConfig {
                k_rollouts: 1,
                selection: WidthSelectionMode::BestQ,
            },
            seed,
        );
        k1_quality += path_quality(&marginals, &path1);

        // K=64
        let path64 = best_of_k_rollouts(
            &marginals_refs,
            &config,
            &screener,
            &sde_config,
            &WidthScaleConfig {
                k_rollouts: 64,
                selection: WidthSelectionMode::BestQ,
            },
            seed,
        );
        k64_quality += path_quality(&marginals, &path64);
    }

    k1_quality /= n as f32;
    k64_quality /= n as f32;
    let gain = (k64_quality - k1_quality) / k1_quality * 100.0;

    println!("  K=1  avg quality:  {k1_quality:.6}");
    println!("  K=64 avg quality:  {k64_quality:.6}");
    println!("  Width gain:        {gain:+.2}%");
}

// ── Benchmark 2: Depth Scaling (T steps) vs γ ───────────────────

#[test]
fn bench_ptrm_depth_scaling() {
    println!("\n🧪 PTRM Depth Scaling Benchmark (T=lookahead steps)");
    println!("{}", "═".repeat(80));

    let (base_config, marginals) = make_marginals();
    let greedy = greedy_path(&marginals);
    let screener = NoScreeningPruner;

    let t_values: &[usize] = &[1, 2, 4, 8];
    let gamma_values: &[f32] = &[0.0, 0.2, 0.5, 1.0];

    println!(
        "{:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "T", "γ", "Quality", "Top1 Agr", "TreeSize", "Unique", "Latency(µs)"
    );
    println!("{}", "─".repeat(80));

    for &gamma in gamma_values {
        let sde_config = SdeConfig {
            gamma,
            ..Default::default()
        };

        for &t in t_values {
            // Override draft_lookahead to control depth
            let mut config = base_config.clone();
            config.draft_lookahead = t.min(marginals.len());

            let start = std::time::Instant::now();
            let n_trials = 50;
            let mut qualities: Vec<f32> = Vec::with_capacity(n_trials);
            let mut agreements: Vec<f32> = Vec::with_capacity(n_trials);
            let mut tree_sizes: Vec<usize> = Vec::with_capacity(n_trials);
            let mut unique_paths: HashSet<Vec<usize>> = HashSet::new();

            for trial in 0..n_trials {
                let mut rng = Rng::new(42 + trial as u64);
                let tree = build_dd_tree_sde(
                    &marginals_refs_truncated(&marginals, config.draft_lookahead),
                    &config,
                    &screener,
                    false,
                    &sde_config,
                    &mut rng,
                );
                let path = extract_best_path(&tree);
                qualities.push(path_quality(&marginals, &path));
                agreements.push(top1_agreement(&greedy, &path));
                tree_sizes.push(tree.len());
                unique_paths.insert(path);
            }
            let elapsed = start.elapsed();

            let avg_quality = qualities.iter().sum::<f32>() / qualities.len() as f32;
            let avg_agreement = agreements.iter().sum::<f32>() / agreements.len() as f32;
            let avg_tree = tree_sizes.iter().sum::<usize>() as f32 / tree_sizes.len() as f32;
            let latency_us = elapsed.as_micros() as f64 / n_trials as f64;

            println!(
                "{:>6} {:>6.1} {:>10.6} {:>10.4} {:>10.1} {:>10} {:>12.1}",
                t,
                gamma,
                avg_quality,
                avg_agreement,
                avg_tree,
                unique_paths.len(),
                latency_us
            );
        }
        println!();
    }

    // ── Summary: T=1 vs T=8 at best γ ─────────────────────────
    println!("\n📊 Depth Scaling Summary (T=1→8 at γ=0.5)");
    println!("{}", "─".repeat(50));

    let sde_config = SdeConfig {
        gamma: 0.5,
        ..Default::default()
    };

    let mut t1_quality = 0.0f32;
    let mut t8_quality = 0.0f32;
    let n = 50;

    for trial in 0..n {
        // T=1
        let mut config1 = base_config.clone();
        config1.draft_lookahead = 1;
        let mut rng1 = Rng::new(100 + trial as u64);
        let tree1 = build_dd_tree_sde(
            &marginals_refs_truncated(&marginals, 1),
            &config1,
            &screener,
            false,
            &sde_config,
            &mut rng1,
        );
        let path1 = extract_best_path(&tree1);
        t1_quality += path_quality(&marginals, &path1);

        // T=8
        let mut config8 = base_config.clone();
        config8.draft_lookahead = 8.min(marginals.len());
        let mut rng8 = Rng::new(100 + trial as u64);
        let tree8 = build_dd_tree_sde(
            &marginals_refs_truncated(&marginals, config8.draft_lookahead),
            &config8,
            &screener,
            false,
            &sde_config,
            &mut rng8,
        );
        let path8 = extract_best_path(&tree8);
        t8_quality += path_quality(&marginals, &path8);
    }

    t1_quality /= n as f32;
    t8_quality /= n as f32;
    let gain = (t8_quality - t1_quality) / t1_quality * 100.0;

    println!("  T=1  avg quality:  {t1_quality:.6}");
    println!("  T=8  avg quality:  {t8_quality:.6}");
    println!("  Depth gain:        {gain:+.2}%");
}

// ── Benchmark 3: Selection Modes ────────────────────────────────

#[test]
fn bench_ptrm_selection_modes() {
    println!("\n🧪 PTRM Selection Modes: BestQ vs MostFrequent");
    println!("{}", "═".repeat(80));

    let (config, marginals) = make_marginals();
    let marginals_refs: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let greedy = greedy_path(&marginals);
    let screener = NoScreeningPruner;

    let sde_config = SdeConfig {
        gamma: 0.5,
        ..Default::default()
    };

    let k_values: &[usize] = &[2, 4, 8, 16, 32, 64];

    println!(
        "{:>6} {:>12} {:>10} {:>10} {:>12}",
        "K", "Mode", "Quality", "Top1 Agr", "Latency(µs)"
    );
    println!("{}", "─".repeat(60));

    for &k in k_values {
        for mode in [WidthSelectionMode::BestQ, WidthSelectionMode::MostFrequent] {
            let width_config = WidthScaleConfig {
                k_rollouts: k,
                selection: mode,
            };

            let start = std::time::Instant::now();
            let n_trials = 30;
            let mut qualities: Vec<f32> = Vec::with_capacity(n_trials);
            let mut agreements: Vec<f32> = Vec::with_capacity(n_trials);

            for trial in 0..n_trials {
                let path = best_of_k_rollouts(
                    &marginals_refs,
                    &config,
                    &screener,
                    &sde_config,
                    &width_config,
                    42 + trial as u64,
                );
                qualities.push(path_quality(&marginals, &path));
                agreements.push(top1_agreement(&greedy, &path));
            }
            let elapsed = start.elapsed();

            let avg_quality = qualities.iter().sum::<f32>() / qualities.len() as f32;
            let avg_agreement = agreements.iter().sum::<f32>() / agreements.len() as f32;
            let latency_us = elapsed.as_micros() as f64 / n_trials as f64;
            let mode_str = match mode {
                WidthSelectionMode::BestQ => "BestQ",
                WidthSelectionMode::MostFrequent => "MostFreq",
                #[cfg(feature = "eqr_convergence")]
                WidthSelectionMode::Top1Converged => "Top1Conv",
            };

            println!(
                "{:>6} {:>12} {:>10.6} {:>10.4} {:>12.1}",
                k, mode_str, avg_quality, avg_agreement, latency_us
            );
        }
        println!();
    }
}

// ── Benchmark 4: EarlyStopGate Impact ───────────────────────────

#[test]
fn bench_ptrm_early_stop_gate() {
    println!("\n🧪 PTRM EarlyStopGate: Threshold vs Tree Size");
    println!("{}", "═".repeat(80));

    let (config, marginals) = make_marginals();
    let _screener = NoScreeningPruner;

    let sde_config = SdeConfig {
        gamma: 0.5,
        ..Default::default()
    };

    let thresholds: &[f32] = &[0.0, 0.1, 0.2, 0.3, 0.5, 0.7];
    let k_values: &[usize] = &[1, 8, 16];

    println!(
        "{:>6} {:>12} {:>10} {:>10} {:>10}",
        "K", "Threshold", "Avg TreeSz", "Quality", "Reduction%"
    );
    println!("{}", "─".repeat(60));

    for &k in k_values {
        // Baseline tree size at threshold=0.0
        let baseline_tree = {
            let gate = EarlyStopGate {
                inner: NoScreeningPruner,
                confidence_threshold: 0.0,
                enabled: true,
            };
            let mut rng = Rng::new(42);
            let tree = build_dd_tree_sde(
                &marginals.iter().map(|m| m.as_slice()).collect::<Vec<_>>(),
                &config,
                &gate,
                false,
                &sde_config,
                &mut rng,
            );
            tree.len() as f32
        };

        for &threshold in thresholds {
            let gate = EarlyStopGate {
                inner: NoScreeningPruner,
                confidence_threshold: threshold,
                enabled: threshold > 0.0,
            };

            let n_trials = 30;
            let mut tree_sizes: Vec<usize> = Vec::with_capacity(n_trials);
            let mut qualities: Vec<f32> = Vec::with_capacity(n_trials);

            for trial in 0..n_trials {
                let mut rng = Rng::new(42 + trial as u64);
                let marginals_refs: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

                // Inject noise for each rollout
                let noisy = inject_sde_noise(&marginals_refs, &sde_config, &mut rng);
                let noisy_refs: Vec<&[f32]> = noisy.iter().map(|m| m.as_slice()).collect();

                let tree = microgpt_rs::speculative::dd_tree::build_dd_tree_screened(
                    &noisy_refs,
                    &config,
                    &gate,
                    false,
                );
                let path = extract_best_path(&tree);
                tree_sizes.push(tree.len());
                qualities.push(path_quality(&marginals, &path));
            }

            let avg_tree = tree_sizes.iter().sum::<usize>() as f32 / tree_sizes.len() as f32;
            let avg_quality = qualities.iter().sum::<f32>() / qualities.len() as f32;
            let reduction = (1.0 - avg_tree / baseline_tree) * 100.0;

            println!(
                "{:>6} {:>12.1} {:>10.1} {:>10.6} {:>10.1}",
                k, threshold, avg_tree, avg_quality, reduction
            );
        }
        println!();
    }
}

// ── Benchmark 5: GOAT Proof — Width vs Depth Head-to-Head ───────

#[test]
fn bench_ptrm_goat_proof_width_vs_depth() {
    println!("\n🐐 PTRM GOAT Proof: Width (K) vs Depth (T) Scaling");
    println!("{}", "═".repeat(80));

    let (base_config, marginals) = make_marginals();
    let screener = NoScreeningPruner;
    let sde_config = SdeConfig {
        gamma: 0.5,
        ..Default::default()
    };

    let n_trials = 100;

    // ── Width scaling: K=1 to K=64, T fixed at 8 ───────────────
    println!("\n📊 Width Scaling (T=8 fixed, K varies)");
    println!("{:>6} {:>10} {:>12}", "K", "Quality", "Latency(µs)");
    println!("{}", "─".repeat(40));

    let mut width_gains: Vec<(usize, f32)> = Vec::new();
    let base_quality;

    {
        // K=1 baseline
        let mut total_q = 0.0f32;
        let start = std::time::Instant::now();
        for trial in 0..n_trials {
            let path = best_of_k_rollouts(
                &marginals.iter().map(|m| m.as_slice()).collect::<Vec<_>>(),
                &base_config,
                &screener,
                &sde_config,
                &WidthScaleConfig {
                    k_rollouts: 1,
                    selection: WidthSelectionMode::BestQ,
                },
                200 + trial as u64,
            );
            total_q += path_quality(&marginals, &path);
        }
        let elapsed = start.elapsed();
        base_quality = total_q / n_trials as f32;
        println!(
            "{:>6} {:>10.6} {:>12.1}",
            1,
            base_quality,
            elapsed.as_micros() as f64 / n_trials as f64
        );
        width_gains.push((1, 0.0));
    }

    for &k in &[2, 4, 8, 16, 32, 64] {
        let mut total_q = 0.0f32;
        let start = std::time::Instant::now();
        for trial in 0..n_trials {
            let path = best_of_k_rollouts(
                &marginals.iter().map(|m| m.as_slice()).collect::<Vec<_>>(),
                &base_config,
                &screener,
                &sde_config,
                &WidthScaleConfig {
                    k_rollouts: k,
                    selection: WidthSelectionMode::BestQ,
                },
                200 + trial as u64,
            );
            total_q += path_quality(&marginals, &path);
        }
        let elapsed = start.elapsed();
        let avg_q = total_q / n_trials as f32;
        let gain = (avg_q - base_quality) / base_quality * 100.0;
        width_gains.push((k, gain));
        println!(
            "{:>6} {:>10.6} {:>12.1}",
            k,
            avg_q,
            elapsed.as_micros() as f64 / n_trials as f64
        );
    }

    // ── Depth scaling: T=1 to T=8, K=1 fixed ───────────────────
    println!("\n📊 Depth Scaling (K=1 fixed, T varies)");
    println!("{:>6} {:>10} {:>12}", "T", "Quality", "Latency(µs)");
    println!("{}", "─".repeat(40));

    let mut depth_gains: Vec<(usize, f32)> = Vec::new();
    let base_quality_t;

    {
        // T=1 baseline
        let mut config1 = base_config.clone();
        config1.draft_lookahead = 1;
        let mut total_q = 0.0f32;
        let start = std::time::Instant::now();
        for trial in 0..n_trials {
            let mut rng = Rng::new(200 + trial as u64);
            let tree = build_dd_tree_sde(
                &marginals_refs_truncated(&marginals, 1),
                &config1,
                &screener,
                false,
                &sde_config,
                &mut rng,
            );
            let path = extract_best_path(&tree);
            total_q += path_quality(&marginals, &path);
        }
        let elapsed = start.elapsed();
        base_quality_t = total_q / n_trials as f32;
        println!(
            "{:>6} {:>10.6} {:>12.1}",
            1,
            base_quality_t,
            elapsed.as_micros() as f64 / n_trials as f64
        );
        depth_gains.push((1, 0.0));
    }

    for &t in &[2, 4, 8] {
        let mut config_t = base_config.clone();
        config_t.draft_lookahead = t.min(marginals.len());
        let mut total_q = 0.0f32;
        let start = std::time::Instant::now();
        for trial in 0..n_trials {
            let mut rng = Rng::new(200 + trial as u64);
            let tree = build_dd_tree_sde(
                &marginals_refs_truncated(&marginals, config_t.draft_lookahead),
                &config_t,
                &screener,
                false,
                &sde_config,
                &mut rng,
            );
            let path = extract_best_path(&tree);
            total_q += path_quality(&marginals, &path);
        }
        let elapsed = start.elapsed();
        let avg_q = total_q / n_trials as f32;
        let gain = (avg_q - base_quality_t) / base_quality_t * 100.0;
        depth_gains.push((t, gain));
        println!(
            "{:>6} {:>10.6} {:>12.1}",
            t,
            avg_q,
            elapsed.as_micros() as f64 / n_trials as f64
        );
    }

    // ── GOAT Verdict ─────────────────────────────────────────────
    let max_width_gain = width_gains.last().map(|(_, g)| *g).unwrap_or(0.0);
    let max_depth_gain = depth_gains.last().map(|(_, g)| *g).unwrap_or(0.0);
    let ratio = if max_depth_gain.abs() > 0.01 {
        max_width_gain / max_depth_gain
    } else {
        f32::INFINITY
    };

    println!("\n🏆 GOAT Verdict");
    println!("{}", "═".repeat(50));
    println!("  Width  K=1→64: {max_width_gain:+.2}% gain");
    println!("  Depth  T=1→8:  {max_depth_gain:+.2}% gain");
    println!("  Ratio (W/D):   {ratio:.2}×");
    if ratio >= 3.0 {
        println!("  ✅ WIDTH >> DEPTH confirmed (PTRM: 9.2× ratio)");
    } else if ratio > 1.0 {
        println!("  ⚠️  Width > Depth, but ratio < 3× (task-dependent)");
    } else {
        println!("  ❌ Width scaling did not dominate on this task");
    }
}

/// Helper: truncate marginals to `n` depths for depth-controlled experiments.
fn marginals_refs_truncated(marginals: &[Vec<f32>], n: usize) -> Vec<&[f32]> {
    marginals.iter().take(n).map(|m| m.as_slice()).collect()
}
