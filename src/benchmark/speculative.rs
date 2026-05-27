use super::{BenchCategory, BenchResult};
use crate::speculative::{
    BinaryScreeningPruner, LeviathanVerifier, NoPruner, NoScreeningPruner, SimulatedVerifier,
    SpeculativeContext, TreeBuilder, dflash_predict_ar_with, dflash_predict_parallel,
    dflash_predict_with, extract_best_path_into, sample_from_distribution,
    speculative_step_conditioned_with, speculative_step_rollback_with, speculative_step_verifier,
};
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use crate::types::{Config, Rng, softmax_scaled};
use rayon::prelude::*;
use std::time::Instant;

pub fn bench_ar(
    weights: &TransformerWeights,
    config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);

    for _ in 0..warmup {
        cache.reset();
        let logits = forward(&mut ctx, weights, &mut cache, 0, 0, config);
        softmax_scaled(logits, 1.0 / config.temperature);
    }

    let start = Instant::now();
    for _ in 0..iters {
        cache.reset();
        let logits = forward(&mut ctx, weights, &mut cache, 0, 0, config);
        softmax_scaled(logits, 1.0 / config.temperature);
    }
    let elapsed = start.elapsed();

    let tps = iters as f64 / elapsed.as_secs_f64();
    BenchResult {
        label: "Transformer AR".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: 1.0,
        color: (70, 130, 180),
        category: BenchCategory::Speculative,
        feature_dim: "SD".into(),
    }
}

pub fn bench_dflash(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut sctx = SpeculativeContext::new(draft_config);

    for _ in 0..warmup {
        sctx.reset();
        dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    }

    let start = Instant::now();
    for _ in 0..iters {
        sctx.reset();
        let _steps = dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    }
    let elapsed = start.elapsed();

    let tps = iters as f64 / elapsed.as_secs_f64();
    BenchResult {
        label: "DFlash".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: draft_config.draft_lookahead as f64,
        color: (255, 99, 71),
        category: BenchCategory::TreeBuild,
        feature_dim: "SD".into(),
    }
}

/// Benchmark DFlash parallel vs sequential.
///
/// Measures `dflash_predict_with` (seq) against `dflash_predict_parallel` (rayon).
/// Parallel uses `map_init` for per-thread ForwardContext + KV cache.
/// For micro/draft models (n_embd ≤ parallel_threshold), parallel falls back to seq.
pub fn bench_dflash_parallel(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    // ── Sequential ──
    let mut sctx = SpeculativeContext::new(draft_config);

    for _ in 0..warmup {
        sctx.reset();
        dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    }

    let start = Instant::now();
    for _ in 0..iters {
        sctx.reset();
        let _steps = dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    }
    let elapsed_seq = start.elapsed();

    let seq_tps = iters as f64 / elapsed_seq.as_secs_f64();
    let seq_br = BenchResult {
        label: "DFlash (seq)".into(),
        throughput: seq_tps,
        time_per_step_us: elapsed_seq.as_micros() as f64 / iters as f64,
        avg_acceptance_len: draft_config.draft_lookahead as f64,
        color: (255, 99, 71),
        category: BenchCategory::TreeBuild,
        feature_dim: "SD".into(),
    };

    // ── Parallel ──
    for _ in 0..warmup {
        std::hint::black_box(dflash_predict_parallel(draft_weights, draft_config, 0, 0));
    }

    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(dflash_predict_parallel(draft_weights, draft_config, 0, 0));
    }
    let elapsed_par = start.elapsed();

    let par_tps = iters as f64 / elapsed_par.as_secs_f64();
    let par_br = BenchResult {
        label: "DFlash (par)".into(),
        throughput: par_tps,
        time_per_step_us: elapsed_par.as_micros() as f64 / iters as f64,
        avg_acceptance_len: draft_config.draft_lookahead as f64,
        color: (255, 140, 71),
        category: BenchCategory::TreeBuild,
        feature_dim: "SD".into(),
    };

    let speedup = par_tps / seq_tps;
    println!(
        "  DFlash par vs seq: {:.2}× ({:.0} vs {:.0} ops/s, threshold={})",
        speedup, par_tps, seq_tps, draft_config.parallel_threshold
    );

    (seq_br, par_br)
}

pub fn bench_ddtree(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut sctx = SpeculativeContext::new(draft_config);
    sctx.reset();
    dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    let marginals_view: Vec<&[f32]> = (0..sctx.steps_populated)
        .map(|step| sctx.marginal_slice(step, draft_config.vocab_size))
        .collect();

    let mut tree_builder = TreeBuilder::new(draft_config);

    for _ in 0..warmup {
        let _ = tree_builder.build(&marginals_view, draft_config, &NoPruner, false);
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = tree_builder.build(&marginals_view, draft_config, &NoPruner, false);
    }
    let elapsed = start.elapsed();

    let ops = iters as f64 / elapsed.as_secs_f64();
    BenchResult {
        label: "DDTree Build".into(),
        throughput: ops,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: 0.0,
        color: (50, 205, 50),
        category: BenchCategory::TreeBuild,
        feature_dim: "SD".into(),
    }
}

pub fn bench_speculative(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut rng = Rng::new(99);
    let mut verifier = SimulatedVerifier::new(0.75, draft_config);

    for _ in 0..warmup {
        let _ =
            speculative_step_verifier(draft_weights, draft_config, 0, 0, &mut rng, &mut verifier);
    }

    let mut total_accepted = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) =
            speculative_step_verifier(draft_weights, draft_config, 0, 0, &mut rng, &mut verifier);
        total_accepted += accepted.len();
    }
    let elapsed = start.elapsed();

    let tps = total_accepted as f64 / elapsed.as_secs_f64();
    let avg_accept = total_accepted as f64 / iters as f64;
    BenchResult {
        label: "Speculative (Simulated)".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: avg_accept,
        color: (255, 165, 0),
        category: BenchCategory::Speculative,
        feature_dim: "SD".into(),
    }
}

/// Speculative decoding with AR drafting + DDTree + simulated acceptance.
/// Measures pure AR drafting benefit without target model verification cost.
pub fn bench_speculative_ar(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut rng = Rng::new(99);
    let mut sctx = SpeculativeContext::new(draft_config);
    let mut tree_builder = TreeBuilder::new(draft_config);

    for _ in 0..warmup {
        let _ = run_speculative_ar_step(
            &mut sctx,
            &mut tree_builder,
            draft_weights,
            draft_config,
            &mut rng,
        );
    }

    let mut total_accepted = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let accepted = run_speculative_ar_step(
            &mut sctx,
            &mut tree_builder,
            draft_weights,
            draft_config,
            &mut rng,
        );
        total_accepted += accepted;
    }
    let elapsed = start.elapsed();

    let tps = total_accepted as f64 / elapsed.as_secs_f64();
    let avg_accept = total_accepted as f64 / iters as f64;
    BenchResult {
        label: "Speculative (AR Draft)".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: avg_accept,
        color: (255, 200, 0),
        category: BenchCategory::Speculative,
        feature_dim: "SD".into(),
    }
}

/// AR draft + DDTree + simulated acceptance + bonus token.
pub fn run_speculative_ar_step(
    sctx: &mut SpeculativeContext,
    tree_builder: &mut TreeBuilder,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    rng: &mut Rng,
) -> usize {
    // 1. Zero-alloc AR draft
    sctx.reset();
    let steps = dflash_predict_ar_with(sctx, draft_weights, draft_config, 0, 0, rng, None);
    let vocab_size = draft_config.vocab_size;

    // 2. Build tree from marginals
    let marginals_view: Vec<&[f32]> = (0..steps)
        .map(|step| sctx.marginal_slice(step, vocab_size))
        .collect();
    let tree = tree_builder.build(&marginals_view, draft_config, &NoPruner, false);

    // 3. Extract best path
    extract_best_path_into(tree, &mut sctx.path_buf);

    if sctx.path_buf.is_empty() {
        let first_marginal = sctx.marginal_slice(0, vocab_size);
        return sample_from_distribution(
            if first_marginal.is_empty() {
                &[1.0]
            } else {
                first_marginal
            },
            rng,
        );
    }

    // 4. Simulated acceptance: 75% cap
    let acceptance_rate = 0.75;
    let max_accept = ((sctx.path_buf.len() as f32) * acceptance_rate).ceil() as usize;
    sctx.accepted_buf.clear();
    sctx.accepted_buf
        .extend(sctx.path_buf.iter().take(max_accept.max(1)).copied());

    // 5. Bonus token
    if sctx.accepted_buf.len() == max_accept && steps > 0 {
        let last_step = steps - 1;
        let last_marginal = sctx.marginal_slice(last_step, vocab_size);
        let bonus = sample_from_distribution(
            if last_marginal.is_empty() {
                &[1.0]
            } else {
                last_marginal
            },
            rng,
        );
        sctx.accepted_buf.push(bonus);
    }

    sctx.accepted_buf.len()
}

/// Benchmark: Chain-seed DDTree vs regular DDTree.
/// Compares chain_seed=true vs chain_seed=false acceptance length.
pub fn bench_ddtree_chain_seed(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    let mut sctx = SpeculativeContext::new(draft_config);
    sctx.reset();
    dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    let mv: Vec<&[f32]> = (0..sctx.steps_populated)
        .map(|step| sctx.marginal_slice(step, draft_config.vocab_size))
        .collect();

    let mut tree_builder = TreeBuilder::new(draft_config);

    // Warmup
    for _ in 0..warmup {
        let _ = tree_builder.build(&mv, draft_config, &NoPruner, false);
        let _ = tree_builder.build(&mv, draft_config, &NoPruner, true);
    }

    // Benchmark no-chain
    let start = Instant::now();
    let mut total_nodes_no_chain = 0usize;
    for _ in 0..iters {
        let tree = tree_builder.build(&mv, draft_config, &NoPruner, false);
        total_nodes_no_chain += tree.len();
    }
    let elapsed_no_chain = start.elapsed();

    // Benchmark chain-seed
    let start = Instant::now();
    let mut total_nodes_chain = 0usize;
    for _ in 0..iters {
        let tree = tree_builder.build(&mv, draft_config, &NoPruner, true);
        total_nodes_chain += tree.len();
    }
    let elapsed_chain = start.elapsed();

    let ops_no_chain = iters as f64 / elapsed_no_chain.as_secs_f64();
    let ops_chain = iters as f64 / elapsed_chain.as_secs_f64();

    (
        BenchResult {
            label: "DDTree (no chain)".into(),
            throughput: ops_no_chain,
            time_per_step_us: elapsed_no_chain.as_micros() as f64 / iters as f64,
            avg_acceptance_len: total_nodes_no_chain as f64 / iters as f64,
            color: (50, 205, 50),
            category: BenchCategory::TreeBuild,
            feature_dim: "SD".into(),
        },
        BenchResult {
            label: "DDTree (chain-seed)".into(),
            throughput: ops_chain,
            time_per_step_us: elapsed_chain.as_micros() as f64 / iters as f64,
            avg_acceptance_len: total_nodes_chain as f64 / iters as f64,
            color: (0, 200, 100),
            category: BenchCategory::TreeBuild,
            feature_dim: "SD".into(),
        },
    )
}

/// Benchmark: DDTree budget sweep across multiple budgets.
/// Returns results for each budget with and without chain-seed.
/// Benchmark: DDTree with ScreeningPruner vs original ConstraintPruner (Plan 021).
///
/// Returns two results:
/// 1. `build_screened()` with `NoScreeningPruner` (R=1.0 everywhere) — should match original DDTree
/// 2. `build_screened()` with `BinaryScreeningPruner(NoPruner)` — adapter path overhead check
///
/// Both should show zero regression vs the baseline `build()` with `NoPruner`.
pub fn bench_ddtree_screened(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    use crate::speculative::BinaryScreeningPruner;

    let mut sctx = SpeculativeContext::new(draft_config);
    sctx.reset();
    dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    let mv: Vec<&[f32]> = (0..sctx.steps_populated)
        .map(|step| sctx.marginal_slice(step, draft_config.vocab_size))
        .collect();

    let mut tree_builder = TreeBuilder::new(draft_config);

    // Warmup both paths
    for _ in 0..warmup {
        let _ = tree_builder.build_screened(&mv, draft_config, &NoScreeningPruner, false);
        let _ =
            tree_builder.build_screened(&mv, draft_config, &BinaryScreeningPruner(NoPruner), false);
    }

    // Benchmark: NoScreeningPruner (R=1.0, pure screening path, no penalty)
    let start = Instant::now();
    let mut total_nodes_noop = 0usize;
    for _ in 0..iters {
        let tree = tree_builder.build_screened(&mv, draft_config, &NoScreeningPruner, false);
        total_nodes_noop += tree.len();
    }
    let elapsed_noop = start.elapsed();

    // Benchmark: BinaryScreeningPruner adapter (ConstraintPruner → ScreeningPruner)
    let start = Instant::now();
    let mut total_nodes_adapter = 0usize;
    for _ in 0..iters {
        let tree =
            tree_builder.build_screened(&mv, draft_config, &BinaryScreeningPruner(NoPruner), false);
        total_nodes_adapter += tree.len();
    }
    let elapsed_adapter = start.elapsed();

    let ops_noop = iters as f64 / elapsed_noop.as_secs_f64();
    let ops_adapter = iters as f64 / elapsed_adapter.as_secs_f64();

    (
        BenchResult {
            label: "DDTree (screened R=1.0)".into(),
            throughput: ops_noop,
            time_per_step_us: elapsed_noop.as_micros() as f64 / iters as f64,
            avg_acceptance_len: total_nodes_noop as f64 / iters as f64,
            color: (0, 191, 255),
            category: BenchCategory::TreeBuild,
            feature_dim: "SD".into(),
        },
        BenchResult {
            label: "DDTree (screened adapter)".into(),
            throughput: ops_adapter,
            time_per_step_us: elapsed_adapter.as_micros() as f64 / iters as f64,
            avg_acceptance_len: total_nodes_adapter as f64 / iters as f64,
            color: (30, 144, 255),
            category: BenchCategory::TreeBuild,
            feature_dim: "SD".into(),
        },
    )
}

pub fn bench_ddtree_budget_sweep(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    budgets: &[usize],
    warmup: usize,
    iters: usize,
) -> Vec<BenchResult> {
    let mut sctx = SpeculativeContext::new(draft_config);
    sctx.reset();
    dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    let mv: Vec<&[f32]> = (0..sctx.steps_populated)
        .map(|step| sctx.marginal_slice(step, draft_config.vocab_size))
        .collect();

    let mut tree_builder = TreeBuilder::new(draft_config);
    let mut results = Vec::with_capacity(budgets.len() * 2);

    for &budget in budgets {
        let mut sweep_config = draft_config.clone();
        sweep_config.tree_budget = budget;

        // Warmup
        for _ in 0..warmup {
            let _ = tree_builder.build(&mv, &sweep_config, &NoPruner, false);
            let _ = tree_builder.build(&mv, &sweep_config, &NoPruner, true);
        }

        // Benchmark no-chain
        let start = Instant::now();
        let mut total_nodes = 0usize;
        for _ in 0..iters {
            let tree = tree_builder.build(&mv, &sweep_config, &NoPruner, false);
            total_nodes += tree.len();
        }
        let elapsed = start.elapsed();

        results.push(BenchResult {
            label: format!("DDTree budget={budget}"),
            throughput: iters as f64 / elapsed.as_secs_f64(),
            time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
            avg_acceptance_len: total_nodes as f64 / iters as f64,
            color: (100, 149, 237),
            category: BenchCategory::TreeBuild,
            feature_dim: "SD".into(),
        });

        // Benchmark chain-seed
        let start = Instant::now();
        let mut total_nodes = 0usize;
        for _ in 0..iters {
            let tree = tree_builder.build(&mv, &sweep_config, &NoPruner, true);
            total_nodes += tree.len();
        }
        let elapsed = start.elapsed();

        results.push(BenchResult {
            label: format!("DDTree chain budget={budget}"),
            throughput: iters as f64 / elapsed.as_secs_f64(),
            time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
            avg_acceptance_len: total_nodes as f64 / iters as f64,
            color: (60, 179, 113),
            category: BenchCategory::TreeBuild,
            feature_dim: "SD".into(),
        });
    }

    results
}

/// Leviathan Algorithm 1: AR draft + real target p/q verification.
pub fn bench_leviathan(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut rng = Rng::new(99);
    let mut verifier = LeviathanVerifier::new(target_weights, target_config, draft_config);

    for _ in 0..warmup {
        let _ =
            speculative_step_verifier(draft_weights, draft_config, 0, 0, &mut rng, &mut verifier);
    }

    let mut total_accepted = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) =
            speculative_step_verifier(draft_weights, draft_config, 0, 0, &mut rng, &mut verifier);
        total_accepted += accepted.len();
    }
    let elapsed = start.elapsed();

    let tps = total_accepted as f64 / elapsed.as_secs_f64();
    let avg_accept = total_accepted as f64 / iters as f64;
    BenchResult {
        label: "Leviathan (Algorithm 1)".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: avg_accept,
        color: (148, 0, 211),
        category: BenchCategory::Speculative,
        feature_dim: "SD".into(),
    }
}

/// Task 3.8: Benchmark snapshot/rollback overhead vs no-rollback speculative step.
/// Measures the cost of KV-Cache snapshot + restore per speculative step.
/// Snapshot cost: O(n_layer × pos × kv_dim) — cheap at our model scale.
pub fn bench_snapshot_rollback(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    // ── No-rollback: standard Leviathan (resets cache each step) ──
    let mut rng_no_rb = Rng::new(99);
    let mut verifier = LeviathanVerifier::new(target_weights, target_config, draft_config);

    for _ in 0..warmup {
        let _ = speculative_step_verifier(
            draft_weights,
            draft_config,
            0,
            0,
            &mut rng_no_rb,
            &mut verifier,
        );
    }

    let mut total_no_rb = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) = speculative_step_verifier(
            draft_weights,
            draft_config,
            0,
            0,
            &mut rng_no_rb,
            &mut verifier,
        );
        total_no_rb += accepted.len();
    }
    let elapsed_no_rb = start.elapsed();

    let no_rollback = BenchResult {
        label: "Leviathan (no rollback)".into(),
        throughput: total_no_rb as f64 / elapsed_no_rb.as_secs_f64(),
        time_per_step_us: elapsed_no_rb.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_no_rb as f64 / iters as f64,
        color: (180, 100, 220),
        category: BenchCategory::Speculative,
        feature_dim: "SD".into(),
    };

    // ── With rollback: speculative_step_rollback_with (zero-alloc, snapshots + restores) ──
    let mut rng_rb = Rng::new(99);
    let mut target_ctx = ForwardContext::new(target_config);
    let mut target_cache = MultiLayerKVCache::new(target_config);
    let mut draft_sctx = SpeculativeContext::new(draft_config);
    let mut tree_builder = TreeBuilder::new(draft_config);
    let mut probs_buf = vec![0.0f32; target_config.vocab_size];
    let mut residual_buf = vec![0.0f32; target_config.vocab_size];

    for _ in 0..warmup {
        target_cache.reset();
        draft_sctx.reset();
        let _ = speculative_step_rollback_with(
            &mut draft_sctx,
            &mut tree_builder,
            draft_weights,
            draft_config,
            target_weights,
            target_config,
            &mut target_ctx,
            &mut target_cache,
            &mut probs_buf,
            &mut residual_buf,
            0,
            0,
            &mut rng_rb,
        );
    }

    let mut total_rb = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        target_cache.reset();
        draft_sctx.reset();
        let (accepted, _) = speculative_step_rollback_with(
            &mut draft_sctx,
            &mut tree_builder,
            draft_weights,
            draft_config,
            target_weights,
            target_config,
            &mut target_ctx,
            &mut target_cache,
            &mut probs_buf,
            &mut residual_buf,
            0,
            0,
            &mut rng_rb,
        );
        total_rb += accepted.len();
    }
    let elapsed_rb = start.elapsed();

    let with_rollback = BenchResult {
        label: "Leviathan (w/ rollback)".into(),
        throughput: total_rb as f64 / elapsed_rb.as_secs_f64(),
        time_per_step_us: elapsed_rb.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_rb as f64 / iters as f64,
        color: (220, 120, 180),
        category: BenchCategory::Speculative,
        feature_dim: "SD".into(),
    };

    (no_rollback, with_rollback)
}

/// Task 5.4: Benchmark conditioned vs unconditioned draft marginals.
/// Measures acceptance length with target-conditioned draft (DFlash-inspired)
/// vs standard unconditioned draft, using simulated acceptance.
/// Also reports approximate KL divergence from the target distribution.
pub fn bench_conditioned_vs_unconditioned(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    // ── Unconditioned: SimulatedVerifier (standard DFlash) ──
    let mut rng_uncond = Rng::new(99);
    let mut verifier = SimulatedVerifier::new(0.75, draft_config);

    for _ in 0..warmup {
        let _ = speculative_step_verifier(
            draft_weights,
            draft_config,
            0,
            0,
            &mut rng_uncond,
            &mut verifier,
        );
    }

    let mut total_uncond = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) = speculative_step_verifier(
            draft_weights,
            draft_config,
            0,
            0,
            &mut rng_uncond,
            &mut verifier,
        );
        total_uncond += accepted.len();
    }
    let elapsed_uncond = start.elapsed();

    let uncond = BenchResult {
        label: "Spec (unconditioned)".into(),
        throughput: total_uncond as f64 / elapsed_uncond.as_secs_f64(),
        time_per_step_us: elapsed_uncond.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_uncond as f64 / iters as f64,
        color: (255, 140, 0),
        category: BenchCategory::Speculative,
        feature_dim: "SD".into(),
    };

    // ── Conditioned: target hidden state seeds draft KV cache (zero-alloc) ──
    let mut rng_cond = Rng::new(99);
    let mut target_ctx = ForwardContext::new(target_config);
    let mut target_cache = MultiLayerKVCache::new(target_config);
    let mut draft_sctx = SpeculativeContext::new(draft_config);
    let mut tree_builder = TreeBuilder::new(draft_config);
    let mut probs_buf = vec![0.0f32; target_config.vocab_size];

    for _ in 0..warmup {
        target_cache.reset();
        draft_sctx.reset();
        let _ = speculative_step_conditioned_with(
            &mut draft_sctx,
            &mut tree_builder,
            draft_weights,
            draft_config,
            target_weights,
            target_config,
            &mut target_ctx,
            &mut target_cache,
            &mut probs_buf,
            0,
            0,
            &mut rng_cond,
        );
    }

    let mut total_cond = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        target_cache.reset();
        draft_sctx.reset();
        let (accepted, _) = speculative_step_conditioned_with(
            &mut draft_sctx,
            &mut tree_builder,
            draft_weights,
            draft_config,
            target_weights,
            target_config,
            &mut target_ctx,
            &mut target_cache,
            &mut probs_buf,
            0,
            0,
            &mut rng_cond,
        );
        total_cond += accepted.len();
    }
    let elapsed_cond = start.elapsed();

    let cond = BenchResult {
        label: "Spec (conditioned)".into(),
        throughput: total_cond as f64 / elapsed_cond.as_secs_f64(),
        time_per_step_us: elapsed_cond.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_cond as f64 / iters as f64,
        color: (0, 180, 180),
        category: BenchCategory::Speculative,
        feature_dim: "SD".into(),
    };

    (uncond, cond)
}

/// Plan 055 T10: Benchmark MTP on vs off at BPE scale.
/// Compares Leviathan acceptance rate with MTP features enabled (truncate/pad)
/// vs disabled (MTP thresholds set to MAX).
/// Note: Uses truncate/pad fallback — no trained projection weights required.
/// When trained weights are available, acceptance rate should improve further.
pub fn bench_mtp_leviathan(warmup: usize, iters: usize) -> (BenchResult, BenchResult) {
    // BPE target + BPE draft (MTP active by default: n_embd=32 >= threshold=32)
    let bpe_target = Config::bpe();
    let bpe_draft = Config::bpe_draft();

    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&bpe_target, &mut rng);
    let draft_weights = TransformerWeights::new(&bpe_draft, &mut Rng::new(99));

    // MTP OFF config — all thresholds disabled
    let bpe_target_off = Config {
        mtp_activation_threshold: usize::MAX,
        mtp_shared_kv_prompt_threshold: usize::MAX,
        mtp_cluster_vocab_threshold: usize::MAX,
        ..Config::bpe()
    };
    let target_weights_off = TransformerWeights::new(&bpe_target_off, &mut Rng::new(42));

    // ── MTP OFF: standard Leviathan, no target conditioning ──
    let mut verifier_off = LeviathanVerifier::new(&target_weights_off, &bpe_target_off, &bpe_draft);
    let mut rng_off = Rng::new(99);

    for _ in 0..warmup {
        let _ = speculative_step_verifier(
            &draft_weights,
            &bpe_draft,
            0,
            0,
            &mut rng_off,
            &mut verifier_off,
        );
    }

    let mut total_off = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) = speculative_step_verifier(
            &draft_weights,
            &bpe_draft,
            0,
            0,
            &mut rng_off,
            &mut verifier_off,
        );
        total_off += accepted.len();
    }
    let elapsed_off = start.elapsed();

    let off = BenchResult {
        label: "MTP OFF (BPE)".into(),
        throughput: total_off as f64 / elapsed_off.as_secs_f64(),
        time_per_step_us: elapsed_off.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_off as f64 / iters as f64,
        color: (200, 100, 100),
        category: BenchCategory::Speculative,
        feature_dim: "SD".into(),
    };

    // ── MTP ON: truncate/pad fallback (no trained projection weights) ──
    let mut verifier_on = LeviathanVerifier::new(&target_weights, &bpe_target, &bpe_draft);
    let mut rng_on = Rng::new(99);

    for _ in 0..warmup {
        let _ = speculative_step_verifier(
            &draft_weights,
            &bpe_draft,
            0,
            0,
            &mut rng_on,
            &mut verifier_on,
        );
    }

    let mut total_on = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) = speculative_step_verifier(
            &draft_weights,
            &bpe_draft,
            0,
            0,
            &mut rng_on,
            &mut verifier_on,
        );
        total_on += accepted.len();
    }
    let elapsed_on = start.elapsed();

    let on = BenchResult {
        label: "MTP ON truncate/pad (BPE)".into(),
        throughput: total_on as f64 / elapsed_on.as_secs_f64(),
        time_per_step_us: elapsed_on.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_on as f64 / iters as f64,
        color: (100, 200, 100),
        category: BenchCategory::Speculative,
        feature_dim: "SD".into(),
    };

    (off, on)
}

/// Plan 055 T24: Multi-scale MTP benchmark — small_target with shared KV.
/// Uses `Config::small_target()` for both target and draft (matching kv_dim=64)
/// so shared KV cache preloading is active. Compares MTP OFF vs ON with shared KV.
pub fn bench_mtp_shared_kv(warmup: usize, iters: usize) -> (BenchResult, BenchResult) {
    // small_target for both — kv_dim matches so shared KV preload is active
    let config = Config {
        mtp_shared_kv_prompt_threshold: 0, // Always share KV
        ..Config::small_target()
    };
    let config_off = Config {
        mtp_activation_threshold: usize::MAX,
        mtp_shared_kv_prompt_threshold: usize::MAX,
        mtp_cluster_vocab_threshold: usize::MAX,
        ..Config::small_target()
    };

    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&config, &mut rng);
    let target_weights_off = TransformerWeights::new(&config_off, &mut Rng::new(42));

    // Use pos=8 so shared KV preload triggers (pos > threshold=0)
    let pos = 8usize;

    // ── MTP OFF: no shared KV, no conditioning ──
    let mut verifier_off = LeviathanVerifier::new(&target_weights_off, &config_off, &config);
    let mut rng_off = Rng::new(99);

    for _ in 0..warmup {
        let _ = speculative_step_verifier(
            &target_weights_off,
            &config_off,
            0,
            pos,
            &mut rng_off,
            &mut verifier_off,
        );
    }

    let mut total_off = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) = speculative_step_verifier(
            &target_weights_off,
            &config_off,
            0,
            pos,
            &mut rng_off,
            &mut verifier_off,
        );
        total_off += accepted.len();
    }
    let elapsed_off = start.elapsed();

    let off = BenchResult {
        label: "MTP OFF shared KV (small)".into(),
        throughput: total_off as f64 / elapsed_off.as_secs_f64(),
        time_per_step_us: elapsed_off.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_off as f64 / iters as f64,
        color: (180, 120, 120),
        category: BenchCategory::Speculative,
        feature_dim: "SD".into(),
    };

    // ── MTP ON: shared KV + truncate/pad conditioning ──
    let mut verifier_on = LeviathanVerifier::new(&target_weights, &config, &config);
    let mut rng_on = Rng::new(99);

    for _ in 0..warmup {
        let _ = speculative_step_verifier(
            &target_weights,
            &config,
            0,
            pos,
            &mut rng_on,
            &mut verifier_on,
        );
    }

    let mut total_on = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) = speculative_step_verifier(
            &target_weights,
            &config,
            0,
            pos,
            &mut rng_on,
            &mut verifier_on,
        );
        total_on += accepted.len();
    }
    let elapsed_on = start.elapsed();

    let on = BenchResult {
        label: "MTP ON shared KV (small)".into(),
        throughput: total_on as f64 / elapsed_on.as_secs_f64(),
        time_per_step_us: elapsed_on.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_on as f64 / iters as f64,
        color: (120, 180, 120),
        category: BenchCategory::Speculative,
        feature_dim: "SD".into(),
    };

    (off, on)
}
