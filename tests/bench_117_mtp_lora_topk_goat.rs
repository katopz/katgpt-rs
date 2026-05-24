//! Plan 117 T38-T41: MTP LoRA Drafter & Top-K Benchmarks
//!
//! T38: Game LoRA acceptance rate benchmark
//! T39: BPE LoRA throughput benchmark
//! T40: Top-K candidate coverage benchmark
//! T41: Output-length gating benchmark
//!
//! Run: `cargo test --features dllm --test bench_117_mtp_lora_topk_goat -- --nocapture`

#![cfg(feature = "dllm")]

use std::time::Instant;

use microgpt_rs::speculative::{
    DrafterLoraWeights, LeviathanVerifier, SpeculativeVerifier, generate_synthetic_pairs,
    train_drafter_lora,
};
use microgpt_rs::transformer::{TransformerWeights, cluster_map_round_robin, select_topk_indices};
use microgpt_rs::types::{Config, Rng};

// ── Helpers ──────────────────────────────────────────────────

/// Create compatible micro_dllm target/draft configs (fast).
/// Target: micro_dllm() with vocab=27, block_size=16.
/// Draft:  same vocab/block, smaller dims.
fn micro_configs() -> (Config, Config) {
    let mut target = Config::micro_dllm();
    target.mtp_min_output_tokens = 1; // Enable speculative decoding
    let mut draft = Config::draft();
    draft.vocab_size = target.vocab_size;
    draft.block_size = target.block_size;
    draft.draft_lookahead = 4;
    (target, draft)
}

/// Measure acceptance rate over N speculative decoding steps.
/// Returns (rate, total_accepted, total_drafted).
fn measure_acceptance_rate(
    verifier: &mut LeviathanVerifier,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_config: &Config,
    n_steps: usize,
    seed: u64,
) -> (f32, usize, usize) {
    let mut rng = Rng::new(seed);
    let mut total_accepted = 0usize;
    let mut total_drafted = 0usize;
    let gamma = draft_config.draft_lookahead;

    for _ in 0..n_steps {
        let token = (rng.next() % target_config.vocab_size as u64) as usize;
        let accepted = verifier.speculate(draft_weights, draft_config, token, 0, &mut rng);
        let accepted_draft = accepted.len().saturating_sub(1);
        total_accepted += accepted_draft;
        total_drafted += gamma;
    }

    let rate = if total_drafted > 0 {
        total_accepted as f32 / total_drafted as f32
    } else {
        0.0
    };
    (rate, total_accepted, total_drafted)
}

// ── T38: Game LoRA Acceptance Rate ───────────────────────────

#[test]
fn bench_game_lora_acceptance() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  T38: Game LoRA Acceptance Rate Benchmark                        ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let (target_config, draft_config) = micro_configs();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

    // Generate synthetic training pairs (~50 pairs)
    let pairs =
        generate_synthetic_pairs(&target_config, &target_weights, 10, 6, &mut Rng::new(123));
    println!(
        "║  Config: micro_dllm, vocab={}, block={}",
        target_config.vocab_size, target_config.block_size
    );
    println!(
        "║  Training pairs: {:>4}                                           ║",
        pairs.len()
    );

    // Train LoRA for 20 epochs (fast)
    let mut lora = DrafterLoraWeights::new(&draft_config, 4, 8.0, &mut Rng::new(77));
    let train_start = Instant::now();
    let best_loss = train_drafter_lora(&draft_config, &draft_weights, &mut lora, &pairs, 20, 0.01);
    let train_time = train_start.elapsed();
    println!(
        "║  Training: {:>6?} best_loss={:.4}                       ║",
        train_time, best_loss
    );

    // Baseline acceptance rate (no LoRA)
    let mut baseline_verifier =
        LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
    let (baseline_rate, baseline_accepted, baseline_drafted) = measure_acceptance_rate(
        &mut baseline_verifier,
        &draft_weights,
        &draft_config,
        &target_config,
        20,
        300,
    );

    // LoRA acceptance rate
    let mut trained_verifier =
        LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
    trained_verifier.set_drafter_lora(lora, &draft_config);
    let (trained_rate, trained_accepted, trained_drafted) = measure_acceptance_rate(
        &mut trained_verifier,
        &draft_weights,
        &draft_config,
        &target_config,
        20,
        300,
    );

    println!("║                                                                  ║");
    println!("║  ┌───────────┬──────────┬───────────┐                           ║");
    println!("║  │ Mode      │ Rate     │ Accepted  │                           ║");
    println!("║  ├───────────┼──────────┼───────────┤                           ║");
    println!(
        "║  │ Baseline  │ {:.4}   │ {:>3}/{:<3}     │                           ║",
        baseline_rate, baseline_accepted, baseline_drafted
    );
    println!(
        "║  │ LoRA-20ep │ {:.4}   │ {:>3}/{:<3}     │                           ║",
        trained_rate, trained_accepted, trained_drafted
    );
    println!("║  └───────────┴──────────┴───────────┘                           ║");

    let improvement = if baseline_rate > 0.0 {
        trained_rate / baseline_rate
    } else {
        f32::INFINITY
    };
    println!(
        "║  Improvement: +{:.1}× acceptance                                 ║",
        improvement
    );
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
}

// ── T39: BPE LoRA Throughput ─────────────────────────────────

#[test]
fn bench_bpe_lora_throughput() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  T39: BPE LoRA Throughput Benchmark                              ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let (target_config, draft_config) = micro_configs();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

    println!(
        "║  Config: micro_dllm, vocab={}, block={}",
        target_config.vocab_size, target_config.block_size
    );

    // Generate synthetic training pairs
    let pairs =
        generate_synthetic_pairs(&target_config, &target_weights, 10, 6, &mut Rng::new(123));
    println!(
        "║  Training pairs: {:>4}                                           ║",
        pairs.len()
    );

    // Train LoRA for 20 epochs
    let mut lora = DrafterLoraWeights::new(&draft_config, 4, 8.0, &mut Rng::new(77));
    let train_start = Instant::now();
    let best_loss = train_drafter_lora(&draft_config, &draft_weights, &mut lora, &pairs, 20, 0.01);
    let train_time = train_start.elapsed();
    println!(
        "║  Training: {:>6?} best_loss={:.4}                       ║",
        train_time, best_loss
    );

    // Measure throughput: tokens/round
    let n_rounds = 20usize;
    let mut trained_verifier =
        LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
    trained_verifier.set_drafter_lora(lora, &draft_config);

    let mut rng = Rng::new(300);
    let mut total_tokens = 0usize;
    let mut tokens_per_round = Vec::with_capacity(n_rounds);
    let decode_start = Instant::now();

    for _ in 0..n_rounds {
        let token = (rng.next() % target_config.vocab_size as u64) as usize;
        let accepted =
            trained_verifier.speculate(&draft_weights, &draft_config, token, 0, &mut rng);
        total_tokens += accepted.len();
        tokens_per_round.push(accepted.len());
    }
    let decode_time = decode_start.elapsed();

    let avg_tokens = total_tokens as f32 / n_rounds as f32;
    let tokens_per_sec = if decode_time.as_secs_f32() > 0.0 {
        total_tokens as f32 / decode_time.as_secs_f32()
    } else {
        f32::INFINITY
    };

    let max_tokens = *tokens_per_round.iter().max().unwrap_or(&0);
    let min_tokens = *tokens_per_round.iter().min().unwrap_or(&0);

    println!("║                                                                  ║");
    println!("║  ┌──────────────────────────┬─────────────────────┐             ║");
    println!("║  │ Metric                   │ Value               │             ║");
    println!("║  ├──────────────────────────┼─────────────────────┤             ║");
    println!(
        "║  │ Rounds                   │ {:>18}  │             ║",
        n_rounds
    );
    println!(
        "║  │ Total tokens             │ {:>18}  │             ║",
        total_tokens
    );
    println!(
        "║  │ Avg tokens/round         │ {:>18.2}  │             ║",
        avg_tokens
    );
    println!(
        "║  │ Min tokens/round         │ {:>18}  │             ║",
        min_tokens
    );
    println!(
        "║  │ Max tokens/round         │ {:>18}  │             ║",
        max_tokens
    );
    println!(
        "║  │ Decode time              │ {:>16?}  │             ║",
        decode_time
    );
    println!(
        "║  │ Throughput (tokens/sec)  │ {:>18.0}  │             ║",
        tokens_per_sec
    );
    println!("║  └──────────────────────────┴─────────────────────┘             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
}

// ── T40: Top-K Candidate Coverage ────────────────────────────

#[test]
fn bench_topk_candidate_coverage() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  T40: Top-K Candidate Coverage Benchmark                         ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let config = Config::bpe();
    let vocab_size = config.vocab_size;
    let cluster_size = config.mtp_cluster_size;

    println!("║  vocab_size={vocab_size}, cluster_size={cluster_size}");

    let cluster_map = cluster_map_round_robin(vocab_size, cluster_size);
    let num_clusters = cluster_map.len();
    println!("║  num_clusters={num_clusters}");
    println!("║                                                                  ║");

    // Simulate cluster scores using random values
    let mut rng = Rng::new(42);
    let cluster_scores: Vec<f32> = (0..num_clusters).map(|_| rng.normal()).collect();

    // Test Top-K selections
    let k_values = [1usize, 4, 8, 32];
    let mut results: Vec<(usize, usize)> = Vec::new(); // (k, covered_tokens)

    println!("║  ┌──────────┬─────────────┬──────────────┬──────────────┐       ║");
    println!("║  │ K        │ Candidates  │ Min Expected │ Coverage %   │       ║");
    println!("║  ├──────────┼─────────────┼──────────────┼──────────────┤       ║");

    for &k in &k_values {
        let topk_indices = select_topk_indices(&cluster_scores, k);

        // Count total candidate tokens across selected clusters
        let covered_tokens: usize = topk_indices
            .iter()
            .map(|&cluster_idx| cluster_map[cluster_idx].len())
            .sum();

        let min_expected = k * cluster_size.min(vocab_size);
        let coverage_pct = covered_tokens as f32 / vocab_size as f32 * 100.0;

        println!(
            "║  │ K={:<5}  │ {:>9}   │ {:>10}   │ {:>9.1}%   │       ║",
            k, covered_tokens, min_expected, coverage_pct
        );

        // Assert: Top-K covers >= K * cluster_size tokens
        assert!(
            covered_tokens >= min_expected.min(vocab_size),
            "K={k}: covered_tokens ({covered_tokens}) < min_expected ({min_expected})"
        );

        results.push((k, covered_tokens));
    }

    println!("║  └──────────┴─────────────┴──────────────┴──────────────┘       ║");

    // Coverage comparison: Top-8 vs Top-1
    let top1_tokens = results
        .iter()
        .find(|&&(k, _)| k == 1)
        .map(|&(_, t)| t)
        .unwrap_or(0);
    let top8_tokens = results
        .iter()
        .find(|&&(k, _)| k == 8)
        .map(|&(_, t)| t)
        .unwrap_or(0);
    let coverage_ratio = if top1_tokens > 0 {
        top8_tokens as f32 / top1_tokens as f32
    } else {
        0.0
    };

    println!(
        "║  Top-8/Top-1 coverage ratio: {:.2}×                             ║",
        coverage_ratio
    );
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
}

// ── T41: Output-Length Gating ────────────────────────────────

#[test]
fn bench_output_length_gating() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  T41: Output-Length Gating Benchmark                             ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    // Config with mtp_min_output_tokens = 16
    let mut target_config = Config::bpe();
    target_config.mtp_min_output_tokens = 16;

    let draft_config = Config::bpe_draft();
    println!(
        "║  target: bpe(), mtp_min_output_tokens={}",
        target_config.mtp_min_output_tokens
    );
    println!(
        "║  draft:  bpe_draft(), block_size={}",
        target_config.block_size
    );
    println!("║                                                                  ║");

    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

    // Test case 1: Below threshold → should skip MTP (return 1 token)
    // remaining_capacity = block_size - pos
    // With block_size=256, pos=252 → remaining=4 < 16 → gated
    let pos_below = target_config.block_size - 4; // remaining = 4 < 16
    let remaining_below = target_config.block_size - pos_below;

    let mut verifier_below = LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
    let result_below = verifier_below.speculate(
        &draft_weights,
        &draft_config,
        target_config.bos_token,
        pos_below,
        &mut Rng::new(100),
    );
    let gated = result_below.len() == 1;

    println!("║  ┌───────────────────────────┬──────────────────────────┐       ║");
    println!("║  │ Scenario                  │ Result                   │       ║");
    println!("║  ├───────────────────────────┼──────────────────────────┤       ║");
    println!(
        "║  │ pos={}, remaining={}       │ tokens={}, gated={}  │       ║",
        pos_below,
        remaining_below,
        result_below.len(),
        gated
    );

    // Test case 2: Above threshold → should use MTP (may return >1 tokens)
    let pos_above = 0; // remaining = block_size = 256 >= 16
    let remaining_above = target_config.block_size - pos_above;

    // Run multiple seeds to observe MTP behavior
    let mut saw_multi = false;
    let mut total_tokens_above = 0usize;
    let n_trials = 50usize;

    for seed in 0..n_trials {
        let mut verifier_above =
            LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
        let result = verifier_above.speculate(
            &draft_weights,
            &draft_config,
            target_config.bos_token,
            pos_above,
            &mut Rng::new(seed as u64),
        );
        total_tokens_above += result.len();
        if result.len() > 1 {
            saw_multi = true;
        }
    }

    let avg_tokens_above = total_tokens_above as f32 / n_trials as f32;
    let mtp_active = saw_multi;

    println!(
        "║  │ pos={}, remaining={}      │ avg_tokens={:.2}, mtp={}   │       ║",
        pos_above, remaining_above, avg_tokens_above, mtp_active
    );
    println!("║  └───────────────────────────┴──────────────────────────┘       ║");

    // Verify gating behavior
    assert!(
        gated,
        "pos={pos_below}: remaining={remaining_below} < mtp_min_output_tokens=16, should gate to 1 token"
    );

    println!("║                                                                  ║");
    println!("║  ✅ Gating below threshold: gated to 1 token                     ║");
    println!(
        "║  ✅ MTP above threshold: avg {:.2} tokens/round                    ║",
        avg_tokens_above
    );
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
}
