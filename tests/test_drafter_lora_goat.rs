//! GOAT Proof — Drafter LoRA Training (Plan 117, Phase 1, T8–T12)
//!
//! Proofs:
//! T8:  Training converges — loss decreases on 100+ replay pairs
//! T9:  LoRA-trained drafter improves acceptance rate over random baseline (GOAT)
//! T10: Quality guarantee — LoRA + target verification produces valid output
//! T11: Game pipeline integration — game() target + game_draft LoRA
//! T12: BPE pipeline integration — bpe() target + bpe_draft() LoRA
//!
//! Run with:
//!   cargo test --test test_drafter_lora.goat -- --nocapture
//!   cargo test --test test_drafter_lora.goat -- benchmark --nocapture

use katgpt_rs::speculative::{
    DrafterForwardContext, DrafterLoraWeights, LeviathanVerifier, SpeculativeVerifier,
    TrainingPair, generate_synthetic_pairs, train_drafter_lora,
};
use katgpt_rs::transformer::TransformerWeights;
use katgpt_rs::types::{Config, Rng};

// ── Helpers ──────────────────────────────────────────────────

/// Create compatible target/draft configs for micro scale.
/// Both have vocab_size=27 and block_size=16.
fn micro_configs() -> (Config, Config) {
    let mut target = Config::micro();
    target.mtp_min_output_tokens = 1; // Enable speculative decoding in tests
    let mut draft = Config::draft();
    // Ensure same vocab/block for speculative decoding compatibility
    draft.vocab_size = target.vocab_size;
    draft.block_size = target.block_size;
    draft.draft_lookahead = 4;
    (target, draft)
}

/// Create compatible game target/draft configs.
/// Target: game() with vocab=10, n_embd=16.
/// Draft:  same vocab, n_embd=4 (4× smaller).
fn game_configs() -> (Config, Config) {
    let mut target = Config::game();
    target.mtp_min_output_tokens = 1; // Enable speculative decoding in tests
    let mut draft = target.clone();
    draft.n_embd = 4;
    draft.n_head = 2;
    draft.head_dim = 2;
    draft.mlp_hidden = 16;
    draft.n_kv_head = 2;
    draft.draft_lookahead = 4;
    (target, draft)
}

/// Create compatible BPE target/draft configs.
/// Both have vocab_size=4096.
fn bpe_configs() -> (Config, Config) {
    let target = Config::bpe();
    let mut draft = Config::bpe_draft();
    draft.draft_lookahead = 4;
    (target, draft)
}

/// Numerically stable cross-entropy: -log(softmax(logits)[target]).
fn cross_entropy(logits: &[f32], target: usize) -> f32 {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();
    -(logits[target] - max_val) + sum_exp.ln()
}

/// Compute average cross-entropy loss across training pairs using LoRA drafter.
fn compute_avg_loss(
    draft_config: &Config,
    draft_weights: &TransformerWeights,
    lora: &DrafterLoraWeights,
    pairs: &[TrainingPair],
) -> f32 {
    let mut ctx = DrafterForwardContext::new(draft_config, lora.q_lora.rank);
    let mut total = 0.0f32;
    let mut count = 0usize;

    for pair in pairs {
        let n = pair.input_tokens.len();
        if n == 0 {
            continue;
        }

        // Forward through all input tokens
        for (pos, &tok) in pair.input_tokens[..n.saturating_sub(1)].iter().enumerate() {
            ctx.forward_lora(draft_config, draft_weights, lora, tok, pos);
        }

        // Use last token's logits for loss
        let last_tok = pair.input_tokens[n - 1];
        let logits = ctx.forward_lora(draft_config, draft_weights, lora, last_tok, n - 1);
        total += cross_entropy(logits, pair.target_token);
        count += 1;
    }

    if count > 0 {
        total / count as f32
    } else {
        f32::NAN
    }
}

/// Measure acceptance rate over N speculative decoding steps.
///
/// Returns (rate, total_accepted, total_drafted).
/// Each step starts at pos=0 with a random input token.
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
        // accepted.len() = accepted_draft + 1 (bonus if all accepted, replacement otherwise)
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

// ── T8: Training Converges ────────────────────────────────────

#[test]
fn test_drafter_lora_training_converges() {
    let (target_config, draft_config) = micro_configs();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

    // Generate 100+ training pairs from target model
    let pairs =
        generate_synthetic_pairs(&target_config, &target_weights, 25, 6, &mut Rng::new(123));
    assert!(
        pairs.len() >= 100,
        "Should have ≥100 training pairs, got {}",
        pairs.len()
    );

    // Initialize LoRA
    let mut lora = DrafterLoraWeights::new(&draft_config, 4, 8.0, &mut Rng::new(77));

    // Measure initial loss (before training)
    let initial_loss = compute_avg_loss(&draft_config, &draft_weights, &lora, &pairs);
    assert!(
        initial_loss.is_finite(),
        "Initial loss should be finite, got {initial_loss}"
    );

    // Train for 15 epochs
    let best_loss = train_drafter_lora(&draft_config, &draft_weights, &mut lora, &pairs, 15, 0.01);

    // Measure final loss (after training)
    let final_loss = compute_avg_loss(&draft_config, &draft_weights, &lora, &pairs);
    assert!(
        final_loss.is_finite(),
        "Final loss should be finite, got {final_loss}"
    );

    // Loss should decrease (or at least not increase significantly)
    assert!(
        final_loss < initial_loss,
        "Training should reduce loss: initial={initial_loss:.4} final={final_loss:.4}"
    );

    eprintln!(
        "T8 ✓ Loss decreased: {:.4} → {:.4} (best_epoch={:.4})",
        initial_loss, final_loss, best_loss
    );
}

// ── T9: GOAT Proof — Acceptance Rate Improvement ──────────────

#[test]
fn test_drafter_lora_improves_acceptance() {
    let (target_config, draft_config) = micro_configs();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

    // Generate training pairs
    let pairs =
        generate_synthetic_pairs(&target_config, &target_weights, 30, 5, &mut Rng::new(123));

    // 1. Measure baseline acceptance rate (no LoRA)
    let mut baseline_verifier =
        LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
    let (baseline_rate, baseline_accepted, baseline_drafted) = measure_acceptance_rate(
        &mut baseline_verifier,
        &draft_weights,
        &draft_config,
        &target_config,
        100,
        200, // fixed seed for reproducibility
    );

    // 2. Train LoRA on drafter
    let mut lora = DrafterLoraWeights::new(&draft_config, 4, 8.0, &mut Rng::new(77));
    train_drafter_lora(&draft_config, &draft_weights, &mut lora, &pairs, 20, 0.01);

    // 3. Measure trained acceptance rate (with LoRA)
    let mut trained_verifier =
        LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
    trained_verifier.set_drafter_lora(lora, &draft_config);
    let (trained_rate, trained_accepted, trained_drafted) = measure_acceptance_rate(
        &mut trained_verifier,
        &draft_weights,
        &draft_config,
        &target_config,
        100,
        200, // same seed for fair comparison
    );

    eprintln!(
        "T9 GOAT: baseline={:.3} ({}/{}) \
         trained={:.3} ({}/{})",
        baseline_rate,
        baseline_accepted,
        baseline_drafted,
        trained_rate,
        trained_accepted,
        trained_drafted
    );

    // LoRA-trained drafter should improve acceptance rate
    assert!(
        trained_rate > baseline_rate,
        "LoRA-trained acceptance ({:.3}) should exceed baseline ({:.3})",
        trained_rate,
        baseline_rate
    );
}

// ── T10: Quality Guarantee ────────────────────────────────────

#[test]
fn test_drafter_lora_preserves_output() {
    let (target_config, draft_config) = micro_configs();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

    // Train a LoRA drafter
    let pairs =
        generate_synthetic_pairs(&target_config, &target_weights, 10, 4, &mut Rng::new(123));
    let mut lora = DrafterLoraWeights::new(&draft_config, 4, 8.0, &mut Rng::new(77));
    train_drafter_lora(&draft_config, &draft_weights, &mut lora, &pairs, 10, 0.01);

    // Create verifier with LoRA
    let mut verifier = LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
    verifier.set_drafter_lora(lora, &draft_config);

    // Run 50 speculative decoding steps, verify ALL output tokens are valid
    let mut step_rng = Rng::new(500);
    for step in 0..50 {
        let token = (step_rng.next() % target_config.vocab_size as u64) as usize;
        let accepted = verifier.speculate(&draft_weights, &draft_config, token, 0, &mut step_rng);

        // Quality guarantee: every returned token must be a valid vocab index
        for (i, &tok) in accepted.iter().enumerate() {
            assert!(
                tok < target_config.vocab_size,
                "Step {step}, token [{i}]: {tok} >= vocab_size {}",
                target_config.vocab_size
            );
        }

        // Must return at least 1 token
        assert!(
            !accepted.is_empty(),
            "Step {step}: must return at least 1 token"
        );
    }

    eprintln!("T10 ✓ All 50 steps produced valid tokens (quality guaranteed by construction)");
}

// ── T11: Game Pipeline Integration ────────────────────────────

#[test]
fn test_game_pipeline_drafter_lora() {
    let (target_config, draft_config) = game_configs();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

    // Simulate game replay: board cells (0-8) + action (9 = BOS)
    let vocab = target_config.vocab_size;
    let sequences: Vec<Vec<usize>> = (0..10)
        .map(|seed| {
            let mut seq_rng = Rng::new(seed + 1000);
            (0..8)
                .map(|_| (seq_rng.next() % vocab as u64) as usize)
                .collect()
        })
        .collect();

    // Generate training pairs from target
    let pairs = katgpt_rs::speculative::generate_training_pairs_from_replays(
        &target_config,
        &target_weights,
        &sequences,
    );
    assert!(!pairs.is_empty(), "Should generate game training pairs");

    // Train LoRA drafter
    let mut lora = DrafterLoraWeights::new(&draft_config, 4, 8.0, &mut Rng::new(77));
    let best_loss = train_drafter_lora(&draft_config, &draft_weights, &mut lora, &pairs, 10, 0.01);
    assert!(best_loss.is_finite(), "Game training should converge");

    // Run speculative decoding with LoRA drafter
    let mut verifier = LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
    verifier.set_drafter_lora(lora, &draft_config);

    let mut game_rng = Rng::new(999);
    for step in 0..20 {
        let token = (game_rng.next() % vocab as u64) as usize;
        let accepted = verifier.speculate(&draft_weights, &draft_config, token, 0, &mut game_rng);

        for &tok in &accepted {
            assert!(tok < vocab, "Game output token {tok} >= vocab {vocab}");
        }
        assert!(!accepted.is_empty(), "Step {step}: must return ≥1 token");
    }

    eprintln!(
        "T11 ✓ Game pipeline: {} training pairs, loss={:.4}, 20 steps valid",
        pairs.len(),
        best_loss
    );
}

// ── T12: BPE Pipeline Integration ─────────────────────────────

#[test]
fn test_bpe_pipeline_drafter_lora() {
    let (target_config, draft_config) = bpe_configs();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

    // Simulate BPE token sequences (minimal — BPE FD training is O(1152 params) per pair)
    let vocab = target_config.vocab_size;
    let sequences: Vec<Vec<usize>> = (0..1)
        .map(|seed| {
            let mut seq_rng = Rng::new(seed + 2000);
            (0..3)
                .map(|_| (seq_rng.next() % vocab as u64) as usize)
                .collect()
        })
        .collect();

    // Generate training pairs
    let pairs = katgpt_rs::speculative::generate_training_pairs_from_replays(
        &target_config,
        &target_weights,
        &sequences,
    );
    assert!(!pairs.is_empty(), "Should generate BPE training pairs");

    // Train LoRA drafter (BPE is ~4× larger than micro: 1152 LoRA params)
    // Use 1 epoch on minimal pairs — just verifying pipeline wiring, not convergence
    let mut lora = DrafterLoraWeights::new(&draft_config, 4, 8.0, &mut Rng::new(77));
    let best_loss = train_drafter_lora(&draft_config, &draft_weights, &mut lora, &pairs, 1, 0.01);
    assert!(best_loss.is_finite(), "BPE training should converge");

    // Run speculative decoding with LoRA drafter
    let mut verifier = LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
    verifier.set_drafter_lora(lora, &draft_config);

    let mut bpe_rng = Rng::new(888);
    for step in 0..10 {
        let token = (bpe_rng.next() % vocab as u64) as usize;
        let accepted = verifier.speculate(&draft_weights, &draft_config, token, 0, &mut bpe_rng);

        for &tok in &accepted {
            assert!(tok < vocab, "BPE output token {tok} >= vocab {vocab}");
        }
        assert!(!accepted.is_empty(), "Step {step}: must return ≥1 token");
    }

    eprintln!(
        "T12 ✓ BPE pipeline: {} training pairs, loss={:.4}, 10 steps valid (1 epoch, wiring test)",
        pairs.len(),
        best_loss
    );
}

// ── Benchmark (opt-in) ───────────────────────────────────────

#[test]
fn benchmark() {
    let args = std::env::args().collect::<Vec<_>>();
    if !args.iter().any(|a| a == "benchmark") {
        eprintln!("Skipping benchmark. Run with -- benchmark to enable.");
        return;
    }

    let (target_config, draft_config) = micro_configs();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

    // Generate training pairs
    let pairs =
        generate_synthetic_pairs(&target_config, &target_weights, 50, 6, &mut Rng::new(123));
    eprintln!("Training pairs: {}", pairs.len());

    // Train LoRA
    let mut lora = DrafterLoraWeights::new(&draft_config, 4, 8.0, &mut Rng::new(77));
    let train_start = std::time::Instant::now();
    let best_loss = train_drafter_lora(&draft_config, &draft_weights, &mut lora, &pairs, 20, 0.01);
    let train_time = train_start.elapsed();
    eprintln!("Training: {:?}, best_loss={:.4}", train_time, best_loss);

    // Baseline acceptance rate
    let mut baseline_verifier =
        LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
    let baseline_start = std::time::Instant::now();
    let (baseline_rate, ..) = measure_acceptance_rate(
        &mut baseline_verifier,
        &draft_weights,
        &draft_config,
        &target_config,
        200,
        300,
    );
    let baseline_time = baseline_start.elapsed();

    // LoRA acceptance rate
    let mut trained_verifier =
        LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
    trained_verifier.set_drafter_lora(lora, &draft_config);
    let trained_start = std::time::Instant::now();
    let (trained_rate, ..) = measure_acceptance_rate(
        &mut trained_verifier,
        &draft_weights,
        &draft_config,
        &target_config,
        200,
        300,
    );
    let trained_time = trained_start.elapsed();

    eprintln!(
        "Baseline: rate={:.3}, time={:?}",
        baseline_rate, baseline_time
    );
    eprintln!(
        "Trained:  rate={:.3}, time={:?}",
        trained_rate, trained_time
    );
    let improvement = if baseline_rate > 0.0 {
        trained_rate / baseline_rate
    } else {
        f32::INFINITY
    };
    eprintln!("Improvement: +{:.1}× acceptance", improvement);
}
