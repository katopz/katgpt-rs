//! GOAT Proof 109 T7: DMax SPD — Quality Comparison vs Binary D2F
//!
//! Proofs:
//! 1. SPD maintains quality under aggressive parallelism (within ±20pp of binary)
//! 2. Hybrid embeddings carry meaningful uncertainty (non-zero confidence)
//! 3. Convergence check maintains quality (with vs without consistency check)
//! 4. Contiguous prefix preserves quality at τ=0
//!
//! Summary test prints combined results table.
//!
//! Run with:
//!   cargo test --features dmax_spd --test test_dmax_spd -- --nocapture

#![cfg(feature = "dmax_spd")]

use katgpt_rs::dllm::{
    D2fContext, denoising_accuracy, generate_pattern_dataset, train_mini_dllm,
};
use katgpt_rs::speculative::d2f::{
    D2fDecodeConfig, SoftDecodeConfig, d2f_decode_block, d2f_decode_block_soft,
};
use katgpt_rs::speculative::types::NoPruner;
use katgpt_rs::transformer::TransformerWeights;
use katgpt_rs::types::{Config, Rng};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Train a mini dLLM and return (config, weights, test_data).
fn setup_trained_model() -> (Config, TransformerWeights, Vec<Vec<usize>>) {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let effective_vocab = config.vocab_size.saturating_sub(1);

    let train_data = generate_pattern_dataset(&mut rng, 50, config.block_size, effective_vocab);
    let test_data = generate_pattern_dataset(&mut rng, 20, config.block_size, effective_vocab);
    let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);

    (config, weights, test_data)
}

/// Create decode config with block_size=4.
fn make_decode_config() -> D2fDecodeConfig {
    D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 0.3,
        block_size: 4,
        temperature: 0.8,
        ..D2fDecodeConfig::default()
    }
}

/// Measure accuracy of decoded block vs target prefix.
fn block_accuracy(tokens: &[usize], target: &[usize], block_size: usize) -> f32 {
    let predicted: Vec<usize> = tokens.iter().take(block_size).copied().collect();
    let target_block: Vec<usize> = target.iter().take(block_size).copied().collect();
    denoising_accuracy(&predicted, &target_block)
}

/// Validate all tokens are within vocab bounds.
fn assert_valid_tokens(tokens: &[usize], vocab_size: usize) {
    for &tok in tokens {
        assert!(tok < vocab_size, "token {tok} >= vocab_size {vocab_size}");
    }
}

// ---------------------------------------------------------------------------
// Proof 1: SPD maintains quality under aggressive parallelism
// ---------------------------------------------------------------------------

#[test]
fn proof_1_spd_quality_vs_binary() {
    let (config, weights, test_data) = setup_trained_model();
    let decode_config = make_decode_config();
    let block_size = decode_config.block_size;
    let soft_config = SoftDecodeConfig::aggressive();

    let n_blocks = 20usize;
    let mut binary_correct = 0usize;
    let mut binary_total = 0usize;
    let mut spd_correct = 0usize;
    let mut spd_total = 0usize;

    for (i, target) in test_data.iter().take(n_blocks).enumerate() {
        // Binary D2F decode
        {
            let mut rng = Rng::new(i as u64 * 100 + 1);
            let result = d2f_decode_block(&weights, &config, &decode_config, &NoPruner, &mut rng);
            assert_valid_tokens(&result.tokens, config.vocab_size);
            let acc = block_accuracy(&result.tokens, target, block_size);
            binary_correct += (acc * block_size as f32) as usize;
            binary_total += block_size;
        }

        // SPD soft decode
        {
            let mut ctx = D2fContext::new(&config);
            let mut rng = Rng::new(i as u64 * 100 + 1);
            let result = d2f_decode_block_soft(
                &mut ctx,
                &weights,
                &config,
                &decode_config,
                &NoPruner,
                &soft_config,
                &mut rng,
            );
            assert_valid_tokens(&result.tokens, config.vocab_size);
            let acc = block_accuracy(&result.tokens, target, block_size);
            spd_correct += (acc * block_size as f32) as usize;
            spd_total += block_size;
        }
    }

    let binary_acc = if binary_total > 0 {
        binary_correct as f32 / binary_total as f32
    } else {
        0.0
    };
    let spd_acc = if spd_total > 0 {
        spd_correct as f32 / spd_total as f32
    } else {
        0.0
    };
    let delta_pp = (spd_acc - binary_acc) * 100.0;

    println!("\n  Proof 1: SPD Quality vs Binary D2F (aggressive preset)");
    println!(
        "    Binary accuracy:  {:.1}% ({}/{})",
        binary_acc * 100.0,
        binary_correct,
        binary_total
    );
    println!(
        "    SPD accuracy:     {:.1}% ({}/{})",
        spd_acc * 100.0,
        spd_correct,
        spd_total
    );
    println!("    Δ (SPD - Binary): {delta_pp:+.1}pp");
    println!("    Blocks tested:    {n_blocks}");

    // GOAT gate: SPD within ±20pp of binary
    assert!(
        spd_acc >= binary_acc - 0.20,
        "SPD accuracy ({:.1}%) more than 20pp below binary ({:.1}%)",
        spd_acc * 100.0,
        binary_acc * 100.0,
    );

    println!("    GOAT gate: ✅ PASS — SPD within ±20pp of binary");
}

// ---------------------------------------------------------------------------
// Proof 2: Hybrid embeddings carry meaningful uncertainty
// ---------------------------------------------------------------------------

#[test]
fn proof_2_hybrid_embedding_confidence() {
    let (config, weights, test_data) = setup_trained_model();
    let decode_config = make_decode_config();
    let soft_config = SoftDecodeConfig::default();

    let n_blocks = 10usize;
    let mut total_confidence = 0.0f32;
    let mut n_valid_blocks = 0usize;

    println!("\n  Proof 2: Hybrid Embedding Confidence (default preset)");

    for (i, _target) in test_data.iter().take(n_blocks).enumerate() {
        let mut ctx = D2fContext::new(&config);
        let mut rng = Rng::new(i as u64 * 200 + 7);
        let result = d2f_decode_block_soft(
            &mut ctx,
            &weights,
            &config,
            &decode_config,
            &NoPruner,
            &soft_config,
            &mut rng,
        );

        assert_valid_tokens(&result.tokens, config.vocab_size);

        let has_history = !result.confidence_history.is_empty();
        let conf = result.confidence_history.first().copied().unwrap_or(0.0);
        let valid_conf = (0.0..=1.0).contains(&conf);

        println!(
            "    Block {i}: confidence_history len={}, first={:.4}, valid={valid_conf}",
            result.confidence_history.len(),
            conf,
        );

        if has_history {
            total_confidence += conf;
            n_valid_blocks += 1;
        }
    }

    let avg_confidence = if n_valid_blocks > 0 {
        total_confidence / n_valid_blocks as f32
    } else {
        0.0
    };

    println!("    Valid blocks:       {n_valid_blocks}/{n_blocks}");
    println!("    Average confidence: {avg_confidence:.4}");

    // At micro scale, confidence may be low but should be > 0 (at least some signal)
    assert!(
        avg_confidence > 0.0,
        "average confidence ({avg_confidence:.4}) should be > 0.0 — at least some signal present",
    );

    // Note: at micro scale, confidence may not monotonically increase — recorded honestly
    if avg_confidence > 0.5 {
        println!("    Signal strength: STRONG (avg > 0.5)");
    } else if avg_confidence > 0.1 {
        println!("    Signal strength: MODERATE (0.1 < avg ≤ 0.5)");
    } else {
        println!("    Signal strength: WEAK (avg ≤ 0.1) — expected at micro_dllm scale");
    }

    println!("    GOAT gate: ✅ PASS — confidence > 0.0, signal detected");
}

// ---------------------------------------------------------------------------
// Proof 3: Convergence check maintains quality
// ---------------------------------------------------------------------------

#[test]
fn proof_3_convergence_check_quality() {
    let (config, weights, test_data) = setup_trained_model();
    let decode_config = make_decode_config();
    let block_size = decode_config.block_size;

    let n_blocks = 10usize;

    let soft_with_check = SoftDecodeConfig {
        consistency_check: true,
        ..SoftDecodeConfig::default()
    };

    let soft_no_check = SoftDecodeConfig {
        consistency_check: false,
        ..SoftDecodeConfig::default()
    };

    let mut check_correct = 0usize;
    let mut check_total = 0usize;
    let mut no_check_correct = 0usize;
    let mut no_check_total = 0usize;

    for (i, target) in test_data.iter().take(n_blocks).enumerate() {
        // With convergence check
        {
            let mut ctx = D2fContext::new(&config);
            let mut rng = Rng::new(i as u64 * 300 + 3);
            let result = d2f_decode_block_soft(
                &mut ctx,
                &weights,
                &config,
                &decode_config,
                &NoPruner,
                &soft_with_check,
                &mut rng,
            );
            assert_valid_tokens(&result.tokens, config.vocab_size);
            let acc = block_accuracy(&result.tokens, target, block_size);
            check_correct += (acc * block_size as f32) as usize;
            check_total += block_size;
        }

        // Without convergence check
        {
            let mut ctx = D2fContext::new(&config);
            let mut rng = Rng::new(i as u64 * 300 + 3);
            let result = d2f_decode_block_soft(
                &mut ctx,
                &weights,
                &config,
                &decode_config,
                &NoPruner,
                &soft_no_check,
                &mut rng,
            );
            assert_valid_tokens(&result.tokens, config.vocab_size);
            let acc = block_accuracy(&result.tokens, target, block_size);
            no_check_correct += (acc * block_size as f32) as usize;
            no_check_total += block_size;
        }
    }

    let check_acc = if check_total > 0 {
        check_correct as f32 / check_total as f32
    } else {
        0.0
    };
    let no_check_acc = if no_check_total > 0 {
        no_check_correct as f32 / no_check_total as f32
    } else {
        0.0
    };
    let delta_pp = (check_acc - no_check_acc) * 100.0;

    println!("\n  Proof 3: Convergence Check Quality");
    println!(
        "    With check:    {:.1}% ({}/{})",
        check_acc * 100.0,
        check_correct,
        check_total
    );
    println!(
        "    Without check: {:.1}% ({}/{})",
        no_check_acc * 100.0,
        no_check_correct,
        no_check_total
    );
    println!("    Δ (with - without): {delta_pp:+.1}pp");
    println!("    Note: steps_used reports max_steps; early exit not yet tracked in result");

    // Convergence check should not hurt quality by more than 15pp
    assert!(
        check_acc >= no_check_acc - 0.15,
        "convergence check accuracy ({:.1}%) more than 15pp below no-check ({:.1}%)",
        check_acc * 100.0,
        no_check_acc * 100.0,
    );

    println!("    GOAT gate: ✅ PASS — convergence check within ±15pp of no-check");
}

// ---------------------------------------------------------------------------
// Proof 4: Contiguous prefix preserves quality at τ=0
// ---------------------------------------------------------------------------

#[test]
fn proof_4_contiguous_prefix_at_zero_threshold() {
    let (config, weights, test_data) = setup_trained_model();
    let decode_config = make_decode_config();
    let block_size = decode_config.block_size;

    let n_blocks = 10usize;

    let soft_with_prefix = SoftDecodeConfig {
        decode_threshold: 0.0,
        contiguous_prefix: true,
        ..SoftDecodeConfig::default()
    };

    let soft_no_prefix = SoftDecodeConfig {
        decode_threshold: 0.0,
        contiguous_prefix: false,
        ..SoftDecodeConfig::default()
    };

    let mut prefix_correct = 0usize;
    let mut prefix_total = 0usize;
    let mut no_prefix_correct = 0usize;
    let mut no_prefix_total = 0usize;

    for (i, target) in test_data.iter().take(n_blocks).enumerate() {
        // With contiguous prefix
        {
            let mut ctx = D2fContext::new(&config);
            let mut rng = Rng::new(i as u64 * 400 + 5);
            let result = d2f_decode_block_soft(
                &mut ctx,
                &weights,
                &config,
                &decode_config,
                &NoPruner,
                &soft_with_prefix,
                &mut rng,
            );
            assert_valid_tokens(&result.tokens, config.vocab_size);
            let acc = block_accuracy(&result.tokens, target, block_size);
            prefix_correct += (acc * block_size as f32) as usize;
            prefix_total += block_size;
        }

        // Without contiguous prefix (all-confident)
        {
            let mut ctx = D2fContext::new(&config);
            let mut rng = Rng::new(i as u64 * 400 + 5);
            let result = d2f_decode_block_soft(
                &mut ctx,
                &weights,
                &config,
                &decode_config,
                &NoPruner,
                &soft_no_prefix,
                &mut rng,
            );
            assert_valid_tokens(&result.tokens, config.vocab_size);
            let acc = block_accuracy(&result.tokens, target, block_size);
            no_prefix_correct += (acc * block_size as f32) as usize;
            no_prefix_total += block_size;
        }
    }

    let prefix_acc = if prefix_total > 0 {
        prefix_correct as f32 / prefix_total as f32
    } else {
        0.0
    };
    let no_prefix_acc = if no_prefix_total > 0 {
        no_prefix_correct as f32 / no_prefix_total as f32
    } else {
        0.0
    };
    let delta_pp = (prefix_acc - no_prefix_acc) * 100.0;

    println!("\n  Proof 4: Contiguous Prefix at τ=0");
    println!(
        "    Contiguous prefix: {:.1}% ({}/{})",
        prefix_acc * 100.0,
        prefix_correct,
        prefix_total
    );
    println!(
        "    All-confident:     {:.1}% ({}/{})",
        no_prefix_acc * 100.0,
        no_prefix_correct,
        no_prefix_total
    );
    println!("    Δ (prefix - all):  {delta_pp:+.1}pp");

    // Contiguous prefix should not degrade quality by more than 10pp
    assert!(
        prefix_acc >= no_prefix_acc - 0.10,
        "contiguous prefix accuracy ({:.1}%) more than 10pp below all-confident ({:.1}%)",
        prefix_acc * 100.0,
        no_prefix_acc * 100.0,
    );

    println!("    GOAT gate: ✅ PASS — contiguous prefix within ±10pp of all-confident");
}

// ---------------------------------------------------------------------------
// Summary: Combined results table
// ---------------------------------------------------------------------------

#[test]
fn summary_dmax_spd_goat_results() {
    let (config, weights, test_data) = setup_trained_model();
    let decode_config = make_decode_config();
    let block_size = decode_config.block_size;

    // --- Proof 1: SPD vs Binary ---
    let soft_aggressive = SoftDecodeConfig::aggressive();
    let n_p1 = 20usize;
    let (p1_binary_acc, p1_spd_acc) = {
        let mut bin_c = 0usize;
        let mut bin_t = 0usize;
        let mut spd_c = 0usize;
        let mut spd_t = 0usize;
        for (i, target) in test_data.iter().take(n_p1).enumerate() {
            {
                let mut rng = Rng::new(i as u64 * 100 + 1);
                let r = d2f_decode_block(&weights, &config, &decode_config, &NoPruner, &mut rng);
                let a = block_accuracy(&r.tokens, target, block_size);
                bin_c += (a * block_size as f32) as usize;
                bin_t += block_size;
            }
            {
                let mut ctx = D2fContext::new(&config);
                let mut rng = Rng::new(i as u64 * 100 + 1);
                let r = d2f_decode_block_soft(
                    &mut ctx,
                    &weights,
                    &config,
                    &decode_config,
                    &NoPruner,
                    &soft_aggressive,
                    &mut rng,
                );
                let a = block_accuracy(&r.tokens, target, block_size);
                spd_c += (a * block_size as f32) as usize;
                spd_t += block_size;
            }
        }
        let ba = if bin_t > 0 {
            bin_c as f32 / bin_t as f32
        } else {
            0.0
        };
        let sa = if spd_t > 0 {
            spd_c as f32 / spd_t as f32
        } else {
            0.0
        };
        (ba, sa)
    };

    // --- Proof 2: Confidence ---
    let soft_default = SoftDecodeConfig::default();
    let n_p2 = 10usize;
    let p2_avg_confidence = {
        let mut total = 0.0f32;
        let mut count = 0usize;
        for i in 0..n_p2 {
            let mut ctx = D2fContext::new(&config);
            let mut rng = Rng::new(i as u64 * 200 + 7);
            let r = d2f_decode_block_soft(
                &mut ctx,
                &weights,
                &config,
                &decode_config,
                &NoPruner,
                &soft_default,
                &mut rng,
            );
            if let Some(&c) = r.confidence_history.first() {
                total += c;
                count += 1;
            }
        }
        if count > 0 { total / count as f32 } else { 0.0 }
    };

    // --- Proof 3: Convergence check ---
    let soft_with_check = SoftDecodeConfig {
        consistency_check: true,
        ..SoftDecodeConfig::default()
    };
    let soft_no_check = SoftDecodeConfig {
        consistency_check: false,
        ..SoftDecodeConfig::default()
    };
    let n_p3 = 10usize;
    let (p3_check_acc, p3_no_check_acc) = {
        let mut cc = 0usize;
        let mut ct = 0usize;
        let mut nc = 0usize;
        let mut nt = 0usize;
        for (i, target) in test_data.iter().take(n_p3).enumerate() {
            {
                let mut ctx = D2fContext::new(&config);
                let mut rng = Rng::new(i as u64 * 300 + 3);
                let r = d2f_decode_block_soft(
                    &mut ctx,
                    &weights,
                    &config,
                    &decode_config,
                    &NoPruner,
                    &soft_with_check,
                    &mut rng,
                );
                let a = block_accuracy(&r.tokens, target, block_size);
                cc += (a * block_size as f32) as usize;
                ct += block_size;
            }
            {
                let mut ctx = D2fContext::new(&config);
                let mut rng = Rng::new(i as u64 * 300 + 3);
                let r = d2f_decode_block_soft(
                    &mut ctx,
                    &weights,
                    &config,
                    &decode_config,
                    &NoPruner,
                    &soft_no_check,
                    &mut rng,
                );
                let a = block_accuracy(&r.tokens, target, block_size);
                nc += (a * block_size as f32) as usize;
                nt += block_size;
            }
        }
        let ca = if ct > 0 { cc as f32 / ct as f32 } else { 0.0 };
        let na = if nt > 0 { nc as f32 / nt as f32 } else { 0.0 };
        (ca, na)
    };

    // --- Proof 4: Contiguous prefix ---
    let soft_prefix = SoftDecodeConfig {
        decode_threshold: 0.0,
        contiguous_prefix: true,
        ..SoftDecodeConfig::default()
    };
    let soft_no_prefix = SoftDecodeConfig {
        decode_threshold: 0.0,
        contiguous_prefix: false,
        ..SoftDecodeConfig::default()
    };
    let n_p4 = 10usize;
    let (p4_prefix_acc, p4_no_prefix_acc) = {
        let mut pc = 0usize;
        let mut pt = 0usize;
        let mut nc = 0usize;
        let mut nt = 0usize;
        for (i, target) in test_data.iter().take(n_p4).enumerate() {
            {
                let mut ctx = D2fContext::new(&config);
                let mut rng = Rng::new(i as u64 * 400 + 5);
                let r = d2f_decode_block_soft(
                    &mut ctx,
                    &weights,
                    &config,
                    &decode_config,
                    &NoPruner,
                    &soft_prefix,
                    &mut rng,
                );
                let a = block_accuracy(&r.tokens, target, block_size);
                pc += (a * block_size as f32) as usize;
                pt += block_size;
            }
            {
                let mut ctx = D2fContext::new(&config);
                let mut rng = Rng::new(i as u64 * 400 + 5);
                let r = d2f_decode_block_soft(
                    &mut ctx,
                    &weights,
                    &config,
                    &decode_config,
                    &NoPruner,
                    &soft_no_prefix,
                    &mut rng,
                );
                let a = block_accuracy(&r.tokens, target, block_size);
                nc += (a * block_size as f32) as usize;
                nt += block_size;
            }
        }
        let pa = if pt > 0 { pc as f32 / pt as f32 } else { 0.0 };
        let na = if nt > 0 { nc as f32 / nt as f32 } else { 0.0 };
        (pa, na)
    };

    // --- Print summary table ---
    println!();
    println!("  ┌────────────────────────────────────────────────────────────────────────────┐");
    println!("  │ GOAT Proof 109 T7: DMax SPD Quality Comparison (micro_dllm)               │");
    println!("  ├────────────────────────────────────────────────────────────────────────────┤");
    println!("  │ Proof │ Metric                  │ Value A       │ Value B       │ Δ        │");
    println!("  │───────│─────────────────────────│───────────────│───────────────│──────────│");
    println!(
        "  │ P1    │ SPD vs Binary (aggr.)   │ SPD {:5.1}%     │ Bin {:5.1}%     │ {:+5.1}pp  │",
        p1_spd_acc * 100.0,
        p1_binary_acc * 100.0,
        (p1_spd_acc - p1_binary_acc) * 100.0,
    );
    println!(
        "  │ P2    │ Avg Confidence (default)│ {:5.1}%         │               │          │",
        p2_avg_confidence * 100.0,
    );
    println!(
        "  │ P3    │ Convergence Check       │ On  {:5.1}%    │ Off {:5.1}%    │ {:+5.1}pp  │",
        p3_check_acc * 100.0,
        p3_no_check_acc * 100.0,
        (p3_check_acc - p3_no_check_acc) * 100.0,
    );
    println!(
        "  │ P4    │ Contiguous Prefix τ=0   │ Yes {:5.1}%    │ No  {:5.1}%    │ {:+5.1}pp  │",
        p4_prefix_acc * 100.0,
        p4_no_prefix_acc * 100.0,
        (p4_prefix_acc - p4_no_prefix_acc) * 100.0,
    );
    println!("  └────────────────────────────────────────────────────────────────────────────┘");
    println!(
        "    Config: vocab={}, block={}, n_layer={}, n_head={}, decode_block={}",
        config.vocab_size,
        config.block_size,
        config.n_layer,
        config.n_head,
        decode_config.block_size,
    );
    println!("    Training: 200 epochs, lr=0.01, mask_ratio=0.3, seed=42");
    println!(
        "    Decode:   denoise_steps={}, confidence_threshold={}, temperature={}",
        decode_config.denoise_steps, decode_config.confidence_threshold, decode_config.temperature,
    );

    // Evaluate all gates
    let p1_pass = p1_spd_acc >= p1_binary_acc - 0.20;
    let p2_pass = p2_avg_confidence > 0.0;
    let p3_pass = p3_check_acc >= p3_no_check_acc - 0.15;
    let p4_pass = p4_prefix_acc >= p4_no_prefix_acc - 0.10;

    println!();
    println!("    Gates:");
    println!(
        "      P1 SPD vs Binary (±20pp):  {}",
        if p1_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "      P2 Confidence (>0.0):      {}",
        if p2_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "      P3 Convergence (±15pp):    {}",
        if p3_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "      P4 Prefix τ=0 (±10pp):    {}",
        if p4_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    assert!(p1_pass, "P1 gate failed: SPD accuracy below 20pp threshold");
    assert!(p2_pass, "P2 gate failed: confidence not > 0.0");
    assert!(
        p3_pass,
        "P3 gate failed: convergence check degraded quality >15pp"
    );
    assert!(
        p4_pass,
        "P4 gate failed: contiguous prefix degraded quality >10pp"
    );

    println!();
    println!("    Result: ✅ ALL GOAT PROOFS PASSED (4/4)");
}
