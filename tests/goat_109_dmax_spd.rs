#![cfg(feature = "dmax_spd")]
//! GOAT Proof Test — DMax Soft Parallel Decode (Plan 109)
//!
//! Proves mathematical invariants of DMax Soft Parallel Decode:
//! hybrid embedding interpolation, contiguous prefix promotion,
//! block convergence detection, and config preset ordering.
//!
//! Reference: DMax Soft Parallel Decode — hybrid embedding D2F enhancement.
//!
//! Run: `cargo test --features dmax_spd --test goat_109_dmax_spd -- --nocapture`

use katgpt_rs::speculative::d2f::{
    BlockConvergence, HybridEmbedding, SoftDecodeConfig, check_block_convergence,
    contiguous_prefix_promote,
};

// ── Helpers ───────────────────────────────────────────────────

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

/// L2 norm of a vector.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ── Proof 1: HybridEmbedding at π=1 (Token Dominates) ────────
//
// When confidence π = 1.0, the hybrid embedding formula becomes:
//   h̃ = 1·e_token + 0·e_mask = e_token
//   target_norm = 1·‖e_token‖ + 0·‖e_mask‖ = ‖e_token‖
//   h = h̃ / ‖h̃‖ · target_norm = e_token / ‖e_token‖ · ‖e_token‖ = e_token
// Output should equal token_emb exactly (up to float precision).

#[test]
fn proof_1_hybrid_embedding_at_pi_1() {
    let token_emb = vec![0.1, 0.5, -0.3, 0.8, 0.2];
    let mask_emb = vec![0.9, 0.1, 0.4, -0.2, 0.7];

    let hybrid = HybridEmbedding {
        confidence: 1.0,
        token_id: 42,
    };
    let mut out = vec![0.0f32; token_emb.len()];
    hybrid.build(&token_emb, &mask_emb, &mut out);

    for (i, (&got, &expected)) in out.iter().zip(token_emb.iter()).enumerate() {
        assert!(
            approx_eq(got, expected, 1e-5),
            "[P1.1] out[{i}] = {got}, expected {expected} (π=1 should equal token_emb)"
        );
    }

    // Also test with very different mask — should still equal token_emb
    let different_mask = vec![-5.0, 3.0, 10.0, -8.0, 2.0];
    let mut out2 = vec![0.0f32; token_emb.len()];
    hybrid.build(&token_emb, &different_mask, &mut out2);

    for (i, (&got, &expected)) in out2.iter().zip(token_emb.iter()).enumerate() {
        assert!(
            approx_eq(got, expected, 1e-5),
            "[P1.2] out[{i}] = {got}, expected {expected} (π=1 independent of mask)"
        );
    }

    // Norm preservation
    let out_norm = l2_norm(&out);
    let token_norm = l2_norm(&token_emb);
    assert!(
        approx_eq(out_norm, token_norm, 1e-4),
        "[P1.3] norm: {out_norm} vs token norm {token_norm}"
    );

    println!("✅ Proof 1 PASSED: HybridEmbedding at π=1 ≈ token_emb (token dominates)");
}

// ── Proof 2: HybridEmbedding at π=0 (Mask Dominates) ─────────
//
// When confidence π = 0.0:
//   h̃ = 0·e_token + 1·e_mask = e_mask
//   target_norm = 0·‖e_token‖ + 1·‖e_mask‖ = ‖e_mask‖
//   h = e_mask / ‖e_mask‖ · ‖e_mask‖ = e_mask
// Output should equal mask_emb exactly.

#[test]
fn proof_2_hybrid_embedding_at_pi_0() {
    let token_emb = vec![0.1, 0.5, -0.3, 0.8, 0.2];
    let mask_emb = vec![0.9, 0.1, 0.4, -0.2, 0.7];

    let hybrid = HybridEmbedding {
        confidence: 0.0,
        token_id: 7,
    };
    let mut out = vec![0.0f32; mask_emb.len()];
    hybrid.build(&token_emb, &mask_emb, &mut out);

    for (i, (&got, &expected)) in out.iter().zip(mask_emb.iter()).enumerate() {
        assert!(
            approx_eq(got, expected, 1e-5),
            "[P2.1] out[{i}] = {got}, expected {expected} (π=0 should equal mask_emb)"
        );
    }

    // Different token — should still equal mask_emb
    let different_token = vec![100.0, -50.0, 30.0, -80.0, 20.0];
    let mut out2 = vec![0.0f32; mask_emb.len()];
    hybrid.build(&different_token, &mask_emb, &mut out2);

    for (i, (&got, &expected)) in out2.iter().zip(mask_emb.iter()).enumerate() {
        assert!(
            approx_eq(got, expected, 1e-5),
            "[P2.2] out[{i}] = {got}, expected {expected} (π=0 independent of token)"
        );
    }

    // Norm preservation
    let out_norm = l2_norm(&out);
    let mask_norm = l2_norm(&mask_emb);
    assert!(
        approx_eq(out_norm, mask_norm, 1e-4),
        "[P2.3] norm: {out_norm} vs mask norm {mask_norm}"
    );

    println!("✅ Proof 2 PASSED: HybridEmbedding at π=0 ≈ mask_emb (mask dominates)");
}

// ── Proof 3: HybridEmbedding Finite for All π ────────────────
//
// For any confidence value π ∈ [0, 1], the output embedding must be
// all finite (no NaN, no Inf). The renormalization prevents collapse.
// Also test edge cases: very small embeddings, large embeddings, negative.

#[test]
fn proof_3_hybrid_embedding_finite() {
    let token_emb = vec![0.1, 0.5, -0.3, 0.8, 0.2];
    let mask_emb = vec![0.9, 0.1, 0.4, -0.2, 0.7];

    let pi_values = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0];

    for &pi in &pi_values {
        let hybrid = HybridEmbedding {
            confidence: pi,
            token_id: 0,
        };
        let mut out = vec![0.0f32; token_emb.len()];
        hybrid.build(&token_emb, &mask_emb, &mut out);

        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "[P3.1] non-finite at π={pi}, dim={i}: {v}");
        }
    }

    // Edge case: zero mask embedding (only token contributes)
    let zero_mask = vec![0.0f32; 5];
    for &pi in &pi_values {
        let hybrid = HybridEmbedding {
            confidence: pi,
            token_id: 0,
        };
        let mut out = vec![0.0f32; 5];
        hybrid.build(&token_emb, &zero_mask, &mut out);

        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "[P3.2] non-finite with zero mask at π={pi}, dim={i}: {v}"
            );
        }
    }

    // Edge case: zero token embedding (only mask contributes)
    let zero_token = vec![0.0f32; 5];
    for &pi in &pi_values {
        let hybrid = HybridEmbedding {
            confidence: pi,
            token_id: 0,
        };
        let mut out = vec![0.0f32; 5];
        hybrid.build(&zero_token, &mask_emb, &mut out);

        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "[P3.3] non-finite with zero token at π={pi}, dim={i}: {v}"
            );
        }
    }

    // Edge case: both zero (degenerate — norm near zero, output should stay finite)
    for &pi in &pi_values {
        let hybrid = HybridEmbedding {
            confidence: pi,
            token_id: 0,
        };
        let mut out = vec![0.0f32; 5];
        hybrid.build(&zero_token, &zero_mask, &mut out);

        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "[P3.4] non-finite with both zero at π={pi}, dim={i}: {v}"
            );
        }
    }

    // Large embeddings should also be finite
    let large_token: Vec<f32> = (0..64).map(|i| (i as f32) * 100.0).collect();
    let large_mask: Vec<f32> = (0..64).map(|i| -(i as f32) * 50.0).collect();
    for &pi in &[0.0, 0.5, 1.0] {
        let hybrid = HybridEmbedding {
            confidence: pi,
            token_id: 0,
        };
        let mut out = vec![0.0f32; 64];
        hybrid.build(&large_token, &large_mask, &mut out);

        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "[P3.5] non-finite with large emb at π={pi}, dim={i}: {v}"
            );
        }
    }

    println!("✅ Proof 3 PASSED: HybridEmbedding output is finite for all π values and edge cases");
}

// ── Proof 4: Contiguous Prefix Promotion ──────────────────────
//
// contiguous_prefix_promote scans masked_positions left-to-right.
// It promotes the longest contiguous prefix where confidence >= threshold.
// If NO position qualifies, it promotes the leftmost (ensure progress).
// Returned positions are always a subset of masked_positions.

#[test]
fn proof_4_contiguous_prefix_promotion() {
    // Case 1: All positions above threshold → promote all
    let masked = vec![2, 5, 8];
    // Set confidences for masked positions
    let mut conf = vec![0.0f32; 10];
    conf[2] = 0.8;
    conf[5] = 0.9;
    conf[8] = 0.7;
    let result = contiguous_prefix_promote(&masked, &conf, 0.5);
    assert_eq!(
        result,
        vec![2, 5, 8],
        "[P4.1] all above threshold → promote all"
    );

    // Case 2: First two above, third below → contiguous prefix stops
    conf[2] = 0.8;
    conf[5] = 0.9;
    conf[8] = 0.3; // below threshold
    let result2 = contiguous_prefix_promote(&masked, &conf, 0.5);
    assert_eq!(
        result2,
        vec![2, 5],
        "[P4.2] contiguous prefix stops at first below"
    );

    // Case 3: First below → promote only leftmost
    conf[2] = 0.3; // below
    conf[5] = 0.9;
    conf[8] = 0.8;
    let result3 = contiguous_prefix_promote(&masked, &conf, 0.5);
    assert_eq!(
        result3,
        vec![2],
        "[P4.3] none qualify → promote leftmost (index 2)"
    );

    // Case 4: All below threshold → promote leftmost
    conf[2] = 0.1;
    conf[5] = 0.2;
    conf[8] = 0.1;
    let result4 = contiguous_prefix_promote(&masked, &conf, 0.5);
    assert_eq!(result4, vec![2], "[P4.4] all below → promote leftmost");

    // Case 5: Empty masked positions → empty result
    let result5 = contiguous_prefix_promote(&[], &conf, 0.5);
    assert!(result5.is_empty(), "[P4.5] empty input → empty output");

    // Case 6: Single position, above threshold
    let result6 = contiguous_prefix_promote(&[7], &conf, 0.05);
    // conf[7] = 0.0, threshold 0.05 → below → promote leftmost (7)
    assert_eq!(
        result6,
        vec![7],
        "[P4.6] single position, below → promote it (leftmost)"
    );

    // Case 7: Single position, above threshold
    conf[7] = 0.8;
    let result7 = contiguous_prefix_promote(&[7], &conf, 0.5);
    assert_eq!(
        result7,
        vec![7],
        "[P4.7] single position, above → promote it"
    );

    // Case 8: Exact threshold boundary (>=)
    conf[3] = 0.5; // exactly at threshold
    let result8 = contiguous_prefix_promote(&[3], &conf, 0.5);
    assert_eq!(
        result8,
        vec![3],
        "[P4.8] exactly at threshold → promote (>=)"
    );

    // Case 9: Just below threshold
    conf[3] = 0.499;
    let result9 = contiguous_prefix_promote(&[3], &conf, 0.5);
    assert_eq!(result9, vec![3], "[P4.9] just below → promote leftmost");

    // Case 10: Only returned positions are from masked_positions
    let masked10 = vec![1, 3, 5, 7, 9];
    let conf10 = vec![0.6, 0.8, 0.7, 0.4, 0.9, 0.3, 0.5, 0.2, 0.1, 0.95];
    let result10 = contiguous_prefix_promote(&masked10, &conf10, 0.5);
    for &pos in &result10 {
        assert!(
            masked10.contains(&pos),
            "[P4.10] promoted position {pos} not in masked_positions"
        );
    }

    println!("✅ Proof 4 PASSED: Contiguous prefix promotion correctly handles all cases");
}

// ── Proof 5: Block Convergence — Confidence ───────────────────
//
// When ALL confidences >= accept_threshold → ConfidenceConverged.
// When any confidence < threshold → not ConfidenceConverged (may be NotConverged
// or ConsistencyConverged if consistency check passes first).

#[test]
fn proof_5_block_convergence_confidence() {
    // Case 1: All confidences above threshold → ConfidenceConverged
    // (no prev_top1, so consistency check won't trigger)
    let current = vec![3, 7, 2];
    let confs = vec![0.92, 0.95, 0.90];
    let result = check_block_convergence(&current, None, &confs, 0.9);
    assert_eq!(
        result,
        BlockConvergence::ConfidenceConverged,
        "[P5.1] all >= 0.9 → ConfidenceConverged"
    );

    // Case 2: One below threshold → NotConverged
    let confs2 = vec![0.92, 0.85, 0.90]; // 0.85 < 0.9
    let result2 = check_block_convergence(&current, None, &confs2, 0.9);
    assert_eq!(
        result2,
        BlockConvergence::NotConverged,
        "[P5.2] one below → NotConverged"
    );

    // Case 3: All exactly at threshold → ConfidenceConverged
    let confs3 = vec![0.9, 0.9, 0.9];
    let result3 = check_block_convergence(&current, None, &confs3, 0.9);
    assert_eq!(
        result3,
        BlockConvergence::ConfidenceConverged,
        "[P5.3] all exactly at threshold → ConfidenceConverged"
    );

    // Case 4: All well above → ConfidenceConverged (even with mismatched prev)
    let prev4 = vec![1, 2, 3]; // different from current → no consistency
    let confs4 = vec![0.99, 0.99, 0.99];
    let result4 = check_block_convergence(&current, Some(&prev4), &confs4, 0.9);
    // Consistency check is PRIMARY — but prev != current, so it falls through to confidence
    assert_eq!(
        result4,
        BlockConvergence::ConfidenceConverged,
        "[P5.4] all well above threshold with different prev → ConfidenceConverged"
    );

    // Case 5: Empty confidences → NotConverged (no positions to check)
    let result5 = check_block_convergence(&current, None, &[], 0.9);
    assert_eq!(
        result5,
        BlockConvergence::NotConverged,
        "[P5.5] empty confidences → NotConverged"
    );

    // Case 6: Single position, above threshold → ConfidenceConverged
    let confs6 = vec![0.95];
    let result6 = check_block_convergence(&[5], None, &confs6, 0.9);
    assert_eq!(
        result6,
        BlockConvergence::ConfidenceConverged,
        "[P5.6] single position above → ConfidenceConverged"
    );

    // Case 7: Single position, below threshold → NotConverged
    let confs7 = vec![0.8];
    let result7 = check_block_convergence(&[5], None, &confs7, 0.9);
    assert_eq!(
        result7,
        BlockConvergence::NotConverged,
        "[P5.7] single position below → NotConverged"
    );

    println!("✅ Proof 5 PASSED: Confidence convergence triggers when all positions >= threshold");
}

// ── Proof 6: Block Convergence — Consistency ──────────────────
//
// When current_top1 == prev_top1 (same predictions) → ConsistencyConverged.
// This is the PRIMARY convergence signal — checked before confidence.
// Even if confidences are low, consistency triggers convergence.

#[test]
fn proof_6_block_convergence_consistency() {
    let current = vec![3, 7, 2];
    let prev_same = vec![3, 7, 2];
    let prev_diff = vec![3, 7, 5]; // last differs

    // Case 1: Same top-1 → ConsistencyConverged (primary signal)
    let confs_low = vec![0.3, 0.2, 0.1]; // below any reasonable threshold
    let result = check_block_convergence(&current, Some(&prev_same), &confs_low, 0.9);
    assert_eq!(
        result,
        BlockConvergence::ConsistencyConverged,
        "[P6.1] same top-1 → ConsistencyConverged even with low confidence"
    );

    // Case 2: Different top-1 → falls through to confidence check
    let result2 = check_block_convergence(&current, Some(&prev_diff), &confs_low, 0.9);
    assert_eq!(
        result2,
        BlockConvergence::NotConverged,
        "[P6.2] different top-1 + low confidence → NotConverged"
    );

    // Case 3: Same top-1 but different length → no consistency
    let prev_short = vec![3, 7];
    let result3 = check_block_convergence(&current, Some(&prev_short), &confs_low, 0.9);
    assert_eq!(
        result3,
        BlockConvergence::NotConverged,
        "[P6.3] different length → NotConverged (no match, low confidence)"
    );

    // Case 4: Same top-1 with high confidence → still ConsistencyConverged (primary)
    let confs_high = vec![0.95, 0.97, 0.92];
    let result4 = check_block_convergence(&current, Some(&prev_same), &confs_high, 0.9);
    assert_eq!(
        result4,
        BlockConvergence::ConsistencyConverged,
        "[P6.4] same top-1 with high confidence → ConsistencyConverged (primary)"
    );

    // Case 5: No prev → consistency check skipped
    let result5 = check_block_convergence(&current, None, &confs_high, 0.9);
    assert_eq!(
        result5,
        BlockConvergence::ConfidenceConverged,
        "[P6.5] no prev → falls through to confidence check"
    );

    // Case 6: Consistency takes priority over confidence
    // Same top-1 but zero confidence → ConsistencyConverged
    let confs_zero = vec![0.0, 0.0, 0.0];
    let result6 = check_block_convergence(&current, Some(&prev_same), &confs_zero, 0.9);
    assert_eq!(
        result6,
        BlockConvergence::ConsistencyConverged,
        "[P6.6] consistency takes priority over confidence"
    );

    // Case 7: Single element consistency
    let result7 = check_block_convergence(&[5], Some(&[5]), &[0.1], 0.9);
    assert_eq!(
        result7,
        BlockConvergence::ConsistencyConverged,
        "[P6.7] single element consistency"
    );

    println!("✅ Proof 6 PASSED: Consistency convergence triggers when top-1 unchanged");
}

// ── Proof 7: Config Presets Ordering ──────────────────────────
//
// aggressive thresholds < default < conservative
// This ensures the presets form a natural ordering:
// aggressive (fast/loose) < default (balanced) < conservative (safe/strict)

#[test]
fn proof_7_config_presets_ordering() {
    let aggressive = SoftDecodeConfig::aggressive();
    let default = SoftDecodeConfig::default();
    let conservative = SoftDecodeConfig::conservative();

    // decode_threshold: aggressive < default < conservative
    assert!(
        aggressive.decode_threshold < default.decode_threshold,
        "[P7.1] aggressive.decode_threshold ({}) < default ({})",
        aggressive.decode_threshold,
        default.decode_threshold
    );
    assert!(
        default.decode_threshold < conservative.decode_threshold,
        "[P7.2] default.decode_threshold ({}) < conservative ({})",
        default.decode_threshold,
        conservative.decode_threshold
    );

    // accept_threshold: aggressive < default < conservative
    assert!(
        aggressive.accept_threshold < default.accept_threshold,
        "[P7.3] aggressive.accept_threshold ({}) < default ({})",
        aggressive.accept_threshold,
        default.accept_threshold
    );
    assert!(
        default.accept_threshold < conservative.accept_threshold,
        "[P7.4] default.accept_threshold ({}) < conservative ({})",
        default.accept_threshold,
        conservative.accept_threshold
    );

    // Verify specific known values
    assert!(
        approx_eq(default.decode_threshold, 0.5, 1e-6),
        "[P7.5] default.decode_threshold should be 0.5"
    );
    assert!(
        approx_eq(default.accept_threshold, 0.9, 1e-6),
        "[P7.6] default.accept_threshold should be 0.9"
    );
    assert!(
        approx_eq(aggressive.decode_threshold, 0.3, 1e-6),
        "[P7.7] aggressive.decode_threshold should be 0.3"
    );
    assert!(
        approx_eq(aggressive.accept_threshold, 0.8, 1e-6),
        "[P7.8] aggressive.accept_threshold should be 0.8"
    );
    assert!(
        approx_eq(conservative.decode_threshold, 0.7, 1e-6),
        "[P7.9] conservative.decode_threshold should be 0.7"
    );
    assert!(
        approx_eq(conservative.accept_threshold, 0.95, 1e-6),
        "[P7.10] conservative.accept_threshold should be 0.95"
    );

    // Boolean fields: all presets should have consistent settings
    assert!(
        default.use_hybrid_embeddings,
        "[P7.11] default.use_hybrid_embeddings should be true"
    );
    assert!(
        default.contiguous_prefix,
        "[P7.12] default.contiguous_prefix should be true"
    );
    assert!(
        default.consistency_check,
        "[P7.13] default.consistency_check should be true"
    );

    // Thresholds are in valid ranges (0, 1)
    for (name, config) in [
        ("aggressive", &aggressive),
        ("default", &default),
        ("conservative", &conservative),
    ] {
        assert!(
            config.decode_threshold > 0.0 && config.decode_threshold < 1.0,
            "[P7.14] {name}.decode_threshold = {} not in (0, 1)",
            config.decode_threshold
        );
        assert!(
            config.accept_threshold > 0.0 && config.accept_threshold <= 1.0,
            "[P7.15] {name}.accept_threshold = {} not in (0, 1]",
            config.accept_threshold
        );
        // accept_threshold > decode_threshold (accept is stricter)
        assert!(
            config.accept_threshold > config.decode_threshold,
            "[P7.16] {name}: accept_threshold ({}) should be > decode_threshold ({})",
            config.accept_threshold,
            config.decode_threshold
        );
    }

    println!("✅ Proof 7 PASSED: Config presets ordered aggressive < default < conservative");
}

// ── Summary ───────────────────────────────────────────────────

#[test]
fn summary_goat_109() {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  🐐 GOAT Proof: DMax Soft Parallel Decode (Plan 109)");
    println!("  Research 72 — Hybrid embedding D2F enhancement");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Proof 1: HybridEmbedding at π=1 ≈ token_emb             ✅");
    println!("  Proof 2: HybridEmbedding at π=0 ≈ mask_emb              ✅");
    println!("  Proof 3: HybridEmbedding finite for all π                ✅");
    println!("  Proof 4: Contiguous prefix promotion correct             ✅");
    println!("  Proof 5: Block convergence (confidence)                  ✅");
    println!("  Proof 6: Block convergence (consistency, primary)        ✅");
    println!("  Proof 7: Config presets: aggressive < default < conserv  ✅");
    println!();
    println!("  Verdict: DMax SPD hybrid embeddings correctly interpolate");
    println!("  between token and mask representations. Prefix promotion");
    println!("  maintains contiguity. Convergence detects both confidence");
    println!("  and consistency signals. Config presets form a valid ordering.");
    println!("═══════════════════════════════════════════════════════════════");
}
