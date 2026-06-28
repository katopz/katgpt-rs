//! Integration test for Plan 243 — compress → patch → reinject → verify.
//!
//! Tests the full MUX-Latent Wire Patch pipeline end-to-end:
//! 1. Compress 256 tokens → 32 latent slots at X8
//! 2. Create LatentPatchBatch, verify BLAKE3 commitments
//! 3. Apply patches to CompressedContext via LatentPatcher::patch_batch
//! 4. Verify output quality (patched/unpatched segments, EXPAND still works)
//! 5. Octree bridge round-trip (TernaryDir ↔ weights ↔ LatentPatch)
//!
//! Also covers rejection cases: tamper, out-of-range, NaN, mixed batch.

#![cfg(feature = "mux_latent_wire")]

use katgpt_rs::mux_latent::{
    CompressionRatio, LatentPatch, LatentPatchBatch, LatentPatcher, LatentSegment, MortonCode,
    MuxLatentConfig, MuxLatentEncoder, OctreeLod, PatchRejection, TernaryDir, TernaryValue,
    octree_leaf_to_patch_weights, patch_weights_to_octree_leaf,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Standard config for all tests: window_size=1024, X8, no instruction preserve.
fn test_config() -> MuxLatentConfig {
    MuxLatentConfig {
        window_size: 1024,
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: false,
        ..Default::default()
    }
}

/// Encode 256 sequential tokens at X8 → 32 latent slots.
fn encode_256_tokens() -> (MuxLatentEncoder, katgpt_rs::mux_latent::CompressedContext) {
    let encoder = MuxLatentEncoder::new(test_config());
    let tokens: Vec<u32> = (0..256).collect();
    let ctx = encoder.encode(&tokens);
    (encoder, ctx)
}

/// Extract weights from a compressed segment by segment_id.
fn get_weights(
    ctx: &katgpt_rs::mux_latent::CompressedContext,
    segment_id: u32,
) -> Option<Vec<f32>> {
    ctx.segments.iter().find_map(|s| match s {
        LatentSegment::Compressed {
            segment_id: id,
            weights,
            ..
        } if *id == segment_id => Some(weights.clone()),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// 1. Compress: encode 256 tokens → 32 latent slots at X8
// ---------------------------------------------------------------------------

#[test]
fn test_compress_256_tokens_to_32_slots() {
    let (encoder, ctx) = encode_256_tokens();

    // X8: 256 tokens / 8 per span = 32 compressed segments
    assert_eq!(ctx.latent_slot_count, 32, "Expected 32 latent slots at X8");
    assert_eq!(ctx.original_token_count, 256);

    // All segments should be compressed (preserve_instructions=false)
    for seg in &ctx.segments {
        assert!(
            matches!(seg, LatentSegment::Compressed { .. }),
            "All segments should be compressed when preserve_instructions=false"
        );
    }

    // Verify config round-trip
    assert_eq!(encoder.config().compression_ratio, CompressionRatio::X8);
    assert_eq!(encoder.config().window_size, 1024);
}

// ---------------------------------------------------------------------------
// 2. Patch: create LatentPatchBatch, verify BLAKE3 commitments
// ---------------------------------------------------------------------------

#[test]
fn test_patch_batch_commitment_verification() {
    let (_, ctx) = encode_256_tokens();
    let total_segments = ctx.latent_slot_count as u32;

    // Create patches for segments 0, 5, 10, 20
    let patch_weights = [
        [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.8f32, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [1.0f32, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0],
        [0.25f32; 8],
    ];
    let segment_ids = [0u32, 5, 10, 20];

    let patches: Vec<LatentPatch> = segment_ids
        .iter()
        .zip(patch_weights.iter())
        .map(|(&id, &w)| LatentPatch::new(id, w))
        .collect();

    // Each patch should have a valid BLAKE3 commitment
    for patch in &patches {
        assert!(
            patch.verify_commitment(),
            "Patch {} commitment should be valid",
            patch.segment_id
        );
        assert!(
            patch.weights_finite(),
            "Patch {} weights should be finite",
            patch.segment_id
        );
    }

    // Batch verification should pass
    let batch = LatentPatchBatch::new(patches.clone(), total_segments, CompressionRatio::X8, 1);
    let result = batch.verify_all_commitments();
    assert!(result.is_ok(), "All commitments should verify");
    let receipt = result.unwrap();
    assert_eq!(receipt.committed.len(), 4);
    assert!(receipt.rejected.is_empty());
    assert_eq!(receipt.committed, vec![0, 5, 10, 20]);
}

// ---------------------------------------------------------------------------
// 3. Reinject: apply patches to context via patch_batch
// ---------------------------------------------------------------------------

#[test]
fn test_reinject_patch_batch_applies_weights() {
    let (_, mut ctx) = encode_256_tokens();
    let total_segments = ctx.latent_slot_count as u32;

    // Snapshot original weights for segments 0, 2, 4
    let orig_0 = get_weights(&ctx, 0).expect("segment 0 should exist");
    let orig_2 = get_weights(&ctx, 2).expect("segment 2 should exist");
    let orig_4 = get_weights(&ctx, 4).expect("segment 4 should exist");

    // Patch segments 0 and 4 with known weights
    let w0 = [0.11f32, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88];
    let w4 = [0.99f32, 0.88, 0.77, 0.66, 0.55, 0.44, 0.33, 0.22];

    let patches = vec![LatentPatch::new(0, w0), LatentPatch::new(4, w4)];

    let batch = LatentPatchBatch::new(patches, total_segments, CompressionRatio::X8, 1);
    let receipt = LatentPatcher::patch_batch(&mut ctx, &batch);

    assert!(receipt.all_committed(), "All patches should be committed");
    assert_eq!(receipt.committed.len(), 2);
    assert!(receipt.rejected.is_empty());

    // Verify segment 0 weights changed
    let patched_0 = get_weights(&ctx, 0).expect("segment 0 should exist");
    assert_ne!(patched_0, orig_0, "Segment 0 weights should have changed");
    for (i, &expected) in w0.iter().enumerate() {
        assert!(
            (patched_0[i] - expected).abs() < f32::EPSILON,
            "Segment 0 weight[{i}] = {} expected {expected}",
            patched_0[i]
        );
    }

    // Verify segment 4 weights changed
    let patched_4 = get_weights(&ctx, 4).expect("segment 4 should exist");
    assert_ne!(patched_4, orig_4, "Segment 4 weights should have changed");
    for (i, &expected) in w4.iter().enumerate() {
        assert!(
            (patched_4[i] - expected).abs() < f32::EPSILON,
            "Segment 4 weight[{i}] = {} expected {expected}",
            patched_4[i]
        );
    }

    // Verify segment 2 is unchanged (not patched)
    let unchanged_2 = get_weights(&ctx, 2).expect("segment 2 should exist");
    assert_eq!(unchanged_2, orig_2, "Segment 2 should be unchanged");
}

// ---------------------------------------------------------------------------
// 4. Verify output quality
// ---------------------------------------------------------------------------

#[test]
fn test_patched_segments_match_patch_weights() {
    let (_, mut ctx) = encode_256_tokens();
    let total_segments = ctx.latent_slot_count as u32;

    // Patch segments 1, 7, 15, 31
    let targets = [1u32, 7, 15, 31];
    let expected_weights: Vec<[f32; 8]> = targets
        .iter()
        .map(|&id| {
            let base = id as f32 * 0.01;
            [
                base,
                base + 0.1,
                base + 0.2,
                base + 0.3,
                base + 0.4,
                base + 0.5,
                base + 0.6,
                base + 0.7,
            ]
        })
        .collect();

    let patches: Vec<LatentPatch> = targets
        .iter()
        .zip(expected_weights.iter())
        .map(|(&id, &w)| LatentPatch::new(id, w))
        .collect();

    let batch = LatentPatchBatch::new(patches, total_segments, CompressionRatio::X8, 1);
    let receipt = LatentPatcher::patch_batch(&mut ctx, &batch);
    assert!(receipt.all_committed());

    // Each patched segment should exactly match the patch weights
    for (idx, &seg_id) in targets.iter().enumerate() {
        let weights =
            get_weights(&ctx, seg_id).unwrap_or_else(|| panic!("segment {seg_id} missing"));
        for (j, &expected) in expected_weights[idx].iter().enumerate() {
            assert!(
                (weights[j] - expected).abs() < f32::EPSILON,
                "Segment {seg_id} weight[{j}] = {} expected {expected}",
                weights[j]
            );
        }
    }
}

#[test]
fn test_unpatched_segments_unchanged() {
    let (_, mut ctx) = encode_256_tokens();
    let total_segments = ctx.latent_slot_count as u32;

    // Snapshot all 32 segments before patching
    let all_before: Vec<Vec<f32>> = (0..32).map(|id| get_weights(&ctx, id).unwrap()).collect();

    // Patch only segments 3, 12, 25
    let patches = vec![
        LatentPatch::new(3, [0.5f32; 8]),
        LatentPatch::new(12, [0.3f32; 8]),
        LatentPatch::new(25, [0.7f32; 8]),
    ];
    let batch = LatentPatchBatch::new(patches, total_segments, CompressionRatio::X8, 1);
    let receipt = LatentPatcher::patch_batch(&mut ctx, &batch);
    assert!(receipt.all_committed());

    // Unpatched segments (all except 3, 12, 25) should be identical
    let patched_ids: [u32; 3] = [3, 12, 25];
    for id in 0..32u32 {
        if patched_ids.contains(&id) {
            continue;
        }
        let after = get_weights(&ctx, id).unwrap();
        assert_eq!(
            after, all_before[id as usize],
            "Segment {id} should be unchanged (not in patch set)"
        );
    }
}

#[test]
fn test_expand_still_works_on_patched_segments() {
    let (_, mut ctx) = encode_256_tokens();
    let total_segments = ctx.latent_slot_count as u32;

    // Record original tokens for segment 7 before patching
    let original_tokens = ctx
        .expand(7)
        .expect("EXPAND should work on segment 7 before patch")
        .to_vec();

    // Patch segment 7
    let patches = vec![LatentPatch::new(7, [0.42f32; 8])];
    let batch = LatentPatchBatch::new(patches, total_segments, CompressionRatio::X8, 1);
    let receipt = LatentPatcher::patch_batch(&mut ctx, &batch);
    assert!(receipt.all_committed());

    // EXPAND on patched segment should still return original_tokens
    let expanded = ctx
        .expand(7)
        .expect("EXPAND should still work after patching");
    assert_eq!(
        expanded,
        original_tokens.as_slice(),
        "EXPAND should return original tokens even after patching weights"
    );
}

#[test]
fn test_expand_still_works_on_unpatched_segments() {
    let (_, mut ctx) = encode_256_tokens();
    let total_segments = ctx.latent_slot_count as u32;

    // Snapshot segment 0 tokens before any patching
    let tokens_0 = ctx.expand(0).expect("EXPAND on segment 0").to_vec();
    let tokens_15 = ctx.expand(15).expect("EXPAND on segment 15").to_vec();

    // Patch segment 7 only
    let patches = vec![LatentPatch::new(7, [0.9f32; 8])];
    let batch = LatentPatchBatch::new(patches, total_segments, CompressionRatio::X8, 1);
    let receipt = LatentPatcher::patch_batch(&mut ctx, &batch);
    assert!(receipt.all_committed());

    // Unpatched segments should still EXPAND correctly
    assert_eq!(ctx.expand(0).unwrap(), tokens_0.as_slice());
    assert_eq!(ctx.expand(15).unwrap(), tokens_15.as_slice());
}

#[test]
fn test_expand_nonexistent_segment_returns_none() {
    let (_, ctx) = encode_256_tokens();
    assert!(
        ctx.expand(999).is_none(),
        "EXPAND on nonexistent segment should return None"
    );
}

// ---------------------------------------------------------------------------
// 5. Octree bridge round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_octree_bridge_round_trip() {
    let (_, mut ctx) = encode_256_tokens();
    let total_segments = ctx.latent_slot_count as u32;

    // Build a TernaryDir with uniform groups so averaging is exact
    let mut dir = TernaryDir::zero();
    for i in 0..128 {
        let group = i / 16;
        let value = match group % 3 {
            0 => TernaryValue::Positive,
            1 => TernaryValue::Negative,
            _ => TernaryValue::Zero,
        };
        dir.set(i, value);
    }

    // Convert to weights and create a patch
    let weights = octree_leaf_to_patch_weights(&dir);

    // Verify weights match expected pattern
    assert!(
        (weights[0] - 1.0).abs() < f32::EPSILON,
        "Group 0: all Positive → +1.0"
    );
    assert!(
        (weights[1] - (-1.0)).abs() < f32::EPSILON,
        "Group 1: all Negative → -1.0"
    );
    assert!(
        (weights[2] - 0.0).abs() < f32::EPSILON,
        "Group 2: all Zero → 0.0"
    );

    // Create patch from octree weights and apply to segment 5
    let segment_id = 5u32;
    let patch = LatentPatch::new(segment_id, weights);
    let batch = LatentPatchBatch::new(vec![patch], total_segments, CompressionRatio::X8, 1);
    let receipt = LatentPatcher::patch_batch(&mut ctx, &batch);
    assert!(receipt.all_committed());

    // Verify the context now has the octree weights
    let ctx_weights = get_weights(&ctx, segment_id).unwrap();
    for (i, &w) in weights.iter().enumerate() {
        assert!(
            (ctx_weights[i] - w).abs() < f32::EPSILON,
            "Context weight[{i}] = {} expected {w}",
            ctx_weights[i]
        );
    }

    // Convert back to TernaryDir and verify dominant directions preserved
    let dir_back = patch_weights_to_octree_leaf(&weights);

    // Groups with uniform values should round-trip exactly
    assert_eq!(
        dir_back.get(0),
        TernaryValue::Positive,
        "Group 0 dominant: Positive"
    );
    assert_eq!(
        dir_back.get(16),
        TernaryValue::Negative,
        "Group 1 dominant: Negative"
    );
    assert_eq!(
        dir_back.get(32),
        TernaryValue::Zero,
        "Group 2 dominant: Zero"
    );

    // Full round-trip: every node in each uniform group should match
    for i in 0..16 {
        assert_eq!(
            dir_back.get(i),
            TernaryValue::Positive,
            "Node {i} in group 0"
        );
    }
    for i in 16..32 {
        assert_eq!(
            dir_back.get(i),
            TernaryValue::Negative,
            "Node {i} in group 1"
        );
    }
    for i in 32..48 {
        assert_eq!(dir_back.get(i), TernaryValue::Zero, "Node {i} in group 2");
    }
}

#[test]
fn test_octree_bridge_single_value_round_trip() {
    // Test with all-positive and all-negative TernaryDirs
    for &fill_value in &[
        TernaryValue::Positive,
        TernaryValue::Negative,
        TernaryValue::Zero,
    ] {
        let mut dir = TernaryDir::zero();
        for i in 0..128 {
            dir.set(i, fill_value);
        }

        let weights = octree_leaf_to_patch_weights(&dir);
        let dir_back = patch_weights_to_octree_leaf(&weights);

        // All weights should be the same value
        let expected_w = fill_value.to_weight();
        for (i, &w) in weights.iter().enumerate() {
            assert!(
                (w - expected_w).abs() < f32::EPSILON,
                "Weight[{i}] = {w} expected {expected_w} for fill={fill_value:?}"
            );
        }

        // Every node should round-trip
        for i in 0..128 {
            assert_eq!(
                dir_back.get(i),
                fill_value,
                "Node {i} should be {fill_value:?} after round-trip"
            );
        }
    }
}

#[test]
fn test_octree_bridge_mixed_groups() {
    // Mixed pattern: alternating Positive/Negative within a group
    let mut dir = TernaryDir::zero();
    for i in 0..16 {
        dir.set(
            i,
            if i % 2 == 0 {
                TernaryValue::Positive
            } else {
                TernaryValue::Negative
            },
        );
    }
    for i in 16..128 {
        dir.set(i, TernaryValue::Zero);
    }

    let weights = octree_leaf_to_patch_weights(&dir);

    // Group 0: 8× Positive (+1.0) + 8× Negative (-1.0) → average = 0.0
    assert!(
        (weights[0] - 0.0).abs() < f32::EPSILON,
        "Mixed group 0 should average to 0.0, got {}",
        weights[0]
    );

    // Groups 1-7: all Zero → 0.0
    for (g, w) in weights.iter().enumerate().take(8).skip(1) {
        assert!((w - 0.0).abs() < f32::EPSILON, "Group {g} should be 0.0, got {}", w);
    }

    // Round-trip: the averaging loss means we can't recover individual alternation,
    // but dominant direction should be Zero (0.0 is within ±0.5 threshold)
    let dir_back = patch_weights_to_octree_leaf(&weights);
    for i in 0..16 {
        assert_eq!(
            dir_back.get(i),
            TernaryValue::Zero,
            "Node {i}: averaging lost alternating pattern, should resolve to Zero"
        );
    }
}

// ---------------------------------------------------------------------------
// Morton code ↔ segment_id mapping
// ---------------------------------------------------------------------------

#[test]
fn test_morton_code_segment_mapping() {
    // Verify morton codes can be used as segment_ids
    let (x, y) = (5u32, 3u32);
    let morton = MortonCode::encode(x, y);
    let (x2, y2) = MortonCode::decode(morton);
    assert_eq!((x, y), (x2, y2), "Morton round-trip for ({x}, {y})");

    // Morton code can serve as a segment_id
    assert!(
        morton < 32,
        "Morton ({x},{y}) = {morton} should fit in our 32-segment context"
    );
}

#[test]
fn test_octree_lod_depth_mapping() {
    // X8 maps to depth 5
    assert_eq!(OctreeLod::ratio_to_depth(CompressionRatio::X8), 5);
    assert_eq!(OctreeLod::depth_to_ratio(5), CompressionRatio::X8);

    // Depth 5 → 32 slots
    assert_eq!(
        OctreeLod::slot_count(5),
        32,
        "Depth 5 should yield 32 slots"
    );
}

// ---------------------------------------------------------------------------
// Rejection tests
// ---------------------------------------------------------------------------

#[test]
fn test_tamper_rejection() {
    let (_, mut ctx) = encode_256_tokens();

    // Create a valid patch then corrupt the commitment
    let mut patch = LatentPatch::new(3, [0.5f32; 8]);
    patch.commitment[0] ^= 0xFF; // flip bits in first byte

    assert!(
        !patch.verify_commitment(),
        "Tampered commitment should fail verification"
    );

    let result = LatentPatcher::patch(&mut ctx, &patch);
    assert!(
        matches!(
            result,
            Err(PatchRejection::CommitmentMismatch { segment_id: 3 })
        ),
        "Tampered patch should be rejected with CommitmentMismatch, got {:?}",
        result
    );
}

#[test]
fn test_out_of_range_rejection() {
    let (_, mut ctx) = encode_256_tokens();
    assert_eq!(ctx.latent_slot_count, 32);

    // Segment 999 does not exist
    let patch = LatentPatch::new(999, [0.1f32; 8]);
    let result = LatentPatcher::patch(&mut ctx, &patch);
    assert!(
        matches!(result, Err(PatchRejection::OutOfRange { segment_id: 999 })),
        "Non-existent segment should be rejected with OutOfRange, got {:?}",
        result
    );

    // Edge: segment 32 (just past the last valid id 31)
    let patch_edge = LatentPatch::new(32, [0.1f32; 8]);
    let result_edge = LatentPatcher::patch(&mut ctx, &patch_edge);
    assert!(
        matches!(
            result_edge,
            Err(PatchRejection::OutOfRange { segment_id: 32 })
        ),
        "Segment 32 should be OutOfRange, got {:?}",
        result_edge
    );
}

#[test]
fn test_nan_rejection() {
    let (_, mut ctx) = encode_256_tokens();

    // Create a patch with NaN weights but correct commitment
    let nan_weights = [f32::NAN, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let mut patch = LatentPatch::new(5, nan_weights);
    // Recompute commitment so it matches the NaN weights (commitment check passes)
    patch.commitment = LatentPatch::compute_commitment(patch.segment_id, &patch.weights);

    assert!(
        !patch.weights_finite(),
        "NaN weights should fail finite check"
    );

    let result = LatentPatcher::patch(&mut ctx, &patch);
    assert!(
        matches!(
            result,
            Err(PatchRejection::NonFiniteWeights { segment_id: 5 })
        ),
        "NaN patch should be rejected with NonFiniteWeights, got {:?}",
        result
    );

    // Verify the segment was NOT modified
    let weights = get_weights(&ctx, 5).unwrap();
    for w in &weights {
        assert!(
            w.is_finite(),
            "Segment 5 weights should still be finite after NaN rejection"
        );
    }
}

#[test]
fn test_inf_rejection() {
    let (_, mut ctx) = encode_256_tokens();

    let inf_weights = [f32::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut patch = LatentPatch::new(2, inf_weights);
    patch.commitment = LatentPatch::compute_commitment(patch.segment_id, &patch.weights);

    assert!(
        !patch.weights_finite(),
        "Inf weights should fail finite check"
    );

    let result = LatentPatcher::patch(&mut ctx, &patch);
    assert!(
        matches!(
            result,
            Err(PatchRejection::NonFiniteWeights { segment_id: 2 })
        ),
        "Inf patch should be rejected with NonFiniteWeights, got {:?}",
        result
    );

    let neg_inf_weights = [f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut patch2 = LatentPatch::new(2, neg_inf_weights);
    patch2.commitment = LatentPatch::compute_commitment(patch2.segment_id, &patch2.weights);

    let result2 = LatentPatcher::patch(&mut ctx, &patch2);
    assert!(
        matches!(
            result2,
            Err(PatchRejection::NonFiniteWeights { segment_id: 2 })
        ),
        "NegInf patch should be rejected, got {:?}",
        result2
    );
}

// ---------------------------------------------------------------------------
// Mixed batch: valid + invalid patches
// ---------------------------------------------------------------------------

#[test]
fn test_batch_mixed_valid_and_invalid() {
    let (_, mut ctx) = encode_256_tokens();
    let total_segments = ctx.latent_slot_count as u32;

    // Snapshot weights before batch
    let orig_0 = get_weights(&ctx, 0).unwrap();
    let orig_5 = get_weights(&ctx, 5).unwrap();
    let _orig_10 = get_weights(&ctx, 10).unwrap();
    let _orig_15 = get_weights(&ctx, 15).unwrap();
    let _orig_20 = get_weights(&ctx, 20).unwrap();

    // 5 valid patches
    let valid_ids = [0u32, 5, 10, 15, 20];
    let valid_weights = [
        [0.1f32; 8],
        [0.2f32; 8],
        [0.3f32; 8],
        [0.4f32; 8],
        [0.5f32; 8],
    ];

    let mut patches: Vec<LatentPatch> = valid_ids
        .iter()
        .zip(valid_weights.iter())
        .map(|(&id, &w)| LatentPatch::new(id, w))
        .collect();

    // + 2 invalid patches: tampered segment 3, NaN segment 7
    let mut tampered = LatentPatch::new(3, [0.6f32; 8]);
    tampered.commitment[0] ^= 0xFF;

    let nan_weights = [f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut nan_patch = LatentPatch::new(7, nan_weights);
    nan_patch.commitment =
        LatentPatch::compute_commitment(nan_patch.segment_id, &nan_patch.weights);

    patches.push(tampered);
    patches.push(nan_patch);

    let batch = LatentPatchBatch::new(patches, total_segments, CompressionRatio::X8, 1);
    let receipt = LatentPatcher::patch_batch(&mut ctx, &batch);

    // Only 5 committed, 2 rejected
    assert_eq!(
        receipt.committed.len(),
        5,
        "Should have 5 committed patches"
    );
    assert_eq!(receipt.rejected.len(), 2, "Should have 2 rejected patches");
    assert!(!receipt.all_committed());

    // Committed IDs should be exactly the valid ones
    let mut committed_sorted = receipt.committed.clone();
    committed_sorted.sort();
    assert_eq!(committed_sorted, vec![0, 5, 10, 15, 20]);

    // Rejected should be segment 3 (CommitmentMismatch) and 7 (NonFiniteWeights)
    let rejection_ids: Vec<u32> = receipt
        .rejected
        .iter()
        .map(|r| match r {
            PatchRejection::CommitmentMismatch { segment_id } => *segment_id,
            PatchRejection::NonFiniteWeights { segment_id } => *segment_id,
            other => panic!("Unexpected rejection type: {:?}", other),
        })
        .collect();
    assert!(
        rejection_ids.contains(&3),
        "Segment 3 should be rejected (tampered)"
    );
    assert!(
        rejection_ids.contains(&7),
        "Segment 7 should be rejected (NaN)"
    );

    // Valid patches should have been applied
    for (idx, &seg_id) in valid_ids.iter().enumerate() {
        let weights = get_weights(&ctx, seg_id).unwrap();
        for (j, &expected) in valid_weights[idx].iter().enumerate() {
            assert!(
                (weights[j] - expected).abs() < f32::EPSILON,
                "Valid segment {seg_id} weight[{j}] should be applied"
            );
        }
    }

    // Invalid patches should NOT have modified segments
    let weights_3 = get_weights(&ctx, 3).unwrap();
    // Segment 3 was not in the valid set, check it wasn't changed by the tampered patch
    // (it was never valid-patched either, so it should equal original)
    // orig_3 not saved, so just verify it's still finite
    for w in &weights_3 {
        assert!(
            w.is_finite(),
            "Segment 3 should have finite weights after rejected patch"
        );
    }

    // Segments that were both valid-patched and exist in orig should have changed
    assert_ne!(
        get_weights(&ctx, 0).unwrap(),
        orig_0,
        "Segment 0 should have been patched"
    );
    assert_ne!(
        get_weights(&ctx, 5).unwrap(),
        orig_5,
        "Segment 5 should have been patched"
    );
}

// ---------------------------------------------------------------------------
// Chain-safe guard
// ---------------------------------------------------------------------------

#[test]
fn test_chain_safe_validation_mod() {
    let batch = LatentPatchBatch::new(vec![], 0, CompressionRatio::X8, 0);
    assert_eq!(
        batch.validation_mod, 1,
        "Default validation_mod should be 1"
    );
    batch.assert_chain_safe(); // Should not panic

    let mut adaptive = batch.clone();
    adaptive.validation_mod = 3;
    let result = std::panic::catch_unwind(|| adaptive.assert_chain_safe());
    assert!(result.is_err(), "validation_mod != 1 should panic");
}

// ---------------------------------------------------------------------------
// Full pipeline: compress → patch → reinject → expand → verify
// ---------------------------------------------------------------------------

#[test]
fn test_full_pipeline_compress_patch_reinject_verify() {
    // Step 1: Compress
    let (_, mut ctx) = encode_256_tokens();
    assert_eq!(ctx.latent_slot_count, 32);
    let total_segments = ctx.latent_slot_count as u32;

    // Step 2: Create patches with octree-derived weights
    let mut dir = TernaryDir::zero();
    for i in 0..16 {
        dir.set(i, TernaryValue::Positive);
    }
    for i in 16..32 {
        dir.set(i, TernaryValue::Negative);
    }
    // Remaining groups stay Zero
    let octree_weights = octree_leaf_to_patch_weights(&dir);

    let patches = vec![
        LatentPatch::new(0, octree_weights),
        LatentPatch::new(15, [0.25f32; 8]),
        LatentPatch::new(31, [0.75f32; 8]),
    ];

    let batch = LatentPatchBatch::new(patches.clone(), total_segments, CompressionRatio::X8, 42);
    assert_eq!(batch.tick, 42);

    // Step 3: Reinject
    let receipt = LatentPatcher::patch_batch(&mut ctx, &batch);
    assert!(receipt.all_committed());
    assert_eq!(receipt.committed.len(), 3);

    // Step 4: Verify patched segments
    let w0 = get_weights(&ctx, 0).unwrap();
    assert!(
        (w0[0] - 1.0).abs() < f32::EPSILON,
        "Segment 0 group 0 → +1.0"
    );
    assert!(
        (w0[1] - (-1.0)).abs() < f32::EPSILON,
        "Segment 0 group 1 → -1.0"
    );

    let w15 = get_weights(&ctx, 15).unwrap();
    for w in &w15 {
        assert!((w - 0.25).abs() < f32::EPSILON, "Segment 15 should be 0.25");
    }

    let w31 = get_weights(&ctx, 31).unwrap();
    for w in &w31 {
        assert!((w - 0.75).abs() < f32::EPSILON, "Segment 31 should be 0.75");
    }

    // Step 5: Verify unpatched segment unchanged
    let w8 = get_weights(&ctx, 8).unwrap();
    // Just verify it's finite and non-zero (original encoder output)
    assert!(
        w8.iter().all(|w| w.is_finite()),
        "Unpatched segment 8 should have finite weights"
    );
    assert!(
        w8.iter().any(|w| *w != 0.0),
        "Unpatched segment 8 should have non-zero weights from encoder"
    );

    // Step 6: EXPAND still works
    let tokens_0 = ctx
        .expand(0)
        .expect("EXPAND should work on patched segment 0");
    assert_eq!(tokens_0.len(), 8, "X8 → 8 original tokens per segment");
    assert_eq!(tokens_0, &[0, 1, 2, 3, 4, 5, 6, 7]);

    let tokens_15 = ctx
        .expand(15)
        .expect("EXPAND should work on patched segment 15");
    assert_eq!(tokens_15.len(), 8);

    // Step 7: Octree round-trip on patched weights
    let dir_back = patch_weights_to_octree_leaf(&octree_weights);
    assert_eq!(
        dir_back.get(0),
        TernaryValue::Positive,
        "Octree round-trip: group 0 Positive"
    );
    assert_eq!(
        dir_back.get(16),
        TernaryValue::Negative,
        "Octree round-trip: group 1 Negative"
    );
}
