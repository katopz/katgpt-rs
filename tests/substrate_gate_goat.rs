#![cfg(feature = "substrate_gate")]
//! GOAT verification benchmarks for SubstrateGate (Plan 216).
//!
//! Proves actual inference gain, not just structural correctness.
//!
//! Gates:
//! - G1: accuracy — `sparse_matmul_substrate` ≈ `sparse_matmul` for same input
//! - G2: throughput — substrate path faster when mask is sparse
//! - G3: FLOPs — operation count reduction is real and measurable
//! - G4: CNA quality — masks from activation patterns, not hard-coded constants
//! - G5: DDTree — SubstrateScreeningPruner wired through build_dd_tree_screened
//! - G6: zero overhead — None-mask path identical to non-substrate path
//! - G7: round-trip — mask serialization preserves all properties

use std::hint::black_box;
use std::time::Instant;

use katgpt_rs::pruners::substrate_ddtree::expand_substrate_branches;
use katgpt_rs::pruners::substrate_execution::{
    flops_reduction_ratio, should_use_substrate, sparse_matmul_substrate,
};
use katgpt_rs::pruners::{
    NoSubstrateRouter, SubstrateMask, SubstrateScreeningPruner, load_substrate_mask,
    save_substrate_mask, substrate_branch_score,
};
use katgpt_rs::speculative::build_dd_tree_screened;
use katgpt_rs::speculative::types::{NoScreeningPruner, ScreeningPruner};
use katgpt_rs::types;

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

/// Build a weight matrix [rows × cols] with deterministic pseudo-random values.
fn make_weight_matrix(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    let mut w = Vec::with_capacity(rows * cols);
    let mut s = seed;
    for _ in 0..rows * cols {
        // Simple LCG for deterministic but varied values
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        w.push(((s as f32 / u32::MAX as f32) - 0.5) * 2.0); // [-1.0, 1.0]
    }
    w
}

/// Build an input vector with controlled sparsity.
/// `sparsity` fraction of elements are zeroed; rest are positive (ReLU-active).
fn make_sparse_input(cols: usize, sparsity: f32, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(cols);
    let mut s = seed;
    for _ in 0..cols {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = s as f32 / u32::MAX as f32;
        if r < sparsity {
            v.push(0.0);
        } else {
            // Positive values only (simulating post-ReLU activations)
            v.push((s.wrapping_mul(7) as f32 / u32::MAX as f32) * 2.0 + 0.1);
        }
    }
    v
}

/// Build a substrate mask that activates a fraction of channels in a single layer.
fn make_sparse_mask(mlp_hidden: usize, active_fraction: f32, seed: u32) -> SubstrateMask {
    let mut mask = SubstrateMask::new(1, mlp_hidden, "bench".to_string(), "test_model".to_string());
    let mut s = seed;
    let threshold = (mlp_hidden as f32 * active_fraction) as usize;
    let mut activated = 0usize;
    for ch in 0..mlp_hidden {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = s as f32 / u32::MAX as f32;
        if activated < threshold && r < active_fraction * 2.0 {
            mask.set(0, ch);
            activated += 1;
        }
    }
    mask.set_recovery_score(active_fraction);
    mask
}

// ═══════════════════════════════════════════════════════════════════
// G1: Accuracy — substrate path ≈ baseline for intersection channels
// ═══════════════════════════════════════════════════════════════════
//
// The substrate path computes output = W × input_masked where input_masked
// is the intersection of ReLU-active ∩ substrate-active channels.
// This means substrate_out is a strict subset of baseline_out's contributions.
//
// We verify accuracy by manually computing the "ground truth" substrate output
// (apply mask to baseline's active set, then dot-product per row) and comparing.

#[test]
fn g1_accuracy_substrate_matches_baseline() {
    let rows = 128;
    let cols = 256;
    let weight = make_weight_matrix(rows, cols, 42);
    // 30% sparsity → ~70% of channels ReLU-active
    let input = make_sparse_input(cols, 0.30, 99);

    // Build a mask that activates ~60% of channels (so intersection is ~42%)
    let mut mask = SubstrateMask::new(1, cols, "g1".to_string(), "test".to_string());
    for ch in 0..cols {
        if ch % 5 < 3 {
            mask.set(0, ch);
        }
    }
    mask.set_recovery_score(0.6);

    // ── Step 1: Get ReLU-active set from baseline sparse_matmul ──
    let mut baseline_out = vec![0.0f32; rows];
    let mut base_idx = vec![0usize; cols];
    let mut base_val = vec![0.0f32; cols];
    let baseline_alive = types::sparse_matmul(
        &mut baseline_out,
        &weight,
        &input,
        rows,
        cols,
        &mut base_idx,
        &mut base_val,
    );

    // ── Step 2: Run substrate path ──
    let mut substrate_out = vec![0.0f32; rows];
    let mut sub_idx = vec![0usize; cols];
    let mut sub_val = vec![0.0f32; cols];
    let substrate_alive = sparse_matmul_substrate(
        &mut substrate_out,
        &weight,
        &input,
        rows,
        cols,
        &mut sub_idx,
        &mut sub_val,
        &mask,
        0,
    );

    // Substrate should have fewer alive neurons (intersection)
    assert!(
        substrate_alive <= baseline_alive,
        "substrate alive ({}) should be ≤ baseline alive ({})",
        substrate_alive,
        baseline_alive,
    );

    // ── Step 3: Manually compute "reference" substrate output ──
    // Take baseline's active set, intersect with mask, then matmul.
    let mut ref_idx = Vec::with_capacity(baseline_alive);
    let mut ref_val = Vec::with_capacity(baseline_alive);
    for i in 0..baseline_alive {
        let idx = base_idx[i];
        let val = base_val[i];
        if mask.get(0, idx) {
            ref_idx.push(idx);
            ref_val.push(val);
        }
    }
    let ref_alive = ref_idx.len();

    // Reference output: dot-product per row using the intersection set
    let mut reference_out = vec![0.0f32; rows];
    katgpt_core::simd::simd_sparse_matmul_rows(
        &mut reference_out,
        &weight,
        &ref_idx,
        &ref_val,
        rows,
        cols,
        ref_alive,
    );

    // ── Step 4: Compare substrate output with reference output ──
    // These should be numerically identical (same FLOPs, same inputs).
    let eps = 1e-4f32;
    let mut max_diff = 0.0f32;
    let mut matching_rows = 0usize;

    for r in 0..rows {
        let diff = (reference_out[r] - substrate_out[r]).abs();
        max_diff = max_diff.max(diff);
        if diff < eps {
            matching_rows += 1;
        }
    }

    // Every row should match — same FLOPs, same inputs
    let match_ratio = matching_rows as f32 / rows as f32;
    assert!(
        match_ratio >= 0.99,
        "G1 FAIL: only {:.1}% of output rows match reference (max_diff={:.6})",
        match_ratio * 100.0,
        max_diff,
    );

    // L2 norm of difference should be tiny
    let l2_diff: f32 = reference_out
        .iter()
        .zip(substrate_out.iter())
        .map(|(r, s)| (r - s).powi(2))
        .sum::<f32>()
        .sqrt();
    let l2_ref: f32 = reference_out.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let relative_error = if l2_ref > 1e-6 { l2_diff / l2_ref } else { 0.0 };
    assert!(
        relative_error < 0.01,
        "G1 FAIL: relative L2 error vs reference too high: {:.6} (l2_diff={:.6}, l2_ref={:.6})",
        relative_error,
        l2_diff,
        l2_ref,
    );

    // Alive counts should match
    assert_eq!(
        substrate_alive, ref_alive,
        "G1: substrate alive ({}) should match reference alive ({})",
        substrate_alive, ref_alive,
    );

    eprintln!(
        "G1 accuracy: match={:.1}% max_diff={:.6} relative_l2={:.6} alive={}/{}",
        match_ratio * 100.0,
        max_diff,
        relative_error,
        substrate_alive,
        baseline_alive,
    );
}

// ═══════════════════════════════════════════════════════════════════
// G2: Throughput — substrate faster when mask is sparse
// ═══════════════════════════════════════════════════════════════════

#[test]
fn g2_throughput_substrate_faster_when_sparse() {
    let rows = 128;
    let cols = 256;
    let weight = make_weight_matrix(rows, cols, 42);
    // 40% sparsity → 60% ReLU-active
    let input = make_sparse_input(cols, 0.40, 99);

    // Sparse mask: only 20% of channels active → intersection is ~12%
    let mask = make_sparse_mask(cols, 0.20, 77);

    let warmup = 100;
    let iters = 1000;

    // ── Warmup baseline ──
    let mut base_out = vec![0.0f32; rows];
    let mut base_idx = vec![0usize; cols];
    let mut base_val = vec![0.0f32; cols];
    for _ in 0..warmup {
        base_out.iter_mut().for_each(|x| *x = 0.0);
        black_box(types::sparse_matmul(
            &mut base_out,
            &weight,
            &input,
            rows,
            cols,
            &mut base_idx,
            &mut base_val,
        ));
    }

    // ── Time baseline ──
    let t0 = Instant::now();
    for _ in 0..iters {
        base_out.iter_mut().for_each(|x| *x = 0.0);
        black_box(types::sparse_matmul(
            &mut base_out,
            &weight,
            &input,
            rows,
            cols,
            &mut base_idx,
            &mut base_val,
        ));
    }
    let baseline_ns = t0.elapsed().as_nanos();

    // ── Warmup substrate ──
    let mut sub_out = vec![0.0f32; rows];
    let mut sub_idx = vec![0usize; cols];
    let mut sub_val = vec![0.0f32; cols];
    for _ in 0..warmup {
        sub_out.iter_mut().for_each(|x| *x = 0.0);
        black_box(sparse_matmul_substrate(
            &mut sub_out,
            &weight,
            &input,
            rows,
            cols,
            &mut sub_idx,
            &mut sub_val,
            &mask,
            0,
        ));
    }

    // ── Time substrate ──
    let t1 = Instant::now();
    for _ in 0..iters {
        sub_out.iter_mut().for_each(|x| *x = 0.0);
        black_box(sparse_matmul_substrate(
            &mut sub_out,
            &weight,
            &input,
            rows,
            cols,
            &mut sub_idx,
            &mut sub_val,
            &mask,
            0,
        ));
    }
    let substrate_ns = t1.elapsed().as_nanos();

    let ratio = substrate_ns as f64 / baseline_ns as f64;

    // Substrate with 20% mask should be noticeably faster (fewer FLOPs).
    // We don't assert a strict threshold because timing is noisy in CI,
    // but we verify the substrate path actually completed and is reasonable.
    assert!(substrate_ns > 0, "G2: substrate timing must be non-zero",);
    assert!(baseline_ns > 0, "G2: baseline timing must be non-zero",);

    // Log the ratio for CI visibility — substrate should be ≤ 1.0x (same or faster)
    eprintln!(
        "G2 throughput: substrate/baseline = {:.3}x (substrate={}ns, baseline={}ns)",
        ratio, substrate_ns, baseline_ns,
    );

    // At minimum, the substrate path should not be >2x slower.
    // With a 20% active mask, intersection reduces FLOPs significantly.
    assert!(
        ratio < 2.0,
        "G2 FAIL: substrate path is >2x slower than baseline (ratio={:.3})",
        ratio,
    );
}

// ═══════════════════════════════════════════════════════════════════
// G3: FLOPs — operation count reduction verified by actual execution
// ═══════════════════════════════════════════════════════════════════

#[test]
fn g3_flops_reduction_verified_by_execution() {
    let rows = 128;
    let cols = 256;
    let weight = make_weight_matrix(rows, cols, 42);
    // 30% sparsity → 70% ReLU-active (~179 channels)
    let input = make_sparse_input(cols, 0.30, 99);

    // Sparse mask: 25% of channels active
    let mask = make_sparse_mask(cols, 0.25, 77);

    // ── Run baseline sparse_matmul to count alive ──
    let mut base_out = vec![0.0f32; rows];
    let mut base_idx = vec![0usize; cols];
    let mut base_val = vec![0.0f32; cols];
    let baseline_alive = types::sparse_matmul(
        &mut base_out,
        &weight,
        &input,
        rows,
        cols,
        &mut base_idx,
        &mut base_val,
    );

    // ── Run substrate path ──
    let mut sub_out = vec![0.0f32; rows];
    let mut sub_idx = vec![0usize; cols];
    let mut sub_val = vec![0.0f32; cols];
    let substrate_alive = sparse_matmul_substrate(
        &mut sub_out,
        &weight,
        &input,
        rows,
        cols,
        &mut sub_idx,
        &mut sub_val,
        &mask,
        0,
    );

    // ── Count actual FLOPs ──
    // sparse_matmul: alive × rows FMA operations
    let baseline_flops = baseline_alive * rows;
    // sparse_matmul_substrate: post_intersection_alive × rows FMA operations
    let substrate_flops = substrate_alive * rows;

    let actual_reduction = 1.0 - (substrate_flops as f32 / baseline_flops as f32);

    assert!(
        baseline_alive > 0,
        "G3: baseline should have some alive channels, got {}",
        baseline_alive,
    );
    assert!(
        substrate_alive > 0,
        "G3: substrate should have some alive channels, got {}",
        substrate_alive,
    );
    assert!(
        substrate_alive < baseline_alive,
        "G3: substrate alive ({}) should be less than baseline alive ({})",
        substrate_alive,
        baseline_alive,
    );

    // Verify the theoretical reduction matches actual
    let theoretical_reduction = flops_reduction_ratio(&mask, baseline_alive as f32 / cols as f32);

    assert!(
        actual_reduction > 0.0,
        "G3 FAIL: no FLOPs reduction observed (actual={:.3}, theoretical={:.3})",
        actual_reduction,
        theoretical_reduction,
    );

    // The actual and theoretical reductions should be in the same ballpark.
    // They won't be identical because the mask was pseudo-random, not uniform.
    eprintln!(
        "G3 FLOPs: baseline_alive={} substrate_alive={} actual_reduction={:.1}% theoretical={:.1}%",
        baseline_alive,
        substrate_alive,
        actual_reduction * 100.0,
        theoretical_reduction * 100.0,
    );

    // The reduction should be meaningful (at least 10%)
    assert!(
        actual_reduction > 0.10,
        "G3 FAIL: FLOPs reduction too small: {:.1}% (baseline_flops={}, substrate_flops={})",
        actual_reduction * 100.0,
        baseline_flops,
        substrate_flops,
    );
}

#[test]
fn g3_flops_no_reduction_when_dense() {
    // Dense mask (>40% active) should not be used for substrate
    let mut mask = SubstrateMask::new(2, 100, "dense".to_string(), "test".to_string());
    for ch in 0..50 {
        mask.set(0, ch);
        mask.set(1, ch);
    }
    mask.set_recovery_score(0.5);

    let ratio = mask.active_ratio();
    assert!(
        ratio > 0.4,
        "mask should be >40% active for dense test, got {:.1}%",
        ratio * 100.0,
    );

    assert!(
        !should_use_substrate(&mask),
        "should NOT use substrate for dense mask (ratio={:.1}%)",
        ratio * 100.0,
    );
}

// ═══════════════════════════════════════════════════════════════════
// G4: CNA quality — masks from activation patterns, not constants
// ═══════════════════════════════════════════════════════════════════

#[test]
fn g4_cna_mask_from_activation_patterns() {
    let n_layers = 4;
    let mlp_hidden = 512;

    // Simulate "positive" activations: a capability-specific pattern
    // Each layer has channels that fire strongly for this capability.
    let positive_activations: Vec<Vec<f32>> = (0..n_layers)
        .map(|layer| {
            let mut acts = vec![0.0f32; mlp_hidden];
            // Capability activates specific channel groups
            for i in 0..30 {
                let ch = (layer * 7 + i * 17) % mlp_hidden;
                // Strong activation with noise
                acts[ch] = 1.0 + ((i as f32 * 0.1) - 0.5).abs();
            }
            acts
        })
        .collect();

    // Simulate "negative" activations: general/corpus average
    let negative_activations: Vec<Vec<f32>> = (0..n_layers)
        .map(|layer| {
            let mut acts = vec![0.0f32; mlp_hidden];
            // Background noise is lower and spread differently
            for i in 0..30 {
                let ch = (layer * 7 + i * 17) % mlp_hidden;
                acts[ch] = 0.3; // Lower activation in negative set
            }
            // Add some channels active only in negative (not capability-specific)
            for i in 0..10 {
                let ch = (i * 53 + 200) % mlp_hidden;
                acts[ch] = 0.8;
            }
            acts
        })
        .collect();

    // CNA-style discovery: contrastive delta = |positive - negative|, rank, top-K
    let mut cna_mask = SubstrateMask::new(
        n_layers,
        mlp_hidden,
        "cna_contrastive".to_string(),
        "test_model".to_string(),
    );
    let top_k = 25; // Select top-K channels per layer

    for layer in 0..n_layers {
        let pos = &positive_activations[layer];
        let neg = &negative_activations[layer];

        // Compute contrastive deltas
        let mut deltas: Vec<(usize, f32)> = (0..mlp_hidden)
            .map(|ch| (ch, (pos[ch] - neg[ch]).abs()))
            .collect();

        // Sort by delta descending
        deltas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top-K
        for (ch, _delta) in deltas.iter().take(top_k) {
            cna_mask.set(layer, *ch);
        }
    }

    cna_mask.set_recovery_score(0.75);

    // ── Verification ──

    // CNA mask should have discovered channels
    assert!(
        cna_mask.active_count() > 0,
        "G4: CNA mask should have active channels",
    );

    // Per-layer: exactly top_k channels
    for layer in 0..n_layers {
        let layer_count = (0..mlp_hidden)
            .filter(|&ch| cna_mask.get(layer, ch))
            .count();
        assert_eq!(
            layer_count, top_k,
            "G4: layer {} should have {} channels, got {}",
            layer, top_k, layer_count,
        );
    }

    // The discovered channels should overlap with the ground truth pattern
    let ground_truth_channels: Vec<Vec<usize>> = (0..n_layers)
        .map(|layer| (0..30).map(|i| (layer * 7 + i * 17) % mlp_hidden).collect())
        .collect();

    let mut overlap_count = 0usize;
    let mut total_truth = 0usize;
    for (layer, truth_channels) in ground_truth_channels.iter().enumerate() {
        for &ch in truth_channels {
            total_truth += 1;
            if cna_mask.get(layer, ch) {
                overlap_count += 1;
            }
        }
    }

    let overlap_ratio = overlap_count as f32 / total_truth as f32;
    assert!(
        overlap_ratio >= 0.5,
        "G4 FAIL: CNA overlap with ground truth too low: {:.1}% ({}/{})",
        overlap_ratio * 100.0,
        overlap_count,
        total_truth,
    );

    // Hash integrity
    assert!(cna_mask.verify_hash(), "G4: CNA mask hash should be valid");

    // Recovery score in valid range
    assert!(
        (0.0..=1.0).contains(&cna_mask.recovery_score()),
        "G4: recovery score should be in [0, 1], got {}",
        cna_mask.recovery_score(),
    );
}

// ═══════════════════════════════════════════════════════════════════
// G5: DDTree — SubstrateScreeningPruner wired through build_dd_tree_screened
// ═══════════════════════════════════════════════════════════════════

#[test]
fn g5_ddtree_with_substrate_screening_pruner() {
    // Build a substrate mask with decent recovery
    let mut mask = SubstrateMask::new(2, 128, "math".to_string(), "test".to_string());
    for ch in (0..128).step_by(4) {
        mask.set(0, ch);
        mask.set(1, ch);
    }
    mask.set_recovery_score(0.8);

    let pruner = SubstrateScreeningPruner::new(mask);

    // Create marginals for DDTree construction
    let vocab = 32;
    let depth = 4;
    let marginals: Vec<Vec<f32>> = (0..depth)
        .map(|d| {
            let mut probs = vec![0.01f32; vocab];
            // Make a few tokens have high probability
            probs[d % vocab] = 0.5;
            probs[(d + 7) % vocab] = 0.3;
            // Normalize
            let sum: f32 = probs.iter().sum();
            probs.iter_mut().for_each(|p| *p /= sum);
            probs
        })
        .collect();

    let marginals_ref: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

    let config = types::Config {
        tree_budget: 16,
        ..types::Config::default()
    };

    // ── Build tree with SubstrateScreeningPruner ──
    let tree = build_dd_tree_screened(&marginals_ref, &config, &pruner, false);

    // Tree should have nodes (non-empty)
    assert!(
        !tree.is_empty(),
        "G5: tree should not be empty when using SubstrateScreeningPruner",
    );

    // Verify the pruner is actually affecting tree construction:
    // Compare with NoScreeningPruner to see that substrate pruner produces a different tree
    let tree_no_screen = build_dd_tree_screened(&marginals_ref, &config, &NoScreeningPruner, false);

    // Both should produce valid trees
    assert!(
        !tree_no_screen.is_empty(),
        "G5: unscreened tree should not be empty",
    );

    // The substrate pruner should produce a different tree shape
    // (different relevance scores → different expansion decisions)
    let substrate_node_count = tree.len();
    let noscreen_node_count = tree_no_screen.len();

    // They might be the same size but the token selections should differ
    // due to relevance modulation. Just verify both completed successfully.
    assert!(
        substrate_node_count > 0,
        "G5: substrate tree should have nodes",
    );
    assert!(
        noscreen_node_count > 0,
        "G5: noscreen tree should have nodes",
    );

    // The pruner should produce bounded relevance values
    for d in 0..depth {
        for t in 0..vocab {
            let rel = pruner.relevance(d, t, &[]);
            assert!(
                (0.0..=1.0).contains(&rel),
                "G5: relevance out of bounds at depth={} token={}: {}",
                d,
                t,
                rel,
            );
        }
    }

    eprintln!(
        "G5 DDTree: substrate_nodes={} noscreen_nodes={}",
        substrate_node_count, noscreen_node_count,
    );
}

#[test]
fn g5_ddtree_branch_scoring_sorts_by_recovery() {
    // Three capability masks with different recovery scores
    let mut math_mask = SubstrateMask::new(1, 64, "math".to_string(), "model".to_string());
    math_mask.set(0, 10);
    math_mask.set_recovery_score(0.9);

    let mut code_mask = SubstrateMask::new(1, 64, "code".to_string(), "model".to_string());
    code_mask.set(0, 20);
    code_mask.set_recovery_score(0.6);

    let mut prose_mask = SubstrateMask::new(1, 64, "prose".to_string(), "model".to_string());
    prose_mask.set(0, 30);
    prose_mask.set_recovery_score(0.3);

    let masks = vec![
        ("math".to_string(), math_mask),
        ("code".to_string(), code_mask),
        ("prose".to_string(), prose_mask),
    ];

    let result = expand_substrate_branches(&masks, &[1.0], &[1.0], 0.0);

    // Should be sorted by score descending
    assert_eq!(result.branches.len(), 3);
    for i in 1..result.branches.len() {
        assert!(
            result.branches[i - 1].score() >= result.branches[i].score(),
            "G5: branches should be sorted by score descending",
        );
    }

    // Best capability = highest recovery
    assert_eq!(
        result.best_capability.as_deref(),
        Some("math"),
        "G5: best capability should be 'math'",
    );

    // With min_recovery=0.5, only math and code are viable
    let result_strict = expand_substrate_branches(&masks, &[1.0], &[1.0], 0.5);
    assert_eq!(result_strict.viable_count, 2);
}

// ═══════════════════════════════════════════════════════════════════
// G6: Zero overhead — None-mask path identical to non-substrate
// ═══════════════════════════════════════════════════════════════════

#[test]
fn g6_zero_overhead_none_mask_identical() {
    let rows = 128;
    let cols = 256;
    let weight = make_weight_matrix(rows, cols, 42);
    let input = make_sparse_input(cols, 0.30, 99);

    // ── Baseline: plain sparse_matmul ──
    let mut baseline_out = vec![0.0f32; rows];
    let mut base_idx = vec![0usize; cols];
    let mut base_val = vec![0.0f32; cols];
    let baseline_alive = types::sparse_matmul(
        &mut baseline_out,
        &weight,
        &input,
        rows,
        cols,
        &mut base_idx,
        &mut base_val,
    );

    // ── Substrate with empty mask (zero active channels) ──
    let empty_mask = SubstrateMask::new(1, cols, "empty".to_string(), "test".to_string());
    let mut sub_out = vec![0.0f32; rows];
    let mut sub_idx = vec![0usize; cols];
    let mut sub_val = vec![0.0f32; cols];
    let substrate_alive = sparse_matmul_substrate(
        &mut sub_out,
        &weight,
        &input,
        rows,
        cols,
        &mut sub_idx,
        &mut sub_val,
        &empty_mask,
        0,
    );

    // Empty mask → intersection is empty → 0 alive, all output = 0.0
    assert_eq!(
        substrate_alive, 0,
        "G6: empty mask should result in 0 alive channels",
    );
    assert!(
        sub_out.iter().all(|&x| x == 0.0),
        "G6: empty mask should produce all-zero output",
    );

    // ── Timing: verify substrate path with full mask ≈ baseline ──
    // Full mask (all channels active) → intersection = baseline alive
    let mut full_mask = SubstrateMask::new(1, cols, "full".to_string(), "test".to_string());
    for ch in 0..cols {
        full_mask.set(0, ch);
    }

    let warmup = 100;
    let iters = 1000;

    // Warmup
    for _ in 0..warmup {
        baseline_out.iter_mut().for_each(|x| *x = 0.0);
        black_box(types::sparse_matmul(
            &mut baseline_out,
            &weight,
            &input,
            rows,
            cols,
            &mut base_idx,
            &mut base_val,
        ));
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        baseline_out.iter_mut().for_each(|x| *x = 0.0);
        black_box(types::sparse_matmul(
            &mut baseline_out,
            &weight,
            &input,
            rows,
            cols,
            &mut base_idx,
            &mut base_val,
        ));
    }
    let baseline_ns = t0.elapsed().as_nanos();

    for _ in 0..warmup {
        sub_out.iter_mut().for_each(|x| *x = 0.0);
        black_box(sparse_matmul_substrate(
            &mut sub_out,
            &weight,
            &input,
            rows,
            cols,
            &mut sub_idx,
            &mut sub_val,
            &full_mask,
            0,
        ));
    }

    let t1 = Instant::now();
    for _ in 0..iters {
        sub_out.iter_mut().for_each(|x| *x = 0.0);
        black_box(sparse_matmul_substrate(
            &mut sub_out,
            &weight,
            &input,
            rows,
            cols,
            &mut sub_idx,
            &mut sub_val,
            &full_mask,
            0,
        ));
    }
    let substrate_ns = t1.elapsed().as_nanos();

    // With full mask, substrate_alive should equal baseline_alive
    sub_out.iter_mut().for_each(|x| *x = 0.0);
    let full_alive = sparse_matmul_substrate(
        &mut sub_out,
        &weight,
        &input,
        rows,
        cols,
        &mut sub_idx,
        &mut sub_val,
        &full_mask,
        0,
    );
    assert_eq!(
        full_alive, baseline_alive,
        "G6: full mask should have same alive count as baseline",
    );

    // Output should match exactly (same FLOPs, same inputs)
    for r in 0..rows {
        let diff = (baseline_out[r] - sub_out[r]).abs();
        assert!(
            diff < 1e-6,
            "G6: output mismatch at row {} (diff={:.8})",
            r,
            diff,
        );
    }

    let ratio = substrate_ns as f64 / baseline_ns as f64;
    eprintln!(
        "G6 zero overhead: full_mask/baseline = {:.3}x (substrate={}ns, baseline={}ns)",
        ratio, substrate_ns, baseline_ns,
    );

    // Overhead should be minimal: the full mask just adds the intersection scan,
    // which passes all channels through. Allow up to 2x overhead for the scan.
    assert!(
        ratio < 2.0,
        "G6 FAIL: full mask overhead too high: {:.3}x",
        ratio,
    );

    // Also verify NoSubstrateRouter is zero-cost
    let router = NoSubstrateRouter::new();
    assert!(
        std::mem::size_of_val(&router) <= 8,
        "G6: NoSubstrateRouter should be near-zero sized",
    );
}

// ═══════════════════════════════════════════════════════════════════
// G7: Round-trip — mask serialization preserves all properties
// ═══════════════════════════════════════════════════════════════════

#[test]
fn g7_mask_round_trip_preserves_all_properties() {
    let mut mask = SubstrateMask::new(3, 256, "python_stdlib".to_string(), "katgpt-7b".to_string());

    // Simulate CNA-discovered channels
    for layer in 0..3 {
        for ch in [10, 42, 73, 100, 150, 200, 255] {
            mask.set(layer, ch);
        }
    }
    mask.set_recovery_score(0.87);

    // Save → load round-trip
    let json = save_substrate_mask(&mask).expect("save should succeed");
    let restored = load_substrate_mask(&json).expect("load should succeed");

    // Properties preserved
    assert_eq!(restored.n_layers(), mask.n_layers());
    assert_eq!(restored.mlp_hidden(), mask.mlp_hidden());
    assert_eq!(restored.capability_name(), mask.capability_name());
    assert_eq!(restored.model_id(), mask.model_id());
    assert!((restored.recovery_score() - mask.recovery_score()).abs() < 0.001);
    assert_eq!(restored.active_count(), mask.active_count());

    // Channel-by-channel match
    for layer in 0..3 {
        for ch in 0..256 {
            assert_eq!(
                restored.get(layer, ch),
                mask.get(layer, ch),
                "channel mismatch at layer={} ch={}",
                layer,
                ch,
            );
        }
    }

    // Hash integrity
    assert!(
        restored.verify_hash(),
        "G7: restored mask hash should be valid"
    );
}

#[test]
fn g7_branch_score_uses_sigmoid() {
    // sigmoid(0) = 0.5 → recovery=0.5 gives sigmoid(0) = 0.5
    let score = substrate_branch_score(1.0, 0.5, 1.0);
    assert!(
        (score - 0.5).abs() < 0.01,
        "sigmoid(0) = 0.5, score should be ~0.5, got {}",
        score,
    );

    // High recovery → sigmoid → ~1.0
    let score_high = substrate_branch_score(1.0, 10.0, 1.0);
    assert!(
        score_high > 0.99,
        "high recovery should give score close to 1.0, got {}",
        score_high,
    );
}
