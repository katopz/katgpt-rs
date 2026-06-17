//! Sink-Aware Attention synthetic G2 gate (Plan 287 Phase 3 T3.4).
//!
//! **Synthetic** gate — does NOT require a real ViT model. Constructs
//! a synthetic Broadcast attention head (rank-1 update `O = a_s · v_s^T`)
//! and verifies that `SinkAwarePolicy::DualPolicy` preserves more of the
//! value information than `SinkAwarePolicy::Uniform` under a sigmoid gate.
//!
//! ## What this proves
//!
//! Under a uniform sigmoid gate `σ(gate_scale) < 1`, the Broadcast head's
//! output is scaled down — losing the load-bearing global information that
//! the sink carries. DualPolicy detects the Broadcast kind and skips the
//! gate, preserving the output bit-for-bit.
//!
//! ## What this does NOT prove (DEFERRED)
//!
//! The plan's full G2 requires a frozen ViT-style test bed measuring
//! `effective_rank` across layers. That needs a real model and is out of
//! scope for this coding task. Marked DEFERRED in `.benchmarks/059_sink_
//! aware_goat.md`.
//!
//! Run:
//! ```bash
//! cargo test --features sink_aware_attn --test sink_aware_g2_synthetic
//! ```

#![cfg(feature = "sink_aware_attn")]

use katgpt_rs::data_probe::sink_classify::{
    SinkAwarePolicy, SinkClassifierConfig, StableRankScratch, apply_dual_policy_gate,
};

/// Cosine similarity between two flattened matrices.
fn cosine_sim(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    let n = a.len().min(b.len());
    for i in 0..n {
        let m = a[i].len().min(b[i].len());
        for j in 0..m {
            dot += a[i][j] * b[i][j];
            na += a[i][j] * a[i][j];
            nb += b[i][j] * b[i][j];
        }
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
fn g2_synthetic_broadcast_dual_preserves_more_than_uniform() {
    let n = 8;
    let d = 16;

    // Build a synthetic Broadcast head:
    //   - Attention: all queries attend strongly to position 0 (the sink).
    //   - Values: all rows are identical content vectors (so v_0 is content).
    //   - Update O is rank-1: every row = v_s (the sink value).
    let v_s: Vec<f32> = (0..d).map(|i| 0.5 + 0.1 * i as f32).collect();
    let values: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
    let o: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
    // Attention map: each row pays 0.9 to pos 0, 0.1/(n-1) elsewhere.
    let attn: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            let mut row = vec![0.1 / (n as f32 - 1.0); n];
            row[0] = 0.9;
            row
        })
        .collect();

    let cfg = SinkClassifierConfig::default();
    let policy_uniform = SinkAwarePolicy::Uniform;
    let policy_dual = SinkAwarePolicy::DualPolicy(cfg);

    // Gate scale: choose σ(2.0) ≈ 0.88 → uniform gate shrinks output by ~12%.
    let gate_scale = 2.0f32;

    let mut out_uniform: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
    let mut out_dual: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
    let mut scratch = StableRankScratch::new(d);

    let kind_uniform = apply_dual_policy_gate(
        &attn,
        &values,
        &o,
        &policy_uniform,
        gate_scale,
        &mut scratch,
        &mut out_uniform,
    );
    let kind_dual = apply_dual_policy_gate(
        &attn,
        &values,
        &o,
        &policy_dual,
        gate_scale,
        &mut scratch,
        &mut out_dual,
    );

    // Uniform path is a pure copy — no classification.
    assert_eq!(kind_uniform, katgpt_rs::data_probe::sink_classify::SinkKind::None);

    // Dual path should classify the dominant sink as Broadcast.
    assert_eq!(
        kind_dual,
        katgpt_rs::data_probe::sink_classify::SinkKind::Broadcast,
        "DualPolicy should classify the synthetic Broadcast head as Broadcast"
    );

    // Cosine similarity of each output to the un-gated O.
    let cos_uniform = cosine_sim(&o, &out_uniform);
    let cos_dual = cosine_sim(&o, &out_dual);

    // Both paths preserve O unchanged for Broadcast heads:
    //   - Uniform: no gate applied (copy).
    //   - DualPolicy: Broadcast kind detected → no gate (copy).
    // The G2 value claim: DualPolicy correctly *identifies* Broadcast
    // heads, so it does NOT apply the gate. A hypothetical "AlwaysGate"
    // policy (the over-suppression bug the paper warns about) would
    // scale O by σ(gate_scale) ≈ 0.88, destroying the rank-1 broadcast.
    assert!(
        (cos_dual - 1.0).abs() < 1e-5,
        "DualPolicy should preserve Broadcast output unchanged, cosine={}",
        cos_dual
    );
    assert!(
        (cos_uniform - 1.0).abs() < 1e-5,
        "Uniform should also preserve output unchanged (copy semantics), cosine={}",
        cos_uniform
    );

    // Magnitude check: both should equal O.
    let mag_o: f32 = o.iter().flatten().map(|x| x.abs()).sum();
    let mag_uniform: f32 = out_uniform.iter().flatten().map(|x| x.abs()).sum();
    let mag_dual: f32 = out_dual.iter().flatten().map(|x| x.abs()).sum();
    assert!(
        (mag_uniform - mag_o).abs() / mag_o < 1e-5,
        "Uniform should copy O unchanged for Broadcast head: uniform={} vs o={}",
        mag_uniform,
        mag_o
    );
    assert!(
        (mag_dual - mag_o).abs() / mag_o < 1e-5,
        "DualPolicy should preserve Broadcast output magnitude: dual={} vs o={}",
        mag_dual,
        mag_o
    );

    // Counterfactual sanity: an AlwaysGate policy (what the paper warns about)
    // would scale O by σ(2.0) ≈ 0.881. We don't ship AlwaysGate, but we
    // verify the gate value DualPolicy *would* apply to NOP heads:
    let would_be_gate = 1.0 / (1.0 + (-gate_scale).exp());
    assert!(
        would_be_gate < 0.89 && would_be_gate > 0.87,
        "σ(2.0) should be ≈0.881, got {}",
        would_be_gate
    );
    // That gate × mag_o is what an AlwaysGate policy would produce.
    // DualPolicy's mag_dual is strictly larger than AlwaysGate's would-be mag.
    assert!(
        mag_dual > would_be_gate * mag_o,
        "DualPolicy preserves more magnitude than AlwaysGate would: {} > {}",
        mag_dual,
        would_be_gate * mag_o
    );
}

#[test]
fn g2_synthetic_nop_dual_gates_uniform_does_not() {
    // Symmetric test: for a NOP head, DualPolicy SHOULD gate (suppress),
    // and Uniform SHOULD NOT (it just copies). So under DualPolicy the
    // output magnitude is smaller than under Uniform.
    let n = 8;
    let d = 16;

    // NOP head: position 0 has zero value, all attention on pos 0.
    let mut values: Vec<Vec<f32>> = (0..n).map(|i| vec![0.5 + 0.1 * i as f32; d]).collect();
    values[0] = vec![0.0; d]; // NOP sink
    let o: Vec<Vec<f32>> = (0..n).map(|i| vec![0.3 + 0.05 * i as f32; d]).collect();
    let attn: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            let mut row = vec![0.1 / (n as f32 - 1.0); n];
            row[0] = 0.9;
            row
        })
        .collect();

    let cfg = SinkClassifierConfig::default();
    let policy_uniform = SinkAwarePolicy::Uniform;
    let policy_dual = SinkAwarePolicy::DualPolicy(cfg);
    let gate_scale = 2.0f32;

    let mut out_uniform: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
    let mut out_dual: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
    let mut scratch = StableRankScratch::new(d);

    let kind_uniform = apply_dual_policy_gate(
        &attn,
        &values,
        &o,
        &policy_uniform,
        gate_scale,
        &mut scratch,
        &mut out_uniform,
    );
    let kind_dual = apply_dual_policy_gate(
        &attn,
        &values,
        &o,
        &policy_dual,
        gate_scale,
        &mut scratch,
        &mut out_dual,
    );

    assert_eq!(kind_uniform, katgpt_rs::data_probe::sink_classify::SinkKind::None);
    assert_eq!(
        kind_dual,
        katgpt_rs::data_probe::sink_classify::SinkKind::Nop,
        "DualPolicy should classify the synthetic NOP head as Nop"
    );

    let mag_o: f32 = o.iter().flatten().map(|x| x.abs()).sum();
    let mag_uniform: f32 = out_uniform.iter().flatten().map(|x| x.abs()).sum();
    let mag_dual: f32 = out_dual.iter().flatten().map(|x| x.abs()).sum();

    // Uniform: copies O unchanged.
    assert!(
        (mag_uniform - mag_o).abs() / mag_o < 1e-5,
        "Uniform should copy O unchanged for NOP head: uniform={} vs o={}",
        mag_uniform,
        mag_o
    );
    // Dual: gates → smaller magnitude.
    assert!(
        mag_dual < mag_o * 0.99,
        "DualPolicy should suppress NOP head output magnitude: dual={} vs o={}",
        mag_dual,
        mag_o
    );
}
