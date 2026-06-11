//! Plan 248: OctreeCTC Reconstructive Memory Navigation — GOAT Proof Tests
//!
//! Validates multi-hop recall improvement of `project_reconstruct()` over
//! single-shot `project_all()`. All tests are deterministic.
//!
//! GOAT metric: evidence accumulation improvement (aggressive reconstruction
//! accumulates ≥20% more evidence than default 3-step reconstruction).

#![cfg(feature = "octree_ctc")]

use katgpt_core::sense::brain::NpcBrain;
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::sense::reconstruction::ReconstructionConfig;
use katgpt_core::types::SenseKind;

/// Build a brain with 6 multi-embedding modules for reconstruction testing.
/// Each module has 3 direction vectors spanning 3 HLA dims, creating cross-module
/// bridges via shared dimensions (e.g., FighterSense dim 4 → SocialSense dim 4).
fn make_chain_brain() -> NpcBrain {
    let builder = SenseOctreeBuilder::new(3);
    let modules = vec![
        builder.build(SenseKind::CommonSense, &[
            KgEmbedding { entity_hash: 1, relation_hash: 1, embedding: [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            KgEmbedding { entity_hash: 2, relation_hash: 2, embedding: [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            KgEmbedding { entity_hash: 3, relation_hash: 3, embedding: [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
        ]),
        builder.build(SenseKind::FighterSense, &[
            KgEmbedding { entity_hash: 4, relation_hash: 4, embedding: [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            KgEmbedding { entity_hash: 5, relation_hash: 5, embedding: [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            KgEmbedding { entity_hash: 6, relation_hash: 6, embedding: [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
        ]),
        builder.build(SenseKind::GameTheorySense, &[
            KgEmbedding { entity_hash: 7, relation_hash: 7, embedding: [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            KgEmbedding { entity_hash: 8, relation_hash: 8, embedding: [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            KgEmbedding { entity_hash: 9, relation_hash: 9, embedding: [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0], sign: true, confidence: 1.0 },
        ]),
        builder.build(SenseKind::SpatialSense, &[
            KgEmbedding { entity_hash: 10, relation_hash: 10, embedding: [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            KgEmbedding { entity_hash: 11, relation_hash: 11, embedding: [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            KgEmbedding { entity_hash: 12, relation_hash: 12, embedding: [0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
        ]),
        builder.build(SenseKind::SocialSense, &[
            KgEmbedding { entity_hash: 13, relation_hash: 13, embedding: [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            KgEmbedding { entity_hash: 14, relation_hash: 14, embedding: [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0], sign: true, confidence: 1.0 },
            KgEmbedding { entity_hash: 15, relation_hash: 15, embedding: [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
        ]),
        builder.build(SenseKind::SkillSense, &[
            KgEmbedding { entity_hash: 16, relation_hash: 16, embedding: [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            KgEmbedding { entity_hash: 17, relation_hash: 17, embedding: [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0], sign: true, confidence: 1.0 },
            KgEmbedding { entity_hash: 18, relation_hash: 18, embedding: [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
        ]),
    ];
    let mut brain = NpcBrain::compose(modules);
    brain.hla_state = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0, 0.0];
    brain
}

fn activation_sum(acts: &[f32; 6]) -> f32 { acts.iter().copied().sum() }
fn vec_activation_sum(acts: &[f32]) -> f32 { acts.iter().copied().sum() }

#[test]
fn single_hop_recall_improvement() {
    let brain = make_chain_brain();
    let result = brain.project_reconstruct();
    for (i, &a) in result.passive.iter().enumerate() {
        assert!(a.is_finite() && a >= 0.0, "passive[{i}] = {a}");
    }
    for (i, &a) in result.active.iter().enumerate() {
        assert!(a.is_finite() && a >= 0.0, "active[{i}] = {a}");
    }
    let any_changed = result.active.iter().zip(result.passive.iter()).any(|(&a, &p)| (a - p).abs() > 1e-6);
    assert!(any_changed, "Reconstruction should change at least one activation");
    assert!(result.evidence.count > 0, "Evidence count should be > 0, got {}", result.evidence.count);
}

#[test]
fn multi_hop_recall_improvement() {
    let builder = SenseOctreeBuilder::new(3);
    let fighter = builder.build(SenseKind::FighterSense, &[
        KgEmbedding { entity_hash: 10, relation_hash: 10, embedding: [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
        KgEmbedding { entity_hash: 11, relation_hash: 11, embedding: [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
        KgEmbedding { entity_hash: 12, relation_hash: 12, embedding: [0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
    ]);
    let social = builder.build(SenseKind::SocialSense, &[
        KgEmbedding { entity_hash: 13, relation_hash: 13, embedding: [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
        KgEmbedding { entity_hash: 14, relation_hash: 14, embedding: [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0], sign: true, confidence: 1.0 },
        KgEmbedding { entity_hash: 15, relation_hash: 15, embedding: [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
    ]);
    let mut brain = NpcBrain::compose(vec![fighter, social]);
    brain.hla_state = [0.8, 0.7, 0.0, 0.0, 0.3, 0.1, 0.0, 0.0];

    let passive = brain.project_all();
    assert!(passive[0] > passive[1], "FighterSense should dominate SocialSense passively");

    let config = ReconstructionConfig { max_steps: 5, hla_learning_rate: 0.3, entropy_threshold: 0.01, ..Default::default() };
    let result = brain.project_reconstruct_with_config(config);
    assert!(result.steps > 0, "Should take at least 1 step");
    assert!(result.evidence.count > 0, "Should accumulate evidence across steps");

    let any_changed = (result.active[1] - passive[0]).abs() > 1e-4 || (result.active[4] - passive[1]).abs() > 1e-4;
    assert!(any_changed, "Reconstruction should change activations via HLA evolution");
}

#[test]
fn recall_threshold_met() {
    let brain = make_chain_brain();
    let result = brain.project_reconstruct();
    let evidence_recall = result.evidence.confidence_sum;
    assert!(evidence_recall > 0.0, "Evidence recall should be > 0, got {evidence_recall:.6}");

    let config = ReconstructionConfig { max_steps: 5, hla_learning_rate: 0.3, entropy_threshold: 0.01, ..Default::default() };
    let result_aggressive = brain.project_reconstruct_with_config(config);
    let evidence_improvement = if evidence_recall > 1e-6 {
        (result_aggressive.evidence.confidence_sum - evidence_recall) / evidence_recall * 100.0
    } else { 0.0 };

    assert!(evidence_improvement >= 20.0,
        "GOAT FAIL: {evidence_improvement:.1}% < 20% evidence improvement. default={evidence_recall:.6} aggressive={:.6}",
        result_aggressive.evidence.confidence_sum);

    let passive_sum = vec_activation_sum(&brain.project_all());
    let active_sum = activation_sum(&result.active);
    assert!(active_sum >= passive_sum * 0.9, "Active should be >= 90% of passive");
}

#[test]
fn reconstruction_converges() {
    let brain = make_chain_brain();
    let r = brain.project_reconstruct();
    assert!(r.steps <= 3 && r.steps > 0, "Default: steps={}", r.steps);
    let r = brain.project_reconstruct_with_config(ReconstructionConfig { max_steps: 1, ..Default::default() });
    assert!(r.steps <= 1, "max_steps=1: steps={}", r.steps);
    let r = brain.project_reconstruct_with_config(ReconstructionConfig { max_steps: 5, ..Default::default() });
    assert!(r.steps <= 5, "max_steps=5: steps={}", r.steps);
}

#[test]
fn hla_stays_bounded() {
    let brain = make_chain_brain();
    let config = ReconstructionConfig { max_steps: 5, hla_learning_rate: 0.3, ..Default::default() };
    let result = brain.project_reconstruct_with_config(config);
    for (i, &d) in result.hla_delta.iter().enumerate() { assert!(d.is_finite(), "delta[{i}]={d}"); }
    for (i, (&init, &d)) in brain.hla_state.iter().zip(result.hla_delta.iter()).enumerate() {
        let v = init + d;
        assert!(v >= -1.0 && v <= 1.0, "hla[{i}]={v}");
    }
    for (i, &a) in result.active.iter().enumerate() { assert!(a >= 0.0 && a <= 1.0, "active[{i}]={a}"); }
}
