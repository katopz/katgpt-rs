//! Integration tests for the personality-weighted composition primitive
//! (Plan 297 — Phases 1, 2, 3, 4 GOAT gates).
//!
//! These cover:
//! - Phase 1 T1.V2: compose kernel correctness (zero weights, extreme ±, belief decay)
//! - Phase 2 T2.V1: drift rule (positive/negative surprise, clamp, EMA)
//! - Phase 4 G1: `compose_tau_infinity_uniform` — no personality at τ → ∞
//! - Phase 4 G4-supporting: sigmoid stability at extreme inputs

use crate::personality_composition::PersonalityWeightedComposition;
use crate::personality_composition::kernel::tests::StaticLayer;
use crate::personality_composition::sigmoid::sigmoid;
use crate::personality_composition::trait_def::LayerDirectionSource;
use crate::personality_composition::types::PersonalityConfig;

// ─── Helpers ───────────────────────────────────────────────────────────────

/// A direction vector with 1.0 at index `i` and 0 elsewhere.
fn unit_direction_32(i: usize) -> [f32; 32] {
    let mut d = [0.0f32; 32];
    d[i] = 1.0;
    d
}

/// A direction vector with `v` at every element.
fn uniform_direction_32(v: f32) -> [f32; 32] {
    [v; 32]
}

// ─── Phase 1 T1.V2: compose correctness ────────────────────────────────────

#[test]
fn compose_zero_weights_uniform() {
    // Plan 297 T1.V2 / R276 §5: when all wᵢ = 0 and τ finite, each sigmoid(0/τ)
    // = 0.5, so output = 0.5 × Σ dᵢ.
    let kernel = PersonalityWeightedComposition::<3, 32>::uniform();
    let l0 = StaticLayer::new(unit_direction_32(0));
    let l1 = StaticLayer::new(unit_direction_32(1));
    let l2 = StaticLayer::new(unit_direction_32(2));
    let layers: [&dyn LayerDirectionSource; 3] = [&l0, &l1, &l2];

    let mut scratch = [0.0f32; 32];
    let mut out = [0.0f32; 32];
    kernel.compose_into(&layers, &mut scratch, &mut out);

    // Each layer contributes 0.5 × dᵢ. d0 = e_0, d1 = e_1, d2 = e_2.
    // Output should be 0.5 at indices 0, 1, 2 and 0 elsewhere.
    assert!((out[0] - 0.5).abs() < 1e-6, "out[0] = {}", out[0]);
    assert!((out[1] - 0.5).abs() < 1e-6, "out[1] = {}", out[1]);
    assert!((out[2] - 0.5).abs() < 1e-6, "out[2] = {}", out[2]);
    for (j, &v) in out.iter().enumerate().skip(3) {
        assert!(v.abs() < 1e-6, "out[{j}] should be 0, got {v}");
    }
}

#[test]
fn compose_extreme_positive_weight_selects_layer() {
    // Plan 297 T1.V2: large wᵢ → sigmoid(wᵢ/τ) ≈ 1.0 → output ≈ dᵢ.
    let kernel = PersonalityWeightedComposition::<3, 32>::new(
        PersonalityConfig::default(),
        [50.0, -50.0, -50.0],
    );
    let l0 = StaticLayer::new(uniform_direction_32(1.0));
    let l1 = StaticLayer::new(uniform_direction_32(1.0));
    let l2 = StaticLayer::new(uniform_direction_32(1.0));
    let layers: [&dyn LayerDirectionSource; 3] = [&l0, &l1, &l2];

    let mut scratch = [0.0f32; 32];
    let mut out = [0.0f32; 32];
    kernel.compose_into(&layers, &mut scratch, &mut out);

    // Layer 0: sigmoid(50/1) ≈ 1.0 → contributes ~1.0 × d0 = [1.0; 32]
    // Layers 1, 2: sigmoid(-50/1) ≈ 0.0 → contribute ~0
    for (j, &v) in out.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-4, "out[{j}] should be ~1.0, got {v}");
    }
}

#[test]
fn compose_extreme_negative_weight_zeros_layer() {
    // Plan 297 T1.V2: large negative wᵢ → sigmoid(wᵢ/τ) ≈ 0 → layer ~0.
    let kernel =
        PersonalityWeightedComposition::<2, 32>::new(PersonalityConfig::default(), [-50.0, 50.0]);
    let l0 = StaticLayer::new(uniform_direction_32(1.0));
    let l1 = StaticLayer::new(uniform_direction_32(2.0));
    let layers: [&dyn LayerDirectionSource; 2] = [&l0, &l1];

    let mut scratch = [0.0f32; 32];
    let mut out = [0.0f32; 32];
    kernel.compose_into(&layers, &mut scratch, &mut out);

    // Layer 0 contributes ~0 (suppressed).
    // Layer 1 contributes ~1.0 × [2.0; 32] = [2.0; 32].
    for (j, &v) in out.iter().enumerate() {
        assert!((v - 2.0).abs() < 1e-4, "out[{j}] should be ~2.0, got {v}");
    }
}

#[test]
fn compose_belief_confidence_decay_shrinks_contribution() {
    // Plan 297 T1.V2: layer with belief_confidence = 0.1 contributes 10% of full.
    let kernel = PersonalityWeightedComposition::<2, 32>::new(
        PersonalityConfig::default(),
        [50.0, 50.0], // both layers fully gated ON
    );
    let l0 = StaticLayer::new(uniform_direction_32(1.0)); // confidence 1.0 (default)
    let l1 = StaticLayer::new(uniform_direction_32(1.0)).with_confidence(0.1);
    let layers: [&dyn LayerDirectionSource; 2] = [&l0, &l1];

    let mut scratch = [0.0f32; 32];
    let mut out = [0.0f32; 32];
    kernel.compose_into(&layers, &mut scratch, &mut out);

    // Layer 0: sigmoid(50/1) ≈ 1.0, confidence 1.0 → ~1.0 × [1.0; 32]
    // Layer 1: sigmoid(50/1) ≈ 1.0, confidence 0.1 → ~0.1 × [1.0; 32]
    // Total: ~1.1 × [1.0; 32]
    for (j, &v) in out.iter().enumerate() {
        assert!((v - 1.1).abs() < 1e-3, "out[{j}] should be ~1.1, got {v}");
    }
}

// ─── Phase 2 T2.V1: drift rule ─────────────────────────────────────────────

#[test]
fn drift_positive_surprise_reinforces() {
    // Plan 297 T2.V1: r_obs > r_expected, positive d_recent → w increases.
    let config = PersonalityConfig {
        alpha: 0.1, // amplify for test visibility
        ..Default::default()
    };
    let mut kernel = PersonalityWeightedComposition::<1, 32>::new(config, [0.0]);

    // d_recent = [1.0; 32] (sum = 32.0), r_expected = 0.0 initially.
    let l0 = StaticLayer::new(uniform_direction_32(1.0)).with_recent(uniform_direction_32(1.0));
    let layers: [&dyn LayerDirectionSource; 1] = [&l0];

    let w_before = kernel.w_snapshot()[0];
    kernel.drift(&layers, 1.0); // r_observed = 1.0, surprise = 1.0 - 0.0 = 1.0
    let w_after = kernel.w_snapshot()[0];

    // Δw = alpha * surprise * Σ d_recent = 0.1 * 1.0 * 32.0 = 3.2.
    assert!(
        w_after > w_before,
        "positive surprise should increase w: before={w_before}, after={w_after}"
    );
    assert!(
        (w_after - 3.2).abs() < 1e-5,
        "Δw should be 3.2, got {w_after}"
    );
}

#[test]
fn drift_negative_surprise_penalizes() {
    // Plan 297 T2.V1: r_obs < r_expected, positive d_recent → w decreases.
    let config = PersonalityConfig {
        alpha: 0.1,
        ..Default::default()
    };
    let mut kernel = PersonalityWeightedComposition::<1, 32>::new(config, [0.0]);

    let l0 = StaticLayer::new(uniform_direction_32(1.0)).with_recent(uniform_direction_32(1.0));
    let layers: [&dyn LayerDirectionSource; 1] = [&l0];

    // First drift to set r_expected to ~0.05 (= 0.05 * 1.0).
    kernel.drift(&layers, 1.0);

    // Now r_observed = 0.0 < r_expected → negative surprise.
    let w_before = kernel.w_snapshot()[0];
    kernel.drift(&layers, 0.0);
    let w_after = kernel.w_snapshot()[0];

    assert!(
        w_after < w_before,
        "negative surprise should decrease w: before={w_before}, after={w_after}"
    );
}

#[test]
fn drift_clamps_to_w_max() {
    // Plan 297 T2.V1: repeated positive drift saturates at w_max.
    let config = PersonalityConfig {
        alpha: 1.0, // huge step
        w_max: 2.0,
        ..Default::default()
    };
    let mut kernel = PersonalityWeightedComposition::<1, 32>::new(config, [0.0]);

    let l0 = StaticLayer::new(uniform_direction_32(1.0)).with_recent(uniform_direction_32(1.0));
    let layers: [&dyn LayerDirectionSource; 1] = [&l0];

    // Drift many times with positive surprise — should saturate at w_max.
    for _ in 0..100 {
        kernel.drift(&layers, 100.0);
    }

    let w = kernel.w_snapshot()[0];
    assert!(
        (w - 2.0).abs() < 1e-5,
        "w should saturate at w_max=2.0, got {w}"
    );
}

#[test]
fn drift_ema_tracks_recent_reward() {
    // Plan 297 T2.V1: r_expected converges to r_observed over many steps.
    let config = PersonalityConfig {
        ema_decay: 0.5, // faster convergence for the test
        ..Default::default()
    };
    let mut kernel = PersonalityWeightedComposition::<1, 32>::new(config, [0.0]);

    // Layer with no recent direction → no w drift, but r_expected still tracks.
    let l0 = StaticLayer::new(uniform_direction_32(1.0)); // no with_recent → empty recent_direction
    let layers: [&dyn LayerDirectionSource; 1] = [&l0];

    // Feed constant reward = 1.0 — r_expected should converge to 1.0.
    for _ in 0..50 {
        kernel.drift(&layers, 1.0);
    }

    let r_exp = kernel.r_expected()[0];
    assert!(
        (r_exp - 1.0).abs() < 1e-3,
        "r_expected should converge to 1.0, got {r_exp}"
    );

    // w should NOT have moved (empty recent_direction → zero delta).
    assert!(
        kernel.w_snapshot()[0].abs() < 1e-6,
        "w should not drift with empty recent_direction"
    );
}

// ─── Phase 4 G1: tau_infinity_uniform ──────────────────────────────────────

#[test]
fn g1_compose_tau_infinity_uniform() {
    // Plan 297 Phase 4 G1: when τ → ∞, all weights contribute 0.5, output =
    // 0.5 × Σ dᵢ (no personality). Divergence only with finite τ.
    let config = PersonalityConfig {
        tau: f32::INFINITY,
        ..Default::default()
    };
    let kernel = PersonalityWeightedComposition::<3, 32>::new(
        config,
        [10.0, -10.0, 5.0], // extreme weights — should be IGNORED at τ = ∞
    );

    let l0 = StaticLayer::new(uniform_direction_32(1.0));
    let l1 = StaticLayer::new(uniform_direction_32(1.0));
    let l2 = StaticLayer::new(uniform_direction_32(1.0));
    let layers: [&dyn LayerDirectionSource; 3] = [&l0, &l1, &l2];

    let mut scratch = [0.0f32; 32];
    let mut out = [0.0f32; 32];
    kernel.compose_into(&layers, &mut scratch, &mut out);

    // At τ = ∞, sigmoid(w/∞) = sigmoid(0) = 0.5 for all layers regardless of w.
    // Total = 0.5 × 3 × [1.0; 32] = [1.5; 32].
    for (j, &v) in out.iter().enumerate() {
        assert!(
            (v - 1.5).abs() < 1e-4,
            "out[{j}] should be 1.5 (no-personality baseline), got {v}"
        );
    }
}

// ─── Phase 4 sigmoid stability (R276 §5) ──────────────────────────────────

#[test]
fn sigmoid_stable_for_extreme_inputs() {
    // R276 §5 / Plan 297: no overflow/NaN for |x| > 50.
    for &x in &[100.0f32, -100.0, 1000.0, -1000.0, 1e10, -1e10] {
        let s = sigmoid(x);
        assert!(s.is_finite(), "sigmoid({x}) is not finite: {s}");
        assert!((0.0..=1.0).contains(&s), "sigmoid({x}) out of [0,1]: {s}");
    }
}

// ─── Phase 2: empty recent_direction semantics ─────────────────────────────

#[test]
fn drift_empty_recent_direction_zero_delta() {
    // A layer whose recent_direction() returns &[] should not influence w,
    // but r_expected should still update.
    let config = PersonalityConfig::default();
    let mut kernel = PersonalityWeightedComposition::<1, 32>::new(config, [0.5]);

    let l0 = StaticLayer::new(uniform_direction_32(1.0)); // no recent
    let layers: [&dyn LayerDirectionSource; 1] = [&l0];

    kernel.drift(&layers, 1.0);

    // w should be unchanged.
    assert!((kernel.w_snapshot()[0] - 0.5).abs() < 1e-6);
    // r_expected should have moved.
    assert!(kernel.r_expected()[0] > 0.0);
}

// ─── Phase 1: multi-layer integration smoke test ───────────────────────────

#[test]
fn compose_three_layers_mixed_weights_and_directions() {
    // End-to-end: 3 layers with distinct directions and weights.
    let config = PersonalityConfig::default();
    let kernel = PersonalityWeightedComposition::<3, 32>::new(config, [0.0, 4.0, -4.0]);

    let l0 = StaticLayer::new(unit_direction_32(0)); // e_0, w=0 → 0.5 × e_0
    let l1 = StaticLayer::new(unit_direction_32(1)); // e_1, w=4 → sigmoid(4) ≈ 0.982 × e_1
    let l2 = StaticLayer::new(unit_direction_32(2)); // e_2, w=-4 → sigmoid(-4) ≈ 0.018 × e_2
    let layers: [&dyn LayerDirectionSource; 3] = [&l0, &l1, &l2];

    let mut scratch = [0.0f32; 32];
    let mut out = [0.0f32; 32];
    kernel.compose_into(&layers, &mut scratch, &mut out);

    let expected_0 = sigmoid(0.0 / 1.0);
    let expected_1 = sigmoid(4.0 / 1.0);
    let expected_2 = sigmoid(-4.0 / 1.0);

    assert!((out[0] - expected_0).abs() < 1e-5, "out[0] mismatch");
    assert!((out[1] - expected_1).abs() < 1e-5, "out[1] mismatch");
    assert!((out[2] - expected_2).abs() < 1e-5, "out[2] mismatch");
    for (j, &v) in out.iter().enumerate().skip(3) {
        assert!(v.abs() < 1e-6, "out[{j}] should be 0");
    }
}

// ─── Phase 2: restore_w round-trip ─────────────────────────────────────────

#[test]
fn restore_w_roundtrips_cleanly() {
    let mut kernel =
        PersonalityWeightedComposition::<3, 32>::new(PersonalityConfig::default(), [0.1, 0.2, 0.3]);
    let snapshot = *kernel.w_snapshot();

    // Mutate.
    kernel.restore_w([0.9, 0.8, 0.7]);
    assert_ne!(kernel.w_snapshot(), &snapshot);

    // Restore.
    kernel.restore_w(snapshot);
    assert_eq!(kernel.w_snapshot(), &snapshot);
}
