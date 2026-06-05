//! GOAT Proof: NDS Curvature Proxy — Modelless Inference-Time Budget Control (Plan 186)
//!
//! Verifies that:
//! - G1:  nds_proxy peaked distribution [0.9, 0.05, 0.05] → high (> 0.5)
//! - G2:  nds_proxy flat distribution [0.33, 0.33, 0.34] → low (< 0.5)
//! - G3:  nds_proxy empty → 0.5
//! - G4:  nds_proxy single element → 0.0 (geometric mean = arithmetic mean)
//! - G5:  spectral_balance_score balanced [10, 10, 10] → ~1.0
//! - G6:  spectral_balance_score imbalanced [100, 1, 1] → low
//! - G7:  spectral_balance_score empty → 1.0
//! - G8:  layer_nds_depth first layer → Boundary
//! - G9:  layer_nds_depth last layer → Boundary
//! - G10: layer_nds_depth middle → Middle
//! - G11: layer_nds_depth deep (≥70%) → Deep
//! - G12: SpectralFlatnessBudget::budget_scale(nds=1.0) → 1.0
//! - G13: SpectralFlatnessBudget::budget_scale(nds=0.0) → 1.0 + max_boost

#![cfg(feature = "nds_proxy")]

use katgpt_rs::pruners::{
    LayerDepth, NdsBudgetModifier, SpectralFlatnessBudget, compute_nds_proxy, layer_nds_depth,
    spectral_balance_score,
};

// ── G1: Peaked distribution → high NDS ──────────────────────────

#[test]
fn test_g1_nds_proxy_peaked_high() {
    let result = compute_nds_proxy(&[0.9, 0.05, 0.05]);
    assert!(
        result > 0.5,
        "G1: peaked distribution should have NDS > 0.5, got {result}"
    );
    println!("✅ G1: peaked [0.9, 0.05, 0.05] → NDS = {result:.4} (> 0.5)");
}

// ── G2: Flat distribution → low NDS ─────────────────────────────

#[test]
fn test_g2_nds_proxy_flat_low() {
    let result = compute_nds_proxy(&[0.33, 0.33, 0.34]);
    assert!(
        result < 0.5,
        "G2: flat distribution should have NDS < 0.5, got {result}"
    );
    println!("✅ G2: flat [0.33, 0.33, 0.34] → NDS = {result:.4} (< 0.5)");
}

// ── G3: Empty → 0.5 (default uncertainty) ───────────────────────

#[test]
fn test_g3_nds_proxy_empty() {
    let result = compute_nds_proxy(&[]);
    assert_eq!(
        result, 0.5,
        "G3: empty distribution should return 0.5, got {result}"
    );
    println!("✅ G3: empty → NDS = 0.5 (default uncertainty)");
}

// ── G4: Single element → 0.0 (GM = AM) ──────────────────────────

#[test]
fn test_g4_nds_proxy_single_element() {
    let result = compute_nds_proxy(&[0.5]);
    assert!(
        result.abs() < 1e-6,
        "G4: single element should have NDS ≈ 0.0 (GM = AM), got {result}"
    );
    println!("✅ G4: single element [0.5] → NDS = {result:.6} (≈ 0.0)");
}

// ── G5: Balanced visits → ~1.0 ──────────────────────────────────

#[test]
fn test_g5_spectral_balance_balanced() {
    let result = spectral_balance_score(&[10, 10, 10]);
    assert!(
        (result - 1.0).abs() < 1e-6,
        "G5: balanced visits should score ~1.0, got {result}"
    );
    println!("✅ G5: balanced [10, 10, 10] → balance = {result:.4}");
}

// ── G6: Imbalanced visits → low score ────────────────────────────

#[test]
fn test_g6_spectral_balance_imbalanced() {
    let result = spectral_balance_score(&[100, 1, 1]);
    assert!(
        result < 0.5,
        "G6: imbalanced visits should score low, got {result}"
    );
    println!("✅ G6: imbalanced [100, 1, 1] → balance = {result:.4} (< 0.5)");
}

// ── G7: Empty visits → 1.0 ──────────────────────────────────────

#[test]
fn test_g7_spectral_balance_empty() {
    let result = spectral_balance_score(&[]);
    assert_eq!(
        result, 1.0,
        "G7: empty visits should score 1.0, got {result}"
    );
    println!("✅ G7: empty → balance = 1.0");
}

// ── G8: First layer → Boundary ──────────────────────────────────

#[test]
fn test_g8_layer_depth_first_boundary() {
    assert_eq!(
        layer_nds_depth(0, 12),
        LayerDepth::Boundary,
        "G8: first layer should be Boundary"
    );
    println!("✅ G8: layer 0/12 → Boundary");
}

// ── G9: Last layer → Boundary ───────────────────────────────────

#[test]
fn test_g9_layer_depth_last_boundary() {
    assert_eq!(
        layer_nds_depth(11, 12),
        LayerDepth::Boundary,
        "G9: last layer should be Boundary"
    );
    println!("✅ G9: layer 11/12 → Boundary");
}

// ── G10: Middle layer → Middle ──────────────────────────────────

#[test]
fn test_g10_layer_depth_middle() {
    assert_eq!(
        layer_nds_depth(3, 12),
        LayerDepth::Middle,
        "G10: middle layer should be Middle"
    );
    println!("✅ G10: layer 3/12 → Middle");
}

// ── G11: Deep layer (≥70%) → Deep ──────────────────────────────

#[test]
fn test_g11_layer_depth_deep() {
    // 70% of 12 = 8.4 → layer_idx 9 >= 8
    assert_eq!(
        layer_nds_depth(9, 12),
        LayerDepth::Deep,
        "G11: deep layer (≥70%) should be Deep"
    );
    println!("✅ G11: layer 9/12 (≥70%) → Deep");
}

// ── G12: Confident (NDS=1.0) → budget_scale = 1.0 ──────────────

#[test]
fn test_g12_budget_scale_confident() {
    let modifier = SpectralFlatnessBudget::new(0.5);
    let scale = modifier.budget_scale(1.0);
    assert!(
        (scale - 1.0).abs() < 1e-6,
        "G12: confident (NDS=1.0) should scale to 1.0, got {scale}"
    );
    println!("✅ G12: NDS=1.0 → budget_scale = {scale:.4} (= 1.0)");
}

// ── G13: Uncertain (NDS=0.0) → budget_scale = 1.0 + max_boost ──

#[test]
fn test_g13_budget_scale_uncertain() {
    let max_boost = 0.5f32;
    let modifier = SpectralFlatnessBudget::new(max_boost);
    let scale = modifier.budget_scale(0.0);
    let expected = 1.0 + max_boost;
    assert!(
        (scale - expected).abs() < 1e-6,
        "G13: uncertain (NDS=0.0) should scale to 1.0 + max_boost = {expected}, got {scale}"
    );
    println!("✅ G13: NDS=0.0 → budget_scale = {scale:.4} (= 1.0 + {max_boost})");
}
