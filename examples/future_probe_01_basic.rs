//! Future Probe basic example (Plan 292 Phase 2 T2.7, Research 267).
//!
//! Demonstrates the [`FutureBehaviorProbe`] primitive on synthetic data:
//! 1. Construct a synthetic probe (random direction vector).
//! 2. Forecast future behavior probability on synthetic activations.
//! 3. Verify aligned activations forecast → 1.0, anti-aligned → 0.0.
//! 4. Save / load round-trip + tamper rejection.
//! 5. Atomic hot-swap demonstration.
//!
//! Mirrors `examples/cna_01_discovery.rs` for structure. No real model —
//! synthetic activations only (the goal is to show the primitive's contract,
//! not to steer a real LLM).
//!
//! Run: `cargo run --example future_probe_01_basic --features future_probe`
//!
//! # What This Proves
//!
//! - **Forecast correctness**: σ(w · act + b) on the boundary cases.
//! - **Feature class tag**: `Prediction`, not `Detection`.
//! - **BLAKE3 commitment**: save/load round-trip stable, tamper-rejected.
//! - **Freeze/thaw**: `swap_direction` is atomic for readers.
//!
//! # What This Does NOT Prove
//!
//! - **Real-model steering** — requires a trained probe + real activations
//!   (Phase 4 GOAT gate runs that).
//! - **FPCG selector** — that's the next example (`fpcg_01_basic.rs`).

use katgpt_rs::pruners::future_probe::{FutureBehaviorProbe, ProbeLoadError};
use katgpt_core::traits::ScreeningPruner;

fn main() {
    println!("=== FutureBehaviorProbe Example (Plan 292 Phase 2) ===\n");

    // ── 1. Synthetic probe: direction = [1, 0, 0, 0], bias = -1.0 ─────
    // Meaning: "future behavior probability rises with activation[0]".
    let d_model = 4;
    let direction = vec![1.0, 0.0, 0.0, 0.0];
    let probe = FutureBehaviorProbe::new(direction, -1.0, 7, "refusal");
    println!(
        "Probe: d_model={} layer=7 behavior=\"refusal\" bias=-1.0",
        d_model
    );
    println!("Artifact hash (first 8 bytes): {}", first_8_hex(&probe.artifact_hash()));
    println!("FeatureClass tag: {:?} (Plan 292 Phase 1)", probe.feature_class());
    println!();

    // ── 2. Forecast on boundary activations ─────────────────────────
    let act_aligned = vec![10.0, 0.0, 0.0, 0.0]; // strong positive signal
    let act_anti = vec![-10.0, 0.0, 0.0, 0.0]; // strong negative signal
    let act_neutral = vec![0.0, 99.0, 0.0, 0.0]; // orthogonal to direction

    let f_aligned = probe.forecast(&act_aligned);
    let f_anti = probe.forecast(&act_anti);
    let f_neutral = probe.forecast(&act_neutral);

    println!("Aligned activation  [10, 0, 0, 0]:   forecast = {f_aligned}");
    println!("Anti-aligned act.  [-10, 0, 0, 0]:   forecast = {f_anti}");
    println!("Neutral activation [0, 99, 0, 0]:    forecast = {f_neutral} (= σ(bias))");
    println!();

    assert!(
        f_aligned.probability > 0.99,
        "aligned activation should give p > 0.99, got {}",
        f_aligned.probability
    );
    assert!(
        f_anti.probability < 0.01,
        "anti-aligned should give p < 0.01, got {}",
        f_anti.probability
    );
    // Neutral: dot = 0, bias = -1 → σ(-1) ≈ 0.2689.
    let expected_neutral = 1.0 / (1.0 + 1.0_f32.exp());
    assert!(
        (f_neutral.probability - expected_neutral).abs() < 1e-6,
        "neutral should be σ(bias) = {expected_neutral}, got {}",
        f_neutral.probability
    );
    println!("✓ All boundary forecasts match σ(w · act + b) contract.\n");

    // ── 3. Save / load round-trip + tamper rejection ────────────────
    let bytes = probe.save_to_bytes();
    println!("Serialized probe: {} bytes (FPPB format)", bytes.len());
    let loaded = FutureBehaviorProbe::load_from_bytes(&bytes).expect("round-trip load");
    assert_eq!(loaded.artifact_hash(), probe.artifact_hash());
    assert_eq!(loaded.behavior(), "refusal");
    assert_eq!(loaded.layer(), 7);
    println!("✓ Save/load round-trip preserves probe (hash stable).");

    let mut tampered = bytes.clone();
    tampered[21] ^= 0xFF; // flip a direction byte
    match FutureBehaviorProbe::load_from_bytes(&tampered) {
        Err(ProbeLoadError::HashMismatch { .. }) => {
            println!("✓ Tampered artifact refused (HashMismatch — BLAKE3 commitment works).");
        }
        other => panic!("expected HashMismatch, got {other:?}"),
    }
    println!();

    // ── 4. Atomic hot-swap ──────────────────────────────────────────
    println!("Before swap: artifact_hash = {}", first_8_hex(&probe.artifact_hash()));
    let probe_v2 = FutureBehaviorProbe::new(vec![-1.0, 0.0, 0.0, 0.0], 1.0, 7, "aggression");
    let hash_v2 = probe_v2.artifact_hash();
    probe.swap_direction(probe_v2);
    println!("After swap:  artifact_hash = {}", first_8_hex(&probe.artifact_hash()));
    assert_eq!(probe.artifact_hash(), hash_v2, "swap must replace the hash");

    // After swap, the same activation gives a different forecast.
    let f_after = probe.forecast(&act_aligned);
    println!("Post-swap forecast on [10, 0, 0, 0]: {f_after}");
    // v2: direction [-1, 0, 0, 0], activation [10, 0, 0, 0] → dot = -10, bias = +1 → logit = -9 → σ ≈ 0.0001.
    assert!(
        f_after.probability < 0.001,
        "post-swap forecast should be near 0 (v2 inverts the signal), got {}",
        f_after.probability
    );
    println!("✓ swap_direction replaced the probe atomically (forecast inverted).");

    println!();
    println!("=== Phase 2 FutureBehaviorProbe primitive verified ===");
    println!("Next: Phase 3 — FpcgSelector samples candidates and picks by forecast score.");
}

fn first_8_hex(h: &[u8; 32]) -> String {
    h.iter().take(8).map(|b| format!("{b:02x}")).collect()
}
