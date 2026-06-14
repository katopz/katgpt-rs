//! CHIAR Collapse Discovery Demo — Automated operator promotion.
//!
//! Run: `cargo run --example chiaroscuro_03_collapse_discovery --features chiaroscuro`
//!
//! Demonstrates paper's Remark 1: routing collapse as discovery mechanism.
//!
//! Sets up a 3-operator router (DCT, Quantized-via-KVarN-analog, FullAttn) and
//! feeds it a stream that naturally favors only 2 ops. The harness detects the
//! collapse, identifies the survivor subset, and recommends demoting the
//! unused op (paper's RBF removal).

#[cfg(feature = "chiaroscuro")]
fn main() {
    use katgpt_rs::chiaroscuro::{
        collapse::CollapseDiscoveryHarness,
        op_trait::{ChiaroscuroOp, ChiaroscuroRouter, DctMixOp, FullAttnOp},
        tau::DEFAULT_TAU_HI,
    };

    println!("=== CHIAR Collapse Discovery Demo (Plan 269, Fusion C) ===\n");
    println!("Paper's finding: 3-operator router (DCT+RBF+Attn) collapses to");
    println!("DCT+Attn. RBF is consistently rejected → removing it improves quality.\n");

    // Build 2-op router: DctMix (low entropy) + FullAttn (high entropy).
    // The "RBF" middle op is intentionally not included — this demo shows the
    // harness detecting collapse to a 1-op or 2-op subset.
    let ops: Vec<Box<dyn ChiaroscuroOp>> = vec![
        Box::new(DctMixOp::default()),
        Box::new(FullAttnOp::default()),
    ];
    let router = ChiaroscuroRouter::new(ops);
    let mut harness = CollapseDiscoveryHarness::new(router, 100, 0.10);

    // Feed a stream that favors only DctMix (all smooth tokens).
    println!("Phase 1: Feeding 200 smooth (low-H) tokens...");
    for _ in 0..200 {
        // Constant embedding → H ≈ 0 → routes to DctMix.
        harness.observe(&[0.5_f32; 64]);
    }

    let total = harness.total_observations();
    let u = harness.router.utilization_entropy();
    println!("  Observations: {total}");
    println!("  Utilization entropy U = {u:.4}");
    println!("  Per-op counts: {:?}", harness.router.utilization_counts());

    if let Some(promotion) = harness.check_collapse() {
        println!("\n✓ Collapse detected!");
        println!("  Survivors (keep): {:?}", promotion.keep);
        println!("  Demote: {:?}", promotion.demote);
        println!("  Recommendation: remove ops at indices {:?} from the router.", promotion.demote);
        println!("  This mirrors the paper's RBF removal — improves quality");
        println!("  by avoiding wasted capacity on redundant operators.");
    } else {
        println!("\n✗ No collapse detected (U too high or not enough samples).");
    }

    // Phase 2: Reset and feed mixed stream — no collapse expected.
    println!("\nPhase 2: Reset and feed mixed (50/50 smooth/complex) stream...");
    harness.reset();
    let mut rng = fastrand::Rng::with_seed(7);
    for _ in 0..200 {
        // Alternate smooth and random.
        harness.observe(&[0.5_f32; 64]);
        let x: Vec<f32> = (0..64).map(|_| rng.f32() * 2.0 - 1.0).collect();
        harness.observe(&x);
    }
    let u2 = harness.router.utilization_entropy();
    println!("  Observations: {}", harness.total_observations());
    println!("  Utilization entropy U = {u2:.4}");
    println!("  Per-op counts: {:?}", harness.router.utilization_counts());

    let snap = harness.current_snapshot();
    if snap.collapsed() {
        println!("\n✗ Unexpected collapse on balanced stream.");
    } else {
        println!("\n✓ No collapse — both operators used (U close to 1.0).");
        println!("  This confirms the harness doesn't false-positive on naturalistic mix.");
    }

    // Phase 3: Show how τ interacts with regime.
    println!("\nPhase 3: Default τ_hi = {DEFAULT_TAU_HI}");
    println!("  Tokens with H > τ_hi route to FullAttn.");
    println!("  In production, τ is calibrated online via StreamingTauCalibrator.");
}

#[cfg(not(feature = "chiaroscuro"))]
fn main() {
    eprintln!("This example requires the `chiaroscuro` feature.");
    eprintln!("Run with: cargo run --example chiaroscuro_03_collapse_discovery --features chiaroscuro");
}
