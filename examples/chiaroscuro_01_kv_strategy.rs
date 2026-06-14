//! CHIAR-KV Demo — Per-token spectral entropy routing to KV cache strategies.
//!
//! Run: `cargo run --example chiaroscuro_01_kv_strategy --features chiaroscuro`
//!
//! Shows before/after KV cache sizes when applying per-token storage strategy:
//! - Smooth tokens (low H) → DCT-truncated (≈4× compression for d=256)
//! - Mid tokens → SpectralQuant (4× compression, delegated)
//! - Complex tokens (high H) → FullPrecision (no compression)
//!
//! Compare with the no-CHIAR baseline where every token is FullPrecision.

#[cfg(feature = "chiaroscuro")]
fn main() {
    use katgpt_rs::chiaroscuro::{
        kv::{ChiaroscuroKvStrategy, DEFAULT_DCT_TRUNCATED_COEFFS},
        tau::StreamingTauCalibrator,
    };

    println!("=== CHIAR-KV Cache Strategy Demo (Plan 269, Fusion A) ===\n");

    // Simulate a stream of 1000 key embeddings.
    // Mix: 60% smooth (constant), 30% mid (sinusoid), 10% complex (noise).
    let d = 256_usize;
    let mut rng = fastrand::Rng::with_seed(42);
    let keys: Vec<Vec<f32>> = (0..1000)
        .map(|i| match i % 10 {
            0..=5 => vec![0.5_f32; d],                                   // smooth
            6..=8 => (0..d).map(|j| ((j as f32) * 0.1).sin()).collect(), // mid
            _ => (0..d).map(|_| rng.f32() * 2.0 - 1.0).collect(),        // complex
        })
        .collect();

    // Calibrate τ on the stream.
    let mut calibrator = StreamingTauCalibrator::default();
    for k in &keys {
        calibrator.observe_embedding(k);
    }
    let tau_lo = calibrator.tau_lo_mut();
    let tau_hi = calibrator.tau_hi_mut();
    println!("Calibrated τ_lo = {tau_lo:.4}, τ_hi = {tau_hi:.4}");
    println!("Observations: {} (window {})\n", calibrator.count(), calibrator.window_len());

    // Apply per-token strategy.
    let mut counts = [0u64; ChiaroscuroKvStrategy::NUM_VARIANTS];
    for k in &keys {
        let s = ChiaroscuroKvStrategy::decide_from_key(k, tau_lo, tau_hi);
        counts[s.as_index()] += 1;
    }

    let total = keys.len() as u64;
    println!("Per-token strategy distribution:");
    for (i, label) in ["DctTruncated", "Quantized   ", "FullPrecision"].iter().enumerate() {
        let c = counts[i];
        let pct = 100.0 * c as f32 / total as f32;
        let ratio = ChiaroscuroKvStrategy::from_index(i)
            .unwrap()
            .compression_ratio(d, DEFAULT_DCT_TRUNCATED_COEFFS);
        println!("  {label}: {c:>4} ({pct:>5.1}%)  compression ratio = {ratio:.2}×");
    }

    // Compute effective KV cache size.
    let baseline_bytes = total as usize * d * 2; // f16 baseline
    let mut chiar_bytes = 0usize;
    for k in &keys {
        let s = ChiaroscuroKvStrategy::decide_from_key(k, tau_lo, tau_hi);
        chiar_bytes += match s {
            ChiaroscuroKvStrategy::DctTruncated => DEFAULT_DCT_TRUNCATED_COEFFS * 4 + 4,
            ChiaroscuroKvStrategy::Quantized => d * 2 / 4, // ~4× compression
            ChiaroscuroKvStrategy::FullPrecision => d * 2,
        };
    }
    let compression = baseline_bytes as f32 / chiar_bytes as f32;
    println!("\nKV cache size (d={d}, {total} tokens):");
    println!("  Baseline (all f16):   {baseline_bytes:>10} bytes");
    println!("  CHIAR-KV (mixed):     {chiar_bytes:>10} bytes");
    println!("  Effective compression: {compression:.2}×");
    println!("\nTheorem 1 guarantee: DctTruncated tokens have reconstruction");
    println!("error bounded by their spectral tail energy (smooth → ≈0 error).");
}

#[cfg(not(feature = "chiaroscuro"))]
fn main() {
    eprintln!("This example requires the `chiaroscuro` feature.");
    eprintln!("Run with: cargo run --example chiaroscuro_01_kv_strategy --features chiaroscuro");
}
