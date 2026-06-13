//! GPart Isometric Partition Adapter Demo (Plan 257).
//!
//! Demonstrates:
//! - Construction from seed + θ
//! - apply() on sample weights
//! - commitment() / verify()
//! - Binary save/load roundtrip
//!
//! Run: `cargo run --example gpart_adapter_demo --features gpart_adapter`

#[cfg(feature = "gpart_adapter")]
fn main() {
    use katgpt_core::GpartAdapter;
    use std::env;

    // 1. Construct adapter from seed + θ
    let seed = 42u64;
    let d = 8;
    let theta: Vec<f32> = (1..=d).map(|i| i as f32 * 0.1).collect();
    let adapter = GpartAdapter { d, seed, theta };

    println!("=== GPart Adapter Demo ===");
    println!(
        "d={}, seed={}, θ={:?}",
        adapter.d, adapter.seed, adapter.theta
    );
    println!("Storage: {} bytes", adapter.storage_bytes());

    // 2. Apply to sample weights
    let mut weights = vec![0.0f32; 64];
    adapter.apply(&mut weights);
    println!(
        "\nApplied to {} weights (first 8): {:?}",
        weights.len(),
        &weights[..8]
    );

    // 3. Verify isometry: ||Pθ||² = ||θ||²
    let isometric = adapter.check_isometry(256);
    println!(
        "Isometry check (N=256): {}",
        if isometric { "PASS" } else { "FAIL" }
    );

    // 4. Commitment + verify
    let commit = adapter.commitment();
    println!("\nBLAKE3 commitment: {:02x?}", &commit[..8]);
    println!("Verify (correct): {}", adapter.verify(&commit));

    let mut tampered = commit;
    tampered[0] ^= 0xFF;
    println!("Verify (tampered): {}", adapter.verify(&tampered));

    // 5. Save/load roundtrip
    let tmp = env::temp_dir().join("gpart_adapter_demo.bin");
    adapter.save(&tmp).expect("save failed");
    let loaded = GpartAdapter::load(&tmp).expect("load failed");
    println!(
        "\nSave/load roundtrip: d={}, seed={}, θ match={}",
        loaded.d,
        loaded.seed,
        loaded.theta == adapter.theta
    );
    std::fs::remove_file(&tmp).ok();

    // 6. Determinism check
    let mut w1 = vec![0.0f32; 64];
    let mut w2 = vec![0.0f32; 64];
    adapter.apply(&mut w1);
    adapter.apply(&mut w2);
    println!("Determinism: {}", if w1 == w2 { "PASS" } else { "FAIL" });

    println!("\n=== Demo Complete ===");
}

#[cfg(not(feature = "gpart_adapter"))]
fn main() {
    eprintln!("This example requires the 'gpart_adapter' feature.");
    eprintln!("Run with: cargo run --example gpart_adapter_demo --features gpart_adapter");
}
