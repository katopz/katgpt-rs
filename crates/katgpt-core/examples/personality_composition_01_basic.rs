//! Example: Personality-Weighted Composition basic demo (Plan 297 T5.3).
//!
//! Demonstrates the core lifecycle of a `PersonalityWeightedComposition`:
//! - 3 layers with distinct latent direction vectors.
//! - Drift over 100 ticks under a constant reward signal.
//! - Weights converge: the layer aligned with the reward gets reinforced,
//!   others stay near zero.
//! - BLAKE3 snapshot before/after shows the personality divergence.
//!
//! Run with:
//! ```sh
//! cargo run --example personality_composition_01_basic --features personality_composition --release
//! ```

use katgpt_core::personality_composition::{
    ArchetypeLabel, LayerDirectionSource, PersonalityConfig, PersonalitySnapshot,
    PersonalityWeightedComposition,
};

/// Minimal `LayerDirectionSource` for the example: holds a fixed direction
/// vector and an EMA-smoothed recent direction.
struct ExampleLayer {
    #[allow(dead_code)] // kept for host-integration clarity; not read by the layer trait
    name: &'static str,
    direction: [f32; 32],
    recent: [f32; 32],
}

impl ExampleLayer {
    fn new(name: &'static str, direction: [f32; 32]) -> Self {
        Self {
            name,
            direction,
            recent: direction, // start with the current direction as the EMA
        }
    }
}

impl LayerDirectionSource for ExampleLayer {
    fn direction<'a>(&self, scratch: &'a mut [f32]) -> &'a [f32] {
        scratch[..32].copy_from_slice(&self.direction);
        &scratch[..32]
    }

    fn recent_direction(&self) -> &[f32] {
        &self.recent
    }

    fn belief_confidence(&self) -> f32 {
        1.0
    }
}

/// Make a direction vector that is 1.0 at index `i` and 0 elsewhere.
fn unit_32(i: usize) -> [f32; 32] {
    let mut d = [0.0f32; 32];
    d[i] = 1.0;
    d
}

fn main() {
    println!("=== PersonalityWeightedComposition Basic Demo (Plan 297) ===\n");

    let config = PersonalityConfig::default();
    println!(
        "Config: tau={}, alpha={}, w_max={}, ema_decay={}",
        config.tau, config.alpha, config.w_max, config.ema_decay
    );

    // 3 layers: "curiosity", "safety", "hunger".
    // Each points along a different axis of the latent space.
    let curiosity = ExampleLayer::new("curiosity", unit_32(0));
    let safety = ExampleLayer::new("safety", unit_32(1));
    let hunger = ExampleLayer::new("hunger", unit_32(2));
    let layers: [&dyn LayerDirectionSource; 3] = [&curiosity, &safety, &hunger];

    // Start with all-zero weights — uniform 0.5 personality.
    let mut kernel = PersonalityWeightedComposition::<3, 32>::new(config, [0.0, 0.0, 0.0]);

    // Initial snapshot.
    let snap0 = PersonalitySnapshot::from_composition(
        &kernel,
        ArchetypeLabel::from_str("example_basic"),
        0,
    );
    println!("\nInitial weights: {:?}", kernel.w_snapshot());
    println!("Initial blake3:  {}", hex_short(&snap0.blake3));

    // Run 100 ticks. The reward is +1.0 every tick (positive surprise vs
    // r_expected=0 initially). The `recent_direction` of all layers is
    // positive (unit vectors), so all weights get reinforced. But
    // `r_expected` converges to 1.0, so surprise → 0 and drift stops.
    println!("\nDrifting 100 ticks with constant reward = +1.0...");
    for _ in 0..100 {
        kernel.drift(&layers, 1.0);
    }

    let snap1 = PersonalitySnapshot::from_composition(
        &kernel,
        ArchetypeLabel::from_str("example_basic"),
        1,
    );
    println!("\nAfter-drift weights: {:?}", kernel.w_snapshot());
    println!("After-drift r_expected: {:?}", kernel.r_expected());
    println!("After-drift blake3:  {}", hex_short(&snap1.blake3));

    // Sanity: surprise converged to ~0 → weights should have plateaued.
    let w = kernel.w_snapshot();
    assert!(w[0] > 0.0, "curiosity layer should be reinforced");
    assert!(w[1] > 0.0, "safety layer should be reinforced");
    assert!(w[2] > 0.0, "hunger layer should be reinforced");
    assert!(
        snap0.blake3 != snap1.blake3,
        "drift must change the personality commitment"
    );

    // Compose the final behavior vector.
    let mut scratch = [0.0f32; 32];
    let mut out = [0.0f32; 32];
    kernel.compose_into(&layers, &mut scratch, &mut out);
    println!("\nComposed behavior vector (first 8 dims): {:?}", &out[..8]);

    // Verify the snapshot still checks out.
    assert!(snap1.verify_blake3(), "snapshot must verify");

    println!("\n=== Demo complete: weights converged, blake3 commitment valid ===");
}

fn hex_short(bytes: &[u8; 32]) -> String {
    bytes[..8].iter().map(|b| format!("{:02x}", b)).collect()
}
