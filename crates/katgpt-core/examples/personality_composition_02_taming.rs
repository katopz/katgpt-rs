//! Example: Personality-Weighted Composition — Taming demo (Plan 297 T5.4).
//!
//! Demonstrates the *open-primitive half* of the wildlife-taming story. The
//! host (riir-ai Plan 327) owns the species-swap; here we only show the
//! personality kernel dynamics:
//!
//! - A wildlife entity has a `COMPANIONS(player)` layer whose weight starts
//!   negative (fearful of the player).
//! - Each tick the player feeds the wildlife (`reward = +1.0`); drift pushes
//!   `w_COMPANIONS(player)` upward.
//! - After enough ticks, `w_COMPANIONS(player)` crosses `τ_tame` — the host
//!   would now trigger the species-swap (handled in riir-ai).
//!
//! The `τ_tame` threshold is host-defined; here we use `τ_tame = 1.0` as a
//! reasonable sigmoid gate (sigmoid(1.0/1.0) ≈ 0.73 → the COMPANIONS layer
//! is "mostly on").
//!
//! Run with:
//! ```sh
//! cargo run --example personality_composition_02_taming --features personality_composition --release
//! ```

use katgpt_core::personality_composition::{
    ArchetypeLabel, LayerDirectionSource, PersonalityConfig, PersonalityWeightedComposition,
};

/// The "tame threshold" — when `w_COMPANIONS(player)` crosses this, the host
/// (riir-ai) triggers the species-swap from wildlife → pet. This is a
/// host-defined constant; katgpt-rs only supplies the open primitive.
const TAU_TAME: f32 = 1.0;

/// 3-layer model for the wildlife entity (simplified from the 9-layer
/// production Entity Cognition Stack in riir-ai Research 146):
/// - SENSE (immediate environment) — always on, full confidence.
/// - SAFETY (self-preservation) — always on.
/// - COMPANIONS(player) — the taming target; starts suppressed.
struct WildlifeLayers {
    sense: [f32; 32],
    safety: [f32; 32],
    companions: [f32; 32],
    /// EMA of the COMPANIONS direction (for drift).
    companions_recent: [f32; 32],
}

impl WildlifeLayers {
    fn new() -> Self {
        // SENSE: along axis 0.
        let mut sense = [0.0f32; 32];
        sense[0] = 1.0;
        // SAFETY: along axis 1.
        let mut safety = [0.0f32; 32];
        safety[1] = 1.0;
        // COMPANIONS(player): along axis 2 — "approach the player".
        let mut companions = [0.0f32; 32];
        companions[2] = 1.0;
        Self {
            sense,
            safety,
            companions,
            companions_recent: companions,
        }
    }
}

struct SenseLayer<'a>(&'a WildlifeLayers);
struct SafetyLayer<'a>(&'a WildlifeLayers);
struct CompanionsLayer<'a>(&'a WildlifeLayers);

impl<'a> LayerDirectionSource for SenseLayer<'a> {
    fn direction<'b>(&self, scratch: &'b mut [f32]) -> &'b [f32] {
        scratch[..32].copy_from_slice(&self.0.sense);
        &scratch[..32]
    }
    fn belief_confidence(&self) -> f32 {
        1.0 // plasma-tier: always fully confident
    }
}

impl<'a> LayerDirectionSource for SafetyLayer<'a> {
    fn direction<'b>(&self, scratch: &'b mut [f32]) -> &'b [f32] {
        scratch[..32].copy_from_slice(&self.0.safety);
        &scratch[..32]
    }
    fn belief_confidence(&self) -> f32 {
        1.0
    }
}

impl<'a> LayerDirectionSource for CompanionsLayer<'a> {
    fn direction<'b>(&self, scratch: &'b mut [f32]) -> &'b [f32] {
        scratch[..32].copy_from_slice(&self.0.companions);
        &scratch[..32]
    }
    fn recent_direction(&self) -> &[f32] {
        &self.0.companions_recent
    }
    fn belief_confidence(&self) -> f32 {
        // The host could decay this when the player is out of sight; here we
        // assume the player is present throughout the feeding.
        1.0
    }
}

fn main() {
    println!("=== PersonalityWeightedComposition Taming Demo (Plan 297 T5.4) ===\n");
    println!("Goal: show w_COMPANIONS(player) rising above tau_tame = {TAU_TAME}\n");

    let config = PersonalityConfig {
        alpha: 0.05, // a bit faster than default for demo pacing
        ..Default::default()
    };
    println!(
        "Config: tau={}, alpha={}, w_max={}, ema_decay={}",
        config.tau, config.alpha, config.w_max, config.ema_decay
    );

    // Initial weights: SENSE on, SAFETY on, COMPANIONS(player) SUPPRESSED
    // (the wildlife is fearful of the player at spawn).
    let w_initial = [0.5, 0.5, -3.0];
    let mut kernel = PersonalityWeightedComposition::<3, 32>::new(config, w_initial);
    println!("\nInitial weights: {:?}", kernel.w_snapshot());
    println!(
        "  (COMPANIONS(player) starts at -3.0 — sigmoid(-3/1) = {:.4})",
        katgpt_core::personality_composition::sigmoid(-3.0)
    );

    let layers_storage = WildlifeLayers::new();

    // Construct the layer array inline — each wrapper borrows `layers_storage`
    // for the lifetime of the composition.
    let sense = SenseLayer(&layers_storage);
    let safety = SafetyLayer(&layers_storage);
    let companions = CompanionsLayer(&layers_storage);
    let layers: [&dyn LayerDirectionSource; 3] = [&sense, &safety, &companions];

    // Feed the wildlife every tick. The reward `+1.0` is positive surprise
    // against `r_expected = 0` initially → drift pushes all layers' weights
    // up. COMPANIONS(player) is the most suppressed layer (starts at -3.0),
    // so its relative movement is the most dramatic.
    let mut tick = 0;
    let mut tame_tick = None;

    for t in 0..200 {
        tick = t;
        // The host supplies `r_observed = +1.0` (food eaten, positive payoff).
        kernel.drift(&layers, 1.0);

        // Check the taming threshold.
        let w_companions = kernel.w_snapshot()[2];
        if w_companions >= TAU_TAME && tame_tick.is_none() {
            tame_tick = Some(t);
        }

        // Early-exit once tamed (and a few extra ticks to show stability).
        if let Some(tt) = tame_tick
            && t >= tt + 10
        {
            break;
        }
    }

    let w = kernel.w_snapshot();
    println!("\nAfter {tick} ticks of feeding:");
    println!("  w_SENSE             = {:.4}", w[0]);
    println!("  w_SAFETY            = {:.4}", w[1]);
    println!("  w_COMPANIONS(player)= {:.4}", w[2]);

    let sigmoid_companions = katgpt_core::personality_composition::sigmoid(w[2] / config.tau);
    println!(
        "  sigmoid(w_COMPANIONS/tau) = {:.4}  (the COMPANIONS layer gate value)",
        sigmoid_companions
    );

    match tame_tick {
        Some(t) => {
            println!("\n✓ TAMED at tick {t}: w_COMPANIONS(player) crossed tau_tame = {TAU_TAME}");
            println!("  The host (riir-ai Plan 327) would now trigger the species-swap:");
            println!("    wildlife → pet (game-specific logic, not in this open primitive).");

            // Snapshot the tamed personality.
            let snap = katgpt_core::personality_composition::PersonalitySnapshot::from_composition(
                &kernel,
                ArchetypeLabel::from_str("wildlife_tamed"),
                1,
            );
            assert!(snap.verify_blake3(), "tamed snapshot must verify");
            println!("\n  Tamed-personality blake3: {}", hex_short(&snap.blake3));
        }
        None => {
            println!("\n✗ Not tamed after {tick} ticks — try increasing alpha or feeding longer.");
        }
    }

    println!("\n=== Demo complete ===");
}

fn hex_short(bytes: &[u8; 32]) -> String {
    bytes[..8].iter().map(|b| format!("{:02x}", b)).collect()
}
