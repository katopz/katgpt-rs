//! Plan 221 Example: Sense Composition + GM Override Demo

use katgpt_core::sense::brain::NpcBrain;
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::types::SenseKind;

fn main() {
    println!("=== Plan 221: Sense Composition + GM Override Demo ===\n");

    // Build sense modules
    let builder = SenseOctreeBuilder::new(3);

    let fighter_emb = KgEmbedding {
        entity_hash: 1,
        relation_hash: 1,
        embedding: [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        sign: true,
    };
    let spatial_emb = KgEmbedding {
        entity_hash: 2,
        relation_hash: 2,
        embedding: [0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        sign: true,
    };
    let social_emb = KgEmbedding {
        entity_hash: 3,
        relation_hash: 3,
        embedding: [0.1, 0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
        sign: true,
    };

    let fighter = builder.build(SenseKind::FighterSense, &[fighter_emb]);
    let spatial = builder.build(SenseKind::SpatialSense, &[spatial_emb]);
    let social = builder.build(SenseKind::SocialSense, &[social_emb]);

    // Create brain with 3 modules
    let mut brain = NpcBrain::compose(vec![fighter, spatial, social]);
    brain.hla_state = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Autonomous projection
    println!("--- Autonomous Mode ---");
    let results = brain.project_all();
    for (i, val) in results.iter().enumerate() {
        println!("  Module {}: activation = {:.4}", i, val);
    }

    // GM override: pin fighter_sense to 0.9
    println!("\n--- GM Override: Pin FighterSense = 0.9 ---");
    brain.pin_sense(SenseKind::FighterSense, 0.9);
    let results = brain.project_all();
    println!(
        "  FighterSense: {:.4} (pinned)",
        results.get(0).unwrap_or(&0.0)
    );
    for (i, val) in results.iter().enumerate() {
        println!("  Module {}: activation = {:.4}", i, val);
    }

    // Scripted mode
    println!("\n--- Scripted Mode ---");
    brain.pin_sense(SenseKind::SpatialSense, 0.5);
    brain.disable_autonomous(42);
    let results = brain.project_all();
    println!("  Autonomous disabled, scripted ID = 42");
    for (i, val) in results.iter().enumerate() {
        println!("  Module {}: activation = {:.4}", i, val);
    }

    // Re-enable
    brain.enable_autonomous();
    println!("\n--- Autonomous Restored ---");
    let results = brain.project_all();
    for (i, val) in results.iter().enumerate() {
        println!("  Module {}: activation = {:.4}", i, val);
    }

    println!("\nDone.");
}
