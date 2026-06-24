//! Plan 320 Phase 7 T7.2 — CUCG ShardFreezeRubric isomorphism example.
//!
//! Demonstrates the G7 Super-GOAT claim: the CUCG shard-freeze rubric
//! produces the same decisions as riir-neuron-db's `can_freeze` gate,
//! because they are the same Boolean formula:
//!
//!   can_freeze = (n_wake_events >= intrinsic_dim) && (spectral_flatness < 0.3)
//!             = P0 && P1
//!
//! Run (default features):
//! ```bash
//! cargo run --example cucg_shard_freeze_isomorphism
//! ```

use katgpt_rs::compaction::rubrics::shard_freeze::{
    SHARD_FREEZE_FLATNESS_THRESHOLD, ShardFreezeFeatures, ShardFreezeRubric,
};
use katgpt_rs::compaction::{Backstop, ClosedUnitCompactionGate, FireRule, RubricScratch};

fn main() {
    println!("═══ CUCG × can_freeze Isomorphism (G7, Plan 320) ═══");
    println!();
    println!(
        "can_freeze = (n >= d) && (flatness < {})",
        SHARD_FREEZE_FLATNESS_THRESHOLD
    );
    println!();

    let gate = ClosedUnitCompactionGate::builder(ShardFreezeRubric::new())
        .fire_rule(FireRule::shard_freeze_rule_2()) // P0 ∧ P1
        .backstop(Backstop::None)
        .build();

    let mut scratch = RubricScratch::with_capacity(4, 4);

    let cases: &[(&str, ShardFreezeFeatures)] = &[
        (
            "well-sampled + converged",
            ShardFreezeFeatures::new(12, 8, 0.10),
        ),
        (
            "well-sampled + not converged",
            ShardFreezeFeatures::new(12, 8, 0.50),
        ),
        (
            "under-sampled + converged",
            ShardFreezeFeatures::new(3, 8, 0.10),
        ),
        (
            "under-sampled + not converged",
            ShardFreezeFeatures::new(3, 8, 0.50),
        ),
    ];

    for (label, f) in cases {
        scratch.clear();
        scratch.usize_buf.push(f.n_wake_events);
        scratch.usize_buf.push(f.intrinsic_dim);
        scratch.f32_buf.push(f.spectral_flatness);

        let decision = gate.evaluate(b"shard", 0, 1_000_000, None, &mut scratch);

        // Compute can_freeze the "riir-neuron-db way" for comparison.
        let can_freeze = f.n_wake_events >= f.intrinsic_dim
            && f.spectral_flatness < SHARD_FREEZE_FLATNESS_THRESHOLD;

        let cucg_freeze = decision.is_compress();
        let match_str = if cucg_freeze == can_freeze {
            "✓ MATCH"
        } else {
            "✗ MISMATCH"
        };

        println!("{label}:");
        println!(
            "  n={}, d={}, flatness={}",
            f.n_wake_events, f.intrinsic_dim, f.spectral_flatness
        );
        println!("  can_freeze formula: {can_freeze}");
        println!(
            "  CUCG decision:      {} ({})",
            cucg_freeze,
            if cucg_freeze {
                "Compress/freeze"
            } else {
                "Continue/no-freeze"
            }
        );
        println!("  {match_str}");
        println!();
    }

    println!("The two gates are the same primitive — trajectory compaction");
    println!("and shard consolidation freeze are instances of one rubric-gated");
    println!("structural-safety check.");
}
