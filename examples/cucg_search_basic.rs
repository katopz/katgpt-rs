//! Plan 320 Phase 7 T7.1 — CUCG SearchRubric basic example.
//!
//! Demonstrates the paper's C1/C2/C3/N1 search rubric on a synthetic
//! trajectory. Shows how a caller sources scalar features, populates the
//! scratch, and reads the Compress/Continue decision + audit record.
//!
//! Run (default features — closed_unit_compaction is now default-on):
//! ```bash
//! cargo run --example cucg_search_basic
//! ```

use katgpt_rs::compaction::rubrics::search::{SearchRubric, TrajectoryFeatures};
use katgpt_rs::compaction::{Backstop, ClosedUnitCompactionGate, FireRule, RubricScratch};

fn main() {
    println!("═══ CUCG SearchRubric Basic Example (Plan 320) ═══");
    println!();

    let rubric = SearchRubric::default();
    let gate = ClosedUnitCompactionGate::builder(rubric)
        .fire_rule(FireRule::search_rule_4()) // C1 ∧ C2 ∧ C3 ∧ ¬N1
        .backstop(Backstop::token_pct(0.30)) // safety net at 30% ctx window
        .build();

    let mut scratch = RubricScratch::with_capacity(8, 2);

    // Three probe points: warmup, mid-derivation, safe point.
    let probes: &[(&str, TrajectoryFeatures)] = &[
        (
            "warmup (low coherence)",
            TrajectoryFeatures::new(0.3, 16.0, 0.1, 4.0),
        ),
        (
            "mid-derivation (high novelty)",
            TrajectoryFeatures::new(0.8, 4.0, 1.2, 5.0),
        ),
        (
            "safe point (verified fact)",
            TrajectoryFeatures::new(0.8, 4.0, 1.2, 0.3),
        ),
    ];

    for (label, features) in probes {
        scratch.clear();
        scratch.f32_buf.extend_from_slice(&[
            features.coherence,
            features.intrinsic_rank,
            features.divergence_since_last,
            features.novelty_rate,
        ]);
        scratch.usize_buf.push(1024); // span_end

        let decision = gate.evaluate(b"trajectory prefix", 500, 4096, None, &mut scratch);
        let audit = decision.audit();

        println!("Probe: {label}");
        println!(
            "  Features: coherence={}, rank={}, div={}, novelty={}",
            features.coherence,
            features.intrinsic_rank,
            features.divergence_since_last,
            features.novelty_rate
        );
        println!(
            "  Verdict mask: 0b{:04b} (C1,C2,C3,N1)",
            audit.fire_rule_eval.yes_mask
        );
        println!(
            "  Decision: {}",
            if decision.is_compress() {
                "COMPRESS (safe to summarize)"
            } else if decision.is_continue() {
                "CONTINUE (not yet safe)"
            } else {
                "FORCED (backstop)"
            }
        );
        println!();
    }
}
