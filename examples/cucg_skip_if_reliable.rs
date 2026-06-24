//! Plan 320 Phase 7 T7.3 — CUCG skip-if-reliable suppression example.
//!
//! Demonstrates G2: the skip-if-reliable fuse (paper §4.1 skip-if-correct
//! oracle, modelless via CLR). When the CLR reliability vote exceeds the
//! threshold, a would-be Compress decision is suppressed to Continue —
//! the current context is reliable enough that compaction can be deferred.
//!
//! Run (default features):
//! ```bash
//! cargo run --example cucg_skip_if_reliable
//! ```

use katgpt_rs::compaction::rubrics::search::SearchRubric;
use katgpt_rs::compaction::{Backstop, ClosedUnitCompactionGate, FireRule, RubricScratch};

fn main() {
    println!("═══ CUCG skip-if-reliable Suppression (G2, Plan 320) ═══");
    println!();

    // Two gates: one without the skip fuse, one with.
    let gate_plain = ClosedUnitCompactionGate::builder(SearchRubric::default())
        .fire_rule(FireRule::search_rule_4())
        .backstop(Backstop::None)
        .build();

    let gate_skip = ClosedUnitCompactionGate::builder(SearchRubric::default())
        .fire_rule(FireRule::search_rule_4())
        .backstop(Backstop::None)
        .skip_if_reliable(0.8) // suppress Compress when CLR vote > 0.8
        .build();

    let mut scratch = RubricScratch::with_capacity(8, 2);

    // A safe-point trajectory (rubric would fire Compress).
    let safe_point_features = [0.8_f32, 4.0, 1.2, 0.3]; // coherence, rank, div, novelty

    let clr_votes: &[(&str, f32)] = &[
        ("no CLR vote", 0.0),    // None path — not suppressed
        ("low reliability", 0.5), // below 0.8 — not suppressed
        ("high reliability", 0.9), // above 0.8 — SUPPRESSED
        ("very high reliability", 0.99),
    ];

    for (label, vote) in clr_votes {
        scratch.clear();
        scratch.f32_buf.extend_from_slice(&safe_point_features);

        // Plain gate (no skip fuse).
        let d_plain = gate_plain.evaluate(b"traj", 0, 10_000, Some(*vote), &mut scratch);

        // Skip-fuse gate.
        scratch.clear();
        scratch.f32_buf.extend_from_slice(&safe_point_features);
        let d_skip = gate_skip.evaluate(b"traj", 0, 10_000, Some(*vote), &mut scratch);

        let plain_str = if d_plain.is_compress() { "Compress" } else { "Continue" };
        let skip_str = if d_skip.is_compress() { "Compress" } else { "Continue (suppressed)" };
        let suppressed = d_plain.is_compress() && !d_skip.is_compress();

        println!("CLR vote = {} ({label}):", vote);
        println!("  plain gate:  {plain_str}");
        println!("  skip gate:   {skip_str}");
        if suppressed {
            println!("  → SUPPRESSED (context is reliable, defer compaction)");
        }
        println!();
    }
}
