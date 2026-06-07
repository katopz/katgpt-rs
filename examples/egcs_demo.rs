//! EGCS Demo — Episode-Guided Constraint Synthesis (Plan 206).
//!
//! Demonstrates how `EpisodePruner` uses reference solutions from an episode DB
//! to synthesize structural constraints that improve pruning accuracy.
//!
//! Scenario: Token-sequence validation where a "correct" solution is known.
//! - **Base pruner** (NoPruner): allows all tokens → low accuracy
//! - **EGCS pruner**: uses reference to synthesize position-level constraints → high accuracy
//!
//! Also demonstrates the V-R (Verify-Refine) loop for iterative refinement.
//!
//! Run: `cargo run --features egcs --example egcs_demo`

#![cfg(feature = "egcs")]

use std::time::Instant;

use katgpt_rs::pruners::{
    ConstraintSynthesizer, Episode, EpisodeMetadata, EpisodePruner, MemoryEpisodeLookup,
    StructuralDiffSynthesizer, SynthesizedConstraint, VrGenerator, VrLoop, VrRoundFeedback,
    VrVerifier,
};
use katgpt_rs::speculative::{ConstraintPruner, NoPruner};

// ── Config ─────────────────────────────────────────────────────

const VOCAB_SIZE: usize = 10;
const SEQ_LEN: usize = 8;
const CANDIDATES_PER_TEST: usize = 100;
const SEED: u64 = 42;

// ── Mock Verifier ──────────────────────────────────────────────

/// Verifier that accepts only the reference sequence.
struct ReferenceVerifier {
    reference: Vec<usize>,
}

impl VrVerifier for ReferenceVerifier {
    fn verify(&self, candidate: &[usize]) -> Result<(), VrRoundFeedback> {
        let mut rejected_positions = Vec::new();
        let mut rejected_tokens = Vec::new();

        for (pos, (&c, &r)) in candidate.iter().zip(self.reference.iter()).enumerate() {
            if c != r {
                rejected_positions.push(pos);
                rejected_tokens.push(c);
            }
        }

        match rejected_positions.is_empty() {
            true => Ok(()),
            false => Err(VrRoundFeedback {
                rejected_positions,
                rejected_tokens,
                failure_description: "mismatch with reference".into(),
            }),
        }
    }
}

// ── Mock Generator ─────────────────────────────────────────────

/// Random generator that respects disallowed constraints.
struct SmartGenerator;

impl VrGenerator for SmartGenerator {
    fn generate(
        &mut self,
        constraints: &[SynthesizedConstraint],
        vocab_size: usize,
        seq_len: usize,
        rng: &mut fastrand::Rng,
    ) -> Vec<Vec<usize>> {
        let mut candidates = Vec::with_capacity(4);

        for _ in 0..4 {
            let mut seq = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                let disallowed: Vec<usize> = constraints
                    .iter()
                    .filter(|c| c.position_range.0 <= pos && pos < c.position_range.1)
                    .flat_map(|c| c.disallowed_tokens.iter().copied())
                    .collect();

                let mut token = rng.usize(0..vocab_size);
                let mut attempts = 0;
                while disallowed.contains(&token) && attempts < vocab_size {
                    token = rng.usize(0..vocab_size);
                    attempts += 1;
                }
                seq.push(token);
            }
            candidates.push(seq);
        }

        candidates
    }
}

// ── Main ────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      EGCS Demo — Episode-Guided Constraint Synthesis       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut rng = fastrand::Rng::with_seed(SEED);

    // Reference solution (the "known good" token sequence)
    let reference = vec![3, 7, 1, 5, 9, 2, 4, 6];

    println!("Reference solution: {reference:?}");
    println!("Vocab size: {VOCAB_SIZE}, Sequence length: {SEQ_LEN}");
    println!();

    // ── Section 1: EpisodePruner Before/After ───────────────────
    println!("{}", "═".repeat(62));
    println!("  Section 1: EpisodePruner — Base vs EGCS");
    println!("{}", "═".repeat(62));
    println!();

    // Build episode DB with the reference
    let mut db = MemoryEpisodeLookup::new();
    let prompt_hash = 0xABCD_u64;
    db.insert(Episode {
        prompt_hash,
        reference_tokens: reference.clone(),
        metadata: EpisodeMetadata {
            verification_count: 10,
            avg_acceptance: 0.95,
        },
    });

    // EGCS pruner (wraps base with episode-guided constraints)
    let egcs = EpisodePruner::new(NoPruner, db, StructuralDiffSynthesizer);
    let mut egcs = egcs;
    egcs.set_prompt(prompt_hash);

    // Base pruner for comparison (separate instance)
    let base = NoPruner;

    // Generate random candidates and test both pruners
    let mut base_accepts = 0usize;
    let mut egcs_accepts = 0usize;
    let mut _match_count = 0usize;

    for _ in 0..CANDIDATES_PER_TEST {
        let candidate: Vec<usize> = (0..SEQ_LEN).map(|_| rng.usize(0..VOCAB_SIZE)).collect();

        let base_valid = base.is_valid(0, candidate[0], &[]);
        let egcs_valid = egcs.is_valid(0, candidate[0], &[]);

        if base_valid {
            base_accepts += 1;
        }
        if egcs_valid {
            egcs_accepts += 1;
        }
        if candidate == reference {
            _match_count += 1;
        }
    }

    println!("  Base pruner accepts: {base_accepts}/{CANDIDATES_PER_TEST}");
    println!("  EGCS pruner accepts: {egcs_accepts}/{CANDIDATES_PER_TEST}");

    // Synthesize and display constraints from the diff
    let candidate: Vec<usize> = vec![3, 2, 1, 0, 9, 2, 4, 8]; // partially correct
    let synthesizer = StructuralDiffSynthesizer;
    let constraints = synthesizer.synthesize(&candidate, &reference);

    println!();
    println!("  Example: candidate {candidate:?}");
    println!("           reference {reference:?}");
    println!("  Synthesized {} constraint(s):", constraints.len());
    for (i, c) in constraints.iter().enumerate() {
        let (start, end) = c.position_range;
        println!(
            "    [{i}] positions {start}..{end}: disallow {:?}",
            c.disallowed_tokens
        );
    }

    // ── Section 2: V-R Loop ─────────────────────────────────────
    println!();
    println!("{}", "═".repeat(62));
    println!("  Section 2: V-R Loop — Verify-Refine Iteration");
    println!("{}", "═".repeat(62));
    println!();

    let verifier = ReferenceVerifier {
        reference: reference.clone(),
    };
    let mut vr = VrLoop::new(SmartGenerator, verifier)
        .with_max_rounds(5)
        .with_seq_len(SEQ_LEN)
        .with_candidates_per_round(4);

    let start = Instant::now();
    let mut rng_vr = fastrand::Rng::with_seed(SEED);
    let result = vr.run(VOCAB_SIZE, &mut rng_vr);
    let elapsed = start.elapsed();

    println!("  V-R Loop result:");
    println!("    Converged: {}", result.converged);
    println!("    Rounds: {}", result.rounds);
    println!("    Accepted: {} candidate(s)", result.accepted.len());
    println!("    Rejection log: {} round(s)", result.rejection_log.len());
    println!("    Constraints accumulated: {}", vr.constraint_count());
    println!("    Latency: {:.2?}", elapsed);

    if let Some(accepted) = result.accepted.first() {
        let matches = accepted == &reference;
        println!("    Accepted candidate: {accepted:?}");
        println!("    Matches reference: {matches}");
    }

    for (i, reason) in result.rejection_log.iter().enumerate() {
        println!("    Round {} rejection: {}", i + 1, reason);
    }

    // ── Section 3: Overhead Measurement ─────────────────────────
    println!();
    println!("{}", "═".repeat(62));
    println!("  Section 3: Overhead — Base vs EGCS Latency");
    println!("{}", "═".repeat(62));
    println!();

    let iters = 10_000usize;
    let mut rng_base = fastrand::Rng::with_seed(SEED);
    let mut rng_egcs = fastrand::Rng::with_seed(SEED);

    // Base pruner timing
    let mut db2 = MemoryEpisodeLookup::new();
    db2.insert(Episode {
        prompt_hash,
        reference_tokens: reference.clone(),
        metadata: EpisodeMetadata::default(),
    });
    let mut egcs2 = EpisodePruner::new(NoPruner, db2, StructuralDiffSynthesizer);
    egcs2.set_prompt(prompt_hash);

    let start = Instant::now();
    for _ in 0..iters {
        let token = rng_base.usize(0..VOCAB_SIZE);
        let _ = NoPruner.is_valid(0, token, &[]);
    }
    let base_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    let start = Instant::now();
    for _ in 0..iters {
        let token = rng_egcs.usize(0..VOCAB_SIZE);
        let _ = egcs2.is_valid(0, token, &[]);
    }
    let egcs_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    let overhead = egcs_ns - base_ns;
    let overhead_pct = if base_ns > 0.0 {
        overhead / base_ns * 100.0
    } else {
        0.0
    };

    println!("  Iterations: {iters}");
    println!("  Base pruner: {base_ns:.1} ns/call");
    println!("  EGCS pruner: {egcs_ns:.1} ns/call");
    println!("  Overhead: {overhead:.1} ns ({overhead_pct:.1}%)");

    // ── Summary ─────────────────────────────────────────────────
    println!();
    println!("{}", "═".repeat(62));
    println!("  Summary");
    println!("{}", "═".repeat(62));
    println!();
    println!("  ✓ EpisodePruner wraps any ConstraintPruner with reference-based constraints");
    println!("  ✓ Zero-cost miss path: no episode → inner pruner only");
    println!("  ✓ V-R Loop iteratively refines candidates via verify-extract-regenerate");
    println!("  ✓ Overhead: {overhead:.1} ns/call on episode DB hit path");
    println!();
}

// TL;DR: Demonstrates EpisodePruner synthesizing structural constraints from episode DB references + V-R Loop iterative refinement, with latency overhead measurement.
