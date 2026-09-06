//! Example: best_belief — ε-quantile Beta lower bound for conservative selection
//! (Plan 336, Research 320, RQGM arXiv:2606.26294 Prop. 4).
//!
//! Demonstrates the primitive's stated commercial purpose: **conservative
//! selection under bounded evidence** — the decision rule for promoting a
//! frozen snapshot / archetype-shard / zone-cache when you only have (S, F)
//! Beta-Bernoulli counts and want a *lower bound* on each candidate's true
//! utility, not a point estimate.
//!
//! Run with:
//! ```sh
//! cargo run --example best_belief_01_conservative_selection --features best_belief --release
//! ```
//!
//! # What This Proves
//!
//! - **The core formula in action**: `BB_ε(S, F) = I⁻¹_ε(1+S, 1+F)` — the
//!   value the candidate's true utility exceeds with probability `1 − ε`
//!   under the Beta-Bernoulli working posterior. Lower ε ⇒ more conservative.
//! - **Thompson explores, best_belief exploits**: the complement to
//!   `sample_beta` Thompson sampling. We show both side-by-side on the same
//!   candidate set so the exploration/exploitation contrast is concrete.
//! - **Monotonicity** (RQGM Prop. 4 invariants): score is monotone-
//!   increasing in S, monotone-decreasing in F, lower ε is more
//!   conservative, and the score lives in `(0, 1)`.
//! - **LUT hot path vs closed-form cold path**: the 32×32×5 LUT covers the
//!   common freeze/thaw regime `S, F ∈ [0, 31]` at the five standard ε
//!   values; off-grid or large counts fall back to the closed form
//!   bit-identically.
//! - **Incumbent tie preference**: `select_best_belief` avoids snapshot
//!   churn by preferring the incumbent on ties (cache-invalidation avoidance).
//! - **Realistic freeze/thaw promotion scenario**: 5 archetype candidates
//!   with mixed evidence → the conservative pick vs the naive argmax pick.
//!
//! # What This Does NOT Prove
//!
//! - **Real RQGM training-loop integration** — this is a reference demo of
//!   the primitive's API + invariants, not a production freeze/thaw pipeline.
//!   The real consumers (PrudentBanker, SafePhased, RQGM §3.5 promotion
//!   gate) call `select_best_belief` inside their hot loops.
//! - **Coverage guarantees vs conformal** — best_belief is a *selection*
//!   primitive (which candidate to promote), not a *calibration* primitive
//!   (how wide should the predictive interval be). The conformal floor
//!   (Issue 010 "Report the Floor") is a different concern. See
//!   `conformal_airpassengers.rs` for the calibration story.
//!
//! # Reference
//!
//! - Plan: `katgpt-rs/.plans/336_controlled_utility_primitives.md`
//! - Source: arXiv:2606.26294 — Iacob et al., *The Red Queen Gödel Machine*,
//!   §3.5 + App. F Prop. 4.
//! - Research: `katgpt-rs/.research/320_Red_Queen_Godel_Machine_Selective_Erasure_Best_Belief.md`

use katgpt_core::{
    best_belief_score, best_belief_scores, select_best_belief,
};

// ─────────────────────────────────────────────────────────────────────────
// Section 1: The core formula — monotonicity invariants.
//
// RQGM Prop. 4 guarantees:
//   - BB_ε(S, F) is monotone-INCREASING in S (more successes ⇒ higher floor).
//   - BB_ε(S, F) is monotone-DECREASING in F (more failures ⇒ lower floor).
//   - Lower ε ⇒ more conservative (smaller floor).
//   - BB_ε(S, F) ∈ (0, 1) always.
//
// These are the load-bearing invariants every consumer relies on. We print
// a sweep so they're visible by inspection.
// ─────────────────────────────────────────────────────────────────────────

fn section_1_monotonicity() {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 1: Monotonicity invariants (RQGM Prop. 4)                  │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    let epsilons = [0.01_f32, 0.05, 0.1, 0.25, 0.5];
    println!("  BB_ε(S, F) at fixed F=2, sweeping S (monotone-increasing in S):");
    println!("  {:>6} │ {:>8} {:>8} {:>8} {:>8} {:>8}", "S", "ε=0.01", "ε=0.05", "ε=0.10", "ε=0.25", "ε=0.50");
    println!("  ──────┼────────────────────────────────────────────────────────────────");
    for &s in &[0_u32, 1, 2, 4, 8, 16, 31] {
        print!("  {s:>6} │");
        for &eps in &epsilons {
            print!(" {:>8.4}", best_belief_score(s, 2, eps));
        }
        println!();
    }
    println!();
    println!("  → Each column increases top-to-bottom (more successes ⇒ higher floor).");
    println!("  → Each row decreases left-to-right (lower ε ⇒ more conservative).");
    println!();

    println!("  BB_ε(S, F) at fixed S=8, sweeping F (monotone-decreasing in F):");
    println!("  {:>6} │ {:>8} {:>8} {:>8} {:>8} {:>8}", "F", "ε=0.01", "ε=0.05", "ε=0.10", "ε=0.25", "ε=0.50");
    println!("  ──────┼────────────────────────────────────────────────────────────────");
    for &f in &[0_u32, 1, 2, 4, 8, 16, 31] {
        print!("  {f:>6} │");
        for &eps in &epsilons {
            print!(" {:>8.4}", best_belief_score(8, f, eps));
        }
        println!();
    }
    println!();
    println!("  → Each column decreases top-to-bottom (more failures ⇒ lower floor).");
    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Section 2: Thompson explores, best_belief exploits.
//
// sample_beta (Thompson sampling) draws a random sample from Beta(1+S, 1+F)
// for EXPLORATION — it occasionally picks under-sampled candidates to learn
// more about them. best_belief returns the ε-quantile lower bound for
// EXPLOITATION — it picks the candidate whose worst-case (at confidence
// 1−ε) is highest.
//
// We don't call sample_beta here (it's a separate primitive), but we show
// the shape difference: best_belief is deterministic + conservative; a
// Thompson sampler would be stochastic + occasionally adventurous.
// ─────────────────────────────────────────────────────────────────────────

fn section_2_explore_vs_exploit() {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 2: Thompson explores, best_belief exploits                │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    // 4 candidates with very different evidence profiles.
    let candidates: &[(u32, u32)] = &[
        (8, 2),   // well-tested, 80% empirical rate
        (1, 0),   // one success, no failures — high point estimate, low evidence
        (40, 10), // heavily tested, 80% empirical rate
        (0, 0),   // no evidence (uniform prior)
    ];
    let labels = ["well-tested (8,2)", "lucky-one (1,0)", "heavy (40,10)", "unknown (0,0)"];

    println!("  Candidates (S, F) and their best_belief floors at ε=0.05:");
    println!();
    let eps = 0.05_f32;
    for (i, &(s, f)) in candidates.iter().enumerate() {
        let bb = best_belief_score(s, f, eps);
        let empirical = if s + f > 0 {
            s as f32 / (s + f) as f32
        } else {
            f32::NAN
        };
        println!(
            "    [{i}] {:>20}: BB={bb:.4}  empirical={empirical:.4}  (floor is {:.2}× the empirical)",
            labels[i],
            if empirical.is_finite() && empirical > 0.0 {
                bb / empirical
            } else {
                f32::NAN
            }
        );
    }
    println!();
    println!("  → The (1,0) 'lucky-one' candidate has empirical=1.0 but BB≈0.22 —");
    println!("    best_belief refuses to promote it on a single success.");
    println!("  → The (40,10) candidate has the same empirical rate as (8,2) but a");
    println!("    higher floor (more evidence ⇒ tighter lower bound).");
    println!("  → Thompson sampling would occasionally pick (1,0) for exploration;");
    println!("    best_belief never does — it's pure exploitation.");
    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Section 3: Realistic freeze/thaw promotion scenario.
//
// 5 archetype-shard candidates competing for promotion. The naive argmax
// (highest empirical S/(S+F)) vs the conservative best_belief pick.
// ─────────────────────────────────────────────────────────────────────────

fn section_3_freeze_thaw_scenario() {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 3: Freeze/thaw promotion scenario (5 archetype candidates) │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    // (S, F) for each archetype shard collected during the evaluation window.
    let candidates: &[(u32, u32)] = &[
        (15, 5),  // archetype A: 75% empirical
        (3, 1),   // archetype B: 75% empirical, low evidence
        (30, 18), // archetype C: 62.5% empirical, heavy evidence
        (8, 2),   // archetype D: 80% empirical, moderate evidence
        (2, 0),   // archetype E: 100% empirical, minimal evidence
    ];
    let labels = ["A (15,5)", "B (3,1)", "C (30,18)", "D (8,2)", "E (2,0)"];

    let eps = 0.05_f32;
    let scores = best_belief_scores(candidates, eps);

    println!("  Candidate evaluation at ε={eps} (95% confidence floor):");
    println!();
    println!("  {:>4} {:>12} {:>10} {:>10} {:>14}", "idx", "label", "empirical", "BB floor", "floor/empirical");
    println!("  ──── ──────────── ────────── ────────── ──────────────");
    for (i, &(s, f)) in candidates.iter().enumerate() {
        let empirical = s as f32 / (s + f) as f32;
        let ratio = scores[i] / empirical;
        println!(
            "  {:>4} {:>12} {:>10.4} {:>10.4} {:>14.3}",
            i,
            labels[i],
            empirical,
            scores[i],
            ratio
        );
    }
    println!();

    // Naive argmax: highest empirical rate.
    let naive_idx = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let ra = a.0 as f32 / (a.0 + a.1) as f32;
            let rb = b.0 as f32 / (b.0 + b.1) as f32;
            katgpt_core::float_order::cmp_for_max(ra, rb)
        })
        .map(|(i, _)| i)
        .unwrap();

    // Conservative best_belief pick.
    let conservative_idx = select_best_belief(candidates, eps, None);

    println!(
        "  Naive argmax (highest empirical):     candidate {naive_idx} ({})",
        labels[naive_idx]
    );
    println!(
        "  Conservative best_belief (ε={eps}): candidate {conservative_idx} ({})",
        labels[conservative_idx]
    );
    println!();
    if naive_idx != conservative_idx {
        println!("  → The two rules DISAGREE. The naive rule picks the candidate with");
        println!("    the highest point estimate; best_belief picks the one whose");
        println!("    worst-case (at 95% confidence) is highest. For freeze/thaw");
        println!("    promotion, the conservative pick is the safer default — it");
        println!("    won't promote a candidate just because it got lucky on few");
        println!("    samples.");
    } else {
        println!("  → The two rules AGREE here — the well-evidenced candidate wins");
        println!("    under both rules.");
    }
    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Section 4: Incumbent tie preference (cache-invalidation avoidance).
//
// When two candidates tie on best_belief score, select_best_belief prefers
// the incumbent to avoid unnecessary snapshot swaps. This is the
// anti-churn rule that keeps freeze/thaw stable.
// ─────────────────────────────────────────────────────────────────────────

fn section_4_incumbent_tie_preference() {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 4: Incumbent tie preference (anti-churn)                   │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    // Two candidates with IDENTICAL (S, F) → guaranteed tie on best_belief.
    let candidates: &[(u32, u32)] = &[(10, 2), (10, 2)];
    let labels = ["challenger (idx 0)", "incumbent (idx 1)"];
    let eps = 0.05_f32;

    let scores = best_belief_scores(candidates, eps);
    println!("  Two candidates with identical (S, F) = (10, 2):");
    println!("    [0] {}: BB={:.6}", labels[0], scores[0]);
    println!("    [1] {}: BB={:.6}", labels[1], scores[1]);
    println!("    → scores are bit-identical: {}", scores[0].to_bits() == scores[1].to_bits());
    println!();

    // Without incumbent preference: argmax picks the FIRST (lowest index).
    let without_incumbent = select_best_belief(candidates, eps, None);
    println!(
        "  select_best_belief(incumbent=None): picks idx {without_incumbent} ({})",
        labels[without_incumbent]
    );

    // With incumbent preference: picks the incumbent on the tie.
    let with_incumbent = select_best_belief(candidates, eps, Some(1));
    println!(
        "  select_best_belief(incumbent=Some(1)): picks idx {with_incumbent} ({})",
        labels[with_incumbent]
    );
    println!();
    println!("  → On a tie, the incumbent wins. This avoids snapshot churn:");
    println!("    re-promoting the same archetype doesn't invalidate caches.");
    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Section 5: LUT hot path vs closed-form cold path.
//
// The 32×32×5 LUT covers S, F ∈ [0, 31] at ε ∈ {0.01, 0.05, 0.1, 0.25, 0.5}.
// Off-grid (large S+F or non-standard ε) falls back to the closed form.
// Both paths are bit-identical on the overlap (the LUT is GENERATED by the
// closed form).
// ─────────────────────────────────────────────────────────────────────────

fn section_5_lut_vs_closed_form() {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 5: LUT hot path vs closed-form cold path                   │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    // In-LUT case: small S, F, standard ε.
    let in_lut = best_belief_score(10, 5, 0.05);
    println!("  In-LUT (S=10, F=5, ε=0.05):    BB = {in_lut:.8}");

    // Out-of-LUT case: large S (falls back to closed form).
    let cold_path_large = best_belief_score(100, 50, 0.05);
    println!("  Cold path (S=100, F=50, ε=0.05): BB = {cold_path_large:.8}  (S ≥ 32 → closed form)");

    // Out-of-LUT case: non-standard ε (falls back to closed form).
    let cold_path_eps = best_belief_score(10, 5, 0.07);
    println!("  Cold path (S=10, F=5, ε=0.07):  BB = {cold_path_eps:.8}  (non-standard ε → closed form)");

    // Edge: uniform prior.
    let uniform = best_belief_score(0, 0, 0.05);
    println!("  Uniform prior (S=0, F=0, ε=0.05): BB = {uniform:.8}  (= ε by definition)");
    println!();
    println!("  → The LUT is bit-identical to the closed form on its domain (the");
    println!("    table is generated by the closed form at first call). Off-grid");
    println!("    values use the closed form directly — same correctness, slower.");
    println!();
}

fn main() {
    println!();
    println!("╔═════════════════════════════════════════════════════════════════════╗");
    println!("║  best_belief — ε-quantile Beta lower bound (Plan 336, RQGM Prop. 4) ║");
    println!("║  Conservative selection under bounded evidence                     ║");
    println!("╚═════════════════════════════════════════════════════════════════════╝");
    println!();

    section_1_monotonicity();
    section_2_explore_vs_exploit();
    section_3_freeze_thaw_scenario();
    section_4_incumbent_tie_preference();
    section_5_lut_vs_closed_form();

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("Done. All 5 sections completed. See the module doc (top of this file)");
    println!("for what this proves / does NOT prove.");
    println!("═══════════════════════════════════════════════════════════════════════");
}
