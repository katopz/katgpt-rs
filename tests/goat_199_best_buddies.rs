//! GOAT Proof for Plan 199: Best Buddies Drafting
//!
//! Criteria: acceptance rate improvement ≥ 5% when best_buddies filtering is enabled.
//!
//! Methodology:
//!   1. Generate synthetic draft/target marginals over V=128 vocab, D=10 lookahead
//!   2. Mix agreement: ~60% positions have high Pearson (peak token same), ~40% differ
//!   3. For each position, "expected acceptance rate" = Σ draft[t] * target[t]
//!      (probability that a token sampled from draft is also high-prob under target)
//!   4. Compare expected acceptance of raw draft marginals vs BB-filtered marginals
//!   5. Assert: BB-filtered marginals have ≥ 5% higher expected acceptance
//!
//! Why this works: BB filtering dampens positions where draft and target disagree,
//! redistributing probability mass toward tokens the target also prefers. This directly
//! translates to higher speculative acceptance rate in production.
//!
//! ```sh
//! cargo test --features "speculative_generator,best_buddies" --test goat_199_best_buddies -- --nocapture
//! ```

#![cfg(all(feature = "speculative_generator", feature = "best_buddies"))]

use katgpt_core::traits::{BestBuddyAligner, pearson_correlation};
use katgpt_core::{Config, NoPruner};
use katgpt_rs::speculative::{
    MarginalBestBuddyAligner, MarginalTokenGenerator, TokenConstraintPruner,
    build_dd_tree_speculative, build_dd_tree_speculative_best_buddies, extract_best_path,
};

const VOCAB: usize = 128;
const DEPTH: usize = 10;
const ROUNDS: usize = 500;

/// Create a peaked distribution over `vocab` tokens with a given peak index.
/// Peak gets ~50% mass, runner-up gets ~30%, rest share the remainder.
fn peaked_marginal(vocab: usize, peak: usize, second_peak: usize) -> Vec<f32> {
    let mut m = vec![0.01f32; vocab];
    m[peak] = 0.50;
    m[second_peak % vocab] = 0.30;
    let sum: f32 = m.iter().sum();
    m.iter_mut().for_each(|v| *v /= sum);
    m
}

/// Create partially-correlated draft/target marginals for a single round.
///
/// ~60% of positions have high draft↔target agreement (same peak token).
/// ~40% have low agreement (target prefers a different token).
fn make_scenario(seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let agree_count = (DEPTH as f64 * 0.6).round() as usize;

    let mut draft = Vec::with_capacity(DEPTH);
    let mut target = Vec::with_capacity(DEPTH);

    for pos in 0..DEPTH {
        let draft_peak = (seed as usize + pos * 7) % VOCAB;
        let draft_second = (draft_peak + 3) % VOCAB;

        let dm = peaked_marginal(VOCAB, draft_peak, draft_second);
        draft.push(dm);

        if pos < agree_count {
            // High agreement: target has same peak
            let tm = peaked_marginal(VOCAB, draft_peak, (draft_peak + 1) % VOCAB);
            target.push(tm);
        } else {
            // Low agreement: target has different peak
            let target_peak = (draft_peak + VOCAB / 2) % VOCAB;
            let tm = peaked_marginal(VOCAB, target_peak, (target_peak + 5) % VOCAB);
            target.push(tm);
        }
    }

    (draft, target)
}

/// Expected acceptance rate for a position:
/// Σ_marginal[t] * target[t] = probability that a token sampled from marginal
/// is also high-probability under the target model.
fn expected_acceptance(marginal: &[f32], target: &[f32]) -> f64 {
    marginal
        .iter()
        .zip(target.iter())
        .map(|(&m, &t)| (m * t) as f64)
        .sum()
}

#[test]
fn goat_199_acceptance_rate_improvement() {
    println!("\n🧪 GOAT 199 — Best Buddies: Acceptance Rate Improvement ≥ 5%");
    println!("{}", "═".repeat(70));

    let config = Config {
        vocab_size: VOCAB,
        draft_lookahead: DEPTH,
        tree_budget: 64,
        ..Config::draft()
    };

    let mut baseline_total = 0.0f64;
    let mut bb_total = 0.0f64;
    let mut baseline_path_total = 0.0f64;
    let mut bb_path_total = 0.0f64;

    for round in 0..ROUNDS {
        let seed = round as u64 * 17 + 31;
        let (draft, target) = make_scenario(seed);

        let draft_slices: Vec<&[f32]> = draft.iter().map(|m| m.as_slice()).collect();
        let target_slices: Vec<&[f32]> = target.iter().map(|m| m.as_slice()).collect();

        // ── Marginal-level acceptance (the core metric) ──
        // Baseline: raw draft marginals vs target
        let raw_acceptance: f64 = (0..DEPTH)
            .map(|i| expected_acceptance(&draft[i], &target[i]))
            .sum();
        baseline_total += raw_acceptance;

        // BB-filtered: apply filter_marginals, then measure acceptance
        let mut aligner = MarginalBestBuddyAligner::new(0.3);
        let filtered = aligner.filter_marginals(&draft_slices, &target_slices);
        let bb_acceptance: f64 = (0..DEPTH)
            .map(|i| expected_acceptance(&filtered[i], &target[i]))
            .sum();
        bb_total += bb_acceptance;

        // ── Path-level acceptance (end-to-end DDTree) ──
        // Baseline path
        {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut sampler = MarginalTokenGenerator { top_k: 10 };
            let pruner = TokenConstraintPruner::new(NoPruner);
            let tree =
                build_dd_tree_speculative(&mut sampler, &pruner, &draft_slices, &config, &mut rng);
            let path = extract_best_path(&tree);
            let path_score: f64 = path
                .iter()
                .enumerate()
                .map(|(d, &t)| {
                    if d < target.len() && t < VOCAB {
                        target[d][t] as f64
                    } else {
                        0.0
                    }
                })
                .sum();
            baseline_path_total += path_score;
        }

        // BB-filtered path
        {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut sampler = MarginalTokenGenerator { top_k: 10 };
            let pruner = TokenConstraintPruner::new(NoPruner);
            let mut aligner2 = MarginalBestBuddyAligner::new(0.3);
            let tree = build_dd_tree_speculative_best_buddies(
                &mut sampler,
                &pruner,
                &draft_slices,
                &target_slices,
                &mut aligner2,
                &config,
                &mut rng,
            );
            let path = extract_best_path(&tree);
            let path_score: f64 = path
                .iter()
                .enumerate()
                .map(|(d, &t)| {
                    if d < target.len() && t < VOCAB {
                        target[d][t] as f64
                    } else {
                        0.0
                    }
                })
                .sum();
            bb_path_total += path_score;
        }
    }

    let baseline_mean = baseline_total / ROUNDS as f64;
    let bb_mean = bb_total / ROUNDS as f64;
    let marginal_improvement = (bb_mean - baseline_mean) / baseline_mean * 100.0;

    let baseline_path_mean = baseline_path_total / ROUNDS as f64;
    let bb_path_mean = bb_path_total / ROUNDS as f64;
    let path_improvement = (bb_path_mean - baseline_path_mean) / baseline_path_mean * 100.0;

    println!("   Rounds:                    {ROUNDS}");
    println!("   Vocab:                     {VOCAB}");
    println!("   Depth:                     {DEPTH}");
    println!("   Agreement positions:       60% (6/10)");
    println!();
    println!("   ── Marginal-Level Acceptance ──");
    println!("   Baseline (raw draft):      {baseline_mean:.6}");
    println!("   BB-filtered:               {bb_mean:.6}");
    println!("   Improvement:               {marginal_improvement:+.2}%");
    println!();
    println!("   ── Path-Level Acceptance (DDTree end-to-end) ──");
    println!("   Baseline path score:       {baseline_path_mean:.4}");
    println!("   BB path score:             {bb_path_mean:.4}");
    println!("   Improvement:               {path_improvement:+.2}%");

    // GOAT gate: marginal-level acceptance must improve by ≥5%
    assert!(
        marginal_improvement >= 5.0,
        "GOAT 199 FAIL: marginal acceptance improvement must be ≥5%, got {marginal_improvement:.2}%. \
         baseline={baseline_mean:.6}, bb={bb_mean:.6}"
    );

    println!();
    println!("   ✅ GOAT 199 PASS: acceptance rate improvement ≥5%");
}

#[test]
fn goat_199_pearson_correlation_accuracy() {
    println!("\n🧪 GOAT 199 — Pearson Correlation Sanity Check");
    println!("{}", "═".repeat(70));

    // Identical distributions → Pearson ≈ 1.0
    let a = vec![0.1f32, 0.2, 0.3, 0.4];
    let corr = pearson_correlation(&a, &a);
    assert!(
        (corr - 1.0).abs() < 1e-6,
        "identical distributions should have corr=1.0, got {corr}"
    );

    // Anti-correlated → Pearson ≈ -1.0
    let b = vec![0.4f32, 0.3, 0.2, 0.1];
    let corr = pearson_correlation(&a, &b);
    assert!(
        (corr + 1.0).abs() < 1e-6,
        "reversed distributions should have corr=-1.0, got {corr}"
    );

    // Zero correlation → Pearson ≈ 0.0
    let c = vec![1.0f32, -1.0, 1.0, -1.0];
    let d = vec![1.0f32, 1.0, -1.0, -1.0];
    let corr = pearson_correlation(&c, &d);
    assert!(
        corr.abs() < 1e-6,
        "orthogonal distributions should have corr≈0.0, got {corr}"
    );

    println!("   ✅ Pearson correlation accuracy verified (identity, anti, orthogonal)");
}

#[test]
fn goat_199_mutual_agreement_filtering() {
    println!("\n🧪 GOAT 199 — Mutual Agreement Filter Quality");
    println!("{}", "═".repeat(70));

    let mut aligner = MarginalBestBuddyAligner::new(0.3);

    // Position 0: high agreement (identical peaks)
    let draft_pos0 = peaked_marginal(VOCAB, 5, 10);
    let target_pos0 = peaked_marginal(VOCAB, 5, 8);

    // Position 1: low agreement (different peaks)
    let draft_pos1 = peaked_marginal(VOCAB, 5, 10);
    let target_pos1 = peaked_marginal(VOCAB, 60, 65);

    let draft_slices: Vec<&[f32]> = vec![&draft_pos0, &draft_pos1];
    let target_slices = vec![target_pos0.as_slice(), target_pos1.as_slice()];

    let filtered = aligner.filter_marginals(&draft_slices, &target_slices);

    assert_eq!(filtered.len(), 2, "should have 2 filtered marginals");

    // Position 0 (high agreement): filtered should preserve the peak
    let peak0_pos = filtered[0]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(
        peak0_pos, 5,
        "high-agreement position should preserve draft peak at token 5"
    );

    // Check that mutual agreement scores differ
    let score_high = aligner.mutual_agreement(&draft_pos0, &target_pos0);
    let score_low = aligner.mutual_agreement(&draft_pos1, &target_pos1);

    assert!(
        score_high > score_low,
        "high agreement ({score_high:.4}) should exceed low agreement ({score_low:.4})"
    );

    // Verify acceptance improvement at marginal level
    let raw_acc_pos0 = expected_acceptance(&draft_pos0, &target_pos0);
    let filt_acc_pos0 = expected_acceptance(&filtered[0], &target_pos0);
    let raw_acc_pos1 = expected_acceptance(&draft_pos1, &target_pos1);
    let filt_acc_pos1 = expected_acceptance(&filtered[1], &target_pos1);

    println!("   High agreement score: {score_high:.4}");
    println!("   Low agreement score:  {score_low:.4}");
    println!("   Peak preserved at high-agreement position: token {peak0_pos}");
    println!();
    println!(
        "   Position 0 (high agree) acceptance: raw={raw_acc_pos0:.6}, filtered={filt_acc_pos0:.6}"
    );
    println!(
        "   Position 1 (low agree)  acceptance: raw={raw_acc_pos1:.6}, filtered={filt_acc_pos1:.6}"
    );

    // BB filtering should not degrade high-agreement positions
    assert!(
        filt_acc_pos0 >= raw_acc_pos0 * 0.99,
        "high-agreement position should not be degraded: {filt_acc_pos0:.6} < {raw_acc_pos0:.6}"
    );

    println!("   ✅ Mutual agreement filter correctly distinguishes agreement levels");
}

// TL;DR: GOAT proof for Plan 199 Best Buddies. Measures acceptance rate delta
// between raw draft marginals and BB-filtered marginals against target distributions.
// Passes when BB filtering improves expected acceptance by ≥5% over 500 rounds.
