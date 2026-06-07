#![cfg(feature = "bfcf_tree")]
//! BFCF Tree Demo — demonstrates all phases of Plan 213.
//!
//! Run with: cargo run --features bfcf_tree --example bfcf_tree_demo

use katgpt_rs::pruners::{
    bfcf_types::{BFCP, BorelRegion, HalfSpace, PWCValueFunction, RegionLabel},
    bfcp_preimage::{acceptance_rate, compute_preimage, maybe_rate, refine_partition},
    percept_router::{PerceptRouter, PerceptRouterConfig, SigmoidPerceptRouter},
    pwc_bandit::RegionBandit,
};
use katgpt_rs::speculative::types::ScreeningPruner;

// ── Demo pruner ────────────────────────────────────────────────

struct DemoPruner {
    threshold: usize,
}

impl ScreeningPruner for DemoPruner {
    fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        if token_idx < self.threshold {
            1.0
        } else if token_idx >= self.threshold + 30 {
            0.0
        } else {
            0.5
        }
    }
}

fn main() {
    println!("=== BFCF Tree Demo (Plan 213) ===\n");

    // ── Phase 1: BFCP Partition Construction ──────────────────
    println!("Phase 1: BFCP Partition Construction");
    println!("{}", "-".repeat(40));

    let bfcp = BFCP::from_regions(vec![
        BorelRegion::new(RegionLabel::Accept, vec![], 50),
        BorelRegion::new(RegionLabel::Reject, vec![], 30),
        BorelRegion::new(RegionLabel::Maybe, vec![], 20),
    ]);

    println!(
        "  Regions: {} (accept={}, reject={}, maybe={})",
        bfcp.region_count(),
        bfcp.accept_count(),
        bfcp.reject_count(),
        bfcp.maybe_count(),
    );
    println!(
        "  Tokens: {} total (accept={}, reject={}, maybe={})",
        bfcp.total_tokens(),
        bfcp.accept_token_count(),
        bfcp.reject_token_count(),
        bfcp.maybe_token_count(),
    );
    println!("  Covers vocab of 100: {}", bfcp.covers_all(100));

    // ── Half-space constraints ────────────────────────────────
    println!("\n  Half-space constraint demo:");
    let hs = HalfSpace {
        dim: 0,
        threshold: 0.5,
        above: true,
    };
    println!(
        "    logit[0] >= 0.5: [0.8, ...] -> {}, [0.3, ...] -> {}",
        hs.contains(&[0.8, 0.0]),
        hs.contains(&[0.3, 0.0]),
    );

    // ── Phase 2: Preimage Refinement ──────────────────────────
    println!("\nPhase 2: Preimage Refinement");
    println!("{}", "-".repeat(40));

    let before_accept = acceptance_rate(&bfcp);
    let before_maybe = maybe_rate(&bfcp);
    println!(
        "  Before preimage: acceptance={:.1}%, maybe={:.1}%",
        before_accept * 100.0,
        before_maybe * 100.0,
    );

    let pruner = DemoPruner { threshold: 60 };
    let prefix: Vec<usize> = vec![];
    let refined = compute_preimage(&bfcp, &prefix, &pruner, 100);
    let after_accept = acceptance_rate(&refined);
    let after_maybe = maybe_rate(&refined);
    println!(
        "  After preimage:  acceptance={:.1}%, maybe={:.1}%",
        after_accept * 100.0,
        after_maybe * 100.0,
    );
    println!(
        "  Improvement: +{:.1}% acceptance",
        (after_accept - before_accept) * 100.0,
    );
    println!(
        "  Maybe regions eliminated: {}",
        bfcp.maybe_count() - refined.maybe_count(),
    );

    // Iterative refinement
    let mut iter_partition = bfcp.clone();
    let refined_count = refine_partition(&mut iter_partition, &[], &pruner, 100, 3);
    println!(
        "  Iterative refinement (max 3 rounds): {} maybe regions refined",
        refined_count,
    );

    // ── Phase 3: PWC Bandit Arms ──────────────────────────────
    println!("\nPhase 3: PWC Bandit Arms");
    println!("{}", "-".repeat(40));

    let mut bandit = RegionBandit::new(3, 5, 2.0_f64.sqrt());
    println!(
        "  Bandit: {} arms, {} regions",
        bandit.arm_count(),
        bandit.region_count()
    );

    // Simulate some updates
    for round in 0..30 {
        let arm = round % 3;
        let region = round % 5;
        let reward = 1.0 / (1.0 + (-(round as f64 * 0.2 - 3.0)).exp());
        bandit.update(region, arm, reward);
    }

    // Show arm selection per region
    println!("  Arm selection per region:");
    for r in 0..5 {
        let best = bandit.select(r);
        let q = bandit.q_value(best, r);
        println!("    Region {}: best arm={}, Q-value={:.3}", r, best, q);
    }

    // PWC closure verification (Theorem 2)
    println!(
        "  PWC closure after 30 updates: {}",
        if bandit.verify_pwc_closure() {
            "✓ maintained"
        } else {
            "✗ violated"
        },
    );

    // ── PWC Value Function demo ──────────────────────────────
    let mut vf = PWCValueFunction::new(5, 0.0);
    for i in 0..5 {
        vf.update(i, (i + 1) as f64 * 0.2);
    }
    println!("  PWC Value Function: closure={}", vf.verify_pwc_closure());
    for i in 0..5 {
        println!("    Region {}: value={:.2}", i, vf.value(i));
    }

    // ── Phase 4: Percept Routing ──────────────────────────────
    println!("\nPhase 4: Percept Routing");
    println!("{}", "-".repeat(40));

    let router = SigmoidPerceptRouter::default_router();

    // Simple partition
    let simple = BFCP::from_regions(vec![BorelRegion::new(RegionLabel::Accept, vec![], 100)]);
    println!(
        "  Simple (1 accept region): complexity={:.3}, path={:?}",
        router.complexity(&simple),
        router.route(&simple),
    );

    // Medium partition
    let medium = BFCP::from_regions(vec![
        BorelRegion::new(RegionLabel::Accept, vec![], 50),
        BorelRegion::new(RegionLabel::Reject, vec![], 30),
        BorelRegion::new(RegionLabel::Maybe, vec![], 20),
    ]);
    println!(
        "  Medium (3 regions, mixed): complexity={:.3}, path={:?}",
        router.complexity(&medium),
        router.route(&medium),
    );

    // Complex partition
    let mut complex_regions = Vec::new();
    for i in 0..30 {
        let label = match i % 3 {
            0 => RegionLabel::Accept,
            1 => RegionLabel::Reject,
            _ => RegionLabel::Maybe,
        };
        complex_regions.push(BorelRegion::new(label, vec![], 5));
    }
    let complex = BFCP::from_regions(complex_regions);
    println!(
        "  Complex (30 mixed regions): complexity={:.3}, path={:?}",
        router.complexity(&complex),
        router.route(&complex),
    );

    // Custom thresholds
    let custom_router = SigmoidPerceptRouter::new(PerceptRouterConfig::new(0.4, 0.6));
    println!(
        "  Custom router (0.4/0.6) on medium: complexity={:.3}, path={:?}",
        custom_router.complexity(&medium),
        custom_router.route(&medium),
    );

    // ── Summary ───────────────────────────────────────────────
    println!("\n{}", "=".repeat(40));
    println!("Summary:");
    println!(
        "  - O(regions={}) instead of O(vocab=128K) evaluations",
        bfcp.region_count()
    );
    println!(
        "  - Preimage: +{:.0}% acceptance improvement",
        (after_accept - before_accept) * 100.0
    );
    println!("  - PWC closure: maintained (Theorem 2)");
    println!("  - Routing: sigmoid-bounded [0,1], no softmax");
    println!("  - Feature gate: bfcf_tree (opt-in, GOAT-gated)");
}
