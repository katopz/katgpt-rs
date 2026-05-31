#![cfg(feature = "sr2am_configurator")]
//! SR²AM Configurator Bandit Tests (Plan 112, Research 076)
//!
//! Tests for the bandit-based configurator that learns per-turn planning decisions
//! (PlanNew/PlanExtend/PlanSkip) for the DDTree speculative decoding path.
//!
//! Run: `cargo test --features sr2am_configurator --test test_sr2am_configurator.goat -- --nocapture`

use katgpt_core::{ConfiguratorContext, PlanningDecision};
use katgpt_rs::pruners::ConfiguratorBandit;
use katgpt_rs::speculative::entropy_truncate_horizon;

// ── Helpers ───────────────────────────────────────────────────

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

// ── PlanningDecision Variants ─────────────────────────────────

#[test]
fn test_planning_decision_variants() {
    // Verify all three variants exist and are distinct
    let decisions = [
        PlanningDecision::PlanNew,
        PlanningDecision::PlanExtend,
        PlanningDecision::PlanSkip,
    ];

    // Each variant should be unique
    for (i, a) in decisions.iter().enumerate() {
        for (j, b) in decisions.iter().enumerate() {
            match i == j {
                true => assert_eq!(a, b),
                false => assert_ne!(a, b),
            }
        }
    }
}

#[test]
fn test_planning_decision_copy_clone() {
    let d = PlanningDecision::PlanNew;
    let d2 = d; // Copy
    assert_eq!(d, d2);

    let d3 = d;
    assert_eq!(d, d3);
}

#[test]
fn test_planning_decision_debug_format() {
    assert!(format!("{:?}", PlanningDecision::PlanNew).contains("PlanNew"));
    assert!(format!("{:?}", PlanningDecision::PlanExtend).contains("PlanExtend"));
    assert!(format!("{:?}", PlanningDecision::PlanSkip).contains("PlanSkip"));
}

// ── ConfiguratorContext ───────────────────────────────────────

#[test]
fn test_configurator_context_fields() {
    let ctx = ConfiguratorContext::new(3, 7);
    assert_eq!(ctx.domain, 3);
    assert_eq!(ctx.entropy_bin, 7);
}

#[test]
fn test_configurator_context_equality() {
    let ctx1 = ConfiguratorContext::new(0, 5);
    let ctx2 = ConfiguratorContext::new(0, 5);
    let ctx3 = ConfiguratorContext::new(1, 5);

    assert_eq!(ctx1, ctx2);
    assert_ne!(ctx1, ctx3);
}

#[test]
fn test_configurator_context_hashable() {
    use std::collections::HashMap;

    let ctx = ConfiguratorContext::new(0, 3);
    let mut map = HashMap::new();
    map.insert(ctx, PlanningDecision::PlanSkip);
    assert_eq!(map.get(&ctx), Some(&PlanningDecision::PlanSkip));
}

// ── Entropy Binning ───────────────────────────────────────────

#[test]
fn test_entropy_bin_boundaries() {
    assert_eq!(ConfiguratorBandit::entropy_bin(0.0), 0);
    assert_eq!(ConfiguratorBandit::entropy_bin(0.05), 0);
    assert_eq!(ConfiguratorBandit::entropy_bin(0.1), 1);
    assert_eq!(ConfiguratorBandit::entropy_bin(0.15), 1);
    assert_eq!(ConfiguratorBandit::entropy_bin(0.5), 5);
    assert_eq!(ConfiguratorBandit::entropy_bin(0.99), 9);
    // Clamped to max bin
    assert_eq!(ConfiguratorBandit::entropy_bin(1.0), 9);
    assert_eq!(ConfiguratorBandit::entropy_bin(2.5), 9);
    assert_eq!(ConfiguratorBandit::entropy_bin(10.0), 9);
}

#[test]
fn test_entropy_bin_monotonic_within_range() {
    let mut prev_bin = 0;
    for i in 0..20 {
        let entropy = i as f32 * 0.05;
        let bin = ConfiguratorBandit::entropy_bin(entropy);
        assert!(
            bin >= prev_bin,
            "bin should be non-decreasing: entropy={entropy}, bin={bin}, prev_bin={prev_bin}"
        );
        prev_bin = bin;
    }
}

// ── Entropy Truncate Horizon ──────────────────────────────────

#[test]
fn test_entropy_truncate_horizon_low_entropy() {
    // Below threshold → no truncation
    assert_eq!(entropy_truncate_horizon(1.0, 8), 8);
    assert_eq!(entropy_truncate_horizon(0.0, 8), 8);
    assert_eq!(entropy_truncate_horizon(2.49, 8), 8);
}

#[test]
fn test_entropy_truncate_horizon_high_entropy() {
    // Above threshold → truncated to 2
    assert_eq!(entropy_truncate_horizon(3.0, 8), 2);
    assert_eq!(entropy_truncate_horizon(5.0, 8), 2);
    assert_eq!(entropy_truncate_horizon(2.51, 8), 2);
}

#[test]
fn test_entropy_truncate_horizon_respects_max() {
    // If max_horizon < TRUNCATED_HORIZON, use max_horizon
    assert_eq!(entropy_truncate_horizon(3.0, 1), 1);
    assert_eq!(entropy_truncate_horizon(3.0, 2), 2);
    assert_eq!(entropy_truncate_horizon(3.0, 0), 0);
}

#[test]
fn test_entropy_truncate_horizon_at_boundary() {
    // Exactly at threshold → not truncated (strict >)
    assert_eq!(entropy_truncate_horizon(2.5, 8), 8);
    // Just above → truncated
    assert_eq!(entropy_truncate_horizon(2.5001, 8), 2);
}

// ── Reward Signal ─────────────────────────────────────────────

#[test]
fn test_reward_signal_quality_dominates() {
    let reward = ConfiguratorBandit::reward_signal(0.8, 0.1, 0.1);
    assert!(reward > 0.0, "quality=0.8 - 0.1*0.1 = 0.79 > 0");
    assert!(approx_eq(reward, 0.79, 1e-6));
}

#[test]
fn test_reward_signal_cost_dominates() {
    let reward = ConfiguratorBandit::reward_signal(0.01, 1.0, 0.1);
    assert!(reward < 0.0, "quality=0.01 - 0.1*1.0 = -0.09 < 0");
    assert!(approx_eq(reward, -0.09, 1e-6));
}

#[test]
fn test_reward_signal_zero_beta_ignores_cost() {
    let reward = ConfiguratorBandit::reward_signal(0.5, 1.0, 0.0);
    assert!(approx_eq(reward, 0.5, 1e-6));
}

#[test]
fn test_reward_signal_zero_quality() {
    let reward = ConfiguratorBandit::reward_signal(0.0, 0.5, 0.1);
    assert!(approx_eq(reward, -0.05, 1e-6));
}

#[test]
fn test_reward_signal_high_beta_penalizes_cost() {
    let reward = ConfiguratorBandit::reward_signal(0.5, 1.0, 1.0);
    assert!(approx_eq(reward, -0.5, 1e-6));
}

// ── ConfiguratorBandit Selection ──────────────────────────────

#[test]
fn test_configurator_bandit_new() {
    let bandit = ConfiguratorBandit::new();
    assert_eq!(bandit.num_contexts(), 0);
}

#[test]
fn test_configurator_bandit_default() {
    let bandit = ConfiguratorBandit::default();
    assert_eq!(bandit.num_contexts(), 0);
}

#[test]
fn test_configurator_bandit_explores_all_arms() {
    let mut bandit = ConfiguratorBandit::new();
    let ctx = ConfiguratorContext::new(0, 5);

    // UCB1 gives f32::MAX to unvisited arms, so first 3 selects cover all arms
    let mut seen_new = false;
    let mut seen_extend = false;
    let mut seen_skip = false;
    let mut seen_spechop = false;

    for _ in 0..4 {
        let decision = bandit.select(ctx);
        match decision {
            PlanningDecision::PlanNew => seen_new = true,
            PlanningDecision::PlanExtend => seen_extend = true,
            PlanningDecision::PlanSkip => seen_skip = true,
            PlanningDecision::SpecHop { .. } => seen_spechop = true,
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::HarnessUpdate => {}
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::WeightUpdate => {}
        }
        bandit.update(ctx, decision, 0.5);
    }

    assert!(seen_new, "should have tried PlanNew");
    assert!(seen_extend, "should have tried PlanExtend");
    assert!(seen_skip, "should have tried PlanSkip");
    assert!(seen_spechop, "should have tried SpecHop");
}

#[test]
fn test_configurator_bandit_converges_to_best_arm() {
    let mut bandit = ConfiguratorBandit::new();
    let ctx = ConfiguratorContext::new(0, 5);

    // PlanSkip gets consistently high rewards
    for _ in 0..100 {
        let decision = bandit.select(ctx);
        let reward = match decision {
            PlanningDecision::PlanSkip => 1.0,
            _ => 0.0,
        };
        bandit.update(ctx, decision, reward);
    }

    let skip_visits = bandit.visit_count(ctx, PlanningDecision::PlanSkip);
    assert!(
        skip_visits > 30,
        "PlanSkip should dominate after 100 rounds, got {skip_visits} visits"
    );
}

#[test]
fn test_configurator_bandit_selects_plan_skip_at_low_entropy() {
    let mut bandit = ConfiguratorBandit::new();
    let ctx = ConfiguratorContext::new(0, 0);

    // Train: PlanSkip is best at low entropy
    for _ in 0..200 {
        let decision = bandit.select(ctx);
        let reward = match decision {
            PlanningDecision::PlanSkip => 0.9,
            PlanningDecision::PlanExtend => 0.3,
            PlanningDecision::PlanNew => 0.1,
            PlanningDecision::SpecHop { .. } => 0.2,
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::HarnessUpdate => 0.2,
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::WeightUpdate => 0.1,
        };
        bandit.update(ctx, decision, reward);
    }

    let skip_visits = bandit.visit_count(ctx, PlanningDecision::PlanSkip);
    let new_visits = bandit.visit_count(ctx, PlanningDecision::PlanNew);
    assert!(
        skip_visits > new_visits,
        "PlanSkip should dominate at low entropy: skip={skip_visits} > new={new_visits}"
    );
}

#[test]
fn test_configurator_bandit_selects_plan_new_at_high_entropy() {
    let mut bandit = ConfiguratorBandit::new();
    let ctx = ConfiguratorContext::new(0, 9);

    // Train: PlanNew is best at high entropy
    for _ in 0..200 {
        let decision = bandit.select(ctx);
        let reward = match decision {
            PlanningDecision::PlanNew => 0.9,
            PlanningDecision::PlanExtend => 0.3,
            PlanningDecision::PlanSkip => 0.1,
            PlanningDecision::SpecHop { .. } => 0.2,
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::HarnessUpdate => 0.2,
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::WeightUpdate => 0.1,
        };
        bandit.update(ctx, decision, reward);
    }

    let new_visits = bandit.visit_count(ctx, PlanningDecision::PlanNew);
    let skip_visits = bandit.visit_count(ctx, PlanningDecision::PlanSkip);
    assert!(
        new_visits > skip_visits,
        "PlanNew should dominate at high entropy: new={new_visits} > skip={skip_visits}"
    );
}

// ── Context Isolation ─────────────────────────────────────────

#[test]
fn test_configurator_bandit_context_isolation() {
    let mut bandit = ConfiguratorBandit::new();
    let ctx_low = ConfiguratorContext::new(0, 1);
    let ctx_high = ConfiguratorContext::new(0, 8);

    // Train low entropy to prefer PlanSkip
    for _ in 0..50 {
        let decision = bandit.select(ctx_low);
        let reward = match decision {
            PlanningDecision::PlanSkip => 1.0,
            _ => 0.0,
        };
        bandit.update(ctx_low, decision, reward);
    }

    // Train high entropy to prefer PlanNew
    for _ in 0..50 {
        let decision = bandit.select(ctx_high);
        let reward = match decision {
            PlanningDecision::PlanNew => 1.0,
            _ => 0.0,
        };
        bandit.update(ctx_high, decision, reward);
    }

    // Contexts should have independent Q-values
    let skip_q_low = bandit
        .q_value(ctx_low, PlanningDecision::PlanSkip)
        .unwrap_or(0.0);
    let new_q_low = bandit
        .q_value(ctx_low, PlanningDecision::PlanNew)
        .unwrap_or(0.0);
    assert!(
        skip_q_low > new_q_low,
        "low entropy ctx should prefer PlanSkip: skip_q={skip_q_low} > new_q={new_q_low}"
    );

    let new_q_high = bandit
        .q_value(ctx_high, PlanningDecision::PlanNew)
        .unwrap_or(0.0);
    let skip_q_high = bandit
        .q_value(ctx_high, PlanningDecision::PlanSkip)
        .unwrap_or(0.0);
    assert!(
        new_q_high > skip_q_high,
        "high entropy ctx should prefer PlanNew: new_q={new_q_high} > skip_q={skip_q_high}"
    );
}

#[test]
fn test_configurator_bandit_domain_isolation() {
    let mut bandit = ConfiguratorBandit::new();
    let ctx_a = ConfiguratorContext::new(0, 5);
    let ctx_b = ConfiguratorContext::new(1, 5);

    // Same entropy bin, different domain
    bandit.update(ctx_a, PlanningDecision::PlanNew, 1.0);
    bandit.update(ctx_b, PlanningDecision::PlanSkip, 1.0);

    let q_a = bandit
        .q_value(ctx_a, PlanningDecision::PlanNew)
        .unwrap_or(0.0);
    let q_b = bandit
        .q_value(ctx_b, PlanningDecision::PlanSkip)
        .unwrap_or(0.0);

    assert!(approx_eq(q_a, 1.0, 1e-6), "domain 0 PlanNew q={q_a}");
    assert!(approx_eq(q_b, 1.0, 1e-6), "domain 1 PlanSkip q={q_b}");
}

// ── Q-Value Update (Incremental Mean) ────────────────────────

#[test]
fn test_q_value_incremental_mean() {
    let mut bandit = ConfiguratorBandit::new();
    let ctx = ConfiguratorContext::new(0, 5);

    bandit.update(ctx, PlanningDecision::PlanSkip, 1.0);
    let q = bandit.q_value(ctx, PlanningDecision::PlanSkip).unwrap();
    assert!(approx_eq(q, 1.0, 1e-6), "after 1 update: q={q}");

    bandit.update(ctx, PlanningDecision::PlanSkip, 0.0);
    let q = bandit.q_value(ctx, PlanningDecision::PlanSkip).unwrap();
    assert!(approx_eq(q, 0.5, 1e-6), "after 2 updates: q={q}");

    bandit.update(ctx, PlanningDecision::PlanSkip, 1.0);
    let q = bandit.q_value(ctx, PlanningDecision::PlanSkip).unwrap();
    assert!(
        approx_eq(q, 2.0 / 3.0, 1e-5),
        "after 3 updates: expected ~0.667, got {q}"
    );
}

#[test]
fn test_unvisited_context_returns_defaults() {
    let bandit = ConfiguratorBandit::new();
    let ctx = ConfiguratorContext::new(99, 5);

    assert_eq!(bandit.q_value(ctx, PlanningDecision::PlanNew), None);
    assert_eq!(bandit.visit_count(ctx, PlanningDecision::PlanNew), 0);
    assert_eq!(bandit.total_pulls(ctx), 0);
}

#[test]
fn test_num_contexts_tracks_entries() {
    let mut bandit = ConfiguratorBandit::new();
    assert_eq!(bandit.num_contexts(), 0);

    let ctx1 = ConfiguratorContext::new(0, 3);
    bandit.update(ctx1, PlanningDecision::PlanNew, 0.5);
    assert_eq!(bandit.num_contexts(), 1);

    let ctx2 = ConfiguratorContext::new(1, 7);
    bandit.update(ctx2, PlanningDecision::PlanSkip, 0.8);
    assert_eq!(bandit.num_contexts(), 2);

    // Same context doesn't add new entry
    bandit.update(ctx1, PlanningDecision::PlanExtend, 0.3);
    assert_eq!(bandit.num_contexts(), 2);
}

// ── Visit Count Tracking ──────────────────────────────────────

#[test]
fn test_visit_counts_per_arm() {
    let mut bandit = ConfiguratorBandit::new();
    let ctx = ConfiguratorContext::new(0, 5);

    bandit.update(ctx, PlanningDecision::PlanNew, 0.5);
    bandit.update(ctx, PlanningDecision::PlanNew, 0.6);
    bandit.update(ctx, PlanningDecision::PlanSkip, 0.9);

    assert_eq!(bandit.visit_count(ctx, PlanningDecision::PlanNew), 2);
    assert_eq!(bandit.visit_count(ctx, PlanningDecision::PlanExtend), 0);
    assert_eq!(bandit.visit_count(ctx, PlanningDecision::PlanSkip), 1);
    assert_eq!(bandit.total_pulls(ctx), 3);
}
