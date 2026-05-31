#![cfg(feature = "sr2am_configurator")]
//! GOAT Proof — SR²AM Configurator Bandit (Plan 112)
//!
//! Proves the ConfiguratorBandit correctly regulates planning in game-like
//! contexts using entropy-based context binning and UCB1 arm selection.
//!
//! Run: `cargo test --features sr2am_configurator --test bench_112_sr2am_configurator_goat -- --nocapture`

use katgpt_core::{ConfiguratorContext, PlanningDecision};
use katgpt_rs::pruners::ConfiguratorBandit;

const ROUNDS: usize = 1000;
const BETA: f32 = 0.1;

/// Simulate a game turn with configurable entropy level.
/// Returns (quality_gain, token_cost) based on the planning decision.
fn simulate_turn(decision: PlanningDecision, entropy: f32) -> (f32, f32) {
    match decision {
        PlanningDecision::PlanNew => {
            // Fresh tree: high cost, best quality at high entropy
            let quality = if entropy > 0.5 { 0.8 } else { 0.3 };
            (quality, 1.0)
        }
        PlanningDecision::PlanExtend => {
            // Extend tree: medium cost, decent quality
            (0.5, 0.5)
        }
        PlanningDecision::PlanSkip => {
            // Skip planning: zero cost, good quality at low entropy
            let quality = if entropy < 0.3 { 0.7 } else { 0.2 };
            (quality, 0.0)
        }
        PlanningDecision::SpecHop { k } => {
            // SpecHop: speculative threads, moderate cost, best when tool-bound
            let quality = if entropy < 0.3 { 0.6 } else { 0.3 };
            let cost = 0.1 * (k.min(8) as f32);
            (quality, cost)
        }
        #[cfg(feature = "sia_feedback")]
        PlanningDecision::HarnessUpdate => {
            // Harness update: moderate cost, medium quality
            (0.4, 0.5)
        }
        #[cfg(feature = "sia_feedback")]
        PlanningDecision::WeightUpdate => {
            // Weight update: high cost, good quality when stalled
            (0.6, 2.0)
        }
    }
}

// ── G1: Arm selection learns multiple contexts ──────────────────

#[test]
fn proof_1_arm_selection_based_on_context() {
    let mut bandit = ConfiguratorBandit::new();

    // Train on 1000 rounds with varying entropy
    for round in 0..ROUNDS {
        let entropy = (round as f32 / ROUNDS as f32).min(1.0);
        let entropy_bin = ConfiguratorBandit::entropy_bin(entropy);
        let ctx = ConfiguratorContext::new(0, entropy_bin);

        let decision = bandit.select(ctx);
        let (quality, cost) = simulate_turn(decision, entropy);
        let reward = ConfiguratorBandit::reward_signal(quality, cost, BETA);
        bandit.update(ctx, decision, reward);
    }

    // Verify the bandit has learned multiple contexts
    assert!(
        bandit.num_contexts() >= 3,
        "[G1 FAIL] Expected ≥3 contexts, got {}",
        bandit.num_contexts()
    );
    println!(
        "[G1] ✅ Bandit learned {} contexts across entropy spectrum",
        bandit.num_contexts()
    );
}

// ── G2: Low entropy → PlanSkip ─────────────────────────────────

#[test]
fn proof_2_low_entropy_prefers_skip() {
    let mut bandit = ConfiguratorBandit::new();
    let ctx_low = ConfiguratorContext::new(0, 0); // entropy ≈ 0

    // Train: PlanSkip is best at low entropy
    for _ in 0..500 {
        let decision = bandit.select(ctx_low);
        let (quality, cost) = simulate_turn(decision, 0.05);
        let reward = ConfiguratorBandit::reward_signal(quality, cost, BETA);
        bandit.update(ctx_low, decision, reward);
    }

    // After training, PlanSkip should have highest Q-value at low entropy
    let q_skip = bandit
        .q_value(ctx_low, PlanningDecision::PlanSkip)
        .unwrap_or(0.0);
    let q_new = bandit
        .q_value(ctx_low, PlanningDecision::PlanNew)
        .unwrap_or(0.0);
    let q_extend = bandit
        .q_value(ctx_low, PlanningDecision::PlanExtend)
        .unwrap_or(0.0);

    assert!(
        q_skip >= q_new && q_skip >= q_extend,
        "[G2 FAIL] PlanSkip Q={q_skip:.3} not highest vs PlanNew={q_new:.3}, PlanExtend={q_extend:.3} at low entropy"
    );
    println!(
        "[G2] ✅ Low entropy prefers PlanSkip: Q_skip={q_skip:.3} ≥ Q_new={q_new:.3}, Q_extend={q_extend:.3}"
    );
}

// ── G3: High entropy → PlanNew ─────────────────────────────────

#[test]
fn proof_3_high_entropy_prefers_new() {
    let mut bandit = ConfiguratorBandit::new();
    let ctx_high = ConfiguratorContext::new(0, 9); // entropy ≈ 1.0

    // Train: PlanNew is best at high entropy
    for _ in 0..500 {
        let decision = bandit.select(ctx_high);
        let (quality, cost) = simulate_turn(decision, 0.95);
        let reward = ConfiguratorBandit::reward_signal(quality, cost, BETA);
        bandit.update(ctx_high, decision, reward);
    }

    // After training, PlanNew should have highest Q-value at high entropy
    let q_new = bandit
        .q_value(ctx_high, PlanningDecision::PlanNew)
        .unwrap_or(0.0);
    let q_skip = bandit
        .q_value(ctx_high, PlanningDecision::PlanSkip)
        .unwrap_or(0.0);
    let q_extend = bandit
        .q_value(ctx_high, PlanningDecision::PlanExtend)
        .unwrap_or(0.0);

    assert!(
        q_new >= q_skip && q_new >= q_extend,
        "[G3 FAIL] PlanNew Q={q_new:.3} not highest vs PlanSkip={q_skip:.3}, PlanExtend={q_extend:.3} at high entropy"
    );
    println!(
        "[G3] ✅ High entropy prefers PlanNew: Q_new={q_new:.3} ≥ Q_skip={q_skip:.3}, Q_extend={q_extend:.3}"
    );
}

// ── G4: Reward signal tradeoff ─────────────────────────────────

#[test]
fn proof_4_reward_signal_tradeoff() {
    // Verify reward = quality_gain - beta * token_cost

    // High quality, low cost → positive reward
    let r1 = ConfiguratorBandit::reward_signal(0.8, 0.1, BETA);
    assert!(
        r1 > 0.0,
        "[G4a] High quality low cost should be positive: {r1}"
    );

    // Low quality, high cost → negative reward
    let r2 = ConfiguratorBandit::reward_signal(0.01, 1.0, BETA);
    assert!(
        r2 < 0.0,
        "[G4b] Low quality high cost should be negative: {r2}"
    );

    // Skip (zero cost) → reward = quality_gain
    let r3 = ConfiguratorBandit::reward_signal(0.5, 0.0, BETA);
    assert!(
        (r3 - 0.5).abs() < 1e-6,
        "[G4c] Zero cost reward should equal quality: {r3}"
    );

    println!("[G4] ✅ Reward signal correctly trades off quality vs cost");
    println!("     quality=0.8, cost=0.1 → reward={r1:.3}");
    println!("     quality=0.01, cost=1.0 → reward={r2:.3}");
    println!("     quality=0.5, cost=0.0 → reward={r3:.3}");
}

// ── G5: Context isolation — different domains learn different policies ──

#[test]
fn proof_5_context_isolation() {
    let mut bandit = ConfiguratorBandit::new();
    let ctx_game = ConfiguratorContext::new(0, 5);
    let ctx_code = ConfiguratorContext::new(1, 5);

    // Train game domain: PlanSkip is best
    for _ in 0..200 {
        let decision = bandit.select(ctx_game);
        let reward = match decision {
            PlanningDecision::PlanSkip => 1.0,
            _ => 0.0,
        };
        bandit.update(ctx_game, decision, reward);
    }

    // Train code domain: PlanNew is best
    for _ in 0..200 {
        let decision = bandit.select(ctx_code);
        let reward = match decision {
            PlanningDecision::PlanNew => 1.0,
            _ => 0.0,
        };
        bandit.update(ctx_code, decision, reward);
    }

    // Game domain should prefer PlanSkip
    let game_skip = bandit
        .q_value(ctx_game, PlanningDecision::PlanSkip)
        .unwrap_or(0.0);
    let game_new = bandit
        .q_value(ctx_game, PlanningDecision::PlanNew)
        .unwrap_or(0.0);
    assert!(
        game_skip > game_new,
        "[G5a] Game domain should prefer PlanSkip: {game_skip:.3} vs {game_new:.3}"
    );

    // Code domain should prefer PlanNew
    let code_new = bandit
        .q_value(ctx_code, PlanningDecision::PlanNew)
        .unwrap_or(0.0);
    let code_skip = bandit
        .q_value(ctx_code, PlanningDecision::PlanSkip)
        .unwrap_or(0.0);
    assert!(
        code_new > code_skip,
        "[G5b] Code domain should prefer PlanNew: {code_new:.3} vs {code_skip:.3}"
    );

    println!(
        "[G5] ✅ Context isolation: game→PlanSkip (Q={game_skip:.3} vs {game_new:.3}), code→PlanNew (Q={code_new:.3} vs {code_skip:.3})"
    );
}

// ── G6: Decision distribution shows meaningful plan_skip savings (>20%) ──

#[test]
fn proof_6_plan_skip_savings() {
    let mut bandit = ConfiguratorBandit::new();
    let mut skip_count = 0usize;
    let mut new_count = 0usize;
    let mut extend_count = 0usize;

    // Simulate 1000 game turns with natural entropy distribution
    for round in 0..ROUNDS {
        // Simulate entropy distribution: most turns are low entropy (confident)
        let entropy = match round % 10 {
            0..=2 => 0.1, // 30% very confident
            3..=5 => 0.3, // 30% somewhat confident
            6..=7 => 0.5, // 20% medium
            8..=9 => 0.8, // 20% uncertain
            _ => 0.5,
        };
        let entropy_bin = ConfiguratorBandit::entropy_bin(entropy);
        let ctx = ConfiguratorContext::new(0, entropy_bin);

        let decision = bandit.select(ctx);
        match decision {
            PlanningDecision::PlanSkip => skip_count += 1,
            PlanningDecision::PlanNew => new_count += 1,
            PlanningDecision::PlanExtend => extend_count += 1,
            PlanningDecision::SpecHop { .. } => {}
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::HarnessUpdate => {}
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::WeightUpdate => {}
        }

        let (quality, cost) = simulate_turn(decision, entropy);
        let reward = ConfiguratorBandit::reward_signal(quality, cost, BETA);
        bandit.update(ctx, decision, reward);
    }

    let total = skip_count + new_count + extend_count;
    let skip_pct = (skip_count as f64 / total as f64) * 100.0;
    let new_pct = (new_count as f64 / total as f64) * 100.0;
    let extend_pct = (extend_count as f64 / total as f64) * 100.0;

    println!("[G6] Decision distribution over {total} turns:");
    println!("     PlanSkip:   {skip_count} ({skip_pct:.1}%)");
    println!("     PlanNew:    {new_count} ({new_pct:.1}%)");
    println!("     PlanExtend: {extend_count} ({extend_pct:.1}%)");

    // GOAT gate: at least 20% plan_skip savings
    // With 60% low-entropy turns, bandit should learn to skip frequently
    assert!(
        skip_pct >= 20.0,
        "[G6 FAIL] PlanSkip savings {skip_pct:.1}% < 20%"
    );

    println!("[G6] ✅ PlanSkip savings: {skip_pct:.1}% ≥ 20%");
}
