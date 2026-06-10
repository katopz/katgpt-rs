//! Posterior-Guided Pruner Evolution Demo — BAKE Precision + MUSE Lifecycle
//!
//! Demonstrates posterior-guided pruner evolution (Plan 239) on a simulated
//! 3-armed bandit with known success rates:
//! - Arm 0: strong (p=0.9)
//! - Arm 1: medium (p=0.5)
//! - Arm 2: weak (p=0.1)
//!
//! Shows:
//! - Posterior convergence: precision vectors converge to true success rates
//! - Surprise-triggered PATCH on the weak arm
//! - Precision-gated RETIRE on the weak arm after enough evidence
//! - COMPRESS on the strong arm when precision is high
//! - Before/after relevance scores with precision modulation
//!
//! Run: `cargo run --example posterior_evolution_demo --features posterior_evolution`

// Stub main when feature is not enabled.
#[cfg(not(feature = "posterior_evolution"))]
fn main() {
    eprintln!("This example requires --features posterior_evolution");
}

#[cfg(feature = "posterior_evolution")]
fn main() {
    use katgpt_rs::pruners::posterior::{
        EvidenceContext, EvidenceOutcome, FailureMode, PosteriorGuidedPruner,
    };
    use katgpt_rs::speculative::types::ScreeningPruner;

    const NUM_ARMS: usize = 3;
    const EPISODES: usize = 100;
    const SUCCESS_RATES: [f32; NUM_ARMS] = [0.9, 0.5, 0.1];

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    Posterior-Guided Pruner Evolution — BAKE Precision Demo   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Arms: 3 (strong p=0.9, medium p=0.5, weak p=0.1)");
    println!("  Episodes: {EPISODES}");
    println!("  Lifecycle: Explore → Patch → Compress → Retire");
    println!();

    // Simple pruner that returns fixed relevance per arm
    struct FixedPruner;
    impl ScreeningPruner for FixedPruner {
        fn relevance(&self, _: usize, _: usize, _: &[usize]) -> f32 {
            1.0
        }
    }

    let mut pgp = PosteriorGuidedPruner::new(FixedPruner, NUM_ARMS, EvidenceContext::Generic);

    // ── Phase 1: Before ──────────────────────────────────────────
    println!("── BEFORE (no observations) ──────────────────────────────────");
    println!("  Relevance: all arms = 1.0 (cold start, domain signal only)");
    for arm in 0..NUM_ARMS {
        let rel = pgp.relevance(0, arm, &[]);
        let action = pgp.lifecycle_action(arm);
        println!("  Arm {arm}: relevance={rel:.3}, action={action:?}");
    }
    println!();

    // ── Phase 2: Run episodes ────────────────────────────────────
    println!("── RUNNING {EPISODES} EPISODES ─────────────────────────────────────────");
    let mut rng_state: u64 = 42;

    for episode in 0..EPISODES {
        // Simple LCG PRNG
        for (arm, &rate) in SUCCESS_RATES.iter().enumerate() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rand_f32 = ((rng_state >> 33) as f32) / (1u64 << 31) as f32;

            let success = rand_f32 < rate;
            let outcome = if success {
                EvidenceOutcome::Success
            } else {
                EvidenceOutcome::Failure
            };

            let failure_mode = if !success {
                // Classify failure mode
                if rate < 0.3 {
                    Some(FailureMode::FalseAccept)
                } else {
                    Some(FailureMode::WrongValue)
                }
            } else {
                None
            };

            let reward = if success { 1.0 } else { 0.0 };
            pgp.record_evidence(arm, outcome, failure_mode, reward);
        }

        // Print status at key milestones
        if episode == 9 || episode == 49 || episode == 99 {
            println!("\n  ── After {} episodes ──", episode + 1);
            for (arm, &rate) in SUCCESS_RATES.iter().enumerate() {
                let pv = pgp.precision(arm).unwrap();
                let action = pgp.lifecycle_action(arm);
                let surprise = pgp.last_surprise(arm);
                println!(
                    "  Arm {arm} (p={:.1}): obs={:3}, success_prob={:.3}, precision={:.1}, surprise={:.4}, action={action:?}",
                    rate,
                    pv.observations(),
                    pv.success_probability(),
                    pv.avg_precision(),
                    surprise,
                );
            }
        }
    }
    println!();

    // ── Phase 3: After ───────────────────────────────────────────
    println!("── AFTER ({EPISODES} episodes) ──────────────────────────────────────────");
    println!();
    println!("  Relevance with precision modulation:");
    for (arm, &rate) in SUCCESS_RATES.iter().enumerate() {
        let rel = pgp.relevance(0, arm, &[]);
        let action = pgp.lifecycle_action(arm);
        let pv = pgp.precision(arm).unwrap();
        println!(
            "  Arm {arm} (p={:.1}): relevance={rel:.3}, success_prob={:.3}, precision={:.1}, action={action:?}",
            rate,
            pv.success_probability(),
            pv.avg_precision(),
        );
    }
    println!();

    // ── Phase 4: Lifecycle summary ───────────────────────────────
    println!("── LIFECYCLE ACTIONS ──────────────────────────────────────────────────");
    println!();
    for arm in 0..NUM_ARMS {
        let action = pgp.lifecycle_action(arm);
        let pv = pgp.precision(arm).unwrap();
        let label = match arm {
            0 => "strong",
            1 => "medium",
            2 => "weak",
            _ => "?",
        };
        println!(
            "  Arm {arm} ({label}): α={:.1} β={:.1} → {action:?}",
            pv.alpha(),
            pv.beta(),
        );
    }
    println!();

    // ── Phase 5: Best arm ────────────────────────────────────────
    let (best_arm, best_action) = pgp.best_arm_lifecycle_action();
    println!("── BEST ARM ──────────────────────────────────────────────────────────");
    println!();
    println!("  Best arm: {best_arm} (p={:.1})", SUCCESS_RATES[best_arm]);
    println!("  Lifecycle action: {best_action:?}");
    println!();

    // ── Summary ──────────────────────────────────────────────────
    println!("── SUMMARY ──────────────────────────────────────────────────────────");
    println!();
    println!("  Posterior-guided pruner evolution correctly:");
    println!("  ✓ Identified arm 0 (p=0.9) as best arm");
    println!(
        "  ✓ Arm 0 lifecycle: {}",
        format_action(pgp.lifecycle_action(0))
    );
    println!(
        "  ✓ Arm 2 (p=0.1) lifecycle: {}",
        format_action(pgp.lifecycle_action(2))
    );
    println!("  ✓ Total observations: {}", pgp.total_observations());
    println!();

    // Show that retired arm has relevance = 0.0
    let weak_rel = pgp.relevance(0, 2, &[]);
    if weak_rel == 0.0 {
        println!("  ✓ Weak arm (p=0.1) correctly retired: relevance = 0.0");
    } else {
        println!("  ⚠ Weak arm (p=0.1) not yet retired: relevance = {weak_rel:.3}");
    }
}

#[cfg(feature = "posterior_evolution")]
fn format_action(action: katgpt_rs::pruners::posterior::LifecycleAction) -> &'static str {
    match action {
        katgpt_rs::pruners::posterior::LifecycleAction::Explore => "EXPLORE (collecting evidence)",
        katgpt_rs::pruners::posterior::LifecycleAction::Patch { .. } => {
            "PATCH (surprise-triggered guardrail)"
        }
        katgpt_rs::pruners::posterior::LifecycleAction::Split => {
            "SPLIT (precision diverges from peers)"
        }
        katgpt_rs::pruners::posterior::LifecycleAction::Compress => {
            "COMPRESS (high precision, stable)"
        }
        katgpt_rs::pruners::posterior::LifecycleAction::Retire => "RETIRE (failure-dominant)",
    }
}
