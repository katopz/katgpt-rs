//! Validator Agent Benchmark — agent-discovered vs hand-crafted vs random (Issue 052, Task C10)
//!
//! Run: `cargo test --features bomber-agent bench_validator_agent -- --nocapture`

#[cfg(feature = "bomber-agent")]
use katgpt_rs::pruners::bomber::{
    AgentLoop, ValidatorCandidate, ValidatorRule, evaluate_validator,
};

#[cfg(feature = "bomber-agent")]
fn hand_crafted_baseline() -> ValidatorCandidate {
    ValidatorCandidate {
        id: "baseline".into(),
        generation: 0,
        rules: vec![
            ValidatorRule::AvoidBlast { lookahead: 3 },
            ValidatorRule::DistanceFromBomb { min_distance: 2 },
            ValidatorRule::SeekPowerUp { priority: 1.5 },
            ValidatorRule::AvoidDeadEnd { lookahead: 2 },
        ],
    }
}

#[cfg(feature = "bomber-agent")]
fn random_baseline() -> ValidatorCandidate {
    ValidatorCandidate {
        id: "random".into(),
        generation: 0,
        rules: vec![],
    }
}

#[cfg(feature = "bomber-agent")]
fn rules_count(rules: &[ValidatorRule]) -> usize {
    rules.len()
}

#[cfg(feature = "bomber-agent")]
fn improvement_pct(agent: f32, baseline: f32) -> String {
    match baseline {
        b if b == 0.0 => format!("+{:.1}", agent),
        b => format!("{:+.1}%", (agent - b) / b.abs() * 100.0),
    }
}

#[cfg(feature = "bomber-agent")]
#[test]
fn bench_validator_agent() {
    let eval_rounds: u32 = 100;

    // ── Hand-crafted baseline ─────────────────────────────────
    println!("\n🔬 Evaluating hand-crafted baseline ({eval_rounds} rounds)...");
    let baseline_candidate = hand_crafted_baseline();
    let baseline_eval = evaluate_validator(&baseline_candidate, eval_rounds);

    // ── Agent-discovered ───────────────────────────────────────
    println!("🤖 Running agent loop (5 gen, pop 8, 50 rounds/eval)...");
    let agent_result = AgentLoop::new()
        .max_generations(5)
        .population_size(8)
        .rounds_per_eval(50)
        .run();

    // Re-evaluate best with same rounds for fair comparison
    println!("🔬 Re-evaluating agent best ({eval_rounds} rounds)...");
    let agent_eval = evaluate_validator(&agent_result.best_candidate, eval_rounds);

    // ── Random baseline ────────────────────────────────────────
    println!("🎲 Evaluating random baseline ({eval_rounds} rounds)...");
    let rand_candidate = random_baseline();
    let rand_eval = evaluate_validator(&rand_candidate, eval_rounds);

    // ── Comparison table ───────────────────────────────────────
    let separator = "═".repeat(50);
    println!("\n{separator}");
    println!("Validator Agent Benchmark ({eval_rounds} rounds each)");
    println!("{separator}");

    println!("\nHand-crafted baseline:");
    println!(
        "  Rules: {}  Avg Score: {:+.1}  Survival: {:.0}%  Kills: {:.2}/round",
        rules_count(&baseline_candidate.rules),
        baseline_eval.avg_score,
        baseline_eval.survival_rate * 100.0,
        baseline_eval.kill_rate,
    );

    println!("\nAgent-discovered (5 gen, pop 8):");
    println!(
        "  Rules: {}  Avg Score: {:+.1}  Survival: {:.0}%  Kills: {:.2}/round",
        rules_count(&agent_result.best_candidate.rules),
        agent_eval.avg_score,
        agent_eval.survival_rate * 100.0,
        agent_eval.kill_rate,
    );
    println!(
        "  Generations: {}  Candidates evaluated: {}",
        agent_result.generations_run, agent_result.total_candidates_evaluated,
    );

    println!("\nRandom (no rules):");
    println!(
        "  Rules: {}  Avg Score: {:+.1}  Survival: {:.0}%  Kills: {:.2}/round",
        rules_count(&rand_candidate.rules),
        rand_eval.avg_score,
        rand_eval.survival_rate * 100.0,
        rand_eval.kill_rate,
    );

    println!(
        "\nAgent improvement over baseline: {} score ({})",
        format!("{:+.1}", agent_eval.avg_score - baseline_eval.avg_score),
        improvement_pct(agent_eval.avg_score, baseline_eval.avg_score),
    );

    // ── Assertions ─────────────────────────────────────────────
    // Agent-discovered should perform at least as well as random baseline
    assert!(
        agent_eval.avg_score >= rand_eval.avg_score,
        "Agent ({:.1}) should >= Random ({:.1})",
        agent_eval.avg_score,
        rand_eval.avg_score,
    );

    println!("\n✅ Benchmark complete.");
}
