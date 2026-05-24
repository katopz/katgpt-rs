//! Bomber Arena — Agent Validator Loop (Issue 052, Task C8)
//!
//! Demonstrates the agent loop: TemplateProposer → evaluate → iterate.
//! Runs multiple generations of rule candidates, selects top performers,
//! mutates from failure traces, and outputs the best discovered validator.
//!
//! Run: `cargo run --example bomber_08_agent_loop --features bomber-agent --quiet`

use katgpt_rs::pruners::bomber::{AgentLoop, ValidatorRule};

fn main() {
    println!("Bomber Arena — Agent Validator Loop");
    println!("═══════════════════════════════════");

    let agent = AgentLoop::new()
        .max_generations(10)
        .rounds_per_eval(50)
        .population_size(10);

    let result = agent.run();

    println!("\n🏆 Best Validator Discovered:");
    println!("   ID: {}", result.best_candidate.id);
    println!("   Generation: {}", result.best_candidate.generation);
    println!("   Rules: {} rule(s)", result.best_candidate.rules.len());

    for (i, rule) in result.best_candidate.rules.iter().enumerate() {
        let desc = match rule {
            ValidatorRule::AvoidBlast { lookahead } => {
                format!("AvoidBlast {{ lookahead: {lookahead} }}")
            }
            ValidatorRule::DistanceFromBomb { min_distance } => {
                format!("DistanceFromBomb {{ min_distance: {min_distance} }}")
            }
            ValidatorRule::SeekPowerUp { priority } => {
                format!("SeekPowerUp {{ priority: {priority:.2} }}")
            }
            ValidatorRule::AvoidDeadEnd { lookahead } => {
                format!("AvoidDeadEnd {{ lookahead: {lookahead} }}")
            }
            ValidatorRule::BlockOpponent { aggression } => {
                format!("BlockOpponent {{ aggression: {aggression:.2} }}")
            }
        };
        println!("     [{i}] {desc}");
    }

    println!("\n📊 Evaluation Metrics:");
    println!(
        "   Avg Score:       {:.1}",
        result.best_evaluation.avg_score
    );
    println!(
        "   Survival Rate:   {:.1}%",
        result.best_evaluation.survival_rate * 100.0
    );
    println!(
        "   Kill Rate:       {:.2}/round",
        result.best_evaluation.kill_rate
    );
    println!(
        "   Failure Traces:  {} rounds with fatal moves",
        result.best_evaluation.failure_traces.len()
    );

    println!("\n🔄 Optimization Summary:");
    println!("   Generations:            {}", result.generations_run);
    println!(
        "   Candidates evaluated:   {}",
        result.total_candidates_evaluated
    );

    // Classify failure patterns in best candidate
    let failure_count = result.best_evaluation.failure_traces.len();
    let rounds = result.best_evaluation.rounds;
    let safe_round_pct = match rounds {
        0 => 100.0,
        _ => ((rounds - failure_count as u32) as f32 / rounds as f32) * 100.0,
    };
    println!("   Safe rounds:            {safe_round_pct:.1}%");

    println!("\n✅ Done.");
}
