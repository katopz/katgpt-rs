//! MeMo Reflection QA — Go domain example (Plan 094).
//!
//! Demonstrates the 5-step reflection QA pipeline on synthetic Go game data.
//! Run: `cargo run --example go_09_reflection_qa --features memo_reflections,go`

#[cfg(feature = "memo_reflections")]
fn main() {
    use katgpt_rs::pruners::{GameStateSnapshot, ReflectionDomain, synthesize_reflections};

    // Generate synthetic Go game replay data (100 moves)
    let states: Vec<GameStateSnapshot> = (0..100)
        .map(|i| {
            let row = (i * 3 + 7) % 9;
            let col = (i * 5 + 3) % 9;
            let captured = match i % 7 {
                0 => 1,
                _ => 0,
            };
            GameStateSnapshot {
                tick: i as u32,
                state_description: format!(
                    "board_move_{i}: stone_at({row},{col}) black_stones={b} white_stones={w} captured={c}",
                    row = row,
                    col = col,
                    b = i / 2 + captured,
                    w = (i + 1) / 2,
                    c = captured,
                ),
                action_description: Some(match i == 50 {
                    true => "pass".to_string(),
                    false => format!("place_stone({row},{col})"),
                }),
                outcome_description: match i {
                    99 => Some("game_over: black_wins_by_2.5".to_string()),
                    _ if captured > 0 => Some(format!(
                        "captured_{n}_stones_at({row},{col})",
                        n = captured
                    )),
                    _ => None,
                },
                score: match i {
                    0..=29 => 0.3,
                    30..=69 => 0.5,
                    _ => 0.7,
                },
            }
        })
        .collect();

    println!("=== MeMo Reflection QA — Go Domain ===\n");
    println!("Input: {} game state snapshots\n", states.len());

    let result = synthesize_reflections(&states, ReflectionDomain::Go);

    println!("Output: {} reflection QA pairs\n", result.pairs.len());
    println!("Step breakdown:");
    println!("  Direct extraction:  {}", result.step_counts[0]);
    println!("  Indirect extraction: {}", result.step_counts[1]);
    println!("  Consolidation:      {}", result.step_counts[2]);
    println!("  Verification:       {}", result.step_counts[3]);
    println!("  Entity surfacing:   {}", result.step_counts[4]);
    println!("  Cross-game synth:   {}", result.step_counts[5]);
    println!(
        "\nVerification rate: {:.1}%",
        result.verification_rate * 100.0
    );

    // Show sample pairs
    println!("\n--- Sample QA Pairs ---");
    for pair in result.pairs.iter().take(5) {
        println!("\n[{step}] Q: {q}", step = pair.step, q = pair.question);
        println!("       A: {a}", a = pair.answer);
    }
}

#[cfg(not(feature = "memo_reflections"))]
fn main() {
    println!("This example requires the `memo_reflections` feature.");
    println!("Run: cargo run --example go_09_reflection_qa --features memo_reflections,go");
}
