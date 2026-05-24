//! MeMo Reflection QA — Bomber domain example (Plan 094).
//!
//! Demonstrates the 5-step reflection QA pipeline on synthetic bomber game data.
//! Run: `cargo run --example bomber_13_reflection_qa --features memo_reflections`

#[cfg(feature = "memo_reflections")]
fn main() {
    use katgpt_rs::pruners::{GameStateSnapshot, ReflectionDomain, synthesize_reflections};

    // Generate synthetic bomber game replay data (100 rounds)
    let states: Vec<GameStateSnapshot> = (0..100)
        .map(|i| {
            let x = i % 13;
            let y = (i * 3) % 11;
            GameStateSnapshot {
                tick: i as u32,
                state_description: format!(
                    "grid({x},{y}) walls={w} bombs={b} enemies={e}",
                    x = x,
                    y = y,
                    w = (i % 7) + 2,
                    b = i % 4,
                    e = (i % 3) + 1,
                ),
                action_description: Some(match i % 6 {
                    0 => "move_north".to_string(),
                    1 => "move_south".to_string(),
                    2 => "move_east".to_string(),
                    3 => "move_west".to_string(),
                    4 => "place_bomb".to_string(),
                    _ => "wait".to_string(),
                }),
                outcome_description: match i % 4 {
                    0 => Some(format!("explosion at ({x},{y})", x = x, y = y)),
                    _ if i == 99 => Some("victory".to_string()),
                    _ => None,
                },
                score: match i > 80 {
                    true => 0.9,
                    false => 0.3 + (i as f32 / 100.0),
                },
            }
        })
        .collect();

    println!("=== MeMo Reflection QA — Bomber Domain ===\n");
    println!("Input: {} game state snapshots\n", states.len());

    let result = synthesize_reflections(&states, ReflectionDomain::Bomber);

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
    println!("Run: cargo run --example bomber_13_reflection_qa --features memo_reflections");
}
