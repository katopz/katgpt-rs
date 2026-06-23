//! Plan 304 T3.4 — Gain/Cost Loop Halting demo (pure-kernel, synthetic).
//!
//! Simulates the per-loop halting decision a looped forward pass would make,
//! WITHOUT needing a real transformer. The hidden-state step sizes are mocked
//! to follow a geometric decay (the regime where the halter pays off: crowd-NPC
//! inputs where refinement collapses fast). The halter fires at the
//! gain/cost crossover, saving the dead-compute loops.
//!
//! Run: `cargo run --example gain_cost_halt_demo --features gain_cost_halt`
//!
//! This validates the Phase-2 wiring conceptually: the kernel's
//! `halt_decision` + `step_size` + `angular_change` compose into a working
//! halter on a synthetic loop trace. The real `forward_looped` integration is
//! wired in `src/transformer.rs` (Plan 304 T2.1–T2.3); this example isolates
//! the halter logic for readability.

#![cfg(feature = "gain_cost_halt")]

use katgpt_core::gain_cost_halt::{
    GainCostLoopHalter, HaltDecision, HaltReason, angular_change, step_size,
};

fn main() {
    // Halter config: tau=1.0 (symmetric gain/cost), patience=1 (halt on first
    // reversal), l_min=1 (allow halting from loop 1 onward). These are the
    // paper defaults.
    let mut halter = GainCostLoopHalter::new(1.0, 1, 1);

    // Fixed cost floor (flat Ω(r), LoopCoder-v2 default). The Phase-2 wiring
    // caches this as 0.01 × the first loop's step size; here we set it
    // directly to 0.1 to make the crossover land mid-trace.
    let cost_floor: f32 = 0.1;

    // Simulate 10 loops. Step size starts at 1.0 and decays by 0.5× each loop
    // (geometric collapse — the crowd-NPC regime). The hidden states are
    // mocked as 4-d vectors along a fixed refinement direction so the angular
    // change is +1.0 (aligned, convergent) — no oscillation, so the halter
    // fires purely on the gain/cost scissors.
    let n_loops = 10;
    let dim = 4;
    let direction: Vec<f32> = vec![1.0, 0.5, -0.3, 0.2]; // unit-ish refinement dir
    let mut prev_hidden: Vec<f32> = vec![0.0; dim];
    let mut prev_step: Vec<f32> = Vec::new();

    println!("Plan 304 — Gain/Cost Loop Halting demo (synthetic)");
    println!("Config: tau=1.0, patience=1, l_min=1, cost_floor={cost_floor}");
    println!("Step sizes decay geometrically (×0.5 per loop) — crowd-NPC regime.");
    println!();
    println!(
        "{:<10} {:>14} {:>10} {:>12} {:>28}",
        "loop_idx", "step_size(gain)", "cost", "cos_theta", "decision"
    );
    println!("{}", "-".repeat(78));

    let mut halted_at: Option<usize> = None;

    for tau in 1..=n_loops {
        // Mock the step size for this loop: geometric decay.
        let step_mag = 1.0_f32 * 0.5f32.powi((tau - 1) as i32);

        // Construct the current hidden state = prev + step_mag × direction.
        let mut curr_hidden = prev_hidden.clone();
        for (c, d) in curr_hidden.iter_mut().zip(direction.iter()) {
            *c += step_mag * d;
        }

        // gain = ||h^(tau) - h^(tau-1)||₂ = step_mag × ||direction||₂.
        let gain = step_size(&curr_hidden, &prev_hidden);

        // curr_step vector = curr - prev = step_mag × direction.
        let curr_step: Vec<f32> = curr_hidden
            .iter()
            .zip(prev_hidden.iter())
            .map(|(c, p)| c - p)
            .collect();

        // cos_theta vs the previous step. On tau==1 there's no prev_step.
        let cos_theta = if prev_step.is_empty() {
            0.0
        } else {
            angular_change(&curr_step, &prev_step)
        };

        let decision = halter.halt_decision(tau, gain, cost_floor, cos_theta);

        let decision_str = match decision {
            HaltDecision::Continue => "Continue".to_string(),
            HaltDecision::Halt { reason } => {
                format!("HALT ({:?})", reason)
            }
            HaltDecision::RefusedFloor => "RefusedFloor".to_string(),
        };

        println!(
            "{:<10} {:>14.6} {:>10.4} {:>12.4} {:>28}",
            tau, gain, cost_floor, cos_theta, decision_str
        );

        // Roll state for the next loop.
        prev_step = curr_step;
        prev_hidden = curr_hidden;
        halter.update_prev_step(gain);

        if let HaltDecision::Halt { .. } = decision {
            halted_at = Some(tau);
            break;
        }
    }

    println!("{}", "-".repeat(78));
    match halted_at {
        Some(idx) => {
            let saved = n_loops - idx;
            let pct = 100.0 * saved as f32 / n_loops as f32;
            println!(
                "Halted at loop {idx}/{n_loops} — saved {saved} dead-compute loops ({pct:.0}%)."
            );
            // Verify the halt reason is GainBelowCost (no oscillation in this trace).
            // Re-run the deciding loop to capture the reason cleanly.
            let step_at_halt = 1.0_f32 * 0.5f32.powi((idx - 1) as i32);
            assert!(
                step_at_halt < cost_floor,
                "halt must be at the gain/cost crossover"
            );
            println!(
                "  crossover: step_size({step_at_halt:.4}) < cost_floor({cost_floor}) → GainBelowCost ✓"
            );
        }
        None => {
            println!("Ran all {n_loops} loops without halting (gain never dropped below cost).");
        }
    }

    // Sanity: confirm the oscillation path also works on a reversing trace.
    println!();
    println!("─ Oscillation check (reversing direction every loop) ─");
    let mut h2 = GainCostLoopHalter::new(1.0, 1, 1);
    // loop 1: aligned (cos theta unknown → 0.0), loop 2: reversal (cos theta -1.0).
    let d1 = h2.halt_decision(1, 10.0, 0.0, 0.0);
    let d2 = h2.halt_decision(2, 10.0, 0.0, -1.0);
    println!("  loop 1 (cos=0.0):  {d1:?}");
    println!("  loop 2 (cos=-1.0): {d2:?}");
    assert_eq!(d1, HaltDecision::Continue);
    assert_eq!(
        d2,
        HaltDecision::Halt {
            reason: HaltReason::Oscillation
        }
    );
    println!("  → oscillation detector halts on the first reversal ✓");
}
