//! Head budget solver example (Plan 271 Phase 3).
//!
//! Demonstrates the nonuniform per-head budget allocation pipeline:
//! 1. Generate synthetic per-head sensitivity curves (some flat, some steep).
//! 2. Solve for a target overall ratio of 0.05 (aggressive compaction).
//! 3. Print the resulting per-head shares.
//! 4. Wrap the result in a `HeadBudgetSchedule`, serialize to postcard,
//!    round-trip back, and verify the BLAKE3 hash.
//!
//! Run with:
//! ```bash
//! cargo run --example attn_match_head_budget --features attn_match --release
//! ```

use katgpt_rs::attn_match::head_budget::{
    HeadBudgetSchedule, HeadBudgetSolver, HeadSensitivityCurve,
};

fn main() {
    println!("=== Attention Matching — Head Budget Solver (Plan 271 Phase 3) ===\n");

    // 1. Build synthetic sensitivity curves.
    // Heads 0, 2, 4 are sensitive (steep quality loss as ratio drops).
    // Heads 1, 3, 5 are flat (barely affected by compaction).
    let num_layers = 2;
    let num_heads = 6;
    let ratios = vec![0.05, 0.1, 0.25, 0.5, 0.75, 1.0];
    let curves: Vec<HeadSensitivityCurve> = (0..num_layers * num_heads)
        .map(|i| {
            let head_id = i;
            let deltas: Vec<f32> = ratios
                .iter()
                .map(|&r| {
                    let one_minus_r = (1.0f32 - r).max(0.0);
                    if head_id % 2 == 0 {
                        // Sensitive head: steep curve.
                        one_minus_r.powf(1.5) * 0.8
                    } else {
                        // Flat head: barely changes.
                        one_minus_r * 0.1
                    }
                })
                .collect();
            HeadSensitivityCurve::new(head_id, ratios.clone(), deltas)
        })
        .collect();

    println!(
        "Model: {} layers × {} heads = {} curves",
        num_layers,
        num_heads,
        curves.len()
    );
    println!("Target overall ratio: 0.05 (aggressive 20:1 compaction)\n");

    // Show the input curves.
    println!("Input sensitivity curves (ratio → quality delta):");
    for c in &curves {
        let kind = if c.head_id % 2 == 0 { "steep" } else { "flat" };
        print!("  head {:>2} [{}]: ", c.head_id, kind);
        for (r, d) in c.ratios.iter().zip(c.deltas.iter()) {
            print!("({:.2}→{:.3}) ", r, d);
        }
        println!();
    }
    println!();

    // 2. Solve for target ratio 0.05.
    let solver = HeadBudgetSolver::new(curves, num_layers, num_heads).with_step_size(0.01);
    let target_ratio = 0.05f32;
    let shares = solver.solve(target_ratio);

    // 3. Print the resulting per-head shares.
    println!("Solved per-head shares:");
    println!("  {:>10} {:>10} {:>12} {:>12}", "layer", "head", "share", "multiplier");
    for (i, &s) in shares.iter().enumerate() {
        let layer = i / num_heads;
        let head = i % num_heads;
        let mult = s / target_ratio;
        println!("  {:>10} {:>10} {:>12.4} {:>12.2}x", layer, head, s, mult);
    }

    let avg: f32 = shares.iter().sum::<f32>() / shares.len() as f32;
    let min = shares.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = shares.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!();
    println!(
        "  avg={:.4} (target {:.4}), min={:.4}, max={:.4}, spread={:.4}",
        avg,
        target_ratio,
        min,
        max,
        max - min
    );

    // Verify local optimality.
    let optimal = solver.is_locally_optimal(&shares);
    println!("  locally optimal (no improving single swap): {}", optimal);

    // 4. Wrap in a schedule, serialize, round-trip, verify.
    println!();
    println!("=== Schedule serialization ===");
    let schedule = HeadBudgetSchedule::new("synthetic-6h-2L".into(), shares.clone());
    println!("  model_id  : {}", schedule.model_id);
    println!("  version   : {}", schedule.version);
    println!(
        "  blake3    : {}",
        hex_short(&schedule.blake3_hash)
    );
    println!("  verify    : {}", schedule.verify());

    let bytes = schedule.to_postcard().expect("serialize");
    println!("  postcard  : {} bytes", bytes.len());

    let recovered = HeadBudgetSchedule::from_postcard(&bytes).expect("deserialize");
    println!(
        "  roundtrip : model={} shares_match={} hash_match={} verify={}",
        recovered.model_id == schedule.model_id,
        recovered.shares == schedule.shares,
        recovered.blake3_hash == schedule.blake3_hash,
        recovered.verify(),
    );

    // Tamper demo.
    let mut tampered = recovered.clone();
    tampered.shares[0] = 0.99;
    println!("  tamper    : shares[0]→0.99, verify={}", tampered.verify());

    println!();
    println!("Done. Sensitive (even-indexed) heads receive more budget than flat ones.");
}

/// Format a BLAKE3 hash as a short hex prefix for display.
fn hex_short(hash: &[u8; 32]) -> String {
    let mut s = String::with_capacity(16);
    for b in &hash[..7] {
        s.push_str(&format!("{:02x}", b));
    }
    s.push('…');
    s
}
