//! Plan 265 Phase 3 Example: CCCP vs. NoPruner DDTree expansion.
//!
//! Demonstrates the before/after DDTree expansion count when using the
//! Collider-Consistency ConstraintPruner (Fusion C) versus NoPruner.
//! CCCP rejects branches that complete no task collider, reducing the
//! tree expansion count at parity with the bandit-only baseline.

fn main() {
    #[cfg(feature = "collider_consistency")]
    {
        use katgpt_rs::collider_pruner::{
            ColliderConstraint, ColliderConstraintConfig, InterleavedTaskBenchmark,
        };

        println!("=== Plan 265 Phase 3: CCCP vs. NoPruner ===\n");

        // 5 tasks interleaved over 20 steps, d_hidden=20, segment_len=4.
        let bench = InterleavedTaskBenchmark::generate(20, 5, 4, 20);
        let parent_hidden: Vec<&[f32]> =
            bench.boundary_hidden.iter().map(|v| v.as_slice()).collect();
        println!(
            "Synthetic benchmark: {} boundaries, {} tasks, d_hidden=20.",
            bench.boundaries.len(),
            bench.n_tasks
        );

        let constraint = ColliderConstraint::new(
            bench.boundaries.clone(),
            (1..=bench.n_tasks).collect(),
            ColliderConstraintConfig::default(),
        );

        // Simulate DDTree expansion at each boundary.
        let mut nopruner_expansions = 0_usize;
        let mut cccp_expansions = 0_usize;
        for (i, h) in bench.boundary_hidden.iter().enumerate() {
            // Two candidates per boundary: correct-task extension vs. dead-task.
            let correct_task = i % bench.n_tasks;
            let dead_task = (i + 1) % bench.n_tasks;
            let sub = h.len() / bench.n_tasks;
            let mut correct_h = vec![0.0_f32; h.len()];
            let mut dead_h = vec![0.0_f32; h.len()];
            for j in correct_task * sub..(correct_task + 1) * sub {
                if j < h.len() {
                    correct_h[j] = 1.0;
                }
            }
            for j in dead_task * sub..(dead_task + 1) * sub {
                if j < h.len() {
                    dead_h[j] = 1.0;
                }
            }
            let depth = bench.boundaries[i] - 1;

            // NoPruner accepts both.
            nopruner_expansions += 2;

            // CCCP: reject dead-task branches (low collider-preservation score).
            // `is_valid_with_hidden` was the planned API; the shipped primitive
            // exposes `collider_preservation_score` (sigmoid in [0,1]). We treat
            // score > 0.5 as "valid" for this demo.
            if constraint.collider_preservation_score(depth, &parent_hidden, &correct_h) > 0.5 {
                cccp_expansions += 1;
            }
            if constraint.collider_preservation_score(depth, &parent_hidden, &dead_h) > 0.5 {
                cccp_expansions += 1;
            }
        }

        let reduction = 100.0 * (1.0 - cccp_expansions as f32 / nopruner_expansions as f32);
        println!("\nResults (before → after CCCP pruning):");
        println!("  NoPruner expansions: {nopruner_expansions}");
        println!("  CCCP expansions:     {cccp_expansions}");
        println!("  Reduction:           {reduction:.1}%");

        // No-task fast path: measure overhead.
        let noop = ColliderConstraint::default();
        let iters = 100_000_usize;
        let start = std::time::Instant::now();
        let mut sink = 0_u64;
        for i in 0..iters {
            // `is_noop()` is the no-collider fast path (the analog of the
            // planned `is_valid` trivial-accept).
            let v = noop.is_noop();
            sink = sink.wrapping_add(v as u64);
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() as f64 / iters as f64;
        let target_ns = if cfg!(debug_assertions) { 50.0 } else { 5.0 };
        assert_ne!(sink, u64::MAX);
        println!(
            "\nNo-task fast path: {per_call_ns:.2} ns per is_noop call (target < {target_ns:.0} ns)"
        );

        println!("\nDone.");
    }

    #[cfg(not(feature = "collider_consistency"))]
    println!(
        "Enable feature: cargo run --example cccp_vs_nopruner --features collider_consistency"
    );
}
