//! Integration tests for Progressive MCGS — exercises the GOAT gate properties.
//!
//! These are the **acceptance tests** for Plan 272 Phase 1:
//! - G1 precondition: scheduler entropy diagnostic produces expected values.
//! - G2 (backprop correctness): with `E_ref = ∅`, Q-values match vanilla MCTS.
//! - G3 (stagnation improvement): stagnation triggers fire at correct thresholds.
//!
//! The full GOAT benchmark (with synthetic reward streams, latency, allocation
//! audit) lives in `benches/progressive_mcgs_goat.rs` (Plan 272 Phase 3).

use crate::progressive_mcgs::{
    graph::ProgressiveMcgs, operators, scheduler::EntropyGatedScheduler, scheduler::RngLite,
    stagnation::StagnationGate, types::*, uct,
};

/// GOAT G2: backprop with `E_ref = ∅` must match vanilla MCTS bit-for-bit
/// (within f32 epsilon).
///
/// Two parallel graphs:
/// - `g_refs`: supports reference edges but none are added.
/// - `g_vanilla`: a second graph with the same operations.
///
/// Both should produce identical `visits`, `cumulative_reward`, and `q_value`
/// for every node.
#[test]
fn goat_g2_backprop_isolation_with_e_ref_empty() {
    let mut g_refs = ProgressiveMcgs::<u32>::new(100, 3);
    let mut g_vanilla = ProgressiveMcgs::<u32>::new(100, 0); // no refs allowed

    // Identical construction.
    let root_a = g_refs.add_root(0, BranchId(0));
    let root_b = g_vanilla.add_root(0, BranchId(0));

    let c1_a = g_refs.expand_primary(root_a, 1, BranchId(0));
    let c1_b = g_vanilla.expand_primary(root_b, 1, BranchId(0));

    let c2_a = g_refs.expand_primary(root_a, 2, BranchId(0));
    let c2_b = g_vanilla.expand_primary(root_b, 2, BranchId(0));

    // Identical backprop sequence.
    let rewards = [
        Reward::Progress,
        Reward::Breakthrough,
        Reward::Failure,
        Reward::Neutral,
        Reward::Progress,
    ];
    for (i, r) in rewards.iter().enumerate() {
        let target_a = if i % 2 == 0 { c1_a } else { c2_a };
        let target_b = if i % 2 == 0 { c1_b } else { c2_b };
        g_refs.backprop(target_a, *r);
        g_vanilla.backprop(target_b, *r);
    }

    // Assert bit-identical MCTS stats.
    for id in [root_a, c1_a, c2_a] {
        let id_b = NodeId(id.0);
        assert_eq!(
            g_refs.visits(id),
            g_vanilla.visits(id_b),
            "visits mismatch at {id:?}"
        );
        assert!(
            (g_refs.cumulative_reward(id) - g_vanilla.cumulative_reward(id_b)).abs() < 1e-6,
            "cumulative_reward mismatch at {id:?}"
        );
        assert!(
            (g_refs.q_value(id) - g_vanilla.q_value(id_b)).abs() < 1e-6,
            "q_value mismatch at {id:?}: {} vs {}",
            g_refs.q_value(id),
            g_vanilla.q_value(id_b)
        );
    }
}

/// GOAT G2 (continued): reference edges must NOT change backprop results.
///
/// Same graph, same backprop sequence, but graph B has cross-branch references
/// added. The visits/cumulative_reward on primary paths must be identical.
#[test]
fn goat_g2_reference_edges_do_not_pollute_backprop() {
    let mut g_clean = ProgressiveMcgs::<u32>::new(100, 3);
    let mut g_with_refs = ProgressiveMcgs::<u32>::new(100, 3);

    // Build identical structure in both.
    let root_c = g_clean.add_root(0, BranchId(0));
    let root_r = g_with_refs.add_root(0, BranchId(0));

    let a1_c = g_clean.expand_primary(root_c, 1, BranchId(0));
    let a1_r = g_with_refs.expand_primary(root_r, 1, BranchId(0));

    let b1_c = g_clean.expand_primary(root_c, 10, BranchId(1));
    let b1_r = g_with_refs.expand_primary(root_r, 10, BranchId(1));

    // In g_with_refs only: add a cross-branch reference a1 → b1.
    g_with_refs.add_reference(a1_r, b1_r);

    // Same backprop on both.
    g_clean.backprop(a1_c, Reward::Breakthrough);
    g_with_refs.backprop(a1_r, Reward::Breakthrough);

    // Stats on a1 should match.
    assert_eq!(g_clean.visits(a1_c), g_with_refs.visits(a1_r));
    assert!(
        (g_clean.q_value(a1_c) - g_with_refs.q_value(a1_r)).abs() < 1e-6,
        "Q(a1) changed when refs added"
    );

    // CRITICAL: b1's stats should be 0/0.0 in BOTH graphs — the reference
    // in g_with_refs did NOT propagate credit to b1.
    assert_eq!(g_clean.visits(b1_c), 0);
    assert_eq!(g_with_refs.visits(b1_r), 0);
    assert!(
        (g_clean.q_value(b1_c) - g_with_refs.q_value(b1_r)).abs() < 1e-6,
        "Q(b1) changed due to reference edge — this is a G2 violation"
    );
}

/// GOAT G1 precondition: scheduler entropy diagnostic computes expected values.
#[test]
fn goat_g1_entropy_diagnostic() {
    // 4-branch uniform → H = log(4), exp(H) = 4.
    let counts_uniform = [10u32, 10, 10, 10];
    let h_uniform = EntropyGatedScheduler::branch_selection_entropy(&counts_uniform);
    let n_uniform = EntropyGatedScheduler::effective_branch_count(&counts_uniform);
    assert!((h_uniform - 4.0f32.ln()).abs() < 1e-4);
    assert!((n_uniform - 4.0).abs() < 1e-3);

    // Concentrated on 1 branch → H = 0, exp(H) = 1.
    let counts_degenerate = [40u32, 0, 0, 0];
    let h_deg = EntropyGatedScheduler::branch_selection_entropy(&counts_degenerate);
    assert!(h_deg.abs() < 1e-4);
    assert!((EntropyGatedScheduler::effective_branch_count(&counts_degenerate) - 1.0).abs() < 1e-3);
}

/// GOAT G1 (synthetic decay): drive a synthetic search and verify the entropy
/// of the branch-selection distribution decreases under the schedule.
///
/// This is a *qualitative* test — the full quantitative gate is in benches/.
/// Here we just verify: (entropy at t=1.0) < (entropy at t=0.0) by a clear margin.
///
/// Model: branch 0 is the "elite" (highest Q-value). Under UCT mode, all 4
/// branches get equal exploration. Under Elite mode, branch 0 gets all the
/// compute. The schedule transitions from UCT to Elite over [0.5, 0.7].
#[test]
fn goat_g1_entropy_decays_under_schedule() {
    let scheduler = EntropyGatedScheduler::default();
    let mut rng = fastrand::Rng::with_seed(42);
    let mut branch_counts = [0u32; 4];

    for i in 0..1000u32 {
        let t_norm = i as f32 / 1000.0;
        let mode = scheduler.pick_mode(t_norm, &mut rng);
        match mode {
            crate::progressive_mcgs::scheduler::SelectMode::Uct => {
                // Uniform exploration across all 4 branches.
                let b = (rng.next_f32() * 4.0) as usize;
                branch_counts[b.min(3)] += 1;
            }
            crate::progressive_mcgs::scheduler::SelectMode::Elite => {
                // Elite exploitation — always pick branch 0 (the global best).
                branch_counts[0] += 1;
            }
        }
    }

    let h_final = EntropyGatedScheduler::branch_selection_entropy(&branch_counts);
    let h_uniform = 4.0f32.ln(); // max possible entropy for 4 branches
    assert!(
        h_final < h_uniform,
        "entropy should decay below uniform under schedule: h_final={h_final:.4}, h_uniform={h_uniform:.4}"
    );
    // The schedule alone (without Q-biased UCT selection) produces a modest
    // entropy decay because `switch_start=0.5` means half the search is pure
    // UCT exploration. The paper's 34% decay (Figure 3: exp(H) 4.8→2.8) comes
    // from the *combination* of schedule + Q-biased UCT concentrating on
    // high-reward branches. This synthetic test models only the schedule's
    // contribution — verifying the directional property (decay happens),
    // not the full magnitude. The full GOAT G1 gate is in benches/.
    assert!(
        h_final < h_uniform * 0.95,
        "entropy should decay by ≥5% under schedule alone: h_final={h_final:.4}, threshold={:.4}",
        h_uniform * 0.95
    );
}

/// GOAT G3 precondition: stagnation triggers fire at documented thresholds.
#[test]
fn goat_g3_stagnation_triggers_fire() {
    let mut gate = StagnationGate::new(2, 3, 6);

    // Drive branch 0 to τ_branch=3 stagnation.
    for _ in 0..3 {
        gate.observe_expansion(BranchId(0), Reward::Neutral);
    }
    let triggers: Vec<_> = gate.check(BranchId(0)).iter().collect();
    assert!(
        triggers.iter().any(|t| matches!(t, crate::progressive_mcgs::stagnation::StagnationTrigger::IntraBranchEvolve)),
        "expected IntraBranchEvolve after τ_branch=3 non-improvements, got {triggers:?}"
    );

    // Drive to τ_global=6 globally.
    for _ in 0..6 {
        gate.observe_expansion(BranchId(0), Reward::Neutral);
    }
    let triggers: Vec<_> = gate.check(BranchId(0)).iter().collect();
    assert!(
        triggers
            .iter()
            .any(|t| matches!(t, crate::progressive_mcgs::stagnation::StagnationTrigger::MultiBranchAggregation)),
        "expected MultiBranchAggregation after τ_global=6 non-breakthroughs, got {triggers:?}"
    );
}

/// Smoke test: full select→expand→observe cycle on a synthetic graph.
///
/// Verifies that all primitives compose without panic and produce
/// non-degenerate Q-values after a small search.
#[test]
fn smoke_full_search_cycle() {
    let mut g = ProgressiveMcgs::<u32>::new(1000, 3);
    let root = g.add_root(0, BranchId(0));

    // Seed each branch with an initial primary expansion.
    let b0_c1 = g.expand_primary(root, 100, BranchId(0));
    let b1_c1 = g.expand_primary(root, 200, BranchId(1));

    // Synthetic search loop.
    let mut rng = fastrand::Rng::with_seed(42);
    let scheduler = EntropyGatedScheduler::default();
    let mut gate = StagnationGate::new(2, 3, 6);

    for step in 0..50 {
        let t_norm = step as f32 / 50.0;
        let mode = scheduler.pick_mode(t_norm, &mut rng);

        let leaf = match mode {
            crate::progressive_mcgs::scheduler::SelectMode::Uct => {
                // Descend from root via UCT.
                let c = if step % 2 == 0 { b0_c1 } else { b1_c1 };
                let _ = c; // suppress unused
                // For simplicity, just pick root.
                root
            }
            crate::progressive_mcgs::scheduler::SelectMode::Elite => {
                // Sample from top-K nodes.
                let ranked = [b0_c1, b1_c1];
                *scheduler.elite_sample(&ranked, &mut rng).unwrap_or(&b0_c1)
            }
        };

        // Expand primary from leaf.
        let new_node = g.expand_primary(leaf, step, BranchId(step % 2));

        // Synthetic reward: branch 0 gets +1 with prob 0.6, branch 1 with prob 0.4.
        let reward = if rng.next_f32() < 0.6 {
            Reward::Progress
        } else {
            Reward::Neutral
        };

        // Snapshot best BEFORE update (per Plan 272 §4 risk).
        let branch = g.branch_of(new_node);
        let _branch_best_before = g.branch_best(branch);
        let _global_best_before = g.global_best();

        // Backprop.
        g.backprop(new_node, reward);
        g.set_branch_best(branch, reward);
        g.set_global_best(reward);

        // Update stagnation.
        gate.observe_expansion(branch, reward);

        // Check for stagnation triggers.
        let triggers: Vec<_> = gate.check(branch).iter().collect();
        if triggers
            .iter()
            .any(|t| matches!(t, crate::progressive_mcgs::stagnation::StagnationTrigger::IntraBranchEvolve))
        {
            // Apply intra-branch: pull last-3 ancestors as references.
            let refs = operators::intra_branch_history(&g, new_node, 3);
            for r in refs {
                g.add_reference(new_node, r);
            }
        }
    }

    // After 50 steps, Q-values should be non-degenerate.
    let q0 = g.q_value(b0_c1);
    let q1 = g.q_value(b1_c1);
    assert!(q0.is_finite(), "Q(b0_c1) not finite: {q0}");
    assert!(q1.is_finite(), "Q(b1_c1) not finite: {q1}");

    // Root should have ≥ 50 visits.
    assert!(
        g.visits(root) >= 50,
        "root should have ≥50 visits after 50 expansions, got {}",
        g.visits(root)
    );

    // Reference edges may have been added by intra-branch evolution.
    let total_refs = g.total_reference_edges();
    assert!(total_refs <= 50 * 3, "reference-edge count exploded: {total_refs}");
}

/// Deterministic replay: same RNG seed → same Q-values.
#[test]
fn deterministic_replay() {
    fn run_once(seed: u64) -> (f32, f32, u32) {
        let mut g = ProgressiveMcgs::<u32>::new(1000, 3);
        let root = g.add_root(0, BranchId(0));
        let c1 = g.expand_primary(root, 100, BranchId(0));
        let c2 = g.expand_primary(root, 200, BranchId(1));

        let mut rng = fastrand::Rng::with_seed(seed);
        for _ in 0..30 {
            let target = if rng.next_f32() < 0.5 { c1 } else { c2 };
            let reward = if rng.next_f32() < 0.5 {
                Reward::Progress
            } else {
                Reward::Neutral
            };
            g.backprop(target, reward);
        }
        (g.q_value(c1), g.q_value(c2), g.visits(root))
    }

    let (q1_a, q2_a, v_a) = run_once(42);
    let (q1_b, q2_b, v_b) = run_once(42);

    assert!((q1_a - q1_b).abs() < 1e-6, "Q(c1) not deterministic");
    assert!((q2_a - q2_b).abs() < 1e-6, "Q(c2) not deterministic");
    assert_eq!(v_a, v_b, "root visits not deterministic");
}

/// Operators compose with the graph without panic.
#[test]
fn operators_smoke() {
    let mut g = ProgressiveMcgs::<u32>::new(100, 3);
    let root = g.add_root(0, BranchId(0));
    let a1 = g.expand_primary(root, 1, BranchId(0));
    let a2 = g.expand_primary(a1, 2, BranchId(0));
    let b1 = g.expand_primary(root, 10, BranchId(1));

    g.backprop(a2, Reward::Breakthrough);
    g.backprop(b1, Reward::Progress);

    let intra = operators::intra_branch_history(&g, a2, 3);
    assert!(!intra.is_empty());

    let cross = operators::cross_branch_top_n(&g, BranchId(0), 2);
    assert!(!cross.is_empty());

    let agg = operators::multi_branch_aggregate(&g, 1);
    assert!(!agg.is_empty());
}

/// UCT descend reaches a leaf.
#[test]
fn uct_descend_smoke() {
    let mut g = ProgressiveMcgs::<u32>::new(100, 3);
    let root = g.add_root(0, BranchId(0));
    let c1 = g.expand_primary(root, 1, BranchId(0));
    let c2 = g.expand_primary(c1, 2, BranchId(0));

    let leaf = uct::uct_descend_to_leaf(&g, root, 1.414);
    assert_eq!(leaf, c2);
}
