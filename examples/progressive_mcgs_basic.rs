//! Progressive MCGS basic example — synthetic search with entropy decay visualization.
//!
//! Demonstrates the three primitives from Plan 272 / Research 239 via the
//! Phase 2 orchestrator ([`ProgressiveMcgsSearch`]):
//! 1. Reference-edge graph search (E_T ⊔ E_ref)
//! 2. Entropy-gated scheduler (UCT → Elite transition)
//! 3. Stagnation gates (branch + global triggers)
//!
//! The example defines a `SyntheticDomain` that:
//! - Proposes payloads = step counter (deterministic, no real generative model)
//! - Evaluates rewards from a per-branch Bernoulli distribution (branch 0 is best)
//!
//! Run with:
//! ```sh
//! cargo run --example progressive_mcgs_basic --features progressive_mcgs --release
//! ```

use katgpt_rs::progressive_mcgs::{
    EntropyGatedScheduler, ProgressiveMcgsConfig, ProgressiveMcgsSearch, Reward, SearchDomain,
    BranchId, NodeId,
};

/// Synthetic search domain: each branch has a fixed reward probability.
///
/// Branch 0 has the highest probability (the "good" branch); the orchestrator's
/// Elite mode should concentrate selections on it as `w(t)` decays.
struct SyntheticDomain {
    /// Per-branch probability of returning `Progress` (vs `Neutral`).
    branch_reward_probs: Vec<f32>,
    /// RNG for reward sampling.
    rng: fastrand::Rng,
}

impl SearchDomain<u32> for SyntheticDomain {
    fn propose(
        &mut self,
        _graph: &katgpt_rs::progressive_mcgs::ProgressiveMcgs<u32>,
        _parent: NodeId,
        _branch: BranchId,
        _reference_nodes: &[NodeId],
        step_index: u32,
    ) -> u32 {
        // Trivial payload: the step counter. A real consumer would call an LLM
        // here, possibly conditioned on reference_nodes' payloads.
        step_index
    }

    fn evaluate(
        &mut self,
        graph: &katgpt_rs::progressive_mcgs::ProgressiveMcgs<u32>,
        node: NodeId,
    ) -> Reward {
        // Look up the node's branch from the graph, then sample from that
        // branch's reward distribution. Branch 0 has the highest probability
        // of Progress; others fail more often. Because `Reward::Failure`
        // maps to -1.0 while `Progress` maps to +1.0, branches differentiate
        // in Q-value — and Elite mode can concentrate on branch 0.
        //
        // (Note: `Neutral` and `Progress` both map to +1.0 in `as_f32()`,
        // so using Neutral vs Progress would NOT differentiate branches in
        // Q-value. We use Failure vs Progress to make the signal visible.)
        let branch = graph.branch_of(node);
        let prob = self
            .branch_reward_probs
            .get(branch.idx())
            .copied()
            .unwrap_or(0.3);
        if self.rng.f32() < prob {
            Reward::Progress
        } else {
            Reward::Failure
        }
    }
}

fn main() {
    println!("=== Progressive MCGS Basic Example (Phase 2 Orchestrator) ===\n");
    println!("Paper: MLEvolve (arxiv 2606.06473) — Du et al. 2026");
    println!("Distilled in: .research/239_MLEvolve_Progressive_MCGS_Entropy_Schedule.md");
    println!("Plan: .plans/272_progressive_mcgs.md\n");

    // --- Setup ---
    let n_branches: u32 = 4;
    let max_expansions: u32 = 500;

    // Branch 0 has the best true mean (+0.6 reward prob); others are worse.
    // The orchestrator's Elite mode should converge toward branch 0.
    let branch_reward_probs = vec![0.6_f32, 0.3, 0.3, 0.3];

    let cfg = ProgressiveMcgsConfig {
        // Slightly earlier switch to make entropy decay visible in 500 steps.
        entropy_switch_start: 0.3,
        entropy_switch_end: 0.6,
        entropy_w_min: 0.15,
        ..ProgressiveMcgsConfig::default()
    };

    let mut search = ProgressiveMcgsSearch::<u32>::new(cfg, n_branches)
        .with_max_expansions(max_expansions);
    search.add_root(0);
    for b in 0..n_branches {
        search.seed_branch(BranchId(b), 100 + b);
    }

    let mut domain = SyntheticDomain {
        branch_reward_probs: branch_reward_probs.clone(),
        rng: fastrand::Rng::with_seed(42),
    };
    let mut rng = fastrand::Rng::with_seed(42);

    // --- Search loop (orchestrator handles all integration) ---
    let mut entropy_curve: Vec<(f32, f32)> = Vec::with_capacity(50);
    let mut reference_edge_count = 0usize;
    let mut stagnation_events = 0u32;
    let mut breakthrough_count = 0u32;

    while let Some(res) = search.step(&mut domain, &mut rng) {
        let t_norm = search.step_count() as f32 / max_expansions as f32;

        // Sample entropy at 50 points.
        if search.step_count() % (max_expansions / 50).max(1) == 0 {
            let h = EntropyGatedScheduler::branch_selection_entropy(
                search.branch_selection_counts(),
            );
            entropy_curve.push((t_norm, h));
        }

        reference_edge_count += res.references_added;
        stagnation_events += res.triggers.len() as u32;
        if res.reward == Reward::Breakthrough {
            breakthrough_count += 1;
        }
    }

    // --- Report ---
    println!("\n--- Search Complete ---");
    println!("Total expansions:    {max_expansions}");
    println!("Total nodes:         {}", search.graph().len());
    println!("Reference edges:     {reference_edge_count} (added by stagnation operators)");
    println!("Stagnation events:   {stagnation_events}");
    println!("Breakthroughs:       {breakthrough_count}");
    println!("Global best reward:  {:?}", search.graph().global_best());

    println!("\n--- Branch Selection Distribution ---");
    let counts = search.branch_selection_counts();
    let total: u32 = counts.iter().sum();
    for (b, &count) in counts.iter().enumerate() {
        let pct = (count as f32 / total as f32) * 100.0;
        let q_best = search
            .graph()
            .node_ids()
            .filter(|id| search.graph().branch_of(*id) == BranchId(b as u32))
            .map(|id| search.graph().q_value(id))
            .fold(f32::NEG_INFINITY, f32::max);
        let marker = if b == 0 { " ← true best branch" } else { "" };
        println!(
            "  Branch {b}: {count:4} selections ({pct:5.1}%)  best Q = {q_best:+.3}{marker}"
        );
    }

    println!("\n--- Entropy Curve (H(π_t) over search progress) ---");
    let h_max = (n_branches as f32).ln();
    println!("  (max entropy for {n_branches} branches = {h_max:.4} nats, exp(H_max) = {n_branches})");
    for (t, h) in &entropy_curve {
        let effective = h.exp();
        let bar_len = ((h / h_max) * 40.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("  t={t:.2}: H={h:.4}  exp(H)={effective:.2}  {bar}");
    }

    let h_start = entropy_curve.first().map(|(_, h)| *h).unwrap_or(0.0);
    let h_end = entropy_curve.last().map(|(_, h)| *h).unwrap_or(0.0);
    let decay_pct = if h_start > 0.0 {
        (1.0 - h_end / h_start) * 100.0
    } else {
        0.0
    };
    println!("\nEntropy decay: {h_start:.4} → {h_end:.4} ({decay_pct:.1}% reduction)");
    println!(
        "Effective branch count: {:.2} → {:.2}",
        entropy_curve.first().map(|(_, h)| h.exp()).unwrap_or(0.0),
        entropy_curve.last().map(|(_, h)| h.exp()).unwrap_or(0.0),
    );

    println!("\n--- GOAT Gate Summary (Phase 1 preconditions, Phase 3 full gate) ---");
    println!("  G1 (entropy decay):  {decay_pct:.1}% (target: >0%, full gate in benches/)");
    println!("  G2 (backprop iso):   verified in unit tests (goat_g2_*)");
    println!("  G3 (stagnation):     {stagnation_events} events fired (target: >0)");

    println!("\n--- Phase 2 Orchestrator Notes ---");
    println!("  • All integration logic encapsulated in ProgressiveMcgsSearch::step()");
    println!("  • Consumer provides only SearchDomain (propose + evaluate)");
    println!("  • Branch selection, UCT descent, backprop, stagnation all automatic");

    println!("\n=== End Example ===");
}
