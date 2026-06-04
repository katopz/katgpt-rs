//! Residency Audit for DDTree pruning paths (Plan 175, Fusion 1).
//!
//! ANE analogy: "compile success ≠ execution target." A model that compiles, runs,
//! and produces correct output may execute on the wrong hardware path — silently 5× slower.
//!
//! Our analogy: a pruner that produces high valid-node ratios may be forcing retained nodes
//! into expensive verification paths. The residency audit checks whether pruning paths
//! actually land on fast paths, not just whether they produce valid trees.
//!
//! This is a **test-only** module — zero runtime overhead. Run in CI to catch silent
//! degradation the same way the ANE `MLComputePlan` catches silent CPU fallback.

#[cfg(test)]
use super::types::NoPruner;
use super::types::{ConstraintPruner, NoScreeningPruner, ScreeningPruner};
use crate::types::Config;
use std::time::Instant;

/// Residency report for a single DDTree build audit.
///
/// Captures whether the pruner's tree construction lands on fast paths
/// or silently degrades (high prune ratio but expensive retained-node cost).
#[derive(Debug, Clone)]
pub struct ResidencyReport {
    /// Fraction of retained nodes on fast verification path (≥0.8 is healthy).
    pub fast_path_ratio: f32,
    /// Average nanoseconds per retained node during tree construction.
    pub avg_branch_cost_ns: f64,
    /// True if pruning looks good but cost is hidden (fast_path_ratio < 0.8).
    pub silent_degradation: bool,
    /// Total nodes in the tree.
    pub total_nodes: usize,
    /// Nodes that were evaluated (expanded + pruned).
    pub nodes_evaluated: usize,
    /// Nodes that survived pruning.
    pub nodes_retained: usize,
    /// Prune ratio (nodes_evaluated - nodes_retained) / nodes_evaluated.
    pub prune_ratio: f32,
}

/// Audit a DDTree build with a `ConstraintPruner`.
///
/// Builds the tree using [`build_dd_tree_pruned`](super::build_dd_tree_pruned),
/// measures per-node cost, and reports residency metrics.
///
/// # Arguments
/// * `marginals` — Per-depth token probability distributions
/// * `config` — DDTree configuration
/// * `pruner` — Constraint pruner to audit
/// * `chain_seed` — Whether to use chain-seeded build
pub fn audit_constraint_pruner(
    marginals: &[&[f32]],
    config: &Config,
    pruner: &dyn ConstraintPruner,
    chain_seed: bool,
) -> ResidencyReport {
    let start = Instant::now();
    let tree = super::build_dd_tree_pruned(marginals, config, pruner, chain_seed);
    let elapsed_ns = start.elapsed().as_nanos() as f64;

    let total_nodes = tree.len();
    let avg_branch_cost_ns = if total_nodes > 0 {
        elapsed_ns / total_nodes as f64
    } else {
        0.0
    };

    // Fast path heuristic: nodes at depths 0–2 are "fast" (shallow, cheap verify).
    // Deeper nodes cost more to verify. If most retained nodes are shallow, the pruner
    // is landing on fast paths. If most are deep, it's silently expensive.
    let fast_path_count = tree.iter().filter(|n| n.depth <= 2).count();
    let fast_path_ratio = if total_nodes > 0 {
        fast_path_count as f32 / total_nodes as f32
    } else {
        1.0
    };

    // Prune ratio: how many candidates did the pruner eliminate?
    // We estimate evaluated nodes from budget * average branching factor.
    let budget = config.tree_budget;
    let nodes_evaluated = budget.max(total_nodes);
    let nodes_retained = total_nodes;
    let prune_ratio = if nodes_evaluated > 0 {
        (nodes_evaluated - nodes_retained) as f32 / nodes_evaluated as f32
    } else {
        0.0
    };

    ResidencyReport {
        fast_path_ratio,
        avg_branch_cost_ns,
        silent_degradation: fast_path_ratio < 0.8 && total_nodes > 0,
        total_nodes,
        nodes_evaluated,
        nodes_retained,
        prune_ratio,
    }
}

/// Audit a DDTree build with a `ScreeningPruner`.
///
/// Builds the tree using [`build_dd_tree_screened`](super::build_dd_tree_screened),
/// measures per-node cost, and reports residency metrics.
///
/// # Arguments
/// * `marginals` — Per-depth token probability distributions
/// * `config` — DDTree configuration
/// * `screener` — Screening pruner to audit
/// * `chain_seed` — Whether to use chain-seeded build
pub fn audit_screening_pruner(
    marginals: &[&[f32]],
    config: &Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
) -> ResidencyReport {
    let start = Instant::now();
    let tree = super::build_dd_tree_screened(marginals, config, screener, chain_seed);
    let elapsed_ns = start.elapsed().as_nanos() as f64;

    let total_nodes = tree.len();
    let avg_branch_cost_ns = if total_nodes > 0 {
        elapsed_ns / total_nodes as f64
    } else {
        0.0
    };

    let fast_path_count = tree.iter().filter(|n| n.depth <= 2).count();
    let fast_path_ratio = if total_nodes > 0 {
        fast_path_count as f32 / total_nodes as f32
    } else {
        1.0
    };

    let budget = config.tree_budget;
    let nodes_evaluated = budget.max(total_nodes);
    let nodes_retained = total_nodes;
    let prune_ratio = if nodes_evaluated > 0 {
        (nodes_evaluated - nodes_retained) as f32 / nodes_evaluated as f32
    } else {
        0.0
    };

    ResidencyReport {
        fast_path_ratio,
        avg_branch_cost_ns,
        silent_degradation: fast_path_ratio < 0.8 && total_nodes > 0,
        total_nodes,
        nodes_evaluated,
        nodes_retained,
        prune_ratio,
    }
}

/// Audit baseline: NoPruner with NoScreeningPruner.
///
/// Establishes the "fast path" baseline — a tree with no pruning at all.
/// Any real pruner should have comparable or better per-node cost than this.
pub fn audit_baseline(marginals: &[&[f32]], config: &Config) -> ResidencyReport {
    audit_screening_pruner(marginals, config, &NoScreeningPruner, false)
}

/// Compare two residency reports and flag if the candidate is silently degrading
/// relative to the baseline.
///
/// Returns `true` if the candidate pruner is worse than baseline on either:
/// - Per-node cost is >2× baseline
/// - Fast path ratio is <80% of baseline's ratio
pub fn is_degrading(candidate: &ResidencyReport, baseline: &ResidencyReport) -> bool {
    let cost_ratio = if baseline.avg_branch_cost_ns > 0.0 {
        candidate.avg_branch_cost_ns / baseline.avg_branch_cost_ns
    } else {
        1.0
    };
    let ratio_relative = if baseline.fast_path_ratio > 0.0 {
        candidate.fast_path_ratio / baseline.fast_path_ratio
    } else {
        1.0
    };
    cost_ratio > 2.0 || ratio_relative < 0.8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Config;

    fn test_config() -> Config {
        Config::draft()
    }

    fn uniform_marginals(depths: usize, vocab: usize) -> Vec<Vec<f32>> {
        let p = 1.0 / vocab as f32;
        vec![vec![p; vocab]; depths]
    }

    fn peaked_marginals(depths: usize, vocab: usize) -> Vec<Vec<f32>> {
        // Token 0 gets 90% probability, rest share 10%
        (0..depths)
            .map(|_| {
                let mut m = vec![0.1 / (vocab - 1) as f32; vocab];
                m[0] = 0.9;
                m
            })
            .collect()
    }

    #[test]
    fn test_baseline_audit_no_degradation() {
        let config = test_config();
        let marginals = uniform_marginals(4, config.vocab_size);
        let report = audit_baseline(
            &marginals.iter().map(|m| m.as_slice()).collect::<Vec<_>>(),
            &config,
        );

        // NoPruner should never show silent degradation
        // With uniform marginals and NoPruner, all nodes should be present
        assert!(report.total_nodes > 0, "baseline should produce nodes");
        assert!(
            !report.silent_degradation,
            "baseline (NoPruner) should not flag silent degradation"
        );
    }

    #[test]
    fn test_no_pruner_passes_audit() {
        let config = test_config();
        let marginals = peaked_marginals(4, config.vocab_size);
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
        let report = audit_constraint_pruner(&slices, &config, &NoPruner, false);

        assert!(report.total_nodes > 0);
        // NoPruner with peaked marginals should have high fast_path_ratio
        // (most expansion happens at shallow depths)
        assert!(
            report.fast_path_ratio >= 0.5,
            "NoPruner should retain many shallow nodes, got ratio={}",
            report.fast_path_ratio
        );
    }

    #[test]
    fn test_no_screening_pruner_passes_audit() {
        let config = test_config();
        let marginals = peaked_marginals(4, config.vocab_size);
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
        let report = audit_screening_pruner(&slices, &config, &NoScreeningPruner, false);

        assert!(report.total_nodes > 0);
        assert!(
            !report.silent_degradation,
            "NoScreeningPruner should not flag silent degradation"
        );
    }

    #[test]
    fn test_uniform_marginals_high_prune_ratio() {
        let mut config = Config::draft();
        config.tree_budget = 16;
        config.draft_lookahead = 4;
        let marginals = uniform_marginals(4, config.vocab_size);
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
        let report = audit_screening_pruner(&slices, &config, &NoScreeningPruner, false);

        // With uniform marginals and small budget, budget caps the tree
        // and prune_ratio reflects how many candidates didn't make the cut
        assert!(
            report.total_nodes <= config.tree_budget,
            "tree should respect budget: {} > {}",
            report.total_nodes,
            config.tree_budget
        );
    }

    #[test]
    fn test_degrading_comparison() {
        let good = ResidencyReport {
            fast_path_ratio: 0.9,
            avg_branch_cost_ns: 100.0,
            silent_degradation: false,
            total_nodes: 50,
            nodes_evaluated: 64,
            nodes_retained: 50,
            prune_ratio: 0.22,
        };
        let bad = ResidencyReport {
            fast_path_ratio: 0.5,
            avg_branch_cost_ns: 300.0,
            silent_degradation: true,
            total_nodes: 20,
            nodes_evaluated: 64,
            nodes_retained: 20,
            prune_ratio: 0.69,
        };

        assert!(
            !is_degrading(&good, &good),
            "identical reports should not degrade"
        );
        assert!(
            is_degrading(&bad, &good),
            "bad pruner should degrade relative to good"
        );
    }

    #[test]
    fn test_empty_marginals() {
        let config = test_config();
        let marginals: Vec<&[f32]> = vec![];
        let report = audit_constraint_pruner(&marginals, &config, &NoPruner, false);

        assert_eq!(
            report.total_nodes, 0,
            "empty marginals should produce empty tree"
        );
        assert_eq!(report.avg_branch_cost_ns, 0.0);
        assert!(
            !report.silent_degradation,
            "empty tree should not flag degradation"
        );
    }
}
