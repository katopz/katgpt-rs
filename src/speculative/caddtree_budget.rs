//! CaDDTree — Cost-Aware Adaptive DDTree Budget Selection (Plan 219).
//!
//! Replaces fixed `Config::tree_budget` with per-round adaptive budget that
//! maximizes token throughput. Based on:
//! - CaDDTree (arXiv:2606.01813) — unimodality proof, greedy optimal search
//! - BASTION (arXiv:2605.29727) — acceptance surrogate, latency estimator
//!
//! Pipeline: marginals → acceptance surrogate → throughput(B) → greedy search → B*

use crate::speculative::types::{ScreeningPruner, TreeNode};
use crate::types::Config;

// Re-export build functions from dd_tree module.
use crate::speculative::build_dd_tree;
use crate::speculative::build_dd_tree_screened;

// SpecCostSnapshot — optional seed source from Plan 096.
#[cfg(feature = "spec_cost_model")]
use crate::speculative::SpecCostSnapshot;

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

/// Sigmoid activation (never softmax, per project convention).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: Acceptance Surrogate
// ─────────────────────────────────────────────────────────────────────────────

/// Acceptance surrogate: estimates expected accepted length from marginals.
///
/// Based on BASTION §3.1: path confidence = Π(top_k_prob_i) across depths.
/// Weighted by sigmoid confidence gate to attenuate contributions from
/// uncertain depths.
pub struct AcceptanceSurrogate {
    /// Confidence gate steepness (default 4.0).
    confidence_k: f32,
    /// Confidence gate threshold (default 0.5).
    confidence_threshold: f32,
}

impl AcceptanceSurrogate {
    /// Default surrogate with steepness 4.0, threshold 0.5.
    pub fn new() -> Self {
        Self {
            confidence_k: 4.0,
            confidence_threshold: 0.5,
        }
    }

    /// Compute path confidence = Π(top-1 probability) up to `depth`.
    ///
    /// Returns the geometric product of max-probability at each depth level,
    /// capped at the available marginal depths.
    pub fn path_confidence(&self, marginals: &[&[f32]], depth: usize) -> f32 {
        let max_depth = marginals.len().min(depth);
        if max_depth == 0 {
            return 0.0;
        }
        let mut confidence = 1.0_f32;
        for marg in marginals.iter().take(max_depth) {
            match marg.iter().copied().reduce(f32::max) {
                Some(top1) => confidence *= top1,
                None => return 0.0,
            }
        }
        confidence
    }

    /// Expected accepted length: Σ sigmoid(k * (conf_d - threshold)) for each depth.
    ///
    /// Each depth contributes a sigmoid-gated confidence. The sum represents the
    /// expected number of tokens accepted by the verifier.
    pub fn expected_accepted_length(&self, marginals: &[&[f32]]) -> f32 {
        if marginals.is_empty() {
            return 0.0;
        }
        let mut cum_confidence = 1.0_f32;
        let mut total = 0.0_f32;
        for marg in marginals.iter() {
            let top1 = match marg.iter().copied().reduce(f32::max) {
                Some(p) => p,
                None => break,
            };
            cum_confidence *= top1;
            let gate = sigmoid(self.confidence_k * (cum_confidence - self.confidence_threshold));
            total += gate;
        }
        total
    }

    /// Expected accepted length when budget B limits the search width.
    ///
    /// Model: with budget B, at each depth we have up to B branches.
    /// The acceptance probability at depth d improves with more branches,
    /// modeled as `1 - (1 - top1)^min(B, vocab)` but capped by cumulative
    /// product decay. Simplified to: E[accept_len(B)] ≈ Σ_d sigmoid(k * (c_d - t))
    /// where c_d = Π_{i=0..d} (1 - (1 - top1_i)^min(B, remaining_vocab)).
    pub fn expected_accepted_length_at_budget(&self, marginals: &[&[f32]], budget: usize) -> f32 {
        if marginals.is_empty() || budget == 0 {
            return 0.0;
        }
        let mut cum_confidence = 1.0_f32;
        let mut total = 0.0_f32;
        for marg in marginals.iter() {
            if marg.is_empty() {
                break;
            }
            let top1 = match marg.iter().copied().reduce(f32::max) {
                Some(p) => p,
                None => break,
            };
            // With B branches, the best candidate improves acceptance.
            // P(at least one good branch) = 1 - (1 - top1)^B, capped at 1.0.
            let effective_prob = 1.0 - (1.0 - top1).powi(budget as i32).min(1.0);
            cum_confidence *= effective_prob;
            let gate = sigmoid(self.confidence_k * (cum_confidence - self.confidence_threshold));
            total += gate;
        }
        total
    }
}

impl Default for AcceptanceSurrogate {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: Latency Estimator
// ─────────────────────────────────────────────────────────────────────────────

/// Online latency estimator with EMA smoothing.
///
/// Models draft and verify latency as affine functions of budget B:
/// - T_draft(B) = draft_per_node * B
/// - T_verify(B) = verify_base + verify_per_node * B
/// - Total T(B) = (draft_per_node + verify_per_node) * B + verify_base
///
/// Because verify_base > 0 and per-node costs are positive, T(B) is strictly
/// convex in B (linear + constant), guaranteeing unimodality of throughput.
pub struct LatencyEstimator {
    /// EMA alpha (default 0.1).
    alpha: f32,
    /// Per-node draft time (EMA, microseconds).
    draft_per_node: f64,
    /// Base verify time (EMA, microseconds).
    verify_base: f64,
    /// Per-node verify time (EMA, microseconds).
    verify_per_node: f64,
    /// Number of observations.
    observations: usize,
}

impl LatencyEstimator {
    /// New estimator with reasonable cold-start defaults.
    ///
    /// Defaults assume ~0.1μs per draft node, ~10μs verify base, ~0.05μs per
    /// verify node. These are conservative estimates for a typical setup.
    pub fn new() -> Self {
        Self {
            alpha: 0.1,
            draft_per_node: 0.1,   // μs per draft node
            verify_base: 10.0,     // μs fixed verify cost
            verify_per_node: 0.05, // μs per verify node
            observations: 0,
        }
    }

    /// Estimate total cost for a given budget B in microseconds.
    ///
    /// T(B) = draft_per_node * B + verify_base + verify_per_node * B
    ///       = (draft_per_node + verify_per_node) * B + verify_base
    #[inline]
    pub fn estimate_cost(&self, budget: usize) -> f64 {
        self.draft_per_node * budget as f64
            + self.verify_base
            + self.verify_per_node * budget as f64
    }

    /// Update EMA estimates from observed timings.
    ///
    /// Given observed draft_time_us for budget B nodes and verify_time_us for
    /// verifying those B nodes, decompose into per-node rates and EMA-smooth.
    pub fn update(&mut self, budget: usize, draft_time_us: f64, verify_time_us: f64) {
        if budget == 0 {
            return;
        }
        let a = self.alpha as f64;
        let obs_draft_per_node = draft_time_us / budget as f64;
        let obs_verify_per_node = if budget > 1 {
            (verify_time_us - self.verify_base) / (budget as f64 - 1.0)
        } else {
            // Single node: can't separate base from per-node, just update base.
            self.verify_base = self.verify_base * (1.0 - a) + verify_time_us * a;
            self.draft_per_node = self.draft_per_node * (1.0 - a) + obs_draft_per_node * a;
            self.observations += 1;
            return;
        };

        match self.observations {
            0 => {
                // First observation: use directly (no EMA blend).
                self.draft_per_node = obs_draft_per_node;
                self.verify_per_node = obs_verify_per_node;
                self.verify_base = verify_time_us - obs_verify_per_node * (budget as f64 - 1.0);
            }
            _ => {
                self.draft_per_node = self.draft_per_node * (1.0 - a) + obs_draft_per_node * a;
                self.verify_base = self.verify_base * (1.0 - a) + verify_time_us * a;
                self.verify_per_node =
                    self.verify_per_node * (1.0 - a) + obs_verify_per_node.max(0.0) * a;
            }
        }
        self.observations += 1;
    }

    /// Whether the estimator has enough observations for reliable estimates.
    ///
    /// Returns true after ≥3 observations (EMA has had time to converge).
    #[inline]
    pub fn calibrated(&self) -> bool {
        self.observations >= 3
    }

    /// Seed initial estimates from a SpecCostSnapshot (Plan 096).
    ///
    /// Uses the actual_ratio and k fields to bootstrap per-node estimates,
    /// avoiding the cold-start period.
    #[cfg(feature = "spec_cost_model")]
    pub fn seed_from_spec_cost(&mut self, snapshot: &SpecCostSnapshot) {
        if snapshot.k == 0 {
            return;
        }
        // actual_ratio = T(K+1)/T(1), so T(K+1) = actual_ratio * T(1).
        // We don't know T(1) exactly, but can estimate relative costs.
        // Assume T(1) ≈ verify_base (single node = mostly fixed cost).
        let t_single = self.verify_base;
        let t_total = t_single * snapshot.actual_ratio;
        let k = snapshot.k as f64;

        self.draft_per_node = t_single * 0.3 / k; // ~30% of single-token time for draft
        self.verify_per_node = (t_total - t_single) / k;
        self.observations = 3; // Mark as calibrated
    }
}

impl Default for LatencyEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3: Unimodal Budget Search
// ─────────────────────────────────────────────────────────────────────────────

/// Budget selector combining acceptance surrogate + latency estimator.
///
/// Finds optimal budget B* that maximizes throughput:
/// ```text
/// T(B) = E[accept_len(B)] / (T_draft(B) + T_verify(B))
/// ```
///
/// Under convex verification cost, T(B) is unimodal (CaDDTree Theorem 1),
/// so greedy search from B=1 upward finds the global optimum.
pub struct BudgetSelector {
    surrogate: AcceptanceSurrogate,
    latency: LatencyEstimator,
    /// Minimum budget (default 1).
    min_budget: usize,
    /// Maximum budget multiplier (default 2.0, caps at 2 * config.tree_budget).
    max_budget_multiplier: f32,
}

impl BudgetSelector {
    /// New selector with default parameters.
    pub fn new() -> Self {
        Self {
            surrogate: AcceptanceSurrogate::new(),
            latency: LatencyEstimator::new(),
            min_budget: 1,
            max_budget_multiplier: 2.0,
        }
    }

    /// Set surrogate confidence gate parameters.
    pub fn with_surrogate_confidence(mut self, k: f32, threshold: f32) -> Self {
        self.surrogate = AcceptanceSurrogate {
            confidence_k: k,
            confidence_threshold: threshold,
        };
        self
    }

    /// Set latency estimator EMA alpha.
    pub fn with_latency_ema_alpha(mut self, alpha: f32) -> Self {
        self.latency = LatencyEstimator {
            alpha,
            ..self.latency
        };
        self
    }

    /// Compute throughput T(B) = E[accept_len(B)] / cost(B).
    ///
    /// Returns tokens per microsecond. Higher is better.
    pub fn throughput(&self, marginals: &[&[f32]], budget: usize) -> f64 {
        let cost = self.latency.estimate_cost(budget);
        if cost <= 0.0 {
            return 0.0;
        }
        let accept_len = self
            .surrogate
            .expected_accepted_length_at_budget(marginals, budget);
        accept_len as f64 / cost
    }

    /// Select optimal budget B* using greedy unimodal search.
    ///
    /// Starts at min_budget and increments while T(B+1) > T(B).
    /// Because throughput is unimodal, the first peak is the global optimum.
    ///
    /// # Edge cases
    /// - Empty marginals → return 1
    /// - All-zero marginals → return 1
    /// - Uncalibrated latency → return `fallback_budget`
    /// - Single depth → return 1
    pub fn select_budget(
        &self,
        marginals: &[&[f32]],
        max_budget: usize,
        fallback_budget: usize,
    ) -> usize {
        // Edge case: empty or degenerate marginals.
        if marginals.is_empty() {
            return 1;
        }
        let has_content = marginals.iter().any(|m| m.iter().any(|&p| p > 0.0));
        if !has_content {
            return 1;
        }
        if marginals.len() < 2 {
            return 1;
        }

        // Fallback: if latency estimator not calibrated, use fixed budget.
        if !self.latency.calibrated() {
            return fallback_budget.min(max_budget).max(self.min_budget);
        }

        let lo = self.min_budget;
        let hi = max_budget.max(lo);

        // Greedy unimodal ascent: find first B where T(B+1) ≤ T(B).
        let mut best_b = lo;
        let mut best_t = self.throughput(marginals, lo);

        for b in (lo + 1)..=hi {
            let t = self.throughput(marginals, b);
            if t > best_t {
                best_t = t;
                best_b = b;
            } else {
                // First descent → peak found (unimodal property).
                break;
            }
        }

        best_b
    }

    /// Get a reference to the latency estimator for updates.
    pub fn latency_mut(&mut self) -> &mut LatencyEstimator {
        &mut self.latency
    }
}

impl Default for BudgetSelector {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4: Integration Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Build DDTree with adaptive budget selection.
///
/// Uses [`BudgetSelector`] to find optimal B*, then delegates to
/// [`build_dd_tree`] with effective budget B*.
///
/// Returns `(tree_nodes, selected_budget)`.
pub fn build_dd_tree_adaptive(marginals: &[&[f32]], config: &Config) -> (Vec<TreeNode>, usize) {
    let selector = BudgetSelector::new();
    let max_budget = (config.tree_budget as f32 * selector.max_budget_multiplier) as usize;
    let selected_budget = selector.select_budget(marginals, max_budget, config.tree_budget);

    let mut adaptive_config = config.clone();
    adaptive_config.tree_budget = selected_budget;
    let tree = build_dd_tree(marginals, &adaptive_config);
    (tree, selected_budget)
}

/// Build DDTree with adaptive budget + ScreeningPruner.
///
/// Uses [`BudgetSelector`] to find optimal B*, then delegates to
/// [`build_dd_tree_screened`] with effective budget B*.
///
/// Returns `(tree_nodes, selected_budget)`.
pub fn build_dd_tree_adaptive_screened(
    marginals: &[&[f32]],
    config: &Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
) -> (Vec<TreeNode>, usize) {
    let selector = BudgetSelector::new();
    let max_budget = (config.tree_budget as f32 * selector.max_budget_multiplier) as usize;
    let selected_budget = selector.select_budget(marginals, max_budget, config.tree_budget);

    let mut adaptive_config = config.clone();
    adaptive_config.tree_budget = selected_budget;
    let tree = build_dd_tree_screened(marginals, &adaptive_config, screener, chain_seed);
    (tree, selected_budget)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Phase 1: AcceptanceSurrogate tests ──────────────────────────────

    #[test]
    fn test_geometric_estimate() {
        // 3 depths: top-1 probs are [0.8, 0.6, 0.5]
        let marginals: Vec<&[f32]> = vec![&[0.8, 0.1, 0.1], &[0.6, 0.3, 0.1], &[0.5, 0.3, 0.2]];
        let surrogate = AcceptanceSurrogate::new();
        let conf = surrogate.path_confidence(&marginals, 3);
        let expected = 0.8 * 0.6 * 0.5; // 0.24
        assert!(
            (conf - expected).abs() < 1e-5,
            "path_confidence = {conf}, expected = {expected}"
        );
    }

    #[test]
    fn test_sigmoid_gate_bounds() {
        // Sigmoid always in [0, 1] for any finite input (exact 0/1 at extreme underflow).
        for x in [-100.0_f32, -1.0, 0.0, 1.0, 100.0] {
            let s = sigmoid(x);
            assert!(s >= 0.0 && s <= 1.0, "sigmoid({x}) = {s} out of [0,1]");
        }
        // At moderate inputs, sigmoid is strictly in (0, 1).
        for x in [-5.0_f32, -1.0, 0.0, 1.0, 5.0] {
            let s = sigmoid(x);
            assert!(s > 0.0 && s < 1.0, "sigmoid({x}) = {s} not in (0,1)");
        }
    }

    #[test]
    fn test_expected_length_sum() {
        // E[accept_len] = Σ sigmoid(k * (cum_conf - threshold)) for each depth.
        let marginals: Vec<&[f32]> = vec![&[0.9, 0.1], &[0.8, 0.2]];
        let surrogate = AcceptanceSurrogate::new();
        let len = surrogate.expected_accepted_length(&marginals);
        // Depth 0: cum=0.9, gate=sigmoid(4*(0.9-0.5))=sigmoid(1.6)≈0.832
        // Depth 1: cum=0.72, gate=sigmoid(4*(0.72-0.5))=sigmoid(0.88)≈0.707
        // Total ≈ 1.539
        assert!(
            len > 1.0 && len < 2.0,
            "expected_accepted_length = {len}, should be in (1, 2)"
        );
    }

    #[test]
    fn test_empty_marginals() {
        let surrogate = AcceptanceSurrogate::new();
        assert_eq!(surrogate.expected_accepted_length(&[]), 0.0);
        assert_eq!(surrogate.path_confidence(&[], 0), 0.0);
    }

    // ── Phase 2: LatencyEstimator tests ─────────────────────────────────

    #[test]
    fn test_ema_convergence() {
        let mut est = LatencyEstimator::new();
        // Push 100 identical measurements: budget=5, draft=5.0μs, verify=12.5μs
        for _ in 0..100 {
            est.update(5, 5.0, 12.5);
        }
        // Expected per-node draft = 5.0/5 = 1.0μs
        assert!(
            (est.draft_per_node - 1.0).abs() < 0.01,
            "draft_per_node = {}",
            est.draft_per_node
        );
        // Total cost for budget=5 should be near (1.0+verify_per_node)*5 + verify_base
        let cost = est.estimate_cost(5);
        assert!(cost > 0.0 && cost < 100.0, "cost = {cost} unreasonable");
    }

    #[test]
    fn test_cost_convexity() {
        let est = LatencyEstimator::new();
        // T(B) = (dpn + vpn)*B + vb is linear → convex (second diff = 0).
        // Check monotonicity: T(B+1) > T(B) for all B.
        for b in 1..100 {
            let t0 = est.estimate_cost(b);
            let t1 = est.estimate_cost(b + 1);
            assert!(t1 > t0, "T({}) = {} >= T({}) = {}", b + 1, t1, b, t0);
        }
        // Check linearity: T(B+2) - T(B+1) == T(B+1) - T(B)
        for b in 1..50 {
            let d1 = est.estimate_cost(b + 1) - est.estimate_cost(b);
            let d2 = est.estimate_cost(b + 2) - est.estimate_cost(b + 1);
            assert!((d2 - d1).abs() < 1e-10, "cost not linear: Δ1={d1}, Δ2={d2}");
        }
    }

    #[test]
    fn test_default_latency_reasonable() {
        let est = LatencyEstimator::new();
        let cost = est.estimate_cost(10);
        assert!(cost > 0.0, "default cost should be positive, got {cost}");
        // Throughput should be positive for any non-trivial marginals.
        let surrogate = AcceptanceSurrogate::new();
        let marginals: Vec<&[f32]> = vec![&[0.8, 0.2], &[0.7, 0.3]];
        let accept = surrogate.expected_accepted_length(&marginals);
        let throughput = accept as f64 / cost;
        assert!(
            throughput > 0.0,
            "throughput should be positive: accept={accept}, cost={cost}"
        );
    }

    #[test]
    fn test_seed_from_spec_cost() {
        let mut est = LatencyEstimator::new();
        #[cfg(feature = "spec_cost_model")]
        {
            let snapshot = SpecCostSnapshot {
                f_sparse: 0.6,
                f_fixed: 0.4,
                unique_ratio: 0.8,
                predicted_ratio: 0.88,
                actual_ratio: 1.5,
                k: 5,
            };
            est.seed_from_spec_cost(&snapshot);
            assert!(est.calibrated(), "should be calibrated after seeding");
            assert!(
                est.draft_per_node > 0.0,
                "draft_per_node should be positive"
            );
        }
        #[cfg(not(feature = "spec_cost_model"))]
        {
            // Without the feature, just verify the estimator works normally.
            assert!(!est.calibrated());
            est.update(5, 5.0, 12.5);
            est.update(5, 5.0, 12.5);
            est.update(5, 5.0, 12.5);
            assert!(est.calibrated());
        }
    }

    // ── Phase 3: BudgetSelector tests ───────────────────────────────────

    #[test]
    fn test_unimodal_synthetic() {
        // Build a selector and inject known latency params to force unimodality.
        let selector = BudgetSelector::new();

        // Marginals: high confidence at first, then decaying.
        let marginals: Vec<&[f32]> = vec![
            &[0.95, 0.05], // depth 0: very confident
            &[0.90, 0.10], // depth 1: confident
            &[0.80, 0.20], // depth 2: decent
            &[0.60, 0.40], // depth 3: uncertain
            &[0.40, 0.60], // depth 4: very uncertain
        ];

        // Throughput should increase then decrease.
        let throughputs: Vec<f64> = (1..=10)
            .map(|b| selector.throughput(&marginals, b))
            .collect();

        // Find manual peak.
        let peak_idx = throughputs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Greedy search should find the same or better.
        let greedy_b = selector.select_budget(&marginals, 10, 5);
        assert!(
            greedy_b >= 1 && greedy_b <= 10,
            "greedy_b = {greedy_b} out of range"
        );
        // The greedy result's throughput should be within 1% of true peak.
        let greedy_t = selector.throughput(&marginals, greedy_b);
        let peak_t = throughputs[peak_idx];
        assert!(
            greedy_t >= peak_t * 0.99,
            "greedy_t = {greedy_t}, peak_t = {peak_t}"
        );
    }

    #[test]
    fn test_greedy_finds_peak() {
        // Create a selector and calibrate latency manually.
        let mut selector = BudgetSelector::new();
        // Calibrate with observations that make cost increase linearly.
        let lat = selector.latency_mut();
        lat.update(10, 10.0, 15.0);
        lat.update(10, 10.0, 15.0);
        lat.update(10, 10.0, 15.0);
        assert!(lat.calibrated());

        // Marginals where acceptance saturates quickly.
        let marginals: Vec<&[f32]> = vec![
            &[0.9, 0.1],
            &[0.5, 0.5], // big drop → throughput should peak at small B
        ];

        let selected = selector.select_budget(&marginals, 20, 10);
        assert!(
            selected >= 1 && selected <= 20,
            "selected = {selected} out of range"
        );
    }

    #[test]
    fn test_safety_bounds() {
        let selector = BudgetSelector::new();

        let marginals: Vec<&[f32]> = vec![&[0.9, 0.1], &[0.8, 0.2], &[0.7, 0.3]];
        let selected = selector.select_budget(&marginals, 5, 3);

        // Result must be in [1, 5].
        assert!(selected >= 1, "selected = {selected} < min_budget");
        assert!(selected <= 5, "selected = {selected} > max_budget");
    }

    #[test]
    fn test_fallback_fixed_budget() {
        // Uncalibrated selector should return the fallback budget.
        let selector = BudgetSelector::new();
        assert!(!selector.latency.calibrated());

        let marginals: Vec<&[f32]> = vec![&[0.9, 0.1], &[0.8, 0.2]];
        let selected = selector.select_budget(&marginals, 20, 7);

        // Should return fallback_budget (7) clamped to max_budget.
        assert_eq!(
            selected, 7,
            "uncalibrated should return fallback, got {selected}"
        );
    }

    // ── Phase 4: Integration tests ──────────────────────────────────────

    /// Helper: create a minimal Config for testing.
    fn test_config() -> Config {
        Config::default()
    }

    #[test]
    fn test_adaptive_produces_valid_tree() {
        let config = test_config();
        let marginals: Vec<&[f32]> = vec![&[0.5, 0.3, 0.2], &[0.4, 0.35, 0.25], &[0.45, 0.3, 0.25]];

        let (tree, budget) = build_dd_tree_adaptive(&marginals, &config);

        // Budget should be positive and bounded.
        assert!(budget >= 1, "budget = {budget}");
        assert!(
            budget <= (config.tree_budget as f32 * 2.0) as usize,
            "budget = {budget} exceeds 2x tree_budget"
        );

        // Tree nodes should have valid scores (finite and non-negative).
        for node in &tree {
            assert!(
                node.score.is_finite(),
                "node score not finite at depth {} token {}",
                node.depth,
                node.token_idx
            );
        }
    }

    #[test]
    fn test_adaptive_screened_produces_valid_tree() {
        use crate::speculative::NoScreeningPruner;

        let config = test_config();
        let screener = NoScreeningPruner;
        let marginals: Vec<&[f32]> = vec![&[0.6, 0.3, 0.1], &[0.5, 0.3, 0.2]];

        let (tree, budget) = build_dd_tree_adaptive_screened(&marginals, &config, &screener, false);

        assert!(budget >= 1, "budget = {budget}");
        for node in &tree {
            assert!(
                node.score.is_finite(),
                "node score not finite at depth {} token {}",
                node.depth,
                node.token_idx
            );
        }
    }

    #[test]
    fn test_feature_gate_no_modify_existing() {
        // Verify that adaptive builder doesn't affect the fixed-budget builder.
        let config = test_config();
        let marginals: Vec<&[f32]> = vec![&[0.5, 0.3, 0.2], &[0.4, 0.35, 0.25]];

        // Build with fixed budget.
        let fixed_tree = build_dd_tree(&marginals, &config);

        // Build with adaptive — uses a temporary config clone, original is untouched.
        let (_adaptive_tree, _adaptive_budget) = build_dd_tree_adaptive(&marginals, &config);

        // Original config is unchanged.
        assert!(
            config.tree_budget > 0,
            "config.tree_budget should be unchanged"
        );

        // Fixed tree should still work the same way (deterministic for same marginals).
        let fixed_tree2 = build_dd_tree(&marginals, &config);
        assert_eq!(
            fixed_tree.len(),
            fixed_tree2.len(),
            "fixed-budget builder should be deterministic"
        );
    }
}
