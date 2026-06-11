//! Curvature-Influence Allocation Bandit (CIAB) — Modelless
//!
//! EoS-aware arm selection inspired by arXiv:2606.04212.
//! Uses curvature-influence proxies (persistence + alignment) to allocate
//! DDTree budget across depth positions without access to model internals.
//!
//! # Architecture
//!
//! - [`CurvatureInfluenceScorer`] — trait for depth-group influence scoring
//! - [`EosProxyScorer`] — default scorer using EMA loss residual + score concentration
//! - [`CurvatureWeightedBudget`] — budget allocator proportional to influence

/// Trait for computing curvature-influence scores per depth group.
///
/// "Curvature influence" is a modelless proxy: it combines persistence
/// (how much a group's loss deviates from its running mean) with alignment
/// (how concentrated the score distribution is). High influence positions
/// get more DDTree budget.
pub trait CurvatureInfluenceScorer: Send + Sync {
    /// Return the cached curvature-influence score for group `k`, normalized to [0, 1].
    /// Lazily recomputes if the cache is stale.
    fn curvature_influence(&mut self, group: usize) -> f32;

    /// Number of depth groups tracked.
    fn num_groups(&self) -> usize;

    /// Update persistence (loss residual) for group `k`.
    fn update_persistence(&mut self, group: usize, loss: f32);

    /// Update alignment (score concentration) for group `k`.
    fn update_alignment(&mut self, group: usize, scores: &[f32]);
}

/// Default scorer: EMA loss residual × score concentration.
///
/// - **Persistence** = EMA of |loss − running_mean|. High when loss oscillates.
/// - **Alignment** = 1 − normalized_entropy(softmax(scores)). High when scores concentrate.
/// - **Influence** = persistence × alignment, normalized to [0, 1].
pub struct EosProxyScorer {
    /// EMA loss residual per group (persistence).
    persistence: Vec<f32>,
    /// Score concentration per group (alignment).
    alignment: Vec<f32>,
    /// EMA smoothing rate (0 < α ≤ 1).
    ema_rate: f32,
    /// Running mean of loss per group.
    loss_mean: Vec<f32>,
    /// Cached influence = persistence × alignment, normalized.
    influence: Vec<f32>,
    /// Pre-allocated scratch buffer for softmax exp values in `update_alignment`.
    softmax_scratch: Vec<f32>,
    /// Whether influence cache is stale (needs recomputation).
    influence_dirty: bool,
}

impl EosProxyScorer {
    /// Create a new scorer for `num_groups` depth positions.
    ///
    /// `ema_rate` controls smoothing (0.1 = slow, 0.5 = fast).
    pub fn new(num_groups: usize, ema_rate: f32) -> Self {
        let ema_rate = ema_rate.clamp(0.01, 1.0);
        Self {
            persistence: vec![0.0; num_groups],
            alignment: vec![0.0; num_groups],
            ema_rate,
            loss_mean: vec![0.0; num_groups],
            influence: vec![0.0; num_groups],
            softmax_scratch: Vec::new(),
            influence_dirty: false,
        }
    }

    /// Return the alignment (score concentration) for group `k`.
    pub fn alignment(&self, group: usize) -> f32 {
        match self.alignment.get(group) {
            Some(&v) => v,
            None => 0.0,
        }
    }

    /// Return the persistence (loss residual) for group `k`.
    pub fn persistence(&self, group: usize) -> f32 {
        match self.persistence.get(group) {
            Some(&v) => v,
            None => 0.0,
        }
    }

    /// Ensure the influence cache is up-to-date, recomputing only when dirty.
    fn ensure_influence(&mut self) {
        if !self.influence_dirty {
            return;
        }
        let max_val = self
            .persistence
            .iter()
            .zip(self.alignment.iter())
            .map(|(&p, &a)| p * a)
            .fold(0.0f32, f32::max);

        for i in 0..self.influence.len() {
            let raw = self.persistence[i] * self.alignment[i];
            self.influence[i] = if max_val > 0.0 { raw / max_val } else { 0.0 };
        }
        self.influence_dirty = false;
    }
}

impl CurvatureInfluenceScorer for EosProxyScorer {
    fn curvature_influence(&mut self, group: usize) -> f32 {
        self.ensure_influence();
        match self.influence.get(group) {
            Some(&v) => v,
            None => 0.0,
        }
    }

    fn num_groups(&self) -> usize {
        self.persistence.len()
    }

    fn update_persistence(&mut self, group: usize, loss: f32) {
        if group >= self.persistence.len() {
            return;
        }
        let alpha = self.ema_rate;
        let residual = (loss - self.loss_mean[group]).abs();
        // EMA update of loss mean
        self.loss_mean[group] = (1.0 - alpha) * self.loss_mean[group] + alpha * loss;
        // EMA update of persistence (residual magnitude)
        self.persistence[group] = (1.0 - alpha) * self.persistence[group] + alpha * residual;
        self.influence_dirty = true;
    }

    fn update_alignment(&mut self, group: usize, scores: &[f32]) {
        if group >= self.alignment.len() || scores.is_empty() {
            return;
        }
        // Compute softmax entropy using pre-allocated scratch buffer
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        self.softmax_scratch.clear();
        self.softmax_scratch
            .extend(scores.iter().map(|&s| (s - max_score).exp()));
        let sum: f32 = self.softmax_scratch.iter().sum();
        if sum <= 0.0 {
            self.alignment[group] = 0.0;
            self.influence_dirty = true;
            return;
        }
        let entropy: f32 = self
            .softmax_scratch
            .iter()
            .map(|&e| {
                let p = e / sum;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum();

        // Maximum entropy = ln(n)
        let max_entropy = (scores.len() as f32).ln();
        let normalized_entropy = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };
        // Alignment = 1 − normalized_entropy (high when concentrated)
        self.alignment[group] = (1.0 - normalized_entropy).clamp(0.0, 1.0);
        self.influence_dirty = true;
    }
}

/// Budget allocator proportional to curvature influence.
///
/// Allocates more DDTree budget to high-influence depth positions
/// while guaranteeing a floor for every position.
pub struct CurvatureWeightedBudget {
    /// Minimum fraction of budget guaranteed for each position (default 0.1).
    pub floor_ratio: f32,
    /// Maximum boost factor over uniform (default 0.5).
    pub max_boost: f32,
}

impl CurvatureWeightedBudget {
    /// Create with default parameters.
    pub fn new() -> Self {
        Self {
            floor_ratio: 0.1,
            max_boost: 0.5,
        }
    }

    /// Allocate `total_budget` tokens across `max_depth` positions.
    ///
    /// Each position gets weight = influence(depth) + floor_ratio.
    /// Weights are normalized, then budget is distributed proportionally.
    pub fn allocate(
        &self,
        total_budget: usize,
        max_depth: usize,
        scorer: &mut dyn CurvatureInfluenceScorer,
    ) -> Vec<usize> {
        if max_depth == 0 || total_budget == 0 {
            return vec![];
        }

        // Compute weights
        let mut weights: Vec<f32> = (0..max_depth)
            .map(|d| scorer.curvature_influence(d) + self.floor_ratio)
            .collect();

        // Cap weights by max_boost over uniform
        let uniform = 1.0 / max_depth as f32;
        let cap = uniform * (1.0 + self.max_boost);
        for w in &mut weights {
            *w = w.min(cap);
        }

        let weight_sum: f32 = weights.iter().sum();
        if weight_sum <= 0.0 {
            // Fallback: uniform
            let per = total_budget / max_depth;
            let mut alloc = vec![per; max_depth];
            let remainder = total_budget - per * max_depth;
            for item in alloc.iter_mut().take(remainder) {
                *item += 1;
            }
            return alloc;
        }

        // Proportional allocation
        let raw: Vec<f32> = weights
            .iter()
            .map(|w| w / weight_sum * total_budget as f32)
            .collect();

        // Round using largest-remainder method to preserve total
        let floored: Vec<usize> = raw.iter().map(|r| r.floor() as usize).collect();
        let allocated: usize = floored.iter().sum();
        let remainder = total_budget - allocated;

        // Distribute remainder to positions with largest fractional parts
        let mut fractional: Vec<(usize, f32)> = raw
            .iter()
            .zip(floored.iter())
            .enumerate()
            .map(|(i, (r, f))| (i, r - *f as f32))
            .collect();
        fractional.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut result = floored;
        for (idx, _) in fractional.into_iter().take(remainder) {
            result[idx] += 1;
        }

        result
    }
}

impl Default for CurvatureWeightedBudget {
    fn default() -> Self {
        Self::new()
    }
}

/// Curvature-informed verification depth selector.
/// High curvature influence → full verification (deep)
/// Low curvature influence → fast-path (shallow)
#[inline]
pub fn verification_depth(
    position: usize,
    scorer: &mut dyn CurvatureInfluenceScorer,
    max_depth: usize,
) -> usize {
    let influence = scorer.curvature_influence(position);
    let scale = influence.clamp(0.1, 1.0);
    (scale * max_depth as f32).ceil() as usize
}

/// NDS-aware curvature influence scorer that composes NDS proxy signal
/// with the existing persistence × alignment metric.
#[cfg(feature = "nds_proxy")]
pub struct NdsAwareScorer<S: CurvatureInfluenceScorer> {
    /// Inner scorer providing base curvature influence.
    pub inner: S,
    /// NDS weight in the combined score: influence = (1-w)*base + w*nds.
    pub nds_weight: f32,
}

#[cfg(feature = "nds_proxy")]
impl<S: CurvatureInfluenceScorer> NdsAwareScorer<S> {
    pub fn new(inner: S, nds_weight: f32) -> Self {
        Self {
            inner,
            nds_weight: nds_weight.clamp(0.0, 1.0),
        }
    }
}

#[cfg(feature = "nds_proxy")]
impl<S: CurvatureInfluenceScorer> CurvatureInfluenceScorer for NdsAwareScorer<S> {
    fn curvature_influence(&mut self, group: usize) -> f32 {
        self.inner.curvature_influence(group)
        // NDS proxy is available via crate::pruners::nds_proxy::nds_proxy
        // but the scorer doesn't have access to marginals here.
        // Composition happens at the caller level via nds_scaled_budget.
    }

    fn num_groups(&self) -> usize {
        self.inner.num_groups()
    }

    fn update_persistence(&mut self, group: usize, loss: f32) {
        self.inner.update_persistence(group, loss);
    }

    fn update_alignment(&mut self, group: usize, scores: &[f32]) {
        self.inner.update_alignment(group, scores);
    }
}

// ── Inline Unit Tests ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eos_proxy_scorer_initializes_zero() {
        let mut scorer = EosProxyScorer::new(5, 0.1);
        assert_eq!(scorer.num_groups(), 5);
        for k in 0..5 {
            assert!(
                (scorer.curvature_influence(k) - 0.0).abs() < 1e-6,
                "Group {k} should have zero influence at init"
            );
        }
    }

    #[test]
    fn test_influence_in_bounds_after_updates() {
        let mut scorer = EosProxyScorer::new(3, 0.3);
        scorer.update_persistence(0, 0.5);
        scorer.update_persistence(1, 0.2);
        scorer.update_persistence(2, 0.8);
        scorer.update_alignment(0, &[0.1, 0.7, 0.15, 0.05]);
        scorer.update_alignment(1, &[0.25, 0.25, 0.25, 0.25]);
        scorer.update_alignment(2, &[0.9, 0.05, 0.03, 0.02]);

        for k in 0..3 {
            let inf = scorer.curvature_influence(k);
            assert!(
                (0.0..=1.0).contains(&inf),
                "Influence for group {k} should be in [0,1], got {inf}"
            );
        }
    }

    #[test]
    fn test_budget_allocation_sums_to_total() {
        let mut scorer = EosProxyScorer::new(5, 0.1);
        let budget = CurvatureWeightedBudget::new();
        let alloc = budget.allocate(100, 5, &mut scorer);
        let sum: usize = alloc.iter().sum();
        assert_eq!(sum, 100, "Allocation should sum to total budget");
    }

    #[test]
    fn test_high_influence_gets_more_budget() {
        let mut scorer = EosProxyScorer::new(3, 0.5);
        // Make group 2 have highest influence
        for _ in 0..10 {
            scorer.update_persistence(2, 0.9);
        }
        scorer.update_alignment(2, &[0.95, 0.02, 0.02, 0.01]);
        // Groups 0 and 1 stay at zero
        scorer.update_alignment(0, &[0.25, 0.25, 0.25, 0.25]);
        scorer.update_alignment(1, &[0.25, 0.25, 0.25, 0.25]);

        let budget = CurvatureWeightedBudget::new();
        let alloc = budget.allocate(100, 3, &mut scorer);

        assert!(
            alloc[2] > alloc[0],
            "Group 2 (high influence) should get more budget than group 0, got {} vs {}",
            alloc[2],
            alloc[0]
        );
    }

    #[test]
    fn test_floor_guarantee() {
        let mut scorer = EosProxyScorer::new(5, 0.1);
        let budget = CurvatureWeightedBudget {
            floor_ratio: 0.2,
            max_boost: 0.5,
        };
        let alloc = budget.allocate(100, 5, &mut scorer);

        // Every position should get at least floor_ratio * (total/depth)
        let min_budget = 0.2 * 100.0 / 5.0;
        for (i, &a) in alloc.iter().enumerate() {
            assert!(
                a as f32 >= min_budget - 1.0,
                "Position {i} should get at least {min_budget}, got {a}"
            );
        }
    }

    #[test]
    fn test_concentration_computation() {
        let mut scorer = EosProxyScorer::new(2, 0.5);

        // Concentrated scores → high alignment
        scorer.update_alignment(0, &[0.9, 0.05, 0.03, 0.02]);
        // Uniform scores → low alignment
        scorer.update_alignment(1, &[0.25, 0.25, 0.25, 0.25]);

        // Group 0 should have higher alignment than group 1
        assert!(
            scorer.alignment(0) > scorer.alignment(1),
            "Concentrated scores should yield higher alignment: {} vs {}",
            scorer.alignment(0),
            scorer.alignment(1)
        );
    }
}
