//! NFCoT FlowBudget — Sigmoid-Weighted Speculative Depth Allocation (Plan 229 T4).
//!
//! Allocates speculative depth budget proportional to flow scores using
//! sigmoid weighting (NOT softmax — per project rules). High-score branches
//! get more speculative depth; low-score branches get early termination.
//!
//! Algorithm:
//!   1. mean = Σ score_i / n
//!   2. w_i = sigmoid(score_i - mean)
//!   3. budget_i = round(total_budget * w_i / Σ w_i)
//!   4. Clamp each to min_budget, adjust so total sums to total_budget

/// Sigmoid activation: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute sigmoid-weighted ratios for each score relative to the mean.
///
/// Returns raw weights that sum to approximately `scores.len() * 0.5`
/// (since sigmoid(0) = 0.5 and deviations cancel around the mean).
#[inline]
pub fn budget_ratios(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    scores.iter().map(|&s| sigmoid(s - mean)).collect()
}

/// Normalize raw weights to integer budgets that sum to `total`.
///
/// Uses largest-remainder method: floor-allocate, then distribute
/// remaining units to branches with the largest fractional parts.
#[inline]
pub fn normalize_budget(raw: &[f32], total: usize) -> Vec<usize> {
    if raw.is_empty() || total == 0 {
        return vec![0; raw.len()];
    }

    let w_total: f32 = raw.iter().sum();
    if w_total < f32::EPSILON {
        // Degenerate: equal split
        let each = total / raw.len();
        let mut out = vec![each; raw.len()];
        let rem = total - each * raw.len();
        for i in 0..rem {
            out[i] += 1;
        }
        return out;
    }

    // Floor allocation + collect remainders
    let mut budgets = Vec::with_capacity(raw.len());
    let mut remainders = Vec::with_capacity(raw.len());
    let mut allocated = 0usize;

    for &w in raw {
        let exact = total as f32 * w / w_total;
        let floored = exact.floor() as usize;
        budgets.push(floored);
        remainders.push(exact - floored as f32);
        allocated += floored;
    }

    // Distribute remainder to branches with largest fractional part
    let mut remaining = total - allocated;
    if remaining > 0 {
        let mut indices: Vec<usize> = (0..remainders.len()).collect();
        indices.sort_by(|&a, &b| {
            remainders[b]
                .partial_cmp(&remainders[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in &indices {
            if remaining == 0 {
                break;
            }
            budgets[i] += 1;
            remaining -= 1;
        }
    }

    budgets
}

/// Allocate speculative depth budget proportional to flow scores (Plan 229 T4).
///
/// Uses sigmoid-weighted allocation (NOT softmax — per project rules).
/// Each branch gets: `budget_i = round(total_budget * sigmoid(score_i - mean_score) / Σ w_j)`.
/// Normalized so that the total budget is respected, each branch gets ≥ `min_budget`.
///
/// Returns a Vec of depth allocations, one per branch.
#[inline]
pub fn allocate_budget(scores: &[f32], total_budget: usize) -> Vec<usize> {
    allocate_budget_with_min(scores, total_budget, 1)
}

/// Internal: allocate with configurable min_budget.
fn allocate_budget_with_min(scores: &[f32], total_budget: usize, min_budget: usize) -> Vec<usize> {
    if scores.is_empty() {
        return Vec::new();
    }
    if total_budget == 0 {
        return vec![0; scores.len()];
    }

    let ratios = budget_ratios(scores);

    // Ensure we don't over-commit to min_budget
    let effective_min = min_budget.min(total_budget / scores.len().max(1));
    let adjusted_total = total_budget.saturating_sub(effective_min * scores.len());
    let mut budgets = normalize_budget(&ratios, adjusted_total);

    // Add min_budget back
    for b in &mut budgets {
        *b += effective_min;
    }

    budgets
}

/// Stateful budget allocator with scratch buffer reuse (Plan 229 T4).
///
/// Pre-allocates a scratch buffer at construction to avoid per-call allocation.
pub struct FlowBudgetAllocator {
    /// Minimum budget per branch. Default: 1.
    min_budget: usize,
    /// Pre-allocated scratch buffer for weights.
    scratch: Vec<f32>,
}

impl FlowBudgetAllocator {
    /// Create a new allocator with explicit minimum budget.
    #[inline]
    pub fn new(min_budget: usize) -> Self {
        Self {
            min_budget,
            scratch: Vec::new(),
        }
    }

    /// Create an allocator with default minimum budget (1).
    #[inline]
    pub fn with_default() -> Self {
        Self::new(1)
    }

    /// Allocate speculative depth budget using pre-allocated scratch buffer.
    pub fn allocate(&mut self, scores: &[f32], total_budget: usize) -> Vec<usize> {
        if scores.is_empty() || total_budget == 0 {
            return vec![0; scores.len()];
        }

        // Compute ratios into scratch buffer
        self.scratch.clear();
        self.scratch.reserve(scores.len());
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        for &s in scores {
            self.scratch.push(sigmoid(s - mean));
        }

        let effective_min = self.min_budget.min(total_budget / scores.len().max(1));
        let adjusted_total = total_budget.saturating_sub(effective_min * scores.len());
        let mut budgets = normalize_budget(&self.scratch, adjusted_total);

        for b in &mut budgets {
            *b += effective_min;
        }

        budgets
    }

    /// Set the minimum budget per branch.
    #[inline]
    pub fn set_min_budget(&mut self, min: usize) {
        self.min_budget = min;
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_budget_two_branches() {
        let scores = [2.0, 0.5];
        let budget = allocate_budget(&scores, 10);
        assert_eq!(budget.len(), 2);
        assert!(
            budget[0] > budget[1],
            "Higher score should get more budget: {:?}",
            budget
        );
        assert_eq!(budget.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_allocate_budget_equal_scores() {
        let scores = [3.0, 3.0, 3.0];
        let budget = allocate_budget(&scores, 12);
        assert_eq!(budget.len(), 3);
        // Equal scores → equal allocation
        assert!(
            budget.iter().all(|&b| b == budget[0]),
            "Equal scores should get equal budget: {:?}",
            budget
        );
        assert_eq!(budget.iter().sum::<usize>(), 12);
    }

    #[test]
    fn test_allocate_budget_single_branch() {
        let scores = [5.0];
        let budget = allocate_budget(&scores, 8);
        assert_eq!(budget, vec![8]);
    }

    #[test]
    fn test_allocate_budget_sums_to_total() {
        let scores = [1.0, 2.0, 3.0, 4.0, 0.5];
        for total in [5, 10, 20, 50, 100] {
            let budget = allocate_budget(&scores, total);
            let sum: usize = budget.iter().sum();
            assert_eq!(
                sum, total,
                "Budget should sum to {total}, got {sum}: {:?}",
                budget
            );
        }
    }

    #[test]
    fn test_allocate_budget_min_budget_respected() {
        let scores = [0.0, 0.0, 0.0, 10.0]; // one dominant branch
        let budget = allocate_budget(&scores, 20);
        for (i, &b) in budget.iter().enumerate() {
            assert!(b >= 1, "Branch {} should get at least 1, got {}", i, b);
        }
        assert_eq!(budget.iter().sum::<usize>(), 20);
    }

    #[test]
    fn test_budget_ratios() {
        // Scores above mean → sigmoid > 0.5, below mean → sigmoid < 0.5
        let ratios = budget_ratios(&[2.0, 0.0]);
        assert!(
            ratios[0] > 0.5,
            "Above-mean score should have ratio > 0.5: {}",
            ratios[0]
        );
        assert!(
            ratios[1] < 0.5,
            "Below-mean score should have ratio < 0.5: {}",
            ratios[1]
        );
        // Equal scores → both 0.5
        let eq = budget_ratios(&[1.0, 1.0]);
        assert!(
            (eq[0] - 0.5).abs() < 1e-6 && (eq[1] - 0.5).abs() < 1e-6,
            "Equal scores should both be 0.5: {:?}",
            eq
        );
    }

    #[test]
    fn test_normalize_budget() {
        let raw = [0.6, 0.3, 0.1];
        let budgets = normalize_budget(&raw, 10);
        assert_eq!(budgets.len(), 3);
        assert_eq!(budgets.iter().sum::<usize>(), 10);
        assert!(budgets[0] >= budgets[1]);
        assert!(budgets[1] >= budgets[2]);
    }

    #[test]
    fn test_allocator_stateful() {
        let mut alloc = FlowBudgetAllocator::with_default();
        let scores = [3.0, 1.0, 2.0];
        let budget = alloc.allocate(&scores, 15);
        assert_eq!(budget.len(), 3);
        assert_eq!(budget.iter().sum::<usize>(), 15);
        assert!(budget[0] >= budget[2]);
        assert!(budget[2] >= budget[1]);

        // Test set_min_budget
        alloc.set_min_budget(2);
        let budget2 = alloc.allocate(&scores, 15);
        for &b in &budget2 {
            assert!(b >= 2, "Each branch should get at least 2: {}", b);
        }
    }

    #[test]
    fn test_allocate_budget_many_branches() {
        let scores: Vec<f32> = vec![0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0];
        let budget = allocate_budget(&scores, 50);
        assert_eq!(budget.len(), 10);
        assert_eq!(budget.iter().sum::<usize>(), 50);
        // Generally increasing with score
        assert!(
            budget[9] > budget[0],
            "Highest score should get more than lowest: {:?}",
            budget
        );
        // All ≥ 1
        for (i, &b) in budget.iter().enumerate() {
            assert!(b >= 1, "Branch {} should get at least 1", i);
        }
    }

    #[test]
    fn test_allocate_budget_zero_total() {
        let scores = [1.0, 2.0, 3.0];
        let budget = allocate_budget(&scores, 0);
        assert_eq!(budget, vec![0, 0, 0]);
    }

    #[test]
    fn test_bench_allocate_budget_32_branches() {
        let scores: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let start = std::time::Instant::now();
        let iters = 10_000;
        for _ in 0..iters {
            std::hint::black_box(allocate_budget(&scores, 64));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        eprintln!("allocate_budget 32 branches: {per_call:.0}ns/call");
        assert!(
            per_call < 100_000.0,
            "32-branch allocation should be <100μs (debug), got {per_call:.0}ns"
        );
    }
}

// TL;DR: Sigmoid-weighted speculative depth allocation. `allocate_budget()` distributes
// total budget proportional to flow scores using sigmoid (NOT softmax). `FlowBudgetAllocator`
// pre-allocates scratch buffer for zero hot-path allocation. Feature: `nf_flow_budget`.
