//! Adaptive decode budget — PFlash complexity signal for DDTree budget scaling.
//!
//! Uses either the prompt compression ratio (a free byproduct of prefill scoring)
//! or the first-marginal Shannon entropy to dynamically scale DDTree budget per-prompt.
//! Simple prompts → less search. Complex → more.
//!
//! # Modes
//!
//! - **Compression** (Plan 167): uses block selection ratio as complexity signal
//! - **Entropy** (Plan 175 Fusion 2, RangeBudget): uses first-marginal Shannon entropy.
//!   Low entropy → model is confident → small budget (greedy). High entropy → model is
//!   uncertain → large budget (speculative). ANE analogy: RangeDim compiles T∈[1..4] into
//!   one artifact — we compile budget∈[base/2, base*2] from one entropy reading.
//!
//! # Feature flag
//! `budget_adaptation` — Plan 167, Research R050

use crate::speculative::types::BudgetAdaptation;

/// Entropy threshold in nats for budget scaling.
///
/// Below this, the model is considered "confident" (budget scales down).
/// Above this, the model is "uncertain" (budget scales up).
///
/// Chosen to match `entropy_truncate_horizon`'s threshold (2.5 nats) as a reference point.
/// For typical LLM vocabularies: max entropy = ln(vocab_size).
/// V=27 → 3.3 nats, V=32000 → 10.4 nats. Threshold 3.0 nats captures the "easy vs hard"
/// boundary for most single-token decisions.
const ENTROPY_THRESHOLD_NATS: f32 = 3.0;

/// Derive per-prompt tree_budget from base + complexity signal.
///
/// Returns budget clamped to [base/2, base*2].
///
/// # Arguments
/// * `base_budget` — default tree budget from domain config
/// * `signal` — complexity signal. Interpretation depends on `mode`:
///   - `Compression`: compression ratio r ∈ (0, 1]
///   - `Entropy`: first-marginal Shannon entropy in nats (≥ 0)
/// * `mode` — adaptation strategy
///
/// # Scaling curves
///
/// **Compression mode:**
/// ```text
/// r=0.0 → scale=0.5  (budget halved, simple prompt)
/// r=0.5 → scale=1.25 (budget slightly above base)
/// r=1.0 → scale=2.0  (budget doubled, complex prompt)
/// ```
///
/// **Entropy mode (RangeBudget, Plan 175 Fusion 2):**
/// ```text
/// H=0.0 → scale=0.5  (budget halved, deterministic / greedy)
/// H=1.5 → scale=1.0  (budget = base, moderate uncertainty)
/// H=3.0 → scale=2.0  (budget doubled, high uncertainty / speculative)
/// H>3.0 → scale=2.0  (clamped at max)
/// ```
pub fn adaptive_tree_budget(base_budget: usize, signal: f32, mode: BudgetAdaptation) -> usize {
    match mode {
        BudgetAdaptation::Off => base_budget,
        BudgetAdaptation::Compression => {
            let r = signal.clamp(0.0, 1.0);
            // Linear scale: f(0)=0.5, f(0.5)=1.25, f(1)=2.0
            let scale = 0.5 + 1.5 * r;
            let adapted = (base_budget as f32 * scale) as usize;
            adapted.max(base_budget / 2).min(base_budget * 2)
        }
        BudgetAdaptation::Entropy => {
            // RangeBudget: normalize entropy to [0,1], then same scaling curve
            let t = (signal / ENTROPY_THRESHOLD_NATS).clamp(0.0, 1.0);
            let scale = 0.5 + 1.5 * t;
            let adapted = (base_budget as f32 * scale) as usize;
            adapted.max(base_budget / 2).min(base_budget * 2)
        }
    }
}

/// Compute Shannon entropy in nats from a probability distribution.
///
/// Returns H = -Σ p·ln(p) where the sum is over non-zero probabilities.
/// The input should be a normalized probability distribution (sums to ~1.0),
/// but the function handles unnormalized inputs gracefully.
///
/// # Examples
/// - Deterministic (one outcome): H ≈ 0
/// - Uniform over V items: H = ln(V)
#[inline]
pub fn shannon_entropy(probs: &[f32]) -> f32 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Derive entropy signal from first marginal for budget adaptation.
///
/// Convenience wrapper: computes entropy of the first marginal distribution
/// (depth-0 token probabilities) and returns it as the signal for Entropy mode.
///
/// This is the "RangeDim read" — one entropy value determines the entire budget
/// for this query's DDTree expansion.
#[inline]
pub fn entropy_signal(first_marginal: &[f32]) -> f32 {
    shannon_entropy(first_marginal)
}

/// Derive compression ratio from block selection results.
///
/// Given the total number of blocks and the number selected by `block_select`,
/// returns the fraction r ∈ (0, 1] that passed the importance filter.
///
/// This is a zero-alloc computation — just a division.
#[inline]
pub fn compression_ratio(selected_count: usize, total_count: usize) -> f32 {
    if total_count == 0 {
        return 1.0; // no blocks = nothing to compress, treat as complex
    }
    (selected_count as f32) / (total_count as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_budget_off_returns_base() {
        assert_eq!(adaptive_tree_budget(100, 0.5, BudgetAdaptation::Off), 100);
    }

    #[test]
    fn test_adaptive_budget_compression_midpoint() {
        let budget = adaptive_tree_budget(100, 0.5, BudgetAdaptation::Compression);
        // scale = 0.5 + 1.5*0.5 = 1.25 → 125
        assert_eq!(budget, 125);
    }

    #[test]
    fn test_adaptive_budget_compression_low() {
        let budget = adaptive_tree_budget(100, 0.0, BudgetAdaptation::Compression);
        // scale = 0.5 + 1.5*0.0 = 0.5 → clamped to max(50, 50) = 50
        assert_eq!(budget, 50);
    }

    #[test]
    fn test_adaptive_budget_compression_high() {
        let budget = adaptive_tree_budget(100, 1.0, BudgetAdaptation::Compression);
        // scale = 0.5 + 1.5*1.0 = 2.0 → 200
        assert_eq!(budget, 200);
    }

    #[test]
    fn test_adaptive_budget_clamped_lower() {
        // Even with r=0.01, budget shouldn't go below base/2
        let budget = adaptive_tree_budget(100, 0.01, BudgetAdaptation::Compression);
        assert!(budget >= 50, "budget {} < base/2 = 50", budget);
    }

    #[test]
    fn test_adaptive_budget_clamped_upper() {
        // Even with r=1.0, budget shouldn't exceed base*2
        let budget = adaptive_tree_budget(100, 1.0, BudgetAdaptation::Compression);
        assert!(budget <= 200, "budget {} > base*2 = 200", budget);
    }

    // ── Entropy mode tests (Plan 175 Fusion 2: RangeBudget) ────────

    #[test]
    fn test_entropy_zero_nats_halves_budget() {
        // H=0: deterministic, model is confident → budget halved
        let budget = adaptive_tree_budget(100, 0.0, BudgetAdaptation::Entropy);
        assert_eq!(budget, 50, "H=0 should halve budget");
    }

    #[test]
    fn test_entropy_half_threshold_near_base() {
        // H=1.5: half of threshold → scale = 0.5 + 1.5*0.5 = 1.25 → 125
        let budget = adaptive_tree_budget(100, 1.5, BudgetAdaptation::Entropy);
        assert_eq!(budget, 125, "H=1.5 should give 1.25× budget");
    }

    #[test]
    fn test_entropy_at_threshold_doubles_budget() {
        // H=3.0: at threshold → scale = 0.5 + 1.5*1.0 = 2.0 → 200
        let budget = adaptive_tree_budget(100, 3.0, BudgetAdaptation::Entropy);
        assert_eq!(budget, 200, "H=3.0 should double budget");
    }

    #[test]
    fn test_entropy_above_threshold_clamps() {
        // H=10.0: above threshold → clamped at base*2
        let budget = adaptive_tree_budget(100, 10.0, BudgetAdaptation::Entropy);
        assert!(
            budget <= 200,
            "H>threshold should clamp at base*2, got {}",
            budget
        );
    }

    #[test]
    fn test_entropy_below_threshold_halves() {
        // H=-1.0: negative (invalid) → clamped to 0 → scale 0.5 → 50
        let budget = adaptive_tree_budget(100, -1.0, BudgetAdaptation::Entropy);
        assert_eq!(budget, 50, "negative entropy should clamp to base/2");
    }

    #[test]
    fn test_entropy_scaling_monotonic() {
        let budgets: Vec<usize> = (0..=10)
            .map(|h| adaptive_tree_budget(1000, h as f32 * 0.4, BudgetAdaptation::Entropy))
            .collect();
        for w in budgets.windows(2) {
            assert!(w[0] <= w[1], "not monotonic: {} > {}", w[0], w[1]);
        }
    }

    // ── Shannon entropy tests ──────────────────────────────────────

    #[test]
    fn test_shannon_entropy_deterministic() {
        // One-hot: H should be ~0
        let mut probs = vec![0.0f32; 10];
        probs[3] = 1.0;
        let h = shannon_entropy(&probs);
        assert!(
            h < 0.01,
            "deterministic distribution should have ~0 entropy, got {}",
            h
        );
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform over 4 items: H = ln(4) ≈ 1.386
        let probs = vec![0.25f32; 4];
        let h = shannon_entropy(&probs);
        assert!(
            (h - 4.0f32.ln()).abs() < 0.01,
            "uniform(4) should have H=ln(4)≈1.386, got {}",
            h
        );
    }

    #[test]
    fn test_shannon_entropy_uniform_27() {
        // Uniform over 27 items (draft vocab): H = ln(27) ≈ 3.296
        let probs = vec![1.0f32 / 27.0; 27];
        let h = shannon_entropy(&probs);
        assert!(
            (h - 27.0f32.ln()).abs() < 0.01,
            "uniform(27) should have H=ln(27)≈3.296, got {}",
            h
        );
    }

    #[test]
    fn test_shannon_entropy_empty() {
        let h = shannon_entropy(&[]);
        assert_eq!(h, 0.0, "empty distribution should have 0 entropy");
    }

    #[test]
    fn test_entropy_signal_peaked() {
        // Peaked distribution (90% on one token): H should be low
        let mut probs = vec![0.1 / 26.0f32; 27];
        probs[0] = 0.9;
        let h = entropy_signal(&probs);
        assert!(
            h < 1.0,
            "peaked distribution should have low entropy, got {}",
            h
        );
    }

    #[test]
    fn test_entropy_signal_uniform() {
        // Uniform: H should be high (above threshold for budget doubling)
        let probs = vec![1.0f32 / 27.0; 27];
        let h = entropy_signal(&probs);
        assert!(
            h > ENTROPY_THRESHOLD_NATS,
            "uniform(27) should exceed threshold"
        );
    }

    // ── Compression ratio tests ────────────────────────────────────

    #[test]
    fn test_compression_ratio_normal() {
        assert!((compression_ratio(5, 10) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_compression_ratio_zero_total() {
        assert_eq!(compression_ratio(0, 0), 1.0);
    }

    #[test]
    fn test_compression_ratio_all_selected() {
        assert!((compression_ratio(10, 10) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaling_curve_monotonic() {
        let budgets: Vec<usize> = (0..=10)
            .map(|r| adaptive_tree_budget(1000, r as f32 / 10.0, BudgetAdaptation::Compression))
            .collect();
        for w in budgets.windows(2) {
            assert!(w[0] <= w[1], "not monotonic: {} > {}", w[0], w[1]);
        }
    }
}
