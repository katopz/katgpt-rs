//! Boltzmann (softmax) sampler with temperature-controlled exploration.
//!
//! Implements temperature-scaled softmax sampling:
//! ```text
//! P(z) = exp(U_z / τ) / Σ exp(U_j / τ)
//! ```
//!
//! - τ → 0: greedy (argmax)
//! - τ → ∞: uniform random
//! - τ = 1.0: standard softmax
//!
//! Based on OPUS paper (arXiv:2602.05400): Boltzmann sampling with redundancy
//! penalty outperforms greedy top-k by +1.26 avg points.

use katgpt_types::Rng;

// ── Boltzmann Sampling ──────────────────────────────────────────

/// Sample a single arm from a Boltzmann distribution over utilities.
///
/// Returns the index of the sampled arm. Temperature controls exploration:
/// - `temperature` ≈ 0.0 → greedy (nearly always picks max utility)
/// - `temperature` = 1.0 → standard softmax
/// - `temperature` → ∞  → uniform random
///
/// # Edge Cases
///
/// - Empty `utilities` → panics (no arms to sample)
/// - `temperature` ≤ 0 → treated as greedy (argmax)
/// - All utilities equal → uniform regardless of temperature
/// - NaN/Inf utilities → filtered out (treated as -∞)
///
/// # Panics
///
/// Panics if `utilities` is empty.
pub fn boltzmann_sample(utilities: &[f32], temperature: f32, rng: &mut Rng) -> usize {
    assert!(!utilities.is_empty(), "cannot sample from empty utilities");

    // Greedy: temperature ≤ 0 or very close to zero
    if temperature <= 1e-8 {
        return argmax_safe(utilities);
    }

    let n = utilities.len();

    // Single arm: always return 0
    if n == 1 {
        return 0;
    }

    // Compute logits = U_z / τ with numerical stability
    let max_u = max_safe(utilities);
    let inv_temp = 1.0 / temperature;
    let mut logits: Vec<f32> = utilities
        .iter()
        .map(|&u| {
            if u.is_finite() {
                (u - max_u) * inv_temp
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();

    // Exp-sum in log space
    let mut sum_exp = 0.0f32;
    for l in &mut logits {
        *l = l.exp();
        sum_exp += *l;
    }

    // Fallback: all logits underflowed → uniform
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return (rng.uniform() * n as f32) as usize;
    }

    // CDF-based sampling
    let target = rng.uniform() * sum_exp;
    let mut cumulative = 0.0f32;
    for (i, &prob) in logits.iter().enumerate() {
        cumulative += prob;
        if target <= cumulative {
            return i;
        }
    }

    // Fallback: last arm (floating point edge case)
    n - 1
}

/// Sample `k` arms without replacement from a Boltzmann distribution.
///
/// Uses iterative rescaling: after selecting an arm, its probability is set
/// to zero and remaining probabilities are renormalized. This produces a
/// valid distribution over size-`k` subsets.
///
/// # Guarantees
///
/// - No duplicate indices in output
/// - Output length = min(k, utilities.len())
/// - Deterministic given same seed + inputs
///
/// # Panics
///
/// Panics if `utilities` is empty.
pub fn boltzmann_sample_batch(
    utilities: &[f32],
    temperature: f32,
    k: usize,
    rng: &mut Rng,
) -> Vec<usize> {
    assert!(!utilities.is_empty(), "cannot sample from empty utilities");

    let n = utilities.len();
    let k_actual = k.min(n);

    // Optimization: if k >= n, return all indices
    if k_actual >= n {
        return (0..n).collect();
    }

    // Optimization: greedy batch for very low temperature
    if temperature <= 1e-8 {
        return greedy_top_k(utilities, k_actual);
    }

    // Compute initial Boltzmann probabilities
    let max_u = max_safe(utilities);
    let inv_temp = 1.0 / temperature;
    let mut probs: Vec<f32> = utilities
        .iter()
        .map(|&u| {
            if u.is_finite() {
                ((u - max_u) * inv_temp).exp()
            } else {
                0.0
            }
        })
        .collect();

    let mut selected = Vec::with_capacity(k_actual);
    let mut available: Vec<usize> = (0..n).collect();

    for _ in 0..k_actual {
        // Renormalize remaining probabilities
        let sum: f32 = available.iter().map(|&i| probs[i]).sum();
        if sum <= 0.0 || !sum.is_finite() {
            // Fallback: uniform among remaining
            let idx = (rng.uniform() * available.len() as f32) as usize;
            let arm = available[idx];
            selected.push(arm);
            available.retain(|&x| x != arm);
            continue;
        }

        // CDF-based sampling among remaining
        let target = rng.uniform() * sum;
        let mut cumulative = 0.0f32;
        let mut chosen_idx = 0;
        for (j, &arm) in available.iter().enumerate() {
            cumulative += probs[arm];
            if target <= cumulative {
                chosen_idx = j;
                break;
            }
        }

        let arm = available[chosen_idx];
        selected.push(arm);
        probs[arm] = 0.0; // Remove from future selection
        available.remove(chosen_idx);
    }

    selected
}

/// Compute Boltzmann probabilities without sampling.
///
/// Useful for introspection and debugging. Returns normalized probabilities.
///
/// # Panics
///
/// Panics if `utilities` is empty.
pub fn boltzmann_probabilities(utilities: &[f32], temperature: f32) -> Vec<f32> {
    assert!(
        !utilities.is_empty(),
        "cannot compute probs for empty utilities"
    );

    if temperature <= 1e-8 {
        let best = argmax_safe(utilities);
        let mut probs = vec![0.0f32; utilities.len()];
        probs[best] = 1.0;
        return probs;
    }

    let max_u = max_safe(utilities);
    let inv_temp = 1.0 / temperature;
    let mut probs: Vec<f32> = utilities
        .iter()
        .map(|&u| {
            if u.is_finite() {
                ((u - max_u) * inv_temp).exp()
            } else {
                0.0
            }
        })
        .collect();

    let sum: f32 = probs.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        for p in &mut probs {
            *p /= sum;
        }
    } else {
        // Fallback: uniform
        let uniform = 1.0 / probs.len() as f32;
        probs.fill(uniform);
    }

    probs
}

// ── Helpers ─────────────────────────────────────────────────────

/// Argmax that handles NaN/Inf: prefers finite values, breaks ties by index.
fn argmax_safe(values: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v.is_finite() && (v > best_val || !best_val.is_finite()) {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Max that handles NaN/Inf: returns maximum finite value, or 0.0 if none.
fn max_safe(values: &[f32]) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for &v in values {
        if v.is_finite() && v > m {
            m = v;
        }
    }
    if m.is_finite() { m } else { 0.0 }
}

/// Greedy top-k selection (deterministic, no RNG).
fn greedy_top_k(utilities: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = utilities
        .iter()
        .enumerate()
        .map(|(i, &u)| (i, if u.is_finite() { u } else { f32::NEG_INFINITY }))
        .collect();
    // Partial sort: we only need top-k
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_rng() -> Rng {
        Rng::new(42)
    }

    // ── Single Sample ──────────────────────────────────────

    #[test]
    fn test_boltzmann_single_arm() {
        let mut rng = seeded_rng();
        let idx = boltzmann_sample(&[0.5], 1.0, &mut rng);
        assert_eq!(idx, 0, "single arm must always return index 0");
    }

    #[test]
    fn test_boltzmann_greedy_low_temperature() {
        let mut rng = seeded_rng();
        let utilities = &[0.1, 0.9, 0.3, 0.7];
        // Temperature near zero → should always pick arm 1 (max = 0.9)
        for _ in 0..100 {
            let idx = boltzmann_sample(utilities, 1e-10, &mut rng);
            assert_eq!(idx, 1, "greedy must pick argmax arm 1");
        }
    }

    #[test]
    fn test_boltzmann_uniform_high_temperature() {
        let mut rng = Rng::new(123);
        let utilities = &[1.0, 0.0, 0.0, 0.0];
        // Temperature very high → nearly uniform, arm 0 gets ~25%
        let mut counts = [0usize; 4];
        let n_trials = 10_000;
        for _ in 0..n_trials {
            let idx = boltzmann_sample(utilities, 1_000_000.0, &mut rng);
            counts[idx] += 1;
        }
        // Each arm should get roughly 25% ± 5%
        for (i, &count) in counts.iter().enumerate() {
            let frac = count as f32 / n_trials as f32;
            assert!(
                (0.20..0.30).contains(&frac),
                "arm {i}: frac={frac:.3}, expected ~0.25"
            );
        }
    }

    #[test]
    fn test_boltzmann_soft_preferences() {
        let mut rng = Rng::new(456);
        let utilities = &[0.0, 1.0]; // arm 1 is better
        let mut counts = [0usize; 2];
        let n_trials = 10_000;
        // τ=1.0: softmax of [0, 1] → P(1) ≈ 0.731
        for _ in 0..n_trials {
            let idx = boltzmann_sample(utilities, 1.0, &mut rng);
            counts[idx] += 1;
        }
        let frac_1 = counts[1] as f32 / n_trials as f32;
        // Should be roughly sigmoid(1) ≈ 0.731 ± 0.05
        assert!(
            (0.68..0.78).contains(&frac_1),
            "arm 1 fraction={frac_1:.3}, expected ~0.731"
        );
    }

    #[test]
    fn test_boltzmann_equal_utilities() {
        let mut rng = Rng::new(789);
        let utilities = &[0.5, 0.5, 0.5];
        let mut counts = [0usize; 3];
        let n_trials = 3_000;
        for _ in 0..n_trials {
            let idx = boltzmann_sample(utilities, 1.0, &mut rng);
            counts[idx] += 1;
        }
        // Equal utilities → uniform distribution
        for (i, &count) in counts.iter().enumerate() {
            let frac = count as f32 / n_trials as f32;
            assert!(
                (0.28..0.38).contains(&frac),
                "equal utilities: arm {i} frac={frac:.3}, expected ~0.333"
            );
        }
    }

    #[test]
    fn test_boltzmann_nan_utility() {
        let mut rng = seeded_rng();
        let utilities = &[f32::NAN, 0.5, 0.8];
        // NaN should be ignored; should pick arm 2 (max finite)
        let idx = boltzmann_sample(utilities, 1e-10, &mut rng);
        assert_eq!(idx, 2, "NaN arm should be skipped, pick arm 2");
    }

    #[test]
    fn test_boltzmann_all_nan() {
        let mut rng = seeded_rng();
        let utilities = &[f32::NAN, f32::NAN];
        // All NaN → fallback to any valid index
        let idx = boltzmann_sample(utilities, 1.0, &mut rng);
        assert!(idx < 2, "should return valid index even with all NaN");
    }

    #[test]
    fn test_boltzmann_probabilities_sum_to_one() {
        let probs = boltzmann_probabilities(&[0.1, 0.5, 0.3, 0.8], 1.0);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "probabilities must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_boltzmann_probabilities_greedy() {
        let probs = boltzmann_probabilities(&[0.1, 0.9, 0.3], 1e-10);
        assert_eq!(probs[0], 0.0);
        assert_eq!(probs[1], 1.0);
        assert_eq!(probs[2], 0.0);
    }

    // ── Batch Sampling ─────────────────────────────────────

    #[test]
    fn test_batch_no_duplicates() {
        let mut rng = seeded_rng();
        let utilities = &[0.1, 0.2, 0.3, 0.4, 0.5];
        let selected = boltzmann_sample_batch(utilities, 1.0, 3, &mut rng);
        assert_eq!(selected.len(), 3, "should select exactly 3 arms");
        let mut seen = std::collections::HashSet::new();
        for &idx in &selected {
            assert!(seen.insert(idx), "duplicate arm {idx} in batch selection");
        }
    }

    #[test]
    fn test_batch_k_exceeds_arms() {
        let mut rng = seeded_rng();
        let utilities = &[0.1, 0.2, 0.3];
        let selected = boltzmann_sample_batch(utilities, 1.0, 10, &mut rng);
        assert_eq!(selected.len(), 3, "k > n → select all n arms");
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_batch_greedy_top_k() {
        let mut rng = seeded_rng();
        let utilities = &[0.1, 0.9, 0.3, 0.7, 0.2];
        let selected = boltzmann_sample_batch(utilities, 1e-10, 2, &mut rng);
        assert_eq!(selected.len(), 2);
        // Greedy should pick indices 1 (0.9) and 3 (0.7)
        assert!(selected.contains(&1), "greedy batch must include arm 1");
        assert!(selected.contains(&3), "greedy batch must include arm 3");
    }

    #[test]
    fn test_batch_diversity_at_high_temperature() {
        let utilities = &[1.0, 0.0, 0.0, 0.0];
        let mut arm_counts = [0usize; 4];
        let n_trials = 1_000;
        for seed in 0..n_trials {
            let mut r = Rng::new(seed);
            let selected = boltzmann_sample_batch(utilities, 100.0, 2, &mut r);
            for &arm in &selected {
                arm_counts[arm] += 1;
            }
        }
        // At high temperature, all arms should be selected sometimes
        for (i, &count) in arm_counts.iter().enumerate() {
            assert!(count > 0, "arm {i} was never selected in {n_trials} trials");
        }
    }

    #[test]
    fn test_batch_single_arm() {
        let mut rng = seeded_rng();
        let utilities = &[0.5];
        let selected = boltzmann_sample_batch(utilities, 1.0, 1, &mut rng);
        assert_eq!(selected, vec![0]);
    }

    // ── Probability Correctness ────────────────────────────

    #[test]
    fn test_boltzmann_two_arm_analytical() {
        // Two arms: U = [0, 1], τ = 0.5
        // logits after max-subtract: arm0 = -2.0, arm1 = 0.0
        // P(0) = e^(-2) / (e^(-2) + e^0) = 1 / (1 + e^2) ≈ 0.1192
        // P(1) = e^0 / (e^(-2) + e^0) = e^2 / (1 + e^2) ≈ 0.8808
        let probs = boltzmann_probabilities(&[0.0, 1.0], 0.5);
        let e2 = (2.0f32).exp();
        let expected_p1 = e2 / (1.0 + e2);
        let p1 = probs[1];
        assert!(
            (p1 - expected_p1).abs() < 1e-4,
            "P(1)={p1:.4}, expected={expected_p1:.4}"
        );
    }

    #[test]
    fn test_boltzmann_distribution_converges_to_analytical() {
        let mut rng = Rng::new(777);
        let utilities = &[0.0, 1.0, 2.0];
        let temperature = 1.0;
        let probs = boltzmann_probabilities(utilities, temperature);

        let n_trials = 50_000;
        let mut counts = [0usize; 3];
        for _ in 0..n_trials {
            let idx = boltzmann_sample(utilities, temperature, &mut rng);
            counts[idx] += 1;
        }

        for (i, &count) in counts.iter().enumerate() {
            let empirical = count as f32 / n_trials as f32;
            let analytical = probs[i];
            let diff = (empirical - analytical).abs();
            assert!(
                diff < 0.02,
                "arm {i}: empirical={empirical:.4}, analytical={analytical:.4}, diff={diff:.4}"
            );
        }
    }
}
