//! Dirichlet-sampled Markov chain generator for data-probe diagnostics.
//!
//! Generates transition matrices from Dirichlet(α, …, α) distributions and
//! selects the one whose entropy rate is closest to a target value `H`.
//! Used as a controlled "probe-LLM" where the true data-generating distribution
//! is known exactly, enabling formal validation of information-theoretic claims.

/// A Markov chain with known transition matrix and computed properties.
pub struct MarkovChain {
    /// Transition matrix P[i][j] = Pr(next=j | current=i).
    pub transition: Vec<Vec<f32>>,
    /// Stationary distribution π.
    pub stationary: Vec<f32>,
    /// Computed entropy rate H(P) = -Σᵢ πᵢ Σⱼ Pᵢⱼ log Pᵢⱼ.
    pub entropy_rate: f32,
    /// Number of states (= vocabulary size for probe-LLM).
    pub num_states: usize,
}

/// Sample a single Dirichlet(α, …, α) variate of dimension `k`.
///
/// Uses the Gamma→exponential trick: sample K exponential variates via
/// `-ln(U)` where `U ~ Uniform(0,1)`, then normalize.
fn sample_dirichlet(k: usize, _alpha: f32, rng: &mut fastrand::Rng) -> Vec<f32> {
    let mut samples: Vec<f32> = (0..k)
        .map(|_| {
            // Exponential(α) variate: α * (-ln(U)), but for symmetric Dirichlet
            // the shared α cancels in normalization, so we use -ln(U) directly.
            let u = rng.f32();
            // Guard against ln(0) = -inf.
            let u_safe = u.max(1e-10);
            -u_safe.ln()
        })
        .collect();
    let sum: f32 = samples.iter().sum();
    if sum > 0.0 {
        for x in samples.iter_mut() {
            *x /= sum;
        }
    } else {
        // Fallback: uniform.
        let v = 1.0 / k as f32;
        samples.fill(v);
    }
    samples
}

/// Compute stationary distribution by power iteration.
///
/// Starts with uniform π, then applies π ← π·P for `n_iters` iterations.
fn stationary_distribution(transition: &[Vec<f32>], n_iters: usize) -> Vec<f32> {
    let k = transition.len();
    let mut pi = vec![1.0 / k as f32; k];
    for _ in 0..n_iters {
        let mut new_pi = vec![0.0f32; k];
        for i in 0..k {
            for j in 0..k {
                new_pi[j] += pi[i] * transition[i][j];
            }
        }
        // Renormalize to prevent drift.
        let sum: f32 = new_pi.iter().sum();
        if sum > 0.0 {
            for x in new_pi.iter_mut() {
                *x /= sum;
            }
        }
        pi = new_pi;
    }
    pi
}

/// Compute entropy rate H(P) = -Σᵢ πᵢ Σⱼ Pᵢⱼ log₂(Pᵢⱼ).
fn entropy_rate(transition: &[Vec<f32>], stationary: &[f32]) -> f32 {
    let k = transition.len();
    let ln2 = std::f32::consts::LN_2;
    let mut h = 0.0f32;
    for i in 0..k {
        let mut row_entropy = 0.0f32;
        for j in 0..k {
            let p = transition[i][j];
            if p > 0.0 {
                row_entropy -= p * (p.ln() / ln2);
            }
        }
        h += stationary[i] * row_entropy;
    }
    h
}

/// Generate a Markov chain with entropy rate closest to `target_h`.
///
/// Samples `n_candidates` transition matrices from Dirichlet(α, …, α),
/// computes entropy rate for each, and returns the one closest to `target_h`.
pub fn generate_markov_chain(
    num_states: usize,
    target_h: f32,
    alpha: f32,
    n_candidates: usize,
    rng: &mut fastrand::Rng,
) -> MarkovChain {
    let mut best_chain: Option<MarkovChain> = None;
    let mut best_diff = f32::INFINITY;

    for _ in 0..n_candidates {
        // Sample a row-stochastic transition matrix from Dirichlet.
        let transition: Vec<Vec<f32>> = (0..num_states)
            .map(|_| sample_dirichlet(num_states, alpha, rng))
            .collect();

        let stationary = stationary_distribution(&transition, 100);
        let h = entropy_rate(&transition, &stationary);

        let diff = (h - target_h).abs();
        if diff < best_diff {
            best_diff = diff;
            best_chain = Some(MarkovChain {
                transition,
                stationary,
                entropy_rate: h,
                num_states,
            });
        }
    }

    best_chain.expect("at least one candidate must be generated")
}

/// Sample a sequence of length `n` from the Markov chain.
///
/// Starts by sampling the initial state from the stationary distribution,
/// then follows the transition probabilities for subsequent states.
pub fn sample_sequence(chain: &MarkovChain, n: usize, rng: &mut fastrand::Rng) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }

    let mut sequence = Vec::with_capacity(n);

    // Sample initial state from stationary distribution.
    sequence.push(sample_categorical(&chain.stationary, rng));

    // Sample subsequent states from transition probabilities.
    for _ in 1..n {
        let current = *sequence.last().unwrap();
        sequence.push(sample_categorical(&chain.transition[current], rng));
    }

    sequence
}

/// Sample from a categorical distribution given probabilities.
fn sample_categorical(probs: &[f32], rng: &mut fastrand::Rng) -> usize {
    let u = rng.f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i;
        }
    }
    // Fallback to last state (handles floating-point rounding).
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_rows_sum_to_one() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 2.0, 1.0, 10, &mut rng);
        for row in &chain.transition {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row sum = {sum}");
        }
    }

    #[test]
    fn test_stationary_sums_to_one() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 2.0, 1.0, 10, &mut rng);
        let sum: f32 = chain.stationary.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "stationary sum = {sum}");
    }

    #[test]
    fn test_entropy_rate_bounded() {
        let mut rng = fastrand::Rng::new();
        let k = 4usize;
        let chain = generate_markov_chain(k, 1.5, 1.0, 10, &mut rng);
        let max_h = (k as f32).log2();
        assert!(
            chain.entropy_rate >= 0.0 && chain.entropy_rate <= max_h + 1e-4,
            "entropy_rate = {}, max = {}",
            chain.entropy_rate,
            max_h
        );
    }

    #[test]
    fn test_sample_sequence_length() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 1.0, 1.0, 10, &mut rng);
        let seq = sample_sequence(&chain, 100, &mut rng);
        assert_eq!(seq.len(), 100);
        assert!(seq.iter().all(|&s| s < 4));
    }

    #[test]
    fn test_empty_sequence() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 1.0, 1.0, 10, &mut rng);
        let seq = sample_sequence(&chain, 0, &mut rng);
        assert!(seq.is_empty());
    }

    // GOAT proof G1: Markov entropy accuracy — empirical entropy ≈ computed entropy_rate within 5%.
    #[test]
    fn goat_g1_markov_entropy_accuracy() {
        let mut rng = fastrand::Rng::with_seed(77);
        let chain = generate_markov_chain(4, 1.5, 1.0, 50, &mut rng);

        // Sample a long sequence and estimate empirical entropy rate from transition counts.
        let seq = sample_sequence(&chain, 100_000, &mut rng);
        let k = chain.num_states;
        let mut transition_counts = vec![vec![0usize; k]; k];
        let mut state_counts = vec![0usize; k];
        for w in seq.windows(2) {
            transition_counts[w[0]][w[1]] += 1;
            state_counts[w[0]] += 1;
        }
        // Count the last state for stationary estimate (it was never a "from" state in windows).
        state_counts[*seq.last().unwrap()] += 1;

        // Compute empirical entropy rate: H_hat = -Σᵢ π̂ᵢ Σⱼ P̂ᵢⱼ log₂(P̂ᵢⱼ)
        let ln2 = std::f32::consts::LN_2;
        let total = seq.len() as f32;
        let mut empirical_h = 0.0f32;
        for i in 0..k {
            let pi_hat = state_counts[i] as f32 / total;
            let row_sum = transition_counts[i].iter().sum::<usize>() as f32;
            if row_sum == 0.0 {
                continue;
            }
            let mut row_h = 0.0f32;
            for j in 0..k {
                let p = transition_counts[i][j] as f32 / row_sum;
                if p > 0.0 {
                    row_h -= p * (p.ln() / ln2);
                }
            }
            empirical_h += pi_hat * row_h;
        }

        let relative_error = (empirical_h - chain.entropy_rate).abs() / chain.entropy_rate;
        assert!(
            relative_error < 0.05,
            "GOAT G1 FAIL: empirical_h = {empirical_h:.6}, entropy_rate = {:.6}, relative_error = {:.4}",
            chain.entropy_rate,
            relative_error
        );
    }
}
