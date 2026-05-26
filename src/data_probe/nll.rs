//! NLL computation against a known Markov chain.
//!
//! Computes negative log-likelihood of sequences under the true generating
//! distribution, enabling direct comparison with model-estimated perplexity.
//! The first token uses the stationary distribution as prior; subsequent tokens
//! use the transition probability from the previous state.

use super::markov::MarkovChain;

/// Average NLL of sequence against Markov chain: -log p(xⁿ)/n.
///
/// Returns `f32::INFINITY` for empty sequences.
/// Uses natural log internally, matching information-theoretic convention.
pub fn average_nll(chain: &MarkovChain, sequence: &[usize]) -> f32 {
    if sequence.is_empty() {
        return f32::INFINITY;
    }
    let total = nll_profile(chain, sequence);
    total.iter().sum::<f32>() / total.len() as f32
}

/// Full NLL profile: per-position negative log-probabilities.
///
/// Position 0 uses the stationary distribution π[x₀].
/// Position t > 0 uses the transition P[x_{t-1}][x_t].
/// Returns -ln(p) for each position.
pub fn nll_profile(chain: &MarkovChain, sequence: &[usize]) -> Vec<f32> {
    sequence
        .iter()
        .enumerate()
        .map(|(t, &state)| {
            let prob = if t == 0 {
                // First token: stationary distribution prior.
                chain.stationary[state]
            } else {
                // Subsequent tokens: transition probability.
                let prev = sequence[t - 1];
                chain.transition[prev][state]
            };
            if prob > 0.0 {
                -prob.ln()
            } else {
                f32::INFINITY
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_probe::markov::{generate_markov_chain, sample_sequence};

    #[test]
    fn test_average_nll_near_entropy_rate() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(8, 2.0, 1.0, 50, &mut rng);
        // With a long enough sequence, average NLL ≈ entropy rate.
        let seq = sample_sequence(&chain, 10_000, &mut rng);
        let avg_nll = average_nll(&chain, &seq);
        // Convert entropy rate from log₂ to ln for comparison.
        let h_ln = chain.entropy_rate * std::f32::consts::LN_2;
        assert!(
            (avg_nll - h_ln).abs() < 0.3,
            "avg_nll = {avg_nll}, H_ln = {h_ln}, diff = {}",
            (avg_nll - h_ln).abs()
        );
    }

    #[test]
    fn test_nll_profile_length() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 1.0, 1.0, 10, &mut rng);
        let seq = sample_sequence(&chain, 50, &mut rng);
        let profile = nll_profile(&chain, &seq);
        assert_eq!(profile.len(), 50);
        assert!(profile.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_empty_sequence_nll() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 1.0, 1.0, 10, &mut rng);
        assert!(average_nll(&chain, &[]).is_infinite());
        assert!(nll_profile(&chain, &[]).is_empty());
    }

    #[test]
    fn test_single_token_nll() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 1.0, 1.0, 10, &mut rng);
        let seq = vec![0];
        let profile = nll_profile(&chain, &seq);
        assert_eq!(profile.len(), 1);
        // Should be -ln(π[0]).
        let expected = -chain.stationary[0].ln();
        assert!((profile[0] - expected).abs() < 1e-4);
    }

    // GOAT proof G2: NLL convergence — mean NLL of 10K samples → entropy_rate within ε=0.1.
    #[test]
    fn goat_g2_nll_convergence() {
        let mut rng = fastrand::Rng::with_seed(88);
        let chain = generate_markov_chain(4, 1.5, 1.0, 50, &mut rng);

        // Convert entropy rate from bits to nats for direct comparison with NLL.
        let h_nats = chain.entropy_rate * std::f32::consts::LN_2;

        // Sample 10K sequences and compute mean NLL.
        let n_samples = 10_000;
        let seq_len = 100;
        let mut total_nll = 0.0f32;
        for _ in 0..n_samples {
            let seq = sample_sequence(&chain, seq_len, &mut rng);
            total_nll += average_nll(&chain, &seq);
        }
        let mean_nll = total_nll / n_samples as f32;

        let diff = (mean_nll - h_nats).abs();
        assert!(
            diff < 0.1,
            "GOAT G2 FAIL: mean_nll = {mean_nll:.6}, h_nats = {h_nats:.6}, diff = {diff:.6}"
        );
    }
}
