//! Three-way regime classification for information-theoretic diagnostics.
//!
//! Classifies sequences into Conservative / Typical / Uncertain regimes
//! based on whether their NLL falls within ε of the source entropy rate H.
//! This implements the typical-set framework from information theory:
//!
//! - **Conservative**: NLL < H − ε  (surprisingly compressible)
//! - **Typical**: H − ε ≤ NLL ≤ H + ε  (AEP typical set)
//! - **Uncertain**: NLL > H + ε  (higher information than expected)

use super::markov::MarkovChain;
use super::nll::average_nll;

/// Regime classification based on NLL vs entropy rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Regime {
    /// NLL < H − ε (surprisingly compressible).
    Conservative,
    /// H − ε ≤ NLL ≤ H + ε (AEP typical set).
    Typical,
    /// NLL > H + ε (higher information than expected).
    Uncertain,
    /// Entropy exceeds critical threshold — solver should switch (Plan 222).
    CriticalInterval,
}

/// Classify entropy as critical interval.
/// Returns true if entropy exceeds threshold.
#[inline]
pub fn is_critical(entropy: f32, h_critical: f32) -> bool {
    entropy >= h_critical
}

/// Classify a single sequence's regime based on its average NLL vs entropy rate.
///
/// Uses natural log internally. The `epsilon` parameter is in nats.
#[inline]
pub fn classify_regime(chain: &MarkovChain, sequence: &[usize], epsilon: f32) -> Regime {
    if sequence.is_empty() {
        // Empty sequences carry no information — classify as conservative.
        return Regime::Conservative;
    }

    let avg_nll = average_nll(chain, sequence);
    // Convert entropy rate from log₂ to nats.
    let h_nats = chain.entropy_rate * std::f32::consts::LN_2;

    if avg_nll < h_nats - epsilon {
        Regime::Conservative
    } else if avg_nll > h_nats + epsilon {
        Regime::Uncertain
    } else {
        Regime::Typical
    }
}

/// Distribution of regime classifications across multiple sequences.
pub struct RegimeDistribution {
    /// Number of sequences classified as Conservative.
    pub n_conservative: usize,
    /// Number of sequences classified as Typical.
    pub n_typical: usize,
    /// Number of sequences classified as Uncertain.
    pub n_uncertain: usize,
    /// Mean NLL across all sequences (in nats).
    pub mean_nll: f32,
}

/// Compute regime distribution across a batch of sequences.
///
/// Computes average NLL once per non-empty sequence (avoids double-computing
/// via `classify_regime` which would call `average_nll` internally).
pub fn regime_distribution(
    chain: &MarkovChain,
    sequences: &[Vec<usize>],
    epsilon: f32,
) -> RegimeDistribution {
    let mut n_conservative = 0usize;
    let mut n_typical = 0usize;
    let mut n_uncertain = 0usize;
    let mut total_nll = 0.0f32;
    let mut n_valid = 0usize;

    // Pre-compute h_nats once for all sequences.
    let h_nats = chain.entropy_rate * std::f32::consts::LN_2;

    for seq in sequences {
        if seq.is_empty() {
            n_conservative += 1;
            continue;
        }

        let avg_nll = average_nll(chain, seq);
        total_nll += avg_nll;
        n_valid += 1;

        let regime = if avg_nll < h_nats - epsilon {
            Regime::Conservative
        } else if avg_nll > h_nats + epsilon {
            Regime::Uncertain
        } else {
            Regime::Typical
        };

        match regime {
            Regime::Conservative => n_conservative += 1,
            Regime::Typical => n_typical += 1,
            Regime::Uncertain => n_uncertain += 1,
            Regime::CriticalInterval => n_uncertain += 1, // Critical intervals are high-entropy, treated as uncertain for counting
        }
    }

    let mean_nll = if n_valid > 0 {
        total_nll / n_valid as f32
    } else {
        0.0
    };

    RegimeDistribution {
        n_conservative,
        n_typical,
        n_uncertain,
        mean_nll,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_probe::markov::{generate_markov_chain, sample_sequence};

    #[test]
    fn test_most_sequences_typical() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(8, 2.0, 1.0, 50, &mut rng);
        let sequences: Vec<Vec<usize>> = (0..100)
            .map(|_| sample_sequence(&chain, 1000, &mut rng))
            .collect();
        let dist = regime_distribution(&chain, &sequences, 0.5);
        // The vast majority of long sequences from the true distribution
        // should be typical (law of large numbers / AEP).
        assert!(
            dist.n_typical > 80,
            "expected >80 typical, got {}/{}/{}/{}",
            dist.n_conservative,
            dist.n_typical,
            dist.n_uncertain,
            sequences.len()
        );
    }

    #[test]
    fn test_regime_counts_sum() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 1.0, 1.0, 10, &mut rng);
        let sequences: Vec<Vec<usize>> = (0..50)
            .map(|_| sample_sequence(&chain, 100, &mut rng))
            .collect();
        let dist = regime_distribution(&chain, &sequences, 0.1);
        assert_eq!(dist.n_conservative + dist.n_typical + dist.n_uncertain, 50);
    }

    #[test]
    fn test_empty_sequence_conservative() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 1.0, 1.0, 10, &mut rng);
        assert_eq!(classify_regime(&chain, &[], 1.0), Regime::Conservative);
    }

    /// Greedy sampling: always pick the highest-probability transition from each state.
    /// Initial state sampled from the stationary distribution.
    fn greedy_sequence(chain: &MarkovChain, n: usize, rng: &mut fastrand::Rng) -> Vec<usize> {
        if n == 0 {
            return Vec::new();
        }
        // Use sample_sequence to get an initial state from the stationary distribution.
        let start = sample_sequence(chain, 1, rng)[0];
        let mut seq = vec![start];
        for _ in 1..n {
            let current = *seq.last().unwrap();
            let next = chain.transition[current]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            seq.push(next);
        }
        seq
    }

    // GOAT proof G3: Greedy sampling → Conservative regime (>80%).
    #[test]
    fn goat_g3_greedy_sampling_conservative() {
        let mut rng = fastrand::Rng::with_seed(42);
        // Low-entropy chain: transitions are concentrated, so greedy picks dominant path.
        // Use alpha=0.05 for very peaked transitions, ensuring greedy NLL << entropy rate.
        let chain = generate_markov_chain(4, 0.5, 0.05, 100, &mut rng);

        // Use a small epsilon (in nats) so the conservative zone is reachable.
        // Greedy NLL per step ≈ -ln(max_prob) << H_nats.
        let epsilon = 0.1;
        let n_samples = 200;
        let seq_len = 1000;
        let mut n_conservative = 0;
        for _ in 0..n_samples {
            let seq = greedy_sequence(&chain, seq_len, &mut rng);
            if classify_regime(&chain, &seq, epsilon) == Regime::Conservative {
                n_conservative += 1;
            }
        }
        let ratio = n_conservative as f32 / n_samples as f32;
        assert!(
            ratio > 0.8,
            "GOAT G3 FAIL: conservative ratio = {ratio:.2} ({n_conservative}/{n_samples}), expected > 0.80"
        );
    }

    // GOAT proof G4: T=1 sampling → Typical regime (>60%).
    #[test]
    fn goat_g4_typical_sampling() {
        let mut rng = fastrand::Rng::with_seed(123);
        let chain = generate_markov_chain(8, 2.0, 1.0, 50, &mut rng);

        let n_samples = 200;
        let seq_len = 1000;
        let epsilon = 0.5;
        let sequences: Vec<Vec<usize>> = (0..n_samples)
            .map(|_| sample_sequence(&chain, seq_len, &mut rng))
            .collect();
        let dist = regime_distribution(&chain, &sequences, epsilon);
        let ratio = dist.n_typical as f32 / n_samples as f32;
        assert!(
            ratio > 0.6,
            "GOAT G4 FAIL: typical ratio = {ratio:.2} ({}/{n_samples}), expected > 0.60",
            dist.n_typical
        );
    }

    // GOAT proof G5: Uniform random sampling → Uncertain regime (>80%).
    #[test]
    fn goat_g5_uniform_sampling_uncertain() {
        let mut rng = fastrand::Rng::with_seed(456);
        // Low-entropy chain: uniform random tokens have much higher NLL than H.
        let chain = generate_markov_chain(4, 0.8, 0.1, 50, &mut rng);

        let n_samples = 200;
        let seq_len = 100;
        let epsilon = 0.3;
        let mut n_uncertain = 0;
        for _ in 0..n_samples {
            // Generate uniform random sequence independent of the chain.
            let seq: Vec<usize> = (0..seq_len)
                .map(|_| rng.usize(0..chain.num_states))
                .collect();
            if classify_regime(&chain, &seq, epsilon) == Regime::Uncertain {
                n_uncertain += 1;
            }
        }
        let ratio = n_uncertain as f32 / n_samples as f32;
        assert!(
            ratio > 0.8,
            "GOAT G5 FAIL: uncertain ratio = {ratio:.2} ({n_uncertain}/{n_samples}), expected > 0.80"
        );
    }
}
