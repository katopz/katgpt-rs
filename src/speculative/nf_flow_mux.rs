//! NFCoT FlowMUX — Flow scoring for MUX multiplexed trajectories (Plan 229 T6).
//!
//! Scores vocabulary-superposition (MUX) trajectories using NF flow density.
//! MUX compresses low-importance reasoning steps into continuous superposition tokens.
//! FlowScore provides density estimation over the hybrid continuous+discrete trajectory.
//!
//! Requires: `nf_flow_score` + `mux_pruner` features.

/// Score for a MUX multiplexed position.
#[derive(Clone, Debug)]
pub struct MuxFlowScore {
    /// Flow density at this position.
    pub density: f32,
    /// Whether this position was multiplexed (continuous) vs discrete.
    pub is_multiplexed: bool,
}

/// Score a trajectory that mixes discrete tokens with MUX superposition positions.
///
/// For discrete positions: use standard flow_score log-prob contribution.
/// For MUX (multiplexed) positions: use entropy of the superposition distribution.
///
/// `marginals` — per-position marginal distributions
/// `selected` — per-position selected token index (or 0 for MUX positions)
/// `is_mux` — per-position flag: true if MUX superposition, false if discrete
pub fn score_mux_trajectory(
    marginals: &[Vec<f32>],
    selected: &[usize],
    is_mux: &[bool],
) -> Vec<MuxFlowScore> {
    assert_eq!(marginals.len(), selected.len());
    assert_eq!(marginals.len(), is_mux.len());

    let mut scores = Vec::with_capacity(marginals.len());

    for (i, (marg, &sel)) in marginals.iter().zip(selected.iter()).enumerate() {
        let density = if is_mux[i] {
            // For MUX positions: use entropy of the superposition as density
            // Higher entropy = more spread = lower confidence = MUX doing its job
            super::nf_flow::categorical_entropy(marg)
        } else {
            // For discrete positions: standard flow score contribution
            if sel < marg.len() {
                marg[sel].max(1e-10f32).ln()
            } else {
                0.0
            }
        };

        scores.push(MuxFlowScore {
            density,
            is_multiplexed: is_mux[i],
        });
    }

    scores
}

/// Aggregate MUX flow scores into a single trajectory score.
///
/// Sums discrete flow scores and adds a MUX bonus for each superposition position.
/// The MUX bonus is sigmoid(density) — higher entropy superposition = more information preserved.
pub fn aggregate_mux_score(scores: &[MuxFlowScore]) -> f32 {
    let mut total = 0.0f32;
    for s in scores {
        if s.is_multiplexed {
            // MUX bonus: sigmoid(entropy) — values (0.5, 1.0) for positive entropy
            total += super::nf_flow::sigmoid(s.density);
        } else {
            total += s.density;
        }
    }
    total
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mux_trajectory_all_discrete() {
        // No MUX positions → standard log-prob scoring
        let marginals = vec![
            vec![0.7, 0.2, 0.1],
            vec![0.1, 0.8, 0.1],
            vec![0.3, 0.3, 0.4],
        ];
        let selected = vec![0, 1, 2];
        let is_mux = vec![false, false, false];

        let scores = score_mux_trajectory(&marginals, &selected, &is_mux);

        assert_eq!(scores.len(), 3);
        assert!(!scores[0].is_multiplexed);
        assert!(!scores[1].is_multiplexed);
        assert!(!scores[2].is_multiplexed);
        // Discrete: density = ln(marg[sel])
        assert!((scores[0].density - 0.7f32.ln()).abs() < 1e-5);
        assert!((scores[1].density - 0.8f32.ln()).abs() < 1e-5);
        assert!((scores[2].density - 0.4f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_mux_trajectory_all_mux() {
        // All MUX → entropy-based scoring
        let marginals = vec![
            vec![0.5, 0.5],               // H = ln(2) ≈ 0.693
            vec![1.0, 0.0],               // H ≈ 0 (degenerate)
            vec![0.25, 0.25, 0.25, 0.25], // H = ln(4) ≈ 1.386
        ];
        let selected = vec![0, 0, 0];
        let is_mux = vec![true, true, true];

        let scores = score_mux_trajectory(&marginals, &selected, &is_mux);

        assert_eq!(scores.len(), 3);
        assert!(scores[0].is_multiplexed);
        assert!(scores[1].is_multiplexed);
        assert!(scores[2].is_multiplexed);
        // Uniform 2-way: H = ln(2)
        assert!((scores[0].density - 2f32.ln()).abs() < 1e-4);
        // Degenerate: H ≈ 0
        assert!(scores[1].density.abs() < 1e-4);
        // Uniform 4-way: H = ln(4)
        assert!((scores[2].density - 4f32.ln()).abs() < 1e-4);
    }

    #[test]
    fn test_mux_trajectory_mixed() {
        // Mixed discrete+MUX
        let marginals = vec![
            vec![0.9, 0.05, 0.05], // discrete
            vec![0.5, 0.5],        // MUX
            vec![0.1, 0.9],        // discrete
        ];
        let selected = vec![0, 0, 1];
        let is_mux = vec![false, true, false];

        let scores = score_mux_trajectory(&marginals, &selected, &is_mux);

        assert_eq!(scores.len(), 3);
        assert!(!scores[0].is_multiplexed);
        assert!(scores[1].is_multiplexed);
        assert!(!scores[2].is_multiplexed);
        // Position 0: discrete ln(0.9)
        assert!((scores[0].density - 0.9f32.ln()).abs() < 1e-5);
        // Position 1: MUX entropy = ln(2)
        assert!((scores[1].density - 2f32.ln()).abs() < 1e-4);
        // Position 2: discrete ln(0.9)
        assert!((scores[2].density - 0.9f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_aggregate_mux_score() {
        // Mixed: one discrete (density=-1), one MUX (entropy=0.693 → sigmoid≈0.667)
        let scores = vec![
            MuxFlowScore {
                density: (-1.0f32),
                is_multiplexed: false,
            },
            MuxFlowScore {
                density: 2f32.ln(),
                is_multiplexed: true,
            },
        ];
        let agg = aggregate_mux_score(&scores);
        let expected = -1.0 + super::super::nf_flow::sigmoid(2f32.ln());
        assert!((agg - expected).abs() < 1e-5);
    }

    #[test]
    fn test_mux_trajectory_empty() {
        let scores = score_mux_trajectory(&[], &[], &[]);
        assert!(scores.is_empty());
        assert_eq!(aggregate_mux_score(&scores), 0.0);
    }

    #[test]
    #[should_panic]
    fn test_mux_trajectory_length_mismatch() {
        let marginals = vec![vec![0.5, 0.5]];
        let selected = vec![0];
        let is_mux = vec![true, false]; // mismatch: 2 vs 1
        let _ = score_mux_trajectory(&marginals, &selected, &is_mux);
    }
}

// TL;DR: FlowMUX scores MUX vocabulary-superposition trajectories using NF flow density.
// Discrete positions scored via log-prob, MUX positions via categorical entropy.
// Aggregate uses sigmoid(entropy) bonus for superposition positions.
// Feature-gated behind `nf_flow_score` + `mux_pruner`, default OFF until GOAT.
