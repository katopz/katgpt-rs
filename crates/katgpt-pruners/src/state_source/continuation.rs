/// Modelless OPD analogue: validator provides short validated continuations
/// from student-sampled states.
#[derive(Debug, Clone)]
pub struct ContinuationScore {
    /// Token sequence of the continuation.
    pub tokens: Vec<usize>,
    /// Fraction of steps that were valid extensions.
    pub valid_density: f32,
    /// Deepest depth reached before dead-end (or max_depth if still live).
    pub reachable_depth: usize,
}

/// Score continuations from a prefix state.
///
/// This is a generic trait that game-specific validators implement.
/// No model is required — validity is checked by the game rules alone.
pub trait ContinuationScorer {
    /// Check if a single continuation step is valid from the given state.
    fn is_valid_step(&self, prefix: &[usize], action: usize) -> bool;

    /// Number of possible actions at a given state.
    fn n_actions(&self, prefix: &[usize]) -> usize;

    /// Generate scored continuations from a prefix.
    ///
    /// Default implementation enumerates valid actions from the prefix,
    /// extends each candidate up to `max_depth`, and scores by valid-continuation
    /// density.
    fn score_continuations(
        &self,
        prefix: &[usize],
        max_depth: usize,
        n_candidates: usize,
        rng: &mut fastrand::Rng,
    ) -> Vec<ContinuationScore> {
        let mut results = Vec::with_capacity(n_candidates);
        let n_act = self.n_actions(prefix);

        if n_act == 0 || max_depth == 0 {
            return results;
        }

        // Collect valid actions from the initial prefix.
        let valid_actions: Vec<usize> = (0..n_act)
            .filter(|&a| self.is_valid_step(prefix, a))
            .collect();

        if valid_actions.is_empty() {
            return results;
        }

        for _ in 0..n_candidates {
            let mut tokens = prefix.to_vec();
            let mut valid_extensions: usize = 0;
            let mut total_attempted: usize = 0;
            let mut reachable_depth: usize = 0;

            for depth in 0..max_depth {
                let cur_n = self.n_actions(&tokens);
                if cur_n == 0 {
                    break;
                }

                // Sample a random action.
                let action = rng.usize(0..cur_n);
                total_attempted += 1;

                if self.is_valid_step(&tokens, action) {
                    tokens.push(action);
                    valid_extensions += 1;
                    reachable_depth = depth + 1;
                } else {
                    // Dead-end at this depth.
                    break;
                }
            }

            let continuation_tokens: Vec<usize> = tokens[prefix.len()..].to_vec();

            let valid_density = if total_attempted > 0 {
                valid_extensions as f32 / total_attempted as f32
            } else {
                0.0
            };

            results.push(ContinuationScore {
                tokens: continuation_tokens,
                valid_density,
                reachable_depth,
            });
        }

        results
    }

    /// Score a beam by its valid-continuation density.
    ///
    /// Returns `base_score` scaled by the mean valid density of sampled
    /// continuations from the beam's endpoint.
    fn beam_score(
        &self,
        beam: &[usize],
        base_score: f32,
        max_depth: usize,
        rng: &mut fastrand::Rng,
    ) -> f32 {
        if beam.is_empty() {
            return base_score;
        }

        let continuations = self.score_continuations(beam, max_depth, 4, rng);

        if continuations.is_empty() {
            return base_score * 0.5; // No valid actions → penalize.
        }

        let mean_density: f32 =
            continuations.iter().map(|c| c.valid_density).sum::<f32>() / continuations.len() as f32;

        base_score * mean_density
    }
}
