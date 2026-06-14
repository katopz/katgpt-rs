//! NFCoT × QGF Fusion — Q-gradient-guided generation with flow-density scoring (Plan 268 T6).
//!
//! Composes [`QGuidedDrafter`] (Plan 268 F1) with [`NfFlowScore`] (Plan 229).
//! The drafter steers generation using the Q-critic gradient; the scorer ranks
//! the resulting candidates by normalizing-flow density, augmented with the
//! same Q-gradient signal.
//!
//! # The Synergy (Research 236 §8)
//!
//! NFCoT FlowScore alone is **MARGINAL** — it scores candidates *post-hoc*
//! but does not steer generation. QGF alone steers generation but has no
//! density-aware ranking signal. Together:
//!
//! 1. **QGF steers** — tilts the marginal toward high-Q actions before sampling.
//! 2. **NFCoT scores** — ranks the tilted candidates by flow density + QGF bonus.
//!
//! The combined scorer (`NfFlowScore::score_with_qgf`) applies the Q-gradient
//! as an additive log-probability bonus at the projection position, which is
//! mathematically equivalent to scoring the tilted marginal with vanilla
//! `flow_score`. See [`super::nf_flow::score_with_qgf`].
//!
//! # Feature Gates
//!
//! Requires both `nf_flow_score` and `qgf_drafter`. Neither is default-ON;
//! both are GOAT-gated. The fusion module inherits both gates.
//!
//! # Zero-Cost When Disabled
//!
//! When `guidance_weight == 0.0`, the drafter is a pass-through and the scorer
//! reduces to vanilla `NfFlowScore::score`. The fusion adds zero overhead in
//! the unguided regime.

use katgpt_core::qgf::QGuidedDrafter;
use katgpt_core::SpeculativeGenerator;

use super::nf_flow::NfFlowScore;
use super::nf_flow_generator::{FlowScoredError, ScoredToken};
use super::spec_generator::{TokenCondition, TokenGenError, TokenOutput};

// ── NfQgfDrafter ──────────────────────────────────────────────────

/// Q-gradient-guided drafter with NFCoT flow-density scoring.
///
/// Wraps a [`QGuidedDrafter`] (QGF guidance) with an [`NfFlowScore`]
/// (NFCoT ranking). At each generation step:
///
/// 1. Generate candidates via the QGF drafter (guidance applied if active).
/// 2. Query the Q-gradient at the projection (top-1 candidate).
/// 3. Score every candidate with NFCoT base score + QGF bonus.
/// 4. Return candidates sorted by descending combined score.
///
/// When `guidance_weight == 0.0`, output is identical to `FlowScoredGenerator`
/// (NFCoT alone). When `guidance_weight > 0`, QGF steers both generation
/// and scoring.
///
/// # Generics
///
/// - `G` — the reference [`SpeculativeGenerator`] (DDTree, PrecisionAware, etc.).
/// - `O` — the [`QGradientOracle`](katgpt_core::QGradientOracle) providing `∇_a Q`.
pub struct NfQgfDrafter<G, O> {
    /// The QGF-guided drafter — applies Q-gradient tilt during generation.
    pub drafter: QGuidedDrafter<G, O>,
    /// The NFCoT scorer — ranks candidates by flow density + QGF bonus.
    pub scorer: NfFlowScore,
}

impl<G, O> NfQgfDrafter<G, O> {
    /// Construct a fusion drafter from a QGF drafter.
    ///
    /// The NFCoT scorer is created fresh with default scratch capacity.
    #[inline]
    pub fn new(drafter: QGuidedDrafter<G, O>) -> Self {
        Self {
            drafter,
            scorer: NfFlowScore::new(),
        }
    }

    /// Access the inner QGF drafter.
    #[inline]
    pub fn drafter(&self) -> &QGuidedDrafter<G, O> {
        &self.drafter
    }

    /// Mutable access to the inner QGF drafter.
    #[inline]
    pub fn drafter_mut(&mut self) -> &mut QGuidedDrafter<G, O> {
        &mut self.drafter
    }

    /// Access the NFCoT scorer.
    #[inline]
    pub fn scorer(&self) -> &NfFlowScore {
        &self.scorer
    }
}

impl<G, O> NfQgfDrafter<G, O>
where
    G: SpeculativeGenerator<
        Condition = TokenCondition,
        Output = TokenOutput,
        Error = TokenGenError,
    >,
    O: katgpt_core::QGradientOracle<State = TokenCondition, Action = TokenOutput>,
{
    /// Construct from raw generator + oracle, with zero guidance weight.
    ///
    /// Equivalent to `NfQgfDrafter::new(QGuidedDrafter::new(generator, oracle))`.
    /// Use `.with_weight(w)` on the result to enable QGF guidance.
    #[inline]
    pub fn from_parts(generator: G, oracle: O) -> Self {
        Self::new(QGuidedDrafter::new(generator, oracle))
    }

    /// Builder: set the QGF guidance weight `1/β`.
    #[inline]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.drafter = self.drafter.with_weight(weight);
        self
    }

    /// Builder: set the QGF guidance period.
    #[inline]
    pub fn with_period(mut self, period: usize) -> Self {
        self.drafter = self.drafter.with_period(period);
        self
    }

    /// Generate candidates with QGF guidance, then score + rank by NFCoT+QGF.
    ///
    /// Pipeline:
    /// 1. `drafter.generate_guided(condition, rng, step)` → candidates.
    /// 2. `drafter.oracle.q_gradient_at(condition, &candidates[0])` → gradient.
    /// 3. `scorer.score_with_qgf(marginals, selected, gradient, weight)` per candidate.
    /// 4. Sort by descending combined score.
    ///
    /// # Arguments
    ///
    /// - `condition` — token context (parent tokens, depth, marginals).
    /// - `marginals_history` — per-position marginals for NFCoT scoring.
    ///   Should include the current position's marginal as the last entry.
    /// - `rng` — RNG for the generator's sampling.
    /// - `step` — generation step index (for QGF period gating).
    ///
    /// # Errors
    ///
    /// - [`FlowScoredError::NoMarginalsHistory`] if `marginals_history` is empty.
    /// - [`FlowScoredError::InnerError`] if the generator fails.
    pub fn generate_guided_scored(
        &mut self,
        condition: &TokenCondition,
        marginals_history: &[Vec<f32>],
        rng: &mut fastrand::Rng,
        step: usize,
    ) -> Result<Vec<ScoredToken>, FlowScoredError> {
        if marginals_history.is_empty() {
            return Err(FlowScoredError::NoMarginalsHistory);
        }

        // 1. Generate candidates (QGF guidance applied inside generate_guided).
        let candidates = self
            .drafter
            .generate_guided(condition, rng, step)
            .map_err(FlowScoredError::InnerError)?;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Query the Q-gradient at the projection (first candidate).
        // Only query when guidance is active — avoids oracle call cost when
        // guidance_weight == 0 or step is outside the period.
        let apply_guidance = self.drafter.should_apply_guidance(step);
        let weight = self.drafter.guidance_weight;

        // 3. Score each candidate. Build the selected trajectory by appending
        //    the candidate's token_idx to the parent tokens.
        //
        // Optimization: query the gradient once (at the projection) and reuse
        //    for all candidates. The gradient is per-action, not per-candidate.
        //
        // Optimization: reuse a single `selected` buffer across candidates
        //    instead of cloning `parent_tokens` per candidate. The buffer is
        //    `parent_tokens.len() + 1` long and is `clear()`-ed each iteration,
        //    keeping allocation to one `with_capacity` per call.
        let mut selected: Vec<usize> = Vec::with_capacity(condition.parent_tokens.len() + 1);
        let mut scored: Vec<ScoredToken> = if apply_guidance {
            let gradient = self.drafter.oracle.q_gradient_at(condition, &candidates[0]);
            candidates
                .into_iter()
                .map(|token| {
                    selected.clear();
                    selected.extend_from_slice(&condition.parent_tokens);
                    selected.push(token.token_idx);
                    let fs = self
                        .scorer
                        .score_with_qgf(marginals_history, &selected, &gradient, weight);
                    ScoredToken {
                        token,
                        flow_score: fs,
                    }
                })
                .collect()
        } else {
            // No guidance this step — vanilla NFCoT scoring (QGF bonus = 0).
            candidates
                .into_iter()
                .map(|token| {
                    selected.clear();
                    selected.extend_from_slice(&condition.parent_tokens);
                    selected.push(token.token_idx);
                    let fs = self.scorer.score(marginals_history, &selected);
                    ScoredToken {
                        token,
                        flow_score: fs,
                    }
                })
                .collect()
        };

        // 4. Sort by descending combined score.
        scored.sort_unstable_by(|a, b| {
            b.flow_score
                .partial_cmp(&a.flow_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored)
    }

    /// Select the best candidate by combined NFCoT+QGF score.
    ///
    /// Convenience wrapper around [`Self::generate_guided_scored`] that
    /// returns only the top-1 candidate. Returns `None` if the generator
    /// produces no candidates.
    #[inline]
    pub fn generate_guided_best(
        &mut self,
        condition: &TokenCondition,
        marginals_history: &[Vec<f32>],
        rng: &mut fastrand::Rng,
        step: usize,
    ) -> Result<Option<ScoredToken>, FlowScoredError> {
        // `generate_guided_scored` sorts by descending score, so the first
        // element is the best. `Vec::pop()` would return the worst (last).
        let scored = self.generate_guided_scored(condition, marginals_history, rng, step)?;
        Ok(scored.first().cloned())
    }
}

// ── SpeculativeGenerator delegation ───────────────────────────────

impl<G, O> SpeculativeGenerator for NfQgfDrafter<G, O>
where
    G: SpeculativeGenerator<
        Condition = TokenCondition,
        Output = TokenOutput,
        Error = TokenGenError,
    >,
    O: katgpt_core::QGradientOracle<State = TokenCondition, Action = TokenOutput>,
{
    type Condition = TokenCondition;
    type Output = TokenOutput;
    type Error = TokenGenError;

    #[inline]
    fn generate(
        &mut self,
        condition: &Self::Condition,
        rng: &mut fastrand::Rng,
    ) -> Result<Vec<Self::Output>, Self::Error> {
        // Delegate to the QGF drafter at step 0 (no period gating by default).
        self.drafter.generate_guided(condition, rng, 0)
    }

    #[inline]
    fn generate_batch(
        &mut self,
        conditions: &[Self::Condition],
        rng: &mut fastrand::Rng,
    ) -> Result<Vec<Vec<Self::Output>>, Self::Error> {
        // Delegate batch to the inner generator via the drafter.
        conditions.iter().map(|c| self.generate(c, rng)).collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::QGradientOracle;

    // ── Mock generator ─────────────────────────────────────────────

    /// Mock generator that returns deterministic candidates with known token_idx.
    struct MockGenerator;

    impl SpeculativeGenerator for MockGenerator {
        type Condition = TokenCondition;
        type Output = TokenOutput;
        type Error = TokenGenError;

        fn generate(
            &mut self,
            _condition: &Self::Condition,
            _rng: &mut fastrand::Rng,
        ) -> Result<Vec<Self::Output>, Self::Error> {
            Ok(vec![
                TokenOutput {
                    token_idx: 0,
                    log_prob: 0.5,
                },
                TokenOutput {
                    token_idx: 1,
                    log_prob: 0.9,
                },
                TokenOutput {
                    token_idx: 2,
                    log_prob: 0.3,
                },
            ])
        }
    }

    /// Mock oracle that returns a known per-action gradient.
    /// Gradient: [g0, g1, g2] — one entry per action index.
    #[derive(Clone)]
    struct MockOracle {
        gradient: Vec<f32>,
    }

    impl QGradientOracle for MockOracle {
        type State = TokenCondition;
        type Action = TokenOutput;

        fn q_gradient_at(&self, _state: &Self::State, _action: &Self::Action) -> Vec<f32> {
            // Return the full gradient vector (caller indexes by action).
            self.gradient.clone()
        }

        fn q_gradient_into(
            &self,
            _state: &Self::State,
            _action: &Self::Action,
            out: &mut [f32],
        ) {
            let n = out.len().min(self.gradient.len());
            out[..n].copy_from_slice(&self.gradient[..n]);
        }
    }

    fn make_condition(parents: &[usize]) -> TokenCondition {
        TokenCondition {
            parent_tokens: parents.to_vec(),
            depth: parents.len(),
            marginals: vec![0.3, 0.5, 0.2],
        }
    }

    // ── Tests: zero guidance weight → matches NFCoT alone ──────────

    #[test]
    fn test_zero_weight_matches_nfcoot_alone() {
        // With guidance_weight = 0, NfQgfDrafter should produce the same
        // ranking as FlowScoredGenerator (NFCoT alone).
        let marginals_history = vec![vec![0.9f32, 0.05, 0.05], vec![0.3, 0.5, 0.2]];
        let oracle = MockOracle {
            gradient: vec![1.0, -1.0, 0.5],
        };

        let drafter = QGuidedDrafter::new(MockGenerator, oracle); // weight = 0
        let mut fusion = NfQgfDrafter::new(drafter);

        let mut rng = fastrand::Rng::new();
        let scored = fusion
            .generate_guided_scored(&make_condition(&[]), &marginals_history, &mut rng, 0)
            .unwrap();

        assert_eq!(scored.len(), 3);
        // With weight = 0, the top candidate should match NFCoT-only ranking.
        // NFCoT favors token 0 (peaked marginal at position 0).
        assert_eq!(scored[0].token.token_idx, 0, "zero-weight top should match NFCoT");
    }

    // ── Tests: QGF + NFCoT > NFCoT alone (ranking flips) ───────────

    #[test]
    fn test_qgf_flips_ranking_when_gradient_strong() {
        // Construct a scenario where NFCoT alone picks candidate A,
        // but a strong Q-gradient endorses candidate B.
        //
        // marginals: [0.9, 0.05, 0.05] → NFCoT strongly prefers token 0
        //   (base log-prob of token 0 is much higher).
        // gradient: [-10.0, 20.0, -5.0] → Q-critic strongly prefers token 1.
        //
        // With sufficient guidance weight, QGF+NFCoT should rank token 1 first.
        let marginals_history = vec![vec![0.9f32, 0.05, 0.05]];
        let oracle = MockOracle {
            gradient: vec![-10.0, 20.0, -5.0],
        };

        // Baseline: NFCoT alone (weight = 0).
        let drafter_nf = QGuidedDrafter::new(MockGenerator, oracle.clone());
        let mut fusion_nf = NfQgfDrafter::new(drafter_nf);
        let mut rng = fastrand::Rng::new();
        let scored_nf = fusion_nf
            .generate_guided_scored(&make_condition(&[]), &marginals_history, &mut rng, 0)
            .unwrap();
        assert_eq!(
            scored_nf[0].token.token_idx, 0,
            "NFCoT alone should prefer token 0 (highest base log-prob)"
        );

        // With QGF: strong gradient flips the ranking to token 1.
        let drafter_qgf = QGuidedDrafter::new(MockGenerator, oracle).with_weight(5.0);
        let mut fusion_qgf = NfQgfDrafter::new(drafter_qgf);
        let mut rng2 = fastrand::Rng::new();
        let scored_qgf = fusion_qgf
            .generate_guided_scored(&make_condition(&[]), &marginals_history, &mut rng2, 0)
            .unwrap();
        assert_eq!(
            scored_qgf[0].token.token_idx, 1,
            "QGF+NFCoT should flip ranking to token 1 (Q-critic preferred)"
        );
    }

    // ── Tests: QGF + NFCoT > QGF alone (NFCoT adds ranking signal) ─

    #[test]
    fn test_nfcoot_breaks_ties_when_gradient_uniform() {
        // When the Q-gradient is uniform (all actions equally preferred by Q),
        // QGF alone cannot discriminate — but NFCoT can via flow density.
        //
        // marginals: [0.1, 0.9] → NFCoT prefers token 1 (higher base log-prob).
        // gradient: [1.0, 1.0] → uniform → QGF adds the same bonus to both.
        //
        // QGF alone: both candidates get +w*1.0 → tie.
        // QGF+NFCoT: NFCoT's base breaks the tie → token 1 wins.
        let marginals_history = vec![vec![0.1f32, 0.9]];
        let oracle = MockOracle {
            gradient: vec![1.0, 1.0], // uniform
        };

        let drafter = QGuidedDrafter::new(MockGenerator, oracle).with_weight(2.0);
        let mut fusion = NfQgfDrafter::new(drafter);
        let mut rng = fastrand::Rng::new();

        // Use a 2-token mock: override the generator's output by checking
        // only tokens 0 and 1. The mock returns 3 tokens; we verify ranking.
        let scored = fusion
            .generate_guided_scored(&make_condition(&[]), &marginals_history, &mut rng, 0)
            .unwrap();

        // Token 1 should rank higher than token 0 (NFCoT base breaks the tie).
        let rank_of_0 = scored.iter().position(|s| s.token.token_idx == 0).unwrap();
        let rank_of_1 = scored.iter().position(|s| s.token.token_idx == 1).unwrap();
        assert!(
            rank_of_1 < rank_of_0,
            "NFCoT should rank token 1 above token 0 (tie-break by base log-prob), got ranks: 1={rank_of_1}, 0={rank_of_0}"
        );
    }

    // ── Tests: period gating ───────────────────────────────────────

    #[test]
    fn test_period_skip_uses_vanilla_nfcoot() {
        // When step is outside the guidance period, QGF is skipped and the
        // scorer uses vanilla NFCoT (no QGF bonus).
        let marginals_history = vec![vec![0.9f32, 0.05, 0.05]];
        let oracle = MockOracle {
            gradient: vec![-10.0, 20.0, -5.0],
        };

        // Period = 2 → guidance only at even steps.
        let drafter = QGuidedDrafter::new(MockGenerator, oracle)
            .with_weight(5.0)
            .with_period(2);
        let mut fusion = NfQgfDrafter::new(drafter);
        let mut rng = fastrand::Rng::new();

        // Step 1 (odd) → no guidance → vanilla NFCoT → token 0 wins.
        let scored = fusion
            .generate_guided_scored(&make_condition(&[]), &marginals_history, &mut rng, 1)
            .unwrap();
        assert_eq!(
            scored[0].token.token_idx, 0,
            "step 1 (odd, period=2) → no QGF → NFCoT alone → token 0"
        );

        // Step 2 (even) → guidance active → token 1 wins.
        let scored = fusion
            .generate_guided_scored(&make_condition(&[]), &marginals_history, &mut rng, 2)
            .unwrap();
        assert_eq!(
            scored[0].token.token_idx, 1,
            "step 2 (even, period=2) → QGF active → token 1"
        );
    }

    // ── Tests: builder + API surface ───────────────────────────────

    #[test]
    fn test_from_parts_and_builders() {
        let oracle = MockOracle {
            gradient: vec![1.0, 0.0, -1.0],
        };
        let fusion = NfQgfDrafter::from_parts(MockGenerator, oracle)
            .with_weight(1.5)
            .with_period(3);

        assert!((fusion.drafter.guidance_weight - 1.5).abs() < 1e-6);
        assert_eq!(fusion.drafter.guidance_period, 3);
    }

    #[test]
    fn test_generate_guided_best_returns_top() {
        let marginals_history = vec![vec![0.9f32, 0.05, 0.05]];
        let oracle = MockOracle {
            gradient: vec![1.0, 0.0, -1.0],
        };
        let drafter = QGuidedDrafter::new(MockGenerator, oracle).with_weight(1.0);
        let mut fusion = NfQgfDrafter::new(drafter);
        let mut rng = fastrand::Rng::new();

        let best = fusion
            .generate_guided_best(&make_condition(&[]), &marginals_history, &mut rng, 0)
            .unwrap();
        assert!(best.is_some(), "should return a best candidate");
        let best = best.unwrap();
        // Token 0 has both highest NFCoT base AND highest gradient → clear winner.
        assert_eq!(best.token.token_idx, 0);
    }

    #[test]
    fn test_empty_marginals_returns_error() {
        let oracle = MockOracle {
            gradient: vec![1.0],
        };
        let drafter = QGuidedDrafter::new(MockGenerator, oracle);
        let mut fusion = NfQgfDrafter::new(drafter);
        let mut rng = fastrand::Rng::new();

        let result = fusion.generate_guided_scored(&make_condition(&[]), &[], &mut rng, 0);
        assert_eq!(result, Err(FlowScoredError::NoMarginalsHistory));
    }

    #[test]
    fn test_speculative_generator_delegation() {
        // The SpeculativeGenerator trait impl should delegate to the drafter.
        let oracle = MockOracle {
            gradient: vec![1.0, 0.0, -1.0],
        };
        let drafter = QGuidedDrafter::new(MockGenerator, oracle).with_weight(0.0);
        let mut fusion = NfQgfDrafter::new(drafter);
        let mut rng = fastrand::Rng::new();

        let result = fusion.generate(&make_condition(&[]), &mut rng).unwrap();
        assert_eq!(result.len(), 3, "trait delegation should return all candidates");
    }

    // ── Tests: Sudoku-like scenario (the unblock) ──────────────────

    #[test]
    fn test_sudoku_like_qgf_nfcoot_synergy() {
        // Simulate a Sudoku-like constraint scenario:
        // - Position 0: peaked marginal [0.85, 0.05, 0.10] (a "given" clue).
        //   NFCoT base strongly prefers token 0 (the clue value).
        // - Position 1: flatter marginal [0.3, 0.4, 0.3] (an empty cell).
        //   NFCoT base slightly prefers token 1.
        //
        // The Q-critic gradient at position 1 (the projection) is:
        //   [-2.0, 5.0, -1.0] → strongly endorses token 1 (the correct fill).
        //
        // NFCoT alone at position 1: token 1 wins by a small margin (0.4 vs 0.3).
        // QGF+NFCoT: token 1 wins by a large margin (QGF bonus amplifies).
        //
        // The "unblock" is that QGF makes the correct Sudoku fill unambiguous,
        // where NFCoT alone was uncertain.
        let marginals_history = vec![
            vec![0.85f32, 0.05, 0.10], // position 0: clue (peaked)
            vec![0.30f32, 0.40, 0.30], // position 1: empty cell (flat)
        ];
        let oracle = MockOracle {
            gradient: vec![-2.0, 5.0, -1.0], // Q-critic: token 1 is correct
        };

        // Baseline: NFCoT alone (weight = 0).
        let drafter_nf = QGuidedDrafter::new(MockGenerator, oracle.clone());
        let mut fusion_nf = NfQgfDrafter::new(drafter_nf);
        let mut rng1 = fastrand::Rng::new();
        let scored_nf = fusion_nf
            .generate_guided_scored(&make_condition(&[0]), &marginals_history, &mut rng1, 0)
            .unwrap();

        // NFCoT alone: token 1 wins narrowly over token 0 at position 1.
        let nf_top = scored_nf[0].token.token_idx;
        let nf_margin = scored_nf[0].flow_score - scored_nf[1].flow_score;
        assert_eq!(nf_top, 1, "NFCoT alone: token 1 should be top");
        // Margin should be small (flat marginal → small base log-prob difference).

        // With QGF: token 1 wins by a much larger margin.
        let drafter_qgf = QGuidedDrafter::new(MockGenerator, oracle).with_weight(2.0);
        let mut fusion_qgf = NfQgfDrafter::new(drafter_qgf);
        let mut rng2 = fastrand::Rng::new();
        let scored_qgf = fusion_qgf
            .generate_guided_scored(&make_condition(&[0]), &marginals_history, &mut rng2, 0)
            .unwrap();

        let qgf_top = scored_qgf[0].token.token_idx;
        let qgf_margin = scored_qgf[0].flow_score - scored_qgf[1].flow_score;
        assert_eq!(qgf_top, 1, "QGF+NFCoT: token 1 should still be top");
        assert!(
            qgf_margin > nf_margin,
            "QGF+NFCoT margin ({qgf_margin:.4}) should exceed NFCoT-alone margin ({nf_margin:.4}) \
             — QGF amplifies the correct candidate's lead"
        );
    }
}

// TL;DR: NfQgfDrafter fuses QGuidedDrafter (Plan 268 F1) with NfFlowScore (Plan 229).
// QGF steers generation via gradient tilt; NFCoT scores candidates by flow density + QGF bonus.
// When guidance_weight == 0, output matches NFCoT alone. When > 0, QGF amplifies the correct
// candidate's lead. Feature-gated on nf_flow_score + qgf_drafter, default OFF until GOAT.
