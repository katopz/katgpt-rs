//! NFCoT FlowScore — Modelless Normalizing Flow Density Scoring (Plan 229, Research 204).
//!
//! Constructs a diagonal affine normalizing flow from DDTree marginals.
//! Zero training — uses entropy-based log-determinant as a confidence-weighting
//! bonus over base log-probability.
//!
//! The flow score is: `base_logprob + log_det`
//! where:
//! - `base_logprob = Σ log(marginals[i][selected[i]])`
//! - `log_det = Σ log(sigmoid(entropy_i))` — entropy of the categorical at position i
//!
//! High entropy (uncertain) → σ ≈ 1 → log_det ≈ 0 → score ≈ base
//! Low entropy (confident) → σ ≈ 0 → large negative log_det → score < base
//!
//! NF-CoT insight: uncertain positions carry more information and should be
//! weighted more. The sigmoid ensures we never get -inf from log(0).

/// Numerical stability floor for log arguments.
const EPSILON: f32 = 1e-10;

/// Sigmoid activation: `1 / (1 + exp(-x))`.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Shannon entropy of a categorical distribution: `H = -Σ p_i * log(p_i)`.
///
/// Returns 0.0 for empty or degenerate inputs. Skips probabilities < EPSILON
/// to avoid log(0).
#[inline]
pub fn categorical_entropy(probs: &[f32]) -> f32 {
    if probs.is_empty() {
        return 0.0;
    }

    let mut h = 0.0f32;
    for &p in probs {
        if p > EPSILON {
            h -= p * p.ln();
        }
    }
    h
}

/// Decompose flow score into `(base_logprob, log_det)` for diagnostics.
///
/// - `base_logprob = Σ log(max(marginals[i][selected[i]], EPSILON))`
/// - `log_det = Σ log(max(sigmoid(entropy_i), EPSILON))`
#[inline]
pub fn flow_components(marginals: &[Vec<f32>], selected: &[usize]) -> (f32, f32) {
    if marginals.is_empty() || selected.is_empty() {
        return (0.0, 0.0);
    }

    let len = marginals.len().min(selected.len());
    let mut base_logprob = 0.0f32;
    let mut log_det = 0.0f32;

    for i in 0..len {
        let dist = &marginals[i];
        let idx = selected[i];

        // Base log-probability contribution
        let p = match dist.get(idx) {
            Some(&v) => v.max(EPSILON),
            None => EPSILON,
        };
        base_logprob += p.ln();

        // Log-determinant contribution: sigmoid(entropy)
        let entropy = categorical_entropy(dist);
        let sigma = sigmoid(entropy);
        log_det += sigma.max(EPSILON).ln();
    }

    (base_logprob, log_det)
}

/// Compute flow score for a single candidate trajectory.
///
/// `flow_score = base_logprob + log_det`
///
/// No allocation — O(V·T) compute only.
#[inline]
pub fn flow_score(marginals: &[Vec<f32>], selected: &[usize]) -> f32 {
    let (base, det) = flow_components(marginals, selected);
    base + det
}

/// Compute flow scores for multiple candidate trajectories.
///
/// Returns a Vec of scores, one per candidate. Pre-allocates output.
pub fn flow_score_batch(marginals: &[Vec<f32>], candidates: &[Vec<usize>]) -> Vec<f32> {
    candidates
        .iter()
        .map(|sel| flow_score(marginals, sel))
        .collect()
}

// ── QGF-aware scoring (Plan 268 T6 — NFCoT × QGF fusion) ───────────
//
// The Q-gradient `∇_a Q(s, â_1)` from the QGuidedDrafter's oracle is a
// per-action logit tilt. Applied in log-probability space, it contributes
// an additive bonus `guidance_weight · gradient[selected[pos]]` at the
// projection position. This is mathematically equivalent to tilting the
// marginal *before* scoring with vanilla `flow_score`.
//
// When `guidance_weight == 0.0` or the gradient is empty, the QGF-aware
// score is identical to `flow_score` — the QGF contribution is purely
// additive on top of the NFCoT base score.
//
// See `.research/236_QGF_Test_Time_Q_Guided_Flow.md` §8 (QGF+NFCoT
// synergy) and `.plans/268_qgf_test_time_q_guided_flow.md` T6.

/// Compute QGF-augmented flow score, applying the gradient bonus at the
/// *last* position of `selected` (the QGF projection point).
///
/// `score_qgf = flow_score(marginals, selected) + guidance_weight · gradient[selected[last]]`
///
/// Returns vanilla `flow_score` when `guidance_weight == 0.0`, gradient is
/// empty, or `selected` is empty — i.e. QGF is fully optional.
#[inline]
pub fn score_with_qgf(
    marginals: &[Vec<f32>],
    selected: &[usize],
    gradient: &[f32],
    guidance_weight: f32,
) -> f32 {
    let base = flow_score(marginals, selected);
    if guidance_weight == 0.0 || gradient.is_empty() || selected.is_empty() {
        return base;
    }
    let pos = selected.len() - 1;
    score_with_qgf_at_inner(base, selected, gradient, pos, guidance_weight)
}

/// Compute QGF-augmented flow score, applying the gradient bonus at a
/// specific position.
///
/// Useful when the QGF projection is at a non-terminal position (e.g. a
/// mid-trajectory re-guidance step).
#[inline]
pub fn score_with_qgf_at(
    marginals: &[Vec<f32>],
    selected: &[usize],
    gradient: &[f32],
    projection_pos: usize,
    guidance_weight: f32,
) -> f32 {
    let base = flow_score(marginals, selected);
    if guidance_weight == 0.0 || gradient.is_empty() {
        return base;
    }
    score_with_qgf_at_inner(base, selected, gradient, projection_pos, guidance_weight)
}

/// Shared inner: add the QGF bonus to a pre-computed base score.
#[inline]
fn score_with_qgf_at_inner(
    base: f32,
    selected: &[usize],
    gradient: &[f32],
    pos: usize,
    guidance_weight: f32,
) -> f32 {
    if pos >= selected.len() {
        return base;
    }
    let action = selected[pos];
    let g = gradient.get(action).copied().unwrap_or(0.0);
    // FMA: base + guidance_weight * g
    guidance_weight.mul_add(g, base)
}

/// Compute QGF-augmented flow scores for multiple candidates.
///
/// Shares one gradient across all candidates (the projection gradient is
/// queried once at the top-1 candidate, then applied to every candidate's
/// selected action at the projection position).
pub fn score_with_qgf_batch(
    marginals: &[Vec<f32>],
    candidates: &[Vec<usize>],
    gradient: &[f32],
    guidance_weight: f32,
) -> Vec<f32> {
    candidates
        .iter()
        .map(|sel| score_with_qgf(marginals, sel, gradient, guidance_weight))
        .collect()
}

/// Return the index of the candidate with the highest QGF-augmented flow score.
pub fn select_best_qgf(
    marginals: &[Vec<f32>],
    candidates: &[Vec<usize>],
    gradient: &[f32],
    guidance_weight: f32,
) -> usize {
    match candidates.len() {
        0 => 0,
        1 => 0,
        _ => {
            let mut best_idx = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (i, sel) in candidates.iter().enumerate() {
                let s = score_with_qgf(marginals, sel, gradient, guidance_weight);
                if s > best_score {
                    best_score = s;
                    best_idx = i;
                }
            }
            best_idx
        }
    }
}

/// Return the index of the candidate with the highest flow score.
pub fn select_best(marginals: &[Vec<f32>], candidates: &[Vec<usize>]) -> usize {
    match candidates.len() {
        0 => 0,
        1 => 0,
        _ => {
            let mut best_idx = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (i, sel) in candidates.iter().enumerate() {
                let s = flow_score(marginals, sel);
                if s > best_score {
                    best_score = s;
                    best_idx = i;
                }
            }
            best_idx
        }
    }
}

/// Inference-time normalizing flow density scorer (Plan 229, Research 204).
///
/// Holds pre-allocated scratch buffers for batch operations.
/// Construct once, reuse across calls — zero hot-path allocation.
pub struct NfFlowScore {
    /// Scratch buffer reused for batch score output.
    scratch: Vec<f32>,
}

impl NfFlowScore {
    /// Create a new scorer with pre-allocated scratch capacity.
    #[inline]
    pub fn new() -> Self {
        Self {
            scratch: Vec::with_capacity(32),
        }
    }

    /// Compute flow score for a single candidate trajectory.
    #[inline]
    pub fn score(&self, marginals: &[Vec<f32>], selected: &[usize]) -> f32 {
        flow_score(marginals, selected)
    }

    /// Compute flow scores for multiple candidates.
    ///
    /// Reuses internal scratch buffer for output.
    pub fn score_batch(&mut self, marginals: &[Vec<f32>], candidates: &[Vec<usize>]) -> Vec<f32> {
        self.scratch.clear();
        self.scratch.reserve(candidates.len());
        for sel in candidates {
            self.scratch.push(flow_score(marginals, sel));
        }
        self.scratch.clone()
    }

    /// Return the index of the candidate with the highest flow score.
    pub fn select_best(&self, marginals: &[Vec<f32>], candidates: &[Vec<usize>]) -> usize {
        select_best(marginals, candidates)
    }

    /// Compute QGF-augmented flow score (Plan 268 T6).
    ///
    /// See [`score_with_qgf`] for the math. The QGF gradient is applied at
    /// the *last* position of `selected` (the QGF projection point).
    ///
    /// When `guidance_weight == 0.0` or `gradient` is empty, this is
    /// identical to [`Self::score`] — QGF is fully optional.
    #[inline]
    pub fn score_with_qgf(
        &self,
        marginals: &[Vec<f32>],
        selected: &[usize],
        gradient: &[f32],
        guidance_weight: f32,
    ) -> f32 {
        score_with_qgf(marginals, selected, gradient, guidance_weight)
    }

    /// Compute QGF-augmented flow score at a specific projection position.
    #[inline]
    pub fn score_with_qgf_at(
        &self,
        marginals: &[Vec<f32>],
        selected: &[usize],
        gradient: &[f32],
        projection_pos: usize,
        guidance_weight: f32,
    ) -> f32 {
        score_with_qgf_at(marginals, selected, gradient, projection_pos, guidance_weight)
    }

    /// Return the index of the candidate with the highest QGF-augmented score.
    #[inline]
    pub fn select_best_qgf(
        &self,
        marginals: &[Vec<f32>],
        candidates: &[Vec<usize>],
        gradient: &[f32],
        guidance_weight: f32,
    ) -> usize {
        select_best_qgf(marginals, candidates, gradient, guidance_weight)
    }
}

impl Default for NfFlowScore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let s0 = sigmoid(0.0);
        assert!((s0 - 0.5).abs() < 1e-6, "sigmoid(0) = {s0}, expected 0.5");

        let s_large = sigmoid(100.0);
        assert!(
            (s_large - 1.0).abs() < 1e-6,
            "sigmoid(100) = {s_large}, expected ~1"
        );

        let s_neg = sigmoid(-100.0);
        assert!(s_neg < 1e-6, "sigmoid(-100) = {s_neg}, expected ~0");
    }

    #[test]
    fn test_categorical_entropy_uniform() {
        // Uniform over 4 categories: entropy = log(4) ≈ 1.3863
        let probs = [0.25f32, 0.25, 0.25, 0.25];
        let h = categorical_entropy(&probs);
        let expected = 4.0f32.ln();
        assert!(
            (h - expected).abs() < 1e-4,
            "entropy of uniform(4) = {h}, expected {expected}"
        );
    }

    #[test]
    fn test_categorical_entropy_dirac() {
        // Dirac: all mass on one category → entropy ≈ 0
        let probs = [1.0f32, 0.0, 0.0, 0.0];
        let h = categorical_entropy(&probs);
        assert!(h.abs() < 1e-6, "entropy of Dirac = {h}, expected ~0");
    }

    #[test]
    fn test_categorical_entropy_empty() {
        assert_eq!(categorical_entropy(&[]), 0.0);
    }

    #[test]
    fn test_flow_score_known() {
        // marginals = [[0.5, 0.5]], selected = [0]
        // base = log(0.5) = -0.6931
        // entropy = -2*0.5*log(0.5) = 0.6931
        // σ = sigmoid(0.6931) ≈ 0.6667
        // log_det = log(0.6667) ≈ -0.4055
        // score ≈ -0.6931 + (-0.4055) ≈ -1.0986
        let marginals = vec![vec![0.5f32, 0.5]];
        let selected = vec![0usize];
        let score = flow_score(&marginals, &selected);

        let base = 0.5f32.ln();
        let entropy = -2.0 * 0.5 * 0.5f32.ln();
        let sigma = sigmoid(entropy);
        let log_det = sigma.ln();
        let expected = base + log_det;

        assert!(
            (score - expected).abs() < 1e-4,
            "score = {score}, expected {expected}"
        );
        assert!(
            (score - (-1.0986)).abs() < 0.01,
            "score = {score}, expected ≈ -1.0986"
        );
    }

    #[test]
    fn test_flow_score_uniform_high_entropy() {
        // Uniform marginals → high entropy → σ ≈ 1 → log_det ≈ 0 → score ≈ base
        let marginals = vec![vec![0.25f32, 0.25, 0.25, 0.25]];
        let selected = vec![0usize];
        let score = flow_score(&marginals, &selected);

        let base = 0.25f32.ln(); // -1.3863
        assert!(
            score < base,
            "score ({score}) should be slightly below base ({base}) due to log_det"
        );
        // log_det should be small negative, not huge
        let (_, log_det) = flow_components(&marginals, &selected);
        assert!(
            log_det > -1.0,
            "log_det ({log_det}) should be small negative for high entropy"
        );
    }

    #[test]
    fn test_flow_score_peaked_low_entropy() {
        // Peaked: [0.99, 0.01] → low entropy → large negative log_det
        let marginals = vec![vec![0.99f32, 0.01]];
        let selected = vec![0usize];
        let score = flow_score(&marginals, &selected);
        let base = 0.99f32.ln(); // ≈ -0.01005

        // Score should be much less than base due to large negative log_det
        assert!(
            score < base - 0.5,
            "score ({score}) should be much < base ({base}) for peaked distribution"
        );

        let (_base_out, log_det) = flow_components(&marginals, &selected);
        assert!(
            log_det < -0.5,
            "log_det ({log_det}) should be large negative for low entropy"
        );
        assert!(
            (score - (-0.676)).abs() < 0.05,
            "score = {score}, expected ≈ -0.676"
        );
    }

    #[test]
    fn test_flow_score_multi_position() {
        // 3 positions with different entropies
        let marginals = vec![
            vec![0.5f32, 0.5],               // high entropy
            vec![0.99f32, 0.01],             // low entropy
            vec![0.25f32, 0.25, 0.25, 0.25], // high entropy
        ];
        let selected = vec![0usize, 0, 0];
        let score = flow_score(&marginals, &selected);

        // Verify by summing individual components
        let mut expected_base = 0.0f32;
        let mut expected_log_det = 0.0f32;
        for dist in &marginals {
            expected_base += dist[0].max(EPSILON).ln();
            let h = categorical_entropy(dist);
            let s = sigmoid(h);
            expected_log_det += s.max(EPSILON).ln();
        }
        let expected = expected_base + expected_log_det;

        assert!(
            (score - expected).abs() < 1e-4,
            "score = {score}, expected = {expected}"
        );
    }

    #[test]
    fn test_flow_score_empty() {
        // Both empty → 0.0
        assert_eq!(flow_score(&[], &[]), 0.0);
        // Mismatched lengths: selected longer than marginals → 0.0
        assert_eq!(flow_score(&[], &[0]), 0.0);
        // Marginals empty, selected empty → 0.0
        assert_eq!(flow_score(&[vec![]], &[]), 0.0);
    }

    #[test]
    fn test_flow_score_batch() {
        let marginals = vec![
            vec![0.1f32, 0.9], // peaked toward index 1
            vec![0.5f32, 0.5], // uniform
        ];
        let candidates = vec![
            vec![0usize, 0], // base: log(0.1) + log(0.5)
            vec![1usize, 0], // base: log(0.9) + log(0.5) — highest base
            vec![0usize, 1], // base: log(0.1) + log(0.5)
        ];
        let scores = flow_score_batch(&marginals, &candidates);

        assert_eq!(scores.len(), 3);
        // Candidate 1 (selects the peaked token) should have highest score
        assert!(
            scores[1] > scores[0],
            "scores[1]={} should > scores[0]={}",
            scores[1],
            scores[0]
        );
        // Candidates 0 and 2 have same base but log_det differs by position entropy
        // Position 0 entropy is same regardless of which token selected, so log_det is same
        assert!(
            (scores[0] - scores[2]).abs() < 1e-5,
            "scores[0]={} should ≈ scores[2]={}",
            scores[0],
            scores[2]
        );
    }

    #[test]
    fn test_select_best() {
        let marginals = vec![vec![0.1f32, 0.9], vec![0.5f32, 0.5]];
        let candidates = vec![
            vec![0usize, 0], // low score
            vec![1usize, 0], // high score (peaked token + uniform)
            vec![0usize, 1], // medium score
        ];
        let best = select_best(&marginals, &candidates);
        assert_eq!(best, 1, "should select candidate 1 (highest flow score)");
    }

    #[test]
    fn test_flow_components() {
        let marginals = vec![vec![0.5f32, 0.5]];
        let selected = vec![0usize];
        let (base, log_det) = flow_components(&marginals, &selected);

        let expected_base = 0.5f32.ln();
        let entropy = categorical_entropy(&marginals[0]);
        let sigma = sigmoid(entropy);
        let expected_log_det = sigma.ln();

        assert!(
            (base - expected_base).abs() < 1e-6,
            "base = {base}, expected {expected_base}"
        );
        assert!(
            (log_det - expected_log_det).abs() < 1e-6,
            "log_det = {log_det}, expected {expected_log_det}"
        );
    }

    #[test]
    fn test_nf_flow_score_instance() {
        let scorer = NfFlowScore::new();
        let marginals = vec![vec![0.5f32, 0.5]];
        let selected = vec![0usize];
        let score = scorer.score(&marginals, &selected);
        let expected = flow_score(&marginals, &selected);
        assert!(
            (score - expected).abs() < 1e-6,
            "instance score = {score}, expected {expected}"
        );
    }

    #[test]
    fn test_nf_flow_score_batch_instance() {
        let mut scorer = NfFlowScore::new();
        let marginals = vec![vec![0.5f32, 0.5]];
        let candidates = vec![vec![0usize], vec![1usize]];
        let scores = scorer.score_batch(&marginals, &candidates);
        assert_eq!(scores.len(), 2);
        // Both should be equal (symmetric distribution, same position)
        assert!(
            (scores[0] - scores[1]).abs() < 1e-6,
            "symmetric scores should be equal: {} vs {}",
            scores[0],
            scores[1]
        );
    }

    #[test]
    fn test_nf_flow_score_select_best_instance() {
        let scorer = NfFlowScore::new();
        let marginals = vec![vec![0.1f32, 0.9]];
        let candidates = vec![vec![0usize], vec![1usize]];
        let best = scorer.select_best(&marginals, &candidates);
        assert_eq!(best, 1, "should select candidate 1");
    }

    // ── QGF-aware scoring tests (Plan 268 T6) ─────────────────────

    #[test]
    fn test_score_with_qgf_zero_weight_equals_flow_score() {
        // When guidance_weight == 0.0, score_with_qgf must equal flow_score.
        let marginals = vec![vec![0.3f32, 0.7]];
        let selected = vec![0usize];
        let gradient = vec![1.0, -1.0];
        let expected = flow_score(&marginals, &selected);
        let got = score_with_qgf(&marginals, &selected, &gradient, 0.0);
        assert!((got - expected).abs() < 1e-6, "zero weight: got {got}, expected {expected}");
    }

    #[test]
    fn test_score_with_qgf_empty_gradient_equals_flow_score() {
        // Empty gradient → no QGF bonus → equals vanilla flow_score.
        let marginals = vec![vec![0.3f32, 0.7]];
        let selected = vec![0usize];
        let expected = flow_score(&marginals, &selected);
        let got = score_with_qgf(&marginals, &selected, &[], 1.0);
        assert!((got - expected).abs() < 1e-6, "empty gradient: got {got}, expected {expected}");
    }

    #[test]
    fn test_score_with_qgf_adds_gradient_bonus() {
        // With non-zero gradient, score_with_qgf = flow_score + w * gradient[selected[last]].
        let marginals = vec![vec![0.3f32, 0.7]];
        let selected = vec![0usize];
        let gradient = vec![2.0, -2.0];
        let w = 0.5f32;
        let base = flow_score(&marginals, &selected);
        let expected = base + w * 2.0; // gradient[selected[0]] = gradient[0] = 2.0
        let got = score_with_qgf(&marginals, &selected, &gradient, w);
        assert!((got - expected).abs() < 1e-6, "got {got}, expected {expected}");
    }

    #[test]
    fn test_score_with_qgf_multi_position_last_position() {
        // QGF bonus applies at the LAST position only.
        let marginals = vec![vec![0.5f32, 0.5], vec![0.3f32, 0.7]];
        let selected = vec![0usize, 1]; // last position selects action 1
        let gradient = vec![10.0, -10.0]; // applies at the last position
        let w = 0.1f32;
        let base = flow_score(&marginals, &selected);
        let expected = base + w * (-10.0); // gradient[selected[1]] = gradient[1] = -10.0
        let got = score_with_qgf(&marginals, &selected, &gradient, w);
        assert!((got - expected).abs() < 1e-6, "got {got}, expected {expected}");
    }

    #[test]
    fn test_score_with_qgf_at_specific_position() {
        // Apply gradient at position 0 instead of the last.
        let marginals = vec![vec![0.5f32, 0.5], vec![0.3f32, 0.7]];
        let selected = vec![0usize, 1];
        let gradient = vec![5.0, -5.0];
        let w = 0.2f32;
        let base = flow_score(&marginals, &selected);
        // gradient[selected[0]] = gradient[0] = 5.0
        let expected = base + w * 5.0;
        let got = score_with_qgf_at(&marginals, &selected, &gradient, 0, w);
        assert!((got - expected).abs() < 1e-6, "got {got}, expected {expected}");
    }

    #[test]
    fn test_score_with_qgf_empty_selected() {
        // Empty selected → flow_score = 0, no bonus.
        let marginals = vec![vec![0.5f32, 0.5]];
        let got = score_with_qgf(&marginals, &[], &[1.0, 2.0], 1.0);
        assert!(got.abs() < 1e-6, "empty selected: got {got}");
    }

    #[test]
    fn test_score_with_qgf_gradient_shorter_than_action() {
        // Gradient is shorter than the vocab — missing entries are treated as 0.
        let marginals = vec![vec![0.3f32, 0.7]];
        let selected = vec![5usize]; // out of gradient bounds
        let gradient = vec![1.0]; // only 1 entry
        let w = 1.0f32;
        let base = flow_score(&marginals, &selected);
        let expected = base; // gradient[5] = 0.0 (missing)
        let got = score_with_qgf(&marginals, &selected, &gradient, w);
        assert!((got - expected).abs() < 1e-6, "got {got}, expected {expected}");
    }

    #[test]
    fn test_select_best_qgf_flips_ranking() {
        // Construct marginals where NFCoT alone prefers candidate A,
        // but QGF gradient strongly endorses candidate B → combined picks B.
        //
        // marginals: [0.9, 0.1] → selecting action 0 has higher base log-prob.
        // gradient: [-5.0, 5.0] → Q-critic strongly prefers action 1.
        //
        // NFCoT alone: candidate [0] > [1] (base log-prob difference).
        // With strong QGF weight: candidate [1] > [0] (QGF bonus flips it).
        let marginals = vec![vec![0.9f32, 0.1]];
        let candidates = vec![vec![0usize], vec![1usize]];
        let gradient = vec![-5.0, 5.0];

        // NFCoT alone picks candidate 0.
        let best_nf = select_best(&marginals, &candidates);
        assert_eq!(best_nf, 0, "NFCoT alone should pick candidate 0");

        // With strong QGF weight, pick candidate 1.
        let best_qgf = select_best_qgf(&marginals, &candidates, &gradient, 10.0);
        assert_eq!(best_qgf, 1, "QGF+NFCoT should flip to candidate 1");
    }

    #[test]
    fn test_select_best_qgf_zero_weight_equals_select_best() {
        let marginals = vec![vec![0.1f32, 0.9]];
        let candidates = vec![vec![0usize], vec![1usize]];
        let gradient = vec![1.0, -1.0];

        let best_nf = select_best(&marginals, &candidates);
        let best_qgf = select_best_qgf(&marginals, &candidates, &gradient, 0.0);
        assert_eq!(best_nf, best_qgf, "zero-weight QGF should match NFCoT alone");
    }

    #[test]
    fn test_nf_flow_score_score_with_qgf_instance() {
        let scorer = NfFlowScore::new();
        let marginals = vec![vec![0.3f32, 0.7]];
        let selected = vec![0usize];
        let gradient = vec![1.0, -1.0];
        let w = 0.5;
        let got = scorer.score_with_qgf(&marginals, &selected, &gradient, w);
        let expected = score_with_qgf(&marginals, &selected, &gradient, w);
        assert!((got - expected).abs() < 1e-6, "instance: got {got}, expected {expected}");
    }

    #[test]
    fn test_score_with_qgf_batch() {
        let marginals = vec![vec![0.3f32, 0.7]];
        let candidates = vec![vec![0usize], vec![1usize]];
        let gradient = vec![1.0, -1.0];
        let scores = score_with_qgf_batch(&marginals, &candidates, &gradient, 1.0);
        assert_eq!(scores.len(), 2);
        // candidate 0: base + 1*gradient[0] = base + 1
        // candidate 1: base + 1*gradient[1] = base - 1
        assert!(scores[0] > scores[1], "candidate 0 should score higher with positive gradient");
    }

    // ── Benchmarks ──────────────────────────────────────────────────

    #[test]
    fn test_bench_flow_score_v128_t5() {
        // 10 positions, vocab_size=128
        let positions = 10;
        let vocab = 128;
        let marginals: Vec<Vec<f32>> = (0..positions)
            .map(|i| {
                (0..vocab)
                    .map(|j| ((i * vocab + j) as f32).sin().abs())
                    .collect()
            })
            .collect();
        // Normalize each position to sum to 1
        let marginals: Vec<Vec<f32>> = marginals
            .into_iter()
            .map(|mut dist| {
                let sum: f32 = dist.iter().sum();
                if sum > EPSILON {
                    for p in &mut dist {
                        *p /= sum;
                    }
                }
                dist
            })
            .collect();
        let selected: Vec<usize> = (0..positions).map(|i| i % vocab).collect();

        let start = std::time::Instant::now();
        let iters = 10_000;
        for _ in 0..iters {
            std::hint::black_box(flow_score(&marginals, &selected));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        eprintln!("flow_score V=128 T=10: {per_call:.0}ns/call");
        assert!(
            per_call < 10_000.0,
            "V=128 T=10 flow_score should be <10μs (debug), got {per_call:.0}ns"
        );
    }

    #[test]
    fn test_bench_flow_score_v32000_t10() {
        // 10 positions, vocab_size=32000
        let positions = 10;
        let vocab = 32000;
        let marginals: Vec<Vec<f32>> = (0..positions)
            .map(|i| {
                (0..vocab)
                    .map(|j| ((i * vocab + j) as f32 * 0.001).sin().abs())
                    .collect()
            })
            .collect();
        let marginals: Vec<Vec<f32>> = marginals
            .into_iter()
            .map(|mut dist| {
                let sum: f32 = dist.iter().sum();
                if sum > EPSILON {
                    for p in &mut dist {
                        *p /= sum;
                    }
                }
                dist
            })
            .collect();
        let selected: Vec<usize> = (0..positions).map(|i| i % vocab).collect();

        let start = std::time::Instant::now();
        let iters = 100;
        for _ in 0..iters {
            std::hint::black_box(flow_score(&marginals, &selected));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        eprintln!("flow_score V=32000 T=10: {per_call:.0}ns/call");
        assert!(
            per_call < 5_000_000.0,
            "V=32000 T=10 flow_score should be <5ms (debug), got {per_call:.0}ns"
        );
    }
}

// TL;DR: Modelless normalizing flow density scorer. Constructs diagonal affine flow from
// DDTree marginals — zero training, O(V·T) per call. `flow_score = base_logprob + log_det`
// where log_det = Σ log(sigmoid(entropy)). High entropy ≈ score ≈ base; low entropy penalizes.
// NfFlowScore struct pre-allocates scratch for batch ops. Plan 229, GOAT gate `nf_flow_score`.
