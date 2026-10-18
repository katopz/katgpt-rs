//! branch_routing — post-candidate branch router (Plan 377 Phase 2).
//!
//! Distilled from Local Branch Routing (arXiv:2606.25354, Yin et al. June
//! 2026). The modelless inference mechanism: forward K candidate next-tokens,
//! score each post-candidate hidden state, commit the argmax (or
//! perturbed-argmax sample).
//!
//! Generalizes `ColliderPruner::batch_is_valid_with_hidden` (in katgpt-rs
//! root) from binary prune/keep to relative route-and-commit. The PoC at
//! `riir-ai/crates/riir-poc/benches/lbr_modelless_goat.rs` confirmed a
//! modelless quality gain of +9pp to +26pp across 5 noise cells
//! (see `.research/376` §8).
//!
//! ## Why dot-product, not set-attention
//!
//! The PoC found that cross-candidate set-attention adds ZERO modelless
//! value with identity projections (within ±1pp of the independent router
//! across all noise cells, v1 and v2). The paper's set-attention gains
//! (Figure 5) require trained Q/K/V projections → riir-train. This open
//! primitive ships the simplified dot-product router only; the
//! set-attention variant is a riir-train follow-up.
//!
//! ## Sigmoid, not softmax
//!
//! Per AGENTS.md §2: sampling uses **Logistic noise** (the sigmoid analog
//! of Gumbel noise for softmax). The Logistic(0, β) distribution has CDF
//! `sigmoid(x/β)`, so adding Logistic noise to each score and taking
//! argmax produces a sigmoid-family categorical sample without ever
//! invoking `exp` or normalizing via softmax. Temperature → 0 recovers
//! deterministic argmax; temperature → ∞ approaches uniform.
//!
//! ## Opt-in
//!
//! Gated by `local_branch_routing`. Promotion to `default` requires the
//! Plan 377 Phase 3 GOAT gate (G1 correctness, G2 router latency <1µs at
//! K=3 D=64, G3 K=1 bit-identical to standard decode, G4 alloc-free hot
//! path, G5 modelless, G6 sigmoid-not-softmax).
//!
//! Post-Candidate Branch Router — distilled from Local Branch Routing
//! (arXiv:2606.25354, Yin et al. June 2026). The modelless inference mechanism
//! distilled to its open primitive: forward K candidate next-tokens, score each
//! post-candidate hidden state by dot-product onto a frozen direction, commit
//! the argmax (or perturbed-argmax sample with Logistic noise — the sigmoid
//! analog of Gumbel-max).
//!
//! Generalizes the shipped ColliderPruner::batch_is_valid_with_hidden from
//! binary prune/keep to relative route-and-commit. PoC-confirmed modelless
//! quality gain of +9pp to +26pp across 5 noise cells (Plan 377 Phase 1,
//! riir-ai/crates/riir-poc). Set-attention variant adds zero modelless value
//! (PoC §8 — within ±1pp of the dot-product router across v1 and v2) and stays
//! a riir-train follow-up (needs trained Q/K/V projections).
//!
//! Sigmoid (NEVER softmax) per AGENTS.md §2: sampling uses Logistic(0, β)
//! noise whose CDF is sigmoid(x/β), making the categorical sample a
//! sigmoid-family operation without any exp/softmax normalization.
//!
//! DEFAULT-ON (Plan 377 Phase 3, 2026-07-04): GOAT PASS — G1 correctness ≥90%, G2 router
//! latency <1µs at K=3 D=64, G3 K=1 bit-identical to standard decode, G4
//! alloc-free hot path, G5 modelless, G6 sigmoid-not-softmax).

// ────────────────────────────────────────────────────────────────────────────
// PostCandidateRouter trait
// ────────────────────────────────────────────────────────────────────────────

/// Post-candidate branch router.
///
/// Given K forwarded candidate hidden states (and the parent hidden states
/// that produced them), returns the index of the candidate to commit.
///
/// Generalizes `ColliderPruner::batch_is_valid_with_hidden` (binary
/// prune/keep over `collider_preservation_score`) to relative
/// route-and-commit. Implementations:
///
/// - [`DotProductRouter`] — dot-product onto a frozen direction vector
///   (the proven modelless primitive; PoC-best at 53 ns / +9–26 pp quality
///   gain).
/// - [`ColliderRouterAdapter`] — wraps any [`PreservationScorer`] (e.g.
///   `ColliderConstraint` in katgpt-rs root) as a router. Competitive at
///   low noise, degrades at high noise per PoC §8.
///
/// # Contract
///
/// - `candidates_hidden` MUST be non-empty; an empty slice returns 0
///   (defensive — callers should supply ≥1 candidate).
/// - `parent_hidden` carries the conditioning states (depth history); the
///   dot-product router ignores it, the collider adapter forwards it to
///   the scorer.
/// - K=1 mode (single candidate) returns 0 bit-identically across all
///   implementations — this is the G3 GOAT gate (no regression vs.
///   standard decode).
pub trait PostCandidateRouter {
    /// Argmax route — deterministic, returns the best candidate index.
    ///
    /// Zero-allocation hot path.
    fn route_argmax(&self, parent_hidden: &[&[f32]], candidates_hidden: &[&[f32]]) -> usize;

    /// Perturbed-argmax sample — stochastic via Logistic noise (the
    /// sigmoid-family analog of Gumbel-max for softmax).
    ///
    /// `temperature` controls exploration:
    /// - `temperature → 0` recovers `route_argmax` (noise vanishes).
    /// - `temperature → ∞` approaches uniform sampling (noise dominates).
    /// - `temperature = 1` is the canonical Logistic(0,1) perturbation.
    ///
    /// Sigmoid (NOT softmax) per AGENTS.md §2 — the perturbation comes
    /// from the Logistic distribution whose CDF is sigmoid, so the
    /// sampling scheme is in the sigmoid family without ever calling
    /// `exp` or normalizing via softmax.
    fn route_sampled(
        &self,
        parent_hidden: &[&[f32]],
        candidates_hidden: &[&[f32]],
        temperature: f32,
        rng: &mut fastrand::Rng,
    ) -> usize;
}

// ────────────────────────────────────────────────────────────────────────────
// DotProductRouter
// ────────────────────────────────────────────────────────────────────────────

/// Default post-candidate router: dot-product onto a frozen direction vector.
///
/// The direction is the "good continuation" vector — typically a learned or
/// task-conditioned embedding of where the latent state should head. For
/// HLA / per-NPC routing, this is the NPC's committed personality direction;
/// for decoder branch routing, the model's continuation embedding.
///
/// Construction allocates once (the direction is copied into a `Box<[f32]>`);
/// `route_argmax` / `route_sampled` are zero-allocation.
///
/// # PoC validation (Plan 377 Phase 1)
///
/// IndependentRouter (this exact pattern) was the best modelless router
/// across all 5 noise cells: 79.6%–84.6% accuracy vs. 58.4%–70.6% for the
/// discrete CoT baseline (+9pp to +26pp). Latency 53 ns at D=16. See
/// `.research/376` §8 for the raw table.
pub struct DotProductRouter {
    /// Frozen scoring direction. Owned to keep the router self-contained
    /// (does not borrow caller state).
    direction: Box<[f32]>,
}

impl DotProductRouter {
    /// Construct from a direction slice. The direction is copied once into
    /// a `Box<[f32]>`; hot-path methods are zero-allocation.
    #[inline]
    pub fn new(direction: &[f32]) -> Self {
        Self {
            direction: direction.to_vec().into_boxed_slice(),
        }
    }

    /// Construct from an already-boxed direction (avoids the copy when the
    /// caller already owns the data).
    #[inline]
    pub fn from_boxed(direction: Box<[f32]>) -> Self {
        Self { direction }
    }

    /// Borrow the frozen direction.
    #[inline]
    pub fn direction(&self) -> &[f32] {
        &self.direction
    }

    /// Score a single candidate by dot-product with the direction. Truncates
    /// to the shorter of `candidate` / `self.direction` — callers should
    /// ensure matching dimensionality.
    #[inline]
    fn score(&self, candidate: &[f32]) -> f32 {
        let len = candidate.len().min(self.direction.len());
        crate::simd::simd_dot_f32(&candidate[..len], &self.direction[..len], len)
    }
}

impl PostCandidateRouter for DotProductRouter {
    #[inline]
    fn route_argmax(&self, _parent_hidden: &[&[f32]], candidates_hidden: &[&[f32]]) -> usize {
        if candidates_hidden.is_empty() {
            return 0;
        }
        let mut best_idx = 0;
        let mut best_score = self.score(candidates_hidden[0]);
        for (i, cand) in candidates_hidden.iter().enumerate().skip(1) {
            let s = self.score(cand);
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }
        best_idx
    }

    #[inline]
    fn route_sampled(
        &self,
        _parent_hidden: &[&[f32]],
        candidates_hidden: &[&[f32]],
        temperature: f32,
        rng: &mut fastrand::Rng,
    ) -> usize {
        if candidates_hidden.len() <= 1 {
            return 0;
        }
        perturbed_argmax(
            candidates_hidden.len(),
            |i| self.score(candidates_hidden[i]),
            temperature,
            rng,
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ColliderRouterAdapter (generic over PreservationScorer)
// ────────────────────────────────────────────────────────────────────────────

/// Decoupling trait: anything that scores a candidate's "preservation" of
/// some tracked property given parent hidden states and a decode depth.
///
/// The canonical implementor is `ColliderConstraint` in katgpt-rs root
/// (`collider_preservation_score(depth, parent_hidden, cand_hidden) -> f32`),
/// but katgpt-core cannot depend on the root crate, so the bound is generic.
/// Consumers impl this trait on their collider type to wire it into
/// [`ColliderRouterAdapter`].
///
/// Higher score = more preserving / more relevant. The adapter routes by
/// argmax over this score.
pub trait PreservationScorer {
    /// Score ∈ [0, 1] (higher = more preserving). Implementors should
    /// document their normalization; `ColliderConstraint` returns a
    /// sigmoid in [0, 1].
    fn preservation_score(
        &self,
        depth: usize,
        parent_hidden: &[&[f32]],
        cand_hidden: &[f32],
    ) -> f32;
}

/// Adapter: wraps any [`PreservationScorer`] as a [`PostCandidateRouter`].
///
/// Routes by argmax over `preservation_score`. Per PoC §8, this pattern is
/// competitive with `DotProductRouter` at low noise (within 1pp) but
/// degrades at high noise (the partial-correlation denominator is unstable
/// when candidate-parent correlation is high; ties the baseline at
/// σ_pre=σ_post=1.0). Ship as an alternative router for low-noise regimes,
/// not the default.
///
/// The `depth` is fixed at construction — typical usage constructs one
/// adapter per decode depth.
pub struct ColliderRouterAdapter<PS: PreservationScorer> {
    scorer: PS,
    depth: usize,
}

impl<PS: PreservationScorer> ColliderRouterAdapter<PS> {
    /// Construct with the underlying scorer and the fixed decode depth at
    /// which to evaluate it.
    #[inline]
    pub fn new(scorer: PS, depth: usize) -> Self {
        Self { scorer, depth }
    }

    /// Borrow the underlying scorer.
    #[inline]
    pub fn scorer(&self) -> &PS {
        &self.scorer
    }

    /// Decode depth at which this adapter scores candidates.
    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }
}

impl<PS: PreservationScorer> PostCandidateRouter for ColliderRouterAdapter<PS> {
    #[inline]
    fn route_argmax(&self, parent_hidden: &[&[f32]], candidates_hidden: &[&[f32]]) -> usize {
        if candidates_hidden.is_empty() {
            return 0;
        }
        let mut best_idx = 0;
        let mut best_score =
            self.scorer
                .preservation_score(self.depth, parent_hidden, candidates_hidden[0]);
        for (i, cand) in candidates_hidden.iter().enumerate().skip(1) {
            let s = self
                .scorer
                .preservation_score(self.depth, parent_hidden, cand);
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }
        best_idx
    }

    #[inline]
    fn route_sampled(
        &self,
        parent_hidden: &[&[f32]],
        candidates_hidden: &[&[f32]],
        temperature: f32,
        rng: &mut fastrand::Rng,
    ) -> usize {
        if candidates_hidden.len() <= 1 {
            return 0;
        }
        let depth = self.depth;
        let scorer = &self.scorer;
        perturbed_argmax(
            candidates_hidden.len(),
            |i| scorer.preservation_score(depth, parent_hidden, candidates_hidden[i]),
            temperature,
            rng,
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// perturbed_argmax — Logistic-noise sampler (the sigmoid analog of Gumbel-max)
// ────────────────────────────────────────────────────────────────────────────

/// Perturbed argmax with Logistic(0, temperature) noise.
///
/// For each candidate i, draw `g_i ~ Logistic(0, temperature)` via the
/// inverse-CDF method: `g = β · (ln(u) − ln(1−u))` where `u ~ Uniform(0,1)`.
/// The Logistic(0, β) distribution has CDF `sigmoid(x / β)`, so this is the
/// canonical sigmoid-family perturbation (the analog of adding Gumbel(0,1)
/// noise for softmax sampling).
///
/// Returns `argmax_i (score(i) + g_i)`.
///
/// # Temperature behavior
///
/// - `temperature → 0⁺`: noise vanishes, recovers plain argmax.
/// - `temperature = 1`: canonical Logistic(0,1) perturbation.
/// - `temperature → ∞`: noise dominates, approaches uniform.
///
/// # Zero-allocation
///
/// Stack-only — no heap allocation. Bounded K ≤ 4096 (the cap matches the
/// `MAX_UNEXPANDED` headroom in MCTSNode); larger slices are truncated.
#[inline]
fn perturbed_argmax(
    k: usize,
    mut score: impl FnMut(usize) -> f32,
    temperature: f32,
    rng: &mut fastrand::Rng,
) -> usize {
    debug_assert!(k > 0, "perturbed_argmax: k must be > 0");
    // Guard against non-finite temperature (NaN, Inf, negative) — fall back
    // to deterministic argmax. The clamp `max(1e-6)` prevents division by
    // zero and bounds the noise scale to a sane regime.
    let beta = if temperature.is_finite() && temperature > 0.0 {
        temperature
    } else {
        // Pathological temperature: fall back to argmax (β → 0).
        let mut best_idx = 0;
        let mut best = score(0);
        for i in 1..k {
            let s = score(i);
            if s > best {
                best = s;
                best_idx = i;
            }
        }
        return best_idx;
    };
    let mut best_idx = 0;
    let mut best_score = f32::NEG_INFINITY;
    for i in 0..k {
        let raw = score(i);
        let g = sample_logistic(beta, rng);
        let perturbed = raw + g;
        if perturbed > best_score {
            best_score = perturbed;
            best_idx = i;
        }
    }
    best_idx
}

/// Sample from Logistic(0, β) via the inverse-CDF method.
///
/// The Logistic(0, β) distribution has CDF `F(x) = sigmoid(x / β)`, so the
/// inverse CDF is `F⁻¹(u) = β · (ln(u) − ln(1−u))` for `u ∈ (0, 1)`. This
/// is the canonical "sigmoid distribution" sample.
///
/// Redraws `u` until it lands in the open interval (0, 1) — fastrand's
/// `f32()` can return 0.0 (which would yield `-inf` from `ln(0)`) or 1.0
/// (which would yield `+inf` from `ln(1−1)`). The redraw loop mirrors the
/// Gumbel pattern in `ac_prefix::types::gumbel_max_sample`.
#[inline]
fn sample_logistic(beta: f32, rng: &mut fastrand::Rng) -> f32 {
    let mut u = rng.f32();
    while u <= 0.0 || u >= 1.0 {
        u = rng.f32();
    }
    // g = β · logit(u) = β · (ln(u) − ln(1−u)). Both terms are real-valued
    // and finite for u ∈ (0, 1) after the redraw guard.
    beta * (u.ln() - (1.0 - u).ln())
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────────

    /// Empty parent slice (the dot-product router ignores parent).
    fn empty_parent() -> Vec<&'static [f32]> {
        Vec::new()
    }

    // ── DotProductRouter: route_argmax ──────────────────────────────────────

    #[test]
    fn dot_product_argmax_picks_highest_dot_product() {
        // direction = [1, 0]: only the first coordinate matters.
        let router = DotProductRouter::new(&[1.0, 0.0]);
        let cands: Vec<&[f32]> = vec![&[0.1, 0.9], &[0.9, 0.1], &[0.5, 0.5]];
        let parent = empty_parent();
        assert_eq!(router.route_argmax(&parent, &cands), 1);
    }

    #[test]
    fn dot_product_argmax_empty_candidates_returns_zero() {
        // Contract: defensive return on empty input.
        let router = DotProductRouter::new(&[1.0, 0.0]);
        let parent = empty_parent();
        assert_eq!(router.route_argmax(&parent, &[]), 0);
    }

    #[test]
    fn dot_product_argmax_k1_returns_zero() {
        // G3 gate: K=1 mode degenerates to standard decode — router has no
        // choice, returns 0 bit-identically.
        let router = DotProductRouter::new(&[1.0, 0.0]);
        let parent = empty_parent();
        let cands: Vec<&[f32]> = vec![&[0.42, 0.42]];
        assert_eq!(router.route_argmax(&parent, &cands), 0);
    }

    #[test]
    fn dot_product_argmax_handles_unequal_lengths() {
        // Candidate longer than direction: extra coords ignored (truncated).
        let router = DotProductRouter::new(&[1.0, 1.0]);
        let cands: Vec<&[f32]> = vec![&[0.5, 0.5, 99.0], &[0.1, 0.1, 99.0]];
        let parent = empty_parent();
        // dot(cand0, dir) = 0.5+0.5 = 1.0; dot(cand1, dir) = 0.2 → cand 0 wins.
        assert_eq!(router.route_argmax(&parent, &cands), 0);
    }

    #[test]
    fn dot_product_argmax_negative_scores_pick_least_negative() {
        // All-negative dot products: argmax picks the largest (least negative).
        let router = DotProductRouter::new(&[-1.0, -1.0]);
        let cands: Vec<&[f32]> = vec![&[0.9, 0.9], &[0.1, 0.1], &[0.5, 0.5]];
        let parent = empty_parent();
        // dot(cand0, dir) = -1.8; cand1 = -0.2 (largest); cand2 = -1.0.
        assert_eq!(router.route_argmax(&parent, &cands), 1);
    }

    #[test]
    fn dot_product_direction_accessor() {
        let router = DotProductRouter::new(&[1.0, 2.0, 3.0]);
        assert_eq!(router.direction(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn dot_product_from_boxed_avoids_copy() {
        let boxed: Box<[f32]> = vec![1.0, 0.0].into_boxed_slice();
        let router = DotProductRouter::from_boxed(boxed);
        assert_eq!(router.direction(), &[1.0, 0.0]);
    }

    // ── DotProductRouter: route_sampled ─────────────────────────────────────

    #[test]
    fn dot_product_sampled_low_temp_approximates_argmax() {
        // With very low temperature, Logistic noise is negligible; the
        // argmax winner should dominate over many trials.
        let router = DotProductRouter::new(&[1.0, 0.0]);
        let cands: Vec<&[f32]> = vec![&[0.1, 0.9], &[0.9, 0.1], &[0.5, 0.5]];
        let parent = empty_parent();
        let mut rng = fastrand::Rng::with_seed(42);
        let mut counts = [0_usize; 3];
        for _ in 0..1000 {
            let idx = router.route_sampled(&parent, &cands, 0.001, &mut rng);
            counts[idx] += 1;
        }
        // Argmax winner is idx=1 (dot=0.9); with near-zero noise it should
        // win ≥95% of the time.
        assert!(
            counts[1] >= 950,
            "low-temp counts = {counts:?}, expected idx=1 >= 950"
        );
    }

    #[test]
    fn dot_product_sampled_high_temp_approaches_uniform() {
        // With high temperature, Logistic noise dominates → near-uniform.
        let router = DotProductRouter::new(&[1.0, 0.0]);
        let cands: Vec<&[f32]> = vec![&[0.1, 0.9], &[0.9, 0.1], &[0.5, 0.5]];
        let parent = empty_parent();
        let mut rng = fastrand::Rng::with_seed(42);
        let mut counts = [0_usize; 3];
        for _ in 0..3000 {
            let idx = router.route_sampled(&parent, &cands, 100.0, &mut rng);
            counts[idx] += 1;
        }
        // Uniform expectation = 1000 each; allow ±25% deviation.
        for (i, &c) in counts.iter().enumerate() {
            assert!(
                c > 750 && c < 1250,
                "high-temp counts = {counts:?}: idx={i} count={c} outside [750, 1250]"
            );
        }
    }

    #[test]
    fn dot_product_sampled_k1_returns_zero() {
        // K=1 degenerate case.
        let router = DotProductRouter::new(&[1.0, 0.0]);
        let parent = empty_parent();
        let cands: Vec<&[f32]> = vec![&[0.42, 0.42]];
        let mut rng = fastrand::Rng::with_seed(0);
        assert_eq!(router.route_sampled(&parent, &cands, 1.0, &mut rng), 0);
    }

    #[test]
    fn dot_product_sampled_empty_returns_zero() {
        let router = DotProductRouter::new(&[1.0, 0.0]);
        let parent = empty_parent();
        let mut rng = fastrand::Rng::with_seed(0);
        assert_eq!(router.route_sampled(&parent, &[], 1.0, &mut rng), 0);
    }

    #[test]
    fn dot_product_sampled_is_deterministic_with_seed() {
        // Same seed → identical sample sequence (sync-boundary friendly).
        let router = DotProductRouter::new(&[1.0, 0.0]);
        let cands: Vec<&[f32]> = vec![&[0.1, 0.9], &[0.9, 0.1], &[0.5, 0.5]];
        let parent = empty_parent();
        let mut r1 = fastrand::Rng::with_seed(123);
        let mut r2 = fastrand::Rng::with_seed(123);
        for _ in 0..100 {
            let a = router.route_sampled(&parent, &cands, 0.5, &mut r1);
            let b = router.route_sampled(&parent, &cands, 0.5, &mut r2);
            assert_eq!(a, b, "sampled routes diverge with same seed");
        }
    }

    #[test]
    fn dot_product_sampled_nan_temperature_falls_back_to_argmax() {
        // Pathological temperature (NaN, negative, zero, infinity) must not
        // panic; falls back to deterministic argmax.
        let router = DotProductRouter::new(&[1.0, 0.0]);
        let cands: Vec<&[f32]> = vec![&[0.1, 0.9], &[0.9, 0.1], &[0.5, 0.5]];
        let parent = empty_parent();
        let mut rng = fastrand::Rng::with_seed(0);
        for &bad_temp in &[f32::NAN, -1.0, 0.0, f32::INFINITY] {
            let idx = router.route_sampled(&parent, &cands, bad_temp, &mut rng);
            assert_eq!(
                idx, 1,
                "NaN/negative temp should fall back to argmax (idx=1), got {idx}"
            );
        }
    }

    // ── ColliderRouterAdapter ──────────────────────────────────────────────

    /// Test scorer: returns the candidate's first coordinate as the score.
    struct FirstCoordScorer;
    impl PreservationScorer for FirstCoordScorer {
        #[inline]
        fn preservation_score(&self, _depth: usize, _parent: &[&[f32]], cand: &[f32]) -> f32 {
            cand.first().copied().unwrap_or(0.0)
        }
    }

    #[test]
    fn collider_adapter_argmax_routes_by_preservation_score() {
        let adapter = ColliderRouterAdapter::new(FirstCoordScorer, 0);
        let cands: Vec<&[f32]> = vec![&[0.1], &[0.9], &[0.5]];
        let parent = empty_parent();
        // Score = first coord → cand 1 wins (0.9).
        assert_eq!(adapter.route_argmax(&parent, &cands), 1);
    }

    #[test]
    fn collider_adapter_k1_returns_zero() {
        let adapter = ColliderRouterAdapter::new(FirstCoordScorer, 5);
        let parent = empty_parent();
        let cands: Vec<&[f32]> = vec![&[0.42]];
        assert_eq!(adapter.route_argmax(&parent, &cands), 0);
    }

    #[test]
    fn collider_adapter_empty_returns_zero() {
        let adapter = ColliderRouterAdapter::new(FirstCoordScorer, 0);
        let parent = empty_parent();
        assert_eq!(adapter.route_argmax(&parent, &[]), 0);
    }

    #[test]
    fn collider_adapter_forwards_depth_and_parent_to_scorer() {
        /// Scorer that records the (depth, parent_len) tuple it was called
        /// with, returning a constant so route_argmax always picks idx=0.
        struct RecordingScorer {
            depth_seen: std::sync::Mutex<Option<usize>>,
            parent_len_seen: std::sync::Mutex<Option<usize>>,
        }
        impl PreservationScorer for RecordingScorer {
            fn preservation_score(&self, depth: usize, parent: &[&[f32]], _cand: &[f32]) -> f32 {
                *self.depth_seen.lock().unwrap() = Some(depth);
                *self.parent_len_seen.lock().unwrap() = Some(parent.len());
                0.5
            }
        }
        let scorer = RecordingScorer {
            depth_seen: std::sync::Mutex::new(None),
            parent_len_seen: std::sync::Mutex::new(None),
        };
        let adapter = ColliderRouterAdapter::new(scorer, 7);
        let parent_state: Vec<f32> = vec![1.0, 2.0, 3.0];
        let parent_refs: Vec<&[f32]> = vec![&parent_state];
        let cands: Vec<&[f32]> = vec![&[0.5]];
        let _ = adapter.route_argmax(&parent_refs, &cands);
        assert_eq!(*adapter.scorer().depth_seen.lock().unwrap(), Some(7));
        assert_eq!(*adapter.scorer().parent_len_seen.lock().unwrap(), Some(1));
    }

    #[test]
    fn collider_adapter_sampled_low_temp_approximates_argmax() {
        let adapter = ColliderRouterAdapter::new(FirstCoordScorer, 0);
        let cands: Vec<&[f32]> = vec![&[0.1], &[0.9], &[0.5]];
        let parent = empty_parent();
        let mut rng = fastrand::Rng::with_seed(42);
        let mut counts = [0_usize; 3];
        for _ in 0..1000 {
            let idx = adapter.route_sampled(&parent, &cands, 0.001, &mut rng);
            counts[idx] += 1;
        }
        assert!(
            counts[1] >= 950,
            "collider low-temp counts = {counts:?}, expected idx=1 >= 950"
        );
    }

    #[test]
    fn collider_adapter_accessors() {
        let adapter = ColliderRouterAdapter::new(FirstCoordScorer, 3);
        assert_eq!(adapter.depth(), 3);
        // Type-name access on the underlying scorer is not directly testable
        // without Debug, but the borrow compiles.
        let _scorer: &FirstCoordScorer = adapter.scorer();
    }

    // ── Logistic sampling sanity ───────────────────────────────────────────

    #[test]
    fn sample_logistic_finite_for_valid_u() {
        // The redraw loop guarantees u ∈ (0, 1), so ln(u) and ln(1−u) are
        // finite. Confirm across many draws.
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..10_000 {
            let g = sample_logistic(1.0, &mut rng);
            assert!(g.is_finite(), "non-finite Logistic sample");
        }
    }

    #[test]
    fn sample_logistic_mean_near_zero() {
        // Logistic(0, 1) has mean 0. Empirical mean over many draws should
        // be close (±0.1 for 50k draws).
        let mut rng = fastrand::Rng::with_seed(11);
        let n = 50_000;
        let mut sum = 0.0_f64;
        for _ in 0..n {
            sum += sample_logistic(1.0, &mut rng) as f64;
        }
        let mean = sum / n as f64;
        assert!(
            mean.abs() < 0.1,
            "Logistic(0,1) empirical mean = {mean}, expected ≈ 0"
        );
    }

    // ── Trait-object dispatch ───────────────────────────────────────────────

    #[test]
    fn post_candidate_router_is_object_safe() {
        // The trait must be object-safe so consumers can hold
        // `Box<dyn PostCandidateRouter>` for runtime dispatch.
        let router: Box<dyn PostCandidateRouter> = Box::new(DotProductRouter::new(&[1.0, 0.0]));
        let cands: Vec<&[f32]> = vec![&[0.1, 0.9], &[0.9, 0.1]];
        let parent = empty_parent();
        assert_eq!(router.route_argmax(&parent, &cands), 1);
        let mut rng = fastrand::Rng::with_seed(0);
        let _ = router.route_sampled(&parent, &cands, 0.1, &mut rng);
    }
}
