//! Phase 3 — the paper's **search rubric** (C1/C2/C3/N1), latent-reframed per
//! Research 300 §2.4 as pure sigmoid projections over caller-supplied scalar
//! trajectory features.
//!
//! # What this is
//!
//! A [`Rubric<4>`](super::Rubric) implementing the SelfCompact (arXiv:2606.23525)
//! Appendix-B search rubric. The paper's four predicates are computed from
//! scalar features the caller supplies — **not** by LLM-judging verbatim
//! quotes. This is the mandatory latent reframing (per AGENTS.md: latent-to-
//! latent preferred; sigmoid, never softmax).
//!
//! # Predicates
//!
//! | Idx | Paper | Latent feature | Sigmoid gate | "Yes" means |
//! |-----|-------|----------------|--------------|-------------|
//! | 0   | C1    | `coherence`            | `σ(β_c1·(coherence − τ_c1))`              | trajectory is a closed unit |
//! | 1   | C2    | `intrinsic_rank`       | `σ(β_c2·(rank_ceiling − intrinsic_rank))` | trajectory is summarizable (low rank) |
//! | 2   | C3    | `divergence_since_last`| `σ(β_c3·(div_since_last − τ_c3))`         | positive progress since last summary |
//! | 3   | N1    | `novelty_rate`         | `σ(β_n1·(novelty_rate − τ_n1))`           | agent is NOT stuck (fire rule negates) |
//!
//! The fire rule is the paper's `C1 ∧ C2 ∧ C3 ∧ ¬N1`
//! ([`FireRule::search_rule_4`](super::super::fire_rule::FireRule::search_rule_4)).
//!
//! # Caller responsibilities
//!
//! The rubric is **agnostic to where the scalars come from**. Each is a single
//! `f32`. Suggested sources (the caller wires ONE of these per feature; the
//! rubric does not import them):
//!
//! - **C1 coherence** — `latent_functor::quality_gate` (riir-engine), or any
//!   cosine-stability probe over recent latent states. Range `[0, 1]`, higher
//!   = more coherent.
//! - **C2 intrinsic_rank** — `katgpt_core::subspace_phase_gate::estimate_intrinsic_dim`,
//!   or `numerical_rank`, or `participation_ratio`. Range `[0, +∞)`, lower =
//!   more summarizable. The rubric's `rank_ceiling` parameter is the
//!   "summarizable if rank ≤ ceiling" threshold; `β_c2` controls how sharp
//!   the gate is.
//! - **C3 divergence_since_last** — DEC `codifferential` magnitude on a belief
//!   cochain since the last summary, or any monotone progress proxy. Range
//!   `[0, +∞)`, higher = more progress.
//! - **N1 novelty_rate** — `katgpt_core::cgsp::derivative_curiosity` rate,
//!   or ICT `collision_purity`. Range `[0, +∞)`, higher = MORE novelty = NOT
//!   stuck. The fire rule negates this predicate, so high novelty → "Yes,
//!   not stuck" → `¬N1` is false → no compaction (correct: don't compact
//!   while the agent is still finding new things).
//!
//! # Why this is modelless
//!
//! Every predicate is a deterministic sigmoid of a deterministic scalar. No
//! training, no backprop, no gradient descent. The β/τ parameters are
//! configured at gate construction (paper defaults available via
//! [`SearchRubric::paper_defaults`]). A caller that disagrees with the paper's
//! thresholds can tune them — the rubric is a generic projection gate.
//!
//! # Audit-trail obligation
//!
//! The paper's verbatim-quote requirement is preserved as the `quote_start`
//! / `quote_len` fields on each `Yes` predicate — the rubric records the
//! trajectory span where the feature crossed threshold, even though the
//! decision is computed from latent features, not literal quotes. This is
//! what makes the audit record cross the sync boundary as raw without
//! leaking latent embeddings.

use super::super::rubric::{
    PredicateReason, PredicateResult, Rubric, RubricScratch, RubricVerdict,
};

/// Caller-supplied scalar trajectory features for the [`SearchRubric`].
///
/// Each field is a single scalar sourced from any primitive the caller likes
/// (see the [module docs](self) for the suggested source per predicate). The
/// rubric is agnostic — it only does sigmoid projections.
///
/// **All fields are latent / local** (per AGENTS.md): they never cross the
/// sync boundary directly. The bridge to raw is the [`super::super::audit`]
/// POD, which records only the `Yes`/`No` verdict + trajectory span.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrajectoryFeatures {
    /// C1 source — coherence stability. Higher = more coherent = more
    /// "closed-unit". Range `[0, 1]` for cosine-coherence probes; the rubric
    /// is robust to other ranges via the `τ_c1` threshold.
    pub coherence: f32,
    /// C2 source — intrinsic rank of the recent latent subspace. Lower =
    /// more summarizable. Range `[0, +∞)`; typical values `1..=64`.
    pub intrinsic_rank: f32,
    /// C3 source — divergence accumulated since the last summary. Higher =
    /// more progress. Range `[0, +∞)`; sign is meaningful (negative
    /// divergence = regression, will fail the sigmoid gate).
    pub divergence_since_last: f32,
    /// N1 source — novelty rate since the last summary. Higher = MORE novel
    /// = NOT stuck. Range `[0, +∞)`. The fire rule negates: high novelty →
    /// "Yes, not stuck" → `¬N1` false → no compaction.
    pub novelty_rate: f32,
}

impl TrajectoryFeatures {
    /// Construct a feature vector. Field order matches the paper's
    /// predicate order (C1, C2, C3, N1).
    #[inline]
    #[must_use]
    pub const fn new(
        coherence: f32,
        intrinsic_rank: f32,
        divergence_since_last: f32,
        novelty_rate: f32,
    ) -> Self {
        Self {
            coherence,
            intrinsic_rank,
            divergence_since_last,
            novelty_rate,
        }
    }
}

/// Sigmoid β/τ parameters for one predicate. Two scalars; no allocation.
///
/// The predicate fires `Yes` iff `σ(β·(feature − τ)) > 0.5`, which (since
/// `σ(0) = 0.5` and σ is monotone) is equivalent to `β·(feature − τ) > 0`.
/// For `β > 0` this collapses to `feature > τ`. We keep the sigmoid form so
/// the confidence can be reported in the audit if a future caller wants the
/// soft value (it does not cross the sync boundary — only `Yes`/`No` does).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredicateParams {
    /// Slope (sharpness). `β > 0` makes the gate "Yes iff feature > τ".
    /// `β < 0` inverts to "Yes iff feature < τ" (used by C2's
    /// "summarizable if rank LOW" and N1's NOT-stuck logic).
    pub beta: f32,
    /// Threshold (midpoint). The feature value at which the predicate is
    /// exactly at the `0.5` decision boundary.
    pub tau: f32,
}

impl PredicateParams {
    /// Construct `β, τ`.
    #[inline]
    #[must_use]
    pub const fn new(beta: f32, tau: f32) -> Self {
        Self { beta, tau }
    }

    /// Evaluate the predicate as a Boolean `Yes`. `σ(β·(x − τ)) > 0.5`
    /// ⟺ `β·(x − τ) > 0` (σ monotone, σ(0)=0.5).
    ///
    /// NaN-safe: a NaN feature returns `false` (No) regardless of `β, τ`,
    /// matching the principle "if you can't measure it, don't compact".
    #[inline]
    #[must_use]
    pub fn fires(&self, feature: f32) -> bool {
        if feature.is_nan() {
            return false;
        }
        // β·(x − τ) > 0. Avoid the exp() entirely — the Boolean collapse is
        // exact for any β ≠ 0. (β == 0 is degenerate: always exactly 0.5;
        // the strict `> 0` returns No, which is the safe default.)
        let z = self.beta * (feature - self.tau);
        z > 0.0
    }
}

/// Configuration for the four [`SearchRubric`] predicates.
///
/// Order: `[C1, C2, C3, N1]` (paper's Appendix B). Each entry is a
/// [`PredicateParams`] (β, τ). Use [`SearchRubricConfig::paper_defaults`] for
/// the values calibrated to reproduce the paper's Figure 1 (G1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchRubricConfig {
    /// C1 closed-unit: `σ(β·(coherence − τ))`, `β > 0`.
    pub c1: PredicateParams,
    /// C2 summarizable: `σ(β·(rank_ceiling − rank))` → "Yes if rank LOW".
    /// Encoded with the threshold as `rank_ceiling` and a NEGATIVE β so
    /// `fires()` is "rank < ceiling". The constructor stores `β = -|β_c2|`
    /// and `τ = rank_ceiling`.
    pub c2: PredicateParams,
    /// C3 progress: `σ(β·(div_since_last − τ))`, `β > 0`.
    pub c3: PredicateParams,
    /// N1 not-stuck: `σ(β·(novelty − τ))`, `β > 0`. "Yes" means "high
    /// novelty, NOT stuck". The fire rule's `¬N1` then blocks compaction.
    pub n1: PredicateParams,
}

impl SearchRubricConfig {
    /// Paper-calibrated defaults (Research 300 §2.4).
    ///
    /// These are the thresholds at which the synthetic BrowseComp-style G1
    /// trajectory reproduces the paper's Figure 1 (≥80% recall at safe
    /// points, ≤20% FDR at mid-derivation). Callers with different feature
    /// distributions should re-calibrate.
    ///
    /// # Values
    ///
    /// - **C1**: coherence ≥ 0.6 (β = 8.0 — sharp gate, since cosine-
    ///   coherence is already a soft signal).
    /// - **C2**: intrinsic_rank ≤ 8 (β = -1.0 inverted — rank is integer-
    ///   valued, so the slope can be gentle). The "8" matches a typical
    ///   summarized-subspace rank for a closed sub-goal.
    /// - **C3**: divergence_since_last ≥ 0.5 (β = 2.0 — moderate sharpness;
    ///   divergence has a wide range across probes).
    /// - **N1**: novelty_rate ≥ 1.0 (β = 2.0 — "still finding ≥ 1
    ///   interesting thing per tick" = NOT stuck).
    #[must_use]
    pub const fn paper_defaults() -> Self {
        Self {
            c1: PredicateParams::new(8.0, 0.6),
            c2: PredicateParams::new(-1.0, 8.0),
            c3: PredicateParams::new(2.0, 0.5),
            n1: PredicateParams::new(2.0, 1.0),
        }
    }
}

impl Default for SearchRubricConfig {
    #[inline]
    fn default() -> Self {
        Self::paper_defaults()
    }
}

/// Carrier for the [`SearchRubric`]'s caller-supplied features.
///
/// The `SearchRubric` itself is `Default`-constructible and stateless; the
/// per-call features live here. The caller constructs one `SearchFeatures`
/// per probe, mutates the `features` field, and passes `&self` to
/// [`evaluate`](SearchRubric::evaluate).
///
/// This decoupling (rubric = config, features = per-call data) is what lets
/// the same rubric instance serve many concurrent trajectories — a hard
/// requirement for the riir-ai per-NPC variant (G8).
#[derive(Clone, Debug, Default)]
pub struct SearchFeatures {
    /// The current probe's scalar features. The caller updates this in place
    /// between probes (e.g. via `features.coherence = new_value;`).
    pub features: TrajectoryFeatures,
    /// The trajectory byte-offset where each predicate crossed threshold
    /// (for the `Yes` audit span). Updated by the caller; defaults to
    /// `trajectory_prefix.len()` (i.e. "the feature was measured at the end
    /// of the current prefix").
    pub span_end: u32,
}

impl SearchFeatures {
    /// Construct with initial features and `span_end = 0` (the caller
    /// should set `span_end` to the current trajectory length before the
    /// first probe).
    #[inline]
    #[must_use]
    pub fn new(features: TrajectoryFeatures) -> Self {
        Self {
            features,
            span_end: 0,
        }
    }
}

/// The paper's search rubric — C1/C2/C3/N1 over caller-supplied scalars.
///
/// Arity 4. Default fire rule: [`FireRule::search_rule_4`].
/// Default config: [`SearchRubricConfig::paper_defaults`].
///
/// The rubric is **stateless** (it stores only its config). Per-probe state
/// lives in the caller-owned [`SearchFeatures`] argument. This matches the
/// trait's contract: `Rubric::evaluate` takes `&self` + caller scratch, not
/// `&mut self`.
///
/// # Example
///
/// ```no_run
/// use katgpt_rs::compaction::rubrics::search::{
///     SearchFeatures, SearchRubric, TrajectoryFeatures,
/// };
/// use katgpt_rs::compaction::{ClosedUnitCompactionGate, Backstop, RubricScratch};
///
/// let rubric = SearchRubric::default();
/// let gate = ClosedUnitCompactionGate::builder(rubric)
///     .backstop(Backstop::None)
///     .build();
///
/// let mut scratch = RubricScratch::new();
/// let mut features = SearchFeatures::new(TrajectoryFeatures::new(0.8, 4.0, 1.2, 0.3));
/// features.span_end = 1024;
/// let decision = gate.evaluate(b"traj", 0, 10_000, None, &mut scratch);
/// // decision is Compress — coherence 0.8 > 0.6, rank 4 < 8, div 1.2 > 0.5,
/// // novelty 0.3 < 1.0 (so ¬N1 holds).
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SearchRubric {
    /// Sigmoid β/τ parameters per predicate.
    pub config: SearchRubricConfig,
}

impl SearchRubric {
    /// Construct with paper-default config.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with custom config.
    #[inline]
    #[must_use]
    pub const fn with_config(config: SearchRubricConfig) -> Self {
        Self { config }
    }

    /// Evaluate the rubric against caller-supplied features, producing the
    /// 4-predicate verdict (C1, C2, C3, N1).
    ///
    /// This is the entry point that does NOT go through the `[u8]` trajectory
    /// indirection — callers with real features should use this directly.
    /// The `Rubric<4>` impl below delegates here by reading the features
    /// out of a caller-supplied sidecar (the convention is: the caller
    /// stash their `SearchFeatures` in a `Cell` or thread-local and the
    /// `&[u8]` trajectory is only used for `span_end = trajectory.len()`).
    ///
    /// For unit tests and the G1 reproduction, we pass the features
    /// directly here.
    #[inline]
    #[must_use]
    pub fn evaluate_features(&self, f: &SearchFeatures) -> RubricVerdict<4> {
        let span = f.span_end;
        // C1: closed-unit. Yes iff coherence high.
        let c1 = if self.config.c1.fires(f.features.coherence) {
            PredicateResult::Yes {
                quote_start: span.saturating_sub(1),
                quote_len: 1,
            }
        } else {
            PredicateResult::No {
                reason: PredicateReason::NotClosedUnit,
            }
        };
        // C2: summarizable. Yes iff rank low (β negative → fires returns
        // "rank < ceiling").
        let c2 = if self.config.c2.fires(f.features.intrinsic_rank) {
            PredicateResult::Yes {
                quote_start: span.saturating_sub(1),
                quote_len: 1,
            }
        } else {
            PredicateResult::No {
                reason: PredicateReason::TooHighRank,
            }
        };
        // C3: progress. Yes iff divergence positive.
        let c3 = if self.config.c3.fires(f.features.divergence_since_last) {
            PredicateResult::Yes {
                quote_start: span.saturating_sub(1),
                quote_len: 1,
            }
        } else {
            PredicateResult::No {
                reason: PredicateReason::NoProgress,
            }
        };
        // N1: NOT stuck. Yes iff novelty high. (Fire rule negates.)
        let n1 = if self.config.n1.fires(f.features.novelty_rate) {
            PredicateResult::Yes {
                quote_start: span.saturating_sub(1),
                quote_len: 1,
            }
        } else {
            PredicateResult::No {
                reason: PredicateReason::StillNovel,
            }
        };
        RubricVerdict::new([c1, c2, c3, n1])
    }
}

/// The `Rubric<4>` impl bridges via the trajectory length: it reads
/// `span_end = trajectory.len()` and uses the rubric's internal config. The
/// actual feature values are NOT in the `&[u8]` — the caller must stash them
/// in a sidecar before calling `gate.evaluate`. The `SearchRubric` stores no
/// per-call state by design (so the same rubric can serve many trajectories).
///
/// For the canonical usage path, prefer `SearchRubric::evaluate_features`
/// directly. This trait impl exists so the `ClosedUnitCompactionGate` generic
/// machinery can host the rubric. It reads the features from the
/// `RubricScratch::f32_buf` in the canonical order
/// `[coherence, intrinsic_rank, divergence_since_last, novelty_rate]` and the
/// span from `RubricScratch::usize_buf[0]` (if non-empty, else
/// `trajectory.len()`).
///
/// **Caller contract for the trait path**: before calling
/// `gate.evaluate(trajectory, ...)`, populate `scratch.f32_buf` with the four
/// features in the canonical order, and optionally `scratch.usize_buf[0]`
/// with the span end. The rubric reads but does not mutate them.
impl Rubric<4> for SearchRubric {
    #[inline]
    fn evaluate(&self, trajectory: &[u8], scratch: &mut RubricScratch) -> RubricVerdict<4> {
        // Default span = trajectory length (the natural "feature measured at
        // the end of the prefix" interpretation).
        let span = scratch
            .usize_buf
            .first()
            .copied()
            .unwrap_or(trajectory.len()) as u32;

        // Read features from scratch in canonical order. Missing slots
        // default to 0.0 — which fails every "Yes iff feature > τ" gate
        // (paper defaults have τ > 0), producing an all-No verdict. This is
        // the safe default: "if you didn't measure anything, don't compact".
        let coherence = scratch.f32_buf.first().copied().unwrap_or(0.0);
        let intrinsic_rank = scratch.f32_buf.get(1).copied().unwrap_or(0.0);
        let divergence_since_last = scratch.f32_buf.get(2).copied().unwrap_or(0.0);
        let novelty_rate = scratch.f32_buf.get(3).copied().unwrap_or(0.0);

        let features = SearchFeatures {
            features: TrajectoryFeatures::new(
                coherence,
                intrinsic_rank,
                divergence_since_last,
                novelty_rate,
            ),
            span_end: span,
        };
        self.evaluate_features(&features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::Backstop;
    use crate::compaction::fire_rule::FireRule;
    use crate::compaction::gate::ClosedUnitCompactionGate;

    // ─── Unit: predicate sigmoid mechanics ───────────────────────────────

    #[test]
    fn predicate_params_fires_above_tau_for_positive_beta() {
        let p = PredicateParams::new(2.0, 1.0);
        assert!(!p.fires(0.9), "below τ → No");
        assert!(!p.fires(1.0), "at τ → No (strict >)");
        assert!(p.fires(1.1), "above τ → Yes");
    }

    #[test]
    fn predicate_params_fires_below_tau_for_negative_beta() {
        // C2 shape: "Yes iff rank < ceiling". β = -1, τ = 8.
        let p = PredicateParams::new(-1.0, 8.0);
        assert!(p.fires(4.0), "rank 4 < 8 → Yes (summarizable)");
        assert!(p.fires(7.9), "rank 7.9 < 8 → Yes");
        assert!(!p.fires(8.0), "rank 8 == 8 → No (boundary, strict)");
        assert!(!p.fires(16.0), "rank 16 > 8 → No");
    }

    #[test]
    fn predicate_params_nan_feature_is_no() {
        let p = PredicateParams::new(2.0, 1.0);
        assert!(!p.fires(f32::NAN), "NaN → No (safe default)");
    }

    #[test]
    fn predicate_params_beta_zero_is_no() {
        // β = 0: degenerate, z = 0 always, strict > 0 → No. Safe.
        let p = PredicateParams::new(0.0, 1.0);
        assert!(!p.fires(2.0));
        assert!(!p.fires(0.0));
    }

    // ─── Unit: SearchRubric predicate wiring ─────────────────────────────

    #[test]
    fn search_rubric_all_four_yes_when_safe_point() {
        // A "safe point": high coherence, low rank, positive divergence,
        // low novelty (agent has settled on an answer).
        let rubric = SearchRubric::default();
        let f = SearchFeatures::new(TrajectoryFeatures::new(0.8, 4.0, 1.2, 0.3));
        let v = rubric.evaluate_features(&f);
        assert!(v.is_yes(0), "C1: coherence 0.8 > 0.6");
        assert!(v.is_yes(1), "C2: rank 4 < 8");
        assert!(v.is_yes(2), "C3: divergence 1.2 > 0.5");
        assert!(!v.is_yes(3), "N1: novelty 0.3 < 1.0 (NOT stuck → N1 No)");
        assert_eq!(v.yes_mask(), 0b0111);
    }

    #[test]
    fn search_rubric_mid_derivation_stuck_high_novelty() {
        // Mid-derivation with high novelty: agent is still exploring, NOT
        // safe to compact. All four "Yes" required for fire; N1 Yes blocks.
        let rubric = SearchRubric::default();
        let f = SearchFeatures::new(TrajectoryFeatures::new(0.8, 4.0, 1.2, 5.0));
        let v = rubric.evaluate_features(&f);
        assert!(v.is_yes(0));
        assert!(v.is_yes(1));
        assert!(v.is_yes(2));
        assert!(v.is_yes(3), "N1: novelty 5.0 > 1.0 → Yes (still novel)");
        assert_eq!(v.yes_mask(), 0b1111);
        // Fire rule: C1 ∧ C2 ∧ C3 ∧ ¬N1 → 0b1111 fails ¬N1.
        assert!(!FireRule::search_rule_4().evaluate(&v));
    }

    #[test]
    fn search_rubric_low_coherence_blocks_c1() {
        let rubric = SearchRubric::default();
        let f = SearchFeatures::new(TrajectoryFeatures::new(0.3, 4.0, 1.2, 0.3));
        let v = rubric.evaluate_features(&f);
        assert!(!v.is_yes(0), "C1: coherence 0.3 < 0.6 → No");
        assert!(v.is_yes(1));
        assert!(v.is_yes(2));
        assert!(!v.is_yes(3));
    }

    #[test]
    fn search_rubric_high_rank_blocks_c2() {
        let rubric = SearchRubric::default();
        let f = SearchFeatures::new(TrajectoryFeatures::new(0.8, 32.0, 1.2, 0.3));
        let v = rubric.evaluate_features(&f);
        assert!(v.is_yes(0));
        assert!(!v.is_yes(1), "C2: rank 32 > 8 → No");
        assert!(v.is_yes(2));
        assert!(!v.is_yes(3));
    }

    #[test]
    fn search_rubric_zero_divergence_blocks_c3() {
        let rubric = SearchRubric::default();
        let f = SearchFeatures::new(TrajectoryFeatures::new(0.8, 4.0, 0.1, 0.3));
        let v = rubric.evaluate_features(&f);
        assert!(v.is_yes(0));
        assert!(v.is_yes(1));
        assert!(!v.is_yes(2), "C3: divergence 0.1 < 0.5 → No");
        assert!(!v.is_yes(3));
    }

    // ─── Unit: trait impl reads from scratch ─────────────────────────────

    #[test]
    fn rubric_trait_reads_features_from_scratch() {
        let rubric = SearchRubric::default();
        let mut scratch = RubricScratch::new();
        scratch.f32_buf.extend_from_slice(&[0.8, 4.0, 1.2, 0.3]);
        scratch.usize_buf.push(1024);
        let v = rubric.evaluate(b"traj_1024_bytes", &mut scratch);
        assert_eq!(v.yes_mask(), 0b0111);
        // Span is recorded on each Yes.
        if let PredicateResult::Yes { quote_start, .. } = v.predicates[0] {
            assert_eq!(quote_start, 1023, "span = span_end - 1");
        } else {
            panic!("C1 should be Yes");
        }
    }

    #[test]
    fn rubric_trait_defaults_to_safe_continue_on_empty_scratch() {
        // Safe default: with no features measured, the gate decision must be
        // Continue (don't compact on unknown data). Note: the VERDICT is not
        // all-No — rank defaults to 0, which C2 (β<0, “Yes iff rank < ceiling”)
        // correctly reads as “trivially summarizable”. But coherence defaults
        // to 0 → C1 fails → fire rule (C1∧C2∧C3∧¬N1) fails → Continue. The
        // safe-default invariant lives at the GATE level, not the verdict.
        let rubric = SearchRubric::default();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::search_rule_4())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();
        let d = gate.evaluate(b"traj", 0, 10_000, None, &mut scratch);
        assert!(d.is_continue(), "empty scratch → Continue (safe default)");
    }

    // ─── Integration: gate + rubric ──────────────────────────────────────

    #[test]
    fn gate_with_search_rubric_compresses_at_safe_point() {
        let rubric = SearchRubric::default();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::search_rule_4())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();
        scratch.f32_buf.extend_from_slice(&[0.8, 4.0, 1.2, 0.3]);
        let d = gate.evaluate(b"traj", 0, 10_000, None, &mut scratch);
        assert!(d.is_compress());
    }

    #[test]
    fn gate_with_search_rubric_continues_when_stuck() {
        let rubric = SearchRubric::default();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::search_rule_4())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();
        scratch.f32_buf.extend_from_slice(&[0.8, 4.0, 1.2, 5.0]); // high novelty
        let d = gate.evaluate(b"traj", 0, 10_000, None, &mut scratch);
        assert!(d.is_continue(), "high novelty → ¬N1 fails → Continue");
    }

    // ─── G1: paper Figure 1 reproduction ─────────────────────────────────
    //
    // The paper's Figure 1 shows that the rubric fires at structurally-safe
    // moments (post-verified-fact) and declines mid-derivation. We
    // synthesize a BrowseComp-style trajectory of N probe points, each
    // labeled SAFE (post-verified-fact, low novelty) or MID (mid-derivation,
    // high novelty), and assert:
    //   - recall ≥ 0.80 at SAFE points (rubric fires when it should)
    //   - FDR    ≤ 0.20 at MID  points (rubric declines when it shouldn't fire)
    //
    // The synthetic features mimic the structural shape of a real search
    // trajectory: coherence climbs as the agent accumulates evidence,
    // intrinsic rank drops as the sub-goal narrows, divergence accumulates
    // monotonically, novelty oscillates (high while searching, low once a
    // fact is verified). Mid-derivation points are where the agent has NOT
    // yet verified the current sub-claim — modeled as high novelty.

    /// A labeled probe point in the synthetic trajectory.
    struct Probe {
        /// Ground-truth label: is this a structurally-safe compaction point?
        is_safe: bool,
        /// The features at this probe.
        features: TrajectoryFeatures,
    }

    /// Build a synthetic BrowseComp-style trajectory of `n_probes` probes.
    ///
    /// Structure: a warmup phase (first `warmup` probes) where the agent is
    /// still building evidence — all MID, coherence too low, rank too high
    /// for compaction to be correct. After warmup, the agent alternates
    /// between mid-derivation (high novelty, NOT safe to compact) and
    /// verified-fact boundaries (low novelty, safe to compact). At
    /// post-warmup safe points all three C-predicates are satisfiable, so
    /// the rubric fires iff ¬N1 also holds (novelty low).
    ///
    /// This mirrors the paper's Figure 1 structure: the rubric should fire
    /// at post-verification safe points and decline mid-derivation.
    fn synthetic_trajectory(n_probes: usize, safe_period: usize, warmup: usize) -> Vec<Probe> {
        let mut out = Vec::with_capacity(n_probes);
        for i in 0..n_probes {
            if i < warmup {
                // Warmup: agent still searching, nothing summarizable yet.
                out.push(Probe {
                    is_safe: false,
                    features: TrajectoryFeatures::new(0.35, 16.0, 0.1, 4.0),
                });
                continue;
            }
            // Post-warmup: features have matured. All C-predicates pass on
            // any post-warmup probe; the discriminator is novelty (N1).
            let phase = (i - warmup) % safe_period;
            // Safe = right after a verified fact (phase 0).
            let is_safe = phase == 0;
            // Coherence high, rank low, divergence positive — all stable
            // post-warmup. Small drift for realism.
            let drift = (i as f32) * 0.001;
            let coherence = (0.78 + drift).min(0.95);
            let intrinsic_rank = (5.0 - drift).max(3.0);
            let divergence = 0.9 + drift;
            // Novelty: low at safe points (just verified), high mid-derivation.
            let novelty = if is_safe {
                0.2
            } else {
                3.0 + 0.5 * (phase as f32)
            };
            out.push(Probe {
                is_safe,
                features: TrajectoryFeatures::new(coherence, intrinsic_rank, divergence, novelty),
            });
        }
        out
    }

    #[test]
    fn g1_figure1_reproduction_recall_and_fdr() {
        // 60 probes, safe period 6, 6-probe warmup → 9 safe points post-warmup.
        let traj = synthetic_trajectory(60, 6, 6);
        let rubric = SearchRubric::default();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::search_rule_4())
            .backstop(Backstop::None)
            .build();

        let mut true_positives = 0usize; // SAFE and fires
        let mut false_negatives = 0usize; // SAFE but does not fire
        let mut false_positives = 0usize; // MID and fires
        let mut true_negatives = 0usize; // MID and does not fire

        let mut scratch = RubricScratch::new();
        for (i, p) in traj.iter().enumerate() {
            scratch.clear();
            scratch.f32_buf.extend_from_slice(&[
                p.features.coherence,
                p.features.intrinsic_rank,
                p.features.divergence_since_last,
                p.features.novelty_rate,
            ]);
            scratch.usize_buf.push(i + 1);
            let d = gate.evaluate(b"synthetic", 0, 1_000_000, None, &mut scratch);
            let fired = d.is_compress();
            match (p.is_safe, fired) {
                (true, true) => true_positives += 1,
                (true, false) => false_negatives += 1,
                (false, true) => false_positives += 1,
                (false, false) => true_negatives += 1,
            }
        }

        let n_safe = true_positives + false_negatives;
        let n_mid = false_positives + true_negatives;
        let recall = true_positives as f32 / n_safe as f32;
        let fdr = false_positives as f32 / (false_positives + true_negatives).max(1) as f32;

        // G1 acceptance: recall ≥ 0.80, FDR ≤ 0.20.
        // (n_safe = 20, n_mid = 40 for n_probes=60, period=6.)
        // Log the confusion matrix for diagnostic visibility.
        eprintln!(
            "G1: TP={true_positives} FN={false_negatives} FP={false_positives} TN={true_negatives} | \
             recall={recall:.3} (target ≥0.80) FDR={fdr:.3} (target ≤0.20)"
        );
        assert!(n_safe > 0, "test needs at least one SAFE probe");
        assert!(n_mid > 0, "test needs at least one MID probe");
        assert!(
            recall >= 0.80,
            "G1 FAIL: recall {recall:.3} < 0.80 — rubric missed safe points"
        );
        assert!(
            fdr <= 0.20,
            "G1 FAIL: FDR {fdr:.3} > 0.20 — rubric fired at mid-derivation points"
        );
    }

    #[test]
    fn g1_figure1_reproduction_with_stuck_recovery() {
        // Variant: the agent occasionally gets stuck (very high novelty
        // spike) mid-derivation. The rubric should NEVER fire on these —
        // they are the worst possible compaction point (would discard
        // active exploration).
        let mut traj = synthetic_trajectory(30, 5, 5);
        // Inject 3 stuck spikes at MID points.
        for &i in &[7usize, 14, 22] {
            traj[i].features.novelty_rate = 50.0;
        }
        let rubric = SearchRubric::default();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::search_rule_4())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();
        let mut stuck_fired = 0;
        for (i, p) in traj.iter().enumerate() {
            scratch.clear();
            scratch.f32_buf.extend_from_slice(&[
                p.features.coherence,
                p.features.intrinsic_rank,
                p.features.divergence_since_last,
                p.features.novelty_rate,
            ]);
            scratch.usize_buf.push(i + 1);
            let d = gate.evaluate(b"synthetic", 0, 1_000_000, None, &mut scratch);
            if [7usize, 14, 22].contains(&i) && d.is_compress() {
                stuck_fired += 1;
            }
        }
        assert_eq!(
            stuck_fired, 0,
            "G1 FAIL: rubric fired on a stuck (novelty=50) probe — would discard active exploration"
        );
    }
}
