//! Similarity Inference — modelless endogenous correlation device for embedded equilibrium.
//!
//! Maintains a **similarity posterior** `ω ∈ (0, 1)` between a focal decision-maker
//! and one partner, updated incrementally from a stream of joint-action observations,
//! plus a **cooperation gate** ([`embedded_best_response`]) that switches from
//! competitive-best-response (Nash) to cooperative-best-response (CCE) when `ω`
//! crosses a payoff-derived threshold. The primitive composes a Bayesian posterior
//! update + a sigmoid cooperation gate + a best-response comparator — zero game
//! semantics, zero entity-kind assumptions, pure closed-form math.
//!
//! # What this is
//!
//! A **modelless inference-time mechanism** distilled from Meulemans et al.
//! (arXiv:2608.03958 §H + §I, *Paradigms of Intelligence*, 4 Aug 2026). The
//! focal agent maintains `ω_T` — its posterior belief that the partner shares
//! its functional identity (same policy / same shard / same committed
//! personality). Observing one's own action played by the partner is a
//! "miracle" under the independent-policy hypothesis but expected under the
//! shared-identity hypothesis, so each such observation pushes `ω` toward 1.
//! When `ω` crosses a payoff-derived threshold, the focal switches from
//! competitive-best-response (defect) to cooperative-best-response (cooperate).
//!
//! # What this is NOT
//!
//! See `katgpt-rs/.plans/526_similarity_inference_primitive.md` and
//! `.research/471` §3 for the full novelty-gate analysis. Summary:
//!
//! - **NOT a new equilibrium concept.** "Embedded equilibrium" reduces to Nash
//!   (decoupled case) or correlated equilibrium (coupled case); both are
//!   already shipped (`CceLp<N,A>`, Plan 295, DEFAULT-ON). The equilibrium
//!   reached by this primitive IS a CCE.
//! - **NOT a replacement for the LP-CCE Moderator (R143/R274).** The moderator
//!   uses an *exogenous* designer-set correlation device `ζ`; this primitive
//!   infers an *endogenous* correlation device `ω` from interaction history.
//!   The two compose: when crowd `ω` crosses threshold, the moderator's
//!   `Γ₀` can switch endogenously. See `.research/471` §3.5.
//! - **NOT a sync-boundary primitive.** `ω` is a per-focal latent scalar that
//!   stays local; only the final cooperate/defect action crosses the sync
//!   boundary as a raw u8 (per AGENTS.md §"Sync Boundary Rule").
//! - **NOT applicable to physical-domain state.** Position, HP, wallet
//!   balance stay raw + synced. `ω` is a *semantic/social* belief about
//!   partner identity, in the latent domain.
//!
//! # Closed-form posterior (paper §H.2)
//!
//! Under a shared-shard hypothesis class with prior `α = P(shared)`:
//!
//! ```text
//! ω_T = α / (α + (1−α) · W(æ_<T))
//! ```
//!
//! where `W(æ_<T) = Π_t P(a_i_t, a_j_t | situation_t)` is the joint-action
//! likelihood under the *independent-policy* marginal. For the canonical
//! "matched-action" evidence (the partner played the same discrete action as
//! the focal in the same situation), each match contributes a factor `1/|A|`,
//! giving `W = |A|^(−T)` and thus:
//!
//! ```text
//! ω_T = α / (α + (1−α) · |A|^(−T))
//! ```
//!
//! For the canonical 2-action Prisoner's Dilemma (`|A|=2`): `ω_T = α / (α +
//! (1−α) · 2^(−T))`, which is the analytical form verified in the G1 test
//! ([`posterior::tests`] `g1_matches_analytical_omega`).
//!
//! `SimilarityPosterior` accumulates `log W` incrementally (no replay of full
//! history) — O(1) per observation regardless of `T`.
//!
//! # Embedded best-response (paper §H.3)
//!
//! Given `ω`, the partner's predicted action distribution is a mixture
//! conditioned on the focal's contemplated action:
//!
//! - If similar (prob `ω`): the partner mirrors the focal's contemplated action.
//! - If independent (prob `1−ω`): the partner plays the exogenous marginal
//!   `q` (default uniform).
//!
//! ```text
//! P̂(a_partner = a' | a_self = a) = ω · δ(a', a) + (1−ω) · q(a')
//! ```
//!
//! The focal picks the action maximizing expected payoff under this coupled
//! predictive model:
//!
//! ```text
//! a* = argmax_a  Σ_{a'}  P̂(a_partner = a' | a_self = a) · R(a, a')
//! ```
//!
//! For canonical PD (R=2, S=0, T=3, P=1) with uniform marginal: the threshold
//! is exactly `ω > 0.5` — verified in [`best_response::tests`]
//! `g8_cooperates_iff_omega_above_half_pd`.
//!
//! # Allocation discipline
//!
//! Per AGENTS.md hot-loop rules:
//! - [`SimilarityPosterior::observe`] is O(1): one fma on `log_w_independent`.
//! - [`SimilarityPosterior::predictive_similarity`] is O(D): one dot-product
//!   + one sigmoid per contemplated action (the continuous-embedding path).
//! - [`embedded_best_response`] is O(|A|²): payoff-matrix scan only.
//! - The discrete-action closed-form path (the G1-validated one) is fully
//!   allocation-free by construction.
//!
//! # Sigmoid, not softmax
//!
//! Per AGENTS.md constraint #2: `ω` is a posterior probability (scalar in
//! `(0,1)`), not a categorical. The cooperation decision is a step function
//! `ω > threshold` (a scalar comparison), not a softmax over actions. Action
//! selection uses plain expected-payoff argmax, not softmax sampling.
//!
//! # Substrate check (substrate-first skill, run before implementation)
//!
//! - **Searched for:** `SimilarityPosterior`, `similarity_inference`,
//!   `embedded_best_response`, `JointActionHistory` across `*.rs` in all 7
//!   repos → zero hits (no prior implementation).
//! - **PayoffTable<N>:** exists in `riir-ai/crates/riir-games-shared/src/payoff/`
//!   but is **combat-specific** (f64, `UnitSpec`, armor classes). Wrong shape
//!   for abstract normal-form games. Plan T1.6 explicitly says "grep first per
//!   substrate-first"; the existing type is domain-bound (combat) — the
//!   abstract game-theoretic matrix needs a separate generic type. We define
//!   [`PayoffMatrix<A>`] locally (f32, no combat semantics).
//! - **Decision:** BUILD NEW (the mechanism is genuinely novel per
//!   `.research/471` §2.3 + §3.5; the closest shipped cousin `CceLp<N,A>`
//!   uses *exogenous* designer correlation, not *endogenous* inferred
//!   correlation).
//! - **Architectural rules checked:**
//!   - Domain classification ✓ — `ω` is semantic/social (latent), not physical.
//!   - Sync boundary ✓ — `ω` stays local; only the final u8 action crosses.
//!   - Sigmoid not softmax ✓ — see above.
//!
//! # Phase 1 GOAT gate (this module)
//!
//! - **G1** — closed-form `ω_T` matches analytical `α/(α+(1−α)·2^(−T))` to f32
//!   epsilon for T=0..50, α=0.1; `embedded_best_response` cooperates iff
//!   `ω > 0.5` for canonical PD.
//! - **G2** — emergent-cooperation PoC (Plan 526 Phase 2, separate file).
//! - **G4** — alloc-free hot path (audited by construction; bench in Phase 4).
//!
//! # References
//!
//! - **Plan:** `katgpt-rs/.plans/526_similarity_inference_primitive.md`
//! - **Research:** `katgpt-rs/.research/471_Similarity_Inference_Embedded_Equilibrium.md`
//! - **Private guide:** `riir-ai/.research/335_Similarity_Inference_Emergent_Cooperation_Guide.md`
//! - **Source paper:** [arXiv:2608.03958](https://arxiv.org/abs/2608.03958) —
//!   Meulemans, Wołczyk, Weis, Nasser, et al. *Paradigms of Intelligence:
//!   A game theory for foundation models shows new paths to rational
//!   cooperation through similarity inference.* 4 Aug 2026.
//! - **Theory preprint:** [arXiv:2511.22226](https://arxiv.org/abs/2511.22226)
//!   — Meulemans et al. *Embedded Universal Predictive Intelligence.* 2025.
//! - **Closest non-Nashian prior art:** Oesterheld, Treutlein, Grosse,
//!   Conitzer, Foerster. *Similarity-based cooperative equilibrium.* NeurIPS
//!   2024. (Requires externally-provided similarity scores; this primitive
//!   *infers* them.)
//!
//! Similarity Inference — endogenous correlation device for embedded equilibrium
//! (Plan 526, Research 471, arXiv:2608.03958 Meulemans et al. *Paradigms of
//! Intelligence* Aug 2026). Per-focal `SimilarityPosterior ω ∈ (0,1)` updated
//! incrementally from joint-action history via the paper's §H.2 closed form
//! `ω_T = α/(α+(1−α)·|A|^(−T))`; `embedded_best_response` switches from
//! competitive-best-response (Nash) to cooperative-best-response (CCE) when ω
//! crosses a payoff-derived threshold (exactly 0.5 for canonical PD). Pure
//! modelless closed-form math; O(1) observe, O(A²) best-response, zero allocs
//! after construction. The mechanism is genuinely novel per R471 §3.5: the
//! shipped `CceLp<N,A>` (Plan 295, DEFAULT-ON) uses an *exogenous*
//! designer-set correlation device ζ; this primitive *infers* an *endogenous*
//! correlation device ω from interaction history. NOT a sync-boundary primitive
//! (ω is latent per-focal; only the final cooperate/defect u8 action crosses).
//! Sigmoid, not softmax (ω is a posterior probability, not a categorical).
//! OPT-IN (demoted 2026-09-04 per riir-ai Issue 867 T1.3 — 24 days DEFAULT-ON
//! 2026-08-11..09-03 with zero consumers workspace-wide) — Plan 526 Phase 1-5
//! GOAT G1–G8 ALL PASS (Bench 579: G1 closed-form reproduction, G2 emergent
//! cooperation, G5 indirect inference, G7 UQ floor, G8 PD threshold); see the
//! Cargo.toml feature-def comment for the full gate + demotion record.

pub mod best_response;
pub mod posterior;

#[cfg(test)]
mod poc;
#[cfg(test)]
mod tests;

pub use best_response::{
    PayoffMatrix, canonical_pd, embedded_best_response, embedded_best_response_into,
};
pub use posterior::SimilarityPosterior;

/// Errors raised by the similarity-inference primitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimilarityError {
    /// `prior_alpha` outside `(0, 1)`. The prior must be a valid probability
    /// strictly between 0 and 1 (0 → never believe partner is similar;
    /// 1 → certain a priori, no evidence can update).
    InvalidPriorAlpha {
        given: u32, // raw bits — f32 doesn't impl Eq
    },
    /// `n_actions` is zero. A game with zero actions is ill-defined.
    EmptyActionSet,
    /// Payoff-matrix shape mismatch: the supplied matrix is not `A × A`.
    PayoffShapeMismatch { expected: usize, got: usize },
    /// Partner-marginal length mismatch (expected `A`, got otherwise).
    MarginalShapeMismatch { expected: usize, got: usize },
}

/// A read-only stream of joint-action observations between one focal agent
/// and one partner.
///
/// Each observation is a triple `(self_action, partner_action, situation)` of
/// `&[f32]` embeddings. The trait is borrows-only — implementations own the
/// storage (KG-triple encounter log, mind-reading channel, trial hash-chain).
///
/// Phase 1 ships a single concrete posterior ([`SimilarityPosterior`]) that
/// consumes observations incrementally and does not require the caller to
/// implement this trait. The trait exists for future consumers that want to
/// replay a stored history through a fresh posterior (Phase 3 indirect
/// inference).
pub trait JointActionHistory {
    /// Push a new observation: the focal played `self_a`, the partner played
    /// `partner_a`, both in context `situation`. All three slices are
    /// read-only borrows; the implementation copies what it needs.
    fn push(&mut self, self_a: &[f32], partner_a: &[f32], situation: &[f32]);

    /// Number of observations stored so far.
    fn len(&self) -> usize;

    /// Whether the history is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
