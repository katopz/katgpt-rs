//! Core types for the paired loss gap diagnostic (Plan 335 Phase 1 T1.2).
//!
//! Generic over nothing — all types work on `&[f32]` log-prob traces and
//! `&[TokenClass]` tag arrays. No game/chain/shard semantics.
//!
//! # Latent vs Raw (AGENTS.md)
//!
//! - `PairedLossGap::deltas` → raw (output of forward passes; the consumer
//!   owns the raw-vs-latent decision upstream). This primitive operates on
//!   whatever log-prob trace the consumer hands it.
//! - `ClassSizeBound::log_v_tau` → raw (theoretical bound; a closed-form log
//!   of a vocabulary size). Not synced — it's a constant annotation.
//! - `TokenClass` → raw (a tag label). Not synced — consumer-side metadata.
//!
//! # Why these types live here (not in a consumer repo)
//!
//! All four types are generic math/data structures with zero game/chain/shard
//! semantics. Any consumer (riir-ai NPC runtime GOAT gates, riir-chain LatCal
//! theoretical footnotes, katgpt-rs root A/B evals) can use them. See
//! Research 319 §2.1 ("Generic: works on any pair of log-prob traces").

/// The per-token paired loss gap trace `Δ_i = ℓ_A − ℓ_B`.
///
/// Constructed once from two equal-length log-probability traces via
/// [`PairedLossGap::from_log_probs`]. The deltas are the only mutable state;
/// all query methods (`mean_gap`, `mean_gap_for_class`, `filtered_mean`) are
/// `&self` and allocate zero heap memory on the hot path (they use iterator
/// folds over the cached deltas).
///
/// **Sign convention:** `Δ_i > 0` means model A assigned LOWER probability
/// (higher loss) than model B at position i — i.e., position i is
/// **B-favored**. The paper (Li & Merrill 2026) uses A = Transformer, B =
/// Hybrid, so `Δ_i > 0` = hybrid-favored. Callers keep whichever convention
/// they want; the math is symmetric.
#[derive(Clone, Debug)]
pub struct PairedLossGap {
    /// Per-token `Δ_i = ℓ_A[i] − ℓ_B[i]`. Length L. Owned (allocated once at
    /// construction by `from_log_probs` via `Vec::with_capacity(L)`).
    pub(crate) deltas: Vec<f32>,
}

/// Token class tag for stratified aggregation (paper §3 + §6).
///
/// The paper's three-way aggregate is Content/Function/Other. We add
/// BracketOpen/BracketClose to capture the state-update vs state-closure
/// asymmetry (paper §4 Pattern ii: openers are hybrid-favored, closers are
/// transformer-favored), and CopyN(n) to capture repeated n-gram reuse
/// (paper §4 Pattern iii: hybrid advantage vanishes on copy positions).
///
/// `CopyN(n)` marks a position completing a repeated n-gram of length `n` in
/// the visible prefix (paper's COPY_k feature). With this enum, copy status
/// is **merged** into the class — a position is EITHER Content OR CopyN, not
/// both. This is a deliberate simplification: it makes the `TopKNoCopy` filter
/// naturally exclude all copy positions (they're disjoint from Content/
/// Function). The paper tracks copy orthogonally; our merged enum gives the
/// same filtered-aggregate result for the synthetic G1 fixture (Phase 2 may
/// revisit if a richer tagger needs orthogonal copy tracking).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenClass {
    /// Open-class content word (state-conditioned readout — paper Pattern i).
    Content,
    /// Closed-class function word.
    Function,
    /// Neither content nor function (e.g., punctuation, whitespace).
    Other,
    /// Opening delimiter — initiates a new region/scope (state update).
    /// Paper Pattern ii: openers are hybrid-favored.
    BracketOpen,
    /// Closing delimiter — satisfies an established structural obligation
    /// (state closure determined by visible opener). Paper Pattern ii:
    /// closers are transformer-favored.
    BracketClose,
    /// Position completing a repeated n-gram of length `n` in the visible
    /// prefix. Paper Pattern iii: hybrid advantage vanishes here (visible-
    /// prefix retrieval suffices). `n ≥ 2` (a 1-gram "repeat" is trivial).
    CopyN(usize),
}

/// The Proposition 1 class-size bound (paper §5).
///
/// `DKL(p⋆_τ ‖ p_ϕ,τ) ≤ log|V_τ|` — the reducible loss from any richer
/// feature map `ϕ` is bounded by the log-vocabulary-size of the target class.
/// For small `V_τ` (physical domain: boolean, u8, grid coords), the bound is
/// near-zero → raw commitment is information-theoretically sufficient. For
/// large `V_τ` (semantic domain: open-class content), the bound is loose →
/// latent encoding earns its keep. See Research 319 §2.2 for the raw-vs-latent
/// justification mapping.
///
/// **Important:** this is a *bound*, not an equality (Research 319 §5 R4).
/// `reducible_loss_ceiling()` returns the worst-case upper bound; the actual
/// reducible loss can be much smaller. Don't overclaim that raw commitment is
/// *optimal* — only that the *room for latent encoding to help* is bounded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassSizeBound {
    /// `log|V_τ|` — the natural log of the class vocabulary size. The
    /// Proposition 1 upper bound on `DKL(p⋆_τ ‖ p_ϕ,τ)`.
    pub log_v_tau: f32,
}

impl ClassSizeBound {
    /// Compute the Proposition 1 bound for a class with `v_tau` possible
    /// values. `log_v_tau = (v_tau as f32).ln()`. O(1).
    ///
    /// # Examples
    /// - `v_tau = 2` (boolean) → `log_v_tau ≈ 0.693` — physical domain, raw
    ///   commitment sufficient.
    /// - `v_tau = 256` (u8) → `log_v_tau ≈ 5.545`.
    /// - `v_tau = 50_000` (open-class noun) → `log_v_tau ≈ 10.82` — semantic
    ///   domain, latent encoding earns its keep.
    #[inline]
    pub fn for_vocab_size(v_tau: usize) -> Self {
        // v_tau = 0 → undefined (log 0). Guard: return +inf bound (no room
        // claimed, no overclaim). v_tau = 1 → log 1 = 0 (deterministic class,
        // zero reducible loss — correct).
        let log_v_tau = if v_tau == 0 {
            f32::INFINITY
        } else {
            (v_tau as f32).ln()
        };
        Self { log_v_tau }
    }

    /// The Proposition 1 upper bound on `DKL(p⋆_τ ‖ p_ϕ,τ)` — i.e., the
    /// worst-case room for ANY richer feature map (including a learned latent
    /// representation) to beat the class-only predictor. Returns `log_v_tau`.
    ///
    /// A class with `reducible_loss_ceiling() ≈ 0` (small `V_τ`) has no room
    /// for latent encoding to help — raw commitment is sufficient. A class
    /// with a large ceiling has room to grow.
    #[inline]
    pub fn reducible_loss_ceiling(&self) -> f32 {
        self.log_v_tau
    }
}

/// The filtered-eval mode (paper §6).
///
/// All three filters are computed from the same per-token NLL — negligible
/// overhead, capability-resolved view. The paper shows `TOP-K∩NO-COPY`
/// roughly doubles the Transformer–Hybrid separation vs `ALL_TOKENS` on 1B
/// pretraining runs (Figure 7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilterKind {
    /// Standard aggregate — mean over ALL tokens. The `ALL_TOKENS` baseline.
    /// Equivalent to [`PairedLossGap::mean_gap`].
    AllTokens,
    /// Paper's `TOP-K∩NO-COPY`: the K most-Δ-favored open-class (Content/
    /// Function) classes, excluding CopyN positions with n ≤ max_ngram.
    ///
    /// With the merged [`TokenClass`] enum (CopyN is disjoint from Content/
    /// Function), the CopyN exclusion is automatically satisfied — all
    /// CopyN positions are already excluded by the Content/Function mask.
    /// `max_ngram` is retained for API fidelity to the paper and for forward-
    /// compat with orthogonal-copy taggers; it has no effect with the merged
    /// enum.
    TopKNoCopy {
        /// Number of open-class candidates to select (paper uses K=10 POS
        /// families; our enum has 2 open-class candidates: Content, Function).
        /// If `k ≥ 2`, both are selected. If `k = 1`, only the more-Δ-favored.
        k: usize,
        /// Exclude CopyN(n) positions with `n ≤ max_ngram`. No-op with the
        /// merged enum (CopyN is already disjoint). Retained for API fidelity.
        max_ngram: usize,
    },
    /// Paper's `COPY-N-ONLY`: positions completing a repeated N-gram of
    /// length exactly `n`. Isolates visible-prefix retrieval (paper Pattern
    /// iii: hybrid advantage vanishes here).
    CopyNOnly {
        /// The exact n-gram length to isolate (paper uses N=5).
        n: usize,
    },
}
