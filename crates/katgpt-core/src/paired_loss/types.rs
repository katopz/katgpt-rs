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
///
/// # Layout (Phase 2 perf)
///
/// `#[repr(u8)]` + `CopyN(u8)` makes this enum exactly **2 bytes**
/// (1-byte discriminant + 1-byte payload). This matters for the G2 perf
/// gate: a `Vec<TokenClass>` of length 8192 is 16 KiB (fits in L1) instead
/// of 128 KiB (L2 territory with the default 16-byte layout). The n-gram
/// length is capped at 255 — more than enough (paper uses N=5; in practice
/// n ≤ 8). `n ≥ 256` saturates to 255 (the copy status matters more than the
/// exact n for the filtered aggregates).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TokenClass {
    /// Open-class content word (state-conditioned readout — paper Pattern i).
    Content = 0,
    /// Closed-class function word.
    Function = 1,
    /// Neither content nor function (e.g., punctuation, whitespace).
    Other = 2,
    /// Opening delimiter — initiates a new region/scope (state update).
    /// Paper Pattern ii: openers are hybrid-favored.
    BracketOpen = 3,
    /// Closing delimiter — satisfies an established structural obligation
    /// (state closure determined by visible opener). Paper Pattern ii:
    /// closers are transformer-favored.
    BracketClose = 4,
    /// Position completing a repeated n-gram of length `n` in the visible
    /// prefix. Paper Pattern iii: hybrid advantage vanishes here (visible-
    /// prefix retrieval suffices). `n ≥ 2` (a 1-gram "repeat" is trivial).
    /// `n` is capped at 255 (`u8`); larger n saturates (see Layout note).
    CopyN(u8) = 5,
}

impl TokenClass {
    /// `true` if this class is an "open-class" candidate for the `TopKNoCopy`
    /// filter — i.e., Content or Function. These are the state-conditioned
    /// readout positions where the paper's Patterns i/ii show the largest
    /// Transformer–Hybrid separation.
    ///
    /// Used by [`super::gap::PairedLossGap::filtered_mean`] to build a
    /// branchless mask for the single-pass fast path.
    #[inline(always)]
    pub fn is_open_class(self) -> bool {
        matches!(self, TokenClass::Content | TokenClass::Function)
    }

    /// Human-readable short label for display in reports/examples (Plan 335
    /// Phase 3). Used by [`super::ClassGapReport`] pretty-printing and the
    /// `paired_loss_0*` examples. Keeps display logic in one place rather than
    /// re-implementing `match` at each consumer.
    ///
    /// `CopyN(n)` renders as `"CopyN"` (the `n` is intentionally omitted —
    /// the display label is a fixed `&'static str`; the full n is available
    /// via the variant itself).
    #[inline]
    pub fn label(self) -> &'static str {
        match self {
            TokenClass::Content => "Content",
            TokenClass::Function => "Function",
            TokenClass::Other => "Other",
            TokenClass::BracketOpen => "BracketOpen",
            TokenClass::BracketClose => "BracketClose",
            TokenClass::CopyN(_) => "CopyN",
        }
    }
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

/// Reusable scratch buffer for the zero-alloc SIMD hot path (Plan 335
/// Phase 2 T2.2).
///
/// [`PairedLossGap::filtered_mean`] and [`PairedLossGap::filtered_mean_with_scratch`]
/// both compute a masked sum over `deltas`. The SIMD backend
/// (`simd_masked_sum_count_f32`) operates on a `&[u8]` mask, which is built
/// from the `&[TokenClass]` array. Building the mask allocates if done
/// naively; this scratch buffer reuses the allocation across calls.
///
/// # Usage
///
/// ```ignore
/// let mut scratch = FilterScratch::default();
/// let mean = gap.filtered_mean_with_scratch(&classes, filter, &mut scratch);
/// // Subsequent calls reuse the same mask buffer — zero net allocs.
/// let mean2 = gap.filtered_mean_with_scratch(&classes, other_filter, &mut scratch);
/// ```
///
/// The scratch buffer grows to fit the largest `classes` array seen; it does
/// not shrink. For a fixed-L eval loop, it allocates once on the first call
/// and is free thereafter.
#[derive(Clone, Debug, Default)]
pub struct FilterScratch {
    /// Reusable mask buffer (1 byte per token). Grown as needed via
    /// `resize`; never shrunk. Reused across `filtered_mean_with_scratch`
    /// calls to achieve zero-alloc hot path.
    pub(crate) mask_buf: Vec<u8>,
}

impl FilterScratch {
    /// Construct an empty scratch buffer (no pre-allocation).
    #[inline]
    pub const fn new() -> Self {
        Self { mask_buf: Vec::new() }
    }

    /// Construct with a pre-allocated capacity. Avoids the first-call grow.
    #[inline]
    pub fn with_capacity(l: usize) -> Self {
        Self {
            mask_buf: Vec::with_capacity(l),
        }
    }
}

/// Per-class annotation row pairing the observed mean gap with the
/// Proposition 1 class-size bound (Plan 335 Phase 3 T3.1).
///
/// Produced by [`super::gap::PairedLossGap::annotate_with_class_bounds`].
/// The `gap_to_bound_ratio = mean_gap / log_v_tau` tells you how much of
/// the theoretical ceiling the observed A/B gap has consumed:
///
/// - **ratio → 1** → near the Proposition 1 ceiling. A richer feature map
///   (e.g., latent encoding, recurrence) has captured most of the
///   theoretically available room — little left to gain.
/// - **ratio → 0** → far from the ceiling. Room remains for a richer feature
///   to help on this class.
/// - **ratio < 0** → the A/B is "backwards" on this class (model A is better
///   than B). The richer feature is *not* helping here; investigate the
///   sign before concluding anything about the bound.
/// - **ratio > 1** → would exceed the Proposition 1 bound. Shouldn't happen
///   for a valid `ClassSizeBound` (V_τ chosen too small) — the bound is
///   generous, not tight; revisit the assumed V_τ.
/// - **NaN** → no [`ClassSizeBound`] was provided for this class. `mean_gap`
///   and `count` are still valid; only the bound/ratio are undefined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassGapRow {
    /// The class this row annotates.
    pub class: TokenClass,
    /// Number of positions in `classes` with this tag.
    pub count: u32,
    /// Mean `Δ_i = ℓ_A − ℓ_B` over positions of this class. Equal to
    /// [`super::gap::PairedLossGap::mean_gap_for_class`] for this class.
    /// Sign: positive = B-favored.
    pub mean_gap: f32,
    /// `log|V_τ|` from the [`ClassSizeBound`] for this class, or `NaN` if no
    /// bound was supplied.
    pub log_v_tau: f32,
    /// `mean_gap / log_v_tau`. See struct docs for the interpretation.
    /// `NaN` when `log_v_tau` is `NaN` (no bound) or zero (`V_τ = 1`, a
    /// deterministic class — `mean_gap` should also be ~0 in that case).
    pub gap_to_bound_ratio: f32,
}

/// Per-class Proposition 1 annotation report (Plan 335 Phase 3 T3.1).
///
/// One [`ClassGapRow`] per distinct [`TokenClass`] present in the input
/// `classes` array. Rows are sorted by `gap_to_bound_ratio` **descending**
/// (NaN-aware: rows with NaN ratio sort last), so the classes nearest their
/// Proposition 1 ceiling appear first — the actionable diagnostic ("where
/// has the richer feature already saturated the available room?").
///
/// Produced by
/// [`super::gap::PairedLossGap::annotate_with_class_bounds`].
///
/// # Zero-alloc note
///
/// This is a **cold-path reporting API**, not a hot-path query. It allocates
/// the `rows` Vec once (one `Vec::with_capacity(distinct_classes)` + one
/// `HashMap` for accumulation). Use it once per eval report, not per token.
/// The hot path is [`super::gap::PairedLossGap::filtered_mean_with_scratch`].
#[derive(Clone, Debug, Default)]
pub struct ClassGapReport {
    /// One row per distinct class present in the input, sorted by
    /// `gap_to_bound_ratio` descending (NaN last).
    pub rows: Vec<ClassGapRow>,
}

impl ClassGapReport {
    /// Look up the row for a specific class. Returns `None` if the class was
    /// not present in the input `classes` array. O(rows.len()) linear scan —
    /// `rows` is small (≤ ~10 distinct classes in practice).
    #[inline]
    pub fn row_for(&self, class: TokenClass) -> Option<&ClassGapRow> {
        self.rows.iter().find(|r| r.class == class)
    }

    /// `true` if no classes were annotated (empty input `classes` array).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Number of annotated classes.
    #[inline]
    pub fn len(&self) -> usize {
        self.rows.len()
    }
}
