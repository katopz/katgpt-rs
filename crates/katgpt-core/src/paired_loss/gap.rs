//! Paired loss gap math (Plan 335 Phase 1 T1.3–T1.6).
//!
//! All methods are `&self`, operate on the cached `deltas` vec, and allocate
//! zero heap memory (iterator folds, no intermediate `Vec`). The only
//! allocation in the whole module is the one-time `Vec::with_capacity(L)` in
//! [`PairedLossGap::from_log_probs`].
//!
//! # SIMD
//!
//! `mean_gap` uses [`crate::simd::simd_sum_f32`] for the horizontal reduction
//! (the hot-path op where SIMD matters most). `from_log_probs` uses a direct
//! subtract loop — LLVM auto-vectorizes `dst[i] = a[i] - b[i]` trivially on
//! f32, and the one-pass construction avoids the zero-then-fma dance.

use crate::simd::simd_sum_f32;
use crate::paired_loss::types::{FilterKind, PairedLossGap, TokenClass};

impl PairedLossGap {
    /// Construct the per-token gap trace from two log-probability sequences.
    ///
    /// `Δ_i = ℓ_A[i] − ℓ_B[i]` for `i in 0..L`. The two slices MUST be
    /// equal-length (panics otherwise — a length mismatch is a caller bug,
    /// not a recoverable condition).
    ///
    /// O(L) subtract, one allocation (`Vec::with_capacity(L)`). The
    /// allocation is necessary — it IS the output. Subsequent query methods
    /// are zero-alloc.
    #[inline]
    pub fn from_log_probs(log_probs_a: &[f32], log_probs_b: &[f32]) -> Self {
        assert_eq!(
            log_probs_a.len(),
            log_probs_b.len(),
            "PairedLossGap::from_log_probs: log-prob traces must have equal length \
             (got {} vs {})",
            log_probs_a.len(),
            log_probs_b.len()
        );
        let len = log_probs_a.len();
        let mut deltas = Vec::with_capacity(len);
        // Direct subtract — LLVM auto-vectorizes f32 dst[i]=a[i]-b[i].
        for i in 0..len {
            deltas.push(log_probs_a[i] - log_probs_b[i]);
        }
        Self { deltas }
    }

    /// Raw read access to the per-token `Δ_i` trace (for consumers that want
    /// to compute their own aggregates). Length L.
    #[inline]
    pub fn deltas(&self) -> &[f32] {
        &self.deltas
    }

    /// Number of tokens in the trace.
    #[inline]
    pub fn len(&self) -> usize {
        self.deltas.len()
    }

    /// `true` if the trace is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }

    /// The aggregate mean gap `Δ̄ = mean(Δ_i)` — the `ALL_TOKENS` filter
    /// (paper §3). O(L) SIMD horizontal sum, zero allocation.
    ///
    /// Returns `0.0` for an empty trace (mathematically undefined; returning
    /// 0.0 avoids NaN propagation in benchmark stats). Callers that need to
    /// distinguish "empty" from "zero gap" should check [`Self::is_empty`].
    #[inline]
    pub fn mean_gap(&self) -> f32 {
        let len = self.deltas.len();
        if len == 0 {
            return 0.0;
        }
        simd_sum_f32(&self.deltas) / (len as f32)
    }

    /// Tag-stratified raw mean — the mean `Δ_i` over positions whose class
    /// equals `target` (paper §3 Analysis I). O(L) single-pass fold, zero
    /// allocation.
    ///
    /// For `target = TokenClass::CopyN(n)`, matches positions with that
    /// EXACT `n` (e.g., `CopyN(5)` matches only `CopyN(5)`, not `CopyN(4)`).
    /// This mirrors the paper's `COPY-N-ONLY` filter (exact N).
    ///
    /// Returns `0.0` if no positions match `target` (empty bucket). Callers
    /// that care can pre-count matches.
    #[inline]
    pub fn mean_gap_for_class(&self, classes: &[TokenClass], target: TokenClass) -> f32 {
        debug_assert_eq!(
            classes.len(),
            self.deltas.len(),
            "mean_gap_for_class: classes.len() ({}) != deltas.len() ({})",
            classes.len(),
            self.deltas.len()
        );
        let (sum, count) = self
            .deltas
            .iter()
            .zip(classes)
            .fold((0.0f32, 0u32), |(s, c), (&d, cls)| {
                if *cls == target {
                    (s + d, c + 1)
                } else {
                    (s, c)
                }
            });
        if count == 0 {
            0.0
        } else {
            sum / (count as f32)
        }
    }

    /// Filtered aggregate mean (paper §6) — amplifies small architecture gaps
    /// that aggregate loss hides. O(L) per filter mode, zero allocation
    /// (iterator folds, no mask `Vec`).
    ///
    /// - [`FilterKind::AllTokens`]: delegates to [`Self::mean_gap`].
    /// - [`FilterKind::TopKNoCopy`]: the K most-Δ-favored open-class
    ///   (Content/Function) classes, excluding CopyN positions. With the
    ///   merged [`TokenClass`] enum, CopyN is already disjoint from Content/
    ///   Function, so the filter selects positions whose class is in the
    ///   top-K open-class candidates by mean Δ.
    /// - [`FilterKind::CopyNOnly`]: positions with class `CopyN(n)` (exact n).
    ///
    /// Returns `0.0` for an empty mask (no positions match the filter).
    #[inline]
    pub fn filtered_mean(&self, classes: &[TokenClass], filter: FilterKind) -> f32 {
        debug_assert_eq!(
            classes.len(),
            self.deltas.len(),
            "filtered_mean: classes.len() ({}) != deltas.len() ({})",
            classes.len(),
            self.deltas.len()
        );
        match filter {
            FilterKind::AllTokens => self.mean_gap(),
            FilterKind::CopyNOnly { n } => {
                self.mean_gap_for_class(classes, TokenClass::CopyN(n))
            }
            FilterKind::TopKNoCopy { k, max_ngram: _ } => {
                self.filtered_mean_topk_nocopy(classes, k)
            }
        }
    }

    /// The `TOP-K∩NO-COPY` core. See [`FilterKind::TopKNoCopy`] doc.
    ///
    /// Candidates: Content, Function (the open-class families where state-
    /// conditioned readout matters — paper Pattern i). Select top-K by mean Δ
    /// (largest Δ = most B-favored). With the merged enum, CopyN/Other/
    /// brackets are naturally excluded.
    #[inline]
    fn filtered_mean_topk_nocopy(&self, classes: &[TokenClass], k: usize) -> f32 {
        // Step 1: per-candidate mean Δ (single pass, fold per candidate).
        // Two candidates → unroll; no heap alloc.
        let (sum_c, cnt_c) = self.class_sum_count(classes, TokenClass::Content);
        let (sum_f, cnt_f) = self.class_sum_count(classes, TokenClass::Function);

        // Step 2: rank candidates by mean Δ (descending = most B-favored first).
        // Stack array, sort 2 elements — zero heap alloc.
        let mean_c = if cnt_c > 0 { sum_c / (cnt_c as f32) } else { f32::NEG_INFINITY };
        let mean_f = if cnt_f > 0 { sum_f / (cnt_f as f32) } else { f32::NEG_INFINITY };
        // Order: [(mean, variant), ...] descending by mean. NEG_INFINITY sorts
        // last, so empty candidates never win a slot.
        let mut ranked = [(mean_c, TokenClass::Content), (mean_f, TokenClass::Function)];
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));

        // Step 3: select top-K candidate classes. If k ≥ candidate count,
        // all (non-empty) candidates are selected.
        let take = k.min(ranked.len());
        let selected = &ranked[..take];

        // Step 4: single-pass masked sum over positions whose class is in the
        // selected set. CopyN positions are naturally excluded (they're not
        // Content/Function). Zero alloc.
        let (sum, count) = self.deltas.iter().zip(classes).fold(
            (0.0f32, 0u32),
            |(s, c), (&d, cls)| {
                if selected.iter().any(|(_, variant)| variant == cls) {
                    (s + d, c + 1)
                } else {
                    (s, c)
                }
            },
        );
        if count == 0 {
            0.0
        } else {
            sum / (count as f32)
        }
    }

    /// Helper: sum + count of `Δ_i` where `classes[i] == target`. Single pass.
    #[inline]
    fn class_sum_count(&self, classes: &[TokenClass], target: TokenClass) -> (f32, u32) {
        self.deltas
            .iter()
            .zip(classes)
            .fold((0.0f32, 0u32), |(s, c), (&d, cls)| {
                if *cls == target {
                    (s + d, c + 1)
                } else {
                    (s, c)
                }
            })
    }
}
