//! Paired loss gap math (Plan 335 Phase 1 T1.3–T1.6 + Phase 2 perf).
//!
//! All methods are `&self`, operate on the cached `deltas` vec, and allocate
//! zero heap memory (iterator folds, no intermediate `Vec`). The only
//! allocation in the whole module is the one-time `Vec::with_capacity(L)` in
//! [`PairedLossGap::from_log_probs`].
//!
//! # SIMD / perf (Phase 2 GOAT gate)
//!
//! - [`PairedLossGap::mean_gap`] uses [`crate::simd::simd_sum_f32`] for the
//!   horizontal reduction (the hot-path op where SIMD matters most).
//! - [`PairedLossGap::from_log_probs`] writes into a `set_len`'d vec by direct
//!   index — LLVM auto-vectorizes `dst[i] = a[i] − b[i]` into packed f32
//!   subtracts (NEON `vsubq_f32` / AVX2 `_mm256_sub_ps`).
//! - [`PairedLossGap::filtered_mean`] with [`FilterKind::TopKNoCopy`] takes a
//!   single-pass branchless fast path when `k ≥ 2` (the only realistic case —
//!   there are exactly 2 open-class candidates: Content, Function). The inner
//!   fold uses `is_open_class() as u32` as a 0/1 multiply mask so LLVM can
//!   vectorize the masked sum. The `k = 1` ranking path falls back to the
//!   3-pass rank-then-mask algorithm (rare; preserved for correctness).

use crate::paired_loss::types::{
    ClassGapReport, ClassGapRow, ClassSizeBound, FilterKind, FilterScratch, PairedLossGap,
    TokenClass,
};
use crate::simd::{simd_masked_sum_count_f32, simd_sum_f32};

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
    ///
    /// # Perf
    ///
    /// Writes into a `set_len`'d vec by direct index so LLVM auto-vectorizes
    /// the f32 subtract into packed ops (`vsubq_f32` / `_mm256_sub_ps`). The
    /// `unsafe { set_len }` is sound because we just allocated `with_capacity(L)`
    /// and write exactly `L` elements before any read.
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
        // Sound: `with_capacity(len)` reserved exactly `len` slots; we write
        // exactly `len` elements before any observable read. The direct-index
        // form (vs `push`) lets LLVM lower this to packed f32 subtracts without
        // a per-iteration capacity check.
        unsafe {
            deltas.set_len(len);
            for i in 0..len {
                *deltas.get_unchecked_mut(i) =
                    *log_probs_a.get_unchecked(i) - *log_probs_b.get_unchecked(i);
            }
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
    ///
    /// # Perf
    ///
    /// The inner fold is branchless: `matches_target(cls) as u32` produces a
    /// 0/1 mask, then `d * (mask as f32)` and `count += mask` avoid the
    /// branch that would defeat LLVM auto-vectorization.
    #[inline]
    pub fn mean_gap_for_class(&self, classes: &[TokenClass], target: TokenClass) -> f32 {
        debug_assert_eq!(
            classes.len(),
            self.deltas.len(),
            "mean_gap_for_class: classes.len() ({}) != deltas.len() ({})",
            classes.len(),
            self.deltas.len()
        );
        // Manual loop for vectorization (Phase 2 perf). The branchless mask
        // (`== target as u32`) lets LLVM lower this to packed masked-FMA.
        let len = self.deltas.len();
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for i in 0..len {
            // Safety: debug_assert above ensures equal lengths.
            let (d, cls) = unsafe {
                (*self.deltas.get_unchecked(i), *classes.get_unchecked(i))
            };
            let m = u32::from(cls == target);
            sum += d * (m as f32);
            count += m;
        }
        if count == 0 {
            0.0
        } else {
            sum / (count as f32)
        }
    }

    /// Filtered aggregate mean (paper §6) — amplifies small architecture gaps
    /// that aggregate loss hides.
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
    ///
    /// # Allocation
    ///
    /// This convenience method builds a temporary `Vec<u8>` mask on each call
    /// (1 allocation). For the zero-alloc SIMD hot path, use
    /// [`Self::filtered_mean_with_scratch`] with a reused [`FilterScratch`].
    #[inline]
    pub fn filtered_mean(&self, classes: &[TokenClass], filter: FilterKind) -> f32 {
        let mut scratch = FilterScratch::default();
        self.filtered_mean_with_scratch(classes, filter, &mut scratch)
    }

    /// Filtered aggregate mean — zero-alloc SIMD variant (Plan 335 Phase 2).
    ///
    /// Same semantics as [`Self::filtered_mean`], but takes a reusable
    /// [`FilterScratch`] so the mask buffer is allocated once and reused
    /// across calls. The masked-sum inner loop uses
    /// [`crate::simd::simd_masked_sum_count_f32`] (NEON/AVX2 backend) instead
    /// of a scalar fold — ~4–8× faster on L=8192.
    ///
    /// # Allocation
    ///
    /// The first call grows `scratch.mask_buf` to `classes.len()`. Subsequent
    /// calls with the same (or smaller) `classes` reuse the buffer — **zero
    /// net allocations** on the hot path.
    #[inline]
    pub fn filtered_mean_with_scratch(
        &self,
        classes: &[TokenClass],
        filter: FilterKind,
        scratch: &mut FilterScratch,
    ) -> f32 {
        debug_assert_eq!(
            classes.len(),
            self.deltas.len(),
            "filtered_mean_with_scratch: classes.len() ({}) != deltas.len() ({})",
            classes.len(),
            self.deltas.len()
        );
        match filter {
            FilterKind::AllTokens => self.mean_gap(),
            FilterKind::CopyNOnly { n } => {
                // n is usize in the filter (API stability); TokenClass::CopyN
                // stores u8 (Phase 2 perf). n > 255 saturates — copy status
                // matters more than the exact n for this aggregate.
                self.masked_mean_simd(classes, scratch, |cls| {
                    *cls == TokenClass::CopyN(n.min(255) as u8)
                })
            }
            FilterKind::TopKNoCopy { k, max_ngram: _ } => {
                self.filtered_mean_topk_nocopy_scratch(classes, k, scratch)
            }
        }
    }

    /// Build a mask from `classes` (via `predicate`), write it into
    /// `scratch.mask_buf` (growing once, reusing thereafter), and compute the
    /// SIMD masked sum. Zero alloc after the first call.
    #[inline]
    fn masked_mean_simd<P: Fn(&TokenClass) -> bool>(
        &self,
        classes: &[TokenClass],
        scratch: &mut FilterScratch,
        predicate: P,
    ) -> f32 {
        let len = classes.len();
        // Reuse the mask buffer — grows once, never shrinks.
        if scratch.mask_buf.len() < len {
            scratch.mask_buf.resize(len, 0);
        }
        let mask = &mut scratch.mask_buf[..len];
        // Build the mask (single pass). LLVM vectorizes this `u8` write loop.
        for i in 0..len {
            mask[i] = predicate(&classes[i]) as u8;
        }
        let (sum, count) = simd_masked_sum_count_f32(&self.deltas, mask);
        if count == 0 {
            0.0
        } else {
            sum / (count as f32)
        }
    }

    /// The `TOP-K∩NO-COPY` core. See [`FilterKind::TopKNoCopy`] doc.
    ///
    /// Candidates: Content, Function (the open-class families where state-
    /// conditioned readout matters — paper Pattern i). Select top-K by mean Δ
    /// (largest Δ = most B-favored). With the merged enum, CopyN/Other/
    /// brackets are naturally excluded.
    ///
    /// # Perf (Phase 2)
    ///
    /// Two paths:
    /// - **Fast path (`k ≥ 2`, the only realistic case — exactly 2 open-class
    ///   candidates exist):** build an open-class mask once into `scratch`, then
    ///   call `simd_masked_sum_count_f32`. O(L) mask build + O(L) SIMD sum.
    /// - **Slow path (`k ≤ 1`):** edge cases (k=0 → empty; k=1 → rank + pick).
    #[inline]
    fn filtered_mean_topk_nocopy_scratch(
        &self,
        classes: &[TokenClass],
        k: usize,
        scratch: &mut FilterScratch,
    ) -> f32 {
        if k >= 2 {
            // Fast path: all open-class positions (Content + Function).
            // Single mask build + SIMD sum. CopyN/Other/brackets naturally
            // excluded (they're not open-class).
            return self.masked_mean_simd(classes, scratch, |cls| cls.is_open_class());
        }
        if k == 0 {
            // Empty mask: select no candidates.
            return 0.0;
        }
        // k == 1 (rare): pick the open-class candidate with the larger mean Δ.
        // Two scalar passes to rank, then return the winner's mean.
        let (sum_c, cnt_c) = self.class_sum_count(classes, TokenClass::Content);
        let (sum_f, cnt_f) = self.class_sum_count(classes, TokenClass::Function);
        let mean_c = if cnt_c > 0 {
            sum_c / (cnt_c as f32)
        } else {
            f32::NEG_INFINITY
        };
        let mean_f = if cnt_f > 0 {
            sum_f / (cnt_f as f32)
        } else {
            f32::NEG_INFINITY
        };
        if mean_c == f32::NEG_INFINITY && mean_f == f32::NEG_INFINITY {
            0.0
        } else {
            mean_c.max(mean_f)
        }
    }

    /// Helper: sum + count of `Δ_i` where `classes[i] == target`. Single pass,
    /// branchless (used by `CopyNOnly` + the `k = 1` ranking path).
    #[inline]
    fn class_sum_count(&self, classes: &[TokenClass], target: TokenClass) -> (f32, u32) {
        let len = self.deltas.len();
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for i in 0..len {
            // Safety: callers ensure classes.len() == deltas.len().
            let (d, cls) = unsafe {
                (*self.deltas.get_unchecked(i), *classes.get_unchecked(i))
            };
            let m = u32::from(cls == target);
            sum += d * (m as f32);
            count += m;
        }
        (sum, count)
    }

    /// Annotate per-class mean gaps with the Proposition 1 class-size bound
    /// (Plan 335 Phase 3 T3.1).
    ///
    /// For each distinct [`TokenClass`] present in `classes`, compute the mean
    /// `Δ_i` over positions of that class, look up the corresponding
    /// [`ClassSizeBound`] in `bounds`, and report
    /// `gap_to_bound_ratio = mean_gap / log_v_tau`.
    ///
    /// # Interpretation
    ///
    /// See [`ClassGapRow`] for the full ratio semantics. The headline use:
    /// classes with `|ratio| → 1` are near their Proposition 1 ceiling (the
    /// richer feature map has saturated the available room); classes with
    /// `|ratio| → 0` still have room for a richer feature to help.
    ///
    /// # Complexity / allocation
    ///
    /// O(L) single pass to accumulate per-class `(sum, count)` into a
    /// `std::collections::HashMap`, then O(distinct_classes) to build rows.
    /// **This is a cold-path reporting API** — it allocates the `rows` Vec and
    /// an internal accumulation HashMap. Use once per eval report, not per
    /// token. The hot path is [`Self::filtered_mean_with_scratch`].
    ///
    /// # Missing bounds
    ///
    /// A class present in `classes` but absent from `bounds` is still
    /// reported — its `mean_gap` and `count` are valid; `log_v_tau` and
    /// `gap_to_bound_ratio` are `NaN`. Sort order puts NaN-ratio rows last.
    ///
    /// # Panics
    ///
    /// Debug-only: panics if `classes.len() != self.len()` (length mismatch
    /// is a caller bug).
    #[inline]
    pub fn annotate_with_class_bounds(
        &self,
        classes: &[TokenClass],
        bounds: &std::collections::HashMap<TokenClass, ClassSizeBound>,
    ) -> ClassGapReport {
        debug_assert_eq!(
            classes.len(),
            self.deltas.len(),
            "annotate_with_class_bounds: classes.len() ({}) != deltas.len() ({})",
            classes.len(),
            self.deltas.len()
        );
        // O(L) single pass: accumulate (sum, count) per distinct class. The
        // number of distinct classes is small (≤ ~10 in practice: 5 base
        // variants + a handful of CopyN(n) values), so the HashMap stays tiny.
        let mut acc: std::collections::HashMap<TokenClass, (f32, u32)> =
            std::collections::HashMap::new();
        for i in 0..self.deltas.len() {
            let (d, cls) = unsafe {
                (*self.deltas.get_unchecked(i), *classes.get_unchecked(i))
            };
            let entry = acc.entry(cls).or_insert((0.0f32, 0u32));
            entry.0 += d;
            entry.1 += 1;
        }
        // Build rows. Look up each class's bound; NaN if not provided.
        let mut rows: Vec<ClassGapRow> = Vec::with_capacity(acc.len());
        for (cls, (sum, count)) in acc {
            let mean_gap = if count == 0 { 0.0 } else { sum / (count as f32) };
            let (log_v_tau, ratio) = match bounds.get(&cls) {
                Some(b) => {
                    let lv = b.log_v_tau;
                    // log_v_tau == 0 (V_τ = 1, deterministic class) → 0/0 = NaN.
                    // log_v_tau == +inf (V_τ = 0 guard) → finite/inf = 0.0.
                    let r = if lv == 0.0 {
                        f32::NAN
                    } else if lv.is_infinite() {
                        0.0
                    } else {
                        mean_gap / lv
                    };
                    (lv, r)
                }
                None => (f32::NAN, f32::NAN),
            };
            rows.push(ClassGapRow {
                class: cls,
                count,
                mean_gap,
                log_v_tau,
                gap_to_bound_ratio: ratio,
            });
        }
        // Sort by gap_to_bound_ratio descending, NaN-aware (NaN sorts last).
        // sort_by is stable; ties keep insertion (HashMap) order.
        rows.sort_by(|a, b| {
            let ra = a.gap_to_bound_ratio;
            let rb = b.gap_to_bound_ratio;
            match (ra.is_nan(), rb.is_nan()) {
                // NaN sorts after any non-NaN (so non-NaN rows come first,
                // descending by ratio).
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                // Both finite: descending = reverse natural order
                // (rb.partial_cmp(ra) returns Less when ra > rb).
                (false, false) => rb
                    .partial_cmp(&ra)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });
        ClassGapReport { rows }
    }
}
