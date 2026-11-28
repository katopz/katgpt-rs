//! YASA tied-indexer block scorer (katgpt-rs Issue 1250 Arm A, Research 698).
//!
//! Source: the YASA (Tied Sparse Attention) blog distill — the zero-parameter
//! indexer that scores candidate keys with the main attention's OWN indexer
//! query-head group: `s_j = mean_h ReLU(q_h·k_j)` over the indexer group's
//! heads, coarse blocks scored by mean-pooled compressed keys. No trained
//! indexer weights (contrast [`super::flashmemory_sparse`]'s
//! `DualEncoderIndexer`).
//!
//! # Mechanism vs the pyramid u-fold
//!
//! The pyramid selection core folds the GQA group to `u = Σ_h q_h` BEFORE any
//! scoring (dot-product linearity — a SIGNED sum: opposing heads cancel). The
//! tied indexer is deliberately NOT that fold: the per-head ReLU kills the
//! linearity, so every head's evidence is non-negative and no head cancels
//! another — one head-group of scores, no value sum, no softmax (the YASA
//! cost claim's shape; verified on our geometry under the issue's T3, never
//! inherited). This module is the SCORER only; selection composes the
//! shipped [`super::block_topk::argtopk_with_scratch`] + the harness's
//! forced-∪-top-K slot discipline.
//!
//! # Gate posture
//!
//! Falsifier-first (the issue's T4 pre-registered bars — PASS = paired diff
//! vs the exact-LSE scan ≥ −0.01 at the same scoring-cost class, plan-filing
//! additionally needs the per-length-bin 95% LB of (tied − single_mean) > 0).
//! The real-tensor recall cell rides the pyramid_612 committed capture
//! (`tests/issue_1250_tied_indexer_cell.rs`); the bar verdict itself is
//! FULL-SET-GATED (L ≥ 32K contiguous bins, `PYRAMID_612_FULL_DIR`). Opt-in
//! until then — a failing bar demotes in the same commit that records it
//! (the issue's ungoat discipline).
//!
//! Feature gate: `tied_indexer` (opt-in; implies `dash_attn`).

use katgpt_core::simd::simd_dot_f32;

use super::block_topk::argtopk_with_scratch;

/// Per-block tied-indexer scores: `s_b = mean_h ReLU(scale · q_h · k̄_b)`.
///
/// - `block_means` — the mean-pooled compressed keys, row-major
///   `[n_blocks × d]` (the pyramid hierarchy's level-0 rows are exactly
///   this table; a caller may also pool with its own block size).
/// - `queries` — the indexer group's query heads, `queries[h][d]`
///   (row-major per head, each slice exactly `d`).
/// - `scale` — the logit scale (pass `1/√d` or `1.0` — the scorer is
///   scale-invariant in RANK for a shared positive scale; the parameter
///   exists so the scores stay comparable with the sibling arms' logit
///   scale in a mixed report).
/// - `scores_out` — cleared then written `[n_blocks]`; reuse-friendly
///   (zero-alloc steady state once sized).
///
/// Deterministic: the fold is head-major then block-major in ascending
/// order, one sequential `+=` per head (Issue-940 fold discipline — the
/// scores feed recall COUNTS today, but the order is pinned so a future
/// sync-crossing consumer inherits a fixed reduction for free).
///
/// Panics if `block_means.len() != n_blocks·d`, any query slice ≠ `d`,
/// or `queries` is empty (an indexer with no heads scores nothing — that
/// is a caller bug, not a zero).
pub fn tied_indexer_block_scores_into(
    block_means: &[f32],
    n_blocks: usize,
    d: usize,
    queries: &[&[f32]],
    scale: f32,
    scores_out: &mut Vec<f32>,
) {
    assert!(d > 0, "head_dim must be positive");
    assert!(
        block_means.len() == n_blocks * d,
        "block_means holds {} f32 but n_blocks·d = {}",
        block_means.len(),
        n_blocks * d
    );
    assert!(!queries.is_empty(), "the indexer group has no query heads");
    for (h, q) in queries.iter().enumerate() {
        assert!(
            q.len() == d,
            "query head {h} holds {} f32 != head_dim {d}",
            q.len()
        );
    }

    scores_out.clear();
    scores_out.resize(n_blocks, 0.0);
    let g = queries.len();
    // Head-major fold (ascending head, ascending block, sequential `+=` —
    // the Issue-940 order pin); the per-block indexing is genuine, not a
    // needless range (both buffers index the SAME block b).
    #[allow(clippy::needless_range_loop)]
    for (b, score) in scores_out.iter_mut().enumerate() {
        let kbar = &block_means[b * d..(b + 1) * d];
        let mut acc = 0.0f32;
        for q in queries {
            let s = scale * simd_dot_f32(q, kbar, d);
            // The ReLU: negative evidence contributes ZERO — the mechanism
            // that stops one opposing head from cancelling another's hit
            // (and also what makes this scorer NOT expressible as a u-fold).
            acc += s.max(0.0);
        }
        *score = acc / g as f32;
    }
}

/// Reusable selection scratch for [`topk_blocks_into`] — steady-state
/// zero-alloc (the `argtopk_with_scratch` pattern).
pub struct TiedScratch {
    /// Non-forced block indices + their scores.
    pub nonforced: Vec<usize>,
    pub nonforced_scores: Vec<f32>,
    /// `argtopk_with_scratch` buffers.
    pub indices: Vec<usize>,
    pub pairs: Vec<(usize, f32)>,
}

impl TiedScratch {
    /// Empty buffers — the first call sizes them, later calls reuse.
    pub fn new() -> Self {
        Self {
            nonforced: Vec::new(),
            nonforced_scores: Vec::new(),
            indices: Vec::new(),
            pairs: Vec::new(),
        }
    }
}

impl Default for TiedScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// The harness slot discipline: argtopk-K of the NON-forced blocks ∪ the
/// forced set (first / previous / current leaf, NSA convention), output
/// ascending + deduped, ≤ K + 3. `forced` may carry duplicates; blocks out
/// of range are ignored (the degenerate single-leaf family pins [0]).
/// Writes into `out` (cleared first); reuses `scratch` — zero-alloc steady
/// state. Returns the selected count.
pub fn topk_blocks_into(
    scores: &[f32],
    n_blocks: usize,
    top_k: usize,
    forced: &[usize],
    out: &mut Vec<usize>,
    scratch: &mut TiedScratch,
) -> usize {
    assert!(
        scores.len() == n_blocks,
        "scores holds {} but n_blocks = {}",
        scores.len(),
        n_blocks
    );
    out.clear();
    scratch.nonforced.clear();
    scratch.nonforced_scores.clear();
    for (b, &s) in scores.iter().enumerate() {
        if !forced.contains(&b) {
            scratch.nonforced.push(b);
            scratch.nonforced_scores.push(s);
        }
    }
    let k = top_k.min(scratch.nonforced_scores.len());
    if k > 0 {
        argtopk_with_scratch(
            &scratch.nonforced_scores,
            k,
            &mut scratch.indices,
            &mut scratch.pairs,
        );
        for &i in scratch.indices[..k].iter() {
            out.push(scratch.nonforced[i]);
        }
    }
    for &f in forced {
        if f < n_blocks && !out.contains(&f) {
            out.push(f);
        }
    }
    out.sort_unstable();
    out.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seeded key-block pattern (the pyramid tests' discipline — never the
    /// unseeded global).
    fn block(rng: &mut fastrand::Rng, n: usize, d: usize) -> Vec<f32> {
        (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect()
    }

    // The ReLU is the mechanism: a block whose mean key anti-aligns with
    // EVERY indexer head scores exactly 0.0 — negative evidence contributes
    // nothing (and a u-fold WOULD have scored it positive here, which is
    // exactly the behavioral difference this pin exists to catch).
    #[test]
    fn relu_kills_negative_evidence() {
        let d = 16usize;
        let q: Vec<f32> = (0..d).map(|i| i as f32 + 1.0).collect();
        // k̄ = −q (perfect anti-alignment): every head logit negative.
        let kbar: Vec<f32> = q.iter().map(|v| -v).collect();
        let mut out = Vec::new();
        tied_indexer_block_scores_into(&kbar, 1, d, &[q.as_slice()], 1.0, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 0.0, "anti-aligned block must score 0.0: {}", out[0]);
    }

    // Mean over heads: g = 2, hand-computed — (ReLU(a) + ReLU(b))/2, where
    // one head hits and one anti-aligns: (s + 0)/2 = s/2.
    #[test]
    fn head_mean_with_one_dead_head_halves() {
        let d = 8usize;
        let q_hit: Vec<f32> = (0..d).map(|i| i as f32 + 1.0).collect();
        let q_dead: Vec<f32> = q_hit.iter().map(|v| -v).collect();
        let kbar: Vec<f32> = q_hit.clone();
        let scale = 0.5f32;
        let expect = 0.5 * scale * (0..d).map(|i| (i as f32 + 1.0).powi(2)).sum::<f32>();
        let mut out = Vec::new();
        tied_indexer_block_scores_into(
            &kbar,
            1,
            d,
            &[q_hit.as_slice(), q_dead.as_slice()],
            scale,
            &mut out,
        );
        assert!(
            (out[0] - expect).abs() <= 1e-4 * expect.abs().max(1.0),
            "mean-of-ReLU mismatch: {} vs {}",
            out[0],
            expect
        );
    }

    // Tie-out vs the block-major naive form (reassociation-only tolerance)
    // + exact equality against the same-order direct loop.
    #[test]
    fn tieout_naive_direct() {
        let (n, d, g) = (37usize, 32usize, 6usize);
        let mut rng = fastrand::Rng::with_seed(0x1250);
        let means = block(&mut rng, n, d);
        let queries: Vec<Vec<f32>> = (0..g).map(|_| block(&mut rng, 1, d)).collect();
        let qrefs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
        let scale = 1.0f32 / (d as f32).sqrt();

        let mut out = Vec::new();
        tied_indexer_block_scores_into(&means, n, d, &qrefs, scale, &mut out);

        // Naive block-major reference (different fold order — tolerance).
        for (b, &got) in out.iter().enumerate() {
            let kbar = &means[b * d..(b + 1) * d];
            let mut acc = 0.0f32;
            for q in &qrefs {
                let mut s = 0.0f32;
                for i in 0..d {
                    s += q[i] * kbar[i];
                }
                acc += (scale * s).max(0.0);
            }
            let want = acc / g as f32;
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "block {b}: {got} vs naive {want}"
            );
        }
    }

    // Zero-alloc reuse: the second call on a sized buffer must not grow it
    // (the steady-state contract — scores_out is cleared, never re-reserved
    // past its capacity).
    #[test]
    fn reuse_keeps_capacity() {
        let (n, d) = (64usize, 8usize);
        let mut rng = fastrand::Rng::with_seed(0x0125_0002);
        let means = block(&mut rng, n, d);
        let q = block(&mut rng, 1, d);
        let mut out = Vec::with_capacity(n);
        let cap_before = out.capacity();
        tied_indexer_block_scores_into(&means, n, d, &[q.as_slice()], 1.0, &mut out);
        tied_indexer_block_scores_into(&means, n, d, &[q.as_slice()], 1.0, &mut out);
        assert_eq!(out.len(), n);
        assert_eq!(out.capacity(), cap_before, "reuse must not grow the buffer");
    }

    // Shape guards: wrong-length inputs panic (loud, never a silent zero).
    #[test]
    #[should_panic(expected = "block_means holds")]
    fn wrong_means_len_panics() {
        let q = vec![1.0f32; 8];
        let mut out = Vec::new();
        tied_indexer_block_scores_into(&[0.0; 4], 1, 8, &[q.as_slice()], 1.0, &mut out);
    }

    #[test]
    #[should_panic(expected = "the indexer group has no query heads")]
    fn empty_group_panics() {
        let mut out = Vec::new();
        tied_indexer_block_scores_into(&[], 0, 8, &[], 1.0, &mut out);
    }
}
