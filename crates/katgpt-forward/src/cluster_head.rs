//! Bound-based stage-1 selection for the clustered LM head (Issue 657).
//!
//! # The bound
//!
//! [`clustered_lm_head`](crate::forward::clustered_lm_head) ranks clusters by
//! `⟨h, centroid_c⟩`, which — because the centroid is the mean of the cluster's
//! LM-head rows — is exactly the cluster's **mean logit**, while the question
//! stage 1 must answer is which cluster contains the **max**. Write
//! `w_t = centroid_c + r_t` and let `radius_c = max‖r_t‖`
//! ([`cluster_radii_from_map`](crate::cluster_build::cluster_radii_from_map)).
//! Cauchy–Schwarz gives, for every `t` in cluster `c`:
//!
//! ```text
//! logit[t] = ⟨h, centroid_c⟩ + ⟨h, r_t⟩ ≤ ⟨h, centroid_c⟩ + ‖h‖ · radius_c
//! ```
//!
//! Cost is one `f32` per cluster and one FMA per cluster in stage 1.
//!
//! # Read this before using `TopK` with radii
//!
//! Issue 657 proposed the bound as a better *ranking* function. **Benchmark 658
//! measured that it is not.** `radius_c` varies little across clusters while
//! `‖h‖` is shared, so the added term is near-constant and mostly swamps the
//! signal: on the structured fixture, bound-ranked `TopK` scores *below* the
//! plain mean ranking at every budget (0.41 vs 0.675 at 25% active). Plan 574's
//! recall failure turned out to be a degenerate k-means seeding, fixed in
//! [`cluster_build`](crate::cluster_build) — not a scoring defect.
//!
//! [`ClusterStop::TopK`] with `radii: Some(..)` therefore exists to keep that
//! comparison runnable, not because it is the configuration to ship.
//!
//! # What the bound is actually for
//!
//! [`ClusterStop::Admissible`] — the thing the mean score cannot do at all.
//! Visit clusters in descending bound order and stop as soon as the next bound
//! falls at or below the best **exact** logit already found. Every unvisited
//! cluster provably holds nothing larger, so the argmax **cannot** be missed:
//! recall is 1.0 by construction rather than by measurement, and the budget
//! becomes a reported outcome instead of a silent correctness risk.
//!
//! Its cost is data-dependent, and the spread is the whole story: **7.30%** of
//! the vocabulary touched on a clustered head, **99.99%** on a geometrically
//! flat one.
//!
//! # Do not enable this unconditionally
//!
//! Issue 661 measured the crossover: the clustered path beats the full head
//! only below roughly **21–34% active**, and the loss above it is severe
//! (0.11–0.35×). Enable below ~15% active; use the full head above ~25%.
//!
//! The wall-clock win is also well below the FLOP reduction — 2.21× measured
//! against 11.64× of arithmetic saved — and the shortfall is **locality**, not
//! threading. Cluster members are non-contiguous token IDs, so stage 2 gathers
//! scattered rows at 20.6 GB/s where the full head streams at 108.0 GB/s; the
//! 5.26× ratio accounts for the gap exactly. Wave-parallelism was tried and is
//! a wash (Issue 661 §661a). Issue 666 permutes the rows into cluster order,
//! which is the fix that addresses the actual cost.
//!
//! See `.benchmarks/658_clustered_lm_head_admissible_goat.md`.
//!
//! Both stop rules are deterministic functions of shipped weights —
//! **modelless**, no training. The hot path allocates nothing; all scratch is
//! caller-owned.

use crate::cluster_build::ClusterLayout;
use katgpt_core::simd::simd_dot_f32;
use rayon::prelude::*;

/// Borrowed view of the three load-time clustered-LM-head artifacts.
///
/// Grouped so the hot path takes one borrow instead of three positional
/// slices that are trivially swappable at the call site.
#[derive(Clone, Copy)]
pub struct ClusterHeadView<'a> {
    /// `[num_clusters, n_embd]` per-cluster centroids of the LM-head rows.
    pub classifier: &'a [f32],
    /// `[num_clusters]` residual radii. `None` reproduces the mean-logit
    /// ranking exactly — the A-side of the comparison, not a fallback to be
    /// relied on in production.
    pub radii: Option<&'a [f32]>,
    /// Token IDs per cluster.
    pub map: &'a [Vec<usize>],
}

/// Caller-owned scratch, so the hot path stays allocation-free.
pub struct ClusterScratch<'a> {
    /// At least `map.len()` wide — stage-1 scores.
    pub scores: &'a mut [f32],
    /// Reused `(index, score)` pairs for selection.
    pub indexed: &'a mut Vec<(usize, f32)>,
    /// Receives the clusters actually evaluated, in visit order.
    pub selected: &'a mut Vec<usize>,
    /// Token IDs of the wave currently being evaluated.
    pub gathered: &'a mut Vec<usize>,
    /// Per-token dot products for the current wave, index-aligned with
    /// [`gathered`](Self::gathered). Separate from `logits` so the parallel
    /// write target is contiguous and provably disjoint.
    pub dots: &'a mut Vec<f32>,
}

/// Tokens a wave must reach before rayon is worth its ~5 µs of overhead.
///
/// Mirrors `simd_matmul_rows_parallel`'s own 512-row cutoff, so a wave that
/// would have been serial inside the full head stays serial here too.
const PARALLEL_MIN_TOKENS: usize = 512;

/// When to stop admitting clusters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterStop {
    /// Fixed budget: take the `k` highest-scoring clusters. Evaluated as one
    /// wave, so it parallelizes unconditionally.
    TopK(usize),
    /// Exact: stop when the bound can no longer beat the best exact logit
    /// found. Recall is 1.0 by construction.
    ///
    /// `wave` clusters are evaluated per round before the stop condition is
    /// re-checked. The check is on the *leading* bound of each wave, so a wave
    /// may compute clusters it did not strictly need — extra work, never a
    /// missed argmax. Exactness is independent of `wave`.
    ///
    /// - `wave: 1` is the sequential reference: minimum work, no parallelism.
    /// - Larger waves trade a few redundant clusters for rayon width. The
    ///   highest-bound cluster almost always holds the argmax, so `best_exact`
    ///   is near-final after the first wave and the second check prunes hard.
    ///
    /// `wave: 0` is treated as 1.
    Admissible { wave: usize },
}

/// What the selection actually cost, so callers can report the operating point
/// instead of assuming it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClusterCost {
    /// Clusters whose exact logits were computed.
    pub clusters: usize,
    /// Tokens whose exact logits were computed.
    pub tokens: usize,
}

/// Exact logits for one cluster; returns the largest logit written.
///
/// Shared with [`clustered_lm_head`](crate::forward::clustered_lm_head) so the
/// two stage-2 loops cannot drift apart.
#[inline(always)]
pub(crate) fn fill_cluster_exact(
    logits: &mut [f32],
    hidden: &[f32],
    lm_head: &[f32],
    tokens: &[usize],
    vocab_size: usize,
    n_embd: usize,
) -> f32 {
    let mut best = f32::NEG_INFINITY;
    for &token_idx in tokens {
        if token_idx >= vocab_size {
            continue;
        }
        let row_off = token_idx * n_embd;
        let dot = simd_dot_f32(&lm_head[row_off..row_off + n_embd], &hidden[..n_embd], n_embd);
        // SAFETY: `token_idx < vocab_size` and `logits` is `vocab_size` wide.
        unsafe {
            *logits.get_unchecked_mut(token_idx) = dot;
        }
        best = best.max(dot);
    }
    best
}

/// Stage 1: score every cluster and sort `scratch.indexed` by score descending.
///
/// Shared by the scattered and packed stage-2 implementations so the ranking —
/// the part that decides *which* clusters are visited, and therefore whether
/// the result is exact — cannot drift between them.
///
/// A full sort (rather than a partial select) is required for `Admissible`, and
/// at cluster counts of 256–2048 against a stage-2 cost of
/// `active_tokens × n_embd` FMAs it is not the bottleneck.
fn rank_clusters(
    hidden: &[f32],
    classifier: &[f32],
    radii: Option<&[f32]>,
    num_clusters: usize,
    n_embd: usize,
    scratch: &mut ClusterScratch<'_>,
) {
    let scores = &mut scratch.scores[..num_clusters];

    // ‖h‖ is shared by every cluster's bound — one sqrt per call, not per
    // cluster.
    let h_norm = match radii {
        None => 0.0,
        Some(_) => simd_dot_f32(&hidden[..n_embd], &hidden[..n_embd], n_embd).max(0.0).sqrt(),
    };

    for (c, score) in scores.iter_mut().enumerate() {
        let row_off = c * n_embd;
        let mean = simd_dot_f32(&classifier[row_off..row_off + n_embd], &hidden[..n_embd], n_embd);
        *score = match radii {
            None => mean,
            Some(radii) => mean + h_norm * radii[c],
        };
    }

    scratch.indexed.resize(num_clusters, (0, 0.0));
    for (i, &s) in scores.iter().enumerate() {
        scratch.indexed[i] = (i, s);
    }
    scratch.indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
}

/// `(limit, wave, stop_checked)` for a stop rule.
///
/// `TopK` is one wave over its whole budget with no stop check; `Admissible`
/// walks the full list in `wave`-sized rounds, re-checking between them. One
/// loop shape serves both.
fn wave_plan(stop: ClusterStop, num_clusters: usize) -> (usize, usize, bool) {
    match stop {
        ClusterStop::TopK(k) => {
            let k = k.min(num_clusters);
            (k, k.max(1), false)
        }
        ClusterStop::Admissible { wave } => (num_clusters, wave.max(1), true),
    }
}

/// Two-stage clustered LM head with bound-ranked stage-1 selection.
///
/// **Scattered layout** — cluster members are token IDs into the original
/// `lm_head`, so stage 2 gathers non-contiguous rows. See
/// [`clustered_lm_head_packed`] for the cluster-contiguous variant, which is
/// the one to use when the memory cost is acceptable.
///
/// See the module docs for the bound and the two stop rules. Unvisited tokens
/// are left at `-inf`, matching
/// [`clustered_lm_head`](crate::forward::clustered_lm_head).
// 8 args: 3 tensors + 2 shapes + the artifact view, stop rule and scratch. The
// artifacts and scratch are already grouped into structs; collapsing further
// would hide the shapes the caller must keep consistent.
#[allow(clippy::too_many_arguments)]
pub fn clustered_lm_head_bounded(
    logits: &mut [f32],
    hidden: &[f32],
    lm_head: &[f32],
    head: ClusterHeadView<'_>,
    vocab_size: usize,
    n_embd: usize,
    stop: ClusterStop,
    mut scratch: ClusterScratch<'_>,
) -> ClusterCost {
    let num_clusters = head.map.len();
    rank_clusters(hidden, head.classifier, head.radii, num_clusters, n_embd, &mut scratch);

    logits.fill(f32::NEG_INFINITY);
    scratch.selected.clear();

    let (limit, wave, checked) = wave_plan(stop, num_clusters);

    let mut cost = ClusterCost::default();
    let mut best_exact = f32::NEG_INFINITY;
    let mut visited = 0usize;

    while visited < limit {
        // Bounds are non-increasing over `indexed`, so if the leading bound of
        // this wave cannot beat an exact logit we already hold, nothing from
        // here on can either. This is what makes the mode exact rather than
        // merely cheap.
        if checked && scratch.indexed[visited].1 <= best_exact {
            break;
        }
        let end = (visited + wave).min(limit);

        scratch.gathered.clear();
        for &(cluster_idx, _) in &scratch.indexed[visited..end] {
            let tokens = &head.map[cluster_idx];
            scratch.gathered.extend(tokens.iter().copied().filter(|&t| t < vocab_size));
            scratch.selected.push(cluster_idx);
        }
        cost.clusters += end - visited;
        cost.tokens += scratch.gathered.len();

        scratch.dots.resize(scratch.gathered.len(), 0.0);
        let row = |t: usize| {
            let off = t * n_embd;
            simd_dot_f32(&lm_head[off..off + n_embd], &hidden[..n_embd], n_embd)
        };
        if scratch.gathered.len() >= PARALLEL_MIN_TOKENS { scratch
                .gathered
                .par_iter()
                .zip(scratch.dots.par_iter_mut())
                .for_each(|(&t, d)| *d = row(t)); } else {
                for (&t, d) in scratch.gathered.iter().zip(scratch.dots.iter_mut()) {
                    *d = row(t);
                }
            }

        // Scatter is serial and memory-bound only — the dots are already
        // computed. Token IDs are unique across clusters, so no write races
        // even though the writes are scattered.
        for (&t, &d) in scratch.gathered.iter().zip(scratch.dots.iter()) {
            // SAFETY: `t < vocab_size` (filtered above) and `logits` is
            // `vocab_size` wide.
            unsafe {
                *logits.get_unchecked_mut(t) = d;
            }
            best_exact = best_exact.max(d);
        }

        visited = end;
    }

    cost
}

/// Borrowed view of the **packed** clustered-LM-head artifacts (Issue 666).
///
/// The classifier and radii are unchanged — only the token→row indirection is
/// replaced by [`ClusterLayout`]'s contiguous spans.
#[derive(Clone, Copy)]
pub struct PackedHeadView<'a> {
    /// `[num_clusters, n_embd]` per-cluster centroids.
    pub classifier: &'a [f32],
    /// `[num_clusters]` residual radii. Must be built from the **same** map the
    /// layout was, or the bound stops being admissible.
    pub radii: Option<&'a [f32]>,
    /// Cluster-contiguous rows.
    pub layout: &'a ClusterLayout,
}

/// Clustered LM head over a cluster-contiguous layout (Issue 666).
///
/// Identical selection to [`clustered_lm_head_bounded`] — the two share
/// [`rank_clusters`] and [`wave_plan`] — but stage 2 reads each cluster as one
/// contiguous span of [`ClusterLayout::permuted`] and scatters the results back
/// through [`ClusterLayout::token_of_row`], instead of gathering scattered rows.
///
/// Every logit this computes is **bit-identical** to the scattered path's: the
/// same `simd_dot_f32` over the same row against the same hidden state, only
/// read from a different address. That is asserted, not assumed.
///
/// Stage 1 costs nothing extra; the whole change is the memory access pattern.
#[allow(clippy::too_many_arguments)]
pub fn clustered_lm_head_packed(
    logits: &mut [f32],
    hidden: &[f32],
    head: PackedHeadView<'_>,
    vocab_size: usize,
    n_embd: usize,
    stop: ClusterStop,
    mut scratch: ClusterScratch<'_>,
) -> ClusterCost {
    let layout = head.layout;
    let num_clusters = layout.offsets.len();
    rank_clusters(hidden, head.classifier, head.radii, num_clusters, n_embd, &mut scratch);

    logits.fill(f32::NEG_INFINITY);
    scratch.selected.clear();

    let (limit, wave, checked) = wave_plan(stop, num_clusters);

    let mut cost = ClusterCost::default();
    let mut best_exact = f32::NEG_INFINITY;
    let mut visited = 0usize;

    while visited < limit {
        // Same admissibility argument as the scattered path: bounds are
        // non-increasing over `indexed`, so once the leading bound of a wave
        // cannot beat an exact logit already held, nothing later can either.
        if checked && scratch.indexed[visited].1 <= best_exact {
            break;
        }
        let end = (visited + wave).min(limit);

        // Total width first, so `dots` is sized once per wave rather than
        // grown per cluster.
        let total: usize =
            scratch.indexed[visited..end].iter().map(|&(c, _)| layout.offsets[c].1).sum();
        scratch.dots.resize(total, 0.0);
        cost.clusters += end - visited;
        cost.tokens += total;

        let mut written = 0usize;
        for &(cluster_idx, _) in &scratch.indexed[visited..end] {
            let (start, len) = layout.offsets[cluster_idx];
            scratch.selected.push(cluster_idx);
            if len == 0 {
                continue;
            }
            // The payoff: one contiguous `[len, n_embd]` block, which is the
            // access pattern `standard_lm_head` already sustains 108 GB/s on.
            // `simd_matmul_rows_parallel` self-selects rayon above 512 rows, so
            // a cluster large enough to be worth threading gets it for free and
            // a small one is not charged the scheduling overhead.
            katgpt_core::simd::simd_matmul_rows_parallel(
                &mut scratch.dots[written..written + len],
                &layout.permuted[start * n_embd..(start + len) * n_embd],
                &hidden[..n_embd],
                len,
                n_embd,
            );
            written += len;
        }

        // Scatter back to token order. Memory-bound only — the dots are already
        // computed — and token IDs are unique across clusters, so the scattered
        // writes cannot collide.
        let mut at = 0usize;
        for &(cluster_idx, _) in &scratch.indexed[visited..end] {
            let (start, len) = layout.offsets[cluster_idx];
            for i in 0..len {
                let t = layout.token_of_row[start + i];
                let d = scratch.dots[at + i];
                // `cluster_layout_from_map` already filters out-of-range tokens,
                // so this holds for every row the layout contains.
                if t < vocab_size {
                    // SAFETY: `t < vocab_size` and `logits` is `vocab_size` wide.
                    unsafe { *logits.get_unchecked_mut(t) = d };
                }
                best_exact = best_exact.max(d);
            }
            at += len;
        }

        visited = end;
    }

    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster_build::{
        LayoutRefusal, TiedPolicy, cluster_classifier_from_map, cluster_layout_from_map,
        cluster_map_from_embeddings, cluster_radii_from_map,
    };
    use crate::forward::standard_lm_head;

    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            // float_order: a NaN score must never win the argmax.
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map_or(0, |(i, _)| i)
    }

    /// Planted-group LM head, the same geometry Benchmark 657 uses.
    fn planted(vocab: usize, n_embd: usize, groups: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg(seed);
        let mut centres = vec![0.0f32; groups * n_embd];
        for slot in centres.iter_mut() {
            *slot = rng.next_f32();
        }
        let mut w = vec![0.0f32; vocab * n_embd];
        for t in 0..vocab {
            let g = t % groups;
            for j in 0..n_embd {
                w[t * n_embd + j] = centres[g * n_embd + j] + 0.05 * rng.next_f32();
            }
        }
        w
    }

    fn artifacts(
        w: &[f32],
        vocab: usize,
        n_embd: usize,
        cluster_size: usize,
    ) -> (Vec<Vec<usize>>, Vec<f32>, Vec<f32>) {
        let map = cluster_map_from_embeddings(w, vocab, n_embd, cluster_size);
        let cls = cluster_classifier_from_map(w, &map, n_embd);
        let radii = cluster_radii_from_map(w, &map, &cls, n_embd);
        (map, cls, radii)
    }

    /// The load-bearing claim: `Admissible` never misses the argmax.
    #[test]
    fn admissible_mode_never_misses_the_argmax() {
        let (vocab, n_embd) = (1024, 64);
        let w = planted(vocab, n_embd, 32, 0xA11CE);
        let (map, cls, radii) = artifacts(&w, vocab, n_embd, 32);

        let mut rng = Lcg(0xC0FFEE);
        let mut logits = vec![0.0f32; vocab];
        let mut truth = vec![0.0f32; vocab];
        let mut scores = vec![0.0f32; map.len()];
        let (mut indexed, mut selected) = (Vec::new(), Vec::new());
        let (mut gathered, mut dots) = (Vec::new(), Vec::new());

        for _ in 0..64 {
            let hidden: Vec<f32> = (0..n_embd).map(|_| rng.next_f32()).collect();
            standard_lm_head(&mut truth, &hidden, &w, vocab, n_embd);

            clustered_lm_head_bounded(
                &mut logits,
                &hidden,
                &w,
                ClusterHeadView { classifier: &cls, radii: Some(&radii), map: &map },
                vocab,
                n_embd,
                ClusterStop::Admissible { wave: 1 },
                ClusterScratch {
                    scores: &mut scores,
                    indexed: &mut indexed,
                    selected: &mut selected,
                    gathered: &mut gathered,
                    dots: &mut dots,
                },
            );
            assert_eq!(
                argmax(&logits),
                argmax(&truth),
                "admissible selection must be exact"
            );
        }
    }

    /// Wave size is a work/parallelism knob, never a correctness knob: every
    /// wave must return the same argmax as `wave: 1`, and larger waves may only
    /// touch *more* tokens, never fewer.
    #[test]
    fn wave_size_changes_cost_but_never_the_answer() {
        let (vocab, n_embd) = (2048, 64);
        let w = planted(vocab, n_embd, 64, 0x5EED);
        let (map, cls, radii) = artifacts(&w, vocab, n_embd, 32);

        let mut rng = Lcg(0x1234);
        let mut truth = vec![0.0f32; vocab];
        let mut logits = vec![0.0f32; vocab];
        let mut scores = vec![0.0f32; map.len()];
        let (mut indexed, mut selected) = (Vec::new(), Vec::new());
        let (mut gathered, mut dots) = (Vec::new(), Vec::new());

        for _ in 0..24 {
            let hidden: Vec<f32> = (0..n_embd).map(|_| rng.next_f32()).collect();
            standard_lm_head(&mut truth, &hidden, &w, vocab, n_embd);
            let want = argmax(&truth);

            let mut baseline_tokens = 0usize;
            for (i, wave) in [1usize, 2, 8, 64, 4096].into_iter().enumerate() {
                let cost = clustered_lm_head_bounded(
                    &mut logits,
                    &hidden,
                    &w,
                    ClusterHeadView { classifier: &cls, radii: Some(&radii), map: &map },
                    vocab,
                    n_embd,
                    ClusterStop::Admissible { wave },
                    ClusterScratch {
                        scores: &mut scores,
                        indexed: &mut indexed,
                        selected: &mut selected,
                        gathered: &mut gathered,
                        dots: &mut dots,
                    },
                );
                assert_eq!(argmax(&logits), want, "wave {wave} changed the argmax");
                match i {
                    0 => baseline_tokens = cost.tokens,
                    _ => assert!(
                        cost.tokens >= baseline_tokens,
                        "wave {wave} touched {} tokens, fewer than wave 1's {baseline_tokens} — \
                         a larger wave can only over-compute, never skip work",
                        cost.tokens
                    ),
                }
            }
        }
    }

    /// The parallel path must be bit-identical to the serial one. It only
    /// engages above [`PARALLEL_MIN_TOKENS`], so this fixture is sized to
    /// straddle that cutoff.
    #[test]
    fn parallel_wave_matches_serial_wave_bit_for_bit() {
        let (vocab, n_embd) = (4096, 64);
        let w = planted(vocab, n_embd, 32, 0xBEEF);
        // 256-token clusters: wave 1 stays under the 512-token cutoff (serial),
        // wave 8 clears it (parallel).
        let (map, cls, radii) = artifacts(&w, vocab, n_embd, 256);

        let hidden: Vec<f32> = (0..n_embd).map(|j| 0.03 * (j as f32) - 0.9).collect();
        let mut scores = vec![0.0f32; map.len()];
        let (mut indexed, mut selected) = (Vec::new(), Vec::new());
        let (mut gathered, mut dots) = (Vec::new(), Vec::new());

        let run = |wave: usize,
                       scores: &mut Vec<f32>,
                       indexed: &mut Vec<(usize, f32)>,
                       selected: &mut Vec<usize>,
                       gathered: &mut Vec<usize>,
                       dots: &mut Vec<f32>| {
            let mut logits = vec![0.0f32; vocab];
            clustered_lm_head_bounded(
                &mut logits,
                &hidden,
                &w,
                ClusterHeadView { classifier: &cls, radii: Some(&radii), map: &map },
                vocab,
                n_embd,
                ClusterStop::TopK(wave),
                ClusterScratch { scores, indexed, selected, gathered, dots },
            );
            logits
        };

        let serial = run(1, &mut scores, &mut indexed, &mut selected, &mut gathered, &mut dots);
        let parallel = run(8, &mut scores, &mut indexed, &mut selected, &mut gathered, &mut dots);

        // The parallel run is a superset: every finite logit the serial run
        // produced must be present and identical.
        let mut shared = 0usize;
        for (t, &s) in serial.iter().enumerate() {
            if s.is_finite() {
                assert_eq!(parallel[t], s, "logit[{t}] differs serial vs parallel");
                shared += 1;
            }
        }
        assert!(shared >= PARALLEL_MIN_TOKENS, "fixture must cross the parallel cutoff");
    }

    /// **The Issue 666 obligation.** The packed layout may only change *where*
    /// rows are read from, never what is computed. Every logit must match the
    /// scattered path bit-for-bit, and the same clusters must be visited.
    ///
    /// Issue 657's fix passed every plausibility check and was still attacking
    /// the wrong defect; the only thing that caught it was running the
    /// alternative side by side. Same discipline here.
    #[test]
    fn packed_layout_is_bit_identical_to_scattered() {
        let (vocab, n_embd) = (2048, 64);
        let w = planted(vocab, n_embd, 64, 0x0FFE);
        let (map, cls, radii) = artifacts(&w, vocab, n_embd, 32);
        let layout =
            cluster_layout_from_map(&w, &map, vocab, n_embd, TiedPolicy::Accept).expect("layout");

        let mut rng = Lcg(0xABCD);
        let mut scores = vec![0.0f32; map.len()];
        let (mut indexed, mut selected) = (Vec::new(), Vec::new());
        let (mut gathered, mut dots) = (Vec::new(), Vec::new());

        for _ in 0..24 {
            let hidden: Vec<f32> = (0..n_embd).map(|_| rng.next_f32()).collect();

            for stop in [ClusterStop::TopK(5), ClusterStop::Admissible { wave: 4 }] {
                let mut scattered = vec![0.0f32; vocab];
                let cost_s = clustered_lm_head_bounded(
                    &mut scattered,
                    &hidden,
                    &w,
                    ClusterHeadView { classifier: &cls, radii: Some(&radii), map: &map },
                    vocab,
                    n_embd,
                    stop,
                    ClusterScratch {
                        scores: &mut scores,
                        indexed: &mut indexed,
                        selected: &mut selected,
                        gathered: &mut gathered,
                        dots: &mut dots,
                    },
                );
                let visited_s = selected.clone();

                let mut packed = vec![0.0f32; vocab];
                let cost_p = clustered_lm_head_packed(
                    &mut packed,
                    &hidden,
                    PackedHeadView { classifier: &cls, radii: Some(&radii), layout: &layout },
                    vocab,
                    n_embd,
                    stop,
                    ClusterScratch {
                        scores: &mut scores,
                        indexed: &mut indexed,
                        selected: &mut selected,
                        gathered: &mut gathered,
                        dots: &mut dots,
                    },
                );

                assert_eq!(packed, scattered, "packed logits differ for {stop:?}");
                assert_eq!(cost_p, cost_s, "packed cost differs for {stop:?}");
                assert_eq!(*selected, visited_s, "packed visited different clusters");
            }
        }
    }

    /// The permutation must not silently drop or duplicate a token.
    #[test]
    fn layout_covers_every_token_exactly_once() {
        let (vocab, n_embd) = (512, 32);
        let w = planted(vocab, n_embd, 16, 0x1111);
        let map = cluster_map_from_embeddings(&w, vocab, n_embd, 32);
        let layout =
            cluster_layout_from_map(&w, &map, vocab, n_embd, TiedPolicy::Accept).expect("layout");

        let mut seen = vec![0usize; vocab];
        for &t in &layout.token_of_row {
            seen[t] += 1;
        }
        assert!(seen.iter().all(|&c| c == 1), "every token must occupy exactly one row");

        // Spans must tile the row space with no gap or overlap.
        let mut next = 0usize;
        for &(start, len) in &layout.offsets {
            assert_eq!(start, next, "cluster spans must be contiguous and in order");
            next += len;
        }
        assert_eq!(next, vocab, "spans must cover every row");

        // Each permuted row must be its token's original row.
        for (r, &t) in layout.token_of_row.iter().enumerate() {
            assert_eq!(
                &layout.permuted[r * n_embd..(r + 1) * n_embd],
                &w[t * n_embd..(t + 1) * n_embd],
                "row {r} is not token {t}'s LM-head row"
            );
        }
    }

    /// Tied embeddings must be refused by default: the permuted copy cannot
    /// alias `wte`, so it is a second allocation of the largest tensor.
    #[test]
    fn tied_embeddings_are_refused_unless_explicitly_accepted() {
        let (vocab, n_embd) = (256, 16);
        let w = planted(vocab, n_embd, 8, 0x2222);
        let map = cluster_map_from_embeddings(&w, vocab, n_embd, 32);

        // Tied: `lm_head` and `wte` are the same storage.
        let refused = cluster_layout_from_map(&w, &map, vocab, n_embd, TiedPolicy::Refuse { wte: &w });
        match refused {
            Err(LayoutRefusal::TiedEmbeddings { extra_bytes }) => assert_eq!(
                extra_bytes,
                vocab * n_embd * size_of::<f32>(),
                "refusal must report the real cost so the caller can decide"
            ),
            other => panic!("tied weights must be refused, got {other:?}"),
        }

        // Explicit opt-in still builds.
        let accepted = cluster_layout_from_map(&w, &map, vocab, n_embd, TiedPolicy::Accept);
        assert!(accepted.is_ok(), "Accept must override the tie check");

        // Untied: a genuinely separate buffer is allowed through.
        let wte = planted(vocab, n_embd, 8, 0x3333);
        let untied =
            cluster_layout_from_map(&w, &map, vocab, n_embd, TiedPolicy::Refuse { wte: &wte });
        assert!(untied.is_ok(), "untied weights must not be refused");
    }

    /// A logit that *is* computed must equal the full head's, bit for bit —
    /// pruning may only withhold values, never perturb them.
    #[test]
    fn computed_logits_are_bit_identical_to_the_full_head() {
        let (vocab, n_embd) = (512, 32);
        let w = planted(vocab, n_embd, 16, 0xBEEF);
        let (map, cls, radii) = artifacts(&w, vocab, n_embd, 32);
        let hidden: Vec<f32> = (0..n_embd).map(|j| 0.1 * (j as f32) - 0.5).collect();

        let mut truth = vec![0.0f32; vocab];
        standard_lm_head(&mut truth, &hidden, &w, vocab, n_embd);

        let mut logits = vec![0.0f32; vocab];
        let mut scores = vec![0.0f32; map.len()];
        let (mut indexed, mut selected) = (Vec::new(), Vec::new());
        let (mut gathered, mut dots) = (Vec::new(), Vec::new());
        clustered_lm_head_bounded(
            &mut logits,
            &hidden,
            &w,
            ClusterHeadView { classifier: &cls, radii: Some(&radii), map: &map },
            vocab,
            n_embd,
            ClusterStop::TopK(4),
            ClusterScratch {
                scores: &mut scores,
                indexed: &mut indexed,
                selected: &mut selected,
                gathered: &mut gathered,
                dots: &mut dots,
            },
        );

        let mut checked = 0usize;
        for (t, &got) in logits.iter().enumerate() {
            if got.is_finite() {
                assert_eq!(got, truth[t], "logit[{t}] differs from the full head");
                checked += 1;
            }
        }
        assert!(checked > 0, "top-4 clusters must produce some finite logits");
    }

    /// `radii: None` must reproduce the shipped mean-logit ranking, so the
    /// bench's A/B comparison is a change of one term and nothing else.
    #[test]
    fn radii_none_matches_the_mean_logit_ranking() {
        let (vocab, n_embd) = (512, 32);
        let w = planted(vocab, n_embd, 16, 0xD00D);
        let (map, cls, _) = artifacts(&w, vocab, n_embd, 32);
        let hidden: Vec<f32> = (0..n_embd).map(|j| 0.05 * (j as f32) - 0.8).collect();

        let mut want = vec![0.0f32; vocab];
        let mut scores = vec![0.0f32; map.len()];
        let (mut idx_buf, mut out_buf) = (Vec::new(), Vec::new());
        crate::forward::clustered_lm_head(
            &mut want, &hidden, &w, &cls, &map, vocab, n_embd, 3, &mut scores, &mut idx_buf,
            &mut out_buf,
        );

        let mut got = vec![0.0f32; vocab];
        let (mut indexed, mut selected) = (Vec::new(), Vec::new());
        let (mut gathered, mut dots) = (Vec::new(), Vec::new());
        clustered_lm_head_bounded(
            &mut got,
            &hidden,
            &w,
            ClusterHeadView { classifier: &cls, radii: None, map: &map },
            vocab,
            n_embd,
            ClusterStop::TopK(3),
            ClusterScratch {
                scores: &mut scores,
                indexed: &mut indexed,
                selected: &mut selected,
                gathered: &mut gathered,
                dots: &mut dots,
            },
        );
        assert_eq!(got, want, "radii=None must be the mean-logit path exactly");
    }

    /// Radii are a genuine upper bound: no member logit may exceed its
    /// cluster's bound. If this fails, `Admissible` is not exact.
    #[test]
    fn radius_bound_dominates_every_member_logit() {
        let (vocab, n_embd) = (512, 32);
        let w = planted(vocab, n_embd, 16, 0xFACE);
        let (map, cls, radii) = artifacts(&w, vocab, n_embd, 32);
        let hidden: Vec<f32> = (0..n_embd).map(|j| 0.2 * (j as f32) - 3.0).collect();
        let h_norm = simd_dot_f32(&hidden, &hidden, n_embd).sqrt();

        for (c, tokens) in map.iter().enumerate() {
            let mean = simd_dot_f32(&cls[c * n_embd..(c + 1) * n_embd], &hidden, n_embd);
            let bound = mean + h_norm * radii[c];
            for &t in tokens {
                let logit = simd_dot_f32(&w[t * n_embd..(t + 1) * n_embd], &hidden, n_embd);
                // f32 slack: the bound is computed in the same precision as the
                // logits, so allow one ulp-ish of accumulated rounding.
                assert!(
                    logit <= bound + 1e-3 * bound.abs().max(1.0),
                    "cluster {c} token {t}: logit {logit} exceeds bound {bound}"
                );
            }
        }
    }
}
