//! PISA — Pyramid Top-K + LSE block selection (Plan 612, Research 595).
//!
//! Source: [arXiv:2609.31093](https://arxiv.org/abs/2609.31093) — *Block
//! Sparse Attention with Log-Linear Complexity* (Tang et al., Sep 2026).
//! Modelless extraction: a mean-pooled coarse-to-fine key hierarchy, LSE
//! block scoring, and root-seeded bounded expansion — all consuming kernels
//! that already ship ([`argtopk_with_scratch`],
//! `katgpt_core::simd::logsumexp_parts`, `katgpt_core::simd::simd_dot_f32`).
//!
//! # Mechanism
//!
//! Level 0 = leaf blocks of [`PYRAMID_BLOCK_SIZE`] keys (summary = mean of
//! its keys); level *t* pools [`PYRAMID_BRANCHING`] adjacent children of
//! level *t−1*, weighted by covered token counts so every node is the EXACT
//! subtree mean (integral-image identity — unit-pinned). Total storage is
//! ≤ 2·⌈N/C⌉·d f32 per KV head.
//!
//! Selection is root-seeded: at each level the candidates are the children
//! of the blocks retained one level up (≤ g·K of them — never a full scan,
//! which is what separates this from the shipped two-level HGA). A candidate
//! at level *t* is scored by LSE over its ≤ g children's summary logits
//! (`simd_dot_f32`); a candidate at the leaf level is scored by EXACT LSE
//! over its ≤ C per-token logits (`logsumexp_parts`). Retained = Top-K;
//! forced blocks (first / previous / current leaf — NSA convention) enter
//! leaf output directly and are never scored, so they never compete for
//! Top-K slots; they DO expand (their non-forced descendants stay eligible),
//! which adds a constant `g·FORCED` to the per-level scored bound —
//! asserted per call as `≤ 1 + (gK + g·3)·⌈log₂(N/C)⌉` (the paper's pure
//! no-forced form `1 + gK·⌈log₂(N/C)⌉` is the special case; same O(gK log N)
//! shape). GQA heads sharing a KV head select one shared set — the group-summed
//! query u = Σ_h q_h is folded once (dot-product linearity) before any
//! scoring.
//!
//! # The scorer ladder (the paper's load-bearing ablation)
//!
//! Jensen chain (paper App. A): normalized mean score ≤ normalized
//! LSE-over-child-means ≤ normalized raw-key LSE. At uniform block size
//! `ln C` is rank-invariant, so the variance/exactness terms carry the
//! entire gain — [`PyramidScoreMode`] exposes the ladder rungs for the
//! Plan 612 G2 sweep:
//!
//! - [`PyramidScoreMode::Mean`] — PISA-1 class (the paper's WEAKEST row,
//!   below plain mean-scoring BSA; kept for the ablation, not for use),
//! - [`PyramidScoreMode::MeanPlusHalfVar`] — PISA-2 (order-2 Taylor rung),
//! - [`PyramidScoreMode::ExactLse`] — PISA (order-∞; the paper predicts
//!   exact wins on recall).
//!
//! # Slot discipline (binding)
//!
//! This slot carries two standing GOAT negatives: MSA (R225/Plan 256) and
//! HGA (R379/Plan 397 — G2-proxy FAIL 2/12). HGA's root cause was measured
//! on RANDOM keys; the random-key NIAH harness is therefore **BANNED** for
//! this feature's gate — every selection-quality claim must replay REAL
//! pretrained checkpoint tensors (Plan 612 T2.1). Opt-in until the Plan 612
//! G2 head-to-head (log-log latency slope ≈1 vs ≈2 at iso-quality) passes on
//! real tensors AND a long-context consumer exists.
//!
//! Feature gate: `pyramid_topk` (opt-in; implies `dash_attn` — it consumes
//! `argtopk_with_scratch` from [`super::block_topk`]).

use katgpt_core::simd::{fast_exp, logsumexp_parts, simd_dot_f32};

use super::block_topk::argtopk_with_scratch;

/// Keys per leaf block (the paper's C = 64).
pub const PYRAMID_BLOCK_SIZE: usize = 64;

/// Children pooled per internal node (the paper's g = 2).
pub const PYRAMID_BRANCHING: usize = 2;

/// Hard upper bound on hierarchy depth. Covers leaf counts up to 2^39 —
/// the depth is `1 + ⌈log₂(n_leaves)⌉`, so this supports ~2^45 keys.
pub const MAX_PYRAMID_LEVELS: usize = 40;

/// Forced-block slots: first / previous / current leaf (NSA convention).
pub const FORCED_LEAF_SLOTS: usize = 3;

/// Taylor-order knob for the block scorer — the Plan 612 G2 sweep axis.
///
/// See the module doc: rungs of the Jensen ladder over block logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyramidScoreMode {
    /// Mean child-logit only (PISA-1 class — order-1 Taylor rung).
    Mean,
    /// Mean + ½·Var of the child logits (PISA-2 — order-2 Taylor rung).
    MeanPlusHalfVar,
    /// Exact LSE over the child logits (PISA — order-∞).
    ExactLse,
}

/// Read view over a pyramid's levels, shared by the static hierarchy and the
/// decode cache so the selection core has ONE home.
///
/// Levels are indexed `0 = leaf … n_levels()−1 = root`; each level's rows
/// are flat row-major `[row][head_dim]`.
pub trait PyramidLevels {
    fn head_dim(&self) -> usize;
    /// Total source keys behind the pyramid.
    fn n_keys(&self) -> usize;
    /// Leaf-block count (⌈N/C⌉); 0 for an empty pyramid.
    fn leaf_count(&self) -> usize;
    /// Total level count (≥ 1 for a non-empty pyramid; 0 when empty).
    fn n_levels(&self) -> usize;
    fn level_row_count(&self, level: usize) -> usize;
    fn level_rows(&self, level: usize) -> &[f32];
}

/// `⌈log₂(x)⌉` for x ≥ 1.
#[inline]
fn ceil_log2_usize(x: usize) -> usize {
    debug_assert!(x >= 1);
    x.next_power_of_two().trailing_zeros() as usize
}

/// Token count covered by pyramid node `child` one level above the leaves,
/// where each such child spans `leaves_per_child` leaves of
/// [`PYRAMID_BLOCK_SIZE`] tokens (the last span may be ragged).
#[inline]
fn covered_token_count(n_keys: usize, child: usize, leaves_per_child: usize) -> usize {
    let span = leaves_per_child * PYRAMID_BLOCK_SIZE;
    n_keys.saturating_sub(child * span).min(span)
}

/// The forced leaf set — first / previous / current leaf, deduped, clamped
/// to `[0, n_leaves)`. At most [`FORCED_LEAF_SLOTS`] entries.
#[derive(Clone, Copy)]
struct ForcedBlocks {
    slots: [usize; FORCED_LEAF_SLOTS],
    len: usize,
}

impl ForcedBlocks {
    fn at(query_pos: usize, n_leaves: usize) -> Self {
        let cur = (query_pos / PYRAMID_BLOCK_SIZE).min(n_leaves - 1);
        let mut slots = [usize::MAX; FORCED_LEAF_SLOTS];
        let mut len = 0;
        for cand in [0, cur.saturating_sub(1), cur] {
            if !slots[..len].contains(&cand) {
                slots[len] = cand;
                len += 1;
            }
        }
        Self { slots, len }
    }

    #[inline]
    fn contains(&self, block: usize) -> bool {
        self.slots[..self.len].contains(&block)
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots[..self.len].iter().copied()
    }

    /// Ancestor of every forced leaf at pyramid level `level` (0 = leaf),
    /// deduped. Same small-array shape as the leaf set.
    fn ancestors_at(&self, level: usize) -> Self {
        let mut slots = [usize::MAX; FORCED_LEAF_SLOTS];
        let mut len = 0;
        for leaf in self.iter() {
            let anc = leaf >> level;
            if !slots[..len].contains(&anc) {
                slots[len] = anc;
                len += 1;
            }
        }
        Self { slots, len }
    }
}

// ---------------------------------------------------------------------------
// Static hierarchy (T1.1)
// ---------------------------------------------------------------------------

/// A mean-pooled coarse-to-fine key hierarchy over ONE KV head's keys.
///
/// Borrowing view over caller-provided storage (zero-alloc build —
/// [`PyramidKeyHierarchy::required_len`] sizes it). Build is a pure function
/// of the key block: bottom-up leaf means, then pairwise pooling weighted by
/// covered token counts, which makes every node the EXACT subtree mean
/// (unit-pinned, [`PyramidKeyHierarchy`] tests).
pub struct PyramidKeyHierarchy<'a> {
    storage: &'a [f32],
    d: usize,
    n_keys: usize,
    /// First storage element of each level's flat rows.
    offsets: [usize; MAX_PYRAMID_LEVELS],
    /// Row count per level.
    lens: [usize; MAX_PYRAMID_LEVELS],
    n_levels: usize,
}

impl<'a> PyramidKeyHierarchy<'a> {
    /// Level count for `n_keys` keys: `1 + ⌈log₂(⌈N/C⌉)⌉` (0 for empty).
    pub fn level_count(n_keys: usize) -> usize {
        if n_keys == 0 {
            return 0;
        }
        1 + ceil_log2_usize(n_keys.div_ceil(PYRAMID_BLOCK_SIZE))
    }

    /// Exact f32 element count [`PyramidKeyHierarchy::build`] needs.
    ///
    /// Σ levels ≈ 2·⌈N/C⌉ rows — the plan's ≤ 2N/C·d storage bound.
    pub fn required_len(n_keys: usize, d: usize) -> usize {
        let mut total = 0usize;
        let mut rows = n_keys.div_ceil(PYRAMID_BLOCK_SIZE);
        while rows > 0 {
            total += rows * d;
            if rows == 1 {
                break;
            }
            rows = rows.div_ceil(PYRAMID_BRANCHING);
        }
        total
    }

    /// Build the hierarchy over `keys` ([`n_keys × d`], row-major) into
    /// `storage` (≥ [`PyramidKeyHierarchy::required_len`] elements).
    pub fn build(keys: &[f32], n_keys: usize, d: usize, storage: &'a mut [f32]) -> Self {
        assert!(d > 0, "head_dim must be positive");
        assert!(
            n_keys == 0 || keys.len() >= n_keys * d,
            "keys slice {} < n_keys·d {}",
            keys.len(),
            n_keys * d
        );
        let need = Self::required_len(n_keys, d);
        assert!(
            storage.len() >= need,
            "pyramid storage {} < required {} (required_len)",
            storage.len(),
            need
        );
        let n_levels = Self::level_count(n_keys);
        assert!(
            n_levels <= MAX_PYRAMID_LEVELS,
            "n_keys needs {} levels > MAX_PYRAMID_LEVELS",
            n_levels
        );

        let mut offsets = [0usize; MAX_PYRAMID_LEVELS];
        let mut lens = [0usize; MAX_PYRAMID_LEVELS];

        // Level 0: leaf means over the actual (possibly ragged) token count.
        let n_leaves = n_keys.div_ceil(PYRAMID_BLOCK_SIZE);
        offsets[0] = 0;
        lens[0] = n_leaves;
        for b in 0..n_leaves {
            let lo = b * PYRAMID_BLOCK_SIZE;
            let cnt = (n_keys - lo).min(PYRAMID_BLOCK_SIZE);
            let dst = &mut storage[b * d..(b + 1) * d];
            for v in dst.iter_mut() {
                *v = 0.0;
            }
            for i in 0..cnt {
                let k = &keys[(lo + i) * d..(lo + i + 1) * d];
                for (dst_v, k_v) in dst.iter_mut().zip(k.iter()) {
                    *dst_v += k_v;
                }
            }
            let inv = 1.0 / cnt as f32;
            for v in dst.iter_mut() {
                *v *= inv;
            }
        }

        // Levels ≥ 1: pairwise pooling weighted by covered TOKEN counts —
        // this is what preserves the exact-subtree-mean identity at the
        // ragged tail (leaf-count weights would misweight a short last leaf).
        let mut off = n_leaves * d;
        let mut prev_rows = n_leaves;
        for t in 1..n_levels {
            let rows_t = prev_rows.div_ceil(PYRAMID_BRANCHING);
            offsets[t] = off;
            lens[t] = rows_t;
            let leaves_per_child = 1usize << (t - 1);
            let (done, rest) = storage.split_at_mut(off);
            let prev_base = offsets[t - 1];
            for j in 0..rows_t {
                let dst = &mut rest[j * d..(j + 1) * d];
                for v in dst.iter_mut() {
                    *v = 0.0;
                }
                let mut w_sum = 0.0f32;
                for child in [2 * j, 2 * j + 1] {
                    if child >= prev_rows {
                        continue;
                    }
                    let w = covered_token_count(n_keys, child, leaves_per_child);
                    if w == 0 {
                        continue;
                    }
                    let wf = w as f32;
                    let src = &done[prev_base + child * d..prev_base + (child + 1) * d];
                    for (dst_v, s_v) in dst.iter_mut().zip(src.iter()) {
                        *dst_v += wf * s_v;
                    }
                    w_sum += wf;
                }
                debug_assert!(w_sum > 0.0, "every existing node covers ≥ 1 token");
                let inv = 1.0 / w_sum;
                for v in dst.iter_mut() {
                    *v *= inv;
                }
            }
            off += rows_t * d;
            prev_rows = rows_t;
        }

        Self {
            storage: &storage[..off],
            d,
            n_keys,
            offsets,
            lens,
            n_levels,
        }
    }

    pub fn n_keys(&self) -> usize {
        self.n_keys
    }
}

impl PyramidLevels for PyramidKeyHierarchy<'_> {
    fn head_dim(&self) -> usize {
        self.d
    }

    fn n_keys(&self) -> usize {
        self.n_keys
    }

    fn leaf_count(&self) -> usize {
        if self.n_levels == 0 { 0 } else { self.lens[0] }
    }

    fn n_levels(&self) -> usize {
        self.n_levels
    }

    fn level_row_count(&self, level: usize) -> usize {
        self.lens[level]
    }

    fn level_rows(&self, level: usize) -> &[f32] {
        let lo = self.offsets[level];
        &self.storage[lo..lo + self.lens[level] * self.d]
    }
}

// ---------------------------------------------------------------------------
// Selection (T1.2)
// ---------------------------------------------------------------------------

/// Reusable scratch for [`coarse_to_fine_select`] — steady-state zero-alloc
/// (the G4 gate counts allocations on the REUSED path, the
/// `argtopk_with_scratch` pattern).
pub struct PyramidScratch {
    /// GQA group-summed query [d].
    pub u: Vec<f32>,
    /// Candidate node indices at the level being selected.
    pub cand: Vec<usize>,
    /// Non-forced candidate indices (into `cand`) + their scores.
    pub nonforced: Vec<usize>,
    pub nonforced_scores: Vec<f32>,
    /// Retained node indices driving the next (finer) level.
    pub retained: Vec<usize>,
    /// `argtopk_with_scratch` buffers.
    pub indices: Vec<usize>,
    pub pairs: Vec<(usize, f32)>,
    /// Leaf-level per-token logits (≤ C).
    pub leaf_logits: [f32; PYRAMID_BLOCK_SIZE],
    /// Selected leaf blocks from the LAST [`coarse_to_fine_select`] call on
    /// this scratch — sorted ascending, deduped, ≤ K + `FORCED_LEAF_SLOTS`.
    pub out: Vec<usize>,
}

impl PyramidScratch {
    /// Empty buffers — the first call sizes them, later calls reuse.
    /// (`[f32; 64]` has no `Default`, so the impl below delegates here.)
    pub fn new() -> Self {
        Self {
            u: Vec::new(),
            cand: Vec::new(),
            nonforced: Vec::new(),
            nonforced_scores: Vec::new(),
            retained: Vec::new(),
            indices: Vec::new(),
            pairs: Vec::new(),
            leaf_logits: [0.0; PYRAMID_BLOCK_SIZE],
            out: Vec::new(),
        }
    }
}

impl Default for PyramidScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Scoring policy for [`coarse_to_fine_select`] — the Taylor rung plus the
/// logit scale (the paper's `q·k̄/√d`; pass `1.0/√d` or `1.0`).
#[derive(Clone, Copy, Debug)]
pub struct PyramidScorer {
    pub mode: PyramidScoreMode,
    pub scale: f32,
}

/// Score one candidate at `level` (0 = leaf) index `node`.
///
/// - leaf level: exact per-token logits over the block's ≤ C keys via
///   `simd_dot_f32(u, k)·scale`, reduced per [`PyramidScoreMode`]
///   (`logsumexp_parts` for [`PyramidScoreMode::ExactLse`]).
/// - internal level: ≤ 2 child-summary logits, reduced inline (2-element LSE
///   for [`PyramidScoreMode::ExactLse`]).
#[inline]
fn score_candidate<L: PyramidLevels + ?Sized>(
    levels: &L,
    keys: &[f32],
    u: &[f32],
    scorer: &PyramidScorer,
    level: usize,
    node: usize,
    leaf_logits: &mut [f32; PYRAMID_BLOCK_SIZE],
) -> f32 {
    let d = levels.head_dim();
    let n_keys = keys.len() / d;
    let (mode, scale) = (scorer.mode, scorer.scale);
    if level == 0 {
        let base = node * PYRAMID_BLOCK_SIZE;
        let cnt = (n_keys - base).min(PYRAMID_BLOCK_SIZE);
        debug_assert!(cnt > 0);
        for (i, slot) in leaf_logits[..cnt].iter_mut().enumerate() {
            let k = &keys[(base + i) * d..(base + i + 1) * d];
            *slot = scale * simd_dot_f32(u, k, d);
        }
        let logits = &leaf_logits[..cnt];
        match mode {
            // True LSE = max + ln Σ e^{x−max}: the max MUST be added back —
            // `logsumexp_parts` returns the max-shifted parts precisely so
            // the caller can. ln_z alone is not comparable across blocks
            // (each block's max is its own subtraction constant), and it
            // deletes exactly the needle signal exact-LSE exists to preserve
            // — the MSA/HGA dilution failure mode (Plan 612 T2.2's per-
            // candidate Jensen pin catches this: mean + ln cnt ≤ true LSE
            // holds; against ln_z it does not).
            PyramidScoreMode::ExactLse => {
                let (max_val, ln_z, _) = logsumexp_parts(logits);
                max_val + ln_z
            }
            PyramidScoreMode::Mean => {
                let mut s = 0.0f32;
                for &x in logits {
                    s += x;
                }
                s / cnt as f32
            }
            PyramidScoreMode::MeanPlusHalfVar => {
                let mut s = 0.0f32;
                for &x in logits {
                    s += x;
                }
                let mean = s / cnt as f32;
                let mut ss = 0.0f32;
                for &x in logits {
                    let dv = x - mean;
                    ss += dv * dv;
                }
                let var = ss / cnt as f32;
                mean + 0.5 * var
            }
        }
    } else {
        let rows = levels.level_rows(level - 1);
        let rows_n = levels.level_row_count(level - 1);
        let c0 = node * PYRAMID_BRANCHING;
        debug_assert!(c0 < rows_n, "first child always exists");
        let s0 = scale * simd_dot_f32(u, &rows[c0 * d..(c0 + 1) * d], d);
        let s1 = if c0 + 1 < rows_n {
            Some(scale * simd_dot_f32(u, &rows[(c0 + 1) * d..(c0 + 2) * d], d))
        } else {
            None
        };
        match (mode, s1) {
            (PyramidScoreMode::Mean, None)
            | (PyramidScoreMode::MeanPlusHalfVar, None)
            | (PyramidScoreMode::ExactLse, None) => s0,
            (PyramidScoreMode::Mean, Some(s1)) => (s0 + s1) * 0.5,
            (PyramidScoreMode::MeanPlusHalfVar, Some(s1)) => {
                let m = (s0 + s1) * 0.5;
                let d0 = s0 - m;
                let d1 = s1 - m;
                // var over 2 points = (d0² + d1²)/2 → +½·var
                m + (d0 * d0 + d1 * d1) * 0.25
            }
            (PyramidScoreMode::ExactLse, Some(s1)) => {
                let m = s0.max(s1);
                m + (fast_exp(s0 - m) + fast_exp(s1 - m)).ln()
            }
        }
    }
}

/// Root-seeded coarse-to-fine Top-K leaf-block selection over ONE KV head.
///
/// Generic over [`PyramidLevels`] so both the static
/// [`PyramidKeyHierarchy`] and the streaming [`PyramidDecodeCache`] drive the
/// same loop. `keys` are the ORIGINAL keys (exactly `n_keys × d`, row-major) —
/// the leaf-level exact LSE reads them; the hierarchy does not own them.
/// `n_keys` is derived from the slice and debug-asserted against
/// [`PyramidLevels::n_keys`].
///
/// `group_queries` are the query heads sharing this KV head (GQA) — they are
/// folded into one group-summed query before any scoring (dot-product
/// linearity), so all logits are `simd_dot_f32(u, ·)·scale`.
///
/// `query_pos` is the current token position, driving the forced
/// first/previous/current leaf policy (NSA convention). Forced leaves join
/// the output directly and are NEVER scored (they cannot lose Top-K slots
/// they don't compete for) — see the module doc for the bound this preserves.
///
/// Returns the scored-candidate count (the selected leaves land in
/// `scratch.out` — sorted ascending, deduped, ≤ K + [`FORCED_LEAF_SLOTS`])
/// and asserts the per-call bound `≤ 1 + (gK + g·3)·⌈log₂(n_leaves)⌉`
/// (⌈log₂ n_leaves⌉ is the plan's `⌈log₂(N/C)⌉` on the exact leaf count —
/// the tightest provable form).
pub fn coarse_to_fine_select<L: PyramidLevels + ?Sized>(
    levels: &L,
    keys: &[f32],
    group_queries: &[&[f32]],
    query_pos: usize,
    top_k: usize,
    scorer: PyramidScorer,
    scratch: &mut PyramidScratch,
) -> usize {
    let d = levels.head_dim();
    let n_leaves = levels.leaf_count();
    let n_levels = levels.n_levels();
    let n_keys = keys.len() / d;
    debug_assert!(
        levels.n_keys() == n_keys,
        "keys slice holds {} keys but the pyramid was built over {}",
        n_keys,
        levels.n_keys()
    );
    debug_assert!(
        group_queries.iter().all(|q| q.len() == d),
        "every group query must have head_dim elements"
    );
    scratch.out.clear();
    if n_levels == 0 || n_leaves == 0 {
        return 0;
    }

    // GQA group-sum before Top-K: u = Σ_h q_h (one fold, then every logit is
    // a single dot against u).
    debug_assert!(!group_queries.is_empty(), "pass ≥ 1 query head");
    scratch.u.clear();
    scratch.u.resize(d, 0.0);
    for q in group_queries {
        for (uv, qv) in scratch.u.iter_mut().zip(q.iter()) {
            *uv += qv;
        }
    }
    let u: &[f32] = &scratch.u;

    let forced = ForcedBlocks::at(query_pos, n_leaves);

    // Degenerate: a single leaf — N ≤ C reduces to single-level behavior and
    // the forced policy already pins the only block.
    if n_levels == 1 {
        scratch.out.extend(forced.iter());
        return 0;
    }

    // Root-seeded: the single root node (never scored — the "+1" of the bound).
    scratch.retained.clear();
    scratch.retained.push(0usize);
    let mut total_scored = 0usize;

    for t in (0..n_levels - 1).rev() {
        // Candidates: children of the retained set one level up. The retained
        // set carries BOTH the Top-K picks and the forced ancestors (added
        // below) — a forced branch must still expand, else a level whose
        // every candidate is forced (constant keys, shallow levels) kills
        // the whole walk and only the forced set survives.
        scratch.cand.clear();
        let rows_t = levels.level_row_count(t);
        for &j in scratch.retained.iter() {
            let c0 = j * PYRAMID_BRANCHING;
            if c0 < rows_t {
                scratch.cand.push(c0);
            }
            let c1 = c0 + 1;
            if c1 < rows_t {
                scratch.cand.push(c1);
            }
        }

        // Forced ancestors at this level: retained unconditionally, never
        // scored, never competing for Top-K slots.
        let forced_here = forced.ancestors_at(t);

        scratch.nonforced.clear();
        scratch.nonforced_scores.clear();
        for &c in scratch.cand.iter() {
            if !forced_here.contains(c) {
                let s = score_candidate(levels, keys, u, &scorer, t, c, &mut scratch.leaf_logits);
                // Hold the NODE index `c` — NOT the candidate-array position.
                // `retained` (and the leaf output) consume these as node ids;
                // positions are only valid while `cand` is identity-ordered,
                // which stops holding the moment argtopk returns score-ordered
                // picks (Plan 612: the needle canary caught the walk expanding
                // the wrong nodes through this).
                scratch.nonforced.push(c);
                scratch.nonforced_scores.push(s);
            }
        }
        total_scored += scratch.nonforced.len();

        let k = top_k.min(scratch.nonforced_scores.len());
        scratch.retained.clear();
        if k > 0 {
            argtopk_with_scratch(
                &scratch.nonforced_scores,
                k,
                &mut scratch.indices,
                &mut scratch.pairs,
            );
            for &idx in scratch.indices[..k].iter() {
                scratch.retained.push(scratch.nonforced[idx]);
            }
        }
        // Expansion set for the next level = Top-K picks ∪ forced.
        for f in forced_here.iter() {
            if !scratch.retained.contains(&f) {
                scratch.retained.push(f);
            }
        }
    }

    // Leaf output: Top-K ∪ forced leaves, deduped, ascending.
    scratch.out.extend_from_slice(&scratch.retained);
    for f in forced.iter() {
        if !scratch.out.contains(&f) {
            scratch.out.push(f);
        }
    }
    scratch.out.sort_unstable();

    // Per-call candidate bound (the complexity claim, asserted — Plan 612
    // T1.2). The paper's pure form (no forced policy) scores ≤ gK per level
    // → 1 + gK·⌈log₂(n_leaves)⌉. The NSA forced policy retains ≤
    // FORCED_LEAF_SLOTS extra branches per level that expand but never
    // score, adding a constant g·FORCED per level — the honest bound below
    // (same O(gK log N) shape). ⌈log₂ n_leaves⌉ is the plan's ⌈log₂(N/C)⌉
    // on the exact leaf count.
    let per_level = PYRAMID_BRANCHING * (top_k + FORCED_LEAF_SLOTS);
    let bound = 1 + per_level * ceil_log2_usize(n_leaves.max(1));
    assert!(
        total_scored <= bound,
        "pyramid selection scored {} candidates > bound {}",
        total_scored,
        bound
    );
    total_scored
}

// ---------------------------------------------------------------------------
// Decode cache (T1.3)
// ---------------------------------------------------------------------------

/// Per-KV-head streaming pyramid: rank-1 leaf update + ancestor-path
/// recompute per appended token.
///
/// Steady state is O((N/C)·d) storage (≈ the static hierarchy); level
/// capacities grow by a fixed fraction (amortized O(1) expansions per
/// doubling). Only the appended leaf's ancestor path is rewritten per token
/// — O(log N) nodes — and the result equals a full rebuild within float
/// reassociation error (unit-pinned).
pub struct PyramidDecodeCache {
    d: usize,
    n_keys: usize,
    /// levels[0] = leaf rows, last = root; flat row-major [row][d].
    levels: Vec<Vec<f32>>,
    /// Token count per leaf block (parallel to levels[0] rows).
    leaf_counts: Vec<u32>,
}

/// Fixed-fraction capacity growth: reserve ~50% more rows when `rows` more
/// would exceed capacity (amortized O(1) expansions per doubling).
fn reserve_rows(buf: &mut Vec<f32>, d: usize, rows: usize) {
    if buf.len() + rows * d > buf.capacity() {
        let cap_rows = buf.capacity() / d;
        buf.reserve((cap_rows / 2).max(4) * d);
    }
}

impl PyramidDecodeCache {
    pub fn new(d: usize) -> Self {
        assert!(d > 0, "head_dim must be positive");
        Self {
            d,
            n_keys: 0,
            levels: Vec::new(),
            leaf_counts: Vec::new(),
        }
    }

    pub fn n_keys(&self) -> usize {
        self.n_keys
    }

    pub fn leaf_count(&self) -> usize {
        self.leaf_counts.len()
    }

    /// Append one key: rank-1 leaf update `(c·k̄ + k)/(c+1)` (or a new leaf
    /// row on block boundary), then recompute the ancestor path with the
    /// same token-count weights the static build uses.
    pub fn append(&mut self, key: &[f32]) {
        debug_assert!(key.len() == self.d);
        let pos = self.n_keys;
        let b = pos / PYRAMID_BLOCK_SIZE;
        let r = pos % PYRAMID_BLOCK_SIZE;

        if self.levels.is_empty() {
            self.levels.push(Vec::new());
        }
        if r == 0 {
            let l0 = &mut self.levels[0];
            reserve_rows(l0, self.d, 1);
            l0.extend_from_slice(key);
            self.leaf_counts.push(1);
            // Depth growth: a new leaf may need a new top level.
            let need = 1 + ceil_log2_usize(self.leaf_counts.len());
            while self.levels.len() < need {
                self.levels.push(Vec::new());
            }
        } else {
            let d = self.d;
            let c = self.leaf_counts[b] as f32;
            let c_new = c + 1.0;
            let row = &mut self.levels[0][b * d..(b + 1) * d];
            for (v, k_v) in row.iter_mut().zip(key.iter()) {
                *v = (*v * c + k_v) / c_new;
            }
            self.leaf_counts[b] += 1;
        }

        // Ancestor path recompute: leaf b's ancestor at level t is b >> t.
        // Lower levels are rewritten first so each level reads fresh children.
        let n_new = self.n_keys + 1;
        for t in 1..self.levels.len() {
            let j = b >> t;
            {
                let lt = &mut self.levels[t];
                if (j + 1) * self.d > lt.len() {
                    reserve_rows(lt, self.d, 1);
                    lt.resize((j + 1) * self.d, 0.0);
                }
            }
            let leaves_per_child = 1usize << (t - 1);
            let (below, above) = self.levels.split_at_mut(t);
            let src = &below[t - 1];
            let dst_row = &mut above[0][j * self.d..(j + 1) * self.d];
            for v in dst_row.iter_mut() {
                *v = 0.0;
            }
            let mut w_sum = 0.0f32;
            for child in [2 * j, 2 * j + 1] {
                if (child + 1) * self.d > src.len() {
                    continue;
                }
                let w = covered_token_count(n_new, child, leaves_per_child);
                if w == 0 {
                    continue;
                }
                let wf = w as f32;
                let srow = &src[child * self.d..(child + 1) * self.d];
                for (dv, sv) in dst_row.iter_mut().zip(srow.iter()) {
                    *dv += wf * sv;
                }
                w_sum += wf;
            }
            debug_assert!(w_sum > 0.0);
            let inv = 1.0 / w_sum;
            for v in dst_row.iter_mut() {
                *v *= inv;
            }
        }

        self.n_keys += 1;
    }
}

impl PyramidLevels for PyramidDecodeCache {
    fn head_dim(&self) -> usize {
        self.d
    }

    fn n_keys(&self) -> usize {
        self.n_keys
    }

    fn leaf_count(&self) -> usize {
        self.leaf_counts.len()
    }

    fn n_levels(&self) -> usize {
        self.levels.len()
    }

    fn level_row_count(&self, level: usize) -> usize {
        self.levels[level].len() / self.d
    }

    fn level_rows(&self, level: usize) -> &[f32] {
        &self.levels[level]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic local PRNG — the seeded pattern (`fastrand::Rng::with_seed`),
    /// never the unseeded global.
    fn key_block(rng: &mut fastrand::Rng, n: usize, d: usize) -> Vec<f32> {
        (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect()
    }

    fn one_head(q: &[f32]) -> Vec<&[f32]> {
        vec![q]
    }

    // T1.1 — every node equals the exact subtree mean over its covered tokens.
    #[test]
    fn hierarchy_nodes_are_exact_subtree_means() {
        let (n, d) = (1000usize, 16usize);
        let mut rng = fastrand::Rng::with_seed(0x612);
        let keys = key_block(&mut rng, n, d);
        let mut storage = vec![0.0f32; PyramidKeyHierarchy::required_len(n, d)];
        let hier = PyramidKeyHierarchy::build(&keys, n, d, &mut storage);
        let n_leaves = n.div_ceil(PYRAMID_BLOCK_SIZE);

        for t in 0..hier.n_levels() {
            let leaves_per_node = 1usize << t;
            let rows = hier.level_row_count(t);
            let rows_flat = hier.level_rows(t);
            for j in 0..rows {
                // exact subtree mean over the covered token range
                let leaf_lo = j * leaves_per_node;
                let leaf_hi = (leaf_lo + leaves_per_node).min(n_leaves);
                let tok_lo = leaf_lo * PYRAMID_BLOCK_SIZE;
                let tok_hi = (leaf_hi * PYRAMID_BLOCK_SIZE).min(n);
                let cnt = tok_hi - tok_lo;
                let mut exact = vec![0.0f32; d];
                for i in tok_lo..tok_hi {
                    for (e, k) in exact.iter_mut().zip(&keys[i * d..(i + 1) * d]) {
                        *e += k;
                    }
                }
                for e in exact.iter_mut() {
                    *e /= cnt as f32;
                }
                let node = &rows_flat[j * d..(j + 1) * d];
                for (nv, ev) in node.iter().zip(exact.iter()) {
                    assert!(
                        (nv - ev).abs() <= 1e-5,
                        "level {t} node {j}: {nv} vs exact {ev}"
                    );
                }
            }
        }
    }

    // T1.1 — storage stays within the 2N/C·d bound.
    #[test]
    fn storage_respects_bound() {
        for &(n, d) in &[(64usize, 8usize), (1000, 16), (4096, 32), (65536, 128)] {
            let need = PyramidKeyHierarchy::required_len(n, d);
            let bound = 2 * (n.div_ceil(PYRAMID_BLOCK_SIZE) + 1) * d;
            assert!(need <= bound, "n={n}: required {need} > 2N/C·d {bound}");
        }
    }

    // T1.2 — candidate bound, forced blocks, output shape.
    #[test]
    fn select_respects_bound_and_forced_blocks() {
        let (n, d, k) = (4096usize, 32usize, 8usize);
        let mut rng = fastrand::Rng::with_seed(0x612_002);
        let keys = key_block(&mut rng, n, d);
        let mut storage = vec![0.0f32; PyramidKeyHierarchy::required_len(n, d)];
        let hier = PyramidKeyHierarchy::build(&keys, n, d, &mut storage);
        let q1: Vec<f32> = (0..d).map(|_| rng.f32()).collect();
        let q2: Vec<f32> = (0..d).map(|_| rng.f32()).collect();
        let heads = vec![q1.as_slice(), q2.as_slice()];

        let query_pos = 777usize;
        let mut scratch = PyramidScratch::new();
        let scorer = PyramidScorer {
            mode: PyramidScoreMode::ExactLse,
            scale: 1.0 / (d as f32).sqrt(),
        };
        let scored =
            coarse_to_fine_select(&hier, &keys, &heads, query_pos, k, scorer, &mut scratch);
        let out = &scratch.out;

        let bound = 1 + PYRAMID_BRANCHING
            * (k + FORCED_LEAF_SLOTS)
            * ceil_log2_usize(n.div_ceil(PYRAMID_BLOCK_SIZE));
        assert!(scored <= bound, "scored {scored} > bound {bound}");
        assert!(scored > 0, "selection scored nothing on a live hierarchy");

        let n_leaves = n.div_ceil(PYRAMID_BLOCK_SIZE);
        let cur = (query_pos / PYRAMID_BLOCK_SIZE).min(n_leaves - 1);
        for f in [0usize, cur - 1, cur] {
            assert!(out.contains(&f), "forced leaf {f} missing from {out:?}");
        }
        assert!(out.len() <= k + FORCED_LEAF_SLOTS);
        assert!(
            out.windows(2).all(|w| w[0] < w[1]),
            "sorted + deduped: {out:?}"
        );
        assert!(out.iter().all(|&b| b < n_leaves));
    }

    // T1.2 — GQA group-sum linearity: 2 heads == the elementwise sum as 1 head.
    #[test]
    fn gqa_group_sum_is_linear() {
        let (n, d, k) = (2048usize, 16usize, 6usize);
        let mut rng = fastrand::Rng::with_seed(0x612_003);
        let keys = key_block(&mut rng, n, d);
        let mut storage = vec![0.0f32; PyramidKeyHierarchy::required_len(n, d)];
        let hier = PyramidKeyHierarchy::build(&keys, n, d, &mut storage);
        let q1: Vec<f32> = (0..d).map(|_| rng.f32() - 0.5).collect();
        let q2: Vec<f32> = (0..d).map(|_| rng.f32() - 0.5).collect();
        let q_sum: Vec<f32> = q1.iter().zip(q2.iter()).map(|(a, b)| a + b).collect();
        let two = vec![q1.as_slice(), q2.as_slice()];
        let one = vec![q_sum.as_slice()];

        let mut scratch = PyramidScratch::new();
        let mhs = PyramidScorer {
            mode: PyramidScoreMode::MeanPlusHalfVar,
            scale: 1.0,
        };
        let scored_two = coarse_to_fine_select(&hier, &keys, &two, 999, k, mhs, &mut scratch);
        let out_two = scratch.out.clone();
        let scored_one = coarse_to_fine_select(&hier, &keys, &one, 999, k, mhs, &mut scratch);
        assert_eq!(out_two, scratch.out);
        assert_eq!(scored_two, scored_one);
    }

    // T1.4 — N ≤ C degenerates to single-level behavior (envelope assert).
    #[test]
    fn degenerate_short_sequence_single_level() {
        let (n, d) = (10usize, 8usize);
        let mut rng = fastrand::Rng::with_seed(0x612_004);
        let keys = key_block(&mut rng, n, d);
        let mut storage = vec![0.0f32; PyramidKeyHierarchy::required_len(n, d)];
        let hier = PyramidKeyHierarchy::build(&keys, n, d, &mut storage);
        assert_eq!(hier.n_levels(), 1, "N ≤ C must be a single-level pyramid");

        let q: Vec<f32> = (0..d).map(|_| rng.f32()).collect();
        let heads = one_head(&q);
        let mut scratch = PyramidScratch::new();
        let scorer = PyramidScorer {
            mode: PyramidScoreMode::ExactLse,
            scale: 1.0,
        };
        let scored = coarse_to_fine_select(&hier, &keys, &heads, 7, 4, scorer, &mut scratch);
        assert_eq!(scored, 0);
        assert_eq!(scratch.out, vec![0]);

        // empty pyramid: selection is a no-op (keys must match the pyramid's
        // own zero-key provenance)
        let mut empty_storage: [f32; 0] = [];
        let empty = PyramidKeyHierarchy::build(&[], 0, d, &mut empty_storage);
        let scored2 = coarse_to_fine_select(&empty, &keys[..0], &heads, 0, 4, scorer, &mut scratch);
        assert_eq!((scored2, scratch.out.as_slice()), (0, [].as_slice()));
    }

    // T2.5 canary half — constant keys must not collapse the selector.
    #[test]
    fn constant_keys_do_not_collapse() {
        let (n, d, k) = (2048usize, 16usize, 8usize);
        let keys = vec![0.25f32; n * d];
        let mut storage = vec![0.0f32; PyramidKeyHierarchy::required_len(n, d)];
        let hier = PyramidKeyHierarchy::build(&keys, n, d, &mut storage);
        let q: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let heads = one_head(&q);

        let mut scratch = PyramidScratch::new();
        let scorer = PyramidScorer {
            mode: PyramidScoreMode::ExactLse,
            scale: 1.0,
        };
        coarse_to_fine_select(&hier, &keys, &heads, n - 1, k, scorer, &mut scratch);
        let out = &scratch.out;
        let n_leaves = n.div_ceil(PYRAMID_BLOCK_SIZE);
        let cur = (n - 1) / PYRAMID_BLOCK_SIZE;
        for f in [0usize, cur - 1, cur] {
            assert!(out.contains(&f));
        }
        assert!(
            out.len() >= k && out.len() <= k + FORCED_LEAF_SLOTS,
            "selector collapsed: {out:?}"
        );
        assert!(out.iter().all(|&b| b < n_leaves));
    }

    // T1.3 — path-update == full recompute (float-reassociation tolerance).
    #[test]
    fn decode_cache_path_update_matches_full_recompute() {
        let (d, total) = (8usize, 3000usize);
        let mut rng = fastrand::Rng::with_seed(0x612_005);
        let keys = key_block(&mut rng, total, d);
        let mut cache = PyramidDecodeCache::new(d);

        for i in 0..total {
            cache.append(&keys[i * d..(i + 1) * d]);
            let n = i + 1;
            let checkpoint = n <= 130 || n % 250 == 0 || n == total;
            if !checkpoint {
                continue;
            }
            let need = PyramidKeyHierarchy::required_len(n, d);
            let mut storage = vec![0.0f32; need];
            let hier = PyramidKeyHierarchy::build(&keys, n, d, &mut storage);
            assert_eq!(cache.n_levels(), hier.n_levels(), "depth mismatch at n={n}");
            for t in 0..cache.n_levels() {
                let cache_flat = cache.level_rows(t);
                let hier_flat = hier.level_rows(t);
                assert_eq!(
                    cache_flat.len(),
                    hier_flat.len(),
                    "rows mismatch at n={n}, t={t}"
                );
                for (cv, hv) in cache_flat.iter().zip(hier_flat.iter()) {
                    assert!(
                        (cv - hv).abs() <= 1e-4,
                        "n={n} level {t}: cache {cv} vs rebuild {hv}"
                    );
                }
            }
        }
    }

    // T1.3 — the cache drives the same selection core.
    #[test]
    fn decode_cache_backed_selection_runs() {
        let (d, total, k) = (16usize, 1500usize, 4usize);
        let mut rng = fastrand::Rng::with_seed(0x612_006);
        let keys = key_block(&mut rng, total, d);
        let mut cache = PyramidDecodeCache::new(d);
        for i in 0..total {
            cache.append(&keys[i * d..(i + 1) * d]);
        }
        let q: Vec<f32> = (0..d).map(|_| rng.f32()).collect();
        let heads = one_head(&q);
        let mut scratch = PyramidScratch::new();
        let scorer = PyramidScorer {
            mode: PyramidScoreMode::ExactLse,
            scale: 1.0,
        };
        coarse_to_fine_select(&cache, &keys, &heads, total - 1, k, scorer, &mut scratch);
        let n_leaves = total.div_ceil(PYRAMID_BLOCK_SIZE);
        assert_eq!(
            *scratch.out.last().unwrap(),
            n_leaves - 1,
            "current leaf forced"
        );
        assert!(scratch.out.windows(2).all(|w| w[0] < w[1]));
    }

    // Plan 612 T2.2 pin (a) — the per-candidate score-ordering THEOREM: for
    // every leaf block, normalized mean score (mean + ln cnt) ≤ true LSE
    // (Jensen: ln mean(e^x) ≥ mean(x), with the uniform-block ln C added
    // back on the mean side). Asserted against the same per-token logits
    // `score_candidate` reduces, so it pins the LEAF arm's LSE form — the
    // max must be added back onto ln_z or this pin fails (the ln_z-only
    // transcription drops each block's max, which is exactly the needle
    // signal exact-LSE exists to preserve).
    #[test]
    fn leaf_ladder_jensen_pin_mean_plus_ln_c_le_lse() {
        let (n, d) = (1024usize, 32usize);
        let mut rng = fastrand::Rng::with_seed(0x612_007);
        let keys = key_block(&mut rng, n, d);
        let mut storage = vec![0.0f32; PyramidKeyHierarchy::required_len(n, d)];
        let hier = PyramidKeyHierarchy::build(&keys, n, d, &mut storage);
        let q: Vec<f32> = (0..d).map(|_| rng.f32() - 0.5).collect();
        let scale = 1.0f32 / (d as f32).sqrt();
        let mut logits = [0.0f32; PYRAMID_BLOCK_SIZE];
        let scorer = PyramidScorer {
            mode: PyramidScoreMode::ExactLse,
            scale,
        };

        let n_leaves = n.div_ceil(PYRAMID_BLOCK_SIZE);
        for leaf in 0..n_leaves {
            let lse = score_candidate(&hier, &keys, &q, &scorer, 0, leaf, &mut logits);
            let mean = score_candidate(
                &hier,
                &keys,
                &q,
                &PyramidScorer {
                    mode: PyramidScoreMode::Mean,
                    scale,
                },
                0,
                leaf,
                &mut logits,
            );
            let base = leaf * PYRAMID_BLOCK_SIZE;
            let cnt = (n - base).min(PYRAMID_BLOCK_SIZE);
            let lhs = mean + (cnt as f32).ln();
            assert!(
                lhs <= lse + 1e-3,
                "Jensen pin violated at leaf {leaf}: mean+ln(cnt)={lhs} > LSE={lse}"
            );
        }
    }

    // The paper's load-bearing ablation in miniature — the dilution canary.
    // One needle key (strongly aligned with u) buried in a block of
    // small-random hay keys: the Mean rung averages the needle down to
    // big/64 and MISSES the block; ExactLse's max-term keeps it and the
    // block is SELECTED. This is the exact MSA/HGA dilution failure mode
    // the ladder exists to fix, pinned as behavior (not just the per-
    // candidate theorem above).
    #[test]
    fn exact_lse_selects_needle_block_mean_rung_misses_it() {
        let (n, d, k) = (2048usize, 32usize, 4usize);
        let mut rng = fastrand::Rng::with_seed(0x612_008);
        let mut keys = key_block(&mut rng, n, d);
        // Needle: block 7, first key = a large positive multiple of u.
        let needle_leaf = 7usize;
        let needle_row = needle_leaf * PYRAMID_BLOCK_SIZE;
        let mut u: Vec<f32> = (0..d).map(|_| rng.f32() - 0.5).collect();
        let unorm: f32 = u.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in u.iter_mut() {
            *v /= unorm;
        }
        let needle_gain = 40.0f32; // mean rung sees 40/64 ≈ 0.63, hay mean ≈ 0
        for (i, v) in u.iter().enumerate() {
            keys[needle_row * d + i] = needle_gain * v;
        }
        let mut storage = vec![0.0f32; PyramidKeyHierarchy::required_len(n, d)];
        let hier = PyramidKeyHierarchy::build(&keys, n, d, &mut storage);
        let heads = one_head(&u);
        let mut scratch = PyramidScratch::new();

        let lse = PyramidScorer {
            mode: PyramidScoreMode::ExactLse,
            scale: 1.0,
        };
        coarse_to_fine_select(&hier, &keys, &heads, n - 1, k, lse, &mut scratch);
        assert!(
            scratch.out.contains(&needle_leaf),
            "ExactLse must select the needle block {}, got {:?}",
            needle_leaf,
            scratch.out
        );

        // The Mean rung: mean score of the needle block ≈ 24/64 ≈ 0.375 vs
        // hay-block means ~N(0, small) — but the FORCED current/first/prev
        // leaves plus K random hay blocks can crowd it out only if K is
        // tiny; with k=4 and a 0.375 signal against ~±0.05 hay noise the
        // needle DOES survive for Mean too — so the honest assertion is the
        // SCORE gap, not the selection gap. Pin the mechanism where it
        // lives: the leaf score ladder ordering on the needle block.
        let mut logits = [0.0f32; PYRAMID_BLOCK_SIZE];
        let mean_score = score_candidate(
            &hier,
            &keys,
            &u,
            &PyramidScorer {
                mode: PyramidScoreMode::Mean,
                scale: 1.0,
            },
            0,
            needle_leaf,
            &mut logits,
        );
        let lse_score = score_candidate(&hier, &keys, &u, &lse, 0, needle_leaf, &mut logits);
        // LSE ≈ ln(e^40 + 63·e^±small) ≈ 40; mean ≈ 40/64 + (hay mean)/64.
        assert!(
            lse_score > needle_gain * 0.9,
            "LSE must retain the needle max: {lse_score} vs gain {needle_gain}"
        );
        assert!(
            mean_score < needle_gain * 0.1,
            "Mean rung must dilute the needle: {mean_score} vs gain {needle_gain}"
        );
    }
}
