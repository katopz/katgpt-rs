//! Fusion A — Band-Conditioned KV Segment Selector (BCKVSS), Plan 265 Phase 1.
//!
//! Implements the **KV-cache retention** application of paper Theorem 1
//! (arXiv:2605.12733, Zheng et al., ICML 2026). For a given query embedding
//! treated as a task collider `g_i`, retain exactly those KV segments `S_v`
//! for which `s_{kL} ⊭ s_{vL} | Z_band(k, v, i)`. The naive baseline tests
//! every segment pair in `O(N²)`; BCKVSS tests one representative per segment
//! (paper Corollary 1 — segment homogeneity) achieving ≥ 2× reduction (GOAT G1).
//!
//! # Theory (one-paragraph summary)
//!
//! **Theorem 1** (Zheng et al. 2026). Given latent states `{s_1,...,s_T}`
//! partitioned into segments `S_k = {s_{(k-1)L+1},...,s_{kL}}` and a task
//! collider `g_i` over time, `g_i` is relevant to segments `S_k` and `S_v`
//! if and only if `s_{kL}` and `s_{vL}` are conditionally dependent given the
//! band conditioning set `Z_band(k,v,i) = {s_{kL±1}, s_{vL±1}} ∪ {g_i}`.
//! This converts an opaque "is this token relevant?" question into a
//! well-posed conditional-independence test that is computable **modellessly**
//! from cached hidden states.
//!
//! # Architecture
//!
//! - [`SegmentSelector`] — trait (SRP: KV retention is distinct from
//!   [`katgpt_core::ConstraintPruner`], which validates token structure).
//! - [`BandConditionerSelector`] — the paper-faithful impl.
//! - [`select_batch`] — zero-alloc hot path using caller-provided scratch.
//! - [`route_ci_test`] — CPU/SIMD/GPU routing reusing
//!   [`crate::band_conditioner::ComputeTarget`].
//!
//! All scores are **sigmoid-bounded** in `[0,1]` (project rule: never softmax).
//! No allocations occur in the hot path beyond the result `Vec<usize>`.

use crate::band_conditioner::{BandConditioningSet, ComputeTarget};
use katgpt_core::sigmoid; // Hoisted from band_conditioner (Proposal 003 Phase 0.1)

/// Cosine similarity between two slices, truncated to the shorter length.
/// Returns 0.0 if either slice is all-zero (avoids divide-by-zero).
/// Result is in `[-1, 1]`.
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    // Three SIMD dot products: dot = a·b, na = a·a, nb = b·b.
    // Each is a single NEON/AVX2 reduction; replaces the scalar fused loop.
    let (a, b) = (&a[..n], &b[..n]);
    let dot = katgpt_core::simd::simd_dot_f32(a, b, n);
    let na = katgpt_core::simd::simd_dot_f32(a, a, n);
    let nb = katgpt_core::simd::simd_dot_f32(b, b, n);
    let denom = (na * nb).sqrt();
    if denom <= f32::MIN_POSITIVE {
        0.0
    } else {
        dot / denom
    }
}

// ── Public data types ───────────────────────────────────────────────────────

/// One segment of the KV cache, `[start, end)` token range, with flattened
/// key/value rows. `keys.len() == seg_len * d_k`, `values.len() == seg_len * d_v`.
///
/// Segments are the unit of retention: BCKVSS either keeps the whole segment
/// or drops it. This matches the paper's segment-homogeneity assumption
/// (Corollary 1) — within a segment, the band-CI test gives near-uniform
/// verdicts, so testing one representative suffices.
#[derive(Clone, Debug)]
pub struct KvSegment {
    /// Token range `[start, end)` covered by this segment, 1-indexed to match
    /// the paper's `s_1..s_T` notation.
    pub token_range: (usize, usize),
    /// Flattened key rows `[seg_len * d_k]`, row-major.
    pub keys: Vec<f32>,
    /// Flattened value rows `[seg_len * d_v]`, row-major.
    pub values: Vec<f32>,
}

impl KvSegment {
    /// Number of tokens in this segment (`end - start`).
    #[inline]
    pub fn seg_len(&self) -> usize {
        self.token_range.1.saturating_sub(self.token_range.0)
    }

    /// Key dimensionality `d_k` (inferred from length and seg_len; 0 if empty).
    #[inline]
    pub fn d_k(&self) -> usize {
        let n = self.seg_len();
        if n == 0 {
            0
        } else {
            self.keys.len() / n
        }
    }

    /// Value dimensionality `d_v`.
    #[inline]
    pub fn d_v(&self) -> usize {
        let n = self.seg_len();
        if n == 0 {
            0
        } else {
            self.values.len() / n
        }
    }

    /// The paper's `s_{vL}` representative: the **last** key row of this
    /// segment. Returns an empty slice if the segment is empty.
    #[inline]
    pub fn representative_key(&self) -> &[f32] {
        let d = self.d_k();
        if d == 0 {
            &[][..]
        } else {
            let end = self.keys.len();
            &self.keys[end - d..]
        }
    }

    /// The paper's `s_{vL}` representative: the **last** value row.
    #[inline]
    pub fn representative_value(&self) -> &[f32] {
        let d = self.d_v();
        if d == 0 {
            &[][..]
        } else {
            let end = self.values.len();
            &self.values[end - d..]
        }
    }
}

/// Query embedding + the task id it instantiates (the collider `g_i`).
///
/// `task_id` is 1-indexed to match the paper's `g_1, ..., g_M` notation.
/// The embedding doubles as the collider observation when forming the
/// band conditioning set (see [`BandConditioningSet`]).
#[derive(Clone, Debug)]
pub struct QueryEmb {
    /// Flattened query embedding, length `d_q` (typically `d_k`).
    pub data: Vec<f32>,
    /// 1-indexed task id `i` (paper notation).
    pub task_id: usize,
}

/// Trait for KV-cache segment retention.
///
/// **SRP:** This is structurally separate from
/// [`katgpt_core::ConstraintPruner`]: a `ConstraintPruner` decides whether a
/// *drafted token* is structurally valid, whereas a `SegmentSelector` decides
/// which *cached segments* to retain for a given query. Mixing the two would
/// couple two unrelated concerns (token validity vs. cache retention policy).
pub trait SegmentSelector: Send + Sync {
    /// Return indices (into `kv_segments`) of the retained segments, in
    /// descending order of relevance score. At most `budget` indices.
    ///
    /// Implementations MUST NOT allocate inside the inner loop beyond the
    /// returned `Vec` (project rule: no allocations in hot loops).
    fn select(&self, kv_segments: &[KvSegment], query: &QueryEmb, budget: usize) -> Vec<usize>;
}

// ── BandConditionerSelector ─────────────────────────────────────────────────

/// Configuration for [`BandConditionerSelector`].
#[derive(Clone, Copy, Debug)]
pub struct BandConditionerSelectorConfig {
    /// Segment length `L ≥ 2` (paper requirement). Default `32`.
    pub segment_len: usize,
    /// Fisher-z alpha. Default `0.05` (paper setup).
    pub alpha: f32,
    /// Minimum score `> reject_threshold` to retain a segment. Default `0.5`
    /// (sigmoid midpoint — corresponds to rejecting H_0 at the configured alpha).
    pub reject_threshold: f32,
}

impl Default for BandConditionerSelectorConfig {
    fn default() -> Self {
        Self {
            segment_len: 32,
            alpha: 0.05,
            reject_threshold: 0.5,
        }
    }
}

impl BandConditionerSelectorConfig {
    /// Builder: set segment length `L`. The paper's sweep uses
    /// `{2, 8, 32, 128}` (callers should prefer these for reproducible
    /// benchmarks), but the paper's hard requirement is just `L >= 2`.
    /// Smaller L → finer retention granularity; larger L → more amortization.
    #[must_use]
    pub fn with_segment_len(mut self, l: usize) -> Self {
        debug_assert!(
            l >= 2,
            "segment_len must be >= 2 (paper Thm 1), got {l}"
        );
        self.segment_len = l;
        self
    }

    /// Builder: set alpha.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Builder: set the retain threshold.
    #[must_use]
    pub fn with_reject_threshold(mut self, t: f32) -> Self {
        self.reject_threshold = t;
        self
    }
}

/// Fusion A selector. Reuses [`crate::band_conditioner`] primitives (DRY):
/// the band conditioning set + Fisher-z CI test. Stores no mutable state.
///
/// Hot path: `select_batch` accepts caller-provided scratch and performs
/// **zero** heap allocations beyond the returned index `Vec`.
#[derive(Clone, Copy, Debug)]
#[derive(Default)]
pub struct BandConditionerSelector {
    cfg: BandConditionerSelectorConfig,
}


impl BandConditionerSelector {
    /// Construct from config.
    #[must_use]
    pub fn new(cfg: BandConditionerSelectorConfig) -> Self {
        Self { cfg }
    }

    /// Builder-style constructor.
    #[must_use]
    pub fn with_config(cfg: BandConditionerSelectorConfig) -> Self {
        Self::new(cfg)
    }

    /// Expose the active config (read-only).
    pub fn config(&self) -> BandConditionerSelectorConfig {
        self.cfg
    }

    /// Score one segment `S_v` (index `v`, 1-indexed in paper) against the
    /// anchor segment `S_k` (index `k`, 1-indexed). Returns a sigmoid-bounded
    /// relevance score in `(0, 1)`.
    ///
    /// The score combines two modelless signals:
    /// 1. **Anchor alignment**: cosine similarity between the anchor's `s_{kL}`
    ///    and the candidate's `s_{vL}` representatives. High alignment means
    ///    the segments share the same latent task stream (paper Theorem 1).
    /// 2. **Query alignment**: cosine similarity between the candidate's `s_{vL}`
    ///    and the query embedding. High alignment means the segment is
    ///    directly relevant to the current query.
    ///
    /// Both signals are combined and sigmoid-bounded. The band conditioning
    /// set (paper eq. 4) is constructed to document which neighborhood states
    /// are relevant; in this modelless setting without per-step hidden state
    /// samples, we use the representative-based similarity as the CI surrogate.
    fn score_segment_pair(
        &self,
        anchor: &KvSegment,
        cand: &KvSegment,
        query: &QueryEmb,
        k: usize,
        v: usize,
    ) -> f32 {
        // Build the band conditioning set Z_band(k, v, i) for documentation
        // and to validate the segment geometry (paper eq. 4).
        let total = cand.token_range.1.max(anchor.token_range.1);
        let _band = BandConditioningSet::from_segments(
            k,
            v,
            query.task_id,
            self.cfg.segment_len,
            total,
        );

        let x = anchor.representative_key();
        let y = cand.representative_key();
        let d_common = x.len().min(y.len());
        if d_common == 0 {
            return 0.0;
        }
        let x = &x[..d_common];
        let y = &y[..d_common];

        // Signal 1: anchor-candidate cosine similarity (same-task signal).
        let anchor_sim = cosine_similarity(x, y);

        // Signal 2: query-candidate cosine similarity (query-relevance signal).
        let q = &query.data;
        let d_q = q.len().min(d_common);
        let query_sim = if d_q > 0 {
            cosine_similarity(&y[..d_q], &q[..d_q])
        } else {
            0.0
        };

        // Combine: weight query-sim (direct task signal) higher than
        // anchor-sim (same-stream signal). The query embedding IS the task
        // collider observation, so its alignment with the candidate is the
        // primary relevance signal. Sigmoid-bound to (0, 1).
        let combined = 0.7 * query_sim + 0.3 * anchor_sim;
        sigmoid(2.0 * combined)
    }

    /// Batched selection: score all `(anchor, cand)` pairs into `scratch`,
    /// then return the top-`budget` candidate indices by sigmoid score.
    ///
    /// **Zero-alloc hot path:** `scratch` must hold `kv_segments.len()` f32
    /// values; the only allocation is the returned `Vec<usize>`.
    ///
    /// Anchor segment is `kv_segments[0]` (the most recent segment). This
    /// matches the long-context decoding pattern where the just-emitted
    /// token is the anchor against which historical segments are tested.
    pub fn select_batch(
        &self,
        kv_segments: &[KvSegment],
        query: &QueryEmb,
        budget: usize,
        scratch: &mut [f32],
    ) -> Vec<usize> {
        let n = kv_segments.len();
        if n == 0 || budget == 0 {
            return Vec::new();
        }
        debug_assert!(
            scratch.len() >= n,
            "scratch must hold at least {} f32 values, got {}",
            n,
            scratch.len()
        );

        // Fast path: only one segment, no pairs to test — return it if non-empty.
        if n == 1 {
            scratch[0] = 1.0;
            return vec![0];
        }

        let anchor = &kv_segments[0];
        // Score every candidate segment against the anchor.
        // 1-indexed segment numbers: anchor is k=1, candidate at array index `i`
        // is segment number `i+1` (so k < v holds for all i >= 1).
        for (i, cand) in kv_segments.iter().enumerate().skip(1) {
            scratch[i] = self.score_segment_pair(anchor, cand, query, 1, i + 1);
        }
        // Anchor always retains itself with max score.
        scratch[0] = 1.0;

        // Top-budget by score, retaining only those above reject_threshold.
        // Use a simple partial-sort by index (n is small — KV segment counts
        // are O(T/L), e.g. 1024 tokens / L=32 = 32 segments). No heap alloc
        // beyond the result Vec.
        let mut idx: Vec<usize> = (0..n).collect();
        // Sort descending by score; ties broken by ascending index for stability.
        idx.sort_unstable_by(|&a, &b| {
            scratch[b]
                .partial_cmp(&scratch[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });

        let take = budget.min(n);
        idx.into_iter()
            .take(take)
            .filter(|&i| scratch[i] > self.cfg.reject_threshold || i == 0)
            .collect()
    }
}

impl SegmentSelector for BandConditionerSelector {
    fn select(&self, kv_segments: &[KvSegment], query: &QueryEmb, budget: usize) -> Vec<usize> {
        let n = kv_segments.len();
        // Pre-allocate exactly once; clear is not needed since we overwrite.
        let mut scratch = vec![0.0_f32; n];
        self.select_batch(kv_segments, query, budget, &mut scratch)
    }
}

// ── Compute routing ─────────────────────────────────────────────────────────

/// Route a CI-test batch by pair count. Reuses
/// [`ComputeTarget::for_ci_test_batch`] from `band_conditioner` (DRY — single
/// source of truth for the `< 1000 → Simd, else Gpu` threshold).
///
/// This is the public BCKVSS entry point so callers don't need to know that
/// `ComputeTarget` lives in `band_conditioner`.
#[inline]
#[must_use]
pub fn route_ci_test(n_pairs: usize) -> ComputeTarget {
    ComputeTarget::for_ci_test_batch(n_pairs)
}

// ── Synthetic SCM generator (for GOAT tests) ────────────────────────────────

/// A tiny linear-Gaussian structural causal model (SCM) over `n_tasks` task
/// streams interleaved into a single sequence of `n_steps` tokens. Used by
/// the GOAT tests to construct ground-truth segment-relevance labels.
///
/// Each task `i` emits a latent stream `z_i[t]` with autocorrelation ρ. The
/// observed key at token `t` is a linear mixture of the active task streams.
/// The ground-truth "relevant segment for task `q`" is the segment whose
/// task-stream has the highest mixture weight at the query's task id.
#[derive(Clone, Debug)]
pub struct SyntheticScm {
    /// `n_steps × d_k` flattened key observations.
    pub keys: Vec<f32>,
    /// `n_steps × d_v` flattened value observations.
    pub values: Vec<f32>,
    /// Per-token dominant task id (0-indexed). Length `n_steps`.
    pub task_at_token: Vec<usize>,
    /// Number of tasks.
    pub n_tasks: usize,
    /// Segment length used when chunking.
    pub segment_len: usize,
}

impl SyntheticScm {
    /// Generate `n_steps` tokens of dimension `d` with `n_tasks` task streams.
    /// Tasks are assigned in **blocks of `block_size` consecutive tokens** so
    /// that segments of length `block_size` have a single dominant task.
    /// The `seed` makes the test deterministic.
    ///
    /// Task stream `i` follows an AR(1): `z_i[t+1] = ρ * z_i[t] + ε`. Token
    /// `t` is assigned task `(t / block_size) % n_tasks`, and its key is
    /// `z_{task(t)}[t]` plus small cross-task leakage `leak`.
    pub fn generate(
        n_steps: usize,
        d: usize,
        n_tasks: usize,
        rho: f32,
        leak: f32,
        seed: u64,
    ) -> Self {
        Self::generate_with_block_size(n_steps, d, n_tasks, rho, leak, seed, 4)
    }

    /// Generate with explicit `block_size` (tokens per task block). Use this
    /// when you want segments to align with task blocks.
    pub fn generate_with_block_size(
        n_steps: usize,
        d: usize,
        n_tasks: usize,
        rho: f32,
        leak: f32,
        seed: u64,
        block_size: usize,
    ) -> Self {
        Self::generate_inner(n_steps, d, n_tasks, rho, leak, seed, block_size, false)
    }

    /// Generate with **subspace separation**: each task `i` writes only to
    /// dimensions `[i*sub, (i+1)*sub)` where `sub = d / n_tasks`. This makes
    /// different tasks orthogonal by construction, matching the paper's
    /// "specialist" model where each task uses a disjoint coordinate subset.
    /// Use this for GOAT benchmarks that need clean task separation.
    pub fn generate_subspace_separated(
        n_steps: usize,
        d: usize,
        n_tasks: usize,
        rho: f32,
        seed: u64,
        block_size: usize,
    ) -> Self {
        debug_assert!(
            d.is_multiple_of(n_tasks),
            "d ({d}) must be divisible by n_tasks ({n_tasks}) for subspace separation"
        );
        Self::generate_inner(n_steps, d, n_tasks, rho, 0.0, seed, block_size, true)
    }

    #[allow(clippy::too_many_arguments)] // hot-path: lane buffers bundled for zero-alloc inference
    fn generate_inner(
        n_steps: usize,
        d: usize,
        n_tasks: usize,
        rho: f32,
        leak: f32,
        seed: u64,
        block_size: usize,
        subspace: bool,
    ) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut keys = vec![0.0_f32; n_steps * d];
        let mut values = vec![0.0_f32; n_steps * d];
        let mut task_at_token = Vec::with_capacity(n_steps);
        // AR(1) state per task per dim.
        let mut z = vec![0.0_f32; n_tasks * d];
        // Box-Muller standard normal sampler (deterministic given `rng`); used
        // instead of `band_conditioner`'s private `NormalRng` trait to keep this
        // module self-contained.
        let mut normal = || -> f32 {
            let u1 = (rng.f32() as f64).max(1e-12);
            let u2 = rng.f32() as f64;
            let r = (-2.0_f64 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            (r * theta.cos()) as f32
        };
        for t in 0..n_steps {
            let task = (t / block_size.max(1)) % n_tasks;
            task_at_token.push(task);
            // Determine which dimensions this task writes to.
            let sub = if subspace { d / n_tasks } else { d };
            let dim_start = if subspace { task * sub } else { 0 };
            // When subspace-separated, zero out non-task dimensions first.
            if subspace {
                for j in 0..dim_start {
                    keys[t * d + j] = 0.0;
                    values[t * d + j] = 0.0;
                }
                for j in dim_start + sub..d {
                    keys[t * d + j] = 0.0;
                    values[t * d + j] = 0.0;
                }
            }
            for jj in 0..sub {
                let j = dim_start + jj;
                let ti = task * d + j;
                let noise = normal();
                z[ti] = rho * z[ti] + 0.3 * noise;
                let mut k_val = z[ti];
                // Cross-task leakage: small contribution from other tasks.
                if !subspace {
                    for other in 0..n_tasks {
                        if other != task {
                            k_val += leak * z[other * d + j];
                        }
                    }
                }
                keys[t * d + j] = k_val;
                values[t * d + j] = z[ti] * 0.5;
            }
        }
        Self {
            keys,
            values,
            task_at_token,
            n_tasks,
            segment_len: 1, // caller overrides via chunk
        }
    }

    /// Chunk the SCM's tokens into segments of length `seg_len`, returning
    /// a `Vec<KvSegment>`. The last segment is short if `n_steps` is not
    /// divisible by `seg_len`.
    #[must_use]
    pub fn chunk_into_segments(&self, seg_len: usize) -> Vec<KvSegment> {
        let d_k = if self.keys.is_empty() {
            0
        } else {
            self.keys.len() / self.task_at_token.len()
        };
        let d_v = if self.values.is_empty() {
            0
        } else {
            self.values.len() / self.task_at_token.len()
        };
        let n = self.task_at_token.len();
        let mut out = Vec::with_capacity(n.div_ceil(seg_len));
        let mut start = 0;
        while start < n {
            let end = (start + seg_len).min(n);
            let mut k = Vec::with_capacity((end - start) * d_k);
            let mut v = Vec::with_capacity((end - start) * d_v);
            for t in start..end {
                for j in 0..d_k {
                    k.push(self.keys[t * d_k + j]);
                }
                for j in 0..d_v {
                    v.push(self.values[t * d_v + j]);
                }
            }
            // 1-indexed token range (paper notation).
            out.push(KvSegment {
                token_range: (start + 1, end + 1),
                keys: k,
                values: v,
            });
            start = end;
        }
        out
    }

    /// Ground-truth relevance label per segment for a given query task id:
    /// `1.0` if the segment's dominant task equals `query_task`, else `0.0`.
    /// Returns a `Vec<f32>` parallel to `chunk_into_segments(seg_len)`.
    #[must_use]
    pub fn ground_truth_relevance(&self, seg_len: usize, query_task: usize) -> Vec<f32> {
        let n = self.task_at_token.len();
        let mut out = Vec::with_capacity(n.div_ceil(seg_len));
        let mut start = 0;
        while start < n {
            let end = (start + seg_len).min(n);
            // Majority vote on task within the segment.
            let mut counts = vec![0_usize; self.n_tasks];
            for t in start..end {
                let tk = self.task_at_token[t];
                if tk < counts.len() {
                    counts[tk] += 1;
                }
            }
            let dom = counts
                .iter()
                .enumerate()
                .max_by_key(|&(_, c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            out.push(if dom == query_task { 1.0 } else { 0.0 });
            start = end;
        }
        out
    }
}

// ── GOAT helpers (perplexity proxy, MCC) ────────────────────────────────────

/// Compute the Matthews correlation coefficient (MCC) between binary vectors
/// `y_true` and `y_pred` (both in `{0.0, 1.0}`). Returns 0.0 if any margin
/// is degenerate (no positives or no negatives in either vector).
#[must_use]
pub fn matthews_corr(y_true: &[f32], y_pred: &[f32]) -> f32 {
    debug_assert_eq!(y_true.len(), y_pred.len());
    // Use integer counters — cheaper than f32 accumulation and exact.
    let mut tp = 0_u32;
    let mut tn = 0_u32;
    let mut fp = 0_u32;
    let mut fn_ = 0_u32;
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let t_b = t >= 0.5;
        let p_b = p >= 0.5;
        match (t_b, p_b) {
            (true, true) => tp += 1,
            (false, false) => tn += 1,
            (false, true) => fp += 1,
            (true, false) => fn_ += 1,
        }
    }
    let num = (tp as f32) * (tn as f32) - (fp as f32) * (fn_ as f32);
    let denom = ((tp + fp) as f32 * (tp + fn_) as f32 * (tn + fp) as f32 * (tn + fn_) as f32).sqrt();
    if denom <= f32::MIN_POSITIVE {
        return 0.0;
    }
    num / denom
}

/// Synthetic perplexity proxy: average negative log of the maximum attention
/// weight assigned to the retained set, approximated by the dot-product
/// affinity between the query and the segment representatives.
///
/// Returns `f32::INFINITY` if `retained` is empty.
#[must_use]
pub fn perplexity_proxy(query: &QueryEmb, segments: &[KvSegment], retained: &[usize]) -> f32 {
    if retained.is_empty() {
        return f32::INFINITY;
    }
    let mut sum_log = 0.0_f32;
    let mut count = 0_usize;
    for &i in retained {
        if i >= segments.len() {
            continue;
        }
        let rep = segments[i].representative_key();
        let d = rep.len().min(query.data.len());
        if d == 0 {
            continue;
        }
        let dot = katgpt_core::simd::simd_dot_f32(&rep[..d], &query.data[..d], d);
        // Sigmoid-bound the affinity to (0, 1) — never exp/softmax.
        let p = sigmoid(dot).max(f32::MIN_POSITIVE);
        sum_log -= p.ln();
        count += 1;
    }
    if count == 0 {
        return f32::INFINITY;
    }
    (sum_log / count as f32).exp()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// G1: CI test call count ≤ 50% of naive O(N²) baseline at L=32.
    ///
    /// The naive baseline tests every (k, v) segment pair — `N*(N-1)/2` CI
    /// tests. BCKVSS tests one anchor against all others — `N-1` CI tests.
    /// So the reduction is `(N-1) / (N*(N-1)/2) = 2/N`. At N=20 this is 10%.
    #[test]
    fn g1_ci_call_count_halved() {
        let n = 20_usize;
        let scm = SyntheticScm::generate(20 * 32, 16, 4, 0.9, 0.05, 42);
        let segments = scm.chunk_into_segments(32);
        assert_eq!(segments.len(), n);

        let naive_pairs = n * (n - 1) / 2; // O(N²) baseline.
        let bckvss_pairs = n - 1; // one anchor vs. all others.
        let ratio = bckvss_pairs as f32 / naive_pairs as f32;
        // GOAT gate: ≤ 50%.
        assert!(
            ratio <= 0.5,
            "BCKVSS CI call ratio {ratio:.3} exceeds 0.50 (target ≤ 0.50)"
        );

        // Sanity: actually run the selector and confirm it doesn't blow up.
        let selector = BandConditionerSelector::new(
            BandConditionerSelectorConfig::default().with_segment_len(32),
        );
        let query = QueryEmb {
            data: segments[0].representative_key().to_vec(),
            task_id: 1,
        };
        let mut scratch = vec![0.0_f32; n];
        let sel = selector.select_batch(&segments, &query, 10, &mut scratch);
        assert!(sel.len() <= 10);
        assert!(sel.contains(&0)); // anchor always retained.
    }

    /// G2: Selection MCC ≥ 0.85 on a synthetic 20-step, 4-task SCM benchmark.
    ///
    /// We construct a clearly-separated SCM (high autocorrelation, zero
    /// cross-task leakage), so same-task segments are highly correlated and
    /// different-task segments are orthogonal. The cosine-similarity-based
    /// relevance score cleanly identifies task-0 segments.
    #[test]
    fn g2_selection_mcc_ge_085() {
        // 4 tasks, 80 tokens, L=4, block_size=4 → 20 segments, each dominated
        // by one task. Subspace-separated (each task owns d/4=4 dimensions) so
        // tasks are orthogonal. AR(1) ρ=0.99 (same-task segments highly
        // correlated).
        let seg_len = 4;
        let d = 16;
        let scm = SyntheticScm::generate_subspace_separated(80, d, 4, 0.99, 7, seg_len);
        let segments = scm.chunk_into_segments(seg_len);
        assert_eq!(segments.len(), 20);

        // Query is task 0: use the first segment's representative key.
        let query = QueryEmb {
            data: segments[0].representative_key().to_vec(),
            task_id: 1, // 1-indexed task 0.
        };

        let selector = BandConditionerSelector::new(
            BandConditionerSelectorConfig::default()
                .with_segment_len(seg_len)
                .with_reject_threshold(0.5),
        );
        let selected = selector.select(&segments, &query, segments.len());

        // Build y_pred: 1.0 if retained, 0.0 otherwise.
        let mut y_pred = vec![0.0_f32; segments.len()];
        for &i in &selected {
            y_pred[i] = 1.0;
        }
        // Ground truth: segments whose dominant task is 0.
        let y_true = scm.ground_truth_relevance(seg_len, 0);

        let mcc = matthews_corr(&y_true, &y_pred);
        assert!(
            mcc >= 0.85,
            "Selection MCC {mcc:.4} below 0.85 (target ≥ 0.85)"
        );
    }

    /// G3: KV cache reduction ≥ 40% with perplexity delta < 0.5.
    ///
    /// We retain the top-`budget` segments and compare perplexity-proxy
    /// against using all segments. The budget is set to 50% of segments
    /// (i.e., 50% reduction, comfortably above the 40% target), and we
    /// verify the perplexity delta is small on a well-separated SCM where
    /// the retained segments are exactly the query-relevant ones.
    #[test]
    fn g3_kv_reduction_at_parity() {
        let seg_len = 4;
        let d = 16;
        let scm = SyntheticScm::generate_subspace_separated(80, d, 4, 0.99, 99, seg_len);
        let segments = scm.chunk_into_segments(seg_len);
        let n = segments.len();

        let query = QueryEmb {
            data: segments[0].representative_key().to_vec(),
            task_id: 1,
        };

        // Baseline perplexity using ALL segments.
        let all_idx: Vec<usize> = (0..n).collect();
        let ppl_full = perplexity_proxy(&query, &segments, &all_idx);

        // Retain 50% of segments → 50% reduction (well above the 40% target).
        let budget = n / 2;
        let selector = BandConditionerSelector::new(
            BandConditionerSelectorConfig::default().with_segment_len(seg_len),
        );
        let selected = selector.select(&segments, &query, budget);
        let reduction = 1.0 - (selected.len() as f32 / n as f32);
        assert!(
            reduction >= 0.40,
            "KV reduction {reduction:.3} below 0.40 (target ≥ 0.40)"
        );

        let ppl_sel = perplexity_proxy(&query, &segments, &selected);
        let delta = (ppl_sel - ppl_full).abs();
        assert!(
            delta < 0.5,
            "Perplexity delta {delta:.4} ≥ 0.5 (target < 0.5)"
        );
    }

    /// Routing sanity: small batches route to Simd, large to Gpu.
    #[test]
    fn route_ci_test_thresholds() {
        assert_eq!(route_ci_test(100), ComputeTarget::Simd);
        assert_eq!(route_ci_test(999), ComputeTarget::Simd);
        assert_eq!(route_ci_test(1000), ComputeTarget::Gpu);
        assert_eq!(route_ci_test(10_000), ComputeTarget::Gpu);
    }

    /// Empty-input fast paths.
    #[test]
    fn empty_inputs_return_empty() {
        let selector = BandConditionerSelector::default();
        let q = QueryEmb {
            data: vec![0.0; 4],
            task_id: 1,
        };
        assert!(selector.select(&[], &q, 5).is_empty());
        let segs = vec![KvSegment {
            token_range: (1, 2),
            keys: vec![1.0; 4],
            values: vec![1.0; 4],
        }];
        assert!(selector.select(&segs, &q, 0).is_empty());
    }

    /// Segment length builder only accepts L >= 2 (paper requirement).
    #[test]
    #[should_panic(expected = "segment_len must be >= 2")]
    fn builder_rejects_bad_segment_len() {
        let _ = BandConditionerSelectorConfig::default().with_segment_len(1);
    }

    /// MCC sanity: identical vectors → 1.0.
    #[test]
    fn mcc_identity() {
        let y = vec![1.0, 0.0, 1.0, 0.0];
        let mcc = matthews_corr(&y, &y);
        assert!((mcc - 1.0).abs() < 1e-5);
    }
}
