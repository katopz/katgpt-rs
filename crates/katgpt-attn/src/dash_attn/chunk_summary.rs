//! Learned chunk summaries via head_cls vectors.
//!
//! Each KV-head carries `R` learnable `head_cls` reference queries (ZHA
//! reference-query shape, arXiv:2610.42112 — riir-train Issue 666) used to
//! summarize a chunk of keys via local SDPA:
//! `k̂_r = softmax(q̂_r · K_chunk / √d) · K_chunk` per reference `r`, with
//! the chunk's routing tangent `max_r(q · k̂_r/√d · γ + b̂_r)`
//! ([`tangent_logit_max_r`]). The default `R = 1` layout is the original
//! single-query form byte-for-byte (additive layout law, Issue 666 T2).
//! When `head_cls` is zero-initialized (default), this degenerates to mean
//! pooling for backward compatibility. Trained reference values arrive via
//! the `zha_chunk_sidecar` feature (BLAKE3-verified sidecar file); a build
//! without the feature (or a boot without the sidecar) serves the incumbent
//! byte-identically.

// ---------------------------------------------------------------------------
// ChunkSummaryQuery
// ---------------------------------------------------------------------------

/// Per-KV-head learned query for chunk summarization.
///
/// `head_cls` layout: `[n_refs][n_kv_head][head_dim]`, **ref-major** — at the
/// default `R = 1` this is the original `[n_kv_head * head_dim]` layout
/// byte-for-byte (the additive-layout law, riir-train Issue 666 T2). Each
/// reference query `q̂_r` summarizes a chunk independently
/// (`k̂_r = softmax(q̂_r · K_chunk / √d) · K_chunk`); the ZHA routing tangent
/// for the chunk is `max_r(q · k̂_r/√d · γ + b̂_r)` — see
/// [`tangent_logit_max_r`].
///
/// Zero-initialized by default → mean pooling (backward-compatible); after
/// training, the references learn to attend to the most informative
/// positions within each chunk. Trained values arrive via the
/// `zha_chunk_sidecar` feature (BLAKE3-verified sidecar file); sidecar-absent
/// serving is byte-identical to the incumbent (the free G3 anchor).
#[derive(Clone)]
pub struct ChunkSummaryQuery {
    /// Learned class token embeddings: flat `[n_refs * n_kv_head * head_dim]`,
    /// ref-major.
    pub head_cls: Vec<f32>,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// Reference-query count `R` (the ZHA indexer's per-KV-group shared
    /// queries; arXiv:2610.42112). Private — every constructor pins it; read
    /// via [`Self::n_refs`].
    n_refs: usize,
    /// Cached result of scanning `head_cls` for all-zeros.
    /// Updated on construction; call [`recompute_zero_init_cache`] after any
    /// direct mutation of `head_cls`.
    zero_init_cache: bool,
}

impl ChunkSummaryQuery {
    /// Create with zero-initialized `head_cls` (mean-pooling mode), `R = 1`.
    pub fn new(n_kv_head: usize, head_dim: usize) -> Self {
        Self {
            head_cls: vec![0.0; n_kv_head * head_dim],
            n_kv_head,
            head_dim,
            n_refs: 1,
            zero_init_cache: true,
        }
    }

    /// Create with `R` zero-initialized reference queries per KV head
    /// (mean-pooling mode; every reference produces the same mean-pool
    /// summary and the same `ln(chunk_size)` entropy, so `R > 1` at zero-init
    /// is route-equivalent to `R = 1` — the tangent max over identical refs
    /// is the incumbent's own logit).
    ///
    /// # Panics
    /// Panics if `n_refs == 0` (an empty reference set has no tangent).
    pub fn with_n_refs(n_refs: usize, n_kv_head: usize, head_dim: usize) -> Self {
        assert!(
            n_refs >= 1,
            "n_refs must be >= 1 (an empty reference set has no tangent logit)"
        );
        Self {
            head_cls: vec![0.0; n_refs * n_kv_head * head_dim],
            n_kv_head,
            head_dim,
            n_refs,
            zero_init_cache: true,
        }
    }

    /// Create with random initialization for training (`R = 1`).
    pub fn new_random(
        n_kv_head: usize,
        head_dim: usize,
        rng: &mut katgpt_core::types::Rng,
    ) -> Self {
        let scale = (2.0 / head_dim as f32).sqrt();
        let mut head_cls = Vec::with_capacity(n_kv_head * head_dim);
        for _ in 0..n_kv_head * head_dim {
            head_cls.push(rng.normal() * scale);
        }
        Self {
            head_cls,
            n_kv_head,
            head_dim,
            n_refs: 1,
            zero_init_cache: false,
        }
    }

    /// Reference-query count `R`.
    #[inline]
    pub fn n_refs(&self) -> usize {
        self.n_refs
    }

    /// Crate-internal assembly from raw ref-major parts (the sidecar lane's
    /// post-verification constructor). `head_cls.len()` must equal
    /// `n_refs * n_kv_head * head_dim` — the caller verified the wire shape
    /// before calling.
    pub(crate) fn from_ref_major(
        head_cls: Vec<f32>,
        n_refs: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Self {
        debug_assert_eq!(head_cls.len(), n_refs * n_kv_head * head_dim);
        let mut query = Self {
            head_cls,
            n_kv_head,
            head_dim,
            n_refs,
            zero_init_cache: false,
        };
        query.recompute_zero_init_cache();
        query
    }

    /// Get the reference-0 query slice for a specific head (the incumbent
    /// single-query reading; every `R = 1` caller keeps its exact layout and
    /// values).
    #[inline]
    pub fn head_query(&self, head_idx: usize) -> &[f32] {
        let start = head_idx * self.head_dim;
        &self.head_cls[start..start + self.head_dim]
    }

    /// Get reference `ref_idx`'s query slice for a specific head.
    ///
    /// Ref-major addressing: `(ref_idx * n_kv_head + head_idx) * head_dim`.
    ///
    /// # Panics
    /// Panics if the computed range is out of bounds (a caller mixing `R`,
    /// `n_kv_head` or `head_dim` across construction and consumption is a
    /// contract violation, not a degenerate input).
    #[inline]
    pub fn ref_query(&self, ref_idx: usize, head_idx: usize) -> &[f32] {
        let start = (ref_idx * self.n_kv_head + head_idx) * self.head_dim;
        debug_assert!(
            ref_idx < self.n_refs,
            "ref_query({ref_idx}) out of range (n_refs = {})",
            self.n_refs
        );
        &self.head_cls[start..start + self.head_dim]
    }

    /// Check if head_cls is effectively zero (mean-pooling mode).
    ///
    /// Returns a cached bool — O(1) instead of scanning the entire vector.
    /// Call [`recompute_zero_init_cache`](Self::recompute_zero_init_cache) after any direct mutation of `head_cls`.
    #[inline]
    pub fn is_zero_init(&self) -> bool {
        self.zero_init_cache
    }

    /// Recompute the cached `is_zero_init` result after mutating `head_cls`.
    pub fn recompute_zero_init_cache(&mut self) {
        self.zero_init_cache = self.head_cls.iter().all(|&v| v == 0.0);
    }
}

// ---------------------------------------------------------------------------
// ChunkSummaryCache
// ---------------------------------------------------------------------------

/// Cache for completed chunk summaries: `[n_chunks][n_kv_head][head_dim]`.
///
/// Populated during prefill; reused during decode for routing.
///
/// Also stores the per-chunk-per-head **entropy bias** `b'_c` (HiLS Prop 3.1,
/// Issue 044) alongside each summary key. At zero-init `head_cls`, every
/// chunk's entropy is `ln(chunk_size)` (constant) so the bias is a no-op for
/// routing rankings; it activates when `head_cls` becomes non-trivial.
#[derive(Clone)]
pub struct ChunkSummaryCache {
    /// Summary keys indexed by chunk: `[n_chunks][n_kv_head][head_dim]`.
    pub summaries: Vec<Vec<Vec<f32>>>,
    /// Entropy biases `b'_c` indexed by chunk: `[n_chunks][n_kv_head]`.
    /// (Issue 044 / Research 399.) Dormant at zero-init (constant across chunks).
    pub entropy_biases: Vec<Vec<f32>>,
    pub n_kv_head: usize,
    pub head_dim: usize,
}

impl ChunkSummaryCache {
    /// Create an empty cache.
    pub fn new(n_kv_head: usize, head_dim: usize) -> Self {
        Self {
            summaries: Vec::new(),
            entropy_biases: Vec::new(),
            n_kv_head,
            head_dim,
        }
    }

    /// Pre-allocate for a known number of chunks.
    pub fn allocate(&mut self, n_chunks: usize) {
        // Reuse existing allocation if already the right size
        if self.summaries.len() == n_chunks {
            // Clear in-place without deallocating
            for chunk in &mut self.summaries {
                for head in chunk.iter_mut() {
                    head.fill(0.0);
                }
            }
            for chunk in &mut self.entropy_biases {
                chunk.fill(0.0);
            }
        } else {
            self.summaries = (0..n_chunks)
                .map(|_| {
                    (0..self.n_kv_head)
                        .map(|_| vec![0.0; self.head_dim])
                        .collect()
                })
                .collect();
            self.entropy_biases = (0..n_chunks).map(|_| vec![0.0; self.n_kv_head]).collect();
        }
    }

    /// Append a single chunk summary (one entry per KV head) and its entropy
    /// biases (one scalar per KV head).
    pub fn append(&mut self, summary: Vec<Vec<f32>>, entropy: Vec<f32>) {
        debug_assert_eq!(summary.len(), self.n_kv_head);
        debug_assert_eq!(entropy.len(), self.n_kv_head);
        for head_summary in &summary {
            debug_assert_eq!(head_summary.len(), self.head_dim);
        }
        self.summaries.push(summary);
        self.entropy_biases.push(entropy);
    }

    /// View summaries for a specific chunk (all heads).
    pub fn view(&self, chunk_idx: usize) -> &[Vec<f32>] {
        &self.summaries[chunk_idx]
    }

    /// View entropy biases for a specific chunk (all heads).
    /// Returns `[b'_c(head_0), b'_c(head_1), ...]`.
    pub fn view_entropy(&self, chunk_idx: usize) -> &[f32] {
        &self.entropy_biases[chunk_idx]
    }

    /// Number of cached chunks.
    pub fn n_chunks(&self) -> usize {
        self.summaries.len()
    }

    /// Clear for a new sequence.
    pub fn reset(&mut self) {
        self.summaries.clear();
        self.entropy_biases.clear();
    }
}

// ---------------------------------------------------------------------------
// Chunk summarization kernel
// ---------------------------------------------------------------------------

/// Summarize a chunk via local SDPA: k̄_c = softmax(q̄ · K_chunk / √d) · K_chunk.
///
/// At zero-init `head_cls`: returns mean pooling of all keys in the chunk.
///
/// # Arguments
/// * `query` - The chunk summary query holding learned `head_cls` vectors.
/// * `chunk_keys` - Flat key buffer `[chunk_size * head_dim]` for one KV head.
/// * `chunk_size` - Number of tokens in this chunk.
/// * `head_idx` - Which KV head to summarize.
/// * `head_dim` - Dimension per head (must match query and key layout).
///
/// Prefer [`summarize_chunk_into`] on hot paths to avoid per-call allocation.
/// For the entropy bias `b'_c` (HiLS Prop 3.1, Issue 044), use
/// [`summarize_chunk_with_entropy`] or [`summarize_chunk_into_with_entropy`].
#[inline]
pub fn summarize_chunk(
    query: &ChunkSummaryQuery,
    chunk_keys: &[f32],
    chunk_size: usize,
    head_idx: usize,
    head_dim: usize,
) -> Vec<f32> {
    let (summary, _entropy) =
        summarize_chunk_with_entropy(query, chunk_keys, chunk_size, head_idx, head_dim);
    summary
}

/// Like [`summarize_chunk`] but also returns the entropy bias `b'_c` (HiLS
/// Prop 3.1, Issue 044).
///
/// Returns `(summary_key, entropy_bias)` where `entropy_bias = -Σ p_t log p_t`
/// over the softmax weights used to compute the summary key. At zero-init
/// this equals `ln(chunk_size)` (uniform distribution). For a near-degenerate
/// (peaked) distribution it approaches `0`.
#[inline]
pub fn summarize_chunk_with_entropy(
    query: &ChunkSummaryQuery,
    chunk_keys: &[f32],
    chunk_size: usize,
    head_idx: usize,
    head_dim: usize,
) -> (Vec<f32>, f32) {
    let mut out = vec![0.0f32; head_dim];
    let mut scores_buf = vec![0.0f32; chunk_size.max(1)];
    let mut entropy = 0.0f32;
    summarize_chunk_into_with_entropy(
        query,
        chunk_keys,
        chunk_size,
        head_idx,
        head_dim,
        &mut out,
        &mut scores_buf,
        &mut entropy,
    );
    (out, entropy)
}

/// Zero-alloc variant of [`summarize_chunk`].
///
/// Writes the summary into `out[..head_dim]` and uses `scores_buf` as scratch.
/// Does not compute the entropy bias; use [`summarize_chunk_into_with_entropy`]
/// for that.
pub fn summarize_chunk_into(
    query: &ChunkSummaryQuery,
    chunk_keys: &[f32],
    chunk_size: usize,
    head_idx: usize,
    head_dim: usize,
    out: &mut [f32],
    scores_buf: &mut [f32],
) {
    let mut entropy = 0.0f32;
    summarize_chunk_into_with_entropy(
        query,
        chunk_keys,
        chunk_size,
        head_idx,
        head_dim,
        out,
        scores_buf,
        &mut entropy,
    );
}

/// Zero-alloc variant that also computes the HiLS Prop 3.1 entropy bias
/// `b'_c = -Σ p_t log p_t` (Issue 044).
///
/// Writes the summary into `out[..head_dim]`, the entropy bias into
/// `entropy_out`, and uses `scores_buf` as scratch. The entropy is computed
/// as one reduction over the same softmax weights already used for the
/// summary key — no second attention pass, no allocation.
///
/// At zero-init `head_cls`, the softmax is uniform and `*entropy_out = ln(chunk_size)`
/// (constant across chunks → no ranking change). For a peaked distribution
/// (learned query concentrating on one token), `*entropy_out ≈ 0`.
#[allow(
    clippy::too_many_arguments,
    reason = "hot-path summarizer; args are distinct query/chunk/out/scratch buffers"
)]
pub fn summarize_chunk_into_with_entropy(
    query: &ChunkSummaryQuery,
    chunk_keys: &[f32],
    chunk_size: usize,
    head_idx: usize,
    head_dim: usize,
    out: &mut [f32],
    scores_buf: &mut [f32],
    entropy_out: &mut f32,
) {
    let q = query.head_query(head_idx);
    summarize_ref_into(
        query.is_zero_init(),
        q,
        chunk_keys,
        chunk_size,
        head_dim,
        out,
        scores_buf,
        entropy_out,
    );
}

/// Summarize one chunk with EVERY reference query of a head
/// (`k̂_r = Attn(q̂_r, K_C)` for `r ∈ 0..R` — the ZHA reference-query shape,
/// riir-train Issue 666 T2).
///
/// Writes each reference's summary into `outs` (flat `[R * head_dim]`,
/// ref-major: reference `r`'s key at `outs[r * head_dim..]`) and each
/// reference's entropy bias into `entropies_out` (`[R]`). Zero-alloc.
///
/// At a zero-init query every reference produces the same mean-pool summary
/// and the same `ln(chunk_size)` entropy — the tangent max over identical
/// refs ([`tangent_logit_max_r`]) is then exactly the `R = 1` incumbent's
/// logit, so the multi-ref reading never perturbs the untrained route.
///
/// Prefer [`summarize_chunk_into_with_entropy`] when `query.n_refs() == 1`
/// (the same arithmetic, one indirection less).
#[allow(
    clippy::too_many_arguments,
    reason = "hot-path summarizer; args are distinct query/chunk/out/scratch buffers"
)]
pub fn summarize_chunk_refs_into_with_entropy(
    query: &ChunkSummaryQuery,
    chunk_keys: &[f32],
    chunk_size: usize,
    head_idx: usize,
    head_dim: usize,
    outs: &mut [f32],
    entropies_out: &mut [f32],
    scores_buf: &mut [f32],
) {
    let n_refs = query.n_refs();
    debug_assert_eq!(
        outs.len(),
        n_refs * head_dim,
        "outs must be [R * head_dim], ref-major"
    );
    debug_assert_eq!(entropies_out.len(), n_refs, "entropies_out must be [R]");
    let is_zero = query.is_zero_init();
    for (ref_idx, entropy_out) in entropies_out.iter_mut().enumerate() {
        let out_off = ref_idx * head_dim;
        summarize_ref_into(
            is_zero,
            query.ref_query(ref_idx, head_idx),
            chunk_keys,
            chunk_size,
            head_dim,
            &mut outs[out_off..out_off + head_dim],
            scores_buf,
            entropy_out,
        );
    }
}

/// Allocating twin of [`summarize_chunk_refs_into_with_entropy`].
/// Returns `(flat refs summaries [R * head_dim] ref-major, entropies [R])`.
pub fn summarize_chunk_refs_with_entropy(
    query: &ChunkSummaryQuery,
    chunk_keys: &[f32],
    chunk_size: usize,
    head_idx: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n_refs = query.n_refs();
    let mut outs = vec![0.0f32; n_refs * head_dim];
    let mut entropies = vec![0.0f32; n_refs];
    let mut scores_buf = vec![0.0f32; chunk_size.max(1)];
    summarize_chunk_refs_into_with_entropy(
        query,
        chunk_keys,
        chunk_size,
        head_idx,
        head_dim,
        &mut outs,
        &mut entropies,
        &mut scores_buf,
    );
    (outs, entropies)
}

/// One reference's chunk summary + entropy bias (the shared body of every
/// summarizer variant). `query_is_zero` selects the mean-pool fallback;
/// `q` is the caller's query slice (reference `r`'s, head `head_idx`).
#[allow(
    clippy::too_many_arguments,
    reason = "hot-path summarizer body; args are distinct query/chunk/out/scratch buffers"
)]
fn summarize_ref_into(
    query_is_zero: bool,
    q: &[f32],
    chunk_keys: &[f32],
    chunk_size: usize,
    head_dim: usize,
    out: &mut [f32],
    scores_buf: &mut [f32],
    entropy_out: &mut f32,
) {
    let hd = head_dim;

    // Check if query is zero → mean pooling fallback.
    // Entropy of the uniform distribution over `chunk_size` elements is
    // `ln(chunk_size)` — constant across chunks, so routing rankings are
    // unchanged at zero-init (the "dormant" guarantee, Issue 044 T4).
    if query_is_zero {
        mean_pool_keys_into(chunk_keys, chunk_size, hd, out);
        *entropy_out = (chunk_size.max(1) as f32).ln();
        return;
    }

    // Compute attention scores: q · k_t / sqrt(d)
    let scale = 1.0 / (hd as f32).sqrt();
    debug_assert!(scores_buf.len() >= chunk_size);
    // Cache the number of full hd-wide chunks; reused for the remainder tail.
    let n_full_chunks = chunk_keys.len() / hd;
    // Use the crate SIMD dot kernel (8-wide FMA accumulator) instead of the
    // iterator `.zip().map().sum()` form, which carries a single fadd
    // dependency chain that blocks LLVM auto-vectorization.
    for (t, key_chunk) in chunk_keys.chunks_exact(hd).enumerate() {
        let dot = katgpt_core::simd::simd_dot_f32(q, key_chunk, hd);
        scores_buf[t] = dot * scale;
    }
    // Handle remainder if hd doesn't evenly divide
    let remainder_start = n_full_chunks * hd;
    if remainder_start < chunk_keys.len() {
        let t = n_full_chunks;
        let mut dot = 0.0f32;
        for d in 0..hd {
            let k_val = if remainder_start + d < chunk_keys.len() {
                chunk_keys[remainder_start + d]
            } else {
                0.0
            };
            dot += q[d] * k_val;
        }
        if t < scores_buf.len() {
            scores_buf[t] = dot * scale;
        }
    }

    // Softmax (numerically stable) — scores_buf[..chunk_size] now holds p_t.
    softmax_inplace(&mut scores_buf[..chunk_size]);

    // Entropy bias b'_c = -Σ p_t log p_t (HiLS Prop 3.1, Issue 044).
    // One reduction over the already-L1-resident softmax weights. Zero alloc.
    let mut entropy = 0.0f32;
    for &p in &scores_buf[..chunk_size] {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    *entropy_out = entropy;

    // Weighted sum of keys → summary (SIMD scale-acc inner loop)
    out[..hd].fill(0.0);
    for (score, key_chunk) in scores_buf[..chunk_size]
        .iter()
        .zip(chunk_keys.chunks_exact(hd))
    {
        katgpt_core::simd::simd_fused_scale_acc(&mut out[..hd], key_chunk, *score, hd);
    }
}

/// The ZHA routing tangent logit for ONE chunk: `s = max_r(q · k̂_r/√d · γ +
/// b̂_r)` (research 021 / riir-train Issue 666 — the tiny reference-query
/// score the KL distillation trains against).
///
/// `chunk_refs` is the chunk's flat reference-summary block `[R * head_dim]`
/// (ref-major, exactly what [`summarize_chunk_refs_into_with_entropy`] writes);
/// `chunk_ref_entropies` carries the per-reference entropy biases `b̂_r`
/// (`[R]`; a shorter slice disables the bias for the missing tail, mirroring
/// the router's own entropy handling). The scale mirrors the router's chunk
/// logit `z = q · k̄/√d · γ + b'_c` (`routing::compute_chunk_logits_into`),
/// so at `R = 1` this is that logit read back bit-for-bit.
///
/// NaN propagation: a NaN ref logit POISONS the chunk (returns NaN) — never
/// silently dropped by the max (a corrupted trained sidecar must be visible
/// in the route, not laundered by the selection).
///
/// # Panics
/// Panics on `chunk_refs.is_empty()` or `head_dim == 0` — scoring an empty
/// reference set is a caller contract violation, not a degenerate input.
#[inline]
pub fn tangent_logit_max_r(
    query: &[f32],
    chunk_refs: &[f32],
    chunk_ref_entropies: &[f32],
    head_dim: usize,
    scaling_factor: f32,
) -> f32 {
    tangent_logit_max_r_indexed(
        query,
        chunk_refs,
        chunk_ref_entropies,
        head_dim,
        scaling_factor,
    )
    .0
}

/// [`tangent_logit_max_r`] plus the winning reference index. The max is a
/// strict `>` walk, so ties keep the FIRST index — the determinism law's
/// fixed strict tie rule (the trainer's subgradient consumes this same
/// first-index argmax).
///
/// # Panics
/// Panics on `chunk_refs.is_empty()` or `head_dim == 0`.
#[inline]
pub fn tangent_logit_max_r_indexed(
    query: &[f32],
    chunk_refs: &[f32],
    chunk_ref_entropies: &[f32],
    head_dim: usize,
    scaling_factor: f32,
) -> (f32, usize) {
    assert!(
        !chunk_refs.is_empty(),
        "tangent logit over an empty reference set"
    );
    assert!(head_dim >= 1, "tangent logit needs head_dim >= 1");
    let n_refs = chunk_refs.len() / head_dim;
    debug_assert_eq!(
        chunk_refs.len(),
        n_refs * head_dim,
        "chunk_refs must be a whole number of head_dim rows"
    );
    let scale = 1.0 / (head_dim as f32).sqrt() * scaling_factor;
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for r in 0..n_refs {
        let k_hat = &chunk_refs[r * head_dim..(r + 1) * head_dim];
        let mut logit = katgpt_core::simd::simd_dot_f32(query, k_hat, head_dim) * scale;
        if r < chunk_ref_entropies.len() {
            logit += chunk_ref_entropies[r];
        }
        // NaN poisoning: a NaN logit must SURFACE, never be skipped by the
        // strict-> selection (f32::max semantics would launder it).
        if logit.is_nan() {
            return (logit, r);
        }
        if logit > best {
            best = logit;
            best_idx = r;
        }
    }
    (best, best_idx)
}

/// Mean pooling over chunk keys (zero-init fallback).
#[allow(dead_code)]
fn mean_pool_keys(chunk_keys: &[f32], chunk_size: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; head_dim];
    mean_pool_keys_into(chunk_keys, chunk_size, head_dim, &mut out);
    out
}

/// Zero-alloc mean pooling into pre-allocated buffer.
fn mean_pool_keys_into(chunk_keys: &[f32], chunk_size: usize, head_dim: usize, out: &mut [f32]) {
    out[..head_dim].fill(0.0);
    if chunk_size == 0 {
        return;
    }
    // Accumulate all tokens via SIMD add
    for t in 0..chunk_size {
        let k_start = t * head_dim;
        katgpt_core::simd::simd_add_inplace(
            &mut out[..head_dim],
            &chunk_keys[k_start..k_start + head_dim],
        );
    }
    // Scale once at the end
    let inv = 1.0 / chunk_size as f32;
    katgpt_core::simd::simd_scale_inplace(&mut out[..head_dim], inv);
}

/// In-place softmax with max subtraction for numerical stability.
///
/// Uses reciprocal-multiply (1 division + N multiplies) instead of N divisions.
///
/// SIMD path: all three passes (max, exp+sum, normalize) use the Cephes-backed
/// vectorized kernels from `katgpt_core::simd`. The exp+sum pass is fused into
/// a single buffer traversal via `simd_exp_sum_inplace` (4-/8-wide depending
/// on target). ~1.7× faster than libm `exp` on aarch64 for the exp pass; the
/// max + scale passes auto-vectorize. The numerical result differs from the
/// libm path by ≤1 ULP per element (Cephes accuracy floor); softmax
/// normalization absorbs this into the final reciprocal-multiply.
fn softmax_inplace(scores: &mut [f32]) {
    use katgpt_core::simd::{
        simd_add_scalar_inplace, simd_exp_sum_inplace, simd_max_f32, simd_scale_inplace,
    };

    if scores.is_empty() {
        return;
    }
    let max_val = simd_max_f32(scores);
    simd_add_scalar_inplace(scores, -max_val);
    let sum_exp = simd_exp_sum_inplace(scores);
    if sum_exp > 0.0 {
        simd_scale_inplace(scores, 1.0 / sum_exp);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const N_KV_HEAD: usize = 2;
    const HEAD_DIM: usize = 4;

    #[test]
    fn test_chunk_summary_query_new_zero() {
        let query = ChunkSummaryQuery::new(N_KV_HEAD, HEAD_DIM);
        assert!(query.is_zero_init());
        assert_eq!(query.head_cls.len(), N_KV_HEAD * HEAD_DIM);
        assert_eq!(query.n_kv_head, N_KV_HEAD);
        assert_eq!(query.head_dim, HEAD_DIM);
    }

    #[test]
    fn test_chunk_summary_query_head_slices() {
        let mut query = ChunkSummaryQuery::new(N_KV_HEAD, HEAD_DIM);
        // Write different values per head
        query.head_cls[0..HEAD_DIM].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        query.head_cls[HEAD_DIM..2 * HEAD_DIM].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

        let h0 = query.head_query(0);
        assert_eq!(h0, &[1.0, 2.0, 3.0, 4.0]);
        let h1 = query.head_query(1);
        assert_eq!(h1, &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_chunk_summary_cache_allocate() {
        let mut cache = ChunkSummaryCache::new(N_KV_HEAD, HEAD_DIM);
        cache.allocate(3);
        assert_eq!(cache.n_chunks(), 3);
        // Each chunk has n_kv_head entries, each of length head_dim
        for chunk in &cache.summaries {
            assert_eq!(chunk.len(), N_KV_HEAD);
            for head_summary in chunk {
                assert_eq!(head_summary.len(), HEAD_DIM);
                assert!(head_summary.iter().all(|&x| x == 0.0));
            }
        }
    }

    #[test]
    fn test_chunk_summary_cache_append() {
        let mut cache = ChunkSummaryCache::new(N_KV_HEAD, HEAD_DIM);
        let summary = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let entropy = vec![0.5_f32.ln(), 2.0_f32.ln()];
        cache.append(summary.clone(), entropy.clone());
        assert_eq!(cache.n_chunks(), 1);
        assert_eq!(cache.view(0)[0], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.view_entropy(0), &entropy);
    }

    #[test]
    fn test_chunk_summary_cache_reset() {
        let mut cache = ChunkSummaryCache::new(N_KV_HEAD, HEAD_DIM);
        cache.append(vec![vec![0.0; HEAD_DIM]; N_KV_HEAD], vec![0.0; N_KV_HEAD]);
        cache.append(vec![vec![0.0; HEAD_DIM]; N_KV_HEAD], vec![0.0; N_KV_HEAD]);
        assert_eq!(cache.n_chunks(), 2);
        cache.reset();
        assert_eq!(cache.n_chunks(), 0);
    }

    #[test]
    fn test_summarize_chunk_mean_pool_fallback() {
        let query = ChunkSummaryQuery::new(1, HEAD_DIM);
        // 3 tokens, each with known keys
        let chunk_keys: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 2.0, 0.0, 0.0, // token 1
            0.0, 0.0, 3.0, 0.0, // token 2
        ];
        let summary = summarize_chunk(&query, &chunk_keys, 3, 0, HEAD_DIM);
        // Mean of the 3 vectors: [1/3, 2/3, 1.0, 0.0]
        let expected = [1.0 / 3.0, 2.0 / 3.0, 1.0, 0.0];
        for (i, (&got, &exp)) in summary.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "mean pool mismatch at dim {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_summarize_chunk_with_learned_query() {
        let mut query = ChunkSummaryQuery::new(1, HEAD_DIM);
        // Use large magnitude so softmax concentrates sharply on matching token.
        // [0, 100, 0, 0] · token 1 [0, 2, 0, 0] = 200 (dominant).
        query.head_cls[0..HEAD_DIM].copy_from_slice(&[0.0, 100.0, 0.0, 0.0]);
        query.recompute_zero_init_cache();

        let chunk_keys: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 2.0, 0.0, 0.0, // token 1
            0.0, 0.0, 3.0, 0.0, // token 2
        ];
        let summary = summarize_chunk(&query, &chunk_keys, 3, 0, HEAD_DIM);

        // Attention should heavily concentrate on token 1 → summary ≈ [0, 2, 0, 0]
        let expected = [0.0, 2.0, 0.0, 0.0];
        for (i, (&got, &exp)) in summary.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-2,
                "learned query mismatch at dim {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_summarize_chunk_single_token() {
        let query = ChunkSummaryQuery::new(1, HEAD_DIM);
        let chunk_keys: Vec<f32> = vec![4.0, 5.0, 6.0, 7.0];
        let summary = summarize_chunk(&query, &chunk_keys, 1, 0, HEAD_DIM);
        // Single token mean pool = the token itself
        assert_eq!(summary, &[4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_mean_pool_keys_empty_chunk() {
        let result = mean_pool_keys(&[], 0, 4);
        assert_eq!(result, &[0.0; 4]);
    }

    #[test]
    fn test_softmax_inplace_uniform() {
        let mut scores = vec![1.0, 1.0, 1.0];
        softmax_inplace(&mut scores);
        // All equal → uniform 1/3
        for &s in &scores {
            assert!((s - 1.0 / 3.0).abs() < 1e-6, "expected 1/3, got {s}");
        }
    }

    #[test]
    fn test_softmax_inplace_peaked() {
        let mut scores = vec![10.0, 0.0, 0.0];
        softmax_inplace(&mut scores);
        let dominant = scores[0];
        assert!(
            dominant > 0.99,
            "expected near-1.0 for dominant, got {dominant}"
        );
        assert!(scores[1] < 0.01);
        assert!(scores[2] < 0.01);
        let sum: f32 = scores.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax must sum to 1.0, got {sum}"
        );
    }

    // ── Issue 044: entropy bias `b'_c` tests (T4, T5) ────────────────────

    /// T4: At zero-init `head_cls`, the softmax is uniform → `b'_c = ln(S)`
    /// exactly. This is constant across all chunks of the same size, so
    /// routing rankings are bit-identical to the pre-entropy behavior.
    #[test]
    fn test_entropy_bias_zero_init_is_ln_chunk_size() {
        let query = ChunkSummaryQuery::new(1, HEAD_DIM);
        let chunk_keys: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 2.0, 0.0, 0.0, // token 1
            0.0, 0.0, 3.0, 0.0, // token 2
        ];
        let (_summary, entropy) = summarize_chunk_with_entropy(&query, &chunk_keys, 3, 0, HEAD_DIM);
        // Uniform distribution over 3 tokens: H = ln(3).
        let expected = 3.0f32.ln();
        assert!(
            (entropy - expected).abs() < 1e-6,
            "zero-init entropy should be ln(3) = {expected}, got {entropy}"
        );
    }

    /// T4 continuation: single-token chunk → entropy = ln(1) = 0.
    #[test]
    fn test_entropy_bias_single_token_is_zero() {
        let query = ChunkSummaryQuery::new(1, HEAD_DIM);
        let chunk_keys: Vec<f32> = vec![4.0, 5.0, 6.0, 7.0];
        let (_summary, entropy) = summarize_chunk_with_entropy(&query, &chunk_keys, 1, 0, HEAD_DIM);
        assert!(
            entropy.abs() < 1e-6,
            "single-token entropy should be 0, got {entropy}"
        );
    }

    /// T4 continuation: two zero-init chunks of the same size → same entropy
    /// → no ranking change (the "dormant at zero-init" guarantee).
    #[test]
    fn test_entropy_bias_dormant_constant_across_chunks() {
        let query = ChunkSummaryQuery::new(1, HEAD_DIM);
        let chunk_a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let chunk_b: Vec<f32> = vec![5.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0];
        let (_s_a, e_a) = summarize_chunk_with_entropy(&query, &chunk_a, 3, 0, HEAD_DIM);
        let (_s_b, e_b) = summarize_chunk_with_entropy(&query, &chunk_b, 3, 0, HEAD_DIM);
        assert_eq!(
            e_a, e_b,
            "same-size zero-init chunks must have identical entropy"
        );
    }

    /// T5: With a non-trivial (peaked) `head_cls`, the softmax concentrates on
    /// one token → entropy approaches 0. A uniform-logit chunk has higher
    /// entropy, so the peaked chunk gets a smaller bias.
    #[test]
    fn test_entropy_bias_peaked_query_near_zero() {
        let mut query = ChunkSummaryQuery::new(1, HEAD_DIM);
        // [0, 100, 0, 0] · token 1 [0, 2, 0, 0] = 200 (dominant) → near-degenerate.
        query.head_cls[0..HEAD_DIM].copy_from_slice(&[0.0, 100.0, 0.0, 0.0]);
        query.recompute_zero_init_cache();

        let chunk_keys: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 2.0, 0.0, 0.0, // token 1
            0.0, 0.0, 3.0, 0.0, // token 2
        ];
        let (_summary, entropy) = summarize_chunk_with_entropy(&query, &chunk_keys, 3, 0, HEAD_DIM);
        // Near-degenerate distribution → entropy ≈ 0.
        assert!(
            entropy < 0.01,
            "peaked query should yield near-zero entropy, got {entropy}"
        );
    }

    /// T5 continuation: compare a peaked chunk vs a uniform-logit chunk.
    /// The peaked chunk must have strictly lower entropy.
    #[test]
    fn test_entropy_bias_peaked_lower_than_uniform() {
        // Uniform-logit query: small magnitude → spread softmax → high entropy.
        let mut query_uniform = ChunkSummaryQuery::new(1, HEAD_DIM);
        query_uniform.head_cls[0..HEAD_DIM].copy_from_slice(&[0.01, 0.02, 0.0, 0.0]);
        query_uniform.recompute_zero_init_cache();

        // Peaked query: large magnitude → concentrated softmax → low entropy.
        let mut query_peaked = ChunkSummaryQuery::new(1, HEAD_DIM);
        query_peaked.head_cls[0..HEAD_DIM].copy_from_slice(&[0.0, 100.0, 0.0, 0.0]);
        query_peaked.recompute_zero_init_cache();

        let chunk_keys: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let (_s_u, e_uniform) =
            summarize_chunk_with_entropy(&query_uniform, &chunk_keys, 3, 0, HEAD_DIM);
        let (_s_p, e_peaked) =
            summarize_chunk_with_entropy(&query_peaked, &chunk_keys, 3, 0, HEAD_DIM);
        assert!(
            e_peaked < e_uniform,
            "peaked entropy ({e_peaked}) must be < uniform entropy ({e_uniform})"
        );
    }

    /// Sanity: the entropy variant produces the same summary key as the
    /// non-entropy variant (the entropy computation must not perturb `out`).
    #[test]
    fn test_summarize_chunk_with_entropy_matches_plain() {
        let mut query = ChunkSummaryQuery::new(1, HEAD_DIM);
        query.head_cls[0..HEAD_DIM].copy_from_slice(&[0.0, 100.0, 0.0, 0.0]);
        query.recompute_zero_init_cache();
        let chunk_keys: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let plain = summarize_chunk(&query, &chunk_keys, 3, 0, HEAD_DIM);
        let (entropy_summary, _e) =
            summarize_chunk_with_entropy(&query, &chunk_keys, 3, 0, HEAD_DIM);
        for (i, (&a, &b)) in plain.iter().zip(entropy_summary.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "dim {i}: plain={a}, entropy={b}");
        }
    }

    // ── Issue 666 T2 — the R reference-query layout ─────────────────────

    /// Additive-layout law, part 1: `R = 1` reference addressing is the
    /// original flat layout byte-for-byte.
    #[test]
    fn test_ref_layout_r1_is_the_incumbent_layout() {
        let mut query = ChunkSummaryQuery::new(N_KV_HEAD, HEAD_DIM);
        assert_eq!(query.n_refs(), 1);
        query.head_cls[0..HEAD_DIM].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        query.head_cls[HEAD_DIM..2 * HEAD_DIM].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        for h in 0..N_KV_HEAD {
            assert_eq!(query.ref_query(0, h), query.head_query(h));
        }
    }

    /// Additive-layout law, part 2: the multi-ref summarizer at `R = 1` is
    /// the single-ref path BIT-FOR-BIT (summary + entropy).
    #[test]
    fn test_summarize_refs_r1_bit_identical_to_single_ref() {
        let mut query = ChunkSummaryQuery::new(1, HEAD_DIM);
        query.head_cls[0..HEAD_DIM].copy_from_slice(&[0.0, 100.0, 0.0, 0.0]);
        query.recompute_zero_init_cache();
        let chunk_keys: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let (single, single_e) = summarize_chunk_with_entropy(&query, &chunk_keys, 3, 0, HEAD_DIM);
        let (refs, refs_e) = summarize_chunk_refs_with_entropy(&query, &chunk_keys, 3, 0, HEAD_DIM);
        assert_eq!(refs.len(), HEAD_DIM);
        assert_eq!(
            single.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            refs.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            "R=1 refs summary must be bit-identical to the single-ref path"
        );
        assert_eq!(single_e.to_bits(), refs_e[0].to_bits());
    }

    /// Each reference row of the multi-ref summarizer equals the single-ref
    /// summarizer run with that reference's query.
    #[test]
    fn test_summarize_refs_r4_matches_per_ref_single() {
        let mut query = ChunkSummaryQuery::with_n_refs(4, 1, HEAD_DIM);
        assert_eq!(query.n_refs(), 4);
        assert_eq!(query.head_cls.len(), 4 * HEAD_DIM);
        // Ref 0 stays zero (mean-pool anchor); refs 1..3 peak at different
        // tokens.
        query.head_cls[HEAD_DIM..2 * HEAD_DIM].copy_from_slice(&[100.0, 0.0, 0.0, 0.0]);
        query.head_cls[2 * HEAD_DIM..3 * HEAD_DIM].copy_from_slice(&[0.0, 100.0, 0.0, 0.0]);
        query.head_cls[3 * HEAD_DIM..4 * HEAD_DIM].copy_from_slice(&[0.0, 0.0, 100.0, 0.0]);
        query.recompute_zero_init_cache();
        assert!(!query.is_zero_init());

        let chunk_keys: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let (refs, ref_entropies) =
            summarize_chunk_refs_with_entropy(&query, &chunk_keys, 3, 0, HEAD_DIM);
        assert_eq!(ref_entropies.len(), 4);
        for r in 0..4 {
            // Single-ref oracle: a fresh R=1 query carrying ref r's slice.
            let mut oracle = ChunkSummaryQuery::new(1, HEAD_DIM);
            oracle.head_cls.copy_from_slice(query.ref_query(r, 0));
            oracle.recompute_zero_init_cache();
            let (expect, expect_e) =
                summarize_chunk_with_entropy(&oracle, &chunk_keys, 3, 0, HEAD_DIM);
            let got = &refs[r * HEAD_DIM..(r + 1) * HEAD_DIM];
            assert_eq!(
                expect.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                got.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "ref {r} row must match its single-ref oracle"
            );
            assert_eq!(expect_e.to_bits(), ref_entropies[r].to_bits());
        }
        // Ref 0 (zero) is the mean-pool anchor.
        let mean = [1.0 / 3.0, 2.0 / 3.0, 1.0, 0.0];
        for (i, (&got, &exp)) in refs[0..HEAD_DIM].iter().zip(mean.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "ref 0 mean-pool dim {i}: {got} vs {exp}"
            );
        }
    }

    /// Zero-init `R > 1`: every reference is the same mean-pool summary with
    /// the same `ln(chunk_size)` entropy — route-equivalent to `R = 1`.
    #[test]
    fn test_summarize_refs_zero_init_all_refs_mean_pool() {
        let query = ChunkSummaryQuery::with_n_refs(3, N_KV_HEAD, HEAD_DIM);
        assert!(query.is_zero_init());
        let chunk_keys: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let (refs, ref_entropies) =
            summarize_chunk_refs_with_entropy(&query, &chunk_keys, 3, 0, HEAD_DIM);
        let mean = [1.0 / 3.0, 2.0 / 3.0, 1.0, 0.0];
        for r in 0..3 {
            let got = &refs[r * HEAD_DIM..(r + 1) * HEAD_DIM];
            for (i, (&g, &exp)) in got.iter().zip(mean.iter()).enumerate() {
                assert!((g - exp).abs() < 1e-6, "ref {r} dim {i}: {g} vs {exp}");
            }
            assert!((ref_entropies[r] - 3.0_f32.ln()).abs() < 1e-6);
        }
    }

    /// `tangent_logit_max_r` picks the max of the manual per-ref logits and
    /// the indexed twin keeps the FIRST index on ties.
    #[test]
    fn test_tangent_logit_max_r_manual_and_first_index_tie() {
        let query = [1.0, 0.0, 0.0, 0.0];
        let hd = 4_usize;
        // Refs: dot products 1.0, 3.0, 3.0 (γ=1 → logits 0.5, 1.5, 1.5 at
        // scale 1/√4 = 0.5; entropies add +0.0, +1.0, +1.0 → 0.5, 2.5, 2.5).
        let refs = [1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let ent = [0.0, 1.0, 1.0];
        let (logit, idx) = tangent_logit_max_r_indexed(&query, &refs, &ent, hd, 1.0);
        assert_eq!(idx, 1, "tie between refs 1 and 2 must keep the FIRST index");
        assert!((logit - 2.5).abs() < 1e-6, "max logit: got {logit}");
        assert!((tangent_logit_max_r(&query, &refs, &ent, hd, 1.0) - logit).abs() < 1e-6);
    }

    /// Zero-init `R > 1` tangent identity: max over identical refs + equal
    /// `ln(n)` biases is exactly the `R = 1` incumbent logit.
    #[test]
    fn test_tangent_logit_zero_init_identity_with_r1_incumbent() {
        let query = [0.5, -0.25, 1.0, 2.0];
        let mean: Vec<f32> = vec![1.0 / 3.0, 2.0 / 3.0, 1.0, 0.0];
        let ln3 = 3.0_f32.ln();
        let hd = 4_usize;
        let r1 = tangent_logit_max_r(&query, &mean, &[ln3], hd, 1.0);
        let mut r3_refs = Vec::new();
        r3_refs.extend_from_slice(&mean);
        r3_refs.extend_from_slice(&mean);
        r3_refs.extend_from_slice(&mean);
        let r3 = tangent_logit_max_r(&query, &r3_refs, &[ln3, ln3, ln3], hd, 1.0);
        assert_eq!(
            r1.to_bits(),
            r3.to_bits(),
            "zero-init R>1 tangent must equal the R=1 incumbent"
        );
    }

    /// NaN propagation: one NaN ref poisons the chunk logit (never silently
    /// skipped by the max — a corrupted sidecar must surface).
    #[test]
    fn test_tangent_logit_nan_ref_poisons() {
        let query = [1.0, 0.0, 0.0, 0.0];
        let nan = f32::NAN;
        let refs = [1.0, 0.0, 0.0, 0.0, nan, 0.0, 0.0, 0.0];
        let (logit, idx) = tangent_logit_max_r_indexed(&query, &refs, &[], 4, 1.0);
        assert!(
            logit.is_nan(),
            "NaN ref must poison the tangent, not be skipped"
        );
        assert_eq!(idx, 1);
    }

    #[should_panic(expected = "empty reference set")]
    #[test]
    fn test_tangent_logit_empty_refs_panics() {
        let query = [1.0, 0.0, 0.0, 0.0];
        let _ = tangent_logit_max_r(&query, &[], &[], 4, 1.0);
    }

    #[should_panic(expected = "n_refs must be >= 1")]
    #[test]
    fn test_with_n_refs_zero_panics() {
        let _ = ChunkSummaryQuery::with_n_refs(0, N_KV_HEAD, HEAD_DIM);
    }
}
