//! HOLA Hippocampal Exact KV Cache — **dynamic** (runtime D/W) variant.
//!
//! This is the production consumer variant of [`HippocampalCache`](super::hippocampal_cache::HippocampalCache).
//! The const-generic original uses `[f32; D]` / `[[f32; D]; W]` for stack
//! allocation and cache-line density — ideal for benchmarks and tests where D/W
//! are known at compile time. However, `forward_gdn2` uses **runtime**
//! `config.head_dim`, and for realistic dimensions (D=64–256, W=64) the
//! const-generic arrays would overflow the stack (256×64×4 = 64KB per array ×
//! 4 arrays = 256KB).
//!
//! `HippocampalCacheDyn` uses `Vec<f32>` (heap-allocated, flattened) for storage,
//! supporting any runtime D and W. The **read path is truly alloc-free** —
//! `read_cache_into` / `read_cache_into_fast` use pre-allocated internal scratch
//! buffers and write the output into a caller-provided `&mut [f32]`, preserving
//! the G4 GOAT gate property.
//!
//! # DRY note
//!
//! The heap helpers (`pack`, `unpack`, `sift_up`, `sift_down`) are shared from
//! `hippocampal_cache.rs` (made `pub(crate)`). The streaming-softmax helpers
//! cannot be shared because the const-generic versions operate on `&[f32; D]`
//! (fixed-size arrays) while the dynamic versions operate on `&[f32]` (slices).
//! The logic is identical; only the types differ.
//!
//! HOLA dynamic (runtime D/W) variant — the production consumer for forward_gdn2
//! which uses runtime config.head_dim. Same algorithm as HippocampalCache<D,W>
//! but Vec-based for runtime dimensions. Alloc-free read path (pre-allocated
//! scratch). Plan 395 Phase 5 (Issue 038 production wiring).

#![allow(clippy::needless_range_loop)]

use crate::hippocampal_cache::{pack, sift_down, sift_up, unpack};
use crate::simd::{simd_dot_f32, simd_scale_inplace};
use crate::types::rmsnorm_with_gamma;

// ─── HippocampalCacheDyn ──────────────────────────────────────────────────────

/// HOLA hippocampal exact KV cache — dynamic (runtime D/W) variant.
///
/// See [`hippocampal_cache`](super::hippocampal_cache) module docs for the
/// algorithm. This struct is functionally identical to `HippocampalCache<D, W>`
/// but uses `Vec` for storage, enabling runtime configuration of head dimension
/// `d` and cache capacity `w`.
///
/// # Alloc-free contract
///
/// - `observe()`: **zero allocation** per call. All storage is pre-allocated at
///   construction.
/// - `read_cache_into()` / `read_cache_into_fast()`: **zero allocation** per
///   call. Query/key normalization uses pre-allocated internal scratch buffers
///   (`qt_scratch`, `kt_scratch`); output is written into a caller-provided
///   `&mut [f32]`. The null-sink contribution is computed without allocating a
///   zero vector (val=0 ⟹ only `sum_exp` changes, `out` only rescales).
/// - `reset()`: **zero allocation**. Only resets `heap_len` to 0.
///
/// # `&mut self` on read methods
///
/// The const-generic `HippocampalCache::read_cache_into` takes `&self` because it
/// uses stack-allocated `[f32; D]` scratch. The dynamic variant takes `&mut self`
/// because it uses pre-allocated `Vec` scratch fields. In the production GDN2
/// forward pass, the cache is accessed mutably (sequential per-Q-head readout),
/// so this is not a constraint.
pub struct HippocampalCacheDyn {
    /// Head dimension (key/value/query vector length).
    d: usize,
    /// Cache capacity (max tokens retained).
    w: usize,
    /// Min-heap of `(score_bits, slot_idx)` packed into `u64`. Root = min score.
    /// Only `heap[..heap_len]` is occupied.
    heap: Vec<u64>,
    /// Number of occupied heap entries (0..=w).
    heap_len: usize,
    /// Stored keys, flattened: `keys[slot * d..(slot + 1) * d]`.
    keys: Vec<f32>,
    /// Stored values, flattened: `vals[slot * d..(slot + 1) * d]`.
    vals: Vec<f32>,
    /// Stored scores, indexed by slot.
    scores: Vec<f32>,
    /// Pre-normalized keys `RMSNorm_γ(k)`, flattened. Computed at observe time
    /// using the struct's `gamma` — enables the fast read path.
    keys_norm: Vec<f32>,
    /// Default γ for the decoupled cache-read RMSNorm. Model parameter (not
    /// runtime-learned). Initialized to ones (identity RMSNorm).
    gamma: Vec<f32>,
    // ── Pre-allocated scratch (avoids per-read allocation) ──
    /// Scratch for query normalization during read. `[d]`
    qt_scratch: Vec<f32>,
    /// Scratch for key normalization during slow-path read. `[d]`
    kt_scratch: Vec<f32>,
}

impl HippocampalCacheDyn {
    /// Create a new empty cache with the given γ vector.
    ///
    /// `gamma.len()` must equal `d`.
    pub fn new(d: usize, w: usize, gamma: Vec<f32>) -> Self {
        assert_eq!(gamma.len(), d, "gamma length must equal d");
        Self {
            d,
            w,
            heap: vec![0u64; w],
            heap_len: 0,
            keys: vec![0.0f32; w * d],
            vals: vec![0.0f32; w * d],
            scores: vec![0.0f32; w],
            keys_norm: vec![0.0f32; w * d],
            gamma,
            qt_scratch: vec![0.0f32; d],
            kt_scratch: vec![0.0f32; d],
        }
    }

    /// Create a new empty cache with γ = ones (identity RMSNorm — the modelless
    /// default).
    pub fn new_with_ones_gamma(d: usize, w: usize) -> Self {
        Self::new(d, w, vec![1.0f32; d])
    }

    /// Head dimension.
    #[inline]
    pub fn d(&self) -> usize {
        self.d
    }

    /// Cache capacity.
    #[inline]
    pub fn w(&self) -> usize {
        self.w
    }

    /// The cache's γ vector (for external key pre-normalization).
    ///
    /// Callers that want to amortize key normalization across multiple reads
    /// (e.g. the dual-path tree verifier, where ancestor keys are shared across
    /// sibling subtrees) can use this with `rmsnorm_with_gamma` to pre-normalize
    /// keys once, then call [`read_cache_into_fast_block_prenorm`].
    ///
    /// [`read_cache_into_fast_block_prenorm`]: Self::read_cache_into_fast_block_prenorm
    #[inline]
    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }

    /// Observe a token: compute `score = beta * residual_norm`, and if the
    /// score qualifies for the top-`w`, insert into the cache (evicting the
    /// lowest-score entry if full).
    ///
    /// `k.len()` and `v.len()` must be >= `d`.
    ///
    /// O(log W) per call. Zero allocation.
    pub fn observe(&mut self, k: &[f32], v: &[f32], beta: f32, residual_norm: f32) {
        debug_assert!(k.len() >= self.d, "k.len()={} < d={}", k.len(), self.d);
        debug_assert!(v.len() >= self.d, "v.len()={} < d={}", v.len(), self.d);

        let score = beta * residual_norm;
        if score.is_nan() || score < 0.0 {
            return;
        }
        let score_bits = score.to_bits();
        let d = self.d;

        if self.heap_len < self.w {
            // Fill phase: claim slot `heap_len` sequentially.
            let slot = self.heap_len;
            self.heap[slot] = pack(score_bits, slot as u32);
            sift_up(&mut self.heap[..], slot);
            self.heap_len += 1;
            self.keys[slot * d..(slot + 1) * d].copy_from_slice(&k[..d]);
            self.vals[slot * d..(slot + 1) * d].copy_from_slice(&v[..d]);
            self.scores[slot] = score;
            // Pre-normalize key for the fast read path.
            self.keys_norm[slot * d..(slot + 1) * d].copy_from_slice(&k[..d]);
            rmsnorm_with_gamma(
                &mut self.keys_norm[slot * d..(slot + 1) * d],
                &self.gamma[..],
            );
        } else if self.w > 0 {
            // Full: replace heap-min if new score is strictly higher.
            let (min_bits, min_slot) = unpack(self.heap[0]);
            if score_bits > min_bits {
                self.heap[0] = pack(score_bits, min_slot);
                sift_down(&mut self.heap[..], 0, self.heap_len);
                let slot = min_slot as usize;
                self.keys[slot * d..(slot + 1) * d].copy_from_slice(&k[..d]);
                self.vals[slot * d..(slot + 1) * d].copy_from_slice(&v[..d]);
                self.scores[slot] = score;
                self.keys_norm[slot * d..(slot + 1) * d].copy_from_slice(&k[..d]);
                rmsnorm_with_gamma(
                    &mut self.keys_norm[slot * d..(slot + 1) * d],
                    &self.gamma[..],
                );
            }
            // else: reject — new score too low.
        }
    }

    /// Read the cache via **softmax** attention with decoupled RMSNorm-γ.
    ///
    /// See `HippocampalCache::read_cache_into` for the algorithm. This is the
    /// dynamic (slice-based) variant. Zero allocation — uses internal scratch
    /// buffers and writes into `out`.
    ///
    /// `out.len()` must be >= `d`.
    pub fn read_cache_into(
        &mut self,
        q: &[f32],
        gamma: &[f32],
        block_kv: &[(&[f32], &[f32])],
        out: &mut [f32],
    ) {
        debug_assert!(
            out.len() >= self.d,
            "out.len()={} < d={}",
            out.len(),
            self.d
        );
        let d = self.d;

        if self.heap_len == 0 && block_kv.is_empty() {
            out[..d].fill(0.0);
            return;
        }

        let sqrt_d = (d as f32).sqrt();

        // Normalize query once (into pre-allocated scratch).
        let qt = &mut self.qt_scratch[..d];
        qt.copy_from_slice(&q[..d]);
        rmsnorm_with_gamma(qt, &gamma[..d]);

        let mut max_logit = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        out[..d].fill(0.0);

        // Cache slots.
        for i in 0..self.heap_len {
            let (_, slot) = unpack(self.heap[i]);
            let slot = slot as usize;
            let kt = &mut self.kt_scratch[..d];
            kt.copy_from_slice(&self.keys[slot * d..(slot + 1) * d]);
            rmsnorm_with_gamma(kt, &gamma[..d]);
            let logit = simd_dot_f32(qt, kt, d) / sqrt_d;
            streaming_softmax_acc_dyn(
                &mut out[..d],
                logit,
                &self.vals[slot * d..(slot + 1) * d],
                &mut max_logit,
                &mut sum_exp,
            );
        }

        // Block KV pairs (current chunk).
        for (k, v) in block_kv {
            let kv_len = d.min(k.len()).min(v.len());
            let kt = &mut self.kt_scratch[..d];
            kt[..kv_len].copy_from_slice(&k[..kv_len]);
            // Zero the rest (in case kv_len < d).
            for j in kv_len..d {
                kt[j] = 0.0;
            }
            rmsnorm_with_gamma(kt, &gamma[..d]);
            let logit = simd_dot_f32(qt, kt, d) / sqrt_d;
            streaming_softmax_acc_dyn(
                &mut out[..d],
                logit,
                &v[..kv_len],
                &mut max_logit,
                &mut sum_exp,
            );
        }

        // Null sink: logit = 0.0, v = [0; d]. Contributes weight but zero value.
        // Inlined without allocating a zero vector — when val is all zeros, the
        // streaming softmax only affects sum_exp and potentially rescales out.
        apply_null_sink(&mut out[..d], 0.0, &mut max_logit, &mut sum_exp);

        // Normalize by sum_exp.
        if sum_exp > 0.0 {
            let inv = 1.0 / sum_exp;
            simd_scale_inplace(&mut out[..d], inv);
        }
    }

    /// **Fast read path** — softmax read using the cache's pre-normalized keys
    /// (computed at observe time via `RMSNorm_γ`).
    ///
    /// See `HippocampalCache::read_cache_into_fast` for the algorithm. This is
    /// the dynamic (slice-based) variant. Zero allocation — uses internal
    /// scratch buffer and writes into `out`.
    ///
    /// `out.len()` must be >= `d`. This is the production read path when γ is
    /// fixed (the common case).
    pub fn read_cache_into_fast(&mut self, q: &[f32], out: &mut [f32]) {
        debug_assert!(
            out.len() >= self.d,
            "out.len()={} < d={}",
            out.len(),
            self.d
        );
        let d = self.d;

        if self.heap_len == 0 {
            out[..d].fill(0.0);
            return;
        }

        let sqrt_d = (d as f32).sqrt();

        // Normalize query once (into pre-allocated scratch).
        let qt = &mut self.qt_scratch[..d];
        qt.copy_from_slice(&q[..d]);
        rmsnorm_with_gamma(qt, &self.gamma[..d]);

        let mut max_logit = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        out[..d].fill(0.0);

        // Cache slots — keys already pre-normalized at observe time.
        for i in 0..self.heap_len {
            let (_, slot) = unpack(self.heap[i]);
            let slot = slot as usize;
            let logit = simd_dot_f32(qt, &self.keys_norm[slot * d..(slot + 1) * d], d) / sqrt_d;
            streaming_softmax_acc_dyn(
                &mut out[..d],
                logit,
                &self.vals[slot * d..(slot + 1) * d],
                &mut max_logit,
                &mut sum_exp,
            );
        }

        // Null sink: logit = 0.0, v = [0; d].
        apply_null_sink(&mut out[..d], 0.0, &mut max_logit, &mut sum_exp);

        if sum_exp > 0.0 {
            let inv = 1.0 / sum_exp;
            simd_scale_inplace(&mut out[..d], inv);
        }
    }

    /// **Fast read path with block KV** — like [`read_cache_into_fast`] but also
    /// incorporates external `block_kv` pairs (e.g. ancestor tree tokens) into the
    /// softmax candidate set. Cache slots use pre-normalized keys; block KV pairs
    /// are normalized on-the-fly. Zero allocation.
    ///
    /// This is the dual-path tree-verify read path (Plan 430 T1.3): the block KV
    /// are the ancestor draft tokens along the root→i path, and ancestor masking
    /// is enforced *by construction* (only ancestor tokens are passed in).
    ///
    /// `out.len()` must be >= `d`. Each `block_kv` pair's slices must have
    /// length >= `d` (only the first `d` elements are read).
    pub fn read_cache_into_fast_block(
        &mut self,
        q: &[f32],
        block_kv: &[(&[f32], &[f32])],
        out: &mut [f32],
    ) {
        debug_assert!(
            out.len() >= self.d,
            "out.len()={} < d={}",
            out.len(),
            self.d
        );
        let d = self.d;

        if self.heap_len == 0 && block_kv.is_empty() {
            out[..d].fill(0.0);
            return;
        }

        let sqrt_d = (d as f32).sqrt();

        // Normalize query once (into pre-allocated scratch).
        let qt = &mut self.qt_scratch[..d];
        qt.copy_from_slice(&q[..d]);
        rmsnorm_with_gamma(qt, &self.gamma[..d]);

        let mut max_logit = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        out[..d].fill(0.0);

        // Cache slots — keys already pre-normalized at observe time.
        for i in 0..self.heap_len {
            let (_, slot) = unpack(self.heap[i]);
            let slot = slot as usize;
            let logit = simd_dot_f32(qt, &self.keys_norm[slot * d..(slot + 1) * d], d) / sqrt_d;
            streaming_softmax_acc_dyn(
                &mut out[..d],
                logit,
                &self.vals[slot * d..(slot + 1) * d],
                &mut max_logit,
                &mut sum_exp,
            );
        }

        // Block KV pairs (ancestor tree tokens) — need RMSNorm (not pre-normalized).
        for (k, v) in block_kv {
            let kv_len = d.min(k.len()).min(v.len());
            let kt = &mut self.kt_scratch[..d];
            kt[..kv_len].copy_from_slice(&k[..kv_len]);
            for j in kv_len..d {
                kt[j] = 0.0;
            }
            rmsnorm_with_gamma(kt, &self.gamma[..d]);
            let logit = simd_dot_f32(qt, kt, d) / sqrt_d;
            streaming_softmax_acc_dyn(
                &mut out[..d],
                logit,
                &v[..kv_len],
                &mut max_logit,
                &mut sum_exp,
            );
        }

        // Null sink: logit = 0.0, v = [0; d].
        apply_null_sink(&mut out[..d], 0.0, &mut max_logit, &mut sum_exp);

        if sum_exp > 0.0 {
            let inv = 1.0 / sum_exp;
            simd_scale_inplace(&mut out[..d], inv);
        }
    }

    /// **Fast read path with pre-normalized block KV** — like
    /// [`read_cache_into_fast_block`] but assumes each `block_kv` key has already
    /// been RMSNorm-γ-normalized by the caller (via `rmsnorm_with_gamma(k,
    /// self.gamma())`). This skips the per-block_kv RMSNorm in the inner loop,
    /// amortizing normalization when the same key appears in multiple nodes'
    /// block_kv (the dual-path tree-verify case: ancestor keys are shared across
    /// sibling subtrees).
    ///
    /// **Caller contract:** each `block_kv` key (`block_kv[i].0`) MUST have been
    /// normalized via `rmsnorm_with_gamma(k, self.gamma())` before calling this
    /// method. Values (`block_kv[i].1`) are not normalized. Output is
    /// bit-identical to [`read_cache_into_fast_block`] when the caller's
    /// pre-normalization uses the same gamma.
    ///
    /// Zero allocation. `out.len()` must be >= `d`.
    pub fn read_cache_into_fast_block_prenorm(
        &mut self,
        q: &[f32],
        block_kv: &[(&[f32], &[f32])],
        out: &mut [f32],
    ) {
        debug_assert!(
            out.len() >= self.d,
            "out.len()={} < d={}",
            out.len(),
            self.d
        );
        let d = self.d;

        if self.heap_len == 0 && block_kv.is_empty() {
            out[..d].fill(0.0);
            return;
        }

        let sqrt_d = (d as f32).sqrt();

        // Normalize query once (into pre-allocated scratch).
        let qt = &mut self.qt_scratch[..d];
        qt.copy_from_slice(&q[..d]);
        rmsnorm_with_gamma(qt, &self.gamma[..d]);

        let mut max_logit = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        out[..d].fill(0.0);

        // Cache slots — keys already pre-normalized at observe time.
        for i in 0..self.heap_len {
            let (_, slot) = unpack(self.heap[i]);
            let slot = slot as usize;
            let logit = simd_dot_f32(qt, &self.keys_norm[slot * d..(slot + 1) * d], d) / sqrt_d;
            streaming_softmax_acc_dyn(
                &mut out[..d],
                logit,
                &self.vals[slot * d..(slot + 1) * d],
                &mut max_logit,
                &mut sum_exp,
            );
        }

        // Block KV pairs — keys already pre-normalized by caller. Skip RMSNorm.
        for (k_norm, v) in block_kv {
            let kv_len = d.min(k_norm.len()).min(v.len());
            let logit = simd_dot_f32(qt, &k_norm[..d], d) / sqrt_d;
            streaming_softmax_acc_dyn(
                &mut out[..d],
                logit,
                &v[..kv_len],
                &mut max_logit,
                &mut sum_exp,
            );
        }

        // Null sink: logit = 0.0, v = [0; d].
        apply_null_sink(&mut out[..d], 0.0, &mut max_logit, &mut sum_exp);

        if sum_exp > 0.0 {
            let inv = 1.0 / sum_exp;
            simd_scale_inplace(&mut out[..d], inv);
        }
    }

    /// Reset the cache to empty. Zero allocation.
    pub fn reset(&mut self) {
        self.heap_len = 0;
    }

    /// Number of tokens currently in the cache.
    #[inline]
    pub fn len(&self) -> usize {
        self.heap_len
    }

    /// Whether the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap_len == 0
    }

    /// Get the minimum score currently in the cache (heap root), or `None` if
    /// empty.
    #[inline]
    pub fn min_score(&self) -> Option<f32> {
        if self.heap_len == 0 {
            return None;
        }
        let (bits, _) = unpack(self.heap[0]);
        Some(f32::from_bits(bits))
    }
}

impl Clone for HippocampalCacheDyn {
    fn clone(&self) -> Self {
        Self {
            d: self.d,
            w: self.w,
            heap: self.heap.clone(),
            heap_len: self.heap_len,
            keys: self.keys.clone(),
            vals: self.vals.clone(),
            scores: self.scores.clone(),
            keys_norm: self.keys_norm.clone(),
            gamma: self.gamma.clone(),
            qt_scratch: self.qt_scratch.clone(),
            kt_scratch: self.kt_scratch.clone(),
        }
    }
}

// ─── Streaming softmax helper (dynamic) ──────────────────────────────────────

/// Streaming softmax accumulation (flash-attention style). Rescales the running
/// output and sum when a new max is encountered. Operates on slices — runtime
/// length, zero allocation.
///
/// This is the dynamic counterpart of `streaming_softmax_acc` in
/// `hippocampal_cache.rs`. The logic is identical; only the types differ
/// (`&mut [f32]` vs `&mut [f32; D]`).
#[inline]
fn streaming_softmax_acc_dyn(
    out: &mut [f32],
    logit: f32,
    val: &[f32],
    max_logit: &mut f32,
    sum_exp: &mut f32,
) {
    use crate::simd::fast_exp;

    let d = val.len();
    if logit > *max_logit {
        let rescale = fast_exp(*max_logit - logit);
        *sum_exp = *sum_exp * rescale + 1.0;
        for j in 0..d {
            out[j] = out[j] * rescale + val[j];
        }
        // Remaining out[d..] only rescales (val is zero beyond d).
        for j in d..out.len() {
            out[j] *= rescale;
        }
        *max_logit = logit;
    } else {
        let weight = fast_exp(logit - *max_logit);
        *sum_exp += weight;
        for j in 0..d {
            out[j] += weight * val[j];
        }
    }
}

/// Apply the null-sink contribution (logit=0.0, val=zeros) without allocating a
/// zero vector. When val is all zeros, the streaming softmax update simplifies
/// to: rescale `out` (if 0.0 > max_logit) and add `exp(0 - max_logit)` to
/// `sum_exp`. No val accumulation is needed since val=0.
#[inline]
fn apply_null_sink(out: &mut [f32], logit: f32, max_logit: &mut f32, sum_exp: &mut f32) {
    use crate::simd::fast_exp;
    if logit > *max_logit {
        let rescale = fast_exp(*max_logit - logit);
        *sum_exp = *sum_exp * rescale + 1.0;
        for v in out.iter_mut() {
            *v *= rescale;
        }
        *max_logit = logit;
    } else {
        let weight = fast_exp(logit - *max_logit);
        *sum_exp += weight;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Parity with const-generic version ───────────────────────────────────

    /// Verify the dynamic cache produces the same top-w set as the const-generic
    /// version on identical input.
    #[test]
    fn dyn_matches_const_generic_top_w() {
        use crate::hippocampal_cache::HippocampalCache;

        const D: usize = 8;
        const W: usize = 4;
        let mut rng = fastrand::Rng::with_seed(42);

        let mut cache_cg: HippocampalCache<D, W> = HippocampalCache::new_with_ones_gamma();
        let mut cache_dyn = HippocampalCacheDyn::new_with_ones_gamma(D, W);

        for i in 0..50 {
            let mut k = [0.0f32; D];
            let mut v = [0.0f32; D];
            for d in 0..D {
                k[d] = rng.f32();
                v[d] = rng.f32();
            }
            let score = 0.01 + 0.01 * i as f32;
            cache_cg.observe(&k, &v, score, 1.0);
            cache_dyn.observe(&k, &v, score, 1.0);
        }

        assert_eq!(cache_cg.len(), W);
        assert_eq!(cache_dyn.len(), W);

        // Compare surviving scores (sorted).
        let mut cg_scores: Vec<f32> = cache_cg.slots().map(|(_, _, _, s)| s).collect();
        cg_scores.sort_by(|a, b| a.total_cmp(b));

        // For the dyn version, extract scores from the heap.
        let mut dyn_scores: Vec<f32> = (0..cache_dyn.heap_len)
            .map(|i| {
                let (bits, _) = unpack(cache_dyn.heap[i]);
                f32::from_bits(bits)
            })
            .collect();
        dyn_scores.sort_by(|a, b| a.total_cmp(b));

        assert_eq!(cg_scores.len(), dyn_scores.len());
        for (a, b) in cg_scores.iter().zip(dyn_scores.iter()) {
            assert!((a - b).abs() < 1e-6, "score mismatch: {a} vs {b}");
        }
    }

    /// Verify the dynamic cache read produces the same output as the
    /// const-generic version on identical input.
    #[test]
    fn dyn_matches_const_generic_read() {
        use crate::hippocampal_cache::HippocampalCache;

        const D: usize = 8;
        const W: usize = 4;
        let mut rng = fastrand::Rng::with_seed(42);

        let mut cache_cg: HippocampalCache<D, W> = HippocampalCache::new_with_ones_gamma();
        let mut cache_dyn = HippocampalCacheDyn::new_with_ones_gamma(D, W);

        for i in 0..20 {
            let mut k = [0.0f32; D];
            let mut v = [0.0f32; D];
            for d in 0..D {
                k[d] = rng.f32();
                v[d] = rng.f32();
            }
            let score = 0.01 + 0.01 * i as f32;
            cache_cg.observe(&k, &v, score, 1.0);
            cache_dyn.observe(&k, &v, score, 1.0);
        }

        // Query.
        let mut q = [0.0f32; D];
        for d in 0..D {
            q[d] = rng.f32();
        }
        let gamma = [1.0f32; D];

        let mut out_cg = [0.0f32; D];
        let mut out_dyn = vec![0.0f32; D];
        cache_cg.read_cache_into(&q, &gamma, &[], &mut out_cg);
        cache_dyn.read_cache_into(&q, &gamma, &[], &mut out_dyn);

        for d in 0..D {
            assert!(
                (out_cg[d] - out_dyn[d]).abs() < 1e-6,
                "read mismatch at d={d}: cg={} dyn={}",
                out_cg[d],
                out_dyn[d]
            );
        }
    }

    /// Verify the fast read path matches the slow path.
    #[test]
    fn fast_read_matches_slow_read() {
        const D: usize = 8;
        const W: usize = 4;
        let mut rng = fastrand::Rng::with_seed(42);

        let mut cache = HippocampalCacheDyn::new_with_ones_gamma(D, W);

        for i in 0..20 {
            let mut k = vec![0.0f32; D];
            let mut v = vec![0.0f32; D];
            for d in 0..D {
                k[d] = rng.f32();
                v[d] = rng.f32();
            }
            let score = 0.01 + 0.01 * i as f32;
            cache.observe(&k, &v, score, 1.0);
        }

        let mut q = vec![0.0f32; D];
        for d in 0..D {
            q[d] = rng.f32();
        }
        let gamma = vec![1.0f32; D];

        let mut out_slow = vec![0.0f32; D];
        let mut out_fast = vec![0.0f32; D];
        cache.read_cache_into(&q, &gamma, &[], &mut out_slow);
        cache.read_cache_into_fast(&q, &mut out_fast);

        for d in 0..D {
            assert!(
                (out_slow[d] - out_fast[d]).abs() < 1e-6,
                "fast/slow mismatch at d={d}: slow={} fast={}",
                out_slow[d],
                out_fast[d]
            );
        }
    }

    /// Verify the pre-normalized block_kv read path matches the on-the-fly
    /// normalization path (`read_cache_into_fast_block`). This is the Plan 430
    /// G2 optimization correctness check: pre-normalizing keys with the cache's
    /// gamma must produce bit-identical output to normalizing inside the read.
    #[test]
    fn prenorm_block_matches_block() {
        const D: usize = 8;
        const W: usize = 4;
        let mut rng = fastrand::Rng::with_seed(99);

        let mut cache = HippocampalCacheDyn::new_with_ones_gamma(D, W);
        for i in 0..10 {
            let k: Vec<f32> = (0..D).map(|_| rng.f32()).collect();
            let v: Vec<f32> = (0..D).map(|_| rng.f32()).collect();
            cache.observe(&k, &v, 0.1 + 0.05 * i as f32, 1.0);
        }

        // External block_kv: 3 (k, v) pairs.
        let block_keys: Vec<Vec<f32>> = (0..3)
            .map(|_| (0..D).map(|_| rng.f32()).collect())
            .collect();
        let block_vals: Vec<Vec<f32>> = (0..3)
            .map(|_| (0..D).map(|_| rng.f32()).collect())
            .collect();
        let q: Vec<f32> = (0..D).map(|_| rng.f32()).collect();

        // Non-prenorm: raw block_kv.
        let block_kv_raw: Vec<(&[f32], &[f32])> = block_keys
            .iter()
            .zip(block_vals.iter())
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        let mut out_block = vec![0.0f32; D];
        cache.read_cache_into_fast_block(&q, &block_kv_raw, &mut out_block);

        // Prenorm: normalize keys externally, then use prenorm read.
        let gamma = cache.gamma().to_vec();
        let keys_norm: Vec<Vec<f32>> = block_keys
            .iter()
            .map(|k| {
                let mut kn = k.clone();
                rmsnorm_with_gamma(&mut kn, &gamma);
                kn
            })
            .collect();
        let block_kv_prenorm: Vec<(&[f32], &[f32])> = keys_norm
            .iter()
            .zip(block_vals.iter())
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        let mut out_prenorm = vec![0.0f32; D];
        cache.read_cache_into_fast_block_prenorm(&q, &block_kv_prenorm, &mut out_prenorm);

        for d in 0..D {
            assert!(
                (out_block[d] - out_prenorm[d]).abs() < 1e-6,
                "prenorm/block mismatch at d={d}: block={} prenorm={}",
                out_block[d],
                out_prenorm[d]
            );
        }
    }

    /// Empty cache read returns zeros (the no-regression guarantee).
    #[test]
    fn empty_cache_read_returns_zeros() {
        const D: usize = 8;
        let mut cache = HippocampalCacheDyn::new_with_ones_gamma(D, 4);

        let q = vec![0.5f32; D];
        let mut out = vec![1.0f32; D]; // non-zero to verify it gets zeroed
        cache.read_cache_into_fast(&q, &mut out);

        for d in 0..D {
            assert_eq!(out[d], 0.0, "empty cache should return zeros at d={d}");
        }
    }

    /// Reset clears the cache.
    #[test]
    fn reset_clears_cache() {
        const D: usize = 4;
        let mut cache = HippocampalCacheDyn::new_with_ones_gamma(D, 4);

        let k = vec![1.0f32; D];
        let v = vec![2.0f32; D];
        cache.observe(&k, &v, 0.5, 1.0);
        assert_eq!(cache.len(), 1);

        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        // Read after reset should return zeros.
        let mut out = vec![1.0f32; D];
        cache.read_cache_into_fast(&k, &mut out);
        for d in 0..D {
            assert_eq!(out[d], 0.0);
        }
    }

    /// Top-w eviction: the lowest-score entries are evicted when full.
    #[test]
    fn top_w_eviction() {
        const D: usize = 4;
        const W: usize = 3;
        let mut cache = HippocampalCacheDyn::new_with_ones_gamma(D, W);

        // Insert 5 entries with increasing scores.
        for i in 0..5 {
            let k = vec![i as f32; D];
            let v = vec![i as f32 * 10.0; D];
            let score = (i + 1) as f32; // scores: 1, 2, 3, 4, 5
            cache.observe(&k, &v, score, 1.0);
        }

        assert_eq!(cache.len(), W);

        // Top-3 by score are entries 4, 3, 2 (scores 5, 4, 3).
        let min_score = cache.min_score().unwrap();
        assert!(
            (min_score - 3.0).abs() < 1e-6,
            "min score should be 3.0, got {min_score}"
        );
    }

    /// NaN score is rejected.
    #[test]
    fn nan_score_rejected() {
        const D: usize = 4;
        let mut cache = HippocampalCacheDyn::new_with_ones_gamma(D, 4);

        let k = vec![1.0f32; D];
        let v = vec![2.0f32; D];
        cache.observe(&k, &v, f32::NAN, 1.0);
        assert_eq!(cache.len(), 0, "NaN score should be rejected");
    }

    /// Zero-W cache is a no-op (safe degenerate case).
    #[test]
    fn zero_w_cache_is_noop() {
        const D: usize = 4;
        let mut cache = HippocampalCacheDyn::new_with_ones_gamma(D, 0);

        let k = vec![1.0f32; D];
        let v = vec![2.0f32; D];
        cache.observe(&k, &v, 0.5, 1.0);
        assert_eq!(cache.len(), 0);

        let mut out = vec![1.0f32; D];
        cache.read_cache_into_fast(&k, &mut out);
        for d in 0..D {
            assert_eq!(out[d], 0.0);
        }
    }
}
