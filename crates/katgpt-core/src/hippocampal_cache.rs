//! HOLA — Hippocampal Exact KV Cache for Linear Attention (Plan 395, Research 378).
//!
//! Distilled from Cui 2026, *"A Hippocampus for Linear Attention: An Exact Memory
//! for What the Recurrent State Forgets"* ([arxiv 2607.02303](https://arxiv.org/abs/2607.02303)).
//! Research note: [`katgpt-rs/.research/378_HOLA_Hippocampal_Exact_KV_for_Linear_Attention.md`](../../.research/378_HOLA_Hippocampal_Exact_KV_for_Linear_Attention.md).
//!
//! # What this is
//!
//! A **surprise-evicted bounded exact KV cache** that complements the fixed-size
//! GDN2 recurrent state. The cache stores the top-`w` tokens by intrinsic
//! delta-rule write magnitude `β·‖e‖` (computed *for free* by any delta-rule
//! update — both `β` and `‖e‖` are already on the hot path), and reads them via
//! a **decoupled RMSNorm-γ** sharpened softmax that turns the exact copies into
//! near-argmax retrieval instead of a soft average.
//!
//! The cache is **modelless** at inference time: the eviction policy is
//! parameter-free (no learned scorer), and γ is a model parameter (like any
//! RMSNorm γ), not a runtime-learned value.
//!
//! # Backing store: binary heap
//!
//! The top-`w` set is maintained by a **binary min-heap** keyed by
//! `f32::to_bits(score)` (deterministic, NaN-safe — scores are always
//! non-negative since `β ≥ 0` and `‖e‖ ≥ 0`). Each heap entry packs
//! `(score_bits: u32) || (slot_idx: u32)` into a `u64` for cache-line density.
//!
//! - `observe()`: O(log w) insert or evict-replace.
//! - `read_cache_into()`: O(w · d) attention-style read.
//!
//! The heap was chosen over a sorted-vec variant after the T2.3 micro-bench
//! (Plan 395 Phase 2). At w=64, the binary heap wins on observe (O(log w)
//! sift vs O(w) linear scan), while read is O(w) for both.
//!
//! # Read path: softmax (not sigmoid)
//!
//! **AGENTS.md deviation — documented and justified.** The project rule
//! "Use sigmoid not softmax" applies to **gating/routing** decisions (independent
//! per-option gates). The HOLA cache read is an **attention/retrieval** operation
//! — competitive selection over a candidate set — where softmax is the
//! principled choice (it normalizes, enabling near-argmax retrieval). The paper
//! uses softmax for exactly this reason, and the codebase already uses
//! `softmax`/`softmax_scaled` for attention computation.
//!
//! An independent sigmoid-gated variant (`read_cache_into_sigmoid`) is also
//! provided for comparison. On the G4 retrieval gate, softmax recovers ≥6/8
//! needles (cosine ≥ 0.8) while sigmoid-gated recovers 0/8 — the noise from
//! non-matching slots (each contributing `sigmoid(0) = 0.5 · v_j`) overwhelms
//! the signal. This is a fundamental property: sigmoid is not competitive, so
//! it cannot suppress irrelevant candidates. See G4 verdict for details.
//!
//! # Modelless γ unblock (§3.5)
//!
//! The decoupled RMSNorm-γ needs a value. The paper trains it end-to-end. Per
//! the modelless-unblock protocol, we exhaust deterministic constructions first:
//!
//! - **γ = ones** (identity RMSNorm — the default). Still sharpens because
//!   RMSNorm scales q/k to unit-RMS, giving `‖q̃‖ ≈ √d`, so logits are
//!   `≈ √d · cos(θ)` instead of `τ · cos(θ)`.
//! - **Per-key norm rescale**: `γ_i = √d / max(‖k_i‖, ε)` — the deterministic
//!   sharpening analog. Use `read_cache_into_per_key_rescale`.
//!
//! If both fail G4, γ-tuning is deferred to riir-train (tracked in the G5 issue).
//!
//! HOLA Hippocampal Exact KV Cache — surprise-evicted (β·‖e‖) bounded KV cache with
//! decoupled RMSNorm-γ read (Plan 395, Research 378, arxiv 2607.02303). Complements
//! the GDN2 fixed-size recurrent state with a top-w exact KV set for long-range
//! retrieval. STAYS OPT-IN — G1–G4 GOAT PASS + consumer wiring PASS (modelless
//! gain, Issue 038 production wiring in forward_gdn2 via HippocampalCacheDyn);
//! promotion deferred to G5 riir-train gate (perplexity on real text). Pure
//! stdlib + katgpt-types.

#![allow(clippy::needless_range_loop)]

use crate::sigmoid;
use crate::simd::{simd_dot_f32, simd_scale_inplace};
use crate::types::rmsnorm_with_gamma;

// ─── Heap packing helpers ───────────────────────────────────────────────────

/// Pack `(score_bits, slot_idx)` into a single `u64`. Score bits occupy the
/// high 32 bits so that `u64` comparison orders by score first (min-heap by
/// score), then by slot index for deterministic tie-breaking.
#[inline(always)]
pub(crate) fn pack(score_bits: u32, slot: u32) -> u64 {
    ((score_bits as u64) << 32) | (slot as u64)
}

/// Unpack a `u64` heap entry into `(score_bits, slot_idx)`.
#[inline(always)]
pub(crate) fn unpack(packed: u64) -> (u32, u32) {
    let score_bits = (packed >> 32) as u32;
    let slot = packed as u32;
    (score_bits, slot)
}

/// Sift-up for a min-heap stored as `&mut [u64]`. Bubbles the element at index
/// `i` up until the heap property is restored. O(log n).
#[inline]
pub(crate) fn sift_up(heap: &mut [u64], mut i: usize) {
    while i > 0 {
        let parent = (i - 1) / 2;
        if heap[i] < heap[parent] {
            heap.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

/// Sift-down for a min-heap stored as `&mut [u64]`. Bubbles the element at
/// index `i` down until the heap property is restored. O(log n).
#[inline]
pub(crate) fn sift_down(heap: &mut [u64], mut i: usize, len: usize) {
    loop {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        let mut smallest = i;
        if left < len && heap[left] < heap[smallest] {
            smallest = left;
        }
        if right < len && heap[right] < heap[smallest] {
            smallest = right;
        }
        if smallest == i {
            break;
        }
        heap.swap(i, smallest);
        i = smallest;
    }
}

// ─── HippocampalCache ─────────────────────────────────────────────────────────

/// HOLA hippocampal exact KV cache — top-`w` by `β·‖e‖` with decoupled
/// RMSNorm-γ read.
///
/// Const generics:
/// - `D` — head dimension (key/value/query vector length).
/// - `W` — cache capacity (max tokens retained).
///
/// The cache is a **pure observer** of the delta-rule update — it never
/// perturbs the GDN2 state. When wired into GDN2 (Phase 3), the recurrent step
/// calls `observe()` after each write; the cache is read separately via
/// `read_cache_into()` and the result is added to the GDN2 readout.
///
/// # Determinism contract (HOLA §3.3)
///
/// The same multiset of `(k, v, score)` triples fed in any order produces the
/// **same** surviving cache set (order-independent top-`w` maintenance). This
/// holds when scores are distinct at the `w`/`w+1` boundary; exact ties at the
/// boundary are inherently order-dependent (some token must be evicted).
pub struct HippocampalCache<const D: usize, const W: usize> {
    /// Min-heap of `(score_bits, slot_idx)` packed into `u64`. Root = min score.
    /// Only `heap[..heap_len]` is occupied.
    heap: [u64; W],
    /// Number of occupied heap entries (0..=W).
    heap_len: usize,
    /// Stored keys, indexed by slot.
    keys: [[f32; D]; W],
    /// Stored values, indexed by slot.
    vals: [[f32; D]; W],
    /// Stored scores, indexed by slot (mirrors `heap` score bits for convenience).
    scores: [f32; W],
    /// Pre-normalized keys `RMSNorm_γ(k)` indexed by slot. Computed at observe
    /// time using the struct's `gamma` — enables the fast read path
    /// (`read_cache_into_fast`) which skips per-read RMSNorm on keys.
    keys_norm: [[f32; D]; W],
    /// Default γ for the decoupled cache-read RMSNorm. Model parameter (not
    /// runtime-learned). Initialized to ones (identity RMSNorm).
    gamma: [f32; D],
}

impl<const D: usize, const W: usize> HippocampalCache<D, W> {
    /// Create a new empty cache with the given γ vector.
    pub fn new(gamma: [f32; D]) -> Self {
        Self {
            heap: [0u64; W],
            heap_len: 0,
            keys: [[0.0f32; D]; W],
            vals: [[0.0f32; D]; W],
            scores: [0.0f32; W],
            keys_norm: [[0.0f32; D]; W],
            gamma,
        }
    }

    /// Create a new empty cache with γ = ones (identity RMSNorm — the modelless default).
    pub fn new_with_ones_gamma() -> Self {
        Self::new([1.0f32; D])
    }

    /// Observe a token: compute `score = beta * residual_norm`, and if the
    /// score qualifies for the top-`w`, insert into the cache (evicting the
    /// lowest-score entry if full).
    ///
    /// O(log W) per call. The key and value are copied into the cache. If the
    /// score does not qualify (heap full and score ≤ heap min), this is a no-op.
    ///
    /// # NaN safety
    ///
    /// If `score` is NaN or negative (shouldn't happen with β ≥ 0, ‖e‖ ≥ 0),
    /// the observation is silently rejected — NaN never enters the heap.
    pub fn observe(&mut self, k: &[f32; D], v: &[f32; D], beta: f32, residual_norm: f32) {
        let score = beta * residual_norm;
        if score.is_nan() || score < 0.0 {
            return;
        }
        let score_bits = score.to_bits();

        if self.heap_len < W {
            // Fill phase: claim slot `heap_len` sequentially.
            let slot = self.heap_len;
            self.heap[slot] = pack(score_bits, slot as u32);
            sift_up(&mut self.heap[..], slot);
            self.heap_len += 1;
            self.keys[slot] = *k;
            self.vals[slot] = *v;
            self.scores[slot] = score;
            // Pre-normalize key for the fast read path.
            self.keys_norm[slot].copy_from_slice(k);
            rmsnorm_with_gamma(&mut self.keys_norm[slot][..], &self.gamma[..]);
        } else if W > 0 {
            // Full: replace heap-min if new score is strictly higher.
            let (min_bits, min_slot) = unpack(self.heap[0]);
            if score_bits > min_bits {
                self.heap[0] = pack(score_bits, min_slot);
                sift_down(&mut self.heap[..], 0, self.heap_len);
                let slot = min_slot as usize;
                self.keys[slot] = *k;
                self.vals[slot] = *v;
                self.scores[slot] = score;
                self.keys_norm[slot].copy_from_slice(k);
                rmsnorm_with_gamma(&mut self.keys_norm[slot][..], &self.gamma[..]);
            }
            // else: reject — new score too low.
        }
    }

    /// Read the cache via **softmax** attention with decoupled RMSNorm-γ.
    ///
    /// This is the paper-faithful read path. The output is a softmax-weighted
    /// sum of cached values plus any block KV pairs:
    ///
    /// ```text
    /// q̃ = RMSNorm_γ(q),  k̃_j = RMSNorm_γ(k_j)
    /// logit_j = q̃ · k̃_j / √d
    /// out = Σ_j softmax_j(logit) · v_j
    /// ```
    ///
    /// The candidate set `V_t` = cache slots ∪ `block_kv` ∪ null sink. The null
    /// sink has `v = 0` (contributes weight but no value — needed for correct
    /// normalization).
    ///
    /// Uses a streaming (flash-attention style) softmax — **zero heap
    /// allocation** on the read path. O((W + |block| + 1) · D).
    ///
    /// # Why softmax, not sigmoid (AGENTS.md deviation — documented)
    ///
    /// The AGENTS.md rule "Use sigmoid not softmax" applies to gating/routing
    /// decisions (independent per-option gates). The cache read is an
    /// attention/retrieval operation — competitive selection over a candidate
    /// set — where softmax normalizes, enabling near-argmax retrieval.
    /// Sigmoid-gated accumulation cannot suppress noise from non-matching slots
    /// (each contributes `sigmoid(0) = 0.5 · v_j`), making it unsuitable for
    /// retrieval. See `read_cache_into_sigmoid` for the comparison variant and
    /// the G4 gate results.
    pub fn read_cache_into(
        &self,
        q: &[f32; D],
        gamma: &[f32; D],
        block_kv: &[(&[f32], &[f32])],
        out: &mut [f32; D],
    ) {
        if self.heap_len == 0 && block_kv.is_empty() {
            *out = [0.0; D];
            return;
        }

        let sqrt_d = (D as f32).sqrt();

        // Normalize query once.
        let mut qt = [0.0f32; D];
        qt.copy_from_slice(q);
        rmsnorm_with_gamma(&mut qt[..], &gamma[..]);

        // Streaming softmax (flash-attention style): track running max, sum_exp,
        // and rescaled output. No buffer needed.
        let mut max_logit = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        out.fill(0.0);

        // Cache slots.
        for i in 0..self.heap_len {
            let (_, slot) = unpack(self.heap[i]);
            let slot = slot as usize;
            let mut kt = [0.0f32; D];
            kt.copy_from_slice(&self.keys[slot]);
            rmsnorm_with_gamma(&mut kt[..], &gamma[..]);
            let logit = simd_dot_f32(&qt, &kt, D) / sqrt_d;
            streaming_softmax_acc(out, logit, &self.vals[slot], &mut max_logit, &mut sum_exp);
        }

        // Block KV pairs (current chunk).
        for (k, v) in block_kv {
            let d = D.min(k.len()).min(v.len());
            let mut kt = [0.0f32; D];
            kt[..d].copy_from_slice(&k[..d]);
            rmsnorm_with_gamma(&mut kt[..], &gamma[..]);
            let logit = simd_dot_f32(&qt, &kt, D) / sqrt_d;
            streaming_softmax_acc_slice(out, logit, &v[..d], &mut max_logit, &mut sum_exp);
        }

        // Null sink: logit = 0.0, v = [0; D]. Contributes weight but zero value.
        streaming_softmax_acc(out, 0.0, &[0.0f32; D], &mut max_logit, &mut sum_exp);

        // Normalize by sum_exp.
        if sum_exp > 0.0 {
            let inv = 1.0 / sum_exp;
            simd_scale_inplace(&mut out[..], inv);
        }
    }

    /// **Fast read path** — softmax read using the cache's pre-normalized keys
    /// (computed at observe time via `RMSNorm_γ`). Skips per-read RMSNorm on
    /// cached keys, cutting the read from O(W·D) RMSNorm + O(W·D) dot to just
    /// O(W·D) dot + O(D) for the query RMSNorm.
    ///
    /// This is the production read path when γ is fixed (the common case).
    /// Use `read_cache_into` when you need a different γ at read time.
    pub fn read_cache_into_fast(
        &self,
        q: &[f32; D],
        block_kv: &[(&[f32], &[f32])],
        out: &mut [f32; D],
    ) {
        if self.heap_len == 0 && block_kv.is_empty() {
            *out = [0.0; D];
            return;
        }

        let sqrt_d = (D as f32).sqrt();

        // Normalize query once.
        let mut qt = [0.0f32; D];
        qt.copy_from_slice(q);
        rmsnorm_with_gamma(&mut qt[..], &self.gamma[..]);

        let mut max_logit = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        out.fill(0.0);

        // Cache slots — keys already pre-normalized at observe time.
        for i in 0..self.heap_len {
            let (_, slot) = unpack(self.heap[i]);
            let slot = slot as usize;
            let logit = simd_dot_f32(&qt, &self.keys_norm[slot], D) / sqrt_d;
            streaming_softmax_acc(out, logit, &self.vals[slot], &mut max_logit, &mut sum_exp);
        }

        // Block KV pairs (still need RMSNorm — not pre-normalized).
        for (k, v) in block_kv {
            let d = D.min(k.len()).min(v.len());
            let mut kt = [0.0f32; D];
            kt[..d].copy_from_slice(&k[..d]);
            rmsnorm_with_gamma(&mut kt[..], &self.gamma[..]);
            let logit = simd_dot_f32(&qt, &kt, D) / sqrt_d;
            streaming_softmax_acc_slice(out, logit, &v[..d], &mut max_logit, &mut sum_exp);
        }

        // Null sink.
        streaming_softmax_acc(out, 0.0, &[0.0f32; D], &mut max_logit, &mut sum_exp);

        if sum_exp > 0.0 {
            let inv = 1.0 / sum_exp;
            simd_scale_inplace(&mut out[..], inv);
        }
    }

    /// Read the cache via **sigmoid-gated** accumulation (AGENTS.md literal
    /// compliance — independent per-slot gates).
    ///
    /// ```text
    /// out = Σ_j sigmoid(q̃ · k̃_j / √d) · v_j
    /// ```
    ///
    /// **Warning:** this read path does NOT normalize. Non-matching slots
    /// contribute `sigmoid(≈0) ≈ 0.5 · v_j` each, so with `W` slots the noise
    /// from irrelevant candidates accumulates. On the G4 retrieval gate this
    /// gives cosine ~0.6 (below the 0.8 bar). Use `read_cache_into` (softmax)
    /// for retrieval; this variant is provided for comparison and for use cases
    /// where sigmoid is genuinely appropriate (e.g., soft gating rather than
    /// competitive retrieval).
    pub fn read_cache_into_sigmoid(
        &self,
        q: &[f32; D],
        gamma: &[f32; D],
        block_kv: &[(&[f32], &[f32])],
        out: &mut [f32; D],
    ) {
        if self.heap_len == 0 && block_kv.is_empty() {
            *out = [0.0; D];
            return;
        }

        let sqrt_d = (D as f32).sqrt();

        let mut qt = [0.0f32; D];
        qt.copy_from_slice(q);
        rmsnorm_with_gamma(&mut qt[..], &gamma[..]);

        out.fill(0.0);

        for i in 0..self.heap_len {
            let (_, slot) = unpack(self.heap[i]);
            let slot = slot as usize;
            let mut kt = [0.0f32; D];
            kt.copy_from_slice(&self.keys[slot]);
            rmsnorm_with_gamma(&mut kt[..], &gamma[..]);
            let logit = simd_dot_f32(&qt, &kt, D) / sqrt_d;
            let gate = sigmoid(logit);
            for d in 0..D {
                out[d] += gate * self.vals[slot][d];
            }
        }

        for (k, v) in block_kv {
            let d = D.min(k.len()).min(v.len());
            let mut kt = [0.0f32; D];
            kt[..d].copy_from_slice(&k[..d]);
            rmsnorm_with_gamma(&mut kt[..], &gamma[..]);
            let logit = simd_dot_f32(&qt, &kt, D) / sqrt_d;
            let gate = sigmoid(logit);
            for j in 0..d {
                out[j] += gate * v[j];
            }
        }
    }

    /// Read with per-key norm rescale γ: `γ_i = √d / max(‖k_i‖, ε)`.
    ///
    /// This is the §3.5 modelless-unblock deterministic correction. Instead of
    /// a fixed γ vector, each key's γ is computed from its own norm, pushing
    /// `‖k̃‖` toward `√d` exactly. The query uses γ=ones (RMSNorm normalizes q
    /// to unit-RMS regardless).
    pub fn read_cache_into_per_key_rescale(
        &self,
        q: &[f32; D],
        block_kv: &[(&[f32], &[f32])],
        eps: f32,
        out: &mut [f32; D],
    ) {
        if self.heap_len == 0 && block_kv.is_empty() {
            *out = [0.0; D];
            return;
        }

        let sqrt_d = (D as f32).sqrt();

        // Query: standard RMSNorm with γ=ones.
        let ones = [1.0f32; D];
        let mut qt = [0.0f32; D];
        qt.copy_from_slice(q);
        rmsnorm_with_gamma(&mut qt[..], &ones[..]);

        let mut max_logit = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        out.fill(0.0);

        for i in 0..self.heap_len {
            let (_, slot) = unpack(self.heap[i]);
            let slot = slot as usize;
            // Per-key rescale: k̃_i = k_i · (√d / max(‖k_i‖, ε)) = k_i · √d / ‖k_i‖
            // (when ‖k_i‖ > ε). This is equivalent to scaling k to norm √d.
            let k = &self.keys[slot];
            let mut kt = [0.0f32; D];
            let norm_sq: f32 = (0..D).map(|d| k[d] * k[d]).sum();
            let norm = norm_sq.sqrt().max(eps);
            let scale = sqrt_d / norm;
            for d in 0..D {
                kt[d] = k[d] * scale;
            }
            let logit = simd_dot_f32(&qt, &kt, D) / sqrt_d;
            streaming_softmax_acc(out, logit, &self.vals[slot], &mut max_logit, &mut sum_exp);
        }

        for (k, v) in block_kv {
            let d = D.min(k.len()).min(v.len());
            let mut kt = [0.0f32; D];
            let norm_sq: f32 = (0..d).map(|j| k[j] * k[j]).sum();
            let norm = norm_sq.sqrt().max(eps);
            let scale = sqrt_d / norm;
            for j in 0..d {
                kt[j] = k[j] * scale;
            }
            let logit = simd_dot_f32(&qt, &kt, D) / sqrt_d;
            streaming_softmax_acc_slice(out, logit, &v[..d], &mut max_logit, &mut sum_exp);
        }

        // Null sink.
        streaming_softmax_acc(out, 0.0, &[0.0f32; D], &mut max_logit, &mut sum_exp);

        if sum_exp > 0.0 {
            let inv = 1.0 / sum_exp;
            simd_scale_inplace(&mut out[..], inv);
        }
    }

    /// Reset the cache to empty. Used at sequence/chunk boundaries.
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

    /// Get the default γ vector.
    #[inline]
    pub fn gamma(&self) -> &[f32; D] {
        &self.gamma
    }

    /// Set the default γ vector. Re-normalizes all stored keys.
    pub fn set_gamma(&mut self, gamma: [f32; D]) {
        self.gamma = gamma;
        for i in 0..self.heap_len {
            let (_, slot) = unpack(self.heap[i]);
            let slot = slot as usize;
            self.keys_norm[slot].copy_from_slice(&self.keys[slot]);
            rmsnorm_with_gamma(&mut self.keys_norm[slot][..], &self.gamma[..]);
        }
    }

    /// Iterate over `(slot, &key, &value, score)` for all occupied slots.
    /// Primarily for testing and diagnostics.
    pub fn slots(&self) -> impl Iterator<Item = (usize, &[f32; D], &[f32; D], f32)> + '_ {
        (0..self.heap_len).map(move |i| {
            let (_, slot) = unpack(self.heap[i]);
            let s = slot as usize;
            (s, &self.keys[s], &self.vals[s], self.scores[s])
        })
    }

    /// Get the minimum score currently in the cache (heap root), or `None` if
    /// empty. Useful for diagnostics and the eviction threshold.
    #[inline]
    pub fn min_score(&self) -> Option<f32> {
        if self.heap_len == 0 {
            return None;
        }
        let (bits, _) = unpack(self.heap[0]);
        Some(f32::from_bits(bits))
    }
}

impl<const D: usize, const W: usize> Default for HippocampalCache<D, W> {
    fn default() -> Self {
        Self::new_with_ones_gamma()
    }
}

impl<const D: usize, const W: usize> Clone for HippocampalCache<D, W> {
    fn clone(&self) -> Self {
        Self {
            heap: self.heap,
            heap_len: self.heap_len,
            keys: self.keys,
            vals: self.vals,
            scores: self.scores,
            keys_norm: self.keys_norm,
            gamma: self.gamma,
        }
    }
}

// ─── Streaming softmax helpers ────────────────────────────────────────────────

/// Streaming softmax accumulation (flash-attention style). Rescales the running
/// output and sum when a new max is encountered. Operates on full `[f32; D]` values.
#[inline]
fn streaming_softmax_acc<const D: usize>(
    out: &mut [f32; D],
    logit: f32,
    val: &[f32; D],
    max_logit: &mut f32,
    sum_exp: &mut f32,
) {
    use crate::simd::fast_exp;
    if logit > *max_logit {
        let rescale = fast_exp(*max_logit - logit);
        *sum_exp = *sum_exp * rescale + 1.0;
        for d in 0..D {
            out[d] = out[d] * rescale + val[d];
        }
        *max_logit = logit;
    } else {
        let weight = fast_exp(logit - *max_logit);
        *sum_exp += weight;
        for d in 0..D {
            out[d] += weight * val[d];
        }
    }
}

/// Streaming softmax accumulation for variable-length value slices (block_kv).
#[inline]
fn streaming_softmax_acc_slice<const D: usize>(
    out: &mut [f32; D],
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
        for j in d..D {
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── T1.V1: top-w retention ───────────────────────────────────────────────

    #[test]
    fn t1v1_top_w_retention() {
        let mut cache: HippocampalCache<8, 4> = HippocampalCache::new_with_ones_gamma();
        let mut rng = fastrand::Rng::with_seed(42);

        // Insert 100 tokens with distinct scores.
        let mut entries: Vec<([f32; 8], [f32; 8], f32)> = Vec::with_capacity(100);
        for i in 0..100 {
            let mut k = [0.0f32; 8];
            let mut v = [0.0f32; 8];
            for d in 0..8 {
                k[d] = rng.f32();
                v[d] = rng.f32();
            }
            // Distinct scores: 0.01 * i guarantees uniqueness.
            let score = 0.01 + 0.01 * i as f32;
            entries.push((k, v, score));
            cache.observe(&k, &v, score, 1.0);
        }

        assert_eq!(cache.len(), 4);

        // The top-4 by score are entries 99, 98, 97, 96.
        let top4_scores: Vec<f32> = entries.iter().map(|(_, _, s)| *s).collect();
        let mut expected = top4_scores.clone();
        expected.sort_by(|a, b| b.total_cmp(a));
        let expected_top4: Vec<f32> = expected[..4].to_vec();

        let mut actual_scores: Vec<f32> = cache.slots().map(|(_, _, _, s)| s).collect();
        actual_scores.sort_by(|a, b| b.total_cmp(a));

        for (i, (a, e)) in actual_scores.iter().zip(expected_top4.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "slot {i}: actual score {a} != expected {e}"
            );
        }
    }

    // ── T1.V2: order-independence ─────────────────────────────────────────────

    #[test]
    fn t1v2_order_independence() {
        let mut rng = fastrand::Rng::with_seed(99);
        let entries: Vec<([f32; 8], [f32; 8], f32)> = (0..100)
            .map(|i| {
                let mut k = [0.0f32; 8];
                let mut v = [0.0f32; 8];
                for d in 0..8 {
                    k[d] = rng.f32();
                    v[d] = rng.f32();
                }
                // Distinct scores.
                let score = 0.01 + 0.01 * i as f32;
                (k, v, score)
            })
            .collect();

        // Insert in 5 different random orders. The final cache SET must be identical.
        let mut reference_keys: Vec<[f32; 8]> = Vec::new();

        for trial in 0..5 {
            let mut order: Vec<usize> = (0..100).collect();
            let mut trial_rng = fastrand::Rng::with_seed(trial * 1000);
            // Fisher-Yates shuffle for a random insertion order.
            for j in (1..order.len()).rev() {
                let k = trial_rng.usize(0..=j);
                order.swap(j, k);
            }

            let mut cache: HippocampalCache<8, 4> = HippocampalCache::new_with_ones_gamma();
            for &idx in &order {
                let (k, v, s) = &entries[idx];
                cache.observe(k, v, *s, 1.0);
            }

            // Collect the surviving keys (the cache set identity).
            let mut keys: Vec<[f32; 8]> = cache.slots().map(|(_, k, _, _)| *k).collect();
            keys.sort_by(|a, b| {
                for d in 0..8 {
                    match a[d].partial_cmp(&b[d]) {
                        Some(std::cmp::Ordering::Equal) => continue,
                        Some(other) => return other,
                        None => continue,
                    }
                }
                std::cmp::Ordering::Equal
            });

            if trial == 0 {
                reference_keys = keys;
            } else {
                assert_eq!(
                    keys.len(),
                    reference_keys.len(),
                    "trial {trial}: cache size differs"
                );
                for (i, (a, e)) in keys.iter().zip(reference_keys.iter()).enumerate() {
                    for d in 0..8 {
                        assert!(
                            (a[d] - e[d]).abs() < 1e-6,
                            "trial {trial} slot {i} dim {d}: {a:?} != {e:?}"
                        );
                    }
                }
            }
        }
    }

    // ── T1.V3: read determinism ───────────────────────────────────────────────

    #[test]
    fn t1v3_read_determinism() {
        let mut cache: HippocampalCache<16, 8> = HippocampalCache::new_with_ones_gamma();
        let mut rng = fastrand::Rng::with_seed(7);
        for i in 0..20 {
            let mut k = [0.0f32; 16];
            let mut v = [0.0f32; 16];
            for d in 0..16 {
                k[d] = rng.f32() * 2.0 - 1.0;
                v[d] = rng.f32() * 2.0 - 1.0;
            }
            cache.observe(&k, &v, 0.5 + i as f32 * 0.01, 1.0);
        }

        let q = [0.5f32; 16];
        let gamma = [1.0f32; 16];
        let block: &[(&[f32], &[f32])] = &[];
        let mut out1 = [0.0f32; 16];
        let mut out2 = [0.0f32; 16];

        cache.read_cache_into(&q, &gamma, block, &mut out1);
        cache.read_cache_into(&q, &gamma, block, &mut out2);

        for d in 0..16 {
            assert!(
                out1[d].to_bits() == out2[d].to_bits(),
                "dim {d}: not byte-identical"
            );
        }
    }

    // ── Phase 2 G1: multi-needle retention (4k tokens, 8 needles) ─────────────

    #[test]
    fn g1_multi_needle_retention_4k() {
        let mut cache: HippocampalCache<64, 8> = HippocampalCache::new_with_ones_gamma();
        let mut rng = fastrand::Rng::with_seed(12345);

        // 8 needles with top-8 scores.
        let mut needles: Vec<([f32; 64], f32)> = Vec::with_capacity(8);
        for _ in 0..8 {
            let mut k = [0.0f32; 64];
            for d in 0..64 {
                k[d] = rng.f32();
            }
            let v = [0.0f32; 64]; // value doesn't matter for retention test
            let beta = 0.9f32;
            let residual = 0.8 + rng.f32() * 0.2; // [0.8, 1.0]
            let score = beta * residual; // [0.72, 0.9]
            needles.push((k, score));
            cache.observe(&k, &v, beta, residual);
        }

        // 3992 distractors with low scores.
        for _ in 0..3992 {
            let mut k = [0.0f32; 64];
            let mut v = [0.0f32; 64];
            for d in 0..64 {
                k[d] = rng.f32();
                v[d] = rng.f32();
            }
            let beta = 0.3f32;
            let residual = 0.05 + rng.f32() * 0.15; // [0.05, 0.2]
            // score = [0.015, 0.06] << needle scores [0.72, 0.9]
            cache.observe(&k, &v, beta, residual);
        }

        // All 8 needles must be retained.
        assert_eq!(cache.len(), 8, "cache should be full");

        // Check that every needle key is in the cache.
        let cache_keys: Vec<[f32; 64]> = cache.slots().map(|(_, k, _, _)| *k).collect();
        for (needle_key, needle_score) in &needles {
            let found = cache_keys.iter().any(|ck| {
                ck.iter()
                    .zip(needle_key.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-6)
            });
            assert!(found, "needle with score {needle_score} was evicted!");
        }

        // Check the min score in cache is a needle score (≥ 0.72).
        let min = cache.min_score().unwrap();
        assert!(
            min >= 0.7,
            "min cache score {min} is too low — a distractor leaked in"
        );
    }

    #[test]
    fn g1_distractors_evicted() {
        let mut cache: HippocampalCache<32, 4> = HippocampalCache::new_with_ones_gamma();
        let mut rng = fastrand::Rng::with_seed(77);

        // 4 high-score tokens.
        for i in 0..4 {
            let k = [i as f32; 32];
            let v = [i as f32; 32];
            cache.observe(&k, &v, 0.9, 0.9);
        }
        // 100 distractors.
        for _ in 0..100 {
            let k = [rng.f32(); 32];
            let v = [rng.f32(); 32];
            cache.observe(&k, &v, 0.1, 0.1);
        }

        // All 4 slots must have high scores.
        for (_, _, _, score) in cache.slots() {
            assert!(score > 0.5, "distractor leaked in with score {score}");
        }
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn empty_cache_read_is_zero() {
        let cache: HippocampalCache<8, 4> = HippocampalCache::new_with_ones_gamma();
        let q = [1.0f32; 8];
        let gamma = [1.0f32; 8];
        let mut out = [99.0f32; 8];
        cache.read_cache_into(&q, &gamma, &[], &mut out);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn reset_clears_cache() {
        let mut cache: HippocampalCache<8, 4> = HippocampalCache::new_with_ones_gamma();
        cache.observe(&[1.0; 8], &[2.0; 8], 0.5, 0.5);
        assert_eq!(cache.len(), 1);
        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn nan_score_rejected() {
        let mut cache: HippocampalCache<4, 4> = HippocampalCache::new_with_ones_gamma();
        cache.observe(&[1.0; 4], &[1.0; 4], f32::NAN, 1.0);
        assert_eq!(cache.len(), 0, "NaN score must be rejected");
    }

    #[test]
    fn negative_score_rejected() {
        let mut cache: HippocampalCache<4, 4> = HippocampalCache::new_with_ones_gamma();
        cache.observe(&[1.0; 4], &[1.0; 4], -1.0, 1.0);
        assert_eq!(cache.len(), 0, "negative score must be rejected");
    }

    #[test]
    fn zero_score_accepted() {
        let mut cache: HippocampalCache<4, 4> = HippocampalCache::new_with_ones_gamma();
        cache.observe(&[1.0; 4], &[1.0; 4], 0.0, 0.0);
        assert_eq!(cache.len(), 1, "zero score should be accepted");
    }

    #[test]
    fn w_zero_is_noop() {
        let mut cache: HippocampalCache<4, 0> = HippocampalCache::new([1.0; 4]);
        cache.observe(&[1.0; 4], &[1.0; 4], 1.0, 1.0);
        assert_eq!(cache.len(), 0);
        let mut out = [99.0f32; 4];
        cache.read_cache_into(&[1.0; 4], &[1.0; 4], &[], &mut out);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn eviction_replaces_lowest() {
        let mut cache: HippocampalCache<4, 2> = HippocampalCache::new_with_ones_gamma();
        // Insert scores 0.1, 0.2 (both kept).
        cache.observe(&[1.0; 4], &[10.0; 4], 0.1, 1.0);
        cache.observe(&[2.0; 4], &[20.0; 4], 0.2, 1.0);
        assert_eq!(cache.len(), 2);
        assert!((cache.min_score().unwrap() - 0.1).abs() < 1e-6);

        // Insert score 0.15: evicts the 0.1 entry.
        cache.observe(&[3.0; 4], &[30.0; 4], 0.15, 1.0);
        assert_eq!(cache.len(), 2);
        // Min should now be 0.15.
        assert!(
            (cache.min_score().unwrap() - 0.15).abs() < 1e-6,
            "min score should be 0.15 after evicting 0.1"
        );

        // Insert score 0.05: rejected (below min).
        cache.observe(&[4.0; 4], &[40.0; 4], 0.05, 1.0);
        assert_eq!(cache.len(), 2);
        assert!((cache.min_score().unwrap() - 0.15).abs() < 1e-6);
    }

    #[test]
    fn softmax_read_concentrates_on_matching_key() {
        // 4 slots: one matching key, three random distractors.
        let mut cache: HippocampalCache<32, 4> = HippocampalCache::new_with_ones_gamma();
        let mut rng = fastrand::Rng::with_seed(555);

        // Matching key/value: unit-norm.
        let mut target_k = [0.0f32; 32];
        let mut target_v = [0.0f32; 32];
        for d in 0..32 {
            target_k[d] = rng.f32() * 2.0 - 1.0;
            target_v[d] = rng.f32() * 2.0 - 1.0;
        }
        let kn = (target_k.iter().map(|x| x * x).sum::<f32>()).sqrt();
        let vn = (target_v.iter().sum::<f32>()).sqrt();
        for d in 0..32 {
            target_k[d] /= kn;
            target_v[d] /= vn;
        }
        cache.observe(&target_k, &target_v, 0.9, 0.9);

        // 3 distractor slots.
        for _ in 0..3 {
            let mut k = [0.0f32; 32];
            let mut v = [0.0f32; 32];
            for d in 0..32 {
                k[d] = rng.f32() * 2.0 - 1.0;
                v[d] = rng.f32() * 2.0 - 1.0;
            }
            cache.observe(&k, &v, 0.3, 0.3);
        }

        // Query = target key.
        let gamma = [1.0f32; 32];
        let mut out = [0.0f32; 32];
        cache.read_cache_into(&target_k, &gamma, &[], &mut out);

        // Cosine(out, target_v) should be high.
        let dot: f32 = out.iter().zip(target_v.iter()).map(|(a, b)| a * b).sum();
        let out_norm = (out.iter().map(|x| x * x).sum::<f32>()).sqrt();
        let cos = dot / (out_norm * vn.max(1e-8));
        assert!(
            cos > 0.8,
            "softmax read cosine {cos} should be > 0.8 for matching key"
        );
    }
}

// ─── Sorted-vec variant for T2.3 comparison ──────────────────────────────────

/// Linear-scan sorted-vec cache variant (T2.3 benchmark competitor).
///
/// Maintains the same top-`w` semantics but uses a sorted `scores` array with
/// linear-scan eviction instead of a binary heap. At small `w` the linear scan
/// + cache locality may beat the heap's pointer-chasing.
///
/// This is a benchmark-only competitor — `HippocampalCache` (heap-backed) is the
/// production type unless the T2.3 micro-bench proves otherwise.
pub struct SortedSlotCache<const D: usize, const W: usize> {
    keys: [[f32; D]; W],
    vals: [[f32; D]; W],
    scores: [f32; W],
    len: usize,
}

impl<const D: usize, const W: usize> SortedSlotCache<D, W> {
    pub fn new() -> Self {
        Self {
            keys: [[0.0f32; D]; W],
            vals: [[0.0f32; D]; W],
            scores: [0.0f32; W],
            len: 0,
        }
    }

    /// Observe with the same semantics as `HippocampalCache::observe`.
    /// Uses linear scan: O(W) to find the min, vs O(log W) for the heap.
    pub fn observe(&mut self, k: &[f32; D], v: &[f32; D], beta: f32, residual_norm: f32) {
        let score = beta * residual_norm;
        if score.is_nan() || score < 0.0 {
            return;
        }

        if self.len < W {
            let slot = self.len;
            self.keys[slot] = *k;
            self.vals[slot] = *v;
            self.scores[slot] = score;
            self.len += 1;
        } else if W > 0 {
            // Linear scan for the minimum score.
            let mut min_idx = 0;
            for i in 1..W {
                if self.scores[i] < self.scores[min_idx] {
                    min_idx = i;
                }
            }
            if score > self.scores[min_idx] {
                self.keys[min_idx] = *k;
                self.vals[min_idx] = *v;
                self.scores[min_idx] = score;
            }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

impl<const D: usize, const W: usize> Default for SortedSlotCache<D, W> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod sorted_vec_tests {
    use super::*;

    #[test]
    fn sorted_vec_matches_heap_top_w() {
        let mut rng = fastrand::Rng::with_seed(31);
        let mut heap_cache: HippocampalCache<8, 4> = HippocampalCache::new_with_ones_gamma();
        let mut sorted_cache: SortedSlotCache<8, 4> = SortedSlotCache::new();

        for i in 0..100 {
            let mut k = [0.0f32; 8];
            let mut v = [0.0f32; 8];
            for d in 0..8 {
                k[d] = rng.f32();
                v[d] = rng.f32();
            }
            let score = 0.01 + 0.01 * i as f32;
            heap_cache.observe(&k, &v, score, 1.0);
            sorted_cache.observe(&k, &v, score, 1.0);
        }

        assert_eq!(heap_cache.len(), sorted_cache.len());

        let mut heap_scores: Vec<f32> = heap_cache.slots().map(|(_, _, _, s)| s).collect();
        let mut sorted_scores: Vec<f32> = sorted_cache.scores[..sorted_cache.len()].to_vec();
        heap_scores.sort_by(|a, b| a.total_cmp(b));
        sorted_scores.sort_by(|a, b| a.total_cmp(b));

        for (h, s) in heap_scores.iter().zip(sorted_scores.iter()) {
            assert!((h - s).abs() < 1e-6, "heap {h} != sorted {s}");
        }
    }
}
