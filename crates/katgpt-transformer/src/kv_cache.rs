//! KV cache types for autoregressive generation, paged branching, and Raven routing.
//!
//! All types here are pure data + allocator helpers — no forward logic.
//!
//! # Sliding-window ring caches (aliases: `RingKvCache`, `SlidingWindowCache`,
//! `WindowedKvCache`, `kv_ring`, circular cache, windowed KV cache)
//!
//! [`MultiLayerKVCache`] supports per-layer sliding-window bounding: a layer
//! with `sliding_capacity(l) = W` allocates exactly `W × kv_dim` floats and holds
//! the most recent `W` positions as a **plain-modulo ring buffer**. The aliases
//! above exist so a consumer grepping English names finds this substrate instead
//! of re-implementing a ring (Issue 683 T4).
//!
//! **House convention (Issue 683, adjudicated 2026-08-24): plain modulo.** The
//! consumer's forward code writes K/V at `pos % W` (one slot, no mirror copy) and
//! reads logical position `t` at `t % W`. An earlier design allocated `2·W·kvd`
//! per sliding layer and mirrored every write so any window was always
//! contiguous in the buffer; that mirrored write was never exercised by any
//! live construction path, and the contiguous-read saving was never measured, so
//! the 2× memory bought nothing and the mirroring was removed. Contiguity across
//! a ring wrap is **not** provided — a window straddling the ring boundary must
//! be gathered as two slices (exact contract on the `sliding_capacity` field).

use katgpt_core::types::{self, Config};

/// KV cache for a single layer (autoregressive generation).
pub struct KVCache {
    pub key: Vec<f32>,   // [block_size, kv_dim] where kv_dim = n_kv_head * head_dim
    pub value: Vec<f32>, // [block_size, kv_dim]
}

impl KVCache {
    pub fn new(config: &Config) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            key: vec![0.0; config.block_size * kvd],
            value: vec![0.0; config.block_size * kvd],
        }
    }

    pub fn reset(&mut self) {
        // Eager zeroing — safe default for a shared substrate crate. The no-op
        // optimization (relying on write-before-read invariant) is a consumer-
        // specific perf decision; consumers that provably maintain the invariant
        // can override locally. The conservative behavior avoids stale-KV leaks
        // for consumers that reset between sequences without re-writing every
        // position (e.g. dflash speculative rollback paths).
        self.key.fill(0.0);
        self.value.fill(0.0);
    }

    /// Invalidate only a single position in the KV cache — O(kv_dim) instead of
    /// O(block_size × kv_dim). Used by dflash when only one position is dirty per
    /// step (Issue 053). Also used by consumers that need to clear a rejected
    /// speculative token's KV before the next draft iteration.
    #[inline]
    pub fn invalidate_position(&mut self, pos: usize, kv_dim: usize) {
        let off = pos * kv_dim;
        if off + kv_dim <= self.key.len() {
            self.key[off..off + kv_dim].fill(0.0);
            self.value[off..off + kv_dim].fill(0.0);
        }
    }
}

/// Multi-layer KV cache: one KVCache per transformer layer.
pub struct MultiLayerKVCache {
    pub layers: Vec<KVCache>,
    /// Highest position written + 1 across all layers, for efficient snapshot.
    fill_pos: usize,
    /// Per-layer sliding-window capacity. `0` = unbounded (uses full
    /// `block_size` allocation, the original behavior for all non-Gemma-4 models).
    /// `> 0` = sliding-bounded plain-modulo ring buffer: the layer's physical
    /// buffer is exactly `sliding_capacity × kv_dim` floats and holds the most
    /// recent `sliding_capacity` positions (Issue 683).
    ///
    /// This cache is data + capacity bookkeeping only — it performs no writes
    /// and no reads of its own. The ring convention is the **consumer's**
    /// responsibility:
    ///
    /// * **Write** (consumer forward code): store K/V for logical position
    ///   `pos` at `layers[l].key[(pos % W) * kv_dim ..][..kv_dim]` and the same
    ///   offset in `value` — plain modulo, a single slot.
    /// * **Read** (consumer attention): logical position `t` lives at `t % W`.
    ///   A window `[t_start, pos]` is contiguous in the buffer only while it
    ///   does not straddle the ring boundary (`t_start % W <= pos % W`); a
    ///   straddling window must be gathered as two slices, `[t_start % W .. W)`
    ///   and `[0 .. pos % W + 1)`. Contiguity across a wrap is NOT provided —
    ///   the former 2× mirrored layout that guaranteed it was removed in
    ///   Issue 683 (never exercised, saving never measured).
    ///
    /// ⚠ Downstream note (updated 2026-08-27): riir-ai's `forward_gemma4_impl`
    /// migrated to the plain-modulo convention (riir-ai Issue 752, commit
    /// `fbc598ab9`) — single write at `pos % W` + a two-slice straddling
    /// gather — so caches built by the sliding-bounded constructors are now
    /// SAFE to pass into that forward (pinned bit-identical vs unbounded by
    /// `crates/riir-engine/tests/gemma4_sliding_ring_wrap.rs`). The original
    /// 2026-08-24 warning (mirrored-write branch assuming `2·W·kvd` buffers)
    /// is obsolete.
    pub sliding_capacity: Vec<usize>,
}

impl MultiLayerKVCache {
    pub fn new(config: &Config) -> Self {
        let mut layers = Vec::with_capacity(config.n_layer);
        layers.extend((0..config.n_layer).map(|_| KVCache::new(config)));
        Self {
            layers,
            fill_pos: 0,
            sliding_capacity: vec![0; config.n_layer],
        }
    }

    /// Construct with per-layer KV dimensions. Used by models where each layer
    /// has a different kv_dim (e.g. Gemma 4's alternating sliding/full layers —
    // Issue 577). The vec length MUST equal `config.n_layer`.
    ///
    /// Each cache layer is allocated as `[block_size * per_layer_kv_dim[i]]`.
    pub fn new_with_per_layer_kv_dim(config: &Config, per_layer_kv_dim: &[usize]) -> Self {
        debug_assert_eq!(
            per_layer_kv_dim.len(),
            config.n_layer,
            "per_layer_kv_dim length must equal n_layer"
        );
        let layers = per_layer_kv_dim
            .iter()
            .map(|&kvd| KVCache {
                key: vec![0.0; config.block_size * kvd],
                value: vec![0.0; config.block_size * kvd],
            })
            .collect();
        Self {
            layers,
            fill_pos: 0,
            sliding_capacity: vec![0; config.n_layer],
        }
    }

    /// Construct with per-layer KV dimensions and an explicit position cap.
    ///
    /// Identical to [`new_with_per_layer_kv_dim`] except each layer is sized
    /// `[max_positions * per_layer_kv_dim[i]]` instead of
    /// `[block_size * per_layer_kv_dim[i]]`. Use this for **training** where
    /// the sequence length is fixed and known upfront — avoids reserving the
    /// model's full context window (e.g. Gemma-4-12B `block_size = 262_144` →
    /// ~168 GiB of demand-paged virtual address space that can exceed the
    /// system commit limit when the GPU runtime also holds significant host
    /// memory). `max_positions` should be the training `seq_len`.
    ///
    /// `max_positions` is clamped to `>= 1` to avoid zero-sized allocations.
    pub fn new_with_per_layer_kv_dim_bounded(
        config: &Config,
        per_layer_kv_dim: &[usize],
        max_positions: usize,
    ) -> Self {
        debug_assert_eq!(
            per_layer_kv_dim.len(),
            config.n_layer,
            "per_layer_kv_dim length must equal n_layer"
        );
        let cap = max_positions.max(1);
        let layers = per_layer_kv_dim
            .iter()
            .map(|&kvd| KVCache {
                key: vec![0.0; cap * kvd],
                value: vec![0.0; cap * kvd],
            })
            .collect();
        Self {
            layers,
            fill_pos: 0,
            sliding_capacity: vec![0; config.n_layer],
        }
    }

    /// Construct a sliding-bounded cache for Gemma-4's alternating
    /// Sliding/Full layer pattern (Plan 320 Phase D3 production fix; re-based
    /// onto the plain-modulo 1× ring convention by Issue 683).
    ///
    /// Sliding layers get a **plain-modulo ring buffer** of exactly
    /// `sliding_window * kvd` floats (logical capacity `sliding_window` — one
    /// slot per window position, no mirror region). Full layers get
    /// `block_size * kvd` (unbounded, same as [`new_with_per_layer_kv_dim`]).
    /// The consumer's forward code owns the ring semantics: write at
    /// `pos % sliding_window`, read `t` at `t % sliding_window`, gathering a
    /// wrap-straddling window as two slices (see the `sliding_capacity` field
    /// doc for the full contract).
    ///
    /// Versus the naive all-`block_size` allocation this reduces the 256K-context
    /// KV cache from ~168 GiB to the single-digit GiB range, and the 1×
    /// plain-modulo convention halves the sliding layers' share again versus the
    /// former mirrored layout. See the D3 test (`plan320_d3_kv_cache_256k_budget.rs`)
    /// for the original budget analysis.
    ///
    /// # Arguments
    ///
    /// * `config` — for `n_layer` + `block_size`.
    /// * `per_layer_kv_dim` — per-layer KV dimension (same as
    ///   `new_with_per_layer_kv_dim`).
    /// * `sliding_layers` — boolean mask: `true` = this layer is sliding-window
    ///   (bounded allocation), `false` = full attention (unbounded). The caller
    ///   determines this from the model's layer-type pattern (e.g.
    ///   `config.gemma4_layer_types` in katgpt-core, which isn't visible at this
    ///   substrate layer).
    /// * `sliding_window` — the sliding-window size (Gemma-4 = 1024). Must be > 0
    ///   for sliding layers to be bounded; 0 falls back to unbounded for all layers.
    pub fn new_gemma4_sliding_bounded(
        config: &Config,
        per_layer_kv_dim: &[usize],
        sliding_layers: &[bool],
        sliding_window: usize,
    ) -> Self {
        debug_assert_eq!(
            per_layer_kv_dim.len(),
            config.n_layer,
            "per_layer_kv_dim length must equal n_layer"
        );
        debug_assert_eq!(
            sliding_layers.len(),
            config.n_layer,
            "sliding_layers length must equal n_layer"
        );
        let sw = sliding_window;
        let layers = per_layer_kv_dim
            .iter()
            .enumerate()
            .map(|(i, &kvd)| {
                let is_sliding = sliding_layers.get(i).copied().unwrap_or(false);
                let capacity = if is_sliding && sw > 0 {
                    // Plain-modulo ring: window × kvd, 1× (Issue 683).
                    sw * kvd
                } else {
                    config.block_size * kvd
                };
                KVCache {
                    key: vec![0.0; capacity],
                    value: vec![0.0; capacity],
                }
            })
            .collect();
        // Record per-layer sliding capacity (0 = unbounded, sw = sliding-bounded).
        let sliding_capacity: Vec<usize> = (0..config.n_layer)
            .map(|i| {
                let is_sliding = sliding_layers.get(i).copied().unwrap_or(false);
                if is_sliding && sw > 0 { sw } else { 0 }
            })
            .collect();
        Self {
            layers,
            fill_pos: 0,
            sliding_capacity,
        }
    }

    /// Construct a cache where **every** layer is sliding-bounded at `window`
    /// (Issue 683 T1) — the uniform all-sliding pattern (SSSS-style drafters,
    /// windowed-everywhere architectures; see the module doc's ring-cache
    /// vocabulary).
    ///
    /// Model-agnostic counterpart to [`new_gemma4_sliding_bounded`]: no
    /// layer-type pattern, one window for every layer, per-layer KV width
    /// `types::kv_dim(config)`. Allocation is exactly `window × kv_dim` floats
    /// per layer (plain-modulo ring, 1×) — same convention and the same
    /// allocation path, so "capacity implies allocation" holds by construction.
    /// This is a constructor rather than a post-hoc setter for exactly that
    /// reason: a setter that does not resize the buffers would silently produce
    /// out-of-range reads.
    ///
    /// # Panics
    ///
    /// Panics if `window == 0` (use [`new`](Self::new) for an unbounded cache).
    pub fn new_all_sliding_bounded(config: &Config, window: usize) -> Self {
        assert!(window > 0, "new_all_sliding_bounded: window must be > 0");
        let kvd = types::kv_dim(config);
        let per_layer_kv_dim = vec![kvd; config.n_layer];
        let all_sliding = vec![true; config.n_layer];
        // Reuse the existing allocation path — no duplicated ring allocation.
        Self::new_gemma4_sliding_bounded(config, &per_layer_kv_dim, &all_sliding, window)
    }

    /// Get the sliding-window capacity for a layer (0 = unbounded/full block_size).
    #[inline]
    pub fn sliding_capacity(&self, layer_idx: usize) -> usize {
        self.sliding_capacity.get(layer_idx).copied().unwrap_or(0)
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.fill_pos = 0;
    }

    /// Invalidate only a single position across all layers — O(n_layer × kv_dim).
    /// Much cheaper than full reset O(n_layer × block_size × kv_dim) when only 1
    /// position is dirty. Used by dflash speculative decoding (Issue 053).
    #[inline]
    pub fn invalidate_position(&mut self, pos: usize, kv_dim: usize) {
        for layer in &mut self.layers {
            layer.invalidate_position(pos, kv_dim);
        }
    }

    /// Update fill_pos tracker. Call after writing to the cache at a position.
    pub fn advance_pos(&mut self, pos: usize) {
        self.fill_pos = self.fill_pos.max(pos + 1);
    }

    /// Get the tracked fill position (highest position written + 1).
    #[inline]
    pub fn fill_pos(&self) -> usize {
        self.fill_pos
    }

    /// Set the fill_pos tracker directly, WITHOUT touching the K/V buffers.
    ///
    /// Use this when a caller has already reshaped the buffer contents in-place
    /// (e.g. sliding-window eviction's `copy_within` shift) and only needs to
    /// advance/shrink the logical fill marker. Distinct from `reset()`, which
    /// also zeroes the K/V data — calling `reset()` after an in-place shift
    /// would wipe the just-copied entries (Issue: sleep eviction
    /// sliding_window_retains_recent failure, 2026-06-29).
    #[inline]
    pub fn set_fill_pos(&mut self, pos: usize) {
        self.fill_pos = pos;
    }

    /// Snapshot KV cache state up to position `pos`.
    /// Copies only filled slots [0..pos) per layer — cheap at our model scale.
    ///
    /// **Allocating:** makes `1 + 2*n_layer` heap allocations per call. For the
    /// per-speculation-step hot path, prefer [`snapshot_into`](Self::snapshot_into)
    /// with a hoisted scratch buffer (zero alloc in steady state). This variant
    /// remains for cold paths and convenience wrappers that don't have a
    /// reusable buffer.
    pub fn snapshot(&self, pos: usize, config: &Config) -> KVSnapshot {
        let kd = types::kv_dim(config);
        // Pre-allocate outer Vec to avoid collect() reallocation jitter.
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            // Clamp to the physical buffer: a sliding-bounded layer holds a
            // `window × kv_dim` ring, so only the physical contents can be
            // captured — logical positions beyond the window are not
            // reconstructible from a ring (Issue 683).
            let end = (pos * kd).min(layer.key.len());
            layers.push(KVLayerSnapshot {
                key: layer.key[..end].to_vec(),
                value: layer.value[..end].to_vec(),
            });
        }
        KVSnapshot { pos, layers }
    }

    /// Zero-alloc variant of [`snapshot`](Self::snapshot) that refills a reusable
    /// [`KVSnapshot`] in place. The snapshot's per-layer `key`/`value` buffers
    /// are `resize`d to the new length (reusing their existing allocation when
    /// possible) and overwritten — no new `Vec` is allocated in steady state.
    ///
    /// # Allocation
    ///
    /// On the first call (or when `out` was previously shorter), the inner
    /// Vecs grow. On every subsequent call with the same or smaller `pos`,
    /// the existing allocations are reused — zero new heap allocations. This
    /// is the variant to use on the per-speculation-step hot path.
    ///
    /// # Layer-count changes
    ///
    /// If `out.layers.len() != self.layers.len()`, the outer Vec is resized.
    /// In steady state (same model), this branch is never taken.
    pub fn snapshot_into(&self, pos: usize, config: &Config, out: &mut KVSnapshot) {
        let kd = types::kv_dim(config);
        out.pos = pos;
        if out.layers.len() != self.layers.len() {
            out.layers
                .resize_with(self.layers.len(), || KVLayerSnapshot {
                    key: Vec::new(),
                    value: Vec::new(),
                });
        }
        for (src, dst) in self.layers.iter().zip(out.layers.iter_mut()) {
            // Clamp to the physical buffer (sliding rings hold only the last
            // `window` positions — Issue 683).
            let end = (pos * kd).min(src.key.len());
            dst.key.resize(end, 0.0);
            dst.value.resize(end, 0.0);
            dst.key[..end].copy_from_slice(&src.key[..end]);
            dst.value[..end].copy_from_slice(&src.value[..end]);
        }
    }

    /// Restore KV cache from a snapshot.
    /// Writes snapshot data back and zeros out positions [snapshot.pos..block_size)
    /// to prevent stale data leaking into the next sequence. The tail zeroing is
    /// the conservative default for a shared substrate crate.
    ///
    /// For sliding-bounded layers the restore is a **physical** ring restore:
    /// `end` is clamped to the layer's buffer (and to the captured snapshot
    /// length), so a full-buffer snapshot round-trips the ring contents exactly
    /// and the tail zeroing covers whatever the snapshot did not capture
    /// (Issue 683).
    pub fn restore(&mut self, snapshot: &KVSnapshot, config: &Config) {
        let kd = types::kv_dim(config);
        for (layer, snap_layer) in self.layers.iter_mut().zip(snapshot.layers.iter()) {
            // Clamp to the physical buffer and to what the snapshot actually
            // captured (sliding rings are window-sized — Issue 683).
            let end = (snapshot.pos * kd)
                .min(layer.key.len())
                .min(snap_layer.key.len());
            layer.key[..end].copy_from_slice(&snap_layer.key[..end]);
            layer.value[..end].copy_from_slice(&snap_layer.value[..end]);
            // Zero out the uncaptured tail to prevent stale data from a
            // previous sequence leaking into the restored state.
            layer.key[end..].fill(0.0);
            layer.value[end..].fill(0.0);
        }
    }
}

/// Cheap snapshot of KV cache state up to position `pos`.
/// Only copies filled slots [0..pos) per layer, not the entire block_size buffer.
///
/// `Default` is derived so callers can construct an empty snapshot for use as
/// a reusable scratch buffer with [`MultiLayerKVCache::snapshot_into`] — the
/// zero-alloc variant on the per-speculation-step hot path.
#[derive(Default)]
pub struct KVSnapshot {
    pub pos: usize,
    pub layers: Vec<KVLayerSnapshot>,
}

/// Per-layer snapshot of KV cache data.
#[derive(Default)]
pub struct KVLayerSnapshot {
    pub key: Vec<f32>,   // [pos * kv_dim]
    pub value: Vec<f32>, // [pos * kv_dim]
}

/// Preload drafter's KV cache with target's pre-computed key/value pairs.
///
/// Copies target's KV for positions [0..pos) into drafter's cache.
/// This enables cross-attention: the drafter attends to the target's past KV
/// instead of computing its own from scratch.
///
/// Only active when `target_kv_dim == draft_kv_dim` (dimensions must match).
/// When dimensions don't match, silently returns (drafter computes its own KV).
///
/// Hybrid behavior after preload:
/// - Past positions [0..pos): read from preloaded target KV
/// - New positions [pos..]: computed by drafter during forward pass
pub fn preload_kv_cache(
    draft_cache: &mut MultiLayerKVCache,
    target_cache: &MultiLayerKVCache,
    pos: usize,
    target_config: &Config,
    draft_config: &Config,
) {
    let target_kv_dim = types::kv_dim(target_config);
    let draft_kv_dim = types::kv_dim(draft_config);

    // Dimension guard: can only share when kv_dim matches
    if target_kv_dim != draft_kv_dim {
        return;
    }

    // Layer guard: can only share layers that exist in both caches
    let min_layers = draft_cache.layers.len().min(target_cache.layers.len());

    // Copy KV for positions [0..pos) for each shared layer. Sliding-bounded
    // layers hold `window × kv_dim` rings, so the copy is clamped to both
    // physical buffers — a physical ring transfer, not a logical [0..pos)
    // transfer (Issue 683).
    if pos > 0 {
        for layer_idx in 0..min_layers {
            let draft_layer = &mut draft_cache.layers[layer_idx];
            let target_layer = &target_cache.layers[layer_idx];
            let copy_len = (pos * target_kv_dim)
                .min(draft_layer.key.len())
                .min(target_layer.key.len());
            if copy_len > 0 {
                draft_layer.key[..copy_len].copy_from_slice(&target_layer.key[..copy_len]);
                draft_layer.value[..copy_len].copy_from_slice(&target_layer.value[..copy_len]);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Paged KV cache — DDTree branch exploration (copy-on-write fork)
// ──────────────────────────────────────────────────────────────────────────

/// Page size in tokens (tuneable, must be power of 2).
pub const PAGE_SIZE: usize = 16;

/// Paged KV cache for DDTree branch exploration.
/// Allocates memory in fixed-size pages with copy-on-write fork.
///
/// Page layout per page: `[K_data | V_data]` where each segment is `PAGE_SIZE * kv_dim` floats.
/// This enables sharing prefix pages between branches without cloning data.
///
/// Fields are `pub` because katgpt-rs root tests inspect them directly
/// (`layer_page_tables`, `free_pages`). Consumers should prefer the method API.
/// Field order groups heap pointers first, then `usize` scalars, to minimize padding.
pub struct PagedKVCache {
    /// Pool of pages. Each page: `[PAGE_SIZE * kv_dim * 2]` floats (K then V).
    pub pages: Vec<Vec<f32>>,
    /// Per-layer page tables. `layer_page_tables[layer][seq_idx]` = vec of page indices.
    pub layer_page_tables: Vec<Vec<Vec<usize>>>,
    /// Free list of page indices for reuse.
    pub free_pages: Vec<usize>,
    /// Reference count per page index. Page is free when ref_count == 0.
    /// Maintained on fork (increment shared pages) and rollback (decrement removed pages).
    /// Enables O(1) exclusive-page detection instead of O(N×P×L) HashSet scan (Issue 053).
    pub page_ref_counts: Vec<usize>,
    /// Dimension of each KV entry (`n_kv_head * head_dim`).
    pub kv_dim: usize,
    /// Cached `PAGE_SIZE * kv_dim` — avoids recomputing on every write/read.
    pub kv_page_size: usize,
    /// Total pages ever allocated (monotonically increasing).
    pub total_pages: usize,
}

impl PagedKVCache {
    /// Create a new paged KV cache.
    /// `max_sequences`: initial number of sequence slots (can grow via fork).
    ///
    /// All initial pages start in the free list (ref_count == 0) so they can be
    /// reused immediately by `alloc_page` without growing the pool. This is the
    /// memory-efficient initialization ported from riir-engine.
    pub fn new(config: &Config, max_sequences: usize) -> Self {
        let kvd = types::kv_dim(config);
        let initial_pages_per_layer = config.block_size / PAGE_SIZE + 1;
        let initial_total = initial_pages_per_layer * config.n_layer;

        Self {
            pages: (0..initial_total)
                .map(|_| vec![0.0; PAGE_SIZE * kvd * 2])
                .collect(),
            layer_page_tables: (0..config.n_layer)
                .map(|_| (0..max_sequences).map(|_| Vec::new()).collect())
                .collect(),
            // All initial pages start as free; preallocate to avoid first-grow realloc.
            free_pages: (0..initial_total).collect(),
            page_ref_counts: vec![0; initial_total],
            kv_dim: kvd,
            kv_page_size: PAGE_SIZE * kvd,
            total_pages: initial_total,
        }
    }

    /// Allocate a new page. Reuse from free list or grow the pool.
    fn alloc_page(&mut self) -> usize {
        let idx = if let Some(idx) = self.free_pages.pop() {
            self.pages[idx].fill(0.0);
            idx
        } else {
            self.pages.push(vec![0.0; PAGE_SIZE * self.kv_dim * 2]);
            let idx = self.total_pages;
            self.total_pages += 1;
            self.page_ref_counts.push(0);
            idx
        };
        self.page_ref_counts[idx] += 1;
        idx
    }

    /// Ensure sequence `seq_idx` has enough pages to cover position `pos` for all layers.
    ///
    /// Uses stack-allocated `ArrayVec` for scratch (bounded to 128 layers) — zero
    /// heap allocation in the hot path. Ported from riir-engine.
    pub fn ensure_pages(&mut self, seq_idx: usize, pos: usize) {
        use arrayvec::ArrayVec;
        let pages_needed = pos / PAGE_SIZE + 1;

        // Grow sequence slots if needed (no page allocation, just empty vecs)
        for layer_tables in &mut self.layer_page_tables {
            while seq_idx >= layer_tables.len() {
                layer_tables.push(Vec::new());
            }
        }

        // Collect deficits into a stack-allocated array.
        // Most models have <= 128 layers; ArrayVec avoids per-call heap allocation.
        let mut deficits = ArrayVec::<usize, 128>::new();
        for lt in &self.layer_page_tables {
            deficits.push(pages_needed.saturating_sub(lt[seq_idx].len()));
        }
        let total_new: usize = deficits.iter().copied().sum();

        // Distribute newly-allocated page indices back into the layer tables.
        //
        // Fast path (the common case — autoregressive decode advances `pos` by
        // 1, so each layer's deficit is 0 or 1, total_new ≤ n_layers ≤ 128):
        // allocate into a flat stack `ArrayVec<usize, 128>` and `extend_from_slice`
        // per layer. Zero heap allocation regardless of deficit distribution.
        //
        // Slow path (prefill / large position jump where total_new > 128): fall
        // back to one heap `Vec<usize>` per layer-with-deficit. Matches the
        // previous behavior; the cost is dominated by the page-data allocation
        // itself, not the index Vec.
        if total_new <= 128 {
            let mut flat_new_pages = ArrayVec::<usize, 128>::new();
            for _ in 0..total_new {
                flat_new_pages.push(self.alloc_page());
            }
            let mut cursor = 0usize;
            for (layer_tables, &deficit) in self.layer_page_tables.iter_mut().zip(&deficits) {
                if deficit > 0 {
                    layer_tables[seq_idx]
                        .extend_from_slice(&flat_new_pages[cursor..cursor + deficit]);
                    cursor += deficit;
                }
            }
            debug_assert_eq!(cursor, total_new, "distributed all allocated pages");
        } else {
            // Slow path: per-layer heap Vecs (original behavior).
            let mut new_pages = ArrayVec::<Vec<usize>, 128>::new();
            for &deficit in &deficits {
                let pages: Vec<usize> = (0..deficit).map(|_| self.alloc_page()).collect();
                new_pages.push(pages);
            }
            for (layer_tables, pages) in self.layer_page_tables.iter_mut().zip(new_pages) {
                layer_tables[seq_idx].extend(pages);
            }
        }
    }

    /// Write K and V for a token position in a specific layer.
    /// Layout per page: `[K_data | V_data]` where each is `PAGE_SIZE * kv_dim` floats.
    #[inline]
    pub fn write_kv(&mut self, layer_idx: usize, seq_idx: usize, pos: usize, k: &[f32], v: &[f32]) {
        let page_local = pos % PAGE_SIZE;
        let page_list_idx = pos / PAGE_SIZE;
        let pidx = self.layer_page_tables[layer_idx][seq_idx][page_list_idx];
        let page = &mut self.pages[pidx];
        let k_off = page_local * self.kv_dim;
        let v_off = self.kv_page_size + page_local * self.kv_dim;
        page[k_off..k_off + self.kv_dim].copy_from_slice(k);
        page[v_off..v_off + self.kv_dim].copy_from_slice(v);
    }

    /// Read K and V for a token position in a specific layer.
    #[inline]
    pub fn read_kv(
        &self,
        layer_idx: usize,
        seq_idx: usize,
        pos: usize,
        k: &mut [f32],
        v: &mut [f32],
    ) {
        let page_local = pos % PAGE_SIZE;
        let page_list_idx = pos / PAGE_SIZE;
        let pidx = self.layer_page_tables[layer_idx][seq_idx][page_list_idx];
        let page = &self.pages[pidx];
        let k_off = page_local * self.kv_dim;
        let v_off = self.kv_page_size + page_local * self.kv_dim;
        k.copy_from_slice(&page[k_off..k_off + self.kv_dim]);
        v.copy_from_slice(&page[v_off..v_off + self.kv_dim]);
    }

    /// Fork a sequence with copy-on-write semantics.
    /// Shares prefix pages up to `fork_at_pos`, allocates new pages on demand after fork.
    /// Returns the new sequence index.
    pub fn fork(&mut self, seq_idx: usize, fork_at_pos: usize) -> usize {
        let fork_page = fork_at_pos / PAGE_SIZE;
        let new_seq = self.layer_page_tables[0].len();

        for layer_tables in &mut self.layer_page_tables {
            let source = &layer_tables[seq_idx];
            let shared_pages = source[..fork_page.min(source.len())].to_vec();
            // Increment ref counts for shared pages (Issue 053)
            for &pidx in &shared_pages {
                self.page_ref_counts[pidx] += 1;
            }
            layer_tables.push(shared_pages);
        }

        new_seq
    }

    /// Rollback a sequence to a given position, freeing exclusive pages.
    ///
    /// Truncates page tables to keep only pages covering positions `[0..rollback_to_pos)`.
    /// Pages that are exclusively owned by this sequence (not referenced by any other
    /// sequence in any layer) are returned to the free list for reuse.
    ///
    /// This is the "page table CoW rollback" — no data is copied, only page table
    /// entries are manipulated and exclusive pages are recycled.
    pub fn rollback(&mut self, seq_idx: usize, rollback_to_pos: usize) {
        let keep_count = rollback_to_pos / PAGE_SIZE;

        // Issue 053: use ref counts for O(1) exclusive-page detection instead of
        // building a HashSet by scanning all sequences across all layers (O(N×P×L)).
        // Decrement ref count for each removed page; if count reaches 0, it's exclusive.
        //
        // Pop from the end (no intermediate Vec) — the previous form allocated
        // a `Vec<usize>` per layer per rollback just to iterate it once.
        for layer_tables in &mut self.layer_page_tables {
            if seq_idx >= layer_tables.len() {
                continue;
            }
            let table = &mut layer_tables[seq_idx];
            while table.len() > keep_count {
                // SAFETY: we just checked `table.len() > keep_count`, so the
                // table is non-empty; `pop` returns the last element.
                let pidx = table.pop().expect("checked non-empty above");
                self.page_ref_counts[pidx] -= 1;
                if self.page_ref_counts[pidx] == 0 {
                    self.free_pages.push(pidx);
                }
            }
        }
    }

    /// Reset all sequences and free all pages.
    pub fn reset(&mut self) {
        for layer_tables in &mut self.layer_page_tables {
            for table in layer_tables.iter_mut() {
                self.free_pages.append(table);
            }
        }
        // Zero all ref counts since all page tables are cleared
        self.page_ref_counts.fill(0);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Issue 771 T2 — the prefix-sharing seam for external prefix indexes
    // (katgpt-kv's radix_prefix tree). Pure pool mechanics: the tree owns
    // page INDICES, these calls move ref-count ownership. Address stability:
    // none of these move or reorder `pages` — captured CUDA graphs that
    // baked buffer addresses stay valid (alloc reuses free-list slots and
    // only ever appends to the pool).
    // ──────────────────────────────────────────────────────────────────────

    /// Read `seq_idx`'s page tables for the first `chunks` chunks:
    /// `tables[layer][chunk] = page index`. Caller-owned allocation (a
    /// read for the index side; no ref-count change).
    pub fn chunk_page_tables(&self, seq_idx: usize, chunks: usize) -> Vec<Vec<usize>> {
        self.layer_page_tables
            .iter()
            .map(|lt| lt[seq_idx][..chunks].to_vec())
            .collect()
    }

    /// Take the tree's hold on `tables[layer][chunk]` pages: ref-count++
    /// per entry. Called for the NEW chunks an insert indexes (chunks the
    /// tree did not already hold).
    pub fn retain_chunk_pages(&mut self, tables: &[Vec<usize>]) {
        for lt in tables {
            for &pidx in lt {
                self.page_ref_counts[pidx] += 1;
            }
        }
    }

    /// Drop a hold taken by [`Self::retain_chunk_pages`]: ref-count-- per
    /// entry; a page reaching 0 returns to the free list. Tree eviction
    /// path — never frees a page a live sequence still references (its own
    /// hold keeps the count > 0).
    pub fn release_chunk_pages(&mut self, tables: &[Vec<usize>]) {
        for lt in tables {
            for &pidx in lt {
                self.page_ref_counts[pidx] -= 1;
                if self.page_ref_counts[pidx] == 0 {
                    self.free_pages.push(pidx);
                }
            }
        }
    }

    /// Install `tables` as `seq_idx`'s page tables (ref-count++ per entry —
    /// the adopting request's hold) and grow sequence slots as needed. The
    /// request then prefills ONLY positions `tables' chunk count × PAGE_SIZE
    /// and beyond; writing below that would corrupt every sharer (chunk-floor
    /// matching guarantees the suffix never lands in a shared page).
    pub fn adopt_chunk_pages(&mut self, seq_idx: usize, tables: &[Vec<usize>]) {
        for (layer_tables, lt) in self.layer_page_tables.iter_mut().zip(tables) {
            while seq_idx >= layer_tables.len() {
                layer_tables.push(Vec::new());
            }
            layer_tables[seq_idx] = lt.clone();
            for &pidx in lt {
                self.page_ref_counts[pidx] += 1;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Raven RSM — Routing Slot Memory (O(1) KV replacement for draft model)
// ──────────────────────────────────────────────────────────────────────────

/// Raven Routing Slot Memory — O(1) KV replacement for the draft model.
///
/// Fixed-size `[num_slots × kv_dim]` memory updated via sparse Top-K routing.
/// Unselected slots are completely frozen — perfect for preserving struct
/// definitions and imports while churning through syntax tokens.
pub struct RavenKVCache {
    // ── Vec fields first (ptr+len+cap = 24 bytes, 8-byte aligned) ──
    /// Key memory: [num_slots × kv_dim]
    pub keys: Vec<f32>,
    /// Value memory: [num_slots × kv_dim]
    pub values: Vec<f32>,
    // Pre-allocated buffers for zero-alloc router computation
    pub router_scored: Vec<(usize, f32)>, // [num_slots]
    pub router_r_t: Vec<f32>,             // [num_slots]
    /// Pre-allocated score buffer for raven_readout_into `[num_slots]`
    pub readout_scores: Vec<f32>,
    /// Pre-allocated output buffer for raven_readout_into `[kv_dim]`
    pub readout_output: Vec<f32>,
    // ── usize fields (8-byte aligned, no padding after Vecs) ──
    /// Number of memory slots
    pub num_slots: usize,
    /// Dimension of each KV entry (= kv_dim = n_kv_head × head_dim)
    pub kv_dim: usize,
    /// Top-K slots to update per token
    pub top_k: usize,
    // ── f32 field last (4-byte aligned, no trailing padding on 64-bit) ──
    /// Forget rate for gated update (negative = slower decay)
    pub forget_rate: f32,
}

impl RavenKVCache {
    pub fn new(config: &Config, num_slots: usize, top_k: usize) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            num_slots,
            kv_dim: kvd,
            top_k,
            keys: vec![0.0; num_slots * kvd],
            values: vec![0.0; num_slots * kvd],
            router_scored: vec![(0usize, 0.0f32); num_slots],
            router_r_t: vec![0.0f32; num_slots],
            readout_scores: vec![0.0; num_slots],
            readout_output: vec![0.0; kvd],
            forget_rate: -1.0,
        }
    }

    pub fn reset(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        // Use fill() instead of clear() to preserve pre-allocated capacity.
        // clear() drops len to 0, forcing reallocation on next use via resize.
        self.router_scored.fill((0, 0.0));
        self.router_r_t.fill(0.0);
        self.readout_scores.fill(0.0);
        self.readout_output.fill(0.0);
    }

    /// Export the current routing vector `r_t` (post-router, pre-update).
    /// Returns the normalized Top-K routing weights for all slots.
    /// Used by Phase 3 routed speculation to feed slot selection into anyrag.
    #[inline]
    pub fn r_t(&self) -> &[f32] {
        &self.router_r_t
    }
}

// ── Plan 320 Phase D3: sliding-bounded cache tests ──────────────────────────

#[cfg(test)]
mod sliding_bounded_tests {
    use super::*;

    /// Build a tiny config for testing: 3 layers, block_size=32, kvd=4.
    fn tiny_config() -> Config {
        Config {
            n_layer: 3,
            block_size: 32,
            n_embd: 16,
            n_head: 2,
            head_dim: 4,
            n_kv_head: 1,
            ..Config::micro()
        }
    }

    #[test]
    fn sliding_bounded_allocates_less_than_naive() {
        // G1: sliding-bounded cache should allocate less memory than the naive
        // all-full allocation for sliding layers.
        let config = tiny_config();
        let kvd = 4;
        let sliding_window = 8; // much less than block_size=32

        // Layer 0 and 2 are sliding; layer 1 is full.
        let sliding_layers = vec![true, false, true];
        let per_layer_kvd = vec![kvd; 3];

        // Naive allocation: all layers at block_size.
        let naive = MultiLayerKVCache::new_with_per_layer_kv_dim(&config, &per_layer_kvd);
        let naive_bytes: usize = naive
            .layers
            .iter()
            .map(|l| (l.key.len() + l.value.len()) * std::mem::size_of::<f32>())
            .sum();

        // Sliding-bounded allocation.
        let bounded = MultiLayerKVCache::new_gemma4_sliding_bounded(
            &config,
            &per_layer_kvd,
            &sliding_layers,
            sliding_window,
        );
        let bounded_bytes: usize = bounded
            .layers
            .iter()
            .map(|l| (l.key.len() + l.value.len()) * std::mem::size_of::<f32>())
            .sum();

        // Sliding layers: sw * kvd elements per buffer (key+value) → 2 * sw * kvd
        // Full layer: block_size * kvd elements per buffer → 2 * block_size * kvd
        // Plain-modulo 1× ring (Issue 683) — no mirror region.
        let sliding_layer_elems = sliding_window * kvd * 2; // key+value
        let full_layer_elems = config.block_size * kvd * 2;
        let expected_elems = 2 * sliding_layer_elems + full_layer_elems; // 2 sliding + 1 full
        let expected_bytes = expected_elems * std::mem::size_of::<f32>();
        assert_eq!(
            bounded_bytes, expected_bytes,
            "bounded bytes should match expected: 2 sliding + 1 full layer"
        );

        // The claim in the test's name — `naive_bytes` was computed above but
        // never checked, so G1's actual premise went unasserted.
        assert!(
            bounded_bytes < naive_bytes,
            "sliding-bounded ({bounded_bytes} B) should allocate less than naive ({naive_bytes} B)"
        );
    }

    #[test]
    fn sliding_capacity_reports_correct_values() {
        let config = tiny_config();
        let sliding_layers = vec![true, false, true];
        let per_layer_kvd = vec![4; 3];
        let sw = 8;

        let cache = MultiLayerKVCache::new_gemma4_sliding_bounded(
            &config,
            &per_layer_kvd,
            &sliding_layers,
            sw,
        );

        assert_eq!(cache.sliding_capacity(0), sw, "layer 0 is sliding");
        assert_eq!(cache.sliding_capacity(1), 0, "layer 1 is full");
        assert_eq!(cache.sliding_capacity(2), sw, "layer 2 is sliding");
    }

    #[test]
    fn unbounded_cache_has_zero_sliding_capacity() {
        // The standard new_with_per_layer_kv_dim should have all-zero capacities.
        let config = tiny_config();
        let per_layer_kvd = vec![4; 3];
        let cache = MultiLayerKVCache::new_with_per_layer_kv_dim(&config, &per_layer_kvd);

        for i in 0..config.n_layer {
            assert_eq!(
                cache.sliding_capacity(i),
                0,
                "layer {i} should be unbounded"
            );
        }
    }

    #[test]
    fn bounded_cache_respects_max_positions() {
        // The bounded constructor should size each layer at `max_positions * kvd`
        // instead of `block_size * kvd` (Issue 442 T4 — training KV cache
        // over-allocation that caused the CUDA-backend host OOM).
        let mut config = tiny_config();
        config.block_size = 1024; // simulate Gemma-4's large context window
        let per_layer_kvd = vec![4; 3];
        let max_positions = 32; // training seq_len
        let cache = MultiLayerKVCache::new_with_per_layer_kv_dim_bounded(
            &config,
            &per_layer_kvd,
            max_positions,
        );

        // Each layer should be sized at max_positions * kvd, NOT block_size * kvd.
        for (i, layer) in cache.layers.iter().enumerate() {
            let expected = max_positions * per_layer_kvd[i];
            assert_eq!(
                layer.key.len(),
                expected,
                "layer {i} key: expected {expected} (max_positions*kvd), got {}",
                layer.key.len()
            );
            assert_eq!(
                layer.value.len(),
                expected,
                "layer {i} value: expected {expected} (max_positions*kvd), got {}",
                layer.value.len()
            );
        }

        // Bounded cache should also report zero sliding_capacity (same as unbounded).
        for i in 0..config.n_layer {
            assert_eq!(cache.sliding_capacity(i), 0);
        }
    }

    #[test]
    fn bounded_cache_clamps_zero_max_positions() {
        // max_positions = 0 should be clamped to 1 (avoid zero-sized alloc).
        let config = tiny_config();
        let per_layer_kvd = vec![4; 3];
        let cache =
            MultiLayerKVCache::new_with_per_layer_kv_dim_bounded(&config, &per_layer_kvd, 0);
        for layer in &cache.layers {
            assert_eq!(layer.key.len(), 4, "clamped to 1*kvd");
            assert_eq!(layer.value.len(), 4, "clamped to 1*kvd");
        }
    }

    #[test]
    fn sliding_bounded_plain_modulo_write() {
        // Issue 683 T0(b): the ring convention is plain modulo — a write for
        // logical position `pos` lands at `pos % sw`, a single slot, no mirror.
        // The substrate performs no writes itself; this simulates the
        // consumer-side write the field doc contract describes.
        let config = tiny_config();
        let sw = 8;
        let kvd = 4;
        let sliding_layers = vec![true, false, false];
        let per_layer_kvd = vec![kvd; 3];

        let mut cache = MultiLayerKVCache::new_gemma4_sliding_bounded(
            &config,
            &per_layer_kvd,
            &sliding_layers,
            sw,
        );

        // 1× allocation: exactly sw * kvd floats — no mirror region exists.
        assert_eq!(cache.layers[0].key.len(), sw * kvd);
        assert_eq!(cache.layers[0].value.len(), sw * kvd);
        // Non-sliding layers keep the full block_size allocation.
        assert_eq!(cache.layers[1].key.len(), config.block_size * kvd);

        // Write position 3, then position 3 + sw (one full ring turn later):
        // both map to the SAME physical slot — the defining plain-modulo
        // property (the newest write wins).
        let layer = &mut cache.layers[0];
        for (pos, val) in [(3usize, 42.0f32), (3 + sw, 99.0f32)] {
            let off = (pos % sw) * kvd;
            layer.key[off..off + kvd].fill(val);
            layer.value[off..off + kvd].fill(val);
        }
        let off = (3 % sw) * kvd;
        assert_eq!(layer.key[off], 99.0, "pos and pos+sw share one slot");
        assert_eq!(layer.value[off], 99.0);
    }

    #[test]
    fn sliding_bounded_window_contiguous_when_aligned() {
        // Plain-modulo truth (Issue 683 T3): a window that does NOT straddle
        // the ring boundary (`t_start % sw <= pos % sw`) IS contiguous in the
        // buffer and maps linearly. After exactly 3 full ring turns (positions
        // 0..=3*sw-1), the current window [2*sw, 3*sw-1] is aligned
        // (t_start % sw == 0) and maps to the whole buffer [0, sw*kvd).
        let config = tiny_config();
        let sw = 8;
        let kvd = 4;
        let mut cache = MultiLayerKVCache::new_all_sliding_bounded(&config, sw);

        // Simulate the consumer write path; encode position identity into K.
        for pos in 0..3 * sw {
            for layer in &mut cache.layers {
                let off = (pos % sw) * kvd;
                layer.key[off..off + kvd].fill(pos as f32);
                layer.value[off..off + kvd].fill(pos as f32);
            }
        }

        let pos = 3 * sw - 1;
        let t_start = pos + 1 - sw; // = 2*sw, aligned: t_start % sw == 0
        assert_eq!(t_start % sw, 0);
        assert_eq!(pos % sw, sw - 1);
        assert!(t_start % sw <= pos % sw, "window must not straddle");

        // Contiguous read: ring slot i holds logical position t_start + i.
        let layer = &cache.layers[0];
        for i in 0..sw {
            let got = layer.key[i * kvd];
            let expected = (t_start + i) as f32;
            assert_eq!(
                got,
                expected,
                "ring slot {i} should hold position {}",
                t_start + i
            );
        }
    }

    #[test]
    fn sliding_bounded_straddling_window_needs_two_slice_gather() {
        // Plain-modulo truth (Issue 683 T3): a window that straddles the ring
        // boundary (`t_start % sw > pos % sw`) is NOT contiguous in the buffer.
        // The consumer must gather two slices. This test pins exactly that —
        // including that a naive contiguous read would run off the end of the
        // 1× buffer. (The old mirrored layout guaranteed contiguity here; that
        // guarantee is deliberately gone — the mirror was never exercised and
        // its saving was never measured.)
        let config = tiny_config();
        let sw = 8;
        let kvd = 4;
        let mut cache = MultiLayerKVCache::new_all_sliding_bounded(&config, sw);

        // Run past 3 full ring turns + 5, ending at pos = 3*sw+4 = 28.
        let n = 3 * sw + 5;
        for pos in 0..n {
            for layer in &mut cache.layers {
                let off = (pos % sw) * kvd;
                layer.key[off..off + kvd].fill(pos as f32);
                layer.value[off..off + kvd].fill(pos as f32);
            }
        }

        let pos = n - 1; // 28
        let t_start = pos + 1 - sw; // 21
        let t_n = pos - t_start + 1;
        assert_eq!(t_n, sw);
        assert!(
            t_start % sw > pos % sw,
            "chosen window must straddle ({} % {} = {} > {} % {} = {})",
            t_start,
            sw,
            t_start % sw,
            pos,
            sw,
            pos % sw
        );

        let layer = &cache.layers[0];
        // Honest pin: a contiguous read at (t_start % sw) would overflow the
        // window-sized buffer — contiguity is NOT provided.
        let naive_read_end = (t_start % sw) * kvd + t_n * kvd;
        assert!(
            naive_read_end > layer.key.len(),
            "naive contiguous read ({naive_read_end} floats) must exceed the 1x buffer ({})",
            layer.key.len()
        );

        // The consumer-side two-slice gather returns exactly the logically-
        // expected sequence [t_start..=pos].
        let head = &layer.key[(t_start % sw) * kvd..sw * kvd];
        let tail = &layer.key[..(pos % sw + 1) * kvd];
        let mut gathered: Vec<f32> = Vec::with_capacity(t_n);
        gathered.extend_from_slice(head);
        gathered.extend_from_slice(tail);
        let expected: Vec<f32> = (t_start..=pos).map(|t| t as f32).collect();
        for (i, chunk) in gathered.chunks(kvd).enumerate() {
            assert_eq!(chunk[0], expected[i], "gathered position {i} of the window");
        }
        assert_eq!(gathered.len(), t_n * kvd);
    }

    #[test]
    fn all_sliding_bounded_shape_and_zero_growth() {
        // Issue 683 T1 + T2 — the consumer's gate: run ≫ window positions
        // through the plain-modulo write path and assert every layer's buffer
        // is exactly window × kvd with sliding_capacity(l) == window — zero
        // growth across the run. (T2's assertion is `window * kvd`, not the
        // issue's original `2 * window * kvd`: that text predates the T0(b)
        // reframe to the plain-modulo 1× convention.)
        let config = tiny_config(); // 3 layers, kvd = 4
        let window = 8;
        let kvd = types::kv_dim(&config);
        assert_eq!(kvd, 4);

        let mut cache = MultiLayerKVCache::new_all_sliding_bounded(&config, window);

        // Shape: every layer sliding at `window`, buffer exactly window * kvd.
        for l in 0..config.n_layer {
            assert_eq!(cache.sliding_capacity(l), window, "layer {l} all-sliding");
            assert_eq!(cache.layers[l].key.len(), window * kvd);
            assert_eq!(cache.layers[l].value.len(), window * kvd);
        }

        // Zero growth: run 100 positions (12+ wraps) through the write path.
        let lens_before: Vec<(usize, usize)> = cache
            .layers
            .iter()
            .map(|l| (l.key.len(), l.value.len()))
            .collect();
        for pos in 0..100 {
            for layer in &mut cache.layers {
                let off = (pos % window) * kvd;
                layer.key[off..off + kvd].fill(pos as f32);
                layer.value[off..off + kvd].fill(pos as f32);
            }
            cache.advance_pos(pos);
        }
        let lens_after: Vec<(usize, usize)> = cache
            .layers
            .iter()
            .map(|l| (l.key.len(), l.value.len()))
            .collect();
        assert_eq!(
            lens_before, lens_after,
            "buffers must not grow across the run"
        );
        for l in 0..config.n_layer {
            assert_eq!(cache.layers[l].key.len(), window * kvd);
            assert_eq!(cache.sliding_capacity(l), window);
        }
        assert_eq!(cache.fill_pos(), 100);
    }

    #[test]
    fn all_sliding_bounded_panics_on_zero_window() {
        let config = tiny_config();
        let result = std::panic::catch_unwind(|| {
            MultiLayerKVCache::new_all_sliding_bounded(&config, 0);
        });
        assert!(
            result.is_err(),
            "window == 0 must panic (use `new` for unbounded)"
        );
    }

    #[test]
    fn all_sliding_bounded_snapshot_restore_roundtrip() {
        // Sliding consistency of the snapshot/restore path (Issue 683): the
        // snapshot captures the physical ring contents (clamped to the
        // window-sized buffer) and restore round-trips them exactly, so a
        // speculative rollback recovers the ring state.
        let config = tiny_config();
        let window = 8;
        let kvd = 4;
        let mut cache = MultiLayerKVCache::new_all_sliding_bounded(&config, window);

        // Write 20 positions (2.5 wraps), then snapshot at pos = 20 > window —
        // must not panic despite pos * kvd exceeding the buffer.
        for pos in 0..20 {
            for layer in &mut cache.layers {
                let off = (pos % window) * kvd;
                layer.key[off..off + kvd].fill(pos as f32);
                layer.value[off..off + kvd].fill(pos as f32);
            }
            cache.advance_pos(pos);
        }
        let snapshot = cache.snapshot(20, &config);
        // Full physical ring captured (pos * kvd clamped to window * kvd).
        for layer in &snapshot.layers {
            assert_eq!(layer.key.len(), window * kvd);
            assert_eq!(layer.value.len(), window * kvd);
        }

        // Corrupt, then restore — ring contents come back exactly.
        for layer in &mut cache.layers {
            layer.key.fill(-1.0);
            layer.value.fill(-1.0);
        }
        cache.restore(&snapshot, &config);
        for (l, layer) in cache.layers[..config.n_layer].iter().enumerate() {
            for i in 0..window {
                // Ring slot i holds the most recent position ≡ i (mod window),
                // i.e. the unique p in [pos-window, pos) with p % window == i.
                let expected_pos = 20 - window + ((i + window - ((20 - window) % window)) % window);
                assert_eq!(
                    layer.key[i * kvd],
                    expected_pos as f32,
                    "layer {l} ring slot {i} after restore"
                );
            }
        }
    }
}
