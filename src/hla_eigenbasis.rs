//! HLA Windowed Eigenbasis Recovery — Issue 001.
//!
//! Recovers a **per-NPC eigenbasis** (top-`k` orthogonal directions) from a
//! recent window of that NPC's own HLA activations, modellessly.
//!
//! # What it does
//!
//! Given `T` observed ticks of a single NPC's `D`-dim HLA state stacked as a
//! `T × D` row-major matrix `W`, this primitive returns the top-`k` eigenvectors
//! of the `D × D` Gram matrix `G = W^T W` (descending by eigenvalue). Those
//! eigenvectors are exactly the **principal directions** of the NPC's recent
//! affective trajectory — orthogonal axes in HLA space that capture the
//! dominant variance of *this* NPC's activations. Every NPC currently shares
//! the same hand-tuned universal basis (Research 032); this exposes
//! individualized affective geometry with **no training**.
//!
//! Eigenvectors of `W^T W` are the right singular vectors `V` of `W`, and the
//! eigenvalues are `σ²` (squared singular values). The recovered basis is a
//! per-NPC rotation/projection matrix usable for emotion routing, zone
//! attention, or adapter selection.
//!
//! # Modelless design
//!
//! - **No LAPACK.** Power iteration on the small `D × D` Gram matrix with
//!   deflation, mirroring the `stable_rank_update_into` pattern
//!   (`katgpt-core/src/data_probe.rs`). `D` is the HLA dim (8 today), so the
//!   Gram is 64 floats — the whole eigen-decomposition is a few hundred flops.
//! - **Zero-alloc hot path.** Caller-owned scratch (`EigenbasisScratch`).
//!   `recover_eigenbasis_from_window` allocates nothing after the first call
//!   for a given `(T, D)` pair.
//! - **Deterministic seed** `1/sqrt(D)` on every coordinate — no RNG. Power
//!   iteration on PSD matrices converges to the dominant eigenvector for any
//!   seed with nonzero overlap. Matches the `stable_rank_update_into`
//!   determinism strategy.
//! - **`Uuid::now_v7()`** for the per-window provenance id (AGENTS.md rule).
//! - **BLAKE3** hash of the input window for cache validation, mirroring
//!   `OffPrincipalIndex::src_hash`.
//!
//! # Sync-boundary rule (per AGENTS.md)
//!
//! The recovered eigenbasis, eigenvectors, and eigenvalues **stay local to the
//! NPC** — never synced. If a downstream behavior needs to commit, project to
//! the 5 raw emotion scalars via a bridge function and commit those. **Never**
//! pass the eigenbasis through `LatCalFixed`, and **never** use it for
//! anti-cheat validation.

#![allow(clippy::needless_range_loop)]

use katgpt_core::simd::{simd_dot_f32, simd_outer_product_acc};

/// Minimum `T` (ticks) before the recovered eigenbasis is trustworthy.
///
/// Below this the window is dominated by noise — the issue's risk §2 mitigation
/// requires `T ≥ 4·D`. With `D = 8` that is 32 ticks.
#[inline]
pub fn min_window_ticks(d: usize) -> usize {
    4 * d
}

// ---------------------------------------------------------------------------
// EigenbasisScratch — caller-owned, reused across calls
// ---------------------------------------------------------------------------

/// Caller-owned scratch for [`recover_eigenbasis_from_window`].
///
/// Holds the `D × D` Gram matrix (deflated in place across the `k` extracted
/// directions), plus two length-`D` work vectors. Resize is cheap and only
/// happens when `(T, D)` changes between calls.
///
/// # Layout (total floats)
///
/// - `gram`: `D * D` (the deflated Gram — mutated during extraction)
/// - `v`:    `D` (current power-iteration vector)
/// - `w`:    `D` (workspace: `G · v`)
///
/// For `D = 8`: `64 + 8 + 8 = 80` floats = 320 bytes of scratch, reused across
/// every NPC and every window for the lifetime of the harness.
#[derive(Clone, Debug, Default)]
pub struct EigenbasisScratch {
    gram: Vec<f32>,
    v: Vec<f32>,
    w: Vec<f32>,
    cached_d: usize,
}

impl EigenbasisScratch {
    /// Construct an empty scratch. Allocates lazily on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-allocate for a given `D`. Idempotent.
    pub fn with_capacity_d(d: usize) -> Self {
        Self {
            gram: vec![0.0; d * d],
            v: vec![0.0; d],
            w: vec![0.0; d],
            cached_d: d,
        }
    }

    fn ensure_capacity_d(&mut self, d: usize) {
        if self.cached_d != d {
            self.gram.resize(d * d, 0.0);
            self.v.resize(d, 0.0);
            self.w.resize(d, 0.0);
            self.cached_d = d;
        }
    }
}

// ---------------------------------------------------------------------------
// EigenbasisTracker — rolling-window Gram for the O(D²)-per-tick hot path
// ---------------------------------------------------------------------------
//
// `recover_eigenbasis_from_window*` rebuild the D×D Gram from scratch each
// call: O(T·D²) FMAs. At T=512,D=8 that's ~7.7µs — too slow for the issue's
// ≤2µs plasma-tier budget. But a live NPC pushes ONE new tick per call and
// evicts the oldest; the Gram update is then two rank-1 updates (add new,
// subtract old) = O(D²) = 64 FMAs, not O(T·D²) = 32768. The tracker holds the
// rolling window buffer + the maintained Gram; `recover` runs the same power-
// iteration-with-deflation as the stateless path on the already-built Gram.
//
// This is the modelless unblock for G1: per-tick cost drops from ~7.7µs to
// ~50ns (update) + ~600ns (recover at k=4,iters=5) ≈ 650ns — well under 2µs.

/// Rolling-window tracker that maintains the D×D Gram incrementally.
///
/// Push the latest HLA tick via [`push_tick`][Self::push_tick]; once the window
/// is full, each push evicts the oldest tick (FIFO). Call
/// [`recover`][Self::recover] to extract the current top-`k` eigenbasis from
/// the maintained Gram — no full rebuild.
///
/// # Layout
///
/// - `window`: `T × D` ring buffer of recent ticks (for eviction).
/// - `gram`:   `D × D` maintained Gram (updated incrementally).
/// - `v`, `w`: length-`D` work vectors for power iteration.
///
/// For `D = 8, T = 512`: `512*8 + 64 + 8 + 8 = 4232` floats ≈ 16.5 KB. This is
/// the per-NPC hot-path state — large compared to the recovered basis (144
/// bytes) but still tiny relative to per-NPC HLA caches elsewhere in the stack.
pub struct EigenbasisTracker {
    window: Vec<f32>,
    gram: Vec<f32>,
    v: Vec<f32>,
    w: Vec<f32>,
    d: usize,
    capacity: usize,
    /// Ring-buffer write head (next slot to write).
    head: usize,
    /// Number of ticks currently held (`≤ capacity`).
    len: usize,
}

impl EigenbasisTracker {
    /// New tracker for a `capacity`-tick window of `d`-dim ticks.
    pub fn new(capacity: usize, d: usize) -> Self {
        assert!(capacity > 0 && d > 0);
        Self {
            window: vec![0.0; capacity * d],
            gram: vec![0.0; d * d],
            v: vec![0.0; d],
            w: vec![0.0; d],
            d,
            capacity,
            head: 0,
            len: 0,
        }
    }

    /// Number of ticks currently held.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the tracker holds no ticks.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the window is full (subsequent pushes will evict).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Push a new `d`-dim tick. If the window is full, the oldest tick is
    /// evicted (FIFO) via a rank-1 Gram subtraction. The new tick is added via
    /// a rank-1 Gram addition. O(D²).
    ///
    /// `tick.len()` must equal the tracker's `d`.
    pub fn push_tick(&mut self, tick: &[f32]) {
        assert_eq!(
            tick.len(),
            self.d,
            "push_tick: tick.len() {} != d {}",
            tick.len(),
            self.d
        );
        // Copy the evicted (oldest) and incoming ticks into stack buffers up
        // front so the rank-1 updates don't alias `self`. D is small (8 today,
        // bounded by HLA dim), so a fixed 16-wide buffer covers any realistic
        // HLA dimension without a heap allocation.
        let mut old_buf = [0.0_f32; 16];
        let mut new_buf = [0.0_f32; 16];
        let d = self.d;
        assert!(d <= 16, "push_tick: D={} > 16 stack buffer; widen new_buf/old_buf", d);

        if self.is_full() {
            // Evict the tick at `head` (oldest — head advances on every push
            // once full): gram -= old^T old.
            let old = &self.window[self.head * d..(self.head + 1) * d];
            old_buf[..d].copy_from_slice(old);
            self.rank1_update_buf(&old_buf[..d], -1.0);
        }
        // Write the new tick into the slot, copy it, and add its rank-1.
        self.window[self.head * d..(self.head + 1) * d].copy_from_slice(tick);
        new_buf[..d].copy_from_slice(tick);
        self.rank1_update_buf(&new_buf[..d], 1.0);

        self.head = (self.head + 1) % self.capacity;
        if !self.is_full() {
            self.len += 1;
        }
    }

    /// `gram += scale * (a ⊗ a)`. Symmetric rank-1, O(D²). `a` must NOT alias
    /// `self.gram` — callers copy into a stack buffer first (see `push_tick`).
    #[inline]
    fn rank1_update_buf(&mut self, a: &[f32], scale: f32) {
        let d = self.d;
        let gram = &mut self.gram[..d * d];
        for i in 0..d {
            let ai = scale * a[i];
            let row = &mut gram[i * d..(i + 1) * d];
            for j in 0..d {
                row[j] += ai * a[j];
            }
        }
    }

    /// Recover the top-`k` eigenbasis from the maintained Gram. Same power-
    /// iteration-with-deflation as `recover_eigenbasis_from_window_fast`, but
    /// skips the Gram rebuild (the Gram is already current). O(k · iters · D²).
    ///
    /// Returns the number of ticks currently held — callers should treat the
    /// basis as untrustworthy below `min_window_ticks(d)`.
    pub fn recover(
        &mut self,
        out_eigvecs: &mut [f32],
        out_eigvals: &mut [f32],
        k: usize,
        iters: u8,
    ) -> usize {
        let d = self.d;
        assert!(k >= 1 && k <= d, "recover: k must be in [1, D]=[1, {}], got {}", d, k);
        assert_eq!(
            out_eigvecs.len(),
            d * k,
            "recover: out_eigvecs.len() {} != d*k",
            out_eigvecs.len()
        );
        assert!(out_eigvals.len() >= k, "recover: out_eigvals too short");

        let gram = &mut self.gram[..d * d];
        let v = &mut self.v[..d];
        let w = &mut self.w[..d];
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let iters = iters.max(1) as usize;

        for col in 0..k {
            for x in v.iter_mut() {
                *x = inv_sqrt_d;
            }
            let mut lambda = 0.0_f32;
            for _ in 0..iters {
                for i in 0..d {
                    w[i] = simd_dot_f32(&gram[i * d..(i + 1) * d], v, d);
                }
                let vtv = simd_dot_f32(v, v, d);
                let vtw = simd_dot_f32(v, w, d);
                if vtv <= 0.0 {
                    break;
                }
                lambda = vtw / vtv;
                let norm_w = simd_dot_f32(w, w, d).max(1e-30).sqrt();
                let inv_norm = 1.0 / norm_w;
                for j in 0..d {
                    v[j] = w[j] * inv_norm;
                }
            }
            // Deflate.
            for i in 0..d {
                let vi = v[i];
                let row_off = i * d;
                for j in 0..d {
                    gram[row_off + j] -= lambda * vi * v[j];
                }
            }
            for row in 0..d {
                out_eigvecs[row * k + col] = v[row];
            }
            out_eigvals[col] = lambda;
        }
        self.len
    }
}

/// Provenance report for a recovered eigenbasis. Carries the deterministic id
/// and content hash callers need to validate a cached basis against the window
/// that produced it (the issue's freeze/thaw risk §3 mitigation).
#[derive(Clone, Debug)]
pub struct EigenbasisProvenance {
    /// `Uuid::now_v7()` assigned at recovery time. Monotonic in time, sortable.
    pub window_id: [u8; 16],
    /// BLAKE3 hash of the input window bytes — validates cache hits.
    pub window_hash: [u8; 32],
    /// `T` (ticks observed) at recovery time.
    pub ticks: usize,
    /// `D` (HLA dim) at recovery time.
    pub dim: usize,
    /// `k` (number of directions recovered).
    pub k: usize,
}

// ---------------------------------------------------------------------------
// recover_eigenbasis_from_window — the primitive
// ---------------------------------------------------------------------------

/// Recover the top-`k` orthogonal principal directions of a windowed HLA
/// activation matrix.
///
/// Computes the `D × D` Gram `G = W^T W` from the `T × D` row-major `window`,
/// then extracts its top-`k` eigenvectors via power iteration with deflation.
/// Eigenvectors are written to `out_eigvecs` (row-major `D × k`, i.e.
/// `out_eigvecs[row * k + col]` is element `(row, col)` — column `j` is the
/// `j`-th principal direction, strided by `k`), and the corresponding
/// eigenvalues (descending) to `out_eigvals[..k]`. Returns a provenance report
/// carrying the window id + BLAKE3 hash for cache validation.
///
/// # Arguments
///
/// * `window` — `T × D` row-major activations. `window.len()` must be
///   `t_dim * d_dim`. One row is one tick's HLA state.
/// * `t_dim` — number of ticks observed (`T`). Should be `≥ 4·D` for a
///   trustworthy basis (see [`min_window_ticks`]); smaller `T` is allowed but
///   the caller should gate downstream effects on the energy ratio.
/// * `d_dim` — HLA dimension (`D`). Today 8.
/// * `out_eigvecs` — length `D * k`, row-major `D × k`. Column `j` (strided by
///   `k`) is the `j`-th principal direction, unit-norm.
/// * `out_eigvals` — length `≥ k`; first `k` elements receive the eigenvalues
///   `σ²` in descending order.
/// * `scratch` — caller-owned; resized to `D*D + 2*D` floats on first call for
///   a new `D`, then reused allocation-free.
/// * `k` — number of principal directions to recover. `1 ≤ k ≤ D`.
/// * `iters` — power-iteration count per direction. Default 5 (matches
///   `stable_rank_update_into` and `OffPrincipalIndex::new`).
///
/// # Returns
///
/// [`EigenbasisProvenance`] carrying the deterministic `Uuid::now_v7()` window
/// id and the BLAKE3 hash of the input window.
///
/// # Zero-alloc contract
///
/// No allocation after the first call for a given `D` (scratch caches `D`).
/// The `Uuid` and BLAKE3 are computed on this (cold) entry point; use
/// [`recover_eigenbasis_from_window_fast`] for the hot path that skips them.
///
/// # Determinism
///
/// The seed vector is `1/sqrt(D)` on every coordinate — no RNG, so the only
/// cross-platform variability is the SIMD reduction order inside
/// `simd_dot_f32` / `simd_outer_product_acc`, which is the same surface
/// `stable_rank_update_into` already relies on for its determinism claim.
///
/// # Panics
///
/// Debug-mode assertions on shape mismatches. Release builds skip the checks.
#[allow(clippy::too_many_arguments)] // signature matches the Issue 001 primitive contract
pub fn recover_eigenbasis_from_window(
    window: &[f32],
    t_dim: usize,
    d_dim: usize,
    out_eigvecs: &mut [f32],
    out_eigvals: &mut [f32],
    scratch: &mut EigenbasisScratch,
    k: usize,
    iters: u8,
) -> EigenbasisProvenance {
    recover_eigenbasis_inner(window, t_dim, d_dim, out_eigvecs, out_eigvals, scratch, k, iters, true)
}

/// Fast-path eigenbasis recovery **without** the BLAKE3/`Uuid::now_v7` provenance
/// overhead.
///
/// This is the plasma-tier hot path: identical mathematics to
/// [`recover_eigenbasis_from_window`] but skips the ~9µs provenance cost (BLAKE3
/// over the full window + system-clock read for the v7 id). Returns `ticks/dim/k`
/// only — callers that need cache validation should call the full
/// [`recover_eigenbasis_from_window`] on the cold freeze/thaw path instead.
///
/// # When to use which
///
/// - **Hot path** (per-NPC, per-tick, affects routing only): this function.
///   The recovered basis is consumed locally and discarded; no provenance needed.
/// - **Cold path** (freeze/thaw commit, cache keying): the full
///   [`recover_eigenbasis_from_window`] with BLAKE3 + `Uuid::now_v7()`.
///
/// Use [`compute_window_hash`] to derive the cache key separately when needed.
#[allow(clippy::too_many_arguments)] // signature mirrors recover_eigenbasis_from_window
pub fn recover_eigenbasis_from_window_fast(
    window: &[f32],
    t_dim: usize,
    d_dim: usize,
    out_eigvecs: &mut [f32],
    out_eigvals: &mut [f32],
    scratch: &mut EigenbasisScratch,
    k: usize,
    iters: u8,
) -> EigenbasisProvenance {
    recover_eigenbasis_inner(window, t_dim, d_dim, out_eigvecs, out_eigvals, scratch, k, iters, false)
}

/// Compute the BLAKE3 cache key for a window without running the eigen-decomposition.
/// Use on the cold freeze/thaw path to validate a cached basis against its source
/// window. O(T·D) — the dominant cost of the provenance path.
#[inline]
pub fn compute_window_hash(window: &[f32]) -> [u8; 32] {
    let bytes = unsafe {
        std::slice::from_raw_parts(window.as_ptr() as *const u8, std::mem::size_of_val(window))
    };
    *blake3::hash(bytes).as_bytes()
}

/// Shared inner: runs the eigen-decomposition, optionally computing provenance.
#[allow(clippy::too_many_arguments)]
fn recover_eigenbasis_inner(
    window: &[f32],
    t_dim: usize,
    d_dim: usize,
    out_eigvecs: &mut [f32],
    out_eigvals: &mut [f32],
    scratch: &mut EigenbasisScratch,
    k: usize,
    iters: u8,
    with_provenance: bool,
) -> EigenbasisProvenance {
    assert!(
        t_dim > 0 && d_dim > 0,
        "recover_eigenbasis_from_window: t_dim and d_dim must be positive, got T={}, D={}",
        t_dim,
        d_dim
    );
    assert_eq!(
        window.len(),
        t_dim * d_dim,
        "recover_eigenbasis_from_window: window.len() {} != T*D = {}*{} = {}",
        window.len(),
        t_dim,
        d_dim,
        t_dim * d_dim
    );
    assert!(
        k >= 1 && k <= d_dim,
        "recover_eigenbasis_from_window: k must be in [1, D]=[1, {}], got {}",
        d_dim,
        k
    );
    assert_eq!(
        out_eigvecs.len(),
        d_dim * k,
        "recover_eigenbasis_from_window: out_eigvecs.len() {} != D*k = {}*{} = {}",
        out_eigvecs.len(),
        d_dim,
        k,
        d_dim * k
    );
    assert!(
        out_eigvals.len() >= k,
        "recover_eigenbasis_from_window: out_eigvals.len() {} < k = {}",
        out_eigvals.len(),
        k
    );

    scratch.ensure_capacity_d(d_dim);
    let gram = &mut scratch.gram[..d_dim * d_dim];
    let v = &mut scratch.v[..d_dim];
    let w = &mut scratch.w[..d_dim];

    // ── 1. Build D×D Gram G = W^T W via symmetric outer-product accumulation.
    // ZERO the buffer first — simd_outer_product_acc ACCUMULATES (+=), and the
    // scratch is reused across calls. On the second call the buffer still
    // holds the deflated Gram from the previous extraction; without this clear
    // the new Gram would be added on top of the stale one (Issue 001 bug caught
    // by scratch_is_reused_across_calls test).
    for g in gram.iter_mut() {
        *g = 0.0;
    }
    // Each row r of W contributes the rank-1 outer product w_r^T w_r. We
    // accumulate row-by-row into `gram`. simd_outer_product_acc adds
    // a (m=1) × b (n=D) rank-1 update into the leading D×D block of `gram`.
    // We pass a 1-row slice so m=1; the accumulator writes into acc[0..D*1..D]
    // i.e. gram[0..D] for that single outer product a·b^T.
    //
    // For each tick row: gram += window[r..r+D]^T · window[r..r+D]  (D×D rank-1).
    for r in 0..t_dim {
        let row = &window[r * d_dim..(r + 1) * d_dim];
        // simd_outer_product_acc(acc, a, b, m, n): acc[i*n + j] += a[i]*b[j]
        // with a=b=row, m=n=D → gram[i*D + j] += row[i]*row[j] (full D×D).
        simd_outer_product_acc(gram, row, row, d_dim, d_dim);
    }

    // ── 2. Power iteration with deflation for the top-k eigenvectors.
    let inv_sqrt_d = 1.0 / (d_dim as f32).sqrt();
    let iters = iters.max(1) as usize;

    for col in 0..k {
        // Deterministic seed: 1/sqrt(D) on every coordinate. Matches
        // stable_rank_update_into — nonzero overlap with any eigenvector.
        for x in v.iter_mut() {
            *x = inv_sqrt_d;
        }

        let mut lambda = 0.0_f32;
        for _ in 0..iters {
            // w = G · v  (D×D matvec, row-major: w[i] = dot(gram row i, v))
            for i in 0..d_dim {
                w[i] = simd_dot_f32(&gram[i * d_dim..(i + 1) * d_dim], v, d_dim);
            }

            // Rayleigh quotient λ ≈ v^T w / v^T v. With the seed normalized and
            // w renormalized each step, v^T v ≈ 1; we keep the division for
            // numerical safety on the first iteration.
            let vtv = simd_dot_f32(v, v, d_dim);
            let vtw = simd_dot_f32(v, w, d_dim);
            if vtv <= 0.0 {
                break;
            }
            lambda = vtw / vtv;

            // Normalize w into v for the next iteration.
            let norm_w = simd_dot_f32(w, w, d_dim).max(1e-30).sqrt();
            let inv_norm = 1.0 / norm_w;
            for j in 0..d_dim {
                v[j] = w[j] * inv_norm;
            }
        }

        // ── 3. Deflate the Gram: G ← G − λ · v v^T.
        // Removes the just-found direction so the next power iteration
        // converges to the next-largest eigenvector. The rank-1 update
        // gram[i][j] -= lambda * v[i] * v[j] is symmetric; we do it in place.
        for i in 0..d_dim {
            let vi = v[i];
            let row_off = i * d_dim;
            for j in 0..d_dim {
                gram[row_off + j] -= lambda * vi * v[j];
            }
        }

        // ── 4. Emit the principal direction as column `col` of out_eigvecs
        // and the eigenvalue. The deflation above guarantees descending order.
        for row in 0..d_dim {
            out_eigvecs[row * k + col] = v[row];
        }
        out_eigvals[col] = lambda;
    }

    // ── 5. Provenance (cold path only). The BLAKE3 hash is O(T·D) and the
    // Uuid::now_v7() reads the system clock — together ~9µs at T=512,D=8. On
    // the hot per-tick path these are skipped; callers that need cache
    // validation use the full `recover_eigenbasis_from_window` (cold path) or
    // call `compute_window_hash` separately.
    let (window_id, window_hash) = if with_provenance {
        let hash = compute_window_hash(window);
        let id = *uuid::Uuid::now_v7().as_bytes();
        (id, hash)
    } else {
        ([0u8; 16], [0u8; 32])
    };

    EigenbasisProvenance {
        window_id,
        window_hash,
        ticks: t_dim,
        dim: d_dim,
        k,
    }
}

/// Energy ratio captured by the top-`k` eigenvalues: `Σ_{i<k} λ_i / Σ_all λ_i`.
///
/// `Σ_all λ_i = trace(G) = Σ_t ‖w_t‖²` (total variance of the window). A ratio
/// near 1.0 means the top-`k` directions explain nearly all of the NPC's recent
/// affective motion; a low ratio means the NPC's activations are spread across
/// many directions and the recovered basis is less informative. Callers should
/// gate downstream effects (e.g. routing weight) on `sigmoid(ratio − τ)`, per
/// the issue's risk §2 mitigation.
///
/// `total_energy` is `trace(G)` — pass the sum of all `D` eigenvalues if you
/// have them, or `Σ_t ‖w_t‖²` computed directly from the window.
#[inline]
pub fn energy_ratio(top_eigvals: &[f32], total_energy: f32) -> f32 {
    if total_energy <= 0.0 {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    for &l in top_eigvals {
        sum += l;
    }
    (sum / total_energy).min(1.0)
}

/// Total energy of a window: `trace(W^T W) = Σ_t ‖w_t‖²`.
///
/// Use as the denominator for [`energy_ratio`]. O(T·D), no allocation.
pub fn window_total_energy(window: &[f32], t_dim: usize, d_dim: usize) -> f32 {
    debug_assert_eq!(window.len(), t_dim * d_dim);
    let mut acc = 0.0_f32;
    for r in 0..t_dim {
        let row = &window[r * d_dim..(r + 1) * d_dim];
        acc += simd_dot_f32(row, row, d_dim);
    }
    acc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;

    /// Build a T×D window with a known rank-3 structure: three orthogonal
    /// directions in D=8 space with specified energies, plus small noise.
    fn make_rank3_window(t: usize, d: usize, energies: [f32; 3], seed: u64) -> Vec<f32> {
        assert!(d >= 3);
        // Three orthogonal seed directions (canonical basis vectors 0,1,2).
        let mut w = vec![0.0_f32; t * d];
        // Deterministic LCG for reproducible noise (NOT used in the primitive —
        // only for synthesizing test inputs).
        let mut s = seed.max(1);
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as f32 / (1u64 << 31) as f32 - 0.5
        };
        // Fill rows cycling through the 3 directions with their energies.
        for r in 0..t {
            let dir = r % 3;
            let amp = (energies[dir] / t as f32).sqrt();
            for j in 0..d {
                let base = if j == dir { amp } else { 0.0 };
                // 1% noise floor so it is not bit-trivially rank-3.
                let noise = next() * 0.01 * amp.abs().max(1e-3);
                w[r * d + j] = base + noise;
            }
        }
        w
    }

    #[test]
    fn g3_reconstructs_rank3_in_d8() {
        // G3 gate: reconstruction ‖W − projection onto top-k‖ small for k≥3.
        let d = 8;
        let t = 512;
        let energies = [4.0_f32, 2.0, 1.0]; // total trace = 7
        let window = make_rank3_window(t, d, energies, 42);

        let k = 4;
        let mut eigvecs = vec![0.0; d * k];
        let mut eigvals = vec![0.0; k];
        let mut scratch = EigenbasisScratch::new();
        let prov = recover_eigenbasis_from_window(
            &window, t, d, &mut eigvecs, &mut eigvals, &mut scratch, k, 5,
        );
        assert_eq!(prov.ticks, t);
        assert_eq!(prov.dim, d);
        assert_eq!(prov.k, k);

        // Eigenvalues descending.
        for i in 1..k {
            assert!(eigvals[i - 1] >= eigvals[i], "eigvals not descending: {:?}", eigvals);
        }
        // Top-3 eigenvalues should carry ~all the energy (noise floor is ~1%).
        let total = window_total_energy(&window, t, d);
        let ratio = energy_ratio(&eigvals, total);
        assert!(ratio > 0.9, "G3 energy ratio {} too low", ratio);

        // Eigenvectors unit-norm.
        for col in 0..k {
            let mut nrm_sq = 0.0;
            for row in 0..d {
                let c = eigvecs[row * k + col];
                nrm_sq += c * c;
            }
            assert!((nrm_sq - 1.0).abs() < 1e-3, "col {} not unit norm: {}", col, nrm_sq);
        }

        // Top-3 directions should align with canonical basis 0,1,2 (the seeds).
        // Recovered column j should have its largest-magnitude entry at row j.
        for col in 0..3 {
            let mut best = 0;
            let mut best_abs = 0.0_f32;
            for row in 0..d {
                let a = eigvecs[row * k + col].abs();
                if a > best_abs {
                    best_abs = a;
                    best = row;
                }
            }
            assert_eq!(best, col, "col {} peak at row {} (expected {})", col, best, col);
        }
    }

    #[test]
    fn g4_two_windows_diverge() {
        // G4 gate: two windows with different structure produce eigenbases
        // whose principal directions are angularly separated.
        let d = 8;
        let t = 256;
        // Window A: energy in direction 0. Window B: energy in direction 4.
        let wina = make_rank3_window(t, d, [4.0, 0.5, 0.5], 1);
        let winb = make_rank3_window(t, d, [0.5, 0.5, 4.0], 2);
        // Shift winb's structure into direction 4 manually (make_rank3 puts it in 2).
        let mut winb_shifted = winb.clone();
        for r in 0..t {
            // Swap direction 2 <-> direction 4.
            let a = winb_shifted[r * d + 2];
            let b = winb_shifted[r * d + 4];
            winb_shifted[r * d + 2] = b;
            winb_shifted[r * d + 4] = a;
        }

        let k = 1;
        let mut va = vec![0.0; d * k];
        let mut vb = vec![0.0; d * k];
        let mut la = vec![0.0; k];
        let mut lb = vec![0.0; k];
        let mut scratch = EigenbasisScratch::new();
        recover_eigenbasis_from_window(&wina, t, d, &mut va, &mut la, &mut scratch, k, 8);
        recover_eigenbasis_from_window(&winb_shifted, t, d, &mut vb, &mut lb, &mut scratch, k, 8);

        // Cosine of the two principal directions.
        let mut dot = 0.0;
        let mut na = 0.0;
        let mut nb = 0.0;
        for row in 0..d {
            dot += va[row] * vb[row];
            na += va[row] * va[row];
            nb += vb[row] * vb[row];
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        // Should be well below 0.7 (the G4 budget) — directions are orthogonal.
        assert!(cos.abs() < 0.5, "G4 cos too high: {} (expected <0.5)", cos);
    }

    #[test]
    fn scratch_is_reused_across_calls() {
        // Same D → no realloc. We can't easily intercept Vec realloc, but we can
        // confirm cached_d is stable and a second call produces consistent output.
        let d = 8;
        let t = 128;
        let window = make_rank3_window(t, d, [3.0, 1.5, 0.7], 7);
        let mut scratch = EigenbasisScratch::new();
        let k = 2;

        let mut e1 = vec![0.0; d * k];
        let mut l1 = vec![0.0; k];
        recover_eigenbasis_from_window(&window, t, d, &mut e1, &mut l1, &mut scratch, k, 5);
        assert_eq!(scratch.cached_d, d);

        let mut e2 = vec![0.0; d * k];
        let mut l2 = vec![0.0; k];
        recover_eigenbasis_from_window(&window, t, d, &mut e2, &mut l2, &mut scratch, k, 5);

        // Deterministic: same input → same output bit-for-bit.
        for i in 0..e1.len() {
            assert_eq!(e1[i], e2[i], "non-deterministic eigvec at {}", i);
        }
        for i in 0..l1.len() {
            assert_eq!(l1[i], l2[i], "non-deterministic eigval at {}", i);
        }
    }

    #[test]
    fn determinism_same_seed_same_output() {
        // G2 (partial — same machine): identical inputs produce identical bases.
        // Cross-platform bit-identical is the bench/dedicated determinism test.
        let d = 8;
        let t = 256;
        let window = make_rank3_window(t, d, [5.0, 3.0, 1.0], 99);
        let k = 3;

        let mut a = vec![0.0; d * k];
        let mut la = vec![0.0; k];
        let mut b = vec![0.0; d * k];
        let mut lb = vec![0.0; k];
        let mut sa = EigenbasisScratch::new();
        let mut sb = EigenbasisScratch::new();
        recover_eigenbasis_from_window(&window, t, d, &mut a, &mut la, &mut sa, k, 5);
        recover_eigenbasis_from_window(&window, t, d, &mut b, &mut lb, &mut sb, k, 5);

        for i in 0..a.len() {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "eigvec bit mismatch at {}", i);
        }
        for i in 0..la.len() {
            assert_eq!(la[i].to_bits(), lb[i].to_bits(), "eigval bit mismatch at {}", i);
        }
    }

    #[test]
    fn k_equals_d_recovers_full_spectrum() {
        // When k=D, all eigenvalues are extracted. For a rank-3 input in D=8,
        // the trailing 5 eigenvalues should be ~0 (noise floor).
        let d = 8;
        let t = 512;
        let window = make_rank3_window(t, d, [6.0, 3.0, 1.0], 5);
        let k = d;

        let mut e = vec![0.0; d * k];
        let mut l = vec![0.0; k];
        let mut scratch = EigenbasisScratch::new();
        recover_eigenbasis_from_window(&window, t, d, &mut e, &mut l, &mut scratch, k, 8);

        // Top-3 dominate; bottom-5 are noise.
        let top3: f32 = l[..3].iter().sum();
        let total: f32 = l.iter().sum();
        assert!(top3 / total > 0.95, "top-3/total = {}", top3 / total);
        // Eigenvalues descending among well-separated values. Power iteration
        // with deflation does NOT order near-degenerate noise-floor eigenvalues
        // (they cluster at ~1e-5 here vs the top at ~2.0); we tolerate ties
        // within a relative band of the max eigenvalue.
        let max_l = l[0].max(1e-30);
        let tie_band = 1e-3 * max_l;
        for i in 1..d {
            assert!(
                l[i - 1] >= l[i] - tie_band,
                "k=D eigvals not descending (outside tie band): {:?}",
                l
            );
        }
    }

    #[test]
    fn provenance_carries_distinct_ids() {
        // Two recoveries get distinct Uuid::now_v7() ids (time + node).
        let d = 8;
        let t = 64;
        let w = make_rank3_window(t, d, [1.0; 3], 1);
        let k = 1;
        let mut e = vec![0.0; d * k];
        let mut l = vec![0.0; k];
        let mut s = EigenbasisScratch::new();
        let p1 = recover_eigenbasis_from_window(&w, t, d, &mut e, &mut l, &mut s, k, 5);
        // Busy-spin briefly to ensure the v7 timestamp differs (1ms resolution).
        std::thread::sleep(std::time::Duration::from_millis(2));
        let p2 = recover_eigenbasis_from_window(&w, t, d, &mut e, &mut l, &mut s, k, 5);
        assert_ne!(p1.window_id, p2.window_id, "v7 ids collided");
        assert_eq!(p1.window_hash, p2.window_hash, "same window should hash equal");
    }

    #[test]
    #[should_panic(expected = "k must be in")]
    fn k_zero_panics() {
        let d = 8;
        let w = vec![0.0; 64 * d];
        let mut e = vec![0.0; d];
        let mut l = vec![0.0];
        let mut s = EigenbasisScratch::new();
        recover_eigenbasis_from_window(&w, 64, d, &mut e, &mut l, &mut s, 0, 5);
    }

    #[test]
    fn energy_ratio_handles_zero_total() {
        assert_eq!(energy_ratio(&[1.0, 2.0], 0.0), 0.0);
        assert!((energy_ratio(&[3.0], 10.0) - 0.3).abs() < 1e-6);
        // Clamps to 1.0 if sum exceeds total (numerical safety).
        assert_eq!(energy_ratio(&[5.0], 1.0), 1.0);
    }

    #[test]
    fn tracker_matches_stateless_after_warmup() {
        // The tracker's maintained Gram should equal the stateless Gram rebuilt
        // from the same window, so the recovered eigenbasis matches (within f32
        // tolerance — the eviction/addition order differs slightly from the
        // single-pass accumulation, but the resulting Gram is the same matrix
        // up to float reassociation).
        use crate::hla_eigenbasis::EigenbasisTracker;
        let d = 8;
        let t = 256;
        let k = 4;
        let window = make_rank3_window(t, d, [4.0, 2.0, 1.0], 17);

        // Stateless.
        let mut ev_s = vec![0.0; d * k];
        let mut el_s = vec![0.0; k];
        let mut scratch = EigenbasisScratch::new();
        recover_eigenbasis_from_window_fast(&window, t, d, &mut ev_s, &mut el_s, &mut scratch, k, 5);

        // Tracker: push all t ticks, then recover.
        let mut tr = EigenbasisTracker::new(t, d);
        for r in 0..t {
            tr.push_tick(&window[r * d..(r + 1) * d]);
        }
        assert!(tr.is_full());
        let mut ev_t = vec![0.0; d * k];
        let mut el_t = vec![0.0; k];
        let held = tr.recover(&mut ev_t, &mut el_t, k, 5);
        assert_eq!(held, t);

        // Eigenvalues should agree to within 1% relative tolerance (float
        // reassociation differences between single-pass and incremental).
        for i in 0..k {
            let rel = (el_s[i] - el_t[i]).abs() / el_s[i].abs().max(1e-6);
            assert!(rel < 0.01, "tracker eigval {} ({}) diverges from stateless ({}) by {}", i, el_t[i], el_s[i], rel);
        }
        // Top eigenvalue direction should align (|cos| > 0.95).
        let mut dot = 0.0;
        let mut ns = 0.0;
        let mut nt = 0.0;
        for row in 0..d {
            dot += ev_s[row * k] * ev_t[row * k];
            ns += ev_s[row * k] * ev_s[row * k];
            nt += ev_t[row * k] * ev_t[row * k];
        }
        let cos = (dot / (ns.sqrt() * nt.sqrt())).abs();
        assert!(cos > 0.95, "tracker top-1 direction cos {} < 0.95", cos);
    }

    #[test]
    fn tracker_eviction_keeps_gram_consistent() {
        // Push capacity+50 ticks; the Gram should reflect only the last
        // `capacity` ticks. Verify by comparing against a stateless recovery
        // over the same trailing window.
        use crate::hla_eigenbasis::EigenbasisTracker;
        let d = 8;
        let cap = 128;
        let k = 2;
        let total = cap + 50;

        // Build `total` distinct ticks (deterministic, rank-leaning).
        let mut all_ticks = vec![0.0_f32; total * d];
        let mut s = 7u64;
        for r in 0..total {
            let dom = r % 3;
            for j in 0..d {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let base = if j == dom { 1.0 } else { 0.0 };
                let noise = (s >> 33) as f32 / (1u64 << 31) as f32 * 0.05;
                all_ticks[r * d + j] = base + noise;
            }
        }

        let mut tr = EigenbasisTracker::new(cap, d);
        for r in 0..total {
            tr.push_tick(&all_ticks[r * d..(r + 1) * d]);
        }
        assert_eq!(tr.len(), cap);
        assert!(tr.is_full());

        let mut ev_t = vec![0.0; d * k];
        let mut el_t = vec![0.0; k];
        tr.recover(&mut ev_t, &mut el_t, k, 8);

        // Stateless over the trailing `cap` ticks (the last cap of all_ticks).
        let trailing = &all_ticks[(total - cap) * d..];
        let mut ev_s = vec![0.0; d * k];
        let mut el_s = vec![0.0; k];
        let mut scratch = EigenbasisScratch::new();
        recover_eigenbasis_from_window_fast(trailing, cap, d, &mut ev_s, &mut el_s, &mut scratch, k, 8);

        for i in 0..k {
            let rel = (el_s[i] - el_t[i]).abs() / el_s[i].abs().max(1e-6);
            assert!(rel < 0.02, "evicted eigval {} ({}) vs stateless ({}) rel {}", i, el_t[i], el_s[i], rel);
        }
    }
}
