//! Fitted token-value tables — the modelless primitive halves of Issue 883
//! P1–P3 (Research 587, the V-side twin of Issue 882's anchor scoring).
//!
//! One frozen table product, three consumers, all closed-form on a frozen
//! checkpoint (no gradient, no training — the only runtime mutation is a
//! table swap, the Plan 408 deterministically-constructed-values class):
//!
//! | Task | Primitive | Law |
//! |---|---|---|
//! | **P1** token-mean-removed V quant | [`MeanRemovedValueCache`] (decorates ANY [`QuantizedKVCache`]) · [`remove_token_mean_into`] / [`add_token_mean_inplace`] / [`axpy_mean_restored`] | store `q(V − E^V_l[s])`, read `dequant(q) + E^V_l[s]` |
//! | **P2** fitted K=V+ retrofit | [`v_from_k_plus`] / [`add_scaled_row_inplace`] | `V = K + λ·E_l[s]`, `E_l[s] = mean(V − K ∣ s)` |
//! | **P3** V-cache halving | [`reconstruct_v_from_rope_k`] / [`read_v`] (feature `fitted_v_reconstruct`) | `V_t = G(−θ·p)·K̂_t + λ·E_l[s_t]` |
//!
//! # The table
//!
//! [`FittedTokenTable`] is the frozen form of the P0 calibration substrate
//! ([`LayeredVkCalibration`], `fitted_anchor_table.rs`): one `width`-wide
//! f32 row per (tapped layer, tracked token), James–Stein-shrunk at freeze
//! (`n/(n+λ_js)·mean`, λ_js = 0 ⇒ the plain mean). Untracked tokens have NO
//! row — [`FittedTokenTable::row`] returns `None` and every consumer falls
//! back to its plain path (P1: plain quant; P2: `V := K`; P3: `V := G(−θp)K̂`).
//! That is the `kv_sink_window`-style fallback seam for table-missed tail
//! tokens: a miss is never an error and never a zero-row guess.
//!
//! **Storage dial** (Research 587 §2.1 / Issue 883 P4):
//! `P(K) = b_w · L · K · d_v` bytes for K tracked rows — the paper's
//! full-vocab table is the `K = N` special case. [`FittedTokenTable::bytes`]
//! reports the live figure at `b_w = 4` (f32 rows).
//!
//! # P1 — what is proven and what is measured
//!
//! Law of total variance: `Var(V) = Var(E[V|s]) + E[Var(V|s)]`, so removing
//! the per-token mean can only shrink the residual VARIANCE (equality iff
//! token identity carries nothing). The quantizer-level claim is MEASURED,
//! not proven: absmax/min-max integer quantizers scale by RANGE, and an
//! off-mean occurrence can grow `|V − E^V[s]|` (Bench 895 G1c records it).
//!
//! # G3 kill switches (bit-identity classes)
//!
//! - P1 with a missing row, or a table whose rows are all `+0.0`, stores and
//!   reads bit-identically to the undecorated cache (`v − 0 = v`; the read
//!   add is skipped on a miss).
//! - P2 at `λ = 0` (or a missing row) is a COPY of `K` — the `V := K` path —
//!   bit-identically; the multiply is never executed (so `−0.0` and a
//!   non-finite row cannot leak through `0·x`).
//! - P3 [`VReadPath::FullCache`] copies the stored V bit-identically.
//!
//! All hot-path functions are zero-allocation over caller buffers (G4).

use crate::fitted_anchor_table::LayeredVkCalibration;
use crate::types::QuantizedKVCache;

/// Which fitted signal of a [`LayeredVkCalibration`] to freeze.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkSignal {
    /// `E^V_l[s] = mean(V | s)` — the value-mean table (P1).
    ValueMean,
    /// `E_l[s] = mean(V − K | s)` — the residual table (P2/P3).
    Residual,
}

/// Frozen per-(layer, token) table — one f32 row of `width` per tracked
/// token per tapped layer. Built once (allocating); every read is a slice.
#[derive(Debug, Clone)]
pub struct FittedTokenTable {
    n_layer: usize,
    width: usize,
    rows: usize,
    /// `n_layer × rows × width`, row-major by (layer, row).
    data: Vec<f32>,
    /// token id → tracked row, or `u32::MAX` (untracked → `None`).
    row_of_token: Vec<u32>,
}

impl FittedTokenTable {
    /// Build from explicit rows. `data.len()` must be
    /// `n_layer × rows × width` where `rows` = the number of distinct tracked
    /// rows referenced by `row_of_token` (`max + 1`, 0 when none).
    ///
    /// # Panics
    /// On a size mismatch or a non-finite row entry (a corrupt artifact must
    /// be loud at load, never a silent NaN at read).
    #[must_use]
    pub fn from_rows(n_layer: usize, width: usize, row_of_token: Vec<u32>, data: Vec<f32>) -> Self {
        let rows = row_of_token
            .iter()
            .filter(|&&r| r != u32::MAX)
            .map(|&r| r as usize + 1)
            .max()
            .unwrap_or(0);
        assert_eq!(
            data.len(),
            n_layer * rows * width,
            "fitted_value_table: data len != n_layer × rows × width"
        );
        assert!(
            data.iter().all(|x| x.is_finite()),
            "fitted_value_table: non-finite table entry"
        );
        Self {
            n_layer,
            width,
            rows,
            data,
            row_of_token,
        }
    }

    /// Freeze one signal of a P0 calibration pass, James–Stein-shrunk with
    /// `lambda_js` (`0` ⇒ the plain per-token mean, bit-identical to
    /// `StreamingMeanTable::mean_into`). Rows with no observations freeze
    /// to zeros (an empty key has no direction).
    #[must_use]
    pub fn from_calibration(cal: &LayeredVkCalibration, signal: VkSignal, lambda_js: f32) -> Self {
        let n_layer = cal.layers.len();
        let rows = cal.top_k;
        let width = cal.layers.first().map_or(0, |l| l.v.width());
        let mut data = vec![0.0f32; n_layer * rows * width];
        for (l, tables) in cal.layers.iter().enumerate() {
            let t = match signal {
                VkSignal::ValueMean => &tables.v,
                VkSignal::Residual => &tables.vk,
            };
            for r in 0..rows {
                let base = (l * rows + r) * width;
                t.shrunk_into(r, lambda_js, &mut data[base..base + width]);
            }
        }
        let mut row_of_token = cal.row_of_token.clone();
        for r in &mut row_of_token {
            if *r != u32::MAX && *r as usize >= rows {
                *r = u32::MAX;
            }
        }
        Self::from_rows(n_layer, width, row_of_token, data)
    }

    /// A same-shape table with every row `+0.0` — the P1/P2 G3 fixture.
    #[must_use]
    pub fn zeros_like(&self) -> Self {
        Self {
            data: vec![0.0; self.data.len()],
            ..self.clone()
        }
    }

    /// The row for `(layer, token)`, or `None` when the token is untracked
    /// (or out of vocab) — the consumer's plain-path fallback.
    #[inline]
    #[must_use]
    pub fn row(&self, layer: usize, token: u32) -> Option<&[f32]> {
        debug_assert!(layer < self.n_layer, "layer {layer} ≥ n_layer");
        let r = *self.row_of_token.get(token as usize)?;
        if r == u32::MAX {
            return None;
        }
        let base = (layer * self.rows + r as usize) * self.width;
        Some(&self.data[base..base + self.width])
    }

    /// Tapped layers.
    #[must_use]
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    /// Row width (`d_v`).
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Tracked rows per layer (the top-K residency `K`).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Live table bytes — the storage dial `P(K) = b_w·L·K·d_v` at `b_w = 4`.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// Storage-dial law `P(K) = b_w · L · K · d_v` (bytes) — Issue 883 P4.
#[inline]
#[must_use]
pub const fn storage_bytes(b_w: usize, n_layer: usize, k_rows: usize, d_v: usize) -> usize {
    b_w * n_layer * k_rows * d_v
}

/// FLOP law `ΔF_V ≈ L · S · d_v · (2d − 1)` — the arithmetic the deleted
/// `W_V` GEMV costs (d multiplies + d−1 adds per output element) for `S`
/// tokens through `L` layers (Issue 883 P4, paper §3.3).
#[inline]
#[must_use]
pub const fn v_projection_flops(n_layer: u64, tokens: u64, d_v: u64, d_model: u64) -> u64 {
    n_layer * tokens * d_v * (2 * d_model - 1)
}

/// The droppable-cache fraction `n_v/(n_kv + n_v)` — exactly `1/2` whenever
/// K and V heads are equal in count and width (every grouped-KV arch; GQA
/// changes `n_kv` for BOTH, so the ratio stays 1/2). Takes per-token K and
/// V widths (`n_kv_heads·head_dim_k`, `n_kv_heads·head_dim_v`).
#[inline]
#[must_use]
pub fn v_cache_fraction(k_width: usize, v_width: usize) -> f64 {
    v_width as f64 / (k_width + v_width) as f64
}

// ── P1 — token-mean-removed V quantization ────────────────────────────────

/// `out = v − row` (encode side); `None` ⇒ `out = v` (plain path).
/// Zero-alloc.
#[inline]
pub fn remove_token_mean_into(v: &[f32], row: Option<&[f32]>, out: &mut [f32]) {
    debug_assert_eq!(v.len(), out.len());
    match row {
        None => out.copy_from_slice(v),
        Some(e) => {
            debug_assert_eq!(e.len(), v.len());
            for ((o, &x), &m) in out.iter_mut().zip(v).zip(e) {
                *o = x - m;
            }
        }
    }
}

/// `out += row` (decode side, after the backend's dequant); `None` ⇒ no-op
/// (plain path — the add is never executed, so a miss is bit-identical).
#[inline]
pub fn add_token_mean_inplace(out: &mut [f32], row: Option<&[f32]>) {
    if let Some(e) = row {
        debug_assert_eq!(e.len(), out.len());
        for (o, &m) in out.iter_mut().zip(e) {
            *o += m;
        }
    }
}

/// Fused attention-V epilogue: `acc += w · (v̂ + row)` in ONE pass over the
/// dequantized residual `v̂` (the aggregation step of decode attention);
/// `None` ⇒ `acc += w · v̂` (the plain axpy). Zero-alloc.
#[inline]
pub fn axpy_mean_restored(acc: &mut [f32], w: f32, v_hat: &[f32], row: Option<&[f32]>) {
    debug_assert_eq!(acc.len(), v_hat.len());
    match row {
        None => {
            for (a, &x) in acc.iter_mut().zip(v_hat) {
                *a += w * x;
            }
        }
        Some(e) => {
            debug_assert_eq!(e.len(), v_hat.len());
            for ((a, &x), &m) in acc.iter_mut().zip(v_hat).zip(e) {
                *a += w * (x + m);
            }
        }
    }
}

/// A [`QuantizedKVCache`] decorator that stores `quant(V − E^V_l[s])` in the
/// wrapped backend and reads `dequant + E^V_l[s]` — the P1 product over ANY
/// existing KV quantizer (KVarN, TurboQuant, …); keys pass through.
///
/// The token at each position is supplied by [`set_token`](Self::set_token)
/// before `store_value` (the trait carries no token id). Untracked tokens
/// (or positions whose token was never set) take the plain path.
pub struct MeanRemovedValueCache<'t, C: QuantizedKVCache> {
    inner: C,
    table: &'t FittedTokenTable,
    /// token id per position (`u32::MAX` = unset → plain path).
    tokens: Vec<u32>,
    scratch: Vec<f32>,
}

impl<'t, C: QuantizedKVCache> MeanRemovedValueCache<'t, C> {
    /// Wrap `inner` for sequences up to `max_seq_len` positions. Allocates
    /// the per-position token map + one `width` scratch row, once.
    #[must_use]
    pub fn new(inner: C, table: &'t FittedTokenTable, max_seq_len: usize) -> Self {
        Self {
            inner,
            table,
            tokens: vec![u32::MAX; max_seq_len],
            scratch: vec![0.0; table.width()],
        }
    }

    /// Record the token id at `pos` (must precede that position's
    /// `store_value`; the same id serves every layer).
    #[inline]
    pub fn set_token(&mut self, pos: usize, token: u32) {
        self.tokens[pos] = token;
    }

    /// The wrapped backend.
    #[must_use]
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// The wrapped backend, mutably (its dequant takes `&mut self`).
    pub fn inner_mut(&mut self) -> &mut C {
        &mut self.inner
    }

    /// Unwrap the backend.
    #[must_use]
    pub fn into_inner(self) -> C {
        self.inner
    }

    /// The fused decode-attention V step: `acc += w · (dequant(q_pos) +
    /// E^V_l[s_pos])` — the backend dequantizes the stored residual into
    /// the owned scratch, then ONE [`axpy_mean_restored`] pass restores the
    /// mean and accumulates (no separate add pass over `out`). Zero-alloc.
    pub fn accumulate_value(&mut self, layer: usize, pos: usize, w: f32, acc: &mut [f32]) {
        let row = self.row_at(layer, pos);
        self.inner
            .dequantize_value_into(layer, pos, &mut self.scratch);
        axpy_mean_restored(acc, w, &self.scratch, row);
    }

    #[inline]
    fn row_at(&self, layer: usize, pos: usize) -> Option<&'t [f32]> {
        let tok = *self.tokens.get(pos)?;
        if tok == u32::MAX {
            return None;
        }
        self.table.row(layer, tok)
    }
}

impl<C: QuantizedKVCache> QuantizedKVCache for MeanRemovedValueCache<'_, C> {
    #[inline]
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        self.inner.store_key(layer, pos, key);
    }

    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        match self.row_at(layer, pos) {
            None => self.inner.store_value(layer, pos, value),
            Some(e) => {
                remove_token_mean_into(value, Some(e), &mut self.scratch);
                self.inner.store_value(layer, pos, &self.scratch);
            }
        }
    }

    #[inline]
    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.inner.dequantize_key_into(layer, pos, out);
    }

    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.inner.dequantize_value_into(layer, pos, out);
        add_token_mean_inplace(out, self.row_at(layer, pos));
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.tokens.fill(u32::MAX);
    }

    #[inline]
    fn pos(&self) -> usize {
        self.inner.pos()
    }

    #[inline]
    fn set_pos(&mut self, pos: usize) {
        self.inner.set_pos(pos);
    }
}

// ── P2 — fitted K=V+ retrofit ─────────────────────────────────────────────

/// `out += λ · row` in place; `λ = 0` or `None` ⇒ untouched (the multiply
/// is never executed). The shared tail of P2 and P3.
#[inline]
pub fn add_scaled_row_inplace(out: &mut [f32], row: Option<&[f32]>, lambda: f32) {
    let Some(e) = row else { return };
    if lambda == 0.0 {
        return;
    }
    debug_assert_eq!(e.len(), out.len());
    if lambda == 1.0 {
        // `1.0·m` is exact, so this is the same value — just one op fewer.
        return add_token_mean_inplace(out, row);
    }
    for (o, &m) in out.iter_mut().zip(e) {
        *o += lambda * m;
    }
}

/// P2: serve `V = K + λ·E_l[s]` with `W_V` deleted. `λ = 0` or a missing
/// row is the `V := K` path, bit-identically (a copy). Zero-alloc.
#[inline]
pub fn v_from_k_plus(k: &[f32], row: Option<&[f32]>, lambda: f32, out: &mut [f32]) {
    debug_assert_eq!(k.len(), out.len());
    out.copy_from_slice(k);
    add_scaled_row_inplace(out, row, lambda);
}

// ── P3 — V-cache halving by reconstruction ────────────────────────────────

#[cfg(feature = "fitted_v_reconstruct")]
pub use reconstruct::{VReadPath, read_v, reconstruct_v_from_rope_k};

#[cfg(feature = "fitted_v_reconstruct")]
mod reconstruct {
    use super::add_scaled_row_inplace;
    use crate::position_group_action::PositionGroupAction;

    /// P3 read-path selector — the kill switch.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum VReadPath {
        /// Read the persistent V cache (bit-identical copy) — the control.
        FullCache,
        /// No V cache: `V = G(−θp)·K̂ + λ·E_l[s]` from the cached post-RoPE K.
        Reconstruct {
            /// The P2-measured table scale (λ = 0 ⇒ `V := G(−θp)K̂`).
            lambda: f32,
        },
    }

    /// P3: reconstruct `V_t = G(−θ·p)·K̂_t + λ·E_l[s_t]` from the cached
    /// post-RoPE key `k_cached` (`n_kv_heads × head_dim`, the rotation
    /// applied per head slice by `rope`, whose `dim()` is the head width).
    ///
    /// ONE one-directional inverse rotation per read — the cache write's
    /// forward rotation is the model's own, never re-applied here (Issue
    /// 883 trap 2: no round-trips). `rope` MUST be the exact convention the
    /// cache was written with (interleaved pairs for [`RopeAction`];
    /// a half-split/NeoX cache needs its own action — trap 1, the tap-point
    /// law). Zero-alloc.
    ///
    /// [`RopeAction`]: crate::position_group_action::RopeAction
    #[inline]
    pub fn reconstruct_v_from_rope_k<A: PositionGroupAction>(
        rope: &A,
        pos: f32,
        k_cached: &[f32],
        row: Option<&[f32]>,
        lambda: f32,
        out: &mut [f32],
    ) {
        let hd = rope.dim();
        debug_assert_eq!(k_cached.len(), out.len());
        debug_assert!(
            hd > 0 && k_cached.len().is_multiple_of(hd),
            "kv width not a head multiple"
        );
        for (kh, oh) in k_cached.chunks_exact(hd).zip(out.chunks_exact_mut(hd)) {
            rope.apply_inverse_at(pos, kh, oh);
        }
        add_scaled_row_inplace(out, row, lambda);
    }

    /// Read V through the selected path. `FullCache` requires `v_cached`
    /// (bit-identical copy); `Reconstruct` ignores it.
    ///
    /// # Panics
    /// `FullCache` with `v_cached = None` (a wiring bug — the kill switch
    /// selected a cache that was dropped).
    #[inline]
    pub fn read_v<A: PositionGroupAction>(
        path: VReadPath,
        rope: &A,
        pos: f32,
        k_cached: &[f32],
        v_cached: Option<&[f32]>,
        row: Option<&[f32]>,
        out: &mut [f32],
    ) {
        match path {
            VReadPath::FullCache => {
                out.copy_from_slice(v_cached.expect("VReadPath::FullCache needs the V cache"));
            }
            VReadPath::Reconstruct { lambda } => {
                reconstruct_v_from_rope_k(rope, pos, k_cached, row, lambda, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_2x3() -> FittedTokenTable {
        // 1 layer, width 3, tokens {0→row1, 2→row0}, token 1 untracked.
        FittedTokenTable::from_rows(
            1,
            3,
            vec![1, u32::MAX, 0],
            vec![1.0, 2.0, 3.0, -1.0, 0.5, 0.25],
        )
    }

    #[test]
    fn row_lookup_and_miss() {
        let t = table_2x3();
        assert_eq!(t.rows(), 2);
        assert_eq!(t.row(0, 0), Some(&[-1.0, 0.5, 0.25][..]));
        assert_eq!(t.row(0, 2), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(t.row(0, 1), None, "untracked → plain path");
        assert_eq!(t.row(0, 99), None, "out of vocab → plain path");
        assert_eq!(t.bytes(), storage_bytes(4, 1, 2, 3));
    }

    #[test]
    fn encode_decode_roundtrip_is_exact_for_representable_values() {
        let t = table_2x3();
        let v = [3.0f32, 4.0, 5.0];
        let mut q = [0.0f32; 3];
        remove_token_mean_into(&v, t.row(0, 2), &mut q);
        assert_eq!(q, [2.0, 2.0, 2.0]);
        add_token_mean_inplace(&mut q, t.row(0, 2));
        assert_eq!(q, v);
        // miss ⇒ copy + no-op
        remove_token_mean_into(&v, None, &mut q);
        assert_eq!(q, v);
    }

    #[test]
    fn fused_axpy_matches_two_pass() {
        let t = table_2x3();
        let vh = [0.5f32, -0.5, 1.0];
        let mut a = [1.0f32; 3];
        axpy_mean_restored(&mut a, 0.5, &vh, t.row(0, 0));
        let mut b = vh;
        add_token_mean_inplace(&mut b, t.row(0, 0));
        let expect: Vec<f32> = b.iter().map(|x| 1.0 + 0.5 * x).collect();
        assert_eq!(&a[..], &expect[..]);
    }

    #[test]
    fn v_from_k_plus_lambda_zero_is_a_copy_bitwise() {
        let t = FittedTokenTable::from_rows(1, 3, vec![0], vec![f32::MAX, 1.0, 2.0]);
        let k = [-0.0f32, 1.5, f32::MIN_POSITIVE];
        let mut out = [9.0f32; 3];
        v_from_k_plus(&k, t.row(0, 0), 0.0, &mut out);
        for (a, b) in out.iter().zip(&k) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        v_from_k_plus(&k, None, 1.0, &mut out);
        for (a, b) in out.iter().zip(&k) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        let t2 = table_2x3();
        v_from_k_plus(&[1.0, 1.0, 1.0], t2.row(0, 2), 0.5, &mut out);
        assert_eq!(out, [1.5, 2.0, 2.5]);
    }

    #[test]
    fn freeze_from_calibration_matches_streaming_means() {
        let mut cal = LayeredVkCalibration::from_counts(2, 2, vec![3, 1, 0], 2);
        cal.observe_layer(0, 0, &[1.0, 1.0], &[3.0, 5.0]);
        cal.observe_layer(0, 0, &[1.0, 1.0], &[1.0, 3.0]);
        cal.observe_layer(1, 1, &[0.0, 2.0], &[4.0, 2.0]);
        let tv = FittedTokenTable::from_calibration(&cal, VkSignal::ValueMean, 0.0);
        let tr = FittedTokenTable::from_calibration(&cal, VkSignal::Residual, 0.0);
        assert_eq!(tv.row(0, 0), Some(&[2.0, 4.0][..]));
        assert_eq!(tr.row(0, 0), Some(&[1.0, 3.0][..]));
        assert_eq!(tr.row(1, 1), Some(&[4.0, 0.0][..]));
        assert_eq!(
            tv.row(1, 0),
            Some(&[0.0, 0.0][..]),
            "empty key freezes to zeros"
        );
        assert_eq!(tv.row(0, 2), None, "never-seen token is untracked");
        // James–Stein: n=2, λ=2 ⇒ ½·mean.
        let ts = FittedTokenTable::from_calibration(&cal, VkSignal::Residual, 2.0);
        assert_eq!(ts.row(0, 0), Some(&[0.5, 1.5][..]));
    }

    #[test]
    fn laws() {
        // gemma-2-2b: L=26, d=2304, d_v = 4 kv heads × 256 = 1024.
        assert_eq!(
            v_projection_flops(26, 128, 1024, 2304),
            26 * 128 * 1024 * 4607
        );
        assert!((v_cache_fraction(1024, 1024) - 0.5).abs() < 1e-15);
        // MLA-style asymmetric widths: K carries the rope dims too.
        assert!((v_cache_fraction(192, 128) - 0.4).abs() < 1e-15);
        assert_eq!(storage_bytes(4, 26, 8192, 1024), 4 * 26 * 8192 * 1024);
    }

    #[test]
    #[should_panic(expected = "non-finite table entry")]
    fn nonfinite_table_refused() {
        let _ = FittedTokenTable::from_rows(1, 1, vec![0], vec![f32::NAN]);
    }

    #[cfg(feature = "fitted_v_reconstruct")]
    #[test]
    fn reconstruct_inverts_the_cache_rotation() {
        use crate::position_group_action::{PositionGroupAction, RopeAction};
        let rope = RopeAction::new(4);
        let k_pre = [0.3f32, -1.2, 0.7, 2.0, -0.4, 0.1, 1.1, -0.9];
        let mut k_cached = [0.0f32; 8];
        let (c0, c1) = k_cached.split_at_mut(4);
        rope.apply_at(1234.0, &k_pre[..4], c0);
        rope.apply_at(1234.0, &k_pre[4..], c1);
        let row = [0.5f32; 8];
        let mut out = [0.0f32; 8];
        reconstruct_v_from_rope_k(&rope, 1234.0, &k_cached, Some(&row), 1.0, &mut out);
        for (o, k) in out.iter().zip(&k_pre) {
            assert!((o - (k + 0.5)).abs() < 1e-5, "{o} vs {}", k + 0.5);
        }
        let v = [7.0f32; 8];
        read_v(
            VReadPath::FullCache,
            &rope,
            1234.0,
            &k_cached,
            Some(&v),
            Some(&row),
            &mut out,
        );
        assert_eq!(out, v);
    }
}
