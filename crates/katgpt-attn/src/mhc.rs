//! Manifold-Constrained Hyper-Connections (mHC) — Xie et al., 2026.
//!
//! The N-wide residual-stream mixer behind GLM-5.3-Flash's
//! `Glm5NextTextHyperConnection` (which inherits its forward unchanged from
//! transformers `DeepseekV4HyperConnection`, `modeling_deepseek_v4.py` — the
//! formula source of record; fetched 2026-11-28, riir-infer plan 627 T4b).
//!
//! # The mechanism (per token, all fp32)
//!
//! The residual stream is `N` parallel copies (`hc_mult`, GLM-5.3: N = 4),
//! shape `[N, D]`. Each attention/MLP block site runs ONE hyper-connection:
//!
//! ```text
//! 1. FLATTEN + NORM   flat = UnweightedRMSNorm(streams.flatten())  [N·D]
//!                        x · rsqrt(mean(x²) + rms_eps)  — no learnable weight
//! 2. PROJECT          logits = fn · flat                [(2+N)·N]
//!                        fn is [(2+N)·N, N·D]; base is [(2+N)·N]; scale is [3]
//! 3. SPLIT            pre_w [N] | post_w [N] | comb_w [N·N] (row-major N×N)
//!                        (base splits identically; scale = [pre, post, comb])
//! 4. WEIGHTS          pre  = sigmoid(pre_w·s0 + pre_b) + eps
//!                     post = 2·sigmoid(post_w·s1 + post_b)      (NO +eps)
//!                     comb = softmax_row(comb_w·s2 + comb_b) + eps
//! 5. SINKHORN         comb /= (col_sum + eps)          — ONCE, before the loop
//!                     repeat (iters−1) times:
//!                        comb /= (row_sum + eps)
//!                        comb /= (col_sum + eps)
//!                     → comb is (approximately) doubly stochastic, which keeps
//!                       the residual mixing non-expansive over deep stacks
//! 6. COLLAPSE         collapsed[d] = Σ_k pre[k] · streams[k][d]    [D]
//!                        (the block input)
//! ```
//!
//! The block (attention or MLP) runs on `collapsed`, and the caller mixes the
//! result back with [`mhc_combine`]:
//!
//! ```text
//! new_streams[k][d] = post[k] · block_out[d] + Σ_j comb[j][k] · streams[j][d]
//! ```
//!
//! `comb` is consumed TRANSPOSED — `sum_j comb[j,k] · residual[j,d]`, summed
//! over the FIRST hc axis. Sinkhorn produces a doubly-stochastic but
//! non-symmetric matrix, so the direction matters (the transformers decoder
//! comments this explicitly). GLM's stream head is an unweighted mean over
//! the N streams (`Glm5NextTextHyperHead`); DeepSeek-V4's own head is a
//! learned sigmoid mix — the head is the caller's concern, not this module's.
//!
//! # Shapes (GLM-5.3-Flash, N = 4, D = 4096)
//!
//! `fn` is **[24, 16384]**, `base` **[24]**, `scale` **[3]** — the shape
//! formula `(2 + hc_mult) * hc_mult` rows × `hc_mult * hidden_size` cols is
//! printed in the transformers source. (Plan 627's earlier "[12×4096]" bullet
//! was a mis-derivation, corrected in the plan at claim time.) Checkpoint
//! names: `layers.{i}.hc_attn_{fn,base,scale}` + `hc_ffn_{fn,base,scale}`,
//! main layers only (no MTP copy). The transformers loader keeps
//! `attn_hc.base` / `attn_hc.scale` / `ffn_hc.base` / `ffn_hc.scale` in fp32
//! (`_keep_in_fp32_modules_strict`) and the forward upcasts `fn` to fp32 at
//! compute time — this module is f32 throughout, matching both.
//!
//! # Determinism classification (Issue-940 carve-out)
//!
//! Every reduction here is LOCAL per-token compute — nothing crosses a
//! sync/quorum boundary — so the crate's canonical SIMD helpers
//! (`simd_matmul_rows`, `simd_sum_sq`, `fast_sigmoid`) are consumed directly,
//! same as the pinned gdn2/kda lanes. Same binary + same input ⇒
//! bit-identical output (gate-tested below). The comb softmax subtracts the
//! row max (torch `F.softmax` semantics) and folds sequentially ascending;
//! the sinkhorn divisors all carry `+eps`, mirroring the source exactly.

use katgpt_core::simd::{fast_sigmoid, simd_matmul_rows, simd_sum_sq};

// ─── Config ─────────────────────────────────────────────────────────────────

/// mHC configuration — the `hc_*` config keys plus the shared RMS eps.
#[derive(Clone, Debug)]
pub struct MhcConfig {
    /// Number of residual streams (`hc_mult`, "N" in the doc). GLM-5.3: **4**.
    pub n_streams: usize,
    /// Hidden dim per stream ("D"). GLM-5.3: **4096**.
    pub hidden_size: usize,
    /// Sinkhorn-Knopp normalization iterations (`hc_sinkhorn_iters`).
    /// GLM-5.3: **20** — the source runs ONE column normalization before the
    /// loop and then `iters − 1` row/column pairs (39 normalizations total).
    pub sinkhorn_iters: usize,
    /// Numerical floor `hc_eps`. GLM-5.3: **1e-6**. Added after the softmax,
    /// and inside EVERY sinkhorn divisor.
    pub eps: f32,
    /// RMS epsilon of the input norm (the config's `rms_norm_eps` — the
    /// source's `DeepseekV4UnweightedRMSNorm` variance epsilon).
    pub rms_eps: f32,
}

impl MhcConfig {
    /// Rows of the `fn` projection (and length of `base`): `(2 + N) · N`.
    #[inline]
    pub fn fn_rows(&self) -> usize {
        (2 + self.n_streams) * self.n_streams
    }

    /// Columns of the `fn` projection: `N · D` (the flattened stream width).
    #[inline]
    pub fn fn_cols(&self) -> usize {
        self.n_streams * self.hidden_size
    }

    /// GLM-5.3-Flash values (config defaults, confirmed by the T0 probe):
    /// `hc_mult` 4, `hc_eps` 1e-6, `hc_sinkhorn_iters` 20, `rms_norm_eps`
    /// 1e-5. Callers with a parsed config should construct directly — this
    /// constructor exists for tests and probes.
    pub fn glm5_flash(hidden_size: usize, rms_eps: f32) -> Self {
        Self {
            n_streams: 4,
            hidden_size,
            sinkhorn_iters: 20,
            eps: 1e-6,
            rms_eps,
        }
    }
}

// ─── Weights ──────────────────────────────────────────────────────────────────────────

/// Borrow view of one hyper-connection site's weights — the HOT-PATH form.
/// The projection matrix alone is [(2+N)·N, N·D] ≈ 1.5 MiB at the GLM-5.3
/// shape, so the forward consumes a view, never an owned clone.
#[derive(Clone, Debug)]
pub struct MhcWeightsRef<'a> {
    /// The projection matrix `fn`, row-major `[(2+N)·N, N·D]` (checkpoint
    /// key `*.fn`; `fn` is a reserved word, hence `fn_w`).
    pub fn_w: &'a [f32],
    /// The additive base `[(2+N)·N]`, split `[N | N | N·N]` exactly like the
    /// logits.
    pub base: &'a [f32],
    /// The three output scales `[pre, post, comb]` (checkpoint key `*.scale`).
    pub scale: [f32; 3],
}

impl MhcWeightsRef<'_> {
    /// Length/type validation against the config — refuses loud naming the
    /// first mismatch. A wrong shape must never silently misindex.
    pub fn validate(&self, cfg: &MhcConfig) -> Result<(), String> {
        let rows = cfg.fn_rows();
        let cols = cfg.fn_cols();
        if self.fn_w.len() != rows * cols {
            return Err(format!(
                "fn weight length {}: expected rows*cols = {rows}*{cols} = {}",
                self.fn_w.len(),
                rows * cols
            ));
        }
        if self.base.len() != rows {
            return Err(format!(
                "base length {}: expected (2+N)*N = {rows}",
                self.base.len()
            ));
        }
        Ok(())
    }
}

/// Owned storage form of one site's weights (what a loader builds). The hot
/// path borrows it via [`MhcWeights::view`].
#[derive(Clone, Debug)]
pub struct MhcWeights {
    /// The projection matrix `fn`, row-major `[(2+N)·N, N·D]`.
    pub fn_w: Vec<f32>,
    /// The additive base `[(2+N)·N]`.
    pub base: Vec<f32>,
    /// The three output scales `[pre, post, comb]`.
    pub scale: [f32; 3],
}

impl MhcWeights {
    /// Borrow view for the hot path — zero copy.
    pub fn view(&self) -> MhcWeightsRef<'_> {
        MhcWeightsRef {
            fn_w: &self.fn_w,
            base: &self.base,
            scale: self.scale,
        }
    }

    /// Length/type validation (delegates to the view's gate).
    pub fn validate(&self, cfg: &MhcConfig) -> Result<(), String> {
        self.view().validate(cfg)
    }
}

// ─── Scratch ────────────────────────────────────────────────────────────────

/// Preallocated per-site buffers — the decision path is allocation-free.
///
/// One scratch per concurrent call site; the seat holds two (attn + ffn).
pub struct MhcScratch {
    flat: Vec<f32>,
    logits: Vec<f32>,
    comb: Vec<f32>,
    pre: Vec<f32>,
    post: Vec<f32>,
    collapsed: Vec<f32>,
}

impl MhcScratch {
    /// Allocate the buffers for `cfg` once (`with_capacity` semantics — no
    /// growth on the hot path).
    pub fn new(cfg: &MhcConfig) -> Self {
        Self {
            flat: vec![0.0; cfg.fn_cols()],
            logits: vec![0.0; cfg.fn_rows()],
            comb: vec![0.0; cfg.n_streams * cfg.n_streams],
            pre: vec![0.0; cfg.n_streams],
            post: vec![0.0; cfg.n_streams],
            collapsed: vec![0.0; cfg.hidden_size],
        }
    }

    /// The `post` gate from the last [`mhc_project`] call.
    #[inline]
    pub fn post(&self) -> &[f32] {
        &self.post
    }

    /// The sinkhorn-normalized `comb` from the last [`mhc_project`] call,
    /// row-major `[N, N]` (`comb[i * n + j]`).
    #[inline]
    pub fn comb(&self) -> &[f32] {
        &self.comb
    }

    /// The collapsed block input from the last [`mhc_project`] call.
    #[inline]
    pub fn collapsed(&self) -> &[f32] {
        &self.collapsed
    }
}

// ─── Forward ────────────────────────────────────────────────────────────────

/// The hyper-connection forward (steps 1–6 in the module doc): reads
/// `streams` `[N·D]` (row-major `[N, D]`, stream-major), writes `post` `[N]`,
/// `comb` `[N·N]` and `collapsed` `[D]` into the scratch, and returns
/// `(post, comb, collapsed)` as slices of it. Allocates nothing.
///
/// `collapsed` is the block input; feed the block output and these `post` /
/// `comb` (plus the ORIGINAL `streams` as the residual) to [`mhc_combine`].
pub fn mhc_project<'a>(
    cfg: &MhcConfig,
    w: &MhcWeightsRef<'_>,
    streams: &[f32],
    sc: &'a mut MhcScratch,
) -> (&'a [f32], &'a [f32], &'a [f32]) {
    let n = cfg.n_streams;
    let d = cfg.hidden_size;
    let flat_len = n * d;
    debug_assert_eq!(streams.len(), flat_len, "streams must be [N·D]");

    // 1. Flatten + unweighted RMS norm over the N·D width. torch: variance =
    //    x.float().square().mean(-1) — sum first, ONE division by the width.
    sc.flat.copy_from_slice(streams);
    let sum_sq = simd_sum_sq(&sc.flat, flat_len);
    let variance = sum_sq / flat_len as f32;
    let inv_rms = 1.0 / (variance + cfg.rms_eps).sqrt();
    for v in sc.flat.iter_mut() {
        *v *= inv_rms;
    }

    // 2. Project: [(2+N)·N, N·D] · [N·D] → [(2+N)·N].
    simd_matmul_rows(&mut sc.logits, w.fn_w, &sc.flat, cfg.fn_rows(), flat_len);

    // 3–4. Split + the one-layer-perceptron weight forms. Layout: the first N
    //      logits are pre, the next N post, the last N·N comb (row-major).
    let (scale_pre, scale_post, scale_comb) = (w.scale[0], w.scale[1], w.scale[2]);
    for k in 0..n {
        let pre_w = sc.logits[k];
        let pre_b = w.base[k];
        sc.pre[k] = fast_sigmoid(pre_w * scale_pre + pre_b) + cfg.eps;
    }
    for k in 0..n {
        let post_w = sc.logits[n + k];
        let post_b = w.base[n + k];
        sc.post[k] = 2.0 * fast_sigmoid(post_w * scale_post + post_b);
    }

    // comb logits → row softmax (+eps) → sinkhorn. The softmax subtracts the
    // row max (torch F.softmax semantics — per-row, never across the whole
    // N·N matrix) and folds sequentially ascending.
    for i in 0..n {
        for j in 0..n {
            let idx = 2 * n + i * n + j;
            sc.comb[i * n + j] = sc.logits[idx] * scale_comb + w.base[idx];
        }
    }
    for i in 0..n {
        let row = i * n;
        let mut row_max = f32::NEG_INFINITY;
        for j in 0..n {
            let v = sc.comb[row + j];
            if v > row_max {
                row_max = v;
            }
        }
        let mut row_sum = 0.0f32;
        for j in 0..n {
            let e = (sc.comb[row + j] - row_max).exp();
            sc.comb[row + j] = e;
            row_sum += e;
        }
        for j in 0..n {
            sc.comb[row + j] = sc.comb[row + j] / row_sum + cfg.eps;
        }
    }

    // 5. Sinkhorn-Knopp: one column normalization first, then (iters − 1)
    //    row/column pairs. Every divisor carries +eps, mirroring the source.
    sinkhorn_normalize(cfg, &mut sc.comb);

    // 6. Collapse: collapsed[d] = Σ_k pre[k] · streams[k][d], k ascending.
    //    (The source collapses with PRE here — POST is the combine gate.)
    for dd in 0..d {
        let mut acc = 0.0f32;
        for k in 0..n {
            acc += sc.pre[k] * streams[k * d + dd];
        }
        sc.collapsed[dd] = acc;
    }

    (&sc.post, &sc.comb, &sc.collapsed)
}

/// One full sinkhorn pass: initial column normalize + (iters − 1) row/col
/// pairs, each divisor `(sum + eps)`. Exposed for tests; the forward calls it.
pub fn sinkhorn_normalize(cfg: &MhcConfig, comb: &mut [f32]) {
    // Initial column normalization (dim = -2 in torch: sum over the FIRST
    // axis, i.e. for each column j sum over rows i).
    divide_by_col_sums(cfg, comb);
    for _ in 1..cfg.sinkhorn_iters.max(1) {
        divide_by_row_sums(cfg, comb);
        divide_by_col_sums(cfg, comb);
    }
}

fn divide_by_row_sums(cfg: &MhcConfig, comb: &mut [f32]) {
    let n = cfg.n_streams;
    for i in 0..n {
        let row = i * n;
        let mut s = 0.0f32;
        for j in 0..n {
            s += comb[row + j];
        }
        let inv = 1.0 / (s + cfg.eps);
        for j in 0..n {
            comb[row + j] *= inv;
        }
    }
}

fn divide_by_col_sums(cfg: &MhcConfig, comb: &mut [f32]) {
    let n = cfg.n_streams;
    for j in 0..n {
        let mut s = 0.0f32;
        for i in 0..n {
            s += comb[i * n + j];
        }
        let inv = 1.0 / (s + cfg.eps);
        for i in 0..n {
            comb[i * n + j] *= inv;
        }
    }
}

// ─── Combine ────────────────────────────────────────────────────────────────

/// The decoder-layer combine: `new[k][d] = post[k]·block_out[d] + Σ_j
/// comb[j][k]·residual[j][d]` — comb consumed TRANSPOSED (sum over the FIRST
/// hc axis; the matrix is doubly-stochastic but non-symmetric, so the
/// direction is load-bearing). Writes `out` `[N·D]`; `out` must not alias
/// `residual` (debug-asserted). Allocates nothing.
///
/// The sum folds sequentially ascending j, with the `post·block_out` product
/// as the initial accumulator value — a fused association (the source
/// computes the two tensors separately then adds; the difference is f32
/// rounding order only, inside the 1e-4 capture-parity bar).
pub fn mhc_combine(
    cfg: &MhcConfig,
    post: &[f32],
    comb: &[f32],
    block_out: &[f32],
    residual: &[f32],
    out: &mut [f32],
) {
    let n = cfg.n_streams;
    let d = cfg.hidden_size;
    debug_assert_eq!(post.len(), n, "post must be [N]");
    debug_assert_eq!(comb.len(), n * n, "comb must be [N·N] row-major");
    debug_assert_eq!(block_out.len(), d, "block_out must be [D]");
    debug_assert_eq!(residual.len(), n * d, "residual must be [N·D]");
    debug_assert_eq!(out.len(), n * d, "out must be [N·D]");
    debug_assert!(
        !std::ptr::eq(out.as_ptr(), residual.as_ptr()),
        "out must not alias residual"
    );
    for k in 0..n {
        let gate = post[k];
        for dd in 0..d {
            let mut acc = gate * block_out[dd];
            for j in 0..n {
                acc += comb[j * n + k] * residual[j * d + dd];
            }
            out[k * d + dd] = acc;
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random f32 in [-1, 1] (fixed LCG — no global RNG,
    /// per the unseeded-global-draw rule).
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((self.0 >> 33) & 0xFFFF) as f32 / 65535.0;
            v * 2.0 - 1.0
        }
    }

    fn small_cfg() -> MhcConfig {
        MhcConfig {
            n_streams: 2,
            hidden_size: 3,
            // Enough iterations that the doubly-stochastic property actually
            // converges at this tiny N (the real config's 20 iters converges
            // on N=4; 3 iters leaves percent-scale row drift on skewed 2×2).
            sinkhorn_iters: 50,
            eps: 1e-6,
            rms_eps: 1e-5,
        }
    }

    fn rand_weights(cfg: &MhcConfig, seed: u64) -> MhcWeights {
        let mut lcg = Lcg(seed);
        MhcWeights {
            fn_w: (0..cfg.fn_rows() * cfg.fn_cols())
                .map(|_| lcg.next_f32())
                .collect(),
            base: (0..cfg.fn_rows()).map(|_| lcg.next_f32()).collect(),
            scale: [0.7, 1.3, 0.9],
        }
    }

    /// The comb matrix after sinkhorn is doubly stochastic: every row sum and
    /// every column sum lands at 1 within a tight band (the +eps floors and
    /// the finite iteration count leave ~eps-scale residue).
    #[test]
    fn comb_is_doubly_stochastic_after_sinkhorn() {
        let cfg = small_cfg();
        let w = rand_weights(&cfg, 11);
        w.validate(&cfg).unwrap();
        let mut sc = MhcScratch::new(&cfg);
        let streams: Vec<f32> = (0..cfg.fn_cols())
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let (_, comb, _) = mhc_project(&cfg, &w.view(), &streams, &mut sc);
        let n = cfg.n_streams;
        for i in 0..n {
            let s: f32 = comb[i * n..(i + 1) * n].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "row {i} sum {s} not ~1");
        }
        for j in 0..n {
            let s: f32 = (0..n).map(|i| comb[i * n + j]).sum();
            assert!((s - 1.0).abs() < 1e-4, "col {j} sum {s} not ~1");
        }
    }

    /// Sinkhorn iteration-count semantics: the source runs ONE column
    /// normalization before the loop and then (iters − 1) row/col pairs —
    /// `iters = 1` is the single column pass, not a no-op.
    #[test]
    fn sinkhorn_iters_one_is_a_single_column_pass() {
        let mut cfg = small_cfg();
        cfg.sinkhorn_iters = 1;
        let mut nonuniform = vec![1.0f32, 3.0, 5.0, 7.0];
        sinkhorn_normalize(&cfg, &mut nonuniform);
        // col sums: (1+5)=6 → col0 /6; (3+7)=10 → col1 /10 → [1/6, 3/10, 5/6, 7/10]
        assert!((nonuniform[0] - 1.0 / 6.0).abs() < 1e-6);
        assert!((nonuniform[1] - 3.0 / 10.0).abs() < 1e-6);
        assert!((nonuniform[2] - 5.0 / 6.0).abs() < 1e-6);
        assert!((nonuniform[3] - 7.0 / 10.0).abs() < 1e-6);
        // row sums are NOT 1 after only the column pass.
        let row0: f32 = nonuniform[0] + nonuniform[1];
        assert!(
            (row0 - 1.0).abs() > 1e-3,
            "iters=1 must leave rows unnormalized"
        );
    }

    /// The pre/post weight formulas on a hand-computed fixture: sigmoid form
    /// for pre (with +eps), doubled sigmoid for post (no eps).
    #[test]
    fn pre_and_post_weight_formulas_match_the_source() {
        let cfg = small_cfg();
        let mut w = rand_weights(&cfg, 23);
        // Craft the projection so the pre/post logits are EXACT known values:
        // with zero fn rows, logits are 0 and only base/scale matter.
        for v in w.fn_w.iter_mut() {
            *v = 0.0;
        }
        w.base = vec![0.0; cfg.fn_rows()];
        w.scale = [1.0, 1.0, 1.0];
        let streams = vec![0.5f32; cfg.fn_cols()];
        let mut sc = MhcScratch::new(&cfg);
        let (post, _, _) = mhc_project(&cfg, &w.view(), &streams, &mut sc);
        // pre = sigmoid(0*1 + 0) + eps = 0.5 + 1e-6 — consumed inside the
        // forward, so verify it through the collapsed output instead.
        // post = 2*sigmoid(0) = 1.0 exactly (no eps).
        for (k, &p) in post.iter().enumerate() {
            assert!((p - 1.0).abs() < 1e-7, "post[{k}] = {p}");
        }
        // Now a NON-zero base: pre_b = 0.5 → collapsed = (sigmoid(0.5)+eps)·Σstreams
        let sig = 1.0 / (1.0 + (-0.5f32).exp());
        for k in 0..cfg.n_streams {
            w.base[k] = 0.5; // pre region
            w.base[cfg.n_streams + k] = -2.0; // post region → 2·sigmoid(−2)
        }
        let (post, _, collapsed) = mhc_project(&cfg, &w.view(), &streams, &mut sc);
        let expected_post = 2.0 / (1.0 + 2.0f32.exp());
        for (k, &p) in post.iter().enumerate() {
            assert!((p - expected_post).abs() < 1e-6, "post[{k}] = {p}");
        }
        // collapsed[d] = Σ_k pre[k] · 0.5 = 2 · pre · 0.5 = pre
        let expected_collapsed = sig + cfg.eps;
        for (dd, &c) in collapsed.iter().enumerate() {
            assert!(
                (c - expected_collapsed).abs() < 1e-6,
                "collapsed[{dd}] = {c} vs {expected_collapsed}"
            );
        }
    }

    /// The combine against an independent in-test transcription, including
    /// the TRANSPOSED comb consumption direction (sum over the first axis).
    #[test]
    fn combine_matches_naive_transcription_including_transpose() {
        let cfg = small_cfg();
        let n = cfg.n_streams;
        let d = cfg.hidden_size;
        let post = [1.5f32, 0.25];
        let comb = [0.1, 0.9, 0.6, 0.4]; // row-major, NOT symmetric
        let block_out = [1.0f32, -2.0, 0.5];
        let residual = [1.0f32, 2.0, 3.0, -1.0, 5.0, 4.0];
        let mut out = vec![0.0f32; n * d];
        mhc_combine(&cfg, &post, &comb, &block_out, &residual, &mut out);
        // Naive: new[k][d] = post[k]*out[d] + Σ_j comb[j][k]*res[j][d]
        for k in 0..n {
            for dd in 0..d {
                let mut acc = post[k] * block_out[dd];
                for j in 0..n {
                    acc += comb[j * n + k] * residual[j * d + dd];
                }
                assert!(
                    (out[k * d + dd] - acc).abs() < 1e-6,
                    "out[{k}][{dd}] = {} vs {acc}",
                    out[k * d + dd]
                );
            }
        }
        // Direction check: transposing the comb input must move values.
        let comb_t = [0.1, 0.6, 0.9, 0.4];
        let mut out_t = vec![0.0f32; n * d];
        mhc_combine(&cfg, &post, &comb_t, &block_out, &residual, &mut out_t);
        assert_ne!(out[1], out_t[1], "comb direction is load-bearing");
    }

    /// Combine with an exact identity comb reduces to `post ⊗ block_out +
    /// residual` (the classical residual-add shape when post = 1).
    #[test]
    fn combine_identity_comb_is_residual_plus_gate() {
        let cfg = small_cfg();
        let n = cfg.n_streams;
        let d = cfg.hidden_size;
        let ones = [1.0f32; 2];
        let identity = [1.0f32, 0.0, 0.0, 1.0];
        let block_out = [1.0f32, 2.0, 3.0];
        let residual = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut out = vec![0.0f32; n * d];
        mhc_combine(&cfg, &ones, &identity, &block_out, &residual, &mut out);
        for k in 0..n {
            for dd in 0..d {
                let expected = block_out[dd] + residual[k * d + dd];
                assert!((out[k * d + dd] - expected).abs() < 1e-6);
            }
        }
    }

    /// Same binary + same input ⇒ bit-identical outputs (two fresh runs).
    #[test]
    fn determinism_bit_identical_reruns() {
        let cfg = small_cfg();
        let w = rand_weights(&cfg, 47);
        let streams: Vec<f32> = (0..cfg.fn_cols())
            .map(|i| (i as f32 * 0.037).cos())
            .collect();
        let run = || {
            let mut sc = MhcScratch::new(&cfg);
            let (post, comb, collapsed) = mhc_project(&cfg, &w.view(), &streams, &mut sc);
            let mut out = vec![0.0f32; cfg.fn_cols()];
            mhc_combine(&cfg, post, comb, collapsed, streams.as_slice(), &mut out);
            (post.to_vec(), comb.to_vec(), collapsed.to_vec(), out)
        };
        let a = run();
        let b = run();
        assert_eq!(a.0, b.0, "post bits drifted");
        assert_eq!(a.1, b.1, "comb bits drifted");
        assert_eq!(a.2, b.2, "collapsed bits drifted");
        assert_eq!(a.3, b.3, "combine bits drifted");
    }

    /// Shape validation refuses every wrong surface loud.
    #[test]
    fn validation_refuses_wrong_shapes_loud() {
        let cfg = small_cfg();
        let rows = cfg.fn_rows();
        let cols = cfg.fn_cols();
        let ok = MhcWeights {
            fn_w: vec![0.0; rows * cols],
            base: vec![0.0; rows],
            scale: [1.0; 3],
        };
        assert!(ok.validate(&cfg).is_ok());
        let bad_fn = MhcWeights {
            fn_w: vec![0.0; rows * cols + 1],
            base: vec![0.0; rows],
            scale: [1.0; 3],
        };
        assert!(bad_fn.validate(&cfg).is_err());
        let bad_base = MhcWeights {
            fn_w: vec![0.0; rows * cols],
            base: vec![0.0; rows - 1],
            scale: [1.0; 3],
        };
        assert!(bad_base.validate(&cfg).is_err());
    }

    /// The GLM-5.3 full-size config reproduces the source shape formula:
    /// fn [24, 16384], base [24] — pinned here so a future config drift
    /// cannot silently move the seat's expectations.
    #[test]
    fn glm5_flash_shape_formula() {
        let cfg = MhcConfig::glm5_flash(4096, 1e-5);
        assert_eq!(cfg.fn_rows(), 24);
        assert_eq!(cfg.fn_cols(), 16384);
        let w = MhcWeights {
            fn_w: vec![0.0; 24 * 16384],
            base: vec![0.0; 24],
            scale: [1.0; 3],
        };
        assert!(w.validate(&cfg).is_ok());
    }

    /// The end-to-end forward (project → identity "block" → combine) against
    /// a fully independent in-test transcription of the source formula on a
    /// small random fixture — catches transcription bugs, not just internal
    /// consistency.
    #[test]
    fn full_forward_matches_independent_transcription() {
        // Tiny dims so the naive transcription stays readable.
        let cfg = MhcConfig {
            n_streams: 2,
            hidden_size: 2,
            sinkhorn_iters: 2,
            eps: 1e-6,
            rms_eps: 1e-5,
        };
        let w = rand_weights(&cfg, 101);
        w.validate(&cfg).unwrap();
        let n = cfg.n_streams;
        let d = cfg.hidden_size;
        let flat_len = n * d;
        let streams: Vec<f32> = (0..flat_len).map(|i| (i as f32 * 0.11).sin()).collect();

        let mut sc = MhcScratch::new(&cfg);
        let (post, comb, collapsed) = mhc_project(&cfg, &w.view(), &streams, &mut sc);
        let (post_h, comb_h, collapsed_h) = (post.to_vec(), comb.to_vec(), collapsed.to_vec());

        // ── independent transcription ──
        // norm: variance = Σx²/N·D then x·(1/sqrt(var+eps))
        let sum_sq: f32 = streams.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sum_sq / flat_len as f32 + cfg.rms_eps).sqrt();
        let flat: Vec<f32> = streams.iter().map(|v| v * inv).collect();
        // logits[r] = Σ_c fn_w[r*cols+c] * flat[c]
        let rows = cfg.fn_rows();
        let cols = cfg.fn_cols();
        let logits: Vec<f32> = (0..rows)
            .map(|r| {
                let row_w = &w.fn_w[r * cols..(r + 1) * cols];
                row_w.iter().zip(flat.iter()).map(|(wv, fv)| wv * fv).sum()
            })
            .collect();
        // comb row-softmax + eps, then sinkhorn(2): col ÷, then (row ÷, col ÷)
        let mut comb_naive = [[0.0f32; 2]; 2];
        for i in 0..n {
            let mut mx = f32::NEG_INFINITY;
            for j in 0..n {
                let v = logits[2 * n + i * n + j] * w.scale[2] + w.base[2 * n + i * n + j];
                comb_naive[i][j] = v;
                mx = mx.max(v);
            }
            let mut s = 0.0f32;
            for cell in comb_naive[i].iter_mut().take(n) {
                *cell = (*cell - mx).exp();
                s += *cell;
            }
            for cell in comb_naive[i].iter_mut().take(n) {
                *cell = *cell / s + cfg.eps;
            }
        }
        // sinkhorn: col, row, col (iters=2 → 1 col pass + 1 (row,col) pair)
        let norm_cols = |m: &mut [[f32; 2]; 2]| {
            for j in 0..n {
                let s = m[0][j] + m[1][j] + cfg.eps;
                for row in m.iter_mut().take(n) {
                    row[j] /= s;
                }
            }
        };
        let norm_rows = |m: &mut [[f32; 2]; 2]| {
            for row in m.iter_mut().take(n) {
                let s = row[0] + row[1] + cfg.eps;
                for cell in row.iter_mut().take(n) {
                    *cell /= s;
                }
            }
        };
        norm_cols(&mut comb_naive);
        norm_rows(&mut comb_naive);
        norm_cols(&mut comb_naive);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (comb_h[i * n + j] - comb_naive[i][j]).abs() < 1e-5,
                    "comb[{i}][{j}] = {} vs {}",
                    comb_h[i * n + j],
                    comb_naive[i][j]
                );
            }
        }
        // post
        for k in 0..n {
            let expected = 2.0 / (1.0 + (-(logits[n + k] * w.scale[1] + w.base[n + k])).exp());
            assert!((post_h[k] - expected).abs() < 1e-6);
        }
        // collapsed = Σ_k pre[k]·streams[k]
        for dd in 0..d {
            let mut acc = 0.0f32;
            for k in 0..n {
                let pre = 1.0 / (1.0 + (-(logits[k] * w.scale[0] + w.base[k])).exp()) + cfg.eps;
                acc += pre * streams[k * d + dd];
            }
            assert!(
                (collapsed_h[dd] - acc).abs() < 1e-5,
                "collapsed[{dd}] = {} vs {acc}",
                collapsed_h[dd]
            );
        }
        // combine with the identity block (block_out = collapsed)
        let mut out = vec![0.0f32; flat_len];
        mhc_combine(&cfg, &post_h, &comb_h, &collapsed_h, &streams, &mut out);
        for k in 0..n {
            for dd in 0..d {
                let mut acc = post_h[k] * collapsed_h[dd];
                for j in 0..n {
                    acc += comb_h[j * n + k] * streams[j * d + dd];
                }
                assert!(
                    (out[k * d + dd] - acc).abs() < 1e-5,
                    "out[{k}][{dd}] = {} vs {acc}",
                    out[k * d + dd]
                );
            }
        }
    }
}
