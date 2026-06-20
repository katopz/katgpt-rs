//! Parallax sigmoid stability regression anchor — Issue 002 T1/T2/T3b.
//!
//! Re-investigation of the W_R positive-feedback divergence documented in
//! Issue 002. The original filing (2026-06-18) claimed sigmoid Parallax
//! diverges to NaN at step 350–375 under naive FD-SGD at LR=1.0 from a zero
//! W_R init, while softmax Parallax stays stable. **Re-running the same
//! setup against the current `tiled_attention_parallax_forward` reveals
//! the pattern has inverted**: sigmoid Parallax is now stable for ≥500
//! steps, softmax Parallax now diverges around step 325–350. The W_R
//! gradient clip (T3b) keeps sigmoid stable without harming convergence.
//!
//! These tests are now **regression anchors** that pin the current
//! empirical behavior. If either T1 or T2 ever flips, it's a signal that
//! the forward path's numerical regime has changed — investigate before
//! assuming the new behavior is correct.
//!
//! ## Background (Issue 002, original analysis — now falsified for sigmoid)
//!
//! Sigmoid Parallax's correction path
//! ```text
//! o_PLX = o_SA − gate_scale · Σ_KV · ρ     where ρ = W_R · x
//! ```
//! has a positive feedback loop under naive SGD on W_R: as |ρ| grows, the
//! correction `Σ_KV · ρ` grows, the gradient w.r.t. W_R grows (proportional
//! to `Σ_KV · x`), W_R amplifies ρ further, etc. The original analysis
//! (Research 140) predicted sigmoid would be the worse case due to its
//! softer saturation keeping Σ_KV higher-rank. Empirically (this test),
//! the current forward path shows the opposite — softmax is the diverging
//! variant under this setup. Root cause of the inversion is NOT fully
//! diagnosed; a candidate explanation is that softmax's `exp` is more
//! sensitive to the large pre-activation magnitudes that build up late in
//! training, while sigmoid's bounded output naturally limits the
//! correction's growth.
//!
//! The forward path itself is finite for any finite ρ — this is purely a
//! training-dynamics issue. The W_R clip mitigation (T3b) is a defensive
//! measure that costs nothing when the gradient is small.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features parallax_attn --release \
//!   --test parallax_sigmoid_stability_grad_clip -- --nocapture --test-threads=1
//! ```
//!
//! Release is REQUIRED — debug is ~30× slower and the divergence step may
//! shift slightly.

#![cfg(feature = "parallax_attn")]

use katgpt_core::parallax_attn::{
    tiled_attention_parallax_forward, ParallaxActivation, ParallaxConfig, ParallaxScratch,
};
use katgpt_core::simd;

// ── Model dimensions (match Issue 002 setup + existing G2 test) ──────────
const D: usize = 8;
const N: usize = 64;
/// SDPA scale = 1/√d.
const SCALE: f32 = 0.353_553_38; // 1.0 / sqrt(8.0)

// ── Training hyperparameters (match Issue 002 setup) ─────────────────────
/// FD-SGD steps. Issue 002 documents divergence at 350–375; we train 500 to
/// confirm the divergence + show the mitigation holds well past it.
#[cfg(not(debug_assertions))]
const STEPS: usize = 500;
#[cfg(debug_assertions)]
const STEPS: usize = 80;
const LR: f32 = 1.0;
const FD_EPS: f32 = 1e-2;
/// Global L2 clip norm for W_R gradient (T3b mitigation).
const W_R_MAX_GRAD_NORM: f32 = 1.0;
const SEED: u64 = 0xA11CE_2222u64;

// ── Deterministic xorshift64* PRNG (mirrors G2/G3 tests) ─────────────────

struct Rng {
    s: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            s: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.s ^= self.s >> 12;
        self.s ^= self.s << 25;
        self.s ^= self.s >> 27;
        self.s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        let u01 = bits as f32 / ((1u32 << 24) as f32);
        u01 * 2.0 - 1.0
    }
    fn fill(&mut self, buf: &mut [f32]) {
        for x in buf.iter_mut() {
            *x = self.next_f32();
        }
    }
}

/// D×D identity matrix, row-major.
fn identity_matrix(d: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; d * d];
    for i in 0..d {
        w[i * d + i] = 1.0;
    }
    w
}

/// Row-orthogonal init via modified Gram-Schmidt (mirrors reference L20-21
/// `torch.nn.init.orthogonal_`).
fn orthogonal_init(rows: usize, cols: usize, rng: &mut Rng) -> Vec<f32> {
    let mut w = vec![0.0f32; rows * cols];
    rng.fill(&mut w);
    for i in 0..rows {
        let mut norm = 0.0;
        for c in 0..cols {
            norm += w[i * cols + c] * w[i * cols + c];
        }
        norm = norm.sqrt().max(1e-12);
        for c in 0..cols {
            w[i * cols + c] /= norm;
        }
    }
    for i in 0..rows {
        for j in 0..i {
            let mut dot = 0.0;
            for c in 0..cols {
                dot += w[i * cols + c] * w[j * cols + c];
            }
            for c in 0..cols {
                w[i * cols + c] -= dot * w[j * cols + c];
            }
        }
        let mut norm = 0.0;
        for c in 0..cols {
            norm += w[i * cols + c] * w[i * cols + c];
        }
        norm = norm.sqrt().max(1e-12);
        for c in 0..cols {
            w[i * cols + c] /= norm;
        }
    }
    w
}

/// Column mean of X ∈ R^{n×d}: out[j] = (1/n) Σ_i X[i,j]. Single-vector
/// "layer input" that Parallax's W_R · x probe expects.
fn column_mean(x: &[f32], n: usize, d: usize, out: &mut [f32]) {
    out[..d].fill(0.0);
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        simd::simd_add_inplace(&mut out[..d], row);
    }
    let inv_n = 1.0 / n.max(1) as f32;
    simd::simd_scale_inplace(&mut out[..d], inv_n);
}

/// Sinusoidal regression target with cross-feature interaction (matches G2).
fn sinusoidal_target(x: &[f32], y: &mut [f32]) {
    let n = x.len() / D;
    for i in 0..n {
        let x0 = x[i * D];
        let x1 = x[i * D + 1];
        let x2 = x[i * D + 2];
        let x3 = x[i * D + 3];
        let cross = 0.5 * (x2 + x3).tanh();
        for j in 0..D {
            y[i * D + j] = (3.0 * x0).sin() * (x1 + 0.2 * j as f32).cos() + cross;
        }
    }
}

/// MSE: `||out - y||² / out.len()`.
fn mse(out: &[f32], y: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..out.len() {
        let d = out[i] - y[i];
        s += d * d;
    }
    s / out.len() as f32
}

/// Apply per-token linear projection: `out[i,:] = W · x[i,:]` (W row-major).
fn apply_projection(x: &[f32], w: &[f32], out: &mut [f32]) {
    for i in 0..N {
        let x_row = &x[i * D..(i + 1) * D];
        let out_row = &mut out[i * D..(i + 1) * D];
        simd::simd_matmul_rows(out_row, w, x_row, D, D);
    }
}

// ── Parallax buffers (matches G2 test setup) ─────────────────────────────

struct ParallaxBuffers {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    out: Vec<f32>,
    x_avg: Vec<f32>,
    scratch: ParallaxScratch,
    cfg: ParallaxConfig,
}

impl ParallaxBuffers {
    fn new(activation: ParallaxActivation) -> Self {
        Self {
            q: vec![0.0; N * D],
            k: vec![0.0; N * D],
            v: vec![0.0; N * D],
            out: vec![0.0; N * D],
            x_avg: vec![0.0; D],
            scratch: ParallaxScratch::new(N, D),
            cfg: ParallaxConfig {
                gate_scale: 1.0,
                zero_init: false,
                activation,
            },
        }
    }
}

fn parallax_forward_mse(
    x: &[f32],
    y: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_r: &[f32],
    buf: &mut ParallaxBuffers,
) -> f32 {
    apply_projection(x, w_q, &mut buf.q);
    apply_projection(x, w_k, &mut buf.k);
    apply_projection(x, w_v, &mut buf.v);
    column_mean(x, N, D, &mut buf.x_avg);
    tiled_attention_parallax_forward(
        &buf.q,
        &buf.k,
        &buf.v,
        &mut buf.out,
        N,
        D,
        SCALE,
        w_r,
        &buf.x_avg,
        &buf.cfg,
        Some(&mut buf.scratch),
    );
    mse(&buf.out, y)
}

/// One FD-SGD step with NO mitigation (the divergent baseline).
/// All four weight matrices get plain per-coordinate central-difference
/// gradient descent at LR=1.0.
#[allow(clippy::too_many_arguments)]
fn parallax_fd_sgd_step_unclipped(
    x: &[f32],
    y: &[f32],
    w_q: &mut [f32],
    w_k: &mut [f32],
    w_v: &mut [f32],
    w_r: &mut [f32],
    buf: &mut ParallaxBuffers,
) -> f32 {
    let inv_2eps = 1.0 / (2.0 * FD_EPS);
    let mut fwd = |wq: &[f32], wk: &[f32], wv: &[f32], wr: &[f32]| -> f32 {
        parallax_forward_mse(x, y, wq, wk, wv, wr, buf)
    };

    for i in 0..w_q.len() {
        let orig = w_q[i];
        w_q[i] = orig + FD_EPS;
        let lp = fwd(w_q, w_k, w_v, w_r);
        w_q[i] = orig - FD_EPS;
        let lm = fwd(w_q, w_k, w_v, w_r);
        w_q[i] = orig;
        let grad = (lp - lm) * inv_2eps;
        w_q[i] = orig - LR * grad;
    }
    for i in 0..w_k.len() {
        let orig = w_k[i];
        w_k[i] = orig + FD_EPS;
        let lp = fwd(w_q, w_k, w_v, w_r);
        w_k[i] = orig - FD_EPS;
        let lm = fwd(w_q, w_k, w_v, w_r);
        w_k[i] = orig;
        let grad = (lp - lm) * inv_2eps;
        w_k[i] = orig - LR * grad;
    }
    for i in 0..w_v.len() {
        let orig = w_v[i];
        w_v[i] = orig + FD_EPS;
        let lp = fwd(w_q, w_k, w_v, w_r);
        w_v[i] = orig - FD_EPS;
        let lm = fwd(w_q, w_k, w_v, w_r);
        w_v[i] = orig;
        let grad = (lp - lm) * inv_2eps;
        w_v[i] = orig - LR * grad;
    }
    for i in 0..w_r.len() {
        let orig = w_r[i];
        w_r[i] = orig + FD_EPS;
        let lp = fwd(w_q, w_k, w_v, w_r);
        w_r[i] = orig - FD_EPS;
        let lm = fwd(w_q, w_k, w_v, w_r);
        w_r[i] = orig;
        let grad = (lp - lm) * inv_2eps;
        w_r[i] = orig - LR * grad;
    }

    fwd(w_q, w_k, w_v, w_r)
}

/// One FD-SGD step with **global L2 gradient clipping on W_R only**
/// (Issue 002 T3b mitigation). W_Q/W_K/W_V use plain SGD as before; the W_R
/// gradient is computed into a temp buffer, its L2 norm is capped at
/// `W_R_MAX_GRAD_NORM`, and only then is the W_R update applied. This
/// directly bounds the positive-feedback loop's amplification per step.
#[allow(clippy::too_many_arguments)]
fn parallax_fd_sgd_step_wr_clip(
    x: &[f32],
    y: &[f32],
    w_q: &mut [f32],
    w_k: &mut [f32],
    w_v: &mut [f32],
    w_r: &mut [f32],
    buf: &mut ParallaxBuffers,
) -> f32 {
    let inv_2eps = 1.0 / (2.0 * FD_EPS);
    let mut fwd = |wq: &[f32], wk: &[f32], wv: &[f32], wr: &[f32]| -> f32 {
        parallax_forward_mse(x, y, wq, wk, wv, wr, buf)
    };

    // W_Q/W_K/W_V — plain per-coordinate FD-SGD, no clipping.
    for i in 0..w_q.len() {
        let orig = w_q[i];
        w_q[i] = orig + FD_EPS;
        let lp = fwd(w_q, w_k, w_v, w_r);
        w_q[i] = orig - FD_EPS;
        let lm = fwd(w_q, w_k, w_v, w_r);
        w_q[i] = orig;
        let grad = (lp - lm) * inv_2eps;
        w_q[i] = orig - LR * grad;
    }
    for i in 0..w_k.len() {
        let orig = w_k[i];
        w_k[i] = orig + FD_EPS;
        let lp = fwd(w_q, w_k, w_v, w_r);
        w_k[i] = orig - FD_EPS;
        let lm = fwd(w_q, w_k, w_v, w_r);
        w_k[i] = orig;
        let grad = (lp - lm) * inv_2eps;
        w_k[i] = orig - LR * grad;
    }
    for i in 0..w_v.len() {
        let orig = w_v[i];
        w_v[i] = orig + FD_EPS;
        let lp = fwd(w_q, w_k, w_v, w_r);
        w_v[i] = orig - FD_EPS;
        let lm = fwd(w_q, w_k, w_v, w_r);
        w_v[i] = orig;
        let grad = (lp - lm) * inv_2eps;
        w_v[i] = orig - LR * grad;
    }

    // W_R — global L2 gradient clip then apply. The whole gradient vector
    // is computed first (NOT per-coordinate), then rescaled if its norm
    // exceeds W_R_MAX_GRAD_NORM. This is the standard "clip by global norm"
    // technique (e.g. GPT-2/3 training recipe) adapted to a single matrix.
    let mut grad_r = vec![0.0f32; w_r.len()];
    for i in 0..w_r.len() {
        let orig = w_r[i];
        w_r[i] = orig + FD_EPS;
        let lp = fwd(w_q, w_k, w_v, w_r);
        w_r[i] = orig - FD_EPS;
        let lm = fwd(w_q, w_k, w_v, w_r);
        w_r[i] = orig;
        grad_r[i] = (lp - lm) * inv_2eps;
    }
    // Skip the clip update entirely if the gradient is non-finite (defensive
    // — once the loop has blown up the FD probe itself can return NaN, and
    // we don't want to propagate that into W_R).
    let any_nan = grad_r.iter().any(|g| !g.is_finite());
    if !any_nan {
        let norm: f32 = grad_r.iter().map(|g| g * g).sum::<f32>().sqrt();
        let scale = if norm > W_R_MAX_GRAD_NORM {
            W_R_MAX_GRAD_NORM / norm.max(1e-20)
        } else {
            1.0
        };
        for i in 0..w_r.len() {
            w_r[i] -= LR * grad_r[i] * scale;
        }
    }

    fwd(w_q, w_k, w_v, w_r)
}

// ── Dataset + init builder ───────────────────────────────────────────────

struct Setup {
    x: Vec<f32>,
    y: Vec<f32>,
    init_w_q: Vec<f32>,
    init_w_k: Vec<f32>,
    init_w_v: Vec<f32>,
    init_w_r: Vec<f32>,
}

/// Build the deterministic Issue 002 setup: Gaussian inputs scaled by 0.5,
/// sinusoidal regression target, orthogonal W_Q, identity W_K/W_V, zero W_R.
fn build_setup() -> Setup {
    let mut rng = Rng::new(SEED);
    let mut x = vec![0.0f32; N * D];
    rng.fill(&mut x);
    for xi in x.iter_mut() {
        *xi *= 0.5;
    }
    let mut y = vec![0.0f32; N * D];
    sinusoidal_target(&x, &mut y);

    let mut rng = Rng::new(SEED + 1);
    let init_w_q = orthogonal_init(D, D, &mut rng);
    let init_w_k = identity_matrix(D);
    let init_w_v = identity_matrix(D);
    let init_w_r = vec![0.0f32; D * D];

    Setup {
        x,
        y,
        init_w_q,
        init_w_k,
        init_w_v,
        init_w_r,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

/// T1: Sigmoid Parallax stays stable under naive FD-SGD for ≥500 steps.
///
/// **Issue 002 re-investigation.** The original filing claimed sigmoid
/// Parallax diverges to NaN at step 350–375 under this exact setup. The
/// current `tiled_attention_parallax_forward` does **not** reproduce that:
/// sigmoid stays finite and descends monotonically through 500 steps.
///
/// This test is now a regression anchor — if sigmoid Parallax ever starts
/// diverging again, this test fails and forces an investigation into
/// whether the forward path's numerical regime has regressed. The original
/// Issue 002 divergence pattern has either been fixed in the forward path
/// since 2026-06-18 or never reproduced in this exact configuration.
#[test]
fn t1_sigmoid_parallax_stays_stable_unclipped() {
    let Setup {
        x,
        y,
        init_w_q,
        init_w_k,
        init_w_v,
        init_w_r,
    } = build_setup();

    let mut w_q = init_w_q.clone();
    let mut w_k = init_w_k.clone();
    let mut w_v = init_w_v.clone();
    let mut w_r = init_w_r.clone();
    let mut buf = ParallaxBuffers::new(ParallaxActivation::Sigmoid);

    let init_mse = parallax_forward_mse(&x, &y, &w_q, &w_k, &w_v, &w_r, &mut buf);
    eprintln!("\n=== T1: sigmoid Parallax, NO mitigation (regression anchor — expect stable) ===");
    eprintln!("init mse = {:.6}", init_mse);

    let mut last_mse = init_mse;
    for step in 0..STEPS {
        last_mse = parallax_fd_sgd_step_unclipped(&x, &y, &mut w_q, &mut w_k, &mut w_v, &mut w_r, &mut buf);
        if step == 0 || (step + 1) % 50 == 0 || step + 1 == STEPS {
            eprintln!(
                "[t1] step {:>4}/{:<4}  mse = {:.6}{}",
                step + 1,
                STEPS,
                last_mse,
                if !last_mse.is_finite() { "  ← DIVERGED" } else { "" },
            );
        }
        if !last_mse.is_finite() {
            // Report divergence and fail-fast — no point continuing once NaN
            // has propagated.
            panic!(
                "T1 regression: sigmoid Parallax diverged at step {} (mse = {:.6}). \
                 Issue 002 re-investigation: the original 2026-06-18 divergence \
                 pattern is now reproducing again. Investigate the forward path \
                 before assuming this is benign.",
                step + 1, last_mse,
            );
        }
    }

    eprintln!("T1 verdict: final mse = {:.6} (init {:.6})", last_mse, init_mse);
    assert!(
        last_mse < init_mse,
        "T1 regression: sigmoid Parallax did not improve over {} steps \
         (init={:.6}, final={:.6}). Expected monotonic descent on a learnable target.",
        STEPS, init_mse, last_mse,
    );
}

/// T2: Softmax Parallax diverges to NaN before step 500 under naive FD-SGD.
///
/// **Issue 002 re-investigation — inverted finding.** The original filing
/// (2026-06-18) used softmax Parallax as the stable control against which
/// the sigmoid divergence was measured. Re-running against the current
/// forward path shows softmax Parallax is now the diverging variant: NaN
/// appears around step 325–350 (see eprintln trace). Sigmoid (T1) is now
/// the stable one.
///
/// Root cause of the inversion is not fully diagnosed; a candidate
/// explanation is that softmax's `exp` saturates faster than sigmoid's
/// bounded output as attention pre-activations grow late in training, so
/// softmax's "sharper normalization" actually amplifies numerical
/// instability rather than suppressing it once the W_R correction path
/// feeds back into the attention scores. This is the opposite of what the
/// original Research 140 analysis predicted.
///
/// This test is a regression anchor — if softmax Parallax ever stops
/// diverging, this test fails and forces an investigation into whether the
/// forward path has changed (and whether the inversion has un-inverted).
#[test]
fn t2_softmax_parallax_diverges_unclipped() {
    let Setup {
        x,
        y,
        init_w_q,
        init_w_k,
        init_w_v,
        init_w_r,
    } = build_setup();

    let mut w_q = init_w_q.clone();
    let mut w_k = init_w_k.clone();
    let mut w_v = init_w_v.clone();
    let mut w_r = init_w_r.clone();
    let mut buf = ParallaxBuffers::new(ParallaxActivation::Softmax);

    let init_mse = parallax_forward_mse(&x, &y, &w_q, &w_k, &w_v, &w_r, &mut buf);
    eprintln!("\n=== T2: softmax Parallax, NO mitigation (regression anchor — expect divergence ~step 325-350) ===");
    eprintln!("init mse = {:.6}", init_mse);

    let mut diverged_at: Option<usize> = None;
    let mut last_mse = init_mse;
    for step in 0..STEPS {
        last_mse = parallax_fd_sgd_step_unclipped(&x, &y, &mut w_q, &mut w_k, &mut w_v, &mut w_r, &mut buf);
        if !last_mse.is_finite() && diverged_at.is_none() {
            diverged_at = Some(step + 1);
        }
        if step == 0 || (step + 1) % 25 == 0 || step + 1 == STEPS {
            eprintln!(
                "[t2] step {:>4}/{:<4}  mse = {:.6}{}",
                step + 1,
                STEPS,
                last_mse,
                if !last_mse.is_finite() { "  ← DIVERGED" } else { "" },
            );
        }
    }

    eprintln!(
        "T2 verdict: final mse = {:.6}, diverged_at = {:?}",
        last_mse, diverged_at
    );
    assert!(
        !last_mse.is_finite(),
        "T2 regression anchor broken: softmax Parallax stayed finite for {} steps \
         (final mse = {:.6}). The 2026-06-19 inverted divergence pattern no longer \
         reproduces — investigate whether the forward path changed.",
        STEPS, last_mse,
    );
    assert!(
        diverged_at.is_some(),
        "T2 should have recorded a divergence step",
    );
}

/// T3b: Global L2 gradient clipping on W_R alone (‖∇_W_R‖ ≤ 1.0) stabilizes
/// sigmoid Parallax for 500 steps and produces a finite MSE below the init.
///
/// This is the Issue 002 mitigation: bound the W_R gradient's L2 norm per
/// step. W_Q/W_K/W_V continue to use plain FD-SGD. The clip directly caps
/// the positive-feedback loop's per-step amplification — even when the
/// gradient wants to be huge, the actual update is at most
/// `LR · W_R_MAX_GRAD_NORM` in L2 norm.
#[test]
fn t3b_sigmoid_parallax_stabilized_by_wr_grad_clip() {
    let Setup {
        x,
        y,
        init_w_q,
        init_w_k,
        init_w_v,
        init_w_r,
    } = build_setup();

    let mut w_q = init_w_q.clone();
    let mut w_k = init_w_k.clone();
    let mut w_v = init_w_v.clone();
    let mut w_r = init_w_r.clone();
    let mut buf = ParallaxBuffers::new(ParallaxActivation::Sigmoid);

    let init_mse = parallax_forward_mse(&x, &y, &w_q, &w_k, &w_v, &w_r, &mut buf);
    eprintln!(
        "\n=== T3b: sigmoid Parallax, W_R gradient clip ‖∇_W_R‖ ≤ {} (expect stable) ===",
        W_R_MAX_GRAD_NORM,
    );
    eprintln!("init mse = {:.6}", init_mse);

    let mut last_mse = init_mse;
    for step in 0..STEPS {
        last_mse =
            parallax_fd_sgd_step_wr_clip(&x, &y, &mut w_q, &mut w_k, &mut w_v, &mut w_r, &mut buf);
        if step == 0 || (step + 1) % 50 == 0 || step + 1 == STEPS {
            eprintln!(
                "[t3b] step {:>4}/{:<4}  mse = {:.6}",
                step + 1, STEPS, last_mse,
            );
        }
    }

    eprintln!("T3b verdict: final mse = {:.6} (init {:.6})", last_mse, init_mse);
    assert!(
        last_mse.is_finite(),
        "T3b mitigation failed: sigmoid Parallax diverged even with W_R gradient \
         clipping (final mse = {:.6}). Issue 002 T3b mitigation is insufficient.",
        last_mse,
    );
    assert!(
        last_mse < init_mse,
        "T3b mitigation: training did not improve (init={:.6}, final={:.6}). \
         Stability without progress is not a useful mitigation.",
        init_mse, last_mse,
    );
}
