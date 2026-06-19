//! Functional Attention (FUNCATTN) — T5.1 SpectralQuant composition gate
//! (Plan 286 Phase 5).
//!
//! ## Hypothesis (from Plan 286 T5.1)
//!
//! Pre-rotating FUNCATTN's basis weights `w_basis` onto a calibrated
//! eigenbasis (from `katgpt_rs::spectralquant::calibrate_eigenbasis`) makes
//! FUNCATTN more expressive per parameter on **anisotropic** input
//! distributions. The adaptive partition `Φ[n, :]` can then distinguish
//! tokens by their principal-component scores instead of arbitrary
//! (isotropic-random) linear combinations.
//!
//! ## Setup
//!
//! - **Anisotropic dataset**: tokens drawn from `N(0, Σ)` with Σ having one
//!   dominant direction (eigenvalue ratio ≈ 16:1 between top and bottom).
//!   This is the regime where eigenbasis alignment should help most — the
//!   random-orthogonal `w_basis` init wastes most of its `k` rows on low-
//!   variance directions.
//! - **Target**: a smooth nonlinear function of the **top principal
//!   component** only: `Y[i, j] = sin(π·pc1[i]) · cos(0.5·j)`. The target
//!   signal lives entirely in the high-variance direction, so a model that
//!   can read PC1 directly has a structural advantage.
//! - **Same parameter budget**: identical orthogonal init for vanilla and
//!   eigen-aligned `w_basis`, identical `w_q/w_k/w_v`. Same FD-SGD steps.
//! - The only difference: eigen-aligned variant has `pre_rotate_basis_weights_into`
//!   applied to its `w_basis` once, before training, using the eigenvectors
//!   from `calibrate_eigenbasis(&samples, D)`.
//!
//! ## Gate (G6 — new gate added by T5.1)
//!
//! - **PASS**: eigen-aligned MSE ≤ vanilla MSE × 0.9 (≥10% improvement).
//! - **TIE**: ratio in [0.9, 1.1] — no clear benefit; document null result
//!   and keep the composition as an opt-in helper (no automatic promotion).
//! - **FAIL**: ratio > 1.1 — eigen-alignment *hurts* in this regime.
//!
//! The hard gate asserts only **mechanics**: both variants must learn
//! (final MSE < trivial-predictor MSE) and the composition must be
//! deterministic (same eigenvectors → same rotated weights). The PASS/TIE/FAIL
//! verdict on the *ratio* is reported via eprintln and recorded in the plan,
//! **not** asserted — matching the G2 pattern (research question, not
//! correctness question). Per the plan's Gain-tier opt-in policy, even a TIE
//! keeps the helper shipped; FAIL triggers an `.issues/` entry to investigate
//! the regime where alignment hurts.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features funcattn,spectral_quant --release \
//!   --test funcattn_t5_1_eigenbasis_compose -- --nocapture
//! ```
//!
//! (Release strongly recommended: 2 variants × ~200 steps × ~96 params × 2
//! FD evals ≈ 80k forward passes plus 1 eigendecomposition. Debug ~30× slower.)

#![cfg(all(feature = "funcattn", feature = "spectral_quant"))]

use katgpt_core::funcattn::{
    funcattn_forward, pre_rotate_basis_weights_into, FuncAttnBasis, FuncAttnConfig,
    FuncAttnScratch,
};
use katgpt_rs::spectralquant::calibrate_eigenbasis;

// ── Model dimensions ─────────────────────────────────────────────────────
const D: usize = 8;
const K: usize = 4;
const N: usize = 32;

// ── Training hyperparameters ─────────────────────────────────────────────
/// FD-SGD steps. 200 in release, 80 in debug (debug builds are ~30× slower).
#[cfg(not(debug_assertions))]
const STEPS: usize = 200;
#[cfg(debug_assertions)]
const STEPS: usize = 80;

const LR: f32 = 1.0;
const FD_EPS: f32 = 1e-2;
const ALPHA: f32 = 0.01;
const TEMPERATURE: f32 = 0.1;
const SEED: u64 = 0xCAFE_BABE_1234u64;

// ── Deterministic xorshift64* PRNG (matches G3/G2 test convention) ───────

struct Rng {
    s: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            s: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.s ^= self.s >> 12;
        self.s ^= self.s << 25;
        self.s ^= self.s >> 27;
        self.s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f32(&mut self) -> f32 {
        // Uniform [-1, 1)
        let bits = (self.next_u64() >> 40) as u32;
        let u01 = bits as f32 / ((1u32 << 24) as f32);
        u01 * 2.0 - 1.0
    }
    fn next_normal(&mut self) -> f32 {
        // Box-Muller transform: two uniforms → one standard normal.
        // We consume two PRNG draws; deterministic given state.
        // Use the same top-24-bits convention as `next_f32` to stay in [0, 1).
        let u1_raw = (self.next_u64() >> 40) as u32;
        let u2_raw = (self.next_u64() >> 40) as u32;
        // Guard against u1 == 0 (ln(0) = -inf); clamp to a tiny positive.
        let u1 = (u1_raw.max(1) as f32) / ((1u32 << 24) as f32);
        let u2 = (u2_raw as f32) / ((1u32 << 24) as f32);
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
    fn fill(&mut self, buf: &mut [f32]) {
        for x in buf.iter_mut() {
            *x = self.next_f32();
        }
    }
}

// ── Helper matrices ──────────────────────────────────────────────────────

fn identity_matrix(d: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; d * d];
    for i in 0..d {
        w[i * d + i] = 1.0;
    }
    w
}

/// Row-orthogonal init via modified Gram-Schmidt (matches G3 convention).
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
            let dot: f32 = (0..cols).map(|c| w[i * cols + c] * w[j * cols + c]).sum();
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

// ── Anisotropic dataset ──────────────────────────────────────────────────
//
// Tokens x ~ N(0, Σ) where Σ = diag(eigenvalues). We use eigenvalues
// {16, 4, 1, 1, 1, 1, 1, 1} so the top principal component carries ~64%
// of the variance. The target depends only on PC1 (the high-variance axis):
//
//   Y[i, j] = sin(π · pc1[i] / scale_pc1) · cos(0.5 · j)
//
// where pc1[i] = x[i, 0] (axis 0 = top eigendirection, since we draw with
// per-axis std = sqrt(eigenvalue)).

const EIGENVALUES: [f32; D] = [16.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
const PC1_AXIS: usize = 0; // axis with eigenvalue 16

fn anisotropic_dataset(rng: &mut Rng) -> (Vec<f32>, Vec<f32>) {
    // Returns (x, y) with x shape (N, D), y shape (N, D).
    let mut x = vec![0.0f32; N * D];
    for i in 0..N {
        for j in 0..D {
            // x[i, j] ~ N(0, eigenvalue[j]) → std = sqrt(eigenvalue[j]).
            let std = EIGENVALUES[j].sqrt();
            x[i * D + j] = rng.next_normal() * std;
        }
    }

    // Target depends on PC1 only. We use the *unnormalized* PC1 value
    // (which has std=4 since eigenvalue=16) so sin(π·x/4) gives ~1 full
    // cycle across the ±2σ range.
    let pc1_std = EIGENVALUES[PC1_AXIS].sqrt();
    let mut y = vec![0.0f32; N * D];
    for i in 0..N {
        let pc1 = x[i * D + PC1_AXIS];
        let base = (std::f32::consts::PI * pc1 / (2.0 * pc1_std)).sin();
        for j in 0..D {
            y[i * D + j] = base * (0.5 * j as f32).cos();
        }
    }
    (x, y)
}

// ── Training infrastructure (mirrors G3 test) ────────────────────────────

fn forward_mse(
    x_basis: &[f32],
    x_value: &[f32],
    y: &[f32],
    w_basis: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    scratch: &mut FuncAttnScratch,
    out: &mut [f32],
) -> f32 {
    let cfg = FuncAttnConfig {
        d: D,
        k: K,
        basis: FuncAttnBasis::Sigmoid,
        alpha: ALPHA,
        temperature: TEMPERATURE,
        cholesky_jitter: 1e-6,
    };
    funcattn_forward(x_basis, x_value, w_basis, w_q, w_k, w_v, &cfg, scratch, out)
        .expect("forward should succeed");
    let mut s = 0.0f32;
    for i in 0..out.len() {
        let diff = out[i] - y[i];
        s += diff * diff;
    }
    s / (N * D) as f32
}

fn fd_sgd_step(
    x_basis: &[f32],
    x_value: &[f32],
    y: &[f32],
    w_basis: &mut [f32],
    w_q: &mut [f32],
    w_k: &mut [f32],
    w_v: &mut [f32],
    scratch: &mut FuncAttnScratch,
    out: &mut [f32],
) -> f32 {
    let inv_2eps = 1.0 / (2.0 * FD_EPS);
    for i in 0..w_basis.len() {
        let orig = w_basis[i];
        w_basis[i] = orig + FD_EPS;
        let lp = forward_mse(x_basis, x_value, y, w_basis, w_q, w_k, w_v, scratch, out);
        w_basis[i] = orig - FD_EPS;
        let lm = forward_mse(x_basis, x_value, y, w_basis, w_q, w_k, w_v, scratch, out);
        w_basis[i] = orig;
        let grad = (lp - lm) * inv_2eps;
        w_basis[i] = orig - LR * grad;
    }
    for i in 0..w_q.len() {
        let orig = w_q[i];
        w_q[i] = orig + FD_EPS;
        let lp = forward_mse(x_basis, x_value, y, w_basis, w_q, w_k, w_v, scratch, out);
        w_q[i] = orig - FD_EPS;
        let lm = forward_mse(x_basis, x_value, y, w_basis, w_q, w_k, w_v, scratch, out);
        w_q[i] = orig;
        let grad = (lp - lm) * inv_2eps;
        w_q[i] = orig - LR * grad;
    }
    for i in 0..w_k.len() {
        let orig = w_k[i];
        w_k[i] = orig + FD_EPS;
        let lp = forward_mse(x_basis, x_value, y, w_basis, w_q, w_k, w_v, scratch, out);
        w_k[i] = orig - FD_EPS;
        let lm = forward_mse(x_basis, x_value, y, w_basis, w_q, w_k, w_v, scratch, out);
        w_k[i] = orig;
        let grad = (lp - lm) * inv_2eps;
        w_k[i] = orig - LR * grad;
    }
    for i in 0..w_v.len() {
        let orig = w_v[i];
        w_v[i] = orig + FD_EPS;
        let lp = forward_mse(x_basis, x_value, y, w_basis, w_q, w_k, w_v, scratch, out);
        w_v[i] = orig - FD_EPS;
        let lm = forward_mse(x_basis, x_value, y, w_basis, w_q, w_k, w_v, scratch, out);
        w_v[i] = orig;
        let grad = (lp - lm) * inv_2eps;
        w_v[i] = orig - LR * grad;
    }
    forward_mse(x_basis, x_value, y, w_basis, w_q, w_k, w_v, scratch, out)
}

fn train_variant(
    label: &str,
    x_basis: &[f32],
    x_value: &[f32],
    y: &[f32],
    init_w_basis: &[f32],
    init_w_q: &[f32],
    init_w_k: &[f32],
    init_w_v: &[f32],
) -> f32 {
    let mut w_basis = init_w_basis.to_vec();
    let mut w_q = init_w_q.to_vec();
    let mut w_k = init_w_k.to_vec();
    let mut w_v = init_w_v.to_vec();
    let mut scratch = FuncAttnScratch::new(N, D, K);
    let mut out = vec![0.0f32; N * D];

    let mut last_mse = forward_mse(
        x_basis, x_value, y, &w_basis, &w_q, &w_k, &w_v, &mut scratch, &mut out,
    );
    eprintln!("[{}] init mse = {:.6}", label, last_mse);

    for step in 0..STEPS {
        last_mse = fd_sgd_step(
            x_basis,
            x_value,
            y,
            &mut w_basis,
            &mut w_q,
            &mut w_k,
            &mut w_v,
            &mut scratch,
            &mut out,
        );
        if step == 0 || (step + 1) % 25 == 0 || step + 1 == STEPS {
            eprintln!("[{}] step {:>4}/{:<4}  mse = {:.6}", label, step + 1, STEPS, last_mse);
        }
    }
    last_mse
}

// ── G6 gate ──────────────────────────────────────────────────────────────

#[test]
fn g6_eigenbasis_aligned_beats_vanilla_on_anisotropic() {
    let mut rng = Rng::new(SEED);

    // Build the anisotropic dataset. We use the same (x, y) for both variants
    // and for the SpectralQuant calibration — calibration is "offline" and
    // uses the same tokens we'll train on, which is the typical deployment
    // pattern (calibrate once on a representative corpus).
    let (x, y) = anisotropic_dataset(&mut rng);

    // Calibrate the eigenbasis from the same tokens (this is what
    // SpectralQuant does in production — calibrate once, deploy forever).
    // Each token is a row of `x`; pass them as `Vec<f32>` samples.
    let samples: Vec<Vec<f32>> = (0..N).map(|i| x[i * D..(i + 1) * D].to_vec()).collect();
    let calibration = calibrate_eigenbasis(&samples, D);
    let eigenvectors = &calibration.eigenvectors;

    eprintln!("\n=== G6: FUNCATTN T5.1 eigenbasis composition gate ===");
    eprintln!(
        "model: n={}, d={}, k={}, steps={} (FD-SGD, LR={}, FD_EPS={}, τ={})",
        N, D, K, STEPS, LR, FD_EPS, TEMPERATURE
    );
    eprintln!(
        "calibration: n_samples={}, d_eff={:.2}, top_eig={:.4}, bot_eig={:.4}, spectral_gap={:?}",
        calibration.n_samples,
        calibration.d_eff,
        calibration.eigenvalues[0],
        calibration.eigenvalues[D - 1],
        calibration.spectral_gap
    );
    eprintln!(
        "eigenvalues: {:?}",
        calibration.eigenvalues
    );

    // Identical orthogonal + identity inits for both variants.
    let init_w_basis = orthogonal_init(K, D, &mut rng);
    let init_w_q: Vec<f32> = identity_matrix(D);
    let init_w_k: Vec<f32> = identity_matrix(D);
    let init_w_v: Vec<f32> = identity_matrix(D);

    // Eigen-aligned variant: copy the init, then pre-rotate.
    let mut init_w_basis_aligned = init_w_basis.clone();
    pre_rotate_basis_weights_into(&mut init_w_basis_aligned, eigenvectors, K, D);

    // Vanilla FUNCATTN.
    let vanilla_mse = train_variant(
        "vanilla  ",
        &x, &x, &y, &init_w_basis, &init_w_q, &init_w_k, &init_w_v,
    );

    // Eigen-aligned FUNCATTN.
    let aligned_mse = train_variant(
        "eigen    ",
        &x, &x, &y, &init_w_basis_aligned, &init_w_q, &init_w_k, &init_w_v,
    );

    // ── G6 verdict ───────────────────────────────────────────────────────
    let ratio = aligned_mse / vanilla_mse.max(1e-20);
    eprintln!("\n=== G6 verdict ===");
    eprintln!("  vanilla FUNCATTN    mse = {:.6}", vanilla_mse);
    eprintln!("  eigen-aligned       mse = {:.6}", aligned_mse);
    eprintln!("  aligned / vanilla ratio = {:.4}", ratio);
    eprintln!("  gate: ratio ≤ 0.90 (PASS, ≥10% improvement)");
    eprintln!("        ratio ≤ 1.10 (TIE, no clear benefit — keep helper opt-in)");
    eprintln!("        ratio > 1.10 (FAIL, alignment hurts — escalate issue)");

    // Hard gate: BOTH variants must have learned (mechanics check).
    // We do NOT assert on the ratio — that's a research question reported
    // via eprintln above, not a correctness check. The primitive
    // (pre_rotate_basis_weights_into) is verified lossless by its own unit
    // tests; the G6 ratio just tells us whether the composition helps in
    // this particular regime.
    let y_mean: f32 = y.iter().sum::<f32>() / y.len() as f32;
    let trivial_mse: f32 = y.iter().map(|&v| (v - y_mean) * (v - y_mean)).sum::<f32>() / y.len() as f32;
    eprintln!("  trivial-predictor mse = {:.6} (variance of y)", trivial_mse);
    assert!(
        vanilla_mse.is_finite() && vanilla_mse < trivial_mse,
        "vanilla FUNCATTN did not learn: mse={} vs trivial={}",
        vanilla_mse,
        trivial_mse
    );
    assert!(
        aligned_mse.is_finite() && aligned_mse < trivial_mse,
        "eigen-aligned FUNCATTN did not learn: mse={} vs trivial={}",
        aligned_mse,
        trivial_mse
    );

    // Determinism check: re-running the rotation on the same weights must
    // produce byte-identical results (the primitive is deterministic given
    // the same eigenvectors).
    {
        let mut w_recheck = init_w_basis.clone();
        pre_rotate_basis_weights_into(&mut w_recheck, eigenvectors, K, D);
        let max_diff: f32 = w_recheck
            .iter()
            .zip(init_w_basis_aligned.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "rotation not deterministic: max_diff = {}", max_diff);
        eprintln!("  determinism: rotation reproducible (max_diff = {:.2e})", max_diff);
    }

    if ratio <= 0.90 {
        eprintln!("  → G6 PASS (eigen-aligned ≥10% better than vanilla).");
    } else if ratio <= 1.10 {
        eprintln!(
            "  → G6 TIE (eigen-aligned within ±10% of vanilla — no clear benefit on this dataset)."
        );
    } else {
        eprintln!(
            "  → G6 FAIL (eigen-aligned >10% worse). The primitive is still correct and lossless;"
        );
        eprintln!(
            "    the composition just doesn't help in this regime. Document in plan + .issues/."
        );
    }
}
