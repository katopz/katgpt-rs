//! subspace_intervention — the three-arm subspace-intervention protocol
//! (Issue 779 T1 / Research 557; arXiv:2607.01987, "Understanding Geometric
//! Representations in SSL ViTs via Subspace Intervention", Zhou et al.
//! ECCV 2026).
//!
//! # What ships here
//!
//! | Primitive | Role |
//! |---|---|
//! | [`ridge_probe_fit_into`] | the modelless probe: closed-form ridge fit `W (c×d)` over a labeled activation bank, solved through the SVD of the regularized Gram (right vectors only — see the kinship note) |
//! | [`eval_head_into`] | frozen-head readout: argmax accuracy + per-class recall |
//! | [`project_through_into`] | `Ẑ = Z·B·Bᵀ` (or the orthogonal complement) through a `d×k` orthonormal basis |
//! | [`random_basis_into`] | seeded random orthonormal control basis (the protocol's control arm) |
//! | [`basis_similarity`] | principal-angle similarity `(1/k)·‖V_aᵀV_b‖²_F ∈ (0, 1]` (Li et al. 2021) — the span-stability axis |
//! | [`three_arm_eval`] | the protocol: aligned / random / residual accuracies through the frozen probe head, per candidate `k` |
//! | [`affinity_sweep`] | task→layer affinity: per-layer probe accuracy + per-class peak layer over a packed activation bank |
//!
//! # The protocol (and what the arms mean)
//!
//! Project the features through a candidate `k`-subspace and read them
//! through the FROZEN probe head:
//! - **aligned** — the top-`k` right singular vectors of the probe weights:
//!   how much task signal lives in the probe's own dominant subspace;
//! - **random** — a seeded random orthonormal `k`-basis at matched budget.
//!   It does NOT collapse to chance: a random `k`-subspace of `R^d` retains
//!   ≈`k/d` of both signal and noise (an SNR-preserving cut). The
//!   discriminating signal is the **aligned-vs-random contrast**;
//! - **residual** — the orthogonal complement of the aligned basis: the
//!   signal that is NOT in the aligned subspace.
//!
//! Projection identity: aligned@`k = rank(W)` reproduces the full accuracy
//! exactly (pinning that the arms measure subspace content, not head
//! quality).
//!
//! # Cousins, not duplicates (the fit kinship table)
//!
//! Same closed-form ridge, three layouts — see `karc/hebbian_readout.rs`
//! for the established precedent of documenting same-math fits side by
//! side: `KarcForecaster::fit_ridge` (f64 Cholesky, delay-embedding
//! features), `HebbianKernelMemory::construct` (f32 whitened ridge with a
//! per-fact edit story), and this module's probe (labels-in, SVD-of-Gram
//! solve, right vectors only — the SVD doubles as the aligned-basis
//! source). `interpolation_geometry::intervention_battery` perturbs
//! committed latent STATE (additive/noise arms); this module PROJECTS
//! features through probe-derived subspaces — the paper's protocol exactly.
//!
//! # Determinism + allocation posture
//!
//! All randomness flows through the seeded SplitMix64 stream
//! (house FixtureRng pattern) — identical inputs + seed ⇒ bit-identical
//! outputs. Evaluation paths are zero-alloc with caller-owned scratch
//! ([`InterventionScratch`]); fit-time workspaces live in the same scratch
//! (built once, reused across layers/refits).
//!
//! λ is a probe-time diagnostic instrument — freeze/consolidation-seam
//! adjacency, never a per-tick signal.
//!
//! subspace_intervention — the three-arm subspace-intervention protocol
//! (Issue 779 T1 / Research 557 / arXiv:2607.01987): modelless ridge probe +
//! aligned/random/residual projection triad + task→layer affinity sweep,
//! promoted from the Issue-778 POC harness. OPT-IN (no-default-consumer rule).

#![allow(dead_code)]

use crate::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};

// ── Seeded PRNG (house FixtureRng pattern: local SplitMix64 + Box-Muller) ──

/// Deterministic f32 stream behind [`random_basis_into`] — SplitMix64 +
/// Box–Muller; identical seed ⇒ bit-identical draws.
pub struct InterventionRng(u64);

impl InterventionRng {
    /// New stream from `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// One N(0,1) draw (Box–Muller; deterministic under the stream).
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = core::f32::consts::TAU * u2;
        if self.next_u64() & 1 == 0 {
            r * theta.cos()
        } else {
            r * theta.sin()
        }
    }
}

/// Orthonormalize `cols` (`d×k`, column-major flat) in place via
/// Gram–Schmidt.
pub fn gram_schmidt_into(cols: &mut [f32], d: usize) {
    let k = cols.len() / d;
    for j in 0..k {
        for i in 0..j {
            let mut dot = 0.0_f32;
            for r in 0..d {
                dot += cols[j * d + r] * cols[i * d + r];
            }
            for r in 0..d {
                cols[j * d + r] -= dot * cols[i * d + r];
            }
        }
        let mut norm = 0.0_f32;
        for r in 0..d {
            norm += cols[j * d + r] * cols[j * d + r];
        }
        norm = norm.sqrt().max(1e-12);
        for r in 0..d {
            cols[j * d + r] /= norm;
        }
    }
}

/// Fill `basis_out` (`d×k`, column-major flat) with a seeded random
/// orthonormal basis — the protocol's control arm.
pub fn random_basis_into(basis_out: &mut [f32], d: usize, seed: u64) {
    let mut rng = InterventionRng::new(seed);
    for v in basis_out.iter_mut() {
        *v = rng.next_gaussian();
    }
    gram_schmidt_into(basis_out, d);
}

/// Principal-angle basis similarity `(1/k)·‖V_aᵀV_b‖²_F ∈ (0, 1]`
/// (Li et al. 2021 — the paper's Fig-8 metric family): 1.0 = same span,
/// ≈`k/d` = independent random spans. `va`/`vb` are `d×k` column-major.
#[must_use]
pub fn basis_similarity(va: &[f32], vb: &[f32], d: usize) -> f32 {
    let k = va.len() / d;
    let mut fro = 0.0_f32;
    for j in 0..k {
        for i in 0..k {
            let mut dot = 0.0_f32;
            for r in 0..d {
                dot += va[j * d + r] * vb[i * d + r];
            }
            fro += dot * dot;
        }
    }
    fro / k as f32
}

// ── Scratch ────────────────────────────────────────────────────────────────

/// Caller-owned workspaces for the whole protocol. Build once per
/// `(d, classes, n_train, n_test)`, reuse across layers/refits/evals —
/// every function below allocates nothing.
pub struct InterventionScratch {
    gram: Vec<f32>,         // d×d regularized Gram
    xty: Vec<f32>,          // d×c RHS
    basis: Vec<f32>,        // d×rank aligned basis
    control: Vec<f32>,      // d×k random control basis
    z: Vec<f32>,            // n_test×d projected features
    coeffs: Vec<f32>,       // per-sample k projection coefficients
    eval_hits: Vec<usize>,  // per-class hit counters (eval path, no alloc)
    eval_total: Vec<usize>, // per-class totals
    peak_acc: Vec<f32>,     // affinity running per-class best recall
    g_svd_res: SvdResultScratch,
    g_svd_work: SvdScratch,
    w_svd_res: SvdResultScratch,
    w_svd_work: SvdScratch,
}

impl InterventionScratch {
    /// Scratch sized for `d` features, `classes` heads, and `n_test`
    /// evaluation rows (the projection buffer). Build once, reuse across
    /// layers/refits/evals.
    #[must_use]
    pub fn new(d: usize, classes: usize, n_train: usize, n_test: usize) -> Self {
        let _ = n_train; // fit workspaces size from (d, classes) only
        let rank_cap = classes.min(d);
        Self {
            gram: vec![0.0; d * d],
            xty: vec![0.0; d * classes],
            basis: vec![0.0; d * rank_cap],
            control: vec![0.0; d * rank_cap],
            z: vec![0.0; n_test * d],
            coeffs: vec![0.0; rank_cap],
            eval_hits: vec![0; classes],
            eval_total: vec![0; classes],
            peak_acc: vec![f32::NEG_INFINITY; classes],
            g_svd_res: SvdResultScratch::with_capacity(d, d),
            g_svd_work: SvdScratch::with_capacity(d, d),
            w_svd_res: SvdResultScratch::with_capacity(classes, d),
            w_svd_work: SvdScratch::with_capacity(d, classes),
        }
    }

    /// Frozen-head eval through the scratch's own counters — the external
    /// entry point ([`eval_head_into`] needs the counter slices split out
    /// for the internal borrow structure; callers use this method).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn eval_into(
        &mut self,
        x: &[f32],
        y: &[usize],
        n: usize,
        d: usize,
        w: &[f32],
        classes: usize,
        recall_out: &mut [f32],
    ) -> f32 {
        let InterventionScratch {
            eval_hits,
            eval_total,
            ..
        } = self;
        eval_head_into(x, y, n, d, w, classes, recall_out, eval_hits, eval_total)
    }
}

// ── The modelless probe ────────────────────────────────────────────────────

/// Closed-form ridge probe fit: `W (c×d, row-major)` from `x` (`n×d`
/// row-major) + integer labels, `λ = lambda_scale · mean(diag(XᵀX))`,
/// solved in the eigenbasis of the regularized Gram:
/// `Wᵀ = (G+λI)⁻¹·(XᵀY) = Σⱼ vⱼ·(vⱼᵀ·(XᵀY))/σⱼ` (right vectors only —
/// the Gram is SPD so its SVD right basis diagonalizes it).
///
/// Dual fast path (n < d — the common activation-bank shape): solves the
/// mathematically identical `(K+λI)⁻¹Y` system on the n×n kernel
/// `K = XXᵀ` and back-transforms `W = Yᵀ(K+λI)⁻¹X` — the standard ridge
/// identity `(XᵀX+λI)⁻¹XᵀY = Xᵀ(XXᵀ+λI)⁻¹Y` with the SAME λ (trace(K) ≡
/// trace(G)). Strictly cheaper for n < d: n³ + n²·d instead of d³ + d²·n —
/// the Issue-779-T3 bank (n=288, d=2304) makes this ~500× faster. Pinned
/// against the primal path by `ridge_primal_dual_agree` (same W within f32
/// tolerance, identical predictions).
///
/// Design record (Issue 779 T1): the 778 POC harness carried an algebra
/// slip — it formed `G⁺·(XᵀY)` first and then divided by `σⱼ²`, which
/// collapses to the class-sum readout `W = XᵀY` (a serviceable
/// nearest-mean probe, which is why the POC's gates passed) — NOT ridge.
/// This is the true solve; the triad semantics are unchanged.
#[allow(clippy::too_many_arguments)]
pub fn ridge_probe_fit_into(
    x: &[f32],
    y: &[usize],
    n: usize,
    d: usize,
    classes: usize,
    lambda_scale: f32,
    scratch: &mut InterventionScratch,
    w_out: &mut [f32],
) {
    debug_assert_eq!(x.len(), n * d);
    debug_assert_eq!(y.len(), n);
    debug_assert!(w_out.len() >= classes * d);
    if n < d {
        ridge_probe_fit_dual_into(x, y, n, d, classes, lambda_scale, scratch, w_out);
    } else {
        ridge_probe_fit_primal_into(x, y, n, d, classes, lambda_scale, scratch, w_out);
    }
}

/// The n≥d primal of [`ridge_probe_fit_into`] — d×d Gram eigen-solve.
#[allow(clippy::too_many_arguments)]
fn ridge_probe_fit_primal_into(
    x: &[f32],
    y: &[usize],
    n: usize,
    d: usize,
    classes: usize,
    lambda_scale: f32,
    scratch: &mut InterventionScratch,
    w_out: &mut [f32],
) {
    let g = &mut scratch.gram[..d * d];
    g.fill(0.0);
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        for a in 0..d {
            let xa = row[a];
            for b in 0..d {
                g[a * d + b] += xa * row[b];
            }
        }
    }
    let trace: f32 = (0..d).map(|a| g[a * d + a]).sum();
    let lambda = lambda_scale * trace / d as f32;
    for a in 0..d {
        g[a * d + a] += lambda;
    }
    let m = &mut scratch.xty[..d * classes];
    m.fill(0.0);
    for i in 0..n {
        let ci = y[i];
        for a in 0..d {
            m[a * classes + ci] += x[i * d + a];
        }
    }
    thin_svd_into(g, d, d, &mut scratch.g_svd_res, &mut scratch.g_svd_work);
    let len = scratch.g_svd_res.len();
    let sigma_max = scratch.g_svd_res.singular_value(0).max(1e-12);
    for wv in w_out[..classes * d].iter_mut() {
        *wv = 0.0;
    }
    for j in 0..len {
        let vj = scratch.g_svd_res.right_singular_vector(j);
        // true solve: divide by σⱼ ONCE (Wᵀ = Σ vⱼ·(vⱼᵀM)/σⱼ)
        let inv_s = 1.0 / scratch.g_svd_res.singular_value(j).max(1e-9 * sigma_max);
        for c in 0..classes {
            let mut proj = 0.0_f32;
            for a in 0..d {
                proj += vj[a] * m[a * classes + c];
            }
            let s = proj * inv_s;
            if s == 0.0 {
                continue;
            }
            for a in 0..d {
                w_out[c * d + a] += vj[a] * s;
            }
        }
    }
}

/// The n<d dual of [`ridge_probe_fit_into`] — same `w_out` contract, same λ
/// formula (divided by `d`, matching the primal), n×n kernel factorization.
/// Uses the `d`-sized scratch buffers' prefixes (n·n ≤ d·d, n·c ≤ d·c).
#[allow(clippy::too_many_arguments)]
fn ridge_probe_fit_dual_into(
    x: &[f32],
    y: &[usize],
    n: usize,
    d: usize,
    classes: usize,
    lambda_scale: f32,
    scratch: &mut InterventionScratch,
    w_out: &mut [f32],
) {
    debug_assert!(n < d);
    // K = X·Xᵀ + λI (n×n, symmetric PSD)
    let k = &mut scratch.gram[..n * n];
    k.fill(0.0);
    for i in 0..n {
        let ri = &x[i * d..(i + 1) * d];
        for j in 0..n {
            let rj = &x[j * d..(j + 1) * d];
            let mut s = 0.0_f32;
            for a in 0..d {
                s += ri[a] * rj[a];
            }
            k[i * n + j] = s;
        }
    }
    // Same λ as the primal: λ_scale · trace(G)/d, and trace(K) ≡ trace(G).
    let trace: f32 = (0..n).map(|i| k[i * n + i]).sum();
    let lambda = lambda_scale * trace / d as f32;
    for i in 0..n {
        k[i * n + i] += lambda;
    }
    // One-hot RHS (n×c), row-major per-sample
    let yh = &mut scratch.xty[..n * classes];
    yh.fill(0.0);
    for i in 0..n {
        yh[i * classes + y[i]] = 1.0;
    }
    // α = (K+λI)⁻¹·Y = Σⱼ vⱼ·(vⱼᵀY)/σⱼ (K+λI is SPD — same one-sided trick)
    let mut alpha = vec![0.0_f32; n * classes];
    thin_svd_into(k, n, n, &mut scratch.g_svd_res, &mut scratch.g_svd_work);
    let len = scratch.g_svd_res.len();
    let sigma_max = scratch.g_svd_res.singular_value(0).max(1e-12);
    for j in 0..len {
        let vj = scratch.g_svd_res.right_singular_vector(j);
        let inv_s = 1.0 / scratch.g_svd_res.singular_value(j).max(1e-9 * sigma_max);
        for c in 0..classes {
            let mut proj = 0.0_f32;
            for i in 0..n {
                proj += vj[i] * yh[i * classes + c];
            }
            let s = proj * inv_s;
            if s == 0.0 {
                continue;
            }
            for i in 0..n {
                alpha[i * classes + c] += vj[i] * s;
            }
        }
    }
    // W = αᵀ·X (c×d row-major)
    for wv in w_out[..classes * d].iter_mut() {
        *wv = 0.0;
    }
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        for c in 0..classes {
            let a_ic = alpha[i * classes + c];
            if a_ic == 0.0 {
                continue;
            }
            for a in 0..d {
                w_out[c * d + a] += a_ic * row[a];
            }
        }
    }
}

/// Frozen-head readout: argmax accuracy of `x` (`n×d`) through `w`
/// (`c×d`); per-class recall into `recall_out` (len `classes`). The
/// per-class counters are caller-passed (the scratch fields
/// `eval_hits`/`eval_total`) so the eval path allocates nothing —
/// field-level borrows keep it borrow-checker-clean at the call sites.
#[allow(clippy::too_many_arguments)]
pub fn eval_head_into(
    x: &[f32],
    y: &[usize],
    n: usize,
    d: usize,
    w: &[f32],
    classes: usize,
    recall_out: &mut [f32],
    hits: &mut [usize],
    total: &mut [usize],
) -> f32 {
    debug_assert_eq!(x.len(), n * d);
    debug_assert!(recall_out.len() >= classes);
    debug_assert!(hits.len() >= classes);
    debug_assert!(total.len() >= classes);
    hits[..classes].fill(0);
    total[..classes].fill(0);
    let mut correct = 0usize;
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let mut best_c = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for c in 0..classes {
            let mut dot = 0.0_f32;
            for (a, &xa) in row.iter().enumerate() {
                dot += xa * w[c * d + a];
            }
            if dot > best_v {
                best_v = dot;
                best_c = c;
            }
        }
        total[y[i]] += 1;
        if best_c == y[i] {
            hits[y[i]] += 1;
            correct += 1;
        }
    }
    for c in 0..classes {
        recall_out[c] = if total[c] == 0 {
            0.0
        } else {
            hits[c] as f32 / total[c] as f32
        };
    }
    correct as f32 / n as f32
}

/// Project `x` (`n×d` row-major) through a `d×k` orthonormal basis
/// (column-major flat, `basis[j·d + a]`): `Ẑ = Z·B·Bᵀ`; `complement = true`
/// → `Ẑ = Z − Z·B·Bᵀ`. Writes `n·d` values into `z_out`.
#[allow(clippy::too_many_arguments)]
pub fn project_through_into(
    x: &[f32],
    z_out: &mut [f32],
    basis: &[f32],
    n: usize,
    d: usize,
    k: usize,
    complement: bool,
    coeffs: &mut [f32],
) {
    debug_assert_eq!(x.len(), n * d);
    debug_assert!(z_out.len() >= n * d);
    debug_assert!(basis.len() >= d * k);
    debug_assert!(coeffs.len() >= k);
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        for j in 0..k {
            let mut dot = 0.0_f32;
            for (a, &xa) in row.iter().enumerate() {
                dot += xa * basis[j * d + a];
            }
            coeffs[j] = dot;
        }
        for a in 0..d {
            let mut acc = if complement { row[a] } else { 0.0 };
            let sign = if complement { -1.0 } else { 1.0 };
            for j in 0..k {
                acc += sign * coeffs[j] * basis[j * d + a];
            }
            z_out[i * d + a] = acc;
        }
    }
}

/// The three-arm intervention against a CALLER-SUPPLIED aligned basis
/// (`d×rank_cap`, column-major, orthonormal — e.g. a calibrated
/// eigenbasis, the FUNCATTN/spectral_pre_rotate arm): for each `k` in
/// `ks`, the aligned arm projects through the first `k` columns, the
/// random arm through a seeded control at matched `k`, the residual arm
/// through the complement of the first `k` columns — all read through
/// the frozen head `w`. Returns the usable rank (`rank_cap`).
#[allow(clippy::too_many_arguments)]
pub fn three_arm_eval_on_basis(
    x_test: &[f32],
    y: &[usize],
    n: usize,
    d: usize,
    w: &[f32],
    classes: usize,
    ks: &[usize],
    basis: &[f32],
    rank_cap: usize,
    seed: u64,
    scratch: &mut InterventionScratch,
    aligned_out: &mut [f32],
    random_out: &mut [f32],
    residual_out: &mut [f32],
    recall_buf: &mut [f32],
) -> usize {
    debug_assert_eq!(x_test.len(), n * d);
    debug_assert!(basis.len() >= d * rank_cap);
    let InterventionScratch {
        z,
        coeffs,
        control,
        eval_hits,
        eval_total,
        ..
    } = scratch;
    for (ki, &k) in ks.iter().enumerate() {
        let k = k.min(rank_cap);
        // aligned (caller basis)
        project_through_into(
            x_test,
            &mut z[..n * d],
            &basis[..d * k],
            n,
            d,
            k,
            false,
            coeffs,
        );
        aligned_out[ki] = eval_head_into(
            &z[..n * d],
            y,
            n,
            d,
            w,
            classes,
            recall_buf,
            eval_hits,
            eval_total,
        );
        // random control (seed varies with k: independent arms)
        random_basis_into(
            &mut control[..d * k],
            d,
            seed ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        project_through_into(
            x_test,
            &mut z[..n * d],
            &control[..d * k],
            n,
            d,
            k,
            false,
            coeffs,
        );
        random_out[ki] = eval_head_into(
            &z[..n * d],
            y,
            n,
            d,
            w,
            classes,
            recall_buf,
            eval_hits,
            eval_total,
        );
        // residual (complement of the aligned basis)
        project_through_into(
            x_test,
            &mut z[..n * d],
            &basis[..d * k],
            n,
            d,
            k,
            true,
            coeffs,
        );
        residual_out[ki] = eval_head_into(
            &z[..n * d],
            y,
            n,
            d,
            w,
            classes,
            recall_buf,
            eval_hits,
            eval_total,
        );
    }
    rank_cap
}

/// The three-arm intervention at one bank, through the frozen probe head
/// `w` (`c×d`): for each candidate `k` in `ks`, write
/// `aligned_out[ki]` / `random_out[ki]` / `residual_out[ki]` = test
/// accuracy of the projected features. Returns the probe rank
/// (`min(c, d)`, the SVD length of `w` — the `k` at which the projection
/// identity holds). The aligned basis is the top-`k` right singular
/// vectors of `w`; the control is a seeded random basis at matched `k`.
///
/// This is the probe-basis specialization of [`three_arm_eval_on_basis`]
/// (the protocol's aligned arm = the probe's own dominant subspace).
#[allow(clippy::too_many_arguments)]
pub fn three_arm_eval(
    x_test: &[f32],
    y: &[usize],
    n: usize,
    d: usize,
    w: &[f32],
    classes: usize,
    ks: &[usize],
    seed: u64,
    scratch: &mut InterventionScratch,
    aligned_out: &mut [f32],
    random_out: &mut [f32],
    residual_out: &mut [f32],
    recall_buf: &mut [f32],
) -> usize {
    debug_assert_eq!(x_test.len(), n * d);
    let rank_cap = classes.min(d);
    thin_svd_into(
        &w[..classes * d],
        classes,
        d,
        &mut scratch.w_svd_res,
        &mut scratch.w_svd_work,
    );
    let rank = scratch.w_svd_res.len().min(rank_cap);
    let basis = &mut scratch.basis[..d * rank];
    for j in 0..rank {
        basis[j * d..(j + 1) * d].copy_from_slice(scratch.w_svd_res.right_singular_vector(j));
    }
    // NOTE: inlined (not delegated to three_arm_eval_on_basis) so the
    // aligned basis stays in scratch — the whole call is zero-alloc (G4).
    let InterventionScratch {
        z,
        coeffs,
        control,
        eval_hits,
        eval_total,
        ..
    } = scratch;
    for (ki, &k) in ks.iter().enumerate() {
        let k = k.min(rank);
        // aligned
        project_through_into(x_test, &mut z[..n * d], basis, n, d, k, false, coeffs);
        aligned_out[ki] = eval_head_into(
            &z[..n * d],
            y,
            n,
            d,
            w,
            classes,
            recall_buf,
            eval_hits,
            eval_total,
        );
        // random control (seed varies with k: independent arms)
        random_basis_into(
            &mut control[..d * k],
            d,
            seed ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        project_through_into(
            x_test,
            &mut z[..n * d],
            &control[..d * k],
            n,
            d,
            k,
            false,
            coeffs,
        );
        random_out[ki] = eval_head_into(
            &z[..n * d],
            y,
            n,
            d,
            w,
            classes,
            recall_buf,
            eval_hits,
            eval_total,
        );
        // residual (complement of the aligned basis)
        project_through_into(x_test, &mut z[..n * d], basis, n, d, k, true, coeffs);
        residual_out[ki] = eval_head_into(
            &z[..n * d],
            y,
            n,
            d,
            w,
            classes,
            recall_buf,
            eval_hits,
            eval_total,
        );
    }
    rank
}

/// Task→layer affinity sweep over a packed bank `acts` (`n_layers ×
/// n_total × d`, layer-major row-major) with shared `labels`: per layer,
/// fit the probe on the gathered train rows and evaluate on the test rows.
/// Writes `full_acc_out[l]`, `recall_out[l·classes + c]`, `peak_layer_out[c]`
/// (per-class argmax layer) and returns the best layer (argmax
/// `full_acc`). `train_idx`/`test_idx` select rows per layer.
///
/// Allocation note (honest contract): the sweep allocates its gather rows
/// and probe weights ONCE per call (reused across layers); the per-layer
/// fit/eval work is scratch-owned and alloc-free.
#[allow(clippy::too_many_arguments)]
pub fn affinity_sweep(
    acts: &[f32],
    labels: &[usize],
    train_idx: &[usize],
    test_idx: &[usize],
    n_total: usize,
    d: usize,
    classes: usize,
    n_layers: usize,
    lambda_scale: f32,
    scratch: &mut InterventionScratch,
    full_acc_out: &mut [f32],
    recall_out: &mut [f32],
    peak_layer_out: &mut [usize],
    layer_recall_buf: &mut [f32],
) -> usize {
    let n_train = train_idx.len();
    let n_test = test_idx.len();
    debug_assert_eq!(acts.len(), n_layers * n_total * d);
    let y_train: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();
    let y_test: Vec<usize> = test_idx.iter().map(|&i| labels[i]).collect();
    let mut x_train = vec![0.0_f32; n_train * d];
    let mut x_test = vec![0.0_f32; n_test * d];
    let mut w = vec![0.0_f32; classes * d];
    scratch.peak_acc[..classes].fill(f32::NEG_INFINITY);
    for l in 0..n_layers {
        let layer = &acts[l * n_total * d..(l + 1) * n_total * d];
        for (r, &i) in train_idx.iter().enumerate() {
            x_train[r * d..(r + 1) * d].copy_from_slice(&layer[i * d..(i + 1) * d]);
        }
        for (r, &i) in test_idx.iter().enumerate() {
            x_test[r * d..(r + 1) * d].copy_from_slice(&layer[i * d..(i + 1) * d]);
        }
        ridge_probe_fit_into(
            &x_train,
            &y_train,
            n_train,
            d,
            classes,
            lambda_scale,
            scratch,
            &mut w,
        );
        full_acc_out[l] = eval_head_into(
            &x_test,
            &y_test,
            n_test,
            d,
            &w,
            classes,
            layer_recall_buf,
            &mut scratch.eval_hits,
            &mut scratch.eval_total,
        );
        recall_out[l * classes..(l + 1) * classes].copy_from_slice(layer_recall_buf);
        for c in 0..classes {
            if scratch.peak_acc[c] < layer_recall_buf[c] {
                scratch.peak_acc[c] = layer_recall_buf[c];
                peak_layer_out[c] = l;
            }
        }
    }
    (0..n_layers)
        .max_by(|&a, &b| full_acc_out[a].total_cmp(&full_acc_out[b]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 24;
    const C: usize = 4;
    const N_TRAIN: usize = 72; // parity split of N_TOTAL
    const N_TEST: usize = 72;
    const N_TOTAL: usize = N_TRAIN + N_TEST;

    /// Planted low-rank bank: class means on a low-dim shared subspace +
    /// isotropic noise — the minimal geometry where the aligned subspace
    /// must beat random. Single layer (the triad needs no layer stack).
    fn planted_bank(seed: u64, signal: f32) -> (Vec<f32>, Vec<usize>) {
        let mut rng = InterventionRng::new(seed);
        let mut acts = vec![0.0_f32; N_TOTAL * D];
        let mut labels = Vec::with_capacity(N_TOTAL);
        let means: Vec<Vec<f32>> = (0..C)
            .map(|c| {
                let mut v = vec![0.0_f32; D];
                v[c] = signal; // orthonormal one-hot means: rank-2 task subspace
                v
            })
            .collect();
        for i in 0..N_TOTAL {
            // (i/2) % C: the parity train/test split sees every class in
            // both halves (i%C would alias with the split — the first
            // draft's chance-level trap, kept as this note).
            let c = (i / 2) % C;
            labels.push(c);
            for a in 0..D {
                acts[i * D + a] = means[c][a] + 0.25 * rng.next_gaussian();
            }
        }
        (acts, labels)
    }

    /// Gather train/test rows by parity; returns (x_train, x_test, y_train, y_test).
    fn split(acts: &[f32], labels: &[usize]) -> (Vec<f32>, Vec<f32>, Vec<usize>, Vec<usize>) {
        let mut xt = Vec::with_capacity(N_TRAIN * D);
        let mut xe = Vec::with_capacity(N_TEST * D);
        let mut yt = Vec::with_capacity(N_TRAIN);
        let mut ye = Vec::with_capacity(N_TEST);
        for i in 0..N_TOTAL {
            if i % 2 == 1 {
                xt.extend_from_slice(&acts[i * D..(i + 1) * D]);
                yt.push(labels[i]);
            } else {
                xe.extend_from_slice(&acts[i * D..(i + 1) * D]);
                ye.push(labels[i]);
            }
        }
        (xt, xe, yt, ye)
    }

    fn fit_probe(xt: &[f32], yt: &[usize], scratch: &mut InterventionScratch) -> Vec<f32> {
        let mut w = vec![0.0_f32; C * D];
        ridge_probe_fit_into(xt, yt, N_TRAIN, D, C, 0.01, scratch, &mut w);
        w
    }

    #[test]
    fn probe_sanity_two_class_separable() {
        // Minimal sanity: means ±1 on coord 0, no noise ⇒ accuracy 1.0.
        // (n=20 < D=24 — this also exercises the dual dispatch.)
        let x: Vec<f32> = (0..20)
            .flat_map(|i| {
                let mut v = vec![0.0_f32; D];
                v[0] = if i % 2 == 0 { -1.0 } else { 1.0 };
                v
            })
            .collect();
        let y: Vec<usize> = (0..20).map(|i| i % 2).collect();
        let mut scratch = InterventionScratch::new(D, 2, 20, 20);
        let mut w = vec![0.0_f32; 2 * D];
        ridge_probe_fit_into(&x, &y, 20, D, 2, 0.01, &mut scratch, &mut w);
        let mut recall = vec![0.0_f32; 2];
        let acc = eval_head_into(
            &x,
            &y,
            20,
            D,
            &w,
            2,
            &mut recall,
            &mut scratch.eval_hits,
            &mut scratch.eval_total,
        );
        assert!(acc > 0.99, "sanity acc {acc}");
    }

    /// The n<d dual must reproduce the n≥d primal EXACTLY (mathematically)
    /// on the same data — the Issue-779-T3 bank shape (n=288 < d=2304) rides
    /// this dispatch, so the identity is pinned, not assumed: same W within
    /// f32 tolerance, identical predictions on a fresh eval set.
    #[test]
    fn ridge_primal_dual_agree() {
        let (n, d, c) = (16_usize, 24, 4); // n < d
        let mut rng = InterventionRng::new(99);
        // Class-conditional mean shifts (a linear probe reads MEANS — the
        // R557 fixture lesson) + noise.
        let mut x = vec![0.0_f32; n * d];
        let y: Vec<usize> = (0..n).map(|i| (i / 2) % c).collect();
        for i in 0..n {
            for a in 0..d {
                let mean = if a % c == y[i] { 1.5 } else { 0.0 };
                x[i * d + a] = mean + 0.2 * rng.next_gaussian();
            }
        }
        let mut eval_x = vec![0.0_f32; n * d];
        let eval_y: Vec<usize> = (0..n).map(|i| (i / 2) % c).collect();
        for i in 0..n {
            for a in 0..d {
                let mean = if a % c == eval_y[i] { 1.5 } else { 0.0 };
                eval_x[i * d + a] = mean + 0.2 * rng.next_gaussian();
            }
        }
        let mut scratch = InterventionScratch::new(d, c, n, n);
        let mut w_primal = vec![0.0_f32; c * d];
        let mut w_dual = vec![0.0_f32; c * d];
        ridge_probe_fit_primal_into(&x, &y, n, d, c, 0.01, &mut scratch, &mut w_primal);
        ridge_probe_fit_dual_into(&x, &y, n, d, c, 0.01, &mut scratch, &mut w_dual);
        let scale = w_primal
            .iter()
            .fold(0.0_f32, |m, v| m.max(v.abs()))
            .max(1e-12);
        let max_delta = w_primal
            .iter()
            .zip(&w_dual)
            .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            max_delta < 1e-3 * scale,
            "primal/dual W disagree: max_delta {max_delta} vs scale {scale}"
        );
        let mut recall_p = vec![0.0_f32; c];
        let mut recall_d = vec![0.0_f32; c];
        let acc_p = eval_head_into(
            &eval_x,
            &eval_y,
            n,
            d,
            &w_primal,
            c,
            &mut recall_p,
            &mut scratch.eval_hits,
            &mut scratch.eval_total,
        );
        let acc_d = eval_head_into(
            &eval_x,
            &eval_y,
            n,
            d,
            &w_dual,
            c,
            &mut recall_d,
            &mut scratch.eval_hits,
            &mut scratch.eval_total,
        );
        assert!(
            (acc_p - acc_d).abs() <= 1.0 / n as f32,
            "same W to f32 tolerance must not move more than a borderline sample: {acc_p} vs {acc_d}"
        );
        assert!(
            acc_p > 0.9 && acc_d > 0.9,
            "the fixture must be separable ({acc_p}/{acc_d})"
        );
    }

    /// G1(a) — the projection identity: aligned@k=rank reproduces the full
    /// accuracy exactly (the arms measure subspace content, not head
    /// quality).
    #[test]
    fn projection_identity_aligned_at_rank_equals_full() {
        let (acts, labels) = planted_bank(7, 2.0);
        let (xt, xe, yt, ye) = split(&acts, &labels);
        let mut scratch = InterventionScratch::new(D, C, N_TRAIN, N_TEST);
        let w = fit_probe(&xt, &yt, &mut scratch);
        let mut recall = vec![0.0_f32; C];
        let full = eval_head_into(
            &xe,
            &ye,
            N_TEST,
            D,
            &w,
            C,
            &mut recall,
            &mut scratch.eval_hits,
            &mut scratch.eval_total,
        );
        let ks = [1_usize, 2, 4];
        let mut al = vec![0.0_f32; 3];
        let mut ra = vec![0.0_f32; 3];
        let mut re = vec![0.0_f32; 3];
        let rank = three_arm_eval(
            &xe,
            &ye,
            N_TEST,
            D,
            &w,
            C,
            &ks,
            0xDEAD_BEEF,
            &mut scratch,
            &mut al,
            &mut ra,
            &mut re,
            &mut recall,
        );
        assert_eq!(rank, C.min(D));
        let ki_full = ks.iter().position(|&k| k >= rank).expect("some k >= rank");
        assert_eq!(
            al[ki_full], full,
            "aligned@k=rank must reproduce full exactly: {} vs {full}",
            al[ki_full]
        );
    }

    /// G1(b,c) — the aligned-vs-random contrast and the residual collapse:
    /// at the task rank the residual arm falls to chance and aligned beats
    /// random decisively (the planted geometry is rank-2).
    #[test]
    fn aligned_beats_random_and_residual_collapses() {
        let (acts, labels) = planted_bank(7, 2.0);
        let (xt, xe, yt, ye) = split(&acts, &labels);
        let mut scratch = InterventionScratch::new(D, C, N_TRAIN, N_TEST);
        let w = fit_probe(&xt, &yt, &mut scratch);
        let mut recall = vec![0.0_f32; C];
        let ks = [2_usize, 4];
        let mut al = vec![0.0_f32; 2];
        let mut ra = vec![0.0_f32; 2];
        let mut re = vec![0.0_f32; 2];
        three_arm_eval(
            &xe,
            &ye,
            N_TEST,
            D,
            &w,
            C,
            &ks,
            0xDEAD_BEEF,
            &mut scratch,
            &mut al,
            &mut ra,
            &mut re,
            &mut recall,
        );
        assert!(
            al[1] >= ra[1] + 0.15,
            "aligned@k=4 must beat random decisively: {:.3} vs {:.3}",
            al[1],
            ra[1]
        );
        assert!(
            re[1] <= 1.0 / C as f32 + 0.05,
            "residual@k=rank must collapse to chance: {:.3}",
            re[1]
        );
    }

    /// Affinity sweep on a two-layer bank (task at layer 1, noise at layer
    /// 0) — the sweep must pick layer 1.
    #[test]
    fn affinity_sweep_finds_the_task_layer() {
        let (acts_task, labels) = planted_bank(11, 2.0);
        let mut rng = InterventionRng::new(99);
        let mut noise_layer = vec![0.0_f32; N_TOTAL * D];
        for v in noise_layer.iter_mut() {
            *v = 0.25 * rng.next_gaussian();
        }
        let mut packed = vec![0.0_f32; 2 * N_TOTAL * D];
        packed[..N_TOTAL * D].copy_from_slice(&noise_layer);
        packed[N_TOTAL * D..].copy_from_slice(&acts_task);
        let train: Vec<usize> = (0..N_TOTAL).filter(|&i| i % 2 == 1).collect();
        let test: Vec<usize> = (0..N_TOTAL).filter(|&i| i % 2 == 0).collect();
        let mut scratch = InterventionScratch::new(D, C, N_TRAIN, N_TEST);
        let mut full_acc = vec![0.0_f32; 2];
        let mut recall = vec![0.0_f32; 2 * C];
        let mut peaks = vec![0usize; C];
        let mut layer_buf = vec![0.0_f32; C];
        let best = affinity_sweep(
            &packed,
            &labels,
            &train,
            &test,
            N_TOTAL,
            D,
            C,
            2,
            0.01,
            &mut scratch,
            &mut full_acc,
            &mut recall,
            &mut peaks,
            &mut layer_buf,
        );
        assert_eq!(best, 1, "the sweep must pick the task layer");
        assert!(peaks.iter().all(|&p| p == 1));
    }

    /// `basis_similarity`: identical spans → 1.0; independent random spans
    /// → ≈ k/d (the random-random floor).
    #[test]
    fn basis_similarity_bounds() {
        let mut ba = vec![0.0_f32; D * 4];
        let mut bb = vec![0.0_f32; D * 4];
        random_basis_into(&mut ba, D, 1);
        bb.copy_from_slice(&ba);
        let same = basis_similarity(&ba, &bb, D);
        assert!(
            (same - 1.0).abs() < 1e-5,
            "identical spans ⇒ 1.0, got {same}"
        );
        random_basis_into(&mut bb, D, 2);
        let indep = basis_similarity(&ba, &bb, D);
        assert!(
            indep < 0.5,
            "independent random 4-spans in d=24 ⇒ well under 0.5, got {indep:.3}"
        );
    }

    /// `random_basis_into` orthonormality (Gram–Schmidt contract).
    #[test]
    fn random_basis_is_orthonormal() {
        let mut b = vec![0.0_f32; D * 5];
        random_basis_into(&mut b, D, 42);
        for j in 0..5 {
            for i in 0..=j {
                let mut dot = 0.0_f32;
                for r in 0..D {
                    dot += b[j * D + r] * b[i * D + r];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-4, "({i},{j}) dot {dot}");
            }
        }
    }

    /// G4 — alloc-free eval paths (scratch + outs pre-built; the counting
    /// allocator must observe zero). The Issue-741 predicate.
    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn g4_alloc_free_eval_paths() {
        let (acts, labels) = planted_bank(7, 2.0);
        let (xt, xe, yt, ye) = split(&acts, &labels);
        let mut scratch = InterventionScratch::new(D, C, N_TRAIN, N_TEST);
        let mut w = vec![0.0_f32; C * D];
        ridge_probe_fit_into(&xt, &yt, N_TRAIN, D, C, 0.01, &mut scratch, &mut w);
        let mut recall = vec![0.0_f32; C];
        let ks = [2_usize, 4];
        let mut al = vec![0.0_f32; 2];
        let mut ra = vec![0.0_f32; 2];
        let mut re = vec![0.0_f32; 2];
        crate::alloc::reset_alloc_stats();
        let rank = three_arm_eval(
            &xe,
            &ye,
            N_TEST,
            D,
            &w,
            C,
            &ks,
            3,
            &mut scratch,
            &mut al,
            &mut ra,
            &mut re,
            &mut recall,
        );
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert!(rank > 0);
        assert_eq!(count, 0, "three_arm_eval must be alloc-free");
    }
}
