//! sst_cross_tail — the cross-tail interference instrument (Issue 972;
//! katgpt-rs Research 639 addendum, arXiv:2610.31818 "Nestor" Prop 2/5).
//!
//! # The gap this prices
//!
//! [`crate::sst_certified`] certifies one delta against its BASE
//! (`anatomy` / `certify_decoded_tail`) and bounds same-anchor CHAINS
//! additively (`compose_tail`). What nothing measured is delta-vs-DELTA
//! interference inside the free tail: two tail-confined overlays can both
//! pass certification and still collide with each other's function. Nestor's
//! forgetting proposition: after adapting on a new domain, forgetting on old
//! labels in the flat predictor flows through the shared value-map row
//! space, with a leading coefficient scaling like the principal-angle
//! overlap (cos²θ) between the old/new row spaces. This module computes
//! that overlap for two same-anchor tail factors — on the FACTORED PAYLOADS,
//! never ambient m×n matrices.
//!
//! # Why the payloads suffice (the exact-isometry fact)
//!
//! `compose_tail` only ever chains factors sharing the SAME anchor, hence
//! bit-identical orthonormal tail bases. For `Δ = U C Vᵀ` with `VᵀV = I`:
//! `row(Δ) = row(C)·Vᵀ` and `z ↦ zVᵀ` is an isometry, so the deltas'
//! input-side principal angles equal those between `row(C₁)` and `row(C₂)`
//! — computed on the small `f_left × f_right` payloads. Dually, output-side
//! angles equal those between the payload column spaces (via `U`).
//!
//! # The trivial-angles hazard (shapes the output)
//!
//! A typical dense full-rank payload spans all of `R^f`, so PLAIN subspace
//! angles between two payloads are trivially one (cos² = 1 everywhere)
//! whatever the deltas "mean". The PRIMARY outputs are therefore per-side
//! basis-free alignment SCALARS in uncentered-linear-CKA form over the
//! side's Grams — 1.0 exactly when `G₁ ∝ G₂`, 0 exactly for Gram-orthogonal
//! pairs, in [0,1] by Cauchy–Schwarz, and still separating two different
//! full-rank payloads (the full-rank guard arm in the gate battery). The
//! cos² angle SPECTRUM is a secondary report, computed on a
//! σ-truncated eigenbasis with the cutoff recorded beside it.
//!
//! # Report-only API law (T5)
//!
//! The instrument PRICES, never refuses — beyond the input guards that
//! mirror `compose_tail_into`'s compose-refusal classes (anchor digest,
//! shapes, bases, finiteness, zero payload). No gate behavior ships here;
//! the consumer posture (`SstAdmissionReport` cross-tail row) files in
//! riir-ai referencing Issue 972 and stays until an owner-gated promotion.
//!
//! # Envelope
//!
//! f64 Gram accumulation with sequential folds; eigenbases from
//! [`crate::karc::jacobi_eigen`] with sorted (stable, `total_cmp`)
//! eigenpairs and sign-canonicalized columns; zero-alloc over a
//! caller-owned [`CrossTailScratch`] after a single warm call (the
//! `SstScratch` posture — this module keeps its own scratch because
//! `SstScratch`'s buffers are private to `sst_certified`). Deterministic
//! under fixed input. Modelless — no training, no gradient descent.
//! Opt-in under the module's own `sst_project` feature (no new flag).

use crate::karc::jacobi_eigen;
use crate::sst_certified::{SstError, TailFactor};

/// σ-relative floor for the truncated spectral bases — the same semantic
/// (and value) as `sst_certified`'s direction floor: a singular direction
/// enters a truncated basis only above this fraction of the payload's
/// largest singular value. Recorded in every report, never implicit.
const SIGMA_REL_FLOOR: f64 = 1e-6;

/// Jacobi sweeps upper bound (the solver converges in 5–10; the bound is a
/// safety ceiling — the same posture as `sst_certified`).
const JACOBI_MAX_SWEEPS: usize = 60;

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// The cross-tail interference report (Issue 972 T1). Both sides are
/// carried SEPARATELY, never pooled (the `anatomy` two-positive-terms
/// discipline): a merge can collide through `V_⊥` (input side) or through
/// `U_⊥` (output side), and the two alignments are different facts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CrossTailReport {
    /// Input-side alignment scalar (the paper's value-map row-space side):
    /// uncentered linear CKA over the input-side Grams `Gᵢ = CᵢᵀCᵢ` —
    /// `⟨G₁,G₂⟩_F / (‖G₁‖_F·‖G₂‖_F)`, clamped to [0,1]. Basis-free; no
    /// upper-bound guarantee on the measured projection fraction (the
    /// rank-1-vs-identity counterexample, Issue 972 T3) — adjudicate this
    /// scalar on RANKING and separation only.
    pub input_scalar: f64,
    /// Output-side alignment scalar: the same CKA form over the output-side
    /// Grams `Hᵢ = CᵢCᵢᵀ`, clamped to [0,1]. Same ranking-only discipline.
    pub output_scalar: f64,
    /// Input-side cos² principal-angle spectrum (descending), computed on
    /// the σ-truncated ROW-space eigenbases of the two payloads. The HARD
    /// bound lives here: the measured projection-energy fraction of Δ₁
    /// inside Δ₂'s truncated row space is ≤ `input_max_cos2()` plus Δ₁'s
    /// own below-floor energy fraction (the truncation term).
    pub input_spectrum: Vec<f64>,
    /// Output-side cos² spectrum (descending) over the truncated
    /// COLUMN-space eigenbases.
    pub output_spectrum: Vec<f64>,
    /// The recorded σ-relative cutoff (a singular direction enters a
    /// truncated basis only above `sigma_rel_floor · σ_max`). Consumers
    /// rebuilding `P_row` for the bound must use the SAME cutoff.
    pub sigma_rel_floor: f64,
    /// Per-delta energy fraction BELOW the input-side cutoff — `[Δ₁, Δ₂]`.
    /// A payload carrying sub-floor energy relaxes the hard bound by its
    /// own fraction (T3's ε term).
    pub input_energy_below_floor: [f64; 2],
    /// The output-side dual of `input_energy_below_floor`.
    pub output_energy_below_floor: [f64; 2],
}

impl CrossTailReport {
    /// `max cos²` of the input-side spectrum (0.0 when empty).
    #[must_use]
    pub fn input_max_cos2(&self) -> f64 {
        self.input_spectrum.first().copied().unwrap_or(0.0)
    }

    /// `max cos²` of the output-side spectrum (0.0 when empty).
    #[must_use]
    pub fn output_max_cos2(&self) -> f64 {
        self.output_spectrum.first().copied().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Scratch
// ---------------------------------------------------------------------------

/// Caller-owned scratch for the instrument (the `SstScratch` posture:
/// grow-on-demand — after one warm call at the working shape, all later
/// calls allocate ZERO).
#[derive(Default)]
pub struct CrossTailScratch {
    /// Per-side Grams (`G = CᵀC` input-side, `H = CCᵀ` output-side).
    g1: Vec<f64>,
    g2: Vec<f64>,
    eigvals: Vec<f64>,
    eigvecs: Vec<f64>,
    jac: Vec<f64>,
    /// Payload one's sorted + sign-canonicalized eigenbasis (`dim × dim`;
    /// the first `kept` columns are the truncated basis).
    basis: Vec<f64>,
    /// Payload two's eigenbasis.
    basis2: Vec<f64>,
    /// The cross-Gram `M = Q₁ᵀQ₂` (`k₁ × k₂`).
    cross: Vec<f64>,
    /// The Gram of M (`k₁ × k₁`) — the jacobi input for the spectrum.
    mt: Vec<f64>,
    order: Vec<usize>,
}

fn ensure64(v: &mut Vec<f64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// Sort eigenpairs descending (stable, deterministic), gather the
/// permuted columns, canonicalize signs (largest-|entry| positive).
/// Mirrors the `fit_anchor_into` posture so every derived basis is
/// reproducible across runs and processes.
fn sorted_canonical_basis(
    eigvals: &[f64],
    eigvecs: &[f64],
    basis: &mut Vec<f64>,
    order: &mut Vec<usize>,
    dim: usize,
) {
    order.clear();
    order.extend(0..dim);
    order.sort_by(|&x, &y| eigvals[y].total_cmp(&eigvals[x]));
    ensure64(basis, dim * dim);
    {
        let dst = &mut basis[..dim * dim];
        for row in 0..dim {
            for t in 0..dim {
                dst[row * dim + t] = eigvecs[row * dim + order[t]];
            }
        }
        for t in 0..dim {
            let mut best = 0usize;
            let mut best_abs = 0.0f64;
            for i in 0..dim {
                let a = dst[i * dim + t].abs();
                if a > best_abs {
                    best_abs = a;
                    best = i;
                }
            }
            if dst[best * dim + t] < 0.0 {
                for i in 0..dim {
                    dst[i * dim + t] = -dst[i * dim + t];
                }
            }
        }
    }
}

/// Which side of the payload the Gram reads. `Input` forms `G = CᵀC` over
/// the payload COLUMNS (the `f_right × f_right` row-space Gram — the
/// paper's value-map side); `Output` forms `H = CCᵀ` over the payload ROWS
/// (the `f_left × f_left` column-space dual).
#[derive(Clone, Copy)]
enum Side {
    Input,
    Output,
}

impl Side {
    fn dim(self, rows: usize, cols: usize) -> usize {
        match self {
            Self::Input => cols,
            Self::Output => rows,
        }
    }
}

/// One side of the instrument, in one pass: both Grams (into
/// `scratch.g1` / `scratch.g2`), the CKA scalar, and — leaving Q₁ in
/// `scratch.basis` and Q₂ in `scratch.basis2` — the kept-direction counts
/// and the below-floor energy fractions. Refuses a zero payload (the CKA
/// denominator vanishes; nothing is priceable).
struct SideOutcome {
    scalar: f64,
    kept: [usize; 2],
    below: [f64; 2],
}

fn side_grams(
    c1: &[f32],
    c2: &[f32],
    rows: usize,
    cols: usize,
    side: Side,
    scratch: &mut CrossTailScratch,
) -> Result<SideOutcome, SstError> {
    let dim = side.dim(rows, cols);
    // The Gram, upper triangle + mirror, sequential f64 folds (the
    // fit_anchor_into posture; the Issue-940 determinism law).
    let gram_into = |c: &[f32], out: &mut Vec<f64>| {
        ensure64(out, dim * dim);
        let g = &mut out[..dim * dim];
        g.fill(0.0);
        match side {
            Side::Input => {
                // G[a][b] = Σ_t C[t][a]·C[t][b].
                for a in 0..dim {
                    for b in a..dim {
                        let mut acc = 0.0f64;
                        for t in 0..rows {
                            acc += (c[t * cols + a] as f64) * (c[t * cols + b] as f64);
                        }
                        g[a * dim + b] = acc;
                        g[b * dim + a] = acc;
                    }
                }
            }
            Side::Output => {
                // H[a][b] = Σ_t C[a][t]·C[b][t].
                for a in 0..dim {
                    for b in a..dim {
                        let mut acc = 0.0f64;
                        for t in 0..cols {
                            acc += (c[a * cols + t] as f64) * (c[b * cols + t] as f64);
                        }
                        g[a * dim + b] = acc;
                        g[b * dim + a] = acc;
                    }
                }
            }
        }
    };
    gram_into(c1, &mut scratch.g1);
    gram_into(c2, &mut scratch.g2);

    // CKA: ⟨G₁,G₂⟩_F / (‖G₁‖_F·‖G₂‖_F), one sequential index fold.
    let (g1, g2) = (&scratch.g1[..dim * dim], &scratch.g2[..dim * dim]);
    let mut num = 0.0f64;
    let mut n1 = 0.0f64;
    let mut n2 = 0.0f64;
    for i in 0..dim * dim {
        num += g1[i] * g2[i];
        n1 += g1[i] * g1[i];
        n2 += g2[i] * g2[i];
    }
    let (n1, n2) = (n1.sqrt(), n2.sqrt());
    if n1 <= 0.0 || n2 <= 0.0 {
        return Err(SstError::ComposeMismatch(
            "zero payload — no alignment to price",
        ));
    }
    // Cauchy–Schwarz holds exactly; clamp the f64 rounding residue.
    let scalar = (num / (n1 * n2)).clamp(0.0, 1.0);

    // Truncated eigenbasis + below-floor fraction, per payload. The cutoff
    // is λ-relative: σ_i > F·σ_max ⟺ λ_i > F²·λ_max.
    let mut kept = [0usize; 2];
    let mut below = [0.0f64; 2];
    for idx in 0..2 {
        let gram_src: &[f64] = if idx == 0 { g1 } else { g2 };
        ensure64(&mut scratch.eigvals, dim);
        ensure64(&mut scratch.eigvecs, dim * dim);
        ensure64(&mut scratch.jac, dim * dim);
        let trace: f64 = (0..dim).map(|i| gram_src[i * dim + i]).sum();
        let tol = 1e-14 * (trace.abs() + 1.0);
        jacobi_eigen(
            &mut scratch.eigvals[..dim],
            &mut scratch.eigvecs[..dim * dim],
            gram_src,
            &mut scratch.jac[..dim * dim],
            dim,
            tol,
            JACOBI_MAX_SWEEPS,
        );
        let dst = if idx == 0 {
            &mut scratch.basis
        } else {
            &mut scratch.basis2
        };
        sorted_canonical_basis(
            &scratch.eigvals[..dim],
            &scratch.eigvecs[..dim * dim],
            dst,
            &mut scratch.order,
            dim,
        );
        let lam_max = scratch.eigvals[scratch.order[0]].max(0.0);
        let lam_floor = lam_max * SIGMA_REL_FLOOR * SIGMA_REL_FLOOR;
        let mut kept_i = 0usize;
        let mut below_mass = 0.0f64;
        let mut total = 0.0f64;
        for t in 0..dim {
            let lam = scratch.eigvals[scratch.order[t]].max(0.0);
            total += lam;
            if lam > lam_floor {
                kept_i += 1;
            } else {
                below_mass += lam;
            }
        }
        kept[idx] = kept_i;
        below[idx] = if total > 0.0 { below_mass / total } else { 0.0 };
    }
    Ok(SideOutcome {
        scalar,
        kept,
        below,
    })
}

/// The cos² principal-angle spectrum between the two truncated bases left
/// by [`side_grams`] — eigenvalues (descending) of `M Mᵀ` with
/// `M = Q₁ᵀQ₂` restricted to the kept columns. The cos² values are
/// invariant to the bases' column signs, so the canonicalization only
/// fixes reproducibility. Zero-alloc: `scratch.cross` / `scratch.mt` grow
/// once; `out` is cleared and refilled (the caller-reuse contract).
fn side_spectrum(dim: usize, kept: [usize; 2], scratch: &mut CrossTailScratch, out: &mut Vec<f64>) {
    let (k1, k2) = (kept[0], kept[1]);
    out.clear();
    if k1 == 0 || k2 == 0 {
        return;
    }
    // M = Q₁ᵀ·Q₂ over the kept columns (k1 × k2), sequential folds.
    ensure64(&mut scratch.cross, k1 * k2);
    {
        let m = &mut scratch.cross[..k1 * k2];
        m.fill(0.0);
        for i in 0..k1 {
            for j in 0..k2 {
                let mut acc = 0.0f64;
                for t in 0..dim {
                    acc += scratch.basis[t * dim + i] * scratch.basis2[t * dim + j];
                }
                m[i * k2 + j] = acc;
            }
        }
    }
    // Gram M Mᵀ (k1 × k1) — the jacobi input.
    ensure64(&mut scratch.mt, k1 * k1);
    {
        let m = &scratch.cross[..k1 * k2];
        let mt = &mut scratch.mt[..k1 * k1];
        mt.fill(0.0);
        for i in 0..k1 {
            for j in i..k1 {
                let mut acc = 0.0f64;
                for t in 0..k2 {
                    acc += m[i * k2 + t] * m[j * k2 + t];
                }
                mt[i * k1 + j] = acc;
                mt[j * k1 + i] = acc;
            }
        }
    }
    ensure64(&mut scratch.eigvals, k1);
    ensure64(&mut scratch.eigvecs, k1 * k1);
    ensure64(&mut scratch.jac, k1 * k1);
    jacobi_eigen(
        &mut scratch.eigvals[..k1],
        &mut scratch.eigvecs[..k1 * k1],
        &scratch.mt[..k1 * k1],
        &mut scratch.jac[..k1 * k1],
        k1,
        1e-14 * 2.0,
        JACOBI_MAX_SWEEPS,
    );
    // cos² ∈ [0,1] by construction; clamp solver residue, sort descending
    // in place (the scratch buffer is ours to permute).
    for v in &mut scratch.eigvals[..k1] {
        *v = v.clamp(0.0, 1.0);
    }
    scratch.eigvals[..k1].sort_by(|a, b| b.total_cmp(a));
    out.extend_from_slice(&scratch.eigvals[..k1]);
}

// ---------------------------------------------------------------------------
// The instrument
// ---------------------------------------------------------------------------

/// Price the cross-tail interference between two SAME-ANCHOR tail factors
/// (Issue 972 T1). Refuses loud, mirroring
/// [`crate::sst_certified::compose_tail_into`]'s compose-refusal classes —
/// a cross-basis comparison is meaningless without a re-factor, so it never
/// re-factors silently. Zero-alloc over `scratch` after a warm call;
/// `report`'s spectra are cleared and refilled (caller-reuse contract,
/// same posture as `TailFactor::ensure_shapes`).
///
/// # Errors
///
/// [`SstError::ComposeMismatch`] when the factors disagree on anchor
/// digest, shape, or tail bases, carry a non-finite payload, or a zero
/// payload (nothing to price).
pub fn cross_tail_interference_into(
    a: &TailFactor,
    b: &TailFactor,
    scratch: &mut CrossTailScratch,
    report: &mut CrossTailReport,
) -> Result<(), SstError> {
    if a.anchor_id != b.anchor_id {
        return Err(SstError::ComposeMismatch(
            "anchor digest differs across links",
        ));
    }
    if (a.m, a.n, a.f_left, a.f_right) != (b.m, b.n, b.f_left, b.f_right) {
        return Err(SstError::ComposeMismatch(
            "factor shapes differ across links",
        ));
    }
    let fl_fr = a.f_left * a.f_right;
    if a.left_basis[..a.m * a.f_left] != b.left_basis[..a.m * a.f_left]
        || a.right_basis[..a.n * a.f_right] != b.right_basis[..a.n * a.f_right]
    {
        return Err(SstError::ComposeMismatch(
            "tail bases differ (re-anchor or truncation) — re-factor every link against one anchor",
        ));
    }
    if !a.c[..fl_fr].iter().all(|v| v.is_finite()) || !b.c[..fl_fr].iter().all(|v| v.is_finite()) {
        return Err(SstError::ComposeMismatch("non-finite payload"));
    }

    let c1 = &a.c[..fl_fr];
    let c2 = &b.c[..fl_fr];

    let in_side = side_grams(c1, c2, a.f_left, a.f_right, Side::Input, scratch)?;
    let mut input_spectrum = core::mem::take(&mut report.input_spectrum);
    side_spectrum(a.f_right, in_side.kept, scratch, &mut input_spectrum);

    let out_side = side_grams(c1, c2, a.f_left, a.f_right, Side::Output, scratch)?;
    let mut output_spectrum = core::mem::take(&mut report.output_spectrum);
    side_spectrum(a.f_left, out_side.kept, scratch, &mut output_spectrum);

    report.input_scalar = in_side.scalar;
    report.output_scalar = out_side.scalar;
    report.sigma_rel_floor = SIGMA_REL_FLOOR;
    report.input_energy_below_floor = in_side.below;
    report.output_energy_below_floor = out_side.below;
    report.input_spectrum = input_spectrum;
    report.output_spectrum = output_spectrum;
    Ok(())
}

/// Convenience (allocating scratch + report) form of
/// [`cross_tail_interference_into`].
///
/// # Errors
///
/// The input guards — see `cross_tail_interference_into`.
pub fn cross_tail_interference(
    a: &TailFactor,
    b: &TailFactor,
) -> Result<CrossTailReport, SstError> {
    let mut scratch = CrossTailScratch::default();
    let mut report = CrossTailReport::default();
    cross_tail_interference_into(a, b, &mut scratch, &mut report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic fixture RNG (the unseeded-global-RNG gate: a local
    // LCG, never a global draw).
    struct Lcg(u64);
    impl Lcg {
        fn next_u(&mut self) -> f64 {
            // xorshift64* — deterministic, no global state.
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((v >> 11) as f64) / (1u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            // Box–Muller, two uniforms, clamped away from log(0).
            let u1 = self.next_u().max(1e-12);
            let u2 = self.next_u();
            (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
        }
    }

    fn gauss_mat(seed: u64, rows: usize, cols: usize) -> Vec<f64> {
        let mut rng = Lcg(seed | 1);
        (0..rows * cols).map(|_| rng.gauss()).collect()
    }

    fn mat_mul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for t in 0..k {
                    acc += a[i * k + t] * b[t * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    /// The exact-representable Givens fixture (Issue 972 T2): two rank-r
    /// payloads whose ROW spaces sit at principal angle θ in R^f (f = 2r).
    /// `C1 = Q1ᵀ` (identity mixing — entries 0/1, f32-exact);
    /// `C2 = Q2ᵀ` with Q2's i-th vector `c·e_i + s·e_{r+i}` (0° and 90°
    /// built DIRECTLY, skipping trig, so those two are exact). The f32
    /// rounding of c/s keeps every vector inside its (e_i, e_{r+i}) plane,
    /// so the constructed angle survives as
    /// `expected_cos2 = c²/(c²+s²)` computed from the ROUNDED values — the
    /// instrument must match it to f64 rounding. `p` is the exact f×f
    /// projector onto row space 2 (the T3 measured fraction uses it).
    struct GivensFixture {
        c1: Vec<f32>,
        c2: Vec<f32>,
        r: usize,
        f: usize,
        expected_cos2: f64,
        p: Vec<f64>,
    }

    fn exact_givens(r: usize, theta_deg: f64) -> GivensFixture {
        let f = 2 * r;
        let (c, s) = if theta_deg == 0.0 {
            (1.0f64, 0.0f64)
        } else if theta_deg == 90.0 {
            (0.0f64, 1.0f64)
        } else {
            (
                (theta_deg.to_radians().cos() as f32) as f64,
                (theta_deg.to_radians().sin() as f32) as f64,
            )
        };
        let expected_cos2 = c * c / (c * c + s * s);
        let mut c1 = vec![0.0f32; r * f];
        let mut c2 = vec![0.0f32; r * f];
        let mut p = vec![0.0f64; f * f];
        // P is the projector onto row space 2 — built from the UNIT
        // rounded basis vectors (the f32 rounding makes |v| = √(c²+s²) ≠
        // 1; without the normalization P² ≠ P and the measured fraction
        // picks up a spurious (1+δ) factor).
        let norm = (c * c + s * s).sqrt();
        let (cn, sn) = (c / norm, s / norm);
        for i in 0..r {
            c1[i * f + i] = 1.0;
            c2[i * f + i] = c as f32;
            c2[i * f + (r + i)] = s as f32;
            // P = Σᵢ v̂ᵢ v̂ᵢᵀ (no cross terms between different i).
            p[i * f + i] += cn * cn;
            p[i * f + (r + i)] += cn * sn;
            p[(r + i) * f + i] += sn * cn;
            p[(r + i) * f + (r + i)] += sn * sn;
        }
        GivensFixture {
            c1,
            c2,
            r,
            f,
            expected_cos2,
            p,
        }
    }

    fn make_factor(c: Vec<f32>, fl: usize, fr: usize) -> TailFactor {
        TailFactor {
            anchor_id: [7u8; 32],
            m: 8,
            n: 8,
            f_left: fl,
            f_right: fr,
            left_basis: vec![0.5f32; 8 * fl],
            right_basis: vec![0.25f32; 8 * fr],
            c,
            leak_bound: 0.0,
        }
    }

    fn frob(m: &[f64]) -> f64 {
        m.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// T2 — the exact family's input spectrum IS the constructed angle
    /// (every entry: the Givens construction makes all r principal angles
    /// equal), to f64 rounding of f32-exact inputs.
    #[test]
    fn spectrum_matches_constructed_angles_exact_family() {
        for theta in [30.0f64, 45.0, 60.0] {
            let fx = exact_givens(3, theta);
            let a = make_factor(fx.c1.clone(), fx.r, fx.f);
            let b = make_factor(fx.c2.clone(), fx.r, fx.f);
            let rep = cross_tail_interference(&a, &b).expect("prices");
            assert_eq!(rep.input_spectrum.len(), fx.r);
            for v in &rep.input_spectrum {
                assert!(
                    (v - fx.expected_cos2).abs() <= 1e-12,
                    "theta {theta}: {v} vs {}",
                    fx.expected_cos2
                );
            }
            assert_eq!(rep.sigma_rel_floor, 1e-6);
            assert_eq!(rep.input_energy_below_floor, [0.0, 0.0]);
        }
    }

    /// T2 — scalars/spectra monotone in θ, exact family AND gauss-mixed
    /// (the generic-mixing arm floors at the f32 rounding of the mix).
    #[test]
    fn spectrum_monotone_across_angles() {
        for (label, mixing) in [("exact", false), ("gauss", true)] {
            let mut prev = f64::INFINITY;
            for theta in [0.0f64, 30.0, 45.0, 60.0, 90.0] {
                let fx = exact_givens(3, theta);
                let (fl, c1, c2) = if mixing {
                    let a1 = gauss_mat(0xBEEF, 8, 3);
                    let a2 = gauss_mat(0xF00D, 8, 3);
                    let q1t: Vec<f64> = (0..3 * fx.f)
                        .map(|i| if i % fx.f == i / fx.f { 1.0 } else { 0.0 })
                        .collect();
                    let q2t = {
                        let mut q = vec![0.0f64; 3 * fx.f];
                        let (c, s) = (
                            (theta.to_radians().cos() as f32) as f64,
                            (theta.to_radians().sin() as f32) as f64,
                        );
                        for i in 0..3 {
                            q[i * fx.f + i] = c;
                            q[i * fx.f + 3 + i] = s;
                        }
                        q
                    };
                    let m1 = mat_mul(&a1, &q1t, 8, 3, fx.f);
                    let m2 = mat_mul(&a2, &q2t, 8, 3, fx.f);
                    (
                        8,
                        m1.iter().map(|v| *v as f32).collect::<Vec<f32>>(),
                        m2.iter().map(|v| *v as f32).collect::<Vec<f32>>(),
                    )
                } else {
                    (fx.r, fx.c1.clone(), fx.c2.clone())
                };
                let a = make_factor(c1, fl, fx.f);
                let b = make_factor(c2, fl, fx.f);
                let rep = cross_tail_interference(&a, &b).expect("prices");
                let tol = if mixing { 1e-5 } else { 1e-12 };
                let want = if mixing {
                    // The gauss arm's subspaces are f32-rounded mixes; the
                    // constructed angle still leads at 1e-5.
                    (theta.to_radians().cos() as f32) as f64
                        * (theta.to_radians().cos() as f32) as f64
                } else {
                    fx.expected_cos2
                };
                let top = rep.input_spectrum[0];
                assert!(
                    (top - want).abs() <= tol,
                    "{label} theta {theta}: {top} vs {want}"
                );
                assert!(
                    top < prev,
                    "{label} theta {theta}: not monotone ({top} >= {prev})"
                );
                prev = top;
                assert!((0.0..=1.0).contains(&rep.input_scalar));
                assert!((0.0..=1.0).contains(&rep.output_scalar));
            }
        }
    }

    /// T2 — EXACT zero overlap at 90° (built by direct axis construction,
    /// never trig): the CKA numerator accumulates exact-zero products, so
    /// the scalar is 0.0 exactly, and the spectrum is the zero matrix's.
    #[test]
    fn exact_zero_overlap_at_90_degrees() {
        let fx = exact_givens(3, 90.0);
        let a = make_factor(fx.c1.clone(), fx.r, fx.f);
        let b = make_factor(fx.c2.clone(), fx.r, fx.f);
        let rep = cross_tail_interference(&a, &b).expect("prices");
        assert_eq!(rep.input_scalar, 0.0, "Gram-orthogonal pairs price 0");
        for v in &rep.input_spectrum {
            assert!(*v <= 1e-15, "cos² 90 must be exact zero, got {v}");
        }
        assert_eq!(rep.input_energy_below_floor, [0.0, 0.0]);
    }

    /// T2 — equal payloads price the identity: scalar exactly 1.0 (up to
    /// fold rounding), spectrum all ones (M = QᵀQ = I).
    #[test]
    fn equal_payload_identity() {
        let fx = exact_givens(3, 45.0);
        let a = make_factor(fx.c1.clone(), fx.r, fx.f);
        let b = make_factor(fx.c1.clone(), fx.r, fx.f);
        let rep = cross_tail_interference(&a, &b).expect("prices");
        assert!((rep.input_scalar - 1.0).abs() <= 1e-12);
        for v in &rep.input_spectrum {
            assert!((v - 1.0).abs() <= 1e-12, "identical spans: {v}");
        }
    }

    /// T2 — the full-rank guard arm (the trivial-angles killer): two
    /// DIFFERENT full-rank payloads span the SAME R^f, so the spectrum is
    /// trivially all ones — and the scalar must still separate them. A
    /// naive angle-only spec reports "identical" here.
    #[test]
    fn full_rank_guard_separates() {
        let f = 6usize;
        let mut c1 = vec![0.0f32; f * f];
        let mut c2 = vec![0.0f32; f * f];
        for i in 0..f {
            c1[i * f + i] = 1.0;
            c2[i * f + i] = (i + 1) as f32;
        }
        let a = make_factor(c1, f, f);
        let b = make_factor(c2, f, f);
        let rep = cross_tail_interference(&a, &b).expect("prices");
        for v in &rep.input_spectrum {
            assert!((v - 1.0).abs() <= 1e-12, "both span R^f: {v}");
        }
        assert!(
            rep.input_scalar < 0.99,
            "distinct full-rank payloads must not read identical: {}",
            rep.input_scalar
        );
        assert!(rep.input_scalar > 0.0);
    }

    /// T1 — the refusal arms mirror `compose_tail_into`'s classes verbatim,
    /// plus the instrument-specific non-finite / zero-payload guards.
    #[test]
    fn refusal_arms_mirror_compose_classes() {
        let fx = exact_givens(3, 45.0);
        let a = make_factor(fx.c1.clone(), fx.r, fx.f);

        let mut bad = make_factor(fx.c2.clone(), fx.r, fx.f);
        bad.anchor_id = [9u8; 32];
        assert!(matches!(
            cross_tail_interference(&a, &bad),
            Err(SstError::ComposeMismatch(
                "anchor digest differs across links"
            ))
        ));

        let mut bad = make_factor(fx.c2.clone(), fx.r, fx.f);
        bad.f_left += 1;
        bad.left_basis.push(0.5);
        assert!(matches!(
            cross_tail_interference(&a, &bad),
            Err(SstError::ComposeMismatch(
                "factor shapes differ across links"
            ))
        ));

        let mut bad = make_factor(fx.c2.clone(), fx.r, fx.f);
        bad.left_basis[0] += 0.5;
        assert!(matches!(
            cross_tail_interference(&a, &bad),
            Err(SstError::ComposeMismatch(x)) if x.contains("tail bases differ")
        ));

        let mut bad = make_factor(fx.c2.clone(), fx.r, fx.f);
        bad.c[0] = f32::NAN;
        assert!(matches!(
            cross_tail_interference(&a, &bad),
            Err(SstError::ComposeMismatch("non-finite payload"))
        ));

        let zero = make_factor(vec![0.0; fx.r * fx.f], fx.r, fx.f);
        assert!(matches!(
            cross_tail_interference(&a, &zero),
            Err(SstError::ComposeMismatch(x)) if x.contains("zero payload")
        ));
    }

    /// T3 — measured projection-energy fraction with PINNED units: the
    /// exact family's measured fraction equals the constructed cos² (the
    /// identity mixing makes the bound's equality case), and the hard
    /// bound `measured ≤ max cos² + ε₁` holds with ε₁ = 0 by construction.
    #[test]
    fn measured_fraction_hard_bound() {
        for theta in [30.0f64, 45.0, 60.0, 90.0] {
            let fx = exact_givens(3, theta);
            let a = make_factor(fx.c1.clone(), fx.r, fx.f);
            let b = make_factor(fx.c2.clone(), fx.r, fx.f);
            let rep = cross_tail_interference(&a, &b).expect("prices");
            // measured = ‖C1·P_row(C2)‖²_F / ‖C1‖²_F, materialized from
            // the payloads through the shared bases (P = Q2Q2ᵀ, the SAME
            // truncated basis the instrument used — identity mixing keeps
            // every direction above the cutoff, so P IS the truncated
            // projector, not just a rank-r proxy).
            let c1: Vec<f64> = fx.c1.iter().map(|v| f64::from(*v)).collect();
            let mut cp = vec![0.0f64; fx.r * fx.f];
            for i in 0..fx.r {
                for j in 0..fx.f {
                    let mut acc = 0.0f64;
                    for t in 0..fx.f {
                        acc += c1[i * fx.f + t] * fx.p[t * fx.f + j];
                    }
                    cp[i * fx.f + j] = acc;
                }
            }
            let measured = frob(&cp).powi(2) / frob(&c1).powi(2);
            assert!(
                (measured - fx.expected_cos2).abs() <= 1e-10,
                "theta {theta}: measured {measured} vs {}",
                fx.expected_cos2
            );
            assert_eq!(rep.input_energy_below_floor[0], 0.0);
            assert!(
                measured <= rep.input_max_cos2() + rep.input_energy_below_floor[0] + 1e-12,
                "hard bound: {measured} > {} + {}",
                rep.input_max_cos2(),
                rep.input_energy_below_floor[0]
            );
        }
    }

    /// T3 — the two-step closed-form linear recursion (the paper's
    /// gradient-flow leading order, fixture-simulated — NO shipped GD):
    /// two GD steps at rate η toward zero targets over inputs confined to
    /// row space 2 close in closed form to `W_A·(I − ηP)²`. The old-task
    /// drift on inputs confined to row space 1 is then (a) monotone in θ
    /// and (b) leading-order bounded by the priced fraction:
    /// `drift ≤ (2η + η²)² · measured` (drift = ‖C1(−2ηP + η²P²)Q1‖ ≤
    /// (2η+η²)·‖C1·P‖ pointwise, and ‖C1·P·Q1‖ ≤ ‖C1·P‖).
    #[test]
    fn two_step_recursion_ranking_and_leading_order_bound() {
        let eta = 1e-3f64;
        let mut prev = f64::INFINITY;
        for theta in [30.0f64, 45.0, 60.0] {
            let fx = exact_givens(3, theta);
            let a = make_factor(fx.c1.clone(), fx.r, fx.f);
            let b = make_factor(fx.c2.clone(), fx.r, fx.f);
            let rep = cross_tail_interference(&a, &b).expect("prices");
            let c1: Vec<f64> = fx.c1.iter().map(|v| f64::from(*v)).collect();
            // Q1 (f × r, columns = row-space-1 basis): identity rows.
            let mut q1 = vec![0.0f64; fx.f * fx.r];
            for i in 0..fx.r {
                q1[i * fx.f + i] = 1.0;
            }
            // The recursion's drift: D = C1·(−2ηP + η²P²).
            let mut p2 = mat_mul(&fx.p, &fx.p, fx.f, fx.f, fx.f);
            for v in p2.iter_mut() {
                *v *= eta * eta;
            }
            let mut dp = vec![0.0f64; fx.f * fx.f];
            for i in 0..fx.f * fx.f {
                dp[i] = -2.0 * eta * fx.p[i] + p2[i];
            }
            let drift = mat_mul(&c1, &dp, fx.r, fx.f, fx.f);
            // Old-task drift fraction on inputs confined to row space 1:
            // ‖drift·Q1‖² / ‖C1·Q1‖².
            let drift_q = mat_mul(&drift, &q1, fx.r, fx.f, fx.r);
            let c1_q = mat_mul(&c1, &q1, fx.r, fx.f, fx.r);
            let frac = frob(&drift_q).powi(2) / frob(&c1_q).powi(2);
            // (a) monotone in θ (forgetting falls as the subspaces split).
            assert!(frac < prev, "theta {theta}: drift {frac} not < {prev}");
            prev = frac;
            // (b) the leading-order bound against the priced fraction.
            let measured = {
                let mut cp = vec![0.0f64; fx.r * fx.f];
                for i in 0..fx.r {
                    for j in 0..fx.f {
                        let mut acc = 0.0f64;
                        for t in 0..fx.f {
                            acc += c1[i * fx.f + t] * fx.p[t * fx.f + j];
                        }
                        cp[i * fx.f + j] = acc;
                    }
                }
                frob(&cp).powi(2) / frob(&c1).powi(2)
            };
            let lead = (2.0 * eta + eta * eta).powi(2);
            assert!(
                frac <= lead * measured + 1e-12,
                "theta {theta}: drift {frac} > {lead} · {measured}"
            );
            // The priced hard bound rides the same report.
            assert!(
                measured <= rep.input_max_cos2() + rep.input_energy_below_floor[0] + 1e-12,
                "theta {theta}: measured {measured} exceeds the priced bound"
            );
        }
    }

    /// The output-side dual: transposing the exact-family pair swaps the
    /// sides — the transposed pair's OUTPUT spectrum is the constructed
    /// angle, its output scalar equals the original's input scalar (the
    /// same Gram products H = CᵀC), and its INPUT side goes trivial (both
    /// column spaces are all of R^{f_left}).
    #[test]
    fn transposed_pair_dual_consistency() {
        let fx = exact_givens(3, 45.0);
        let a = make_factor(fx.c1.clone(), fx.r, fx.f);
        let b = make_factor(fx.c2.clone(), fx.r, fx.f);
        let orig = cross_tail_interference(&a, &b).expect("prices");

        let mut ca = vec![0.0f32; fx.f * fx.r];
        let mut cb = vec![0.0f32; fx.f * fx.r];
        for i in 0..fx.r {
            for j in 0..fx.f {
                ca[j * fx.r + i] = fx.c1[i * fx.f + j];
                cb[j * fx.r + i] = fx.c2[i * fx.f + j];
            }
        }
        let ta = make_factor(ca, fx.f, fx.r);
        let tb = make_factor(cb, fx.f, fx.r);
        let rep = cross_tail_interference(&ta, &tb).expect("prices");
        assert_eq!(rep.output_spectrum.len(), fx.r);
        assert!(
            (rep.output_max_cos2() - fx.expected_cos2).abs() <= 1e-12,
            "dual spectrum {} vs {}",
            rep.output_max_cos2(),
            fx.expected_cos2
        );
        assert!(
            (rep.output_scalar - orig.input_scalar).abs() <= 1e-12,
            "same Gram products must price identically"
        );
        // The transposed INPUT side is trivial: both column spaces are
        // R^{f_left} (the trivial-angles hazard, demonstrated).
        for v in &rep.input_spectrum {
            assert!((v - 1.0).abs() <= 1e-12, "trivial input side: {v}");
        }
        assert!((rep.input_scalar - 1.0).abs() <= 1e-12);
    }

    /// T4 — determinism: two runs on the same inputs produce bit-identical
    /// reports (sorted eigenpairs, canonical signs, sequential folds — the
    /// Issue-940 law).
    #[test]
    fn determinism_bit_identical_reports() {
        let fx = exact_givens(3, 45.0);
        let a = make_factor(fx.c1.clone(), fx.r, fx.f);
        let b = make_factor(fx.c2.clone(), fx.r, fx.f);
        let r1 = cross_tail_interference(&a, &b).expect("prices");
        let r2 = cross_tail_interference(&a, &b).expect("prices");
        assert_eq!(r1, r2, "same inputs must price bit-identically");
    }

    /// T4 — rotation invariance, two postures: (a) an EXACT-representable
    /// ambient mixing (a column permutation) moves nothing beyond f64
    /// rounding; (b) a general orthogonal mixing floors at the f32 payload
    /// wire (~1e-5) — the issue's 1e-10 assumed f64 payloads; the honest
    /// f32 floor is recorded here.
    #[test]
    fn rotation_invariance() {
        let fx = exact_givens(3, 45.0);
        let a = make_factor(fx.c1.clone(), fx.r, fx.f);
        let b = make_factor(fx.c2.clone(), fx.r, fx.f);
        let base = cross_tail_interference(&a, &b).expect("prices");

        // (a) permutation: exact-invariant (values only move).
        let perm: Vec<usize> = (0..fx.f).map(|i| (i + 1) % fx.f).collect();
        let permute = |c: &[f32], p: &[usize]| -> Vec<f32> {
            (0..fx.r)
                .flat_map(|i| (0..fx.f).map(move |j| c[i * fx.f + p[j]]))
                .collect()
        };
        let pa = make_factor(permute(&fx.c1, &perm), fx.r, fx.f);
        let pb = make_factor(permute(&fx.c2, &perm), fx.r, fx.f);
        let pp = cross_tail_interference(&pa, &pb).expect("prices");
        assert!((pp.input_scalar - base.input_scalar).abs() <= 1e-12);
        assert!((pp.output_scalar - base.output_scalar).abs() <= 1e-12);
        for (x, y) in pp.input_spectrum.iter().zip(&base.input_spectrum) {
            assert!((x - y).abs() <= 1e-12);
        }

        // (b) general rotation: the f32 floor arm.
        let mut rng = Lcg(0xA11CE);
        let mut m = Vec::with_capacity(fx.f * fx.f);
        for _ in 0..fx.f * fx.f {
            m.push(rng.gauss());
        }
        // Modified Gram–Schmidt over the rows.
        let mut r_mat = m.clone();
        for i in 0..fx.f {
            for k in 0..i {
                let dot: f64 = (0..fx.f)
                    .map(|j| r_mat[i * fx.f + j] * r_mat[k * fx.f + j])
                    .sum();
                for j in 0..fx.f {
                    r_mat[i * fx.f + j] -= dot * r_mat[k * fx.f + j];
                }
            }
            let n = frob(&r_mat[i * fx.f..(i + 1) * fx.f]);
            for j in 0..fx.f {
                r_mat[i * fx.f + j] /= n;
            }
        }
        let rotate = |c: &[f32]| -> Vec<f32> {
            let cf: Vec<f64> = c.iter().map(|v| f64::from(*v)).collect();
            let out = mat_mul(&cf, &r_mat, fx.r, fx.f, fx.f);
            out.iter().map(|v| *v as f32).collect()
        };
        let ra = make_factor(rotate(&fx.c1), fx.r, fx.f);
        let rb = make_factor(rotate(&fx.c2), fx.r, fx.f);
        let rp = cross_tail_interference(&ra, &rb).expect("prices");
        assert!(
            (rp.input_scalar - base.input_scalar).abs() <= 1e-5,
            "general rotation, f32 floor: {} vs {}",
            rp.input_scalar,
            base.input_scalar
        );
        assert!(
            (rp.input_spectrum[0] - base.input_spectrum[0]).abs() <= 1e-5,
            "spectrum rotation floor"
        );
    }
}
