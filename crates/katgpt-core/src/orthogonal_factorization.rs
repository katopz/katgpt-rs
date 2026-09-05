//! Orthogonal Factorization Primitives — orthonormalize + activity hinge +
//! Parseval certificate (Issue 687, Research 504 — arXiv:2608.20065
//! "Orthogonal JEPA", Path 0 modelless extraction).
//!
//! The paper's *structure* is closed-form linear algebra with no gradient
//! anywhere, and it fills three documented gaps:
//!
//! 1. **Production direction sets are never orthogonalized.** The 5 HLA
//!    affect directions and the 14 planner drive `dir_vec`s are extracted
//!    contrastively and may correlate; orthogonality is a TEST-ONLY
//!    assumption. Cross-talk between "fear" and "despair" channels is the
//!    paper's monolithic-target failure mode wearing our vocabulary.
//!    [`orthonormalize_into`] fixes that at construction, and its `defect`
//!    output (the paper's `L_orth`) is a one-shot redundancy audit of the
//!    ORIGINAL set.
//! 2. **No per-coordinate activity floor exists.** `effective_rank` is
//!    aggregate (a dead channel hides in a full-rank population — the exact
//!    insufficiency `data_probe/gaussianity.rs` documents); gaussianity is
//!    distribution-shape. The hinge `max(0, γ−σ̂)` gives bounded,
//!    per-(factor,coordinate)-attributed variance deficit — the missing
//!    third axis ([`factor_activity_hinge`]).
//! 3. **No runtime Parseval invariant / exact truncation certificate.**
//!    With an orthonormal-complete basis: `‖z‖² = Σ_k (B_k·z)²`
//!    ([`parseval_energy_check`], O(d)) and dropped-energy truncation error
//!    is an identity, not an approximation ([`kept_energy`]).
//!    [`hadamard_factorize`] builds the integer-core Walsh basis whose
//!    dyadic scale (d = 4^m, incl. d = 64) makes both checks EXACT in f32 —
//!    the cross-platform bit-identity witness.
//!
//! # Conditioning (T4)
//!
//! The paper's conditioning caveat watches the synthesis basis B. With B
//! orthonormal BY CONSTRUCTION, κ(B) = 1 — the caveat converts to a
//! certificate: per-head amplification `‖W_k‖₂ = √λ_max(W_kᵀW_k)` via the
//! pinned Jacobi eigensolver in [`crate::spectral_pencil`] (one call per
//! head, construction-time only — [`head_conditioning`]); composite rollout
//! bound `Π_t max_k‖W_k‖₂` ([`rollout_bound`]). Orthogonality ≠ statistical
//! independence (the paper's own caveat) — the hinge and defect are
//! monitors, not disentanglement proofs.
//!
//! # γ schedule (T2)
//!
//! `γ(n) = max(γ_min, c/√n)` — the minimal schedule that stays above the
//! std-estimator's own sampling noise: for Gaussian coordinates
//! `sd(σ̂) ≈ σ/√(2n)`, so with the default `c = 1.0` a healthy unit-variance
//! channel cannot fire at any n (`1/√n > 0.707/√n`), while a genuinely dead
//! coordinate (σ → 0) fires with hinge value exactly `γ`. Defaults:
//! [`GAMMA_FAC_MIN`] = 0.25, [`GAMMA_SCHED_C`] = 1.0.
//!
//! # Determinism (G1)
//!
//! All reductions are scalar in-order `f64` accumulations over `f32` data
//! with fixed loop order (ascending). Rust's default FP semantics forbid
//! contraction and reassociation, so LLVM cannot vectorize/reorder these
//! reductions — outputs are bit-identical across runs AND platforms. The
//! Hadamard path is exact: ±1/√d entries and dyadic inputs make every
//! intermediate a dyadic rational (no rounding at any width ≥ the operands'
//! mantissas), so Parseval residual and recompose are EXACTLY `0.0` / bit-equal.
//!
//! # Allocation discipline (G4)
//!
//! [`orthonormalize_into`], [`parseval_energy_check`], [`recompose_into`],
//! [`kept_energy`], [`hadamard_factorize`], [`head_conditioning`] and the
//! [`FactorActivityScratch`] steady-state methods perform ZERO heap
//! allocation (caller-owned buffers throughout). [`FactorActivityScratch`]
//! allocates once at construction (the `GaussianityScratch` pattern).
//!
//! # Non-goals
//!
//! Learned/data-adaptive bases + dedicated trained heads → riir-train
//! Plan 351. Consumer wiring (riir-ai affect orthogonalization A/B — a
//! gameplay owner call per the CLR precedent; riir-neuron-db blend
//! interference gate) = separate issues AFTER the Issue 687 GOAT passes.
//! Opt-in behind `orthogonal_factorization` until a consumer promotes.
//!
//! Sigmoid, not softmax (per AGENTS.md) — aggregates are means and hinges;
//! no normalization competition anywhere.

use crate::spectral_pencil::{DenseScratch, jacobi_eigen};

/// Default floor for the factor-activity γ schedule. A factor coordinate
/// carrying less than a quarter of the population's typical unit scale is
/// "under-active" once enough samples accumulate (n ≥ 1/c² = 16 with the
/// default c; before that the schedule sits higher).
pub const GAMMA_FAC_MIN: f32 = 0.25;

/// Default γ-schedule coefficient: `γ ≥ c/√n` sits √2× above the Gaussian
/// σ̂ sampling noise `σ/√(2n)` at unit σ for every n.
pub const GAMMA_SCHED_C: f32 = 1.0;

/// Default relative tolerance for the Parseval energy check. Orthonormal
/// bases with f64 accumulation land ~1e-7; a duplicated/missing basis
/// vector lands O(1).
pub const PARSEVAL_TOL_REL: f32 = 1e-5;

/// A Gram–Schmidt row whose post-projection residual falls below this
/// fraction of its original norm is rank-deficient: it is zeroed (the
/// input `defect` has already fired — the redundancy is in the input set).
const REORTH_RELATIVE_FLOOR: f64 = 1e-6;

/// In-order f64 dot over `f32` operands, 8-lane accumulator pattern.
///
/// Determinism contract (G1 bit-identity across runs AND platforms) forbids
/// LLVM reassociation — but a SOURCE-FIXED association is just as
/// deterministic: element `i` always accumulates into lane `i % 8`, and the
/// lanes always sum in ascending order. The fixed pattern restores 8×
/// add-latency parallelism vs the sequential chain (which Rust must keep
/// scalar: ~3-cycle chained f64 adds × 64 elements would alone exceed the
/// G2 budget across the ~300 dots of a d=64/K=14 call).
///
/// Dyadic-exact operands (Hadamard fixtures) stay exact under ANY
/// association — the exactness anchors are unaffected by construction.
#[inline]
fn dot_f64<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0_f64; 8];
    for (ca, cb) in a.as_chunks::<8>().0.iter().zip(b.as_chunks::<8>().0.iter()) {
        for (slot, (x, y)) in acc.iter_mut().zip(ca.iter().zip(cb.iter())) {
            *slot += f64::from(*x) * f64::from(*y);
        }
    }
    let n = D;
    let tail = n - n % 8;
    for i in tail..n {
        acc[i % 8] += f64::from(a[i]) * f64::from(b[i]);
    }
    acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7]
}

// ──────────────────────────────────────────────────────────────────────────
// T1 — orthonormalize + L_orth defect
// ──────────────────────────────────────────────────────────────────────────

/// The paper's `L_orth` for a vector set (unit-width blocks: `B_k = b_k`):
/// `Σ_k (‖b_k‖² − 1)² + Σ_{i<j} (b_i·b_j)²`.
///
/// The first term penalizes non-unit norms, the second cross-talk — so the
/// score is calibrated for **unit-norm direction sets** (the production
/// shape). Zero iff the set is orthonormal. Use standalone to audit a
/// production direction set without orthogonalizing it.
#[must_use]
pub fn orthogonality_defect<const D: usize>(vectors: &[[f32; D]]) -> f32 {
    let mut defect: f64 = 0.0;
    for v in vectors {
        let n2 = dot_f64(v, v);
        let e = n2 - 1.0;
        defect += e * e;
    }
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            let d = dot_f64(&vectors[i], &vectors[j]);
            defect += d * d;
        }
    }
    defect as f32
}

/// Twice-reorthogonalized modified Gram–Schmidt ("twice is enough" —
/// Girard/Kahan): `out` receives an orthonormal basis for the span of
/// `vectors`, and `defect` receives the INPUT set's [`orthogonality_defect`]
/// — the one-shot redundancy audit that fires on planted collisions.
///
/// - In-place in `out` (no scratch, no allocation); deterministic ascending
///   iteration order; f64 accumulators.
/// - A rank-deficient row (residual ≤ `REORTH_RELATIVE_FLOOR`·‖original‖,
///   e.g. an exact duplicate or a K > D overflow) is zeroed — the defect
///   already carries the finding.
/// - Non-finite inputs produce a non-finite defect (guard with
///   `!defect.is_finite()` — the Batch-54 form) and zeroed rows.
///
/// The OUTPUT basis's own defect is ≤ ~1e-12 by construction (pinned by
/// test); consumers audit inputs, not outputs.
pub fn orthonormalize_into<const D: usize>(
    vectors: &[[f32; D]],
    out: &mut [[f32; D]],
    defect: &mut f32,
) {
    assert_eq!(
        vectors.len(),
        out.len(),
        "orthonormalize_into: out.len() must equal vectors.len()"
    );
    *defect = orthogonality_defect(vectors);

    for i in 0..out.len() {
        out[i] = vectors[i];
        let original_norm = dot_f64(&out[i], &out[i]).sqrt();
        // `prev` = rows 0..i (the settled basis), `row` = the working row.
        let (prev, cur) = out.split_at_mut(i);
        let row = &mut cur[0];
        // Two projection passes: the second removes the first pass's own
        // rounding residue ("twice is enough").
        for _pass in 0..2 {
            for pj in prev.iter() {
                let d = dot_f64(row, pj);
                if d != 0.0 {
                    // Storage-precision elementwise update with the
                    // f64-derived coefficient: the cancellation-critical
                    // dots stay f64; the update is f32 (house GS precedent
                    // — every shipped GS fixture is pure f32) and vectorizes;
                    // the second reorth pass cleans the extra rounding
                    // (measured max |cos| ~1e-7, gate 1e-6).
                    let d32 = d as f32;
                    for (oc, jc) in row.iter_mut().zip(pj.iter()) {
                        *oc -= d32 * *jc;
                    }
                }
            }
        }
        let residual = dot_f64(row, row).sqrt();
        if residual <= REORTH_RELATIVE_FLOOR * original_norm
            || !residual.is_finite()
            || !original_norm.is_finite()
        {
            *row = [0.0; D];
        } else {
            let inv = 1.0 / residual;
            for s in row.iter_mut() {
                *s = (f64::from(*s) * inv) as f32;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T2 — factor-activity variance hinge
// ──────────────────────────────────────────────────────────────────────────

/// Per-coordinate Welford accumulators over a population window of factor
/// coordinates: `K` factors × `r` coordinates each (row-major
/// `sample[k·r + j]`), f64 streams. Allocate once, observe a window, read
/// the hinge, [`reset`](Self::reset) for the next window.
///
/// The vector (multi-stream) shape is new substrate: the shipped Welford
/// copies (`karc/regime_gate`, `hint_regret`, downstream `chiaroscuro`)
/// are single-stream and module-local.
#[derive(Debug, Clone)]
pub struct FactorActivityScratch {
    k: usize,
    r: usize,
    mean: Box<[f64]>,
    m2: Box<[f64]>,
    count: u64,
}

impl FactorActivityScratch {
    /// Scratch for `K` factors × `r` coordinates. `K·r` parallel streams.
    #[must_use]
    pub fn new(k: usize, r: usize) -> Self {
        assert!(k > 0, "FactorActivityScratch: k must be positive");
        assert!(r > 0, "FactorActivityScratch: r must be positive");
        let n = k * r;
        Self {
            k,
            r,
            mean: vec![0.0; n].into_boxed_slice(),
            m2: vec![0.0; n].into_boxed_slice(),
            count: 0,
        }
    }

    /// Observe one sample: all `K` factors' coordinate vectors, flattened
    /// row-major (`len == k·r`, `sample[k·r + j]` = factor `k`'s coordinate
    /// `j`, typically `B_kᵀz` for the current latent `z`). O(K·r) — the
    /// amortized O(N·d) population pass.
    pub fn observe_sample(&mut self, sample: &[f32]) {
        assert_eq!(
            sample.len(),
            self.k * self.r,
            "observe_sample: len must equal k·r = {}",
            self.k * self.r
        );
        self.count = self.count.saturating_add(1);
        let n = self.count as f64;
        for (idx, &x) in sample.iter().enumerate() {
            let x = f64::from(x);
            let delta = x - self.mean[idx];
            self.mean[idx] += delta / n;
            self.m2[idx] += delta * (x - self.mean[idx]);
        }
    }

    /// Samples observed since construction / last reset.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Clear all streams for the next population window.
    pub fn reset(&mut self) {
        for v in self.mean.iter_mut() {
            *v = 0.0;
        }
        for v in self.m2.iter_mut() {
            *v = 0.0;
        }
        self.count = 0;
    }
}

/// The γ schedule: `max(γ_min, c/√n)` — the minimal schedule above the
/// σ̂-estimator's own sampling noise (module docs). `n = 0` is treated as 1.
#[must_use]
pub fn gamma_schedule(gamma_min: f32, c: f32, n: u64) -> f32 {
    let n_eff = (n.max(1)) as f64;
    ((f64::from(gamma_min)).max(f64::from(c) / n_eff.sqrt())) as f32
}

/// Factor-activity hinge report: `(1/Kr) Σ max(0, γ − σ̂_{k,j})` plus the
/// worst (k,j) attribution. All fields plain values — `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct ActivityReport {
    /// Mean hinge over all `K·r` coordinates, bounded `[0, γ]`.
    pub mean_hinge: f32,
    /// Flat argmax index (`k·r + j`) of the per-coordinate hinge.
    pub worst_flat: usize,
    /// The worst coordinate's hinge value (== γ when a channel is dead).
    pub worst_hinge: f32,
    /// The γ the hinge was evaluated at.
    pub gamma: f32,
    /// Samples in the window.
    pub n: u64,
}

/// Per-(factor,coordinate) variance hinge over the observed window:
/// fills `per_coord_out[k·r + j] = max(0, γ − σ̂_{k,j})` (sample std) and
/// returns the mean + worst attribution (the paper's
/// `(1/Kr) Σ max(0, γ_fac − σ̂_{k,j})`).
///
/// Fewer than 2 samples ⇒ NO CLAIM: all hinges 0 (a variance estimate from
/// one sample is noise, and the schedule exists precisely to keep the
/// estimator honest). A dead coordinate (constant across the window) has
/// `m2 == 0` exactly, so its hinge is EXACTLY `γ` — bit-exact firing.
pub fn factor_activity_hinge(
    scratch: &FactorActivityScratch,
    gamma: f32,
    per_coord_out: &mut [f32],
) -> ActivityReport {
    let n_streams = scratch.k * scratch.r;
    assert_eq!(
        per_coord_out.len(),
        n_streams,
        "factor_activity_hinge: per_coord_out.len() must equal k·r = {n_streams}"
    );
    if scratch.count < 2 {
        per_coord_out.fill(0.0);
        return ActivityReport {
            mean_hinge: 0.0,
            worst_flat: 0,
            worst_hinge: 0.0,
            gamma,
            n: scratch.count,
        };
    }
    let denom = f64::from(scratch.count as u32) - 1.0;
    let g = f64::from(gamma);
    let mut sum = 0.0_f64;
    let mut worst_flat = 0_usize;
    let mut worst_hinge = -1.0_f32;
    for (idx, slot) in per_coord_out.iter_mut().enumerate() {
        let sigma = (scratch.m2[idx] / denom).max(0.0).sqrt();
        let h32 = ((g - sigma).max(0.0)) as f32;
        *slot = h32;
        sum += f64::from(h32);
        if h32 > worst_hinge {
            worst_hinge = h32;
            worst_flat = idx;
        }
    }
    ActivityReport {
        mean_hinge: (sum / n_streams as f64) as f32,
        worst_flat,
        worst_hinge,
        gamma,
        n: scratch.count,
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T3 — Parseval invariant + recompose + exact truncation certificate
// ──────────────────────────────────────────────────────────────────────────

/// Parseval energy check result. `passed` uses [`PARSEVAL_TOL_REL`]; callers
/// with tighter contracts compare `residual_rel` directly (the Hadamard
/// witness is exactly `0.0`).
#[derive(Debug, Clone, Copy)]
pub struct ParsevalReport {
    /// `‖z‖²` (f64-accumulated).
    pub z_norm_sq: f32,
    /// `Σ_k (B_k·z)²` (f64-accumulated).
    pub factor_energy_sum: f32,
    /// `|z_norm_sq − factor_energy_sum| / z_norm_sq` (0 for `z = 0`).
    pub residual_rel: f32,
    /// `residual_rel ≤ PARSEVAL_TOL_REL` (and finite).
    pub passed: bool,
}

/// Parseval invariant for an orthonormal-COMPLETE basis (`basis.len() == D`):
/// `‖z‖² ≟ Σ_k (B_k·z)²`. Byproducts the factor coefficients `B_k·z` into
/// `coeffs_out` (one pass, no extra reads) — feed them to
/// [`recompose_into`]/[`kept_energy`] for the reconstruction identity and
/// the exact truncation certificate.
///
/// An incomplete (K < D) or non-orthonormal basis fails by Bessel / by the
/// cross-term leakage respectively — exactly the runtime structural check
/// the shipped per-op norm-preservation gates do not provide.
pub fn parseval_energy_check<const D: usize>(
    z: &[f32; D],
    basis: &[[f32; D]],
    coeffs_out: &mut [f32],
) -> ParsevalReport {
    assert_eq!(
        coeffs_out.len(),
        basis.len(),
        "parseval_energy_check: coeffs_out.len() must equal basis.len()"
    );
    let z2 = dot_f64(z, z);
    let mut total = 0.0_f64;
    for (k, b) in basis.iter().enumerate() {
        let c = dot_f64(b, z);
        coeffs_out[k] = c as f32;
        total += c * c;
    }
    let residual_rel = if z2 > 0.0 {
        ((z2 - total).abs() / z2) as f32
    } else {
        0.0
    };
    ParsevalReport {
        z_norm_sq: z2 as f32,
        factor_energy_sum: total as f32,
        residual_rel,
        passed: residual_rel <= PARSEVAL_TOL_REL && residual_rel.is_finite(),
    }
}

/// Reconstruction: `out = Σ_k coeffs[k]·basis[k]` (f64 accumulation). For
/// an orthonormal-complete basis and `coeffs` from [`parseval_energy_check`],
/// `out == z` — exactly, on dyadic Hadamard fixtures (Prop. 1).
pub fn recompose_into<const D: usize>(basis: &[[f32; D]], coeffs: &[f32], out: &mut [f32; D]) {
    assert_eq!(
        coeffs.len(),
        basis.len(),
        "recompose_into: coeffs.len() must equal basis.len()"
    );
    let mut acc = [0.0_f64; D];
    for (b, &c) in basis.iter().zip(coeffs.iter()) {
        let c64 = f64::from(c);
        for (j, o) in acc.iter_mut().enumerate() {
            *o += c64 * f64::from(b[j]);
        }
    }
    for (o, a) in out.iter_mut().zip(acc.iter()) {
        *o = *a as f32;
    }
}

/// Energy captured by the kept subset: `Σ_{k : kept[k]} coeffs[k]²`. With
/// Parseval (`Σ_k coeffs[k]² == ‖z‖²`), the EXACT truncation certificate:
/// dropped energy = `total − kept_energy(...)`, and
/// `‖z − recompose(kept)‖² == dropped` is an identity for orthonormal-complete
/// bases — not an approximation.
#[must_use]
pub fn kept_energy(coeffs: &[f32], kept: &[bool]) -> f32 {
    assert_eq!(coeffs.len(), kept.len(), "kept_energy: length mismatch");
    let mut s = 0.0_f64;
    for (&c, &k) in coeffs.iter().zip(kept.iter()) {
        if k {
            s += f64::from(c) * f64::from(c);
        }
    }
    s as f32
}

/// Walsh–Hadamard factor basis for `D = 2ⁿ` (natural/Sylvester order):
/// `out[i][j] = ±1/√D` with sign `(−1)^popcount(i & j)` — an
/// orthonormal-complete basis in O(D²) integer sign flips + one scale.
///
/// For `D = 4^m` (incl. the 64-dim `style_weights`/HLA latent) the scale is
/// dyadic (`1/8` at D=64) and every entry/intermediate is exact in f32 —
/// the integer-core cross-platform bit-identity witness for T3. For odd
/// exponents `1/√D` is irrational: correctly-rounded, deterministic, but
/// Parseval holds to rounding (not exactly 0).
///
/// # Panics
/// Panics if `D` is not a power of two.
pub fn hadamard_factorize<const D: usize>(out: &mut [[f32; D]]) {
    assert!(
        D.is_power_of_two(),
        "hadamard_factorize: D must be a power of two (got {D})"
    );
    let scale = 1.0 / (D as f32).sqrt();
    for (i, row) in out.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = if (i & j).count_ones() % 2 == 0 {
                scale
            } else {
                -scale
            };
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T4 — head conditioning certificate (via spectral_pencil)
// ──────────────────────────────────────────────────────────────────────────

/// Construction-time conditioning certificate over a set of factor heads.
/// `per_step_bound == sigma_max`: one composite step amplifies by at most
/// `max_k ‖W_k‖₂`; a T-step rollout by [`rollout_bound`] on it.
///
/// Orthonormal B ⇒ κ(B) = 1 by construction — only the heads remain to
/// certify. Commit σ_max/κ as metadata at construction (the paper's
/// autoregressive-rollout caveat, converted to a certificate).
#[derive(Debug, Clone, Copy)]
pub struct ConditioningCert {
    /// `max_k ‖W_k‖₂` — the worst head's amplification.
    pub sigma_max: f32,
    /// Index of the worst head.
    pub worst_head: usize,
    /// `== sigma_max` (the per-composite-step bound). Feed to
    /// [`rollout_bound`].
    pub per_step_bound: f32,
}

/// Per-head spectral norms `‖W_k‖₂ = √λ_max(W_kᵀW_k)` via the pinned Jacobi
/// eigensolver ([`crate::spectral_pencil::jacobi_eigen`]) on the
/// f64-accumulated Gram — exact (no power-iteration slack), deterministic,
/// construction-time. `heads` yields each head as its rows (`&[[f32; D]]`,
/// `D` = input dimension); `norms_out[k]` receives `‖W_k‖₂`.
///
/// Zero allocation: the D×D Gram is stack-built; `scratch` is the caller's
/// [`DenseScratch`]. An empty head (0 rows) has norm 0.
pub fn head_conditioning<'a, const D: usize, I>(
    heads: I,
    norms_out: &mut [f32],
    scratch: &mut DenseScratch<D>,
) -> ConditioningCert
where
    I: IntoIterator<Item = &'a [[f32; D]]>,
{
    let mut sigma_max = 0.0_f32;
    let mut worst_head = 0_usize;
    for (k, head_rows) in heads.into_iter().enumerate() {
        assert!(
            k < norms_out.len(),
            "head_conditioning: norms_out too small for head {k}"
        );
        // Gram = WᵀW, symmetric, f64 accumulation.
        let mut gram = [[0.0_f32; D]; D];
        for i in 0..D {
            for j in 0..=i {
                let mut s = 0.0_f64;
                for row in head_rows {
                    s += f64::from(row[i]) * f64::from(row[j]);
                }
                let v = s as f32;
                gram[i][j] = v;
                gram[j][i] = v;
            }
        }
        let report = jacobi_eigen(&gram, false, scratch);
        debug_assert!(
            report.converged,
            "head_conditioning: Jacobi did not converge"
        );
        // values sorted ascending; Gram is PSD ⇒ λ_max = values[D−1] ≥ 0.
        let lambda_max = scratch.values[D - 1].max(0.0);
        let norm = lambda_max.sqrt();
        norms_out[k] = norm;
        if norm > sigma_max {
            sigma_max = norm;
            worst_head = k;
        }
    }
    ConditioningCert {
        sigma_max,
        worst_head,
        per_step_bound: sigma_max,
    }
}

/// Composite rollout bound `Π_t max_k‖W_k‖₂ = per_step_bound^T` — the
/// worst-case amplification of a T-step autoregressive rollout through the
/// factorized heads.
#[must_use]
pub fn rollout_bound(per_step_bound: f32, steps: u32) -> f32 {
    per_step_bound.powi(steps as i32)
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic fixture directions (crate Rng — XorShift64, split-mixed).
    fn rng_dirs<const D: usize>(k: usize, seed: u64) -> Vec<[f32; D]> {
        let mut rng = crate::types::Rng::new(seed);
        (0..k)
            .map(|_| {
                let mut v = [0.0_f32; D];
                for s in v.iter_mut() {
                    *s = rng.normal();
                }
                v
            })
            .collect()
    }

    /// Max |cos| over nonzero-row pairs (zero rows skipped — GS rank spill).
    fn max_abs_pair_cos<const D: usize>(basis: &[[f32; D]]) -> f32 {
        let mut m = 0.0_f32;
        for i in 0..basis.len() {
            if dot_f64(&basis[i], &basis[i]) == 0.0 {
                continue;
            }
            for j in (i + 1)..basis.len() {
                if dot_f64(&basis[j], &basis[j]) == 0.0 {
                    continue;
                }
                m = m.max(dot_f64(&basis[i], &basis[j]).abs() as f32);
            }
        }
        m
    }

    /// Dyadic z (multiples of 0.25, |z| ≤ 4) — every Hadamard intermediate
    /// stays a dyadic rational with ≤ 22 significand bits: exact in f32/f64.
    fn dyadic_z<const D: usize>() -> [f32; D] {
        let mut z = [0.0_f32; D];
        for (j, s) in z.iter_mut().enumerate() {
            *s = (((j * 7) % 33) as f32 - 16.0) * 0.25;
        }
        z
    }

    #[test]
    fn gs_orthonormalizes_random_set() {
        let vectors = rng_dirs::<8>(5, 7);
        let mut out = [[0.0_f32; 8]; 5];
        let mut defect = 0.0_f32;
        orthonormalize_into(&vectors, &mut out, &mut defect);
        for row in &out {
            let n2 = dot_f64(row, row);
            assert!((n2 - 1.0).abs() < 1e-6, "norm² = {n2}");
        }
        assert!(max_abs_pair_cos(&out) < 1e-6);
        assert!(orthogonality_defect(&out) < 1e-10);
    }

    #[test]
    fn defect_closed_form_tiny() {
        let ortho = [[1.0_f32, 0.0], [0.0, 1.0]];
        assert_eq!(orthogonality_defect(&ortho), 0.0);
        let non_unit = [[2.0_f32, 0.0]];
        assert_eq!(orthogonality_defect(&non_unit), 9.0); // (4−1)²
        let parallel = [[1.0_f32, 0.0], [1.0, 0.0]];
        assert_eq!(orthogonality_defect(&parallel), 1.0); // (b1·b2)² = 1
        let short = [[0.5_f32, 0.0]];
        assert_eq!(orthogonality_defect(&short), 0.5625); // (0.25−1)²
    }

    #[test]
    fn empty_set_noop() {
        let empty: [[f32; 4]; 0] = [];
        assert_eq!(orthogonality_defect(&empty), 0.0);
        let mut d = 1.0_f32;
        orthonormalize_into(&empty, &mut [], &mut d);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn gs_zeroes_exact_duplicate() {
        let v = rng_dirs::<8>(1, 3);
        let vectors = [v[0], v[0]];
        let mut out = [[0.0_f32; 8]; 2];
        let mut defect = 0.0_f32;
        orthonormalize_into(&vectors, &mut out, &mut defect);
        assert_eq!(dot_f64(&out[1], &out[1]), 0.0, "duplicate row zeroed");
        assert!((dot_f64(&out[0], &out[0]) - 1.0).abs() < 1e-6);
        assert!(defect >= 1.0, "duplicate must fire the input defect");
    }

    #[test]
    fn gs_zeroes_overflow_beyond_rank() {
        let vectors = rng_dirs::<4>(6, 11);
        let mut out = [[0.0_f32; 4]; 6];
        let mut defect = 0.0_f32;
        orthonormalize_into(&vectors, &mut out, &mut defect);
        let zero_rows = out.iter().filter(|r| dot_f64(r, r) == 0.0).count();
        assert!(zero_rows >= 2, "6 vectors in d=4 ⇒ ≥2 rank-spill rows");
        assert!(defect > 0.1);
        assert!(max_abs_pair_cos(&out) < 1e-6);
    }

    #[test]
    fn gs_preserves_span_complete_basis() {
        let vectors = rng_dirs::<8>(8, 21);
        let mut basis = [[0.0_f32; 8]; 8];
        let mut defect = 0.0_f32;
        orthonormalize_into(&vectors, &mut basis, &mut defect);
        let z = dyadic_z::<8>();
        let mut coeffs = [0.0_f32; 8];
        let rep = parseval_energy_check(&z, &basis, &mut coeffs);
        assert!(rep.passed, "residual_rel = {}", rep.residual_rel);
        let mut rec = [0.0_f32; 8];
        recompose_into(&basis, &coeffs, &mut rec);
        for (a, b) in rec.iter().zip(z.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn hadamard_defect_zero_at_d64_tiny_at_d8() {
        let mut h8 = [[0.0_f32; 8]; 8];
        hadamard_factorize(&mut h8);
        assert!(orthogonality_defect(&h8) < 1e-12); // 1/√8 rounds
        let mut h64 = [[0.0_f32; 64]; 64];
        hadamard_factorize(&mut h64);
        assert_eq!(orthogonality_defect(&h64), 0.0); // 0.125 dyadic EXACT
    }

    #[test]
    fn parseval_exact_and_recompose_bit_identical_at_d64() {
        let mut basis = [[0.0_f32; 64]; 64];
        hadamard_factorize(&mut basis);
        let z = dyadic_z::<64>();
        let mut coeffs = [0.0_f32; 64];
        let rep = parseval_energy_check(&z, &basis, &mut coeffs);
        assert_eq!(rep.residual_rel, 0.0, "dyadic witness must be EXACT");
        assert!(rep.passed);
        let mut rec = [0.0_f32; 64];
        recompose_into(&basis, &coeffs, &mut rec);
        for (a, b) in rec.iter().zip(z.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "recompose must be bit-exact");
        }
    }

    #[test]
    fn parseval_catches_duplicate_and_incomplete() {
        let mut basis = [[0.0_f32; 4]; 4];
        hadamard_factorize(&mut basis);
        // Duplicate a row: the energy double-counts along b0.
        let mut dup = basis;
        dup[1] = dup[0];
        let z = [0.5_f32; 4];
        let mut coeffs = [0.0_f32; 4];
        let rep = parseval_energy_check(&z, &dup, &mut coeffs);
        assert!(!rep.passed);
        assert!(rep.residual_rel > 0.1);
        // Incomplete basis (K = D−1) with z in the dropped direction.
        let incomplete = &basis[..3];
        let mut c2 = [0.0_f32; 3];
        let rep2 = parseval_energy_check(&basis[3], incomplete, &mut c2);
        assert!(!rep2.passed);
        assert!((rep2.residual_rel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn truncation_certificate_identity() {
        let mut basis = [[0.0_f32; 64]; 64];
        hadamard_factorize(&mut basis);
        let z = dyadic_z::<64>();
        let mut coeffs = [0.0_f32; 64];
        let rep = parseval_energy_check(&z, &basis, &mut coeffs);
        // Keep the 32 largest-|coeff| factors; drop the rest.
        let mut order: Vec<usize> = (0..64).collect();
        order.sort_by(|&a, &b| coeffs[b].abs().total_cmp(&coeffs[a].abs()));
        let mut kept = [false; 64];
        for &k in &order[..32] {
            kept[k] = true;
        }
        let kept_e = kept_energy(&coeffs, &kept);
        let dropped = rep.factor_energy_sum - kept_e;
        // ‖z − recompose(kept)‖² must equal dropped — the Parseval identity
        // that makes truncation budgets exact rather than approximate.
        let mut masked = [0.0_f32; 64];
        for (m, (kc, &keep)) in masked.iter_mut().zip(coeffs.iter().zip(kept.iter())) {
            if keep {
                *m = *kc;
            }
        }
        let mut rec = [0.0_f32; 64];
        recompose_into(&basis, &masked, &mut rec);
        let mut resid2 = 0.0_f64;
        for j in 0..64 {
            let d = f64::from(z[j] - rec[j]);
            resid2 += d * d;
        }
        assert!(
            (resid2 as f32 - dropped).abs() <= 1e-6 * rep.z_norm_sq + 1e-9,
            "truncation identity: ‖z−rec‖² = {resid2} vs dropped = {dropped}"
        );
    }

    /// G8a: planted near-parallel pair ⇒ input defect fires + GS decorrelates.
    #[test]
    fn defect_fires_on_planted_near_parallel_pair() {
        // Healthy control: an orthonormalized 14-set (defect ~0).
        let base = rng_dirs::<64>(14, 42);
        let mut healthy = [[0.0_f32; 64]; 14];
        let mut d0 = 0.0_f32;
        orthonormalize_into(&base, &mut healthy, &mut d0);
        let d_healthy = orthogonality_defect(&healthy);

        // Planted: replace row 13 with a near-copy of row 0.
        let mut rng = crate::types::Rng::new(43);
        let mut w = healthy[0];
        for s in w.iter_mut() {
            *s += 0.01 * rng.normal();
        }
        let n = dot_f64(&w, &w).sqrt();
        for s in w.iter_mut() {
            *s = (f64::from(*s) / n) as f32;
        }
        let mut planted = healthy;
        planted[13] = w;
        let d_planted = orthogonality_defect(&planted);

        assert!(d_healthy < 1e-6);
        assert!(d_planted > 0.5, "planted pair must fire (got {d_planted})");
        assert!(d_planted > 100.0 * d_healthy);

        // GS decorrelates: output pairwise |cos| < 1e-6, survivor intact.
        let mut out = [[0.0_f32; 64]; 14];
        let mut defect = 0.0_f32;
        orthonormalize_into(&planted, &mut out, &mut defect);
        assert!(max_abs_pair_cos(&out) < 1e-6);
        assert!(
            (dot_f64(&out[13], &out[13]) - 1.0).abs() < 1e-6,
            "not zeroed"
        );
    }

    #[test]
    fn gamma_schedule_values() {
        assert_eq!(gamma_schedule(0.25, 1.0, 0), 1.0);
        assert_eq!(gamma_schedule(0.25, 1.0, 1), 1.0);
        assert_eq!(gamma_schedule(0.25, 1.0, 4), 0.5);
        assert_eq!(gamma_schedule(0.25, 1.0, 16), 0.25);
        assert_eq!(gamma_schedule(0.25, 1.0, 1_000_000), 0.25);
        assert_eq!(gamma_schedule(0.9, 1.0, 100), 0.9); // γ_min dominates
    }

    #[test]
    fn hinge_matches_two_pass_variance() {
        let mut rng = crate::types::Rng::new(99);
        let (k, r) = (2_usize, 3_usize);
        let mut scratch = FactorActivityScratch::new(k, r);
        let mut samples = vec![0.0_f32; 100 * k * r];
        for s in samples.chunks_mut(k * r) {
            for (idx, v) in s.iter_mut().enumerate() {
                // Factor 1's coords live at σ ≈ 0.05 — below γ — so both
                // hinge arms (firing and not firing) are exercised.
                let scale = if idx / r == 1 { 0.05 } else { 1.0 };
                *v = rng.normal() * scale;
            }
            scratch.observe_sample(s);
        }
        let gamma = gamma_schedule(GAMMA_FAC_MIN, GAMMA_SCHED_C, scratch.count());
        let mut per = [0.0_f32; 6];
        let rep = factor_activity_hinge(&scratch, gamma, &mut per);
        assert_eq!(rep.n, 100);

        // Direct two-pass reference (f64 on the same f32 data).
        let n = 100.0_f64;
        for idx in 0..6 {
            let col: Vec<f64> = (0..100).map(|i| f64::from(samples[i * 6 + idx])).collect();
            let mean = col.iter().sum::<f64>() / n;
            let var = col.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
            let want = ((f64::from(gamma) - var.sqrt()).max(0.0)) as f32;
            assert!(
                (per[idx] - want).abs() < 1e-5,
                "idx {idx}: {} vs {want}",
                per[idx]
            );
        }
        let mean_hinge = per.iter().sum::<f32>() / 6.0;
        assert!((rep.mean_hinge - mean_hinge).abs() < 1e-6);
        // Factor 1 fires (σ̂ ≈ 0.05 < γ = 0.25); factor 0 does not.
        assert!(per.iter().skip(3).all(|&h| h > 0.1));
        assert!(per.iter().take(3).all(|&h| h == 0.0));
    }

    /// G8b: planted dead channel ⇒ hinge fires EXACTLY on that coordinate.
    #[test]
    fn hinge_dead_channel_exactly_gamma() {
        let mut rng = crate::types::Rng::new(7);
        let (k, r) = (3_usize, 2_usize);
        let mut scratch = FactorActivityScratch::new(k, r);
        for _ in 0..64 {
            let mut s = [0.0_f32; 6];
            for (idx, v) in s.iter_mut().enumerate() {
                *v = if idx == r { 7.0 } else { rng.normal() };
            }
            scratch.observe_sample(&s);
        }
        let gamma = gamma_schedule(GAMMA_FAC_MIN, GAMMA_SCHED_C, 64);
        assert_eq!(gamma, 0.25);
        let mut per = [0.0_f32; 6];
        let rep = factor_activity_hinge(&scratch, gamma, &mut per);
        // Dead coordinate: m2 == 0.0 exactly in Welford ⇒ hinge EXACTLY γ.
        assert_eq!(per[2].to_bits(), 0.25_f32.to_bits());
        assert_eq!(rep.worst_flat, 2);
        assert_eq!(rep.worst_hinge.to_bits(), 0.25_f32.to_bits());
        // Healthy coordinates (σ ≈ 1 ≫ γ): hinge exactly 0.
        for (idx, &h) in per.iter().enumerate() {
            if idx != 2 {
                assert_eq!(h, 0.0, "healthy idx {idx} fired");
            }
        }
        assert!((rep.mean_hinge - 0.25 / 6.0).abs() < 1e-7);
    }

    #[test]
    fn hinge_no_claim_below_two_samples() {
        let mut scratch = FactorActivityScratch::new(2, 2);
        let mut per = [1.0_f32; 4];
        let rep = factor_activity_hinge(&scratch, 0.5, &mut per);
        assert_eq!(rep.n, 0);
        assert_eq!(rep.mean_hinge, 0.0);
        assert!(per.iter().all(|&h| h == 0.0));
        scratch.observe_sample(&[1.0, 2.0, 3.0, 4.0]);
        let mut per2 = [1.0_f32; 4];
        let rep2 = factor_activity_hinge(&scratch, 0.5, &mut per2);
        assert_eq!(rep2.n, 1);
        assert!(per2.iter().all(|&h| h == 0.0));
    }

    #[test]
    fn head_norm_diag_exact() {
        let head = [[3.0_f32, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.5]];
        let mut scratch = DenseScratch::new();
        let mut norms = [0.0_f32; 1];
        let cert = head_conditioning([&head[..]], &mut norms, &mut scratch);
        assert!((norms[0] - 4.0).abs() < 1e-6);
        assert_eq!(cert.sigma_max, norms[0]);
        assert_eq!(cert.worst_head, 0);
    }

    #[test]
    fn head_norm_rank_one() {
        // W = u vᵀ ⇒ ‖W‖₂ = ‖u‖·‖v‖.
        let u = [1.0_f32, 2.0, 3.0, 0.0];
        let v = [2.0_f32, 0.0, 0.0, 1.0];
        let head: Vec<[f32; 4]> = u
            .iter()
            .map(|&ui| [ui * v[0], ui * v[1], ui * v[2], ui * v[3]])
            .collect();
        let un: f64 = u
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        let vn: f64 = v
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        let mut scratch = DenseScratch::new();
        let mut norms = [0.0_f32; 1];
        let _ = head_conditioning([&head[..]], &mut norms, &mut scratch);
        assert!((f64::from(norms[0]) - un * vn).abs() < 1e-3);
    }

    #[test]
    fn head_norm_orthonormal_is_exactly_one() {
        let mut head = [[0.0_f32; 64]; 64];
        hadamard_factorize(&mut head);
        let mut scratch = DenseScratch::new();
        let mut norms = [0.0_f32; 1];
        let _ = head_conditioning([&head[..]], &mut norms, &mut scratch);
        assert_eq!(norms[0], 1.0); // Gram == I exactly (dyadic)
    }

    #[test]
    fn conditioning_cert_and_rollout_bound() {
        let h0 = [[3.0_f32, 0.0], [0.0, 4.0]];
        let h1 = [[2.0_f32, 0.0], [0.0, 2.0]];
        let h2 = [[9.0_f32, 0.0], [0.0, 1.0]];
        let mut scratch = DenseScratch::new();
        let mut norms = [0.0_f32; 3];
        let cert = head_conditioning([&h0[..], &h1[..], &h2[..]], &mut norms, &mut scratch);
        assert!((norms[0] - 4.0).abs() < 1e-6);
        assert!((norms[1] - 2.0).abs() < 1e-6);
        assert!((norms[2] - 9.0).abs() < 1e-6);
        assert_eq!(cert.worst_head, 2);
        assert!((cert.sigma_max - 9.0).abs() < 1e-6);
        assert!((cert.per_step_bound - 9.0).abs() < 1e-6);
        assert_eq!(rollout_bound(2.0, 10), 1024.0);
    }

    /// G4: zero allocations in steady state (the gaussianity pattern — the
    /// lib test binary installs `alloc::TrackingAllocator` under
    /// cfg(test, debug_assertions); skip with a message if absent).
    // See the note on `latent_confounder_audit`'s twin: `crate::alloc` is
    // `#[cfg(debug_assertions)]` by design, so in release these imports do not
    // resolve and the whole lib-test target fails to compile (Issue 716). The
    // doc comment above says "skip with a message if absent", which was the
    // intent — but an unconditional `use` is a compile error, not a skip.
    #[cfg(debug_assertions)]
    #[test]
    fn g4_zero_alloc_steady_state() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};

        // Fixtures pre-built (their allocs don't count).
        let vectors = rng_dirs::<8>(5, 5);
        let mut out = [[0.0_f32; 8]; 5];
        let mut defect = 0.0_f32;
        let mut activity = FactorActivityScratch::new(2, 3);
        let sample = [0.5_f32; 6];
        let mut per = [0.0_f32; 6];
        let mut basis = [[0.0_f32; 8]; 8];
        hadamard_factorize(&mut basis);
        let z = dyadic_z::<8>();
        let mut coeffs = [0.0_f32; 8];
        let mut rec = [0.0_f32; 8];
        let kept = [true; 8];
        let head = basis;
        let mut norms = [0.0_f32; 1];
        let mut dense = DenseScratch::new();

        // Sentinel: confirm the allocator is installed.
        reset_alloc_stats();
        let _sentinel: Vec<u8> = vec![0u8; 256];
        let (sent_count, _) = get_alloc_stats();
        if sent_count == 0 {
            eprintln!("g4_zero_alloc_steady_state: TrackingAllocator not installed — SKIPPED");
            return;
        }
        drop(_sentinel);

        // Warmup.
        orthonormalize_into(&vectors, &mut out, &mut defect);
        activity.observe_sample(&sample);
        let _ = factor_activity_hinge(&activity, 0.25, &mut per);
        let _ = parseval_energy_check(&z, &basis, &mut coeffs);
        recompose_into(&basis, &coeffs, &mut rec);
        let _ = kept_energy(&coeffs, &kept);
        hadamard_factorize(&mut basis);
        let _ = head_conditioning([&head[..]], &mut norms, &mut dense);

        reset_alloc_stats();
        for _ in 0..100 {
            orthonormalize_into(&vectors, &mut out, &mut defect);
            activity.observe_sample(&sample);
            let _ = factor_activity_hinge(&activity, 0.25, &mut per);
            let _ = parseval_energy_check(&z, &basis, &mut coeffs);
            recompose_into(&basis, &coeffs, &mut rec);
            let _ = kept_energy(&coeffs, &kept);
            hadamard_factorize(&mut basis);
            let _ = head_conditioning([&head[..]], &mut norms, &mut dense);
        }
        let (count, bytes) = get_alloc_stats();
        assert_eq!(
            count, 0,
            "steady-state surface must be alloc-free; observed {count} allocations ({bytes} bytes)"
        );
    }

    #[test]
    #[should_panic(expected = "out.len()")]
    fn gs_shape_mismatch_panics() {
        let vectors = rng_dirs::<4>(3, 1);
        let mut out = [[0.0_f32; 4]; 2];
        let mut defect = 0.0_f32;
        orthonormalize_into(&vectors, &mut out, &mut defect);
    }

    #[test]
    #[should_panic(expected = "observe_sample")]
    fn observe_shape_mismatch_panics() {
        let mut scratch = FactorActivityScratch::new(2, 3);
        scratch.observe_sample(&[0.0; 5]);
    }

    #[test]
    #[should_panic(expected = "per_coord_out.len()")]
    fn hinge_shape_mismatch_panics() {
        let scratch = FactorActivityScratch::new(2, 3);
        let mut per = [0.0_f32; 5];
        let _ = factor_activity_hinge(&scratch, 0.25, &mut per);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn hadamard_non_power_of_two_panics() {
        let mut out = [[0.0_f32; 6]; 6];
        hadamard_factorize(&mut out);
    }
}
