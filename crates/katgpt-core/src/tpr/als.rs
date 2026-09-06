//! Ridge-ALS fit for the TPR binding algebra (Issue 707 T5) — the offline
//! calibration that produces a frozen [`TprArtifact`].
//!
//! **No gradient descent.** Every block is a closed-form ridge solve; the
//! sweep is block-coordinate descent (Gauss-Seidel) over four blocks:
//!
//! 1. **`W, b`** — one ridge normal-equation solve against the current cores,
//!    reusing [`crate::linalg::ridge_solve_direct_f32`] (the KARC kernel).
//! 2. **free cores** — `ĉ_n = (WᵀW + λI)⁻¹Wᵀ(e_n − b)`, ONE Cholesky shared
//!    across all `N` states (the projection factor, cached into the artifact).
//! 3. **fillers** — per-filler `d`-vector solve against the free cores; the
//!    role weights enter only as the scalar `Σ‖r_p‖²`, so the normal matrix
//!    is diagonal and the solve is exact.
//! 4. **roles** — the transpose of block 3 (per-slot `m`-vector), skipped for
//!    [`TprScheme::Orthogonal`] where the roles are fixed one-hots.
//!
//! Blocks 1–2 minimize the state-space objective exactly; blocks 3–4 minimize
//! the core-space surrogate. That composition is *empirically* monotone, not
//! provably so, which is why [`AlsReport::monotone_violations`] is a COUNTER
//! rather than an assert — a fit that violates monotonicity is reported, not
//! silently accepted (Issue 707 T-G1 reads this field).
//!
//! Determinism: fixed iteration order, no rayon, no hashing, a SplitMix64
//! filler init from [`AlsConfig::lcg_seed`]. Two fits of the same input +
//! config produce **bit-identical** artifacts (the T-G1 double-run gate).

use super::types::{
    AlsConfig, AlsInput, AlsReport, L21, SCHEMA_VERSION, TPR_MAX_PROJECTION_K, TprArtifact,
    TprError, TprScheme,
};
use crate::linalg::{cholesky_f32, chol_solve_f32, ridge_solve_direct_f32, spd_inverse_f32};

/// The mutable ALS state one sweep can move, in the order `als_fit`'s
/// destructuring assignment expects: `(fillers, scheme, w, bias, cores)`.
///
/// A named alias rather than the tuple inline, so the best-iterate guard's
/// store and restore cannot drift out of agreement on the field order —
/// every element is a `Vec<f32>` except one, so a transposition would
/// compile and silently restore the wrong block (Issue 712).
type AlsIterate = (Vec<f32>, TprScheme, Vec<f32>, Vec<f32>, Vec<f32>);

/// Minimum projection ridge — guarantees `WᵀW + λI` is positive definite even
/// when the caller asked for `ridge_lambda = 0`.
const MIN_PROJECTION_LAMBDA: f32 = 1e-6;

/// House deterministic RNG (the same SplitMix64 used across katgpt-core test
/// fixtures) — used ONLY for the filler-table initialization.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[-1, 1)`.
    fn next_sym(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        u.mul_add(2.0, -1.0)
    }
}

/// Read `W[i, coord]` from the block-major encoder layout (see
/// [`TprArtifact::w`]). The fit is offline, so it reads through this helper
/// rather than duplicating the runtime path's block-outer loops.
#[inline]
fn w_at(w: &[f32], dim: usize, d: usize, i: usize, coord: usize) -> f32 {
    w[(coord / d) * dim * d + (coord % d) * dim + i]
}

#[inline]
fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Role weights over the `m` core blocks, materialized for the fit (the
/// runtime path avoids this copy; the fit is offline).
fn role_row(scheme: &TprScheme, p: usize, m: usize, out: &mut [f32]) {
    out.fill(0.0);
    match scheme {
        TprScheme::Orthogonal { .. } => out[p] = 1.0,
        TprScheme::RoleVectors { roles, .. } => {
            out.copy_from_slice(&roles[p * m..(p + 1) * m]);
        }
    }
}

/// `core += sign · (r ⊗ f)`.
#[inline]
fn axpy_bind(core: &mut [f32], r: &[f32], f: &[f32], d: usize, sign: f32) {
    for (blk, &rw) in r.iter().enumerate() {
        match rw == 0.0 {
            true => continue,
            false => {
                let w = sign * rw;
                let off = blk * d;
                for (j, &fv) in f.iter().enumerate() {
                    core[off + j] = w.mul_add(fv, core[off + j]);
                }
            }
        }
    }
}

/// Build every state's constrained core `c_n = Σ_j r_{p_j} ⊗ f_{v_j}`.
fn build_cores(
    input: &AlsInput<'_>,
    scheme: &TprScheme,
    fillers: &[f32],
    d: usize,
    m: usize,
    row: &mut [f32],
    cores: &mut [f32],
) {
    let k = m * d;
    cores.fill(0.0);
    for (s, b) in input.bindings.iter().enumerate() {
        let core = &mut cores[s * k..(s + 1) * k];
        for (&p, &v) in b.roles.iter().zip(b.fillers.iter()) {
            role_row(scheme, p as usize, m, row);
            let f = &fillers[v as usize * d..(v as usize + 1) * d];
            axpy_bind(core, row, f, d, 1.0);
        }
    }
}

/// Block 1 — exact ridge solve of `[W b]` against the current cores.
///
/// `X = [cores | 1]` (`N × (K+1)`); the ridge penalizes the `K` weight columns
/// but NOT the intercept, which is the standard centering-equivalent form.
#[allow(clippy::too_many_arguments)]
fn solve_wb(
    states: &[f32],
    cores: &[f32],
    n: usize,
    k: usize,
    d: usize,
    dim: usize,
    lambda: f32,
    gram: &mut [f32],
    cov: &mut [f32],
    wt: &mut [f32],
    l_scratch: &mut [f32],
    z_scratch: &mut [f32],
    w: &mut [f32],
    bias: &mut [f32],
) {
    let p = k + 1;
    gram[..p * p].fill(0.0);
    cov[..p * dim].fill(0.0);
    for s in 0..n {
        let c = &cores[s * k..(s + 1) * k];
        let e = &states[s * dim..(s + 1) * dim];
        // Gram: upper-left K×K from c·cᵀ, last row/col from the intercept.
        for a in 0..k {
            let ca = c[a];
            match ca == 0.0 {
                true => {}
                false => {
                    let row = a * p;
                    for b in 0..k {
                        gram[row + b] = ca.mul_add(c[b], gram[row + b]);
                    }
                    gram[row + k] += ca;
                    gram[k * p + a] += ca;
                }
            }
        }
        gram[k * p + k] += 1.0;
        // Cross-covariance XᵀE.
        for (a, &ca) in c.iter().enumerate() {
            match ca == 0.0 {
                true => {}
                false => {
                    let row = a * dim;
                    for (i, &ev) in e.iter().enumerate() {
                        cov[row + i] = ca.mul_add(ev, cov[row + i]);
                    }
                }
            }
        }
        let row = k * dim;
        for (i, &ev) in e.iter().enumerate() {
            cov[row + i] += ev;
        }
    }
    for a in 0..k {
        gram[a * p + a] += lambda;
    }
    ridge_solve_direct_f32(
        &mut wt[..p * dim],
        l_scratch,
        z_scratch,
        &gram[..p * p],
        &cov[..p * dim],
        p,
        dim,
    );
    for i in 0..dim {
        bias[i] = wt[k * dim + i];
    }
    for j in 0..k {
        // Core coord `j` is column `j % d` of block `j / d` — a contiguous
        // length-`D` run in the column-slice layout.
        let base = (j / d) * dim * d + (j % d) * dim;
        w[base..base + dim].copy_from_slice(&wt[j * dim..(j + 1) * dim]);
    }
}

/// Σ‖e_n − (W·c_n + b)‖² over the corpus.
// (states, cores, n, k, dim, per_state) is the intrinsic shape of a
// row-major SSR over a flat corpus — bundling it into a struct would only
// move the same eight values behind a constructor.
#[allow(clippy::too_many_arguments)]
fn state_ssr(
    states: &[f32],
    cores: &[f32],
    w: &[f32],
    bias: &[f32],
    n: usize,
    k: usize,
    d: usize,
    dim: usize,
    per_state: Option<&mut [f32]>,
) -> f64 {
    let mut total = 0.0f64;
    let mut per = per_state;
    for s in 0..n {
        let c = &cores[s * k..(s + 1) * k];
        let e = &states[s * dim..(s + 1) * dim];
        let mut ss = 0.0f64;
        for i in 0..dim {
            let mut acc = bias[i];
            for (j, &cv) in c.iter().enumerate() {
                acc = w_at(w, dim, d, i, j).mul_add(cv, acc);
            }
            let dv = (e[i] - acc) as f64;
            ss += dv * dv;
        }
        if let Some(p) = per.as_deref_mut() {
            p[s] = (ss as f32).sqrt();
        }
        total += ss;
    }
    total
}

/// Block 2 — free cores `Ĉ = (WᵀW + λI)⁻¹Wᵀ(E − b)`, one shared Cholesky.
#[allow(clippy::too_many_arguments)]
fn free_cores(
    states: &[f32],
    w: &[f32],
    bias: &[f32],
    n: usize,
    k: usize,
    d: usize,
    dim: usize,
    lambda: f32,
    gram: &mut [f32],
    l: &mut [f32],
    rhs: &mut [f32],
    z: &mut [f32],
    out: &mut [f32],
) -> Result<(), TprError> {
    gram[..k * k].fill(0.0);
    for i in 0..dim {
        for a in 0..k {
            let wa = w_at(w, dim, d, i, a);
            match wa == 0.0 {
                true => {}
                false => {
                    let row = a * k;
                    for b in 0..k {
                        gram[row + b] = wa.mul_add(w_at(w, dim, d, i, b), gram[row + b]);
                    }
                }
            }
        }
    }
    for a in 0..k {
        gram[a * k + a] += lambda.max(MIN_PROJECTION_LAMBDA);
    }
    if !all_finite(&gram[..k * k]) {
        return Err(TprError::NonFinite("free-core Gram"));
    }
    // RHS is k × n row-major: rhs[j*n + s] = Wᵀ(e_s − b)[j].
    rhs[..k * n].fill(0.0);
    for s in 0..n {
        let e = &states[s * dim..(s + 1) * dim];
        for i in 0..dim {
            let sv = e[i] - bias[i];
            match sv == 0.0 {
                true => {}
                false => {
                    for j in 0..k {
                        rhs[j * n + s] = w_at(w, dim, d, i, j).mul_add(sv, rhs[j * n + s]);
                    }
                }
            }
        }
    }
    cholesky_f32(l, &gram[..k * k], k);
    chol_solve_f32(&mut out[..k * n], &mut z[..k * n], l, &rhs[..k * n], k, n);
    match all_finite(&out[..k * n]) {
        true => Ok(()),
        false => Err(TprError::NonFinite("free cores")),
    }
}

/// Per-filler and per-role occurrence indices, built once per fit.
struct Occurrences {
    /// `by_filler[v]` = `(state, role_slot)` pairs.
    by_filler: Vec<Vec<(u32, u16)>>,
    /// `by_role[p]` = `(state, filler)` pairs.
    by_role: Vec<Vec<(u32, u16)>>,
}

impl Occurrences {
    fn build(input: &AlsInput<'_>, n_fillers: usize, n_slots: usize) -> Self {
        let mut by_filler = vec![Vec::new(); n_fillers];
        let mut by_role = vec![Vec::new(); n_slots];
        for (s, b) in input.bindings.iter().enumerate() {
            for (&p, &v) in b.roles.iter().zip(b.fillers.iter()) {
                by_filler[v as usize].push((s as u32, p));
                by_role[p as usize].push((s as u32, v));
            }
        }
        Self { by_filler, by_role }
    }
}

/// Working residual `resid[s] = ĉ_s − c_s(θ)` (core space).
fn residual_from(chat: &[f32], cores: &[f32], n: usize, k: usize, resid: &mut [f32]) {
    for i in 0..n * k {
        resid[i] = chat[i] - cores[i];
    }
}

/// Block 3 — per-filler exact solve against the free cores.
#[allow(clippy::too_many_arguments)]
fn filler_block(
    occ: &Occurrences,
    scheme: &TprScheme,
    resid: &mut [f32],
    fillers: &mut [f32],
    d: usize,
    m: usize,
    ridge: f32,
    l21_weights: Option<&[f32]>,
    row: &mut [f32],
    num: &mut [f32],
) {
    let k = m * d;
    for (v, occs) in occ.by_filler.iter().enumerate() {
        if occs.is_empty() {
            continue;
        }
        let f_old: Vec<f32> = fillers[v * d..(v + 1) * d].to_vec();
        // Add every occurrence back into the residual.
        for &(s, p) in occs {
            role_row(scheme, p as usize, m, row);
            let core = &mut resid[s as usize * k..(s as usize + 1) * k];
            axpy_bind(core, row, &f_old, d, 1.0);
        }
        num[..d].fill(0.0);
        let mut den = ridge;
        for &(s, p) in occs {
            role_row(scheme, p as usize, m, row);
            let core = &resid[s as usize * k..(s as usize + 1) * k];
            for (blk, &rw) in row.iter().enumerate() {
                match rw == 0.0 {
                    true => continue,
                    false => {
                        let off = blk * d;
                        for j in 0..d {
                            num[j] = rw.mul_add(core[off + j], num[j]);
                        }
                        den = rw.mul_add(rw, den);
                    }
                }
            }
        }
        for j in 0..d {
            let dj = match l21_weights {
                Some(wts) => den + wts[j],
                None => den,
            };
            fillers[v * d + j] = match dj > 0.0 {
                true => num[j] / dj,
                false => 0.0,
            };
        }
        let f_new: Vec<f32> = fillers[v * d..(v + 1) * d].to_vec();
        for &(s, p) in occs {
            role_row(scheme, p as usize, m, row);
            let core = &mut resid[s as usize * k..(s as usize + 1) * k];
            axpy_bind(core, row, &f_new, d, -1.0);
        }
    }
}

/// Block 4 — per-role-slot exact solve (role-vector schemes only).
// Mirror of `filler_block`'s signature (occurrences, scheme, residual,
// counterpart table, geometry, ridge, two scratch buffers) — kept parallel on
// purpose so the transpose symmetry of the two blocks stays readable.
#[allow(clippy::too_many_arguments)]
fn role_block(
    occ: &Occurrences,
    scheme: &mut TprScheme,
    resid: &mut [f32],
    fillers: &[f32],
    d: usize,
    m: usize,
    ridge: f32,
    row: &mut [f32],
    num: &mut [f32],
) {
    if matches!(scheme, TprScheme::Orthogonal { .. }) {
        return;
    }
    let k = m * d;
    for (p, occs) in occ.by_role.iter().enumerate() {
        if occs.is_empty() {
            continue;
        }
        role_row(scheme, p, m, row);
        let r_old: Vec<f32> = row[..m].to_vec();
        for &(s, v) in occs {
            let f = &fillers[v as usize * d..(v as usize + 1) * d];
            let core = &mut resid[s as usize * k..(s as usize + 1) * k];
            axpy_bind(core, &r_old, f, d, 1.0);
        }
        num[..m].fill(0.0);
        let mut den = ridge;
        for &(s, v) in occs {
            let f = &fillers[v as usize * d..(v as usize + 1) * d];
            let core = &resid[s as usize * k..(s as usize + 1) * k];
            for (blk, nb) in num[..m].iter_mut().enumerate() {
                let off = blk * d;
                let mut acc = 0.0f32;
                for j in 0..d {
                    acc = core[off + j].mul_add(f[j], acc);
                }
                *nb += acc;
            }
            for &fv in f {
                den = fv.mul_add(fv, den);
            }
        }
        let r_new: Vec<f32> = (0..m)
            .map(|i| match den > 0.0 {
                true => num[i] / den,
                false => 0.0,
            })
            .collect();
        scheme.set_role_vec(p, &r_new);
        for &(s, v) in occs {
            let f = &fillers[v as usize * d..(v as usize + 1) * d];
            let core = &mut resid[s as usize * k..(s as usize + 1) * k];
            axpy_bind(core, &r_new, f, d, -1.0);
        }
    }
}

/// Pivoted modified Gram-Schmidt over the fitted role vectors → the signed,
/// slot-permuted orthonormal unbind basis, plus `(μ, min diag)`.
///
/// Pivoting on the largest remaining residual norm makes the basis
/// deterministic and well-conditioned: the slot whose role is *most*
/// independent of the already-chosen directions is orthonormalized first, so
/// a near-degenerate slot never sets the scale.
fn unbind_basis(roles: &[f32], n_slots: usize, m: usize) -> (Vec<f32>, f32, f32) {
    let mut work: Vec<f32> = roles[..n_slots * m].to_vec();
    let mut basis = vec![0.0f32; n_slots * m];
    let mut done = vec![false; n_slots];
    for _ in 0..n_slots {
        // Pivot: largest remaining residual norm (ties → lowest slot id).
        let mut best = usize::MAX;
        let mut best_n2 = -1.0f32;
        for p in 0..n_slots {
            if done[p] {
                continue;
            }
            let n2: f32 = work[p * m..(p + 1) * m].iter().map(|v| v * v).sum();
            if n2 > best_n2 {
                best_n2 = n2;
                best = p;
            }
        }
        if best == usize::MAX {
            break;
        }
        let norm = best_n2.max(0.0).sqrt();
        let q: Vec<f32> = match norm > 1e-12 {
            true => work[best * m..(best + 1) * m]
                .iter()
                .map(|v| v / norm)
                .collect(),
            // Degenerate slot: fall back to the canonical block direction so
            // the basis stays defined (and the coherence reports the damage).
            false => (0..m)
                .map(|i| match i == best % m {
                    true => 1.0,
                    false => 0.0,
                })
                .collect(),
        };
        basis[best * m..(best + 1) * m].copy_from_slice(&q);
        done[best] = true;
        for p in 0..n_slots {
            if done[p] {
                continue;
            }
            let proj: f32 = (0..m).map(|i| work[p * m + i] * q[i]).sum();
            for i in 0..m {
                work[p * m + i] -= proj * q[i];
            }
        }
    }
    // μ over UNIT-NORMALIZED roles (a true coherence in [0,1]); diag over the
    // raw roles (the recovery scale the unbind inherits).
    let mut mu = 0.0f32;
    let mut diag_min = f32::INFINITY;
    for p in 0..n_slots {
        let bp = &basis[p * m..(p + 1) * m];
        let dp: f32 = (0..m).map(|i| roles[p * m + i] * bp[i]).sum();
        diag_min = diag_min.min(dp.abs());
        for q in 0..n_slots {
            if q == p {
                continue;
            }
            let rq = &roles[q * m..(q + 1) * m];
            let nq: f32 = rq.iter().map(|v| v * v).sum::<f32>().sqrt();
            if nq > 1e-12 {
                let c: f32 = (0..m).map(|i| rq[i] / nq * bp[i]).sum::<f32>().abs();
                mu = mu.max(c);
            }
        }
    }
    if !diag_min.is_finite() {
        diag_min = 0.0;
    }
    (basis, mu, diag_min)
}

fn percentile(sorted: &[f32], q: f32) -> f32 {
    match sorted.is_empty() {
        true => 0.0,
        false => {
            let idx = ((sorted.len() - 1) as f32 * q).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        }
    }
}

/// Fit a TPR artifact by ridge-ALS (Issue 707 T5).
///
/// Returns the frozen (BLAKE3-committed) artifact and the certificate bundle
/// the GOAT gates read. Deterministic: same input + config → bit-identical
/// bytes.
pub fn als_fit(
    input: AlsInput<'_>,
    cfg: &AlsConfig,
) -> Result<(TprArtifact, AlsReport), TprError> {
    let dim = input.dim;
    let d = cfg.d;
    let m = cfg.scheme.arity();
    let k = m * d;
    let n = input.n_states();
    let n_slots = cfg.scheme.n_bind_slots();

    if dim == 0 || d == 0 || m == 0 || n == 0 {
        return Err(TprError::DimMismatch {
            what: "fit geometry",
            expected: 1,
            got: 0,
        });
    }
    check(input.states.len(), n * dim, "states")?;
    check(input.bindings.len(), n, "bindings")?;
    if matches!(cfg.scheme, TprScheme::RoleVectors { .. }) && n_slots > m {
        return Err(TprError::DimMismatch {
            what: "bind slots (unbind needs n_slots <= arity)",
            expected: m,
            got: n_slots,
        });
    }
    for b in input.bindings {
        for (&p, &v) in b.roles.iter().zip(b.fillers.iter()) {
            if p as usize >= n_slots {
                return Err(TprError::BadId {
                    what: "role",
                    max: n_slots.saturating_sub(1),
                    got: p as usize,
                });
            }
            if v as usize >= input.n_fillers {
                return Err(TprError::BadId {
                    what: "filler",
                    max: input.n_fillers.saturating_sub(1),
                    got: v as usize,
                });
            }
        }
    }
    if !all_finite(input.states) {
        return Err(TprError::NonFinite("fit states"));
    }

    let n_fillers = input.n_fillers;
    let p_aug = k + 1;

    // Deterministic filler init: SplitMix64 rows, unit-normalized.
    let mut rng = SplitMix64::new(cfg.lcg_seed);
    let mut fillers = vec![0.0f32; n_fillers * d];
    for v in 0..n_fillers {
        let row = &mut fillers[v * d..(v + 1) * d];
        for x in row.iter_mut() {
            *x = rng.next_sym();
        }
        let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nrm = match nrm > 1e-6 {
            true => nrm,
            false => 1.0,
        };
        for x in row.iter_mut() {
            *x /= nrm;
        }
    }

    let mut scheme = cfg.scheme.clone();
    let occ = Occurrences::build(&input, n_fillers, n_slots);

    let mut cores = vec![0.0f32; n * k];
    let mut chat = vec![0.0f32; k * n];
    let mut resid = vec![0.0f32; n * k];
    let mut w = vec![0.0f32; dim * k];
    let mut bias = vec![0.0f32; dim];
    let mut row = vec![0.0f32; m];
    let mut num = vec![0.0f32; d.max(m)];
    let mut gram = vec![0.0f32; p_aug * p_aug];
    let mut cov = vec![0.0f32; p_aug * dim];
    let mut wt = vec![0.0f32; p_aug * dim];
    let mut l_big = vec![0.0f32; p_aug * p_aug];
    let mut z_big = vec![0.0f32; (p_aug * dim).max(k * n)];
    let mut l_k = vec![0.0f32; k * k];
    let mut rhs = vec![0.0f32; k * n];
    let mut free = vec![0.0f32; k * n];

    let mut report = AlsReport::default();

    build_cores(&input, &scheme, &fillers, d, m, &mut row, &mut cores);
    solve_wb(
        input.states,
        &cores,
        n,
        k,
        d,
        dim,
        cfg.ridge_lambda,
        &mut gram,
        &mut cov,
        &mut wt,
        &mut l_big,
        &mut z_big,
        &mut w,
        &mut bias,
    );
    if !all_finite(&w) || !all_finite(&bias) {
        return Err(TprError::NonFinite("W/b block"));
    }
    let mut prev = state_ssr(input.states, &cores, &w, &bias, n, k, d, dim, None);
    report.ssr_per_sweep.push(prev);
    // Monotone tolerance is scaled by the INITIAL objective, not by the
    // current one. Near convergence the SSR is a sum of f32-accumulated
    // near-zero residuals, so its last digits are noise: a `prev·1e-9` bar
    // reports that noise as a violation (measured: a 2e-9 blip on a 1.96e-5
    // objective, i.e. 1e-4 relative, after the fit had already fallen 7
    // orders of magnitude). Against `ssr_0` the bar stays meaningful — a real
    // divergence grows by orders of magnitude, not by 1e-9 of the total
    // explained energy.
    let monotone_tol = (prev.abs() * 1e-9).max(1e-12);

    let (l21_iters, l21_lambda) = match cfg.l21 {
        L21::ReweightedMm { iters, lambda } => (iters.max(1), lambda),
        _ => (1, 0.0),
    };
    let mut l21_w = vec![0.0f32; d];

    let mut sweeps = 0u32;
    // Best-iterate guard (Issue 712): the monotone acceptance bar is scaled
    // by the INITIAL ssr (correctly — near convergence the SSR's last digits
    // are f32 noise), so a small-but-real uphill step near the floor can be
    // accepted and shipped, violating the documented "artifact == trajectory
    // minimum" invariant (measured: sweep 6→7 went 9.7272e-5 → 1.0247e-4,
    // +5.2%, under the ssr_0·1e-9 = 1.23e-5 bar; bench_707 G1 `is_best` red).
    // Track the best accepted iterate and restore it at loop exit — the
    // shipped artifact is then the trajectory minimum BY CONSTRUCTION,
    // independent of fp-path luck.
    let mut best_ssr = prev;
    let mut best_snap: Option<AlsIterate> = None;
    for _ in 0..cfg.max_sweeps {
        sweeps += 1;
        // Snapshot for the descent guard: blocks 3–4 minimize the CORE-space
        // surrogate, not the state-space objective, so a sweep can in
        // principle propose an uphill step. Rather than merely counting that
        // (which would ship the worse fit), the proposal is rejected and the
        // last accepted iterate is returned — the artifact is then, by
        // construction, the minimum of the recorded trajectory.
        let snap = (fillers.clone(), scheme.clone(), w.clone(), bias.clone(), cores.clone());
        free_cores(
            input.states,
            &w,
            &bias,
            n,
            k,
            d,
            dim,
            cfg.ridge_lambda,
            &mut gram,
            &mut l_k,
            &mut rhs,
            &mut z_big,
            &mut free,
        )?;
        // free is k × n row-major; transpose into per-state cores.
        for s in 0..n {
            for j in 0..k {
                chat[s * k + j] = free[j * n + s];
            }
        }
        residual_from(&chat, &cores, n, k, &mut resid);

        for pass in 0..l21_iters {
            let weights = match cfg.l21 {
                L21::ReweightedMm { lambda, .. } => {
                    // MM surrogate weights from the CURRENT filler table.
                    for j in 0..d {
                        let cn: f32 = (0..n_fillers)
                            .map(|v| fillers[v * d + j] * fillers[v * d + j])
                            .sum::<f32>()
                            .sqrt();
                        l21_w[j] = lambda / (cn + 1e-6);
                    }
                    let _ = pass;
                    let _ = l21_lambda;
                    Some(&l21_w[..])
                }
                _ => None,
            };
            filler_block(
                &occ,
                &scheme,
                &mut resid,
                &mut fillers,
                d,
                m,
                cfg.filler_ridge,
                weights,
                &mut row,
                &mut num,
            );
        }
        role_block(
            &occ,
            &mut scheme,
            &mut resid,
            &fillers,
            d,
            m,
            cfg.filler_ridge,
            &mut row,
            &mut num,
        );
        if !all_finite(&fillers) {
            return Err(TprError::NonFinite("filler block"));
        }

        build_cores(&input, &scheme, &fillers, d, m, &mut row, &mut cores);
        solve_wb(
            input.states,
            &cores,
            n,
            k,
            d,
            dim,
            cfg.ridge_lambda,
            &mut gram,
            &mut cov,
            &mut wt,
            &mut l_big,
            &mut z_big,
            &mut w,
            &mut bias,
        );
        if !all_finite(&w) || !all_finite(&bias) {
            return Err(TprError::NonFinite("W/b block"));
        }
        let ssr = state_ssr(input.states, &cores, &w, &bias, n, k, d, dim, None);
        // The rejected proposal is still RECORDED — the trajectory stays
        // auditable — but the parameters are rolled back.
        report.ssr_per_sweep.push(ssr);
        if ssr > prev + monotone_tol {
            report.monotone_violations += 1;
            (fillers, scheme, w, bias, cores) = snap;
            break;
        }
        let improved = prev - ssr;
        let converged = improved <= cfg.tol * prev.abs().max(1e-12);
        prev = ssr;
        if ssr < best_ssr {
            best_ssr = ssr;
            let snap: AlsIterate =
                (fillers.clone(), scheme.clone(), w.clone(), bias.clone(), cores.clone());
            best_snap = Some(snap);
        }
        if converged {
            break;
        }
    }
    // Restore the best accepted iterate when the loop exited on a
    // within-tolerance uphill step (Issue 712). Also covers the
    // monotone-reject path: `snap` there is the last accepted state, which
    // the best-tracking subsumes when an earlier sweep was strictly better.
    if prev > best_ssr && let Some(best) = best_snap.take() {
        (fillers, scheme, w, bias, cores) = best;
        prev = best_ssr;
    }

    // L2,1 prune + exact refit.
    if let L21::PruneRefit { tau_frac } = cfg.l21 {
        let ssr_before = prev;
        let mut col_norm = vec![0.0f32; d];
        for (j, cn) in col_norm.iter_mut().enumerate() {
            *cn = (0..n_fillers)
                .map(|v| fillers[v * d + j] * fillers[v * d + j])
                .sum::<f32>()
                .sqrt();
        }
        let max_norm = col_norm.iter().copied().fold(0.0f32, f32::max);
        let cut = tau_frac * max_norm;
        let mut pruned = 0usize;
        for (j, &cn) in col_norm.iter().enumerate() {
            if cn < cut {
                pruned += 1;
                for v in 0..n_fillers {
                    fillers[v * d + j] = 0.0;
                }
            }
        }
        report.pruned_dims = pruned;
        build_cores(&input, &scheme, &fillers, d, m, &mut row, &mut cores);
        solve_wb(
            input.states,
            &cores,
            n,
            k,
            d,
            dim,
            cfg.ridge_lambda,
            &mut gram,
            &mut cov,
            &mut wt,
            &mut l_big,
            &mut z_big,
            &mut w,
            &mut bias,
        );
        prev = state_ssr(input.states, &cores, &w, &bias, n, k, d, dim, None);
        report.ssr_per_sweep.push(prev);
        report.prune_ssr_increase = prev - ssr_before;
    }

    // Residual certificate.
    let mut per_state = vec![0.0f32; n];
    let final_ssr = state_ssr(
        input.states,
        &cores,
        &w,
        &bias,
        n,
        k,
        d,
        dim,
        Some(&mut per_state),
    );
    let mut sorted = per_state.clone();
    sorted.sort_by(|a, b| crate::float_order::asc(*a, *b));
    let mut mean = vec![0.0f32; dim];
    for s in 0..n {
        for (i, mv) in mean.iter_mut().enumerate() {
            *mv += input.states[s * dim + i];
        }
    }
    for v in mean.iter_mut() {
        *v /= n as f32;
    }
    let mut centered_energy = 0.0f64;
    for s in 0..n {
        for (i, &mv) in mean.iter().enumerate() {
            let dv = (input.states[s * dim + i] - mv) as f64;
            centered_energy += dv * dv;
        }
    }
    let energy_fraction = match centered_energy > 0.0 {
        true => (final_ssr / centered_energy) as f32,
        false => 0.0,
    };

    report.final_ssr = final_ssr;
    report.sweeps = sweeps;
    report.residual_p50 = percentile(&sorted, 0.50);
    report.residual_p99 = percentile(&sorted, 0.99);
    report.residual_max = sorted.last().copied().unwrap_or(0.0);
    report.residual_energy_fraction = energy_fraction;

    // Unbind basis + coherence.
    let (basis, mu, diag_min) = match &scheme {
        TprScheme::Orthogonal { .. } => (None, 0.0f32, 1.0f32),
        TprScheme::RoleVectors { roles, .. } => {
            let (b, mu, dg) = unbind_basis(roles, n_slots, m);
            (Some(b), mu, dg)
        }
    };
    let max_filler_norm = (0..n_fillers)
        .map(|v| {
            fillers[v * d..(v + 1) * d]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
        })
        .fold(0.0f32, f32::max);

    // Cached projection Cholesky (T4).
    let projection_lambda = cfg.ridge_lambda.max(MIN_PROJECTION_LAMBDA);
    let projection_inv = match cfg.build_projection && k <= TPR_MAX_PROJECTION_K {
        false => None,
        true => {
            let mut g = vec![0.0f32; k * k];
            for i in 0..dim {
                for a in 0..k {
                    let wa = w_at(&w, dim, d, i, a);
                    match wa == 0.0 {
                        true => {}
                        false => {
                            let r = a * k;
                            for b in 0..k {
                                g[r + b] = wa.mul_add(w_at(&w, dim, d, i, b), g[r + b]);
                            }
                        }
                    }
                }
            }
            for a in 0..k {
                g[a * k + a] += projection_lambda;
            }
            match all_finite(&g) {
                false => None,
                true => {
                    let mut inv = vec![0.0f32; k * k];
                    let mut l = vec![0.0f32; k * k];
                    let mut inv_l = vec![0.0f32; k * k];
                    spd_inverse_f32(&mut inv, &mut l, &mut inv_l, &g, k);
                    match all_finite(&inv) {
                        true => Some(inv),
                        false => None,
                    }
                }
            }
        }
    };

    let bic_label = match &scheme {
        TprScheme::Orthogonal { arity } => format!("orthogonal:m{arity}:d{d}"),
        TprScheme::RoleVectors { arity, .. } => {
            format!("rolevectors:m{arity}:s{n_slots}:d{d}")
        }
    };

    let mut art = TprArtifact {
        version: SCHEMA_VERSION,
        dim,
        d,
        m,
        n_fillers,
        w,
        bias,
        fillers,
        scheme,
        unbind_basis: basis,
        crosstalk_mu: mu,
        unbind_diag_min: diag_min,
        max_filler_norm,
        residual_p50: report.residual_p50,
        residual_p99: report.residual_p99,
        residual_max: report.residual_max,
        residual_energy_fraction: energy_fraction,
        n_fit_states: n,
        bic_label,
        fit_objective: final_ssr,
        als_sweeps: sweeps,
        monotone_violations: report.monotone_violations,
        pruned_dims: report.pruned_dims,
        projection_inv,
        projection_lambda,
        commitment: [0u8; 32],
    };
    art.freeze();
    Ok((art, report))
}

#[inline]
fn check(got: usize, expected: usize, what: &'static str) -> Result<(), TprError> {
    match got == expected {
        true => Ok(()),
        false => Err(TprError::DimMismatch {
            what,
            expected,
            got,
        }),
    }
}

/// Parameter count of a fitted artifact — the BIC complexity term (T7).
pub fn param_count(art: &TprArtifact) -> usize {
    let structural = match &art.scheme {
        TprScheme::Orthogonal { .. } => 0,
        TprScheme::RoleVectors { arity, roles } => roles.len() / (*arity).max(1) * *arity,
    };
    art.dim * art.core_len() + art.dim + art.n_fillers * art.d + structural
}
