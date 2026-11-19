//! relevance_core — the ACE extraction primitive: the whitened
//! activity × relevance subspace (Plan 640; Research 683, arXiv:2602.22600
//! v3, Schiffman NeurIPS 2026; reference `joshseth/cores @ 527948bb`, MIT —
//! the factored balanced-SVD form is the math of record).
//!
//! # The math (Research 683 §2)
//!
//! Given the activity covariance `Cov ∈ R^{d×d}` of a representation `h` and
//! the accumulated relevance (sensitivity) moment `S = E[Jᵀ J]` of a frozen
//! downstream head `f(h)` (`J = ∂f/∂h`, streamed per sample through
//! [`accumulate_sens`]), the relevance core is the top-r eigenspace of the
//! balanced product
//!
//! ```text
//!   L = Cov^{+1/2}            (sqrt factor; trace-relative jitter, negatives clamped)
//!   M' = L · S · L            (the ONLY new linear algebra — two symmetric multiplies)
//!   M' = V Λ Vᵀ               (the shared basis_split eigen kernel, descending)
//!   Q  = orth(L · V_r)        (map back; modified Gram–Schmidt, f64)
//! ```
//!
//! `energies = Λ` — ACE's σ²: `Cov·S` and `L S L` are similar matrices, so
//! the energies are the eigenvalues of `Cov·S` (directions that BOTH carry
//! activity variance AND move the head).
//!
//! **Signature:** the whitening exponent is `Cov^{+1/2}`, not CCA's
//! `Cov^{−1/2}` — a direction the representation never uses scores zero no
//! matter how sensitive the head is to it (that is the point: relevance of
//! what the model actually *does*, not what it *could* read).
//!
//! **Special cases (the positive controls):** isotropic sensitivity
//! (`S ∝ I`) ⇒ `M' ∝ Cov` ⇒ the core is the PCA basis; isotropic activity
//! (`Cov ∝ I`) ⇒ the core is the top eigenspace of `S`; a rank-1 binary task
//! under a linear head reduces to the difference-of-means direction.
//!
//! # Scope (honest)
//!
//! An OFFLINE analysis primitive — calibration-time `O(d³)`, never a serving
//! path. Research 388's prefilter refutation is the envelope boundary: a
//! relevance subspace is an *instrument* over a frozen head, not a cheaper
//! forward. Opt-in; promotion needs the Phase-1 GOAT gate AND a wired
//! consumer (riir-infer Issue 1237 is the first candidate).
//!
//! # Composition (consume, never fork)
//!
//! Both eigensolves (`Cov` and `M'`) ride [`crate::basis_split`]'s shared
//! kernel (d ≤ 16 Jacobi / otherwise Householder+QL, descending, row
//! eigenvectors) and the [`RankRule::Gap`] verdict IS
//! [`crate::basis_split::check_rank_gate`]. No second solver.
//!
//! # Determinism + allocation
//!
//! f64 end-to-end, sequential folds in ascending index order, first-index
//! tie rule on the energy crossing (Issue 940). All scratch is the
//! caller-owned [`CoreScratch`] (allocated once per `d`); the extract path is
//! zero-alloc.

use crate::basis_split::{
    BasisSplitError, MAX_DIM, UnidentifiedReason, Verdict, check_rank_gate, eigh_rows_desc,
    participation_ratio,
};
use crate::linalg::symmetric_eig::SymmetricEigScratch;

/// Trace-relative jitter added to `Cov`'s eigenvalues before the square root
/// (`λ_i ← max(λ_i, 0) + JITTER_REL · tr(Cov)/d`) — the reference's
/// conditioning guard; small enough to leave a rank-deficient direction at
/// effectively zero activity weight.
pub const JITTER_REL: f64 = 1e-10;

/// A mapped-back core column whose norm falls below this (relative to the
/// largest) is a COLLAPSE — `L` annihilated it — and refuses loud.
pub const COLLAPSE_REL: f64 = 1e-9;

/// Refuse-loud errors. Every variant names the caller-fixable fact.
#[derive(Debug, Clone, PartialEq)]
pub enum RelevanceCoreError {
    /// The shared kernel's refusal (dimension, scratch, …) — forwarded.
    Kernel(BasisSplitError),
    /// A `Cov` / `S` / Jacobian entry is non-finite (flat index disclosed).
    NonFinite { index: usize },
    /// `Cov` has zero (or negative) trace — no activity, no core.
    ZeroActivity,
    /// The whitened relevance moment `L S L` has zero trace — the head is
    /// insensitive to every direction the representation uses.
    ZeroRelevance,
    /// [`RankRule::Gap`] refused: no spectral gap certifies the boundary.
    Unidentified(UnidentifiedReason),
    /// The requested rank is `0` or exceeds `d`.
    RankOutOfRange { r: usize },
    /// `Q = orth(L V_r)` lost a column (`L` annihilated `V_r`'s row `index`).
    Collapsed { index: usize },
}

impl From<BasisSplitError> for RelevanceCoreError {
    fn from(e: BasisSplitError) -> Self {
        Self::Kernel(e)
    }
}

impl core::fmt::Display for RelevanceCoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Kernel(e) => write!(f, "eigen kernel: {e}"),
            Self::NonFinite { index } => write!(f, "input[{index}] is non-finite"),
            Self::ZeroActivity => write!(f, "activity covariance has zero trace"),
            Self::ZeroRelevance => write!(f, "whitened relevance moment has zero trace"),
            Self::Unidentified(r) => write!(f, "rank gate refused: {r:?}"),
            Self::RankOutOfRange { r } => write!(f, "rank {r} out of range"),
            Self::Collapsed { index } => write!(f, "core column {index} collapsed under L"),
        }
    }
}

impl std::error::Error for RelevanceCoreError {}

/// How the core rank `r` is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RankRule {
    /// A caller-fixed rank (`1..=d`).
    Fixed(usize),
    /// Smallest `r` whose cumulative energy share reaches `threshold`
    /// ([`core_rank_energy`], the reference `_select_rank` semantics).
    Energy(f64),
    /// The largest-eigengap boundary, refuse-loud below `tau_gap`
    /// ([`check_rank_gate`] — the basis_split identifiability verdict).
    Gap { tau_gap: f64 },
}

/// The extract report. The basis rows land in the caller's `q_out` and the
/// full descending spectrum in `energies_out`.
#[derive(Debug, Clone, PartialEq)]
pub struct CoreExtract {
    /// Selected core rank (rows of `q_out` written).
    pub r: usize,
    /// `λ_r / λ_{r+1}` at the chosen boundary (`+inf` past an exact zero,
    /// `NaN` when `r == d`).
    pub gap_ratio: f64,
    /// Cumulative energy share captured by the top-r (`Σ_{i<r} λ_i / Σ λ⁺`).
    pub energy_share: f64,
    /// Participation ratio of the energy spectrum — REPORT-ONLY.
    pub participation_ratio: f64,
}

/// Caller-owned scratch for [`extract_core_from_matrices`], sized once per
/// `d`. Every buffer is overwritten per call.
pub struct CoreScratch {
    d: usize,
    sym: SymmetricEigScratch,
    jacobi: Vec<f64>,
    l: Vec<f64>,
    work: Vec<f64>,
    m_white: Vec<f64>,
    eigvals: Vec<f64>,
    eigvecs: Vec<f64>,
}

impl CoreScratch {
    /// Allocates every buffer for dimension `d` (the only allocation site).
    pub fn new(d: usize) -> Self {
        let mut sym = SymmetricEigScratch::new();
        sym.ensure_capacity(d);
        Self {
            d,
            sym,
            jacobi: vec![0.0; d * d],
            l: vec![0.0; d * d],
            work: vec![0.0; d * d],
            m_white: vec![0.0; d * d],
            eigvals: vec![0.0; d],
            eigvecs: vec![0.0; d * d],
        }
    }

    /// The dimension this scratch was sized for.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// The row-major `Cov^{1/2}` factor of the last extract (diagnostic; the
    /// Phase-1 PoC reads it to map whitened quantities back).
    pub fn sqrt_factor(&self) -> &[f64] {
        &self.l
    }
}

fn check_finite(a: &[f64]) -> Result<(), RelevanceCoreError> {
    match a.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(RelevanceCoreError::NonFinite { index }),
        None => Ok(()),
    }
}

/// Running relevance moment `S += jᵀ j` for ONE Jacobian row `j_row`
/// (length `d`), so callers stream per-sample Jacobian rows without
/// materializing `J ∈ R^{N·K×d}`. Sequential f64 fold, upper triangle +
/// mirror (exactly symmetric). `n_seen` counts accumulated rows; the scale
/// of `S` never moves the core basis (energies scale with it), so dividing
/// by `n_seen` is the caller's choice. Zero-alloc.
pub fn accumulate_sens(
    sens: &mut [f64],
    d: usize,
    j_row: &[f64],
    n_seen: &mut u64,
) -> Result<(), RelevanceCoreError> {
    if d == 0 || d > MAX_DIM {
        return Err(BasisSplitError::DimOutOfRange.into());
    }
    if sens.len() != d * d || j_row.len() != d {
        return Err(BasisSplitError::DimensionMismatch { expected: d * d }.into());
    }
    check_finite(j_row)?;
    for i in 0..d {
        let ji = j_row[i];
        if ji == 0.0 {
            continue;
        }
        for k in i..d {
            let c = ji * j_row[k];
            sens[i * d + k] += c;
            if k != i {
                sens[k * d + i] += c;
            }
        }
    }
    *n_seen += 1;
    Ok(())
}

/// Smallest rank whose cumulative share of the POSITIVE energy reaches
/// `threshold` (sequential cumsum, first-index crossing). Floor 1; a
/// spectrum with no positive energy returns 1 (the extract refuses that
/// case earlier as [`RelevanceCoreError::ZeroRelevance`]).
/// `energies_desc` must be sorted descending.
pub fn core_rank_energy(energies_desc: &[f64], threshold: f64) -> usize {
    let total: f64 = energies_desc.iter().map(|&e| e.max(0.0)).sum();
    if total.is_nan() || total <= 0.0 {
        return 1;
    }
    let mut cum = 0.0;
    for (i, &e) in energies_desc.iter().enumerate() {
        cum += e.max(0.0);
        if cum / total >= threshold {
            return i + 1;
        }
    }
    energies_desc.len().max(1)
}

/// Symmetric product `out = A · B · A` for row-major d×d `A` (symmetric) and
/// `B`, via `work = B · A` then `out = A · work`. Sequential folds.
fn sandwich(a: &[f64], b: &[f64], d: usize, work: &mut [f64], out: &mut [f64]) {
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0;
            for l in 0..d {
                acc += b[i * d + l] * a[l * d + j];
            }
            work[i * d + j] = acc;
        }
    }
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0;
            for l in 0..d {
                acc += a[i * d + l] * work[l * d + j];
            }
            out[i * d + j] = acc;
        }
    }
    // Pin exact symmetry (the two products drift by rounding).
    for i in 0..d {
        for j in (i + 1)..d {
            let m = 0.5 * (out[i * d + j] + out[j * d + i]);
            out[i * d + j] = m;
            out[j * d + i] = m;
        }
    }
}

/// Extracts the relevance core from a finalized activity covariance `cov`
/// and relevance moment `sens` (both row-major d×d, symmetric PSD).
///
/// Writes the r core rows (orthonormal, row-major `r × d`) into `q_out`
/// (length ≥ `r*d`; size it `d*d` when the rule picks `r`) and the full
/// descending energy spectrum into `energies_out` (length ≥ `d`).
///
/// Refuse-loud: non-finite input, zero activity, zero relevance, a
/// [`RankRule::Gap`] boundary with no certified gap, a rank out of range, or
/// a collapsed mapped-back column — never a silent basis. Zero-alloc.
pub fn extract_core_from_matrices(
    cov: &[f64],
    sens: &[f64],
    d: usize,
    rule: RankRule,
    q_out: &mut [f64],
    energies_out: &mut [f64],
    scratch: &mut CoreScratch,
) -> Result<CoreExtract, RelevanceCoreError> {
    if d == 0 || d > MAX_DIM {
        return Err(BasisSplitError::DimOutOfRange.into());
    }
    if cov.len() != d * d || sens.len() != d * d || scratch.d != d {
        return Err(BasisSplitError::DimensionMismatch { expected: d * d }.into());
    }
    if energies_out.len() < d {
        return Err(BasisSplitError::ScratchTooShort { required: d }.into());
    }
    check_finite(cov)?;
    check_finite(sens)?;
    let trace: f64 = (0..d).map(|i| cov[i * d + i]).sum();
    if trace.is_nan() || trace <= 0.0 {
        return Err(RelevanceCoreError::ZeroActivity);
    }

    let s = scratch;

    // (1) Cov = U Λ Uᵀ → L = U diag(sqrt(max(λ,0) + jitter)) Uᵀ.
    eigh_rows_desc(
        cov,
        d,
        &mut s.eigvals,
        &mut s.eigvecs,
        &mut s.sym,
        &mut s.jacobi,
    )?;
    let jitter = JITTER_REL * trace / d as f64;
    s.l.fill(0.0);
    for k in 0..d {
        let root = (s.eigvals[k].max(0.0) + jitter).sqrt();
        let u = &s.eigvecs[k * d..(k + 1) * d];
        for i in 0..d {
            let ui = root * u[i];
            for (dst, &uj) in s.l[i * d + i..(i + 1) * d].iter_mut().zip(&u[i..]) {
                *dst += ui * uj;
            }
        }
    }
    for i in 0..d {
        for j in (i + 1)..d {
            s.l[j * d + i] = s.l[i * d + j];
        }
    }

    // (2) M' = L S L — the only new linear algebra.
    sandwich(&s.l, sens, d, &mut s.work, &mut s.m_white);
    let rel_trace: f64 = (0..d).map(|i| s.m_white[i * d + i]).sum();
    if rel_trace.is_nan() || rel_trace <= 0.0 {
        return Err(RelevanceCoreError::ZeroRelevance);
    }

    // (3) M' = V Λ Vᵀ through the shared kernel; rank by rule.
    eigh_rows_desc(
        &s.m_white,
        d,
        &mut s.eigvals,
        &mut s.eigvecs,
        &mut s.sym,
        &mut s.jacobi,
    )?;
    energies_out[..d].copy_from_slice(&s.eigvals[..d]);
    let r = match rule {
        RankRule::Fixed(r) => r,
        RankRule::Energy(threshold) => core_rank_energy(&s.eigvals[..d], threshold),
        RankRule::Gap { tau_gap } => match check_rank_gate(&s.eigvals[..d], None, tau_gap).0 {
            Verdict::SufficientVariation { k, .. } => k,
            Verdict::Unidentified(reason) => return Err(RelevanceCoreError::Unidentified(reason)),
            Verdict::Unwhitened => unreachable!("check_rank_gate never emits Unwhitened"),
        },
    };
    if r == 0 || r > d {
        return Err(RelevanceCoreError::RankOutOfRange { r });
    }
    if q_out.len() < r * d {
        return Err(BasisSplitError::ScratchTooShort { required: r * d }.into());
    }

    // (4) Q = orth(L V_r): row b = L v_b (L symmetric), then modified
    // Gram–Schmidt in ascending row order.
    let mut max_norm = 0.0f64;
    for b in 0..r {
        let v = &s.eigvecs[b * d..(b + 1) * d];
        let row = &mut q_out[b * d..(b + 1) * d];
        for (i, out) in row.iter_mut().enumerate() {
            let l_row = &s.l[i * d..(i + 1) * d];
            *out = l_row.iter().zip(v).map(|(a, b)| a * b).sum();
        }
        let n = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        max_norm = max_norm.max(n);
    }
    for b in 0..r {
        for p in 0..b {
            let (done, rest) = q_out.split_at_mut(b * d);
            let qp = &done[p * d..(p + 1) * d];
            let row = &mut rest[..d];
            let dot: f64 = qp.iter().zip(row.iter()).map(|(a, c)| a * c).sum();
            for (x, &y) in row.iter_mut().zip(qp) {
                *x -= dot * y;
            }
        }
        let row = &mut q_out[b * d..(b + 1) * d];
        let n = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n.is_nan() || n <= COLLAPSE_REL * max_norm {
            return Err(RelevanceCoreError::Collapsed { index: b });
        }
        for x in row.iter_mut() {
            *x /= n;
        }
    }

    let positive: f64 = s.eigvals[..d].iter().map(|&e| e.max(0.0)).sum();
    let captured: f64 = s.eigvals[..r].iter().map(|&e| e.max(0.0)).sum();
    Ok(CoreExtract {
        r,
        gap_ratio: if r < d {
            s.eigvals[r - 1] / s.eigvals[r]
        } else {
            f64::NAN
        },
        energy_share: captured / positive,
        participation_ratio: participation_ratio(&s.eigvals[..d]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn gauss(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
        }
    }

    /// Random orthonormal frame (rows) by Gram–Schmidt.
    fn frame(d: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg(seed);
        let mut q: Vec<f64> = (0..d * d).map(|_| rng.gauss()).collect();
        for i in 0..d {
            for p in 0..i {
                let dot: f64 = (0..d).map(|t| q[i * d + t] * q[p * d + t]).sum();
                for t in 0..d {
                    q[i * d + t] -= dot * q[p * d + t];
                }
            }
            let n: f64 = (0..d).map(|t| q[i * d + t].powi(2)).sum::<f64>().sqrt();
            for t in 0..d {
                q[i * d + t] /= n;
            }
        }
        q
    }

    /// Σ_k lam[k] q_k q_kᵀ.
    fn compose(q: &[f64], lam: &[f64], d: usize) -> Vec<f64> {
        let mut m = vec![0.0; d * d];
        for (k, &l) in lam.iter().enumerate() {
            for i in 0..d {
                for j in 0..d {
                    m[i * d + j] += l * q[k * d + i] * q[k * d + j];
                }
            }
        }
        m
    }

    fn ident(d: usize, c: f64) -> Vec<f64> {
        (0..d * d)
            .map(|t| if t / d == t % d { c } else { 0.0 })
            .collect()
    }

    /// Σ_{b,k} (q_b · ref_k)² — r for a perfect span match.
    fn capture(q: &[f64], r: usize, refs: &[f64], k: usize, d: usize) -> f64 {
        let mut c = 0.0;
        for b in 0..r {
            for j in 0..k {
                let dot: f64 = (0..d).map(|t| q[b * d + t] * refs[j * d + t]).sum();
                c += dot * dot;
            }
        }
        c
    }

    fn run(
        cov: &[f64],
        sens: &[f64],
        d: usize,
        rule: RankRule,
    ) -> (CoreExtract, Vec<f64>, Vec<f64>) {
        let mut sc = CoreScratch::new(d);
        let (mut q, mut e) = (vec![0.0; d * d], vec![0.0; d]);
        let rep = extract_core_from_matrices(cov, sens, d, rule, &mut q, &mut e, &mut sc).unwrap();
        (rep, q, e)
    }

    /// Positive control: isotropic activity ⇒ the core is S's top eigenspace
    /// (rotated, both solver arms).
    #[test]
    fn isotropic_activity_recovers_the_sensitivity_top_space() {
        for &d in &[8usize, 40] {
            let f = frame(d, 0x640_0001 + d as u64);
            let lam: Vec<f64> = (0..d)
                .map(|k| if k < 3 { 50.0 - k as f64 } else { 0.1 })
                .collect();
            let sens = compose(&f, &lam, d);
            let (rep, q, e) = run(&ident(d, 2.0), &sens, d, RankRule::Gap { tau_gap: 10.0 });
            assert_eq!(rep.r, 3, "d={d}");
            assert!((capture(&q, 3, &f, 3, d) - 3.0).abs() < 1e-9, "d={d}");
            // energies = eig(Cov·S) = 2·lam (to the JITTER_REL·tr/d shift)
            assert!((e[0] - 100.0).abs() < 1e-7 * 100.0, "d={d} e0={}", e[0]);
        }
    }

    /// Positive control: isotropic sensitivity ⇒ the core is the PCA basis.
    #[test]
    fn isotropic_sensitivity_recovers_pca() {
        let d = 24;
        let f = frame(d, 0x640_0002);
        let lam: Vec<f64> = (0..d)
            .map(|k| if k < 2 { 9.0 - k as f64 } else { 0.05 })
            .collect();
        let cov = compose(&f, &lam, d);
        let (rep, q, _) = run(&cov, &ident(d, 1.0), d, RankRule::Fixed(2));
        assert_eq!(rep.r, 2);
        assert!((capture(&q, 2, &f, 2, d) - 2.0).abs() < 1e-8);
    }

    /// The defining property (the PCA separator): a high-variance direction
    /// the head ignores is excluded; a low-variance direction the head reads
    /// is kept — and a direction the representation never uses scores zero
    /// however sensitive the head is (the Cov^{+1/2} signature).
    #[test]
    fn decoy_variance_is_excluded_and_unused_directions_score_zero() {
        let d = 16;
        let f = frame(d, 0x640_0003);
        // f0 = decoy (variance 100, sensitivity 0); f1 = relevant (variance
        // 1, sensitivity 1); f2 = unused (variance 0, sensitivity 1000).
        let mut cov_l = vec![0.01; d];
        cov_l[0] = 100.0;
        cov_l[1] = 1.0;
        cov_l[2] = 0.0;
        let mut sens_l = vec![0.0; d];
        sens_l[1] = 1.0;
        sens_l[2] = 1000.0;
        let cov = compose(&f, &cov_l, d);
        let sens = compose(&f, &sens_l, d);
        let (_, q, e) = run(&cov, &sens, d, RankRule::Fixed(1));
        let row1: f64 = (0..d).map(|t| q[t] * f[d + t]).sum();
        assert!(
            row1.abs() > 1.0 - 1e-6,
            "top core row must be the relevant axis"
        );
        assert!(
            e[1] < 1e-6 * e[0],
            "unused + decoy directions score ~0, got {e:?}"
        );
    }

    /// Energies are the eigenvalues of Cov·S: Σλ = tr(Cov·S).
    #[test]
    fn energies_sum_to_trace_of_cov_times_sens() {
        let d = 20;
        let (fa, fb) = (frame(d, 0x640_0004), frame(d, 0x640_0005));
        let cov = compose(&fa, &(0..d).map(|k| 1.0 + k as f64).collect::<Vec<_>>(), d);
        let sens = compose(
            &fb,
            &(0..d).map(|k| 0.5 + (k % 3) as f64).collect::<Vec<_>>(),
            d,
        );
        let (_, _, e) = run(&cov, &sens, d, RankRule::Energy(0.9));
        let tr: f64 = (0..d)
            .map(|i| {
                (0..d)
                    .map(|l| cov[i * d + l] * sens[l * d + i])
                    .sum::<f64>()
            })
            .sum();
        let sum: f64 = e.iter().sum();
        assert!((sum - tr).abs() < 1e-6 * tr, "Σλ {sum} vs tr {tr}");
    }

    /// Q rows orthonormal + bitwise deterministic across two runs.
    #[test]
    fn core_is_orthonormal_and_bitwise_deterministic() {
        let d = 32;
        let (fa, fb) = (frame(d, 0x640_0006), frame(d, 0x640_0007));
        let cov = compose(
            &fa,
            &(0..d).map(|k| 1.0 / (1.0 + k as f64)).collect::<Vec<_>>(),
            d,
        );
        let sens = compose(&fb, &(0..d).map(|k| (d - k) as f64).collect::<Vec<_>>(), d);
        let (r1, q1, e1) = run(&cov, &sens, d, RankRule::Fixed(6));
        let (r2, q2, e2) = run(&cov, &sens, d, RankRule::Fixed(6));
        assert_eq!(r1, r2);
        assert_eq!(
            q1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            q2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            e1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            e2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        for a in 0..6 {
            for b in 0..6 {
                let dot: f64 = (0..d).map(|t| q1[a * d + t] * q1[b * d + t]).sum();
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-10, "Q[{a}]·Q[{b}] = {dot}");
            }
        }
    }

    #[test]
    fn accumulate_sens_streams_the_outer_product_sum() {
        let d = 5;
        let mut s = vec![0.0; d * d];
        let mut n = 0u64;
        let rows = [[1.0, 2.0, 0.0, -1.0, 0.5], [0.0, 1.0, 3.0, 0.0, -2.0]];
        for row in &rows {
            accumulate_sens(&mut s, d, row, &mut n).unwrap();
        }
        assert_eq!(n, 2);
        for i in 0..d {
            for j in 0..d {
                let want: f64 = rows.iter().map(|r| r[i] * r[j]).sum();
                assert_eq!(s[i * d + j], want);
            }
        }
        assert_eq!(
            accumulate_sens(&mut s, d, &[f64::NAN, 0.0, 0.0, 0.0, 0.0], &mut n),
            Err(RelevanceCoreError::NonFinite { index: 0 })
        );
    }

    #[test]
    fn core_rank_energy_first_crossing_with_floor() {
        assert_eq!(core_rank_energy(&[6.0, 3.0, 1.0], 0.6), 1);
        assert_eq!(core_rank_energy(&[6.0, 3.0, 1.0], 0.61), 2);
        assert_eq!(core_rank_energy(&[6.0, 3.0, 1.0], 0.9), 2);
        assert_eq!(core_rank_energy(&[6.0, 3.0, 1.0], 1.0), 3);
        assert_eq!(core_rank_energy(&[0.0, 0.0], 0.5), 1);
    }

    #[test]
    fn refusals_are_loud() {
        let d = 6;
        let mut sc = CoreScratch::new(d);
        let (mut q, mut e) = (vec![0.0; d * d], vec![0.0; d]);
        let i = ident(d, 1.0);
        let z = vec![0.0; d * d];
        let go =
            |c: &[f64], s: &[f64], rule, sc: &mut CoreScratch, q: &mut [f64], e: &mut [f64]| {
                extract_core_from_matrices(c, s, d, rule, q, e, sc)
            };
        assert_eq!(
            go(&z, &i, RankRule::Fixed(1), &mut sc, &mut q, &mut e),
            Err(RelevanceCoreError::ZeroActivity)
        );
        assert_eq!(
            go(&i, &z, RankRule::Fixed(1), &mut sc, &mut q, &mut e),
            Err(RelevanceCoreError::ZeroRelevance)
        );
        assert_eq!(
            go(&i, &i, RankRule::Fixed(0), &mut sc, &mut q, &mut e),
            Err(RelevanceCoreError::RankOutOfRange { r: 0 })
        );
        // Flat spectrum: no gap certifies any boundary.
        assert!(matches!(
            go(
                &i,
                &i,
                RankRule::Gap { tau_gap: 2.0 },
                &mut sc,
                &mut q,
                &mut e
            ),
            Err(RelevanceCoreError::Unidentified(_))
        ));
        let mut bad = i.clone();
        bad[7] = f64::INFINITY;
        assert_eq!(
            go(&bad, &i, RankRule::Fixed(1), &mut sc, &mut q, &mut e),
            Err(RelevanceCoreError::NonFinite { index: 7 })
        );
        let mut wrong = CoreScratch::new(d + 1);
        assert!(matches!(
            go(&i, &i, RankRule::Fixed(1), &mut wrong, &mut q, &mut e),
            Err(RelevanceCoreError::Kernel(
                BasisSplitError::DimensionMismatch { .. }
            ))
        ));
    }
}
