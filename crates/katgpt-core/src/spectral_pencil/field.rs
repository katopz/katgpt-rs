//! `field` — the spectral gate as an [`ArchetypeFieldSource`] (riir-ai Issue
//! 736 Phase B1 / Research 495 §5 fusion): a pencil genome becomes a
//! committed-blend archetype field whose [`lipschitz_bound`] is a DERIVED
//! certificate (the ‖Aᵢ‖₂ envelope), upgrading FAME Lemma 1's inputs from
//! hand-claimed constants to closed-form bounds.
//!
//! # The field
//!
//! `F(z) = λk(A(z[..N])) · z` — **spectral amplification**: the gate reads
//! the first `N` coordinates of the latent state as its belief features and
//! scales the whole state by the k-th eigenvalue (the spectral neuron's
//! activation as a growth rate). Requires `N ≤ D`.
//!
//! Eigenvalues are invariant under the pencil's orthogonal gauge, so the
//! field stores the CANONICAL form ([`gauge::canonicalize`]) — same-input
//! commitment bytes are stable, and canonical storage is a fixed point up
//! to the internal packed↔full round-trip float noise (~1 ulp on
//! off-diagonals).
//!
//! # The certificate (closed-form, construction-time)
//!
//! With exact norms `‖A₀‖₂`, `‖Aᵢ‖₂` ([`bounds::norm_jacobi_exact`]) and
//! the certified box `‖z‖₂ ≤ R`:
//!
//! * **Gate Lipschitz** (Weyl 1-Lipschitz of λk in the spectral norm +
//!   Cauchy–Schwarz on `Σ δᵢAᵢ`): `L_g = √(Σᵢ ‖Aᵢ‖₂²)`.
//! * **Growth envelope** (triangle inequality on `‖A(x)‖₂`):
//!   `G = ‖A₀‖₂ + R·L_g ≥ sup_box |λk(A(z))|`.
//! * **Field Lipschitz** (product rule on `F(z) = g(z)·z`, both factors
//!   bounded on the box): `L = G + R·L_g = ‖A₀‖₂ + 2·R·L_g`.
//!
//! The certificate is valid on `‖z‖₂ ≤ R` only — [`evolve`] never enforces
//! the box (the caller owns the state domain); [`certificate`] exposes the
//! components for diagnostics and GM surfacing.
//!
//! # Cost + allocation
//!
//! One `evolve` = one values-only pinned-Jacobi eval (paper §7.3: dense
//! d=16 ≈ 8K FLOPs; d=32 ≈ 44K) through a **stack** `DenseScratch` (~8.3 KB
//! at D=32) — zero heap allocation by construction (no `Box`/`Vec`/format
//! anywhere on the path; inherits the Issue 676 G4 eval audit, Bench 671).
//! Construction canonicalizes + solves `(N+1)` eigens for the exact norms —
//! a one-time cost amortized over the field's frozen lifetime.
//!
//! # Honest boundaries
//!
//! * Cross-**representation** commitment byte-stability (two conjugated
//!   parametrizations of the same function hashing identically) is NOT
//!   claimed: repeated eigenvalues in `A₀` leave a free in-block gauge
//!   ([`gauge`] doc), and float noise in different Jacobi paths defeats
//!   raw-byte hashing. Same-input determinism IS claimed. The quantized
//!   canonical-bytes policy for cross-representation stability belongs to
//!   the per-NPC genome Pod (riir-ai Issue 736 B3).
//! * The composed blend-level safety bound over these fields is
//!   `Σ_k gate_k·L_k` (triangle inequality over the gated sum) — see
//!   [`CommittedFieldBlend::lipschitz_bound`].
//!
//! [`evolve`]: ArchetypeFieldSource::evolve
//! [`lipschitz_bound`]: ArchetypeFieldSource::lipschitz_bound
//! [`CommittedFieldBlend::lipschitz_bound`]: crate::committed_field_blend::CommittedFieldBlend::lipschitz_bound

use crate::committed_field_blend::ArchetypeFieldSource;
use crate::spectral_pencil::bounds::norm_jacobi_exact;
use crate::spectral_pencil::dense::DenseScratch;
use crate::spectral_pencil::gauge;
use crate::spectral_pencil::{DensePencil, SymPacked};

/// The construction-time certificate of one [`SpectralField`].
///
/// Every component is a closed-form function of the pencil's coefficients
/// (exact Jacobi norms) and the certified box radius — no sampling, no
/// hand constants. This is the FAME Lemma 1 input upgrade: hosts used to
/// report `|scale|`, `1.0`, `0.0`, or `∞`; a spectral field reports the
/// derived envelope.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralFieldCert<const N: usize> {
    /// `‖A₀‖₂` (exact, Jacobi).
    pub norm_a0: f32,
    /// `‖Aᵢ‖₂` per feature matrix (exact, Jacobi).
    pub feature_norms: [f32; N],
    /// `L_g = √(Σᵢ ‖Aᵢ‖₂²)` — ℓ2 Lipschitz of `x ↦ λk(A(x))` (Weyl).
    pub gate_lipschitz: f32,
    /// `G = ‖A₀‖₂ + R·L_g` — sup of `|λk(A(z))|` on `‖z‖₂ ≤ R`.
    pub growth_envelope: f32,
    /// `L = ‖A₀‖₂ + 2·R·L_g` — Lipschitz of the field on `‖z‖₂ ≤ R`.
    pub field_lipschitz: f32,
    /// The certified box radius `R` (ℓ2, the certificate's domain).
    pub box_r2: f32,
}

/// A spectral-neuron archetype field: `F(z) = λk(A(z[..N]))·z` over a
/// canonical-gauge dense pencil, with a derived Lipschitz certificate.
///
/// Frozen by contract ([`ArchetypeFieldSource`]): constructed once from a
/// genome, evaluated without interior state. See the [module
/// docs](self) for the certificate math and honest boundaries.
///
/// # Example
///
/// ```
/// use katgpt_core::committed_field_blend::ArchetypeFieldSource;
/// use katgpt_core::spectral_pencil::field::SpectralField;
/// use katgpt_core::spectral_pencil::{DensePencil, init::seeded_dense};
///
/// let init = seeded_dense::<8, 4>(b"npc-42", 7);
/// let pencil = DensePencil { a0: init.a0, a: init.a };
/// let field = SpectralField::new(&pencil, 7, 5.0);
///
/// // Derived certificate — not a hand constant.
/// assert!(field.certificate().field_lipschitz.is_finite());
/// assert!(field.lipschitz_bound() == field.certificate().field_lipschitz);
///
/// // The field: eigenvalue-scaled state.
/// let z = [0.5_f32; 8];
/// let mut dz = [0.0_f32; 8];
/// let out = katgpt_core::committed_field_blend::ArchetypeFieldSource::evolve(
///     &field, &z, &mut dz,
/// );
/// let g = field.gate(&z);
/// assert!((out[3] - g * z[3]).abs() < 1e-6);
/// ```
#[derive(Clone, Debug)]
pub struct SpectralField<const D: usize, const N: usize> {
    /// Canonical-gauge pencil (orthogonal invariance ⇒ same function).
    pencil: DensePencil<D, N>,
    /// 0-indexed eigenvalue the gate reads.
    k: usize,
    cert: SpectralFieldCert<N>,
    commitment: [u8; 32],
}

impl<const D: usize, const N: usize> SpectralField<D, N> {
    /// Compile-time arity guard: features are `z[..N]`, so `N ≤ D`.
    const N_LE_D: () = assert!(N <= D, "SpectralField requires N <= D");

    /// Build the field from a pencil genome, canonicalizing the gauge and
    /// deriving the exact certificate. Bit-reproducible for the same input.
    ///
    /// `k` is the 0-indexed eigenvalue (the temperament rung); `box_r2`
    /// is the certified `‖z‖₂` radius the certificate is valid on (for a
    /// feature box of `‖x‖∞ ≤ R∞`, `box_r2 = R∞·√N` is the compatible
    /// ℓ2 radius; the seeded-init guarantee domain uses
    /// [`crate::spectral_pencil::init::BOX_R`]).
    #[must_use]
    pub fn new(pencil: &DensePencil<D, N>, k: usize, box_r2: f32) -> Self {
        let () = Self::N_LE_D;
        let mut scratch = DenseScratch::<D>::new();

        // Canonical gauge (idempotent, pinned) — same-input stable bytes.
        let (a0, a) = gauge::canonicalize(&pencil.a0, &pencil.a, &mut scratch);

        // Exact norms on the canonical form (conjugation-invariant values).
        let norm_a0 = norm_jacobi_exact(&a0, &mut scratch);
        let mut feature_norms = [0.0_f32; N];
        for (m, out) in a.iter().zip(feature_norms.iter_mut()) {
            *out = norm_jacobi_exact(m, &mut scratch);
        }
        let sum_sq: f64 = feature_norms
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum();
        let gate_lipschitz = (sum_sq as f32).sqrt();
        let growth_envelope = norm_a0 + box_r2 * gate_lipschitz;
        let field_lipschitz = growth_envelope + box_r2 * gate_lipschitz;

        // Commitment: BLAKE3 over (tag, k, box, canonical packed bytes).
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"spectral_field/v1");
        hasher.update(&(k as u64).to_le_bytes());
        hasher.update(&box_r2.to_le_bytes());
        hash_packed::<D>(&mut hasher, &a0);
        for m in &a {
            hash_packed::<D>(&mut hasher, m);
        }
        let commitment = *hasher.finalize().as_bytes();

        Self {
            pencil: DensePencil { a0, a },
            k: k.min(D - 1),
            cert: SpectralFieldCert {
                norm_a0,
                feature_norms,
                gate_lipschitz,
                growth_envelope,
                field_lipschitz,
                box_r2,
            },
            commitment,
        }
    }

    /// The derived certificate (FAME Lemma 1 input).
    #[must_use]
    pub fn certificate(&self) -> &SpectralFieldCert<N> {
        &self.cert
    }

    /// The stored canonical pencil (for attribution /
    /// [`crate::spectral_pencil::attribution::attribute`] — B2's ledger
    /// surface consumes this without duplicating anything).
    #[must_use]
    pub fn pencil(&self) -> &DensePencil<D, N> {
        &self.pencil
    }

    /// The temperament rung (0-indexed k).
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// The raw gate scalar `λk(A(z[..N]))` — the think-brain seam output.
    /// Scalar only, per the domain-discipline rule; allocates nothing.
    #[must_use]
    pub fn gate(&self, z: &[f32]) -> f32 {
        let mut x = [0.0_f32; N];
        for (xi, &zi) in x.iter_mut().zip(z.iter().take(N)) {
            *xi = zi;
        }
        let mut scratch = DenseScratch::<D>::new();
        self.pencil.eval(&x, self.k, &mut scratch).value
    }
}

impl<const D: usize, const N: usize> ArchetypeFieldSource<D> for SpectralField<D, N> {
    fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
        debug_assert!(z.len() >= D, "z must be at least D={D}");
        debug_assert!(
            dz_scratch.len() >= D,
            "dz_scratch must be at least D={D} elements"
        );
        let s = self.gate(z);
        for j in 0..D {
            dz_scratch[j] = s * z[j];
        }
        &mut dz_scratch[..D]
    }

    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    fn lipschitz_bound(&self) -> f32 {
        self.cert.field_lipschitz
    }
}

/// Hash one packed matrix's raw f32 bytes (deterministic serialization of
/// the canonical form — the √2 off-diagonal packing is part of the data).
fn hash_packed<const D: usize>(hasher: &mut blake3::Hasher, m: &SymPacked<D>) {
    for row in m.data.iter() {
        for v in row.iter() {
            hasher.update(&v.to_le_bytes());
        }
    }
}

/// The PoC-validated production temperament shape: D=8 ladder (8 rungs),
/// N=4 belief features (threat/evidence/safety/fatigue at the consumer).
pub type SpectralField8x4 = SpectralField<8, 4>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::committed_field_blend::CommittedFieldBlend;
    use crate::spectral_pencil::init::seeded_dense;
    use crate::spectral_pencil::shape::{nsd_diagonal_feature, rank_one_feature};

    /// Family-M-shaped genome (the PoC construction, public pieces only):
    /// seeded ladder A₀ + rank-one PSD threat/evidence + NSD safety/fatigue.
    fn family_m(seed: &[u8], k: usize) -> DensePencil<8, 4> {
        let init = seeded_dense::<8, 4>(seed, k);
        let beta_t = 0.9_f32;
        let beta_e = 0.4_f32;
        let mut dir = [0.0_f32; 8];
        let mut n2 = 0.0_f64;
        let mut rng = seed_u64(seed);
        for v in dir.iter_mut() {
            *v = next_unit(&mut rng);
            n2 += f64::from(*v) * f64::from(*v);
        }
        let inv = (n2.sqrt() as f32).recip();
        for v in dir.iter_mut() {
            *v *= inv;
        }
        let mut a0 = rank_one_feature::<8>(beta_t, &dir);
        for (i, row) in a0.data.iter_mut().enumerate() {
            row[i] += 0.05;
        }
        let mut a1 = rank_one_feature::<8>(beta_e, &dir);
        for (i, row) in a1.data.iter_mut().enumerate() {
            row[i] += 0.08;
        }
        DensePencil {
            a0: init.a0,
            a: [
                a0,
                a1,
                nsd_diagonal_feature::<8>(&[0.15; 8]),
                nsd_diagonal_feature::<8>(&[0.1; 8]),
            ],
        }
    }

    fn seed_u64(bytes: &[u8]) -> u64 {
        let h = blake3::hash(bytes);
        u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap())
    }

    /// Deterministic LCG in `[-1, 1)` (no fastrand dep here — keep the
    /// test dependency surface of this module empty).
    fn next_unit(rng: &mut u64) -> f32 {
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*rng >> 33) as f32 / 2.0_f32.powi(31)) * 2.0 - 1.0
    }

    /// Sample a point in the ℓ2 ball of radius `r` (rejection, bounded).
    fn sample_in_ball(rng: &mut u64, r: f32) -> [f32; 8] {
        for _ in 0..64 {
            let mut p = [0.0_f32; 8];
            let mut n2 = 0.0_f64;
            for v in p.iter_mut() {
                *v = next_unit(rng) * r;
                n2 += f64::from(*v) * f64::from(*v);
            }
            if (n2 as f32) <= r * r {
                return p;
            }
        }
        [0.0; 8] // bounded fallback (measure-zero path)
    }

    const R: f32 = 5.0;

    /// G1 — the field certificate holds: `‖F(z)−F(y)‖₂ ≤ L·‖z−y‖₂` on the
    /// certified box, zero violations across genomes × temperament rungs ×
    /// sampled pairs (the PoC envelope gate lifted to field level).
    #[test]
    fn certificate_field_lipschitz_holds() {
        let mut violations = 0_u32;
        let mut rng = 0x5EC7_0001_u64;
        for genome in 0..12 {
            let seed = format!("field-g1-{genome}").into_bytes();
            for k in [0_usize, 3, 7] {
                let field = SpectralField::new(&family_m(&seed, k), k, R);
                let l = field.certificate().field_lipschitz;
                assert!(l.is_finite() && l > 0.0);
                let mut dz_z = [0.0_f32; 8];
                let mut dz_y = [0.0_f32; 8];
                for _ in 0..80 {
                    let z = sample_in_ball(&mut rng, R);
                    let y = sample_in_ball(&mut rng, R);
                    ArchetypeFieldSource::evolve(&field, &z, &mut dz_z);
                    ArchetypeFieldSource::evolve(&field, &y, &mut dz_y);
                    let dxy: f64 = z
                        .iter()
                        .zip(y.iter())
                        .map(|(a, b)| {
                            let d = a - b;
                            f64::from(d) * f64::from(d)
                        })
                        .sum::<f64>()
                        .sqrt();
                    if dxy < 1e-9 {
                        continue; // degenerate pair
                    }
                    let df: f64 = dz_z
                        .iter()
                        .zip(dz_y.iter())
                        .map(|(a, b)| {
                            let d = a - b;
                            f64::from(d) * f64::from(d)
                        })
                        .sum::<f64>()
                        .sqrt();
                    if df > f64::from(l) * dxy + 1e-3 {
                        violations += 1;
                    }
                }
            }
        }
        assert_eq!(
            violations, 0,
            "field certificate violated {violations} times — the ‖A₀‖+2R·L_g envelope is wrong"
        );
    }

    /// G1b — the gate-level Weyl certificate holds directly:
    /// `|g(z)−g(y)| ≤ L_g·‖z[..N]−y[..N]‖₂`.
    #[test]
    fn certificate_gate_lipschitz_holds() {
        let mut violations = 0_u32;
        let mut rng = 0x6A7E_0002_u64;
        for genome in 0..12 {
            let seed = format!("field-g1b-{genome}").into_bytes();
            for k in [0_usize, 3, 7] {
                let field = SpectralField::new(&family_m(&seed, k), k, R);
                let lg = field.certificate().gate_lipschitz;
                for _ in 0..80 {
                    let z = sample_in_ball(&mut rng, R);
                    let y = sample_in_ball(&mut rng, R);
                    let dxy: f64 = z
                        .iter()
                        .take(4)
                        .zip(y.iter().take(4))
                        .map(|(a, b)| {
                            let d = a - b;
                            f64::from(d) * f64::from(d)
                        })
                        .sum::<f64>()
                        .sqrt();
                    if dxy < 1e-9 {
                        continue;
                    }
                    let dg = f64::from(field.gate(&z) - field.gate(&y)).abs();
                    if dg > f64::from(lg) * dxy + 1e-4 {
                        violations += 1;
                    }
                }
            }
        }
        assert_eq!(
            violations, 0,
            "gate certificate violated {violations} times — the √Σ‖Aᵢ‖² envelope is wrong"
        );
    }

    /// G1c — the growth envelope dominates the gate on the box:
    /// `|λk(A(z))| ≤ ‖A₀‖₂ + R·L_g`.
    #[test]
    fn growth_envelope_dominates_gate() {
        let mut violations = 0_u32;
        let mut rng = 0x6047_0003_u64;
        for genome in 0..8 {
            let seed = format!("field-g1c-{genome}").into_bytes();
            for k in [0_usize, 3, 7] {
                let field = SpectralField::new(&family_m(&seed, k), k, R);
                let g_max = field.certificate().growth_envelope;
                for _ in 0..60 {
                    let z = sample_in_ball(&mut rng, R);
                    if field.gate(&z).abs() > g_max + 1e-3 {
                        violations += 1;
                    }
                }
            }
        }
        assert_eq!(
            violations, 0,
            "growth envelope violated {violations} times — triangle bound is wrong"
        );
    }

    /// B1's composed-safety verification: a 3-temperament blend of spectral
    /// fields satisfies `CommittedFieldBlend::lipschitz_bound` empirically,
    /// and the composed value is the gated SUM (triangle inequality over
    /// `Σ gate_k·f_k`) — the form the 736 B1 fix corrected.
    #[test]
    fn composed_safety_bound_holds() {
        type Blend3x8 = CommittedFieldBlend<3, 8>;
        let genome = family_m(b"field-composed", 3);
        let f0 = SpectralField::new(&genome, 0, R);
        let f1 = SpectralField::new(&genome, 3, R);
        let f2 = SpectralField::new(&genome, 7, R);
        let fields: [&dyn ArchetypeFieldSource<8>; 3] = [&f0, &f1, &f2];

        // Directions + summary chosen so every π clamps to +pi_max → all
        // gates ≈ 1 (the regime that maximally exposes under-reporting).
        let dirs = [
            [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let summary = [10.0_f32; 8];

        let mut blend = Blend3x8::uncommitted();
        blend.commit(&summary, &dirs, &fields, 1);

        // Pin the composed form: Σ gate_k·L_k (≈ sigmoid(10)·ΣL at pi_max).
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let expected_sum: f32 = [
            f0.lipschitz_bound(),
            f1.lipschitz_bound(),
            f2.lipschitz_bound(),
        ]
        .iter()
        .enumerate()
        .map(|(kk, &lk)| sigmoid(blend.pi[kk] / blend.tau) * lk)
        .sum();
        let bound = blend.lipschitz_bound(&fields);
        assert!(
            (bound - expected_sum).abs() < 1e-2 * expected_sum,
            "composed bound {bound} != gated sum {expected_sum}"
        );
        assert!(
            bound > f1.lipschitz_bound(),
            "sum form must exceed the max single term"
        );

        // Empirical: the blended dynamics satisfy the composed bound.
        let mut violations = 0_u32;
        let mut rng = 0xC0AB_0004_u64;
        let mut scratch = [0.0_f32; 8];
        let mut out_z = [0.0_f32; 8];
        let mut out_y = [0.0_f32; 8];
        for _ in 0..400 {
            let z = sample_in_ball(&mut rng, R);
            let y = sample_in_ball(&mut rng, R);
            let dxy: f64 = z
                .iter()
                .zip(y.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    f64::from(d) * f64::from(d)
                })
                .sum::<f64>()
                .sqrt();
            if dxy < 1e-9 {
                continue;
            }
            blend.apply_blended(&fields, &z, &mut scratch, &mut out_z);
            blend.apply_blended(&fields, &y, &mut scratch, &mut out_y);
            let df: f64 = out_z
                .iter()
                .zip(out_y.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    f64::from(d) * f64::from(d)
                })
                .sum::<f64>()
                .sqrt();
            if df > f64::from(bound) * dxy + 1e-3 {
                violations += 1;
            }
        }
        assert_eq!(
            violations, 0,
            "composed safety bound violated {violations}/400 pairs — the gated-sum certificate chain is broken"
        );
    }

    /// Same input → same commitment bytes + bit-identical outputs.
    #[test]
    fn determinism_commitment_and_eval() {
        let genome = family_m(b"field-det", 5);
        let a = SpectralField::new(&genome, 5, R);
        let b = SpectralField::new(&genome, 5, R);
        assert_eq!(a.commitment(), b.commitment());
        let z = [0.3_f32, -0.7, 1.2, -2.0, 0.4, 0.9, -0.1, 2.2];
        let mut da = [0.0_f32; 8];
        let mut db = [0.0_f32; 8];
        let oa = ArchetypeFieldSource::evolve(&a, &z, &mut da);
        let ob = ArchetypeFieldSource::evolve(&b, &z, &mut db);
        assert_eq!(oa, ob);
    }

    /// Canonical storage is a fixed point up to float noise:
    /// re-canonicalizing the stored pencil reproduces it within the
    /// internal packed↔full round-trip re-rounding (~1 ulp on
    /// off-diagonals — the gauge idempotence contract at float precision).
    #[test]
    fn canonical_storage_is_fixed_point() {
        let field = SpectralField::new(&family_m(b"field-fix", 2), 2, R);
        let mut scratch = DenseScratch::<8>::new();
        let (a0, a) = gauge::canonicalize(&field.pencil().a0, &field.pencil().a, &mut scratch);
        let mut max_diff = 0.0_f32;
        for (row_a, row_b) in a0.data.iter().zip(field.pencil().a0.data.iter()) {
            for (x, y) in row_a.iter().zip(row_b.iter()) {
                max_diff = max_diff.max((x - y).abs());
            }
        }
        for (m, stored) in a.iter().zip(field.pencil().a.iter()) {
            for (row_a, row_b) in m.data.iter().zip(stored.data.iter()) {
                for (x, y) in row_a.iter().zip(row_b.iter()) {
                    max_diff = max_diff.max((x - y).abs());
                }
            }
        }
        assert!(
            max_diff < 1e-5,
            "canonical storage not a float-noise fixed point (max diff {max_diff})"
        );
    }

    /// Orthogonal conjugation preserves the FUNCTION and the certificate:
    /// a signed-permutation conjugate evaluates identically (float noise
    /// only) and its norms match.
    #[test]
    fn conjugation_preserves_function_and_certificate() {
        let field = SpectralField::new(&family_m(b"field-conj", 4), 4, R);

        // Signed permutation Q: swap axes 0↔3, negate axis 5.
        // A' = QᵀAQ is computed by index arithmetic — exact in floats.
        let conj = |m: &SymPacked<8>| -> SymPacked<8> {
            let full = m.to_full();
            let map = |i: usize| -> usize {
                match i {
                    0 => 3,
                    3 => 0,
                    j => j,
                }
            };
            let sgn = |i: usize| -> f32 {
                match i {
                    5 => -1.0,
                    _ => 1.0,
                }
            };
            let mut out = [[0.0_f32; 8]; 8];
            for r in 0..8 {
                for c in 0..8 {
                    // (Qᵀ A Q)[r][c] = sgn(r)·sgn(c)·A[map(r)][map(c)]
                    out[r][c] = sgn(r) * sgn(c) * full[map(r)][map(c)];
                }
            }
            SymPacked::pack_from_full(&out)
        };
        let pencil_prime = DensePencil {
            a0: conj(&field.pencil().a0),
            a: {
                let mut out = [SymPacked::<8>::zeroed(); 4];
                for (m, o) in field.pencil().a.iter().zip(out.iter_mut()) {
                    *o = conj(m);
                }
                out
            },
        };
        let field_prime = SpectralField::new(&pencil_prime, 4, R);

        // Same function: gates agree to float noise at several states.
        for t in 0..12_u64 {
            let mut rng = 0xC0A7_1000_u64 + t;
            let z = sample_in_ball(&mut rng, R);
            let g0 = field.gate(&z);
            let g1 = field_prime.gate(&z);
            assert!(
                (g0 - g1).abs() < 1e-3,
                "conjugation broke the function: {g0} vs {g1}"
            );
        }

        // Same certificate values (norms are conjugation-invariant).
        let (c0, c1) = (field.certificate(), field_prime.certificate());
        assert!((c0.norm_a0 - c1.norm_a0).abs() < 1e-3);
        for (a, b) in c0.feature_norms.iter().zip(c1.feature_norms.iter()) {
            assert!((a - b).abs() < 1e-3, "feature norm drifted: {a} vs {b}");
        }
    }

    /// k is clamped into range; commitment distinguishes k + box.
    #[test]
    fn commitment_distinguishes_params() {
        let genome = family_m(b"field-params", 1);
        let a = SpectralField::new(&genome, 1, R);
        let b = SpectralField::new(&genome, 2, R);
        let c = SpectralField::new(&genome, 1, 6.0);
        assert_ne!(a.commitment(), b.commitment(), "k must be committed");
        assert_ne!(
            a.commitment(),
            c.commitment(),
            "box radius must be committed"
        );
        assert_eq!(SpectralField::new(&genome, 99, R).k(), 7);
    }
}
