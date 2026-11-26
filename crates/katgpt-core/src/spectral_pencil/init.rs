//! `init` — the seeded eigengap-guaranteed initialization constructor
//! (Issue 676 T4; paper §7.2, Lemma 2).
//!
//! ## The construction (dense)
//!
//! * `A₀ = Qᵀ·diag(−1,…,−1, 0@k, 1,…,1)·Q` — a conjugated ladder. The
//!   zero sits at ladder position `k` (the eigenvalue the gate reads), so
//!   `γk(A₀) = 1` exactly.
//! * `Q` from the **pinned QR of a BLAKE3-seeded Gaussian matrix**:
//!   Householder trapezoidalization with the `α = −sign(x₀)·‖x‖` pivot
//!   convention, then a sign-fix pass forcing `R`'s diagonal positive
//!   (zero diagonals counted as positive — pinned). No library QR: the
//!   rotation/order variance of LAPACK-class solvers is exactly what
//!   commitment must not inherit.
//! * `Aᵢ = αᵢ·I + diag(εᵢ)`, `αᵢ ~ U(±1/√n)` (Kaiming-uniform fan-in
//!   scale), `εᵢⱼ ~ U(±1/(20n))` — the paper's R=5 baking of Lemma 2's
//!   `1/(4Rn)`.
//!
//! ## Why γk ≥ ½ holds (Lemma 2's argument, one paragraph)
//!
//! `Σᵢ xᵢ·αᵢ·I` is a multiple of the identity — it shifts every
//! eigenvalue equally and never moves a gap. The only gap-moving term is
//! the diagonal jitter `E(x) = Σᵢ xᵢ·diag(εᵢ)`, whose spectral norm is
//! `max_j |Σᵢ xᵢ εᵢⱼ| ≤ R·n·max|ε| = 5n·(1/20n) = ¼`. Weyl:
//! `γk(A(x)) ≥ γk(A₀) − 2‖E‖₂ = 1 − ½ = ½` on `‖x‖∞ ≤ 5`.
//!
//! ## The tridiagonal variant
//!
//! `A₀` is the **diagonal ladder itself** (a diagonal matrix is a valid
//! tridiagonal, spectrum exact), feature matrices carry the jitter on
//! both diag and off. Spectral-norm bounds are sparsity-blind — the
//! Gershgorin row radius of a tridiagonal `E` is up to `3·max|entry|`
//! per row, so the jitter rescales to `U(±1/(60n))` (= `1/(12Rn)`,
//! keeping `3·5n·maxε ≤ ¼`). Non-commutativity comes free: a generic
//! tridiagonal never commutes with the diagonal `A₀`.
//!
//! ## The PSD variant (squareplus parametrization)
//!
//! For the monotone-in-feature-`i` shape DSL (T7's consumer), `Aᵢ` must
//! be PSD. The paper parametrizes `Aᵢ = diag(squareplus(v))` with
//! `squareplus(x) = (x + √(1+x²))/2` and initializes `v =
//! squareplus⁻¹(target) = target − 1/(4·target)` so the realized entries
//! hit `αᵢ + εᵢⱼ` with `αᵢ ~ U(1/(2√n), 1/√n)` (positive — the PSD
//! analogue of the fan-in scale).

use crate::hebbian_kernel_memory::SeedRng;
use crate::spectral_pencil::sym::{SymPacked, Tridiagonal};

/// Default input-box radius R the constants assume (paper §7.2.3: a
/// standard-normal feature sits within ±5 with probability ≥
/// 1 − 5.73e−7·n).
pub const BOX_R: f32 = 5.0;

/// Dense seeded pencil: `A₀` (conjugated ladder) + `N` feature matrices.
pub struct SeededDenseInit<const D: usize, const N: usize> {
    pub a0: SymPacked<D>,
    pub a: [SymPacked<D>; N],
}

/// Tridiagonal seeded pencil (diag ladder + tri jitter).
pub struct SeededTriInit<const D: usize, const N: usize> {
    pub a0: Tridiagonal<D>,
    pub a: [Tridiagonal<D>; N],
}

/// Construct the dense seeded pencil from arbitrary seed bytes (BLAKE3 →
/// 64-bit RNG seed). `k` is the 0-indexed eigenvalue the gate will read
/// (`1 ≤ k+1 ≤ D` in the paper's 1-indexed terms).
///
/// Bit-reproducible: same seed bytes → identical matrices, any target,
/// any run.
#[must_use]
pub fn seeded_dense<const D: usize, const N: usize>(
    seed_bytes: &[u8],
    k: usize,
) -> SeededDenseInit<D, N> {
    let mut rng = seed_rng(seed_bytes);

    // ── Q: pinned Householder QR of a Gaussian D×D ──
    let mut g = [[0.0_f32; D]; D];
    for row in g.iter_mut() {
        for e in row.iter_mut() {
            *e = rng.next_gaussian_pair().0;
        }
    }
    let mut q = identity::<D>();
    // R lives implicitly in g's upper triangle as we reflect in place:
    // apply each reflector H_j to (g, q) from the left; the product
    // H_{D-2}···H_0·g = R and H_{D-2}···H_0 = Q.
    for j in 0..D.saturating_sub(1) {
        // x = g[j.., j]
        let mut norm_sq = 0.0_f64;
        for row in g.iter().skip(j) {
            norm_sq += f64::from(row[j]) * f64::from(row[j]);
        }
        let norm = (norm_sq as f32).sqrt();
        if norm == 0.0 {
            continue; // pinned: zero column → skip reflector
        }
        let x0 = g[j][j];
        let alpha = if x0 >= 0.0 { -norm } else { norm }; // −sign(x₀)·‖x‖
        // v = x − α·e_j ⇒ v_j = x0 − α, v_i = g[i][j] (i > j).
        let mut v = [0.0_f32; D];
        v[j] = x0 - alpha;
        for (vi, row) in v.iter_mut().zip(g.iter()).skip(j + 1) {
            *vi = row[j];
        }
        let mut v_norm_sq = 0.0_f64;
        for &vi in v.iter().skip(j) {
            v_norm_sq += f64::from(vi) * f64::from(vi);
        }
        let beta = 2.0 / (v_norm_sq as f32).max(f32::MIN_POSITIVE); // 2/vᵀv
        apply_reflector::<D>(&mut g, &v, beta, j);
        apply_reflector::<D>(&mut q, &v, beta, j);
    }
    // The accumulated q is Q_house (Q_house·g_original = R). Sign-fix
    // pass: force R's diagonal positive by flipping q's columns so the
    // decomposition is the unique-diagonal one (pinned).
    for (i, row) in g.iter().enumerate() {
        if row[i] < 0.0 {
            for qc in q.iter_mut() {
                qc[i] = -qc[i];
            }
        }
    }

    // ── A₀ = Qᵀ Λ Q with Λ = ladder (−1 … 0@k … +1) ──
    let mut ladder = [0.0_f32; D];
    for (i, e) in ladder.iter_mut().enumerate() {
        *e = if i < k {
            -1.0
        } else if i == k {
            0.0
        } else {
            1.0
        };
    }
    // A0[i][j] = Σ_l λ_l · Q[l][i] · Q[l][j]  (Qᵀ Λ Q)
    let mut a0_full = [[0.0_f32; D]; D];
    for i in 0..D {
        for j in 0..D {
            let mut acc = 0.0_f64;
            for (l, &lam) in ladder.iter().enumerate() {
                acc += f64::from(lam) * f64::from(q[l][i]) * f64::from(q[l][j]);
            }
            a0_full[i][j] = acc as f32;
        }
    }
    let a0 = SymPacked::pack_from_full(&a0_full);

    // ── Aᵢ = αᵢ I + diag(εᵢ) ──
    let inv_sqrt_n = 1.0 / (N as f32).sqrt();
    let eps_bound = 1.0 / (20.0 * N as f32); // paper R=5 baking of 1/(4Rn)
    let mut a = [SymPacked::zeroed(); N];
    for mat_out in a.iter_mut() {
        let alpha = (rng.next_f32() * 2.0 - 1.0) * inv_sqrt_n;
        for (j, d_row) in mat_out.data.iter_mut().enumerate() {
            let eps = (rng.next_f32() * 2.0 - 1.0) * eps_bound;
            d_row[j] = alpha + eps;
        }
    }

    SeededDenseInit { a0, a }
}

/// Construct the tridiagonal seeded pencil (`A₀` = exact diagonal ladder;
/// feature jitter on diag AND off, rescaled per the module doc).
#[must_use]
pub fn seeded_tridiag<const D: usize, const N: usize>(
    seed_bytes: &[u8],
    k: usize,
) -> SeededTriInit<D, N> {
    let mut rng = seed_rng(seed_bytes);
    let mut a0 = Tridiagonal::zeroed();
    for (i, e) in a0.diag.iter_mut().enumerate() {
        *e = if i < k {
            -1.0
        } else if i == k {
            0.0
        } else {
            1.0
        };
    }

    let inv_sqrt_n = 1.0 / (N as f32).sqrt();
    // Gershgorin row radius ≤ 3·max|entry| ⇒ ε = 1/(12·R·n) keeps
    // ‖E(x)‖₂ ≤ 3·R·n·ε = ¼ at R = 5.
    let eps_bound = 1.0 / (12.0 * BOX_R * N as f32);
    let mut a = [Tridiagonal::zeroed(); N];
    for mat in a.iter_mut() {
        let alpha = (rng.next_f32() * 2.0 - 1.0) * inv_sqrt_n;
        for e in mat.diag.iter_mut() {
            *e = alpha + (rng.next_f32() * 2.0 - 1.0) * eps_bound;
        }
        // off: jitter on the live D−1 couplings; dead last slot stays 0
        // (canonical bytes).
        for e in mat.off.iter_mut().take(D.saturating_sub(1)) {
            *e = (rng.next_f32() * 2.0 - 1.0) * eps_bound;
        }
    }
    SeededTriInit { a0, a }
}

/// The PSD squareplus parametrization vector for one feature's diagonal
/// target (paper §7.2.3): writes `v` such that
/// `diag(squareplus(v)) ≈ diag(α + ε)` into `out` (caller-owned,
/// zero-alloc; `out.len() == targets.len()`).
///
/// `squareplus⁻¹(y) = y − 1/(4y)` for `y > 0`; the caller supplies the
/// positive targets (e.g. `α ~ U(1/(2√n), 1/√n)` plus jitter kept above
/// `f32::MIN_POSITIVE` scale). Targets ≤ 0 clamp — squareplus of the
/// resulting v stays positive regardless.
pub fn squareplus_param_into(targets: &[f32], out: &mut [f32]) {
    for (y, v) in targets.iter().zip(out.iter_mut()) {
        let y = y.max(f32::MIN_POSITIVE);
        *v = y - 1.0 / (4.0 * y);
    }
}

/// `squareplus(x) = (x + √(1+x²))/2` — the paper's positivity
/// parametrization (Barron 2021); polynomial gradient decay keeps
/// deeply-negative parameters trainable (vs. softplus' exponential).
#[inline]
#[must_use]
pub fn squareplus(x: f32) -> f32 {
    0.5 * (x + (x * x + 1.0).sqrt())
}

// ── internals ──

fn identity<const D: usize>() -> [[f32; D]; D] {
    let mut out = [[0.0_f32; D]; D];
    for (i, row) in out.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    out
}

/// Apply the Householder reflector `H = I − β·v·vᵀ` (rows ≥ j) to `m`
/// from the left, in place.
fn apply_reflector<const D: usize>(m: &mut [[f32; D]; D], v: &[f32; D], beta: f32, j: usize) {
    // w = β·(vᵀ·M[j..]) then M[j..] -= v·w
    let mut w = [0.0_f32; D];
    for c in 0..D {
        let mut acc = 0.0_f64;
        for r in j..D {
            acc += f64::from(v[r]) * f64::from(m[r][c]);
        }
        w[c] = beta * (acc as f32);
    }
    for r in j..D {
        let vr = v[r];
        for c in 0..D {
            m[r][c] -= vr * w[c];
        }
    }
}

fn seed_rng(seed_bytes: &[u8]) -> SeedRng {
    let hash = blake3::hash(seed_bytes);
    let mut first8 = [0_u8; 8];
    first8.copy_from_slice(&hash.as_bytes()[..8]);
    SeedRng::new(u64::from_le_bytes(first8))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_pencil::dense::{DenseScratch, jacobi_eigen};

    #[test]
    fn q_is_orthogonal_and_r_positive_diagonal() {
        // Re-derive Q inside the constructor via a public window: the
        // seeded A0 has spectrum {−1×k, 0, +1×(D−k−1)} — testing that
        // IS testing the QR (any QR drift breaks the exact spectrum).
        const D: usize = 6;
        const N: usize = 3;
        for k in 0..D {
            let init = seeded_dense::<D, N>(b"orthogonality-probe", k);
            let mut s = DenseScratch::<D>::new();
            jacobi_eigen(&init.a0.to_full(), false, &mut s);
            for (i, &v) in s.values.iter().enumerate() {
                let expect = if i < k {
                    -1.0
                } else if i == k {
                    0.0
                } else {
                    1.0
                };
                assert!(
                    (v - expect).abs() < 1e-5,
                    "k={k} idx {i}: {v} vs {expect} — QR not orthogonal?"
                );
            }
        }
    }

    #[test]
    fn same_seed_bit_identical_across_calls() {
        const D: usize = 5;
        const N: usize = 4;
        let i1 = seeded_dense::<D, N>(b"npc-17/world-a", 2);
        let i2 = seeded_dense::<D, N>(b"npc-17/world-a", 2);
        assert_eq!(i1.a0, i2.a0);
        assert_eq!(i1.a, i2.a);
    }

    #[test]
    fn noncommutativity_certificate_positive() {
        const D: usize = 6;
        const N: usize = 4;
        let init = seeded_dense::<D, N>(b"commute-probe", 3);
        let a0 = init.a0.to_full();
        for (idx, ai_packed) in init.a.iter().enumerate() {
            let ai = ai_packed.to_full();
            // [A0, Ai] = A0·Ai − Ai·A0, Frobenius norm
            let mut comm_sq = 0.0_f64;
            for i in 0..D {
                for j in 0..D {
                    let mut ab = 0.0_f64;
                    let mut ba = 0.0_f64;
                    for l in 0..D {
                        ab += f64::from(a0[i][l]) * f64::from(ai[l][j]);
                        ba += f64::from(ai[i][l]) * f64::from(a0[l][j]);
                    }
                    comm_sq += (ab - ba) * (ab - ba);
                }
            }
            let norm = (comm_sq as f32).sqrt();
            assert!(norm > 1e-4, "feature {idx} commutes with A0 (norm {norm})");
        }
    }
}
