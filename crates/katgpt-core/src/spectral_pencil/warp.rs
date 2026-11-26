//! `warp` — the invertible monotone warp element `g(x) = λk(A + x·B)`,
//! `B ≻ 0` (Issue 676 T9; paper §9 future work, closed-form verified by
//! the No-GD panel via an inertia argument).
//!
//! With `B ≻ 0`, `g` is strictly increasing in `x` (Loewner monotonicity
//! on a line) — and its inverse is **closed-form**:
//!
//! ```text
//! g⁻¹(z) = λ_{d−k+1}( B^{−1/2}·(z·I − A)·B^{−1/2} )
//! ```
//!
//! (1-indexed λ; 0-indexed call sites use `values[D−1−k]`.) One
//! eigen-solve each direction — the first *provably bijective* bridge
//! element in the stack (the bridge-function family is otherwise
//! dot+sigmoid projections and clamps, neither invertible).
//!
//! ## B construction
//!
//! [`MonotoneWarp::from_directions`]: `B = I + Σ βᵢ·dᵢdᵢᵀ` with unit
//! directions — PD by construction (I + PSD ⪰ I ≻ 0), so
//! `B^{−1/2}` exists and is computed by the pinned Jacobi eigenpair
//! (`B = V·diag(w)·Vᵀ ⇒ B^{−1/2} = V·diag(1/√w)·Vᵀ`).

use crate::spectral_pencil::dense::{DenseScratch, jacobi_eigen};
use crate::spectral_pencil::sym::SymPacked;

/// One invertible monotone warp `g(x) = λk(A + xB)`, `B ≻ 0`.
///
/// 0-indexed `k` (`k < D`); the inverse reads the mirrored index
/// `D−1−k`.
pub struct MonotoneWarp<const D: usize> {
    pub a: SymPacked<D>,
    pub b: SymPacked<D>,
    /// Precomputed `B^{−1/2}` (canonical at construction; the warp is
    /// frozen — this is a modelless primitive, no training).
    pub b_inv_sqrt: SymPacked<D>,
    pub k: usize,
}

impl<const D: usize> MonotoneWarp<D> {
    /// Build from `A` and `B = I + Σ βᵢ·dᵢdᵢᵀ` over unit directions.
    /// `dirs.len() == betas.len()`; each `dirs[i].len() ≤ D`.
    #[must_use]
    pub fn from_directions(
        a: SymPacked<D>,
        betas: &[f32],
        dirs: &[[f32; 32]],
        k: usize,
        scratch: &mut DenseScratch<D>,
    ) -> Self {
        // B as a FULL matrix (offs must go through the √2 packing —
        // writing raw offs into SymPacked.data silently halves them in
        // to_full()).
        let mut bf = [[0.0_f32; D]; D];
        for (i, row) in bf.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        for (beta, dir) in betas.iter().zip(dirs.iter()) {
            for i in 0..D {
                for j in 0..D {
                    let di = if i < dir.len() { dir[i] } else { 0.0 };
                    let dj = if j < dir.len() { dir[j] } else { 0.0 };
                    bf[i][j] += beta * di * dj;
                }
            }
        }
        let b = SymPacked::pack_from_full(&bf);
        // B^{−1/2} via pinned Jacobi.
        jacobi_eigen(&bf, true, scratch);
        // V·diag(1/√w)·Vᵀ
        let mut bis = [[0.0_f32; D]; D];
        for (c1, row1) in scratch.v.iter().enumerate() {
            for (c2, row2) in scratch.v.iter().enumerate() {
                let mut acc = 0.0_f64;
                for (l, &vl1) in row1.iter().enumerate() {
                    let w = scratch.values[l].max(f32::MIN_POSITIVE);
                    acc += f64::from(vl1) * (1.0 / f64::from(w).sqrt()) * f64::from(row2[l]);
                }
                bis[c1][c2] = acc as f32;
            }
        }
        let b_inv_sqrt = SymPacked::pack_from_full(&bis);
        Self {
            a,
            b,
            b_inv_sqrt,
            k: k.min(D - 1),
        }
    }

    /// The forward map `g(x) = λk(A + xB)` — one Jacobi solve.
    #[must_use]
    pub fn g(&self, x: f32, scratch: &mut DenseScratch<D>) -> f32 {
        let mut ax = self.a;
        ax.add_scaled_into(&self.b, x);
        let full = ax.to_full();
        jacobi_eigen(&full, false, scratch);
        scratch.values[self.k]
    }

    /// The closed-form inverse `g⁻¹(z) = λ_{d−k+1}(B^{−1/2}(zI − A)B^{−1/2})`
    /// — one Jacobi solve (mirrored index).
    #[must_use]
    pub fn g_inv(&self, z: f32, scratch: &mut DenseScratch<D>) -> f32 {
        // M = z·I − A
        let mut m = self.a;
        m.negate();
        for (i, row) in m.data.iter_mut().enumerate() {
            row[i] += z;
        }
        let fm = m.to_full();
        let fb = self.b_inv_sqrt.to_full();
        // M' = B^{−1/2}·M·B^{−1/2}
        let mut t = [[0.0_f32; D]; D];
        for (r, trow) in t.iter_mut().enumerate() {
            for (c, tv) in trow.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for (l, &fl) in fm[r].iter().enumerate() {
                    acc += f64::from(fl) * f64::from(fb[l][c]);
                }
                *tv = acc as f32;
            }
        }
        let mut out = [[0.0_f32; D]; D];
        for (r, orow) in out.iter_mut().enumerate() {
            for (c, ov) in orow.iter_mut().enumerate() {
                // out[r][c] = Σ_l fb[r][l]·t[l][c] (fb symmetric)
                let mut acc = 0.0_f64;
                for l in 0..D {
                    acc += f64::from(fb[r][l]) * f64::from(t[l][c]);
                }
                *ov = acc as f32;
            }
        }
        jacobi_eigen(&out, false, scratch);
        scratch.values[D - 1 - self.k]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_pencil::init::seeded_dense;

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / 2.0_f32.powi(31)) * 2.0 - 1.0
        }
    }

    /// T9 G1: round-trip g⁻¹(g(x)) == x across random constructions —
    /// 10⁵ release tier / 1k debug.
    #[test]
    #[allow(clippy::needless_range_loop)] // symmetric matrix fill — both indices index `af`
    fn warp_round_trip_holds() {
        const D: usize = 6;
        const TRIALS: usize = if cfg!(debug_assertions) {
            1_000
        } else {
            100_000
        };
        let mut scratch = DenseScratch::<D>::new();
        let mut rng = Lcg(2026);
        for t in 0..TRIALS {
            // random A via the seeded-init QR machinery is expensive per
            // trial; draw a small random symmetric directly.
            let mut af = [[0.0_f32; D]; D];
            for i in 0..D {
                for j in i..D {
                    let v = rng.next_f32();
                    af[i][j] = v;
                    af[j][i] = v;
                }
            }
            let a = crate::spectral_pencil::sym::SymPacked::pack_from_full(&af);
            // B = I + 2 rank-one PSD directions (random betas ≥ 0).
            let betas = [rng.next_f32().abs() + 0.1, rng.next_f32().abs() + 0.1];
            let mut dirs = [[0.0_f32; 32]; 2];
            for d in dirs.iter_mut() {
                let mut n2 = 0.0_f32;
                for e in d.iter_mut().take(D) {
                    *e = rng.next_f32();
                    n2 += *e * *e;
                }
                let n = n2.sqrt().max(f32::MIN_POSITIVE);
                for e in d.iter_mut().take(D) {
                    *e /= n;
                }
            }
            let k = t % D;
            let warp = MonotoneWarp::<D>::from_directions(a, &betas, &dirs, k, &mut scratch);
            let x = rng.next_f32() * 4.0 - 2.0;
            let z = warp.g(x, &mut scratch);
            let x2 = warp.g_inv(z, &mut scratch);
            assert!(
                (x - x2).abs() < 1e-3,
                "trial {t} k {k}: round trip {x} → {z} → {x2}"
            );
        }
    }

    /// Monotonicity of g (Loewner on the line) — the design property
    /// the flow-model consumer (paper §9) needs.
    #[test]
    fn warp_is_strictly_increasing() {
        const D: usize = 5;
        let mut scratch = DenseScratch::<D>::new();
        let init = seeded_dense::<D, 2>(b"warp-monotone", 2);
        let betas = [0.4_f32, 0.9];
        let mut dirs = [[0.0_f32; 32]; 2];
        for (i, d) in dirs.iter_mut().enumerate() {
            for (j, e) in d.iter_mut().enumerate().take(D) {
                *e = ((i * 5 + j * 3) % 7) as f32 / 7.0 - 0.5;
            }
        }
        for k in [0_usize, 1, D / 2, D - 1] {
            let warp = MonotoneWarp::<D>::from_directions(init.a0, &betas, &dirs, k, &mut scratch);
            let mut prev = f32::NEG_INFINITY;
            for step in 0..50 {
                let x = -3.0 + 6.0 * (step as f32) / 49.0;
                let g = warp.g(x, &mut scratch);
                assert!(g >= prev - 1e-6, "k {k} step {step}: {g} < {prev}");
                prev = g;
            }
        }
    }
}
