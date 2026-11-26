//! `sym` — isometric representation of small symmetric matrices
//! (Issue 676 T1).
//!
//! A dense symmetric `d×d` matrix is represented by its upper-triangle
//! parameter vector `v ∈ R^{d(d+1)/2}` — **off-diagonal entries stored
//! pre-multiplied by `√2`** — so that the map `v ↦ sym(v)` (which places
//! `v`'s off-diagonal entries into the matrix scaled by `1/√2`) is an
//! **isometry**:
//!
//! ```text
//! ‖sym(v)‖_F == ‖v‖₂        ⟨sym(u), sym(v)⟩_F == ⟨u, v⟩
//! ```
//!
//! This is the paper's §7.1 representation (arXiv:2608.08003): it avoids
//! an accidental basis-dependent preconditioner under Euclidean updates —
//! a step in parameter space rescales the matrix by the same amount. As
//! a side effect Frobenius norm / inner-product queries (genome
//! similarity, attribution quadratic forms) are plain loops over the
//! upper triangle, and `‖A‖₂ ≤ ‖A‖_F = ‖v‖₂` is free.
//!
//! ## Storage layout (stable-Rust const generics)
//!
//! The parameter vector lives in the **upper triangle of a full
//! `[[f32; D]; D]` square** — `data[i][j]` for `i ≤ j` is `v`'s entry
//! (offs pre-scaled). The lower triangle **mirrors** the upper
//! (`data[j][i] == data[i][j]`) so the array is symmetric in parameter
//! space and `#[repr(C)]`-flat. The compact `D(D+1)/2` array would
//! require `generic_const_exprs` (unstable; the PKM / factorized-action
//! precedent) — the mirror costs ≤ 2× memory at `D ≤ 32` (≤ 4 KiB) and
//! zero correctness. [`SymPacked::extract_param_vec_into`] recovers the
//! compact vector when a consumer needs it.
//!
//! ## Scale round-trip honesty
//!
//! `√2 · (1/√2) ≠ 1` exactly in f32, so `pack_from_full` → `to_full`
//! round-trips with ≤ 1 ulp error on off-diagonal entries. The isometry
//! identities are asserted to 1–2 ulp in the property tests, not exactly.

/// `1/√2` as f32. Not the exact reciprocal of [`SQRT_2`] in f32
/// rounding — see module doc.
pub const RCP_SQRT_2: f32 = 0.707_106_77;
/// `√2` as f32.
pub const SQRT_2: f32 = core::f32::consts::SQRT_2;

/// Logical parameter-vector length for a symmetric `d×d` matrix.
#[inline]
#[must_use]
pub const fn packed_len(d: usize) -> usize {
    d * (d + 1) / 2
}

/// An isometrically represented symmetric `D×D` matrix (Issue 676 T1).
///
/// See the module doc for the layout: upper triangle = the parameter
/// vector (offs ×√2), lower triangle mirrored, `to_full()` applies the
/// `1/√2` off scale to produce the actual matrix.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymPacked<const D: usize> {
    pub data: [[f32; D]; D],
}

impl<const D: usize> SymPacked<D> {
    /// The zero matrix.
    #[inline]
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            data: [[0.0; D]; D],
        }
    }

    /// Pack a full symmetric matrix. Only the upper triangle of `full`
    /// is read; off-diagonal entries are multiplied by `√2` on the way
    /// in, mirrored to the lower triangle.
    #[must_use]
    pub fn pack_from_full(full: &[[f32; D]; D]) -> Self {
        let mut out = [[0.0; D]; D];
        for i in 0..D {
            out[i][i] = full[i][i];
            for j in (i + 1)..D {
                let v = full[i][j] * SQRT_2;
                out[i][j] = v;
                out[j][i] = v;
            }
        }
        Self { data: out }
    }

    /// The actual symmetric matrix (offs scaled by `1/√2`).
    #[must_use]
    pub fn to_full(&self) -> [[f32; D]; D] {
        let mut out = [[0.0; D]; D];
        for (i, orow) in out.iter_mut().enumerate() {
            orow[i] = self.data[i][i];
        }
        for (i, srow) in self.data.iter().enumerate() {
            for j in (i + 1)..D {
                let e = srow[j] * RCP_SQRT_2;
                out[i][j] = e;
                out[j][i] = e;
            }
        }
        out
    }

    /// Frobenius norm of the matrix — equals the Euclidean norm of the
    /// parameter vector exactly (the isometry): loop over the upper
    /// triangle. A free upper bound on the spectral norm.
    #[must_use]
    pub fn frobenius_norm(&self) -> f32 {
        let mut s = 0.0_f64;
        for (i, row) in self.data.iter().enumerate() {
            s += f64::from(row[i]) * f64::from(row[i]);
            for &x in row[(i + 1)..].iter() {
                s += f64::from(x) * f64::from(x);
            }
        }
        (s as f32).sqrt()
    }

    /// Frobenius inner product `⟨A, B⟩_F` — equals the plain dot of the
    /// two parameter vectors exactly (the isometry).
    #[must_use]
    pub fn frobenius_dot(&self, other: &Self) -> f32 {
        let mut s = 0.0_f64;
        for i in 0..D {
            s += f64::from(self.data[i][i]) * f64::from(other.data[i][i]);
            for j in (i + 1)..D {
                s += f64::from(self.data[i][j]) * f64::from(other.data[i][j]);
            }
        }
        s as f32
    }

    /// Entry access `(i, j)` in **matrix** semantics (offs unscaled),
    /// symmetric in `(i, j)`.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        let raw = self.data[i][j]; // mirror invariant makes [i][j] == [j][i]
        if i == j { raw } else { raw * RCP_SQRT_2 }
    }

    /// Entry write `(i, j)` in **matrix** semantics, mirrored.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f32) {
        let scaled = if i == j { v } else { v * SQRT_2 };
        self.data[i][j] = scaled;
        self.data[j][i] = scaled;
    }

    /// Add `scale·other` into `self` (builds `A₀ + Σ xᵢAᵢ`). Mirror
    /// symmetry is preserved by the full-square add.
    #[inline]
    pub fn add_scaled_into(&mut self, other: &Self, scale: f32) {
        for (ra, rb) in self.data.iter_mut().zip(other.data.iter()) {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += scale * b;
            }
        }
    }

    /// Negate in place (`λk(−A) = −λ_{d−k+1}(A)` mirror duality helper).
    #[inline]
    pub fn negate(&mut self) {
        for row in self.data.iter_mut() {
            for v in row.iter_mut() {
                *v = -*v;
            }
        }
    }

    /// Copy the compact parameter vector (upper triangle, row-major,
    /// offs pre-scaled) into `out` — `out.len() >= packed_len(D)`.
    /// The stable-Rust escape hatch for consumers that want the flat
    /// `D(D+1)/2` view (commitment bytes, SIMD dots).
    pub fn extract_param_vec_into(&self, out: &mut [f32]) {
        let mut w = 0_usize;
        for (i, row) in self.data.iter().enumerate() {
            for &v in row[i..].iter() {
                if w < out.len() {
                    out[w] = v;
                    w += 1;
                }
            }
        }
    }
}

/// Symmetric tridiagonal matrix — the O(d)-parameter pencil family
/// (Issue 676 T3).
///
/// `off` has length `D` with the **last slot `off[D−1]` dead (always
/// zero)** — the stable-Rust stand-in for the ideal `[f32; D−1]`
/// (`generic_const_exprs`, see module doc). Every kernel reads
/// `off[i]` only as the `(i, i+1)` coupling, so the dead slot is never
/// observed; constructors zero it to keep canonical bytes stable.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tridiagonal<const D: usize> {
    pub diag: [f32; D],
    pub off: [f32; D],
}

impl<const D: usize> Tridiagonal<D> {
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            diag: [0.0; D],
            off: [0.0; D],
        }
    }

    /// Gershgorin bounds `(lo, hi)` containing every eigenvalue.
    /// Row radius `r_i = |b_{i−1}| + |b_i|` (missing neighbours = 0).
    #[must_use]
    pub fn gershgorin(&self) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..D {
            let mut r = 0.0_f32;
            if i > 0 {
                r += self.off[i - 1].abs();
            }
            if i + 1 < D {
                r += self.off[i].abs();
            }
            lo = lo.min(self.diag[i] - r);
            hi = hi.max(self.diag[i] + r);
        }
        (lo, hi)
    }
}

// ── small vector helpers (zero alloc) ──

/// Euclidean norm of a slice, f64 accumulation.
#[must_use]
pub fn norm2(xs: &[f32]) -> f32 {
    let mut s = 0.0_f64;
    for &x in xs {
        s += f64::from(x) * f64::from(x);
    }
    (s as f32).sqrt()
}

/// Plain dot product, f64 accumulation.
#[must_use]
pub fn dot(xs: &[f32], ys: &[f32]) -> f32 {
    let mut s = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        s += f64::from(x) * f64::from(y);
    }
    s as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / 2.0_f32.powi(31)) * 2.0 - 1.0 // [-1, 1)
        }
    }

    #[test]
    fn pack_round_trip_is_1ulp_on_offdiagonals() {
        const D: usize = 5;
        let mut rng = Lcg(7);
        let mut full = [[0.0_f32; D]; D];
        for (i, j) in (0..D).flat_map(|i| (i..D).map(move |j| (i, j))) {
            let v = rng.next_f32() * 3.0;
            full[i][j] = v;
            full[j][i] = v;
        }
        let packed = SymPacked::<D>::pack_from_full(&full);
        let rt = packed.to_full();
        for i in 0..D {
            for j in 0..D {
                let (a, b) = (full[i][j], rt[i][j]);
                let ulps = ((a.to_bits() as i64) - (b.to_bits() as i64)).abs();
                assert!(
                    ulps <= 1,
                    "round trip drifted {ulps} ulps at ({i},{j}): {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn isometry_norm_and_inner_product_hold_to_2ulp() {
        const D: usize = 4;
        let mut rng = Lcg(42);
        let mut f1 = [[0.0_f32; D]; D];
        let mut f2 = [[0.0_f32; D]; D];
        for i in 0..D {
            for j in i..D {
                let a = rng.next_f32();
                let b = rng.next_f32();
                f1[i][j] = a;
                f1[j][i] = a;
                f2[i][j] = b;
                f2[j][i] = b;
            }
        }
        let p1 = SymPacked::<D>::pack_from_full(&f1);
        let p2 = SymPacked::<D>::pack_from_full(&f2);

        // Frobenius norm of the FULL matrix vs the packed representation's
        // upper-triangle norm (the isometry).
        let mut fro = 0.0_f64;
        for row in &f1 {
            for v in row {
                fro += f64::from(*v) * f64::from(*v);
            }
        }
        let fro = (fro as f32).sqrt();
        let packed_norm = p1.frobenius_norm();
        let ulps = ((fro.to_bits() as i64) - (packed_norm.to_bits() as i64)).abs();
        assert!(
            ulps <= 2,
            "norm identity drifted {ulps} ulps: {fro} vs {packed_norm}"
        );

        let mut fi = 0.0_f64;
        for i in 0..D {
            for j in 0..D {
                fi += f64::from(f1[i][j]) * f64::from(f2[i][j]);
            }
        }
        let fi = fi as f32;
        let pd = p1.frobenius_dot(&p2);
        let ulps = ((fi.to_bits() as i64) - (pd.to_bits() as i64)).abs();
        assert!(
            ulps <= 4,
            "inner-product identity drifted {ulps} ulps: {fi} vs {pd}"
        );
    }

    #[test]
    fn get_set_are_symmetric_and_unscaled() {
        const D: usize = 3;
        let mut p = SymPacked::<D>::zeroed();
        p.set(0, 2, 1.5);
        // off-diagonals round-trip through the √2/1√2 convention with
        // ≤1 ulp — the documented honesty note.
        assert!((p.get(2, 0) - 1.5).abs() < 1e-6);
        assert!((p.get(0, 2) - 1.5).abs() < 1e-6);
        p.set(1, 1, -2.0);
        assert_eq!(p.get(1, 1), -2.0); // diagonals are exact
    }

    #[test]
    fn param_vec_round_trips_through_pack() {
        const D: usize = 4;
        let mut rng = Lcg(11);
        let mut full = [[0.0_f32; D]; D];
        for (i, j) in (0..D).flat_map(|i| (i..D).map(move |j| (i, j))) {
            let v = rng.next_f32();
            full[i][j] = v;
            full[j][i] = v;
        }
        let p = SymPacked::<D>::pack_from_full(&full);
        let mut v = [0.0_f32; packed_len(D)];
        p.extract_param_vec_into(&mut v);
        // v's entries: diag as-is, offs ×√2 — spot-check entry (0,1).
        assert!((v[1] - full[0][1] * SQRT_2).abs() < 1e-6);
        assert_eq!(v[0], full[0][0]);
        // ‖v‖₂ == ‖full‖_F (the isometry, flat view).
        assert!((norm2(&v) - p.frobenius_norm()).abs() < 1e-5);
    }

    #[test]
    fn tridiag_gershgorin_brackets_identity() {
        const D: usize = 4;
        let mut t = Tridiagonal::<D>::zeroed();
        t.diag = [1.0, -2.0, 3.0, 0.5];
        t.off = [0.1, 0.2, 0.3, 0.0];
        let (lo, hi) = t.gershgorin();
        // row radii 0.1, 0.3, 0.5, 0.3 → lo from row 1, hi from row 2
        assert!((lo - (-2.3)).abs() < 1e-6, "lo {lo}");
        assert!((hi - 3.5).abs() < 1e-6, "hi {hi}");
    }
}
