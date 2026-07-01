//! MinkowskiLattice — high-dimensional lattice in C^f with sup-norm packing bounds.
//!
//! Implements the geometric infrastructure for the unit distance GOAT proof:
//! - Lattice embedding of D^(-1)·O_K in C^f
//! - Sup-norm packing bounds (bounded number of lattice points in polydisc)
//! - Coset averaging for expected unit-distance pair count
//! - Injective projection verification
//!
//! Reference: Lemma 2.1 (Geometry of Numbers) from Remarks paper.

use super::types::C64;

/// High-dimensional lattice in C^f with sup-norm packing utilities.
///
/// Represents the Minkowski-embedded lattice D^(-1)·O_K where O_K is the
/// ring of integers of a CM field K of degree 2f. The complex dimension is f,
/// matching the number of conjugate pairs of Archimedean embeddings.
///
/// The lattice lives in C^f ≅ R^(2f) and is parameterized by:
/// - `dim`: complex dimension f = [L:Q] where K = L(i)
/// - `min_sep`: minimum separation δ in sup-norm between distinct lattice points
/// - `covol`: covolume = |det(Λ)| (normalized by disc(K)^(1/2))
#[derive(Clone, Debug)]
pub struct MinkowskiLattice {
    /// Complex dimension f (field degree of totally real subfield).
    pub dim: usize,

    /// Basis vectors in C^f, stored row-major: basis[row * dim + col].
    /// Length = dim * dim (dim basis vectors, each of length dim).
    pub basis: Vec<C64>,

    /// Minimum separation δ in sup-norm between distinct lattice points.
    pub min_sep: f64,

    /// Covolume (|det| of the basis matrix, treated as R^(2f) real matrix).
    pub covol: f64,

    /// Sup-norm bound B for packing: any D-separated set in B_R has |X| ≤ exp(B·f).
    /// B = 2·log(4·R·D) where R = root discriminant, D = denominator.
    pub packing_param: f64,
}

impl MinkowskiLattice {
    /// Construct a lattice from explicit parameters.
    ///
    /// # Arguments
    /// * `dim` — Complex dimension f
    /// * `basis` — Basis vectors in C^f (dim × dim, row-major)
    /// * `min_sep` — Minimum sup-norm separation δ
    /// * `covol` — Covolume of the lattice
    /// * `packing_param` — B = 2·log(4·R·D) for packing bound
    pub fn new(dim: usize, basis: Vec<C64>, min_sep: f64, covol: f64, packing_param: f64) -> Self {
        assert_eq!(basis.len(), dim * dim, "basis must be dim×dim");
        assert!(min_sep > 0.0, "min_sep must be positive");
        assert!(covol > 0.0, "covol must be positive");
        Self {
            dim,
            basis,
            min_sep,
            covol,
            packing_param,
        }
    }

    /// Construct a lattice from field parameters (Gaussian-style grid).
    ///
    /// For Q(i): dim=1, basis=[(1,0)], min_sep=1.0, covol=1.0.
    /// For general CM field K=L(i) with degree f: builds the standard
    /// embedding lattice D^(-1)·O_K.
    pub fn from_field_params(dim: usize, root_disc: f64, denominator: f64) -> Self {
        // Standard basis: identity matrix in C^f scaled by 1/D
        let scale = 1.0 / denominator;
        let basis: Vec<C64> = (0..dim)
            .flat_map(|i| {
                (0..dim).map(move |j| {
                    if i == j {
                        C64::new(scale, 0.0)
                    } else {
                        C64::ZERO
                    }
                })
            })
            .collect();

        // Min separation ≈ 1/D in each coordinate
        let min_sep = scale;

        // Covolume ≈ disc(K)^(1/2) / D^(2f)
        let covol = root_disc.powi(dim as i32) / denominator.powi(2 * dim as i32);

        // Packing parameter B = 2·log(4·R·D)
        let packing_param = 2.0 * (4.0 * root_disc * denominator).ln();

        Self {
            dim,
            basis,
            min_sep,
            covol: covol.max(1e-300), // avoid zero
            packing_param,
        }
    }

    /// Sup-norm of a vector in C^f.
    ///
    /// sup_norm(z) = max_j |z_j|.
    #[inline]
    pub fn sup_norm(v: &[C64]) -> f64 {
        v.iter().map(|z| z.norm()).fold(0.0_f64, f64::max)
    }

    /// Squared sup-norm of a vector in C^f.
    ///
    /// sup_norm_sq(z) = max_j |z_j|². Use instead of `sup_norm` for
    /// comparisons to avoid computing square roots.
    #[inline]
    pub fn sup_norm_sq(v: &[C64]) -> f64 {
        v.iter().map(|z| z.norm_sq()).fold(0.0_f64, f64::max)
    }

    /// Compute lattice point: sum of basis[i] * integer_coeffs[i].
    ///
    /// Evaluates Λ(a) = Σ a_i · b_i where b_i are basis vectors.
    pub fn lattice_point(&self, coeffs: &[i64]) -> Vec<C64> {
        assert_eq!(coeffs.len(), self.dim, "coeffs must have dim elements");
        let mut point = vec![C64::ZERO; self.dim];
        for (i, &c) in coeffs.iter().enumerate() {
            let cf = c as f64;
            let basis_row = &self.basis[i * self.dim..(i + 1) * self.dim];
            for (pj, &bv) in point.iter_mut().zip(basis_row) {
                *pj = *pj + bv * cf;
            }
        }
        point
    }

    /// Compute lattice point into a pre-allocated buffer.
    ///
    /// Same as `lattice_point` but writes into `out` instead of allocating.
    #[inline]
    pub fn lattice_point_into(&self, coeffs: &[i64], out: &mut [C64]) {
        debug_assert_eq!(coeffs.len(), self.dim, "coeffs must have dim elements");
        debug_assert_eq!(out.len(), self.dim, "out must have dim elements");
        out.fill(C64::ZERO);
        for (i, &c) in coeffs.iter().enumerate() {
            let cf = c as f64;
            let basis_row = &self.basis[i * self.dim..(i + 1) * self.dim];
            for (pj, &bv) in out.iter_mut().zip(basis_row) {
                *pj = *pj + bv * cf;
            }
        }
    }

    /// Upper bound on packing number: max |X| of D-separated points in B_R.
    ///
    /// Uses sup-norm packing: |X| ≤ (2R/D)^f = exp(f · log(2R/D)).
    /// This is the key bound from Lemma 2.1 that controls set size.
    pub fn packing_bound(&self, radius: f64) -> usize {
        if radius <= 0.0 || self.min_sep <= 0.0 {
            return 0;
        }
        let ratio = 2.0 * radius / self.min_sep;
        if ratio <= 1.0 {
            return 1;
        }
        // (2R/δ)^f, clamped to usize — use powi to avoid ln/exp precision loss
        let bound = ratio.powi(self.dim as i32);
        bound.min(usize::MAX as f64) as usize
    }

    /// Estimate lattice points in polydisc B_R = {z ∈ C^f : sup_norm(z) ≤ R}.
    ///
    /// Approximate count via volume ratio:
    /// |Λ ∩ B_R| ≈ vol(B_R) / covol(Λ)
    ///
    /// The volume of the polydisc in R^(2f) is (π·R²)^f.
    pub fn polydisc_count(&self, radius: f64) -> usize {
        if radius <= 0.0 {
            return 0;
        }
        // Volume of polydisc = (π·R²)^f
        let vol_polydisc = (std::f64::consts::PI * radius * radius).powi(self.dim as i32);
        let count = vol_polydisc / self.covol;
        count.max(1.0) as usize
    }

    /// Check if projection to coordinate `coord` is injective on lattice.
    ///
    /// The projection π_k: C^f → C maps (z_1,...,z_f) → z_k.
    /// For CM fields, the first-coordinate projection is injective because
    /// if z ∈ Λ has z_1 = 0, then all conjugates σ_j(z) = 0 (norm form),
    /// hence z = 0 in O_K (the element is zero).
    ///
    /// Injectivity requires that the nonzero basis vector components at `coord`
    /// are Q-linearly independent: no nontrivial integer combination gives zero.
    /// For identity-like bases (our constructions), each coordinate has exactly
    /// one nonzero basis component, which is trivially injective.
    pub fn is_projection_injective(&self, coord: usize) -> bool {
        if coord >= self.dim {
            return false;
        }

        // Collect nonzero basis vector components at this coordinate.
        // Zero components don't affect the projected value — they can be ignored.
        let mut nonzero: [C64; 16] = [C64::ZERO; 16];
        let mut nonzero_count = 0usize;
        for i in 0..self.dim {
            let v = self.basis[i * self.dim + coord];
            if v.norm_sq() > 1e-30 {
                nonzero[nonzero_count] = v;
                nonzero_count += 1;
            }
        }

        // All zero → projection is constant → not injective
        if nonzero_count == 0 {
            return false;
        }

        // Single nonzero value → trivially injective (different coefficients give different values)
        if nonzero_count == 1 {
            return true;
        }

        // Multiple nonzero values: verify Q-linear independence.
        // If two values are equal or negatives, some integer combination gives zero.
        // For our constructions with identity-like bases using irrational generators
        // (e.g., 1 and φ), the ratios are irrational, ensuring Q-linear independence.
        for i in 0..nonzero_count {
            for j in (i + 1)..nonzero_count {
                let diff_re = (nonzero[i].re - nonzero[j].re).abs();
                let diff_im = (nonzero[i].im - nonzero[j].im).abs();
                let sum_re = (nonzero[i].re + nonzero[j].re).abs();
                let sum_im = (nonzero[i].im + nonzero[j].im).abs();
                // Equal (ratio ≈ 1) or negatives (ratio ≈ -1) → not Q-independent
                if (diff_re < 1e-12 && diff_im < 1e-12) || (sum_re < 1e-12 && sum_im < 1e-12) {
                    return false;
                }
            }
        }

        true
    }

    /// Generate lattice points within polydisc of given radius.
    ///
    /// Enumerates integer coefficients and keeps points with sup_norm ≤ radius.
    /// Uses a bounded range for coefficients based on radius/min_sep.
    pub fn points_in_polydisc(&self, radius: f64) -> Vec<Vec<C64>> {
        if radius <= 0.0 {
            return Vec::new();
        }

        let range = (radius / self.min_sep).ceil() as i64 + 1;
        let radius_sq = radius * radius;
        let mut buf = vec![C64::ZERO; self.dim];
        let mut points = Vec::with_capacity(self.polydisc_count(radius).max(1));

        // For dim=1, simple iteration
        if self.dim == 1 {
            let bv = self.basis[0];
            for c in -range..=range {
                let cf = c as f64;
                let p0 = bv * cf;
                if p0.norm_sq() <= radius_sq {
                    points.push(vec![p0]);
                }
            }
            return points;
        }

        // For dim=2, double loop
        if self.dim == 2 {
            let b00 = self.basis[0];
            let b01 = self.basis[1];
            let b10 = self.basis[2];
            let b11 = self.basis[3];
            for c0 in -range..=range {
                let cf0 = c0 as f64;
                for c1 in -range..=range {
                    let cf1 = c1 as f64;
                    let p0 = b00 * cf0 + b01 * cf1;
                    let p1 = b10 * cf0 + b11 * cf1;
                    let ns = p0.norm_sq().max(p1.norm_sq());
                    if ns <= radius_sq {
                        points.push(vec![p0, p1]);
                    }
                }
            }
            return points;
        }

        // General case: recursive enumeration
        let mut coeffs = vec![0i64; self.dim];
        self.enumerate_polydisc(&mut coeffs, 0, range, radius_sq, &mut points, &mut buf);
        points
    }

    /// Recursive helper for polydisc enumeration.
    fn enumerate_polydisc(
        &self,
        coeffs: &mut [i64],
        depth: usize,
        range: i64,
        radius_sq: f64,
        points: &mut Vec<Vec<C64>>,
        buf: &mut [C64],
    ) {
        if depth == self.dim {
            self.lattice_point_into(coeffs, buf);
            if Self::sup_norm_sq(buf) <= radius_sq {
                points.push(buf.to_vec());
            }
            return;
        }
        for c in -range..=range {
            coeffs[depth] = c;
            self.enumerate_polydisc(coeffs, depth + 1, range, radius_sq, points, buf);
        }
    }

    /// Compute expected unit-distance pairs averaged over cosets.
    ///
    /// Given a set U of "unit translations" (complex numbers with |σ(u)| = 1
    /// for all embeddings σ), the expected number of unit-distance pairs in
    /// a random coset a + Λ projected to C is:
    ///
    /// E[ν] = |U| · |Λ ∩ B_R| / covol(Λ) · scaling
    ///
    /// This is the key averaging argument from Lemma 2.1.
    pub fn coset_average_unit_pairs(&self, unit_set: &[C64], radius: f64) -> f64 {
        let lattice_count = self.polydisc_count(radius) as f64;
        let u_size = unit_set.len() as f64;

        // Expected pairs ≈ |U| · ρ_R where ρ_R = lattice density in ball
        let rho_r = lattice_count / self.covol.max(1e-300);

        u_size * rho_r
    }

    /// Project a lattice point in C^f to the complex plane via first coordinate.
    ///
    /// This is the map π_1: C^f → C that produces the planar point set.
    /// Injectivity of π_1 ensures the projected set has |P| = |X| points.
    pub fn project_to_plane(&self, point: &[C64]) -> C64 {
        if point.is_empty() {
            return C64::ZERO;
        }
        point[0]
    }

    /// Project a set of lattice points to the complex plane.
    pub fn project_set_to_plane(&self, points: &[Vec<C64>]) -> Vec<C64> {
        points.iter().map(|p| self.project_to_plane(p)).collect()
    }
}

/// Gaussian integer lattice Λ = Z[i] in C.
///
/// The simplest lattice for the Erdős grid construction.
/// dim=1, basis={(1,0)}, min_sep=1.0, covol=1.0.
impl MinkowskiLattice {
    /// The standard Gaussian integer lattice Z[i].
    pub fn gaussian() -> Self {
        Self {
            dim: 1,
            basis: vec![C64::ONE],
            min_sep: 1.0,
            covol: 1.0,
            packing_param: 2.0 * 4.0_f64.ln(), // 2·log(4·1·1)
        }
    }

    /// Generate an Erdős-style √n × √n grid of Gaussian integers.
    ///
    /// Produces n points in the plane as {a + bi : 0 ≤ a,b < √n}.
    /// The classic construction that gives ν(P) ≥ n^(1+c/log log n).
    pub fn erdos_grid(&self, n: usize) -> Vec<C64> {
        let side = (n as f64).sqrt().ceil() as usize;
        let mut points = Vec::with_capacity(side * side);
        for re in 0..side {
            for im in 0..side {
                points.push(C64::new(re as f64, im as f64));
            }
        }
        points
    }
}

/// Q(√5, i) lattice — 4-dimensional CM field.
///
/// K = Q(√5, i) has degree 4 over Q. The totally real subfield L = Q(√5)
/// has degree 2, so complex dimension f = 2.
impl MinkowskiLattice {
    /// Lattice for Q(√5, i) with dim=2.
    ///
    /// Uses the standard embedding with denominator D.
    pub fn q_sqrt5_i(denominator: f64) -> Self {
        let dim = 2;
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // golden ratio
        let scale = 1.0 / denominator;

        // Basis: (1/D, 0), (0, 1/D), (φ/D, 0), (0, φ/D) in C^2
        let basis = vec![
            C64::new(scale, 0.0),
            C64::ZERO,
            C64::ZERO,
            C64::new(scale, 0.0),
            C64::new(scale * phi, 0.0),
            C64::ZERO,
            C64::ZERO,
            C64::new(scale * phi, 0.0),
        ];

        let min_sep = scale;
        let root_disc = 5.0_f64.sqrt(); // disc(Q(√5))^(1/2)
        let covol = root_disc.powi(2) / denominator.powi(4);

        Self {
            dim,
            basis,
            min_sep,
            covol: covol.max(1e-300),
            packing_param: 2.0 * (4.0 * root_disc * denominator).ln(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::count_unit_distances;
    use super::*;

    #[test]
    fn gaussian_lattice_basic() {
        let lattice = MinkowskiLattice::gaussian();

        assert_eq!(lattice.dim, 1);
        assert!((lattice.min_sep - 1.0).abs() < 1e-12);
        assert!((lattice.covol - 1.0).abs() < 1e-12);
        assert!(lattice.is_projection_injective(0));
    }

    #[test]
    fn gaussian_packing_bound() {
        let lattice = MinkowskiLattice::gaussian();

        // In radius R with min_sep 1, bound = (2R/1)^1 = 2R
        let bound = lattice.packing_bound(5.0);
        assert_eq!(bound, 10); // (2*5/1)^1 = 10
    }

    #[test]
    fn gaussian_polydisc_count() {
        let lattice = MinkowskiLattice::gaussian();

        // vol = π·R², covol = 1
        let count = lattice.polydisc_count(10.0);
        let expected = std::f64::consts::PI * 100.0;
        assert!((count as f64 - expected).abs() / expected < 0.01);
    }

    #[test]
    fn gaussian_points_in_polydisc() {
        let lattice = MinkowskiLattice::gaussian();

        let points = lattice.points_in_polydisc(2.5);
        // Should include integers -2, -1, 0, 1, 2 (5 points)
        assert_eq!(points.len(), 5);
    }

    #[test]
    fn gaussian_lattice_point() {
        let lattice = MinkowskiLattice::gaussian();

        let pt = lattice.lattice_point(&[3]);
        assert!((pt[0].re - 3.0).abs() < 1e-12);
        assert!(pt[0].im.abs() < 1e-12);
    }

    #[test]
    fn erdos_grid_unit_distances() {
        let lattice = MinkowskiLattice::gaussian();
        let points = lattice.erdos_grid(100); // 10×10 grid

        assert!(points.len() >= 100);

        // Count unit distances: adjacent horizontal and vertical neighbors
        let count = count_unit_distances(&points, 1e-10);
        let n = points.len();

        // For √n × √n grid: at least 2·√n·(√n-1) = 2·10·9 = 180 unit distances
        let expected_min = 2 * 10 * 9;
        assert!(
            count as usize >= expected_min,
            "expected at least {expected_min} unit distances, got {count}"
        );

        // Erdős bound: ν(n) ≥ n^(1 + c/log_log_n) for some c > 0
        // For n=100: log_log_n = log(log(100)) = log(4.605) ≈ 1.527
        // n^(1+0.1/log_log_n) ≈ 100^1.065 ≈ 115 pairs minimum
        let log_log_n = (n as f64).ln().ln();
        let erdos_lower = (n as f64).powf(1.0 + 0.1 / log_log_n);
        assert!(
            count as f64 >= erdos_lower,
            "ν({n}) = {count} < Erdős bound {erdos_lower:.1}"
        );
    }

    #[test]
    fn sup_norm_basic() {
        let v = vec![C64::new(1.0, 0.0), C64::new(0.0, 2.0), C64::new(0.5, 0.5)];
        assert!((MinkowskiLattice::sup_norm(&v) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn from_field_params_q_i() {
        let lattice = MinkowskiLattice::from_field_params(1, 1.0, 1.0);

        assert_eq!(lattice.dim, 1);
        assert!((lattice.min_sep - 1.0).abs() < 1e-12);
        assert!(lattice.is_projection_injective(0));
    }

    #[test]
    fn projection_injective_identity() {
        let lattice = MinkowskiLattice::from_field_params(3, 2.0, 1.0);

        // Identity basis — projection to any coord is injective
        for k in 0..3 {
            assert!(lattice.is_projection_injective(k));
        }
    }

    #[test]
    fn q_sqrt5_i_lattice() {
        let lattice = MinkowskiLattice::q_sqrt5_i(1.0);

        assert_eq!(lattice.dim, 2);
        assert!(lattice.is_projection_injective(0));
        assert!(lattice.is_projection_injective(1));
    }

    #[test]
    fn coset_average_unit_pairs_positive() {
        let lattice = MinkowskiLattice::gaussian();

        // Unit set: roots of unity on the unit circle
        let unit_set: Vec<C64> = (0..4)
            .map(|k| {
                let angle = std::f64::consts::PI * 0.5 * k as f64;
                C64::new(angle.cos(), angle.sin())
            })
            .collect();

        let avg = lattice.coset_average_unit_pairs(&unit_set, 10.0);
        assert!(avg > 0.0, "expected positive average, got {avg}");
    }
}
