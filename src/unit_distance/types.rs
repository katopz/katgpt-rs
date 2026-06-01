//! Shared types for unit_distance module.
//!
//! Minimal complex arithmetic (C64) and algebraic number theory types
//! for lattice-based GOAT proofs of combinatorial geometry bounds.

use std::ops::{Add, Mul, Neg, Sub};

/// Complex number as a pair of f64 values (re, im).
///
/// Lightweight alternative to `num-complex::Complex64` — avoids adding
/// a dependency for research-only feature-gated code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    /// Zero.
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    /// One.
    pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };
    /// Imaginary unit i.
    pub const I: C64 = C64 { re: 0.0, im: 1.0 };

    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// Complex modulus |z| = sqrt(re² + im²).
    #[inline]
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Squared modulus |z|² = re² + im².
    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Complex conjugate z̄ = re - i·im.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Argument (phase angle) in radians.
    #[inline]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Multiplicative inverse 1/z.
    #[inline]
    pub fn inv(self) -> Self {
        let d = self.norm_sq();
        Self {
            re: self.re / d,
            im: -self.im / d,
        }
    }
}

impl Add for C64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for C64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for C64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Mul<f64> for C64 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl Neg for C64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl std::fmt::Display for C64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im.abs() < 1e-15 {
            write!(f, "{:.6}", self.re)
        } else if self.re.abs() < 1e-15 {
            write!(f, "{:.6}i", self.im)
        } else {
            let sign = if self.im >= 0.0 { '+' } else { '-' };
            write!(f, "{:.6}{}{:.6}i", self.re, sign, self.im.abs())
        }
    }
}

/// Parameters for a CM field K = L(i) where L is totally real.
///
/// A CM (complex multiplication) field is a totally imaginary quadratic
/// extension of a totally real number field. These are the fields used
/// in the unit distance construction.
#[derive(Clone, Debug)]
pub struct CmFieldParams {
    /// Root discriminant rd(K).
    pub root_discriminant: f64,

    /// Rational primes that split completely in K (must be ≡ 1 mod 4).
    pub split_primes: Vec<u64>,

    /// Degree [L:Q] of the totally real subfield.
    /// Total degree of K is 2·f.
    pub degree: usize,

    /// Ideal class number h(K).
    pub class_number: u64,

    /// Denominator D for lattice embedding D^(-1)·O_K.
    pub denominator: u64,
}

impl CmFieldParams {
    /// Total field degree [K:Q] = 2·f.
    #[inline]
    pub fn total_degree(&self) -> usize {
        2 * self.degree
    }

    /// Complex dimension for Minkowski embedding = f.
    #[inline]
    pub fn complex_dim(&self) -> usize {
        self.degree
    }
}

/// A conjugate pair of prime ideals {P, cP} in a CM field.
///
/// In a CM field, complex conjugation c exchanges prime ideals.
/// A pair (P, cP) with P ≠ cP gives two distinct ideals whose
/// product is the rational prime p·O_K.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimePair {
    /// The underlying rational prime p.
    pub prime: u64,

    /// Exponent k for ideal construction: P^ε · cP^(k-ε).
    pub exponent: u64,
}

impl PrimePair {
    pub const fn new(prime: u64, exponent: u64) -> Self {
        Self { prime, exponent }
    }
}

/// Result of counting norm-one elements via class group pigeonhole.
///
/// From Lemma 2.2 of the Remarks paper: given conjugate prime pairs
/// {(P_s, cP_s)} with exponents {k_s}, the number of elements u with
/// |σ(u)| = 1 for all embeddings σ is at least Π(k_s + 1) / h(K).
#[derive(Clone, Debug)]
pub struct PigeonholeResult {
    /// Root discriminant rd(K).
    pub root_discriminant: f64,

    /// Number of conjugate prime pairs used.
    pub num_prime_pairs: usize,

    /// Lower bound on |U| = number of norm-one elements.
    pub unit_set_lower_bound: u64,

    /// Product Π(k_s + 1) — total ideal configurations.
    pub total_configs: u64,

    /// Class number h(K) used as denominator.
    pub class_number: u64,

    /// Denominator D for D^(-1)·O_K lattice embedding.
    pub denominator: u64,
}

/// Point set in C (the complex plane) with unit-distance statistics.
///
/// Represents a projected point set from the Minkowski lattice,
/// together with pre-computed unit distance count.
#[derive(Clone, Debug)]
pub struct PointSet {
    /// Points in C (projected from C^f lattice via first coordinate).
    pub points: Vec<C64>,

    /// Number of unit-distance pairs (|z_i - z_j| = 1).
    pub unit_distance_pairs: u64,
}

impl PointSet {
    /// Number of points.
    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Unit distance density ν(P) = unit_pairs / |P|.
    /// Higher density → more unit distances per point.
    #[inline]
    pub fn unit_distance_density(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        self.unit_distance_pairs as f64 / self.points.len() as f64
    }
}

/// Count unit-distance pairs in a point set.
///
/// A pair (i, j) with i < j is a unit distance if |z_i - z_j| ≈ 1
/// within tolerance `eps`.
pub fn count_unit_distances(points: &[C64], eps: f64) -> u64 {
    let mut count: u64 = 0;
    let sq_eps = 2.0 * eps + eps * eps;
    for i in 0..points.len() {
        let pi = points[i];
        let pi_re = pi.re;
        let pi_im = pi.im;
        for j in (i + 1)..points.len() {
            let pj = points[j];
            let dr = pi_re - pj.re;
            let di = pi_im - pj.im;
            let d_sq = dr * dr + di * di;
            count += ((d_sq - 1.0).abs() < sq_eps) as u64;
        }
    }
    count
}

/// Exponent δ for the lower bound ν(n) ≥ n^(1+δ).
///
/// From the construction: δ = γ / (4·B) where:
/// - γ = t·log(2) - log(h) with t = number of split prime pairs, h = class number
/// - B = 2·log(4·R·D) with R = root discriminant, D = denominator
#[derive(Clone, Copy, Debug)]
pub struct DeltaEstimate {
    /// The computed δ value.
    pub delta: f64,

    /// γ = t·log(2) - log(h).
    pub gamma: f64,

    /// B = 2·log(4·R·D).
    pub b_param: f64,

    /// Root discriminant R.
    pub root_discriminant: f64,

    /// Number of split prime pairs t.
    pub t: usize,

    /// Class number h.
    pub h: u64,

    /// Denominator D.
    pub denominator: u64,
}

impl DeltaEstimate {
    /// Compute δ from field parameters.
    ///
    /// Returns `None` if γ ≤ 0 (construction doesn't yield δ > 0).
    pub fn from_field_params(params: &CmFieldParams) -> Option<Self> {
        let t = params.split_primes.len();
        let h = params.class_number;
        let r = params.root_discriminant;
        let d = params.denominator as f64;

        let gamma = t as f64 * (2.0_f64).ln() - (h as f64).ln();
        let b_param = 2.0 * (4.0 * r * d).ln();

        if gamma <= 0.0 {
            return None;
        }

        let delta = gamma / (4.0 * b_param);

        Some(Self {
            delta,
            gamma,
            b_param,
            t,
            h,
            root_discriminant: r,
            denominator: params.denominator,
        })
    }

    /// Whether δ is positive (construction valid).
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.delta > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c64_arithmetic() {
        let a = C64::new(1.0, 2.0);
        let b = C64::new(3.0, -1.0);

        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-12);
        assert!((sum.im - 1.0).abs() < 1e-12);

        let prod = a * b;
        // (1+2i)(3-i) = 3-i+6i-2i² = 3+5i+2 = 5+5i
        assert!((prod.re - 5.0).abs() < 1e-12);
        assert!((prod.im - 5.0).abs() < 1e-12);

        assert!((a.norm() - 5.0_f64.sqrt()).abs() < 1e-12);
        assert!((a.norm_sq() - 5.0).abs() < 1e-12);

        let conj = a.conj();
        assert!((conj.re - 1.0).abs() < 1e-12);
        assert!((conj.im + 2.0).abs() < 1e-12);
    }

    #[test]
    fn c64_unit_circle() {
        // e^(iπ/3) = 0.5 + i·√3/2
        let z = C64::new(0.5, 3.0_f64.sqrt() / 2.0);
        assert!((z.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn count_unit_distances_basic() {
        // Three points: unit triangle
        let points = vec![C64::ZERO, C64::ONE, C64::new(0.5, 3.0_f64.sqrt() / 2.0)];
        let count = count_unit_distances(&points, 1e-10);
        assert_eq!(count, 3); // all three pairs are unit distance
    }
}
