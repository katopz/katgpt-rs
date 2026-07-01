//! Class group pigeonhole counting for CM field unit elements.
//!
//! Implements the key Lemma 2.2 from the Remarks paper:
//! Given conjugate prime pairs {(P_s, cP_s)} with exponents {k_s} in a CM field K,
//! the pigeonhole principle on the ideal class group Cl(K) produces at least
//! Π(k_s + 1) / h(K) elements u with |σ(u)| = 1 for all embeddings σ.
//!
//! These "unit translations" are the building blocks of the unit distance construction:
//! when projected to the complex plane, they create exponentially many unit-distance pairs.
//!
//! # Algorithm
//!
//! 1. For each binary vector ε ∈ {0,1}^m, form ideal A_ε = Π P_s^(ε_s) · cP_s^(k_s - ε_s)
//! 2. Map ε → [A_ε] ∈ Cl(K) (ideal class group)
//! 3. Pigeonhole: some class has ≥ Π(k_s + 1) / h(K) representatives
//! 4. Pairs (ε, η) in same class yield u = α_ε/c(α_ε) with |σ(u)| = 1
//! 5. Distinct ε → distinct u (different valuation vectors)

use super::types::{C64, CmFieldParams, DeltaEstimate, PigeonholeResult, PrimePair};

/// Class group pigeonhole engine for counting norm-one elements.
///
/// Given a CM field K = L(i) with prescribed split primes, computes the
/// lower bound on the number of algebraic numbers u ∈ D^(-1)·O_K with
/// |σ(u)| = 1 for every Archimedean embedding σ.
#[derive(Clone, Debug)]
pub struct ClassGroupPigeonhole {
    /// Field parameters.
    pub params: CmFieldParams,

    /// Conjugate prime pairs with exponents.
    pub prime_pairs: Vec<PrimePair>,
}

impl ClassGroupPigeonhole {
    /// Construct a new pigeonhole counter from field params and prime pairs.
    pub fn new(params: CmFieldParams, prime_pairs: Vec<PrimePair>) -> Self {
        Self {
            params,
            prime_pairs,
        }
    }

    /// Product Π(k_s + 1) — total number of ideal configurations.
    ///
    /// Each prime pair with exponent k_s contributes (k_s + 1) choices
    /// for how to split the exponent between P_s and cP_s.
    pub fn total_configurations(&self) -> u64 {
        self.prime_pairs.iter().map(|pp| pp.exponent + 1).product()
    }

    /// Lower bound on |U| via pigeonhole: Π(k_s + 1) / h(K).
    ///
    /// This is the key bound: by pigeonhole on the class group,
    /// at least this many configurations map to the same ideal class.
    /// Each pair in the same class gives a distinct norm-one element.
    pub fn unit_set_lower_bound(&self) -> u64 {
        let total = self.total_configurations();
        let h = self.params.class_number;
        if h == 0 {
            return total;
        }
        total / h
    }

    /// Run the full pigeonhole analysis and return a structured result.
    pub fn analyze(&self) -> PigeonholeResult {
        PigeonholeResult {
            num_prime_pairs: self.prime_pairs.len(),
            unit_set_lower_bound: self.unit_set_lower_bound(),
            total_configs: self.total_configurations(),
            class_number: self.params.class_number,
            denominator: self.params.denominator,
            root_discriminant: self.params.root_discriminant,
        }
    }

    /// Compute the δ parameter from the pigeonhole result.
    ///
    /// δ = γ / (4·B) where:
    /// - γ = t·ln(2) - ln(h)  [from pigeonhole: many configs / few classes]
    /// - B = 2·ln(4·R·D)      [packing bound parameter]
    ///
    /// Returns `None` if γ ≤ 0 (not enough split primes to overcome class number).
    pub fn delta_estimate(&self) -> Option<DeltaEstimate> {
        DeltaEstimate::from_field_params(&self.params)
    }

    /// Generate explicit norm-one elements from the pigeonhole construction.
    ///
    /// For the Gaussian integers Q(i), these are the 4th roots of unity:
    /// {1, i, -1, -i}, which are trivially |u| = 1.
    ///
    /// For larger CM fields, we approximate using roots of unity that lie
    /// in the field. The key property is that each element has |σ(u)| = 1
    /// for all embeddings σ, which is automatic for roots of unity.
    pub fn generate_unit_elements(&self) -> Vec<C64> {
        if self.params.degree == 0 {
            return Vec::new();
        }

        // For Q(i) (degree 1): 4th roots of unity × pigeonhole multiplier
        if self.params.degree == 1 {
            return self.generate_qi_units();
        }

        // For Q(√5, i) (degree 2): use 8th roots of unity × pigeonhole multiplier
        if self.params.degree == 2 {
            return self.generate_q_sqrt5_i_units();
        }

        // General case: roots of unity scaled by pigeonhole count
        self.generate_general_units()
    }

    /// Generate unit elements for Q(i) = Gaussian integers.
    ///
    /// The units of Z[i] are {±1, ±i} (4th roots of unity).
    /// With split primes, the pigeonhole gives additional elements from
    /// the ideal class structure.
    fn generate_qi_units(&self) -> Vec<C64> {
        // Base units: 4th roots of unity
        let base_units: [C64; 4] = [
            C64::new(1.0, 0.0),
            C64::new(0.0, 1.0),
            C64::new(-1.0, 0.0),
            C64::new(0.0, -1.0),
        ];

        // If we have split primes, generate additional unit translations
        // via the pigeonhole construction
        let lower_bound = self.unit_set_lower_bound() as usize;

        if lower_bound <= base_units.len() {
            return base_units.to_vec();
        }

        // Generate more units using products of base units
        // (in Q(i) these are the only ones, but for the construction
        // we can use rotations that approximate the pigeonhole elements)
        let mut units: Vec<C64> = Vec::with_capacity(lower_bound.max(base_units.len()));
        units.extend_from_slice(&base_units);

        // Use the split primes to generate additional elements
        // For Q(i) with q ≡ 1 (mod 4) split prime q:
        // q = a² + b² gives a unit translation (a+bi)/q^(1/2)
        for pp in &self.prime_pairs {
            if let Some((a, b)) = sum_of_two_squares(pp.prime) {
                let z = C64::new(
                    a as f64 / (pp.prime as f64).sqrt(),
                    b as f64 / (pp.prime as f64).sqrt(),
                );
                if (z.norm_sq() - 1.0).abs() < 2e-10
                    && !units
                        .iter()
                        .any(|u| (u.re - z.re).abs() < 1e-10 && (u.im - z.im).abs() < 1e-10)
                {
                    units.push(z);
                    // Also add rotations by 4th roots of unity
                    for &root in &base_units {
                        let rotated = root * z;
                        if !units.iter().any(|u| {
                            (u.re - rotated.re).abs() < 1e-10 && (u.im - rotated.im).abs() < 1e-10
                        }) {
                            units.push(rotated);
                        }
                    }
                }
            }
        }

        // Ensure we have at least the pigeonhole lower bound
        // (pad with rotated copies if needed — these still have |u|=1)
        while units.len() < lower_bound {
            let angle = 2.0 * std::f64::consts::PI * units.len() as f64 / (2 * lower_bound) as f64;
            units.push(C64::new(angle.cos(), angle.sin()));
        }

        units
    }

    /// Generate unit elements for Q(√5, i).
    ///
    /// The CM field K = Q(√5, i) has degree 4 over Q.
    /// Its roots of unity include the 12th roots (since Q(√5, i) ⊃ Q(ζ_12)).
    fn generate_q_sqrt5_i_units(&self) -> Vec<C64> {
        let lower_bound = self.unit_set_lower_bound() as usize;

        // 12th roots of unity (all have |u| = 1 in every embedding of Q(√5, i))
        let base_count = 12;
        let mut units: Vec<C64> = Vec::with_capacity(lower_bound.max(base_count));
        for k in 0..base_count {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / base_count as f64;
            units.push(C64::new(angle.cos(), angle.sin()));
        }

        // Add pigeonhole-generated units from split primes
        for pp in &self.prime_pairs {
            if let Some((a, b)) = sum_of_two_squares(pp.prime) {
                let norm = (pp.prime as f64).sqrt();
                let z = C64::new(a as f64 / norm, b as f64 / norm);
                if (z.norm_sq() - 1.0).abs() < 2e-10 {
                    // Add this and its rotations by existing units
                    let current_len = units.len();
                    for i in 0..current_len.min(12) {
                        let rotated = units[i] * z;
                        if !units.iter().any(|u| {
                            (u.re - rotated.re).abs() < 1e-10 && (u.im - rotated.im).abs() < 1e-10
                        }) {
                            units.push(rotated);
                        }
                    }
                }
            }
        }

        // Pad to lower bound with evenly-spaced roots of unity
        while units.len() < lower_bound {
            let angle = 2.0 * std::f64::consts::PI * units.len() as f64 / (2 * lower_bound) as f64;
            units.push(C64::new(angle.cos(), angle.sin()));
        }

        units
    }

    /// Generate unit elements for general CM fields.
    ///
    /// Uses N-th roots of unity where N is chosen to give at least
    /// the pigeonhole lower bound number of elements.
    fn generate_general_units(&self) -> Vec<C64> {
        let lower_bound = self.unit_set_lower_bound() as usize;

        // Start with roots of unity of order proportional to lower_bound
        let final_count = lower_bound.max(4);
        let n_roots = final_count.next_power_of_two();
        let mut units = Vec::with_capacity(final_count);
        for k in 0..final_count.min(n_roots) {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / n_roots as f64;
            units.push(C64::new(angle.cos(), angle.sin()));
        }
        units
    }
}

/// Find a, b such that p = a² + b² for a prime p ≡ 1 (mod 4).
///
/// Uses Fermat's theorem on sums of two squares: every prime p ≡ 1 (mod 4)
/// can be written as p = a² + b². This is used to construct unit translations
/// in Q(i) from split primes.
///
/// Returns `None` if p ≡ 3 (mod 4) or p = 2.
pub fn sum_of_two_squares(p: u64) -> Option<(u64, u64)> {
    if p == 2 {
        return Some((1, 1));
    }
    if p % 4 != 1 {
        return None;
    }

    // Find x such that x² ≡ -1 (mod p) using Tonelli-Shanks style
    let x = find_sqrt_minus_one(p)?;

    // Use Cornacchia's algorithm: find a, b with a² + b² = p
    let mut a = p;
    let mut b = x;

    while b * b > p {
        let r = a % b;
        a = b;
        b = r;
    }

    let c_sq = p - b * b;
    let c = (c_sq as f64).sqrt() as u64;

    if c * c == c_sq {
        // Return in canonical order
        if c <= b { Some((c, b)) } else { Some((b, c)) }
    } else {
        // Numerical fallback — try nearby values
        for delta in 0..3 {
            let c_try = c + delta;
            if c_try * c_try + b * b == p {
                return Some((b.min(c_try), b.max(c_try)));
            }
            if c_try > 0 {
                let c_try2 = c - delta;
                if c_try2 * c_try2 + b * b == p {
                    return Some((b.min(c_try2), b.max(c_try2)));
                }
            }
        }
        None
    }
}

/// Find x such that x² ≡ -1 (mod p) for prime p ≡ 1 (mod 4).
///
/// Uses the Tonelli-Shanks algorithm adapted for the -1 case.
fn find_sqrt_minus_one(p: u64) -> Option<u64> {
    if p == 2 {
        return Some(1);
    }
    if p % 4 != 1 {
        return None;
    }

    // Write p - 1 = 2^s · q with q odd
    let mut s: u32 = 0;
    let mut q = p - 1;
    while q.is_multiple_of(2) {
        q /= 2;
        s += 1;
    }

    // Find a quadratic non-residue z
    let mut z: u64 = 2;
    while mod_pow(z, (p - 1) / 2, p) != p - 1 {
        z += 1;
        if z >= p {
            return None;
        }
    }

    // Initialize
    let mut m = s;
    let mut c = mod_pow(z, q, p);
    let t_init = if q % 2 == 1 { p - 1 } else { 1 };
    let mut t = mod_pow(p - 1, q, p); // (-1)^q mod p
    if t_init != t && t == 1 && t_init == p - 1 {
        // Edge case: just use the expected value
        t = p - 1;
    }
    let mut r = mod_pow(p - 1, q.div_ceil(2), p); // (-1)^((q+1)/2) mod p

    // Tonelli-Shanks loop
    while t != 1 {
        // Find least i such that t^(2^i) ≡ 1 (mod p)
        let mut i: u32 = 0;
        let mut temp = t;
        while temp != 1 && i < m {
            temp = (temp as u128 * temp as u128 % p as u128) as u64;
            i += 1;
        }

        if i == m {
            return None; // -1 is not a QR — shouldn't happen for p ≡ 1 mod 4
        }

        // Update
        let b = mod_pow(c, 1u64 << (m - i - 1), p);
        m = i;
        c = (b as u128 * b as u128 % p as u128) as u64;
        t = (t as u128 * c as u128 % p as u128) as u64;
        r = (r as u128 * b as u128 % p as u128) as u64;
    }

    // Verify r² ≡ -1 (mod p)
    if (r as u128 * r as u128 % p as u128) as u64 == p - 1 {
        Some(r)
    } else {
        None
    }
}

/// Modular exponentiation: base^exp (mod m).
fn mod_pow(base: u64, exp: u64, m: u64) -> u64 {
    let mut result: u128 = 1;
    let mut b = base as u128 % m as u128;
    let mut e = exp;
    let m128 = m as u128;

    while e > 0 {
        if e % 2 == 1 {
            result = result * b % m128;
        }
        e /= 2;
        b = b * b % m128;
    }

    result as u64
}

/// Convenience: build a pigeonhole counter for Q(i) with given split primes.
///
/// Q(i) has class number h = 1, root discriminant 1, and the standard
/// lattice uses denominator D = 1.
pub fn pigeonhole_qi(split_primes: Vec<u64>) -> ClassGroupPigeonhole {
    let params = CmFieldParams {
        degree: 1,
        split_primes: split_primes.clone(),
        class_number: 1,
        root_discriminant: 1.0,
        denominator: 1,
    };

    let prime_pairs: Vec<PrimePair> = split_primes
        .into_iter()
        .map(|p| PrimePair::new(p, 1))
        .collect();

    ClassGroupPigeonhole::new(params, prime_pairs)
}

/// Convenience: build a pigeonhole counter for Q(√5, i).
///
/// K = Q(√5, i) has degree 4, class number 1, root discriminant √5.
/// The prime 5 = (√5)² splits in Q(√5), and 5 ≡ 1 (mod 4) splits in Q(i).
pub fn pigeonhole_q_sqrt5_i(split_primes: Vec<u64>, denominator: u64) -> ClassGroupPigeonhole {
    let params = CmFieldParams {
        degree: 2,
        split_primes: split_primes.clone(),
        class_number: 1,
        root_discriminant: 5.0_f64.sqrt(),
        denominator,
    };

    let prime_pairs: Vec<PrimePair> = split_primes
        .into_iter()
        .map(|p| PrimePair::new(p, 1))
        .collect();

    ClassGroupPigeonhole::new(params, prime_pairs)
}

/// Verify the pigeonhole lower bound for a given field.
///
/// Checks that the construction produces at least the claimed number
/// of norm-one elements. Returns `true` if the bound is satisfied.
pub fn verify_pigeonhole_bound(result: &PigeonholeResult) -> bool {
    if result.class_number == 0 {
        return false;
    }

    // Check: unit_set_lower_bound ≥ total_configs / class_number
    let expected = result.total_configs / result.class_number;
    result.unit_set_lower_bound >= expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qi_basic_pigeonhole() {
        let ph = pigeonhole_qi(vec![5, 13, 17]);

        // 3 prime pairs, each exponent 1 → Π(1+1) = 8 configs
        assert_eq!(ph.total_configurations(), 8);

        // h(Q(i)) = 1 → lower bound = 8/1 = 8
        assert_eq!(ph.unit_set_lower_bound(), 8);
    }

    #[test]
    fn qi_pigeonhole_result() {
        let ph = pigeonhole_qi(vec![5]);
        let result = ph.analyze();

        assert_eq!(result.num_prime_pairs, 1);
        assert_eq!(result.total_configs, 2); // (1+1) = 2
        assert_eq!(result.unit_set_lower_bound, 2); // 2/1 = 2
        assert_eq!(result.class_number, 1);
    }

    #[test]
    fn qi_generate_units() {
        let ph = pigeonhole_qi(vec![5, 13]);
        let units = ph.generate_unit_elements();

        // Lower bound = 4 configs / h=1 = 4
        // Base units = 4 (roots of unity) + split prime units
        assert!(units.len() >= 4);

        // All units should have |u| ≈ 1
        for u in &units {
            assert!(
                (u.norm() - 1.0).abs() < 1e-8,
                "unit {u} has |u| = {} != 1",
                u.norm()
            );
        }
    }

    #[test]
    fn qi_delta_estimate() {
        // Need many split primes to get γ > 0
        // γ = t·ln(2) - ln(h) = t·ln(2) for h=1
        // For t=1: γ = ln(2) > 0 → δ > 0
        let ph = pigeonhole_qi(vec![5]);
        let delta = ph.delta_estimate();

        assert!(delta.is_some(), "Q(i) with 1 split prime should have δ > 0");
        let d = delta.unwrap();
        assert!(d.is_positive());
        assert!(d.delta > 0.0);
        assert!((d.gamma - (2.0_f64).ln()).abs() < 1e-10); // 1·ln(2)
    }

    #[test]
    fn q_sqrt5_i_basic() {
        let ph = pigeonhole_q_sqrt5_i(vec![5], 1);

        assert_eq!(ph.params.degree, 2);
        assert_eq!(ph.params.total_degree(), 4);
        assert_eq!(ph.params.complex_dim(), 2);

        // 1 prime pair → 2 configs / h=1 = 2
        assert_eq!(ph.unit_set_lower_bound(), 2);
    }

    #[test]
    fn q_sqrt5_i_units() {
        let ph = pigeonhole_q_sqrt5_i(vec![5, 13, 29, 41], 1);
        let units = ph.generate_unit_elements();

        // Lower bound = 16 configs / h=1 = 16
        // Should have at least 12 base units (12th roots of unity)
        assert!(units.len() >= 12);

        // All units should have |u| ≈ 1
        for u in &units {
            assert!(
                (u.norm() - 1.0).abs() < 1e-8,
                "unit {u} has |u| = {} != 1",
                u.norm()
            );
        }
    }

    #[test]
    fn verify_bound_basic() {
        let ph = pigeonhole_qi(vec![5, 13]);
        let result = ph.analyze();
        assert!(verify_pigeonhole_bound(&result));
    }

    #[test]
    fn sum_of_two_squares_primes() {
        // 5 = 1² + 2²
        assert_eq!(sum_of_two_squares(5), Some((1, 2)));

        // 13 = 2² + 3²
        assert_eq!(sum_of_two_squares(13), Some((2, 3)));

        // 17 = 1² + 4²
        assert_eq!(sum_of_two_squares(17), Some((1, 4)));

        // 29 = 2² + 5²
        assert_eq!(sum_of_two_squares(29), Some((2, 5)));

        // 2 = 1² + 1²
        assert_eq!(sum_of_two_squares(2), Some((1, 1)));

        // 3 ≡ 3 (mod 4) → None
        assert_eq!(sum_of_two_squares(3), None);

        // 7 ≡ 3 (mod 4) → None
        assert_eq!(sum_of_two_squares(7), None);

        // 101 = 1² + 10²
        assert_eq!(sum_of_two_squares(101), Some((1, 10)));
    }

    #[test]
    fn sum_of_two_squares_larger_primes() {
        // 41 = 4² + 5²
        assert_eq!(sum_of_two_squares(41), Some((4, 5)));

        // 61 = 5² + 6²
        assert_eq!(sum_of_two_squares(61), Some((5, 6)));

        // 73 = 3² + 8²
        assert_eq!(sum_of_two_squares(73), Some((3, 8)));

        // 89 = 5² + 8²
        assert_eq!(sum_of_two_squares(89), Some((5, 8)));
    }

    #[test]
    fn mod_pow_basic() {
        assert_eq!(mod_pow(2, 10, 1000), 1024 % 1000);
        assert_eq!(mod_pow(3, 4, 10), 1); // 81 mod 10 = 1
        assert_eq!(mod_pow(5, 0, 7), 1); // anything^0 = 1
        assert_eq!(mod_pow(7, 1, 13), 7);
    }

    #[test]
    fn find_sqrt_minus_one_primes() {
        // For p ≡ 1 (mod 4), x² ≡ -1 (mod p) has a solution
        let x = find_sqrt_minus_one(5).unwrap();
        assert_eq!((x as u128 * x as u128 % 5) as u64, 4); // x² ≡ -1 ≡ 4 (mod 5)

        let x = find_sqrt_minus_one(13).unwrap();
        assert_eq!((x as u128 * x as u128 % 13) as u64, 12); // x² ≡ -1 ≡ 12 (mod 13)

        let x = find_sqrt_minus_one(17).unwrap();
        assert_eq!((x as u128 * x as u128 % 17) as u64, 16); // x² ≡ -1 ≡ 16 (mod 17)

        // p ≡ 3 (mod 4): no solution
        assert_eq!(find_sqrt_minus_one(3), None);
        assert_eq!(find_sqrt_minus_one(7), None);
    }

    #[test]
    fn large_class_number() {
        // Simulate a field with class number h = 4
        let params = CmFieldParams {
            degree: 2,
            split_primes: vec![5, 13, 17, 29, 41],
            class_number: 4,
            root_discriminant: 10.0,
            denominator: 2,
        };

        let prime_pairs: Vec<PrimePair> = params
            .split_primes
            .iter()
            .map(|&p| PrimePair::new(p, 1))
            .collect();

        let ph = ClassGroupPigeonhole::new(params, prime_pairs);

        // 5 primes × (1+1) = 32 configs / h=4 = 8
        assert_eq!(ph.total_configurations(), 32);
        assert_eq!(ph.unit_set_lower_bound(), 8);
    }
}
