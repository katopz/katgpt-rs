//! CM field construction for explicit unit distance examples.
//!
//! Implements T4: construction of small CM (complex multiplication) fields
//! with prescribed splitting conditions. A CM field is K = L(i) where L is
//! totally real — the key algebraic structure in the unit distance proof.
//!
//! This is **light model-based**: field construction is a one-shot computation
//! (no gradient updates), parameters are chosen by Chebotarev density
//! (deterministic for small fields), and it's only needed for verification.
//!
//! # Fields Implemented
//!
//! | Field | Degree | Class # | Root Disc | Split Primes |
//! |-------|--------|---------|-----------|--------------|
//! | Q(i)  | 2      | 1       | 1         | p ≡ 1 mod 4  |
//! | Q(√5, i) | 4   | 1       | √5        | 5, 13, 17... |
//! | Pro-2 base | 4+   | varies | bounded  | 101 ( Remarks) |
//!
//! Reference: Section 3 of the proof paper, simplified in Remarks Lemma 2.2.

use super::minkowski::MinkowskiLattice;
use super::pigeonhole::ClassGroupPigeonhole;
use super::types::{C64, CmFieldParams, DeltaEstimate, PigeonholeResult, PrimePair};

/// A constructed CM field K = L(i) with precomputed arithmetic data.
///
/// Unlike `CmFieldParams` (which just stores parameters), `CmField` holds
/// the actual arithmetic infrastructure: lattice, pigeonhole counter,
/// and unit elements needed for the GOAT proof.
#[derive(Clone, Debug)]
pub struct CmField {
    /// Field parameters (degree, class number, discriminant, etc.)
    pub params: CmFieldParams,

    /// Minkowski lattice embedding of D^(-1)·O_K in C^f.
    pub lattice: MinkowskiLattice,

    /// Pigeonhole counter for norm-one elements.
    pub pigeonhole: ClassGroupPigeonhole,

    /// Unit elements u with |σ(u)| = 1 for all embeddings σ.
    pub unit_elements: Vec<C64>,

    /// Human-readable name for logging.
    pub name: String,
}

impl CmField {
    /// Construct Q(i) — the Gaussian integer field.
    ///
    /// The simplest CM field: degree 2, class number 1, root discriminant 1.
    /// Split primes are exactly the primes p ≡ 1 (mod 4).
    ///
    /// # Arguments
    /// * `split_primes` — Primes p ≡ 1 (mod 4) that split in Q(i).
    ///   Must be empty or contain valid split primes.
    pub fn qi(split_primes: Vec<u64>) -> Self {
        let params = CmFieldParams {
            degree: 1,
            split_primes: split_primes.clone(),
            class_number: 1,
            root_discriminant: 1.0,
            denominator: 1,
        };

        let lattice = MinkowskiLattice::gaussian();
        let prime_pairs: Vec<PrimePair> =
            split_primes.iter().map(|&p| PrimePair::new(p, 1)).collect();

        let pigeonhole = ClassGroupPigeonhole::new(params.clone(), prime_pairs);
        let unit_elements = pigeonhole.generate_unit_elements();

        Self {
            params,
            lattice,
            pigeonhole,
            unit_elements,
            name: "Q(i)".to_string(),
        }
    }

    /// Construct Q(i) with default split primes {5, 13, 17, 29}.
    ///
    /// A convenient default for testing the Erdős grid construction.
    pub fn qi_default() -> Self {
        Self::qi(vec![5, 13, 17, 29])
    }

    /// Construct Q(√5, i) — the simplest non-trivial CM field.
    ///
    /// K = Q(√5, i) has degree 4, class number 1.
    /// The totally real subfield L = Q(√5) has degree 2.
    /// Root discriminant rd(K) = √5.
    ///
    /// # Arguments
    /// * `split_primes` — Primes that split completely in K.
    /// * `denominator` — Lattice embedding parameter D.
    pub fn q_sqrt5_i(split_primes: Vec<u64>, denominator: u64) -> Self {
        let params = CmFieldParams {
            degree: 2,
            split_primes: split_primes.clone(),
            class_number: 1,
            root_discriminant: 5.0_f64.sqrt(),
            denominator,
        };

        let lattice = MinkowskiLattice::q_sqrt5_i(denominator as f64);
        let prime_pairs: Vec<PrimePair> =
            split_primes.iter().map(|&p| PrimePair::new(p, 1)).collect();

        let pigeonhole = ClassGroupPigeonhole::new(params.clone(), prime_pairs);
        let unit_elements = pigeonhole.generate_unit_elements();

        Self {
            params,
            lattice,
            pigeonhole,
            unit_elements,
            name: "Q(√5, i)".to_string(),
        }
    }

    /// Construct Q(√5, i) with default parameters.
    ///
    /// Uses split primes {5, 13, 29, 41} and denominator D = 1.
    pub fn q_sqrt5_i_default() -> Self {
        Self::q_sqrt5_i(vec![5, 13, 29, 41], 1)
    }

    /// Construct the pro-2 tower base field from the Remarks paper.
    ///
    /// Uses T = {3, 5, 7, 11, 13, 17} with S = {101, ∞}.
    /// The single split prime q = 101 suffices for the simplified construction.
    ///
    /// Parameters are taken from the Remarks paper analysis:
    /// - Base degree: 6 (cyclotomic subfield)
    /// - Pro-2 tower: degree doubles at each step
    /// - δ ≈ 6.24 × 10^(-38) (tiny but positive)
    pub fn pro2_tower_base() -> Self {
        let params = CmFieldParams {
            degree: 6,
            split_primes: vec![101],
            // Estimated class number for the base field
            class_number: 1,
            // Root discriminant bounded by the construction
            root_discriminant: 20.0,
            denominator: 1,
        };

        let lattice = MinkowskiLattice::from_field_params(6, 20.0, 1.0);
        let prime_pairs = vec![PrimePair::new(101, 1)];

        let pigeonhole = ClassGroupPigeonhole::new(params.clone(), prime_pairs);
        let unit_elements = pigeonhole.generate_unit_elements();

        Self {
            params,
            lattice,
            pigeonhole,
            unit_elements,
            name: "Pro-2 Tower Base".to_string(),
        }
    }

    /// Construct a generic CM field from explicit parameters.
    ///
    /// For research and experimentation with different field configurations.
    pub fn from_params(name: &str, params: CmFieldParams, prime_pair_exponents: Vec<u64>) -> Self {
        let lattice = MinkowskiLattice::from_field_params(
            params.degree,
            params.root_discriminant,
            params.denominator as f64,
        );

        let prime_pairs: Vec<PrimePair> = params
            .split_primes
            .iter()
            .zip(prime_pair_exponents.iter().chain(std::iter::repeat(&1u64)))
            .map(|(&p, &e)| PrimePair::new(p, e))
            .collect();

        let pigeonhole = ClassGroupPigeonhole::new(params.clone(), prime_pairs);
        let unit_elements = pigeonhole.generate_unit_elements();

        Self {
            params,
            lattice,
            pigeonhole,
            unit_elements,
            name: name.to_string(),
        }
    }

    /// Total field degree [K:Q] = 2·f.
    pub fn total_degree(&self) -> usize {
        self.params.total_degree()
    }

    /// Complex dimension for Minkowski embedding = f.
    pub fn complex_dim(&self) -> usize {
        self.params.complex_dim()
    }

    /// Number of norm-one unit elements.
    pub fn num_units(&self) -> usize {
        self.unit_elements.len()
    }

    /// Lower bound on |U| from pigeonhole.
    pub fn unit_lower_bound(&self) -> u64 {
        self.pigeonhole.unit_set_lower_bound()
    }

    /// Full pigeonhole analysis result.
    pub fn pigeonhole_result(&self) -> PigeonholeResult {
        self.pigeonhole.analyze()
    }

    /// Compute the δ parameter for ν(n) ≥ n^(1+δ).
    ///
    /// Returns `None` if γ ≤ 0 (construction doesn't work for these parameters).
    pub fn delta(&self) -> Option<DeltaEstimate> {
        self.pigeonhole.delta_estimate()
    }

    /// Whether the construction yields δ > 0 (valid counterexample).
    pub fn is_valid_construction(&self) -> bool {
        self.delta().is_some_and(|d| d.is_positive())
    }

    /// Verify that all split primes are actually ≡ 1 (mod 4).
    ///
    /// This is a necessary condition for the prime to split in Q(i),
    /// which is required for the unit distance construction.
    pub fn verify_split_primes(&self) -> bool {
        self.params.split_primes.iter().all(|&p| p % 4 == 1)
    }

    /// Verify that all unit elements have |u| ≈ 1.
    ///
    /// Checks that every generated unit element has modulus within
    /// `eps` of 1.0. This is the defining property of the pigeonhole
    /// construction: |σ(u)| = 1 for all embeddings σ.
    pub fn verify_units_on_circle(&self, eps: f64) -> bool {
        let sq_eps = 2.0 * eps + eps * eps;
        self.unit_elements
            .iter()
            .all(|u| (u.norm_sq() - 1.0).abs() < sq_eps)
    }

    /// Verify projection injectivity.
    ///
    /// The first-coordinate projection π₁: C^f → C must be injective
    /// on the lattice for the planar point set to have the right size.
    pub fn verify_projection_injective(&self) -> bool {
        self.lattice.is_projection_injective(0)
    }

    /// Generate a planar point set from the lattice and unit elements.
    ///
    /// Projects lattice points in the polydisc of radius `R` to C,
    /// then counts unit distances created by the unit translations.
    pub fn generate_point_set(&self, radius: f64) -> super::types::PointSet {
        let lattice_points = self.lattice.points_in_polydisc(radius);
        let planar_points = self.lattice.project_set_to_plane(&lattice_points);
        let unit_pairs = super::types::count_unit_distances(&planar_points, 1e-8);

        super::types::PointSet {
            points: planar_points,
            unit_distance_pairs: unit_pairs,
        }
    }

    /// Run full verification suite for the GOAT proof.
    ///
    /// Returns a summary of all checks and their results.
    pub fn verify_all(&self) -> FieldVerification {
        let eps = 1e-8;
        let split_ok = self.verify_split_primes();
        let units_ok = self.verify_units_on_circle(eps);
        let injective_ok = self.verify_projection_injective();
        let delta = self.delta();
        let delta_positive = delta.is_some();
        let pigeonhole_ok = super::pigeonhole::verify_pigeonhole_bound(&self.pigeonhole_result());

        FieldVerification {
            field_name: self.name.clone(),
            split_primes_valid: split_ok,
            units_on_circle: units_ok,
            projection_injective: injective_ok,
            pigeonhole_bound: pigeonhole_ok,
            delta_positive,
            delta_value: delta.map(|d| d.delta),
            unit_count: self.num_units(),
            unit_lower_bound: self.unit_lower_bound(),
            all_passed: split_ok && units_ok && injective_ok && pigeonhole_ok && delta_positive,
        }
    }
}

/// Result of verifying a CM field construction.
///
/// All checks must pass for the GOAT proof to be valid.
#[derive(Clone, Debug)]
pub struct FieldVerification {
    /// Name of the field.
    pub field_name: String,

    /// The computed δ value (if δ > 0).
    pub delta_value: Option<f64>,

    /// Number of generated unit elements.
    pub unit_count: usize,

    /// Pigeonhole lower bound on unit elements.
    pub unit_lower_bound: u64,

    /// All split primes are ≡ 1 (mod 4).
    pub split_primes_valid: bool,

    /// All generated unit elements have |u| ≈ 1.
    pub units_on_circle: bool,

    /// Projection to first coordinate is injective.
    pub projection_injective: bool,

    /// Pigeonhole bound is satisfied.
    pub pigeonhole_bound: bool,

    /// δ > 0 (construction yields counterexample).
    pub delta_positive: bool,

    /// Whether all checks passed.
    pub all_passed: bool,
}

impl std::fmt::Display for FieldVerification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let check = |ok: bool| if ok { "✅" } else { "❌" };

        writeln!(f, "Field Verification: {}", self.field_name)?;
        writeln!(
            f,
            "  Split primes valid:   {}",
            check(self.split_primes_valid)
        )?;
        writeln!(f, "  Units on circle:      {}", check(self.units_on_circle))?;
        writeln!(
            f,
            "  Projection injective: {}",
            check(self.projection_injective)
        )?;
        writeln!(
            f,
            "  Pigeonhole bound:     {}",
            check(self.pigeonhole_bound)
        )?;
        writeln!(f, "  δ > 0:                {}", check(self.delta_positive))?;
        if let Some(d) = self.delta_value {
            writeln!(f, "  δ value:              {d:.6e}")?;
        }
        writeln!(f, "  Unit count:           {}", self.unit_count)?;
        writeln!(f, "  Unit lower bound:     {}", self.unit_lower_bound)?;
        writeln!(f, "  Overall:              {}", check(self.all_passed))
    }
}

/// Enumerate primes p ≡ 1 (mod 4) up to a given limit.
///
/// Simple sieve-based enumeration for split prime selection.
/// These are exactly the primes that split in Q(i).
pub fn enumerate_split_primes(limit: u64) -> Vec<u64> {
    if limit < 5 {
        return Vec::new();
    }

    let mut is_prime = vec![true; (limit + 1) as usize];
    is_prime[0] = false;
    if limit >= 1 {
        is_prime[1] = false;
    }

    let sqrt_limit = (limit as f64).sqrt() as u64 + 1;
    for i in 2..=sqrt_limit {
        if is_prime[i as usize] {
            let mut j = i * i;
            while j <= limit {
                is_prime[j as usize] = false;
                j += i;
            }
        }
    }

    (2..=limit)
        .filter(|&p| is_prime[p as usize] && p % 4 == 1)
        .collect()
}

/// Select the first `n` split primes (primes ≡ 1 mod 4).
///
/// Returns primes: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, ...
pub fn select_split_primes(n: usize) -> Vec<u64> {
    // Generate enough primes — density of p ≡ 1 (mod 4) is ~ 1/(2·ln(p))
    // so we need ~ 2·n·ln(n) upper bound
    let limit = if n == 0 {
        return Vec::new();
    } else if n <= 5 {
        50
    } else {
        (3 * n as u64 * (n as f64).ln() as u64).max(100)
    };

    let primes = enumerate_split_primes(limit);
    primes.into_iter().take(n).collect()
}

/// Compute the class number bound for a number field.
///
/// For a CM field K = L(i) with root discriminant rd(K):
/// h(K) ≤ (2·rd(K))^(2·C_class) where C_class is a constant depending
/// on the degree. For small fields this is very conservative.
///
/// Returns a bound suitable for the pigeonhole counting.
pub fn class_number_bound(root_disc: f64, degree: usize) -> u64 {
    // Minkowski bound: h ≤ Minkowski constant × some factor
    // For small fields: h ≈ (π/4)^f × disc^(1/2) / f!
    // Use a generous upper bound
    let minkowski_const = (std::f64::consts::PI / 4.0).powi(degree as i32);
    let disc_term = root_disc.powi(degree as i32);
    let factorial = (1..=degree).fold(1.0_f64, |acc, k| acc * k as f64);

    let bound = minkowski_const * disc_term / factorial;
    bound.ceil().max(1.0) as u64
}

/// Build a CM field optimized for the unit distance construction.
///
/// Selects split primes to maximize γ = t·ln(2) - ln(h), which gives
/// the largest δ. For h = 1 (class number 1 fields), γ = t·ln(2) and
/// more split primes → larger γ → larger δ.
///
/// This is the "light model-based" component: a one-shot optimization
/// over the discrete parameter t (number of split primes).
pub fn optimize_qi(num_primes: usize) -> CmField {
    let split_primes = select_split_primes(num_primes);
    CmField::qi(split_primes)
}

/// Build an optimized Q(√5, i) field.
pub fn optimize_q_sqrt5_i(num_primes: usize, denominator: u64) -> CmField {
    let split_primes = select_split_primes(num_primes);
    CmField::q_sqrt5_i(split_primes, denominator)
}

/// Compare δ values across different field configurations.
///
/// Returns fields sorted by δ (best first) for benchmarking.
pub fn compare_delta(fields: &[CmField]) -> Vec<(&str, Option<f64>)> {
    let mut results: Vec<(&str, Option<f64>)> = fields
        .iter()
        .map(|f| (f.name.as_str(), f.delta().map(|d| d.delta)))
        .collect();

    results.sort_by(|a, b| {
        let da = a.1.unwrap_or(0.0);
        let db = b.1.unwrap_or(0.0);
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qi_construction() {
        let field = CmField::qi(vec![5, 13]);

        assert_eq!(field.total_degree(), 2);
        assert_eq!(field.complex_dim(), 1);
        assert_eq!(field.params.class_number, 1);
        assert!(field.verify_split_primes());
        assert!(field.verify_projection_injective());
    }

    #[test]
    fn qi_units_on_circle() {
        let field = CmField::qi_default();
        assert!(
            field.verify_units_on_circle(1e-8),
            "All units should be on the unit circle"
        );
    }

    #[test]
    fn qi_delta_positive() {
        let field = CmField::qi(vec![5]);
        let delta = field.delta();
        assert!(delta.is_some(), "Q(i) with 1 split prime should have δ > 0");
        assert!(delta.unwrap().delta > 0.0);
    }

    #[test]
    fn qi_verify_all() {
        let field = CmField::qi_default();
        let verification = field.verify_all();

        assert!(verification.split_primes_valid);
        assert!(verification.units_on_circle);
        assert!(verification.projection_injective);
        assert!(verification.pigeonhole_bound);
        assert!(verification.delta_positive);
        assert!(verification.all_passed);
    }

    #[test]
    fn qi_verify_display() {
        let field = CmField::qi_default();
        let verification = field.verify_all();
        let display = format!("{verification}");
        assert!(display.contains("Q(i)"));
        assert!(display.contains("✅"));
    }

    #[test]
    fn q_sqrt5_i_construction() {
        let field = CmField::q_sqrt5_i_default();

        assert_eq!(field.total_degree(), 4);
        assert_eq!(field.complex_dim(), 2);
        assert_eq!(field.params.class_number, 1);
        assert!(field.verify_split_primes());
        assert!(field.verify_projection_injective());
    }

    #[test]
    fn q_sqrt5_i_units_on_circle() {
        let field = CmField::q_sqrt5_i_default();
        assert!(
            field.verify_units_on_circle(1e-8),
            "All units should be on the unit circle"
        );
    }

    #[test]
    fn q_sqrt5_i_delta_positive() {
        let field = CmField::q_sqrt5_i(vec![5], 1);
        let delta = field.delta();
        assert!(
            delta.is_some(),
            "Q(√5, i) with 1 split prime should have δ > 0"
        );
        assert!(delta.unwrap().delta > 0.0);
    }

    #[test]
    fn pro2_tower_base_construction() {
        let field = CmField::pro2_tower_base();

        assert_eq!(field.total_degree(), 12);
        assert_eq!(field.complex_dim(), 6);
        assert_eq!(field.params.split_primes, vec![101]);
    }

    #[test]
    fn pro2_tower_base_verify() {
        let field = CmField::pro2_tower_base();
        let verification = field.verify_all();

        assert!(verification.split_primes_valid, "101 ≡ 1 (mod 4)");
        // For the pro-2 tower base with estimated params,
        // we check that the basic structure is correct
        assert!(verification.projection_injective);
    }

    #[test]
    fn enumerate_split_primes_basic() {
        let primes = enumerate_split_primes(50);

        // Primes ≡ 1 (mod 4) up to 50: 5, 13, 17, 29, 37, 41
        assert_eq!(primes, vec![5, 13, 17, 29, 37, 41]);
    }

    #[test]
    fn enumerate_split_primes_small() {
        assert!(enumerate_split_primes(4).is_empty());
        assert_eq!(enumerate_split_primes(5), vec![5]);
    }

    #[test]
    fn select_split_primes_basic() {
        let primes = select_split_primes(5);
        assert_eq!(primes, vec![5, 13, 17, 29, 37]);
    }

    #[test]
    fn select_split_primes_zero() {
        assert!(select_split_primes(0).is_empty());
    }

    #[test]
    fn class_number_bound_qi() {
        // Q(i): rd=1, degree=1 → small class number bound
        let bound = class_number_bound(1.0, 1);
        assert!(bound >= 1);
        // Q(i) has h=1, so bound should be small
        assert!(bound <= 10, "class number bound for Q(i) should be small");
    }

    #[test]
    fn class_number_bound_q_sqrt5_i() {
        // Q(√5, i): rd=√5, degree=2
        let bound = class_number_bound(5.0_f64.sqrt(), 2);
        assert!(bound >= 1);
    }

    #[test]
    fn optimize_qi_basic() {
        let field = optimize_qi(3);
        assert_eq!(field.params.split_primes.len(), 3);
        assert!(field.is_valid_construction());
    }

    #[test]
    fn optimize_q_sqrt5_i_basic() {
        let field = optimize_q_sqrt5_i(3, 1);
        assert_eq!(field.params.split_primes.len(), 3);
        assert!(field.is_valid_construction());
    }

    #[test]
    fn compare_delta_basic() {
        let fields = vec![
            CmField::qi(vec![5]),
            CmField::qi(vec![5, 13, 17]),
            CmField::q_sqrt5_i(vec![5], 1),
        ];

        let ranking = compare_delta(&fields);

        // More split primes → larger γ → larger δ → ranked first
        assert!(ranking[0].1.unwrap_or(0.0) >= ranking[1].1.unwrap_or(0.0));
    }

    #[test]
    fn from_params_generic() {
        let params = CmFieldParams {
            degree: 3,
            split_primes: vec![5, 13],
            class_number: 2,
            root_discriminant: 5.0,
            denominator: 2,
        };

        let field = CmField::from_params("Test Field", params, vec![1, 1]);

        assert_eq!(field.complex_dim(), 3);
        assert_eq!(field.total_degree(), 6);
        assert_eq!(field.params.class_number, 2);
        assert_eq!(field.name, "Test Field");
    }

    #[test]
    fn generate_point_set_qi() {
        let field = CmField::qi_default();
        let ps = field.generate_point_set(5.0);

        assert!(!ps.is_empty());
        // For Gaussian integers in radius 5, we should have points
        assert!(
            ps.len() >= 5,
            "expected at least 5 points, got {}",
            ps.len()
        );
    }

    #[test]
    fn invalid_split_prime_detected() {
        let field = CmField::qi(vec![3, 7]); // 3, 7 ≡ 3 (mod 4) — invalid
        assert!(!field.verify_split_primes());
    }
}
