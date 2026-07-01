//! Unit Distance GOAT Proof — Number-theoretic lattice constructions.
//!
//! Implements the disproof of Erdős's 1946 unit distance conjecture using
//! CM field constructions from algebraic number theory. The key result:
//! ν(n) ≥ n^(1+δ) for some δ > 0, refuting the conjectured ν(n) ≤ n^(1+o(1)).
//!
//! # Architecture
//!
//! | Component | Type | Description |
//! |-----------|------|-------------|
//! | `MinkowskiLattice` | Modelless (T1) | High-dim lattice with sup-norm packing |
//! | `ClassGroupPigeonhole` | Modelless (T2) | Norm-one element counting via ideal class |
//! | `CmField` | Light model-based (T4) | CM field construction with prescribed splitting |
//! | `TowerSearch` | Model-based (T5) | UCB1 bandit search for optimal tower parameters |
//!
//! # Reference
//!
//! OpenAI (2026), "Planar Point Sets with Many Unit Distances"
//! Alon, Bloom, Gowers et al. (2026), "Remarks on the Disproof..."
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "unit_distance")]` — zero default impact.

#[cfg(feature = "unit_distance")]
mod cm_field;
#[cfg(feature = "unit_distance")]
mod minkowski;
#[cfg(feature = "unit_distance")]
mod pigeonhole;
#[cfg(feature = "unit_distance")]
mod tower_search;
#[cfg(feature = "unit_distance")]
mod types;

#[cfg(feature = "unit_distance")]
pub use cm_field::{
    CmField, FieldVerification, class_number_bound, compare_delta, enumerate_split_primes,
    optimize_q_sqrt5_i, optimize_qi, select_split_primes,
};
#[cfg(feature = "unit_distance")]
pub use minkowski::MinkowskiLattice;
#[cfg(feature = "unit_distance")]
pub use pigeonhole::{
    ClassGroupPigeonhole, pigeonhole_q_sqrt5_i, pigeonhole_qi, sum_of_two_squares,
    verify_pigeonhole_bound,
};
#[cfg(feature = "unit_distance")]
pub use tower_search::{
    TowerArm, TowerBandit, TowerFamily, TowerSearch, TowerSearchConfig, TowerSearchResult,
};
#[cfg(feature = "unit_distance")]
pub use types::{
    C64, CmFieldParams, DeltaEstimate, PigeonholeResult, PointSet, PrimePair, count_unit_distances,
};
