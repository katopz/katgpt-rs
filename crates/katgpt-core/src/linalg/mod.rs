//! Shared linear-algebra kernels extracted for reuse across ridge-style solvers.
//!
//! Currently consumed by [`crate::karc`] (Plan 308, Research 288, arXiv:2606.19984).
//! The f32 Cholesky-based SPD inverse + ridge solve live in [`ridge_solve`].
//!
//! # Why not unify with `peira` yet?
//!
//! `peira.rs` owns an f64 Cholesky path (`invert_spd_into`, `matmul_into`) that is
//! private to that module and tightly coupled to its EMA covariance tracking.
//! Extracting it generically would risk destabilising PEIRA's bit-exact f64
//! numerics (Plan 153 GOAT G4 reproducibility). Per the correctness-first rule in
//! AGENTS.md, this module ships a standalone f32 path for KARC and leaves a
//! `// TODO: unify with peira's f64 path` note rather than touching PEIRA.
//!
//! Unification is tracked as future work once a generic-over-`T: Float` Cholesky
//! is benchmarked to be bit-identical to the current f64 specialisation.

pub mod ridge_solve;

#[cfg(feature = "geometric_product")]
pub mod geometric_product;

pub use ridge_solve::{
    chol_solve_f32, chol_solve_f64, cholesky_f32, cholesky_f64,
    ridge_solve_direct_f32, ridge_solve_direct_f64, ridge_solve_woodbury_f32,
    spd_inverse_f32,
};

// Plan 319 — Channel-wise Clifford Geometric Product (coherence + wedge).
// Re-exported alongside the ridge kernels as a peer linear-algebra primitive.
#[cfg(feature = "geometric_product")]
pub use geometric_product::{
    cyclic_shift_into, geometric_product_into, geometric_product_wedge_into,
};
