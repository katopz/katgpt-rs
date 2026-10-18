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
//!
//! Shared linear-algebra kernels. Originally extracted for `karc`'s ridge-style
//! solvers (Plan 308); the f32 Cholesky/ridge path lives here as a standalone
//! extraction of the PEIRA `(N + λI)⁻¹` pattern — see the module note for why
//! PEIRA's f64 path is left untouched. Plan 319 (Clifford geometric product)
//! and Plan 326 (Tucker/HOSVD tensor factorization) ship peers under `linalg::`
//! — each must also gate this `pub mod` so the crate compiles when only that
//! feature is on. Issue 684 (`svd_cca`) joins the same rule for its
//! `symmetric_eig` consumption.

pub mod ridge_solve;

// Issue 186 (Path B, 2026-07-20) — Householder tridiagonalization + implicit-shift
// QL. Drop-in alternative to `karc::jacobi_eigen` for large symmetric matrices
// (~5-10× faster at n ≥ 256). Always compiled; consumed by `karc::large_dh`
// under the `karc_householder_eig` feature gate.
pub mod symmetric_eig;

#[cfg(feature = "geometric_product")]
pub mod geometric_product;

// Plan 326 — Tucker / HOSVD N-mode tensor factorization (the N-mode
// generalization of `thin_svd_into`). Distilled from TFNO §6.1 as the third
// and final FNO gap (Research 307 §3 candidate plan #3).
#[cfg(feature = "tucker_factorization")]
pub mod tucker;

// Issue 839 — Kronecker-factored tile apply `(A ⊗ B) · x = A Z Bᵀ` as two small
// GEMMs, plus the delegating Walsh–Hadamard fast path. The arbitrary-factor
// generalization of `katgpt-kv::kvarn::hadamard`'s fixed-H tile machinery
// (Research 569, Cactus Needle 3).
#[cfg(feature = "kron_tile")]
pub mod kron_tile;

pub use ridge_solve::{
    NotPositiveDefinite, chol_solve_f32, chol_solve_f64, cholesky_f32, cholesky_f64,
    ridge_solve_direct_f32, ridge_solve_direct_f64, ridge_solve_woodbury_f32, spd_inverse_f32,
    try_cholesky_f32, try_ridge_solve_woodbury_f32,
};

// Issue 186 (Path B) — symmetric eigendecomposition via Householder + QL.
pub use symmetric_eig::{SymmetricEigScratch, symmetric_eig};

// Plan 319 — Channel-wise Clifford Geometric Product (coherence + wedge).
// Re-exported alongside the ridge kernels as a peer linear-algebra primitive.
#[cfg(feature = "geometric_product")]
pub use geometric_product::{
    cyclic_shift_into, geometric_product_into, geometric_product_wedge_into,
};

// Plan 326 — Tucker / HOSVD factorization re-exports. Peer to the SVD
// primitives in `subspace_phase_gate`; this is their N-mode generalization.
#[cfg(feature = "tucker_factorization")]
pub use tucker::{
    MAX_MODES, TuckerConfig, TuckerError, TuckerResult, TuckerResultScratch, TuckerScratch,
    tucker_decompose, tucker_decompose_into, tucker_reconstruct_into,
};

// Issue 839 — Kronecker-factored tile apply, re-exported alongside the other
// linear-algebra kernels.
#[cfg(feature = "kron_tile")]
pub use kron_tile::{
    KronScratch, WHT_TILE_WIDTHS, dense_matvec_into, is_permutation, kron_apply,
    kron_apply_tile_into, kron_dense_into, permute_into, wht_apply_tiles, wht_factor_into,
};
