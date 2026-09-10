//! katgpt-dec — Discrete Exterior Calculus (DEC) substrate.
//!
//! Pure math substrate for Stokes calculus on cell complexes. No app semantics.
//! Spun out of `katgpt-core::dec` (Issue 007 Phase E Tier 1) as a standalone
//! publishable crate mirroring the `katgpt-transformer` template.
//!
//! Based on "Topological Neural Operators" (arXiv:2606.09806).
//!
//! # What's here
//!
//! - **Cell complex** — vertices, edges, faces, volumes with oriented incidence
//! - **Cochain fields** — typed feature vectors on cells of a given rank
//! - **DEC operators** — gradient d₀, curl d₁, divergence d₂, codifferential δₖ
//! - **Hodge Laplacian** — Δₖ = δₖ₊₁dₖ + dₖ₋₁δₖ (conservation-by-construction)
//! - **Stokes calculus** — boundary flux, line integrals, belief-mass divergence
//! - **Hodge decomposition** — exact ⊕ harmonic ⊕ coexact (Helmholtz split)
//! - **Motor-gated field** (opt-in, `motor_gated_field` feature, Plan 357) —
//!   Amari-style neural-field evolution step unifying the Hodge Laplacian with
//!   a per-channel motor gain (`evolve_motor_gated_field`).
//! - **PCA global-function layer** (opt-in, `pca_global` feature, Plan 591) —
//!   wires the DEC aggregates into the CA decision function: one global
//!   evaluation per tick + a per-cell decision pass over the untouched
//!   birth/death kernel (`step_pca_sync`).
//!
//! # Conservation Guarantees
//!
//! The fundamental identity `dₖ₊₁ ∘ dₖ = 0` holds exactly:
//! - `curl(grad) = 0`: gradient fields never have circulation
//! - `div(curl) = 0`: curl fields never have divergence
//!
//! # Usage
//!
//! ```ignore
//! use katgpt_dec::{CellComplex, CochainField, exterior_derivative};
//!
//! // Create a 2D grid cell complex
//! let cx = CellComplex::grid_2d(64, 64);
//!
//! // Define a potential on vertices (rank-0 cochain)
//! let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
//! // ... fill potential values ...
//!
//! // Compute gradient (rank-0 → rank-1)
//! let gradient = exterior_derivative(&cx, &potential);
//!
//! // Compute curl (rank-1 → rank-2) — guaranteed zero if input is a gradient!
//! let curl = exterior_derivative(&cx, &gradient);
//! ```
//!
//! # Backwards compatibility
//!
//! `katgpt-core` re-exports this crate as `katgpt_core::dec` via a
//! `pub use katgpt_dec as dec;` shim, so all historical
//! `katgpt_core::dec::*` paths continue to work unchanged.

pub mod backend;
#[cfg(feature = "grid_3d")]
pub mod birth_death;
#[cfg(feature = "heat_kernel_trajectory")]
pub mod bom_heat_kernel;
pub mod cache;
pub mod flow;
#[cfg(feature = "heat_kernel_trajectory")]
pub mod heat_kernel;
pub mod hodge;
#[cfg(feature = "htno_v_cycle")]
pub mod htno;
#[cfg(feature = "heat_kernel_trajectory")]
pub mod krylov;
#[cfg(feature = "motor_gated_field")]
pub mod motor_gated;
#[cfg(feature = "heat_kernel_trajectory")]
pub mod nonlinear_heat_kernel;
pub mod operators;
#[cfg(feature = "pca_global")]
pub mod pca;
#[cfg(feature = "cochain_point_sampler")]
pub mod point_sampler;
#[cfg(feature = "se2_equivariant_lift")]
pub mod se2_lift;
#[cfg(feature = "sheaf_admm")]
pub mod sheaf_admm;
pub mod simd;
pub mod stokes_calculus;
pub mod types;

pub use backend::{DecBackend, select_backend};
pub use cache::{DecCache, DirtyRegion, affected_vertices, hodge_decompose_cached};
pub use flow::{DecFlowField, coexact_flow, exact_flow, harmonic_flow};
pub use hodge::{
    HodgeComponents, betti_numbers, dec_relevance_score, harmonic_projector, hodge_decompose,
    hodge_energy, hodge_residual, hodge_spectrum,
};
pub use operators::{
    codifferential, exterior_derivative, graph_laplacian, hodge_laplacian, hodge_star,
};
pub use stokes_calculus::{
    belief_mass_divergence, boundary_flux_mass, boundary_flux_mass_indexed,
    boundary_flux_mass_only, circulation_integral, line_integral,
};

#[cfg(feature = "motor_gated_field")]
pub use motor_gated::{evolve_motor_gated_field, relu_gate_into};

// Plan 454 T4 — stochastic birth/death NCA growth step (modelless, opt-in).
// The 3D sibling of `evolve_motor_gated_field`: composes the shipped DEC
// Laplacian (7-point stencil via the `grid_3d` fast path) with a fixed-seed
// SplitMix64 PRNG and a sigmoid alive gate. Zero-alloc, bit-identical under
// a fixed seed (G6 quorum-safety).
//
// Plan 454 T5 — `argmax_block_type` raw → categorical bridge: thresholds the
// continuous cochain into a `u8` block-class per cell. Intended for a future
// civ-engine city-growth consumer (T9 caveat: no such consumer exists today —
// see birth_death.rs docs).
#[cfg(feature = "grid_3d")]
pub use birth_death::{
    BirthDeathParams, SplitMix64, argmax_block_type, stochastic_birth_death_step,
};

// Plan 591 — PCA global-function layer: DEC aggregates as the CA decision
// function's global channel (Programmable CA, arXiv:2609.06102). Opt-in;
// composes the Plan 454 kernel (above) with the always-on DEC operators —
// promote to default only on the Phase 3 GOAT pass.
#[cfg(feature = "pca_global")]
pub use pca::{
    GlobalScalars, GlobalTargetGate, PcaDecision, PcaGlobalFn, PcaScratch, StopWhen, step_pca_sync,
};

// Plan 560 — SE(2)-equivariant lifting layer (Smets §3.4.1).
// DEFAULT-ON (Phase 2 GOAT G1+G2 ALL PASS 2026-07-25).
#[cfg(feature = "se2_equivariant_lift")]
pub use se2_lift::{se2_lift_into, se2_project_integrate_into, se2_project_max_into};

#[cfg(feature = "heat_kernel_trajectory")]
pub use heat_kernel::{
    DecEigendecomposition, K_MAX, NULL_SPACE_THRESHOLD, heat_kernel_trajectory_krylov,
    heat_kernel_trajectory_krylov_into, heat_kernel_trajectory_linear,
    heat_kernel_trajectory_linear_into,
};

#[cfg(feature = "heat_kernel_trajectory")]
pub use krylov::{KRYLOV_K_MAX, krylov_expmv, krylov_expmv_into};

#[cfg(feature = "heat_kernel_trajectory")]
pub use nonlinear_heat_kernel::{
    DEFAULT_N_QUAD, MAX_N_QUAD, NonlinearScratch, expm_source_term_quadrature,
    heat_kernel_trajectory_nonlinear, heat_kernel_trajectory_nonlinear_into,
};

// Plan 359 Phase 4 — BoM trajectory sampling (multi-hypothesis heat kernel).
// Opt-in extension of the linear path: perturbs h₀ along the near-harmonic
// subspace and applies the heat kernel to each of K hypotheses. The
// diversity-for-exploration analog of BoMSampler (Plan 281) in trajectory
// space.
#[cfg(feature = "heat_kernel_trajectory")]
pub use bom_heat_kernel::{
    heat_kernel_trajectory_bom, heat_kernel_trajectory_bom_into, near_harmonic_indices,
};

// Plan 407 — Sheaf-ADMM coordination primitive (modelless, DEFAULT-ON since
// Phase 2 GOAT gate G1–G6 ALL PASS, 2026-07-07, Bench 407).
#[cfg(feature = "sheaf_admm")]
pub use sheaf_admm::{
    AdmmScratch, LocalObjective, SheafMaps, sheaf_admm_step, sheaf_admm_step_cg_into,
    sheaf_admm_step_into, sheaf_admm_step_soft_into,
};

// Multi-scale V-cycle via selector restriction maps — opt-in. Composes the
// single-complex DEC operators into a two-level fine→coarse→fine hierarchy.
#[cfg(feature = "htno_v_cycle")]
pub use htno::{
    VCycleRestriction, VCycleScratch, grid_coarsen_2x2, htno_v_cycle, htno_v_cycle_into,
};

// Continuous cochain point sampler (Plan 422, Research 404) — Whitney/de-Rham
// reconstruction for continuous field queries inside primitives.
#[cfg(feature = "cochain_point_sampler")]
pub use point_sampler::{
    LocalCoordEncode, PointSamplerScratch, lambda_coordinate_quad, lambda_coordinate_tri,
    local_coord_aug_dim, local_coordinate_aug_barycentric, local_coordinate_aug_cartesian,
    local_coordinate_quad, local_coordinate_tri, sample_cochain_at_point_quad_into,
    sample_cochain_at_point_tri_into, sample_point_quad_into, sample_point_tri_into,
};

pub use types::{CellComplex, CoboundaryIndex, CochainField, MAX_RANK};

// Shared test helpers for DEC operator tests (Issue 037 T1 verdict).
// Extracted from 3 duplicated copies in heat_kernel.rs, motor_gated.rs,
// nonlinear_heat_kernel.rs to eliminate ~80 LOC of drift risk and 3 of 4
// clippy::too_many_arguments lints. Declared at crate root so all test modules
// share a single compilation via `crate::test_common::*`.
#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod test_common;
