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
//! - **Terrain cochains** — occupancy/safety/threat/destruction/interest fields
//! - **Lattice utility** — SIMD lattice-edge utility projection (Plan 335)
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
pub mod cache;
pub mod flow;
pub mod hodge;
pub mod lattice_utility;
pub mod operators;
pub mod simd;
pub mod stokes_calculus;
pub mod terrain_cochains;
pub mod types;

pub use backend::{DecBackend, select_backend};
pub use cache::{DecCache, DirtyRegion, affected_vertices, hodge_decompose_cached};
pub use flow::{DecFlowField, coexact_flow, exact_flow, harmonic_flow};
pub use hodge::{
    HodgeComponents, betti_numbers, dec_relevance_score, harmonic_projector, hodge_decompose,
    hodge_energy, hodge_residual, hodge_spectrum,
};
pub use lattice_utility::{HlaToCohainWeights, lattice_edge_utility_into};
pub use operators::{
    codifferential, exterior_derivative, graph_laplacian, hodge_laplacian, hodge_star,
};
pub use stokes_calculus::{
    belief_mass_divergence, boundary_flux_mass, boundary_flux_mass_indexed,
    boundary_flux_mass_only, circulation_integral, line_integral,
};

pub use types::{CellComplex, CoboundaryIndex, CochainField, MAX_RANK};

pub use terrain_cochains::{
    DestructionCochain, InterestCohain, OccupancyCochain, SafetyCochain, ThreatCochain,
};
