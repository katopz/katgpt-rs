//! Discrete Exterior Calculus (DEC) operators on cell complexes.
//!
//! Based on "Topological Neural Operators" (arXiv:2606.09806).
//!
//! Provides inference-time topological operators for game spatial reasoning:
//! - **Cell complex** — vertices, edges, faces, volumes with oriented incidence
//! - **Cochain fields** — typed feature vectors on cells of a given rank
//! - **DEC operators** — gradient d₀, curl d₁, divergence d₂, codifferential δₖ
//! - **Hodge Laplacian** — Δₖ = δₖ₊₁dₖ + dₖ₋₁δₖ (conservation-by-construction)
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
//! use katgpt_core::dec::{CellComplex, CochainField, exterior_derivative};
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

pub mod backend;
pub mod flow;
pub mod hodge;
pub mod operators;
pub mod types;

pub use backend::{DecBackend, select_backend};
pub use flow::{DecFlowField, coexact_flow, exact_flow, harmonic_flow};
pub use hodge::{
    HodgeComponents, betti_numbers, dec_relevance_score, harmonic_projector, hodge_decompose,
    hodge_energy, hodge_residual, hodge_spectrum,
};
pub use operators::{
    codifferential, exterior_derivative, graph_laplacian, hodge_laplacian, hodge_star,
};
pub use types::{CellComplex, CochainField, MAX_RANK};
