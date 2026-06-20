//! CCE — Coarse Correlated Equilibria moderator primitives (Plan 295, Research 274).
//!
//! Generic, game-agnostic implementation of the LP-CCE formulation and
//! no-regret primal-dual learning algorithm from Campi, Cannerozzi, Tzouanas
//! 2026 (arxiv 2606.20062). Three primitives will ship in this module:
//!
//! 1. `ExternalRegret` — closed-form external-regret functional on a finite
//!    deviation class, plus uniqueness check (Assumption 6.2) and linear
//!    derivative (Lemma 6.5).
//! 2. `CceLp<N, A>` — finite occupation-measure LP solver (Phase 2).
//! 3. `CcePrimalDual` — Bregman-regularized primal-dual iterator with
//!    `O(N⁻¹ᐟ²)` averaged-iterate convergence (Phase 2).
//!
//! **Phase 1 ships only `ExternalRegret` + core types.** Phase 2 adds the LP
//! solver and primal-dual iterator. Phase 3 adds benchmarks and examples.
//!
//! ## Convention
//!
//! `gamma` is the **cost** functional (minimize). The CCE LP minimizes
//! `gamma0(ρ)` subject to `gamma(ρ) ≤ gamma_dev(ρ, κ)` for all `κ ∈ D`.
//!
//! External regret `ER(ρ) = max_{κ ∈ D} (γ(ρ) − γ_dev(ρ, κ))`:
//! - `ER = 0` at Nash (marginal CCE).
//! - `ER < 0` at a strict CCE (every deviation strictly worse).
//! - `ER > 0` is NOT a CCE (profitable deviation exists).
//!
//! ## Sigmoid-only rule
//!
//! This module contains no activations. The CCE formulation is purely linear
//! algebra over occupation measures. Sigmoid gates appear in the riir-ai
//! runtime binding (Plan 325), not here.
//!
//! ## Latent-space contract
//!
//! This is the **public open primitive** — pure generic math, MIT-licensed,
//! no game semantics. The latent-space reframing (state = HLA bucket, action =
//! CGSP conjecturer arm, signal = zone-mood latent scalar) lives in riir-ai
//! Plan 325. See `AGENTS.md` "Latent vs Raw Space Rules" for the boundary.

pub mod bregman;
pub mod external_regret;
pub mod lp;
pub mod primal_dual;
pub mod types;

pub use bregman::{BregmanPotential, Euclidean, Kl};
pub use external_regret::ExternalRegret;
pub use lp::{CceLp, CceLpError};
pub use primal_dual::{CcePrimalDual, ConvergenceReportRaw, StepReport};
pub use types::{
    ActionSpace, Deviation, DeviationClass, OccupationMeasure, OccupationMeasureError,
    PayoffTensor, StateSpace,
};
