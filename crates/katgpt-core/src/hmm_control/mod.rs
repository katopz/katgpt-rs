//! HMM homeostatic control — the exact setpoint-reachability drive solver
//! (Plan 590 / Research 543; paper: [arXiv:2609.07508](https://arxiv.org/abs/2609.07508),
//! Moreno-Bote, "Homeostasis Revisited and Reformulated Through Hidden
//! Markov Model Control", 2026).
//!
//! **The one-line selling point:** the provably-optimal DETERMINISTIC
//! policy for "maximize the probability of the desired observation at
//! every step of the horizon" — multiplicative backward messages
//! β_t(x) = P(y_{t:T} = y_d | x) bounded [0, 1] by construction, no
//! discount factor (the horizon replaces γ), and risk sensitivity by
//! construction (transition mass weighs the argmax). The setpoint-drive
//! sibling of [`crate::mop`] (range drives: entropy-optimal exploration;
//! this module: exact goal attainment — the paper proves the variational/
//! free-energy route is suboptimal + stochastic for this objective).
//!
//! # What ships here
//!
//! - [`HmmControlSolver<N, A, T>`](solve::HmmControlSolver) — the paper's
//!   Eqs. 14-15 backward recursion over a frozen tabular kernel
//!   `p[N][A][N]` + per-step emission `e[T][N][A]`. Zero allocation (const
//!   -generic arrays, returned by value), single backward sweep (no
//!   iteration, no scratch).
//! - [`HmmSolution`](types::HmmSolution) — `beta[T][N]` (reachability
//!   messages) + `policy[T][N]` (deterministic argmax actions, ties to the
//!   lowest index).
//! - [`validate_tables`](types::validate_tables) — opt-in input contract
//!   (off the hot path); [`invariant_emission`](types::invariant_emission)
//!   — the time-invariant emission helper.
//!
//! # Modelless + sync boundary
//!
//! Pure deterministic math on caller-owned arrays. Nothing crosses a sync
//! boundary; the action is raw-deterministic, and any caller-side affect
//! projection of β (e.g. a fear scalar) crosses only as a bounded scalar
//! after its own gated bridge — same discipline as `mop`.
//!
//! # UQ floor ("Report the Floor") — N/A
//!
//! β is an exact model-computed probability, not a calibrated estimate
//! over data. The conformal-naive floor does not apply.
//!
//! # References
//!
//! - Research: `katgpt-rs/.research/543_HMM_Homeostatic_Control_Deterministic_Drives.md`
//! - Private runtime guide: `riir-ai/.research/370_per_npc_homeostatic_drive_control_guide.md`
//!   (the two-mode homeostat composition with `mop`)
//! - Classical lineage (honesty): stochastic shortest-path reachability —
//!   the novelty claim is scoped to the game-runtime fusion, not the math.

pub mod solve;
pub mod types;

pub use solve::HmmControlSolver;
pub use types::{HmmInputError, HmmSolution, invariant_emission, validate_tables};
