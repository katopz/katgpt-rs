//! MOP — Maximum Occupancy Principle value-iteration primitive
//! (Plan 573 / Research 478; paper: [arXiv:2205.10316](https://arxiv.org/abs/2205.10316),
//! Ramírez-Ruiz et al., Nat. Commun. 15, 6368 (2024), CC-BY 4.0).
//!
//! **The one-line selling point:** a reward-free optimal policy — the
//! closed-form Bellman-like fixed point that maximizes the entropy of the
//! future action-state path, yielding emergent survival (absorbing states
//! have exactly zero value), persistent behavioral stochasticity, and a
//! tunable risk knob (β) — all from a frozen transition model, zero
//! training, zero gradient descent.
//!
//! # What ships here
//!
//! - [`MopSolver<N, A>`](solve::MopSolver) — the paper's Eq. 7 fixed-point
//!   map in log-space LSE form over a frozen tabular kernel
//!   `p[N][A][N]` + availability mask. Converges unconditionally (paper
//!   Theorem 3); absorbing/terminal states pinned to `V = 0` bit-exact.
//! - [`pi_star`](solve::MopSolver::pi_star) — the optimal categorical
//!   policy, closed-form from the fixed point.
//! - [`arenas`] — the shared gridworld/ring domain builders (tests, benches,
//!   and riir-ai's private parity harness).
//!
//! # Modelless + sync boundary
//!
//! Pure deterministic math on caller-owned arrays (zero allocation with the
//! caller-provided [`MopScratch`](solve::MopScratch)). Nothing crosses a
//! sync boundary; π\* is a local control policy, not a synced distribution.
//!
//! # Softmax exemption (house "sigmoid, never softmax" rule)
//!
//! π\*'s `exp/Z` normalization is the paper's exact categorical-distribution
//! math (Eq. 5) — it must sum to 1 over available actions. The house rule
//! governs semantic scalar projections (emotion gates, attention boosts),
//! which this is not. **Do not "fix" this normalization to sigmoid** — it
//! would corrupt the math.
//!
//! # UQ floor ("Report the Floor") — N/A
//!
//! MOP claims no predictive distribution, interval, coverage, or calibrated
//! uncertainty: `V*` is a path-occupancy value and π\* a control policy,
//! validated on behavior gates (riir-ai Bench 679), not forecast
//! calibration. The conformal-naive floor does not apply.
//!
//! # References
//!
//! - Research: `katgpt-rs/.research/478_MOP_Maximum_Occupancy_Principle.md`
//!   (Super-GOAT verdict: 3/4 gates hard PASS + G4 pass-with-caveat)
//! - Private runtime guide: `riir-ai/.research/338_per_npc_mop_runtime_guide.md`
//!   + wiring plan `riir-ai/.plans/538_per_npc_mop_runtime.md` (the consumer)
//! - Defend-wrong PoC: `riir-ai/crates/riir-poc/src/mop_poc.rs` (Bench 679;
//!   the permanent §3.6 regression check — NOT a source template for this
//!   crate; this implementation is fresh from the paper's math)
//!
//! MOP — Maximum Occupancy Principle value-iteration primitive (Plan 573 /
//! Research 478, arXiv:2205.10316). The paper's Eq. 7 fixed-point map in
//! log-space LSE form over a frozen tabular kernel — reward-free optimal
//! policy with emergent survival (absorbing states V=0 bit-exact) and a β
//! risk knob. Opt-in `mop_path_entropy`. Consumers: riir-ai Plan 538.

pub mod arenas;
pub mod solve;
pub mod types;

pub use solve::{MopScratch, MopSolver, action_entropy_nats, state_conditional_entropy};
pub use types::{MopConfig, MopConfigError, MopSolution};
