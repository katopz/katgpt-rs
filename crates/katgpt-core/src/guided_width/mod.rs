//! Guided width rollouts on the belief host — Issue 895 (Research 590, the
//! GRAM re-distill, arXiv:2605.19376; supersedes Research 058 on this host
//! only). Plan 095's pending width-vs-depth G1/G3 are completed by Bench 898.
//!
//! GRAM makes recursive latent refinement *stochastic and wide*: each
//! high-level transition is `h_t = u_t + ε_t` — a deterministic proposal plus
//! guidance noise `ε ~ N(μ(u), σ²)` — and N parallel trajectories are ranked
//! pre-decode. This module ships the modelless version for a host with **no
//! reward signal** (per-NPC fog-of-war belief deliberation), as operators that
//! consume the shipped substrate rather than fork it:
//!
//! | Task | Operator | Consumes |
//! |---|---|---|
//! | T1 (a) | [`perturb::Transversal`] — `ε = σ_t·P_⊥ v`, `P_⊥ = I − ûûᵀ` | [`crate::diversity::temp::blake3_noise_fill`] (the ε source) |
//! | T1 (b) | `hodge_arm::MassConserving` — ε ∈ coexact ∪ harmonic, `δ₁ε ≡ 0` | `katgpt-dec` `codifferential_into` (feature `guided_width_hodge`) |
//! | T2 | [`types::StagnationGate`] — `σ_t = σ_max·sigmoid(α(w_stuck − w₀))` | `fast_sigmoid` |
//! | T3 | [`score::latent_value_into`] — self-consistency + convergence residual + frozen direction | sigmoid scoring, [`crate::float_order`] |
//! | T4 | [`init::sobol_init_into`] + [`init::diverse_set_into`] | [`crate::speculative::qmc::SobolQmc`], [`crate::diversity::temp::select_diverse_subset_in_place`] |
//! | T5 | [`table::DirectionTable`] + [`table::DirectionPosterior`] | `thin_svd_into` (the kernel `katgpt-canon::fit_joint_svd_pair` itself delegates to), [`crate::best_belief::best_belief_score`] |
//! | T6 | trap-kill-reallocate inside [`rollout::guided_width_rollouts`] | [`crate::saddle_escape::FlipDetector`], [`crate::saddle_escape::apply_kick`] |
//!
//! # Why `thin_svd_into` and not `fit_joint_svd_pair` (substrate-first)
//!
//! `katgpt-canon` depends on `katgpt-core`, so calling `fit_joint_svd_pair`
//! from here is a package cycle. That function is the TWO-source
//! specialisation (`[A | B]` joint SVD + Procrustes) of the one kernel both
//! use — `subspace_phase_gate::thin_svd_into`; the direction table is
//! single-source (successful `Δh` rows), so it consumes that kernel one level
//! down. No second SVD exists.
//!
//! # Kill switch (G3)
//!
//! `n_branches ≤ 1` OR `sigma_max == 0` runs the incumbent path verbatim:
//! `k_steps` calls of the caller's deterministic step on a copy of `h0` — no
//! noise, no guidance, no trap machinery. Bit-identical to the host's
//! `evolve_belief` loop by construction (`belief_host` test). Guidance
//! magnitudes are `κ·σ`, so `σ = 0` also zeroes the μ≠0 term.
//!
//! # Table-absent fallback (T5)
//!
//! `Hooks::guidance == None` (or a zero-direction table) takes the exact
//! isotropic+transversal code path — bit-identical, pinned by test.
//!
//! # Boundary
//!
//! Public modelless substrate: no training, no game vocabulary, sigmoid never
//! softmax. Everything here is **think-brain latent** — nothing crosses a
//! sync boundary; a consumer that surfaces anything syncs scalars only. The
//! consumer (per-NPC belief deliberation) is riir-ai Issue 1008. The DDTree /
//! logit lane is NOT touched (Research 058's covered verdict stands).
//!
//! # Posture
//!
//! Opt-in `guided_width_rollouts` (+ `guided_width_hodge` for arm (b)).
//! Research 058 §8.3's "do NOT make guided noise the default" governs unless
//! Bench 898 records a both-families win (the pre-stated demote condition).

pub mod init;
pub mod perturb;
pub mod rollout;
pub mod score;
pub mod table;
pub mod types;

#[cfg(feature = "guided_width_hodge")]
pub mod hodge_arm;

#[cfg(feature = "sense_composition")]
pub mod belief_host;

pub use init::{diverse_set_into, sobol_init_into};
pub use perturb::{Perturbation, Transversal, transversal_perturbation_into};
pub use rollout::{belief_key, guided_width_rollouts};
pub use score::{latent_value_into, select_best};
pub use table::{DirectionFitScratch, DirectionPosterior, DirectionTable, ThawError};
pub use types::{
    GuidedWidthConfig, GuidedWidthScratch, Guidance, Hooks, LatentValueConfig, MAX_BRANCHES,
    MAX_DIRECTIONS, RolloutReport, StagnationGate, TrapProbe, TrapReallocConfig,
};
