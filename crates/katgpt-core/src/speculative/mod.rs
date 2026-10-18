//! Speculative-decoding substrate (Plan 008 Step 5).
//!
//! Pure substrate types for speculative decoding: data types, configs,
//! algorithms, and trait implementations that depend only on
//! [`crate::types::Config`], [`crate::traits`], and std.
//!
//! ## What lives here
//! - [`types`]: `TreeNode` + `TokenPath` (Issue 670 vocab-widened path),
//!   `DraftResult`, `DraftEvent`, `RejectionReason`,
//!   `DecodeStrategy`, `SdeConfig`, `EarlyStopGate<P>`, `FlashPrefillConfig`,
//!   `BlockScores`, LDT conflict detector (`ConflictDetector`,
//!   `EntropyConflictDetector`), `TesNode`, `TrajectoryCredit`, and various
//!   feature-gated config types (DFlare fusion/kv-routing/progressive-budget,
//!   `LdtPruneConfig`, `SpecCostSnapshot`, `RoutingOverlapSnapshot`,
//!   `StabilitySnapshot`).
//! - [`sampling`]: CDF-based sampling primitives (`sample_from_distribution`,
//!   `sample_residual_distribution`, `sample_residual_distribution_into`).
//!
//! ## What does NOT live here (stays in consumer crates)
//! - The companion traits (`ConstraintPruner`, `ScreeningPruner`, `DominoPruner`,
//!   `NoPruner`, `NoScreeningPruner`, `BinaryScreeningPruner`) — already in
//!   [`crate::traits`] since Plan 107 Phase 0.
//! - Composition types that need `katgpt-transformer`:
//!   `SpeculativeContext` (moved to `katgpt-forward` Plan 393 — composes
//!   `ForwardContext`), `DDTreeBranchCache` — these need `ForwardContext`,
//!   `MultiLayerKVCache`, `PagedKVCache`, `forward_paged`.
//! - Consumer-crate-specific composition: `TesConfig` (needs `BanditStrategy`),
//!   `SelfSpecConfig` (needs `D2fDecodeConfig`).
//! - The DDTree builders (`build_dd_tree*`, `TreeBuilder`) — composition that
//!   drives the substrate; stays in the consumer.
//!
//! ## Feature gating
//! Always-on (no feature gate on the module itself — same pattern as
//! [`crate::simd`], [`crate::types`], [`crate::traits`], [`crate::hla`]).
//! Individual types are gated by their respective feature flags, forwarded
//! from the consumer via `katgpt-core/<feature>` (e.g. `katgpt-core/elf_sde`
//! gates `EarlyStopGate`).
//!
//! Plan 370 — QMC noise-fill convenience entry point (constructs the right
//! QmcSource from a QmcMethod tag + seed, zero-alloc). Used by
//! MultiHypothesisBoMMinimaxPlanner when NoiseQueryConfig::qmc_method is Some.

pub mod sampling;
pub mod types;

// QMC uniform sources (Plan 367, Research 367 — QuasiMoTTo). Opt-in behind
// `qmc_sampling`. Produces correlated-but-marginally-exact k-point batches that
// are a drop-in for i.i.d. `rng.uniform()` in K-rollout paths.
#[cfg(feature = "qmc_sampling")]
pub mod qmc;

// Plan 367 Fusion E — QmcHalter: sample-efficiency-aware halting for QMC
// rollout budgets (Research 367 §2.3 Fusion E + Research 205 §1 union bound).
// The sample-efficiency-aware analog of `GainCostLoopHalter` (Plan 304) —
// decides whether to draw more QMC rollouts based on the union-bound ceiling
// `min(1, k·p)` vs the empirical coverage from the actual point set.
#[cfg(feature = "qmc_sampling")]
pub mod qmc_halter;

// Re-export the substrate API at `katgpt_core::speculative::*` for ergonomic
// imports (`use katgpt_core::speculative::{TreeNode, DraftResult};`).
pub use types::*;

// Sampling primitives are always-on (depend only on `crate::types::Rng`).
// Re-exported here for one-stop access. `sample_residual_distribution` is the
// allocating convenience wrapper (deprecated in favor of `_into`), but the
// re-export itself is intentional — downstream crates may still reference it.
#[allow(deprecated)]
pub use sampling::{
    sample_from_distribution, sample_residual_distribution, sample_residual_distribution_into,
};

// Plan 367 Phase 2 — QMC arithmetic-coding descend with coordinate carry.
// Gated on `qmc_sampling` for feature isolation (G6). Drop-in per-step
// descend operator that turns any `qmc::QmcSource` into a token-sequence
// sampler; bit-identical to `sample_from_distribution` on fresh i.i.d. draws.
#[cfg(feature = "qmc_sampling")]
pub use sampling::sample_from_distribution_qmc;
// Plan 367 Phase 3 — K-rollout QMC sampler (draws K correlated-but-
// marginally-exact sequences from a per-position distribution array).
#[cfg(feature = "qmc_sampling")]
pub use sampling::sample_k_from_distribution_qmc;

// Plan 367 Phase 4 — QMC → Gaussian noise query fill (Fusion A: QmcBoMSampler).
// Inverse normal CDF (probit) + gaussianize + fill_noise_queries helper. The
// `sample_k_states_qmc` convenience wrapper additionally requires
// `bom_sampling` (the BoMSampler trait + NoiseQueryConfig live there).
//
// Plan 370 — `fill_noise_queries_gaussian_qmc_by_method` is the QmcMethod-tag
// convenience entry point used by MultiHypothesisBoMMinimaxPlanner.
#[cfg(feature = "qmc_sampling")]
pub use qmc::{
    BootstrapEstimate, QmcSource, contiguous_block_bootstrap_pass_at_m,
    dyadic_bootstrap_pass_at_m_lattice, fill_noise_queries_gaussian_qmc,
    gaussianize_uniforms_inplace, inverse_normal_cdf,
};

// Plan 367 Fusion E — QmcHalter re-exports. Sample-efficiency-aware halting
// for QMC rollout budgets (R367 §2.3 Fusion E + R205 §1 union bound).
#[cfg(all(feature = "qmc_sampling", feature = "bom_sampling"))]
pub use qmc::{fill_noise_queries_gaussian_qmc_by_method, sample_k_states_qmc};
#[cfg(feature = "qmc_sampling")]
pub use qmc_halter::{
    QmcHaltDecision, QmcHaltReason, QmcHalter, count_hits_1d, iid_at_least_one, union_bound_ceiling,
};
