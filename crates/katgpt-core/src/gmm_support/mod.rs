//! `gmm_support` — the density-ratio support gate primitive (Plan 618 /
//! Research 604, arXiv:2610.02126 "Local Support Learning").
//!
//! The continuous-latent two-density member of the shipped gate family:
//! a diagonal-GMM pair (positive fit on the suite/adapter's own training
//! distribution, negative on a generic reference corpus capturing
//! distribution WIDTH, not identity) whose log-ratio `ℓ(x) = log Φ_pos(x)
//! − log Φ_neg(x)` gates adapter/answer admission to the training
//! support. Modelless throughout: EM statistics, a seeded JL projection,
//! closed-form `O(K·E)` eval, an EMA scalar, and an offline-certifiable
//! excess bound — no gradient descent anywhere.
//!
//! # Layout
//!
//! | Module | Role |
//! |---|---|
//! | [`mixture`] | [`DiagGmm`] — the mixture-density extension of the `RegionSubspaceField` class (per-component variances, logsumexp density, BLAKE3 commitment) |
//! | [`em`] | [`fit_diag_gmm`] — EM refinement seeded from the shipped deterministic k-means |
//! | [`projector`] | [`JlProjector`] — the revived JL projection at E ≥ 32, packed sign-bit storage |
//! | [`gate`] | [`SupportGate`] + [`GateSmoother`] — the log-ratio gate and its EMA |
//! | [`certify`] | the T6/T7 excess/deficit certification instruments + the leak-by-construction canary |
//! | [`fixture`] | deterministic corpus fixtures + the CLT-12 sampler |
//!
//! # Fit-space law (measured at the first consumer)
//!
//! Fit/eval in the JL-PROJECTED space (E ≈ 32–64), never raw D-space —
//! raw hashed-bag space rejects Gaussianity universally (0/154 label
//! pools; riir-reflex Issue 066 PRE-CHECK). The projection is part of
//! the gate's frozen definition.
//!
//! # Sigmoid, not softmax
//!
//! The gate's confidence bridge is `sigmoid(ℓ/τ)` — a two-density ratio
//! is one scalar, not a categorical. The mixture's internal logsumexp is
//! the defining normalization of a mixture (the
//! `distributional_steering` caveat-4 precedent), not a semantic
//! softmax.
//!
//! # Consumers (wiring in their own repos)
//!
//! riir-reflex Issue 066 (the fused-abstain density half), the
//! riir-refine `contrastive_scope` continuous-latent sibling (guide
//! row), riir-engine gated `LoRAHotSwap` dispatch (guide row,
//! `riir-ai/.research/394`). Opt-in (`gmm_support`) per the
//! no-default-consumer rule — promotion rides a consumer's GOAT.
//!
//! Density-ratio support gate (Plan 618 / Research 604, arXiv:2610.02126
//! "Local Support Learning") — the continuous-latent two-density member
//! of the gate family: a diagonal-GMM pair (positive fit on the
//! suite/adapter's own training distribution, negative on a generic
//! reference corpus) whose log-ratio gates admission to the training
//! support, with the revived JL projector at E ≥ 32 (packed sign-bit
//! storage), EM refinement seeded from the shipped k-means, an EMA
//! smoother, and the offline excess/deficit certification instruments
//! (the leak-by-construction canary included). Fit in the PROJECTED
//! space, never raw (the measured fit-space law). Opt-in
//! (`gmm_support`) per the no-default-consumer rule — promotion rides a
//! consumer's GOAT (riir-reflex Issue 066 is the first).

pub mod certify;
pub mod em;
pub mod fixture;
pub mod gate;
pub mod mixture;
pub mod projector;

pub use certify::{
    CertificationReport, CertifyConfig, CertifyError, LinearDiscriminatorGate, SupportDecision,
    bound_acceptance, canary_fire_rate, certify,
};
pub use em::{EmConfig, EmError, fit_diag_gmm};
pub use fixture::{FixtureSet, build_fixture};
pub use gate::{DEFAULT_TAU, GateError, GateSmoother, SupportGate};
pub use mixture::{DiagGmm, MixtureError, compute_mixture_commitment};
pub use projector::JlProjector;

#[cfg(test)]
mod tests;
