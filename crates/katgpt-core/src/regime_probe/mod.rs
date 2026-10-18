//! Regime probes for frozen predictors (Issue 740, Research 541).
//!
//! Distilled from *"Language Diffusion Models are Associative Memories
//! Capable of Retrieving Unseen Data"* (Pham, Zaki, Ambrogioni, Krotov,
//! Negri — EMNLP 2026; [arXiv:2604.26841](https://arxiv.org/abs/2604.26841)).
//! Every probe consumes only output categoricals a serving path already
//! produces (rows M1–M4 of Research 541 §2) and emits **raw scalars**
//! (entropy nats, a two-sample gap, a recovery rate, a corruption-tolerance
//! bound) — the sanctioned bridge direction: latent read → scalar out. No
//! probe output crosses a sync boundary; regime labels are local
//! instrumentation.
//!
//! # Module layout
//!
//! - [`entropy`]  — T1/M1: per-position conditional entropy of a categorical
//!   (max-shift + log-sum-exp via the shared `simd::logsumexp_parts` kernel
//!   factored from `breakeven/fidelity.rs::cross_entropy`).
//! - [`gap`]      — T2/M2: two-sample entropy-gap detector (reference corpus
//!   vs generated sequences), bit-deterministic, BLAKE3-artifact output.
//! - [`basin`]    — T3/M3: corrupt → renovate → recover basin probe (paper
//!   eq 12) over a caller-supplied frozen renovator (trait seam shaped like
//!   `ugc_schedule::UgcDenoiser`).
//! - [`gardner`]  — T4/M4: Gardner capacity curve `γ_c(κ)` from
//!   `1/γ_c = (1+κ²)Φ(κ) + κφ(κ)`, inverted once into a `OnceLock` LUT;
//!   `basin_radius_bound(γ)` via `κ > 2√ρ`.
//!
//! # Normalization honesty
//!
//! Categorical posteriors here are softmax-of-logits **where the input is a
//! score vector** (the probe's job is consuming serving-path categoricals).
//! Two-way posteriors are written as a single `sigmoid(score difference)` —
//! sigmoid, never a multi-way softmax gate (AGENTS.md §2). Where a predictor
//! emits log-probabilities, feeding them as logits is exact: `softmax(log p)`
//! recovers `p` (entropy is invariant to the constant normalizer).
//!
//! # Feature gate
//!
//! Gated `regime_probe` (opt-in). Promotion to default requires the GOAT
//! gates (G1 discriminative validity, G2 bound-holds, G3 bit-determinism,
//! G4 zero-alloc) **and** a later promotion decision — see
//! `.benchmarks/702_regime_probe_goat.md`.
//!
//! Issue 740 — Regime probes for frozen predictors (Research 541, arXiv:2604.26841
//! UDDMs-as-associative-memories): per-position conditional entropy (shared
//! logsumexp kernel with breakeven/fidelity), two-sample entropy-gap detector,
//! corrupt→renovate→recover basin probe (eq 12), and the Gardner capacity LUT
//! (1/γ_c = (1+κ²)Φ(κ) + κφ(κ), basin radius via κ > 2√ρ). Every probe consumes
//! only serving-path output categoricals and emits RAW scalars (entropy nats,
//! gap, ρ bound) — the sanctioned latent-read→scalar-out bridge direction. Zero
//! deps (blake3 already non-optional; RNG is the data_probe-markov fastrand
//! convention). GOAT-gated on the constructed memorizer/generalizer pair (G1),
//! the hebbian_kernel_memory bound-holds sweep (G2), bit-determinism (G3),
//! zero-alloc scratch protocol (G4): .benchmarks/702_regime_probe_goat.md.
//! Opt-in (no-default-consumer rule); promotion needs the GOAT gates AND a
//! later promotion decision.

pub mod basin;
pub mod entropy;
pub mod gap;
pub mod gardner;

pub use basin::{BasinReport, BasinScratch, FrozenRenovator, basin_probe, basin_probe_into};
pub use entropy::{conditional_entropies_into, conditional_entropy_nats, mean_conditional_entropy};
pub use gap::{EntropyGapReport, entropy_gap, entropy_gap_into};
pub use gardner::{
    KAPPA_GRID_POINTS, KAPPA_MAX, basin_radius_bound, basin_radius_from_kappa, gamma_capacity,
    kappa_max, kappa_max_bisection, phi_cdf, phi_pdf,
};
