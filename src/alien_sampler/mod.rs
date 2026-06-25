//! # Alien Sampler Primitive — Coherence × Availability Frontier Ranking
//!
//! **Open primitive** distilled from arXiv:2603.01092 (Artiles et al., "The
//! Alien Space of Science", May 2026). Generic, modelless within-pool ranking
//! that fuses a **coherence** score with an **unavailability** score via
//! within-pool z-scoring, then ranks candidates by `Fβ = (1−β)·zC + β·zU`.
//!
//! Game-side wiring (NPC population banks, CGSP Conjecturer binding, zone
//! emission feeds) lives in `riir-ai` Plan 312+ — this crate stays
//! **math-only, MIT, no game IP**.
//!
//! ## Source paper
//! [arxiv 2603.01092](https://arxiv.org/abs/2603.01092) — Artiles et al., "The
//! Alien Space of Science" (May 2026). Distills the paper's coherence ×
//! (1−availability) frontier search into a generic sampler.
//!
//! ## Two-axis decomposition
//!
//! The paper's core insight is that candidate quality in a population-diversity
//! setting decomposes into two near-independent axes:
//!
//! 1. **Coherence** — how internally consistent / on-personality / high-guide
//!    a candidate is. Caller-supplied via [`traits::CoherenceScorer`].
//! 2. **Availability** — how represented the candidate is in the reference
//!    community's repertoire. Caller-supplied via
//!    [`traits::AvailabilityScorer`]; the sampler negates it to produce
//!    "alien-ness" (low availability = alien).
//!
//! The two axes are fused via within-pool z-scoring (population std) and a
//! single linear combination `Fβ = (1−β)·zC + β·zU`. β=0 is coherence-only
//! (motif collapse); β=1 is availability-only (random alien); β=0.7 (paper
//! default) is the empirically-validated frontier.
//!
//! ## Load-bearing rule: median-of-top-m
//!
//! The paper's critical ablation (§1.4) shows that the **median over top-m
//! cosine retrievals** is load-bearing — substituting a density estimator
//! (kernel density, normalized softmax over similarities) fails the
//! motif-collapse test. [`median_top_m::MedianTopMAvailability`] implements
//! this exact rule: for each candidate, compute cosine similarity against a
//! precomputed [`median_top_m::MedianTopMAvailability::bank`] of community
//! embeddings, take the top-m via `select_nth_unstable` (O(n) partial sort),
//! and return the median. m=10 (paper default).
//!
//! ## Zero-allocation hot path
//!
//! The sampler is designed for per-tick / per-NPC / per-cycle use:
//! - [`sampler::AlienSampler::rank_into`] takes caller-owned scratch buffers
//!   and output buffer — no allocation inside the scoring loop.
//! - [`median_top_m::MedianTopMAvailability::availability_embedded_with_scratch`]
//!   takes a caller-owned cosine scratch buffer — no allocation per candidate.
//! - The trait-compatible entry points ([`sampler::AlienSampler::rank`] and
//!   `AvailabilityScorer::availability`) allocate once for the output / cosine
//!   scratch — convenient for cold paths and tests.
//!
//! ## Determinism
//!
//! Bit-identical output for bit-identical input. No RNG, no thread-local
//! state, no clock. The sort is a stable sort on `total_cmp` of f32 scores;
//! ties break by input index (ascending). Required for replay / sync / audit
//! per AGENTS.md.
//!
//! ## GOAT gates (Plan 311 Phase 3)
//!
//! The primitive is opt-in behind the `alien_sampler` feature until the Phase
//! 3 GOAT gate passes on a synthetic motif-collapse scenario:
//! - **G1 (motif collapse):** top-10 direction concentration across a 100-NPC
//!   zone. Treatment (AlienSampler β=0.7) must be ≤ 50% of OPUS-style scalar
//!   local-redundancy baseline's concentration. Paper analog: 95.7%→34.3%.
//! - **G2 (quality preservation):** mean coherence of selected directions.
//!   Treatment must be ≥ 90% of coherence-only baseline.
//! - **G3 (perf):** per-cycle wall time ≤ 5× scalar-redundancy baseline.
//! - **G4 (latent boundary):** no `Vec<f32>` escapes the `rank()` call
//!   boundary in the public API (the open primitive has no sync concept).
//!
//! See `.benchmarks/311_alien_sampler_goat.md` for the gate results.
//!
//! ## API surface
//!
//! - [`sampler::AlienSampler`]`<V, C, A>` — the sampler itself.
//! - [`traits::CoherenceScorer`] / [`traits::AvailabilityScorer`] — the two
//!   axis traits (caller-implemented).
//! - [`median_top_m::MedianTopMAvailability`] — the paper's load-bearing
//!   median-of-top-m cosine availability scorer (ships `impl
//!   AvailabilityScorer<f32>`).
//! - [`types::AlienConfig`] / [`types::ScoredCandidate`] / [`types::AlienSamplerError`]
//!   — config + output + error types.
//!
//! ## Private boundary (NOT in this crate)
//!
//! - NPC wiring (zone banks populated from NPC behavior emissions) → `riir-ai`
//!   Plan 312.
//! - Latent/raw sync boundary enforcement → `riir-ai` runtime (this crate has
//!   no sync concept).
//! - Autoregressive coherence transformer training → `riir-train`.
//! - LatCal commitment of Fβ scores for chain provenance → speculative, not
//!   needed until a chain consumer asks.
//!
//! ## References
//!
//! - Plan: `katgpt-rs/.plans/311_alien_sampler_primitive.md`
//! - Research: `katgpt-rs/.research/293_Alien_Science_Coherence_Availability_Frontier.md`
//! - Cousin (scalar-redundancy baseline): `katgpt-rs/.research/089_OPUS_*.md`
//!   (OPUS local-redundancy penalty — what this primitive's GOAT gate beats)
//! - Cousin (sigmoid-gated atom composition):
//!   `katgpt-rs/.research/276_Personality_Weighted_Latent_Layer_Composition.md`
//! - Cousin (per-tick decision primitive):
//!   `katgpt-rs/.research/281_Per_Tick_Salience_Tri_Gate_Speak_Silent_Delegate.md`

pub mod median_top_m;
pub mod sampler;
pub mod traits;
pub mod types;

pub use median_top_m::MedianTopMAvailability;
pub use sampler::AlienSampler;
pub use traits::{AvailabilityScorer, CoherenceScorer};
pub use types::{AlienConfig, AlienSamplerError, ScoredCandidate};

#[cfg(test)]
mod integration_tests;
