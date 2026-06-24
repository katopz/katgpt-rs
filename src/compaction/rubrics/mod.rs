//! Concrete [`Rubric`](super::Rubric) implementations — the latent-reframed
//! paper rubrics (Phase 3+) plus the cross-domain isomorphism rubrics (Phase 5).
//!
//! Each rubric in this module is a **pure, deterministic, modelless sigmoid
//! projection** over caller-supplied scalar features. The rubric never reaches
//! across crates to source its features — that decoupling is what keeps the
//! primitive testable, modelless, and sync-safe (per AGENTS.md: SOLID, DRY,
//! Modular, Generic, Decouple; and the modelless-first mandate).
//!
//! # Source-primitive map (Phase 3+, Research 300 §2.4)
//!
//! The rustdoc on each rubric documents which existing primitive the caller
//! *should* source each scalar from. The rubric itself is agnostic — feed it
//! synthetic features in tests, real features from any primitive in prod.
//!
//! | Paper predicate | Latent feature | Suggested source primitive |
//! |-----------------|----------------|----------------------------|
//! | C1 closed-unit  | coherence stability | `latent_functor/quality_gate` (riir-ai), or any coherence probe |
//! | C2 summarizable | intrinsic rank (inverted) | `katgpt-core::subspace_phase_gate::estimate_intrinsic_dim` |
//! | C3 progress     | divergence-since-last | `katgpt-core::dec` codifferential on a belief cochain |
//! | N1 stuck        | novelty rate (inverted) | `katgpt-core::cgsp::derivative_curiosity`, or ICT `collision_purity` |
//! | P0 input suff.  | n_wake_events ≥ intrinsic_dim | `katgpt-core::subspace_phase_gate::phase_transition_gate` |
//! | P1 output conv. | spectral flatness < τ | riir-neuron-db `ConsolidationPipeline::can_freeze` |

pub mod search;
