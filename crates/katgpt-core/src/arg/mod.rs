//! ARG Protocol Primitives — open generic types distilled from the ARG Standard
//! (Iris Technologies, 2026; https://protocol.airistech.ai/arg-core.html).
//!
//! Plan 327 Phases 1-2, Research 309 — open half of the ARG × Latent Substrate
//! Super-GOAT fusion. Five generic protocol primitives (no game / chain /
//! shard semantics):
//!
//! - [`policy`] — `PolicyEnvelope`, `PolicyState`, `PolicyConstraints`,
//!   `ResponseMode`. Step 1 hard gate.
//! - [`taxonomy`] — `TaxonomyNode`, `TaxonomyValidator`, `LabelId`, `LabelSet`.
//!   Step 3 deterministic label-set validation producing `L_final`.
//! - [`lifecycle`] — `LifecycleState`, `RedirectTable`. Step E `ACTIVE →
//!   DEPRECATED → REMOVED` with redirect/alias preserving episodic-record
//!   interpretability under split/merge.
//! - [`candidate`] — `TypedOfflineCandidate`, `CandidateIntent`, `CandidateKind`,
//!   `EvidenceId`. Step C typed offline candidate (the structural delta).
//! - [`scorer`] — `OfflineCandidateScorer`, `Evidence`, `InfoOutcomeStatus`,
//!   `GainComponents`, `ScoredCandidate`. Step C scoring with the G5
//!   silence-bias penalty (`silence ≠ confirmed success`).
//!
//! Phase 3 will ship the fifth primitive (`InfoRegistry` with two-phase dedup).
//! Private runtime composition with HLA / Entity Cognition Stack / VMG /
//! Sub-Goal Compaction lives in `riir-ai/.plans/337_arg_runtime_wiring.md`.
//!
//! All primitives are pure types + validators. No LLM in the hot path. The
//! protocol permits LLM escalation (ARG OW-3.2 bounded proposer) — this crate
//! rejects it; the plasma → hot → warm → cold tier cascade in riir-ai is the
//! substitute.

pub mod candidate;
pub mod lifecycle;
pub mod policy;
pub mod scorer;
pub mod taxonomy;

pub use candidate::{CandidateIntent, CandidateKind, EvidenceId, TypedOfflineCandidate};
pub use lifecycle::{LifecycleState, RedirectTable};
pub use policy::{
    PolicyConstraints, PolicyDecision, PolicyEnvelope, PolicyState, ResponseMode, ShouldProceed,
};
pub use scorer::{
    DEFAULT_AUTO_COMMIT_THRESHOLD, Evidence, GainComponents, InfoOutcomeStatus,
    OfflineCandidateScorer, ScoredCandidate,
};
pub use taxonomy::{
    LabelId, LabelSet, TaxonomyKind, TaxonomyNode, TaxonomyValidator, ValidationError,
    ValidationResult, ValidationScratch,
};
