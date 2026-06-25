//! ARG Protocol Primitives — open generic types distilled from the ARG Standard
//! (Iris Technologies, 2026; https://protocol.airistech.ai/arg-core.html).
//!
//! Plan 327 Phases 1-3, Research 309 — open half of the ARG × Latent Substrate
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
//! - [`registry`] — `InfoRegistry`, `InfoKey`, `InfoUnit`, `MatchResult`,
//!   `CompareFn`. Step 9 + Step C two-phase dedup (primary key index +
//!   secondary payload-hash collision index) with grey-zone review.
//!
//! All five primitives shipped. Private runtime composition with HLA / Entity
//! Cognition Stack / VMG / Sub-Goal Compaction lives in
//! `riir-ai/.plans/337_arg_runtime_wiring.md`.
//!
//! All primitives are pure types + validators. No LLM in the hot path. The
//! protocol permits LLM escalation (ARG OW-3.2 bounded proposer) — this crate
//! rejects it; the plasma → hot → warm → cold tier cascade in riir-ai is the
//! substitute.

pub mod candidate;
pub mod lifecycle;
pub mod policy;
pub mod registry;
pub mod scorer;
pub mod taxonomy;

pub use candidate::{CandidateIntent, CandidateKind, EvidenceId, TypedOfflineCandidate};
pub use lifecycle::{LifecycleState, RedirectTable};
pub use policy::{
    PolicyConstraints, PolicyDecision, PolicyEnvelope, PolicyState, ResponseMode, ShouldProceed,
};
pub use registry::{
    AccessScope, CompareFn, CompareResult, InfoKey, InfoRegistry, InfoType, InfoUnit,
    LabelSignature, MatchResult, MatchScratch, PayloadHash, PayloadHashCompare, Provenance,
};
pub use scorer::{
    DEFAULT_AUTO_COMMIT_THRESHOLD, Evidence, GainComponents, InfoOutcomeStatus,
    OfflineCandidateScorer, ScoredCandidate,
};
pub use taxonomy::{
    LabelId, LabelSet, TaxonomyKind, TaxonomyNode, TaxonomyValidator, ValidationError,
    ValidationResult, ValidationScratch,
};
