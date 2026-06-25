//! ARG Protocol Primitives — open generic types distilled from the ARG Standard
//! (Iris Technologies, 2026; https://protocol.airistech.ai/arg-core.html).
//!
//! Plan 327 Phase 1, Research 309 — open half of the ARG × Latent Substrate
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
//!
//! Phase 2/3 ships the remaining two (`TypedOfflineCandidate` + `InfoRegistry`).
//! Private runtime composition with HLA / Entity Cognition Stack / VMG /
//! Sub-Goal Compaction lives in `riir-ai/.plans/337_arg_runtime_wiring.md`.
//!
//! All primitives are pure types + validators. No LLM in the hot path. The
//! protocol permits LLM escalation (ARG OW-3.2 bounded proposer) — this crate
//! rejects it; the plasma → hot → warm → cold tier cascade in riir-ai is the
//! substitute.

pub mod lifecycle;
pub mod policy;
pub mod taxonomy;

pub use lifecycle::{LifecycleState, RedirectTable};
pub use policy::{
    PolicyConstraints, PolicyDecision, PolicyEnvelope, PolicyState, ResponseMode, ShouldProceed,
};
pub use taxonomy::{
    LabelId, LabelSet, TaxonomyKind, TaxonomyNode, TaxonomyValidator, ValidationError,
    ValidationResult, ValidationScratch,
};
