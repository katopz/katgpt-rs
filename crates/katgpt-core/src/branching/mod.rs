//! Non-Interference Memory Branches — continual-adaptation primitive
//! distilled from RIZZ (Goel et al., Oxford, Jun 2026, arXiv:2606.20638).
//!
//! Plan 329 / Research 310 — Super-GOAT fusion of BAKE × CLR × MCGS × Engram
//! × ARG × closure-instrument × Salience. Five generic open primitives (no
//! game / chain / shard semantics):
//!
//! - [`types`] — `BranchId`, `CognitiveBranch`, `EpisodicEntry`, `ProceduralRule`,
//!   `FailureEntry`, `BranchStats`, `BranchLifecycle`.
//! - [`bank`] — `BranchBank`: bounded bank of persistent cognitive branches with
//!   spawn / merge / prune lifecycle. Free-list slot reuse; pre-allocated
//!   capacity.
//! - [`router`] — `BranchRouter`: dot-product snap routing with optional
//!   Jaccard token-overlap fallback. Zero-alloc hot path.
//! - [`verifier`] — `VerifierGate`: reward + curiosity + branch-centroid
//!   quarantine write gate. Composes with CLR `should_write_memory(r_k, S_LP)`.
//! - [`projection`] — `NonInterferenceProjection`: orthogonal latent subspaces
//!   per branch (D-dim direction vectors; interference = |dot(dir_i, dir_j)|).
//! - [`compiler`] — `BudgetCompiler`: priority-cascade context compiler under
//!   fixed byte budget (scope_ctx → procedural → episodic → cross-positive →
//!   failures → working_memory → query).
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! | Quantity | Space | Synced? |
//! |----------|-------|---------|
//! | `BranchBank` slot array + free-list | **Raw** | YES (deterministic structure) |
//! | `BranchId`, `BranchLifecycle`, `BranchStats` | **Raw** | YES |
//! | `spawn_anchor`, `EpisodicEntry.embedding` | **Latent** | NO (projection vectors) |
//! | `token_signature` | **Raw** | YES (deterministic hashes) |
//! | `EpisodicEntry.payload` | **Caller-defined** | Caller decides |
//!
//! Nothing in this module crosses the `SyncBlock → ChainConsensus` boundary
//! by default. The caller decides what to sync.
//!
//! # Feature gate
//!
//! Entire module is `#[cfg(feature = "non_interference_branches")]`. Zero cost
//! when disabled — every public API vanishes from the build. **DEFAULT-ON**
//! since Plan 329 Phase 3 (2026-06-26) — G1–G5 GOAT gate ALL PASS (8 orthogonal
//! directions in D=8, route 301.5ns<1µs, 0 allocs/100 calls, [] deps, 101/101
//! tests). The feature is in `default = [...]` of both katgpt-core and katgpt-rs
//! root. See `.benchmarks/329_*` for the GOAT gate evidence.
//!
//! # Composition
//!
//! When the `arg_protocol` feature is also enabled, `BranchLifecycle` is a
//! type alias for `crate::arg::LifecycleState` (the same enum used by the ARG
//! protocol's ontology lifecycle). This makes branch lifecycle state
//! committable and redirect-resolvable via the ARG `RedirectTable`.
//!
//! # References
//!
//! - Plan: `katgpt-rs/.plans/329_non_interference_memory_branches.md`
//! - Research: `katgpt-rs/.research/310_RIZZ_Non_Interference_Memory_Branches.md`
//! - Source paper: [arXiv:2606.20638](https://arxiv.org/abs/2606.20638)
//! - Private guide: `riir-ai/.research/161_Per_NPC_Cognitive_Branch_Continual_Adaptation_Guide.md`
//! - Fusion cousins: Plan 236 (BAKE), Plan 284/316 (CLR), progressive_mcgs/,
//!   Plan 299 (Engram), Plan 327 (ARG), Plan 290 (closure-instrument),
//!   Plan 303 (Salience)
//!
//! Non-Interference Memory Branches — Super-GOAT fusion (Plan 329, Research 310,
//! arXiv:2606.20638 Goel et al. Oxford Jun 2026). Five generic open primitives:
//! BranchBank (bounded persistent CognitiveBranch bank with spawn/merge/prune
//! lifecycle), BranchRouter (dot-product snap + Jaccard fallback), VerifierGate
//! (reward + curiosity + centroid-quarantine write gate, composes with CLR),
//! NonInterferenceProjection (orthogonal latent subspace per branch),
//! BudgetCompiler (priority-cascade context compiler under fixed budget). Fuses
//! BAKE × CLR × MCGS × Engram × ARG × closure-instrument × Salience into a new
//! capability class: per-NPC continual adaptation without catastrophic
//! forgetting. Composes with arg_protocol LifecycleState when both features on.
//! DEFAULT-ON (Plan 329 Phase 3, 2026-06-26): G1–G5 ALL PASS.

pub mod bank;
pub mod compiler;
#[cfg(test)]
mod composition;
pub mod projection;
pub mod router;
pub mod types;
pub(crate) mod util;
pub mod verifier;

// ── Public API re-exports ─────────────────────────────────────────────────
//
// Mirrors the idiom used by other katgpt-core feature modules (`arg`,
// `bisimulation`, `closure`): `pub use` the most common types at the module
// root so callers can write
// `katgpt_core::branching::BranchBank` instead of
// `katgpt_core::branching::bank::BranchBank`.

pub use bank::{BranchBank, DEFAULT_MAX_BRANCHES};
pub use compiler::{
    BudgetCompiler, CompiledContext, CompiledItem, DEFAULT_BUDGET_BYTES, PriorityTier,
    RetrievedMaterials,
};
pub use projection::{
    AssignError, AssignResult, DEFAULT_ASSIGN_MAX_INTERFERENCE, DEFAULT_ORTHOGONAL_EPSILON,
    DEFAULT_PROJECTION_DIM, NonInterferenceProjection, max_orthogonal_branches,
};
pub use router::{BranchRouter, DEFAULT_TAU_JACCARD, DEFAULT_TAU_SNAP, RouteMode, RouteResult};
pub use types::{
    BRANCH_TYPES_WIRE_VERSION, BranchId, BranchLifecycle, BranchStats, CognitiveBranch,
    EpisodicCodec, EpisodicEntry, FailureEntry, ProceduralRule,
};
pub use verifier::{
    DEFAULT_QUARANTINE_CENTROID_THRESH, DEFAULT_TAU_CURIOSITY, DEFAULT_TAU_WRITE, VerifierGate,
    WriteDecision,
};
