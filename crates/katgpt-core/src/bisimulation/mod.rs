//! `bisimulation` — generic modelless primitive for quotienting observed
//! transition graphs into bisimulation-equivalent state classes and inferring
//! abstract operator schemas.
//!
//! # Plan 324 / Research 308 / arXiv:2602.19260
//!
//! Source paper: Duggan, Lorang, Lu, Scheutz, "The Price Is Not Right" (Tufts,
//! Feb 2026, [arXiv:2602.19260](https://arxiv.org/pdf/2602.19260)). The paper
//! compares a fine-tuned VLA (π0 / OpenPi) against a neuro-symbolic
//! manipulation (NSM) pipeline on Towers of Hanoi and finds NSM dominates on
//! structured long-horizon tasks: 95% vs 34% (3-block), 78% vs 0% (unseen
//! 4-block), with ~80× less training energy and ~10× less inference energy.
//!
//! This module ships the **deterministic half** of the NSM pipeline:
//!
//! 1. [`TransitionGraph`] — observed `(state, op, state')` triples, sorted
//!    and indexed for O(log) adjacency.
//! 2. [`BisimulationQuotient`] — partition of states into equivalence classes
//!    via Paige-Tarjan partition refinement (O((S+E) log S)).
//! 3. [`OperatorSchema`] —
//!    one abstract operator per edge-label, with preconditions (src classes)
//!    and effects ((src, dst) class pairs).
//! 4. A BFS [`planner`](crate::bisimulation::planner::plan) over the quotient
//!    (sufficient for G3 plan validity; MetricFF-grade planning is out of
//!    scope).
//!
//! What this primitive **does NOT** ship (out of scope per Plan 324):
//!
//! - Diffusion-policy skill training (→ riir-train).
//! - LLM-based symbolic abstraction / ASP solver (the heavier-weight CWM
//!   path; see [`crate::induced_cwm`]).
//! - LatCal chain-commitment wiring (the `blake3` field is the bridge
//!   artifact; actually wiring it into a chain block is riir-chain's job).
//! - Real-world robot / game integration (engine primitive only).
//!
//! # Relationship to Induced CWM (Plan 296, Research 275)
//!
//! [`crate::induced_cwm`] is the **closest cousin** and ships the broader
//! capability class ("system observes structured task, induces verifiable
//! rules, plans via search") as a Super-GOAT. CWM induces **executable
//! code** via an LLM refinement loop; this primitive induces an **operator
//! schema** via a deterministic graph algorithm.
//!
//! | Aspect | Induced CWM (Plan 296) | This primitive (Plan 324) |
//! |--------|------------------------|---------------------------|
//! | Induction engine | LLM (offline) | Deterministic (Paige-Tarjan) |
//! | Output | Python/`GameState` impl | PDDL-like `OperatorSchema` |
//! | Plan-time compute | MCTS/ISMCTS over induced code | BFS over quotient |
//! | Cold-tier cost | LLM call | None (pure graph algorithm) |
//! | Hot-tier cost | `advance()` (per-tick) | `class_of(state)` (O(1) lookup) |
//!
//! The runtime picks per task: code induction for rich domains, operator
//! induction for structured/combinatorial domains. They are complementary,
//! not redundant.
//!
//! # Closes Research 264 §2.2 gaps
//!
//! Research 264 (Closure-Expansion Instrument) flagged two unshipped gaps:
//!
//! 1. **PTG data structure** (Primitive Transition Graph as explicit runtime
//!    data structure) — ✅ closed by [`TransitionGraph`].
//! 2. **Motif mining loop** (recurring sub-path consolidation) — ✅ closed by
//!    [`BisimulationQuotient`] (the quotient collapses recurring motifs into
//!    a single class; operator inference lifts them to abstract operators).
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! | Quantity | Space | Synced? |
//! |----------|-------|---------|
//! | `TransitionGraph` edges | **Raw** | YES (deterministic observed facts) |
//! | `BisimulationQuotient.state_to_class` | **Raw** | YES (deterministic) |
//! | `BisimulationQuotient.blake3` | **Raw** | YES (chain-committable) |
//! | `OperatorSchema.blake3` | **Raw** | YES (chain-committable) |
//! | Class-id semantics (what a class "means") | **Latent** | NO (consumer's job) |
//! | `class_of(state)` hot-path lookup | **Raw** | YES (O(1) index, audit-frequent) |
//!
//! The bisimulation algorithm is **purely raw-side**: it operates on observed
//! transition triples and produces deterministic quotient partitions. The
//! *interpretation* of a class (e.g., "this class represents the '2-disk
//! Hanoi state with largest disk on peg 3'") is a latent-space mapping the
//! consumer maintains separately — this primitive never sees it.
//!
//! # Feature gate
//!
//! Gated behind the `bisimulation_operator_inference` Cargo feature
//! (default-off). The primitive is opt-in by design: downstream pipelines
//! (riir-ai NPC runtime, riir-chain LatCal consumer, etc.) opt in by
//! enabling the feature. Promotion to default-on is **not planned** — this
//! is a primitive, not a default-on capability (same policy as Induced CWM,
//! Plan 296).
//!
//! # References
//!
//! - Plan: `katgpt-rs/.plans/324_bisimulation_operator_inference.md`
//! - Research: `katgpt-rs/.research/308_NSM_VLA_Price_Is_Not_Right_Bisimulation_Operator_Inference.md`
//! - Source paper: [arXiv:2602.19260](https://arxiv.org/pdf/2602.19260)
//! - Underlying NSM method: [arXiv:2508.21501](https://arxiv.org/abs/2508.21501)
//! - Closest cousin: [`crate::induced_cwm`] (Plan 296, Research 275)
//! - Gap closed: `katgpt-rs/.research/264_Compositional_Open_Ended_Intelligence_Framework.md` §2.2
//!
//! ── Bisimulation Operator Inference (Plan 324, Research 308, arxiv 2602.19260) ─
//!
//! Open primitive: quotient an observed transition graph into bisimulation-
//! equivalent state classes (signature-based partition refinement, O((S+E)
//! log² S log d)) and infer an abstract PDDL-like operator schema. The
//! lighter-weight PDDL-side counterpart to Induced CWM (Plan 296): where CWM
//! induces executable *code* via an LLM, this induces an *operator schema* via
//! a deterministic graph algorithm. Closes Research 264 §2.2 gaps #1 (PTG) +
//! #2 (motif mining). Opt-in by design — downstream pipelines (riir-ai NPC
//! runtime, riir-chain LatCal consumer) opt in by enabling the feature.
//!
//! Issue 586 — rule-application consistency metric (BDH-CQ §6.4 analog):
//! 3-bin strict/partial/none task histogram + sigmoid-guarded gap +
//! structure-preservation breakdown + complexity-cluster regime detection.
//! Gates `infer_operators` output via `promotion_verdict`; the
//! ComplexityClustered regime is the exemplar-seeking trigger consumed by
//! riir-ai Issue 672.

#[cfg(feature = "operator_consistency")]
pub mod consistency;
pub mod graph;
pub mod operator;
pub mod planner;
pub mod refine;
pub mod types;

// ── Public API re-exports ─────────────────────────────────────────────────
//
// Mirrors the idiom used by other katgpt-core feature modules (e.g.
// `induced_cwm`, `micro_belief`): `pub use` the most common types at the
// module root so callers can write
// `katgpt_core::bisimulation::BisimulationQuotient` instead of
// `katgpt_core::bisimulation::refine::BisimulationQuotient`.

#[cfg(feature = "operator_consistency")]
pub use consistency::{
    ApplicationOutcome, ConsistencyGateConfig, ConsistencyRegime, ConsistencyReport,
    PromotionVerdict, promotion_verdict, rule_consistency,
};
pub use graph::{TransitionGraph, TransitionGraphBuilder};
pub use operator::{OperatorDef, OperatorSchema, infer_operators};
pub use planner::{Plan, plan};
pub use refine::{BisimulationQuotient, partition_refine};
pub use types::{OperatorLabel, QuotientEdge, StateClassId, StateId, Transition};

// ── Quotient definition lives here (re-exported from refine) ───────────────
//
// `BisimulationQuotient` is the central type of this module. It's defined
// in `refine.rs` because the field layout is tightly coupled to the
// Paige-Tarjan algorithm's output; see `refine::BisimulationQuotient` for
// the per-field documentation.
