//! **Causal-ID — Algorithmic Syntactic Causal Identification**
//! (Plan 457, Research 450, arXiv:2403.09580 Cakiqi & Little 2024).
//!
//! A modelless primitive for identifying interventional signatures on
//! Acyclic Directed Mixed Graphs (ADMGs) with unobserved confounders.
//! Distilled from the Cakiqi-Little Theorem 1 syntactic identification
//! algorithm — pure graph rewriting, no gradient descent.
//!
//! ## What this ships
//!
//! - [`Admg`] / [`NodeId`] / [`AdmgSignature`] — the substrate types
//!   ([`types`] module).
//! - [`identify`] — the top-level driver implementing the recursive
//!   Shpitser-Pearl ID algorithm (six steps: marginalise, restrict,
//!   drop, district, fix, recurse).
//! - [`fixing`] — graph-rewriting primitives ([`Admg::districts`],
//!   [`Admg::ancestors`], [`Admg::fix_node`], [`fixing::try_fixseq`]).
//! - [`extract_relevant_subgraph`] — bounded-BFS subgraph extractor
//!   (caveat #2 mitigation: keep the `O(k²)`–`O(k³)` algorithm on a
//!   ≤32-node subgraph).
//!
//! ## Why this is in katgpt-core
//!
//! The identification algorithm is pure graph rewriting on structural
//! types — no game-specific vocabulary, no game-state dependency. It is
//! the modelless half of the counterfactual reasoning story; the
//! ADMG-from-KgTriple construction + GM tool wiring live in
//! `riir-ai/crates/riir-engine/src/causal_id/` (Plan 457 Phase 3+).
//!
//! ## Provenance (the soundness correction)
//!
//! The Issue 545 PoC caught a soundness bug in the original one-pass
//! formulation: it computed districts of `G[Y⋆]` instead of `G[V]`,
//! returning `NotIdentifiable` for the classic front-door case. The
//! corrected recursive formulation (districts of `G[V]`, hedge FAIL
//! condition) is what ships here. See [`identify`] module docs for the
//! six-step algorithm.
//!
//! ## Why modelless
//!
//! Every primitive here is pure graph rewriting. No backprop, no learned
//! parameters, no gradient descent. The ADMG is constructed downstream
//! (Plan 457 Phase 3, riir-ai) from a `KgTriple` corpus + a confounder
//! injection layer; this crate only owns the identification math.
//!
//! ## Status
//!
//! DEFAULT-ON (Plan 457 Phase 5 promotion, 2026-07-18). Plan 457 Phase 1 —
//! implementation complete. Phase 2 GOAT gate G1+G2+G3 PASS; G4 was initially
//! DEFERRED with offline-only rationale (8.40µs/32-node identify is well
//! outside the 20Hz tick), then closed 2026-07-18 as an INFORMATIONAL PASS
//! (Issue 183 / Benchmark 465: Scratch refactor cut per-call allocs 284→198
//! (−30%) and latency 8.26→6.07 µs (−27%) on the 32-node scenario; remaining
//! graph-construction allocs out of G4 scope). Phase 4 T4.5 synthetic Consumer
//! A bench (`.benchmarks/464_causal_id_consumer_a_synthetic.md`) cleared the T4.7
//! promotion gate: 71.7% non-trivial Ok rate (43/60 queries) on a 100-node
//! game-world KG with 3 faction confounder cliques, 43 actionable
//! interventional signatures S1 (Canvas FlowGraph reachability) cannot
//! derive. Consumer B T4.6 (sleep-cycle trace) remains BLOCKED on real-trace
//! capture + new counterfactual-claim infrastructure (T4.3); per Plan 457
//! §T4.7 the promotion criterion is Consumer A OR Consumer B, so Consumer A
//! passing alone is sufficient.
//!
//! The Issue 545 PoC proved the algorithm strictly dominates Canvas
//! FlowGraph reachability on a 13-node game KG with a `NPC1 ↔ NPC2`
//! confounder (S2 produces a 5-node interventional signature that correctly
//! excludes NPC1; S1 yields only a boolean `reaches=true` and cannot see
//! the confounder).
//!
//! Causal-ID — Algorithmic Syntactic Causal Identification (Plan 457,
//! Research 450, arXiv:2403.09580 Cakiqi & Little 2024). Pure modelless
//! graph rewriting on Acyclic Directed Mixed Graphs (ADMGs) with bidirected
//! confounders: `identify(Y, do(A))` returns the interventional signature
//! backbone `Y⋆ = An(Y in G[V\A])` via the recursive Shpitser-Pearl ID
//! algorithm (Cakiqi-Little Theorem 1 distillation). The Issue 545
//! defend-wrong PoC proved S2 strictly dominates Canvas FlowGraph
//! reachability on a 13-node game KG with a `NPC1 ↔ NPC2` confounder: S2
//! produces a 5-node interventional signature that correctly excludes NPC1
//! (the confounder neighbor); S1 yields only a boolean reaches=true and
//! cannot see the confounder. Sibling of `causal_head_importance` (activation
//! patching) + `canvas` (declared directed-only topology) — causal_id adds
//! the bidirected-confounder dimension they cannot see. Pure modelless
//! (graph rewriting on BLAKE3-committed NodeId set). Offline-only (24µs on
//! 13 nodes is well outside the 20Hz tick; subgraph extraction mandatory to
//! stay ≤32 nodes per query). DEFAULT-ON since 2026-07-18 (Plan 457 Phase 5
//! promotion): Plan 457 Phase 2 GOAT G1+G2+G3+G4 ALL PASS; §T4.7 promotion
//! gate PASS on Consumer A (synthetic 100-node game-world KG, OR criterion —
//! Consumer B T4.6 sleep-cycle remains BLOCKED on real-trace capture but does
//! not block promotion).

pub mod fixing;
pub mod identify;
pub mod subgraph;
pub mod types;

pub use fixing::try_fixseq;
pub use identify::identify;
pub use subgraph::extract_relevant_subgraph;
pub use types::{Admg, AdmgSignature, INLINE_SIGNATURE_CAP, IdentificationError, NodeId};
