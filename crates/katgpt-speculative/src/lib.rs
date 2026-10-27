//! katgpt-speculative — shared speculative decoding + DDTree substrate.
//!
//! Issue 013 (2026-06-29): collapses the fork between `katgpt-rs/src/speculative/`
//! and `riir-engine/src/{dd_tree,dflash}.rs`. The core DDTree algorithm lives
//! here so improvements propagate to both consumers.
//!
//! # What lives here
//!
//! - **DDTree core** (`dd_tree`): `build_dd_tree`, `build_dd_tree_pruned`,
//!   `build_dd_tree_screened`, `build_dd_tree_balanced`, `TreeBuilder`,
//!   `extract_parent_tokens`, `extract_best_path`, `merge_retrieved_branches`,
//!   `find_valid_sequence`, etc. Pure algorithm over pre-computed marginals.
//!   No `forward` dependency.
//!
//! # What does NOT live here
//!
//! - **Types** (`TreeNode`, `DraftResult`, `ConstraintPruner`, …) → already in
//!   `katgpt_core::speculative::types` + `katgpt_core::traits` (Plan 008 Phase 2.5).
//! - **Sampling** (`sample_from_distribution`) → already in
//!   `katgpt_core::speculative::sampling` (Plan 008 Phase 2.6).
//! - **DFlash** (`dflash_predict_*_with`) → the three zero-alloc `_with`
//!   cores live here (Issue 013 Phase B). They are generic over a `DflashCtx` +
//!   `DflashCache` backend trait pair and a `forward_fn` closure, because the
//!   underlying `ForwardContext` / `MultiLayerKVCache` / `TransformerWeights`
//!   types are crate-specific. The thin wrappers (`dflash_predict`, `_ar`,
//!   `_conditioned`, `_parallel`) and feature-gated variants (`_domino`,
//!   `_routing`, `_fusion`) stay in each consumer.
//! - **Feature-gated DDTree variants** (`build_dd_tree_belief`, `_speculative`,
//!   `_kurtosis`, `_domino`, `_manifold`, `_lodestar`, `_gdsd`, …) → stay in
//!   `katgpt-rs/src/speculative/dd_tree.rs` because they reference root-only
//!   sibling modules (`super::belief_drafter`, `super::spec_generator`, etc.).
//! - **Bigram Markov head** (`bigram_markov` feature, Issue 659) → the
//!   modelless sequential drafter: deterministic CSR top-m bigram table +
//!   zero-alloc greedy-chain marginal emission, dropping straight into
//!   `build_dd_tree`. The trained-MLP sibling is `belief_drafter`.

pub mod blueprint;
pub mod branch_confidence;
pub mod correlation_budget;
pub mod dd_tree;
pub mod decomp_reviewer;
pub mod dflash;
// Issue 917 T3 — EB-Sampler-class entropy-bounded lane wiring: the DDTree
// expansion variant whose per-expansion child count is the EB prefix and the
// DFlash block commit that picks the drafted block's committed prefix by the
// same Σ H − max H ≤ γ bound. Consumes katgpt_core::entropy_bounded_commit
// (T1). Additive — the legacy builders and dflash cores are untouched.
#[cfg(feature = "entropy_bounded_commit")]
pub mod entropy_bounded;
pub mod nf_flow;
pub mod nf_flow_budget;
pub mod nf_flow_fold;
pub mod nf_flow_gate;
pub mod nf_flow_mux;
pub mod pathway_tracker;
pub mod prefix_scheduler;
pub mod vocab_coreset;

// ── Plan 386 (2026-07-05): modules moved from katgpt-rs/src/speculative/. ──
// Each gated to mirror its historical root feature gate. Root re-exports
// preserve `katgpt_rs::speculative::<module>::*` paths.

// AcceptanceForecast — entropy-bounded acceptance-rate forecast (ungated).
pub mod acceptance_forecast;
// Best Buddies Drafting — mutual NN filter (Plan 199).
#[cfg(feature = "best_buddies")]
pub mod best_buddies;
// Domino Causal Correction — PrefixCorrectionTable + domino_score (Plan 197).
#[cfg(feature = "domino_correction")]
pub mod domino;
// SpeculativeGenerator token-domain (Plan 193).
#[cfg(feature = "speculative_generator")]
pub mod spec_generator;
// Answer Extract — parallel_probe answer extraction (Plan 133).
#[cfg(feature = "parallel_probe")]
pub mod answer_extract;
// DendriticGate NMDA-inspired adaptive tree branching (Plan 260).
#[cfg(feature = "dendritic_gate")]
pub mod dendritic_gate;
// Kurtosis Gate — polarization-driven speculative decoding (Plan 203b).
#[cfg(feature = "kurtosis_gate")]
pub mod kurtosis_gate;
// Self-Learning Selectivity Router (Plan 204).
#[cfg(feature = "selectivity_router")]
pub mod selectivity_router;
// NextLat Belief-State Speculative Drafter (Plan 217). belief_cache is the
// lock-free papaya-backed drafter cache; belief_drafter is the MLP itself.
#[cfg(feature = "belief_drafter")]
pub mod belief_cache;
#[cfg(feature = "belief_drafter")]
pub mod belief_drafter;
// Weak-side probe artifact (Issue 865 T2): the BLAKE3-committed freeze/thaw
// wire for the probe_guidance decode lane — a LatentDynamicsMLP connector +
// shared trunk lm_head + tap metadata, sealed and verified on load. Gated on
// belief_drafter because the connector IS that MLP class (reused, not
// re-invented); training lives in the riir-train nextlat_* lane pattern.
#[cfg(feature = "probe_artifact")]
pub mod probe_artifact;
// Bigram Markov head — the modelless sequential drafter (Issue 659,
// Research 316 §3.5 path 2). belief_drafter above is the trained-MLP
// drafter (NextLat); this is its table-lookup sibling: deterministic CSR
// top-m bigram table + zero-alloc greedy-chain marginal emission into the
// build_dd_tree seam.
#[cfg(feature = "bigram_markov")]
pub mod bigram_markov;
// Pair-scored path selection — the modelless DFlash 2 selector (Issue 671,
// Research 490): t-step marginal propagation (the U_t parallel drafter) +
// adjacent-pair coherence from the same table, walked greedily.
#[cfg(feature = "bigram_markov")]
pub mod pair_select;
// Grid re-selection — the Issue 966 DCOCO-class corrector: top-K pools over
// the DRAFTER's own step-major marginals (dflash output) + prefix-first
// scored walk via pair_select's shared selection core. Implies bigram_markov
// (the transition prior).
#[cfg(feature = "grid_reselect")]
pub mod grid_reselect;
// NFCoT FlowScore Generator + QGF Fusion (Plan 229 / Plan 268 T6). nf_flow
// (the scorer core) is already ungated above; these two compose it with
// spec_generator / QGuidedDrafter and are gated on both parents.
#[cfg(all(feature = "nf_flow_score", feature = "speculative_generator"))]
pub mod nf_flow_generator;
#[cfg(all(feature = "nf_flow_score", feature = "qgf_drafter"))]
pub mod nf_flow_qgf;

// PPoT core primitives (Issue 003): types/knowledge/entropy/rank moved here;
// the `resample` orchestrator stays in katgpt-rs root and consumes these via
// re-export. Gated because the moved files carry internal `#[cfg(feature = "ppot")]`
// gates (adaptive-knowledge path) that require the feature to be live.
#[cfg(feature = "ppot")]
pub mod ppot;

// ── Proposal 003 Phase 6 absorptions (2026-07-04) ──────────────────────────
// Modules moved from katgpt-rs/src/. Each is feature-gated to mirror its
// historical root gate. Root re-exports preserve `katgpt_rs::<module>::*` paths.

// distill umbrella — ilc (Phase 6) + trd (Plan 384) live here. peira lives
// in katgpt-spectral; root re-exports it at katgpt_rs::distill::peira.
// Gate on any distill submodule feature so the umbrella compiles whenever
// at least one sub-module is requested.
#[cfg(any(feature = "ilc_distill", feature = "trd_refined_draft"))]
pub mod distill;

// RTPurbo retrieval-head sparse decode (Plan 126).
#[cfg(feature = "rt_turbo")]
pub mod rt_turbo;

// PASD boundary-aware draft scoring (Plan 227 Phase 4).
#[cfg(feature = "precision_aware_draft")]
pub mod precision_aware_draft;

// SpecHop continuous multi-hop speculation pipeline (Plan 131).
#[cfg(feature = "spechop")]
pub mod spechop;

// Speculative Reconciliation Engine (Plan 177). Originally ungated in root
// (the `spec_reconciliation = []` feature was vestigial — the module compiled
// unconditionally). Preserved here as ungated to match historical behavior.
pub mod spec_reconciliation;

// Re-export katgpt_core's speculative types + traits so consumers can import
// everything from one place (`katgpt_speculative::{TreeNode, ConstraintPruner, …}`).
pub use katgpt_core::speculative::types::*;
pub use katgpt_core::traits::{
    BinaryScreeningPruner, CompletionHorizon, ConstraintPruner, DominoPruner, NoPruner,
    NoScreeningPruner, ScreeningPruner,
};

// ── Phase 12 T4.5 (2026-07-04): modules moved from katgpt-rs/src/. ──
// Progressive MCGS — Monte Carlo Graph Search (Plan 148). Self-contained.
#[cfg(feature = "progressive_mcgs")]
pub mod progressive_mcgs;
// Chain Fold — step-boundary-aware context folding (Plan 195, GOAT 16/16).
// Needs katgpt-kv/still_kv for chain_folder's KV compaction types.
#[cfg(feature = "chain_fold")]
pub mod fold;

// ── Plan 387 (2026-07-05): Phase 2 speculative cluster move. ──
// 10 modules moved from katgpt-rs/src/speculative/. Leaf-only deps verified
// via line-range grep (Plan 386 R296-class lesson). Root re-exports preserve
// `katgpt_rs::speculative::<module>::*` paths.

// LDT α-operator conflict-clause pruning (Plan 088, GOAT 7/7, default-on).
// Forwards to katgpt-core/lattice_deduction for the trait half.
#[cfg(feature = "lattice_deduction")]
pub mod alpha;

// LDT conflict-triggered targeted candidate reopening (Issue 956 T2).
// Composes the leaf's own `alpha::ConflictClauseDB` + katgpt-core's
// `EntropyConflictDetector` over the shared screened builder. Default-off
// (`ReopenConfig::default()` is the byte-identical plain build).
#[cfg(feature = "lattice_deduction")]
pub mod lattice_reopen;

// Compression-adaptive decode budget (Plan 167). Always-on in root; mirrors
// that here. Uses katgpt_core::speculative::types::BudgetAdaptation.
pub mod budget;

// Budget compat shims (Plan 167). Always-on; uses crate::budget::*.
pub mod budget_compat;

// CaDDTree cost-aware adaptive DDTree budget (Plan 219, GOAT 7/7, default-on).
// Uses crate::dd_tree::build_dd_tree + katgpt_core::mux_demux.
#[cfg(feature = "caddtree_budget")]
pub mod caddtree_budget;

// Flow-based ScreeningPruner (Plan 030 bandit cluster). Gated by `bandit`
// (local switch in this crate — does NOT pull katgpt-pruners to avoid cycle).
#[cfg(feature = "bandit")]
pub mod flow_pruner;

// PEIRA distill ScreeningPruner (Plan 153, GOAT 7/7, default-on). Gated by
// `peira_distill` (tracking flag in this crate).
#[cfg(feature = "peira_distill")]
pub mod peira_pruner;

// Precision-Aware Speculative Generator (Plan 227 Phase 4). Composes
// precision_aware_draft + spec_generator (both leaf-local).
#[cfg(all(feature = "precision_aware_draft", feature = "speculative_generator"))]
pub mod precision_aware_generator;

// Residency audit — KV cache residency tracking. Always-on in root.
pub mod residency_audit;

// Trust-Region Adaptive Speculation (Plan 182). Always-on module in root
// (the `trust_region_spec` feature gates the bandit-backed routing path in
// root's re-export; the module substrate is always compiled).
pub mod trust_region;

// Domino LoRA causal correction adapter (Plan 231).
#[cfg(feature = "domino_lora")]
pub mod domino_lora;

// AND-OR DDTree builder (Plan 190 T2). Uses ScreeningPruner::relevance() to
// detect low-confidence regions and ProofGoalCache for subgoal memoization.
// Gated by `and_or_dtree` (mirrors root gate).
#[cfg(feature = "and_or_dtree")]
pub mod and_or_builder;

// ECHO Environment Predictor (Plan 247). Forward model + consistency gate
// that wraps EnvPredictorPruner as a ScreeningPruner. The BanditPruner
// integration glue lives in katgpt-pruners::echo_env_integration.
#[cfg(feature = "echo_env_predictor")]
pub mod echo_env;

// Adaptive Chain-of-Thought controller (Plan 194). Decides per-query whether
// to think (latent reasoning) or answer directly. Gated by `thinking_cot`
// (mirrors root gate).
#[cfg(feature = "thinking_cot")]
pub mod thinking_controller;

// ── Plan 390 (2026-07-05): speculative Phase 5 — prefill substrate unblock. ──
// Prefill pure substrate (PrefillScorer trait + RandomScorer/UniformScorer +
// pure compression/selection helpers + orchestrators). The forward-coupled
// impls (AttentionScorer, BlockAttentionScorer) and the entmax-coupled
// block_select_entmax stay in root's src/speculative/prefill.rs.
// Always-on (not feature-gated) because the trait signature uses
// TransformerWeights (always-available) and the substrate functions are pure.
// Root re-exports via `pub use prefill::*` so historical
// `katgpt_rs::speculative::prefill::*` paths resolve.
pub mod prefill;

// ── Plan 389 (2026-07-05): speculative Phase 4 — parallel_probe unblock. ──
// SpeculativeVerifier trait — hosted here (not katgpt-core) because the
// signature uses TransformerWeights from katgpt-transformer (above katgpt-core).
// The impls (SimulatedVerifier, LeviathanVerifier) stay in root's verifier.rs
// because they consume `crate::transformer::forward` (forward-cycle blocker).
// Always-on (not feature-gated) because root's verifier.rs is always compiled
// and its impls reference this trait — gating it would break no-default builds.
pub mod verifier_trait;
// Re-export the trait at the crate root for ergonomic access
// (`katgpt_speculative::SpeculativeVerifier`).
pub use verifier_trait::SpeculativeVerifier;

// Parallel-Probe 2D controller (Plan 133). Moved from root's
// `src/speculative/parallel_probe.rs`. Root re-exports preserve
// `katgpt_rs::speculative::parallel_probe::*` paths.
#[cfg(feature = "parallel_probe")]
pub mod parallel_probe;

// Weaver inference-only logit corrector (Issue 131). Loads trained Weaver
// weights from a safetensors checkpoint, runs the 7-step forward pass
// (conditioning → causal attention → SwiGLU → top-K gather → residual add
// → renormalize), and applies the residual correction to DFlash draft
// logits over the top-K candidate set.
#[cfg(feature = "weaver_runtime")]
pub mod weaver;
