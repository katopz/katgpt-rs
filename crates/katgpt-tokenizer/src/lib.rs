//! katgpt-tokenizer — BPE, ToaST split-tree, and ConvexTok LP vocabulary optimization.
//!
//! Standalone modelless tokenizer crate extracted from `katgpt-rs/src/tokenizer/`
//! (Issue 014, 2026-06-29). Leaf crate — no `katgpt-*` dependencies.
//!
//! # Features
//!
//! - *(default)* BPE encoder/decoder (`BpeTokenizer`, `BpeTokenizerImpl`, `BpeTrainer`).
//! - `toast_tokenizer` — ToaST split-tree tokenization (Plan 122) + Double-Array
//!   Trie vocab lookup (auto-routed above `DATRIE_VOCAB_THRESHOLD`, Plan 137).
//! - `convex_tok` — ConvexTok LP vocabulary optimizer (Plan 127); implies
//!   `toast_tokenizer` and pulls `good_lp`/HiGHS.
//! - `datrie_vocab` — alias for `toast_tokenizer` (kept for back-compat with
//!   the katgpt-rs feature surface).

mod bpe;
mod types;

pub use bpe::{BpeTokenizerImpl, BpeTrainer};
pub use types::{BpeTokenizer, MergeRule};

#[cfg(feature = "toast_tokenizer")]
mod toast_builder;
#[cfg(feature = "toast_tokenizer")]
mod toast_inference;
#[cfg(feature = "toast_tokenizer")]
mod toast_types;

#[cfg(feature = "toast_tokenizer")]
pub use toast_builder::SplitTreeBuilder;
#[cfg(feature = "toast_tokenizer")]
pub use toast_inference::ToastTokenizerImpl;
#[cfg(feature = "toast_tokenizer")]
pub use toast_types::{DATRIE_VOCAB_THRESHOLD, SplitNode, SplitTree, ToastTokenizer};

// ── Double-Array Trie Vocab Lookup (Plan 137, Research 137) ──
//
// Compiled under `toast_tokenizer` — auto-built when vocab > DATRIE_VOCAB_THRESHOLD.
// Threshold-routed: no separate feature gate needed.

#[cfg(feature = "toast_tokenizer")]
mod datrie;
#[cfg(feature = "toast_tokenizer")]
pub use datrie::{DatrieTreeIndex, DatrieVocab};

// ── ConvexTok LP Vocabulary Optimizer (Plan 127, Research 087) ──

#[cfg(feature = "convex_tok")]
mod convex_certify;
#[cfg(feature = "convex_tok")]
mod convex_graph;
#[cfg(feature = "convex_tok")]
mod convex_rounding;
#[cfg(feature = "convex_tok")]
mod convex_solver;
#[cfg(feature = "convex_tok")]
mod convex_toast_bridge;
#[cfg(feature = "convex_tok")]
mod convex_types;

#[cfg(feature = "convex_tok")]
pub use convex_certify::Certifier;
#[cfg(feature = "convex_tok")]
pub use convex_graph::GraphBuilder;
#[cfg(feature = "convex_tok")]
pub use convex_rounding::Rounder;
#[cfg(feature = "convex_tok")]
pub use convex_solver::ConvexSolver;
#[cfg(feature = "convex_tok")]
pub use convex_toast_bridge::{ConvexToToastBridge, SpecialTokens};
#[cfg(feature = "convex_tok")]
pub use convex_types::{
    ColourId, FreeEdgeId, LpSolution, OptimalityCert, PricedEdgeId, RoundedVocabulary,
    RoundingScheme, TokenisationGraph, VertexId,
};
