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

// ── fast_bpe — gigatoken-derived fast BPE encode path (Issue 191, Research 456) ──
//
// Vendored from gigatoken's pure-Rust `bpe/` core (PairRankTable + branchless
// merge cores + ShortPretokenCache substrate). Ships behind the `fast_bpe`
// feature flag; the GOAT gate lives in `tests/fast_bpe_goat.rs`.

#[cfg(feature = "fast_bpe")]
mod fast_bpe;

#[cfg(feature = "fast_bpe")]
pub use bpe::FastBpeEncoder;
#[cfg(feature = "fast_bpe")]
pub use fast_bpe::{MergeScratch, PairRankTable, PairRankTableBuildError, TokenId as FastTokenId};

// ── ConvexTok LP Vocabulary Optimizer (Plan 127, Research 087) ──

#[cfg(all(feature = "convex_tok", not(target_arch = "wasm32")))]
mod convex_certify;
#[cfg(all(feature = "convex_tok", not(target_arch = "wasm32")))]
mod convex_graph;
#[cfg(all(feature = "convex_tok", not(target_arch = "wasm32")))]
mod convex_rounding;
#[cfg(all(feature = "convex_tok", not(target_arch = "wasm32")))]
mod convex_solver;
#[cfg(all(feature = "convex_tok", not(target_arch = "wasm32")))]
mod convex_toast_bridge;
#[cfg(all(feature = "convex_tok", not(target_arch = "wasm32")))]
mod convex_types;

#[cfg(all(feature = "convex_tok", not(target_arch = "wasm32")))]
pub use convex_certify::Certifier;
#[cfg(all(feature = "convex_tok", not(target_arch = "wasm32")))]
pub use convex_graph::GraphBuilder;
#[cfg(all(feature = "convex_tok", not(target_arch = "wasm32")))]
pub use convex_rounding::Rounder;
#[cfg(all(feature = "convex_tok", not(target_arch = "wasm32")))]
pub use convex_solver::ConvexSolver;
#[cfg(all(feature = "convex_tok", not(target_arch = "wasm32")))]
pub use convex_toast_bridge::{ConvexToToastBridge, SpecialTokens};
#[cfg(all(feature = "convex_tok", not(target_arch = "wasm32")))]
pub use convex_types::{
    ColourId, FreeEdgeId, LpSolution, OptimalityCert, PricedEdgeId, RoundedVocabulary,
    RoundingScheme, TokenisationGraph, VertexId,
};
