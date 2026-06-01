//! Tokenizer module — BPE, ToaST split-tree, and ConvexTok LP vocabulary optimization.

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
