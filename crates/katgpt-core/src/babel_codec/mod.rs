//! BabelCodec — Readability-relaxed semantic codec (Plan 331, Research 312).
//!
//! Distillation of the **modelless subset** of BabelTele
//! (Zhu et al., SJTU, [arXiv:2606.19857](https://arxiv.org/abs/2606.19857), Jun 2026).
//! The paper's headline 3.6× compression is **LLM-prompted** (not a deterministic
//! function). What we ship here is the deterministic fixed-rule subset
//! (paper Appendix C.2.8, BT-P8 "Fixed Symbolic Mapping Rules") plus a generic
//! latent-projection facade that re-packages existing `DensityBudget`-style
//! infrastructure under one trait.
//!
//! # What ships
//!
//! - [`BabelCodec`] — generic trait: `compress` / `decompress` / `last_ratio` /
//!   `commit` / `verify`.
//! - [`BabelPair`] — `{ compressor, reader }` analog of `LoraPair` (Plan 025).
//! - [`fixed_rule::FixedRuleTextCodec`] — deterministic BT-P8 text codec.
//!   Bijective on its supported schema (KG triples, entity-attribute pairs,
//!   config strings, conditional branches, comparisons).
//! - [`sigmoid_latent::SigmoidLatentCodec`] — generic-trait facade over the
//!   existing latent projection pattern (`DensityBudget` + `extract_hla_slice`,
//!   Plan 311). Re-skin for API uniformity, NOT new capability.
//! - [`commitment::BabelCommitment`] — BLAKE3 `[u8; 32]` newtype over the
//!   compressed bytes. The proposed LatCal chain-commitment bridge
//!   (originally Issue 002) was closed as moot — Plan 331 G2 FAILED (1.14×
//!   vs 2× bar). The commitment primitive is retained as independently useful.
//!
//! # Constraints (per AGENTS.md + Plan 331)
//!
//! - **Modelless**: no training, no backprop, no gradient descent. All operations
//!   are deterministic closed-form mappings.
//! - **Sigmoid, NOT softmax**: any squashing uses the stable `sigmoid` in
//!   [`sigmoid_latent`].
//! - **BLAKE3** for commitment (NOT SHA1/SHA256) — matches the existing crate
//!   convention.
//! - **Opt-in feature `babel_codec`**: NOT in `default`. The G2 gate
//!   (≥ 2× compression on the real Seal-style corpus) killed CompressionDrafter
//!   twice (Plan 285/287); BabelCodec must beat that bar before promotion.
//!
//! # References
//!
//! - Research: [katgpt-rs/.research/312_BabelTele_Readability_Relaxed_Semantic_Codec.md](../../../.research/312_BabelTele_Readability_Relaxed_Semantic_Codec.md)
//! - Plan: [katgpt-rs/.plans/331_babel_codec_readability_relaxed_semantic_codec.md](../../../.plans/331_babel_codec_readability_relaxed_semantic_codec.md)
//! - Compressor/reader pair analog: Plan 025 (`LoraPair { reader, writer }`)
//! - Latent-level cousin (already shipped): `NpcLatentMessage { hla_slice }` +
//!   `DensityBudget` (riir-ai Plan 311, Research 133)
//! - CompressionDrafter failure precedent: Plan 285/287,
//!   [`.benchmarks/285_compression_drafter_goat.md`](../../../.benchmarks/285_compression_drafter_goat.md)

pub mod commitment;
pub mod fixed_rule;
pub mod sigmoid_latent;

pub use commitment::BabelCommitment;
pub use fixed_rule::FixedRuleTextCodec;
pub use sigmoid_latent::{CompressedLatent, SigmoidLatentCodec};

/// Readability-relaxed semantic codec.
///
/// `compress` projects an input into a model-native compressed representation
/// that sacrifices human readability while preserving recoverable semantics.
/// `decompress` is the deterministic inverse (where defined by the impl).
///
/// The trait is generic over:
/// - `Input` — the source representation (`&str`, `&[f32; D]`, etc.).
/// - `Compressed` — the compressed representation (`Vec<u8>`, fixed-size buffer).
/// - `Reader` — auxiliary read-side state (a projection matrix, a parser
///   context, etc.). Use `()` when no reader state is needed.
///
/// # Contract
///
/// - `compress` is deterministic: same `(self, input)` → same `Compressed`.
/// - `decompress(reader, compress(x))` recovers `x` on the schema-covered
///   subset (G1 fidelity gate). Implementations document which subset is
///   bit-identical vs lossy.
/// - `last_ratio` returns the byte/element ratio achieved on the most recent
///   `compress` call (`compressed_size / original_size`, lower = better).
/// - `commit` / `verify` produce a [`BabelCommitment`] (BLAKE3 of the
///   compressed bytes). Cross-architecture deterministic (BLAKE3 is portable).
pub trait BabelCodec {
    /// Source representation.
    type Input;
    /// Compressed representation.
    type Compressed;
    /// Read-side auxiliary state (use `()` if none).
    type Reader;

    /// Readability-relaxed semantic projection. Deterministic.
    fn compress(&mut self, input: &Self::Input) -> Self::Compressed;

    /// Recover semantics. Deterministic inverse on the schema-covered subset.
    fn decompress(reader: &Self::Reader, c: &Self::Compressed) -> Self::Input;

    /// Compression ratio of the most recent `compress` call
    /// (`compressed_size / original_size` — lower is better; `< 1.0` is a win).
    fn last_ratio(&self) -> f32;

    /// BLAKE3 commitment of the most recent `compress` output.
    fn commit(&self) -> BabelCommitment;

    /// Recompute the commitment of `c` and compare.
    fn verify(&self, c: &Self::Compressed, commitment: &BabelCommitment) -> bool;
}

/// Compressor / reader pair — the structural analog of `LoraPair { reader, writer }`
/// (Plan 025). One party compresses; the recipient's reader decompresses.
///
/// Per Research 312 §1.3, BabelTele representations compressed by one model are
/// decodable by heterogeneous readers (78–110% retained accuracy across
/// Gemini/GPT/Qwen/Kimi/Claude). The pair abstraction is real: a per-NPC
/// "dialect" can compress while a different-NPC reader decompresses without
/// pairwise training.
///
/// For the deterministic fixed-rule codec shipped here, `compressor` and
/// `reader` are the same codec instance (the BT-P8 mapping is its own inverse's
/// context). The pair shape is preserved for the learned-codec future path
/// (→ riir-train). The LatCal chain-commitment bridge (originally Issue 002)
/// was closed as moot — Plan 331 G2 FAILED.
pub struct BabelPair<C, R> {
    /// Compressor side. Owns the projection / mapping.
    pub compressor: C,
    /// Reader side. Owns the inverse-projection context.
    pub reader: R,
}

impl<C, R> BabelPair<C, R> {
    /// Construct a compressor / reader pair.
    pub fn new(compressor: C, reader: R) -> Self {
        Self { compressor, reader }
    }
}
