//! `product_key_memory` — O(√N) Factored Retrieval Memory (open primitive).
//!
//! Plan 408 / Research 387. Implements the **retrieval factorization** half of
//! the FwPKM paper (Lample et al. 2019 §2.2; Zhao & Jones 2026 distillation).
//! The gradient-descent half of the paper is forbidden per AGENTS.md
//! constraint #1 and replaced by the shipped δ-rule analog (Plan 053) —
//! this primitive implements ONLY the inference-time factored retrieval.
//!
//! # The complexity-class gap this fills
//!
//! The katgpt retrieval stack today has four distinct complexity classes:
//!
//! | Retriever | Cost | Slot ceiling | Sparsity axis |
//! |---|---|---|---|
//! | Raven RSM | O(1) routing | ~10³ experts | conditional computation |
//! | Engram | O(1) hash | ~10⁵ slots (hash-collides above) | content-addressed |
//! | δ-Mem | O(r) associative | rank-r bounded | associative |
//! | **PKM (this crate)** | **O(√N) factored** | **~10⁶ slots** | **similarity-ranked** |
//!
//! PKM is the only retriever that scales to ~10⁶ slots at sub-linear cost. It
//! splits a `d_k`-dim query into two `d_k/2`-dim halves, scores two √N-row
//! codebooks, and takes the top-k of the `k×k` Cartesian product — yielding
//! `2√N + k²` scoring cost instead of `N`.
//!
//! # Modelless mandate (§3.5)
//!
//! The FwPKM paper's `L_mem` GD on value rows, `L_addr` GD on keys, and the
//! n-iter TTT loop are ALL forbidden. They are replaced by shipped substrates:
//!
//! | Forbidden paper mechanism | Modelless replacement (shipped) |
//! |---|---|
//! | `L_mem` GD on V | `DeltaMemoryState::write_segment` δ-rule (Plan 053) |
//! | `L_addr` GD on K | TEMP `sleep_diverse` diversity selector (Plan 005) |
//! | n-iter TTT loop | Sleep Consolidation N-pass (Plan 154) |
//!
//! The optional δ-rule write path over the PKM value table lands in Phase 5
//! (`product_key_memory_episodic`) and is bit-identical to one GD step at
//! η=1 — but is NOT iterated, so it is modelless.
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! - Slot keys + values (latent patterns) → latent, frozen, BLAKE3-committed.
//! - PKM table commitment root (Phase 4) → raw, syncable audit artifact.
//! - Top-k weights → latent scalar; bridge at the sync boundary.
//!
//! # Phase status (this module)
//!
//! - ✅ Phase 1 (skeleton + types) — [`types`].
//! - ✅ Phase 2 (retrieval kernel) — [`kernel`].
//! - ✅ Phase 3 (GOAT gate G1–G4) — `.benchmarks/408_pkm_goat.md`.
//! - ✅ Phase 4 (freeze/thaw wrapper) — [`freeze`] (gated
//!   `product_key_memory_freeze`).
//! - ✅ Phase 5 (δ-rule write gate, F1 fusion) — [`episodic`] (gated
//!   `product_key_memory_episodic`).
//! - ✅ Phase 6 (example + docs) — `examples/product_key_memory_demo.rs` +
//!   `.docs/28_product_key_memory.md` + README Feature Showcase entry.
//! - ⏳ Phase 7 (private fusions) — deferred to riir-* repos.
//!
//! # CRITICAL — never softmax at the *gate* level
//!
//! Per AGENTS.md, every *probability/relevance gate* in this codebase uses
//! sigmoid, not softmax. The top-k *normalization* within PKM (paper §2.2) is
//! a different concern: it is a ranking normalization over the k²-restricted
//! candidate set, NOT a probability claim. Plan 408 T2.1 step 6 documents the
//! deviation from the global sigmoid rule for this restricted case and keeps
//! softmax there for ranking fidelity vs the paper's reference implementation.
//!
//! # References
//!
//! - Plan: `katgpt-rs/.plans/408_Product_Key_Memory_Primitive.md`
//! - Research: `katgpt-rs/.research/387_Fast_Weight_Product_Key_Memory_PKM.md`
//! - Source paper: [arXiv:2601.00671](https://arxiv.org/abs/2601.00671) —
//!   Zhao & Jones, "Fast-weight Product Key Memory", Sakana AI, Feb 2026
//!   (distills the PKM factorization from Lample et al. 2019 §2.2).
//!
//! ── Product Key Memory — O(√N) Factored Retrieval (Plan 408, Research 387) ─
//!
//! Open MIT-licensed primitive: the fourth complexity class in the retrieval
//! stack (Raven O(1) / Engram O(1)-hash / δ-Mem O(r) / PKM O(√N)). Splits a
//! d_k-dim query into two halves, scores two √N codebooks, takes top-k of the
//! k² Cartesian product — yielding `2√N + k²` cost instead of `N`. Scales to
//! ~10⁶ slots at sub-linear retrieval cost.
//!
//! Modelless (constraint #1): the FwPKM paper's gradient-descent half (L_mem
//! GD on V, L_addr GD on K, n-iter TTT) is forbidden. Replaced by shipped
//! δ-rule analog (Plan 053). This primitive ships ONLY the inference-time
//! factored retrieval; the optional δ-rule write path lands in Phase 5
//! (product_key_memory_episodic).
//!
//! Phase 1 (this commit): types only — const-generic
//! `ProductKeyMemory<SQRT_N, D_K, D_V>`, `ScoreFn` (Dot/Idw), fixed-size
//! `PkQuery<K>`. Leaf-clean (zero deps). Phase 2 ships the kernel + GOAT gate.
//! DEFAULT-ON since 2026-07-07 (Plan 408 Phase 3 GOAT): G1 latency 1670×
//! speedup, G2 top-k Jaccard 1.0000 vs brute-force, G3 IDW centroid-ness PASS,
//! G4 0 allocs/1000 steady-state query_into calls. See `.benchmarks/408_pkm_goat.md`.
//!
//! Phase 4 (F4 fusion) — freeze/thaw wrapper around ProductKeyMemory. Gated
//! separately so the leaf-clean retrieval primitive (above) stays usable
//! without the Arc<RwLock<Arc<...>>> + BLAKE3 commitment machinery. See
//! `product_key_memory/freeze.rs` for the pattern rationale (mirrors
//! `induced_cwm/hot_swap.rs`).
//!
//! Plan 408 Phase 5 — δ-rule write gate over PKM (F1 fusion: PKM × δ-Mem).
//! PkmEpisodicStore wraps FrozenProductKeyMemory + a mutable working copy.
//! Gated on `product_key_memory_episodic` (implies `product_key_memory_freeze`).

// Phase 4 — freeze/thaw wrapper (Plan 408 T4.1). Gated separately so the
// leaf-clean retrieval primitive stays usable without pulling in the
// `Arc<RwLock<Arc<...>>>` + BLAKE3 commitment machinery. The wrapper re-uses
// the Phase 1/2 retrieval kernel unchanged.
#[cfg(feature = "product_key_memory_freeze")]
pub mod freeze;
#[cfg(feature = "product_key_memory_freeze")]
pub use freeze::FrozenProductKeyMemory;

// Phase 5 — δ-rule write gate (Plan 408 T5.1, F1 fusion: PKM × δ-Mem).
// PkmEpisodicStore wraps FrozenProductKeyMemory + a mutable working copy,
// applying `V[idx] += gate * (target - V[idx])` on the top-k value rows.
// Modelless: one δ-rule step per call (not iterated); the curiosity `gate`
// is sourced externally. Gated on the freeze wrapper (the publish path
// delegates to `FrozenProductKeyMemory::commit`).
#[cfg(feature = "product_key_memory_episodic")]
pub mod episodic;
#[cfg(feature = "product_key_memory_episodic")]
pub use episodic::{BackgroundAccessStats, PkmEpisodicStore};

pub mod kernel;
pub mod types;

pub use kernel::{PkmScratch, score_dot, score_idw};
pub use types::{D_K_FLOOR, PkEntry, PkQuery, ProductKeyMemory, SQRT_N_FLOOR, ScoreFn};
