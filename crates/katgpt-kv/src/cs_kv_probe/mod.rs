//! CS-KV-Importance Probe + Density-Budget Interpolator.
//!
//! Modelless primitives distilled from arxiv 2606.13594 (Research 247,
//! "See What I See, Know What I Think"). MIT-licensed, no game semantics.
//!
//! # The three primitives
//!
//! 1. **[`CsKvProbe`]** — compressed-sensing KV-group importance probe. Given a
//!    black-box eval function, `M` ablation masks, `N` episodes →
//!    [`KvGroupRanking`]. Pure inference, zero training; the only "learning" is
//!    one coordinate-descent Lasso solve on a fixed measurement matrix.
//! 2. **[`DensityBudget`]** — the `K(ca)` interpolator. Given `ca ∈ [0,1]`,
//!    `K_sparse`, `K_dense`, `D` → integer top-K budget. One scalar in, one out.
//! 3. **[`GatedKvSlice`]** — applies a ranking + budget to a KV cache via the
//!    SP-KV `soft_gate_bias` convention (`log(s + ε)`). **Sigmoid-compatible,
//!    never softmax.** Zero-allocation apply path (`&mut [f32]` out).
//!
//! # Feature gate
//!
//! Enabled via the `cs_kv_probe` feature in `Cargo.toml` — **opt-in**, NOT in
//! `default` or `full` until the Plan 280 G2 gate (sparse-vs-dense duality
//! shape at our dimensionality) passes.
//!
//! # References
//!
//! - Plan: `katgpt-rs/.plans/280_cs_kv_importance_probe.md`
//! - Research: `katgpt-rs/.research/247_Dense_Latent_Heterogeneous_Communication_CS_Probe.md`
//! - SP-KV gate convention: `src/sp_kv/utility_predictor.rs` (`soft_gate_bias`)

pub mod budget;
pub mod gate;
pub mod lasso;
pub mod probe;
pub mod types;

// DensityBudget's struct lives in `types.rs`; its impl is split (constructor +
// Default in types, the K(ca) interpolator in budget). Re-export from `budget`
// so the interpolator API and the type travel together.
pub use budget::DensityBudget;
pub use gate::GatedKvSlice;
pub use lasso::lasso;
pub use probe::{CsKvProbe, CsProbeConfig, sample_masks};
pub use types::{AblationMask, Episode, KvGroupRanking};

// Position-disentangled cross-shape KV transport (Plan 280 §2.2 Primitive 3):
// `shard_kv::rope::{undo_rope, reapply_rope}` (Plan 147, GOAT-proved) is the
// canonical RoPE strip/restore pair. We DO NOT reinvent it here. A hard
// re-export would couple `cs_kv_probe` to the `shard_kv` feature; instead,
// callers who need cross-shape transport should depend on `shard_kv`
// directly and use `katgpt_rs::shard_kv::{undo_rope, reapply_rope}`. The
// probe itself never touches RoPE — it operates on already-flattened
// `Episode::kv_cache` slices supplied by the caller.
//
// When Plan 311 (riir-ai) wires the NPC runtime, it will pull both features
// and compose them at the call site, not via a re-export here.
