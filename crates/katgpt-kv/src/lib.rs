//! katgpt-kv — KV-cache namespace.
//!
//! All KV-cache compression, compaction, projection-sharing, and quantization
//! backends extracted from `katgpt-rs/src/` (Issue 015 Phase 3). Each backend
//! is gated by its historical feature flag, preserving the pre-extraction
//! semantics 1:1.
//!
//! # Modules
//!
//! | Module | Feature | Origin | Plan |
//! |--------|---------|--------|------|
//! | `kv_share` | `kv_share` | `src/kv_share.rs` | Plan 185 — Q-K=V projection sharing (50% cache reduction) |
//! | `osc_kv` | `osc_kv` | `src/osc_kv.rs` | Plan 189 — Oscillatory KV cache, IMEX discretization |
//! | `cs_kv_probe` | `cs_kv_probe` | `src/cs_kv_probe/` | Plan 280 — Compressed-sensing KV importance probe |
//! | `shard_kv` | `shard_kv` | `src/shard_kv/` | Plan 147 — ShardKV asymmetric K/V compression |
//! | `sp_kv` | `sp_kv` | `src/sp_kv/` | Plan 070 — SP-KV self-pruned key-value attention |
//! | `still_kv` | `still_kv` | `src/still_kv/` | Plan 245 — StillKV perceiver-based compaction |
//! | `kvarn` | `kvarn` | `src/kvarn/` | Research 159 — KVarN variance-normalized quantization |
//! | `targeted_precision` | `targeted_precision` | `src/targeted_precision.rs` | Plan 227 Phase 2 — per-head bit allocation |
//!
//! # Cross-crate deps
//!
//! - `katgpt-core` — SIMD kernels, `types::*` re-export (Rng, Config, kv_dim, QuantizedKVCache)
//! - `katgpt-types` — `QuantizedKVCache` trait (Issue 015 Phase 1)
//! - `katgpt-spectral` — `spectralquant::*` re-export (shard_kv K-path + kvarn via targeted_precision)
//!
//! # Re-export shim
//!
//! The root `katgpt-rs` crate re-exports each sub-module behind its feature
//! flag as `katgpt_rs::{kv_share, osc_kv, ...}`, preserving back-compat with
//! all existing call sites in `tests/`, `examples/`, and `src/` consumers
//! (`fold`, `attn_match`).

#![allow(unexpected_cfgs)]

#[cfg(feature = "kv_share")]
pub mod kv_share;
#[cfg(feature = "osc_kv")]
pub mod osc_kv;
#[cfg(feature = "cs_kv_probe")]
pub mod cs_kv_probe;
#[cfg(feature = "shard_kv")]
pub mod shard_kv;
#[cfg(feature = "sp_kv")]
pub mod sp_kv;
#[cfg(feature = "still_kv")]
pub mod still_kv;
#[cfg(feature = "kvarn")]
pub mod kvarn;
#[cfg(feature = "targeted_precision")]
pub mod targeted_precision;
