//! Higher-order Linear Attention (HLA) — O(1) inference cache substrate.
//!
//! Implements second-order HLA (symmetric + asymmetric AHLA) as an alternative
//! to standard KV-cache attention. Achieves O(1) per-token memory independent
//! of sequence length, replacing the growing KV cache with fixed-size prefix
//! sufficient statistics that capture higher-order query-key interactions.
//!
//! # What lives here (substrate)
//!
//! This module contains the **pure substrate** half of HLA:
//! - [`types`] — cache state structs (`HlaLayerState`, `MultiLayerHlaCache`,
//!   `MultiLayerAhlaCache`, Parallax variants, `HlaVariant`)
//! - [`kernel`] — zero-alloc streaming recurrence kernels (`hla_state_update`,
//!   `hla_readout`, `ahla_step`, full-layer helpers)
//!
//! Both depend only on `crate::simd` and `crate::types::Config` — no root-only
//! cognitive modules, no `ForwardContext`, no role transport. This is the
//! publishable-leaf half: any crate can `cargo add katgpt-core` and get the
//! HLA substrate without pulling the engine.
//!
//! # What stays in the root crate (composition)
//!
//! - `forward_hla` / `forward_ahla` / `generate_*` — composition functions that
//!   wire HLA kernels into a full transformer forward pass. They depend on
//!   `ForwardContext` (root-only, has pruner fields), so they live in
//!   `katgpt-rs/src/hla/forward.rs` and re-export the substrate from here.
//!
//! # What stays in riir-engine (cognitive extensions)
//!
//! - `*_role_aware` kernel variants + `forward_hla_role_aware` — apply diagonal
//!   role transport to keys per head. Depend on `crate::role_transport::*`
//!   (Category C private). Live in `riir-engine/src/hla/` behind the
//!   `hla_role_aware` feature.
//! - `ThirdOrderMoment` + `third_order_update` / `third_order_readout` —
//!   compressed key-key-value interactions (Plan 151 T13). riir-engine only.
//!
//! Reference: Zhang, Qin, Wang, Gu (2026). "Higher-order Linear Attention."
//! See `.research/28_Higher_order_Linear_Attention.md` for full derivation.
//!
//! # Origin
//!
//! Moved from `katgpt-rs/src/hla/{types,kernel}.rs` (Plan 008 Phase 1 Step 4,
//! 2026-06-28). The forward functions stayed in the root crate — same split
//! pattern as Plan 008 Step 2 (`katgpt-transformer` got the substrate types,
//! root kept the forward composition).

pub mod kernel;
pub mod types;

pub use kernel::{
    ahla_denom, ahla_layer_step, ahla_step, hla_denom, hla_layer_readout, hla_layer_update,
    hla_readout, hla_readout_normalized, hla_state_update,
};
pub use types::{
    AhlaLayerState, AhlaQHeadState, HlaLayerState, HlaQHeadState, HlaVariant, MultiLayerAhlaCache,
    MultiLayerHlaCache, MultiLayerParallaxAhlaCache, ParallaxAhlaLayerState,
    ParallaxAhlaQHeadState,
};
