//! katgpt-attn — Attention stack primitives.
//!
//! Extracted from `katgpt-rs/src/` (Proposal 003 Phase 2). This crate owns the
//! attention *kernel* and *types* layers; the transformer composition layer
//! (functions that take `ForwardContext`) stays in the root crate.
//!
//! # Modules
//!
//! | Module | Feature | Notes |
//! |--------|---------|-------|
//! | [`gdn2`] | `gdn2_attention` | GDN2 recurrent attention kernel + types. `forward.rs` stays root. |
//! | [`dash_attn`] | `dash_attn` | DashAttention sparse routing kernels. `forward.rs`/`tests.rs`/`meta_router`/`sat_analysis` stay root (cross-domain deps). |
//! | [`chiaroscuro`] | `chiaroscuro` | Per-token DCT spectral entropy operator routing. |
//! | [`rat_bridge`] | `rat_plus_bridge` | RAT+ recurrence bridge — dilated inference via GDN2 state. |
//! | [`ega_attn`] | `ega_attn` | Energy-Gated Attention — spectral salience gating. |
//! | [`diagonal_gate`] | `diagonal_gate` | Shared DiagonalGate abstraction (GDN2 + Wall). |
//! | [`static_cal`] | `static_cal_tables` | Pre-computed per-head attention scales. |
//! | [`funcattn_compose`] | `funcattn_freeze_thaw` / `funcattn_spectral_pre_rotate` / `funcattn_chiar_blend` | FuncAttn composition layer (Plan 286 Phase 5). |
//!
//! # Relationship to katgpt-core
//!
//! The base attention primitives (`attention`, `parallax_attn`, `set_attention`,
//! `funcattn`) live in `katgpt-core` and are NOT moved here — moving them would
//! invert the dependency DAG (katgpt-core can't depend on katgpt-attn). They
//! remain at `katgpt_core::attention`, etc. This crate adds the root-level
//! attention modules that sit above katgpt-core in the stack.

#[cfg(feature = "gdn2_attention")]
pub mod gdn2;

#[cfg(feature = "diagonal_gate")]
pub mod diagonal_gate;

#[cfg(feature = "dash_attn")]
pub mod dash_attn;

#[cfg(feature = "chiaroscuro")]
pub mod chiaroscuro;

#[cfg(feature = "rat_plus_bridge")]
pub mod rat_bridge;

#[cfg(feature = "ega_attn")]
pub mod ega_attn;

#[cfg(feature = "static_cal_tables")]
pub mod static_cal;

#[cfg(any(
    feature = "funcattn_freeze_thaw",
    feature = "funcattn_spectral_pre_rotate",
    feature = "funcattn_chiar_blend"
))]
pub mod funcattn_compose;
