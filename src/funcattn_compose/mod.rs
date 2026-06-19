//! Functional Attention composition layer — Plan 286 Phase 5 (T5.1–T5.3).
//!
//! Composes the [`crate::funcattn`] primitive (shipped in `katgpt-core`) with
//! three sibling shipped primitives, all opt-in. The FUNCATTN forward hot path
//! lives in `katgpt-core::funcattn` (zero-alloc, G5-verified); this module only
//! adds composition *glue* and *runtime* artifacts — it never touches the
//! per-token decode path unless the caller explicitly invokes a composition.
//!
//! # Why this is a separate module from `katgpt-core::funcattn`
//!
//! `katgpt-core` is a dependency of this crate, not the reverse. Two of the
//! three compositions wire FUNCATTN to primitives that live in this outer
//! crate ([`crate::spectralquant`] and [`crate::chiaroscuro`]), so the
//! composition layer must live here. The freeze/thaw snapshot (T5.3) lives
//! here too for Phase-5 cohesion and because its atomic hot-swap is a runtime
//! concern (the bridge to riir-ai Plan 318).
//!
//! # Features (each opt-in — NOT in `default`, NOT in `full`)
//!
//! | Feature | Composes | Task |
//! |---------|----------|------|
//! | `funcattn_spectral_pre_rotate` | FUNCATTN × SpectralQuant eigenbasis | T5.1 |
//! | `funcattn_chiar_blend`         | FUNCATTN × CHIAR spectral-entropy routing | T5.2 |
//! | `funcattn_freeze_thaw`         | FUNCATTN × BLAKE3-committed snapshot hot-swap | T5.3 |
//! | `funcattn_compose`             | parent — enables all three | — |
//!
//! Per Plan 286 Phase 5 ("Each opt-in") and the plan's Gain-tier verdict, none
//! of these promote to default-on until a composition-specific GOAT gate proves
//! a gain. They ship as opt-in experiments whose mechanics are verified here.
//!
//! # Latent vs raw boundary
//!
//! All three compositions operate on latent weight/activation tensors only.
//! Nothing crosses the sync boundary (no `MapPos`, no `SyncBlock`). The
//! freeze/thaw snapshot's **weights blob is latent and never synced**; only its
//! BLAKE3 commitment + version would be emitted as an audit event by a runtime
//! consumer (riir-ai), matching the `micro_belief::snapshot` contract.

#[cfg(feature = "funcattn_spectral_pre_rotate")]
pub mod spectral_pre_rotate;

#[cfg(feature = "funcattn_chiar_blend")]
pub mod chiar_blend;

#[cfg(feature = "funcattn_freeze_thaw")]
pub mod freeze_thaw;
