//! Gated DeltaNet-2 (GDN2) — O(1) decode kernel + types.
//!
//! This module owns the GDN2 recurrent attention substrate (kernel + types).
//! The composition layer (`forward_gdn2`, which takes `ForwardContext`) stays
//! in the root crate (`katgpt_rs::gdn2::forward`).
//!
//! See the root `gdn2/mod.rs` for the full architecture documentation.
//! Reference: Yang, Zhang, Kautz (2024). "Gated Delta Networks."

pub mod kernel;
pub mod types;

pub use kernel::{gdn2_recurrent_step, gdn2_state_readout, gdn2_state_update, l2_normalize, sigmoid};
pub use types::{Gdn2GateConfig, Gdn2HeadState, Gdn2LayerState, MultiLayerGdn2Cache};
