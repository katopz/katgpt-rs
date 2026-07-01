//! RAT+ Recurrence Bridge — Modelless dilated inference
//!
//! Wires GDN2 recurrent state as bridge for sparse dilated attention.
//! No retraining — pure inference-time adaptation.
//! Target: 8-64× attention FLOPs reduction with <2% quality degradation.
//!
//! Plan 225, Research 201.

#[cfg(feature = "rat_plus_bridge")]
mod bridge;
#[cfg(feature = "rat_plus_bridge")]
mod dilated_kv;
#[cfg(feature = "rat_plus_bridge")]
mod fuse;

#[cfg(feature = "rat_plus_bridge")]
mod dilation_router;
#[cfg(feature = "rat_plus_bridge")]
mod vortex;

#[cfg(feature = "rat_plus_bridge")]
pub use bridge::*;
#[cfg(feature = "rat_plus_bridge")]
pub use dilated_kv::*;
#[cfg(feature = "rat_plus_bridge")]
pub use dilation_router::*;
#[cfg(feature = "rat_plus_bridge")]
pub use fuse::*;
#[cfg(feature = "rat_plus_bridge")]
pub use vortex::*;
