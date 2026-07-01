//! katgpt-quant — Quantization codecs for KV cache compression.
//!
//! Five codecs extracted from `katgpt-rs/src/` (Proposal 003 Phase 1):
//! - **TurboQuant** (`turboquant`): random rotation + uniform codebook. Legacy baseline.
//! - **PlanarQuant** (`planar_quant`): 2D Givens rotation — O(d) vs TQ O(d²).
//! - **IsoQuant** (`iso_quant`): 4D quaternion rotation — O(d), 512 FMAs for d=128.
//! - **OCTOPUS** (`octopus`): octahedral triplet codec — data-oblivious, dominates SQ.
//! - **Hybrid OCT-PQ** (`hybrid_oct_pq`): OCT encoding + PQ rotation.
//!
//! All codecs depend on `katgpt-core` for SIMD kernels + shared types. The
//! inter-codec dependency chain is: turboquant (base) ← planar_quant, iso_quant;
//! octopus (standalone); hybrid_oct_pq (planar_quant + octopus).

#[cfg(feature = "turboquant")]
pub mod turboquant;

#[cfg(feature = "planar_quant")]
pub mod planar_quant;

#[cfg(feature = "iso_quant")]
pub mod iso_quant;

#[cfg(feature = "octopus")]
pub mod octopus;

#[cfg(feature = "hybrid_oct_pq")]
pub mod hybrid_oct_pq;
