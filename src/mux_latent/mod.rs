//! MUX-Latent Context Compression — inference-time context compression via
//! vocabulary superposition. Distilled from LCLM (arXiv:2606.09659).
//!
//! Architecture: Input tokens → MUX superposition encoder → latent slots
//!                                                    → domain_latent mid-layer injection
//!
//! No training required. Uses existing MUX infrastructure for zero-allocation
//! position-weighted superposition with lossless separation guarantee.

mod buffer;
mod config;
mod context;
mod encoder;
mod expand;
mod inject;
mod spectral_lod;

pub use buffer::{BufferStats, EvictionPolicy, LatentContextBuffer};
pub use config::MuxLatentConfig;
pub use context::{CompressedContext, LatentSegment};
pub use encoder::MuxLatentEncoder;
pub use expand::expand_segment;
pub use inject::{CompressionSummary, LatentPrefillAdapter, MixedPrefillSequence, PrefillEntry};
pub use spectral_lod::SpectralLOD;
