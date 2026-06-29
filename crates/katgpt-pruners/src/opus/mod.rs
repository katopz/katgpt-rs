//! OPUS-Inspired Boltzmann + Redundancy Selection (Plan 129).
//!
//! Based on OPUS paper (arXiv:2602.05400): Boltzmann sampling with redundancy
//! penalty outperforms greedy top-k by +1.26 avg points on real benchmarks.
//!
//! # Architecture
//!
//! - [`CountSketch`] — O(d) → O(m) dimensionality reduction with unbiased inner-product estimation
//! - [`boltzmann_sample`] / [`boltzmann_sample_batch`] — temperature-controlled softmax sampling
//! - [`OpusBanditPruner`] — wraps [`BanditPruner`] with redundancy penalty + Boltzmann selection
//! - [`OpusConfig`] — configuration with paper defaults (τ=0.9, m=8192, ρ=0.5)
//!
//! # Module Structure
//!
//! ```text
//! src/pruners/opus/
//! ├── mod.rs           # This file — index only
//! ├── types.rs         # OpusConfig, OpusBanditPruner<P>, OpusRedundantEnv
//! ├── count_sketch.rs  # CountSketch projection (standalone, reusable)
//! └── boltzmann.rs     # Boltzmann sampling with batch without-replacement
//! ```

pub mod boltzmann;
pub mod count_sketch;
pub mod types;

pub use boltzmann::{boltzmann_probabilities, boltzmann_sample, boltzmann_sample_batch};
pub use count_sketch::{CountSketch, exact_inner_product, squared_norm};
pub use types::{OpusBanditPruner, OpusConfig, OpusRedundantEnv};
