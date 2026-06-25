//! Spectral primitives — Fourier-basis algebra on discrete samples.
//!
//! Umbrella module for narrow modelless spectral operators distilled from
//! the FNO practical-perspective survey (Research 307). Each operator ships
//! behind its own feature flag and is independently GOAT-gated.
//!
//! ## Why this module exists
//!
//! The FNO paper's modelless toolkit (spectral differentiation, spectral
//! interpolation, Fourier continuation, Tucker/HOSVD) is *partially* covered
//! by existing primitives:
//!
//! - `cross_resolution_transport` (Plan 310, DEFAULT-ON) — FNO super-resolution.
//! - `funcattn` (Plan 286) — SpectralConv analog.
//! - DEC `exterior_derivative` (Plan 251) — spectral differentiation in DEC vocabulary.
//! - `flow::fft_smooth` (Plan 242) — low-pass FFT.
//! - `LatCalSpectralFixed` (riir-chain Plan 265) — fixed-point Fourier commitment.
//!
//! The pieces that are **genuinely missing** (Research 307 §3) ship here:
//!
//! - `continuation` — Fourier continuation for non-periodic latent fields
//!   (Plan 323, feature `fourier_continuation`).
//! - `differentiation` — standalone FFT-based spectral differentiation for
//!   periodic uniform 1D grids (Plan 325, feature `spectral_differentiation`).
//!   The specialized 1D-periodic case where DEC `exterior_derivative` is
//!   overkill.
//!
//! Future narrow gaps (Tucker factorization for `NeuronShard` compaction)
//! would land under this same umbrella as a separate feature.

#[cfg(feature = "fourier_continuation")]
pub mod continuation;

#[cfg(feature = "spectral_differentiation")]
pub mod differentiation;
