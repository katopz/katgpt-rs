//! Key selection: choosing the compact keys `Ck ⊂ K`.
//!
//! Two families:
//! - [`highest_attn`]: top-t keys by aggregated attention score (fastest)
//! - [`omp`]: Orthogonal Matching Pursuit on the mass feature matrix (best quality)
//!
//! Per the paper, both produce a `(selected_indices, weights)` pair where the
//! weights `w = exp(β)` directly solve the NNLS mass-matching problem on the
//! selected subset.

pub mod highest_attn;
pub mod omp;

use crate::types::{KeySelector, ScoreMethod};

pub use highest_attn::select_highest_attn_keys;
pub use omp::select_omp_keys;

/// Output of a key selection algorithm.
#[derive(Debug, Clone)]
pub struct KeySelection {
    /// Selected indices into the original `K` (length `t`).
    pub indices: Vec<usize>,
    /// Per-selected-key weights `w = exp(β)` (length `t`). May be all 1.0 if
    /// the selector does not produce weights (e.g., raw HighestAttnKeys).
    pub weights: Vec<f32>,
}

/// Discriminator for runtime selection of selector kind.
///
/// This mirrors [`KeySelector`] but is used where the caller already has a
/// `KeySelector` value rather than a `&AmConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KeySelectorKind {
    HighestAttn = 0,
    Omp = 1,
    OmpFast = 2,
}

impl From<KeySelector> for KeySelectorKind {
    #[inline]
    fn from(s: KeySelector) -> Self {
        match s {
            KeySelector::HighestAttnKeys => Self::HighestAttn,
            KeySelector::Omp => Self::Omp,
            KeySelector::OmpFast => Self::OmpFast,
        }
    }
}

impl From<ScoreMethod> for KeySelectorKind {
    /// Convenience: ScoreMethod → selector kind defaults to HighestAttn.
    #[inline]
    fn from(_: ScoreMethod) -> Self {
        Self::HighestAttn
    }
}
