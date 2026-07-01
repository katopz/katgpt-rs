//! Type definitions for the Alien Sampler primitive.
//!
//! See [`crate::alien_sampler`] for the full module doc + paper citation.

/// Configuration for [`super::sampler::AlienSampler`].
///
/// All fields are validated at construction time (see
/// [`super::sampler::AlienSampler::new`]).
///
/// Reference: Plan 311 (T1.3), Research 293.
#[derive(Clone, Copy, Debug)]
pub struct AlienConfig {
    /// Fusion weight in `[0, 1]`. `0.0` = coherence-only, `1.0` =
    /// availability-only (i.e. maximally alien / unsaturated by the community
    /// bank). Paper default `0.7`.
    ///
    /// The sampler computes `Fβ = (1−β)·zC + β·zU` where `zU = −zA` (unavailability
    /// is the negation of availability per the trait contract).
    pub beta: f32,
    /// Top-`m` parameter forwarded to any [`super::traits::AvailabilityScorer`]
    /// that implements the median-of-top-m rule (notably
    /// [`super::median_top_m::MedianTopMAvailability`]). Paper default `10`.
    ///
    /// Ignored by scorers that do not consume `m` (e.g. a custom scalar
    /// availability). Stored here so callers can build a sampler + scorer pair
    /// from a single config without re-threading the parameter.
    pub top_m: usize,
}

impl AlienConfig {
    /// Paper-default config: `beta = 0.7`, `top_m = 10`.
    #[must_use]
    pub const fn paper_default() -> Self {
        Self { beta: 0.7, top_m: 10 }
    }

    /// Coherence-only config: `beta = 0.0`. Useful as the GOAT-gate baseline
    /// arm (Plan 311 T3.2 Arm A).
    #[must_use]
    pub const fn coherence_only() -> Self {
        Self { beta: 0.0, top_m: 10 }
    }

    /// Availability-only config: `beta = 1.0`. Diagnostic only — rarely what a
    /// caller wants (it ignores coherence entirely), but useful for verifying
    /// the fusion math in tests.
    #[must_use]
    pub const fn availability_only() -> Self {
        Self { beta: 1.0, top_m: 10 }
    }
}

impl Default for AlienConfig {
    #[inline]
    fn default() -> Self {
        Self::paper_default()
    }
}

/// A scored candidate: `(score, idx)` pair returned by
/// [`super::sampler::AlienSampler::rank`].
///
/// `idx` is the caller's index into the input candidate slice; `score` is the
/// fused z-scored Fβ value (higher = more alien-coherent = ranked earlier).
///
/// Stored as a struct (not a bare tuple) so field names are self-documenting
/// at call sites and the type can grow debug provenance later without breaking
/// destructuring.
///
/// Reference: Plan 311 (T1.3), Research 293.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct ScoredCandidate {
    /// Fused Fβ score (z-scored, dimensionless).
    pub score: f32,
    /// Index into the input `candidates: &[Vec<V>]` slice.
    pub idx: usize,
}

impl ScoredCandidate {
    /// Construct from raw components.
    #[inline]
    #[must_use]
    pub const fn new(score: f32, idx: usize) -> Self {
        Self { score, idx }
    }
}

/// Partial ordering by score descending. Convenient for `sort_by` / `sort_unstable_by`.
impl Eq for ScoredCandidate {}

impl Ord for ScoredCandidate {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Higher score first. f32 has no total Ord; we use total_cmp which
        // gives a deterministic ordering (NaN sorts last, which is fine — NaN
        // inputs are guarded upstream in the sampler).
        other.score.total_cmp(&self.score)
    }
}

impl PartialOrd for ScoredCandidate {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Error type for sampler construction and ranking.
///
/// The sampler is a hot-path primitive; construction errors are programmer
/// bugs (panic on `new`), but rank-time errors on bad scratch buffer lengths
/// return `Err` so callers can recover without unwinding.
///
/// Reference: Plan 311 (T1.3), Research 293.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AlienSamplerError {
    /// Scratch buffer length did not match the candidate count.
    ScratchLengthMismatch {
        /// Expected length (= candidates.len()).
        expected: usize,
        /// Actual length passed by the caller.
        got: usize,
        /// Which buffer was wrong.
        which: &'static str,
    },
}

impl core::fmt::Display for AlienSamplerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ScratchLengthMismatch { expected, got, which } => {
                write!(
                    f,
                    "alien_sampler: {which} scratch length mismatch (expected {expected}, got {got})"
                )
            }
        }
    }
}

impl std::error::Error for AlienSamplerError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paper_default_matches_paper() {
        let cfg = AlienConfig::paper_default();
        assert!((cfg.beta - 0.7).abs() < 1e-6);
        assert_eq!(cfg.top_m, 10);
    }

    #[test]
    fn scored_candidate_descending_order() {
        let mut v = [ScoredCandidate::new(0.1, 0),
            ScoredCandidate::new(0.9, 1),
            ScoredCandidate::new(0.5, 2)];
        v.sort();
        assert_eq!(v[0].idx, 1); // 0.9
        assert_eq!(v[1].idx, 2); // 0.5
        assert_eq!(v[2].idx, 0); // 0.1
    }

    #[test]
    fn error_displays() {
        let e = AlienSamplerError::ScratchLengthMismatch {
            expected: 10,
            got: 5,
            which: "coherence",
        };
        let s = format!("{e}");
        assert!(s.contains("expected 10"));
        assert!(s.contains("got 5"));
        assert!(s.contains("coherence"));
    }
}
