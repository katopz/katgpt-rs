//! ICT detector output types.
//!
//! Plan 294, Research 270 §2.3. Decoupled from [`crate::ict::detector`] so
//! callers can hold a `BranchingReport` without pulling in the detector's
//! scratch buffers.

/// Per-call output of [`crate::ict::detector::BranchingDetector::observe_and_detect`].
///
/// Carries three signals (one per trajectory-step column):
///
/// - `mask[k]`: **the ICT selector.** `true` iff trajectory `k` is in the
///   top-`k_percent` of JS-divergence-to-group-mean — i.e. it is one of the
///   ~10% of trajectories that are genuinely diverging from the population
///   mean at this step. This is the "spend cognitive budget here" bit.
/// - `beta_per_step[k]`: collision purity β of the population mean at step
///   `k`. Drop this into the H₁→H₂ Bebop upgrade ([`crate::ict::bebop_upgrade`])
///   or any other entropy-driven gate that should be using β.
/// - `uniqueness_scores[k]`: the raw JS-divergence-to-mean `u_{k,s}` per
///   trajectory. Useful for diagnostics, sorting, and the G3 Spearman
///   correlation test.
///
/// # Design notes
///
/// The three Vecs are owned (not borrowed) because the report is meant to
/// outlive the detector's scratch buffers — callers may store reports across
/// ticks. For zero-alloc hot paths see `BranchingDetector`'s scratch-mask
/// accessor; this struct is the "snapshot for downstream" shape.
///
/// # References
///
/// - Research 270 §2.3 — primitive signatures.
/// - Research 270 §2.4 — runtime fusion recipe (the *when* of CLR/HLA/KG).
/// - Plan 294 — implementation plan + GOAT gates.
/// - arxiv 2606.19771 — source paper (Feng et al., 18 Jun 2026).
/// - riir-ai `.research/142_*.md` — private NPC guide (the moat).
#[derive(Debug, Clone)]
pub struct BranchingReport {
    /// Per-trajectory branching mask. `mask.len() == k_trajectories`.
    pub mask: Vec<bool>,
    /// Per-step collision purity β of the population mean. Length is the
    /// number of trajectories (one β per trajectory column).
    pub beta_per_step: Vec<f32>,
    /// Per-trajectory JS-divergence-to-mean uniqueness scores. Same length
    /// as `mask`.
    pub uniqueness_scores: Vec<f32>,
}

impl BranchingReport {
    /// Construct an empty report (zero-length vectors). Useful as a default
    /// or for taking capacity via `std::mem::replace`.
    pub fn empty() -> Self {
        Self {
            mask: Vec::new(),
            beta_per_step: Vec::new(),
            uniqueness_scores: Vec::new(),
        }
    }

    /// Number of trajectories flagged as branching points.
    #[inline]
    pub fn branching_count(&self) -> usize {
        self.mask.iter().filter(|m| **m).count()
    }

    /// Fraction of trajectories flagged. In `[0, 1]`. Returns 0.0 for
    /// empty reports (no division-by-zero).
    #[inline]
    pub fn branching_fraction(&self) -> f32 {
        if self.mask.is_empty() {
            return 0.0;
        }
        self.branching_count() as f32 / self.mask.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_report_has_zero_count_and_fraction() {
        let r = BranchingReport::empty();
        assert_eq!(r.branching_count(), 0);
        assert_eq!(r.branching_fraction(), 0.0);
    }

    #[test]
    fn branching_count_and_fraction_basic() {
        let r = BranchingReport {
            mask: vec![true, false, true, false, true],
            beta_per_step: vec![0.5; 5],
            uniqueness_scores: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        };
        assert_eq!(r.branching_count(), 3);
        assert!((r.branching_fraction() - 0.6).abs() < 1e-6);
    }
}
