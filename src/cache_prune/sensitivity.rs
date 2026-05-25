//! CachePrune Sensitivity Module (Plan 140).
//!
//! Provides a generic [`SensitivityDetector`] trait for identifying sensitive tokens
//! in prompts. Game-specific implementations live in `riir-ai`. This module supplies
//! only the trait definition, two trivial baseline implementations
//! ([`StrictDetector`] / [`OpenDetector`]), and [`MaskedSegment`] for segmenting a
//! token stream along sensitivity boundaries.
//!
//! Reference: arXiv:2605.23640

/// Trait for identifying sensitive tokens in a prompt.
/// Implementations are domain-specific and live in riir-ai.
pub trait SensitivityDetector: Send + Sync {
    /// Name of the detector (for logging).
    fn name(&self) -> &str;

    /// Produce a binary sensitivity mask for the token sequence.
    /// `mask[i] = true` means token `i` is sensitive (excluded from cross-user sharing).
    fn detect(&self, tokens: &[u32], text: &str) -> Vec<bool>;
}

/// Default: strict masking (everything is sensitive).
pub struct StrictDetector;

/// Default: nothing is sensitive (open sharing).
pub struct OpenDetector;

impl SensitivityDetector for StrictDetector {
    fn name(&self) -> &str {
        "strict"
    }

    fn detect(&self, tokens: &[u32], _text: &str) -> Vec<bool> {
        vec![true; tokens.len()]
    }
}

impl SensitivityDetector for OpenDetector {
    fn name(&self) -> &str {
        "open"
    }

    fn detect(&self, tokens: &[u32], _text: &str) -> Vec<bool> {
        vec![false; tokens.len()]
    }
}

/// A segment of tokens bounded by sensitive tokens.
pub struct MaskedSegment {
    pub start: usize,
    pub end: usize,
    pub is_reusable: bool,
    pub contextualization_score: f32,
    pub recompute_indices: Vec<usize>,
}

impl MaskedSegment {
    /// Derive masked segments from a sensitivity mask.
    ///
    /// Scans the mask and creates segments of consecutive non-sensitive tokens.
    /// Each non-sensitive run becomes a `MaskedSegment` with `is_reusable = true`.
    /// Runs shorter than `min_length` are marked `is_reusable = false`.
    pub fn from_mask(mask: &[bool], min_length: usize) -> Vec<Self> {
        let mut segments = Vec::new();
        let n = mask.len();
        let mut i = 0;

        while i < n {
            // Skip sensitive tokens.
            if mask[i] {
                i += 1;
                continue;
            }

            // Start of a non-sensitive run.
            let start = i;
            while i < n && !mask[i] {
                i += 1;
            }
            let end = i; // exclusive

            let len = end - start;
            let is_reusable = len >= min_length;

            segments.push(MaskedSegment {
                start,
                end,
                is_reusable,
                contextualization_score: 0.0,
                recompute_indices: Vec::new(),
            });
        }

        segments
    }

    /// Length of this segment.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether this segment is empty.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_detector_returns_all_true() {
        let det = StrictDetector;
        let tokens: &[u32] = &[1, 2, 3, 4, 5];
        let mask = det.detect(tokens, "hello");
        assert_eq!(mask, vec![true, true, true, true, true]);
    }

    #[test]
    fn open_detector_returns_all_false() {
        let det = OpenDetector;
        let tokens: &[u32] = &[10, 20, 30];
        let mask = det.detect(tokens, "world");
        assert_eq!(mask, vec![false, false, false]);
    }

    #[test]
    fn from_mask_alternating_sensitivity() {
        // S, NS, S, NS, NS  →  mask = [true, false, true, false, false]
        let mask = vec![true, false, true, false, false];
        let segs = MaskedSegment::from_mask(&mask, 1);

        assert_eq!(segs.len(), 2);
        // First non-sensitive run: index 1 only.
        assert_eq!(segs[0].start, 1);
        assert_eq!(segs[0].end, 2);
        assert!(segs[0].is_reusable);
        // Second non-sensitive run: indices 3–4.
        assert_eq!(segs[1].start, 3);
        assert_eq!(segs[1].end, 5);
        assert!(segs[1].is_reusable);
    }

    #[test]
    fn from_mask_min_length_filters_short_segments() {
        // Three non-sensitive runs: lengths 1, 3, 2.
        // mask: [false, true, false, false, false, true, false, false]
        let mask = vec![false, true, false, false, false, true, false, false];
        let segs = MaskedSegment::from_mask(&mask, 2);

        assert_eq!(segs.len(), 3);
        // Run 0: length 1 < 2 → not reusable.
        assert!(!segs[0].is_reusable);
        assert_eq!(segs[0].len(), 1);
        // Run 1: length 3 >= 2 → reusable.
        assert!(segs[1].is_reusable);
        assert_eq!(segs[1].len(), 3);
        // Run 2: length 2 >= 2 → reusable.
        assert!(segs[2].is_reusable);
        assert_eq!(segs[2].len(), 2);
    }

    #[test]
    fn from_mask_all_sensitive_produces_empty() {
        let mask = vec![true, true, true, true];
        let segs = MaskedSegment::from_mask(&mask, 1);
        assert!(segs.is_empty());
    }

    #[test]
    fn from_mask_all_non_sensitive_produces_one_big_segment() {
        let mask = vec![false, false, false, false, false];
        let segs = MaskedSegment::from_mask(&mask, 1);

        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start, 0);
        assert_eq!(segs[0].end, 5);
        assert!(segs[0].is_reusable);
        assert_eq!(segs[0].len(), 5);
        assert!(!segs[0].is_empty());
    }
}
