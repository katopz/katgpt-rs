//! Step boundary detection — Plan 195 T1.
//!
//! Detects reasoning step boundaries in CoT text by scanning for structural
//! markers: double newlines (`\n\n`), think-tag transitions (`<think...>`,
//! `</think...>`), and other reasoning delimiters.

use super::types::StepBoundary;

/// Text markers that indicate a reasoning step boundary.
const BOUNDARY_MARKERS: &[&str] = &["\n\n", "</think", "<think"];

/// Detects reasoning step boundaries in a CoT text string.
///
/// Scans for `\n\n`, `</think`, and `<think` markers. Tag transitions
/// (`<think` and `</think`) are marked as anchors (must-keep boundaries).
///
/// Returns a sorted `Vec<StepBoundary>` ordered by token position.
pub fn detect_step_boundaries(text: &str) -> Vec<StepBoundary> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut raw_boundaries: Vec<(usize, bool)> = Vec::new();
    let mut search_start = 0;

    while search_start < text.len() {
        let mut best_pos = usize::MAX;
        let mut best_is_anchor = false;

        for marker in BOUNDARY_MARKERS {
            if let Some(offset) = text[search_start..].find(marker) {
                let pos = search_start + offset;
                if pos < best_pos {
                    best_pos = pos;
                    best_is_anchor = is_anchor_marker(marker);
                }
            }
        }

        if best_pos == usize::MAX {
            break;
        }

        // Deduplicate: skip if same position as last entry.
        match raw_boundaries.last() {
            Some(&(last_pos, _)) if last_pos == best_pos => {}
            _ => raw_boundaries.push((best_pos, best_is_anchor)),
        }

        // Advance past the marker to avoid infinite loop on zero-length matches.
        search_start = best_pos + 1;
    }

    // Always include position 0 as the first step boundary.
    if raw_boundaries.first().is_none_or(|&(pos, _)| pos > 0) {
        raw_boundaries.insert(0, (0, true));
    }

    // Map to StepBoundary with sequential step indices.
    raw_boundaries
        .into_iter()
        .enumerate()
        .map(|(step_index, (token_pos, is_anchor))| {
            StepBoundary::new(token_pos, step_index, is_anchor)
        })
        .collect()
}

/// Check if a marker is an anchor (must-keep reasoning boundary).
#[inline]
fn is_anchor_marker(marker: &str) -> bool {
    matches!(marker, "<think" | "</think")
}

/// Count the number of reasoning steps in text.
pub fn count_steps(text: &str) -> usize {
    detect_step_boundaries(text).len()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text() {
        let boundaries = detect_step_boundaries("");
        assert!(boundaries.is_empty());
    }

    #[test]
    fn test_no_boundaries() {
        let text = "Hello world this is a single step";
        let boundaries = detect_step_boundaries(text);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].token_pos, 0);
        assert!(boundaries[0].is_anchor); // Position 0 is always anchor
    }

    #[test]
    fn test_double_newline_boundaries() {
        let text = "Step 1\n\nStep 2\n\nStep 3";
        let boundaries = detect_step_boundaries(text);

        assert!(boundaries.len() >= 3);

        // First boundary is always position 0.
        assert_eq!(boundaries[0].token_pos, 0);

        // Check that double newlines create non-anchor boundaries.
        let non_anchors: Vec<_> = boundaries.iter().filter(|b| !b.is_anchor).collect();
        assert!(
            non_anchors.len() >= 2,
            "Expected at least 2 non-anchor boundaries from \\n\\n, got {}",
            non_anchors.len()
        );
    }

    #[test]
    fn test_think_tag_anchors() {
        let text = "<think_reasoning>Let me think</think_reasoning>Answer is 42";
        let boundaries = detect_step_boundaries(text);

        // Should have anchor boundaries at think-tag positions.
        let anchors: Vec<_> = boundaries.iter().filter(|b| b.is_anchor).collect();
        assert!(
            anchors.len() >= 2,
            "Expected at least 2 anchor boundaries from think tags, got {}",
            anchors.len()
        );
    }

    #[test]
    fn test_mixed_boundaries() {
        let text = "Step 1\n\nStep 2\n\n<think_analysis>Deep thought</think_analysis>\n\nStep 3";
        let boundaries = detect_step_boundaries(text);

        assert!(boundaries.len() >= 4);

        // Verify step indices are sequential.
        for (i, b) in boundaries.iter().enumerate() {
            assert_eq!(b.step_index, i);
        }
    }

    #[test]
    fn test_count_steps() {
        let text = "Step 1\n\nStep 2\n\nStep 3";
        let count = count_steps(text);
        assert!(count >= 3);
    }

    #[test]
    fn test_sequential_step_indices() {
        let text = "a\n\nb\n\nc\n\nd\n\ne";
        let boundaries = detect_step_boundaries(text);
        for (i, b) in boundaries.iter().enumerate() {
            assert_eq!(b.step_index, i);
        }
    }

    #[test]
    fn test_consecutive_newlines_no_duplicates() {
        // Three consecutive newlines = two overlapping \n\n boundaries.
        // Position 6: \n\n (chars 6,7), Position 7: \n\n (chars 7,8).
        // With +1 advance, both are found → 3 boundaries total (pos 0 + 2).
        let text = "Step 1\n\n\nStep 2";
        let boundaries = detect_step_boundaries(text);
        // Position 0 + two \n\n positions = 3 boundaries.
        assert_eq!(boundaries.len(), 3);
    }
}
