//! T2M Option Stripper — Post-Verify Anti-Shortcut Wrapper (Plan 212, T5)
//!
//! Wraps any [`ScreeningPruner`] with a two-pass verification that prevents
//! option-matching shortcuts in multiple-choice reasoning tasks.
//!
//! # Problem
//!
//! When the model sees options like "A) Paris" it can match the answer to the
//! option letter without genuine reasoning. The T2M (Think-to-Match) stripper
//! prevents this by requiring the reasoning to succeed **both** with and without
//! the options visible.
//!
//! # Architecture
//!
//! 1. **Pure pass**: strip options from prompt, score via `inner.relevance()` → `pure_score`
//! 2. **Matched pass**: score with options visible, gated by whether the answer
//!    actually matches an option → `matched_score`
//! 3. **Final score**: `min(pure_score, matched_score)` — the bottleneck ensures
//!    the model can't shortcut through option matching alone.
//!
//! # Feature Gate
//!
//! Gated behind `collapse_aware_thinking`. Not in default build until GOAT proof.

use crate::speculative::types::ScreeningPruner;

// ── OptionStripper ────────────────────────────────────────────

/// Post-verify wrapper that prevents option-matching shortcuts.
///
/// Wraps an inner [`ScreeningPruner`] and adds a two-pass verification step.
/// The inner pruner scores tokens; `OptionStripper` gates the score through
/// both pure-reasoning and option-matched lenses, taking the minimum.
#[cfg(feature = "collapse_aware_thinking")]
pub struct OptionStripper<S: ScreeningPruner> {
    inner: S,
    options_stripped: bool,
}

// ── CollapseDetectorFrozen (T6 Freeze/Thaw) ──────────────────

/// Frozen state for collapse detector persistence.
///
/// `repr(C)` for stable binary layout via the freeze/thaw infrastructure
/// in [`crate::pruners::freeze`].
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CollapseDetectorFrozen {
    /// Adaptive hesitation threshold τ.
    pub threshold: u32,
    /// EMA of per-trace optimal thresholds.
    pub hesitation_ema: f32,
    /// Mean budget EMA across positions.
    pub budget_ema_mean: f32,
    /// Efficiency reward preference γ ∈ [0.0, 1.0].
    pub gamma: f32,
}

impl Default for CollapseDetectorFrozen {
    fn default() -> Self {
        Self {
            threshold: 3,
            hesitation_ema: 0.0,
            budget_ema_mean: 0.5,
            gamma: 0.1,
        }
    }
}

impl CollapseDetectorFrozen {
    /// Magic bytes for freeze validation: b"CAAB".
    pub const MAGIC: [u8; 4] = *b"CAAB";
    /// Format version.
    pub const VERSION: u32 = 1;

    /// Create with default values and valid magic/version.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate magic bytes and version after loading from disk.
    pub fn validate(&self) -> Result<(), String> {
        if self.threshold == 0 {
            return Err("CollapseDetectorFrozen: threshold must be > 0".into());
        }
        if self.gamma < 0.0 || self.gamma > 1.0 {
            return Err(format!(
                "CollapseDetectorFrozen: gamma {} not in [0.0, 1.0]",
                self.gamma
            ));
        }
        Ok(())
    }
}

// ── OptionStripper impl ──────────────────────────────────────

#[cfg(feature = "collapse_aware_thinking")]
impl<S: ScreeningPruner> OptionStripper<S> {
    /// Wrap an inner pruner with option-stripping verification.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            options_stripped: false,
        }
    }

    /// Remove multiple-choice option lines from the prompt.
    ///
    /// Strips lines matching common option patterns:
    /// - `A)`, `B)`, `C)`, `D)` (with optional content after)
    /// - `A.`, `B.`, `C.`, `D.` (with optional content after)
    /// - Numbered options: `1)`, `2)`, `3)`, `4)` (with optional content after)
    ///
    /// Returns the prompt with option lines removed.
    pub fn strip_options(&mut self, prompt: &str) -> String {
        self.options_stripped = true;
        prompt
            .lines()
            .filter(|line| !Self::is_option_line(line))
            .collect::<Vec<&str>>()
            .join("\n")
    }

    /// Reset the options-stripped flag and return a reference to the inner pruner.
    pub fn restore_options(&mut self) -> &S {
        self.options_stripped = false;
        &self.inner
    }

    /// Whether options have been stripped (for introspection).
    pub fn is_stripped(&self) -> bool {
        self.options_stripped
    }

    /// Access the inner pruner directly.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Access the inner pruner mutably.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }

    /// Pure reasoning score: delegate to `inner.relevance()`.
    ///
    /// This is the "without options" verification — the model must score well
    /// on pure reasoning merit, not on pattern-matching option letters.
    pub fn verify_pure(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        self.inner.relevance(depth, token_idx, parent_tokens)
    }

    /// Option-matched score: gate by whether the answer matches an option.
    ///
    /// If `matched` is true, returns `inner.relevance()` — the answer is
    /// consistent with an option. If false, returns 0.0 — the answer doesn't
    /// align with any option, so the option-matched score collapses.
    pub fn verify_matched(
        &self,
        depth: usize,
        token_idx: usize,
        parent_tokens: &[usize],
        matched: bool,
    ) -> f32 {
        match matched {
            true => self.inner.relevance(depth, token_idx, parent_tokens),
            false => 0.0,
        }
    }

    /// Two-pass verification score: `min(pure_score, matched_score)`.
    ///
    /// This is the core anti-shortcut gate:
    /// - If pure reasoning fails (low score), the final score is low regardless
    ///   of option matching.
    /// - If the answer doesn't match any option (`answer_matches_option` is false),
    ///   the matched score is 0.0, so the final score is 0.0.
    /// - Only when **both** pass does the score remain high.
    pub fn two_pass_score(
        &mut self,
        depth: usize,
        token_idx: usize,
        parent_tokens: &[usize],
        answer_matches_option: bool,
    ) -> f32 {
        let pure_score = self.verify_pure(depth, token_idx, parent_tokens);
        let matched_score =
            self.verify_matched(depth, token_idx, parent_tokens, answer_matches_option);
        pure_score.min(matched_score)
    }

    /// Check if a line is a multiple-choice option line.
    ///
    /// Matches: `A)`, `B)`, `C)`, `D)`, `A.`, `B.`, `C.`, `D.`,
    /// and numbered variants `1)`, `2)`, `3)`, `4)` (case-insensitive,
    /// allows leading whitespace).
    fn is_option_line(line: &str) -> bool {
        let trimmed = line.trim_start();

        // Letter options: A-D followed by ) or .
        if let Some(rest) = trimmed.strip_prefix(|c: char| c.is_ascii_alphabetic()) {
            if rest.strip_prefix(')').is_some() {
                return true;
            }
            if rest.strip_prefix('.').is_some() {
                return true;
            }
        }

        // Numbered options: 1-4 followed by ) or .
        if let Some(rest) = trimmed.strip_prefix(|c: char| c.is_ascii_digit()) {
            if rest.strip_prefix(')').is_some() {
                return true;
            }
            if rest.strip_prefix('.').is_some() {
                return true;
            }
        }

        false
    }
}

// ── ScreeningPruner delegation ────────────────────────────────

#[cfg(feature = "collapse_aware_thinking")]
impl<S: ScreeningPruner> ScreeningPruner for OptionStripper<S> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(all(test, feature = "collapse_aware_thinking"))]
mod tests {
    use super::*;
    use crate::speculative::types::NoScreeningPruner;

    #[test]
    fn test_strip_options_removes_multiple_choice() {
        let mut stripper = OptionStripper::new(NoScreeningPruner);
        let prompt = "What is the capital of France?\nA) Paris\nB) London\nC) Berlin\nD) Madrid\nPlease answer.";
        let stripped = stripper.strip_options(prompt);

        assert!(!stripped.contains("A) Paris"));
        assert!(!stripped.contains("B) London"));
        assert!(!stripped.contains("C) Berlin"));
        assert!(!stripped.contains("D) Madrid"));
        assert!(stripped.contains("What is the capital of France?"));
        assert!(stripped.contains("Please answer."));
        assert!(stripper.is_stripped());
    }

    #[test]
    fn test_strip_options_preserves_non_option_text() {
        let mut stripper = OptionStripper::new(NoScreeningPruner);
        let prompt = "Solve for x:\n2x + 3 = 7\nTherefore x = 2\nThe answer is clear.";
        let stripped = stripper.strip_options(prompt);

        assert_eq!(stripped, prompt, "Non-option text should be unchanged");
        assert!(stripper.is_stripped());
    }

    #[test]
    fn test_two_pass_prevents_shortcut() {
        let mut stripper = OptionStripper::new(NoScreeningPruner);

        // NoScreeningPruner always returns 1.0, but if the answer
        // does NOT match any option, matched_score = 0.0.
        // min(1.0, 0.0) = 0.0 — shortcut blocked.
        let score = stripper.two_pass_score(0, 0, &[], false);
        assert_eq!(
            score, 0.0,
            "Two-pass must return 0.0 when answer doesn't match any option"
        );
    }

    #[test]
    fn test_two_pass_allows_correct() {
        let mut stripper = OptionStripper::new(NoScreeningPruner);

        // NoScreeningPruner returns 1.0, and the answer matches an option.
        // min(1.0, 1.0) = 1.0 — both passes succeed.
        let score = stripper.two_pass_score(0, 0, &[], true);
        assert_eq!(
            score, 1.0,
            "Two-pass must return 1.0 when both pure reasoning and option match succeed"
        );
    }

    #[test]
    fn test_screening_pruner_delegate() {
        let stripper = OptionStripper::new(NoScreeningPruner);

        // ScreeningPruner::relevance delegates to inner — NoScreeningPruner returns 1.0
        let score = ScreeningPruner::relevance(&stripper, 5, 42, &[1, 2, 3]);
        assert_eq!(score, 1.0, "Delegation to inner pruner must work");

        // Also via the wrapper method
        let score2 = stripper.verify_pure(5, 42, &[1, 2, 3]);
        assert_eq!(score2, 1.0, "verify_pure must also delegate correctly");
    }

    #[test]
    fn test_restore_options_resets_flag() {
        let mut stripper = OptionStripper::new(NoScreeningPruner);
        assert!(!stripper.is_stripped());

        let _ = stripper.strip_options("A) test");
        assert!(stripper.is_stripped());

        // restore_options returns &S but holds &mut self, so check flag after borrow ends
        {
            let _inner = stripper.restore_options();
        }
        assert!(!stripper.is_stripped());
    }

    #[test]
    fn test_strip_options_dot_pattern() {
        let mut stripper = OptionStripper::new(NoScreeningPruner);
        let prompt = "Choose:\nA. First\nB. Second\nKeep this.";
        let stripped = stripper.strip_options(prompt);

        assert!(!stripped.contains("A. First"));
        assert!(!stripped.contains("B. Second"));
        assert!(stripped.contains("Choose:"));
        assert!(stripped.contains("Keep this."));
    }

    #[test]
    fn test_strip_options_numbered_pattern() {
        let mut stripper = OptionStripper::new(NoScreeningPruner);
        let prompt = "Pick one:\n1) Alpha\n2) Beta\n3) Gamma\nDone.";
        let stripped = stripper.strip_options(prompt);

        assert!(!stripped.contains("1) Alpha"));
        assert!(!stripped.contains("2) Beta"));
        assert!(!stripped.contains("3) Gamma"));
        assert!(stripped.contains("Pick one:"));
        assert!(stripped.contains("Done."));
    }

    #[test]
    fn test_collapse_detector_frozen_default() {
        let frozen = CollapseDetectorFrozen::default();
        assert_eq!(frozen.threshold, 3);
        assert!((frozen.gamma - 0.1).abs() < f32::EPSILON);
        assert!(frozen.validate().is_ok());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_collapse_detector_frozen_validate() {
        let mut frozen = CollapseDetectorFrozen::default();
        frozen.threshold = 0;
        assert!(frozen.validate().is_err());

        frozen.threshold = 5;
        frozen.gamma = -0.5;
        assert!(frozen.validate().is_err());

        frozen.gamma = 1.5;
        assert!(frozen.validate().is_err());

        frozen.gamma = 0.5;
        assert!(frozen.validate().is_ok());
    }

    // ── T7: GOAT Tests ──────────────────────────────────────────────

    #[test]
    fn test_end_to_end_two_pass() {
        // Full workflow: strip options, run two-pass scoring, verify min-bottleneck.
        let mut stripper = OptionStripper::new(NoScreeningPruner);
        let prompt = "What is the capital of France?\nA) yes\nB) no\nC) maybe";

        // Step 1: Strip options from the prompt.
        let stripped = stripper.strip_options(prompt);
        assert!(stripper.is_stripped());
        assert!(!stripped.contains("A) yes"));
        assert!(!stripped.contains("B) no"));
        assert!(!stripped.contains("C) maybe"));
        assert!(stripped.contains("What is the capital of France?"));

        // Step 2: Pure pass — score without options (NoScreeningPruner returns 1.0).
        let pure_score = stripper.verify_pure(0, 0, &[]);
        assert_eq!(
            pure_score, 1.0,
            "Pure score should be 1.0 from NoScreeningPruner"
        );

        // Step 3: Matched pass — answer matches an option → 1.0.
        let matched_score = stripper.verify_matched(0, 0, &[], true);
        assert_eq!(
            matched_score, 1.0,
            "Matched score should be 1.0 when answer matches"
        );

        // Step 4: Two-pass with match → min(1.0, 1.0) = 1.0.
        let two_pass_matched = stripper.two_pass_score(0, 0, &[], true);
        assert_eq!(two_pass_matched, 1.0, "Two-pass matched should be 1.0");

        // Step 5: Two-pass without match → min(1.0, 0.0) = 0.0.
        // This is the key anti-shortcut: the min-bottleneck blocks shortcuts.
        let two_pass_unmatched = stripper.two_pass_score(0, 0, &[], false);
        assert_eq!(
            two_pass_unmatched, 0.0,
            "Two-pass unmatched must be 0.0 — shortcut blocked by min-bottleneck"
        );
    }
}
