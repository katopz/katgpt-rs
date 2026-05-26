//! Answer extraction strategies for parallel-probe speculative decoding (Plan 133 T2).
//!
//! Provides a generic [`AnswerExtractor`] trait and several concrete implementations for
//! pulling structured answers out of raw token sequences. These extractors feed into the
//! [`ParallelProbeController`](super::parallel_probe::ParallelProbeController) for consensus-based
//! early stopping and branch pruning.
//!
//! ## Design constraints
//!
//! - **Zero external deps** — uses only `std` string methods, no regex crate.
//! - **Stateless** — extractors are lightweight and carry no mutable state.
//! - **Composable** — callers can chain extractors or swap strategies per domain.

// ── Trait ──────────────────────────────────────────────────────

/// Trait for extracting a structured answer from a decoded text sequence.
///
/// Implementations parse the raw decoded text and return the first recognizable answer,
/// or `None` if no answer pattern is found.
pub trait AnswerExtractor: Send + Sync {
    /// Extract an answer from the decoded `text`.
    ///
    /// `tokens` is the raw token-id sequence (available for token-level extractors),
    /// `text` is the decoded string. Most implementations only need `text`.
    fn extract_answer(&self, tokens: &[usize], text: &str) -> Option<String>;
}

// ── RegexAnswerExtractor ───────────────────────────────────────

/// Pattern-based answer extractor that recognises common answer formats.
///
/// Supported patterns (checked in priority order):
///
/// 1. LaTeX `\boxed{...}` — extracts content inside braces.
/// 2. `"The answer is ..."` — captures text after the prefix until end-of-line or period.
/// 3. Numeric patterns — standalone integers, decimals, and fractions like `3/4`.
///
/// All matching is case-insensitive where applicable and uses only `std` string methods
/// (no regex crate).
///
/// # Examples
///
/// ```
/// use katgpt_rs::speculative::answer_extract::{AnswerExtractor, RegexAnswerExtractor};
///
/// let ext = RegexAnswerExtractor;
/// assert_eq!(ext.extract_answer(&[], "The result is \\boxed{42}"), Some("42".to_string()));
/// assert_eq!(ext.extract_answer(&[], "The answer is 3.14"), Some("3.14".to_string()));
/// ```
#[derive(Clone, Debug, Default)]
pub struct RegexAnswerExtractor;

impl RegexAnswerExtractor {
    /// Create a new `RegexAnswerExtractor`.
    pub fn new() -> Self {
        Self
    }

    /// Try to extract the content of `\boxed{...}`.
    ///
    /// Handles nested braces one level deep (e.g. `\boxed{\frac{1}{2}}`).
    fn extract_boxed(text: &str) -> Option<String> {
        // Look for \boxed{ — find the opening brace and match closing brace.
        let needle = "\\boxed{";
        let start = text.find(needle)?;
        let content_start = start + needle.len();

        // Simple brace-counting: track depth so nested braces work.
        let mut depth = 1usize;
        let mut content_end = content_start;
        for ch in text[content_start..].chars() {
            if ch == '{' {
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            content_end += ch.len_utf8();
        }

        if depth == 0 {
            Some(text[content_start..content_end].trim().to_string())
        } else {
            // Unmatched braces — take to end of line as fallback.
            let line_end = text[content_start..]
                .find('\n')
                .map(|i| content_start + i)
                .unwrap_or(text.len());
            Some(text[content_start..line_end].trim().to_string())
        }
    }

    /// Try to extract the value after "the answer is" (case-insensitive).
    fn extract_answer_is(text: &str) -> Option<String> {
        let lower = text.to_ascii_lowercase();
        // Try several common phrasings.
        let prefixes = ["the answer is ", "the answer is:", "answer: ", "answer:"];
        for prefix in &prefixes {
            if let Some(pos) = lower.find(prefix) {
                let value_start = pos + prefix.len();
                if value_start >= text.len() {
                    continue;
                }
                let remainder = &text[value_start..];
                // Take until newline or sentence-ending period (a `.` NOT followed by a digit).
                let end = Self::find_sentence_end(remainder).unwrap_or(remainder.len());
                let candidate = remainder[..end].trim().trim_matches(',').trim();
                if !candidate.is_empty() {
                    return Some(candidate.to_string());
                }
            }
        }
        None
    }

    /// Find the index of the first sentence-ending period in `text`.
    ///
    /// A period that is part of a decimal number (followed by a digit) is NOT treated as
    /// a sentence end.
    fn find_sentence_end(text: &str) -> Option<usize> {
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'.' {
                // Not a sentence end if followed by a digit (decimal point).
                let next_is_digit = i + 1 < bytes.len() && (bytes[i + 1] as char).is_ascii_digit();
                if !next_is_digit {
                    return Some(i);
                }
            } else if bytes[i] == b'\n' {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// Try to extract a standalone numeric value (integer, decimal, fraction, or negative).
    fn extract_numeric(text: &str) -> Option<String> {
        // Walk through the text looking for a numeric token.
        // A numeric token starts with a digit or `-` followed by a digit.
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let ch = bytes[i] as char;
            // Look for start of number: digit, or minus sign followed by digit.
            if ch.is_ascii_digit()
                || (ch == '-' && i + 1 < bytes.len() && (bytes[i + 1] as char).is_ascii_digit())
            {
                let start = i;
                let mut end = i;

                // Consume optional sign.
                if (bytes[end] as char) == '-' {
                    end += 1;
                }

                // Consume integer part.
                while end < bytes.len() && (bytes[end] as char).is_ascii_digit() {
                    end += 1;
                }

                // Optional decimal part.
                if end < bytes.len() && bytes[end] == b'.' {
                    let dot_pos = end;
                    end += 1;
                    while end < bytes.len() && (bytes[end] as char).is_ascii_digit() {
                        end += 1;
                    }
                    // If no digits after dot, this was a sentence-ending period — back up.
                    if end == dot_pos + 1 {
                        end = dot_pos;
                    }
                }

                // Optional fraction part: /digits
                if end < bytes.len() && bytes[end] == b'/' {
                    let frac_start = end;
                    end += 1;
                    if end < bytes.len() && (bytes[end] as char).is_ascii_digit() {
                        while end < bytes.len() && (bytes[end] as char).is_ascii_digit() {
                            end += 1;
                        }
                    } else {
                        end = frac_start; // Not a fraction — back up.
                    }
                }

                // Check word boundary: the char before and after should not be alphanumeric.
                let before_ok = start == 0 || !(bytes[start - 1] as char).is_ascii_alphanumeric();
                let after_ok = end >= bytes.len() || !(bytes[end] as char).is_ascii_alphanumeric();

                if before_ok && after_ok && end > start {
                    let num = &text[start..end];
                    // Must contain at least one digit.
                    if num.chars().any(|c| c.is_ascii_digit()) {
                        return Some(num.to_string());
                    }
                }

                i = end;
            } else {
                i += 1;
            }
        }
        None
    }
}

impl AnswerExtractor for RegexAnswerExtractor {
    fn extract_answer(&self, _tokens: &[usize], text: &str) -> Option<String> {
        // Priority 1: \boxed{...}
        if let Some(answer) = Self::extract_boxed(text) {
            return Some(answer);
        }
        // Priority 2: "The answer is ..."
        if let Some(answer) = Self::extract_answer_is(text) {
            return Some(answer);
        }
        // Priority 3: standalone numeric.
        Self::extract_numeric(text)
    }
}

// ── ThinkTokenExtractor ────────────────────────────────────────

/// Extracts the text after the `</think` boundary produced by reasoning models.
///
/// Many reasoning-capable models emit a chain-of-thought block enclosed in
/// `<think⟩...⟨/think>` (or variants like `</think`), then produce the final answer
/// after the closing tag. This extractor returns everything after the **last** closing
/// tag, trimmed.
///
/// # Examples
///
/// ```
/// use katgpt_rs::speculative::answer_extract::{AnswerExtractor, ThinkTokenExtractor};
///
/// let ext = ThinkTokenExtractor;
/// let text = "<think\nstep 1...\n</think\n\n42";
/// assert_eq!(ext.extract_answer(&[], text), Some("42".to_string()));
/// ```
#[derive(Clone, Debug, Default)]
pub struct ThinkTokenExtractor;

impl ThinkTokenExtractor {
    /// Create a new `ThinkTokenExtractor`.
    pub fn new() -> Self {
        Self
    }

    /// Closing tag variants to search for (most common first).
    const CLOSING_TAGS: &'static [&'static str] = &["</think", "<｜end▁of▁thinking｜", "</thought"];
}

impl AnswerExtractor for ThinkTokenExtractor {
    fn extract_answer(&self, _tokens: &[usize], text: &str) -> Option<String> {
        // Find the last occurrence of any closing tag.
        let mut best_end = 0usize; // Position *after* the tag.
        let mut found = false;

        for tag in Self::CLOSING_TAGS {
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find(tag) {
                let abs_start = search_from + pos;
                let abs_end = abs_start + tag.len();
                if abs_end >= best_end {
                    best_end = abs_end;
                    found = true;
                }
                search_from = abs_end;
            }
        }

        if !found {
            return None;
        }

        // Content starts after the tag — skip optional `>` and whitespace.
        let remainder = &text[best_end..];
        let content = remainder.trim_start_matches('>').trim_start();

        if content.is_empty() {
            return None;
        }

        // Take up to the first double newline or end — the "answer" portion.
        let end = content.find("\n\n").unwrap_or(content.len());
        let answer = content[..end].trim();
        if answer.is_empty() {
            None
        } else {
            Some(answer.to_string())
        }
    }
}

// ── DiscreteActionExtractor ────────────────────────────────────

/// Extracts discrete action indices from game/reinforcement-learning domains.
///
/// Looks for patterns like `Action: 3`, `action=7`, or standalone small integers
/// bounded by `max_actions`. Useful when the model is playing a game with a fixed
/// action space and the answer is simply which action to take.
///
/// # Examples
///
/// ```
/// use katgpt_rs::speculative::answer_extract::{AnswerExtractor, DiscreteActionExtractor};
///
/// let ext = DiscreteActionExtractor::new(9);
/// assert_eq!(ext.extract_answer(&[], "I choose action 5"), Some("5".to_string()));
/// ```
#[derive(Clone, Debug)]
pub struct DiscreteActionExtractor {
    /// Maximum number of actions (exclusive upper bound).
    /// Extracted values must be in `0..max_actions`.
    pub max_actions: usize,
}

impl DiscreteActionExtractor {
    /// Create a new extractor for an action space of size `max_actions`.
    ///
    /// Only integers in `[0, max_actions)` are accepted as valid actions.
    pub fn new(max_actions: usize) -> Self {
        Self { max_actions }
    }
}

impl AnswerExtractor for DiscreteActionExtractor {
    fn extract_answer(&self, _tokens: &[usize], text: &str) -> Option<String> {
        let lower = text.to_ascii_lowercase();

        // Priority 1: explicit "action: N" or "action=N" patterns.
        let action_prefixes = ["action: ", "action=", "move: ", "move="];
        for prefix in &action_prefixes {
            if let Some(pos) = lower.find(prefix) {
                let value_start = pos + prefix.len();
                let remainder = &text[value_start..];
                if let Some(action) = Self::parse_action_int(remainder, self.max_actions) {
                    return Some(action.to_string());
                }
            }
        }

        // Priority 2: find the last integer in text that fits in action space.
        // "Last" because game trajectories often list several moves before the final one.
        let mut last_valid: Option<usize> = None;
        let bytes = lower.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if (bytes[i] as char).is_ascii_digit() {
                let start = i;
                while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                    i += 1;
                }
                // Check word boundary.
                let before_ok = start == 0 || !(bytes[start - 1] as char).is_ascii_alphanumeric();
                let after_ok = i >= bytes.len() || !(bytes[i] as char).is_ascii_alphanumeric();
                if before_ok && after_ok
                    && let Ok(val) = lower[start..i].parse::<usize>()
                        && val < self.max_actions {
                            last_valid = Some(val);
                        }
            } else {
                i += 1;
            }
        }

        last_valid.map(|v| v.to_string())
    }
}

impl DiscreteActionExtractor {
    /// Parse the first integer from `text` that fits in `[0, max_actions)`.
    fn parse_action_int(text: &str, max_actions: usize) -> Option<usize> {
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if (bytes[i] as char).is_ascii_digit() {
                let start = i;
                while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                    i += 1;
                }
                if let Ok(val) = text[start..i].parse::<usize>()
                    && val < max_actions {
                        return Some(val);
                    }
            } else {
                i += 1;
            }
        }
        None
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RegexAnswerExtractor ──────────────────────────────────

    #[test]
    fn test_boxed_simple() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "The result is \\boxed{42}"),
            Some("42".to_string())
        );
    }

    #[test]
    fn test_boxed_expression() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "\\boxed{\\frac{1}{2}}"),
            Some("\\frac{1}{2}".to_string())
        );
    }

    #[test]
    fn test_boxed_nested_braces() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "\\boxed{x^{2} + 1}"),
            Some("x^{2} + 1".to_string())
        );
    }

    #[test]
    fn test_answer_is_pattern() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "After calculation, the answer is 17"),
            Some("17".to_string())
        );
    }

    #[test]
    fn test_answer_is_colon() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "Answer: 3.14"),
            Some("3.14".to_string())
        );
    }

    #[test]
    fn test_answer_is_case_insensitive() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "THE ANSWER IS: -5"),
            Some("-5".to_string())
        );
    }

    #[test]
    fn test_numeric_integer() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "The value is 42."),
            Some("42".to_string())
        );
    }

    #[test]
    fn test_numeric_decimal() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "Result: 3.14159 end"),
            Some("3.14159".to_string())
        );
    }

    #[test]
    fn test_numeric_fraction() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "We get 3/4 as the answer"),
            Some("3/4".to_string())
        );
    }

    #[test]
    fn test_numeric_negative() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "Temperature: -10 degrees"),
            Some("-10".to_string())
        );
    }

    #[test]
    fn test_no_answer_found() {
        let ext = RegexAnswerExtractor;
        assert_eq!(ext.extract_answer(&[], "hello world no numbers here"), None);
    }

    #[test]
    fn test_empty_text() {
        let ext = RegexAnswerExtractor;
        assert_eq!(ext.extract_answer(&[], ""), None);
    }

    #[test]
    fn test_boxed_priority_over_numeric() {
        let ext = RegexAnswerExtractor;
        // The number 99 appears first, but \boxed{7} should take priority.
        assert_eq!(
            ext.extract_answer(&[], "99 is wrong, \\boxed{7} is correct"),
            Some("7".to_string())
        );
    }

    #[test]
    fn test_answer_is_priority_over_numeric() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "100 birds, the answer is 50"),
            Some("50".to_string())
        );
    }

    #[test]
    fn test_multiple_answers_picks_first_pattern() {
        let ext = RegexAnswerExtractor;
        // \boxed{} takes priority over "the answer is".
        assert_eq!(
            ext.extract_answer(&[], "\\boxed{A} and the answer is B"),
            Some("A".to_string())
        );
    }

    #[test]
    fn test_boxed_trailing_content() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "\\boxed{hello} more text"),
            Some("hello".to_string())
        );
    }

    #[test]
    fn test_answer_is_with_period() {
        let ext = RegexAnswerExtractor;
        assert_eq!(
            ext.extract_answer(&[], "The answer is 42. Done."),
            Some("42".to_string())
        );
    }

    // ── ThinkTokenExtractor ───────────────────────────────────

    #[test]
    fn test_think_basic() {
        let ext = ThinkTokenExtractor;
        assert_eq!(
            ext.extract_answer(&[], "<think\nreasoning...\n</think\n\n42"),
            Some("42".to_string())
        );
    }

    #[test]
    fn test_think_no_tag() {
        let ext = ThinkTokenExtractor;
        assert_eq!(ext.extract_answer(&[], "No think tag here"), None);
    }

    #[test]
    fn test_think_empty_after_tag() {
        let ext = ThinkTokenExtractor;
        assert_eq!(ext.extract_answer(&[], "<think\n...\n</think\n\n   "), None);
    }

    #[test]
    fn test_think_uses_last_tag() {
        let ext = ThinkTokenExtractor;
        let text = "<think\nfirst\n</think\nwrong\n<think\nsecond\n</think\n\ncorrect";
        assert_eq!(ext.extract_answer(&[], text), Some("correct".to_string()));
    }

    #[test]
    fn test_think_multiline_answer() {
        let ext = ThinkTokenExtractor;
        let text = "</think\n\nThe answer is 42\n\nExplanation follows";
        // Should stop at double newline.
        assert_eq!(
            ext.extract_answer(&[], text),
            Some("The answer is 42".to_string())
        );
    }

    #[test]
    fn test_think_empty_text() {
        let ext = ThinkTokenExtractor;
        assert_eq!(ext.extract_answer(&[], ""), None);
    }

    // ── DiscreteActionExtractor ───────────────────────────────

    #[test]
    fn test_action_explicit() {
        let ext = DiscreteActionExtractor::new(9);
        assert_eq!(
            ext.extract_answer(&[], "I choose action: 5"),
            Some("5".to_string())
        );
    }

    #[test]
    fn test_action_equals() {
        let ext = DiscreteActionExtractor::new(9);
        assert_eq!(
            ext.extract_answer(&[], "Best action=3 for this state"),
            Some("3".to_string())
        );
    }

    #[test]
    fn test_action_implicit_last_valid() {
        let ext = DiscreteActionExtractor::new(9);
        // Multiple numbers: 10 is out of range, 7 and 3 are valid.
        // Should pick the last valid one.
        assert_eq!(
            ext.extract_answer(&[], "Compare 10, 7, and 3"),
            Some("3".to_string())
        );
    }

    #[test]
    fn test_action_out_of_range() {
        let ext = DiscreteActionExtractor::new(5);
        assert_eq!(ext.extract_answer(&[], "Action: 9"), None);
    }

    #[test]
    fn test_action_zero() {
        let ext = DiscreteActionExtractor::new(3);
        assert_eq!(ext.extract_answer(&[], "action=0"), Some("0".to_string()));
    }

    #[test]
    fn test_action_no_number() {
        let ext = DiscreteActionExtractor::new(9);
        assert_eq!(ext.extract_answer(&[], "No numbers here"), None);
    }

    #[test]
    fn test_action_boundary_exclusive() {
        let ext = DiscreteActionExtractor::new(3);
        // 3 is NOT in [0, 3).
        assert_eq!(ext.extract_answer(&[], "action: 3"), None);
        assert_eq!(ext.extract_answer(&[], "action: 2"), Some("2".to_string()));
    }
}
