//! Observation verification for speculative hop predictions.
//!
//! Implements the verification layer that compares target tool observations
//! against speculative predictions. Based on paper Appendix D.4:
//! - Exact match (after normalization)
//! - Short-answer exact match
//! - Numeric consistency
//! - Substring match
//! - Token-set Jaccard similarity ≥ 0.55
//! - Refusal pattern detection

use std::collections::HashSet;
use std::sync::LazyLock;

/// Trait for verifying speculative observations against target observations.
///
/// Paper Section 4: when the target tool returns, the verifier checks
/// equivalence → `true` = commit branch, `false` = rollback.
pub trait ObservationVerifier: Send + Sync {
    /// Verify whether speculative observation matches target.
    ///
    /// Returns `true` if observations are equivalent (commit branch),
    /// `false` if they differ (rollback branch).
    fn verify(&self, o_target: &str, o_spec: &str) -> bool;
}

/// Common refusal patterns indicating the tool/model declined to answer.
const REFUSAL_PATTERNS: &[&str] = &[
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "sorry",
    "as an ai",
    "as a language model",
    "i apologize",
    "not able to",
    "against my",
];

/// Common English stopwords to remove before computing Jaccard similarity.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any", "few", "more", "most", "other",
    "some", "such", "no", "only", "own", "same", "than", "too", "very", "just", "because", "if",
    "then", "else", "when", "where", "why", "how", "what", "which", "who", "whom", "this", "that",
    "these", "those", "it", "its", "he", "she", "they", "them", "we", "you", "i", "me", "my",
    "your", "his", "her", "our", "their",
];

/// Pre-computed HashSet of stopwords for O(1) lookups instead of O(n) linear scan.
static STOPWORD_SET: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| STOPWORDS.iter().copied().collect());

/// Rule-based verifier implementing paper Appendix D.4 heuristics.
///
/// Checks (in order of decreasing cost, early-exit on pass):
/// 1. Exact match after normalization
/// 2. Short-answer exact match (< 10 chars)
/// 3. Refusal pattern detection (both refused → match, one refused → mismatch)
/// 4. Numeric consistency (extract all numbers, compare sets)
/// 5. Substring containment
/// 6. Token-set Jaccard similarity ≥ 0.55
#[derive(Clone, Debug)]
pub struct RuleBasedVerifier {
    /// Minimum Jaccard similarity threshold for paraphrase matching.
    /// Paper default: 0.55.
    pub jaccard_threshold: f64,
    /// Maximum length for "short answer" exact matching.
    /// Paper default: 10 characters.
    pub short_answer_max_len: usize,
}

impl Default for RuleBasedVerifier {
    fn default() -> Self {
        Self {
            jaccard_threshold: 0.55,
            short_answer_max_len: 10,
        }
    }
}

impl ObservationVerifier for RuleBasedVerifier {
    fn verify(&self, o_target: &str, o_spec: &str) -> bool {
        let target_norm = normalize(o_target);
        let spec_norm = normalize(o_spec);

        // Rule 1: Exact match after normalization
        if target_norm == spec_norm {
            return true;
        }

        // Rule 2: Short-answer exact match
        if target_norm.len() <= self.short_answer_max_len
            && spec_norm.len() <= self.short_answer_max_len
        {
            return target_norm == spec_norm;
        }

        // Rule 3: Refusal pattern detection
        let target_refused = is_refusal(&target_norm);
        let spec_refused = is_refusal(&spec_norm);
        match (target_refused, spec_refused) {
            (true, true) => return true, // Both refused → equivalent
            (true, false) | (false, true) => return false, // One refused, one didn't → mismatch
            (false, false) => {}         // Neither refused → continue checks
        }

        // Rule 4: Numeric consistency
        if numeric_consistent(&target_norm, &spec_norm) {
            return true;
        }

        // Rule 5: Substring containment
        if target_norm.contains(&spec_norm) || spec_norm.contains(&target_norm) {
            return true;
        }

        // Rule 6: Token-set Jaccard similarity
        let jaccard = token_set_jaccard(&target_norm, &spec_norm);
        jaccard >= self.jaccard_threshold
    }
}

/// Normalize text for comparison: lowercase, collapse whitespace, trim.
fn normalize(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_whitespace() {
                ' '
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Check whether text matches a refusal pattern.
fn is_refusal(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    REFUSAL_PATTERNS
        .iter()
        .any(|pattern| lower.contains(pattern))
}

/// Extract all numeric values (integers and decimals) from text.
fn extract_numbers(text: &str) -> Vec<String> {
    let mut numbers = Vec::new();
    let mut current = String::new();
    let mut has_digit = false;

    for ch in text.chars() {
        match ch {
            '0'..='9' => {
                current.push(ch);
                has_digit = true;
            }
            '.' if has_digit => {
                current.push(ch);
            }
            '-' if current.is_empty() => {
                current.push(ch);
            }
            _ => {
                if has_digit {
                    numbers.push(current.clone());
                }
                current.clear();
                has_digit = false;
            }
        }
    }
    if has_digit {
        numbers.push(current);
    }
    numbers
}

/// Check numeric consistency: both texts have the same set of numbers.
fn numeric_consistent(target: &str, spec: &str) -> bool {
    let target_nums = extract_numbers(target);
    let spec_nums = extract_numbers(spec);

    if target_nums.is_empty() && spec_nums.is_empty() {
        return false; // No numbers to compare → not a numeric match
    }

    // Both must have at least one number, and sets must be equal
    if target_nums.is_empty() || spec_nums.is_empty() {
        return false;
    }

    // Check that all numbers in target appear in spec and vice versa
    let mut target_sorted = target_nums;
    let mut spec_sorted = spec_nums;
    target_sorted.sort();
    spec_sorted.sort();
    target_sorted == spec_sorted
}

/// Tokenize text into a set of words (for Jaccard computation).
/// Returns a `HashSet<&str>` for O(1) membership checks in Jaccard.
fn tokenize_set<'a>(text: &'a str, scratch: &mut HashSet<&'a str>) {
    scratch.clear();
    text.split_whitespace()
        .filter(|word| !STOPWORD_SET.contains(word))
        .for_each(|word| {
            scratch.insert(word);
        });
}

/// Compute token-set Jaccard similarity between two normalized texts.
///
/// J(A, B) = |A ∩ B| / |A ∪ B|
///
/// Stopwords are removed before computing to focus on content words.
pub fn token_set_jaccard(target: &str, spec: &str) -> f64 {
    let mut target_set = HashSet::new();
    tokenize_set(target, &mut target_set);
    let mut spec_set = HashSet::new();
    tokenize_set(spec, &mut spec_set);

    if target_set.is_empty() && spec_set.is_empty() {
        return 1.0; // Both empty → identical
    }
    if target_set.is_empty() || spec_set.is_empty() {
        return 0.0; // One empty, one not → no overlap
    }

    let intersection = target_set.intersection(&spec_set).count();
    let union = target_set.len() + spec_set.len() - intersection;
    if union == 0 {
        return 1.0;
    }

    intersection as f64 / union as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── T10: Unit tests ─────────────────────────────────────────

    fn verifier() -> RuleBasedVerifier {
        RuleBasedVerifier::default()
    }

    #[test]
    fn test_identical_observations_true() {
        assert!(verifier().verify("The answer is 42", "The answer is 42"));
        assert!(verifier().verify("hello world", "hello world"));
    }

    #[test]
    fn test_different_numbers_false() {
        // Different numbers → numeric inconsistency → no other match → false
        assert!(!verifier().verify("The answer is 42", "The answer is 43"));
    }

    #[test]
    fn test_paraphrased_true() {
        // Paraphrased with enough token overlap → Jaccard ≥ 0.55
        let target = "Paris is the capital city of France located in Europe";
        let spec = "Paris is the capital of France in Europe";
        assert!(verifier().verify(target, spec));
    }

    #[test]
    fn test_short_answer_mismatch_false() {
        assert!(!verifier().verify("yes", "no"));
        assert!(!verifier().verify("42", "43"));
        assert!(!verifier().verify("true", "false"));
    }

    #[test]
    fn test_refusal_pattern_false() {
        // One is a refusal, the other is not → mismatch
        assert!(!verifier().verify("I cannot answer that question", "The answer is 42"));
        assert!(!verifier().verify("The answer is 42", "I apologize but I cannot help"));
    }

    #[test]
    fn test_both_refusals_true() {
        // Both are refusals → equivalent
        assert!(verifier().verify("I cannot answer that", "I'm unable to help with that"));
    }

    #[test]
    fn test_case_insensitive_match() {
        assert!(verifier().verify("The Answer Is 42", "the answer is 42"));
        assert!(verifier().verify("HELLO WORLD", "hello world"));
    }

    #[test]
    fn test_whitespace_normalization() {
        assert!(verifier().verify("The   answer   is   42", "The answer is 42"));
        assert!(verifier().verify("  hello  world  ", "hello world"));
    }

    #[test]
    fn test_substring_match() {
        assert!(verifier().verify("The capital of France is Paris", "capital of France"));
    }

    #[test]
    fn test_numeric_consistency_same_numbers() {
        assert!(verifier().verify("Result: 3.14", "Got 3.14"));
        assert!(verifier().verify(
            "Coordinates 40.7128 -74.0060",
            "Location at -74.0060 40.7128"
        ));
    }

    // ── Normalize ───────────────────────────────────────────────

    #[test]
    fn test_normalize_collapse_whitespace() {
        assert_eq!(normalize("  hello   world  "), "hello world");
    }

    #[test]
    fn test_normalize_lowercase() {
        assert_eq!(normalize("HELLO World"), "hello world");
    }

    #[test]
    fn test_normalize_empty() {
        assert_eq!(normalize(""), "");
        assert_eq!(normalize("   "), "");
    }

    // ── Refusal detection ───────────────────────────────────────

    #[test]
    fn test_is_refusal_positive() {
        assert!(is_refusal("I cannot answer"));
        assert!(is_refusal("I apologize but..."));
        assert!(is_refusal("as an AI, I..."));
    }

    #[test]
    fn test_is_refusal_negative() {
        assert!(!is_refusal("The answer is 42"));
        assert!(!is_refusal("Paris is the capital of France"));
    }

    // ── Number extraction ───────────────────────────────────────

    #[test]
    fn test_extract_numbers_simple() {
        assert_eq!(extract_numbers("42"), vec!["42"]);
        assert_eq!(extract_numbers("3.14"), vec!["3.14"]);
    }

    #[test]
    fn test_extract_numbers_multiple() {
        let nums = extract_numbers("x=42 y=3.14 z=-7");
        assert!(nums.contains(&"42".to_string()));
        assert!(nums.contains(&"3.14".to_string()));
    }

    #[test]
    fn test_extract_numbers_none() {
        assert!(extract_numbers("no numbers here").is_empty());
    }

    // ── Token-set Jaccard ───────────────────────────────────────

    #[test]
    fn test_jaccard_identical() {
        let j = token_set_jaccard("hello world", "hello world");
        assert!((j - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_no_overlap() {
        let j = token_set_jaccard("alpha beta", "gamma delta");
        assert!((j).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_partial() {
        // "paris capital france" vs "paris capital europe"
        // stopword-free: {paris, capital, france} ∩ {paris, capital, europe} = 2
        // union = 3 + 3 - 2 = 4, J = 2/4 = 0.5
        let j = token_set_jaccard(
            "paris is the capital of france",
            "paris is the capital of europe",
        );
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_both_empty() {
        let j = token_set_jaccard("", "");
        assert!((j - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_stopwords_removed() {
        // "a the is" → empty after stopword removal
        let j = token_set_jaccard("a the is", "a the is");
        assert!((j - 1.0).abs() < 1e-10);
    }
}
