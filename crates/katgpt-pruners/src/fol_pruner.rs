//! FOL Constraint Extraction — Modelless Prompt→Constraint Pipeline (Plan 209, Phase 1).
//!
//! Parses Rust prompts into first-order logic constraints without using an LLM.
//! Uses a static keyword→token index lookup table and wraps any inner `ConstraintPruner`.
//!
//! # Architecture
//!
//! ```text
//! Prompt → extract_fol_constraints(prompt, vocab) → Vec<FolConstraint>
//!                                                      │
//!                                            FolPruner { inner, constraints }
//!                                                      │
//!                                            DDTree → FolPruner.is_valid()
//!                                                      ├── check constraints first
//!                                                      └── delegate to inner
//! ```
//!
//! Zero cost on miss path (empty constraints → inner only).
//! Feature-gated behind `fol_constraints`.

use katgpt_speculative::ConstraintPruner;

// ── FolConstraint ──────────────────────────────────────────────────

/// A first-order logic constraint extracted from a prompt.
///
/// Constrains token choices at specific depth ranges based on keyword
/// patterns detected in the prompt text (e.g., "async function", "no unsafe").
#[derive(Clone, Debug)]
pub struct FolConstraint {
    /// Position range where constraint applies [start, end).
    pub depth_range: (usize, usize),
    /// Token indices allowed at these positions.
    pub allowed: Vec<usize>,
    /// Token indices disallowed at these positions.
    pub disallowed: Vec<usize>,
    /// Confidence in this constraint [0, 1].
    pub confidence: f32,
}

// ── Static Keyword Table ──────────────────────────────────────────

/// Prompt pattern → vocab token strings to search for.
///
/// Each entry maps a regex-like prompt pattern to vocab token strings.
/// During extraction, prompt is scanned for the pattern; matching tokens
/// are resolved to their indices in the vocabulary.
///
/// Format: `(prompt_pattern, &[vocab_token_string])`
/// - Positive patterns: tokens are added to `allowed`.
/// - Negation patterns (prefix "no "): tokens are added to `disallowed`.
/// - Type patterns (prefix "returns "): tokens for the type are added to `allowed`.
static RUST_KEYWORD_TABLE: &[(&str, &[&str])] = &[
    // ── Keywords ──
    ("async function", &["async", "fn"]),
    ("async fn", &["async", "fn"]),
    ("pub async fn", &["pub", "async", "fn"]),
    ("pub fn", &["pub", "fn"]),
    ("fn ", &["fn"]),
    ("function", &["fn"]),
    ("struct ", &["struct"]),
    ("enum ", &["enum"]),
    ("impl ", &["impl"]),
    ("trait ", &["trait"]),
    ("mod ", &["mod"]),
    ("use ", &["use"]),
    ("const ", &["const"]),
    ("static ", &["static"]),
    ("let ", &["let"]),
    ("mut ", &["mut"]),
    ("match ", &["match"]),
    ("if ", &["if"]),
    ("else ", &["else"]),
    ("loop ", &["loop"]),
    ("while ", &["while"]),
    ("for ", &["for"]),
    ("return ", &["return"]),
    ("break ", &["break"]),
    ("continue ", &["continue"]),
    ("where ", &["where"]),
    ("type ", &["type"]),
    ("pub struct", &["pub", "struct"]),
    ("pub enum", &["pub", "enum"]),
    ("pub trait", &["pub", "trait"]),
    ("pub impl", &["pub", "impl"]),
    ("pub mod", &["pub", "mod"]),
    // ── Types ──
    ("returns Result", &["Result"]),
    ("returns Option", &["Option"]),
    ("returns Vec", &["Vec"]),
    ("returns String", &["String"]),
    ("returns bool", &["bool"]),
    ("returns i32", &["i32"]),
    ("returns u32", &["u32"]),
    ("returns f64", &["f64"]),
    ("returns usize", &["usize"]),
    ("Result<", &["Result", "Ok", "Err"]),
    ("Option<", &["Option", "Some", "None"]),
    ("Vec<", &["Vec"]),
    ("String", &["String"]),
    ("HashMap", &["HashMap"]),
    ("BTreeMap", &["BTreeMap"]),
    // ── Negation patterns ──
    ("no unsafe", &["unsafe"]),
    ("no panic", &["panic", "unwrap", "expect"]),
    ("no unwrap", &["unwrap"]),
    ("no expect", &["expect"]),
    ("no clone", &["clone"]),
    ("no copy", &["copy"]),
    ("no mut", &["mut"]),
    // ── Attribute patterns ──
    ("#[derive(", &["derive"]),
    ("#[test]", &["test"]),
    ("#[cfg(", &["cfg"]),
    ("#[inline]", &["inline"]),
    // ── Error handling ──
    ("? operator", &["?", "try"]),
    ("error handling", &["Result", "Ok", "Err", "?"]),
    // ── Async patterns ──
    ("async move", &["async", "move"]),
    (".await", &["await"]),
    ("tokio", &["tokio", "async", "await"]),
    ("spawn", &["spawn"]),
    // ── Lifetime/Generics ──
    ("lifetime", &["'", "lifetime"]),
    ("generic", &["<", ">"]),
    // ── Visibility ──
    ("private", &[]),
    ("crate visibility", &["crate"]),
    ("super", &["super"]),
];

// ── Extraction ─────────────────────────────────────────────────────

/// Extract FOL constraints from a Rust prompt by keyword matching.
///
/// Scans the prompt against the static keyword table, resolves matching
/// token strings to their indices in the vocabulary, and produces
/// `FolConstraint` entries.
///
/// - "async function" → allowed tokens matching async/fn keywords
/// - "returns Result<T,E>" → constraint for Result-related tokens
/// - "no unsafe" → disallowed token for unsafe
///
/// Returns empty vec for empty prompts (zero alloc on miss path).
pub fn extract_fol_constraints(prompt: &str, vocab: &[String]) -> Vec<FolConstraint> {
    if prompt.is_empty() || vocab.is_empty() {
        return Vec::new();
    }

    let prompt_lower = prompt.to_ascii_lowercase();
    let mut constraints = Vec::new();

    for &(pattern, token_strings) in RUST_KEYWORD_TABLE {
        let pattern_lower = pattern.to_ascii_lowercase();

        match prompt_lower.contains(&pattern_lower) {
            false => continue,
            true => {
                let is_negation = pattern.starts_with("no ");
                let resolved = resolve_token_indices(token_strings, vocab);

                match resolved.is_empty() {
                    true => continue,
                    false => {
                        let confidence = compute_confidence(pattern, &prompt_lower);

                        constraints.push(FolConstraint {
                            depth_range: (0, usize::MAX),
                            allowed: match is_negation {
                                true => Vec::new(),
                                false => resolved.clone(),
                            },
                            disallowed: match is_negation {
                                true => resolved,
                                false => Vec::new(),
                            },
                            confidence,
                        });
                    }
                }
            }
        }
    }

    constraints
}

/// Resolve token strings to their indices in the vocabulary.
///
/// Performs case-sensitive matching. Returns all matching indices.
/// Pre-allocated with capacity based on input size.
fn resolve_token_indices(token_strings: &[&str], vocab: &[String]) -> Vec<usize> {
    let mut indices = Vec::with_capacity(token_strings.len());

    for &token_str in token_strings {
        for (idx, vocab_token) in vocab.iter().enumerate() {
            match vocab_token == token_str {
                true => {
                    indices.push(idx);
                    break; // first match only per token string
                }
                false => continue,
            }
        }
    }

    indices
}

/// Compute confidence score for a matched pattern.
///
/// Longer/more specific patterns get higher confidence.
/// Uses sigmoid of pattern length as a smooth confidence function.
fn compute_confidence(pattern: &str, _prompt: &str) -> f32 {
    let len = pattern.len() as f32;
    // Sigmoid: σ(x) = 1 / (1 + exp(-x))
    // Use pattern length scaled so ~10 chars → ~0.7 confidence
    let x = (len - 5.0) * 0.5;
    1.0 / (1.0 + (-x).exp())
}

// ── FolPruner ──────────────────────────────────────────────────────

/// FOL constraint pruner — wraps inner pruner with keyword-extracted constraints.
///
/// Applies first-order logic constraints extracted from the prompt to prune
/// DDTree branch candidates. Falls back to inner pruner alone when no
/// constraints are active (zero-cost miss path).
///
/// Feature-gated behind `fol_constraints`.
#[cfg(feature = "fol_constraints")]
pub struct FolPruner<P: ConstraintPruner> {
    /// Inner pruner (base structural validity).
    inner: P,
    /// Extracted FOL constraints.
    constraints: Vec<FolConstraint>,
}

#[cfg(feature = "fol_constraints")]
impl<P: ConstraintPruner> FolPruner<P> {
    /// Create a new `FolPruner` wrapping `inner` with the given constraints.
    pub fn new(inner: P, constraints: Vec<FolConstraint>) -> Self {
        Self { inner, constraints }
    }

    /// Create a `FolPruner` by extracting constraints from a prompt.
    pub fn from_prompt(inner: P, prompt: &str, vocab: &[String]) -> Self {
        let constraints = extract_fol_constraints(prompt, vocab);
        Self { inner, constraints }
    }

    /// Check if a token at a given depth violates any FOL constraint.
    fn is_rejected_by_constraints(&self, depth: usize, token_idx: usize) -> bool {
        for c in &self.constraints {
            match depth >= c.depth_range.0 && depth < c.depth_range.1 {
                false => continue,
                true => {
                    // If allowed is non-empty, token must be in it
                    if !c.allowed.is_empty() && !c.allowed.contains(&token_idx) {
                        return true;
                    }
                    // If token is explicitly disallowed, reject
                    if c.disallowed.contains(&token_idx) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

#[cfg(feature = "fol_constraints")]
impl<P: ConstraintPruner> ConstraintPruner for FolPruner<P> {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Fast path: no constraints → delegate to inner only
        if self.constraints.is_empty() {
            return self.inner.is_valid(depth, token_idx, parent_tokens);
        }

        // Check FOL constraints first
        if self.is_rejected_by_constraints(depth, token_idx) {
            return false;
        }

        // Delegate to inner
        self.inner.is_valid(depth, token_idx, parent_tokens)
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        // Delegate to inner batch first
        self.inner
            .batch_is_valid(depth, candidates, parent_tokens, results);

        // Fast path: no constraints → inner results are final
        if self.constraints.is_empty() {
            return;
        }

        // Apply FOL constraints in batch
        let len = candidates.len().min(results.len());
        for i in 0..len {
            match results[i] {
                false => continue, // already rejected by inner
                true => {
                    if self.is_rejected_by_constraints(depth, candidates[i]) {
                        results[i] = false;
                    }
                }
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Pruner that accepts everything (for testing FOL layer in isolation).
    struct AcceptAllPruner;

    impl ConstraintPruner for AcceptAllPruner {
        fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
            true
        }
    }

    /// Build a minimal Rust-like vocab for testing.
    fn test_vocab() -> Vec<String> {
        vec![
            "async".into(),
            "fn".into(),
            "pub".into(),
            "struct".into(),
            "unsafe".into(),
            "Result".into(),
            "Option".into(),
            "Ok".into(),
            "Err".into(),
            "Some".into(),
            "None".into(),
            "match".into(),
            "let".into(),
            "mut".into(),
            "return".into(),
            "unwrap".into(),
            "expect".into(),
            "panic".into(),
            "clone".into(),
            "Vec".into(),
            "?".into(),
            "await".into(),
            "move".into(),
            "impl".into(),
            "enum".into(),
            "trait".into(),
            "use".into(),
            "mod".into(),
            "where".into(),
            "test".into(),
        ]
    }

    // ── extract_fol_constraints tests ──

    #[test]
    fn test_empty_prompt_zero_constraints() {
        let vocab = test_vocab();
        let constraints = extract_fol_constraints("", &vocab);
        assert!(constraints.is_empty());
    }

    #[test]
    fn test_empty_vocab_zero_constraints() {
        let constraints = extract_fol_constraints("async fn foo() {}", &[]);
        assert!(constraints.is_empty());
    }

    #[test]
    fn test_async_function_extracts_keywords() {
        let vocab = test_vocab();
        let constraints =
            extract_fol_constraints("Write an async function that processes data", &vocab);

        // Should find "async function" → allowed: [async_idx, fn_idx]
        let async_fn_constraint = constraints
            .iter()
            .find(|c| !c.allowed.is_empty() && c.allowed.contains(&0) && c.allowed.contains(&1));

        match async_fn_constraint {
            Some(c) => {
                assert!(c.allowed.contains(&0)); // async
                assert!(c.allowed.contains(&1)); // fn
                assert!(c.confidence > 0.0);
            }
            None => panic!("expected async/fn constraint not found"),
        }
    }

    #[test]
    fn test_no_unsafe_produces_negation() {
        let vocab = test_vocab();
        let constraints = extract_fol_constraints("safe Rust: no unsafe code allowed", &vocab);

        // Should find "no unsafe" → disallowed: [unsafe_idx]
        let negation = constraints.iter().find(|c| !c.disallowed.is_empty());

        match negation {
            Some(c) => {
                assert!(c.disallowed.contains(&4)); // unsafe
                assert!(c.allowed.is_empty());
            }
            None => panic!("expected negation constraint for 'unsafe' not found"),
        }
    }

    #[test]
    fn test_returns_result_extracts_type_tokens() {
        let vocab = test_vocab();
        let constraints = extract_fol_constraints(
            "Write a function returns Result<T, E> for error handling",
            &vocab,
        );

        // Should find "returns Result" and "Result<" and "error handling"
        let has_result = constraints.iter().any(|c| c.allowed.contains(&5)); // Result
        assert!(has_result, "expected Result token in allowed constraints");
    }

    #[test]
    fn test_no_panic_produces_negation() {
        let vocab = test_vocab();
        let constraints = extract_fol_constraints("Robust code: no panic, no unwrap", &vocab);

        // "no panic" → disallowed: [panic, unwrap, expect]
        let no_panic = constraints
            .iter()
            .find(|c| c.disallowed.contains(&15)) // unwrap is index 15
            .or_else(|| constraints.iter().find(|c| c.disallowed.contains(&16))); // expect is index 16

        assert!(
            no_panic.is_some(),
            "expected negation constraint for panic/unwrap"
        );
    }

    #[test]
    fn test_unrelated_prompt_minimal_constraints() {
        let vocab = test_vocab();
        let constraints = extract_fol_constraints("The weather is nice today", &vocab);
        // No Rust keywords matched → zero constraints
        assert!(constraints.is_empty());
    }

    #[test]
    fn test_confidence_increases_with_pattern_length() {
        let short_conf = compute_confidence("fn ", "");
        let long_conf = compute_confidence("async function", "");
        // Longer patterns should have higher confidence (sigmoid of length)
        assert!(
            long_conf > short_conf,
            "longer pattern should have higher confidence: {long_conf} vs {short_conf}"
        );
    }

    // ── FolPruner integration tests ──

    #[cfg(feature = "fol_constraints")]
    #[test]
    fn test_fol_pruner_delegates_to_inner_when_no_constraints() {
        let inner = AcceptAllPruner;
        let pruner = FolPruner::new(inner, Vec::new());

        // No constraints → inner accepts everything
        assert!(pruner.is_valid(0, 42, &[]));
        assert!(pruner.is_valid(5, 0, &[1, 2, 3, 4, 5]));
        assert!(pruner.is_valid(100, 999, &[]));
    }

    #[cfg(feature = "fol_constraints")]
    #[test]
    fn test_fol_pruner_rejects_disallowed_tokens() {
        let inner = AcceptAllPruner;
        let constraints = vec![FolConstraint {
            depth_range: (0, usize::MAX),
            allowed: Vec::new(),
            disallowed: vec![4], // unsafe
            confidence: 0.9,
        }];
        let pruner = FolPruner::new(inner, constraints);

        // Inner accepts everything, but FOL constraint disallows token 4
        assert!(!pruner.is_valid(0, 4, &[])); // unsafe → rejected
        assert!(pruner.is_valid(0, 0, &[])); // async → allowed
        assert!(pruner.is_valid(0, 1, &[])); // fn → allowed
    }

    #[cfg(feature = "fol_constraints")]
    #[test]
    fn test_fol_pruner_allows_only_whitelisted() {
        let inner = AcceptAllPruner;
        let constraints = vec![FolConstraint {
            depth_range: (0, usize::MAX),
            allowed: vec![0, 1], // async, fn only
            disallowed: Vec::new(),
            confidence: 0.8,
        }];
        let pruner = FolPruner::new(inner, constraints);

        // Only tokens 0 and 1 are allowed at any depth
        assert!(pruner.is_valid(0, 0, &[])); // async → allowed
        assert!(pruner.is_valid(0, 1, &[])); // fn → allowed
        assert!(!pruner.is_valid(0, 2, &[])); // pub → rejected (not in allowed)
        assert!(!pruner.is_valid(0, 3, &[])); // struct → rejected
    }

    #[cfg(feature = "fol_constraints")]
    #[test]
    fn test_fol_pruner_respects_depth_range() {
        let inner = AcceptAllPruner;
        let constraints = vec![FolConstraint {
            depth_range: (2, 5), // only applies at depths [2, 5)
            allowed: vec![0],
            disallowed: Vec::new(),
            confidence: 0.7,
        }];
        let pruner = FolPruner::new(inner, constraints);

        // Outside range → no constraint applied, inner accepts
        assert!(pruner.is_valid(0, 99, &[])); // depth 0 → no constraint
        assert!(pruner.is_valid(1, 99, &[])); // depth 1 → no constraint
        assert!(pruner.is_valid(5, 99, &[])); // depth 5 → outside range

        // Inside range → constraint applied
        assert!(pruner.is_valid(2, 0, &[])); // depth 2, token 0 → allowed
        assert!(!pruner.is_valid(2, 99, &[])); // depth 2, token 99 → not in allowed
        assert!(!pruner.is_valid(4, 99, &[])); // depth 4, token 99 → not in allowed
    }

    #[cfg(feature = "fol_constraints")]
    #[test]
    fn test_fol_pruner_batch_is_valid() {
        let inner = AcceptAllPruner;
        let constraints = vec![FolConstraint {
            depth_range: (0, usize::MAX),
            allowed: Vec::new(),
            disallowed: vec![4], // unsafe
            confidence: 0.9,
        }];
        let pruner = FolPruner::new(inner, constraints);

        let candidates = vec![0, 1, 4, 3, 4, 5];
        let mut results = vec![true; 6];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);

        assert!(results[0]); // async → allowed
        assert!(results[1]); // fn → allowed
        assert!(!results[2]); // unsafe → rejected
        assert!(results[3]); // struct → allowed
        assert!(!results[4]); // unsafe → rejected
        assert!(results[5]); // Result → allowed
    }

    // ── GOAT Proof: Constraint Extraction Accuracy ≥80% (Plan 209, T5.2) ──

    #[test]
    fn goat_constraint_extraction_accuracy() {
        // Corpus: (prompt, expected_allowed_keywords)
        // Keywords must exist in test_vocab() for resolution.
        let corpus: &[(&str, &[&str])] = &[
            ("async fn", &["async", "fn"]),
            ("pub async fn", &["pub", "async", "fn"]),
            ("fn main", &["fn"]),
            ("struct Foo", &["struct"]),
            ("enum Bar", &["enum"]),
            ("impl Display", &["impl"]),
            ("trait Send", &["trait"]),
            ("match x", &["match"]),
            ("if let Some", &["if", "let"]),
            ("no unsafe", &["unsafe"]), // negation → disallowed
            ("pub struct Config", &["pub", "struct"]),
            (
                "async function returning Result",
                &["async", "fn", "Result"],
            ),
            ("impl Iterator", &["impl"]),
            ("where T: Clone", &["where"]),
            ("pub enum Error", &["pub", "enum"]),
            ("async move", &["async", "move"]),
            ("const MAX", &["const"]),
            ("type Alias", &["type"]),
            ("use std", &["use"]),
            ("mod tests", &["mod"]),
            ("pub trait", &["pub", "trait"]),
            ("no unsafe code", &["unsafe"]), // negation
            ("Result<T, E>", &["Result", "Ok", "Err"]),
            ("Option<T>", &["Option", "Some", "None"]),
            ("Vec<String>", &["Vec"]),
            ("no panic", &["panic", "unwrap", "expect"]),
            ("error handling", &["Result", "Ok", "Err", "?"]),
            ("pub fn new", &["pub", "fn"]),
            ("fn default", &["fn"]),
            ("else branch", &["else"]),
            ("mut x", &["mut"]),
            ("return value", &["return"]),
            ("let x", &["let"]),
            ("impl FromStr", &["impl"]),
            ("trait IntoIterator", &["trait"]),
            ("pub async fn connect", &["pub", "async", "fn"]),
            ("struct Config { verbose: bool }", &["struct"]),
            ("enum Command", &["enum"]),
            ("fn clone", &["fn"]),
            ("no unwrap", &["unwrap"]),
            ("no expect", &["expect"]),
            ("no clone", &["clone"]),
            ("#[test]", &["test"]),
            ("HashMap", &[]), // not in test_vocab → no constraint
            ("private", &[]), // pattern maps to empty tokens
            ("tokio runtime", &["async", "await"]),
            ("spawn task", &["spawn"]), // not in vocab → resolves empty
            ("fn with_capacity", &["fn"]),
            ("pub impl", &["pub", "impl"]),
            ("pub mod", &["pub", "mod"]),
            ("", &[]), // empty → no constraints
        ];

        let vocab = test_vocab();
        let mut correct = 0;
        let total = corpus.len();

        for (prompt, expected) in corpus {
            let constraints = extract_fol_constraints(prompt, &vocab);

            if expected.is_empty() {
                // Expect no constraints or all with empty allowed+disallowed
                let any_meaningful = constraints
                    .iter()
                    .any(|c| !c.allowed.is_empty() || !c.disallowed.is_empty());
                if !any_meaningful {
                    correct += 1;
                }
            } else {
                // Check each expected keyword appears in some constraint's allowed or disallowed
                let all_found = expected.iter().all(|kw| {
                    let idx = vocab.iter().position(|v| v == *kw);
                    idx.is_some_and(|i| {
                        constraints
                            .iter()
                            .any(|c| c.allowed.contains(&i) || c.disallowed.contains(&i))
                    })
                });
                if all_found {
                    correct += 1;
                }
            }
        }

        let accuracy = correct as f32 / total as f32;
        assert!(
            accuracy >= 0.80,
            "constraint extraction accuracy {:.0}% < 80% ({}/{})",
            accuracy * 100.0,
            correct,
            total
        );
    }

    #[cfg(feature = "fol_constraints")]
    #[test]
    fn test_fol_pruner_from_prompt() {
        let vocab = test_vocab();
        let pruner = FolPruner::from_prompt(AcceptAllPruner, "no unsafe code here", &vocab);

        // Should have at least one constraint disallowing unsafe (index 4)
        assert!(!pruner.is_valid(0, 4, &[])); // unsafe → rejected
        assert!(pruner.is_valid(0, 0, &[])); // async → allowed
    }
}
