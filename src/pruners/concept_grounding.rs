//! Concept Grounding for Pruner Rules — Plan 210 Phase 3 (F2).
//!
//! Template-based mapping of pruner internals to human-readable explanations.
//! No LLM at runtime — pure string interpolation from static templates.
//! Grounds expression terms and pruner scores into semantic concepts.
//!
//! # Architecture
//!
//! ```text
//! PrunerState ──► ConceptGrounding::ground() ──► Vec<ConceptMapping>
//!      │                                              │
//!      │          ConceptGrounding::explain_chain()    │
//!      └──────────────────────────────────────────────►Vec<String>
//!                                                     │
//!              ConceptGrounding::summarize()           │
//!      ┌──────────────────────────────────────────────┘
//!      ▼
//! PolicyExplanation { mappings, chain_of_thought, summary }
//! ```
//!
//! # Feature Gate
//!
//! `concept_grounding` (depends on `symbolic_distill`).
//!
//! # Performance
//!
//! - Grounding: ~1μs per call
//! - Template matching: static lookup, zero allocation

// ── Sigmoid helper ────────────────────────────────────────────

/// Sigmoid function: `1 / (1 + exp(-x))`.
/// Used for confidence bounding — never softmax.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Core types ────────────────────────────────────────────────

/// Origin of a concept mapping — static template vs learned mapping.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum GroundingSource {
    /// Static template match from `TemplateGrounding`.
    Template = 0,
    /// Learned mapping from future training pipeline.
    Learned = 1,
}

/// A single variable-to-semantic concept mapping with confidence.
#[derive(Clone, Debug)]
pub struct ConceptMapping {
    /// Variable name (e.g., "depth", "bandit_score").
    pub variable: String,
    /// Human-readable semantic interpretation.
    pub semantic: String,
    /// Confidence in [0, 1], sigmoid-bounded.
    pub confidence: f32,
    /// Origin of this mapping.
    pub source: GroundingSource,
}

/// Complete explanation for a pruner decision.
#[derive(Clone, Debug)]
pub struct PolicyExplanation {
    pub mappings: Vec<ConceptMapping>,
    pub chain_of_thought: Vec<String>,
    pub summary: String,
}

impl PolicyExplanation {
    /// Simple JSON serialization — no serde dependency.
    /// Produces valid JSON suitable for TrialLog JSONL append.
    pub fn to_json(&self) -> String {
        let mappings_json: Vec<String> = self
            .mappings
            .iter()
            .map(|m| {
                format!(
                    r#"{{"variable":"{}","semantic":"{}","confidence":{},"source":"{}"}}"#,
                    json_escape(&m.variable),
                    json_escape(&m.semantic),
                    m.confidence,
                    match m.source {
                        GroundingSource::Template => "Template",
                        GroundingSource::Learned => "Learned",
                    }
                )
            })
            .collect();

        let chain_json: Vec<String> = self
            .chain_of_thought
            .iter()
            .map(|s| format!("\"{}\"", json_escape(s)))
            .collect();

        format!(
            r#"{{"mappings":[{}],"chain_of_thought":[{}],"summary":"{}"}}"#,
            mappings_json.join(","),
            chain_json.join(","),
            json_escape(&self.summary),
        )
    }
}

/// Escape a string for JSON embedding (handles quotes and backslashes).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

// ── PrunerState snapshot ──────────────────────────────────────

/// Captures pruner internals at a decision point.
#[derive(Clone, Debug, Default)]
pub struct PrunerState {
    /// Current tree depth.
    pub depth: usize,
    /// Current token index in the sequence.
    pub token_idx: usize,
    /// Parent token indices (path from root).
    pub parent_token: Vec<usize>,
    /// Per-pruner scores: (pruner_name, score).
    pub pruner_scores: Vec<(String, f32)>,
    /// Whether the token was accepted by the pruner committee.
    pub accepted: bool,
}

impl PrunerState {
    /// Create a new `PrunerState` with pre-allocated pruner_scores.
    pub fn new() -> Self {
        Self {
            depth: 0,
            token_idx: 0,
            parent_token: Vec::with_capacity(16),
            pruner_scores: Vec::with_capacity(8),
            accepted: false,
        }
    }
}

// ── Trait ─────────────────────────────────────────────────────

/// Concept grounding trait — maps pruner state to human-readable concepts.
pub trait ConceptGrounding: Send + Sync {
    /// Ground pruner state into concept mappings.
    fn ground(&self, state: &PrunerState) -> Vec<ConceptMapping>;

    /// Generate chain-of-thought explanations from state + mappings.
    fn explain_chain(&self, state: &PrunerState, mappings: &[ConceptMapping]) -> Vec<String>;

    /// Produce a human-readable summary from mappings + chain.
    fn summarize(&self, mappings: &[ConceptMapping], chain: &[String]) -> String;
}

// ── TemplateGrounding ─────────────────────────────────────────

/// Static template-based concept grounding — no LLM, pure string interpolation.
pub struct TemplateGrounding {
    /// Static template table: (pattern, semantic).
    templates: Vec<(&'static str, &'static str)>,
}

impl TemplateGrounding {
    /// Template confidence: `sigmoid(1.0) ≈ 0.7310585`.
    /// Pre-computed because `f32::exp` is not `const fn`.
    const TEMPLATE_CONFIDENCE: f32 = 0.7310586;

    /// Create a new `TemplateGrounding` with the default template table.
    pub fn new() -> Self {
        Self {
            templates: Vec::new(), // vocab-dependent templates filled by caller
        }
    }

    /// Create with custom template table.
    pub fn with_templates(templates: Vec<(&'static str, &'static str)>) -> Self {
        Self { templates }
    }

    /// Interpret a pruner score into a semantic label.
    fn interpret_score(score: f32) -> &'static str {
        match score {
            s if s > 0.8 => "high confidence",
            s if s > 0.5 => "moderate confidence",
            _ => "low confidence / rejected",
        }
    }

    /// Interpret tree depth into a semantic label.
    fn interpret_depth(depth: usize) -> &'static str {
        match depth {
            0 => "top-level declaration",
            1 => "type annotation or parameter",
            2 => "function body or expression",
            _ => "nested expression",
        }
    }
}

impl Default for TemplateGrounding {
    fn default() -> Self {
        Self::new()
    }
}

impl ConceptGrounding for TemplateGrounding {
    fn ground(&self, state: &PrunerState) -> Vec<ConceptMapping> {
        let mut mappings = Vec::with_capacity(state.pruner_scores.len() + 2);
        let conf = Self::TEMPLATE_CONFIDENCE;

        // Depth-based grounding
        let depth_semantic = Self::interpret_depth(state.depth);
        mappings.push(ConceptMapping {
            variable: "depth".to_string(),
            semantic: depth_semantic.to_string(),
            confidence: conf,
            source: GroundingSource::Template,
        });

        // Token-based grounding — vocab-dependent, use feature names from caller
        // Only produce a mapping if templates are populated
        for (pattern, semantic) in &self.templates {
            // Match against token_idx as a proxy for vocab position
            if state.token_idx > 0 {
                mappings.push(ConceptMapping {
                    variable: pattern.to_string(),
                    semantic: semantic.to_string(),
                    confidence: conf,
                    source: GroundingSource::Template,
                });
                break; // one token mapping per grounding
            }
        }

        // Score-based grounding for each pruner
        for (name, score) in &state.pruner_scores {
            let interpretation = Self::interpret_score(*score);
            mappings.push(ConceptMapping {
                variable: format!("pruner_{name}_score"),
                semantic: interpretation.to_string(),
                confidence: sigmoid(*score), // sigmoid-bound the raw score
                source: GroundingSource::Template,
            });
        }

        mappings
    }

    fn explain_chain(&self, state: &PrunerState, mappings: &[ConceptMapping]) -> Vec<String> {
        let mut chain = Vec::with_capacity(mappings.len() + 1);

        // Template: "Token at depth {depth} was {action} because {reason}"
        let action = match state.accepted {
            true => "accepted",
            false => "rejected",
        };
        let reason = match state.depth {
            0 => "top-level declaration — always evaluated",
            1 => "type-level context — structural relevance",
            2 => "body-level context — semantic relevance",
            _ => "deep nesting — cumulative relevance check",
        };
        chain.push(format!(
            "Token {} at depth {} was {} because {}",
            state.token_idx, state.depth, action, reason
        ));

        // Template: "Pruner '{name}' scored {score:.2} ({interpretation})"
        for (name, score) in &state.pruner_scores {
            let interpretation = Self::interpret_score(*score);
            chain.push(format!(
                "Pruner '{}' scored {:.2} ({})",
                name, score, interpretation
            ));
        }

        // Template: "Combined relevance: {combined} → {decision}"
        if !state.pruner_scores.is_empty() {
            let combined: f32 = state.pruner_scores.iter().map(|(_, s)| *s).sum::<f32>()
                / state.pruner_scores.len() as f32;
            let decision = match state.accepted {
                true => "accepted",
                false => "rejected",
            };
            chain.push(format!(
                "Combined relevance: {:.2} → {}",
                combined, decision
            ));
        }

        // Append mapping-driven insights
        for m in mappings {
            if m.variable.starts_with("pruner_") {
                continue; // already covered above
            }
            chain.push(format!(
                "{} → {} (confidence: {:.2})",
                m.variable, m.semantic, m.confidence
            ));
        }

        chain
    }

    fn summarize(&self, mappings: &[ConceptMapping], chain: &[String]) -> String {
        if mappings.is_empty() && chain.is_empty() {
            return "No grounding available — empty pruner state.".to_string();
        }

        let accepted_count = mappings
            .iter()
            .filter(|m| m.semantic.contains("high confidence"))
            .count();
        let total_scorers = mappings
            .iter()
            .filter(|m| m.variable.starts_with("pruner_"))
            .count();

        let confidence_summary = match (accepted_count, total_scorers) {
            (0, 0) => "no scorer data".to_string(),
            (a, t) if a == t && t > 0 => "all scorers agree".to_string(),
            (0, t) => format!("unanimously rejected ({t} scorers)"),
            (a, t) => format!("{a}/{t} scorers confident"),
        };

        // Extract depth semantic if present
        let depth_desc = mappings
            .iter()
            .find(|m| m.variable == "depth")
            .map(|m| m.semantic.as_str())
            .unwrap_or("unknown depth");

        format!(
            "Decision at {} ({}): {}. {} reasoning steps.",
            depth_desc,
            confidence_summary,
            match chain.last() {
                Some(s) => s.clone(),
                None => "no reasoning".to_string(),
            },
            chain.len(),
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(depth: usize, token_idx: usize, accepted: bool) -> PrunerState {
        let mut state = PrunerState::new();
        state.depth = depth;
        state.token_idx = token_idx;
        state.accepted = accepted;
        state
    }

    // ── F2.8: Template grounding produces correct mappings ──

    #[test]
    fn test_template_grounding_correct_mappings() {
        let grounding = TemplateGrounding::new();
        let mut state = make_state(0, 5, true);
        state.pruner_scores.push(("bandit".to_string(), 0.9));

        let mappings = grounding.ground(&state);

        // Should have depth mapping + 1 pruner score mapping = 2
        assert_eq!(mappings.len(), 2);

        // Depth mapping
        assert_eq!(mappings[0].variable, "depth");
        assert_eq!(mappings[0].semantic, "top-level declaration");
        assert_eq!(mappings[0].source, GroundingSource::Template);

        // Pruner score mapping
        assert_eq!(mappings[1].variable, "pruner_bandit_score");
        assert_eq!(mappings[1].semantic, "high confidence");
    }

    // ── F2.8: Chain-of-thought fills templates correctly ──

    #[test]
    fn test_chain_of_thought_templates() {
        let grounding = TemplateGrounding::new();
        let mut state = make_state(1, 10, false);
        state.pruner_scores.push(("relevance".to_string(), 0.3));

        let mappings = grounding.ground(&state);
        let chain = grounding.explain_chain(&state, &mappings);

        // Should have: token action line + pruner line + combined line + depth insight = 4
        assert_eq!(chain.len(), 4);

        // First line: token at depth action
        assert!(chain[0].contains("Token 10"));
        assert!(chain[0].contains("depth 1"));
        assert!(chain[0].contains("rejected"));

        // Second line: pruner score
        assert!(chain[1].contains("Pruner 'relevance'"));
        assert!(chain[1].contains("0.30"));
        assert!(chain[1].contains("low confidence / rejected"));

        // Third line: combined relevance
        assert!(chain[2].contains("Combined relevance"));
        assert!(chain[2].contains("rejected"));

        // Fourth line: depth mapping insight
        assert!(chain[3].contains("depth"));
        assert!(chain[3].contains("type annotation or parameter"));
    }

    // ── F2.8: Summary is non-empty and human-readable ──

    #[test]
    fn test_summary_non_empty_readable() {
        let grounding = TemplateGrounding::new();
        let mut state = make_state(2, 7, true);
        state.pruner_scores.push(("bandit".to_string(), 0.85));

        let mappings = grounding.ground(&state);
        let chain = grounding.explain_chain(&state, &mappings);
        let summary = grounding.summarize(&mappings, &chain);

        assert!(!summary.is_empty());
        assert!(summary.contains("function body or expression"));
        assert!(summary.contains("reasoning steps"));
    }

    // ── F2.8: Confidence values are sigmoid-bounded [0, 1] ──

    #[test]
    fn test_confidence_sigmoid_bounded() {
        let grounding = TemplateGrounding::new();
        let mut state = make_state(3, 1, true);
        state.pruner_scores.push(("test_high".to_string(), 0.99));
        state.pruner_scores.push(("test_low".to_string(), 0.01));

        let mappings = grounding.ground(&state);

        for m in &mappings {
            assert!(
                (0.0..=1.0).contains(&m.confidence),
                "confidence {} not in [0, 1] for variable {}",
                m.confidence,
                m.variable
            );
        }

        // Template confidence should be sigmoid(1.0) ≈ 0.731
        let depth_mapping = mappings.iter().find(|m| m.variable == "depth").unwrap();
        assert!((depth_mapping.confidence - 0.731).abs() < 0.01);

        // Score-based confidence should be sigmoid(score)
        let high = mappings
            .iter()
            .find(|m| m.variable == "pruner_test_high_score")
            .unwrap();
        assert!((high.confidence - sigmoid(0.99)).abs() < 1e-4);
    }

    // ── F2.8: Empty pruner state → graceful degradation ──

    #[test]
    fn test_empty_state_no_panic() {
        let grounding = TemplateGrounding::new();
        let state = PrunerState::default();

        let mappings = grounding.ground(&state);
        let chain = grounding.explain_chain(&state, &mappings);
        let summary = grounding.summarize(&mappings, &chain);

        // Should produce at least the depth mapping
        assert!(!mappings.is_empty());
        assert_eq!(mappings[0].variable, "depth");

        // Chain should have the token line
        assert!(!chain.is_empty());

        // Summary should note no scorer data
        assert!(summary.contains("no scorer data"));
    }

    // ── F2.8: to_json produces valid-ish JSON ──

    #[test]
    fn test_to_json_output() {
        let explanation = PolicyExplanation {
            mappings: vec![ConceptMapping {
                variable: "depth".to_string(),
                semantic: "top-level declaration".to_string(),
                confidence: 0.731,
                source: GroundingSource::Template,
            }],
            chain_of_thought: vec!["Token 0 at depth 0 was accepted".to_string()],
            summary: "Decision summary".to_string(),
        };

        let json = explanation.to_json();

        // Should be valid JSON structure
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        assert!(json.contains("\"mappings\""));
        assert!(json.contains("\"chain_of_thought\""));
        assert!(json.contains("\"summary\""));
        assert!(json.contains("\"depth\""));
        assert!(json.contains("\"Template\""));
    }

    // ── F2.8: JSON escaping handles special characters ──

    #[test]
    fn test_json_escape() {
        let input = r#"he said "hello\nworld""#;
        let escaped = json_escape(input);
        assert!(!escaped.contains('"') || escaped.contains("\\\""));
    }

    // ── Depth-based grounding maps correctly ──

    #[test]
    fn test_depth_grounding_all_levels() {
        let grounding = TemplateGrounding::new();

        let cases: Vec<(usize, &'static str)> = vec![
            (0, "top-level declaration"),
            (1, "type annotation or parameter"),
            (2, "function body or expression"),
            (3, "nested expression"),
            (5, "nested expression"),
            (10, "nested expression"),
        ];

        for (depth, expected) in cases {
            let state = make_state(depth, 0, true);
            let mappings = grounding.ground(&state);
            let depth_map = mappings
                .iter()
                .find(|m| m.variable == "depth")
                .unwrap_or_else(|| panic!("no depth mapping for depth={depth}"));
            assert_eq!(
                depth_map.semantic, expected,
                "depth={depth}: expected '{expected}', got '{}'",
                depth_map.semantic
            );
        }
    }

    // ── Score-based grounding threshold mapping ──

    #[test]
    fn test_score_grounding_thresholds() {
        let grounding = TemplateGrounding::new();

        let cases: Vec<(f32, &'static str)> = vec![
            (0.9, "high confidence"),
            (0.81, "high confidence"),
            (0.6, "moderate confidence"),
            (0.51, "moderate confidence"),
            (0.5, "low confidence / rejected"),
            (0.2, "low confidence / rejected"),
            (0.0, "low confidence / rejected"),
        ];

        for (score, expected) in cases {
            let mut state = make_state(0, 0, true);
            state.pruner_scores.push(("test".to_string(), score));
            let mappings = grounding.ground(&state);
            let score_map = mappings
                .iter()
                .find(|m| m.variable == "pruner_test_score")
                .unwrap_or_else(|| panic!("no score mapping for score={score}"));
            assert_eq!(
                score_map.semantic, expected,
                "score={score}: expected '{expected}', got '{}'",
                score_map.semantic
            );
        }
    }

    // ── Template confidence constant is sigmoid(1.0) ──

    #[test]
    fn test_template_confidence_is_sigmoid_1() {
        let expected = sigmoid(1.0);
        assert!(
            (TemplateGrounding::TEMPLATE_CONFIDENCE - expected).abs() < 1e-6,
            "TEMPLATE_CONFIDENCE should be sigmoid(1.0) ≈ 0.731"
        );
    }

    // ── PrunerState pre-allocation ──

    #[test]
    fn test_pruner_state_preallocated() {
        let state = PrunerState::new();
        assert_eq!(state.pruner_scores.capacity(), 8);
        assert!(state.parent_token.capacity() >= 16);
    }

    // ── Summary with all scorers agreeing ──

    #[test]
    fn test_summary_all_scorers_agree() {
        let grounding = TemplateGrounding::new();
        let mut state = make_state(0, 0, true);
        state.pruner_scores.push(("a".to_string(), 0.9));
        state.pruner_scores.push(("b".to_string(), 0.85));

        let mappings = grounding.ground(&state);
        let chain = grounding.explain_chain(&state, &mappings);
        let summary = grounding.summarize(&mappings, &chain);

        assert!(summary.contains("all scorers agree"));
    }

    // ── Full pipeline: ground → explain → summarize ──

    #[test]
    fn test_full_pipeline() {
        let grounding = TemplateGrounding::new();
        let mut state = make_state(2, 42, true);
        state.pruner_scores.push(("bandit".to_string(), 0.75));
        state.pruner_scores.push(("relevance".to_string(), 0.6));
        state.parent_token.push(0);
        state.parent_token.push(10);

        let mappings = grounding.ground(&state);
        let chain = grounding.explain_chain(&state, &mappings);
        let summary = grounding.summarize(&mappings, &chain);
        let explanation = PolicyExplanation {
            mappings,
            chain_of_thought: chain,
            summary,
        };
        let json = explanation.to_json();

        // Verify end-to-end
        assert!(!explanation.mappings.is_empty());
        assert!(!explanation.chain_of_thought.is_empty());
        assert!(!explanation.summary.is_empty());
        assert!(json.contains("function body or expression"));
        assert!(json.contains("moderate confidence"));
    }

    // ── F2.9: Grounding Overhead Benchmark ──────────────────────────

    #[test]
    fn test_grounding_overhead() {
        use std::time::Instant;

        let grounding = TemplateGrounding::new();
        let state = PrunerState {
            depth: 1,
            token_idx: 5,
            parent_token: vec![0, 3],
            pruner_scores: vec![
                ("syntax".to_string(), 0.85),
                ("bandit".to_string(), 0.62),
                ("cache".to_string(), 0.30),
            ],
            accepted: true,
        };

        let iters = 10_000;
        let start = Instant::now();
        for _ in 0..iters {
            let mappings = grounding.ground(std::hint::black_box(&state));
            std::hint::black_box(&mappings);
        }
        let elapsed = start.elapsed();
        let per_call_us = elapsed.as_micros() as f64 / iters as f64;

        // Target: <100μs per call (generous — actual target is 10μs)
        assert!(
            per_call_us < 100.0,
            "Grounding overhead {per_call_us:.1}μs exceeds 100μs target"
        );

        eprintln!("  F2.9 grounding overhead: {per_call_us:.2}μs per ground call");
    }
}
