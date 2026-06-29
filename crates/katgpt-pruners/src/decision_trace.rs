//! DecisionTrace — interpretable decision traces from DDTree exploration (Plan 209 T4).
//!
//! Extracts human-readable decision traces showing rules applied, alternatives rejected,
//! and overall confidence. This is an **opt-in debug/audit feature** with no accuracy
//! or performance benefit — it exists solely for transparency.
//!
//! Feature-gated behind `decision_trace` — depends on `rule_extraction`.

use super::rule_extractor::ExtractedRule;

// ── Sigmoid ────────────────────────────────────────────────────────

/// Sigmoid function: `1 / (1 + exp(-x))`. Bounded to (0, 1).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── DecisionTrace ──────────────────────────────────────────────────

/// Human-readable decision trace from DDTree exploration.
///
/// Captures which rules were applied, which alternatives were rejected,
/// and an overall confidence score computed via sigmoid aggregation.
#[derive(Clone, Debug)]
pub struct DecisionTrace {
    /// Rules that were applied (matched conditions).
    pub rules_applied: Vec<ExtractedRule>,
    /// Rules that were considered but not applied.
    pub alternatives_rejected: Vec<ExtractedRule>,
    /// Overall confidence in this decision ∈ [0, 1].
    pub confidence: f32,
}

impl DecisionTrace {
    /// Render a human-readable decision trace.
    ///
    /// Maps token indices to vocab strings where available, shows score
    /// comparisons for rejected alternatives.
    ///
    /// # Example output
    /// ```text
    /// Decision trace: "Chose token at depth 2 because:
    ///   Rule 1: ⟨depth=0, token='fn'⟩ ∧ ⟨depth=1, token='Result'⟩ → depth=2, token='match' (score=0.85, support=5)
    ///   Alternative: ⟨depth=0, token='fn'⟩ ∧ ⟨depth=1, token='Result'⟩ → depth=2, token='if' (score=0.62, rejected: score too low)
    /// Confidence: 0.85"
    /// ```
    pub fn to_string(&self, vocab: &[String]) -> String {
        let mut lines =
            Vec::with_capacity(2 + self.rules_applied.len() + self.alternatives_rejected.len());

        match (
            self.rules_applied.is_empty(),
            self.alternatives_rejected.is_empty(),
        ) {
            (true, true) => {
                lines.push("Decision trace: (no rules applied, no alternatives)".to_string());
            }
            (true, false) => {
                lines.push(
                    "Decision trace: (no rules applied, alternatives were considered)".to_string(),
                );
                for (i, alt) in self.alternatives_rejected.iter().enumerate() {
                    lines.push(format!(
                        "  Alternative {}: {} (score={:.2}, rejected: no rule matched)",
                        i + 1,
                        Self::format_rule(alt, vocab),
                        alt.score,
                    ));
                }
            }
            (false, _) => {
                let action_depth = self.rules_applied[0].action.0;
                let action_token = resolve_token(self.rules_applied[0].action.1, vocab);
                lines.push(format!(
                    "Decision trace: \"Chose token at depth {} ('{}') because:",
                    action_depth, action_token,
                ));
                for (i, rule) in self.rules_applied.iter().enumerate() {
                    lines.push(format!(
                        "  Rule {}: {} (score={:.2}, support={})",
                        i + 1,
                        Self::format_rule(rule, vocab),
                        rule.score,
                        rule.support,
                    ));
                }
                for (i, alt) in self.alternatives_rejected.iter().enumerate() {
                    let best_score = self.rules_applied[0].score;
                    let reason = rejection_reason(best_score, alt.score);
                    lines.push(format!(
                        "  Alternative {}: {} (score={:.2}, rejected: {})",
                        i + 1,
                        Self::format_rule(alt, vocab),
                        alt.score,
                        reason,
                    ));
                }
            }
        }

        lines.push(format!("Confidence: {:.2}", self.confidence));
        lines.join("\n")
    }

    /// Format a single rule as: `⟨depth=D, token='T'⟩ ∧ ... → depth=D, token='T'`
    fn format_rule(rule: &ExtractedRule, vocab: &[String]) -> String {
        let conditions: Vec<String> = rule
            .conditions
            .iter()
            .map(|(depth, tok_idx)| {
                let token = resolve_token(*tok_idx, vocab);
                format!("⟨depth={}, token='{}'⟩", depth, token)
            })
            .collect();

        let action_token = resolve_token(rule.action.1, vocab);
        let action_str = format!("depth={}, token='{}'", rule.action.0, action_token);

        match conditions.is_empty() {
            true => format!("→ {}", action_str),
            false => format!("{} → {}", conditions.join(" ∧ "), action_str),
        }
    }
}

// ── DecisionTraceBuilder ───────────────────────────────────────────

/// Builder for constructing `DecisionTrace` instances.
///
/// Collects applied and rejected rules, then computes confidence as
/// the sigmoid of the mean applied-rule score.
pub struct DecisionTraceBuilder {
    applied: Vec<ExtractedRule>,
    rejected: Vec<ExtractedRule>,
}

impl DecisionTraceBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            applied: Vec::new(),
            rejected: Vec::new(),
        }
    }

    /// Record a rule that was applied (matched conditions).
    pub fn applied(&mut self, rule: ExtractedRule) -> &mut Self {
        self.applied.push(rule);
        self
    }

    /// Record a rule that was considered but rejected.
    pub fn rejected(&mut self, rule: ExtractedRule) -> &mut Self {
        self.rejected.push(rule);
        self
    }

    /// Build the `DecisionTrace`.
    ///
    /// Confidence is computed as `sigmoid(mean_score_of_applied_rules)`.
    /// If no rules were applied, confidence is 0.0.
    /// If exactly one rule was applied, confidence is `sigmoid(rule.score)`.
    /// Otherwise, confidence is `sigmoid(mean(scores))`.
    pub fn build(&self) -> DecisionTrace {
        let confidence = match self.applied.len() {
            0 => 0.0,
            1 => sigmoid(self.applied[0].score),
            _ => {
                let mean_score =
                    self.applied.iter().map(|r| r.score).sum::<f32>() / self.applied.len() as f32;
                sigmoid(mean_score)
            }
        };

        DecisionTrace {
            rules_applied: self.applied.clone(),
            alternatives_rejected: self.rejected.clone(),
            confidence,
        }
    }
}

impl Default for DecisionTraceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Resolve a token index to a vocab string, or `<idx:N>` if out of range.
fn resolve_token(idx: usize, vocab: &[String]) -> String {
    match vocab.get(idx) {
        Some(s) => s.clone(),
        None => format!("<idx:{}>", idx),
    }
}

/// Determine why an alternative was rejected relative to the best applied score.
fn rejection_reason(best_score: f32, alt_score: f32) -> &'static str {
    match alt_score {
        s if s < best_score * 0.5 => "score too low",
        s if s < best_score => "score too low",
        _ => "score too low", // Should not happen if best > alt, but safe default
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rule(depth: usize, tok: usize, score: f32, support: u32) -> ExtractedRule {
        ExtractedRule::new(vec![(0, 1), (1, 2)], (depth, tok), score, support)
    }

    fn sample_vocab() -> Vec<String> {
        vec![
            "fn".to_string(),
            "Result".to_string(),
            "match".to_string(),
            "if".to_string(),
        ]
    }

    #[test]
    fn empty_trace_produces_valid_output() {
        let trace = DecisionTraceBuilder::new().build();
        let output = trace.to_string(&sample_vocab());

        assert!(
            output.contains("(no rules applied"),
            "Empty trace should mention no rules: got\n{}",
            output,
        );
        assert!(
            output.contains("Confidence: 0.00"),
            "Empty trace should have zero confidence: got\n{}",
            output,
        );
    }

    #[test]
    fn trace_with_applied_rules_shows_them_correctly() {
        let rule = sample_rule(2, 2, 0.85, 5);
        let trace = DecisionTraceBuilder::new().applied(rule.clone()).build();
        let output = trace.to_string(&sample_vocab());

        assert!(
            output.contains("Rule 1:"),
            "Should show Rule 1: got\n{}",
            output,
        );
        assert!(
            output.contains("score=0.85"),
            "Should show score: got\n{}",
            output,
        );
        assert!(
            output.contains("support=5"),
            "Should show support: got\n{}",
            output,
        );
        assert!(
            output.contains("token='match'"),
            "Should resolve token name: got\n{}",
            output,
        );
    }

    #[test]
    fn confidence_is_sigmoid_bounded() {
        // Single rule with high score
        let trace = DecisionTraceBuilder::new()
            .applied(sample_rule(2, 2, 5.0, 1))
            .build();
        assert!(
            (0.0..=1.0).contains(&trace.confidence),
            "Confidence should be in [0,1]: got {}",
            trace.confidence,
        );
        assert!(
            trace.confidence > 0.99,
            "High input score should give confidence near 1.0: got {}",
            trace.confidence,
        );

        // Negative score → low confidence
        let trace = DecisionTraceBuilder::new()
            .applied(sample_rule(2, 2, -5.0, 1))
            .build();
        assert!(
            trace.confidence < 0.01,
            "Negative score should give confidence near 0.0: got {}",
            trace.confidence,
        );

        // Empty → exactly 0
        let trace = DecisionTraceBuilder::new().build();
        assert_eq!(
            trace.confidence, 0.0,
            "Empty trace should have confidence 0.0"
        );
    }

    #[test]
    fn to_string_maps_token_indices_to_vocab() {
        let rule = ExtractedRule::new(vec![(0, 0), (1, 1)], (2, 2), 0.9, 3);
        let trace = DecisionTraceBuilder::new().applied(rule).build();
        let output = trace.to_string(&sample_vocab());

        assert!(
            output.contains("token='fn'"),
            "Should map token index 0 → 'fn': got\n{}",
            output,
        );
        assert!(
            output.contains("token='Result'"),
            "Should map token index 1 → 'Result': got\n{}",
            output,
        );
        assert!(
            output.contains("token='match'"),
            "Should map token index 2 → 'match': got\n{}",
            output,
        );
    }

    #[test]
    fn to_string_handles_out_of_range_token() {
        let rule = ExtractedRule::new(
            vec![(0, 99)], // index 99 is out of range
            (1, 100),
            0.5,
            1,
        );
        let trace = DecisionTraceBuilder::new().applied(rule).build();
        let output = trace.to_string(&sample_vocab());

        assert!(
            output.contains("<idx:99>"),
            "Should fall back to <idx:N> for out-of-range: got\n{}",
            output,
        );
        assert!(
            output.contains("<idx:100>"),
            "Should fall back for action token: got\n{}",
            output,
        );
    }

    #[test]
    fn builder_constructs_trace_correctly() {
        let r1 = sample_rule(2, 2, 0.85, 5);
        let r2 = sample_rule(2, 3, 0.62, 2);
        let alt = sample_rule(2, 3, 0.30, 1);

        let trace = DecisionTraceBuilder::new()
            .applied(r1.clone())
            .applied(r2.clone())
            .rejected(alt.clone())
            .build();

        assert_eq!(trace.rules_applied.len(), 2, "Should have 2 applied rules");
        assert_eq!(
            trace.alternatives_rejected.len(),
            1,
            "Should have 1 rejected rule"
        );
        assert!(
            (0.0..=1.0).contains(&trace.confidence),
            "Confidence should be in [0,1]: got {}",
            trace.confidence,
        );

        // Verify rules are in order
        assert_eq!(trace.rules_applied[0].action, (2, 2));
        assert_eq!(trace.rules_applied[1].action, (2, 3));
        assert_eq!(trace.alternatives_rejected[0].action, (2, 3));
    }

    #[test]
    fn builder_default_trait() {
        let builder = DecisionTraceBuilder::default();
        let trace = builder.build();
        assert!(trace.rules_applied.is_empty());
        assert!(trace.alternatives_rejected.is_empty());
        assert_eq!(trace.confidence, 0.0);
    }

    #[test]
    fn alternatives_only_trace() {
        let alt = sample_rule(2, 3, 0.40, 1);
        let trace = DecisionTraceBuilder::new().rejected(alt).build();
        let output = trace.to_string(&sample_vocab());

        assert!(
            output.contains("no rules applied, alternatives were considered"),
            "Should mention alternatives only: got\n{}",
            output,
        );
        assert!(
            output.contains("Alternative 1:"),
            "Should show Alternative 1: got\n{}",
            output,
        );
    }

    #[test]
    fn sigmoid_bounded_unit_interval() {
        for x in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let s = sigmoid(x);
            assert!(
                (0.0..=1.0).contains(&s),
                "sigmoid({}) = {} not in [0,1]",
                x,
                s,
            );
        }
        // Symmetry check: sigmoid(x) + sigmoid(-x) ≈ 1.0
        for x in [0.5, 1.0, 2.0, 5.0] {
            let sum = sigmoid(x) + sigmoid(-x);
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "sigmoid({}) + sigmoid(-{}) = {} ≠ 1.0",
                x,
                x,
                sum,
            );
        }
    }
}
