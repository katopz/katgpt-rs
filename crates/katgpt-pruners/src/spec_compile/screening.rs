//! ScreeningPruner implementation for CompiledSpec.
//!
//! Graded relevance scoring via sigmoid over matching rule count.
//! Allowlist rules yield binary {0.0, 1.0}; ambiguous/no-rule cases
//! use a continuous sigmoid signal centered at match count 0.5.

use katgpt_core::traits::ScreeningPruner;

use super::types::*;

impl ScreeningPruner for CompiledSpec {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if self.global_blocked.contains(token_idx) {
            return 0.0;
        }

        if !self.global_allowed.is_empty() {
            return if self.global_allowed.contains(token_idx) {
                1.0
            } else {
                0.0
            };
        }

        let mut match_count: usize = 0;
        let mut first_applicable: Option<&SpecRule> = None;

        for rule in &self.rules {
            if let Some(rule_depth) = rule.depth
                && rule_depth != depth {
                    continue;
                }

            if !prefix_matches(&rule.prefix, parent_tokens) {
                continue;
            }

            match_count += 1;

            if first_applicable.is_none() {
                first_applicable = Some(rule);
            }
        }

        if match_count == 0 {
            return 1.0;
        }

        if let Some(rule) = first_applicable {
            let in_bitmap = rule.allowed.contains(token_idx);
            return if rule.is_allowlist {
                // Allowlist: binary relevance
                if in_bitmap { 1.0 } else { 0.0 }
            } else {
                // Blocklist: in-bitmap is hard 0.0, out-of-bitmap is sigmoid
                if in_bitmap {
                    0.0
                } else {
                    sigmoid(match_count as f32 - 0.5)
                }
            };
        }

        // Fallback (should not reach here when match_count > 0)
        1.0
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn prefix_matches(prefix: &[PrefixEntry], parent_tokens: &[usize]) -> bool {
    if prefix.is_empty() {
        return true;
    }

    for entry in prefix {
        match parent_tokens.get(entry.depth) {
            Some(&token) if token == entry.token_idx => continue,
            _ => return false,
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_classification_spec() -> CompiledSpec {
        let allowed = CompactBitmap::from_token_indices(
            [
                b'p', b'o', b's', b'i', b't', b'v', b'e', b'n', b'g', b'a', b'r', b' ', b'\n',
            ]
            .iter()
            .map(|&b| b as usize),
        );

        CompiledSpec {
            spec_hash: [0u8; 32],
            rules: vec![SpecRule {
                depth: None,
                prefix: Vec::new(),
                allowed,
                is_allowlist: true,
            }],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        }
    }

    fn make_extraction_spec() -> CompiledSpec {
        CompiledSpec {
            spec_hash: [1u8; 32],
            rules: vec![
                SpecRule {
                    depth: Some(0),
                    prefix: Vec::new(),
                    allowed: CompactBitmap::from_token_indices(
                        [b'{', b'['].iter().map(|&b| b as usize),
                    ),
                    is_allowlist: false,
                },
                SpecRule {
                    depth: None,
                    prefix: Vec::new(),
                    allowed: CompactBitmap::from_token_indices(
                        (b'a'..=b'z')
                            .chain(b'0'..=b'9')
                            .chain([b'{', b'}', b'[', b']', b':', b',', b'"', b' ', b'\n'])
                            .map(|b| b as usize),
                    ),
                    is_allowlist: false,
                },
            ],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        }
    }

    #[test]
    fn test_classification_relevance() {
        let spec = make_classification_spec();

        let rel_allowed = spec.relevance(0, b'p' as usize, &[]);
        assert!(
            (rel_allowed - 1.0).abs() < f32::EPSILON,
            "allowed token should have relevance 1.0, got {rel_allowed}"
        );

        let rel_blocked = spec.relevance(0, b'x' as usize, &[]);
        assert!(
            (rel_blocked - 0.0).abs() < f32::EPSILON,
            "blocked token should have relevance 0.0, got {rel_blocked}"
        );
    }

    #[test]
    fn test_extraction_relevance_blocklist() {
        let spec = make_extraction_spec();

        let rel = spec.relevance(0, b'a' as usize, &[]);
        assert!(
            rel > 0.0,
            "blocklist token not in bitmap should have positive relevance, got {rel}"
        );

        let rel_blocked = spec.relevance(0, b'{' as usize, &[]);
        assert!(
            (rel_blocked - 0.0).abs() < f32::EPSILON,
            "blocklist token in bitmap should have relevance 0.0, got {rel_blocked}"
        );
    }

    #[test]
    fn test_empty_spec_returns_one() {
        let spec = CompiledSpec {
            spec_hash: [0u8; 32],
            rules: Vec::new(),
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        };

        assert!(
            (spec.relevance(0, 0, &[]) - 1.0).abs() < f32::EPSILON,
            "empty spec should return 1.0"
        );
        assert!(
            (spec.relevance(5, 255, &[1, 2, 3, 4, 5]) - 1.0).abs() < f32::EPSILON,
            "empty spec should return 1.0 at any depth"
        );
    }

    #[test]
    fn test_global_blocked_returns_zero() {
        let spec = CompiledSpec {
            spec_hash: [0u8; 32],
            rules: Vec::new(),
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::from_token_indices(
                [b'<', b'>'].iter().map(|&b| b as usize),
            ),
        };

        assert!(
            (spec.relevance(0, b'<' as usize, &[]) - 0.0).abs() < f32::EPSILON,
            "globally blocked should be 0.0"
        );
        assert!(
            (spec.relevance(0, b'>' as usize, &[]) - 0.0).abs() < f32::EPSILON,
            "globally blocked should be 0.0"
        );
        assert!(
            (spec.relevance(0, b'a' as usize, &[]) - 1.0).abs() < f32::EPSILON,
            "non-blocked with empty global_allowed should be 1.0"
        );
    }

    #[test]
    fn test_global_allowed_membership() {
        let spec = CompiledSpec {
            spec_hash: [0u8; 32],
            rules: Vec::new(),
            vocab_size: 256,
            global_allowed: CompactBitmap::from_token_indices(
                [b'a', b'b', b'c'].iter().map(|&b| b as usize),
            ),
            global_blocked: CompactBitmap::empty(),
        };

        assert!(
            (spec.relevance(0, b'a' as usize, &[]) - 1.0).abs() < f32::EPSILON,
            "in global_allowed should be 1.0"
        );
        assert!(
            (spec.relevance(0, b'z' as usize, &[]) - 0.0).abs() < f32::EPSILON,
            "not in global_allowed should be 0.0"
        );
    }

    #[test]
    fn test_sigmoid_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(5.0) > 0.99);
        assert!(sigmoid(-5.0) < 0.01);
        assert!((sigmoid(1.0) - (1.0 / (1.0 + (-1.0f32).exp()))).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ambiguous_rule_uses_sigmoid() {
        let spec = CompiledSpec {
            spec_hash: [2u8; 32],
            rules: vec![SpecRule {
                depth: None,
                prefix: Vec::new(),
                allowed: CompactBitmap::from_token_indices(
                    [b'<', b'>'].iter().map(|&b| b as usize),
                ),
                is_allowlist: false,
            }],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        };

        let expected = sigmoid(1.0f32 - 0.5);
        let rel = spec.relevance(0, b'a' as usize, &[]);
        assert!(
            (rel - expected).abs() < 1e-5,
            "ambiguous blocklist token should use sigmoid, expected {expected}, got {rel}"
        );
    }
}
