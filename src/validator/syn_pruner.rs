use super::partial_parser::PartialParser;
use super::types::PruneResult;
use crate::speculative::types::ConstraintPruner;
use crate::tokenizer::BpeTokenizer;
use crate::tokenizer::BpeTokenizerImpl;
use std::sync::Arc;
use std::sync::Mutex;

/// Two-tier syntax pruner for Validator.
///
/// Tier 0: Bracket balancer DFA (PartialParser) — O(n), rejects clearly broken code.
/// Tier 1: `syn` parse attempt — accurate, but expensive. Only called if Tier 0 passes.
pub struct SynPruner {
    tokenizer: Arc<BpeTokenizer>,
    parser: Mutex<PartialParser>,
    scratch_tokens: Mutex<Vec<usize>>,
}

impl SynPruner {
    pub fn new(tokenizer: Arc<BpeTokenizer>) -> Self {
        Self {
            tokenizer,
            parser: Mutex::new(PartialParser::new()),
            scratch_tokens: Mutex::new(Vec::with_capacity(64)),
        }
    }

    /// Validate a complete code string through both tiers.
    pub fn validate(&self, code: &str) -> PruneResult {
        // Tier 0: Bracket balance
        if !self
            .parser
            .lock()
            .expect("parser mutex poisoned")
            .is_valid(code)
        {
            return PruneResult {
                is_valid: false,
                error_kind: super::types::ErrorKind::UnbalancedBrackets,
            };
        }

        // Tier 1: syn parse
        match syn::parse_str::<syn::Stmt>(code) {
            Ok(_) => PruneResult {
                is_valid: true,
                error_kind: super::types::ErrorKind::None,
            },
            Err(e) => PruneResult {
                is_valid: false,
                error_kind: super::types::ErrorKind::SynError(e.to_string()),
            },
        }
    }

    /// Quick Tier 0 check only (for DDTree hot path).
    pub fn is_valid_quick(&self, code: &str) -> bool {
        self.parser
            .lock()
            .expect("parser mutex poisoned")
            .is_valid(code)
    }
}

impl ConstraintPruner for SynPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let mut all_tokens = self
            .scratch_tokens
            .lock()
            .expect("scratch_tokens mutex poisoned");
        all_tokens.clear();
        all_tokens.extend_from_slice(parent_tokens);
        all_tokens.push(token_idx);

        let code = BpeTokenizerImpl::decode(&self.tokenizer, &all_tokens);

        // Only do Tier 0 (bracket balance) in the hot path.
        // Tier 1 (syn) is too expensive for every DDTree node.
        // Reuse the existing parser via Mutex to avoid per-call allocation.
        let mut parser = self.parser.lock().expect("parser mutex poisoned");
        parser.is_valid(&code)
    }

    #[cfg(feature = "hoare_pruner")]
    fn propagate(&mut self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let mut all_tokens = self
            .scratch_tokens
            .lock()
            .expect("scratch_tokens mutex poisoned");
        all_tokens.clear();
        all_tokens.extend_from_slice(parent_tokens);
        all_tokens.push(token_idx);

        let code = BpeTokenizerImpl::decode(&self.tokenizer, &all_tokens);

        let mut parser = self.parser.lock().expect("parser mutex poisoned");
        parser.reset();
        let valid = parser.is_valid(&code);

        const MAX_BRACKET_DEPTH: i32 = 32;
        valid && parser.total_depth() <= MAX_BRACKET_DEPTH
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syn_pruner_accepts_valid_rust() {
        let tokenizer = Arc::new(crate::tokenizer::BpeTrainer::train("fn let mut x", 64));
        let pruner = SynPruner::new(tokenizer);

        let result = pruner.validate("let x = 42;");
        assert!(result.is_valid, "expected valid for 'let x = 42;'");
        assert_eq!(result.error_kind, super::super::types::ErrorKind::None);

        let result = pruner.validate("fn main() { }");
        assert!(result.is_valid, "expected valid for 'fn main() {{ }}'");

        let result = pruner.validate("let s = \"hello\";");
        assert!(result.is_valid, "expected valid for string literal");
    }

    #[test]
    fn test_syn_pruner_rejects_invalid_rust() {
        let tokenizer = Arc::new(crate::tokenizer::BpeTrainer::train("fn let mut x", 64));
        let pruner = SynPruner::new(tokenizer);

        let result = pruner.validate("let = ;");
        assert!(!result.is_valid, "expected invalid for 'let = ;'");

        match result.error_kind {
            super::super::types::ErrorKind::SynError(msg) => {
                assert!(!msg.is_empty(), "syn error should have a message");
            }
            other => panic!("expected SynError, got {other:?}"),
        }
    }

    #[test]
    fn test_syn_pruner_bracket_tier_rejects() {
        let tokenizer = Arc::new(crate::tokenizer::BpeTrainer::train("fn let { }", 64));
        let pruner = SynPruner::new(tokenizer);

        // Unmatched closing brace — Tier 0 should reject before syn sees it
        let result = pruner.validate("fn main() { } }");
        assert!(!result.is_valid, "expected invalid for unbalanced braces");
        assert_eq!(
            result.error_kind,
            super::super::types::ErrorKind::UnbalancedBrackets
        );

        // Unmatched closing paren
        let result = pruner.validate("foo())");
        assert!(!result.is_valid, "expected invalid for unbalanced parens");
        assert_eq!(
            result.error_kind,
            super::super::types::ErrorKind::UnbalancedBrackets
        );
    }
}
