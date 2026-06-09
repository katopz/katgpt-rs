//! HoarePruner — predicate propagation across DDTree steps (Plan 223, Phase 2).
//!
//! Tracks semantic state with BLAKE3 hashing and propagates predicates
//! through multi-step DDTree paths to catch structural errors early.

/// BLAKE3 hash of semantic state.
pub type StateHash = [u8; 32];

/// Semantic state tracked across DDTree steps.
#[derive(Clone, Debug)]
pub struct SemanticState {
    /// Stack depth for bracket tracking.
    pub bracket_depth: u32,
    /// Open brackets (e.g., '(', '[', '{') as bitmask.
    pub open_brackets: u32,
    /// Keywords seen so far (bitmask for up to 32 keywords).
    pub keyword_mask: u32,
    /// BLAKE3 hash of state for quick comparison.
    pub hash: StateHash,
}

impl SemanticState {
    pub fn initial() -> Self {
        let mut state = Self {
            bracket_depth: 0,
            open_brackets: 0,
            keyword_mask: 0,
            hash: [0u8; 32],
        };
        state.rehash();
        state
    }

    /// Recompute BLAKE3 hash over current state fields.
    pub fn rehash(&mut self) {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.bracket_depth.to_le_bytes());
        hasher.update(&self.open_brackets.to_le_bytes());
        hasher.update(&self.keyword_mask.to_le_bytes());
        self.hash = *hasher.finalize().as_bytes();
    }

    /// Push a bracket onto the state.
    pub fn push_bracket(&mut self, bracket: u8) {
        self.bracket_depth += 1;
        self.open_brackets |= 1 << (bracket % 32);
        self.rehash();
    }

    /// Pop a bracket from the state.
    pub fn pop_bracket(&mut self, bracket: u8) {
        self.bracket_depth = self.bracket_depth.saturating_sub(1);
        self.open_brackets &= !(1 << (bracket % 32));
        self.rehash();
    }

    /// Mark a keyword as seen.
    pub fn mark_keyword(&mut self, keyword_idx: u8) {
        if keyword_idx < 32 {
            self.keyword_mask |= 1 << keyword_idx;
        }
        self.rehash();
    }
}

/// Predicate for Hoare-style verification across steps.
#[derive(Clone, Debug)]
pub enum Predicate {
    /// Base predicate: bracket depth must be <= max.
    BracketDepthLe(u32),
    /// Base predicate: specific keyword must have been seen.
    KeywordSeen(u8),
    /// Conjunction: both must hold.
    And(Box<Predicate>, Box<Predicate>),
    /// Disjunction: at least one must hold.
    Or(Box<Predicate>, Box<Predicate>),
}

impl Predicate {
    /// Evaluate predicate against current semantic state.
    pub fn evaluate(&self, state: &SemanticState) -> bool {
        match self {
            Predicate::BracketDepthLe(max) => state.bracket_depth <= *max,
            Predicate::KeywordSeen(idx) => *idx < 32 && (state.keyword_mask & (1 << *idx)) != 0,
            Predicate::And(l, r) => l.evaluate(state) && r.evaluate(state),
            Predicate::Or(l, r) => l.evaluate(state) || r.evaluate(state),
        }
    }
}

/// HoarePruner — propagates predicates across DDTree paths.
#[derive(Clone, Debug)]
pub struct HoarePruner {
    /// Current semantic state.
    state: SemanticState,
    /// Active predicates to check.
    predicates: Vec<Predicate>,
    /// Number of violations detected.
    violations: usize,
}

impl HoarePruner {
    pub fn new(predicates: Vec<Predicate>) -> Self {
        Self {
            state: SemanticState::initial(),
            predicates,
            violations: 0,
        }
    }

    /// Get current semantic state.
    pub fn state(&self) -> &SemanticState {
        &self.state
    }

    /// Propagate state with a new token. Returns true if all predicates hold.
    pub fn propagate(&mut self, token: &str) -> bool {
        // Update state based on token
        match token {
            "(" | "[" | "{" => self.state.push_bracket(token.as_bytes()[0]),
            ")" | "]" | "}" => self.state.pop_bracket(token.as_bytes()[0]),
            _ => {}
        }

        // Check all predicates
        let all_hold = self.predicates.iter().all(|p| p.evaluate(&self.state));
        if !all_hold {
            self.violations += 1;
        }
        all_hold
    }

    /// Propagate state with a single char — avoids allocation vs `propagate(&str)`.
    pub fn propagate_char(&mut self, ch: char) -> bool {
        match ch {
            '(' | '[' | '{' => self.state.push_bracket(ch as u8),
            ')' | ']' | '}' => self.state.pop_bracket(ch as u8),
            _ => {}
        }

        let all_hold = self.predicates.iter().all(|p| p.evaluate(&self.state));
        if !all_hold {
            self.violations += 1;
        }
        all_hold
    }

    /// Number of violations detected so far.
    pub fn violations(&self) -> usize {
        self.violations
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.state = SemanticState::initial();
        self.violations = 0;
    }

    /// Map token_idx to bracket character for ConstraintPruner impl.
    /// 0='(', 1=')', 2='[', 3=']', 4='{', 5='}'
    fn token_idx_to_char(idx: usize) -> Option<char> {
        match idx {
            0 => Some('('),
            1 => Some(')'),
            2 => Some('['),
            3 => Some(']'),
            4 => Some('{'),
            5 => Some('}'),
            _ => None,
        }
    }
}

#[cfg(feature = "hoare_pruner")]
impl crate::speculative::types::ConstraintPruner for HoarePruner {
    fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
        self.predicates.iter().all(|p| p.evaluate(&self.state))
    }

    fn propagate(&mut self, _depth: usize, token_idx: usize, _parent_token: &[usize]) -> bool {
        if let Some(ch) = Self::token_idx_to_char(token_idx) {
            self.propagate_char(ch)
        } else {
            // Non-bracket token: just check predicates against current state
            self.predicates.iter().all(|p| p.evaluate(&self.state))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bracket_depth_predicate() {
        let pred = Predicate::BracketDepthLe(2);
        let state = SemanticState::initial();
        assert!(pred.evaluate(&state));

        let mut state = SemanticState::initial();
        state.push_bracket(b'(');
        state.push_bracket(b'(');
        state.push_bracket(b'(');
        assert!(!pred.evaluate(&state));
    }

    #[test]
    fn test_predicate_propagation() {
        let pred = Predicate::BracketDepthLe(2);
        let mut pruner = HoarePruner::new(vec![pred]);

        assert!(pruner.propagate("("));
        assert!(pruner.propagate("("));
        assert!(!pruner.propagate("(")); // depth 3 > 2
        assert_eq!(pruner.violations(), 1);
    }

    #[test]
    fn test_and_or_predicates() {
        let pred = Predicate::And(
            Box::new(Predicate::BracketDepthLe(5)),
            Box::new(Predicate::Or(
                Box::new(Predicate::KeywordSeen(0)),
                Box::new(Predicate::BracketDepthLe(0)),
            )),
        );
        let state = SemanticState::initial();
        assert!(pred.evaluate(&state)); // depth 0 <= 5 AND (kw0 not seen OR depth 0 <= 0)
    }

    #[test]
    fn test_state_hash_changes() {
        let s1 = SemanticState::initial();
        let mut s2 = SemanticState::initial();
        s2.push_bracket(b'(');
        assert_ne!(s1.hash, s2.hash);
    }

    #[test]
    fn test_multistep_propagation_violation() {
        let pred = Predicate::BracketDepthLe(2);
        let mut pruner = HoarePruner::new(vec![pred]);

        // Path: "(", "(", "x", "(", "x", ")"
        assert!(pruner.propagate("(")); // depth 1
        assert!(pruner.propagate("(")); // depth 2
        assert!(pruner.propagate("x")); // depth 2, non-bracket
        assert!(!pruner.propagate("(")); // depth 3 > 2, VIOLATION
        assert!(!pruner.propagate("x")); // still depth 3, still invalid
        assert!(pruner.propagate(")")); // depth 2, valid again

        assert_eq!(pruner.violations(), 2);
        assert_eq!(pruner.state().bracket_depth, 2);
    }

    #[test]
    fn test_propagate_overhead_benchmark() {
        use std::time::Instant;

        let pred = Predicate::BracketDepthLe(10);
        let mut pruner = HoarePruner::new(vec![pred]);

        let tokens = ["(", ")", "[", "]", "{", "}", "x", "y"];
        let n = 10_000;

        // Warmup
        for _ in 0..100 {
            for &t in &tokens {
                let _ = pruner.propagate(t);
            }
            pruner.reset();
        }

        let start = Instant::now();
        for _ in 0..n {
            for &t in &tokens {
                let _ = pruner.propagate(t);
            }
            pruner.reset();
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() as f64 / (n as f64 * tokens.len() as f64);

        // Debug builds have no optimizations — use generous threshold.
        // Release builds should be well under 200ns.
        let budget = if cfg!(debug_assertions) {
            5000.0
        } else {
            200.0
        };
        assert!(
            per_call_ns < budget,
            "propagate overhead {per_call_ns:.1}ns exceeds {budget:.0}ns budget"
        );

        eprintln!(
            "  propagate: {per_call_ns:.1}ns/call ({n} iterations, {} tokens each)",
            tokens.len()
        );
    }
}
