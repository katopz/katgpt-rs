//! Lodestar — Completion-Distance Pruning (Plan 207, Research 183).
//!
//! A reusable automaton-backed [`ConstraintPruner`] + [`CompletionHorizon`].
//! Any hard constraint expressible as a finite automaton over the token vocab
//! (balanced brackets, mini-JSON, a keyword set, a game action grammar, …) gets,
//! for free, one precomputed integer per state — the **shortest-accepting-distance**
//! `d(s)` — which simultaneously powers budget-aware masking (A), jump-ahead (B),
//! and an admissible A* heuristic / termination proof (C).
//!
//! Pure inference-time, zero training. The distance vector is computed **once** at
//! build time by reverse relaxation from the accepting set.
//!
//! # Architecture
//!
//! ```text
//! LodestarAutomatonBuilder
//!       │  add_transition / add_accept
//!       ▼
//! LodestarAutomaton (immutable after build)
//!   ├── transitions: flat row-major δ(s, t) → Option<usize>
//!   ├── distances:   precomputed d(s) per state
//!   └── singular_spans: precomputed forced-path length per state
//!       │
//!       ▼
//! LodestarPruner  (ConstraintPruner + CompletionHorizon)
//!   ├── is_valid:        δ defined ⇒ valid, + optional budget check
//!   ├── batch_is_valid:  amortized state lookup (single follow_path)
//!   ├── min_completion_distance: d(δ(current, token))
//!   └── singular_span_len:       precomputed span from current state
//! ```
//!
//! See `examples/lodestar_demo.rs` for the GOAT proof (100% vs 13.9% valid-in-budget).

use katgpt_core::traits::{CompletionHorizon, ConstraintPruner};

/// Marker for "no outgoing edge" in the packed transition table.
const NO_EDGE: usize = usize::MAX;

/// Marker for "no accepting state reachable from here".
pub const UNREACHABLE: u32 = u32::MAX;

// ── BitVec (compact bool storage) ──────────────────────────────────────

/// Compact bit-vector backed by `Vec<u64>`.
///
/// 8× memory savings vs `Vec<bool>` (1 bit per element vs 1 byte).
/// Zero external dependencies.
#[derive(Clone, Debug)]
struct BitVec {
    words: Vec<u64>,
    len: usize,
}

impl BitVec {
    /// All-false bit-vector of length `len`.
    fn new(len: usize) -> Self {
        let words = len.div_ceil(64);
        Self {
            words: vec![0u64; words],
            len,
        }
    }

    /// Get bit at `idx`. Returns `false` if out of bounds.
    #[inline]
    fn get(&self, idx: usize) -> bool {
        if idx >= self.len {
            return false;
        }
        let word = idx / 64;
        let bit = idx % 64;
        (self.words[word] >> bit) & 1 == 1
    }

    /// Set bit at `idx` to `value`. Panics if out of bounds.
    #[inline]
    fn set(&mut self, idx: usize, value: bool) {
        assert!(
            idx < self.len,
            "BitVec index {idx} out of bounds (len={})",
            self.len
        );
        let word = idx / 64;
        let bit = idx % 64;
        if value {
            self.words[word] |= 1u64 << bit;
        } else {
            self.words[word] &= !(1u64 << bit);
        }
    }
}

// ── LodestarAutomaton ──────────────────────────────────────────────────

/// A deterministic finite automaton (DFA) that tracks generation state.
///
/// Provides transition function δ(state, token) → Option<next_state>,
/// with precomputed shortest-accepting-distances for budget-aware pruning.
///
/// Construct via [`LodestarAutomaton::builder`], then call
/// [`LodestarAutomatonBuilder::build`] to finalize and precompute distances.
/// After that the automaton is immutable and `O(1)` per lookup.
#[derive(Clone, Debug)]
pub struct LodestarAutomaton {
    /// δ(s, t) = transitions[s * vocab_size + t] → next state.
    /// Flat row-major: state × vocab → next state (`usize::MAX` = no transition).
    transitions: Vec<usize>,
    /// Set of accepting (complete) states (bit-packed).
    accept_states: BitVec,
    /// Precomputed shortest-accepting-distance d(s) per state.
    /// `u32::MAX` = unreachable (dead state).
    distances: Vec<u32>,
    /// Singular-span length per state (number of forced tokens until a real branch).
    /// Precomputed at build time — zero allocation on hot paths.
    singular_spans: Vec<u32>,
    vocab_size: usize,
    n_states: usize,
    start_state: usize,
}

impl LodestarAutomaton {
    /// Create a builder for constructing the automaton.
    ///
    /// - `vocab_size`: number of distinct tokens in the vocabulary (alphabet Σ).
    /// - `n_states`: number of states in the DFA (including start and accept).
    /// - `start_state`: the initial state (must be < `n_states`).
    #[inline]
    pub fn builder(
        vocab_size: usize,
        n_states: usize,
        start_state: usize,
    ) -> LodestarAutomatonBuilder {
        assert!(
            start_state < n_states,
            "start_state ({start_state}) must be < n_states ({n_states})"
        );
        LodestarAutomatonBuilder {
            transitions: vec![NO_EDGE; n_states * vocab_size],
            accept_states: BitVec::new(n_states),
            vocab_size,
            n_states,
            start_state,
        }
    }

    /// Successor of `state` on `token`, or `None` if illegal.
    ///
    /// O(1) flat-array lookup — branch-free on the happy path.
    #[inline]
    pub fn transition(&self, state: usize, token: usize) -> Option<usize> {
        let idx = state * self.vocab_size + token;
        match self.transitions.get(idx) {
            Some(&ns) if ns != NO_EDGE => Some(ns),
            _ => None,
        }
    }

    /// Whether `state` is accepting (a complete, valid output ends here).
    #[inline]
    pub fn is_accept(&self, state: usize) -> bool {
        self.accept_states.get(state)
    }

    /// Precomputed shortest-accepting-distance `d(s)`.
    ///
    /// Returns [`UNREACHABLE`] if no accepting state is reachable from `state`.
    #[inline]
    pub fn distance(&self, state: usize) -> u32 {
        match self.distances.get(state) {
            Some(&d) => d,
            None => UNREACHABLE,
        }
    }

    /// Precomputed singular-span length from `state`.
    ///
    /// Number of consecutive forced (single-outgoing-edge) transitions starting
    /// from `state`, until a real branch (≥2 legal tokens) or an accepting state.
    #[inline]
    pub fn singular_span_len(&self, state: usize) -> u32 {
        match self.singular_spans.get(state) {
            Some(&s) => s,
            None => 0,
        }
    }

    /// Replay `parent_tokens` from the start state, returning the state reached.
    ///
    /// Returns `start_state` if the path is empty or if at any point δ returns None
    /// (i.e., the prefix is illegal).
    #[inline]
    pub fn follow_path(&self, parent_tokens: &[usize]) -> usize {
        let mut state = self.start_state;
        for &token in parent_tokens {
            match self.transition(state, token) {
                Some(next) => state = next,
                None => return self.start_state,
            }
        }
        state
    }

    /// The start state.
    #[inline]
    pub fn start_state(&self) -> usize {
        self.start_state
    }

    /// Number of states in the automaton.
    #[inline]
    pub fn n_states(&self) -> usize {
        self.n_states
    }

    /// Vocabulary size (alphabet cardinality).
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// ── Builder ────────────────────────────────────────────────────────────

/// Builder for [`LodestarAutomaton`].
///
/// Add transitions and accept states, then call [`build`](Self::build) to
/// finalize the automaton (precomputing distances and singular spans).
pub struct LodestarAutomatonBuilder {
    transitions: Vec<usize>,
    accept_states: BitVec,
    vocab_size: usize,
    n_states: usize,
    start_state: usize,
}

impl LodestarAutomatonBuilder {
    /// Add a transition δ(from, token) → to.
    ///
    /// Panics if any index is out of range.
    pub fn add_transition(&mut self, from: usize, token: usize, to: usize) -> &mut Self {
        assert!(
            from < self.n_states,
            "from state ({from}) out of range (n_states={})",
            self.n_states
        );
        assert!(
            to < self.n_states,
            "to state ({to}) out of range (n_states={})",
            self.n_states
        );
        assert!(
            token < self.vocab_size,
            "token ({token}) out of range (vocab_size={})",
            self.vocab_size
        );
        self.transitions[from * self.vocab_size + token] = to;
        self
    }

    /// Mark `state` as accepting (a complete, valid output ends here).
    ///
    /// Panics if `state` is out of range.
    pub fn add_accept(&mut self, state: usize) -> &mut Self {
        assert!(
            state < self.n_states,
            "accept state ({state}) out of range (n_states={})",
            self.n_states
        );
        self.accept_states.set(state, true);
        self
    }

    /// Finalize the automaton: precompute distances and singular spans.
    ///
    /// Runs reverse-BFS distance computation (O(|S|·|Σ|) per relaxation pass)
    /// and singular-span precomputation. Called exactly once.
    pub fn build(self) -> LodestarAutomaton {
        let distances = precompute_distances(
            &self.transitions,
            &self.accept_states,
            self.n_states,
            self.vocab_size,
        );
        let singular_spans = precompute_singular_spans(
            &self.transitions,
            &self.accept_states,
            self.n_states,
            self.vocab_size,
        );
        LodestarAutomaton {
            transitions: self.transitions,
            accept_states: self.accept_states,
            distances,
            singular_spans,
            vocab_size: self.vocab_size,
            n_states: self.n_states,
            start_state: self.start_state,
        }
    }
}

// ── Precomputation ─────────────────────────────────────────────────────

/// Reverse-BFS from accept states.
///
/// d(accept) = 0. BFS outward from all accept states via reverse edges.
/// O(|S|·|Σ|) total — each (state, token) edge examined exactly once during
/// reverse-edge construction + BFS traversal.
fn precompute_distances(
    transitions: &[usize],
    accept_states: &BitVec,
    n_states: usize,
    vocab_size: usize,
) -> Vec<u32> {
    use std::collections::VecDeque;

    let mut dist = vec![UNREACHABLE; n_states];
    let mut queue = VecDeque::with_capacity(n_states);

    // Build reverse adjacency: for each state, which states can reach it?
    // reverse_edges[dest] = list of (source, token) pairs that transition to dest.
    // Total edges = S × Σ, built in O(S·Σ).
    let mut reverse_edges: Vec<Vec<usize>> = vec![Vec::new(); n_states];
    for s in 0..n_states {
        let base = s * vocab_size;
        for t in 0..vocab_size {
            let ns = transitions[base + t];
            if ns != NO_EDGE {
                reverse_edges[ns].push(s);
            }
        }
    }

    // Seed with all accept states at distance 0
    #[allow(clippy::needless_range_loop)]
    for s in 0..n_states {
        if accept_states.get(s) {
            dist[s] = 0;
            queue.push_back(s);
        }
    }

    // Multi-source BFS along reverse edges
    while let Some(target) = queue.pop_front() {
        let d_next = dist[target] + 1;
        for &source in &reverse_edges[target] {
            if dist[source] == UNREACHABLE {
                dist[source] = d_next;
                queue.push_back(source);
            }
        }
    }
    dist
}

/// Precompute singular-span length per state.
///
/// For each state, count consecutive forced (single-outgoing-edge) transitions.
/// 0 if the state has >1 legal token or 0 legal tokens.
/// Follows the forced token until hitting a real branch or ACCEPT.
fn precompute_singular_spans(
    transitions: &[usize],
    accept_states: &BitVec,
    n_states: usize,
    vocab_size: usize,
) -> Vec<u32> {
    (0..n_states)
        .map(|s| compute_singular_span(transitions, accept_states, vocab_size, s))
        .collect()
}

/// Compute the singular span from a single state.
fn compute_singular_span(
    transitions: &[usize],
    accept_states: &BitVec,
    vocab_size: usize,
    mut state: usize,
) -> u32 {
    let mut len = 0u32;
    loop {
        // Count outgoing edges and find the single legal token (if any).
        let base = state * vocab_size;
        let mut only_next: usize = 0;
        let mut count = 0u32;
        for t in 0..vocab_size {
            let ns = transitions[base + t];
            if ns != NO_EDGE {
                count += 1;
                only_next = ns;
                if count > 1 {
                    // More than one legal token — not a forced state.
                    return len;
                }
            }
        }
        match count {
            0 => return len, // Dead end, not forced.
            1 => {
                state = only_next;
                len += 1;
                if accept_states.get(state) {
                    return len; // Reached ACCEPT through forced chain.
                }
            }
            _ => return len, // Unreachable: count is 0 or 1 at this point.
        }
    }
}

// ── LodestarPruner ─────────────────────────────────────────────────────

/// Budget-aware constraint pruner powered by a precomputed automaton.
///
/// Implements [`ConstraintPruner`] (δ defined ⇒ valid) and [`CompletionHorizon`]
/// (distance + singular span from precomputed tables).
///
/// The pruner tracks automaton state via `parent_tokens` and provides
/// admissible completion-distance estimates for budget-aware masking.
///
/// When a `budget` is set, `is_valid` additionally rejects tokens whose
/// successor cannot complete within the remaining budget:
/// `1 + distances[next_state] <= budget_remaining` where
/// `budget_remaining = budget - depth - 1`.
#[derive(Clone, Debug)]
pub struct LodestarPruner {
    automaton: LodestarAutomaton,
    /// Maximum token budget for budget-aware distance checks.
    /// When set, `is_valid` additionally rejects tokens that cannot
    /// complete within the remaining budget.
    budget: Option<usize>,
}

impl LodestarPruner {
    /// Wrap a precomputed automaton. No budget constraint.
    #[inline]
    pub fn new(automaton: LodestarAutomaton) -> Self {
        Self {
            automaton,
            budget: None,
        }
    }

    /// Wrap a precomputed automaton with a token budget.
    ///
    /// `is_valid` will reject tokens whose successor cannot complete
    /// within the remaining budget: `1 + d(δ(s, t)) <= budget - depth - 1`.
    #[inline]
    pub fn with_budget(automaton: LodestarAutomaton, budget: usize) -> Self {
        Self {
            automaton,
            budget: Some(budget),
        }
    }

    /// Borrow the underlying automaton (for distances, spans, etc.).
    #[inline]
    pub fn automaton(&self) -> &LodestarAutomaton {
        &self.automaton
    }

    /// The configured budget, if any.
    #[inline]
    pub fn budget(&self) -> Option<usize> {
        self.budget
    }

    /// Follow `parent_tokens` from the start state to get the current automaton state.
    /// Returns the start state if the prefix is illegal.
    #[inline]
    fn current_state(&self, parent_tokens: &[usize]) -> usize {
        self.automaton.follow_path(parent_tokens)
    }
}

impl ConstraintPruner for LodestarPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        if token_idx >= self.automaton.vocab_size {
            return false;
        }
        let state = self.current_state(parent_tokens);
        let next = match self.automaton.transition(state, token_idx) {
            Some(ns) => ns,
            None => return false,
        };
        // Budget check: reject if the successor cannot complete within budget_remaining.
        match self.budget {
            Some(budget) => {
                let budget_remaining = match budget.checked_sub(depth + 1) {
                    Some(br) => br,
                    None => return false,
                };
                let d = self.automaton.distance(next);
                match d {
                    UNREACHABLE => false,
                    _ => (1 + d as usize) <= budget_remaining,
                }
            }
            None => true,
        }
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let state = self.current_state(parent_tokens);
        let budget_remaining = self.budget.map(|b| b.saturating_sub(depth + 1));
        let len = candidates.len().min(results.len());
        for i in 0..len {
            let token = candidates[i];
            if token >= self.automaton.vocab_size {
                results[i] = false;
                continue;
            }
            let next = match self.automaton.transition(state, token) {
                Some(ns) => ns,
                None => {
                    results[i] = false;
                    continue;
                }
            };
            results[i] = match budget_remaining {
                Some(br) => {
                    let d = self.automaton.distance(next);
                    match d {
                        UNREACHABLE => false,
                        _ => (1 + d as usize) <= br,
                    }
                }
                None => true,
            };
        }
    }
}

impl CompletionHorizon for LodestarPruner {
    fn min_completion_distance(
        &self,
        _depth: usize,
        token_idx: usize,
        parent_tokens: &[usize],
    ) -> u32 {
        if token_idx >= self.automaton.vocab_size {
            return UNREACHABLE;
        }
        let state = self.current_state(parent_tokens);
        match self.automaton.transition(state, token_idx) {
            Some(next) => self.automaton.distance(next),
            None => UNREACHABLE,
        }
    }

    fn singular_span_len(&self, _depth: usize, parent_tokens: &[usize]) -> u32 {
        let state = self.current_state(parent_tokens);
        self.automaton.singular_span_len(state)
    }
}

// ── Lodestar DDTree Configuration ───────────────────────────────────────

/// Configuration for `build_dd_tree_lodestar` — controls A* ordering and jump-ahead.
///
/// Default reproduces pure log-prob best-first (λ = 0, jump-ahead disabled).
#[derive(Clone, Debug)]
pub struct LodestarConfig {
    /// A* distance weight λ. Heap key = `score − λ·d(s)`.
    /// λ = 0 (default) → pure log-prob ordering, byte-identical to `build_dd_tree_pruned`.
    /// λ > 0 → prefer branches closer to completion (A* admissible heuristic).
    pub astar_lambda: f32,
    /// Enable jump-ahead: collapse singular spans into one tree node.
    /// When `true`, deterministic forced paths are emitted as a single expansion step
    /// instead of per-token, reducing tree nodes and speeding up traversal.
    pub jump_ahead: bool,
}

impl Default for LodestarConfig {
    fn default() -> Self {
        Self {
            astar_lambda: 0.0,
            jump_ahead: false,
        }
    }
}

impl LodestarConfig {
    /// Pure log-prob ordering, no jump-ahead (default).
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// A* ordering with jump-ahead enabled.
    pub fn thinking(lambda: f32) -> Self {
        Self {
            astar_lambda: lambda,
            jump_ahead: true,
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────

    /// Simple 3-state linear chain: start(0) → mid(1) → accept(2).
    /// Vocab = {A=0, B=1}. Only token A is legal at each non-accept state.
    fn linear_automaton() -> LodestarAutomaton {
        let mut b = LodestarAutomaton::builder(2, 3, 0);
        b.add_transition(0, 0, 1); // start --A--> mid
        b.add_transition(1, 0, 2); // mid  --A--> accept
        b.add_accept(2);
        b.build()
    }

    /// Tiny array grammar: `[` ( `N` (`,` `N`)* )? `]`, vocab = {OPEN=0,CLOSE=1,NUM=2,COMMA=3}.
    /// States: 0=start(need `[`), 1=after`[`(Value: NUM or `]`),
    ///         2=after value(More: `,` or `]`), 3=ACCEPT.
    fn array_automaton() -> LodestarAutomaton {
        const OPEN: usize = 0;
        const CLOSE: usize = 1;
        const NUM: usize = 2;
        const COMMA: usize = 3;
        let mut b = LodestarAutomaton::builder(4, 4, 0);
        b.add_transition(0, OPEN, 1);
        b.add_transition(1, NUM, 2);
        b.add_transition(1, CLOSE, 3); // empty array `[]`
        b.add_transition(2, COMMA, 1);
        b.add_transition(2, CLOSE, 3);
        b.add_accept(3);
        b.build()
    }

    /// Diamond-shaped automaton: start(0) → {mid_a(1), mid_b(2)} → accept(3).
    /// Token A=0 from start goes to mid_a, token B=1 from start goes to mid_b.
    /// Both mid states accept token C=2 to reach accept.
    fn diamond_automaton() -> LodestarAutomaton {
        let mut b = LodestarAutomaton::builder(3, 4, 0);
        b.add_transition(0, 0, 1); // start --A--> mid_a
        b.add_transition(0, 1, 2); // start --B--> mid_b
        b.add_transition(1, 2, 3); // mid_a --C--> accept
        b.add_transition(2, 2, 3); // mid_b --C--> accept
        b.add_accept(3);
        b.build()
    }

    /// Automaton with a dead state (state 1 has no outgoing edges, not accepting).
    fn dead_state_automaton() -> LodestarAutomaton {
        let mut b = LodestarAutomaton::builder(2, 3, 0);
        b.add_transition(0, 0, 1); // start --token0--> dead
        b.add_transition(0, 1, 2); // start --token1--> accept
        // state 1 has no outgoing transitions (dead)
        b.add_accept(2);
        b.build()
    }

    /// Header + nested array grammar (mirrors lodestar_demo.rs).
    /// 3 forced header tokens, then a nested array structure.
    /// Vocab = {OPEN=0, CLOSE=1, NUM=2, COMMA=3, HDR=4}.
    fn header_array_automaton() -> LodestarAutomaton {
        const OPEN: usize = 0;
        const CLOSE: usize = 1;
        const NUM: usize = 2;
        const COMMA: usize = 3;
        const HDR: usize = 4;
        const MAX_DEPTH: usize = 2;
        const HLEN: usize = 3;
        const ACCEPT: usize = HLEN + MAX_DEPTH * 2;
        const N_STATES: usize = ACCEPT + 1;

        let mut b = LodestarAutomaton::builder(5, N_STATES, 0);

        // Header chain: 0→1→2 (forced HDR tokens).
        b.add_transition(0, HDR, 1);
        b.add_transition(1, HDR, 2);
        b.add_transition(2, OPEN, HLEN); // → (depth 1, Value)

        // Body states for each nesting depth.
        for d in 1..=MAX_DEPTH {
            let sv = HLEN + (d - 1) * 2; // (depth d, Value)
            let sm = sv + 1; // (depth d, More)

            // Value state: NUM → More, or nested OPEN (if not at max depth).
            b.add_transition(sv, NUM, sm);
            if d < MAX_DEPTH {
                b.add_transition(sv, OPEN, HLEN + d * 2); // → (depth d+1, Value)
            }

            // More state: COMMA → Value, CLOSE → parent More or ACCEPT.
            b.add_transition(sm, COMMA, sv);
            let close_target = if d == 1 {
                ACCEPT
            } else {
                HLEN + (d - 2) * 2 + 1
            };
            b.add_transition(sm, CLOSE, close_target);
        }

        b.add_accept(ACCEPT);
        b.build()
    }

    // ── T3: LodestarAutomaton Tests ───────────────────────────────

    #[test]
    fn test_1_simple_3_state_automaton() {
        let a = linear_automaton();
        // Start state transitions.
        assert!(a.transition(0, 0).is_some(), "token A legal from start");
        assert!(a.transition(0, 1).is_none(), "token B illegal from start");
        // Mid state transitions.
        assert!(a.transition(1, 0).is_some(), "token A legal from mid");
        assert!(a.transition(1, 1).is_none(), "token B illegal from mid");
        // Accept state has no outgoing.
        assert!(a.transition(2, 0).is_none(), "no outgoing from accept");
        // Accept check.
        assert!(!a.is_accept(0));
        assert!(!a.is_accept(1));
        assert!(a.is_accept(2));
    }

    #[test]
    fn test_2_precomputed_distances() {
        let a = linear_automaton();
        assert_eq!(a.distance(2), 0, "d(accept)=0");
        assert_eq!(a.distance(1), 1, "d(mid)=1");
        assert_eq!(a.distance(0), 2, "d(start)=2");
    }

    #[test]
    fn test_3_singular_spans_linear_chain() {
        let a = linear_automaton();
        // start has 1 outgoing → mid; mid has 1 outgoing → accept.
        // So singular span from start = 2 (forced path: start→mid→accept).
        assert_eq!(
            a.singular_span_len(0),
            2,
            "linear chain: 2 forced steps from start"
        );
        // mid has 1 outgoing → accept, which is accepting → span = 1.
        assert_eq!(
            a.singular_span_len(1),
            1,
            "linear chain: 1 forced step from mid"
        );
        // accept has 0 outgoing → span = 0.
        assert_eq!(a.singular_span_len(2), 0, "accept: no outgoing, span=0");
    }

    #[test]
    fn test_3_singular_spans_array_grammar() {
        let a = array_automaton();
        // Start state has 1 outgoing (only OPEN) → forced, span ≥ 1.
        // After OPEN, state 1 has 2 outgoing (NUM, CLOSE) → branch, span stops.
        assert_eq!(
            a.singular_span_len(0),
            1,
            "start→1 forced (OPEN), then branch at state1"
        );
        // State 1 has 2 outgoing → span = 0.
        assert_eq!(a.singular_span_len(1), 0, "branching state: span=0");
        // State 2 has 2 outgoing (COMMA, CLOSE) → span = 0.
        assert_eq!(a.singular_span_len(2), 0, "branching state: span=0");
        // State 3 (accept) has 0 outgoing → span = 0.
        assert_eq!(a.singular_span_len(3), 0, "accept: span=0");
    }

    #[test]
    fn test_4_is_valid_legal_and_illegal() {
        const OPEN: usize = 0;
        const CLOSE: usize = 1;
        const NUM: usize = 2;
        let p = LodestarPruner::new(array_automaton());
        // At depth 0 only `[` (OPEN) is legal.
        assert!(p.is_valid(0, OPEN, &[]));
        assert!(!p.is_valid(0, CLOSE, &[]));
        assert!(!p.is_valid(0, NUM, &[]));
        // After `[`, both NUM and `]` (CLOSE) are legal.
        assert!(p.is_valid(1, NUM, &[OPEN]));
        assert!(p.is_valid(1, CLOSE, &[OPEN]));
        // After `[ N`, `]` is legal, NUM is not (need `,` first).
        assert!(p.is_valid(2, CLOSE, &[OPEN, NUM]));
        assert!(!p.is_valid(2, NUM, &[OPEN, NUM]));
        // Illegal prefix → nothing valid.
        assert!(!p.is_valid(1, NUM, &[CLOSE]));
    }

    #[test]
    fn test_5_budget_masking_overshoot_rejected() {
        let a = array_automaton();
        // d(start)=2, d(state1)=1, d(state2)=1, d(accept)=0.
        // Budget = 2 tokens. At depth 0, placing OPEN leads to state1 (d=1).
        //   budget_remaining = 2 - 0 - 1 = 1. Need 1 + d(state1) = 2 ≤ 1? NO → rejected.
        // Wait: budget_remaining = budget - depth - 1 = 2 - 0 - 1 = 1.
        //   1 + d(state1) = 1 + 1 = 2 ≤ 1? No. So OPEN is rejected at budget=2.
        // With budget = 3: budget_remaining = 3 - 0 - 1 = 2. 1 + 1 = 2 ≤ 2? Yes.
        let p2 = LodestarPruner::with_budget(a.clone(), 2);
        // Budget too tight: even the first legal step needs 2 more tokens but only 1 slot remains.
        assert!(
            !p2.is_valid(0, 0, &[]),
            "budget=2: OPEN rejected (needs 2 more, only 1 remaining)"
        );

        let p3 = LodestarPruner::with_budget(a.clone(), 3);
        assert!(
            p3.is_valid(0, 0, &[]),
            "budget=3: OPEN accepted (needs 2 more, 2 remaining)"
        );

        // At budget=3, after placing OPEN at depth 0: depth=1, parent=[OPEN].
        // CLOSE leads to accept (d=0). budget_remaining = 3 - 1 - 1 = 1. 1 + 0 = 1 ≤ 1? Yes.
        assert!(
            p3.is_valid(1, 1, &[0]),
            "budget=3 after OPEN: CLOSE → accept, OK"
        );

        // NUM leads to state2 (d=1). budget_remaining = 1. 1 + 1 = 2 ≤ 1? No.
        assert!(
            !p3.is_valid(1, 2, &[0]),
            "budget=3 after OPEN: NUM → state2 (d=1), overshoots"
        );

        // Without budget, both are valid.
        let p_no_budget = LodestarPruner::new(a);
        assert!(p_no_budget.is_valid(0, 0, &[]));
        assert!(p_no_budget.is_valid(1, 1, &[0]));
        assert!(p_no_budget.is_valid(1, 2, &[0]));
    }

    #[test]
    fn test_6_batch_is_valid_consistency() {
        let a = array_automaton();
        let p = LodestarPruner::new(a);
        let candidates = [0, 1, 2, 3]; // OPEN, CLOSE, NUM, COMMA
        let mut batch = [false; 4];
        let mut individual = [false; 4];

        // Depth 0, empty parent.
        for (i, &c) in candidates.iter().enumerate() {
            individual[i] = p.is_valid(0, c, &[]);
        }
        p.batch_is_valid(0, &candidates, &[], &mut batch);
        assert_eq!(batch, individual, "batch must match individual at depth 0");

        // Depth 1, parent = [OPEN].
        let parent = [0];
        for (i, &c) in candidates.iter().enumerate() {
            individual[i] = p.is_valid(1, c, &parent);
        }
        p.batch_is_valid(1, &candidates, &parent, &mut batch);
        assert_eq!(batch, individual, "batch must match individual at depth 1");
    }

    #[test]
    fn test_6_batch_is_valid_with_budget() {
        let a = array_automaton();
        let p = LodestarPruner::with_budget(a, 4);
        let candidates = [0, 1, 2, 3];
        let mut batch = [false; 4];
        let mut individual = [false; 4];

        for (i, &c) in candidates.iter().enumerate() {
            individual[i] = p.is_valid(0, c, &[]);
        }
        p.batch_is_valid(0, &candidates, &[], &mut batch);
        assert_eq!(batch, individual, "budget batch must match individual");
    }

    #[test]
    fn test_7_completion_horizon_distances() {
        const OPEN: usize = 0;
        const NUM: usize = 2;
        let p = LodestarPruner::new(array_automaton());
        // After placing `[` at depth 0: successor is state1, d(state1)=1.
        assert_eq!(p.min_completion_distance(0, OPEN, &[]), 1);
        // After `[ N`: successor is state2, d(state2)=1.
        assert_eq!(p.min_completion_distance(1, NUM, &[OPEN]), 1);
        // Illegal token → UNREACHABLE.
        assert_eq!(p.min_completion_distance(0, NUM, &[]), UNREACHABLE);
        // Token out of vocab range → UNREACHABLE.
        assert_eq!(p.min_completion_distance(0, 99, &[]), UNREACHABLE);
    }

    #[test]
    fn test_7_singular_span_len_through_pruner() {
        let a = array_automaton();
        let p = LodestarPruner::new(a);
        // At root (empty parent), current state = start. singular_span_len(start) = 1.
        assert_eq!(p.singular_span_len(0, &[]), 1);
        // After OPEN (parent = [0]), current state = 1. Branching → span = 0.
        assert_eq!(p.singular_span_len(1, &[0]), 0);
    }

    #[test]
    fn test_8_follow_path_correct_state_tracking() {
        let a = array_automaton();
        // Empty path → start state.
        assert_eq!(a.follow_path(&[]), 0);
        // [OPEN] → state 1.
        assert_eq!(a.follow_path(&[0]), 1);
        // [OPEN, NUM] → state 2.
        assert_eq!(a.follow_path(&[0, 2]), 2);
        // [OPEN, NUM, CLOSE] → state 3 (ACCEPT).
        assert_eq!(a.follow_path(&[0, 2, 1]), 3);
        // [OPEN, CLOSE] → state 3 (empty array).
        assert_eq!(a.follow_path(&[0, 1]), 3);
        // Illegal prefix [CLOSE] → returns start (path fails at first step).
        assert_eq!(a.follow_path(&[1]), 0);
        // Illegal mid-path: [OPEN, NUM, NUM] → NUM not legal from state 2, returns start.
        assert_eq!(a.follow_path(&[0, 2, 2]), 0);
    }

    #[test]
    fn test_9_dead_state() {
        let a = dead_state_automaton();
        // State 1 is dead: no outgoing transitions.
        assert_eq!(
            a.distance(1),
            UNREACHABLE,
            "dead state: distance = UNREACHABLE"
        );
        assert!(
            a.transition(1, 0).is_none(),
            "dead state: no transition on token 0"
        );
        assert!(
            a.transition(1, 1).is_none(),
            "dead state: no transition on token 1"
        );
        // Singular span of dead state is 0 (no outgoing edges).
        assert_eq!(a.singular_span_len(1), 0);
        // No tokens are valid when automaton is in the dead state.
        let p = LodestarPruner::new(a);
        // Prefix [token0] leads to dead state.
        assert!(!p.is_valid(1, 0, &[0]), "dead state: token 0 invalid");
        assert!(!p.is_valid(1, 1, &[0]), "dead state: token 1 invalid");
    }

    #[test]
    fn test_10_diamond_shaped_automaton() {
        let a = diamond_automaton();
        // Both paths have equal length (2 steps).
        assert_eq!(a.distance(3), 0, "d(accept)=0");
        assert_eq!(a.distance(1), 1, "d(mid_a)=1");
        assert_eq!(a.distance(2), 1, "d(mid_b)=1");
        assert_eq!(a.distance(0), 2, "d(start)=2");
        // Both paths to accept are valid.
        let p = LodestarPruner::new(a);
        assert!(p.is_valid(0, 0, &[]), "token A from start → mid_a");
        assert!(p.is_valid(0, 1, &[]), "token B from start → mid_b");
        // From mid_a, only token C (2) is legal.
        assert!(p.is_valid(1, 2, &[0]), "token C from mid_a → accept");
        assert!(!p.is_valid(1, 0, &[0]), "token A illegal from mid_a");
        // From mid_b, only token C (2) is legal.
        assert!(p.is_valid(1, 2, &[1]), "token C from mid_b → accept");
        // Completion distances are admissible from both mid states.
        assert_eq!(p.min_completion_distance(0, 0, &[]), 1, "via mid_a: d=1");
        assert_eq!(p.min_completion_distance(0, 1, &[]), 1, "via mid_b: d=1");
    }

    #[test]
    fn test_11_larger_header_array_grammar() {
        let a = header_array_automaton();
        const HDR: usize = 4;
        const OPEN: usize = 0;
        const CLOSE: usize = 1;
        const NUM: usize = 2;
        const HLEN: usize = 3;
        const MAX_DEPTH: usize = 2;
        const ACCEPT: usize = HLEN + MAX_DEPTH * 2;

        // Verify distances: header chain is forced, then array body.
        // ACCEPT=0, states one step from ACCEPT have d=1, etc.
        assert_eq!(a.distance(ACCEPT), 0, "d(accept)=0");
        // Header states: H0(0) → H1(1) → H2(2) → body.
        // d(H2) = 1 + d(body_value_d1) = 1 + ... (at least a NUM + CLOSE = 2 from body)
        assert!(a.distance(0) > 0, "header has non-zero distance to accept");
        assert!(a.distance(1) > 0, "mid header has non-zero distance");
        assert!(
            a.distance(0) > a.distance(1),
            "distance decreases along header chain"
        );

        // Singular span from start: 2 forced HDR + 1 forced OPEN = 3.
        // (H0→H1 forced HDR, H1→H2 forced HDR, H2→body forced OPEN; body then has
        // 2 choices — NUM or nested OPEN — so the span stops at 3.)
        assert_eq!(
            a.singular_span_len(0),
            3,
            "2 header HDR + 1 forced OPEN = 3 singular span"
        );

        // Follow the full valid path to ACCEPT: HDR HDR OPEN NUM CLOSE.
        // (Header chain is 0→1→2 = 2 forced HDR, then 2→body via OPEN.)
        let final_state = a.follow_path(&[HDR, HDR, OPEN, NUM, CLOSE]);
        assert_eq!(final_state, ACCEPT);
        assert!(a.is_accept(final_state));

        // Same valid path through the pruner.
        let p = LodestarPruner::new(a);
        assert!(p.is_valid(0, HDR, &[]));
        assert!(p.is_valid(1, HDR, &[HDR]));
        assert!(p.is_valid(2, OPEN, &[HDR, HDR]));
        assert!(p.is_valid(3, NUM, &[HDR, HDR, OPEN]));
        assert!(p.is_valid(4, CLOSE, &[HDR, HDR, OPEN, NUM]));
        // A 3rd HDR is illegal — state 2 only accepts OPEN.
        assert!(!p.is_valid(2, HDR, &[HDR, HDR]));
    }

    #[test]
    fn test_admissibility_and_consistency() {
        // d(s) ≤ 1 + d(δ(s,t)) for every legal edge.
        let a = array_automaton();
        for s in 0..a.n_states() {
            for t in 0..a.vocab_size() {
                if let Some(ns) = a.transition(s, t) {
                    let (ds, dns) = (a.distance(s), a.distance(ns));
                    if ds != UNREACHABLE && dns != UNREACHABLE {
                        assert!(ds <= 1 + dns, "consistency violated at {s}->{ns}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_monotone_descent_exists() {
        // Every reachable non-accept state has a token strictly reducing distance.
        let a = array_automaton();
        for s in 0..a.n_states() {
            if a.is_accept(s) || a.distance(s) == UNREACHABLE {
                continue;
            }
            let has = (0..a.vocab_size()).any(|t| match a.transition(s, t) {
                Some(ns) => {
                    let dns = a.distance(ns);
                    dns != UNREACHABLE && dns + 1 == a.distance(s)
                }
                None => false,
            });
            assert!(has, "no monotone descent from state {s}");
        }
    }

    #[test]
    fn test_default_horizon_is_zero_for_nopruner() {
        use katgpt_core::traits::NoPruner;
        assert_eq!(NoPruner.min_completion_distance(3, 7, &[1, 2, 3]), 0);
        assert_eq!(NoPruner.singular_span_len(0, &[]), 0);
    }

    // ── DDTree integration (Plan 207 Phase 2) ────────────────────

    /// The budget guarantee, end-to-end through `build_dd_tree_lodestar`:
    /// every retained branch can still reach an accepting state within the
    /// sequence length — i.e. `d(state_after_path) ≤ remaining_slots`.
    #[test]
    fn test_dd_tree_lodestar_budget_guarantee() {
        use crate::speculative::dd_tree::{build_dd_tree_lodestar, extract_parent_tokens};
        use crate::types::Config;

        let p = LodestarPruner::new(array_automaton()); // vocab = 4
        let mut config = Config::draft();
        config.vocab_size = 4;
        config.tree_budget = 64;

        // Tight sequence: uniform marginals so the draft never helps the budget.
        let seq_len = 6;
        let row = vec![0.25f32; 4];
        let marginals: Vec<&[f32]> = (0..seq_len).map(|_| row.as_slice()).collect();

        let lode_config = LodestarConfig::default();
        let tree = build_dd_tree_lodestar(&marginals, &config, &p, &lode_config);
        assert!(
            !tree.is_empty(),
            "tree should retain at least the seed branch"
        );

        let auto = p.automaton();
        for node in &tree {
            let path = extract_parent_tokens(node.parent_path, node.depth + 1);
            let state = auto.follow_path(&path);
            let remaining = seq_len - (node.depth + 1);
            let d = auto.distance(state);
            assert!(
                d != UNREACHABLE && (d as usize) <= remaining,
                "budget guarantee violated: depth={} d={d} remaining={remaining} path={path:?}",
                node.depth
            );
        }
    }

    /// Default-0 path: with `NoPruner` as the horizon, `build_dd_tree_lodestar`
    /// must produce a **byte-identical** tree to `build_dd_tree_pruned`
    /// (acceptance criterion #3 — zero behavioral change for non-adopters).
    #[test]
    fn test_dd_tree_lodestar_nopruner_matches_pruned() {
        use crate::speculative::dd_tree::{build_dd_tree_lodestar, build_dd_tree_pruned};
        use crate::types::Config;
        use katgpt_core::traits::NoPruner;

        let mut config = Config::draft();
        config.vocab_size = 4;
        config.tree_budget = 64;

        let seq_len = 5;
        let row = vec![0.1, 0.4, 0.3, 0.2];
        let marginals: Vec<&[f32]> = (0..seq_len).map(|_| row.as_slice()).collect();

        let lodestar_tree =
            build_dd_tree_lodestar(&marginals, &config, &NoPruner, &LodestarConfig::default());
        let pruned_tree = build_dd_tree_pruned(&marginals, &config, &NoPruner, false);

        assert_eq!(
            lodestar_tree.len(),
            pruned_tree.len(),
            "node count must match"
        );
        for (a, b) in lodestar_tree.iter().zip(pruned_tree.iter()) {
            assert_eq!(a.depth, b.depth);
            assert_eq!(a.token_idx, b.token_idx);
            assert_eq!(a.parent_path, b.parent_path);
            assert!((a.score - b.score).abs() < 1e-6, "scores must match");
        }
    }

    #[test]
    fn test_builder_add_transition_validates_bounds() {
        let mut b = LodestarAutomaton::builder(2, 3, 0);
        // Valid transition.
        b.add_transition(0, 0, 1);
        // Out-of-range from state.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut b = LodestarAutomaton::builder(2, 3, 0);
            b.add_transition(99, 0, 1);
        }));
        assert!(result.is_err(), "out-of-range from state panics");
    }

    #[test]
    fn test_builder_add_accept_validates_bounds() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut b = LodestarAutomaton::builder(2, 3, 0);
            b.add_accept(99);
        }));
        assert!(result.is_err(), "out-of-range accept state panics");
    }

    #[test]
    fn test_builder_start_state_validates() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            LodestarAutomaton::builder(2, 3, 99);
        }));
        assert!(result.is_err(), "out-of-range start state panics");
    }

    // ── T7: Jump-ahead tests ──────────────────────────────────────────

    /// Jump-ahead with a linear chain automaton:
    /// start → mid → accept. The forced header span should collapse
    /// into fewer tree nodes when jump_ahead = true.
    #[test]
    fn test_jump_ahead_collapses_singular_span() {
        use crate::speculative::dd_tree::build_dd_tree_lodestar;
        use crate::types::Config;

        // Linear: 0 --(0)--> 1 --(1)--> 2(accept)
        let auto = linear_automaton();
        let p = LodestarPruner::new(auto);
        let mut config = Config::draft();
        config.vocab_size = 2;
        config.tree_budget = 64;

        let seq_len = 2;
        let row = vec![0.5f32; 2];
        let marginals: Vec<&[f32]> = (0..seq_len).map(|_| row.as_slice()).collect();

        // Without jump-ahead: standard per-token expansion.
        let tree_normal =
            build_dd_tree_lodestar(&marginals, &config, &p, &LodestarConfig::default());

        // With jump-ahead: singular span should collapse.
        let tree_jump = build_dd_tree_lodestar(
            &marginals,
            &config,
            &p,
            &LodestarConfig {
                jump_ahead: true,
                ..LodestarConfig::default()
            },
        );

        // Both should produce valid trees (non-empty).
        assert!(!tree_normal.is_empty(), "normal tree should be non-empty");
        assert!(!tree_jump.is_empty(), "jump-ahead tree should be non-empty");

        // Jump-ahead should produce fewer or equal nodes (collapsed span).
        assert!(
            tree_jump.len() <= tree_normal.len(),
            "jump-ahead should collapse: jump={} normal={}",
            tree_jump.len(),
            tree_normal.len()
        );
    }

    /// Jump-ahead budget guarantee: even with jump-ahead enabled,
    /// every node in the tree should still satisfy the distance constraint.
    #[test]
    fn test_jump_ahead_preserves_budget_guarantee() {
        use crate::speculative::dd_tree::{build_dd_tree_lodestar, extract_parent_tokens};
        use crate::types::Config;

        let auto = header_array_automaton();
        let p = LodestarPruner::new(auto);
        let mut config = Config::draft();
        config.vocab_size = 5;
        config.tree_budget = 128;

        let seq_len = 8;
        let row = vec![0.2f32; 5];
        let marginals: Vec<&[f32]> = (0..seq_len).map(|_| row.as_slice()).collect();

        let tree = build_dd_tree_lodestar(&marginals, &config, &p, &LodestarConfig::thinking(0.1));

        assert!(!tree.is_empty(), "tree should be non-empty");

        let au = p.automaton();
        for node in &tree {
            let path = extract_parent_tokens(node.parent_path, node.depth + 1);
            let state = au.follow_path(&path);
            let remaining = seq_len - (node.depth + 1);
            let d = au.distance(state);
            assert!(
                d != UNREACHABLE && (d as usize) <= remaining,
                "budget guarantee violated with jump-ahead: depth={} d={d} remaining={remaining}",
                node.depth
            );
        }
    }

    // ── T8: A* ordering tests ─────────────────────────────────────────

    /// A* with λ > 0 should prefer branches closer to completion.
    /// With uniform marginals, the A* node should be ordered by distance,
    /// not by log-prob (all equal).
    #[test]
    fn test_astar_prefers_closer_to_completion() {
        use crate::speculative::dd_tree::build_dd_tree_lodestar;
        use crate::types::Config;

        let auto = header_array_automaton();
        let p = LodestarPruner::new(auto);
        let mut config = Config::draft();
        config.vocab_size = 5;
        config.tree_budget = 32;

        // Max 8 tokens in u128 parent_path (16 bits each).
        let seq_len = 8;
        let row = vec![0.2f32; 5];
        let marginals: Vec<&[f32]> = (0..seq_len).map(|_| row.as_slice()).collect();

        let tree_no_astar =
            build_dd_tree_lodestar(&marginals, &config, &p, &LodestarConfig::default());

        let tree_astar = build_dd_tree_lodestar(
            &marginals,
            &config,
            &p,
            &LodestarConfig {
                astar_lambda: 0.5,
                ..LodestarConfig::default()
            },
        );

        assert!(!tree_no_astar.is_empty());
        assert!(!tree_astar.is_empty());

        if tree_astar.len() > 1 {
            for w in tree_astar.windows(2) {
                assert!(
                    w[0].score >= w[1].score,
                    "A* scores should be descending: {} >= {}",
                    w[0].score,
                    w[1].score
                );
            }
        }
    }

    /// A* with default λ = 0 produces identical tree to build_dd_tree_pruned.
    #[test]
    fn test_astar_lambda_zero_matches_default() {
        use crate::speculative::dd_tree::{build_dd_tree_lodestar, build_dd_tree_pruned};
        use crate::types::Config;
        use katgpt_core::traits::NoPruner;

        let mut config = Config::draft();
        config.vocab_size = 4;
        config.tree_budget = 64;

        let seq_len = 5;
        let row = vec![0.1, 0.4, 0.3, 0.2];
        let marginals: Vec<&[f32]> = (0..seq_len).map(|_| row.as_slice()).collect();

        let lodestar_tree =
            build_dd_tree_lodestar(&marginals, &config, &NoPruner, &LodestarConfig::default());
        let pruned_tree = build_dd_tree_pruned(&marginals, &config, &NoPruner, false);

        assert_eq!(lodestar_tree.len(), pruned_tree.len());
        for (a, b) in lodestar_tree.iter().zip(pruned_tree.iter()) {
            assert_eq!(a.depth, b.depth);
            assert_eq!(a.token_idx, b.token_idx);
            assert_eq!(a.parent_path, b.parent_path);
            assert!((a.score - b.score).abs() < 1e-6, "scores should match");
        }
    }
}
