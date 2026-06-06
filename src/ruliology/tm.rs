//! TM Strategy — Turing Machine as a SimpleProgram bandit arm.
//!
//! Implements the simplest Turing machines (1-state, 2-symbol) for Wolfram's ruliology.
//! 36 possible 1-state machines (6 configs per symbol × 2 symbols).
//!
//! Plan 188 Phase 3.

use crate::ruliology::types::SimpleProgram;

/// Default tape width for TM simulation.
const DEFAULT_TAPE_WIDTH: usize = 7;

// ── TmStrategy ─────────────────────────────────────────────────

/// A Turing machine strategy for 2-action iterated games.
///
/// Each state has: for symbol 0 and symbol 1, (write, direction, next_state).
/// Direction: 0 = left, 1 = right, 2 = stay.
///
/// With 1 state and 2 symbols, there are 36 possible machines:
/// Each symbol has 2 (write) × 3 (direction) × 1 (next_state) = 6 configs.
/// Total: 6 × 6 = 36 machines.
///
/// The tape is initialized from opponent history, and the TM runs one step
/// per game round, returning the symbol written.
#[derive(Clone, Debug)]
pub struct TmStrategy {
    /// For each symbol (0, 1): (write, direction, next_state).
    /// direction: 0 = left, 1 = right, 2 = stay.
    transitions: [(u8, u8, u8); 2],
    /// Current head position.
    head: usize,
    /// Tape (grows as needed).
    tape: Vec<u8>,
    /// Current state.
    state: u8,
    /// Number of states.
    n_states: u8,
    /// Complexity (cached at construction).
    complexity: f32,
}

impl TmStrategy {
    /// Create a new TM strategy.
    ///
    /// `transitions[0]` = (write, direction, next_state) for symbol 0.
    /// `transitions[1]` = (write, direction, next_state) for symbol 1.
    /// `n_states` = number of states (1 for simplest TMs).
    /// `tape_width` = initial tape size.
    pub fn new(transitions: [(u8, u8, u8); 2], n_states: u8, tape_width: usize) -> Self {
        let complexity = Self::compute_complexity(&transitions, n_states);
        let w = tape_width.max(3);
        Self {
            transitions,
            head: w / 2,
            tape: vec![0u8; w],
            state: 0,
            n_states,
            complexity,
        }
    }

    /// Transition table.
    #[inline]
    pub fn transitions(&self) -> &[(u8, u8, u8); 2] {
        &self.transitions
    }

    /// Current head position.
    #[inline]
    pub fn head(&self) -> usize {
        self.head
    }

    /// Current tape contents.
    #[inline]
    pub fn tape(&self) -> &[u8] {
        &self.tape
    }

    /// Current state.
    #[inline]
    pub fn state(&self) -> u8 {
        self.state
    }

    /// Number of states.
    #[inline]
    pub fn n_states(&self) -> u8 {
        self.n_states
    }

    /// Complexity = distinct transition behaviors / max possible.
    ///
    /// Each transition has (write, direction, next_state). We count distinct
    /// triplets across all symbol entries, normalized by total entries.
    fn compute_complexity(transitions: &[(u8, u8, u8); 2], n_states: u8) -> f32 {
        let mut vals = Vec::with_capacity(6);
        for &(w, d, ns) in transitions {
            vals.push(w);
            vals.push(d);
            vals.push(ns);
        }
        vals.sort();
        vals.dedup();

        let distinct = vals.len() as f32;
        // Max distinct: write has 2 values, direction has 3, next_state has n_states.
        let max_distinct = (2 + 3 + n_states as usize) as f32;
        if max_distinct == 0.0 {
            return 0.0;
        }
        distinct / max_distinct
    }

    /// Enumerate all 1-state, 2-symbol Turing machines.
    ///
    /// For 1 state, 2 symbols:
    /// - Each symbol has: write ∈ {0,1}, direction ∈ {0,1,2}, next_state ∈ {0}
    /// - So each symbol has 2 × 3 × 1 = 6 configs
    /// - Total: 6 × 6 = 36 machines
    pub fn enumerate_1_state() -> Vec<TmStrategy> {
        let mut machines = Vec::with_capacity(36);

        for sym0_config in 0..6 {
            for sym1_config in 0..6 {
                let t0 = Self::decode_transition(sym0_config, 1);
                let t1 = Self::decode_transition(sym1_config, 1);
                machines.push(TmStrategy::new([t0, t1], 1, DEFAULT_TAPE_WIDTH));
            }
        }

        machines
    }

    /// Decode a packed config index into (write, direction, next_state).
    ///
    /// Encoding: config = write + direction * 2 + next_state * 6
    /// For 1 state: next_state is always 0.
    fn decode_transition(config: u8, n_states: u8) -> (u8, u8, u8) {
        let write = config % 2;
        let direction = (config / 2) % 3;
        let next_state = (config / 6) % n_states.max(1);
        (write, direction, next_state)
    }

    /// Reset TM to initial state for a new game.
    pub fn reset(&mut self) {
        self.state = 0;
        self.head = self.tape.len() / 2;
        self.tape.fill(0);
    }

    /// Ensure tape is large enough for head position.
    fn ensure_tape_size(&mut self) {
        if self.head >= self.tape.len() {
            // Double the tape, placing new cells (0) at the end.
            let old_len = self.tape.len();
            self.tape.resize(old_len * 2, 0);
        }
    }
}

impl SimpleProgram for TmStrategy {
    /// Produce next action given opponent's action history.
    ///
    /// 1. Write opponent's last move into the tape at the current head position
    ///    (or leave tape as-is if no history).
    /// 2. Read symbol at head position.
    /// 3. Apply transition: write symbol, move head, change state.
    /// 4. Return the symbol written.
    fn next_action(&mut self, opponent_history: &[u8]) -> u8 {
        // If there's opponent history, write the last move to the tape at head.
        // This initializes the tape from the game context.
        if let Some(&last) = opponent_history.last() {
            let val = if last > 0 { 1 } else { 0 };
            self.ensure_tape_size();
            self.tape[self.head] = val;
        }

        self.ensure_tape_size();

        // Read current symbol.
        let symbol = self.tape[self.head];

        // Look up transition for this symbol.
        let (write, direction, next_state) = self.transitions[symbol as usize];

        // Write new symbol.
        self.tape[self.head] = write;

        // Move head.
        let tape_len = self.tape.len();
        match direction {
            0 => {
                // Left.
                self.head = if self.head == 0 {
                    tape_len - 1
                } else {
                    self.head - 1
                };
            }
            1 => {
                // Right.
                self.head = (self.head + 1) % tape_len;
            }
            _ => {
                // Stay — no movement.
            }
        }

        // Change state (clamped to valid range).
        self.state = if next_state < self.n_states {
            next_state
        } else {
            0
        };

        write
    }

    /// Blake3 hash of transition table → u64 ID.
    fn id(&self) -> u64 {
        let mut hasher = blake3::Hasher::new();
        for &(w, d, ns) in &self.transitions {
            hasher.update(&[w, d, ns]);
        }
        hasher.update(&[self.n_states]);

        let hash = hasher.finalize();
        let bytes: [u8; 8] = hash.as_bytes()[..8].try_into().unwrap_or([0u8; 8]);
        u64::from_le_bytes(bytes)
    }

    /// Complexity score (cached at construction).
    #[inline]
    fn complexity(&self) -> f32 {
        self.complexity
    }
}

impl PartialEq for TmStrategy {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}
impl Eq for TmStrategy {}

impl std::hash::Hash for TmStrategy {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tm_enumerate_1_state_count() {
        let machines = TmStrategy::enumerate_1_state();
        assert_eq!(
            machines.len(),
            36,
            "1-state 2-symbol TMs have exactly 36 machines"
        );
    }

    #[test]
    fn test_tm_next_action_basic() {
        // TM that always writes 1 and moves right, regardless of input.
        // Symbol 0: (write=1, dir=right=1, next=0)
        // Symbol 1: (write=1, dir=right=1, next=0)
        let mut tm = TmStrategy::new([(1, 1, 0), (1, 1, 0)], 1, 7);

        // No history: tape is all 0s, head reads 0, writes 1, moves right.
        let action = tm.next_action(&[]);
        assert_eq!(action, 1, "should write 1");

        // After one step: tape has 1 at position 3 (center of 7), head at 4.
        // History [1]: write 1 at position 4, read 0 (tape[4] is still 0), write 1.
        let action = tm.next_action(&[1]);
        assert_eq!(action, 1, "should still write 1");
    }

    #[test]
    fn test_tm_next_action_identity() {
        // TM that always writes what it reads and stays: identity machine.
        // Symbol 0: (write=0, dir=stay=2, next=0)
        // Symbol 1: (write=1, dir=stay=2, next=0)
        let mut tm = TmStrategy::new([(0, 2, 0), (1, 2, 0)], 1, 7);

        // No history: tape all 0s, reads 0, writes 0.
        assert_eq!(tm.next_action(&[]), 0);

        // History [1]: writes 1 at head, reads 1 (just wrote), writes 1.
        assert_eq!(tm.next_action(&[1]), 1);
    }

    #[test]
    fn test_tm_next_action_invert() {
        // TM that inverts: reads 0 → writes 1, reads 1 → writes 0.
        // Symbol 0: (write=1, dir=stay=2, next=0)
        // Symbol 1: (write=0, dir=stay=2, next=0)
        let mut tm = TmStrategy::new([(1, 2, 0), (0, 2, 0)], 1, 7);

        // No history: reads 0, writes 1.
        assert_eq!(tm.next_action(&[]), 1);

        // History [0]: writes 0 at head, reads 0, writes 1.
        assert_eq!(tm.next_action(&[0]), 1);

        // History [1]: writes 1 at head, reads 1, writes 0.
        let mut tm2 = TmStrategy::new([(1, 2, 0), (0, 2, 0)], 1, 7);
        assert_eq!(tm2.next_action(&[1]), 0);
    }

    #[test]
    fn test_tm_head_movement_left() {
        // TM that writes 1 and moves left.
        let mut tm = TmStrategy::new([(1, 0, 0), (1, 0, 0)], 1, 7);
        let initial_head = tm.head();

        tm.next_action(&[]);
        assert_eq!(tm.head(), initial_head - 1, "head should move left");

        // Move to position 0 then left should wrap.
        tm.head = 0;
        tm.next_action(&[]);
        assert_eq!(tm.head(), 6, "head should wrap to end");
    }

    #[test]
    fn test_tm_head_movement_right() {
        // TM that writes 0 and moves right.
        let mut tm = TmStrategy::new([(0, 1, 0), (0, 1, 0)], 1, 7);
        let initial_head = tm.head();

        tm.next_action(&[]);
        assert_eq!(tm.head(), initial_head + 1, "head should move right");

        // Move to last position then right should wrap.
        tm.head = 6;
        tm.next_action(&[]);
        assert_eq!(tm.head(), 0, "head should wrap to start");
    }

    #[test]
    fn test_tm_reset() {
        // TM: symbol 0 → write 1, move right; symbol 1 → write 1, move right.
        // Always moves right, never stays.
        let mut tm = TmStrategy::new([(1, 1, 0), (1, 1, 0)], 1, 7);

        // Play a few rounds — head moves right each time.
        tm.next_action(&[]);
        tm.next_action(&[1]);
        assert_ne!(tm.head(), 3, "head should have moved from center");

        tm.reset();
        assert_eq!(tm.state(), 0);
        assert_eq!(tm.head(), 3, "head should reset to center");
        assert!(
            tm.tape().iter().all(|&c| c == 0),
            "tape should be all zeros"
        );
    }

    #[test]
    fn test_tm_complexity_range() {
        let machines = TmStrategy::enumerate_1_state();
        for tm in &machines {
            let c = tm.complexity();
            assert!((0.0..=1.0).contains(&c), "complexity out of range: {c}");
        }
    }

    #[test]
    fn test_tm_all_distinct_ids() {
        let machines = TmStrategy::enumerate_1_state();
        let ids: Vec<u64> = machines.iter().map(|m| m.id()).collect();
        let unique_ids: std::collections::HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(
            ids.len(),
            unique_ids.len(),
            "duplicate IDs in TM enumeration"
        );
    }

    #[test]
    fn test_tm_decode_transition() {
        // Config 0: write=0, dir=0, next=0
        assert_eq!(TmStrategy::decode_transition(0, 1), (0, 0, 0));
        // Config 1: write=1, dir=0, next=0
        assert_eq!(TmStrategy::decode_transition(1, 1), (1, 0, 0));
        // Config 2: write=0, dir=1, next=0
        assert_eq!(TmStrategy::decode_transition(2, 1), (0, 1, 0));
        // Config 3: write=1, dir=1, next=0
        assert_eq!(TmStrategy::decode_transition(3, 1), (1, 1, 0));
        // Config 4: write=0, dir=2, next=0
        assert_eq!(TmStrategy::decode_transition(4, 1), (0, 2, 0));
        // Config 5: write=1, dir=2, next=0
        assert_eq!(TmStrategy::decode_transition(5, 1), (1, 2, 0));
    }
}

// TL;DR: TmStrategy — 1-state 2-symbol Turing machine as SimpleProgram. 36 possible machines. Reads opponent history onto tape, applies one transition per round, returns written symbol. Wrap-around tape with left/right/stay movement.
