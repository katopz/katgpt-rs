//! SpecDFA — Deterministic Finite Automaton for format spec compilation.
//!
//! Plan 259 Phase 3 (T9/T10): Compile format specs (email, phone, date, URL) into
//! DFA state transitions. Each DFA state exposes a `CompactBitmap` of allowed chars,
//! enabling O(1) `ConstraintPruner::is_valid` and BFS-based `CompletionHorizon`.
//!
//! Character-level tokens: byte values 0–255 as token indices.

use katgpt_core::traits::{CompletionHorizon, ConstraintPruner};

use super::types::CompactBitmap;

// ── DFA core types ──────────────────────────────────────────────

/// DFA state identifier. `u16` supports up to 65535 states.
pub type DfaState = u16;

/// Sentinel for the dead (non-accepting, no-exit) state.
pub const DEAD_STATE: DfaState = u16::MAX;

/// Maximum state count before we reject the DFA as too complex.
const MAX_STATE_COUNT: DfaState = 1000;

/// A single DFA transition: from one state, characters in `char_class` lead to `to_state`.
#[derive(Clone, Debug)]
pub struct DfaTransition {
    pub from_state: DfaState,
    /// Set of byte values (0–255) that trigger this transition.
    pub char_class: CompactBitmap,
    pub to_state: DfaState,
}

/// Compiled DFA for a format spec.
///
/// Transitions are sorted by `from_state` for efficient binary-search lookup.
/// `accept_states` is a bitmap of state IDs that represent valid completions.
#[derive(Clone, Debug)]
pub struct SpecDFA {
    /// Transitions sorted by `from_state` (secondary: insertion order within same state).
    pub transitions: Vec<DfaTransition>,
    /// Bitmap of accepting state IDs.
    pub accept_states: CompactBitmap,
    /// Total number of states (including start).
    pub state_count: DfaState,
    /// Initial state.
    pub start_state: DfaState,
    /// Human-readable format name (e.g. "email", "url").
    pub format_name: String,
}

/// Errors from format spec compilation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SpecCompileError {
    /// Format name not recognised.
    UnknownFormat = 0,
    /// DFA exceeded `MAX_STATE_COUNT`.
    DfaTooComplex = 1,
    /// Spec string is malformed.
    InvalidSpec = 2,
}

impl std::fmt::Display for SpecCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecCompileError::UnknownFormat => write!(f, "unknown format spec"),
            SpecCompileError::DfaTooComplex => write!(f, "DFA too complex (exceeds state limit)"),
            SpecCompileError::InvalidSpec => write!(f, "invalid spec"),
        }
    }
}

impl std::error::Error for SpecCompileError {}

// ── SpecDFA core methods ────────────────────────────────────────

impl SpecDFA {
    /// Walk the DFA from `start_state` through each byte in `chars`,
    /// returning the resulting state or `DEAD_STATE` if no transition exists.
    pub fn current_state(&self, chars: &[u8]) -> DfaState {
        let mut state = self.start_state;
        for &ch in chars {
            state = match self.next_state(state, ch) {
                Some(s) => s,
                None => return DEAD_STATE,
            };
        }
        state
    }

    /// Look up the transition for `ch` from `state`.
    #[inline]
    fn next_state(&self, state: DfaState, ch: u8) -> Option<DfaState> {
        // Binary-search for the range of transitions from `state`.
        let transitions = &self.transitions;
        let Ok(idx) = transitions.binary_search_by(|t| t.from_state.cmp(&state)) else {
            return None;
        };

        // Scan left to find the start of the range.
        let mut start = idx;
        while start > 0 && transitions[start - 1].from_state == state {
            start -= 1;
        }
        // Scan right to find the end.
        let mut end = idx + 1;
        while end < transitions.len() && transitions[end].from_state == state {
            end += 1;
        }

        // Check each transition from this state.
        for t in &transitions[start..end] {
            if t.char_class.contains(ch as usize) {
                return Some(t.to_state);
            }
        }
        None
    }

    /// Collect all byte values that have a transition from `state`.
    /// Returns a `CompactBitmap` suitable for `ConstraintPruner` token-set queries.
    pub fn allowed_chars(&self, state: DfaState) -> CompactBitmap {
        let mut allowed = CompactBitmap::empty();

        let transitions = &self.transitions;
        let Ok(idx) = transitions.binary_search_by(|t| t.from_state.cmp(&state)) else {
            return allowed;
        };

        let mut start = idx;
        while start > 0 && transitions[start - 1].from_state == state {
            start -= 1;
        }
        let mut end = idx + 1;
        while end < transitions.len() && transitions[end].from_state == state {
            end += 1;
        }

        for t in &transitions[start..end] {
            allowed.union_with(&t.char_class);
        }
        allowed
    }

    /// BFS shortest path from `state` to any accept state.
    /// Returns `u32::MAX` if unreachable.
    fn bfs_min_distance(&self, state: DfaState) -> u32 {
        if self.accept_states.contains(state as usize) {
            return 0;
        }
        if state == DEAD_STATE {
            return u32::MAX;
        }

        // BFS queue: (state, distance)
        let mut queue: Vec<(DfaState, u32)> = Vec::with_capacity(self.state_count as usize);
        let mut visited: Vec<bool> = vec![false; self.state_count as usize];
        queue.push((state, 0u32));
        visited[state as usize] = true;

        let mut head = 0usize;
        while head < queue.len() {
            let (s, d) = queue[head];
            head += 1;

            // Get transitions from s.
            if let Ok(idx) = self.transitions.binary_search_by(|t| t.from_state.cmp(&s)) {
                let mut start = idx;
                while start > 0 && self.transitions[start - 1].from_state == s {
                    start -= 1;
                }
                let mut end = idx + 1;
                while end < self.transitions.len() && self.transitions[end].from_state == s {
                    end += 1;
                }

                for t in &self.transitions[start..end] {
                    let ns = t.to_state;
                    if ns == DEAD_STATE {
                        continue;
                    }
                    if (ns as usize) < visited.len() && !visited[ns as usize] {
                        if self.accept_states.contains(ns as usize) {
                            return d + 1;
                        }
                        visited[ns as usize] = true;
                        queue.push((ns, d + 1));
                    }
                }
            }
        }
        u32::MAX
    }

    /// Count the deterministic singular-span length from `state`.
    /// A singular span is a chain where each state has exactly one outgoing transition.
    /// Bounded to prevent infinite loops on cycles (max `state_count` steps).
    fn singular_span_from(&self, state: DfaState) -> u32 {
        if state == DEAD_STATE {
            return 0;
        }
        let max_steps = self.state_count as u32;
        let mut current = state;
        let mut span = 0u32;

        for _ in 0..max_steps {
            let out = self.outgoing_count(current);
            if out != 1 {
                break;
            }
            // Exactly one transition — follow it.
            span += 1;
            if let Ok(idx) = self
                .transitions
                .binary_search_by(|t| t.from_state.cmp(&current))
            {
                let mut start = idx;
                while start > 0 && self.transitions[start - 1].from_state == current {
                    start -= 1;
                }
                if start < self.transitions.len() && self.transitions[start].from_state == current {
                    let next = self.transitions[start].to_state;
                    if next == DEAD_STATE {
                        break;
                    }
                    current = next;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        span
    }

    /// Count outgoing transitions from a state.
    fn outgoing_count(&self, state: DfaState) -> usize {
        let Ok(idx) = self
            .transitions
            .binary_search_by(|t| t.from_state.cmp(&state))
        else {
            return 0;
        };
        let mut start = idx;
        while start > 0 && self.transitions[start - 1].from_state == state {
            start -= 1;
        }
        let mut end = idx + 1;
        while end < self.transitions.len() && self.transitions[end].from_state == state {
            end += 1;
        }
        end - start
    }

    /// Sort transitions by `from_state` for binary-search lookup.
    fn ensure_sorted(&mut self) {
        self.transitions.sort_by_key(|t| t.from_state);
    }
}

// ── ConstraintPruner impl ───────────────────────────────────────

impl ConstraintPruner for SpecDFA {
    /// Check if `token_idx` is a valid byte from the current DFA state.
    /// Replays parent_tokens from start each call (simple but correct).
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // DFA operates on byte-level tokens only.
        if token_idx > 255 {
            return false;
        }

        // Replay parent tokens to determine current state.
        // parent_tokens may contain usize values that are byte values (0–255).
        let state = if parent_tokens.is_empty() {
            self.start_state
        } else {
            let bytes: Vec<u8> = parent_tokens
                .iter()
                .map(|&t| if t > 255 { 0u8 } else { t as u8 })
                .collect();
            self.current_state(&bytes)
        };

        if state == DEAD_STATE {
            return false;
        }

        // Check if token_idx (as a byte) is allowed from the current state.
        self.next_state(state, token_idx as u8).is_some()
    }
}

// ── CompletionHorizon impl ──────────────────────────────────────

impl CompletionHorizon for SpecDFA {
    /// BFS shortest path from the state reached after placing `token_idx` to any accept state.
    fn min_completion_distance(
        &self,
        _depth: usize,
        token_idx: usize,
        parent_tokens: &[usize],
    ) -> u32 {
        if token_idx > 255 {
            return u32::MAX;
        }

        // Compute the state after placing token_idx.
        let mut bytes: Vec<u8> = Vec::with_capacity(parent_tokens.len() + 1);
        for &t in parent_tokens {
            bytes.push(if t > 255 { 0u8 } else { t as u8 });
        }
        bytes.push(token_idx as u8);

        let state = self.current_state(&bytes);

        if state == DEAD_STATE {
            return u32::MAX;
        }

        self.bfs_min_distance(state)
    }

    /// Length of the deterministic singular span from the state reached by parent_tokens.
    fn singular_span_len(&self, _depth: usize, parent_tokens: &[usize]) -> u32 {
        let state = if parent_tokens.is_empty() {
            self.start_state
        } else {
            let bytes: Vec<u8> = parent_tokens
                .iter()
                .map(|&t| if t > 255 { 0u8 } else { t as u8 })
                .collect();
            self.current_state(&bytes)
        };

        if state == DEAD_STATE {
            return 0;
        }

        self.singular_span_from(state)
    }
}

// ── FormatDfaBuilder ────────────────────────────────────────────

/// Builder for format-specific DFAs.
pub struct FormatDfaBuilder;

impl FormatDfaBuilder {
    /// Build an email format DFA.
    ///
    /// States: start → local_part → at_sign → domain → dot_tld → accept
    ///
    /// Roughly validates `local@domain.tld` structure.
    pub fn email_dfa() -> SpecDFA {
        // State assignments
        const S_START: DfaState = 0;
        const S_LOCAL: DfaState = 1;
        const S_AT: DfaState = 2;
        const S_DOMAIN: DfaState = 3;
        const S_DOT_TLD: DfaState = 4;
        const S_TLD: DfaState = 5;
        const STATE_COUNT: DfaState = 6;

        // Character classes
        let alphanumeric = Self::chars_class((b'a'..=b'z').chain(b'A'..=b'Z').chain(b'0'..=b'9'));
        let local_chars = {
            let mut bm = alphanumeric.clone();
            // Allow dots, underscores, hyphens, plus in local part.
            for &c in b".-_+" {
                bm.insert(c as usize);
            }
            bm
        };
        let domain_chars = {
            let mut bm = alphanumeric.clone();
            bm.insert(b'-' as usize);
            bm
        };
        let tld_chars = Self::chars_class((b'a'..=b'z').chain(b'A'..=b'Z'));

        // Allow: incremental push is clearer than a single vec![] for this many transitions.
        #[allow(clippy::vec_init_then_push)]
        let mut transitions = Vec::with_capacity(8);
        // start → local_part (first char must be alphanumeric)
        transitions.push(DfaTransition {
            from_state: S_START,
            char_class: alphanumeric.clone(),
            to_state: S_LOCAL,
        });
        // local_part loops (local chars)
        transitions.push(DfaTransition {
            from_state: S_LOCAL,
            char_class: local_chars.clone(),
            to_state: S_LOCAL,
        });
        // local_part → @ sign
        transitions.push(DfaTransition {
            from_state: S_LOCAL,
            char_class: Self::single_char(b'@'),
            to_state: S_AT,
        });
        // @ → domain (first domain char must be alphanumeric)
        transitions.push(DfaTransition {
            from_state: S_AT,
            char_class: alphanumeric.clone(),
            to_state: S_DOMAIN,
        });
        // domain loops
        transitions.push(DfaTransition {
            from_state: S_DOMAIN,
            char_class: domain_chars,
            to_state: S_DOMAIN,
        });
        // domain → dot_tld
        transitions.push(DfaTransition {
            from_state: S_DOMAIN,
            char_class: Self::single_char(b'.'),
            to_state: S_DOT_TLD,
        });
        // dot_tld → tld
        transitions.push(DfaTransition {
            from_state: S_DOT_TLD,
            char_class: tld_chars.clone(),
            to_state: S_TLD,
        });
        // tld loops (multi-char TLD)
        transitions.push(DfaTransition {
            from_state: S_TLD,
            char_class: tld_chars,
            to_state: S_TLD,
        });

        let mut accept = CompactBitmap::empty();
        accept.insert(S_TLD as usize);

        SpecDFA {
            transitions,
            accept_states: accept,
            state_count: STATE_COUNT,
            start_state: S_START,
            format_name: "email".to_string(),
        }
    }

    /// Build a phone number DFA.
    ///
    /// States: start → area_code → exchange → number → accept
    ///
    /// Roughly validates patterns like `(123) 456-7890`, `123-456-7890`, `+1-234-567-8901`.
    pub fn phone_dfa() -> SpecDFA {
        const S_START: DfaState = 0;
        const S_AREA_OPEN: DfaState = 1; // after '(' or '+'
        const S_AREA_DIGIT: DfaState = 2; // collecting area code digits
        const S_AREA_CLOSE: DfaState = 3; // after ')' or first separator
        const S_EXCHANGE: DfaState = 4; // exchange digits
        const S_SEP: DfaState = 5; // separator after exchange
        const S_NUMBER: DfaState = 6; // subscriber number digits
        const S_ACCEPT: DfaState = 7;
        const STATE_COUNT: DfaState = 8;

        let digits = Self::chars_class(b'0'..=b'9');
        let separators = Self::chars_class([b'-', b' ', b'.'].into_iter());

        let mut transitions = Vec::with_capacity(16);

        // start → digit (area code) or '(' or '+'
        transitions.push(DfaTransition {
            from_state: S_START,
            char_class: digits.clone(),
            to_state: S_AREA_DIGIT,
        });
        transitions.push(DfaTransition {
            from_state: S_START,
            char_class: Self::single_char(b'('),
            to_state: S_AREA_OPEN,
        });
        transitions.push(DfaTransition {
            from_state: S_START,
            char_class: Self::single_char(b'+'),
            to_state: S_AREA_OPEN,
        });

        // area_open → digits
        transitions.push(DfaTransition {
            from_state: S_AREA_OPEN,
            char_class: digits.clone(),
            to_state: S_AREA_DIGIT,
        });

        // area_digit loops + → close/sep after 3 digits (simplified: allow loop + transition)
        transitions.push(DfaTransition {
            from_state: S_AREA_DIGIT,
            char_class: digits.clone(),
            to_state: S_AREA_DIGIT,
        });
        transitions.push(DfaTransition {
            from_state: S_AREA_DIGIT,
            char_class: Self::single_char(b')'),
            to_state: S_AREA_CLOSE,
        });
        transitions.push(DfaTransition {
            from_state: S_AREA_DIGIT,
            char_class: separators.clone(),
            to_state: S_AREA_CLOSE,
        });

        // area_close → exchange digits
        transitions.push(DfaTransition {
            from_state: S_AREA_CLOSE,
            char_class: digits.clone(),
            to_state: S_EXCHANGE,
        });
        transitions.push(DfaTransition {
            from_state: S_AREA_CLOSE,
            char_class: separators.clone(),
            to_state: S_AREA_CLOSE,
        });

        // exchange loops
        transitions.push(DfaTransition {
            from_state: S_EXCHANGE,
            char_class: digits.clone(),
            to_state: S_EXCHANGE,
        });
        // exchange → separator
        transitions.push(DfaTransition {
            from_state: S_EXCHANGE,
            char_class: separators.clone(),
            to_state: S_SEP,
        });

        // sep → number
        transitions.push(DfaTransition {
            from_state: S_SEP,
            char_class: digits.clone(),
            to_state: S_NUMBER,
        });

        // number loops + → accept
        transitions.push(DfaTransition {
            from_state: S_NUMBER,
            char_class: digits,
            to_state: S_NUMBER,
        });
        transitions.push(DfaTransition {
            from_state: S_NUMBER,
            char_class: separators,
            to_state: S_ACCEPT,
        });

        let mut accept = CompactBitmap::empty();
        accept.insert(S_ACCEPT as usize);
        // Also accept after sufficient digits in number state (simplified).
        accept.insert(S_NUMBER as usize);

        SpecDFA {
            transitions,
            accept_states: accept,
            state_count: STATE_COUNT,
            start_state: S_START,
            format_name: "phone".to_string(),
        }
    }

    /// Build a date format DFA.
    ///
    /// States: start → year → dash1 → month → dash2 → day → accept
    ///
    /// Validates `YYYY-MM-DD` pattern.
    pub fn date_dfa() -> SpecDFA {
        const S_START: DfaState = 0;
        const S_YEAR: DfaState = 1;
        const S_DASH1: DfaState = 2;
        const S_MONTH: DfaState = 3;
        const S_DASH2: DfaState = 4;
        const S_DAY: DfaState = 5;
        const S_ACCEPT: DfaState = 6;
        const STATE_COUNT: DfaState = 7;

        let digits = Self::chars_class(b'0'..=b'9');
        let dash = Self::single_char(b'-');

        // Allow: incremental push is clearer than a single vec![] for this many transitions.
        #[allow(clippy::vec_init_then_push)]
        let mut transitions = Vec::with_capacity(8);

        // start → year digit
        transitions.push(DfaTransition {
            from_state: S_START,
            char_class: digits.clone(),
            to_state: S_YEAR,
        });
        // year loops (4 digits)
        transitions.push(DfaTransition {
            from_state: S_YEAR,
            char_class: digits.clone(),
            to_state: S_YEAR,
        });
        // year → dash1
        transitions.push(DfaTransition {
            from_state: S_YEAR,
            char_class: dash.clone(),
            to_state: S_DASH1,
        });
        // dash1 → month
        transitions.push(DfaTransition {
            from_state: S_DASH1,
            char_class: digits.clone(),
            to_state: S_MONTH,
        });
        // month loops (2 digits)
        transitions.push(DfaTransition {
            from_state: S_MONTH,
            char_class: digits.clone(),
            to_state: S_MONTH,
        });
        // month → dash2
        transitions.push(DfaTransition {
            from_state: S_MONTH,
            char_class: dash,
            to_state: S_DASH2,
        });
        // dash2 → day
        transitions.push(DfaTransition {
            from_state: S_DASH2,
            char_class: digits.clone(),
            to_state: S_DAY,
        });
        // day loops (2 digits) → accept
        transitions.push(DfaTransition {
            from_state: S_DAY,
            char_class: digits,
            to_state: S_ACCEPT,
        });

        let mut accept = CompactBitmap::empty();
        accept.insert(S_ACCEPT as usize);

        SpecDFA {
            transitions,
            accept_states: accept,
            state_count: STATE_COUNT,
            start_state: S_START,
            format_name: "date".to_string(),
        }
    }

    /// Build a URL format DFA.
    ///
    /// States: start → scheme → colon → slashes → host → path → accept
    ///
    /// Roughly validates `https://host/path` structure.
    pub fn url_dfa() -> SpecDFA {
        const S_START: DfaState = 0;
        const S_SCHEME: DfaState = 1;
        const S_COLON: DfaState = 2;
        const S_SLASH1: DfaState = 3;
        const S_SLASH2: DfaState = 4;
        const S_HOST: DfaState = 5;
        const S_PATH: DfaState = 6;
        const S_ACCEPT: DfaState = 7;
        const STATE_COUNT: DfaState = 8;

        let alpha = Self::chars_class((b'a'..=b'z').chain(b'A'..=b'Z'));
        let alphanumeric = Self::chars_class((b'a'..=b'z').chain(b'A'..=b'Z').chain(b'0'..=b'9'));
        let scheme_chars = {
            let mut bm = alpha.clone();
            bm.insert(b'+' as usize);
            bm.insert(b'-' as usize);
            bm.insert(b'.' as usize);
            bm
        };
        let host_chars = {
            let mut bm = alphanumeric.clone();
            bm.insert(b'-' as usize);
            bm.insert(b'.' as usize);
            bm
        };
        let path_chars = {
            let mut bm = alphanumeric.clone();
            for &c in b"/-_.~?=&%" {
                bm.insert(c as usize);
            }
            bm
        };

        let mut transitions = Vec::with_capacity(16);

        // start → scheme (letter)
        transitions.push(DfaTransition {
            from_state: S_START,
            char_class: alpha,
            to_state: S_SCHEME,
        });
        // scheme loops (scheme chars)
        transitions.push(DfaTransition {
            from_state: S_SCHEME,
            char_class: scheme_chars,
            to_state: S_SCHEME,
        });
        // scheme → colon
        transitions.push(DfaTransition {
            from_state: S_SCHEME,
            char_class: Self::single_char(b':'),
            to_state: S_COLON,
        });
        // colon → slash1
        transitions.push(DfaTransition {
            from_state: S_COLON,
            char_class: Self::single_char(b'/'),
            to_state: S_SLASH1,
        });
        // slash1 → slash2
        transitions.push(DfaTransition {
            from_state: S_SLASH1,
            char_class: Self::single_char(b'/'),
            to_state: S_SLASH2,
        });
        // slash2 → host
        transitions.push(DfaTransition {
            from_state: S_SLASH2,
            char_class: alphanumeric.clone(),
            to_state: S_HOST,
        });
        // host loops
        transitions.push(DfaTransition {
            from_state: S_HOST,
            char_class: host_chars,
            to_state: S_HOST,
        });
        // host → path
        transitions.push(DfaTransition {
            from_state: S_HOST,
            char_class: Self::single_char(b'/'),
            to_state: S_PATH,
        });
        // path loops
        transitions.push(DfaTransition {
            from_state: S_PATH,
            char_class: path_chars,
            to_state: S_PATH,
        });
        // path → accept (any path char accepted)
        transitions.push(DfaTransition {
            from_state: S_PATH,
            char_class: Self::single_char(b'/'),
            to_state: S_ACCEPT,
        });

        let mut accept = CompactBitmap::empty();
        accept.insert(S_ACCEPT as usize);
        // Host with no path is also valid.
        accept.insert(S_HOST as usize);
        // A non-empty path is valid.
        accept.insert(S_PATH as usize);

        SpecDFA {
            transitions,
            accept_states: accept,
            state_count: STATE_COUNT,
            start_state: S_START,
            format_name: "url".to_string(),
        }
    }

    // ── helpers ──────────────────────────────────────────────

    /// Build a `CompactBitmap` from an iterator of byte values.
    fn chars_class(chars: impl Iterator<Item = u8>) -> CompactBitmap {
        CompactBitmap::from_token_indices(chars.map(|c| c as usize))
    }

    /// Single-character bitmap.
    fn single_char(c: u8) -> CompactBitmap {
        let mut bm = CompactBitmap::empty();
        bm.insert(c as usize);
        bm
    }
}

// ── Top-level compile function ──────────────────────────────────

/// Compile a format description string into a `SpecDFA`.
///
/// Recognised formats: `"email"`, `"phone"`, `"date"`, `"url"`.
pub fn compile_format_spec(format_desc: &str) -> Result<SpecDFA, SpecCompileError> {
    let mut dfa = match format_desc.trim().to_lowercase().as_str() {
        "email" => FormatDfaBuilder::email_dfa(),
        "phone" => FormatDfaBuilder::phone_dfa(),
        "date" => FormatDfaBuilder::date_dfa(),
        "url" => FormatDfaBuilder::url_dfa(),
        _ => return Err(SpecCompileError::UnknownFormat),
    };

    if dfa.state_count > MAX_STATE_COUNT {
        return Err(SpecCompileError::DfaTooComplex);
    }

    dfa.ensure_sorted();
    Ok(dfa)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Email DFA tests ─────────────────────────────────────

    #[test]
    fn test_email_accepts_valid() {
        let dfa = FormatDfaBuilder::email_dfa();
        let input = b"user@example.com";
        let state = dfa.current_state(input);
        assert!(
            dfa.accept_states.contains(state as usize),
            "email 'user@example.com' should reach accept state, got state {state}"
        );
    }

    #[test]
    fn test_email_rejects_no_at() {
        let dfa = FormatDfaBuilder::email_dfa();
        let input = b"invalid";
        let state = dfa.current_state(input);
        assert!(
            !dfa.accept_states.contains(state as usize),
            "email 'invalid' (no @) should NOT be in accept state, got state {state}"
        );
    }

    #[test]
    fn test_email_rejects_no_tld() {
        let dfa = FormatDfaBuilder::email_dfa();
        let input = b"user@example";
        let state = dfa.current_state(input);
        assert!(
            !dfa.accept_states.contains(state as usize),
            "email 'user@example' (no .tld) should NOT be in accept state, got state {state}"
        );
    }

    #[test]
    fn test_email_accepts_complex_local() {
        let dfa = FormatDfaBuilder::email_dfa();
        let input = b"first.last+tag@domain.org";
        let state = dfa.current_state(input);
        assert!(
            dfa.accept_states.contains(state as usize),
            "email 'first.last+tag@domain.org' should reach accept state, got state {state}"
        );
    }

    // ── Phone DFA tests ─────────────────────────────────────

    #[test]
    fn test_phone_accepts_with_parens() {
        let dfa = FormatDfaBuilder::phone_dfa();
        let input = b"(123) 456-7890";
        let state = dfa.current_state(input);
        assert!(
            dfa.accept_states.contains(state as usize),
            "phone '(123) 456-7890' should reach accept state, got state {state}"
        );
    }

    #[test]
    fn test_phone_accepts_dashes() {
        let dfa = FormatDfaBuilder::phone_dfa();
        let input = b"123-456-7890";
        let state = dfa.current_state(input);
        assert!(
            dfa.accept_states.contains(state as usize),
            "phone '123-456-7890' should reach accept state, got state {state}"
        );
    }

    // ── Date DFA tests ──────────────────────────────────────

    #[test]
    fn test_date_accepts_valid() {
        let dfa = FormatDfaBuilder::date_dfa();
        let input = b"2024-01-15";
        let state = dfa.current_state(input);
        assert!(
            dfa.accept_states.contains(state as usize),
            "date '2024-01-15' should reach accept state, got state {state}"
        );
    }

    #[test]
    fn test_date_rejects_no_dashes() {
        let dfa = FormatDfaBuilder::date_dfa();
        let input = b"20240115";
        let state = dfa.current_state(input);
        assert!(
            !dfa.accept_states.contains(state as usize),
            "date '20240115' (no dashes) should NOT reach accept state, got state {state}"
        );
    }

    // ── URL DFA tests ───────────────────────────────────────

    #[test]
    fn test_url_accepts_https() {
        let dfa = FormatDfaBuilder::url_dfa();
        let input = b"https://example.com/path";
        let state = dfa.current_state(input);
        assert!(
            dfa.accept_states.contains(state as usize),
            "url 'https://example.com/path' should reach accept state, got state {state}"
        );
    }

    #[test]
    fn test_url_accepts_host_only() {
        let dfa = FormatDfaBuilder::url_dfa();
        let input = b"http://example.com";
        let state = dfa.current_state(input);
        assert!(
            dfa.accept_states.contains(state as usize),
            "url 'http://example.com' (host-only) should reach accept state, got state {state}"
        );
    }

    // ── ConstraintPruner tests ──────────────────────────────

    #[test]
    fn test_pruner_valid_chars_email() {
        let dfa = FormatDfaBuilder::email_dfa();

        // At start, alphanumeric chars should be valid.
        assert!(dfa.is_valid(0, b'u' as usize, &[]));
        assert!(dfa.is_valid(0, b'a' as usize, &[]));
        assert!(dfa.is_valid(0, b'0' as usize, &[]));

        // '@' at start should be invalid.
        assert!(!dfa.is_valid(0, b'@' as usize, &[]));
    }

    #[test]
    fn test_pruner_invalid_after_dead() {
        let dfa = FormatDfaBuilder::email_dfa();

        // After "user@", domain chars should be valid.
        let parents: Vec<usize> = vec![
            b'u' as usize,
            b's' as usize,
            b'e' as usize,
            b'r' as usize,
            b'@' as usize,
        ];
        assert!(dfa.is_valid(6, b'e' as usize, &parents));

        // But '@' again should be invalid (no '@' transition from domain).
        assert!(!dfa.is_valid(6, b'@' as usize, &parents));
    }

    #[test]
    fn test_pruner_rejects_tokens_above_255() {
        let dfa = FormatDfaBuilder::email_dfa();
        assert!(!dfa.is_valid(0, 256, &[]));
        assert!(!dfa.is_valid(0, 1000, &[]));
    }

    // ── CompletionHorizon tests ─────────────────────────────

    #[test]
    fn test_horizon_distance_decreases() {
        let dfa = FormatDfaBuilder::email_dfa();

        // Empty: distance to accept from start (must traverse local@domain.tld).
        let d0 = dfa.min_completion_distance(0, b'u' as usize, &[]);

        // After "user@": closer to accept.
        let parents_after_at: Vec<usize> = vec![
            b'u' as usize,
            b's' as usize,
            b'e' as usize,
            b'r' as usize,
            b'@' as usize,
        ];
        let d1 = dfa.min_completion_distance(6, b'e' as usize, &parents_after_at);

        // After "user@example.": even closer (just need TLD).
        let parents_after_dot: Vec<usize> = vec![
            b'u' as usize,
            b's' as usize,
            b'e' as usize,
            b'r' as usize,
            b'@' as usize,
            b'e' as usize,
            b'x' as usize,
            b'a' as usize,
            b'm' as usize,
            b'p' as usize,
            b'l' as usize,
            b'e' as usize,
            b'.' as usize,
        ];
        let d2 = dfa.min_completion_distance(14, b'c' as usize, &parents_after_dot);

        assert!(
            d0 > d1,
            "distance should decrease: start={d0} > after_at={d1}"
        );
        assert!(
            d1 > d2,
            "distance should decrease: after_at={d1} > after_dot={d2}"
        );
    }

    #[test]
    fn test_horizon_returns_max_for_dead_state() {
        let dfa = FormatDfaBuilder::email_dfa();
        // '@' at depth 0 leads to dead state (no transition from start).
        let dist = dfa.min_completion_distance(0, b'@' as usize, &[]);
        assert_eq!(dist, u32::MAX, "dead state should return u32::MAX");
    }

    // ── compile_format_spec tests ───────────────────────────

    #[test]
    fn test_compile_email() {
        let dfa = compile_format_spec("email").unwrap();
        assert_eq!(dfa.format_name, "email");
        assert!(dfa.state_count > 0);
    }

    #[test]
    fn test_compile_phone() {
        let dfa = compile_format_spec("phone").unwrap();
        assert_eq!(dfa.format_name, "phone");
    }

    #[test]
    fn test_compile_date() {
        let dfa = compile_format_spec("date").unwrap();
        assert_eq!(dfa.format_name, "date");
    }

    #[test]
    fn test_compile_url() {
        let dfa = compile_format_spec("url").unwrap();
        assert_eq!(dfa.format_name, "url");
    }

    #[test]
    fn test_compile_unknown() {
        let err = compile_format_spec("unknown_format").unwrap_err();
        assert_eq!(err, SpecCompileError::UnknownFormat);
    }

    #[test]
    fn test_compile_trim_case_insensitive() {
        let dfa = compile_format_spec("  EMAIL  ").unwrap();
        assert_eq!(dfa.format_name, "email");
    }

    // ── Allowed chars tests ─────────────────────────────────

    #[test]
    fn test_allowed_chars_start_state() {
        let dfa = FormatDfaBuilder::email_dfa();
        let allowed = dfa.allowed_chars(dfa.start_state);

        // Should include alphanumeric.
        assert!(allowed.contains(b'a' as usize));
        assert!(allowed.contains(b'Z' as usize));
        assert!(allowed.contains(b'0' as usize));
        // Should NOT include '@' at start.
        assert!(!allowed.contains(b'@' as usize));
    }

    #[test]
    fn test_allowed_chars_dead_state() {
        let dfa = FormatDfaBuilder::email_dfa();
        let allowed = dfa.allowed_chars(DEAD_STATE);
        assert!(
            allowed.is_empty(),
            "dead state should have no allowed chars"
        );
    }

    // ── Singular span tests ─────────────────────────────────

    #[test]
    fn test_singular_span_url_colon_slash() {
        let dfa = FormatDfaBuilder::url_dfa();

        // After scheme (e.g. "http"), the next steps are ':' → '/' → '/' which is singular.
        // At colon state (S_COLON=2), only '/' is valid → singular.
        let span = dfa.singular_span_from(2);
        assert!(
            span >= 1,
            "colon state should have singular span >= 1, got {span}"
        );
    }

    // ── Error display test ──────────────────────────────────

    #[test]
    fn test_error_display() {
        assert_eq!(
            format!("{}", SpecCompileError::UnknownFormat),
            "unknown format spec"
        );
        assert_eq!(
            format!("{}", SpecCompileError::DfaTooComplex),
            "DFA too complex (exceeds state limit)"
        );
        assert_eq!(format!("{}", SpecCompileError::InvalidSpec), "invalid spec");
    }
}
