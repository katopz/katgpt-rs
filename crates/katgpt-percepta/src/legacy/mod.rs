//! Percepta-style O(log N) 2D Attention via Convex Hull KV Cache.
//!
//! Standard transformer attention computes Q·K for all N past keys → O(N) per step.
//! Percepta restricts attention heads to d=2, making the dot product a 2D geometric
//! projection. When keys form a convex hull, finding the maximum attention score
//! becomes ternary search over a unimodal (bitonic) sequence → O(log N).
//!
//! Integration points with katgpt-rs:
//! - DDTree branch pruning: validate drafted tokens before target verification
//! - Deterministic Validator: encode state-machine rules as 2D key embeddings
//! - "Free embedding" bridge: project hidden states to 2D for fast retrieval

/// 2D vector for geometric attention operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Dot product — the core attention score in 2D.
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Z-component of cross product AB × AC.
    /// Positive = left turn, Negative = right turn, Zero = collinear.
    #[inline]
    pub fn cross_z(a: &Self, b: &Self, c: &Self) -> f32 {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }
}

/// Specialized KV Cache for 2D attention heads.
/// Maintains the upper convex hull of keys for O(log N) attention lookup.
///
/// Keys must have monotonically non-decreasing X coordinates — natural for
/// sequential execution traces where position encodes time step.
pub struct KVCache2D {
    keys: Vec<Vec2>,
    values: Vec<usize>,
    upper_hull: Vec<usize>,
}

impl Default for KVCache2D {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache2D {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            upper_hull: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            upper_hull: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn hull_len(&self) -> usize {
        self.upper_hull.len()
    }

    /// Append a key-value pair. Amortized O(1) hull maintenance via Graham Scan.
    ///
    /// For keys with monotonically increasing X:
    /// - Points creating non-right turns (collinear or concave) are removed
    /// - The upper hull captures the "skyline" of the key distribution
    pub fn append(&mut self, key: Vec2, value: usize) {
        let idx = self.keys.len();
        self.keys.push(key);
        self.values.push(value);

        // Maintain upper convex hull: pop points violating convexity
        while self.upper_hull.len() >= 2 {
            let len = self.upper_hull.len();
            let a = &self.keys[self.upper_hull[len - 2]];
            let b = &self.keys[self.upper_hull[len - 1]];
            let c = &key;

            // Right turn (cross < 0) preserves convexity. Remove otherwise.
            if Vec2::cross_z(a, b, c) >= 0.0 {
                self.upper_hull.pop();
            } else {
                break;
            }
        }
        self.upper_hull.push(idx);
    }

    /// Standard O(N) attention: linear scan over all keys.
    /// Baseline for correctness verification.
    pub fn linear_attention(&self, query: &Vec2) -> (f32, usize) {
        if self.keys.is_empty() { (f32::NEG_INFINITY, 0) } else {
                let mut max_score = f32::NEG_INFINITY;
                let mut best_idx = 0;
                for (i, key) in self.keys.iter().enumerate() {
                    let score = query.dot(key);
                    if score > max_score {
                        max_score = score;
                        best_idx = i;
                    }
                }
                (max_score, self.values[best_idx])
            }
    }

    /// O(log N) attention via ternary search over the convex hull.
    ///
    /// The dot product of a fixed query against points on a convex hull
    /// forms a unimodal (bitonic) sequence: it rises to a peak then falls.
    /// Ternary search finds the peak in O(log H) where H = hull size.
    pub fn fast_attention(&self, query: &Vec2) -> (f32, usize) {
        let n = self.upper_hull.len();
        match n {
            0 => (f32::NEG_INFINITY, 0),
            1 => {
                let idx = self.upper_hull[0];
                (query.dot(&self.keys[idx]), self.values[idx])
            }
            2 => {
                let idx0 = self.upper_hull[0];
                let idx1 = self.upper_hull[1];
                let s0 = query.dot(&self.keys[idx0]);
                let s1 = query.dot(&self.keys[idx1]);
                if s0 >= s1 { (s0, self.values[idx0]) } else { (s1, self.values[idx1]) }
            }
            _ => {
                let mut left = 0usize;
                let mut right = n - 1;

                // Ternary search on unimodal dot-product sequence
                while right - left > 2 {
                    let third = (right - left) / 3;
                    let m1 = left + third;
                    let m2 = right - third;

                    let s1 = query.dot(&self.keys[self.upper_hull[m1]]);
                    let s2 = query.dot(&self.keys[self.upper_hull[m2]]);

                    if s1 < s2 { left = m1 } else { right = m2 }
                }

                // Scan the remaining 1–3 candidates
                let mut max_score = f32::NEG_INFINITY;
                let mut best_idx = self.upper_hull[left];

                for i in left..=right {
                    let idx = self.upper_hull[i];
                    let score = query.dot(&self.keys[idx]);
                    if score > max_score {
                        max_score = score;
                        best_idx = idx;
                    }
                }

                (max_score, self.values[best_idx])
            }
        }
    }

    /// Get hull indices (for debugging/testing).
    pub fn hull_indices(&self) -> &[usize] {
        &self.upper_hull
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.upper_hull.clear();
    }

    /// Get all keys (for debugging/testing).
    pub fn keys(&self) -> &[Vec2] {
        &self.keys
    }

    /// Get all values (for debugging/testing).
    pub fn values(&self) -> &[usize] {
        &self.values
    }
}

// ── 9×9 Sudoku: Public API for examples ──────────────────────────

/// 9×9 Sudoku board. 0 = empty cell, 1-9 = digit.
#[derive(Clone, Debug)]
pub struct Sudoku9x9 {
    pub grid: [[u8; 9]; 9],
}

impl Sudoku9x9 {
    /// Create from a 9×9 grid. 0 = empty.
    pub fn new(grid: [[u8; 9]; 9]) -> Self {
        Self { grid }
    }

    /// Arto Inkala's famous "World's Hardest Sudoku" (21 clues).
    pub fn arto_inkala() -> Self {
        Self::new([
            [8, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 6, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 9, 0, 2, 0, 0],
            [0, 5, 0, 0, 0, 7, 0, 0, 0],
            [0, 0, 0, 0, 4, 5, 7, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 3, 0],
            [0, 0, 1, 0, 0, 0, 0, 6, 8],
            [0, 0, 8, 5, 0, 0, 0, 1, 0],
            [0, 9, 0, 0, 0, 0, 4, 0, 0],
        ])
    }

    /// The exact puzzle from Percepta's transformer-vm `manifest.yaml` (30 clues).
    /// Source: <https://github.com/Percepta-Core/transformer-vm/blob/main/transformer_vm/examples/manifest.yaml>
    /// String: `530070000600195000098000060800060003400803001700020006060000280000419005000080079`
    pub fn percepta_reference() -> Self {
        Self::new([
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ])
    }

    /// Check if placing `digit` at (row, col) violates Sudoku rules.
    /// The "rules engine" — deterministic constraint satisfaction.
    pub fn is_valid_move(&self, row: usize, col: usize, digit: u8) -> bool {
        if digit == 0 {
            return false;
        }
        // Row constraint
        for c in 0..9 {
            if self.grid[row][c] == digit {
                return false;
            }
        }
        // Column constraint
        for r in 0..9 {
            if self.grid[r][col] == digit {
                return false;
            }
        }
        // 3×3 box constraint
        let box_r = (row / 3) * 3;
        let box_c = (col / 3) * 3;
        for r in 0..3 {
            for c in 0..3 {
                if self.grid[box_r + r][box_c + c] == digit {
                    return false;
                }
            }
        }
        true
    }

    /// Count given clues (non-zero cells).
    pub fn clue_count(&self) -> usize {
        self.grid
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v > 0)
            .count()
    }

    /// Check if the board is fully solved.
    pub fn is_solved(&self) -> bool {
        self.grid.iter().flat_map(|row| row.iter()).all(|&v| v > 0) && self.is_valid_solution()
    }

    /// Find next empty cell, returns (row, col) or None.
    pub fn next_empty(&self) -> Option<(usize, usize)> {
        for r in 0..9 {
            for c in 0..9 {
                if self.grid[r][c] == 0 {
                    return Some((r, c));
                }
            }
        }
        None
    }

    /// Pretty-print the board as a string.
    pub fn display(&self) -> String {
        let mut s = String::with_capacity(256);
        for r in 0..9 {
            if r > 0 && r % 3 == 0 {
                s.push_str("------+-------+------\n");
            }
            for c in 0..9 {
                if c > 0 && c % 3 == 0 {
                    s.push_str("| ");
                }
                match self.grid[r][c] {
                    0 => s.push_str(". "),
                    d => {
                        // '1'..='9' correspond to ASCII 49..=57.
                        s.push((b'0' + d) as char);
                        s.push(' ');
                    }
                }
            }
            s.push('\n');
        }
        s
    }

    /// Solve with KVCache2D trace. Returns true if solved.
    pub fn solve(&mut self, cache: &mut KVCache2D, step: &mut usize) -> bool {
        let filled = self.clue_count();
        cache.append(Vec2::new(*step as f32, filled as f32), *step);
        *step += 1;

        let Some((row, col)) = self.next_empty() else {
            return true;
        };

        for digit in 1..=9u8 {
            if self.is_valid_move(row, col, digit) {
                self.grid[row][col] = digit;
                if self.solve(cache, step) {
                    return true;
                }
                self.grid[row][col] = 0;
            }
        }
        false
    }

    /// Fast solver: MRV cell selection + bitmask candidate tracking +
    /// naked-singles constraint propagation.
    ///
    /// Returns `(solved, steps)`. `steps` counts recursive entries (apples-to-
    /// apples with `solve()`'s step counter). On Inkala this is typically
    /// ~100-1000× fewer steps than `solve()` because:
    ///   - Naked singles (cells with 1 candidate) are filled without branching.
    ///   - MRV picks the most-constrained cell first, shrinking the branching
    ///     factor dramatically.
    ///   - Candidate bitmasks make `is_valid` an O(1) bit check, not an O(27)
    ///     row/col/box scan.
    ///
    /// Modelless: pure deterministic rules engine, no training. Satisfies the
    /// modelless-first mandate (Issue 005 Option A+B).
    pub fn solve_fast(&mut self) -> (bool, usize) {
        // Candidate bitmask per cell: bit (d-1) set ⇒ digit d is a candidate.
        // 0b111111111 = all 9 digits valid.
        let mut cands = [[0u16; 9]; 9];
        // multi-array coordinated init: cands[r][c] derived from self.grid[r][c]
        #[allow(clippy::needless_range_loop)]
        for r in 0..9 {
            for c in 0..9 {
                cands[r][c] = if self.grid[r][c] == 0 { 0b111111111 } else { 0 };
            }
        }
        // Seed candidates from the initial clues.
        for r in 0..9 {
            for c in 0..9 {
                let d = self.grid[r][c];
                if d > 0 {
                    Self::eliminate(&mut cands, r, c, d);
                }
            }
        }
        let mut steps = 0usize;
        let solved = self.solve_fast_rec(&mut cands, &mut steps);
        (solved, steps)
    }

    /// Recursive core for `solve_fast`. Uses full-snapshot backtrack:
    /// snapshots `cands` and `grid` at entry, restores both on any false return.
    /// 162 bytes (cands) + 81 bytes (grid) per frame is trivial vs. the perf win.
    fn solve_fast_rec(&mut self, cands: &mut [[u16; 9]; 9], steps: &mut usize) -> bool {
        *steps += 1;

        // ── Naked-singles propagation: fill every cell with exactly 1 candidate. ──
        // This cascades (filling one single often creates new singles). Loop until
        // no progress. Any dead cell (0 candidates) aborts this branch.
        loop {
            let mut filled_one = false;
            let mut dead = false;
            for r in 0..9 {
                for c in 0..9 {
                    if self.grid[r][c] != 0 {
                        continue;
                    }
                    let mask = cands[r][c];
                    if mask == 0 {
                        dead = true;
                        break;
                    }
                    if mask.is_power_of_two() {
                        let d = mask.trailing_zeros() as u8 + 1;
                        self.grid[r][c] = d;
                        Self::eliminate(cands, r, c, d);
                        filled_one = true;
                    }
                }
                if dead {
                    break;
                }
            }
            if dead {
                return false;
            }
            if !filled_one {
                break;
            }
        }

        // ── Find the MRV cell: empty cell with fewest candidates. ──
        let mut best: Option<(usize, usize, u16)> = None; // (r, c, mask)
        // multi-array scan: self.grid[r][c] and cands[r][c] read in lockstep,
        // and (r,c) is captured into `best` for the branching phase below.
        #[allow(clippy::needless_range_loop)]
        for r in 0..9 {
            for c in 0..9 {
                if self.grid[r][c] != 0 {
                    continue;
                }
                let mask = cands[r][c];
                let n = mask.count_ones();
                if n == 0 {
                    return false; // dead cell
                }
                if best.is_none_or(|(_, _, bm)| n < bm.count_ones()) {
                    best = Some((r, c, mask));
                    if n == 2 {
                        break; // can't beat 2 candidates; early-exit the scan
                    }
                }
            }
        }

        let Some((row, col, mask)) = best else {
            // No empty cells left → solved.
            return true;
        };

        // ── Branch on the MRV cell's candidates. ──
        // Full-snapshot backtrack: capture cands + grid state, try each
        // candidate, restore on failure. This is correct by construction — no
        // need for tracked elimination/restore bookkeeping.
        let mut bits = mask;
        while bits != 0 {
            let bit = bits.isolate_lowest_one(); // lowest set bit
            bits ^= bit;
            let d = bit.trailing_zeros() as u8 + 1;

            // Snapshot before placing.
            let cands_snap = *cands;
            let grid_snap = self.grid;

            self.grid[row][col] = d;
            Self::eliminate(cands, row, col, d);

            if self.solve_fast_rec(cands, steps) {
                return true;
            }

            // Restore the full state for the next candidate.
            *cands = cands_snap;
            self.grid = grid_snap;
        }
        false
    }

    /// Eliminate digit `d` from the candidates of all peers (row, col, box) of
    /// `(row, col)`. Does NOT touch `(row, col)` itself.
    #[inline]
    fn eliminate(cands: &mut [[u16; 9]; 9], row: usize, col: usize, d: u8) {
        let bit = 1u16 << (d - 1);
        for cell in &mut cands[row] {
            *cell &= !bit;
        }
        for row_cells in cands.iter_mut() {
            row_cells[col] &= !bit;
        }
        let box_r = (row / 3) * 3;
        let box_c = (col / 3) * 3;
        for r in 0..3 {
            for c in 0..3 {
                cands[box_r + r][box_c + c] &= !bit;
            }
        }
    }

    /// Validate a complete board satisfies all constraints.
    fn is_valid_solution(&self) -> bool {
        for r in 0..9 {
            let mut seen = [false; 10];
            for c in 0..9 {
                let d = self.grid[r][c] as usize;
                if d == 0 || seen[d] {
                    return false;
                }
                seen[d] = true;
            }
        }
        for c in 0..9 {
            let mut seen = [false; 10];
            for r in 0..9 {
                let d = self.grid[r][c] as usize;
                if d == 0 || seen[d] {
                    return false;
                }
                seen[d] = true;
            }
        }
        for box_r in (0..9).step_by(3) {
            for box_c in (0..9).step_by(3) {
                let mut seen = [false; 10];
                for r in 0..3 {
                    for c in 0..3 {
                        let d = self.grid[box_r + r][box_c + c] as usize;
                        if d == 0 || seen[d] {
                            return false;
                        }
                        seen[d] = true;
                    }
                }
            }
        }
        true
    }
}

// ── Symbolic Validator: Deterministic Rules Engine ──────────────────

/// Neuro-symbolic intercept: prunes LLM-drafted tokens against
/// deterministic constraints. Invalid moves get probability 0.0.
///
/// This is the bridge between speculative decoding (DDTree) and
/// the Percepta execution trace. The LLM proposes, the rules dispose.
pub struct SymbolicValidator;

impl SymbolicValidator {
    /// Filter drafted (digit, log_prob) pairs through Sudoku constraints.
    /// Returns only valid moves, sorted by probability descending.
    ///
    /// In a real system: the fast draft model proposes logits,
    /// this intercept prunes invalid branches *before* target verification.
    pub fn prune_drafts(
        state: &Sudoku9x9,
        row: usize,
        col: usize,
        logits: &[(u8, f32)],
    ) -> Vec<(u8, f32)> {
        let mut valid: Vec<(u8, f32)> = logits
            .iter()
            .filter(|(digit, _)| state.is_valid_move(row, col, *digit))
            .copied()
            .collect();
        valid.sort_by(|a, b| b.1.total_cmp(&a.1));
        valid
    }
}

// ── Streaming Solver: Step-by-step "thinking" output ─────────────

/// Events emitted during streaming solve.
#[derive(Debug)]
pub enum SolveEvent {
    /// Attempting to place a digit.
    Try {
        row: usize,
        col: usize,
        digit: u8,
        depth: usize,
    },
    /// Placement accepted, moving deeper.
    Accepted {
        row: usize,
        col: usize,
        digit: u8,
        filled: usize,
    },
    /// Contradiction found — this branch is dead.
    Contradiction {
        row: usize,
        col: usize,
        digit: u8,
        depth: usize,
    },
    /// Backtracking from a dead end.
    Backtrack {
        row: usize,
        col: usize,
        depth: usize,
    },
    /// Puzzle solved.
    Solved {
        steps: usize,
        hull_size: usize,
        total_trace: usize,
    },
}

/// Solver that emits events for streaming display.
/// Produces the "LLM thinking" output pattern from the Percepta demo.
pub struct StreamingSolver {
    pub state: Sudoku9x9,
    pub cache: KVCache2D,
    pub step: usize,
    pub events: Vec<SolveEvent>,
    /// CHT-based hard attention head for O(log N) queries on arbitrary 2D points.
    /// Records the same `(step, filled)` trace as `cache`.
    /// Only available with `percepta` feature flag.
    #[cfg(feature = "percepta")]
    pub cht_head: super::hull::HardAttentionHead,
}

impl StreamingSolver {
    pub fn new(grid: [[u8; 9]; 9]) -> Self {
        Self {
            state: Sudoku9x9::new(grid),
            cache: KVCache2D::new(),
            step: 0,
            events: Vec::new(),
            #[cfg(feature = "percepta")]
            cht_head: super::hull::HardAttentionHead::new(),
        }
    }

    /// Solve and collect streaming events.
    pub fn solve_streaming(&mut self) -> bool {
        let filled = self.state.clue_count();
        self.solve_recursive(0, filled)
    }

    fn solve_recursive(&mut self, depth: usize, filled: usize) -> bool {
        self.cache
            .append(Vec2::new(self.step as f32, filled as f32), self.step);

        // Mirror trace into CHT head (feature-gated).
        // Key: (step, filled_count), Value: step index, Seq: step for tie-breaking.
        #[cfg(feature = "percepta")]
        self.cht_head.insert(
            [self.step as f64, filled as f64],
            [self.step as f64, 0.0],
            self.step as i32,
        );

        self.step += 1;

        let Some((row, col)) = self.state.next_empty() else {
            self.events.push(SolveEvent::Solved {
                steps: self.step,
                hull_size: self.cache.hull_len(),
                total_trace: self.cache.len(),
            });
            return true;
        };

        for digit in 1..=9u8 {
            self.events.push(SolveEvent::Try {
                row,
                col,
                digit,
                depth,
            });

            if self.state.is_valid_move(row, col, digit) {
                self.state.grid[row][col] = digit;
                // Incremental: we just placed exactly one digit into an empty cell,
                // so the new filled count is `filled + 1`. Avoids a full 81-cell
                // scan on every accepted move in the backtracking recursion.
                let new_filled = filled + 1;
                self.events.push(SolveEvent::Accepted {
                    row,
                    col,
                    digit,
                    filled: new_filled,
                });

                if self.solve_recursive(depth + 1, new_filled) {
                    return true;
                }

                self.state.grid[row][col] = 0;
                self.events.push(SolveEvent::Backtrack { row, col, depth });
            } else {
                self.events.push(SolveEvent::Contradiction {
                    row,
                    col,
                    digit,
                    depth,
                });
            }
        }
        false
    }

    /// Format events as concise streaming "thinking" text.
    /// Matches the Percepta web demo style: ~25 lines showing
    /// early exploration, key backtracks, convergence, and solution.
    pub fn format_events(&self) -> String {
        let mut out = String::new();
        if self.events.is_empty() {
            return out;
        }

        // Collect key moments from the event stream
        let mut accepted_idx = 0usize;
        let mut accepted_events: Vec<(usize, usize, u8, usize, usize)> = Vec::new();

        for event in &self.events {
            match event {
                SolveEvent::Accepted {
                    row,
                    col,
                    digit,
                    filled,
                } => {
                    accepted_events.push((*row, *col, *digit, *filled, accepted_idx));
                    accepted_idx += 1;
                }
                SolveEvent::Backtrack { .. } => {}
                _ => {}
            }
        }

        // Phrases for varied output
        const OK_PHRASES: &[&str] = &[
            "No immediate violations.",
            "Looks consistent.",
            "Still consistent.",
            "No violations so far.",
            "That works.",
            "Looks good.",
        ];

        // Select ~20 key placements: first 4, last 5, and evenly spaced middle ones
        let n = accepted_events.len();
        let mut shown_indices: Vec<usize> = Vec::new();

        if n <= 20 {
            // Show all if few enough
            shown_indices = (0..n).collect();
        } else {
            // First 4
            shown_indices.extend(0..4usize.min(n));
            // Last 5
            let last_start = n.saturating_sub(5);
            // Middle: evenly spaced, ~11 points
            let middle_count = 11usize;
            if n > 20 {
                for i in 0..middle_count {
                    let idx = 4 + ((n - 9) as f64 * i as f64 / middle_count as f64) as usize;
                    if idx < last_start && !shown_indices.contains(&idx) {
                        shown_indices.push(idx);
                    }
                }
            }
            // Last 5
            for i in last_start..n {
                if !shown_indices.contains(&i) {
                    shown_indices.push(i);
                }
            }
            shown_indices.sort();
        }

        // Track depth changes for backtrack annotations
        let mut prev_filled = 0usize;
        let mut shown_count = 0usize;

        // Pre-size output buffer: each shown event is ~50 chars.
        out.reserve(shown_indices.len() * 64);

        // write!/writeln! into a String never returns Err.
        use std::fmt::Write as _;

        for &idx in &shown_indices {
            let (row, col, digit, filled, _seq) = accepted_events[idx];
            shown_count += 1;

            // Detect backtrack: filled count decreased from previous shown
            if filled < prev_filled && shown_count > 1 {
                let drop = prev_filled - filled;
                if drop >= 3 {
                    let _ = writeln!(
                        out,
                        "Undoing row {} col {}. Going back up.",
                        row + 1,
                        col + 1,
                    );
                } else {
                    let _ = writeln!(
                        out,
                        "Trying another path at row {}, col {}.",
                        row + 1,
                        col + 1,
                    );
                }
            }

            let _ = writeln!(out, "Trying {digit} at row {}, col {}.", row + 1, col + 1);
            let phrase = OK_PHRASES[shown_count % OK_PHRASES.len()];
            let _ = writeln!(out, "{phrase} ({filled}/81 resolved)");
            prev_filled = filled;
        }

        // Always show the Solved event
        for event in &self.events {
            if let SolveEvent::Solved {
                steps,
                hull_size,
                total_trace,
            } = event
            {
                let ratio = *total_trace as f64 / *hull_size as f64;
                let _ = writeln!(
                    out,
                    "\n✅ Solved in {steps} steps!\n\
                     Hull compression: {hull_size} vertices \
                     from {total_trace} trace entries ({ratio:.1}x)"
                );
            }
        }

        out
    }

    // ── CHT Integration Methods (feature-gated) ──────────────

    /// Total entries in the CHT hard attention head.
    #[cfg(feature = "percepta")]
    pub fn cht_size(&self) -> usize {
        self.cht_head.size()
    }

    /// Verify CHT queries match legacy cache on standard query directions.
    ///
    /// Only tests directions where legacy is expected to be correct (`qy >= 0`).
    /// Returns `(matches, total_queries)`. Perfect parity = `(N, N)`.
    ///
    /// Note: `qy < 0` queries are *not* tested here because legacy only
    /// maintains the upper hull — those queries return wrong results with
    /// `KVCache2D` but correct results with `HardAttentionHead`.
    #[cfg(feature = "percepta")]
    pub fn verify_cht_parity(&self) -> (usize, usize) {
        // Only upper-hull directions (qy > 0) and horizontal (qy == 0)
        // where legacy's Graham scan is correct.
        let queries: [[f64; 2]; 6] = [
            [1.0, 0.0],  // rightmost kx
            [0.0, 1.0],  // highest ky
            [1.0, 1.0],  // diagonal
            [5.0, 10.0], // steep positive
            [10.0, 1.0], // shallow positive
            [-1.0, 0.0], // leftmost kx (edge case)
        ];
        let mut matches = 0usize;
        for q in &queries {
            let legacy_val = {
                let query = Vec2::new(q[0] as f32, q[1] as f32);
                let (_, val) = self.cache.fast_attention(&query);
                val
            };
            let cht_val = self
                .cht_head
                .query(*q, super::types::TieBreak::Latest)
                .map(|v| v[0] as usize);

            if let Some(cv) = cht_val
                && cv == legacy_val
            {
                matches += 1;
            }
        }
        (matches, queries.len())
    }
}

#[cfg(test)]
mod tests;
