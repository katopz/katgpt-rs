# katgpt-rs: Sudoku Domain — Solver, Constraint Pruning, TUI

## Overview
Sudoku serves as the proof-of-concept domain for Deterministic Validator — demonstrating that a deterministic rules engine can validate LLM-drafted tokens inside the speculative decoding loop. The 9×9 solver with streaming output matches the web-demo experience from the Percepta blog.

## Sudoku9x9 (`percepta/legacy.rs`)

```rust
pub struct Sudoku9x9 {
    pub grid: [[u8; 9]; 9],
}
```

### Key Methods
- `new(grid: [[u8; 9]; 9]) -> Self` — create from grid (0 = empty)
- `arto_inkala() -> Self` — Arto Inkala's "World's Hardest Sudoku" (21 clues)
- `percepta_reference() -> Self` — puzzle from Percepta transformer-vm manifest (30 clues)
- `is_valid_move(row, col, digit) -> bool` — row/col/3×3 box checks
- `clue_count() -> usize` — count non-zero cells
- `solve(&mut self, cache: &mut KVCache2D, step: &mut usize) -> bool` — backtracking solver
- `is_solved() -> bool`
- `next_empty() -> Option<(usize, usize)>` — find next empty cell
- `display() -> String` — pretty-printed grid with box separators

### Reference Puzzle (Arto Inkala, 21 clues)
```
8 . . | . . . | . . .
. . 3 | 6 . . | . . .
. 7 . | . 9 . | 2 . .
------+-------+------
. 5 . | . . 7 | . . .
. . . | . 4 5 | 7 . .
. . . | 1 . . | . 3 .
------+-------+------
. . 1 | . . . | . 6 8
. . 8 | 5 . . | . 1 .
. 9 . | . . . | 4 . .
```

### Results
- 49,559 backtracking steps to solve
- 7 hull vertices → 7079.9× compression via convex hull
- O(49559) → O(log 7) ≈ O(3) attention speedup

## StreamingSolver (`percepta/legacy.rs`)

Emits `SolveEvent`s during solve for real-time visualization:

```rust
pub enum SolveEvent {
    Try { row: usize, col: usize, digit: u8, depth: usize },
    Accepted { row: usize, col: usize, digit: u8, filled: usize },
    Contradiction { row: usize, col: usize, digit: u8, depth: usize },
    Backtrack { row: usize, col: usize, depth: usize },
    Solved { steps: usize, hull_size: usize, total_trace: usize },
}

pub struct StreamingSolver {
    pub state: Sudoku9x9,
    pub cache: KVCache2D,
    pub step: usize,
    pub events: Vec<SolveEvent>,
    #[cfg(feature = "percepta")]
    pub cht_head: super::hull::HardAttentionHead,
}
```

- `format_events()` produces concise web-demo-style output
- Shows first 4 placements, evenly spaced middle, last 5
- `cht_head` mirrors the `(step, filled)` trace for O(log N) hard attention queries (gated behind `"percepta"` feature)

## SudokuPruner (`pruners/sudoku_pruner.rs`, behind `"sudoku"` feature)

Implements `ConstraintPruner` (from `katgpt-core::traits`) for DDTree branch validation:

### Path-Aware Pruning
- Signature: `is_valid(depth, token_idx, parent_tokens: &[usize])`
- Checks digit against initial board AND all parent tokens in the path
- Catches cross-depth conflicts: depth 0 places `4` at (0,1), depth 1 tries `4` at (0,2) → pruned
- Result: **100% valid branches** (all cross-depth conflicts eliminated)

```rust
#[cfg(feature = "sudoku")]
pub struct SudokuPruner {
    board: Sudoku9x9,
    positions: Vec<(usize, usize)>,  // depth → (row, col) mapping
}
```

### Key Methods
- `new(board: Sudoku9x9) -> Self` — auto-discovers empty cells in row-major order
- `empty_count() -> usize` — number of empty cells (= max DDTree depth)
- `position_at(depth) -> Option<(usize, usize)>` — depth → (row, col) mapping
- `board() -> &Sudoku9x9` — access underlying board state

### ConstraintPruner Implementation
Path-aware conflict checking is inline in `is_valid()`:
1. Reject token 0 (empty/padding)
2. Map depth → (row, col) via `positions`
3. Check digit against initial board (`board.is_valid_move`)
4. Iterate parent tokens: if same digit shares row/col/box with any parent → reject
- Incremental: O(parent_tokens.len()) per check = O(lookahead) — negligible

## DDTree Integration

`build_dd_tree_pruned(marginals, config, pruner)` in `dd_tree.rs`:
- Before adding children to heap, filters through `pruner.is_valid(depth, token, parent_tokens)`
- Invalid branches never enter the tree → saves verification budget
- Works with both chain-seed and REST merge

### Pruning Results
| Metric | No Pruner | Static Pruner | Path-Aware Pruner |
|--------|-----------|---------------|-------------------|
| Valid branches | 52% | ~80% | **100%** |
| Invalid pruned | 0 | ~20% | **48%** |
| Cross-depth conflicts | Not caught | Not caught | **All caught** |

## Examples

### `sudoku_01_9x9.rs`
- Solves Arto Inkala puzzle with streaming "thinking" output
- Shows hull compression stats and O(log N) attention
- No feature flag required (Sudoku9x9 is always public)
- Run: `cargo run --example sudoku_01_9x9`

### `sudoku_02_speculative.rs` (behind `"sudoku"` feature)
- Simulated draft model marginals (uniform over valid digits)
- Compares DDTree: without vs with Deterministic Validator pruning
- Token distribution table shows which digits pruned per depth
- Run: `cargo run --features sudoku --example sudoku_02_speculative`

### `sudoku_03_tui.rs` (behind `"sudoku"` feature)
- Ratatui-based TUI with real-time solver visualization
- Color-coded cells: Green (clue), Cyan (accepted), Yellow (trying), Red (contradiction)
- Two tabs: 9×9 solver vs Speculative mode
- Channel-based: `mpsc` for streaming events from solver thread
- Stats bar: tok/s, tokens, lines/s
- Run: `cargo run --features sudoku --release --example sudoku_03_tui`

### `sudoku_04_percepta_vs.rs` (behind `"sudoku"` feature)
- Compares native vs Percepta execution approaches

## Design Lessons
1. **ConstraintPruner is domain-agnostic** — same trait serves Sudoku and Rust AST (SynPruner)
2. **Path context is essential** — static pruning misses cross-depth conflicts
3. **Incremental validation is fast** — O(lookahead) per check, no board copies needed
4. **Deterministic Validator promise delivered** — LLM drafts, deterministic rules validate, 100% valid outputs
