# Issue 005: Modelless Sudoku Speculate Drafter (MRV + Constraint Propagation)

## Origin

Bench 334 (`334_sudoku_speculate_perf.md`) showed that the speculate way
**cannot beat backtrack** on Arto Inkala's hardest Sudoku:

- **2.430 ms** backtrack (49,559 steps) vs speculate hitting dead-ends in
  11–31 µs then falling back to backtrack.
- Root cause: uniform marginals give the drafter zero signal.

## Optimization Opportunity (Modelless, per §3.5)

Before deferring to riir-train, exhaust modelless paths. Two deterministic
drafter improvements that could push `p_accept` from 1/9 toward 1/1:

### Option A — MRV cell ordering

Reorder `SudokuPruner::positions` by **Minimum Remaining Values**: cells with
the fewest valid digits go first. The drafter then proposes forced moves (1
valid digit) first, which always commit.

- **Effort**: small — change `SudokuPruner::new()` to sort by MRV.
- **Risk**: changes the depth→(row,col) mapping; existing examples/tests
  assume row-major order. Audit all call sites.
- **Expected gain**: Inkala has many forced cells in the opening; MRV could
  commit ~10–15 cells with `p_accept = 1` before hitting any real branching.

### Option B — Constraint propagation as drafter signal

Replace uniform marginals with a **naked/hidden singles detector**: any cell
with exactly 1 valid digit gets `p = 1.0` for that digit, others `0.0`. Cells
with >1 valid digit stay uniform. This makes the drafter a pure deterministic
rules engine — no training, no gradient descent.

- **Effort**: medium — needs a singles-finder that re-scans the board each
  round (or incrementally updates a candidate-bitmask per cell).
- **Risk**: bitmask bookkeeping bugs. Mitigate with the existing
  `has_dead_end()` helper as a sanity check.
- **Expected gain**: could solve most "easy" cells without speculation, leaving
  backtrack to handle only the truly ambiguous ones.

### Option C (deferred → riir-train)

A trained draft model that proposes the RIGHT digit. Out of scope for modelless.

## GOAT Gate (if implemented)

- [ ] G1: solve correctness — board `is_solved()` returns true.
- [ ] G2: perf — speculate_solve time < backtrack time on Inkala.
- [ ] G3: no-regression — `sudoku_02_speculative` example still works.
- [ ] G4: zero-alloc drafter (reuse scratch, no per-round Vec).
- [ ] G5: feature isolation — gate behind `sudoku_mrv` / `sudoku_cp` features.

## Architectural Blocker (separate)

`TreeNode.parent_path: u128` hard-caps lookahead at 8. To enable deeper
speculation (e.g. solve a 9-cell row in one tree), the layout needs:
- `parent_path` widened to `u256` (16 tokens) or
- Variable-length path encoding (heap-allocated, breaks the Pod story).

This is a separate, larger change — out of scope for the drafter optimization.

## Priority

Low — Sudoku is a proof-of-concept domain, not production. The bench already
answers the perf question (2.430 ms backtrack; speculate can't beat it without
a smarter drafter). Filing for completeness; no plan needed unless we want to
demo a speculate-beats-backtrack result.
