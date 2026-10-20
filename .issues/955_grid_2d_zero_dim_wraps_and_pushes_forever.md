**Status:** Open — latent defect found en route to riir-ai Issue 1100 T0 (2026-10-20); repro + proposed one-line guard below. No release hazard today (every shipped caller passes w,h ≥ 1), but the failure shape is a silent infinite push loop, which is the worst way to fail.

# Issue 955 — `CellComplex::grid_2d(0, h)` / `(w, 0)` wraps its loop bounds and pushes forever (release-mode OOM)

## The defect

`katgpt-dec/src/types.rs::grid_2d` computes `w - 1` and `h - 1` on `usize` for
the edge/face counts and the three loop bounds. With `w = 0` or `h = 0` the
subtraction wraps (`usize::MAX`) in release (overflow-checks off):

- `n_h_edges = (w-1) * h` → `usize::MAX * 0` = 0 (and the mirror) — the COUNTS
  stay zero, so the `reserve_exact` calls pre-allocate nothing and nothing
  looks wrong;
- the horizontal-edge loop (`for y in 0..h`, `for x in 0..(w-1)`) is skipped
  when its OUTER bound is the zero dimension;
- the **B₂ face loop** (`for y in 0..(h-1) { for x in 0..(w-1) { push } }`) is
  the trap: with `h = 0` (any `w`) the OUTER bound wraps to `usize::MAX` and
  the inner bound is `w - 1` (also `usize::MAX` when `w = 0`, or finite
  otherwise — either way the outer alone never terminates). Each iteration
  pushes 4 triplets → `grow_one` → realloc → unbounded memory growth → the
  process is OOM-killed (measured: SIGKILL in ~8 s on an M3 Max, ~1844/2123
  profiler samples in `RawVec::finish_grow` under `grid_2d`).

In a debug build the same call panics on the overflow — loud, fine. The
release shape (silent hang → OOM kill) is the dangerous one.

## Repro

```rust
let cx = katgpt_dec::CellComplex::grid_2d(0, 0); // hangs in --release, panics in debug
// equally: grid_2d(5, 0) — the B₂ outer loop wraps with a finite inner
```

Found from the consumer side: a caller reading a not-yet-allocated field
naturally has dims `(0, 0)` and passes them through; riir-ai's route-avoid
band trace hit exactly this (its first dwell preceded the first real refresh)
and died before a `dims == (0, 0)` guard was added there.

## Proposed fix (one line, plus the zeros become honest errors)

```rust
pub fn grid_2d(w: usize, h: usize) -> Self {
    assert!(w > 0 && h > 0, "grid_2d needs a non-empty grid, got {w}x{h}");
    ...
```

`grid_2d` has no meaningful empty complex (a 0-dim grid has no cells of any
rank and no operators apply), so an assert is the contract, not a limitation.
Alternative: early-return a `Self::new(0, 0, 0, 0)` — but a silent empty
complex would resurface downstream as the `graph_laplacian_into` edge-list
path on an empty boundary set, i.e. the same silence one layer later. Prefer
the assert. Add the degenerate-input arms to the grid self-tests
(`grid_2d(1,1)` should keep working — it does today: both subtractions give 0).

## Context

Discovered while wiring riir-ai Issue 1100 T0 (the route-avoid band trace);
the consumer-side guard + the negative-result record live there. This issue
owns the substrate half. Trivial fix; no GOAT implications (no perf surface);
the assert cost is one branch on a constructor.
