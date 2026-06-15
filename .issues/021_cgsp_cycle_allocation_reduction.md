# Issue 021: CGSP `cycle()` per-cycle allocation reduction

**Date:** 2026-06-15
**Discovered by:** Plan 274 Phase 3 GOAT gate (T3.6 / P3)
**Severity:** Low (plasma-tier acceptable, blocks TRUE zero-alloc claim)
**Feature:** `cgsp`

## Problem

`CgspLoop::cycle()` allocates **~56 times per cycle** in steady state, not
zero as the plan T1.4 originally claimed. The plan's "zero-allocation in
steady state" invariant is empirically false.

Measured in `.benchmarks/274_cgsp_goat.md` §P3:
- 55.91 allocs/cycle
- 3480 bytes/cycle
- k=8 candidates per cycle, pool_size=64

The per-cycle cost is still 844ns (under the 1µs plasma budget), so this is
not blocking. But the plan claimed zero-alloc and we should either honour
that claim or update the plan honestly.

## Root cause

Two allocation sites in `src/cgsp/loop_.rs::cycle()`:

### Site 1: `scratch.candidates.resize(k, placeholder)` (line ~215)

```rust
scratch.candidates.resize(k, Candidate::new(Direction::zeros(target.dim()), usize::MAX));
```

`ScratchBuffers::reset()` calls `.clear()` on each Vec (preserving capacity),
then `cycle()` calls `.resize(k, placeholder)`. Going from len=0 to len=k
clones the `placeholder` Candidate k times. Each clone allocates a new
`Vec<f32>` for `Direction::coords`.

**Fix:** Instead of clear+resize, use a fill pattern that reuses existing
slots. Pre-fill the Vec once in `ScratchBuffers::new()` to length k, then
overwrite in-place each cycle (no clear, no resize). Or use
`mem::swap`/`mem::take` with a pool of pre-allocated Direction buffers.

### Site 2: `candidates[i].clone()` for solver attempt (line ~273)

```rust
let cand = candidates[i].clone();
let rate = self.solver.attempt(target, &cand);
```

This clone exists to work around a borrow-checker conflict: `self.solver`
needs `&mut self` while `candidates` (which is a slice of `scratch`) is
already borrowed mutably. Cloning the Candidate detaches it from the scratch
borrow so the solver can be called.

**Fix:** Refactor the borrow split. Options:
- Extract the `&mut self.solver` borrow BEFORE entering the
  candidates-loop, storing it in a local. (May require restructuring the
  `ScratchBuffers` destructuring.)
- Change the `Solver::attempt` trait signature to take `&Direction` instead
  of `&Candidate`, and pass `&candidates[i].direction` directly without
  cloning.
- Use `std::mem::take` to move the Candidate out of the slot temporarily,
  then put it back after the solver call (avoids allocation but adds a
  write-back).

## Proposed approach

Option A (smallest change, biggest win): change `Solver::attempt` to take
`&Direction` + `pool_index: usize` instead of `&Candidate`. This removes
Site 2 entirely and makes the trait cleaner.

```rust
pub trait Solver {
    fn attempt(&mut self, target: &Target, candidate_direction: &Direction, pool_index: usize) -> f32;
}
```

Option B (fixes Site 1): make `ScratchBuffers::reset` a no-op when the Vec
is already at length k. Add a `ScratchBuffers::ensure_len(k)` that fills
slots once. The conjecturer writes into slots in-place rather than relying
on resize semantics.

Combining A + B would bring per-cycle allocations down to ~0 (only the
`cdf_scratch` growth on the very first cycle remains, which is amortised).

## Acceptance criteria

- [ ] Per-cycle allocations < 5 (down from 56) in the P3 gate test
- [ ] G4 per-cycle overhead does not regress (still ≤ 1µs)
- [ ] All 29 existing cgsp unit tests still pass
- [ ] All 9 GOAT gate tests still pass

## References

- Plan: `.plans/274_curiosity_guided_self_play.md` (T3.6)
- Benchmark: `.benchmarks/274_cgsp_goat.md` §P3
- Implementation: `src/cgsp/loop_.rs` lines ~215, ~273
- Test: `tests/bench_274_cgsp_goat.rs::p3_allocation_audit_steady_state`

## TL;DR

CGSP's `cycle()` allocates ~56 times per cycle, not zero. Two sites cause
this: (1) clear+resize pattern on `scratch.candidates`, (2) Candidate clone
to dodge a borrow-checker conflict. Both are fixable. Not blocking (844ns
includes them), but blocks the "zero-alloc" plan claim. Low priority unless
CGSP sees heavy per-tick use in riir-ai Plan 299.
