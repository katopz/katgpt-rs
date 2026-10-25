# Issue 964 — katgpt-assign: precompute the solver's static invariants (bbtree-class ordered-structure audit)

**Status:** IN PROGRESS — findings filed by the bbtree distill session (workspace distill Research 322); every fix behind its own GOAT gate.
**Claimed:** katgpt-rs-m3-964a, epoch=1792932000, expires 2026-10-27T12:40:00Z
**Date:** 2026-10-25
**Type:** perf / refactor (zero-risk class: O(m log m) per call on STATIC data)

## Finding (static analysis, not measurements)

The `katgpt-assign` solver recomputes work that is invariant across recomputes —
it depends only on the compiled problem spec, never on the assignment state:

1. **Sorted/deduped member list rebuilt on every heat recompute** —
   `crates/katgpt-assign/src/solver.rs:610-613` (inside `refresh_heat`'s
   `Spec::GroupCount` arm): `members.clone()` + `sort_unstable()` + `dedup()`
   run on EVERY heat recompute. The sorted/deduped member list is invariant —
   it is a pure function of the compiled spec. The same clone+sort+dedup is
   repeated at `crates/katgpt-assign/src/solver.rs:685-688` (`heat_of_container`,
   the read-only `explain` surface). Current cost: an allocation + an
   `O(m log m)` sort per spec per recompute (and per explained container).

2. **`heat_order` re-sorted from scratch per recompute** —
   `crates/katgpt-assign/src/solver.rs:649-652`: `clear()` + `extend(0..b)` +
   `sort_unstable_by_key` over `b` containers on every `refresh_heat()`. The
   `0..b` baseline and the seeded `tie_perm` are static; only the `heat` key
   moves between recomputes.

This is the audit's "O(m log m) per call on static data" class — the classic
cache-conscious ordered-structure finding (bbtree). Zero-risk fix: the
precomputed values are byte-identical to what the hot path recomputes today.

## Proposed tasks

- [ ] **(a)** Precompute the sorted/deduped member list once at
      `specs::compile()` (`crates/katgpt-assign/src/specs.rs:85`) — the
      builder-pattern/pre-compute rule the repo already follows. `Compiled`
      (`crates/katgpt-assign/src/specs.rs:30`) gains the per-`GroupCount`-spec
      sorted+deduped member lists; the solver holds a reference. Both call
      sites (`refresh_heat` + `heat_of_container`) consume it — the per-call
      clone + sort + dedup disappear from the hot path.
- [ ] **(b)** `heat_order`: precompute the container-index order (the static
      `0..b` + `tie_perm` baseline) at compile; the per-recompute sort over the
      moving `heat` key stays. If profiling warrants (measure first — only if
      the sort shows up), swap it for a bucket/radix pass over the `i64` heat
      values instead of a comparison sort.
- [ ] **(c)** GOAT gate before any promotion, per repo Feature Flag Discipline:
      a solver recompute-loop microbench before/after — **G1** identical
      assignment output (the solver has parity gates; reuse them);
      **G2** `--release`, quiet box, box state (free RAM / power mode /
      concurrent jobs) recorded beside the numbers; **G3** no regression on
      the existing suite; **G4** the `members.clone()` allocation disappears
      from the hot path (alloc-count axis).

## References

- Blog: <https://bbtree.xiangpeng.systems/post/>
- Source: <https://github.com/XiangpengHao/bbtree> @ `8adae97a0ed97e5c98fa2790312eb0aec9719846`
  (crate `bbtree` v0.1.5, MIT OR Apache-2.0, zero-dep, no_std, stable Rust —
  cache-conscious ordered map: precompute/sort-once where the data is static).
