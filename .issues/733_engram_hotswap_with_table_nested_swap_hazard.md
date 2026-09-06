# Issue 733 — EngramHotSwap::with_table does not hold the writer lock; a nested same-thread swap drops the old table under a live borrow

**Status:** OPEN (2026-09-07) — latent soundness hazard, found by riir-chain Plan 046 while trying to force an orphan-envelope test through a "locked" hotswap (`riir-chain 438b4d44`, `.plans/046` §2b side finding).

## The hazard

`EngramHotSwap::with_table` (`crates/katgpt-core/src/engram/hotswap.rs` L140)
spin-waits only while a writer is mid-swap (`lock.load(Acquire) == true`), then
loads the pointer **without acquiring the writer lock**. Its safety comment
argues "no subsequent swap can be in flight while we hold the borrow" — but
that argument only covers OTHER threads. From the SAME thread, a nested call
is trivially reachable:

```rust
hotswap.with_table(|_t| {
    // lock == false here, so swap's compare_exchange succeeds...
    let _ = hotswap.swap(Box::new(new_table)); // ...and swap drops the OLD
                                               // table after store(false)
}); // <- the closure's &dyn EngramTable borrow now points at freed memory
```

`swap` (L84) drops the old pointer after `lock.store(false, Release)` — the
drop the nested call performs is exactly the box the outer `with_table` borrow
is reading. This is a use-after-free shaped hole gated only by the borrow
checker NOT being able to see it (the pointer is behind `AtomicPtr`, the
borrow is unlinked from it).

The rustc miri/mutable-borrow lint family cannot catch this because
`with_table(&self, ...)` never takes a mutable borrow — the aliasing is
invisible at the type level.

## Repro shape

Same-thread `with_table` → `swap` nested. (riir-chain's
`mcp_tests_engram.rs::produce_before_locked_publish_leaves_a_consistent_orphan_envelope`
originally did exactly this expecting publish to FAIL; it succeeded instead —
which is how the hazard surfaced. The test was rewritten to the honest orphan
shape, so the shipped test does NOT exercise the hazard.)

## Fix directions (pick one, prove with a test)

1. **with_table takes the writer lock** for the closure's duration
   (`compare_exchange(false→true)` + a Release store at the end). Cost: every
   reader takes a writer-lock round trip; the hot-path lookup path
   (`lookup_into` via cache hierarchy) does NOT go through `with_table`, so
   the cost lands only on closure readers — acceptable, but re-measure the
   hotswap G5 concurrent-reader gate.
2. **with_table takes &mut self** — statically serializes against same-thread
   `swap(&self)`? No: both take `&self`, so this does NOT compose; rejected
   unless `swap` also moves to `&mut self` (a real API break — audit the
   Arc<EngramHotSwap> sharing first).
3. **swap detects re-entrance** (thread-local in-swap flag) and returns
   `Err(new_table)` — smallest blast radius, keeps `&self`, makes the nested
   case fail closed like a contended swap does.

Whatever lands must add the nested-call repro as a test (debug_assertions
catches the drop-under-borrow only if the fix makes it unreachable; the test's
job is to pin WHICH behavior is contractual: `Err` from the nested swap).

## blast radius check done

`with_table` callers: katgpt-core tests + riir-chain
(`commitment_fast`/`with_table` reads) + riir-ai consumers — all READ-only
closures today; no production code nests a swap inside `with_table` (the
supervisor publishes AFTER its dispatch borrow ends). So the hazard is latent,
not live — but it is one innocent-looking helper away from live.
