//! `EngramHotSwap` — atomic runtime table replacement.
//!
//! Plan 299 Phase 5 T5.1–T5.4, T5.7–T5.8. Mirrors the
//! `SenseHotSwap` pattern (moved to riir-engine::sense::hotswap in Issue 007 Phase C):
//! `AtomicPtr<Box<dyn EngramTable>>` + `AtomicBool` lock. The lock blocks
//! readers during swap (Option A per plan T5.4) — acceptable because table
//! updates are infrequent. Crossbeam-epoch reclamation is NOT used; the
//! writer briefly sets the lock, readers spin-wait, then the writer drops
//! the old table after the new pointer is published.
//!
//! # Memory reclamation
//!
//! The writer is responsible for dropping the old table. We use a simple
//! protocol:
//! 1. Writer: `lock.compare_exchange(false, true, AcqRel, Acquire)` — acquire
//!    exclusive swap permission.
//! 2. Writer: `ptr.swap(new_ptr, AcqRel)` — atomic pointer publication.
//! 3. Writer: `current_commitment.store(..., Release)` — update fast id.
//! 4. Writer: `lock.store(false, Release)` — release.
//! 5. Writer: `drop(Box::from_raw(old_ptr))` — free the old table.
//!
//! Readers spin on `lock.load(Acquire) == false` before loading the pointer.
//! Because the lock blocks readers BEFORE step 5, no reader can be holding
//! a borrowed reference to the old table when it's dropped. This is safe
//! under the assumption that `with_table` calls are short (microseconds) —
//! table updates are infrequent enough that this never stalls writers
//! meaningfully.
//!
//! # CRITICAL — never softmax
//!
//! Per AGENTS.md this module contains **no `softmax` symbol**. It's pure
//! atomic-pointer plumbing.
//!
//! # Latent vs raw boundary
//!
//! - The [`EngramTable`](super::EngramTable) trait object is **latent** —
//!   slot contents never cross the sync boundary.
//! - The `current_commitment: AtomicU64` is the **raw** low-8-bytes of the
//!   table's BLAKE3 root. It exists for fast identity checks across the
//!   sync boundary (clients can verify the table they observe matches a
//!   committed snapshot without learning the slot contents).

use super::EngramTable;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};

// Per-thread `with_table` depth (Issue 733). A nested same-thread call
// would spin forever against the lock its own closure holds — depth > 0
// on entry panics instead of hanging. The borrow guard clears it.
thread_local! {
    static WITH_TABLE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Releases the writer lock + the depth flag when a `with_table` closure
/// exits — normally or by panic. A leaked lock would spin every future
/// reader forever, so this MUST be panic-safe (plain Drop is).
struct ReadBorrowGuard<'a> {
    lock: &'a AtomicBool,
}

impl Drop for ReadBorrowGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.lock.store(false, Ordering::Release);
        WITH_TABLE_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

/// Atomic runtime replacement for `dyn EngramTable`.
///
/// See the module docs for the memory reclamation protocol.
pub struct EngramHotSwap {
    /// Atomic pointer to the boxed table. We use `AtomicPtr<Box<dyn EngramTable>>`
    /// (so the inner type is `Sized`): each `Box::into_raw` call produces a
    /// `*mut Box<dyn EngramTable>`, which fits in an `AtomicPtr`. The slight
    /// cost is a double indirection (`AtomicPtr → Box → dyn EngramTable`),
    /// but the box itself is a single wide pointer so the cost is one cache
    /// line load (acceptable for a control-plane operation).
    table: AtomicPtr<Box<dyn EngramTable>>,
    /// Lock flag — set to `true` during swap to block readers. Readers
    /// spin-wait on this.
    lock: AtomicBool,
    /// Low 8 bytes of the current table's BLAKE3 commitment, as u64. Used
    /// for fast identity checks without recomputing the full root.
    current_commitment: AtomicU64,
}

impl EngramHotSwap {
    /// Construct with an initial table. Computes the initial commitment.
    pub fn new(initial: Box<dyn EngramTable>) -> Self {
        let commitment = commitment_low_u64(initial.commitment());
        // Double-box: `Box::into_raw(Box::new(initial))` produces a
        // `*mut Box<dyn EngramTable>` matching the AtomicPtr type.
        Self {
            table: AtomicPtr::new(Box::into_raw(Box::new(initial))),
            lock: AtomicBool::new(false),
            current_commitment: AtomicU64::new(commitment),
        }
    }

    /// Atomically swap in a new table. Returns `Err(new_table)` if the lock
    /// is held (another writer is mid-swap). The caller may retry.
    ///
    /// On success, the old table is dropped (after the new pointer is
    /// published and the lock released — readers see either old or new,
    /// never a mix).
    pub fn swap(&self, new_table: Box<dyn EngramTable>) -> Result<(), Box<dyn EngramTable>> {
        // Acquire the writer lock. `compare_exchange` returns Err if another
        // writer beat us to it — surface that to the caller.
        if self
            .lock
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(new_table);
        }

        // We hold the lock — readers are spinning (or haven't started). Safe
        // to swap the pointer and drop the old table.
        let new_commitment = commitment_low_u64(new_table.commitment());
        let new_ptr: *mut Box<dyn EngramTable> = Box::into_raw(Box::new(new_table));
        let old_ptr = self.table.swap(new_ptr, Ordering::AcqRel);

        // Publish the new commitment. Readers that load the new pointer will
        // also see the new commitment (Release/Acquire pairing).
        self.current_commitment
            .store(new_commitment, Ordering::Release);

        // Release the writer lock — readers can proceed.
        self.lock.store(false, Ordering::Release);

        // Free the old table. No reader can be holding a borrow at this
        // point: every `with_table` closure HOLDS this lock for its whole
        // duration (Issue 733), so winning the exchange above proves no
        // closure borrow is live — the drop below cannot race one. (Pre-733
        // this argument was aspirational: the closure only spin-checked the
        // lock at entry, so a swap could drop the table under a live borrow.
        // The lock-holding closure is what makes it true.)
        // SAFETY: `old_ptr` was allocated via `Box::into_raw(Box::new(...))`
        // by us (either in `new` or a prior `swap`). We have exclusive access
        // (the writer lock guarantees no other writer is touching it; the
        // reader closure holds the same lock, so none is borrowing it).
        unsafe {
            drop(Box::from_raw(old_ptr));
        }

        Ok(())
    }

    /// Run a closure with the current table. Spin-waits if a swap is in
    /// flight. **Holds the writer lock for the closure's duration** — the
    /// guarantee that makes the borrow sound by construction (Issue 733).
    ///
    /// Returns whatever `f` returns. The closure receives a `&dyn EngramTable`
    /// borrow that is valid for the duration of `f` only — the caller MUST
    /// not escape the reference (no `unsafe` lifetime extension). The
    /// `EngramTable` trait is `Send + Sync`, so reads from other threads are
    /// safe, but the borrow itself is local to the calling thread.
    ///
    /// # Why holding the lock is the fix (Issue 733)
    ///
    /// The pre-733 shape spin-waited on the writer lock at ENTRY only, then
    /// loaded the pointer unlocked. Two unsound interleavings fell out:
    ///
    /// 1. **Same-thread nesting** — a `swap` inside the closure found the
    ///    lock free (the closure never took it), succeeded, and dropped the
    ///    old table under the closure's live borrow. Use-after-free.
    /// 2. **Cross-thread** — a `swap` on another thread during the closure
    ///    likewise found the lock free and dropped the loaded table mid-read.
    ///    The old in-code essay acknowledged this ("NOT formally safe under
    ///    all interleavings") and bet on swap latency hiding it.
    ///
    /// With the lock held for the closure's duration, `swap`'s
    /// `compare_exchange(false → true)` fails for BOTH cases and returns
    /// `Err(new_table)` — the same fail-closed contract as a contended swap.
    /// The borrow is exclusive by construction, and `swap`'s "the lock
    /// blocked all readers" drop-safety argument is now literally true.
    ///
    /// # Cost
    ///
    /// One uncontended CAS on entry + one Release store on exit per closure
    /// call, and readers are now mutually exclusive with each other. The hot
    /// lookup path does NOT go through here (`lookup_into` via the
    /// [`ZipfianCacheHierarchy`] reads a shared snapshot directly); closure
    /// readers are control-plane (dispatch/verify), where a nanosecond-scale
    /// RMW is noise. Re-validated by `g5_concurrent_reader_writer_no_torn_reads`.
    ///
    /// # Nested calls
    ///
    /// A nested `with_table` on the SAME thread would spin forever against a
    /// lock its own thread holds — instead of hanging, it panics loudly
    /// (thread-local depth flag). Nest a plain read through the borrow you
    /// already have instead.
    ///
    /// [`ZipfianCacheHierarchy`]: super::ZipfianCacheHierarchy
    pub fn with_table<R>(&self, f: impl FnOnce(&dyn EngramTable) -> R) -> R {
        // Nested same-thread calls would self-deadlock on the lock below —
        // fail loud instead of hanging. The guard clears the flag (panic-safe).
        WITH_TABLE_DEPTH.with(|d| {
            assert_eq!(
                d.get(),
                0,
                "EngramHotSwap::with_table called nested on the same thread — \n                 the closure already holds a borrow; nest a read through it instead (Issue 733)"
            );
            d.set(1);
        });

        // Acquire the writer lock for the closure's duration. `swap` and
        // `try_lock` CAS the same flag, so both fail closed while a closure
        // borrow is live — that is the soundness guarantee.
        while self
            .lock
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // Hint to the CPU that we're in a spin loop — avoids a memory
            // fence mis-speculation and reduces power on Intel HT.
            std::hint::spin_loop();
        }

        // Drop guard releases the lock + the depth flag even if `f` panics —
        // a leaked lock would spin every future reader forever.
        let _guard = ReadBorrowGuard { lock: &self.lock };

        // SAFETY: `ptr` was allocated via `Box::into_raw` by us (either in
        // `new` or a prior `swap`). No swap can be in flight while we hold
        // the borrow: every swap (this thread or another) must win the same
        // `compare_exchange(false → true)` we now hold, and it drops the old
        // pointer only AFTER its own successful exchange. Holding the lock
        // therefore excludes the drop for the borrow's whole lifetime —
        // the guarantee the pre-733 unlocked load could not make.
        //
        // `ptr` points to a `Box<dyn EngramTable>` (double-boxed on insert).
        // Dereference twice (through the double-box) to get `&dyn EngramTable`
        // directly for the caller.
        let ptr = self.table.load(Ordering::Acquire);
        let table: &dyn EngramTable = unsafe { &**ptr };
        f(table)
    }

    /// Fast identity check: low 8 bytes of the current table's BLAKE3 root.
    ///
    /// Cheap atomic load — no hashing. Useful for cross-thread "did the
    /// table change since I last looked?" checks without recomputing the
    /// full commitment.
    #[inline]
    pub fn commitment_fast(&self) -> u64 {
        self.current_commitment.load(Ordering::Acquire)
    }

    /// Acquire the writer lock manually (rare — for swap-like operations
    /// outside of [`swap`](Self::swap)). Release with [`unlock`](Self::unlock).
    ///
    /// Returns `false` if the lock was already held.
    pub fn try_lock(&self) -> bool {
        self.lock
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Release the writer lock acquired via [`try_lock`](Self::try_lock).
    pub fn unlock(&self) {
        self.lock.store(false, Ordering::Release);
    }
}

impl Drop for EngramHotSwap {
    fn drop(&mut self) {
        // We're dropping the hotswap — free the boxed table. No readers can
        // be active (we have &mut self, so exclusive access).
        let ptr = self.table.load(Ordering::Acquire);
        // SAFETY: `ptr` was allocated via `Box::into_raw` by us. We have
        // exclusive access via `&mut self`.
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

/// Extract the low 8 bytes of a BLAKE3 root as a u64 (little-endian).
///
/// This is a fast identity check — collisions across distinct tables are
/// astronomically unlikely (64-bit prefix of a 256-bit hash).
#[inline]
fn commitment_low_u64(root: [u8; 32]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&root[..8]);
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::{EngramHash, EngramTableBuilder, EngramTableId, K_MAX};

    /// Build a small test table with `n_populated` populated slots.
    fn make_table(n_slots: usize, d: usize, n_populated: usize) -> Box<dyn EngramTable> {
        let mut b = EngramTableBuilder::new(n_slots, d);
        for i in 0..n_populated as u64 {
            let pat: Vec<f32> = (0..d).map(|j| (i as f32) * 0.1 + j as f32 * 0.01).collect();
            b.add_pattern(EngramHash(i), &pat);
        }
        Box::new(b.build())
    }

    #[test]
    fn initial_commitment_is_set() {
        // T5.7: same content → same EngramTableId. The hotswap's fast
        // commitment should match the table's actual commitment low-u64.
        let table = make_table(64, 4, 4);
        let expected = commitment_low_u64(table.commitment());
        let hs = EngramHotSwap::new(table);
        assert_eq!(
            hs.commitment_fast(),
            expected,
            "initial commitment_fast must match the table's actual root low-u64"
        );
    }

    #[test]
    fn same_content_same_commitment_low_u64() {
        // Two tables with identical contents → same low-u64 commitment.
        let t1 = make_table(64, 4, 4);
        let t2 = make_table(64, 4, 4);
        let c1 = commitment_low_u64(t1.commitment());
        let c2 = commitment_low_u64(t2.commitment());
        assert_eq!(c1, c2, "same contents → same commitment low-u64");
    }

    #[test]
    fn swap_updates_commitment_fast() {
        // T5.7: after swap, commitment_fast reflects the new table.
        let t1 = make_table(64, 4, 2);
        let t2 = make_table(64, 4, 8); // different contents
        let expected_new = commitment_low_u64(t2.commitment());
        let hs = EngramHotSwap::new(t1);
        assert_ne!(
            hs.commitment_fast(),
            expected_new,
            "sanity: t1 and t2 must differ"
        );
        assert!(hs.swap(t2).is_ok(), "swap should succeed (no contention)");
        assert_eq!(
            hs.commitment_fast(),
            expected_new,
            "after swap, commitment_fast must reflect the new table"
        );
    }

    #[test]
    fn with_table_reads_current_table() {
        let t1 = make_table(64, 4, 4);
        let hs = EngramHotSwap::new(t1);
        let num_slots = hs.with_table(|t| t.num_slots());
        assert_eq!(num_slots, 64);
        let dim = hs.with_table(|t| t.dim());
        assert_eq!(dim, 4);
    }

    #[test]
    fn with_table_sees_swapped_table() {
        let t1 = make_table(64, 4, 1);
        let t2 = make_table(128, 8, 2);
        let hs = EngramHotSwap::new(t1);
        assert_eq!(hs.with_table(|t| t.num_slots()), 64);
        assert!(hs.swap(t2).is_ok(), "swap ok");
        assert_eq!(hs.with_table(|t| t.num_slots()), 128);
        assert_eq!(hs.with_table(|t| t.dim()), 8);
    }

    #[test]
    fn thousand_swaps_no_leak_smoke() {
        // T5.7: 1000 swaps in a row — smoke check for leaks. This is not a
        // real leak detector (would need valgrind/Miri for that); it just
        // verifies the swap/drop protocol runs without panic or hang for
        // 1000 iterations. Run with `cargo test -- --ignored --nocapture`
        // to see timing.
        //
        // To get a real leak check, run:
        //   cargo +nightly miri test --features engram \
        //     -p katgpt-core engram::hotswap::tests::thousand_swaps_no_leak_smoke
        // (if Miri is available; it isn't on the default toolchain).
        let t0 = make_table(64, 4, 4);
        let hs = EngramHotSwap::new(t0);
        for i in 0..1000 {
            let new_t = make_table(64, 4, (i % 8) as usize + 1);
            assert!(
                hs.swap(new_t).is_ok(),
                "swap must not fail when uncontended"
            );
        }
        // No assertion — the test is "didn't panic / leak to OOM".
    }

    #[test]
    fn engram_table_id_verify_after_swap() {
        // T5.7: after swap, EngramTableId::from_table still verifies.
        let t1 = make_table(64, 4, 4);
        let t1_id = hs_with_id(t1.as_ref());
        let t2 = make_table(64, 4, 4); // same contents as t1
        let hs = EngramHotSwap::new(t1);
        // Swap in t2 (same contents). id1_pre should still verify.
        assert!(hs.swap(t2).is_ok(), "swap");
        let id_post = hs.with_table(EngramTableId::from_table);
        assert_eq!(t1_id, id_post, "same contents → same EngramTableId");
    }

    fn hs_with_id(t: &dyn EngramTable) -> EngramTableId {
        EngramTableId::from_table(t)
    }

    #[test]
    fn try_lock_unlock_round_trip() {
        let t = make_table(32, 4, 1);
        let hs = EngramHotSwap::new(t);
        assert!(hs.try_lock(), "first try_lock succeeds");
        assert!(!hs.try_lock(), "second try_lock fails (already locked)");
        hs.unlock();
        assert!(hs.try_lock(), "try_lock succeeds after unlock");
        hs.unlock();
    }

    #[test]
    fn swap_fails_when_locked() {
        let t1 = make_table(32, 4, 1);
        let t2 = make_table(64, 4, 2);
        let hs = EngramHotSwap::new(t1);
        hs.try_lock(); // hold the lock
        let result = hs.swap(t2);
        assert!(result.is_err(), "swap must fail when locked");
        hs.unlock();
    }

    #[test]
    fn nested_swap_inside_with_table_fails_closed_and_borrow_stays_valid() {
        // Issue 733 — THE repro, now contractual. Pre-733 the nested swap
        // SUCCEEDED and dropped the old table under the closure's live
        // borrow (use-after-free). Post-733 the closure holds the writer
        // lock, so the nested swap's CAS fails → Err — and the borrow stays
        // valid: reads through it AFTER the failed nested swap still return
        // the ORIGINAL table's contents.
        let hs = EngramHotSwap::new(make_table(32, 4, 1));
        let before = hs.with_table(|t| t.commitment());
        let t2 = make_table(64, 4, 2);
        let t2_commitment = t2.commitment();
        hs.with_table(|t| {
            let result = hs.swap(t2);
            assert!(result.is_err(), "nested swap must fail closed (Issue 733)");
            // The borrow is still the ORIGINAL table — reads are valid and
            // unchanged (the failed swap returned ownership to us; the old
            // table was NOT dropped).
            assert_eq!(t.commitment(), before, "borrow must stay the original table");
            assert_eq!(t.num_slots(), 32);
        });
        // After the closure: the table is still the original (the nested
        // swap was refused, not deferred).
        assert_eq!(
            hs.commitment_fast(),
            commitment_low_u64(before),
            "refused nested swap must not change the hotswap"
        );
        assert_ne!(
            commitment_low_u64(t2_commitment),
            hs.commitment_fast(),
            "the returned table must NOT have been swapped in"
        );
    }

    #[test]
    #[should_panic(expected = "nested on the same thread")]
    fn nested_with_table_panics_loud_instead_of_self_deadlocking() {
        // Post-733 the closure holds the writer lock; a nested with_table on
        // the same thread would spin forever against it. The depth flag
        // turns the hang into a loud panic.
        let hs = EngramHotSwap::new(make_table(32, 4, 1));
        hs.with_table(|_t| {
            hs.with_table(|_inner| {});
        });
    }

    #[test]
    fn cross_thread_swap_during_live_closure_fails_closed() {
        // The OTHER half of the Issue 733 hazard: a swap from another thread
        // while a closure borrow is live. Post-733 the closure holds the
        // writer lock, so the foreign swap must get Err immediately (fail
        // closed, like any contended swap) — never drop the table mid-read.
        use std::sync::atomic::{AtomicBool, Ordering as AOrd};
        use std::sync::Arc;

        let hs = Arc::new(EngramHotSwap::new(make_table(32, 4, 1)));
        let closure_active = Arc::new(AtomicBool::new(false));
        let swap_attempted = Arc::new(AtomicBool::new(false));
        let swap_was_err = Arc::new(AtomicBool::new(false));

        let hs_w = Arc::clone(&hs);
        let active = Arc::clone(&closure_active);
        let attempted = Arc::clone(&swap_attempted);
        let attempted_inner = Arc::clone(&swap_attempted);
        let was_err = Arc::clone(&swap_was_err);
        let spawner = std::thread::spawn(move || {
            hs_w.with_table(|t| {
                let before = t.commitment();
                active.store(true, AOrd::Release);
                // Hold the closure open until the background swap attempt
                // has been made and observed.
                while !attempted_inner.load(AOrd::Acquire) {
                    std::hint::spin_loop();
                }
                // Borrow STILL valid after the foreign swap attempt — and
                // it must be the ORIGINAL table (the swap was refused).
                assert_eq!(t.commitment(), before);
                before
            })
        });

        while !closure_active.load(AOrd::Acquire) {
            std::hint::spin_loop();
        }
        // Foreign swap attempt — the closure holds the lock, so this must
        // fail closed WITHOUT dropping the borrowed table.
        let result = hs.swap(make_table(64, 4, 2));
        was_err.store(result.is_err(), AOrd::Release);
        attempted.store(true, AOrd::Release);

        let before = spawner.join().expect("closure thread must not panic");
        assert!(was_err.load(AOrd::Acquire), "foreign swap during a live closure must Err");
        assert_eq!(hs.commitment_fast(), commitment_low_u64(before));
    }

    #[test]
    #[ignore = "G5: slow concurrent reader/writer test — run with --ignored"]
    fn g5_concurrent_reader_writer_no_torn_reads() {
        // T5.8 / G5: spawn N reader threads doing lookups + 1 writer doing
        // swaps. Run for ~2 seconds wall-clock. Assert: zero panics, zero
        // torn reads (each reader's lookup returns a valid slot vector —
        // never a mix of old/new table contents).
        //
        // Note on Option A race (see with_table docs): if this test fails
        // intermittently, that's the signal to upgrade to crossbeam-epoch.
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AOrd};
        use std::thread;
        use std::time::Duration;

        const N_READERS: usize = 4;

        let t0 = make_table(128, 8, 4);
        let hs = Arc::new(EngramHotSwap::new(t0));
        let stop = Arc::new(AtomicBool::new(false));
        let panic_count = Arc::new(AtomicUsize::new(0));
        let lookup_count = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::with_capacity(N_READERS);

        // Spawn readers.
        for _ in 0..N_READERS {
            let hs = Arc::clone(&hs);
            let stop = Arc::clone(&stop);
            let panic_count = Arc::clone(&panic_count);
            let lookup_count = Arc::clone(&lookup_count);
            handles.push(thread::spawn(move || {
                let keys = [EngramHash(0); K_MAX];
                let mut out = vec![0.0f32; K_MAX * 8];
                while !stop.load(AOrd::Relaxed) {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        hs.with_table(|t| t.lookup_into(&keys, &mut out))
                    }));
                    if let Ok(hits) = result {
                        // Sanity: hits ≤ K_MAX. If hits > K_MAX, we have
                        // a torn read (corrupted vtable / data).
                        if hits > K_MAX {
                            panic_count.fetch_add(1, AOrd::Relaxed);
                        }
                        lookup_count.fetch_add(1, AOrd::Relaxed);
                    } else {
                        panic_count.fetch_add(1, AOrd::Relaxed);
                    }
                }
            }));
        }

        // Spawn writer.
        let hs_w = Arc::clone(&hs);
        let stop_w = Arc::clone(&stop);
        let writer = thread::spawn(move || {
            let mut swap_count = 0usize;
            while !stop_w.load(AOrd::Relaxed) {
                // Vary contents slightly so commitments change.
                let new_t = make_table(128, 8, (swap_count % 8) + 1);
                match hs_w.swap(new_t) {
                    Ok(()) => swap_count += 1,
                    Err(returned_t) => {
                        // Lock contention — rare, retry next iteration with
                        // a fresh table. Drop the returned one.
                        drop(returned_t);
                        std::thread::yield_now();
                    }
                }
                if swap_count >= 100 {
                    break; // writer does at most 100 swaps
                }
            }
            swap_count
        });

        // Run for ~2 seconds.
        thread::sleep(Duration::from_secs(2));
        stop.store(true, AOrd::Release);

        // Join all readers.
        for h in handles {
            h.join().expect("reader thread must not panic");
        }
        let swaps_done = writer.join().expect("writer thread must not panic");

        let panics = panic_count.load(AOrd::Acquire);
        let lookups = lookup_count.load(AOrd::Acquire);
        assert_eq!(panics, 0, "no panics / torn reads allowed; got {panics}");
        assert!(lookups > 0, "readers must have made some progress");
        assert!(swaps_done > 0, "writer must have done some swaps");
        eprintln!("g5: {swaps_done} swaps, {lookups} lookups, 0 torn reads over ~2s");
    }
}
