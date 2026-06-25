//! Fixed-capacity ring buffer of pending delegate tokens (Plan 303 T3.2).
//!
//! Zero-allocation handoff between the per-tick salience decision and the
//! caller-owned async spawn. This crate does **not** spawn async tasks — see
//! the contract note in [`PendingDelegateQueue`] docs (Plan 303 T3.3).
//!
//! # Example: caller pattern (Plan 303 T3.4)
//!
//! ```no_run
//! use katgpt_rs::salience::{
//!     FoldbackTarget, PendingDelegateQueue, SalienceDecision, SalienceTriGate,
//! };
//!
//! // 1. Construct the gate once (caller owns the direction vectors).
//! const D: usize = 8;
//! let gate: SalienceTriGate<&'static str, D> = SalienceTriGate::new(
//!     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // d_speak
//!     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // d_delegate
//!     0.3, 0.2, // w_z, w_c
//!     2.0, 2.0, // beta_speak, beta_delegate
//!     0.5, 0.5, // tau_speak, tau_delegate
//!     0.2, 0.6, // floor_speak, ceil_delegate
//! );
//!
//! // 2. Caller-owned queue (this crate does NOT spawn async — T3.3).
//! let mut pending: PendingDelegateQueue<&'static str> = PendingDelegateQueue::new();
//!
//! // 3. Per-tick decision. If the gate emits Delegate, push the token.
//! let tick = 42u64;
//! let a = [0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//! let decision = gate.decide(&a, 0.5, 0.5, "delegate-payload", tick);
//! if let SalienceDecision::Delegate(payload) = decision {
//!     let token = gate
//!         .build_delegate_token(payload, tick, 0, FoldbackTarget::ActivationState);
//!     // Push is fallible — caller decides drop-oldest vs refuse-the-decision.
//!     let _ = pending.push(token);
//! }
//!
//! // 4. Caller's runtime spawns the async task and pops from `pending`.
//! //    On completion the caller applies foldback per `token.foldback_target`.
//! while let Some(token) = pending.pop() {
//!     // spawn_async(async move { /* call backend, fold result back */ });
//!     # let _ = token;
//! }
//! ```
//!
//! Reference: Plan 303 T3.2 / T3.3 / T3.4.

#![cfg(feature = "salience_tri_gate")]

use crate::salience::types::DelegateToken;

/// Fixed-capacity ring buffer of pending [`DelegateToken`]s.
///
/// **Contract (Plan 303 T3.3):** this crate does **not** spawn async tasks.
/// The caller (riir-ai runtime, Plan 330) owns the spawn. This queue is just
/// the typed handoff — the caller pushes tokens here when the gate emits
/// `Delegate`, pops them when spawning async work, and removes them on
/// completion to apply foldback.
///
/// Default capacity is 2 (the common case: one in-flight delegate + one
/// queued). Callers that expect more concurrency should raise `CAP`.
///
/// **Zero-allocation:** `slots` is a fixed-size array of
/// `Option<DelegateToken<A>>`. `push` returns `Err(token)` when full — the
/// caller decides policy (drop oldest, drop newest, refuse the decision, …).
///
/// # Ring-buffer layout
///
/// - `head` = next write slot (advances mod `CAP` on every successful push).
/// - `len`  = number of live tokens (`0..=CAP`).
/// - Oldest live slot (next to be popped, FIFO) = `(head + CAP - len) % CAP`.
///
/// `head` and `len` are `u8`; this constrains `CAP <= 255` (asserted in
/// [`new`](Self::new)). The default of 2 covers the typical "one in-flight +
/// one queued" case.
///
/// Reference: Plan 303 T3.2.
#[derive(Clone, Debug)]
pub struct PendingDelegateQueue<A: Clone, const CAP: usize = 2> {
    slots: [Option<DelegateToken<A>>; CAP],
    head: u8,
    len: u8,
}

impl<A: Clone, const CAP: usize> PendingDelegateQueue<A, CAP> {
    /// Construct an empty queue.
    ///
    /// # Panics
    /// Debug-build only: panics if `CAP > 255` (the `head` / `len` fields are
    /// `u8`). Production callers should use `CAP <= 255`; the default `CAP = 2`
    /// is always safe.
    #[must_use]
    pub fn new() -> Self {
        debug_assert!(
            CAP <= u8::MAX as usize,
            "PendingDelegateQueue: CAP must be <= 255 (head/len are u8), got {CAP}"
        );
        Self {
            // Inline const block (stable since Rust 1.79; this crate's edition
            // is 2024 so MSRV ≥ 1.85 ⇒ available). Each element is
            // independently const-evaluated, so non-Copy `DelegateToken<A>`
            // works without a `Copy` bound on `A`.
            slots: [const { None }; CAP],
            head: 0,
            len: 0,
        }
    }

    /// Push a token onto the queue (append at the tail).
    ///
    /// Returns `Err(token)` if the queue is full — the caller decides policy
    /// (drop oldest, drop newest, refuse the decision, …). We never silently
    /// drop; the failed-push token is returned by value so the caller can
    /// retry or log.
    #[inline]
    pub fn push(&mut self, token: DelegateToken<A>) -> Result<(), DelegateToken<A>> {
        if self.len as usize >= CAP {
            return Err(token);
        }
        let h = self.head as usize;
        self.slots[h] = Some(token);
        // Advance head mod CAP. The branch-free form avoids a predictable
        // branch in the hot push path; CAP is a const generic so the divisor
        // is known at compile time.
        self.head = ((h + 1) % CAP) as u8;
        self.len += 1;
        Ok(())
    }

    /// Pop the oldest token (FIFO). Returns `None` if empty.
    ///
    /// The returned slot is set back to `None` so the `DelegateToken<A>`
    /// payload drops here (correct Drop semantics if `A` owns resources).
    #[inline]
    pub fn pop(&mut self) -> Option<DelegateToken<A>> {
        if self.len == 0 {
            return None;
        }
        // Oldest live slot = (head + CAP - len) mod CAP.
        let oldest = (self.head as usize + CAP - self.len as usize) % CAP;
        self.len -= 1;
        self.slots[oldest].take()
    }

    /// `true` iff the queue holds zero tokens.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Current number of live tokens.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Fixed capacity (the `CAP` const generic).
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        CAP
    }

    /// Drop all pending tokens and reset `head` / `len`.
    ///
    /// Each slot is set back to `None` so payloads' `Drop` runs (matters when
    /// `A` owns resources — e.g. an `Arc<Request>`).
    #[inline]
    pub fn clear(&mut self) {
        for slot in self.slots.iter_mut() {
            *slot = None;
        }
        self.head = 0;
        self.len = 0;
    }
}

impl<A: Clone, const CAP: usize> Default for PendingDelegateQueue<A, CAP> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::salience::types::FoldbackTarget;

    /// Build a token with a small payload for tests.
    fn tok(payload: u32, tick: u64) -> DelegateToken<u32> {
        DelegateToken {
            payload,
            issued_tick: tick,
            holding_reply_idx: 0,
            foldback_target: FoldbackTarget::ActivationState,
        }
    }

    #[test]
    fn push_pop_fifo_order() {
        let mut q: PendingDelegateQueue<u32, 2> = PendingDelegateQueue::new();
        assert!(q.push(tok(10, 1)).is_ok());
        assert!(q.push(tok(20, 2)).is_ok());

        // FIFO: 10 must come out before 20.
        let a = q.pop().expect("first pop");
        assert_eq!(a.payload, 10);
        assert_eq!(a.issued_tick, 1);
        let b = q.pop().expect("second pop");
        assert_eq!(b.payload, 20);
        assert_eq!(b.issued_tick, 2);
        assert!(q.is_empty());
    }

    #[test]
    fn push_when_full_returns_err_with_token() {
        let mut q: PendingDelegateQueue<u32, 2> = PendingDelegateQueue::new();
        assert!(q.push(tok(1, 0)).is_ok());
        assert!(q.push(tok(2, 0)).is_ok());
        assert_eq!(q.len(), 2);

        // Third push must fail and hand the token back by value.
        match q.push(tok(99, 0)) {
            Err(returned) => assert_eq!(returned.payload, 99),
            Ok(()) => panic!("push should have failed when full"),
        }
        // Length unchanged after a failed push.
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn cap_2_holds_exactly_2() {
        let mut q: PendingDelegateQueue<u32, 2> = PendingDelegateQueue::new();
        assert_eq!(q.capacity(), 2);
        assert_eq!(q.len(), 0);
        assert!(q.push(tok(1, 0)).is_ok());
        assert_eq!(q.len(), 1);
        assert!(q.push(tok(2, 0)).is_ok());
        assert_eq!(q.len(), 2);
        assert_eq!(q.capacity(), 2);
    }

    #[test]
    fn pop_on_empty_returns_none() {
        let mut q: PendingDelegateQueue<u32, 2> = PendingDelegateQueue::new();
        assert!(q.pop().is_none());
        // Still empty after a no-op pop.
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn clear_resets_all_slots() {
        let mut q: PendingDelegateQueue<u32, 2> = PendingDelegateQueue::new();
        assert!(q.push(tok(1, 0)).is_ok());
        assert!(q.push(tok(2, 0)).is_ok());
        assert_eq!(q.len(), 2);

        q.clear();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert!(q.pop().is_none());
        // Capacity is structural, not state — unchanged by clear.
        assert_eq!(q.capacity(), 2);
    }

    #[test]
    fn reuse_after_clear() {
        // After clear the queue must be usable again — push and pop should
        // behave as if freshly constructed. (Catches a regression where clear
        // forgets to reset head/len and the ring indexing goes off-rail.)
        let mut q: PendingDelegateQueue<u32, 2> = PendingDelegateQueue::new();
        assert!(q.push(tok(1, 0)).is_ok());
        assert!(q.push(tok(2, 0)).is_ok());
        q.clear();

        assert!(q.push(tok(42, 7)).is_ok());
        assert_eq!(q.len(), 1);
        let popped = q.pop().expect("pop after clear+push");
        assert_eq!(popped.payload, 42);
        assert_eq!(popped.issued_tick, 7);
        assert!(q.is_empty());
    }

    #[test]
    fn wraparound_preserves_fifo() {
        // Stress the ring wraparound: push 2, pop 1, push 1 (forces head to
        // wrap), then pop both. Oldest-first order must hold.
        let mut q: PendingDelegateQueue<u32, 2> = PendingDelegateQueue::new();
        assert!(q.push(tok(1, 0)).is_ok());
        assert!(q.push(tok(2, 0)).is_ok());
        assert_eq!(q.pop().unwrap().payload, 1); // pop oldest, head still at 0
        assert!(q.push(tok(3, 0)).is_ok()); // writes to slots[0], wraps
        assert_eq!(q.pop().unwrap().payload, 2); // 2 was the older of the two
        assert_eq!(q.pop().unwrap().payload, 3);
        assert!(q.is_empty());
    }

    #[test]
    fn default_is_empty() {
        let q: PendingDelegateQueue<u32, 2> = PendingDelegateQueue::default();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert_eq!(q.capacity(), 2);
    }

    #[test]
    fn larger_cap_holds_more() {
        // Sanity-check the const-generic CAP works beyond the default 2.
        let mut q: PendingDelegateQueue<u32, 4> = PendingDelegateQueue::new();
        assert_eq!(q.capacity(), 4);
        for i in 0..4u32 {
            assert!(q.push(tok(i, 0)).is_ok(), "push {i} failed");
        }
        // 5th must fail.
        assert!(q.push(tok(99, 0)).is_err());
        // FIFO drain.
        for i in 0..4u32 {
            assert_eq!(q.pop().unwrap().payload, i, "FIFO broke at {i}");
        }
        assert!(q.is_empty());
    }
}
