//! `InducedCwmSlot` — atomic hot-swap slot for an induced CWM kernel
//! (Plan 296 Phase 4).
//!
//! Paper §3.2 (arxiv 2510.04542): a Code World Model is *induced* (offline,
//! cold-tier) and then *deployed* (online, hot-tier). The deployment cycle
//! involves hot-swapping an existing kernel for a new one when the induction
//! process produces a better candidate. Crucially:
//!
//! - **Readers** (the hot path, ~20Hz) must never see a torn snapshot —
//!   either the old kernel or the new one, never a half-written mix.
//! - **Writers** (induction events, cold-tier, ~minutes apart) need to
//!   atomically replace the kernel and bump the version.
//!
//! This module ships the [`InducedCwmSlot`] that implements this contract.
//!
//! # Pattern: same as `LoRAHotSwap` / `LoRAWeightVersion` / `micro_belief`
//!
//! Per Plan 296 §T4.2, this is the SAME atomic-swap pattern used by:
//!
//! - [`riir_engine::episode_buffer::LoRAWeightVersion`] (ArcSwap-backed A/B
//!   LoRA weight swap, riir-ai Plan 092).
//! - [`crate::micro_belief::snapshot::MicroRecurrentKernelSnapshot`] (BLAKE3
//!   snapshot with `u64 version`, the precedent Phase 1's `CwmCommitment`
//!   follows).
//!
//! No new concurrency primitive is introduced. The implementation uses
//! `std::sync::Arc<std::sync::RwLock<Option<...>>>` because:
//!
//! 1. **It's in `std`** — zero new dependencies. `arc-swap` is a
//!    `riir-engine` dep but NOT a `katgpt-core` dep; adding it for one
//!    struct is scope-creep.
//! 2. **The hot path tolerates `RwLock` read-lock cost** (~10ns on x86_64
//!    uncontended) because readers clone the kernel out (one `K: clone()`
//!    per tick), not because they hold the lock for long. The read critical
//!    section is just `lock().clone()` — microseconds at most even for a
//!    KB-scale kernel.
//! 3. **Writers are rare** (minutes-scale cadence) so `RwLock` writer
//!    contention is not a concern. `ArcSwap` would shave nanoseconds per
//!    read but adds a dependency for no measurable gain at this layer.
//!
//! If a future profile shows `RwLock` read contention on the hot path,
//! swap to `arc-swap` (it's a drop-in: `RwLock<Option<T>>` →
//! `ArcSwapOption<T>`, `read().clone()` → `load().clone()`).
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! | Quantity | Space | Synced? |
//! |----------|-------|---------|
//! | `K` (the kernel object) | Raw | YES (via BLAKE3 commitment, not by value) |
//! | `CwmCommitment.blake3` | Raw | YES (audit event) |
//! | `CwmCommitment.version` | Raw | YES (monotonic counter) |
//! | `InducedCwmSlot.inner` | — | NO (slot is process-local; the *commitment* is the syncable artifact) |
//!
//! The slot itself does not cross the sync boundary. The
//! [`CwmCommitment`](crate::induced_cwm::CwmCommitment) it returns does.
//!
//! # Hot-path vs cold-path
//!
//! | Layer | What | Tier |
//! |-------|------|------|
//! | Induction event (writer) | LLM call → kernel impl + canonical bytes | Cold (offline/background) |
//! | [`InducedCwmSlot::induce`] | Store kernel + compute commitment | Cold (event) |
//! | [`InducedCwmSlot::current`] | Read-lock + clone out | **Hot** (once per search, not per tick) |
//! | Game tick on cloned kernel | `kernel.advance(state, &action, pid)` | **Hot** (20Hz) |
//!
//! Note: [`InducedCwmSlot::current`] is "hot" relative to the induction
//! cadence, but it's NOT on the 20Hz tick path. Callers should clone the
//! kernel out once at the start of a search, then call `advance()` on the
//! clone — not call `current()` on every tick. This mirrors how `mcts_search`
//! works: take `state: &S` once, then advance the clone.
//!
//! # References
//!
//! - Plan: [`crate::induced_cwm`] §Phase 4
//! - Source paper: [arxiv 2510.04542](https://arxiv.org/pdf/2510.04542) §3.2
//! - Precedent (riir-ai): `riir_engine::episode_buffer::LoRAWeightVersion`
//! - Precedent (katgpt-core): [`crate::micro_belief::snapshot`]
//! - Commitment artifact: [`crate::induced_cwm::CwmCommitment`]
//! - Kernel trait: [`crate::induced_cwm::InducedCwmKernel`]

use std::sync::{Arc, RwLock};

use crate::induced_cwm::{CwmCommitment, InducedCwmKernel};

// ── T4.1 — InducedCwmSlot ─────────────────────────────────────────────────

/// Hot-swap slot for an induced CWM kernel.
///
/// See the [module docs](self) for the full pattern and rationale. The
/// short version: readers ([`current`](Self::current)) never see a torn
/// snapshot; writers ([`induce`](Self::induce)) atomically replace the
/// kernel and bump the version.
///
/// # Type parameters
///
/// * `K` — the kernel type. Must be `InducedCwmKernel + Clone + Send + Sync`.
///   `Clone` is needed because [`current`](Self::current) clones the kernel
///   out of the lock to release it quickly. `Send + Sync` is needed because
///   the slot is shared across threads via the inner `Arc`.
///
/// # Example
///
/// ```ignore
/// use katgpt_core::induced_cwm::{InducedCwmSlot, InducedCwmKernel};
/// # use katgpt_core::traits::GameState;
/// # #[derive(Clone)]
/// # struct MyKernel;
/// # impl GameState for MyKernel { /* ... */ type Action = (); /* ... */ }
/// # impl InducedCwmKernel for MyKernel {
/// #     fn canonical_bytes(&self) -> Vec<u8> { vec![] }
/// # }
///
/// let slot = InducedCwmSlot::<MyKernel>::new();
/// assert!(slot.current().is_none());
///
/// let kernel_v1 = MyKernel;
/// let commitment_v1 = slot.induce(kernel_v1, /* version */ 1, /* tick */ 0);
/// assert_eq!(slot.current().unwrap().1.version, 1);
///
/// let kernel_v2 = MyKernel;
/// let commitment_v2 = slot.induce(kernel_v2, /* version */ 2, /* tick */ 100);
/// assert_eq!(slot.current().unwrap().1.version, 2);
/// assert_eq!(slot.current_blake3(), Some(commitment_v2.blake3));
/// ```
pub struct InducedCwmSlot<K: InducedCwmKernel + Send + Sync> {
    /// Inner storage. `None` = no kernel induced yet (slot freshly
    /// constructed, or kernel has been retired pending re-induction).
    ///
    /// `RwLock` rather than `ArcSwap` because `katgpt-core` doesn't depend
    /// on `arc-swap` (see the module docs for the rationale). The read
    /// critical section is one `clone()`, so contention is not a concern
    /// at this layer.
    inner: Arc<RwLock<Option<(K, CwmCommitment)>>>,
}

// Manual `Clone` — `#[derive(Clone)]` would require `K: Clone`, but cloning
// the slot only needs to bump the `Arc` refcount. K's `Clone` is only needed
// for `current()`, not for cloning the slot itself.
impl<K: InducedCwmKernel + Send + Sync> Clone for InducedCwmSlot<K> {
    fn clone(&self) -> Self {
        // Cloning the slot shares the underlying storage — both clones see
        // the same induced kernel. This is the "fan-out" pattern: a top-level
        // slot gets cloned into per-worker slots, all of which observe the
        // same hot-swaps.
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<K: InducedCwmKernel + Send + Sync> InducedCwmSlot<K> {
    /// Construct an empty slot (no kernel induced yet).
    pub fn new() -> Self {
        Self { inner: Arc::new(RwLock::new(None)) }
    }

    /// Construct a slot pre-loaded with `kernel` at `version` / `tick`.
    ///
    /// Convenience for tests and for the "load from disk on startup" path —
    /// equivalent to [`new`](Self::new) + [`induce`](Self::induce) but
    /// avoids the temporary empty state.
    pub fn with_kernel(kernel: K, version: u64, created_at_tick: u64) -> Self {
        let commitment = CwmCommitment::from_kernel(&kernel, version, created_at_tick);
        Self { inner: Arc::new(RwLock::new(Some((kernel, commitment)))) }
    }

    /// Hot-swap the kernel.
    ///
    /// Computes the new commitment from `kernel`'s canonical bytes, stores
    /// `(kernel, commitment)` atomically (under the write lock), and returns
    /// the commitment. Readers that called [`current`](Self::current) before
    /// this call see the old kernel; readers that call after see the new one.
    ///
    /// # Arguments
    ///
    /// * `kernel` — the new kernel to install.
    /// * `version` — caller-managed monotonic contents ordinal. SHOULD be
    ///   strictly greater than the previous version. The slot does NOT
    ///   enforce monotonicity — that's the caller's job (typically the
    ///   induction pipeline increments a counter).
    /// * `created_at_tick` — global game tick at which the induction event
    ///   occurred. Recorded in the commitment for audit / replay.
    ///
    /// # Returns
    ///
    /// The freshly-computed [`CwmCommitment`] for the new kernel. Callers
    /// should pass this to the chain-consensus layer (riir-ai Plan 326) so
    /// other nodes can verify the swap.
    pub fn induce(&self, kernel: K, version: u64, created_at_tick: u64) -> CwmCommitment {
        let commitment = CwmCommitment::from_kernel(&kernel, version, created_at_tick);
        // Write lock — blocks new readers until the swap is done. The
        // critical section is just `*guard = Some(...)` (a Vec push at
        // worst, depending on `K`'s move semantics). Microseconds.
        let mut guard = self.inner.write().expect("InducedCwmSlot lock poisoned");
        // Clone the commitment into the slot — we return the original to
        // the caller. CwmCommitment is small (32 + 8 + 8 = 48 bytes), so the
        // clone is cheaper than restructuring the API to return a borrow.
        *guard = Some((kernel, commitment.clone()));
        commitment
    }

    /// Atomically read the current kernel + commitment, cloning both out.
    ///
    /// Returns `None` if no kernel has been induced yet.
    ///
    /// # Hot-path note
    ///
    /// This call takes the read lock for the duration of one `K::clone()`.
    /// For KB-scale kernels this is sub-microsecond uncontended. Callers on
    /// the 20Hz tick path should clone once at the start of a search, not
    /// per-tick — see the module docs.
    ///
    /// # Panics
    ///
    /// Panics if the inner lock is poisoned (a writer panicked while holding
    /// the write lock). This is unrecoverable — the slot is corrupt.
    pub fn current(&self) -> Option<(K, CwmCommitment)> {
        let guard = self.inner.read().expect("InducedCwmSlot lock poisoned");
        guard.as_ref().map(|(k, c)| (k.clone(), c.clone()))
    }

    /// Cheap accessor for the current kernel's BLAKE3 commitment hash.
    ///
    /// Returns `None` if no kernel has been induced. Cheaper than
    /// [`current`](Self::current) when the caller only needs the hash for
    /// an audit check — no `K::clone()`, just a `[u8; 32]` copy.
    pub fn current_blake3(&self) -> Option<[u8; 32]> {
        let guard = self.inner.read().expect("InducedCwmSlot lock poisoned");
        guard.as_ref().map(|(_, c)| c.blake3)
    }

    /// Cheap accessor for the current commitment (no kernel clone).
    ///
    /// Returns `None` if no kernel has been induced. Useful for audit /
    /// logging paths that want the full commitment artifact but don't need
    /// the kernel itself.
    pub fn current_commitment(&self) -> Option<CwmCommitment> {
        let guard = self.inner.read().expect("InducedCwmSlot lock poisoned");
        guard.as_ref().map(|(_, c)| c.clone())
    }

    /// Cheap accessor for the current kernel's version counter.
    ///
    /// Returns `None` if no kernel has been induced.
    pub fn current_version(&self) -> Option<u64> {
        let guard = self.inner.read().expect("InducedCwmSlot lock poisoned");
        guard.as_ref().map(|(_, c)| c.version)
    }

    /// Returns `true` iff no kernel is currently induced.
    ///
    /// Cheap (one lock acquisition, no clone).
    pub fn is_empty(&self) -> bool {
        let guard = self.inner.read().expect("InducedCwmSlot lock poisoned");
        guard.is_none()
    }

    /// Get the number of strong references to the inner storage.
    ///
    /// Diagnostic / test utility. Returns 1 for a freshly-constructed slot,
    /// 2 immediately after [`Clone::clone`], etc. Useful for asserting that
    /// a slot has been uniquely owned (e.g. before `Drop`).
    pub fn arc_strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl<K: InducedCwmKernel + Send + Sync> Default for InducedCwmSlot<K> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::induced_cwm::InducedCwmKernel;
    use crate::traits::GameState;

    // ── Mock kernel (re-used from Phase 1 tests pattern) ──────────────
    //
    // Simple state type whose canonical bytes encode a `step_size`. Two
    // kernels with different `step_size` produce different BLAKE3 hashes.

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct MockKernel {
        step_size: u32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum MockAction {
        Nop,
    }

    impl GameState for MockKernel {
        type Action = MockAction;

        fn available_actions(&self, _player_id: u8) -> Vec<Self::Action> {
            vec![MockAction::Nop]
        }

        fn advance(&self, _action: &Self::Action, _player_id: u8) -> Self {
            *self // trivial — this mock is about the slot, not the game
        }

        fn is_terminal(&self) -> bool {
            false
        }

        fn reward(&self, _player_id: u8) -> f32 {
            0.0
        }

        fn tick(&self) -> u32 {
            0
        }
    }

    impl InducedCwmKernel for MockKernel {
        fn canonical_bytes(&self) -> Vec<u8> {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(b"mock_slot_v1");
            bytes.extend_from_slice(&self.step_size.to_le_bytes());
            bytes
        }
    }

    // ── T4.1 — slot lifecycle ─────────────────────────────────────────

    #[test]
    fn slot_starts_empty() {
        let slot: InducedCwmSlot<MockKernel> = InducedCwmSlot::new();
        assert!(slot.is_empty());
        assert!(slot.current().is_none());
        assert!(slot.current_blake3().is_none());
        assert!(slot.current_commitment().is_none());
        assert!(slot.current_version().is_none());
    }

    #[test]
    fn slot_default_is_empty() {
        let slot: InducedCwmSlot<MockKernel> = InducedCwmSlot::default();
        assert!(slot.is_empty());
    }

    #[test]
    fn slot_with_kernel_not_empty() {
        let slot = InducedCwmSlot::with_kernel(MockKernel { step_size: 1 }, 7, 100);
        assert!(!slot.is_empty());
        assert_eq!(slot.current_version(), Some(7));
    }

    // ── T4.3 — induce A → read → induce B → read returns B ───────────

    #[test]
    fn induce_swaps_kernel_and_bumps_version() {
        let slot: InducedCwmSlot<MockKernel> = InducedCwmSlot::new();

        // Induce kernel A (step_size=3, version=1, tick=10).
        let kernel_a = MockKernel { step_size: 3 };
        let commitment_a = slot.induce(kernel_a, 1, 10);
        assert_eq!(commitment_a.version, 1);
        assert_eq!(commitment_a.created_at_tick, 10);

        // Read returns A.
        let (current_a, commitment_a_read) = slot.current().unwrap();
        assert_eq!(current_a.step_size, 3);
        assert_eq!(commitment_a_read.version, 1);
        assert_eq!(commitment_a_read.blake3, commitment_a.blake3);

        // Induce kernel B (step_size=5, version=2, tick=20).
        let kernel_b = MockKernel { step_size: 5 };
        let commitment_b = slot.induce(kernel_b, 2, 20);
        assert_eq!(commitment_b.version, 2);

        // Read returns B.
        let (current_b, commitment_b_read) = slot.current().unwrap();
        assert_eq!(current_b.step_size, 5);
        assert_eq!(commitment_b_read.version, 2);
        assert_eq!(commitment_b_read.blake3, commitment_b.blake3);

        // BLAKE3 differs (different step_size → different canonical bytes).
        assert_ne!(
            commitment_a.blake3, commitment_b.blake3,
            "BLAKE3 must differ when step_size differs (different canonical bytes)"
        );

        // current_blake3 tracks the latest.
        assert_eq!(slot.current_blake3(), Some(commitment_b.blake3));
    }

    #[test]
    fn induce_same_kernel_keeps_blake3_stable() {
        // Two inductions of the same canonical kernel (same step_size) must
        // produce identical BLAKE3, even though `version` differs. This is
        // the G4 gate (commitment integrity) — same logical kernel → same
        // BLAKE3 across re-runs.
        let slot: InducedCwmSlot<MockKernel> = InducedCwmSlot::new();

        let kernel = MockKernel { step_size: 7 };
        let c1 = slot.induce(kernel, 1, 0);
        let c2 = slot.induce(kernel, 2, 100);

        assert_eq!(c1.blake3, c2.blake3, "same canonical kernel → same BLAKE3");
        assert_ne!(c1.version, c2.version, "version must differ");
    }

    // ── T4.4 — serde roundtrip on CwmCommitment ──────────────────────

    #[test]
    fn commitment_serde_roundtrip_preserves_fields() {
        // Build a slot, induce a kernel, serialise the commitment,
        // deserialise it, and assert all three fields are preserved.
        let slot = InducedCwmSlot::with_kernel(MockKernel { step_size: 11 }, 42, 1234);

        let original = slot.current_commitment().unwrap();

        // Serialise via serde_json (postcard / bincode would also work — the
        // CwmCommitment struct derives `serde::Serialize + Deserialize` and
        // has no platform-specific layout quirks beyond the [u8; 32] BLAKE3
        // field, which serde_json serialises as a sequence of bytes).
        let json = serde_json::to_string(&original).expect("serde_json::to_string");
        let restored: CwmCommitment =
            serde_json::from_str(&json).expect("serde_json::from_str");

        assert_eq!(original, restored, "roundtrip must preserve all fields");
        assert_eq!(restored.blake3, original.blake3);
        assert_eq!(restored.version, original.version);
        assert_eq!(restored.created_at_tick, original.created_at_tick);
    }

    #[test]
    fn commitment_serde_roundtrip_via_postcard_preserves_fields() {
        // Postcard is the production-tier serialiser for katgpt-core. Verify
        // the roundtrip via postcard too — catches any serde tag issue that
        // serde_json (with its more lenient representation) might paper over.
        let slot = InducedCwmSlot::with_kernel(MockKernel { step_size: 13 }, 99, 5678);
        let original = slot.current_commitment().unwrap();

        let bytes = postcard::to_allocvec(&original).expect("postcard::to_allocvec");
        let restored: CwmCommitment = postcard::from_bytes(&bytes).expect("postcard::from_bytes");

        assert_eq!(original, restored, "postcard roundtrip must preserve all fields");
    }

    #[test]
    fn commitment_matches_kernel_after_induce() {
        // The commitment returned by induce must match the kernel that was
        // just induced — that's the whole point of the BLAKE3 commitment.
        let slot: InducedCwmSlot<MockKernel> = InducedCwmSlot::new();
        let kernel = MockKernel { step_size: 17 };
        let commitment = slot.induce(kernel, 1, 0);

        // Read back the kernel and verify the commitment matches it.
        let (current_kernel, _) = slot.current().unwrap();
        assert!(
            commitment.matches_kernel(&current_kernel),
            "commitment must match the kernel it was computed from"
        );

        // Induce a different kernel — the OLD commitment must NOT match the
        // NEW kernel.
        let _commitment_b = slot.induce(MockKernel { step_size: 99 }, 2, 100);
        let (new_kernel, _) = slot.current().unwrap();
        assert!(
            !commitment.matches_kernel(&new_kernel),
            "old commitment must NOT match new kernel after swap"
        );
    }

    // ── Clone fan-out ─────────────────────────────────────────────────

    #[test]
    fn cloned_slot_shares_storage() {
        // Cloning the slot should produce a new slot handle that observes
        // the same swaps — both clones see the same kernel after an induce
        // on either one.
        let slot_a: InducedCwmSlot<MockKernel> = InducedCwmSlot::new();
        let slot_b = slot_a.clone();

        // arc_strong_count went 1 → 2 after clone.
        assert_eq!(slot_a.arc_strong_count(), 2);

        // Induce on slot_a, observe on slot_b.
        let c = slot_a.induce(MockKernel { step_size: 5 }, 1, 0);
        assert_eq!(slot_b.current_version(), Some(1));
        assert_eq!(slot_b.current_blake3(), Some(c.blake3));

        // Induce on slot_b, observe on slot_a.
        let c2 = slot_b.induce(MockKernel { step_size: 6 }, 2, 100);
        assert_eq!(slot_a.current_version(), Some(2));
        assert_eq!(slot_a.current_blake3(), Some(c2.blake3));
    }

    // ── Concurrency: read-during-write does not corrupt ──────────────

    #[test]
    fn concurrent_reads_during_induce_see_consistent_snapshots() {
        // Multiple readers reading concurrently with a writer must each see
        // either the old or the new kernel — never a torn mix. This test
        // does NOT verify atomicity under contention (that's a property of
        // RwLock itself, well-tested in std) — it verifies that the API
        // shape allows concurrent reads without deadlock or panic.
        use std::thread;

        let slot = Arc::new(InducedCwmSlot::with_kernel(MockKernel { step_size: 1 }, 0, 0));

        // Spawn N reader threads, each reading `current()` in a tight loop.
        // Concurrently (well, sequentially here — true contention needs a
        // longer critical section) induce a new kernel. The readers must
        // not panic and must always see a well-formed `(step_size, version)`
        // pair.
        let mut handles = Vec::new();
        for _ in 0..4 {
            let s = Arc::clone(&slot);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    if let Some((k, c)) = s.current() {
                        // The invariant: any (step_size, version) pair the
                        // reader sees must be internally consistent — the
                        // version counter is what we mutate, so a reader
                        // can see any version, but the (step_size, version)
                        // pair must match what was actually stored at some
                        // point. (We don't assert stronger than this because
                        // we can't predict ordering across threads.)
                        let _ = (k.step_size, c.version);
                    }
                }
            }));
        }

        // Writer induces several kernels.
        for i in 1..=10u64 {
            slot.induce(MockKernel { step_size: i as u32 }, i, i * 10);
        }

        for h in handles {
            h.join().expect("reader thread panicked");
        }

        // All readers finished without panic — the slot is consistent.
        assert_eq!(slot.current_version(), Some(10));
    }
}
