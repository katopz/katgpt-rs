//! In-memory frozen engram pattern table.
//!
//! Plan 299 Phase 2 T2.1–T2.6. A flat `Box<[f32]>` row-major array of
//! `N × D` slot vectors, looked up by `hash mod N` (direct index, O(1)).
//!
//! # Hot-path contract
//!
//! [`InMemoryEngramTable::lookup_into`] is **zero-allocation**: caller
//! provides an `out` slice of size `K_MAX * D` and the implementation does
//! K_MAX direct slice-copies from the flat slots array. No `Vec`, no
//! `HashMap`, no papaya — direct array indexing is faster than any hash map
//! for fixed-modulus lookup (per AGENTS.md "Don't linear scan for hot-path
//! queries" — we don't even scan, we O(1) index).
//!
//! # Collision handling
//!
//! Last-write-wins at build time. Multi-head retrieval in Phase 2 (K_MAX
//! heads, distinct primes) dilutes collisions to a quality issue; the
//! sigmoid gate in Phase 3 filters any residual noise.

use super::{EngramHash, EngramTable, HashHead, K_MAX};
use std::sync::OnceLock;

/// Frozen in-memory engram pattern table.
///
/// Construct via [`EngramTableBuilder`]. After `build()`, the slots and
/// heads are immutable; the only lazy state is the cached BLAKE3
/// commitment (computed on first call to [`commitment`](EngramTable::commitment)).
pub struct InMemoryEngramTable {
    /// Flat `N × D` row-major slot array. Slot `i` occupies
    /// `slots[i*D..(i+1)*D]`. Empty / unpopulated slots are all-zeros.
    slots: Box<[f32]>,
    /// K_MAX hash heads used by the *caller* to produce the hash keys
    /// passed to `lookup_into`. Stored here for diagnostics + commitment
    /// (the heads configuration is part of the table identity).
    heads: Box<[HashHead; K_MAX]>,
    /// Number of slots N (slots.len() / D).
    n_slots: usize,
    /// Slot dimensionality D.
    d: usize,
    /// Lazy-cached BLAKE3 commitment. First call to `commitment()` computes
    /// and stores; subsequent calls return the cached value.
    commitment_cache: OnceLock<[u8; 32]>,
}

impl InMemoryEngramTable {
    /// Convenience constructor — equivalent to
    /// `EngramTableBuilder::new(n_slots, d).build()` with default heads.
    #[inline]
    pub fn builder() -> EngramTableBuilder {
        EngramTableBuilder::new(1024, 32)
    }

    /// Read-only access to the heads configuration (for tests + diagnostics).
    #[inline]
    pub fn heads(&self) -> &[HashHead; K_MAX] {
        &self.heads
    }
}

impl EngramTable for InMemoryEngramTable {
    #[inline]
    fn lookup_into(&self, hash_keys: &[EngramHash; K_MAX], out: &mut [f32]) -> usize {
        let d = self.d;
        debug_assert!(
            out.len() >= K_MAX * d,
            "lookup_into: out.len()={} must be ≥ K_MAX*D = {}*{} = {}",
            out.len(),
            K_MAX,
            d,
            K_MAX * d
        );

        let n = self.n_slots;
        // Guard against n == 0 to avoid division-by-zero. Treat empty tables
        // as all-zero outputs.
        if n == 0 {
            for v in out[..K_MAX * d].iter_mut() {
                *v = 0.0;
            }
            return 0;
        }

        let mut hits = 0usize;
        // K_MAX is const (16); the inner slice copy is what dominates.
        // `copy_from_slice` is a memcpy — no per-element overhead.
        for k in 0..K_MAX {
            let slot_idx = (hash_keys[k].0 as usize) % n;
            let src = &self.slots[slot_idx * d..(slot_idx + 1) * d];
            let dst = &mut out[k * d..(k + 1) * d];
            dst.copy_from_slice(src);
            // Hit = any non-zero element in the slot. SIMD-friendly check
            // would be `simd_sum_abs_f32(src) > 0.0` but for K_MAX=16 the
            // scalar scan is cheap and branch-predictor-friendly.
            if src.iter().any(|&v| v != 0.0) {
                hits += 1;
            }
        }
        hits
    }

    #[inline]
    fn commitment(&self) -> [u8; 32] {
        *self.commitment_cache.get_or_init(|| {
            // Compute the Merkle root over the flat slots array. The heads
            // configuration is NOT part of the commitment — two tables with
            // identical slot contents but different heads (e.g. re-hashed
            // during a hot-swap) share the same commitment. This matches
            // the plan T5.6 contract (slots-only leaves).
            super::commitment::build_merkle_root(&self.slots, self.d)
        })
    }

    #[inline]
    fn num_slots(&self) -> usize {
        self.n_slots
    }

    #[inline]
    fn dim(&self) -> usize {
        self.d
    }
}

// EngramTable requires Send + Sync. InMemoryEngramTable is Send+Sync if its
// fields are: Box<[f32]> (Send+Sync), Box<[HashHead; K_MAX]> (Send+Sync),
// OnceLock<[u8;32]> (Send+Sync). All good — no manual unsafe needed.

/// Builder for [`InMemoryEngramTable`].
///
/// Accumulates `(hash_key, pattern)` writes into a flat slot array using
/// last-write-wins collision handling, then freezes the table on `build()`.
///
/// # Example
///
/// ```ignore
/// use katgpt_core::engram::{EngramTableBuilder, EngramHash};
/// let mut b = EngramTableBuilder::new(1024, 32);
/// b.add_pattern(EngramHash(7), &[1.0f32; 32]);
/// let table = b.build();
/// ```
pub struct EngramTableBuilder {
    slots: Box<[f32]>,
    heads: Box<[HashHead; K_MAX]>,
    n_slots: usize,
    d: usize,
}

impl EngramTableBuilder {
    /// Create an empty builder for an `n_slots × d` table.
    ///
    /// Picks a default head configuration: each head gets a distinct prime
    /// modulus ≥ `n_slots` (next prime ≥ n_slots, picked from a small table
    /// for test convenience) and a per-head seed derived from a fixed base.
    /// Callers needing custom heads can swap them after build via direct
    /// field access on the resulting table — for now this is the canonical
    /// path.
    #[inline]
    pub fn new(n_slots: usize, d: usize) -> Self {
        // Allocate zero-initialized slots. We need zero-init because the
        // last-write-wins collision semantics require unpopulated slots to
        // be zero (so `lookup_into` can detect "miss" by all-zero).
        let slots =
            vec![0.0f32; n_slots.checked_mul(d).expect("n_slots*d overflow")].into_boxed_slice();
        let heads = Box::new(default_heads(n_slots));
        Self {
            slots,
            heads,
            n_slots,
            d,
        }
    }

    /// Override the default head configuration. Caller is responsible for
    /// ensuring distinct primes / seeds per head for the K-head independence
    /// property.
    #[inline]
    pub fn with_heads(mut self, heads: [HashHead; K_MAX]) -> Self {
        *self.heads = heads;
        self
    }

    /// Write a pattern into the slot at `hash_key mod n_slots`.
    ///
    /// Last-write-wins: if two writes land on the same slot, the second wins
    /// and the first is silently overwritten. `pattern.len()` MUST equal
    /// `d` (debug_asserted).
    #[inline]
    pub fn add_pattern(&mut self, hash_key: EngramHash, pattern: &[f32]) {
        debug_assert_eq!(
            pattern.len(),
            self.d,
            "add_pattern: pattern.len()={} must equal d={}",
            pattern.len(),
            self.d
        );
        if self.n_slots == 0 {
            return; // No slots to write to — silently drop. Caller error.
        }
        let slot_idx = (hash_key.0 as usize) % self.n_slots;
        let dst = &mut self.slots[slot_idx * self.d..(slot_idx + 1) * self.d];
        dst.copy_from_slice(pattern);
    }

    /// Freeze into an immutable [`InMemoryEngramTable`]. The commitment is
    /// NOT computed here — it's lazy, computed on first
    /// `commitment()` call.
    #[inline]
    pub fn build(self) -> InMemoryEngramTable {
        InMemoryEngramTable {
            slots: self.slots,
            heads: self.heads,
            n_slots: self.n_slots,
            d: self.d,
            commitment_cache: OnceLock::new(),
        }
    }
}

/// Default K_MAX head configuration. Picks a prime ≥ n_slots per head, with
/// per-head seeds derived from a fixed base. Distinct primes per head is the
/// key property that gives K_MAX independent hashes.
fn default_heads(n_slots: usize) -> [HashHead; K_MAX] {
    let mut heads = [HashHead {
        n: 0,
        k: 0,
        modulus: 1,
        seed: 0,
    }; K_MAX];
    // Use a base modulus ≥ max(n_slots, 2) to avoid mod-by-1 degeneracy.
    let base = (n_slots.max(2)) as u64;
    for k in 0..K_MAX {
        let prime = next_prime(base + k as u64); // distinct prime per head
        heads[k] = HashHead {
            n: 0,
            k: k as u8,
            modulus: prime,
            seed: 0x4242_4242_4242_4242u64
                .wrapping_add((k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        };
    }
    heads
}

/// Smallest prime ≥ n. Naive trial division — fine for one-time build, not
/// on the lookup hot path. If n < 2 returns 2.
fn next_prime(n: u64) -> u64 {
    let mut candidate = n.max(2);
    loop {
        if is_prime(candidate) {
            return candidate;
        }
        candidate += 1;
    }
}

/// Naive primality test, good enough for build-time use on small numbers.
#[inline]
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true; // 2, 3
    }
    if n % 2 == 0 {
        return false;
    }
    let mut i: u64 = 3;
    while i.saturating_mul(i) <= n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_table_all_zeros_out() {
        // T2.5: empty table → all zeros out, 0 hits returned.
        let table = EngramTableBuilder::new(1024, 4).build();
        let keys = [EngramHash(0); K_MAX];
        let mut out = [0.0f32; K_MAX * 4];
        let hits = table.lookup_into(&keys, &mut out);
        assert_eq!(hits, 0, "empty table must report 0 hits");
        assert!(out.iter().all(|&v| v == 0.0), "empty table → all zeros");
    }

    #[test]
    fn single_slot_populated_lookup_hits() {
        // T2.5: write to slot 7, look up hash(7) → hits=1 (rest zero).
        let mut b = EngramTableBuilder::new(16, 4);
        b.add_pattern(EngramHash(7), &[1.0, 2.0, 3.0, 4.0]);
        let table = b.build();

        let mut keys = [EngramHash(0); K_MAX];
        keys[0] = EngramHash(7); // head 0 → slot 7
        // Other heads land wherever their hash mod 16 sends them; with
        // default seeds they may or may not collide, but only slot 7 has
        // non-zero contents.

        let mut out = [0.0f32; K_MAX * 4];
        let hits = table.lookup_into(&keys, &mut out);
        assert!(hits >= 1, "at least head 0 must hit slot 7");
        // head 0 must have the pattern we wrote:
        assert_eq!(&out[0..4], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn k_head_retrieval_fills_all_k_slots() {
        // T2.5: lookup always fills K_MAX * D floats in `out`, regardless of
        // hit count.
        let mut b = EngramTableBuilder::new(32, 8);
        // Populate a few distinct slots.
        for i in 0..8u64 {
            let mut pat = [0.0f32; 8];
            pat[i as usize] = 1.0;
            b.add_pattern(EngramHash(i), &pat);
        }
        let table = b.build();

        let keys = [EngramHash(0); K_MAX]; // all heads → slot 0
        let mut out = [f32::NAN; K_MAX * 8]; // start with NaN to detect unwritten
        let _hits = table.lookup_into(&keys, &mut out);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "every slot in out must be written"
        );
        // Slot 0 was populated with pat[0]=1.0:
        for k in 0..K_MAX {
            assert_eq!(out[k * 8], 1.0, "head {k} slot 0 first element");
        }
    }

    #[test]
    fn commitment_deterministic_same_contents_same_blake3() {
        // T2.5: same contents → same BLAKE3 commitment.
        let mut b1 = EngramTableBuilder::new(16, 4);
        let mut b2 = EngramTableBuilder::new(16, 4);
        for i in 0..4u64 {
            let pat = [i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32];
            b1.add_pattern(EngramHash(i), &pat);
            b2.add_pattern(EngramHash(i), &pat);
        }
        let t1 = b1.build();
        let t2 = b2.build();
        assert_eq!(
            t1.commitment(),
            t2.commitment(),
            "same contents → same commitment"
        );
    }

    #[test]
    fn commitment_is_cached() {
        // First call computes; second returns cached. We can't time, but we
        // can verify they're equal + that the cache is populated.
        let table = EngramTableBuilder::new(8, 2).build();
        let a = table.commitment();
        let b = table.commitment();
        assert_eq!(a, b);
        assert!(
            table.commitment_cache.get().is_some(),
            "cache must be populated"
        );
    }

    #[test]
    fn zero_slot_table_safe_lookup() {
        // Edge case: n_slots=0. Must not divide-by-zero.
        let table = EngramTableBuilder::new(0, 4).build();
        let keys = [EngramHash(123); K_MAX];
        let mut out = [1.0f32; K_MAX * 4];
        let hits = table.lookup_into(&keys, &mut out);
        assert_eq!(hits, 0);
        assert!(out.iter().all(|&v| v == 0.0), "n_slots=0 → all zeros out");
    }

    #[test]
    fn is_prime_basic() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(!is_prime(9));
        assert!(is_prime(17));
        assert!(!is_prime(21));
    }

    #[test]
    fn next_prime_monotonic() {
        assert_eq!(next_prime(1), 2);
        assert_eq!(next_prime(2), 2);
        assert_eq!(next_prime(3), 3);
        assert_eq!(next_prime(4), 5);
        assert_eq!(next_prime(8), 11);
        assert!(is_prime(next_prime(100)));
    }
}
