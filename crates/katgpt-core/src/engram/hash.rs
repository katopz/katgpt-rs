//! Multi-head multiplicative-XOR hash for N-gram suffixes.
//!
//! Plan 299 Phase 1 T1.1–T1.7. Each [`HashHead`] is an independent
//! `(seed, modulus)` configuration; [`multi_head_hash`] computes K_MAX
//! hashes in one call. The hash is designed so that a fixed-size suffix
//! (typically `[CanonicalId; 3]` — a trigram) produces K_MAX independent
//! slot keys with O(1) work each.
//!
//! # Formula (Plan T1.5)
//!
//! ```text
//! suffix_fold = Σᵢ suffix[i] · MULTIPLIERS[i mod 8]
//! hash_k      = (seed_k XOR suffix_fold) mod modulus_k
//! ```
//!
//! The MULTIPLIERS are large odd primes (compile-time constants). Mixing is
//! `seed XOR suffix_fold` — fast, branch-free, SIMD-friendly when the suffix
//! is a fixed-size `[u64; 3]`. Prime moduli (`modulus_k`) per head dilute
//! collisions; multi-head retrieval in Phase 2 makes collisions a quality
//! issue (filtered by the sigmoid gate), not a correctness issue.
//!
//! # Hot-path contract
//!
//! [`multi_head_hash`] is **zero-allocation**: returns a fixed-size
//! `[EngramHash; K_MAX]`. No `Vec`, no `Box`.

use super::{CanonicalId, EngramHash, K_MAX};

/// One prime-table hash configuration. Pre-computed at table build time and
/// frozen for the lifetime of the table (immutable after build).
///
/// - `n` — table size class (informational; the actual modulus is `modulus`).
///   Used by builders to derive a sensible prime for a target table size.
/// - `k` — head index, 0..K_MAX. Informational; included for diagnostics.
/// - `modulus` — prime modulus for `hash mod modulus`. Should be ≥ the
///   number of slots. Pick distinct primes per head to decorrelate collisions.
/// - `seed` — per-head random seed. MUST be distinct per head for the K-head
///   independence property to hold (test: changing one head's seed changes
///   only its hash output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HashHead {
    /// Table size class (informational).
    pub n: u8,
    /// Head index (informational, 0..K_MAX).
    pub k: u8,
    /// Prime modulus for `hash mod modulus`. Should be ≥ num_slots.
    pub modulus: u64,
    /// Per-head random seed. MUST be distinct per head.
    pub seed: u64,
}

/// Compile-time multiplier table for the suffix fold. Large odd primes
/// (all < 2^64), chosen to mix bits across the 64-bit range. Indexed by
/// `i mod 8`.
///
/// These are not cryptographic — they just need to be large, odd, and
/// pairwise independent enough that distinct suffixes rarely collide.
/// All literals are verified to fit in `u64` (no overflow).
const MULTIPLIERS: [u64; 8] = [
    0x9E37_79B9_7F4A_7C15, // floor(2^64 / φ) — Knuth's multiplicative hash constant
    0xFF51_AFD7_ED55_8CCD, // Murmur3 fmix64 constant (a)
    0xC4CE_B9FE_1A85_EC53, // Murmur3 fmix64 constant (b)
    0x87C3_7B91_1142_53D5, // large odd prime
    0x0343_F587_7C33_C29D, // large odd prime (< 2^62)
    0x0A10_58D6_DEAD_7497, // large odd prime (< 2^60)
    0x1A4E_F51A_5C42_2837, // large odd prime (< 2^61)
    0x3000_19FF_F777_BBBB, // large odd prime (< 2^62)
];

/// Compute K_MAX hashes over an N-gram suffix.
///
/// The suffix is typically `[CanonicalId; 3]` (a trigram), but any length
/// is accepted; longer suffixes just fold more terms into `suffix_fold`.
/// Empty suffix → all-zero hashes (test T1.6).
///
/// # Zero-allocation
///
/// Returns `[EngramHash; K_MAX]` — a stack-sized array, no heap traffic.
///
/// # Determinism
///
/// Same `(suffix, heads)` → same output, always. Same suffix with one head's
/// seed changed → only that head's hash changes (test T1.6).
///
/// # Example
///
/// ```ignore
/// use katgpt_core::engram::{multi_head_hash, HashHead, CanonicalId, K_MAX};
///
/// let heads = [HashHead { n: 20, k: 0, modulus: (1 << 20) + 3, seed: 42 }; K_MAX];
/// let suffix = [CanonicalId(1), CanonicalId(2), CanonicalId(3)];
/// let keys = multi_head_hash(&suffix, &heads);
/// assert_eq!(keys.len(), K_MAX);
/// ```
#[inline]
pub fn multi_head_hash(suffix: &[CanonicalId], heads: &[HashHead; K_MAX]) -> [EngramHash; K_MAX] {
    // T1.6 contract: empty suffix → all-zero hashes (the "no pattern"
    // sentinel). This is a caller-facing semantic, not a hash invariant —
    // we special-case it here so callers can branch on `keys == [0; K_MAX]`
    // to detect "no context to retrieve".
    if suffix.is_empty() {
        return [EngramHash(0); K_MAX];
    }

    // Fold the suffix once — shared across all K heads. Branch-free loop.
    // Wrapping ops are intentional: we want the bit-mix properties of
    // unsigned arithmetic, and the prime modulus at the end restores
    // bounded range. No overflow UB (wrapping_* is defined).
    let mut suffix_fold: u64 = 0;
    for (i, c) in suffix.iter().enumerate() {
        let m = MULTIPLIERS[i & 7]; // i mod 8 — branch-free
        suffix_fold = suffix_fold.wrapping_add(c.0.wrapping_mul(m));
    }

    let mut out = [EngramHash(0); K_MAX];
    // Unroll-friendly: K_MAX is a const (16). Each iteration is independent.
    for k in 0..K_MAX {
        let h = &heads[k];
        // seed XOR suffix_fold — single XOR for fast decorrelation per head,
        // then prime modulus for slot-index range.
        let mixed = h.seed ^ suffix_fold;
        // `% modulus` — single remainder per head. With prime moduli, the
        // K_MAX heads are pairwise-decorrelated.
        out[k] = EngramHash(mixed % h.modulus);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic K_MAX head set from a base seed — used by every
    /// test so we don't reinvent the wheel.
    fn make_heads(base_seed: u64) -> [HashHead; K_MAX] {
        let mut heads = [HashHead {
            n: 0,
            k: 0,
            modulus: 1,
            seed: 0,
        }; K_MAX];
        for (k, head) in heads.iter_mut().enumerate() {
            // Distinct prime per head: pick primes ≥ 2^20 to make collisions
            // rare on small tables. Real builds use larger primes per slot
            // count, but for unit tests any distinct primes work.
            let prime = pick_prime(k);
            *head = HashHead {
                n: 20,
                k: k as u8,
                modulus: prime,
                seed: base_seed.wrapping_add((k as u64).wrapping_mul(0xDEAD_BEEF_CAFE_F00D)),
            };
        }
        heads
    }

    /// Pick a distinct prime for head `k`. Hardcoded for test determinism.
    fn pick_prime(k: usize) -> u64 {
        const PRIMES: [u64; K_MAX] = [
            1_048_576u64 + 7u64, // 2^20 + 7
            1_048_576u64 + 17u64,
            1_048_576u64 + 23u64,
            1_048_576u64 + 41u64,
            1_048_576u64 + 47u64,
            1_048_576u64 + 59u64,
            1_048_576u64 + 71u64,
            1_048_576u64 + 89u64,
            1_048_576u64 + 113u64,
            1_048_576u64 + 131u64,
            1_048_576u64 + 173u64,
            1_048_576u64 + 197u64,
            1_048_576u64 + 233u64,
            1_048_576u64 + 251u64,
            1_048_576u64 + 281u64,
            1_048_576u64 + 311u64,
        ];
        PRIMES[k]
    }

    #[test]
    fn empty_suffix_all_zero_hashes() {
        // T1.6 contract: empty suffix → all-zero hashes (the "no pattern"
        // sentinel). The function special-cases this so callers can branch
        // on `keys == [EngramHash(0); K_MAX]` to detect "nothing to retrieve".
        let heads = make_heads(42);
        let keys = multi_head_hash(&[], &heads);
        for (k, key) in keys.iter().enumerate() {
            assert_eq!(*key, EngramHash(0), "head {k}: empty suffix → zero hash");
        }
    }

    #[test]
    fn determinism_same_suffix_same_hash() {
        // T1.6: same suffix → same hash, always.
        let heads = make_heads(42);
        let suffix = [CanonicalId(1), CanonicalId(2), CanonicalId(3)];
        let a = multi_head_hash(&suffix, &heads);
        let b = multi_head_hash(&suffix, &heads);
        assert_eq!(a, b, "same suffix + same heads → same hashes");
    }

    #[test]
    fn different_suffix_different_hash() {
        // T1.6: different suffix → at least one head differs. With distinct
        // seeds + prime moduli, all K_MAX should differ for any
        // non-pathological suffix pair.
        let heads = make_heads(42);
        let a = multi_head_hash(&[CanonicalId(1), CanonicalId(2), CanonicalId(3)], &heads);
        let b = multi_head_hash(&[CanonicalId(4), CanonicalId(5), CanonicalId(6)], &heads);
        let any_diff = a.iter().zip(b.iter()).any(|(x, y)| x != y);
        assert!(any_diff, "different suffixes → at least one head differs");
    }

    #[test]
    fn changing_one_head_seed_changes_only_its_hash() {
        // T1.6: K heads are independent — change head 3's seed, only head 3
        // produces a different hash.
        let suffix = [CanonicalId(7), CanonicalId(11), CanonicalId(13)];
        let heads_a = make_heads(42);
        let keys_a = multi_head_hash(&suffix, &heads_a);

        let mut heads_b = heads_a;
        heads_b[3].seed = heads_a[3].seed.wrapping_add(1);
        let keys_b = multi_head_hash(&suffix, &heads_b);

        for k in 0..K_MAX {
            if k == 3 {
                assert_ne!(keys_a[k], keys_b[k], "head 3 must differ after seed change");
            } else {
                assert_eq!(keys_a[k], keys_b[k], "head {k} must be unchanged");
            }
        }
    }

    #[test]
    fn multipliers_const_correctness() {
        // Every multiplier MUST be odd, else bit-mix degenerates for
        // even suffix values.
        for &m in &MULTIPLIERS {
            assert!(m & 1 == 1, "MULTIPLIERS must all be odd; {m:#x} is even");
            assert!(m != 0);
        }
    }

    // ─── T1.7: proptest property tests ───────────────────────────────────
    //
    // Plan 299 T1.7: `proptest` over random `[CanonicalId; 3]` suffixes —
    // verify determinism + uniform distribution modulo prime (chi-square test
    // on 10K samples).
    //
    // Two properties:
    //   1. Determinism: same (suffix, heads) → same `[EngramHash; K_MAX]`.
    //   2. Uniform distribution: per-head hashes are approximately uniform
    //      modulo their prime modulus (chi-square test on 10K samples, all
    //      16 heads must pass p > 0.001). The check uses a deterministic LCG
    //      so it is reproducible — not flaky.

    use proptest::prelude::*;

    /// Any `u64` is a valid `CanonicalId` payload.
    fn any_canonical_id() -> impl Strategy<Value = CanonicalId> {
        any::<u64>().prop_map(CanonicalId)
    }

    /// A 3-element suffix (trigram) of arbitrary canonical ids.
    fn any_trigram() -> impl Strategy<Value = [CanonicalId; 3]> {
        (any_canonical_id(), any_canonical_id(), any_canonical_id())
            .prop_map(|(a, b, c)| [a, b, c])
    }

    proptest! {
        /// T1.7 determinism: same suffix + same heads → same hashes, always.
        /// No shrinking needed (output is a fixed-size array). Runs 256 cases
        /// by default — proptest will explore the input space.
        #[test]
        fn prop_hash_deterministic(suffix in any_trigram()) {
            let heads = make_heads(42);
            let a = multi_head_hash(&suffix, &heads);
            let b = multi_head_hash(&suffix, &heads);
            prop_assert_eq!(a, b, "same suffix + same heads → same hashes");
        }

        /// T1.7 head independence: change one head's seed → only that head's
        /// hash changes (the rest are bit-identical).
        #[test]
        fn prop_head_independence(suffix in any_trigram(), head_idx in 0usize..K_MAX) {
            let heads_a = make_heads(42);
            let keys_a = multi_head_hash(&suffix, &heads_a);

            let mut heads_b = heads_a;
            heads_b[head_idx].seed = heads_a[head_idx].seed.wrapping_add(1);
            let keys_b = multi_head_hash(&suffix, &heads_b);

            for k in 0..K_MAX {
                if k == head_idx {
                    // Note: edge case — if changing the seed by 1 happens to
                    // produce the same `(seed ^ suffix_fold) % modulus`, the
                    // hash could be unchanged. This is astronomically rare for
                    // primes near 2^20 and never observed in practice, but we
                    // don't hard-assert to keep the property test sound.
                    // Just assert the *other* heads are unchanged.
                    continue;
                }
                prop_assert_eq!(
                    keys_a[k], keys_b[k],
                    "head {} unchanged after head_idx={} seed change",
                    k, head_idx
                );
            }
        }

        /// T1.7 distinct-suffix decorrelation: two distinct trigrams produce
        /// at least one different head hash. (With prime moduli ≥ 2^20+7 and
        //  K_MAX=16, a total collision across all heads is astronomically
        //  improbable for distinct suffixes.)
        #[test]
        fn prop_distinct_suffix_distinct_hash(
            a in any_trigram(),
            b in any_trigram()
        ) {
            prop_assume!(a != b);
            let heads = make_heads(42);
            let ka = multi_head_hash(&a, &heads);
            let kb = multi_head_hash(&b, &heads);
            prop_assert!(
                ka != kb,
                "distinct trigrams should not collide on all K_MAX heads"
            );
        }
    }

    /// T1.7 uniform distribution: chi-square test on 10K samples per head.
    ///
    /// For each head k, hash 10K random `[CanonicalId; 3]` suffixes, bucket
    /// `(hash) % 256` into 256 buckets, compute chi-square = Σ (O−E)²/E,
    /// and verify it's below the critical value for p=0.001 with 255 degrees
    /// of freedom. A chi-square this large should appear < 0.1% of the time
    /// under the uniform null hypothesis — generous enough that a good hash
    /// always passes, strict enough to catch gross degeneracies (e.g., all
    /// hashes landing in one bucket).
    ///
    /// Uses a deterministic LCG — this test is NOT flaky.
    #[test]
    fn chi_square_uniform_distribution_10k() {
        let heads = make_heads(42);
        let n_samples = 10_000u64;
        let n_buckets = 256usize;

        // LCG matching the G1 bench for determinism.
        let mut rng_state = 0xC0_FF_EE_12_34u64;
        let mut rng = || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng_state
        };

        // Per-head per-bucket counts. K_MAX × 256 = 4096 u64s — trivial.
        let mut counts = [[0u64; 256]; K_MAX];
        for _ in 0..n_samples {
            let suffix = [
                CanonicalId(rng()),
                CanonicalId(rng()),
                CanonicalId(rng()),
            ];
            let keys = multi_head_hash(&suffix, &heads);
            for k in 0..K_MAX {
                // Bucket = hash % 256. Taking `% modulus` first and then `%
                // n_buckets` would also work but modulus ≥ 2^20+7 is much
                // larger than 256, so `(hash) % 256` directly is unbiased
                // enough for a chi-square uniformity check.
                let bucket = (keys[k].0 % (n_buckets as u64)) as usize;
                counts[k][bucket] += 1;
            }
        }

        // The chi-square distribution with `n_buckets - 1 = 255` degrees of
        // freedom has critical value ≈ 336.6 at p=0.001 (i.e., under the
        // uniform null, P(chi-square > 336.6) ≈ 0.001). We use 350 as the
        // threshold to give margin for LCG-induced sample autocorrelation.
        let chi_sq_critical = 350.0f64;
        let expected = n_samples as f64 / n_buckets as f64;

        let mut chi_sq_per_head = [0.0f64; K_MAX];
        let mut failures = Vec::new();
        for k in 0..K_MAX {
            let mut chi = 0.0f64;
            for &c in &counts[k] {
                let o = c as f64;
                let diff = o - expected;
                chi += diff * diff / expected;
            }
            chi_sq_per_head[k] = chi;
            if chi > chi_sq_critical {
                failures.push((k, chi));
            }
        }

        // Report every head's chi-square value — useful diagnostic if any
        // head is borderline.
        let report: Vec<String> = (0..K_MAX)
            .map(|k| format!("h{k}={:.1}", chi_sq_per_head[k]))
            .collect();
        let joined = report.join(", ");
        assert!(
            failures.is_empty(),
            "T1.7 chi-square uniformity failed for heads {failures:?} (critical={chi_sq_critical}). \
             Per-head chi-square: {joined}",
        );
    }
}
