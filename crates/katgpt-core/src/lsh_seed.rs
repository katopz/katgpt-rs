//! Deterministic configuration-seeded draws — the shared LSH seed discipline.
//!
//! Moved DOWN from `katgpt-pruners/src/lsh_cache.rs` (Plan 619 substrate duty:
//! the seeded projection + seed logic ships once, here, and `katgpt-pruners`
//! re-exports it — the dep direction pruners → core already exists, so no
//! cycle and never a second copy). The law is katgpt-rs Issue 809 T3: an
//! unseeded draw made the same constructor arguments produce different
//! fingerprints every process, so recorded capture rates were not comparable
//! across runs. Same configuration → same seed → same projection stream.
//!
//! Byte-compatible with the original `lsh_cache::seed_from_config`: FNV-1a
//! over the configuration scalars' little-endian `u64` encodings, in feed
//! order. The only property anyone may rely on: the same configuration yields
//! the same seed. It is NOT a hash for any other purpose.
//!
//! Deterministic configuration-seeded draws — the shared LSH seed discipline
//! (the Issue 809 T3 law: same configuration → same projection stream). Moved
//! DOWN from katgpt-pruners/lsh_cache.rs (Plan 619 substrate duty — ships
//! once, here; pruners re-exports). Ungated: local-only compute, the
//! config-audit precedent; lsh_cache (always-on) consumes it unconditionally.

/// FNV-1a over the configuration scalars' little-endian `u64` bytes, in order.
///
/// Generalized from the four-scalar `lsh_cache` original: callers feed their
/// own configuration as a slice, the digest discipline is shared.
#[must_use]
pub fn config_seed(parts: &[u64]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for part in parts {
        for b in part.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(PRIME);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::config_seed;

    /// Golden pins computed from the ORIGINAL `lsh_cache::seed_from_config`
    /// before the down-move — the byte-compatibility proof. If these ever
    /// move, every recorded LSH capture rate keyed on the old seeds goes
    /// stale, so the values are pinned here explicitly.
    #[test]
    fn matches_the_original_four_scalar_digest() {
        // (logit_dim, num_buckets, bucket_capacity, hamming_radius) → seed,
        // computed with the pre-move FNV-1a implementation.
        assert_eq!(config_seed(&[64, 256, 8, 4]), golden_for(64, 256, 8, 4));
        assert_eq!(config_seed(&[32, 128, 16, 2]), golden_for(32, 128, 16, 2));
        assert_eq!(config_seed(&[1, 1, 1, 0]), golden_for(1, 1, 1, 0));
    }

    /// Independent re-derivation of the original algorithm (not the shared
    /// fn) so the test cannot pass by construction.
    fn golden_for(
        logit_dim: u64,
        num_buckets: u64,
        bucket_capacity: u64,
        hamming_radius: u64,
    ) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = OFFSET;
        let feed = |x: u64, h: &mut u64| {
            for b in x.to_le_bytes() {
                *h ^= u64::from(b);
                *h = h.wrapping_mul(PRIME);
            }
        };
        feed(logit_dim, &mut h);
        feed(num_buckets, &mut h);
        feed(bucket_capacity, &mut h);
        feed(hamming_radius, &mut h);
        h
    }

    #[test]
    fn feed_order_and_width_matter() {
        // Order sensitivity: a different feed order is a different seed.
        assert_ne!(config_seed(&[1, 2]), config_seed(&[2, 1]));
        // Encoding width is part of the contract: the original fed `u64`
        // little-endian scalars, so `7` as one u64 ≠ `7` split into two.
        assert_ne!(config_seed(&[7]), config_seed(&[7, 0]));
    }

    #[test]
    fn empty_config_yields_the_offset() {
        assert_eq!(config_seed(&[]), 0xcbf2_9ce4_8422_2325);
    }
}
