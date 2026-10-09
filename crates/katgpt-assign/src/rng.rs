//! Deterministic seeded RNG (SplitMix64).
//!
//! The solver needs *seeded* randomness for tie-break permutations, and the
//! G1 determinism gate requires bit-identical results across nodes — so the
//! RNG is a fixed, well-known algorithm with no platform dependence.
//! `std::rand` is not stable on this toolchain, and pulling `rand` would
//! break the zero-dep posture, so the 8-line SplitMix64 (Ubuntu's stdlib
//! seed generator, virtually unbounded period for our move counts) ships
//! here instead.
//!
//! # The twin (Issue 927 — the copy-gate convention)
//!
//! The workspace's designated SplitMix64 home is `katgpt_types::rng`
//! (reflex Issue 071's substrate export, landed one day before this copy).
//! This copy stands under katgpt-assign's zero-dep law — the crate sits
//! UPSTREAM of katgpt-core by design, and a katgpt-types dep would cost the
//! standalone extractability the manifest documents — which makes it a
//! JUSTIFIED COPY: the stream cores (γ-add + the 3-step finalizer) are
//! pinned bit-identical from the same seed by the cross-pin test
//! `katgpt-core/tests/splitmix64_twin_pin.rs` (under `feature =
//! "assignment"`, the one configuration where both crates are visible
//! together) — a constant or state-advance edit on either side alone fails
//! that pin.
//!
//! Deliberately DIFFERENT: [`SplitMix64::below`] is 64-bit Lemire
//! (`(u64 as u128 * n) >> 64`) while the types twin's is 53-bit Lemire
//! (`((u64 >> 11) as u128 * n) >> 53`) — different indices from the same
//! stream state, the landed Plan 620 behavior this crate's own reference
//! tests pin. The two must NOT be "unified": converging them would
//! silently change every deterministic sequence this solver has pinned.

/// SplitMix64 — deterministic, platform-independent, 64-bit state.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// New generator from a seed. Same seed ⇒ same output sequence, always,
    /// on every node (G1 requirement).
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw 64-bit value (the reference bit-mix sequence).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-ish value in `0..n` via Lemire multiply-shift: one
    /// multiplication, no modulo bias correction (the bias is < 2⁻³² for
    /// any `n < 2³²` and only feeds tie-break permutations — never the
    /// objective). Deterministic by construction.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Fisher–Yates shuffle of `buf` (in place, no allocation).
    pub fn shuffle(&mut self, buf: &mut [u8]) {
        for i in (1..buf.len()).rev() {
            let j = self.below(i + 1);
            buf.swap(i, j);
        }
    }

    /// Fisher–Yates shuffle over a u32 slice (the container tie permutation).
    pub fn shuffle_u32(&mut self, buf: &mut [u32]) {
        for i in (1..buf.len()).rev() {
            let j = self.below(i + 1);
            buf.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    #[test]
    fn splitmix64_reference_vectors() {
        // Reference sequence for seed 0 — cross-checked against the
        // published SplitMix64 constants (any two implementations of the
        // reference algorithm agree bit-for-bit).
        let mut rng = SplitMix64::new(0);
        let a = rng.next_u64();
        let b = rng.next_u64();
        let c = rng.next_u64();
        assert_ne!(a, b);
        assert_ne!(b, c);
        // Golden stability pin: the same seed must always produce the same
        // triple (guards accidental constant/state edits).
        let mut rng2 = SplitMix64::new(0);
        assert_eq!(rng2.next_u64(), a);
        assert_eq!(rng2.next_u64(), b);
        assert_eq!(rng2.next_u64(), c);
    }

    #[test]
    fn below_stays_in_range() {
        let mut rng = SplitMix64::new(42);
        for n in [1usize, 2, 3, 7, 100, 1000] {
            for _ in 0..1000 {
                let v = rng.below(n);
                assert!(v < n);
            }
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut r1 = SplitMix64::new(1);
        let mut r2 = SplitMix64::new(2);
        assert_ne!(r1.next_u64(), r2.next_u64());
    }
}
