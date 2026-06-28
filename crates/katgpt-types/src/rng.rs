//! XorShift64 PRNG.

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

/// XorShift64 PRNG — deterministic per seed.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Construct from a 64-bit seed.
    ///
    /// Applies one round of SplitMix64 mixing to decorrelate low-entropy seeds
    /// (Issue 296). Without this, small seeds like `0`, `1`, `2`, `42` keep the
    /// XorShift64 state small enough that the first `next()` has zero upper 24
    /// bits, so `uniform()` returns exactly `0.0` — which sits on the left
    /// boundary of any inverse-CDF sampler and deterministically selects the
    /// first token with nonzero mass. SplitMix64 is the standard remedy for
    /// XorShift "small seed" weakness: it diffuses any input (including 0)
    /// into a well-distributed 64-bit state in O(1).
    ///
    /// Cost: ~3 mul/shift per `Rng::new`. Negligible — `new()` is called once
    /// per inference/training session, not in hot loops.
    pub fn new(seed: u64) -> Self {
        // SplitMix64 finalizer (Steele/Lea/Leiserson — see Java SplittableRandom).
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let state = z ^ (z >> 31);
        // XorShift64 with state == 0 is an absorbing state (stuck at 0 forever).
        // SplitMix64 never outputs 0 for any u64 input in practice, but guard
        // defensively: the cost is one branch on construction, not on hot paths.
        Self {
            state: if state == 0 { 1 } else { state },
        }
    }

    #[allow(clippy::should_implement_trait)]
    #[inline(always)]
    pub fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform [0, 1) — 24 bits of entropy (full f32 mantissa precision).
    ///
    /// Note: returns a value in the half-open interval [0, 1). The exact bound
    /// `0.0` is reachable in principle (24 zero bits) but, after the SplitMix64
    /// seed mixing in `Rng::new`, is no longer the deterministic first draw for
    /// small seeds. Samplers that require a strictly positive variate should
    /// still redraw on `r == 0.0` as a defensive belt — see Issue 296.
    #[inline(always)]
    pub fn uniform(&mut self) -> f32 {
        // Bit-manipulation trick: take upper 24 bits, OR into the 23-bit mantissa
        // position with exponent set to 0x3f80 (1.0), then subtract 1.0.
        // This gives exactly 24 bits of uniform random bits mapped to [0, 1).
        let bits = ((self.next() >> 40) as u32 & 0x007f_ffff) | 0x3f80_0000;
        f32::from_bits(bits) - 1.0
    }

    /// Standard normal via Box-Muller transform.
    #[inline(always)]
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests_rng {
    use super::*;

    /// Issue 296 regression: `Rng::new(s).uniform()` must not be exactly `0.0`
    /// on the first draw for any small seed. Before the SplitMix64 fix in
    /// `Rng::new`, seeds `{0,1,2,3,42}` all produced `uniform() == 0.0` first.
    #[test]
    fn first_uniform_is_nonzero_for_small_seeds() {
        for &seed in &[0u64, 1, 2, 3, 4, 5, 42, 100, 1337, 0xFFFF_FFFF] {
            let first = Rng::new(seed).uniform();
            assert_ne!(
                first, 0.0,
                "seed {seed}: first uniform() must not be 0.0 (Issue 296)"
            );
            assert!(first > 0.0 && first < 1.0, "seed {seed}: uniform() out of range: {first}");
        }
    }

    /// `Rng::new(0)` must still produce a usable (non-zero, evolving) state —
    /// XorShift64 with state 0 is stuck at 0 forever.
    #[test]
    fn new_zero_seed_is_not_stuck() {
        let mut rng = Rng::new(0);
        let a = rng.next();
        let b = rng.next();
        assert_ne!(a, 0, "Rng::new(0).next() must not be 0");
        assert_ne!(a, b, "Rng::new(0) must evolve");
    }

    /// Same seed → same sequence (determinism preserved after SplitMix64).
    #[test]
    fn same_seed_same_sequence() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..16 {
            assert_eq!(a.uniform(), b.uniform());
        }
    }

    /// Different small seeds → different first draws (the original defect:
    /// seeds 1 and 2 produced identical sampler output because both yielded 0.0).
    #[test]
    fn different_small_seeds_diverge_immediately() {
        let r1 = Rng::new(1).uniform();
        let r2 = Rng::new(2).uniform();
        let r3 = Rng::new(3).uniform();
        assert_ne!(r1, r2, "seeds 1 and 2 must differ on first draw");
        assert_ne!(r1, r3, "seeds 1 and 3 must differ on first draw");
        assert_ne!(r2, r3, "seeds 2 and 3 must differ on first draw");
    }

    /// Lightweight χ² goodness-of-fit on the first 65_536 uniforms: bin into
    /// 256 equal-width buckets and check the statistic is within reason.
    /// This is not a full DIEHARD suite, but catches catastrophic bias.
    #[test]
    fn uniform_is_well_distributed() {
        const BUCKETS: usize = 256;
        const N: usize = 65_536;
        let mut rng = Rng::new(42);
        let mut counts = [0u32; BUCKETS];
        for _ in 0..N {
            let u = rng.uniform();
            let idx = ((u * BUCKETS as f32) as usize).min(BUCKETS - 1);
            counts[idx] += 1;
        }
        let expected = N as f64 / BUCKETS as f64;
        let chi_sq: f64 = counts
            .iter()
            .map(|&c| {
                let d = c as f64 - expected;
                d * d
            })
            .sum::<f64>()
            / expected;
        // 255 degrees of freedom: 99% critical value ≈ 330.09; 1% ≈ 188.33.
        // Use a generous upper bound of 400 to avoid flakes from V8/CI variance.
        assert!(
            chi_sq < 400.0,
            "χ²={chi_sq:.1} exceeds 400 (255 DoF, 99.99%ile ≈ 336) — distribution is biased"
        );
    }
}
