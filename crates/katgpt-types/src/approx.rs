//! Checked approximation contracts — the ONE ULP metric behind the
//! test/bench float-exactness gates (Issue 944, from ink-lang Research 475).
//!
//! Three copies of the same two's-complement bit-distance formula used to
//! live beside their gates (`bench_844`'s `ulp_diff`,
//! `simd/activations.rs`'s `ulp_diff`, `bench_871`'s `ulp_delta`), each with
//! its own unstated edge semantics. This module is the consolidation: one
//! metric with one stated law, plus the declarative [`ApproxContract`] the
//! gates measure against their named references.
//!
//! Test/bench-only by convention (no hot-path caller), deliberately UNGATED:
//! a feature gate here would compile every migrated gate to a silent
//! `ok. 0 passed` the moment the feature is off (the green-zero class).
//! Zero-alloc on the measure path; the only allocation is a panic message.

/// Monotone ULP distance between two finite `f32` values.
///
/// The law (one law, stated here — the three former copies disagreed):
///
/// - **Signed zeros share one position.** `ulp_distance(0.0, -0.0)` is
///   `Some(0)`, not the ~2³¹ artifact the raw two's-complement map gives.
/// - **Non-finite fails closed.** Either side NaN or ±Inf → `None`. The raw
///   bit map over NaN payloads and infinity encodings is a number that means
///   nothing; refusing beats returning it.
/// - **Finite inputs**: the classic two's-complement bit distance,
///   bit-identical to all three pre-consolidation copies on their same-sign
///   domains (bench_871's documented same-sign precondition is subsumed —
///   the law no longer depends on it for the zeros it excluded).
///
/// ```
/// use katgpt_types::approx::ulp_distance;
/// assert_eq!(ulp_distance(1.0, 1.0), Some(0));
/// assert_eq!(
///     ulp_distance(1.0, f32::from_bits(1.0f32.to_bits() + 1)),
///     Some(1)
/// );
/// assert_eq!(ulp_distance(0.0, -0.0), Some(0));
/// assert_eq!(ulp_distance(f32::NAN, 1.0), None);
/// ```
pub fn ulp_distance(a: f32, b: f32) -> Option<u32> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    // One position for ±0: `== 0.0` is true for both zeros and for nothing
    // else, so this fold is exactly the one-position law.
    let a = if a == 0.0 { 0.0 } else { a };
    let b = if b == 0.0 { 0.0 } else { b };
    Some((a.to_bits() as i32).abs_diff(b.to_bits() as i32))
}

/// A named ULP bound measured over a domain — the declarative half of an
/// approximation contract (ink's `approximates ref within ulps K`).
///
/// A gate accumulates [`Self::measure`] across its sweep and asserts the
/// bound once:
///
/// ```
/// use katgpt_types::approx::ApproxContract;
/// let mut contract = ApproxContract::new("sigmoid vs f64 reference");
/// for x in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
///     let reference = (1.0f64 / (1.0 + (-(x as f64)).exp())) as f32;
///     contract.measure((1.0f64 / (1.0 + (-(x as f64)).exp())) as f32, reference);
/// }
/// contract.assert_within(4);
/// ```
///
/// Fail-closed on all three silent-pass shapes: a non-finite pair poisons
/// the contract, an EMPTY contract refuses to assert (a sweep loop that
/// never ran would otherwise pass), and a violated bound names the worst
/// pair in the panic.
#[derive(Debug)]
pub struct ApproxContract {
    name: &'static str,
    /// The worst finite measurement: (distance, actual, reference).
    worst: Option<(u32, f32, f32)>,
    poisoned: bool,
}

impl ApproxContract {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            worst: None,
            poisoned: false,
        }
    }

    /// Record one measured pair. Zero-alloc.
    pub fn measure(&mut self, actual: f32, reference: f32) {
        match ulp_distance(actual, reference) {
            None => self.poisoned = true,
            Some(d) => {
                if self.worst.is_none_or(|(w, _, _)| d > w) {
                    self.worst = Some((d, actual, reference));
                }
            }
        }
    }

    /// The worst measured distance, or `None` when nothing finite was
    /// measured or the contract is poisoned.
    pub fn max_ulps(&self) -> Option<u32> {
        if self.poisoned {
            None
        } else {
            self.worst.map(|(d, _, _)| d)
        }
    }

    /// Whether [`Self::assert_within`] would pass — the benches' verdict-bit
    /// half. `false` when poisoned or empty.
    pub fn passes(&self, bound_ulps: u32) -> bool {
        self.max_ulps().is_some_and(|d| d <= bound_ulps)
    }

    /// Panic unless every measured pair is within `bound_ulps` ULPs of its
    /// reference. Panics on a poisoned or empty contract too — loud, with
    /// the reason named.
    pub fn assert_within(&self, bound_ulps: u32) {
        if self.poisoned {
            panic!(
                "approx contract `{}` measured a NON-FINITE pair — fail closed",
                self.name
            );
        }
        match self.worst {
            None => panic!(
                "approx contract `{}` measured NOTHING — an empty sweep is not a pass",
                self.name
            ),
            Some((d, actual, reference)) if d > bound_ulps => panic!(
                "approx contract `{}` violated: worst {d} ULP (actual {actual:e}, \
                 reference {reference:e}) > bound {bound_ulps} ULP",
                self.name
            ),
            Some(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ApproxContract, ulp_distance};

    #[test]
    fn signed_zeros_share_one_position() {
        assert_eq!(ulp_distance(0.0, -0.0), Some(0));
        assert_eq!(ulp_distance(-0.0, -0.0), Some(0));
        assert_eq!(ulp_distance(0.0, 0.0), Some(0));
    }

    #[test]
    fn identical_to_the_raw_two_complement_map_on_finite_domains() {
        // The consolidation must not move any existing gate's measurement:
        // on finite non-zero pairs (every existing domain) the shared metric
        // is bit-identical to the three former local copies. Deterministic
        // stride, no RNG (the global_rng_gate class).
        let raw = |a: f32, b: f32| (a.to_bits() as i32).abs_diff(b.to_bits() as i32);
        let mut bits = 0x0080_0000u32; // smallest normal magnitude
        while bits < 0x7F00_0000 {
            let a = f32::from_bits(bits);
            let b = f32::from_bits(bits.wrapping_add(0x0143));
            let c = f32::from_bits(bits.wrapping_add(0x7B21));
            assert_eq!(ulp_distance(a, b), Some(raw(a, b)));
            assert_eq!(ulp_distance(a, c), Some(raw(a, c)));
            let na = f32::from_bits(bits | 0x8000_0000);
            let nb = f32::from_bits(bits.wrapping_add(0x0143) | 0x8000_0000);
            assert_eq!(ulp_distance(na, nb), Some(raw(na, nb)));
            bits = bits.wrapping_add(0x04D1_3F0D); // odd stride, ~26 stops
        }
    }

    #[test]
    fn adjacent_values_are_one_ulp() {
        let next = f32::from_bits(1.0f32.to_bits() + 1);
        assert_eq!(ulp_distance(1.0, next), Some(1));
        let prev = f32::from_bits(1.0f32.to_bits() - 1);
        assert_eq!(ulp_distance(1.0, prev), Some(1));
        assert_eq!(ulp_distance(1.0, 1.0), Some(0));
    }

    #[test]
    fn non_finite_fails_closed() {
        for a in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(ulp_distance(a, 1.0), None);
            assert_eq!(ulp_distance(1.0, a), None);
            assert_eq!(ulp_distance(a, a), None);
        }
    }

    #[test]
    fn contract_passes_a_true_bound() {
        let mut contract = ApproxContract::new("exact");
        let next = f32::from_bits(1.0f32.to_bits() + 1);
        contract.measure(next, 1.0);
        contract.measure(1.0, 1.0);
        assert_eq!(contract.max_ulps(), Some(1));
        assert!(contract.passes(4));
        contract.assert_within(4);
    }

    #[test]
    fn contract_names_a_seeded_violation() {
        let mut contract = ApproxContract::new("seeded violation");
        contract.measure(1.0, 1.0);
        contract.measure(2.0, 1.0); // ~8.4M ULP — a real gap
        assert!(!contract.passes(4));
        let msg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            contract.assert_within(4);
        }))
        .expect_err("the violated contract must panic");
        let msg = msg
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| msg.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .expect("string panic payload");
        assert!(msg.contains("worst"), "{msg}");
        assert!(msg.contains("seeded violation"), "{msg}");
    }

    #[test]
    fn contract_refuses_non_finite_and_empty() {
        let mut poisoned = ApproxContract::new("poison");
        poisoned.measure(f32::NAN, 1.0);
        assert!(!poisoned.passes(4));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| poisoned.assert_within(4)))
                .is_err()
        );

        let empty = ApproxContract::new("empty");
        assert!(!empty.passes(4));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| empty.assert_within(4)))
                .is_err()
        );
    }
}
