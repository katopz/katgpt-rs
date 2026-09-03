//! Nearest-rank percentiles with tail support — the workspace's one
//! percentile-of-record helper (promoted from five in-stack copies,
//! riir-ai Issue 861 T4 → the 5th-copy trigger → this module).
//!
//! The defect this shape exists to prevent: `sorted[(n * p) as usize]`
//! and `sorted[n * 99 / 100]` both land on `n - 1` — the **maximum** —
//! for every `n <= 1/(1-p)` (n ≤ 100 at p99, n ≤ 20 at p95, n ≤ 1000
//! at p999). Below that boundary the site reports one observation under
//! a percentile's name, and a `.min(len - 1)` clamp prevents a panic,
//! not a wrong statistic. A `p99 < budget` assert built on the naive
//! index becomes *stricter* (false RED), but it can no longer notice a
//! real regression: the tail is decided by a single sample.
//!
//! The quantity this helper returns alongside the value is **tail
//! support** = `n - idx`: the number of samples at or above the
//! reported rank (1 at n=100/p99, 2 at n=200, 10 at n=1000). Anything
//! under 10 is weak whether or not it is degenerate — print it, don't
//! hide it (the riir-ai Issue 853 T3 discipline).
//!
//! Contract:
//!
//! - `sorted` MUST be pre-sorted ascending by the caller (the helper
//!   indexes; it never compares, so it is generic over the element and
//!   float callers keep their own `total_cmp` sort — NaN placement is
//!   the caller's ordering decision, see
//!   [`crate::float_order`]). An unsorted slice returns a wrong value
//!   of the right type with no panic — sort at the call site.
//! - `p` is a fraction in `(0, 1]` (`0.99`, not `99`).
//! - Empty input: `debug_assert!` + a loud `assert!` (an empty sample
//!   set has no percentile, and a silent 0.0 would be a fabricated
//!   measurement).
//!
//! Promoted copies (all delegate here now): the canonical
//! `riir-games-shared/src/stats.rs` (`SeriesStats::p99` + Display
//! support), `riir-engine` `bench_336` (f32 + internal `total_cmp`
//! sort), `riir-rag` `p76_g2c_perf_benchmark` (u64),
//! `riir-games-civ` `bench_392_motivation_goat` (u64), `riir-gpu`
//! `ternary_dispatch_bench` (f64).
//!
//! Pure integer/floating arithmetic on `p` and `n` only — zero deps,
//! zero-cost-unless-invoked, ungated (the `float_order` precedent).

/// Nearest-rank percentile of a pre-sorted (ascending) slice, with the
/// rank's tail support.
///
/// Returns `(value, support)` where `support = n - idx` is the number
/// of samples at or above the returned rank: `1` means "this is the
/// maximum of n ≤ 1/(1-p) samples — the percentile name is decoration",
/// `≥ 10` means the tail has real statistical footing. `p` is a
/// fraction in `(0, 1]`; the rank is `ceil(p·n)` clamped to `1..=n`,
/// 1-based, so `p=1.0` is the maximum with support 1 and `p=0.5` at
/// even `n` is the upper median (the convention every promoted copy
/// already shipped — see `riir-games-shared/src/stats.rs`).
///
/// The rank is deliberately the **ceiling** form: the boundary trap
/// this module exists for is `floor(p·n) == n-1` for every
/// `n ≤ 1/(1-p)`, which reports the maximum under a percentile's name.
/// Callers asserting `p99 < budget` should gate on
/// `support >= 10` (⇒ `n ≥ 1000`) and fall back to an unconditional
/// p50/min budget when the tail is unavailable — the
/// `riir-mmorpg-examples` Issue 093 D1 shape.
#[inline]
pub fn nearest_rank<T: Copy>(sorted: &[T], p: f64) -> (T, usize) {
    let n = sorted.len();
    assert!(
        n > 0,
        "nearest_rank on an empty sample set — an empty sample has no percentile",
    );
    debug_assert!(
        p > 0.0 && p <= 1.0,
        "nearest_rank p is a fraction in (0, 1], got {p}",
    );
    let idx = ((p * n as f64).ceil() as usize).clamp(1, n) - 1;
    (sorted[idx], n - idx)
}

#[cfg(test)]
mod tests {
    use super::nearest_rank;

    #[test]
    fn ends_are_exact() {
        let v: Vec<f64> = (0..10).map(f64::from).collect();
        assert_eq!(nearest_rank(&v, 1.0), (9.0, 1), "p100 is the max");
        assert_eq!(
            nearest_rank(&v, 0.5),
            (4.0, 6),
            "p50 of 0..10: ceil(5)=5 → index 4 — the upper-median convention \
             the promoted copies shipped, support 6"
        );
        assert_eq!(
            nearest_rank(&v, 0.99),
            (9.0, 1),
            "p99 of 10 samples IS the max — support says so"
        );
    }

    #[test]
    fn support_grows_with_n_at_fixed_p() {
        // The whole point: at n=100 p99 already has support 2 (the ceiling
        // form leaves the max one rank EARLIER than the naive index —
        // ceil(0.99·100) = 99 exactly, since 0.99·100 is an exact integer
        // tie), and the tail reaches support 11 at n=1000. The max
        // (support 1) only survives for n ≤ 99 at p99, because
        // ceil(p·n) == n ⟺ n < 1/(1−p) and the p·n = n−1 exact tie lands
        // on the ceiling's low side.
        let v: Vec<f64> = (0..1000).map(f64::from).collect();
        let (val, sup) = nearest_rank(&v, 0.99);
        assert_eq!(val, 989.0, "ceil(0.99*1000)=990 → 0-based index 989");
        assert_eq!(sup, 11);
        let v100: Vec<f64> = (0..100).map(f64::from).collect();
        assert_eq!(
            nearest_rank(&v100, 0.99),
            (98.0, 2),
            "the exact-tie boundary: NOT the max (support 2) where the naive \
             floor index would have reported the max"
        );
        let v99: Vec<f64> = (0..99).map(f64::from).collect();
        assert_eq!(
            nearest_rank(&v99, 0.99),
            (98.0, 1),
            "below the tie boundary the ceiling form still returns the max — \
             support is what tells the caller"
        );
    }

    #[test]
    fn works_for_integer_elements() {
        // The u64 copies (riir-rag p76, civ bench_392) — no Ord bound
        // needed: the helper only indexes a pre-sorted slice.
        let v: Vec<u64> = vec![5, 3, 9, 1]; // caller pre-sorts → [1, 3, 5, 9]
        let sorted = [1u64, 3, 5, 9];
        let _ = v; // the contract note: sorting is the caller's job
        assert_eq!(nearest_rank(&sorted, 0.95), (9, 1));
    }

    #[test]
    fn empty_input_is_loud() {
        let result = std::panic::catch_unwind(|| nearest_rank::<f64>(&[], 0.99));
        assert!(result.is_err(), "empty input must assert, not fabricate 0.0");
    }

    #[test]
    fn generic_over_f32_with_caller_total_cmp_sort() {
        // The bench_336 shape: f32 + caller-side total_cmp (NaN placement
        // is the caller's ordering decision — float_order's territory).
        let mut v: Vec<f32> = vec![3.0, 1.0, 2.0];
        v.sort_unstable_by(f32::total_cmp);
        assert_eq!(nearest_rank(&v, 0.5), (2.0, 2));
    }
}
