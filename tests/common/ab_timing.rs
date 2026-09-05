//! Interleaved median-of-ratios A/B timing — the load-invariant treatment for
//! wall-clock gates (katgpt-rs Issue 723 **Class A**, T7).
//!
//! Include it the same way as `common/alloc_tracking.rs`:
//!
//! ```ignore
//! #[path = "common/ab_timing.rs"]
//! mod ab_timing;
//! ```
//!
//! ## Why a shared module and not a per-file loop
//!
//! Issue 723's first full-workspace *execution* found 8 targets whose verdict
//! the **box** decides rather than the code: each timed arm A to completion,
//! then arm B to completion, and asserted on the single ratio. Two sequential
//! 2M-iteration arms of the *same* work measured +5.2% and +21.7% thirty
//! seconds apart (Issue 723 T5), so on a box at load 8–16 a single ratio is
//! not a measurement of the primitive — it is a measurement of the scheduler.
//! A gate whose verdict the box decides is a gate that cannot run in CI, which
//! is the finding either way.
//!
//! Three defenses, all of them learned the expensive way and none of them
//! subsuming another:
//!
//! 1. **Interleaving.** `ROUNDS` back-to-back `(a-chunk, b-chunk)` pairs, one
//!    ratio per pair. Adjacent chunks share a load window, so a drift that
//!    moves both arms cancels in the ratio instead of landing entirely on the
//!    arm that happened to run second.
//! 2. **Median across pairs**, never the mean and never a single pair — a
//!    preemption spike lands in one round and the median discards it.
//! 3. **A loud zero.** rustc 1.98.1 + fat LTO eliminates inlined-callee work
//!    whose outer result is dead, *even through a `black_box` inside the
//!    callee* (Issue 723 T5: a direct call with a used result measured
//!    16.6 µs; `let _ = f()` over the same fn in the same binary read ~0).
//!    A vanished arm reads 0 ns and the ratio collapses to `NaN`/`inf`, which
//!    fails every comparison — Issue 723's own **Class A2**. Here an
//!    all-zero arm is a named FAIL that tells the reader the *instrument*
//!    broke, never a verdict about the code. Callers must still make their
//!    arms data-dependent and consume the sink; interleaving cannot resurrect
//!    work the optimiser deleted.
//!
//! Reporting discipline: [`AbRatio::report`] prints the per-round range next
//! to the median, because a median inside a 0.9–1.1 band and a median inside a
//! 0.3–3.0 band are not the same claim even when they are the same number.

#![allow(dead_code)]

use std::time::{Duration, Instant};

/// Outcome of an interleaved A/B run. `ratio = b / a`, so `a` is the
/// **baseline/denominator** arm and `b` the **candidate/numerator** arm:
/// a median of `1.20` reads "b costs 20% more than a".
pub struct AbRatio {
    /// One ratio per surviving round, sorted ascending.
    pub ratios: Vec<f64>,
    /// Median of `ratios` — the quantity a gate should assert on.
    pub median: f64,
    /// Total nanoseconds spent in the `a` arm across all rounds.
    pub a_total_ns: u128,
    /// Total nanoseconds spent in the `b` arm across all rounds.
    pub b_total_ns: u128,
    /// Rounds actually executed (surviving rounds is `ratios.len()`).
    pub rounds: usize,
    /// Iterations per arm per round.
    pub iters_per_round: usize,
}

impl AbRatio {
    /// `(median - 1) * 100` — the candidate's overhead over the baseline, in
    /// percent. Negative means the candidate is faster.
    pub fn overhead_pct(&self) -> f64 {
        (self.median - 1.0) * 100.0
    }

    /// Lowest per-round ratio (never `NaN`: an empty `ratios` is a hard FAIL
    /// inside [`ab_median_ratio`], so this is only called on a live result).
    pub fn min(&self) -> f64 {
        self.ratios[0]
    }

    /// Highest per-round ratio.
    pub fn max(&self) -> f64 {
        self.ratios[self.ratios.len() - 1]
    }

    /// Mean nanoseconds per `a` iteration across every round.
    pub fn a_ns_per_iter(&self) -> f64 {
        self.a_total_ns as f64 / (self.rounds * self.iters_per_round) as f64
    }

    /// Mean nanoseconds per `b` iteration across every round.
    pub fn b_ns_per_iter(&self) -> f64 {
        self.b_total_ns as f64 / (self.rounds * self.iters_per_round) as f64
    }

    /// Print the median WITH its per-round range. A median alone hides
    /// whether the rounds agreed, and the range is what says whether the box
    /// was quiet enough for the number to mean anything.
    pub fn report(&self, label: &str) {
        println!(
            "   {label}: {} x {} iters/arm — a {:.1} ns/iter, b {:.1} ns/iter",
            self.rounds,
            self.iters_per_round,
            self.a_ns_per_iter(),
            self.b_ns_per_iter(),
        );
        println!(
            "   {label}: ratio b/a median {:.4} (rounds {:.4} .. {:.4}, {} of {} survived) \
             = {:+.1}%",
            self.median,
            self.min(),
            self.max(),
            self.ratios.len(),
            self.rounds,
            self.overhead_pct(),
        );
    }
}

/// Interleaved median-of-ratios over two single-iteration closures.
///
/// Runs `warmup` iterations of each arm, then `rounds` back-to-back
/// `(a × iters, b × iters)` pairs, taking one `b/a` ratio per pair and
/// returning their median. Each closure receives a monotonically increasing
/// iteration index so it can vary its input — constant input is what lets the
/// optimiser hoist the whole arm out of the loop (Issue 723 Class A2).
///
/// # Panics
///
/// - `rounds == 0` or `iters == 0` (nothing would be measured).
/// - Every round produced a zero-nanosecond arm: the work was eliminated or
///   the chunk is below timer resolution. That is an instrument failure and
///   is reported as one — raise `iters`, make the arm data-dependent, and
///   consume its sink.
pub fn ab_median_ratio<A, B>(
    rounds: usize,
    iters: usize,
    warmup: usize,
    mut a: A,
    mut b: B,
) -> AbRatio
where
    A: FnMut(usize),
    B: FnMut(usize),
{
    assert!(rounds > 0, "ab_median_ratio needs rounds > 0");
    assert!(iters > 0, "ab_median_ratio needs iters > 0");

    for i in 0..warmup {
        a(i);
        b(i);
    }

    let mut ratios: Vec<f64> = Vec::with_capacity(rounds);
    let mut a_total_ns = 0u128;
    let mut b_total_ns = 0u128;

    for r in 0..rounds {
        let base = (r + 1) * iters;

        let t = Instant::now();
        for i in 0..iters {
            a(base + i);
        }
        let a_ns = t.elapsed().as_nanos();

        let t = Instant::now();
        for i in 0..iters {
            b(base + i);
        }
        let b_ns = t.elapsed().as_nanos();

        a_total_ns += a_ns;
        b_total_ns += b_ns;
        if a_ns > 0 && b_ns > 0 {
            ratios.push(b_ns as f64 / a_ns as f64);
        }
    }

    assert!(
        !ratios.is_empty(),
        "A/B instrument failure: every one of {rounds} rounds measured 0 ns in an arm \
         over {iters} iters (work eliminated by the optimiser, or the chunk is below \
         timer resolution) — fix the harness, do not read a verdict out of it",
    );

    ratios.sort_by(|x, y| x.total_cmp(y));
    let median = ratios[ratios.len() / 2];

    AbRatio {
        ratios,
        median,
        a_total_ns,
        b_total_ns,
        rounds,
        iters_per_round: iters,
    }
}

/// Best-of-N wall time in microseconds over a closure that **returns its own
/// timed [`Duration`]**.
///
/// For an *absolute* latency budget there is no second arm to ratio against,
/// so the load-invariant quantity is the **minimum**: contention can only ever
/// add time, so the smallest of N samples is the closest observation of the
/// machine's true cost and the only one a busy box cannot inflate. A p50 over
/// 5 samples on a box at load 8–16 is a measurement of the scheduler.
///
/// The closure owns the clock rather than the harness bracketing it, because
/// the common case needs **per-iteration setup excluded from the timed
/// region** (restoring operands that the primitive scales in place, re-zeroing
/// an output buffer) and setup + work as two separate closures cannot both
/// hold `&mut` to the same state. Returning the `Duration` keeps the borrow in
/// one closure:
///
/// ```ignore
/// let us = best_of_us(3, 20, || {
///     a.copy_from_slice(&a0);          // setup — not timed
///     let t = Instant::now();
///     work(&mut a);
///     t.elapsed()                       // timed region only
/// });
/// ```
///
/// # Panics
///
/// `iters == 0`, or every timed call measured 0 ns (see [`ab_median_ratio`]).
pub fn best_of_us<F>(warmup: usize, iters: usize, mut timed: F) -> f64
where
    F: FnMut() -> Duration,
{
    assert!(iters > 0, "best_of_us needs iters > 0");

    for _ in 0..warmup {
        timed();
    }

    let mut best_ns = u128::MAX;
    for _ in 0..iters {
        let ns = timed().as_nanos();
        if ns < best_ns {
            best_ns = ns;
        }
    }

    assert!(
        best_ns > 0,
        "timing instrument failure: every one of {iters} timed calls measured 0 ns \
         (work eliminated by the optimiser, or below timer resolution) — raise the \
         work per call and consume the result",
    );

    best_ns as f64 / 1000.0
}
