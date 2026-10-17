//! Plan 617 T3.1 — the LoopCD fixture NON-DEGENERACY gate.
//!
//! The Bench-847/850 lesson, mechanized: a saturated toy makes every
//! guided-arm comparison vacuous (850 passed over zero close decisions), and
//! a monotone-contraction fixture makes `z_R − z_k` parallel to `z_R` so
//! greedy argmax cannot move for a STRUCTURAL reason. This gate commits the
//! fixture's non-degeneracy evidence BEFORE any guided arm runs (the
//! ordering is the plan's verdict-review note; the GOAT bench record states
//! it):
//!
//! - (a) a real depth→accuracy slope exists (accuracy monotone nondecreasing
//!   across depths 1..=MAX_LOOPS, with a minimum full-range gain);
//! - (b) the close-decision stratum is non-empty with a minimum count —
//!   cases whose final two-answer margin is below [`CLOSE_MARGIN_EPS`];
//! - (c) the disagreement floor: a minimum count where
//!   `argmax(z_1) ≠ argmax(z_MAX_LOOPS)` (the paper's own disagreement-rate
//!   metric at k=1);
//! - plus the sanity frame: finite logits everywhere, byte-deterministic
//!   re-runs, no tie in any case's ground truth (by construction), and the
//!   query row never wins the decode.
//!
//! Run: `cargo test --release --features lt2_looped --test loopcd_fixture_gate`
//! (unguided only — the fixture is frozen before the first guided arm).

#![cfg(feature = "lt2_looped")]

#[path = "common/loopcd_fixture.rs"]
mod loopcd_fixture;

use loopcd_fixture::{LoopCdFixture, MAX_LOOPS, all_cases, answer_margin, argmax};

/// Close-decision threshold on the final two-answer margin. The stratum the
/// GOAT's T3.3(c) leg reads; a saturated fixture would empty it.
const CLOSE_MARGIN_EPS: f32 = 0.5;
/// Minimum close-stratum count (b).
const MIN_CLOSE: usize = 5;
/// Minimum disagreement count (c).
const MIN_DISAGREE: usize = 10;
/// Minimum full-range accuracy gain (a): acc(MAX_LOOPS) − acc(1).
const MIN_SLOPE: f64 = 0.25;

#[test]
fn fixture_is_non_degenerate() {
    let fixture = LoopCdFixture::new();
    let cases = all_cases();
    assert!(
        cases.len() >= 40,
        "case space unexpectedly small: {}",
        cases.len()
    );

    // Per-depth accuracy + per-case diagnostics.
    let mut acc = [0.0f64; MAX_LOOPS];
    let mut close_at_max = 0usize;
    let mut disagree = 0usize;
    let mut per_case: Vec<(usize, i32, usize, f32, usize, usize)> = Vec::new();
    let mut logits_at_max: Vec<f32> = Vec::new();

    for case in &cases {
        let mut amax = [0usize; MAX_LOOPS];
        for (d, logits) in (0..MAX_LOOPS).map(|d| (d, fixture.run_case(case, d + 1))) {
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "non-finite logits at depth {} for {:?}",
                d + 1,
                case.prefix
            );
            amax[d] = argmax(&logits);
            acc[d] += f64::from(amax[d] == case.answer);
            if d + 1 == MAX_LOOPS {
                logits_at_max = logits;
            }
        }
        // Determinism: the same case at the same depth re-runs byte-identical.
        let again = fixture.run_case(case, MAX_LOOPS);
        assert_eq!(
            again, logits_at_max,
            "fixture must be byte-deterministic for {:?}",
            case.prefix
        );

        let margin_max = answer_margin(&logits_at_max);
        if margin_max.abs() < CLOSE_MARGIN_EPS {
            close_at_max += 1;
        }
        if amax[0] != amax[MAX_LOOPS - 1] {
            disagree += 1;
        }
        per_case.push((
            case.prefix.len(),
            case.imbalance,
            amax[0],
            margin_max,
            amax[MAX_LOOPS - 1],
            usize::from(amax[MAX_LOOPS - 1] == case.answer),
        ));
    }

    let n = cases.len() as f64;
    for a in &mut acc {
        *a /= n;
    }

    // Diagnostics first (the strata table tuning reads) — assertions after,
    // so a failing floor still prints the evidence it failed on.
    println!("== LoopCD fixture non-degeneracy table ==");
    println!(
        "cases={} close(max)={close_at_max} disagree(1 vs {MAX_LOOPS})={disagree}",
        cases.len()
    );
    println!("accuracy by depth: {:?}", acc);
    println!("len imb  amax(1) margin(max) amax(max) correct(max)");
    for row in &per_case {
        println!(
            "{:>3} {:>4} {:>7} {:>12.4} {:>9} {:>11}",
            row.0, row.1, row.2, row.3, row.4, row.5
        );
    }

    // ── (a) the depth→accuracy slope ──
    for d in 1..MAX_LOOPS {
        assert!(
            acc[d] >= acc[d - 1] - 1e-9,
            "accuracy must be monotone nondecreasing: acc({}) = {:.4} < acc({}) = {:.4}",
            d + 1,
            acc[d],
            d,
            acc[d - 1]
        );
    }
    let slope = acc[MAX_LOOPS - 1] - acc[0];
    assert!(
        slope >= MIN_SLOPE,
        "depth→accuracy slope {:.4} < floor {MIN_SLOPE} — the fixture is saturated \
         and every guided-arm comparison would be vacuous",
        slope
    );

    // ── (b) the close-decision stratum ──
    assert!(
        close_at_max >= MIN_CLOSE,
        "close stratum {close_at_max} < floor {MIN_CLOSE} — a saturated toy has no \
         close decisions and the T3.3(c) leg would pass over zero rows (Bench 850's \
         exact failure mode)"
    );

    // ── (c) the disagreement floor ──
    assert!(
        disagree >= MIN_DISAGREE,
        "disagreement {disagree} < floor {MIN_DISAGREE} — without it z_R − z_k is \
         parallel to z_R and greedy argmax is unchanged by construction"
    );

    // The query row never wins a decode.
    for case in &cases {
        let logits_at_max = fixture.run_case(case, MAX_LOOPS);
        let amax_max = argmax(&logits_at_max);
        assert_ne!(
            amax_max,
            loopcd_fixture::QUERY,
            "query row won the decode for {:?}",
            case.prefix
        );
    }
}
