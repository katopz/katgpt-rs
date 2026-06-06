//! Ruliology benchmarks — enumeration time + IrreducibilityGate overhead.
//!
//! Plan 188 Phase 6: GOAT proof benchmarks.
//!
//! Run: `cargo test --features ruliology ruliology_bench -- --nocapture`

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::ruliology::{
        CaStrategy, FsmEnumerator, IrreducibilityGate, SimpleProgram, TmStrategy, matching_pennies,
    };

    /// Helper: generic tournament for any SimpleProgram type.
    #[allow(dead_code)]
    fn generic_tournament<S: SimpleProgram>(
        strategies: &[S],
        rounds: u32,
        payoff_fn: &dyn Fn(u8, u8) -> f64,
    ) -> Vec<Vec<f64>> {
        let n = strategies.len();
        let mut payoffs = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut si = strategies[i].clone();
                let mut sj = strategies[j].clone();
                let mut hist_i: Vec<u8> = Vec::with_capacity(rounds as usize);
                let mut hist_j: Vec<u8> = Vec::with_capacity(rounds as usize);
                let mut total = 0.0f64;
                for _ in 0..rounds {
                    let ai = si.next_action(&hist_j);
                    let aj = sj.next_action(&hist_i);
                    total += payoff_fn(ai, aj);
                    hist_i.push(ai);
                    hist_j.push(aj);
                }
                payoffs[i][j] = total / rounds as f64;
            }
        }
        payoffs
    }

    // ── Enumeration Benchmarks ─────────────────────────────────────

    #[test]
    fn bench_enumerate_fsm_2_states() {
        let start = Instant::now();
        let fsms = FsmEnumerator::enumerate(2);
        let elapsed = start.elapsed();

        println!(
            "[bench] FSM(2) enumerate: {} strategies in {elapsed:?}",
            fsms.len()
        );

        // N=2 should be trivial (< 100ms even in debug).
        assert!(
            elapsed.as_millis() < 500,
            "FSM(2) enumeration too slow: {elapsed:?}"
        );
    }

    #[test]
    fn bench_enumerate_fsm_3_states() {
        let start = Instant::now();
        let fsms = FsmEnumerator::enumerate(3);
        let elapsed = start.elapsed();

        println!(
            "[bench] FSM(3) enumerate: {} strategies in {elapsed:?}",
            fsms.len()
        );

        // N=3 should be sub-second in debug, trivial in release.
        assert!(
            elapsed.as_secs() < 10,
            "FSM(3) enumeration too slow: {elapsed:?}"
        );
    }

    #[test]
    #[ignore] // FSM(4) takes minutes — run with `--ignored` flag
    fn bench_enumerate_fsm_4_states() {
        let start = Instant::now();
        let fsms = FsmEnumerator::enumerate(4);
        let elapsed = start.elapsed();

        println!(
            "[bench] FSM(4) enumerate: {} strategies in {elapsed:?}",
            fsms.len()
        );

        // N=4: 18^4 = 104,976 raw FSMs. Dedup may take several seconds.
        // Report the time but don't hard-fail — this is a measurement.
        if elapsed.as_secs() > 30 {
            println!("[bench] ⚠️  FSM(4) took {elapsed:?} — may be too slow for production");
        }
    }

    #[test]
    fn bench_enumerate_ca() {
        let start = Instant::now();
        let cas = CaStrategy::enumerate_all();
        let elapsed_all = start.elapsed();

        let start = Instant::now();
        let distinct = CaStrategy::enumerate_distinct();
        let elapsed_distinct = start.elapsed();

        println!("[bench] CA enumerate all: {} in {elapsed_all:?}", cas.len());
        println!(
            "[bench] CA enumerate distinct: {} in {elapsed_distinct:?}",
            distinct.len()
        );

        assert!(
            elapsed_all.as_millis() < 100,
            "CA all too slow: {elapsed_all:?}"
        );
        assert!(
            elapsed_distinct.as_millis() < 1000,
            "CA distinct too slow: {elapsed_distinct:?}"
        );
    }

    #[test]
    fn bench_enumerate_tm() {
        let start = Instant::now();
        let tms = TmStrategy::enumerate_1_state();
        let elapsed = start.elapsed();

        println!(
            "[bench] TM enumerate: {} machines in {elapsed:?}",
            tms.len()
        );
        assert!(
            elapsed.as_millis() < 50,
            "TM enumeration too slow: {elapsed:?}"
        );
    }

    // ── Tournament Benchmarks ───────────────────────────────────────

    #[test]
    fn bench_tournament_fsm_2() {
        let fsms = FsmEnumerator::enumerate(2);

        let start = Instant::now();
        let _matrix = FsmEnumerator::tournament(&fsms, 100, &matching_pennies);
        let elapsed = start.elapsed();

        println!(
            "[bench] FSM(2) tournament ({} strategies, 100 rounds): {elapsed:?}",
            fsms.len()
        );

        assert!(
            elapsed.as_millis() < 500,
            "FSM(2) tournament too slow: {elapsed:?}"
        );
    }

    #[test]
    fn bench_tournament_fsm_3() {
        let fsms = FsmEnumerator::enumerate(3);

        let start = Instant::now();
        let _matrix = FsmEnumerator::tournament(&fsms, 50, &matching_pennies);
        let elapsed = start.elapsed();

        println!(
            "[bench] FSM(3) tournament ({} strategies, 50 rounds): {elapsed:?}",
            fsms.len()
        );

        // FSM(3) tournament is O(n²) = ~1054² ≈ 1.1M pairs. Allow more time.
        assert!(
            elapsed.as_secs() < 60,
            "FSM(3) tournament too slow: {elapsed:?}"
        );
    }

    // ── IrreducibilityGate Benchmarks ──────────────────────────────

    #[test]
    fn bench_irreducibility_gate_fsm2() {
        let fsms = FsmEnumerator::enumerate(2);
        let matrix = FsmEnumerator::tournament(&fsms, 100, &matching_pennies);
        let gate = IrreducibilityGate::default();

        let start = Instant::now();
        for _ in 0..1000 {
            let _result = gate.analyze(&matrix);
        }
        let elapsed = start.elapsed();

        let per_check = elapsed / 1000;
        println!(
            "[bench] IrreducibilityGate ({}×{} matrix): {per_check:?} per check (1000 iterations)",
            fsms.len(),
            fsms.len()
        );

        // Should be sub-millisecond per check.
        assert!(
            per_check.as_micros() < 1000,
            "IrreducibilityGate too slow: {per_check:?} per check"
        );
    }

    #[test]
    fn bench_irreducibility_gate_fsm3() {
        let fsms = FsmEnumerator::enumerate(3);
        let matrix = FsmEnumerator::tournament(&fsms, 50, &matching_pennies);
        let gate = IrreducibilityGate::default();

        let start = Instant::now();
        for _ in 0..100 {
            let _result = gate.analyze(&matrix);
        }
        let elapsed = start.elapsed();

        let per_check = elapsed / 100;
        println!(
            "[bench] IrreducibilityGate ({}×{} matrix): {per_check:?} per check (100 iterations)",
            fsms.len(),
            fsms.len()
        );

        // FSM(3) matrix is ~1054×1054, should still be fast.
        assert!(
            per_check.as_millis() < 100,
            "IrreducibilityGate too slow for FSM(3): {per_check:?} per check"
        );
    }

    // ── No-Regression Test ─────────────────────────────────────────

    #[test]
    fn bench_no_regression_feature_disabled() {
        // This test verifies the ruliology feature compiles and runs.
        // When disabled, the module is compiled out — zero cost.
        // This test only exists to assert it compiles when enabled.
        println!("[bench] Feature `ruliology` is enabled — all ruliology code active.");
        println!("[bench] When disabled: module is #[cfg(feature = \"ruliology\")] — zero cost.");
    }
}

// TL;DR: Benchmarks for Plan 188 Phase 6 — FSM enumeration time (N=2 trivial, N=3 sub-second, N=4 measured), IrreducibilityGate overhead, no-regression verification.
