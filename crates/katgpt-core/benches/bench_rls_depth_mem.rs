//! Plan 630 Phase 2 — `rls_depth_mem` G2 micro-bench (T2.3).
//!
//! Windowed-traffic vs materializing E intermediate states at E = 4: one
//! lane reads each of the E stages through a [`WindowedReader`]
//! (base read + j correction terms, ZERO state copies); the other lane
//! materializes each stage's full state on a fresh clone (clone RANK² +
//! replay the write + read RANK²) and reads it. Per single-stage read the
//! materialized form can win (one RANK² dot vs base + corrections); the
//! window exists for the SWEEP — all E stages' reads without materializing
//! E states. Both numbers are the record; the Phase-3 T3.1 GOAT is the
//! decision gate.
//!
//! Run: `cargo bench -p katgpt-core --features rls_depth_mem --bench
//! bench_rls_depth_mem` (cargo bench is release by construction).

#![cfg(feature = "rls_depth_mem")]

use katgpt_core::delta_mem::rls::{RlsCtx, RlsMemoryState, WindowedReader};
use std::hint::black_box;

const RANK: usize = 8;
const BASE_WRITES: usize = 64;
const E: usize = 4;
const ITERS: usize = 20_000;

/// Deterministic xorshift64* — benches cannot import the crate's regular
/// deps, and an unseeded global draw is the Issue-809 class.
struct Xs {
    s: u64,
}

impl Xs {
    fn next_u(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f(&mut self) -> f32 {
        (self.next_u() >> 40) as f32 / (1u64 << 24) as f32 - 1.0
    }
}

fn l2_normalize(v: &mut [f32]) {
    let mut s = 0.0f32;
    for &x in v.iter() {
        s += x * x;
    }
    let n = s.sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// Median over `iters` runs, ns. The closure returns an f64 the timer folds
/// into the sample so the sweep cannot be dead-code-eliminated (the
/// absent-work class — the timed-region loud-zero law).
fn time_median_ns(f: &mut dyn FnMut() -> f64, iters: usize) -> f64 {
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = std::time::Instant::now();
        let r = f();
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        times.push((elapsed, r));
    }
    times.sort_by(|a, b| a.0.total_cmp(&b.0));
    times[times.len() / 2].0
}

fn main() {
    // Deterministic stream: BASE_WRITES + E (key, value) pairs + one query.
    // The caller-L2 law is the GENERATOR's contract (the envelope lesson).
    let mut xs = Xs {
        s: 0x9E37_79B9_7F4A_7C15,
    };
    let gen_pair = |xs: &mut Xs| {
        let mut k = [0.0f32; RANK];
        for x in k.iter_mut() {
            *x = xs.next_f();
        }
        l2_normalize(&mut k);
        let mut v = [0.0f32; RANK];
        for x in v.iter_mut() {
            *x = 0.5 * xs.next_f();
        }
        (k, v)
    };
    let mut st = RlsMemoryState::<RANK>::new();
    for _ in 0..BASE_WRITES {
        let (k, v) = gen_pair(&mut xs);
        st.write(&k, &v);
    }
    let base = st.clone();
    // E journaled writes on the live state; the RAW pairs are kept so the
    // materialized lane can replay the same writes on its clones.
    let mut pairs: Vec<([f32; RANK], [f32; RANK])> = Vec::with_capacity(E);
    let mut journal: Vec<RlsCtx<RANK>> = Vec::with_capacity(E);
    for _ in 0..E {
        let (k, v) = gen_pair(&mut xs);
        let mut ctx = RlsCtx::<RANK>::default();
        st.write_ctx(&k, &v, &mut ctx);
        journal.push(ctx);
        pairs.push((k, v));
    }
    let mut query = [0.0f32; RANK];
    for x in query.iter_mut() {
        *x = xs.next_f();
    }
    l2_normalize(&mut query);
    // Pre-built reader (construction allocates once — outside the measured
    // section; read_into itself is the zero-alloc hot path).
    let mut reader = WindowedReader::<RANK>::new(&base, E);
    for ctx in journal.iter().copied() {
        reader.push(ctx);
    }
    let mut out_a = [0.0f32; RANK];
    let mut out_b = [0.0f32; RANK];

    let windowed = time_median_ns(
        &mut || {
            let mut acc = 0.0f64;
            for j in 1..=E {
                // Stage-j read through a window of j corrections: the caller
                // narrows the window instead of holding stage states.
                reader.clear();
                for ctx in journal.iter().take(j).copied() {
                    reader.push(ctx);
                }
                reader.read_into(black_box(&query), &mut out_a);
                acc += out_a.iter().map(|&x| x as f64).sum::<f64>();
            }
            acc
        },
        ITERS,
    );

    let materialized = time_median_ns(
        &mut || {
            let mut acc = 0.0f64;
            // Incremental materialization: each arriving write costs one
            // full-state clone (the traffic the window avoids), then the
            // stage read. 1 write + 1 clone + 1 read per stage.
            let mut stage = base.clone();
            for (k, v) in pairs.iter() {
                stage.write(k, v);
                let snapshot = stage.clone();
                snapshot.read_into(black_box(&query), &mut out_b);
                acc += out_b.iter().map(|&x| x as f64).sum::<f64>();
                black_box(&snapshot);
            }
            acc
        },
        ITERS,
    );

    println!("rls_depth_mem windowed-vs-materialized @ RANK={RANK} E={E} (median of {ITERS}):");
    println!("  windowed     {windowed:9.1} ns / E-stage sweep");
    println!("  materialized {materialized:9.1} ns / E-stage sweep");
    println!(
        "  ratio windowed/materialized = {:.3}",
        windowed / materialized
    );
    println!(
        "  (materialized/stage = 1 write + clone RANK^2 + read RANK^2; windowed/stage = base read + j correction terms, 0 copies)"
    );
}
