//! Plan 630 Phases 2+3 — `rls_depth_mem` G2 benches (T2.3 + T3.1).
//!
//! FOUR interleaved arms, one sampling loop — sequential A/B arms measure
//! the BOX (measured +5.2%/+21.7% drift thirty seconds apart), so every
//! comparison here alternates samples arm-by-arm and reads percentiles off
//! each arm's own sample vector:
//!
//! 1. **rls write+read** — `RlsMemoryState::write` + `read_into`, rank 8.
//! 2. **delta write+read** — the shipped `DeltaMemoryState` at matched
//!    rank (the Phase-3 T3.1 GOAT comparison; RLS does strictly more work
//!    — the P downdate).
//! 3. **windowed sweep** — E=4 stage reads through a [`WindowedReader`]
//!    (base + j correction terms, zero state copies).
//! 4. **materialized sweep** — E=4 stages via incremental full-state
//!    materialization (1 write + clone RANK² + read per stage).
//!
//! p50/p99 over 20k samples/arm: p50 support 10k, p99 support 200 — both
//! far from the n−1 max trap (a "p99" over ≤100 samples IS the max).
//! REPORT-ONLY: the Phase-3 verdict is decided from the run and recorded in
//! the plan, never asserted into the bench.
//!
//! Run: `cargo bench -p katgpt-core --features rls_depth_mem --bench
//! bench_rls_depth_mem` (cargo bench is release by construction).

#![cfg(feature = "rls_depth_mem")]

use katgpt_core::delta_mem::rls::{RlsCtx, RlsMemoryState, WindowedReader};
use katgpt_core::delta_mem::{DeltaMemoryConfig, DeltaMemoryState};
use std::hint::black_box;

const RANK: usize = 8;
const BASE_WRITES: usize = 64;
const E: usize = 4;
const ITERS_PER_ARM: usize = 20_000;
const BATCH: usize = 32;
const WARMUP: usize = 200;
const PAIRS_N: usize = 256;
const QUERIES_N: usize = 64;

/// Deterministic xorshift64* — benches cannot import the crate's regular
/// deps, and an unseeded global draw is the Issue-809 class.
struct Xs {
    s: u64,
}

impl Xs {
    fn next_f(&mut self) -> f32 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32 - 1.0
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

/// One interleaved pass over ALL arms — `arms[i]` sampled when
/// `iteration % n_arms == i`. Each timed sample runs the arm's closure
/// `BATCH` times (the Windows timer quantizes at ~100 ns — a single ~100 ns
/// op reports exactly 100.0 and p99 collapses onto p50; batching restores
/// ~3 ns effective resolution). Per-op ns = elapsed / BATCH. Returns
/// per-arm sample vectors; the closures fold a result into the return so
/// no arm can be dead-code-eliminated (the loud-zero law).
fn sample_interleaved(arms: &mut [&mut dyn FnMut() -> f64], iters_per_arm: usize) -> Vec<Vec<f64>> {
    let n = arms.len();
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(iters_per_arm); n];
    for i in 0..iters_per_arm * n {
        let a = i % n;
        let start = std::time::Instant::now();
        let mut r = 0.0f64;
        for _ in 0..BATCH {
            r += arms[a]();
        }
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000_000.0 / BATCH as f64;
        samples[a].push(elapsed + r * 0.0); // r folded, never dead
    }
    samples
}

/// Percentile of a sample vector, ns — nearest rank `ceil(p·n) − 1`; with
/// 20k samples p50 support = 10k and p99 support = 200 (percentile-index
/// law: the truncating form lands on the MAX at small n).
fn percentile_ns(mut samples: Vec<f64>, p: f64) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    let idx = ((n as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    samples[idx]
}

fn main() {
    // ── Fixtures (the caller-L2 law is the GENERATOR's contract — the
    //    envelope lesson; unnormalized keys destabilize the forward). ──
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
    let gen_query = |xs: &mut Xs| {
        let mut q = [0.0f32; RANK];
        for x in q.iter_mut() {
            *x = xs.next_f();
        }
        l2_normalize(&mut q);
        q
    };

    // RLS state: BASE_WRITES + E journaled writes (the Phase-2 fixtures).
    let mut st = RlsMemoryState::<RANK>::new();
    for _ in 0..BASE_WRITES {
        let (k, v) = gen_pair(&mut xs);
        st.write(&k, &v);
    }
    let base = st.clone();
    let mut pairs: Vec<([f32; RANK], [f32; RANK])> = Vec::with_capacity(E);
    let mut journal: Vec<RlsCtx<RANK>> = Vec::with_capacity(E);
    for _ in 0..E {
        let (k, v) = gen_pair(&mut xs);
        let mut ctx = RlsCtx::<RANK>::default();
        st.write_ctx(&k, &v, &mut ctx);
        journal.push(ctx);
        pairs.push((k, v));
    }
    let mut reader = WindowedReader::<RANK>::new(&base, E);
    for ctx in journal.iter().copied() {
        reader.push(ctx);
    }

    // Delta state at matched rank, warmed past the error-history growth
    // (Vec::new() → cap 512 allocates ~9 times; WARMUP covers it so the
    // timed window is steady-state for BOTH arms).
    let mut delta = DeltaMemoryState::new(DeltaMemoryConfig {
        rank: RANK,
        ..DeltaMemoryConfig::default()
    });
    for _ in 0..WARMUP {
        let (k, v) = gen_pair(&mut xs);
        delta.write(&k, &v);
    }

    // Cycling write/query fixtures (content is timing-irrelevant — neither
    // write has data-dependent control flow — but pre-generated so no
    // generator state rides the timed loop).
    let mut wr_pairs: Vec<([f32; RANK], [f32; RANK])> = Vec::with_capacity(PAIRS_N);
    for _ in 0..PAIRS_N {
        wr_pairs.push(gen_pair(&mut xs));
    }
    let mut queries: Vec<[f32; RANK]> = Vec::with_capacity(QUERIES_N);
    for _ in 0..QUERIES_N {
        queries.push(gen_query(&mut xs));
    }
    let mut out_r = [0.0f32; RANK];
    let mut out_d = [0.0f32; RANK];
    let mut out_a = [0.0f32; RANK];
    let mut out_b = [0.0f32; RANK];

    // ── Interleaved warmup (caches, branch predictors, the delta Vec). ──
    {
        let mut arms: [&mut dyn FnMut() -> f64; 4] = [
            &mut || {
                st.write(&wr_pairs[0].0, &wr_pairs[0].1);
                st.read_into(&queries[0], &mut out_r);
                out_r.iter().map(|&x| x as f64).sum::<f64>()
            },
            &mut || {
                delta.write(&wr_pairs[0].0, &wr_pairs[0].1);
                delta.read_into(&queries[0], &mut out_d);
                out_d.iter().map(|&x| x as f64).sum::<f64>()
            },
            &mut || {
                reader.read_into(&queries[0], &mut out_a);
                out_a.iter().map(|&x| x as f64).sum::<f64>()
            },
            &mut || {
                let mut stage = base.clone();
                for (k, v) in pairs.iter() {
                    stage.write(k, v);
                    let snapshot = stage.clone();
                    snapshot.read_into(&queries[0], &mut out_b);
                    black_box(&snapshot);
                }
                out_b.iter().map(|&x| x as f64).sum::<f64>()
            },
        ];
        for _ in 0..WARMUP {
            for arm in arms.iter_mut() {
                arm();
            }
        }
    }

    // ── The interleaved measurement (one loop, four arms). Each arm owns a
    //    rotating index; black_box(idx) makes the load addresses opaque so
    //    the fixture loads cannot be hoisted out of the timed region (the
    //    arguments-side loud-zero defence). ──
    let mut ridx = 0usize;
    let mut didx = 0usize;
    let mut qidx = 0usize;
    let mut arms: [&mut dyn FnMut() -> f64; 4] = [
        // 1. rls write+read
        &mut || {
            ridx = (ridx + 1) % PAIRS_N;
            qidx = (qidx + 1) % QUERIES_N;
            let i = black_box(ridx);
            let q = black_box(qidx);
            st.write(&wr_pairs[i].0, &wr_pairs[i].1);
            st.read_into(&queries[q], &mut out_r);
            out_r.iter().map(|&x| x as f64).sum::<f64>()
        },
        // 2. delta write+read (matched rank)
        &mut || {
            didx = (didx + 1) % PAIRS_N;
            let i = black_box(didx);
            delta.write(&wr_pairs[i].0, &wr_pairs[i].1);
            delta.read_into(&queries[0], &mut out_d);
            out_d.iter().map(|&x| x as f64).sum::<f64>()
        },
        // 3. windowed sweep (E stages through the reader)
        &mut || {
            let mut acc = 0.0f64;
            for j in 1..=E {
                reader.clear();
                for ctx in journal.iter().take(j).copied() {
                    reader.push(ctx);
                }
                reader.read_into(&queries[j % QUERIES_N], &mut out_a);
                acc += out_a.iter().map(|&x| x as f64).sum::<f64>();
            }
            acc
        },
        // 4. materialized sweep (1 write + clone RANK^2 + read per stage)
        &mut || {
            let mut acc = 0.0f64;
            let mut stage = base.clone();
            for (k, v) in pairs.iter() {
                stage.write(k, v);
                let snapshot = stage.clone();
                snapshot.read_into(&queries[0], &mut out_b);
                acc += out_b.iter().map(|&x| x as f64).sum::<f64>();
                black_box(&snapshot);
            }
            acc
        },
    ];
    let samples = sample_interleaved(&mut arms, ITERS_PER_ARM);

    let rls_p50 = percentile_ns(samples[0].clone(), 0.50);
    let rls_p99 = percentile_ns(samples[0].clone(), 0.99);
    let delta_p50 = percentile_ns(samples[1].clone(), 0.50);
    let delta_p99 = percentile_ns(samples[1].clone(), 0.99);
    let windowed_p50 = percentile_ns(samples[2].clone(), 0.50);
    let materialized_p50 = percentile_ns(samples[3].clone(), 0.50);

    println!(
        "rls_depth_mem G2/T3.1 + T2.3 @ RANK={RANK} E={E} — interleaved, {ITERS_PER_ARM} samples/arm (4090 box):"
    );
    println!("  T3.1 write+read  delta-rule p50 {delta_p50:8.1} p99 {delta_p99:8.1} ns");
    println!(
        "  T3.1 write+read  rls        p50 {rls_p50:8.1} p99 {rls_p99:8.1} ns  (ratio p50 {:.2}×, p99 {:.2}×)",
        rls_p50 / delta_p50,
        rls_p99 / delta_p99
    );
    println!(
        "  T2.3 E-stage sweep  windowed p50 {windowed_p50:8.1} ns  materialized p50 {materialized_p50:8.1} ns  (ratio {:.2}×)",
        windowed_p50 / materialized_p50
    );
    println!(
        "  (report-only — the Phase-3 verdict is recorded in .plans/630, never asserted here)"
    );
}
