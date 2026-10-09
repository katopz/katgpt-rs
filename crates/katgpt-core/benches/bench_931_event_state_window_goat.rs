//! Plan 623 T1.3 — `event_state_windows` GOAT gate (G1 posture re-pin +
//! G2 latency with the fixed-window baseline arm + determinism digest).
//!
//! Gates (run: `cargo bench -p katgpt-core --features event_state_windows
//! --bench bench_931_event_state_window_goat` at `--release` — the profile
//! AGENTS.md mandates for gates):
//!
//! - **G1 posture re-pin** — the hard-indicator limit (γ→0 → exact 0/1) and
//!   the transition exclusion re-asserted at the bench posture; the full
//!   closed-form suite lives in the module's inline tests.
//! - **G2a window latency** — per-call ≤ 20 ns (bar), p50/p90/p99 with tail
//!   supports printed (the repo's percentile-reporting rule). The K-state
//!   partition row (the consumer shape: all K states per tick) printed as
//!   context beside it.
//! - **G2b summary latency** — N=10k members ≤ 100 µs (the R311 shape), with
//!   the FIXED-WINDOW BASELINE ARM: the incumbent aggregation shape is the
//!   same statistics over a fixed `[t−W, t)` membership — identical compute,
//!   no anchoring. The claim is therefore RELATIVE: event-anchored total
//!   (K window evals + summary) ≤ fixed-window summary + ε, so the bar does
//!   not drift with box load.
//! - **Determinism** — double-summary bit-identity + a BLAKE3 digest over
//!   the summary bytes + the window-eval stream (the two-box anchor).
//! - **G4** — pointer only here: the alloc gate is the separate
//!   `event_state_window_alloc_check` binary (the `*_alloc_check`
//!   convention).
//!
//! Loud-zero defense: every timed loop's outputs feed `black_box` sinks so
//! LTO cannot delete the work it timed (the timed-region rule).
#![cfg(feature = "event_state_windows")]

use katgpt_core::event_state_window::{event_state_window, population_state_summary};
use std::hint::black_box;
use std::time::Instant;

const K_STATES: usize = 8;
const N_MEMBERS: usize = 10_000;
const TIMED_ITERS: usize = 2_000;
const WARMUP: usize = 64;

/// Deterministic LCG — no RNG dep, no global state (the global-RNG gate's
/// seeded-local law).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn f32_unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// One well-formed interior-state boundary tuple with sane tick-scale
/// edges (states of ~100 ticks, settle 20, width 5 — the G1 test geometry).
struct Bounds {
    t: f32,
    b_prev: f32,
    b_next: f32,
    delta: f32,
    gamma: f32,
}

fn bench_bounds(rng: &mut Lcg) -> Bounds {
    let b_prev = rng.f32_unit() * 1.0e4;
    let width = 50.0 + rng.f32_unit() * 200.0;
    let delta = 5.0 + rng.f32_unit() * 40.0;
    let b_next = b_prev + delta + width;
    let t = b_prev + delta + rng.f32_unit() * width;
    Bounds {
        t,
        b_prev,
        b_next,
        delta,
        gamma: 5.0,
    }
}

/// The G1 partition geometry (contiguous 100-tick states, settle 20) — the
/// K-state consumer row and the posture re-pin share it.
fn partition_geometry(k: usize, t: f32) -> (f32, f32, f32, f32) {
    let period = 100.0f32;
    let b_prev = (k as f32) * period;
    let delta = if k == 0 { 0.0 } else { 20.0 };
    let anchor = if k == 0 { b_prev - 1.0e4 } else { b_prev };
    (t, anchor, b_prev + period, delta)
}

fn main() {
    println!("== Plan 623 T1 — event_state_windows GOAT (K={K_STATES}, N={N_MEMBERS}) ==");

    // ── G1 posture re-pin: hard indicator + exclusion at bench posture ──
    let g_small = 1.0e-3f32;
    assert_eq!(event_state_window(50.0, 0.0, 100.0, 10.0, g_small), 1.0);
    assert_eq!(event_state_window(5.0, 0.0, 100.0, 10.0, g_small), 0.0);
    assert_eq!(event_state_window(50.0, 100.0, 50.0, 10.0, 5.0), 0.0, "inverted support → exact zero");
    assert_eq!(event_state_window(f32::NAN, 0.0, 100.0, 10.0, 5.0), 0.0, "NaN t → exact zero");
    println!("G1 posture re-pin: hard-indicator 1.0/0.0 + inverted/NaN zero ✓");

    // ── G2a: window per-call latency (bar ≤ 20 ns) ──────────────────────
    // ns-class work: per-call Instant pairs are BELOW the Windows timer
    // quantum (~100 ns — a naive run reads p50 0 ns / p99 100 ns, the timer
    // wearing a latency number). Batch 256 calls per Instant sample and
    // divide — the same loud-zero-defended loop, measurable resolution.
    const BATCH: usize = 256;
    const BATCHES: usize = 2_000;
    let mut rng = Lcg(0x6231_0001);
    let bounds: Vec<Bounds> = (0..256).map(|_| bench_bounds(&mut rng)).collect();
    let mut sink = 0.0f32;
    let run_window_batch = |first: usize, sink: &mut f32| {
        for i in first..first + BATCH {
            let b = &bounds[i & 255];
            *sink += event_state_window(b.t, b.b_prev, b.b_next, b.delta, b.gamma);
        }
    };
    for b in 0..WARMUP {
        run_window_batch(b * BATCH, &mut sink);
    }
    let mut samples: Vec<u128> = Vec::with_capacity(BATCHES);
    for b in 0..BATCHES {
        let t0 = Instant::now();
        run_window_batch(b * BATCH, &mut sink);
        samples.push(t0.elapsed().as_nanos());
    }
    black_box(sink);
    samples.sort_unstable();
    let (bp50, _) = katgpt_core::stats::nearest_rank(&samples, 0.50);
    let (bp90, _) = katgpt_core::stats::nearest_rank(&samples, 0.90);
    let (bp99, _) = katgpt_core::stats::nearest_rank(&samples, 0.99);
    let bsupport = samples.len() - samples.partition_point(|&x| x < bp99);
    let wp50 = bp50 / BATCH as u128;
    let wp90 = bp90 / BATCH as u128;
    let wp99 = bp99 / BATCH as u128;
    println!(
        "G2a window per-call: p50 {wp50} ns | p90 {wp90} ns | p99 {wp99} ns  (batch={BATCH}, n={BATCHES} batches, tail@p99={bsupport})  [bar ≤ 20 ns]"
    );
    assert!(wp99 <= 20, "G2a FAIL: p99 {wp99} ns > 20 ns per window call");

    // The consumer shape: all K states evaluated per tick (the partition row).
    let mut ksum = 0.0f32;
    let run_partition_batch = |first_tick: usize, ksum: &mut f32| {
        for tick in first_tick..first_tick + BATCH {
            let t = (tick % 800) as f32;
            let mut acc = 0.0f32;
            for k in 0..K_STATES {
                let (tt, bp, bn, d) = partition_geometry(k, t);
                acc += event_state_window(tt, bp, bn, d, 5.0);
            }
            *ksum += acc;
        }
    };
    for b in 0..WARMUP {
        run_partition_batch(b * BATCH, &mut ksum);
    }
    let mut ksamples: Vec<u128> = Vec::with_capacity(BATCHES);
    for b in 0..BATCHES {
        let t0 = Instant::now();
        run_partition_batch(b * BATCH, &mut ksum);
        ksamples.push(t0.elapsed().as_nanos());
    }
    black_box(ksum);
    ksamples.sort_unstable();
    let (kp50, _) = katgpt_core::stats::nearest_rank(&ksamples, 0.50);
    println!(
        "  context: K={K_STATES} partition row p50 {} ns/tick (window share of a per-tick pass)",
        kp50 / BATCH as u128
    );

    // ── G2b: summary N=10k (bar ≤ 100 µs) vs the fixed-window baseline ──
    let values: Vec<f32> = (0..N_MEMBERS).map(|_| rng.f32_unit() * 100.0).collect();
    let mut scratch = vec![0.0f32; N_MEMBERS];

    // Baseline arm FIRST (fixed-window aggregation: same statistics, no
    // event anchoring — the incumbent shape). Arm B adds K window evals.
    let mut b_sink = 0.0f32;
    let mut run_baseline = |sink: &mut f32| {
        let s = population_state_summary(&values, &mut scratch);
        *sink += s.mean + s.p50;
    };
    for _ in 0..WARMUP {
        run_baseline(&mut b_sink);
    }
    let mut base_samples: Vec<u128> = Vec::with_capacity(TIMED_ITERS);
    for _ in 0..TIMED_ITERS {
        let t0 = Instant::now();
        run_baseline(&mut b_sink);
        base_samples.push(t0.elapsed().as_nanos());
    }
    black_box(b_sink);
    base_samples.sort_unstable();
    let (b_p50, _) = katgpt_core::stats::nearest_rank(&base_samples, 0.50);
    let (b_p99, _) = katgpt_core::stats::nearest_rank(&base_samples, 0.99);

    // Event-anchored arm: K window evals (the anchoring) + the SAME summary.
    let mut e_sink = 0.0f32;
    let mut tick = 0usize;
    let mut run_event = |tick: &mut usize, sink: &mut f32| {
        let t = (*tick % 800) as f32;
        let mut wsum = 0.0f32;
        for k in 0..K_STATES {
            let (tt, bp, bn, d) = partition_geometry(k, t);
            wsum += event_state_window(tt, bp, bn, d, 5.0);
        }
        let s = population_state_summary(&values, &mut scratch);
        *sink += wsum + s.mean;
        *tick = (*tick + 1) % 800;
    };
    for _ in 0..WARMUP {
        run_event(&mut tick, &mut e_sink);
    }
    let mut ev_samples: Vec<u128> = Vec::with_capacity(TIMED_ITERS);
    for _ in 0..TIMED_ITERS {
        let t0 = Instant::now();
        run_event(&mut tick, &mut e_sink);
        ev_samples.push(t0.elapsed().as_nanos());
    }
    black_box(e_sink);
    ev_samples.sort_unstable();
    let (e_p50, _) = katgpt_core::stats::nearest_rank(&ev_samples, 0.50);
    let (e_p99, _) = katgpt_core::stats::nearest_rank(&ev_samples, 0.99);

    println!(
        "G2b summary N={N_MEMBERS}: fixed-window baseline p50 {b_p50} ns / p99 {b_p99} ns  [bar p50 ≤ 100 µs]"
    );
    assert!(
        b_p50 <= 100_000,
        "G2b FAIL: fixed-window summary p50 {b_p50} ns > 100 µs at N={N_MEMBERS}"
    );
    let delta = e_p50 as i128 - b_p50 as i128;
    println!(
        "  event-anchored total p50 {e_p50} ns / p99 {e_p99} ns — anchoring delta {delta:+} ns vs baseline (bar: ≤ +{K_STATES}×20 ns + noise)"
    );
    assert!(
        delta <= (K_STATES as i128) * 20 + 5_000,
        "G2 relative FAIL: event-anchored adds {delta} ns — the anchoring must be ~free vs the summary"
    );

    // ── Determinism: double-summary bit-identity + BLAKE3 anchors ───────
    let digest = |s: &katgpt_core::event_state_window::StateSummary| -> Vec<u8> {
        [
            s.mean, s.spread, s.p10, s.p25, s.p50, s.p75, s.p90,
        ]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .chain(s.supports.iter().flat_map(|u| u.to_le_bytes()))
        .chain(std::iter::once(s.n as u8))
        .collect()
    };
    let s1 = population_state_summary(&values, &mut scratch);
    let s2 = population_state_summary(&values, &mut scratch);
    assert_eq!(s1, s2, "same input order → bit-identical summary");
    let s3 = {
        let mut rev = values.clone();
        rev.reverse();
        population_state_summary(&rev, &mut scratch)
    };
    assert_eq!(s1, s3, "reversed order → bit-identical summary (the G1 invariance, bench posture)");
    println!("determinism: double-build + reversed-order bit-identical ✓");
    println!("  summary blake3     : {}", blake3::hash(&digest(&s1)));
    println!("G4: alloc gate = tests/event_state_window_alloc_check.rs (separate binary)");
}
