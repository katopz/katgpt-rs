//! GOAT gate — permanent attention sinks + a bounded KV window
//! (Issue 841, Research 571 §B-8).
//!
//! ```bash
//! cargo bench -p katgpt-core --features usage_rate_eviction \
//!     --bench bench_841_kv_sink_window_goat
//! # G4 needs an allocator in the profile:
//! cargo bench -p katgpt-core --features usage_rate_eviction,alloc_tracking \
//!     --bench bench_841_kv_sink_window_goat
//! ```
//!
//! # What is being gated
//!
//! The product claim is **not** "eviction is faster". It is
//! `live_slots <= n_sink + window` for every step and every sequence length,
//! which makes decode-time KV memory a function of the POLICY rather than of
//! how long the conversation has run. The latency result is a COROLLARY of
//! that: eviction planning is O(live), so without a ceiling its per-step cost
//! grows with the sequence and with one it is flat. G2 measures the
//! corollary because it is the part a box can invalidate; G1 measures the
//! claim.
//!
//! **G1 the ceiling** — a 20 000-step adversarial admission sequence whose
//! scorer ranks the sinks WORST, asserting the bound at every step and that
//! no admitted sink is ever evicted. Plus: the steady state must sit AT the
//! ceiling, because a policy that evicted everything would satisfy a bound.
//!
//! **G2 flatness** — per-step planning cost at growing sequence lengths,
//! bounded vs unbounded. Read as a SHAPE (does the bounded arm stay flat),
//! not as a speedup: the two arms do different amounts of work by
//! construction, which is the finding.
//!
//! **G3 reduction** — `SinkWindowPolicy::UNBOUNDED` must reproduce
//! `kv_eviction::select_evict_into` INDEX FOR INDEX. An "off" switch that is
//! a second code path is not an off switch.
//!
//! **G4 alloc-free** — zero allocations across a steady-state eviction loop.
//!
//! **G5 fidelity honesty** — the module must report `Lossy` when it cannot
//! prove otherwise. This gate exists because the *promotion* decision rests
//! on it, and a permissive default here would exempt a real lossy surface
//! from the Plan-585 runaway gate.

use std::hint::black_box;
use std::time::{Duration, Instant};

use katgpt_core::kv_eviction::select_evict_into;
use katgpt_core::kv_sink_window::{
    SinkWindowPolicy, SlotClass, WindowFidelity, select_evict_windowed, sink_pin_mask_into,
};

#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
#[global_allocator]
static BENCH_ALLOC: katgpt_core::alloc::TrackingAllocator = katgpt_core::alloc::TrackingAllocator;

const N_SINK: usize = 4;
const WINDOW: usize = 256;

/// Best-of-`iters` minimum nanoseconds. The MINIMUM, not the mean: a loaded
/// box can only make a round slower, so the minimum is the load-robust
/// statistic (AGENTS.md § *A latency number without its BOX STATE*).
fn best_of_ns<F: FnMut() -> Duration>(warmup: usize, iters: usize, mut timed: F) -> f64 {
    for _ in 0..warmup {
        timed();
    }
    let mut best = u128::MAX;
    for _ in 0..iters {
        best = best.min(timed().as_nanos());
    }
    best as f64
}

/// A deliberately hostile scorer: the sinks look WORST and old rows look
/// best, so anything ordering by score alone evicts exactly the wrong rows.
fn hostile_scores(live: &[u64], out: &mut Vec<f32>) {
    out.clear();
    out.extend(live.iter().map(|&pos| match (pos as usize) < N_SINK {
        true => -1.0,
        false => 1e6 - pos as f32,
    }));
}

fn g1_ceiling(failures: &mut Vec<String>) {
    let p = SinkWindowPolicy::new(N_SINK, WINDOW);
    let mut live: Vec<u64> = Vec::new();
    let mut scores = Vec::new();
    let mut evict = Vec::new();
    let steps = 20_000u64;

    let mut worst_live = 0usize;
    let mut breach = 0usize;
    let mut sink_lost = 0usize;

    for step in 0..steps {
        live.push(step);
        hostile_scores(&live, &mut scores);
        let k = p.evictions_needed(live.len());
        select_evict_windowed(&p, &live, step, &scores, k, &mut evict);
        evict.sort_unstable();
        for &i in evict.iter().rev() {
            live.remove(i);
        }
        worst_live = worst_live.max(live.len());
        if live.len() > p.capacity() {
            breach += 1;
        }
        for s in 0..(N_SINK as u64).min(step + 1) {
            if !live.contains(&s) {
                sink_lost += 1;
            }
        }
    }

    let at_ceiling = live.len() == p.capacity();
    let ok = breach == 0 && sink_lost == 0 && at_ceiling;
    println!(
        "G1 ceiling             {steps} steps, peak live {worst_live} vs ceiling {} \
         ({breach} breach, {sink_lost} sink loss, steady state AT ceiling: {at_ceiling}) → {}",
        p.capacity(),
        verdict(ok)
    );
    if !ok {
        failures.push(format!(
            "G1: {breach} ceiling breach(es), {sink_lost} sink loss(es), at_ceiling {at_ceiling}"
        ));
    }

    // The memory statement the ceiling buys, as a ratio rather than a bound.
    let unbounded_rows = steps as usize;
    println!(
        "G1 memory              after {steps} tokens: {} live rows vs {unbounded_rows} unbounded \
         — {:.0}x fewer, and CONSTANT in the sequence length",
        live.len(),
        unbounded_rows as f64 / live.len() as f64
    );
}

fn g2_flatness(failures: &mut Vec<String>) {
    println!("G2 per-step planning cost as the sequence grows");
    let bounded = SinkWindowPolicy::new(N_SINK, WINDOW);
    let unbounded = SinkWindowPolicy::UNBOUNDED;
    let mut scores = Vec::new();
    let mut evict = Vec::new();
    let mut first: Option<f64> = None;
    let mut last = 0.0f64;

    for n in [1_000usize, 4_000, 16_000] {
        // Unbounded: the live set IS the sequence.
        let live_u: Vec<u64> = (0..n as u64).collect();
        hostile_scores(&live_u, &mut scores);
        let cur = n as u64 - 1;
        let ns_u = best_of_ns(2, 12, || {
            let t = Instant::now();
            select_evict_windowed(&unbounded, &live_u, cur, &scores, 1, &mut evict);
            let d = t.elapsed();
            black_box(&evict);
            d
        });

        // Bounded: the live set is the ceiling, whatever n is.
        let live_b: Vec<u64> = ((n - bounded.capacity()) as u64..n as u64).collect();
        let mut sb = Vec::new();
        hostile_scores(&live_b, &mut sb);
        let ns_b = best_of_ns(2, 12, || {
            let t = Instant::now();
            select_evict_windowed(&bounded, &live_b, cur, &sb, 1, &mut evict);
            let d = t.elapsed();
            black_box(&evict);
            d
        });

        first.get_or_insert(ns_b);
        last = ns_b;
        println!(
            "   seq {n:>6}  unbounded {ns_u:>10.0} ns/step   bounded {ns_b:>8.0} ns/step \
             → {:.0}x",
            ns_u / ns_b
        );
    }

    // The gate is FLATNESS of the bounded arm across a 16x sequence growth,
    // not a speedup: the arms do different work by construction.
    let drift = last / first.unwrap_or(last);
    let ok = drift <= 2.0;
    println!(
        "G2 bounded flatness    16x sequence growth moves the bounded arm {drift:.2}x \
         (budget 2.0x, loaded box) → {}",
        verdict(ok)
    );
    if !ok {
        failures.push(format!(
            "G2: the bounded arm is not flat — {drift:.2}x over 16x sequence growth"
        ));
    }
}

fn g3_reduction(failures: &mut Vec<String>) {
    // Index for index, over scores carrying ties, zeros, a NaN and an
    // infinity — the shapes an ordering can disagree about.
    let scores = vec![
        0.5f32,
        0.1,
        0.1,
        f32::NAN,
        0.9,
        0.0,
        0.3,
        0.1,
        f32::INFINITY,
        -0.0,
    ];
    let positions: Vec<u64> = (0..scores.len() as u64).collect();
    let unpinned = vec![false; scores.len()];
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut mismatch = 0usize;
    for k in 0..=scores.len() + 2 {
        select_evict_windowed(
            &SinkWindowPolicy::UNBOUNDED,
            &positions,
            scores.len() as u64 - 1,
            &scores,
            k,
            &mut a,
        );
        select_evict_into(&scores, k, &unpinned, &mut b);
        if a != b {
            mismatch += 1;
        }
    }

    // And the sink-only composition against the shipped selector under this
    // module's own pin mask.
    let p = SinkWindowPolicy::new(3, usize::MAX);
    let mut mask = Vec::new();
    sink_pin_mask_into(&p, &positions, scores.len() as u64 - 1, &mut mask);
    let mut mismatch_pin = 0usize;
    for k in 0..=scores.len() + 2 {
        select_evict_windowed(&p, &positions, scores.len() as u64 - 1, &scores, k, &mut a);
        select_evict_into(&scores, k, &mask, &mut b);
        if a != b {
            mismatch_pin += 1;
        }
    }

    let ok = mismatch == 0 && mismatch_pin == 0;
    println!(
        "G3 reduction           UNBOUNDED vs select_evict_into: {mismatch} mismatch over 13 k; \
         sink-only vs the same under its pin mask: {mismatch_pin} → {}",
        verdict(ok)
    );
    if !ok {
        failures.push(format!(
            "G3: the off switch is a second code path ({mismatch} + {mismatch_pin} mismatches)"
        ));
    }
}

fn g4_alloc(failures: &mut Vec<String>) {
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    {
        let p = SinkWindowPolicy::new(N_SINK, WINDOW);
        let mut live: Vec<u64> = (0..p.capacity() as u64).collect();
        let mut scores = Vec::new();
        let mut evict = Vec::new();
        let mut mask = Vec::new();

        // Warm every buffer to its steady-state capacity first — a Vec
        // growing into its working size is a property of the first call, not
        // of the loop.
        for step in 0..64u64 {
            live.push(p.capacity() as u64 + step);
            hostile_scores(&live, &mut scores);
            select_evict_windowed(&p, &live, step, &scores, 1, &mut evict);
            sink_pin_mask_into(&p, &live, step, &mut mask);
            live.remove(evict[0]);
        }

        katgpt_core::alloc::reset_alloc_stats();
        for step in 1_000..3_000u64 {
            live.push(step);
            hostile_scores(&live, &mut scores);
            let k = p.evictions_needed(live.len());
            select_evict_windowed(&p, &live, step, &scores, k, &mut evict);
            sink_pin_mask_into(&p, &live, step, &mut mask);
            for &i in evict.iter() {
                live.remove(i);
            }
        }
        let (n_allocs, bytes) = katgpt_core::alloc::get_alloc_stats();
        let ok = n_allocs == 0;
        println!(
            "G4 alloc-free          {n_allocs} alloc(s), {bytes} byte(s) over 2000 \
             steady-state eviction steps → {}",
            verdict(ok)
        );
        if !ok {
            failures.push(format!(
                "G4: {n_allocs} allocation(s) in the steady-state loop"
            ));
        }
    }
    #[cfg(not(any(debug_assertions, feature = "alloc_tracking")))]
    {
        let _ = &mut *failures;
        println!(
            "G4 alloc-free          ⛔ NOT MEASURED — this profile compiles no allocator. \
             Re-run with `--features alloc_tracking`; a green run without it is not a G4 pass."
        );
    }
}

fn g5_fidelity_honesty(failures: &mut Vec<String>) {
    let p = SinkWindowPolicy::new(N_SINK, WINDOW);
    let unknown_is_lossy = p.fidelity(None) == WindowFidelity::Lossy;
    let short_is_lossy = p.fidelity(Some(WINDOW + 1)) == WindowFidelity::Lossy;
    let clearing_is_lossless = p.fidelity(Some(WINDOW)).is_lossless();
    // The classification order, which is what makes the sink guarantee hold
    // at the sequence length the policy exists for.
    let sink_survives_scrolling = p.classify(0, 1_000_000) == SlotClass::Sink;

    let ok = unknown_is_lossy && short_is_lossy && clearing_is_lossless && sink_survives_scrolling;
    println!(
        "G5 fidelity honesty    unknown d_max → Lossy: {unknown_is_lossy}; window < d_max → \
         Lossy: {short_is_lossy}; window >= d_max → Lossless: {clearing_is_lossless}; \
         a scrolled-away sink is still a sink: {sink_survives_scrolling} → {}",
        verdict(ok)
    );
    if !ok {
        failures.push("G5: the fidelity verdict is permissive where it must refuse".into());
    }
}

fn main() {
    println!("Bench 841 — KV permanent sinks + bounded window (Issue 841)");
    println!(
        "policy: n_sink {N_SINK}, window {WINDOW}, ceiling {}",
        N_SINK + WINDOW
    );
    let t0 = Instant::now();
    let mut failures: Vec<String> = Vec::new();

    g1_ceiling(&mut failures);
    println!();
    g2_flatness(&mut failures);
    println!();
    g3_reduction(&mut failures);
    println!();
    g4_alloc(&mut failures);
    println!();
    g5_fidelity_honesty(&mut failures);

    println!("\n(total {:.1}s)", t0.elapsed().as_secs_f64());
    println!(
        "\n⛔ NOT PROMOTED, and not for caution: with an unknown d_max this is a LOSSY KV\n\
         surface, and AGENTS.md requires the Plan-585 runaway gate on a SEALED LONG-CONTEXT\n\
         EVAL before any lossy policy goes default-on. This repo has no such eval, so the\n\
         promotion is blocked by a missing measurement rather than by a failing one.\n\
         `kv_eviction::runaway_gate` is the instrument; the corpus is the open task."
    );
    match failures.is_empty() {
        true => println!("\n✓ Bench 841 PASSED — every measured gate holds"),
        false => {
            for f in &failures {
                println!("✗ {f}");
            }
            println!("\n✗ Bench 841 FAILED — {} gate(s)", failures.len());
            std::process::exit(1);
        }
    }
}

fn verdict(ok: bool) -> &'static str {
    match ok {
        true => "PASS",
        false => "FAIL",
    }
}
