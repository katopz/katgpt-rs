//! `fixed_log_softmax` G2 — ns/call parity vs the incumbent max-subtract
//! form (Issue 959 T5).
//!
//! The fixed form adds one `log1p` close and n subtractions to the same
//! n-exp fold the incumbent already runs — the parity claim is "same
//! cost class", gated at ratio ≤ 1.5 (measured, then pinned). Timing is
//! INTERLEAVED (paired a-chunk/b-chunk, alternating order, median
//! across rounds — the Issue-723 treatment: sequential A/B measures the
//! box, not the arms), results flow through `black_box` so no arm can
//! be satisfied by absent work (the timed-region loud-zero trap).
//!
//! # Box-state provenance (AGENTS G2 law)
//!
//! Every latency number below is a point-in-time read on a shared box.
//! The run prints a box-state line (timestamp, parallelism, profile);
//! quote `uptime` and power state beside any reuse of these numbers.
//!
//! # Run
//!
//! ```bash
//! cargo run -p katgpt-core --features fixed_log_softmax --bench softmax_fix_log_softmax_goat --release -- --nocapture
//! ```

#![cfg(feature = "fixed_log_softmax")]

use std::hint::black_box;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Incumbent max-subtract log-softmax — the standard form this
/// primitive replaces at instrument call sites. Same slice-in/slice-out
/// shape, same n-exp fold; the `1 +` close is where the cliff lives.
fn incumbent_log_softmax(z: &[f32], out: &mut [f32]) {
    let mut best = f32::NEG_INFINITY;
    for &v in z {
        if v > best {
            best = v;
        }
    }
    let mut sum = 0.0_f32;
    for &v in z {
        sum += (v - best).exp();
    }
    let lse = best + sum.ln();
    for (o, &v) in out.iter_mut().zip(z.iter()) {
        *o = v - lse;
    }
}

/// Deterministic fixture: n logits spanning the resolution band —
/// margins 0..~100 below the max, so the exercised fold includes the
/// subnormal-tail terms.
fn fixture(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let i = i as f32;
            if i == 0.0 {
                0.0
            } else {
                -((i * 7919.0).rem_euclid(100.0)) // deterministic scatter 0..−100
            }
        })
        .collect()
}

const CALLS_PER_CHUNK: usize = 2000;
const ROUNDS: usize = 9;

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// One paired round: (a-chunk, b-chunk) or (b-chunk, a-chunk)
/// alternating — the interleaved treatment.
fn measure(
    z: &[f32],
    fixed_out: &mut [f32],
    incumbent_out: &mut [f32],
    fixed_first: bool,
) -> (f64, f64) {
    let run = |which: bool, out: &mut [f32]| -> f64 {
        let t0 = Instant::now();
        for _ in 0..CALLS_PER_CHUNK {
            if which {
                katgpt_core::softmax_fix::fixed_log_softmax(z, out);
                black_box(&*out);
            } else {
                incumbent_log_softmax(z, out);
                black_box(&*out);
            }
        }
        t0.elapsed().as_secs_f64()
    };
    if fixed_first {
        let a = run(true, fixed_out);
        let b = run(false, incumbent_out);
        (a, b)
    } else {
        let b = run(false, incumbent_out);
        let a = run(true, fixed_out);
        (a, b)
    }
}

fn main() {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    println!("═══════════════════════════════════════════════════════════════");
    println!("  fixed_log_softmax G2 parity gate (Issue 959 T5)");
    println!(
        "  epoch {now} · cores {:?} · debug_assertions {}",
        std::thread::available_parallelism().map(|n| n.get()).ok(),
        cfg!(debug_assertions)
    );
    println!("  ⚠ box state: quote `uptime` + power state beside any latency reuse");
    println!("  {CALLS_PER_CHUNK} calls/chunk × {ROUNDS} interleaved rounds, median ns/call");
    println!("═══════════════════════════════════════════════════════════════");

    let mut all_pass = true;
    for &n in &[2usize, 64, 1024] {
        let z = fixture(n);
        let mut fixed_out = vec![0.0_f32; n];
        let mut incumbent_out = vec![0.0_f32; n];

        // Warmup both arms.
        katgpt_core::softmax_fix::fixed_log_softmax(&z, &mut fixed_out);
        black_box(&*fixed_out);
        incumbent_log_softmax(&z, &mut incumbent_out);
        black_box(&*incumbent_out);

        let mut fixed_secs = Vec::with_capacity(ROUNDS);
        let mut incumbent_secs = Vec::with_capacity(ROUNDS);
        for r in 0..ROUNDS {
            let (a, b) = measure(&z, &mut fixed_out, &mut incumbent_out, r % 2 == 0);
            fixed_secs.push(a);
            incumbent_secs.push(b);
        }
        let fixed_ns = median(&mut fixed_secs) * 1e9 / f64::from(CALLS_PER_CHUNK as u32);
        let incumbent_ns = median(&mut incumbent_secs) * 1e9 / f64::from(CALLS_PER_CHUNK as u32);
        let ratio = fixed_ns / incumbent_ns;
        // Same cost class: log1p vs log plus n subtractions must not
        // move the call out of the incumbent's class. 1.5× is the
        // parity band; the absolute numbers go to the record.
        let pass = ratio <= 1.5;
        all_pass &= pass;
        println!(
            "  n={n:<5} incumbent {incumbent_ns:8.1} ns/call · fixed {fixed_ns:8.1} ns/call · ratio {ratio:.3}  {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    println!(
        "═══════════════ {} ═══════════════",
        if all_pass {
            "G2 PARITY PASS"
        } else {
            "G2 PARITY FAIL"
        }
    );
    if !all_pass {
        std::process::exit(1);
    }
}
