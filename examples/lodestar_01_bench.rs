//! Lodestar Completion-Distance Pruning Benchmark — Plan 207, T13.
//!
//! Micro-benchmarks for LodestarPruner overhead: per-call `is_valid`,
//! batch validation, CompletionHorizon methods, end-to-end DDTree builds,
//! and `follow_path` inner-loop cost.
//!
//! ```sh
//! cargo run --release --example lodestar_01_bench --features lodestar
//! ```

#![cfg(feature = "lodestar")]

use std::time::Instant;

use katgpt_rs::pruners::{LodestarAutomaton, LodestarConfig, LodestarPruner};
use katgpt_rs::speculative::types::CompletionHorizon;
use katgpt_rs::speculative::{
    ConstraintPruner, NoPruner, build_dd_tree_lodestar, build_dd_tree_pruned,
};
use katgpt_rs::types::Config;

// ── Config ─────────────────────────────────────────────────────

const WARMUP: u64 = 50;
const N_ITERS: u64 = 500;

// ── Automaton: header + nested array ───────────────────────────
//
// Vocabulary: OPEN=0, CLOSE=1, NUM=2, COMMA=3, HDR=4
// Grammar:   H H H [ N , N , ... ]  (nested arrays up to depth 3)
// States:    0,1,2 = header;  3+(d-1)*2 = Value@depth d;  +1 = More@depth d
//            ACCEPT = 3 + 3*2 = 9,  N_STATES = 10

const OPEN: usize = 0;
const CLOSE: usize = 1;
const NUM: usize = 2;
const COMMA: usize = 3;
const HDR: usize = 4;
const VOCAB: usize = 5;
const MAX_DEPTH: usize = 3;
const HLEN: usize = 3;

const fn s_value(d: usize) -> usize {
    HLEN + (d - 1) * 2
}
const fn s_more(d: usize) -> usize {
    HLEN + (d - 1) * 2 + 1
}
const ACCEPT: usize = HLEN + MAX_DEPTH * 2; // = 9
const N_STATES: usize = ACCEPT + 1; // = 10
const START: usize = 0;

/// Build the header+array automaton used throughout the benchmark suite.
fn build_automaton() -> LodestarAutomaton {
    let mut b = LodestarAutomaton::builder(VOCAB, N_STATES, START);

    // Header chain: 0 →HDR→ 1 →HDR→ 2 →OPEN→ s_value(1)
    b.add_transition(0, HDR, 1);
    b.add_transition(1, HDR, 2);
    b.add_transition(2, OPEN, s_value(1));

    // Body states for each nesting depth.
    for d in 1..=MAX_DEPTH {
        let sv = s_value(d);
        let sm = s_more(d);

        // Value state: accept NUM → More, accept OPEN → deeper Value (if not max).
        b.add_transition(sv, NUM, sm);
        if d < MAX_DEPTH {
            b.add_transition(sv, OPEN, s_value(d + 1));
        }

        // More state: COMMA → Value, CLOSE → parent More or ACCEPT.
        b.add_transition(sm, COMMA, sv);
        let close_target = if d == 1 { ACCEPT } else { s_more(d - 1) };
        b.add_transition(sm, CLOSE, close_target);
    }

    b.add_accept(ACCEPT);
    b.build()
}

// ── Helpers ────────────────────────────────────────────────────

/// Simple stats from a sample of timing measurements (nanoseconds).
#[allow(dead_code)]
struct Stats {
    p50: f64,
    p99: f64,
    mean: f64,
    min: f64,
}

fn compute_stats(samples: &[f64]) -> Stats {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    Stats {
        p50: sorted[n / 2],
        p99: sorted[(n * 99) / 100].min(sorted[n - 1]),
        mean: sorted.iter().sum::<f64>() / n as f64,
        min: sorted[0],
    }
}

/// Build parent_tokens that reach a given state at a given depth.
/// Returns a path that exercises the automaton at realistic depths.
fn make_parent_tokens(depth: usize) -> Vec<usize> {
    match depth {
        0 => vec![],
        2 => vec![HDR, HDR],                   // state 2 (need OPEN next)
        5 => vec![HDR, HDR, OPEN, NUM, COMMA], // s_value(1) after "H H [ N ,"
        8 => vec![HDR, HDR, OPEN, NUM, COMMA, NUM, COMMA, NUM], // s_more(1)
        _ => vec![],
    }
}

/// Create normalized descending-harmonics marginals for `seq_len` positions × `vocab` tokens.
fn make_marginals(seq_len: usize, vocab: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(seq_len);
    for _ in 0..seq_len {
        let mut row = Vec::with_capacity(vocab);
        let mut sum = 0.0f32;
        for t in 0..vocab {
            let v = 1.0 / ((t + 1) as f32);
            row.push(v);
            sum += v;
        }
        for v in &mut row {
            *v /= sum;
        }
        out.push(row);
    }
    out
}

// ── Bench 1: Per-call is_valid overhead ────────────────────────

fn bench_is_valid() {
    println!("── Bench 1: Per-call is_valid — LodestarPruner vs NoPruner ──\n");

    let auto = build_automaton();
    let pruner = LodestarPruner::with_budget(auto, 20);
    let no_pruner = NoPruner;

    let depths: &[usize] = &[0, 2, 5, 8];

    println!(
        "  {:>12} {:>8} {:>12} {:>12} {:>12}",
        "Pruner", "Depth", "p50 (ns)", "mean (ns)", "min (ns)"
    );
    println!("  {}", "-".repeat(60));

    for &depth in depths {
        let parent_tokens = make_parent_tokens(depth);

        // LodestarPruner
        let mut lode_samples = Vec::with_capacity(N_ITERS as usize);
        for _ in 0..WARMUP {
            for token in 0..VOCAB {
                std::hint::black_box(pruner.is_valid(depth, token, &parent_tokens));
            }
        }
        for _ in 0..N_ITERS {
            let start = Instant::now();
            for token in 0..VOCAB {
                std::hint::black_box(pruner.is_valid(depth, token, &parent_tokens));
            }
            lode_samples.push(start.elapsed().as_nanos() as f64 / VOCAB as f64);
        }
        let lode_stats = compute_stats(&lode_samples);
        println!(
            "  {:>12} {:>8} {:>12.1} {:>12.1} {:>12.1}",
            "LodestarPruner", depth, lode_stats.p50, lode_stats.mean, lode_stats.min,
        );

        // NoPruner
        let mut no_samples = Vec::with_capacity(N_ITERS as usize);
        for _ in 0..WARMUP {
            for token in 0..VOCAB {
                std::hint::black_box(no_pruner.is_valid(depth, token, &parent_tokens));
            }
        }
        for _ in 0..N_ITERS {
            let start = Instant::now();
            for token in 0..VOCAB {
                std::hint::black_box(no_pruner.is_valid(depth, token, &parent_tokens));
            }
            no_samples.push(start.elapsed().as_nanos() as f64 / VOCAB as f64);
        }
        let no_stats = compute_stats(&no_samples);
        println!(
            "  {:>12} {:>8} {:>12.1} {:>12.1} {:>12.1}",
            "NoPruner", depth, no_stats.p50, no_stats.mean, no_stats.min,
        );
    }
    println!();
}

// ── Bench 2: batch_is_valid overhead ───────────────────────────

fn bench_batch_is_valid() {
    println!("── Bench 2: batch_is_valid — LodestarPruner vs NoPruner ──\n");

    let auto = build_automaton();
    let pruner = LodestarPruner::with_budget(auto, 20);
    let no_pruner = NoPruner;

    let candidates: Vec<usize> = (0..VOCAB).collect();
    let depths: &[usize] = &[0, 2, 5, 8];

    println!(
        "  {:>12} {:>8} {:>12} {:>12} {:>12}",
        "Pruner", "Depth", "p50 (ns)", "mean (ns)", "min (ns)"
    );
    println!("  {}", "-".repeat(60));

    for &depth in depths {
        let parent_tokens = make_parent_tokens(depth);
        let mut results = vec![false; VOCAB];

        // LodestarPruner batch
        let mut lode_samples = Vec::with_capacity(N_ITERS as usize);
        for _ in 0..WARMUP {
            pruner.batch_is_valid(depth, &candidates, &parent_tokens, &mut results);
            std::hint::black_box(&results);
        }
        for _ in 0..N_ITERS {
            let start = Instant::now();
            pruner.batch_is_valid(depth, &candidates, &parent_tokens, &mut results);
            lode_samples.push(start.elapsed().as_nanos() as f64);
            std::hint::black_box(&results);
        }
        let lode_stats = compute_stats(&lode_samples);
        println!(
            "  {:>12} {:>8} {:>12.1} {:>12.1} {:>12.1}",
            "LodestarPruner", depth, lode_stats.p50, lode_stats.mean, lode_stats.min,
        );

        // NoPruner batch
        let mut no_samples = Vec::with_capacity(N_ITERS as usize);
        for _ in 0..WARMUP {
            no_pruner.batch_is_valid(depth, &candidates, &parent_tokens, &mut results);
            std::hint::black_box(&results);
        }
        for _ in 0..N_ITERS {
            let start = Instant::now();
            no_pruner.batch_is_valid(depth, &candidates, &parent_tokens, &mut results);
            no_samples.push(start.elapsed().as_nanos() as f64);
            std::hint::black_box(&results);
        }
        let no_stats = compute_stats(&no_samples);
        println!(
            "  {:>12} {:>8} {:>12.1} {:>12.1} {:>12.1}",
            "NoPruner", depth, no_stats.p50, no_stats.mean, no_stats.min,
        );
    }
    println!();
}

// ── Bench 3: CompletionHorizon method overhead ─────────────────

fn bench_completion_horizon() {
    println!("── Bench 3: CompletionHorizon — min_completion_distance + singular_span_len ──\n");

    let auto = build_automaton();
    let pruner = LodestarPruner::new(auto);

    let depths: &[usize] = &[0, 2, 5, 8];

    println!(
        "  {:>30} {:>8} {:>12} {:>12} {:>12}",
        "Method", "Depth", "p50 (ns)", "mean (ns)", "min (ns)"
    );
    println!("  {}", "-".repeat(68));

    for &depth in depths {
        let parent_tokens = make_parent_tokens(depth);

        // min_completion_distance per token
        let mut mcd_samples = Vec::with_capacity(N_ITERS as usize);
        for _ in 0..WARMUP {
            for token in 0..VOCAB {
                std::hint::black_box(pruner.min_completion_distance(depth, token, &parent_tokens));
            }
        }
        for _ in 0..N_ITERS {
            let start = Instant::now();
            for token in 0..VOCAB {
                std::hint::black_box(pruner.min_completion_distance(depth, token, &parent_tokens));
            }
            mcd_samples.push(start.elapsed().as_nanos() as f64 / VOCAB as f64);
        }
        let mcd_stats = compute_stats(&mcd_samples);
        println!(
            "  {:>30} {:>8} {:>12.1} {:>12.1} {:>12.1}",
            "min_completion_distance", depth, mcd_stats.p50, mcd_stats.mean, mcd_stats.min,
        );

        // singular_span_len
        let mut ssl_samples = Vec::with_capacity(N_ITERS as usize);
        for _ in 0..WARMUP {
            std::hint::black_box(pruner.singular_span_len(depth, &parent_tokens));
        }
        for _ in 0..N_ITERS {
            let start = Instant::now();
            std::hint::black_box(pruner.singular_span_len(depth, &parent_tokens));
            ssl_samples.push(start.elapsed().as_nanos() as f64);
        }
        let ssl_stats = compute_stats(&ssl_samples);
        println!(
            "  {:>30} {:>8} {:>12.1} {:>12.1} {:>12.1}",
            "singular_span_len", depth, ssl_stats.p50, ssl_stats.mean, ssl_stats.min,
        );
    }
    println!();
}

// ── Bench 4: End-to-end DDTree builds ─────────────────────────

fn bench_end_to_end() {
    println!("── Bench 4: End-to-end build_dd_tree_lodestar vs build_dd_tree_pruned ──\n");

    let seq_len = 8;
    let marginals = make_marginals(seq_len, VOCAB);
    let refs: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    let mut config = Config::draft();
    config.vocab_size = VOCAB;
    config.tree_budget = 64;

    let auto = build_automaton();
    let pruner = LodestarPruner::with_budget(auto, seq_len);

    // ── (a) Baseline: build_dd_tree_pruned with NoPruner ──────────
    for _ in 0..WARMUP {
        let tree = build_dd_tree_pruned(&refs, &config, &NoPruner, false);
        std::hint::black_box(tree);
    }
    let mut baseline_samples = Vec::with_capacity(N_ITERS as usize);
    for _ in 0..N_ITERS {
        let start = Instant::now();
        let tree = build_dd_tree_pruned(&refs, &config, &NoPruner, false);
        baseline_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        std::hint::black_box(tree);
    }

    // ── (b) build_dd_tree_lodestar with NoPruner (default-0, should be ~identical) ──
    for _ in 0..WARMUP {
        let tree = build_dd_tree_lodestar(&refs, &config, &NoPruner, &LodestarConfig::default());
        std::hint::black_box(tree);
    }
    let mut nope_samples = Vec::with_capacity(N_ITERS as usize);
    for _ in 0..N_ITERS {
        let start = Instant::now();
        let tree = build_dd_tree_lodestar(&refs, &config, &NoPruner, &LodestarConfig::default());
        nope_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        std::hint::black_box(tree);
    }

    // ── (c) build_dd_tree_lodestar with LodestarPruner + budget ───
    for _ in 0..WARMUP {
        let tree = build_dd_tree_lodestar(&refs, &config, &pruner, &LodestarConfig::default());
        std::hint::black_box(tree);
    }
    let mut pruned_samples = Vec::with_capacity(N_ITERS as usize);
    for _ in 0..N_ITERS {
        let start = Instant::now();
        let tree = build_dd_tree_lodestar(&refs, &config, &pruner, &LodestarConfig::default());
        pruned_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        std::hint::black_box(tree);
    }

    // ── (d) build_dd_tree_lodestar with full thinking mode ────────
    for _ in 0..WARMUP {
        let tree = build_dd_tree_lodestar(&refs, &config, &pruner, &LodestarConfig::thinking(0.5));
        std::hint::black_box(tree);
    }
    let mut think_samples = Vec::with_capacity(N_ITERS as usize);
    for _ in 0..N_ITERS {
        let start = Instant::now();
        let tree = build_dd_tree_lodestar(&refs, &config, &pruner, &LodestarConfig::thinking(0.5));
        think_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        std::hint::black_box(tree);
    }

    // ── Report ────────────────────────────────────────────────────
    let baseline = compute_stats(&baseline_samples);
    let nope = compute_stats(&nope_samples);
    let pruned = compute_stats(&pruned_samples);
    let think = compute_stats(&think_samples);

    println!(
        "  Config: seq_len={seq_len}, vocab={VOCAB}, tree_budget={}",
        config.tree_budget
    );
    println!("  Warmup: {WARMUP}, Measure: {N_ITERS}\n");

    println!(
        "  {:>32} {:>10} {:>10} {:>10} {:>10}",
        "Path", "p50 (μs)", "mean (μs)", "min (μs)", "Δmean%"
    );
    println!("  {}", "-".repeat(76));
    println!(
        "  {:>32} {:>10.2} {:>10.2} {:>10.2} {:>10}",
        "pruned + NoPruner (baseline)", baseline.p50, baseline.mean, baseline.min, "—",
    );
    println!(
        "  {:>32} {:>10.2} {:>10.2} {:>10.2} {:>9.1}%",
        "lodestar + NoPruner (default-0)",
        nope.p50,
        nope.mean,
        nope.min,
        (nope.mean - baseline.mean) / baseline.mean * 100.0,
    );
    println!(
        "  {:>32} {:>10.2} {:>10.2} {:>10.2} {:>9.1}%",
        "lodestar + LodestarPruner",
        pruned.p50,
        pruned.mean,
        pruned.min,
        (pruned.mean - baseline.mean) / baseline.mean * 100.0,
    );
    println!(
        "  {:>32} {:>10.2} {:>10.2} {:>10.2} {:>9.1}%",
        "lodestar + thinking(0.5)",
        think.p50,
        think.mean,
        think.min,
        (think.mean - baseline.mean) / baseline.mean * 100.0,
    );
    println!();
}

// ── Bench 5: follow_path overhead at various depths ────────────

fn bench_follow_path() {
    println!("── Bench 5: automaton.follow_path at various depths ──\n");

    let auto = build_automaton();
    let depths: &[usize] = &[0, 2, 5, 8];

    println!(
        "  {:>8} {:>12} {:>12} {:>12}",
        "Depth", "p50 (ns)", "mean (ns)", "min (ns)"
    );
    println!("  {}", "-".repeat(48));

    for &depth in depths {
        let parent_tokens = make_parent_tokens(depth);

        // Warmup
        for _ in 0..WARMUP {
            std::hint::black_box(auto.follow_path(&parent_tokens));
        }

        let mut samples = Vec::with_capacity(N_ITERS as usize);
        for _ in 0..N_ITERS {
            let start = Instant::now();
            std::hint::black_box(auto.follow_path(&parent_tokens));
            samples.push(start.elapsed().as_nanos() as f64);
        }

        let stats = compute_stats(&samples);
        println!(
            "  {:>8} {:>12.1} {:>12.1} {:>12.1}",
            depth, stats.p50, stats.mean, stats.min,
        );
    }
    println!();
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Lodestar Completion-Distance Pruning Benchmark — Plan 207 T13");
    println!("  Warmup: {WARMUP} iters, Measure: {N_ITERS} iters");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    bench_is_valid();
    bench_batch_is_valid();
    bench_completion_horizon();
    bench_end_to_end();
    bench_follow_path();

    // ── Verdict ────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Verdict — GOAT criterion: per-step overhead < ~50 ns");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Run on this machine and check that all per-call numbers");
    println!("  (Bench 1-3, 5) show p50 < 50 ns. End-to-end (Bench 4)");
    println!("  overhead should be < ~5% for default-0 and < ~20% for");
    println!("  full thinking mode vs baseline.");
    println!();
    println!("  Expected: PASS ✅  (Lodestar is O(1) per lookup, precomputed)");
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmark Complete");
    println!("═══════════════════════════════════════════════════════════════");
}

// TL;DR: 5-benchmark suite for Plan 207 Lodestar — per-call is_valid,
// batch_is_valid, CompletionHorizon overhead, end-to-end DDTree builds
// (pruned vs lodestar with NoPruner/LodestarPruner/thinking), and
// follow_path inner-loop cost. Run with:
// cargo run --release --example lodestar_01_bench --features lodestar
