//! Percepta Head-to-Head: Our Rust Hull Attention vs Their Python+C++ Transformer
//!
//! ⚠️  UNFAIR COMPARISON — different algorithms, different machines.
//!
//! Theirs: Transformer executes WASM bytecodes (1 byte = 1 token) at ~30K tok/s.
//! Ours:   Rust backtracking with O(log N) hull attention compression.
//!
//! This proves Rust is fast, NOT that our algorithm beats theirs.
//! The FAIR comparison comes after Plan 064: we RIIR their transformer-vm,
//! then run same algorithm, same inputs, same machine: Rust tok/s vs C++ tok/s.
//!
//! What Percepta CANNOT do (but we can in riir-ai):
//! - Bomberman with learning/adaptation (lora.bin + validator.wasm + bandit)
//! - Self-play improvement (G-Zero, HL learning)
//! - Dynamic rule hotswap at runtime
//! - Real model inference with trained weights
//!
//! Run: cargo run --example sudoku_04_percepta_vs

use std::time::{Duration, Instant};

use katgpt_rs::percepta::{SolveEvent, StreamingSolver, Sudoku9x9};

/// Format duration with appropriate precision: µs when < 1ms, decimal ms when < 1s, else seconds.
fn fmt_duration(d: Duration) -> String {
    let micros = d.as_micros();
    if micros < 1000 {
        format!("{micros}µs")
    } else if micros < 1_000_000 {
        format!("{:.2}ms", d.as_secs_f64() * 1000.0)
    } else {
        format!("{:.2}s", d.as_secs_f64())
    }
}

/// The exact puzzle from Percepta's transformer-vm manifest.yaml
/// Source: https://github.com/Percepta-Core/transformer-vm/blob/main/transformer_vm/examples/manifest.yaml
/// "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
fn percepta_puzzle() -> Sudoku9x9 {
    let s = "530070000600195000098000060800060003400803001700020006060000280000419005000080079";
    let mut grid = [[0u8; 9]; 9];
    for (i, ch) in s.chars().enumerate() {
        grid[i / 9][i % 9] = ch.to_digit(10).unwrap() as u8;
    }
    Sudoku9x9::new(grid)
}

/// Benchmark a single solve, return (solved, duration, steps, hull_size, total_trace).
fn bench_solve(grid: [[u8; 9]; 9]) -> (bool, std::time::Duration, usize, usize, usize) {
    let mut solver = StreamingSolver::new(grid);
    let start = Instant::now();
    let solved = solver.solve_streaming();
    let elapsed = start.elapsed();

    let event = solver.events.last();
    let (steps, hull_size, total_trace) = match event {
        Some(SolveEvent::Solved {
            steps,
            hull_size,
            total_trace,
        }) => (*steps, *hull_size, *total_trace),
        _ => (solver.step, solver.cache.hull_len(), solver.cache.len()),
    };

    (solved, elapsed, steps, hull_size, total_trace)
}

/// Run N iterations and return median duration.
fn bench_median(
    grid: [[u8; 9]; 9],
    iterations: usize,
) -> (bool, std::time::Duration, usize, usize, usize, f64) {
    let mut results = Vec::with_capacity(iterations);
    let mut solved = false;
    let mut steps = 0;
    let mut hull_size = 0;
    let mut total_trace = 0;

    for _ in 0..iterations {
        let (s, dur, st, hs, tt) = bench_solve(grid);
        solved = s;
        steps = st;
        hull_size = hs;
        total_trace = tt;
        results.push(dur);
    }

    results.sort();
    let median = results[iterations / 2];
    let steps_per_sec = steps as f64 / median.as_secs_f64();

    (solved, median, steps, hull_size, total_trace, steps_per_sec)
}

fn main() {
    println!("⚔️  Percepta Head-to-Head: Rust Hull Attention vs Python+C++ Transformer");
    println!("{}", "═".repeat(70));
    println!();
    println!("  Their setup:  Python+C++ transformer executing WASM bytecodes");
    println!("  Our setup:    Pure Rust, 2D convex hull attention (Graham Scan + ternary search)");
    println!("  Their speed:  ~30,000 tok/s (C++ engine with BLAS + CHT hull cache)");
    println!("  Their tokens: ~900,000 tokens for Sudoku (each token = 1 byte of machine state)");
    println!();

    // ── Puzzle 1: Percepta's Reference Puzzle ──────────────────────
    println!("📋 Round 1: Percepta's Reference Puzzle (from manifest.yaml)");
    println!("{}", "─".repeat(70));

    let percepta = percepta_puzzle();
    let percepta_clues = percepta.clue_count();
    println!(
        "  Puzzle:  530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    );
    println!("  Clues:   {percepta_clues}");
    print!("{}", percepta.display());

    let iterations = 100;
    println!("  Benchmarking {iterations} iterations...");
    let (solved, median, steps, hull_size, total_trace, our_tps) =
        bench_median(percepta.grid, iterations);

    if !solved {
        println!("  ❌ FAILED TO SOLVE!");
        return;
    }

    // Percepta reports ~30K tok/s for their C++ engine on Sudoku
    // Their puzzle generates ~900K tokens of execution trace
    let their_tps = 30_000.0;
    let their_tokens = 900_000.0;
    let their_time_secs = their_tokens / their_tps;

    let our_time_fmt = fmt_duration(median);

    println!();
    println!("  ✅ Solved!");
    println!("  Steps (backtracking): {steps}");
    println!(
        "  Hull compression:     {total_trace} → {hull_size} vertices ({:.1}x)",
        total_trace as f64 / hull_size as f64
    );
    println!(
        "  Attention:            O({total_trace}) → O(log {hull_size}) ≈ O({})",
        (hull_size as f64).log2().ceil() as usize
    );
    println!();
    println!("  ┌─────────────────────────────────────────────────────┐");
    println!("  │                    SPEED COMPARISON                 │");
    println!("  ├─────────────────────────────────────────────────────┤");
    println!(
        "  │  Percepta (C++):  {their_tps:>10.0} tok/s  │  {their_time_secs:>8.1}s (900K tokens) │"
    );
    println!("  │  Ours (Rust):     {our_tps:>10.0} steps/s │  {our_time_fmt:>12}        │");
    println!("  └─────────────────────────────────────────────────────┘");

    if median.as_secs_f64() < their_time_secs {
        let ratio = their_time_secs / median.as_secs_f64();
        println!("  🏆 We are {:.0}× FASTER", ratio);
    } else {
        println!(
            "  ⚠️  They beat us (different algorithm — they're executing WASM in a transformer)"
        );
    }

    // ── Puzzle 2: Arto Inkala (World's Hardest) ────────────────────
    println!();
    println!("📋 Round 2: Arto Inkala — World's Hardest Sudoku (21 clues)");
    println!("{}", "─".repeat(70));

    let arto = Sudoku9x9::arto_inkala();
    println!("  Clues:   21 (harder than Percepta's puzzle)");
    print!("{}", arto.display());

    println!("  Benchmarking {iterations} iterations...");
    let (solved2, median2, steps2, hull2, trace2, our_tps2) = bench_median(arto.grid, iterations);

    if !solved2 {
        println!("  ❌ FAILED TO SOLVE!");
        return;
    }

    println!();
    println!("  ✅ Solved!");
    println!("  Steps (backtracking): {steps2}");
    println!(
        "  Hull compression:     {trace2} → {hull2} vertices ({:.1}x)",
        trace2 as f64 / hull2 as f64
    );
    println!("  Our throughput:       {our_tps2:.0} steps/s");
    println!("  Our time:             {}", fmt_duration(median2));

    // ── Summary ────────────────────────────────────────────────────
    println!();
    println!("📊 Summary");
    println!("{}", "─".repeat(70));
    println!("  ┌──────────────────────┬───────────────┬──────────────┬──────────────┐");
    println!("  │ Puzzle               │ Steps         │ Our Time     │ Throughput   │");
    println!("  ├──────────────────────┼───────────────┼──────────────┼──────────────┤");
    println!(
        "  │ Percepta reference   │ {steps:>11} │ {:>12} │ {our_tps:>8.0}/s │",
        fmt_duration(median)
    );
    println!(
        "  │ Arto Inkala (harder) │ {steps2:>11} │ {:>12} │ {our_tps2:>8.0}/s │",
        fmt_duration(median2)
    );
    println!("  └──────────────────────┴───────────────┴──────────────┴──────────────┘");
    println!();
    println!("  ⚠️  UNFAIR — different algorithms, different machines.");
    println!("     Percepta: transformer executes WASM bytecodes (1 byte = 1 token).");
    println!("     Ours:     Rust backtracking with O(log N) hull attention compression.");
    println!("     Speed difference is mostly algorithm, not language.");
    println!();
    println!("  FAIR comparison: after Plan 064 (RIIR transformer-vm), we run same");
    println!("  algorithm, same inputs, same machine. Rust tok/s vs C++ tok/s. Then we");
    println!("  know if Rust beats C++, not just if backtracking beats WASM interpretation.");
    println!();
    println!("  What Percepta CANNOT do (but we can in riir-ai):");
    println!("     • Bomberman with learning (lora.bin + validator.wasm + bandit)");
    println!("     • Self-play improvement (G-Zero, heuristic learning)");
    println!("     • Dynamic rule hotswap at runtime");
    println!("     • Real model inference with trained weights");
    println!();
    println!("✨ Done.");
}
