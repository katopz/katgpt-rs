//! Monopoly FSM Benchmark — Performance Measurement
//!
//! Measures game engine throughput, per-turn timing, and AI decision latency.
//!
//! Run: `cargo run --example monopoly_04_bench --features monopoly --quiet`

use std::time::Instant;

use fastrand::Rng;
use katgpt_core::stats::nearest_rank;
use katgpt_rs::pruners::monopoly::{
    GreedyPlayer, HLPlayer, MonopolyPlayer, RandomPlayer, ValidatorPlayer, run_game,
};

const WARMUP: usize = 10;
const BENCH_GAMES: usize = 100;
const MAX_TURNS: u32 = 300;
const SEED: u64 = 42;

fn main() {
    let mut rng = Rng::with_seed(SEED);

    // ── Header ──
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Monopoly FSM Benchmark — {BENCH_GAMES} Games (warmup={WARMUP})");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // ── Warmup ──
    print!("  Warmup ({WARMUP} games)... ");
    for i in 0..WARMUP {
        let mut players: [Box<dyn MonopolyPlayer>; 4] = [
            Box::new(RandomPlayer::new(0)),
            Box::new(GreedyPlayer::new(1)),
            Box::new(ValidatorPlayer::new(2)),
            Box::new(HLPlayer::new(3)),
        ];
        let _ = run_game(SEED + i as u64, &mut players, &mut rng, MAX_TURNS);
    }
    println!("done");
    println!();

    // ── Benchmark: 100 games ──
    let mut total_turns = 0u32;
    let mut total_events = 0usize;
    let mut min_time = u128::MAX;
    let mut max_time = 0u128;
    let mut game_times = Vec::with_capacity(BENCH_GAMES);

    let bench_start = Instant::now();
    for i in 0..BENCH_GAMES {
        let mut players: [Box<dyn MonopolyPlayer>; 4] = [
            Box::new(RandomPlayer::new(0)),
            Box::new(GreedyPlayer::new(1)),
            Box::new(ValidatorPlayer::new(2)),
            Box::new(HLPlayer::new(3)),
        ];
        let game_start = Instant::now();
        let result = run_game(
            SEED + WARMUP as u64 + i as u64,
            &mut players,
            &mut rng,
            MAX_TURNS,
        );
        let elapsed = game_start.elapsed().as_micros();

        total_turns += result.total_turns;
        total_events += result.events.len();
        game_times.push(elapsed);
        min_time = min_time.min(elapsed);
        max_time = max_time.max(elapsed);
    }
    let bench_elapsed = bench_start.elapsed();

    let avg_game_us = game_times.iter().sum::<u128>() / BENCH_GAMES as u128;
    let avg_turn_us = if total_turns > 0 {
        bench_elapsed.as_micros() / total_turns as u128
    } else {
        0
    };
    let throughput = BENCH_GAMES as f64 / bench_elapsed.as_secs_f64();

    // ── Results ──
    println!("─── Game Performance ────────────────────────────────────────────");
    println!(
        "  Total time:           {:.2}s",
        bench_elapsed.as_secs_f64()
    );
    println!("  Games/second:         {throughput:.1}");
    println!(
        "  Avg game:             {avg_game_us}µs ({:.2}ms)",
        avg_game_us as f64 / 1000.0
    );
    println!("  Min game:             {min_time}µs");
    println!("  Max game:             {max_time}µs");
    println!(
        "  Avg turns/game:       {:.1}",
        total_turns as f64 / BENCH_GAMES as f64
    );
    println!(
        "  Avg events/game:      {:.0}",
        total_events as f64 / BENCH_GAMES as f64
    );
    println!();
    println!("─── Turn Performance ────────────────────────────────────────────");
    println!(
        "  Avg turn:             {avg_turn_us}µs ({:.2}ms)",
        avg_turn_us as f64 / 1000.0
    );
    println!("  Target:               <1ms/turn (1000µs)");
    if avg_turn_us < 1000 {
        println!(
            "  Status:               ✅ PASS ({:.1}x under target)",
            1000.0 / avg_turn_us as f64
        );
    } else {
        println!("  Status:               ❌ FAIL");
    }
    println!();

    // ── Latency Distribution ──
    game_times.sort();
    // `scripts/percentile_index_audit.py`: `xs[n * 99 / 100]` is `xs[n - 1]`
    // -- the MAX -- for every n <= 100, and BENCH_GAMES is 100, so the printed
    // p99 WAS the max. p90 is fine (support 10); p99 is nearest-rank with its
    // support printed, because at this sample count it rests on 2 games.
    let p50 = game_times[BENCH_GAMES * 50 / 100];
    let (p90, p90_s) = nearest_rank(&game_times, 0.90);
    let (p99, p99_s) = nearest_rank(&game_times, 0.99);
    println!("─── Latency Distribution ────────────────────────────────────────");
    println!("  p50: {p50}µs ({:.2}ms)", p50 as f64 / 1000.0);
    println!("  p90: {p90}µs ({:.2}ms)  [support {p90_s}]", p90 as f64 / 1000.0);
    println!("  p99: {p99}µs ({:.2}ms)  [support {p99_s}]", p99 as f64 / 1000.0);
}
