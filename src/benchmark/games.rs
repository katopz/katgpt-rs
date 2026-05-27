//! E2E Game timing benchmarks: transformer generate through plasma/hot/warm/cold cache states.
//!
//! Measures end-to-end time for generating sequences via `generate_into` with
//! different cache thermal states:
//! - **Plasma**: fresh cache, first generation (cold start + SIMD warm)
//! - **Hot**: cache already populated with recent tokens
//! - **Warm**: cache reset but allocator is hot (pages still in L2/L3)
//! - **Cold**: full cache reallocation between runs (simulates cold start)

use super::{BenchCategory, BenchResult};
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, generate_into};
use crate::types::{Config, Rng};
use std::cell::UnsafeCell;
use std::time::Instant;

/// Wrapper to make `UnsafeCell` `Sync` (SAFETY: only used in single-threaded benchmarks).
struct SyncCell<T>(UnsafeCell<T>);
unsafe impl<T> Sync for SyncCell<T> {}
impl<T> SyncCell<T> {
    const fn new(v: T) -> Self {
        SyncCell(UnsafeCell::new(v))
    }
    fn get(&self) -> *mut T {
        self.0.get()
    }
}

/// Cache thermal state for E2E game timing.
#[derive(Clone, Copy, Debug)]
enum CacheState {
    /// Fresh allocation, first generation. Simulates absolute cold start.
    Plasma,
    /// Cache reused across generations (warm allocator + pre-populated).
    Hot,
    /// Cache reset between generations (allocator warm, cache empty).
    Warm,
    /// Full cache drop + reallocation between generations.
    Cold,
}

impl CacheState {
    fn label(self) -> &'static str {
        match self {
            CacheState::Plasma => "plasma",
            CacheState::Hot => "hot",
            CacheState::Warm => "warm",
            CacheState::Cold => "cold",
        }
    }
}

/// Run E2E transformer generation benchmarks through different cache thermal states.
///
/// For each cache state, generates a fixed number of tokens and measures throughput.
/// Returns one `BenchResult` per (game_profile, cache_state) combination.
///
/// Game profiles use different token counts to represent different game types:
/// - Sudoku: short, structured (16 tokens)
/// - Go: medium, strategic (32 tokens)
/// - Monopoly: long, multi-player (64 tokens)
/// - Bomber: real-time, bursty (8 tokens)
pub fn bench_e2e_game_timing(_config: &Config) -> Vec<BenchResult> {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let game_profiles: &[(&str, usize)] =
        &[("Sudoku", 16), ("Go", 32), ("Monopoly", 64), ("Bomber", 8)];

    let cache_states = [
        CacheState::Plasma,
        CacheState::Hot,
        CacheState::Warm,
        CacheState::Cold,
    ];

    let warmup_rounds = 3;
    let bench_rounds = 20;

    let mut results = Vec::new();

    println!("\n🎮 E2E Game Timing (plasma/hot/warm/cold)...");

    for &(game_name, n_tokens) in game_profiles {
        println!("   {game_name} ({n_tokens} tok)...");

        for &state in &cache_states {
            // Warmup
            for _ in 0..warmup_rounds {
                let mut ctx = ForwardContext::new(&config);
                let mut cache = MultiLayerKVCache::new(&config);
                let mut rng_w = Rng::new(42);
                let mut tokens = Vec::with_capacity(n_tokens);
                generate_into(
                    &mut ctx,
                    &mut cache,
                    &weights,
                    &config,
                    &mut rng_w,
                    n_tokens,
                    &mut tokens,
                );
            }

            // Benchmark
            let start = Instant::now();
            for i in 0..bench_rounds {
                match state {
                    CacheState::Cold => {
                        // Drop and reallocate everything
                        let mut ctx = ForwardContext::new(&config);
                        let mut cache = MultiLayerKVCache::new(&config);
                        let mut rng_b = Rng::new(42 + i);
                        let mut tokens = Vec::with_capacity(n_tokens);
                        generate_into(
                            &mut ctx,
                            &mut cache,
                            &weights,
                            &config,
                            &mut rng_b,
                            n_tokens,
                            &mut tokens,
                        );
                    }
                    CacheState::Plasma => {
                        // First generation with fresh alloc (but not dropped between iterations)
                        let mut ctx = ForwardContext::new(&config);
                        let mut cache = MultiLayerKVCache::new(&config);
                        let mut rng_b = Rng::new(42 + i);
                        let mut tokens = Vec::with_capacity(n_tokens);
                        cache.reset();
                        generate_into(
                            &mut ctx,
                            &mut cache,
                            &weights,
                            &config,
                            &mut rng_b,
                            n_tokens,
                            &mut tokens,
                        );
                    }
                    CacheState::Warm => {
                        // Reuse context, reset cache (allocator hot)
                        static CTX_WARM: std::sync::OnceLock<SyncCell<Option<ForwardContext>>> =
                            std::sync::OnceLock::new();
                        static CACHE_WARM: std::sync::OnceLock<
                            SyncCell<Option<MultiLayerKVCache>>,
                        > = std::sync::OnceLock::new();
                        // SAFETY: benchmark is single-threaded
                        let ctx_cell = CTX_WARM
                            .get_or_init(|| SyncCell::new(Some(ForwardContext::new(&config))));
                        let cache_cell = CACHE_WARM
                            .get_or_init(|| SyncCell::new(Some(MultiLayerKVCache::new(&config))));
                        unsafe {
                            let ctx = (*ctx_cell.get()).as_mut().unwrap();
                            let cache = (*cache_cell.get()).as_mut().unwrap();
                            cache.reset();
                            let mut rng_b = Rng::new(42 + i);
                            let mut tokens = Vec::with_capacity(n_tokens);
                            generate_into(
                                ctx,
                                cache,
                                &weights,
                                &config,
                                &mut rng_b,
                                n_tokens,
                                &mut tokens,
                            );
                        }
                    }
                    CacheState::Hot => {
                        // Reuse context and cache (tokens still present)
                        static CTX_HOT: std::sync::OnceLock<SyncCell<Option<ForwardContext>>> =
                            std::sync::OnceLock::new();
                        static CACHE_HOT: std::sync::OnceLock<SyncCell<Option<MultiLayerKVCache>>> =
                            std::sync::OnceLock::new();
                        unsafe {
                            let ctx_cell = CTX_HOT
                                .get_or_init(|| SyncCell::new(Some(ForwardContext::new(&config))));
                            let cache_cell = CACHE_HOT.get_or_init(|| {
                                SyncCell::new(Some(MultiLayerKVCache::new(&config)))
                            });
                            let ctx = (*ctx_cell.get()).as_mut().unwrap();
                            let cache = (*cache_cell.get()).as_mut().unwrap();
                            // Don't reset cache — simulate hot path with accumulated state
                            let mut rng_b = Rng::new(42 + i);
                            let mut tokens = Vec::with_capacity(n_tokens);
                            generate_into(
                                ctx,
                                cache,
                                &weights,
                                &config,
                                &mut rng_b,
                                n_tokens,
                                &mut tokens,
                            );
                        }
                    }
                }
            }
            let elapsed = start.elapsed();

            let total_tokens = bench_rounds as f64 * n_tokens as f64;
            let tps = total_tokens / elapsed.as_secs_f64();
            let us_per_token = elapsed.as_micros() as f64 / (bench_rounds as f64 * n_tokens as f64);

            let state_label = state.label();

            // Color coding: plasma=white-blue, hot=red, warm=orange, cold=blue
            let color = match state {
                CacheState::Plasma => (173, 216, 230), // light blue
                CacheState::Hot => (255, 69, 0),       // red-orange
                CacheState::Warm => (255, 165, 0),     // orange
                CacheState::Cold => (70, 130, 180),    // steel blue
            };

            results.push(BenchResult {
                label: format!("{game_name}/{state_label}"),
                throughput: tps,
                time_per_step_us: us_per_token,
                avg_acceptance_len: n_tokens as f64,
                color,
                category: BenchCategory::E2EGame,
                feature_dim: "Game".into(),
            });
        }
    }

    // Print summary table
    println!(
        "\n   {:<20} {:>12} {:>12} {:>10}",
        "Profile/State", "tok/s", "μs/tok", "Tokens"
    );
    println!("   {}", "-".repeat(56));
    for r in &results {
        println!(
            "   {:<20} {:>12.0} {:>12.2} {:>10.0}",
            r.label, r.throughput, r.time_per_step_us, r.avg_acceptance_len,
        );
    }

    results
}
