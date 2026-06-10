//! GOAT benchmark for 001 pruners optimization.
//!
//! Measures performance gains from:
//!   C-1: Flat array BomberState (1 memcpy vs 13 heap allocations)
//!   C-2: MCTS search throughput with BomberHeuristic
//!   C-3: Go influence multi-source BFS
//!   H-1: BFCP cache pipeline with Arc<BFCP> clones
//!   H-2: BanditPruner scratch buffer reuse
//!
//! Run with: cargo test --features "bomber go" bench_001_pruners_goat -- --nocapture

#[cfg(all(feature = "bomber", feature = "go"))]
use std::time::Instant;

#[cfg(all(feature = "bomber", feature = "go"))]
use katgpt_rs::pruners::bomber::ArenaGrid;
#[cfg(all(feature = "bomber", feature = "go"))]
use katgpt_rs::pruners::game_state::{BomberHeuristic, BomberState, StateHeuristic, mcts_search};
#[cfg(all(feature = "bomber", feature = "go"))]
use katgpt_rs::pruners::go::state::{GoHeuristic, GoState};
#[cfg(all(feature = "bomber", feature = "go"))]
use katgpt_rs::pruners::{
    BFCP, BanditPruner, BanditStrategy, BfcpRegionCache, BorelRegion, RegionLabel,
    blake3_logit_hash,
};
#[cfg(all(feature = "bomber", feature = "go"))]
use katgpt_rs::speculative::ScreeningPruner;

#[cfg(all(feature = "bomber", feature = "go"))]
use fastrand::Rng;

// ── Benchmark 1: BomberState clone throughput (C-1) ──────────

#[cfg(all(feature = "bomber", feature = "go"))]
#[test]
fn bench_001_bomber_state_clone() {
    let grid = ArenaGrid::generate(42);
    let state = BomberState::from_grid(&grid);
    let n: u64 = 10_000;

    let start = Instant::now();
    for _ in 0..n {
        let cloned = state.clone();
        std::hint::black_box(&cloned);
    }
    let elapsed = start.elapsed();
    let per_clone = elapsed / n as u32;
    let clones_per_sec = n as f64 / elapsed.as_secs_f64();
    let ns_per_clone = elapsed.as_nanos() as f64 / n as f64;

    println!("\n🧪 Benchmark 1: BomberState Clone (C-1) — {n} iterations");
    println!("{}", "═".repeat(60));
    println!("Total:        {elapsed:?}");
    println!("Per clone:    {per_clone:?}");
    println!("Clones/sec:   {clones_per_sec:.0}");
    println!("ns/clone:     {ns_per_clone:.1}");
    println!("Layout:       flat [Cell; 169] ≈ 1 memcpy of 169 bytes");

    // Flat array: 169 bytes memcpy should be < 5µs even on slow machines
    assert!(
        per_clone.as_micros() < 5,
        "Clone too slow: {per_clone:?} >= 5µs"
    );
}

// ── Benchmark 2: MCTS search throughput (C-1 + C-2) ──────────

#[cfg(all(feature = "bomber", feature = "go"))]
#[test]
fn bench_001_mcts_search_throughput() {
    let grid = ArenaGrid::generate(42);
    let state = BomberState::from_grid(&grid);
    let heuristic = BomberHeuristic;
    let n: u64 = 100;
    let budget = 500;
    let rollout_depth = 10;

    let mut rng = Rng::new();
    let start = Instant::now();
    for _ in 0..n {
        let action = mcts_search(
            &state,
            0,
            budget,
            rollout_depth,
            &|s, pid| heuristic.evaluate(s, pid),
            &mut rng,
        );
        std::hint::black_box(action);
    }
    let elapsed = start.elapsed();
    let per_search = elapsed / n as u32;
    let searches_per_sec = n as f64 / elapsed.as_secs_f64();
    let nodes_per_sec = searches_per_sec * budget as f64;

    println!("\n🧪 Benchmark 2: MCTS Search Throughput (C-1 + C-2) — {n} iterations");
    println!("{}", "═".repeat(60));
    println!("Total:          {elapsed:?}");
    println!("Per search:     {per_search:?}");
    println!("Budget:         {budget} nodes/search, depth={rollout_depth}");
    println!("Searches/sec:   {searches_per_sec:.1}");
    println!("Nodes/sec:      {nodes_per_sec:.0}");

    // Each search builds a tree of ~budget nodes via advance().
    // With 500 budget, expect < 500ms per search.
    assert!(
        per_search.as_millis() < 500,
        "MCTS too slow: {per_search:?} >= 500ms"
    );
}

// ── Benchmark 3: Go influence() / evaluate() (C-3) ──────────

#[cfg(all(feature = "bomber", feature = "go"))]
#[test]
fn bench_001_go_heuristic_evaluate() {
    let mut state = GoState::new(9);
    // Place some stones to make influence() non-trivial
    let _ = state.play_move(2, 2); // Black
    let _ = state.play_move(6, 6); // White
    let _ = state.play_move(2, 6); // Black
    let _ = state.play_move(6, 2); // White
    let _ = state.play_move(4, 4); // Black
    let _ = state.play_move(4, 5); // White

    let heuristic = GoHeuristic;
    let n: u64 = 1_000;

    let start = Instant::now();
    for _ in 0..n {
        let score = heuristic.evaluate(&state, 0);
        std::hint::black_box(score);
    }
    let elapsed = start.elapsed();
    let per_eval = elapsed / n as u32;
    let us_per_eval = elapsed.as_micros() as f64 / n as f64;

    println!("\n🧪 Benchmark 3: GoHeuristic::evaluate() (C-3) — {n} iterations");
    println!("{}", "═".repeat(60));
    println!("Total:          {elapsed:?}");
    println!("Per evaluate:   {per_eval:?}");
    println!("µs/evaluate:    {us_per_eval:.2}");
    println!("Board:          9×9 with 6 stones");
    println!("Method:         multi-source BFS for influence()");

    // Multi-source BFS on 9×9 should be < 200µs per evaluate
    assert!(
        per_eval.as_micros() < 200,
        "Evaluate too slow: {per_eval:?} >= 200µs"
    );
}

// ── Benchmark 4: BFCP cache pipeline (H-1) ──────────

/// Minimal ScreeningPruner for BanditPruner benchmark (H-2).
#[cfg(all(feature = "bomber", feature = "go"))]
struct UniformPruner;

#[cfg(all(feature = "bomber", feature = "go"))]
impl ScreeningPruner for UniformPruner {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        0.5
    }
}

#[cfg(all(feature = "bomber", feature = "go"))]
#[test]
fn bench_001_bfcp_cache_pipeline() {
    let mut cache = BfcpRegionCache::new(50);
    let n: u64 = 10_000;
    let dims = 8;

    let mut rng = Rng::new();
    let start = Instant::now();
    for i in 0..n {
        // Generate 8-dim logit vector
        let logits: Vec<f32> = (0..dims).map(|_| rng.f32()).collect();
        let hash = blake3_logit_hash(&logits);

        // Create a minimal BFCP partition
        let region = BorelRegion::new(RegionLabel::Accept, vec![], 1);
        let partition = std::sync::Arc::new(BFCP::from_regions(vec![region]));

        cache.insert(hash, partition);

        // Lookup some previously inserted entries
        if i > 0 && i % 7 == 0 {
            let _ = cache.lookup(&hash);
        }
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / n as u32;
    let ops_per_sec = n as f64 / elapsed.as_secs_f64();

    let rate = cache.hit_rate();
    println!("\n🧪 Benchmark 4: BFCP Cache Pipeline (H-1) — {n} operations");
    println!("{}", "═".repeat(60));
    println!("Total:          {elapsed:?}");
    println!("Per op:         {per_op:?}");
    println!("Ops/sec:        {ops_per_sec:.0}");
    println!("Hit rate:       {rate:.3}");
    println!("Cache capacity: 50, logits: {dims}-dim");
    println!("Arc<BFCP> clones should be cheap (refcount bump)");

    // Each op is insert + occasional lookup with Arc clone.
    // Should be < 50µs per op (BLAKE3 hash + HashMap insert).
    assert!(
        per_op.as_micros() < 50,
        "Cache op too slow: {per_op:?} >= 50µs"
    );
}

// ── Benchmark 5: BanditPruner scratch buffer reuse (H-2) ──────────

#[cfg(all(feature = "bomber", feature = "go"))]
#[test]
fn bench_001_bandit_pruner_relevance() {
    let num_arms = 64;
    let pruner = BanditPruner::new(UniformPruner, BanditStrategy::Ucb1, num_arms);
    let n: u64 = 10_000;

    let start = Instant::now();
    for i in 0..n {
        let token_idx = (i as usize) % num_arms;
        let score = pruner.relevance(0, token_idx, &[]);
        std::hint::black_box(score);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / n as u32;
    let calls_per_sec = n as f64 / elapsed.as_secs_f64();

    println!("\n🧪 Benchmark 5: BanditPruner relevance() (H-2) — {n} calls");
    println!("{}", "═".repeat(60));
    println!("Total:          {elapsed:?}");
    println!("Per call:       {per_call:?}");
    println!("Calls/sec:      {calls_per_sec:.0}");
    println!("Arms:           {num_arms}");
    println!("Strategy:       UCB1");
    println!("Scratch buffers pre-allocated (no per-call Vec alloc)");

    // UCB1 relevance is O(1) per arm with pre-allocated scratch buffers.
    // Should be < 10µs per call.
    assert!(
        per_call.as_micros() < 10,
        "Relevance too slow: {per_call:?} >= 10µs"
    );
}
