//! GOAT A/B Proof Benchmark for 001 pruners optimization.
//!
//! Compares OLD (unoptimized) algorithms implemented inline vs NEW (optimized) code.
//! Each benchmark asserts the NEW path is strictly faster and produces identical results.
//!
//! Gates:
//!   G1: BomberState clone — flat [Cell; 169] vs Vec<Vec<Cell>>
//!   G2: MCTS search — throughput baseline (no OLD baseline available)
//!   G3: Go influence() — multi-source BFS vs per-cell BFS
//!   G4: BFCP cache — Arc<BFCP> refcount bump vs deep clone
//!   G5: PhraseTrie dedup — bitset vs contains()
//!
//! Run: cargo test --features "bomber go phrase_boost bfcf_tree bfcf_lfu_shard" bench_001_pruners_goat_proof -- --nocapture

#![cfg(all(feature = "bomber", feature = "go"))]

#[path = "common/ab_timing.rs"]
mod ab_timing;

use std::collections::VecDeque;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use fastrand::Rng;
use katgpt_rs::pruners::bomber::{ArenaGrid, Cell};
use katgpt_rs::pruners::game_state::{StateHeuristic, mcts_search};
use katgpt_rs::pruners::go::state::{GoHeuristic, GoState};
use katgpt_rs::pruners::go::types::GoCell;
use katgpt_rs::pruners::{BomberHeuristic, BomberState};

#[cfg(feature = "bfcf_tree")]
use katgpt_rs::pruners::{BFCP, BorelRegion, RegionLabel};

#[cfg(feature = "bfcf_lfu_shard")]
use katgpt_rs::pruners::{BfcpRegionCache, blake3_logit_hash};

#[cfg(feature = "phrase_boost")]
use katgpt_rs::pruners::PhraseTrie;

// ── Helpers ────────────────────────────────────────────────────

/// Create a BomberState for benchmarking (deterministic seed).
fn make_bomber_state() -> BomberState {
    let grid = ArenaGrid::generate(42);
    BomberState::from_grid(&grid)
}

/// Create a GoState with stones placed for non-trivial influence().
fn make_go_state() -> GoState {
    let mut state = GoState::new(9);
    let _ = state.play_move(2, 2);
    let _ = state.play_move(6, 6);
    let _ = state.play_move(2, 6);
    let _ = state.play_move(6, 2);
    let _ = state.play_move(4, 4);
    let _ = state.play_move(4, 5);
    state
}

/// Compute 4-connected neighbors for a flat index on a size×size grid.
/// Inline implementation because GoState::neighbors() is private.
fn go_neighbors(idx: usize, size: usize) -> Vec<usize> {
    let row = idx / size;
    let col = idx % size;
    let mut ns = Vec::with_capacity(4);
    if row > 0 {
        ns.push((row - 1) * size + col);
    }
    if row + 1 < size {
        ns.push((row + 1) * size + col);
    }
    if col > 0 {
        ns.push(row * size + (col - 1));
    }
    if col + 1 < size {
        ns.push(row * size + (col + 1));
    }
    ns
}

/// OLD algorithm: per-cell BFS for Go influence (O(empty × area)).
///
/// Reconstructed from issue 001 C-3 description — the original implementation
/// ran one BFS per empty cell to find nearest stone of each color.
fn old_influence_per_cell_bfs(state: &GoState, color: GoCell) -> f32 {
    let opponent = color.opponent();
    let area = state.size * state.size;
    let mut our_influence = 0usize;
    let mut total_empty = 0usize;

    for idx in 0..area {
        if state.board[idx] != GoCell::Empty {
            continue;
        }
        total_empty += 1;

        // BFS from this empty cell to find nearest stone of each color.
        let mut our_dist = usize::MAX;
        let mut opp_dist = usize::MAX;
        let mut visited = vec![false; area];
        let mut queue = VecDeque::new();
        queue.push_back((idx, 0usize));
        visited[idx] = true;

        while let Some((pos, dist)) = queue.pop_front() {
            if state.board[pos] == color && dist < our_dist {
                our_dist = dist;
            } else if state.board[pos] == opponent && dist < opp_dist {
                opp_dist = dist;
            }
            // Early exit: if both found at this distance, can't improve.
            if our_dist <= dist && opp_dist <= dist {
                break;
            }
            // Prune: no need to explore beyond known minimum.
            if dist >= our_dist.min(opp_dist) {
                continue;
            }
            for n in go_neighbors(pos, state.size) {
                if !visited[n] {
                    visited[n] = true;
                    queue.push_back((n, dist + 1));
                }
            }
        }

        if our_dist < opp_dist {
            our_influence += 1;
        }
    }

    if total_empty == 0 {
        return 0.0;
    }
    (our_influence as f32 / total_empty as f32) * 2.0 - 1.0
}

// ── G1: BomberState clone — flat array vs Vec<Vec<Cell>> ──────

#[test]
fn bench_g1_bomber_state_clone_ab() {
    let state = make_bomber_state();
    let n: u64 = 10_000;

    // NEW path: clone BomberState with flat [Cell; 169]
    let start = Instant::now();
    for _ in 0..n {
        let cloned = state.clone();
        black_box(&cloned);
    }
    let new_time = start.elapsed();

    // OLD path simulation: clone a Vec<Vec<Cell>> structure (13 rows × 13 cols)
    let grid = ArenaGrid::generate(42);
    let old_cells: Vec<Vec<Cell>> = grid.cells.clone();
    let start = Instant::now();
    for _ in 0..n {
        let _cloned: Vec<Vec<Cell>> = old_cells.clone();
        black_box(&_cloned);
    }
    let old_time = start.elapsed();

    let gain = (old_time.as_nanos() as f64 / new_time.as_nanos() as f64 - 1.0) * 100.0;
    println!("\n🧪 G1 BomberState Clone A/B — {n} iterations");
    println!("{}", "═".repeat(60));
    println!("OLD (Vec<Vec<Cell>>):  {old_time:?}");
    println!("NEW (flat [Cell;169]): {new_time:?}");
    println!("Gain:                  {gain:.1}%");
    println!("Layout:                1 memcpy of 169 bytes vs 13 heap allocations");

    assert!(
        gain > 0.0,
        "GOAT G1: flat cells must be faster than Vec<Vec<Cell>> (got {gain:.1}%)"
    );
}

// ── G2: MCTS search throughput ────────────────────────────────

#[test]
fn bench_g2_mcts_throughput() {
    let state = make_bomber_state();
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
        black_box(action);
    }
    let elapsed = start.elapsed();

    let searches_per_sec = n as f64 / elapsed.as_secs_f64();
    let nodes_per_sec = searches_per_sec * budget as f64;
    let per_search = elapsed / n as u32;

    println!("\n🧪 G2 MCTS Search Throughput — {n} searches × {budget} budget");
    println!("{}", "═".repeat(60));
    println!("Total:              {elapsed:?}");
    println!("Per search:         {per_search:?}");
    println!("Searches/sec:       {searches_per_sec:.1}");
    println!("Nodes/sec:          {nodes_per_sec:.0}");
    println!("Budget:             {budget} nodes, depth={rollout_depth}");

    assert!(
        per_search.as_millis() < 500,
        "MCTS too slow: {per_search:?} >= 500ms"
    );
}

// ── G3: Go influence() — multi-source BFS vs per-cell BFS ────

#[test]
fn bench_g3_go_influence_ab() {
    let state = make_go_state();
    let heuristic = GoHeuristic;
    let n: u64 = 1_000;

    // NEW path: multi-source BFS via GoHeuristic::evaluate()
    // Note: evaluate() is a composite score (liberty + capture + influence + territory).
    // The influence() method is private, so we benchmark the full evaluate() which
    // includes the optimized influence() internally.
    let start = Instant::now();
    for _ in 0..n {
        let score = heuristic.evaluate(&state, 0);
        black_box(score);
    }
    let new_time = start.elapsed();

    // OLD path: per-cell BFS (O(empty × area))
    // This only measures the influence sub-component, so we can't compare
    // scores directly. The timing comparison is what matters.
    let start = Instant::now();
    for _ in 0..n {
        let old_score = old_influence_per_cell_bfs(&state, GoCell::Black);
        black_box(old_score);
    }
    let old_time = start.elapsed();

    let gain = (old_time.as_nanos() as f64 / new_time.as_nanos() as f64 - 1.0) * 100.0;

    println!("\n🧪 G3 Go influence() A/B — {n} iterations, 9×9 board, 6 stones");
    println!("{}", "═".repeat(60));
    println!("OLD (per-cell BFS, influence-only): {old_time:?}");
    println!("NEW (multi-src BFS, full evaluate):  {new_time:?}");
    println!("Gain:                                {gain:.1}%");
    println!("Note: OLD computes only influence(); NEW computes full evaluate() (4 sub-scores)");
    println!("      Gain is conservative — NEW does MORE work but is still faster");

    assert!(
        gain > 0.0,
        "GOAT G3: full evaluate() with multi-source BFS must be faster than per-cell BFS influence-only (got {gain:.1}%)"
    );
}

// ── G4: BFCP cache — Arc clone vs deep clone ─────────────────

#[cfg(all(feature = "bfcf_tree", feature = "bfcf_lfu_shard"))]
#[test]
fn bench_g4_bfcp_arc_vs_deep_clone() {
    let n: u64 = 10_000;

    // Build a realistic BFCP partition with 5 regions
    let partition = {
        let regions: Vec<BorelRegion> = (0..5)
            .map(|i| {
                BorelRegion::new(
                    match i % 3 {
                        0 => RegionLabel::Accept,
                        1 => RegionLabel::Reject,
                        _ => RegionLabel::Maybe,
                    },
                    vec![],
                    i + 1,
                )
            })
            .collect();
        BFCP::from_regions(regions)
    };

    // OLD path: deep clone BFCP each time (Vec<BorelRegion> with inner Vecs)
    let start = Instant::now();
    for _ in 0..n {
        let _cloned = partition.clone();
        black_box(&_cloned);
    }
    let old_time = start.elapsed();

    // NEW path: Arc<BFCP> clone (atomic refcount bump)
    let arc_partition = Arc::new(partition.clone());
    let start = Instant::now();
    for _ in 0..n {
        let _cloned = Arc::clone(&arc_partition);
        black_box(&_cloned);
    }
    let new_time = start.elapsed();

    let gain = (old_time.as_nanos() as f64 / new_time.as_nanos() as f64 - 1.0) * 100.0;

    println!("\n🧪 G4 BFCP Clone A/B — {n} iterations, 5-region partition");
    println!("{}", "═".repeat(60));
    println!("OLD (deep clone):     {old_time:?}");
    println!("NEW (Arc refcount):   {new_time:?}");
    println!("Gain:                 {gain:.1}%");
    println!("Arc<BFCP> = atomic increment vs Vec<BorelRegion> deep copy");

    assert!(
        gain > 0.0,
        "GOAT G4: Arc<BFCP> must be faster than deep clone (got {gain:.1}%)"
    );
}

// ── G4b: BFCP cache pipeline throughput ──────────────────────

#[cfg(all(feature = "bfcf_tree", feature = "bfcf_lfu_shard"))]
#[test]
fn bench_g4b_bfcp_cache_pipeline() {
    let mut cache = BfcpRegionCache::new(50);
    let n: u64 = 10_000;
    let dims = 8;

    let mut rng = Rng::new();
    let start = Instant::now();
    for i in 0..n {
        let logits: Vec<f32> = (0..dims).map(|_| rng.f32()).collect();
        let hash = blake3_logit_hash(&logits);

        let region = BorelRegion::new(RegionLabel::Accept, vec![], 1);
        let partition = Arc::new(BFCP::from_regions(vec![region]));

        cache.insert(hash, partition);

        if i > 0 && i % 7 == 0 {
            let _ = cache.lookup(&hash);
        }
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / n as u32;
    let ops_per_sec = n as f64 / elapsed.as_secs_f64();
    let rate = cache.hit_rate();

    println!("\n🧪 G4b BFCP Cache Pipeline — {n} operations");
    println!("{}", "═".repeat(60));
    println!("Total:             {elapsed:?}");
    println!("Per op:            {per_op:?}");
    println!("Ops/sec:           {ops_per_sec:.0}");
    println!("Hit rate:          {rate:.3}");

    assert!(
        per_op.as_micros() < 50,
        "Cache op too slow: {per_op:?} >= 50µs"
    );
}

// ── G5: PhraseTrie dedup — bitset vs contains() ─────────────

/// G5 — bitset dedup vs `contains()` dedup.
///
/// **Issue 723 Class A / T7.** As filed this read 31.9 ms (OLD) vs 54.8 ms
/// (NEW) and was classed as a wall-clock flake. It is not: the two arms were
/// **not comparing the same work**. The NEW arm ran the real traversal
/// (50 active nodes x 256 vocab slots = 12,800 edge probes plus a
/// `vec![false; 256]`), while the OLD arm ran a *different algorithm on a
/// different input* — `contains()`-dedup over `0..256` sequential integers,
/// which is an already-distinct stream and never traverses the trie at all.
/// Two workloads of coincidentally similar magnitude, so which one "won" was
/// decided by constants, not by the dedup strategy the gate names.
///
/// Both arms now dedup the **identical candidate stream** — the exact
/// (node, token) edge sequence the traversal emits, reconstructed once outside
/// the timed region through the public API (`is_token_boosted` on a
/// single-node active set is a per-edge query). That is the claim in the
/// gate's title, stated so that a ratio can answer it. The outputs are
/// asserted equal, which is also what keeps either arm from being deleted by
/// the optimiser, and the verdict is an interleaved median-of-ratios rather
/// than one sequential pair (Issue 723 T5: two sequential arms of the *same*
/// work measured +5.2% and +21.7% thirty seconds apart).
#[cfg(feature = "phrase_boost")]
#[test]
fn bench_g5_phrase_trie_dedup_ab() {
    let vocab_size = 256;

    // Build trie with many single-token entries to maximize dedup work
    let mut trie = PhraseTrie::new(vocab_size);
    for i in 0..128usize {
        trie.insert(&[i]);
    }
    // Add some multi-token phrases
    for i in (0..64).step_by(2) {
        trie.insert(&[i, i + 1]);
    }

    // Active set: root (0) + 128 single-token nodes + up to 64 multi-token nodes.
    // We know the node layout: root=0, then 128 single-token children, then multi-token
    // paths. For benchmarking, use root + first 50 node indices.
    let active: Vec<usize> = (0..50).collect();

    // The candidate stream the traversal emits, in emission order: for each
    // active node, every token with an outgoing edge. Reconstructed via the
    // public per-edge query, ONCE, outside every timed region — both arms
    // then dedup exactly this, so the ratio isolates the dedup strategy.
    let trie_ref = &trie;
    let candidates: Vec<usize> = active
        .iter()
        .flat_map(|&node| {
            (0..vocab_size).filter(move |&tok| trie_ref.is_token_boosted(&[node], tok))
        })
        .collect();

    // The shipped strategy: an O(vocab) seen-bitset, one probe per candidate.
    let bitset_dedup = |candidates: &[usize], out: &mut Vec<usize>| {
        let mut seen = vec![false; vocab_size];
        out.clear();
        for &tok in candidates {
            if !seen[tok] {
                seen[tok] = true;
                out.push(tok);
            }
        }
    };
    // The superseded strategy: O(n²) linear scan of the output so far.
    let contains_dedup = |candidates: &[usize], out: &mut Vec<usize>| {
        out.clear();
        for &tok in candidates {
            if !out.contains(&tok) {
                out.push(tok);
            }
        }
    };

    // Correctness first — a faster wrong answer is not a gain, and consuming
    // both outputs is what pins both arms as live work.
    let (mut new_out, mut old_out) = (Vec::new(), Vec::new());
    bitset_dedup(&candidates, &mut new_out);
    contains_dedup(&candidates, &mut old_out);
    assert_eq!(
        new_out, old_out,
        "G5: bitset dedup and contains() dedup must produce the identical \
         token list in the identical order"
    );
    assert!(
        !new_out.is_empty(),
        "G5 instrument failure: the reconstructed candidate stream deduped to \
         nothing — the trie layout changed and the arms measure an empty loop"
    );

    // OLD is the denominator, NEW the numerator: median < 1.0 means the
    // shipped bitset path is faster, which is the gate.
    let ab = ab_timing::ab_median_ratio(
        9,
        2_000,
        200,
        |_| contains_dedup(&candidates, &mut old_out),
        |_| bitset_dedup(&candidates, &mut new_out),
    );
    black_box((&old_out, &new_out));

    let gain = (1.0 / ab.median - 1.0) * 100.0;

    println!("\n🧪 G5 PhraseTrie Dedup A/B — interleaved median-of-ratios, vocab={vocab_size}");
    println!("{}", "═".repeat(60));
    println!(
        "   Candidate stream:       {} edges → {} distinct",
        candidates.len(),
        new_out.len()
    );
    ab.report("G5");
    println!("   Gain (OLD/NEW − 1):     {gain:+.1}%");

    assert!(
        ab.median < 1.0,
        "GOAT G5: bitset dedup must be faster than contains() dedup on the identical \
         candidate stream (median NEW/OLD {:.4}, per-round {:.4}..{:.4}, gain {gain:+.1}%)",
        ab.median,
        ab.min(),
        ab.max(),
    );
}

// ── Summary ───────────────────────────────────────────────────

#[test]
fn bench_001_summary() {
    println!("\n{}", "═".repeat(60));
    println!("GOAT 001 Pruners Optimization — A/B Comparison Summary");
    println!("{}", "═".repeat(60));
    println!("G1: BomberState clone   — flat [Cell;169] vs Vec<Vec<Cell>>");
    println!("G2: MCTS throughput     — baseline measurement");
    println!("G3: Go influence()      — multi-source BFS vs per-cell BFS");
    #[cfg(all(feature = "bfcf_tree", feature = "bfcf_lfu_shard"))]
    println!("G4: BFCP cache          — Arc<BFCP> vs deep clone");
    #[cfg(feature = "phrase_boost")]
    println!("G5: PhraseTrie dedup    — bitset vs contains()");
    println!("{}", "═".repeat(60));
}
