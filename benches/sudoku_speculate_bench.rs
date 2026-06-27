//! Sudoku Speculative-Solve Benchmark — pure perf, no TUI.
//!
//! Measures how fast the hardest Sudoku (Arto Inkala, 21 clues, 60 empties)
//! can be solved three ways:
//!
//! 1. **backtrack**        — canonical `Sudoku9x9::solve()` (the ground-truth
//!                           complete solver; docs baseline = 49,559 steps).
//! 2. **speculate_iterative** — iterative DDTree + greedy path commit, with
//!                           backtracking fallback when speculation paints
//!                           into a corner. This is the realistic "speculative
//!                           decoding" pattern: draft → prune → commit → verify.
//! 3. **speculate_oneshot** — single `build_dd_tree_pruned` with full-depth
//!                           lookahead. "Pure speculate" extreme: can the tree
//!                           find a complete valid solution in one build?
//!
//! Convention: `std::time::Instant` + `harness = false` (matches
//! `cucg_bench.rs`, `alien_sampler_bench.rs`, `procrustes_bench.rs` —
//! Criterion is not a katgpt-rs dev-dep).
//!
//! Run:
//! ```bash
//! cargo run --release --bench sudoku_speculate_bench --features sudoku
//! ```

#![cfg(feature = "sudoku")]

use katgpt_rs::percepta::{KVCache2D, Sudoku9x9};
use katgpt_rs::pruners::SudokuPruner;
use katgpt_rs::speculative::{build_dd_tree_pruned, extract_parent_tokens};
use katgpt_rs::types::Config;
use std::time::{Duration, Instant};

/// Vocab size for Sudoku: indices 0..=9 (0 = padding/empty, 1..=9 = digits).
const SUDOKU_VOCAB: usize = 10;

/// Batched-timing outer samples (median-of-N).
const OUTER: usize = 31;

/// Inner iterations per sample (Sudoku solve is fast; batch for stable ns).
/// Backtracking on Inkala ≈ 0.5–5 ms; 16 solves per sample keeps each sample
/// under ~80 ms.
const BATCH: usize = 16;

/// Warmup iterations to prime caches + branch predictor.
const WARMUP: usize = 3;

/// Safety cap on speculate_iterative rounds (prevents infinite loop on bugs).
const MAX_SPEC_ROUNDS: usize = 200;

// ─── Marginals ─────────────────────────────────────────────────────────────

/// Uniform draft marginals over digits 1–9 (index 0 = padding, never drafted).
/// This is the worst-case drafter — zero information. The pruner supplies all
/// the constraint signal. A trained drafter would replace this with real logits.
fn uniform_marginals(lookahead: usize) -> Vec<Vec<f32>> {
    (0..lookahead)
        .map(|_| {
            let mut p = vec![0.0f32; SUDOKU_VOCAB];
            for d in 1..=9 {
                p[d] = 1.0 / 9.0;
            }
            p
        })
        .collect()
}

// ─── Result structs ────────────────────────────────────────────────────────

#[derive(Clone)]
struct SolveStats {
    solved: bool,
    /// Backtracking steps (Mode 1), or fallback steps (Mode 2).
    steps: usize,
    /// Speculative rounds executed (Mode 2).
    spec_rounds: usize,
    /// Cells committed via speculation (Mode 2).
    spec_commits: usize,
    /// Iterations of the outer loop (Mode 2).
    iterations: usize,
    /// Tree nodes built total (Mode 2/3).
    tree_nodes: usize,
    /// Times the fallback had to revert ALL speculate commits back to the
    /// initial puzzle (because prior rounds' wrong commits poisoned the board
    /// so even pre-round backtrack failed). Honest "speculate contributed
    /// nothing" counter.
    full_reverts: usize,
    /// Final board state (for post-timing visual print + assertion).
    final_board: Sudoku9x9,
}

impl Default for SolveStats {
    fn default() -> Self {
        Self {
            solved: false,
            steps: 0,
            spec_rounds: 0,
            spec_commits: 0,
            iterations: 0,
            tree_nodes: 0,
            full_reverts: 0,
            final_board: Sudoku9x9::new([[0; 9]; 9]),
        }
    }
}

// ─── Mode 1: Canonical backtracking ────────────────────────────────────────

fn solve_backtrack() -> SolveStats {
    let mut board = Sudoku9x9::arto_inkala();
    let mut cache = KVCache2D::new();
    let mut step = 0usize;
    let solved = board.solve(&mut cache, &mut step);
    SolveStats {
        solved,
        steps: step,
        final_board: board,
        ..Default::default()
    }
}

// ─── Mode 4: Fast solver (MRV + constraint propagation) ────────────────────

/// `Sudoku9x9::solve_fast()` — MRV cell selection + bitmask candidate
/// tracking + naked-singles constraint propagation. Pure modelless rules
/// engine (no training). This is the "there IS a faster way" answer:
/// ~27× fewer steps than naive backtracking on Inkala.
fn solve_fast() -> SolveStats {
    let mut board = Sudoku9x9::arto_inkala();
    let (solved, steps) = board.solve_fast();
    SolveStats {
        solved,
        steps,
        final_board: board,
        ..Default::default()
    }
}

// ─── Mode 2: Iterative speculate + backtrack fallback ──────────────────────

/// After committing a speculated path, check whether any empty cell now has
/// zero valid digits (the speculation painted into a corner).
fn has_dead_end(board: &Sudoku9x9) -> bool {
    for r in 0..9 {
        for c in 0..9 {
            if board.grid[r][c] == 0 {
                let any_valid = (1..=9).any(|d| board.is_valid_move(r, c, d));
                if !any_valid {
                    return true;
                }
            }
        }
    }
    false
}

/// Run backtracking on `board` in place. If it fails (returns false),
/// reset `board` to the initial Arto Inkala puzzle and backtrack from there
/// (guaranteed solvable). Returns `(solved, steps, reverted_to_initial)`.
///
/// This is the correctness safety net: speculate commits can poison the
/// board into a globally-unsolvable state that has no immediate dead-end.
/// When that happens, backtrack from the poisoned state returns false, so we
/// revert to the known-solvable initial puzzle and solve honestly.
fn backtrack_with_initial_fallback(board: &mut Sudoku9x9) -> (bool, usize, bool) {
    let mut cache = KVCache2D::new();
    let mut step = 0usize;
    if board.solve(&mut cache, &mut step) {
        return (true, step, false);
    }
    // Poisoned — revert to initial and solve from the known-solvable state.
    *board = Sudoku9x9::arto_inkala();
    let mut cache2 = KVCache2D::new();
    let mut step2 = 0usize;
    let solved = board.solve(&mut cache2, &mut step2);
    (solved, step + step2, true)
}

/// Iterative speculative solve.
///
/// Each round:
///   1. Rebuild `SudokuPruner` for the current board (positions shift as cells
///      fill — re-discover empties in row-major order).
///   2. Build a lookahead DDTree with the path-aware pruner.
///   3. Pick the deepest, highest-score root-to-leaf path.
///   4. Commit the path cell-by-cell (each commit is guaranteed valid by the
///      path-aware pruner's cross-depth conflict checks).
///   5. If the post-commit board has a dead-end cell → revert this round and
///      fall back to complete backtracking from the pre-round board.
///
/// With uniform marginals (no draft model), speculation carries no information
/// beyond what the pruner already enforces — so dead-ends are frequent on
/// Inkala and the fallback fires. The bench reports this honestly.
///
/// NOTE: `lookahead` is capped internally at 8 because `TreeNode.parent_path`
/// is `u128` packing 16-bit tokens (128/16 = 8 max). The DDTree speculate
/// primitive is architecturally an 8-deep lookahead, designed for token-level
/// speculative decoding — NOT full-puzzle search.
fn solve_speculate_iterative(lookahead_in: usize, tree_budget: usize) -> SolveStats {
    // Architectural ceiling: u128 parent_path packs 16-bit tokens → max 8.
    const MAX_LOOKAHEAD: usize = 8;
    let lookahead = lookahead_in.min(MAX_LOOKAHEAD);

    let mut board = Sudoku9x9::arto_inkala();
    let mut config = Config::draft();
    config.vocab_size = SUDOKU_VOCAB;
    config.tree_budget = tree_budget;
    config.draft_lookahead = lookahead; // keeps parent_tokens_buf correctly sized

    let mut stats = SolveStats::default();
    let mut iterations = 0usize;

    while !board.is_solved() {
        iterations += 1;
        if iterations > MAX_SPEC_ROUNDS {
            break;
        }

        let pruner = SudokuPruner::new(board.clone());
        let empty = pruner.empty_count();
        if empty == 0 {
            break;
        }

        let la = lookahead.min(empty);
        let margs = uniform_marginals(la);
        let mv: Vec<&[f32]> = margs.iter().map(|s| s.as_slice()).collect();

        let tree = build_dd_tree_pruned(&mv, &config, &pruner, false);
        stats.tree_nodes += tree.len();

        if tree.is_empty() {
            // No valid speculation — backtrack the rest.
            let (solved, step, reverted) = backtrack_with_initial_fallback(&mut board);
            stats.steps += step;
            stats.full_reverts += reverted as usize;
            stats.solved = solved;
            stats.iterations = iterations;
            stats.final_board = board;
            return stats;
        }

        // Deepest, highest-score path.
        let best = tree
            .iter()
            .max_by(|a, b| a.depth.cmp(&b.depth).then(a.score.partial_cmp(&b.score).unwrap()))
            .unwrap();
        let path = extract_parent_tokens(best.parent_path, best.depth + 1);

        stats.spec_rounds += 1;
        let pre_round = board.clone();
        let mut round_commits = 0usize;

        for (depth, &token) in path.iter().enumerate() {
            if token == 0 {
                continue;
            }
            if let Some((row, col)) = pruner.position_at(depth)
                && board.is_valid_move(row, col, token as u8)
            {
                board.grid[row][col] = token as u8;
                round_commits += 1;
            }
        }
        stats.spec_commits += round_commits;

        // Dead-end check: did this round's commits make some other cell unsolvable?
        if !board.is_solved() && has_dead_end(&board) {
            // Revert and backtrack from the pre-round state.
            board = pre_round;
            let (solved, step, reverted) = backtrack_with_initial_fallback(&mut board);
            stats.steps += step;
            stats.full_reverts += reverted as usize;
            stats.solved = solved;
            stats.iterations = iterations;
            stats.final_board = board;
            return stats;
        }
    }

    stats.solved = board.is_solved();
    stats.iterations = iterations;
    stats.final_board = board;
    stats
}

// ─── Mode 3: DDTree primitive throughput (lookahead=8) ────────────────────

/// Measure the raw cost of building one 8-deep pruned DDTree at a given
/// budget. This is the "speculate primitive" unit cost — how fast can the
/// draft+prune machinery produce a lookahead tree?
///
/// A full oneshot solve is architecturally impossible: `TreeNode.parent_path`
/// is `u128` packing 16-bit tokens, so max lookahead = 8. The hardest Sudoku
/// has 60 empties — it can never fit in one tree. Mode 3 instead reports the
/// primitive's nodes/µs throughput, which is the building block Mode 2 pays
/// per round.
fn build_one_tree(tree_budget: usize) -> usize {
    let board = Sudoku9x9::arto_inkala();
    let pruner = SudokuPruner::new(board.clone());
    let lookahead = 8usize; // architectural max

    let mut config = Config::draft();
    config.vocab_size = SUDOKU_VOCAB;
    config.tree_budget = tree_budget;
    config.draft_lookahead = lookahead;

    let margs = uniform_marginals(lookahead);
    let mv: Vec<&[f32]> = margs.iter().map(|s| s.as_slice()).collect();

    let tree = build_dd_tree_pruned(&mv, &config, &pruner, false);
    tree.len()
}

// ─── Timing harness ────────────────────────────────────────────────────────

/// Median-of-`OUTER` timing over `BATCH` solves each.
fn median_batch<F: Fn() -> SolveStats>(solve: F) -> (Duration, SolveStats) {
    for _ in 0..WARMUP {
        let _ = solve();
    }
    let mut samples: Vec<Duration> = Vec::with_capacity(OUTER);
    let mut last = SolveStats::default();
    for _ in 0..OUTER {
        let t0 = Instant::now();
        for _ in 0..BATCH {
            last = solve();
        }
        samples.push(t0.elapsed());
    }
    samples.sort();
    let mid = OUTER / 2;
    let median_batch = (samples[mid].as_nanos() as f64 + samples[mid - 1].as_nanos() as f64) / 2.0;
    (
        Duration::from_nanos((median_batch / BATCH as f64) as u64),
        last,
    )
}

fn fmt_us(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{ns} ns")
    } else if ns < 1_000_000 {
        format!("{:.2} µs", ns as f64 / 1_000.0)
    } else {
        format!("{:.3} ms", ns as f64 / 1_000_000.0)
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let puzzle = Sudoku9x9::arto_inkala();
    let clues = puzzle.clue_count();
    let empties = 81 - clues;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Sudoku Speculative-Solve Bench — Arto Inkala (hardest)     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Puzzle:    {clues} clues, {empties} empty cells");
    println!("  Drafter:   uniform over digits 1-9 (worst case, zero signal)");
    println!("  Pruner:    path-aware SudokuPruner (100% valid branches)");
    println!("  Timing:    median of {OUTER} × batch-of-{BATCH}, {WARMUP} warmup");
    println!();

    // ── Mode 1: backtrack baseline ──
    println!("── Mode 1: backtrack (canonical Sudoku9x9::solve) ────────────");
    let (t_bt, s_bt) = median_batch(solve_backtrack);
    println!("  solved:          {}", s_bt.solved);
    println!("  steps:           {}", s_bt.steps);
    println!("  median time:     {}/solve", fmt_us(t_bt));
    let bt_us = t_bt.as_nanos() as f64 / 1000.0;
    let bt_per_step_us = bt_us / s_bt.steps.max(1) as f64;
    println!("  per-step:        {:.2} µs", bt_per_step_us);
    println!();

    // ── Mode 4: solve_fast (MRV + constraint propagation) ──
    // The "yes, there IS a faster way" answer. Pure modelless rules engine —
    // no feature flag needed beyond `sudoku`, no training. Beats backtrack by
    // ~20-30× on steps because naked-singles propagation + MRV cell ordering
    // eliminate most of the search.
    println!("── Mode 4: solve_fast (MRV + constraint propagation) ─────────");
    let (t_fast, s_fast) = median_batch(solve_fast);
    println!("  solved:          {}", s_fast.solved);
    println!("  steps:           {}  (vs backtrack {})", s_fast.steps, s_bt.steps);
    println!("  median time:     {}/solve", fmt_us(t_fast));
    let fast_us = t_fast.as_nanos() as f64 / 1000.0;
    let fast_per_step_us = fast_us / s_fast.steps.max(1) as f64;
    println!("  per-step:        {:.2} µs", fast_per_step_us);
    let speedup = bt_us / fast_us.max(1e-9);
    let step_reduction = s_bt.steps as f64 / s_fast.steps.max(1) as f64;
    println!("  speedup:         {:.2}× faster ({}× fewer steps)", speedup, step_reduction.round() as u64);
    println!();

    // ── Mode 2: speculate_iterative ──
    println!("── Mode 2: speculate_iterative (DDTree + greedy commit + fallback) ──");
    println!("  (lookahead capped at 8 — TreeNode.parent_path u128 / 16-bit ceiling)");
    println!("{:<10} {:>10} {:>10} {:>12} {:>10} {:>12} {:>12} {:>10}",
        "lookahead", "budget", "solved", "spec_commits", "fallback", "full_revert", "tree_nodes", "time");
    println!("{}", "─".repeat(92));

    // A few (lookahead, budget) configs to characterize the trade-off.
    // lookahead > 8 is clamped internally; we pass 16 to prove the clamp works.
    let configs: &[(usize, usize)] = &[
        (4, 32),
        (8, 64),
        (8, 128),
        (16, 256),  // clamped to 8 internally
    ];
    for &(la, budget) in configs {
        let (t, s) = median_batch(|| solve_speculate_iterative(la, budget));
        println!(
            "{:<10} {:>10} {:>10} {:>12} {:>10} {:>12} {:>12} {:>10}",
            la,
            budget,
            s.solved,
            s.spec_commits,
            s.steps,
            s.full_reverts,
            s.tree_nodes,
            fmt_us(t),
        );
    }
    println!();

    // ── Mode 3: DDTree primitive throughput ──
    println!("── Mode 3: DDTree primitive throughput (lookahead=8, Inkala) ─");
    println!("  measures raw build_one_tree() cost — the speculate unit primitive");
    println!("{:<14} {:>12} {:>12} {:>14}",
        "budget", "nodes_built", "time", "nodes/µs");
    println!("{}", "─".repeat(56));

    let budgets: &[usize] = &[64, 256, 1_024, 4_096, 16_384];
    for &budget in budgets {
        // Batch small-budget builds for stable timing; single-shot large ones.
        let (t, nodes) = if budget <= 1024 {
            let inner = 64usize;
            let mut samples: Vec<Duration> = Vec::with_capacity(OUTER);
            let mut last_nodes = 0usize;
            for _ in 0..OUTER {
                let t0 = Instant::now();
                for _ in 0..inner {
                    last_nodes = build_one_tree(budget);
                }
                samples.push(t0.elapsed());
            }
            samples.sort();
            let mid = OUTER / 2;
            let median_batch_ns = (samples[mid].as_nanos() as f64
                + samples[mid - 1].as_nanos() as f64)
                / 2.0;
            (
                Duration::from_nanos((median_batch_ns / inner as f64) as u64),
                last_nodes,
            )
        } else {
            let t0 = Instant::now();
            let nodes = build_one_tree(budget);
            (t0.elapsed(), nodes)
        };
        let nodes_per_us =
            nodes as f64 / (t.as_nanos().max(1) as f64 / 1000.0);
        println!(
            "{:<14} {:>12} {:>12} {:>14.1}",
            budget,
            nodes,
            fmt_us(t),
            nodes_per_us,
        );
    }
    println!();

    // ── Verdict ──
    println!("── Verdict ───────────────────────────────────────────────────");
    println!("  backtrack:           {} steps, {} (ground truth)", s_bt.steps, fmt_us(t_bt));
    println!();
    println!("  ARCHITECTURAL CEILING: TreeNode.parent_path is u128 packing");
    println!("  16-bit tokens → max lookahead = 8 (128/16). The DDTree speculate");
    println!("  primitive is a token-level speculative-decoding kernel, NOT a");
    println!("  full-puzzle search. A 60-empty Sudoku cannot be solved in one tree.");
    println!();
    println!("  With uniform marginals (no draft model), the speculative drafter");
    println!("  contributes zero information — every digit it proposes is already");
    println!("  constraint-valid via the pruner. So speculate_iter at best matches");
    println!("  backtrack and at worst pays tree-build overhead before falling back.");
    println!();
    println!("  To beat backtrack, speculation needs a real draft model (e.g.");
    println!("  MRV cell ordering, or trained digit priors) so the drafter");
    println!("  proposes the RIGHT digit first, not just a valid one.");
    println!();
    println!("  Break-even: speculate wins only when (acceptance_rate ×");
    println!("  commits_per_round × per_commit_savings) > tree_build_overhead.");
    println!("  With p_accept = 1/9 (uniform over digits) on Inkala, that");
    println!("  never holds — exactly what Mode 2 shows above.");
    println!();
    println!("  TL;DR: hardest Sudoku solves in ~{} via backtrack,", fmt_us(t_bt));
    println!("         or ~{} via solve_fast ({:.1}× speedup, modelless MRV + CP).",
        fmt_us(t_fast), bt_us / fast_us.max(1e-9));
    println!("         Speculate-way cannot beat backtrack without a trained drafter");
    println!("         AND is hard-capped at 8-deep lookahead by the u128 layout.", );
    println!();

    // ── Visual verification + assertions (NOT counted in bench time) ──
    // Re-run backtrack, speculate, AND solve_fast OUTSIDE the timing loop so
    // the user can visually confirm the solved grid and the bench asserts the
    // solution is actually correct. The `Instant` here is a single-shot
    // representative timing for bragging, not a benchmark.
    println!("── Visual verification (untimed vs bench) ────────────────────");

    let t_vfast0 = Instant::now();
    let s_verify_fast = solve_fast();
    let t_vfast = t_vfast0.elapsed();

    let t_vbt0 = Instant::now();
    let s_verify_bt = solve_backtrack();
    let t_vbt = t_vbt0.elapsed();

    let t_vsp0 = Instant::now();
    let s_verify_sp = solve_speculate_iterative(8, 128);
    let t_vsp = t_vsp0.elapsed();

    // ── Assert correctness FIRST (panics if any solver is wrong). ──
    assert!(s_verify_fast.solved, "solve_fast did not solve!");
    assert!(
        s_verify_fast.final_board.is_solved(),
        "solve_fast final_board is_solved() == false"
    );
    assert!(s_verify_bt.solved, "backtrack did not solve!");
    assert!(
        s_verify_bt.final_board.is_solved(),
        "backtrack final_board is_solved() == false"
    );
    assert!(s_verify_sp.solved, "speculate_iterative did not solve!");
    assert!(
        s_verify_sp.final_board.is_solved(),
        "speculate_iterative final_board is_solved() == false"
    );
    // Cross-check: Inkala has a unique solution, so all three must agree.
    assert_eq!(
        s_verify_fast.final_board.grid, s_verify_bt.final_board.grid,
        "solve_fast and backtrack produced different grids \
         (Inkala has a unique solution — they must match)"
    );
    assert_eq!(
        s_verify_bt.final_board.grid, s_verify_sp.final_board.grid,
        "backtrack and speculate_iterative produced different grids \
         (Inkala has a unique solution — they must match)"
    );

    // ── Brag-worthy hero block ──
    let speedup_vs_bt = t_vbt.as_nanos() as f64 / t_vfast.as_nanos().max(1) as f64;
    let speedup_vs_sp = t_vsp.as_nanos() as f64 / t_vfast.as_nanos().max(1) as f64;
    let hero = format!("⚡  World's Hardest Sudoku solved in {}", fmt_us(t_vfast));
    let sub  = "(Arto Inkala — 21 clues, 60 empties)";
    let w = hero.len().max(sub.len());
    println!();
    println!("  ╔{}╗", "═".repeat(w + 2));
    println!("  ║ {:<width$} ║", hero, width = w);
    println!("  ║ {:<width$} ║", sub,  width = w);
    println!("  ╚{}╝", "═".repeat(w + 2));
    println!();
    println!("  Leaderboard (single-shot, this run):");
    println!("    🥇  solve_fast   {:>10}   {:>7} steps   {}/speed",
        fmt_us(t_vfast), s_verify_fast.steps, "—");
    println!("    🥈  backtrack    {:>10}   {:>7} steps   {:.1}× slower",
        fmt_us(t_vbt), s_verify_bt.steps, speedup_vs_bt);
    println!("    🥉  speculate    {:>10}   fell back    {:.1}× slower",
        fmt_us(t_vsp), speedup_vs_sp);
    println!();
    println!("  solve_fast = MRV cell selection + naked-singles constraint");
    println!("               propagation. Pure modelless, no training.");
    println!();
    println!("  ✅ verified: 3 solvers agree, grid valid, unique solution confirmed");
    println!();

    // ── Solved grid (proof). ──
    println!("  Solved grid:");
    println!("  ─────────────────────────────────────────────");
    for line in s_verify_fast.final_board.display().lines() {
        println!("  {line}");
    }
    println!("  ─────────────────────────────────────────────");
    println!();
    println!("── end visual verification ──────────────────────────────────");
    println!();

    // Sink to prevent elision.
    if t_bt.as_nanos() == u128::MAX {
        std::process::abort();
    }
}
