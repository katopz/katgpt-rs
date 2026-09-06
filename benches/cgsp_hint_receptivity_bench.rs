//! CGSP Hint-Receptivity Bench — wires the DDTree Sudoku speculation as a
//! CGSP `Solver` and compares `HintPolicy::OrderOnly` (bandit absorbs rewards,
//! priorities adapt) vs `HintPolicy::Skip` (uniform sampling, no feedback).
//!
//! This is the G-RRM §3 experiment (arXiv 2607.02491): does hint-ordering help
//! or hurt the solver? G-RRM found cadical3 is overhead-dominated (0.896×
//! slowdown from hints); glucose4/backtracking are hint-receptive (up to 33.3×
//! speedup). This bench measures which regime our DDTree speculation falls in
//! when routed through the CGSP loop.
//!
//! Design: each `Solver::attempt()` call clones Arto Inkala's hardest Sudoku,
//! forces a specific digit (mapped from `pool_index`) into the first empty
//! cell, then runs ONE round of DDTree speculation. The return value is
//! `spec_commits / 60` (progress rate, lands in the intermediate-difficulty
//! band). With `OrderOnly`, the bandit learns which digits lead to more
//! committed cells; with `Skip`, priorities stay uniform.
//!
//! Run:
//! ```bash
//! cargo bench --bench cgsp_hint_receptivity_bench --features cgsp,sudoku_mrv,sudoku_cp
//! ```

#![cfg(all(feature = "cgsp", feature = "sudoku"))]

use katgpt_core::cgsp::traits::{HintDeltaBandit, NoOpBatchGate, NoOpDifficultyFilter, Solver};
use katgpt_core::cgsp::types::{
    CycleResult, Direction, HintPolicy, Priority, ScratchBuffers, Target,
};
use katgpt_core::cgsp::{CgspConfig, CgspLoop};
use katgpt_percepta::Sudoku9x9;
use katgpt_rs::pruners::SudokuPruner;
use katgpt_rs::speculative::build_dd_tree_pruned;
use katgpt_rs::types::Config;

/// Vocab: indices 0..=9 (0=empty, 1..=9=digits).
const SUDOKU_VOCAB: usize = 10;

/// Inkala has 60 empty cells.
const INKALA_EMPTIES: usize = 60;

/// Pool size = 9 digits.
const POOL_SIZE: usize = 9;

/// CGSP cycles per A/B run.
const N_CYCLES: usize = 50;

// ── VecBandit (test-only priority table) ──────────────────────────────────

struct VecBandit {
    prios: Vec<f32>,
}

impl VecBandit {
    fn uniform(n: usize) -> Self {
        Self {
            prios: vec![1.0 / n as f32; n],
        }
    }
}

impl HintDeltaBandit for VecBandit {
    fn absorb(&mut self, arm: usize, reward: f32) {
        if let Some(p) = self.prios.get_mut(arm) {
            *p += reward.max(0.0);
        }
    }
    fn priority(&self, arm: usize) -> Priority {
        self.prios.get(arm).copied().unwrap_or(0.0)
    }
    fn priorities(&self) -> &[Priority] {
        &self.prios
    }
    fn priorities_mut(&mut self) -> &mut [Priority] {
        &mut self.prios
    }
}

// ── Sudoku Speculation Solver ────────────────────────────────────────────

/// One round of DDTree speculation from Inkala, with `pool_index` forcing a
/// specific digit into the first empty cell.
///
/// Returns `spec_commits / INKALA_EMPTIES` — the fraction of cells committed
/// in one round. This lands in `[0, ~0.13]` (8-deep ceiling / 60 empties),
/// which sits in CGSP's intermediate-difficulty band.
fn speculate_one_round(first_digit: u8) -> usize {
    let mut board = Sudoku9x9::arto_inkala();

    // Force the digit into the first empty cell (row-major scan).
    'outer: for r in 0..9 {
        for c in 0..9 {
            if board.grid[r][c] == 0 {
                if board.is_valid_move(r, c, first_digit) {
                    board.grid[r][c] = first_digit;
                }
                break 'outer;
            }
        }
    }

    // Build one DDTree speculation round.
    let pruner = SudokuPruner::new_mrv(board.clone());
    let empty = pruner.empty_count();
    if empty == 0 {
        return 0;
    }

    let lookahead = 8usize.min(empty);
    let mut config = Config::draft();
    config.vocab_size = SUDOKU_VOCAB;
    config.tree_budget = 128;
    config.draft_lookahead = lookahead;

    // Constraint-aware marginals: naked singles → p=1.0, else uniform 1/N.
    let margs: Vec<Vec<f32>> = (0..lookahead)
        .map(|depth| {
            if let Some((row, col)) = pruner.position_at(depth) {
                let mut candidates = Vec::new();
                for d in 1..=9u8 {
                    if board.is_valid_move(row, col, d) {
                        candidates.push(d);
                    }
                }
                let n = candidates.len();
                let mut p = vec![0.0f32; SUDOKU_VOCAB];
                if n == 1 {
                    p[candidates[0] as usize] = 1.0; // naked single
                } else if n > 0 {
                    for &d in &candidates {
                        p[d as usize] = 1.0 / n as f32;
                    }
                }
                p
            } else {
                vec![0.0f32; SUDOKU_VOCAB]
            }
        })
        .collect();

    let mv: Vec<&[f32]> = margs.iter().map(|s| s.as_slice()).collect();
    let tree = build_dd_tree_pruned(&mv, &config, &pruner, false);
    if tree.is_empty() {
        return 0;
    }

    // Commit the deepest path.
    let best = tree
        .iter()
        .max_by(|a, b| a.depth.cmp(&b.depth).then(a.score.total_cmp(&b.score)))
        .unwrap();

    let path = katgpt_rs::speculative::extract_parent_tokens(best.parent_path, best.depth + 1);
    let mut commits = 0usize;
    for (depth, &token) in path.iter().enumerate() {
        if token == 0 {
            continue;
        }
        if let Some((row, col)) = pruner.position_at(depth)
            && board.is_valid_move(row, col, token as u8)
        {
            board.grid[row][col] = token as u8;
            commits += 1;
        }
    }
    commits
}

/// CGSP Solver wrapping Sudoku DDTree speculation.
/// `pool_index` maps to digit `pool_index + 1` (1-9).
struct SudokuSpecSolver {
    hint_policy: HintPolicy,
}

impl Solver for SudokuSpecSolver {
    fn attempt(
        &mut self,
        _target: &Target,
        _candidate_direction: &Direction,
        pool_index: usize,
    ) -> f32 {
        let digit = (pool_index + 1).min(9) as u8;
        let commits = speculate_one_round(digit);
        // Progress rate: commits / total empties.
        // Maps to [0, ~0.13] — intermediate-difficulty band.
        commits as f32 / INKALA_EMPTIES as f32
    }

    fn hint_receptivity(&self) -> HintPolicy {
        self.hint_policy
    }
}

// ── Pool: 9 digit-directions (identity basis in 9-dim space) ──────────────

fn make_digit_pool() -> Vec<Direction> {
    (0..POOL_SIZE)
        .map(|i| {
            let mut coords = vec![0.0f32; POOL_SIZE];
            coords[i] = 1.0;
            Direction { coords }
        })
        .collect()
}

/// Trivial guide: returns the digit's own coord value (digit i → priority i).
/// This gives the bandit a weak prior to work with.
struct DigitGuide;

impl katgpt_core::cgsp::traits::QualityGuide for DigitGuide {
    fn score(&self, _target: &Target, candidate: &Direction) -> f32 {
        // Return the max coord (which digit this direction represents).
        candidate.coords.iter().cloned().fold(0.0f32, f32::max)
    }
}

/// Simple conjecturer: samples from the pool weighted by priorities.
struct PoolConjecturer {
    pool: Vec<Direction>,
    rng_state: u64,
}

impl PoolConjecturer {
    fn new(pool: Vec<Direction>, seed: u64) -> Self {
        Self {
            pool,
            rng_state: seed,
        }
    }

    #[inline]
    fn next_rand(&mut self) -> u64 {
        // xorshift64
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }
}

impl katgpt_core::cgsp::traits::CuriosityConjecturer for PoolConjecturer {
    fn sample_candidates(
        &mut self,
        _target: &Target,
        priorities: &[Priority],
        out: &mut [katgpt_core::cgsp::types::Candidate],
        cdf_scratch: &mut Vec<f32>,
    ) {
        let k = out.len();
        let n = self.pool.len();
        // Build CDF from priorities.
        cdf_scratch.clear();
        cdf_scratch.reserve(n + 1);
        let mut cum = 0.0f32;
        cdf_scratch.push(0.0);
        for &p in priorities.iter().take(n) {
            cum += p.max(0.0);
            cdf_scratch.push(cum);
        }
        let total = cum.max(1e-9);

        for slot in out.iter_mut().take(k) {
            let r = (self.next_rand() as f32 / u64::MAX as f32) * total;
            // Binary search for the arm.
            let arm = match cdf_scratch[1..].binary_search_by(|probe| {
                probe.total_cmp(&r)
            }) {
                Ok(i) | Err(i) => i.min(n - 1),
            };
            let dir = self.pool[arm].clone();
            *slot = katgpt_core::cgsp::types::Candidate::new(dir, arm);
        }
    }

    fn pool_size(&self) -> usize {
        self.pool.len()
    }

    fn pool_directions(&self) -> &[Direction] {
        &self.pool
    }
}

// ── A/B Runner ────────────────────────────────────────────────────────────

struct RunOutcome {
    label: String,
    mean_solve_rate: f32,
    priority_max: f32,
    priority_min: f32,
    priority_spread: f32,
}

fn run_ab(label: &str, hint_policy: HintPolicy) -> RunOutcome {
    let pool = make_digit_pool();
    let conj = PoolConjecturer::new(pool.clone(), 42);
    let guide = DigitGuide;
    let solver = SudokuSpecSolver { hint_policy };
    let bandit = VecBandit::uniform(POOL_SIZE);

    let config = CgspConfig {
        k: 4,
        tau_low: 0.05, // low threshold so entropy doesn't trigger collapse on small pool
        exploration_magnitude: 0.2,
        solve_rate_floor: 0.0,
        solve_rate_ceiling: 1.0,
        k_npc: 1,
        staleness_lambda: 0.0,
    };

    let mut lp = CgspLoop::new(conj, guide, solver, bandit, config)
        .with_difficulty_filter(NoOpDifficultyFilter)
        .with_batch_gate(NoOpBatchGate);

    let target = Target::new(Direction {
        coords: vec![1.0; POOL_SIZE],
    });
    let mut scratch = ScratchBuffers::new(POOL_SIZE, POOL_SIZE);

    let mut rate_sum = 0.0f32;
    let mut rate_count = 0u32;

    for _ in 0..N_CYCLES {
        let r: CycleResult = lp.cycle(&target, &mut scratch);
        rate_sum += r.stats.mean_r_synth;
        rate_count += 1;
    }

    let prios = lp.bandit().priorities();
    let pmax = prios.iter().cloned().fold(0.0f32, f32::max);
    let pmin = prios.iter().cloned().fold(f32::MAX, f32::min);

    RunOutcome {
        label: label.to_string(),
        mean_solve_rate: rate_sum / rate_count.max(1) as f32,
        priority_max: pmax,
        priority_min: pmin,
        priority_spread: pmax - pmin,
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CGSP Hint-Receptivity Bench — DDTree Sudoku × G-RRM §3     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Solver:    DDTree speculation (1 round/attempt, Inkala hardest)");
    println!("  Pool:      {POOL_SIZE} digits (1-9), identity basis");
    println!("  Cycles:    {N_CYCLES} × k=4 candidates each");
    println!("  Question:  does priority-weighted sampling (OrderOnly) beat");
    println!("             uniform (Skip) at finding high-commit digits?");
    println!();

    // ── A/B runs ──
    let order_only = run_ab("OrderOnly", HintPolicy::OrderOnly);
    let skip = run_ab("Skip", HintPolicy::Skip);

    // ── Report ──
    println!(
        "┌{:─<14}┬{:─>14}┬{:─>14}┬{:─>14}┬{:─>14}┐",
        "", "", "", "", ""
    );
    println!(
        "│{: <14}│{: >14}│{: >14}│{: >14}│{: >14}│",
        "Policy", "Mean r_synth", "Prio Spread", "Prio Min", "Prio Max"
    );
    println!(
        "├{:─<14}┼{:─>14}┼{:─>14}┼{:─>14}┼{:─>14}┤",
        "", "", "", "", ""
    );
    for o in [&order_only, &skip] {
        println!(
            "│{: <14}│{: >14.6}│{: >14.6}│{: >14.6}│{: >14.6}│",
            o.label, o.mean_solve_rate, o.priority_spread, o.priority_min, o.priority_max
        );
    }
    println!(
        "└{:─<14}┴{:─>14}┴{:─>14}┴{:─>14}┴{:─>14}┘",
        "", "", "", "", ""
    );
    println!();

    // ── Verdict ──
    let spread_diff = order_only.priority_spread - skip.priority_spread;
    let rate_diff = order_only.mean_solve_rate - skip.mean_solve_rate;

    println!("── Verdict (G-RRM §3 regime classification) ──────────────");
    println!(
        "  Priority spread: OrderOnly={:.6} vs Skip={:.6} (Δ={:+.6})",
        order_only.priority_spread, skip.priority_spread, spread_diff
    );
    println!(
        "  Mean r_synth:    OrderOnly={:.6} vs Skip={:.6} (Δ={:+.6})",
        order_only.mean_solve_rate, skip.mean_solve_rate, rate_diff
    );
    println!();

    if order_only.priority_spread > skip.priority_spread + 1e-6 {
        println!("  ✅ OrderOnly CONVERGED priorities (spread > 0).");
        println!("     → Bandit learned to prefer certain digits → solver is HINT-RECEPTIVE.");
        if rate_diff > 1e-6 {
            println!("     → AND mean r_synth improved → hints genuinely help (glucose4 regime).");
        } else {
            println!("     → BUT mean r_synth flat → hints change sampling but not outcomes");
            println!("       (overhead-dominated: cadical3 regime).");
        }
    } else {
        println!("  ⚠️  OrderOnly did NOT converge priorities beyond Skip.");
        println!("     → Either all digits are equally productive, or the difficulty");
        println!("       filter suppressed all feedback. Check the r_synth column.");
    }
    println!();

    // ── Per-digit probe (which digits produce most commits?) ──
    println!("── Per-digit commit probe (ground truth, no CGSP) ────────");
    println!("{:<10} {:>12} {:>14}", "Digit", "Commits", "Progress Rate");
    println!("{}", "─".repeat(38));
    for d in 1..=9u8 {
        let commits = speculate_one_round(d);
        let rate = commits as f32 / INKALA_EMPTIES as f32;
        println!("{d:<10} {commits:>12} {rate:>14.6}");
    }
}
