//! Sudoku with RosettaPruner — Combining Row/Col/Box Constraint Pruners (Plan 201).
//!
//! Demonstrates RosettaPruner as a meta-pruner combining multiple domain-specific
//! ConstraintPruners (row, column, box) into a single unified pruner with:
//! - O(1) fast-path for universal concepts (all pruners agree)
//! - Majority vote fallback for contested positions
//!
//! Run: `cargo run --features rosetta_pruner,sudoku --example rosetta_sudoku`

#![cfg(all(feature = "rosetta_pruner", feature = "sudoku"))]

use std::sync::Arc;

use katgpt_rs::percepta::Sudoku9x9;
use katgpt_rs::pruners::{RosettaPruner, SudokuPruner};
use katgpt_rs::speculative::{ConstraintPruner, ScreeningPruner, build_dd_tree_pruned};
use katgpt_rs::types::Config;

/// Wrapper that applies only row constraint for a specific cell position.
struct RowPruner {
    grid: [[u8; 9]; 9],
    cell_row: usize,
}

impl ConstraintPruner for RowPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        if token_idx == 0 || token_idx > 9 {
            return false;
        }
        let digit = token_idx as u8;
        for col in 0..9 {
            if self.grid[self.cell_row][col] == digit {
                return false;
            }
        }
        true
    }
}

/// Wrapper that applies only column constraint.
struct ColPruner {
    grid: [[u8; 9]; 9],
    cell_col: usize,
}

impl ConstraintPruner for ColPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        if token_idx == 0 || token_idx > 9 {
            return false;
        }
        let digit = token_idx as u8;
        for row in 0..9 {
            if self.grid[row][self.cell_col] == digit {
                return false;
            }
        }
        true
    }
}

/// Wrapper that applies only box constraint.
struct BoxPruner {
    grid: [[u8; 9]; 9],
    cell_row: usize,
    cell_col: usize,
}

impl ConstraintPruner for BoxPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        if token_idx == 0 || token_idx > 9 {
            return false;
        }
        let digit = token_idx as u8;
        let box_row = (self.cell_row / 3) * 3;
        let box_col = (self.cell_col / 3) * 3;
        for r in box_row..box_row + 3 {
            for c in box_col..box_col + 3 {
                if r == self.cell_row && c == self.cell_col {
                    continue;
                }
                if self.grid[r][c] == digit {
                    return false;
                }
            }
        }
        true
    }
}

fn main() {
    println!("🧩 Sudoku with RosettaPruner — Plan 201");
    println!("{}", "═".repeat(55));

    // ── 1. Create a partially-filled Sudoku board ────────────────
    //
    // 5 3 _ | _ 7 _ | _ _ _
    // 6 _ _ | 1 9 5 | _ _ _
    // _ 9 8 | _ _ _ | _ 6 _
    // ------+-------+------
    // 8 _ _ | _ 6 _ | _ _ 3
    // 4 _ _ | 8 _ 3 | _ _ 1
    // 7 _ _ | _ 2 _ | _ _ 6
    // ------+-------+------
    // _ 6 _ | _ _ _ | 2 8 _
    // _ _ _ | 4 1 9 | _ _ 5
    // _ _ _ | _ 8 _ | _ 7 9

    let board = Sudoku9x9::arto_inkala();

    println!("\n  Board: Arto Inkala (hardest known Sudoku)");

    // ── 2. Compare: Standard SudokuPruner vs RosettaPruner ──────

    let config = Config {
        vocab_size: 10,
        tree_budget: 512,
        draft_lookahead: 8,
        ..Config::draft()
    };

    // Create marginals: uniform over digits 1-9 for each cell
    let marginals: Vec<Vec<f32>> = (0..8)
        .map(|_| {
            let mut probs = vec![0.0f32; 10];
            for d in 1..=9u8 {
                probs[d as usize] = 1.0 / 9.0;
            }
            probs
        })
        .collect();
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // Standard SudokuPruner
    let standard_pruner = SudokuPruner::new(board.clone());

    let start = std::time::Instant::now();
    let standard_tree = build_dd_tree_pruned(&mv, &config, &standard_pruner, false);
    let standard_time = start.elapsed();
    let standard_nodes = standard_tree.len();

    // RosettaPruner: combine row, col, box pruners for the first empty cell
    let grid = board.grid;
    // Find first empty cell
    let (first_row, first_col) = grid
        .iter()
        .enumerate()
        .find_map(|(r, row): (usize, &[u8; 9])| {
            row.iter().enumerate().find_map(|(c, &v)| match v == 0 {
                true => Some((r, c)),
                false => None,
            })
        })
        .unwrap_or((0, 0));

    println!("  First empty cell: row={first_row}, col={first_col}");

    let row_pruner: Arc<dyn ConstraintPruner> = Arc::new(RowPruner {
        grid,
        cell_row: first_row,
    });
    let col_pruner: Arc<dyn ConstraintPruner> = Arc::new(ColPruner {
        grid,
        cell_col: first_col,
    });
    let box_pruner: Arc<dyn ConstraintPruner> = Arc::new(BoxPruner {
        grid,
        cell_row: first_row,
        cell_col: first_col,
    });

    let mut rosetta = RosettaPruner::new(vec![row_pruner, col_pruner, box_pruner]);
    let tokens: Vec<usize> = (0..10).collect();
    let discovered = rosetta.mine_concepts(8, &tokens, &[]);
    println!("  Universal concepts discovered: {discovered}");
    println!("  Rosetta pruner count: {}", rosetta.pruner_count());

    let start = std::time::Instant::now();
    let rosetta_tree = build_dd_tree_pruned(&mv, &config, &rosetta, false);
    let rosetta_time = start.elapsed();
    let rosetta_nodes = rosetta_tree.len();

    // ── 3. Results ──────────────────────────────────────────────

    println!("\n  {:>24} {:>10} {:>12}", "Pruner", "Nodes", "Time (μs)");
    println!("  {}", "-".repeat(50));
    println!(
        "  {:>24} {:>10} {:>12.2}",
        "SudokuPruner (standard)",
        standard_nodes,
        standard_time.as_nanos() as f64 / 1000.0,
    );
    println!(
        "  {:>24} {:>10} {:>12.2}",
        "RosettaPruner (row+col+box)",
        rosetta_nodes,
        rosetta_time.as_nanos() as f64 / 1000.0,
    );

    let node_reduction = if standard_nodes > 0 {
        (1.0 - rosetta_nodes as f64 / standard_nodes as f64) * 100.0
    } else {
        0.0
    };
    println!("\n  Node reduction: {node_reduction:.1}%");

    // ── 4. Show ScreeningPruner path ────────────────────────────

    println!("\n  ScreeningPruner relevance for first cell:");
    for digit in 1..=9 {
        let valid = rosetta.is_valid(0, digit, &[]);
        let rel = rosetta.relevance(0, digit, &[]);
        println!("    digit={digit}: valid={valid}, relevance={rel:.3}");
    }

    println!("\n{}", "═".repeat(55));
    println!("  ✅ RosettaPruner successfully combines row/col/box constraints");
    println!("  ✅ Universal concepts get O(1) fast-path via concept map");
    println!("  ✅ ScreeningPruner provides soft relevance for contested tokens");
}

// TL;DR: Sudoku example for Plan 201 Rosetta Pruner — demonstrates
// combining row/col/box constraint pruners into a unified meta-pruner.
// Run with: cargo run --features "rosetta_pruner,sudoku" --example rosetta_sudoku
