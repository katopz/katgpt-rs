//! Domino Sudoku: DDTree + DominoPruner prefix-conditioned correction.
//!
//! Demonstrates modelless domino correction with Sudoku constraint pruning:
//! - **Before**: standard DDTree with SudokuPruner (100% valid, known result)
//! - **After**: DDTree with DominoPruner showing same validity but fewer nodes explored
//!
//! Metric: nodes_explored, valid_rate, time_per_tree
//! Expected: same 100% valid, ~10-20% fewer nodes (prefix-aware pruning eliminates branches earlier)
//!
//! Run: cargo run --example domino_sudoku --features domino_correction,sudoku

#[cfg(not(all(feature = "domino_correction", feature = "sudoku")))]
fn main() {
    eprintln!("This example requires both `domino_correction` and `sudoku` features.");
    eprintln!("Run with: cargo run --example domino_sudoku --features domino_correction,sudoku");
}

#[cfg(all(feature = "domino_correction", feature = "sudoku"))]
fn main() {
    use katgpt_rs::percepta::Sudoku9x9;
    use katgpt_rs::pruners::SudokuPruner;
    use katgpt_rs::speculative::{
        ConstraintPruner, PrefixCorrectionTable, build_dd_tree_pruned, domino_correct_marginals,
        domino_score, extract_parent_tokens, prefix_hash,
    };
    use katgpt_rs::types::Config;

    println!("🁣 Domino Sudoku: Prefix-Conditioned Causal Correction");
    println!("{}", "═".repeat(60));

    let board = Sudoku9x9::arto_inkala();
    let clues = board.clue_count();
    let empty = 81 - clues;
    println!("\n📝 Arto Inkala Puzzle — {clues} clues, {empty} empty cells\n");
    print!("{}", board.display());

    let pruner = SudokuPruner::new(board.clone());

    // ── 1. Simulate draft model marginals ──────────────────────────
    let lookahead = 6usize.min(pruner.empty_count());
    let vocab_size = 10; // digits 0-9 (0 = padding)

    let raw_marginals: Vec<Vec<f32>> = (0..lookahead)
        .map(|_| {
            let mut probs = vec![0.0f32; vocab_size];
            for d in 1..=9u8 {
                probs[d as usize] = 1.0 / 9.0;
            }
            probs
        })
        .collect();

    let marginal_slices: Vec<&[f32]> = raw_marginals.iter().map(|m| m.as_slice()).collect();

    let config = Config {
        tree_budget: 64,
        draft_lookahead: lookahead,
        vocab_size,
        ..Config::default()
    };

    // ── 2. Baseline: Standard DDTree with SudokuPruner ──────────────
    let start = std::time::Instant::now();
    let baseline_tree = build_dd_tree_pruned(&marginal_slices, &config, &pruner, true);
    let baseline_time = start.elapsed();
    let baseline_nodes = baseline_tree.len();

    let baseline_valid = baseline_tree
        .iter()
        .filter(|n| n.depth > 0)
        .map(|n| {
            let parent_tokens = extract_parent_tokens(n.parent_path, n.depth);
            pruner.is_valid(n.depth, n.token_idx, &parent_tokens)
        })
        .collect::<Vec<_>>();
    let baseline_valid_rate =
        baseline_valid.iter().filter(|&&v| v).count() as f64 / baseline_valid.len().max(1) as f64;

    // ── 3. Build prefix correction table from constraint patterns ───
    // Simulate a correction table: for common prefix patterns,
    // suppress tokens that violate row/col/box constraints
    let mut correction_builder = PrefixCorrectionTable::builder(vocab_size);
    for depth in 1..lookahead {
        // For each depth, compute a correction that suppresses invalid tokens
        let mut correction = vec![0.0f32; vocab_size];
        for token in 1..=9 {
            // Token 0 is padding, skip
            let is_valid = pruner.is_valid(depth, token, &[0; 0]);
            if !is_valid {
                correction[token] = -0.1; // Suppress invalid tokens
            }
        }
        // Use depth as a simple key for demonstration
        correction_builder =
            correction_builder.add_correction_raw(prefix_hash(&[depth]), correction);
    }
    let table = correction_builder.build();

    // ── 4. Apply domino correction to marginals ─────────────────────
    let mut corrected_marginals = raw_marginals.clone();
    let sampled_tokens: Vec<usize> = (1..=lookahead).collect();
    domino_correct_marginals(&mut corrected_marginals, &sampled_tokens, &table);

    let corrected_slices: Vec<&[f32]> = corrected_marginals.iter().map(|m| m.as_slice()).collect();

    // ── 5. Domino DDTree ────────────────────────────────────────────
    let start = std::time::Instant::now();
    let domino_tree = build_dd_tree_pruned(&corrected_slices, &config, &pruner, true);
    let domino_time = start.elapsed();
    let domino_nodes = domino_tree.len();

    let domino_valid = domino_tree
        .iter()
        .filter(|n| n.depth > 0)
        .map(|n| {
            let parent_tokens = extract_parent_tokens(n.parent_path, n.depth);
            pruner.is_valid(n.depth, n.token_idx, &parent_tokens)
        })
        .collect::<Vec<_>>();
    let domino_valid_rate =
        domino_valid.iter().filter(|&&v| v).count() as f64 / domino_valid.len().max(1) as f64;

    // ── 6. Results ──────────────────────────────────────────────────
    println!("\n📊 Results");
    println!("{}", "─".repeat(60));
    println!(
        "{:<25} {:>12} {:>12} {:>12}",
        "Metric", "Baseline", "Domino", "Δ"
    );
    println!("{}", "─".repeat(60));

    let node_delta = domino_nodes as f64 - baseline_nodes as f64;
    let node_pct = if baseline_nodes > 0 {
        node_delta / baseline_nodes as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "{:<25} {:>12} {:>12} {:>11.1}%",
        "Nodes explored", baseline_nodes, domino_nodes, node_pct
    );

    let time_delta = domino_time.as_secs_f64() - baseline_time.as_secs_f64();
    let time_pct = if baseline_time.as_secs_f64() > 0.0 {
        time_delta / baseline_time.as_secs_f64() * 100.0
    } else {
        0.0
    };
    println!(
        "{:<25} {:>10.1}µs {:>10.1}µs {:>11.1}%",
        "Build time",
        baseline_time.as_secs_f64() * 1e6,
        domino_time.as_secs_f64() * 1e6,
        time_pct
    );

    println!(
        "{:<25} {:>11.1}% {:>11.1}% {:>12}",
        "Valid rate",
        baseline_valid_rate * 100.0,
        domino_valid_rate * 100.0,
        if (baseline_valid_rate - domino_valid_rate).abs() < 0.01 {
            "same ✓"
        } else {
            "diff ⚠"
        }
    );

    println!("{}", "─".repeat(60));

    // ── 7. Domino score demo ────────────────────────────────────────
    println!("\n🧮 Domino Score Demo");
    let base = -2.5f32;
    println!("  base_score = {base}");
    for depth in 0..=4 {
        let strength = 0.8f32;
        let scored = domino_score(base, depth, strength);
        println!("  depth={depth}, strength={strength} → score={scored:.4}");
    }

    println!(
        "\n✅ Domino correction is modelless — no training, no LoRA, pure pattern extraction."
    );
}
