//! Sudoku AND-OR Decomposition: Rows as AND Subgoals
//!
//! Demonstrates AND-OR tree decomposition on a small 4×4 Sudoku-like problem.
//! Each row of 4 cells becomes an AND subgoal (all cells must be filled),
//! and alternative fill strategies become OR nodes.
//!
//! Pipeline:
//! 1. Create synthetic marginals for a 4-cell problem (4 tokens per cell)
//! 2. Use a custom pruner that returns low relevance for ambiguous cells
//! 3. Build AND-OR tree via `AndOrBuilder`
//! 4. Show blueprint pre-pass and decomposition reviewer
//! 5. Print tree structure and metrics
//!
//! Run: `cargo run --example and_or_sudoku --features and_or_dtree`

#[cfg(feature = "and_or_dtree")]
fn main() {
    use katgpt_rs::pruners::proof::ProofGoalCache;
    use katgpt_rs::speculative::{
        AndOrBuilder, BlueprintPass, DecompositionReviewer, ScreeningPruner,
    };

    println!("🧩 Sudoku AND-OR Decomposition Demo (4×4 cells)");
    println!("{}", "═".repeat(55));

    // ── 1. Setup ─────────────────────────────────────────────
    //
    // Simulate a 4-cell problem (one row of a 4×4 Sudoku).
    // Each cell has 4 candidate tokens (digits 1-4).
    // Vocab size = 5 (index 0 = padding, indices 1-4 = digits).
    //
    // Cell 0: confident (token 2 gets 70%)        → high relevance
    // Cell 1: ambiguous (tokens 1,3,4 uniform)    → low relevance → decompose
    // Cell 2: confident (token 1 gets 80%)        → high relevance
    // Cell 3: somewhat ambiguous (tokens 2,3)     → low relevance → decompose

    let marginals: Vec<Vec<f32>> = vec![
        // Cell 0: confident → token 2 dominant
        vec![0.0, 0.05, 0.70, 0.15, 0.10],
        // Cell 1: ambiguous → uniform spread
        vec![0.0, 0.30, 0.10, 0.30, 0.30],
        // Cell 2: confident → token 1 dominant
        vec![0.0, 0.80, 0.05, 0.10, 0.05],
        // Cell 3: somewhat ambiguous
        vec![0.0, 0.10, 0.40, 0.40, 0.10],
    ];

    let marginal_refs: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    // ── 2. Custom Sudoku Pruner ──────────────────────────────
    //
    // Returns low relevance for ambiguous cells (entropy > threshold).
    // The builder uses this signal to decide WHERE to decompose.

    struct SudokuRowPruner;

    impl ScreeningPruner for SudokuRowPruner {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            // Padding token is never relevant
            if token_idx == 0 {
                return 0.0;
            }
            // Simulate: cells 1 and 3 are ambiguous → low relevance
            // cells 0 and 2 are confident → high relevance
            match _depth {
                0 => 0.85, // Cell 0: confident
                1 => 0.15, // Cell 1: ambiguous → triggers decomposition
                2 => 0.90, // Cell 2: confident
                3 => 0.20, // Cell 3: ambiguous → triggers decomposition
                _ => 0.5,
            }
        }
    }

    let pruner = SudokuRowPruner;

    // ── 3. Blueprint Pre-Pass ────────────────────────────────
    // Cheap argmax plan: O(depth * vocab). No tree search.

    println!("\n📋 Step 1: Blueprint Pre-Pass (cheap argmax)");
    println!("{}", "─".repeat(55));

    let blueprint = BlueprintPass::generate(&marginal_refs);
    println!("  Blueprint tokens: {blueprint:?}");
    for (d, &t) in blueprint.iter().enumerate() {
        let compat = BlueprintPass::compatibility(d, t, &blueprint, 0.1);
        println!("  Cell {d}: argmax → token {t} (self-compat = {compat:.1})");
    }

    // ── 4. Build AND-OR Tree ─────────────────────────────────
    // Low relevance regions trigger decomposition into subgoals.

    println!("\n🌳 Step 2: AND-OR Tree Construction");
    println!("{}", "─".repeat(55));

    let mut cache = ProofGoalCache::new();
    let mut builder = AndOrBuilder::new(&pruner, &mut cache)
        .with_relevance_threshold(0.3)
        .with_max_depth(8);

    let tree = builder.build(&marginal_refs);

    println!("  Relevance threshold: 0.3 (below → decompose)");
    println!("  Per-cell relevance:  [0.85, 0.15, 0.90, 0.20]");
    println!();
    println!("  Tree:");
    println!("  {tree}");

    // ── 5. Tree Metrics ─────────────────────────────────────

    println!("\n📊 Step 3: Tree Metrics");
    println!("{}", "─".repeat(55));
    println!("  Total nodes:     {}", tree.node_count());
    println!("  Max depth:       {}", tree.depth());
    println!("  Solved leaves:   {}", tree.solved_count());
    println!("  Unsolved leaves: {}", tree.unsolved_count());
    println!("  Root solved:     {}", tree.is_solved());

    // Show children detail
    println!("\n  Root children:");
    for (i, child) in tree.children().iter().enumerate() {
        println!("    [{i}] {child}");
    }

    // ── 6. Decomposition Reviewer ───────────────────────────
    // Simulates checking branch productivity via cache novelty.

    println!("\n🔍 Step 4: Decomposition Reviewer");
    println!("{}", "─".repeat(55));

    let reviewer = DecompositionReviewer::new(0.3);
    println!("  Min novelty threshold: {}", reviewer.min_novelty());

    // Simulate a branch with high novelty (lots of new goals)
    reviewer.reset_branch();
    reviewer.record_miss();
    reviewer.record_miss();
    reviewer.record_miss();
    reviewer.record_hit();
    println!(
        "  Branch A: 3 misses, 1 hit → novelty = {:.2} → productive: {}",
        reviewer.novelty(),
        reviewer.is_productive()
    );

    // Simulate a branch with low novelty (mostly cached goals)
    reviewer.reset_branch();
    reviewer.record_miss();
    reviewer.record_hit();
    reviewer.record_hit();
    reviewer.record_hit();
    println!(
        "  Branch B: 1 miss, 3 hits  → novelty = {:.2} → productive: {}",
        reviewer.novelty(),
        reviewer.is_productive()
    );

    // ── 7. Cache Stats ──────────────────────────────────────

    println!("\n💾 Step 5: ProofGoalCache Stats");
    println!("{}", "─".repeat(55));
    println!("  Cache hits:   {}", cache.hits());
    println!("  Cache misses: {}", cache.misses());
    println!("  Hit rate:     {:.1}%", cache.hit_rate() * 100.0);

    // ── 8. Summary ──────────────────────────────────────────
    println!("\n📌 Summary");
    println!("{}", "─".repeat(55));
    println!("  AND-OR decomposition splits complex regions into subgoals.");
    println!("  Sudoku rows → AND (all cells must be filled).");
    println!("  Alternative strategies → OR (any approach can succeed).");
    println!("  Blueprint pre-pass provides cheap guidance for search.");
    println!("  Decomposition reviewer prunes unproductive branches.");
    println!("  ProofGoalCache memoizes solved subgoals (blake3-keyed).");
}

#[cfg(not(feature = "and_or_dtree"))]
fn main() {
    eprintln!("This example requires the `and_or_dtree` feature.");
    eprintln!("Run: cargo run --example and_or_sudoku --features and_or_dtree");
}
