//! AND-OR DDTree API Demo — Tree Construction & Metrics Walkthrough
//!
//! Demonstrates the core `AndOrNode<G, S>` API from katgpt-core:
//! - OR/AND/Leaf node construction
//! - Pushing children
//! - Solved propagation (OR: any child, AND: all children)
//! - Tree metrics (node_count, depth, solved_count)
//!
//! Run: `cargo run --example and_or_demo --features and_or_dtree`

#[cfg(feature = "and_or_dtree")]
fn main() {
    use katgpt_core::AndOrNode;

    println!("🌳 AND-OR DDTree API Demo");
    println!("{}", "═".repeat(50));

    // ── 1. Build a small tree manually ──────────────────────
    //
    // Structure:
    //   OR(root)
    //   ├── AND(strategy-A)
    //   │   ├── LEAF(subgoal-1) ✅ solved
    //   │   └── LEAF(subgoal-2) ✅ solved
    //   └── LEAF(strategy-B) — unsolved
    //
    // The root OR is solved because strategy-A (AND) is solved
    // (all its children are solved).

    // Leaves first
    let leaf1 = AndOrNode::solved_leaf("subgoal-1", vec![0, 1]);
    let leaf2 = AndOrNode::solved_leaf("subgoal-2", vec![2, 3]);

    // AND node: both children must succeed
    let mut and_node = AndOrNode::and("strategy-A");
    and_node.push_child(leaf1);
    and_node.push_child(leaf2);

    // Mark both children as solved in the AND node
    assert!(and_node.mark_child_solved(0), "child 0 should be markable");
    assert!(and_node.mark_child_solved(1), "child 1 should be markable");

    // OR alternative: unsolved leaf
    let leaf_alt = AndOrNode::unsolved_leaf("strategy-B");

    // Root OR node: any child can succeed
    let mut root = AndOrNode::or("root");
    root.push_child(and_node);
    root.push_child(leaf_alt);
    root.set_best(0); // strategy-A is the best so far

    // ── 2. Display tree ─────────────────────────────────────
    println!("\n📋 Tree Structure:");
    println!("{}", "─".repeat(50));
    println!("  {root}");

    println!("\n  Children:");
    for (i, child) in root.children().iter().enumerate() {
        println!("    [{i}] {child}");
    }

    // ── 3. Metrics ──────────────────────────────────────────
    println!("\n📊 Tree Metrics:");
    println!("{}", "─".repeat(50));
    println!("  Total nodes:    {}", root.node_count());
    println!("  Max depth:      {}", root.depth());
    println!("  Solved leaves:  {}", root.solved_count());
    println!("  Unsolved leaves: {}", root.unsolved_count());
    println!("  Root solved:    {}", root.is_solved());

    // ── 4. Verify solved propagation ────────────────────────
    println!("\n✅ Solved Propagation:");
    println!("{}", "─".repeat(50));

    let and_child = root.child(0).unwrap();
    println!("  AND node solved:  {}", and_child.is_solved()); // true
    println!("  OR alt solved:    {}", root.child(1).unwrap().is_solved()); // false
    println!("  Root (OR) solved: {}", root.is_solved()); // true — any child

    // ── 5. Mutations ────────────────────────────────────────
    println!("\n🔧 Mutations:");
    println!("{}", "─".repeat(50));

    // Solve the alternative strategy
    root.child_mut(1).unwrap().set_solution(vec![5, 6]);
    println!(
        "  Set solution on alt leaf → alt solved: {}",
        root.child(1).unwrap().is_solved()
    );

    // Add a sketch to the AND node
    root.child_mut(0).unwrap().set_sketch(vec![0, 1, 2, 3]);
    println!("  Set sketch on AND node");

    // Extract and re-print
    if let Some(solution) = root.child_mut(1).unwrap().take_solution() {
        println!("  Extracted alt solution: {solution:?}");
    }

    // ── 6. Summary ──────────────────────────────────────────
    println!("\n📌 Summary:");
    println!("  AndOrNode<G,S> is generic over any goal and solution type.");
    println!("  OR:  solved if ANY child solves (alternative strategies)");
    println!("  AND: solved if ALL children solve (decomposed subgoals)");
    println!("  Leaf: atomic goal, solved if solution.is_some()");
    println!();
    println!("  In DDTree: low-relevance regions → AND decomposition,");
    println!("  high-relevance → leaf (solved directly). Root is always OR.");
}

#[cfg(not(feature = "and_or_dtree"))]
fn main() {
    eprintln!("This example requires the `and_or_dtree` feature.");
    eprintln!("Run: cargo run --example and_or_demo --features and_or_dtree");
}
