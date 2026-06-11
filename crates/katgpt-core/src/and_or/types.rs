//! Generic AND-OR tree node for hierarchical goal decomposition.
//!
//! Inspired by LEAP's AND-OR DAG with hierarchical memoization (arXiv 2606.03303).
//! Generic over goal type `G` and solution type `S` so the same structure serves:
//! - **Proof search** (katgpt-rs): `G = Subgoal`, `S = Vec<usize>` (token path)
//! - **Game strategy** (riir-ai): `G = StrategySubgoal`, `S = StrategySolution`
//!
//! # Layout
//!
//! ```text
//! AndOrNode::Or       — any child can succeed (alternative strategies)
//! AndOrNode::And      — all children must succeed (decomposition)
//! AndOrNode::Leaf     — atomic goal, solved or unsolved
//! ```

use std::fmt;

/// AND-OR tree node for hierarchical goal decomposition.
///
/// # Type parameters
///
/// - `G`: Goal descriptor (subgoal, strategy subgoal, etc.)
/// - `S`: Solution type (token path, action sequence, etc.)
///
/// # Invariants
///
/// - `Or` nodes: `best` is always `< children.len()` when `Some`
/// - `And` nodes: `solved.len() == children.len()` always
/// - `Leaf` nodes: no children
#[derive(Debug, Clone)]
pub enum AndOrNode<G, S> {
    /// OR node: any child can succeed. Represents alternative strategies.
    Or {
        goal: G,
        children: Vec<AndOrNode<G, S>>,
        /// Index of best child so far (highest cumulative relevance).
        best: Option<usize>,
    },
    /// AND node: all children must succeed. Represents a decomposition.
    And {
        goal: G,
        /// Partial solution assuming subgoals succeed.
        sketch: Option<S>,
        children: Vec<AndOrNode<G, S>>,
        /// Per-child solved status. `solved.len() == children.len()` invariant.
        solved: Vec<bool>,
        /// Number of `true` entries in `solved`. Maintained incrementally.
        solved_count: usize,
    },
    /// Leaf: a solved or unsolved atomic goal.
    Leaf { goal: G, solution: Option<S> },
}

impl<G, S> AndOrNode<G, S> {
    // ── Constructors ─────────────────────────────────────────────

    /// Create an OR node with no children.
    #[inline]
    pub fn or(goal: G) -> Self {
        Self::Or {
            goal,
            children: Vec::new(),
            best: None,
        }
    }

    /// Create an AND node with no children.
    #[inline]
    pub fn and(goal: G) -> Self {
        Self::And {
            goal,
            sketch: None,
            children: Vec::new(),
            solved: Vec::new(),
            solved_count: 0,
        }
    }

    /// Create a solved leaf.
    #[inline]
    pub fn solved_leaf(goal: G, solution: S) -> Self {
        Self::Leaf {
            goal,
            solution: Some(solution),
        }
    }

    /// Create an unsolved leaf.
    #[inline]
    pub fn unsolved_leaf(goal: G) -> Self {
        Self::Leaf {
            goal,
            solution: None,
        }
    }

    // ── Accessors ────────────────────────────────────────────────

    /// Returns `true` if this node is fully solved.
    ///
    /// - `Or`: solved if any child is solved (or `best` points to a solved child)
    /// - `And`: solved if all children are solved
    /// - `Leaf`: solved if `solution.is_some()`
    #[inline]
    pub fn is_solved(&self) -> bool {
        match self {
            Self::Or { children, best, .. } => match best {
                &Some(idx) => children.get(idx).is_some_and(|c| c.is_solved()),
                None => children.iter().any(|c| c.is_solved()),
            },
            Self::And {
                children,
                solved_count,
                ..
            } => {
                if children.is_empty() {
                    return false;
                }
                *solved_count == children.len()
            }
            Self::Leaf { solution, .. } => solution.is_some(),
        }
    }

    /// Borrow the goal from any node variant.
    pub fn goal(&self) -> &G {
        match self {
            Self::Or { goal, .. } | Self::And { goal, .. } | Self::Leaf { goal, .. } => goal,
        }
    }

    /// Mutably borrow the goal.
    pub fn goal_mut(&mut self) -> &mut G {
        match self {
            Self::Or { goal, .. } | Self::And { goal, .. } | Self::Leaf { goal, .. } => goal,
        }
    }

    /// Number of direct children. 0 for leaves.
    #[inline]
    pub fn child_count(&self) -> usize {
        match self {
            Self::Or { children, .. } | Self::And { children, .. } => children.len(),
            Self::Leaf { .. } => 0,
        }
    }

    /// Borrow a child by index. Returns `None` for leaves or out-of-bounds.
    #[inline]
    pub fn child(&self, idx: usize) -> Option<&AndOrNode<G, S>> {
        match self {
            Self::Or { children, .. } | Self::And { children, .. } => children.get(idx),
            Self::Leaf { .. } => None,
        }
    }

    /// Mutably borrow a child by index.
    #[inline]
    pub fn child_mut(&mut self, idx: usize) -> Option<&mut AndOrNode<G, S>> {
        match self {
            Self::Or { children, .. } | Self::And { children, .. } => children.get_mut(idx),
            Self::Leaf { .. } => None,
        }
    }

    /// Iterate over direct children.
    #[inline]
    pub fn children(&self) -> &[AndOrNode<G, S>] {
        match self {
            Self::Or { children, .. } | Self::And { children, .. } => children,
            Self::Leaf { .. } => &[],
        }
    }

    // ── Mutators ─────────────────────────────────────────────────

    /// Add a child to an OR or AND node. No-op for leaves.
    ///
    /// For AND nodes, this also appends `false` to `solved`.
    pub fn push_child(&mut self, child: AndOrNode<G, S>) {
        match self {
            Self::Or { children, .. } => {
                children.push(child);
            }
            Self::And {
                children,
                solved,
                solved_count,
                ..
            } => {
                children.push(child);
                solved.push(false);
                // solved_count unchanged — new child is unsolved.
                let _ = solved_count;
            }
            Self::Leaf { .. } => {}
        }
    }

    /// Mark child `idx` as solved in an AND node.
    /// Returns `false` if not an AND node or `idx` out of bounds.
    pub fn mark_child_solved(&mut self, idx: usize) -> bool {
        match self {
            Self::And {
                children: _,
                solved,
                solved_count,
                ..
            } => {
                if idx < solved.len() {
                    if !solved[idx] {
                        solved[idx] = true;
                        *solved_count += 1;
                    }
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Set `best` index for an OR node.
    /// Returns `false` if not an OR node or `idx` out of bounds.
    pub fn set_best(&mut self, idx: usize) -> bool {
        match self {
            Self::Or { children, best, .. } => {
                if idx < children.len() {
                    *best = Some(idx);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Set solution on a leaf. No-op for non-leaves.
    pub fn set_solution(&mut self, solution: S) {
        if let Self::Leaf { solution: slot, .. } = self {
            *slot = Some(solution);
        }
    }

    /// Set the sketch (partial solution) on an AND node. No-op for non-AND.
    pub fn set_sketch(&mut self, sketch: S) {
        if let Self::And { sketch: slot, .. } = self {
            *slot = Some(sketch);
        }
    }

    /// Extract the solution from a leaf, replacing with `None`.
    /// Returns `None` for non-leaf nodes.
    pub fn take_solution(&mut self) -> Option<S> {
        if let Self::Leaf { solution, .. } = self {
            solution.take()
        } else {
            None
        }
    }

    /// Extract the sketch from an AND node, replacing with `None`.
    /// Returns `None` for non-AND nodes.
    pub fn take_sketch(&mut self) -> Option<S> {
        if let Self::And { sketch, .. } = self {
            sketch.take()
        } else {
            None
        }
    }

    // ── Tree metrics ─────────────────────────────────────────────

    /// Total number of nodes in this subtree (including self).
    #[inline]
    pub fn node_count(&self) -> usize {
        1 + match self {
            Self::Or { children, .. } | Self::And { children, .. } => {
                children.iter().map(|c| c.node_count()).sum::<usize>()
            }
            Self::Leaf { .. } => 0,
        }
    }

    /// Maximum depth of the tree (0 for leaves, 1 + max child depth otherwise).
    #[inline]
    pub fn depth(&self) -> usize {
        match self {
            Self::Or { children, .. } | Self::And { children, .. } => children
                .iter()
                .map(|c| c.depth())
                .max()
                .map_or(0, |d| 1 + d),
            Self::Leaf { .. } => 0,
        }
    }

    /// Count of solved leaves in this subtree.
    ///
    /// If you also need `unsolved_count`, prefer [`leaf_stats`](Self::leaf_stats)
    /// which fuses both in a single traversal.
    #[inline]
    pub fn solved_count(&self) -> usize {
        match self {
            Self::Or { children, .. } | Self::And { children, .. } => {
                children.iter().map(|c| c.solved_count()).sum()
            }
            Self::Leaf { solution, .. } => solution.is_some() as usize,
        }
    }

    /// Count of unsolved leaves in this subtree.
    ///
    /// If you also need `solved_count`, prefer [`leaf_stats`](Self::leaf_stats)
    /// which fuses both in a single traversal.
    #[inline]
    pub fn unsolved_count(&self) -> usize {
        match self {
            Self::Or { children, .. } | Self::And { children, .. } => {
                children.iter().map(|c| c.unsolved_count()).sum()
            }
            Self::Leaf { solution, .. } => solution.is_none() as usize,
        }
    }

    /// Fused `(solved_count, unsolved_count)` in a single traversal.
    ///
    /// Equivalent to `(self.solved_count(), self.unsolved_count())` but only
    /// walks the tree once.
    #[inline]
    pub fn leaf_stats(&self) -> (usize, usize) {
        match self {
            Self::Or { children, .. } | Self::And { children, .. } => {
                let mut solved = 0usize;
                let mut unsolved = 0usize;
                for child in children {
                    let (s, u) = child.leaf_stats();
                    solved += s;
                    unsolved += u;
                }
                (solved, unsolved)
            }
            Self::Leaf { solution, .. } => match solution.is_some() {
                true => (1, 0),
                false => (0, 1),
            },
        }
    }
}

impl<G: fmt::Debug, S: fmt::Debug> fmt::Display for AndOrNode<G, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Or {
                goal,
                children,
                best,
                ..
            } => {
                write!(
                    f,
                    "OR(goal={:?}, children={}, best={:?})",
                    goal,
                    children.len(),
                    best
                )
            }
            Self::And {
                goal,
                children,
                solved_count,
                solved,
                ..
            } => {
                write!(
                    f,
                    "AND(goal={:?}, children={}, solved={}/{})",
                    goal,
                    children.len(),
                    solved_count,
                    solved.len()
                )
            }
            Self::Leaf { goal, solution, .. } => {
                write!(
                    f,
                    "LEAF(goal={:?}, {})",
                    goal,
                    if solution.is_some() {
                        "solved"
                    } else {
                        "unsolved"
                    }
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: simple goal type for tests.
    #[derive(Debug, Clone, PartialEq)]
    struct Goal(String);

    /// Helper: simple solution type for tests.
    #[derive(Debug, Clone, PartialEq)]
    struct Solution(Vec<usize>);

    fn g(s: &str) -> Goal {
        Goal(s.to_string())
    }

    fn sol(tokens: &[usize]) -> Solution {
        Solution(tokens.to_vec())
    }

    // ── Construction tests ────────────────────────────────────────

    #[test]
    fn test_or_node_empty() {
        let node: AndOrNode<Goal, Solution> = AndOrNode::or(g("root"));
        assert!(!node.is_solved());
        assert_eq!(node.child_count(), 0);
        assert_eq!(node.depth(), 0);
    }

    #[test]
    fn test_and_node_empty() {
        let node: AndOrNode<Goal, Solution> = AndOrNode::and(g("root"));
        assert!(!node.is_solved()); // empty AND = no children to solve
        assert_eq!(node.child_count(), 0);
    }

    #[test]
    fn test_solved_leaf() {
        let node = AndOrNode::solved_leaf(g("goal"), sol(&[1, 2, 3]));
        assert!(node.is_solved());
        assert_eq!(node.depth(), 0);
        assert_eq!(node.node_count(), 1);
    }

    #[test]
    fn test_unsolved_leaf() {
        let node: AndOrNode<Goal, Solution> = AndOrNode::unsolved_leaf(g("goal"));
        assert!(!node.is_solved());
    }

    // ── Child access tests ────────────────────────────────────────

    #[test]
    fn test_push_child_or() {
        let mut root: AndOrNode<Goal, Solution> = AndOrNode::or(g("root"));
        root.push_child(AndOrNode::solved_leaf(g("a"), sol(&[1])));
        root.push_child(AndOrNode::unsolved_leaf(g("b")));
        assert_eq!(root.child_count(), 2);
        assert!(root.child(0).unwrap().is_solved());
        assert!(!root.child(1).unwrap().is_solved());
    }

    #[test]
    fn test_push_child_and() {
        let mut root: AndOrNode<Goal, Solution> = AndOrNode::and(g("root"));
        root.push_child(AndOrNode::solved_leaf(g("a"), sol(&[1])));
        root.push_child(AndOrNode::unsolved_leaf(g("b")));
        assert_eq!(root.child_count(), 2);
    }

    #[test]
    fn test_push_child_leaf_noop() {
        let mut node = AndOrNode::solved_leaf(g("leaf"), sol(&[1]));
        node.push_child(AndOrNode::unsolved_leaf(g("x")));
        assert_eq!(node.child_count(), 0); // leaf has no children
    }

    // ── Solved propagation tests ──────────────────────────────────

    #[test]
    fn test_or_solved_when_any_child_solved() {
        let mut root: AndOrNode<Goal, Solution> = AndOrNode::or(g("root"));
        root.push_child(AndOrNode::unsolved_leaf(g("a")));
        root.push_child(AndOrNode::solved_leaf(g("b"), sol(&[2])));
        assert!(root.is_solved()); // OR: any child solved
    }

    #[test]
    fn test_or_solved_via_best() {
        let mut root: AndOrNode<Goal, Solution> = AndOrNode::or(g("root"));
        root.push_child(AndOrNode::unsolved_leaf(g("a")));
        root.push_child(AndOrNode::solved_leaf(g("b"), sol(&[2])));
        root.push_child(AndOrNode::unsolved_leaf(g("c")));
        assert!(root.set_best(1));
        assert!(root.is_solved()); // best points to solved child
    }

    #[test]
    fn test_or_not_solved_when_best_points_unsolved() {
        let mut root: AndOrNode<Goal, Solution> = AndOrNode::or(g("root"));
        root.push_child(AndOrNode::solved_leaf(g("a"), sol(&[1])));
        root.push_child(AndOrNode::unsolved_leaf(g("b")));
        assert!(root.set_best(1)); // best points to unsolved
        assert!(!root.is_solved()); // best child isn't solved
    }

    #[test]
    fn test_and_solved_when_all_children_solved() {
        let mut root: AndOrNode<Goal, Solution> = AndOrNode::and(g("root"));
        root.push_child(AndOrNode::unsolved_leaf(g("a")));
        root.push_child(AndOrNode::unsolved_leaf(g("b")));
        assert!(!root.is_solved());

        assert!(root.mark_child_solved(0));
        assert!(!root.is_solved()); // only 1/2 solved

        // Solve child 0's leaf
        root.child_mut(0).unwrap().set_solution(sol(&[1]));
        root.mark_child_solved(0);
        // child 1 still unsolved
        assert!(!root.is_solved());

        root.child_mut(1).unwrap().set_solution(sol(&[2]));
        root.mark_child_solved(1);
        assert!(root.is_solved()); // all children solved
    }

    // ── Setters/mutators ──────────────────────────────────────────

    #[test]
    fn test_set_solution_leaf() {
        let mut node: AndOrNode<Goal, Solution> = AndOrNode::unsolved_leaf(g("g"));
        node.set_solution(sol(&[42]));
        assert!(node.is_solved());
    }

    #[test]
    fn test_set_sketch_and() {
        let mut node: AndOrNode<Goal, Solution> = AndOrNode::and(g("g"));
        node.set_sketch(sol(&[1, 2, 3]));
        assert_eq!(node.take_sketch(), Some(sol(&[1, 2, 3])));
        assert_eq!(node.take_sketch(), None);
    }

    #[test]
    fn test_take_solution() {
        let mut node = AndOrNode::solved_leaf(g("g"), sol(&[1, 2]));
        assert_eq!(node.take_solution(), Some(sol(&[1, 2])));
        assert_eq!(node.take_solution(), None);
        assert!(!node.is_solved());
    }

    #[test]
    fn test_set_best_out_of_bounds() {
        let mut root: AndOrNode<Goal, Solution> = AndOrNode::or(g("root"));
        root.push_child(AndOrNode::unsolved_leaf(g("a")));
        assert!(!root.set_best(5)); // out of bounds
    }

    #[test]
    fn test_mark_child_solved_non_and() {
        let mut node: AndOrNode<Goal, Solution> = AndOrNode::or(g("root"));
        assert!(!node.mark_child_solved(0)); // not an AND node
    }

    // ── Tree metrics ──────────────────────────────────────────────

    #[test]
    fn test_node_count_complex_tree() {
        // OR → [AND → [Leaf, Leaf], Leaf]
        let mut root: AndOrNode<Goal, Solution> = AndOrNode::or(g("root"));
        let mut and_node: AndOrNode<Goal, Solution> = AndOrNode::and(g("decomp"));
        and_node.push_child(AndOrNode::unsolved_leaf(g("sub1")));
        and_node.push_child(AndOrNode::unsolved_leaf(g("sub2")));
        root.push_child(and_node);
        root.push_child(AndOrNode::unsolved_leaf(g("alt")));
        assert_eq!(root.node_count(), 5); // root + and + 2 leaves + alt leaf
        assert_eq!(root.depth(), 2); // root → and → leaf
    }

    #[test]
    fn test_solved_unsolved_counts() {
        let mut root: AndOrNode<Goal, Solution> = AndOrNode::or(g("root"));
        root.push_child(AndOrNode::solved_leaf(g("a"), sol(&[1])));
        root.push_child(AndOrNode::unsolved_leaf(g("b")));
        root.push_child(AndOrNode::solved_leaf(g("c"), sol(&[3])));
        assert_eq!(root.solved_count(), 2);
        assert_eq!(root.unsolved_count(), 1);
    }

    #[test]
    fn test_leaf_stats_matches_individual() {
        let mut root: AndOrNode<Goal, Solution> = AndOrNode::or(g("root"));
        root.push_child(AndOrNode::solved_leaf(g("a"), sol(&[1])));
        root.push_child(AndOrNode::unsolved_leaf(g("b")));
        root.push_child(AndOrNode::solved_leaf(g("c"), sol(&[3])));

        let (s, u) = root.leaf_stats();
        assert_eq!(s, root.solved_count());
        assert_eq!(u, root.unsolved_count());
        assert_eq!((s, u), (2, 1));
    }

    // ── Goal access ───────────────────────────────────────────────

    #[test]
    fn test_goal_access() {
        let node: AndOrNode<Goal, Solution> = AndOrNode::or(g("my_goal"));
        assert_eq!(node.goal().0, "my_goal");
    }

    #[test]
    fn test_goal_mut() {
        let mut node: AndOrNode<Goal, Solution> = AndOrNode::or(g("old"));
        node.goal_mut().0 = "new".to_string();
        assert_eq!(node.goal().0, "new");
    }

    // ── Display ───────────────────────────────────────────────────

    #[test]
    fn test_display_or() {
        let node: AndOrNode<Goal, Solution> = AndOrNode::or(g("root"));
        let s = format!("{node}");
        assert!(s.contains("OR"));
        assert!(s.contains("root"));
    }

    #[test]
    fn test_display_and() {
        let mut node: AndOrNode<Goal, Solution> = AndOrNode::and(g("decomp"));
        node.push_child(AndOrNode::solved_leaf(g("a"), sol(&[1])));
        node.mark_child_solved(0);
        let s = format!("{node}");
        assert!(s.contains("AND"));
        assert!(s.contains("solved=1/1"));
    }

    #[test]
    fn test_display_leaf() {
        let node = AndOrNode::solved_leaf(g("goal"), sol(&[1]));
        let s = format!("{node}");
        assert!(s.contains("LEAF"));
        assert!(s.contains("solved"));
    }
}
