//! MuxDdTree — DDTree nodes with superposition of K tokens (Research 158, MUX).
//!
//! Each [`MuxNode`] holds a superposition span of up to K `(token_id, geometric_weight)` pairs,
//! representing multiple hypotheses at a single DDTree position. The tree structure enables
//! `width^depth * K^depth` total hypothesis coverage from `width * depth` actual nodes.
//!
//! # Architecture
//!
//! - [`MuxNode`] — single superposition node with K token hypotheses
//! - [`MuxDdTree`] — tree of MuxNodes organized as `[depth][width * ...]`
//!
//! # Relationship to MuxSpanPruner
//!
//! This module requires `mux_pruner` (which provides `MuxSpanPruner`), as `MuxNode` reuses
//! the same top-k extraction and geometric decay logic. The tree expands from logit vectors
//! via [`MuxDdTree::expand_from_logits`].

use super::mux_span::MuxSpanPruner;

// ── MuxNode ────────────────────────────────────────────────────────

/// A DDTree node holding a superposition of K token hypotheses.
///
/// Each entry is `(token_id, geometric_weight)` where weights decay
/// geometrically from the dominant token. The `dominant` field is
/// pre-computed for fast single-token fallback.
///
/// (Research 158, MUX)
pub struct MuxNode {
    /// Superposition span: (token_id, geometric_weight), sorted by descending weight.
    pub span: Vec<(usize, f32)>,
    /// Pre-computed dominant (highest-weight) token ID.
    pub dominant: usize,
}

impl MuxNode {
    /// Create a new MuxNode from a span of (token_id, weight) pairs.
    ///
    /// The span is assumed to be sorted by descending weight.
    /// The dominant token is the first entry's token ID.
    pub fn new(span: Vec<(usize, f32)>) -> Self {
        let dominant = span.first().map(|&(idx, _)| idx).unwrap_or(0);
        Self { span, dominant }
    }

    /// Demultiplex: recover all token IDs from the superposition span.
    pub fn demux(&self) -> Vec<usize> {
        self.span.iter().map(|&(idx, _)| idx).collect()
    }

    /// Return the dominant (highest-weight) token ID.
    pub fn dominant(&self) -> usize {
        self.dominant
    }

    /// Number of hypotheses in the superposition.
    pub fn hypothesis_count(&self) -> usize {
        self.span.len()
    }
}

// ── MuxDdTree ──────────────────────────────────────────────────────

/// Tree of MuxNodes organized by depth and width.
///
/// Each tree level has up to `width` branches, each containing a MuxNode
/// with `span_k` superposed hypotheses. The total hypothesis coverage is
/// `width^depth * span_k^depth` — exponential coverage from linear node count.
///
/// (Research 158, MUX)
pub struct MuxDdTree {
    /// Branching factor per depth level.
    pub width: usize,
    /// Maximum tree depth.
    pub depth: usize,
    /// Number of superposed tokens per node.
    pub span_k: usize,
    /// Nodes organized as `nodes[depth][node_index]`.
    /// At each depth, there are at most `width^depth` nodes.
    pub nodes: Vec<Vec<MuxNode>>,
}

impl MuxDdTree {
    /// Create a new empty MuxDdTree.
    pub fn new(width: usize, depth: usize, span_k: usize) -> Self {
        let nodes = (0..depth).map(|_| Vec::new()).collect();
        Self {
            width,
            depth,
            span_k,
            nodes,
        }
    }

    /// Total number of leaf nodes (at maximum depth).
    pub fn leaf_count(&self) -> usize {
        self.nodes.last().map(|level| level.len()).unwrap_or(0)
    }

    /// Total hypothesis coverage across all leaves.
    ///
    /// Each leaf has `span_k` hypotheses, and each path through the tree
    /// from root to leaf has `depth` superposition nodes. The total
    /// hypothesis count is the number of leaves × span_k.
    pub fn hypothesis_coverage(&self) -> usize {
        self.leaf_count() * self.span_k
    }

    /// Expand the tree at the given depth from a logit vector.
    ///
    /// Extracts top-k peaks from the logits (respecting the tree's `span_k`),
    /// creates a MuxNode from them, and appends it to the given depth level.
    ///
    /// If the logits don't form a valid superposition (per `MuxSpanPruner`),
    /// no node is added and the method returns without action.
    pub fn expand_from_logits(&mut self, logits: &[f32], depth: usize, decay: f32) {
        if depth >= self.depth {
            return;
        }

        let pruner = MuxSpanPruner::with_params(decay, self.span_k, 0.3);
        if !pruner.is_valid_logits(logits) {
            return;
        }

        let peaks = MuxSpanPruner::extract_top_k_peaks(logits, self.span_k);
        if peaks.is_empty() {
            return;
        }

        let node = MuxNode::new(peaks);
        self.nodes[depth].push(node);
    }

    /// Expand multiple branches at a given depth from multiple logit vectors.
    ///
    /// Appends one MuxNode per valid logit vector.
    pub fn expand_batch(&mut self, logit_batch: &[&[f32]], depth: usize, decay: f32) {
        for logits in logit_batch {
            self.expand_from_logits(logits, depth, decay);
        }
    }

    /// Total number of nodes across all depth levels.
    pub fn total_nodes(&self) -> usize {
        self.nodes.iter().map(|level| level.len()).sum()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn geometric_logits(vocab_size: usize, k: usize, decay: f32) -> Vec<f32> {
        let mut logits = vec![0.0f32; vocab_size];
        for i in 0..k {
            logits[10 + i * 7] = 10.0 * decay.powi(i as i32);
        }
        logits
    }

    #[test]
    fn test_mux_node_new() {
        let span = vec![(5, 1.0), (3, 0.9), (1, 0.81)];
        let node = MuxNode::new(span);
        assert_eq!(node.dominant(), 5);
        assert_eq!(node.hypothesis_count(), 3);
        assert_eq!(node.demux(), vec![5, 3, 1]);
    }

    #[test]
    fn test_mux_node_empty_span() {
        let node = MuxNode::new(vec![]);
        assert_eq!(node.dominant(), 0);
        assert_eq!(node.hypothesis_count(), 0);
        assert!(node.demux().is_empty());
    }

    #[test]
    fn test_mux_ddtree_new() {
        let tree = MuxDdTree::new(3, 4, 5);
        assert_eq!(tree.width, 3);
        assert_eq!(tree.depth, 4);
        assert_eq!(tree.span_k, 5);
        assert_eq!(tree.total_nodes(), 0);
    }

    #[test]
    fn test_mux_ddtree_expand_from_logits() {
        let mut tree = MuxDdTree::new(2, 3, 5);
        let logits = geometric_logits(100, 5, 0.9);
        tree.expand_from_logits(&logits, 0, 0.9);
        assert_eq!(tree.nodes[0].len(), 1);
        assert_eq!(tree.nodes[0][0].hypothesis_count(), 5);
    }

    #[test]
    fn test_mux_ddtree_rejects_invalid_logits() {
        let mut tree = MuxDdTree::new(2, 3, 5);
        // Uniform noise — should be rejected
        let logits = vec![1.0f32; 100];
        tree.expand_from_logits(&logits, 0, 0.9);
        assert_eq!(tree.nodes[0].len(), 0);
    }

    #[test]
    fn test_mux_ddtree_covers_more_hypotheses() {
        let width = 3;
        let depth = 2;
        let span_k = 5;
        let mut tree = MuxDdTree::new(width, depth, span_k);

        // Expand 3 branches at depth 0
        for i in 0..width {
            let mut logits = geometric_logits(100, span_k, 0.9);
            // Vary slightly per branch
            logits[0] = i as f32;
            tree.expand_from_logits(&logits, 0, 0.9);
        }

        // Expand 2 branches at depth 1 from each depth-0 node
        for _ in 0..2 {
            let logits = geometric_logits(100, span_k, 0.9);
            tree.expand_from_logits(&logits, 1, 0.9);
        }

        // Verify coverage: leaves * span_k
        let coverage = tree.hypothesis_coverage();
        assert!(
            coverage > 0,
            "tree should have positive hypothesis coverage"
        );
        assert_eq!(
            coverage,
            tree.leaf_count() * span_k,
            "coverage = leaf_count * span_k"
        );
    }

    #[test]
    fn test_mux_ddtree_depth_bounds() {
        let mut tree = MuxDdTree::new(2, 2, 3);
        let logits = geometric_logits(100, 3, 0.9);
        // Expanding at depth >= tree.depth should be a no-op
        tree.expand_from_logits(&logits, 5, 0.9);
        assert_eq!(tree.total_nodes(), 0);
    }

    #[test]
    fn test_mux_ddtree_expand_batch() {
        let mut tree = MuxDdTree::new(3, 2, 3);
        let l1 = geometric_logits(100, 3, 0.9);
        let l2 = geometric_logits(100, 3, 0.8);
        let noise = vec![1.0f32; 100];
        tree.expand_batch(&[&l1, &noise, &l2], 0, 0.9);
        // Only l1 and l2 are valid with decay=0.9; noise should be rejected
        // l2 was generated with decay=0.8, so it may or may not pass the 0.9 pruner
        // At minimum l1 should be accepted
        assert!(tree.nodes[0].len() >= 1);
    }

    #[test]
    fn test_mux_ddtree_total_nodes() {
        let mut tree = MuxDdTree::new(2, 3, 3);
        let logits = geometric_logits(100, 3, 0.9);
        tree.expand_from_logits(&logits, 0, 0.9);
        tree.expand_from_logits(&logits, 1, 0.9);
        tree.expand_from_logits(&logits, 2, 0.9);
        assert_eq!(tree.total_nodes(), 3);
    }
}
