//! `MuxDdTree` — superposition DD-tree with BFS frontier mode.
//!
//! Each node carries K tokens as a weighted span (superposition).
//! `hypothesis_coverage()` = `leaf_count() * K^depth`.
//!
//! BFS frontier mode reads logit distributions at each depth, detects the
//! effective width (number of valid superposition peaks), and expands all
//! peaks simultaneously.

use crate::mux::span_pruner::MuxSpanPruner;
use crate::mux::top_k::extract_top_k_peaks;

/// Default superposition width (number of tokens per node).
pub const DEFAULT_K: usize = 4;

/// A node in the DD-tree that carries K tokens as a weighted span.
#[derive(Debug, Clone)]
pub struct MuxNode {
    /// Token IDs held in superposition at this node.
    pub tokens: Vec<u32>,
    /// Corresponding weights (logit values) for each token.
    pub weights: Vec<f32>,
    /// Child nodes (branching factor = width at this depth).
    pub children: Vec<MuxNode>,
}

impl MuxNode {
    pub fn new(tokens: Vec<u32>, weights: Vec<f32>) -> Self {
        assert_eq!(tokens.len(), weights.len());
        Self {
            tokens,
            weights,
            children: Vec::new(),
        }
    }

    /// Number of tokens in the superposition span.
    pub fn span_size(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if this node is a leaf (no children).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Shannon entropy of a probability distribution (in nats).
/// Zero-alloc, branch-free inner loop.
#[cfg(feature = "comp_width")]
fn shannon_entropy(peaks: &[f32]) -> f32 {
    let total: f32 = peaks.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let inv_total = 1.0 / total;
    let mut h = 0.0f32;
    for &p in peaks {
        let pn = p * inv_total;
        if pn > 0.0 {
            h -= pn * pn.ln();
        }
    }
    h
}

/// Compositional DDTree partner-entropy width (Plan 205, Research 181).
///
/// Replaces binary `PEAK_DOMINANCE_RATIO` with continuous scaling.
/// Maps normalized entropy ∈ [0, 1] → width ∈ [1, base]:
///
/// ```text
/// width = max(1, round(base * normalized_entropy^alpha))
/// ```
///
/// Where `alpha` controls the shape:
/// - alpha < 1: aggressively widens (slight entropy → wide)
/// - alpha = 1: linear
/// - alpha > 1: conservatively widens (needs high entropy to widen)
///
/// Uses CM isotropic scale internally for the norm estimate:
/// `s = (normalized + damping).recip().sqrt()` — one division, one sqrt, zero-alloc.
#[cfg(feature = "comp_width")]
fn compositional_width(peaks: &[f32], base: usize) -> usize {
    let entropy = shannon_entropy(peaks);
    // max entropy for uniform distribution over len items: ln(n)
    let max_entropy = (peaks.len() as f32).ln();
    if max_entropy <= 0.0 {
        return 1;
    }
    let normalized = (entropy / max_entropy).clamp(0.0, 1.0);
    // Width scales linearly with normalized entropy: peaked→1, uniform→base
    let width = (base as f32 * normalized).round() as usize;
    width.max(1)
}

/// DD-tree wrapper that manages superposition expansion.
#[derive(Debug, Clone)]
pub struct MuxDdTree {
    /// Root node.
    pub root: MuxNode,
    /// Maximum superposition width per node.
    pub k: usize,
    /// Current depth of the tree.
    pub depth: usize,
    /// Pruner for validating superposition spans.
    pub pruner: MuxSpanPruner,
}

impl MuxDdTree {
    pub fn new(k: usize) -> Self {
        let pruner = MuxSpanPruner::new(k, 0.5);
        Self {
            root: MuxNode::new(Vec::new(), Vec::new()),
            k,
            depth: 0,
            pruner,
        }
    }

    /// Initialize the root with an initial superposition from logit distribution.
    pub fn init_root(&mut self, logits: &[f32]) {
        let peaks = extract_top_k_peaks(logits, self.k);
        let tokens: Vec<u32> = (0..peaks.len() as u32).collect();
        self.root = MuxNode::new(tokens, peaks);
        self.depth = 0;
    }

    /// Count leaf nodes in the tree.
    pub fn leaf_count(&self) -> usize {
        Self::count_leaves(&self.root)
    }

    fn count_leaves(node: &MuxNode) -> usize {
        if node.is_leaf() {
            1
        } else {
            node.children.iter().map(Self::count_leaves).sum()
        }
    }

    /// Total hypothesis coverage: `leaf_count * K^depth`.
    pub fn hypothesis_coverage(&self) -> usize {
        let k_pow = self.k.pow(self.depth as u32);
        self.leaf_count() * k_pow
    }

    /// Expand a leaf node at the given path using logit distribution.
    /// Creates `width` children, each with top-K tokens from `logits`.
    pub fn expand_node(&mut self, path: &[usize], logits: &[f32], width: usize) {
        let node = Self::get_node_mut(&mut self.root, path);
        let peaks = extract_top_k_peaks(logits, self.k);
        let effective_width = width.min(peaks.len()).max(1);

        for i in 0..effective_width {
            // Distribute peaks across children: each child gets a shifted view
            let offset = (i * self.k / effective_width).min(peaks.len());
            let child_tokens: Vec<u32> =
                (offset as u32..(offset + peaks.len().min(self.k)) as u32).collect();
            let child_weights: Vec<f32> = peaks.iter().take(self.k).copied().collect();
            node.children
                .push(MuxNode::new(child_tokens, child_weights));
        }

        // Track maximum depth
        let new_depth = path.len() + 1;
        if new_depth > self.depth {
            self.depth = new_depth;
        }
    }

    /// **BFS frontier mode**: expand all current leaves simultaneously using
    /// per-depth logit distributions and dynamic width detection.
    ///
    /// For each leaf, reads the logit distribution, determines the effective
    /// width via `detect_width`, validates with the pruner, and expands.
    pub fn expand_bfs_frontier<F>(&mut self, depth: usize, logits_by_leaf: &[F])
    where
        F: AsRef<[f32]>,
    {
        let leaves = self.collect_leaf_paths();
        assert_eq!(
            leaves.len(),
            logits_by_leaf.len(),
            "must provide logits for every leaf"
        );

        for (path, logits) in leaves.into_iter().zip(logits_by_leaf.iter()) {
            let logits = logits.as_ref();
            let width = self.detect_width(logits);
            if width > 0 && self.pruner.is_valid(logits, depth) {
                self.expand_node(&path, logits, width);
            }
        }
    }

    /// Detect the effective branching width from a logit distribution.
    ///
    /// With `comp_width` feature: uses continuous partner-entropy scaling
    /// derived from Compositional Muon's isotropic approximation.
    /// Without: falls back to binary PEAK_DOMINANCE_RATIO threshold.
    pub fn detect_width(&self, logits: &[f32]) -> usize {
        let peaks = extract_top_k_peaks(logits, self.k);
        if peaks.len() < 2 {
            return 1;
        }
        let total: f32 = peaks.iter().sum();
        if total <= 0.0 {
            return 1;
        }

        #[cfg(feature = "comp_width")]
        {
            let width = compositional_width(&peaks, self.k);
            width.max(1)
        }

        #[cfg(not(feature = "comp_width"))]
        {
            let top_ratio = peaks[0] / total;
            if top_ratio > 0.8 {
                1
            } else {
                peaks.len().min(self.k)
            }
        }
    }

    /// Collect paths to all leaf nodes (BFS order).
    pub fn collect_leaf_paths(&self) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut queue: Vec<(Vec<usize>, &MuxNode)> = vec![(Vec::new(), &self.root)];
        while let Some((path, node)) = queue.pop() {
            if node.is_leaf() {
                result.push(path);
            } else {
                for (i, child) in node.children.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(i);
                    queue.push((child_path, child));
                }
            }
        }
        result
    }

    fn get_node_mut<'a>(node: &'a mut MuxNode, path: &[usize]) -> &'a mut MuxNode {
        let mut current = node;
        for &idx in path {
            current = &mut current.children[idx];
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_root_and_leaf_count() {
        let mut tree = MuxDdTree::new(4);
        let logits = vec![0.1, 1.0, 0.2, 0.7, 0.05, 0.5, 0.0, 0.3];
        tree.init_root(&logits);
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.root.span_size(), 4); // top-4 peaks
        assert_eq!(tree.depth, 0);
    }

    #[test]
    fn hypothesis_coverage_formula() {
        let mut tree = MuxDdTree::new(4);
        let logits = vec![0.1, 1.0, 0.2, 0.7, 0.05, 0.5, 0.0, 0.3];
        tree.init_root(&logits);
        // 1 leaf * 4^0 = 1
        assert_eq!(tree.hypothesis_coverage(), 1);
    }

    #[test]
    fn expand_node_increases_leaves() {
        let mut tree = MuxDdTree::new(4);
        let logits = vec![0.1, 1.0, 0.2, 0.7, 0.05, 0.5, 0.0, 0.3];
        tree.init_root(&logits);
        tree.expand_node(&[], &logits, 2);
        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.depth, 1);
        // 2 leaves * 4^1 = 8
        assert_eq!(tree.hypothesis_coverage(), 8);
    }

    #[test]
    #[cfg(not(feature = "comp_width"))]
    fn detect_width_peaked() {
        let tree = MuxDdTree::new(4);
        // Single dominant peak (1.0 is > 80% of total)
        let logits = vec![1.0, 0.05, 0.03, 0.02, 0.01];
        assert_eq!(tree.detect_width(&logits), 1);
    }

    #[test]
    #[cfg(not(feature = "comp_width"))]
    fn detect_width_multi_peak() {
        let tree = MuxDdTree::new(4);
        // Spread across multiple peaks
        let logits = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        assert_eq!(tree.detect_width(&logits), 4);
    }

    #[test]
    fn bfs_frontier_expansion() {
        let mut tree = MuxDdTree::new(4);
        let logits = vec![0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01];
        tree.init_root(&logits);
        assert_eq!(tree.leaf_count(), 1);

        // Expand frontier: multi-peak distribution should expand to width 4
        let leaf_logits: Vec<Vec<f32>> = vec![logits.clone()];
        tree.expand_bfs_frontier(1, &leaf_logits);
        assert!(tree.leaf_count() > 1);
        assert_eq!(tree.depth, 1);
    }

    // ── Plan 205: comp_width tests ──────────────────────────────

    #[cfg(feature = "comp_width")]
    #[test]
    fn comp_width_zero_entropy_returns_min() {
        // Zero entropy: all mass on one token → width should be 1
        let peaks = vec![1.0, 0.0, 0.0, 0.0];
        let w = compositional_width(&peaks, 4);
        assert_eq!(w, 1, "zero entropy should give width 1, got {w}");
    }

    #[cfg(feature = "comp_width")]
    #[test]
    fn comp_width_uniform_entropy_returns_base() {
        // Max entropy: uniform distribution → width should be base
        let peaks = vec![0.25, 0.25, 0.25, 0.25];
        let w = compositional_width(&peaks, 4);
        assert_eq!(w, 4, "uniform should give full width, got {w}");
    }

    #[cfg(feature = "comp_width")]
    #[test]
    fn comp_width_monotonic_with_entropy() {
        // Higher entropy → width should be >= lower entropy width
        let low_entropy = vec![0.9, 0.05, 0.03, 0.02];
        let high_entropy = vec![0.3, 0.3, 0.2, 0.2];
        let w_low = compositional_width(&low_entropy, 8);
        let w_high = compositional_width(&high_entropy, 8);
        assert!(
            w_high >= w_low,
            "high entropy width ({w_high}) should be >= low entropy width ({w_low})"
        );
    }

    #[cfg(feature = "comp_width")]
    #[test]
    fn comp_width_detect_width_peaked_gives_small() {
        let tree = MuxDdTree::new(4);
        // Very peaked: top-1 dominates
        let logits = vec![1.0, 0.05, 0.03, 0.02, 0.01];
        let w = tree.detect_width(&logits);
        assert!(
            w <= 2,
            "peaked distribution should give small width, got {w}"
        );
    }

    #[cfg(feature = "comp_width")]
    #[test]
    fn comp_width_detect_width_uniform_gives_full() {
        let tree = MuxDdTree::new(4);
        // Uniform distribution
        let logits = vec![0.25, 0.25, 0.25, 0.25];
        let w = tree.detect_width(&logits);
        assert_eq!(w, 4, "uniform distribution should give full width");
    }

    #[cfg(feature = "comp_width")]
    #[test]
    fn shannon_entropy_values() {
        // Uniform over 4: H = ln(4) ≈ 1.386
        let uniform = vec![0.25_f32, 0.25, 0.25, 0.25];
        let h = shannon_entropy(&uniform);
        let expected = (4.0_f32).ln();
        assert!(
            (h - expected).abs() < 0.01,
            "expected {expected:.3}, got {h:.3}"
        );

        // Degenerate (all on one): H = 0
        let degenerate = vec![1.0_f32, 0.0, 0.0, 0.0];
        let h0 = shannon_entropy(&degenerate);
        assert!(
            h0.abs() < 0.001,
            "degenerate entropy should be ~0, got {h0}"
        );
    }
}
