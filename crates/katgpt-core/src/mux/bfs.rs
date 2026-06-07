//! `MuxBfs` — dynamic-width BFS expansion strategy for MUX tree search.
//!
//! Reads logit distributions to determine how many branches to expand
//! at each depth of the DD-tree.

use crate::mux::dd_tree::MuxDdTree;
#[cfg(not(feature = "comp_width"))]
use crate::mux::top_k::extract_top_k_peaks;

#[cfg(not(feature = "comp_width"))]
/// Ratio threshold for deciding peaked vs multi-peak distribution.
const PEAK_DOMINANCE_RATIO: f32 = 0.8;

/// BFS engine that drives dynamic-width expansion of a `MuxDdTree`.
#[derive(Debug, Clone)]
pub struct MuxBfs {
    /// Maximum superposition width (matches tree K).
    pub k: usize,
}

impl MuxBfs {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Detect effective branching width from a logit distribution.
    ///
    /// With `comp_width` feature: delegates to `MuxDdTree::detect_width`
    /// which uses continuous partner-entropy scaling.
    /// Without: binary threshold on top-1 dominance ratio.
    pub fn detect_width(&self, logits: &[f32]) -> usize {
        #[cfg(feature = "comp_width")]
        {
            let tree = MuxDdTree::new(self.k);
            tree.detect_width(logits)
        }

        #[cfg(not(feature = "comp_width"))]
        {
            let peaks = extract_top_k_peaks(logits, self.k);
            if peaks.len() < 2 {
                return 1;
            }
            let total: f32 = peaks.iter().sum();
            if total <= 0.0 {
                return 1;
            }
            let top_ratio = peaks[0] / total;
            if top_ratio > PEAK_DOMINANCE_RATIO {
                1
            } else {
                peaks.len().min(self.k)
            }
        }
    }

    /// Run one BFS expansion step on the tree: expand all leaves with
    /// the provided per-leaf logit distributions.
    pub fn step(&self, tree: &mut MuxDdTree, depth: usize, logits_by_leaf: &[Vec<f32>]) {
        let leaves = tree.collect_leaf_paths();
        assert_eq!(
            leaves.len(),
            logits_by_leaf.len(),
            "logits count must match leaf count"
        );

        for i in 0..leaves.len() {
            let width = self.detect_width(&logits_by_leaf[i]);
            if tree.pruner.is_valid(&logits_by_leaf[i], depth) {
                tree.expand_node(&leaves[i], &logits_by_leaf[i], width);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_peaked() {
        let bfs = MuxBfs::new(4);
        let logits = vec![1.0, 0.05, 0.03, 0.02];
        assert_eq!(bfs.detect_width(&logits), 1);
    }

    #[test]
    fn detect_multi_peak() {
        let bfs = MuxBfs::new(4);
        let logits = vec![0.5, 0.4, 0.3, 0.2];
        assert_eq!(bfs.detect_width(&logits), 4);
    }

    #[test]
    fn bfs_step_expands_tree() {
        let bfs = MuxBfs::new(4);
        let mut tree = MuxDdTree::new(4);
        let logits = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        tree.init_root(&logits);
        assert_eq!(tree.leaf_count(), 1);

        bfs.step(&mut tree, 1, &[logits.clone()]);
        assert!(tree.leaf_count() > 1);
    }
}
