//! `MuxBfs` — dynamic-width BFS expansion strategy for MUX tree search.
//!
//! Reads logit distributions to determine how many branches to expand
//! at each depth of the DD-tree.

use crate::mux::dd_tree::MuxDdTree;
#[cfg(feature = "comp_width")]
use crate::mux::dd_tree::compositional_width;
#[cfg(feature = "comp_width")]
use crate::mux::top_k::{MAX_TOP_K, extract_top_k_into};
#[cfg(not(feature = "comp_width"))]
use crate::mux::top_k::{MAX_TOP_K, extract_top_k_into};

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
    /// Without: binary threshold on top-1 dominance ratio. Zero-alloc.
    pub fn detect_width(&self, logits: &[f32]) -> usize {
        let mut buf = [0.0f32; MAX_TOP_K];
        let peaks = extract_top_k_into(logits, self.k, &mut buf);
        self.detect_width_with_peaks(peaks)
    }

    /// Same contract as [`Self::detect_width`] but accepts pre-extracted
    /// top-K peaks. Avoids the redundant `extract_top_k_into` when the
    /// caller already has the peaks (e.g. for `is_valid_with_peaks` or
    /// `MuxDdTree::expand_node_with_peaks` in the same BFS step).
    #[inline]
    pub fn detect_width_with_peaks(&self, peaks: &[f32]) -> usize {
        #[cfg(feature = "comp_width")]
        {
            if peaks.len() < 2 {
                return 1;
            }
            let total: f32 = peaks.iter().sum();
            if total <= 0.0 {
                return 1;
            }
            compositional_width(peaks, self.k).max(1)
        }

        #[cfg(not(feature = "comp_width"))]
        {
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
    ///
    /// Top-K peaks are extracted once per leaf and reused across width
    /// detection, pruning, and node expansion — 3× fewer `extract_top_k_into`
    /// passes compared to the naïve per-call extraction.
    pub fn step(&self, tree: &mut MuxDdTree, depth: usize, logits_by_leaf: &[Vec<f32>]) {
        let leaves = tree.collect_leaf_paths_flat();
        assert_eq!(
            leaves.len(),
            logits_by_leaf.len(),
            "logits count must match leaf count"
        );

        for (i, logits) in logits_by_leaf.iter().enumerate() {
            // Extract once — reused for width, validity, and expansion.
            let mut buf = [0.0f32; MAX_TOP_K];
            let peaks = extract_top_k_into(logits, tree.k, &mut buf);
            let width = self.detect_width_with_peaks(peaks);
            if tree.pruner.is_valid_with_peaks(peaks) {
                tree.expand_node_with_peaks(leaves.path(i), peaks, width);
            }
        }
        let _ = depth; // preserved for API compat; pruner ignores depth
    }

    /// BFS step with dendritic-gated dynamic width.
    ///
    /// Each expansion's width is modulated by `gate.compute_gate()`:
    /// `comp_width = (width as f32 * nmda_gate).max(1.0) as usize`
    ///
    /// Minimum width is 1 — always expand at least one candidate.
    /// Feature-gated behind `dendritic_gate`.
    #[cfg(feature = "dendritic_gate")]
    pub fn step_dendritic(
        &self,
        tree: &mut MuxDdTree,
        depth: usize,
        logits_by_leaf: &[Vec<f32>],
        gate: &crate::dendritic_gate::DendriticGate,
        entropy_by_leaf: &[f32],
        coincidence_by_leaf: &[f32],
    ) {
        let leaves = tree.collect_leaf_paths_flat();
        assert_eq!(
            leaves.len(),
            logits_by_leaf.len(),
            "logits count must match leaf count"
        );

        for (i, logits) in logits_by_leaf.iter().enumerate() {
            // Extract once — reused for width and expansion (pruner check skipped here
            // since the original used is_valid inside the guard; preserved below).
            let mut buf = [0.0f32; MAX_TOP_K];
            let peaks = extract_top_k_into(logits, tree.k, &mut buf);
            if tree.pruner.is_valid_with_peaks(peaks) {
                let base_width = self.detect_width_with_peaks(peaks);
                let nmda_gate = gate.compute_gate(
                    *entropy_by_leaf.get(i).unwrap_or(&1.0),
                    *coincidence_by_leaf.get(i).unwrap_or(&0.5),
                );
                let gated_width = ((base_width as f32) * nmda_gate).max(1.0) as usize;
                tree.expand_node_with_peaks(leaves.path(i), peaks, gated_width);
            }
        }
        let _ = depth; // preserved for API compat; pruner ignores depth
    }

    /// Zero-alloc variant of `step` that reuses a caller-provided `LeafPaths` buffer.
    ///
    /// The `leaves` buffer is cleared and refilled each call, retaining its heap
    /// capacity across BFS steps — the biggest allocation hot-spot in the BFS loop.
    pub fn step_into(
        &self,
        tree: &mut MuxDdTree,
        depth: usize,
        logits_by_leaf: &[Vec<f32>],
        leaves: &mut crate::mux::dd_tree::LeafPaths,
    ) {
        tree.collect_leaf_paths_flat_into(leaves);
        assert_eq!(
            leaves.len(),
            logits_by_leaf.len(),
            "logits count must match leaf count"
        );

        for (i, logits) in logits_by_leaf.iter().enumerate() {
            // Extract once — reused for width, validity, and expansion.
            let mut buf = [0.0f32; MAX_TOP_K];
            let peaks = extract_top_k_into(logits, tree.k, &mut buf);
            let width = self.detect_width_with_peaks(peaks);
            if tree.pruner.is_valid_with_peaks(peaks) {
                tree.expand_node_with_peaks(leaves.path(i), peaks, width);
            }
        }
        let _ = depth; // preserved for API compat; pruner ignores depth
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

        bfs.step(&mut tree, 1, std::slice::from_ref(&logits));
        assert!(tree.leaf_count() > 1);
    }
}
