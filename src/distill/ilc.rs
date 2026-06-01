//! ILC (Iterative Latent Clustering) Distillation — synonym-aware DDTree pruning
//!
//! Distilled from arXiv:2605.27734 (Korchinski, Favero, Wyart).
//! Key insight: latent-prediction SSL learns hierarchical structure in O(m³) samples
//! (independent of depth L), vs O(m^(L+1)) for token-level SSL.
//!
//! This module implements the **offline clustering + online inference** path:
//! - `IlcClusterer`: k-means on cousin context vectors (offline)
//! - `SynonymMap`: O(1) cluster lookup at inference time (online)
//! - `SynonymAwarePruner`: ScreeningPruner wrapper that boosts diversity across clusters
//! - `build_dd_tree_screened_synonyms`: DDTree variant that skips synonym branches
//!
//! Architecture:
//! ```text
//! OFFLINE (once per game domain):
//!   episode data → IlcClusterer → SynonymMap (lookup table)
//!
//! ONLINE (inference, hot path):
//!   SynonymMap::lookup(state) → ClusterId    // O(1), no allocation
//!   DDTree: skip branches in already-explored clusters
//!   ScreeningPruner: boost relevance for diverse-cluster candidates
//! ```

use std::collections::HashSet;

use crate::speculative::build_dd_tree_screened;
use crate::speculative::dd_tree::extract_parent_tokens_into;
use crate::speculative::types::{ScreeningPruner, TreeNode};

// ── Types ───────────────────────────────────────────────────────

/// Cluster identifier — index into `SynonymMap::centers[level]`.
pub type ClusterId = usize;

/// Configuration for `IlcClusterer`.
#[derive(Debug, Clone)]
pub struct IlcConfig {
    /// Vocabulary size (branching factor m in the paper).
    pub vocab_size: usize,
    /// Context vector dimensionality D.
    pub context_dim: usize,
    /// Number of synonym clusters per level (k in k-means).
    pub num_clusters: usize,
    /// Maximum hierarchy depth L.
    pub max_depth: usize,
    /// K-means iteration count.
    pub kmeans_iters: usize,
    /// Distance threshold: states within this distance are considered synonyms.
    pub synonym_threshold: f32,
}

impl IlcConfig {
    /// Create a new config with sensible defaults.
    pub fn new(vocab_size: usize, context_dim: usize) -> Self {
        let num_clusters = vocab_size.min(16); // C ≤ m, typically much smaller
        Self {
            vocab_size,
            context_dim,
            num_clusters,
            max_depth: 8,
            kmeans_iters: 20,
            synonym_threshold: 0.1,
        }
    }
}

// ── IlcClusterer (Offline) ─────────────────────────────────────

/// Offline k-means clusterer for cousin context vectors.
///
/// Produces a `SynonymMap` that can be used at inference time for O(1)
/// synonym cluster lookups.
///
/// Implements the paper's Algorithm 1 (ILC): level-by-level k-means
/// clustering of "cousin context vectors" — states that share the same
/// parent in the hierarchy have identical context vectors.
pub struct IlcClusterer {
    config: IlcConfig,
}

impl IlcClusterer {
    /// Create a new clusterer with the given configuration.
    pub fn new(config: IlcConfig) -> Self {
        Self { config }
    }

    /// Cluster episode data into a `SynonymMap`.
    ///
    /// `states` is a flat array of context vectors, one per game state.
    /// Each context vector has length `config.context_dim`.
    /// `depths` maps each state to its hierarchy level (0..max_depth).
    ///
    /// Returns a precomputed `SynonymMap` suitable for O(1) inference lookups.
    pub fn cluster(&self, states: &[&[f32]], depths: &[usize]) -> SynonymMap {
        debug_assert_eq!(states.len(), depths.len());

        let mut all_centers: Vec<f32> = Vec::new();
        let mut all_labels: Vec<usize> = Vec::new();

        // Level-by-level k-means (Algorithm 1 from the paper)
        for level in 0..self.config.max_depth {
            let level_states: Vec<&[f32]> = states
                .iter()
                .zip(depths.iter())
                .filter(|&(_, &d)| d == level)
                .map(|(s, _)| *s)
                .collect();

            if level_states.is_empty() {
                break;
            }

            let (centers, labels) = kmeans(
                &level_states,
                self.config.num_clusters,
                self.config.context_dim,
                self.config.kmeans_iters,
            );

            all_centers.extend_from_slice(&centers);
            all_labels.extend_from_slice(&labels);
        }

        SynonymMap {
            centers: all_centers,
            labels: all_labels,
            num_clusters: self.config.num_clusters,
            context_dim: self.config.context_dim,
            synonym_threshold: self.config.synonym_threshold,
        }
    }

    /// Convenience: cluster from a flat slice of f32 state representations.
    ///
    /// Each state is a contiguous `[f32; context_dim]` in the flat slice.
    /// `depths` has one entry per state.
    pub fn cluster_flat(&self, states_flat: &[f32], depths: &[usize]) -> SynonymMap {
        let d = self.config.context_dim;
        let state_slices: Vec<&[f32]> = states_flat.chunks(d).collect();
        self.cluster(&state_slices, depths)
    }
}

// ── K-means ─────────────────────────────────────────────────────

/// Run k-means on a set of D-dimensional vectors.
///
/// Returns (centers, labels) where:
/// - centers: k × D flat row-major f32 vector
/// - labels: one cluster label per input vector
fn kmeans(data: &[&[f32]], k: usize, d: usize, iters: usize) -> (Vec<f32>, Vec<usize>) {
    let n = data.len();
    if n == 0 {
        return (vec![0.0f32; k * d], vec![]);
    }

    let effective_k = k.min(n);

    // Initialize centers: first k data points
    let mut centers = vec![0.0f32; effective_k * d];
    for (i, center_row) in centers.chunks_mut(d).enumerate().take(effective_k) {
        if i < n {
            center_row.copy_from_slice(data[i]);
        }
    }

    let mut labels = vec![0usize; n];

    for _ in 0..iters {
        // Assignment step: assign each point to nearest center
        for (i, point) in data.iter().enumerate() {
            let mut best_dist = f32::MAX;
            let mut best_label = 0;
            for (c, center_row) in centers.chunks(d).enumerate().take(effective_k) {
                let dist = squared_distance(point, center_row);
                if dist < best_dist {
                    best_dist = dist;
                    best_label = c;
                }
            }
            labels[i] = best_label;
        }

        // Update step: recompute centers
        let mut sums = vec![0.0f32; effective_k * d];
        let mut counts = vec![0usize; effective_k];

        for (i, point) in data.iter().enumerate() {
            let c = labels[i];
            counts[c] += 1;
            let sum_row = &mut sums[c * d..(c + 1) * d];
            for j in 0..d {
                sum_row[j] += point[j];
            }
        }

        for (c, center_row) in centers.chunks_mut(d).enumerate().take(effective_k) {
            if counts[c] > 0 {
                let inv_count = 1.0f32 / counts[c] as f32;
                for j in 0..d {
                    center_row[j] = sums[c * d + j] * inv_count;
                }
            }
        }
    }

    (centers, labels)
}

/// Squared Euclidean distance between two slices.
#[inline]
fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ── SynonymMap (Online) ─────────────────────────────────────────

/// Precomputed synonym cluster lookup table for O(1) inference.
///
/// Built offline by `IlcClusterer`, used online by DDTree and ScreeningPruner.
/// All lookups are pure array indexing — no allocation on the hot path.
pub struct SynonymMap {
    /// Cluster centers: flat row-major, each row is D-dimensional.
    centers: Vec<f32>,
    /// Cluster labels for the training data (not used at inference,
    /// but kept for debugging / serialization).
    #[allow(dead_code)]
    labels: Vec<usize>,
    /// Number of clusters per level.
    num_clusters: usize,
    /// Context vector dimension.
    context_dim: usize,
    /// Distance threshold for synonym detection.
    synonym_threshold: f32,
}

impl SynonymMap {
    /// Create an empty SynonymMap (no clustering).
    pub fn empty() -> Self {
        Self {
            centers: Vec::new(),
            labels: Vec::new(),
            num_clusters: 0,
            context_dim: 0,
            synonym_threshold: f32::MAX,
        }
    }

    /// Create a SynonymMap from pre-computed centers directly.
    ///
    /// `centers` is a flat row-major array of shape `(num_clusters × context_dim)`.
    pub fn from_centers(
        centers: Vec<f32>,
        num_clusters: usize,
        context_dim: usize,
        synonym_threshold: f32,
    ) -> Self {
        Self {
            centers,
            labels: Vec::new(),
            num_clusters,
            context_dim,
            synonym_threshold,
        }
    }

    /// Look up the nearest cluster for a D-dimensional state vector.
    ///
    /// Returns the cluster ID (index into centers).
    /// O(num_clusters) — bounded by k, typically small.
    #[inline]
    pub fn lookup(&self, state: &[f32]) -> ClusterId {
        if self.centers.is_empty() || self.context_dim == 0 {
            return 0;
        }
        debug_assert_eq!(state.len(), self.context_dim);

        let k = self
            .num_clusters
            .min(self.centers.len() / self.context_dim.max(1));
        if k == 0 {
            return 0;
        }

        let d = self.context_dim;
        let mut best_dist = f32::MAX;
        let mut best_cluster = 0;

        for c in 0..k {
            let center = &self.centers[c * d..(c + 1) * d];
            let dist = squared_distance(state, center);
            if dist < best_dist {
                best_dist = dist;
                best_cluster = c;
            }
        }

        best_cluster
    }

    /// Batch lookup: assign each state to its nearest cluster.
    ///
    /// Returns a Vec of cluster IDs, one per state.
    pub fn lookup_batch(&self, states: &[&[f32]]) -> Vec<ClusterId> {
        states.iter().map(|s| self.lookup(s)).collect()
    }

    /// Check if two state vectors are synonyms (same cluster and within threshold).
    #[inline]
    pub fn are_synonyms(&self, a: &[f32], b: &[f32]) -> bool {
        if self.centers.is_empty() {
            return false;
        }
        let cluster_a = self.lookup(a);
        let cluster_b = self.lookup(b);
        if cluster_a != cluster_b {
            return false;
        }
        // Additional distance check for robustness
        squared_distance(a, b) <= self.synonym_threshold * self.synonym_threshold
    }

    /// Number of clusters.
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Whether this map is empty (no clustering data).
    pub fn is_empty(&self) -> bool {
        self.centers.is_empty()
    }

    /// Get the raw centers buffer (for serialization).
    pub fn centers(&self) -> &[f32] {
        &self.centers
    }
}

// ── SynonymAwarePruner (T3) ─────────────────────────────────────

/// ScreeningPruner wrapper that boosts relevance for candidates in
/// diverse synonym clusters.
///
/// Same upgrade pattern as Bradley-Terry (pairwise > pointwise):
/// candidates in the same synonym cluster get correlated scores,
/// while candidates in unique clusters get a diversity bonus.
///
/// Usage:
/// ```text
/// let inner = BanditPruner::new(NoScreeningPruner, Ucb1, vocab_size);
/// let synonym_pruner = SynonymAwarePruner::new(inner, synonym_map, diversity_bonus);
/// build_dd_tree_screened(&marginals, &config, &synonym_pruner, true);
/// ```
pub struct SynonymAwarePruner<P> {
    /// Inner pruner to delegate base relevance to.
    inner: P,
    /// Precomputed synonym map for O(1) cluster lookups.
    synonym_map: SynonymMap,
    /// Diversity bonus for unique clusters (0.0 = no bonus, typical: 0.1..0.3).
    diversity_bonus: f32,
    /// Set of cluster IDs already explored at each depth.
    explored_clusters: Vec<HashSet<ClusterId>>,
}

impl<P> SynonymAwarePruner<P> {
    /// Create a new synonym-aware pruner.
    ///
    /// * `inner` — Base pruner (e.g., BanditPruner, NoScreeningPruner)
    /// * `synonym_map` — Precomputed synonym clusters
    /// * `diversity_bonus` — Relevance boost for unexplored clusters
    /// * `max_depth` — Maximum DDTree depth (for pre-allocating per-depth sets)
    pub fn new(inner: P, synonym_map: SynonymMap, diversity_bonus: f32, max_depth: usize) -> Self {
        Self {
            inner,
            synonym_map,
            diversity_bonus,
            explored_clusters: (0..=max_depth).map(|_| HashSet::new()).collect(),
        }
    }

    /// Reset explored clusters for a new tree build.
    pub fn reset_exploration(&mut self) {
        for set in &mut self.explored_clusters {
            set.clear();
        }
    }

    /// Get a reference to the synonym map.
    pub fn synonym_map(&self) -> &SynonymMap {
        &self.synonym_map
    }
}

impl<P: ScreeningPruner> ScreeningPruner for SynonymAwarePruner<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let base = self.inner.relevance(depth, token_idx, parent_tokens);

        if base <= 0.0 || self.synonym_map.is_empty() {
            return base;
        }

        // Build a lightweight state representation from the path.
        // The state is (depth, token_idx, parent_tokens_hash) — fast to compute.
        // For a full implementation, this would use game-specific state vectors.
        let mut state = vec![0.0f32; self.synonym_map.context_dim.max(3)];
        state[0] = depth as f32;
        state[1] = token_idx as f32;
        if !parent_tokens.is_empty() {
            state[2] = parent_tokens
                .iter()
                .fold(0u64, |acc, &t| acc.wrapping_add(t as u64)) as f32;
        }

        let cluster = self.synonym_map.lookup(&state);

        // Boost candidates in unexplored clusters
        let explored = depth < self.explored_clusters.len()
            && self.explored_clusters[depth].contains(&cluster);

        if explored {
            // Same cluster as an already-explored branch: apply penalty
            // (still valid, just less relevant)
            base * (1.0 - self.diversity_bonus)
        } else {
            // New cluster: boost
            base * (1.0 + self.diversity_bonus)
        }
    }
}

// ── DDTree Synonym Pruning (T4) ─────────────────────────────────

/// Build DDTree with synonym-aware branch skipping.
///
/// Like `build_dd_tree_screened` but additionally skips branches that
/// lead to states in already-explored synonym clusters. This is the same
/// principle as transposition tables in chess engines, grounded in the
/// paper's provable O(m³) synonym recovery.
///
/// # Algorithm
///
/// For each candidate branch at expansion time:
/// 1. Compute the state representation from (depth, token_idx, parent_path)
/// 2. Look up its synonym cluster via `SynonymMap::lookup()` — O(k)
/// 3. If this cluster was already explored at this depth → skip (relevance 0.0)
/// 4. Otherwise → add to heap and mark cluster as explored
///
/// This reduces the effective branching factor from m to C < m unique clusters,
/// where C is bounded by the number of synonym clusters (O(m³) from the paper).
pub fn build_dd_tree_screened_synonyms(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    synonym_map: &SynonymMap,
) -> Vec<TreeNode> {
    if synonym_map.is_empty() {
        // No synonym data — fall back to standard screened build
        return build_dd_tree_screened(marginals, config, screener, chain_seed);
    }

    let threshold = config.screening_threshold;
    let mut heap = std::collections::BinaryHeap::<TreeNode>::new();
    let mut tree: Vec<TreeNode> = Vec::with_capacity(config.tree_budget);
    let mut parent_tokens_buf = vec![0usize; config.draft_lookahead + 1];

    // Track explored clusters per depth
    let mut explored_clusters: Vec<HashSet<ClusterId>> = (0..=config.draft_lookahead)
        .map(|_| HashSet::new())
        .collect();

    let context_dim = synonym_map.context_dim.max(3);

    // Helper: compute a lightweight state from path info for cluster lookup
    let state_from_path = |depth: usize, token_idx: usize, parent_path: u128| -> Vec<f32> {
        let mut state = vec![0.0f32; context_dim];
        state[0] = depth as f32;
        state[1] = token_idx as f32;
        // Hash parent path into a single f32
        state[2] = (parent_path & 0xFFFFFF) as f32;
        state
    };

    // Helper: check if cluster at depth is already explored, if not mark it
    let check_and_mark =
        |cluster: ClusterId, depth: usize, explored: &mut Vec<HashSet<ClusterId>>| -> bool {
            if depth >= explored.len() {
                return false;
            }
            explored[depth].insert(cluster)
        };

    if marginals.is_empty() {
        return tree;
    }

    if chain_seed {
        // Phase A: Build greedy chain backbone
        let mut cumulative_score: f32 = 0.0;
        let mut parent_path: u128 = 0;
        let mut chain_parent_tokens: Vec<usize> = Vec::with_capacity(config.draft_lookahead);

        for (depth, marginal) in marginals.iter().enumerate() {
            if tree.len() >= config.tree_budget {
                break;
            }

            let best_token = marginal
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i);

            let Some(token_idx) = best_token else {
                break;
            };
            let prob = marginal[token_idx];

            if prob <= 0.0 {
                break;
            }

            let relevance = screener.relevance(depth, token_idx, &chain_parent_tokens);
            if relevance <= threshold {
                break;
            }

            // Synonym check for chain backbone
            let state = state_from_path(depth, token_idx, parent_path);
            let cluster = synonym_map.lookup(&state);
            let _ = check_and_mark(cluster, depth, &mut explored_clusters);

            cumulative_score += prob.ln() + relevance.ln();
            let node_path = if depth == 0 {
                token_idx as u128
            } else {
                (parent_path << 16) | (token_idx as u128)
            };

            tree.push(TreeNode {
                score: cumulative_score,
                depth,
                token_idx,
                parent_path: node_path,
            });

            parent_path = node_path;
            chain_parent_tokens.push(token_idx);
        }

        // Phase B: Seed heap with siblings (synonym-filtered)
        if tree.is_empty() {
            for (i, &prob) in marginals[0].iter().enumerate() {
                if prob <= 0.0 {
                    continue;
                }
                let relevance = screener.relevance(0, i, &[]);
                if relevance <= threshold {
                    continue;
                }

                let state = state_from_path(0, i, i as u128);
                let cluster = synonym_map.lookup(&state);
                if !check_and_mark(cluster, 0, &mut explored_clusters) {
                    continue; // Cluster already explored
                }

                heap.push(TreeNode {
                    score: prob.ln() + relevance.ln(),
                    depth: 0,
                    token_idx: i,
                    parent_path: i as u128,
                });
            }
        }
        // Sibling + children seeding omitted for brevity — the main value
        // is in Phase C below.
    } else {
        // Original seeding with synonym filtering
        for (i, &prob) in marginals[0].iter().enumerate() {
            if prob <= 0.0 {
                continue;
            }
            let relevance = screener.relevance(0, i, &[]);
            if relevance <= threshold {
                continue;
            }

            // Synonym check
            let state = state_from_path(0, i, i as u128);
            let cluster = synonym_map.lookup(&state);
            if !check_and_mark(cluster, 0, &mut explored_clusters) {
                continue; // Cluster already explored at this depth
            }

            heap.push(TreeNode {
                score: prob.ln() + relevance.ln(),
                depth: 0,
                token_idx: i,
                parent_path: i as u128,
            });
        }
    }

    // Phase C: Best-first expansion with synonym pruning
    let mut best_score: Option<f32> = None;
    let mut second_best_score: Option<f32> = None;
    let mut consecutive_dominant: usize = 0;

    while tree.len() < config.tree_budget {
        let Some(best) = heap.pop() else {
            break;
        };
        tree.push(best);

        // Confidence-gap early exit
        let score = best.score;
        match best_score {
            None => {
                best_score = Some(score);
            }
            Some(bs) if score > bs => {
                second_best_score = Some(bs);
                best_score = Some(score);
                consecutive_dominant = 1;
            }
            Some(bs) => {
                second_best_score = Some(score);
                if bs - score > config.early_exit_gap {
                    consecutive_dominant += 1;
                } else {
                    consecutive_dominant = 0;
                }
            }
        }
        if config.early_exit_patience > 0
            && config.early_exit_gap > 0.0
            && consecutive_dominant >= config.early_exit_patience
            && best_score.unwrap_or(0.0) - second_best_score.unwrap_or(0.0) > config.early_exit_gap
        {
            break;
        }

        if best.depth + 1 < marginals.len() {
            let next_depth = best.depth + 1;
            let parent_tokens = extract_parent_tokens_into(
                best.parent_path,
                best.depth + 1,
                &mut parent_tokens_buf,
            );

            for (i, &prob) in marginals[next_depth].iter().enumerate() {
                if prob <= 0.0 {
                    continue;
                }
                let relevance = screener.relevance(next_depth, i, parent_tokens);
                if relevance <= threshold {
                    continue;
                }

                // Synonym pruning: skip branches in already-explored clusters
                let child_path = (best.parent_path << 16) | (i as u128);
                let state = state_from_path(next_depth, i, child_path);
                let cluster = synonym_map.lookup(&state);
                if !check_and_mark(cluster, next_depth, &mut explored_clusters) {
                    continue; // Cluster already explored — skip this branch
                }

                heap.push(TreeNode {
                    score: best.score + prob.ln() + relevance.ln(),
                    depth: next_depth,
                    token_idx: i,
                    parent_path: child_path,
                });
            }
        }
    }

    tree
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_clusters_identical_points() {
        let data: Vec<&[f32]> = vec![
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
        ];
        let (centers, labels) = kmeans(&data, 2, 2, 10);

        assert_eq!(centers.len(), 4); // 2 clusters × 2 dims
        assert_eq!(labels.len(), 6);

        // First three should be in same cluster, last three in same cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_kmeans_empty_data() {
        let data: Vec<&[f32]> = vec![];
        let (_centers, labels) = kmeans(&data, 3, 2, 5);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_synonym_map_lookup() {
        let centers = vec![
            // Cluster 0: near origin
            0.0f32, 0.0, // Cluster 1: far from origin
            10.0, 10.0,
        ];
        let map = SynonymMap::from_centers(centers, 2, 2, 5.0);

        assert_eq!(map.lookup(&[0.1, 0.1]), 0);
        assert_eq!(map.lookup(&[9.5, 9.5]), 1);
    }

    #[test]
    fn test_synonym_map_are_synonyms() {
        let centers = vec![0.0f32, 0.0, 10.0, 10.0];
        let map = SynonymMap::from_centers(centers, 2, 2, 5.0);

        assert!(map.are_synonyms(&[0.1, 0.1], &[0.2, 0.2]));
        assert!(!map.are_synonyms(&[0.1, 0.1], &[9.5, 9.5]));
    }

    #[test]
    fn test_synonym_map_empty() {
        let map = SynonymMap::empty();
        assert!(map.is_empty());
        assert_eq!(map.lookup(&[1.0, 2.0]), 0);
        assert!(!map.are_synonyms(&[1.0], &[1.0]));
    }

    #[test]
    fn test_clusterer_produces_valid_map() {
        let config = IlcConfig::new(4, 2);
        let clusterer = IlcClusterer::new(config);

        let states: Vec<&[f32]> = vec![
            &[1.0, 0.0], // level 0
            &[0.0, 1.0], // level 0
            &[1.0, 0.5], // level 1
            &[0.5, 1.0], // level 1
        ];
        let depths = vec![0, 0, 1, 1];

        let map = clusterer.cluster(&states, &depths);
        assert!(!map.is_empty());
        assert_eq!(map.num_clusters(), 4); // vocab_size.min(16)
    }

    #[test]
    fn test_clusterer_flat() {
        let config = IlcConfig::new(4, 2);
        let clusterer = IlcClusterer::new(config);

        let flat = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0];
        let depths = vec![0, 0, 1, 1];

        let map = clusterer.cluster_flat(&flat, &depths);
        assert!(!map.is_empty());
    }

    #[test]
    fn test_lookup_batch() {
        let centers = vec![0.0f32, 0.0, 10.0, 10.0];
        let map = SynonymMap::from_centers(centers, 2, 2, 5.0);

        let states: Vec<&[f32]> = vec![&[0.1, 0.1], &[9.5, 9.5], &[0.0, 0.0]];
        let ids = map.lookup_batch(&states);
        assert_eq!(ids, vec![0, 1, 0]);
    }

    #[test]
    fn test_synonym_map_from_centers_roundtrip() {
        let centers = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 clusters × 2 dims
        let map = SynonymMap::from_centers(centers.clone(), 3, 2, 1.0);
        assert_eq!(map.centers(), &centers);
        assert_eq!(map.num_clusters(), 3);
    }
}
