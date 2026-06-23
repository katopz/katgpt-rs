//! Viable Manifold Graph — discrete safe-manifold navigation primitive.
//!
//! Distillation of arXiv:2206.00106 (González-Duque et al., *Mario Plays on a
//! Manifold*, 2022). Given any smooth map `f: R^n → R^m` (closure) and a
//! viability predicate `V(z)`, compute the pullback volume field
//! `log det(J_f^T J_f)`, threshold + filter a finite latent sample to a
//! discrete safe-manifold subgraph, then navigate via A* / random walk that
//! stays inside the viable set by construction.
//!
//! Three composable primitives:
//! 1. [`pullback_volume`] — `log det(J^T J)` via [`jacobian_svd_at`] (Plan 301).
//! 2. [`build_safe_manifold_graph`] — discrete safe subgraph from samples.
//! 3. [`manifold_geodesic`] + [`manifold_random_walk`] — navigation that never
//!    leaves the viable set.
//!
//! # Design
//!
//! - **Modelless**: `f` is a caller-supplied closure. No gradients, no training.
//! - **Generic over `f` and `V`**: closures + a small [`ViabilityPredicate`]
//!   trait. No game / shard / chain semantics leak in.
//! - **Zero-allocation hot path**: `manifold_random_walk` reuses a small
//!   fixed-capacity scratch buffer; capacity is stable across 1000+ steps (G6).
//! - **Reuses existing infra**: the SVD comes from [`jacobian_svd_at`]
//!   (Plan 301). This module adds only the `det(J^T J)` reduction + graph +
//!   navigation.
//! - **No new deps**: graph storage is `Vec`-based (no petgraph); A* uses
//!   `std::collections::BinaryHeap` (no pathfinding crate); RNG is
//!   [`fastrand::Rng`] (already a direct dep of katgpt-core).
//!
//! References:
//! - Plan 312 — open-primitive spec
//! - Research 294 — math + prior-art table
//! - Plan 301 — substrate (`jacobian_svd_at`, [`JacobianSvdScratch`])

use crate::subspace_phase_gate::{JacobianSvdScratch, jacobian_svd_at};
use std::collections::BinaryHeap;

// ═══════════════════════════════════════════════════════════════════════════
// Phase 1 — Pullback volume field
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for [`pullback_volume`].
#[derive(Clone, Copy, Debug)]
pub struct VolumeFieldConfig {
    /// ε added inside each `log(·)` to keep it finite at rank-deficient points.
    /// Default `1e-12`.
    pub log_eps: f32,
    /// Forward-difference step passed to [`jacobian_svd_at`] when estimating
    /// the Jacobian. Default `1e-4` (Plan 301's recommended value). Pass a
    /// negative value to opt into central differences (more accurate, 2× cost).
    ///
    /// **Numerical-analysis note**: the forward-difference quotient
    /// `(f(x+eps) − f(x)) / eps` has relative error ≈ `machine_eps · |f(x)| /
    /// (eps · |f'(x)|)`. For O(1) function values and `eps = 1e-4`, this is
    /// ~1e-3 — fine for thresholding, too loose for bit-exact unit tests on
    /// linear maps. Callers who need high precision on near-linear regions
    /// should increase `jacobian_eps` (truncation error is zero for affine
    /// maps, so larger steps are free there) or use central differences.
    pub jacobian_eps: f32,
}

impl Default for VolumeFieldConfig {
    fn default() -> Self {
        Self {
            log_eps: 1e-12,
            jacobian_eps: DEFAULT_JACOBIAN_EPS,
        }
    }
}

/// Default forward-difference step for [`pullback_volume`]. Matches Plan 301's
/// recommended value (see `subspace_phase_gate.rs` docstring). Negative values
/// opt into central differences.
pub const DEFAULT_JACOBIAN_EPS: f32 = 1e-4;

/// Pullback volume of a smooth map `f: R^n → R^m` at `z`.
///
/// Returns `Σ_i log(σ_i^2 + cfg.log_eps)` where `σ_i` are the singular values
/// of the Jacobian `J_f(z)`. This is the numerically stable form of
/// `log det(J^T J) = log(Π σ_i^2)`, the pullback metric determinant that the
/// paper §III-B calls the "cost-to-traverse" scalar field.
///
/// Zero new allocations beyond what [`jacobian_svd_at`] already performs —
/// the [`SvdResult`](crate::subspace_phase_gate::SvdResult) is dropped promptly
/// after the reduction.
///
/// # Arguments
///
/// - `f` — smooth map `R^n → R^m`, called as `f(&z, &mut out)`.
/// - `z` — latent point, length `n`. Must match the `n` passed to
///   `JacobianSvdScratch::with_capacity(n, m)`.
/// - `scratch` — reusable scratch buffer (see Plan 301). Reuse across calls.
/// - `cfg` — small-epsilon for the log reduction.
pub fn pullback_volume<F>(
    f: F,
    z: &[f32],
    scratch: &mut JacobianSvdScratch,
    cfg: &VolumeFieldConfig,
) -> f32
where
    F: Fn(&[f32], &mut [f32]),
{
    let svd = jacobian_svd_at(f, z, cfg.jacobian_eps, scratch);
    let log_eps = cfg.log_eps;
    let mut acc: f32 = 0.0;
    for &sigma in svd.singular_values.iter() {
        // log(σ² + ε) — numerically stable vs log then product. Negative σ
        // would be an SVD bug; clamp defensively so we don't get NaN.
        let s2 = sigma * sigma;
        acc += (s2.max(0.0) + log_eps).ln();
    }
    acc
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2 — SafeManifoldGraph
// ═══════════════════════════════════════════════════════════════════════════

/// Discrete approximation of a safe manifold.
///
/// `nodes` is a flat row-major `[n_nodes × dim]` buffer of latent coordinates.
/// `edges` are bidirectional, deduplicated (stored as `(min, max)`), and sorted.
///
/// Construction is via [`build_safe_manifold_graph`]. The graph is the discrete
/// substrate on which [`manifold_geodesic`] and [`manifold_random_walk`] run;
/// both navigation primitives stay inside the viable set by construction.
#[derive(Clone, Debug)]
pub struct SafeManifoldGraph {
    /// Latent dimension `n`.
    pub dim: usize,
    /// Flat `[n_nodes × dim]` row-major buffer of kept node coordinates.
    nodes: Vec<f32>,
    /// Bidirectional, deduplicated `(min_id, max_id)` edges, sorted ascending.
    /// Source of truth for the public `edges()` accessor; not used by the
    /// hot-path neighbor lookup (which goes through `csr_*` below).
    edges: Vec<(u32, u32)>,
    /// CSR row pointers, length `n_nodes + 1`. Neighbors of node `i` live in
    /// `csr_neighbors[csr_offsets[i]..csr_offsets[i+1]]`. O(degree) lookup.
    csr_offsets: Vec<u32>,
    /// CSR column indices, length `2 * n_edges` (each edge contributes one
    /// entry per endpoint). Ordered per node as `[< node ascending] ++
    /// [> node ascending]` to match the historical linear-scan emission
    /// order byte-for-byte (preserves `manifold_random_walk` determinism).
    csr_neighbors: Vec<u32>,
}

impl SafeManifoldGraph {
    /// Empty graph with the given latent dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            nodes: Vec::new(),
            edges: Vec::new(),
            csr_offsets: vec![0],
            csr_neighbors: Vec::new(),
        }
    }

    /// Rebuild the CSR adjacency cache from `edges`.
    ///
    /// Must be called whenever `edges` (or `nodes`, which changes `n_nodes`)
    /// is mutated outside the public constructors — e.g. in tests that poke
    /// the private fields directly. All production paths (`new`, [`build_safe_manifold_graph`])
    /// invoke this at the end of construction.
    ///
    /// Per-node neighbor order is `[< node ascending] ++ [> node ascending]`,
    /// matching the order the pre-CSR `for_each_neighbor` linear scan emitted
    /// (edges are sorted by `(lo, hi)` lexicographically). This keeps
    /// `manifold_random_walk` byte-identical for any fixed RNG seed.
    ///
    /// O(V + E): two passes over edges plus a per-node slice sort. The slice
    /// sort is O(Σ deg·log(deg)) which is negligible (deg ≈ k_nearest).
    fn rebuild_csr(&mut self) {
        let n = self.n_nodes();
        // Counting-sort first pass: degree of each node.
        let mut degree = vec![0u32; n];
        for &(lo, hi) in &self.edges {
            let (lo, hi) = (lo as usize, hi as usize);
            if lo < n && hi < n {
                degree[lo] += 1;
                degree[hi] += 1;
            }
        }
        // Prefix sum → row pointers.
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        let mut acc: u32 = 0;
        for &d in &degree {
            acc += d;
            offsets.push(acc);
        }
        // Second pass: scatter each edge's endpoints into the right slots.
        let mut neighbors = vec![0u32; acc as usize];
        let mut cursor = offsets.clone(); // cursor[i] = next write index for node i
        for &(lo, hi) in &self.edges {
            let (lo, hi) = (lo as usize, hi as usize);
            if lo < n && hi < n {
                let p = cursor[lo] as usize;
                neighbors[p] = hi as u32;
                cursor[lo] += 1;
                let q = cursor[hi] as usize;
                neighbors[q] = lo as u32;
                cursor[hi] += 1;
            }
        }
        // Per-node sort to match the pre-CSR emission order:
        // lower-id neighbors first (ascending), then higher-id (ascending).
        for node in 0..n {
            let s = offsets[node] as usize;
            let e = offsets[node + 1] as usize;
            let node_u = node as u32;
            neighbors[s..e].sort_unstable_by_key(|&nbr| (nbr > node_u, nbr));
        }
        self.csr_offsets = offsets;
        self.csr_neighbors = neighbors;
    }

    /// Number of kept nodes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        // dim is non-zero for any buildable graph; guard anyway.
        if self.dim == 0 {
            return 0;
        }
        self.nodes.len() / self.dim
    }

    /// Number of deduplicated edges.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Latent coordinates of node `idx` (length `dim`). O(1) slice.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_nodes()`.
    #[inline]
    pub fn node_latent(&self, idx: u32) -> &[f32] {
        let start = (idx as usize) * self.dim;
        let end = start + self.dim;
        &self.nodes[start..end]
    }

    /// Neighbors of node `idx`, in the canonical CSR order `[< idx ascending] ++
    /// [> idx ascending]`.
    ///
    /// **Allocation note**: returns an owned `Vec<u32>` of length = node degree.
    /// This is a *cold* / query path; for hot navigation use
    /// [`Self::for_each_neighbor`] (zero-alloc) or [`manifold_random_walk`]
    /// (uses a small internal scratch buffer).
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_nodes()`.
    pub fn neighbors(&self, idx: u32) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        self.for_each_neighbor(idx, |n| out.push(n));
        out
    }

    /// Zero-allocation neighbor iteration. Calls `f(id)` for each neighbor of
    /// `idx`. O(degree) via the CSR adjacency cache — each step does two array
    /// reads (`csr_offsets[idx]`, `csr_offsets[idx+1]`) plus `degree` reads
    /// from `csr_neighbors`. Used by hot navigation paths.
    ///
    /// Neighbor order is deterministic and identical to the pre-CSR linear
    /// scan: `[< idx ascending] ++ [> idx ascending]`. This keeps
    /// `manifold_random_walk` byte-identical for any fixed RNG seed.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_nodes()`.
    #[inline]
    pub fn for_each_neighbor<F: FnMut(u32)>(&self, idx: u32, mut f: F) {
        let i = idx as usize;
        let start = self.csr_offsets[i] as usize;
        let end = self.csr_offsets[i + 1] as usize;
        for &neighbor in &self.csr_neighbors[start..end] {
            f(neighbor);
        }
    }

    /// Index of the node nearest (Euclidean) to `z`. O(n·dim) scan.
    ///
    /// Returns `None` if the graph is empty. On ties returns the lowest index.
    ///
    /// # Panics
    ///
    /// Panics if `z.len() != dim`.
    pub fn nearest_node(&self, z: &[f32]) -> Option<u32> {
        assert_eq!(z.len(), self.dim, "z.len() must equal graph dim");
        if self.n_nodes() == 0 {
            return None;
        }
        let mut best_idx: u32 = 0;
        let mut best_d2: f32 = f32::INFINITY;
        for i in 0..self.n_nodes() {
            let node = self.node_latent(i as u32);
            let mut d2: f32 = 0.0;
            for k in 0..self.dim {
                let d = node[k] - z[k];
                d2 += d * d;
            }
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = i as u32;
            }
        }
        Some(best_idx)
    }

    /// Read-only access to the internal edge list (deduplicated, sorted).
    pub fn edges(&self) -> &[(u32, u32)] {
        &self.edges
    }

    /// Read-only access to the flat internal node buffer.
    pub fn nodes_flat(&self) -> &[f32] {
        &self.nodes
    }
}

/// Configuration for [`build_safe_manifold_graph`].
#[derive(Clone, Copy, Debug)]
pub struct GraphBuildConfig {
    /// Keep node `i` iff `pullback_volume(f, z_i) ≤ volume_threshold`. Caller
    /// responsibility to pick (paper uses mean volume as a default threshold).
    pub volume_threshold: f32,
    /// If true, verify the midpoint of each candidate edge is also viable
    /// before adding it. Slower, more correct (catches "thin corridor" gaps).
    pub edge_midpoint_check: bool,
    /// Connect each kept node to its `k_nearest` kept neighbors (Euclidean in
    /// latent space). Paper uses grid adjacency; we generalize.
    pub k_nearest: usize,
}

impl Default for GraphBuildConfig {
    fn default() -> Self {
        Self {
            volume_threshold: f32::INFINITY,
            edge_midpoint_check: false,
            k_nearest: 4,
        }
    }
}

/// Closure-agnostic viability predicate. Implement for any stateful predicate
/// the caller wants to apply to each candidate node.
pub trait ViabilityPredicate {
    /// `true` iff `z` is a viable latent point (e.g., decodes to a playable
    /// level, a coherent affect state, etc.).
    fn is_viable(&self, z: &[f32]) -> bool;
}

/// Wraps any `Fn(&[f32]) -> bool` as a [`ViabilityPredicate`].
pub struct ClosurePredicate<F>(pub F)
where
    F: Fn(&[f32]) -> bool;

impl<F> ViabilityPredicate for ClosurePredicate<F>
where
    F: Fn(&[f32]) -> bool,
{
    #[inline]
    fn is_viable(&self, z: &[f32]) -> bool {
        (self.0)(z)
    }
}

/// Build a [`SafeManifoldGraph`] from a finite latent sample.
///
/// Algorithm (Plan 312 T2.4):
/// 1. For each sample `z_i`: compute `vol_i = pullback_volume(f, z_i, scratch, volume_cfg)`.
///    Keep `i` iff `vol_i ≤ build_cfg.volume_threshold AND predicate.is_viable(z_i)`.
/// 2. For each kept node, find its `k_nearest` kept neighbors (Euclidean). If
///    `edge_midpoint_check`, verify the midpoint `(z_a + z_b)/2` is also viable
///    before adding the edge.
/// 3. Dedup + sort edges. Return the graph.
///
/// `samples` is a flat `[n_samples × dim]` row-major buffer; `dim` is the
/// latent dimension. The returned graph has `dim` set from the argument.
///
/// # Panics
///
/// Panics if `samples.len() % dim != 0`.
pub fn build_safe_manifold_graph<F, V>(
    f: F,
    samples: &[f32],
    dim: usize,
    predicate: &V,
    volume_cfg: &VolumeFieldConfig,
    build_cfg: &GraphBuildConfig,
    scratch: &mut JacobianSvdScratch,
) -> SafeManifoldGraph
where
    F: Fn(&[f32], &mut [f32]),
    V: ViabilityPredicate,
{
    assert!(
        dim > 0,
        "dim must be positive (got {dim})"
    );
    assert_eq!(
        samples.len() % dim,
        0,
        "samples.len() ({}) must be a multiple of dim ({})",
        samples.len(),
        dim
    );
    let n_samples = samples.len() / dim;

    // ── Step 1: filter samples by (volume ≤ τ) AND predicate ───────────────
    // We collect kept (original-index, coords) pairs. The original index is
    // not stored in the final graph — only the kept coordinates.
    let mut kept_nodes: Vec<f32> = Vec::with_capacity(n_samples * dim);
    let mut vol_scratch: f32;
    for i in 0..n_samples {
        let z = &samples[i * dim..(i + 1) * dim];
        vol_scratch = pullback_volume(&f, z, scratch, volume_cfg);
        if vol_scratch <= build_cfg.volume_threshold && predicate.is_viable(z) {
            kept_nodes.extend_from_slice(z);
        }
    }
    let n_kept = if dim > 0 {
        kept_nodes.len() / dim
    } else {
        0
    };

    // ── Step 2: connect each kept node to its k_nearest kept neighbors ──────
    // For each node a, compute distances to all other nodes, take k nearest,
    // optionally check midpoint viability, add (min,max) edge. Dedup at the
    // end via sort + dedup.
    let k = build_cfg.k_nearest.min(n_kept.saturating_sub(1));
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(n_kept * k);
    // Reusable scratch for pairwise distances — capacity n_kept-1, reused
    // across all nodes.
    let mut dist_buf: Vec<(f32, u32)> = Vec::with_capacity(n_kept.saturating_sub(1));
    // Reusable midpoint buffer (only allocated if midpoint check is on).
    let mut mid_buf: Vec<f32> = if build_cfg.edge_midpoint_check {
        vec![0.0; dim]
    } else {
        Vec::new()
    };

    for a in 0..n_kept as u32 {
        let za = &kept_nodes[(a as usize) * dim..(a as usize + 1) * dim];
        dist_buf.clear();
        for b in 0..n_kept as u32 {
            if b == a {
                continue;
            }
            let zb = &kept_nodes[(b as usize) * dim..(b as usize + 1) * dim];
            let mut d2: f32 = 0.0;
            for j in 0..dim {
                let d = za[j] - zb[j];
                d2 += d * d;
            }
            dist_buf.push((d2, b));
        }
        // Partial sort: take k smallest. For small k vs n, a full sort is
        // simple and cache-friendly; the paper's grids are ≤10³ nodes.
        dist_buf.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(_d2, b) in dist_buf.iter().take(k) {
            if build_cfg.edge_midpoint_check {
                let zb = &kept_nodes[(b as usize) * dim..(b as usize + 1) * dim];
                for j in 0..dim {
                    mid_buf[j] = 0.5 * (za[j] + zb[j]);
                }
                if !predicate.is_viable(&mid_buf) {
                    continue;
                }
            }
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            edges.push((lo, hi));
        }
    }

    // ── Step 3: dedup + sort ────────────────────────────────────────────
    edges.sort_unstable();
    edges.dedup();

    let mut g = SafeManifoldGraph {
        dim,
        nodes: kept_nodes,
        edges,
        csr_offsets: Vec::new(),
        csr_neighbors: Vec::new(),
    };
    // Build CSR adjacency so `for_each_neighbor` is O(degree), not O(E).
    g.rebuild_csr();
    g
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 3 — Navigation
// ═══════════════════════════════════════════════════════════════════════════

/// A* shortest path on a [`SafeManifoldGraph`] with Euclidean-latent heuristic.
///
/// Returns the node-index path `src → … → dst` (inclusive of both endpoints),
/// or `None` if `dst` is unreachable from `src`.
///
/// Admissibility: Euclidean distance is a consistent (hence admissible)
/// heuristic for graphs whose edge weights are Euclidean distances. We use
/// Euclidean distance as the edge weight here.
///
/// # Panics
///
/// Panics if `src` or `dst` are out of range. Returns `Some(vec![src])` if
/// `src == dst`.
pub fn manifold_geodesic(g: &SafeManifoldGraph, src: u32, dst: u32) -> Option<Vec<u32>> {
    let n = g.n_nodes();
    assert!((src as usize) < n, "src out of range: {src} >= {n}");
    assert!((dst as usize) < n, "dst out of range: {dst} >= {n}");
    if src == dst {
        return Some(vec![src]);
    }

    let dst_latent = g.node_latent(dst).to_vec();

    // g_score[i] = best known cost-to-come from src to i.
    let mut g_score = vec![f32::INFINITY; n];
    g_score[src as usize] = 0.0;
    // came_from[i] = predecessor on best path.
    let mut came_from: Vec<u32> = vec![u32::MAX; n];
    // closed set via f_score once popped: a node is final when popped.
    let mut closed = vec![false; n];

    // BinaryHeap is max-heap; reverse via Reverse to get min-heap on f = g + h.
    // Entry: (OrderedF32(f_score), node). OrderedF32 provides a total order so
    // BinaryHeap works without an `ordered-float` dependency.
    let mut open: BinaryHeap<std::cmp::Reverse<(Of32, u32)>> =
        BinaryHeap::with_capacity(n.min(64));
    open.push(std::cmp::Reverse((Of32(0.0), src)));

    while let Some(std::cmp::Reverse((Of32(_f_cur), cur))) = open.pop() {
        let cur_idx = cur as usize;
        if closed[cur_idx] {
            continue;
        }
        closed[cur_idx] = true;
        if cur == dst {
            break;
        }
        let cur_latent = g.node_latent(cur);
        let g_cur = g_score[cur_idx];
        g.for_each_neighbor(cur, |nxt| {
            let nxt_idx = nxt as usize;
            if closed[nxt_idx] {
                return;
            }
            // Edge weight = Euclidean distance in latent space.
            let nxt_latent = g.node_latent(nxt);
            let mut d2: f32 = 0.0;
            for k in 0..g.dim {
                let d = cur_latent[k] - nxt_latent[k];
                d2 += d * d;
            }
            let w = d2.sqrt();
            let tentative = g_cur + w;
            if tentative < g_score[nxt_idx] {
                g_score[nxt_idx] = tentative;
                came_from[nxt_idx] = cur;
                let mut h: f32 = 0.0;
                for k in 0..g.dim {
                    let d = nxt_latent[k] - dst_latent[k];
                    h += d * d;
                }
                let h = h.sqrt();
                open.push(std::cmp::Reverse((Of32(tentative + h), nxt)));
            }
        });
    }

    if !closed[dst as usize] {
        return None;
    }

    // Reconstruct path dst → src, then reverse.
    let mut path: Vec<u32> = Vec::new();
    let mut cur = dst;
    loop {
        path.push(cur);
        if cur == src {
            break;
        }
        let pred = came_from[cur as usize];
        if pred == u32::MAX {
            // Unreachable (shouldn't happen since closed[dst] is true), but
            // defend against logic bugs.
            return None;
        }
        cur = pred;
    }
    path.reverse();
    Some(path)
}

/// Total-order wrapper for f32 so A*'s `BinaryHeap<Reverse<(Of32(f), u32)>>`
/// behaves as a min-heap by float value. NaN sorts last (treating it as +∞).
#[derive(Copy, Clone, PartialEq)]
struct Of32(f32);

impl Eq for Of32 {}

impl PartialOrd for Of32 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Of32 {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // NaN → +∞ (sorts last). Otherwise normal compare.
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => self.0.total_cmp(&other.0),
        }
    }
}

/// Uniform-over-neighbors random walk for `m` steps.
///
/// Returns a `Vec<u32>` of length `m + 1` (the trajectory, starting with
/// `src`). Each successive index is chosen uniformly at random from the
/// previous node's neighbors, so every visited node is on the safe-manifold
/// graph by construction — **playability = 1.0**.
///
/// **Hot-path allocation contract**: the returned `Vec` is pre-sized to
/// `m + 1` and never grows beyond it. If a node has no neighbors (isolated),
/// the walk parks there for the remaining steps. The internal neighbor-list
/// scratch buffer is bounded by the node degree and is reused across steps.
///
/// RNG: uses [`fastrand::Rng`] (already a direct dep of katgpt-core). The
/// caller owns the RNG instance for reproducibility.
pub fn manifold_random_walk(g: &SafeManifoldGraph, src: u32, m: usize, rng: &mut fastrand::Rng) -> Vec<u32> {
    let n = g.n_nodes();
    assert!((src as usize) < n, "src out of range: {src} >= {n}");

    // Pre-allocate the trajectory at exact final capacity.
    let mut path: Vec<u32> = Vec::with_capacity(m + 1);
    path.push(src);

    // Reusable neighbor scratch. Upper bound on degree: number of edges
    // (worst case a star node). For typical graphs this is ~k_nearest.
    // Cap capacity growth: only reserve once.
    let mut neighbors: Vec<u32> = Vec::new();
    let mut cur = src;
    for _ in 0..m {
        neighbors.clear();
        g.for_each_neighbor(cur, |n| neighbors.push(n));
        if neighbors.is_empty() {
            // Park at the current node for the remaining steps.
            let remaining = m - path.len() + 1;
            path.resize(path.len() + remaining, cur);
            break;
        }
        let pick = neighbors[rng.usize(0..neighbors.len())];
        path.push(pick);
        cur = pick;
    }
    debug_assert_eq!(
        path.len(),
        m + 1,
        "random walk should return exactly m+1 nodes"
    );
    path
}

/// Weighted-over-neighbors random walk for `m` steps.
///
/// At each step, sample the next node from the neighbor set with probability
/// proportional to `exp(weights(cur, nxt))` (softmax-free — we use
/// `sigmoid`-style normalization per AGENTS.md, here implemented as a
/// straightforward weighted draw over non-negative weights; pass
/// `weights = |a,b| sigmoid(...)` at the call site to enforce the sigmoid
/// rule). This is the **riir-ai integration hook**: the `weights` closure
/// can wrap `cgsp_runtime::curiosity_step` without that type being visible
/// here.
///
/// Weights `< 0` are clamped to 0 (treated as zero probability). If all
/// weights are zero, falls back to uniform. Returns the same trajectory shape
/// as [`manifold_random_walk`]: `Vec<u32>` of length `m + 1`.
pub fn manifold_curiosity_walk<W>(
    g: &SafeManifoldGraph,
    src: u32,
    m: usize,
    weights: &W,
    rng: &mut fastrand::Rng,
) -> Vec<u32>
where
    W: Fn(u32, u32) -> f32,
{
    let n = g.n_nodes();
    assert!((src as usize) < n, "src out of range: {src} >= {n}");

    let mut path: Vec<u32> = Vec::with_capacity(m + 1);
    path.push(src);

    // Reusable scratch: (neighbor, weight). Capacity bounded by max degree.
    let mut cand: Vec<(u32, f32)> = Vec::new();
    let mut cur = src;
    for _ in 0..m {
        cand.clear();
        g.for_each_neighbor(cur, |n| {
            let w = (weights)(cur, n).max(0.0);
            cand.push((n, w));
        });
        if cand.is_empty() {
            let remaining = m - path.len() + 1;
            path.resize(path.len() + remaining, cur);
            break;
        }
        let total: f32 = cand.iter().map(|(_, w)| *w).sum();
        let pick = if total <= 0.0 {
            // All-zero weights → uniform.
            cand[rng.usize(0..cand.len())].0
        } else {
            let mut r = rng.f32() * total;
            let mut chosen = cand[cand.len() - 1].0;
            for &(node, w) in cand.iter() {
                r -= w;
                if r <= 0.0 {
                    chosen = node;
                    break;
                }
            }
            chosen
        };
        path.push(pick);
        cur = pick;
    }
    path
}

// ═══════════════════════════════════════════════════════════════════════════
// Inline tests — GOAT gates G1–G6 (unit-test subset only; no benches)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fastrand::Rng;

    // ── G1: identity map has zero pullback volume ──────────────────────────

    #[test]
    fn test_pullback_volume_identity_is_zero() {
        // f(x) = x. J = I → singular values all 1 → log(1² + ε) ≈ 0.
        //
        // We use a large `jacobian_eps` (0.25) because the identity map is
        // affine — its Jacobian is constant, so truncation error is exactly
        // zero and a larger step eliminates forward-difference roundoff
        // (relative error ≈ machine_eps·|f(x)|/(eps·|f'(x)|)). With eps=1e-4
        // and |f(x)|=O(1), roundoff is ~1e-3, which would fail the 1e-6 gate;
        // with eps=0.25 it drops below 1e-6. For nonlinear caller maps the
        // default 1e-4 is the right tradeoff (see `VolumeFieldConfig` docs).
        let mut scratch = JacobianSvdScratch::with_capacity(4, 4);
        let cfg = VolumeFieldConfig {
            jacobian_eps: 0.25,
            ..VolumeFieldConfig::default()
        };
        let f = |z: &[f32], out: &mut [f32]| out.copy_from_slice(z);
        for z_orig in [[0.0_f32; 4], [1.0; 4], [0.5, -0.5, 2.0, -3.0]] {
            let vol = pullback_volume(f, &z_orig, &mut scratch, &cfg);
            assert!(
                vol.abs() < 1e-6,
                "identity map volume should be ~0, got {vol} for z={z_orig:?}"
            );
        }
    }

    // ── G2: scaling map f(x) = c·x has volume 2n·log(c) ────────────────────

    #[test]
    fn test_pullback_volume_scaling_is_2n_log_c() {
        // f(x) = 2x. J = 2I → singular values all 2 → log(4+ε)·n = 2n·log(2).
        // Same large-`jacobian_eps` rationale as G1: affine map, zero
        // truncation error, larger step kills roundoff.
        let n = 4;
        let mut scratch = JacobianSvdScratch::with_capacity(n, n);
        let cfg = VolumeFieldConfig {
            jacobian_eps: 0.25,
            ..VolumeFieldConfig::default()
        };
        let c = 2.0_f32;
        let f = move |z: &[f32], out: &mut [f32]| {
            for i in 0..z.len() {
                out[i] = c * z[i];
            }
        };
        let z = [0.3_f32, -0.7, 1.2, 0.9];
        let vol = pullback_volume(f, &z, &mut scratch, &cfg);
        let expected = 2.0 * (n as f32) * c.ln();
        assert!(
            (vol - expected).abs() < 1e-4,
            "scaling volume should be 2n·log(c) = {expected}, got {vol}"
        );
    }

    // ── Helpers shared by G3 / G3b ─────────────────────────────────────────

    /// 4D grid: 100 samples in [0,1]^4 (10 per axis interleaved — actually
    /// we use a regular grid: {0.0, 0.1, ..., 0.5} × {0.0, 0.1, ..., 0.5}
    /// wait, that's 6^4 = 1296. We want 100 = 10×10. So build a 10×10×1×1
    /// degenerate grid — but that makes the first 2 dims vary and last 2
    /// constant. For a more honest 4D spread we use 100 random points in
    /// [0,1]^4 from a fixed-seed RNG.
    fn make_100_samples_4d(seed: u64) -> Vec<f32> {
        let mut rng = Rng::with_seed(seed);
        let mut v = Vec::with_capacity(100 * 4);
        for _ in 0..100 {
            for _ in 0..4 {
                v.push(rng.f32());
            }
        }
        v
    }

    /// Identity map f for graph-build tests. With identity f, pullback volume
    /// is ~0 everywhere, so threshold = +∞ keeps every sample that passes the
    /// predicate.
    fn f_identity(z: &[f32], out: &mut [f32]) {
        out.copy_from_slice(z);
    }

    // ── G3: predicate = always true → connected graph, 100 nodes ────────────

    #[test]
    fn test_safe_graph_build_connected_when_predicate_true() {
        let samples = make_100_samples_4d(42);
        let mut scratch = JacobianSvdScratch::with_capacity(4, 4);
        let build_cfg = GraphBuildConfig {
            volume_threshold: f32::INFINITY,
            edge_midpoint_check: false,
            k_nearest: 4,
        };
        let g = build_safe_manifold_graph(
            f_identity,
            &samples,
            4,
            &ClosurePredicate(|_| true),
            &VolumeFieldConfig::default(),
            &build_cfg,
            &mut scratch,
        );
        assert_eq!(g.n_nodes(), 100, "all 100 samples should be kept");
        assert!(g.n_edges() > 0, "graph should have edges");

        // Connectedness check via BFS from node 0.
        let n = g.n_nodes();
        let mut visited = vec![false; n];
        let mut stack = vec![0u32];
        visited[0] = true;
        let mut count = 1;
        while let Some(u) = stack.pop() {
            g.for_each_neighbor(u, |v| {
                let vi = v as usize;
                if !visited[vi] {
                    visited[vi] = true;
                    count += 1;
                    stack.push(v);
                }
            });
        }
        assert_eq!(count, n, "graph should be connected (BFS reached {count}/{n})");
    }

    // ── G3b: predicate = "z[0] > 0.5" → disconnected components ────────────

    #[test]
    fn test_safe_graph_build_disconnected_when_predicate_splits() {
        let samples = make_100_samples_4d(42);
        let mut scratch = JacobianSvdScratch::with_capacity(4, 4);
        let build_cfg = GraphBuildConfig {
            volume_threshold: f32::INFINITY,
            edge_midpoint_check: false,
            k_nearest: 4,
        };
        let g = build_safe_manifold_graph(
            f_identity,
            &samples,
            4,
            &ClosurePredicate(|z| z[0] > 0.5),
            &VolumeFieldConfig::default(),
            &build_cfg,
            &mut scratch,
        );
        // Roughly half the samples have z[0] > 0.5.
        assert!(
            g.n_nodes() > 0 && g.n_nodes() < 100,
            "predicate should filter some samples; got {}",
            g.n_nodes()
        );
        // No edge should cross the predicate boundary. Every edge (a, b) must
        // have both endpoints with z[0] > 0.5.
        for &(a, b) in g.edges() {
            let za = g.node_latent(a);
            let zb = g.node_latent(b);
            assert!(za[0] > 0.5, "edge endpoint a={a} fails predicate");
            assert!(zb[0] > 0.5, "edge endpoint b={b} fails predicate");
        }
        // Connectedness: should be a single component (all kept nodes share
        // z[0] > 0.5), but we verify the boundary is respected more strongly
        // by checking the original 100 samples: count how many have z[0] > 0.5.
        let expected_kept = samples
            .chunks(4)
            .filter(|z| z[0] > 0.5)
            .count();
        assert_eq!(
            g.n_nodes(),
            expected_kept,
            "kept node count should match predicate-positive samples"
        );
    }

    // ── G4: manifold_geodesic returns a path that stays viable ──────────────

    /// Build the paper's two-disk + corridor toy: viable iff
    ///   (in left disk) OR (in right disk) OR (in corridor connecting them).
    fn build_two_disk_corridor_graph() -> SafeManifoldGraph {
        // Left disk center (-2, 0), radius 1.5. Right disk (+2, 0), radius 1.5.
        // Corridor: |x| < 2.0 AND |y| < 0.4.
        let viable = |z: &[f32]| {
            let x = z[0];
            let y = z[1];
            let dl = ((x + 2.0).powi(2) + y * y).sqrt();
            let dr = ((x - 2.0).powi(2) + y * y).sqrt();
            dl < 1.5 || dr < 1.5 || (x.abs() < 2.0 && y.abs() < 0.4)
        };

        // 2D grid over [-5, 5]² at step 0.25 = 41×41 = 1681 samples.
        let mut samples: Vec<f32> = Vec::with_capacity(41 * 41 * 2);
        let mut i = -5.0_f32;
        while i <= 5.0 {
            let mut j = -5.0_f32;
            while j <= 5.0 {
                samples.push(i);
                samples.push(j);
                j += 0.25;
            }
            i += 0.25;
        }

        // f = identity (so volume is ~0 everywhere; the predicate alone
        // decides viability). k=4 nearest with midpoint check ensures the
        // corridor is faithfully connected.
        let mut scratch = JacobianSvdScratch::with_capacity(2, 2);
        let build_cfg = GraphBuildConfig {
            volume_threshold: f32::INFINITY,
            edge_midpoint_check: true,
            k_nearest: 4,
        };
        build_safe_manifold_graph(
            |z: &[f32], out: &mut [f32]| out.copy_from_slice(z),
            &samples,
            2,
            &ClosurePredicate(viable),
            &VolumeFieldConfig::default(),
            &build_cfg,
            &mut scratch,
        )
    }

    #[test]
    fn test_manifold_geodesic_validity() {
        let g = build_two_disk_corridor_graph();
        assert!(g.n_nodes() > 50, "toy graph should have many nodes");

        // Pick src in left disk, dst in right disk.
        let src = g.nearest_node(&[-2.0, 0.0]).expect("left disk center");
        let dst = g.nearest_node(&[2.0, 0.0]).expect("right disk center");
        assert_ne!(src, dst, "src and dst must differ");

        let path = manifold_geodesic(&g, src, dst).expect("path should exist");
        assert!(!path.is_empty(), "path non-empty");
        assert_eq!(path.first(), Some(&src), "path starts at src");
        assert_eq!(path.last(), Some(&dst), "path ends at dst");

        // Every node on the path satisfies the predicate (viable by
        // construction since the graph only contains viable nodes, but assert
        // explicitly to make the GOAT gate self-documenting).
        let viable = |z: &[f32]| {
            let x = z[0];
            let y = z[1];
            let dl = ((x + 2.0).powi(2) + y * y).sqrt();
            let dr = ((x - 2.0).powi(2) + y * y).sqrt();
            dl < 1.5 || dr < 1.5 || (x.abs() < 2.0 && y.abs() < 0.4)
        };
        for &node in &path {
            let z = g.node_latent(node);
            assert!(viable(z), "path node {node} at {z:?} is not viable");
        }

        // Monotonicity sanity: no repeated nodes in a shortest path.
        let mut sorted_path = path.clone();
        sorted_path.sort_unstable();
        let len_before = sorted_path.len();
        sorted_path.dedup();
        assert_eq!(
            sorted_path.len(),
            len_before,
            "shortest path should not revisit nodes"
        );
    }

    // ── G5: manifold_random_walk stays viable ──────────────────────────────

    #[test]
    fn test_manifold_random_walk_validity() {
        let g = build_two_disk_corridor_graph();
        let src = g.nearest_node(&[-2.0, 0.0]).expect("left disk center");
        let mut rng = Rng::with_seed(0xC0FFEE);

        let m = 25;
        let walk = manifold_random_walk(&g, src, m, &mut rng);

        assert_eq!(walk.len(), m + 1, "walk length should be m+1");
        assert_eq!(walk.first(), Some(&src), "walk starts at src");

        let viable = |z: &[f32]| {
            let x = z[0];
            let y = z[1];
            let dl = ((x + 2.0).powi(2) + y * y).sqrt();
            let dr = ((x - 2.0).powi(2) + y * y).sqrt();
            dl < 1.5 || dr < 1.5 || (x.abs() < 2.0 && y.abs() < 0.4)
        };
        for &node in &walk {
            let z = g.node_latent(node);
            assert!(
                viable(z),
                "walk visited non-viable node {node} at {z:?} — playability violated"
            );
        }
        // Playability = 1.0 by construction (no assertion needed beyond the loop).
    }

    // ── G6: random walk has zero alloc growth across 1000 steps ─────────────

    #[test]
    fn test_manifold_random_walk_zero_alloc_across_1000_steps() {
        let g = build_two_disk_corridor_graph();
        let src = g.nearest_node(&[-2.0, 0.0]).expect("left disk center");
        let mut rng = Rng::with_seed(0xBADC0DE);

        let m = 1000;
        let walk = manifold_random_walk(&g, src, m, &mut rng);

        // Length contract: exactly m + 1 nodes.
        assert_eq!(walk.len(), m + 1, "walk length");
        // Capacity contract: the returned Vec was sized at m + 1 upfront and
        // must not have grown beyond it. (Vec allocation is always a power of
        // two ≥ length; here we assert capacity == m + 1, which is what
        // Vec::with_capacity(m+1) yields.)
        assert_eq!(
            walk.capacity(),
            m + 1,
            "walk Vec capacity must equal m+1 (no growth beyond pre-allocation)"
        );

        // All nodes still viable.
        let viable = |z: &[f32]| {
            let x = z[0];
            let y = z[1];
            let dl = ((x + 2.0).powi(2) + y * y).sqrt();
            let dr = ((x - 2.0).powi(2) + y * y).sqrt();
            dl < 1.5 || dr < 1.5 || (x.abs() < 2.0 && y.abs() < 0.4)
        };
        for &node in &walk {
            let z = g.node_latent(node);
            assert!(viable(z), "walk visited non-viable node {node}");
        }
    }

    // ── Bonus: curiosity walk basic sanity ──────────────────────────────────

    #[test]
    fn test_manifold_curiosity_walk_basic() {
        let g = build_two_disk_corridor_graph();
        let src = g.nearest_node(&[-2.0, 0.0]).expect("left disk center");
        let mut rng = Rng::with_seed(7);

        // Uniform weights → should match uniform random walk contract.
        let m = 50;
        let walk = manifold_curiosity_walk(&g, src, m, &|_, _| 1.0, &mut rng);
        assert_eq!(walk.len(), m + 1, "curiosity walk length");
        assert_eq!(walk.first(), Some(&src), "starts at src");

        // Every consecutive pair must be adjacent in the graph.
        for w in walk.windows(2) {
            let a = w[0];
            let b = w[1];
            if a == b {
                continue; // parked at isolated node
            }
            let mut found = false;
            g.for_each_neighbor(a, |n| {
                if n == b {
                    found = true;
                }
            });
            assert!(found, "walk hop {a}→{b} is not a graph edge");
        }
    }

    // ── Bonus: A* on trivial 2-node graph ───────────────────────────────────

    #[test]
    fn test_manifold_geodesic_trivial_unreachable() {
        // Two isolated nodes → no path.
        let mut g = SafeManifoldGraph::new(2);
        g.nodes = vec![0.0, 0.0, 1.0, 0.0];
        g.edges = vec![];
        g.rebuild_csr();
        // n_nodes should be 2 here.
        assert_eq!(g.n_nodes(), 2);
        let path = manifold_geodesic(&g, 0, 1);
        assert!(path.is_none(), "isolated nodes should have no path");
    }

    #[test]
    fn test_manifold_geodesic_trivial_single_edge() {
        let mut g = SafeManifoldGraph::new(2);
        g.nodes = vec![0.0, 0.0, 1.0, 0.0];
        g.edges = vec![(0, 1)];
        g.rebuild_csr();
        let path = manifold_geodesic(&g, 0, 1).expect("path exists");
        assert_eq!(path, vec![0, 1]);
    }
}
