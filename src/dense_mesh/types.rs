//! Types for DenseMesh — latent node network.

// Uses std (not alloc) — katgpt-rs is a std binary crate.

// ---------------------------------------------------------------------------
// DenseHidden — latent channel between nodes
// ---------------------------------------------------------------------------

/// A dense hidden-state vector exchanged between DenseMesh nodes.
///
/// **LATENT domain** (AGENTS.md): never crosses `SyncBlock` / chain quorum.
/// Only the input embedding node produces the first `DenseHidden`, and only
/// the output de-embedding node consumes the last one to emit tokens.
///
/// Stored as `Box<[f32]>` for stable layout across passes. Reuse via
/// [`MeshScratch`] to keep the hot path zero-allocation (plasma tier).
#[derive(Clone, Debug)]
pub struct DenseHidden {
    /// Flattened hidden state `[seq_len * hidden_dim]`.
    pub data: Box<[f32]>,
    /// Sequence length (number of token positions).
    pub seq_len: usize,
    /// Hidden dimension per position.
    pub hidden_dim: usize,
}

impl DenseHidden {
    /// Allocate a zeroed hidden state for `seq_len` positions × `hidden_dim`.
    pub fn zeros(seq_len: usize, hidden_dim: usize) -> Self {
        let len = seq_len.checked_mul(hidden_dim).expect("hidden size overflow");
        Self {
            data: vec![0.0f32; len].into_boxed_slice(),
            seq_len,
            hidden_dim,
        }
    }

    /// View as flattened slice.
    pub fn rows(&self) -> &[f32] {
        &self.data
    }

    /// View as mutable flattened slice.
    pub fn rows_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Reset all values to zero (for scratch reuse — plasma tier).
    pub fn clear(&mut self) {
        for v in self.data.iter_mut() {
            *v = 0.0;
        }
    }

    /// Total element count.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Add another hidden state into this one (paper §3.1.3 aggregation = sum).
    ///
    /// Panics in debug if shapes differ. Use for aggregating multiple incoming
    /// edges at a single vertex.
    pub fn add_assign(&mut self, other: &DenseHidden) {
        debug_assert_eq!(self.seq_len, other.seq_len);
        debug_assert_eq!(self.hidden_dim, other.hidden_dim);
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            *dst += *src;
        }
    }
}

// ---------------------------------------------------------------------------
// Topology — layer widths
// ---------------------------------------------------------------------------

/// Role of a layer in the DenseMesh topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LayerRole {
    /// Input layer: embeds tokens -> first `DenseHidden`.
    Input,
    /// Hidden layer: consumes + produces `DenseHidden`.
    Hidden,
    /// Output layer: de-embeds final `DenseHidden` -> tokens.
    Output,
}

/// Layer-wise topology spec (paper §3.1.3).
///
/// Default `1/4/4/4/1` mirrors the paper. Width is runtime-configurable for
/// adaptive-depth inference (BreakevenRouter / CollapseAware integration).
#[derive(Clone, Debug)]
pub struct Topology {
    /// Layer widths, e.g. `[1, 4, 4, 4, 1]`.
    pub widths: Vec<usize>,
}

impl Topology {
    /// Chain topology `[1, 1]` — minimal 2-layer mesh, baseline for gate 1.
    /// One input node, one output node, one edge between them.
    pub fn chain() -> Self {
        Self {
            widths: vec![1, 1],
        }
    }

    /// Diamond `[1, 2, 1]`.
    pub fn diamond() -> Self {
        Self {
            widths: vec![1, 2, 1],
        }
    }

    /// Wide `[1, 4, 4, 4, 1]` (paper default).
    pub fn wide() -> Self {
        Self {
            widths: vec![1, 4, 4, 4, 1],
        }
    }

    /// Number of layers.
    pub fn depth(&self) -> usize {
        self.widths.len()
    }

    /// Width of layer `l`.
    pub fn width(&self, layer: usize) -> usize {
        self.widths[layer]
    }

    /// Role of layer `l`.
    pub fn role(&self, layer: usize) -> LayerRole {
        match layer {
            0 => LayerRole::Input,
            l if l + 1 == self.widths.len() => LayerRole::Output,
            _ => LayerRole::Hidden,
        }
    }

    /// Total number of edges in the layer-wise fully-connected graph.
    /// For adjacent layers (w_l, w_{l+1}), edge count is w_l * w_{l+1}.
    pub fn edge_count(&self) -> usize {
        self.widths.windows(2).map(|w| w[0] * w[1]).sum()
    }
}

impl Default for Topology {
    fn default() -> Self {
        Self::chain()
    }
}

// ---------------------------------------------------------------------------
// ComputeTarget — CPU/GPU/ANE routing
// ---------------------------------------------------------------------------

/// Where a layer's forward pass runs.
///
/// Per optimisation.md: GPU launch overhead is ~50μs — only worth it when
/// width >= 4 (amortise across parallel branches). ANE wins on fixed-shape
/// final decode (per Research 155 / 223).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ComputeTarget {
    /// CPU — single small op, no launch overhead.
    Cpu,
    /// GPU — data-parallel branches (width >= 4).
    Gpu,
    /// Apple Neural Engine — final decode on Apple Silicon.
    Ane,
}

// ---------------------------------------------------------------------------
// MeshConfig — topology + thresholds + compute routing
// ---------------------------------------------------------------------------

/// Configuration for a DenseMesh forward pass.
///
/// Built once (cold tier), reused across queries. Width is runtime-adaptive
/// via [`Topology`]; thresholds govern compute routing.
#[derive(Clone, Debug)]
pub struct MeshConfig {
    /// Topology (layer widths). May be swapped per-query by EdgeBandit.
    pub topology: Topology,
    /// Width threshold above which GPU dispatch is worthwhile.
    /// Default 4 (per optimisation.md ~50us launch cost).
    pub gpu_width_threshold: usize,
    /// Whether to route the output layer to ANE (Apple Silicon only).
    pub prefer_ane_output: bool,
    /// Enable rayon vertex parallelism across hidden nodes when a hidden
    /// layer's width ≥ [`MeshConfig::gpu_width_threshold`] (Issue 020, Path A).
    ///
    /// Defaults to `false` to preserve single-threaded scaling measurements
    /// (e.g. `prof_dense_mesh`). Callers that want the paper's vertex-parameter-
    /// sharing parallel speedup (Gate 4) must opt in. Each rayon worker borrows
    /// the shared `&dyn DenseNode` and must pick up its own per-thread state
    /// via `rayon::current_thread_index()` (see `TransformerNode::ctx_pool`).
    pub enable_vertex_parallelism: bool,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            topology: Topology::chain(),
            gpu_width_threshold: 4,
            prefer_ane_output: true,
            enable_vertex_parallelism: false,
        }
    }
}

impl MeshConfig {
    /// Pick compute target for a layer of given width and role.
    pub fn pick_compute(&self, width: usize, role: LayerRole) -> ComputeTarget {
        if role == LayerRole::Output && self.prefer_ane_output {
            return ComputeTarget::Ane;
        }
        if width >= self.gpu_width_threshold {
            ComputeTarget::Gpu
        } else {
            ComputeTarget::Cpu
        }
    }
}

// ---------------------------------------------------------------------------
// MeshScratch — reusable plasma-tier buffers
// ---------------------------------------------------------------------------

/// Pre-allocated scratch space for DenseMesh forward passes.
///
/// Allocated once (via [`MeshScratch::new`]), reused across calls via `clear()`.
/// Eliminates per-token allocation in the hot path (per optimisation.md).
pub struct MeshScratch {
    /// Aggregation buffer — accumulates incoming edge outputs at each vertex.
    pub aggregate: DenseHidden,
    /// Per-edge output buffer — reused for each edge.route() call.
    pub edge_output: DenseHidden,
    /// Rank scratch buffer for LoRA edges — holds intermediate `[rank]` vector.
    /// Sized to the largest rank across all LoRA edges; reused per position.
    pub rank_buf: Vec<f32>,
}

impl MeshScratch {
    /// Allocate scratch for the given hidden shape.
    pub fn new(seq_len: usize, hidden_dim: usize) -> Self {
        Self {
            aggregate: DenseHidden::zeros(seq_len, hidden_dim),
            edge_output: DenseHidden::zeros(seq_len, hidden_dim),
            rank_buf: Vec::new(),
        }
    }

    /// Allocate scratch with a rank buffer pre-sized (for LoRA edges).
    pub fn with_rank_capacity(seq_len: usize, hidden_dim: usize, max_rank: usize) -> Self {
        Self {
            aggregate: DenseHidden::zeros(seq_len, hidden_dim),
            edge_output: DenseHidden::zeros(seq_len, hidden_dim),
            rank_buf: vec![0.0f32; max_rank],
        }
    }

    /// Reset buffers for reuse (plasma tier pattern).
    pub fn clear(&mut self) {
        self.aggregate.clear();
        self.edge_output.clear();
        // rank_buf is overwritten by LoRA edges before use; no need to clear.
    }
}

// ---------------------------------------------------------------------------
// Bridge functions — latent <-> raw (AGENTS.md compliance)
// ---------------------------------------------------------------------------

/// Project a dense hidden state to a raw scalar via dot-product + sigmoid.
///
/// **Bridge** (AGENTS.md): `raw <- sigmoid(w . latent)`. Use for chain commit
/// of a derived scalar (e.g., confidence, intent). NEVER use this to
/// reconstruct `MapPos` or balances — those stay raw end-to-end.
///
/// Sigmoid (not softmax) per AGENTS.md rule.
#[inline]
pub fn latent_to_raw_scalar(latent: &[f32], direction: &[f32]) -> f32 {
    debug_assert_eq!(latent.len(), direction.len());
    let mut dot = 0.0f32;
    let n = latent.len();
    let mut i = 0;
    // Chunked loop for SIMD auto-vectorisation (per optimisation.md).
    while i + 8 <= n {
        for j in 0..8 {
            dot += latent[i + j] * direction[i + j];
        }
        i += 8;
    }
    while i < n {
        dot += latent[i] * direction[i];
        i += 1;
    }
    // Numerically stable sigmoid.
    if dot >= 0.0 {
        1.0 / (1.0 + (-dot).exp())
    } else {
        let e = dot.exp();
        e / (1.0 + e)
    }
}

/// Lift a raw scalar into a dense direction by scaling `direction` by `scalar`.
///
/// **Bridge** (AGENTS.md): `latent <- scalar * direction`. Use to condition a
/// DenseMesh node from a raw input (e.g., wallet balance, HP) without leaking
/// raw values into the latent channel as plaintext.
#[inline]
pub fn raw_to_latent_projection(scalar: f32, direction: &[f32], out: &mut [f32]) {
    debug_assert_eq!(direction.len(), out.len());
    let n = direction.len();
    let mut i = 0;
    while i + 8 <= n {
        for j in 0..8 {
            out[i + j] = scalar * direction[i + j];
        }
        i += 8;
    }
    while i < n {
        out[i] = scalar * direction[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_hidden_zeros_and_clear() {
        let mut h = DenseHidden::zeros(4, 8);
        assert_eq!(h.len(), 32);
        for v in h.rows() {
            assert_eq!(*v, 0.0);
        }
        for v in h.rows_mut() {
            *v = 1.0;
        }
        h.clear();
        for v in h.rows() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_dense_hidden_add_assign() {
        let mut a = DenseHidden::zeros(2, 4);
        let mut b = DenseHidden::zeros(2, 4);
        for v in a.rows_mut() {
            *v = 1.0;
        }
        for v in b.rows_mut() {
            *v = 2.0;
        }
        a.add_assign(&b);
        for v in a.rows() {
            assert_eq!(*v, 3.0);
        }
    }

    #[test]
    fn test_topology_edge_count() {
        assert_eq!(Topology::chain().edge_count(), 1);
        assert_eq!(Topology::diamond().edge_count(), 2 + 2);
        assert_eq!(
            Topology::wide().edge_count(),
            4 + 4 * 4 + 4 * 4 + 4
        );
    }

    #[test]
    fn test_topology_roles() {
        let t = Topology::wide(); // [1,4,4,4,1]
        assert_eq!(t.role(0), LayerRole::Input);
        assert_eq!(t.role(1), LayerRole::Hidden);
        assert_eq!(t.role(2), LayerRole::Hidden);
        assert_eq!(t.role(3), LayerRole::Hidden);
        assert_eq!(t.role(4), LayerRole::Output);
        // Chain [1,1] has just input and output.
        let c = Topology::chain();
        assert_eq!(c.role(0), LayerRole::Input);
        assert_eq!(c.role(1), LayerRole::Output);
    }

    #[test]
    fn test_mesh_config_pick_compute() {
        let cfg = MeshConfig::default();
        assert_eq!(cfg.pick_compute(1, LayerRole::Hidden), ComputeTarget::Cpu);
        assert_eq!(cfg.pick_compute(4, LayerRole::Hidden), ComputeTarget::Gpu);
        assert_eq!(cfg.pick_compute(1, LayerRole::Output), ComputeTarget::Ane);
        assert_eq!(cfg.pick_compute(1, LayerRole::Input), ComputeTarget::Cpu);
    }

    #[test]
    fn test_latent_to_raw_scalar_sigmoid_range() {
        let direction = [1.0f32; 8];
        let latent_pos = [2.0f32; 8];
        let latent_neg = [-2.0f32; 8];
        let pos = latent_to_raw_scalar(&latent_pos, &direction);
        let neg = latent_to_raw_scalar(&latent_neg, &direction);
        // sigmoid(16) ≈ 0.99999, sigmoid(-16) ≈ 0.00001.
        assert!(pos > 0.99 && pos < 1.0, "pos = {pos}");
        assert!(neg > 0.0 && neg < 0.01, "neg = {neg}");
    }

    #[test]
    fn test_latent_to_raw_scalar_zero() {
        let direction = [1.0f32; 8];
        let latent_zero = [0.0f32; 8];
        let z = latent_to_raw_scalar(&latent_zero, &direction);
        // sigmoid(0) = 0.5.
        assert!((z - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_raw_to_latent_projection_scales() {
        let direction = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 8];
        raw_to_latent_projection(2.0, &direction, &mut out);
        assert_eq!(out, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }
}
