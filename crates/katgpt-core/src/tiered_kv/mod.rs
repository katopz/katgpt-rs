//! Tiered Hot/Warm/Cold K/V Store — the route-and-fetch substrate for
//! sparse long-context attention (Plan 397, Research 379).
//!
//! Distilled from Frank, Fedosov, Grinenko (BMW Group) 2026, *"Hierarchical
//! Global Attention"* ([arxiv 2606.30709](https://arxiv.org/abs/2606.30709)).
//!
//! # What this is
//!
//! A generic abstraction for a 3-temperature K/V store that lets a sparse
//! attention primitive fetch only the routed working set from a tiered backing
//! store, instead of materializing the entire historical K/V cache.
//!
//! - **Hot** — always-resident: chunk/group summaries + sink chunk + local window.
//!   Small (summaries are sub-chunk centroids; sink+local are bounded windows).
//! - **Warm** — bounded LRU shard cache for recently-routed token chunks.
//! - **Cold** — full token K/V in process RAM (reference impl). Production
//!   mmap/NVMe-backed variant is a riir-ai/riir-neuron-db follow-up.
//!
//! The trait is intentionally generic over `D` (head dim) and does not mandate
//! RoPE, entmax, or any attention-specific policy — the routing decision is
//! made by the caller (e.g. HGA's `forward_hga`), the store only fetches.
//!
//! # Why always-on (not feature-gated)
//!
//! `TieredKvStore` is a generic trait + reference impl with no attention-layer
//! dependencies. It's useful for any future route-and-fetch pattern, not just
//! HGA. Keep it ungated; the HGA-specific code that consumes it is gated by `hga`.
//!
//! Tiered Hot/Warm/Cold K/V Store — the route-and-fetch substrate for sparse
//! long-context attention (Plan 397, Research 379, arxiv 2606.30709). Generic
//! trait + in-memory reference impl. Always-on (no feature gate) because it's
//! a generic primitive with no attention-layer deps; the HGA-specific consumer
//! is gated by `hga`.

pub mod in_memory;

pub use in_memory::InMemoryTieredKvStore;

/// Sink + local window specification — the always-visible token set.
///
/// In HGA's defaults (paper §3.3): sink = first 2 chunks, local = last 8 chunks.
/// These tokens always enter the output softmax regardless of routing — they
/// ground the query in the sequence's start (sink) and immediate context (local).
#[derive(Clone, Debug)]
pub struct SinkLocalSet {
    /// Chunk indices of the sink (typically the first few chunks — attention sink).
    pub sink_chunks: Vec<usize>,
    /// Chunk indices of the local window (typically the last few chunks — recent context).
    pub local_chunks: Vec<usize>,
}

impl SinkLocalSet {
    /// Construct from explicit chunk lists.
    pub fn new(sink_chunks: Vec<usize>, local_chunks: Vec<usize>) -> Self {
        Self {
            sink_chunks,
            local_chunks,
        }
    }

    /// All always-visible chunk indices (sink ∪ local), deduplicated + sorted.
    pub fn all_chunks(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.sink_chunks.len() + self.local_chunks.len());
        out.extend_from_slice(&self.sink_chunks);
        out.extend_from_slice(&self.local_chunks);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Whether a chunk is in the always-visible set.
    pub fn contains(&self, chunk_idx: usize) -> bool {
        self.sink_chunks.contains(&chunk_idx) || self.local_chunks.contains(&chunk_idx)
    }
}

/// Route budget — how many chunks and groups the routing pass may select.
///
/// `usize::MAX` means "all" (full-coverage mode = causal SDPA, used by G1).
#[derive(Clone, Copy, Debug)]
pub struct RouteBudget {
    /// Max number of chunks to select at stage 1 (chunk-level entmax).
    pub k_c: usize,
    /// Max number of groups to select at stage 2 (within selected chunks).
    pub k_g: usize,
}

impl RouteBudget {
    /// Full coverage — select all chunks and all groups. Degenerates to causal SDPA.
    pub const FULL: Self = Self {
        k_c: usize::MAX,
        k_g: usize::MAX,
    };
}

/// The fetched working set — real token K/V from sink+local+routed chunks.
///
/// This is the output of `route_and_fetch`. The caller runs standard softmax
/// attention (SDPA) over these exact token K/V pairs. **Summary K/V never
/// enters the output** — the HGA "summary-keys-route, real-keys-compute" rule.
pub struct WorkingSet {
    /// Flattened keys `[n_tokens * D]` (row-major: token 0, then token 1, ...).
    pub keys: Vec<f32>,
    /// Flattened values `[n_tokens * D]` (aligned with keys).
    pub values: Vec<f32>,
    /// Number of tokens in the working set.
    pub n_tokens: usize,
    /// Per-token position (for RoPE application at read time if needed).
    pub positions: Vec<usize>,
}

impl WorkingSet {
    /// Empty working set.
    pub fn empty() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            n_tokens: 0,
            positions: Vec::new(),
        }
    }

    /// Append a token's K/V to the working set.
    pub fn push_token(&mut self, key: &[f32], value: &[f32], position: usize) {
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        self.positions.push(position);
        self.n_tokens += 1;
    }
}

/// Which groups within which chunks are selected by stage-2 routing.
///
/// Sparse representation: each entry is `(chunk_idx, first_group, n_groups)`
/// — a contiguous group range selected within that chunk.
pub struct GroupSelection {
    /// `(chunk_idx, first_group, n_groups)` — selected groups within a chunk.
    pub selections: Vec<(usize, usize, usize)>,
}

impl GroupSelection {
    /// Empty selection (no groups selected).
    pub fn empty() -> Self {
        Self {
            selections: Vec::new(),
        }
    }

    /// All groups within all chunks (full coverage — G1 test mode).
    pub fn all_groups(n_chunks: usize, n_groups_per_chunk: usize) -> Self {
        let selections = (0..n_chunks).map(|c| (c, 0, n_groups_per_chunk)).collect();
        Self { selections }
    }

    /// Add a contiguous group range within a chunk.
    pub fn add_range(&mut self, chunk_idx: usize, first_group: usize, n_groups: usize) {
        self.selections.push((chunk_idx, first_group, n_groups));
    }
}

/// Generic tiered K/V store trait.
///
/// The store holds real token K/V in the cold tier and summary/summary-derived
/// routing data in the hot tier. `fetch_working_set` returns the exact-token
/// working set for the output softmax.
///
/// The actual routing policy (chunk entmax, group dot-product) is implemented
/// by the HGA forward path (`katgpt_attn::hga_forward`), NOT by this trait.
/// The store provides the data; the forward path provides the policy. This
/// keeps the trait generic and reusable.
pub trait TieredKvStore {
    /// Head dimension `D`.
    fn head_dim(&self) -> usize;

    /// Chunk size `C` (tokens per chunk).
    fn chunk_size(&self) -> usize;

    /// Group size `GS` (tokens per group; must divide chunk_size).
    fn group_size(&self) -> usize;

    /// Number of chunks stored so far.
    fn n_chunks(&self) -> usize;

    /// Append a chunk of token K/V to the cold tier and compute/append its
    /// group summaries to the hot tier.
    ///
    /// - `keys` — flattened `[C * D]` key vectors (already RoPE-rotated at their positions).
    /// - `values` — flattened `[C * D]` value vectors.
    /// - `positions` — `[C]` token positions.
    fn append_chunk(&mut self, keys: &[f32], values: &[f32], positions: &[usize]);

    /// Fetch the full working set: sink + local + routed chunks' selected tokens.
    ///
    /// `selected_chunks` — chunks selected by stage-1 chunk routing (plus sink+local).
    /// `group_selection` — which groups within those chunks survived stage-2 group routing.
    fn fetch_working_set(
        &self,
        sink_local: &SinkLocalSet,
        selected_chunks: &[usize],
        group_selection: &GroupSelection,
    ) -> WorkingSet;
}

#[cfg(test)]
mod tests;
