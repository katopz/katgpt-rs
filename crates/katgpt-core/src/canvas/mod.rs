//! Canvas Schema Compiler — declared causal topology for attention masks
//! (Plan 419, Research 398, Valdez *Canvas Engineering* July 2026).
//!
//! ## What this is
//!
//! The **modelless half** of canvas engineering: a typed [`CanvasSchema`]
//! compiler that lowers a declared region layout + directed topology into
//! (a) an [`AttentionMaskSpec`] consumable by existing sparse-attention paths
//! (AC-Prefix, VortexFlow), (b) a [`LossWeightMask`] for training-time
//! callers, and (c) a [`reachability_horizon`] / [`can_reach`] primitive
//! proving the **exact marginal independence** guarantee for binary masks.
//! Plus a [`transfer_distance`] semantic-type compatibility scalar.
//!
//! ## The load-bearing guarantee
//!
//! For a **binary** mask (`weight ∈ {0, 1}`): if there is no directed path
//! `a → b` in the information-flow graph, then region `a` cannot influence
//! region `b`. This is exact marginal independence, **by construction** —
//! it holds because an absent edge is a hard prohibition in the mask.
//!
//! This ships on structural/correctness merits, like the DEC `d∘d=0` identity
//! (Plan 251). The *behavioral* gain (paper §5's 1.73× parameter efficiency)
//! is training-dependent and tracked separately in Issue 043 (resolved + removed).
//!
//! ## Quick start
//!
//! ### (a) Declare + compile a canvas
//!
//! ```
//! use katgpt_core::canvas::{
//!     compile_schema, dense, region_indices, CanvasBounds, CanvasLayout, CanvasSchema,
//!     RegionId, RegionSpec, AttentionFnFamily,
//! };
//!
//! // Two regions: output [0,2) and input [2,4) on a 1×1×4 canvas.
//! let layout = CanvasLayout {
//!     t: 1, h: 1, w: 4, d_model: 8,
//!     regions: vec![
//!         RegionSpec::new("out", CanvasBounds::new(0,1,0,1,0,2), 1, true, 1.0, None, AttentionFnFamily::Cross),
//!         RegionSpec::new("in",  CanvasBounds::new(0,1,0,1,2,4), 1, false, 0.0, None, AttentionFnFamily::Cross),
//!     ],
//! };
//! let schema = CanvasSchema { layout, topology: dense(&[RegionId(0), RegionId(1)]) };
//! let compiled = compile_schema(&schema);
//! assert_eq!(compiled.mask.n_positions, 4);
//! ```
//!
//! ### (b) Check the reachability guarantee
//!
//! With an **isolated** topology (no cross-region edges), region 0 cannot
//! reach region 1 at any horizon:
//!
//! ```
//! use katgpt_core::canvas::{
//!     build_flow_graph, can_reach, isolated, CanvasBounds, CanvasLayout, RegionId, RegionSpec,
//!     AttentionFnFamily,
//! };
//!
//! let layout = CanvasLayout {
//!     t: 1, h: 1, w: 4, d_model: 8,
//!     regions: vec![
//!         RegionSpec::new("a", CanvasBounds::new(0,1,0,1,0,2), 1, false, 0.0, None, AttentionFnFamily::Cross),
//!         RegionSpec::new("b", CanvasBounds::new(0,1,0,1,2,4), 1, false, 0.0, None, AttentionFnFamily::Cross),
//!     ],
//! };
//! let topo = isolated(&[RegionId(0), RegionId(1)]);
//! let g = build_flow_graph(&topo, 2);
//! // No directed path a → b ⇒ exact marginal independence (holds for all horizons).
//! assert!(!can_reach(&g, RegionId(0), RegionId(1), 1));
//! assert!(!can_reach(&g, RegionId(0), RegionId(1), 1000));
//! ```
//!
//! ## Why modelless
//!
//! Every primitive here is a pure function over index sets + graphs. Zero
//! backprop, zero weight mutation. The compiled mask is structure; the
//! *interpretation* of that structure (attention computation) lives in the
//! consumer's existing attention path.
//!
//! ## §3.5 modelless-unblock relevance
//!
//! Canvas engineering's behavioral headline (parameter efficiency, cortical
//! fit) is genuinely training-dependent (Research 398 §2.2 exhausted the three
//! modelless paths). But the *compiler* + *reachability guarantee* ship
//! modellessly on correctness merits — the same framing as the DEC identity
//! operators. A systematically biased topology can be *declared* and *verified*
//! modellessly; only *exploiting* it for a behavioral gain needs riir-train.
//!
//! ## Fusion (the Super-GOAT angle, tracked in Issue 043 — resolved + removed)
//!
//! - **F1**: Canvas compiler × DEC reachability → declared causal topology on
//!   latent positions with graph-theoretic independence guarantees.
//! - **F2**: Canvas × `region_subspace_bridge` (Plan 416) → per-NPC compiled
//!   cognitive stack (perception → affect → action).
//! - **F3**: Canvas schema × freeze/thaw → schema-keyed latent exchange
//!   (same-schema NPCs swap latents directly).
//!
//! ## Status
//!
//! Phase 1–4 in progress (Plan 419). Opt-in via `canvas_schema` until the
//! GOAT gate (G1–G6) passes. The reachability soundness test (G1) is the
//! load-bearing gate.
//!
//! See: `katgpt-rs/.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md`
//! See: `katgpt-rs/.plans/419_canvas_schema_compiler.md`
//!
//! Canvas Schema Compiler — declared causal topology for attention masks
//! (Plan 419, Research 398, Valdez *Canvas Engineering* July 2026). The
//! modelless half: a typed CanvasSchema compiler that lowers a declared region
//! layout + directed topology into an AttentionMaskSpec (consumable by
//! AC-Prefix / VortexFlow), a LossWeightMask, and a can_reach / reachability
//! primitive proving exact marginal independence for binary masks (absent
//! edge ⟹ no influence, by construction). Plus a transfer_distance semantic-
//! type compatibility scalar. Pure structure compilation, zero gradient
//! descent. STAYS OPT-IN — G1–G6 GOAT gates ALL PASS (Bench 419, 2026-07-09);
//! not promoted to default because the fusion PoC resolved inconclusively
//! (Issue 043, removed) and the primitive's constituents are already default-on
//! with runtime consumers.

pub mod mask;
pub mod reachability;
pub mod transfer;
pub mod types;

// Re-export the public API at the module root for ergonomic access.
pub use mask::{build_attention_mask, build_loss_weight_mask, temporal_aligns};
pub use reachability::{
    FlowGraph, TransitiveClosure, build_flow_graph, can_reach, reachability_horizon,
};
pub use transfer::{compatible_regions, compatible_regions_in_layout, transfer_distance};
pub use types::{
    AttentionFnFamily, AttentionMaskSpec, CanvasBounds, CanvasLayout, CanvasSchema, CanvasTopology,
    CompiledCanvas, Connection, LossWeightMask, RegionId, RegionSpec, SEMANTIC_EMBED_DIM,
    SemanticType,
};

use core::ops::Range;

// ─── T1.3: region index arithmetic ──────────────────────────────────────────

/// Compute the contiguous flat-position range occupied by a region.
///
/// Per paper §2.3, a region's index set `I_r` is the set of flat positions
/// inside its axis-aligned bounds. Because regions are axis-aligned boxes and
/// the flat layout is row-major `(t, h, w)`, `I_r` is **not** contiguous in
/// general (a region spanning multiple frames is a stride-separated union of
/// per-frame blocks).
///
/// **This implementation returns the *offset* range `[base, base + volume)`**
/// where `base` is the first position of the region's bounds and `volume` is
/// the region's cell count. This is the convention used throughout the
/// compiler: each region owns a contiguous slab `[base, base + volume)` of the
/// flat index, and positions are assigned in declaration order with no gaps.
/// This makes the mask builder's inner loop a tight `for i in range` with no
/// stride arithmetic, at the cost of requiring regions to be laid out
/// contiguously by the schema author (overlapping/gappy regions are a caller
/// responsibility, same as the paper's own assumption).
///
/// # Allocation
///
/// Zero. Returns a `Range<usize>` by value.
///
/// # Panics
///
/// None. A region whose bounds fall outside the canvas contributes a
/// (possibly empty) range; the compiler does not validate bounds here.
pub fn region_indices(spec: &RegionSpec, layout: &CanvasLayout) -> Range<usize> {
    // Base offset of the region's lower corner in the flat (t,h,w) layout.
    let b = spec.bounds;
    let hw = (layout.h as usize).saturating_mul(layout.w as usize);
    let base = (b.t0 as usize) * hw + (b.h0 as usize) * (layout.w as usize) + b.w0 as usize;
    let volume = b.volume();
    base..base + volume
}

// ─── T1.4: topology constructors ────────────────────────────────────────────

/// Fully-connected topology over the given regions: every region queries every
/// region (including itself). Paper §2.2 "dense".
///
/// For `n` regions this emits `n²` connections.
pub fn dense(regions: &[RegionId]) -> CanvasTopology {
    let mut connections = Vec::with_capacity(regions.len() * regions.len());
    for &dst in regions {
        for &src in regions {
            connections.push(Connection::new(src, dst));
        }
    }
    CanvasTopology { connections }
}

/// Block-diagonal (isolated) topology: each region self-attends only, no
/// cross-region edges. Paper §2.2 "isolated".
///
/// This is the topology that makes the reachability guarantee vacuously
/// strongest: no region can influence any other.
pub fn isolated(regions: &[RegionId]) -> CanvasTopology {
    let mut connections = Vec::with_capacity(regions.len());
    for &r in regions {
        connections.push(Connection::new(r, r)); // self-attention
    }
    CanvasTopology { connections }
}

/// Hub-and-spoke topology: the hub queries every spoke and every spoke queries
/// the hub (bidirectional hub access), but spokes do not query each other.
/// Paper §2.2 "hub-spoke".
pub fn hub_spoke(hub: RegionId, spokes: &[RegionId]) -> CanvasTopology {
    let mut connections = Vec::with_capacity(1 + 2 * spokes.len());
    // Hub self-attends.
    connections.push(Connection::new(hub, hub));
    for &s in spokes {
        // Spoke queries hub (hub → spoke as content source).
        connections.push(Connection::new(hub, s));
        // Hub queries spoke.
        connections.push(Connection::new(s, hub));
        // Spoke self-attends.
        connections.push(Connection::new(s, s));
    }
    CanvasTopology { connections }
}

/// Causal chain topology `A → B → C → …` (information flow). Each region
/// self-attends and *queries its predecessor*, so information flows forward
/// along the chain. Paper §2.2 "chain".
///
/// For `chain = [r0, r1, …, rn]` this emits, for each region, a self-connection
/// `(r, r)`, and for each `k ≥ 1` the connection `Connection(r_k, r_{k-1})` —
/// i.e. `r_k` queries `r_{k-1}`. Because information flows `dst → src` (the
/// key-value region influences the querier; see
/// [`crate::canvas::reachability`]), the resulting information-flow
/// arcs are `r_{k-1} → r_k`, i.e. `r0 → r1 → … → rn`.
///
/// This direction is what makes Plan 419 T3.6 hold: for `causal_chain([A,B,C])`,
/// `can_reach(A, C, 1) == false` but `can_reach(A, C, 2) == true`. Self-loops do
/// not affect cross-region reachability.
pub fn causal_chain(chain: &[RegionId]) -> CanvasTopology {
    let mut connections = Vec::with_capacity(chain.len() * 2);
    for (k, &r) in chain.iter().enumerate() {
        connections.push(Connection::new(r, r)); // self-attention
        if k > 0 {
            // r queries its predecessor ⟹ info arc predecessor → r.
            connections.push(Connection::new(r, chain[k - 1]));
        }
    }
    CanvasTopology { connections }
}

/// Causal-temporal topology: each region self-attends within the same frame and
/// cross-attends to itself in the *previous* frame only (no future leakage).
/// Paper §2.2 "causal temporal" — the default for temporal canvases.
///
/// Emits, per region: a same-frame self-connection `(t_src=None, t_dst=None)`
/// and a previous-frame self-connection with `(t_src=Some(-1), t_dst=Some(0))`
/// (the source frame is one behind the destination frame, for a shared ref).
pub fn causal_temporal(regions: &[RegionId]) -> CanvasTopology {
    let mut connections = Vec::with_capacity(regions.len() * 2);
    for &r in regions {
        // Same-frame self-attention.
        connections.push(Connection::new(r, r));
        // Previous-frame self-attention: src frame = ref-1, dst frame = ref+0.
        connections.push(Connection {
            src: r,
            dst: r,
            weight: 1.0,
            t_src: Some(-1),
            t_dst: Some(0),
            fn_family: None,
        });
    }
    CanvasTopology { connections }
}

// ─── T2.5: the top-level compiler ───────────────────────────────────────────

/// THE COMPILER. Pure structure, zero gradient descent.
///
/// Lowers a [`CanvasSchema`] into a [`CompiledCanvas`]: per-region flat index
/// ranges, a sparse [`AttentionMaskSpec`], and a [`LossWeightMask`].
///
/// # Allocation
///
/// Three allocations (region_indices, mask edges, loss weights), all at
/// schema-load time. Subsequent queries (`can_reach`, `transfer_distance`) are
/// alloc-free against the compiled artifact.
///
/// # Example
///
/// See the module-level quick-start.
pub fn compile_schema(schema: &CanvasSchema) -> CompiledCanvas {
    let region_indices: Vec<Range<usize>> = schema
        .layout
        .regions
        .iter()
        .map(|r| region_indices(r, &schema.layout))
        .collect();
    let mask = build_attention_mask(&schema.topology, &region_indices, &schema.layout);
    let loss_mask = build_loss_weight_mask(&schema.layout, &region_indices);
    CompiledCanvas {
        region_indices,
        mask,
        loss_mask,
    }
}

/// Convenience: build the flow graph directly from a topology + region count.
///
/// This is a thin alias for [`reachability::build_flow_graph`], re-exported at
/// the module root so callers can `use katgpt_core::canvas::build_flow_graph`.
/// (The `_via_compile` name exists only in tests as a second alias to avoid a
/// name collision in the test module.)
#[doc(hidden)]
pub fn build_flow_graph_via_compile(topology: &CanvasTopology, n_regions: usize) -> FlowGraph {
    build_flow_graph(topology, n_regions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_indices_returns_contiguous_slab() {
        let layout = CanvasLayout {
            t: 1,
            h: 1,
            w: 4,
            d_model: 8,
            regions: vec![],
        };
        // Region covering w=[2,4) on a 1×1×4 canvas → base 2, volume 2.
        let spec = RegionSpec::new(
            "r",
            CanvasBounds::new(0, 1, 0, 1, 2, 4),
            1,
            false,
            0.0,
            None,
            AttentionFnFamily::Cross,
        );
        let range = region_indices(&spec, &layout);
        assert_eq!(range, 2..4);
    }

    #[test]
    fn region_indices_2d_layout_base_arithmetic() {
        // 1×2×2 canvas (h=2, w=2): positions are
        //   (h0,w0)=0, (h0,w1)=1, (h1,w0)=2, (h1,w1)=3.
        let layout = CanvasLayout {
            t: 1,
            h: 2,
            w: 2,
            d_model: 8,
            regions: vec![],
        };
        // Region covering h=[1,2), w=[0,2) → base = 1*2 + 0 = 2, volume = 1*2 = 2.
        let spec = RegionSpec::new(
            "r",
            CanvasBounds::new(0, 1, 1, 2, 0, 2),
            1,
            false,
            0.0,
            None,
            AttentionFnFamily::Cross,
        );
        let range = region_indices(&spec, &layout);
        assert_eq!(range, 2..4);
    }

    #[test]
    fn dense_topology_edge_count() {
        let topo = dense(&[RegionId(0), RegionId(1), RegionId(2)]);
        assert_eq!(topo.n_edges(), 9); // 3×3
    }

    #[test]
    fn isolated_topology_edge_count() {
        let topo = isolated(&[RegionId(0), RegionId(1), RegionId(2)]);
        assert_eq!(topo.n_edges(), 3); // self only
        for c in &topo.connections {
            assert_eq!(c.src, c.dst);
        }
    }

    #[test]
    fn hub_spoke_topology_structure() {
        let topo = hub_spoke(RegionId(0), &[RegionId(1), RegionId(2)]);
        // Hub self + 2 spokes × (spoke→hub, hub→spoke, spoke self) = 1 + 6 = 7.
        assert_eq!(topo.n_edges(), 7);
    }

    #[test]
    fn causal_chain_topology_structure() {
        let topo = causal_chain(&[RegionId(0), RegionId(1), RegionId(2)]);
        // Each region self-attends (3) + chain[1] queries chain[0], chain[2] queries
        // chain[1] (2) = 5. Information arcs (dst→src) are 0→1→2.
        assert_eq!(topo.n_edges(), 5);
        assert!(
            topo.connections
                .contains(&Connection::new(RegionId(1), RegionId(0)))
        );
        assert!(
            topo.connections
                .contains(&Connection::new(RegionId(2), RegionId(1)))
        );
    }

    #[test]
    fn causal_temporal_emits_same_and_previous_frame() {
        let topo = causal_temporal(&[RegionId(0)]);
        assert_eq!(topo.n_edges(), 2);
        // First: same-frame self (no offsets).
        assert!(topo.connections[0].t_src.is_none());
        // Second: previous-frame self (t_src=-1, t_dst=0).
        assert_eq!(topo.connections[1].t_src, Some(-1));
        assert_eq!(topo.connections[1].t_dst, Some(0));
    }
}
