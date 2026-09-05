//! Mask + loss-weight compilation for the Canvas Schema Compiler.
//!
//! Lowers a [`CanvasTopology`] into a sparse [`AttentionMaskSpec`] and a
//! [`LossWeightMask`] over the flat position index. This is the "backend" of
//! [`super::compile_schema`].
//!
//! ## Allocation discipline
//!
//! [`build_attention_mask`] does a pre-scan to size the edge vector exactly,
//! then fills it — one allocation total. [`build_loss_weight_mask`] allocates
//! the length-`N` weight vector once. Neither is a hot path; both run at
//! schema-load time.

use core::ops::Range;

use super::types::{AttentionMaskSpec, CanvasLayout, CanvasTopology, Connection, LossWeightMask};

/// The temporal-alignment predicate `A_τ` from paper §2.3.
///
/// For a connection with temporal offsets `(t_src, t_dst)`, two positions
/// `(t_i, t_j)` align iff there exists a reference `ref` such that
/// `t_i = ref + t_src ∧ t_j = ref + t_dst`.
///
/// - If **both** offsets are `None`: always align (unconstrained).
/// - If exactly one is `None`: align iff the set one matches (the other is
///   free, so any `ref` satisfying the set offset works — but the two
///   positions must still share a `ref`; for a single set offset `t_src = k`,
///   this means `t_i = ref + k` for *some* `ref`, which is always satisfiable
///   by choosing `ref = t_i - k`. The unset side is then `t_j = ref + (free)`,
///   which is unconstrained. **Conclusion: a single `None` side is
///   unconstrained for that side.** In practice the paper always sets both or
///   neither; this implementation handles the asymmetric case correctly.)
/// - If **both** are set: `t_i - t_src == t_j - t_dst` (they share a `ref`).
///
/// `t_i` is the query position's frame, `t_j` the key position's frame.
#[inline]
pub fn temporal_aligns(t_src: Option<i32>, t_dst: Option<i32>, t_i: u32, t_j: u32) -> bool {
    match (t_src, t_dst) {
        (None, None) => true,
        // Single-sided: the set side determines ref, the free side is unconstrained.
        (Some(_), None) | (None, Some(_)) => true,
        (Some(k_src), Some(k_dst)) => {
            // t_i - k_src == t_j - k_dst  ⟺  t_i - t_j == k_src - k_dst
            (t_i as i64 - t_j as i64) == (k_src as i64 - k_dst as i64)
        }
    }
}

/// Build the sparse attention mask from a topology + per-region index ranges.
///
/// For each present `Connection { src, dst, weight, t_src, t_dst, .. }`:
/// - For each query position `i ∈ region_indices[src]`, key position
///   `j ∈ region_indices[dst]`:
///   - If [`temporal_aligns`] `(t_src, t_dst, frame(i), frame(j))`: emit
///     `(i, j, weight)`.
///
/// **Direction (paper §2.2, authoritative):** `Connection(src, dst)` licenses
/// `src` tokens to *query* `dst` keys/values. So in the mask `M_ij`, the
/// **query** (row) index `i` comes from `src` and the **key** (column) index
/// `j` comes from `dst`. The emitted triple is `(query=src_pos, key=dst_pos,
/// weight)`. Information flows `dst → src` (the key region influences the
/// querier), which is the direction the reachability graph tracks.
///
/// This convention is what makes Plan 419 T3.6 hold: for `causal_chain([A,B,C])`
/// each region queries its predecessor, so information flows `A → B → C` and
/// `can_reach(A, C, 2) == true`.
///
/// # Allocation
///
/// One allocation: `edges.reserve_exact(total_pairs)` where `total_pairs` is
/// pre-computed by summing `|src|·|dst|` over all present connections. Then
/// fill. Zero per-edge allocation.
///
/// # Panics
///
/// Panics if a connection's `src`/`dst` index is out of bounds for
/// `region_indices`, or if `region_indices[r]` extends past the layout's flat
/// position count (checked via the frame lookup only when temporal offsets are
/// set; structural bounds are the caller's responsibility).
pub fn build_attention_mask(
    topology: &CanvasTopology,
    region_indices: &[Range<usize>],
    layout: &CanvasLayout,
) -> AttentionMaskSpec {
    let n_positions = layout.n_positions();
    let h_extent = layout.h as usize;

    // Pre-scan: total edge count = Σ |dst|·|src| over present connections.
    let total_pairs: usize = topology
        .connections
        .iter()
        .filter(|c| c.is_present())
        .map(|c| {
            let src_range = &region_indices[c.src.get()];
            let dst_range = &region_indices[c.dst.get()];
            src_range.len().saturating_mul(dst_range.len())
        })
        .sum();

    let mut edges: Vec<(usize, usize, f32)> = Vec::with_capacity(total_pairs);

    for &conn in &topology.connections {
        if !conn.is_present() {
            continue;
        }
        let Connection {
            src,
            dst,
            weight,
            t_src,
            t_dst,
            ..
        } = conn;
        let src_range = &region_indices[src.get()];
        let dst_range = &region_indices[dst.get()];

        let both_temporal_set = t_src.is_some() && t_dst.is_some();

        if !both_temporal_set {
            // Fast path: emit the full cartesian product, no per-pair frame check.
            // `Range<usize>` is a by-value iterator — iterate it directly, no alloc.
            for i in src_range.clone() {
                // query = src position (paper §2.2: src queries dst)
                for j in dst_range.clone() {
                    // key = dst position
                    edges.push((i, j, weight));
                }
            }
        } else {
            // Slow path: per-pair temporal alignment.
            for i in src_range.clone() {
                for j in dst_range.clone() {
                    let t_i = frame_of(i, h_extent, layout.w as usize);
                    let t_j = frame_of(j, h_extent, layout.w as usize);
                    if temporal_aligns(t_src, t_dst, t_i, t_j) {
                        edges.push((i, j, weight));
                    }
                }
            }
        }
    }

    AttentionMaskSpec { n_positions, edges }
}

/// Recover the temporal frame index of a flat position.
///
/// Flat layout is row-major over `(t, h, w)`: position `p` maps to
/// `(t, h, w)` where `p = t · (H·W) + h · W + w`. So `t = p / (H·W)`.
#[inline]
fn frame_of(flat: usize, h_extent: usize, w_extent: usize) -> u32 {
    let hw = h_extent.saturating_mul(w_extent);
    (flat.checked_div(hw).unwrap_or(0)) as u32
}

/// Build the per-position loss-weight mask.
///
/// `ω_i = Σ_r 1[i ∈ I_r] · loss_weight_r · 1[is_output_r]`.
///
/// Non-output regions contribute nothing. Positions covered by multiple
/// output regions sum their weights (paper §2.3 does not forbid overlap;
/// overlapping regions are the caller's responsibility).
///
/// # Allocation
///
/// One allocation: the length-`N` weight vector, zero-initialized, then
/// accumulation in place.
pub fn build_loss_weight_mask(
    layout: &CanvasLayout,
    region_indices: &[Range<usize>],
) -> LossWeightMask {
    let n = layout.n_positions();
    let mut weights = vec![0.0_f32; n];

    for (r_idx, region) in layout.regions.iter().enumerate() {
        if !region.is_output {
            continue;
        }
        let range = &region_indices[r_idx];
        let w = region.loss_weight;
        for i in range.clone() {
            // Guard against a region range that overshoots N (defensive;
            // well-formed schemas keep ranges inside [0, N)).
            if i < n {
                weights[i] += w;
            }
        }
    }

    LossWeightMask { weights }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::types::{
        AttentionFnFamily, CanvasBounds, CanvasTopology, RegionId, RegionSpec,
    };
    use crate::canvas::{
        CanvasLayout, CanvasSchema, causal_chain, compile_schema, dense, isolated,
    };

    /// A 2-region canvas: region 0 = output, region 1 = input-only.
    fn two_region_layout() -> CanvasLayout {
        CanvasLayout {
            t: 1,
            h: 1,
            w: 4,
            d_model: 8,
            regions: vec![
                RegionSpec {
                    name: "out",
                    bounds: CanvasBounds::new(0, 1, 0, 1, 0, 2),
                    period: 1,
                    is_output: true,
                    loss_weight: 1.0,
                    semantic_type: None,
                    default_attn: AttentionFnFamily::Cross,
                },
                RegionSpec {
                    name: "in",
                    bounds: CanvasBounds::new(0, 1, 0, 1, 2, 4),
                    period: 1,
                    is_output: false,
                    loss_weight: 0.0,
                    semantic_type: None,
                    default_attn: AttentionFnFamily::Cross,
                },
            ],
        }
    }

    #[test]
    fn temporal_aligns_both_none_is_unconstrained() {
        assert!(temporal_aligns(None, None, 0, 0));
        assert!(temporal_aligns(None, None, 5, 3));
    }

    #[test]
    fn temporal_aligns_both_set_shares_reference() {
        // ref = 0: t_i = t_src = 2, t_j = t_dst = 5 → (2,5) aligns.
        assert!(temporal_aligns(Some(2), Some(5), 2, 5));
        // (3, 6): ref = 1, t_src=2→3 ✓, t_dst=5→6 ✓ → aligns.
        assert!(temporal_aligns(Some(2), Some(5), 3, 6));
        // (2, 6): t_i - t_j = -4, k_src - k_dst = -3 → no.
        assert!(!temporal_aligns(Some(2), Some(5), 2, 6));
    }

    #[test]
    fn loss_weight_mask_zeroes_non_output_regions() {
        let layout = two_region_layout();
        let indices: Vec<Range<usize>> = layout
            .regions
            .iter()
            .map(|r| crate::canvas::region_indices(r, &layout))
            .collect();
        let mask = build_loss_weight_mask(&layout, &indices);
        // Region 0 (output) covers positions [0,2), region 1 (input) covers [2,4).
        assert_eq!(mask.weights.len(), 4);
        assert!((mask.weights[0] - 1.0).abs() < 1e-6);
        assert!((mask.weights[1] - 1.0).abs() < 1e-6);
        assert!(mask.weights[2].abs() < 1e-6); // non-output region
        assert!(mask.weights[3].abs() < 1e-6);
    }

    #[test]
    fn build_mask_dense_topology_emits_full_product() {
        // Two regions of size 2 each. Dense topology → 2×2 = 4 edges (each region
        // also self-attends: connections from A→A, A→B, B→A, B→B).
        let layout = two_region_layout();
        let indices: Vec<Range<usize>> = layout
            .regions
            .iter()
            .map(|r| crate::canvas::region_indices(r, &layout))
            .collect();
        // dense over both regions: 4 connections, each contributing 2×2 = 4 → 16 edges.
        let topo = dense(&[RegionId(0), RegionId(1)]);
        let mask = build_attention_mask(&topo, &indices, &layout);
        assert_eq!(mask.n_positions, 4);
        assert_eq!(mask.n_edges(), 16);
    }

    #[test]
    fn build_mask_isolated_topology_is_block_diagonal() {
        let layout = two_region_layout();
        let indices: Vec<Range<usize>> = layout
            .regions
            .iter()
            .map(|r| crate::canvas::region_indices(r, &layout))
            .collect();
        let topo = isolated(&[RegionId(0), RegionId(1)]);
        let mask = build_attention_mask(&topo, &indices, &layout);
        // Each region self-attends: 2×2 + 2×2 = 8 edges.
        assert_eq!(mask.n_edges(), 8);
        // No cross edges.
        for &(i, j, _) in &mask.edges {
            let i_in_r0 = i < 2;
            let j_in_r0 = j < 2;
            assert_eq!(i_in_r0, j_in_r0, "cross-region edge leaked");
        }
    }

    #[test]
    fn causal_chain_topology_produces_directed_edges() {
        let layout = CanvasLayout {
            t: 1,
            h: 1,
            w: 3,
            d_model: 8,
            regions: vec![
                RegionSpec::new(
                    "a",
                    CanvasBounds::new(0, 1, 0, 1, 0, 1),
                    1,
                    false,
                    0.0,
                    None,
                    AttentionFnFamily::Cross,
                ),
                RegionSpec::new(
                    "b",
                    CanvasBounds::new(0, 1, 0, 1, 1, 2),
                    1,
                    false,
                    0.0,
                    None,
                    AttentionFnFamily::Cross,
                ),
                RegionSpec::new(
                    "c",
                    CanvasBounds::new(0, 1, 0, 1, 2, 3),
                    1,
                    false,
                    0.0,
                    None,
                    AttentionFnFamily::Cross,
                ),
            ],
        };
        let indices: Vec<Range<usize>> = layout
            .regions
            .iter()
            .map(|r| crate::canvas::region_indices(r, &layout))
            .collect();
        // causal_chain(A→B→C): every region self-attends (3) plus chain[1] queries
        // chain[0] and chain[2] queries chain[1] (2). Each region has size 1,
        // so each present connection contributes exactly 1 edge. Total = 5.
        let topo = causal_chain(&[RegionId(0), RegionId(1), RegionId(2)]);
        let mask = build_attention_mask(&topo, &indices, &layout);
        assert_eq!(mask.n_edges(), 5);
        // Of those 5 edges, the 2 *directed* (non-self) edges are query=B reads
        // key=A, and query=C reads key=B (paper convention: src queries dst;
        // causal_chain emits Connection(current, predecessor)).
        let non_self: Vec<_> = mask
            .edges
            .iter()
            .filter(|(i, j, _)| i != j)
            .copied()
            .collect();
        assert_eq!(
            non_self.len(),
            2,
            "expected exactly 2 non-self edges, got {non_self:?}"
        );
        assert!(non_self.contains(&(1, 0, 1.0)), "B(1) should query A(0)");
        assert!(non_self.contains(&(2, 1, 1.0)), "C(2) should query B(1)");
    }

    #[test]
    fn compile_schema_round_trips() {
        let layout = two_region_layout();
        let schema = CanvasSchema {
            layout: layout.clone(),
            topology: dense(&[RegionId(0), RegionId(1)]),
        };
        let compiled = compile_schema(&schema);
        assert_eq!(compiled.region_indices.len(), 2);
        assert_eq!(compiled.mask.n_positions, 4);
        assert_eq!(compiled.loss_mask.weights.len(), 4);
    }

    #[test]
    fn empty_topology_produces_empty_mask() {
        let layout = two_region_layout();
        let schema = CanvasSchema {
            layout,
            topology: CanvasTopology::new(),
        };
        let compiled = compile_schema(&schema);
        assert_eq!(compiled.mask.n_edges(), 0);
    }
}
