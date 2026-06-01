//! Per-head sparsity analysis using Summed-Area Tables for DashAttention.
//!
//! Wires `cache_prune::SummedAreaTable` into DashAttention's sparsity pipeline,
//! providing O(1) intra/inter attention ratio queries per head (or per-segment).
//!
//! Feature-gated with both `dash_attn` and `cache_prune` (Plan 140, T17).

use crate::cache_prune::SummedAreaTable;

/// Sparsity profile for a single attention head (or segment).
///
/// Captures how self-contained a head's attention pattern is:
/// high `contextualization_score` means the head mostly attends within
/// its own segment — a reusable KV-cache candidate.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HeadSparsityInfo {
    /// Total attention this head directs toward positions within its own segment.
    pub intra_attention: f32,
    /// Total attention this head directs toward prefix context (positions before its segment).
    pub inter_attention: f32,
    /// `intra_attention - inter_attention`. Positive ⇒ self-contained.
    pub contextualization_score: f32,
    /// `true` when `contextualization_score > 0` — the head is a reusable segment candidate.
    pub is_self_contained: bool,
}

/// Compute per-head (or per-segment) sparsity profiles from an n×n attention matrix.
///
/// The matrix is divided into `head_count` equal-sized contiguous segments along
/// the diagonal. For each segment `[l, r)`, the SAT is used to compute:
/// - **intra_attention**: sum of attention from rows `[l, r)` to columns `[l, r)`
/// - **inter_attention**: sum of attention from rows `[l, r)` to columns `[0, l)`
///
/// # Panics
///
/// Panics if `attention` is empty, non-square, or not evenly divisible by `head_count`.
pub fn head_sparsity_profile(attention: &[Vec<f32>], head_count: usize) -> Vec<HeadSparsityInfo> {
    let n = attention.len();
    assert!(!attention.is_empty(), "attention matrix must not be empty");
    assert!(
        attention.iter().all(|row| row.len() == n),
        "attention matrix must be square ({}×{})",
        n,
        attention[0].len()
    );
    assert!(
        n.is_multiple_of(head_count),
        "matrix size ({n}) must be evenly divisible by head_count ({head_count})"
    );

    let segment_len = n / head_count;

    // SAT mutates in-place — clone to preserve the caller's data.
    // This is necessary because SummedAreaTable::build requires &mut data for
    // in-place prefix-sum construction. The caller could instead provide a
    // pre-allocated mutable copy to avoid this allocation.
    let mut sat_data = attention.to_vec();
    let sat = SummedAreaTable::build(&mut sat_data);

    (0..head_count)
        .map(|h| {
            let l = h * segment_len;
            let r = l + segment_len;

            let intra = sat.intra_attention(l, r);
            let inter = sat.inter_attention(l, r);
            let score = intra - inter;

            HeadSparsityInfo {
                intra_attention: intra,
                inter_attention: inter,
                contextualization_score: score,
                is_self_contained: score > 0.0,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 16×16 synthetic attention matrix with 4 equal segments (heads).
    ///
    /// Segment layout (each segment is 4 rows × 4 cols):
    ///
    /// - **Seg 0** (rows 0..3): uniform 0.25 everywhere — mixed attention.
    /// - **Seg 1** (rows 4..7): rows 4..7 attend only to cols 4..7 (value 0.25) — self-contained.
    /// - **Seg 2** (rows 8..11): rows 8..11 attend only to cols 0..3 (value 0.25) — fully prefix.
    /// - **Seg 3** (rows 12..15): diagonal-heavy — each row attends to its own col (value 1.0).
    fn synthetic_16x4() -> Vec<Vec<f32>> {
        let n = 16usize;
        let seg = 4usize;
        let mut m = vec![vec![0.0f32; n]; n];

        // Seg 0: uniform 0.25 in rows 0..3 (every column)
        for row in &mut m[..seg] {
            for val in row.iter_mut() {
                *val = 0.25;
            }
        }

        // Seg 1: rows 4..7 attend only to cols 4..7 (self-contained block)
        for row in &mut m[seg..2 * seg] {
            for val in row[seg..2 * seg].iter_mut() {
                *val = 0.25;
            }
        }

        // Seg 2: rows 8..11 attend only to cols 0..3 (fully prefix)
        for row in &mut m[2 * seg..3 * seg] {
            for val in row[..seg].iter_mut() {
                *val = 0.25;
            }
        }

        // Seg 3: diagonal-heavy — each row in 12..15 has 1.0 at its own column, 0 elsewhere
        for (i, row) in m.iter_mut().enumerate().take(4 * seg).skip(3 * seg) {
            row[i] = 1.0;
        }

        m
    }

    #[test]
    fn test_head_sparsity_profile_16x4() {
        let attn = synthetic_16x4();
        let profiles = head_sparsity_profile(&attn, 4);

        assert_eq!(profiles.len(), 4);

        // --- Seg 0 (rows 0..3, cols 0..3 intra; cols 0..15 total) ---
        // intra: rows 0..3 × cols 0..3 → 4×4 cells × 0.25 = 4.0
        // inter: l=0 so inter is always 0.0 (no prefix)
        // score: 4.0 - 0.0 = 4.0 → self-contained
        assert!(
            (profiles[0].intra_attention - 4.0).abs() < 1e-4,
            "seg0 intra: got {}",
            profiles[0].intra_attention
        );
        assert!(
            (profiles[0].inter_attention - 0.0).abs() < 1e-4,
            "seg0 inter: got {}",
            profiles[0].inter_attention
        );
        assert!(
            (profiles[0].contextualization_score - 4.0).abs() < 1e-4,
            "seg0 score: got {}",
            profiles[0].contextualization_score
        );
        assert!(
            profiles[0].is_self_contained,
            "seg0 should be self-contained"
        );

        // --- Seg 1 (rows 4..7, cols 4..7 intra; cols 0..3 inter) ---
        // intra: rows 4..7 × cols 4..7 → 4×4 × 0.25 = 4.0
        // inter: rows 4..7 × cols 0..3 → 4×4 × 0.0 = 0.0
        // score: 4.0 → self-contained
        assert!(
            (profiles[1].intra_attention - 4.0).abs() < 1e-4,
            "seg1 intra: got {}",
            profiles[1].intra_attention
        );
        assert!(
            (profiles[1].inter_attention - 0.0).abs() < 1e-4,
            "seg1 inter: got {}",
            profiles[1].inter_attention
        );
        assert!(
            profiles[1].is_self_contained,
            "seg1 should be self-contained"
        );

        // --- Seg 2 (rows 8..11, cols 8..11 intra; cols 0..7 inter) ---
        // intra: rows 8..11 × cols 8..11 → all zeros = 0.0
        // inter: rows 8..11 × cols 0..7 = rows 8..11 × cols 0..3 (= 4×4×0.25) + cols 4..7 (0) = 4.0
        // score: 0.0 - 4.0 = -4.0 → NOT self-contained
        assert!(
            (profiles[2].intra_attention - 0.0).abs() < 1e-4,
            "seg2 intra: got {}",
            profiles[2].intra_attention
        );
        assert!(
            (profiles[2].inter_attention - 4.0).abs() < 1e-4,
            "seg2 inter: got {}",
            profiles[2].inter_attention
        );
        assert!(
            (profiles[2].contextualization_score - (-4.0)).abs() < 1e-4,
            "seg2 score: got {}",
            profiles[2].contextualization_score
        );
        assert!(
            !profiles[2].is_self_contained,
            "seg2 should NOT be self-contained"
        );

        // --- Seg 3 (rows 12..15, cols 12..15 intra; cols 0..11 inter) ---
        // intra: diagonal only → m[12][12] + m[13][13] + m[14][14] + m[15][15] = 4 × 1.0 = 4.0
        // inter: rows 12..15 × cols 0..11 → all zeros = 0.0
        // score: 4.0 → self-contained
        assert!(
            (profiles[3].intra_attention - 4.0).abs() < 1e-4,
            "seg3 intra: got {}",
            profiles[3].intra_attention
        );
        assert!(
            (profiles[3].inter_attention - 0.0).abs() < 1e-4,
            "seg3 inter: got {}",
            profiles[3].inter_attention
        );
        assert!(
            (profiles[3].contextualization_score - 4.0).abs() < 1e-4,
            "seg3 score: got {}",
            profiles[3].contextualization_score
        );
        assert!(
            profiles[3].is_self_contained,
            "seg3 should be self-contained"
        );
    }

    #[test]
    #[should_panic(expected = "must be evenly divisible")]
    fn test_uneven_head_count_panics() {
        let attn = vec![vec![1.0f32; 4]; 4];
        // 4 is not divisible by 3
        let _ = head_sparsity_profile(&attn, 3);
    }

    #[test]
    fn test_single_head_full_matrix() {
        // 4×4 all-ones → intra = total = 16, inter = 0 (l=0)
        let attn = vec![vec![1.0f32; 4]; 4];
        let profiles = head_sparsity_profile(&attn, 1);

        assert_eq!(profiles.len(), 1);
        assert!((profiles[0].intra_attention - 16.0).abs() < 1e-4);
        assert!((profiles[0].inter_attention - 0.0).abs() < 1e-4);
        assert!(profiles[0].is_self_contained);
    }
}
