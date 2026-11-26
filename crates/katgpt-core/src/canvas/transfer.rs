//! Semantic-type compatibility via frozen embeddings (Plan 419 Phase 4).
//!
//! [`transfer_distance`] is a modelless routing scalar: given two
//! [`SemanticType`]s with frozen embeddings, it returns `1 - cosine(a, b)`.
//! Identical types → 0; orthogonal types → 1; opposite types → 2.
//!
//! This is the schema-ABI compatibility check from paper §2.4 Table 1: two
//! regions with compatible types can exchange latents without retraining.
//! Modelless (pure cosine, zero allocation, zero training).

use super::types::{CanvasLayout, CanvasSchema, RegionId, SemanticType};

/// Compute `1.0 - cosine(a, b)` between two semantic types' frozen embeddings.
///
/// Returns:
/// - `0.0` for identical (parallel) embeddings.
/// - `1.0` for orthogonal embeddings.
/// - `2.0` for anti-parallel embeddings.
///
/// Returns `1.0` (maximally distant) if either embedding is the zero vector
/// (cosine is undefined; we treat zero-norm as "no information" → maximally
/// distant, the conservative routing choice).
///
/// # Allocation
///
/// Zero. Operates on the fixed-size `[f32; SEMANTIC_EMBED_DIM]` slices in place.
#[inline]
pub fn transfer_distance(a: &SemanticType, b: &SemanticType) -> f32 {
    cosine_distance(&a.frozen_embedding, &b.frozen_embedding)
}

/// `1 - cosine` on two slices. Internal helper shared by [`transfer_distance`]
/// and [`compatible_regions`].
#[inline]
pub(crate) fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    // Single-pass dot + squared norms accumulated in **f64** — no allocation,
    // and overflow-safe for f32 inputs (a coordinate of 1e20 squares to 1e40,
    // which overflows f32::MAX ≈ 3.4e38 to inf, but is exact in f64). This keeps
    // the cosine well-defined for large-magnitude parallel embeddings, which a
    // pure-f32 reduction would silently turn into inf/inf = NaN.
    let mut dot: f64 = 0.0;
    let mut na: f64 = 0.0;
    let mut nb: f64 = 0.0;
    let len = a.len().min(b.len());
    for k in 0..len {
        let x = a[k] as f64;
        let y = b[k] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 || !denom.is_finite() {
        // Zero-norm → cosine undefined → maximally distant (conservative).
        // (denom is finite for all finite f32 inputs under f64 accumulation, so
        // the non-finite branch is a defensive no-op kept for robustness.)
        return 1.0;
    }
    let cos = dot / denom;
    // Guard against tiny numerical overshoot beyond [-1, 1]. NaN is unreachable
    // here for finite inputs (guarded above), but clamp defensively anyway.
    if cos.is_nan() {
        return 1.0;
    }
    let cos = cos.clamp(-1.0, 1.0);
    (1.0 - cos) as f32
}

/// Returns all region pairs `(a, b)` with `a < b` whose semantic types are
/// both present and whose [`transfer_distance`] is `≤ max_distance`.
///
/// This is the schema-ABI compatibility check (paper §2.4 Table 1): pairs
/// below the threshold can exchange latents without retraining. Only the upper
/// triangle is returned (unordered pairs), since distance is symmetric.
///
/// # Allocation
///
/// Allocates the result `Vec`. The scan itself is O(n_regions²) with no
/// per-pair allocation.
pub fn compatible_regions(schema: &CanvasSchema, max_distance: f32) -> Vec<(RegionId, RegionId)> {
    compatible_regions_in_layout(&schema.layout, max_distance)
}

/// Layout-only variant of [`compatible_regions`] (the topology is irrelevant
/// to type compatibility). Exposed for callers that hold a layout directly.
pub fn compatible_regions_in_layout(
    layout: &CanvasLayout,
    max_distance: f32,
) -> Vec<(RegionId, RegionId)> {
    let n = layout.regions.len();
    // Upper bound on pairs: n*(n-1)/2. Reserve once.
    let max_pairs = n.saturating_mul(n) / 2;
    let mut out: Vec<(RegionId, RegionId)> = Vec::with_capacity(max_pairs);
    for i in 0..n {
        let Some(t_i) = layout.regions[i].semantic_type.as_ref() else {
            continue;
        };
        let t_j_iter = layout.regions[(i + 1)..n].iter();
        for (j_off, region_j) in t_j_iter.enumerate() {
            let j = i + 1 + j_off;
            let Some(t_j) = region_j.semantic_type.as_ref() else {
                continue;
            };
            if transfer_distance(t_i, t_j) <= max_distance {
                out.push((RegionId(i), RegionId(j)));
            }
        }
    }
    out.shrink_to_fit();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::SemanticType;
    use crate::canvas::types::SEMANTIC_EMBED_DIM;

    #[test]
    fn identical_embeddings_have_zero_distance() {
        let a = SemanticType::basis("camera", 0);
        // Same basis axis → identical.
        let b = SemanticType::basis("cam2", 0);
        assert!((transfer_distance(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_embeddings_have_unit_distance() {
        let a = SemanticType::basis("camera", 0);
        let b = SemanticType::basis("joints", 1);
        assert!((transfer_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn antiparallel_embeddings_have_distance_two() {
        let mut emb = [0.0_f32; SEMANTIC_EMBED_DIM];
        emb[0] = 1.0;
        let a = SemanticType::new("pos", emb);
        emb[0] = -1.0;
        let b = SemanticType::new("neg", emb);
        assert!((transfer_distance(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_is_maximally_distant() {
        let zero = SemanticType::new("zero", [0.0_f32; SEMANTIC_EMBED_DIM]);
        let a = SemanticType::basis("x", 0);
        // zero vs anything → 1.0 (conservative).
        assert!((transfer_distance(&zero, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn non_unit_embeddings_normalize_correctly() {
        // [2,0,...] and [1,0,...] are parallel → distance 0 after normalization.
        let mut e1 = [0.0_f32; SEMANTIC_EMBED_DIM];
        e1[0] = 2.0;
        let mut e2 = [0.0_f32; SEMANTIC_EMBED_DIM];
        e2[0] = 1.0;
        let a = SemanticType::new("a", e1);
        let b = SemanticType::new("b", e2);
        assert!(transfer_distance(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_distance_parallel_representable_is_zero() {
        // Two parallel vectors with representable norm: [3,4,0,...] has norm 5.
        // dot = 25, na = nb = 25, denom = 25, cos = 1.0 exactly → distance 0.
        // This verifies the clamp does not spuriously push a clean 1.0 below 1.0.
        let mut e = [0.0_f32; SEMANTIC_EMBED_DIM];
        e[0] = 3.0;
        e[1] = 4.0;
        let a = SemanticType::new("a", e);
        let b = SemanticType::new("b", e);
        assert!(transfer_distance(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_distance_overflow_returns_exact_zero_via_f64() {
        // [1e20, 0, ...] squares to 1e40, which overflows f32::MAX ≈ 3.4e38 to
        // inf — so a pure-f32 reduction yields inf/inf = NaN. The f64
        // accumulation path keeps the cosine well-defined: the two vectors are
        // parallel, so the true distance is exactly 0. We return that exact 0,
        // NOT a conservative 1.0 (the vectors are identical-direction, not
        // maximally distant).
        let mut e = [0.0_f32; SEMANTIC_EMBED_DIM];
        e[0] = 1e20;
        let a = SemanticType::new("a", e);
        let b = SemanticType::new("b", e);
        let d = transfer_distance(&a, &b);
        assert!(
            d.abs() < 1e-3,
            "parallel large-magnitude vectors should be distance ~0, got {d}"
        );
        assert!(!d.is_nan(), "distance must never be NaN");
    }
}
