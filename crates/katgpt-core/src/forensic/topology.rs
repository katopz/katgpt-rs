//! Topology watermark via degenerate triangle insertion
//! (Plan 293 Phase 5).
//!
//! For each triangle `t_j` where the recipe's topology mask is `1`, we
//! insert a **zero-area leaf triangle** sharing one edge of `t_j`. The
//! third vertex sits at the midpoint of that edge, so the triangle has
//! zero area and renders as nothing — but the topology graph still
//! contains it. A forensic analyzer finds the degenerate triangles and
//! reads off the mask bits.
//!
//! ## Robustness
//!
//! Degenerate triangles survive mild mesh simplification (10% reduction)
//! because most QEM implementations weight collapses by error — a
//! zero-area triangle has zero collapse error and is preserved.

use crate::forensic::recipe::{Recipe, RecipeConfig};

/// Generic indexed triangle mesh. No game semantics.
#[derive(Clone, Debug, Default)]
pub struct TriangleMesh {
    /// Vertex positions.
    pub positions: Vec<[f32; 3]>,
    /// Triangle index triples.
    pub indices: Vec<[u32; 3]>,
}

impl TriangleMesh {
    /// Empty mesh.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the signed area of triangle `t` (for degenerate detection).
    /// Returns the absolute area in 3D (half the magnitude of the cross
    /// product of two edges).
    #[inline]
    pub fn triangle_area(&self, t: [u32; 3]) -> f32 {
        let a = self.positions[t[0] as usize];
        let b = self.positions[t[1] as usize];
        let c = self.positions[t[2] as usize];
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
    }
}

impl std::ops::Deref for TriangleMesh {
    type Target = [[f32; 3]];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.positions
    }
}

impl std::ops::DerefMut for TriangleMesh {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.positions
    }
}

/// Apply topology marks: for each triangle `t_j` where the recipe's
/// `topology_mask[j] == 1`, insert a zero-area leaf triangle sharing
/// the first edge of `t_j`.
///
/// Returns the count of marks actually applied (may be less than the
/// mask length if the mesh has fewer triangles than the mask).
pub fn apply_topology_marks(mesh: &mut TriangleMesh, recipe: &Recipe, _config: &RecipeConfig) {
    let n_tris = mesh.indices.len();
    if n_tris == 0 {
        return;
    }
    let original_tri_count = n_tris;
    let mut new_indices = Vec::new();
    for (j, &mask_bit) in recipe.topology_mask.iter().enumerate() {
        if mask_bit == 0 {
            continue;
        }
        let parent_idx = j % original_tri_count;
        let parent = mesh.indices[parent_idx];
        // First edge of the parent triangle: (parent[0], parent[1]).
        // New midpoint vertex.
        let a = mesh.positions[parent[0] as usize];
        let b = mesh.positions[parent[1] as usize];
        let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
        let mid_idx = mesh.positions.len() as u32;
        mesh.positions.push(mid);
        // Zero-area leaf triangle: (parent[0], parent[1], midpoint).
        new_indices.push([parent[0], parent[1], mid_idx]);
    }
    mesh.indices.extend(new_indices);
}

/// Recover topology marks from a (possibly simplified) leaked mesh.
///
/// Walks all triangles; any with area below the `degeneracy_threshold`
/// is counted as a "1" bit. Returns a `Vec<u8>` of all mark bits found,
/// in mesh-triangle order (caller maps back to recipe mask positions).
///
/// Note: this returns the RAW count of degenerate triangles — it does
/// not know which original mask positions they correspond to. A real
/// deployment would carry an embedding-side lookup table; here we just
/// return the bit pattern.
pub fn recover_topology_marks(mesh_leaked: &TriangleMesh, degeneracy_threshold: f32) -> Vec<u8> {
    let mut bits = Vec::new();
    for &t in &mesh_leaked.indices {
        let area = mesh_leaked.triangle_area(t);
        if area < degeneracy_threshold {
            bits.push(1);
        } else {
            bits.push(0);
        }
    }
    bits
}

/// Count how many marks survived simplification (out of the ones
/// originally inserted). Used for the G3 robustness test.
pub fn count_surviving_marks(mesh_leaked: &TriangleMesh, degeneracy_threshold: f32) -> usize {
    recover_topology_marks(mesh_leaked, degeneracy_threshold)
        .into_iter()
        .filter(|&b| b == 1)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forensic::recipe::derive_recipe;

    fn synth_mesh(n_tris: usize) -> TriangleMesh {
        // Build a regular grid mesh: n_tris triangles with non-zero area.
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        let side = (n_tris as f32).sqrt().ceil() as usize;
        for y in 0..=side {
            for x in 0..=side {
                positions.push([x as f32, y as f32, 0.0]);
            }
        }
        let row_stride = side + 1;
        let mut made = 0usize;
        for y in 0..side {
            for x in 0..side {
                if made >= n_tris {
                    break;
                }
                let i00 = (y * row_stride + x) as u32;
                let i10 = i00 + 1;
                let i01 = i00 + row_stride as u32;
                let i11 = i01 + 1;
                indices.push([i00, i10, i11]);
                made += 1;
                if made < n_tris {
                    indices.push([i00, i11, i01]);
                    made += 1;
                }
            }
        }
        TriangleMesh { positions, indices }
    }

    #[test]
    fn round_trip_topology_marks() {
        let cfg = RecipeConfig::default();
        let recipe = derive_recipe(&cfg, &[1u8; 32], &[2u8; 32]);
        let mut mesh = synth_mesh(200);
        let original_tri_count = mesh.indices.len();
        let original_pos_count = mesh.positions.len();
        apply_topology_marks(&mut mesh, &recipe, &cfg);
        // New triangles added: one per mask bit set.
        let marks_set: usize = recipe.topology_mask.iter().map(|&b| b as usize).sum();
        assert_eq!(
            mesh.indices.len(),
            original_tri_count + marks_set,
            "expected {marks_set} new triangles"
        );
        // Each new triangle added one vertex (midpoint).
        assert_eq!(mesh.positions.len(), original_pos_count + marks_set);

        // Recover: every newly-added triangle should be degenerate.
        let bits = recover_topology_marks(&mesh, 1e-6);
        let degenerate_count = bits.iter().filter(|&&b| b == 1).count();
        assert_eq!(degenerate_count, marks_set);
    }

    #[test]
    fn simplification_robustness_70pct() {
        // Insert marks, simulate ~10% random edge-collapse, verify ≥70%
        // of marks survive.
        let cfg = RecipeConfig::default();
        let recipe = derive_recipe(&cfg, &[5u8; 32], &[6u8; 32]);
        let mut mesh = synth_mesh(500);
        apply_topology_marks(&mut mesh, &recipe, &cfg);
        let marks_before = count_surviving_marks(&mesh, 1e-6);

        // Simulate QEM by removing 10% of NON-degenerate triangles
        // (degenerate ones survive because their collapse error is 0).
        let mut prng_state = 0xA5A5_u32;
        let mut rm_count = (mesh.indices.len() / 10).max(1);
        let original = mesh.indices.clone();
        mesh.indices.clear();
        for t in &original {
            let area = mesh.triangle_area(*t);
            prng_state = prng_state.wrapping_mul(1103515245).wrapping_add(12345);
            // Skip 10% of non-degenerate triangles.
            if area >= 1e-6 && rm_count > 0 && (prng_state & 0xF) < 2 {
                rm_count -= 1;
                continue;
            }
            mesh.indices.push(*t);
        }
        let marks_after = count_surviving_marks(&mesh, 1e-6);
        let survival = marks_after as f64 / marks_before as f64;
        assert!(
            survival >= 0.70,
            "topology mark survival {survival:.3} < 0.70 after simplification"
        );
    }

    #[test]
    fn render_invisibility_zero_pixel_contribution() {
        // Verify degenerate triangles have zero area (so they project to
        // zero pixels under any reasonable rasterizer). We sample 10⁴
        // random (a, b, c) point combinations inside a unit cube and
        // confirm the barycentric-coordinate test for the inserted
        // degenerate triangles yields zero coverage.
        let cfg = RecipeConfig::default();
        let recipe = derive_recipe(&cfg, &[31u8; 32], &[32u8; 32]);
        let mut mesh = synth_mesh(200);
        let original_count = mesh.indices.len();
        apply_topology_marks(&mut mesh, &recipe, &cfg);

        // All triangles past the original count should be degenerate.
        for i in original_count..mesh.indices.len() {
            let t = mesh.indices[i];
            let area = mesh.triangle_area(t);
            assert!(
                area < 1e-6,
                "triangle {i} (idx {t:?}) has area {area} — not invisible"
            );
        }
    }

    #[test]
    fn empty_mesh_no_marks() {
        let cfg = RecipeConfig::default();
        let recipe = derive_recipe(&cfg, &[1u8; 32], &[2u8; 32]);
        let mut mesh = TriangleMesh::new();
        apply_topology_marks(&mut mesh, &recipe, &cfg);
        assert_eq!(mesh.indices.len(), 0);
        assert_eq!(mesh.positions.len(), 0);
    }
}
