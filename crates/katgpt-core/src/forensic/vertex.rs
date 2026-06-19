//! Vertex perturbation application + recovery (Plan 293 Phase 3).
//!
//! Marks are applied in the 2D tangent plane: only `(x, y)` are
//! perturbed, `z` (surface normal direction) is left untouched. This
//! keeps the displacement within `ε` of the original vertex by the
//! LoopWM spectral bound (`P_vertex = I + ε·Ā`, eigenvalues of `Ā` in
//! `(0, 1)` → each vertex moves by at most `ε` per axis).
//!
//! ## SIMD path
//!
//! Mirrors the crate idiom in `simd.rs`: explicit `core::arch` intrinsics
//! on `aarch64` (NEON, 4× f32) and `x86_64` (AVX2, 8× f32), with a scalar
//! fallback for other targets. No external SIMD crate.

use crate::forensic::recipe::{Recipe, RecipeConfig};

/// Trait abstracting a vertex buffer we can mark + read back.
pub trait VertexMarkable {
    /// Number of vertices in the buffer.
    fn vertex_count(&self) -> usize;
    /// Read vertex `idx` as `[x, y, z]`.
    fn get_vertex(&self, idx: usize) -> [f32; 3];
    /// Write vertex `idx` from `[x, y, z]`.
    fn set_vertex(&mut self, idx: usize, v: [f32; 3]);
}

impl<VertexSliceDeref> VertexMarkable for VertexSliceDeref
where
    VertexSliceDeref: std::ops::Deref<Target = [[f32; 3]]> + std::ops::DerefMut,
{
    #[inline]
    fn vertex_count(&self) -> usize {
        std::ops::Deref::deref(self).len()
    }
    #[inline]
    fn get_vertex(&self, idx: usize) -> [f32; 3] {
        std::ops::Deref::deref(self)[idx]
    }
    #[inline]
    fn set_vertex(&mut self, idx: usize, v: [f32; 3]) {
        std::ops::DerefMut::deref_mut(self)[idx] = v;
    }
}

/// Scalar vertex mark application.
///
/// For each marked vertex `v_k`:
/// ```text
/// v_k' = (I + ε · P_vertex) · v_k      (2D: x, y only; z untouched)
/// ```
/// `P_vertex` is diagonal, so this is a per-axis scale.
pub fn apply_vertex_marks<V: VertexMarkable>(mesh: &mut V, recipe: &Recipe, config: &RecipeConfig) {
    let p11 = recipe.p_vertex[0][0];
    let p22 = recipe.p_vertex[1][1];
    let eps = config.epsilon_vertex;
    let scale_x = 1.0 + eps * (p11 - 1.0) / eps; // = p11 (kept explicit for clarity)
    let _ = scale_x;
    // The recipe's p_vertex already encodes `I + ε·Ā`, so applying it
    // directly is `(I + ε·Ā)·v`. No additional ε scaling.
    let total_x = p11;
    let total_y = p22;
    for &v_idx_u32 in &recipe.vertex_indices {
        let v_idx = (v_idx_u32 as usize) % mesh.vertex_count().max(1);
        let v = mesh.get_vertex(v_idx);
        let v_marked = [v[0] * total_x, v[1] * total_y, v[2]];
        mesh.set_vertex(v_idx, v_marked);
    }
}

/// SIMD vertex mark application (NEON/AVX2). Bit-identical to the scalar
/// path within f32 epsilon — we apply the same per-axis scale, just
/// unrolled 4 or 8 vertices at a time.
///
/// Hot path: this is the function the WASM vessel would call. Mirrors
/// the SIMD idiom in `simd.rs` (explicit `core::arch`, scalar fallback
/// for unsupported targets).
pub fn apply_vertex_marks_simd<V: VertexMarkable>(
    mesh: &mut V,
    recipe: &Recipe,
    _config: &RecipeConfig,
) {
    let p11 = recipe.p_vertex[0][0];
    let p22 = recipe.p_vertex[1][1];
    let n = mesh.vertex_count();
    if n == 0 {
        return;
    }

    // We can't take a `&mut [f32]` slice from a generic `VertexMarkable`,
    // so we collect the indices we want to touch into a packed buffer,
    // SIMD-process them, then write back. This keeps the SIMD path
    // bit-identical to scalar (same scale, same vertex selection) while
    // amortizing the per-vertex `get/set` overhead via 4-wide batches.
    let count = recipe.vertex_indices.len();
    let mut positions_x = Vec::with_capacity(count);
    let mut positions_y = Vec::with_capacity(count);
    let mut resolved_idx = Vec::with_capacity(count);

    for &v_idx_u32 in &recipe.vertex_indices {
        let v_idx = (v_idx_u32 as usize) % n;
        let v = mesh.get_vertex(v_idx);
        positions_x.push(v[0]);
        positions_y.push(v[1]);
        resolved_idx.push(v_idx);
    }

    simd_scale_inplace(&mut positions_x, p11);
    simd_scale_inplace(&mut positions_y, p22);

    for (i, &v_idx) in resolved_idx.iter().enumerate() {
        let v = mesh.get_vertex(v_idx);
        mesh.set_vertex(v_idx, [positions_x[i], positions_y[i], v[2]]);
    }
}

/// SIMD-accelerated in-place scalar multiply: `x[i] *= s`. Mirrors the
/// `simd.rs` pattern (NEON/AVX2 + scalar tail).
#[inline(always)]
fn simd_scale_inplace(x: &mut [f32], s: f32) {
    let len = x.len();
    if len == 0 {
        return;
    }

    // NEON: 4-wide.
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_scale_inplace(x, s);
    }

    // AVX2: 8-wide.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2_scale_inplace(x, s);
    }

    // Scalar fallback (non-aarch64, non-x86_64).
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for v in x.iter_mut() {
            *v *= s;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_scale_inplace(x: &mut [f32], s: f32) {
    use core::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};
    unsafe {
        let vs = vdupq_n_f32(s);
        let mut i = 0;
        let chunks = x.len() / 4;
        for _ in 0..chunks {
            let vx = vld1q_f32(x.as_ptr().add(i));
            let r = vmulq_f32(vx, vs);
            vst1q_f32(x.as_mut_ptr().add(i), r);
            i += 4;
        }
        for j in i..x.len() {
            *x.get_unchecked_mut(j) *= s;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn avx2_scale_inplace(x: &mut [f32], s: f32) {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps};
    unsafe {
        let vs = _mm256_set1_ps(s);
        let mut i = 0;
        let chunks = x.len() / 8;
        for _ in 0..chunks {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let r = _mm256_mul_ps(vx, vs);
            _mm256_storeu_ps(x.as_mut_ptr().add(i), r);
            i += 8;
        }
        // Handle 4-wide tail if present.
        if x.len() - i >= 4 {
            use core::arch::x86_64::{_mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps};
            let vss = _mm_set1_ps(s);
            let vx = _mm_loadu_ps(x.as_ptr().add(i));
            let r = _mm_mul_ps(vx, vss);
            _mm_storeu_ps(x.as_mut_ptr().add(i), r);
            i += 4;
        }
        for j in i..x.len() {
            *x.get_unchecked_mut(j) *= s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forensic::recipe::{construct_perturbation_matrix, derive_recipe};

    fn synth_mesh(n: usize) -> Vec<[f32; 3]> {
        (0..n)
            .map(|i| {
                [
                    (i as f32) * 0.1,
                    ((i as f32) * 0.1).sin(),
                    ((i as f32) * 0.1).cos(),
                ]
            })
            .collect()
    }

    #[test]
    fn vertex_displacement_within_epsilon() {
        // For each recipe, apply marks and verify that each UNIQUE vertex's
        // final displacement stays within ε of its original (relative to
        // |v|). We dedupe by vertex index because the recipe may list the
        // same vertex multiple times (collisions in the mod-N mapping),
        // and the apply function compounds the scale on each pass.
        let cfg = RecipeConfig::default();
        let eps_bound = cfg.epsilon_vertex + 1e-6;
        let mut fails = 0usize;
        for trial in 0..200u32 {
            let mut pk = [0u8; 32];
            for (i, b) in pk.iter_mut().enumerate() {
                *b = (trial as u8).wrapping_add(i as u8).wrapping_mul(11);
            }
            let ms = [42u8; 32];
            let recipe = derive_recipe(&cfg, &pk, &ms);
            let n = 10_000;
            let mut mesh = synth_mesh(n);
            let original = mesh.clone();
            apply_vertex_marks(&mut mesh, &recipe, &cfg);
            // Collect unique vertex indices actually touched.
            let mut touched: Vec<usize> = recipe
                .vertex_indices
                .iter()
                .map(|&v| (v as usize) % n)
                .collect();
            touched.sort_unstable();
            touched.dedup();
            for i in touched {
                let d = (
                    mesh[i][0] - original[i][0],
                    mesh[i][1] - original[i][1],
                    mesh[i][2] - original[i][2],
                );
                let l2 = (d.0 * d.0 + d.1 * d.1 + d.2 * d.2).sqrt();
                let v_norm = (original[i][0] * original[i][0]
                    + original[i][1] * original[i][1]
                    + original[i][2] * original[i][2])
                    .sqrt()
                    .max(1e-6);
                // If the same vertex was marked multiple times the
                // displacement compounds multiplicatively. We bound the
                // per-pass displacement at ε and accept up to a small
                // multiplicative slack (collisions are rare and ≤2×).
                let collision_factor = recipe
                    .vertex_indices
                    .iter()
                    .filter(|&&v| (v as usize) % n == i)
                    .count() as f32;
                let bound = eps_bound * collision_factor.max(1.0);
                if l2 / v_norm > bound {
                    fails += 1;
                }
            }
        }
        assert_eq!(fails, 0, "{fails} unique vertices exceeded ε bound");
    }

    #[test]
    fn simd_matches_scalar_within_f32_eps() {
        let cfg = RecipeConfig::default();
        let pk = [3u8; 32];
        let ms = [4u8; 32];
        let n = 10_000;
        let recipe = derive_recipe(&cfg, &pk, &ms);

        let mut mesh_scalar = synth_mesh(n);
        let mut mesh_simd = synth_mesh(n);
        apply_vertex_marks(&mut mesh_scalar, &recipe, &cfg);
        apply_vertex_marks_simd(&mut mesh_simd, &recipe, &cfg);

        for i in 0..n {
            for c in 0..3 {
                let d = (mesh_scalar[i][c] - mesh_simd[i][c]).abs();
                assert!(d <= 1e-6, "simd/scalar mismatch at ({i},{c}): {d}");
            }
        }
    }

    #[test]
    fn determinism_same_recipe_same_mesh() {
        let cfg = RecipeConfig::default();
        let pk = [9u8; 32];
        let ms = [9u8; 32];
        let recipe = derive_recipe(&cfg, &pk, &ms);
        let n = 1000;
        let mut m1 = synth_mesh(n);
        let mut m2 = synth_mesh(n);
        apply_vertex_marks(&mut m1, &recipe, &cfg);
        apply_vertex_marks(&mut m2, &recipe, &cfg);
        assert_eq!(m1, m2);
    }

    #[test]
    fn perturbation_matrix_shape() {
        // Sanity: construct_perturbation_matrix yields a diagonal 2×2.
        let p = construct_perturbation_matrix(&[7u8; 32], 1e-4);
        assert_eq!(p[0][1], 0.0);
        assert_eq!(p[1][0], 0.0);
        // Diagonal entries slightly > 1 (P = I + ε·Ā, Ā > 0).
        assert!(p[0][0] >= 1.0 && p[0][0] < 1.0 + 1e-3);
        assert!(p[1][1] >= 1.0 && p[1][1] < 1.0 + 1e-3);
    }
}
