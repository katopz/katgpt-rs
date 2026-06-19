//! Forensic recovery pipeline + recipient attribution
//! (Plan 293 Phase 6).
//!
//! Given a leaked asset and the unmarked reference, recover the codeword
//! that was embedded, then attribute it to a specific recipient via a
//! `RecipientRegistry` lookup. Confidence is sigmoid-gated per
//! AGENTS.md (no softmax).

use crate::forensic::recipe::{Recipe, RecipeConfig};
use crate::forensic::texture::{Dct8x8Block, recover_dct_marks};
use crate::forensic::topology::{TriangleMesh, count_surviving_marks};

/// A leaked asset (mesh + texture blocks) we want to attribute.
#[derive(Clone, Debug)]
pub struct LeakedContent {
    /// Mesh with embedded vertex + topology marks.
    pub mesh: TriangleMesh,
    /// Texture DCT blocks with embedded DCT marks.
    pub texture_blocks: Vec<Dct8x8Block>,
}

/// What evidence was matched during recovery — for audit trails.
#[derive(Clone, Debug, Default)]
pub struct RecoveryEvidence {
    /// Recovered codeword bits.
    pub codeword: Vec<u8>,
    /// Number of vertex bits successfully recovered (sign-of-displacement).
    pub vertex_bits: usize,
    /// Number of DCT bits successfully recovered.
    pub dct_bits: usize,
    /// Number of topology bits successfully recovered.
    pub topology_bits: usize,
    /// Tardos accusation statistic S_j.
    pub accusation_score: f64,
    /// Tardos accusation threshold Z.
    pub accusation_threshold: f64,
}

/// Attribution result.
#[derive(Clone, Debug)]
pub struct RecoveryResult {
    /// Attributed recipient pubkey hash.
    pub recipient_pubkey: [u8; 32],
    /// Sigmoid-gated confidence in [0, 1]. AGENTS.md: sigmoid not softmax.
    pub confidence: f32,
    /// Evidence trail.
    pub evidence: RecoveryEvidence,
}

/// Recover `P_vertex` via closed-form 1D least squares (diagonal case).
///
/// Given `v_leak = (I + ε·Ā) · v_ref` with `P_vertex` diagonal, the
/// least-squares fit per axis is:
///
/// ```text
/// scale_x = Σ v_leak_x · v_ref_x / Σ v_ref_x²
/// scale_y = Σ v_leak_y · v_ref_y / Σ v_ref_y²
/// ```
///
/// The recovered matrix is `[[scale_x, 0], [0, scale_y]]`. This is the
/// closed-form minimum of `‖V_leak − D · V_ref‖²` for diagonal `D`.
pub fn recover_p_vertex(
    mesh_leaked: &TriangleMesh,
    mesh_reference: &TriangleMesh,
    vertex_indices: &[u32],
) -> [[f32; 2]; 2] {
    let n = mesh_reference.positions.len();
    let mut sx_num = 0.0f64;
    let mut sx_den = 0.0f64;
    let mut sy_num = 0.0f64;
    let mut sy_den = 0.0f64;
    for &v_idx_u32 in vertex_indices {
        let i = (v_idx_u32 as usize) % n.max(1);
        let leak = mesh_leaked.positions[i];
        let reference = mesh_reference.positions[i];
        sx_num += (leak[0] as f64) * (reference[0] as f64);
        sx_den += (reference[0] as f64) * (reference[0] as f64);
        sy_num += (leak[1] as f64) * (reference[1] as f64);
        sy_den += (reference[1] as f64) * (reference[1] as f64);
    }
    let scale_x = if sx_den > 1e-12 {
        (sx_num / sx_den) as f32
    } else {
        1.0
    };
    let scale_y = if sy_den > 1e-12 {
        (sy_num / sy_den) as f32
    } else {
        1.0
    };
    [[scale_x, 0.0], [0.0, scale_y]]
}

/// Recover the **DCT-channel codeword** from a leaked asset.
///
/// The DCT channel is the primary recoverable codeword: each marked
/// coefficient carries one bit (sign of `(leaked − reference)`). The
/// vertex channel contributes a single global bit (sign of the
/// recovered `P_vertex`), and topology contributes a count — both are
/// confirmation signals, not part of the lookup key. This matches the
/// plan's intent: DCT marks are the high-bandwidth, recompression-robust
/// carrier; vertex + topology are auxiliary.
///
/// Returns a `Vec<u8>` of length `recipe.dct_indices.len()`.
pub fn recover_codeword(
    leaked: &LeakedContent,
    reference: &LeakedContent,
    recipe: &Recipe,
    _degeneracy_threshold: f32,
) -> Vec<u8> {
    recover_dct_marks(&leaked.texture_blocks, &reference.texture_blocks, recipe)
}

/// Extract the auxiliary evidence (P_vertex sign bit + topology mark
/// count) for the `RecoveryEvidence` trail.
pub fn recover_auxiliary_evidence(
    leaked: &LeakedContent,
    reference: &LeakedContent,
    recipe: &Recipe,
    degeneracy_threshold: f32,
) -> (u8, usize) {
    let p_recovered =
        recover_p_vertex(&leaked.mesh, &reference.mesh, &recipe.vertex_indices);
    let p_sign_bit: u8 = if p_recovered[0][0] > 1.0 { 1 } else { 0 };
    let topo_marks = count_surviving_marks(&leaked.mesh, degeneracy_threshold);
    (p_sign_bit, topo_marks)
}

/// Recipient registry abstraction. A real deployment (riir-ai Plan 322)
/// would implement this against an NFT registry; for the open primitive
/// we leave it as a trait so consumers can wire their own store.
pub trait RecipientRegistry {
    /// Look up the recipient pubkey hash for a given codeword, if any.
    fn lookup_by_codeword(&self, codeword: &[u8]) -> Option<[u8; 32]>;
    /// Number of registered recipients.
    fn n_recipients(&self) -> usize;
}

/// Numerically stable sigmoid: `σ(x) = 1 / (1 + e^(-x))`.
/// AGENTS.md rule: use sigmoid, not softmax.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Attribute a leaked asset to a recipient.
///
/// Pipeline:
/// 1. Recover the DCT-channel codeword (the primary carrier).
/// 2. Look up the codeword in the registry.
/// 3. If found, compute the Tardos accusation score against the matched
///    recipient's full codeword, and gate the confidence via
///    `σ(accusation_score − threshold)` per AGENTS.md.
///
/// Returns `None` if the codeword does not match any registered
/// recipient.
pub fn attribute(
    leaked: &LeakedContent,
    reference: &LeakedContent,
    recipe: &Recipe,
    registry: &dyn RecipientRegistry,
    _config: &RecipeConfig,
    degeneracy_threshold: f32,
) -> Option<RecoveryResult> {
    let codeword = recover_codeword(leaked, reference, recipe, degeneracy_threshold);
    let recipient_pubkey = registry.lookup_by_codeword(&codeword)?;

    // Auxiliary evidence (P_vertex sign + topology mark count).
    let (p_sign_bit, topo_marks) =
        recover_auxiliary_evidence(leaked, reference, recipe, degeneracy_threshold);

    // Tardos accusation: score of the recovered codeword against the
    // matched recipient's slot. The recovered codeword is the DCT
    // channel, which lives at Tardos positions
    // `[v_offset .. v_offset+L_dct]` in the full codeword. We use
    // `accusation_sum_offset` to compare at the correct positions.
    // Per Plan 293 T6.4, confidence is σ(accusation_sum) — the raw
    // sigmoid of the accusation statistic (AGENTS.md: sigmoid not
    // softmax).
    let v_offset = recipe.vertex_indices.len();
    let accusation_score =
        recipe.codebook.accusation_sum_offset(&codeword, recipe.recipient_idx, v_offset);
    let threshold = recipe.codebook.accusation_threshold_for_len(codeword.len());
    let confidence = sigmoid(accusation_score) as f32;

    Some(RecoveryResult {
        recipient_pubkey,
        confidence,
        evidence: RecoveryEvidence {
            codeword,
            vertex_bits: p_sign_bit as usize,
            dct_bits: recipe.dct_indices.len(),
            topology_bits: topo_marks,
            accusation_score,
            accusation_threshold: threshold,
        },
    })
}

// ─── In-memory test registry ────────────────────────────────────────────

/// A simple in-memory registry mapping recipient pubkeys to their
/// codewords (computed from the recipe). Used by the open-primitive
/// tests and the demo.
pub struct InMemoryRegistry {
    entries: Vec<([u8; 32], Vec<u8>)>,
}

impl InMemoryRegistry {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a recipient with their precomputed codeword.
    pub fn register(&mut self, pubkey: [u8; 32], codeword: Vec<u8>) {
        self.entries.push((pubkey, codeword));
    }

    /// Register by deriving the DCT-channel codeword from the recipe.
    /// Only the DCT-channel bits are stored (the recoverable subset);
    /// vertex and topology channels are auxiliary evidence, not lookup
    /// keys. Bits at duplicate resolved positions reflect the FIRST
    /// occurrence (matching `apply_dct_marks` / `recover_dct_marks`).
    /// `n_blocks_hint` should match the texture block count the recipe
    /// will be applied against.
    pub fn register_recipe(&mut self, pubkey: [u8; 32], recipe: &Recipe, n_blocks_hint: usize) {
        let v_offset = recipe.vertex_indices.len();
        let n_blocks = n_blocks_hint.max(1);
        // For each resolved (block, coef) position, the canonical bit
        // is the codeword bit of the FIRST dct_indices entry mapping to
        // that position (same convention as apply_dct_marks). All
        // later entries hitting the same position read this same bit.
        let mut first_bit: std::collections::HashMap<(usize, u8), u8> =
            std::collections::HashMap::new();
        for (k, &(block_idx_u32, coef_idx)) in recipe.dct_indices.iter().enumerate() {
            let block_idx = (block_idx_u32 as usize) % n_blocks;
            first_bit
                .entry((block_idx, coef_idx))
                .or_insert(recipe.codeword[v_offset + k]);
        }
        let dct_codeword: Vec<u8> = recipe
            .dct_indices
            .iter()
            .map(|&(block_idx_u32, coef_idx)| {
                let block_idx = (block_idx_u32 as usize) % n_blocks;
                *first_bit.get(&(block_idx, coef_idx)).unwrap_or(&0)
            })
            .collect();
        self.entries.push((pubkey, dct_codeword));
    }
}

impl Default for InMemoryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RecipientRegistry for InMemoryRegistry {
    fn lookup_by_codeword(&self, codeword: &[u8]) -> Option<[u8; 32]> {
        // Exact-match lookup. For forensic-grade deployment this would
        // be a fuzzy Tardos nearest-neighbor search, but exact match
        // suffices for the open primitive's tests and demo.
        for (pk, cw) in &self.entries {
            if cw == codeword {
                return Some(*pk);
            }
        }
        None
    }

    fn n_recipients(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forensic::recipe::derive_recipe;
    use crate::forensic::texture::{Dct8x8Block, apply_dct_marks};
    use crate::forensic::topology::{TriangleMesh, apply_topology_marks};
    use crate::forensic::vertex::{apply_vertex_marks, apply_vertex_marks_simd};

    fn synth_mesh(n_verts: usize, n_tris: usize) -> TriangleMesh {
        let side = (n_verts as f32).sqrt().ceil() as usize;
        let mut positions = Vec::new();
        for y in 0..=side {
            for x in 0..=side {
                positions.push([x as f32, y as f32, 0.0]);
                if positions.len() >= n_verts {
                    break;
                }
            }
            if positions.len() >= n_verts {
                break;
            }
        }
        let mut indices = Vec::new();
        let row_stride = (side + 1) as u32;
        let mut made = 0usize;
        for y in 0..side {
            for x in 0..side {
                if made >= n_tris {
                    break;
                }
                let i00 = (y as u32) * row_stride + x as u32;
                indices.push([i00, i00 + 1, i00 + row_stride]);
                made += 1;
            }
        }
        TriangleMesh { positions, indices }
    }

    fn synth_texture(n_blocks: usize) -> Vec<Dct8x8Block> {
        (0..n_blocks)
            .map(|i| {
                let mut d = [0.0f32; 64];
                for j in 0..64 {
                    d[j] = ((i + j) as f32) * 0.5;
                }
                Dct8x8Block { data: d }
            })
            .collect()
    }

    fn end_to_end(with_simd: bool) -> (RecoveryResult, [u8; 32]) {
        let cfg = RecipeConfig::default();
        let pubkey = [7u8; 32];
        let master = [9u8; 32];
        let recipe = derive_recipe(&cfg, &pubkey, &master);

        let n_verts = 1000;
        let n_tris = 500;
        let n_blocks = 200;

        let mut mesh = synth_mesh(n_verts, n_tris);
        let mesh_reference = mesh.clone();
        let mut texture = synth_texture(n_blocks);
        let texture_reference = texture.clone();

        if with_simd {
            apply_vertex_marks_simd(&mut mesh, &recipe, &cfg);
        } else {
            apply_vertex_marks(&mut mesh, &recipe, &cfg);
        }
        apply_dct_marks(&mut texture, &recipe, &cfg);
        apply_topology_marks(&mut mesh, &recipe, &cfg);

        let leaked = LeakedContent {
            mesh,
            texture_blocks: texture,
        };
        let reference = LeakedContent {
            mesh: mesh_reference,
            texture_blocks: texture_reference,
        };

        let mut registry = InMemoryRegistry::new();
        registry.register_recipe(pubkey, &recipe, n_blocks);

        let result = attribute(
            &leaked,
            &reference,
            &recipe,
            &registry,
            &cfg,
            1e-6,
        )
        .expect("attribution must succeed");
        (result, pubkey)
    }

    #[test]
    fn end_to_end_correct_recipient_high_confidence() {
        let (result, expected_pk) = end_to_end(false);
        assert_eq!(result.recipient_pubkey, expected_pk);
        assert!(
            result.confidence > 0.5,
            "confidence {} not > 0.5",
            result.confidence
        );
    }

    #[test]
    fn end_to_end_simd_path() {
        let (result, expected_pk) = end_to_end(true);
        assert_eq!(result.recipient_pubkey, expected_pk);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn wrong_recipient_low_confidence() {
        // Register a DIFFERENT recipient than the one whose recipe was
        // applied. The lookup should either fail (codeword doesn't
        // match) or return a confidence < 0.5.
        let cfg = RecipeConfig::default();
        let real_pk = [1u8; 32];
        let other_pk = [2u8; 32];
        let master = [9u8; 32];

        let recipe = derive_recipe(&cfg, &real_pk, &master);

        let n_verts = 1000;
        let n_tris = 500;
        let n_blocks = 200;

        let mut mesh = synth_mesh(n_verts, n_tris);
        let mesh_reference = mesh.clone();
        let mut texture = synth_texture(n_blocks);
        let texture_reference = texture.clone();

        apply_vertex_marks(&mut mesh, &recipe, &cfg);
        apply_dct_marks(&mut texture, &recipe, &cfg);
        apply_topology_marks(&mut mesh, &recipe, &cfg);

        let leaked = LeakedContent {
            mesh,
            texture_blocks: texture,
        };
        let reference = LeakedContent {
            mesh: mesh_reference,
            texture_blocks: texture_reference,
        };

        // Register the WRONG recipient.
        let other_recipe = derive_recipe(&cfg, &other_pk, &master);
        let mut registry = InMemoryRegistry::new();
        registry.register_recipe(other_pk, &other_recipe, n_blocks);

        let outcome = attribute(&leaked, &reference, &recipe, &registry, &cfg, 1e-6);
        // The codeword from `real_pk`'s recipe won't match `other_pk`'s
        // registered codeword under exact-match lookup. If by some
        // accident it matches, the confidence must be low.
        match outcome {
            None => { /* expected — codeword doesn't match */ }
            Some(r) => {
                assert!(r.confidence < 0.5, "wrong-recipient confidence too high");
            }
        }
    }

    #[test]
    fn sigmoid_stability() {
        // Sanity: sigmoid is bounded in (0, 1) and centered at 0.5 for x=0.
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(1e6) <= 1.0);
        assert!(sigmoid(-1e6) >= 0.0);
        // Monotonic.
        assert!(sigmoid(1.0) > sigmoid(0.0));
        assert!(sigmoid(0.0) > sigmoid(-1.0));
    }

    #[test]
    fn recover_p_vertex_fits_diagonal() {
        // Synthetic: apply a known diagonal P_vertex to a mesh, then
        // recover it.
        let n = 100;
        let mut mesh = TriangleMesh::default();
        for i in 0..n {
            mesh.positions
                .push([(i as f32) * 0.1, (i as f32) * 0.2, (i as f32) * 0.3]);
        }
        let reference = mesh.clone();
        let scale_x = 1.0001f32;
        let scale_y = 1.0002f32;
        for p in mesh.positions.iter_mut() {
            p[0] *= scale_x;
            p[1] *= scale_y;
        }
        let vertex_indices: Vec<u32> = (0..n as u32).collect();
        let recovered = recover_p_vertex(&mesh, &reference, &vertex_indices);
        let err_x = (recovered[0][0] - scale_x).abs();
        let err_y = (recovered[1][1] - scale_y).abs();
        assert!(err_x < 1e-5, "scale_x recovery err {err_x}");
        assert!(err_y < 1e-5, "scale_y recovery err {err_y}");
    }
}
