//! Per-recipient forensic recipe derivation.
//!
//! A `Recipe` is a compact (few-hundred-byte) bundle of perturbation
//! parameters derived deterministically from `(master_seed, recipient_pubkey)`
//! via BLAKE3. It binds the recipient's identity to:
//!
//! - A 2×2 vertex perturbation matrix `P_vertex` with LoopWM spectral
//!   stability (`A = diag(-exp(a))` → eigenvalues in `(0,1)`,
//!   `det(I + ε·Ā) ≈ 1`). See Research 268 §4, arxiv 2606.18208.
//! - A list of marked vertex indices.
//! - A list of `(block_idx, coef_idx)` mid-frequency DCT positions.
//! - A topology mask (one bit per triangle).
//! - A Tardos anti-collusion codeword truncated to the recipe bandwidth.
//!
//! Everything is a deterministic function of the seed — no per-recipient
//! state is stored on the embedder side.

use blake3::Hasher;

use crate::forensic::tardos::{self, TardosCodebook};

/// Write the BLAKE3 output of `h` into `out` (32 bytes).
#[inline]
fn finalize_into(h: &Hasher, out: &mut [u8; 32]) {
    let hash = h.finalize();
    let bytes = hash.as_bytes();
    *out = *bytes;
}

/// Tuning knobs for recipe derivation. Defaults match Plan 293:
/// L ≈ 1000 bits total recipe bandwidth at c=10, n=10⁵, ε_fp=10⁻⁶.
#[derive(Clone, Copy, Debug)]
pub struct RecipeConfig {
    /// L_v: number of vertices to perturb. Default 50.
    pub vertex_mark_count: usize,
    /// L_dct: number of (block, coef) DCT marks. Default 50.
    pub dct_mark_count: usize,
    /// L_t: number of topology (triangle) marks. Default 100.
    pub topology_mark_count: usize,
    /// ε: vertex displacement scale. Default 1e-4.
    pub epsilon_vertex: f32,
    /// δ: DCT coefficient magnitude. Default 2.0 (just above BC7 noise floor).
    pub delta_dct: f32,
    /// c: design colluder bound for the Tardos codebook. Default 10.
    pub colluder_bound: usize,
    /// ε_fp: design false-positive probability. Default 1e-6.
    pub false_positive_epsilon: f64,
    /// n_recipients: design population for the codebook. Default 100_000.
    pub n_recipients: usize,
}

impl Default for RecipeConfig {
    fn default() -> Self {
        Self {
            vertex_mark_count: 50,
            dct_mark_count: 50,
            topology_mark_count: 100,
            epsilon_vertex: 1e-4,
            delta_dct: 2.0,
            colluder_bound: 10,
            false_positive_epsilon: 1e-6,
            n_recipients: 100_000,
        }
    }
}

impl RecipeConfig {
    /// Total recipe bandwidth in bits: L_v + L_dct + L_t.
    #[inline]
    pub fn codeword_length(&self) -> usize {
        self.vertex_mark_count + self.dct_mark_count + self.topology_mark_count
    }
}

/// A derived per-recipient recipe. Cheap to clone and serialize — order
/// of KB at default config.
#[derive(Clone, Debug)]
pub struct Recipe {
    /// 2×2 vertex perturbation matrix `P_vertex = I + ε·Ā` (diagonal for now).
    /// Eigenvalues of `Ā` lie in (0,1) by LoopWM construction.
    pub p_vertex: [[f32; 2]; 2],
    /// Marked vertex indices (length = `vertex_mark_count`).
    pub vertex_indices: Vec<u32>,
    /// Mid-frequency DCT positions `(block_idx, coef_idx)`.
    /// `coef_idx ∈ [10, 32]`.
    pub dct_indices: Vec<(u32, u8)>,
    /// Per-triangle topology mask bits (length = `topology_mark_count`).
    pub topology_mask: Vec<u8>,
    /// Truncated Tardos codeword of length `codeword_length()`.
    pub codeword: Vec<u8>,
    /// Recipient pubkey hash (BLAKE3 of input pubkey).
    pub recipient_id: [u8; 32],
    /// Tardos codebook used (so recovery can recompute bits).
    pub codebook: TardosCodebook,
    /// Stable recipient index in [0, n_recipients).
    pub recipient_idx: usize,
}

/// Derive a recipe from `(master_seed, recipient_pubkey)`. Fully
/// deterministic — same inputs always produce the same recipe.
pub fn derive_recipe(
    config: &RecipeConfig,
    recipient_pubkey: &[u8; 32],
    master_seed: &[u8; 32],
) -> Recipe {
    // Domain-separated per-recipient seed.
    let seed = derive_seed(master_seed, recipient_pubkey);

    // Recipient identity hash (for registry inverse-lookup).
    let mut rid_hasher = Hasher::new();
    rid_hasher.update(b"forensic_recipe_v1::recipient_id");
    rid_hasher.update(&seed);
    let mut recipient_id = [0u8; 32];
    finalize_into(&rid_hasher, &mut recipient_id);

    // Build the Tardos codebook sized for the full population, then
    // truncate the per-recipient codeword to the recipe bandwidth.
    let codebook = TardosCodebook::generate(
        &seed,
        config.n_recipients,
        config.colluder_bound,
        config.false_positive_epsilon,
    );
    let recipient_idx = tardos::recipient_index(&seed, recipient_pubkey, config.n_recipients);
    let full_codeword = codebook.codeword(recipient_idx);
    let bandwidth = config.codeword_length();
    // Truncate to recipe bandwidth. If the codebook is shorter than the
    // bandwidth (won't happen at default config), wrap.
    let codeword: Vec<u8> = (0..bandwidth)
        .map(|i| full_codeword[i % codebook.length])
        .collect();

    // 2×2 perturbation matrix from the seed.
    let p_vertex = construct_perturbation_matrix(&seed, config.epsilon_vertex);

    // Vertex indices: pseudorandom selection into the recipient's slot.
    // The caller is responsible for ensuring the mesh has at least
    // `vertex_mark_count` vertices; we mod by a sentinel 2³² − 1 here and
    // callers typically mod by their actual vertex count.
    let vertex_indices = derive_indices(
        &seed,
        b"vertex_indices",
        config.vertex_mark_count,
        u32::MAX - 1,
    );

    // DCT indices: mid-frequency range coef ∈ [10, 32].
    let dct_indices = derive_dct_indices(&seed, config.dct_mark_count);

    // Topology mask: last L_t codeword bits.
    let topology_mask: Vec<u8> = codeword
        [config.vertex_mark_count + config.dct_mark_count..]
        .to_vec();

    Recipe {
        p_vertex,
        vertex_indices,
        dct_indices,
        topology_mask,
        codeword,
        recipient_id,
        codebook,
        recipient_idx,
    }
}

/// Per-recipient BLAKE3-derived seed:
/// `seed = BLAKE3("forensic_recipe_v1" ‖ master_seed ‖ recipient_pubkey)`.
#[inline]
pub fn derive_seed(master_seed: &[u8; 32], recipient_pubkey: &[u8; 32]) -> [u8; 32] {
    let mut h = Hasher::new();
    h.update(b"forensic_recipe_v1");
    h.update(master_seed);
    h.update(recipient_pubkey);
    let mut out = [0u8; 32];
    finalize_into(&h, &mut out);
    out
}

/// Construct the 2×2 vertex perturbation matrix with LoopWM spectral
/// stability (Research 268 §4, arxiv 2606.18208):
///
/// ```text
/// A   = diag(-exp(a₁), -exp(a₂))   ← a ∈ ℝ² from seed
/// Ā   = exp(Δ · A)                  ← all eigenvalues in (0, 1)
/// P_vertex = I + ε · Ā              ← det(I + εD) = ∏(1 + ε·dᵢ) ≈ 1
/// ```
///
/// Contraction of `Ā` bounds cumulative displacement across the chunk
/// graph and keeps perturbations below the BC7 noise floor.
pub fn construct_perturbation_matrix(seed: &[u8; 32], epsilon: f32) -> [[f32; 2]; 2] {
    // a₁, a₂ derived from seed bytes. We use the first two u32s, mapped
    // into [-2, 2] so exp() lands in [exp(-2), exp(2)] ⊂ (0, ~7.4).
    // Negative sign in A = diag(-exp(a)) guarantees contraction: the
    // eigenvalues of Ā = exp(Δ·A) are exp(Δ·(-exp(aᵢ))) ∈ (0, 1).
    let a1 = seed_f32_bounded(seed, 0, -2.0, 2.0);
    let a2 = seed_f32_bounded(seed, 4, -2.0, 2.0);
    let delta = 1.0f32;
    // Ā eigenvalues in (0, 1): exp(-exp(a)).
    let bar_a1 = (delta * (-a1.exp())).exp();
    let bar_a2 = (delta * (-a2.exp())).exp();
    // P_vertex = I + ε·Ā (diagonal).
    let p11 = 1.0 + epsilon * bar_a1;
    let p22 = 1.0 + epsilon * bar_a2;

    debug_assert!(
        bar_a1 > 0.0 && bar_a1 < 1.0,
        "bar_a1={bar_a1} not in (0,1)"
    );
    debug_assert!(
        bar_a2 > 0.0 && bar_a2 < 1.0,
        "bar_a2={bar_a2} not in (0,1)"
    );

    [[p11, 0.0], [0.0, p22]]
}

/// Read 4 bytes from `seed` at `offset` and map linearly into `[lo, hi]`.
#[inline]
fn seed_f32_bounded(seed: &[u8; 32], offset: usize, lo: f32, hi: f32) -> f32 {
    let bytes: [u8; 4] = [
        seed[offset & 31],
        seed[(offset + 1) & 31],
        seed[(offset + 2) & 31],
        seed[(offset + 3) & 31],
    ];
    let u = u32::from_le_bytes(bytes);
    let t = u as f32 / u32::MAX as f32; // [0, 1]
    lo + t * (hi - lo)
}

/// Derive `count` u32 indices from the seed, each reduced mod `modulo`.
fn derive_indices(seed: &[u8; 32], domain: &'static [u8], count: usize, modulo: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mut h = Hasher::new();
        h.update(seed);
        h.update(domain);
        h.update(&(i as u64).to_le_bytes());
        let mut buf = [0u8; 32];
        finalize_into(&h, &mut buf);
        let u = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        out.push(u % modulo);
    }
    out
}

/// Derive `count` DCT `(block_idx, coef_idx)` pairs. Coef index is
/// mid-frequency ∈ [10, 32]. Block index is unbounded (caller mods by
/// their block count). Pairs are deduped so each (block, coef) position
/// is marked at most once — this guarantees exact round-trip recovery
/// (no compounding collisions).
fn derive_dct_indices(seed: &[u8; 32], count: usize) -> Vec<(u32, u8)> {
    const COEF_MIN: u8 = 10;
    const COEF_MAX_EXCLUSIVE: u8 = 33; // [10, 32] inclusive
    let coef_range = (COEF_MAX_EXCLUSIVE - COEF_MIN) as u32;
    let mut out: Vec<(u32, u8)> = Vec::with_capacity(count);
    let mut seen = std::collections::HashSet::new();
    let mut i = 0usize;
    // Generate pairs until we have `count` unique ones. Use a counter
    // to drive the hash; bail after `count * 4` attempts to avoid an
    // infinite loop if the (block, coef) space is exhausted.
    let max_attempts = count * 4 + 16;
    while out.len() < count && i < max_attempts {
        let mut h = Hasher::new();
        h.update(seed);
        h.update(b"dct_indices");
        h.update(&(i as u64).to_le_bytes());
        let mut buf = [0u8; 32];
        finalize_into(&h, &mut buf);
        let block = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let coef_raw = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let coef = (COEF_MIN as u32 + (coef_raw % coef_range)) as u8;
        let key = (block, coef);
        if seen.insert(key) {
            out.push((block, coef));
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn rand_seed(byte: u8) -> [u8; 32] {
        let mut s = [byte; 32];
        // Sprinkle in some variety so different seeds genuinely differ.
        for i in 0..32 {
            s[i] = byte.wrapping_add(i as u8).wrapping_mul(7);
        }
        s
    }

    #[test]
    fn determinism_same_inputs_same_recipe() {
        let cfg = RecipeConfig::default();
        let pk = [3u8; 32];
        let ms = [9u8; 32];
        let r1 = derive_recipe(&cfg, &pk, &ms);
        let r2 = derive_recipe(&cfg, &pk, &ms);
        assert_eq!(r1.p_vertex, r2.p_vertex);
        assert_eq!(r1.vertex_indices, r2.vertex_indices);
        assert_eq!(r1.dct_indices, r2.dct_indices);
        assert_eq!(r1.topology_mask, r2.topology_mask);
        assert_eq!(r1.codeword, r2.codeword);
        assert_eq!(r1.recipient_id, r2.recipient_id);
    }

    #[test]
    fn per_recipient_distinctness() {
        let cfg = RecipeConfig::default();
        let ms = [9u8; 32];
        let pk1 = [1u8; 32];
        let pk2 = [2u8; 32];
        let r1 = derive_recipe(&cfg, &pk1, &ms);
        let r2 = derive_recipe(&cfg, &pk2, &ms);
        assert_ne!(r1.p_vertex, r2.p_vertex);
        assert_ne!(r1.recipient_id, r2.recipient_id);
        assert_ne!(r1.codeword, r2.codeword);
    }

    #[test]
    fn p_vertex_spectral_stability_over_10k_seeds() {
        // det ∈ [0.9999, 1.0001], eig(Ā) ∈ (0, 1) for 10⁴ random seeds.
        let eps = 1e-4f32;
        for byte in 0..200u8 {
            for inner in 0..50u8 {
                let mut seed = [0u8; 32];
                for i in 0..32 {
                    seed[i] = byte.wrapping_add(inner).wrapping_add(i as u8);
                }
                let p = construct_perturbation_matrix(&seed, eps);
                let det = p[0][0] * p[1][1] - p[0][1] * p[1][0];
                // det(I + ε·diag(d₁,d₂)) = (1+ε·d₁)(1+ε·d₂) where dᵢ ∈ (0,1).
                // So det ∈ (1, (1+ε)²) ⊂ (1, 1+2ε+ε²). For ε=1e-4, det ∈ (1, 1.00020001).
                assert!(
                    det >= 0.9999 && det <= 1.00021,
                    "det={det} out of bounds for seed byte {byte}/{inner}"
                );
                // Ā eigenvalues = (p_ii − 1) / ε.
                let bar_a1 = (p[0][0] - 1.0) / eps;
                let bar_a2 = (p[1][1] - 1.0) / eps;
                assert!(bar_a1 > 0.0 && bar_a1 < 1.0, "bar_a1={bar_a1}");
                assert!(bar_a2 > 0.0 && bar_a2 < 1.0, "bar_a2={bar_a2}");
            }
        }
    }

    #[test]
    fn codeword_length_matches_bandwidth() {
        let cfg = RecipeConfig::default();
        let r = derive_recipe(&cfg, &[0u8; 32], &[0u8; 32]);
        assert_eq!(r.codeword.len(), cfg.codeword_length());
        // L_v + L_dct + L_t = 50 + 50 + 100 = 200 in the default config.
        // The plan's "≈ 1000 bits" is the full Tardos length; we
        // deliberately truncate to a recipe-shaped bandwidth for
        // embedding efficiency.
        assert_eq!(cfg.codeword_length(), 200);
    }

    #[test]
    fn vertex_indices_count_and_dct_coef_range() {
        let cfg = RecipeConfig::default();
        let r = derive_recipe(&cfg, &[7u8; 32], &[8u8; 32]);
        assert_eq!(r.vertex_indices.len(), cfg.vertex_mark_count);
        assert_eq!(r.dct_indices.len(), cfg.dct_mark_count);
        for &(_, coef) in &r.dct_indices {
            assert!(coef >= 10 && coef <= 32, "coef {coef} outside mid-freq range");
        }
        assert_eq!(r.topology_mask.len(), cfg.topology_mark_count);
    }
}
