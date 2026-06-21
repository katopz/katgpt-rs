//! Curator verification layer for the Merkle octree system (Plan 253).
//!
//! Phase 2 (Curator Verification) and Phase 3 (Curator Bandit):
//! - **CuratorVerifier**: modelless verification of sense data committed to a Merkle octree.
//! - **MerkleFrozenEnvelope**: freeze/thaw with Merkle root integrity.
//! - **CuratorBandit**: Thompson-sampling reputation tracker for curator accuracy.
//! - **verification_weight**: AbsorbCompress-style weight multiplier.
//!
//! GOAT targets: verify < 2µs, freeze/thaw < 1µs, sample+update < 100ns.
//! Feature-gated behind `merkle_octree`.

use std::collections::HashMap;

use crate::merkle::{HASH_SIZE, MERKLE_OCTREE_INTERNAL, MERKLE_OCTREE_LEAVES, MerkleOctree};
use crate::types::{SenseModule, TernaryDir};

// ---------------------------------------------------------------------------
// T4: CuratorVerifier — modelless verification
// ---------------------------------------------------------------------------

/// Result of curator verification.
#[derive(Clone, Copy, Debug)]
pub struct CuratorVerdict {
    /// KG consistency: dot-product of first direction with itself (self-similarity).
    /// Non-zero when directions carry meaningful signal.
    pub kg_consistency: f32,
    /// Latent conditioning: sigmoid(dot(query_vector, direction[0])) in [0, 1].
    pub latent_conditioning: f32,
    /// Spectral flatness: variance of leaf hashes exceeds entropy floor.
    pub spectral_ok: bool,
    /// Overall pass/fail.
    pub pass_: bool,
}

/// Modelless verifier for sense data committed to a Merkle octree.
///
/// Checks three properties without any model weights:
/// 1. KG consistency: dot-product similarity between direction vectors.
/// 2. Spectral flatness: variance of leaf hashes exceeds entropy floor.
/// 3. Latent conditioning: sigmoid projection of query onto direction is valid.
#[derive(Clone, Debug)]
pub struct CuratorVerifier {
    /// Minimum acceptable KG consistency (dot-product similarity).
    pub kg_consistency_threshold: f32,
    /// Minimum acceptable spectral flatness (variance of leaf hashes).
    pub spectral_floor: f32,
    /// Query vector for latent conditioning check.
    pub query_vector: [f32; 8],
}

impl Default for CuratorVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl CuratorVerifier {
    /// Create a new verifier with sensible defaults.
    ///
    /// - `kg_consistency_threshold = 0.1`
    /// - `spectral_floor = 1e-6`
    /// - `query_vector = [1.0; 8]`
    pub fn new() -> Self {
        Self {
            kg_consistency_threshold: 0.1,
            spectral_floor: 1e-6,
            query_vector: [1.0; 8],
        }
    }

    /// Verify a sense module against its Merkle octree.
    ///
    /// No allocation, no model weights. GOAT target: < 2µs.
    pub fn verify_module(&self, module: &SenseModule, tree: &MerkleOctree) -> CuratorVerdict {
        // 1. KG consistency: dot-product of direction[0] with itself.
        //    Self-similarity = ||dir||² in ternary sign space.
        let kg_consistency = match module.n_directions {
            0 => 0.0,
            _ => {
                let dir = &module.directions[0];
                Self::ternary_dot_self(dir)
            }
        };

        // 2. Spectral flatness: variance of leaf hashes (first 8 bytes LE as u64).
        let spectral_ok = Self::check_spectral(tree, self.spectral_floor);

        // 3. Latent conditioning: sigmoid(dot(query_vector, direction[0])).
        let latent_conditioning = match module.n_directions {
            0 => 0.5, // neutral sigmoid(0)
            _ => {
                let dir = &module.directions[0];
                let dot = self.query_direction_dot(dir);
                Self::sigmoid(dot)
            }
        };

        let pass_ = kg_consistency >= self.kg_consistency_threshold
            && spectral_ok
            && (0.0..=1.0).contains(&latent_conditioning);

        CuratorVerdict {
            kg_consistency,
            spectral_ok,
            latent_conditioning,
            pass_,
        }
    }

    /// Compute dot-product of a ternary direction with itself.
    ///
    /// For each dim: sign = (pos_bit - neg_bit) ∈ {-1, 0, +1}.
    /// Dot-self = Σ sign² × row_scale² = Σ (has_sign) × row_scale².
    /// Only dimensions where pos or neg is set contribute.
    #[inline(always)]
    fn ternary_dot_self(dir: &TernaryDir) -> f32 {
        let active = (dir.pos_bits | dir.neg_bits).count_ones() as f32;
        active * dir.row_scale * dir.row_scale
    }

    /// Check spectral flatness of leaf hashes.
    ///
    /// Treats each `[u8; 32]` leaf hash as a u64 (first 8 bytes LE),
    /// then computes mean and variance. Leaf hashes with sufficient
    /// entropy should have non-trivial variance.
    #[inline]
    fn check_spectral(tree: &MerkleOctree, floor: f32) -> bool {
        // Process 64 leaf hashes as u64 via direct byte read.
        // Single-pass Welford-style variance: numerically stable.
        let n = MERKLE_OCTREE_LEAVES as f64;
        let mut mean = 0.0f64;
        let mut m2 = 0.0f64;
        for i in 0..MERKLE_OCTREE_LEAVES {
            let leaf = &tree.hashes[MERKLE_OCTREE_INTERNAL + 1 + i];
            // Read first 8 bytes as little-endian u64 — same codegen as
            // `read_unaligned` on little-endian targets, but safe and portable.
            let val = u64::from_le_bytes(leaf[..8].try_into().unwrap());
            let val = val as f64;
            // Welford online variance
            let delta = val - mean;
            mean += delta / (i as f64 + 1.0);
            let delta2 = val - mean;
            m2 += delta * delta2;
        }

        let variance = m2 / n;
        variance >= floor as f64
    }

    /// Dot-product of query_vector with ternary direction signs.
    ///
    /// For each dim i: sign_i = (pos_bit_i - neg_bit_i) ∈ {-1, 0, +1}.
    /// dot = Σ query[i] × sign_i × row_scale.
    #[inline(always)]
    fn query_direction_dot(&self, dir: &TernaryDir) -> f32 {
        let mut dot = 0.0f32;
        for i in 0..8 {
            let pos = ((dir.pos_bits >> i) & 1) as f32;
            let neg = ((dir.neg_bits >> i) & 1) as f32;
            let sign = pos - neg;
            dot += self.query_vector[i] * sign;
        }
        dot * dir.row_scale
    }

    /// Sigmoid projection — delegates to shared crate::simd::fast_sigmoid
    /// (bounded (0,1), libm-exp accuracy).
    #[inline(always)]
    fn sigmoid(x: f32) -> f32 {
        crate::simd::fast_sigmoid(x)
    }
}

// ---------------------------------------------------------------------------
// T5: MerkleFrozenEnvelope — freeze/thaw with Merkle root
// ---------------------------------------------------------------------------

/// Frozen target data: token IDs and weights, similar to MuxTarget but standalone.
/// Avoids coupling to the `mux_freeze_thaw` feature.
#[derive(Clone, Debug)]
pub struct FrozenTarget {
    /// Depth at which this pattern was recorded.
    pub depth: usize,
    /// Token IDs in the superposition.
    pub tokens: Vec<u32>,
    /// Weights for each token.
    pub weights: Vec<f32>,
}

impl FrozenTarget {
    /// Create a new frozen target.
    pub fn new(tokens: Vec<u32>, weights: Vec<f32>, depth: usize) -> Self {
        Self {
            depth,
            tokens,
            weights,
        }
    }
}

/// Frozen Merkle envelope: pairs frozen data with its Merkle root for verification.
#[derive(Clone, Debug)]
pub struct MerkleEnvelope {
    /// Lookup key.
    pub key: u64,
    /// The frozen target data.
    pub target: FrozenTarget,
    /// BLAKE3 Merkle root at time of freeze.
    pub merkle_root: [u8; HASH_SIZE],
}

/// Store for Merkle-verified frozen envelopes.
///
/// Extends the `MuxPatternStore` freeze/thaw pattern but is standalone
/// (does not depend on `mux` feature at the module level).
#[derive(Clone, Debug, Default)]
pub struct MerkleFrozenStore {
    envelopes: HashMap<u64, Vec<MerkleEnvelope>>,
}

impl MerkleFrozenStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Freeze a target with its Merkle root.
    ///
    /// Stores the envelope under the given key. Later, `thaw_and_verify`
    /// can check integrity against an expected root.
    pub fn freeze_with_root(
        &mut self,
        key: u64,
        target: FrozenTarget,
        merkle_root: [u8; HASH_SIZE],
    ) {
        let envelope = MerkleEnvelope {
            key,
            target,
            merkle_root,
        };
        self.envelopes.entry(key).or_default().push(envelope);
    }

    /// Thaw envelopes for a key and verify against an expected Merkle root.
    ///
    /// Returns a list of `(target reference, root_matches)` pairs.
    /// `root_matches` is `true` when the envelope's Merkle root equals `expected_root`.
    pub fn thaw_and_verify(
        &self,
        key: u64,
        expected_root: &[u8; HASH_SIZE],
    ) -> Vec<(&FrozenTarget, bool)> {
        match self.envelopes.get(&key) {
            Some(envelopes) => envelopes
                .iter()
                .map(|e| (&e.target, e.merkle_root == *expected_root))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Number of distinct keys stored.
    pub fn key_count(&self) -> usize {
        self.envelopes.len()
    }

    /// Total number of envelopes across all keys.
    pub fn envelope_count(&self) -> usize {
        self.envelopes.values().map(|v| v.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// T7: CuratorBandit — Thompson sampling reputation tracker
// ---------------------------------------------------------------------------

/// Per-curator bandit arm state for Thompson sampling.
///
/// Beta(α, β) posterior over curator correctness.
/// Prior: α=1, β=1 (uniform).
#[derive(Clone, Copy, Debug, Default)]
pub struct CuratorArm {
    /// Alpha (success count) for Beta distribution.
    pub alpha: f32,
    /// Beta (failure count) for Beta distribution.
    pub beta: f32,
}

/// Bandit-based curator reputation tracker.
///
/// Uses Thompson sampling with a fast posterior-mean + noise approximation
/// for sub-100ns sample. EMA decay handles concept drift.
#[derive(Clone, Debug)]
pub struct CuratorBandit {
    /// Per-curator arm states.
    arms: Vec<CuratorArm>,
    /// EMA decay rate for concept drift handling.
    pub ema_decay: f32,
    /// RNG for Thompson sampling.
    rng: fastrand::Rng,
}

impl CuratorBandit {
    /// Create a new bandit with `n_curators` arms, each with uniform prior α=1, β=1.
    pub fn new(n_curators: usize) -> Self {
        let arms = vec![
            CuratorArm {
                alpha: 1.0,
                beta: 1.0,
            };
            n_curators
        ];
        Self {
            arms,
            ema_decay: 0.99,
            rng: fastrand::Rng::new(),
        }
    }

    /// Thompson sample from the Beta(α, β) posterior for a curator.
    ///
    /// Uses the posterior mean + exploration noise approximation:
    /// `mean = α / (α + β)`, noise = `U(-1, 1) / (α + β + 1)`.
    /// GOAT target: < 50ns.
    ///
    /// Returns 0.5 for out-of-bounds curator IDs (neutral / unknown).
    pub fn sample(&mut self, curator_id: usize) -> f32 {
        match self.arms.get(curator_id) {
            Some(arm) => {
                let total = arm.alpha + arm.beta;
                let mean = arm.alpha / total;
                // Exploration noise: uniform perturbation scaled by uncertainty.
                // Higher total → lower noise → more exploitation.
                let noise = (self.rng.f32() * 2.0 - 1.0) / (total + 1.0);
                (mean + noise).clamp(0.0, 1.0)
            }
            None => 0.5,
        }
    }

    /// Update the bandit arm for a curator based on verification outcome.
    ///
    /// `correct = true` → α += 1 (success).
    /// `correct = false` → β += 1 (failure).
    /// GOAT target: < 50ns.
    pub fn update(&mut self, curator_id: usize, correct: bool) {
        if let Some(arm) = self.arms.get_mut(curator_id) {
            match correct {
                true => arm.alpha += 1.0,
                false => arm.beta += 1.0,
            }
        }
    }

    /// Get the reputation (posterior mean) for a curator.
    ///
    /// Returns α / (α + β). Higher = more accurate.
    /// Returns 0.5 for out-of-bounds curator IDs.
    pub fn reputation(&self, curator_id: usize) -> f32 {
        match self.arms.get(curator_id) {
            Some(arm) => arm.alpha / (arm.alpha + arm.beta),
            None => 0.5,
        }
    }

    /// Apply EMA decay to all arms for concept drift handling.
    ///
    /// `α *= ema_decay`, `β *= ema_decay`. This gradually forgets
    /// old observations so the bandit adapts to changing curator accuracy.
    pub fn decay(&mut self) {
        for arm in &mut self.arms {
            arm.alpha *= self.ema_decay;
            arm.beta *= self.ema_decay;
        }
    }

    /// Number of curator arms.
    pub fn len(&self) -> usize {
        self.arms.len()
    }

    /// Whether the bandit has any arms.
    pub fn is_empty(&self) -> bool {
        self.arms.is_empty()
    }
}

// ---------------------------------------------------------------------------
// T8: AbsorbCompress integration — verification weight
// ---------------------------------------------------------------------------

/// Verification weight multiplier based on curator accuracy.
///
/// - **High accuracy (>80%)**: amplified weight, 1.0..2.0.
/// - **Medium accuracy (50-80%)**: linear pass-through, 0.5..0.8.
/// - **Low accuracy (<50%)**: probation, weight → 0.
#[inline]
pub fn verification_weight(reputation: f32) -> f32 {
    // Branchless: piecewise linear with sigmoid-like transition
    // Below 0.5: 0.0 (probation)
    // 0.5..0.8: linear ramp (reputation)
    // Above 0.8: 1.0 + (reputation - 0.8) * 5.0 (1.0..2.0)
    let prob_mask = (reputation >= 0.5) as u32 as f32;
    let boosted_mask = (reputation > 0.8) as u32 as f32;
    // For 0.5..0.8: linear_val = reputation, for > 0.8: base = 1.0
    let linear_val = reputation * prob_mask * (1.0 - boosted_mask);
    // FMA: reputation * 5.0 - 4.0 (== 1.0 + (reputation - 0.8) * 5.0).
    let boosted_val = reputation.mul_add(5.0, -4.0) * boosted_mask;
    linear_val + boosted_val
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle::MERKLE_OCTREE_LEAVES;

    /// Helper: build a tree with varied leaf data (non-zero spectral variance).
    fn varied_tree() -> MerkleOctree {
        let mut leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        for i in 0..MERKLE_OCTREE_LEAVES {
            let mut buf = [0u8; 32];
            buf[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            leaf_hashes[i] = *blake3::hash(&buf).as_bytes();
        }
        MerkleOctree::build_from_leaves(&leaf_hashes)
    }

    /// Helper: build a tree with all-zero leaves (low spectral variance).
    fn zero_leaf_tree() -> MerkleOctree {
        let leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        MerkleOctree::build_from_leaves(&leaf_hashes)
    }

    /// Helper: build a SenseModule with a single meaningful direction.
    fn module_with_direction() -> SenseModule {
        use crate::types::{SenseKind, TernaryDir};
        let mut dirs = [TernaryDir::zero(); 8];
        // Direction 0: pos_bits = 0b11 (dims 0,1 positive), neg_bits = 0b100 (dim 2 negative)
        dirs[0] = TernaryDir {
            pos_bits: 0b011,
            neg_bits: 0b100,
            row_scale: 1.0,
        };
        let mut module = SenseModule {
            octree_bits: [0; 4],
            directions: dirs,
            confidence: 0.9,
            kind: SenseKind::SpatialSense,
            version: 1,
            octree_depth: 3,
            n_directions: 1,
            _reserved: 0,
            commitment: [0u8; 32],
        };
        module.commit();
        module
    }

    /// Helper: build a SenseModule with no directions (empty).
    fn empty_module() -> SenseModule {
        use crate::types::{SenseKind, TernaryDir};
        let mut module = SenseModule {
            octree_bits: [0; 4],
            directions: [TernaryDir::zero(); 8],
            confidence: 0.0,
            kind: SenseKind::SpatialSense,
            version: 1,
            octree_depth: 3,
            n_directions: 0,
            _reserved: 0,
            commitment: [0u8; 32],
        };
        module.commit();
        module
    }

    // ---- T4: CuratorVerifier tests ----

    #[test]
    fn test_curator_verifier_consistent_kg() {
        let verifier = CuratorVerifier::new();
        let module = module_with_direction();
        let tree = varied_tree();
        let verdict = verifier.verify_module(&module, &tree);

        // Module has direction with row_scale=1.0, 3 active dims → consistency = 3.0
        assert!(
            verdict.kg_consistency >= verifier.kg_consistency_threshold,
            "kg_consistency {} should be >= {}",
            verdict.kg_consistency,
            verifier.kg_consistency_threshold,
        );
        assert!(
            verdict.latent_conditioning >= 0.0 && verdict.latent_conditioning <= 1.0,
            "latent_conditioning should be in [0,1], got {}",
            verdict.latent_conditioning,
        );
        assert!(
            verdict.spectral_ok,
            "varied tree should have spectral variance"
        );
        assert!(verdict.pass_, "consistent module + varied tree should pass");
    }

    #[test]
    fn test_curator_verifier_inconsistent_kg() {
        let verifier = CuratorVerifier::new();
        let module = empty_module();
        let tree = varied_tree();
        let verdict = verifier.verify_module(&module, &tree);

        // Empty module has n_directions=0, so kg_consistency=0.0
        assert_eq!(
            verdict.kg_consistency, 0.0,
            "empty module should have zero consistency"
        );
        assert!(
            verdict.kg_consistency < verifier.kg_consistency_threshold,
            "empty module should fail KG threshold"
        );
        assert!(!verdict.pass_, "empty module should not pass");
    }

    #[test]
    fn test_curator_verifier_spectral_anomaly() {
        let verifier = CuratorVerifier::new();
        let module = module_with_direction();
        let tree = zero_leaf_tree(); // all-zero leaves → zero variance
        let verdict = verifier.verify_module(&module, &tree);

        // All-zero leaf hashes have zero variance → spectral check fails
        assert!(
            !verdict.spectral_ok,
            "all-zero leaves should fail spectral check"
        );
        assert!(!verdict.pass_, "spectral anomaly should fail overall");
    }

    // ---- T5: MerkleFrozenStore tests ----

    #[test]
    fn test_merkle_frozen_store_freeze_thaw() {
        let mut store = MerkleFrozenStore::new();
        let root = [0xABu8; HASH_SIZE];
        let target = FrozenTarget::new(vec![1, 2, 3], vec![0.5, 0.3, 0.2], 0);

        store.freeze_with_root(42, target.clone(), root);

        let results = store.thaw_and_verify(42, &root);
        assert_eq!(results.len(), 1);
        assert!(results[0].1, "root should match");
        assert_eq!(results[0].0.tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_merkle_frozen_store_wrong_root() {
        let mut store = MerkleFrozenStore::new();
        let root = [0xABu8; HASH_SIZE];
        let wrong_root = [0xCDu8; HASH_SIZE];
        let target = FrozenTarget::new(vec![1, 2, 3], vec![0.5, 0.3, 0.2], 0);

        store.freeze_with_root(42, target, root);

        let results = store.thaw_and_verify(42, &wrong_root);
        assert_eq!(results.len(), 1);
        assert!(!results[0].1, "wrong root should not match");
    }

    // ---- T7: CuratorBandit tests ----

    #[test]
    fn test_curator_bandit_convergence() {
        let mut bandit = CuratorBandit::new(3);

        // Simulate 100 correct verifications for curator 0
        for _ in 0..100 {
            bandit.update(0, true);
        }
        // Mix in 20 failures for curator 1
        for _ in 0..20 {
            bandit.update(1, false);
        }

        let rep_0 = bandit.reputation(0);
        assert!(
            rep_0 > 0.75,
            "after 100 correct updates, reputation should be > 0.75, got {}",
            rep_0,
        );

        let rep_1 = bandit.reputation(1);
        assert!(
            rep_1 < 0.5,
            "after 20 failures, reputation should be < 0.5, got {}",
            rep_1,
        );
    }

    #[test]
    fn test_curator_bandit_probation() {
        let mut bandit = CuratorBandit::new(2);

        // Many failures for curator 0
        for _ in 0..50 {
            bandit.update(0, false);
        }

        let rep = bandit.reputation(0);
        assert!(
            rep < 0.5,
            "after 50 failures, reputation should be < 0.5, got {}",
            rep,
        );

        let weight = verification_weight(rep);
        assert_eq!(
            weight, 0.0,
            "low-accuracy curator should be on probation (weight=0)"
        );
    }

    #[test]
    fn test_curator_bandit_out_of_bounds() {
        let bandit = CuratorBandit::new(2);
        assert_eq!(
            bandit.reputation(99),
            0.5,
            "out-of-bounds should return 0.5"
        );
    }

    #[test]
    fn test_curator_bandit_decay() {
        let mut bandit = CuratorBandit::new(1);

        // Build up some counts
        for _ in 0..10 {
            bandit.update(0, true);
        }
        let pre_alpha = bandit.arms[0].alpha;
        let pre_beta = bandit.arms[0].beta;

        bandit.decay();

        assert!(
            bandit.arms[0].alpha < pre_alpha,
            "alpha should decay: {} < {}",
            bandit.arms[0].alpha,
            pre_alpha,
        );
        assert!(
            bandit.arms[0].beta < pre_beta,
            "beta should decay: {} < {}",
            bandit.arms[0].beta,
            pre_beta,
        );
    }

    // ---- T8: verification_weight tests ----

    #[test]
    fn test_verification_weight_thresholds() {
        // High accuracy: amplified
        let w_high = verification_weight(0.9);
        assert_eq!(w_high, 1.0 + (0.9 - 0.8) * 5.0);
        assert!(w_high > 1.0, "high accuracy should amplify");
        assert!(w_high <= 2.0, "max amplification is 2.0");

        // Very high accuracy (edge case: reputation=1.0)
        let w_max = verification_weight(1.0);
        assert_eq!(w_max, 2.0, "reputation 1.0 → weight 2.0");

        // Medium accuracy: linear pass-through
        let w_mid = verification_weight(0.65);
        assert_eq!(w_mid, 0.65, "medium accuracy should pass through linearly");

        // Boundary: exactly 0.5
        let w_boundary = verification_weight(0.5);
        assert_eq!(w_boundary, 0.5, "exactly 0.5 is not probation");

        // Low accuracy: probation
        let w_low = verification_weight(0.3);
        assert_eq!(w_low, 0.0, "low accuracy should be probation (weight=0)");

        // Very low accuracy
        let w_zero = verification_weight(0.1);
        assert_eq!(w_zero, 0.0, "very low accuracy should be probation");

        // Boundary: exactly 0.8 → linear pass-through (0.8 > 0.8 is false)
        let w_eighty = verification_weight(0.8);
        assert_eq!(
            w_eighty, 0.8,
            "exactly 0.8 → linear pass-through (amplification starts above 0.8)"
        );
    }

    #[test]
    fn test_curator_bandit_sample_in_range() {
        let mut bandit = CuratorBandit::new(1);

        // Sample many times — all should be in [0, 1]
        for _ in 0..1000 {
            let s = bandit.sample(0);
            assert!(
                (0.0..=1.0).contains(&s),
                "sample should be in [0,1], got {}",
                s,
            );
        }
    }
}
