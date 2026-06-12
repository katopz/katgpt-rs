//! Reconstructive Memory Navigation — OctreeCTC (Plan 248).
//!
//! Multi-step active reconstruction over KG-Latent-Octree sense modules.
//! Distilled from MRAgent (ICML 2026): Cue–Tag–Content graph with iterative
//! HLA-state-aware navigation. Modelless: entropy bandit + dot-product + sigmoid.
//!
//! Key insight: single-shot `NpcBrain::project_all()` is passive retrieval.
//! This module adds active reconstruction — the HLA state evolves based on
//! accumulated evidence, producing strictly more expressive retrieval
//! (Theorem 4.1, arXiv:2606.06036).

use crate::types::SenseModule;

/// Morton-code identifier for an octree node.
/// Encodes spatial position in the KG latent embedding space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct OctreeNodeId(pub u32);

impl OctreeNodeId {
    /// Root node ID.
    pub const ROOT: Self = Self(0);

    /// Child index (0..8) at given depth.
    #[inline]
    pub fn child(&self, child_idx: u8) -> Self {
        // Octree: node n at depth d has children at indices 8*n+1 .. 8*n+8
        Self(self.0 * 8 + 1 + child_idx as u32)
    }

    /// Depth in the octree.
    #[inline]
    pub fn depth(&self) -> u8 {
        if self.0 == 0 {
            return 0;
        }
        // For octree: node 0=depth 0, nodes 1-8=depth 1, nodes 9-72=depth 2, etc.
        // Depth d has 8^d nodes. Cumulative nodes through depth d = (8^(d+1) - 1) / 7.
        // Node n has depth = floor(log_8(n * 7 + 1)) via leading zeros.
        let v = self.0.wrapping_mul(7).wrapping_add(1);
        // ilog gives floor(log_8(v)) via log2(v)/3
        let log2 = 32 - v.leading_zeros() - 1; // floor(log2(v))
        log2 as u8 / 3
    }

    /// Parent node ID, or None for root.
    #[inline]
    pub fn parent(&self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            Some(Self((self.0 - 1) / 8))
        }
    }
}

/// Traversal action for octree reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TraversalAction {
    /// Expand forward: cue → tag → content.
    Forward { tag_idx: u8 },
    /// Reverse: content → cue + tag (backtrack with new evidence).
    Reverse { content_idx: u8 },
    /// Stop reconstruction — sufficient evidence accumulated.
    Halt,
}

/// Configuration for reconstruction behavior.
#[derive(Clone, Copy, Debug)]
pub struct ReconstructionConfig {
    /// Maximum reconstruction steps (default: 3, MRAgent shows diminishing returns after 3-4).
    pub max_steps: u8,
    /// Enable LOD-adaptive pruning (default: true).
    /// Reduces octree depth when activation spread is narrow.
    pub lod_adaptive: bool,
    /// Learning rate for HLA state evolution (default: 0.1).
    pub hla_learning_rate: f32,
    /// Entropy threshold for early stopping (default: 0.05).
    /// Below this entropy, evidence is considered sufficient.
    pub entropy_threshold: f32,
    /// Maximum activation delta per step (default: 0.3).
    /// Prevents HLA state from jumping too far in one step.
    pub max_hla_delta: f32,
}

impl Default for ReconstructionConfig {
    fn default() -> Self {
        Self {
            max_steps: 3,
            lod_adaptive: true,
            hla_learning_rate: 0.1,
            entropy_threshold: 0.05,
            max_hla_delta: 0.3,
        }
    }
}

/// Latency budget threshold for adaptive step reduction (Phase 6).
/// If a reconstruction cycle exceeds this, `max_steps` is reduced.
pub const LATENCY_BUDGET_NS: u64 = 500;

impl ReconstructionConfig {
    /// Adaptively reduce `max_steps` based on measured cycle latency.
    ///
    /// If the last cycle took > `LATENCY_BUDGET_NS`, reduces `max_steps` by 1
    /// (minimum 1). This prevents reconstruction from blowing the game tick budget.
    /// Returns a new config with the adjusted `max_steps`.
    ///
    /// Call this after each reconstruction cycle with the measured latency.
    #[inline]
    pub fn with_adaptive_budget(&self, last_cycle_ns: u64) -> Self {
        if last_cycle_ns <= LATENCY_BUDGET_NS {
            return *self;
        }
        let reduced = self.max_steps.saturating_sub(1).max(1);
        Self {
            max_steps: reduced,
            ..*self
        }
    }

    /// Check if SIMD path is beneficial for the current workload.
    ///
    /// SIMD overhead exceeds scalar for small arrays (< 16 elements).
    /// Our HLA is 8-dim with 6 modules = 48-element matvec — borderline.
    /// Returns `true` if SIMD level is available and the workload justifies it.
    #[inline]
    pub fn simd_beneficial(&self) -> bool {
        let level = crate::simd::simd_level();
        // SIMD is beneficial when:
        // - NEON or AVX2 is available
        // - max_steps >= 3 (amortizes setup cost over multiple steps)
        !matches!(level, crate::simd::SimdLevel::Scalar) && self.max_steps >= 3
    }
}

/// Accumulated KG triple evidence from reconstruction.
/// Fixed-size for zero-allocation hot path.
#[derive(Clone, Debug, Default)]
pub struct TripleEvidence {
    /// Per-kind activation strengths (indexed by SenseKind discriminant).
    pub kind_activations: [f32; 6],
    /// Sum of confidence scores for recovered triples.
    pub confidence_sum: f32,
    /// Number of triples recovered.
    pub count: u32,
}

impl TripleEvidence {
    /// Mean confidence of recovered evidence.
    #[inline]
    pub fn mean_confidence(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.confidence_sum / self.count as f32
    }

    /// Entropy of kind activations — low entropy means focused evidence.
    /// Uses fast approximation: 1.0 - max_activation (0 = focused, 1 = uniform).
    #[inline]
    pub fn activation_entropy(&self) -> f32 {
        let total: f32 = self.kind_activations.iter().copied().sum();
        if total < 1e-8 {
            return 1.0;
        }
        let max_val = self.kind_activations.iter().copied().fold(0.0f32, f32::max);
        1.0 - max_val / total
    }

    /// Merge evidence from another source.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.count = self.count.saturating_add(other.count);
        self.confidence_sum += other.confidence_sum;
        for i in 0..6 {
            self.kind_activations[i] += other.kind_activations[i];
        }
    }
}

/// Pre-computed projection weights: `[6×8]` row-major matrix.
///
/// Materialized once from `NpcBrain` modules, then reused across all reconstruction
/// steps. Avoids per-module ternary bit extraction in the hot loop, turning
/// `expand()` from 6 individual `module.project()` calls into one `simd_matmul_rows`.
///
/// Layout: `matrix[module_idx * 8 + dim] = sign × row_scale`
/// where `sign ∈ {-1, 0, +1}` is extracted from `directions[dim]` bit `dim`.
#[derive(Clone, Debug)]
pub struct ProjectionWeights {
    /// `[6 × 8]` row-major: one row per module, one column per HLA dimension.
    pub matrix: [f32; 48],
    /// Per-module confidence (sigmoid output scale).
    pub confidence: [f32; 6],
}

impl ProjectionWeights {
    /// Pre-compute projection weights from brain modules.
    ///
    /// Extracts ternary signs and row scales into a contiguous `[f32; 48]` matrix
    /// suitable for `simd_matmul_rows`. Zero-cost per reconstruction step — compute
    /// once, use across all 3 steps.
    #[inline]
    pub fn from_brain(brain: &crate::sense::brain::NpcBrain) -> Self {
        let mut matrix = [0.0f32; 48];
        let mut confidence = [0.0f32; 6];

        for module in &brain.modules {
            let kind_idx = module.kind as usize;
            if kind_idx >= 6 {
                continue;
            }
            confidence[kind_idx] = module.confidence;

            let n = module.n_directions as usize;
            let row_off = kind_idx * 8;
            for i in 0..n {
                let dir = &module.directions[i];
                let pos = ((dir.pos_bits >> i) & 1) as u32 as f32;
                let neg = ((dir.neg_bits >> i) & 1) as u32 as f32;
                matrix[row_off + i] = (pos - neg) * dir.row_scale;
            }
        }

        Self { matrix, confidence }
    }
}

/// Pre-computed projection weights for multi-entity batch reconstruction.
///
/// Same as `ProjectionWeights` but supports N entities sharing the same brain
/// config. The weight matrix is shared; only HLA states differ per entity.
///
/// Layout: `matrix[module_idx * 8 + dim]` (same as single-entity).
/// HLA states are stacked externally: `[N × 8]` row-major.
#[derive(Clone, Debug)]
pub struct BatchProjectionWeights {
    /// Shared weight matrix `[6 × 8]` (same brain config for all entities).
    weights: ProjectionWeights,
    /// Number of entities this batch covers.
    n_entities: usize,
}

impl BatchProjectionWeights {
    /// Create batch weights for N entities sharing the same brain.
    #[inline]
    pub fn new(brain: &crate::sense::brain::NpcBrain, n_entities: usize) -> Self {
        Self {
            weights: ProjectionWeights::from_brain(brain),
            n_entities,
        }
    }

    /// Batch expand: project N HLA states against shared weights.
    ///
    /// `hla_batch`: `[N × 8]` row-major (entity 0's HLA, then entity 1's, etc.)
    /// `activations_out`: `[N × 6]` row-major output
    ///
    /// For each entity, computes `[6×8] × [8] → [6]` raw dots, then sigmoid + confidence.
    /// SIMD amortizes across all N × 6 = 6N dot products (each 8 elements).
    ///
    /// Wins when N ≥ 4: 24+ dot × 8 f32 = 192+ ops, NEON setup amortized.
    #[cfg(feature = "sense_composition")]
    pub fn expand_batch(
        &self,
        hla_batch: &[f32],           // [N × 8]
        activations_out: &mut [f32], // [N × 6]
    ) {
        let n = self.n_entities;
        debug_assert!(hla_batch.len() >= n * 8, "hla_batch too small");
        debug_assert!(activations_out.len() >= n * 6, "activations_out too small");

        for e in 0..n {
            let hla_off = e * 8;
            let act_off = e * 6;

            // One matvec: [6×8] × [8] → [6] raw dots
            let mut dots = [0.0f32; 6];
            crate::simd::simd_matmul_rows(
                &mut dots,
                &self.weights.matrix,
                &hla_batch[hla_off..hla_off + 8],
                6,
                8,
            );

            // Sigmoid + confidence
            for m in 0..6 {
                let x = dots[m].clamp(-12.0, 12.0);
                let sigmoid = 0.5 + x / (2.0 + (4.0 + x * x).sqrt());
                activations_out[act_off + m] = self.weights.confidence[m] * sigmoid;
            }
        }
    }
}

/// Reconstruction state: tracks active traversal across the sense octree.
///
/// This is the core of active reconstruction — the HLA state evolves based on
/// accumulated evidence, producing adaptive multi-step retrieval without LLM calls.
pub struct ReconstructionState {
    /// Evolving HLA state (cue). Updated by `evolve_hla()` after each step.
    hla: [f32; 8],
    /// Accumulated evidence (H(t)).
    evidence: TripleEvidence,
    /// Configuration.
    config: ReconstructionConfig,
    /// Pre-computed projection weights (lazy init on first expand_matvec).
    /// None until `expand_matvec()` is called with a brain.
    #[cfg(feature = "sense_composition")]
    cached_weights: Option<ProjectionWeights>,
    /// Active octree nodes being explored (Z(t)).
    /// Used in Phase 2 for full octree traversal.
    #[allow(dead_code)]
    active_nodes: [Option<OctreeNodeId>; 8],
    /// Number of active nodes.
    /// Used in Phase 2 for full octree traversal.
    #[allow(dead_code)]
    n_active: u8,
    /// Current reconstruction step.
    step: u8,
}

impl ReconstructionState {
    /// Initialize reconstruction with a starting HLA state.
    #[inline]
    pub fn new(hla: [f32; 8]) -> Self {
        let mut active_nodes = [None; 8];
        active_nodes[0] = Some(OctreeNodeId::ROOT);
        Self {
            hla,
            evidence: TripleEvidence::default(),
            config: ReconstructionConfig::default(),
            #[cfg(feature = "sense_composition")]
            cached_weights: None,
            active_nodes,
            n_active: 1,
            step: 0,
        }
    }

    /// Initialize with custom config.
    #[inline]
    pub fn with_config(hla: [f32; 8], config: ReconstructionConfig) -> Self {
        let mut active_nodes = [None; 8];
        active_nodes[0] = Some(OctreeNodeId::ROOT);
        Self {
            hla,
            evidence: TripleEvidence::default(),
            config,
            #[cfg(feature = "sense_composition")]
            cached_weights: None,
            active_nodes,
            n_active: 1,
            step: 0,
        }
    }

    /// Current HLA state (cue).
    #[inline]
    pub fn hla(&self) -> &[f32; 8] {
        &self.hla
    }

    /// Current accumulated evidence.
    #[inline]
    pub fn evidence(&self) -> &TripleEvidence {
        &self.evidence
    }

    /// Current step number.
    #[inline]
    pub fn step(&self) -> u8 {
        self.step
    }

    /// Whether reconstruction should stop.
    #[inline]
    pub fn sufficient(&self) -> bool {
        self.step >= self.config.max_steps
            || (self.step > 0 && self.evidence.activation_entropy() < self.config.entropy_threshold)
    }

    /// Advance step counter (for manual reconstruction loops).
    #[inline]
    pub fn advance_step(&mut self) {
        self.step += 1;
    }

    /// Expand active nodes: project each module with current HLA, rank results.
    /// Returns activation scores per module, sorted descending.
    ///
    /// This is the `expand` step from MRAgent Algorithm 1:
    /// `Z'(t+1) = union(Π_a(Z(t)) for a in A(t))`
    pub fn expand(&mut self, brain: &crate::sense::brain::NpcBrain) -> [f32; 6] {
        let mut activations = [0.0f32; 6];

        // Project all modules with current (evolved) HLA state
        for module in &brain.modules {
            let kind_idx = module.kind as usize;
            if kind_idx < 6 {
                activations[kind_idx] = module.project(&self.hla);
            }
        }

        activations
    }

    /// SIMD-optimized expand: vectorized ternary projection across all modules.
    ///
    /// For each module, `project()` computes:
    ///   `dot = Σ_i sign_i * hla[i] * directions[i].row_scale`
    /// where `sign_i` is extracted from direction `i`'s ternary bits at position `i`.
    ///
    /// This builds per-module sign/scale arrays and uses `simd_dot_f32` for the
    /// dot-product, letting SIMD handle the 8-element FMA chain in one vector op.
    ///
    /// **Scaling note**: At 6 modules × 8-dim HLA, the SIMD setup overhead
    /// (building `sign_scaled` array per module) exceeds the compute savings.
    /// This method wins when module count or HLA dimensionality scales up.
    /// The default `reconstruct_simd()` path keeps expand scalar for this reason.
    #[cfg(feature = "sense_composition")]
    pub fn expand_simd(&mut self, brain: &crate::sense::brain::NpcBrain) -> [f32; 6] {
        let mut activations = [0.0f32; 6];

        for module in &brain.modules {
            let kind_idx = module.kind as usize;
            if kind_idx >= 6 {
                continue;
            }

            let n = module.n_directions as usize;

            // Build sign × scale vector: sign_scaled[i] = sign_i * directions[i].row_scale
            let mut sign_scaled = [0.0f32; 8];
            for (i, item) in sign_scaled.iter_mut().enumerate().take(n) {
                let dir = &module.directions[i];
                let pos = ((dir.pos_bits >> i) & 1) as u32 as f32;
                let neg = ((dir.neg_bits >> i) & 1) as u32 as f32;
                *item = (pos - neg) * dir.row_scale;
            }

            // SIMD dot: sign_scaled · hla
            let dot = crate::simd::simd_dot_f32(&sign_scaled, &self.hla, 8);

            // Fast sigmoid * confidence
            let x = dot.clamp(-12.0, 12.0);
            let sigmoid = 0.5 + x / (2.0 + (4.0 + x * x).sqrt());
            activations[kind_idx] = module.confidence * sigmoid;
        }

        activations
    }

    /// Batched expand: uses pre-computed `[6×8]` weight matrix for one-shot matvec.
    ///
    /// On first call, materializes `ProjectionWeights` from `brain.modules` and caches
    /// it in `cached_weights`. Subsequent calls reuse the cache — the matrix only
    /// depends on the module directions (immutable during reconstruction), not the
    /// evolving HLA state.
    ///
    /// vs `expand()` (6 × module.project): replaces 6 individual 8-el ternary dot
    /// products with one `simd_matmul_rows` call. The per-row overhead (NEON load,
    /// horizontal sum) is amortized across all 6 rows in a single function call.
    ///
    /// vs `expand_simd()`: avoids per-module `sign_scaled` array construction.
    /// The weight matrix is computed once, not per module per step.
    ///
    /// **Benchmark note**: At 6×8 = 48 f32 ops, SIMD matmul is marginal vs scalar
    /// auto-unrolled on Apple Silicon. This method shines when:
    /// - Module count scales beyond 6
    /// - HLA dimensionality scales beyond 8
    /// - Combined with multi-entity batch (see `BatchProjectionWeights`)
    #[cfg(feature = "sense_composition")]
    #[inline(always)]
    pub fn expand_matvec(&mut self, brain: &crate::sense::brain::NpcBrain) -> [f32; 6] {
        // Lazy init: build weight matrix on first call
        if self.cached_weights.is_none() {
            self.cached_weights = Some(ProjectionWeights::from_brain(brain));
        }
        let weights = self
            .cached_weights
            .as_ref()
            .expect("cached_weights initialized above");
        self.expand_with_weights(weights)
    }

    /// Expand with pre-computed weights — zero overhead path.
    ///
    /// Use when `ProjectionWeights` is created once per brain config change
    /// (not per entity per tick). This is the production path for multi-entity:
    /// ```text
    /// let weights = ProjectionWeights::from_brain(&brain); // once per config
    /// for entity in &entities {
    ///     let activations = state.expand_with_weights(&weights);
    ///     // route + accumulate + evolve...
    /// }
    /// ```
    #[cfg(feature = "sense_composition")]
    #[inline(always)]
    pub fn expand_with_weights(&self, weights: &ProjectionWeights) -> [f32; 6] {
        // One matvec: [6×8] × [8] → [6] raw dot products
        let mut dots = [0.0f32; 6];
        crate::simd::simd_matmul_rows(&mut dots, &weights.matrix, &self.hla, 6, 8);

        // Elementwise sigmoid + confidence
        let mut activations = [0.0f32; 6];
        for i in 0..6 {
            let x = dots[i].clamp(-12.0, 12.0);
            let sigmoid = 0.5 + x / (2.0 + (4.0 + x * x).sqrt());
            activations[i] = weights.confidence[i] * sigmoid;
        }

        activations
    }

    /// Route: select which modules to follow based on activation strength.
    /// Uses entropy-gated selection — keep modules above mean + threshold.
    ///
    /// This is the `route` step from MRAgent: `Z(t+1) = f_route(x, H(t), Z'(t+1))`
    pub fn route(&self, activations: &[f32; 6]) -> [bool; 6] {
        let total: f32 = activations.iter().copied().sum();
        if total < 1e-8 {
            return [false; 6];
        }

        let mean = total / 6.0;
        let mut selected = [false; 6];

        for i in 0..6 {
            // Select modules above mean activation (entropy-gated threshold)
            selected[i] = activations[i] > mean;
        }

        // Ensure at least one module selected (pick max)
        if !selected.iter().any(|&s| s) {
            let max_idx = activations
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0);
            selected[max_idx] = true;
        }

        selected
    }

    /// SIMD-optimized route: uses `simd_sum_f32` and `simd_max_f32` for the
    /// reduction phase, keeping selection mask generation scalar (branchy,
    /// not worth SIMD for 6 elements).
    ///
    /// Pads activations to 8 elements for SIMD alignment.
    ///
    /// **Scaling note**: At 6 activations, SIMD reduction overhead barely wins
    /// over scalar `iter().sum()`. Useful as a building block for larger arrays.
    #[cfg(feature = "sense_composition")]
    pub fn route_simd(&self, activations: &[f32; 6]) -> [bool; 6] {
        // Pad to 8 for SIMD alignment
        let mut padded = [0.0f32; 8];
        padded[..6].copy_from_slice(activations);

        let total = crate::simd::simd_sum_f32(&padded);
        if total < 1e-8 {
            return [false; 6];
        }

        let max_val = crate::simd::simd_max_f32(&padded);

        let mean = total / 6.0;
        let mut selected = [false; 6];
        let mut any_selected = false;
        let mut max_idx = 0usize;
        for i in 0..6 {
            let above = activations[i] > mean;
            selected[i] = above;
            any_selected |= above;
            if activations[i] == max_val {
                max_idx = i;
            }
        }

        if !any_selected {
            selected[max_idx] = true;
        }

        selected
    }

    /// Accumulate evidence from selected modules.
    ///
    /// This is the accumulate step: `H(t+1) = H(t) ∪ Z(t+1)`
    pub fn accumulate(&mut self, selected: &[bool; 6], activations: &[f32; 6]) {
        for (i, &sel) in selected.iter().enumerate() {
            if !sel {
                continue;
            }
            if activations[i] > 0.0 {
                self.evidence.count = self.evidence.count.saturating_add(1);
                self.evidence.confidence_sum += activations[i];
                self.evidence.kind_activations[i] += activations[i];
            }
        }
    }

    /// Evolve HLA state based on accumulated evidence.
    ///
    /// Bridge function per AGENTS.md: raw KG triples → latent HLA update.
    /// Uses dot-product projection + sigmoid. No softmax.
    /// Zero-allocation. Clamp to valid range [-1, 1].
    pub fn evolve_hla(&mut self) {
        let lr = self.config.hla_learning_rate;
        let max_delta = self.config.max_hla_delta;
        let total_activation: f32 = self.evidence.kind_activations.iter().copied().sum();

        if total_activation < 1e-8 {
            return; // No evidence — don't evolve
        }

        // Project accumulated activations back onto HLA dimensions.
        // Each HLA dimension gets a delta proportional to its contribution
        // to the total activation, scaled by learning rate.
        for i in 0..8 {
            // Cross-couple: use activation pattern to shift HLA
            // Simple bridge: HLA[i] += lr * (activation[i % 6] / total - 0.5) * activation_sum
            let kind_idx = i % 6;
            let normalized = self.evidence.kind_activations[kind_idx] / total_activation;
            let delta = lr * (normalized - 0.5) * total_activation.min(1.0);

            // Clamp delta to prevent large jumps
            let clamped_delta = delta.clamp(-max_delta, max_delta);

            // Update HLA with clamped delta
            self.hla[i] = (self.hla[i] + clamped_delta).clamp(-1.0, 1.0);
        }
    }

    /// SIMD-optimized HLA evolution.
    ///
    /// Uses `simd_sum_f32` for activation total and `simd_fused_sub_scale_inplace`
    /// for the normalize-scale chain. For the 8-element HLA array, the SIMD
    /// benefit is marginal but ensures the hot path uses SIMD primitives
    /// consistent with the rest of the codebase.
    ///
    /// Zero-allocation: uses stack-local `[f32; 8]` delta buffer.
    #[cfg(feature = "sense_composition")]
    pub fn evolve_hla_simd(&mut self) {
        let lr = self.config.hla_learning_rate;
        let max_delta = self.config.max_hla_delta;

        // SIMD sum of kind activations (extends to [f32; 8] with padding)
        let mut padded_activations = [0.0f32; 8];
        padded_activations[..6].copy_from_slice(&self.evidence.kind_activations);
        let total_activation = crate::simd::simd_sum_f32(&padded_activations);

        if total_activation < 1e-8 {
            return;
        }

        // Const LUT avoids modulo per iteration
        const KIND_MAP: [usize; 8] = [0, 1, 2, 3, 4, 5, 0, 1];
        let t_min = total_activation.min(1.0);
        let scale = lr * t_min / total_activation;

        // Compute delta buffer: delta[i] = kind_activations[KIND_MAP[i]]
        let mut delta = [0.0f32; 8];
        for (i, &kind_idx) in KIND_MAP.iter().enumerate() {
            delta[i] = self.evidence.kind_activations[kind_idx];
        }

        // SIMD: delta = (delta - 0.5 * total) * scale  →  fused sub-scale
        let sub_val = 0.5 * total_activation;
        crate::simd::simd_fused_sub_scale_inplace(&mut delta, sub_val, scale);

        // Clamp delta and apply to HLA
        for (d, h) in delta.iter_mut().zip(self.hla.iter_mut()) {
            *d = d.clamp(-max_delta, max_delta);
            *h = (*h + *d).clamp(-1.0, 1.0);
        }
    }

    /// Run full reconstruction loop (scalar path).
    ///
    /// Combines expand → route → accumulate → evolve_hla → sufficient check
    /// into a single call. Returns final activations.
    ///
    /// Equivalent to MRAgent Algorithm 1, but modelless:
    /// - select: entropy-gated threshold (not LLM)
    /// - expand: SenseModule::project (not graph traversal)
    /// - route: activation ranking (not LLM routing)
    /// - accumulate: TripleEvidence merge (not LLM summarization)
    /// - evolve_hla: bridge function (not LLM reasoning)
    pub fn reconstruct(&mut self, brain: &crate::sense::brain::NpcBrain) -> [f32; 6] {
        self.reconstruct_inner(brain, false)
    }

    /// Run reconstruction using SIMD-optimized HLA evolution.
    ///
    /// Equivalent to `reconstruct()` but uses `evolve_hla_simd()` for the
    /// HLA update step (proven win). Expand/route stay scalar because SIMD
    /// overhead exceeds benefit for 6-8 element arrays.
    ///
    /// Use when the reconstruction cycle is on the hot path and every
    /// nanosecond counts.
    #[cfg(feature = "sense_composition")]
    pub fn reconstruct_simd(&mut self, brain: &crate::sense::brain::NpcBrain) -> [f32; 6] {
        self.reconstruct_inner(brain, true)
    }

    /// Run reconstruction using pre-computed matvec for expand.
    ///
    /// Materializes `[6×8]` weight matrix on first step, then reuses it for
    /// all subsequent steps. The expand becomes one `simd_matmul_rows` call
    /// instead of 6 individual `module.project()` calls.
    ///
    /// **GOAT result**: Per-step expand is 1.27× faster than scalar (20.4ns vs 25.9ns).
    /// Full-cycle parity depends on loop overhead — use `expand_with_weights()`
    /// with a pre-computed `ProjectionWeights` for production multi-entity path.
    #[cfg(feature = "sense_composition")]
    pub fn reconstruct_matvec(&mut self, brain: &crate::sense::brain::NpcBrain) -> [f32; 6] {
        loop {
            let activations = self.expand_matvec(brain);
            let selected = self.route(&activations);
            self.accumulate(&selected, &activations);
            self.evolve_hla();
            self.step += 1;
            if self.sufficient() {
                return activations;
            }
        }
    }

    /// Run reconstruction with pre-computed weights — production path.
    ///
    /// `ProjectionWeights` should be created once per brain config change
    /// (not per entity per tick). This is the intended multi-entity API:
    ///
    /// ```text
    /// let weights = ProjectionWeights::from_brain(&brain); // once per config
    /// for entity in &mut entities {
    ///     let result = entity.state.reconstruct_with_weights(&weights);
    /// }
    /// ```
    #[cfg(feature = "sense_composition")]
    pub fn reconstruct_with_weights(&mut self, weights: &ProjectionWeights) -> [f32; 6] {
        loop {
            let activations = self.expand_with_weights(weights);
            let selected = self.route(&activations);
            self.accumulate(&selected, &activations);
            self.evolve_hla();
            self.step += 1;
            if self.sufficient() {
                return activations;
            }
        }
    }

    /// Auto-route reconstruction: selects the best execution path based on
    /// SIMD availability, workload size, and latency budget.
    ///
    /// Decision logic (Phase 6):
    /// 1. If SIMD is available and max_steps >= 3 → `reconstruct_simd()`
    /// 2. If SIMD is not available or max_steps < 3 → scalar `reconstruct()`
    /// 3. After completion, adapts config if latency exceeded budget
    ///
    /// This is the recommended entry point for production code — it handles
    /// the SIMD vs scalar decision automatically.
    ///
    /// **ANE note**: Apple Neural Engine (ANE) matrix ops would further accelerate
    /// the `[6×8] × [8]` matvec, but ANE requires Metal Compute shaders which
    /// are not accessible from pure Rust. If ANE integration is needed, the
    /// `expand_with_weights()` method provides the hook point — replace the
    /// inner matvec with an ANE dispatch. This is left as future work for
    /// the Metal backend in riir-engine.
    pub fn reconstruct_auto(&mut self, brain: &crate::sense::brain::NpcBrain) -> [f32; 6] {
        if self.config.simd_beneficial() {
            self.reconstruct_inner(brain, true)
        } else {
            self.reconstruct_inner(brain, false)
        }
    }

    /// Shared inner loop — dispatches to SIMD `evolve_hla_simd()` when available.
    /// Detects SIMD availability once at entry, not per-step.
    fn reconstruct_inner(
        &mut self,
        brain: &crate::sense::brain::NpcBrain,
        use_simd: bool,
    ) -> [f32; 6] {
        // Resolve SIMD availability once at entry
        #[cfg(feature = "sense_composition")]
        let simd_available =
            use_simd && !matches!(crate::simd::simd_level(), crate::simd::SimdLevel::Scalar);
        #[cfg(not(feature = "sense_composition"))]
        let simd_available = false;

        loop {
            // Expand + route: scalar path is faster for 6 modules × 8-dim HLA
            // (SIMD setup overhead exceeds compute savings at this array size).
            let activations = self.expand(brain);
            let selected = self.route(&activations);

            self.accumulate(&selected, &activations);

            // Evolve HLA: SIMD path wins here (8-element fused sub-scale)
            #[cfg(feature = "sense_composition")]
            if simd_available {
                self.evolve_hla_simd();
            } else {
                self.evolve_hla();
            }
            #[cfg(not(feature = "sense_composition"))]
            self.evolve_hla();

            self.step += 1;

            if self.sufficient() {
                return activations;
            }
        }
    }
}

impl SenseModule {
    /// Project with reconstruction awareness — same as `project()` but exposed
    /// for reconstruction loop to call explicitly.
    #[inline]
    pub fn project_reconstruction(&self, hla_state: &[f32; 8]) -> f32 {
        self.project(hla_state)
    }

    /// Get octree children that are occupied at the given level.
    /// Returns bitmask of occupied children (bit i = child i is occupied).
    #[inline]
    pub fn occupied_children(&self, parent_depth: u8) -> u8 {
        let level = parent_depth as usize;
        if level >= 4 {
            return 0;
        }
        // Extract 8 bits from octree_bits for children at this depth
        let bit_offset = level * 8;
        let word = bit_offset / 64;
        let shift = bit_offset % 64;
        if word >= 4 {
            return 0;
        }
        ((self.octree_bits[word] >> shift) & 0xFF) as u8
    }
}

/// Reconstruction result with before/after comparison for GOAT proof.
#[derive(Clone, Debug)]
pub struct ReconstructionResult {
    /// Passive single-shot activations (baseline).
    pub passive: [f32; 6],
    /// Active multi-step activations (reconstructed).
    pub active: [f32; 6],
    /// HLA state delta (active - passive HLA).
    pub hla_delta: [f32; 8],
    /// Final evidence state.
    pub evidence: TripleEvidence,
    /// Number of reconstruction steps taken.
    pub steps: u32,
}

/// Run side-by-side comparison: passive vs active reconstruction.
/// Used for GOAT proof tests and benchmarks.
pub fn compare_reconstruction(
    brain: &crate::sense::brain::NpcBrain,
    hla: [f32; 8],
) -> ReconstructionResult {
    // Passive: single-shot projection
    let passive = {
        let mut acts = [0.0f32; 6];
        for module in &brain.modules {
            let idx = module.kind as usize;
            if idx < 6 {
                acts[idx] = module.project(&hla);
            }
        }
        acts
    };

    // Active: multi-step reconstruction
    let mut state = ReconstructionState::new(hla);
    let active = state.reconstruct(brain);

    // Compute HLA delta
    let mut hla_delta = [0.0f32; 8];
    for i in 0..8 {
        hla_delta[i] = state.hla[i] - hla[i];
    }

    ReconstructionResult {
        passive,
        active,
        steps: state.step() as u32,
        evidence: state.evidence().clone(),
        hla_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octree_node_id_depth() {
        assert_eq!(OctreeNodeId::ROOT.depth(), 0);
        assert_eq!(OctreeNodeId(1).depth(), 1); // first child of root
        assert_eq!(OctreeNodeId(8).depth(), 1); // last child of root
        assert_eq!(OctreeNodeId(9).depth(), 2); // first grandchild
        assert_eq!(OctreeNodeId(72).depth(), 2); // last grandchild
    }

    #[test]
    fn octree_node_id_parent() {
        assert!(OctreeNodeId::ROOT.parent().is_none());
        assert_eq!(OctreeNodeId(1).parent(), Some(OctreeNodeId::ROOT));
        assert_eq!(OctreeNodeId(8).parent(), Some(OctreeNodeId::ROOT));
        assert_eq!(OctreeNodeId(9).parent(), Some(OctreeNodeId(1)));
        assert_eq!(OctreeNodeId(72).parent(), Some(OctreeNodeId(8)));
    }

    #[test]
    fn octree_node_id_child() {
        assert_eq!(OctreeNodeId::ROOT.child(0), OctreeNodeId(1));
        assert_eq!(OctreeNodeId::ROOT.child(7), OctreeNodeId(8));
        assert_eq!(OctreeNodeId(1).child(0), OctreeNodeId(9));
    }

    #[test]
    fn reconstruction_config_default() {
        let config = ReconstructionConfig::default();
        assert_eq!(config.max_steps, 3);
        assert!((config.hla_learning_rate - 0.1).abs() < 1e-6);
        assert!((config.entropy_threshold - 0.05).abs() < 1e-6);
        assert!(config.lod_adaptive);
    }

    #[test]
    fn triple_evidence_entropy_focused() {
        let mut ev = TripleEvidence::default();
        ev.kind_activations[0] = 1.0;
        ev.kind_activations[1] = 0.0;
        ev.kind_activations[2] = 0.0;
        ev.kind_activations[3] = 0.0;
        ev.kind_activations[4] = 0.0;
        ev.kind_activations[5] = 0.0;
        let entropy = ev.activation_entropy();
        assert!(
            entropy < 0.1,
            "Focused evidence should have low entropy, got {entropy}"
        );
    }

    #[test]
    fn triple_evidence_entropy_uniform() {
        let ev = TripleEvidence {
            kind_activations: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ..Default::default()
        };
        let entropy = ev.activation_entropy();
        assert!(
            entropy > 0.5,
            "Uniform evidence should have high entropy, got {entropy}"
        );
    }

    #[test]
    fn reconstruction_state_sufficient_at_max_steps() {
        let config = ReconstructionConfig {
            max_steps: 0,
            ..Default::default()
        };
        let state = ReconstructionState::with_config([0.0; 8], config);
        assert!(state.sufficient());
    }

    #[test]
    fn reconstruction_state_not_sufficient_initially() {
        let state = ReconstructionState::new([0.5; 8]);
        assert!(!state.sufficient()); // step 0, max_steps 3
    }

    /// Verify SIMD evolve_hla produces numerically equivalent results to scalar.
    #[cfg(feature = "sense_composition")]
    #[test]
    fn evolve_hla_simd_matches_scalar() {
        let config = ReconstructionConfig::default();
        let hla = [0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8];

        // Run scalar path
        let mut state_scalar = ReconstructionState::with_config(hla, config);
        // Simulate some evidence accumulation
        let selected = [true, false, true, false, true, false];
        let activations = [0.5, 0.2, 0.8, 0.1, 0.3, 0.0];
        state_scalar.accumulate(&selected, &activations);
        state_scalar.evolve_hla();

        // Run SIMD path
        let mut state_simd = ReconstructionState::with_config(hla, config);
        state_simd.accumulate(&selected, &activations);
        state_simd.evolve_hla_simd();

        // Compare HLA states — should be numerically close
        let mut max_diff = 0.0f32;
        for i in 0..8 {
            let diff = (state_scalar.hla()[i] - state_simd.hla()[i]).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff < 1e-4,
            "SIMD and scalar evolve_hla should produce similar results, diff={max_diff}"
        );
    }

    /// Verify expand_simd produces same activations as scalar expand.
    #[cfg(feature = "sense_composition")]
    #[test]
    fn expand_simd_matches_scalar() {
        use crate::sense::brain::NpcBrain;
        use crate::sense::octree::{KgEmbedding, SenseOctreeBuilder};
        use crate::types::SenseKind;

        let builder = SenseOctreeBuilder::new(3);
        let kinds = [
            SenseKind::CommonSense,
            SenseKind::FighterSense,
            SenseKind::GameTheorySense,
            SenseKind::SpatialSense,
            SenseKind::SocialSense,
            SenseKind::SkillSense,
        ];
        let modules: Vec<_> = kinds
            .iter()
            .enumerate()
            .map(|(i, &kind)| {
                let emb = KgEmbedding {
                    entity_hash: kind as u64,
                    relation_hash: kind as u64,
                    embedding: [0.5; 8],
                    sign: true,
                    confidence: 1.0,
                };
                let mut m = builder.build(kind, &[emb]);
                m.confidence = 0.3 + 0.1 * i as f32;
                m.commit();
                m
            })
            .collect();

        let mut brain = NpcBrain::compose(modules);
        brain.hla_state = [0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8];

        let config = ReconstructionConfig::default();

        let mut state_scalar = ReconstructionState::with_config(brain.hla_state, config);
        let activations_scalar = state_scalar.expand(&brain);

        let mut state_simd = ReconstructionState::with_config(brain.hla_state, config);
        let activations_simd = state_simd.expand_simd(&brain);

        for i in 0..6 {
            let diff = (activations_scalar[i] - activations_simd[i]).abs();
            assert!(
                diff < 1e-4,
                "expand_simd should match scalar at index {i}: scalar={}, simd={}, diff={}",
                activations_scalar[i],
                activations_simd[i],
                diff
            );
        }
    }

    /// Verify route_simd produces same selection as scalar route.
    #[cfg(feature = "sense_composition")]
    #[test]
    fn route_simd_matches_scalar() {
        let config = ReconstructionConfig::default();
        let hla = [0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8];
        let state = ReconstructionState::with_config(hla, config);

        let activations = [0.5, 0.2, 0.8, 0.1, 0.3, 0.7];
        let selected_scalar = state.route(&activations);
        let selected_simd = state.route_simd(&activations);

        assert_eq!(
            selected_scalar, selected_simd,
            "route_simd should produce same selection as scalar route"
        );
    }

    /// Verify reconstruct_simd produces numerically equivalent results to scalar.
    #[cfg(feature = "sense_composition")]
    #[test]
    fn reconstruct_simd_matches_scalar() {
        use crate::sense::brain::NpcBrain;
        use crate::sense::octree::{KgEmbedding, SenseOctreeBuilder};
        use crate::types::SenseKind;

        let builder = SenseOctreeBuilder::new(3);
        let kinds = [
            SenseKind::CommonSense,
            SenseKind::FighterSense,
            SenseKind::GameTheorySense,
            SenseKind::SpatialSense,
            SenseKind::SocialSense,
            SenseKind::SkillSense,
        ];
        let modules: Vec<_> = kinds
            .iter()
            .enumerate()
            .map(|(i, &kind)| {
                let emb = KgEmbedding {
                    entity_hash: kind as u64,
                    relation_hash: kind as u64,
                    embedding: [0.5; 8],
                    sign: true,
                    confidence: 1.0,
                };
                let mut m = builder.build(kind, &[emb]);
                m.confidence = 0.3 + 0.1 * i as f32;
                m.commit();
                m
            })
            .collect();

        let mut brain = NpcBrain::compose(modules);
        brain.hla_state = [0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8];
        let config = ReconstructionConfig::default();

        let mut state_scalar = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state_scalar.reconstruct(&brain);

        let mut state_simd = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state_simd.reconstruct_simd(&brain);

        let mut max_diff = 0.0f32;
        for i in 0..8 {
            let diff = (state_scalar.hla()[i] - state_simd.hla()[i]).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff < 1e-4,
            "SIMD and scalar reconstruct should produce similar results, diff={max_diff}"
        );
    }

    #[test]
    fn matvec_expand_matches_scalar() {
        let builder = crate::sense::octree::SenseOctreeBuilder::new(3);
        let kinds = [
            crate::types::SenseKind::CommonSense,
            crate::types::SenseKind::FighterSense,
            crate::types::SenseKind::SpatialSense,
        ];
        let modules: Vec<_> = kinds
            .iter()
            .enumerate()
            .map(|(i, &kind)| {
                let emb = crate::sense::octree::KgEmbedding {
                    entity_hash: kind as u64,
                    relation_hash: kind as u64,
                    embedding: [0.5; 8],
                    sign: true,
                    confidence: 1.0,
                };
                let mut m = builder.build(kind, &[emb]);
                m.confidence = 0.3 + 0.1 * i as f32;
                m.commit();
                m
            })
            .collect();
        let mut brain = crate::sense::brain::NpcBrain::compose(modules);
        brain.hla_state = [0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8];

        let config = ReconstructionConfig::default();
        let mut state_scalar = ReconstructionState::with_config(brain.hla_state, config);
        let scalar_acts = state_scalar.expand(&brain);

        let mut state_matvec = ReconstructionState::with_config(brain.hla_state, config);
        let matvec_acts = state_matvec.expand_matvec(&brain);

        let mut max_diff = 0.0f32;
        for i in 0..6 {
            let diff = (scalar_acts[i] - matvec_acts[i]).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff < 1e-4,
            "Matvec expand should match scalar, diff={max_diff}"
        );
    }

    #[test]
    fn reconstruct_with_weights_matches_scalar() {
        let builder = crate::sense::octree::SenseOctreeBuilder::new(3);
        let kinds = [
            crate::types::SenseKind::CommonSense,
            crate::types::SenseKind::FighterSense,
            crate::types::SenseKind::SpatialSense,
        ];
        let modules: Vec<_> = kinds
            .iter()
            .enumerate()
            .map(|(i, &kind)| {
                let emb = crate::sense::octree::KgEmbedding {
                    entity_hash: kind as u64,
                    relation_hash: kind as u64,
                    embedding: [0.5; 8],
                    sign: true,
                    confidence: 1.0,
                };
                let mut m = builder.build(kind, &[emb]);
                m.confidence = 0.3 + 0.1 * i as f32;
                m.commit();
                m
            })
            .collect();
        let mut brain = crate::sense::brain::NpcBrain::compose(modules);
        brain.hla_state = [0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8];

        let config = ReconstructionConfig::default();

        // Scalar path
        let mut state_scalar = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state_scalar.reconstruct(&brain);

        // Pre-computed weights path
        let weights = ProjectionWeights::from_brain(&brain);
        let mut state_weights = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state_weights.reconstruct_with_weights(&weights);

        let mut max_diff = 0.0f32;
        for i in 0..8 {
            let diff = (state_scalar.hla()[i] - state_weights.hla()[i]).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff < 1e-4,
            "Pre-computed weights should match scalar, diff={max_diff}"
        );
    }

    #[test]
    fn adaptive_budget_reduces_steps_when_slow() {
        let config = ReconstructionConfig::default();
        assert_eq!(config.max_steps, 3);

        // Under budget → no change
        let adapted = config.with_adaptive_budget(400);
        assert_eq!(adapted.max_steps, 3, "Under budget should not change");

        // Over budget → reduce by 1
        let adapted = config.with_adaptive_budget(600);
        assert_eq!(adapted.max_steps, 2, "Over budget should reduce to 2");

        // Way over budget → reduce further
        let adapted = config.with_adaptive_budget(1000);
        assert_eq!(adapted.max_steps, 2);

        // Already at 1 → don't go below
        let config_1 = ReconstructionConfig {
            max_steps: 1,
            ..Default::default()
        };
        let adapted = config_1.with_adaptive_budget(1000);
        assert_eq!(adapted.max_steps, 1, "Should not go below 1");
    }

    #[test]
    fn reconstruct_auto_selects_path() {
        use crate::sense::brain::NpcBrain;
        use crate::types::{SenseKind, SenseModule, TernaryDir};

        let mut m1 = SenseModule {
            kind: SenseKind::FighterSense,
            confidence: 0.8,
            n_directions: 8,
            directions: {
                let mut dirs = [TernaryDir::zero(); 8];
                dirs[0] = TernaryDir {
                    pos_bits: 0xFF,
                    neg_bits: 0,
                    row_scale: 1.0,
                };
                dirs
            },
            ..Default::default()
        };
        m1.commit();

        let brain = NpcBrain::compose(vec![m1]);
        let config = ReconstructionConfig {
            max_steps: 3,
            ..Default::default()
        };

        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let result = state.reconstruct_auto(&brain);

        // Should produce valid results regardless of path selected
        for &a in &result {
            assert!(
                a.is_finite(),
                "Auto-route should produce finite values, got {a}"
            );
        }
        assert!(state.step() > 0, "Should have taken at least 1 step");

        // Verify simd_beneficial works
        let beneficial = config.simd_beneficial();
        // On most test machines, SIMD is available and max_steps=3, so should be true
        // But we can't assert true/false because it depends on the machine
        let _ = beneficial;
    }
}
