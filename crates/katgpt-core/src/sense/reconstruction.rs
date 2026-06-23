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

#![allow(clippy::needless_range_loop)]

use crate::types::SenseModule;

#[cfg(feature = "temporal_deriv")]
use crate::temporal_deriv::TemporalDerivativeKernel;

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
    /// Fast EMA coefficient for the HLA surprise kernel (default: 0.3).
    ///
    /// Plan 277 Fusion F1: the dual `(fast − slow)` band-pass derivative
    /// tracks *how fast* the HLA is changing (vs `evolve_hla` which tracks
    /// *what is*). Paper's canonical ~10× ratio vs `temporal_deriv_alpha_slow`
    /// (O'Reilly 2026, arXiv:2606.08720).
    #[cfg(feature = "temporal_deriv")]
    pub temporal_deriv_alpha_fast: f32,
    /// Slow EMA coefficient for the HLA surprise kernel (default: 0.03).
    ///
    /// See `temporal_deriv_alpha_fast` — the ~10× ratio is the paper's
    /// canonical separation of time constants.
    #[cfg(feature = "temporal_deriv")]
    pub temporal_deriv_alpha_slow: f32,
    /// Self-advantage recursion gate threshold (Plan 283 T5.1).
    ///
    /// Default: `f32::NAN` (disabled). When finite (e.g., `0.01`), the
    /// reconstruction loop halts early if the advantage margin for the
    /// top-routed module drops below this threshold — i.e., the current
    /// step did not improve the prediction for the candidate module above
    /// the population average (dead compute detected).
    ///
    /// This is the 4th early-stop criterion, complementary to `max_steps`,
    /// `entropy_threshold`, and `with_adaptive_budget`. Unlike those (which
    /// ask "is this step done?" or "is this step slow?"), this asks
    /// "did this step help?" — an improvement signal.
    ///
    /// # Input semantics
    ///
    /// The canonical `AdvantageMarginGate` (root crate, `src/pruners/self_advantage.rs`)
    /// consumes raw policy logits. Here, module activations are sigmoid-bounded
    /// `[0, 1]` and treated as logits over 6 module candidates. The advantage
    /// math (`log π+ − log π̂`) is invariant to the input scale — it only
    /// measures relative shifts between steps. The threshold needs separate
    /// tuning from the LLM-logit benchmark because the dynamic range differs.
    #[cfg(feature = "self_advantage_gate")]
    pub advantage_margin_threshold: f32,
}

impl Default for ReconstructionConfig {
    fn default() -> Self {
        Self {
            max_steps: 3,
            lod_adaptive: true,
            hla_learning_rate: 0.1,
            entropy_threshold: 0.05,
            max_hla_delta: 0.3,
            #[cfg(feature = "temporal_deriv")]
            temporal_deriv_alpha_fast: 0.3,
            #[cfg(feature = "temporal_deriv")]
            temporal_deriv_alpha_slow: 0.03,
            // Plan 283 T5.1: DEFAULT-ON (0.01) after GOAT gate T5.1.4 passed
            // (2.50× steps saved, 100% argmax match, 0ns overhead — see
            // `.benchmarks/057_self_advantage_hla_gate.md`). Set to NaN to
            // disable the 4th early-stop criterion (byte-identical to baseline).
            #[cfg(feature = "self_advantage_gate")]
            advantage_margin_threshold: 0.01,
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
        // Single pass: compute sum + max simultaneously
        let mut total = 0.0f32;
        let mut max_val = 0.0f32;
        for &v in &self.kind_activations {
            total += v;
            max_val = max_val.max(v);
        }
        if total < 1e-8 {
            return 1.0;
        }
        1.0 - max_val / total
    }

    /// Merge evidence from another source.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.count = self.count.saturating_add(other.count);
        self.confidence_sum += other.confidence_sum;
        // Direct indexing for fixed-size array — LLVM unrolls fully.
        for i in 0..6 {
            self.kind_activations[i] += other.kind_activations[i];
        }
    }

    /// Map HLA dimension index (0..8) → source SenseKind index (0..6).
    ///
    /// The HLA is 8-dimensional but only 6 SenseKinds feed it, so dims 6 and 7
    /// reuse kinds 0 and 1. This is the **single source of truth** for that
    /// gather — used by both [`ReconstructionState::evolve_hla`] and
    /// [`ReconstructionState::evolve_hla_simd`] via [`kind_activations_padded`].
    ///
    /// (Plan 276 T2.1 — do NOT duplicate this constant elsewhere.)
    pub const KIND_MAP: [usize; 8] = [0, 1, 2, 3, 4, 5, 0, 1];

    /// Gather the 6 per-kind activations into the 8-element HLA input vector.
    ///
    /// Returns `[k0,k1,k2,k3,k4,k5,k0,k1]` — the activations laid out per
    /// [`KIND_MAP`](Self::KIND_MAP). This is the `input` passed to the shared
    /// leaky-integrator core ([`crate::leaky_core::leaky_step`]).
    ///
    /// NOTE: the normalization `total` is NOT `Σ padded` — it is `Σ padded[..6]`
    /// (the 6 distinct source activations). See [`evolve_hla`] and the
    /// `leaky_core` module docs for why `total` is supplied separately.
    #[inline]
    pub fn kind_activations_padded(&self) -> [f32; 8] {
        let k = &self.kind_activations;
        // Direct indexing over a const LUT — fully unrolled, zero allocation.
        let m = Self::KIND_MAP;
        [
            k[m[0]], k[m[1]], k[m[2]], k[m[3]], k[m[4]], k[m[5]], k[m[6]], k[m[7]],
        ]
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
                // Bit masked to 0/1 — `as u32` is redundant.
                let pos = ((dir.pos_bits >> i) & 1) as f32;
                let neg = ((dir.neg_bits >> i) & 1) as f32;
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

            // Sigmoid + confidence. Uses crate::simd::fast_sigmoid for numerical
            // equivalence with scalar `SenseModule::project` (the rational
            // approximation overshoots (0,1) for |x| > 2.67 — see simd.rs docs).
            // NOTE: `simd_sigmoid_inplace` was benchmarked and rejected for the
            // 6-element expand path — see `expand_with_weights` for details.
            for m in 0..6 {
                activations_out[act_off + m] =
                    self.weights.confidence[m] * crate::simd::fast_sigmoid(dots[m]);
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
    /// Dual fast/slow EMA surprise kernel observing the HLA output channel
    /// (Plan 277 Fusion F1, gated by `temporal_deriv`).
    ///
    /// `None` until first observation only if the feature is disabled at
    /// compile time (then the field is absent entirely — zero cost). When the
    /// feature is ON the kernel is always `Some` and is fed every `evolve_hla`
    /// tick. Tracks *how fast* the HLA is changing; `evolve_hla` itself tracks
    /// *what is*.
    #[cfg(feature = "temporal_deriv")]
    surprise: Option<TemporalDerivativeKernel<8>>,
    /// Last `(fast − slow)` derivative written by `evolve_hla` (zero-init).
    ///
    /// Feature-gated for byte-identical layout when `temporal_deriv` is OFF.
    /// Only written under `temporal_deriv`; read via [`surprise_vector`](Self::surprise_vector).
    #[cfg(feature = "temporal_deriv")]
    last_surprise: [f32; 8],
}

impl ReconstructionState {
    /// Initialize reconstruction with a starting HLA state.
    #[inline]
    pub fn new(hla: [f32; 8]) -> Self {
        Self::with_config(hla, ReconstructionConfig::default())
    }

    /// Initialize with custom config.
    #[inline]
    pub fn with_config(hla: [f32; 8], config: ReconstructionConfig) -> Self {
        let mut active_nodes = [None; 8];
        active_nodes[0] = Some(OctreeNodeId::ROOT);

        // Build the surprise kernel from config alphas. Done outside the
        // struct literal so `config` can move cleanly; alphas are Copy.
        #[cfg(feature = "temporal_deriv")]
        let surprise = Some(TemporalDerivativeKernel::new(
            config.temporal_deriv_alpha_fast,
            config.temporal_deriv_alpha_slow,
        ));

        Self {
            hla,
            evidence: TripleEvidence::default(),
            config,
            #[cfg(feature = "sense_composition")]
            cached_weights: None,
            active_nodes,
            n_active: 1,
            step: 0,
            #[cfg(feature = "temporal_deriv")]
            surprise,
            #[cfg(feature = "temporal_deriv")]
            last_surprise: [0.0; 8],
        }
    }

    /// Current HLA state (cue).
    #[inline]
    pub fn hla(&self) -> &[f32; 8] {
        &self.hla
    }

    /// Mutable access to the HLA state (cue).
    ///
    /// Exposed for **post-evolve steering** (Plan 309 Phase 5 / Research 153):
    /// after `evolve_hla()` has reconciled the HLA against accumulated evidence,
    /// a game-runtime caller applies an additive latent-field overlay directly
    /// onto this slice via `apply_latent_steering` / `apply_latent_steering_weighted`.
    ///
    /// This is a **latent-only, think-brain** mutation: it never enters `SyncBlock`,
    /// never crosses the sync boundary, and never touches the frozen personality
    /// shard — only the mutable per-tick belief state. Per AGENTS.md, only the 5
    /// scalar emotion projections cross sync afterwards via the existing bridge.
    ///
    /// Game runtimes that do not use latent-field steering have no reason to
    /// call this — the default read path is [`hla`](Self::hla).
    #[inline]
    pub fn hla_mut(&mut self) -> &mut [f32; 8] {
        &mut self.hla
    }

    /// Last `(fast − slow)` surprise derivative written by `evolve_hla`.
    ///
    /// Returns `None` when the `temporal_deriv` feature is off (no surprise
    /// channel exists); returns `Some(&[f32; 8])` otherwise. The vector is
    /// zero-initialized until the first `evolve_hla` tick.
    ///
    /// Plan 277 Fusion F1, T2.3.
    #[inline]
    pub fn surprise_vector(&self) -> Option<&[f32; 8]> {
        #[cfg(feature = "temporal_deriv")]
        {
            Some(&self.last_surprise)
        }
        #[cfg(not(feature = "temporal_deriv"))]
        {
            None
        }
    }

    /// L2 norm of the current `(fast − slow)` HLA surprise derivative.
    ///
    /// Returns `0.0` when the `temporal_deriv` feature is off; otherwise
    /// delegates to [`TemporalDerivativeKernel::surprise_norm`]. Bounded
    /// scalar suitable for syncing as a raw summary statistic (per
    /// AGENTS.md latent→raw bridge rules).
    ///
    /// Plan 277 Fusion F1, T2.4.
    #[inline]
    pub fn surprise_norm(&self) -> f32 {
        #[cfg(feature = "temporal_deriv")]
        {
            match self.surprise {
                Some(ref s) => s.surprise_norm(),
                None => 0.0,
            }
        }
        #[cfg(not(feature = "temporal_deriv"))]
        {
            0.0
        }
    }

    /// Inject a direct additive delta into the HLA state (per-dim clamped to
    /// `[-1, 1]`). Does NOT touch evidence and does NOT observe into the
    /// surprise kernel — call [`evolve_hla`](Self::evolve_hla) afterward to
    /// feed the updated HLA into the surprise channel.
    ///
    /// Use cases: scripted narrative events (combat onset, loot drops,
    /// encounters), benchmark event injection (Plan 277 G2 gate), and debug
    /// introspection. The HLA field is otherwise private and only mutated
    /// through `evolve_hla`.
    #[inline]
    pub fn inject_hla_delta(&mut self, delta: [f32; 8]) {
        for i in 0..8 {
            self.hla[i] = (self.hla[i] + delta[i]).clamp(-1.0, 1.0);
        }
    }

    /// Directly overwrite the per-SenseKind activation drive signal.
    ///
    /// Plan 331 Phase 1 audit helper: lets [`crate::sense::reconstruction_depth_invariance`]
    /// inject a controlled per-tick stimulus into the leaky integrator without
    /// going through `accumulate()` (which *adds* rather than *sets*). Only the
    /// `kind_activations` field is touched — `confidence_sum` / `count` are
    /// unchanged (they do not feed `evolve_hla`'s math).
    ///
    /// `pub(crate)` so the audit module can use it without leaking a generic
    /// setter into the public API.
    #[cfg(feature = "depth_invariance")]
    #[inline]
    pub(crate) fn set_kind_activations(&mut self, activations: [f32; 6]) {
        self.evidence.kind_activations = activations;
    }

    // NOTE: The previous `pub(crate) fn hla_mut` (Plan 331 Phase 1, gated by
    // `depth_invariance`) was removed — it duplicated the unconditional
    // `pub fn hla_mut` added by Plan 309 Phase 5 for latent-field steering.
    // `pub` is a superset of `pub(crate)`, so the unconditional method serves
    // both callers (`reconstruction_depth_invariance::evolve_hla_regularized`
    // and `apply_latent_steering`). Keeping both triggered E0592 whenever
    // `depth_invariance` was enabled (its default state).

    /// Feed the current HLA into the surprise kernel and cache the derivative
    /// in `last_surprise`. Private — `evolve_hla` and `evolve_hla_simd` are
    /// the only callers, which keeps a single observation point per tick.
    ///
    /// One predicted branch (`if let Some`); no allocation. The kernel is
    /// always `Some` under `temporal_deriv`, but the `Option` keeps the field
    /// initializable to `None` for future opt-out paths.
    #[cfg(feature = "temporal_deriv")]
    #[inline]
    fn observe_surprise_inner(&mut self) {
        if let Some(ref mut s) = self.surprise {
            self.last_surprise = s.observe(&self.hla);
        }
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

    /// 4th early-stop criterion (Plan 283 T5.1): self-advantage recursion gate.
    ///
    /// Returns `true` if reconstruction should halt because the current step
    /// is dead compute — the top-routed module's activation did not improve
    /// above the population average relative to the previous step.
    ///
    /// Returns `false` to continue. No-op when:
    /// - `config.advantage_margin_threshold` is NaN (disabled), OR
    /// - `prev` is `None` (first step — no prior to compare against).
    ///
    /// Uses a stack-local `[f32; 18]` scratch (3 × 6 elements) — zero allocation.
    /// The math is a minimal inline of the canonical `self_advantage_margin`
    /// (root crate, `src/pruners/self_advantage.rs`), kept here because
    /// katgpt-core cannot depend on the root crate. The two implementations
    /// are mathematically equivalent; the canonical one is the tested,
    /// benchmarked reference.
    #[cfg(feature = "self_advantage_gate")]
    #[inline]
    fn advantage_gate_halt(&self, prev: Option<&[f32; 6]>, curr: &[f32; 6]) -> bool {
        let threshold = self.config.advantage_margin_threshold;
        if threshold.is_nan() {
            return false; // disabled
        }
        let Some(prev) = prev else {
            return false; // first step — no prior
        };
        // Stack-local scratch: [pre_lsm | post_lsm | advantage] each [f32; 6].
        let mut scratch = [0.0f32; 18];
        let margin = advantage_margin_hla(prev, curr, argmax6(curr), &mut scratch);
        margin < threshold
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
                let pos = ((dir.pos_bits >> i) & 1) as f32;
                let neg = ((dir.neg_bits >> i) & 1) as f32;
                *item = (pos - neg) * dir.row_scale;
            }

            // SIMD dot: sign_scaled · hla
            let dot = crate::simd::simd_dot_f32(&sign_scaled, &self.hla, 8);

            // Fast sigmoid * confidence. Uses crate::simd::fast_sigmoid for
            // numerical equivalence with the scalar `SenseModule::project` path
            // (the previous rational approximation `0.5 + x/(2+sqrt(4+x^2))`
            // overshoots (0,1) for |x| > 2.67 — see simd.rs docs).
            activations[kind_idx] = module.confidence * crate::simd::fast_sigmoid(dot);
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

        // Elementwise sigmoid + confidence. Uses crate::simd::fast_sigmoid for
        // numerical equivalence with scalar `SenseModule::project` (the rational
        // approximation `0.5 + x/(2+sqrt(4+x^2))` overshoots (0,1) for |x| > 2.67).
        //
        // NOTE: `simd_sigmoid_inplace` was benchmarked here and REJECTED. On
        // Apple Silicon NEON, libm `expf` is fast enough (~5 ns/call) that the
        // 6-call scalar loop beats the SIMD polynomial path — the Cephes setup
        // (10+ `vdupq` constants) + 1 SIMD chunk + 2-element scalar tail costs
        // more than 6 libm calls. Padding to 8 helped the per-step expand
        // (~25% faster) but regressed the full reconstruction cycle (~40 ns
        // slower) due to register/icache pressure in the hot loop. The scalar
        // `fast_sigmoid` loop is the GOAT here.
        let mut activations = [0.0f32; 6];
        for i in 0..6 {
            activations[i] = weights.confidence[i] * crate::simd::fast_sigmoid(dots[i]);
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

        // Precompute reciprocal to avoid per-element division
        let mean = total * (1.0 / 6.0);

        // Single pass: build selection mask + track max activation
        let mut selected = [false; 6];
        let mut any_selected = false;
        let mut max_idx = 0usize;
        let mut max_val = activations[0];

        for i in 0..6 {
            let above = activations[i] > mean;
            selected[i] = above;
            any_selected |= above;
            if activations[i] > max_val {
                max_val = activations[i];
                max_idx = i;
            }
        }

        // Ensure at least one module selected (pick max)
        if !any_selected {
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

        let mean = total * (1.0 / 6.0);
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
    ///
    /// Plan 276 T2.1: now a thin delegate over the shared leaky-integrator core
    /// ([`crate::leaky_core::leaky_step`]). The `KIND_MAP` gather lives in
    /// [`TripleEvidence::kind_activations_padded`], and the normalization total
    /// is `Σ kind_activations[0..6]` (the 6 source activations — NOT the
    /// gathered 8; see `leaky_core` module docs). Behavior is byte-identical to
    /// the previous inline implementation.
    pub fn evolve_hla(&mut self) {
        // total is over the 6 distinct SenseKind activations — preserved exactly
        // from the original inline math. Do not change to Σ padded (would alter
        // scale and break the shipped HLA benchmarks).
        let total: f32 = self.evidence.kind_activations.iter().copied().sum();
        let input = self.evidence.kind_activations_padded();
        crate::leaky_core::leaky_step(
            &mut self.hla,
            &input,
            total,
            self.config.hla_learning_rate,
            self.config.max_hla_delta,
        );

        // Fusion F1 (Plan 277): feed the post-update HLA into the surprise
        // kernel. Runs even when `leaky_step` no-op'd (total < 1e-8) — the
        // kernel still needs to observe the (unchanged) signal so its EMAs
        // converge and the derivative decays to zero on a stationary HLA.
        #[cfg(feature = "temporal_deriv")]
        self.observe_surprise_inner();
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

        // SIMD sum of kind activations (zero-padded to [f32; 8] — sum is over the
        // 6 distinct activations, matching the scalar path's Σ[0..6]).
        let mut padded_activations = [0.0f32; 8];
        padded_activations[..6].copy_from_slice(&self.evidence.kind_activations);
        let total_activation = crate::simd::simd_sum_f32(&padded_activations);

        if total_activation < 1e-8 {
            return;
        }

        let t_min = total_activation.min(1.0);
        let scale = lr * t_min / total_activation;

        // Gather: KIND_MAP = [0,1,2,3,4,5,0,1] → lives once in
        // TripleEvidence::kind_activations_padded() (Plan 276 T2.2 dedup).
        let mut delta = self.evidence.kind_activations_padded();

        // SIMD: delta = (delta - 0.5 * total) * scale  →  fused sub-scale
        let sub_val = 0.5 * total_activation;
        crate::simd::simd_fused_sub_scale_inplace(&mut delta, sub_val, scale);

        // Clamp delta and apply to HLA
        for (d, h) in delta.iter_mut().zip(self.hla.iter_mut()) {
            *d = d.clamp(-max_delta, max_delta);
            *h = (*h + *d).clamp(-1.0, 1.0);
        }

        // Fusion F1 (Plan 277): SIMD path must feed the surprise kernel too —
        // numerical equivalence with the scalar path is asserted in
        // `evolve_hla_surprise_simd_matches_scalar`.
        #[cfg(feature = "temporal_deriv")]
        self.observe_surprise_inner();
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
    /// HLA update step (proven win). Both paths now use `expand_matvec` for
    /// the expand step (sense_composition feature); `use_simd` only toggles
    /// the HLA evolution kernel.
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
    /// **Note**: Since the matvec path is now numerically equivalent to the
    /// scalar path (both use `crate::simd::fast_sigmoid`), `reconstruct_inner`
    /// dispatches to `expand_matvec` directly when `sense_composition` is on.
    /// This function is kept for explicit callers that want the matvec path
    /// without depending on feature-flag dispatch.
    ///
    /// **GOAT result**: Per-step expand is 1.27× faster than scalar (20.4ns vs 25.9ns).
    /// Full-cycle parity depends on loop overhead — use `expand_with_weights()`
    /// with a pre-computed `ProjectionWeights` for production multi-entity path.
    #[cfg(feature = "sense_composition")]
    pub fn reconstruct_matvec(&mut self, brain: &crate::sense::brain::NpcBrain) -> [f32; 6] {
        // Plan 283 T5.1: previous-step activations for the advantage gate.
        // `None` on the first iteration (no prior to compare).
        #[cfg(feature = "self_advantage_gate")]
        let mut prev_activations: Option<[f32; 6]> = None;

        loop {
            let activations = self.expand_matvec(brain);
            let selected = self.route(&activations);
            self.accumulate(&selected, &activations);
            self.evolve_hla();
            self.step += 1;
            if self.sufficient() {
                return activations;
            }
            // 4th early-stop: advantage-margin gate (dead compute).
            #[cfg(feature = "self_advantage_gate")]
            {
                if self.advantage_gate_halt(prev_activations.as_ref(), &activations) {
                    return activations;
                }
                prev_activations = Some(activations);
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
        // Plan 283 T5.1: previous-step activations for the advantage gate.
        #[cfg(feature = "self_advantage_gate")]
        let mut prev_activations: Option<[f32; 6]> = None;

        loop {
            let activations = self.expand_with_weights(weights);
            let selected = self.route(&activations);
            self.accumulate(&selected, &activations);
            self.evolve_hla();
            self.step += 1;
            if self.sufficient() {
                return activations;
            }
            // 4th early-stop: advantage-margin gate (dead compute).
            #[cfg(feature = "self_advantage_gate")]
            {
                if self.advantage_gate_halt(prev_activations.as_ref(), &activations) {
                    return activations;
                }
                prev_activations = Some(activations);
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
        let use_simd = self.config.simd_beneficial();
        self.reconstruct_inner(brain, use_simd)
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

        // Plan 283 T5.1: previous-step activations for the advantage gate.
        #[cfg(feature = "self_advantage_gate")]
        let mut prev_activations: Option<[f32; 6]> = None;

        loop {
            // Expand + route. When `sense_composition` is enabled, use the
            // matvec path: weights are cached on first call (one-shot
            // `simd_matmul_rows` for all 6 modules) and reused for every
            // subsequent step. Benchmarks show ~1.27× per-step expand speedup
            // (20.4ns vs 25.9ns). Numerically equivalent to scalar `expand`
            // since both paths now use `crate::simd::fast_sigmoid`.
            //
            // Constraint: a `ReconstructionState` is bound to one brain
            // configuration once `expand_matvec` is called (it caches the
            // weight matrix). Different brains require different states.
            #[cfg(feature = "sense_composition")]
            let activations = self.expand_matvec(brain);
            #[cfg(not(feature = "sense_composition"))]
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

            // 4th early-stop: advantage-margin gate (dead compute).
            // Halts if the current step did not improve the prediction for
            // the top-routed module above the population average.
            #[cfg(feature = "self_advantage_gate")]
            {
                if self.advantage_gate_halt(prev_activations.as_ref(), &activations) {
                    return activations;
                }
                prev_activations = Some(activations);
            }
        }
    }
}

// ── Plan 283 T5.1: Self-advantage gate helpers ─────────────────
// Minimal inline of the canonical advantage-margin math (root crate
// `src/pruners/self_advantage.rs`). Kept here because katgpt-core cannot
// depend on the root crate. Mathematically equivalent; the canonical
// version is the tested, benchmarked reference.

/// Argmax of a 6-element array — used to select the gate's candidate
/// (the top-routed module). Unrolled for speed.
#[cfg(feature = "self_advantage_gate")]
#[inline]
fn argmax6(v: &[f32; 6]) -> usize {
    let mut idx = 0usize;
    let mut max = v[0];
    if v[1] > max {
        max = v[1];
        idx = 1;
    }
    if v[2] > max {
        max = v[2];
        idx = 2;
    }
    if v[3] > max {
        max = v[3];
        idx = 3;
    }
    if v[4] > max {
        max = v[4];
        idx = 4;
    }
    if v[5] > max {
        idx = 5;
    }
    idx
}

/// Advantage margin for the HLA reconstruction gate (Eq. 18, arxiv:2511.16886).
///
/// `margin(candidate) = A(candidate) − E_{a∼π+}[A(a)]`
///
/// where `A(a) = log π+(a) − log π̂(a)` is the self-advantage of action `a`,
/// `π̂` is the pre-step distribution (`prev`), and `π+` is the post-step
/// distribution (`curr`). The expectation is under `π+` and equals
/// `KL(π+ ‖ π̂)` by the standard identity.
///
/// Inputs are treated as logits (module activations are sigmoid-bounded
/// `[0, 1]`; the advantage math is invariant to absolute scale — it
/// measures relative shifts between steps).
///
/// # Scratch layout
///
/// `[pre_lsm(6) | post_lsm(6) | advantage(6)]` = 18 f32s. Written and read
/// within this function — no allocation.
#[cfg(feature = "self_advantage_gate")]
#[inline]
fn advantage_margin_hla(prev: &[f32; 6], curr: &[f32; 6], candidate: usize, scratch: &mut [f32; 18]) -> f32 {
    debug_assert!(candidate < 6);

    // Scratch layout: [pre_lsm(6) | post_lsm(6) | advantage(6)] = 18 f32s.
    // Use split_at_mut to get non-overlapping mutable borrows.
    let (pre_lsm, rest) = scratch.split_at_mut(6);
    let (post_lsm, adv) = rest.split_at_mut(6);

    // log-softmax for pre and post (max-subtracted for numerical stability).
    log_softmax_into6(prev, pre_lsm);
    log_softmax_into6(curr, post_lsm);

    // Advantage: A(a) = post_lsm(a) − pre_lsm(a).
    adv[0] = post_lsm[0] - pre_lsm[0];
    adv[1] = post_lsm[1] - pre_lsm[1];
    adv[2] = post_lsm[2] - pre_lsm[2];
    adv[3] = post_lsm[3] - pre_lsm[3];
    adv[4] = post_lsm[4] - pre_lsm[4];
    adv[5] = post_lsm[5] - pre_lsm[5];

    // E_{a∼π+}[A(a)] = Σ_a π+(a) · A(a) = Σ_a exp(post_lsm[a]) · adv[a] = KL(π+‖π̂).
    let expectation = post_lsm[0].exp() * adv[0]
        + post_lsm[1].exp() * adv[1]
        + post_lsm[2].exp() * adv[2]
        + post_lsm[3].exp() * adv[3]
        + post_lsm[4].exp() * adv[4]
        + post_lsm[5].exp() * adv[5];

    adv[candidate] - expectation
}

/// In-place log-softmax for a 6-element vector.
///
/// Writes `log_softmax(x)[i] = x[i] − logsumexp(x)` into `out`.
/// Max-subtracted for numerical stability. Both slices must have length 6.
#[cfg(feature = "self_advantage_gate")]
#[inline]
fn log_softmax_into6(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), 6);
    debug_assert_eq!(out.len(), 6);

    let mut max_val = x[0];
    if x[1] > max_val {
        max_val = x[1];
    }
    if x[2] > max_val {
        max_val = x[2];
    }
    if x[3] > max_val {
        max_val = x[3];
    }
    if x[4] > max_val {
        max_val = x[4];
    }
    if x[5] > max_val {
        max_val = x[5];
    }

    // Σ exp(x[i] − max)
    let lse = (x[0] - max_val).exp()
        + (x[1] - max_val).exp()
        + (x[2] - max_val).exp()
        + (x[3] - max_val).exp()
        + (x[4] - max_val).exp()
        + (x[5] - max_val).exp();
    let log_lse = lse.ln();

    out[0] = x[0] - max_val - log_lse;
    out[1] = x[1] - max_val - log_lse;
    out[2] = x[2] - max_val - log_lse;
    out[3] = x[3] - max_val - log_lse;
    out[4] = x[4] - max_val - log_lse;
    out[5] = x[5] - max_val - log_lse;
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
    let hla_delta = {
        let mut d = [0.0f32; 8];
        let evolved = state.hla();
        for i in 0..8 {
            d[i] = evolved[i] - hla[i];
        }
        d
    };

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

    /// Plan 277 T2.7 — surprise channel numerical equivalence between scalar
    /// `evolve_hla` and SIMD `evolve_hla_simd` paths.
    ///
    /// Both paths feed the same HLA into the `TemporalDerivativeKernel<8>` via
    /// `observe_surprise_inner()`. Since the HLA itself is numerically
    /// equivalent (asserted by `evolve_hla_simd_matches_scalar`) and the
    /// surprise kernel is deterministic given its input, the surprise vectors
    /// and norms MUST match to within f32 rounding tolerance.
    ///
    /// Guards against a regression where someone edits only one path's surprise
    /// observe call (e.g. moves it before the leaky step, or removes it).
    /// Requires both `sense_composition` (for the SIMD path) and
    /// `temporal_deriv` (for the surprise channel).
    #[cfg(all(feature = "sense_composition", feature = "temporal_deriv"))]
    #[test]
    fn evolve_hla_surprise_simd_matches_scalar() {
        let config = ReconstructionConfig::default();
        let hla = [0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8];
        let selected = [true, false, true, false, true, false];
        let activations = [0.5, 0.2, 0.8, 0.1, 0.3, 0.0];

        // Run 10 ticks on each path so the surprise kernel builds up nonzero
        // (fast − slow) state — a single tick would only test the initial
        // transient, not the steady-state divergence robustness.
        let mut state_scalar = ReconstructionState::with_config(hla, config);
        let mut state_simd = ReconstructionState::with_config(hla, config);
        for tick in 0..10u32 {
            // Vary activations each tick to produce a non-trivial trajectory.
            let scale = 0.5 + 0.1 * tick as f32;
            let acts = [
                activations[0] * scale,
                activations[1] * scale,
                activations[2] * scale,
                activations[3] * scale,
                activations[4] * scale,
                activations[5] * scale,
            ];
            state_scalar.accumulate(&selected, &acts);
            state_scalar.evolve_hla();
            state_simd.accumulate(&selected, &acts);
            state_simd.evolve_hla_simd();
        }

        // Compare the surprise vectors.
        let sv_scalar = state_scalar
            .surprise_vector()
            .expect("temporal_deriv on → surprise_vector is Some");
        let sv_simd = state_simd
            .surprise_vector()
            .expect("temporal_deriv on → surprise_vector is Some");
        let mut max_vec_diff = 0.0f32;
        for i in 0..8 {
            let diff = (sv_scalar[i] - sv_simd[i]).abs();
            max_vec_diff = max_vec_diff.max(diff);
        }
        assert!(
            max_vec_diff < 1e-5,
            "surprise vector scalar vs SIMD diverged: max_diff={max_vec_diff:e}"
        );

        // Compare the surprise norms.
        let n_scalar = state_scalar.surprise_norm();
        let n_simd = state_simd.surprise_norm();
        let norm_diff = (n_scalar - n_simd).abs();
        assert!(
            norm_diff < 1e-5,
            "surprise_norm scalar vs SIMD diverged: scalar={n_scalar:e}, simd={n_simd:e}, diff={norm_diff:e}"
        );
    }

    /// Plan 277 Fusion F1 T2.6 — G2 gate: synthetic emotional-event trace.
    ///
    /// Embeds three step-change events into a 1000-tick HLA trace and verifies
    /// the dual `(fast − slow)` surprise channel detects them while the raw
    /// HLA L2 norm does not. This is the core G2 proof that the derivative is
    /// an *orthogonal* signal to magnitude — it fires on change, not on level.
    ///
    /// Events (combat onset, loot drop, encounter) are injected as additive
    /// deltas on one HLA dimension at t=200, 500, 800 via `inject_hla_delta`.
    /// Between events the trace is held stationary so the EMAs converge and
    /// the derivative decays back to ~0 — the canonical neocortical
    /// prediction-error shape (O'Reilly 2026).
    ///
    /// Gates:
    /// - Recall ≥ 80%: every event window (±10 ticks) contains a local
    ///   `surprise_norm` peak. 3/3 events → 100% expected.
    /// - False positives ≤ 10%: fewer than 10% of trace ticks outside event
    ///   windows may be local peaks above the detection threshold.
    /// - Raw vs derivative orthogonality: the raw HLA L2 norm is monotone
    ///   non-decreasing across each event (it steps up and stays), so it CANNOT
    ///   peak inside the event window — the derivative peaks where raw does not.
    #[cfg(feature = "temporal_deriv")]
    #[test]
    fn surprise_detects_emotional_events_g2_gate() {
        // --- Trace configuration -------------------------------------------------
        const TRACE_LEN: usize = 1000;
        const EVENTS: [usize; 3] = [200, 500, 800];
        const WINDOW: usize = 10; // ±ticks around each event
        // Detection threshold: a tick is a "peak" if surprise_norm exceeds this
        // AND is a local maximum vs its immediate neighbors. Calibrated to sit
        // well above the converged-stationary floor and well below the event
        // spike magnitude given α_fast=0.3, α_slow=0.03.
        const DETECT_THRESHOLD: f32 = 0.05;
        const RECALL_MIN: f32 = 0.80; // ≥80% per the G2 spec
        const FP_MAX_FRAC: f32 = 0.10; // ≤10% false positives

        let config = ReconstructionConfig::default();
        let mut state = ReconstructionState::with_config([0.0; 8], config);

        // Per-tick recordings.
        let mut surprise_trace = [0.0f32; TRACE_LEN];
        let mut raw_norm_trace = [0.0f32; TRACE_LEN];

        for t in 0..TRACE_LEN {
            // Inject scripted event deltas exactly at the event tick.
            // Combat onset: dim 0 jumps +0.6 (arousal).
            // Loot drop:    dim 1 jumps +0.4 (valence).
            // Encounter:    dim 2 jumps +0.5 (social attention).
            let delta = match t {
                200 => [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                500 => [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                800 => [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                _ => [0.0; 8],
            };
            if delta != [0.0; 8] {
                state.inject_hla_delta(delta);
            }

            // Drive a no-op evolve_hla tick: zero evidence → leaky_step does
            // nothing to HLA, but the surprise kernel still observes the
            // (possibly just-injected) current HLA. This is exactly the scalar
            // path's documented behavior (observe runs even on a no-op step).
            state.evolve_hla();

            // Record surprise and raw HLA L2 norm.
            surprise_trace[t] = state.surprise_norm();
            let hla = state.hla();
            let sq: f32 = hla.iter().map(|x| x * x).sum();
            raw_norm_trace[t] = sq.max(0.0).sqrt();
        }

        // --- Detect local peaks in surprise_trace -------------------------------
        // A peak at t requires surprise_trace[t] > threshold AND ≥ both
        // neighbors (clamped at trace ends). We then count peaks that fall
        // inside any event window vs outside.
        let mut peaks_outside: usize = 0;
        let mut surprise_peak_ticks: Vec<usize> = Vec::new();
        for t in 0..TRACE_LEN {
            if surprise_trace[t] <= DETECT_THRESHOLD {
                continue;
            }
            let prev = if t == 0 { f32::MIN } else { surprise_trace[t - 1] };
            let next = if t + 1 == TRACE_LEN {
                f32::MIN
            } else {
                surprise_trace[t + 1]
            };
            if surprise_trace[t] >= prev && surprise_trace[t] >= next {
                surprise_peak_ticks.push(t);
                if !EVENTS.iter().any(|&e| t.abs_diff(e) <= WINDOW) {
                    peaks_outside += 1;
                }
            }
        }

        // --- G2 gate: recall ----------------------------------------------------
        // Count how many distinct event windows contain ≥1 peak.
        let mut events_hit = 0usize;
        for &e in &EVENTS {
            let lo = e.saturating_sub(WINDOW);
            let hi = (e + WINDOW).min(TRACE_LEN - 1);
            if (lo..=hi).any(|t| surprise_peak_ticks.contains(&t)) {
                events_hit += 1;
            }
        }
        let recall = events_hit as f32 / EVENTS.len() as f32;
        assert!(
            recall >= RECALL_MIN,
            "G2 recall {recall:.2} < {RECALL_MIN}: hit {events_hit}/{} events; peaks at {:?}",
            EVENTS.len(),
            surprise_peak_ticks
        );

        // --- G2 gate: false positives -------------------------------------------
        // Out-of-window peaks must be < 10% of trace length.
        let fp_frac = peaks_outside as f32 / TRACE_LEN as f32;
        assert!(
            fp_frac <= FP_MAX_FRAC,
            "G2 false-positive fraction {fp_frac:.3} > {FP_MAX_FRAC}: {peaks_outside} out-of-window peaks at {:?}",
            surprise_peak_ticks
                .iter()
                .copied()
                .filter(|t| !EVENTS.iter().any(|e| t.abs_diff(*e) <= WINDOW))
                .collect::<Vec<_>>()
        );

        // --- G2 proof: raw HLA norm does NOT peak at events ---------------------
        // The raw norm is monotone non-decreasing by construction (each event
        // only adds magnitude and nothing subtracts). Therefore the global raw
        // peak is at the last tick, far from any event. We assert:
        //   (a) raw norm at the last tick ≥ raw norm at every event tick, and
        //   (b) the surprise global peak IS near an event (within WINDOW of
        //       some event), proving the two signals peak at different places.
        let raw_argmax = (0..TRACE_LEN)
            .max_by(|&a, &b| {
                raw_norm_trace[a]
                    .partial_cmp(&raw_norm_trace[b])
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .expect("non-empty trace");
        let surprise_argmax = (0..TRACE_LEN)
            .max_by(|&a, &b| {
                surprise_trace[a]
                    .partial_cmp(&surprise_trace[b])
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .expect("non-empty trace");

        // Raw norm peaks at (or near) the last tick — never at an event.
        let raw_near_event = EVENTS
            .iter()
            .any(|&e| raw_argmax.abs_diff(e) <= WINDOW);
        assert!(
            !raw_near_event,
            "G2: raw HLA norm must NOT peak near an event; raw_argmax={raw_argmax}, events={EVENTS:?}"
        );

        // Surprise norm peaks near an event.
        let surprise_near_event = EVENTS
            .iter()
            .any(|&e| surprise_argmax.abs_diff(e) <= WINDOW);
        assert!(
            surprise_near_event,
            "G2: surprise_norm must peak near an event; surprise_argmax={surprise_argmax}, events={EVENTS:?}"
        );

        // Orthogonality: the two argmaxes must be far apart (different places).
        let argmax_gap = raw_argmax.abs_diff(surprise_argmax);
        assert!(
            argmax_gap > WINDOW,
            "G2: surprise peak and raw peak must be >{WINDOW} ticks apart; gap={argmax_gap} (raw={raw_argmax}, surprise={surprise_argmax})"
        );
    }

    /// Plan 276 T2.3 — zero-behavior-change regression gate.
    ///
    /// Runs a known evidence pattern through the refactored `evolve_hla`
    /// (now a delegate over `crate::leaky_core::leaky_step`) and asserts the
    /// resulting HLA is **bit-for-bit identical** to the original inline math
    /// (sum-over-6 total, `KIND_MAP = [0,1,2,3,4,5,0,1]` gather, config-sourced
    /// lr / max_delta). Runs whenever `sense` compiles — `micro_belief` is NOT
    /// required, which proves the ungated core keeps `sense` decoupled.
    #[test]
    fn evolve_hla_is_byte_identical_to_inline_reference() {
        let config = ReconstructionConfig::default();
        let lr = config.hla_learning_rate;
        let max_delta = config.max_hla_delta;
        let init_hla = [0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8];

        // Accumulate a non-trivial evidence pattern with non-zero k0/k1 so the
        // KIND_MAP wrap (dims 6,7) is exercised and the sum-over-6 vs
        // sum-over-8 distinction actually matters.
        let selected = [true, true, true, true, true, true];
        let activations = [0.5, 0.2, 0.8, 0.1, 0.3, 0.4];

        // --- Actual: refactored delegate path ---
        let mut state_actual = ReconstructionState::with_config(init_hla, config);
        state_actual.accumulate(&selected, &activations);
        state_actual.evolve_hla();

        // --- Reference: verbatim copy of the PRE-refactor evolve_hla body ---
        // total is Σ kind_activations[0..6]; gather uses KIND_MAP wrap.
        const KIND_MAP: [usize; 8] = [0, 1, 2, 3, 4, 5, 0, 1];
        let mut kind_activations = [0.0f32; 6];
        for (i, &sel) in selected.iter().enumerate() {
            if sel && activations[i] > 0.0 {
                kind_activations[i] += activations[i];
            }
        }
        let mut hla_ref = init_hla;
        let total_activation: f32 = kind_activations.iter().copied().sum();
        assert!(
            total_activation >= 1e-8,
            "test fixture must accumulate non-trivial evidence"
        );
        let t_min = total_activation.min(1.0);
        let scale = lr * t_min / total_activation;
        let half_total = 0.5 * total_activation;
        for i in 0..8 {
            let normalized = kind_activations[KIND_MAP[i]];
            let delta = scale * (normalized - half_total);
            let clamped_delta = delta.clamp(-max_delta, max_delta);
            hla_ref[i] = (hla_ref[i] + clamped_delta).clamp(-1.0, 1.0);
        }

        // Bit-for-bit equality — not approximate. Any drift is a regression.
        assert_eq!(
            *state_actual.hla(),
            hla_ref,
            "T2.3: refactored evolve_hla must be byte-identical to the inline reference"
        );
    }

    /// Plan 276 T2.1 — the `KIND_MAP` gather helper must produce the wrapped
    /// 8-element layout from the 6 per-kind activations. Single-source-of-truth
    /// guard: if someone edits `TripleEvidence::KIND_MAP`, this catches it.
    #[test]
    fn kind_activations_padded_matches_kind_map() {
        let ev = TripleEvidence {
            kind_activations: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            ..Default::default()
        };
        let padded = ev.kind_activations_padded();
        assert_eq!(padded, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2]);
        assert_eq!(TripleEvidence::KIND_MAP, [0, 1, 2, 3, 4, 5, 0, 1]);
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

    // ── Plan 283 T5.1: self-advantage gate tests ──────────────────

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn argmax6_finds_correct_index() {
        assert_eq!(argmax6(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 5);
        assert_eq!(argmax6(&[0.6, 0.5, 0.4, 0.3, 0.2, 0.1]), 0);
        assert_eq!(argmax6(&[0.1, 0.1, 0.9, 0.1, 0.1, 0.1]), 2);
        // Ties → first occurrence wins (consistent with `>` comparison).
        assert_eq!(argmax6(&[0.5, 0.5, 0.1, 0.1, 0.1, 0.1]), 0);
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn log_softmax_into6_sums_to_one_after_exp() {
        let x = [0.3, 0.7, 0.1, 0.5, 0.4, 0.2];
        let mut out = [0.0f32; 6];
        log_softmax_into6(&x, &mut out);
        let sum: f32 = out.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "exp(log_softmax) must sum to 1, got {sum}");
        for &v in &out {
            assert!(v <= 0.0, "log-prob must be ≤ 0, got {v}");
        }
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn advantage_margin_zero_when_identical() {
        let prev = [0.5, 0.2, 0.8, 0.1, 0.3, 0.4];
        let curr = prev;
        let mut scratch = [0.0f32; 18];
        for candidate in 0..6 {
            let m = advantage_margin_hla(&prev, &curr, candidate, &mut scratch);
            assert!(m.abs() < 1e-6, "identical steps → zero margin, got {m} for candidate {candidate}");
        }
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn advantage_margin_positive_when_post_sharpens_candidate() {
        let prev = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        let curr = [0.1, 0.1, 0.9, 0.1, 0.1, 0.1];
        let mut scratch = [0.0f32; 18];
        let m = advantage_margin_hla(&prev, &curr, 2, &mut scratch);
        assert!(m > 0.0, "sharpening the candidate must give positive margin, got {m}");
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn advantage_margin_negative_when_post_shifts_away_from_candidate() {
        let prev = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1];
        let curr = [0.1, 0.1, 0.9, 0.1, 0.1, 0.1];
        let mut scratch = [0.0f32; 18];
        let m = advantage_margin_hla(&prev, &curr, 0, &mut scratch);
        assert!(m < 0.0, "shifting away from candidate must give negative margin, got {m}");
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn gate_halt_disabled_when_threshold_nan() {
        // Explicit NaN threshold disables the gate (default is now 0.01 after T5.1.4).
        let config = ReconstructionConfig {
            advantage_margin_threshold: f32::NAN,
            ..Default::default()
        };
        let state = ReconstructionState::with_config([0.0; 8], config);
        let prev = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        let curr = [0.1, 0.1, 0.9, 0.1, 0.1, 0.1];
        assert!(!state.advantage_gate_halt(Some(&prev), &curr), "disabled gate never halts");
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn gate_default_threshold_is_0_01() {
        // Plan 283 T5.1.4: promoted to default-on after GOAT gate passed
        // (2.50× steps saved, 100% argmax match, 0ns overhead).
        let config = ReconstructionConfig::default();
        assert!((config.advantage_margin_threshold - 0.01).abs() < 1e-6,
            "default threshold should be 0.01 after T5.1.4 promotion, got {}", config.advantage_margin_threshold);
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn gate_halt_never_on_first_step() {
        let config = ReconstructionConfig {
            advantage_margin_threshold: 0.01,
            ..Default::default()
        };
        let state = ReconstructionState::with_config([0.0; 8], config);
        let curr = [0.1, 0.1, 0.9, 0.1, 0.1, 0.1];
        assert!(!state.advantage_gate_halt(None, &curr), "first step (prev=None) never halts");
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn gate_halt_when_dead_compute_detected() {
        let config = ReconstructionConfig {
            advantage_margin_threshold: 0.01,
            ..Default::default()
        };
        let state = ReconstructionState::with_config([0.0; 8], config);
        let prev = [0.5, 0.2, 0.8, 0.1, 0.3, 0.4];
        let curr = prev;
        assert!(state.advantage_gate_halt(Some(&prev), &curr), "identical steps must trigger halt");
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn gate_no_halt_when_candidate_improves() {
        let config = ReconstructionConfig {
            advantage_margin_threshold: 0.01,
            ..Default::default()
        };
        let state = ReconstructionState::with_config([0.0; 8], config);
        let prev = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        let curr = [0.1, 0.1, 0.9, 0.1, 0.1, 0.1];
        assert!(!state.advantage_gate_halt(Some(&prev), &curr), "improving step must not halt");
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn reconstruct_with_gate_enabled_runs_to_completion() {
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
            max_steps: 5,
            advantage_margin_threshold: 0.01,
            ..Default::default()
        };
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let result = state.reconstruct(&brain);

        for &a in &result {
            assert!(a.is_finite(), "gate-enabled reconstruction must produce finite activations, got {a}");
        }
        assert!(state.step() > 0, "should have taken at least 1 step");
        assert!(state.step() <= 5, "should not exceed max_steps");
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn gate_on_preserves_argmax_vs_disabled() {
        // Plan 283 T5.1.4 GOAT gate G2: gate ON (default 0.01) must produce
        // the same argmax as gate OFF (NaN), even though it may halt earlier.
        // This is the quality guarantee that makes the gate safe to default-on.
        use crate::sense::brain::NpcBrain;
        use crate::types::{SenseKind, SenseModule, TernaryDir};

        let mut m1 = SenseModule {
            kind: SenseKind::CommonSense,
            confidence: 0.7,
            n_directions: 8,
            directions: {
                let mut dirs = [TernaryDir::zero(); 8];
                dirs[0] = TernaryDir {
                    pos_bits: 0x01,
                    neg_bits: 0x04,
                    row_scale: 0.5,
                };
                dirs[1] = TernaryDir {
                    pos_bits: 0x02,
                    neg_bits: 0x08,
                    row_scale: 0.5,
                };
                dirs
            },
            ..Default::default()
        };
        m1.commit();

        let brain = NpcBrain::compose(vec![m1]);

        // Gate OFF (NaN).
        let config_off = ReconstructionConfig {
            advantage_margin_threshold: f32::NAN,
            ..Default::default()
        };
        let mut state_off = ReconstructionState::with_config(brain.hla_state, config_off);
        let result_off = state_off.reconstruct(&brain);

        // Gate ON (default 0.01).
        let mut state_on = ReconstructionState::with_config(brain.hla_state, ReconstructionConfig::default());
        let result_on = state_on.reconstruct(&brain);

        // Argmax must match (quality guarantee).
        let argmax_off = result_off.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let argmax_on = result_on.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        assert_eq!(argmax_off, argmax_on,
            "gate ON must preserve argmax vs OFF (quality guarantee)");
    }
}
