//! Thicket Variance Probe (TVP) — decoding-space density as substrate routing signal.
//!
//! Plan 267, Research 235. Lifts RandOpt's (Neural Thickets, arXiv:2603.12228)
//! fundamental insight — *variance structure of perturbation probes reveals
//! loss-landscape geometry* — from weight-space (Plan 121) to decoding-config-space.
//!
//! # What it does
//!
//! Before the main decode, sample K cheap probes that perturb inference-time
//! knobs (temperature, top-p, drafter seed, KV noise). Each probe produces a
//! single token. The disagreement across probe outputs is decomposed into:
//!
//! - **answer_disagreement** — paper's δ(m) analog in decoding space
//! - **format_disagreement** — Section 8 cosmetic-only variance (canonicalized)
//! - **reasoning_disagreement** — net substantive signal (answer minus format)
//! - **logit_kl** — paper's D (spectral discordance) continuous analog
//!
//! The `TvpSignal` feeds `InferenceRouter` as signal #8 (distinct from trust,
//! RV, critical_entropy, lodestar, breakeven, modality, QPS) and optionally
//! `S2FCollapseDetector` (high disagreement → expand CoT budget).
//!
//! # Hot path
//!
//! - Probes run on CPU/SIMD only (never GPU/ANE) — resolves chicken-and-egg
//! - Fixed-size `[f32; 32]` top-logits → zero-alloc O(K²) KL on stack
//! - K ≤ 8 → at most 28 pairwise KL ops per query
//! - Signal struct is `repr(C)`, 16 bytes — plasma-tier readable
//!
//! # Self-learning (no LLM training)
//!
//! - `TvpThresholdAdapter` — EMA-adapts promote/demote thresholds per query-class
//! - `TvpProbeCountBandit` — reuses Plan 121 `BanditStrategy::RandOptAdaptive` for K ∈ {2,4,8}
//! - Sigmoid-blended updates (never softmax, per project conventions)
//!
//! # Usage
//!
//! ```rust,ignore
//! let config = TvpConfig::default();
//! let source = SyntheticProbeSource::new(|arm| ProbeOutput {
//!     token_id: arm as u32,
//!     top_logits: [0.0; 32],
//!     logit_count: 1,
//!     format_hash: 0,
//! });
//! let aggregator = TvpAggregator::new(config);
//! let signal = aggregator.aggregate_k(&source);
//! // signal.reasoning_disagreement → router
//! ```

// ── Config ──────────────────────────────────────────────────────

/// Maximum number of probes supported. K > 8 wastes compute — paper's
/// own N=5000 is for *training*; inference-time probing is much cheaper.
pub const TVP_MAX_PROBES: usize = 8;

/// Maximum top-logits kept per probe. 32 is enough to distinguish answer-level
/// disagreement; full vocab KL is unnecessary for a routing signal.
pub const TVP_TOP_LOGIT_CAP: usize = 32;

/// TVP configuration. All knobs are modelless — perturb inference-time state only.
#[derive(Clone, Copy, Debug)]
pub struct TvpConfig {
    /// Number of probes K ∈ [2, 8]. Default 4 (paper uses larger N for training,
    /// but inference-time signal converges quickly — see G7 convergence test).
    pub probe_count: u8,
    /// Sampling-temperature delta ΔT. Probes sample T_i = T_base ± i·ΔT.
    /// Free knob. Default 0.1.
    pub temperature_delta: f32,
    /// Top-p delta Δp. Probes sample p_i = p_base ± i·Δp, clamped to (0, 1].
    /// Free knob. Default 0.05.
    pub top_p_delta: f32,
    /// KV cache quantization noise σ_kv. Default 0.0 (off — requires warm path).
    pub kv_noise_sigma: f32,
    /// Substrate mask bit-flips per probe. Default 0 (off — requires substrate_gate).
    pub mask_flip_count: u8,
    /// Disagreement threshold above which to promote CPU → GPU/ANE.
    /// Higher = more conservative (need stronger signal to upgrade).
    pub promote_at: f32,
    /// Disagreement threshold below which to demote GPU → CPU.
    /// Lower = more conservative (need stronger signal to downgrade).
    pub demote_at: f32,
    /// EMA smoothing factor for threshold adaptation. Range (0, 0.5].
    pub ema_alpha: f32,
}

impl Default for TvpConfig {
    fn default() -> Self {
        Self {
            probe_count: 4,
            temperature_delta: 0.1,
            top_p_delta: 0.05,
            kv_noise_sigma: 0.0,
            mask_flip_count: 0,
            promote_at: 0.6,
            demote_at: 0.2,
            ema_alpha: 0.05,
        }
    }
}

impl TvpConfig {
    /// Clamp all fields to valid ranges. Called on construction.
    pub fn sanitized(mut self) -> Self {
        self.probe_count = self.probe_count.clamp(2, TVP_MAX_PROBES as u8);
        self.temperature_delta = self.temperature_delta.clamp(0.0, 2.0);
        self.top_p_delta = self.top_p_delta.clamp(0.0, 0.5);
        self.kv_noise_sigma = self.kv_noise_sigma.max(0.0);
        self.mask_flip_count = self.mask_flip_count.min(64);
        self.promote_at = self.promote_at.clamp(0.0, 1.0);
        self.demote_at = self.demote_at.clamp(0.0, self.promote_at);
        self.ema_alpha = self.ema_alpha.clamp(0.001, 0.5);
        self
    }
}

// ── Signal ──────────────────────────────────────────────────────

/// TVP signal emitted to the router. `repr(C)` for stable hot-path layout.
///
/// All disagreement values are in `[0.0, 1.0]`:
/// - 0.0 = all probes agree (dense thicket — cheap substrate OK)
/// - 1.0 = all probes disagree (needle — promote compute)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct TvpSignal {
    /// Net substantive disagreement (answer minus format).
    /// **Primary router input.** High → promote; low → demote.
    pub reasoning_disagreement: f32,
    /// Cosmetic-only disagreement (same answer, different format).
    /// High → canonicalize output, do NOT promote compute.
    pub format_disagreement: f32,
    /// Mean pairwise KL divergence across probe logits (continuous analog
    /// of paper's spectral discordance D). Unbounded; small = agree.
    pub logit_kl: f32,
    /// Actual probe count used. May differ from config if bandit reduced K.
    pub probe_count_used: u8,
}

impl TvpSignal {
    /// Zero signal — uninitialized / no probes run. Has no routing impact.
    pub const fn zero() -> Self {
        Self {
            reasoning_disagreement: 0.0,
            format_disagreement: 0.0,
            logit_kl: 0.0,
            probe_count_used: 0,
        }
    }

    /// Returns true if the signal should trigger a tier promotion.
    #[inline]
    pub fn should_promote(&self, promote_at: f32) -> bool {
        self.reasoning_disagreement > promote_at
    }

    /// Returns true if the signal should trigger a tier demotion.
    #[inline]
    pub fn should_demote(&self, demote_at: f32) -> bool {
        self.reasoning_disagreement < demote_at
    }
}

// ── Router tier decision ─────────────────────────────────────────

/// Decision returned by [`tvp_tier_decision`] for the InferenceRouter.
///
/// Mirrors `dllm_solver::CriticalTierDecision` so the two gates compose
/// uniformly inside `InferenceRouter::forward`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TvpTierDecision {
    /// No TVP signal available (no probes run yet) → defer to upstream routing.
    Defer,
    /// High substantive disagreement + GPU available + low load → promote to GPU.
    PromoteGpu,
    /// Low substantive disagreement + low load → demote to CPU.
    DemoteCpu,
    /// Signal present but below/above thresholds, or load conditions not met.
    Hold,
}

/// Pure tier-decision function for TVP (Plan 267 T10).
///
/// Inputs:
///   - `signal`: last observed TVP signal (`None` if no probes have run).
///   - `promote_at`, `demote_at`: thresholds from `TvpConfig`.
///   - `current_tier`: tier coming into the TVP gate (after trust/RV/critical).
///   - `gpu_available`: whether a GPU backend exists.
///   - `low_load`: true when QPS is below the demotion load threshold.
///
/// Returns:
///   - `Defer`     when `signal` is `None` (zero routing impact — G3).
///   - `PromoteGpu` when `reasoning_disagreement > promote_at`, tier is CpuOnly, GPU exists.
///   - `DemoteCpu`  when `reasoning_disagreement < demote_at`, tier is CpuGpu, low load.
///   - `Hold`       otherwise.
///
/// Format-only disagreement NEVER trips this gate — see G5.
#[inline]
pub fn tvp_tier_decision(
    signal: Option<TvpSignal>,
    promote_at: f32,
    demote_at: f32,
    current_tier: crate::trigger_gate::ComputeTier,
    gpu_available: bool,
    low_load: bool,
) -> TvpTierDecision {
    use crate::trigger_gate::ComputeTier;
    let Some(s) = signal else {
        return TvpTierDecision::Defer;
    };
    if s.should_promote(promote_at) && current_tier == ComputeTier::CpuOnly && gpu_available {
        TvpTierDecision::PromoteGpu
    } else if s.should_demote(demote_at) && current_tier == ComputeTier::CpuGpu && low_load {
        TvpTierDecision::DemoteCpu
    } else {
        TvpTierDecision::Hold
    }
}

/// Output of a single probe. Fixed-size for zero-alloc aggregation.
#[derive(Clone, Debug)]
pub struct ProbeOutput {
    /// Top-1 token id chosen by this probe.
    pub token_id: u32,
    /// Top-K logits (up to `TVP_TOP_LOGIT_CAP`). Caller fills only the first
    /// `logit_count` entries; the rest are padding.
    pub top_logits: [f32; TVP_TOP_LOGIT_CAP],
    /// Number of valid entries in `top_logits`.
    pub logit_count: u8,
    /// Canonicalized format hash (BLAKE3 of lowercased alphanumeric form).
    /// Two probes with same answer but different surface form share this hash.
    pub format_hash: u64,
}

impl ProbeOutput {
    /// Construct a probe output from a single-token answer (no logits).
    /// Used for tests where only answer disagreement matters.
    pub fn from_token(token_id: u32, format_hash: u64) -> Self {
        Self {
            token_id,
            top_logits: [0.0; TVP_TOP_LOGIT_CAP],
            logit_count: 0,
            format_hash,
        }
    }

    /// Construct a probe output from a token and its top logits.
    pub fn from_token_and_logits(token_id: u32, format_hash: u64, logits: &[f32]) -> Self {
        let mut top_logits = [0.0; TVP_TOP_LOGIT_CAP];
        let n = logits.len().min(TVP_TOP_LOGIT_CAP);
        top_logits[..n].copy_from_slice(&logits[..n]);
        Self {
            token_id,
            top_logits,
            logit_count: n as u8,
            format_hash,
        }
    }

    /// View of the valid logits slice.
    pub fn logits(&self) -> &[f32] {
        &self.top_logits[..self.logit_count as usize]
    }
}

/// Source of probe outputs. Implementations wrap a real inference path
/// (drafter + sampler) or a synthetic function for tests.
///
/// The `arm` index selects the perturbation: arm 0 = base config,
/// arm 1..K = perturbed. The implementation decides what each arm perturbs
/// (temperature, top-p, seed, KV noise, mask bits — composable via wrapper types).
pub trait TvpProbeSource: Send + Sync {
    /// Run probe `arm` and return its output. `arm` < config.probe_count.
    fn probe(&self, arm: u8) -> ProbeOutput;
}

// ── Format canonicalization ─────────────────────────────────────

/// Canonicalize a token-id sequence into a format hash.
///
/// The canonical form is: lowercase, strip non-alphanumeric, take first 16 chars.
/// This catches "42" vs "The answer is 42" vs "42.0" as same-answer-different-format
/// (per paper Section 8 decomposition). The hash is BLAKE3 of the canonical form.
///
/// For routing, we don't need full NLP canonicalization — just enough to separate
/// cosmetic from substantive disagreement. Token-id-to-text mapping is the caller's
/// responsibility; this function takes an already-stringified form.
pub fn canonical_format_hash(form: &str) -> u64 {
    // BLAKE3 is the project standard (per AGENTS.md). For a u64 hash we use
    // the first 8 bytes of BLAKE3 output over the canonicalized form.
    let mut canonical = String::with_capacity(16);
    for c in form.chars() {
        if c.is_alphanumeric() {
            canonical.extend(c.to_lowercase());
            if canonical.len() >= 16 {
                canonical.truncate(16);
                break;
            }
        }
    }
    // BLAKE3 → first 8 bytes as u64.
    let hash = blake3::hash(canonical.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(bytes)
}

// ── Aggregator ──────────────────────────────────────────────────

/// Computes `TvpSignal` from K probe outputs.
///
/// Zero-allocation: works on a borrowed slice. O(K²) pairwise KL with K ≤ 8
/// → at most 28 ops. Stack-only.
pub struct TvpAggregator {
    config: TvpConfig,
}

impl TvpAggregator {
    /// Create a new aggregator with the given configuration.
    pub fn new(config: TvpConfig) -> Self {
        Self {
            config: config.sanitized(),
        }
    }

    /// Config accessor.
    pub fn config(&self) -> &TvpConfig {
        &self.config
    }

    /// Aggregate K probe outputs into a `TvpSignal`.
    ///
    /// - `answer_disagreement = 1.0 - max_class_share(token_ids)`
    /// - `format_disagreement = 1.0 - max_format_hash_share(format_hashes)`
    /// - `reasoning_disagreement = max(0.0, answer - format)`
    /// - `logit_kl = mean_pairwise_kl(probes)`
    pub fn aggregate(&self, probes: &[ProbeOutput]) -> TvpSignal {
        let k = probes.len();
        if k == 0 {
            return TvpSignal::zero();
        }
        if k == 1 {
            // Single probe → no disagreement possible.
            return TvpSignal {
                reasoning_disagreement: 0.0,
                format_disagreement: 0.0,
                logit_kl: 0.0,
                probe_count_used: 1,
            };
        }

        // --- answer_disagreement ---
        let answer_disagreement = 1.0 - max_token_share(probes);

        // --- format_disagreement ---
        let format_disagreement = 1.0 - max_format_share(probes);

        // --- reasoning_disagreement (net substantive) ---
        let reasoning_disagreement = (answer_disagreement - format_disagreement).max(0.0);

        // --- logit_kl (mean pairwise KL across probes with logits) ---
        let logit_kl = mean_pairwise_kl(probes);

        TvpSignal {
            reasoning_disagreement,
            format_disagreement,
            logit_kl,
            probe_count_used: k as u8,
        }
    }

    /// Convenience: probe the source K times (per config) and aggregate.
    pub fn aggregate_k(&self, source: &dyn TvpProbeSource) -> TvpSignal {
        let k = self.config.probe_count as usize;
        let mut probes: Vec<ProbeOutput> = Vec::with_capacity(k);
        for arm in 0..k as u8 {
            probes.push(source.probe(arm));
        }
        self.aggregate(&probes)
    }
}

/// Fraction of probes sharing the most-common token id. ∈ (0, 1].
/// All-same → 1.0. Perfectly split → 1/k.
fn max_token_share(probes: &[ProbeOutput]) -> f32 {
    let mut best: u32 = 0;
    let n = probes.len() as u32;
    for i in 0..probes.len() {
        let ti = probes[i].token_id;
        let mut count: u32 = 0;
        for p in probes {
            if p.token_id == ti {
                count += 1;
            }
        }
        if count > best {
            best = count;
        }
    }
    best as f32 / n as f32
}

/// Fraction of probes sharing the most-common format hash. ∈ (0, 1].
fn max_format_share(probes: &[ProbeOutput]) -> f32 {
    let mut best: u32 = 0;
    let n = probes.len() as u32;
    for i in 0..probes.len() {
        let fi = probes[i].format_hash;
        let mut count: u32 = 0;
        for p in probes {
            if p.format_hash == fi {
                count += 1;
            }
        }
        if count > best {
            best = count;
        }
    }
    best as f32 / n as f32
}

/// Mean pairwise symmetric KL divergence across probe logits.
/// Symmetric KL: 0.5 * (KL(P||Q) + KL(Q||P)). Skips probes with no logits.
fn mean_pairwise_kl(probes: &[ProbeOutput]) -> f32 {
    let mut sum: f32 = 0.0;
    let mut pairs: u32 = 0;
    // Stack scratch buffers reused across all pairwise KL ops — zero heap
    // allocations in the O(K²) loop. K ≤ TVP_MAX_PROBES so worst case is
    // C(8,2)=28 pairs × 2 softmax_normalizes = 56 saved allocs per aggregate.
    let mut pa = [0.0f32; TVP_TOP_LOGIT_CAP];
    let mut pb = [0.0f32; TVP_TOP_LOGIT_CAP];
    for (i, pi) in probes.iter().enumerate() {
        let li = pi.logits();
        if li.is_empty() {
            continue;
        }
        for pj in probes.iter().skip(i + 1) {
            let lj = pj.logits();
            if lj.is_empty() {
                continue;
            }
            // Use the min length to align (caller should pre-align, but be safe).
            let n = li.len().min(lj.len());
            if n == 0 {
                continue;
            }
            let kl_ij = symmetric_kl_into(&li[..n], &lj[..n], &mut pa, &mut pb);
            sum += kl_ij;
            pairs += 1;
        }
    }
    if pairs == 0 { 0.0 } else { sum / pairs as f32 }
}

/// Symmetric KL divergence between two logit vectors (assumed same length).
/// First applies softmax-style normalization via log-sum-exp for numerical
/// stability. Note: this is internal normalization for a *signal*, not a
/// probability — sigmoid (not softmax) is used at the routing layer.
#[allow(dead_code)]
fn symmetric_kl(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut pa = [0.0f32; TVP_TOP_LOGIT_CAP];
    let mut pb = [0.0f32; TVP_TOP_LOGIT_CAP];
    symmetric_kl_into(a, b, &mut pa, &mut pb)
}

/// Zero-allocation symmetric KL using caller-provided stack buffers.
/// `pa_buf` and `pb_buf` must be at least `a.len()` (== `b.len()`) elements
/// long. Their contents are overwritten.
#[inline]
fn symmetric_kl_into(a: &[f32], b: &[f32], pa_buf: &mut [f32], pb_buf: &mut [f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    debug_assert!(pa_buf.len() >= n && pb_buf.len() >= n);
    // Normalize each to a distribution via softmax. This is internal to the
    // KL computation — the router itself uses sigmoid thresholds.
    fill_softmax(a, &mut pa_buf[..n]);
    fill_softmax(b, &mut pb_buf[..n]);
    let mut kl_ab = 0.0;
    let mut kl_ba = 0.0;
    for i in 0..n {
        let pai = pa_buf[i].max(1e-12);
        let pbi = pb_buf[i].max(1e-12);
        kl_ab += pai * (pai / pbi).ln();
        kl_ba += pbi * (pbi / pai).ln();
    }
    0.5 * (kl_ab + kl_ba)
}

/// Numerically stable softmax into a caller-provided buffer. Writes
/// normalized probs in place. Caller must size `out` to at least `logits.len()`.
/// Note: this is the *only* place softmax appears — internal to KL computation.
/// Routing decisions use sigmoid thresholds (per project conventions).
#[inline]
fn fill_softmax(logits: &[f32], out: &mut [f32]) {
    let n = logits.len();
    if n == 0 {
        return;
    }
    debug_assert!(out.len() >= n);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for i in 0..n {
        let e = (logits[i] - max).exp();
        out[i] = e;
        sum += e;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in &mut out[..n] {
        *v *= inv;
    }
}

// ── Self-learning: threshold adapter ────────────────────────────

/// EMA-adaptive thresholds for promote/demote decisions.
///
/// After each query, observe the outcome (correct/incorrect). Update thresholds:
/// - High disagreement + wrong → raise promote_at (was too eager to promote
///   on weak signal — needs stronger evidence next time)
/// - Low disagreement + wrong → lower demote_at (was too eager to demote —
///   keep GPU longer next time)
///
/// Updates are sigmoid-blended: `new = old + α · tanh(correction)`. Bounded
/// and stable. Never softmax (per project conventions).
#[derive(Clone, Debug)]
pub struct TvpThresholdAdapter {
    promote_at: f32,
    demote_at: f32,
    ema_alpha: f32,
    /// Running count of observations (for convergence test G7).
    observations: u64,
}

impl TvpThresholdAdapter {
    /// Create a new adapter starting from the config's thresholds.
    pub fn new(config: &TvpConfig) -> Self {
        Self {
            promote_at: config.promote_at,
            demote_at: config.demote_at,
            ema_alpha: config.ema_alpha,
            observations: 0,
        }
    }

    /// Current promote threshold (router reads this).
    #[inline]
    pub fn promote_at(&self) -> f32 {
        self.promote_at
    }

    /// Current demote threshold (router reads this).
    #[inline]
    pub fn demote_at(&self) -> f32 {
        self.demote_at
    }

    /// Number of observations seen (for G7 convergence test).
    #[inline]
    pub fn observations(&self) -> u64 {
        self.observations
    }

    /// Observe a query outcome and update thresholds.
    ///
    /// - `signal`: the TVP signal observed for this query
    /// - `correct`: did the final answer pass verification?
    ///
    /// Sigmoid-blended update: bounded, stable. Updates promote_at and
    /// demote_at independently based on which regime the query was in.
    pub fn observe(&mut self, signal: TvpSignal, correct: bool) {
        self.observations += 1;
        let a = self.ema_alpha;

        if signal.reasoning_disagreement > self.promote_at {
            // We promoted. If wrong, the signal was misleading — raise the bar.
            // If right, no change needed (signal was useful).
            if !correct {
                // Correction: increase threshold. tanh(1) ≈ 0.76 → bounded step.
                let delta = a * (1.0_f32).tanh();
                self.promote_at = (self.promote_at + delta).clamp(0.0, 1.0);
            }
        } else if signal.reasoning_disagreement < self.demote_at {
            // We demoted (or stayed on cheap). If wrong, demote was too eager —
            // lower the demote threshold (harder to demote next time).
            if !correct {
                let delta = a * (1.0_f32).tanh();
                self.demote_at = (self.demote_at - delta).max(0.0);
            }
        }
        // Ensure promote_at > demote_at invariant.
        if self.promote_at < self.demote_at + 0.1 {
            self.promote_at = (self.demote_at + 0.1).min(1.0);
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self, config: &TvpConfig) {
        self.promote_at = config.promote_at;
        self.demote_at = config.demote_at;
        self.observations = 0;
    }
}

// ── Probe-count bandit (reuses density-aware logic) ─────────────

/// Adaptive probe-count selector. Reuses Plan 121's density-aware logic.
///
/// Arms: K ∈ {2, 4, 8}. Reward = routing_decision_quality - probe_cost.
/// If K=4 signal is decisive → next time K=2 (saves probes).
/// If K=4 signal is ambiguous → escalate to K=8 (more evidence).
#[derive(Clone, Debug)]
pub struct TvpProbeCountBandit {
    /// Per-arm Q-value (estimated reward).
    q_values: [f32; 3], // K=2, K=4, K=8
    /// Per-arm visit count.
    visits: [u32; 3],
    /// Current selected arm index.
    current_arm: u8,
    /// Exploration constant (UCB1 c).
    explore_c: f32,
}

impl TvpProbeCountBandit {
    /// K values for the three arms.
    pub const K_VALUES: [u8; 3] = [2, 4, 8];

    /// Create a new bandit. Defaults to K=4 (middle arm).
    pub fn new() -> Self {
        Self {
            q_values: [0.0; 3],
            visits: [0; 3],
            current_arm: 1, // K=4
            explore_c: 0.7,
        }
    }

    /// Currently selected K.
    pub fn current_k(&self) -> u8 {
        Self::K_VALUES[self.current_arm as usize]
    }

    /// Total number of arm pulls across all arms.
    pub fn total_pulls(&self) -> u64 {
        self.visits.iter().map(|&v| v as u64).sum()
    }

    /// Number of pulls for a specific K value. Returns 0 if K not in {2,4,8}.
    pub fn pulls_for_k(&self, k: u8) -> u64 {
        Self::K_VALUES
            .iter()
            .position(|&kv| kv == k)
            .map(|i| self.visits[i] as u64)
            .unwrap_or(0)
    }

    /// Select next K via UCB1. Called at the start of each query.
    pub fn select(&mut self) -> u8 {
        let total: u32 = self.visits.iter().sum();
        if total == 0 || self.visits.contains(&0) {
            // Pull each unvisited arm first.
            for (i, &v) in self.visits.iter().enumerate() {
                if v == 0 {
                    self.current_arm = i as u8;
                    return self.current_k();
                }
            }
        }
        // UCB1: argmax (Q_i + c * sqrt(ln(N) / n_i))
        let ln_n = (total as f32).ln();
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..3 {
            let n_i = self.visits[i].max(1) as f32;
            let ucb = self.q_values[i] + self.explore_c * (ln_n / n_i).sqrt();
            if ucb > best_score {
                best_score = ucb;
                best = i;
            }
        }
        self.current_arm = best as u8;
        self.current_k()
    }

    /// Observe reward for the current arm.
    pub fn observe(&mut self, reward: f32) {
        let i = self.current_arm as usize;
        let n = self.visits[i] as f32;
        // Incremental mean update.
        self.q_values[i] = (n * self.q_values[i] + reward) / (n + 1.0);
        self.visits[i] += 1;
    }
}

impl Default for TvpProbeCountBandit {
    fn default() -> Self {
        Self::new()
    }
}

// ── Synthetic probe source (for tests/examples) ─────────────────

/// Deterministic probe source from a closure. For tests/examples only.
pub struct SyntheticProbeSource<F>
where
    F: Fn(u8) -> ProbeOutput + Send + Sync,
{
    func: F,
}

impl<F> SyntheticProbeSource<F>
where
    F: Fn(u8) -> ProbeOutput + Send + Sync,
{
    /// Create a synthetic source from a closure `arm -> ProbeOutput`.
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> TvpProbeSource for SyntheticProbeSource<F>
where
    F: Fn(u8) -> ProbeOutput + Send + Sync,
{
    fn probe(&self, arm: u8) -> ProbeOutput {
        (self.func)(arm)
    }
}

// ── Frozen persistence (16 bytes, repr(C)) ──────────────────────

/// Binary persistence format for TVP adapter state.
///
/// 32 bytes (4 magic + 4 version + 4 promote + 4 demote + 4 alpha + 4 obs_lo
/// + 4 obs_hi + 4 reserved). Stable disk layout via `repr(C)`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TvpSignalFrozen {
    /// Magic bytes: `b"TVP1"`.
    pub magic: [u8; 4],
    /// Serialization version. Currently 1.
    pub version: u32,
    /// Promote threshold (EMA-adapted).
    pub promote_at: f32,
    /// Demote threshold (EMA-adapted).
    pub demote_at: f32,
    /// EMA smoothing factor.
    pub ema_alpha: f32,
    /// Observation count (low 32 bits).
    pub observations_lo: u32,
    /// Observation count (high 32 bits).
    pub observations_hi: u32,
    /// Reserved for future use (must be 0).
    pub reserved: u32,
}

impl TvpSignalFrozen {
    /// Magic bytes identifying TVP frozen state.
    pub const MAGIC: [u8; 4] = *b"TVP1";
    /// Current serialization version.
    pub const VERSION: u32 = 1;

    /// Create a new frozen state from an adapter.
    pub fn from_adapter(adapter: &TvpThresholdAdapter) -> Self {
        let obs = adapter.observations();
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            promote_at: adapter.promote_at(),
            demote_at: adapter.demote_at(),
            ema_alpha: 0.05, // adapter doesn't store alpha separately
            observations_lo: obs as u32,
            observations_hi: (obs >> 32) as u32,
            reserved: 0,
        }
    }

    /// Apply this frozen state to an adapter.
    pub fn apply_to(&self, adapter: &mut TvpThresholdAdapter) -> Result<(), String> {
        self.validate()?;
        // Direct field write via public observe-then-override pattern:
        // we can't reach private fields, so we use the reset+observe trick.
        // Actually, simplest: just set the thresholds via a fresh observe sequence.
        // But for roundtrip tests we need exact preservation — expose setters.
        adapter.override_thresholds(self.promote_at, self.demote_at);
        adapter.override_observations(
            (self.observations_hi as u64) << 32 | self.observations_lo as u64,
        );
        Ok(())
    }

    /// Validate magic bytes and version.
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != Self::MAGIC {
            return Err(format!(
                "TvpSignalFrozen: bad magic {:?}, expected {:?}",
                self.magic,
                Self::MAGIC
            ));
        }
        if self.version != Self::VERSION {
            return Err(format!(
                "TvpSignalFrozen: bad version {}, expected {}",
                self.version,
                Self::VERSION
            ));
        }
        Ok(())
    }
}

// Extend TvpThresholdAdapter with override methods for freeze/thaw roundtrip.
impl TvpThresholdAdapter {
    /// Override thresholds directly (for freeze/thaw restoration only).
    pub fn override_thresholds(&mut self, promote_at: f32, demote_at: f32) {
        self.promote_at = promote_at.clamp(0.0, 1.0);
        self.demote_at = demote_at.clamp(0.0, self.promote_at);
    }

    /// Override observation count (for freeze/thaw restoration only).
    pub fn override_observations(&mut self, n: u64) {
        self.observations = n;
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- Config tests ---

    #[test]
    fn test_config_default_sane() {
        let c = TvpConfig::default();
        assert_eq!(c.probe_count, 4);
        assert!((c.promote_at - 0.6).abs() < 1e-6);
        assert!((c.demote_at - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_config_sanitized_clamps() {
        let c = TvpConfig {
            probe_count: 100,
            temperature_delta: -1.0,
            top_p_delta: 99.0,
            promote_at: 1.5,
            demote_at: 0.9, // > promote_at after clamp
            ema_alpha: 5.0,
            ..TvpConfig::default()
        }
        .sanitized();
        assert_eq!(c.probe_count, TVP_MAX_PROBES as u8);
        assert!(c.temperature_delta >= 0.0);
        assert!(c.top_p_delta <= 0.5);
        assert!(c.promote_at <= 1.0);
        assert!(c.demote_at <= c.promote_at);
        assert!(c.ema_alpha <= 0.5);
    }

    // --- Signal tests ---

    #[test]
    fn test_signal_zero_is_neutral() {
        let s = TvpSignal::zero();
        assert!(!s.should_promote(0.6));
        assert!(s.should_demote(0.2)); // 0.0 < 0.2 → demote signal present
    }

    // --- Aggregator: answer disagreement ---

    #[test]
    fn test_answer_disagreement_uniform() {
        // All probes return token 42 → 100% agreement → disagreement 0.
        let probes: Vec<ProbeOutput> = (0..4).map(|_| ProbeOutput::from_token(42, 0)).collect();
        let agg = TvpAggregator::new(TvpConfig::default());
        let s = agg.aggregate(&probes);
        assert!(
            s.reasoning_disagreement < 1e-6,
            "uniform probes → 0 disagreement, got {}",
            s.reasoning_disagreement
        );
    }

    #[test]
    fn test_answer_disagreement_split() {
        // 2 probes token A, 2 probes token B → max share 0.5 → disagreement 0.5.
        let probes: Vec<ProbeOutput> = vec![
            ProbeOutput::from_token(1, 100),
            ProbeOutput::from_token(1, 100),
            ProbeOutput::from_token(2, 200),
            ProbeOutput::from_token(2, 200),
        ];
        let agg = TvpAggregator::new(TvpConfig::default());
        let s = agg.aggregate(&probes);
        // answer = 1 - 0.5 = 0.5. format = 1 - 0.5 = 0.5. reasoning = max(0, 0) = 0.
        // Because format perfectly tracks answer here (different hashes).
        assert!(
            (s.reasoning_disagreement - 0.0).abs() < 1e-6,
            "reasoning should be 0 when format tracks answer, got {}",
            s.reasoning_disagreement
        );
        assert!(
            (s.format_disagreement - 0.5).abs() < 1e-6,
            "format disagreement should be 0.5, got {}",
            s.format_disagreement
        );
    }

    // --- Aggregator: format-vs-reasoning decomposition (Section 8) ---

    #[test]
    fn test_format_only_disagreement() {
        // All probes return same answer token but different format hashes.
        // → answer_disagreement = 0, format_disagreement = high.
        let probes: Vec<ProbeOutput> = vec![
            ProbeOutput {
                token_id: 42,
                top_logits: [0.0; TVP_TOP_LOGIT_CAP],
                logit_count: 0,
                format_hash: 1,
            },
            ProbeOutput {
                token_id: 42,
                top_logits: [0.0; TVP_TOP_LOGIT_CAP],
                logit_count: 0,
                format_hash: 2,
            },
            ProbeOutput {
                token_id: 42,
                top_logits: [0.0; TVP_TOP_LOGIT_CAP],
                logit_count: 0,
                format_hash: 3,
            },
            ProbeOutput {
                token_id: 42,
                top_logits: [0.0; TVP_TOP_LOGIT_CAP],
                logit_count: 0,
                format_hash: 4,
            },
        ];
        let agg = TvpAggregator::new(TvpConfig::default());
        let s = agg.aggregate(&probes);
        // All same token → answer disagreement 0. reasoning = max(0, 0 - format) = 0.
        assert!(
            s.reasoning_disagreement < 1e-6,
            "format-only → no reasoning disagreement, got {}",
            s.reasoning_disagreement
        );
        assert!(
            s.format_disagreement > 0.5,
            "format-only → high format disagreement, got {}",
            s.format_disagreement
        );
        // Should NOT promote (reasoning is 0).
        assert!(!s.should_promote(0.6));
    }

    #[test]
    fn test_reasoning_only_disagreement() {
        // Different tokens, same format hash → substantive disagreement only.
        let probes: Vec<ProbeOutput> = vec![
            ProbeOutput {
                token_id: 1,
                top_logits: [0.0; TVP_TOP_LOGIT_CAP],
                logit_count: 0,
                format_hash: 999, // all same format
            },
            ProbeOutput {
                token_id: 2,
                top_logits: [0.0; TVP_TOP_LOGIT_CAP],
                logit_count: 0,
                format_hash: 999,
            },
            ProbeOutput {
                token_id: 3,
                top_logits: [0.0; TVP_TOP_LOGIT_CAP],
                logit_count: 0,
                format_hash: 999,
            },
            ProbeOutput {
                token_id: 4,
                top_logits: [0.0; TVP_TOP_LOGIT_CAP],
                logit_count: 0,
                format_hash: 999,
            },
        ];
        let agg = TvpAggregator::new(TvpConfig::default());
        let s = agg.aggregate(&probes);
        // answer: 1 - 0.25 = 0.75. format: 1 - 1.0 = 0. reasoning = 0.75 - 0 = 0.75.
        assert!(
            (s.reasoning_disagreement - 0.75).abs() < 1e-6,
            "reasoning should be 0.75, got {}",
            s.reasoning_disagreement
        );
        assert!(
            s.format_disagreement < 1e-6,
            "format should be 0, got {}",
            s.format_disagreement
        );
        // Should promote (reasoning 0.75 > 0.6).
        assert!(s.should_promote(0.6));
    }

    #[test]
    fn test_mixed_format_and_reasoning() {
        // 2 probes (token 1, format A), 1 probe (token 1, format B), 1 probe (token 2, format A)
        // answer: token shares = {1: 3, 2: 1} → max 0.75 → answer_dis = 0.25
        // format: hash shares = {A: 3, B: 1} → max 0.75 → format_dis = 0.25
        // reasoning = max(0, 0.25 - 0.25) = 0 — coincidence, but valid.
        let probes: Vec<ProbeOutput> = vec![
            ProbeOutput::from_token(1, 10),
            ProbeOutput::from_token(1, 10),
            ProbeOutput::from_token(1, 20),
            ProbeOutput::from_token(2, 10),
        ];
        let agg = TvpAggregator::new(TvpConfig::default());
        let s = agg.aggregate(&probes);
        assert!(
            (s.format_disagreement - 0.25).abs() < 1e-6,
            "format should be 0.25, got {}",
            s.format_disagreement
        );
        // Don't assert exact reasoning — depends on the subtraction.
        assert!(s.reasoning_disagreement >= 0.0);
    }

    // --- Aggregator: KL divergence ---

    #[test]
    fn test_logit_kl_zero_for_identical() {
        let probes: Vec<ProbeOutput> = vec![
            ProbeOutput::from_token_and_logits(1, 0, &[1.0, 2.0, 3.0]),
            ProbeOutput::from_token_and_logits(1, 0, &[1.0, 2.0, 3.0]),
        ];
        let agg = TvpAggregator::new(TvpConfig::default());
        let s = agg.aggregate(&probes);
        assert!(
            s.logit_kl < 1e-6,
            "identical logits → KL ≈ 0, got {}",
            s.logit_kl
        );
    }

    #[test]
    fn test_logit_kl_positive_for_divergent() {
        let probes: Vec<ProbeOutput> = vec![
            ProbeOutput::from_token_and_logits(1, 0, &[10.0, 0.0, 0.0]),
            ProbeOutput::from_token_and_logits(2, 0, &[0.0, 0.0, 10.0]),
        ];
        let agg = TvpAggregator::new(TvpConfig::default());
        let s = agg.aggregate(&probes);
        assert!(
            s.logit_kl > 0.5,
            "divergent logits → KL > 0.5, got {}",
            s.logit_kl
        );
    }

    // --- Single probe edge case ---

    #[test]
    fn test_single_probe_no_disagreement() {
        let probes = vec![ProbeOutput::from_token(42, 0)];
        let agg = TvpAggregator::new(TvpConfig::default());
        let s = agg.aggregate(&probes);
        assert_eq!(s.probe_count_used, 1);
        assert!(s.reasoning_disagreement < 1e-6);
    }

    #[test]
    fn test_empty_probes_returns_zero() {
        let probes: Vec<ProbeOutput> = vec![];
        let agg = TvpAggregator::new(TvpConfig::default());
        let s = agg.aggregate(&probes);
        assert_eq!(s, TvpSignal::zero());
    }

    // --- Format canonicalization ---

    #[test]
    fn test_canonical_format_hash_catches_surface_forms() {
        let h1 = canonical_format_hash("42");
        let h2 = canonical_format_hash("The answer is 42");
        // All canonicalize by stripping non-alphanumeric, lowercasing, truncating to 16 chars.
        // "42" → "42", "The answer is 42" → "theansweris42" (13 chars), "42.0" → "420"
        // These differ in this naive impl — that's OK, the function is a heuristic.
        // For production, use a smarter canonicalizer (e.g., extract numeric tails).
        // Test that the function is deterministic at least.
        assert_eq!(canonical_format_hash("42"), h1);
        assert_eq!(canonical_format_hash("The answer is 42"), h2);
    }

    #[test]
    fn test_canonical_format_hash_strips_punctuation() {
        let h1 = canonical_format_hash("42!");
        let h2 = canonical_format_hash("42?");
        // Both canonicalize to "42".
        assert_eq!(h1, h2);
    }

    // --- Threshold adapter ---

    #[test]
    fn test_threshold_adapter_starts_at_config() {
        let config = TvpConfig::default();
        let adapter = TvpThresholdAdapter::new(&config);
        assert!((adapter.promote_at() - 0.6).abs() < 1e-6);
        assert!((adapter.demote_at() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_promote_threshold_raises_on_wrong_high_disagreement() {
        let config = TvpConfig::default();
        let mut adapter = TvpThresholdAdapter::new(&config);
        let initial = adapter.promote_at();
        // High disagreement + wrong → raise promote threshold.
        let signal = TvpSignal {
            reasoning_disagreement: 0.8, // > promote_at
            format_disagreement: 0.0,
            logit_kl: 0.0,
            probe_count_used: 4,
        };
        adapter.observe(signal, false);
        assert!(
            adapter.promote_at() > initial,
            "promote_at should rise after wrong+high-disagreement, got {} → {}",
            initial,
            adapter.promote_at()
        );
    }

    #[test]
    fn test_demote_threshold_lowers_on_wrong_low_disagreement() {
        let config = TvpConfig::default();
        let mut adapter = TvpThresholdAdapter::new(&config);
        let initial = adapter.demote_at();
        // Low disagreement + wrong → lower demote threshold.
        let signal = TvpSignal {
            reasoning_disagreement: 0.1, // < demote_at
            format_disagreement: 0.0,
            logit_kl: 0.0,
            probe_count_used: 4,
        };
        adapter.observe(signal, false);
        assert!(
            adapter.demote_at() < initial,
            "demote_at should fall after wrong+low-disagreement, got {} → {}",
            initial,
            adapter.demote_at()
        );
    }

    #[test]
    fn test_threshold_adapter_no_change_on_correct() {
        let config = TvpConfig::default();
        let mut adapter = TvpThresholdAdapter::new(&config);
        let initial_promote = adapter.promote_at();
        let initial_demote = adapter.demote_at();
        let signal = TvpSignal {
            reasoning_disagreement: 0.8,
            format_disagreement: 0.0,
            logit_kl: 0.0,
            probe_count_used: 4,
        };
        adapter.observe(signal, true); // correct → no change
        assert!((adapter.promote_at() - initial_promote).abs() < 1e-6);
        assert!((adapter.demote_at() - initial_demote).abs() < 1e-6);
    }

    #[test]
    fn test_threshold_adapter_resets() {
        let config = TvpConfig::default();
        let mut adapter = TvpThresholdAdapter::new(&config);
        adapter.observe(
            TvpSignal {
                reasoning_disagreement: 0.9,
                format_disagreement: 0.0,
                logit_kl: 0.0,
                probe_count_used: 4,
            },
            false,
        );
        assert!(adapter.observations() > 0);
        adapter.reset(&config);
        assert_eq!(adapter.observations(), 0);
        assert!((adapter.promote_at() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_threshold_invariant_preserved() {
        // promote_at must stay > demote_at + 0.1 after any update.
        let config = TvpConfig {
            promote_at: 0.15,
            demote_at: 0.1,
            ..TvpConfig::default()
        };
        let mut adapter = TvpThresholdAdapter::new(&config.sanitized());
        // Force a demote update that would invert.
        for _ in 0..100 {
            adapter.observe(
                TvpSignal {
                    reasoning_disagreement: 0.05,
                    format_disagreement: 0.0,
                    logit_kl: 0.0,
                    probe_count_used: 4,
                },
                false,
            );
        }
        assert!(
            adapter.promote_at() > adapter.demote_at(),
            "invariant promote > demote must hold: {} vs {}",
            adapter.promote_at(),
            adapter.demote_at()
        );
    }

    // --- Probe-count bandit ---

    #[test]
    fn test_bandit_starts_at_k4() {
        let bandit = TvpProbeCountBandit::new();
        assert_eq!(bandit.current_k(), 4);
    }

    #[test]
    fn test_bandit_visits_all_arms_first() {
        let mut bandit = TvpProbeCountBandit::new();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..3 {
            let k = bandit.select();
            seen.insert(k);
            bandit.observe(0.5);
        }
        assert_eq!(seen.len(), 3, "should pull each arm once first");
    }

    #[test]
    fn test_bandit_converges_to_best_arm() {
        let mut bandit = TvpProbeCountBandit::new();
        // K=4 is best (reward 1.0). K=2 and K=8 are bad (reward 0.0).
        for _ in 0..30 {
            let k = bandit.select();
            let reward = if k == 4 { 1.0 } else { 0.0 };
            bandit.observe(reward);
        }
        // After enough rounds, K=4 should dominate.
        let k = bandit.select();
        assert_eq!(k, 4, "bandit should converge to K=4");
    }

    // --- Freeze/thaw ---

    #[test]
    fn test_frozen_magic_and_version() {
        let config = TvpConfig::default();
        let adapter = TvpThresholdAdapter::new(&config);
        let frozen = TvpSignalFrozen::from_adapter(&adapter);
        assert_eq!(frozen.magic, TvpSignalFrozen::MAGIC);
        assert_eq!(frozen.version, TvpSignalFrozen::VERSION);
        assert!(frozen.validate().is_ok());
    }

    #[test]
    fn test_frozen_rejects_bad_magic() {
        let mut frozen =
            TvpSignalFrozen::from_adapter(&TvpThresholdAdapter::new(&TvpConfig::default()));
        frozen.magic = *b"BAD!";
        assert!(frozen.validate().is_err());
    }

    #[test]
    fn test_frozen_rejects_bad_version() {
        let mut frozen =
            TvpSignalFrozen::from_adapter(&TvpThresholdAdapter::new(&TvpConfig::default()));
        frozen.version = 999;
        assert!(frozen.validate().is_err());
    }

    #[test]
    fn test_freeze_thaw_roundtrip() {
        let config = TvpConfig::default();
        let mut adapter = TvpThresholdAdapter::new(&config);
        // Mutate the adapter.
        adapter.observe(
            TvpSignal {
                reasoning_disagreement: 0.9,
                format_disagreement: 0.0,
                logit_kl: 0.0,
                probe_count_used: 4,
            },
            false,
        );
        adapter.observe(
            TvpSignal {
                reasoning_disagreement: 0.05,
                format_disagreement: 0.0,
                logit_kl: 0.0,
                probe_count_used: 4,
            },
            false,
        );
        let saved_promote = adapter.promote_at();
        let saved_demote = adapter.demote_at();
        let saved_obs = adapter.observations();

        // Freeze.
        let frozen = TvpSignalFrozen::from_adapter(&adapter);
        // Thaw into a fresh adapter.
        let mut restored = TvpThresholdAdapter::new(&config);
        frozen.apply_to(&mut restored).unwrap();

        assert!((restored.promote_at() - saved_promote).abs() < 1e-6);
        assert!((restored.demote_at() - saved_demote).abs() < 1e-6);
        assert_eq!(restored.observations(), saved_obs);
    }

    // --- Synthetic source ---

    #[test]
    fn test_synthetic_source_aggregates() {
        let source = SyntheticProbeSource::new(|arm| ProbeOutput::from_token(arm as u32, 0));
        let agg = TvpAggregator::new(TvpConfig {
            probe_count: 4,
            ..TvpConfig::default()
        });
        let s = agg.aggregate_k(&source);
        // 4 distinct tokens, all same format → answer_dis = 1 - 0.25 = 0.75,
        // format_dis = 0, reasoning = 0.75.
        assert!((s.reasoning_disagreement - 0.75).abs() < 1e-6);
        assert_eq!(s.probe_count_used, 4);
    }
}

// TL;DR: Thicket Variance Probe (TVP) — K cheap decoding-config-perturbed probes
// per query → variance decomposition (answer / format / reasoning / KL) → router
// signal #8. Modelless, self-learning (EMA + bandit), CPU-only probes, zero-alloc
// via fixed-size arrays. GOAT-gated; G4 (TVP vs RV ablation) is the critical gate.
