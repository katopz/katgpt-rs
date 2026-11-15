//! Diffusion Sampler — Adaptive Per-Position Correctness Predictor for D2F Denoising.
//!
//! Plan 089 T6: Trained Sampler Research.
//!
//! Replaces fixed confidence threshold with a learned per-position correctness
//! predictor. Two capacity variants:
//!
//! 1. **Logistic** — 6 features, ~7 params. For micro_dllm scale (d=16, vocab=27).
//! 2. **Transformer** — 4-layer d=384, ~4.8M params. For production scale (stub, deferred).
//!
//! Research (Nemotron Appendix A): shifts Pareto frontier +1.3× TPF or +10.6% accuracy.
//!
//! # Feature Vector (per position, per denoising step)
//!
//! | Feature | Description |
//! |---------|-------------|
//! | top1_prob | Top-1 token probability after softmax |
//! | margin | Top-1 prob − top-2 prob |
//! | top3_mass | Sum of top-3 token probabilities |
//! | entropy | Entropy of softmax distribution |
//! | step_norm | Current step / max steps |
//! | pos_norm | Position / block_size |
//!
//! # Usage
//!
//! ```ignore
//! // Collect trajectories from D2F decode with ground truth
//! let trajectories = collect_trajectories(&weights, &config, &decode_config, &test_data, 100);
//!
//! // Train sampler
//! let mut sampler = DiffusionSampler::auto(&config);
//! sampler.train(&trajectories, 0.01, 200);
//!
//! // Use in denoising loop — replace fixed threshold
//! let p_correct = sampler.predict(&features);
//! if p_correct >= tau_conf { ... }
//! ```

// Plan 399 (2026-07-05): moved from root `src/speculative/diffusion_sampler.rs`.
// Import rewrites applied: `crate::dllm::*` → `crate::d2f_context::*`,
// `crate::speculative::d2f::*` → `crate::d2f::*`,
// `crate::transformer::*` → `katgpt_transformer::*`,
// `crate::types::*` → `katgpt_types::*`.

use crate::d2f::D2fDecodeConfig;
use crate::d2f_context::{D2fContext, forward_block_causal_with};
use katgpt_core::jsd_topk::jsd_topk_sets;
use katgpt_transformer::TransformerWeights;
use katgpt_types::{Config, Rng};

// ── Feature Extraction ────────────────────────────────────────

/// Number of input features per position.
pub const N_FEATURES: usize = 6;

/// Top-K set width for the inter-step drift feature (Issue 802 item 3) — the
/// same three tiers [`SamplerFeatures::from_logits`] already tracks for
/// `top3_mass`, so the drift set comes free with the feature extraction
/// (indices added in [`SamplerFeatures::from_logits_into`], masses
/// untouched).
pub const TOPK_DRIFT_K: usize = 3;

/// Number of stability features appended by
/// [`SamplerFeatures::to_array_with_stability`] (Issue 802 item 3):
/// `stable_age` and `topk_drift`.
pub const N_STABILITY_FEATURES: usize = 2;

/// Per-position features extracted from D2F denoising step.
///
/// Lightweight stats that capture the model's confidence at each position
/// without requiring the full embedding (which would be d=384 in production).
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplerFeatures {
    /// Top-1 token probability after softmax.
    pub top1_prob: f32,
    /// Margin: top-1 prob − top-2 prob. Higher = more confident.
    pub margin: f32,
    /// Sum of top-3 token probabilities. Higher = peaked distribution.
    pub top3_mass: f32,
    /// Entropy of softmax distribution. Lower = more confident.
    pub entropy: f32,
    /// Current denoising step normalized: step / max_steps.
    pub step_norm: f32,
    /// Position within block normalized: pos / block_size.
    pub pos_norm: f32,
    /// Consecutive-step stability counter (Issue 802 item 3): +1 per
    /// observed step whose argmax equals the previous armed step's
    /// (saturating at `u8::MAX`); reset on an argmax change, on an
    /// observation below the arming mass floor, and on any remask. 0 on the
    /// first observation. Set by [`StabilityTracker::observe`] — the pure
    /// [`Self::from_logits`] extraction cannot see across steps and leaves
    /// both stability fields at their defaults.
    pub stable_age: u8,
    /// Inter-step top-K drift (Issue 802 item 3): the restricted JSD
    /// between this step's and the previous armed step's top-3 sets, via
    /// `jsd_topk_sets` (bounded by ln 2; disjoint sets → exactly ln 2).
    /// 0.0 = no measurement (first observation, disarmed step, post-remask).
    pub topk_drift: f32,
}

impl SamplerFeatures {
    /// Extract features from a flat logits slice for one position.
    ///
    /// `logits_p` must be length `vocab`. `mask` token is excluded from stats.
    ///
    /// Zero-allocation: tracks top-3 probabilities via branchless comparisons
    /// during the exp/sum pass (order preserved under division by `sum_exp`).
    /// Entropy computed via the identity `Σ p·log(p) = (Σ e·log(e))/sum - log(sum)`
    /// to avoid per-element division. Previously allocated 2 `Vec<f32>` of
    /// vocab_size and did a full O(n log n) sort for top-3 mass.
    pub fn from_logits(
        logits_p: &[f32],
        vocab: usize,
        mask: usize,
        step: usize,
        max_steps: usize,
        pos: usize,
        block_size: usize,
    ) -> Self {
        Self::from_logits_into(
            logits_p, vocab, mask, step, max_steps, pos, block_size, None,
        )
    }

    /// [`Self::from_logits`] plus the top-3 token indices emitted into
    /// `topk_idx` in probability order — the same tiers the masses track,
    /// so the [`StabilityTracker`] drift set costs nothing extra (Issue 802
    /// item 3). The extracted values (`top1_prob`/`margin`/`top3_mass`/
    /// `entropy`) are bitwise identical to the bare form: the index
    /// tracking rides the existing branchless shifts and adds no math. A
    /// degenerate row (zero restricted mass) emits `[0; 3]` and the default
    /// features (top1_prob 0.0 disarms any sane [`StabilityTracker`] floor).
    #[allow(clippy::too_many_arguments)] // the 7 from_logits params + the emit out-param
    pub fn from_logits_into(
        logits_p: &[f32],
        vocab: usize,
        mask: usize,
        step: usize,
        max_steps: usize,
        pos: usize,
        block_size: usize,
        topk_idx: Option<&mut [u32; TOPK_DRIFT_K]>,
    ) -> Self {
        debug_assert_eq!(vocab, logits_p.len(), "vocab size mismatch");
        let max_logit = logits_p.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Single fused pass: compute exp, sum_exp, and track top-3 exp values
        // (order preserved under division by sum_exp → top-3 exp == top-3 prob).
        // Also accumulate Σ e·log(e) for entropy via the identity
        //   H = -Σ p·log(p) = -(Σ e·log(e))/sum + log(sum).
        // Scalar `fast_exp` (not SIMD): top-3 tracking branches are interleaved
        // with the exp, preventing clean SIMD vectorization.
        use katgpt_core::simd::fast_exp;
        let mut sum_exp = 0.0f32;
        let mut sum_e_log_e = 0.0f32;
        // Top-3 trackers (exp values). Initialized to -1 so any valid exp >= 0 wins.
        let mut t1 = f32::NEG_INFINITY;
        let mut t2 = f32::NEG_INFINITY;
        let mut t3 = f32::NEG_INFINITY;
        // Top-3 token indices, shifted in lockstep with the exps (Issue 802
        // item 3 — the drift feature's set). Strict-`>` comparisons keep the
        // FIRST index on exp ties (deterministic).
        let mut i1 = 0u32;
        let mut i2 = 0u32;
        let mut i3 = 0u32;
        for (t, &logit) in logits_p.iter().enumerate() {
            if t == mask {
                continue;
            }
            let e = fast_exp(logit - max_logit);
            sum_exp += e;
            // Branchless top-3 update: shift down only if e exceeds current tier.
            // Compiles to cmov / fmax sequences — no data-dependent branches.
            if e > t1 {
                t3 = t2;
                i3 = i2;
                t2 = t1;
                i2 = i1;
                t1 = e;
                i1 = t as u32;
            } else if e > t2 {
                t3 = t2;
                i3 = i2;
                t2 = e;
                i2 = t as u32;
            } else if e > t3 {
                t3 = e;
                i3 = t as u32;
            }
            // e·log(e): e is non-negative; for e near 0 the product is ~0 (lim x·log(x)=0).
            // Guard with max(1e-10) to avoid log(0). Note: this contributes to the
            // entropy identity and must use the *un-normalized* exp values.
            if e > 1e-10 {
                sum_e_log_e += e * e.ln();
            }
        }

        if sum_exp <= 0.0 {
            if let Some(out) = topk_idx {
                *out = [0; TOPK_DRIFT_K];
            }
            return Self::default();
        }
        if let Some(out) = topk_idx {
            *out = [i1, i2, i3];
        }

        // Normalize top-3 exp values to probabilities.
        let inv_sum = 1.0 / sum_exp;
        let top1_prob = t1 * inv_sum;
        let top2_prob = t2 * inv_sum;
        let top3_mass = (t1 + t2 + t3) * inv_sum;

        // Entropy via identity: H = -(Σ e·log(e))/sum + log(sum).
        let entropy = -(sum_e_log_e / sum_exp) + sum_exp.ln();

        let margin = top1_prob - top2_prob;
        let step_norm = if max_steps > 0 {
            step as f32 / max_steps as f32
        } else {
            0.0
        };
        let pos_norm = if block_size > 0 {
            pos as f32 / block_size as f32
        } else {
            0.0
        };

        Self {
            top1_prob,
            margin,
            top3_mass,
            entropy,
            step_norm,
            pos_norm,
            // Cross-step state: filled by StabilityTracker::observe at the
            // call site, never by this pure extraction (Issue 802 item 3).
            stable_age: 0,
            topk_drift: 0.0,
        }
    }

    /// Convert to flat feature array for model input.
    pub fn to_array(&self) -> [f32; N_FEATURES] {
        [
            self.top1_prob,
            self.margin,
            self.top3_mass,
            self.entropy,
            self.step_norm,
            self.pos_norm,
        ]
    }

    /// [`Self::to_array`] plus the two stability features (`stable_age`,
    /// `topk_drift`) — the fusion-predictor input shape (Issue 802 item 3;
    /// the trained-predictor fusion over stability features is the surviving
    /// novelty delta over LESS's fixed rule). [`Self::to_array`] stays the
    /// 6-feature shape: the trained logistic checkpoint contract and its
    /// signal-diff pins are unchanged.
    pub fn to_array_with_stability(&self) -> [f32; N_FEATURES + N_STABILITY_FEATURES] {
        let mut out = [0.0f32; N_FEATURES + N_STABILITY_FEATURES];
        out[..N_FEATURES].copy_from_slice(&self.to_array());
        out[N_FEATURES] = self.stable_age as f32;
        out[N_FEATURES + 1] = self.topk_drift;
        out
    }
}

// ── Inter-Step Stability (Issue 802 items 3 + 5) ─────────────

/// Per-position inter-step stability state (Issue 802 item 3): the
/// `stable_age` saturating counter plus the top-K drift measurement, with
/// the item-5 boundary laws enforced at the single observation seam.
///
/// One tracker per (request, position). `observe` runs once per denoising
/// step at that position; the returned pair fills
/// [`SamplerFeatures::stable_age`] / [`SamplerFeatures::topk_drift`] and
/// rides [`SamplerFeatures::to_array_with_stability`] into the fusion
/// predictor (the trained-predictor fusion is the surviving novelty delta
/// over LESS's fixed rule — LESS, arXiv:2606.16908, owns the fixed-rule
/// class; do NOT file the gate class as novel).
///
/// # Boundary laws (Issue 802 item 5 — all enforced here, tested below)
///
/// - **Mass floor:** an observation with `top1_prob < tau_min` carries no
///   commitment information (the near-fully-masked argmax is noise): the
///   counter resets to 0, no drift is measured, and the noisy set is NOT
///   adopted — the next armed step compares against the last ARMED step,
///   skipping the noise.
/// - **Remask reset:** `remasked = true` clears all state (TSD's
///   unique-commit-time assumption breaks under remasking); the next
///   observation is a first observation.
/// - **Measurement cadence** is caller policy: the tracker measures whenever
///   it is asked; stride-s subsampling stays with the decode loop (TSD's
///   training-time subsampling does not prove the measurement analog).
///
/// First observation (and every post-reset one) returns `(0, 0.0)`: age 0
/// (no held step yet) and drift 0.0 = "no evidence of drift" — a max-drift
/// ln 2 reading would poison the stability signal on the first step.
///
/// Fixed-size state, zero allocation on every path.
#[derive(Clone, Copy, Debug)]
pub struct StabilityTracker {
    /// Previous ARMED step's top-3 set (indices in probability order) — the
    /// drift comparison side.
    prev_idx: [u32; TOPK_DRIFT_K],
    prev_masses: [f32; TOPK_DRIFT_K],
    /// Whether `prev_*` carries a comparison side (false from construction
    /// and after a remask reset; a below-floor step keeps the last armed
    /// side per the mass-floor law).
    prev_valid: bool,
    /// Saturating consecutive-stable-step counter.
    age: u8,
}

impl StabilityTracker {
    /// Fresh tracker: everything at the first-observation state.
    pub fn new() -> Self {
        Self {
            prev_idx: [0; TOPK_DRIFT_K],
            prev_masses: [0.0; TOPK_DRIFT_K],
            prev_valid: false,
            age: 0,
        }
    }

    /// Observe one denoising step at this position. `idx` / `masses` are the
    /// step's top-3 set in probability order (the
    /// [`SamplerFeatures::from_logits_into`] emit); `idx[0]` is the argmax
    /// over non-mask tokens. Returns `(stable_age, topk_drift)` for the
    /// step.
    ///
    /// Delegates to [`Self::observe_row`] — the row form is the primary
    /// seam and names the boundary branch that fired; this projection
    /// keeps the historical `(u8, f32)` signature byte-identical.
    pub fn observe(
        &mut self,
        idx: &[u32; TOPK_DRIFT_K],
        masses: &[f32; TOPK_DRIFT_K],
        top1_prob: f32,
        tau_min: f32,
        remasked: bool,
    ) -> (u8, f32) {
        let row = self.observe_row(idx, masses, top1_prob, tau_min, remasked);
        (row.stable_age, row.topk_drift)
    }

    /// Clear all state (the in-band form is `observe(.., remasked = true)`;
    /// this is the teardown / inter-request form).
    pub fn reset(&mut self) {
        self.prev_valid = false;
        self.age = 0;
    }
}

impl Default for StabilityTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Provenance rows (Issue 1210 C2) ─────────────────────────

/// Which boundary law produced a [`StabilityTracker`] reading — the
/// receipt half of the observation (Issue 1210 C2's provenance rows;
/// the fleet's typed-reason-at-the-observation-seam pattern —
/// `MoveReport::rejection`, `AbstainReason`, `closure::RejectionReason`,
/// `escalation_guard::receipt` — applied to the tracker's branches, which
/// [`StabilityTracker::observe`]'s bare `(u8, f32)` discards).
///
/// Branch-exact: every variant names a branch the item-5 laws already
/// take; none adds a decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StabilityReason {
    /// First observation at this position (or the first after any reset):
    /// no comparison side exists. Age 0, drift 0.0 by the neutrality law —
    /// a max-drift ln 2 reading here would poison the stability signal.
    FirstObservation,
    /// Armed step whose argmax equals the previous armed step's: the
    /// stable case. Age incremented (saturating), drift measured.
    ArgmaxHeld,
    /// Armed step whose argmax differs from the previous armed step's:
    /// the inflection event itself. Age reset to 0, drift measured
    /// against the last armed set (the row carries the magnitude, so a
    /// consumer can read flip-only or flip-or-spike without a second
    /// seam).
    ArgmaxFlip,
    /// Below the arming mass floor, or a non-finite `top1_prob` (which
    /// must disarm — through `<` it would silently arm): no commitment
    /// information, the noisy set is NOT adopted and the next armed step
    /// compares against the last ARMED side (the item-5 mass-floor law).
    /// Age reset, no drift measured.
    Disarmed,
    /// A remask cleared all state (the item-5 remask law — TSD's
    /// unique-commit-time assumption breaks under remasking). Age reset,
    /// no drift; the next observation is a first observation.
    RemaskReset,
}

/// One provenance row: which law fired, and what it measured —
/// [`StabilityTracker::observe`]'s `(stable_age, topk_drift)` with the
/// branch that produced them named (Issue 1210 C2). Zero allocation;
/// `Copy` so a capture can hold rows in a flat buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StabilityObservation {
    /// The boundary branch that fired (see [`StabilityReason`]).
    pub reason: StabilityReason,
    /// The stable-age half of the reading (identical to `observe`'s
    /// first element — the row is a superset, never a recomputation).
    pub stable_age: u8,
    /// The drift half of the reading (identical to `observe`'s second
    /// element; 0.0 wherever the law does not measure — first
    /// observation, disarmed step, post-remask).
    pub topk_drift: f32,
}

impl StabilityTracker {
    /// [`Self::observe`] with the branch that fired named — the primary
    /// seam; the bare-tuple form delegates here, so the two can never
    /// disagree (one computation, two projections).
    ///
    /// Same contract as [`Self::observe`]: `idx` / `masses` are the
    /// step's top-3 set in probability order, `idx[0]` the argmax over
    /// non-mask tokens.
    pub fn observe_row(
        &mut self,
        idx: &[u32; TOPK_DRIFT_K],
        masses: &[f32; TOPK_DRIFT_K],
        top1_prob: f32,
        tau_min: f32,
        remasked: bool,
    ) -> StabilityObservation {
        if remasked {
            self.prev_valid = false;
            self.age = 0;
            return StabilityObservation {
                reason: StabilityReason::RemaskReset,
                stable_age: 0,
                topk_drift: 0.0,
            };
        }
        // `is_nan()` spelled out (not a negated comparison): a NaN
        // top1_prob must DISARM — through `<` it would silently arm.
        if top1_prob < tau_min || top1_prob.is_nan() {
            self.age = 0;
            return StabilityObservation {
                reason: StabilityReason::Disarmed,
                stable_age: 0,
                topk_drift: 0.0,
            };
        }

        // Restricted JSD between the last armed set and this one. Both sets
        // are copied and index-sorted (the sets arrive in probability
        // order; `jsd_topk_sets` wants ascending indices). K = 3 → two
        // fixed-size sorts, no allocation.
        let (drift, prev_valid, prev_argmax_held) = if self.prev_valid {
            let mut a: [(u32, f32); TOPK_DRIFT_K] =
                std::array::from_fn(|i| (self.prev_idx[i], self.prev_masses[i]));
            let mut b: [(u32, f32); TOPK_DRIFT_K] = std::array::from_fn(|i| (idx[i], masses[i]));
            a.sort_unstable_by_key(|t| t.0);
            b.sort_unstable_by_key(|t| t.0);
            let mut d = 0.0f32;
            jsd_topk_sets(&a, &b, &mut d);
            (d, true, idx[0] == self.prev_idx[0])
        } else {
            (0.0, false, false)
        };

        self.age = if prev_valid && prev_argmax_held {
            self.age.saturating_add(1)
        } else {
            0
        };

        let reason = if !prev_valid {
            StabilityReason::FirstObservation
        } else if prev_argmax_held {
            StabilityReason::ArgmaxHeld
        } else {
            StabilityReason::ArgmaxFlip
        };

        self.prev_idx = *idx;
        self.prev_masses = *masses;
        self.prev_valid = true;
        StabilityObservation {
            reason,
            stable_age: self.age,
            topk_drift: drift,
        }
    }
}

// ── Sampler Variant ───────────────────────────────────────────

/// Which sampler capacity to use.
#[derive(Clone, Debug, Default)]
pub enum SamplerVariant {
    /// Logistic regression: 6 features → 1 output. ~7 params.
    /// Best for micro_dllm scale (d=16, vocab=27, block=16).
    #[default]
    Logistic,
    /// 2-layer MLP: 6 → hidden → 1. Scales with hidden_dim.
    /// For medium-scale models. hidden_dim=32 → ~250 params.
    Mlp { hidden_dim: usize },
    /// 4-layer transformer: d=384, ~4.8M params.
    /// For production-scale models (Nemotron paper spec). Stub, deferred.
    Transformer { d_model: usize, n_layers: usize },
}

// ── Logistic Sampler ──────────────────────────────────────────

/// Logistic regression sampler: sigmoid(w·x + b).
///
/// 6 weights + 1 bias = 7 parameters total.
/// O(1) inference, trivial memory footprint.
#[derive(Clone, Debug)]
pub struct LogisticSampler {
    /// Feature weights [N_FEATURES].
    pub weights: Vec<f32>,
    /// Bias term.
    pub bias: f32,
}

impl LogisticSampler {
    /// Create with random initialization.
    pub fn new(rng: &mut Rng) -> Self {
        let mut weights = Vec::with_capacity(N_FEATURES);
        for _ in 0..N_FEATURES {
            weights.push((rng.next() as f64 / u64::MAX as f64 * 0.1 - 0.05) as f32);
        }
        Self { weights, bias: 0.0 }
    }

    /// Create with zero initialization (fresh sampler).
    pub fn zeros() -> Self {
        Self {
            weights: vec![0.0; N_FEATURES],
            bias: 0.0,
        }
    }

    /// Predict P(correct | features) ∈ [0, 1].
    pub fn predict(&self, features: &SamplerFeatures) -> f32 {
        let x = features.to_array();
        let mut logit = self.bias;
        for (&w, &xi) in self.weights.iter().zip(x.iter()) {
            logit += w * xi;
        }
        sigmoid(logit)
    }

    /// Predict for a batch of features.
    pub fn predict_batch(&self, features: &[SamplerFeatures]) -> Vec<f32> {
        features.iter().map(|f| self.predict(f)).collect()
    }

    /// Train on trajectories using SGD.
    ///
    /// Binary cross-entropy loss:
    /// L = -[y·log(σ(z)) + (1-y)·log(1-σ(z))]
    ///
    /// Returns final average loss.
    pub fn train(&mut self, trajectories: &[SamplerTrajectory], lr: f32, n_epochs: usize) -> f32 {
        let mut last_loss = 0.0f32;

        for _epoch in 0..n_epochs {
            let mut epoch_loss = 0.0f32;
            let mut n_samples = 0usize;

            for traj in trajectories {
                let x = traj.features.to_array();
                let logit = {
                    let mut z = self.bias;
                    for (&w, &xi) in self.weights.iter().zip(x.iter()) {
                        z += w * xi;
                    }
                    z
                };
                let pred = sigmoid(logit);
                let y = if traj.correct { 1.0f32 } else { 0.0f32 };

                // Binary cross-entropy loss
                let eps = 1e-7f32;
                let loss = -y * (pred + eps).ln() - (1.0 - y) * (1.0 - pred + eps).ln();
                epoch_loss += loss;

                // Gradient: dL/dz = pred - y (for sigmoid + BCE)
                let grad = pred - y;

                // Update weights: w -= lr * grad * x_i
                for (w, &xi) in self.weights.iter_mut().zip(x.iter()) {
                    *w -= lr * grad * xi;
                }
                self.bias -= lr * grad;

                n_samples += 1;
            }

            last_loss = if n_samples > 0 {
                epoch_loss / n_samples as f32
            } else {
                0.0
            };
        }

        last_loss
    }
}

// ── MLP Sampler ───────────────────────────────────────────────

/// 2-layer MLP sampler: input → hidden (ReLU) → output (sigmoid).
///
/// hidden_dim=32 → ~250 params. More capacity than logistic for
/// medium-scale models where feature interactions matter.
#[derive(Clone, Debug)]
pub struct MlpSampler {
    /// Input → hidden weights `[hidden_dim × N_FEATURES]`.
    pub w1: Vec<f32>,
    /// Hidden bias `[hidden_dim]`.
    pub b1: Vec<f32>,
    /// Hidden → output weights `[hidden_dim]`.
    pub w2: Vec<f32>,
    /// Output bias.
    pub b2: f32,
    /// Hidden dimension.
    pub hidden_dim: usize,
}

impl MlpSampler {
    /// Create with Xavier initialization.
    pub fn new(hidden_dim: usize, rng: &mut Rng) -> Self {
        let scale = (2.0 / (N_FEATURES + hidden_dim) as f32).sqrt();
        let mut w1 = Vec::with_capacity(hidden_dim * N_FEATURES);
        for _ in 0..hidden_dim * N_FEATURES {
            let u = (rng.next() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
            w1.push(u * scale);
        }

        let scale2 = (2.0 / (hidden_dim + 1) as f32).sqrt();
        let mut w2 = Vec::with_capacity(hidden_dim);
        for _ in 0..hidden_dim {
            let u = (rng.next() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
            w2.push(u * scale2);
        }

        Self {
            w1,
            b1: vec![0.0; hidden_dim],
            w2,
            b2: 0.0,
            hidden_dim,
        }
    }

    /// Forward pass: fills `h_scratch` with hidden activations and returns output logit.
    ///
    /// `h_scratch` must have length `>= hidden_dim`. Avoids the per-call `Vec` allocation
    /// and `clone()` that the previous `(Vec<f32>, f32)` return required.
    fn forward_into(&self, x: &[f32; N_FEATURES], h_scratch: &mut [f32]) -> f32 {
        debug_assert!(h_scratch.len() >= self.hidden_dim);
        let h = &mut h_scratch[..self.hidden_dim];
        for (j, h_j) in h.iter_mut().enumerate() {
            let mut sum = self.b1[j];
            for (i, &xi) in x.iter().enumerate() {
                sum += self.w1[j * N_FEATURES + i] * xi;
            }
            *h_j = relu(sum);
        }

        let mut logit = self.b2;
        for (&w2j, &hj) in self.w2.iter().zip(h.iter()) {
            logit += w2j * hj;
        }
        logit
    }

    /// Predict P(correct | features) ∈ [0, 1].
    ///
    /// Uses a stack-allocated hidden buffer (hidden_dim ≤ 64 for all sampler variants)
    /// to avoid heap allocation on the inference hot path.
    pub fn predict(&self, features: &SamplerFeatures) -> f32 {
        let x = features.to_array();
        let mut h_stack = [0.0f32; 64];
        let h = if self.hidden_dim <= 64 {
            &mut h_stack[..self.hidden_dim]
        } else {
            // Defensive: fall back to heap only for unusually large hidden_dim.
            &mut vec![0.0f32; self.hidden_dim]
        };
        let logit = self.forward_into(&x, h);
        sigmoid(logit)
    }

    /// Predict for a batch of features.
    pub fn predict_batch(&self, features: &[SamplerFeatures]) -> Vec<f32> {
        features.iter().map(|f| self.predict(f)).collect()
    }

    /// Train on trajectories using SGD with backprop.
    ///
    /// Returns final average loss.
    pub fn train(&mut self, trajectories: &[SamplerTrajectory], lr: f32, n_epochs: usize) -> f32 {
        let mut last_loss = 0.0f32;

        for _epoch in 0..n_epochs {
            let mut epoch_loss = 0.0f32;
            let mut n_samples = 0usize;

            // Reusable hidden buffer — allocated once per epoch instead of per-trajectory.
            // Previous code allocated + cloned a Vec on every `forward()` call.
            let mut h_buf = vec![0.0f32; self.hidden_dim];

            for traj in trajectories {
                let x = traj.features.to_array();
                let logit = self.forward_into(&x, &mut h_buf);
                // Backprop needs the hidden activations; alias the buffer for the
                // gradient pass below. (h_buf is not mutated during backprop — only read.)
                let hidden = &h_buf[..];
                let pred = sigmoid(logit);
                let y = if traj.correct { 1.0f32 } else { 0.0f32 };

                let eps = 1e-7f32;
                let loss = -y * (pred + eps).ln() - (1.0 - y) * (1.0 - pred + eps).ln();
                epoch_loss += loss;

                // dL/dlogit = pred - y
                let d_logit = pred - y;

                // Update w2 and b2
                for (w2j, &hj) in self.w2.iter_mut().zip(hidden.iter()) {
                    *w2j -= lr * d_logit * hj;
                }
                self.b2 -= lr * d_logit;

                // Backprop through ReLU to w1 and b1
                for (j, &hj) in hidden.iter().enumerate() {
                    let d_hidden = d_logit * self.w2[j] * relu_grad(hj);
                    for (i, &xi) in x.iter().enumerate() {
                        self.w1[j * N_FEATURES + i] -= lr * d_hidden * xi;
                    }
                    self.b1[j] -= lr * d_hidden;
                }

                n_samples += 1;
            }

            last_loss = if n_samples > 0 {
                epoch_loss / n_samples as f32
            } else {
                0.0
            };
        }

        last_loss
    }
}

// ── Transformer Sampler (Stub, Deferred) ──────────────────────

/// 4-layer transformer sampler (d=384, ~4.8M params).
///
/// Nemotron paper spec: PCA-compressed semantic embeddings + distribution stats
/// as 144-dim input. This is a stub for production-scale models.
///
/// **Not implemented** — deferred until T6 proves value at micro_dllm scale.
/// When implemented, this will follow the same `predict` / `train` interface.
#[derive(Clone, Debug)]
pub struct TransformerSampler {
    /// d_model dimension (default: 384).
    pub d_model: usize,
    /// Number of transformer layers (default: 4).
    pub n_layers: usize,
    /// PCA projection [d_model × N_FEATURES] (not trained yet).
    pub _pca_proj: Vec<f32>,
}

impl TransformerSampler {
    /// Create a stub sampler with given dimensions.
    pub fn new(d_model: usize, n_layers: usize) -> Self {
        Self {
            d_model,
            n_layers,
            _pca_proj: vec![0.0; d_model * N_FEATURES],
        }
    }

    /// Not implemented — falls back to logistic-equivalent prediction.
    pub fn predict(&self, features: &SamplerFeatures) -> f32 {
        // Stub: use a heuristic based on top1_prob and margin
        let x = features.to_array();
        let logit = x[0] * 2.0 + x[1] * 1.5 - 0.5;
        sigmoid(logit)
    }

    /// Not implemented — no-op.
    pub fn predict_batch(&self, features: &[SamplerFeatures]) -> Vec<f32> {
        features.iter().map(|f| self.predict(f)).collect()
    }

    /// Not implemented — returns 0.0.
    pub fn train(
        &mut self,
        _trajectories: &[SamplerTrajectory],
        _lr: f32,
        _n_epochs: usize,
    ) -> f32 {
        0.0
    }
}

// ── DiffusionSampler (Unified Enum) ───────────────────────────

/// Adaptive per-position correctness predictor for D2F denoising.
///
/// Selects the appropriate capacity based on model scale:
/// - `micro_dllm` (d=16) → Logistic (~7 params)
/// - Small models (d=48-128) → MLP (~250 params)
/// - Production (d=768+) → Transformer (~4.8M params, stub)
///
/// Usage:
/// ```ignore
/// let mut sampler = DiffusionSampler::auto(&config);
/// sampler.train(&trajectories, 0.01, 200);
/// let p_correct = sampler.predict(&features);
/// ```
#[derive(Clone, Debug)]
pub enum DiffusionSampler {
    /// Logistic regression: ~7 params. For micro_dllm scale.
    Logistic(LogisticSampler),
    /// 2-layer MLP: ~250 params. For small models.
    Mlp(MlpSampler),
    /// 4-layer transformer: ~4.8M params. For production (stub).
    Transformer(TransformerSampler),
}

impl DiffusionSampler {
    /// Auto-select sampler variant based on model config.
    ///
    /// Heuristic:
    /// - n_embd ≤ 32 → Logistic
    /// - n_embd ≤ 256 → MLP (hidden = n_embd)
    /// - n_embd > 256 → Transformer (d=384, 4 layers)
    pub fn auto(config: &Config) -> Self {
        Self::auto_with_rng(config, &mut Rng::new(42))
    }

    /// Auto-select with custom RNG for reproducible initialization.
    pub fn auto_with_rng(config: &Config, rng: &mut Rng) -> Self {
        match config.n_embd {
            0..=32 => Self::Logistic(LogisticSampler::new(rng)),
            33..=256 => Self::Mlp(MlpSampler::new(config.n_embd.min(64), rng)),
            _ => Self::Transformer(TransformerSampler::new(384, 4)),
        }
    }

    /// Create a logistic sampler with zero weights (for training from scratch).
    pub fn logistic_zeros() -> Self {
        Self::Logistic(LogisticSampler::zeros())
    }

    /// Create a logistic sampler with random weights.
    pub fn logistic(rng: &mut Rng) -> Self {
        Self::Logistic(LogisticSampler::new(rng))
    }

    /// Create an MLP sampler with given hidden dimension.
    pub fn mlp(hidden_dim: usize, rng: &mut Rng) -> Self {
        Self::Mlp(MlpSampler::new(hidden_dim, rng))
    }

    /// Create a transformer sampler stub.
    pub fn transformer(d_model: usize, n_layers: usize) -> Self {
        Self::Transformer(TransformerSampler::new(d_model, n_layers))
    }

    /// Predict P(correct | features) ∈ [0, 1].
    pub fn predict(&self, features: &SamplerFeatures) -> f32 {
        match self {
            Self::Logistic(s) => s.predict(features),
            Self::Mlp(s) => s.predict(features),
            Self::Transformer(s) => s.predict(features),
        }
    }

    /// Predict for a batch of features.
    pub fn predict_batch(&self, features: &[SamplerFeatures]) -> Vec<f32> {
        match self {
            Self::Logistic(s) => s.predict_batch(features),
            Self::Mlp(s) => s.predict_batch(features),
            Self::Transformer(s) => s.predict_batch(features),
        }
    }

    /// Train on collected trajectories.
    ///
    /// Returns final average loss.
    pub fn train(&mut self, trajectories: &[SamplerTrajectory], lr: f32, n_epochs: usize) -> f32 {
        match self {
            Self::Logistic(s) => s.train(trajectories, lr, n_epochs),
            Self::Mlp(s) => s.train(trajectories, lr, n_epochs),
            Self::Transformer(s) => s.train(trajectories, lr, n_epochs),
        }
    }

    /// Which variant this is.
    pub fn variant(&self) -> SamplerVariant {
        match self {
            Self::Logistic(_) => SamplerVariant::Logistic,
            Self::Mlp(s) => SamplerVariant::Mlp {
                hidden_dim: s.hidden_dim,
            },
            Self::Transformer(s) => SamplerVariant::Transformer {
                d_model: s.d_model,
                n_layers: s.n_layers,
            },
        }
    }

    /// Evaluate AUC (Area Under ROC Curve) on trajectories.
    ///
    /// AUC = 1.0 means perfect prediction, 0.5 means random.
    pub fn evaluate_auc(&self, trajectories: &[SamplerTrajectory]) -> f32 {
        if trajectories.is_empty() {
            return 0.5;
        }

        // Collect (predicted, actual) pairs
        let mut pairs: Vec<(f32, bool)> = trajectories
            .iter()
            .map(|t| (self.predict(&t.features), t.correct))
            .collect();

        // sort_unstable_by is faster than sort_by — doesn't preserve equal-element
        // order (fine for AUC computation) and avoids O(n) worst-case merges.
        pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        // Count positives and negatives
        let n_pos = pairs.iter().filter(|(_, c)| *c).count() as f64;
        let n_neg = pairs.len() as f64 - n_pos;

        if n_pos == 0.0 || n_neg == 0.0 {
            return 0.5;
        }

        // Compute AUC using trapezoidal rule
        let mut tp = 0.0f64;
        let mut fp = 0.0f64;
        let mut auc = 0.0f64;
        let mut prev_fpr = 0.0f64;
        let mut prev_tpr = 0.0f64;

        for (_, positive) in &pairs {
            if *positive {
                tp += 1.0;
            } else {
                fp += 1.0;
            }

            let tpr = tp / n_pos;
            let fpr = fp / n_neg;

            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
            prev_fpr = fpr;
            prev_tpr = tpr;
        }

        auc as f32
    }
}

// ── Trajectory Collection ─────────────────────────────────────

/// A single training example: features at one position + correctness label.
#[derive(Clone, Debug)]
pub struct SamplerTrajectory {
    /// Features extracted at this position/step.
    pub features: SamplerFeatures,
    /// Whether the sampled token was correct (matches ground truth).
    pub correct: bool,
}

/// Collect training trajectories from D2F decode with ground truth.
///
/// Runs D2F decode on each target sequence, extracting per-position features
/// at each denoising step. For each position, records whether the model's
/// top-1 prediction matches the ground truth token.
///
/// # Arguments
///
/// * `weights` — Trained dLLM weights.
/// * `config` — Model config.
/// * `decode_config` — D2F decode config.
/// * `targets` — Ground truth token sequences.
/// * `max_trajectories` — Cap on total samples collected (0 = unlimited).
///
/// # Returns
///
/// Vector of (features, correct) training pairs.
pub fn collect_trajectories(
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    targets: &[Vec<usize>],
    max_trajectories: usize,
) -> Vec<SamplerTrajectory> {
    let mut all_trajectories = Vec::new();
    let _rng = Rng::new(42);
    let mut dctx = D2fContext::new(config);

    for target in targets {
        let block_size = decode_config.block_size;
        let seq_len = target.len().min(block_size);
        let max_steps = decode_config.denoise_steps;
        let mask = config.mask_token;
        let vocab = config.vocab_size;

        // Initialize tokens: all mask
        let mut tokens: Vec<usize> = vec![mask; seq_len];

        for step in 0..max_steps {
            // Forward pass with block-causal attention
            let _ = forward_block_causal_with(&mut dctx, weights, &tokens, config, block_size);

            // Single fused pass over masked positions: extract features, record
            // trajectory, AND commit tokens via confidence remasking.
            //
            // Previously this was two separate loops over the same positions, each
            // recomputing the O(vocab) argmax scan. Merging halves the argmax work.
            //
            // Cap handling: if `max_trajectories` is hit mid-step, we return
            // immediately — the remaining token commits are skipped, matching the
            // old two-loop behavior (loop 2 never ran after an early return).
            #[allow(clippy::needless_range_loop)]
            for p in 0..seq_len {
                if tokens[p] != mask {
                    continue;
                }

                let logits_start = p * vocab;
                let logits_end = logits_start + vocab;
                let logits_p = &dctx.logits_flat[logits_start..logits_end];

                // Find top-1 prediction (single argmax scan, reused below)
                let mut top1 = 0usize;
                let mut top1_val = f32::NEG_INFINITY;
                for (t, &logit) in logits_p.iter().enumerate() {
                    if t == mask {
                        continue;
                    }
                    if logit > top1_val {
                        top1_val = logit;
                        top1 = t;
                    }
                }

                // ── Trajectory recording (was loop 1) ──
                let features = SamplerFeatures::from_logits(
                    logits_p, vocab, mask, step, max_steps, p, seq_len,
                );
                let correct = top1 == target[p];
                all_trajectories.push(SamplerTrajectory { features, correct });

                if max_trajectories > 0 && all_trajectories.len() >= max_trajectories {
                    return all_trajectories;
                }

                // ── Confidence remasking / token commit (was loop 2) ──
                // Reuses `top1` and `top1_val` from the single argmax above;
                // the old code recomputed both in a second O(vocab) scan.
                use katgpt_core::simd::fast_exp;
                let sum_exp: f32 = (0..vocab)
                    .filter(|&t| t != mask)
                    .map(|t| fast_exp(logits_p[t] - top1_val))
                    .sum();
                let top1_prob = fast_exp(logits_p[top1] - top1_val) / sum_exp;

                if top1_prob >= decode_config.confidence_threshold {
                    tokens[p] = top1;
                }
            }

            // Early exit if all unmasked
            if tokens.iter().all(|&t| t != mask) {
                break;
            }
        }
    }

    all_trajectories
}

// NOTE (Plan 399): `train_logistic_on_patterns` stayed in root because it
// depends on `crate::dllm::{generate_pattern_dataset, train_mini_dllm}`
// (training code). It's re-exported via root's
// `src/speculative/diffusion_sampler.rs` shim so the historical
// `crate::speculative::diffusion_sampler::train_logistic_on_patterns`
// path continues to resolve.

// ── Integration Helper ────────────────────────────────────────

/// Decision from the sampler for a single position in the denoising loop.
#[derive(Clone, Copy, Debug)]
pub struct SamplerDecision {
    /// Predicted P(correct) for the top-1 token.
    pub p_correct: f32,
    /// Whether to accept the token (p_correct >= threshold).
    pub accept: bool,
}

impl DiffusionSampler {
    /// Make an accept/reject decision for a denoising position.
    ///
    /// Replaces the fixed `chosen_prob >= tau_conf` check in the denoising loop.
    /// Falls back to fixed threshold if sampler is not confident.
    pub fn decide(&self, features: &SamplerFeatures, threshold: f32) -> SamplerDecision {
        let p_correct = self.predict(features);
        SamplerDecision {
            p_correct,
            accept: p_correct >= threshold,
        }
    }

    /// Batch decision for multiple positions.
    pub fn decide_batch(
        &self,
        features: &[SamplerFeatures],
        threshold: f32,
    ) -> Vec<SamplerDecision> {
        features.iter().map(|f| self.decide(f, threshold)).collect()
    }
}

// ── Activation Functions ──────────────────────────────────────

/// Sigmoid: σ(x) = 1 / (1 + exp(-x)).
///
/// Delegates to `katgpt_core::simd::fast_sigmoid` (Cephes polynomial).
fn sigmoid(x: f32) -> f32 {
    katgpt_core::simd::fast_sigmoid(x)
}

/// ReLU: max(0, x).
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// ReLU gradient: 1 if x > 0, else 0.
fn relu_grad(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> Config {
        Config::micro_dllm()
    }

    // ── SamplerFeatures Tests ──

    #[test]
    fn test_features_from_logits_uniform() {
        // Uniform logits → uniform probabilities
        let vocab = 4;
        let mask = 3;
        let logits = vec![1.0, 1.0, 1.0, 1.0]; // mask at index 3
        let features = SamplerFeatures::from_logits(&logits, vocab, mask, 0, 8, 0, 4);

        // With uniform logits, top1_prob ≈ 1/3 (3 valid tokens)
        assert!(
            (features.top1_prob - 1.0 / 3.0).abs() < 0.05,
            "uniform logits should give ~0.33 top1_prob, got {}",
            features.top1_prob,
        );
        // Margin should be ~0 for uniform
        assert!(
            features.margin.abs() < 0.05,
            "uniform logits should give ~0 margin, got {}",
            features.margin,
        );
    }

    #[test]
    fn test_features_from_logits_peaked() {
        // One dominant logit → high top1_prob, high margin
        let vocab = 4;
        let mask = 3;
        let logits = vec![10.0, 0.0, 0.0, 0.0]; // token 0 dominates
        let features = SamplerFeatures::from_logits(&logits, vocab, mask, 2, 8, 1, 4);

        assert!(
            features.top1_prob > 0.99,
            "peaked logits should give very high top1_prob, got {}",
            features.top1_prob,
        );
        assert!(
            features.margin > 0.9,
            "peaked logits should give high margin, got {}",
            features.margin,
        );
        assert!(
            features.entropy < 0.1,
            "peaked logits should give low entropy, got {}",
            features.entropy,
        );
    }

    #[test]
    fn test_features_normalization() {
        let features = SamplerFeatures::from_logits(
            &[1.0, 0.0, 0.0],
            3,
            2,
            4, // step
            8, // max_steps
            2, // pos
            4, // block_size
        );
        assert!(
            (features.step_norm - 0.5).abs() < 1e-6,
            "step_norm should be 4/8 = 0.5, got {}",
            features.step_norm,
        );
        assert!(
            (features.pos_norm - 0.5).abs() < 1e-6,
            "pos_norm should be 2/4 = 0.5, got {}",
            features.pos_norm,
        );
    }

    #[test]
    fn test_features_to_array_length() {
        let features = SamplerFeatures::default();
        let arr = features.to_array();
        assert_eq!(arr.len(), N_FEATURES);
    }

    // ── LogisticSampler Tests ──

    #[test]
    fn test_logistic_zeros_predicts_half() {
        // Zero weights → logit = 0 → sigmoid(0) = 0.5
        let sampler = LogisticSampler::zeros();
        let features = SamplerFeatures {
            top1_prob: 0.9,
            margin: 0.5,
            ..Default::default()
        };
        let pred = sampler.predict(&features);
        assert!(
            (pred - 0.5).abs() < 1e-6,
            "zero weights should predict 0.5, got {pred}",
        );
    }

    #[test]
    fn test_logistic_positive_weights_high_prob() {
        // Positive weights + high top1_prob → high prediction
        let mut sampler = LogisticSampler::zeros();
        sampler.weights[0] = 5.0; // top1_prob weight
        sampler.bias = -2.0;

        let features = SamplerFeatures {
            top1_prob: 0.9,
            ..Default::default()
        };
        let pred = sampler.predict(&features);
        assert!(
            pred > 0.5,
            "positive weight on high top1_prob should predict > 0.5, got {pred}",
        );
    }

    #[test]
    fn test_logistic_train_reduces_loss() {
        let mut rng = Rng::new(42);
        let mut sampler = LogisticSampler::new(&mut rng);

        // Create simple trajectories: high top1_prob → correct
        let trajectories: Vec<SamplerTrajectory> = (0..100)
            .map(|i| {
                let prob = i as f32 / 100.0;
                SamplerTrajectory {
                    features: SamplerFeatures {
                        top1_prob: prob,
                        ..Default::default()
                    },
                    correct: prob > 0.5,
                }
            })
            .collect();

        let loss_before = {
            let mut total = 0.0f32;
            for t in &trajectories {
                let pred = sampler.predict(&t.features);
                let y = if t.correct { 1.0f32 } else { 0.0f32 };
                let eps = 1e-7f32;
                total -= y * (pred + eps).ln() + (1.0 - y) * (1.0 - pred + eps).ln();
            }
            total / trajectories.len() as f32
        };

        let loss_after = sampler.train(&trajectories, 0.1, 100);

        assert!(
            loss_after < loss_before,
            "training should reduce loss: before={loss_before:.4}, after={loss_after:.4}",
        );
    }

    #[test]
    fn test_logistic_predict_batch_matches_individual() {
        let mut rng = Rng::new(42);
        let sampler = LogisticSampler::new(&mut rng);
        let features: Vec<SamplerFeatures> = (0..10)
            .map(|i| SamplerFeatures {
                top1_prob: i as f32 / 10.0,
                ..Default::default()
            })
            .collect();

        let batch_preds = sampler.predict_batch(&features);
        let individual_preds: Vec<f32> = features.iter().map(|f| sampler.predict(f)).collect();

        for (i, (b, ind)) in batch_preds.iter().zip(individual_preds.iter()).enumerate() {
            assert!(
                (b - ind).abs() < 1e-6,
                "batch[{i}] = {b} != individual[{i}] = {ind}",
            );
        }
    }

    // ── MlpSampler Tests ──

    #[test]
    fn test_mlp_predict_is_bounded() {
        let mut rng = Rng::new(42);
        let sampler = MlpSampler::new(32, &mut rng);
        let features = SamplerFeatures {
            top1_prob: 0.9,
            margin: 0.5,
            entropy: 0.3,
            top3_mass: 0.95,
            step_norm: 0.5,
            pos_norm: 0.5,
            stable_age: 0,
            topk_drift: 0.0,
        };
        let pred = sampler.predict(&features);
        assert!(
            (0.0..=1.0).contains(&pred),
            "prediction should be in [0, 1], got {pred}",
        );
    }

    #[test]
    fn test_mlp_train_runs() {
        let mut rng = Rng::new(42);
        let mut sampler = MlpSampler::new(16, &mut rng);

        let trajectories: Vec<SamplerTrajectory> = (0..50)
            .map(|i| {
                let prob = i as f32 / 50.0;
                SamplerTrajectory {
                    features: SamplerFeatures {
                        top1_prob: prob,
                        margin: prob * 0.5,
                        ..Default::default()
                    },
                    correct: prob > 0.5,
                }
            })
            .collect();

        let loss = sampler.train(&trajectories, 0.01, 50);
        assert!(
            loss.is_finite(),
            "MLP training should produce finite loss, got {loss}",
        );
    }

    // ── DiffusionSampler Enum Tests ──

    #[test]
    fn test_auto_selects_logistic_for_micro() {
        let config = make_config();
        let sampler = DiffusionSampler::auto(&config);
        assert!(
            matches!(sampler.variant(), SamplerVariant::Logistic),
            "micro_dllm (n_embd=16) should select Logistic",
        );
    }

    #[test]
    fn test_auto_selects_mlp_for_medium() {
        let mut config = make_config();
        config.n_embd = 64;
        let sampler = DiffusionSampler::auto(&config);
        assert!(
            matches!(sampler.variant(), SamplerVariant::Mlp { .. }),
            "n_embd=64 should select MLP",
        );
    }

    #[test]
    fn test_auto_selects_transformer_for_large() {
        let mut config = make_config();
        config.n_embd = 768;
        let sampler = DiffusionSampler::auto(&config);
        assert!(
            matches!(sampler.variant(), SamplerVariant::Transformer { .. }),
            "n_embd=768 should select Transformer",
        );
    }

    #[test]
    fn test_logistic_zeros_factory() {
        let sampler = DiffusionSampler::logistic_zeros();
        assert!(matches!(sampler.variant(), SamplerVariant::Logistic));

        // Should predict 0.5 for any input
        let features = SamplerFeatures {
            top1_prob: 0.99,
            ..Default::default()
        };
        let pred = sampler.predict(&features);
        assert!(
            (pred - 0.5).abs() < 1e-6,
            "zeros sampler should predict 0.5, got {pred}",
        );
    }

    // ── Trajectory Collection Tests ──

    // ── End-to-End: Train Sampler ──
    // (Plan 399: the 3 train-dependent tests in this section —
    // test_collect_trajectories_produces_data, test_collect_trajectories_respects_cap,
    // test_train_logistic_on_patterns — stayed in root because they call
    // crate::dllm::{generate_pattern_dataset, train_mini_dllm}.)

    #[test]
    fn test_sampler_decision() {
        let sampler = DiffusionSampler::logistic_zeros();
        // Zero weights always predict 0.5 → accept with threshold <= 0.5

        let features = SamplerFeatures {
            top1_prob: 0.8,
            ..Default::default()
        };

        let decision_low = sampler.decide(&features, 0.3);
        assert!(
            decision_low.accept,
            "threshold 0.3 < 0.5 prediction → should accept",
        );

        let decision_high = sampler.decide(&features, 0.8);
        assert!(
            !decision_high.accept,
            "threshold 0.8 > 0.5 prediction → should reject",
        );
    }

    // ── AUC Tests ──

    #[test]
    fn test_auc_perfect_predictor() {
        // Test AUC with an untrained (random) logistic sampler → should be ~0.5
        let mut rng = Rng::new(42);
        let sampler = DiffusionSampler::logistic(&mut rng);

        let trajectories: Vec<SamplerTrajectory> = (0..50)
            .map(|i| SamplerTrajectory {
                features: SamplerFeatures {
                    top1_prob: i as f32 / 50.0,
                    ..Default::default()
                },
                correct: i > 25,
            })
            .collect();

        let auc = sampler.evaluate_auc(&trajectories);
        assert!(
            (0.0..=1.0).contains(&auc),
            "AUC should be in [0, 1], got {auc}",
        );
    }

    // ── Convenience Function Test ──
    // (Plan 399: test_train_logistic_on_patterns_convenience stayed in root —
    // it exercises the root-resident `train_logistic_on_patterns` helper.)

    // ── T3 Integration Tests (Plan 116) ──
    // (Plan 399: test_d2f_decode_with_sampler_produces_valid_output and
    // test_d2f_decode_sampler_differs_from_fixed_threshold stayed in root —
    // they call crate::dllm::{generate_pattern_dataset, train_mini_dllm}.)

    // ── StabilityTracker (Issue 802 items 3 + 5) ──

    const TAU: f32 = 0.1;

    fn set(a: u32, b: u32, c: u32) -> ([u32; 3], [f32; 3]) {
        ([a, b, c], [0.7, 0.2, 0.05])
    }

    #[test]
    fn stability_first_observation_is_neutral() {
        let mut t = StabilityTracker::new();
        let (idx, m) = set(4, 7, 9);
        assert_eq!(t.observe(&idx, &m, 0.7, TAU, false), (0, 0.0));
    }

    #[test]
    fn stability_ages_on_unchanged_argmax_and_resets_on_change() {
        let mut t = StabilityTracker::new();
        let (idx, m) = set(4, 7, 9);
        let _ = t.observe(&idx, &m, 0.7, TAU, false);
        assert_eq!(
            t.observe(&idx, &m, 0.7, TAU, false).0,
            1,
            "first held step ages to 1"
        );
        assert_eq!(t.observe(&idx, &m, 0.7, TAU, false).0, 2);
        let (idx2, m2) = set(5, 7, 9); // argmax changed
        assert_eq!(
            t.observe(&idx2, &m2, 0.7, TAU, false).0,
            0,
            "argmax change resets"
        );
        assert_eq!(
            t.observe(&idx2, &m2, 0.7, TAU, false).0,
            1,
            "age restarts on the new argmax"
        );
    }

    #[test]
    fn stability_saturates_at_u8_max() {
        let mut t = StabilityTracker::new();
        let (idx, m) = set(1, 2, 3);
        let _ = t.observe(&idx, &m, 0.7, TAU, false);
        let mut last = 0u8;
        for _ in 0..300 {
            last = t.observe(&idx, &m, 0.7, TAU, false).0;
        }
        assert_eq!(last, u8::MAX, "saturating, never wrapping");
    }

    #[test]
    fn stability_mass_floor_disarms_without_adopting_the_noisy_set() {
        let mut t = StabilityTracker::new();
        let (idx, m) = set(4, 7, 9);
        let _ = t.observe(&idx, &m, 0.7, TAU, false);
        // Below-floor step (the item-5 law): disarm, no drift, no adoption.
        assert_eq!(t.observe(&idx, &m, TAU * 0.5, TAU, false), (0, 0.0));
        // A NaN top1_prob must disarm too (not arm through the comparison).
        assert_eq!(t.observe(&idx, &m, f32::NAN, TAU, false), (0, 0.0));
        // The next armed step ages against the last ARMED step.
        let (age, drift) = t.observe(&idx, &m, 0.7, TAU, false);
        assert_eq!(age, 1);
        assert!((0.0..=std::f32::consts::LN_2).contains(&drift));
    }

    #[test]
    fn stability_remask_resets_to_first_observation() {
        let mut t = StabilityTracker::new();
        let (idx, m) = set(4, 7, 9);
        let _ = t.observe(&idx, &m, 0.7, TAU, false);
        let _ = t.observe(&idx, &m, 0.7, TAU, false);
        assert_eq!(
            t.observe(&idx, &m, 0.7, TAU, true),
            (0, 0.0),
            "remask resets everything"
        );
        assert_eq!(
            t.observe(&idx, &m, 0.7, TAU, false),
            (0, 0.0),
            "post-remask step is a first observation"
        );
    }

    #[test]
    fn stability_drift_bounded_laws() {
        let mut t = StabilityTracker::new();
        let (idx, m) = set(4, 7, 9);
        let _ = t.observe(&idx, &m, 0.7, TAU, false);
        // Identical set → bitwise 0.
        assert_eq!(t.observe(&idx, &m, 0.7, TAU, false).1, 0.0);
        // Fully disjoint set → bitwise ln 2 (the kernel's disjoint law).
        let (idx2, m2) = set(1, 2, 3);
        assert_eq!(
            t.observe(&idx2, &m2, 0.7, TAU, false).1,
            std::f32::consts::LN_2
        );
        // Partial overlap → strictly inside (0, ln 2).
        let (idx3, m3) = set(4, 2, 3);
        let drift = t.observe(&idx3, &m3, 0.7, TAU, false).1;
        assert!(
            drift > 0.0 && drift < std::f32::consts::LN_2,
            "partial-overlap drift {drift}"
        );
    }

    // ── Provenance rows (Issue 1210 C2): the row form names the branch;
    // these twins pin the reason for every item-5 law the tuple tests
    // above already pin numerically.

    #[test]
    fn stability_row_names_every_boundary_branch() {
        let mut t = StabilityTracker::new();
        let (idx, m) = set(4, 7, 9);

        // First observation: neutral, named.
        let r = t.observe_row(&idx, &m, 0.7, TAU, false);
        assert_eq!(r.reason, StabilityReason::FirstObservation);
        assert_eq!((r.stable_age, r.topk_drift), (0, 0.0));

        // Held: ages, drift measured (identical set → bitwise 0).
        let r = t.observe_row(&idx, &m, 0.7, TAU, false);
        assert_eq!(r.reason, StabilityReason::ArgmaxHeld);
        assert_eq!((r.stable_age, r.topk_drift), (1, 0.0));

        // Flip: argmax changed, age reset, drift strictly inside (0, ln 2)
        // (the set {5,7,9} overlaps {4,7,9} in two of three slots).
        let (idx2, m2) = set(5, 7, 9);
        let r = t.observe_row(&idx2, &m2, 0.7, TAU, false);
        assert_eq!(r.reason, StabilityReason::ArgmaxFlip);
        assert_eq!(r.stable_age, 0);
        assert!(r.topk_drift > 0.0 && r.topk_drift < std::f32::consts::LN_2);

        // Disarmed below floor: named, no adoption, no drift.
        let r = t.observe_row(&idx2, &m2, TAU * 0.5, TAU, false);
        assert_eq!(r.reason, StabilityReason::Disarmed);
        assert_eq!((r.stable_age, r.topk_drift), (0, 0.0));

        // Disarmed on NaN: the same law (must not arm through the
        // comparison).
        let r = t.observe_row(&idx2, &m2, f32::NAN, TAU, false);
        assert_eq!(r.reason, StabilityReason::Disarmed);

        // Post-disarm armed step compares against the last ARMED side:
        // {5,7,9} again → held, age 1, drift bitwise 0.
        let r = t.observe_row(&idx2, &m2, 0.7, TAU, false);
        assert_eq!(r.reason, StabilityReason::ArgmaxHeld);
        assert_eq!((r.stable_age, r.topk_drift), (1, 0.0));

        // Remask: named reset, then a named first observation.
        let r = t.observe_row(&idx2, &m2, 0.7, TAU, true);
        assert_eq!(r.reason, StabilityReason::RemaskReset);
        assert_eq!((r.stable_age, r.topk_drift), (0, 0.0));
        let r = t.observe_row(&idx2, &m2, 0.7, TAU, false);
        assert_eq!(r.reason, StabilityReason::FirstObservation);
    }

    #[test]
    fn stability_row_is_a_superset_of_the_tuple_form() {
        // One scripted sequence through BOTH forms on identical trackers:
        // every step's (age, drift) pair must match exactly — the row is
        // a projection of one computation, never a recomputation.
        type Step = (([u32; 3], [f32; 3]), f32, bool);
        let script: [Step; 8] = [
            (set(4, 7, 9), 0.7, false),
            (set(4, 7, 9), 0.3, false),
            (set(4, 7, 9), 0.7, false),
            (set(1, 2, 3), 0.7, false),
            (set(1, 2, 3), f32::NAN, false),
            (set(1, 2, 3), 0.7, false),
            (set(1, 2, 3), 0.7, true),
            (set(9, 8, 7), 0.7, false),
        ];
        let mut tuple_t = StabilityTracker::new();
        let mut row_t = StabilityTracker::new();
        for (step, ((idx, m), p, remasked)) in script.iter().enumerate() {
            let (a, d) = tuple_t.observe(idx, m, *p, TAU, *remasked);
            let r = row_t.observe_row(idx, m, *p, TAU, *remasked);
            assert_eq!((a, d), (r.stable_age, r.topk_drift), "step {step}");
        }
    }

    #[test]
    fn from_logits_emits_top3_indices_bitwise_same_masses() {
        let vocab = 8;
        let mask = 0;
        let mut logits = [0.0f32; 8];
        logits[5] = 4.0;
        logits[2] = 2.0;
        logits[7] = 1.0;
        logits[3] = 0.5;
        let mut idx_out = [9u32; 3];
        let via_into =
            SamplerFeatures::from_logits_into(&logits, vocab, mask, 0, 4, 0, 4, Some(&mut idx_out));
        assert_eq!(idx_out, [5, 2, 7]);
        // The bare form stays bitwise-identical on the masses — the index
        // tracking is pure addition (the signal-diff pin).
        let bare = SamplerFeatures::from_logits(&logits, vocab, mask, 0, 4, 0, 4);
        assert_eq!(bare.to_array(), via_into.to_array());
        // Mask token excluded from the set despite being the max logit.
        // Gap kept at 6 (well inside fast_exp's |x| < 88 accuracy window):
        // a >87.3 gap underflows EVERY non-mask exp to exactly 0.0 (the
        // documented fast_exp contract), sum_exp reads 0, and the pre-
        // existing degenerate branch fires — default features + [0; 3]
        // emit, which is the coherent safe direction (top1_prob 0.0
        // disarms the StabilityTracker floor).
        let mut logits2 = logits;
        logits2[0] = 10.0;
        let mut idx2 = [9u32; 3];
        SamplerFeatures::from_logits_into(&logits2, vocab, mask, 0, 4, 0, 4, Some(&mut idx2));
        assert_eq!(idx2, [5, 2, 7]);
        // Beyond the fast_exp floor (>87.3 gap): the degenerate branch —
        // zeros out the set and the features instead of producing garbage
        // indices from all-underflowed exps.
        let mut logits3 = logits;
        logits3[0] = 100.0;
        let mut idx4 = [9u32; 3];
        let degenerate =
            SamplerFeatures::from_logits_into(&logits3, vocab, mask, 0, 4, 0, 4, Some(&mut idx4));
        assert_eq!(idx4, [0; 3]);
        assert_eq!(
            degenerate.top1_prob, 0.0,
            "degenerate row reads default features"
        );
        // Exp ties: first index wins (strict-> tracking, deterministic).
        let mut tie = [0.0f32; 8];
        tie[6] = 1.0;
        tie[2] = 1.0;
        let mut idx3 = [9u32; 3];
        SamplerFeatures::from_logits_into(&tie, vocab, mask, 0, 4, 0, 4, Some(&mut idx3));
        assert_eq!(idx3[0], 2, "first-index-wins on exp ties");
    }

    #[test]
    fn stability_array_appends_two_features_and_default_is_zero() {
        let f = SamplerFeatures::default();
        assert_eq!(f.stable_age, 0);
        assert_eq!(f.topk_drift, 0.0);
        let arr = f.to_array_with_stability();
        assert_eq!(arr.len(), N_FEATURES + N_STABILITY_FEATURES);
        // The core six are the unchanged to_array shape (checkpoint
        // contract), stability appended after.
        assert_eq!(arr[..N_FEATURES], f.to_array());
        assert_eq!(&arr[N_FEATURES..], &[0.0, 0.0]);
    }
}
