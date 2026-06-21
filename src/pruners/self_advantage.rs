//! Self-advantage computation from latent recursion pre/post logits.
//!
//! Distilled from [arxiv:2511.16886](https://arxiv.org/abs/2511.16886) —
//! "Latent Reasoning in TRMs is Secretly a Policy Improvement Operator"
//! (Asadulaev et al., ICML 2026). See `.research/250_*.md`, Plan 283.
//!
//! A single model, run twice (pre-recursion and post-recursion), produces a
//! self-advantage signal via log-ratio. No teacher, no oracle, no value
//! function. The math is structurally identical to SDPG's `centered_log_ratio`
//! (Plan 180), but sources both distributions from the same model's two passes
//! instead of oracle-vs-student bandits.
//!
//! # Three primitives
//!
//! | Primitive | Math | Operation |
//! |-----------|------|----------|
//! | [`self_advantage`] | `A(a) = log π+(a) − log π̂(a)` | Log-ratio of post/pre log-softmax |
//! | [`self_advantage_margin`] | `A(y*) − E_{a∼π+}[A(a)]` | Dead-compute detector (Eq. 18) |
//! | [`product_policy_log`] | `(1−w)·log π̂ + w·log π+` | Product-policy interpolation (Eq. 16) |
//!
//! # Zero allocation
//!
//! All functions write into caller-provided scratch buffers. The scratch layout
//! for [`self_advantage`] / [`self_advantage_margin`] is `[pre_lsm | post_lsm |
//! advantage]`, each of length `n = logits.len()`. Total: `3 * n`.

// ── Private helpers ─────────────────────────────────────────────

/// Compute log-softmax of `logits` into `out`.
///
/// Numerically stable: subtracts max before exp, then normalizes in log space.
/// `log_softmax(x)[i] = (x[i] - max(x)) - log(sum(exp(x[j] - max(x))))`
#[inline]
fn log_softmax_into(logits: &[f32], out: &mut [f32]) {
    let n = logits.len();
    debug_assert_eq!(out.len(), n);
    if n == 0 {
        return;
    }

    // Pass 1: find max for numerical stability.
    let mut max_val = f32::NEG_INFINITY;
    for &v in logits {
        if v > max_val {
            max_val = v;
        }
    }
    // Guard against all-NEG_INFINITY (shouldn't happen with real logits).
    if max_val == f32::NEG_INFINITY {
        let ln_n = (n as f32).ln();
        for slot in out.iter_mut() {
            *slot = -ln_n;
        }
        return;
    }

    // Pass 2: accumulate shifted exp + write shifted logits.
    let mut lse = 0.0f32; // Σ exp(x[i] - max), un-logged
    for i in 0..n {
        let shifted = logits[i] - max_val;
        out[i] = shifted;
        lse += shifted.exp();
    }
    let log_lse = lse.ln();

    // Pass 3: finalize log-softmax.
    for slot in out.iter_mut() {
        *slot -= log_lse;
    }
}

// ── Phase 1: Self-Advantage ─────────────────────────────────────

/// Compute the self-advantage `A(a) = log π+(a) − log π̂(a)` for all actions.
///
/// Returns a mutable slice into the advantage region of `scratch`
/// (`scratch[2*n..3*n]`). The caller should not modify `scratch` while
/// holding this reference.
///
/// # Arguments
///
/// * `pre_logits` — reference policy logits `π̂` (pre-recursion readout).
/// * `post_logits` — improved policy logits `π+` (post-recursion readout).
/// * `scratch` — buffer of length `>= 3 * pre_logits.len()`.
///   Layout after call: `[pre_logsoftmax | post_logsoftmax | advantage]`.
///
/// # Positive / negative semantics
///
/// * `A[a] > 0` — recursion step increased relative log-prob of action `a`.
/// * `A[a] < 0` — recursion step decreased relative log-prob of action `a`.
/// * `A[a] ≈ 0` — no change (dead compute for this action).
pub fn self_advantage<'a>(
    pre_logits: &[f32],
    post_logits: &[f32],
    scratch: &'a mut [f32],
) -> &'a mut [f32] {
    let n = pre_logits.len();
    assert_eq!(post_logits.len(), n);
    assert!(n > 0, "self_advantage: empty logits");
    assert!(
        scratch.len() >= 3 * n,
        "scratch must be >= 3 * logits.len() (got {}, need {})",
        scratch.len(),
        3 * n,
    );

    let (pre_lsm, rest) = scratch.split_at_mut(n);
    let (post_lsm, adv) = rest.split_at_mut(n);

    log_softmax_into(pre_logits, pre_lsm);
    log_softmax_into(post_logits, post_lsm);

    // Chunked subtraction for SIMD auto-vectorization (4-wide f32).
    // adv[i] = post_lsm[i] - pre_lsm[i]
    let chunks = n / 4;
    let mut i = 0;
    while i < chunks * 4 {
        adv[i] = post_lsm[i] - pre_lsm[i];
        adv[i + 1] = post_lsm[i + 1] - pre_lsm[i + 1];
        adv[i + 2] = post_lsm[i + 2] - pre_lsm[i + 2];
        adv[i + 3] = post_lsm[i + 3] - pre_lsm[i + 3];
        i += 4;
    }
    while i < n {
        adv[i] = post_lsm[i] - pre_lsm[i];
        i += 1;
    }

    adv
}

/// Advantage margin for a specific candidate action (Eq. 18).
///
/// `margin(candidate) = A(candidate) − E_{a∼π+}[A(a)]`
///
/// The expectation is computed under the post-recursion policy `π+`
/// (equivalently `w = 1.0` in the product-policy family). By the identity
/// `Σ_a π+(a)·log(π+(a)/π̂(a)) = KL(π+ ‖ π̂)`, this simplifies to:
///
/// ```text
/// margin(candidate) = log(π+(candidate) / π̂(candidate)) − KL(π+ ‖ π̂)
/// ```
///
/// # Returns
///
/// * **Positive** — the recursion step preferentially benefits `candidate`
///   above average. Accept the step.
/// * **Zero** — neutral. `candidate` tracks the mean improvement.
/// * **Negative** — the recursion step is dead compute (or harmful) for
///   `candidate`. Skip.
///
/// # Arguments
///
/// Same scratch contract as [`self_advantage`] (`>= 3 * n`).
pub fn self_advantage_margin(
    pre_logits: &[f32],
    post_logits: &[f32],
    candidate: usize,
    scratch: &mut [f32],
) -> f32 {
    let n = pre_logits.len();
    assert!(candidate < n, "candidate {} out of range (n={})", candidate, n);

    // Populate scratch: [pre_lsm | post_lsm | advantage].
    // We discard the returned &mut to release the borrow before reading.
    let _ = self_advantage(pre_logits, post_logits, scratch);

    // Re-borrow immutably for the expectation sum.
    let post_lsm = &scratch[n..2 * n];
    let adv = &scratch[2 * n..3 * n];

    // E_{a∼π+}[A(a)] = Σ_a exp(post_lsm[a]) * adv[a]
    //               = KL(π+ ‖ π̂)   [identity, see module docs]
    //
    // Chunked accumulation for SIMD.
    let chunks = n / 4;
    let mut expectation = 0.0f32;
    let mut i = 0;
    while i < chunks * 4 {
        expectation += post_lsm[i].exp() * adv[i];
        expectation += post_lsm[i + 1].exp() * adv[i + 1];
        expectation += post_lsm[i + 2].exp() * adv[i + 2];
        expectation += post_lsm[i + 3].exp() * adv[i + 3];
        i += 4;
    }
    while i < n {
        expectation += post_lsm[i].exp() * adv[i];
        i += 1;
    }

    adv[candidate] - expectation
}

// ── Phase 3: Product-Policy (Eq. 16) ────────────────────────────

/// Product-policy interpolation in log space (Eq. 16).
///
/// Writes `(1−w)·log π̂(a) + w·log π+(a)` into `out`. To obtain the
/// normalized product policy `π_w(a) ∝ π̂(a)^{1−w} · π+(a)^w`, exponentiate
/// and renormalize (or pass `out` through a softmax).
///
/// # Trust weight `w`
///
/// * `w = 0.0` — skip reasoning entirely (return `π̂`).
/// * `w = 0.5` — geometric mean of pre/post.
/// * `w = 1.0` — full reasoning (return `π+`).
/// * `w > 1.0` — extrapolation: trust reasoning *beyond* the model's own
///   update (sharpening).
/// * `w < 0.0` — invert the reasoning step (experimental).
///
/// # Zero allocation
///
/// Only stack-local temporaries plus the output buffer. No scratch needed
/// beyond `out`.
pub fn product_policy_log(pre_logits: &[f32], post_logits: &[f32], w: f32, out: &mut [f32]) {
    let n = pre_logits.len();
    assert_eq!(post_logits.len(), n);
    assert_eq!(out.len(), n);
    assert!(n > 0, "product_policy_log: empty logits");

    let one_minus_w = 1.0 - w;

    // Compute log partition functions for both distributions.
    let pre_max = pre_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let post_max = post_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let pre_lse: f32 = pre_logits.iter().map(|&v| (v - pre_max).exp()).sum::<f32>().ln();
    let post_lse: f32 = post_logits
        .iter()
        .map(|&v| (v - post_max).exp())
        .sum::<f32>()
        .ln();

    let pre_log_z = pre_max + pre_lse; // log Σ exp(pre)
    let post_log_z = post_max + post_lse;

    // out[a] = (1-w) * (pre_logits[a] - pre_log_z) + w * (post_logits[a] - post_log_z)
    //        = (1-w) * log π̂(a) + w * log π+(a)
    let chunks = n / 4;
    let mut i = 0;
    while i < chunks * 4 {
        out[i] = one_minus_w * (pre_logits[i] - pre_log_z) + w * (post_logits[i] - post_log_z);
        out[i + 1] =
            one_minus_w * (pre_logits[i + 1] - pre_log_z) + w * (post_logits[i + 1] - post_log_z);
        out[i + 2] =
            one_minus_w * (pre_logits[i + 2] - pre_log_z) + w * (post_logits[i + 2] - post_log_z);
        out[i + 3] =
            one_minus_w * (pre_logits[i + 3] - pre_log_z) + w * (post_logits[i + 3] - post_log_z);
        i += 4;
    }
    while i < n {
        out[i] = one_minus_w * (pre_logits[i] - pre_log_z) + w * (post_logits[i] - post_log_z);
        i += 1;
    }
}

// ── Phase 2: AdvantageMarginGate (Eq. 18 wrapper) ──────────────
// Feature-gated: requires `self_advantage_gate` Cargo feature.
// The gate is a standalone struct — it does NOT implement ScreeningPruner
// because ScreeningPruner::relevance() has no logits access. Instead, the
// recursion loop calls `should_recurse(pre_logits, post_logits, candidate)`
// after each step and breaks early when dead compute is detected.

/// Dead-compute gate for recursion loops (Eq. 18).
///
/// Distilled from [arxiv:2511.16886](https://arxiv.org/abs/2511.16886).
/// See `.research/250_*.md`, Plan 283.
///
/// After each recursion step, the caller invokes [`should_recurse`](Self::should_recurse)
/// with the pre-recursion logits (`π̂`), post-recursion logits (`π+`), and
/// the candidate action index. The gate computes the advantage margin
/// `A(candidate) − E_{a∼π+}[A(a)]` and returns `true` if the step improved
/// the candidate's prediction above the threshold, or `false` if the step
/// is dead compute.
///
/// # Integration pattern
///
/// ```text
/// let mut gate = AdvantageMarginGate::default();
/// for step in 0..max_steps {
///     let pre_logits = capture_logits(&model);
///     model.recurse();
///     let post_logits = capture_logits(&model);
///     if !gate.should_recurse(&pre_logits, &post_logits, candidate) {
///         break; // dead compute detected — skip remaining steps
///     }
/// }
/// ```
///
/// # Zero allocation in steady state
///
/// The internal scratch buffer is lazily sized on the first call and reused
/// across all subsequent calls. After the first call, `should_recurse()`
/// performs zero heap allocations.
#[cfg(feature = "self_advantage_gate")]
#[derive(Debug, Clone)]
pub struct AdvantageMarginGate {
    /// Margin threshold for accepting a recursion step.
    /// Default: `0.01` — small positive margin that rejects dead-compute steps
    /// where the candidate's improvement merely ties the population average.
    /// The mathematically centered value from Eq. 18 is `0.0`, but that never
    /// fires for convergent recursion (every step trivially beats zero mean),
    /// so the practical default is `0.01` (validated by the GOAT gate bench to
    /// give a 5×+ forward-pass reduction at 100% argmax quality). Negative
    /// thresholds are more permissive (accept even slightly harmful steps);
    /// larger positive thresholds are stricter.
    pub margin_threshold: f32,
    /// Runtime toggle. Default: `true`. When `false`, `should_recurse()`
    /// always returns `true` (passthrough).
    pub enabled: bool,
    /// Scratch buffer for advantage computation (lazily sized to `3 * n`).
    scratch: Vec<f32>,
}

#[cfg(feature = "self_advantage_gate")]
impl Default for AdvantageMarginGate {
    #[inline]
    fn default() -> Self {
        Self {
            // Practical default per Plan 283 Finding #1: 0.0 never fires for
            // convergent recursion; 0.01 gives 5×+ reduction at 100% quality
            // (validated by self_advantage_gate_bench GOAT gate).
            margin_threshold: 0.01,
            enabled: true,
            scratch: Vec::new(),
        }
    }
}

#[cfg(feature = "self_advantage_gate")]
impl AdvantageMarginGate {
    /// Create a gate with a custom margin threshold.
    #[inline]
    pub fn new(margin_threshold: f32) -> Self {
        Self {
            margin_threshold,
            enabled: true,
            scratch: Vec::new(),
        }
    }

    /// Ensure the scratch buffer can hold `3 * n` elements.
    #[inline]
    fn ensure_scratch(&mut self, n: usize) {
        let needed = 3 * n;
        if self.scratch.len() < needed {
            self.scratch.resize(needed, 0.0);
        }
    }

    /// Decide whether to continue recursing after this step.
    ///
    /// Returns `true` if the advantage margin for `candidate` is
    /// `>= margin_threshold` (or if the gate is disabled). Returns `false`
    /// if dead compute is detected (margin below threshold).
    ///
    /// # Arguments
    ///
    /// * `pre_logits` — logits before the recursion step (`π̂`).
    /// * `post_logits` — logits after the recursion step (`π+`).
    /// * `candidate` — index of the candidate action to evaluate.
    #[inline]
    pub fn should_recurse(
        &mut self,
        pre_logits: &[f32],
        post_logits: &[f32],
        candidate: usize,
    ) -> bool {
        if !self.enabled {
            return true;
        }
        self.ensure_scratch(pre_logits.len());
        let margin =
            self_advantage_margin(pre_logits, post_logits, candidate, &mut self.scratch);
        margin >= self.margin_threshold
    }

    /// Compute the advantage margin without making a gate decision.
    ///
    /// Useful for logging, debugging, or adaptive threshold tuning.
    #[inline]
    pub fn margin(
        &mut self,
        pre_logits: &[f32],
        post_logits: &[f32],
        candidate: usize,
    ) -> f32 {
        self.ensure_scratch(pre_logits.len());
        self_advantage_margin(pre_logits, post_logits, candidate, &mut self.scratch)
    }
}

// ── Phase 3: ProductPolicySharpen (Eq. 16 wrapper) ─────────────
// Feature-gated: requires `product_policy_sharpen` Cargo feature.

/// Controllable product-policy sharpening wrapper (Eq. 16).
///
/// Distilled from [arxiv:2511.16886](https://arxiv.org/abs/2511.16886).
/// See `.research/250_*.md`, Plan 283.
///
/// After each recursion step, the caller invokes [`sharpen`](Self::sharpen)
/// with pre-recursion logits (`π̂`), post-recursion logits (`π+`), and an
/// output buffer. The wrapper writes the interpolated log-policy
/// `(1−w)·log π̂ + w·log π+` into the output buffer.
///
/// # Trust weight `w`
///
/// * `0.0` — skip reasoning (return `π̂`).
/// * `0.5` — geometric mean.
/// * `1.0` — full reasoning (return `π+`).
/// * `>1.0` — extrapolation: sharpen beyond the model's own update.
#[cfg(feature = "product_policy_sharpen")]
#[derive(Debug, Clone)]
pub struct ProductPolicySharpen {
    /// Trust weight `w` for the product-policy interpolation.
    pub w: f32,
}

#[cfg(feature = "product_policy_sharpen")]
impl Default for ProductPolicySharpen {
    #[inline]
    fn default() -> Self {
        Self { w: 1.0 }
    }
}

#[cfg(feature = "product_policy_sharpen")]
impl ProductPolicySharpen {
    /// Create a sharpening wrapper with trust weight `w`.
    #[inline]
    pub fn new(w: f32) -> Self {
        Self { w }
    }

    /// Apply product-policy interpolation.
    ///
    /// Writes `(1−w)·log π̂(a) + w·log π+(a)` into `out`. The caller
    /// exponentiates and normalizes to obtain `π_w(a) ∝ π̂(a)^{1−w}·π+(a)^w`.
    ///
    /// Zero allocation — writes directly into the caller-provided buffer.
    #[inline]
    pub fn sharpen(&self, pre_logits: &[f32], post_logits: &[f32], out: &mut [f32]) {
        product_policy_log(pre_logits, post_logits, self.w, out);
    }

    /// Apply product-policy interpolation and normalize to probabilities.
    ///
    /// Writes `π_w(a) = softmax((1−w)·log π̂ + w·log π+)` into `out`.
    /// This is a convenience method for callers that want the final
    /// probability distribution directly.
    pub fn sharpen_normalized(&self, pre_logits: &[f32], post_logits: &[f32], out: &mut [f32]) {
        product_policy_log(pre_logits, post_logits, self.w, out);
        // Softmax in-place.
        let max_val = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = out.iter().map(|&v| (v - max_val).exp()).sum();
        if sum > 0.0 {
            let log_sum = sum.ln();
            for v in out.iter_mut() {
                *v = (*v - max_val - log_sum).exp();
            }
        }
    }
}

// ── Phase 5 T5.3: AdvantageDirectionSnapshot (freeze/thaw) ──────
// Feature-gated: requires `advantage_freeze_thaw` Cargo feature.
//
// Per-NPC personality fingerprint: the running EMA of the self-advantage
// direction A(·) = log π+ - log π̂ over many ticks. Captures "what kinds
// of observations improve this NPC's predictions" as a versioned,
// BLAKE3-committed latent direction vector.
//
// Design decision: EMA aggregation. The advantage A(·) is per-step
// per-candidate (n values per step, where n = vocab size). EMA smooths
// transient noise into a single direction. Decay λ defaults to 0.1
// (i.e., new observations weigh 10x the cumulative history). Conservative
// default — revisit if a consumer needs a different aggregation (Top-K,
// mean direction, etc.).

/// Snapshot version format for `AdvantageDirectionSnapshot`.
/// Bump if the weights_blob layout changes.
pub const ADVANTAGE_SNAPSHOT_VERSION: u64 = 1;

/// Running EMA aggregator for the self-advantage direction.
///
/// Maintains a single `direction: Vec<f32>` of length `n` (vocab size).
/// Each call to `update(advantage_slice)` blends the new observation into
/// the running direction via `direction ← (1-λ)·direction + λ·advantage`.
///
/// First call initializes `direction` to a copy of `advantage` (no history
/// to decay from). Subsequent calls apply the EMA.
///
/// # Decay λ
///
/// * `λ = 0.1` (default) — new observations weigh 10× cumulative history.
///   Smooths transient noise, captures slow drift.
/// * `λ = 1.0` — no memory, direction = last observation only.
/// * `λ → 0` — perfect memory, direction = average over all history.
///
/// # Zero allocation in steady state
///
/// The `direction` buffer is sized once on the first `update()` and reused.
/// `update()` performs zero heap allocations after the first call.
#[cfg(feature = "advantage_freeze_thaw")]
#[derive(Debug, Clone)]
pub struct AdvantageDirectionAccumulator {
    /// Running EMA of the advantage direction. Length = vocab size.
    pub direction: Vec<f32>,
    /// EMA decay factor (0, 1]. New observations weigh `λ`, history weighs `1-λ`.
    pub lambda: f32,
    /// Number of updates observed. Used for snapshot metadata.
    pub updates: u64,
}

#[cfg(feature = "advantage_freeze_thaw")]
impl Default for AdvantageDirectionAccumulator {
    #[inline]
    fn default() -> Self {
        Self::new(0.1)
    }
}

#[cfg(feature = "advantage_freeze_thaw")]
impl AdvantageDirectionAccumulator {
    /// Create an accumulator with EMA decay `lambda`.
    ///
    /// `lambda` must be in `(0.0, 1.0]`. Values outside this range panic in
    /// debug mode.
    #[inline]
    pub fn new(lambda: f32) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&lambda),
            "lambda must be in (0, 1], got {}",
            lambda
        );
        Self {
            direction: Vec::new(),
            lambda,
            updates: 0,
        }
    }

    /// Blend a new advantage observation into the running direction.
    ///
    /// First call initializes `direction` to a copy of `advantage`.
    /// Subsequent calls apply `direction ← (1-λ)·direction + λ·advantage`.
    ///
    /// Panics if `advantage.len()` differs from the direction length set
    /// by the first call.
    pub fn update(&mut self, advantage: &[f32]) {
        let n = advantage.len();
        assert!(n > 0, "AdvantageDirectionAccumulator::update: empty slice");
        if self.direction.is_empty() {
            // First observation: initialize.
            self.direction.clear();
            self.direction.extend_from_slice(advantage);
        } else {
            assert_eq!(
                self.direction.len(),
                n,
                "advantage length {} differs from direction length {}",
                n,
                self.direction.len()
            );
            let one_minus_lambda = 1.0 - self.lambda;
            // Chunked loop for auto-vectorization (4-wide).
            let chunks = n / 4;
            let mut i = 0;
            while i < chunks * 4 {
                self.direction[i] = one_minus_lambda * self.direction[i] + self.lambda * advantage[i];
                self.direction[i + 1] =
                    one_minus_lambda * self.direction[i + 1] + self.lambda * advantage[i + 1];
                self.direction[i + 2] =
                    one_minus_lambda * self.direction[i + 2] + self.lambda * advantage[i + 2];
                self.direction[i + 3] =
                    one_minus_lambda * self.direction[i + 3] + self.lambda * advantage[i + 3];
                i += 4;
            }
            while i < n {
                self.direction[i] = one_minus_lambda * self.direction[i] + self.lambda * advantage[i];
                i += 1;
            }
        }
        self.updates += 1;
    }

    /// Current EMA direction. Empty if `update()` has never been called.
    pub fn direction(&self) -> &[f32] {
        &self.direction
    }

    /// Snapshot the current direction as a BLAKE3-committed freeze/thaw artifact.
    ///
    /// The snapshot owns a copy of the direction bytes (LE f32 per element).
    /// Use `commit()` after construction (or after any modification) to
    /// compute the BLAKE3 hash.
    pub fn snapshot(&self, version: u64) -> AdvantageDirectionSnapshot {
        let mut weights_blob = Vec::with_capacity(self.direction.len() * 4);
        for &v in &self.direction {
            weights_blob.extend_from_slice(&v.to_le_bytes());
        }
        AdvantageDirectionSnapshot::from_parts(
            self.direction.len(),
            self.lambda,
            self.updates,
            weights_blob,
            [0u8; 32],
            version,
        )
    }
}

/// A versioned, BLAKE3-committed snapshot of an advantage direction vector.
///
/// The freeze/thaw artifact for an NPC's "what improves me" fingerprint.
/// Two NPCs of the same type can diverge by holding different snapshot
/// versions — one may have learned that combat observations improve its
/// predictions, another that exploration observations do.
///
/// # Sync boundary
///
/// - The **direction blob** is latent, local, never synced.
/// - The **BLAKE3 commitment + version** IS synced as an audit event when
///   a personality update occurs — clients can verify that the NPC they
///   observe matches a committed personality, without learning the direction.
///
/// # Commitment scheme
///
/// BLAKE3 over `dim_le_bytes || lambda_le_bytes || updates_le_bytes || weights_blob`:
///
/// ```text
/// hasher.update(&(self.dim as u64).to_le_bytes());
/// hasher.update(&self.lambda.to_le_bytes());
/// hasher.update(&self.updates.to_le_bytes());
/// hasher.update(&self.weights_blob);
/// ```
///
/// Matches the streaming-`Hasher` pattern used by `MicroRecurrentKernelSnapshot`
/// (`katgpt-rs/crates/katgpt-core/src/micro_belief/snapshot.rs`).
#[cfg(feature = "advantage_freeze_thaw")]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AdvantageDirectionSnapshot {
    /// Vocab size the direction was accumulated against.
    pub dim: usize,
    /// EMA decay λ used during accumulation. Part of the commitment so a
    /// snapshot cannot be silently replayed with a different λ.
    pub lambda: f32,
    /// Number of `update()` calls observed before the snapshot. Provenance
    /// metadata (helps detect under-trained personalities).
    pub updates: u64,
    /// Serialised direction: `dim` LE f32 values, 4 bytes each.
    pub weights_blob: Vec<u8>,
    /// BLAKE3 commitment over `(dim, lambda, updates, weights_blob)`.
    /// Filled by `commit()`; zeroed during hashing.
    pub blake3: [u8; 32],
    /// Monotonic version counter (caller-managed). NOT part of the BLAKE3
    /// input — two snapshots with identical direction but different versions
    /// are the *same* personality at different points in time.
    pub version: u64,
}

#[cfg(feature = "advantage_freeze_thaw")]
impl AdvantageDirectionSnapshot {
    /// Build a snapshot from raw parts WITHOUT computing the commitment.
    ///
    /// Useful for deserialisation paths where the commitment is already known
    /// (e.g. loading from disk). Call `commit()` afterwards if you need to
    /// recompute, or `verify()` to check integrity.
    pub fn from_parts(
        dim: usize,
        lambda: f32,
        updates: u64,
        weights_blob: Vec<u8>,
        blake3: [u8; 32],
        version: u64,
    ) -> Self {
        Self { dim, lambda, updates, weights_blob, blake3, version }
    }

    /// Compute (or recompute) the BLAKE3 commitment over
    /// `(dim, lambda, updates, weights_blob)`.
    ///
    /// Idempotent: calling twice produces the same hash. The existing
    /// `self.blake3` is zeroed internally before hashing so the commitment
    /// never feeds back into itself.
    pub fn commit(&mut self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(self.dim as u64).to_le_bytes());
        hasher.update(&self.lambda.to_le_bytes());
        hasher.update(&self.updates.to_le_bytes());
        hasher.update(&self.weights_blob);
        let hash = *hasher.finalize().as_bytes();
        self.blake3 = hash;
        hash
    }

    /// Recompute the commitment and compare with the stored `self.blake3`.
    ///
    /// Returns `true` iff the stored direction produces the stored hash.
    /// A `false` result indicates tampering or corruption.
    pub fn verify(&self) -> bool {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(self.dim as u64).to_le_bytes());
        hasher.update(&self.lambda.to_le_bytes());
        hasher.update(&self.updates.to_le_bytes());
        hasher.update(&self.weights_blob);
        let recomputed = *hasher.finalize().as_bytes();
        recomputed == self.blake3
    }

    /// Decode the `weights_blob` back into an `Vec<f32>` direction vector.
    ///
    /// Returns `Err` if the blob length is not `dim * 4`.
    pub fn decode_direction(&self) -> Result<Vec<f32>, &'static str> {
        if self.weights_blob.len() != self.dim * 4 {
            return Err("weights_blob length mismatch");
        }
        let mut out = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let start = i * 4;
            let bytes: [u8; 4] = self.weights_blob[start..start + 4].try_into().map_err(|_| "slice conversion")?;
            out.push(f32::from_le_bytes(bytes));
        }
        Ok(out)
    }

    /// Restore an accumulator from this snapshot.
    ///
    /// The restored accumulator has the same `lambda`, `direction`, and
    /// `updates` count. Future `update()` calls will continue the EMA from
    /// the snapshotted state.
    pub fn restore_accumulator(&self) -> Result<AdvantageDirectionAccumulator, &'static str> {
        let direction = self.decode_direction()?;
        Ok(AdvantageDirectionAccumulator {
            direction,
            lambda: self.lambda,
            updates: self.updates,
        })
    }
}

#[cfg(all(test, feature = "advantage_freeze_thaw"))]
mod advantage_freeze_thaw_tests {
    use super::*;

    #[test]
    fn accumulator_first_update_initializes() {
        let mut acc = AdvantageDirectionAccumulator::default();
        acc.update(&[1.0, -0.5, 0.25, -0.125]);
        assert_eq!(acc.direction(), &[1.0, -0.5, 0.25, -0.125]);
        assert_eq!(acc.updates, 1);
    }

    #[test]
    fn accumulator_ema_converges_to_steady_input() {
        // With lambda=0.1 and a constant input, the EMA should converge
        // toward that constant.
        let mut acc = AdvantageDirectionAccumulator::new(0.1);
        let target = [3.0, -1.0, 2.0, 0.5];
        for _ in 0..200 {
            acc.update(&target);
        }
        // After 200 updates with λ=0.1, EMA is essentially at the target.
        for (i, &t) in target.iter().enumerate() {
            let diff = (acc.direction()[i] - t).abs();
            assert!(diff < 0.01, "idx {}: EMA {} not converged to {}", i, acc.direction()[i], t);
        }
    }

    #[test]
    fn accumulator_ema_smooths_noise() {
        // Alternating +1/-1 input should smooth toward the mean (0).
        // With λ=0.5 the EMA steady-state oscillation amplitude is λ/(2-λ)=1/3,
        // which does NOT demonstrate smoothing. Use the default λ=0.1: amplitude
        // is 0.1/1.9 ≈ ±0.053, well within the < 0.2 threshold.
        let mut acc = AdvantageDirectionAccumulator::new(0.1);
        for step in 0..100 {
            let v = if step % 2 == 0 { 1.0 } else { -1.0 };
            acc.update(&[v]);
        }
        let d = acc.direction()[0];
        assert!(d.abs() < 0.2, "alternating input should smooth toward 0, got {}", d);
    }

    #[test]
    fn snapshot_roundtrips_commit_verify() {
        let mut acc = AdvantageDirectionAccumulator::default();
        acc.update(&[0.5, -0.25, 0.125]);
        acc.update(&[0.6, -0.30, 0.150]);

        let mut snap = acc.snapshot(1);
        snap.commit();
        assert!(snap.verify(), "freshly-committed snapshot must verify");
        assert_eq!(snap.dim, 3);
        assert_eq!(snap.updates, 2);
        assert_ne!(snap.blake3, [0u8; 32], "blake3 must be non-zero after commit");
    }

    #[test]
    fn commit_is_idempotent() {
        let mut acc = AdvantageDirectionAccumulator::default();
        acc.update(&[0.5, -0.25]);
        let mut snap = acc.snapshot(1);
        let h1 = snap.commit();
        let h2 = snap.commit();
        assert_eq!(h1, h2, "commit must be idempotent");
    }

    #[test]
    fn tampered_blob_fails_verify() {
        let mut acc = AdvantageDirectionAccumulator::default();
        acc.update(&[1.0, 2.0, 3.0]);
        let mut snap = acc.snapshot(1);
        snap.commit();
        // Tamper: flip a byte.
        snap.weights_blob[0] ^= 0xFF;
        assert!(!snap.verify(), "tampered snapshot must fail verify");
    }

    #[test]
    fn decode_direction_roundtrips() {
        let mut acc = AdvantageDirectionAccumulator::default();
        let original = [0.5, -0.25, 0.125, -0.0625, 1.0];
        acc.update(&original);
        let mut snap = acc.snapshot(1);
        snap.commit();
        let decoded = snap.decode_direction().expect("decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn restore_accumulator_continues_ema() {
        // Snapshot, restore, continue updating — the restored accumulator
        // should pick up where it left off.
        let mut acc = AdvantageDirectionAccumulator::default();
        acc.update(&[1.0, 0.0]);
        acc.update(&[1.0, 0.0]);
        let mut snap = acc.snapshot(1);
        snap.commit();

        let mut restored = snap.restore_accumulator().expect("restore");
        assert_eq!(restored.updates, 2);
        assert_eq!(restored.direction(), acc.direction());

        // Continue updating.
        restored.update(&[1.0, 0.0]);
        assert_eq!(restored.updates, 3);
    }

    #[test]
    fn lambda_is_part_of_commitment() {
        // Two snapshots with identical direction but different λ must have
        // different BLAKE3 hashes.
        let mut acc1 = AdvantageDirectionAccumulator::new(0.1);
        let mut acc2 = AdvantageDirectionAccumulator::new(0.5);
        acc1.update(&[1.0, 2.0]);
        acc2.update(&[1.0, 2.0]);  // same initial direction, different λ

        let mut snap1 = acc1.snapshot(1);
        let mut snap2 = acc2.snapshot(1);
        snap1.commit();
        snap2.commit();
        assert_ne!(snap1.blake3, snap2.blake3, "different λ must yield different commitments");
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for f32 comparisons in cross-validation.
    const EPS: f32 = 1e-5;

    fn make_scratch(n: usize) -> Vec<f32> {
        vec![0.0; 3 * n]
    }

    // ── T1.3: self_advantage correctness ────────────────────────

    #[test]
    fn test_identical_pre_post_zero_advantage() {
        // Identical pre/post → all advantages zero (dead compute correctly detected).
        let pre = [1.0, 2.0, 3.0, 0.5];
        let post = [1.0, 2.0, 3.0, 0.5];
        let mut scratch = make_scratch(pre.len());
        let adv = self_advantage(&pre, &post, &mut scratch);
        for a in adv {
            assert!(a.abs() < EPS, "expected zero advantage, got {}", a);
        }
    }

    #[test]
    fn test_post_sharpens_candidate_positive_advantage() {
        // Post sharpens toward index 1: its logit increases.
        let pre = [1.0, 1.0, 1.0];
        let post = [1.0, 5.0, 1.0]; // index 1 boosted
        let mut scratch = make_scratch(pre.len());
        let adv = self_advantage(&pre, &post, &mut scratch);
        assert!(
            adv[1] > EPS,
            "candidate 1 should have positive advantage, got {}",
            adv[1]
        );
        // Others should be negative (mass moved away from them).
        assert!(adv[0] < -EPS, "candidate 0 should be negative, got {}", adv[0]);
        assert!(adv[2] < -EPS, "candidate 2 should be negative, got {}", adv[2]);
    }

    #[test]
    fn test_post_shifts_away_candidate_negative_advantage() {
        // Post shifts away from index 0: its logit decreases.
        let pre = [5.0, 1.0, 1.0];
        let post = [1.0, 5.0, 1.0]; // index 0 suppressed
        let mut scratch = make_scratch(pre.len());
        let adv = self_advantage(&pre, &post, &mut scratch);
        assert!(
            adv[0] < -EPS,
            "candidate 0 should have negative advantage (harmful step), got {}",
            adv[0]
        );
    }

    #[test]
    fn test_extreme_logits_no_overflow() {
        // Extreme logits: numerical stability check.
        let pre = [1e4_f32, -1e4, 0.0, 5e3];
        let post = [-1e4_f32, 1e4, 0.0, 5e3];
        let mut scratch = make_scratch(pre.len());
        let adv = self_advantage(&pre, &post, &mut scratch);
        for a in adv {
            assert!(a.is_finite(), "advantage must be finite, got {}", a);
            assert!(
                a.abs() < 1e5,
                "advantage magnitude should be bounded, got {}",
                a
            );
        }
    }

    #[test]
    fn test_single_element() {
        // Degenerate: single action. Advantage should be 0 (log-ratio of
        // two identical distributions over a singleton is always 0).
        let pre = [42.0_f32];
        let post = [99.0_f32];
        let mut scratch = make_scratch(1);
        let adv = self_advantage(&pre, &post, &mut scratch);
        assert!(adv[0].abs() < EPS, "singleton advantage must be 0, got {}", adv[0]);
    }

    // ── T1.3: self_advantage_margin correctness ─────────────────

    #[test]
    fn test_margin_zero_when_identical() {
        // Identical pre/post → zero margin (dead compute for any candidate).
        let pre = [1.0, 2.0, 3.0, 0.5];
        let post = [1.0, 2.0, 3.0, 0.5];
        let mut scratch = make_scratch(pre.len());
        for c in 0..pre.len() {
            let m = self_advantage_margin(&pre, &post, c, &mut scratch);
            assert!(m.abs() < EPS, "margin for {} should be 0, got {}", c, m);
        }
    }

    #[test]
    fn test_margin_positive_for_boosted_candidate() {
        // Post sharpens toward index 2 → margin should be positive for index 2.
        let pre = [1.0, 1.0, 1.0, 1.0];
        let post = [1.0, 1.0, 5.0, 1.0];
        let mut scratch = make_scratch(pre.len());
        let m = self_advantage_margin(&pre, &post, 2, &mut scratch);
        assert!(
            m > EPS,
            "margin for boosted candidate should be positive, got {}",
            m
        );
    }

    #[test]
    fn test_margin_negative_for_suppressed_candidate() {
        // Post shifts away from index 0 → margin should be negative for index 0.
        let pre = [5.0, 1.0, 1.0, 1.0];
        let post = [1.0, 1.0, 1.0, 5.0];
        let mut scratch = make_scratch(pre.len());
        let m = self_advantage_margin(&pre, &post, 0, &mut scratch);
        assert!(
            m < -EPS,
            "margin for suppressed candidate should be negative, got {}",
            m
        );
    }

    #[test]
    fn test_margin_sum_over_candidates_is_zero() {
        // Σ_a π+(a) * margin(a) = 0 by construction (margin is mean-centered
        // under π+). We verify via a weighted sum.
        let pre = [0.5, 1.5, 2.0, 0.8, 1.2];
        let post = [1.8, 0.3, 2.5, 0.1, 1.0];
        let n = pre.len();
        let mut scratch = make_scratch(n);

        // Populate scratch to get post_lsm for the π+ weights.
        let _ = self_advantage(&pre, &post, &mut scratch);
        let post_lsm = &scratch[n..2 * n].to_vec();

        let mut weighted_sum = 0.0f32;
        for (c, &post_lsm_c) in post_lsm.iter().enumerate().take(n) {
            let m = self_advantage_margin(&pre, &post, c, &mut scratch);
            weighted_sum += post_lsm_c.exp() * m;
        }
        assert!(
            weighted_sum.abs() < 1e-4,
            "Σ π+(a)·margin(a) should be ≈ 0, got {}",
            weighted_sum
        );
    }

    // ── T1.4: Cross-validation against SDPG centered_log_ratio ──
    // Requires the `sdpg_bandit` feature for access to the shipped reference
    // implementation of `centered_log_ratio` (Plan 180, Research 160).

    #[cfg(feature = "sdpg_bandit")]
    #[test]
    fn test_self_advantage_plus_clr_is_constant() {
        use crate::pruners::centered_log_ratio;
        // Property: A[a] + centered_log_ratio[a] = KL(π+ ‖ π̂) for all a.
        // This follows from:
        //   A[a] = log(π+(a)/π̂(a))
        //   clr[a] = KL(π+‖π̂) - log(π+(a)/π̂(a))
        //   ⟹ A[a] + clr[a] = KL(π+‖π̂)  [constant across a]
        let pre = [0.3, 1.7, 2.1, 0.9, 1.5]; // student = pre
        let post = [1.9, 0.4, 2.8, 0.2, 1.1]; // teacher = post
        let n = pre.len();
        let mut scratch = make_scratch(n);
        let adv = self_advantage(&pre, &post, &mut scratch).to_vec();

        let clr = centered_log_ratio(&pre, &post, 1.0);

        let sums: Vec<f32> = adv.iter().zip(clr.iter()).map(|(&a, &c)| a + c).collect();
        let first = sums[0];
        for (i, &s) in sums.iter().enumerate() {
            assert!(
                (s - first).abs() < 1e-4,
                "A[{}] + clr[{}] = {} diverges from {} (KL should be constant)",
                i,
                i,
                s,
                first
            );
        }
    }

    #[cfg(feature = "sdpg_bandit")]
    #[test]
    fn test_margin_is_negation_of_clr() {
        use crate::pruners::centered_log_ratio;
        // Property: margin(candidate) = -centered_log_ratio(candidate)
        // (with student=pre, teacher=post, temperature=1.0).
        let pre = [0.3, 1.7, 2.1, 0.9, 1.5];
        let post = [1.9, 0.4, 2.8, 0.2, 1.1];
        let n = pre.len();
        let clr = centered_log_ratio(&pre, &post, 1.0);

        for c in 0..n {
            let mut scratch = make_scratch(n);
            let m = self_advantage_margin(&pre, &post, c, &mut scratch);
            assert!(
                (m + clr[c]).abs() < 1e-4,
                "margin({}) + clr({}) = {} should be ≈ 0",
                c,
                c,
                m + clr[c]
            );
        }
    }

    #[cfg(feature = "sdpg_bandit")]
    #[test]
    fn test_clr_cross_validation_with_temperature_sweep() {
        use crate::pruners::centered_log_ratio;
        // The identity margin = -clr holds at τ=1.0. At other temperatures
        // the distributions differ, so we only check τ=1.0 here.
        // This test documents the scope of the cross-validation.
        let pre = [1.0, 2.0, 3.0];
        let post = [3.0, 1.0, 2.0];
        let clr = centered_log_ratio(&pre, &post, 1.0);
        let mut scratch = make_scratch(pre.len());
        for c in 0..pre.len() {
            let m = self_advantage_margin(&pre, &post, c, &mut scratch);
            assert!((m + clr[c]).abs() < 1e-4, "τ=1.0 identity broken at {}", c);
        }
    }

    // ── T3.1: product_policy_log correctness ────────────────────

    #[test]
    fn test_product_policy_w_zero_returns_pre_logsoftmax() {
        // w=0 → out = log π̂ (pre log-softmax).
        let pre = [1.0, 2.0, 3.0, 0.5];
        let post = [3.0, 1.0, 2.0, 5.0];
        let mut out = vec![0.0; pre.len()];
        product_policy_log(&pre, &post, 0.0, &mut out);

        // Compute expected: log softmax of pre.
        let mut expected = vec![0.0; pre.len()];
        log_softmax_into(&pre, &mut expected);
        for i in 0..pre.len() {
            assert!((out[i] - expected[i]).abs() < EPS, "w=0 mismatch at {}", i);
        }
    }

    #[test]
    fn test_product_policy_w_one_returns_post_logsoftmax() {
        // w=1 → out = log π+ (post log-softmax).
        let pre = [1.0, 2.0, 3.0, 0.5];
        let post = [3.0, 1.0, 2.0, 5.0];
        let mut out = vec![0.0; pre.len()];
        product_policy_log(&pre, &post, 1.0, &mut out);

        let mut expected = vec![0.0; post.len()];
        log_softmax_into(&post, &mut expected);
        for i in 0..post.len() {
            assert!((out[i] - expected[i]).abs() < EPS, "w=1 mismatch at {}", i);
        }
    }

    #[test]
    fn test_product_policy_w_half_is_geometric_mean() {
        // w=0.5 → out = 0.5 * (log π̂ + log π+) = log sqrt(π̂ · π+).
        // This is the log of the geometric mean of the two distributions.
        let pre = [1.0, 2.0, 3.0];
        let post = [3.0, 1.0, 2.0];
        let mut out = vec![0.0; pre.len()];
        product_policy_log(&pre, &post, 0.5, &mut out);

        let mut pre_lsm = vec![0.0; pre.len()];
        let mut post_lsm = vec![0.0; post.len()];
        log_softmax_into(&pre, &mut pre_lsm);
        log_softmax_into(&post, &mut post_lsm);
        for i in 0..pre.len() {
            let expected = 0.5 * (pre_lsm[i] + post_lsm[i]);
            assert!((out[i] - expected).abs() < EPS, "w=0.5 mismatch at {}", i);
        }
    }

    #[test]
    fn test_product_policy_w_two_extrapolates() {
        // w=2.0 → extrapolation: out = -log π̂ + 2·log π+.
        // The output should sharpen toward π+ beyond the post distribution.
        let pre = [1.0, 1.0, 1.0];
        let post = [3.0, 1.0, 1.0]; // sharpens toward 0
        let mut out = vec![0.0; pre.len()];
        product_policy_log(&pre, &post, 2.0, &mut out);

        // After softmax, index 0 should be more peaked than post alone.
        let mut post_sm = vec![0.0; post.len()];
        let max = post.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = post.iter().map(|&v| (v - max).exp()).sum();
        for i in 0..post.len() {
            post_sm[i] = ((post[i] - max) / 1.0).exp() / sum;
        }

        let out_max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let out_sum: f32 = out.iter().map(|&v| (v - out_max).exp()).sum();
        let out_sm: Vec<f32> = out.iter().map(|&v| (v - out_max).exp() / out_sum).collect();

        assert!(
            out_sm[0] > post_sm[0],
            "w=2 should sharpen index 0 beyond post: out={} post={}",
            out_sm[0],
            post_sm[0]
        );
    }

    #[test]
    fn test_product_policy_extreme_logits() {
        let pre = [1e4_f32, -1e4, 0.0];
        let post = [-1e4_f32, 1e4, 0.0];
        let mut out = vec![0.0; pre.len()];
        product_policy_log(&pre, &post, 1.5, &mut out);
        for &v in &out {
            assert!(v.is_finite(), "product_policy output must be finite");
        }
    }

    // ── Numerical stability ─────────────────────────────────────

    #[test]
    fn test_log_softmax_sums_to_one_after_exp() {
        let logits = [1.0, 2.0, 3.0, -1.0, 0.5];
        let mut lsm = vec![0.0; logits.len()];
        log_softmax_into(&logits, &mut lsm);
        let sum: f32 = lsm.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < EPS, "exp(log_softmax) must sum to 1, got {}", sum);
    }

    // ── Phase 2: AdvantageMarginGate tests ─────────────────────

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn test_gate_default_rejects_zero_margin_step() {
        // Default threshold 0.01 (Plan 283 Finding #1): identical pre/post
        // → margin 0 → 0 >= 0.01 is false → step rejected as dead compute.
        // This is the entire point of the gate: a recursion step that didn't
        // move the candidate's prediction above population average should
        // not be re-run.
        let mut gate = AdvantageMarginGate::default();
        let pre = [1.0, 2.0, 3.0];
        let post = [1.0, 2.0, 3.0];
        assert!(!gate.should_recurse(&pre, &post, 0), "zero-margin step must be rejected by default");
        assert!(!gate.should_recurse(&pre, &post, 1), "zero-margin step must be rejected by default");
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn test_gate_threshold_zero_accepts_zero_margin() {
        // Explicit threshold 0.0 (the centered default from Eq. 18): identical
        // pre/post → margin 0 → 0 >= 0 → true. Kept as a sanity check that the
        // math is unchanged — only the *default* changed.
        let mut gate = AdvantageMarginGate::new(0.0);
        let pre = [1.0, 2.0, 3.0];
        let post = [1.0, 2.0, 3.0];
        assert!(gate.should_recurse(&pre, &post, 0));
        assert!(gate.should_recurse(&pre, &post, 1));
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn test_gate_blocks_dead_compute() {
        // Post shifts away from candidate 0 → negative margin → blocked.
        let mut gate = AdvantageMarginGate::default();
        let pre = [5.0, 1.0, 1.0];
        let post = [1.0, 5.0, 1.0];
        assert!(!gate.should_recurse(&pre, &post, 0), "dead compute for candidate 0");
        // Candidate 1 was boosted → should pass.
        assert!(gate.should_recurse(&pre, &post, 1), "improvement for candidate 1");
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn test_gate_disabled_always_passes() {
        let mut gate = AdvantageMarginGate { enabled: false, ..Default::default() };
        let pre = [5.0, 1.0];
        let post = [1.0, 5.0]; // shifts away from 0
        assert!(gate.should_recurse(&pre, &post, 0), "disabled gate must pass");
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn test_gate_strict_threshold_rejects_marginal_improvement() {
        // Set a high threshold — only large improvements pass.
        let mut gate = AdvantageMarginGate::new(5.0);
        let pre = [1.0, 1.0, 1.0];
        let post = [1.5, 1.0, 1.0]; // small boost to candidate 0
        assert!(!gate.should_recurse(&pre, &post, 0), "small improvement below strict threshold");
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn test_gate_reuses_scratch_across_calls() {
        // Verify that repeated calls don't panic and give consistent results.
        let mut gate = AdvantageMarginGate::default();
        let pre = [1.0, 2.0, 3.0, 4.0, 5.0];
        let post = [5.0, 4.0, 3.0, 2.0, 1.0];
        for _ in 0..100 {
            let _ = gate.should_recurse(&pre, &post, 0);
            let _ = gate.should_recurse(&pre, &post, 4);
        }
        // After 200 calls, scratch should be sized exactly once.
        assert_eq!(gate.scratch.len(), 3 * pre.len());
    }

    #[cfg(feature = "self_advantage_gate")]
    #[test]
    fn test_gate_margin_matches_standalone() {
        // The gate's margin() should match the standalone self_advantage_margin().
        let pre = [0.5, 1.5, 2.0, 0.8];
        let post = [1.8, 0.3, 2.5, 0.1];
        let mut gate = AdvantageMarginGate::default();
        let gate_margin = gate.margin(&pre, &post, 2);

        let mut scratch = make_scratch(pre.len());
        let standalone = self_advantage_margin(&pre, &post, 2, &mut scratch);
        assert!((gate_margin - standalone).abs() < EPS);
    }

    // ── Phase 3: ProductPolicySharpen tests ────────────────────

    #[cfg(feature = "product_policy_sharpen")]
    #[test]
    fn test_sharpen_w_one_matches_post() {
        let sharpener = ProductPolicySharpen::new(1.0);
        let pre = [1.0, 2.0, 3.0];
        let post = [3.0, 1.0, 2.0];
        let mut out = vec![0.0; pre.len()];
        sharpener.sharpen(&pre, &post, &mut out);

        let mut expected = vec![0.0; post.len()];
        log_softmax_into(&post, &mut expected);
        for i in 0..pre.len() {
            assert!((out[i] - expected[i]).abs() < EPS);
        }
    }

    #[cfg(feature = "product_policy_sharpen")]
    #[test]
    fn test_sharpen_normalized_sums_to_one() {
        let sharpener = ProductPolicySharpen::new(0.7);
        let pre = [1.0, 2.0, 3.0, 0.5];
        let post = [3.0, 1.0, 2.0, 5.0];
        let mut out = vec![0.0; pre.len()];
        sharpener.sharpen_normalized(&pre, &post, &mut out);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "normalized output must sum to 1, got {}", sum);
    }

    #[cfg(feature = "product_policy_sharpen")]
    #[test]
    fn test_sharpen_extrapolation_sharpens_beyond_post() {
        // w=2.0 should sharpen index 0 beyond what post alone gives.
        let sharpener = ProductPolicySharpen::new(2.0);
        let pre = [1.0, 1.0, 1.0];
        let post = [3.0, 1.0, 1.0];
        let mut out = vec![0.0; pre.len()];
        sharpener.sharpen_normalized(&pre, &post, &mut out);

        // Post alone: softmax([3,1,1]) → index 0 prob
        let max = 3.0_f32;
        let sum: f32 = [(3.0 - max).exp(), (1.0 - max).exp(), (1.0 - max).exp()].iter().sum();
        let post_prob_0 = (3.0 - max).exp() / sum;

        assert!(
            out[0] > post_prob_0,
            "w=2 sharpening should exceed post-only prob: got {} vs {}",
            out[0],
            post_prob_0
        );
    }
}
