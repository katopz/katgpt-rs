//! SipIt-style Transformer Inversion — modelless, O(T·|V|) exact prompt recovery.
//!
//! Given a frozen decoder-only transformer's forward function and the layer-ℓ
//! per-position hidden state matrix `H̆^(ℓ) ∈ R^{T×d}`, recover the discrete
//! input token sequence `s = ⟨s₁, …, s_T⟩` exactly via per-position vocabulary
//! search. Theorem 2.2 of Nikolaou et al. ICLR 2026 (arXiv:2510.15511) proves
//! that standard decoder-only text transformers are almost-surely injective
//! under continuous parameter initialization — different prompts produce
//! different last-token states — so the prompt is uniquely recoverable from
//! the activation up to float tolerance.
//!
//! # What this is
//!
//! A **modelless inference-time algorithm.** The gradient (when used) is on
//! a continuous proxy embedding `e^(j)` at position `t`, never on model
//! parameters. No backprop through weights. Per the research workflow §3.5
//! Path 0: genuinely modelless.
//!
//! A **generic vocabulary-indexed search.** Given (a) a forward function
//! `F(v; π, t) → R^d` that returns the layer-ℓ hidden state at position `t`
//! when the prefix is `π` and the current token is `v`, (b) the observed
//! per-position states `H̆`, (c) a tolerance `ε`, and (d) a policy `Π` that
//! enumerates `V \ C` (vocabulary minus already-tried candidates) — recovers
//! `s` exactly via local verifier tests.
//!
//! # What this is NOT
//!
//! See `katgpt-rs/.plans/561_transformer_inversion_sipit_open_primitive.md`
//! §"IS NOT" for the full list of rejected fusions. Summary:
//!
//! - **NOT applicable to HLA.** HLA is a sigmoid-bounded per-NPC 8-dim
//!   belief kernel, not a decoder-only text transformer. Theorem 2.2
//!   requires real-analytic activations + a vocabulary-indexed embedding
//!   lookup; HLA has neither.
//! - **NOT a sync-boundary compression primitive.** Sync commits raw
//!   scalars (5 f32 = 20 bytes) or weight shards; transmitting a layer-ℓ
//!   hidden state matrix `T×d×4` bytes would be a ~7000× bandwidth
//!   increase, not a decrease.
//! - **NOT a lossless activation-hashing scheme.** Theorem 2.2 is
//!   measure-zero over the **parameter space**; it is NOT bit-exact over
//!   f32 representations (the paper uses `torch.allclose(rtol=1e-5,
//!   atol=1e-8)` empirically). BLAKE3 over f32 activations will collide
//!   for any two prompts whose activations fall within float precision.
//!
//! # Robustness (paper Theorem 3.2)
//!
//! Recovery holds under perturbation: if the observed state `ę_t = h̆_t + e_t`
//! has noise `‖e_t‖_∞ < Δ_π,t / 2` (where `Δ_π,t` is the margin — the
//! minimum L∞ distance from the true token's state to any other token's
//! state at position `t` under prefix `π`), then recovery still works. The
//! tolerance `ε` in [`InversionConfig`] should be set to `Δ_π,t / 2` for
//! guaranteed recovery under worst-case noise of that magnitude.
//!
//! **Quantization relationship (paper Table 2):** FP4/INT8 quantization
//! preserves injectivity in practice — the paper reports it more than doubles
//! the minimum distance between token states (the margin `Δ_π,t`), making
//! quantized models *easier* to invert, not harder. This is because
//! quantization noise acts as a regularizer that pushes apart nearly-
//! degenerate token states. The primitive's tolerance-based acceptance check
//! naturally handles quantization: set `ε` to the quantization noise floor
//! (typically ~1e-2 for INT8, ~1e-1 for FP4) and recovery proceeds as usual.
//!
//! # Allocation discipline
//!
//! Per AGENTS.md hot-loop rules:
//! - Prefix `π` reuses a single `Vec<u32>` that grows by 1 per outer iter.
//! - Visited set `C` is a `Vec<bool>` of length `|V|`, reset per position.
//! - Gradient proxy `e^(j)` (Phase 2) reuses a single `Vec<f32>` of length `d`.
//! - `InversionForward::hidden_at_into` writes into a caller-supplied
//!   `&mut [f32]` scratch — no per-trial allocation.
//!
//! # References
//!
//! - **Plan:** `katgpt-rs/.plans/561_transformer_inversion_sipit_open_primitive.md`
//! - **Research:** `katgpt-rs/.research/232_Task_Relevant_Identifiability_Specialist.md`
//!   (Gain-Redirects line); cross-refs in `.research/158` (MUX) +
//!   `.research/244` (FaithfulnessProbe).
//! - **Source paper:** arXiv:2510.15511 — Nikolaou, Mencattini, Crisostomi,
//!   Santilli, Panagakis, Rodolà, *Language Models are Injective and Hence
//!   Invertible*, ICLR 2026.
//! - **Reference impl:** <https://github.com/giorgosnikolaou/SIPIT>
//!
//! # Phase 1 GOAT gate (this module)
//!
//! - **G1** — exact recovery on a toy 2-layer decoder-only transformer
//!   (GELU activation, d=16, |V|=32, T=8), random init. Three sub-tests:
//!   `g1_exact_recovery_random_init`, `g1_recovers_when_two_prompts_differ_only_at_position_t`
//!   (paper Lemma D.2 causality), `g1_no_false_positive_on_mismatched_observed`.
//! - **G2** — latency; deferred to Phase 3 (`benches/inversion_bench.rs`).
//! - **G4** — alloc-free hot path; deferred to Phase 3 (`dhat` bench).
//!
//! Phase 2 adds the gradient-guided policy (paper Alg 3) behind the
//! `grad_policy` sub-feature.
//!
//! Transformer Inversion — SipIt open primitive (Plan 561, Research 232
//! Gain-Redirects, arXiv:2510.15511 Nikolaou et al. ICLR 2026). Modelless,
//! O(T·|V|) exact prompt recovery from a layer-ℓ hidden state matrix via
//! per-position vocabulary search. Public-engine adoption hook for
//! transparency / interpretability / audit tooling on standard decoder-only
//! text transformers; NOT applicable to HLA (sigmoid-bounded 8-dim belief
//! kernel, not a real-analytic text transformer) and NOT a sync-boundary
//! compression primitive (transmitting a T×d hidden-state matrix is a
//! ~7000× bandwidth INCREASE over the 20-byte scalar sync).
//!
//! Phase 1 (this module): skeleton + RandomPolicy (uniform-without-
//! replacement) + G1 exact-recovery tests on a toy 2-layer GELU transformer.
//! Phase 2 (gated on `grad_policy` sub-feature): GradientGuidedPolicy
//! (paper Alg 3). Phase 3 (deferred): G2 latency + G4 alloc-free benches.
//!
//! Gain-tier parking rationale: §1.55 mandates the verdict (modelless +
//! not shipped → Gain). The Gain tier routing "Plan only, behind feature
//! flag" implies code ships feature-gated; promotion to default-on
//! requires a concrete consumer that demonstrates a gain at the GOAT gate
//! (Plan 561 T5.1 — unmet as of 2026-07-29: zero production consumers across
//! the 7-repo stack; re-verified via grep for `transformer_inversion` /
//! `katgpt_core::inversion` across all 7 repos — only self-references in
//! this module + bench_561 + this lib.rs export + the
//! `examples/transformer_inversion_01_forensics.rs` reference harness).
//! The feature ships as adoption-hook infrastructure; the forensics demo
//! (2026-07-29) closes the documentation gap (every public primitive in
//! katgpt-rs ships an example harness) but is NOT a production consumer.
//! Re-evaluate at the 2026-10-26 timeout (T5.2).

pub mod policy;
pub mod recovery;
pub mod verifier;

#[cfg(test)]
mod tests;

pub use policy::{InversionPolicy, RandomPolicy};
pub use recovery::{invert_sequence, invert_sequence_into};
pub use verifier::{AcceptanceRegion, accept_observation, accept_observation_into};

#[cfg(feature = "grad_policy")]
pub use policy::GradientGuidedPolicy;
#[cfg(feature = "grad_policy")]
pub use recovery::{invert_sequence_grad, invert_sequence_grad_into};

/// Observed per-position layer-ℓ hidden states `H̆^(ℓ) ∈ R^{T×d}`, row-major.
///
/// Borrows the underlying buffer; does not own it. `states.len()` must equal
/// `t_len * d_len`. The state at position `t` is `&states[t*d_len .. (t+1)*d_len]`.
#[derive(Clone, Copy, Debug)]
pub struct ObservedStates<'a> {
    pub states: &'a [f32],
    pub t_len: usize,
    pub d_len: usize,
}

impl<'a> ObservedStates<'a> {
    /// Construct from a flat row-major buffer. Returns `Err` on shape mismatch.
    pub fn from_row_major(
        states: &'a [f32],
        t_len: usize,
        d_len: usize,
    ) -> Result<Self, InversionError> {
        let expected = t_len
            .checked_mul(d_len)
            .ok_or(InversionError::ShapeOverflow)?;
        if states.len() != expected {
            return Err(InversionError::ShapeMismatch {
                expected,
                got: states.len(),
            });
        }
        Ok(Self {
            states,
            t_len,
            d_len,
        })
    }

    /// Return the state row at position `t`. Panics if `t >= t_len` (caller bug).
    #[inline]
    pub fn row(&self, t: usize) -> &[f32] {
        &self.states[t * self.d_len..(t + 1) * self.d_len]
    }
}

/// Configuration for SipIt-style inversion.
#[derive(Clone, Debug)]
pub struct InversionConfig {
    /// Acceptance tolerance ε. Theory: ε < Δ_π,t / 2 where Δ_π,t is the
    /// minimum distance between the observed state and any other token's
    /// state at position `t` under prefix `π`. Practice: small + backoff.
    ///
    /// The acceptance check is `‖h̆_t − F(v; π, t)‖_∞ ≤ ε` (L∞ norm); see
    /// [`verifier::AcceptanceRegion`].
    pub tolerance: f32,

    /// Max vocabulary trials per position before declaring failure.
    /// Default `usize::MAX` = whole vocabulary. Tightening this trades
    /// recall for early-failure signal (useful when the gradient-guided
    /// policy of Phase 2 is expected to find the token in <0.5% of |V|).
    pub max_trials_per_position: usize,

    /// Policy for candidate enumeration. See [`InversionPolicy`].
    pub policy: InversionPolicy,
}

impl Default for InversionConfig {
    fn default() -> Self {
        Self {
            // Paper §E.1 uses rtol=1e-5, atol=1e-8 for the empirical
            // collision check on f32 activations; the L∞ tolerance here is
            // a different (more conservative) bound. 1e-3 is a reasonable
            // starting point for toy transformers in f32 — adjust per-model.
            tolerance: 1e-3,
            max_trials_per_position: usize::MAX,
            policy: InversionPolicy::Random,
        }
    }
}

/// Forward signature: given prefix `π` and a candidate token `v` at position
/// `t`, write the layer-ℓ hidden state at position `t` into `out`.
///
/// The caller supplies this. It wraps their transformer's forward pass up to
/// layer ℓ (with prefix conditioning). No autodiff required for the random
/// policy. The `out` buffer has length `d` (the model's hidden dimension);
/// implementations must not allocate.
///
/// # Errors
///
/// Implementations should return [`InversionError::ForwardFailed`] if the
/// forward pass cannot be evaluated (e.g., invalid token id, OOM).
pub trait InversionForward {
    fn hidden_at_into(
        &self,
        prefix: &[u32],
        candidate: u32,
        position: usize,
        out: &mut [f32],
    ) -> Result<(), InversionError>;
}

/// Gradient + projection hooks (only used by `GradientGuidedPolicy`,
/// Phase 2). The caller owns the transformer's embedding matrix and
/// differentiates its forward pass; the primitive consumes the loss
/// gradient + nearest-token projection without any autodiff dependency.
///
/// Phase 1 random policy never calls this; it is only reached when
/// [`InversionConfig::policy`] is [`InversionPolicy::GradientGuided`] and
/// the caller invokes [`recovery::invert_sequence_grad_into`].
#[cfg(feature = "grad_policy")]
pub trait InversionGradient {
    /// Gradient of `L(e) = ½·‖h̆_t − F(e; π, t)‖²` with respect to the proxy
    /// embedding `e`, evaluated at `proxy`. `observed_state` is the target
    /// row `h̆_t` (passed in so the caller can compute the residual).
    ///
    /// Writes the length-`d` gradient vector into `out`. Implementations
    /// may use analytical, autodiff, or finite-difference gradients — the
    /// primitive is agnostic. Returns [`InversionError::ForwardFailed`] if
    /// the underlying evaluation fails.
    fn grad_hidden_at_into(
        &self,
        prefix: &[u32],
        observed_state: &[f32],
        proxy: &[f32],
        position: usize,
        out: &mut [f32],
    ) -> Result<(), InversionError>;

    /// Project a continuous proxy embedding to the nearest vocabulary token:
    /// returns `argmin_v ‖proxy − embedding[v]‖²`. The caller owns the
    /// embedding matrix and chooses the search strategy (linear scan for
    /// small `|V|`, KD-tree / FAISS / etc. for large `|V|`).
    ///
    /// This is the bridge from the continuous gradient trajectory back to
    /// the discrete vocabulary — paper Alg 3 step "project to nearest vocab
    /// embedding". Allocation-free implementations are preferred but not
    /// required (this is called at most `max_grad_steps / projection_period`
    /// times per position, not per gradient step).
    fn nearest_token(&self, proxy: &[f32]) -> Result<u32, InversionError>;

    /// Initialize the proxy embedding for a new position. Default: zeros.
    ///
    /// Paper §E.1 recommends the mean of all vocabulary embeddings (closer
    /// to every individual embedding than zeros, so the gradient basin is
    /// more likely to contain the correct token). Callers that can compute
    /// the mean cheaply should override this; the default zeros works for
    /// symmetric embedding distributions but may converge slower.
    ///
    /// Called once per position by the driver; allocation-free (writes into
    /// the reused `proxy` buffer).
    fn init_proxy_into(&self, out: &mut [f32]) -> Result<(), InversionError> {
        for x in out.iter_mut() {
            *x = 0.0;
        }
        Ok(())
    }
}

/// Outcome of an inversion run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InversionResult {
    /// Exact recovery (within tolerance) at every position.
    Recovered(Vec<u32>),
    /// Could not verify any candidate at `failed_position` within
    /// `max_trials_per_position` (or the whole vocabulary was exhausted).
    Failed {
        failed_position: usize,
        candidates_tried: usize,
    },
}

/// Errors returned by the inversion driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InversionError {
    /// `states.len() != t_len * d_len`.
    ShapeMismatch { expected: usize, got: usize },
    /// `t_len * d_len` overflowed `usize`.
    ShapeOverflow,
    /// The caller's `InversionForward` impl returned an error.
    ForwardFailed,
    /// `scratch.len() != d_len` (the scratch buffer must hold one row).
    ScratchLenMismatch { expected: usize, got: usize },
    /// Empty vocabulary (`vocab_size == 0`) or `t_len == 0`.
    EmptyInput,
}

#[cfg(test)]
impl std::fmt::Display for InversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => write!(
                f,
                "ObservedStates shape mismatch: expected {expected} floats, got {got}"
            ),
            Self::ShapeOverflow => write!(f, "ObservedStates t_len * d_len overflowed usize"),
            Self::ForwardFailed => write!(f, "InversionForward::hidden_at_into failed"),
            Self::ScratchLenMismatch { expected, got } => write!(
                f,
                "scratch buffer must hold d_len={expected} floats, got {got}"
            ),
            Self::EmptyInput => write!(f, "empty vocabulary or empty observed sequence"),
        }
    }
}
