//! Set-Refine — set-conditioned state refinement for decision planes
//! (Plan 627, reflex Issue 090 C2; downstream of the Jeva Prop-1 finding,
//! arXiv:2610.18773).
//!
//! ## The capability being priced
//!
//! An independent per-option scorer — `score_i = dot(x, k_i)` over a fixed
//! state embedding `x` — structurally CANNOT reverse an A-vs-B preference
//! when the candidate set changes (softmax log-odds cancel the shared
//! denominator; monotone per-option scores preserve order). Jeva buys
//! set-dependence with one trained candidate-conditioned attention block.
//! The modelless analog priced here:
//!
//! ```text
//! x' = x + (gamma/n) * sum_i w_i * (k_i - x),   w_i = sigmoid(dot(x, k_i)/tau)
//! ```
//!
//! (NEVER softmax — per-option independent gates, the AGENTS.md section-2
//! law.) The refined state `x'` is then scored by the SAME independent
//! scorer, making its scores set-dependent through the state alone.
//! Options are immutable — only the state moves.
//!
//! ## Substrate law (Plan 627 section-1 amendment)
//!
//! This module is a THIN COMPOSITION over [`crate::set_attention`]
//! (Plan 354) — no fresh kernel. The operator above is the single-query /
//! identity-projection special case of `set_sigmoid_attention_into`:
//! `h_i + (gamma/N) * sum_j alpha_ij * (v_j - h_i)` with
//! `alpha_ij = sigmoid((q_i . k_j)/sqrt(k)*beta)`. Mapping: state `x` =
//! row 0, options `k_i` = rows 1..=N, `w_q = w_k = identity_projection(d, d)`,
//! `w_v = None` (identity values — the state row's self-term contributes
//! `alpha_00 * (x - x) = 0`), `beta = sqrt(d)/tau` so the gate argument is
//! exactly `dot(x, k_i)/tau`. Decision embeddings run d = 256 (reflex
//! `EMBED_DIM`), past the legacy d <= 16 stack cap — hence the wide
//! caller-scratch entry point
//! [`set_sigmoid_attention_wide_into`](crate::set_attention::set_sigmoid_attention_wide_into)
//! this plan adds; at d <= 16 the two entry points are bit-identical.
//!
//! ## Invariance scope (the A2 exemption — reflex Issue 090 law)
//!
//! Set-dependent preference reversal REQUIRES option interaction, so a
//! working `set_refine` deliberately BREAKS decoy-invariance (reflex probe
//! A2) BY DESIGN and may move A1/A3 wherever coverage reads refined
//! scores. What it preserves: permutation invariance (the sum over options
//! is order-symmetric — S2) and value-blind argmax (the operator reads no
//! value vector — C1). The S1-S5 suite (`tests/set_refine_suite.rs`) is
//! the gate A2 cannot provide.
//!
//! ## Determinism
//!
//! Same inputs + same option order produce bit-identical output (the
//! kernel is a deterministic fold). Permuted option order agrees to float
//! precision (~1e-6; peer-order float addition — the Plan 354 G1 law).
//! The refined state is a LOCAL latent feeding the caller's scorer;
//! nothing here crosses a sync boundary (the Issue-940 classification).
//!
//! ## Zero-alloc law
//!
//! [`refine_state_into`] allocates nothing: all buffers are caller-owned
//! via [`SetRefineScratch`] (grow-only, reused across decisions), matching
//! the kernel's G4 law. The `gamma == 0.0` short-circuit copies the state
//! verbatim — the byte-identity twin of the engine-side flag-off gate (S5).

use crate::set_attention::{
    SetAttentionConfig, identity_projection_into, set_sigmoid_attention_wide_into,
};

/// Gate weight `w_i = sigmoid(dot(x, k_i) / tau)` — the analytic arm of the
/// suite (S3/S4) and the tuning surface for consumers. Delegates to the
/// SAME `fast_sigmoid` the kernel uses (~1 ULP), so analytic assertions
/// match the kernel's alpha to rounding.
#[inline]
pub fn gate_weight(x: &[f32], k: &[f32], tau: f32) -> f32 {
    debug_assert_eq!(x.len(), k.len(), "gate_weight: x and k must share dim");
    let mut dot = 0.0f32;
    for m in 0..x.len() {
        dot += x[m] * k[m];
    }
    crate::simd::fast_sigmoid(dot / tau)
}

/// Configuration for [`refine_state_into`]. `Copy` — build once, pass by
/// value (the `SetAttentionConfig` convention).
///
/// - `tau` — gate temperature: `w_i = sigmoid(dot(x, k_i)/tau)`. Smaller
///   tau sharpens gates (only near-attended options pull); larger softens.
/// - `gamma` — residual pull scale. The kernel normalizes by `n` (option
///   count + 1), so gamma is invariant to set size: "how much the attended
///   option cloud moves the state, on average". `0.0` = OFF (byte-identity
///   short-circuit, S5).
/// - `d` — embedding dimensionality (no cap; the wide kernel path).
#[derive(Clone, Copy, Debug)]
pub struct SetRefineConfig {
    pub tau: f32,
    pub gamma: f32,
    pub d: usize,
}

impl SetRefineConfig {
    /// Construct with explicit tau/gamma/d.
    #[inline]
    pub const fn new(tau: f32, gamma: f32, d: usize) -> Self {
        Self { tau, gamma, d }
    }

    /// The kernel-side beta that realizes `w_i = sigmoid(dot/tau)`:
    /// kernel argument is `(q.k)/sqrt(d) * beta` with identity projections,
    /// so `beta = sqrt(d)/tau`.
    #[inline]
    pub fn beta(&self) -> f32 {
        (self.d as f32).sqrt() / self.tau
    }

    /// The kernel config (dense path — decision sets are small; `top_k`
    /// stays `None`).
    #[inline]
    pub fn attention_config(&self) -> SetAttentionConfig {
        SetAttentionConfig::new(self.beta(), self.gamma)
    }
}

/// Errors returned by [`refine_state_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetRefineError {
    /// `options.len()` is not a multiple of `cfg.d`.
    OptionsLenMismatch { got: usize, d: usize },
    /// `options` is empty (`n_opts == 0`) — a decision with no options is
    /// not a decision; refining against an empty set is refused, not
    /// silently identity.
    NoOptions,
    /// `state.len() != cfg.d` or `out.len() != cfg.d`.
    DimMismatch { expected: usize, got: usize },
    /// The underlying kernel rejected the shapes (should not happen with a
    /// consistent [`SetRefineScratch`] — surfaced for completeness).
    Kernel(crate::set_attention::SetAttentionError),
}

/// Caller-owned, grow-only scratch for [`refine_state_into`] — the
/// zero-alloc law. Create once (sized for the expected max option count),
/// reuse across decisions; [`SetRefineScratch::ensure`] grows it when a
/// larger set arrives.
///
/// Buffers (for `n_opts` options, `d` dims, `n = n_opts + 1` rows):
/// `rows` n*d (packed `[state; options...]`), `out_rows` n*d, `q` and
/// `k_scratch` n*d each (identity-projection products), `alpha` n,
/// `sum_av` d (the wide accumulation scratch), `w_q`/`w_k` d*d (identity
/// projections, built once at [`SetRefineScratch::new`]).
pub struct SetRefineScratch {
    d: usize,
    rows: Vec<f32>,
    out_rows: Vec<f32>,
    q: Vec<f32>,
    k_scratch: Vec<f32>,
    alpha: Vec<f32>,
    sum_av: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
}

impl SetRefineScratch {
    /// New scratch for `d` dims and `max_options` options (grows on demand).
    pub fn new(d: usize, max_options: usize) -> Self {
        let n = max_options + 1;
        let nd = n * d;
        let mut w_q = vec![0.0f32; d * d];
        identity_projection_into(&mut w_q, d, d);
        let mut w_k = vec![0.0f32; d * d];
        identity_projection_into(&mut w_k, d, d);
        Self {
            d,
            rows: vec![0.0f32; nd],
            out_rows: vec![0.0f32; nd],
            q: vec![0.0f32; nd],
            k_scratch: vec![0.0f32; nd],
            alpha: vec![0.0f32; n],
            sum_av: vec![0.0f32; d],
            w_q,
            w_k,
        }
    }

    /// Grow (never shrink) to fit `n_opts` options at the construction `d`.
    pub fn ensure(&mut self, n_opts: usize) {
        let n = n_opts + 1;
        let nd = n * self.d;
        if self.rows.len() < nd {
            self.rows.resize(nd, 0.0);
            self.out_rows.resize(nd, 0.0);
            self.q.resize(nd, 0.0);
            self.k_scratch.resize(nd, 0.0);
        }
        if self.alpha.len() < n {
            self.alpha.resize(n, 0.0);
        }
    }
}

/// Refine ONE state embedding against an option set:
/// `x' = x + (gamma/n) * sum_i w_i * (k_i - x)` with
/// `w_i = sigmoid(dot(x, k_i)/tau)`.
///
/// - `state` — `d` floats, the state embedding (read-only).
/// - `options` — `n_opts * d` row-major option embeddings (read-only —
///   S4 immutability by construction: they are copied into scratch and
///   never written back).
/// - `out` — `d` floats, the refined state.
///
/// `gamma == 0.0` short-circuits to a verbatim copy (byte-identity, S5).
/// Zero allocations with a pre-`ensure`d scratch.
///
/// # Errors
/// [`SetRefineError::NoOptions`] on an empty option set;
/// [`SetRefineError::OptionsLenMismatch`] when `options.len()` is not a
/// multiple of `d`; [`SetRefineError::DimMismatch`] on `state`/`out`
/// length violations; [`SetRefineError::Kernel`] propagated from the
/// underlying attention kernel.
pub fn refine_state_into(
    state: &[f32],
    options: &[f32],
    out: &mut [f32],
    cfg: &SetRefineConfig,
    scratch: &mut SetRefineScratch,
) -> Result<(), SetRefineError> {
    let d = cfg.d;
    if state.len() != d {
        return Err(SetRefineError::DimMismatch {
            expected: d,
            got: state.len(),
        });
    }
    if out.len() != d {
        return Err(SetRefineError::DimMismatch {
            expected: d,
            got: out.len(),
        });
    }
    if options.is_empty() {
        return Err(SetRefineError::NoOptions);
    }
    if !options.len().is_multiple_of(d) {
        return Err(SetRefineError::OptionsLenMismatch {
            got: options.len(),
            d,
        });
    }
    // The OFF posture is a verbatim copy — the S5 byte-identity twin (a
    // gamma=0 kernel pass would also be identity to float rounding, but a
    // copy is identity to the BYTE, -0.0 included).
    if cfg.gamma == 0.0 {
        out.copy_from_slice(state);
        return Ok(());
    }
    let n_opts = options.len() / d;
    let n = n_opts + 1;
    scratch.ensure(n_opts);
    // Pack [state; options...] — the option rows are read from `options`
    // (never written): immutability by construction.
    scratch.rows[..d].copy_from_slice(state);
    scratch.rows[d..d + options.len()].copy_from_slice(options);
    set_sigmoid_attention_wide_into(
        &scratch.rows[..n * d],
        &scratch.w_q,
        &scratch.w_k,
        None,
        &mut scratch.out_rows[..n * d],
        &cfg.attention_config(),
        n,
        d,
        d,
        &mut scratch.q[..n * d],
        &mut scratch.k_scratch[..n * d],
        &mut scratch.alpha,
        &mut scratch.sum_av,
    )
    .map_err(SetRefineError::Kernel)?;
    out.copy_from_slice(&scratch.out_rows[..d]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn gamma_zero_is_verbatim_copy() {
        let d = 8;
        let state: Vec<f32> = (0..d).map(|i| i as f32 * 0.5 - 1.0).collect();
        let options: Vec<f32> = (0..d * 3).map(|i| (i % 7) as f32 * 0.25).collect();
        let mut out = vec![0.0f32; d];
        let cfg = SetRefineConfig::new(1.0, 0.0, d);
        let mut sc = SetRefineScratch::new(d, 3);
        refine_state_into(&state, &options, &mut out, &cfg, &mut sc).unwrap();
        assert_eq!(
            bits(&out),
            bits(&state),
            "S5: byte-identity at gamma=0"
        );
    }

    #[test]
    fn empty_options_refused() {
        let d = 4;
        let mut out = vec![0.0f32; d];
        let cfg = SetRefineConfig::new(1.0, 0.5, d);
        let mut sc = SetRefineScratch::new(d, 2);
        let r = refine_state_into(&vec![0.1; d], &[], &mut out, &cfg, &mut sc);
        assert_eq!(r, Err(SetRefineError::NoOptions));
    }

    #[test]
    fn non_multiple_options_refused() {
        let d = 4;
        let mut out = vec![0.0f32; d];
        let cfg = SetRefineConfig::new(1.0, 0.5, d);
        let mut sc = SetRefineScratch::new(d, 2);
        let r = refine_state_into(&vec![0.1; d], &vec![0.2; d + 1], &mut out, &cfg, &mut sc);
        assert_eq!(
            r,
            Err(SetRefineError::OptionsLenMismatch { got: d + 1, d })
        );
    }
}
