//! Sigmoid fusion kernel for engram patterns.
//!
//! Plan 299 Phase 3 T3.1–T3.5. Implements the context-aware sigmoid gate
//! that fuses a retrieved pattern into a hidden state.
//!
//! # CRITICAL: sigmoid, not softmax, per AGENTS.md
//!
//! This entire module is sigmoid-based. The gate is a single scalar
//! `σ(dot(q_norm, k_norm) / τ) ∈ (0, 1)`. **There is no `softmax` symbol
//! anywhere in this file.** Softmax would be both slower (D-way exp) and
//! wrong (per AGENTS.md, sigmoid is the canonical sparse-gating primitive
//! in this stack — see temporal_deriv, faithfulness/gate, off_principal).
//!
//! # Formula (Plan T3.3)
//!
//! ```text
//! q_norm = RMSNorm(q)
//! k_norm = RMSNorm(k)
//! gate   = sigmoid(dot(q_norm, k_norm) / τ)     // ∈ (0, 1)
//! for j in 0..D:
//!     out[j] = gate * v[j]
//! ```
//!
//! The output is `gate * v` — caller adds it as a residual into the hidden
//! state. `τ = √D` matches the paper (so `dot/τ` is roughly cosine-scaled).
//!
//! # Hot-path contract
//!
//! [`sigmoid_fuse_into`] is **zero-allocation**. The caller provides the
//! `out` buffer; the kernel writes exactly `D` floats to it. RMSNorm uses
//! [`simd::simd_sum_sq`] for the pass-1 reduction (NEON/AVX2-accelerated).
//!
//! # TODO (Phase 3 follow-on, deferred)
//!
//! - T3.6 multi-branch variant `sigmoid_fuse_multi_branch_into` (M distinct
//!   gates sharing one `v`). Default M=1 (single-branch); mHC users opt in
//!   to M=4. Deferred — file when first consumer needs it.
//! - T3.7 depthwise causal conv `conv_causal_into` (paper §2.3 eq 5).
//!   Deferred — file when first consumer needs it.

use crate::simd::{fast_sigmoid, simd_dot_f32, simd_sum_sq};

/// Shared value `v` across `M` branches, each with its own query/key pair.
///
/// Computes per-branch gate and scales the **same** `v` by each gate.
/// Default `M = 1` reduces to a single call to [`sigmoid_fuse_into`]. The
/// paper uses `M = 4` (mHC backbone §2.4) — the shared value is the fused
/// `(W_V · e)`; the keys are per-branch `(W_K^(m) · e)` projections.
///
/// # CRITICAL — sigmoid, not softmax, per AGENTS.md
///
/// Each branch computes its own independent scalar gate
/// `σ(dot(q_norm, k_norm) / τ)`. There is **no `softmax` symbol** in this
/// function — softmax would imply competition between branches (the
/// Engram paper specifically avoids this — branches are additive, not
/// mutually-exclusive).
///
/// # Zero-allocation
///
/// Uses the same fused-RMSNorm + dot trick as [`sigmoid_fuse_into`] — no
/// intermediate `q_norm` / `k_norm` buffers are materialized. The caller
/// provides the `out_per_branch` slices.
///
/// # Arguments
///
/// - `q_per_branch` — M query slices, each of length D.
/// - `k_per_branch` — M key slices, each of length D.
/// - `v` — single shared value slice of length D.
/// - `out_per_branch` — M output slices, each of length D. Written as
///   `out_per_branch[m][j] = gate_m * v[j]`.
/// - `config` — fusion config (tau, rmsnorm_eps).
///
/// # Panics (debug only)
///
/// `debug_assert!` checks that all slices have equal length D and that
/// `q_per_branch.len() == k_per_branch.len() == out_per_branch.len()`.
///
/// # Plan reference
///
/// Plan 299 Phase 3 T3.6. Default M=1 must reduce to a single call to
/// [`sigmoid_fuse_into`] — verified by the `m1_matches_single_branch` unit
/// test.
#[inline]
pub fn sigmoid_fuse_multi_branch_into(
    q_per_branch: &[&[f32]],
    k_per_branch: &[&[f32]],
    v: &[f32],
    out_per_branch: &mut [&mut [f32]],
    config: &SigmoidFusionConfig,
) {
    let m = q_per_branch.len();
    debug_assert_eq!(
        k_per_branch.len(),
        m,
        "sigmoid_fuse_multi_branch_into: k_per_branch.len() must equal q_per_branch.len()"
    );
    debug_assert_eq!(
        out_per_branch.len(),
        m,
        "sigmoid_fuse_multi_branch_into: out_per_branch.len() must equal q_per_branch.len()"
    );
    if m == 0 {
        return;
    }
    let d = v.len();
    debug_assert!(
        q_per_branch.iter().all(|q| q.len() == d),
        "sigmoid_fuse_multi_branch_into: all q slices must have length D = v.len()"
    );
    debug_assert!(
        k_per_branch.iter().all(|k| k.len() == d),
        "sigmoid_fuse_multi_branch_into: all k slices must have length D = v.len()"
    );
    debug_assert!(
        out_per_branch.iter().all(|o| o.len() == d),
        "sigmoid_fuse_multi_branch_into: all out slices must have length D = v.len()"
    );
    if d == 0 {
        return;
    }

    // For each branch, compute the gate and scale v by it.
    // The inner scale-loop is small-D-friendly and auto-vectorizes.
    for branch in 0..m {
        let q = q_per_branch[branch];
        let k = k_per_branch[branch];
        let out = &mut out_per_branch[branch];

        // Fused RMSNorm + dot: dot(q_norm, k_norm) = dot(q, k) * inv_rms_q * inv_rms_k.
        let sum_sq_q = simd_sum_sq(q, d);
        let sum_sq_k = simd_sum_sq(k, d);
        let inv_rms_q = 1.0 / ((sum_sq_q / d as f32) + config.rmsnorm_eps).sqrt();
        let inv_rms_k = 1.0 / ((sum_sq_k / d as f32) + config.rmsnorm_eps).sqrt();
        let raw_dot = simd_dot_f32(q, k, d);
        let normalized_dot = raw_dot * inv_rms_q * inv_rms_k;

        // CRITICAL: sigmoid, not softmax. Each branch computes its own
        // independent scalar gate — no inter-branch competition.
        let gate = fast_sigmoid(normalized_dot / config.tau);

        // Slice both sides to D once so the store loop is bounds-check-free
        // and auto-vectorizes.
        for (o, &vj) in out[..d].iter_mut().zip(v[..d].iter()) {
            *o = gate * vj;
        }
    }
}

/// Configuration for the sigmoid fusion kernel.
///
/// Defaults (per Plan T3.1):
/// - `tau = √D` — scales the dot product so `gate = σ(cosine-like)` for
///   unit-norm q,k. The default `tau = √32` matches the most common hidden
///   dim in this stack; host MUST override if D differs (e.g. pass
///   `(D as f32).sqrt()`).
/// - `rmsnorm_eps = 1e-6` — guard against zero-RMS vectors.
#[derive(Debug, Clone, Copy)]
pub struct SigmoidFusionConfig {
    /// Inverse-temperature for the sigmoid gate. `dot(q_norm, k_norm) / tau`.
    pub tau: f32,
    /// RMSNorm epsilon (numerical guard against zero-RMS vectors).
    pub rmsnorm_eps: f32,
    /// Additive bias on the gate logit: `gate = sigmoid(dot(q_norm,k_norm)/tau
    /// + logit_bias)`. Default `0.0` — bit-identical to the legacy gate
    ///   (`x + 0.0` is an exact f32 identity, including the `-0.0` edge: both
    ///   branch sides of `fast_sigmoid` map ±0.0 to the same `0.5`).
    ///
    /// Plan 364 Phase 4 (arXiv:2608.15062 GRT recipe — "bias toward identity,
    /// specialize gradually"): the xHC sparse-write gate inits CLOSED at
    /// `logit_bias = -4` (`g ≈ 0.018`) so the model boots as the no-write
    /// identity and the gated writes specialize gradually. Also the inject
    /// point for per-step gate-logit noise (εg, train-only). The hot path
    /// costs one f32 add; the default path stays bit-identical.
    ///
    /// Issue 819 T3 (Research 566, arXiv:2601.15380): this constant bias is
    /// the constant-prior special case of the sigmoid KL-prior gate
    /// `g* = σ(s/τ + logit π_j)` — per-key priors π_j live in
    /// `ParallaxConfig::prior_logits` (`prior_logit_lane`), and a per-call
    /// prior for THIS single-key kernel is expressible by assignment
    /// (`config.logit_bias = lane[j]` — the xHC forward already composes
    /// per-step bias this way). The negative-bias default is a stability
    /// requirement, not a training convenience: the sigmoid margin law
    /// bounds context-noise sensitivity by `‖Δo‖ ≲ ε·(L−1)·e^{ω−δ}`, and with
    /// a uniform prior (δ = 0) sigmoid context mass grows LINEARLY in L —
    /// the prior margin is the only structural protection a sigmoid gate
    /// stack has. Any structural margin δ ≳ ω + ln L is load-bearing.
    /// Law + derivation: `parallax_attn` module docs / Research 566 §2.
    pub logit_bias: f32,
}

impl Default for SigmoidFusionConfig {
    #[inline]
    fn default() -> Self {
        // Default tau = √32 — the common hidden dim in this stack. Host
        // overrides when D differs. Hardcoded sqrt for const-fn friendliness
        // (no runtime sqrt at default construction).
        Self {
            tau: (32.0f32).sqrt(),
            rmsnorm_eps: 1e-6,
            logit_bias: 0.0,
        }
    }
}

/// RMSNorm `x → x / √(mean(x²) + eps)`, writing the result into `out`.
///
/// Plan T3.2. In-place-safe: `out` MAY alias `x` (read-then-write per
/// element, no cross-element aliasing). Uses `simd::simd_sum_sq` for the
/// pass-1 reduction. Zero-allocation.
///
/// `out.len()` MUST equal `x.len()` (debug_asserted). Empty `x` is a no-op.
#[inline]
pub fn rmsnorm_into(x: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    debug_assert_eq!(out.len(), n, "rmsnorm_into: out.len() must equal x.len()");
    // Pass 1: sum of squares (SIMD-accelerated via crate::simd).
    let sum_sq = simd_sum_sq(x, n);
    // inv_rms = 1 / sqrt(mean(x²) + eps). Stay f32 to avoid f64 round-trip
    // (per types/math.rs rmsnorm comment).
    let inv_rms = 1.0 / ((sum_sq / n as f32) + eps).sqrt();
    // Pass 2: scale x → out. We can't use simd_scale_inplace directly since
    // out ≠ x in general, so do a manual copy-with-scale. Zipping the two
    // pre-sliced halves keeps the loop bounds-check-free.
    for (o, &xi) in out[..n].iter_mut().zip(x.iter()) {
        *o = xi * inv_rms;
    }
}

/// Context-aware sigmoid-gated fusion of `v` into `out`.
///
/// Plan T3.3. CRITICAL: uses sigmoid, not softmax, per AGENTS.md.
///
/// Computes:
/// ```text
/// q_norm = RMSNorm(q, eps)
/// k_norm = RMSNorm(k, eps)
/// gate   = sigmoid(simd_dot_f32(q_norm, k_norm) / tau)
/// out[j] = gate * v[j]   for j in 0..D
/// ```
///
/// # Zero-allocation
///
/// Caller provides `out` of size `D`. The kernel uses no scratch — the
/// `q_norm` / `k_norm` writes happen directly into prefix regions of `out`
/// ONLY when the caller has aliased intentionally; in the standard
/// (non-aliasing) case the kernel does two small fixed-size stack arrays
/// via `MaybeUninit`-free const-size arrays. Since D is unknown at compile
/// time, we use a single pass that recomputes inv_rms twice — but this is
/// cheaper than allocating two D-sized buffers.
///
/// Implementation note: we compute the dot product by folding inv_rms_q and
/// inv_rms_k into the per-element products, avoiding the need for
/// intermediate q_norm/k_norm buffers entirely. This is the "fused RMSNorm
/// + dot" trick — mathematically equivalent, allocation-free.
///
/// # Arguments
///
/// - `q`, `k`, `v` — slices of length D (the hidden-state dim). All MUST be
///   equal length (debug_asserted).
/// - `out` — output slice of length D. MUST NOT alias `v` (writes happen
///   after reads for each element, so aliasing is actually safe, but prefer
///   separate buffers).
/// - `config` — see [`SigmoidFusionConfig`].
#[inline]
pub fn sigmoid_fuse_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    config: &SigmoidFusionConfig,
) {
    let d = q.len();
    debug_assert_eq!(k.len(), d, "sigmoid_fuse_into: k.len() must equal q.len()");
    debug_assert_eq!(v.len(), d, "sigmoid_fuse_into: v.len() must equal q.len()");
    debug_assert_eq!(
        out.len(),
        d,
        "sigmoid_fuse_into: out.len() must equal q.len()"
    );
    if d == 0 {
        return;
    }

    // Fused RMSNorm + dot product, no intermediate buffers:
    //   dot(q_norm, k_norm) = dot(q * inv_rms_q, k * inv_rms_k)
    //                       = inv_rms_q * inv_rms_k * dot(q, k)
    //
    // Wait — that's only true for plain scaling. RMSNorm IS plain scaling
    // (no mean-subtraction, unlike LayerNorm), so the algebra holds:
    //   q_norm[i] = q[i] * inv_rms_q
    //   dot(q_norm, k_norm) = (Σ q[i]*k[i]) * inv_rms_q * inv_rms_k
    //
    // So we can compute dot(q, k) once via simd_dot_f32, multiply by the
    // two inv_rms scalars, and never materialize q_norm/k_norm.
    let sum_sq_q = simd_sum_sq(q, d);
    let sum_sq_k = simd_sum_sq(k, d);
    let inv_rms_q = 1.0 / ((sum_sq_q / d as f32) + config.rmsnorm_eps).sqrt();
    let inv_rms_k = 1.0 / ((sum_sq_k / d as f32) + config.rmsnorm_eps).sqrt();

    let raw_dot = simd_dot_f32(q, k, d);
    let normalized_dot = raw_dot * inv_rms_q * inv_rms_k;

    // CRITICAL: sigmoid, not softmax, per AGENTS.md.
    // `logit_bias` (Plan 364 Phase 4) shifts the gate logit additively; the
    // default `0.0` is an exact f32 identity — bit-identical to the
    // pre-field kernel at every input.
    let gate = fast_sigmoid(normalized_dot / config.tau + config.logit_bias);

    // Write gate * v[j] into out — SIMD-friendly stride-1 write. We use a
    // manual loop (not simd_scale_inplace) because src and dst are
    // different slices; slicing both to D up front removes the per-element
    // bounds checks so the small D=32 loop auto-vectorizes cleanly.
    for (o, &vj) in out[..d].iter_mut().zip(v[..d].iter()) {
        *o = gate * vj;
    }
}

/// [`sigmoid_fuse_into`] with an extra scalar on the gate — returns the
/// **unscaled** gate.
///
/// Issue 656 (counterfactual privilege gating). Computes:
/// ```text
/// gate   = sigmoid(dot(q_norm, k_norm) / tau)     // identical to sigmoid_fuse_into
/// out[j] = (gate * gate_scale) * v[j]
/// return gate
/// ```
///
/// # Why a sibling instead of a parameter on [`sigmoid_fuse_into`]
///
/// The shipped default path must stay byte-for-byte untouched (Issue 656 T2:
/// "existing gate math untouched"). `gate_scale = 1.0` here is bit-identical
/// to [`sigmoid_fuse_into`] — pinned by the
/// `scaled_fuse_with_unit_scale_is_bit_identical` test in
/// [`privilege`](super::privilege).
///
/// # Why the scale is applied to the scalar, not the vector
///
/// Folding it into the gate costs **one f32 multiply per call**. Applying it
/// as a second pass over `out` would cost `D` multiplies plus a second store.
/// That difference is the whole reason the privilege gate can meet a tight
/// hot-path overhead budget.
///
/// # Returning the gate
///
/// The caller needs the *unscaled* similarity gate to implement the
/// gate-on-gate floor (deciding whether a head is worth counterfactual
/// scoring). Recovering it from `out` would require dividing by
/// `gate_scale · ‖v‖`, which is undefined when either is zero.
///
/// # CRITICAL — sigmoid, not softmax, per AGENTS.md
///
/// Same single scalar gate as [`sigmoid_fuse_into`]. No `softmax` symbol.
///
/// # Zero-allocation
///
/// Same fused-RMSNorm + dot trick as [`sigmoid_fuse_into`] — no scratch.
/// Compiled under `engram_privilege` (the Issue-656 consumer) OR
/// `engram_tripwire` (the Issue-837 detector — the returned unscaled gate IS
/// the tripwire's per-source consumption weight).
#[cfg(any(feature = "engram", feature = "engram_tripwire"))]
#[inline]
pub fn sigmoid_fuse_scaled_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    config: &SigmoidFusionConfig,
    gate_scale: f32,
) -> f32 {
    let d = q.len();
    debug_assert_eq!(
        k.len(),
        d,
        "sigmoid_fuse_scaled_into: k.len() must equal q.len()"
    );
    debug_assert_eq!(
        v.len(),
        d,
        "sigmoid_fuse_scaled_into: v.len() must equal q.len()"
    );
    debug_assert!(
        out.len() >= d,
        "sigmoid_fuse_scaled_into: out.len() must be ≥ q.len()"
    );
    if d == 0 {
        return 0.0;
    }

    // Fused RMSNorm + dot — see `sigmoid_fuse_into` for the algebra.
    let sum_sq_q = simd_sum_sq(q, d);
    let sum_sq_k = simd_sum_sq(k, d);
    let inv_rms_q = 1.0 / ((sum_sq_q / d as f32) + config.rmsnorm_eps).sqrt();
    let inv_rms_k = 1.0 / ((sum_sq_k / d as f32) + config.rmsnorm_eps).sqrt();

    let raw_dot = simd_dot_f32(q, k, d);
    let normalized_dot = raw_dot * inv_rms_q * inv_rms_k;

    // CRITICAL: sigmoid, not softmax, per AGENTS.md.
    // Same logit_bias as [`sigmoid_fuse_into`] (default 0.0 = identity).
    let gate = fast_sigmoid(normalized_dot / config.tau + config.logit_bias);
    // The scale rides on the scalar — one multiply, not D.
    let scaled = gate * gate_scale;

    for (o, &vj) in out[..d].iter_mut().zip(v[..d].iter()) {
        *o = scaled * vj;
    }
    gate
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a config with a given D so the tau default makes sense.
    fn cfg_for_dim(d: usize) -> SigmoidFusionConfig {
        SigmoidFusionConfig {
            tau: (d as f32).sqrt(),
            rmsnorm_eps: 1e-6,
            logit_bias: 0.0,
        }
    }

    /// Plan 364 Phase 4 — the `logit_bias` field at its default `0.0` must be
    /// bit-identical to the LEGACY gate math (`fast_sigmoid(ndot/tau)`, the
    /// pre-field formula) at every input, including the `-0.0` edge
    /// (`-0.0 + 0.0 = +0.0` in IEEE; both must map to the same gate bits).
    #[test]
    fn logit_bias_zero_is_bit_identical_at_every_input() {
        let d = 32;
        let cfg = cfg_for_dim(d);
        // A corpus of adversarial inputs: zeros (exact ±0.0 normalized dot),
        // subnormals, large magnitudes, mixed signs, deterministic variety.
        let corpus: Vec<Vec<f32>> = vec![
            vec![0.0; d],
            vec![-0.0; d],
            vec![f32::MIN_POSITIVE; d],
            vec![1e-30; d],
            vec![1e6; d],
            vec![-1e6; d],
            (0..d)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            (0..d)
                .map(|i| (((i * 37 + 11) % 23) as i32 - 11) as f32 / 7.0)
                .collect(),
        ];
        for q in &corpus {
            for k in &corpus {
                let v = vec![1.5f32; d];
                let mut out = vec![0.0f32; d];
                sigmoid_fuse_into(q, k, &v, &mut out, &cfg);
                // Inline LEGACY reference — the pre-field formula, computed
                // from the same primitives the kernel uses.
                let sum_sq_q = simd_sum_sq(q, d);
                let sum_sq_k = simd_sum_sq(k, d);
                let inv_q = 1.0 / ((sum_sq_q / d as f32) + cfg.rmsnorm_eps).sqrt();
                let inv_k = 1.0 / ((sum_sq_k / d as f32) + cfg.rmsnorm_eps).sqrt();
                let ndot = simd_dot_f32(q, k, d) * inv_q * inv_k;
                let gate_ref = fast_sigmoid(ndot / cfg.tau);
                // gate · v[0] is injective in gate for v[0] = 1.5 ≠ 0.
                let got = out[0] / 1.5;
                assert_eq!(
                    got.to_bits(),
                    gate_ref.to_bits(),
                    "q[0]={} k[0]={}",
                    q[0],
                    k[0]
                );
            }
        }
    }

    /// Plan 364 Phase 4 — a non-zero `logit_bias` shifts the gate exactly as
    /// `sigmoid(z + b)`: for orthogonal q/k (raw dot = 0) the gate is
    /// σ(b), strictly monotone across the sweep set, with the −4 point
    /// (the xHC "start CLOSED" init) at σ(−4) ≈ 0.018.
    #[test]
    fn logit_bias_shifts_gate_monotonically() {
        let d = 16;
        // Disjoint supports → raw dot exactly 0 → gate = σ(b) exactly.
        let q: Vec<f32> = (0..d).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let k: Vec<f32> = (0..d).map(|i| if i % 2 == 1 { 1.0 } else { 0.0 }).collect();
        let v = vec![1.0f32; d];
        let mut prev = -1.0f32;
        for &b in &[-4.0f32, -2.0, 0.0, 2.0, 4.0] {
            let cfg = SigmoidFusionConfig {
                tau: 1.0,
                rmsnorm_eps: 1e-6,
                logit_bias: b,
            };
            let mut out = vec![0.0f32; d];
            sigmoid_fuse_into(&q, &k, &v, &mut out, &cfg);
            let g = out[0]; // gate · 1.0
            assert!(
                g > prev,
                "gate must be strictly monotone in logit_bias: b={b} g={g}"
            );
            let expect = 1.0 / (1.0 + (-b).exp());
            assert!(
                (g - expect).abs() < 1e-5,
                "b={b}: gate {g} vs σ(b) {expect}"
            );
            prev = g;
        }
        // The −4 closed-init value, pinned against the closed-form:
        let cfg_closed = SigmoidFusionConfig {
            tau: 1.0,
            rmsnorm_eps: 1e-6,
            logit_bias: -4.0,
        };
        let mut out = vec![0.0f32; d];
        sigmoid_fuse_into(&q, &k, &v, &mut out, &cfg_closed);
        assert!(
            (out[0] - 0.017_986_21).abs() < 1e-4, // σ(-4) ≈ 0.01799
            "closed-init gate must be σ(-4), got {}",
            out[0]
        );
    }

    #[test]
    fn q_equals_k_gate_near_one() {
        // T3.5: q == k → after RMSNorm, dot ≈ D (cosine ≈ 1).
        //       gate = sigmoid(D / √D) = sigmoid(√D) — large → ≈ 1.0.
        let d = 16;
        let cfg = cfg_for_dim(d);
        let q: Vec<f32> = (1..=d).map(|i| i as f32).collect();
        let k = q.clone();
        let v: Vec<f32> = (1..=d).map(|i| (i as f32) * 0.1).collect();
        let mut out = vec![0.0f32; d];
        sigmoid_fuse_into(&q, &k, &v, &mut out, &cfg);
        // gate ≈ sigmoid(√16) = sigmoid(4) ≈ 0.982
        let gate = out[0] / v[0];
        assert!(
            (gate - 1.0).abs() < 0.05,
            "q==k → gate near 1.0, got {gate}"
        );
    }

    #[test]
    fn q_opposite_k_gate_near_zero() {
        // T3.5: q == -k → after RMSNorm, dot ≈ -D (cosine ≈ -1).
        //       gate ≈ sigmoid(-√D) → ≈ 0.0.
        let d = 16;
        let cfg = cfg_for_dim(d);
        let q: Vec<f32> = (1..=d).map(|i| i as f32).collect();
        let k: Vec<f32> = q.iter().map(|x| -x).collect();
        let v: Vec<f32> = (1..=d).map(|i| (i as f32) * 0.1).collect();
        let mut out = vec![0.0f32; d];
        sigmoid_fuse_into(&q, &k, &v, &mut out, &cfg);
        // gate ≈ sigmoid(-4) ≈ 0.018
        let gate = out[0] / v[0];
        assert!(gate < 0.05, "q==-k → gate near 0.0, got {gate}");
    }

    #[test]
    fn q_orthogonal_k_gate_near_half() {
        // T3.5: q ⊥ k → after RMSNorm, dot ≈ 0 → gate ≈ sigmoid(0) = 0.5.
        // Build an explicit orthogonal pair in even dim: [a, b, 0, 0, ...]
        // vs [0, 0, c, d, ...].
        let d = 16;
        let cfg = cfg_for_dim(d);
        let mut q = vec![0.0f32; d];
        let mut k = vec![0.0f32; d];
        for (i, qi) in q.iter_mut().take(d / 2).enumerate() {
            *qi = (i as f32) + 1.0;
        }
        for (i, ki) in k[d / 2..d].iter_mut().enumerate() {
            *ki = ((i + d / 2) as f32) + 1.0;
        }
        let v = vec![1.0f32; d];
        let mut out = vec![0.0f32; d];
        sigmoid_fuse_into(&q, &k, &v, &mut out, &cfg);
        let gate = out[0]; // v[0]=1.0
        assert!((gate - 0.5).abs() < 1e-4, "q⊥k → gate ≈ 0.5, got {gate}");
    }

    #[test]
    fn ranking_preserved_spearman_high() {
        // T3.5: for fixed q, varying k, the sigmoid gate ranking matches
        // cosine ranking (rank-correlation > 0.95).
        //
        // Build a fixed q, then 10 candidate k vectors with monotonically
        // increasing cosine similarity. Sigmoid gate should preserve that
        // ordering (it's monotonic in the dot product of normalized
        // vectors, and RMSNorm preserves cosine ordering).
        let d = 32;
        let cfg = cfg_for_dim(d);
        // q = unit-ish random vector
        let q: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.1).sin()).collect();

        // Build 10 k vectors with progressively smaller angle to q by
        // interpolating q with an orthogonal direction.
        let mut orth = vec![0.0f32; d];
        for i in 0..d / 2 {
            orth[i] = q[i + d / 2];
            orth[i + d / 2] = -q[i];
        }

        let mut gates: Vec<f32> = Vec::with_capacity(10);
        let mut cosines: Vec<f32> = Vec::with_capacity(10);
        for t in 0..10 {
            let tf = t as f32 / 9.0; // 0.0 ..= 1.0
            // k = (1-t)*orth + t*q — cosine with q grows monotonically in t.
            let k: Vec<f32> = (0..d).map(|i| (1.0 - tf) * orth[i] + tf * q[i]).collect();
            let v = vec![1.0f32; d];
            let mut out = vec![0.0f32; d];
            sigmoid_fuse_into(&q, &k, &v, &mut out, &cfg);
            gates.push(out[0]); // gate (v[0]=1)

            // cosine(q, k)
            let dot_qk: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            let nq: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nk: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
            cosines.push(dot_qk / (nq * nk + 1e-12));
        }

        // Spearman rank-correlation: count concordant vs discordant pairs.
        let n = gates.len();
        let mut concordant = 0isize;
        let mut discordant = 0isize;
        for i in 0..n {
            for j in (i + 1)..n {
                let g_sign = (gates[i] - gates[j]).signum();
                let c_sign = (cosines[i] - cosines[j]).signum();
                if g_sign == c_sign && g_sign != 0.0 {
                    concordant += 1;
                } else if g_sign == -c_sign && g_sign != 0.0 {
                    discordant += 1;
                }
            }
        }
        let total = concordant + discordant;
        assert!(total > 0, "must have at least one comparable pair");
        let rho = (concordant - discordant) as f64 / total as f64;
        assert!(rho > 0.95, "Spearman ρ must be > 0.95, got {rho}");
    }

    #[test]
    fn empty_inputs_are_noop() {
        let cfg = SigmoidFusionConfig::default();
        let q: [f32; 0] = [];
        let k: [f32; 0] = [];
        let v: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        sigmoid_fuse_into(&q, &k, &v, &mut out, &cfg); // must not panic
    }

    #[test]
    fn rmsnorm_zero_input_is_zero_output() {
        let x = vec![0.0f32; 8];
        let mut out = vec![0.0f32; 8];
        rmsnorm_into(&x, 1e-6, &mut out);
        // mean(x²)+eps = eps; inv_rms = 1/sqrt(eps) is large, but 0*large=0.
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn rmsnorm_unit_vector() {
        // [1,0,0,0]: mean(x²)=0.25, rms=0.5, output = [2,0,0,0]
        let x = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut out = vec![0.0f32; 4];
        rmsnorm_into(&x, 1e-6, &mut out);
        assert!((out[0] - 2.0).abs() < 1e-3, "got {}", out[0]);
        assert!(out[1..].iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn m1_multi_branch_matches_single_branch() {
        // T3.6: M=1 must reduce bit-identically to a single call to
        // sigmoid_fuse_into.
        let d = 16;
        let cfg = cfg_for_dim(d);
        let q: Vec<f32> = (1..=d).map(|i| i as f32).collect();
        let k: Vec<f32> = (0..d).map(|i| (i as f32) * 0.3 + 1.0).collect();
        let v: Vec<f32> = (0..d).map(|i| (i as f32) * 0.1).collect();

        // Single-branch reference.
        let mut out_ref = vec![0.0f32; d];
        sigmoid_fuse_into(&q, &k, &v, &mut out_ref, &cfg);

        // Multi-branch with M=1.
        let mut out_mb = vec![0.0f32; d];
        let q_slices = [&q[..]];
        let k_slices = [&k[..]];
        let mut out_slices = [out_mb.as_mut_slice()];
        sigmoid_fuse_multi_branch_into(&q_slices, &k_slices, &v, &mut out_slices, &cfg);

        assert_eq!(out_mb, out_ref, "M=1 must match single-branch output");
    }

    #[test]
    fn m4_q_equals_k_all_gates_near_one() {
        // T3.6: M=4, q_i = k_i for all i → all gates near 1.0.
        let d = 16;
        let cfg = cfg_for_dim(d);
        let q0: Vec<f32> = (1..=d).map(|i| i as f32).collect();
        let q1: Vec<f32> = (0..d).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let q2: Vec<f32> = (0..d).map(|i| (i as f32) * 0.7 + 2.0).collect();
        let q3: Vec<f32> = (0..d).map(|i| (i as f32) * 0.9 + 3.0).collect();
        let v: Vec<f32> = vec![1.0f32; d];

        let q_slices = [&q0[..], &q1[..], &q2[..], &q3[..]];
        // k_i = q_i → dot(q_norm, k_norm) ≈ 1 → gate ≈ sigmoid(√d) ≈ 1.
        let k_slices = [&q0[..], &q1[..], &q2[..], &q3[..]];
        // Flat output buffer + per-branch views via split_at_mut (avoids
        // multiple simultaneous &mut borrows of a Vec).
        let mut out_buf = vec![0.0f32; 4 * d];
        let (out0, rest) = out_buf.split_at_mut(d);
        let (out1, rest) = rest.split_at_mut(d);
        let (out2, out3) = rest.split_at_mut(d);
        let mut out_slices = [out0, out1, out2, out3];
        sigmoid_fuse_multi_branch_into(&q_slices, &k_slices, &v, &mut out_slices, &cfg);

        for m in 0..4 {
            // gate * v[j] = gate (since v[j]=1) → gate ≈ out_buf[m*d].
            let gate = out_buf[m * d];
            assert!(
                (gate - 1.0).abs() < 0.05,
                "branch {m}: gate ≈ 1, got {gate}"
            );
        }
    }

    #[test]
    fn m4_orthogonal_q_k_all_gates_near_half() {
        // T3.6: M=4, q_i ⊥ k_i for all i → all gates near 0.5.
        let d = 16;
        let cfg = cfg_for_dim(d);
        // Build 4 orthogonal (q_i, k_i) pairs: q has first half non-zero,
        // k has second half non-zero.
        let mk_orth_pair = |seed: f32| -> (Vec<f32>, Vec<f32>) {
            let mut q = vec![0.0f32; d];
            let mut k = vec![0.0f32; d];
            for (i, qi) in q.iter_mut().take(d / 2).enumerate() {
                *qi = seed + i as f32;
            }
            for (i, ki) in k[d / 2..d].iter_mut().enumerate() {
                *ki = seed + (i + d / 2) as f32;
            }
            (q, k)
        };
        let (q0, k0) = mk_orth_pair(1.0);
        let (q1, k1) = mk_orth_pair(2.0);
        let (q2, k2) = mk_orth_pair(3.0);
        let (q3, k3) = mk_orth_pair(4.0);
        let v: Vec<f32> = vec![1.0f32; d];

        let q_slices = [&q0[..], &q1[..], &q2[..], &q3[..]];
        let k_slices = [&k0[..], &k1[..], &k2[..], &k3[..]];
        let mut out_buf = vec![0.0f32; 4 * d];
        let (out0, rest) = out_buf.split_at_mut(d);
        let (out1, rest) = rest.split_at_mut(d);
        let (out2, out3) = rest.split_at_mut(d);
        let mut out_slices = [out0, out1, out2, out3];
        sigmoid_fuse_multi_branch_into(&q_slices, &k_slices, &v, &mut out_slices, &cfg);

        for m in 0..4 {
            let gate = out_buf[m * d]; // v[0] = 1
            assert!(
                (gate - 0.5).abs() < 1e-3,
                "branch {m}: gate ≈ 0.5, got {gate}"
            );
        }
    }

    #[test]
    fn m0_multi_branch_is_noop() {
        // Edge case: M=0 → no work.
        let cfg = SigmoidFusionConfig::default();
        let v: [f32; 0] = [];
        let q_slices: &[&[f32]] = &[];
        let k_slices: &[&[f32]] = &[];
        let out_slices: &mut [&mut [f32]] = &mut [];
        sigmoid_fuse_multi_branch_into(q_slices, k_slices, &v, out_slices, &cfg);
    }

    // Note: we intentionally do NOT import or reference simd_scale_inplace
    // here even though it's used internally — that's an impl detail. But we
    // keep the import alive via a no-op test.
    #[test]
    fn simd_scale_inplace_is_available() {
        // Sanity: the SIMD primitive we depend on is reachable.
        use crate::simd::simd_scale_inplace;
        let mut x = [2.0f32, 4.0, 8.0];
        simd_scale_inplace(&mut x, 0.5);
        assert_eq!(&x, &[1.0, 2.0, 4.0]);
    }
}
