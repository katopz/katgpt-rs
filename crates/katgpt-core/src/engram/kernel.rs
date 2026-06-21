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
        }
    }
}

/// RMSNorm `x → x / √(mean(x²) + eps)`, writing the result into `out`.
///
/// Plan T3.2. In-place-safe: `out` MAY alias `x` (read-then-write per
/// element, no cross-element aliasing). Uses [`simd::simd_sum_sq`] for the
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
    // out ≠ x in general, so do a manual copy-with-scale.
    for i in 0..n {
        out[i] = x[i] * inv_rms;
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
    let gate = fast_sigmoid(normalized_dot / config.tau);

    // Write gate * v[j] into out — SIMD-friendly stride-1 write. We use a
    // manual loop (not simd_scale_inplace) because src and dst are
    // different slices; a small D=32 loop auto-vectorizes cleanly.
    for j in 0..d {
        out[j] = gate * v[j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a config with a given D so the tau default makes sense.
    fn cfg_for_dim(d: usize) -> SigmoidFusionConfig {
        SigmoidFusionConfig {
            tau: (d as f32).sqrt(),
            rmsnorm_eps: 1e-6,
        }
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
        for i in 0..d / 2 {
            q[i] = (i as f32) + 1.0;
        }
        for i in d / 2..d {
            k[i] = (i as f32) + 1.0;
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
