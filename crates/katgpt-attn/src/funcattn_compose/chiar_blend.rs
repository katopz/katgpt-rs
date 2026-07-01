//! T5.2 — Compose FUNCATTN with CHIAR spectral-entropy operator routing.
//!
//! Plan 286 T5.2: route between FUNCATTN and a fallback operator (Parallax or
//! SDPA) by **per-token spectral entropy** H(x). FUNCATTN for low-entropy
//! (structured) tokens, the fallback for high-entropy (chaotic) tokens.
//!
//! # Why a blend, not a hard split
//!
//! FUNCATTN is a whole-sequence operator: its adaptive basis Φ and the
//! Tikhonov solve `C = Q̃ · reg⁻¹ · K̃ᵀ` depend on *all* n tokens jointly. It
//! cannot be applied to an arbitrary subset of tokens without recomputing the
//! basis, so a hard per-token partition would forfeit FUNCATTN's signal. The
//! honest modelless design that preserves per-token entropy routing is a
//! **soft sigmoid blend** of two whole-sequence outputs:
//!
//! ```text
//! gate[n]   = sigmoid((H(x_n) − τ) · β)      // ∈ (0, 1)
//! out[n,:]  = gate[n] · fallback[n,:] + (1 − gate[n]) · funcattn_out[n,:]
//! ```
//!
//! Low-entropy tokens (`H < τ`) → gate ≈ 0 → FUNCATTN dominates. High-entropy
//! tokens (`H > τ`) → gate ≈ 1 → fallback dominates. This is per-token,
//! modelless (no learned router), and uses **sigmoid, never softmax** (AGENTS.md).
//!
//! # CHIAR trait integration
//!
//! [`FuncAttnChiaroscuroOp`] implements CHIAR's [`ChiaroscuroOp`] trait so
//! FUNCATTN plugs into CHIAR's [`ChiaroscuroRouter`] for utilization tracking
//! and collapse discovery (Plan 269 Fusion B/C). It is a **routing anchor** —
//! like CHIAR's own [`FullAttnOp`], its `forward_token` is an identity copy
//! because the real FUNCATTN operator runs cross-token. The actual per-token
//! routing decision is the [`blend_by_entropy_into`] gate above.
//!
//! # Fallback choice
//!
//! The plan names Parallax as the high-entropy fallback. Parallax (Plan 135)
//! ships sigmoid-basis but its audit (2026-05-30) found **no gain** without
//! Muon-trained weights, and the G2 caveat (Plan 286 T3.2) documents sigmoid
//! Parallax diverging to NaN under naive FD-SGD. The blend here is therefore
//! **fallback-agnostic**: the caller passes any sequence-level operator's
//! output as `fallback_out`. The recommended robust default is SDPA
//! (`tiled_attention`, already a `funcattn` feature dependency); Parallax can
//! be wired by a caller that has trained its weights.
//!
//! # Cost
//!
//! The blend costs 2× attention compute (both operators run on the full
//! sequence) plus `n` DCT entropy computations. It is a quality/latency
//! tradeoff knob, not a speedup — opt-in, off by default. The per-token DCT
//! is `O(d log d)` (negligible vs `O(n·d·k)` FUNCATTN + `O(n²·d)` SDPA).

use crate::chiaroscuro::entropy::{sigmoid, spectral_entropy_dct_into};
use crate::chiaroscuro::op_trait::ChiaroscuroOp;
use rustfft::{FftPlanner, num_complex::Complex32};

/// Spectral-entropy crossover below which a token is "structured" (FUNCATTN)
/// and above which it is "chaotic" (fallback). Default matches CHIAR's `τ_hi`
/// cluster midpoint (0.865, Plan 269). Tokens with `H ≈ τ` get a near-50/50
/// blend; the [`FuncAttnChiarBlendConfig::beta`] sharpness controls the width.
pub const DEFAULT_FUNCATTN_CHIAR_TAU: f32 = 0.865;

/// Default gate sharpness. `β = 12` makes the gate transition over roughly
/// `±0.15` in H(x) — a soft but decisive switch. Higher = harder routing.
pub const DEFAULT_FUNCATTN_CHIAR_BETA: f32 = 12.0;

/// Configuration for the per-token entropy blend.
///
/// `tau` is the entropy crossover; `beta` is the sigmoid slope (sharpness).
/// Both are fixed inference-time hyperparameters (modelless — no learning).
#[derive(Debug, Clone, Copy)]
pub struct FuncAttnChiarBlendConfig {
    /// Entropy crossover τ. Tokens with `H(x) < τ` lean FUNCATTN; `H(x) > τ`
    /// lean fallback.
    pub tau: f32,
    /// Sigmoid sharpness β. `β → ∞` is a hard threshold at `H = τ`; `β → 0`
    /// is a uniform 50/50 blend.
    pub beta: f32,
}

impl Default for FuncAttnChiarBlendConfig {
    fn default() -> Self {
        Self { tau: DEFAULT_FUNCATTN_CHIAR_TAU, beta: DEFAULT_FUNCATTN_CHIAR_BETA }
    }
}

/// CHIAR routing anchor for FUNCATTN.
///
/// Declares FUNCATTN's eligibility as the low-entropy (`H ≤ entropy_hi`)
/// operator in a [`crate::chiaroscuro::ChiaroscuroRouter`]. This lets CHIAR's
/// collapse-discovery harness observe FUNCATTN's utilization alongside
/// `DctMixOp` / `FullAttnOp`. The actual per-token routing is the
/// [`blend_by_entropy_into`] sigmoid gate, not this anchor's `forward_token`.
pub struct FuncAttnChiaroscuroOp {
    entropy_hi: f32,
}

impl FuncAttnChiaroscuroOp {
    /// Create the anchor with an upper entropy bound (default `0.865`).
    pub fn new(entropy_hi: f32) -> Self {
        Self { entropy_hi }
    }
}

impl Default for FuncAttnChiaroscuroOp {
    fn default() -> Self {
        Self::new(DEFAULT_FUNCATTN_CHIAR_TAU)
    }
}

impl ChiaroscuroOp for FuncAttnChiaroscuroOp {
    /// FUNCATTN is eligible for the lowest-entropy tokens (structured signal).
    #[inline]
    fn entropy_lo(&self) -> f32 {
        0.0
    }

    #[inline]
    fn entropy_hi(&self) -> f32 {
        self.entropy_hi
    }

    /// FUNCATTN is `O(n·d·k + d³)` — cheaper than full `O(n²·d)` attention for
    /// large n, but it runs on the whole sequence. Cost is relative to
    /// `FullAttnOp` (= 1.0): for typical `k ≪ n`, FUNCATTN is markedly cheaper,
    /// so we report a sub-1.0 relative cost. Kept conservative at 0.5 since the
    /// blend still pays for the fallback operator.
    #[inline]
    fn relative_cost(&self) -> f32 {
        0.5
    }

    fn name(&self) -> &'static str {
        "FuncAttn"
    }

    /// Identity copy — FUNCATTN is cross-token; the real per-token decision is
    /// [`blend_by_entropy_into`]. Mirrors CHIAR's `FullAttnOp::forward_token`.
    fn forward_token(&self, x: &[f32], out: &mut [f32]) {
        let n = x.len().min(out.len());
        out[..n].copy_from_slice(&x[..n]);
    }
}

/// Compute per-token spectral entropies `H(x_n)` for an `(n, d)` input, zero-alloc.
///
/// Writes `n` entropy values to `out`. Reuses the caller-provided `scratch` and
/// `planner` across calls to amortize allocations (mirrors CHIAR's
/// `spectral_entropy_dct_into` contract).
pub fn compute_token_entropies_into(
    x: &[f32],
    n: usize,
    d: usize,
    scratch: &mut Vec<Complex32>,
    planner: &mut FftPlanner<f32>,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), n * d, "x must be (n, d)");
    debug_assert_eq!(out.len(), n, "out must be length n");
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        out[i] = spectral_entropy_dct_into(row, scratch, planner);
    }
}

/// Per-token sigmoid blend of FUNCATTN and a fallback operator's outputs.
///
/// For each token `n`:
/// ```text
/// gate[n]  = sigmoid((H(x_n) − τ) · β)
/// out[n,:] = gate[n] · fallback[n,:] + (1 − gate[n]) · funcattn_out[n,:]
/// ```
/// Low-entropy tokens → FUNCATTN; high-entropy → fallback. Zero-alloc.
///
/// # Arguments
/// * `funcattn_out` — FUNCATTN forward output, `(n, d)`.
/// * `fallback_out` — fallback operator (SDPA/Parallax) forward output, `(n, d)`.
/// * `entropies`    — per-token `H(x)`, length `n` (from [`compute_token_entropies_into`]
///   or CHIAR's `spectral_entropy_dct`).
/// * `cfg`          — blend config (τ crossover, β sharpness).
/// * `out`          — output `(n, d)`, pre-allocated by caller. May alias
///   `funcattn_out` or `fallback_out` (single-pass write, no read-after-write
///   hazard per element).
pub fn blend_by_entropy_into(
    funcattn_out: &[f32],
    fallback_out: &[f32],
    entropies: &[f32],
    cfg: &FuncAttnChiarBlendConfig,
    out: &mut [f32],
) {
    let n = entropies.len();
    debug_assert_eq!(funcattn_out.len(), fallback_out.len());
    debug_assert_eq!(funcattn_out.len() % n, 0, "output must be (n, d)");
    debug_assert_eq!(out.len(), funcattn_out.len());
    if n == 0 {
        return;
    }
    let d = funcattn_out.len() / n;
    for i in 0..n {
        // gate ∈ (0,1): H >> τ → ~1 (fallback), H << τ → ~0 (funcattn).
        let gate = sigmoid((entropies[i] - cfg.tau) * cfg.beta);
        let inv_gate = 1.0 - gate;
        let f_row = &funcattn_out[i * d..(i + 1) * d];
        let b_row = &fallback_out[i * d..(i + 1) * d];
        let o_row = &mut out[i * d..(i + 1) * d];
        for j in 0..d {
            o_row[j] = gate * b_row[j] + inv_gate * f_row[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chiaroscuro::op_trait::ChiaroscuroRouter;

    fn lcg_fill(out: &mut [f32], seed: u32) {
        let mut s = seed;
        for x in out.iter_mut() {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            *x = ((s >> 8) as f32 / 16777216.0) - 0.5;
        }
    }

    #[test]
    fn op_anchor_routes_low_entropy_to_funcattn() {
        // Plug FuncAttnChiaroscuroOp + FullAttnOp into a CHIAR router and
        // verify a low-entropy token routes to FUNCATTN, high to FullAttn.
        use crate::chiaroscuro::op_trait::FullAttnOp;
        let mut router = ChiaroscuroRouter::new(vec![
            Box::new(FuncAttnChiaroscuroOp::default()),
            Box::new(FullAttnOp::default()),
        ]);
        // Constant vector → H ≈ 0 (structured) → FUNCATTN (index 0).
        let low = vec![0.5f32; 64];
        let idx_low = router.route(&low);
        assert_eq!(idx_low, 0, "low-entropy token should route to FuncAttn");

        // Pseudo-random vector → H closer to 1 (chaotic) → FullAttn (index 1).
        let mut high = vec![0.0f32; 128];
        lcg_fill(&mut high, 7);
        let idx_high = router.route(&high);
        assert_eq!(idx_high, 1, "high-entropy token should route to FullAttn");
    }

    #[test]
    fn op_forward_token_is_identity() {
        let op = FuncAttnChiaroscuroOp::default();
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        op.forward_token(&x, &mut out);
        assert_eq!(out, x, "anchor forward_token must be identity copy");
    }

    #[test]
    fn blend_low_entropy_yields_funcattn_output() {
        // H = 0 for all tokens → gate ≈ 0 → out ≈ funcattn_out. Use a test
        // config whose τ straddles {0,1} so the sigmoid saturates crisply.
        let n = 4;
        let d = 3;
        let funcattn_out: Vec<f32> = (0..n * d).map(|i| i as f32).collect();
        let fallback_out: Vec<f32> = (0..n * d).map(|i| 100.0 + i as f32).collect();
        let entropies = vec![0.0f32; n];
        let cfg = FuncAttnChiarBlendConfig { tau: 0.5, beta: 20.0 };
        let mut out = vec![0.0f32; n * d];
        blend_by_entropy_into(&funcattn_out, &fallback_out, &entropies, &cfg, &mut out);
        for (o, f) in out.iter().zip(funcattn_out.iter()) {
            assert!((o - f).abs() < 1e-2, "H=0 → out≈funcattn (got {o}, want {f})");
        }
    }

    #[test]
    fn blend_high_entropy_yields_fallback_output() {
        // H = 1 for all tokens → gate ≈ 1 → out ≈ fallback_out.
        let n = 4;
        let d = 3;
        let funcattn_out: Vec<f32> = (0..n * d).map(|i| i as f32).collect();
        let fallback_out: Vec<f32> = (0..n * d).map(|i| 100.0 + i as f32).collect();
        let entropies = vec![1.0f32; n];
        let cfg = FuncAttnChiarBlendConfig { tau: 0.5, beta: 20.0 };
        let mut out = vec![0.0f32; n * d];
        blend_by_entropy_into(&funcattn_out, &fallback_out, &entropies, &cfg, &mut out);
        for (o, b) in out.iter().zip(fallback_out.iter()) {
            assert!((o - b).abs() < 1e-2, "H=1 → out≈fallback (got {o}, want {b})");
        }
    }

    #[test]
    fn blend_mid_entropy_is_convex_combination() {
        // H = τ exactly → gate = sigmoid(0) = 0.5 → out = average of the two.
        let n = 1;
        let d = 2;
        let funcattn_out = vec![0.0f32, 0.0];
        let fallback_out = vec![10.0f32, 20.0];
        let cfg = FuncAttnChiarBlendConfig::default();
        let entropies = vec![cfg.tau];
        let mut out = vec![0.0f32; n * d];
        blend_by_entropy_into(&funcattn_out, &fallback_out, &entropies, &cfg, &mut out);
        // out = 0.5*fallback + 0.5*funcattn = 0.5*fallback.
        assert!((out[0] - 5.0).abs() < 1e-4, "H=τ → 50/50 blend (got {})", out[0]);
        assert!((out[1] - 10.0).abs() < 1e-4, "H=τ → 50/50 blend (got {})", out[1]);
    }

    #[test]
    fn compute_entropies_constants_are_low() {
        // Constant token rows → H ≈ 0.
        let n = 3;
        let d = 64;
        let x = vec![0.42f32; n * d];
        let mut entropies = vec![0.0f32; n];
        let mut scratch = Vec::new();
        let mut planner = FftPlanner::new();
        compute_token_entropies_into(&x, n, d, &mut scratch, &mut planner, &mut entropies);
        for h in &entropies {
            assert!(*h < 0.1, "constant token H should be ≈ 0 (got {h})");
        }
    }

    #[test]
    fn blend_is_continuous_in_entropy() {
        // A small entropy perturbation must move the output by a small amount.
        let funcattn_out = vec![0.0f32];
        let fallback_out = vec![1.0f32];
        let cfg = FuncAttnChiarBlendConfig::default();
        let mut out_lo = vec![0.0f32];
        let mut out_hi = vec![0.0f32];
        blend_by_entropy_into(&funcattn_out, &fallback_out, &[cfg.tau - 0.01], &cfg, &mut out_lo);
        blend_by_entropy_into(&funcattn_out, &fallback_out, &[cfg.tau + 0.01], &cfg, &mut out_hi);
        let delta = (out_hi[0] - out_lo[0]).abs();
        assert!(delta > 0.0 && delta < 0.5, "small entropy change → small output change (delta={delta})");
    }
}
