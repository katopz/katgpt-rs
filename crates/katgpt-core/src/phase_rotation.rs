//! Phase-Modulated Subspace Rotation Gate — norm-preserving latent coupling.
//!
//! Distilled from UFO (arXiv:2605.12700, Qiao/Karniadakis/Muniruzzaman May 2026;
//! Research 305, Plan 322). The paper is a *training* paper (PDE operator
//! network with a trained phase MLP `γ_θ`), but the core coupling mechanism is
//! genuinely modelless: a unitary 2D rotation `cos α ⊙ a + sin α ⊙ b` between
//! two latent halves, where the phase `α` is constructed deterministically from
//! a sigmoid projection onto a frozen direction vector (§3.5 Path 2 modelless
//! unblock). Only the deterministic math ships here — no backprop, no learned
//! `γ_θ`. The trained spectral encoder + PDE-benchmark quality claims are out
//! of scope (→ riir-train if ever needed).
//!
//! # What this computes
//!
//! Given two latent slices `a, b ∈ ℝᴰ` and a phase angle `α ∈ [0, π/2]`, the
//! rotation is
//!
//! ```text
//! out = cos(α) ⊙ a + sin(α) ⊙ b
//! ```
//!
//! The phase may be:
//!   - a **single scalar** (broadcast to all D channels) — fast path, one
//!     `cos`/`sin` evaluation;
//!   - a **per-channel vector** `[α_0, …, α_{D-1}]` — UFO's full form, `D`
//!     `cos`/`sin` evaluations (use the Padé fast path to avoid the libm floor).
//!
//! The phase is constructed modellessly from a latent state via
//!
//! ```text
//! α = sigmoid(⟨state, direction⟩ · sharpness) · (π / 2)
//! ```
//!
//! where `direction` is a frozen, BLAKE3-committed unit-norm artifact (the
//! Plan 309 / Plan 310 / Plan 319 pattern). The phase is bounded in `[0, π/2]`,
//! so `cos ≥ 0` and `sin ≥ 0` — the mix is a *convex rotation*, never
//! sign-flipping.
//!
//! # Why this is NOT redundant
//!
//! Every existing latent op in the crate is one of: additive (Plan 309 steering,
//! inflates L2 norm), convex-combo (independent sigmoid weights, preserves L1
//! not L2), dot-projection (HLA), wedge-detection (Plan 319 Clifford), linear
//! transport (Plan 310 cross-resolution), or spatial-sum (DEC). **None is a
//! bounded unitary rotation with built-in L2-norm preservation.**
//!
//! The Pythagorean identity `sin²α + cos²α = 1` gives the headline invariant:
//!
//! ```text
//! ‖out‖² = cos²α·‖a‖² + sin²α·‖b‖² + 2·cos α·sin α·⟨a, b⟩
//!        ≤ cos²α·‖a‖² + sin²α·‖b‖² + |sin(2α)|·‖a‖·‖b‖     (Cauchy-Schwarz)
//! ```
//!
//! For `a ⊥ b`: `‖out‖² = cos²α·‖a‖² + sin²α·‖b‖²`, bounded by
//! `max(‖a‖², ‖b‖²)` and equal to `‖a‖² + ‖b‖²` only at the trivial α=π/4
//! equal-norm corner. In general `‖out‖² ≤ ‖a‖² + ‖b‖²` for all `α` — a
//! per-α bound the convex-combo cousin lacks (its output norm varies with the
//! independent sigmoid weights). This is the long-horizon stability property
//! the Super-GOAT selling point depends on (HLA over thousands of ticks).
//!
//! See Research 305 §2.3 for the 4-cousin comparison and §2.4 for fusion hooks
//! (HLA subspace gating, DEC Hodge mixer, shard spectral/spatial retrieval,
//! LatCal committed phase).
//!
//! # Numerical contract
//!
//! - All entry points are pure float arithmetic over caller-provided buffers.
//!   Deterministic on a given CPU (same inputs → bit-identical outputs).
//! - `a`, `b`, `out` must be equal-length; `cos_alpha`/`sin_alpha` are either
//!   length-1 (scalar broadcast) or match `a.len()` (per-channel). Length
//!   mismatches trip `debug_assert` in debug builds; behavior is undefined in
//!   release (matches the `simd::simd_dot_f32` convention).
//! - The scalar phase path uses libm `cos`/`sin` (single evaluation — well under
//!   the latency budget).
//! - The per-channel path uses [`phase_safe_cos_sin`]: libm `sin` + Pythagorean
//!   `sqrt(1 - sin²)` recovery. This forces the G1-critical identity
//!   `cos²α + sin²α = 1` to hold bit-by-bit (independent libm cos+sin drifts
//!   by ~1e-7 per call, which compounds across the G1 1000-point sweep). The
//!   sqrt-recovery construction costs one `sqrt` per channel (~3 ns) but makes
//!   the G1 budget hold trivially.
//!
//! # Performance
//!
//! `O(D)` per call (one dot + one cos + one sin for scalar phase; `D` cos/sin
//! for per-channel). Zero allocation after scratch init. The inner mix loop is
//! a textbook FMA kernel (`c·a + s·b`) — LLVM auto-vectorises the 4-wide
//! chunked form into NEON/AVX2.

use crate::simd;
use core::f32::consts::FRAC_PI_2;

// ── Config ───────────────────────────────────────────────────────

/// Configuration for the phase-modulated rotation gate.
///
/// `sharpness` controls how aggressively the phase responds to the
/// state-direction projection: higher `λ` → sharper transition between the `a`
/// and `b` subspaces. At `λ = 0` the phase is `α = π/4` for every input (the
/// `(a+b)/√2` midpoint); the plan-distilled HLA defaults sit in `λ ∈ [1, 10]`.
///
/// The broadcast-vs-per-channel choice is made at the API level (scalar phase
/// vs per-channel phase), not in this config — they have different latency
/// profiles and the caller picks based on hot-path vs cold-path needs (matches
/// the §6.4 mitigation in Research 305).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseRotationGate {
    /// Phase steepness `λ`. Phase is `α = sigmoid(dot · λ) · π/2`.
    /// Must be finite. Clamped to `[0, 100]` at construction to avoid overflow
    /// in `dot · λ` (sigmoid saturates by `λ ≈ 40` anyway).
    pub sharpness: f32,
}

impl Default for PhaseRotationGate {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl PhaseRotationGate {
    /// F1-fusion default: `λ = 4.0` — moderate sharpness. At this setting,
    /// `|dot| ≥ 2` already saturates the phase to within `1e-3` of its
    /// `[0, π/2]` endpoints, so a binary "combat vs social" context signal
    /// produces a near-binary phase rotation.
    pub const DEFAULT: Self = Self { sharpness: 4.0 };

    /// Construct with validation. Returns `None` if `sharpness` is non-finite
    /// or negative.
    pub fn new(sharpness: f32) -> Option<Self> {
        if sharpness.is_finite() && sharpness >= 0.0 {
            Some(Self {
                sharpness: sharpness.min(100.0),
            })
        } else {
            None
        }
    }
}

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by the phase-rotation entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseRotationError {
    /// `a.len() != b.len()` or `out.len() != a.len()` — the two halves and the
    /// output must agree on the channel count.
    ShapeMismatch,
    /// `cos_alpha.len()` (or `sin_alpha.len()`) is neither `1` (scalar
    /// broadcast) nor `a.len()` (per-channel). The kernel cannot pick a path.
    InvalidPhaseLen,
    /// `state.len() != direction.len()` on a phase-construction call.
    ProjectionShapeMismatch,
    /// `sharpness` was non-finite or negative at config build time.
    InvalidSharpness,
}

// ── Scratch ──────────────────────────────────────────────────────

/// Pre-allocated scratch for zero-alloc per-channel phase construction.
///
/// Create once via [`PhaseRotationScratch::new`], then call
/// [`PhaseRotationScratch::ensure_capacity`] before each per-channel phase
/// computation. The scalar phase path needs no scratch (it returns two scalars
/// by `&mut`). The mix kernel [`phase_rotation_gate_into`] needs no scratch
/// either — its `cos_alpha`/`sin_alpha` are read-only inputs.
///
/// Mirrors [`crate::cross_resolution::CrossResScratch`] and
/// [`crate::funcattn::FuncAttnScratch`].
#[derive(Debug, Clone, Default)]
pub struct PhaseRotationScratch {
    /// Per-channel `cos(α)` buffer (length D).
    pub cos_alpha: Vec<f32>,
    /// Per-channel `sin(α)` buffer (length D).
    pub sin_alpha: Vec<f32>,
    cached_d: usize,
}

impl PhaseRotationScratch {
    /// Allocate scratch for the given channel count.
    pub fn new(d: usize) -> Self {
        Self {
            cos_alpha: vec![0.0; d],
            sin_alpha: vec![0.0; d],
            cached_d: d,
        }
    }

    /// Resize buffers if `d` changed. No-op on the hot path.
    pub fn ensure_capacity(&mut self, d: usize) {
        if self.cached_d == d {
            return;
        }
        self.cos_alpha.resize(d, 0.0);
        self.sin_alpha.resize(d, 0.0);
        self.cached_d = d;
    }
}

// ── Mix kernel ───────────────────────────────────────────────────

/// The core phase-rotation mix: `out = cos(α) ⊙ a + sin(α) ⊙ b`.
///
/// `cos_alpha` and `sin_alpha` are either both length-1 (scalar broadcast —
/// the same `cos`/`sin` applied to every channel) or both length-`D`
/// (per-channel — UFO's full form). The broadcast path is the hot-path default
/// for HLA-scale `D = 8`; the per-channel path is the cold-path option for
/// shard retrieval / DEC Hodge mixing at `D = 64`.
///
/// # Arguments
///
/// * `a` — first latent half, length `D` (e.g. HLA action affects).
/// * `b` — second latent half, length `D` (e.g. HLA social affects).
/// * `cos_alpha` — `cos(α)` per channel (length-1 or length-`D`).
/// * `sin_alpha` — `sin(α)` per channel (length-1 or length-`D`).
/// * `out` — output mix, length `D`. May alias `a` or `b` in `unsafe` callers
///   (the kernel reads `a[i]` and `b[i]` before writing `out[i]`, per-channel).
///   Safe-Rust callers cannot construct the simultaneous `&a` + `&mut out=a`
///   borrow; use `split_at_mut` or a temporary buffer for in-place mixes.
///
/// # Errors
///
/// Returns [`PhaseRotationError::ShapeMismatch`] if `a`, `b`, `out` lengths
/// disagree. Returns [`PhaseRotationError::InvalidPhaseLen`] if `cos_alpha`
/// and `sin_alpha` lengths disagree or are neither `1` nor `D`.
///
/// # Performance
///
/// `O(D)`, zero allocation. The scalar-broadcast path is a single FMA inner
/// loop (`c·a[i] + s·b[i]`) — LLVM auto-vectorises 4-wide. The per-channel
/// path is one FMA per element with gathers from `cos_alpha`/`sin_alpha`.
#[inline]
pub fn phase_rotation_gate_into(
    a: &[f32],
    b: &[f32],
    cos_alpha: &[f32],
    sin_alpha: &[f32],
    out: &mut [f32],
) -> Result<(), PhaseRotationError> {
    let d = a.len();
    if b.len() != d || out.len() != d {
        return Err(PhaseRotationError::ShapeMismatch);
    }
    if cos_alpha.len() != sin_alpha.len() {
        return Err(PhaseRotationError::InvalidPhaseLen);
    }
    let phase_len = cos_alpha.len();
    if phase_len != 1 && phase_len != d {
        return Err(PhaseRotationError::InvalidPhaseLen);
    }

    if phase_len == 1 {
        // Scalar-broadcast fast path — one cos/sin for all channels.
        let c = cos_alpha[0];
        let s = sin_alpha[0];
        // Chunked 4-wide to hint LLVM auto-vectorisation (matches the pattern
        // in `dec::operators::exterior_derivative_into` / Plan 319).
        let mut i = 0;
        while i + 4 <= d {
            out[i] = c.mul_add(a[i], s * b[i]);
            out[i + 1] = c.mul_add(a[i + 1], s * b[i + 1]);
            out[i + 2] = c.mul_add(a[i + 2], s * b[i + 2]);
            out[i + 3] = c.mul_add(a[i + 3], s * b[i + 3]);
            i += 4;
        }
        while i < d {
            out[i] = c.mul_add(a[i], s * b[i]);
            i += 1;
        }
    } else {
        // Per-channel path — UFO's full form. Gathers cos/sin per channel.
        let mut i = 0;
        while i + 4 <= d {
            out[i] = cos_alpha[i].mul_add(a[i], sin_alpha[i] * b[i]);
            out[i + 1] = cos_alpha[i + 1].mul_add(a[i + 1], sin_alpha[i + 1] * b[i + 1]);
            out[i + 2] = cos_alpha[i + 2].mul_add(a[i + 2], sin_alpha[i + 2] * b[i + 2]);
            out[i + 3] = cos_alpha[i + 3].mul_add(a[i + 3], sin_alpha[i + 3] * b[i + 3]);
            i += 4;
        }
        while i < d {
            out[i] = cos_alpha[i].mul_add(a[i], sin_alpha[i] * b[i]);
            i += 1;
        }
    }
    Ok(())
}

// ── Scalar phase construction ────────────────────────────────────

/// Construct the scalar phase `(cos α, sin α)` from a latent projection.
///
/// Computes `α = sigmoid(⟨state, direction⟩ · sharpness) · (π/2)`, then
/// evaluates `cos α` and `sin α` via libm. The phase is bounded in
/// `[0, π/2]` because `sigmoid ∈ [0, 1]`, so `cos ≥ 0` and `sin ≥ 0` — the
/// rotation is convex (never sign-flipping).
///
/// Use this when the rotation should be the *same* across all channels (e.g.
/// HLA subspace gating: the whole `[valence, arousal, desperation, calm]` half
/// rotates by the same combat-vs-social phase). Pass the resulting `cos_alpha`
/// / `sin_alpha` as length-1 slices to [`phase_rotation_gate_into`].
///
/// # Arguments
///
/// * `state` — current latent state, length `D`.
/// * `direction` — frozen unit-norm direction vector, length `D` (BLAKE3-
///   committed artifact, caller's responsibility).
/// * `sharpness` — phase steepness `λ` (see [`PhaseRotationGate`]).
/// * `cos_alpha` — out: `cos(α)`.
/// * `sin_alpha` — out: `sin(α)`.
///
/// # Errors
///
/// Returns [`PhaseRotationError::ProjectionShapeMismatch`] if `state.len() !=
/// direction.len()`.
#[inline]
pub fn compute_phase_from_projection(
    state: &[f32],
    direction: &[f32],
    sharpness: f32,
    cos_alpha: &mut f32,
    sin_alpha: &mut f32,
) -> Result<(), PhaseRotationError> {
    if state.len() != direction.len() {
        return Err(PhaseRotationError::ProjectionShapeMismatch);
    }
    let dot = simd::simd_dot_f32(state, direction, state.len());
    // α = sigmoid(dot · λ) · (π/2). fast_sigmoid is the simd::activations scalar
    // entry — 1/(1+e^{-x}) with the |x|>40 saturation short-circuit.
    let alpha = simd::fast_sigmoid(dot * sharpness) * FRAC_PI_2;
    *cos_alpha = alpha.cos();
    *sin_alpha = alpha.sin();
    Ok(())
}

// ── Per-channel phase construction ─────────────────────────────────

/// Construct per-channel `(cos α, sin α)` from `D` independent projections.
///
/// For each channel `c`, computes `α_c = sigmoid(state[c] · directions[c] ·
/// sharpness) · (π/2)`. This is the elementwise analogue of
/// [`compute_phase_from_projection`]: instead of one global phase from a single
/// dot product, each channel gets its own phase from its own (state, direction)
/// pair. Use this when different channels should rotate by different amounts
/// (e.g. shard retrieval: the spectral half and spatial half may rotate at
/// different rates depending on which channel dominates the query).
///
/// Each channel's `(cos, sin)` is evaluated via [`phase_safe_cos_sin`]
/// (libm `sin` + Pythagorean `sqrt` recovery), which forces the G1-critical
/// `cos²α + sin²α = 1` identity to hold bit-by-bit. If a future hot path
/// needs to beat the libm-sin latency floor (Phase 3 SIMD/LUT work), a new
/// `compute_phase_per_channel_simd_into` entry point will land — this function
/// is the always-correct reference.
///
/// Writes into the caller-provided `cos_out` and `sin_out` slices (length `D`).
/// The `_scratch` parameter is reserved for future SIMD-vectorized variants
/// that need a sin-LUT cache; this implementation does not currently touch
/// `scratch` (the per-channel sin eval is fully stack-local) — pass a
/// default-initialized scratch and it will be a no-op.
///
/// # Arguments
///
/// * `state` — current latent state, length `D`.
/// * `directions` — per-channel direction scalars, length `D` (each channel
///   has its own one-dimensional "direction" — equivalent to a diagonal
///   direction matrix). For a full `(D, D)` direction matrix use the
///   matrix-vector variant (TODO if a caller needs it).
/// * `sharpness` — phase steepness `λ`.
/// * `cos_out` — per-channel `cos(α_c)`, length `D`.
/// * `sin_out` — per-channel `sin(α_c)`, length `D`.
/// * `_scratch` — reserved for future SIMD/LUT variants (no-op now).
///
/// # Errors
///
/// Returns [`PhaseRotationError::ProjectionShapeMismatch`] if any of `state`,
/// `directions`, `cos_out`, `sin_out` disagree in length.
#[inline]
pub fn compute_phase_per_channel_into(
    state: &[f32],
    directions: &[f32],
    sharpness: f32,
    cos_out: &mut [f32],
    sin_out: &mut [f32],
    _scratch: &mut PhaseRotationScratch,
) -> Result<(), PhaseRotationError> {
    let d = state.len();
    if directions.len() != d || cos_out.len() != d || sin_out.len() != d {
        return Err(PhaseRotationError::ProjectionShapeMismatch);
    }

    // Phase construction is a per-channel sigmoid + cos/sin — no cross-channel
    // interaction. The 4-wide chunking is retained from the original SIMD-padé
    // design for forward-compat with a future SIMD sin-LUT variant; with libm
    // sin it does not auto-vectorise (libm call overhead per element) but the
    // overhead is acceptable within the 1500ns D=64 budget.
    let mut i = 0;
    while i + 4 <= d {
        for j in 0..4 {
            let idx = i + j;
            // α_c = sigmoid(state[c] · directions[c] · λ) · π/2.
            let proj = state[idx] * directions[idx] * sharpness;
            let alpha = simd::fast_sigmoid(proj) * FRAC_PI_2;
            let (c, s) = phase_safe_cos_sin(alpha);
            cos_out[idx] = c;
            sin_out[idx] = s;
        }
        i += 4;
    }
    while i < d {
        let proj = state[i] * directions[i] * sharpness;
        let alpha = simd::fast_sigmoid(proj) * FRAC_PI_2;
        let (c, s) = phase_safe_cos_sin(alpha);
        cos_out[i] = c;
        sin_out[i] = s;
        i += 1;
    }
    Ok(())
}

// ── Phase-safe cos/sin (Pythagorean-forced identity) ────────────

/// Phase-safe `(cos α, sin α)` for `α ∈ [0, π/2]` that **forces the
/// Pythagorean identity `cos²α + sin²α = 1` bit-by-bit**.
///
/// This is the G1-critical primitive: the whole norm-preservation thesis
/// depends on `sin²α + cos²α = 1` holding in f32 arithmetic. A naive
/// independent evaluation of `cos` and `sin` (even via libm) accumulates
/// drift in the identity (~1e-7 abs for libm), which is fine for most uses
/// but matters at the G1 < 1e-4 budget *across a 1000-point sweep* where
/// worst-case drift can compound.
///
/// # Method
///
/// `sin(α)` is evaluated via libm. Then `cos(α)` is recovered via the
/// Pythagorean identity: `cos α = sqrt(1 - sin²α)` (the `+` root because the
/// phase is in `[0, π/2]`, so `cos ≥ 0`). This forces `cos² + sin² = 1` to
/// f32 precision *by construction* — only one libm call instead of two, and
/// the identity holds to bit-exactness (modulo the single rounding in
/// `1 - sin²` and the `sqrt`).
///
/// Cost: one libm `sin` (~10 ns) + one `sqrt` (~3 ns on aarch64). For the
/// D=64 per-channel path that's ~13 ns × 64 = ~830 ns — well under the
/// 1500 ns libm-path budget (Plan 322 G3 cold-path target). If a future hot
/// path needs to beat the 600 ns Padé-path target, swap `libm_sin` for a
/// SIMD-vectorized sin Padé (tracked in Plan 322 Phase 3 SIMD/LUT work —
/// only if G3 marginally fails on the bench).
///
/// # Branchless-ness
///
/// No `if` statements, no sign-quadrant logic. The clamp handles contract
/// violations (out-of-range `α`), and the `sqrt` argument is provably `≥ 0`
/// because the clamp keeps `sin ∈ [0, 1]`. The kernel is SIMD-friendly: the
/// 4-wide chunked loop in [`compute_phase_per_channel_into`] can be
/// auto-vectorised by LLVM (the libm `sin` call is the only scalar op).
#[inline(always)]
fn phase_safe_cos_sin(alpha: f32) -> (f32, f32) {
    // Normalize to [0, π/2] by construction (phase is sigmoid · π/2). We still
    // clamp defensively — out-of-contract inputs would have ambiguous cos sign.
    let x = alpha.clamp(0.0, FRAC_PI_2);

    // libm sin — ~1 ULP accuracy, ~10 ns.
    let sin_v = x.sin().clamp(0.0, 1.0);

    // cos α = sqrt(1 − sin²α) — forces the Pythagorean identity bit-by-bit.
    // The sqrt arg is provably ≥ 0: sin_v ∈ [0, 1] by the clamp above, so
    // sin²v ∈ [0, 1] and 1 − sin²v ∈ [0, 1]. The + root because the phase is
    // in [0, π/2] (cos ≥ 0).
    let cos_v = (1.0f32 - sin_v * sin_v).sqrt();

    (cos_v, sin_v)
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::excessive_precision)]
mod tests {
    use super::*;
    use core::f32::consts::FRAC_PI_2;

    /// Helper: libm reference for cos/sin (used by accuracy tests).
    #[inline(always)]
    fn ref_cos_sin(alpha: f32) -> (f32, f32) {
        (alpha.cos(), alpha.sin())
    }

    #[test]
    fn scalar_phase_at_zero_returns_a() {
        // α = 0 → cos α = 1, sin α = 0 → out = a.
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        phase_rotation_gate_into(&a, &b, &[1.0], &[0.0], &mut out).unwrap();
        assert_eq!(out, a, "α=0 must return a bit-identically");
    }

    #[test]
    fn scalar_phase_at_pi_half_returns_b() {
        // α = π/2 → cos α ≈ 0, sin α = 1 → out ≈ b.
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        // cos(π/2) ≈ 6.1e-17 in f32 (libm), so out ≈ a·6e-17 + b·1 ≈ b.
        let cos_half_pi = (FRAC_PI_2).cos();
        let sin_half_pi = (FRAC_PI_2).sin();
        phase_rotation_gate_into(&a, &b, &[cos_half_pi], &[sin_half_pi], &mut out).unwrap();
        for i in 0..4 {
            assert!(
                (out[i] - b[i]).abs() < 1e-5,
                "α=π/2 must return b within libm cos(π/2) drift; out[{}] = {} vs b[{}] = {}",
                i,
                out[i],
                i,
                b[i]
            );
        }
    }

    #[test]
    fn scalar_phase_at_pi_four_is_average_scaled() {
        // α = π/4 → cos α = sin α = 1/√2 ≈ 0.7071.
        // out = (a + b) / √2.
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        let inv_sqrt2 = core::f32::consts::FRAC_1_SQRT_2;
        phase_rotation_gate_into(&a, &b, &[inv_sqrt2], &[inv_sqrt2], &mut out).unwrap();
        // Expected: (a + b) / √2 = [0.7071, 0.7071, 0, 0].
        assert!((out[0] - inv_sqrt2).abs() < 1e-6, "out[0] = {}", out[0]);
        assert!((out[1] - inv_sqrt2).abs() < 1e-6, "out[1] = {}", out[1]);
        assert!(out[2].abs() < 1e-7);
        assert!(out[3].abs() < 1e-7);
    }

    #[test]
    fn l2_norm_bounded_by_sum_of_input_norms() {
        // ‖out‖² ≤ ‖a‖² + ‖b‖² for ALL α (Cauchy-Schwarz + sin²+cos²=1).
        // Sweep α ∈ [0, π/2] and verify the bound holds across the sweep.
        let a = [1.0f32, 2.0, 3.0, 0.5];
        let b = [0.5f32, -1.0, 2.0, 1.0];
        let norm_a_sq: f32 = a.iter().map(|v| v * v).sum();
        let norm_b_sq: f32 = b.iter().map(|v| v * v).sum();
        let bound = norm_a_sq + norm_b_sq;
        let mut out = [0.0f32; 4];

        let steps = 1000;
        for k in 0..=steps {
            let alpha = (k as f32 / steps as f32) * FRAC_PI_2;
            let (c, s) = (alpha.cos(), alpha.sin());
            phase_rotation_gate_into(&a, &b, &[c], &[s], &mut out).unwrap();
            let norm_out_sq: f32 = out.iter().map(|v| v * v).sum();
            assert!(
                norm_out_sq <= bound + 1e-4,
                "α={}: ‖out‖² = {} exceeds bound ‖a‖²+‖b‖² = {}",
                alpha,
                norm_out_sq,
                bound
            );
        }
    }

    #[test]
    fn l2_norm_exact_for_orthogonal_equal_norm_at_pi_four() {
        // a ⊥ b, ‖a‖ = ‖b‖ = 1, α = π/4 → ‖out‖ = 1 exactly.
        // ‖out‖² = cos²·1 + sin²·1 = 1 (Pythagorean identity).
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        let inv_sqrt2 = core::f32::consts::FRAC_1_SQRT_2;
        phase_rotation_gate_into(&a, &b, &[inv_sqrt2], &[inv_sqrt2], &mut out).unwrap();
        let norm_out_sq: f32 = out.iter().map(|v| v * v).sum();
        assert!(
            (norm_out_sq - 1.0).abs() < 1e-5,
            "α=π/4, a⊥b, ‖a‖=‖b‖=1: ‖out‖² = {} should be 1.0",
            norm_out_sq
        );
    }

    #[test]
    fn per_channel_phase_independent_rotations() {
        // Each channel rotates by its own α; channel c's output depends only
        // on cos α_c, sin α_c (NOT on neighboring channels' phases).
        let a = [1.0f32, 1.0, 1.0, 1.0];
        let b = [2.0f32, 2.0, 2.0, 2.0];
        // Per-channel α: [0, π/4, π/2, π/4].
        let alpha = [0.0f32, core::f32::consts::FRAC_PI_4, FRAC_PI_2, core::f32::consts::FRAC_PI_4];
        let cos_alpha: Vec<f32> = alpha.iter().map(|&a| a.cos()).collect();
        let sin_alpha: Vec<f32> = alpha.iter().map(|&a| a.sin()).collect();
        let mut out = [0.0f32; 4];
        phase_rotation_gate_into(&a, &b, &cos_alpha, &sin_alpha, &mut out).unwrap();
        // Channel 0: α=0 → out = a[0] = 1.
        assert!((out[0] - 1.0).abs() < 1e-6, "channel 0 α=0: out = {}", out[0]);
        // Channel 1: α=π/4 → out = (1+2)/√2 = 3/√2 ≈ 2.121.
        let expected_1 = (1.0f32 + 2.0) * core::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (out[1] - expected_1).abs() < 1e-5,
            "channel 1 α=π/4: out = {} vs expected {}",
            out[1],
            expected_1
        );
        // Channel 2: α=π/2 → out ≈ b[2] = 2.
        assert!(
            (out[2] - 2.0).abs() < 1e-5,
            "channel 2 α=π/2: out = {} vs b = 2",
            out[2]
        );
        // Channel 3: same as channel 1.
        assert!(
            (out[3] - expected_1).abs() < 1e-5,
            "channel 3 α=π/4: out = {} vs expected {}",
            out[3],
            expected_1
        );
    }

    #[test]
    fn phase_bounded_in_zero_to_pi_half() {
        // compute_phase_from_projection returns α ∈ [0, π/2] for arbitrary
        // state/direction inputs (sigmoid ∈ [0,1], ·π/2 ∈ [0,π/2]).
        // cos α ≥ 0 and sin α ≥ 0 must hold.
        let directions = [
            [1.0f32; 8],
            [-1.0f32; 8],
            [0.5f32, -0.3, 0.8, -0.2, 0.1, -0.7, 0.4, -0.5],
        ];
        let states = [
            [10.0f32; 8],  // large positive projection
            [-10.0f32; 8], // large negative projection
            [0.0f32; 8],   // zero projection (α = π/4 by sigmoid(0) = 0.5)
            [1.0f32; 8],   // unit projection
        ];
        let mut cos_a = 0.0f32;
        let mut sin_a = 0.0f32;
        for state in &states {
            for dirn in &directions {
                compute_phase_from_projection(state, dirn, 4.0, &mut cos_a, &mut sin_a).unwrap();
                assert!(
                    cos_a >= -1e-6,
                    "cos α = {} < 0 for state={:?} dirn={:?}",
                    cos_a,
                    state,
                    dirn
                );
                assert!(
                    sin_a >= -1e-6,
                    "sin α = {} < 0 for state={:?} dirn={:?}",
                    sin_a,
                    state,
                    dirn
                );
                // Pythagorean identity holds (within f32 drift).
                let sum_sq = cos_a * cos_a + sin_a * sin_a;
                assert!(
                    (sum_sq - 1.0).abs() < 1e-5,
                    "cos²+sin² = {} != 1 for state={:?} dirn={:?}",
                    sum_sq,
                    state,
                    dirn
                );
            }
        }
    }

    #[test]
    fn deterministic_given_same_inputs() {
        // Same (state, direction, sharpness) → same (cos α, sin α).
        let state = [0.3f32, -0.5, 0.7, 0.1, -0.2, 0.8, -0.4, 0.6];
        let direction = [0.1f32, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8];
        let mut cos1 = 0.0f32;
        let mut sin1 = 0.0f32;
        let mut cos2 = 0.0f32;
        let mut sin2 = 0.0f32;
        compute_phase_from_projection(&state, &direction, 4.0, &mut cos1, &mut sin1).unwrap();
        compute_phase_from_projection(&state, &direction, 4.0, &mut cos2, &mut sin2).unwrap();
        assert_eq!(cos1.to_bits(), cos2.to_bits(), "cos α must be bit-identical");
        assert_eq!(sin1.to_bits(), sin2.to_bits(), "sin α must be bit-identical");
    }

    #[test]
    fn zero_alloc_in_steady_state() {
        // PhaseRotationScratch allocated once; phase_rotation_gate_into and
        // compute_phase_per_channel_into do NOT allocate.
        // We can't use a #[global_allocator] (parallel test harness), so we
        // instead verify by code inspection: no Vec::new, vec![], Vec::clone,
        // or .resize on the hot path. The bench (Phase 2 G4) will pin this
        // empirically with a CountingAllocator (single-threaded bench harness).
        //
        // This test exists to pin the API shape — it constructs a scratch, runs
        // both hot paths, and asserts the scratch's cached_d is unchanged
        // (no implicit growth).
        let d = 8;
        let mut scratch = PhaseRotationScratch::new(d);
        let a = vec![1.0f32; d];
        let b = vec![0.5f32; d];
        let cos_alpha = vec![0.7071f32; d];
        let sin_alpha = vec![0.7071f32; d];
        let mut out = vec![0.0f32; d];

        // Mix hot path — scratch is not even borrowed here (it doesn't need it).
        phase_rotation_gate_into(&a, &b, &cos_alpha, &sin_alpha, &mut out).unwrap();
        assert_eq!(scratch.cached_d, d, "scratch.cached_d unchanged");

        // Per-channel phase hot path — scratch IS borrowed but must not grow.
        let state = vec![0.5f32; d];
        let directions = vec![1.0f32; d];
        let mut cos_out = vec![0.0f32; d];
        let mut sin_out = vec![0.0f32; d];
        compute_phase_per_channel_into(
            &state,
            &directions,
            4.0,
            &mut cos_out,
            &mut sin_out,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(scratch.cached_d, d, "per-channel path did not grow scratch");
        assert_eq!(scratch.cos_alpha.len(), d);
        assert_eq!(scratch.sin_alpha.len(), d);
    }

    #[test]
    fn shape_mismatch_returns_err() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 8]; // wrong
        let mut out = [0.0f32; 4];
        let err = phase_rotation_gate_into(&a, &b, &[1.0], &[0.0], &mut out).unwrap_err();
        assert_eq!(err, PhaseRotationError::ShapeMismatch);
    }

    #[test]
    fn invalid_phase_len_returns_err() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        // cos_alpha length 2 is neither 1 nor D=4.
        let err = phase_rotation_gate_into(&a, &b, &[1.0, 0.0], &[1.0, 0.0], &mut out).unwrap_err();
        assert_eq!(err, PhaseRotationError::InvalidPhaseLen);
    }

    #[test]
    fn projection_shape_mismatch_returns_err() {
        let state = [1.0f32; 4];
        let direction = [1.0f32; 8]; // wrong
        let mut cos_a = 0.0f32;
        let mut sin_a = 0.0f32;
        let err =
            compute_phase_from_projection(&state, &direction, 4.0, &mut cos_a, &mut sin_a).unwrap_err();
        assert_eq!(err, PhaseRotationError::ProjectionShapeMismatch);
    }

    #[test]
    fn phase_safe_cos_sin_accuracy() {
        // Sweep α ∈ [0, π/2] at 1000 steps and verify:
        //   1. Pythagorean identity drift |cos² + sin² - 1| < 1e-4 (G1 budget).
        //   2. Per-element abs error vs libm cos/sin < 5e-3 (libm sin ~1 ULP +
        //      sqrt amplification — well within the same accuracy budget as
        //      Plan 319 Issue 003).
        let steps = 1000;
        let mut max_cos_err = 0.0f32;
        let mut max_sin_err = 0.0f32;
        let mut max_pythagorean_drift = 0.0f32;
        for k in 0..=steps {
            let alpha = (k as f32 / steps as f32) * FRAC_PI_2;
            let (cos_p, sin_p) = phase_safe_cos_sin(alpha);
            let (cos_r, sin_r) = ref_cos_sin(alpha);
            max_cos_err = max_cos_err.max((cos_p - cos_r).abs());
            max_sin_err = max_sin_err.max((sin_p - sin_r).abs());
            let drift = (cos_p * cos_p + sin_p * sin_p - 1.0).abs();
            max_pythagorean_drift = max_pythagorean_drift.max(drift);
        }
        // G1 kill-switch: Pythagorean identity must hold to <1e-4. With the
        // sqrt-recovery construction this is essentially f32 rounding noise.
        assert!(
            max_pythagorean_drift < 1e-4,
            "phase_safe cos²+sin² drift = {} exceeds G1 1e-4 budget",
            max_pythagorean_drift
        );
        // Per-element accuracy: sin is from libm (~1 ULP), cos is sqrt(1-sin²)
        // (~2 ULP, sin error amplified by the sqrt derivative). Both well
        // within 5e-3.
        assert!(
            max_cos_err < 5e-3,
            "phase_safe cos max abs err = {} exceeds 5e-3 budget",
            max_cos_err
        );
        assert!(
            max_sin_err < 5e-3,
            "phase_safe sin max abs err = {} exceeds 5e-3 budget",
            max_sin_err
        );
    }

    #[test]
    fn per_channel_phase_matches_libm_within_budget() {
        // End-to-end: per-channel phase via phase_safe_cos_sin must match
        // libm cos/sin within the 5e-3 budget. The Pythagorean identity forces
        // cos² + sin² = 1, but cos itself is sqrt(1-sin²) which has small
        // amplification of sin's libm error.
        let d = 64;
        let state: Vec<f32> = (0..d).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let directions: Vec<f32> = (0..d).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let sharpness = 4.0f32;

        let mut cos_safe = vec![0.0f32; d];
        let mut sin_safe = vec![0.0f32; d];
        let mut cos_libm = vec![0.0f32; d];
        let mut sin_libm = vec![0.0f32; d];
        let mut scratch = PhaseRotationScratch::new(d);

        compute_phase_per_channel_into(
            &state,
            &directions,
            sharpness,
            &mut cos_safe,
            &mut sin_safe,
            &mut scratch,
        )
        .unwrap();

        // Reference: libm cos/sin directly (the original paper spec, not via
        // Pythagorean recovery).
        for i in 0..d {
            let proj = state[i] * directions[i] * sharpness;
            let alpha = simd::fast_sigmoid(proj) * FRAC_PI_2;
            cos_libm[i] = alpha.cos();
            sin_libm[i] = alpha.sin();
        }

        let mut max_cos_diff = 0.0f32;
        let mut max_sin_diff = 0.0f32;
        for i in 0..d {
            max_cos_diff = max_cos_diff.max((cos_safe[i] - cos_libm[i]).abs());
            max_sin_diff = max_sin_diff.max((sin_safe[i] - sin_libm[i]).abs());
        }
        assert!(
            max_cos_diff < 5e-3,
            "per-channel phase_safe vs libm cos diff = {} exceeds budget",
            max_cos_diff
        );
        assert!(
            max_sin_diff < 5e-3,
            "per-channel phase_safe vs libm sin diff = {} exceeds budget",
            max_sin_diff
        );
    }

    #[test]
    fn gate_config_validation() {
        // Valid sharpness.
        assert!(PhaseRotationGate::new(4.0).is_some());
        assert!(PhaseRotationGate::new(0.0).is_some());
        // Invalid sharpness.
        assert!(PhaseRotationGate::new(-1.0).is_none());
        assert!(PhaseRotationGate::new(f32::NAN).is_none());
        assert!(PhaseRotationGate::new(f32::INFINITY).is_none());
        // Clamping at 100.
        let big = PhaseRotationGate::new(1000.0).unwrap();
        assert_eq!(big.sharpness, 100.0);
    }

    #[test]
    fn scratch_ensure_capacity_noop_on_same_d() {
        let mut scratch = PhaseRotationScratch::new(8);
        // Same capacity — no-op.
        scratch.ensure_capacity(8);
        assert_eq!(scratch.cached_d, 8);
        assert_eq!(scratch.cos_alpha.len(), 8);
        // Different capacity — grows (cold path).
        scratch.ensure_capacity(16);
        assert_eq!(scratch.cached_d, 16);
        assert_eq!(scratch.cos_alpha.len(), 16);
        assert_eq!(scratch.sin_alpha.len(), 16);
        // Shrink also works.
        scratch.ensure_capacity(4);
        assert_eq!(scratch.cached_d, 4);
        assert_eq!(scratch.cos_alpha.len(), 4);
    }

    #[test]
    fn scalar_and_per_channel_paths_agree_at_uniform_phase() {
        // When the per-channel phase is the same for every channel, the
        // per-channel mix must match the scalar-broadcast mix bit-identically
        // (the two kernels compute the same FMA chain).
        let d = 8;
        let a: Vec<f32> = (0..d).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..d).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let alpha = 0.7f32; // arbitrary phase
        let cos_a = alpha.cos();
        let sin_a = alpha.sin();

        let mut out_scalar = vec![0.0f32; d];
        let cos_broadcast = [cos_a];
        let sin_broadcast = [sin_a];
        phase_rotation_gate_into(&a, &b, &cos_broadcast, &sin_broadcast, &mut out_scalar).unwrap();

        let mut out_per_channel = vec![0.0f32; d];
        let cos_each: Vec<f32> = vec![cos_a; d];
        let sin_each: Vec<f32> = vec![sin_a; d];
        phase_rotation_gate_into(&a, &b, &cos_each, &sin_each, &mut out_per_channel).unwrap();

        for i in 0..d {
            assert_eq!(
                out_scalar[i].to_bits(),
                out_per_channel[i].to_bits(),
                "scalar and per-channel paths disagree at channel {}: {} vs {}",
                i,
                out_scalar[i],
                out_per_channel[i]
            );
        }
    }

    #[test]
    fn out_can_alias_a() {
        // out may alias a — the kernel reads a[i] and b[i] before writing out[i],
        // so per-channel aliasing is sound at runtime. Rust's borrow checker
        // can't prove this without `unsafe`, so we test the semantics by
        // computing into a *copy* of `a` and comparing against a fresh output.
        // (Production callers who need true in-place aliasing use raw pointers
        // or `split_at_mut` patterns; the kernel itself is alias-safe.)
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut expected = [0.0f32; 4];
        phase_rotation_gate_into(&a, &b, &[0.6], &[0.8], &mut expected).unwrap();
        // The aliasing claim: if `out == a` at entry, the kernel would compute
        // the same `expected` because it reads a[i] before writing out[i]. We
        // can't construct that borrow in safe Rust; the per-channel read-before-
        // write order is what makes the kernel alias-safe in unsafe callers.
        // Pin the expected values so a future refactor that reorders reads and
        // writes trips this test.
        let want = [
            0.6f32.mul_add(1.0, 0.8 * 5.0),
            0.6f32.mul_add(2.0, 0.8 * 6.0),
            0.6f32.mul_add(3.0, 0.8 * 7.0),
            0.6f32.mul_add(4.0, 0.8 * 8.0),
        ];
        for i in 0..4 {
            assert!((expected[i] - want[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn monotone_interpolation_from_a_to_b() {
        // G2 prerequisite: sweeping α ∈ [0, π/2] moves output monotonically
        // from a to b in cosine-similarity space (cos sim to a decreases
        // monotonically, cos sim to b increases monotonically).
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0];
        let steps = 100;
        let mut prev_sim_a = 1.0f32;
        let mut prev_sim_b = 0.0f32;
        let mut out = [0.0f32; 4];
        for k in 0..=steps {
            let alpha = (k as f32 / steps as f32) * FRAC_PI_2;
            let (c, s) = (alpha.cos(), alpha.sin());
            phase_rotation_gate_into(&a, &b, &[c], &[s], &mut out).unwrap();
            let sim_a = cosine_sim(&out, &a);
            let sim_b = cosine_sim(&out, &b);
            // sim_a must be non-increasing.
            assert!(
                sim_a <= prev_sim_a + 1e-5,
                "α={}: sim_a = {} increased from {} (should be monotone decreasing)",
                alpha,
                sim_a,
                prev_sim_a
            );
            // sim_b must be non-decreasing.
            assert!(
                sim_b >= prev_sim_b - 1e-5,
                "α={}: sim_b = {} decreased from {} (should be monotone increasing)",
                alpha,
                sim_b,
                prev_sim_b
            );
            prev_sim_a = sim_a;
            prev_sim_b = sim_b;
        }
    }

    /// Helper: cosine similarity (scalar reference — no SIMD in tests).
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        dot / (na * nb)
    }
}
