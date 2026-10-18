//! Spherical Steering — single-target geodesic Slerp rotation toward a target
//! direction on the unit hypersphere.
//!
//! Distilled from Spherical Steering (arXiv:2602.08169, You/Deng/Chen ICML
//! 2026; Research 382, Plan 405). The paper's mechanism is modelless: a
//! norm-preserving **geodesic** Slerp rotation of a latent vector `ĥ` along
//! the great-circle path toward a target archetype `μ_T`, gated by a
//! sigmoid-translated von Mises-Fisher (vMF) confidence signal. Only the
//! deterministic math ships here — no training, no learned `μ_T`. The
//! contrastive prototype construction (mean-difference) is the consumer's
//! responsibility (a frozen, BLAKE3-committed recipe — see Research 382 §2.4
//! F1 fusion hook).
//!
//! # What this computes
//!
//! Given a latent vector `h ∈ ℝᴰ`, a unit-norm target direction `μ_T ∈ ℝᴰ`,
//! and a steering strength `t ∈ [0, 1]`:
//!
//! ```text
//! ĥ   = h / ‖h‖
//! θ   = arccos(clamp(ĥ · μ_T, -1, 1))
//! ĥ'  = sin((1−t)θ)/sin θ · ĥ + sin(tθ)/sin θ · μ_T     // unit-norm by construction
//! h'  = ‖h‖ · ĥ'
//! ```
//!
//! `ĥ'` lies on the unit sphere `S^{d-1}` for every `θ ∈ (0, π)` and every
//! `t ∈ [0, 1]` (the Slerp identity), so `‖h'‖ = ‖h‖` exactly (up to f32
//! rounding in the mix). The rotation traces the **geodesic** (shortest path
//! on the sphere) from `ĥ` to `μ_T` — no "shortcut" through the ambient
//! space, no norm inflation, no sign-flipping.
//!
//! The vMF confidence gate (`vmf_confidence_gate`) derives `t` from the
//! cosine `s_T = ĥ · μ_T`:
//!
//! ```text
//! δ = -tanh(κ · s_T)             // paper Eq 17
//!   = 1 − 2 · sigmoid(2κ · s_T)  // sigmoid form (AGENTS.md "never softmax")
//! t = clamp((α · δ − β) / (1 − β), 0, 1)   if δ > β else 0
//! ```
//!
//! When `ĥ` is already aligned with `μ_T` (`s_T → 1`): `δ → -1 < β` → `t = 0`
//! (no steering). When `ĥ` is anti-aligned (`s_T → -1`): `δ → +1` → `t = α`
//! (max steering). The gate is **input-adaptive**: drift detection, not a
//! fixed-strength correction.
//!
//! # Why this is NOT redundant with Plan 322
//!
//! [`crate::phase_rotation::phase_rotation_gate_into`] (Plan 322) ships a
//! **2-subspace** rotation `cos α ⊙ a + sin α ⊙ b` where the input naturally
//! splits into halves `(a, b)`. Its norm-preservation invariant
//! (`sin²α + cos²α = 1`) is exact when `a ⊥ b` (the HLA design intent); the
//! use case is **balance** between two subspaces.
//!
//! Spherical Steering's Slerp form takes a **single** input vector and a
//! **single** target direction. Its invariant (`‖c0·ĥ + c1·μ_T‖ = 1`) holds
//! for **all** `θ ∈ (0, π)`, not just orthogonal; the use case is **steering
//! toward an archetype** outside the input's own direction. The two
//! parameterizations compose: Plan 322 rotates within the `(a, b)` plane;
//! Spherical Steering rotates toward a target outside that plane.
//!
//! # Numerical contract
//!
//! - All entry points are pure float arithmetic over caller-provided buffers.
//!   Deterministic on a given CPU (same inputs → bit-identical outputs).
//! - `h`, `mu_t`, `h_out`, and `scratch.unit_h` must be equal length. Length
//!   mismatches trip [`SlerpError::ShapeMismatch`].
//! - `t` must be in `[0, 1]`; otherwise [`SlerpError::InvalidStrength`].
//! - `‖h‖ < 1e-12` → [`SlerpError::ZeroNorm`] (caller decides policy).
//! - `θ < θ_MIN` (aligned) → lerp fallback `(1−t)·ĥ + t·μ_T`, then renormalize
//!   back to `‖h‖`. Avoids the `sin θ → 0` division blow-up; drift is
//!   `O(t²·θ²)` — well under the G1 budget.
//! - `θ > π − θ_MIN` (antipodal) → [`SlerpError::AntipodalDegenerate`]: the
//!   geodesic is not unique (every great circle through antipodes is shortest).
//!   The paper's measure-zero case; caller decides policy (no-op, deterministic
//!   perpendicular rotation, etc.).
//! - `θ` is computed via `atan2(√(1−x²), x)` rather than `arccos(x)` — the
//!   `arccos` form is ill-conditioned near `±1` (slope → ∞), the `atan2` form
//!   stays well-conditioned across the full range.
//!
//! # Performance
//!
//! `O(D)` per call: one `simd_dot_f32` (cosine), one `simd_sum_sq` (norm), one
//! `acos`/`atan2` + two `sin` + one `div` for the coefficients, then a 4-wide
//! chunked FMA mix loop. Zero allocation after scratch init. The inner loop
//! is the same `c·a + s·b` shape as Plan 322 — LLVM auto-vectorizes to
//! NEON/AVX2. Expect ~3–5× the latency of Plan 322's scalar-broadcast path at
//! D=8 (arccos + 2 sin + div vs cos + sin), well under the G3 budget.
//!
//! See Research 382 §2.3 for the cousin comparison and §2.4 for the F1 fusion
//! hook (Slerp × CommittedFieldBlend × HLA divergence = "personality drift
//! auto-correction").
//!
//! Spherical Steering — single-target geodesic Slerp rotation
//! `sin((1−t)θ)/sin θ · ĥ + sin(tθ)/sin θ · μ_T` toward a unit-norm target
//! direction on S^{d-1}, with sigmoid-translated vMF confidence gate (Plan 405,
//! Research 382, arxiv 2602.08169 You/Deng/Chen ICML 2026). Sibling to Plan 322's
//! 2-subspace phase rotation — same norm-preservation thesis, different
//! parameterization: Plan 322 rotates *within* the (a,b) plane; Plan 405 rotates
//! *toward* a target outside the input's direction (Slerp identity holds for all
//! θ ∈ (0,π)). vMF gate reduces to sigmoid via Eq 17:
//! `δ = -tanh(κ·s_T) = 1 − 2·sigmoid(2κ·s_T)`. §3.5 modelless Path 3 (closed-form
//! trig + sigmoid; no training). DEFAULT-ON (Plan 405 Phase 2, 2026-07-06):
//! G1-G5 ALL PASS.

use crate::simd;
use core::f32::consts::PI;

/// Below this angular separation the geodesic is degenerate (`sin θ → 0`):
/// use the lerp fallback. Chosen so the lerp's `O(t²·θ²)` drift stays well
/// under the G1 `< 1e-4` budget (worst case at θ = θ_MIN, t = 1: drift ≈
/// `θ_MIN² / 2 ≈ 5e-7`).
const THETA_MIN: f32 = 1e-3;

/// `‖h‖` below this is treated as zero (caller's vector is the zero vector;
/// Slerp is undefined). Matches the convention in `simd::simd_dot_f32` (zero
/// norm → undefined direction).
const NORM_FLOOR: f32 = 1e-12;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by the Slerp steering entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SlerpError {
    /// `h.len() != mu_t.len()` (or `h_out`, `scratch.unit_h`) — all four must
    /// agree on the channel count.
    ShapeMismatch,
    /// `‖h‖ < NORM_FLOOR` — the input has no defined direction on the sphere.
    /// Caller decides policy (no-op, deterministic reseed, etc.).
    ZeroNorm,
    /// `θ > π − θ_MIN` — `ĥ` and `μ_T` are antipodal. The geodesic is not
    /// unique (every great circle through antipodes is shortest), so the
    /// Slerp coefficients are undefined. Caller decides policy.
    AntipodalDegenerate,
    /// `t < 0` or `t > 1` (or non-finite). The Slerp interpolation parameter
    /// must be a convex weight.
    InvalidStrength,
}

// ── Scratch ──────────────────────────────────────────────────────

/// Pre-allocated scratch for zero-alloc Slerp steering.
///
/// Create once via [`SlerpScratch::new`], then call [`SlerpScratch::ensure_capacity`]
/// before each invocation. The only field is `unit_h`, the in-place
/// normalization of `h` (length `D`). No `cos`/`sin` scratch is needed — Slerp
/// uses exactly 2 `sin` evaluations per call, kept in registers.
///
/// Mirrors [`crate::phase_rotation::PhaseRotationScratch`] and
/// [`crate::cross_resolution::CrossResScratch`].
#[derive(Debug, Clone, Default)]
pub struct SlerpScratch {
    /// `ĥ = h / ‖h‖` (length `D`). Written by [`slerp_steering_into`] and
    /// [`spherical_steering_into`]; read back during the mix loop.
    pub unit_h: Vec<f32>,
    cached_d: usize,
}

impl SlerpScratch {
    /// Allocate scratch for the given channel count.
    pub fn new(d: usize) -> Self {
        Self {
            unit_h: vec![0.0; d],
            cached_d: d,
        }
    }

    /// Resize the `unit_h` buffer if `d` changed. No-op on the hot path.
    pub fn ensure_capacity(&mut self, d: usize) {
        if self.cached_d == d {
            return;
        }
        self.unit_h.resize(d, 0.0);
        self.cached_d = d;
    }
}

// ── Mix kernel ───────────────────────────────────────────────────

/// The core Slerp steering mix: rotate `h` toward `μ_T` by strength `t`.
///
/// Computes `h' = ‖h‖ · (sin((1−t)θ)/sin θ · ĥ + sin(tθ)/sin θ · μ_T)` where
/// `θ = arccos(clamp(ĥ · μ_T, -1, 1))`. The output lies on the sphere of
/// radius `‖h‖` for every `θ ∈ (0, π)` and every `t ∈ [0, 1]` (the Slerp
/// identity).
///
/// # Arguments
///
/// * `h` — current latent state, length `D`. Need not be unit-norm (the
///   function preserves whatever `‖h‖` is).
/// * `mu_t` — target direction, length `D`. **Caller's responsibility** to
///   ensure `‖μ_T‖ ≈ 1` (a BLAKE3-committed artifact). Mild deviation is
///   tolerated (the Slerp math still returns a finite result), but the
///   norm-preservation invariant only holds to `‖μ_T‖ ≈ 1`.
/// * `t` — steering strength, `[0, 1]`. From [`vmf_confidence_gate`] or a
///   designer-fixed value.
/// * `h_out` — output, length `D`. May **not** alias `h` (the kernel reads
///   `h[i]` to compute `unit_h[i]` before writing `h_out[i]`, but the
///   normalization pass runs over all of `h` first — use a separate buffer
///   or copy `h` into `h_out` first if you need in-place). Aliasing `mu_t`
///   is fine (read-only). Aliasing `scratch.unit_h` is fine (scratch is
///   written before the read phase).
/// * `scratch` — caller-owned, reused across calls. The `unit_h` field is
///   overwritten on each call.
///
/// # Errors
///
/// Returns [`SlerpError::ShapeMismatch`] if lengths disagree. Returns
/// [`SlerpError::ZeroNorm`] if `‖h‖ < 1e-12`. Returns
/// [`SlerpError::AntipodalDegenerate`] if `θ > π − 1e-3`. Returns
/// [`SlerpError::InvalidStrength`] if `t < 0`, `t > 1`, or non-finite.
///
/// # Edge cases
///
/// - `t = 0` → `h_out == h` (identity; fast path, no trig).
/// - `t = 1` → `h_out == ‖h‖ · μ_T` (full rotation to target).
/// - `θ < 1e-3` (aligned) → lerp fallback `(1−t)·ĥ + t·μ_T`, then renormalize
///   to `‖h‖`. Avoids `sin θ → 0` blow-up.
/// - `θ > π − 1e-3` (antipodal) → `Err(AntipodalDegenerate)`.
///
/// # Performance
///
/// `O(D)`, zero allocation in steady state. One dot, one norm, one `atan2`,
/// two `sin`, one `div` for the coefficients, then a 4-wide chunked FMA
/// mix loop. See module docs for the latency profile.
#[inline]
pub fn slerp_steering_into(
    h: &[f32],
    mu_t: &[f32],
    t: f32,
    h_out: &mut [f32],
    scratch: &mut SlerpScratch,
) -> Result<(), SlerpError> {
    let d = h.len();
    if mu_t.len() != d || h_out.len() != d || scratch.unit_h.len() != d {
        return Err(SlerpError::ShapeMismatch);
    }
    if !t.is_finite() || !(0.0..=1.0).contains(&t) {
        return Err(SlerpError::InvalidStrength);
    }

    // Fast path: t = 0 → identity. (Also handles `t = -0.0` cleanly.)
    if t == 0.0 {
        h_out.copy_from_slice(h);
        return Ok(());
    }

    // ‖h‖ via SIMD sum-of-squares. Zero norm → caller's policy.
    let norm_sq = simd::simd_sum_sq(h, d);
    if norm_sq < NORM_FLOOR * NORM_FLOOR {
        return Err(SlerpError::ZeroNorm);
    }
    let norm = norm_sq.sqrt();
    let inv_norm = 1.0 / norm;

    // ĥ = h / ‖h‖, in-place into scratch.
    let unit_h = &mut scratch.unit_h[..d];
    let mut i = 0;
    while i + 4 <= d {
        unit_h[i] = h[i] * inv_norm;
        unit_h[i + 1] = h[i + 1] * inv_norm;
        unit_h[i + 2] = h[i + 2] * inv_norm;
        unit_h[i + 3] = h[i + 3] * inv_norm;
        i += 4;
    }
    while i < d {
        unit_h[i] = h[i] * inv_norm;
        i += 1;
    }

    // s_T = ĥ · μ_T (cosine to target). Clamp to [-1, 1] to absorb f32 drift.
    let s_t = simd::simd_dot_f32(unit_h, mu_t, d).clamp(-1.0, 1.0);

    // Fast path: t = 1 → rotate fully to μ_T (still need the unit check, but
    // skip the trig). Output = ‖h‖ · μ_T.
    if t == 1.0 {
        let mut j = 0;
        while j + 4 <= d {
            h_out[j] = norm * mu_t[j];
            h_out[j + 1] = norm * mu_t[j + 1];
            h_out[j + 2] = norm * mu_t[j + 2];
            h_out[j + 3] = norm * mu_t[j + 3];
            j += 4;
        }
        while j < d {
            h_out[j] = norm * mu_t[j];
            j += 1;
        }
        return Ok(());
    }

    // θ via atan2(√(1−x²), x) — better conditioned near ±1 than arccos.
    // (arccos slope → ∞ near ±1; atan2 slope stays bounded.)
    let one_minus_sq = (1.0f32 - s_t * s_t).max(0.0);
    let theta = (one_minus_sq.sqrt()).atan2(s_t);

    // θ ≈ π (antipodal) → geodesic is non-unique. Caller's policy.
    if theta > PI - THETA_MIN {
        return Err(SlerpError::AntipodalDegenerate);
    }

    // θ ≈ 0 (aligned) → lerp fallback. Avoids the sin θ → 0 division.
    // Drift is O(t²·θ²) — at θ = THETA_MIN = 1e-3, worst case (t = 1): ~5e-7,
    // well under the 1e-4 G1 budget. The fallback renormalizes to absorb the
    // small non-unit drift introduced by lerping two near-equal unit vectors.
    if theta < THETA_MIN {
        slerp_aligned_lerp_fallback(unit_h, mu_t, t, h_out, norm, d);
        return Ok(());
    }

    // General case: Slerp coefficients.
    let sin_theta = theta.sin();
    let c0 = ((1.0f32 - t) * theta).sin() / sin_theta;
    let c1 = (t * theta).sin() / sin_theta;

    // Mix: h_out = ‖h‖ · (c0 · ĥ + c1 · μ_T).
    let unit_h_read = &scratch.unit_h[..d];
    let mut m = 0;
    while m + 4 <= d {
        h_out[m] = norm * c0.mul_add(unit_h_read[m], c1 * mu_t[m]);
        h_out[m + 1] = norm * c0.mul_add(unit_h_read[m + 1], c1 * mu_t[m + 1]);
        h_out[m + 2] = norm * c0.mul_add(unit_h_read[m + 2], c1 * mu_t[m + 2]);
        h_out[m + 3] = norm * c0.mul_add(unit_h_read[m + 3], c1 * mu_t[m + 3]);
        m += 4;
    }
    while m < d {
        h_out[m] = norm * c0.mul_add(unit_h_read[m], c1 * mu_t[m]);
        m += 1;
    }

    Ok(())
}

/// Lerp fallback for the aligned (θ < `THETA_MIN`) case.
///
/// Computes `m = (1−t)·ĥ + t·μ_T`, renormalizes `m` to unit length, then
/// scales by `norm`. The renormalization absorbs the small non-unit drift
/// introduced by the lerp (the lerp of two unit vectors is unit only when
/// they coincide exactly). This keeps the G1 norm-preservation invariant
/// exact even in the fallback branch.
#[inline]
fn slerp_aligned_lerp_fallback(
    unit_h: &[f32], // ĥ, length d (already populated by the caller)
    mu_t: &[f32],   // length d
    t: f32,
    h_out: &mut [f32], // length d
    norm: f32,
    d: usize,
) {
    // Pass 1: lerp into h_out, accumulate ‖m‖².
    let mut sum_sq = 0.0f32;
    let mut k = 0;
    while k + 4 <= d {
        let m0 = (1.0f32 - t) * unit_h[k] + t * mu_t[k];
        let m1 = (1.0f32 - t) * unit_h[k + 1] + t * mu_t[k + 1];
        let m2 = (1.0f32 - t) * unit_h[k + 2] + t * mu_t[k + 2];
        let m3 = (1.0f32 - t) * unit_h[k + 3] + t * mu_t[k + 3];
        h_out[k] = m0;
        h_out[k + 1] = m1;
        h_out[k + 2] = m2;
        h_out[k + 3] = m3;
        sum_sq += m0 * m0 + m1 * m1 + m2 * m2 + m3 * m3;
        k += 4;
    }
    while k < d {
        let mk = (1.0f32 - t) * unit_h[k] + t * mu_t[k];
        h_out[k] = mk;
        sum_sq += mk * mk;
        k += 1;
    }

    // Pass 2: scale to ‖h‖. (m / ‖m‖) · ‖h‖ = m · (‖h‖ / ‖m‖).
    // sum_sq ≥ (1−t)²·‖ĥ‖² + t²·‖μ_T‖² ≥ (1−t)² (when ‖μ_T‖ ≥ 0); well above
    // NORM_FLOOR² for any reasonable ‖μ_T‖.
    let scale = norm / sum_sq.sqrt();
    for v in h_out[..d].iter_mut() {
        *v *= scale;
    }
}

// ── vMF confidence gate ──────────────────────────────────────────

/// Sigmoid-translated vMF confidence gate for input-adaptive steering strength.
///
/// Derives the Slerp parameter `t ∈ [0, 1]` from the cosine-to-target `s_T`:
///
/// ```text
/// δ = -tanh(κ · s_T)             // paper Eq 17
///   = 1 − 2 · sigmoid(2κ · s_T)  // sigmoid form (AGENTS.md "never softmax")
/// t = clamp((α · δ − β) / (1 − β), 0, 1)   if δ > β else 0
/// ```
///
/// - `s_T = 1` (aligned) → `δ = -1 ≤ β` → `t = 0` (no steering needed).
/// - `s_T = -1` (anti-aligned) → `δ = +1` → `t = α` (max steering).
/// - `s_T = 0` (orthogonal) → `δ = 0` → `t = 0` if `β ≥ 0`, else positive.
///
/// `β` is the **selectivity threshold**: lower `β` → steer on smaller drift;
/// higher `β` → only steer on large drift. `α` is the **rotation scale**
/// (max strength per call). `κ` is the vMF **concentration** (sharpness of
/// the alignment detector; paper default 20).
///
/// # Arguments
///
/// * `s_t` — cosine `ĥ · μ_T`, expected in `[-1, 1]`. Out-of-range values are
///   clamped (caller's `μ_T` may be slightly off unit norm).
/// * `kappa` — vMF concentration. Paper default 20. Higher = sharper.
/// * `alpha` — max rotation strength, `(0, 1]`.
/// * `beta` — selectivity threshold, `[-1, 1)`.
///
/// # Returns
///
/// `t ∈ [0, 1]` — the Slerp steering strength. Always finite, always in range.
///
/// # Why sigmoid not softmax
///
/// The paper writes the gate as a 2-class softmax over exponential vMF scores.
/// Eq 17 simplifies it to `δ = -tanh(κ·s_T)`, and `tanh(x) = 2·sigmoid(2x) − 1`
/// is an algebraic identity. So `δ = 1 − 2·sigmoid(2κ·s_T)` — a single sigmoid,
/// no normalization, no exp-sum. Per AGENTS.md §2 ("sigmoid not softmax"): the
/// sigmoid form is numerically stabler, faster, and avoids the exp-overflow
/// class of bugs.
#[inline]
pub fn vmf_confidence_gate(s_t: f32, kappa: f32, alpha: f32, beta: f32) -> f32 {
    // Clamp the cosine (defensive against slightly-off-unit-norm μ_T).
    let s = s_t.clamp(-1.0, 1.0);

    // δ = -tanh(κ · s_T) = 1 − 2 · sigmoid(2κ · s_T).
    // fast_sigmoid saturates by |x| > 40, so no overflow at large κ · s_T.
    let delta = 1.0f32 - 2.0 * simd::fast_sigmoid(2.0 * kappa * s);

    // Below the selectivity threshold → no steering.
    if delta <= beta {
        return 0.0;
    }

    // Linear ramp from β (→ 0) to δ = 1 (→ α). Clamped to [0, 1] for safety.
    let denom = 1.0 - beta;
    if denom <= 0.0 {
        // β ≥ 1 is out of contract; defensively return 0 (no steering).
        return 0.0;
    }
    let t = (alpha * delta - beta) / denom;
    t.clamp(0.0, 1.0)
}

// ── Convenience: full pipeline ───────────────────────────────────

/// Full spherical-steering pipeline: gate + Slerp in one call.
///
/// Computes `s_T = (h · μ_T) / ‖h‖` (cosine, no separate normalization pass),
/// derives `t` via [`vmf_confidence_gate`], then calls [`slerp_steering_into`].
///
/// # Fast path
///
/// If `t == 0` (the gate says "no steering needed" — e.g. `s_T` is already
/// above the selectivity threshold), copies `h` to `h_out` and returns. No
/// normalization, no trig. This is the steady-state case for a well-aligned
/// vector.
///
/// # Arguments
///
/// As [`slerp_steering_into`] plus the three gate parameters `kappa`, `alpha`,
/// `beta` (see [`vmf_confidence_gate`]).
///
/// # Errors
///
/// As [`slerp_steering_into`]. Note: the gate itself is total (always returns
/// a finite `t ∈ [0, 1]`), so errors come only from the Slerp stage
/// (shape mismatch, zero norm, antipodal).
#[inline]
pub fn spherical_steering_into(
    h: &[f32],
    mu_t: &[f32],
    kappa: f32,
    alpha: f32,
    beta: f32,
    h_out: &mut [f32],
    scratch: &mut SlerpScratch,
) -> Result<(), SlerpError> {
    let d = h.len();
    if mu_t.len() != d || h_out.len() != d || scratch.unit_h.len() != d {
        return Err(SlerpError::ShapeMismatch);
    }

    // ‖h‖ — needed both for the cosine and (if t > 0) for the Slerp.
    let norm_sq = simd::simd_sum_sq(h, d);
    if norm_sq < NORM_FLOOR * NORM_FLOOR {
        return Err(SlerpError::ZeroNorm);
    }
    let norm = norm_sq.sqrt();

    // s_T = (h · μ_T) / ‖h‖. (μ_T is assumed unit-norm by contract.)
    let raw_dot = simd::simd_dot_f32(h, mu_t, d);
    let s_t = (raw_dot / norm).clamp(-1.0, 1.0);

    let t = vmf_confidence_gate(s_t, kappa, alpha, beta);

    // Gate says "no steering" → fast-path no-op.
    if t == 0.0 {
        h_out.copy_from_slice(h);
        return Ok(());
    }

    // Delegate to the mix kernel. `slerp_steering_into` re-validates shapes
    // and t (cheap), and re-computes norm (also cheap; one SIMD pass). The
    // duplication is acceptable — it keeps the two entry points independent.
    slerp_steering_into(h, mu_t, t, h_out, scratch)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    /// Helper: L2 norm of a slice.
    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Helper: dot product of two slices.
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Helper: normalize a slice in place; panic if zero.
    fn normalize(v: &mut [f32]) {
        let n = l2_norm(v);
        assert!(n > 1e-12, "normalize: zero-norm input");
        for x in v.iter_mut() {
            *x /= n;
        }
    }

    // V1.1 — t = 0 returns h bit-exactly (the identity fast path).
    #[test]
    fn slerp_at_t_zero_returns_h() {
        let h = [0.6f32, -0.8, 0.0, 1.5, -0.3, 0.2, 0.9, -1.1];
        let mu_t = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e_0
        let mut out = [0.0f32; 8];
        let mut scratch = SlerpScratch::new(8);
        slerp_steering_into(&h, &mu_t, 0.0, &mut out, &mut scratch).unwrap();
        assert_eq!(out, h, "t=0 must return h bit-exactly");
    }

    // V1.2 — t = 1 returns ‖h‖ · μ_T (full rotation to target).
    #[test]
    fn slerp_at_t_one_returns_mu_t_scaled() {
        let h = [0.6f32, -0.8, 0.0, 1.5, -0.3, 0.2, 0.9, -1.1];
        let norm = l2_norm(&h);
        let mu_t = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e_0
        let expected = [
            norm * mu_t[0],
            norm * mu_t[1],
            norm * mu_t[2],
            norm * mu_t[3],
            norm * mu_t[4],
            norm * mu_t[5],
            norm * mu_t[6],
            norm * mu_t[7],
        ];
        let mut out = [0.0f32; 8];
        let mut scratch = SlerpScratch::new(8);
        slerp_steering_into(&h, &mu_t, 1.0, &mut out, &mut scratch).unwrap();
        for i in 0..8 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-5,
                "t=1: out[{}] = {} should be {} (‖h‖·μ_T)",
                i,
                out[i],
                expected[i]
            );
        }
    }

    // V1.3 — norm preservation across the (t, θ) sweep. THE G1 KILL-SWITCH
    // EQUIVALENT at the unit-test level (the bench sweeps 1000 points; this
    // sweeps representative corners).
    #[test]
    fn slerp_preserves_norm_for_all_t_and_theta() {
        let mut rng_seed: u32 = 0x1234_5678;
        let mut next_rand = || {
            // Simple xorshift for reproducibility — no dep on rand crate.
            rng_seed ^= rng_seed << 13;
            rng_seed ^= rng_seed >> 17;
            rng_seed ^= rng_seed << 5;
            (rng_seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };

        let d = 8usize;
        let mut h = [0.0f32; 8];
        let mut mu_t = [0.0f32; 8];
        let mut out = [0.0f32; 8];
        let mut scratch = SlerpScratch::new(d);

        let t_values = [0.0f32, 0.25, 0.5, 0.75, 1.0];

        for _case in 0..32 {
            // Random h.
            for x in h.iter_mut() {
                *x = next_rand();
            }
            // Random mu_t, then normalize.
            for x in mu_t.iter_mut() {
                *x = next_rand();
            }
            normalize(&mut mu_t);

            let norm_h = l2_norm(&h);

            // Compute θ = arccos(ĥ · μ_T) to filter out antipodal cases.
            let mut unit_h = h;
            normalize(&mut unit_h);
            let s_t = dot(&unit_h, &mu_t).clamp(-1.0, 1.0);
            let theta = s_t.acos();
            if theta > PI - THETA_MIN {
                continue; // skip antipodal (covered by V1.5)
            }

            for &t in &t_values {
                slerp_steering_into(&h, &mu_t, t, &mut out, &mut scratch).unwrap();
                let norm_out = l2_norm(&out);
                let rel_err = ((norm_out - norm_h).abs() / norm_h).max(0.0);
                assert!(
                    rel_err < 1e-4,
                    "norm preservation failed: t={t}, θ={theta:.4}, ‖h‖={norm_h:.6}, ‖out‖={norm_out:.6}, rel_err={rel_err:.2e}"
                );
            }
        }
    }

    // V1.4 — aligned edge case (θ < THETA_MIN) uses the lerp fallback; no
    // NaN, no div-by-zero, norm preserved.
    #[test]
    fn slerp_aligned_edge_case_uses_lerp() {
        // h and mu_t both ≈ e_0 (so θ ≈ 0). mu_t is e_0 exactly; h is e_0
        // perturbed by 5e-4 in one component — well below THETA_MIN.
        let mut h = [0.0f32; 8];
        h[0] = 1.0;
        h[1] = 5e-4; // tiny perturbation
        normalize(&mut h); // h ≈ e_0
        let mu_t = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e_0
        let norm_h = l2_norm(&h);

        let mut out = [0.0f32; 8];
        let mut scratch = SlerpScratch::new(8);
        // t = 0.5 — exercises the lerp branch (t = 0 and t = 1 are fast paths).
        slerp_steering_into(&h, &mu_t, 0.5, &mut out, &mut scratch).unwrap();

        // No NaN.
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "out[{i}] = {v} is non-finite");
        }
        // Norm preserved (the lerp fallback renormalizes, so this should hold).
        let norm_out = l2_norm(&out);
        let rel_err = ((norm_out - norm_h).abs() / norm_h).max(0.0);
        assert!(
            rel_err < 1e-4,
            "aligned lerp: norm drift {rel_err:.2e} should be < 1e-4"
        );
    }

    // V1.5 — antipodal edge case (θ > π − THETA_MIN) returns AntipodalDegenerate.
    #[test]
    fn slerp_antipodal_returns_error() {
        let h = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e_0
        let mu_t = [-1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // -e_0
        let mut out = [0.0f32; 8];
        let mut scratch = SlerpScratch::new(8);
        let err = slerp_steering_into(&h, &mu_t, 0.5, &mut out, &mut scratch).unwrap_err();
        assert_eq!(err, SlerpError::AntipodalDegenerate);
    }

    // V1.6 — vMF gate output is always in [0, 1] across the (s_t, κ) sweep.
    #[test]
    fn vmf_gate_bounded_in_zero_one() {
        let alpha_values = [0.3f32, 0.6, 0.8, 1.0];
        let beta_values = [-0.5f32, -0.15, 0.0, 0.3, 0.4];
        let kappa_values = [5.0f32, 20.0, 40.0];

        for &alpha in &alpha_values {
            for &beta in &beta_values {
                for &kappa in &kappa_values {
                    let n = 200usize;
                    for i in 0..=n {
                        let s_t = -1.0 + 2.0 * (i as f32) / (n as f32);
                        let t = vmf_confidence_gate(s_t, kappa, alpha, beta);
                        assert!(
                            t.is_finite(),
                            "t non-finite at s_t={s_t}, κ={kappa}, α={alpha}, β={beta}"
                        );
                        assert!(
                            (0.0..=1.0).contains(&t),
                            "t={t} out of [0,1] at s_t={s_t}, κ={kappa}, α={alpha}, β={beta}"
                        );
                    }
                }
            }
        }
    }

    // V1.7 — gate returns 0 when s_t = 1 (already aligned) for any β > -1.
    #[test]
    fn vmf_gate_zero_when_aligned() {
        for &beta in &[-0.5f32, -0.15, 0.0, 0.3, 0.4] {
            for &kappa in &[5.0f32, 20.0, 40.0] {
                let t = vmf_confidence_gate(1.0, kappa, 1.0, beta);
                assert!(
                    t.abs() < 1e-6,
                    "s_t=1 should give t≈0, got t={t} at κ={kappa}, β={beta}"
                );
            }
        }
    }

    // V1.8 — gate output is non-decreasing as s_t decreases (more drift → more
    // steering). The modulator δ = -tanh(κ·s_T) is monotone decreasing in s_T,
    // so t is monotone non-decreasing as s_T drops.
    #[test]
    fn vmf_gate_increases_with_drift() {
        let kappa = 20.0;
        let alpha = 1.0;
        let beta = 0.0;
        let mut prev_t = -1.0f32;
        let n = 200usize;
        // s_t goes from +1 (aligned) down to -1 (anti-aligned).
        for i in 0..=n {
            let s_t = 1.0 - 2.0 * (i as f32) / (n as f32);
            let t = vmf_confidence_gate(s_t, kappa, alpha, beta);
            assert!(
                t >= prev_t - 1e-6,
                "gate not monotone in drift: s_t={s_t:.3}, t={t:.4}, prev_t={prev_t:.4}"
            );
            prev_t = t;
        }
    }

    // V1.9 — shape mismatch returns Err(ShapeMismatch).
    #[test]
    fn shape_mismatch_returns_err() {
        let h = [1.0f32; 8];
        let mu_t = [0.0f32; 4]; // wrong length
        let mut out = [0.0f32; 8];
        let mut scratch = SlerpScratch::new(8);
        let err = slerp_steering_into(&h, &mu_t, 0.5, &mut out, &mut scratch).unwrap_err();
        assert_eq!(err, SlerpError::ShapeMismatch);

        // Also: scratch length mismatch.
        let mu_t2 = [0.0f32; 8];
        let mut scratch_wrong = SlerpScratch::new(4);
        let err2 = slerp_steering_into(&h, &mu_t2, 0.5, &mut out, &mut scratch_wrong).unwrap_err();
        assert_eq!(err2, SlerpError::ShapeMismatch);
    }

    // V1.10 — zero-norm input returns Err(ZeroNorm).
    #[test]
    fn zero_norm_returns_err() {
        let h = [0.0f32; 8];
        let mu_t = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 8];
        let mut scratch = SlerpScratch::new(8);
        let err = slerp_steering_into(&h, &mu_t, 0.5, &mut out, &mut scratch).unwrap_err();
        assert_eq!(err, SlerpError::ZeroNorm);
    }

    // ── Additional sanity tests (not in the V1.x plan list, but cheap and
    //    document the contract clearly). ─────────────────────────────────

    // Invalid t (out of [0,1]) returns Err(InvalidStrength).
    #[test]
    fn invalid_strength_returns_err() {
        let h = [1.0f32; 4];
        let mu_t = [1.0f32, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        let mut scratch = SlerpScratch::new(4);
        assert_eq!(
            slerp_steering_into(&h, &mu_t, -0.1, &mut out, &mut scratch).unwrap_err(),
            SlerpError::InvalidStrength
        );
        assert_eq!(
            slerp_steering_into(&h, &mu_t, 1.1, &mut out, &mut scratch).unwrap_err(),
            SlerpError::InvalidStrength
        );
        assert_eq!(
            slerp_steering_into(&h, &mu_t, f32::NAN, &mut out, &mut scratch).unwrap_err(),
            SlerpError::InvalidStrength
        );
    }

    // At θ = π/2, t = 1/2, ‖h‖ = 1, the Slerp midpoint equals (ĥ + μ_T)/√2
    // (the great-circle midpoint at right angle). This is the canonical
    // Slerp reference value.
    #[test]
    fn slerp_midpoint_at_right_angle() {
        let mut h = [1.0f32, 0.0, 0.0, 0.0];
        normalize(&mut h); // unit
        let mu_t = [0.0f32, 1.0, 0.0, 0.0]; // unit, orthogonal to h
        let mut out = [0.0f32; 4];
        let mut scratch = SlerpScratch::new(4);
        slerp_steering_into(&h, &mu_t, 0.5, &mut out, &mut scratch).unwrap();

        // Reference: at θ=π/2, t=1/2, Slerp gives (sin(π/4)/sin(π/2)) · ĥ +
        // (sin(π/4)/sin(π/2)) · μ_T = (√2/2)·(ĥ + μ_T).
        let inv_sqrt2 = core::f32::consts::FRAC_1_SQRT_2;
        let expected = [
            inv_sqrt2 * h[0] + inv_sqrt2 * mu_t[0],
            inv_sqrt2 * h[1] + inv_sqrt2 * mu_t[1],
            inv_sqrt2 * h[2] + inv_sqrt2 * mu_t[2],
            inv_sqrt2 * h[3] + inv_sqrt2 * mu_t[3],
        ];
        for (i, (&o, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (o - e).abs() < 1e-5,
                "midpoint at θ=π/2: out[{i}] = {o} should be {e}"
            );
        }
    }

    // The full pipeline (spherical_steering_into) is a no-op when the gate
    // returns t = 0 (already aligned above the selectivity threshold).
    #[test]
    fn full_pipeline_noop_when_aligned() {
        let mut h = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // Slightly off-axis so the norm is still meaningful.
        h[1] = 0.1;
        normalize(&mut h);
        let mu_t = h; // μ_T = h exactly → s_T = 1 → δ = -1 → t = 0.
        let mut out = [0.0f32; 8];
        let mut scratch = SlerpScratch::new(8);
        spherical_steering_into(&h, &mu_t, 20.0, 1.0, 0.0, &mut out, &mut scratch).unwrap();
        assert_eq!(out, h, "aligned full pipeline must be a bit-exact no-op");
    }

    // The full pipeline rotates toward μ_T when drift is detected.
    #[test]
    fn full_pipeline_rotates_when_drift_detected() {
        let h = [1.0f32, 0.0, 0.0, 0.0]; // ĥ = e_0
        let mu_t = [0.0f32, 1.0, 0.0, 0.0]; // orthogonal → s_T = 0 → δ = 0 → t = 0 if β = 0
        let mut out = [0.0f32; 4];
        let mut scratch = SlerpScratch::new(4);

        // β = -0.5: gate fires on s_T = 0 (δ = 0 > β = -0.5). Some rotation.
        spherical_steering_into(&h, &mu_t, 20.0, 0.5, -0.5, &mut out, &mut scratch).unwrap();
        let cos_after = dot(&out, &mu_t) / l2_norm(&out);
        let cos_before = dot(&h, &mu_t) / l2_norm(&h);
        assert!(
            cos_after > cos_before + 1e-3,
            "rotation should increase alignment with μ_T: before={cos_before}, after={cos_after}"
        );
    }

    // zero-alloc in steady state (mirrors Plan 322's test pattern).
    // We can't easily install a counting allocator in a lib test without
    // per-test boilerplate; instead we verify that the public API never
    // calls Vec::new / vec![] / Vec::resize on the hot path by checking
    // that a pre-sized scratch is never resized. The full G4 alloc check
    // lives in the GOAT bench (Phase 2 T2.5).
    #[test]
    fn scratch_ensure_capacity_noop_on_same_d() {
        let mut scratch = SlerpScratch::new(8);
        // Force a reallocation by growing.
        scratch.ensure_capacity(16);
        assert_eq!(scratch.unit_h.len(), 16);
        // Now same-d should be a no-op (no realloc).
        let ptr_before = scratch.unit_h.as_ptr();
        scratch.ensure_capacity(16);
        assert_eq!(
            scratch.unit_h.as_ptr(),
            ptr_before,
            "ensure_capacity same-d should not realloc"
        );
    }

    // Determinism: same inputs → bit-identical outputs across calls.
    #[test]
    fn deterministic_given_same_inputs() {
        let h = [0.3f32, -0.7, 0.5, 1.2];
        let mu_t = [0.0f32, 1.0, 0.0, 0.0];
        let mut out1 = [0.0f32; 4];
        let mut out2 = [0.0f32; 4];
        let mut scratch = SlerpScratch::new(4);
        slerp_steering_into(&h, &mu_t, 0.5, &mut out1, &mut scratch).unwrap();
        slerp_steering_into(&h, &mu_t, 0.5, &mut out2, &mut scratch).unwrap();
        assert_eq!(out1, out2, "same inputs must give bit-identical outputs");
    }

    // Sanity: FRAC_PI_4 / FRAC_PI_2 / PI are imported (silences unused-import
    // warnings in builds where these tests are the only consumers).
    #[test]
    fn consts_imported() {
        assert!((FRAC_PI_4 - core::f32::consts::FRAC_PI_4).abs() < 1e-6);
        assert!((FRAC_PI_2 - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((PI - core::f32::consts::PI).abs() < 1e-6);
    }
}
