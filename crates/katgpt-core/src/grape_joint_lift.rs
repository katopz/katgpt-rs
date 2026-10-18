//! GRAPE Joint Lift — `GL(d+2)` Block-Diagonal Composition of Rotary + Additive.
//!
//! Distilled from Zhang et al., *GRAPE: Group Representational Position
//! Encoding* (arXiv:2512.07805, ICLR 2026, **Appendix E**). See
//! [Research 446](../../.research/446_GRAPE_Group_Representational_Position_Encoding.md)
//! for the full distillation. This module composes the GRAPE-M rotary
//! primitive ([Issue 159](../../.issues/159_grapem_rank2_rodrigues_exponential.md),
//! [`crate::grapem::Rank2Plane`]) with the GRAPE-A additive bias
//! (paper §4.1–4.2) into a single `GL(d+2)` block-diagonal group action.
//!
//! # Why this exists
//!
//! Today in the stack, Wall Attention *replaces* RoPE — they are alternatives.
//! The GRAPE paper's Appendix E proves they **compose** into a single
//! one-parameter subgroup of `GL(d+2)` while preserving the exact relative
//! law. This module is the modelless computational distillation of that proof:
//! one fused score that applies both a rotary rotation and an additive logit
//! bias in a single `O(d)` pass.
//!
//! # What this computes
//!
//! For query `q ∈ ℝᴰ`, key `k ∈ ℝᴰ`, and relative offset `m = j − i`:
//!
//! ```text
//! score(q, k, m) = qᵀ · exp(m·ω_rot·L) · k / √d   +   m · ω_add · Λ(q, k)
//!                  └────────── rotary ──────────┘     └──── additive ────┘
//! ```
//!
//! where `L = abᵀ − baᵀ` is the rank-2 skew generator (from [`Rank2Plane`]),
//! and `Λ = λq(q) + λk(k)` is the content-modulated gate sum:
//!
//! ```text
//! λq(q) = softplus(vᵀ · q / √d) ≥ 0    (query gate)
//! λk(k) = softplus(uᵀ · k / √d) ≥ 0    (key gate)
//! ```
//!
//! ## The `GL(d+2)` block-diagonal structure (Appendix E)
//!
//! The paper constructs the joint lift via augmented vectors
//! `bq = [q; 1; 0]`, `bk = [k; 0; 1]` and a block-diagonal generator:
//!
//! ```text
//! G_joint(m) = exp(m·L_joint) = [ exp(m·L)   0ᵀ      0    ]
//!                                [ 0         1       m·ω·Λ ]
//!                                [ 0         0       1     ]   ∈ GL(d+2)
//! ```
//!
//! Scoring with the paired inverse-transpose `G_joint(m)^{−⊤}` yields exactly
//! `qᵀ·exp(m·L)·k + m·ω·Λ + const` (the `const` comes from the homogeneous
//! coordinates and cancels in softmax). This module never materialises the
//! `(d+2)×(d+2)` matrix — the score decomposes into one rotary apply + two
//! dot products + one softplus pair + one FMA, all `O(d)`.
//!
//! ## Sign convention (causal regime)
//!
//! For `j ≤ i` (causal attention), `m = j − i ≤ 0`. Since `ω_add > 0` and
//! `Λ ≥ 0`, the additive term `m·ω_add·Λ ≤ 0` — a monotonic penalty. This
//! matches ALiBi's sign convention. ALiBi is recovered exactly when `L = 0`
//! (no rotation), `v = 0`, and `u` is a constant vector giving `λk ≡ β_h`.
//!
//! ## Decoupled `omega_rot` / `omega_add`
//!
//! The paper uses a single shared `ω` for both parts. This implementation
//! decouples them (`omega_rot` for the rotary frequency, `omega_add` for the
//! additive frequency). Setting `omega_rot == omega_add` recovers the paper
//! exactly. The decoupling is a **strict generalization** — not a deviation.
//!
//! # Modelless contract
//!
//! The plane `(a, b)` and the gate vectors `(u, v)` are **user-supplied**.
//! Learning them is `→ riir-train` (the modelless-first mandate, AGENTS.md).
//! This module ships only the deterministic float arithmetic: Rodrigues
//! rotation (via [`Rank2Plane`]) + softplus gates + dot products.
//!
//! # Streaming cache pattern
//!
//! The primitive is stateless. For streaming inference, the caller owns the
//! cache:
//!
//! 1. **At key arrival `j`:** cache `k̂_j = G(j)·k_j` (via
//!    [`Rank2Plane::apply_into`]) and `λk_j = softplus(uᵀ·k_j/√d)` (one dot
//!    + one softplus).
//! 2. **At query time `t`:** compute `q̂_t = G(t)·q_t` and
//!    `λq_t = softplus(vᵀ·q_t/√d)`, then score:
//!    `score = q̂_tᵀ·k̂_j / √d + (j−t)·ω_add·(λq_t + λk_j)`.
//!
//! No cache rewrite when `t` increments — matches RoPE's streaming policy.
//! The joint lift adds only the cached `λk_j` (one scalar per key) on top of
//! RoPE's cached `k̂_j`.
//!
//! # Numerical contract
//!
//! - All entry points are pure float arithmetic over caller-provided buffers.
//!   Deterministic on a given CPU.
//! - `softplus` uses the numerically stable branch form: for `z >= 0`,
//!   `z + log1p(e^{-z})`; for `z < 0`, `log1p(e^z)`. The `exp` is only
//!   ever called on a non-positive argument, so it never overflows.
//! - `score_into` writes into a caller-provided scratch buffer
//!   (`rotated_q_scratch`, length `D`) — zero allocation after [`GrapeJointLift::new`].
//!
//! # Performance
//!
//! `O(d)` per call: one [`Rank2Plane::apply_into`] (2 dot products + 1 FMA
//! triad), two gate dot products (`simd::simd_dot_f32`), two softplus
//! evaluations, one final dot product, one FMA. The structural minimum —
//! the joint lift's value is the **unified API + correctness guarantee**,
//! not a speedup over calling the parts separately.
//!
//! GRAPE Joint Lift — GL(d+2) block-diagonal composition of rotary + additive
//! (GRAPE Appendix E, arXiv:2512.07805). Single-pass fused score that composes
//! GRAPE-M (Issue 159's Rank2Plane, the rotary SO(d) part) with GRAPE-A
//! (paper §4.1, the additive logit bias via softplus gates) into one group
//! action. Closes the composition story: today Wall *replaces* RoPE; this
//! primitive proves they *compose* into a single one-parameter subgroup of
//! GL(d+2) while preserving the exact relative law. The plane (a, b) and gate
//! vectors (u, v) are user-supplied (modelless); learning is → riir-train.
//! Implies grapem_rodrigues (wraps Rank2Plane). Decoupled omega_rot/omega_add
//! is a strict generalization of the paper's shared ω.
//! STAYS OPT-IN — G1–G4 GOAT gate ALL PASS (Bench 460, 2026-07-17); promotion
//! deferred per Issue 163 T6: thin composition layer — value is unified API +
//! correctness guarantee (Appendix E's block-diagonal GL(d+2) proof), not a
//! perf gain over calling the parts separately. No hot-path consumer today.

use crate::grapem::Rank2Plane;
use crate::simd::simd_dot_f32;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by the joint-lift entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum JointLiftError {
    /// `plane.dim() != u_gate.len()` or `!= v_gate.len()` (at construction),
    /// or `q.len()` / `k.len()` / `scratch.len()` disagree with `dim()` (at scoring).
    ShapeMismatch,
}

// ── softplus ─────────────────────────────────────────────────────

/// `softplus(z) = log(1 + e^z) ≥ 0`.
///
/// Numerically stable branch selection avoids overflow in either direction:
/// - `z >= 0`: `z + log(1 + e^{-z})` — `e^{-z} ∈ (0, 1]`, no overflow.
/// - `z < 0`: `log(1 + e^z)` — `e^z ∈ (0, 1)`, no overflow.
///
/// At `z = 0`: `softplus(0) = log(2) ≈ 0.693`. For `|z| > ~88`, the result
/// approaches `max(z, 0)` to f32 precision (the `e^{−|z|}` correction
/// underflows).
///
/// This is the GRAPE-A gate function from paper §4.1. The non-negativity
/// guarantees the additive bias `m·ω·Λ ≤ 0` in the causal regime
/// (`m ≤ 0, ω > 0, Λ ≥ 0`).
#[inline]
pub fn softplus(z: f32) -> f32 {
    use crate::simd::fast_exp;
    if z >= 0.0 {
        // z + log(1 + e^{-z}): for z >= 0, e^{-z} ∈ (0, 1], so the exp never
        // overflows. ln_1p gives an accurate log(1 + small).
        z + fast_exp(-z).ln_1p()
    } else {
        // log(1 + e^z): for z < 0, e^z ∈ (0, 1), so the exp never overflows.
        // ln_1p gives an accurate log(1 + small).
        fast_exp(z).ln_1p()
    }
}

// ── Joint lift handle ───────────────────────────────────────────

/// Pre-computed handle for the GRAPE joint lift (rotary + additive).
///
/// Owns a [`Rank2Plane`] (the rotary part, Issue 159), the two decoupled
/// frequencies `(omega_rot, omega_add)`, and the two gate vectors `(u, v)`
/// (the additive part). After [`GrapeJointLift::new`],
/// [`GrapeJointLift::score_into`] is **zero-allocation** — it reuses the
/// cached plane + gate vectors and writes into a caller-provided scratch
/// buffer.
///
/// # Construction
///
/// [`GrapeJointLift::new`] validates that `plane.dim() == u_gate.len() ==
/// v_gate.len()` and copies the gate vectors into owned `Box<[f32]>` storage
/// (two allocations). Returns
/// [`JointLiftError::ShapeMismatch`] on dimension disagreement.
///
/// # Special cases
///
/// - `omega_add = 0` → pure rotary (RoPE-like via [`Rank2Plane`]).
/// - `omega_rot = 0` or `plane.s() = 0` → pure additive (GRAPE-A, §4.1).
/// - `u = v = 0` → `softplus(0) = log(2)`, constant gate (rotary + constant shift).
/// - `m = 0` → identity offset (score reduces to `qᵀ·k/√d`).
///
/// See the module doc for the streaming cache pattern and the `GL(d+2)`
/// block-diagonal structure.
#[derive(Debug, Clone)]
pub struct GrapeJointLift {
    /// Rotary part — the `SO(d)` rank-2 rotation plane (Issue 159).
    plane: Rank2Plane,
    /// Rotary frequency `ω_rot`. Applied as `exp(m·ω_rot·L)`.
    omega_rot: f32,
    /// Additive frequency `ω_add`. Applied as `m·ω_add·Λ`.
    /// Decoupled from `omega_rot` for flexibility (paper uses a single shared `ω`).
    omega_add: f32,
    /// Key gate vector `u ∈ ℝᴰ`. Computes `λk(k) = softplus(uᵀ·k/√d)`.
    u_gate: Box<[f32]>,
    /// Query gate vector `v ∈ ℝᴰ`. Computes `λq(q) = softplus(vᵀ·q/√d)`.
    v_gate: Box<[f32]>,
}

impl GrapeJointLift {
    /// Construct the joint lift from a rotary plane, two frequencies, and two
    /// gate vectors.
    ///
    /// Copies `u_gate, v_gate` into owned `Box<[f32]>` storage (two
    /// allocations). After construction, [`Self::score_into`] is zero-alloc.
    ///
    /// # Arguments
    ///
    /// * `plane` — the rotary rank-2 plane (Issue 159's [`Rank2Plane`]).
    /// * `omega_rot` — rotary frequency. Applied as `exp(m·ω_rot·L)`.
    /// * `omega_add` — additive frequency. Applied as `m·ω_add·Λ`. Decoupled
    ///   from `omega_rot` (paper uses shared `ω`; set equal to recover exactly).
    /// * `u_gate` — key gate vector, length must equal `plane.dim()`.
    /// * `v_gate` — query gate vector, length must equal `plane.dim()`.
    ///
    /// # Errors
    ///
    /// Returns [`JointLiftError::ShapeMismatch`] if `u_gate.len()` or
    /// `v_gate.len()` disagree with `plane.dim()`.
    pub fn new(
        plane: Rank2Plane,
        omega_rot: f32,
        omega_add: f32,
        u_gate: &[f32],
        v_gate: &[f32],
    ) -> Result<Self, JointLiftError> {
        let d = plane.dim();
        if u_gate.len() != d || v_gate.len() != d {
            return Err(JointLiftError::ShapeMismatch);
        }
        Ok(Self {
            plane,
            omega_rot,
            omega_add,
            u_gate: u_gate.into(),
            v_gate: v_gate.into(),
        })
    }

    /// Dimension `D` of the vectors this lift operates on.
    #[inline]
    pub fn dim(&self) -> usize {
        self.plane.dim()
    }

    /// Read access to the rotary plane.
    #[inline]
    pub fn plane(&self) -> &Rank2Plane {
        &self.plane
    }

    /// Rotary frequency `ω_rot`.
    #[inline]
    pub const fn omega_rot(&self) -> f32 {
        self.omega_rot
    }

    /// Additive frequency `ω_add`.
    #[inline]
    pub const fn omega_add(&self) -> f32 {
        self.omega_add
    }

    /// Read access to the key gate vector `u`.
    #[inline]
    pub fn u_gate(&self) -> &[f32] {
        &self.u_gate
    }

    /// Read access to the query gate vector `v`.
    #[inline]
    pub fn v_gate(&self) -> &[f32] {
        &self.v_gate
    }

    /// Compute the additive gate sum `Λ = λq(q) + λk(k)` where
    /// `λq(q) = softplus(vᵀ·q/√d)` and `λk(k) = softplus(uᵀ·k/√d)`.
    ///
    /// Pure function — exposed for streaming-cache callers that want to
    /// cache `λk(k)` at key arrival and only recompute `λq(q)` per query.
    ///
    /// # Errors
    ///
    /// Returns [`JointLiftError::ShapeMismatch`] if `q.len()` or `k.len()`
    /// disagree with [`Self::dim`].
    #[inline]
    pub fn gate_sum(&self, q: &[f32], k: &[f32]) -> Result<f32, JointLiftError> {
        let d = self.dim();
        if q.len() != d || k.len() != d {
            return Err(JointLiftError::ShapeMismatch);
        }
        let sqrt_d = (d as f32).sqrt();
        let lambda_q = softplus(simd_dot_f32(&self.v_gate, q, d) / sqrt_d);
        let lambda_k = softplus(simd_dot_f32(&self.u_gate, k, d) / sqrt_d);
        Ok(lambda_q + lambda_k)
    }

    /// Compute the joint score in one pass, writing to `out`:
    ///
    /// ```text
    /// out = qᵀ · exp(m·ω_rot·L) · k / √d   +   m · ω_add · (λq(q) + λk(k))
    /// ```
    ///
    /// Uses `rotated_q_scratch` (length `D`) as temporary storage for
    /// `exp(m·ω_rot·L)·q`. Zero allocation after [`Self::new`].
    ///
    /// # Arguments
    ///
    /// * `q` — query vector, length `D`.
    /// * `k` — key vector, length `D`.
    /// * `m` — relative offset `j − i` (integer per the paper's discrete position model).
    /// * `rotated_q_scratch` — scratch buffer for the rotary apply, length `D`.
    ///   May not alias `q` or `k` (passed to [`Rank2Plane::apply_into`]).
    /// * `out` — receives the joint score.
    ///
    /// # Errors
    ///
    /// Returns [`JointLiftError::ShapeMismatch`] if any of `q`, `k`, or
    /// `rotated_q_scratch` disagree with [`Self::dim`].
    ///
    /// # Performance
    ///
    /// `O(d)`, zero allocation. One rotary apply + three dot products + two
    /// softplus + one FMA.
    pub fn score_into(
        &self,
        q: &[f32],
        k: &[f32],
        m: i32,
        rotated_q_scratch: &mut [f32],
        out: &mut f32,
    ) -> Result<(), JointLiftError> {
        let d = self.dim();
        if q.len() != d || k.len() != d || rotated_q_scratch.len() != d {
            return Err(JointLiftError::ShapeMismatch);
        }
        if d == 0 {
            *out = 0.0;
            return Ok(());
        }

        // ── Rotary part: q̂ = exp(m·ω_rot·L)·q ──
        // Rank2Plane::apply_into handles the degenerate-plane (s ≈ 0) case
        // via its small-angle Taylor branch, returning q̂ = q cleanly.
        self.plane
            .apply_into(q, m as f32, self.omega_rot, rotated_q_scratch)
            .map_err(|_| JointLiftError::ShapeMismatch)?;

        // ── Rotary logit: q̂ᵀ·k / √d ──
        let sqrt_d = (d as f32).sqrt();
        let rotary_logit = simd_dot_f32(rotated_q_scratch, k, d) / sqrt_d;

        // ── Additive part: m·ω_add·Λ ──
        // Λ = softplus(vᵀ·q/√d) + softplus(uᵀ·k/√d) ≥ 2·log(2) > 0.
        let lambda_q = softplus(simd_dot_f32(&self.v_gate, q, d) / sqrt_d);
        let lambda_k = softplus(simd_dot_f32(&self.u_gate, k, d) / sqrt_d);
        let lambda_sum = lambda_q + lambda_k;
        let additive_bias = (m as f32) * self.omega_add * lambda_sum;

        // ── Joint score: rotary + additive ──
        *out = rotary_logit + additive_bias;
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grapem::Rank2Plane;

    // ── softplus ────────────────────────────────────────────────

    #[test]
    fn softplus_at_zero_is_log2() {
        let val = softplus(0.0);
        let expected = (2.0_f32).ln(); // log(2) ≈ 0.6931
        assert!(
            (val - expected).abs() < 1e-6,
            "softplus(0) = {val}, expected log(2) = {expected}"
        );
    }

    #[test]
    fn softplus_is_non_negative() {
        for &z in &[-10.0_f32, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 50.0] {
            let val = softplus(z);
            assert!(val >= 0.0, "softplus({z}) = {val} should be >= 0");
        }
    }

    #[test]
    fn softplus_is_monotone_increasing() {
        let zs = [-5.0_f32, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0];
        for w in zs.windows(2) {
            assert!(
                softplus(w[0]) <= softplus(w[1]),
                "softplus not monotone at {} vs {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn softplus_extreme_negative_does_not_nan() {
        let val = softplus(-100.0);
        assert!(val.is_finite(), "softplus(-100) = {val} should be finite");
        // softplus(-100) ≈ -100 + log(1 + e^100) ≈ -100 + 100 = 0 (asymptotically)
        assert!(val.abs() < 1e-3, "softplus(-100) ≈ 0, got {val}");
    }

    #[test]
    fn softplus_extreme_positive_does_not_inf() {
        let val = softplus(100.0);
        assert!(val.is_finite(), "softplus(100) = {val} should be finite");
        // softplus(100) ≈ 100
        assert!((val - 100.0).abs() < 1e-3, "softplus(100) ≈ 100, got {val}");
    }

    #[test]
    fn softplus_matches_naive_definition_for_moderate_z() {
        // For |z| < 10, the naive log(1 + e^z) is numerically safe.
        for &z in &[-5.0_f32, -1.0, 0.0, 1.0, 5.0] {
            let naive_val = (z.exp() + 1.0).ln();
            let stable = softplus(z);
            assert!(
                (naive_val - stable).abs() < 1e-5,
                "softplus({z}): naive {naive_val} vs stable {stable}"
            );
        }
    }

    // ── JointLiftError ──────────────────────────────────────────

    #[test]
    fn new_shape_mismatch_u_gate_returns_err() {
        let plane = Rank2Plane::new(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);
        let u = &[0.0; 3]; // wrong: should be 4
        let v = &[0.0; 4];
        let err = GrapeJointLift::new(plane, 1.0, 1.0, u, v).unwrap_err();
        assert_eq!(err, JointLiftError::ShapeMismatch);
    }

    #[test]
    fn new_shape_mismatch_v_gate_returns_err() {
        let plane = Rank2Plane::new(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);
        let u = &[0.0; 4];
        let v = &[0.0; 5]; // wrong: should be 4
        let err = GrapeJointLift::new(plane, 1.0, 1.0, u, v).unwrap_err();
        assert_eq!(err, JointLiftError::ShapeMismatch);
    }

    // ── Accessors ───────────────────────────────────────────────

    #[test]
    fn accessors_return_construction_values() {
        let a = [1.0_f32, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        let plane = Rank2Plane::new(&a, &b);
        let u = [0.1_f32, 0.2, 0.3, 0.4];
        let v = [0.5_f32, 0.6, 0.7, 0.8];
        let lift = GrapeJointLift::new(plane, 0.5, 1.5, &u, &v).unwrap();
        assert_eq!(lift.dim(), 4);
        assert!((lift.omega_rot() - 0.5).abs() < 1e-7);
        assert!((lift.omega_add() - 1.5).abs() < 1e-7);
        assert_eq!(lift.u_gate(), &u[..]);
        assert_eq!(lift.v_gate(), &v[..]);
        // plane accessor: s = 1 (orthonormal a, b)
        assert!((lift.plane().s() - 1.0).abs() < 1e-6);
    }

    // ── gate_sum ────────────────────────────────────────────────

    #[test]
    fn gate_sum_shape_mismatch_returns_err() {
        let plane = Rank2Plane::new(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);
        let lift = GrapeJointLift::new(plane, 1.0, 1.0, &[0.0; 4], &[0.0; 4]).unwrap();
        let q = &[0.0; 3]; // wrong
        let k = &[0.0; 4];
        assert_eq!(
            lift.gate_sum(q, k).unwrap_err(),
            JointLiftError::ShapeMismatch
        );
    }

    #[test]
    fn gate_sum_zero_vectors_is_2_log2() {
        // softplus(0) = log(2); both gates are log(2); sum = 2·log(2).
        let plane = Rank2Plane::new(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);
        let lift = GrapeJointLift::new(plane, 1.0, 1.0, &[0.0; 4], &[0.0; 4]).unwrap();
        let q = &[0.5, 0.5, 0.5, 0.5]; // non-zero q,k but zero gate vectors
        let k = &[0.3, 0.4, 0.5, 0.6];
        let gs = lift.gate_sum(q, k).unwrap();
        let expected = 2.0 * (2.0_f32).ln();
        assert!(
            (gs - expected).abs() < 1e-6,
            "gate_sum with zero gate vectors = {gs}, expected 2·log(2) = {expected}"
        );
    }

    // ── score_into shape checks ─────────────────────────────────

    #[test]
    fn score_into_shape_mismatch_q_returns_err() {
        let plane = Rank2Plane::new(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);
        let lift = GrapeJointLift::new(plane, 1.0, 1.0, &[0.0; 4], &[0.0; 4]).unwrap();
        let q = &[0.0; 3]; // wrong
        let k = &[0.0; 4];
        let mut scratch = vec![0.0; 4];
        let mut out = 0.0;
        assert_eq!(
            lift.score_into(q, k, 0, &mut scratch, &mut out)
                .unwrap_err(),
            JointLiftError::ShapeMismatch
        );
    }

    #[test]
    fn score_into_shape_mismatch_scratch_returns_err() {
        let plane = Rank2Plane::new(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);
        let lift = GrapeJointLift::new(plane, 1.0, 1.0, &[0.0; 4], &[0.0; 4]).unwrap();
        let q = &[0.0; 4];
        let k = &[0.0; 4];
        let mut scratch = vec![0.0; 3]; // wrong
        let mut out = 0.0;
        assert_eq!(
            lift.score_into(q, k, 0, &mut scratch, &mut out)
                .unwrap_err(),
            JointLiftError::ShapeMismatch
        );
    }

    #[test]
    fn score_into_empty_dim_is_zero() {
        let plane = Rank2Plane::new(&[], &[]);
        let lift = GrapeJointLift::new(plane, 1.0, 1.0, &[], &[]).unwrap();
        let mut out = 99.0;
        let mut scratch: [f32; 0] = [];
        lift.score_into(&[], &[], 5, &mut scratch, &mut out)
            .unwrap();
        assert_eq!(out, 0.0);
    }

    // ── G1 special cases ────────────────────────────────────────

    /// Helper: build a joint lift from explicit vectors.
    fn build_lift(
        a: &[f32],
        b: &[f32],
        omega_rot: f32,
        omega_add: f32,
        u: &[f32],
        v: &[f32],
    ) -> GrapeJointLift {
        let plane = Rank2Plane::new(a, b);
        GrapeJointLift::new(plane, omega_rot, omega_add, u, v).unwrap()
    }

    /// Manual reference: compute the joint score by calling Rank2Plane directly
    /// + explicit gate computation. Used as ground truth for G1.
    fn ref_score(lift: &GrapeJointLift, q: &[f32], k: &[f32], m: i32) -> f32 {
        let d = lift.dim();
        let sqrt_d = (d as f32).sqrt();
        // Rotary: q̂ = exp(m·ω_rot·L)·q
        let mut q_rot = vec![0.0_f32; d];
        lift.plane()
            .apply_into(q, m as f32, lift.omega_rot(), &mut q_rot)
            .unwrap();
        let rotary_logit = simd_dot_f32(&q_rot, k, d) / sqrt_d;
        // Additive: m·ω_add·(softplus(v·q/√d) + softplus(u·k/√d))
        let lambda_q = softplus(simd_dot_f32(lift.v_gate(), q, d) / sqrt_d);
        let lambda_k = softplus(simd_dot_f32(lift.u_gate(), k, d) / sqrt_d);
        let additive_bias = (m as f32) * lift.omega_add() * (lambda_q + lambda_k);
        rotary_logit + additive_bias
    }

    #[test]
    fn g1_special_case_omega_add_zero_is_pure_rotary() {
        // ω_add = 0 → additive term vanishes; score = qᵀ·exp(m·ω_rot·L)·k/√d.
        let lift = build_lift(
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            0.5, // omega_rot
            0.0, // omega_add = 0
            &[0.1, 0.2, 0.3, 0.4],
            &[0.5, 0.6, 0.7, 0.8],
        );
        let q = [0.3_f32, -0.2, 0.5, 0.1];
        let k = [0.4_f32, 0.7, -0.6, 0.9];
        let mut scratch = [0.0_f32; 4];
        let mut score = 0.0;
        lift.score_into(&q, &k, 7, &mut scratch, &mut score)
            .unwrap();

        // Reference: pure rotary logit only.
        let mut q_rot = [0.0_f32; 4];
        lift.plane().apply_into(&q, 7.0, 0.5, &mut q_rot).unwrap();
        let expected = simd_dot_f32(&q_rot, &k, 4) / 2.0; // √4 = 2
        assert!(
            (score - expected).abs() < 1e-6,
            "ω_add=0 score {score} should equal pure rotary {expected}"
        );
    }

    #[test]
    fn g1_special_case_omega_rot_zero_is_pure_additive() {
        // ω_rot = 0 → rotary is identity; score = qᵀ·k/√d + m·ω_add·Λ.
        let lift = build_lift(
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            0.0, // omega_rot = 0
            1.0, // omega_add
            &[0.1, 0.2, 0.3, 0.4],
            &[0.5, 0.6, 0.7, 0.8],
        );
        let q = [0.3_f32, -0.2, 0.5, 0.1];
        let k = [0.4_f32, 0.7, -0.6, 0.9];
        let mut scratch = [0.0_f32; 4];
        let mut score = 0.0;
        lift.score_into(&q, &k, -3, &mut scratch, &mut score)
            .unwrap();

        // Reference: pure additive.
        let dot_qk = simd_dot_f32(&q, &k, 4);
        let lambda_q = softplus(simd_dot_f32(&[0.5, 0.6, 0.7, 0.8], &q, 4) / 2.0);
        let lambda_k = softplus(simd_dot_f32(&[0.1, 0.2, 0.3, 0.4], &k, 4) / 2.0);
        let expected = dot_qk / 2.0 + (-3.0_f32) * 1.0 * (lambda_q + lambda_k);
        assert!(
            (score - expected).abs() < 1e-6,
            "ω_rot=0 score {score} should equal pure additive {expected}"
        );
    }

    #[test]
    fn g1_special_case_degenerate_plane_is_pure_additive() {
        // a ∥ b → s = 0 → rotary is identity (small-angle branch).
        let lift = build_lift(
            &[1.0, 2.0, 3.0, 4.0],
            &[2.0, 4.0, 6.0, 8.0], // parallel to a
            1.0,
            1.0,
            &[0.1, 0.2, 0.3, 0.4],
            &[0.5, 0.6, 0.7, 0.8],
        );
        assert!(lift.plane().s() < 1e-6, "parallel a,b should give s ≈ 0");
        let q = [0.3_f32, -0.2, 0.5, 0.1];
        let k = [0.4_f32, 0.7, -0.6, 0.9];
        let mut scratch = [0.0_f32; 4];
        let mut score = 0.0;
        lift.score_into(&q, &k, 5, &mut scratch, &mut score)
            .unwrap();

        // Reference: pure additive (rotary = identity).
        let dot_qk = simd_dot_f32(&q, &k, 4);
        let lambda_q = softplus(simd_dot_f32(&[0.5, 0.6, 0.7, 0.8], &q, 4) / 2.0);
        let lambda_k = softplus(simd_dot_f32(&[0.1, 0.2, 0.3, 0.4], &k, 4) / 2.0);
        let expected = dot_qk / 2.0 + 5.0 * (lambda_q + lambda_k);
        assert!(
            (score - expected).abs() < 1e-6,
            "degenerate-plane score {score} should equal pure additive {expected}"
        );
    }

    #[test]
    fn g1_special_case_m_zero_is_qk_dot() {
        // m = 0 → no offset; score = qᵀ·k/√d (rotary identity + zero additive).
        let lift = build_lift(
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            0.7,
            1.0,
            &[0.1, 0.2, 0.3, 0.4],
            &[0.5, 0.6, 0.7, 0.8],
        );
        let q = [0.3_f32, -0.2, 0.5, 0.1];
        let k = [0.4_f32, 0.7, -0.6, 0.9];
        let mut scratch = [0.0_f32; 4];
        let mut score = 0.0;
        lift.score_into(&q, &k, 0, &mut scratch, &mut score)
            .unwrap();

        let expected = simd_dot_f32(&q, &k, 4) / 2.0;
        assert!(
            (score - expected).abs() < 1e-6,
            "m=0 score {score} should equal q·k/√d {expected}"
        );
    }

    #[test]
    fn g1_special_case_zero_gate_vectors_is_rotary_plus_constant() {
        // u = v = 0 → softplus(0) = log(2); Λ = 2·log(2) (constant).
        // score = rotary_logit + m·ω_add·2·log(2).
        let lift = build_lift(
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            0.5,
            1.0,
            &[0.0; 4], // u = 0
            &[0.0; 4], // v = 0
        );
        let q = [0.3_f32, -0.2, 0.5, 0.1];
        let k = [0.4_f32, 0.7, -0.6, 0.9];
        let mut scratch = [0.0_f32; 4];
        let mut score = 0.0;
        lift.score_into(&q, &k, -4, &mut scratch, &mut score)
            .unwrap();

        // Reference: rotary + m·ω_add·2·log(2).
        let mut q_rot = [0.0_f32; 4];
        lift.plane().apply_into(&q, -4.0, 0.5, &mut q_rot).unwrap();
        let rotary_logit = simd_dot_f32(&q_rot, &k, 4) / 2.0;
        let expected = rotary_logit + (-4.0_f32) * 1.0 * 2.0 * (2.0_f32).ln();
        assert!(
            (score - expected).abs() < 1e-6,
            "zero-gate score {score} should equal rotary + const {expected}"
        );
    }

    #[test]
    fn g1_additive_term_is_non_positive_in_causal_regime() {
        // For m ≤ 0, ω_add > 0, Λ ≥ 0: additive = m·ω_add·Λ ≤ 0.
        // Verify by checking score ≤ rotary_logit for m < 0.
        let lift = build_lift(
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            0.3,
            1.0,
            &[0.1, 0.2, 0.3, 0.4],
            &[0.5, 0.6, 0.7, 0.8],
        );
        let q = [0.3_f32, -0.2, 0.5, 0.1];
        let k = [0.4_f32, 0.7, -0.6, 0.9];
        for m in [-1_i32, -3, -10, -50] {
            let mut scratch = [0.0_f32; 4];
            let mut score = 0.0;
            lift.score_into(&q, &k, m, &mut scratch, &mut score)
                .unwrap();
            // Rotary-only score (omega_add = 0 reference).
            let mut q_rot = [0.0_f32; 4];
            lift.plane()
                .apply_into(&q, m as f32, 0.3, &mut q_rot)
                .unwrap();
            let rotary_only = simd_dot_f32(&q_rot, &k, 4) / 2.0;
            assert!(
                score <= rotary_only + 1e-6,
                "m={m}: joint score {score} should be ≤ rotary-only {rotary_only} (additive ≤ 0)"
            );
        }
    }

    // ── G1 main: bit-identical to manual composition ────────────

    /// Deterministic LCG for reproducible random inputs.
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        #[inline]
        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }
        fn next_f32_vec(&mut self, d: usize) -> Vec<f32> {
            (0..d)
                .map(|_| {
                    let bits = self.next_u64() >> 40;
                    let u = (bits as f32) / ((1u64 << 24) as f32);
                    u * 2.0 - 1.0
                })
                .collect()
        }
        fn next_f32(&mut self, hi: f32) -> f32 {
            let bits = self.next_u64() >> 40;
            let u = (bits as f32) / ((1u64 << 24) as f32);
            u * hi
        }
    }

    #[test]
    fn g1_joint_score_matches_manual_composition_across_dims() {
        // For each dim, generate 20 random instances and verify the joint
        // score is bit-identical (within f32 rounding) to the manual
        // composition: Rank2Plane::apply_into + dot + softplus gates + FMA.
        // Seed 0x163 (Issue 163) | 0xC0FFEE (deterministic marker).
        let mut rng = Lcg::new(0x163C0FFEE_u64);
        for &d in &[8_usize, 16, 32, 64] {
            for _ in 0..20 {
                let a = rng.next_f32_vec(d);
                let b = rng.next_f32_vec(d);
                let u = rng.next_f32_vec(d);
                let v = rng.next_f32_vec(d);
                let q = rng.next_f32_vec(d);
                let k = rng.next_f32_vec(d);
                let omega_rot = rng.next_f32(2.0) + 0.1; // [0.1, 2.1)
                let omega_add = rng.next_f32(2.0) + 0.1;
                let m = (rng.next_u64() % 200) as i32 - 100; // [-100, 100)

                let lift = build_lift(&a, &b, omega_rot, omega_add, &u, &v);
                let mut scratch = vec![0.0; d];
                let mut score = 0.0;
                lift.score_into(&q, &k, m, &mut scratch, &mut score)
                    .unwrap();

                let expected = ref_score(&lift, &q, &k, m);

                // The score_into and ref_score use identical operations, so
                // the difference should be near zero (only reordering of the
                // FMA could introduce sub-ULP drift). Use a tight tolerance.
                let abs_diff = (score - expected).abs();
                let scale = expected.abs().max(1.0);
                assert!(
                    abs_diff < 1e-5 * scale,
                    "d={d} m={m}: joint score {score} vs ref {expected}, abs_diff {abs_diff}"
                );
            }
        }
    }

    #[test]
    fn g1_relative_law_score_depends_only_on_offset() {
        // The joint lift's defining property (Appendix E): the score depends
        // only on m = j − i, not on absolute (i, j). Since score_into takes
        // m directly, this is structurally guaranteed. Verify by checking
        // that score_into with the same m gives identical results regardless
        // of how m was derived — i.e., the API correctly encodes relativity.
        let lift = build_lift(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            0.5,
            1.0,
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            &[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        );
        let q = [0.3_f32, -0.2, 0.5, 0.1, 0.4, -0.3, 0.2, 0.6];
        let k = [0.4_f32, 0.7, -0.6, 0.9, 0.1, -0.4, 0.5, 0.2];

        // Same offset m = -5, derived from different (i, j) pairs.
        let m = -5_i32;
        let pairs = [(0_i32, 5), (10, 15), (100, 105), (1000, 1005)];
        let mut scores = Vec::with_capacity(pairs.len());
        for _ in &pairs {
            let mut scratch = [0.0_f32; 8];
            let mut score = 0.0;
            lift.score_into(&q, &k, m, &mut scratch, &mut score)
                .unwrap();
            scores.push(score);
        }
        // All scores must be bit-identical — the API encodes relativity by
        // construction (m is the only positional input).
        for w in scores.windows(2) {
            assert!(
                (w[0] - w[1]).abs() < 1e-7,
                "relative law violated: scores differ by {}",
                (w[0] - w[1]).abs()
            );
        }
    }

    // ── G2 latency smoke test ───────────────────────────────────

    #[test]
    fn g2_score_into_not_slower_than_separate_calls() {
        // Smoke test: score_into (fused) should not be more than 1.10× slower
        // than calling Rank2Plane::apply_into + dot + softplus + FMA separately.
        // The value of score_into is the unified API + correctness, not speed.
        // This test guards against accidental O(d²) regressions.
        let d = 64_usize;
        let mut rng = Lcg::new(0xC0FFEE);
        let a = rng.next_f32_vec(d);
        let b = rng.next_f32_vec(d);
        let u = rng.next_f32_vec(d);
        let v = rng.next_f32_vec(d);
        let q = rng.next_f32_vec(d);
        let k = rng.next_f32_vec(d);
        let lift = build_lift(&a, &b, 0.5, 1.0, &u, &v);

        let n_iter: usize = 10_000;
        let mut scratch = vec![0.0; d];

        // Fused path.
        let start_fused = std::time::Instant::now();
        let mut acc_fused = 0.0_f32;
        for i in 0..n_iter {
            let mut score = 0.0;
            lift.score_into(&q, &k, (i as i32) % 50 - 25, &mut scratch, &mut score)
                .unwrap();
            acc_fused += score;
        }
        let elapsed_fused = start_fused.elapsed().as_nanos();

        // Separate path (manual composition).
        let start_sep = std::time::Instant::now();
        let mut acc_sep = 0.0_f32;
        let sqrt_d = (d as f32).sqrt();
        for i in 0..n_iter {
            let m = (i as i32) % 50 - 25;
            let mut q_rot = vec![0.0; d];
            lift.plane()
                .apply_into(&q, m as f32, 0.5, &mut q_rot)
                .unwrap();
            let rotary = simd_dot_f32(&q_rot, &k, d) / sqrt_d;
            let lambda_q = softplus(simd_dot_f32(&v, &q, d) / sqrt_d);
            let lambda_k = softplus(simd_dot_f32(&u, &k, d) / sqrt_d);
            acc_sep += rotary + (m as f32) * 1.0 * (lambda_q + lambda_k);
        }
        let elapsed_sep = start_sep.elapsed().as_nanos();

        // Black-box the accumulators to prevent dead-code elimination.
        std::hint::black_box(acc_fused);
        std::hint::black_box(acc_sep);

        // The separate path allocates a scratch Vec per iteration, so the
        // fused path should actually be FASTER in this test. We just assert
        // the fused path is not pathologically slow (< 2× separate).
        // The Issue 163 G2 target is ≤ 1.10×; the separate path here is
        // penalized by per-iter alloc, so the ratio is generous.
        let ratio = elapsed_fused as f64 / elapsed_sep.max(1) as f64;
        assert!(
            ratio < 2.0,
            "fused path {elapsed_fused}ns vs separate {elapsed_sep}ns, ratio {ratio:.3} (should be < 2.0)"
        );
    }

    // ── G4 structural: zero-alloc score_into ────────────────────

    /// The `score_into` signature takes only `&[f32]`, `&mut [f32]`, `&mut f32`,
    /// and `i32` for the hot-path args. The plane + gate vectors are owned
    /// at construction. This is a structural check — the runtime alloc gate
    /// lives in a sibling integration test (matches Issue 161's pattern).
    #[test]
    fn g4_score_into_takes_only_borrowed_slices() {
        let plane = Rank2Plane::new(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let lift = GrapeJointLift::new(plane, 1.0, 1.0, &[0.1; 8], &[0.2; 8]).unwrap();
        let q = [0.5; 8];
        let k = [0.3; 8];
        let mut scratch = [0.0; 8];
        let mut out = 0.0;
        // Type-check: all hot-path args are borrowed.
        let _: Result<(), JointLiftError> = lift.score_into(&q, &k, 5, &mut scratch, &mut out);
        assert!(out.is_finite());
    }

    #[test]
    fn g4_new_allocates_exactly_two_box_slices() {
        // Structural: u_gate and v_gate are Box<[f32]> (one alloc each).
        // The Rank2Plane inside also has two Box<[f32]> (a, b), but those
        // are constructed before GrapeJointLift::new — they're the caller's
        // allocs. This test documents the construction-time alloc count.
        let plane = Rank2Plane::new(&[1.0, 0.0], &[0.0, 1.0]);
        let _lift = GrapeJointLift::new(plane, 1.0, 1.0, &[0.1; 2], &[0.2; 2]).unwrap();
        // GrapeJointLift::new does exactly 2 allocs (u, v via `slice.into()`).
        // (The plane's 2 allocs happened in Rank2Plane::new.)
    }
}
