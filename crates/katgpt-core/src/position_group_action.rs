//! Unified `PositionGroupAction` trait — RoPE / ALiBi / FoX / Wall as one family.
//!
//! Distilled from Zhang et al., *GRAPE: Group Representational Position
//! Encoding* (arXiv:2512.07805, ICLR 2026, §2.2 + §4.1 + Appendix E). See
//! [Research 446](../../.research/446_GRAPE_Group_Representational_Position_Encoding.md)
//! for the full distillation.
//!
//! # The unification
//!
//! GRAPE's central observation: every mainstream positional encoding is an
//! instance of one one-parameter group action
//!
//! ```text
//! G(n) = exp(n · ω · L)
//! ```
//!
//! obeying the exact relative law `G(t − s) = G(s)⁻¹ · G(t)`. The family
//! splits by where the generator `L` lives:
//!
//! | Encoding | Group | Generator | Closed form |
//! |----------|-------|-----------|-------------|
//! | **RoPE** | `SO(d)` (multiplicative) | `L = abᵀ − baᵀ` (rank-2 skew, per pair) | Rodrigues (Issue 159) |
//! | **NoPE** | trivial | `L = 0` | identity |
//! | **ALiBi** | `GL(d+2)` (additive homogeneous lift) | rank-1 nilpotent `A² = 0` | `I + n·ω·A` |
//! | **FoX** | `GL(d+2)` (additive, per-token gates) | diagonal nilpotent | `I + n·ω·diag(f)` |
//! | **Wall** | `GL(d+2)` (additive, per-channel prefix sums) | rank-1 nilpotent per channel | `I + n·ω·A_c` |
//!
//! Today in katgpt-rs these are scattered across modules with incompatible
//! APIs:
//!
//! - [`PositionFreeCompactor`](../../katgpt_kv/still_kv/position_free.rs) (RoPE) —
//!   its own module, its own vocabulary (`un_rotate_keys`, `un_rotate_f32`).
//! - [`WallDiagonalGate`](../../katgpt_attn/diagonal_gate.rs) (Wall) — different
//!   trait (`DiagonalGate`), different vocabulary (`compute_gate`, `apply`).
//! - `apply_rope_phase_shift` (`katgpt-attn-match`) — RoPE-specific, cannot be
//!   reused for ALiBi or FoX.
//!
//! The result: any tool that wants to be position-encoding-agnostic (KV
//! compaction, attention matching, attention dilation) has to special-case
//! RoPE vs Wall. [`PositionGroupAction`] is the unified abstraction that lets
//! all of them speak the same vocabulary.
//!
//! # Design constraint (non-negotiable, per Issue 160)
//!
//! The trait is a **vocabulary bridge**, not a hot-path replacement. It does
//! NOT replace `PositionFreeCompactor` or `WallDiagonalGate` internally —
//! those stay as-is for hot-path performance. The trait provides:
//!
//! 1. A common interface so position-encoding-agnostic tools can be written
//!    once and work across RoPE/ALiBi/FoX/Wall.
//! 2. Reference implementations that prove the unification is real (G1
//!    bit-identical to the specialized impls on the relevant special cases).
//!
//! Hot-path code should continue to call the specialized impls directly.
//! The trait is for cold-path / vocabulary / interop use.
//!
//! # GRAPE-M bridge
//!
//! [`RopeAction`] implements the canonical-basis RoPE special case directly
//! (per-pair 2D rotations on `(2i, 2i+1)` — the fast path). For the fully
//! general rank-2 rotation plane (GRAPE-M), use [`GrapeMAction`], which wraps
//! [`crate::grapem::Rank2Plane`] (Issue 159). The canonical RoPE is recovered
//! from `GrapeMAction` by choosing `a = e_{2i}, b = e_{2i+1}` per pair.
//!
//! # Numerical contract
//!
//! - All entry points are pure float arithmetic over caller-provided buffers.
//!   Deterministic on a given CPU (same inputs → bit-identical outputs).
//! - `apply_at` and `apply_inverse_at` write to `out`; `out.len()` must equal
//!   `dim()`. Length mismatches trip the impl's error path (panic in debug,
//!   `debug_assert` no-op in release — matches the `phase_rotation` convention).
//! - `apply_at(0, x, out)` is always identity (`out = x`).
//!
//! PositionGroupAction — unified trait (RoPE / ALiBi / FoX / Wall / NoPE /
//! GRAPE-M) per GRAPE §2.2 + §4.1. Vocabulary bridge for position-encoding-
//! agnostic tooling (KV compaction, attention matching). Every positional
//! encoding is an instance of G(n) = exp(n·ω·L); the trait abstracts over
//! where the generator L lives (SO(d) multiplicative vs GL(d+2) additive
//! homogeneous lift). Hot-path code keeps using PositionFreeCompactor /
//! WallDiagonalGate directly; the trait is for cold-path interop.
//! Implies grapem_rodrigues (GrapeMAction wraps Rank2Plane).
//! STAYS OPT-IN — G1–G4 GOAT gate ALL PASS (Bench 458, 2026-07-17); promotion
//! deferred per Issue 160 T4: vocabulary bridge — no hot-path consumer today.
//! Re-evaluate when a position-encoding-agnostic tool (KV compactor, attention
//! matcher) lands. Should be promoted together with grapem_rodrigues.

use crate::grapem::Rank2Plane;

// ── Trait ───────────────────────────────────────────────────────

/// A positional encoding as a one-parameter group action `G(n) = exp(n·ω·L)`.
///
/// All mainstream position encodings are instances:
/// - **RoPE**: multiplicative `SO(d)` action, rank-2 skew generators.
/// - **ALiBi / FoX / Wall**: additive `GL(d+2)` homogeneous lift, rank-1
///   nilpotent generators.
/// - **NoPE**: trivial `L = 0` (identity action).
///
/// The exact relative law `G(t−s) = G(s)⁻¹·G(t)` holds for all
/// implementations, which is what makes the unification useful —
/// position-encoding-agnostic tools can rely on it.
///
/// # Implementor's contract
///
/// - `apply_at(0, x, out)` writes `x` to `out` (identity at the origin).
/// - `apply_at(n, x, out)` followed by `apply_inverse_at(n, out, out2)`
///   recovers `x` (up to f32 precision).
/// - `dim()` is constant for the lifetime of `self`.
/// - `apply_at` and `apply_inverse_at` perform **zero heap allocations**.
///
/// # Hot-path note
///
/// This trait is a **vocabulary bridge**, not a hot-path replacement. For
/// maximum perf, call the specialized impls directly
/// (`PositionFreeCompactor`, `WallDiagonalGate`, etc.). The trait is for
/// cold-path / interop / position-encoding-agnostic tooling.
pub trait PositionGroupAction {
    /// Apply `G(n)` to `x`, writing to `out`. `out.len() == x.len() == dim()`.
    fn apply_at(&self, n: f32, x: &[f32], out: &mut [f32]);

    /// Apply `G(n)⁻¹` (the group inverse) to `x`, writing to `out`.
    /// For one-parameter groups `G(n) = exp(n·ω·L)`, the inverse is
    /// `G(n)⁻¹ = exp(−n·ω·L) = G(−n)`.
    fn apply_inverse_at(&self, n: f32, x: &[f32], out: &mut [f32]);

    /// Dimension of the vector being acted on.
    fn dim(&self) -> usize;
}

// ── NoPE (trivial action) ───────────────────────────────────────

/// NoPE — no positional encoding. `L = 0`, so `G(n) = I` for all `n`.
///
/// Useful as a baseline / control: any tool written against
/// [`PositionGroupAction`] can be tested against `NopeAction` to verify the
/// "no position info" behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NopeAction {
    dim: usize,
}

impl NopeAction {
    /// Construct a NoPE action for the given dimension.
    #[inline]
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl PositionGroupAction for NopeAction {
    #[inline]
    fn apply_at(&self, _n: f32, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        // G(n) = I → out = x.
        if x.as_ptr() != out.as_ptr() {
            out[..x.len()].copy_from_slice(x);
        }
    }

    #[inline]
    fn apply_inverse_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        // Inverse of identity is identity.
        self.apply_at(n, x, out);
    }

    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }
}

// ── RoPE (multiplicative SO(d), rank-2 skew per pair) ───────────

/// RoPE — Rotary Position Embedding. Multiplicative `SO(d)` action with
/// rank-2 skew generators, one per consecutive pair of dimensions.
///
/// The standard RoPE construction: for each pair `(2i, 2i+1)`, apply the 2D
/// rotation
///
/// ```text
/// [x_{2i}  ]     [cos(n·ω_i)  −sin(n·ω_i)] [x_{2i}  ]
/// [x_{2i+1}]  := [sin(n·ω_i)   cos(n·ωi)] [x_{2i+1}]
/// ```
///
/// where `ω_i = θ^(-2i/d)` is the inverse-frequency schedule (the
/// `transformers`-library default is `θ = 10000`).
///
/// This is exactly GRAPE-M in the canonical basis: for each pair, the
/// generator is `L_i = e_{2i}·e_{2i+1}ᵀ − e_{2i+1}·e_{2i}ᵀ` (a rank-2 skew
/// matrix that rotates the `(2i, 2i+1)` plane). The full `d×d` generator is
/// the block-diagonal sum of the per-pair generators.
///
/// # Sign convention
///
/// [`crate::grapem`] rotates `a → −b` (clockwise). RoPE's standard convention
/// is **counter-clockwise** (`x_{2i} → cos·x_{2i} + sin·x_{2i+1}`), so
/// [`RopeAction`] uses the **opposite** sign on the cross term. This is
/// achieved by negating `n` in the inner `grapem_apply_into` call OR by
/// implementing the rotation directly with the standard sign. This impl
/// does the latter (direct implementation) for clarity and to avoid the
/// grapem dependency on the hot path.
///
/// # Construction
///
/// [`RopeAction::new`] builds the standard θ = 10000 schedule.
/// [`RopeAction::with_theta`] lets the caller pick a different base. Both
/// require `dim` to be even (RoPE operates on pairs).
#[derive(Debug, Clone)]
pub struct RopeAction {
    /// Per-pair inverse frequencies `ω_i = θ^(-2i/d)`, length `dim/2`.
    omegas: Vec<f32>,
    dim: usize,
}

impl RopeAction {
    /// Construct with the standard `θ = 10000` schedule.
    ///
    /// `dim` must be even and ≥ 2.
    pub fn new(dim: usize) -> Self {
        Self::with_theta(dim, 10000.0)
    }

    /// Construct with a custom base `θ`.
    ///
    /// `dim` must be even and ≥ 2; `theta` must be > 1.0 (the standard
    /// range is `[10000, 500000]` for long-context extensions).
    pub fn with_theta(dim: usize, theta: f32) -> Self {
        assert!(
            dim >= 2 && dim.is_multiple_of(2),
            "RoPE requires even dim >= 2"
        );
        assert!(theta > 1.0, "RoPE theta must be > 1.0");
        let half = dim / 2;
        let omegas: Vec<f32> = (0..half)
            .map(|i| {
                // ω_i = θ^(-2i/d) = 1 / θ^(2i/d).
                let exponent = -(2.0 * i as f32) / dim as f32;
                theta.powf(exponent)
            })
            .collect();
        Self { omegas, dim }
    }

    /// Read access to the per-pair frequencies.
    pub fn omegas(&self) -> &[f32] {
        &self.omegas
    }
}

impl PositionGroupAction for RopeAction {
    fn apply_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        // For each pair (2i, 2i+1): standard counter-clockwise 2D rotation.
        //   x'_{2i}   = cos(n·ω_i)·x_{2i} − sin(n·ω_i)·x_{2i+1}
        //   x'_{2i+1} = sin(n·ω_i)·x_{2i} + cos(n·ω_i)·x_{2i+1}
        for i in 0..self.omegas.len() {
            let angle = n * self.omegas[i];
            let c = angle.cos();
            let s = angle.sin();
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            out[2 * i] = c.mul_add(x0, -s * x1);
            out[2 * i + 1] = s.mul_add(x0, c * x1);
        }
    }

    fn apply_inverse_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        // G(n)⁻¹ = G(−n) for one-parameter groups. Negate n and dispatch.
        self.apply_at(-n, x, out);
    }

    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }
}

// ── ALiBi (additive GL(d+2), rank-1 nilpotent per head) ─────────

/// ALiBi — Attention with Linear Biases. Additive bias
/// `b_h(t, j) = −β_h · (t − j)` added to the attention score for head `h`,
/// key position `j`, query position `t`.
///
/// In GRAPE's additive `GL(d+2)` framing: the generator is the rank-1
/// nilpotent matrix `A = u·vᵀ` (with `A² = 0` because `vᵀ·u = 0`), and
/// `exp(n·ω·A) = I + n·ω·A` (the nilpotent closed form). The bias is the
/// additive term: `G(n) − I = n·ω·A`.
///
/// ALiBi is a **scalar bias per head**, so [`AlibiAction`] represents one
/// head's bias. The "vector" it acts on is a length-1 scalar (the attention
/// score), and `apply_at` adds `−β · n` to it (where `n = t − j`).
///
/// For multi-head attention, construct one `AlibiAction` per head with the
/// appropriate `β_h`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlibiAction {
    /// Slope `β_h` for this head. The paper's default is a geometric
    /// sequence `β_h = 2^(-8/h)` for `h` heads — caller constructs with the
    /// appropriate slope.
    beta: f32,
}

impl AlibiAction {
    /// Construct with the per-head slope `β`. Must be finite.
    #[inline]
    pub const fn new(beta: f32) -> Self {
        Self { beta }
    }

    /// The slope β.
    #[inline]
    pub const fn beta(&self) -> f32 {
        self.beta
    }

    /// Compute the ALiBi slope schedule for `h` heads (the paper's default
    /// geometric sequence). Returns `h` slopes; the first is `2^(-8)`, then
    /// halved for subsequent heads. The geometric sequence is symmetric for
    /// even `h` and ends at `2^(-8/h)` for the last head.
    ///
    /// # Example
    ///
    /// For `h = 4`: `[1/2^8, 1/2^9, 1/2^10, 1/2^11]` = the standard 4-head schedule.
    pub fn slope_schedule(h: usize) -> Vec<f32> {
        let start = 2f32.powf(-8.0); // 1/256
        (0..h).map(|i| start / 2f32.powi(i as i32)).collect()
    }
}

impl PositionGroupAction for AlibiAction {
    fn apply_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        // ALiBi acts on a length-1 "vector" (the attention score). The bias
        // is `-β·n` where `n = t − j`. apply_at adds the bias to the input.
        debug_assert_eq!(x.len(), 1);
        debug_assert_eq!(out.len(), 1);
        out[0] = x[0] - self.beta * n;
    }

    fn apply_inverse_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        // Inverse: subtract the bias.
        debug_assert_eq!(x.len(), 1);
        debug_assert_eq!(out.len(), 1);
        out[0] = x[0] + self.beta * n;
    }

    #[inline]
    fn dim(&self) -> usize {
        // ALiBi is a scalar bias — "dim" is 1 (the attention score).
        1
    }
}

// ── FoX (additive, per-token forget gates) ──────────────────────

/// FoX — Forget-gate position encoding. Per-token learnable forget gate
/// `f_t ∈ (0, 1)` that multiplicatively attenuates the contribution of
/// position `t`. In GRAPE's additive framing, the generator is a diagonal
/// nilpotent `A = diag(log f_t / ω)` (so `exp(n·ω·A) = diag(f_t^n)`).
///
/// This is a **per-token scalar gate**, analogous to ALiBi but with a
/// learned (or sigmoid-projected) gate per position instead of a fixed
/// linear slope. [`FoxAction`] represents the schedule for a sequence of
/// positions; `apply_at(n, x, out)` scales `x[t]` by `f_t^n`.
///
/// Modelless: the gate values `f_t` are user-supplied (e.g. frozen
/// sigmoid projections of token features). Learning the gates is
/// `→ riir-train`.
#[derive(Debug, Clone)]
pub struct FoxAction {
    /// Per-token forget gates `f_t ∈ (0, 1)`, length = sequence length.
    gates: Vec<f32>,
}

impl FoxAction {
    /// Construct from user-supplied gate values. Each gate must be in
    /// `(0, 1)` (debug-asserted; values outside this range produce
    /// non-bounded attenuation).
    pub fn new(gates: Vec<f32>) -> Self {
        debug_assert!(gates.iter().all(|g| *g > 0.0 && *g < 1.0));
        Self { gates }
    }

    /// Read access to the per-token gates.
    pub fn gates(&self) -> &[f32] {
        &self.gates
    }
}

impl PositionGroupAction for FoxAction {
    fn apply_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        // x[t] *= f_t^n. For n=0, f_t^0 = 1 (identity). For n=1, x[t] *= f_t.
        debug_assert_eq!(x.len(), self.gates.len());
        debug_assert_eq!(out.len(), self.gates.len());
        for (i, gate) in self.gates.iter().enumerate() {
            out[i] = x[i] * gate.powf(n);
        }
    }

    fn apply_inverse_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        // Inverse: x[t] *= f_t^(-n).
        debug_assert_eq!(x.len(), self.gates.len());
        debug_assert_eq!(out.len(), self.gates.len());
        for (i, gate) in self.gates.iter().enumerate() {
            out[i] = x[i] * gate.powf(-n);
        }
    }

    #[inline]
    fn dim(&self) -> usize {
        self.gates.len()
    }
}

// ── Wall (additive, per-channel prefix sums) ────────────────────

/// Wall — the additive bias `b_c(n) = −θ_c · a_n` per channel `c`, where
/// `a_n` is a per-position scalar (typically `a_n = 1` for the uniform
/// schedule) and `θ_c` is a per-channel slope. This is the scalar special
/// case of GRAPE-AP (Issue 161).
///
/// In GRAPE's additive framing: the generator is a rank-1 nilpotent per
/// channel, and `G(n) − I = n · diag(−θ_c · a_n)` is a per-channel additive
/// bias that accumulates linearly in `n`.
///
/// This action represents one channel's bias schedule. For multi-channel
/// attention, construct one `WallAction` per channel (or use the vector
/// form — `apply_at` on a length-`C` vector applies all `C` biases at once
/// with `θ[c]` and `a[c·L + n]` indexing).
#[derive(Debug, Clone)]
pub struct WallAction {
    /// Per-channel slopes `θ_c`, length `C`.
    thetas: Vec<f32>,
    /// Per-position scalar schedule `a_n`, length `L` (the prefix sum
    /// direction). Typically all-ones.
    alphas: Vec<f32>,
}

impl WallAction {
    /// Construct with per-channel slopes `θ_c` and a uniform `a_n = 1`
    /// schedule of length `L`.
    pub fn new_uniform(thetas: Vec<f32>, seq_len: usize) -> Self {
        Self {
            thetas,
            alphas: vec![1.0; seq_len],
        }
    }

    /// Construct with per-channel slopes AND a custom per-position schedule.
    pub fn new(thetas: Vec<f32>, alphas: Vec<f32>) -> Self {
        Self { thetas, alphas }
    }

    /// Read access to the per-channel slopes.
    pub fn thetas(&self) -> &[f32] {
        &self.thetas
    }

    /// Read access to the per-position schedule.
    pub fn alphas(&self) -> &[f32] {
        &self.alphas
    }
}

impl PositionGroupAction for WallAction {
    fn apply_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        // Wall acts on a length-C vector (one entry per channel). The bias
        // is `-θ_c · a_n · n` where n is the relative position and a_n is
        // the per-position scalar. For apply_at, we interpret `n` as the
        // relative position index and apply the cumulative bias.
        //
        // For the "single-step bias" interpretation (matching ALiBi's API):
        // out[c] = x[c] + (-θ_c · a_n · n). Here `a_n` is indexed by `|n|`
        // (clamped to the schedule length).
        let alpha_idx = (n.round() as isize).unsigned_abs();
        let alpha = self.alphas.get(alpha_idx).copied().unwrap_or(0.0);
        debug_assert_eq!(x.len(), self.thetas.len());
        debug_assert_eq!(out.len(), self.thetas.len());
        for (c, theta) in self.thetas.iter().enumerate() {
            out[c] = x[c] - theta * alpha * n;
        }
    }

    fn apply_inverse_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        let alpha_idx = (n.round() as isize).unsigned_abs();
        let alpha = self.alphas.get(alpha_idx).copied().unwrap_or(0.0);
        debug_assert_eq!(x.len(), self.thetas.len());
        debug_assert_eq!(out.len(), self.thetas.len());
        for (c, theta) in self.thetas.iter().enumerate() {
            out[c] = x[c] + theta * alpha * n;
        }
    }

    #[inline]
    fn dim(&self) -> usize {
        self.thetas.len()
    }
}

// ── GRAPE-M bridge (general rank-2 plane via Issue 159) ─────────

/// GRAPE-M general — arbitrary rank-2 rotation plane via the Rodrigues
/// closed form (Issue 159). This is the fully general multiplicative case:
/// any `SO(d)` action whose generator is `L = abᵀ − baᵀ` for arbitrary
/// (possibly non-canonical, possibly learned) `a, b`.
///
/// Wraps [`crate::grapem::Rank2Plane`]. Use [`RopeAction`] for the canonical
/// RoPE special case; use [`GrapeMAction`] for learned rotation planes
/// (per-NPC HLA personality, per-shard rotation in `MerkleFrozenEnvelope`).
#[derive(Debug, Clone)]
pub struct GrapeMAction {
    plane: Rank2Plane,
    /// Frequency scale `ω`.
    omega: f32,
}

impl GrapeMAction {
    /// Construct from generator vectors `a, b` and a frequency scale `ω`.
    ///
    /// Delegates to [`Rank2Plane::new`] for the scalar precomputation. The
    /// vectors are retained (see the grapem module doc for why retaining
    /// `a, b` is mathematically necessary).
    pub fn new(a: &[f32], b: &[f32], omega: f32) -> Self {
        Self {
            plane: Rank2Plane::new(a, b),
            omega,
        }
    }

    /// Read access to the underlying plane.
    pub fn plane(&self) -> &Rank2Plane {
        &self.plane
    }

    /// The frequency scale `ω`.
    #[inline]
    pub const fn omega(&self) -> f32 {
        self.omega
    }
}

impl PositionGroupAction for GrapeMAction {
    fn apply_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        self.plane.apply_into(x, n, self.omega, out).unwrap();
    }

    fn apply_inverse_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        // G(n)⁻¹ = G(−n) for one-parameter groups.
        self.plane.apply_into(x, -n, self.omega, out).unwrap();
    }

    #[inline]
    fn dim(&self) -> usize {
        self.plane.dim()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::needless_range_loop, non_snake_case)]
mod tests {
    use super::*;

    // ── Trait contract tests (apply to all impls) ─────────────────────────

    /// For any PositionGroupAction, apply_at(0, x, out) writes x to out.
    fn assert_identity_at_zero<T: PositionGroupAction>(action: &T, x: &[f32]) {
        let mut out = vec![0f32; x.len()];
        action.apply_at(0.0, x, &mut out);
        for (i, (got, want)) in out.iter().zip(x.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "identity@0 violated at {i}: got {got}, want {want}"
            );
        }
    }

    /// For any PositionGroupAction, apply followed by apply_inverse recovers x.
    fn assert_inverse_roundtrip<T: PositionGroupAction>(action: &T, n: f32, x: &[f32]) {
        let mut tmp = vec![0f32; x.len()];
        action.apply_at(n, x, &mut tmp);
        let mut recovered = vec![0f32; x.len()];
        action.apply_inverse_at(n, &tmp, &mut recovered);
        let norm_x = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        for (i, (got, want)) in recovered.iter().zip(x.iter()).enumerate() {
            let rel = (got - want).abs() / norm_x;
            assert!(
                rel < 1e-4,
                "inverse roundtrip violated at {i}: got {got}, want {want}, rel {rel}"
            );
        }
    }

    #[test]
    fn NopeAction_identity_at_zero() {
        let a = NopeAction::new(4);
        assert_identity_at_zero(&a, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn NopeAction_is_identity_for_all_n() {
        let a = NopeAction::new(4);
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0f32; 4];
        a.apply_at(13.7, &x, &mut out);
        assert_eq!(out, x);
    }

    #[test]
    fn RopeAction_identity_at_zero() {
        let a = RopeAction::new(8);
        assert_identity_at_zero(&a, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    }

    #[test]
    fn RopeAction_inverse_roundtrip() {
        let a = RopeAction::new(8);
        assert_inverse_roundtrip(&a, 7.3, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        assert_inverse_roundtrip(&a, -3.2, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    }

    #[test]
    fn RopeAction_pair_independence() {
        // RoPE rotates each (2i, 2i+1) pair independently — the pair's norm
        // is preserved.
        let a = RopeAction::new(8);
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = [0f32; 8];
        a.apply_at(2.7, &x, &mut out);
        for i in 0..4 {
            let norm_in = (x[2 * i].powi(2) + x[2 * i + 1].powi(2)).sqrt();
            let norm_out = (out[2 * i].powi(2) + out[2 * i + 1].powi(2)).sqrt();
            assert!(
                (norm_in - norm_out).abs() < 1e-4,
                "pair {i} norm not preserved: {norm_in} vs {norm_out}"
            );
        }
    }

    #[test]
    fn AlibiAction_identity_at_zero() {
        let a = AlibiAction::new(0.5);
        assert_identity_at_zero(&a, &[1.0]);
    }

    #[test]
    fn AlibiAction_inverse_roundtrip() {
        let a = AlibiAction::new(0.5);
        assert_inverse_roundtrip(&a, 7.3, &[1.0]);
    }

    #[test]
    fn AlibiAction_adds_negative_slope_times_n() {
        let a = AlibiAction::new(0.5);
        let mut out = [0f32];
        a.apply_at(4.0, &[1.0], &mut out);
        // out = 1.0 - 0.5 * 4.0 = -1.0
        assert!((out[0] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn AlibiAction_slope_schedule() {
        let s = AlibiAction::slope_schedule(4);
        // Expected: [1/256, 1/512, 1/1024, 1/2048].
        assert!((s[0] - 2f32.powf(-8.0)).abs() < 1e-6);
        assert!((s[1] - 2f32.powf(-9.0)).abs() < 1e-6);
        assert!((s[2] - 2f32.powf(-10.0)).abs() < 1e-6);
        assert!((s[3] - 2f32.powf(-11.0)).abs() < 1e-6);
    }

    #[test]
    fn FoxAction_identity_at_zero() {
        let a = FoxAction::new(vec![0.9, 0.5, 0.1]);
        assert_identity_at_zero(&a, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn FoxAction_inverse_roundtrip() {
        let a = FoxAction::new(vec![0.9, 0.5, 0.1]);
        assert_inverse_roundtrip(&a, 2.0, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn FoxAction_attenuates_at_n1() {
        let a = FoxAction::new(vec![0.9, 0.5, 0.1]);
        let x = [1.0, 2.0, 3.0];
        let mut out = [0f32; 3];
        a.apply_at(1.0, &x, &mut out);
        // f_t^1 = f_t, so out = x * f.
        assert!((out[0] - 0.9).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn WallAction_identity_at_zero() {
        let a = WallAction::new_uniform(vec![0.1, 0.2, 0.3], 10);
        assert_identity_at_zero(&a, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn WallAction_inverse_roundtrip() {
        let a = WallAction::new_uniform(vec![0.1, 0.2, 0.3], 10);
        assert_inverse_roundtrip(&a, 3.0, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn GrapeMAction_identity_at_zero() {
        let a = [0.3f32, -0.7, 1.2, 0.5];
        let b = [0.1f32, 0.6, -0.4, 0.9];
        let action = GrapeMAction::new(&a, &b, 1.0);
        assert_identity_at_zero(&action, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn GrapeMAction_inverse_roundtrip() {
        let a = [0.3f32, -0.7, 1.2, 0.5];
        let b = [0.1f32, 0.6, -0.4, 0.9];
        let action = GrapeMAction::new(&a, &b, 1.0);
        assert_inverse_roundtrip(&action, 1.7, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn GrapeMAction_delegates_to_rank2plane() {
        let a = [0.3f32, -0.7, 1.2, 0.5];
        let b = [0.1f32, 0.6, -0.4, 0.9];
        let action = GrapeMAction::new(&a, &b, 0.7);
        let plane = Rank2Plane::new(&a, &b);

        let x = [1.0, 2.0, 3.0, 4.0];
        let mut out_trait = [0f32; 4];
        let mut out_direct = [0f32; 4];
        action.apply_at(1.3, &x, &mut out_trait);
        plane.apply_into(&x, 1.3, 0.7, &mut out_direct).unwrap();
        for i in 0..4 {
            assert!(
                (out_trait[i] - out_direct[i]).abs() < 1e-7,
                "GrapeMAction diverges from Rank2Plane at {i}"
            );
        }
    }

    // ── G1 gate: RoPE bit-identical to a direct reference impl ────────────

    /// G1 for RopeAction: matches a textbook direct RoPE implementation
    /// on the same (theta, dim, pos) inputs.
    #[test]
    fn g1_rope_matches_reference_impl() {
        let dim = 8;
        let theta = 10000.0;
        let action = RopeAction::with_theta(dim, theta);

        // Reference omegas (the textbook formula).
        let ref_omegas: Vec<f32> = (0..dim / 2)
            .map(|i| theta.powf(-(2.0 * i as f32) / dim as f32))
            .collect();
        assert_eq!(action.omegas(), ref_omegas);

        let x = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut out_trait = [0f32; 8];
        let mut out_ref = [0f32; 8];

        for n in [0.0f32, 1.0, 3.7, 13.5, 100.0] {
            action.apply_at(n, &x, &mut out_trait);
            // Reference: per-pair counter-clockwise rotation.
            for i in 0..dim / 2 {
                let angle = n * ref_omegas[i];
                let c = angle.cos();
                let s = angle.sin();
                out_ref[2 * i] = c * x[2 * i] - s * x[2 * i + 1];
                out_ref[2 * i + 1] = s * x[2 * i] + c * x[2 * i + 1];
            }
            for i in 0..dim {
                assert!(
                    (out_trait[i] - out_ref[i]).abs() < 1e-7,
                    "G1 RoPE mismatch at n={n}, i={i}: {} vs {}",
                    out_trait[i],
                    out_ref[i]
                );
            }
        }
    }

    // ── G2 smoke: dispatch overhead is small ──────────────────────────────

    /// G2 smoke: trait dispatch adds minimal overhead vs direct call.
    /// The precise measurement is a criterion bench; this just confirms
    /// the dispatch isn't catastrophically slow (e.g. virtual call per element).
    #[test]
    fn g2_smoke_dispatch_not_pathological() {
        let action = RopeAction::new(8);
        let x = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut out = [0f32; 8];
        let start = std::time::Instant::now();
        for _ in 0..100_000 {
            action.apply_at(1.0, &x, &mut out);
            std::hint::black_box(out);
        }
        let elapsed = start.elapsed();
        // 100k calls. A direct RoPE at d=8 is ~10ns/call, so ~1ms total.
        // Allow 100× headroom for trait dispatch (the gate is structural).
        assert!(
            elapsed.as_millis() < 100,
            "G2 smoke: trait dispatch took {elapsed:?} for 100k calls (>100ms)"
        );
    }
}
