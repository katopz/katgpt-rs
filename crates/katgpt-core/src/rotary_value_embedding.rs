//! RoVE — Rotary Value Embeddings Attention.
//!
//! Distilled from García-Castellanos, Weiler, Bekkers, *RoVE: Rotary Value
//! Embeddings Attention for Relative Position-dependent Value Pathways*
//! (arXiv:2606.11275, Jul 2026, [code](https://github.com/AGarciaCast/RoVE)).
//! See [Research 452](../../.research/452_RoVE_Rotary_Value_Embeddings_Attentive_Convolution.md)
//! for the full distillation and [Plan 557](../../.plans/557_rotary_value_embeddings.md)
//! for the execution tracker.
//!
//! # The asymmetry RoVE fixes
//!
//! Standard RoPE rotates queries `q_i` by `R_i` and keys `k_j` by `R_j` before
//! the inner product, making attention scores depend only on the relative
//! offset `δ = j − i`. The **value pathway (OV circuit) is position-blind** —
//! `W_V` is applied identically to every token regardless of its offset from
//! the query.
//!
//! RoVE additionally rotates each value `W_V·x_j` by `R_j` before aggregation,
//! then inverse-rotates the output by `R_{−i}` to put it in the query's frame:
//!
//! ```text
//! ỹ_i = R_{−i} · Σ_j A_ij · R_j · W_V · x_j
//!     = Σ_j A_ij · (R_{j−i} · W_V) · x_j
//!     = Σ_j A_ij · ψ_{j−i} · x_j    ← attentive convolution
//! ```
//!
//! The effective kernel `ψ_δ = R_δ·W_V` is an offset-indexed block-Toeplitz
//! family — converting the attention mixer from Kronecker structure
//! (`A ⊗ W_V`) to gated block-Toeplitz structure.
//!
//! # Three equivalent views
//!
//! 1. **Attentive convolution** — the value kernel becomes a function of the
//!    offset `δ = j − i`, not just the attention weight. This is the paper's
//!    headline framing (§3, Eq. 3).
//! 2. **Matrix mixer** — the attention output `Σ_j A_ij · V_j` is a
//!    matrix-vector product `A · V`. Standard attention factors `A` (RoPE-only
//!    on Q/K) from `V` (position-blind). RoVE couples them: `V` becomes a
//!    function of the row index via `R_j`, and the output is re-indexed by
//!    `R_{−i}`. The mixer structure is no longer `A ⊗ I_d` — it is a gated
//!    block-Toeplitz operator.
//! 3. **Local frame** — each value lives in its token's local frame; the
//!    inverse rotation `R_{−i}` maps the aggregated output back into the
//!    query's frame. This is the geometric view and is the most useful for
//!    FlashAttention compatibility (see below).
//!
//! # FlashAttention compatibility
//!
//! The rotations act on V (before the kernel call) and on the aggregated
//! output (after the kernel call). They **never** touch the `n×n` attention
//! score matrix. This means RoVE can be fused into any FlashAttention-style
//! kernel that already supports per-head V rotation — no `n×n` materialization
//! is needed. Plan 557 Phase 2 G5 verifies the output-equivalence algebraically.
//!
//! # Sibling relationship with [`crate::position_group_action`]
//!
//! RoVE is the **first hot-path consumer** of GRAPE's [`RopeAction`] (Issue 160,
//! Research 446). GRAPE shipped the `PositionGroupAction` trait as a
//! "vocabulary bridge for cold-path code"; RoVE turns it into a real attention
//! variant by calling [`RopeAction::apply_at`] on the V projection and
//! [`RopeAction::apply_inverse_at`] on the post-softmax output.
//!
//! # Inference-only caveat (the open question)
//!
//! The paper validates RoVE as a **training-time architectural choice** — the
//! model is trained from scratch with V rotation. The inference-time retrofit
//! onto RoPE-trained checkpoints is **unvalidated**. The structural argument
//! cuts both ways: RoVE makes the OV circuit offset-aware, but `W_V` was
//! trained under the offset-blind assumption. Plan 557 Phase 5 is a mandatory
//! PoC before any default-on promotion.
//!
//! # Parameter-free
//!
//! Zero new weights. The only config is `theta` (the RoPE base frequency,
//! default 10000.0), and that exists only to thread YaRN-style rescaling in
//! future work (paper Appendix C). The primitive itself is pure float
//! arithmetic over caller-owned buffers.
//!
//! RoVE — Rotary Value Embeddings Attention (Plan 557, Research 452,
//! arXiv:2606.11275). Extends RoPE from Q/K to the V projection + inverse-
//! rotates the aggregated output, yielding an attentive convolution with
//! offset-indexed block-Toeplitz kernel ψ_δ = R_δ·W_V. Parameter-free,
//! FlashAttention-compatible. The first hot-path consumer of GRAPE's
//! PositionGroupAction trait — turns the "vocabulary bridge" into a real
//! attention variant. Implies position_group_action (consumes RopeAction).
//! STAYS OPT-IN until Phase 5 retrofit PoC settles the open question of
//! whether inference-time RoVE onto RoPE-trained checkpoints helps or hurts.

use crate::position_group_action::{PositionGroupAction, RopeAction};

// ── Config ──────────────────────────────────────────────────────

/// Configuration for RoVE.
///
/// RoVE is parameter-free — the only field is `theta` (the RoPE base
/// frequency), which exists for forward-compat with YaRN-style rescaling
/// (paper Appendix C). The default `10000.0` matches the paper's RoPE base
/// and our existing [`crate::position_group_action::RopeAction::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoVeConfig {
    /// RoPE base frequency. Default `10000.0`.
    pub theta: f32,
}

impl Default for RoVeConfig {
    #[inline]
    fn default() -> Self {
        Self { theta: 10000.0 }
    }
}

impl RoVeConfig {
    /// Construct with a custom base frequency.
    #[inline]
    pub const fn new(theta: f32) -> Self {
        Self { theta }
    }

    /// Build a [`RopeAction`] for the given head dimension.
    ///
    /// `dim` must be even and `>= 2` (RoPE operates on pairs). Odd `dim`
    /// panics with a clear message — see G7.
    #[inline]
    pub fn build_rope_action(&self, dim: usize) -> RopeAction {
        RopeAction::with_theta(dim, self.theta)
    }
}

// ── Per-token primitives (zero-allocation) ──────────────────────

/// Rotate the value projection at position `pos` into the global frame.
///
/// Computes `V_rot[pos] = R_pos · V[pos]` in place into `out`.
///
/// **Zero allocation.** Caller-owned `out` buffer (length `dim`).
#[inline]
pub fn rotate_values_into(action: &RopeAction, pos: usize, values: &[f32], out: &mut [f32]) {
    action.apply_at(pos as f32, values, out);
}

/// Inverse-rotate the softmax-aggregated output at query position `pos` from
/// the global frame back into the query's local frame.
///
/// Computes `ỹ[pos] = R_{−pos} · aggregated[pos]` in place into `out`.
///
/// **Zero allocation.** Caller-owned `out` buffer (length `dim`).
#[inline]
pub fn inverse_rotate_output_into(
    action: &RopeAction,
    pos: usize,
    aggregated: &[f32],
    out: &mut [f32],
) {
    action.apply_inverse_at(pos as f32, aggregated, out);
}

// ── Batch primitives (zero-allocation in the loop) ──────────────

/// Batch value rotation for all tokens in a sequence.
///
/// `values` and `out` are flat `[n * dim]` row-major buffers. For each token
/// `t` at position `positions[t]`, computes `out[t] = R_{positions[t]} · values[t]`.
///
/// - **Zero allocation** in the loop (per-token slice borrows only).
/// - This is the API the attention forward path calls once per layer (Phase 3).
///
/// - **Hot-path note (Plan 557 Phase 3):** this scalar path recomputes
///   `cos`/`sin` per pair per token — the dominant cost. For batch rotation
///   of `n` tokens at contiguous positions `0..n`, prefer
///   [`batch_rotate_values_into_fast`] with a precomputed [`RoVeRotationTable`],
///   which eliminates transcendental calls from the inner loop entirely.
///   The scalar path is kept as the bit-identical reference (G1 contract).
pub fn batch_rotate_values_into(
    action: &RopeAction,
    positions: &[usize],
    values: &[f32],
    out: &mut [f32],
    dim: usize,
) {
    debug_assert_eq!(
        values.len(),
        positions.len() * dim,
        "values buffer length mismatch"
    );
    debug_assert_eq!(
        out.len(),
        positions.len() * dim,
        "out buffer length mismatch"
    );
    for (t, &pos) in positions.iter().enumerate() {
        let start = t * dim;
        rotate_values_into(
            action,
            pos,
            &values[start..start + dim],
            &mut out[start..start + dim],
        );
    }
}

/// Batch inverse output rotation for all query positions.
///
/// `aggregated` and `out` are flat `[n * dim]` row-major buffers. For each
/// query `i` at position `positions[i]`, computes
/// `out[i] = R_{−positions[i]} · aggregated[i]`.
///
/// - **Zero allocation** in the loop.
pub fn batch_inverse_rotate_output_into(
    action: &RopeAction,
    positions: &[usize],
    aggregated: &[f32],
    out: &mut [f32],
    dim: usize,
) {
    debug_assert_eq!(
        aggregated.len(),
        positions.len() * dim,
        "aggregated buffer length mismatch"
    );
    debug_assert_eq!(
        out.len(),
        positions.len() * dim,
        "out buffer length mismatch"
    );
    for (t, &pos) in positions.iter().enumerate() {
        let start = t * dim;
        inverse_rotate_output_into(
            action,
            pos,
            &aggregated[start..start + dim],
            &mut out[start..start + dim],
        );
    }
}

// ── Fast batch primitives (precomputed cos/sin table — Phase 3 G2 unblock) ──
//
// The scalar [`batch_rotate_values_into`] / [`batch_inverse_rotate_output_into`]
// recompute `cos(angle)` / `sin(angle)` per pair per token. For `n=1024, d=768`,
// that is `1024 × 384 × 2 = 786,432` transcendental calls per layer. On a
// typical ARM core each `sincos` call is ~30–100 cycles — the transcendentals
// dominate the rotation cost entirely.
//
// The fast path precomputes the cos/sin table ONCE per `(theta, dim, max_pos)`
// triple, then the per-token inner loop is pure `mul_add` arithmetic. This is
// the G2 unblock for Plan 557 Phase 2's honest FAIL (6.45% → target < 5%).
//
// The table is `max_pos × dim` entries, storing interleaved `(cos, sin)` per
// pair — matching the AoS input layout so the rotation inner loop reads
// contiguous `(c, s)` pairs and writes contiguous `(x0', x1')` pairs. The
// inner loop uses `f32::mul_add` which lowers to a single FMA on hardware
// that has it (NEON vfmaq_n_f32_lane, AVX2 vfmadd213ps) — matching the
// scalar reference's rounding exactly (single-rounding FMA semantics).
//
// Auto-vectorization note: the AoS pair layout does NOT auto-vectorize as
// cleanly as SoA, but LLVM's SLP vectorizer does catch the 2-wide pair
// pattern on aarch64 (confirmed via `cargo asm` spot-check). The dominant
// win is eliminating the transcendentals; SIMD on the FMA is a secondary
// gain that the bench measures honestly.

/// Precomputed cos/sin table for RoVE batch rotation.
///
/// Stores `(cos, sin)` for every `(position, pair)` combination, laid out as
/// `max_pos × dim` interleaved pairs: position `pos`, pair `i` lives at
/// `table[pos * dim + 2*i]` (cos) and `table[pos * dim + 2*i + 1]` (sin).
///
/// Build once per forward pass (or once per layer — the table is
/// position-only, so it's reused across layers when positions don't change).
/// The fast batch functions then read from this table with zero
/// transcendental calls in the inner loop.
///
/// **Memory cost:** `max_pos × dim × 4` bytes. For `max_pos=1024, dim=768`,
/// that's 3 MB — acceptable for inference-time scratch (freed when the table
/// drops). For long-context (max_pos=8192+), consider a streaming table that
/// computes cos/sin on demand for a window of positions — not implemented
/// here (out of scope for the G2 unblock).
pub struct RoVeRotationTable {
    /// Interleaved `(cos, sin)` pairs: `table[pos * dim + 2*i] = cos(pos·ω_i)`,
    /// `table[pos * dim + 2*i + 1] = sin(pos·ω_i)`.
    table: Vec<f32>,
    dim: usize,
    max_pos: usize,
}

impl RoVeRotationTable {
    /// Build the cos/sin table for positions `0..max_pos` at the given `dim`
    /// and `theta`.
    ///
    /// `dim` must be even and `>= 2` (delegates to [`RopeAction::with_theta`]
    /// for validation). `max_pos` must be `>= 1`.
    pub fn new(dim: usize, theta: f32, max_pos: usize) -> Self {
        // Validate dim + theta by constructing a RopeAction (panics on invalid).
        let action = RopeAction::with_theta(dim, theta);
        assert!(max_pos >= 1, "max_pos must be >= 1");
        let omegas = action.omegas();

        let mut table = vec![0.0f32; max_pos * dim];
        for pos in 0..max_pos {
            let pos_f = pos as f32;
            let row = &mut table[pos * dim..(pos + 1) * dim];
            for (i, &omega) in omegas.iter().enumerate() {
                let angle = pos_f * omega;
                // Match the scalar RopeAction::apply_at path EXACTLY: separate
                // .cos() and .sin() calls, NOT .sin_cos(). The fused sin_cos()
                // uses a different internal argument-reduction path on some
                // libm implementations and produces a 1-ULP difference vs
                // separate cos()/sin() calls. Bit-identical output to the
                // scalar path requires matching the transcendental call pattern.
                row[2 * i] = angle.cos();
                row[2 * i + 1] = angle.sin();
            }
        }

        Self {
            table,
            dim,
            max_pos,
        }
    }

    /// Dimension of the vectors being rotated.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Maximum position covered by the table (exclusive upper bound).
    #[inline]
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// Raw access to the interleaved `(cos, sin)` table (for testing).
    pub fn as_slice(&self) -> &[f32] {
        &self.table
    }
}

/// Fast batch value rotation using a precomputed cos/sin table.
///
/// For each token `t` at position `positions[t]`, computes
/// `out[t] = R_{positions[t]} · values[t]` using the precomputed table —
/// zero transcendental calls in the inner loop.
///
/// **Panics** if any `positions[t] >= table.max_pos()`.
///
/// **Bit-identical to [`batch_rotate_values_into`]?** Yes — the forward
/// direction uses the same positive-angle `cos`/`sin` and the same `mul_add`
/// rotation formula as the scalar path, so the two paths produce bit-for-bit
/// identical output for the same `(theta, dim, positions)`. This is verified
/// by the G8 test (forward direction, tol 0.0).
pub fn batch_rotate_values_into_fast(
    table: &RoVeRotationTable,
    positions: &[usize],
    values: &[f32],
    out: &mut [f32],
) {
    let dim = table.dim();
    debug_assert_eq!(
        values.len(),
        positions.len() * dim,
        "values buffer length mismatch"
    );
    debug_assert_eq!(
        out.len(),
        positions.len() * dim,
        "out buffer length mismatch"
    );
    let half = dim / 2;
    let tbl = table.as_slice();

    for (t, &pos) in positions.iter().enumerate() {
        assert!(
            pos < table.max_pos(),
            "position {pos} >= table.max_pos() {}",
            table.max_pos()
        );
        let in_start = t * dim;
        let out_start = t * dim;
        let tbl_start = pos * dim;

        // Inner loop: pure mul_add arithmetic, no transcendentals.
        // Reads contiguous `(c, s)` pairs from the table and contiguous
        // `(x0, x1)` pairs from the input — cache-friendly streaming access.
        for i in 0..half {
            let c = tbl[tbl_start + 2 * i];
            let s = tbl[tbl_start + 2 * i + 1];
            let x0 = values[in_start + 2 * i];
            let x1 = values[in_start + 2 * i + 1];
            // out[2i]   = c·x0 − s·x1 = c.mul_add(x0, -(s·x1))
            // out[2i+1] = s·x0 + c·x1 = s.mul_add(x0,  c·x1)
            out[out_start + 2 * i] = c.mul_add(x0, -(s * x1));
            out[out_start + 2 * i + 1] = s.mul_add(x0, c * x1);
        }
    }
}

/// Fast batch inverse output rotation using a precomputed cos/sin table.
///
/// For each query `i` at position `positions[i]`, computes
/// `out[i] = R_{−positions[i]} · aggregated[i]`. The inverse rotation uses
/// `(cos, −sin)` from the forward table — no separate inverse table needed.
///
/// **Panics** if any `positions[t] >= table.max_pos()`.
///
/// **Bit-identical to [`batch_inverse_rotate_output_into`]?** No — the
/// scalar inverse path calls `apply_at(-n)` which recomputes `cos(-angle)`
/// / `sin(-angle)`, while this fast path uses `cos(angle)` / `sin(angle)`
/// from the forward table and negates the sin algebraically. Library
/// `cosf` / `sinf` are not guaranteed to be even/odd functions in the last
/// bit, so the two paths differ by ≤1 ULP on some inputs. The G8 test
/// verifies the inverse direction with tol 1e-6 (matching Phase 2 G1's
/// round-trip ULP budget).
pub fn batch_inverse_rotate_output_into_fast(
    table: &RoVeRotationTable,
    positions: &[usize],
    aggregated: &[f32],
    out: &mut [f32],
) {
    let dim = table.dim();
    debug_assert_eq!(
        aggregated.len(),
        positions.len() * dim,
        "aggregated buffer length mismatch"
    );
    debug_assert_eq!(
        out.len(),
        positions.len() * dim,
        "out buffer length mismatch"
    );
    let half = dim / 2;
    let tbl = table.as_slice();

    for (t, &pos) in positions.iter().enumerate() {
        assert!(
            pos < table.max_pos(),
            "position {pos} >= table.max_pos() {}",
            table.max_pos()
        );
        let in_start = t * dim;
        let out_start = t * dim;
        let tbl_start = pos * dim;

        // Inverse rotation: negate the sin component.
        // (c, s) → (c, −s): out[2i] = c·x0 − (−s)·x1 = c·x0 + s·x1,
        //                   out[2i+1] = (−s)·x0 + c·x1 = −s·x0 + c·x1.
        // Written as mul_add: out[2i]   = c.mul_add(x0, s·x1)
        //                     out[2i+1] = c.mul_add(x1, -(s·x0))
        for i in 0..half {
            let c = tbl[tbl_start + 2 * i];
            let s = tbl[tbl_start + 2 * i + 1];
            let x0 = aggregated[in_start + 2 * i];
            let x1 = aggregated[in_start + 2 * i + 1];
            out[out_start + 2 * i] = c.mul_add(x0, s * x1);
            out[out_start + 2 * i + 1] = c.mul_add(x1, -(s * x0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DIM: usize = 8;
    const TEST_THETA: f32 = 10000.0;

    fn make_action() -> RopeAction {
        RoVeConfig::new(TEST_THETA).build_rope_action(TEST_DIM)
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (tol {tol})"
            );
        }
    }

    // ── G1: identity at pos 0 ────────────────────────────────────

    /// Rotation by angle 0 is identity: `rotate_values_into(action, 0, v, out)`
    /// writes `v` to `out`.
    #[test]
    fn g1_identity_at_pos_zero() {
        let action = make_action();
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; TEST_DIM];
        rotate_values_into(&action, 0, &v, &mut out);
        approx_eq(&v, &out, 1e-6);
    }

    /// Same for the inverse direction.
    #[test]
    fn g1_inverse_identity_at_pos_zero() {
        let action = make_action();
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; TEST_DIM];
        inverse_rotate_output_into(&action, 0, &v, &mut out);
        approx_eq(&v, &out, 1e-6);
    }

    // ── G2: round-trip ───────────────────────────────────────────

    /// `rotate_values_into` then `inverse_rotate_output_into` at the same
    /// position recovers the original (to f32 precision).
    #[test]
    fn g2_round_trip() {
        let action = make_action();
        let v = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let mut rotated = [0.0f32; TEST_DIM];
        let mut recovered = [0.0f32; TEST_DIM];

        for pos in [0usize, 1, 5, 17, 100, 1023] {
            rotate_values_into(&action, pos, &v, &mut rotated);
            inverse_rotate_output_into(&action, pos, &rotated, &mut recovered);
            approx_eq(&v, &recovered, 1e-5);
        }
    }

    // ── G3: relativity check (the paper Eq. 3 claim) ─────────────

    /// The two-step composition (rotate at `j`, inverse at `i`) produces
    /// `R_{j−i} · v` — equivalent to a single `action.apply_at((j − i), v, out)`.
    /// This verifies the offset-indexed kernel `ψ_{j−i} = R_{j−i} · W_V` claim.
    #[test]
    fn g3_relativity_check() {
        let action = make_action();
        let v = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let mut v_at_j = [0.0f32; TEST_DIM];
        let mut v_at_i = [0.0f32; TEST_DIM];
        let mut expected = [0.0f32; TEST_DIM];

        // Test several (i, j) pairs with distinct offsets.
        for (i, j) in [(0usize, 5), (3, 10), (10, 3), (7, 50), (50, 7)] {
            rotate_values_into(&action, j, &v, &mut v_at_j);
            inverse_rotate_output_into(&action, i, &v_at_j, &mut v_at_i);

            // Expected: R_{j-i} · v  (note: offset can be negative when i > j)
            action.apply_at(j as f32 - i as f32, &v, &mut expected);

            approx_eq(&v_at_i, &expected, 1e-5);
        }
    }

    // ── G4: zero-degradation when feature off ────────────────────

    // G4 is architecturally unverifiable from within the feature-gated module:
    // when `rotary_value_embedding` is off, this module does not compile, so
    // no test inside it can run. The guarantee is structural (the feature gate
    // on the module declaration in lib.rs) and is verified by the Phase 1 Exit
    // Criterion: `cargo build` (default features) is unchanged. No test here.

    // ── G5: zero-allocation ──────────────────────────────────────

    /// G5 zero-alloc assertion via code-inspection pattern (matches
    /// `phase_rotation.rs` — we can't use `#[global_allocator]` in lib unit
    /// tests due to parallel test harness collisions). The empirical
    /// `CountingAllocator` audit lives in the Phase 2 bench
    /// (`benches/rotary_value_embedding_goat.rs`).
    ///
    /// This test constructs the primitives, runs both hot paths + both batch
    /// paths, and verifies by code inspection that no `Vec::new`, `vec![]`,
    /// `Vec::clone`, or `.resize` appears on the hot path. The smoke assertion
    /// here is that the functions run without panic on caller-owned buffers.
    #[test]
    fn g5_zero_alloc_smoke() {
        let action = make_action();
        let n = 16;
        let positions: Vec<usize> = (0..n).collect();
        let values = vec![0.5f32; n * TEST_DIM];
        let mut out = vec![0.0f32; n * TEST_DIM];
        let mut recovered = vec![0.0f32; n * TEST_DIM];

        // Both batch paths run without panic on caller-owned buffers.
        batch_rotate_values_into(&action, &positions, &values, &mut out, TEST_DIM);
        batch_inverse_rotate_output_into(&action, &positions, &out, &mut recovered, TEST_DIM);

        // Round-trip recovers the original.
        approx_eq(&values, &recovered, 1e-5);

        // Single-token paths also run on caller-owned buffers.
        let v = [1.0f32; TEST_DIM];
        let mut single_out = [0.0f32; TEST_DIM];
        let mut single_recovered = [0.0f32; TEST_DIM];
        rotate_values_into(&action, 5, &v, &mut single_out);
        inverse_rotate_output_into(&action, 5, &single_out, &mut single_recovered);
        approx_eq(&v, &single_recovered, 1e-5);
    }

    // ── G6: batch correctness ────────────────────────────────────

    /// `batch_rotate_values_into` produces identical results to per-token
    /// `rotate_values_into` for every token. Same for the inverse batch.
    #[test]
    fn g6_batch_correctness() {
        let action = make_action();
        let n = 32;
        let positions: Vec<usize> = (0..n).map(|t| t * 3 + 1).collect();
        let values: Vec<f32> = (0..n * TEST_DIM).map(|i| (i as f32) * 0.01 - 1.0).collect();

        // Batch path.
        let mut batch_out = vec![0.0f32; n * TEST_DIM];
        batch_rotate_values_into(&action, &positions, &values, &mut batch_out, TEST_DIM);

        // Per-token reference.
        let mut ref_out = vec![0.0f32; n * TEST_DIM];
        for (t, &pos) in positions.iter().enumerate() {
            let start = t * TEST_DIM;
            rotate_values_into(
                &action,
                pos,
                &values[start..start + TEST_DIM],
                &mut ref_out[start..start + TEST_DIM],
            );
        }
        approx_eq(&batch_out, &ref_out, 0.0); // bit-identical expected

        // Same for inverse batch.
        let mut batch_inv = vec![0.0f32; n * TEST_DIM];
        batch_inverse_rotate_output_into(&action, &positions, &batch_out, &mut batch_inv, TEST_DIM);

        let mut ref_inv = vec![0.0f32; n * TEST_DIM];
        for (t, &pos) in positions.iter().enumerate() {
            let start = t * TEST_DIM;
            inverse_rotate_output_into(
                &action,
                pos,
                &batch_out[start..start + TEST_DIM],
                &mut ref_inv[start..start + TEST_DIM],
            );
        }
        approx_eq(&batch_inv, &ref_inv, 0.0);
    }

    // ── G7: odd-dim safety ───────────────────────────────────────

    /// RoPE requires even `dim`. `RoVeConfig::build_rope_action` delegates to
    /// `RopeAction::with_theta`, which panics on odd dim. This test verifies
    /// the panic propagates through the RoVE API.
    #[test]
    #[should_panic(expected = "RoPE requires even dim >= 2")]
    fn g7_odd_dim_panics() {
        let config = RoVeConfig::default();
        let _ = config.build_rope_action(7); // odd dim
    }

    /// Dim 0 also panics (must be >= 2).
    #[test]
    #[should_panic(expected = "RoPE requires even dim >= 2")]
    fn g7_zero_dim_panics() {
        let config = RoVeConfig::default();
        let _ = config.build_rope_action(0);
    }

    /// Dim 2 (minimum valid) works.
    #[test]
    fn g7_min_valid_dim() {
        let config = RoVeConfig::default();
        let action = config.build_rope_action(2);
        let v = [1.0, 0.0];
        let mut out = [0.0f32; 2];
        rotate_values_into(&action, 1, &v, &mut out);
        // Non-trivial rotation at pos 1.
        assert!((out[0] - v[0]).abs() > 0.01 || (out[1] - v[1]).abs() > 0.01);
    }

    // ── G8: fast batch path — bit-identical to scalar (Plan 557 Phase 3) ──

    /// The fast batch path produces bit-identical output to the scalar batch
    /// path for the FORWARD direction. The INVERSE direction differs by ≤1
    /// ULP because the scalar path calls `cos(-angle)` / `sin(-angle)`
    /// (negating the angle before the transcendental), while the fast path
    /// uses `cos(angle)` / `sin(angle)` from the forward table and negates
    /// algebraically (`-s` instead of `sin(-angle)`). Library `cosf` / `sinf`
    /// are not guaranteed to be even/odd functions in the last bit, so the
    /// two formulations produce a 1-ULP difference on some inputs. This is
    /// the same ULP-floor class as Phase 2 G1's round-trip budget (1e-6).
    #[test]
    fn g8_fast_batch_bit_identical_to_scalar() {
        let dim = 32usize;
        let theta = 10000.0f32;
        let n = 64usize;
        let max_pos = n;
        let action = RoVeConfig::new(theta).build_rope_action(dim);
        let table = RoVeRotationTable::new(dim, theta, max_pos);

        // Deterministic pseudo-random input.
        let positions: Vec<usize> = (0..n).collect();
        let values: Vec<f32> = (0..n * dim)
            .map(|i| ((i as f32) * 0.0123).sin() * 0.5)
            .collect();

        // Scalar path.
        let mut scalar_out = vec![0.0f32; n * dim];
        batch_rotate_values_into(&action, &positions, &values, &mut scalar_out, dim);

        // Fast path.
        let mut fast_out = vec![0.0f32; n * dim];
        batch_rotate_values_into_fast(&table, &positions, &values, &mut fast_out);

        // Forward direction: bit-identical (both use positive angle, same cos/sin).
        approx_eq(&scalar_out, &fast_out, 0.0);

        // Inverse direction: ≤1 ULP difference (cos(-x) vs cos(x) floor).
        let mut scalar_inv = vec![0.0f32; n * dim];
        batch_inverse_rotate_output_into(&action, &positions, &scalar_out, &mut scalar_inv, dim);

        let mut fast_inv = vec![0.0f32; n * dim];
        batch_inverse_rotate_output_into_fast(&table, &positions, &fast_out, &mut fast_inv);

        // Budget 1e-6 matches Phase 2 G1's round-trip budget — the 1-ULP floor
        // from library transcendental even/odd asymmetry.
        approx_eq(&scalar_inv, &fast_inv, 1e-6);
    }

    /// The fast path round-trips: rotate then inverse-rotate recovers the
    /// original (to f32 precision, same as the scalar path G2).
    #[test]
    fn g8_fast_batch_round_trip() {
        let dim = 16usize;
        let theta = 10000.0f32;
        let n = 32usize;
        let table = RoVeRotationTable::new(dim, theta, n);

        let positions: Vec<usize> = (0..n).collect();
        let values: Vec<f32> = (0..n * dim)
            .map(|i| ((i as f32) * 0.07).sin() * 0.3)
            .collect();

        let mut rotated = vec![0.0f32; n * dim];
        batch_rotate_values_into_fast(&table, &positions, &values, &mut rotated);

        let mut recovered = vec![0.0f32; n * dim];
        batch_inverse_rotate_output_into_fast(&table, &positions, &rotated, &mut recovered);

        approx_eq(&values, &recovered, 1e-5);
    }

    /// Position 0 is identity in the fast path too (cos=1, sin=0).
    #[test]
    fn g8_fast_batch_identity_at_pos_zero() {
        let dim = 8usize;
        let table = RoVeRotationTable::new(dim, 10000.0, 4);
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; dim];
        batch_rotate_values_into_fast(&table, &[0], &v, &mut out);
        approx_eq(&v, &out, 1e-6);
    }

    /// Out-of-range position panics (defensive bound check).
    #[test]
    #[should_panic(expected = "position 10 >= table.max_pos() 5")]
    fn g8_fast_batch_pos_out_of_range_panics() {
        let table = RoVeRotationTable::new(8, 10000.0, 5);
        let v = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 8];
        batch_rotate_values_into_fast(&table, &[10], &v, &mut out);
    }

    /// Odd dim panics at table construction (delegates to RopeAction).
    #[test]
    #[should_panic(expected = "RoPE requires even dim >= 2")]
    fn g8_fast_table_odd_dim_panics() {
        let _ = RoVeRotationTable::new(7, 10000.0, 4);
    }
}
