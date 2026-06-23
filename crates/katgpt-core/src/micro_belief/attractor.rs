//! Family A — attractor loop kernel.
//!
//! `s_t = σ(W_s · s_{t-1} + W_x · x_t + b)`
//!
//! This is the GOAT candidate of Plan 276. Unlike HLA's leaky integrator
//! (`ReconstructionState::evolve_hla`, Family C), the attractor update has
//! fixed-point basins: beliefs exhibit hysteresis — they resist noise until
//! evidence accumulates, then flip. Whether this reduces long-horizon
//! flip-flops vs HLA on a coherence benchmark is the G2.1 GOAT gate.
//!
//! # State range choice
//!
//! We store the belief vector in `(−1, 1)` via `state[i] = 2·σ(·) − 1`, NOT in
//! `(0, 1)` (the raw sigmoid output). This matches the existing `evolve_hla`
//! state range of `[-1, 1]` (see `reconstruction.rs` L646:
//! `self.hla[i] = (...).clamp(-1.0, 1.0)`), so:
//!   - both Family A and Family C kernels can be benchmarked against the same
//!     bridge direction vectors,
//!   - the G2.1 coherence benchmark compares apples to apples (same scalar
//!     range, same projection geometry),
//!   - the AGENTS.md rule that scalar bridges use `sigmoid(dot)` is preserved
//!     (the bridge is applied to the `(-1,1)` state; the bridge's own sigmoid
//!     then maps to `(0,1)` for the synced scalar — the two sigmoids compose
//!     cleanly because both are monotone).
//!
//! The post-update clamp at `±clamp` (default `6.0`) is a no-op safety net for
//! the `(−1, 1)` range — kept so future unbounded-activation families can reuse
//! the same `step()` scaffolding without changing the contract.
//!
//! # Weight layout (R5 mitigation)
//!
//! Generic const exprs (`[f32; DIM]`) are not stable, so weights are stored as
//! row-major `Vec<f32>` of length `dim*dim`. Row `i` is `ws[i*dim .. (i+1)*dim]`.
//! Performance impact at `dim = 32` is negligible: the matvec is 32 dot
//! products of length 32, each dispatched to `simd_dot_f32` which auto-vectorises.

use crate::micro_belief::bridge::project_to_scalars as bridge_project;
use crate::micro_belief::types::{KernelConfig, MicroRecurrentBeliefState, RecurrenceFamily};
use crate::micro_belief::{assume_init_slice, uninit_stack};
use crate::simd::simd_dot_f32;

#[cfg(not(feature = "simd_sigmoid"))]
use crate::simd::fast_sigmoid;

#[cfg(feature = "simd_sigmoid")]
use crate::simd::simd_sigmoid_tanh_clamp_inplace;

/// Static zero buffer for the `q` argument of `simd_sigmoid_tanh_clamp_inplace`
/// when `step()` calls it (q is always zero in step — there is no noise query).
/// 1024 f32 matches the max `dim` supported by the stack buffers.
#[cfg(feature = "simd_sigmoid")]
static ZERO_BUF: [f32; 1024] = [0.0; 1024];

/// Family A attractor kernel: `s_t = 2·σ(W_s·s + W_x·x + b) − 1`.
///
/// See the module-level docs for the state-range choice and weight layout.
///
/// Construct via [`from_seed`](Self::from_seed) (deterministic) and tune via
/// the builder methods. The kernel is frozen after construction — callers MUST
/// NOT mutate `ws` / `wx` / `b` (they are `pub` only for snapshot serialisation
/// convenience; the snapshot path reads them immutably).
#[derive(Clone, Debug)]
pub struct AttractorKernel {
    /// Recurrent weight matrix `W_s`, row-major `dim*dim`.
    pub ws: Vec<f32>,
    /// Input weight matrix `W_x`, row-major `dim*dim`.
    pub wx: Vec<f32>,
    /// Bias vector `b`, length `dim`.
    pub b: Vec<f32>,
    /// Belief-vector dimension.
    pub dim: usize,
    /// Post-activation clamp magnitude (default `6.0`; no-op for the default
    /// `(-1,1)` state range).
    pub clamp: f32,
    /// Seed used to initialise weights (retained for snapshot provenance).
    pub seed: u64,
}

impl AttractorKernel {
    /// Construct a deterministically-initialised attractor kernel.
    ///
    /// Weights are drawn from `fastrand::Rng::with_seed(seed)` in the range
    /// `[-1/sqrt(dim), 1/sqrt(dim)]` (Xavier-like init scaled for a tanh-range
    /// forward pass; keeps the pre-sigmoid activation in a reasonable range so
    /// the kernel doesn't saturate to a fixed point on step 1).
    ///
    /// # Determinism (G1.1)
    ///
    /// Same `seed` + same `dim` always produces bit-identical weights, hence
    /// bit-identical `s_T` for the same input sequence. `fastrand::Rng` is
    /// deterministic and platform-independent by construction.
    pub fn from_seed(seed: u64, dim: usize) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let scale = 1.0 / (dim as f32).sqrt();
        let ws: Vec<_> = (0..dim * dim).map(|_| rng.f32() * 2.0 - 1.0).collect();
        // Multiply by scale at init time so the hot path has one fewer mul.
        let ws: Vec<f32> = ws.into_iter().map(|v| v * scale).collect();
        let wx: Vec<_> = (0..dim * dim).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let wx: Vec<f32> = wx.into_iter().map(|v| v * scale).collect();
        // Bias starts at zero — neutral fixed point at the origin.
        let b = vec![0.0; dim];
        Self {
            ws,
            wx,
            b,
            dim,
            clamp: 6.0,
            seed,
        }
    }

    /// Construct from an explicit [`KernelConfig`].
    pub fn from_config(config: &KernelConfig) -> Self {
        let mut k = Self::from_seed(config.seed, config.dim);
        k.clamp = config.clamp;
        k
    }

    /// Builder: override the post-activation clamp magnitude.
    #[inline]
    pub fn with_clamp(mut self, clamp: f32) -> Self {
        self.clamp = clamp;
        self
    }

    /// Precompute `W_x · x` into `out[..dim]`.
    ///
    /// Used by [`step_with_precomputed_wx`](Self::step_with_precomputed_wx) to
    /// avoid recomputing the input matvec across K iterations in
    /// [`LatentThoughtKernel`](super::latent_thought::LatentThoughtKernel) —
    /// the input `x` is invariant across K inner iterations, only the state
    /// `s` changes.
    ///
    /// # Bit-identical to the `dot_wx` inside `step()`
    ///
    /// Uses the same `simd_dot_f32` reductions in the same chunked-4 outer-loop
    /// pattern as `step()`, so `out[i]` is bit-identical to the `dot_wx`
    /// computed inside [`step()`](Self::step). This preserves the G1.6
    /// K=1-reduces-to-Family-A property.
    #[inline]
    pub fn precompute_wx_dot(&self, input: &[f32], out: &mut [f32]) {
        debug_assert_eq!(input.len(), self.dim, "input/dim mismatch");
        debug_assert!(out.len() >= self.dim, "out too short for dim");
        let dim = self.dim;
        let mut i = 0usize;
        while i + 4 <= dim {
            let wx_r0 = &self.wx[i * dim..(i + 1) * dim];
            let wx_r1 = &self.wx[(i + 1) * dim..(i + 2) * dim];
            let wx_r2 = &self.wx[(i + 2) * dim..(i + 3) * dim];
            let wx_r3 = &self.wx[(i + 3) * dim..(i + 4) * dim];
            out[i] = simd_dot_f32(input, wx_r0, dim);
            out[i + 1] = simd_dot_f32(input, wx_r1, dim);
            out[i + 2] = simd_dot_f32(input, wx_r2, dim);
            out[i + 3] = simd_dot_f32(input, wx_r3, dim);
            i += 4;
        }
        while i < dim {
            let wx_row = &self.wx[i * dim..(i + 1) * dim];
            out[i] = simd_dot_f32(input, wx_row, dim);
            i += 1;
        }
    }

    /// Advance one tick using a precomputed `W_x · x` buffer (from
    /// [`precompute_wx_dot`](Self::precompute_wx_dot)). Skips the input matvec
    /// — saves ~50ns per call at dim=32 (eliminates 32 `simd_dot_f32` calls).
    ///
    /// Used by `LatentThoughtKernel::step` for K > 1: precompute `W_x · x`
    /// once, then call this K times with the changing `state` and the same
    /// precomputed input.
    ///
    /// # Bit-identical to `step()`
    ///
    /// Addition order is preserved: `next[i] = (dot_ws + wx_x[i]) + b[i]`,
    /// matching `step()`'s `(dot_ws + dot_wx) + b[i]` when `wx_x[i]` was
    /// produced by `precompute_wx_dot` (same `simd_dot_f32` reduction).
    #[inline]
    pub fn step_with_precomputed_wx(&self, state: &mut [f32], wx_x: &[f32]) {
        debug_assert_eq!(state.len(), self.dim, "state/dim mismatch");
        debug_assert_eq!(wx_x.len(), self.dim, "wx_x/dim mismatch");
        let dim = self.dim;
        let clamp = self.clamp;

        let mut next_buf = uninit_stack::<1024>();
        // SAFETY: next[..dim] is fully written by the matvec below before any read.
        let next: &mut [f32] = unsafe { assume_init_slice(&mut next_buf, dim) };

        let mut i = 0usize;
        while i + 4 <= dim {
            let ws_r0 = &self.ws[i * dim..(i + 1) * dim];
            let ws_r1 = &self.ws[(i + 1) * dim..(i + 2) * dim];
            let ws_r2 = &self.ws[(i + 2) * dim..(i + 3) * dim];
            let ws_r3 = &self.ws[(i + 3) * dim..(i + 4) * dim];
            let dot_ws_0 = simd_dot_f32(state, ws_r0, dim);
            let dot_ws_1 = simd_dot_f32(state, ws_r1, dim);
            let dot_ws_2 = simd_dot_f32(state, ws_r2, dim);
            let dot_ws_3 = simd_dot_f32(state, ws_r3, dim);
            // Same addition order as step(): (dot_ws + dot_wx) + b.
            next[i] = (dot_ws_0 + wx_x[i]) + self.b[i];
            next[i + 1] = (dot_ws_1 + wx_x[i + 1]) + self.b[i + 1];
            next[i + 2] = (dot_ws_2 + wx_x[i + 2]) + self.b[i + 2];
            next[i + 3] = (dot_ws_3 + wx_x[i + 3]) + self.b[i + 3];
            i += 4;
        }
        while i < dim {
            let ws_row = &self.ws[i * dim..(i + 1) * dim];
            let dot_ws = simd_dot_f32(state, ws_row, dim);
            next[i] = (dot_ws + wx_x[i]) + self.b[i];
            i += 1;
        }

        // Apply the state-writeback chain: (2·σ(·) − 1).clamp(−clamp, clamp).
        #[cfg(feature = "simd_sigmoid")]
        {
            let ptr = next.as_mut_ptr();
            let a_slice: &[f32] = unsafe { core::slice::from_raw_parts(ptr, dim) };
            let out_slice: &mut [f32] = unsafe { core::slice::from_raw_parts_mut(ptr, dim) };
            simd_sigmoid_tanh_clamp_inplace(out_slice, a_slice, &ZERO_BUF[..dim], clamp);
        }
        #[cfg(not(feature = "simd_sigmoid"))]
        {
            for j in 0..dim {
                next[j] = (2.0 * fast_sigmoid(next[j]) - 1.0).clamp(-clamp, clamp);
            }
        }

        state[..dim].copy_from_slice(&next[..dim]);
    }

    /// Serialise the weights to a flat little-endian byte blob for snapshot.
    ///
    /// Layout: `ws (dim*dim*4) || wx (dim*dim*4) || b (dim*4)`, all f32 in
    /// native little-endian. The snapshot module computes BLAKE3 over this
    /// exact byte sequence — any change to the layout MUST bump the snapshot
    /// version AND be reflected in `MicroRecurrentKernelSnapshot::commit`.
    ///
    /// This is NOT on the hot path; snapshots are rare (per-NPC personality
    /// version events).
    pub fn to_snapshot_blob(&self) -> Vec<u8> {
        let total_bytes =
            (self.ws.len() + self.wx.len() + self.b.len()) * core::mem::size_of::<f32>();
        let mut out = Vec::with_capacity(total_bytes);
        for &v in &self.ws {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.wx {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.b {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Recursively advance the kernel for `inputs.len()` ticks and classify the
    /// resulting belief-vector chain with [`crate::classify_chain`].
    ///
    /// The chain `s_0, s_1, …, s_k` (where `s_0 = initial_state` and
    /// `k = inputs.len()`) is captured into a flattened buffer and classified.
    /// Each input `inputs[t]` drives one [`MicroRecurrentBeliefState::step`]
    /// invocation producing `s_{t+1}`.
    ///
    /// **Zero per-step allocation** — double-buffered `s_a` / `s_b` plus a
    /// single up-front `Vec::with_capacity` for the chain. The depth-invariance
    /// `Scratch` is allocated inside this call; tight-loop callers should reuse
    /// one via the raw [`crate::classify_chain`] primitive.
    ///
    /// # Plan 306 Phase 4 (G3 — negative control)
    ///
    /// The attractor update applies `(2·σ(·) − 1).clamp(±clamp)`, bounding
    /// magnitude per construction. We therefore **expect** the diagnostic to
    /// classify as [`DepthInvariant`] on this kernel — the negative control
    /// confirming the classifier does not false-positive on healthy kernels.
    /// The matching `leaky` audit (or an unclamped inline leaky in the test)
    /// provides the positive control.
    ///
    /// [`DepthInvariant`]: crate::DepthInvarianceKind::DepthInvariant
    #[cfg(feature = "depth_invariance")]
    pub fn audit_depth_invariance(
        &self,
        initial_state: &[f32],
        inputs: &[&[f32]],
        cfg: &crate::DepthInvarianceConfig,
    ) -> crate::DepthInvarianceDiagnostic {
        let dim = self.dim;
        assert_eq!(initial_state.len(), dim, "initial_state must have length dim");
        for (i, inp) in inputs.iter().enumerate() {
            assert_eq!(inp.len(), dim, "inputs[{i}] must have length dim");
        }

        let k = inputs.len();
        let k_plus_1 = k + 1;

        let mut chain: Vec<f32> = Vec::with_capacity(k_plus_1 * dim);
        chain.extend_from_slice(initial_state);

        // Double-buffered state — `step()` writes in place, so we copy s_a → s_b
        // then step s_b, then swap. This is the only way to capture s_t and s_{t+1}
        // in the same chain without aliasing step()'s read+write on `state`.
        let mut s_a: Vec<f32> = initial_state.to_vec();
        let mut s_b: Vec<f32> = initial_state.to_vec();

        for inp in inputs {
            s_b.copy_from_slice(&s_a);
            MicroRecurrentBeliefState::step(self, &mut s_b, inp);
            chain.extend_from_slice(&s_b);
            std::mem::swap(&mut s_a, &mut s_b);
        }

        let mut scratch = crate::Scratch::with_capacity(k_plus_1, dim);
        crate::classify_chain(&chain, dim, cfg, &mut scratch)
    }
}

impl MicroRecurrentBeliefState for AttractorKernel {
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    /// Advance one tick: `state[i] = clamp(2·σ(W_s[i]·s + W_x[i]·x + b[i]) − 1, ±clamp)`.
    ///
    /// # Hot-path properties
    ///
    /// - Zero allocation: writes in-place to `state`.
    /// - Deterministic: `simd_dot_f32` and `fast_sigmoid` are both deterministic
    ///   across runs (G1.1).
    /// - Auto-vectorisable: each row is a length-`dim` dot product dispatched to
    ///   `simd_dot_f32`; the outer loop over `dim` rows is chunkable.
    ///
    /// # Range
    ///
    /// Output is in `(−1, 1)` (modulo the no-op clamp at `±6`). See the
    /// module-level docs for why this range was chosen.
    #[inline]
    fn step(&self, state: &mut [f32], input: &[f32]) {
        debug_assert_eq!(state.len(), self.dim, "state/dim mismatch");
        debug_assert_eq!(input.len(), self.dim, "input/dim mismatch");
        let dim = self.dim;
        let clamp = self.clamp;

        // We must not mutate `state` in-place while reading it for the matvec,
        // because row i reads state[j] for all j (including j != i). The
        // dim=32 f32 vector (128 bytes) fits comfortably on the stack.
        //
        // `next` serves double duty: first as the pre-sigmoid activation
        // buffer (dot_ws + dot_wx + b), then as the output of the
        // sigmoid→tanh→clamp pass. The matvec addition order
        // (dot_ws + dot_wx + b) is preserved bit-for-bit. The SIMD sigmoid
        // helper reads a[i] and writes out[i] in the same 4-wide chunk, so
        // in-place aliasing is safe (read completes before write per chunk).
        // Uninitialized 4KB stack buffer — we overwrite next[..dim] via the
        // matvec before any read (the sigmoid pass). Skipping the O(dim) zero-
        // init saves ~40ns/step at dim=32 (Issue 024). Matches the pattern in
        // `katgpt-rs/src/cumprodsum.rs::segsum`.
        let mut next_buf = uninit_stack::<1024>();
        // SAFETY: we write next[..dim] via the matvec below before any read.
        let next: &mut [f32] = unsafe { assume_init_slice(&mut next_buf, dim) };

        // Process rows in chunks of 4 to give LLVM a clear auto-vec hint on
        // the outer loop. The inner reductions are dispatched to simd_dot_f32.
        // Four independent accumulators per pass hide FMA pipeline latency.
        let mut i = 0;
        while i + 4 <= dim {
            // Row slices for W_s and W_x — computed once per row.
            let ws_r0 = &self.ws[i * dim..(i + 1) * dim];
            let ws_r1 = &self.ws[(i + 1) * dim..(i + 2) * dim];
            let ws_r2 = &self.ws[(i + 2) * dim..(i + 3) * dim];
            let ws_r3 = &self.ws[(i + 3) * dim..(i + 4) * dim];
            let wx_r0 = &self.wx[i * dim..(i + 1) * dim];
            let wx_r1 = &self.wx[(i + 1) * dim..(i + 2) * dim];
            let wx_r2 = &self.wx[(i + 2) * dim..(i + 3) * dim];
            let wx_r3 = &self.wx[(i + 3) * dim..(i + 4) * dim];

            // 4 independent dot products over state, 4 over input — FMA-bound.
            let dot_ws_0 = simd_dot_f32(state, ws_r0, dim);
            let dot_ws_1 = simd_dot_f32(state, ws_r1, dim);
            let dot_ws_2 = simd_dot_f32(state, ws_r2, dim);
            let dot_ws_3 = simd_dot_f32(state, ws_r3, dim);
            let dot_wx_0 = simd_dot_f32(input, wx_r0, dim);
            let dot_wx_1 = simd_dot_f32(input, wx_r1, dim);
            let dot_wx_2 = simd_dot_f32(input, wx_r2, dim);
            let dot_wx_3 = simd_dot_f32(input, wx_r3, dim);

            // Pre-sigmoid activation = W_s·s + W_x·x + b.
            // Addition order MUST stay (dot_ws + dot_wx) + b — do not rearrange
            // (Plan 281 G1.3 bit-identical degeneracy with sample_k_states).
            next[i] = dot_ws_0 + dot_wx_0 + self.b[i];
            next[i + 1] = dot_ws_1 + dot_wx_1 + self.b[i + 1];
            next[i + 2] = dot_ws_2 + dot_wx_2 + self.b[i + 2];
            next[i + 3] = dot_ws_3 + dot_wx_3 + self.b[i + 3];
            i += 4;
        }
        // Tail: remaining rows (dim mod 4).
        while i < dim {
            let ws_row = &self.ws[i * dim..(i + 1) * dim];
            let wx_row = &self.wx[i * dim..(i + 1) * dim];
            let dot_ws = simd_dot_f32(state, ws_row, dim);
            let dot_wx = simd_dot_f32(input, wx_row, dim);
            next[i] = dot_ws + dot_wx + self.b[i];
            i += 1;
        }

        // Apply the state-writeback chain: (2·σ(·) − 1).clamp(−clamp, clamp).
        //
        // Under `simd_sigmoid`: a single fused NEON/AVX2 pass over `dim`
        // elements replaces `dim` scalar fast_sigmoid calls. `next` is both
        // the input (pre-sigmoid activation) and output — safe because the
        // SIMD helper reads a[i] and writes out[i] within the same chunk
        // (the vld1q completes before vst1q per 4-element group). q is zero
        // in step() (no noise query) — ZERO_BUF makes the call signature
        // identical to sample_k_states with zero queries, preserving G1.3.
        //
        // Under the default (scalar) path: per-element fast_sigmoid, bit-for-bit
        // unchanged from the pre-simd_sigmoid implementation.
        #[cfg(feature = "simd_sigmoid")]
        {
            // SAFETY: `next` aliases as both `a` (read) and `out` (write). The
            // SIMD helper processes 4 (NEON) or 8 (AVX2) elements per chunk:
            // it loads a[i..i+W] into a register, computes, then stores to
            // out[i..i+W]. No chunk reads elements written by a later chunk,
            // so the aliasing is data-race-free.
            let len = dim;
            let ptr = next.as_mut_ptr();
            let a_slice: &[f32] = unsafe { core::slice::from_raw_parts(ptr, len) };
            let out_slice: &mut [f32] = unsafe { core::slice::from_raw_parts_mut(ptr, len) };
            simd_sigmoid_tanh_clamp_inplace(out_slice, a_slice, &ZERO_BUF[..dim], clamp);
        }
        #[cfg(not(feature = "simd_sigmoid"))]
        {
            for j in 0..dim {
                next[j] = (2.0 * fast_sigmoid(next[j]) - 1.0).clamp(-clamp, clamp);
            }
        }

        // Write back in one pass — avoids read-after-write hazards on `state`
        // (we read all of `state` to compute each row; mutating in place would
        // poison later rows).
        state[..dim].copy_from_slice(&next[..dim]);
    }

    #[inline(always)]
    fn project_to_scalars(&self, state: &[f32], directions: &[f32], dim: usize, out: &mut [f32]) {
        bridge_project(state, directions, dim, out);
    }

    #[inline]
    fn family(&self) -> RecurrenceFamily {
        RecurrenceFamily::Attractor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_seed_is_deterministic() {
        // Same seed → same weights bit-identically.
        let a = AttractorKernel::from_seed(42, 32);
        let b = AttractorKernel::from_seed(42, 32);
        assert_eq!(a.ws, b.ws);
        assert_eq!(a.wx, b.wx);
        assert_eq!(a.b, b.b);
    }

    #[test]
    fn different_seeds_produce_different_weights() {
        let a = AttractorKernel::from_seed(1, 32);
        let b = AttractorKernel::from_seed(2, 32);
        assert_ne!(a.ws, b.ws, "different seeds must produce different weights");
    }

    #[test]
    fn bias_starts_zero() {
        let k = AttractorKernel::from_seed(42, 32);
        assert!(k.b.iter().all(|&v| v == 0.0), "bias must init to zero");
    }

    #[test]
    fn weights_are_xavier_scaled() {
        // Xavier scale = 1/sqrt(dim). At dim=32, scale ≈ 0.1768.
        // Weights are in [-scale, scale].
        let k = AttractorKernel::from_seed(42, 32);
        let scale = 1.0 / (32.0f32).sqrt();
        for &w in &k.ws {
            assert!(
                w.abs() <= scale + 1e-6,
                "weight {w} exceeds Xavier scale {scale}"
            );
        }
    }

    #[test]
    fn step_writes_into_minus_one_to_one_range() {
        let k = AttractorKernel::from_seed(42, 32);
        let mut state = vec![0.0f32; 32];
        let input = vec![0.5f32; 32];
        for _ in 0..100 {
            k.step(&mut state, &input);
            for &v in &state {
                assert!(v > -1.0001 && v < 1.0001, "state out of (-1,1): {v}");
            }
        }
    }

    #[test]
    fn snapshot_blob_layout_is_stable() {
        // Layout: ws (dim*dim*4) || wx (dim*dim*4) || b (dim*4).
        let k = AttractorKernel::from_seed(42, 8);
        let blob = k.to_snapshot_blob();
        let expected_len = (8 * 8 + 8 * 8 + 8) * 4;
        assert_eq!(blob.len(), expected_len);
        // First 4 bytes should be ws[0] little-endian.
        let ws0_bytes = &blob[0..4];
        let ws0 = f32::from_le_bytes([ws0_bytes[0], ws0_bytes[1], ws0_bytes[2], ws0_bytes[3]]);
        assert_eq!(ws0, k.ws[0]);
    }

    #[test]
    fn family_is_attractor() {
        let k = AttractorKernel::from_seed(42, 32);
        assert_eq!(k.family(), RecurrenceFamily::Attractor);
    }

    #[test]
    fn with_clamp_builder() {
        let k = AttractorKernel::from_seed(42, 32).with_clamp(2.0);
        assert_eq!(k.clamp, 2.0);
    }

    #[test]
    fn from_config_uses_config_values() {
        let cfg = KernelConfig::default()
            .with_dim(16)
            .with_seed(99)
            .with_clamp(1.5);
        let k = AttractorKernel::from_config(&cfg);
        assert_eq!(k.dim, 16);
        assert_eq!(k.seed, 99);
        assert_eq!(k.clamp, 1.5);
    }

    /// **Bit-identical guarantee for `step_with_precomputed_wx`.**
    ///
    /// The precomputed path MUST produce the same state as `step()` for the
    /// same `(state, input)` — otherwise the LatentThoughtKernel K>1 path would
    /// diverge from the K×step() reference. We exercise this over 50 recurrent
    /// steps to catch any accumulation-order drift.
    #[test]
    fn step_with_precomputed_wx_matches_step_bit_identical() {
        let k = AttractorKernel::from_seed(42, 32);
        // Non-trivial input (not constant — exercises all rows).
        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01 - 0.15).collect();
        let mut state_direct = vec![0.0f32; 32];
        let mut state_precomputed = vec![0.0f32; 32];
        // Precomputed scratch (only the first dim=32 elements are used).
        let mut wx_x = [0.0f32; 1024];

        for _step in 0..50 {
            // Direct path.
            k.step(&mut state_direct, &input);
            // Precomputed path.
            k.precompute_wx_dot(&input, &mut wx_x[..32]);
            k.step_with_precomputed_wx(&mut state_precomputed, &wx_x[..32]);
        }
        assert_eq!(
            state_direct, state_precomputed,
            "precompute + step_with_precomputed_wx must be bit-identical to step()"
        );
    }
}
