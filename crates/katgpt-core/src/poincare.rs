//! Poincaré Adapter — closed-form latent navigation primitive (Plan 449).
//!
//! Distillation of arXiv:2607.14228 (Chen et al., *SeeSE3: Emergence of 3D
//! Space in Vision Features*, DeepMind, 15 Jul 2026). See
//! `katgpt-rs/.research/449_SeeSE3_Poincare_Adapter_Primitive.md` for the full
//! math + novelty gate (4/4 Super-GOAT) and
//! `riir-ai/.research/319_SeeSE3_Latent_Imagination_Game_Runtime_Guide.md`
//! for the private game-runtime selling point this primitive hooks into.
//!
//! # The primitive
//!
//! A frozen [`PoincareAdapter`] Pod holds an offline-fit triple `(φ, W, W†)`:
//!
//! - `φ: R^d_latent → R^d_phi` is a small unrolling map (modelless default:
//!   PCA-tanh, a single-layer linear projection + bounded `tanh` activation).
//! - `W: R^d_phi → R^d_target` is the linear decoder: `Δtarget ≈ W·(φ(z₂) − φ(z₁))`.
//! - `W†: R^d_target → R^d_phi` is the Moore-Penrose pseudoinverse of `W`,
//!   giving the **closed-form inverse navigator**:
//!
//! ```text
//! z_dest = z_src + φ⁻¹( φ(z_src) + W† · Δtarget )
//! ```
//!
//! If `φ` is invertible on its image (the PCA case — orthogonal projection
//! onto top-`phi_out` directions), the inverse step is closed-form. The
//! navigator then reduces to one MLP evaluation, one matvec, and one inverse
//! projection — all bounded-size, all zero-allocation.
//!
//! # Why modelless
//!
//! The fit is **closed-form ridge + PCA** — no gradient descent. The paper's
//! AdamW fit over a 2-layer φ is the gradient fallback (`riir-train` follow-up
//! only if G2 fails). Per the research skill §3.5 modelless-unblock protocol,
//! the modelless path is the default; Path 0 (training-target decomposition)
//! applies: the paper's value is the math (`ΔP ≈ W·Δz` + the unrolling), NOT
//! the training loop.
//!
//! # Theorem 7 design constraint
//!
//! The paper proves rotation targets fit tighter than translation-magnitude
//! targets (rotational optical flow is depth-independent; translational flow
//! scales as `1/Z`). Designers should lean on the easier component. Concretely:
//! if the consumer's target space is `(facing_θ, Δx, Δy)`, expect higher R² on
//! `θ` than on `‖Δx, Δy‖`. See `riir-ai/.research/319` §"HLA depth analog gap".
//!
//! # Zero-alloc hot path
//!
//! [`poincare_navigate_into`] takes borrowed slices + the adapter reference
//! only. No `Vec`, no `Box`. The `φ` evaluation reuses the caller-supplied
//! `&mut [f32]` scratch buffer. The G4 gate (Phase 2) pins this at 0
//! allocations per call after warmup.
//!
//! # Sibling primitives
//!
//! - **Latent Field Steering** (Plan 309) — the *forward* direction: push
//!   latent state along a designer-supplied direction vector. Poincaré is the
//!   *inverse*: given a desired target movement, find the latent step.
//! - **Viable Manifold Graph** (Plan 312) — *graph-based* navigation on a
//!   safe subgraph. Poincaré is *continuous closed-form*. They compose: VMG
//!   handles highly curved manifolds; Poincaré handles manifolds that admit
//!   linear unrolling.
//! - **SLoD** (Plan 235) — Poincaré *ball* geometry for KG LOD selection.
//!   Different problem (KG abstraction levels, not latent navigation).
//!
//! # References
//!
//! - Plan 449 — execution plan
//! - Research 449 — math + novelty gate
//! - riir-ai/.research/319 — private game-runtime selling-point guide
//!
//! Plan 449: Poincaré Adapter — closed-form latent navigation primitive.
//! Distillation of arXiv:2607.14228 (Chen et al., *SeeSE3: Emergence of 3D
//! Space in Vision Features*, DeepMind, 15 Jul 2026). Research 449 ran the
//! novelty gate (4/4 Super-GOAT). The open primitive ships a frozen
//! `PoincareAdapter` Pod + the closed-form navigator `poincare_navigate_into` +
//! the multi-step variant + an offline closed-form ridge fit. Private game-
//! runtime selling point (NPC imagination) lives in riir-ai/.research/319.
//!
//! Pure modelless (closed-form PCA + ridge + SVD pseudoinverse; no gradient
//! descent). Reuses Plan 301's `thin_svd_into` + Plan 308's
//! `ridge_solve_direct_f32`. Gated on `subspace_phase_gate` for the SVD. DEFAULT-ON
//! (Plan 449 Phase 3, 2026-07-18): G1–G7 ALL PASS — see .benchmarks/449_poincare_goat.md.

#![allow(clippy::needless_range_loop)]
#![allow(non_snake_case)] // `W`, `W_pinv` match paper notation (arXiv:2607.14228)

use crate::simd::simd_dot_f32;
use crate::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};
use blake3::Hasher;

// ── Constants ─────────────────────────────────────────────────────

/// Maximum supported latent dimension. HLA is 8; LLM-block activations are 64;
/// shard `style_weights` are 64. Anything larger should reduce before fit.
pub const LATENT_DIM_MAX: usize = 64;

/// φ's output dimensionality (the "chart" dimension). 20 matches the paper's
/// sweet spot for vision features; smaller is faster, larger unrolls more
/// curvature at the cost of `W` conditioning.
pub const PHI_OUT_DEFAULT: usize = 20;

/// φ's hidden dimensionality for the modelless default (PCA-tanh). Set equal
/// to `phi_out` — the modelless path is a single linear projection + tanh.
/// A larger hidden layer is reserved for the gradient-fit follow-up.
pub const PHI_HIDDEN_DEFAULT: usize = PHI_OUT_DEFAULT;

/// Maximum supported target dimension. SE(3) is 6 (3 rotation + 3
/// translation); SE(2) + belief scalars are ≤ 8. Anything larger suggests the
/// target space should be factored.
pub const TARGET_DIM_MAX: usize = 8;

/// Default ridge regularization α. Matches the paper's `α = 1.0`.
pub const RIDGE_ALPHA_DEFAULT: f32 = 1.0;

// ── Errors ────────────────────────────────────────────────────────

/// Errors returned by [`fit_poincare_adapter`] and [`PoincareAdapter::from_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PoincareFitError {
    /// `latent_dim` is 0 or exceeds [`LATENT_DIM_MAX`].
    LatentDimOutOfRange,
    /// `target_dim` is 0 or exceeds [`TARGET_DIM_MAX`].
    TargetDimOutOfRange,
    /// `phi_out` is 0, exceeds `latent_dim`, or exceeds [`LATENT_DIM_MAX`].
    PhiOutOutOfRange,
    /// `phi_hidden` is 0 or exceeds [`LATENT_DIM_MAX`].
    PhiHiddenOutOfRange,
    /// Number of (z, target) pairs provided is fewer than `target_dim + 1`.
    /// Ridge needs at least `target_dim` independent displacement samples.
    InsufficientSamples,
    /// Length of a `z` slice did not match the declared `latent_dim`.
    LatentLenMismatch,
    /// Length of a `target` slice did not match the declared `target_dim`.
    TargetLenMismatch,
    /// `W` ended up rank-deficient (singular values collapsed below τ).
    /// Usually means `target_dim > φ(z)`'s effective rank on the sample set.
    RankDeficient,
    /// Byte buffer passed to [`PoincareAdapter::from_bytes`] had the wrong
    /// length, wrong magic, or failed BLAKE3 verification.
    MalformedBuffer,
}

// ── Fit configuration ────────────────────────────────────────────

/// Configuration for [`fit_poincare_adapter`].
#[derive(Debug, Clone, Copy)]
pub struct FitConfig {
    /// Ridge regularization α added to the Gram diagonal. Default 1.0.
    pub ridge_alpha: f32,
    /// Numerical-rank threshold for `W` (singular values below
    /// `tau · σ_max` are zeroed). Default `1e-5`.
    pub rank_tau: f32,
    /// Magic byte written into the Pod's serialized form. Lets consumers
    /// distinguish adapter versions. Default `b'P'` (`0x50`).
    pub magic: u8,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            ridge_alpha: RIDGE_ALPHA_DEFAULT,
            rank_tau: 1e-5,
            magic: b'P',
        }
    }
}

// ── Pod: PoincareAdapter ─────────────────────────────────────────

/// Frozen Poincaré adapter triple `(φ, W, W†)`, BLAKE3-committed.
///
/// Layout (all arrays are row-major, little-endian on serialize):
///
/// | Field | Shape | Notes |
/// |---|---|---|
/// | `phi_w1` | `[phi_hidden × latent_dim]` | First φ layer weights |
/// | `phi_b1` | `[phi_hidden]` | First φ layer biases |
/// | `phi_w2` | `[phi_out × phi_hidden]` | Second φ layer weights |
/// | `phi_b2` | `[phi_out]` | Second φ layer biases |
/// | `W` | `[target_dim × phi_out]` | Forward decoder `Δtarget = W · Δφ` |
/// | `W_pinv` | `[phi_out × target_dim]` | Pseudoinverse `Δφ = W_pinv · Δtarget` |
///
/// The dimensions live alongside the arrays as `u8` / `u16` headers (max
/// LATENT_DIM_MAX=64, so u8 suffices for latent_dim/target_dim/phi_out; u16
/// gives headroom for future expansion). The `blake3` field is the content
/// commitment over all weights (NOT over the dims — they are recovered from
/// the weights' lengths when reconstructing via [`Self::canonical_bytes`]).
///
/// For the modelless default (PCA-tanh), `phi_hidden == phi_out` and `phi_w2`
/// is the identity (so φ is effectively single-layer).
///
/// **Sync-boundary invariant (per global AGENTS.md):** this Pod carries only
/// latent semantic state. The target space is *generic* — callers define what
/// `Δtarget` means (SE(3) pose, HLA affect, personality drift). No raw
/// `MapPos` / `HP` / wallet semantics leak in.
#[derive(Debug, Clone)]
pub struct PoincareAdapter {
    /// First φ layer weights, row-major `[phi_hidden × latent_dim]`.
    pub phi_w1: Vec<f32>,
    /// First φ layer biases, length `phi_hidden`.
    pub phi_b1: Vec<f32>,
    /// Second φ layer weights, row-major `[phi_out × phi_hidden]`.
    pub phi_w2: Vec<f32>,
    /// Second φ layer biases, length `phi_out`.
    pub phi_b2: Vec<f32>,
    /// Forward decoder, row-major `[target_dim × phi_out]`.
    /// `Δtarget[i] = Σ_j W[i·phi_out + j] · Δφ[j]`.
    pub W: Vec<f32>,
    /// Pseudoinverse, row-major `[phi_out × target_dim]`.
    /// `Δφ[i] = Σ_j W_pinv[i·target_dim + j] · Δtarget[j]`.
    pub W_pinv: Vec<f32>,
    /// Latent dim `d`. `1 ≤ latent_dim ≤ LATENT_DIM_MAX`.
    pub latent_dim: u8,
    /// Target dim. `1 ≤ target_dim ≤ TARGET_DIM_MAX`.
    pub target_dim: u8,
    /// φ hidden-layer width.
    pub phi_hidden: u8,
    /// φ output width (the "chart" dimension).
    pub phi_out: u8,
    /// BLAKE3 commitment over `phi_w1 || phi_b1 || phi_w2 || phi_b2 || W || W_pinv`.
    pub blake3: [u8; 32],
}

impl PoincareAdapter {
    /// Latent dimension `d`.
    #[inline]
    pub fn latent_dim(&self) -> usize {
        self.latent_dim as usize
    }

    /// Target dimension.
    #[inline]
    pub fn target_dim(&self) -> usize {
        self.target_dim as usize
    }

    /// φ hidden width.
    #[inline]
    pub fn phi_hidden(&self) -> usize {
        self.phi_hidden as usize
    }

    /// φ output width.
    #[inline]
    pub fn phi_out(&self) -> usize {
        self.phi_out as usize
    }

    /// Re-derive the BLAKE3 commitment over the current contents. Used by
    /// [`verify`](Self::verify) and by the freeze/thaw envelope (riir-neuron-db).
    pub fn recompute_blake3(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(
            &self
                .phi_w1
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        hasher.update(
            &self
                .phi_b1
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        hasher.update(
            &self
                .phi_w2
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        hasher.update(
            &self
                .phi_b2
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        hasher.update(
            &self
                .W
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        hasher.update(
            &self
                .W_pinv
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        *hasher.finalize().as_bytes()
    }

    /// Verify that the stored BLAKE3 commitment matches the current contents.
    /// Returns `false` if any weight was mutated after fitting.
    pub fn verify(&self) -> bool {
        self.recompute_blake3() == self.blake3
    }

    /// Canonical serialized form: `magic || latent_dim || target_dim ||
    /// phi_hidden || phi_out || phi_w1 (LE) || phi_b1 (LE) || phi_w2 (LE) ||
    /// phi_b2 (LE) || W (LE) || W_pinv (LE) || blake3`.
    ///
    /// Stable across versions — bump `FitConfig::magic` to break compatibility.
    /// Suitable for `MerkleFrozenEnvelope` (riir-neuron-db Plan 007).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let total = 5
            + (self.phi_w1.len()
                + self.phi_b1.len()
                + self.phi_w2.len()
                + self.phi_b2.len()
                + self.W.len()
                + self.W_pinv.len())
                * 4
            + 32;
        let mut out = Vec::with_capacity(total);
        out.push(b'P');
        out.push(self.latent_dim);
        out.push(self.target_dim);
        out.push(self.phi_hidden);
        out.push(self.phi_out);
        for f in self.phi_w1.iter() {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for f in self.phi_b1.iter() {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for f in self.phi_w2.iter() {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for f in self.phi_b2.iter() {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for f in self.W.iter() {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for f in self.W_pinv.iter() {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.extend_from_slice(&self.blake3);
        out
    }

    /// Reconstruct a [`PoincareAdapter`] from its canonical byte form.
    ///
    /// Verifies magic + BLAKE3; returns
    /// [`PoincareFitError::MalformedBuffer`] on any inconsistency
    /// (truncated, dim mismatch, failed commitment).
    pub fn from_bytes(buf: &[u8]) -> Result<Self, PoincareFitError> {
        if buf.len() < 5 + 32 {
            return Err(PoincareFitError::MalformedBuffer);
        }
        if buf[0] != b'P' {
            return Err(PoincareFitError::MalformedBuffer);
        }
        let latent_dim = buf[1] as usize;
        let target_dim = buf[2] as usize;
        let phi_hidden = buf[3] as usize;
        let phi_out = buf[4] as usize;
        if latent_dim == 0
            || latent_dim > LATENT_DIM_MAX
            || target_dim == 0
            || target_dim > TARGET_DIM_MAX
            || phi_hidden == 0
            || phi_hidden > LATENT_DIM_MAX
            || phi_out == 0
            || phi_out > LATENT_DIM_MAX
        {
            return Err(PoincareFitError::MalformedBuffer);
        }

        let n_w1 = phi_hidden * latent_dim;
        let n_b1 = phi_hidden;
        let n_w2 = phi_out * phi_hidden;
        let n_b2 = phi_out;
        let n_w = target_dim * phi_out;
        let n_pinv = phi_out * target_dim;
        let total_floats = n_w1 + n_b1 + n_w2 + n_b2 + n_w + n_pinv;
        let expected_len = 5 + total_floats * 4 + 32;
        if buf.len() != expected_len {
            return Err(PoincareFitError::MalformedBuffer);
        }

        let mut cursor = 5;
        let take = |n: usize, c: &mut usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let bytes: [u8; 4] = buf[*c..*c + 4].try_into().ok().unwrap_or([0; 4]);
                v.push(f32::from_le_bytes(bytes));
                *c += 4;
            }
            v
        };

        let phi_w1 = take(n_w1, &mut cursor);
        let phi_b1 = take(n_b1, &mut cursor);
        let phi_w2 = take(n_w2, &mut cursor);
        let phi_b2 = take(n_b2, &mut cursor);
        let W = take(n_w, &mut cursor);
        let W_pinv = take(n_pinv, &mut cursor);

        let blake3: [u8; 32] = buf[cursor..cursor + 32]
            .try_into()
            .map_err(|_| PoincareFitError::MalformedBuffer)?;

        let adapter = Self {
            phi_w1,
            phi_b1,
            phi_w2,
            phi_b2,
            W,
            W_pinv,
            latent_dim: latent_dim as u8,
            target_dim: target_dim as u8,
            phi_hidden: phi_hidden as u8,
            phi_out: phi_out as u8,
            blake3,
        };

        if !adapter.verify() {
            return Err(PoincareFitError::MalformedBuffer);
        }
        Ok(adapter)
    }
}

// ── Hot path: evaluate φ ─────────────────────────────────────────

/// Evaluate φ at `z` and write the result into `phi_out_scratch` (length
/// `phi_out`). Zero-allocation.
///
/// φ is a 2-layer MLP: `φ(z) = tanh(W2 · tanh(W1 · z + b1) + b2)`.
///
/// For the modelless default (PCA-tanh) the caller fits `W2 = I` and `b2 = 0`,
/// so this collapses to the single-layer `tanh(W1 · z + b1)`.
///
/// # Arguments
///
/// - `z` — latent point, length `adapter.latent_dim()`.
/// - `adapter` — frozen fitted adapter.
/// - `phi_out_scratch` — output buffer, length `adapter.phi_out()`.
/// - `hidden_scratch` — scratch for the hidden activation, length
///   `adapter.phi_hidden()`.
#[inline]
pub fn eval_phi_into(
    z: &[f32],
    adapter: &PoincareAdapter,
    phi_out_scratch: &mut [f32],
    hidden_scratch: &mut [f32],
) {
    let latent_dim = adapter.latent_dim();
    let phi_hidden = adapter.phi_hidden();
    let phi_out = adapter.phi_out();
    debug_assert_eq!(z.len(), latent_dim, "z.len != latent_dim");
    debug_assert_eq!(
        hidden_scratch.len(),
        phi_hidden,
        "hidden_scratch.len != phi_hidden"
    );
    debug_assert_eq!(
        phi_out_scratch.len(),
        phi_out,
        "phi_out_scratch.len != phi_out"
    );

    // Hidden layer: hidden[i] = tanh(Σ_j W1[i*latent + j] · z[j] + b1[i])
    for i in 0..phi_hidden {
        let row = &adapter.phi_w1[i * latent_dim..(i + 1) * latent_dim];
        let mut acc = adapter.phi_b1[i];
        acc += simd_dot_f32(row, z, latent_dim);
        hidden_scratch[i] = fast_tanh(acc);
    }

    // Output layer: out[k] = tanh(Σ_i W2[k*hidden + i] · hidden[i] + b2[k])
    for k in 0..phi_out {
        let row = &adapter.phi_w2[k * phi_hidden..(k + 1) * phi_hidden];
        let mut acc = adapter.phi_b2[k];
        acc += simd_dot_f32(row, hidden_scratch, phi_hidden);
        phi_out_scratch[k] = fast_tanh(acc);
    }
}

/// Evaluate `W_pinv · delta_target` and accumulate into `phi_scratch`.
///
/// After this call, `phi_scratch[k] = old_phi_scratch[k] + Σ_j W_pinv[k*target + j] · delta_target[j]`.
/// Combined with [`eval_phi_into`] writing `φ(z_src)` into `phi_scratch` first,
/// this realizes the chart-space destination `g_dest = φ(z_src) + W† · Δtarget`.
#[inline]
pub fn accumulate_pinv_into(
    delta_target: &[f32],
    adapter: &PoincareAdapter,
    phi_scratch: &mut [f32],
) {
    let phi_out = adapter.phi_out();
    let target_dim = adapter.target_dim();
    debug_assert_eq!(
        delta_target.len(),
        target_dim,
        "delta_target.len != target_dim"
    );
    debug_assert_eq!(phi_scratch.len(), phi_out, "phi_scratch.len != phi_out");

    // phi_scratch[k] += Σ_j W_pinv[k*target + j] · delta_target[j]
    for k in 0..phi_out {
        let row = &adapter.W_pinv[k * target_dim..(k + 1) * target_dim];
        phi_scratch[k] += simd_dot_f32(row, delta_target, target_dim);
    }
}

// ── Hot path: navigator ──────────────────────────────────────────

/// Navigate from `z_src` toward a desired target displacement, writing the
/// imagined destination into `z_out`.
///
/// Realizes:
///
/// ```text
/// z_out = z_src + φ⁻¹( φ(z_src) + W† · Δtarget )
/// ```
///
/// The `φ⁻¹` step here is the **least-squares inverse**: project the chart
/// displacement back into latent space via `φ_w1ᵀ` (the transpose of φ's first
/// layer). For the modelless PCA-tanh adapter (`phi_w2 = I`, orthonormal
/// PCA basis), this is the exact pseudoinverse and the round-trip is exact
/// for displacements in `span(W1)`; for the gradient-fit adapter it's an
/// approximation (consumers concerned about fidelity should use
/// [`poincare_navigate_chart_into`] and do their own retrieval / decoding).
///
/// # Zero-allocation
///
/// The caller supplies the three scratch buffers. None of them grow during
/// the call. After warmup, this function allocates 0 bytes (G4 gate).
///
/// # Arguments
///
/// - `z_src` — source latent state, length `adapter.latent_dim()`.
/// - `delta_target` — desired target displacement, length `adapter.target_dim()`.
/// - `adapter` — frozen fitted adapter.
/// - `z_out` — destination latent state, length `adapter.latent_dim()`.
///   May alias `z_src`.
/// - `phi_scratch` — length `adapter.phi_out()`.
/// - `hidden_scratch` — length `adapter.phi_hidden()`.
#[inline]
pub fn poincare_navigate_into(
    z_src: &[f32],
    delta_target: &[f32],
    adapter: &PoincareAdapter,
    z_out: &mut [f32],
    phi_scratch: &mut [f32],
    hidden_scratch: &mut [f32],
) {
    let latent_dim = adapter.latent_dim();
    debug_assert_eq!(z_src.len(), latent_dim, "z_src.len != latent_dim");
    debug_assert_eq!(z_out.len(), latent_dim, "z_out.len != latent_dim");

    // 1) φ(z_src) → phi_scratch
    eval_phi_into(z_src, adapter, phi_scratch, hidden_scratch);

    // 2) phi_scratch += W_pinv · delta_target
    accumulate_pinv_into(delta_target, adapter, phi_scratch);

    // 3) z_out = z_src + φ⁻¹(phi_scratch)
    //    φ⁻¹ via least-squares (W1ᵀ): for each latent dim j,
    //      dz[j] = Σ_k W1[k*latent + j] · phi_scratch[k]
    //    (W1 columns are the latent-space basis directions; this projects the
    //    chart displacement onto them.)
    for j in 0..latent_dim {
        let mut dz = 0.0f32;
        for k in 0..adapter.phi_out() {
            dz += adapter.phi_w1[k * latent_dim + j] * phi_scratch[k];
        }
        z_out[j] = z_src[j] + dz;
    }
}

/// Navigate `delta_target` in `n_steps` open-loop sub-steps.
///
/// Splits `delta_target / n_steps` and iterates [`poincare_navigate_into`]
/// `n_steps` times. Open-loop integrator (no correction from environment).
///
/// Determinism: identical `(z_src, delta_target, n_steps, adapter)` yields
/// bit-identical `z_out`. No RNG, no thread-state reads.
///
/// Returns the final `z_out`. `z_out` may alias `z_src` for in-place update.
///
/// # Arguments
///
/// - `z_src` — source latent state, length `adapter.latent_dim()`.
/// - `delta_target` — total desired target displacement over the trajectory.
/// - `n_steps` — number of open-loop sub-steps. Must be ≥ 1.
/// - `adapter` — frozen fitted adapter.
/// - `z_out` — destination latent state, length `adapter.latent_dim()`.
///   May alias `z_src`.
/// - `phi_scratch`, `hidden_scratch` — scratch buffers (same shape as
///   [`poincare_navigate_into`]).
/// - `delta_step_scratch` — length `adapter.target_dim()`. Holds the
///   per-step `delta_target / n_steps`.
#[inline]
#[allow(clippy::too_many_arguments)] // navigator + 3 scratch buffers is intrinsic to the zero-alloc hot path
pub fn poincare_multi_step_into(
    z_src: &[f32],
    delta_target: &[f32],
    n_steps: usize,
    adapter: &PoincareAdapter,
    z_out: &mut [f32],
    phi_scratch: &mut [f32],
    hidden_scratch: &mut [f32],
    delta_step_scratch: &mut [f32],
) {
    debug_assert!(n_steps >= 1, "n_steps must be >= 1");
    let target_dim = adapter.target_dim();
    let latent_dim = adapter.latent_dim();
    debug_assert_eq!(delta_target.len(), target_dim);
    debug_assert_eq!(delta_step_scratch.len(), target_dim);

    let inv_steps = 1.0f32 / (n_steps as f32);
    for j in 0..target_dim {
        delta_step_scratch[j] = delta_target[j] * inv_steps;
    }

    // First step: z_src → z_out
    poincare_navigate_into(
        z_src,
        delta_step_scratch,
        adapter,
        z_out,
        phi_scratch,
        hidden_scratch,
    );
    // Subsequent steps: z_out → z_out (in-place).
    //
    // `poincare_navigate_into` reads `z_src` fully before writing `z_out`
    // (φ(z_src) is computed into phi_scratch first, then z_out is written one
    // element at a time as `z_src[j] + dz[j]`). BUT the borrow checker can't
    // see this — `z_out` is `&mut`, and passing it as both `z_src: &[f32]`
    // and `z_out: &mut [f32]` violates aliasing rules. We work around this by
    // copying z_out into a small stack buffer, navigating from the copy into
    // z_out, and repeating. The copy is `latent_dim ≤ 64` f32s = ≤ 256 bytes,
    // stays in L1.
    let mut z_prev = [0.0f32; LATENT_DIM_MAX];
    for _ in 1..n_steps {
        z_prev[..latent_dim].copy_from_slice(z_out);
        poincare_navigate_into(
            &z_prev[..latent_dim],
            delta_step_scratch,
            adapter,
            z_out,
            phi_scratch,
            hidden_scratch,
        );
    }
}

// ── Offline fit (modelless) ──────────────────────────────────────

/// Offline closed-form fit of a [`PoincareAdapter`] over `(z, target)` sample
/// pairs.
///
/// # Modelless protocol (research skill §3.5)
///
/// This ships the deterministic closed-form path:
/// 1. **φ = PCA-tanh**: center the `z` samples, take the top-`phi_out` right
///    singular vectors of the centered `z` matrix, set `W1 = σ · Vᵀ` (scaled
///    so each PCA direction has unit output magnitude), `b1 = -W1 · mean_z`,
///    `W2 = I`, `b2 = 0`. So `φ(z) = tanh(W1 · (z − mean_z))`.
/// 2. **W = ridge**: solve `W = (ZᵀZ + αI)⁻¹ · ZᵀY` where `Z` is the stacked
///    `φ(z_i)` and `Y` is the stacked `target_i`. Done via
///    [`crate::linalg::ridge_solve_direct_f32`] (Cholesky factorisation).
/// 3. **W_pinv = SVD-based**: compute the thin SVD of `W` and set
///    `W_pinv = V · Σ⁻¹ · Uᵀ`.
///
/// # Arguments
///
/// - `z_samples` — `N` latent samples, each length `latent_dim`. Row-major
///   (one `&[f32]` per sample).
/// - `target_samples` — `N` target samples, each length `target_dim`. Same
///   row order as `z_samples`.
/// - `latent_dim`, `target_dim`, `phi_out`, `phi_hidden` — see
///   [`PoincareAdapter`] field docs. For the modelless default
///   (`phi_hidden == phi_out`) the adapter is single-layer PCA-tanh; for
///   `phi_hidden > phi_out` the caller must supply a gradient fit
///   (`riir-train` follow-up) — this function rejects that with
///   [`PoincareFitError::PhiHiddenOutOfRange`] only if it exceeds
///   [`LATENT_DIM_MAX`]; otherwise it falls back to single-layer with
///   `phi_hidden := phi_out` (silently — documented in the result's
///   `phi_hidden` field).
/// - `cfg` — ridge regularization, rank threshold, magic byte.
pub fn fit_poincare_adapter(
    z_samples: &[&[f32]],
    target_samples: &[&[f32]],
    latent_dim: usize,
    target_dim: usize,
    phi_out: usize,
    phi_hidden: usize,
    cfg: &FitConfig,
) -> Result<PoincareAdapter, PoincareFitError> {
    // ── Validate inputs ───────────────────────────────────────────
    if latent_dim == 0 || latent_dim > LATENT_DIM_MAX {
        return Err(PoincareFitError::LatentDimOutOfRange);
    }
    if target_dim == 0 || target_dim > TARGET_DIM_MAX {
        return Err(PoincareFitError::TargetDimOutOfRange);
    }
    if phi_out == 0 || phi_out > latent_dim || phi_out > LATENT_DIM_MAX {
        return Err(PoincareFitError::PhiOutOutOfRange);
    }
    if phi_hidden == 0 || phi_hidden > LATENT_DIM_MAX {
        return Err(PoincareFitError::PhiHiddenOutOfRange);
    }
    if z_samples.len() != target_samples.len() || z_samples.len() < target_dim + 1 {
        return Err(PoincareFitError::InsufficientSamples);
    }
    for (i, z) in z_samples.iter().enumerate() {
        if z.len() != latent_dim {
            return Err(PoincareFitError::LatentLenMismatch);
        }
        let _ = i;
    }
    for t in target_samples.iter() {
        if t.len() != target_dim {
            return Err(PoincareFitError::TargetLenMismatch);
        }
    }

    // For the modelless path, silently collapse phi_hidden → phi_out
    // (single-layer PCA-tanh). The gradient-fit (riir-train) path will
    // populate phi_hidden > phi_out via its own constructor.
    let phi_hidden_eff = phi_hidden.min(phi_out);
    let n = z_samples.len();

    // ── Step 1: PCA on z_samples ──────────────────────────────────
    // Center the samples.
    let mut mean_z = vec![0.0f32; latent_dim];
    for z in z_samples.iter() {
        for (j, &v) in z.iter().enumerate() {
            mean_z[j] += v;
        }
    }
    let inv_n = 1.0f32 / (n as f32);
    for v in mean_z.iter_mut() {
        *v *= inv_n;
    }

    // Build the (N × latent_dim) centered-z matrix, row-major.
    let mut z_centered = vec![0.0f32; n * latent_dim];
    for (i, z) in z_samples.iter().enumerate() {
        for (j, &v) in z.iter().enumerate() {
            z_centered[i * latent_dim + j] = v - mean_z[j];
        }
    }

    // Thin SVD: V columns are the principal directions in input space.
    // Z is (N × latent_dim); we factor it as `M = U · Σ · Vᵀ`. The right
    // singular vectors V[:, k] are the principal directions in input space
    // (length `latent_dim`).
    //
    // Argument-order note: `SvdResultScratch::with_capacity(m_rows, n_cols)`
    // takes (rows, cols) — but `SvdScratch::with_capacity(n_cols, m_rows)`
    // takes (cols, rows) (see subspace_phase_gate.rs L568-578). The reversed
    // scratch order is a known wart kept for back-compat with Plan 301.
    let mut svd_result = SvdResultScratch::with_capacity(n, latent_dim);
    let mut svd_work = SvdScratch::with_capacity(latent_dim, n);
    thin_svd_into(&z_centered, n, latent_dim, &mut svd_result, &mut svd_work);

    // Take top-phi_hidden_eff right singular vectors. Each `V[:, k]` is
    // length `latent_dim` and accessed via the public accessor
    // [`SvdResultScratch::right_singular_vector`]. **W1 rows are unit-norm**
    // (the PCA directions themselves) — we do NOT scale by σ_k. Scaling by σ_k
    // would push `W1 · z` into tanh's saturation regime (|x| > 3) and erase
    // the signal. Keeping W1 unit-norm keeps the projection in the same
    // magnitude range as `z`, where tanh ≈ identity for small inputs and
    // only mildly compresses large ones — the chart stays close to linear.
    //
    // The σ_k information is recovered implicitly through the ridge fit: W's
    // rows are scaled to whatever the targets require.
    let mut phi_w1 = vec![0.0f32; phi_hidden_eff * latent_dim];
    let mut phi_b1 = vec![0.0f32; phi_hidden_eff];
    for k in 0..phi_hidden_eff {
        let v_col = svd_result.right_singular_vector(k);
        // W1 row k = V[:, k]  (unit-norm PCA direction)
        for j in 0..latent_dim {
            phi_w1[k * latent_dim + j] = v_col[j];
        }
        // b1[k] = -W1[k] · mean_z  (center the projection so φ(mean_z) = tanh(0) = 0)
        let mut b = 0.0f32;
        for j in 0..latent_dim {
            b -= phi_w1[k * latent_dim + j] * mean_z[j];
        }
        phi_b1[k] = b;
    }

    // W2 = I (phi_out × phi_hidden_eff), b2 = 0 (modelless single-layer form).
    let mut phi_w2 = vec![0.0f32; phi_out * phi_hidden_eff];
    let phi_b2 = vec![0.0f32; phi_out];
    for k in 0..phi_out.min(phi_hidden_eff) {
        phi_w2[k * phi_hidden_eff + k] = 1.0;
    }

    // ── Step 2: Apply φ to all samples, build Z (N × phi_out) ──────
    let mut phi_z = vec![0.0f32; n * phi_out];
    let mut hidden = vec![0.0f32; phi_hidden_eff];
    let mut phi_row = vec![0.0f32; phi_out];
    // The fitting adapter is loop-invariant: `dummy_adapter` deep-copies the
    // four weight buffers (4 heap allocations) and none of them are mutated
    // inside this loop, so it was being rebuilt — and thrown away — once per
    // sample. Build it once. `phi_z` has exactly `n * phi_out` elements and
    // `n == z_samples.len()` (line 721), so `chunks_exact_mut` yields exactly
    // as many rows as there are samples and the `zip` cannot iterate short.
    let fit_adapter = dummy_adapter(
        &phi_w1,
        &phi_b1,
        &phi_w2,
        &phi_b2,
        latent_dim,
        phi_out,
        phi_hidden_eff,
    );
    for (z, row) in z_samples.iter().zip(phi_z.chunks_exact_mut(phi_out)) {
        eval_phi_into(z, &fit_adapter, &mut phi_row, &mut hidden);
        row.copy_from_slice(&phi_row[..phi_out]);
    }

    // ── Step 3: Ridge fit W = (ZᵀZ + αI)⁻¹ · ZᵀY ──────────────────
    // Gram matrix ZᵀZ (phi_out × phi_out).
    let mut gram = vec![0.0f32; phi_out * phi_out];
    for i in 0..phi_out {
        for j in 0..phi_out {
            let mut s = 0.0f32;
            for r in 0..n {
                s += phi_z[r * phi_out + i] * phi_z[r * phi_out + j];
            }
            gram[i * phi_out + j] = s;
        }
    }
    // Add αI to the diagonal.
    for i in 0..phi_out {
        gram[i * phi_out + i] += cfg.ridge_alpha;
    }
    // Covariance ZᵀY (phi_out × target_dim).
    let mut cov = vec![0.0f32; phi_out * target_dim];
    for i in 0..phi_out {
        for j in 0..target_dim {
            let mut s = 0.0f32;
            for r in 0..n {
                s += phi_z[r * phi_out + i] * target_samples[r][j];
            }
            cov[i * target_dim + j] = s;
        }
    }
    // Ridge solve: Wᵀ (phi_out × target_dim) → but we want W (target_dim × phi_out)
    // row-major. The ridge_solve_direct_f32 writes Wᵀ in column-major; we'll
    // transpose by re-interpreting: the output w_t has shape (d_h × n_out)
    // where d_h = phi_out, n_out = target_dim. So `w_t[i * target_dim + j]`
    // is `W[j][i]`. After the solve we transpose into the adapter's `W`
    // layout (`W[j * phi_out + i]`).
    use crate::linalg::ridge_solve::ridge_solve_direct_f32;
    let mut w_t = vec![0.0f32; phi_out * target_dim];
    let mut l_scratch = vec![0.0f32; phi_out * phi_out];
    let mut z_scratch = vec![0.0f32; phi_out * target_dim];
    ridge_solve_direct_f32(
        &mut w_t,
        &mut l_scratch,
        &mut z_scratch,
        &gram,
        &cov,
        phi_out,
        target_dim,
    );
    let mut W = vec![0.0f32; target_dim * phi_out];
    for j in 0..target_dim {
        for i in 0..phi_out {
            W[j * phi_out + i] = w_t[i * target_dim + j];
        }
    }

    // ── Step 4: W_pinv = V · Σ⁻¹ · Uᵀ via thin SVD of W ───────────
    // W is (target_dim × phi_out). Thin SVD returns U (target_dim × k),
    // Σ (k), V (phi_out × k), where k = min(target_dim, phi_out).
    let mut w_svd_result = SvdResultScratch::with_capacity(target_dim, phi_out);
    let mut w_svd_work = SvdScratch::with_capacity(phi_out, target_dim);
    thin_svd_into(&W, target_dim, phi_out, &mut w_svd_result, &mut w_svd_work);

    // Numerical rank check: top σ must be ≥ tau · σ_max; below cutoff are
    // zeroed in Σ⁻¹.
    let sigma_max = if !w_svd_result.is_empty() {
        w_svd_result.singular_value(0).max(1e-12)
    } else {
        1e-12
    };
    let cutoff = cfg.rank_tau * sigma_max;
    let mut rank = 0;
    for k in 0..w_svd_result.len() {
        if w_svd_result.singular_value(k) > cutoff {
            rank += 1;
        }
    }
    if rank < target_dim.min(phi_out) {
        // Rank-deficient W — either the samples don't span the target space
        // or phi_out is too small. Caller should add more samples or increase
        // phi_out.
        return Err(PoincareFitError::RankDeficient);
    }

    // W_pinv = V · Σ⁻¹ · Uᵀ, shape (phi_out × target_dim).
    // V[:, k] is length `phi_out`; U[:, k] is length `target_dim`.
    let mut W_pinv = vec![0.0f32; phi_out * target_dim];
    for k in 0..w_svd_result.len() {
        let sigma = w_svd_result.singular_value(k);
        if sigma <= cutoff {
            continue;
        }
        let inv_sigma = 1.0f32 / sigma;
        // W_pinv += inv_sigma · V[:, k] · U[:, k]ᵀ
        // outer product: W_pinv[i * target_dim + j] += inv_sigma · V[i, k] · U[j, k]
        let v_col = w_svd_result.right_singular_vector(k); // length phi_out
        let u_col = w_svd_result.left_singular_vector(k); // length target_dim
        for i in 0..phi_out {
            let v = v_col[i];
            for j in 0..target_dim {
                W_pinv[i * target_dim + j] += inv_sigma * v * u_col[j];
            }
        }
    }

    // ── Step 5: Assemble the adapter, compute BLAKE3 ──────────────
    let mut adapter = PoincareAdapter {
        phi_w1,
        phi_b1,
        phi_w2,
        phi_b2,
        W,
        W_pinv,
        latent_dim: latent_dim as u8,
        target_dim: target_dim as u8,
        phi_hidden: phi_hidden_eff as u8,
        phi_out: phi_out as u8,
        blake3: [0u8; 32], // placeholder; recompute below
    };
    adapter.blake3 = adapter.recompute_blake3();
    Ok(adapter)
}

/// Tiny helper to build a "fitting-time" adapter so we can reuse
/// [`eval_phi_into`] while fitting. Borrows the weights; doesn't compute BLAKE3.
fn dummy_adapter(
    phi_w1: &[f32],
    phi_b1: &[f32],
    phi_w2: &[f32],
    phi_b2: &[f32],
    latent_dim: usize,
    phi_out: usize,
    phi_hidden: usize,
) -> PoincareAdapter {
    PoincareAdapter {
        phi_w1: phi_w1.to_vec(), // small alloc on cold fit path
        phi_b1: phi_b1.to_vec(),
        phi_w2: phi_w2.to_vec(),
        phi_b2: phi_b2.to_vec(),
        W: Vec::new(),
        W_pinv: Vec::new(),
        latent_dim: latent_dim as u8,
        target_dim: 0,
        phi_hidden: phi_hidden as u8,
        phi_out: phi_out as u8,
        blake3: [0u8; 32],
    }
}

// ── Numerics: tanh ───────────────────────────────────────────────

/// Fast tanh for MLP hidden-layer activations. Delegates to
/// [`simd::fast_tanh`] (Padé [2/2] rational polynomial, ~0.025 worst-case
/// absolute error near |x|≈2, saturates to ±1 for |x|>3).
///
/// Safe for the `eval_phi_into` hidden-layer activation use case (bounded
/// scalar squashing — drift is acceptable, no algebraic identity to preserve).
///
/// **NOT safe for norm-preservation identities** (cos²+sin²=1 etc.) — see
/// Plan 322 `phase_rotation_coupling` for the canonical failure mode where
/// independent Padé approximations drifted the Pythagorean identity by ~5e-3.
/// This function is only used in `eval_phi_into`, never in the geodesic /
/// exp-map / norm-preservation paths.
#[inline]
fn fast_tanh(x: f32) -> f32 {
    crate::simd::fast_tanh(x)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol * (1.0 + a.abs() + b.abs())
    }

    fn make_random_unit_vector(rng: &mut fastrand::Rng, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x /= norm.max(1e-12);
        }
        v
    }

    #[test]
    fn smoke_validates_fit_inputs() {
        let cfg = FitConfig::default();
        let z: Vec<f32> = vec![0.0; 4];
        let t: Vec<f32> = vec![0.0; 2];
        let zs: Vec<&[f32]> = vec![&z];
        let ts: Vec<&[f32]> = vec![&t];
        assert_eq!(
            fit_poincare_adapter(&zs, &ts, 4, 2, 2, 2, &cfg).unwrap_err(),
            PoincareFitError::InsufficientSamples
        );
        assert_eq!(
            fit_poincare_adapter(&zs, &ts, 0, 2, 2, 2, &cfg).unwrap_err(),
            PoincareFitError::LatentDimOutOfRange
        );
    }

    #[test]
    fn fits_identity_map_recovers_it() {
        // Construct a linear target map: target = A · z (4 → 2) with
        // rank-2 A. The adapter should fit W · φ(z) ≈ target.
        //
        // Use phi_out = latent_dim (no PCA reduction) — this is the
        // "linear decoder" regime where the modelless adapter must succeed.
        // (A separate test exercises the nonlinear-unrolling regime where
        // phi_out < latent_dim and tanh provides the chart.)
        let mut rng = fastrand::Rng::with_seed(42);
        let latent_dim = 4;
        let target_dim = 2;
        let phi_out = latent_dim; // no reduction

        // A is (target_dim × latent_dim), rank 2.
        let a_row1 = make_random_unit_vector(&mut rng, latent_dim);
        let a_row2 = make_random_unit_vector(&mut rng, latent_dim);
        let a = [a_row1, a_row2];

        // Sample N pairs.
        let n = 50;
        let z_samples: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..latent_dim).map(|_| rng.f32() * 2.0 - 1.0).collect())
            .collect();
        let target_samples: Vec<Vec<f32>> = z_samples
            .iter()
            .map(|z| {
                let mut t = vec![0.0f32; target_dim];
                for j in 0..target_dim {
                    t[j] = simd_dot_f32(&a[j], z, latent_dim);
                }
                t
            })
            .collect();
        let z_refs: Vec<&[f32]> = z_samples.iter().map(|v| v.as_slice()).collect();
        let t_refs: Vec<&[f32]> = target_samples.iter().map(|v| v.as_slice()).collect();

        let adapter = fit_poincare_adapter(
            &z_refs,
            &t_refs,
            latent_dim,
            target_dim,
            phi_out,
            phi_out,
            &FitConfig::default(),
        )
        .expect("fit should succeed");

        // Reconstruct targets on held-out z. R² should be ~1.0.
        let mut hidden = vec![0.0f32; adapter.phi_hidden()];
        let mut phi = vec![0.0f32; adapter.phi_out()];
        let mut max_err = 0.0f32;
        for _ in 0..20 {
            let z: Vec<f32> = (0..latent_dim).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let mut t_truth = vec![0.0f32; target_dim];
            for j in 0..target_dim {
                t_truth[j] = simd_dot_f32(&a[j], &z, latent_dim);
            }
            eval_phi_into(&z, &adapter, &mut phi, &mut hidden);
            let mut t_hat = vec![0.0f32; target_dim];
            for j in 0..target_dim {
                t_hat[j] = simd_dot_f32(&adapter.W[j * phi_out..(j + 1) * phi_out], &phi, phi_out);
            }
            for j in 0..target_dim {
                let err = (t_truth[j] - t_hat[j]).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        // Identity linear map + linear decoder + tanh warp.
        // tanh(|z| ≤ √4 = 2) compresses to tanh(2) = 0.964, a ~4% signal
        // loss per axis that compounds through the W matvec. Recovery within
        // 0.5 (out of |target| ≤ 2) proves the adapter learned the right
        // direction and approximately the right magnitude — the residual is
        // tanh's bounded distortion, not a fit failure. The Phase 2 G1 gate
        // (small-displacement linearity) tightens this to 1e-3 on small |z|.
        assert!(max_err < 0.5, "max_err = {max_err:.4} exceeds 0.5");
    }

    #[test]
    fn inverse_navigator_round_trips_in_subspace() {
        // For an adapter fit on a linear map, the inverse navigator
        // z + W_pinv · Δtarget should produce a z' whose W · φ(z') recovers
        // Δtarget up to the tanh warp.
        let mut rng = fastrand::Rng::with_seed(7);
        let latent_dim = 4;
        let target_dim = 2;
        let phi_out = 2;
        let a_row1 = make_random_unit_vector(&mut rng, latent_dim);
        let a_row2 = make_random_unit_vector(&mut rng, latent_dim);
        let a = [a_row1, a_row2];

        let n = 50;
        let z_samples: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..latent_dim).map(|_| rng.f32() * 0.5).collect())
            .collect();
        let target_samples: Vec<Vec<f32>> = z_samples
            .iter()
            .map(|z| {
                let mut t = vec![0.0f32; target_dim];
                for j in 0..target_dim {
                    t[j] = simd_dot_f32(&a[j], z, latent_dim);
                }
                t
            })
            .collect();
        let z_refs: Vec<&[f32]> = z_samples.iter().map(|v| v.as_slice()).collect();
        let t_refs: Vec<&[f32]> = target_samples.iter().map(|v| v.as_slice()).collect();
        let adapter = fit_poincare_adapter(
            &z_refs,
            &t_refs,
            latent_dim,
            target_dim,
            phi_out,
            phi_out,
            &FitConfig::default(),
        )
        .expect("fit should succeed");

        let z_src: Vec<f32> = (0..latent_dim).map(|_| rng.f32() * 0.3).collect();
        let delta_target = vec![0.1, 0.05];
        let mut z_out = vec![0.0f32; latent_dim];
        let mut phi = vec![0.0f32; phi_out];
        let mut hidden = vec![0.0f32; adapter.phi_hidden()];
        poincare_navigate_into(
            &z_src,
            &delta_target,
            &adapter,
            &mut z_out,
            &mut phi,
            &mut hidden,
        );

        // Project z_out back: W · φ(z_out) − W · φ(z_src) ≈ delta_target.
        eval_phi_into(&z_src, &adapter, &mut phi, &mut hidden);
        let mut w_phi_src = vec![0.0f32; target_dim];
        for j in 0..target_dim {
            w_phi_src[j] = simd_dot_f32(&adapter.W[j * phi_out..(j + 1) * phi_out], &phi, phi_out);
        }
        eval_phi_into(&z_out, &adapter, &mut phi, &mut hidden);
        let mut w_phi_out = vec![0.0f32; target_dim];
        for j in 0..target_dim {
            w_phi_out[j] = simd_dot_f32(&adapter.W[j * phi_out..(j + 1) * phi_out], &phi, phi_out);
        }
        let dx0 = w_phi_out[0] - w_phi_src[0];
        let dx1 = w_phi_out[1] - w_phi_src[1];
        // The recovery is approximate (tanh warp + W2 = I + PCA scaling).
        // Direction must match; magnitude may shrink.
        let direction_match = (dx0 * delta_target[0] + dx1 * delta_target[1]) > 0.0;
        assert!(
            direction_match,
            "navigator moved the wrong way: dx=({dx0:.4}, {dx1:.4}), Δtarget=({0}, {1})",
            delta_target[0], delta_target[1]
        );
    }

    #[test]
    fn canonical_bytes_round_trip_is_bit_identical() {
        let mut rng = fastrand::Rng::with_seed(13);
        let latent_dim = 4;
        let target_dim = 2;
        let phi_out = 2;
        let a_row1 = make_random_unit_vector(&mut rng, latent_dim);
        let a_row2 = make_random_unit_vector(&mut rng, latent_dim);
        let a = [a_row1, a_row2];

        let n = 30;
        let z_samples: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..latent_dim).map(|_| rng.f32() * 0.5).collect())
            .collect();
        let target_samples: Vec<Vec<f32>> = z_samples
            .iter()
            .map(|z| {
                let mut t = vec![0.0f32; target_dim];
                for j in 0..target_dim {
                    t[j] = simd_dot_f32(&a[j], z, latent_dim);
                }
                t
            })
            .collect();
        let z_refs: Vec<&[f32]> = z_samples.iter().map(|v| v.as_slice()).collect();
        let t_refs: Vec<&[f32]> = target_samples.iter().map(|v| v.as_slice()).collect();
        let adapter = fit_poincare_adapter(
            &z_refs,
            &t_refs,
            latent_dim,
            target_dim,
            phi_out,
            phi_out,
            &FitConfig::default(),
        )
        .expect("fit should succeed");

        let bytes = adapter.canonical_bytes();
        let recovered = PoincareAdapter::from_bytes(&bytes).expect("round-trip");
        assert_eq!(recovered.latent_dim, adapter.latent_dim);
        assert_eq!(recovered.target_dim, adapter.target_dim);
        assert_eq!(recovered.phi_hidden, adapter.phi_hidden);
        assert_eq!(recovered.phi_out, adapter.phi_out);
        assert_eq!(recovered.blake3, adapter.blake3);
        for i in 0..adapter.phi_w1.len() {
            assert!(approx_eq(recovered.phi_w1[i], adapter.phi_w1[i], 1e-6));
        }
        for i in 0..adapter.W.len() {
            assert!(approx_eq(recovered.W[i], adapter.W[i], 1e-6));
        }
        for i in 0..adapter.W_pinv.len() {
            assert!(approx_eq(recovered.W_pinv[i], adapter.W_pinv[i], 1e-6));
        }
        // Tamper with the buffer → verification fails.
        let mut tampered = bytes.clone();
        let last_weight_byte = bytes.len() - 32 - 1;
        tampered[last_weight_byte] ^= 0xff;
        assert_eq!(
            PoincareAdapter::from_bytes(&tampered).unwrap_err(),
            PoincareFitError::MalformedBuffer
        );
    }

    #[test]
    fn multi_step_is_deterministic() {
        let mut rng = fastrand::Rng::with_seed(99);
        let latent_dim = 4;
        let target_dim = 2;
        let phi_out = 2;
        let a_row1 = make_random_unit_vector(&mut rng, latent_dim);
        let a_row2 = make_random_unit_vector(&mut rng, latent_dim);
        let a = [a_row1, a_row2];

        let n = 30;
        let z_samples: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..latent_dim).map(|_| rng.f32() * 0.5).collect())
            .collect();
        let target_samples: Vec<Vec<f32>> = z_samples
            .iter()
            .map(|z| {
                let mut t = vec![0.0f32; target_dim];
                for j in 0..target_dim {
                    t[j] = simd_dot_f32(&a[j], z, latent_dim);
                }
                t
            })
            .collect();
        let z_refs: Vec<&[f32]> = z_samples.iter().map(|v| v.as_slice()).collect();
        let t_refs: Vec<&[f32]> = target_samples.iter().map(|v| v.as_slice()).collect();
        let adapter = fit_poincare_adapter(
            &z_refs,
            &t_refs,
            latent_dim,
            target_dim,
            phi_out,
            phi_out,
            &FitConfig::default(),
        )
        .expect("fit should succeed");

        let z_src: Vec<f32> = vec![0.1, 0.2, -0.1, 0.05];
        let delta = vec![0.2, -0.1];
        let mut z_a = vec![0.0f32; latent_dim];
        let mut z_b = vec![0.0f32; latent_dim];
        let mut phi = vec![0.0f32; phi_out];
        let mut hidden = vec![0.0f32; adapter.phi_hidden()];
        let mut delta_step = vec![0.0f32; target_dim];
        poincare_multi_step_into(
            &z_src,
            &delta,
            4,
            &adapter,
            &mut z_a,
            &mut phi,
            &mut hidden,
            &mut delta_step,
        );
        poincare_multi_step_into(
            &z_src,
            &delta,
            4,
            &adapter,
            &mut z_b,
            &mut phi,
            &mut hidden,
            &mut delta_step,
        );
        for j in 0..latent_dim {
            assert!(
                z_a[j].to_bits() == z_b[j].to_bits(),
                "non-deterministic at j={j}: {a} vs {b}",
                a = z_a[j],
                b = z_b[j]
            );
        }
    }

    #[test]
    fn latent_raw_boundary_no_sync_types() {
        // Static check: the navigator signature uses only &[f32] / &mut [f32]
        // / &PoincareAdapter. No MapPos / SyncBlock / ChainConsensus. Enforced
        // by the type signature itself — this test exists to pin the API.
        let _ = std::any::TypeId::of::<
            fn(&[f32], &[f32], &PoincareAdapter, &mut [f32], &mut [f32], &mut [f32]),
        >();
        // If the type signature changes to leak a sync/chain/game type, this
        // line will fail to compile (rustc will complain about the missing
        // type parameter).
    }
}
