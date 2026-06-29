//! SSD Block Decomposition — dual-mode chunked semiseparable state space computation.
//!
//! Based on "Transformers are SSMs" (arXiv:2405.21060, Section 6): the block
//! decomposition splits the quadratic SSD computation into four steps:
//!
//! 1. **Diagonal blocks** — intra-chunk quadratic attention via `segsum` + matmul
//! 2. **Right factors** — input → state contribution per chunk via `cumprodsum_scalar`
//! 3. **Center factors** — inter-chunk state propagation via `cumprodsum_scalar`
//! 4. **Left factors** — state → output contribution per chunk
//!
//! For T tokens, state_dim N, head_dim P, block_len Q:
//! - If T ≤ Q: pure quadratic (single chunk, steps 2–4 carry no inter-chunk state)
//! - If T > Q: O(T·Q·P) for diagonal + O(T·N·P) for off-diagonal, vs O(T²·P) naive
//!
//! All temporaries live in [`SsdScratch`]; the forward pass allocates nothing
//! beyond `segsum`'s internal small buffer (Q elements).
//!
//! Reference: Research 230 — Semiseparable State Space Duality, Plan 263.

#![allow(clippy::needless_range_loop)]

use crate::cumprodsum::{cumprodsum_scalar, segsum};

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the SSD block decomposition.
///
/// * `block_len` — chunk size Q (64 for CPU/SIMD, 128 for GPU tensor cores)
/// * `state_dim` — state dimension N (number of key/value features per position)
/// * `head_dim` — head dimension P (output feature size per position)
#[derive(Clone, Copy, Debug)]
pub struct SsdBlockConfig {
    pub block_len: usize,
    pub state_dim: usize,
    pub head_dim: usize,
}

impl SsdBlockConfig {
    /// Default CPU configuration: block_len=64, state_dim=16, head_dim=64.
    pub const fn default_cpu() -> Self {
        Self {
            block_len: 64,
            state_dim: 16,
            head_dim: 64,
        }
    }
}

impl Default for SsdBlockConfig {
    fn default() -> Self {
        Self::default_cpu()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Adaptive chunk-size routing
// ────────────────────────────────────────────────────────────────────────────

/// Adaptive block length based on sequence length.
///
/// * T ≤ 255: full quadratic (single chunk, no inter-chunk overhead)
/// * 256 ≤ T ≤ 2047: block_len=64 (CPU/SIMD sweet spot)
/// * T ≥ 2048: block_len=128 (GPU tensor core sweet spot)
#[inline]
pub fn auto_block_len(seq_len: usize) -> usize {
    match seq_len {
        0..=255 => seq_len.max(1),
        256..=2047 => 64,
        _ => 128,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Scratch buffers (zero-allocation hot path)
// ────────────────────────────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for [`ssd_block_forward`].
///
/// All temporaries are stored here so the forward pass allocates nothing
/// (beyond `segsum`'s internal Q-element buffer). Reuse across calls with
/// different `seq_len` via [`SsdScratch::for_seq_len`].
pub struct SsdScratch {
    /// Diagonal-block (intra-chunk) contribution: `[T * head_dim]`
    chunk_out: Vec<f32>,
    /// Per-chunk terminal states (right factors): `[state_dim * head_dim * n_chunks]`
    /// Layout: `[n][p][chunk]` — chunk-contiguous for `cumprodsum_scalar`.
    chunk_states: Vec<f32>,
    /// Propagated boundary states (center factors): `[state_dim * head_dim * n_chunks]`
    /// Layout: `[n][p][chunk]` — chunk-contiguous for `cumprodsum_scalar`.
    inter_state: Vec<f32>,
    /// Per-chunk decay product: `[n_chunks]`
    chunk_decay: Vec<f32>,
    /// Per-position prefix decay Π_{k=cs}^{cs+i} a[k] within the current chunk: `[block_len]`.
    /// Used by Step 4 to avoid O(block_len²) recomputation of chunk-local decay.
    chunk_decay_prefix: Vec<f32>,
    /// `segsum` matrix for intra-chunk attention: `[block_len * block_len]`
    segsum_buf: Vec<f32>,
    /// Log-decay for current chunk (input to `segsum`): `[block_len]`
    log_a_buf: Vec<f32>,
}

impl SsdScratch {
    /// Create scratch space sized for `config` and `seq_len`.
    pub fn new(config: &SsdBlockConfig, seq_len: usize) -> Self {
        let block_len = config.block_len;
        let head_dim = config.head_dim;
        let state_dim = config.state_dim;
        let n_chunks = seq_len.div_ceil(block_len);
        let np = state_dim * head_dim;

        Self {
            chunk_out: vec![0.0; seq_len * head_dim],
            chunk_states: vec![0.0; np * n_chunks],
            inter_state: vec![0.0; np * n_chunks],
            chunk_decay: vec![0.0; n_chunks],
            chunk_decay_prefix: vec![0.0; block_len],
            segsum_buf: vec![0.0; block_len * block_len],
            log_a_buf: vec![0.0; block_len],
        }
    }

    /// Resize scratch for a new `seq_len`. Reuses capacity when possible.
    pub fn for_seq_len(&mut self, config: &SsdBlockConfig, seq_len: usize) {
        let block_len = config.block_len;
        let head_dim = config.head_dim;
        let state_dim = config.state_dim;
        let n_chunks = seq_len.div_ceil(block_len);
        let np = state_dim * head_dim;

        self.chunk_out.resize(seq_len * head_dim, 0.0);
        self.chunk_states.resize(np * n_chunks, 0.0);
        self.inter_state.resize(np * n_chunks, 0.0);
        self.chunk_decay.resize(n_chunks, 0.0);
        self.chunk_decay_prefix.resize(block_len, 0.0);
        self.segsum_buf.resize(block_len * block_len, 0.0);
        self.log_a_buf.resize(block_len, 0.0);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SSD Block Forward (4-step decomposition)
// ────────────────────────────────────────────────────────────────────────────

/// SSD block forward pass — the 4-step block decomposition.
///
/// Computes the semiseparable state space operation:
/// ```text
/// S[0] = 0
/// S[t] = a[t] · S[t-1] + b[t] ⊗ x[t]     (state update, S is [N×P])
/// y[t] = c[t]^T · S[t]                    (output projection)
/// ```
///
/// Equivalently: `y[t][p] = Σ_{j≤t} (Π_{k=j+1}^{t} a[k]) · (b[j]·c[t]) · x[j][p]`
///
/// The input `a` holds **actual decay factors** in [0, 1] (e.g. `sigmoid(gate)`).
/// Internally, `segsum` receives `ln(a)` so that `exp(segsum(ln(a)))` yields
/// `Π a[k]`, matching the direct products used in steps 2–4 and [`ssd_naive`].
///
/// # Layouts (all row-major)
/// * `x` — `[T * head_dim]`
/// * `a` — `[T]` decay factors (shared across all heads/channels)
/// * `b` — `[T * state_dim]`
/// * `c` — `[T * state_dim]`
/// * `out` — `[T * head_dim]`
///
/// # Zero allocation
/// All temporaries are written to `scratch`. The caller must size it via
/// [`SsdScratch::new`] or [`SsdScratch::for_seq_len`].
#[allow(clippy::too_many_lines)]
pub fn ssd_block_forward(
    x: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    config: &SsdBlockConfig,
    out: &mut [f32],
    scratch: &mut SsdScratch,
) {
    let seq_len = a.len();
    let head_dim = config.head_dim;
    let state_dim = config.state_dim;
    let block_len = config.block_len;

    debug_assert_eq!(x.len(), seq_len * head_dim);
    debug_assert_eq!(b.len(), seq_len * state_dim);
    debug_assert_eq!(c.len(), seq_len * state_dim);
    debug_assert_eq!(out.len(), seq_len * head_dim);
    debug_assert!(block_len > 0 && head_dim > 0 && state_dim > 0);

    if seq_len == 0 {
        return;
    }

    let n_chunks = seq_len.div_ceil(block_len);

    // Zero output and scratch
    out.fill(0.0);
    scratch.chunk_out.fill(0.0);
    scratch.chunk_states.fill(0.0);
    scratch.inter_state.fill(0.0);
    scratch.chunk_decay.fill(0.0);

    // ════════════════════════════════════════════════════════════════════════
    // Step 1: Diagonal blocks — intra-chunk quadratic attention
    // ════════════════════════════════════════════════════════════════════════
    //
    // For each chunk of length Q, compute the contribution from source positions
    // j to output positions t where both j, t are in the same chunk and j ≤ t:
    //
    //   chunk_out[t][p] += mask[t,j] · (b[j]·c[t]) · x[j][p]
    //
    // where mask[t,j] = Π_{k=j+1}^{t} a[k]  (causal decay within chunk).
    //
    // We obtain mask via segsum(ln(a)) + exp():
    //   segsum(ln_a)[t,j] = Σ_{k=j+1}^{t} ln(a[k])
    //   exp(segsum(ln_a)[t,j]) = Π_{k=j+1}^{t} a[k]

    for chunk in 0..n_chunks {
        let cs = chunk * block_len;
        let ce = (cs + block_len).min(seq_len);
        let cl = ce - cs;

        // Convert decay to log-space for segsum
        let log_a = &mut scratch.log_a_buf[..cl];
        for i in 0..cl {
            log_a[i] = a[cs + i].ln();
        }

        // segsum(log_a) → segsum_buf[0..cl*cl]
        let seg = &mut scratch.segsum_buf[..cl * cl];
        segsum(log_a, seg);

        // Quadratic attention within the chunk
        for t_local in 0..cl {
            let t_global = cs + t_local;
            let c_t = &c[t_global * state_dim..t_global * state_dim + state_dim];
            let seg_row = t_local * cl;

            for j_local in 0..=t_local {
                // mask = Π_{k=j+1}^{t} a[k]  (empty product = 1.0 on diagonal).
                // For j == t we hardcode 1.0 to avoid ln(0)=-inf causing NaN
                // in segsum when a[k] = 0.
                let mask = if j_local == t_local {
                    1.0
                } else {
                    seg[seg_row + j_local].exp()
                };
                if !mask.is_finite() || mask == 0.0 {
                    continue;
                }

                let j_global = cs + j_local;
                let b_j = &b[j_global * state_dim..j_global * state_dim + state_dim];

                // dot(b[j], c[t]) — contiguous slices, ideal for the SIMD dot kernel.
                let dot_bc = katgpt_core::simd::simd_dot_f32(b_j, c_t, state_dim);

                let weight = mask * dot_bc;
                let x_j = &x[j_global * head_dim..j_global * head_dim + head_dim];
                let out_t =
                    &mut scratch.chunk_out[t_global * head_dim..t_global * head_dim + head_dim];
                // out_t[p] += weight * x_j[p] — SIMD FMA (was scalar loop).
                katgpt_core::simd::simd_fused_scale_acc(out_t, x_j, weight, head_dim);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Step 2: Right factors — input → state per chunk
    // ════════════════════════════════════════════════════════════════════════
    //
    // For each chunk, compute the terminal state (with zero initial state):
    //
    //   S_intra_end[chunk][n][p] = Σ_{t in chunk} (Π_{k=t+1}^{chunk_end} a[k])
    //                              · b[t][n] · x[t][p]
    //
    // This is exactly cumprodsum within the chunk:
    //   h[t] = a[t]·h[t-1] + b[t][n]·x[t][p], h_init = 0
    //   S_intra_end = h at the last position of the chunk.
    //
    // Also compute chunk_decay[chunk] = Π_{k in chunk} a[k] for step 3.
    //
    // Layout: chunk_states[n][p][chunk] (chunk-contiguous for step 3).

    for chunk in 0..n_chunks {
        let cs = chunk * block_len;
        let ce = (cs + block_len).min(seq_len);
        let cl = ce - cs;

        // Per-chunk decay product: Π a[k] for k in [cs, ce)
        let mut dec = 1.0f32;
        for k in cs..ce {
            dec *= a[k];
        }
        scratch.chunk_decay[chunk] = dec;

        // For each (n, p) channel: run cumprodsum within chunk, read final state
        // Input b[t][n]·x[t][p] is strided, so compute the recurrence inline.
        for n in 0..state_dim {
            for p in 0..head_dim {
                let mut h = 0.0f32;
                for t_local in 0..cl {
                    let t = cs + t_local;
                    // FMA: single rounding per recurrence step.
                    h = a[t].mul_add(h, b[t * state_dim + n] * x[t * head_dim + p]);
                }
                // Store at [n][p][chunk] offset
                scratch.chunk_states[n * head_dim * n_chunks + p * n_chunks + chunk] = h;
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Step 3: Center factors — inter-chunk state propagation
    // ════════════════════════════════════════════════════════════════════════
    //
    // The boundary state before chunk c is:
    //   S_boundary[c] = Π_{chunks < c} a · S_boundary_prev + S_intra_end_prev
    //
    // This is a cumprodsum over chunks with decay = chunk_decay and input =
    // chunk_states (S_intra_end per chunk).
    //
    // For each (n, p) channel, run cumprodsum_scalar across the chunk dimension:
    //   inter_state[0] = chunk_decay[0]·0 + chunk_states[0] = S_boundary[1]
    //   inter_state[c] = chunk_decay[c]·inter_state[c-1] + chunk_states[c]
    //                  = S_boundary[c+1]

    for n in 0..state_dim {
        for p in 0..head_dim {
            let off = n * head_dim * n_chunks + p * n_chunks;
            cumprodsum_scalar(
                &scratch.chunk_decay,
                &scratch.chunk_states[off..off + n_chunks],
                0.0,
                &mut scratch.inter_state[off..off + n_chunks],
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Step 4: Left factors — add inter-chunk contribution to output
    // ════════════════════════════════════════════════════════════════════════
    //
    // For position t in chunk c (c ≥ 1), add the contribution from all earlier
    // chunks:
    //
    //   y[t][p] += (Π_{k=cs_c}^{t} a[k]) · Σ_n c[t][n] · S_boundary[c][n][p]
    //
    // where S_boundary[c] = inter_state[c-1] for c ≥ 1, and S_boundary[0] = 0.
    //
    // (Step 1 already wrote the diagonal-block contribution to chunk_out;
    //  we copy it to `out` first, then add the inter-chunk part.)

    out.copy_from_slice(&scratch.chunk_out[..seq_len * head_dim]);

    for chunk in 1..n_chunks {
        let cs = chunk * block_len;
        let ce = (cs + block_len).min(seq_len);
        let cl = ce - cs;

        // Pre-compute cumulative decay Π_{k=cs}^{cs+i} a[k] for i = 0..cl-1.
        //
        // Previously this was recomputed per t with an inner `for k in cs..=t` loop —
        // O(block_len²) per chunk. The prefix product is O(block_len).
        //
        // SAFETY: `chunk_decay_prefix` is sized for the largest chunk length
        // (block_len) by SsdScratch. We use only `cl` slots here.
        let prefix = &mut scratch.chunk_decay_prefix[..cl];
        let mut acc = 1.0f32;
        for i in 0..cl {
            acc *= a[cs + i];
            prefix[i] = acc;
        }

        for t_local in 0..cl {
            let t = cs + t_local;

            // Decay from chunk start to position t: Π_{k=cs}^{t} a[k]
            // (precomputed above as prefix[t_local]).
            let decay = prefix[t_local];

            if decay == 0.0 {
                continue;
            }

            // S_boundary[chunk] = inter_state[chunk - 1], layout [n][p][chunk]
            let sb_chunk = chunk - 1;
            let c_t = &c[t * state_dim..t * state_dim + state_dim];
            let out_t = &mut out[t * head_dim..t * head_dim + head_dim];

            for p in 0..head_dim {
                let mut dot = 0.0f32;
                for n in 0..state_dim {
                    let off = n * head_dim * n_chunks + p * n_chunks + sb_chunk;
                    dot += c_t[n] * scratch.inter_state[off];
                }
                out_t[p] += decay * dot;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Naive quadratic SSD (for verification)
// ────────────────────────────────────────────────────────────────────────────

/// Naive O(T²) SSD computation for correctness verification.
///
/// Computes: `y[t][p] = Σ_{j=0}^{t} (Π_{k=j+1}^{t} a[k]) · (b[j]·c[t]) · x[j][p]`
///
/// # Layouts (all row-major)
/// * `x` — `[T * head_dim]`
/// * `a` — `[T]` decay factors
/// * `b` — `[T * state_dim]`
/// * `c` — `[T * state_dim]`
/// * `out` — `[T * head_dim]`
pub fn ssd_naive(
    x: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    head_dim: usize,
    state_dim: usize,
    out: &mut [f32],
) {
    let seq_len = a.len();
    debug_assert_eq!(x.len(), seq_len * head_dim);
    debug_assert_eq!(b.len(), seq_len * state_dim);
    debug_assert_eq!(c.len(), seq_len * state_dim);
    debug_assert_eq!(out.len(), seq_len * head_dim);

    if seq_len == 0 {
        return;
    }

    out.fill(0.0);

    for t in 0..seq_len {
        let c_t = &c[t * state_dim..t * state_dim + state_dim];
        let out_t = &mut out[t * head_dim..t * head_dim + head_dim];

        for j in 0..=t {
            // decay = Π_{k=j+1}^{t} a[k]
            let mut decay = 1.0f32;
            for k in (j + 1)..=t {
                decay *= a[k];
            }

            if decay == 0.0 {
                continue;
            }

            // dot(b[j], c[t]) — contiguous slices, ideal for the SIMD dot kernel.
            let b_j = &b[j * state_dim..j * state_dim + state_dim];
            let dot_bc = katgpt_core::simd::simd_dot_f32(b_j, c_t, state_dim);

            let weight = decay * dot_bc;
            let x_j = &x[j * head_dim..j * head_dim + head_dim];
            for p in 0..head_dim {
                out_t[p] += weight * x_j[p];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random data via splitmix64-style LCG.
    fn gen_data(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) * scale
            })
            .collect()
    }

    /// Decay factors in [0.5, 0.99] to avoid underflow.
    fn gen_decay(seed: u64, len: usize) -> Vec<f32> {
        gen_data(seed, len, 0.49).iter().map(|&v| 0.5 + v).collect()
    }

    // ── Required tests ──────────────────────────────────────────────────────

    #[test]
    fn test_block_matches_naive() {
        // T=64, head_dim=8, state_dim=4, block_len=64 (single chunk)
        let t = 64;
        let head_dim = 8;
        let state_dim = 4;
        let block_len = 64;

        let x = gen_data(1, t * head_dim, 1.0);
        let a = gen_decay(2, t);
        let b = gen_data(3, t * state_dim, 1.0);
        let c = gen_data(4, t * state_dim, 1.0);

        let config = SsdBlockConfig {
            block_len,
            state_dim,
            head_dim,
        };

        let mut out_block = vec![0.0f32; t * head_dim];
        let mut scratch = SsdScratch::new(&config, t);
        ssd_block_forward(&x, &a, &b, &c, &config, &mut out_block, &mut scratch);

        let mut out_naive = vec![0.0f32; t * head_dim];
        ssd_naive(&x, &a, &b, &c, head_dim, state_dim, &mut out_naive);

        let tol = 1e-3;
        for i in 0..t * head_dim {
            let err = (out_block[i] - out_naive[i]).abs();
            assert!(
                err < tol,
                "Mismatch at {i}: block={:.6}, naive={:.6}, err={:.2e}",
                out_block[i],
                out_naive[i],
                err
            );
        }
        assert!(
            out_naive.iter().any(|&v| v.abs() > 1e-6),
            "Naive output is all zeros"
        );
    }

    #[test]
    fn test_block_matches_naive_large() {
        // T=128, block_len=64 → two chunks, tests inter-chunk propagation
        let t = 128;
        let head_dim = 8;
        let state_dim = 4;
        let block_len = 64;

        let x = gen_data(10, t * head_dim, 1.0);
        let a = gen_decay(20, t);
        let b = gen_data(30, t * state_dim, 1.0);
        let c = gen_data(40, t * state_dim, 1.0);

        let config = SsdBlockConfig {
            block_len,
            state_dim,
            head_dim,
        };

        let mut out_block = vec![0.0f32; t * head_dim];
        let mut scratch = SsdScratch::new(&config, t);
        ssd_block_forward(&x, &a, &b, &c, &config, &mut out_block, &mut scratch);

        let mut out_naive = vec![0.0f32; t * head_dim];
        ssd_naive(&x, &a, &b, &c, head_dim, state_dim, &mut out_naive);

        let tol = 1e-3;
        for i in 0..t * head_dim {
            let err = (out_block[i] - out_naive[i]).abs();
            assert!(
                err < tol,
                "Mismatch at {i}: block={:.6}, naive={:.6}, err={:.2e}",
                out_block[i],
                out_naive[i],
                err
            );
        }
    }

    #[test]
    fn test_auto_block_len() {
        // T=100 → full quadratic
        assert_eq!(auto_block_len(100), 100);
        assert_eq!(auto_block_len(255), 255);

        // T=500 → 64 (SIMD sweet spot)
        assert_eq!(auto_block_len(500), 64);
        assert_eq!(auto_block_len(2047), 64);

        // T=3000 → 128 (GPU tensor core sweet spot)
        assert_eq!(auto_block_len(3000), 128);
        assert_eq!(auto_block_len(2048), 128);

        // Edge: T=0 → 1
        assert_eq!(auto_block_len(0), 1);
        assert_eq!(auto_block_len(1), 1);
    }

    #[test]
    fn test_scratch_reuse() {
        // SsdScratch reused across calls with different seq_lens
        let config = SsdBlockConfig {
            block_len: 64,
            state_dim: 4,
            head_dim: 8,
        };

        let mut scratch = SsdScratch::new(&config, 64);

        // Call 1: T=64 (single chunk)
        let t1 = 64;
        let x1 = gen_data(1, t1 * 8, 1.0);
        let a1 = gen_decay(2, t1);
        let b1 = gen_data(3, t1 * 4, 1.0);
        let c1 = gen_data(4, t1 * 4, 1.0);
        let mut out1 = vec![0.0f32; t1 * 8];
        ssd_block_forward(&x1, &a1, &b1, &c1, &config, &mut out1, &mut scratch);

        // Resize for T=128 (two chunks)
        let t2 = 128;
        scratch.for_seq_len(&config, t2);
        let x2 = gen_data(10, t2 * 8, 1.0);
        let a2 = gen_decay(20, t2);
        let b2 = gen_data(30, t2 * 4, 1.0);
        let c2 = gen_data(40, t2 * 4, 1.0);
        let mut out2 = vec![0.0f32; t2 * 8];
        ssd_block_forward(&x2, &a2, &b2, &c2, &config, &mut out2, &mut scratch);

        // Verify both match naive
        let mut naive1 = vec![0.0f32; t1 * 8];
        ssd_naive(&x1, &a1, &b1, &c1, 8, 4, &mut naive1);
        for i in 0..t1 * 8 {
            assert!((out1[i] - naive1[i]).abs() < 1e-3, "Call 1 mismatch at {i}");
        }

        let mut naive2 = vec![0.0f32; t2 * 8];
        ssd_naive(&x2, &a2, &b2, &c2, 8, 4, &mut naive2);
        for i in 0..t2 * 8 {
            assert!((out2[i] - naive2[i]).abs() < 1e-3, "Call 2 mismatch at {i}");
        }
    }

    #[test]
    fn test_zero_allocation() {
        // Verify scratch reuse: two calls with same scratch produce identical,
        // correct results.
        let config = SsdBlockConfig {
            block_len: 64,
            state_dim: 4,
            head_dim: 8,
        };

        let t = 128;
        let x = gen_data(1, t * 8, 1.0);
        let a = gen_decay(2, t);
        let b = gen_data(3, t * 4, 1.0);
        let c = gen_data(4, t * 4, 1.0);

        let mut scratch = SsdScratch::new(&config, t);

        let mut out1 = vec![0.0f32; t * 8];
        ssd_block_forward(&x, &a, &b, &c, &config, &mut out1, &mut scratch);

        let mut out2 = vec![0.0f32; t * 8];
        ssd_block_forward(&x, &a, &b, &c, &config, &mut out2, &mut scratch);

        // Deterministic: both calls identical
        for i in 0..t * 8 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-6,
                "Non-deterministic at {i}: {:.6} vs {:.6}",
                out1[i],
                out2[i]
            );
        }

        // Both match naive
        let mut naive = vec![0.0f32; t * 8];
        ssd_naive(&x, &a, &b, &c, 8, 4, &mut naive);
        for i in 0..t * 8 {
            assert!(
                (out1[i] - naive[i]).abs() < 1e-3,
                "Block != naive at {i}: {:.6} vs {:.6}",
                out1[i],
                naive[i]
            );
        }
    }

    // ── Additional edge-case tests ──────────────────────────────────────────

    #[test]
    fn test_block_matches_naive_multi_chunk() {
        // T=200, block_len=64 → 4 chunks (64, 64, 64, 8) — irregular last chunk
        let t = 200;
        let head_dim = 4;
        let state_dim = 4;
        let block_len = 64;

        let x = gen_data(100, t * head_dim, 1.0);
        let a = gen_decay(200, t);
        let b = gen_data(300, t * state_dim, 1.0);
        let c = gen_data(400, t * state_dim, 1.0);

        let config = SsdBlockConfig {
            block_len,
            state_dim,
            head_dim,
        };

        let mut out_block = vec![0.0f32; t * head_dim];
        let mut scratch = SsdScratch::new(&config, t);
        ssd_block_forward(&x, &a, &b, &c, &config, &mut out_block, &mut scratch);

        let mut out_naive = vec![0.0f32; t * head_dim];
        ssd_naive(&x, &a, &b, &c, head_dim, state_dim, &mut out_naive);

        let tol = 1e-2; // Looser due to accumulation over 200 positions
        for i in 0..t * head_dim {
            let err = (out_block[i] - out_naive[i]).abs();
            assert!(
                err < tol,
                "Mismatch at {i}: block={:.6}, naive={:.6}, err={:.2e}",
                out_block[i],
                out_naive[i],
                err
            );
        }
    }

    #[test]
    fn test_single_element() {
        let t = 1;
        let head_dim = 2;
        let state_dim = 2;

        let x = vec![1.0, 0.5];
        let a = vec![0.9];
        let b = vec![1.0, 0.8];
        let c = vec![0.5, 0.3];

        let config = SsdBlockConfig {
            block_len: 64,
            state_dim,
            head_dim,
        };

        let mut out_block = vec![0.0f32; t * head_dim];
        let mut scratch = SsdScratch::new(&config, t);
        ssd_block_forward(&x, &a, &b, &c, &config, &mut out_block, &mut scratch);

        // y[0][p] = dot(b[0],c[0]) * x[0][p] = 0.74 * x[0][p]
        let dot = 1.0 * 0.5 + 0.8 * 0.3; // 0.74
        assert!((out_block[0] - dot * 1.0).abs() < 1e-5);
        assert!((out_block[1] - dot * 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_zero_decay_isolates_positions() {
        // a[k]=0 for k>0 → each position only sees itself
        let t = 8;
        let head_dim = 2;
        let state_dim = 2;
        let block_len = 4;

        let x = gen_data(1, t * head_dim, 1.0);
        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = gen_data(3, t * state_dim, 1.0);
        let c = gen_data(4, t * state_dim, 1.0);

        let config = SsdBlockConfig {
            block_len,
            state_dim,
            head_dim,
        };

        let mut out_block = vec![0.0f32; t * head_dim];
        let mut scratch = SsdScratch::new(&config, t);
        ssd_block_forward(&x, &a, &b, &c, &config, &mut out_block, &mut scratch);

        let mut out_naive = vec![0.0f32; t * head_dim];
        ssd_naive(&x, &a, &b, &c, head_dim, state_dim, &mut out_naive);

        for i in 0..t * head_dim {
            assert!(
                (out_block[i] - out_naive[i]).abs() < 1e-5,
                "Mismatch at {i}: block={:.6}, naive={:.6}",
                out_block[i],
                out_naive[i]
            );
        }
    }

    #[test]
    fn test_no_decay_causal_attention() {
        // a[k]=1.0 → SSD becomes standard causal linear attention
        let t = 64;
        let head_dim = 4;
        let state_dim = 4;
        let block_len = 32;

        let x = gen_data(1, t * head_dim, 1.0);
        let a = vec![1.0f32; t];
        let b = gen_data(3, t * state_dim, 1.0);
        let c = gen_data(4, t * state_dim, 1.0);

        let config = SsdBlockConfig {
            block_len,
            state_dim,
            head_dim,
        };

        let mut out_block = vec![0.0f32; t * head_dim];
        let mut scratch = SsdScratch::new(&config, t);
        ssd_block_forward(&x, &a, &b, &c, &config, &mut out_block, &mut scratch);

        let mut out_naive = vec![0.0f32; t * head_dim];
        ssd_naive(&x, &a, &b, &c, head_dim, state_dim, &mut out_naive);

        for i in 0..t * head_dim {
            assert!(
                (out_block[i] - out_naive[i]).abs() < 1e-2,
                "Mismatch at {i}: block={:.6}, naive={:.6}",
                out_block[i],
                out_naive[i]
            );
        }
    }
}
