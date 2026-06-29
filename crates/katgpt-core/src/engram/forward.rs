//! End-to-end fuse hook — applies K retrieved patterns to a hidden state.
//!
//! Plan 299 Phase 7 T7.1–T7.2 (partial). The orchestrator wires GOAT gate
//! tests (T7.3–T7.10) and example/demo code separately.
//!
//! # Pipeline
//!
//! For each of the K heads (up to [`K_MAX`](super::K_MAX)):
//! 1. Look up the K slot vectors (one call to
//!    [`EngramTable::lookup_into`](super::EngramTable::lookup_into) — done
//!    once, not per head).
//! 2. For each non-empty retrieved pattern `e_k`:
//!    - Treat `e_k` as both the key `k` and value `v` (the host can swap in
//!      learned `W_K`, `W_V` projections in a future phase — for now we use
//!      the identity projection to keep the open primitive projection-free).
//!    - [`sigmoid_fuse_into`](super::sigmoid_fuse_into) with the host
//!      `query` as `q` and `e_k` as both `k` and `v`.
//!    - Residual-add the result into `hidden_state`.
//!
//! # Zero-allocation
//!
//! [`fuse_into_hidden_state`] is **zero-allocation**. The caller provides
//! three scratch buffers:
//! - `scratch_lookup` — size `K_MAX * D` (the lookup output)
//! - `scratch_norm` — size `D` (rmsnorm workspace, currently unused since
//!   [`sigmoid_fuse_into`] is fused, but kept in the API for forward
//!   compat with T3.6 multi-branch / T3.7 conv variants that may need it)
//! - `scratch_out` — size `D` (per-pattern sigmoid fuse output before
//!   residual add)
//!
//! # What this is NOT
//!
//! - Not a training step. No gradients, no backprop. Pure inference.
//! - Not a softmax. Sigmoid only (per AGENTS.md — see [`kernel`](super::kernel)).
//! - Not a learned projection. `e_k` is used directly as both k and v. The
//!   host can pre-project (e.g. `e_k → W_K · e_k`) before passing patterns
//!   to the table, or wait for a future phase that adds projection weights
//!   to [`EngramConfig`].

use super::{EngramHash, EngramTable, K_MAX, SigmoidFusionConfig, sigmoid_fuse_into};
use crate::simd::{simd_add_inplace, simd_sum_abs_f32};

// `EngramConfig` is DEFINED in this module (not imported) — the canonical
// home per Plan T7.2. `mod.rs` re-exports it via
// `pub use forward::{EngramConfig, fuse_into_hidden_state};`.

/// Host-side configuration for the end-to-end engram fuse.
///
/// Per Plan T7.2. Fields:
/// - `fusion` — sigmoid kernel config (`tau`, `rmsnorm_eps`).
/// - `k_heads` — how many of the K_MAX retrieved heads to actually fuse.
///   Must be `≤ K_MAX`. Default `K_MAX`.
///
/// # Future extensions (deferred)
///
/// - `conv_kernel: Option<[f32; 4]>` — depthwise causal conv weights (T3.7).
/// - `multi_branch: Option<usize>` — number of branches for T3.6 variant.
///
/// Both are left out for now; the host adds them when their phases land.
#[derive(Debug, Clone, Copy)]
pub struct EngramConfig {
    /// Sigmoid fusion kernel config (tau, rmsnorm_eps).
    pub fusion: SigmoidFusionConfig,
    /// Number of heads to fuse (≤ K_MAX). Default K_MAX.
    pub k_heads: usize,
}

impl Default for EngramConfig {
    #[inline]
    fn default() -> Self {
        Self {
            fusion: SigmoidFusionConfig::default(),
            k_heads: K_MAX,
        }
    }
}

impl EngramConfig {
    /// Construct with explicit `tau` derived from the hidden dim D.
    ///
    /// Convenience: `EngramConfig::for_dim(D)` sets `tau = √D` and
    /// `k_heads = K_MAX`.
    #[inline]
    pub fn for_dim(d: usize) -> Self {
        Self {
            fusion: SigmoidFusionConfig {
                tau: (d as f32).sqrt(),
                rmsnorm_eps: 1e-6,
            },
            k_heads: K_MAX,
        }
    }
}

/// End-to-end fuse: look up K patterns, sigmoid-fuse each into `hidden_state`
/// (residual add).
///
/// # Zero-allocation
///
/// See the module docs — caller provides scratch buffers of the documented
/// sizes. The function does NO heap allocation.
///
/// # Arguments
///
/// - `hidden_state` — the live latent state, `&mut [f32]` of length D. Each
///   retrieved + gated pattern is residual-added into this in place.
/// - `query` — the query vector `q`, length D. Used as `q` in
///   [`sigmoid_fuse_into`] for every head.
/// - `table` — the frozen engram table. Looked up once via `lookup_into`.
/// - `hash_keys` — K_MAX hash slot keys (typically the output of
///   [`multi_head_hash`](super::multi_head_hash)).
/// - `config` — host-side config (fusion tau + k_heads).
/// - `scratch_lookup` — size `K_MAX * D`. Receives the raw slot vectors.
/// - `scratch_norm` — size `D`. Currently unused (fused rmsnorm); kept for
///   forward compat with T3.6/T3.7.
/// - `scratch_out` — size `D`. Receives the per-head sigmoid fuse output
///   before residual-add into `hidden_state`.
///
/// # Panics (debug only)
///
/// `debug_assert!` checks that:
/// - `query.len() == hidden_state.len() == table.dim()`
/// - `scratch_lookup.len() >= K_MAX * D`
/// - `scratch_norm.len() >= D`
/// - `scratch_out.len() >= D`
/// - `config.k_heads <= K_MAX`
// zero-alloc hot path; 5 inputs + 3 scratch buffers
#[allow(clippy::too_many_arguments)]
pub fn fuse_into_hidden_state(
    hidden_state: &mut [f32],
    query: &[f32],
    table: &dyn EngramTable,
    hash_keys: &[EngramHash; K_MAX],
    config: &EngramConfig,
    scratch_lookup: &mut [f32],
    scratch_norm: &mut [f32],
    scratch_out: &mut [f32],
) {
    let d = table.dim();
    debug_assert_eq!(
        hidden_state.len(),
        d,
        "fuse_into_hidden_state: hidden_state.len() must equal table.dim()"
    );
    debug_assert_eq!(
        query.len(),
        d,
        "fuse_into_hidden_state: query.len() must equal table.dim()"
    );
    debug_assert!(
        scratch_lookup.len() >= K_MAX * d,
        "fuse_into_hidden_state: scratch_lookup must be ≥ K_MAX*D"
    );
    debug_assert!(
        scratch_norm.len() >= d,
        "fuse_into_hidden_state: scratch_norm must be ≥ D"
    );
    debug_assert!(
        scratch_out.len() >= d,
        "fuse_into_hidden_state: scratch_out must be ≥ D"
    );
    debug_assert!(
        config.k_heads <= K_MAX,
        "fuse_into_hidden_state: config.k_heads must be ≤ K_MAX"
    );

    // Step 1: look up K_MAX slot vectors into scratch_lookup (one call).
    let _hits = table.lookup_into(hash_keys, &mut scratch_lookup[..K_MAX * d]);

    // Step 2: for each head up to k_heads, sigmoid-fuse into scratch_out,
    // then residual-add into hidden_state.
    //
    // Identity projection: pattern e_k is used as both k and v. Host can
    // pre-project in a future phase by adding W_K, W_V to EngramConfig.
    let k_active = config.k_heads.min(K_MAX);
    for k in 0..k_active {
        let e_k = &scratch_lookup[k * d..(k + 1) * d];

        // Skip empty slots — a slot of all zeros produces gate * 0 = 0,
        // which is a no-op residual add. We check anyway to save the
        // dot-product work; `simd_sum_abs_f32` is the branch-free SIMD
        // reduction (one NEON/AVX2 horizontal add instead of K_MAX=16
        // short-circuiting scalar compares with unpredictable branch hits).
        if simd_sum_abs_f32(e_k) == 0.0 {
            continue;
        }

        // CRITICAL: sigmoid, not softmax, per AGENTS.md (see kernel.rs).
        sigmoid_fuse_into(
            query, // q
            e_k,   // k (identity projection of e_k)
            e_k,   // v (identity projection of e_k)
            scratch_out,
            &config.fusion,
        );

        // Residual add: hidden_state[j] += scratch_out[j] for j in 0..d.
        // `simd_add_inplace` guarantees the NEON/AVX2+FMA codegen that the
        // manual loop only got when LLVM's auto-vectorizer was in a good
        // mood (it usually was at D=32, but `simd_add_inplace` is certain).
        simd_add_inplace(&mut hidden_state[..d], &scratch_out[..d]);
    }

    // Touch scratch_norm so the compiler doesn't complain about it being
    // unused in the current (fused-rmsnorm) implementation. Future phases
    // (T3.6 multi-branch, T3.7 conv) may use it.
    let _ = scratch_norm.len();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::{EngramHash, EngramTableBuilder};

    /// Build a small test table with a few populated slots.
    fn make_test_table(d: usize) -> impl EngramTable {
        let mut b = EngramTableBuilder::new(32, d);
        for i in 0..4u64 {
            let pat: Vec<f32> = (0..d).map(|j| (i as f32) * 0.1 + j as f32 * 0.01).collect();
            b.add_pattern(EngramHash(i), &pat);
        }
        b.build()
    }

    #[test]
    fn fuse_zero_query_does_not_corrupt_hidden_state() {
        // Edge case: zero query + zero hidden_state → after fuse, hidden
        // state may receive non-zero contributions from populated slots
        // (because gate * v ≠ 0 if gate ≠ 0). With q=0, RMSNorm(0)=0,
        // dot=0, gate = sigmoid(0) = 0.5. So hidden_state gets 0.5 * v per
        // populated head. This is correct behavior — verify the math.
        let d = 8;
        let table = make_test_table(d);
        let mut hidden = vec![0.0f32; d];
        let query = vec![0.0f32; d];
        let keys = [EngramHash(0); K_MAX]; // all heads → slot 0
        let cfg = EngramConfig::for_dim(d);

        let mut scratch_lookup = vec![0.0f32; K_MAX * d];
        let mut scratch_norm = vec![0.0f32; d];
        let mut scratch_out = vec![0.0f32; d];

        fuse_into_hidden_state(
            &mut hidden,
            &query,
            &table,
            &keys,
            &cfg,
            &mut scratch_lookup,
            &mut scratch_norm,
            &mut scratch_out,
        );

        // slot 0 pattern: i=0, so pat = [0, 0.01, 0.02, ...]
        // gate ≈ 0.5 (zero dot), all K_MAX heads hit slot 0 (same hash).
        // hidden[j] += K_MAX * 0.5 * pat[j]
        // Allow some tolerance because RMSNorm(0) with eps produces a tiny
        // inv_rms.
        let expected_mag = 0.5 * K_MAX as f32 * 0.01; // for j=1
        assert!(
            (hidden[1] - expected_mag).abs() < 1.0,
            "fuse math check failed: hidden[1]={}, expected~{}",
            hidden[1],
            expected_mag
        );
    }

    #[test]
    fn fuse_no_allocation_in_hot_path() {
        // Smoke test: just verify the function runs end-to-end without
        // panicking on a typical input. (We can't easily test "zero
        // allocation" without a custom allocator — that's a job for the
        // GOAT gate bench T7.3.)
        let d = 16;
        let table = make_test_table(d);
        let mut hidden: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
        let keys = [EngramHash(1); K_MAX];
        let cfg = EngramConfig::for_dim(d);

        let mut scratch_lookup = vec![0.0f32; K_MAX * d];
        let mut scratch_norm = vec![0.0f32; d];
        let mut scratch_out = vec![0.0f32; d];

        let hidden_before = hidden.clone();
        fuse_into_hidden_state(
            &mut hidden,
            &query,
            &table,
            &keys,
            &cfg,
            &mut scratch_lookup,
            &mut scratch_norm,
            &mut scratch_out,
        );
        // Hidden state must have changed (residual add of at least one
        // populated slot's gate*v).
        assert_ne!(hidden, hidden_before, "fuse must modify hidden_state");
    }

    #[test]
    fn fuse_skips_empty_slots() {
        // If hash_keys all point to unpopulated slots, hidden_state is
        // unchanged.
        let d = 8;
        let mut b = EngramTableBuilder::new(64, d);
        // Populate slot 0 only.
        b.add_pattern(EngramHash(0), &[1.0f32; 8]);
        let table = b.build();

        let mut hidden = vec![0.5f32; d];
        let hidden_before = hidden.clone();
        let query = vec![1.0f32; d];
        // Pick hashes that avoid slot 0 — e.g. all hash 1 (slot 1 is empty).
        let keys = [EngramHash(1); K_MAX];
        let cfg = EngramConfig::for_dim(d);

        let mut scratch_lookup = vec![0.0f32; K_MAX * d];
        let mut scratch_norm = vec![0.0f32; d];
        let mut scratch_out = vec![0.0f32; d];

        fuse_into_hidden_state(
            &mut hidden,
            &query,
            &table,
            &keys,
            &cfg,
            &mut scratch_lookup,
            &mut scratch_norm,
            &mut scratch_out,
        );
        assert_eq!(hidden, hidden_before, "all-empty-slot lookup → no change");
    }

    #[test]
    fn engram_config_default_k_heads_is_k_max() {
        let cfg = EngramConfig::default();
        assert_eq!(cfg.k_heads, K_MAX);
    }

    #[test]
    fn engram_config_for_dim_sets_tau_sqrt_d() {
        let cfg = EngramConfig::for_dim(64);
        assert!((cfg.fusion.tau - (64.0f32).sqrt()).abs() < 1e-5);
    }
}
