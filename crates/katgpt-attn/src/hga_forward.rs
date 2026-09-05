//! HGA forward path — three-stage chunk→group→token routing.
//!
//! Plan 397 T1.8 + T1.9, Research 379 §1.2.
//!
//! # Architecture
//!
//! ```text
//! Stage 1: Chunk-level entmax routing
//!   - Score each chunk's summary against the query (dot-product).
//!   - Apply α-entmax (α=1.5) for sparse chunk selection.
//!   - Select top-k_c chunks.
//!
//! Stage 2: Group-level dot-product routing
//!   - Within selected chunks, score each group's summary against the query.
//!   - Select top-k_g groups (deterministic top-K, no softmax — this is routing,
//!     not attention).
//!
//! Stage 3: Route-and-fetch + exact softmax
//!   - Fetch real token K/V for sink + local + selected groups from the tiered store.
//!   - Run standard causal SDPA (softmax) over the fetched token K/V.
//!   - Summary K/V NEVER enters the output — the "summary-keys-route, real-keys-compute" rule.
//! ```
//!
//! # AGENTS.md compliance
//!
//! - **sigmoid not softmax** — chunk and group selection use dot-product scoring +
//!   top-K / entmax (deterministic routing), NOT softmax. Softmax is used only for
//!   the final output attention over fetched real-token K/V (standard SDPA —
//!   retrieval, not gating).

#![cfg(all(feature = "hga", feature = "dash_attn"))]

use crate::dash_attn::entmax::entmax_1p5;
use katgpt_core::hga::GroupSummaryCache;
use katgpt_core::simd::simd_dot_f32;
use katgpt_core::tiered_kv::{
    GroupSelection, InMemoryTieredKvStore, SinkLocalSet, TieredKvStore, WorkingSet,
};

/// Output of the HGA forward pass: the attention output vector + diagnostics.
pub struct HgaOutput {
    /// `[D]` attention output vector.
    pub out: Vec<f32>,
    /// Number of tokens in the fetched working set.
    pub n_fetched: usize,
    /// Number of chunks selected at stage 1.
    pub n_selected_chunks: usize,
    /// Number of groups selected at stage 2.
    pub n_selected_groups: usize,
}

/// HGA forward pass: three-stage chunk→group→token routing.
///
/// This is a **standalone** function that takes raw tensors — NOT wired into the
/// full transformer ForwardContext dispatch. Phase 2's job is to wire it in.
///
/// # Arguments
///
/// - `query` — `[D]` query vector (already RoPE-rotated at the query position).
/// - `store` — the tiered K/V store (holds all chunks' K/V + summaries).
/// - `group_cache` — the group summary cache (holds per-chunk per-group summaries).
/// - `sink_local` — which chunks are always-visible (sink + local window).
/// - `chunk_summaries` — `[n_chunks * D]` per-chunk summary vectors for stage-1 scoring.
/// - `budget` — route budget (k_c chunks, k_g groups). `RouteBudget::FULL` = causal SDPA.
///
/// # Returns
///
/// `HgaOutput` with the `[D]` attention output and diagnostics.
///
/// # Panics
///
/// If `store.n_chunks() != group_cache.n_chunks()` or dimensions mismatch.
pub fn forward_hga(
    query: &[f32],
    store: &InMemoryTieredKvStore<impl Fn(&[f32], &[usize], usize, usize) -> Vec<f32>>,
    group_cache: &GroupSummaryCache,
    sink_local: &SinkLocalSet,
    chunk_summaries: &[f32],
    budget: katgpt_core::tiered_kv::RouteBudget,
) -> HgaOutput {
    let d = store.head_dim();
    let n_chunks = store.n_chunks();
    assert_eq!(
        group_cache.n_chunks(),
        n_chunks,
        "store/group_cache chunk count mismatch"
    );
    assert_eq!(
        chunk_summaries.len(),
        n_chunks * d,
        "chunk_summaries must be [n_chunks * D]"
    );

    if n_chunks == 0 {
        return HgaOutput {
            out: vec![0.0; d],
            n_fetched: 0,
            n_selected_chunks: 0,
            n_selected_groups: 0,
        };
    }

    // ── Stage 1: chunk-level scoring + entmax + top-k_c ──────────────────────
    let mut chunk_scores = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        let summary = &chunk_summaries[c * d..(c + 1) * d];
        let score = simd_dot_f32(query, summary, d);
        chunk_scores.push(score);
    }

    // Entmax-1.5 over chunk scores for sparse selection.
    let (chunk_probs, _tau) = entmax_1p5(&chunk_scores);

    // Select top-k_c chunks by entmax probability (or all if budget is FULL).
    //
    // The FULL branch is tested first: previously the probability-ranked
    // `chunk_indices_scored` Vec was always built, sorted (O(n log n)) and
    // collected into a `selected_chunks` Vec that the FULL override then threw
    // away. Hoisting the check skips two allocations plus the sort on that path,
    // and drops the always-discarded intermediate collect on the budgeted path.
    // The budgeted path's filter → sort → take sequence is unchanged.
    let selected_chunks: Vec<usize> = if budget.k_c == usize::MAX {
        (0..n_chunks).collect()
    } else {
        let mut chunk_indices_scored: Vec<(usize, f32)> = chunk_probs
            .iter()
            .enumerate()
            .filter(|(_, p)| **p > 0.0)
            .map(|(i, p)| (i, *p))
            .collect();
        chunk_indices_scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let max_chunks = budget.k_c.min(n_chunks);
        chunk_indices_scored.truncate(max_chunks);
        chunk_indices_scored.into_iter().map(|(i, _)| i).collect()
    };

    // ── Stage 2: group-level scoring + top-k_g ───────────────────────────────
    let group_selection = if budget.k_g == usize::MAX {
        // Full coverage: all groups in all selected chunks.
        GroupSelection::all_groups(n_chunks, group_cache.n_groups_per_chunk())
    } else {
        group_cache.select_top_k_groups(query, &selected_chunks, budget.k_g)
    };

    let n_selected_groups: usize = group_selection.selections.iter().map(|(_, _, n)| *n).sum();

    // ── Stage 3: fetch working set + exact softmax SDPA ──────────────────────
    let working_set = store.fetch_working_set(sink_local, &selected_chunks, &group_selection);

    let out = sdpa(query, &working_set, d);

    HgaOutput {
        out,
        n_fetched: working_set.n_tokens,
        n_selected_chunks: selected_chunks.len(),
        n_selected_groups,
    }
}

/// Standard scaled dot-product attention (softmax) over a working set.
///
/// `logit_j = (q · k_j) / sqrt(D)`, `out = Σ_j softmax(logit_j) · v_j`.
///
/// This is standard SDPA — softmax is correct here (retrieval over fetched
/// real tokens, not a gating/routing decision).
fn sdpa(query: &[f32], ws: &WorkingSet, d: usize) -> Vec<f32> {
    if ws.n_tokens == 0 {
        return vec![0.0; d];
    }

    let inv_sqrt_d = 1.0 / (d as f32).sqrt();
    let n = ws.n_tokens;

    // Compute logits with pre-computed inv_sqrt_d (multiply, not divide).
    let mut logits = vec![0.0f32; n];
    for (j, logit) in logits.iter_mut().enumerate().take(n) {
        let k = &ws.keys[j * d..(j + 1) * d];
        *logit = simd_dot_f32(query, k, d) * inv_sqrt_d;
    }

    // Max-shift via SIMD reduction (branch-free horizontal max).
    let max_logit = katgpt_core::simd::simd_max_f32(&logits[..n]);

    // Shifted exp + sum via SIMD primitives.
    katgpt_core::simd::simd_add_scalar_inplace(&mut logits[..n], -max_logit);
    katgpt_core::simd::simd_exp_inplace(&mut logits[..n]);
    let sum_exp = katgpt_core::simd::simd_sum_f32(&logits[..n]);

    // Weighted sum of values: fuse inv into each weight, then SIMD scale-acc.
    let mut out = vec![0.0f32; d];
    if sum_exp > 0.0 {
        let inv = 1.0 / sum_exp;
        for (j, &weight_unscaled) in logits.iter().enumerate().take(n) {
            let weight = weight_unscaled * inv;
            let v = &ws.values[j * d..(j + 1) * d];
            katgpt_core::simd::simd_fused_scale_acc(&mut out[..d], v, weight, d);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::hga::MixedRopeSummarizer;
    use katgpt_core::tiered_kv::RouteBudget;

    /// Simple mean summarizer for the store (matches the one in tiered_kv tests).
    fn mean_summarizer(
        keys_flat: &[f32],
        positions: &[usize],
        group_start: usize,
        n_tokens: usize,
    ) -> Vec<f32> {
        let total_tokens = positions.len();
        let d = keys_flat.len().checked_div(total_tokens).unwrap_or(8);
        let mut summary = vec![0.0f32; d];
        for t in 0..n_tokens {
            let offset = (group_start + t) * d;
            for i in 0..d {
                summary[i] += keys_flat[offset + i];
            }
        }
        let inv = 1.0 / n_tokens as f32;
        for x in summary.iter_mut() {
            *x *= inv;
        }
        summary
    }

    /// Compute chunk summaries (mean of all keys in the chunk) for stage-1 scoring.
    fn compute_chunk_summaries(
        store: &InMemoryTieredKvStore<impl Fn(&[f32], &[usize], usize, usize) -> Vec<f32>>,
    ) -> Vec<f32> {
        let d = store.head_dim();
        let c = store.chunk_size();
        let n = store.n_chunks();
        let mut out = vec![0.0f32; n * d];
        for chunk_idx in 0..n {
            let keys = store.keys_for_chunk(chunk_idx);
            let mut sum = vec![0.0f32; d];
            for t in 0..c {
                for i in 0..d {
                    sum[i] += keys[t * d + i];
                }
            }
            let inv = 1.0 / c as f32;
            for i in 0..d {
                out[chunk_idx * d + i] = sum[i] * inv;
            }
        }
        out
    }

    /// T1.12: full-coverage HGA = causal SDPA within < 1e-5.
    #[test]
    fn forward_hga_full_coverage_equals_causal_sdpa() {
        let d = 8;
        let chunk_size = 4;
        let group_size = 2;
        let rope_theta = 10000.0;

        let summarizer = MixedRopeSummarizer::from_rope_theta(d, rope_theta, group_size);
        let mut group_cache = GroupSummaryCache::new(d, chunk_size, group_size, summarizer);
        let mut store = InMemoryTieredKvStore::new(d, chunk_size, group_size, mean_summarizer);

        // Append 5 chunks (20 tokens total) with deterministic data.
        let mut rng = fastrand::Rng::with_seed(42);
        let mut all_keys = Vec::new();
        let mut all_values = Vec::new();
        for chunk_idx in 0..5 {
            let keys: Vec<f32> = (0..chunk_size * d).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let values: Vec<f32> = (0..chunk_size * d).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let positions: Vec<usize> = (chunk_idx * chunk_size..).take(chunk_size).collect();
            all_keys.extend_from_slice(&keys);
            all_values.extend_from_slice(&values);
            store.append_chunk(&keys, &values, &positions);
            group_cache.append_chunk(&keys, &positions);
        }

        let n_tokens = 5 * chunk_size; // 20 tokens

        // Query at position 20 (after all tokens).
        let query: Vec<f32> = (0..d).map(|_| rng.f32() * 2.0 - 1.0).collect();

        // Full-coverage HGA.
        let sink_local = SinkLocalSet::new(vec![], (0..5).collect()); // all chunks local
        let chunk_sums = compute_chunk_summaries(&store);
        let hga_out = forward_hga(
            &query,
            &store,
            &group_cache,
            &sink_local,
            &chunk_sums,
            RouteBudget::FULL,
        );

        // Reference: causal SDPA over all 20 tokens.
        let sqrt_d = (d as f32).sqrt();
        let mut ref_logits = vec![0.0f32; n_tokens];
        let mut max_logit = f32::NEG_INFINITY;
        for j in 0..n_tokens {
            ref_logits[j] = simd_dot_f32(&query, &all_keys[j * d..(j + 1) * d], d) / sqrt_d;
            if ref_logits[j] > max_logit {
                max_logit = ref_logits[j];
            }
        }
        let mut sum_exp = 0.0f32;
        for l in ref_logits.iter_mut() {
            *l = (*l - max_logit).exp();
            sum_exp += *l;
        }
        let mut ref_out = vec![0.0f32; d];
        let inv = 1.0 / sum_exp;
        for j in 0..n_tokens {
            let w = ref_logits[j] * inv;
            for i in 0..d {
                ref_out[i] += w * all_values[j * d + i];
            }
        }

        // Compare — should match within f32 noise.
        let mut max_diff = 0.0f32;
        for (a, b) in hga_out.out.iter().zip(ref_out.iter()) {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 1e-5,
            "HGA full-coverage output differs from causal SDPA by {max_diff} (should be < 1e-5)"
        );
        assert_eq!(
            hga_out.n_fetched, n_tokens,
            "full coverage should fetch all tokens"
        );
    }

    /// Verify that sparse routing fetches fewer tokens than full coverage.
    #[test]
    fn forward_hga_sparse_fetches_fewer_tokens() {
        let d = 8;
        let chunk_size = 4;
        let group_size = 2;

        let summarizer = MixedRopeSummarizer::from_rope_theta(d, 10000.0, group_size);
        let mut group_cache = GroupSummaryCache::new(d, chunk_size, group_size, summarizer);
        let mut store = InMemoryTieredKvStore::new(d, chunk_size, group_size, mean_summarizer);

        let mut rng = fastrand::Rng::with_seed(99);
        for chunk_idx in 0..10 {
            let keys: Vec<f32> = (0..chunk_size * d).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let values: Vec<f32> = (0..chunk_size * d).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let positions: Vec<usize> = (chunk_idx * chunk_size..).take(chunk_size).collect();
            store.append_chunk(&keys, &values, &positions);
            group_cache.append_chunk(&keys, &positions);
        }

        let query: Vec<f32> = (0..d).map(|_| rng.f32()).collect();
        let sink_local = SinkLocalSet::new(vec![0], vec![9]); // first + last chunk
        let chunk_sums = compute_chunk_summaries(&store);

        // Sparse: 3 chunks, 4 groups.
        let sparse_out = forward_hga(
            &query,
            &store,
            &group_cache,
            &sink_local,
            &chunk_sums,
            katgpt_core::tiered_kv::RouteBudget { k_c: 3, k_g: 4 },
        );

        // Should fetch fewer than the full 40 tokens.
        assert!(
            sparse_out.n_fetched < 40,
            "sparse should fetch < 40 tokens, got {}",
            sparse_out.n_fetched
        );
        // Sink (4) + local (4) + routed groups should be ≥ 8.
        assert!(
            sparse_out.n_fetched >= 8,
            "should fetch at least sink+local = 8 tokens, got {}",
            sparse_out.n_fetched
        );
    }
}
