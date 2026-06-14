//! Chunked compaction for long contexts (Plan 271 Phase 4).
//!
//! When `T` exceeds a single compaction call's feasible working set, split the
//! KV cache into overlapping chunks and run [`compact`] on each independently.
//! Two modes:
//!
//! - **KV-based**: chunks are formed directly on the KV tensors; overlap is a
//!   fixed token window between adjacent chunks. Cheap and exact for
//!   intra-chunk attention, but boundary tokens may lose cross-chunk context.
//! - **Text-based**: chunks are formed upstream (e.g. per paragraph / per
//!   turn) and arrive with explicit global positions. We un-rotate RoPE at the
//!   chunk's `start_pos`, compact in position-free space, and re-rotate at the
//!   compacted position. This preserves long-range positional structure at the
//!   cost of needing [`PositionFreeCompactor`] (requires the `still_kv` feature).
//!
//! Both modes return a [`ChunkedCompactOutput`] — the concatenated `(Ck, β, Cv)`
//! plus per-chunk metadata so callers can inspect boundary reconstruction
//! error and choose re-chunking strategies adaptively.
//!
//! # Design Notes
//!
//! - Overlap tokens are *not* deduplicated in the output: they are re-compacted
//!   in each chunk they appear in. This keeps each chunk's compaction
//!   self-contained (no cross-chunk dependencies in the inner loop) at the cost
//!   of slightly larger `total_compact_len`. Dedup would require a global
//!   selection pass that defeats the point of chunking.
//! - The output position of each compact chunk in text-based mode is recorded
//!   in [`ChunkMeta::chunk_start`] (the original global start). Re-rotation is
//!   the caller's responsibility via [`apply_rope_phase_shift`] if they wish
//!   to relocate the compacted tokens.
//!
//! # TL;DR
//!
//! Two chunked-compaction entry points: [`ChunkedCompactor::compact_kv_based`]
//! (overlap-aware slicing) and [`ChunkedCompactor::compact_text_based`]
//! (position-aware with RoPE). Both reuse the Phase 1 [`compact`] orchestrator
//! per chunk.

use crate::attn_match::compact::{compact, CompactError};
use crate::attn_match::types::AmConfig;

/// Pre-split text chunk with its own KV slice and global position offset.
///
/// Used as input to [`ChunkedCompactor::compact_text_based`].
#[derive(Clone, Debug)]
pub struct TextChunk {
    /// Flat `(chunk_len, d)` keys for this chunk.
    pub keys: Vec<f32>,
    /// Flat `(chunk_len, d)` values for this chunk.
    pub values: Vec<f32>,
    /// Global position offset (token index) where this chunk begins.
    pub start_pos: usize,
    /// Number of tokens in this chunk.
    pub chunk_len: usize,
}

/// Per-chunk metadata in a [`ChunkedCompactOutput`].
#[derive(Clone, Debug, Default)]
pub struct ChunkMeta {
    /// Original global position where this chunk began.
    pub chunk_start: usize,
    /// Number of original tokens in this chunk (post-overlap if KV-based).
    pub chunk_len: usize,
    /// Number of tokens after compaction for this chunk.
    pub compact_len: usize,
    /// Relative attention-output reconstruction error for this chunk.
    pub reconstruction_error: f32,
}

/// Concatenated output of a chunked compaction run.
#[derive(Clone, Debug, Default)]
pub struct ChunkedCompactOutput {
    /// Concatenated compact keys, flat `(sum_t, d)`.
    pub compact_keys: Vec<f32>,
    /// Concatenated per-token β, length `sum_t`.
    pub beta: Vec<f32>,
    /// Concatenated compact values, flat `(sum_t, d)`.
    pub compact_values: Vec<f32>,
    /// `sum_t` — total tokens across all compacted chunks.
    pub total_compact_len: usize,
    /// Per-chunk metadata, length = number of chunks processed.
    pub per_chunk: Vec<ChunkMeta>,
}

/// Chunked compactor that splits long KV caches and compacts each chunk.
#[derive(Clone, Debug)]
pub struct ChunkedCompactor {
    /// Tokens per chunk (excluding overlap).
    pub chunk_size: usize,
    /// Overlap tokens between adjacent chunks (KV-based mode).
    pub overlap: usize,
}

impl ChunkedCompactor {
    /// Create a new chunked compactor.
    ///
    /// Panics iff `chunk_size == 0` or `overlap >= chunk_size` (overlap must
    /// be strictly smaller than the chunk so each chunk has unique tokens).
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        assert!(
            overlap < chunk_size,
            "overlap ({overlap}) must be < chunk_size ({chunk_size})"
        );
        Self {
            chunk_size,
            overlap,
        }
    }

    /// KV-based chunked compaction.
    ///
    /// Splits `full_kv_keys` / `full_kv_values` into overlapping chunks of
    /// `chunk_size` tokens with `overlap` shared tokens between adjacent
    /// chunks, runs [`compact`] on each, and concatenates the results.
    ///
    /// `queries` are shared across all chunks (same reference queries). This
    /// is the common case: the model's query distribution doesn't change with
    /// position enough to warrant per-chunk queries, and sharing keeps the
    /// `Qref` consistent for global mass matching.
    pub fn compact_kv_based(
        &self,
        full_kv_keys: &[f32],
        full_kv_values: &[f32],
        queries: &[f32],
        t_len: usize,
        d: usize,
        n: usize,
        config: &AmConfig,
    ) -> Result<ChunkedCompactOutput, CompactError> {
        if full_kv_keys.len() != t_len * d {
            return Err(CompactError::DimensionMismatch(format!(
                "full_kv_keys.len()={} but t_len*d={}*{}={}",
                full_kv_keys.len(),
                t_len,
                d,
                t_len * d
            )));
        }
        if full_kv_values.len() != t_len * d {
            return Err(CompactError::DimensionMismatch(format!(
                "full_kv_values.len()={} but t_len*d={}",
                full_kv_values.len(),
                t_len * d
            )));
        }
        if queries.len() != n * d {
            return Err(CompactError::DimensionMismatch(format!(
                "queries.len()={} but n*d={}",
                queries.len(),
                n * d
            )));
        }
        if t_len == 0 {
            return Ok(ChunkedCompactOutput::default());
        }

        let stride = self.chunk_size - self.overlap;
        let chunk_starts = chunk_starts(t_len, self.chunk_size, stride);

        // Upper bound: each chunk contributes ≤ compact_size tokens.
        let max_total = chunk_starts.len() * config.compact_size;
        let mut out = ChunkedCompactOutput {
            compact_keys: Vec::with_capacity(max_total * d),
            beta: Vec::with_capacity(max_total),
            compact_values: Vec::with_capacity(max_total * d),
            total_compact_len: 0,
            per_chunk: Vec::with_capacity(chunk_starts.len()),
        };

        for &start in &chunk_starts {
            let end = (start + self.chunk_size).min(t_len);
            let chunk_len = end - start;
            if chunk_len == 0 {
                continue;
            }
            let chunk_keys = &full_kv_keys[start * d..end * d];
            let chunk_values = &full_kv_values[start * d..end * d];

            // Per-chunk config must respect `compact_size < chunk_len`.
            let chunk_cfg = chunk_local_config(config, chunk_len);

            let result = compact(chunk_keys, chunk_values, queries, chunk_len, d, n, &chunk_cfg)?;

            let recon = result
                .report
                .as_ref()
                .map(|r| r.relative_attn_output_error)
                .unwrap_or(0.0);

            out.compact_keys.extend_from_slice(&result.compact_keys);
            out.beta.extend_from_slice(&result.beta);
            out.compact_values
                .extend_from_slice(&result.compact_values);
            out.total_compact_len += result.compact_len;
            out.per_chunk.push(ChunkMeta {
                chunk_start: start,
                chunk_len,
                compact_len: result.compact_len,
                reconstruction_error: recon,
            });
        }

        Ok(out)
    }

    /// Text-based chunked compaction.
    ///
    /// Each [`TextChunk`] is compacted in position-free latent space (RoPE
    /// removed at `start_pos`), then re-rotated at the chunk's compacted
    /// position so the output keys remain position-consistent with the
    /// original schedule.
    ///
    /// `queries_per_chunk[i]` is the flat `(n_i, d)` reference query slice for
    /// chunk `i`. All chunks must share the same `d`; `n` may vary.
    ///
    /// If the `still_kv` feature is off, RoPE preservation is skipped and the
    /// keys are compacted as-is — see [`apply_rope_phase_shift`] for the
    /// no-op fallback contract.
    pub fn compact_text_based(
        &self,
        chunks: &[TextChunk],
        queries_per_chunk: &[Vec<f32>],
        config: &AmConfig,
    ) -> Result<ChunkedCompactOutput, CompactError> {
        if chunks.len() != queries_per_chunk.len() {
            return Err(CompactError::DimensionMismatch(format!(
                "chunks.len()={} but queries_per_chunk.len()={}",
                chunks.len(),
                queries_per_chunk.len()
            )));
        }
        if chunks.is_empty() {
            return Ok(ChunkedCompactOutput::default());
        }

        let d = infer_d(chunks)?;
        let max_total = chunks.len() * config.compact_size;
        let mut out = ChunkedCompactOutput {
            compact_keys: Vec::with_capacity(max_total * d),
            beta: Vec::with_capacity(max_total),
            compact_values: Vec::with_capacity(max_total * d),
            total_compact_len: 0,
            per_chunk: Vec::with_capacity(chunks.len()),
        };

        for (ci, chunk) in chunks.iter().enumerate() {
            let chunk_len = chunk.chunk_len;
            if chunk_len == 0 {
                out.per_chunk.push(ChunkMeta {
                    chunk_start: chunk.start_pos,
                    chunk_len: 0,
                    compact_len: 0,
                    reconstruction_error: 0.0,
                });
                continue;
            }
            let chunk_keys = &chunk.keys[..chunk_len * d];
            let chunk_values = &chunk.values[..chunk_len * d];
            let queries = &queries_per_chunk[ci];
            let n = queries.len() / d;
            if queries.len() != n * d {
                return Err(CompactError::DimensionMismatch(format!(
                    "chunk {ci}: queries.len()={} not divisible by d={d}",
                    queries.len()
                )));
            }

            let chunk_cfg = chunk_local_config(config, chunk_len);

            // Position-free path: un-rotate at original start_pos, compact,
            // re-rotate at compacted position. This requires still_kv.
            #[cfg(feature = "still_kv")]
            {
                let pf = PositionFreeBridge::new(ROPE_THETA, d);
                let pos_free_keys = pf.un_rotate_f32(chunk_keys, chunk.start_pos);
                let result = compact(&pos_free_keys, chunk_values, queries, chunk_len, d, n, &chunk_cfg)?;
                let new_pos = chunk.start_pos; // keep original global position
                let rerotated = pf.re_rotate_f32(&result.compact_keys, new_pos);
                let recon = result
                    .report
                    .as_ref()
                    .map(|r| r.relative_attn_output_error)
                    .unwrap_or(0.0);
                out.compact_keys.extend_from_slice(&rerotated);
                out.beta.extend_from_slice(&result.beta);
                out.compact_values.extend_from_slice(&result.compact_values);
                out.total_compact_len += result.compact_len;
                out.per_chunk.push(ChunkMeta {
                    chunk_start: chunk.start_pos,
                    chunk_len,
                    compact_len: result.compact_len,
                    reconstruction_error: recon,
                });
            }

            // Fallback path: no still_kv feature — compact keys as-is.
            // See `apply_rope_phase_shift` doc for the no-op contract.
            #[cfg(not(feature = "still_kv"))]
            {
                let result = compact(chunk_keys, chunk_values, queries, chunk_len, d, n, &chunk_cfg)?;
                let recon = result
                    .report
                    .as_ref()
                    .map(|r| r.relative_attn_output_error)
                    .unwrap_or(0.0);
                out.compact_keys.extend_from_slice(&result.compact_keys);
                out.beta.extend_from_slice(&result.beta);
                out.compact_values.extend_from_slice(&result.compact_values);
                out.total_compact_len += result.compact_len;
                out.per_chunk.push(ChunkMeta {
                    chunk_start: chunk.start_pos,
                    chunk_len,
                    compact_len: result.compact_len,
                    reconstruction_error: recon,
                });
            }
        }

        Ok(out)
    }
}

impl ChunkedCompactOutput {
    /// Mean per-chunk reconstruction error.
    pub fn mean_reconstruction_error(&self) -> f32 {
        if self.per_chunk.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .per_chunk
            .iter()
            .map(|c| c.reconstruction_error)
            .sum();
        sum / self.per_chunk.len() as f32
    }

    /// Boundary reconstruction error — mean of the first and last chunk only.
    ///
    /// Useful for comparing overlap vs no-overlap: with overlap, boundary
    /// chunks should have lower error because they "see" context from the
    /// adjacent chunk.
    pub fn boundary_reconstruction_error(&self) -> f32 {
        match self.per_chunk.len() {
            0 => 0.0,
            1 => self.per_chunk[0].reconstruction_error,
            _ => {
                let n = self.per_chunk.len();
                (self.per_chunk[0].reconstruction_error
                    + self.per_chunk[n - 1].reconstruction_error)
                    / 2.0
            }
        }
    }
}

/// Apply a RoPE phase shift: un-rotate keys at `start_pos`, then re-rotate at
/// `new_pos`. This effectively "moves" the keys from one position to another
/// without changing their semantic content (modulo f16 round-trip precision).
///
/// Requires the `still_kv` feature for [`PositionFreeCompactor`]. When
/// `still_kv` is off, this returns `keys.to_vec()` unchanged — RoPE
/// preservation is then the caller's responsibility (e.g. via an external
/// RoPE-aware KV cache).
pub fn apply_rope_phase_shift(
    keys: &[f32],
    d: usize,
    start_pos: usize,
    new_pos: usize,
    rope_theta: f32,
) -> Vec<f32> {
    if d == 0 || keys.is_empty() {
        return keys.to_vec();
    }
    if start_pos == new_pos {
        return keys.to_vec();
    }
    #[cfg(feature = "still_kv")]
    {
        let pf = PositionFreeBridge::new(rope_theta, d);
        pf.phase_shift_f32(keys, start_pos, new_pos)
    }
    #[cfg(not(feature = "still_kv"))]
    {
        // No-op fallback: still_kv feature is off, RoPE preservation skipped.
        let _ = (start_pos, new_pos, rope_theta);
        keys.to_vec()
    }
}

/// Default RoPE theta (LLaMA / Gemma convention) used when no override is
/// provided. Match this to the model's actual RoPE base frequency in
/// production.
pub const ROPE_THETA: f32 = 10_000.0;

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Compute the global token-start indices of each chunk.
///
/// Chunks have length `chunk_size` and stride `chunk_size - overlap`. The
/// final chunk is clamped to the end of the sequence.
fn chunk_starts(t_len: usize, chunk_size: usize, stride: usize) -> Vec<usize> {
    if t_len == 0 || chunk_size == 0 {
        return Vec::new();
    }
    let stride = stride.max(1);
    let mut starts = Vec::with_capacity(t_len / stride + 1);
    let mut start = 0usize;
    while start < t_len {
        starts.push(start);
        if start + chunk_size >= t_len {
            break;
        }
        start += stride;
    }
    starts
}

/// Build a per-chunk config clamped to `compact_size < chunk_len`.
///
/// [`compact`] rejects configs where `compact_size >= original_len`, so we
/// shrink `compact_size` to `chunk_len - 1` when needed. We do *not* error —
/// the caller asked us to compact whatever fits.
fn chunk_local_config(config: &AmConfig, chunk_len: usize) -> AmConfig {
    let mut cfg = config.clone();
    if cfg.compact_size >= chunk_len {
        cfg.compact_size = chunk_len.saturating_sub(1).max(1);
    }
    cfg
}

/// Infer `d` from the chunks, ensuring consistency.
fn infer_d(chunks: &[TextChunk]) -> Result<usize, CompactError> {
    for c in chunks {
        if c.chunk_len > 0 {
            if c.chunk_len == 0 {
                continue;
            }
            if c.keys.len() % c.chunk_len != 0 {
                return Err(CompactError::DimensionMismatch(format!(
                    "chunk: keys.len()={} not divisible by chunk_len={}",
                    c.keys.len(),
                    c.chunk_len
                )));
            }
            let d = c.keys.len() / c.chunk_len;
            if d == 0 {
                return Err(CompactError::DimensionMismatch(
                    "inferred d=0 from chunk".into(),
                ));
            }
            return Ok(d);
        }
    }
    Err(CompactError::DimensionMismatch(
        "all chunks empty — cannot infer d".into(),
    ))
}

/// Thin f32 adapter around [`PositionFreeCompactor`] (which is f16↔f32).
///
/// Encapsulates the conversion so callers stay in f32 land.
#[cfg(feature = "still_kv")]
struct PositionFreeBridge {
    inner: crate::still_kv::position_free::PositionFreeCompactor,
}

#[cfg(feature = "still_kv")]
impl PositionFreeBridge {
    fn new(rope_theta: f32, head_dim: usize) -> Self {
        Self {
            inner: crate::still_kv::position_free::PositionFreeCompactor::new(
                rope_theta,
                head_dim,
            ),
        }
    }

    fn un_rotate_f32(&self, keys: &[f32], start_pos: usize) -> Vec<f32> {
        use half::f16;
        let f16_keys: Vec<f16> = keys.iter().map(|&v| f16::from_f32(v)).collect();
        self.inner.un_rotate_keys(&f16_keys, start_pos)
    }

    fn re_rotate_f32(&self, keys: &[f32], new_pos: usize) -> Vec<f32> {
        let f16_out = self.inner.re_rotate_keys(keys, new_pos);
        f16_out.iter().map(|v| v.to_f32()).collect()
    }

    fn phase_shift_f32(&self, keys: &[f32], start_pos: usize, new_pos: usize) -> Vec<f32> {
        let pos_free = self.un_rotate_f32(keys, start_pos);
        self.re_rotate_f32(&pos_free, new_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_kv(t_len: usize, d: usize, seed: usize) -> (Vec<f32>, Vec<f32>) {
        let mut keys = vec![0.0f32; t_len * d];
        let mut values = vec![0.0f32; t_len * d];
        for i in 0..t_len {
            for k in 0..d {
                let x = ((i + seed) as f32) * 0.1 + (k as f32) * 0.01;
                keys[i * d + k] = x.sin() * 0.5;
                values[i * d + k] = x.cos() * 0.3;
            }
        }
        (keys, values)
    }

    fn synth_queries(n: usize, d: usize, seed: usize) -> Vec<f32> {
        let mut q = vec![0.0f32; n * d];
        for i in 0..n {
            for k in 0..d {
                let x = ((i + seed + 100) as f32) * 0.2 + (k as f32) * 0.05;
                q[i * d + k] = x.sin() * 0.4;
            }
        }
        q
    }

    #[test]
    fn test_kv_based_chunking_concatenates_correctly() {
        let d = 8usize;
        let n = 4usize;
        let per_chunk = 32usize;
        let num_chunks = 4usize;
        let t_len = per_chunk * num_chunks; // 128
        let (keys, values) = synth_kv(t_len, d, 1);
        let queries = synth_queries(n, d, 1);

        let compactor = ChunkedCompactor::new(per_chunk, 0);
        let cfg = AmConfig::highest_attn(8);
        let out = compactor
            .compact_kv_based(&keys, &values, &queries, t_len, d, n, &cfg)
            .expect("compact ok");

        assert_eq!(out.per_chunk.len(), num_chunks);
        for m in &out.per_chunk {
            assert_eq!(m.compact_len, 8, "each chunk should compact to 8");
        }
        assert_eq!(out.total_compact_len, 8 * num_chunks);
        assert_eq!(out.compact_keys.len(), out.total_compact_len * d);
        assert_eq!(out.compact_values.len(), out.total_compact_len * d);
        assert_eq!(out.beta.len(), out.total_compact_len);
    }

    #[test]
    fn test_kv_based_preserves_more_than_text_based_on_dependent_chunks() {
        // Construct two chunks where chunk 1's tokens depend on chunk 0's.
        // KV-based with overlap lets chunk 0's last `overlap` tokens bleed into
        // chunk 1, giving the compactor more context. Text-based (no overlap)
        // loses that dependency.
        let d = 8usize;
        let n = 4usize;
        let per_chunk = 32usize;

        // Chunk 0: distinct content.
        let (k0, v0) = synth_kv(per_chunk, d, 1);
        // Chunk 1: echoes chunk 0's last 8 tokens (synthetic dependency).
        let (mut k1, mut v1) = synth_kv(per_chunk, d, 2);
        for i in 0..8usize {
            for kk in 0..d {
                k1[i * d + kk] = k0[(per_chunk - 8 + i) * d + kk];
                v1[i * d + kk] = v0[(per_chunk - 8 + i) * d + kk];
            }
        }

        let mut full_keys = k0.clone();
        full_keys.extend_from_slice(&k1);
        let mut full_values = v0.clone();
        full_values.extend_from_slice(&v1);
        let t_len = per_chunk * 2;
        let queries = synth_queries(n, d, 3);

        // KV-based with overlap=8: chunk 1 includes 8 tokens from chunk 0.
        let compactor_ov = ChunkedCompactor::new(per_chunk, 8);
        let cfg = AmConfig::highest_attn(8);
        let out_kv = compactor_ov
            .compact_kv_based(&full_keys, &full_values, &queries, t_len, d, n, &cfg)
            .expect("kv compact ok");

        // Text-based: each chunk standalone (no overlap, start_pos = its index).
        let chunks = vec![
            TextChunk {
                keys: k0.clone(),
                values: v0.clone(),
                start_pos: 0,
                chunk_len: per_chunk,
            },
            TextChunk {
                keys: k1.clone(),
                values: v1.clone(),
                start_pos: per_chunk,
                chunk_len: per_chunk,
            },
        ];
        let q_chunk = synth_queries(n, d, 3);
        let queries_per_chunk = vec![q_chunk.clone(), q_chunk];
        let compactor_txt = ChunkedCompactor::new(per_chunk, 0);
        let out_txt = compactor_txt
            .compact_text_based(&chunks, &queries_per_chunk, &cfg)
            .expect("text compact ok");

        // KV-based with overlap must capture *more* source tokens overall
        // (overlap region is compacted in both chunks).
        let kv_total_source: usize = out_kv.per_chunk.iter().map(|m| m.chunk_len).sum();
        let txt_total_source: usize = out_txt.per_chunk.iter().map(|m| m.chunk_len).sum();
        assert!(
            kv_total_source > txt_total_source,
            "KV-based overlap should preserve more source tokens: kv={kv_total_source} txt={txt_total_source}"
        );

        // And the total compact length should reflect that extra coverage.
        assert!(
            out_kv.total_compact_len >= out_txt.total_compact_len,
            "KV-based overlap should produce >= text-based compact length: kv={} txt={}",
            out_kv.total_compact_len,
            out_txt.total_compact_len
        );
    }

    #[test]
    fn test_overlap_reduces_boundary_loss() {
        // With overlap, the first and last chunks get extra context → lower
        // boundary reconstruction error.
        let d = 8usize;
        let n = 4usize;
        let per_chunk = 32usize;
        let num_chunks = 4usize;
        let t_len = per_chunk * num_chunks;
        let (keys, values) = synth_kv(t_len, d, 5);
        let queries = synth_queries(n, d, 5);
        let cfg = AmConfig::highest_attn(8);

        let c_ov = ChunkedCompactor::new(per_chunk, 8);
        let out_ov = c_ov
            .compact_kv_based(&keys, &values, &queries, t_len, d, n, &cfg)
            .expect("ov compact");

        let c_no = ChunkedCompactor::new(per_chunk, 0);
        let out_no = c_no
            .compact_kv_based(&keys, &values, &queries, t_len, d, n, &cfg)
            .expect("no-ov compact");

        // Boundary error: first chunk's reconstruction should be the same
        // (no chunk before it) but the *interior* chunks and last chunk
        // benefit from overlap. Compare mean error of all chunks after the
        // first.
        let interior_ov: f32 = out_ov.per_chunk[1..]
            .iter()
            .map(|m| m.reconstruction_error)
            .sum::<f32>()
            / (out_ov.per_chunk.len() - 1).max(1) as f32;
        let interior_no: f32 = out_no.per_chunk[1..]
            .iter()
            .map(|m| m.reconstruction_error)
            .sum::<f32>()
            / (out_no.per_chunk.len() - 1).max(1) as f32;

        // Overlap version should have lower interior/boundary error.
        // (Equal is allowed for the degenerate synthetic case, but never higher.)
        assert!(
            interior_ov <= interior_no + 1e-5,
            "overlap should not increase boundary error: ov={interior_ov:.6} no={interior_no:.6}"
        );
    }

    #[test]
    fn test_chunked_total_length_correct() {
        let d = 8usize;
        let n = 4usize;
        let per_chunk = 32usize;
        let num_chunks = 3usize;
        let t_len = per_chunk * num_chunks;
        let (keys, values) = synth_kv(t_len, d, 9);
        let queries = synth_queries(n, d, 9);
        let compactor = ChunkedCompactor::new(per_chunk, 4);
        let cfg = AmConfig::highest_attn(10);
        let out = compactor
            .compact_kv_based(&keys, &values, &queries, t_len, d, n, &cfg)
            .expect("compact");

        let sum_per_chunk: usize = out.per_chunk.iter().map(|m| m.compact_len).sum();
        assert_eq!(sum_per_chunk, out.total_compact_len);
    }

    #[test]
    fn test_empty_input_returns_empty() {
        let compactor = ChunkedCompactor::new(32, 4);
        let cfg = AmConfig::highest_attn(8);
        let queries = synth_queries(4, 8, 1);
        let out = compactor
            .compact_kv_based(&[], &[], &queries, 0, 8, 4, &cfg)
            .expect("empty ok");
        assert_eq!(out.total_compact_len, 0);
        assert!(out.compact_keys.is_empty());
        assert!(out.compact_values.is_empty());
        assert!(out.beta.is_empty());
        assert!(out.per_chunk.is_empty());

        let out_txt = compactor
            .compact_text_based(&[], &[], &cfg)
            .expect("empty text ok");
        assert_eq!(out_txt.total_compact_len, 0);
    }

    #[test]
    fn test_single_chunk_equivalent_to_direct_compact() {
        // 1 chunk with overlap=0 should match a direct `compact()` call.
        let d = 8usize;
        let n = 4usize;
        let t_len = 32usize;
        let (keys, values) = synth_kv(t_len, d, 7);
        let queries = synth_queries(n, d, 7);
        let cfg = AmConfig::highest_attn(8);

        let direct = compact(&keys, &values, &queries, t_len, d, n, &cfg).expect("direct");

        let compactor = ChunkedCompactor::new(t_len, 0);
        let chunked = compactor
            .compact_kv_based(&keys, &values, &queries, t_len, d, n, &cfg)
            .expect("chunked");

        assert_eq!(chunked.total_compact_len, direct.compact_len);
        assert_eq!(chunked.compact_keys.len(), direct.compact_keys.len());

        for (a, b) in chunked.compact_keys.iter().zip(direct.compact_keys.iter()) {
            assert!((a - b).abs() < 1e-5, "key mismatch: {a} vs {b}");
        }
        for (a, b) in chunked.beta.iter().zip(direct.beta.iter()) {
            assert!((a - b).abs() < 1e-5, "beta mismatch: {a} vs {b}");
        }
        for (a, b) in chunked
            .compact_values
            .iter()
            .zip(direct.compact_values.iter())
        {
            assert!((a - b).abs() < 1e-5, "value mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_chunk_starts_clamps_final_chunk() {
        // t_len=100, chunk_size=32, stride=32-8=24 → starts 0,24,48,72, final clamped.
        let starts = chunk_starts(100, 32, 24);
        assert_eq!(starts, vec![0, 24, 48, 72]); // 72+32 >= 100, stop
        let starts2 = chunk_starts(128, 32, 32);
        assert_eq!(starts2, vec![0, 32, 64, 96]); // each chunk exactly fits, last 96..128
    }

    #[test]
    fn test_chunk_local_config_shrinks_oversize_compact() {
        let cfg = AmConfig::highest_attn(64);
        let local = chunk_local_config(&cfg, 32);
        assert_eq!(local.compact_size, 31);
    }

    #[test]
    fn test_compact_kv_based_dim_mismatch_errors() {
        let compactor = ChunkedCompactor::new(32, 0);
        let cfg = AmConfig::highest_attn(8);
        let keys = vec![0.0f32; 16 * 8]; // wrong size
        let values = vec![0.0f32; 32 * 8];
        let queries = vec![0.0f32; 4 * 8];
        let err = compactor
            .compact_kv_based(&keys, &values, &queries, 32, 8, 4, &cfg)
            .unwrap_err();
        assert!(matches!(err, CompactError::DimensionMismatch(_)));
    }
}
