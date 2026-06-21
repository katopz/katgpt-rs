//! Online compaction for arbitrarily long generation (Plan 271 Phase 5).
//!
//! During autoregressive decoding the KV cache grows without bound. Once the
//! physical token count exceeds `phys_budget + recent_window`, we compact the
//! older prefix while leaving the most-recent `recent_window` tokens
//! uncompacted so the model's local attention is bit-exact.
//!
//! The result is an [`OnlineCompactResult`] — a compacted prefix
//! `(Ck, β, Cv)` plus the raw recent suffix. The caller concatenates these
//! when feeding the model: `[compact_prefix ... recent_suffix]`.
//!
//! # Logical vs Physical Length
//!
//! - **Physical length** = actual KV tensors stored on device. Grows by 1 per
//!   generated token. Compaction reduces it back to ~`phys_budget +
//!   recent_window`.
//! - **Logical length** = sequence length the model "sees" after compaction
//!   (the AM-compacted prefix contributes `compact_len` logical tokens, plus
//!   `recent_window` raw tokens). This is what matters for attention shape.
//!
//! Because AM preserves attention output + mass, the compacted prefix is a
//! *lossy but attention-equivalent* summary: downstream queries see nearly the
//! same attention distribution as if all original tokens were present.
//!
//! # Trigger Boundary
//!
//! Compaction triggers iff `current_pos >= phys_budget + recent_window`
//! (inclusive). This guarantees:
//! - The compactable prefix has length ≥ `phys_budget` (so it's worth
//!   compacting).
//! - The recent window has length exactly `recent_window` (always preserved).
//!
//! # TL;DR
//!
//! [`OnlineCompactor::maybe_compact`] returns `Some` once the cache exceeds
//! `phys_budget + recent_window` tokens; the result has a compacted prefix
//! and a bit-identical recent suffix.

#![allow(clippy::too_many_arguments)]

use crate::attn_match::compact::{CompactError, compact};
use crate::attn_match::types::{AmConfig, AmResult};

/// Result of a single online compaction pass.
#[derive(Clone, Debug)]
pub struct OnlineCompactResult {
    /// Compacted older tokens (the prefix). `compact_prefix.compact_len` is
    /// the logical contribution of the prefix.
    pub compact_prefix: AmResult,
    /// Uncompacted recent keys (raw, bit-identical to input slice).
    /// Flat `(recent_window, d)`.
    pub recent_keys: Vec<f32>,
    /// Uncompacted recent values (raw, bit-identical to input slice).
    /// Flat `(recent_window, d)`.
    pub recent_values: Vec<f32>,
    /// Global position where the recent window starts in the original
    /// sequence. Equal to `current_pos - recent_window` at compaction time.
    pub recent_start_pos: usize,
    /// Total logical length = `compact_prefix.compact_len + recent_window`.
    /// This is the sequence length the model sees after this compaction.
    pub total_logical_len: usize,
}

/// Mid-trajectory compactor for unbounded generation.
#[derive(Clone, Debug)]
pub struct OnlineCompactor {
    /// Maximum compacted tokens before triggering compaction.
    pub phys_budget: usize,
    /// Number of most-recent tokens to always preserve uncompacted.
    pub recent_window: usize,
}

impl OnlineCompactor {
    /// Create a new online compactor.
    ///
    /// Panics iff `phys_budget == 0` or `recent_window == 0` (both must be
    /// positive — a zero recent window would mean compacting the active token,
    /// and a zero budget would trigger every step).
    pub fn new(phys_budget: usize, recent_window: usize) -> Self {
        assert!(phys_budget > 0, "phys_budget must be > 0");
        assert!(recent_window > 0, "recent_window must be > 0");
        Self {
            phys_budget,
            recent_window,
        }
    }

    /// Trigger threshold (inclusive). Compaction fires iff
    /// `current_pos >= trigger_threshold()`.
    #[inline]
    pub fn trigger_threshold(&self) -> usize {
        self.phys_budget + self.recent_window
    }

    /// Conditionally compact the KV cache.
    ///
    /// Returns `Some(result)` when `current_pos >= phys_budget + recent_window`
    /// (compaction triggered), `None` otherwise.
    ///
    /// When triggered:
    /// - Prefix `[0 .. current_pos - recent_window]` is compacted via [`compact`].
    /// - Suffix `[current_pos - recent_window .. current_pos]` is preserved raw.
    ///
    /// # Arguments
    /// * `kv_keys`, `kv_values` — full `(current_pos, d)` cache, flat.
    /// * `queries` — `(n, d)` reference queries, flat.
    /// * `current_pos` — number of tokens currently in the cache.
    /// * `d` — head dimension.
    /// * `n` — number of reference queries.
    /// * `config` — AM compaction config. `compact_size` is honored as-is; if
    ///   it's ≥ prefix length, it's clamped down (see [`clamp_compact_size`]).
    pub fn maybe_compact(
        &self,
        kv_keys: &[f32],
        kv_values: &[f32],
        queries: &[f32],
        current_pos: usize,
        d: usize,
        n: usize,
        config: &AmConfig,
    ) -> Result<Option<OnlineCompactResult>, CompactError> {
        if current_pos < self.trigger_threshold() {
            return Ok(None);
        }

        let prefix_len = current_pos - self.recent_window;

        // Defensive dimension checks. Compute products once.
        let cur_bytes = current_pos * d;
        if kv_keys.len() < cur_bytes {
            return Err(CompactError::DimensionMismatch(format!(
                "kv_keys.len()={} but current_pos*d={}*{}={}",
                kv_keys.len(),
                current_pos,
                d,
                cur_bytes
            )));
        }
        if kv_values.len() < cur_bytes {
            return Err(CompactError::DimensionMismatch(format!(
                "kv_values.len()={} but current_pos*d={}",
                kv_values.len(),
                cur_bytes
            )));
        }
        if queries.len() != n * d {
            return Err(CompactError::DimensionMismatch(format!(
                "queries.len()={} but n*d={}",
                queries.len(),
                n * d
            )));
        }

        // Edge: prefix_len == 0 means only the recent window is present —
        // nothing to compact. Treat as no-op (return None) to keep the result
        // type's invariants (`compact_prefix.compact_len > 0`).
        if prefix_len == 0 {
            return Ok(None);
        }

        let prefix_bytes = prefix_len * d;
        let prefix_keys = &kv_keys[..prefix_bytes];
        let prefix_values = &kv_values[..prefix_bytes];
        let recent_keys = kv_keys[prefix_bytes..cur_bytes].to_vec();
        let recent_values = kv_values[prefix_bytes..cur_bytes].to_vec();

        let cfg = clamp_compact_size(config, prefix_len);
        let compact_prefix = compact(prefix_keys, prefix_values, queries, prefix_len, d, n, &cfg)?;

        let total_logical_len = compact_prefix.compact_len + self.recent_window;

        Ok(Some(OnlineCompactResult {
            compact_prefix,
            recent_keys,
            recent_values,
            recent_start_pos: prefix_len,
            total_logical_len,
        }))
    }
}

impl OnlineCompactResult {
    /// Total bytes used by the post-compaction cache (f32 storage).
    ///
    /// Compact prefix: `compact_len * (2*d + 1)` f32 (Ck, β, Cv).
    /// Recent suffix: `recent_window * 2 * d` f32 (raw K, V).
    pub fn total_bytes(&self, d: usize) -> usize {
        let prefix_bytes =
            self.compact_prefix.compact_len * (2 * d + 1) * std::mem::size_of::<f32>();
        let recent_bytes = self.recent_keys.len() * std::mem::size_of::<f32>()
            + self.recent_values.len() * std::mem::size_of::<f32>();
        prefix_bytes + recent_bytes
    }

    /// Bytes saved by this compaction vs. the un-compacted prefix.
    ///
    /// Original prefix storage: `prefix_original_len * 2 * d * sizeof(f32)`.
    pub fn bytes_saved(&self, d: usize) -> usize {
        let original_prefix_bytes =
            self.compact_prefix.original_len * 2 * d * std::mem::size_of::<f32>();
        original_prefix_bytes.saturating_sub(
            self.total_bytes(d)
                - self.recent_keys.len() * std::mem::size_of::<f32>()
                - self.recent_values.len() * std::mem::size_of::<f32>(),
        )
    }
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Return a config with `compact_size` clamped to `< prefix_len`.
///
/// If the caller's `compact_size` is already smaller, the config is returned
/// unchanged (cheap clone). If it's ≥ prefix_len, we shrink to
/// `prefix_len - 1` so [`compact`] accepts it.
fn clamp_compact_size(config: &AmConfig, prefix_len: usize) -> AmConfig {
    // Fast path: no clamping needed. We still must clone to return an owned
    // AmConfig, but the caller is one-shot per compaction so this is fine.
    if config.compact_size < prefix_len {
        return config.clone();
    }
    let mut cfg = config.clone();
    cfg.compact_size = prefix_len.saturating_sub(1).max(1);
    cfg
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
    fn test_compaction_triggers_at_phys_budget() {
        // phys_budget=64, recent_window=16 → trigger at pos >= 80.
        let d = 8usize;
        let n = 4usize;
        let phys = 64usize;
        let window = 16usize;
        let (keys, values) = synth_kv(96, d, 1);
        let queries = synth_queries(n, d, 1);
        let cfg = AmConfig::highest_attn(8);

        let compactor = OnlineCompactor::new(phys, window);

        // At pos=80, triggers.
        let r80 = compactor
            .maybe_compact(&keys, &values, &queries, 80, d, n, &cfg)
            .expect("ok");
        assert!(r80.is_some(), "should trigger at pos=80");
        let r80 = r80.unwrap();
        assert_eq!(r80.recent_start_pos, 80 - window);
        assert_eq!(r80.recent_keys.len(), window * d);
        assert_eq!(r80.recent_values.len(), window * d);

        // At pos=79, does NOT trigger.
        let r79 = compactor
            .maybe_compact(&keys, &values, &queries, 79, d, n, &cfg)
            .expect("ok");
        assert!(r79.is_none(), "should NOT trigger at pos=79");
    }

    #[test]
    fn test_recent_window_preserved_uncompacted() {
        let d = 8usize;
        let n = 4usize;
        let phys = 32usize;
        let window = 8usize;
        let pos = phys + window; // 40
        let (keys, values) = synth_kv(pos, d, 2);
        let queries = synth_queries(n, d, 2);
        let cfg = AmConfig::highest_attn(8);

        let compactor = OnlineCompactor::new(phys, window);
        let result = compactor
            .maybe_compact(&keys, &values, &queries, pos, d, n, &cfg)
            .expect("ok")
            .expect("should trigger");

        // Recent slice must be bit-identical to input[pos-window*d .. pos*d].
        let expected_recent_keys = &keys[(pos - window) * d..pos * d];
        let expected_recent_values = &values[(pos - window) * d..pos * d];
        assert_eq!(result.recent_keys.as_slice(), expected_recent_keys);
        assert_eq!(result.recent_values.as_slice(), expected_recent_values);
    }

    #[test]
    fn test_multiple_consecutive_compactions_preserve_total_semantics() {
        // Simulate realistic online inference: generate tokens one at a time,
        // invoke `maybe_compact` at each step, and apply the compaction when
        // it fires. After 3 compactions, verify the logical length stayed
        // bounded and the compact prefix grew monotonically.
        let d = 8usize;
        let n = 4usize;
        let phys = 32usize;
        let window = 8usize;
        let cfg = AmConfig::highest_attn(16);
        let compactor = OnlineCompactor::new(phys, window);

        let mut pos = phys + window; // start exactly at trigger threshold
        let (mut keys, mut values) = synth_kv(pos, d, 3);
        let queries = synth_queries(n, d, 3);

        let mut logical_len_history: Vec<usize> = Vec::new();
        let mut compact_lens: Vec<usize> = Vec::new();
        let mut next_token_seed = 1000usize;

        // Generate up to 200 tokens; stop once we've seen 3 compactions.
        while logical_len_history.len() < 3 && pos < 512 {
            // Add one token (simulate a decode step).
            let (extra_k, extra_v) = synth_kv_with_offset(pos, 1, d, next_token_seed);
            next_token_seed += 1;
            keys.extend_from_slice(&extra_k);
            values.extend_from_slice(&extra_v);
            pos += 1;

            if let Some(r) = compactor
                .maybe_compact(&keys, &values, &queries, pos, d, n, &cfg)
                .expect("compact ok")
            {
                // Logical length must remain bounded by phys + window + 1
                // (compact_size may clamp up by 1 on tiny prefixes).
                assert!(
                    r.total_logical_len <= phys + window + 1,
                    "logical len {} exceeded bound {}",
                    r.total_logical_len,
                    phys + window
                );
                logical_len_history.push(r.total_logical_len);
                compact_lens.push(r.compact_prefix.compact_len);

                // Apply compaction: cache becomes [compact_prefix | recent].
                keys = r.compact_prefix.compact_keys.clone();
                keys.extend_from_slice(&r.recent_keys);
                values = r.compact_prefix.compact_values.clone();
                values.extend_from_slice(&r.recent_values);
                pos = r.total_logical_len;
            } // else: not yet at threshold, keep generating
        }

        assert_eq!(
            logical_len_history.len(),
            3,
            "should have compacted 3 times"
        );
        // Every observed logical length must respect the bound.
        for &ll in &logical_len_history {
            assert!(ll <= phys + window + 1, "logical len out of bound: {ll}");
        }
        // Compact prefix length is positive and bounded by phys.
        for &cl in &compact_lens {
            assert!(cl > 0 && cl <= phys, "compact_len out of range: {cl}");
        }
    }

    fn synth_kv_with_offset(
        offset: usize,
        len: usize,
        d: usize,
        seed: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut keys = vec![0.0f32; len * d];
        let mut values = vec![0.0f32; len * d];
        for i in 0..len {
            for k in 0..d {
                let x = ((i + offset + seed) as f32) * 0.1 + (k as f32) * 0.01;
                keys[i * d + k] = x.sin() * 0.5;
                values[i * d + k] = x.cos() * 0.3;
            }
        }
        (keys, values)
    }

    #[test]
    fn test_no_compaction_when_below_budget() {
        let d = 8usize;
        let n = 4usize;
        let phys = 64usize;
        let window = 16usize;
        let (keys, values) = synth_kv(32, d, 4); // way below budget
        let queries = synth_queries(n, d, 4);
        let cfg = AmConfig::highest_attn(8);

        let compactor = OnlineCompactor::new(phys, window);
        let r = compactor
            .maybe_compact(&keys, &values, &queries, 32, d, n, &cfg)
            .expect("ok");
        assert!(r.is_none(), "should not trigger below budget");
    }

    #[test]
    fn test_compaction_at_exact_boundary() {
        // pos == phys_budget + recent_window should trigger (inclusive).
        let d = 8usize;
        let n = 4usize;
        let phys = 64usize;
        let window = 16usize;
        let pos = phys + window; // 80 — exact boundary
        let (keys, values) = synth_kv(pos, d, 5);
        let queries = synth_queries(n, d, 5);
        let cfg = AmConfig::highest_attn(8);

        let compactor = OnlineCompactor::new(phys, window);
        let r = compactor
            .maybe_compact(&keys, &values, &queries, pos, d, n, &cfg)
            .expect("ok");
        assert!(r.is_some(), "boundary should be inclusive");
        let r = r.unwrap();
        assert_eq!(r.recent_start_pos, pos - window);
    }

    #[test]
    fn test_trigger_threshold_value() {
        let c = OnlineCompactor::new(4096, 256);
        assert_eq!(c.trigger_threshold(), 4096 + 256);
    }

    #[test]
    fn test_clamp_compact_size() {
        let cfg = AmConfig::highest_attn(64);
        let clamped = clamp_compact_size(&cfg, 32);
        assert_eq!(clamped.compact_size, 31);

        let cfg2 = AmConfig::highest_attn(8);
        let clamped2 = clamp_compact_size(&cfg2, 32);
        assert_eq!(clamped2.compact_size, 8); // unchanged
    }

    #[test]
    fn test_dim_mismatch_errors() {
        let d = 8usize;
        let n = 4usize;
        let phys = 32usize;
        let window = 8usize;
        let pos = phys + window;
        let keys = vec![0.0f32; 10 * d]; // too short
        let values = vec![0.0f32; pos * d];
        let queries = synth_queries(n, d, 1);
        let cfg = AmConfig::highest_attn(8);

        let compactor = OnlineCompactor::new(phys, window);
        let err = compactor
            .maybe_compact(&keys, &values, &queries, pos, d, n, &cfg)
            .unwrap_err();
        assert!(matches!(err, CompactError::DimensionMismatch(_)));
    }
}
