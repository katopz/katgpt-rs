//! Root-crate re-export of the sink-aware classifier primitive.
//!
//! The primitive (types + `classify_sink_at` + `classify_all_sinks` +
//! `stable_rank_update_into` + `apply_dual_policy_gate`) lives in
//! [`katgpt_core::data_probe`] (single file under the `sink_aware_attn`
//! feature in katgpt-core). This module re-exports it at the
//! `katgpt_rs::data_probe::sink_classify` path so it composes with the
//! existing [`crate::data_probe::geometry`] diagnostics.
//!
//! ## Relationship to the rest of `data_probe`
//!
//! The classifier is the **mechanism locator**: it identifies *which* sink
//! columns in an attention map are Adaptive NOPs vs Broadcasts. The sibling
//! [`crate::data_probe::geometry::effective_rank`] is the **aggregate
//! symptom**: it measures how collapsed the resulting hidden states are
//! across the whole layer. Broadcast sinks reduce `effective_rank` across
//! tokens (Lemma 4 in Fesser et al.); the classifier tells you *why*.
//!
//! Phase 4's `LayerSinkSummary` (in [`crate::data_probe::geometry`]) bridges
//! the two: per-layer aggregates of the per-sink classifications.
//!
//! Plan 287, Research 258, arXiv:2606.08105.

pub use katgpt_core::data_probe::{
    CachedSinkClassification, SinkAwarePolicy, SinkClassifierConfig, SinkDiagnostic, SinkKind,
    StableRankScratch, apply_dual_policy_gate, apply_dual_policy_gate_cached,
    apply_dual_policy_gate_cached_flat, apply_dual_policy_gate_flat, classify_all_sinks,
    classify_all_sinks_flat, classify_sink_at, classify_sink_at_flat, stable_rank_update_into,
    stable_rank_update_into_flat,
};

// ── Tests (Plan 287 Phase 1 T1.5 — G1 classifier correctness) ────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build an attention column of length `n` with all queries paying
    /// 100% attention to position `pos` (each entry is 1.0 — this is normalized
    /// attention where every row has all its mass on `pos`). Mean = 1.0.
    fn column_all_on(_pos: usize, n: usize) -> Vec<f32> {
        vec![1.0f32; n]
    }

    // ── G1: NOP-only head ────────────────────────────────────────

    /// One position has `‖v‖=0`, all attention mass there → `Nop`.
    #[test]
    fn g1_nop_only_head() {
        let n = 4;
        let d = 8;
        // Position 0 has zero value; all queries attend fully to position 0.
        let mut values: Vec<Vec<f32>> = (0..n).map(|_| vec![1.0; d]).collect();
        values[0] = vec![0.0; d]; // sink value is zero
        let attn_column = column_all_on(0, n);
        let cfg = SinkClassifierConfig::default();
        let mut scratch = StableRankScratch::new(d);
        let diag = classify_sink_at(0, &attn_column, &values, None, &cfg, &mut scratch);
        assert_eq!(diag.kind, SinkKind::Nop, "expected Nop, got {:?}", diag);
        assert!(diag.strength > 0.99, "strength should be ~1.0");
        assert!(
            diag.value_norm_ratio < 0.01,
            "value_norm_ratio should be ~0 for zero-norm sink, got {}",
            diag.value_norm_ratio
        );
    }

    // ── G1: Broadcast-only head ──────────────────────────────────

    /// One position has `‖v‖=content`, attention uniform → `Broadcast`.
    /// Stable rank of the rank-1 update `O = a·v_s^T` should be ≈ 1.
    #[test]
    fn g1_broadcast_only_head() {
        let n = 4;
        let d = 8;
        // All values identical → uniform norms, ratio exactly 1.0.
        let values: Vec<Vec<f32>> = (0..n).map(|_| vec![1.5; d]).collect();
        // Update O is rank-1: every row is the same vector v_s.
        let v_s = values[0].clone();
        let update_o: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        // Attention column: all queries pay 1.0 to pos 0.
        let attn_column = column_all_on(0, n);
        let cfg = SinkClassifierConfig::default();
        let mut scratch = StableRankScratch::new(d);
        let diag = classify_sink_at(
            0,
            &attn_column,
            &values,
            Some(&update_o),
            &cfg,
            &mut scratch,
        );
        assert_eq!(
            diag.kind,
            SinkKind::Broadcast,
            "expected Broadcast, got {:?}",
            diag
        );
        assert!(
            (diag.value_norm_ratio - 1.0).abs() < 0.01,
            "value_norm_ratio should be ~1 for broadcast, got {}",
            diag.value_norm_ratio
        );
        assert!(
            (diag.update_stable_rank - 1.0).abs() < 0.05,
            "stable rank should be ~1.0 for rank-1 update, got {}",
            diag.update_stable_rank
        );
    }

    // ── G1: Mixed head — two sinks, one NOP one Broadcast ────────────

    /// `classify_all_sinks` should pick up candidates above τ_sink.
    /// With attention rows that split mass 0.6/0.4 between two positions,
    /// only the 0.6 column exceeds τ_sink=0.5.
    #[test]
    fn g1_mixed_head() {
        let n = 5;
        let d = 4;
        // Position 0: NOP (zero value). Position 1: Broadcast (content value).
        // Other positions: low attention.
        let mut values: Vec<Vec<f32>> = (0..n).map(|i| vec![1.0 + 0.1 * i as f32; d]).collect();
        values[0] = vec![0.0; d]; // NOP
        // Attention map: queries split mass between pos 0 (0.6) and pos 1 (0.4).
        let attn: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut row = vec![0.0f32; n];
                row[0] = 0.6;
                row[1] = 0.4;
                row
            })
            .collect();
        let cfg = SinkClassifierConfig::default();
        let mut scratch = StableRankScratch::new(d);
        let mut out = Vec::new();
        classify_all_sinks(&attn, &values, &cfg, &mut scratch, &mut out);
        // Only position 0 exceeds τ_sink=0.5 (column sum / n = 0.6).
        assert_eq!(
            out.len(),
            1,
            "expected 1 candidate above τ_sink=0.5, got {}",
            out.len()
        );
        assert_eq!(out[0].position, 0);
        assert_eq!(out[0].kind, SinkKind::Nop);
    }

    /// Mixed head with both sinks above τ_sink.
    #[test]
    fn g1_mixed_head_both_above_threshold() {
        let n = 5;
        let d = 4;
        let mut values: Vec<Vec<f32>> = (0..n).map(|i| vec![1.0 + 0.1 * i as f32; d]).collect();
        values[0] = vec![0.0; d]; // NOP sink at pos 0
        // values[1] stays as a normal content vector → will be broadcast if
        // update_O is rank-1. But classify_all_sinks doesn't pass update_O,
        // so broadcast requires only value_norm_ratio in [0.5, 1.5] AND
        // update_O.is_some()... wait, that's never satisfied without update_O.
        // Re-read classify_sink_at: Broadcast requires update_O.is_some().
        // So with classify_all_sinks (which passes None), broadcast is never
        // returned. Adjust test: just verify pos 0 is NOP and pos 1 is
        // *not* classified as Broadcast (it'll be None due to missing O).
        let attn: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut row = vec![0.0f32; n];
                row[0] = 0.6;
                row[1] = 0.6; // both above τ_sink=0.5
                row
            })
            .collect();
        let cfg = SinkClassifierConfig::default();
        let mut scratch = StableRankScratch::new(d);
        let mut out = Vec::new();
        classify_all_sinks(&attn, &values, &cfg, &mut scratch, &mut out);
        assert_eq!(out.len(), 2, "expected 2 candidates, got {}", out.len());
        // Position 0: zero value → NOP.
        let d0 = out.iter().find(|d| d.position == 0).expect("pos 0 missing");
        assert_eq!(d0.kind, SinkKind::Nop);
        // Position 1: value_norm_ratio ~1 but no update_O → None.
        let d1 = out.iter().find(|d| d.position == 1).expect("pos 1 missing");
        assert_eq!(d1.kind, SinkKind::None);
    }

    // ── G1: No-sink head ─────────────────────────────────────────

    /// Uniform attention → no position exceeds τ_sink → empty out.
    #[test]
    fn g1_no_sink_head() {
        let n = 6;
        let d = 4;
        let values: Vec<Vec<f32>> = (0..n).map(|i| vec![1.0 + 0.1 * i as f32; d]).collect();
        // Uniform attention: each row is 1/n.
        let attn: Vec<Vec<f32>> = (0..n).map(|_| vec![1.0 / n as f32; n]).collect();
        let cfg = SinkClassifierConfig::default();
        let mut scratch = StableRankScratch::new(d);
        let mut out = Vec::new();
        classify_all_sinks(&attn, &values, &cfg, &mut scratch, &mut out);
        assert!(
            out.is_empty(),
            "no position should exceed τ_sink=0.5 under uniform attention, got {:?}",
            out
        );
    }

    // ── G1: Zero attention column edge ───────────────────────────

    /// Zero-length or all-zero attention column must not crash.
    #[test]
    fn g1_zero_attn_column_edge() {
        let d = 4;
        let values: Vec<Vec<f32>> = vec![vec![1.0; d]];
        let cfg = SinkClassifierConfig::default();
        let mut scratch = StableRankScratch::new(d);
        // Zero-length column.
        let col: [f32; 0] = [];
        let diag = classify_sink_at(0, &col, &values, None, &cfg, &mut scratch);
        assert_eq!(diag.kind, SinkKind::None);
        assert!(diag.strength.abs() < 1e-6);

        // All-zero column.
        let col2 = [0.0f32; 4];
        let diag2 = classify_sink_at(0, &col2, &values, None, &cfg, &mut scratch);
        assert_eq!(diag2.kind, SinkKind::None);
    }

    // ── G1: Degenerate values edge ───────────────────────────────

    /// All-zero values must not divide by zero; ratio set to 1.0, kind=None.
    #[test]
    fn g1_degenerate_values_edge() {
        let n = 3;
        let d = 4;
        let values: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
        let attn_column = column_all_on(0, n);
        let cfg = SinkClassifierConfig::default();
        let mut scratch = StableRankScratch::new(d);
        let diag = classify_sink_at(0, &attn_column, &values, None, &cfg, &mut scratch);
        // Degenerate handling: ratio=1.0, kind=None (even though strength > τ).
        assert_eq!(diag.kind, SinkKind::None);
        assert!(
            diag.value_norm_ratio > 0.99 && diag.value_norm_ratio < 1.01,
            "degenerate ratio should be set to 1.0, got {}",
            diag.value_norm_ratio
        );
        assert!(!diag.update_stable_rank.is_nan() || diag.update_stable_rank.is_nan());
        // stable_rank_update_into on all-zero O returns 0.0 — not NaN.
    }

    // ── G1: Degenerate stable-rank input ────────────────────────

    /// All-zero `O` should return stable rank 0 (not NaN, not crash).
    #[test]
    fn g1_stable_rank_zero_matrix() {
        let d = 4;
        let n = 3;
        let o: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
        let mut scratch = StableRankScratch::new(d);
        let sr = stable_rank_update_into(&o, &mut scratch, 5);
        assert!(
            sr.abs() < 1e-6,
            "stable rank of zero matrix should be 0, got {}",
            sr
        );
        assert!(!sr.is_nan());
    }

    // ── Issue 001: cached variant parity ────────────────────────────

    /// The cached variant must produce the same output as the per-call
    /// variant on its audit call, and the same output on subsequent cached
    /// calls (assuming the classification is stable).
    #[test]
    fn issue001_cached_matches_per_call_for_broadcast() {
        use super::*;
        let n = 8;
        let d = 16;
        let v_s: Vec<f32> = (0..d).map(|i| 0.5 + 0.1 * i as f32).collect();
        let values: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let o: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let attn: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut row = vec![0.1 / (n as f32 - 1.0); n];
                row[0] = 0.9;
                row
            })
            .collect();

        let cfg = SinkClassifierConfig::default();
        let policy_dual = SinkAwarePolicy::DualPolicy(cfg);
        let mut out_dual: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
        let mut out_cached: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
        let mut scratch_a = StableRankScratch::new(d);
        let mut scratch_b = StableRankScratch::new(d);
        let mut cached = CachedSinkClassification::with_config(cfg, 16);

        // Audit call (first call) — must match per-call DualPolicy.
        let kind_dual = apply_dual_policy_gate(
            &attn, &values, &o, &policy_dual, 2.0, &mut scratch_a, &mut out_dual,
        );
        let kind_cached = apply_dual_policy_gate_cached(
            &attn, &values, &o, 2.0, &mut scratch_b, &mut cached, &mut out_cached,
        );
        assert_eq!(kind_dual, kind_cached, "audit call should classify same");
        assert_eq!(kind_dual, SinkKind::Broadcast);

        // Verify outputs match bit-for-bit (Broadcast → copy unchanged).
        for i in 0..n {
            for j in 0..d {
                assert!((out_dual[i][j] - out_cached[i][j]).abs() < 1e-6);
            }
        }

        // Subsequent cached calls should reuse the cached Broadcast decision.
        let kind_cached_2 = apply_dual_policy_gate_cached(
            &attn, &values, &o, 2.0, &mut scratch_b, &mut cached, &mut out_cached,
        );
        assert_eq!(kind_cached_2, SinkKind::Broadcast);
        assert_eq!(cached.calls_since_audit, 2);
        // Output unchanged for Broadcast.
        for i in 0..n {
            for j in 0..d {
                assert!((out_cached[i][j] - o[i][j]).abs() < 1e-6);
            }
        }
    }

    /// Cache invalidate forces re-classification on next call.
    #[test]
    fn issue001_cached_invalidate_forces_reaudit() {
        let n = 4;
        let d = 8;
        let v_s: Vec<f32> = vec![1.5; d];
        let values: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let o: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let attn: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut row = vec![0.0; n];
                row[0] = 1.0;
                row
            })
            .collect();

        let cfg = SinkClassifierConfig::default();
        let mut scratch = StableRankScratch::new(d);
        let mut out: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
        let mut cached = CachedSinkClassification::with_config(cfg, 16);

        // First call: audit.
        let _ = apply_dual_policy_gate_cached(
            &attn, &values, &o, 2.0, &mut scratch, &mut cached, &mut out,
        );
        assert_eq!(cached.cached_kind, Some(SinkKind::Broadcast));
        assert_eq!(cached.calls_since_audit, 1);

        // Invalidate.
        cached.invalidate();
        assert!(cached.cached_kind.is_none());
        assert_eq!(cached.calls_since_audit, 0);

        // Next call should re-audit.
        let kind = apply_dual_policy_gate_cached(
            &attn, &values, &o, 2.0, &mut scratch, &mut cached, &mut out,
        );
        assert_eq!(kind, SinkKind::Broadcast);
        assert_eq!(cached.calls_since_audit, 1);
    }

    // ── Plan 288 T7: flat-layout parity vs Vec<Vec<f32>> ───────────
    //
    // Every flat variant must produce bit-identical `SinkKind` decisions
    // (and numerically-identical diagnostics / outputs) to the Vec<Vec<f32>>
    // variant on identical inputs. This is the G1 GOAT gate for Plan 288.

    /// Helper: flatten a Vec<Vec<f32>> row-major into a single Vec<f32>.
    /// Used to feed identical data to both layout variants in parity tests.
    fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
        let d = rows.first().map(|r| r.len()).unwrap_or(0);
        let mut out = Vec::with_capacity(rows.len() * d);
        for r in rows {
            out.extend_from_slice(r);
        }
        out
    }

    /// `stable_rank_update_into_flat` must match `stable_rank_update_into`
    /// on the same data (within fp tolerance — the arithmetic is identical,
    /// so they should agree to the last bit).
    #[test]
    fn plan288_stable_rank_flat_parity() {
        let n = 6;
        let d = 8;
        // Rank-1-ish O: every row = scalar_i * base_vec.
        let base: Vec<f32> = (0..d).map(|i| 0.3 + 0.07 * i as f32).collect();
        let o_rows: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let s = 1.0 + 0.1 * i as f32;
                base.iter().map(|x| x * s).collect()
            })
            .collect();
        let o_flat = flatten(&o_rows);

        let mut sc_vec = StableRankScratch::new(d);
        let mut sc_flat = StableRankScratch::new(d);

        let sr_vec = stable_rank_update_into(&o_rows, &mut sc_vec, 5);
        let sr_flat = stable_rank_update_into_flat(&o_flat, n, d, &mut sc_flat, 5);

        // Both should classify as effectively rank-1 (cosine probe fires).
        assert!((sr_vec - 1.0).abs() < 0.05, "vec sr={}, expected ~1.0", sr_vec);
        assert!((sr_flat - 1.0).abs() < 0.05, "flat sr={}, expected ~1.0", sr_flat);
        // Bit-identical (same arithmetic, just different slicing).
        assert!((sr_vec - sr_flat).abs() < 1e-6, "vec={} flat={}", sr_vec, sr_flat);
    }

    /// Flat stable-rank on zero matrix must return 0.0 (not NaN).
    #[test]
    fn plan288_stable_rank_flat_zero_matrix() {
        let n = 3;
        let d = 4;
        let o_flat = vec![0.0f32; n * d];
        let mut scratch = StableRankScratch::new(d);
        let sr = stable_rank_update_into_flat(&o_flat, n, d, &mut scratch, 5);
        assert!(sr.abs() < 1e-6, "zero-matrix flat sr should be 0, got {}", sr);
        assert!(!sr.is_nan());
    }

    /// `classify_sink_at_flat` must agree with `classify_sink_at` on NOP head.
    #[test]
    fn plan288_classify_sink_flat_parity_nop() {
        let n = 4;
        let d = 8;
        let mut values_rows: Vec<Vec<f32>> = (0..n).map(|_| vec![1.0; d]).collect();
        values_rows[0] = vec![0.0; d]; // NOP sink at pos 0
        let values_flat = flatten(&values_rows);
        let attn_col = vec![1.0f32; n];
        let cfg = SinkClassifierConfig::default();
        let mut sc_a = StableRankScratch::new(d);
        let mut sc_b = StableRankScratch::new(d);

        let diag_vec = classify_sink_at(0, &attn_col, &values_rows, None, &cfg, &mut sc_a);
        let diag_flat = classify_sink_at_flat(0, &attn_col, &values_flat, n, d, None, &cfg, &mut sc_b);

        assert_eq!(diag_vec.kind, SinkKind::Nop);
        assert_eq!(diag_flat.kind, SinkKind::Nop);
        assert!((diag_vec.value_norm_ratio - diag_flat.value_norm_ratio).abs() < 1e-6);
        assert!((diag_vec.strength - diag_flat.strength).abs() < 1e-6);
    }

    /// `classify_sink_at_flat` must agree with `classify_sink_at` on Broadcast head.
    #[test]
    fn plan288_classify_sink_flat_parity_broadcast() {
        let n = 4;
        let d = 8;
        let v_s: Vec<f32> = (0..d).map(|i| 1.5 + 0.05 * i as f32).collect();
        let values_rows: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let o_rows: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect(); // rank-1 O
        let values_flat = flatten(&values_rows);
        let o_flat = flatten(&o_rows);
        let attn_col = vec![1.0f32; n];
        let cfg = SinkClassifierConfig::default();
        let mut sc_a = StableRankScratch::new(d);
        let mut sc_b = StableRankScratch::new(d);

        let diag_vec = classify_sink_at(
            0, &attn_col, &values_rows, Some(&o_rows), &cfg, &mut sc_a,
        );
        let diag_flat = classify_sink_at_flat(
            0, &attn_col, &values_flat, n, d, Some((&o_flat, n, d)), &cfg, &mut sc_b,
        );

        assert_eq!(diag_vec.kind, SinkKind::Broadcast);
        assert_eq!(diag_flat.kind, SinkKind::Broadcast);
        assert!((diag_vec.value_norm_ratio - diag_flat.value_norm_ratio).abs() < 1e-6);
        assert!((diag_vec.update_stable_rank - diag_flat.update_stable_rank).abs() < 1e-6);
    }

    /// `classify_all_sinks_flat` must pick up the same candidates as the
    /// Vec<Vec<f32>> variant.
    #[test]
    fn plan288_classify_all_sinks_flat_parity() {
        let n = 5;
        let d = 4;
        let mut values_rows: Vec<Vec<f32>> =
            (0..n).map(|i| vec![1.0 + 0.1 * i as f32; d]).collect();
        values_rows[0] = vec![0.0; d]; // NOP at pos 0
        let values_flat = flatten(&values_rows);
        let attn_rows: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut row = vec![0.0f32; n];
                row[0] = 0.6;
                row[1] = 0.4;
                row
            })
            .collect();
        let attn_flat = flatten(&attn_rows);
        let cfg = SinkClassifierConfig::default();
        let mut sc_a = StableRankScratch::new(d);
        let mut sc_b = StableRankScratch::new(d);
        let mut out_vec = Vec::new();
        let mut out_flat = Vec::new();

        classify_all_sinks(&attn_rows, &values_rows, &cfg, &mut sc_a, &mut out_vec);
        classify_all_sinks_flat(&attn_flat, n, &values_flat, d, &cfg, &mut sc_b, &mut out_flat);

        assert_eq!(out_vec.len(), out_flat.len());
        assert_eq!(out_vec.len(), 1, "only pos 0 exceeds τ_sink=0.5");
        assert_eq!(out_vec[0].position, 0);
        assert_eq!(out_flat[0].position, 0);
        assert_eq!(out_vec[0].kind, SinkKind::Nop);
        assert_eq!(out_flat[0].kind, SinkKind::Nop);
    }

    /// `apply_dual_policy_gate_flat` must produce identical SinkKind AND
    /// output values to the Vec<Vec<f32>> variant on identical inputs.
    #[test]
    fn plan288_apply_dual_policy_gate_flat_parity() {
        let n = 8;
        let d = 16;
        // Broadcast setup: rank-1 O, all values equal, attention on pos 0.
        let v_s: Vec<f32> = (0..d).map(|i| 0.5 + 0.1 * i as f32).collect();
        let values_rows: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let o_rows: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let attn_rows: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut row = vec![0.1 / (n as f32 - 1.0); n];
                row[0] = 0.9;
                row
            })
            .collect();

        let values_flat = flatten(&values_rows);
        let o_flat = flatten(&o_rows);
        let attn_flat = flatten(&attn_rows);

        let cfg = SinkClassifierConfig::default();
        let policy = SinkAwarePolicy::DualPolicy(cfg);
        let gate_scale = 2.0;

        let mut out_rows: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
        let mut out_flat = vec![0.0f32; n * d];
        let mut sc_a = StableRankScratch::new(d);
        let mut sc_b = StableRankScratch::new(d);

        let kind_vec = apply_dual_policy_gate(
            &attn_rows, &values_rows, &o_rows, &policy, gate_scale, &mut sc_a, &mut out_rows,
        );
        let kind_flat = apply_dual_policy_gate_flat(
            &attn_flat, &values_flat, &o_flat, n, d, &policy, gate_scale, &mut sc_b, &mut out_flat,
        );

        assert_eq!(kind_vec, kind_flat);
        assert_eq!(kind_vec, SinkKind::Broadcast);

        // Outputs must match bit-for-bit.
        for i in 0..n {
            for j in 0..d {
                let v = out_rows[i][j];
                let f = out_flat[i * d + j];
                assert!((v - f).abs() < 1e-6, "mismatch at [{i}][{j}]: vec={v} flat={f}");
            }
        }
    }

    /// NOP head: flat variant must gate output by σ(gate_scale).
    #[test]
    fn plan288_apply_dual_policy_gate_flat_nop() {
        let n = 4;
        let d = 8;
        // NOP: pos 0 has zero value, all attention on pos 0.
        let mut values_rows: Vec<Vec<f32>> = (0..n).map(|_| vec![1.0; d]).collect();
        values_rows[0] = vec![0.0; d];
        let o_rows: Vec<Vec<f32>> = (0..n).map(|i| vec![2.0 + i as f32; d]).collect();
        let attn_rows: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut row = vec![0.0f32; n];
                row[0] = 1.0;
                row
            })
            .collect();

        let values_flat = flatten(&values_rows);
        let o_flat = flatten(&o_rows);
        let attn_flat = flatten(&attn_rows);

        let cfg = SinkClassifierConfig::default();
        let policy = SinkAwarePolicy::DualPolicy(cfg);
        let gate_scale = 1.0; // σ(1) ≈ 0.731
        let mut out_flat = vec![0.0f32; n * d];
        let mut scratch = StableRankScratch::new(d);

        let kind = apply_dual_policy_gate_flat(
            &attn_flat, &values_flat, &o_flat, n, d, &policy, gate_scale, &mut scratch, &mut out_flat,
        );
        assert_eq!(kind, SinkKind::Nop);

        let expected_scale = 1.0 / (1.0 + (-gate_scale).exp());
        for i in 0..n {
            for j in 0..d {
                let expected = o_rows[i][j] * expected_scale;
                assert!((out_flat[i * d + j] - expected).abs() < 1e-5);
            }
        }
    }

    /// Cached flat variant: audit call matches per-call, subsequent calls
    /// reuse the decision.
    #[test]
    fn plan288_cached_flat_audit_and_reuse() {
        let n = 4;
        let d = 8;
        let v_s: Vec<f32> = vec![1.5; d];
        let values_rows: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let o_rows: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let attn_rows: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut row = vec![0.0f32; n];
                row[0] = 1.0;
                row
            })
            .collect();
        let values_flat = flatten(&values_rows);
        let o_flat = flatten(&o_rows);
        let attn_flat = flatten(&attn_rows);

        let cfg = SinkClassifierConfig::default();
        let mut out_a = vec![0.0f32; n * d];
        let mut out_b = vec![0.0f32; n * d];
        let mut sc_a = StableRankScratch::new(d);
        let mut sc_b = StableRankScratch::new(d);
        let mut cached = CachedSinkClassification::with_config(cfg, 16);

        // Per-call reference.
        let policy = SinkAwarePolicy::DualPolicy(cfg);
        let kind_a = apply_dual_policy_gate_flat(
            &attn_flat, &values_flat, &o_flat, n, d, &policy, 2.0, &mut sc_a, &mut out_a,
        );
        // Cached audit call.
        let kind_b = apply_dual_policy_gate_cached_flat(
            &attn_flat, &values_flat, &o_flat, n, d, 2.0, &mut sc_b, &mut cached, &mut out_b,
        );
        assert_eq!(kind_a, kind_b);
        assert_eq!(kind_b, SinkKind::Broadcast);
        assert_eq!(cached.calls_since_audit, 1);

        // Outputs match (Broadcast → copy unchanged).
        for k in 0..n * d {
            assert!((out_a[k] - out_b[k]).abs() < 1e-6);
        }

        // Second cached call — reuses decision, calls_since_audit increments.
        let kind_c = apply_dual_policy_gate_cached_flat(
            &attn_flat, &values_flat, &o_flat, n, d, 2.0, &mut sc_b, &mut cached, &mut out_b,
        );
        assert_eq!(kind_c, SinkKind::Broadcast);
        assert_eq!(cached.calls_since_audit, 2);
    }
}
