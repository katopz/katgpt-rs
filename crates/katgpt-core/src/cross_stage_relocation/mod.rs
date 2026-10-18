//! Cross-Stage Residual Relocation Operator + Permeation-Map Diagnostic
//! (Plan 431, Research 417, arXiv:2607.08393 — Dai, Rao, Wang et al.,
//! "Towards Mechanistically Understanding Why Memorized Knowledge Fails to
//! Generalize in LLM Finetuning", HKUST-GZ / HKUST, NeurIPS 2026 submission).
//!
//! Two modelless primitives distilled from the Knowing-Using Gap paper:
//!
//! - **[`PermeationMap`] + [`permeation_scan_into`]** — the safe (diagnostic)
//!   half. A 2D `(source_stage, target_stage)` intervention heatmap that reuses
//!   [`crate::causal_head_importance::direct_effect_importance`] (Plan 358) as
//!   the cell score, plus [`PermeationMap::classify_two_cluster`] for the
//!   paper's two-cluster pattern (§5.4 / Figure 5). No forward-pass machinery
//!   of its own — the caller supplies the patched-forward closure (same
//!   contract as Plan 358's `causal_head_importance::patching`).
//!
//! - **[`RelocateOp`] + [`RelocatePair`] + [`RelocatingForward`]** — the risky
//!   (applied operator) half in [`relocate`]. Snapshots an anchor's residual
//!   state at one stage and overwrites at another during a forward pass. Ships
//!   with the paper's fixed two-pair default `(0.82L→0.45L) + (0.10L→0.45L)`
//!   as [`RelocatePair::LateEarly`] (§5.5, 58–75% oracle headroom recovery on
//!   the paper's LLM benchmark across 6 models × 2 domains).
//!
//! # Modelless discipline
//!
//! Zero training, zero backprop. The diagnostic is a forward-pass-only scan
//! over a caller-supplied patched-forward closure. The operator is two
//! `memcpy`s of an activation buffer plus a forward-pass orchestration. The
//! saturation-epoch / gradient-locality / alignment-aware-training findings
//! stay in Research 417 §1 only — they are riir-train territory.
//!
//! # Caller responsibility
//!
//! As with Plan 358's `direct_effect_importance`, the patched forward pass
//! (snapshot/overwrite hooks on the host's residual stream) is the caller's
//! responsibility — it requires a full transformer-style forward and lives in
//! riir-engine / riir-games. This module is the *operator + scorer*; the
//! forward pass is supplied by the caller via [`RelocatingForward`] (operator)
//! or a closure (diagnostic). This keeps katgpt-core leaf-clean (no
//! transformer dep) and matches the FaithfulnessProbe / causal_head_importance
//! pattern.
//!
//! # Promotion gate
//!
//! Opt-in feature flag `cross_stage_relocation`. Promotion to default requires
//! G1–G6 PASS **and** the §3.6 defend-wrong PoC (Phase 3 in
//! `riir-ai/crates/riir-poc/`) confirming the operator actually recovers
//! capability on a toy domain — not just architectural coverage. The paper
//! proves 58–75% oracle-headroom recovery on LLMs with knowledge injection;
//! our substrate (latent functors, HLA, neuron shards) does not have the same
//! "early MLP / late MLP" structure, so the transfer must be PoC-verified.
//!
//! Cross-Stage Residual Relocation Operator + Permeation-Map Diagnostic
//! (Plan 431, Research 417, arXiv:2607.08393 — Knowing-Using Gap). Two
//! modelless primitives: (1) `permeation_scan_into` — a 2D
//! `(src_stage, dst_stage)` intervention heatmap reusing Plan 358's
//! `direct_effect_importance` as the cell score, plus two-cluster
//! classification; (2) `RelocateOp` — an applied operator that snapshots an
//! anchor's state at one stage and overwrites at another, with the paper's
//! `(0.82L→0.45L) + (0.10L→0.45L)` fixed default. Both behind
//! `cross_stage_relocation` feature flag, opt-in. Implies `causal_head_importance`
//! (the cell-score function). Phase 3 defend-wrong PoC in `riir-poc/` is
//! MANDATORY before any promotion — the 58–75% recovery is a quality claim on
//! the paper's LLM substrate, not ours.

pub mod relocate;

pub use relocate::{RelocateOp, RelocatePair, RelocatingForward};

use crate::causal_head_importance::direct_effect_importance;

/// `L_src × L_dst` matrix of [`direct_effect_importance`] cell scores.
///
/// Cell `(i, j) > 0` means: snapshotting the anchor's state at source stage
/// `i` and overwriting at destination stage `j` increases the readout by
/// `cell(i, j)` (in normalized `[0, 1]` units — it is an effect-importance
/// score, not a raw logit difference).
///
/// # Layout
///
/// Row-major flat layout: `cell(i, j) = cells[i * n_dst + j]`.
///
/// # Construction
///
/// Caller pre-allocates via [`PermeationMap::zeros`] and fills via
/// [`permeation_scan_into`] (zero-alloc — reuses the buffer).
#[derive(Clone, Debug)]
pub struct PermeationMap {
    /// Row-major `[n_src * n_dst]`. `cell(i, j) = cells[i * n_dst + j]`.
    pub cells: Vec<f32>,
    /// Number of source stages (rows).
    pub n_src: usize,
    /// Number of destination stages (cols).
    pub n_dst: usize,
}

impl PermeationMap {
    /// Pre-allocate an `n_src × n_dst` map filled with zeros.
    ///
    /// The intended call pattern is: construct once, reuse across scans via
    /// [`permeation_scan_into`] (which overwrites in place — no growth).
    #[inline]
    pub fn zeros(n_src: usize, n_dst: usize) -> Self {
        Self {
            cells: vec![0.0; n_src.checked_mul(n_dst).expect("n_src * n_dst overflow")],
            n_src,
            n_dst,
        }
    }

    /// Read the cell `(src_stage, dst_stage)`.
    ///
    /// Returns `0.0` for out-of-range indices (the scan produces no signal
    /// there; this matches the paper's "no patch → no effect" baseline).
    #[inline]
    pub fn cell(&self, src_stage: usize, dst_stage: usize) -> f32 {
        if src_stage < self.n_src && dst_stage < self.n_dst {
            self.cells[src_stage * self.n_dst + dst_stage]
        } else {
            0.0
        }
    }

    /// Mutably access the cell `(src_stage, dst_stage)`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of range. Use [`PermeationMap::cell`] for
    /// a clamping read accessor.
    #[inline]
    pub fn cell_mut(&mut self, src_stage: usize, dst_stage: usize) -> &mut f32 {
        assert!(
            src_stage < self.n_src,
            "src_stage {src_stage} out of range (n_src={})",
            self.n_src
        );
        assert!(
            dst_stage < self.n_dst,
            "dst_stage {dst_stage} out of range (n_dst={})",
            self.n_dst
        );
        &mut self.cells[src_stage * self.n_dst + dst_stage]
    }

    /// The paper's two-cluster classification (§5.4 / Figure 5).
    ///
    /// Splits each axis into early / mid / late thirds and reports which
    /// quadrant(s) contain the dominant effective-patch cluster:
    ///
    /// - [`ClusterClass::EarlyToMid`] — only the `early_src × mid_dst`
    ///   quadrant is dominant (paper's surprising cluster: source ≈0.1L →
    ///   target ≈0.45L).
    /// - [`ClusterClass::LateToMid`] — only the `late_src × mid_dst` quadrant
    ///   is dominant (paper's intuitive cluster: source ≈0.8L → target
    ///   ≈0.45L).
    /// - [`ClusterClass::Both`] — both early→mid and late→mid quadrants are
    ///   dominant (the paper's combined two-pair heuristic applies).
    /// - [`ClusterClass::None`] — no clear two-cluster pattern (the map is
    ///   flat, dominated by an off-cluster region, or all-zero).
    ///
    /// "Dominant" means: that quadrant's mean cell score is `≥ τ` AND `≥` the
    /// max mean of all other quadrants. `τ` defaults to `0.1` (a soft floor
    /// that suppresses fp noise — the cells are normalized `[0, 1]` effect
    /// scores, so 0.1 corresponds to a 10% effect). For a map with `< 3`
    /// stages on either axis the thirds degenerate; the classifier still
    /// partitions by the available indices.
    pub fn classify_two_cluster(&self) -> ClusterClass {
        self.classify_two_cluster_with_threshold(0.1)
    }

    /// Same as [`PermeationMap::classify_two_cluster`] but with an explicit
    /// dominance threshold `τ`.
    pub fn classify_two_cluster_with_threshold(&self, threshold: f32) -> ClusterClass {
        // Degenerate case: an empty or all-zero map has no cluster.
        if self.n_src == 0 || self.n_dst == 0 {
            return ClusterClass::None;
        }

        // Partition each axis into early / mid / late thirds.
        let src_bounds = third_bounds(self.n_src);
        let dst_bounds = third_bounds(self.n_dst);

        // Compute the mean cell score within each of the 9 quadrants
        // (src_third × dst_third). Empty quadrants (zero area) get mean 0.
        let mut quadrant_mean = [[0.0f32; 3]; 3]; // [src_third][dst_third]
        for (s_third, (s_lo, s_hi)) in src_bounds.iter().enumerate() {
            for (d_third, (d_lo, d_hi)) in dst_bounds.iter().enumerate() {
                let area = (s_hi.saturating_sub(*s_lo)) * (d_hi.saturating_sub(*d_lo));
                if area == 0 {
                    continue;
                }
                let mut sum = 0.0f32;
                for s in *s_lo..*s_hi {
                    for d in *d_lo..*d_hi {
                        sum += self.cells[s * self.n_dst + d];
                    }
                }
                quadrant_mean[s_third][d_third] = sum / area as f32;
            }
        }

        // The two paper-relevant quadrants:
        //   early_src (0) × mid_dst (1) — the "surprising" cluster
        //   late_src  (2) × mid_dst (1) — the "intuitive" cluster
        let early_to_mid = quadrant_mean[0][1];
        let late_to_mid = quadrant_mean[2][1];

        // The strongest competing quadrant (anything outside the two paper
        // quadrants). Used to enforce "dominant": a paper quadrant must beat
        // every other quadrant to count.
        let mut other_max = 0.0f32;
        for (s, row) in quadrant_mean.iter().enumerate() {
            for (d, &val) in row.iter().enumerate() {
                let is_paper_quadrant = (s == 0 || s == 2) && d == 1;
                if !is_paper_quadrant {
                    other_max = other_max.max(val);
                }
            }
        }

        let early_dominant = early_to_mid >= threshold && early_to_mid >= other_max;
        let late_dominant = late_to_mid >= threshold && late_to_mid >= other_max;

        match (early_dominant, late_dominant) {
            (true, true) => ClusterClass::Both,
            (true, false) => ClusterClass::EarlyToMid,
            (false, true) => ClusterClass::LateToMid,
            (false, false) => ClusterClass::None,
        }
    }
}

/// The paper's two-cluster pattern classification (Research 417 §1.4).
///
/// Returned by [`PermeationMap::classify_two_cluster`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterClass {
    /// Only the `early_src → mid_dst` cluster is dominant (the paper's
    /// "surprising" cluster: source ≈0.1L → target ≈0.45L).
    EarlyToMid,
    /// Only the `late_src → mid_dst` cluster is dominant (the paper's
    /// "intuitive" cluster: source ≈0.8L → target ≈0.45L).
    LateToMid,
    /// Both early→mid and late→mid clusters are dominant (the paper's
    /// combined two-pair heuristic applies).
    Both,
    /// No clear two-cluster pattern.
    None,
}

/// Scan all `(src_stage, dst_stage)` pairs and write the cell scores into
/// `out.cells` (caller pre-allocates).
///
/// For each pair, the caller-supplied `patched_readout(src_stage, dst_stage)`
/// closure must:
///
/// 1. Run a forward pass of the host model with the anchor token's residual
///    state at `dst_stage` overwritten by the anchor's state at `src_stage`
///    (a cross-stage residual relocation — see [`RelocateOp`]).
/// 2. Return the readout `m_patched` (a single f32 — typically a logit
///    difference or span-level score, same convention as Plan 358's
///    `direct_effect_importance`).
///
/// The cell score is then
/// `direct_effect_importance(m_clean, m_corrupt, m_patched)` (Plan 358 Eq 10)
/// — the normalized drop in the readout. A score near `1.0` means relocating
/// `src_stage → dst_stage` alone recovers the capability; near `0.0` means
/// the relocation is ineffective.
///
/// # Zero-alloc guarantee
///
/// Writes into `out.cells` in place — does not grow any `Vec`. The closure
/// may allocate freely (it owns the forward pass); the scan loop itself is
/// allocation-free.
///
/// # Arguments
///
/// - `m_clean` — readout on the clean input (the gold-standard forward pass).
/// - `m_corrupt` — readout on the corrupted input (answer replaced by
///   distractor). Defines the headroom: `m_clean - m_corrupt`.
/// - `patched_readout` — closure returning `m_patched` for each
///   `(src_stage, dst_stage)` pair.
/// - `out` — pre-allocated `PermeationMap` of size `n_src × n_dst`. Must
///   match the dimensions the closure expects.
pub fn permeation_scan_into<F>(
    m_clean: f32,
    m_corrupt: f32,
    patched_readout: F,
    out: &mut PermeationMap,
) where
    F: FnMut(usize, usize) -> f32,
{
    scan_into_inner(
        m_clean,
        m_corrupt,
        patched_readout,
        out,
        /* strict_dims */ false,
    );
}

/// Like [`permeation_scan_into`] but asserts `out.n_src == out.n_dst` (the
/// paper's square `L × L` permeation map on a single model).
///
/// Use this when scanning a model against itself (the common case). Use
/// [`permeation_scan_into`] for the asymmetric case (e.g. cross-model
/// relocation where the source and destination stacks have different depths).
pub fn permeation_scan_square_into<F>(
    m_clean: f32,
    m_corrupt: f32,
    patched_readout: F,
    out: &mut PermeationMap,
) where
    F: FnMut(usize, usize) -> f32,
{
    scan_into_inner(
        m_clean,
        m_corrupt,
        patched_readout,
        out,
        /* strict_dims */ true,
    );
}

fn scan_into_inner<F>(
    m_clean: f32,
    m_corrupt: f32,
    mut patched_readout: F,
    out: &mut PermeationMap,
    strict_dims: bool,
) where
    F: FnMut(usize, usize) -> f32,
{
    if strict_dims {
        assert_eq!(
            out.n_src, out.n_dst,
            "permeation_scan_square_into requires a square map (n_src == n_dst)"
        );
    }
    // The scan loop itself is allocation-free: we overwrite `out.cells` in
    // place. The closure may allocate (it owns the forward pass), but the
    // scan infrastructure adds zero allocations.
    for src in 0..out.n_src {
        for dst in 0..out.n_dst {
            let m_patched = patched_readout(src, dst);
            let score = direct_effect_importance(m_clean, m_corrupt, m_patched);
            out.cells[src * out.n_dst + dst] = score;
        }
    }
}

/// Partition `n` items into three contiguous thirds `[early, mid, late]`.
///
/// Returns `[(lo, hi); 3]` half-open ranges. For `n < 3` the later thirds
/// collapse to empty ranges (e.g. `n=2` → `[(0,1), (1,2), (2,2)]`).
#[inline]
fn third_bounds(n: usize) -> [(usize, usize); 3] {
    if n == 0 {
        return [(0, 0); 3];
    }
    let third = n.div_ceil(3);
    let early = (0, third.min(n));
    let mid = (early.1, (2 * third).min(n));
    let late = (mid.1, n);
    [early, mid, late]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PermeationMap construction & accessors ─────────────────────────────

    #[test]
    fn zeros_has_correct_dimensions() {
        let m = PermeationMap::zeros(4, 6);
        assert_eq!(m.n_src, 4);
        assert_eq!(m.n_dst, 6);
        assert_eq!(m.cells.len(), 24);
        assert!(m.cells.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn cell_read_write_roundtrip() {
        let mut m = PermeationMap::zeros(3, 3);
        *m.cell_mut(1, 2) = 0.5;
        assert_eq!(m.cell(1, 2), 0.5);
        // Out-of-range read returns 0 (clamping).
        assert_eq!(m.cell(5, 5), 0.0);
    }

    #[test]
    #[should_panic(expected = "src_stage 5 out of range")]
    fn cell_mut_panics_on_out_of_range_src() {
        let mut m = PermeationMap::zeros(2, 2);
        let _ = m.cell_mut(5, 0);
    }

    #[test]
    #[should_panic(expected = "dst_stage 9 out of range")]
    fn cell_mut_panics_on_out_of_range_dst() {
        let mut m = PermeationMap::zeros(2, 2);
        let _ = m.cell_mut(0, 9);
    }

    // ── permeation_scan_into ───────────────────────────────────────────────

    #[test]
    fn scan_fills_cells_with_direct_effect_scores() {
        // Synthetic: m_clean=1.0, m_corrupt=0.0 (full headroom).
        // patched_readout returns a known function of (src, dst) so we can
        // verify the cell scores are direct_effect_importance(m_clean,
        // m_corrupt, m_patched).
        let mut map = PermeationMap::zeros(2, 2);
        permeation_scan_into(1.0, 0.0, |src, dst| (src + dst) as f32 * 0.25, &mut map);

        // cell(0,0): m_patched=0.0 → IE=(1-0)/(1-0)=1.0
        assert!((map.cell(0, 0) - 1.0).abs() < 1e-6);
        // cell(0,1): m_patched=0.25 → IE=(1-0.25)/(1-0)=0.75
        assert!((map.cell(0, 1) - 0.75).abs() < 1e-6);
        // cell(1,0): m_patched=0.25 → IE=0.75
        assert!((map.cell(1, 0) - 0.75).abs() < 1e-6);
        // cell(1,1): m_patched=0.5 → IE=0.5
        assert!((map.cell(1, 1) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn scan_overwrites_in_place_zero_alloc() {
        // Run the scan twice on the same buffer; the second run must
        // overwrite (not append), proving the scan is in-place.
        let mut map = PermeationMap::zeros(3, 3);
        permeation_scan_into(1.0, 0.0, |_, _| 0.5, &mut map);
        assert_eq!(map.cells.len(), 9);
        permeation_scan_into(1.0, 0.0, |_, _| 0.25, &mut map);
        assert_eq!(map.cells.len(), 9, "scan grew the buffer — not in-place");
        // All cells should now reflect the second scan (m_patched=0.25 → IE=0.75).
        assert!(map.cells.iter().all(|&v| (v - 0.75).abs() < 1e-6));
    }

    #[test]
    fn scan_handles_clean_equals_corrupt() {
        // m_clean == m_corrupt → direct_effect_importance returns 0 (safe
        // default). The scan should not divide by zero.
        let mut map = PermeationMap::zeros(2, 2);
        permeation_scan_into(0.5, 0.5, |_, _| 0.3, &mut map);
        assert!(map.cells.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn square_scan_rejects_non_square_map() {
        let mut map = PermeationMap::zeros(2, 3);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            permeation_scan_square_into(1.0, 0.0, |_, _| 0.5, &mut map);
        }));
        assert!(result.is_err(), "square scan should reject non-square map");
    }

    // ── classify_two_cluster ───────────────────────────────────────────────

    #[test]
    fn classify_empty_map_is_none() {
        let m = PermeationMap::zeros(0, 0);
        assert_eq!(m.classify_two_cluster(), ClusterClass::None);
    }

    #[test]
    fn classify_all_zero_map_is_none() {
        let m = PermeationMap::zeros(9, 9);
        assert_eq!(m.classify_two_cluster(), ClusterClass::None);
    }

    #[test]
    fn classify_late_to_mid_pattern() {
        // 9x9 map: late_src (rows 6-8) × mid_dst (cols 3-5) is the only
        // hot quadrant. This is the paper's "intuitive" cluster.
        let mut m = PermeationMap::zeros(9, 9);
        for s in 6..9 {
            for d in 3..6 {
                *m.cell_mut(s, d) = 0.8;
            }
        }
        assert_eq!(m.classify_two_cluster(), ClusterClass::LateToMid);
    }

    #[test]
    fn classify_early_to_mid_pattern() {
        // 9x9 map: early_src (rows 0-2) × mid_dst (cols 3-5) is the only
        // hot quadrant. This is the paper's "surprising" cluster.
        let mut m = PermeationMap::zeros(9, 9);
        for s in 0..3 {
            for d in 3..6 {
                *m.cell_mut(s, d) = 0.7;
            }
        }
        assert_eq!(m.classify_two_cluster(), ClusterClass::EarlyToMid);
    }

    #[test]
    fn classify_both_clusters() {
        // 9x9 map: both early→mid and late→mid are hot. This is the paper's
        // combined two-pair case.
        let mut m = PermeationMap::zeros(9, 9);
        for d in 3..6 {
            for s in 0..3 {
                *m.cell_mut(s, d) = 0.7;
            }
            for s in 6..9 {
                *m.cell_mut(s, d) = 0.8;
            }
        }
        assert_eq!(m.classify_two_cluster(), ClusterClass::Both);
    }

    #[test]
    fn classify_off_cluster_dominated_is_none() {
        // If the hot region is in a non-paper quadrant (e.g. late×late),
        // classify_two_cluster should return None.
        let mut m = PermeationMap::zeros(9, 9);
        for s in 6..9 {
            for d in 6..9 {
                *m.cell_mut(s, d) = 0.9;
            }
        }
        assert_eq!(m.classify_two_cluster(), ClusterClass::None);
    }

    #[test]
    fn classify_paper_threshold_suppresses_fp_noise() {
        // Cells just below the threshold (0.05 < 0.1) should not trigger a
        // cluster classification.
        let mut m = PermeationMap::zeros(9, 9);
        for s in 6..9 {
            for d in 3..6 {
                *m.cell_mut(s, d) = 0.05;
            }
        }
        assert_eq!(
            m.classify_two_cluster(),
            ClusterClass::None,
            "sub-threshold noise should not classify as a cluster"
        );
    }

    #[test]
    fn classify_degenerate_small_map() {
        // A 2x2 map: thirds degenerate but the classifier must not panic.
        let mut m = PermeationMap::zeros(2, 2);
        *m.cell_mut(0, 0) = 0.5;
        *m.cell_mut(0, 1) = 0.5;
        // Just verify it runs and returns a valid variant.
        let cls = m.classify_two_cluster();
        assert!(matches!(
            cls,
            ClusterClass::EarlyToMid
                | ClusterClass::LateToMid
                | ClusterClass::Both
                | ClusterClass::None
        ));
    }

    #[test]
    fn classify_threshold_override() {
        // With a high threshold (0.9), a 0.8-strength cluster should be None.
        let mut m = PermeationMap::zeros(9, 9);
        for s in 6..9 {
            for d in 3..6 {
                *m.cell_mut(s, d) = 0.8;
            }
        }
        assert_eq!(
            m.classify_two_cluster_with_threshold(0.9),
            ClusterClass::None
        );
        // With the default threshold (0.1), the same map classifies as LateToMid.
        assert_eq!(m.classify_two_cluster(), ClusterClass::LateToMid);
    }

    // ── third_bounds ───────────────────────────────────────────────────────

    #[test]
    fn third_bounds_9_partitioned_correctly() {
        let [early, mid, late] = third_bounds(9);
        assert_eq!(early, (0, 3));
        assert_eq!(mid, (3, 6));
        assert_eq!(late, (6, 9));
    }

    #[test]
    fn third_bounds_2_degenerates_gracefully() {
        let [early, mid, late] = third_bounds(2);
        assert_eq!(early, (0, 1));
        assert_eq!(mid, (1, 2));
        assert_eq!(late, (2, 2)); // empty
    }

    #[test]
    fn third_bounds_0_all_empty() {
        let [early, mid, late] = third_bounds(0);
        assert_eq!(early, (0, 0));
        assert_eq!(mid, (0, 0));
        assert_eq!(late, (0, 0));
    }
}
