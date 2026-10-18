//! Extension-count (freedom-of-function) selection criterion — Research 486 /
//! Issue 665. Opt-in PoC-gated primitive; promotion to default requires the
//! Issue 665 PoC gate (freedom-guided near-best beats min-loss AND
//! random-near-best under a declared distribution shift).
//!
//! Source: Bennett, "Why the Third Axis Is Freedom" (arXiv:2608.05423).
//! Freedom of function w(π) = Π_c (2^{a_c} − 1), where a_c = the count of
//! permitted outputs at context c over a DECLARED FINITE partition. Weakest
//! *correct* policies are likeliest to remain correct under unknown future
//! demands (Cor 4: generalization probability ∝ freedom in the unseen-context
//! vocabulary) — the anti-Occam position: prefer the weakest hypothesis.
//!
//! # What this is
//!
//! A **modelless, deterministic selection criterion** (not a loss, not a
//! trainer). Among best-of-K candidates within a loss gate of the winner,
//! prefer the one that opens an unoccupied output region (Δ log extension
//! count). This module provides the closed-form scoring; the renoise-CE
//! sibling [`crate::renoise_ce::best_of_n_freedom`] provides the wired
//! selection mode over a caller-owned occupancy table.
//!
//! # Operational conventions (documented deviations from the raw formula)
//!
//! - **Empty contexts (a_c = 0) are excluded from the product** (factor 1):
//!   the raw factor 2^0 − 1 = 0 makes the whole product zero — the
//!   infeasible-policy boundary the paper itself flags (§13). An empty
//!   context means "no permission evidence yet", not "provably infeasible".
//! - **First activation of a context** (0 → 1 cells) is pinned to
//!   [`FIRST_ACTIVATION_GAIN`] = 2.0, strictly above the largest finite
//!   increment ln 3 ≈ 1.0986 (1 → 2 cells). The raw increment is +∞ (from
//!   the excluded state); the pin preserves the ordering "open an unvisited
//!   context first" (Cor 4: unseen contexts dominate future compatibility)
//!   with a finite, stable constant.
//! - **Occupancy is an ESTIMATE of permission**: a cell counts toward a_c
//!   once at least one selected candidate landed in it (threshold 1 — the
//!   minimal thresholded-prototype admissibility rule; the paper's
//!   controller used the same shape with decay).
//!
//! # Vocabulary note
//!
//! `speculative/qmc` uses "min freedom" to mean sample INDEPENDENCE — the
//! opposite notion. API names here say `extension_count` / `freedom_gain`;
//! never reuse the bare word `freedom` for module paths (Issue 665 note).
//!
//! # Hot-path design
//!
//! Free functions are zero-alloc over caller slices. [`freedom_gain`] scans
//! the cell→context partition (O(cells)) — selection-time cost, NOT a
//! hot-path primitive; the [`ExtensionOccupancy`] struct maintains per-
//! context counts incrementally for O(1) gain queries.
//!
//! Extension-count (freedom-of-function) selection criterion — closed-form
//! near-best selection over a declared finite output partition (Research 486,
//! arXiv:2608.05423, Issue 665). Opt-in PoC-gated: promotion to default
//! requires the Issue 665 PoC gate (freedom-guided near-best beats min-loss
//! AND random-near-best under a declared distribution shift) plus a
//! production consumer.

/// Natural-log freedom of a permission profile: `Σ_c ln(2^{a_c} − 1)`.
///
/// `context_counts[c]` = occupied-cell count a_c for context c over the
/// declared partition. Contexts with a_c = 0 are excluded (factor 1 — see
/// module docs). Returns 0.0 for an all-empty profile.
pub fn log_freedom(context_counts: &[u32]) -> f32 {
    let mut total = 0.0f32;
    for &a in context_counts {
        if a == 0 {
            continue;
        }
        total += context_factor_ln(a);
    }
    total
}

/// Pinned gain for the first activation of an empty context (0 → 1 cells).
///
/// Strictly above the largest finite increment ln 3 ≈ 1.0986 (1 → 2 cells),
/// preserving "open an unvisited context first" with a finite constant
/// (the raw increment from the excluded empty state is +∞ — see module docs).
pub const FIRST_ACTIVATION_GAIN: f32 = 2.0;

/// ln(2^a − 1), exact for a < 63 via u64; a ≥ 63 approximated by a·ln 2
/// (2^a − 1 ≈ 2^a; relative error < 2^−62, far below f32 resolution).
#[inline]
fn context_factor_ln(a: u32) -> f32 {
    if a >= 63 {
        a as f32 * std::f32::consts::LN_2
    } else {
        (((1u64 << a) - 1) as f32).ln()
    }
}

/// Δ ln-freedom from activating `candidate_cell`, given per-cell occupancy
/// counts and the declared cell→context partition.
///
/// Returns 0.0 when the candidate's cell is already occupied (no new region
/// opened). Otherwise returns the increment of activating one more cell in
/// the candidate's context: ln(2^{a+1} − 1) − ln(2^a − 1) for a ≥ 1, or
/// [`FIRST_ACTIVATION_GAIN`] when the context is empty (a = 0).
///
/// NOTE: 3-arg form (the Issue 665 sketch had 2) — the partition slice is
/// load-bearing: without it "occupied cell in context a" is indistinguishable
/// from "fresh cell in context a". O(cells) partition scan; use
/// [`ExtensionOccupancy::freedom_gain`] for O(1) maintained-state queries.
pub fn freedom_gain(cell_counts: &[u32], cell_context: &[u16], candidate_cell: usize) -> f32 {
    debug_assert_eq!(
        cell_counts.len(),
        cell_context.len(),
        "occupancy and partition slices must be parallel"
    );
    if candidate_cell >= cell_counts.len() || cell_counts[candidate_cell] > 0 {
        return 0.0;
    }
    let ctx = cell_context[candidate_cell];
    let mut a: u32 = 0;
    for (i, &c) in cell_context.iter().enumerate() {
        if c == ctx && cell_counts[i] > 0 {
            a += 1;
        }
    }
    if a == 0 {
        FIRST_ACTIVATION_GAIN
    } else {
        context_factor_ln(a + 1) - context_factor_ln(a)
    }
}

/// Near-best loss gate (Research 486: "within a loss gate of the winner").
///
/// Lower loss = better throughout. `Relative` assumes non-negative losses
/// (a negative best loss shrinks the gate — use `Absolute` there).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LossGate {
    /// `loss <= best + tol`.
    Absolute(f32),
    /// `loss <= best * (1 + tol)` — non-negative losses only.
    Relative(f32),
}

impl LossGate {
    /// Does `loss` fall within the gate of `best_loss`?
    #[inline]
    pub fn admits(&self, loss: f32, best_loss: f32) -> bool {
        match self {
            Self::Absolute(tol) => loss <= best_loss + tol,
            Self::Relative(tol) => loss <= best_loss * (1.0 + tol),
        }
    }
}

/// Occupancy state over a declared finite output partition, grouped into
/// contexts. Caller-owned and persistent across selections — the criterion
/// is a running-state selection rule (the paper's controller kept a decayed
/// occupancy table across training steps).
///
/// Allocates once at construction; all queries are zero-alloc. Gain queries
/// are O(1) via incrementally maintained per-context counts.
#[derive(Debug, Clone)]
pub struct ExtensionOccupancy {
    /// Per-cell landing counts (`> 0` = occupied).
    cell_counts: Vec<u32>,
    /// Declared cell → context partition.
    cell_context: Vec<u16>,
    /// Per-context occupied-cell counts (a_c), maintained by [`Self::record`].
    context_counts: Vec<u32>,
}

impl ExtensionOccupancy {
    /// New empty occupancy. `cell_context[i]` = context of cell i;
    /// `n_contexts` = number of distinct contexts (≥ max index + 1).
    pub fn new(cell_context: Vec<u16>, n_contexts: usize) -> Self {
        debug_assert!(cell_context.iter().all(|&c| (c as usize) < n_contexts));
        Self {
            cell_counts: vec![0; cell_context.len()],
            cell_context,
            context_counts: vec![0; n_contexts],
        }
    }

    /// Total cells in the declared partition.
    pub fn cells(&self) -> usize {
        self.cell_counts.len()
    }

    /// Landing count of one cell.
    pub fn cell_count(&self, cell: usize) -> u32 {
        self.cell_counts[cell]
    }

    /// Per-context occupied-cell counts (input shape of [`log_freedom`]).
    pub fn context_counts(&self) -> &[u32] {
        &self.context_counts
    }

    /// Δ ln-freedom if `cell` were selected now — O(1). Agrees with the
    /// free function [`freedom_gain`] by construction (pinned by test).
    pub fn freedom_gain(&self, cell: usize) -> f32 {
        if cell >= self.cell_counts.len() || self.cell_counts[cell] > 0 {
            return 0.0;
        }
        let a = self.context_counts[self.cell_context[cell] as usize];
        if a == 0 {
            FIRST_ACTIVATION_GAIN
        } else {
            context_factor_ln(a + 1) - context_factor_ln(a)
        }
    }

    /// Record a landing: increments the cell count and, on first landing,
    /// the context's occupied-cell count.
    pub fn record(&mut self, cell: usize) {
        if cell >= self.cell_counts.len() {
            return;
        }
        self.cell_counts[cell] += 1;
        if self.cell_counts[cell] == 1 {
            self.context_counts[self.cell_context[cell] as usize] += 1;
        }
    }

    /// Current natural-log freedom (see [`log_freedom`]).
    pub fn log_freedom(&self) -> f32 {
        log_freedom(&self.context_counts)
    }

    /// Number of distinct occupied cells.
    pub fn occupied_cells(&self) -> usize {
        self.cell_counts.iter().filter(|&&c| c > 0).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force extension count by subset ENUMERATION (independent
    /// computation path): per context with a permitted outputs, count all
    /// non-empty subsets by walking every bitmask; product across contexts.
    fn brute_force_extension_ln(context_counts: &[u32]) -> f32 {
        let mut product: u64 = 1;
        for &a in context_counts {
            if a == 0 {
                continue;
            }
            assert!(a < 20, "brute-force enumeration needs a < 20");
            let mut non_empty: u64 = 0;
            for mask in 0..(1u64 << a) {
                if mask.count_ones() > 0 {
                    non_empty += 1;
                }
            }
            product *= non_empty;
        }
        (product as f32).ln()
    }

    #[test]
    fn log_freedom_matches_brute_force_enumeration() {
        // Small vocabularies where enumeration is exact (the Issue 665 pin).
        let cases: &[&[u32]] = &[
            &[1],
            &[2],
            &[3],
            &[4],
            &[1, 1],
            &[2, 3],
            &[4, 4, 2],
            &[5, 1, 3],
        ];
        for &counts in cases {
            let got = log_freedom(counts);
            let want = brute_force_extension_ln(counts);
            assert!(
                (got - want).abs() < 1e-4,
                "counts={counts:?}: log_freedom={got} vs brute-force ln={want}"
            );
        }
    }

    #[test]
    fn empty_contexts_excluded_from_product() {
        // a=0 contexts contribute factor 1 (module-doc convention).
        assert_eq!(log_freedom(&[]), 0.0);
        assert_eq!(log_freedom(&[0, 0]), 0.0);
        assert!((log_freedom(&[0, 3]) - log_freedom(&[3])).abs() < 1e-6);
    }

    #[test]
    fn gain_is_positive_and_monotone_decreasing_in_a() {
        // gain(a) = ln(2^{a+1}−1) − ln(2^a−1), a ≥ 1: positive, decreasing.
        let mut prev = f32::INFINITY;
        for a in 1u32..10 {
            let g = context_factor_ln(a + 1) - context_factor_ln(a);
            assert!(g > 0.0, "gain must be positive at a={a}");
            assert!(g < prev, "gain must decrease in a (a={a})");
            prev = g;
        }
        // Largest finite increment is ln 3 (a=1 → 2).
        let g1 = context_factor_ln(2) - context_factor_ln(1);
        assert!((g1 - 3.0f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn occupied_cell_has_zero_gain() {
        let counts = [1, 2, 0, 0];
        let ctx = [0u16, 0, 1, 1];
        assert_eq!(freedom_gain(&counts, &ctx, 0), 0.0);
        assert_eq!(freedom_gain(&counts, &ctx, 1), 0.0);
        // Out-of-bounds candidate: no gain (defensive).
        assert_eq!(freedom_gain(&counts, &ctx, 99), 0.0);
    }

    #[test]
    fn first_activation_dominates_all_finite_gains() {
        // Empty context: pinned FIRST_ACTIVATION_GAIN > ln 3 (max finite).
        let counts = [1, 2, 0];
        let ctx = [0u16, 0, 1];
        let g = freedom_gain(&counts, &ctx, 2);
        assert_eq!(g, FIRST_ACTIVATION_GAIN);
        assert!(g > 3.0f32.ln(), "first activation must outrank a=1 gain");
        // Fresh cell in an occupied context: finite decrement gain.
        let g2 = freedom_gain(&[1, 0], &[0, 0], 1);
        assert!((g2 - 3.0f32.ln()).abs() < 1e-6, "a=1→2 gain is ln 3");
    }

    #[test]
    fn loss_gate_absolute_and_relative() {
        let abs = LossGate::Absolute(0.5);
        assert!(abs.admits(1.0, 0.7));
        assert!(abs.admits(1.2, 0.7)); // exactly at best + tol
        assert!(!abs.admits(1.21, 0.7));
        let rel = LossGate::Relative(0.1);
        assert!(rel.admits(1.1, 1.0));
        assert!(!rel.admits(1.11, 1.0));
    }

    #[test]
    fn struct_gain_matches_free_fn_and_records() {
        // Partition: 2 contexts × 3 cells. Deterministic occupancy pattern.
        let ctx = vec![0u16, 0, 0, 1, 1, 1];
        let mut occ = ExtensionOccupancy::new(ctx.clone(), 2);
        assert_eq!(occ.occupied_cells(), 0);
        assert_eq!(occ.log_freedom(), 0.0);

        occ.record(0); // context 0: first activation
        occ.record(3); // context 1: first activation
        occ.record(0); // repeat landing — no new cell
        assert_eq!(occ.cell_count(0), 2);
        assert_eq!(occ.context_counts(), &[1, 1]);
        assert_eq!(occ.occupied_cells(), 2);

        // Every cell's struct gain must equal the free-fn gain.
        for cell in 0..ctx.len() {
            assert_eq!(
                occ.freedom_gain(cell),
                freedom_gain(&occ_cell_counts(&occ), &ctx, cell),
                "cell {cell}"
            );
        }
        // Fresh cell in context 0 (a=1): ln 3.
        assert!((occ.freedom_gain(1) - 3.0f32.ln()).abs() < 1e-6);
        // log_freedom = 2 × ln(2^1−1) = 0 (both contexts at a=1).
        assert_eq!(occ.log_freedom(), 0.0);

        occ.record(1);
        occ.record(4);
        // Both contexts now a=2: freedom = 2 × ln 3.
        assert!((occ.log_freedom() - 2.0 * 3.0f32.ln()).abs() < 1e-5);
        assert_eq!(occ.context_counts(), &[2, 2]);
    }

    /// Expose the private per-cell counts for the free-fn cross-check.
    fn occ_cell_counts(occ: &ExtensionOccupancy) -> Vec<u32> {
        (0..occ.cells()).map(|c| occ.cell_count(c)).collect()
    }

    #[test]
    fn large_context_counts_do_not_overflow() {
        // a ≥ 63 approximates by a·ln 2; monotone + finite.
        let g1 = log_freedom(&[63, 63]);
        let g2 = log_freedom(&[100, 100]);
        assert!(g1.is_finite() && g2.is_finite());
        assert!(g2 > g1);
    }
}
