//! Usage-rate (mass/age) KV eviction scoring + the generation-runaway canary
//! (Plan 585, Research 523, arXiv:2608.19920 "Learning how to Forget",
//! Seeger et al., AWS 2026).
//!
//! # The score
//!
//! Raw H2O evicts by cumulative attention mass, which is monotonically
//! biased toward residency age: an old-but-cold row (0.001 mass/step × 1000
//! steps) carries the same cumulative mass as a young-but-hot row (0.5/step
//! × 2). The paper's normalized score converts cumulative evidence into a
//! per-step usage rate:
//!
//! ```text
//! score(row, tick) = cum_mass / max(1, tick - admission_tick)
//! ```
//!
//! O(1) per row per step: `observe` accumulates the caller-supplied
//! attention-mass increment; `score` is one divide. **The signal is
//! caller-supplied** (the `causal_head_importance::suspect_indices` house
//! pattern) — katgpt-core stays leaf-clean; mass producers are consumer-side
//! (riir-gpu kernel byproduct, filed as riir-ai Issue 836, pull-gated on this
//! plan's GOAT).
//!
//! # Per-(b,h) by construction
//!
//! The original H2O sums scores across batch before selecting — a pure
//! coarsening, since each head owns its rows. Here the caller slices per
//! (batch, head); nothing in this module crosses that boundary, and
//! [`select_evict`] takes exactly one head's slice.
//!
//! # Sync boundary
//!
//! None — pure scoring over caller-owned state; no sync surfaces are touched.
//!
//! # The canary (Phase 2)
//!
//! [`RunawayStats`] + [`runaway_gate`] encode the paper's R/p128
//! generation-runaway diagnostic as a promotion gate: **any lossy KV policy
//! (eviction / quantization / compaction) promoted to default MUST pass this
//! gate on a sealed long-context eval.** This extends the Issue 750
//! lossy-surface rule (aggregate perplexity can be flat while family-
//! conditional behavior flips) to the generation axis: the runaway signature
//! (output length running to the cap) is invisible to perplexity-style
//! metrics and to tolerant substring metrics — the paper measured 35–128×
//! output/target blowups while SubEM read fine.

use crate::float_order;

/// Per-row usage bookkeeping. One row per live KV slot, per (batch, head);
/// the caller owns the slot indexing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UsageRow {
    /// Cumulative attention mass received since admission.
    pub cum_mass: f32,
    /// Tick at which the slot was (re)admitted. The min-age guard in
    /// [`score`] makes a fresh row score at its mass, never divide by 0.
    pub admission_tick: u64,
}

/// O(1) accumulate: add one attention-mass increment to a row.
///
/// Non-finite or negative increments are IGNORED (the softmax contract says
/// weights are >= 0 and finite; a violation is a caller bug, so this is
/// `debug_assert!`-loud in debug and silently dropped in release rather than
/// poisoning `cum_mass` for the row's whole lifetime).
#[inline]
pub fn observe(row: &mut UsageRow, mass_increment: f32, _tick: u64) {
    if !mass_increment.is_finite() || mass_increment < 0.0 {
        debug_assert!(
            false,
            "kv_eviction::observe: mass_increment must be finite and >= 0, got {mass_increment}"
        );
        return;
    }
    row.cum_mass += mass_increment;
}

/// Usage-rate score: `cum_mass / max(1, age)`. NaN cannot escape (both
/// inputs are guarded by construction — `cum_mass` only ever accumulates
/// finite non-negative increments and starts at 0; the divisor is >= 1).
#[inline]
pub fn score(row: &UsageRow, tick: u64) -> f32 {
    let age = tick.saturating_sub(row.admission_tick);
    row.cum_mass / (age.max(1) as f32)
}

/// Fixed-capacity usage table over KV slots. Allocates ONCE at construction
/// (`Vec::with_capacity`); `reset_row` + `observe` + `scores` never allocate,
/// so the per-step update path is steady-state allocation-free (G4).
///
/// Slot indexing mirrors the caller's cache exactly — this is a side table,
/// not the cache. When a slot is reused by a new token the caller calls
/// [`UsageScoreTable::reset_row`] (re-admission at the current tick).
pub struct UsageScoreTable {
    rows: Vec<UsageRow>,
    len: usize,
}

impl UsageScoreTable {
    /// Allocate once for `cap` slots. All rows start zeroed with
    /// `admission_tick = 0` (the `max(1, …)` age guard keeps them safe even
    /// if scored before their first `reset_row`).
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            rows: vec![UsageRow::default(); cap],
            len: 0,
        }
    }

    /// Live row count (slots are activated in order by [`Self::reset_row`]).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// (Re)admit slot `idx` at `tick`: zero its mass, stamp the tick, grow
    /// the live prefix if `idx == len`. O(1).
    pub fn reset_row(&mut self, idx: usize, tick: u64) {
        debug_assert!(idx < self.rows.len(), "slot {idx} out of capacity");
        self.rows[idx] = UsageRow {
            cum_mass: 0.0,
            admission_tick: tick,
        };
        if idx >= self.len {
            self.len = idx + 1;
        }
    }

    /// Read-only row access (live prefix only in practice; out-of-prefix
    /// rows are zeroed defaults).
    #[inline]
    pub fn row(&self, idx: usize) -> &UsageRow {
        debug_assert!(idx < self.rows.len(), "slot {idx} out of capacity");
        &self.rows[idx]
    }

    /// Mutable row access for the free-fn [`observe`] hot path.
    #[inline]
    pub fn row_mut(&mut self, idx: usize) -> &mut UsageRow {
        debug_assert!(idx < self.rows.len(), "slot {idx} out of capacity");
        &mut self.rows[idx]
    }

    /// Score every row at `tick` into `out` (reused buffer — no per-step
    /// allocation). Only the live prefix is written.
    pub fn scores(&self, tick: u64, out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.rows[..self.len].iter().map(|r| score(r, tick)));
    }
}

/// Lowest-`k` eviction selection among unpinned rows, reusing `out`.
///
/// Output is in **eviction-priority order**: lowest score first, ties broken
/// by ascending index. Deterministic. NaN-safe via
/// [`float_order::cmp_for_min`] (a corrupt score can never be evicted
/// FIRST — it orders last under min-ordering, the conservative direction
/// for a corrupt value). `pinned` may be shorter than `scores` (missing
/// entries read as unpinned) but not longer.
///
/// ZERO allocation: `out` is the workspace (filled with unpinned indices,
/// sorted by looked-up score, truncated to k) — the steady-state update
/// path allocates nothing (G4, test-pinned).
///
/// Per-(b,h) by construction: the caller passes ONE head's slice; nothing
/// here reduces across heads.
pub fn select_evict_into(scores: &[f32], k: usize, pinned: &[bool], out: &mut Vec<usize>) {
    out.clear();
    if k == 0 || scores.is_empty() {
        return;
    }
    out.extend(
        scores
            .iter()
            .enumerate()
            .filter(|(idx, _)| !pinned.get(*idx).copied().unwrap_or(false))
            .map(|(idx, _)| idx),
    );
    if out.len() > 1 {
        // Order by (score asc, index asc) — a total order (float_order keeps
        // NaN from splitting the comparison; the index tie-break finishes
        // it). The sort key is looked up from `scores` — no tuple pairs, no
        // scratch allocation. Sorted unconditionally: the priority-order
        // contract holds even when k >= candidates.
        out.sort_by(|&a, &b| {
            float_order::cmp_for_min(scores[a], scores[b]).then(a.cmp(&b))
        });
    }
    if out.len() > k {
        out.truncate(k);
    }
}

/// Allocating convenience wrapper for [`select_evict_into`].
pub fn select_evict(scores: &[f32], k: usize, pinned: &[bool]) -> Vec<usize> {
    let mut out = Vec::new();
    select_evict_into(scores, k, pinned, &mut out);
    out
}

// ── Phase 2: the generation-runaway canary ───────────────────────────────

/// Generation-runaway statistics over a batch of generations (Plan 585 T2.1;
/// the paper's R/p128 diagnostic).
///
/// The failure mode this catches: under a train/inference attention mismatch
/// (e.g. an eviction policy the model was not trained with), generation
/// degrades into length-runaway — the paper measured output lengths 35–128×
/// the target, saturating the token cap — while perplexity-style and
/// tolerant substring metrics read FINE. `r_median` exposes the length
/// axis; `p_cap` exposes the saturation axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunawayStats {
    /// Median of per-sample `output_len / target_len` ratios. 1.0 = exact.
    /// Samples with `target_len == 0` are skipped (a zero target has no
    /// defined ratio).
    pub r_median: f32,
    /// Fraction of samples whose output length reached `cap` (the
    /// generation-token cap). Saturation is the hard ceiling of the runaway
    /// signature.
    pub p_cap: f32,
    /// Number of samples contributing (zero-target samples excluded).
    pub n: usize,
}

impl RunawayStats {
    /// Pure fn, zero deps. Mismatched slice lengths truncate to the shorter;
    /// zero-target samples are skipped (all skipped ⇒ `n == 0` and both
    /// stats are 0.0 — an empty eval FAILS the gate rather than passing
    /// vacuously, see [`runaway_gate`]).
    pub fn from_generations(output_lens: &[usize], target_lens: &[usize], cap: usize) -> Self {
        let n_pairs = output_lens.len().min(target_lens.len());
        let mut ratios: Vec<f32> = Vec::with_capacity(n_pairs);
        let mut at_cap = 0usize;
        for i in 0..n_pairs {
            let target = target_lens[i];
            if target == 0 {
                continue;
            }
            let out = output_lens[i];
            if cap > 0 && out >= cap {
                at_cap += 1;
            }
            ratios.push(out as f32 / target as f32);
        }
        let n = ratios.len();
        let r_median = if n == 0 {
            0.0
        } else {
            // NaN-safe comparator by convention; ratios from finite usize
            // division cannot be NaN anyway.
            ratios.sort_by(|a, b| float_order::asc(*a, *b));
            if n % 2 == 1 {
                ratios[n / 2]
            } else {
                (ratios[n / 2 - 1] + ratios[n / 2]) / 2.0
            }
        };
        let p_cap = if n == 0 { 0.0 } else { at_cap as f32 / n as f32 };
        Self {
            r_median,
            p_cap,
            n,
        }
    }
}

/// The promotion gate (Plan 585 T2.2): a lossy KV policy may promote to
/// default only if its sealed long-context eval shows `r_median <= r_max`
/// AND `p_cap <= p_cap_max`.
///
/// Encoded rule (doc = contract): **any lossy KV policy (eviction,
/// quantization, compaction) promoted to default MUST pass this gate on a
/// sealed long-context eval** — the generation-axis extension of the Issue
/// 750 lossy-surface rule. Suggested thresholds for a serving eval:
/// `r_max = 1.5`, `p_cap_max = 0.05` (the paper's healthy co-trained arms
/// sit near R ≈ 1; runaway arms sit at 35–128× or pinned at cap).
///
/// An eval with ZERO contributing samples returns `false` — an empty eval
/// must fail the gate, never pass it vacuously (the tile-loop-gate lesson).
pub fn runaway_gate(stats: &RunawayStats, r_max: f32, p_cap_max: f32) -> bool {
    if stats.n == 0 {
        return false;
    }
    stats.r_median <= r_max && stats.p_cap <= p_cap_max
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── T1.2 score / observe ──────────────────────────────────────────

    #[test]
    fn score_is_mass_over_max_one_age() {
        let mut row = UsageRow {
            cum_mass: 0.0,
            admission_tick: 100,
        };
        // Fresh row at tick 100: age 0 -> divisor 1, scores at its mass.
        observe(&mut row, 0.5, 100);
        assert_eq!(score(&row, 100), 0.5);
        // 2 ticks later: age 2.
        observe(&mut row, 0.5, 102);
        assert_eq!(score(&row, 102), 1.0 / 2.0);
    }

    #[test]
    fn observe_guard_predicate_rejects_bad_increments() {
        // The guard predicate itself (debug builds abort on the first bad
        // call by design, so the predicate is exercised directly here).
        assert!(!f32::NAN.is_finite());
        assert!(!f32::INFINITY.is_finite());
        // (The remaining tautological guards — `-0.5 < 0.0`, `0.0.is_finite()` —
        // were folded into this comment; the predicate semantics they documented
        // are non-finite and negative increments fail, finite non-negative pass.)
        // In release the bad increments are dropped, not accumulated:
        if !cfg!(debug_assertions) {
            let mut row = UsageRow {
                cum_mass: 1.0,
                admission_tick: 0,
            };
            observe(&mut row, f32::NAN, 1);
            observe(&mut row, -0.5, 1);
            assert_eq!(row.cum_mass, 1.0);
        }
    }

    #[test]
    fn score_never_divides_by_zero() {
        let row = UsageRow {
            cum_mass: 3.0,
            admission_tick: 1_000,
        };
        // Scoring BEFORE admission_tick: saturating_sub -> 0 -> divisor 1.
        assert_eq!(score(&row, 500), 3.0);
    }

    // ── T1.1 table shape ──────────────────────────────────────────────

    #[test]
    fn table_allocates_once_and_reset_activates_prefix() {
        let mut t = UsageScoreTable::with_capacity(8);
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        t.reset_row(3, 10);
        assert_eq!(t.len(), 4, "reset at idx 3 activates the 0..=3 prefix");
        t.reset_row(1, 12);
        assert_eq!(t.len(), 4);
        assert_eq!(t.row(1).admission_tick, 12);
        assert_eq!(t.row(0).cum_mass, 0.0);
    }

    #[test]
    fn scores_fill_live_prefix_only() {
        let mut t = UsageScoreTable::with_capacity(4);
        t.reset_row(0, 0);
        t.reset_row(1, 0);
        observe(t.row_mut(0), 2.0, 4);
        observe(t.row_mut(1), 1.0, 2);
        let mut out = Vec::new();
        t.scores(4, &mut out);
        // Row 0: mass 2.0 over age 4 = 0.5. Row 1: mass 1.0 over age 4 =
        // 0.25 (admitted at tick 0, observed at tick 2, scored at tick 4).
        assert_eq!(out, vec![0.5, 0.25]);
        assert_eq!(out.len(), t.len());
    }

    // ── T1.3 selection ────────────────────────────────────────────────

    #[test]
    fn select_evict_lowest_k_priority_order() {
        // Ties: rows 0, 2, 4 all score 1.0 -> tie-break ascending index.
        let scores = [1.0, 5.0, 1.0, 5.0, 1.0];
        assert_eq!(select_evict(&scores, 2, &[]), vec![0, 2]);
        // Pinned rows are never selected even when lowest.
        let pinned = [true, false, false, false, false];
        assert_eq!(select_evict(&scores, 2, &pinned), vec![2, 4]);
        // Priority order: lowest score first, ties ascending index.
        assert_eq!(select_evict(&scores, 10, &pinned), vec![2, 4, 1, 3]);
        // k = 0 / empty: no-op.
        assert!(select_evict(&scores, 0, &[]).is_empty());
        assert!(select_evict(&[], 3, &[]).is_empty());
    }

    #[test]
    fn select_evict_nan_cannot_be_evicted_first() {
        // NaN sorts LAST under cmp_for_min (the corrupt value is evicted
        // last, never first), and cannot split the total order.
        let scores = [f32::NAN, 3.0, 1.0];
        assert_eq!(select_evict(&scores, 1, &[]), vec![2]);
        let scores2 = [f32::NAN, f32::NAN, 2.0];
        assert_eq!(select_evict(&scores2, 1, &[]), vec![2]);
        // All-NaN: still returns deterministically without panicking. NaN
        // orders LAST (the conservative direction), so the real value is
        // evicted first and the NaNs follow in index order.
        assert_eq!(select_evict(&scores2, 2, &[]), vec![2, 0]);
    }

    #[test]
    fn select_evict_into_reuses_buffer() {
        let mut out = vec![99usize; 4];
        select_evict_into(&[3.0, 1.0, 2.0], 2, &[], &mut out);
        assert_eq!(out, vec![1, 2]);
        // Second call into the same buffer: cleared first.
        select_evict_into(&[9.0, 0.0], 1, &[], &mut out);
        assert_eq!(out, vec![1]);
    }

    // ── T1.4 property tests ───────────────────────────────────────────

    #[test]
    fn monotone_in_mass() {
        let mut a = UsageRow {
            cum_mass: 0.0,
            admission_tick: 0,
        };
        let mut b = UsageRow {
            cum_mass: 0.0,
            admission_tick: 0,
        };
        observe(&mut a, 1.0, 10);
        observe(&mut b, 5.0, 10);
        assert!(score(&b, 10) > score(&a, 10));
        // Monotonicity holds at every later tick too.
        for tick in [11u64, 50, 1_000] {
            assert!(score(&b, tick) > score(&a, tick));
        }
    }

    #[test]
    fn anti_monotone_in_age() {
        let mut row = UsageRow {
            cum_mass: 0.0,
            admission_tick: 0,
        };
        observe(&mut row, 1.0, 0);
        let s0 = score(&row, 1);
        let s1 = score(&row, 2);
        let s2 = score(&row, 10);
        assert!(s0 > s1 && s1 > s2, "score must decay with age: {s0} {s1} {s2}");
    }

    #[test]
    fn pinned_rows_never_selected_property() {
        // Property over many layouts: for every k, the pinned set and the
        // selection are disjoint.
        let scores: Vec<f32> = (0..32).map(|i| ((i * 7) % 13) as f32 / 3.0).collect();
        let pinned: Vec<bool> = (0..32).map(|i| i % 5 == 0).collect();
        for k in 1..=32 {
            let sel = select_evict(&scores, k, &pinned);
            assert!(sel.len() <= k);
            for &s in &sel {
                assert!(!pinned[s], "pinned row {s} selected at k={k}");
            }
            // Determinism: identical input -> identical output.
            assert_eq!(select_evict(&scores, k, &pinned), sel);
        }
        // Sanity: at k = n the selection is the full unpinned set in
        // eviction-priority order (score asc, ties ascending index).
        let mut expected: Vec<usize> = (0..32).filter(|&i| !pinned[i]).collect();
        expected.sort_by(|&a, &b| {
            float_order::cmp_for_min(scores[a], scores[b]).then(a.cmp(&b))
        });
        assert_eq!(select_evict(&scores, 32, &pinned), expected);
    }

    #[test]
    fn zero_beta_pin_is_no_op_on_selection() {
        // beta=0-pin == no pins at all: same selection.
        let scores = [4.0, 1.0, 3.0, 2.0];
        let no_pins = select_evict(&scores, 2, &[]);
        let zero_pins = select_evict(&scores, 2, &[false; 4]);
        assert_eq!(no_pins, zero_pins);
    }

    #[test]
    fn determinism_bit_identical_across_runs() {
        // LCG stream -> observe/score/select twice -> bit-identical outputs.
        let n = 256;
        let mut t1 = UsageScoreTable::with_capacity(n);
        let mut t2 = UsageScoreTable::with_capacity(n);
        let mut rng = SimpleLcg::new(42);
        for idx in 0..n {
            let tick = rng.next_u64() % 100;
            t1.reset_row(idx, tick);
            t2.reset_row(idx, tick);
        }
        let mut s1 = Vec::new();
        let mut s2 = Vec::new();
        for step in 0..1_000u64 {
            let idx = (rng.next_u64() % n as u64) as usize;
            let mass = (rng.next_u64() % 1000) as f32 / 1000.0;
            // Same draw sequence drives BOTH tables (replay).
            observe(t1.row_mut(idx), mass, step);
            observe(t2.row_mut(idx), mass, step);
        }
        t1.scores(1_000, &mut s1);
        t2.scores(1_000, &mut s2);
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "scores must be bit-identical");
        }
        let pinned = [false; 256];
        let out1 = select_evict(&s1, 16, &pinned);
        let out2 = select_evict(&s2, 16, &pinned);
        assert_eq!(out1, out2);
    }

    // ── T1.5 reference parity ─────────────────────────────────────────

    #[test]
    fn reference_parity_vs_naive_recompute() {
        // Naive: recompute cum_mass and admission from the raw stream each
        // time. The incremental table must produce bit-identical scores.
        let mut rng = SimpleLcg::new(7);
        let n = 64;
        let mut t = UsageScoreTable::with_capacity(n);
        let mut naive_mass = vec![0.0f32; n];
        let mut naive_adm = vec![0u64; n];
        for (idx, adm) in naive_adm.iter_mut().enumerate() {
            let tick = rng.next_u64() % 50;
            t.reset_row(idx, tick);
            *adm = tick;
        }
        let mut naive_scores = Vec::new();
        let mut table_scores = Vec::new();
        let mut step = 50u64;
        for _ in 0..2_000 {
            let idx = (rng.next_u64() % n as u64) as usize;
            let mass = (rng.next_u64() % 100) as f32 / 100.0;
            step += 1;
            observe(t.row_mut(idx), mass, step);
            naive_mass[idx] += mass; // same accumulation order per row
            if step.is_multiple_of(100) {
                naive_scores.clear();
                for i in 0..n {
                    let age = step.saturating_sub(naive_adm[i]).max(1) as f32;
                    naive_scores.push(naive_mass[i] / age);
                }
                t.scores(step, &mut table_scores);
                assert_eq!(table_scores.len(), naive_scores.len());
                for (a, b) in table_scores.iter().zip(naive_scores.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "incremental score diverged from naive recompute"
                    );
                }
            }
        }
    }

    // ── T2.2 / T2.3 canary ────────────────────────────────────────────

    #[test]
    fn runaway_stats_median_and_cap() {
        // Ratios [1.0, 1.0, 1.25, 0.75, 8.0(at cap)] — median 1.0.
        let stats = RunawayStats::from_generations(&[8, 8, 10, 6, 64], &[8, 8, 8, 8, 8], 64);
        assert_eq!(stats.n, 5);
        assert_eq!(stats.p_cap, 1.0 / 5.0);
        assert!(
            (stats.r_median - 1.0).abs() < 1e-6,
            "median of [0.75,1.0,1.0,1.25,8.0] is 1.0, got {}",
            stats.r_median
        );
        // Even count: mean of the two middles.
        let stats2 = RunawayStats::from_generations(&[8, 16], &[8, 8], 64);
        assert!((stats2.r_median - 1.5).abs() < 1e-6);
    }

    #[test]
    fn zero_target_samples_are_skipped() {
        let stats = RunawayStats::from_generations(&[8, 8], &[0, 8], 64);
        assert_eq!(stats.n, 1);
        assert!((stats.r_median - 1.0).abs() < 1e-6);
    }

    #[test]
    fn empty_eval_fails_the_gate_never_passes_vacuously() {
        let stats = RunawayStats::from_generations(&[], &[], 64);
        assert_eq!(stats.n, 0);
        assert!(
            !runaway_gate(&stats, 1.5, 0.05),
            "empty eval must FAIL the gate"
        );
        // All-zero-target eval: same vacuity guard.
        let stats2 = RunawayStats::from_generations(&[8, 8], &[0, 0], 64);
        assert!(!runaway_gate(&stats2, 1.5, 0.05));
    }

    #[test]
    fn planted_over_eviction_fixture_fails_the_gate() {
        // T2.3 non-vacuity: the runaway arm (7/8 outputs run to cap at 8x
        // target) FAILS; the healthy arm PASSES on identical thresholds —
        // the fails-before/passes-after pair.
        let runaway = RunawayStats::from_generations(
            &[64, 64, 64, 64, 64, 64, 64, 8],
            &[8, 8, 8, 8, 8, 8, 8, 8],
            64,
        );
        assert!(
            !runaway_gate(&runaway, 1.5, 0.05),
            "over-eviction arm must FAIL"
        );
        let healthy = RunawayStats::from_generations(
            &[8, 9, 8, 10, 8, 7, 9, 8],
            &[8, 8, 8, 8, 8, 8, 8, 8],
            64,
        );
        assert!(runaway_gate(&healthy, 1.5, 0.05), "healthy arm must PASS");
    }

    #[test]
    fn gate_boundary_conditions() {
        let at_limit = RunawayStats {
            r_median: 1.5,
            p_cap: 0.05,
            n: 100,
        };
        assert!(runaway_gate(&at_limit, 1.5, 0.05), "at-limit is <=, passes");
        let over = RunawayStats {
            r_median: 1.5 + f32::EPSILON,
            p_cap: 0.05,
            n: 100,
        };
        assert!(!runaway_gate(&over, 1.5, 0.05));
    }

    // ── T3.4 G4: zero steady-state allocation on the update path ──────

    #[cfg(debug_assertions)]
    #[test]
    fn update_path_is_allocation_free() {
        // The lib test binary installs TrackingAllocator (TEST_GLOBAL_ALLOC);
        // counters are per-thread. Construct once, then measure the
        // steady-state cycle: reset_row + observe + scores(into reused
        // buffer) + select_evict_into must be exactly 0 allocs.
        let cap = 128;
        let mut t = UsageScoreTable::with_capacity(cap);
        for i in 0..cap {
            t.reset_row(i, i as u64);
        }
        let mut scores = Vec::with_capacity(cap);
        let mut victims = Vec::new();
        let pinned = [false; 128];
        // warm (buffers settle at capacity)
        for step in 0..10u64 {
            t.reset_row(step as usize % cap, step);
            observe(t.row_mut(step as usize % cap), 0.5, step);
            t.scores(step, &mut scores);
            crate::kv_eviction::select_evict_into(&scores, 2, &pinned, &mut victims);
        }
        crate::alloc::reset_alloc_stats();
        for step in 100..200u64 {
            t.reset_row(step as usize % cap, step);
            observe(t.row_mut(step as usize % cap), 0.5, step);
            t.scores(step, &mut scores);
            crate::kv_eviction::select_evict_into(&scores, 2, &pinned, &mut victims);
        }
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(
            count, 0,
            "steady-state update path must not allocate (got {count} allocs)"
        );
    }

    /// Minimal deterministic LCG (xorshift) — the benches/tests house
    /// pattern (bench_313's SimpleRng).
    struct SimpleLcg(u64);
    impl SimpleLcg {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 { 1 } else { seed })
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }
}
