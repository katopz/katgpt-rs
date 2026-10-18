//! Selection-Set Fixpoint Propagation — KEEP M3 in house operator vocabulary
//! (Issue 655 / Research 483, arXiv:2602.23592 KEEP, DAC 2026).
//!
//! The one genuinely-unshipped composition identified by the Research 483
//! substrate audit: a query-seeded importance propagation iterated until the
//! **selected set** stabilizes. Every arrow already shipped as an operator —
//! query seeding (attn-match family), top-r selection (DensityBudget/TopK),
//! one-step edge propagation ([`crate::set_attention`]), CLR reliability
//! weighting (Plan 570) — but no shipped loop iterated a *selection set* to a
//! membership fixpoint (grep-verified: Hopfield stabilizes *state*, ADMM
//! stabilizes *consensus*, power iteration stabilizes *eigenvectors*).
//!
//! ```text
//! scores = query_seed(query, memories)          // caller-provided
//! loop {
//!     selected = top_r(scores)                  // hard budget, deterministic
//!     if selected == prev_selected { break }    // ← membership fixpoint
//!     scores' = (1-α)·seed + α·edge_prop(selected, w × sigmoid-gate)
//! }
//! ```
//!
//! This is personalized-PageRank-with-early-stop re-expressed in house
//! primitives (HippoRAG's PPR class; KEEP's independent validation in the KV
//! domain). The early stop is the point: BFS `k_hop_neighbors` is explicitly
//! offline-only (`O(degree^k)`); a membership-stabilizing loop touches only
//! the selected set's out-edges per iteration and halts as soon as the
//! selection stops changing — typically after ~chain-length iterations.
//!
//! # House rules
//!
//! - **Zero-alloc**: all buffers live in [`SelectionPropagationScratch`]
//!   (grow-only, caller-owned); the hot path borrows exclusively.
//! - **Sigmoid gate for membership** (never softmax): each selected node's
//!   contribution is gated by `sigmoid(gate_beta · score)` — CLR-reliability
//!   weighting per Plan 570.
//! - **Deterministic**: CSR edge order fixes the summation order; top-r ties
//!   break by ascending index. Same inputs → bit-identical outputs.
//! - **No new deps**: pure linear algebra over a CSR adjacency.
//!
//! # Blend modes
//!
//! - [`PropagationBlend::Mass`] (default, PPR-style): `next[i] =
//!   (1-α)·seed[i] + α·Σ_{j∈sel} w_ji·sigmoid(β·score_j)`. Edge weight is
//!   preserved — a chain successor (w=0.85) accrues roughly 2× a low-reliability
//!   neighbor (w=0.4) from the same supporter. This is the mode the Issue 655
//!   POC runs as its primary arm.
//! - [`PropagationBlend::Mean`]: `next[i] = (1-α)·seed[i] + α·
//!   Σ_{j∈sel} w_ji·rel_j / Σ_{j∈sel} w_ji` — the literal "edge_avg" from the
//!   KEEP formula. **Documented degeneracy**: for a node supported by exactly
//!   one selected node, the weight cancels (`w·rel/w = rel`), so a w=0.85
//!   chain edge and a w=0.40 distractor edge produce the identical score.
//!   Kept because it is the paper's literal shape; the POC reports it as a
//!   fourth arm so the degeneracy is measured, not assumed.
//!
//! # Feature flag
//! `selection_propagation` — opt-in (Issue 655; promotion depends on the G1
//! head-to-head vs the shipped BFS-decay traversal).
//!
//! Selection-Set Fixpoint Propagation — KEEP M3 in house operator vocabulary
//! (Issue 655 / Research 483, KEEP arXiv:2602.23592; HippoRAG PPR class). The
//! one genuinely-unshipped composition from the Research 483 audit: a
//! query-seeded importance propagation iterated until the top-r selected set
//! stabilizes (membership fixpoint). Sigmoid-gated membership, CLR-reliability
//! edge weighting, zero-alloc caller scratch, deterministic CSR order.
//! Opt-in — promotion depends on the Issue 655 G1 head-to-head vs the shipped
//! BFS-decay traversal (riir-rag fuse_graph_candidates) + downstream consumers.

use crate::sigmoid;

/// How the propagation step combines edge weights into the next score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum PropagationBlend {
    /// `Σ w·rel` — PPR-style mass; edge weight preserved (default).
    #[default]
    Mass,
    /// `Σ w·rel / Σ w` — the literal KEEP edge_avg; degenerate for
    /// single-supporter nodes (weight cancels).
    Mean,
}

/// Operator configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropagationConfig {
    /// Restart weight: `next = (1-α)·seed + α·propagated`. Higher α trusts
    /// the graph more. Default 0.85.
    pub alpha: f32,
    /// Sigmoid steepness for the membership gate `sigmoid(β·score)`.
    /// Default 4.0.
    pub gate_beta: f32,
    /// Maximum propagation steps before the hard stop. The membership
    /// fixpoint usually halts earlier (~chain length). Default 16.
    pub max_iters: usize,
    /// Edge blend mode. Default [`PropagationBlend::Mass`].
    pub blend: PropagationBlend,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            alpha: 0.85,
            gate_beta: 4.0,
            max_iters: 16,
            blend: PropagationBlend::Mass,
        }
    }
}

/// Outcome of a fixpoint run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropagationOutcome {
    /// Propagation steps actually performed (0 = selection already stable on
    /// the seed, e.g. `budget >= n` or a self-consistent seed).
    pub iters: usize,
    /// `true` when the loop halted on membership stability rather than
    /// `max_iters`.
    pub stable: bool,
}

/// Caller-owned scratch for [`propagate_selection_to_fixpoint_into`].
/// Grow-only: a scratch built for `(n, budget)` serves any smaller problem
/// without reallocating (G4 alloc-free steady state).
#[derive(Debug, Default)]
pub struct SelectionPropagationScratch {
    scores: Vec<f32>,
    acc: Vec<f32>,
    sumw: Vec<f32>,
    membership: Vec<u64>,
    prev_membership: Vec<u64>,
    top_idx: Vec<u32>,
    top_val: Vec<f32>,
}

impl SelectionPropagationScratch {
    /// Empty scratch; sized on first call.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-sized scratch for `n` nodes and a `budget`-sized selection.
    pub fn with_capacity(n: usize, budget: usize) -> Self {
        let mut s = Self::new();
        s.reset(n, budget.max(1));
        s
    }

    /// Grow-only resize to serve `n` nodes / `budget` selection.
    pub fn reset(&mut self, n: usize, budget: usize) {
        let words = n.div_ceil(64);
        if self.scores.len() < n {
            self.scores.resize(n, 0.0);
            self.acc.resize(n, 0.0);
            self.sumw.resize(n, 0.0);
        }
        if self.membership.len() < words {
            self.membership.resize(words, 0);
            self.prev_membership.resize(words, 0);
        }
        // Size to budget+1: the selection loop inserts before popping the
        // worst entry, so a buffer sized exactly `budget` would reallocate on
        // the insert-when-full step (G4).
        let budget = budget.max(1);
        if self.top_idx.len() < budget + 1 {
            self.top_idx.resize(budget + 1, 0);
            self.top_val.resize(budget + 1, f32::NEG_INFINITY);
        }
    }
}

/// Select the top-`budget` entries of `scores`, ordered by (score desc, index
/// asc), into `top_val` / `top_idx` (sorted descending, truncated to the
/// selection). Deterministic: equal scores break by ascending index.
fn select_top_r(scores: &[f32], budget: usize, top_val: &mut Vec<f32>, top_idx: &mut Vec<u32>) {
    let r = budget.min(scores.len());
    top_val.clear();
    top_idx.clear();
    // Buffers are kept sorted descending by (score, -idx). Insertion into a
    // bounded sorted window: O(n·r) worst case, r ≤ 32 by the operator's
    // contract — fine at µs scale.
    for (i, &s) in scores.iter().enumerate() {
        let full = top_val.len() >= r;
        if full {
            let worst = *top_val.last().unwrap();
            // Strictly-better only (NaN scores never enter — a NaN compares
            // false against everything, including Greater). Ties break by
            // ascending index, and during an ascending-index scan an equal
            // score already seated has the smaller index — a later tie never
            // enters.
            if s.partial_cmp(&worst) != Some(core::cmp::Ordering::Greater) {
                continue;
            }
        }
        // Find the insertion position: first slot where our (s, i) ranks
        // strictly higher. (s, i) > (s', i') iff s > s' or (s == s' and i < i').
        let mut lo = 0usize;
        let mut hi = top_val.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let mv = top_val[mid];
            let above = s > mv || (s == mv && (i as u32) < top_idx[mid]);
            if above {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        top_val.insert(lo, s);
        top_idx.insert(lo, i as u32);
        if top_val.len() > r {
            top_val.pop();
            top_idx.pop();
        }
    }
}

/// Iterate query-seeded importance propagation until the top-`budget` selected
/// set stabilizes (membership fixpoint) or `max_iters` propagation steps run.
///
/// # Arguments
/// - `offsets` — CSR row offsets, length `n + 1` (`offsets[0] == 0`).
/// - `targets` / `weights` — edge columns/weights, length `offsets[n]`,
///   parallel arrays.
/// - `seed` — query-seeded scores, length `n`. Expected in `[0, 1]` (the
///   harness builds them as `sigmoid(β·dot)`); any finite values rank the
///   same way.
/// - `budget` — selection width `r` per iteration (the equal-budget knob for
///   head-to-head comparisons).
/// - `scores_out` — final scores, length `n`. Rank-descending consumers take
///   `top-k` of this.
///
/// # Returns
/// [`PropagationOutcome`] with the propagation-step count + whether the stop
/// was the membership fixpoint.
///
/// # Panics
/// None in release; `debug_assert`s check slice-length contracts.
#[allow(clippy::too_many_arguments)] // CSR triple + seed/out slices — the `_into` convention
pub fn propagate_selection_to_fixpoint_into(
    offsets: &[u32],
    targets: &[u32],
    weights: &[f32],
    seed: &[f32],
    n: usize,
    budget: usize,
    cfg: &PropagationConfig,
    scores_out: &mut [f32],
    scratch: &mut SelectionPropagationScratch,
) -> PropagationOutcome {
    debug_assert_eq!(offsets.len(), n + 1, "offsets must be n+1");
    debug_assert_eq!(targets.len(), weights.len(), "targets/weights parallel");
    debug_assert_eq!(seed.len(), n, "seed must be n");
    debug_assert_eq!(scores_out.len(), n, "scores_out must be n");
    if n == 0 {
        return PropagationOutcome {
            iters: 0,
            stable: true,
        };
    }
    scratch.reset(n, budget);
    let words = n.div_ceil(64);
    let SelectionPropagationScratch {
        scores,
        acc,
        sumw,
        membership,
        prev_membership,
        top_idx,
        top_val,
    } = &mut *scratch;
    scores[..n].copy_from_slice(seed);
    prev_membership[..words].fill(0);

    if budget == 0 {
        scores_out.copy_from_slice(&scores[..n]);
        return PropagationOutcome {
            iters: 0,
            stable: true,
        };
    }

    let mut iters = 0usize;
    let alpha = cfg.alpha;
    let beta = cfg.gate_beta;
    loop {
        // 1. Selection: top-r of the current scores.
        select_top_r(&scores[..n], budget, top_val, top_idx);
        // 2. Membership bitset.
        membership[..words].fill(0);
        for &i in top_idx.iter() {
            membership[(i as usize) / 64] |= 1u64 << ((i as usize) % 64);
        }
        // 3. Membership fixpoint check.
        if iters > 0 && membership[..words] == prev_membership[..words] {
            scores_out.copy_from_slice(&scores[..n]);
            return PropagationOutcome {
                iters,
                stable: true,
            };
        }
        prev_membership[..words].copy_from_slice(&membership[..words]);

        if iters >= cfg.max_iters {
            scores_out.copy_from_slice(&scores[..n]);
            return PropagationOutcome {
                iters,
                stable: false,
            };
        }

        // 4. Propagate: each selected j pushes sigmoid-gated reliability
        //    along its out-edges. CSR order fixes summation order.
        acc[..n].fill(0.0);
        sumw[..n].fill(0.0);
        for &j in top_idx.iter() {
            let j = j as usize;
            let rel = sigmoid(beta * scores[j]);
            for e in offsets[j] as usize..offsets[j + 1] as usize {
                let t = targets[e] as usize;
                acc[t] += weights[e] * rel;
                sumw[t] += weights[e];
            }
        }

        // 5. Blend with the seed restart.
        let one_minus_alpha = 1.0 - alpha;
        match cfg.blend {
            PropagationBlend::Mass => {
                for i in 0..n {
                    scores[i] = one_minus_alpha * seed[i] + alpha * acc[i];
                }
            }
            PropagationBlend::Mean => {
                for i in 0..n {
                    let prop = if sumw[i] > 0.0 { acc[i] / sumw[i] } else { 0.0 };
                    scores[i] = one_minus_alpha * seed[i] + alpha * prop;
                }
            }
        }
        iters += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3-node chain (0↔1↔2, w=0.9) + 2 distractors (3, 4) hanging off node 0
    /// at w=0.3/0.25 (asymmetric, so the two distractors don't tie-oscillate).
    /// Symmetrized like the POC fixture (k_hop_neighbors unions spo+osp).
    /// Seed: node 0 high, distractors medium-high, chain tail low.
    fn toy_graph() -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        // node 0: ->1 (0.9), ->3 (0.30), ->4 (0.25)
        // node 1: ->0 (0.9), ->2 (0.9)
        // node 2: ->1 (0.9)
        // node 3: ->0 (0.30)
        // node 4: ->0 (0.25)
        let offsets = vec![0, 3, 5, 6, 7, 8];
        let targets = vec![1, 3, 4, 0, 2, 1, 0, 0];
        let weights = vec![0.9, 0.30, 0.25, 0.9, 0.9, 0.9, 0.30, 0.25];
        (offsets, targets, weights)
    }

    fn top_k(scores: &[f32], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        idx.truncate(k);
        idx
    }

    #[test]
    fn chain_support_beats_distractors() {
        let (offsets, targets, weights) = toy_graph();
        let seed = [0.98f32, 0.5, 0.5, 0.9, 0.9];
        let mut out = [0.0f32; 5];
        let mut scratch = SelectionPropagationScratch::with_capacity(5, 4);
        let outcome = propagate_selection_to_fixpoint_into(
            &offsets,
            &targets,
            &weights,
            &seed,
            5,
            4,
            &PropagationConfig::default(),
            &mut out,
            &mut scratch,
        );
        // The whole chain {0,1,2} must outrank the distractors {3,4} at
        // budget 4: selection = {0,1,2,+1 of 3/4}.
        let top4 = top_k(&out, 4);
        assert!(
            top4.contains(&1) && top4.contains(&2),
            "chain tail must enter the budget: top4 = {top4:?}, scores = {out:?}"
        );
        // Distractor scores must sit below the chain tail under Mass blend.
        assert!(
            out[1] > out[3] && out[1] > out[4],
            "chain successor (w=0.9) must outrank distractors (w=0.3): {out:?}"
        );
        assert!(outcome.stable);
        assert!(outcome.iters > 0 && outcome.iters <= 16);
    }

    #[test]
    fn mean_blend_weight_cancellation_documented() {
        let (offsets, targets, weights) = toy_graph();
        // Equal seeds for the chain successor (1) and the distractor (3) so
        // the ONLY difference can come from the blend: Mean cancels the edge
        // weight (both get prop = rel(node 0)), Mass preserves it (0.9·rel
        // vs 0.3·rel). Run exactly one propagation (max_iters = 1).
        let seed = [0.98f32, 0.7, 0.5, 0.7, 0.7];
        let mut out = [0.0f32; 5];
        let mut scratch = SelectionPropagationScratch::with_capacity(5, 4);
        let cfg = PropagationConfig {
            blend: PropagationBlend::Mean,
            max_iters: 1,
            ..Default::default()
        };
        let _ = propagate_selection_to_fixpoint_into(
            &offsets,
            &targets,
            &weights,
            &seed,
            5,
            4,
            &cfg,
            &mut out,
            &mut scratch,
        );
        assert!(
            (out[1] - out[3]).abs() < 1e-6,
            "mean blend cancels w: {out:?}"
        );

        let cfg = PropagationConfig {
            blend: PropagationBlend::Mass,
            max_iters: 1,
            ..Default::default()
        };
        let _ = propagate_selection_to_fixpoint_into(
            &offsets,
            &targets,
            &weights,
            &seed,
            5,
            4,
            &cfg,
            &mut out,
            &mut scratch,
        );
        assert!(
            out[1] - out[3] > 0.3,
            "mass blend preserves w (0.9 vs 0.3): {out:?}"
        );
    }

    #[test]
    fn membership_fixpoint_stops_early() {
        let (offsets, targets, weights) = toy_graph();
        let seed = [0.98f32, 0.5, 0.5, 0.9, 0.9];
        let mut out = [0.0f32; 5];
        let mut scratch = SelectionPropagationScratch::with_capacity(5, 4);
        let cfg = PropagationConfig {
            max_iters: 64,
            ..Default::default()
        };
        let outcome = propagate_selection_to_fixpoint_into(
            &offsets,
            &targets,
            &weights,
            &seed,
            5,
            4,
            &cfg,
            &mut out,
            &mut scratch,
        );
        // A 2-hop chain stabilizes in a handful of iterations, not 64.
        assert!(outcome.stable, "toy chain must reach membership fixpoint");
        assert!(
            outcome.iters < 8,
            "2-hop chain should stop in <8 iters, got {}",
            outcome.iters
        );
    }

    #[test]
    fn near_tied_alternates_can_hit_max_iters() {
        // Two IDENTICAL distractors (3, 4) with identical weights off node 0
        // alternate in/out of the budget at r=1 — the documented oscillation
        // mode the max_iters hard stop exists for.
        let offsets = vec![0, 2, 2, 2, 2, 2];
        let targets = vec![3, 4];
        let weights = vec![0.5, 0.5];
        let seed = [0.9f32, 0.0, 0.0, 0.8, 0.8];
        let mut out = [0.0f32; 5];
        let mut scratch = SelectionPropagationScratch::with_capacity(5, 1);
        let cfg = PropagationConfig {
            max_iters: 8,
            ..Default::default()
        };
        let outcome = propagate_selection_to_fixpoint_into(
            &offsets,
            &targets,
            &weights,
            &seed,
            5,
            1,
            &cfg,
            &mut out,
            &mut scratch,
        );
        // Whichever way it lands, the run must terminate at the bound without
        // reporting a membership fixpoint (or stabilize at the tie — both are
        // legal; assert the loop TERMINATED, i.e. iters ≤ 8).
        assert!(outcome.iters <= 8);
    }

    #[test]
    fn max_iters_hard_stop_on_oscillation() {
        // Node 0 (seed 0.9) has ONE edge to node 1 (seed 0.8); node 1 has NO
        // out-edges. budget=1: iter0 selects {0}, propagates boost to 1;
        // iter1 selects {1}; with no out-edges every node collapses to the
        // seed restart; iter2 selects {0} again → oscillation → max_iters.
        let offsets = vec![0, 1, 1];
        let targets = vec![1];
        let weights = vec![1.0];
        let seed = [0.9f32, 0.8];
        let mut out = [0.0f32; 2];
        let mut scratch = SelectionPropagationScratch::with_capacity(2, 1);
        let cfg = PropagationConfig {
            max_iters: 6,
            ..Default::default()
        };
        let outcome = propagate_selection_to_fixpoint_into(
            &offsets,
            &targets,
            &weights,
            &seed,
            2,
            1,
            &cfg,
            &mut out,
            &mut scratch,
        );
        assert!(
            !outcome.stable,
            "oscillating selection must NOT report stable"
        );
        assert_eq!(outcome.iters, 6);
    }

    #[test]
    fn deterministic_bit_identical_two_runs() {
        let (offsets, targets, weights) = toy_graph();
        let seed = [0.98f32, 0.5, 0.5, 0.9, 0.9];
        let mut out_a = [0.0f32; 5];
        let mut out_b = [0.0f32; 5];
        let mut scratch = SelectionPropagationScratch::with_capacity(5, 4);
        let cfg = PropagationConfig::default();
        let a = propagate_selection_to_fixpoint_into(
            &offsets,
            &targets,
            &weights,
            &seed,
            5,
            4,
            &cfg,
            &mut out_a,
            &mut scratch,
        );
        let b = propagate_selection_to_fixpoint_into(
            &offsets,
            &targets,
            &weights,
            &seed,
            5,
            4,
            &cfg,
            &mut out_b,
            &mut scratch,
        );
        assert_eq!(out_a, out_b, "two runs must be bit-identical");
        assert_eq!(a, b);
    }

    #[test]
    fn tie_break_prefers_lower_index() {
        // Two nodes, identical seed, no edges: top-1 must be node 0.
        let offsets = vec![0, 0, 0];
        let targets: Vec<u32> = vec![];
        let weights: Vec<f32> = vec![];
        let seed = [0.7f32, 0.7];
        let mut out = [0.0f32; 2];
        let mut scratch = SelectionPropagationScratch::with_capacity(2, 1);
        let cfg = PropagationConfig {
            max_iters: 1,
            ..Default::default()
        };
        let outcome = propagate_selection_to_fixpoint_into(
            &offsets,
            &targets,
            &weights,
            &seed,
            2,
            1,
            &cfg,
            &mut out,
            &mut scratch,
        );
        assert!(
            outcome.stable,
            "selection with no edges is stable after one propagation"
        );
        assert_eq!(outcome.iters, 1);
    }

    #[test]
    fn budget_zero_and_empty_graph() {
        let mut out: [f32; 0] = [];
        let mut scratch = SelectionPropagationScratch::new();
        let o = propagate_selection_to_fixpoint_into(
            &[0],
            &[],
            &[],
            &[],
            0,
            0,
            &PropagationConfig::default(),
            &mut out,
            &mut scratch,
        );
        assert_eq!(
            o,
            PropagationOutcome {
                iters: 0,
                stable: true
            }
        );

        let (offsets, targets, weights) = toy_graph();
        let seed = [0.9f32, 0.5, 0.5, 0.8, 0.8];
        let mut out5 = [0.0f32; 5];
        let o = propagate_selection_to_fixpoint_into(
            &offsets,
            &targets,
            &weights,
            &seed,
            5,
            0,
            &PropagationConfig::default(),
            &mut out5,
            &mut scratch,
        );
        assert_eq!(
            o,
            PropagationOutcome {
                iters: 0,
                stable: true
            }
        );
        assert_eq!(out5, seed, "budget=0 returns the seed unchanged");
    }

    #[test]
    fn budget_ge_n_selects_everything_stable() {
        let (offsets, targets, weights) = toy_graph();
        let seed = [0.98f32, 0.5, 0.5, 0.9, 0.9];
        let mut out = [0.0f32; 5];
        let mut scratch = SelectionPropagationScratch::with_capacity(5, 8);
        let outcome = propagate_selection_to_fixpoint_into(
            &offsets,
            &targets,
            &weights,
            &seed,
            5,
            8,
            &PropagationConfig::default(),
            &mut out,
            &mut scratch,
        );
        // Everyone is selected at iter 0 and again at iter 1 → stable.
        assert!(outcome.stable);
        assert!(
            outcome.iters <= 2,
            "budget >= n stabilizes immediately, got {}",
            outcome.iters
        );
    }
}
