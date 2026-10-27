//! Pair-scored path selection — the modelless DFlash 2 selector (Issue 671,
//! Research 490).
//!
//! DFlash 2 (Inco AI, 2026-08-18) lifts parallel block drafting by scoring
//! adjacent candidate PAIRS instead of trusting per-position argmax picks:
//! `S_t(a,b) = U_t(b) + ⟨A(a) ⊙ H(h_t), B(b)⟩` — a trained rank-256 bilinear
//! coherence term added to the drafter's own per-position logit, walked
//! greedily from the last verified token. "Choosing is cheaper than
//! predicting": +2.0M params / +0.6% latency beats DSpark's +77.8M / +9.6%
//! sequential correction.
//!
//! # Modelless substitution (the Issue 659 precedent)
//!
//! This module composes two signals that already ship, without training:
//!
//! - **`U_t(b)`** — per-position marginal evidence from **t-step forward
//!   propagation** of the [`BigramMarkovTable`] from the last verified token:
//!   `U_t = P(x_t | x_0 = prev)`. Each position's marginal is conditioned on
//!   the verified prefix ONLY — independent of sibling draft picks, exactly
//!   the DFlash parallel-drafter contract ( [`tstep_marginals_into`] ).
//! - **`log P(b|a)`** — adjacent-pair coherence from the same table's
//!   transition rows (the deterministic stand-in for the trained `A·H·B`
//!   bilinear; a full sparse table where DFlash 2 uses rank-256).
//!
//! Score: `S_t(a,b) = ln U_t(b) + λ_t · ln P(b|a)` with `a` the previous
//! PICKED token (the walk's own prefix — this is what the marginals lost and
//! the pair term restores). The walk is greedy best-successor over each
//! position's top-m lattice ([`pair_scored_path_into`]).
//!
//! # The entropy gate (`λ_t`, the `H(h_t)` analog)
//!
//! `λ_t = λ₀ · σ(κ · H_t)` ([`PairGateKind::Entropy`]) — HIGH-entropy
//! positions get a LARGER pair weight: the marginal is diluted (Research 407
//! §1.1's ceiling insight — marginals average over prefixes), so lean on
//! coherence; low-entropy positions trust `U`. NOTE: Issue 671's task text
//! wrote `σ(−κ·H)` — the sign here is set to match Research 490 §2.3-2's
//! stated intent ("high-entropy positions trust the pair term more"), which
//! the literal formula contradicts. T8 measures whether the gate carries
//! signal at all (vs [`PairGateKind::Flat`]); the margin variant
//! ([`PairGateKind::Margin`], `λ₀ · σ(−κ · (p₁ − p₂))`) keeps the
//! ambiguous→coherence direction without the sign question.
//!
//! # Semantics inherited from the table
//!
//! - **Zero rows**: a `prev` with no bigram evidence produces empty
//!   marginals and an empty path — the honest-drafter discipline of
//!   [`crate::bigram_markov`] (propose nothing when there is no information).
//! - **Truncation**: each propagation step keeps the top-m candidates by
//!   accumulated mass; tail mass is dropped, never renormalised (stored mass
//!   ≤ 1.0, mirroring the table's own top-m discipline).
//! - **Missing pairs**: `P(b|a) = 0` (absent from `a`'s top-m row) scores at
//!   [`MISSING_PAIR_LOGP`] — strongly disfavored but beatable by dominant `U`
//!   at small λ (DFlash 2's bilinear scores every pair; a sparse table must
//!   floor).
//! - **Ties**: marginals sort by `(mass desc, token asc)`; the walk takes the
//!   first strict max — deterministic end to end, and `λ = 0` reduces
//!   bit-identically to argmax-of-marginals (the G1 property, Issue 671 T5).
//!
//! Zero-alloc steady state: propagation reuses [`PairSelectScratch`]; the
//! walk writes into a caller-owned path buffer. Cost per draft cycle:
//! O(depth · m²) table reads + O(depth · m) scoring.
//!
//! Feature-gated behind `bigram_markov` with the table — opt-in until the
//! Issue 671 offline gate proves the gain.

use crate::bigram_markov::BigramMarkovTable;
use katgpt_core::simd::fast_sigmoid;

/// `ln P` substitute when the pair `(a, b)` has no evidence in the table
/// (`P(b|a) = 0`). ln(1e-9) ≈ −20.723. Chosen so a λ·floor penalty at
/// λ ∈ [0.5, 2] dominates any realistic `U` gap — absent pairs are
/// disfavoured, not forbidden (DFlash 2's bilinear has no absent case; a
/// sparse table must floor somewhere).
pub const MISSING_PAIR_LOGP: f32 = -20.723_272;

/// How `λ_t` is derived per position (Issue 671 T2 / T8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairGateKind {
    /// `λ_t = λ₀` — constant blend (the T8 control arm).
    Flat,
    /// `λ_t = λ₀ · σ(κ · H_t)` — high-entropy (diluted) marginals lean on
    /// the pair term. The modelless `H(h_t)` analog.
    Entropy,
    /// `λ_t = λ₀ · σ(−κ · (p₁ − p₂))` — ambiguous top-2 leans on the pair
    /// term. Entropy-free variant.
    Margin,
}

/// Selector configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct PairSelectConfig {
    /// Base pair-term weight `λ₀`. `0` reduces to argmax-of-marginals;
    /// large values reduce to bigram-greedy *within the lattice*.
    pub lambda0: f32,
    /// Gate steepness `κ` for the [`Entropy`](PairGateKind::Entropy) and
    /// [`Margin`](PairGateKind::Margin) gates.
    pub kappa: f32,
    pub gate: PairGateKind,
}

impl Default for PairSelectConfig {
    fn default() -> Self {
        Self {
            lambda0: 1.0,
            kappa: 2.0,
            gate: PairGateKind::Flat,
        }
    }
}

/// Reused buffers for marginal propagation (zero-alloc steady state).
#[derive(Debug, Default)]
pub struct PairSelectScratch {
    /// `positions[t]` = top-m `(token, mass)` at step `t+1`, sorted
    /// `(mass desc, token asc)`. Mass = truncated `P(x_{t+1} | prev)`.
    positions: Vec<Vec<(u32, f32)>>,
    /// Per-step accumulation buffer (pre-dedup, ≤ m² entries).
    acc: Vec<(u32, f32)>,
    /// Current frontier (top-m of the previous step; starts `[(prev, 1.0)]`).
    cur: Vec<(u32, f32)>,
}

impl PairSelectScratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-reserve for `depth` positions of `m` candidates.
    pub fn with_capacity(depth: usize, m: usize) -> Self {
        let cap = m.saturating_mul(m);
        Self {
            positions: vec![Vec::with_capacity(m); depth],
            acc: Vec::with_capacity(cap),
            cur: Vec::with_capacity(m + 1),
        }
    }
}

/// Per-position `λ_t` from the config gate (Issue 671 T2).
///
/// Exposed for measurement (T8 compares gate kinds against [`PairGateKind::Flat`]).
/// `candidates` is the position's `(token, mass)` list as produced by
/// [`tstep_marginals_into`].
pub fn gate_lambda(cfg: &PairSelectConfig, candidates: &[(u32, f32)]) -> f32 {
    match cfg.gate {
        PairGateKind::Flat => cfg.lambda0,
        PairGateKind::Entropy => {
            // Shannon entropy of the top-m renormalised distribution.
            let total: f32 = candidates.iter().map(|&(_, p)| p).sum();
            if total <= 0.0 {
                return cfg.lambda0;
            }
            let h: f32 = candidates
                .iter()
                .map(|&(_, p)| {
                    let q = p / total;
                    if q > 0.0 { -q * q.ln() } else { 0.0 }
                })
                .sum();
            cfg.lambda0 * fast_sigmoid(cfg.kappa * h)
        }
        PairGateKind::Margin => {
            if candidates.len() < 2 {
                return cfg.lambda0;
            }
            // Candidates are mass-desc sorted; p1 − p2.
            let margin = candidates[0].1 - candidates[1].1;
            cfg.lambda0 * fast_sigmoid(-cfg.kappa * margin)
        }
    }
}

/// Forward-propagate the table `depth` steps from `prev`, keeping the top-`m`
/// candidates per position — the modelless parallel drafter `U_t`.
///
/// `positions[t]` = `P(x_{t+1} | x_0 = prev)` restricted to its top-m support,
/// sorted `(mass desc, token asc)`. A `prev` with no bigram evidence yields
/// `depth` empty positions (zero-row semantics). Tail mass truncated per
/// step is dropped, never renormalised.
///
/// Deterministic: same table + inputs → bit-identical output.
pub fn tstep_marginals_into<'s>(
    table: &BigramMarkovTable,
    prev: u32,
    depth: usize,
    m: usize,
    scratch: &'s mut PairSelectScratch,
) -> &'s [Vec<(u32, f32)>] {
    scratch.positions.clear();
    scratch.positions.resize_with(depth, Vec::new);
    if depth == 0 {
        return &scratch.positions;
    }

    scratch.cur.clear();
    scratch.cur.push((prev, 1.0));

    for step in 0..depth {
        scratch.acc.clear();
        for &(a, w) in scratch.cur.iter() {
            if let Some((succs, probs)) = table.successors(a) {
                for (&b, &p) in succs.iter().zip(probs.iter()) {
                    scratch.acc.push((b, w * p));
                }
            }
        }
        if scratch.acc.is_empty() {
            // Zero row: this and all deeper positions stay empty.
            break;
        }
        // Merge by token (deterministic), then rank by (mass desc, token asc).
        scratch.acc.sort_unstable_by_key(|&(t, _)| t);
        let mut write = 0;
        for read in 0..scratch.acc.len() {
            let (tok, mass) = scratch.acc[read];
            if write > 0 && scratch.acc[write - 1].0 == tok {
                scratch.acc[write - 1].1 += mass;
            } else {
                scratch.acc[write] = (tok, mass);
                write += 1;
            }
        }
        scratch.acc.truncate(write);
        scratch
            .acc
            .sort_unstable_by(|&(t1, m1), &(t2, m2)| m2.total_cmp(&m1).then(t1.cmp(&t2)));
        scratch.acc.truncate(m);

        scratch.positions[step] = scratch.acc.clone();
        // Continue propagation from the truncated frontier.
        scratch.cur.clear();
        scratch.cur.extend_from_slice(&scratch.acc);
    }

    &scratch.positions
}

/// The per-position selection core shared by both pair-scored walks
/// ([`pair_scored_path_into`] over the bigram-propagated lattice, and the
/// Issue 966 grid-reselection walk over the drafter's own top-K pools).
///
/// Picks `argmax_b [ ln U(b) + lambda · ln P(b|a) ]` over `cands` —
/// `(token, U)` pairs, ANY order (scoring is per-candidate, not rank
/// dependent). Missing pairs score at [`MISSING_PAIR_LOGP`]. First strict
/// max wins (deterministic; `lambda = 0` with `(U desc, token asc)`-sorted
/// candidates reduces bit-identically to argmax-of-marginals — the G1
/// property, Issue 671 T5 — because equal scores keep the earlier entry and
/// sorted order puts the marginal argmax first).
///
/// Empty `cands` returns `u32::MAX` (the caller stalls the walk; the walk
/// adapters treat that as their own empty-position signal).
#[inline]
pub fn pick_pair_scored(
    cands: &[(u32, f32)],
    prev: u32,
    table: &BigramMarkovTable,
    lambda: f32,
) -> u32 {
    if cands.is_empty() {
        return u32::MAX;
    }
    let mut best_tok = cands[0].0;
    let mut best_score = f32::NEG_INFINITY;
    for &(b, u) in cands.iter() {
        let u_logp = u.ln();
        let pair_p = table.probability(prev, b);
        let pair_logp = if pair_p > 0.0 {
            pair_p.ln()
        } else {
            MISSING_PAIR_LOGP
        };
        let score = u_logp + lambda * pair_logp;
        if score > best_score {
            best_score = score;
            best_tok = b;
        }
    }
    best_tok
}

/// Greedy pair-scored walk over the marginal lattice (Issue 671 T1/T3).
///
/// From `a = prev`, each position picks `argmax_b [ ln U_t(b) + λ_t · ln P(b|a) ]`
/// over the position's top-m candidates; `a` becomes the pick. Missing pairs
/// score at [`MISSING_PAIR_LOGP`]. First strict max wins (candidates are
/// mass-desc sorted → deterministic; `λ = 0` is bit-identical to
/// argmax-of-marginals). Empty position (zero row) stalls the walk, matching
/// the greedy-chain discipline.
pub fn pair_scored_path_into(
    table: &BigramMarkovTable,
    prev: u32,
    marginals: &[Vec<(u32, f32)>],
    cfg: &PairSelectConfig,
    path: &mut Vec<u32>,
) {
    path.clear();
    let mut a = prev;
    for pos in marginals {
        if pos.is_empty() {
            break;
        }
        let lambda = gate_lambda(cfg, pos);
        let best_tok = pick_pair_scored(pos, a, table, lambda);
        if best_tok == u32::MAX {
            break;
        }
        path.push(best_tok);
        a = best_tok;
    }
}

/// Allocating convenience wrapper over [`pair_scored_path_into`].
pub fn pair_scored_path(
    table: &BigramMarkovTable,
    prev: u32,
    marginals: &[Vec<(u32, f32)>],
    cfg: &PairSelectConfig,
) -> Vec<u32> {
    let mut path = Vec::with_capacity(marginals.len());
    pair_scored_path_into(table, prev, marginals, cfg, &mut path);
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bigram_markov::BigramMarkovBuilder;

    /// Toy table from explicit bigram counts: `add((prev, next) × n)`.
    fn table_from_pairs(
        pairs: &[(u32, u32, u32)],
        vocab: usize,
        top_m: usize,
    ) -> BigramMarkovTable {
        let mut b = BigramMarkovBuilder::new();
        for &(p, n, count) in pairs {
            for _ in 0..count {
                b.add_sequence(&[p, n]);
            }
        }
        b.build(vocab, top_m)
    }

    /// The divergent fixture: `0→{2 (×6), 1 (×4)}`, `1→3 (×10)`,
    /// `2→{5 (×3), 3 (×1)}`.
    ///
    /// - `U_1` ranks `2` (0.6) over `1` (0.4) → λ=0 path starts `[2, …]`.
    /// - `U_2`: `3` = 0.4·1.0 + 0.6·0.25 = 0.55; `5` = 0.6·0.75 = 0.45
    ///   → argmax-of-marginals picks `3` at depth 2.
    /// - But `P(3|2) = 0.25 < P(5|2) = 0.75`, so a pair-weighted walk picks
    ///   `5` — the two signals genuinely disagree (the selection headroom).
    fn divergent_table() -> BigramMarkovTable {
        table_from_pairs(
            &[(0, 2, 6), (0, 1, 4), (1, 3, 10), (2, 5, 3), (2, 3, 1)],
            8,
            8,
        )
    }

    #[test]
    fn lambda_zero_equals_argmax_of_marginals() {
        // G1 property (Issue 671 T5): λ=0 must reduce to the per-position
        // argmax of the marginals, bit-identical.
        let table = divergent_table();
        let mut scratch = PairSelectScratch::new();
        for prev in [0u32, 1, 2, 3] {
            let marginals = tstep_marginals_into(&table, prev, 6, 8, &mut scratch).to_vec();
            let cfg = PairSelectConfig {
                lambda0: 0.0,
                ..PairSelectConfig::default()
            };
            let path = pair_scored_path(&table, prev, &marginals, &cfg);
            let argmax: Vec<u32> = marginals
                .iter()
                .take_while(|p| !p.is_empty())
                .map(|p| p[0].0)
                .collect();
            assert_eq!(
                path, argmax,
                "λ=0 diverged from argmax-of-marginals at prev={prev}"
            );
        }
    }

    #[test]
    fn pair_term_flips_the_depth2_pick() {
        // The fixture's designed disagreement: λ=0 picks [2, 3]; a λ=1
        // pair-scored walk picks [2, 5] (coherence beats marginal rank).
        let table = divergent_table();
        let mut scratch = PairSelectScratch::new();
        let marginals = tstep_marginals_into(&table, 0, 2, 8, &mut scratch).to_vec();
        assert_eq!(marginals[0][0].0, 2, "U_1 should rank token 2 first");

        let flat0 = PairSelectConfig {
            lambda0: 0.0,
            ..PairSelectConfig::default()
        };
        assert_eq!(pair_scored_path(&table, 0, &marginals, &flat0), vec![2, 3]);

        let flat1 = PairSelectConfig {
            lambda0: 1.0,
            ..PairSelectConfig::default()
        };
        assert_eq!(pair_scored_path(&table, 0, &marginals, &flat1), vec![2, 5]);
    }

    #[test]
    fn huge_lambda_is_bigram_greedy_within_lattice() {
        // λ → ∞ reduces to argmax P(b|a) over the lattice candidates
        // (greedy-chain semantics restricted to the top-m support).
        let table = divergent_table();
        let mut scratch = PairSelectScratch::new();
        let marginals = tstep_marginals_into(&table, 0, 4, 8, &mut scratch).to_vec();
        let cfg = PairSelectConfig {
            lambda0: 1e9,
            ..PairSelectConfig::default()
        };
        let path = pair_scored_path(&table, 0, &marginals, &cfg);
        // Greedy chain from 0: successors(0)[0] = 2, successors(2)[0] = 5.
        assert_eq!(path, vec![2, 5]);
    }

    #[test]
    fn tstep_marginals_match_dense_propagation() {
        // No-truncation regime (m ≥ row length): the sparse propagation must
        // equal explicit dense matrix powers.
        let table = table_from_pairs(
            &[
                (0, 1, 3),
                (0, 2, 1),
                (1, 2, 2),
                (1, 3, 2),
                (2, 0, 1),
                (2, 3, 1),
            ],
            4,
            8,
        );
        // Dense transition matrix P[i][j] = probability(i, j).
        let mut dense = [[0.0f32; 4]; 4];
        for i in 0..4u32 {
            let total = table.row_total(i) as f32;
            if total > 0.0 {
                for j in 0..4u32 {
                    dense[i as usize][j as usize] = table.probability(i, j) / total.max(1.0);
                }
                // Row probabilities are count/row_total over the FULL row
                // (pre-truncation normaliser), so scale back to true mass.
                let mass: f32 = dense[i as usize].iter().sum();
                if mass > 0.0 {
                    for v in &mut dense[i as usize] {
                        *v /= mass;
                    }
                }
            }
        }
        let mut dist = [0.0f32; 4];
        dist[0] = 1.0;
        let mut scratch = PairSelectScratch::new();
        let marginals = tstep_marginals_into(&table, 0, 3, 8, &mut scratch);
        for (step, step_marginals) in marginals.iter().enumerate() {
            // Dense step.
            let mut next = [0.0f32; 4];
            for j in 0..4 {
                let mut acc = 0.0;
                for (i, &w) in dist.iter().enumerate() {
                    acc += w * dense[i][j];
                }
                next[j] = acc;
            }
            dist = next;
            // Compare (dense entries below a tiny floor may be truncated out
            // of the sparse row entirely — only compare tokens the table kept).
            for &(tok, mass) in step_marginals {
                assert!(
                    (mass - dist[tok as usize]).abs() < 1e-5,
                    "step {step}, token {tok}: sparse mass {mass} vs dense {}",
                    dist[tok as usize]
                );
            }
        }
    }

    #[test]
    fn entropy_gate_direction_high_entropy_trusts_pair() {
        // Research 490 §2.3-2: diluted (high-entropy) marginals → larger λ.
        let cfg = PairSelectConfig {
            lambda0: 1.0,
            kappa: 2.0,
            gate: PairGateKind::Entropy,
        };
        let confident = vec![(7u32, 0.98f32), (9, 0.02)];
        let ambiguous = vec![(7u32, 0.5f32), (9, 0.5)];
        let l_conf = gate_lambda(&cfg, &confident);
        let l_amb = gate_lambda(&cfg, &ambiguous);
        assert!(
            l_amb > l_conf,
            "ambiguous marginal should trust the pair term more: {l_amb} vs {l_conf}"
        );
    }

    #[test]
    fn margin_gate_direction_small_margin_trusts_pair() {
        let cfg = PairSelectConfig {
            lambda0: 1.0,
            kappa: 2.0,
            gate: PairGateKind::Margin,
        };
        let clear = vec![(7u32, 0.9f32), (9, 0.05)];
        let tight = vec![(7u32, 0.5f32), (9, 0.48)];
        assert!(gate_lambda(&cfg, &tight) > gate_lambda(&cfg, &clear));
    }

    #[test]
    fn missing_pair_floor_beatable_by_dominant_u_at_small_lambda() {
        // A lattice candidate absent from the predecessor's row can still win
        // when U dominates and λ is small — the floor disfavours, not forbids.
        let table = divergent_table();
        // Hand-built lattice: token 3 (U 0.999) is absent from row 4;
        // token 5 (U 1e-6) is present in row 4.
        let lattice = vec![vec![(3u32, 0.999f32), (5, 1e-6)]];
        let small = PairSelectConfig {
            lambda0: 0.1,
            ..PairSelectConfig::default()
        };
        // prev = 4 has no row in the fixture → every pair floors.
        assert_eq!(table.probability(4, 5), 0.0);
        let path = pair_scored_path(&table, 4, &lattice, &small);
        assert_eq!(
            path,
            vec![3],
            "dominant U must beat the floored pair at λ=0.1"
        );
    }

    #[test]
    fn zero_row_prev_yields_empty_marginals_and_path() {
        let table = divergent_table();
        let mut scratch = PairSelectScratch::new();
        let marginals = tstep_marginals_into(&table, 7, 4, 8, &mut scratch);
        assert!(marginals.iter().all(|p| p.is_empty()));
        let mut path = Vec::new();
        pair_scored_path_into(
            &table,
            7,
            marginals,
            &PairSelectConfig::default(),
            &mut path,
        );
        assert!(path.is_empty());
    }

    #[test]
    fn deterministic_across_runs() {
        let table = divergent_table();
        let cfg = PairSelectConfig::default();
        let mut s1 = PairSelectScratch::new();
        let mut s2 = PairSelectScratch::new();
        let m1 = tstep_marginals_into(&table, 0, 6, 8, &mut s1).to_vec();
        let m2 = tstep_marginals_into(&table, 0, 6, 8, &mut s2).to_vec();
        assert_eq!(m1, m2);
        assert_eq!(
            pair_scored_path(&table, 0, &m1, &cfg),
            pair_scored_path(&table, 0, &m2, &cfg)
        );
    }

    #[test]
    fn scratch_reuse_no_growth_steady_state() {
        // Repeated calls at fixed depth/m must not grow any buffer.
        let table = divergent_table();
        let mut scratch = PairSelectScratch::with_capacity(4, 8);
        let _ = tstep_marginals_into(&table, 0, 4, 8, &mut scratch);
        let cap_after_first = scratch.positions.capacity();
        for prev in [0u32, 1, 2, 0, 2, 1] {
            let _ = tstep_marginals_into(&table, prev, 4, 8, &mut scratch);
        }
        assert_eq!(scratch.positions.capacity(), cap_after_first);
    }
}
