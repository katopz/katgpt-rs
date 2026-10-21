//! T2 — search over bet COMPOSITIONS (Issue 952).
//!
//! A composition is a non-empty set of roster bets (a bitmask over the
//! roster, bit `i` = roster index `i`) whose prediction is the equal-weight
//! mean of its members' predictions. Its score is the leave-one-out
//! backtest CRPS on the SELECTION split. A point prediction is a point-mass
//! distribution, whose CRPS against `y` is exactly `|ŷ − y|` — so the
//! selection score is the LOO mean absolute error, stated in CRPS units
//! because that is the unit the floor gate compares in. The interval is
//! the calibration split's job (`super::BetSearchHead::calibrate`), never
//! the search's.
//!
//! # Cost model
//!
//! [`BetEvaluator`] caches each bet's LOO prediction column the first time
//! any composition needs it (`O(n · bet cost)` once per bet); a composition
//! score is then `O(n · |mask|)` additions over cached columns. The budget
//! the GOAT bench compares at is the number of DISTINCT compositions
//! scored ([`BetEvaluator::evaluations`]), identical for every searcher.
//!
//! # Searchers
//!
//! * `puct_search` (feature `bet_search_puct`) — the shipped `katgpt_core::chance_puct` searcher over a
//!   canonical composition tree (a node adds a roster bet with a HIGHER
//!   index than every member, so each set is reachable by exactly one
//!   path). Priors are the members' negated singleton scores; the value is
//!   the negated composition score; chance nodes are single-outcome
//!   (deterministic game). PUCT is used as an exploration policy: the pick
//!   is the best-scored composition it EVALUATED, not the root action.
//! * [`exhaustive_search`] — the equal-budget control (d): score the first
//!   `evaluations` compositions in canonical order (size ascending, then
//!   mask ascending).
//!
//! Both picks share one tie rule (Issue 940): lowest score, then the
//! LOWEST mask — independent of evaluation order, so two searchers that
//! evaluated the same set agree bit-for-bit.

use core::cell::RefCell;

#[cfg(feature = "bet_search_puct")]
use katgpt_core::chance_puct::{ChanceGame, ChancePuct, ChancePuctConfig, fastrand};

use super::bets::{AuthoredBet, Bet, BetScratch, CompsView};

/// Largest supported roster (a composition is a `u32` mask; the memo is
/// `2^roster` scores).
pub const MAX_ROSTER: usize = 12;

/// Score + provenance of one search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchOutcome {
    /// The picked composition (`0` ⇔ nothing scoreable).
    pub mask: u32,
    /// Its selection-split LOO CRPS (`+inf` when `mask == 0`).
    pub crps: f32,
    /// Distinct compositions scored.
    pub evaluations: u32,
    /// Bets whose LOO column was materialized.
    pub bets_materialized: u32,
}

struct Memo {
    /// `2^roster` composition scores; `NaN` = not yet scored.
    crps: Vec<f32>,
    /// Bet-major LOO prediction columns, `roster × n`.
    loo: Vec<f32>,
    materialized: u32,
    evaluations: u32,
    scratch: BetScratch,
}

/// Scores compositions over one selection corpus, memoized.
pub struct BetEvaluator<'a> {
    roster: &'a [AuthoredBet],
    comps: CompsView<'a>,
    max_depth: u8,
    memo: RefCell<Memo>,
}

impl<'a> BetEvaluator<'a> {
    /// An evaluator over `comps` (the SELECTION split) for `roster`
    /// (truncated to [`MAX_ROSTER`]), compositions of at most `max_depth`
    /// bets.
    pub fn new(roster: &'a [AuthoredBet], comps: CompsView<'a>, max_depth: u8) -> Self {
        let roster = &roster[..roster.len().min(MAX_ROSTER)];
        Self {
            roster,
            comps,
            max_depth: max_depth.max(1),
            memo: RefCell::new(Memo {
                crps: vec![f32::NAN; 1 << roster.len()],
                loo: vec![f32::NAN; roster.len() * comps.n],
                materialized: 0,
                evaluations: 0,
                scratch: BetScratch::new(comps.dim, comps.n),
            }),
        }
    }

    pub fn roster_len(&self) -> usize {
        self.roster.len()
    }

    pub fn max_depth(&self) -> u8 {
        self.max_depth
    }

    /// Distinct compositions scored so far.
    pub fn evaluations(&self) -> u32 {
        self.memo.borrow().evaluations
    }

    /// The selection-split LOO CRPS of `mask` (memoized; `+inf` for the
    /// empty mask or when any member bet cannot predict a row).
    pub fn crps(&self, mask: u32) -> f32 {
        if mask == 0 || self.comps.n < 2 {
            return f32::INFINITY;
        }
        let mut memo = self.memo.borrow_mut();
        let cached = memo.crps[mask as usize];
        if !cached.is_nan() {
            return cached;
        }
        let n = self.comps.n;
        for b in 0..self.roster.len() {
            if mask & (1 << b) != 0 && memo.materialized & (1 << b) == 0 {
                self.materialize(&mut memo, b);
            }
        }
        // Issue 940: sequential fold — rows ascending, members ascending.
        let members = mask.count_ones() as f32;
        let mut err = 0.0_f64;
        for i in 0..n {
            let mut s = 0.0_f32;
            for b in 0..self.roster.len() {
                if mask & (1 << b) != 0 {
                    s += memo.loo[b * n + i];
                }
            }
            err += f64::from((s / members - self.comps.values[i]).abs());
        }
        let score = (err / n as f64) as f32;
        let score = if score.is_finite() {
            score
        } else {
            f32::INFINITY
        };
        memo.crps[mask as usize] = score;
        memo.evaluations += 1;
        score
    }

    fn materialize(&self, memo: &mut Memo, b: usize) {
        let n = self.comps.n;
        let dim = self.comps.dim;
        let Memo { loo, scratch, .. } = memo;
        for i in 0..n {
            let query = &self.comps.raw[i * dim..i * dim + dim];
            loo[b * n + i] = self.roster[b].predict(&self.comps, Some(i), query, scratch);
        }
        memo.materialized |= 1 << b;
    }

    /// Best scored composition so far (lowest score, then lowest mask).
    pub fn best(&self) -> SearchOutcome {
        let memo = self.memo.borrow();
        let mut best = (0_u32, f32::INFINITY);
        for (mask, &s) in memo.crps.iter().enumerate() {
            if s.is_finite() && s < best.1 {
                best = (mask as u32, s);
            }
        }
        SearchOutcome {
            mask: best.0,
            crps: best.1,
            evaluations: memo.evaluations,
            bets_materialized: memo.materialized.count_ones(),
        }
    }
}

/// Every composition of at most `max_depth` members over a roster of
/// `roster_len`, in canonical order: size ascending, then mask ascending.
pub fn canonical_compositions(roster_len: usize, max_depth: u8) -> impl Iterator<Item = u32> {
    let roster_len = roster_len.min(MAX_ROSTER);
    (1..=u32::from(max_depth.max(1)))
        .flat_map(move |size| (1_u32..(1 << roster_len)).filter(move |m| m.count_ones() == size))
}

/// Control (d): score the first `evaluations` canonical compositions.
pub fn exhaustive_search(eval: &BetEvaluator<'_>, evaluations: usize) -> SearchOutcome {
    for mask in canonical_compositions(eval.roster_len(), eval.max_depth()).take(evaluations) {
        eval.crps(mask);
    }
    eval.best()
}

/// One node of the canonical composition tree.
#[cfg(feature = "bet_search_puct")]
#[derive(Clone, Copy)]
struct BetNode<'e, 'a> {
    eval: &'e BetEvaluator<'a>,
    mask: u32,
    /// Lowest roster index a child may add (canonical: strictly above every
    /// member).
    next: u8,
    /// `true` between an `apply` and its single-outcome `resolve`.
    chance: bool,
}

#[cfg(feature = "bet_search_puct")]
impl ChanceGame for BetNode<'_, '_> {
    type Action = u8;
    type Outcome = ();

    fn is_terminal(&self) -> bool {
        false
    }

    /// Higher = better: the negated composition score. The empty root reads
    /// the mean singleton score, so 0.5 after the squash means "as good as
    /// an average bet".
    fn value(&self) -> f32 {
        if self.mask != 0 {
            let s = self.eval.crps(self.mask);
            return if s.is_finite() { -s } else { f32::MIN };
        }
        let mut sum = 0.0_f32;
        let mut count = 0_u32;
        for b in 0..self.eval.roster_len() {
            let s = self.eval.crps(1 << b);
            if s.is_finite() {
                sum += s;
                count += 1;
            }
        }
        if count == 0 {
            f32::MIN
        } else {
            -sum / count as f32
        }
    }

    fn actions(&self, out: &mut Vec<(u8, f32)>) {
        if self.mask.count_ones() >= u32::from(self.eval.max_depth()) {
            return;
        }
        for b in usize::from(self.next)..self.eval.roster_len() {
            let s = self.eval.crps(1 << b);
            let prior = if s.is_finite() { -s } else { f32::MIN };
            out.push((b as u8, prior));
        }
    }

    fn apply(&self, action: u8) -> Self {
        Self {
            mask: self.mask | (1 << action),
            next: action + 1,
            chance: true,
            ..*self
        }
    }

    fn outcomes(&self, out: &mut Vec<((), f32)>) {
        if self.chance {
            out.push(((), 1.0));
        }
    }

    fn resolve(&self, _outcome: ()) -> Self {
        Self {
            chance: false,
            ..*self
        }
    }
}

#[cfg(feature = "bet_search_puct")]
/// PUCT search (budget = simulations; each expands at most one composition)
/// from the empty composition, deterministic for a fixed `seed`.
pub fn puct_search(eval: &BetEvaluator<'_>, cfg: ChancePuctConfig, seed: u64) -> SearchOutcome {
    let root = BetNode {
        eval,
        mask: 0,
        next: 0,
        chance: false,
    };
    let mut searcher = ChancePuct::new(cfg);
    let mut rng = fastrand::Rng::with_seed(seed);
    let _ = searcher.search(&root, &mut rng);
    eval.best()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bet_search::bets::{DEFAULT_BETS, normalize_rows};

    struct Corpus {
        raw: Vec<f32>,
        norm: Vec<f32>,
        values: Vec<f32>,
        scales: Vec<f32>,
    }

    /// value = 3 · x0 exactly: the ratio bet is perfect, everything else is
    /// not.
    fn ratio_corpus(n: usize) -> Corpus {
        let mut raw = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            let x0 = 1.0 + i as f32;
            let x1 = ((i * 7) % 5) as f32 + 1.0;
            raw.extend([x0, x1]);
            values.push(3.0 * x0);
        }
        let scales = vec![1.0, 1.0];
        let mut norm = Vec::new();
        normalize_rows(&raw, &scales, 2, &mut norm);
        Corpus {
            raw,
            norm,
            values,
            scales,
        }
    }

    fn view(c: &Corpus) -> CompsView<'_> {
        CompsView {
            raw: &c.raw,
            norm: &c.norm,
            values: &c.values,
            scales: &c.scales,
            dim: 2,
            n: c.values.len(),
        }
    }

    #[test]
    fn canonical_order_is_size_then_mask() {
        let v: Vec<u32> = canonical_compositions(3, 2).collect();
        assert_eq!(v, vec![0b001, 0b010, 0b100, 0b011, 0b101, 0b110]);
        assert_eq!(canonical_compositions(7, 3).count(), 7 + 21 + 35);
    }

    #[test]
    fn exhaustive_finds_the_exact_bet() {
        let c = ratio_corpus(12);
        let eval = BetEvaluator::new(&DEFAULT_BETS, view(&c), 3);
        let out = exhaustive_search(&eval, usize::MAX);
        assert_eq!(out.evaluations, 63);
        // Ratio scaling (roster index 4) alone is exact; two-feature ratio
        // (index 6) is also exact on a through-origin plane — the lowest
        // mask wins the tie.
        assert_eq!(out.mask, 1 << 4);
        assert!(out.crps < 1e-4, "{}", out.crps);
    }

    #[test]
    fn memo_counts_distinct_compositions_once() {
        let c = ratio_corpus(8);
        let eval = BetEvaluator::new(&DEFAULT_BETS, view(&c), 3);
        let a = eval.crps(0b11);
        let b = eval.crps(0b11);
        assert_eq!(a.to_bits(), b.to_bits());
        assert_eq!(eval.evaluations(), 1);
    }

    #[test]
    #[cfg(feature = "bet_search_puct")]
    fn puct_is_deterministic_and_finds_the_exact_bet() {
        let c = ratio_corpus(16);
        let cfg = ChancePuctConfig {
            budget: 32,
            ..ChancePuctConfig::default()
        };
        let e1 = BetEvaluator::new(&DEFAULT_BETS, view(&c), 3);
        let e2 = BetEvaluator::new(&DEFAULT_BETS, view(&c), 3);
        let a = puct_search(&e1, cfg, 7);
        let b = puct_search(&e2, cfg, 7);
        assert_eq!(a, b);
        assert!(a.crps < 1e-4, "{a:?}");
        assert!(a.evaluations <= 32 + DEFAULT_BETS.len() as u32);
    }

    #[test]
    fn too_few_rows_score_nothing() {
        let c = ratio_corpus(1);
        let eval = BetEvaluator::new(&DEFAULT_BETS, view(&c), 3);
        let out = exhaustive_search(&eval, usize::MAX);
        assert_eq!(out.mask, 0);
        assert!(out.crps.is_infinite());
    }
}
