//! LDT conflict-triggered targeted candidate reopening (Issue 956 T2).
//!
//! Distilled from the candidate-set binary-diffusion evidence law
//! (arXiv:2610.28543 via riir-train Research 480): restore-capable candidate
//! evolution beats prune-only search on hard instances. Our shipped lattice
//! (`build_dd_tree_screened`) is prune-only per pass with whole-node
//! backtracking; this module adds the paper's missing mechanism —
//! **per-position revisability** — as a bounded post-pass:
//!
//! 1. **Elimination-time memory.** The score a candidate had when screening
//!    killed it is replayed at the chain's own context (the score exists at
//!    elimination — `theta_elim ≈ 0.111` implies it). Exact for
//!    parent-independent screeners (the `MassScreener` LDT story);
//!    chain-context approximation for parent-dependent ones (disclosed).
//! 2. **Conflict trigger.** `EntropyConflictDetector::is_conflicted_at_depth`
//!    per depth (prune-rate over the replayed elimination counts; the
//!    detector's entropy half rides along unchanged).
//! 3. **CDCL synthesis for free.** The clause DB already learns WHERE the
//!    conflict was: each fired depth with a chain commitment learns the
//!    singleton clause `{(depth, chain_token)}`. Reopen candidates come from
//!    **clause-implicated positions**; the no-cycle invariant (never reopen
//!    a candidate a live clause excludes) is the singleton probe
//!    `is_violated(&[(d, t)])`.
//! 4. **β ranking.** Reopen the top eliminated candidates ranked `score^β`
//!    (ties: depth, then token — deterministic) — the CDCL phase-saving
//!    analog: restore the most-plausible-eliminated first. The per-round
//!    COUNT is T2b's: fixed `budget_b`, or (η mode) a capped-mean bisection
//!    over the pool's rank scores — propensity `σ(γ·(score − τ))` per
//!    candidate (never softmax), τ solved so the expected count is
//!    `η·headroom`, the deterministic read taking the count above the cut.
//!    Selection itself stays RNG-free (the determinism law); the σ-form is
//!    a smooth drive over the pool's score SHAPE, and `headroom` is the
//!    sigmoid over the measured budget tightness that bounds both the
//!    reopen/re-eliminate thrash and the score-ordered-pop displacement.
//! 5. **Revival semantics.** A reopened candidate re-enters with relevance
//!    1.0 — the screening penalty (`ln R`) is REMOVED, so it competes on its
//!    own LLM mass (`ln P`). The revision says the elimination was wrong,
//!    not that the relevance was higher. The budget-constrained best-first
//!    build then re-decides honestly (the paper's re-corruption step).
//! 6. **Termination (the FSM law applied to solver state).** Bounded rounds
//!    (`max_rounds`), a per-node cap (each candidate reopened at most once
//!    per call — the ledger), the no-cycle clause filter, and an
//!    empty-restoration break (a fired round with no admissible candidate
//!    cannot make progress). No prune/reopen oscillation is representable.
//!
//! `alpha_intersect` itself is untouched — the α-operator stays MONOTONE
//! per pass (the propagator law); this module composes WITH any
//! `ScreeningPruner`, including `AlphaScreeningPruner` (the α-lattice).
//!
//! All code behind `#[cfg(feature = "lattice_deduction")]`. Default-off:
//! `ReopenConfig::default()` has `budget_b = 0`, which IS the plain
//! screened build (byte-identity, G3) — a conflict can never fire because
//! the conflict machinery never runs.

use katgpt_core::simd::fast_sigmoid;
use katgpt_core::speculative::types::ConflictDetector;
use katgpt_core::speculative::types::{EntropyConflictDetector, TreeNode};
use katgpt_core::traits::ScreeningPruner;

use crate::alpha::ConflictClauseDB;
use crate::dd_tree::{TreeBuilder, build_dd_tree_screened};

/// One eliminated candidate, recorded by the elimination-time replay.
///
/// `score` is the relevance the candidate had at elimination (chain
/// context) — the LDT elimination story's own quantity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EliminationRecord {
    pub depth: usize,
    pub token_idx: usize,
    pub score: f32,
}

/// Conflict-triggered reopening configuration (Issue 956 T2 + T2b).
///
/// Default is the OFF posture: `budget_b = 0` runs the plain screened
/// build — byte-identical, and the conflict machinery never executes.
///
/// Two count modes, selected by `eta`:
///
/// - **Fixed-b (T2, `eta <= 0`).** Reopen the top-`budget_b` pool entries
///   every applied round. The measured posture of the T2 bench.
/// - **η-bounded (T2b, `eta > 0`).** `budget_b` demotes to the ABSOLUTE
///   per-round cap; the per-round count comes from a capped-mean bisection
///   over the pool's rank scores: each candidate's reopen propensity is the
///   sigmoid drive `σ(γ·(score − τ))` (never softmax), τ is bisected so the
///   expected count `Σ σ(γ·(sᵢ − τ)) = η·headroom`, and the deterministic
///   read takes the count above the cut. `headroom =
///   σ(headroom_gamma·(tree_budget/distinct_space − headroom_center))` is
///   the FSM-law sigmoid over the MEASURED tightness — the T2 bench's
///   regression boot sat at ratio ≈ 2.08 (budget 150, space V·D = 72) and
///   its gain boot at ≈ 6.94 (budget 500), so the shipped center 4.0 (the
///   geometric midpoint, rounded) and steepness 1.0 read ≈ 0.13 / ≈ 0.95
///   at the two measured boots (provisional calibration; re-pin on new
///   boots). A tight budget therefore spends its reopens (often zero — a
///   zero-count round breaks before re-propagating, the progress law) and
///   a roomy one recovers the full η — bounding both the reopen/re-eliminate
///   thrash AND the score-ordered-pop displacement T2 measured.
#[derive(Debug, Clone)]
pub struct ReopenConfig {
    /// Max reopens per conflict round — in η mode the ABSOLUTE cap the
    /// bisected count is clamped to. 0 disables the whole mechanism
    /// (G3 default-off).
    pub budget_b: usize,
    /// Rank exponent: candidates ranked by `score^β` descending. Values
    /// ≤ 0 or NaN clamp to 0.0 (all-1.0 ranks → deterministic pool order).
    /// β = 1 is the plain score order; larger β steepens toward the
    /// top-scored eliminations (the phase-saving analog). In the
    /// deterministic η mode the selected SET is the top-count by rank, so
    /// β > 0 leaves it unchanged (a monotone warp preserves the cut's
    /// cardinality); β matters for the fixed mode's order and any future
    /// propensity-sampled read.
    pub beta: f32,
    /// Hard cap on applied reopen rounds per call. Each round re-detects
    /// conflicts against the rebuilt tree; the cap plus the ledger plus the
    /// no-cycle filter is the termination proof (no oscillation).
    pub max_rounds: usize,
    /// The conflict detector (LDT θ_cls analog; Plan 170 F4 depth
    /// escalation rides inside).
    pub detector: EntropyConflictDetector,
    /// T2b expected reopens per round at full headroom (the bisection
    /// target before headroom scaling). `<= 0` or NaN selects the fixed-b
    /// mode; `> 0` selects the η-bounded mode (`budget_b` stays the
    /// absolute cap).
    pub eta: f32,
    /// T2b σ steepness over the pool's rank scores, in UNITS OF POOL
    /// SPACING: the effective slope is `γ·n/spread` (n = pool size, spread
    /// = score range), so the σ transition width is `spread/(γ·n)` — one
    /// mean order-statistic gap at γ = 1. A FIXED absolute γ cannot work
    /// across distributions (measured: the bench's ~60-candidate
    /// elimination pools at spread 0.11 strangled a γ = 20 slope to zero
    /// reopens everywhere — the σ tail mass scales with n). Non-finite or
    /// ≤ 0 clamps to the default.
    pub gamma: f32,
    /// T2b headroom-drive center: the tree_budget/distinct_space ratio the
    /// drive reads 0.5 at. Provisional calibration: the T2 boots measured
    /// regression at ≈ 2.08 and gain at ≈ 6.94; 4.0 is their geometric
    /// midpoint rounded.
    pub headroom_center: f32,
    /// T2b headroom-drive steepness over that ratio. Provisional: 1.0
    /// reads ≈ 0.13 / ≈ 0.95 at the two measured boots.
    pub headroom_gamma: f32,
}

impl Default for ReopenConfig {
    fn default() -> Self {
        Self {
            budget_b: 0,
            beta: 1.0,
            max_rounds: 1,
            detector: EntropyConflictDetector::default(),
            eta: 0.0,
            gamma: 20.0,
            headroom_center: 4.0,
            headroom_gamma: 1.0,
        }
    }
}

/// Counters for one `build_dd_tree_screened_reopen` call.
///
/// `rounds` is the reopen-cycles-per-solve counter the T2b anti-thrash
/// budget pins in the G1 record.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ReopenStats {
    /// Depth-conflict events across all rounds.
    pub conflicts_fired: usize,
    /// Candidates actually restored (applied rounds only).
    pub reopens_applied: usize,
    /// Pool candidates skipped by the no-cycle clause filter.
    pub no_cycle_blocked: usize,
    /// Applied reopen rounds (≤ `max_rounds` by construction).
    pub rounds: usize,
}

/// The revival wrapper: lifted candidates read relevance 1.0 (pure LLM
/// mass — the screening penalty removed), everything else passes through.
struct ReopenScreener<'a> {
    inner: &'a dyn ScreeningPruner,
    /// (depth, token) pairs to revive; small (≤ budget_b × rounds), so the
    /// linear probe is the right structure — zero allocation.
    restored: &'a [(usize, usize)],
}

impl ScreeningPruner for ReopenScreener<'_> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if self
            .restored
            .iter()
            .any(|&(d, t)| d == depth && t == token_idx)
        {
            // Revival (Issue 956 T2): the revision says this candidate's
            // elimination was WRONG, so `ln R` drops out of the blended
            // score and the candidate competes on `ln P` alone.
            return 1.0;
        }
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

/// The chain prefix of a built tree: Phase A pushes chain nodes first in
/// strictly increasing depth order, so the chain is the longest prefix of
/// `tree` whose node at index `i` sits at depth `i`.
fn chain_prefix_of(tree: &[TreeNode], out: &mut Vec<usize>) {
    out.clear();
    for (i, node) in tree.iter().enumerate() {
        if node.depth != i {
            break;
        }
        out.push(node.token_idx);
    }
}

/// T2b capped-mean bisection — LIFTED to [`katgpt_core::sigmoid_budget`]
/// (Issue 956 T3's shared leaf; the closure form keeps the call zero-alloc
/// over the reused rank buffer). `gamma` must arrive PRE-SCALED to the pool
/// (`γ·n/spread` — see [`ReopenConfig::gamma`]): a fixed absolute slope is
/// distribution-blind and was measured strangling the drive to zero reopens.
fn bisect_reopen_cut(rank: &[(f32, usize, usize)], gamma: f32, target: f32) -> f32 {
    let s_max = rank.iter().map(|r| r.0).fold(f32::NEG_INFINITY, f32::max);
    let s_min = rank.iter().map(|r| r.0).fold(f32::INFINITY, f32::min);
    katgpt_core::sigmoid_budget::bisect_sigmoid_cut(
        |tau| rank.iter().map(|r| fast_sigmoid(gamma * (r.0 - tau))).sum(),
        s_min,
        s_max,
        gamma,
        target,
    )
}

/// Elimination-time memory at chain context: for every (depth, token) with
/// positive mass, replay the screener's relevance under the chain prefix of
/// that depth and record it when it would be hard-trimmed. Already-reopened
/// candidates read 1.0 through the same wrapper, so later rounds see them
/// as no longer eliminated.
fn elimination_memory(
    marginals: &[&[f32]],
    threshold: f32,
    screener: &ReopenScreener<'_>,
    chain: &[usize],
    records: &mut Vec<EliminationRecord>,
    counts: &mut [usize],
) {
    records.clear();
    counts.fill(0);
    for (depth, marginal) in marginals.iter().enumerate() {
        // The chain prefix available at this depth. Past a chain break the
        // deepest available prefix is the disclosed replay context.
        let parent = &chain[..depth.min(chain.len())];
        for (token_idx, &prob) in marginal.iter().enumerate() {
            if prob <= 0.0 {
                continue;
            }
            let relevance = screener.relevance(depth, token_idx, parent);
            if relevance <= threshold {
                records.push(EliminationRecord {
                    depth,
                    token_idx,
                    score: relevance,
                });
                counts[depth] += 1;
            }
        }
    }
}

/// Conflict-triggered targeted candidate reopening (Issue 956 T2).
///
/// Pass 1 is the plain [`build_dd_tree_screened`]. When no depth conflicts,
/// that tree is returned unchanged — the same nodes in the same order with
/// the same scores (G3). When a depth conflicts, eliminated candidates at
/// clause-implicated positions are revived (top-`budget_b` by `score^β`)
/// and the build re-propagates through the revival wrapper.
///
/// `clause_db` persists across calls (the CDCL learned-clause store); the
/// ledger inside this call keeps the per-node reopen cap.
pub fn build_dd_tree_screened_reopen(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    reopen: &ReopenConfig,
    clause_db: &mut ConflictClauseDB,
    stats: &mut ReopenStats,
) -> Vec<TreeNode> {
    // Default-off: the plain build, byte-identical, machinery never runs.
    if reopen.budget_b == 0 || reopen.max_rounds == 0 {
        return build_dd_tree_screened(marginals, config, screener, chain_seed);
    }

    let threshold = config.screening_threshold;
    // β sanity: ≤ 0 or NaN collapses the ranking to a constant (pool order).
    let beta = if reopen.beta.is_finite() && reopen.beta > 0.0 {
        reopen.beta
    } else {
        0.0
    };
    // T2b mode: eta > 0 (finite) selects the η-bounded count; everything
    // else is the fixed-b posture (T2 byte-for-byte).
    let eta_mode = reopen.eta.is_finite() && reopen.eta > 0.0;
    let gamma = if reopen.gamma.is_finite() && reopen.gamma > 0.0 {
        reopen.gamma
    } else {
        20.0
    };
    let headroom_center = if reopen.headroom_center.is_finite() {
        reopen.headroom_center
    } else {
        4.0
    };
    let headroom_gamma = if reopen.headroom_gamma.is_finite() && reopen.headroom_gamma >= 0.0 {
        reopen.headroom_gamma
    } else {
        1.0
    };

    // T2b headroom denominator: the distinct (depth, token) candidate space
    // (Σ positive-mass tokens — the V·D measure the T2 budget note names).
    // Loop-invariant (marginals never change) — computed once per call.
    let distinct_space: usize = marginals
        .iter()
        .map(|m| m.iter().filter(|&&p| p > 0.0).count())
        .sum();

    let mut builder = TreeBuilder::new(config);
    let mut tree = builder.build_screened(marginals, config, screener, chain_seed);

    // Scratch buffers: allocated once per call, reused across rounds
    // (the reopen pass adds no per-node allocation — G4).
    let mut chain: Vec<usize> = Vec::with_capacity(marginals.len());
    let mut elim_records: Vec<EliminationRecord> = Vec::new();
    let mut elim_counts: Vec<usize> = vec![0; marginals.len()];
    let mut conflicted: Vec<usize> = Vec::new();
    let mut implicated: Vec<usize> = Vec::new();
    let mut restored: Vec<(usize, usize)> = Vec::new();
    let mut ledger: Vec<(usize, usize)> = Vec::new();
    let mut rank: Vec<(f32, usize, usize)> = Vec::new();

    let mut rounds = 0_usize;
    while rounds < reopen.max_rounds {
        // Chain context of the CURRENT tree (round 1: the plain build;
        // later rounds: the rebuilt one — restored tokens may extend it).
        chain_prefix_of(tree, &mut chain);

        // Elimination memory at chain context (the wrapper carries the
        // ledger of already-revived candidates; scoped so the borrow of
        // `restored` ends before it is mutated below).
        let mem_wrapper = ReopenScreener {
            inner: screener,
            restored: &restored,
        };
        elimination_memory(
            marginals,
            threshold,
            &mem_wrapper,
            &chain,
            &mut elim_records,
            &mut elim_counts,
        );

        // Per-depth conflict detection over the replayed prune rates.
        conflicted.clear();
        for (depth, marginal) in marginals.iter().enumerate() {
            let total = marginal.iter().filter(|&&p| p > 0.0).count();
            if reopen.detector.is_conflicted_at_depth(
                std::slice::from_ref(marginal),
                elim_counts[depth],
                total,
                depth,
            ) {
                conflicted.push(depth);
            }
        }
        if conflicted.is_empty() {
            break;
        }
        stats.conflicts_fired += conflicted.len();

        // Learn the commitment in force at each fired depth: the singleton
        // clause {(depth, chain_token)}. A depth the chain never reached has
        // no commitment to blame (the conflict says screening killed
        // everything there) — implicated, but nothing learned.
        for &depth in &conflicted {
            if chain.len() > depth {
                clause_db.learn(std::collections::HashSet::from([(depth, chain[depth])]));
            }
        }

        // Implicated positions: this round's fired depths plus every depth
        // named by a live clause (prior rounds / prior calls; FIFO eviction
        // ages stale clauses out).
        implicated.clear();
        implicated.extend_from_slice(&conflicted);
        for pos in clause_db.implicated_positions() {
            if !implicated.contains(&pos) {
                implicated.push(pos);
            }
        }

        // Reopen pool: eliminations at implicated positions, minus the
        // no-cycle exclusions (a live clause names it), minus the ledger
        // (per-node cap: reopened at most once per call), minus what the
        // tree already carries at that depth.
        rank.clear();
        for rec in &elim_records {
            if !implicated.contains(&rec.depth) {
                continue;
            }
            if clause_db.is_violated(&[(rec.depth, rec.token_idx)]) {
                stats.no_cycle_blocked += 1;
                continue;
            }
            if ledger.contains(&(rec.depth, rec.token_idx)) {
                continue;
            }
            if tree
                .iter()
                .any(|n| n.depth == rec.depth && n.token_idx == rec.token_idx)
            {
                continue;
            }
            rank.push((rec.score.powf(beta), rec.depth, rec.token_idx));
        }
        if rank.is_empty() {
            // Fired but nothing admissible — no progress is possible this
            // call; stop (the anti-oscillation break).
            break;
        }
        rank.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        // The per-round count (T2 fixed-b, or the T2b capped-mean bisection).
        // rank is sorted descending by rank score, so every candidate at or
        // above the cut is a prefix.
        let count = if eta_mode {
            // Headroom drive over the MEASURED tightness: the pop budget
            // relative to the distinct (depth, token) candidate space.
            let ratio = config.tree_budget as f32 / distinct_space.max(1) as f32;
            let headroom = fast_sigmoid(headroom_gamma * (ratio - headroom_center));
            let target = (reopen.eta * headroom)
                .min(reopen.budget_b as f32)
                .min(rank.len() as f32);
            if target <= 0.0 {
                0
            } else {
                // γ in units of pool spacing: the σ transition width is
                // spread/(γ·n) — one mean order-statistic gap at γ = 1.
                let n = rank.len() as f32;
                let s_max = rank.iter().map(|r| r.0).fold(f32::NEG_INFINITY, f32::max);
                let s_min = rank.iter().map(|r| r.0).fold(f32::INFINITY, f32::min);
                let spread = (s_max - s_min).max(f32::EPSILON);
                let gamma_eff = gamma * n / spread;
                let cut = bisect_reopen_cut(&rank, gamma_eff, target);
                rank.iter()
                    .take_while(|r| r.0 >= cut)
                    .count()
                    .min(reopen.budget_b)
            }
        } else {
            reopen.budget_b
        };

        restored.clear();
        for &(_, depth, token_idx) in rank.iter().take(count) {
            restored.push((depth, token_idx));
        }
        if restored.is_empty() {
            // Fired but nothing admissible (η mode can bisect to a zero
            // count at tight headroom) — no progress is possible this call;
            // stop before the re-propagation (the anti-oscillation break;
            // also skips a wasted identical rebuild).
            break;
        }

        // Re-propagate: same builder (clears internally), revival wrapper
        // scoped to the build (the borrow ends with it).
        let build_wrapper = ReopenScreener {
            inner: screener,
            restored: &restored,
        };
        tree = builder.build_screened(marginals, config, &build_wrapper, chain_seed);

        ledger.extend_from_slice(&restored);
        stats.reopens_applied += restored.len();
        rounds += 1;
    }
    stats.rounds = rounds;

    builder.into_tree()
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaScreeningPruner;

    const VOCAB: usize = 4;
    const DEPTHS: usize = 3;

    fn test_config(threshold: f32, budget: usize) -> katgpt_types::Config {
        let mut config = katgpt_types::Config::draft();
        config.vocab_size = VOCAB;
        config.draft_lookahead = DEPTHS;
        config.tree_budget = budget;
        config.screening_threshold = threshold;
        // Deterministic expansion: no early exit.
        config.early_exit_patience = 0;
        config.early_exit_gap = 0.0;
        config
    }

    /// Relevance = the token's own marginal mass (the LDT elimination
    /// story; parent-independent, so the chain-context replay is exact).
    struct MassScreener {
        marginals: Vec<Vec<f32>>,
    }

    impl ScreeningPruner for MassScreener {
        fn relevance(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            self.marginals[depth][token_idx]
        }
    }

    /// The full node identity: depth, token, exact score bits, path.
    fn same_node(a: &TreeNode, b: &TreeNode) -> bool {
        a.depth == b.depth
            && a.token_idx == b.token_idx
            && a.score.to_bits() == b.score.to_bits()
            && a.parent_path == b.parent_path
    }

    fn trees_identical(a: &[TreeNode], b: &[TreeNode]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_node(x, y))
    }

    fn in_tree(tree: &[TreeNode], depth: usize, token: usize) -> bool {
        tree.iter()
            .any(|n| n.depth == depth && n.token_idx == token)
    }

    /// The workhorse instance. At θ=0.15: depth 0 eliminates 3 of 4
    /// (prune 0.75 > 0.60 — fires), depth 1 eliminates 3 of 4 (0.75 >
    /// 0.58 — fires), depth 2 eliminates 2 of 4 (0.5 < 0.56 — silent).
    fn workhorse_marginals() -> Vec<Vec<f32>> {
        vec![
            vec![0.6, 0.1, 0.1, 0.1],
            vec![0.5, 0.12, 0.05, 0.05],
            vec![0.4, 0.3, 0.1, 0.1],
        ]
    }

    fn slices_of(marginals: &[Vec<f32>]) -> Vec<&[f32]> {
        marginals.iter().map(|m| m.as_slice()).collect()
    }

    #[test]
    fn zero_budget_is_byte_identical() {
        let marginals = workhorse_marginals();
        let slices = slices_of(&marginals);
        let screener = MassScreener {
            marginals: marginals.clone(),
        };
        let config = test_config(0.15, 64);

        let plain = build_dd_tree_screened(&slices, &config, &screener, true);
        let mut db = ConflictClauseDB::default();
        let mut stats = ReopenStats::default();
        let reopened = build_dd_tree_screened_reopen(
            &slices,
            &config,
            &screener,
            true,
            &ReopenConfig::default(),
            &mut db,
            &mut stats,
        );

        assert!(
            trees_identical(&plain, &reopened),
            "G3: b=0 must be the plain build"
        );
        assert_eq!(stats, ReopenStats::default(), "no machinery ran");
    }

    #[test]
    fn zero_rounds_cap_is_byte_identical() {
        let marginals = workhorse_marginals();
        let slices = slices_of(&marginals);
        let screener = MassScreener {
            marginals: marginals.clone(),
        };
        let config = test_config(0.15, 64);

        let plain = build_dd_tree_screened(&slices, &config, &screener, true);
        let mut db = ConflictClauseDB::default();
        let mut stats = ReopenStats::default();
        let cfg = ReopenConfig {
            budget_b: 4,
            max_rounds: 0,
            ..ReopenConfig::default()
        };
        let reopened = build_dd_tree_screened_reopen(
            &slices, &config, &screener, true, &cfg, &mut db, &mut stats,
        );

        assert!(trees_identical(&plain, &reopened));
        assert_eq!(stats, ReopenStats::default());
    }

    #[test]
    fn no_conflict_no_change() {
        // Only 2 of 4 eliminated per depth: prune rate 0.5 never crosses
        // the F4 effective maxes (0.60 / 0.58 / 0.56), so nothing fires.
        let marginals = vec![
            vec![0.55, 0.20, 0.13, 0.12],
            vec![0.50, 0.25, 0.13, 0.12],
            vec![0.45, 0.30, 0.13, 0.12],
        ];
        let slices = slices_of(&marginals);
        let screener = MassScreener {
            marginals: marginals.clone(),
        };
        let config = test_config(0.15, 64);

        let plain = build_dd_tree_screened(&slices, &config, &screener, true);
        let mut db = ConflictClauseDB::default();
        let mut stats = ReopenStats::default();
        let cfg = ReopenConfig {
            budget_b: 4,
            ..ReopenConfig::default()
        };
        let reopened = build_dd_tree_screened_reopen(
            &slices, &config, &screener, true, &cfg, &mut db, &mut stats,
        );

        assert!(
            trees_identical(&plain, &reopened),
            "G3: no conflict → unchanged"
        );
        assert_eq!(stats.conflicts_fired, 0);
        assert_eq!(stats.rounds, 0);
    }

    #[test]
    fn conflict_fires_and_reopens() {
        let marginals = workhorse_marginals();
        let slices = slices_of(&marginals);
        let screener = MassScreener {
            marginals: marginals.clone(),
        };
        let config = test_config(0.15, 64);

        let plain = build_dd_tree_screened(&slices, &config, &screener, true);
        assert!(!in_tree(&plain, 0, 1), "precondition: (0,1) eliminated");
        assert!(!in_tree(&plain, 1, 1), "precondition: (1,1) eliminated");

        let mut db = ConflictClauseDB::default();
        let mut stats = ReopenStats::default();
        let cfg = ReopenConfig {
            budget_b: 2,
            beta: 1.0,
            max_rounds: 2,
            ..ReopenConfig::default()
        };
        let reopened = build_dd_tree_screened_reopen(
            &slices, &config, &screener, true, &cfg, &mut db, &mut stats,
        );

        // Round 1 fired at depths 0 and 1; the revival cleared both (the
        // restored read 1.0 → prune rates drop below the maxes), so no
        // round 2 despite the cap of 2.
        assert_eq!(stats.conflicts_fired, 2);
        assert_eq!(stats.rounds, 1);
        assert_eq!(stats.reopens_applied, 2);
        assert_eq!(stats.no_cycle_blocked, 0);
        // Top-2 by score^1: (1,1) at 0.12, then the depth-0 trio tied at
        // 0.10 with (0,1) winning the depth tie-break.
        assert!(in_tree(&reopened, 1, 1), "revived candidate must land");
        assert!(in_tree(&reopened, 0, 1), "revived candidate must land");
        assert!(!in_tree(&reopened, 1, 2), "b=2 leaves (1,2) eliminated");
        // Both fired depths had chain commitments (the argmax chain
        // survives every depth here — the eliminations are all non-argmax
        // tokens), so two singleton clauses learned: {(0,0)} and {(1,0)}.
        assert_eq!(db.len(), 2);

        // Revival semantics: the revived node's score is pure LLM mass on
        // the chain prefix — score0 + ln(0.12), R=1 contributing ln 1 = 0.
        // It enters as a Phase-B sibling of the depth-1 chain node, whose
        // parent score is the depth-0 chain node's.
        let chain0 = &reopened[0]; // depth-0 chain node = argmax(0.6)
        let expected = chain0.score + 0.12_f32.ln();
        let revived = reopened
            .iter()
            .find(|n| n.depth == 1 && n.token_idx == 1)
            .expect("revived node present");
        assert_eq!(
            revived.score.to_bits(),
            expected.to_bits(),
            "revival = chain prefix + ln(P), no screening penalty"
        );
    }

    #[test]
    fn no_cycle_blocks_clause_named_candidate() {
        let marginals = workhorse_marginals();
        let slices = slices_of(&marginals);
        let screener = MassScreener {
            marginals: marginals.clone(),
        };
        let config = test_config(0.15, 64);

        // Pre-learn the exclusion: a live clause names (1,1).
        let mut db = ConflictClauseDB::default();
        use std::collections::HashSet;
        db.learn(HashSet::from([(1_usize, 1_usize)]));

        let mut stats = ReopenStats::default();
        let cfg = ReopenConfig {
            budget_b: 2,
            ..ReopenConfig::default()
        };
        let reopened = build_dd_tree_screened_reopen(
            &slices, &config, &screener, true, &cfg, &mut db, &mut stats,
        );

        assert!(
            stats.no_cycle_blocked >= 1,
            "the clause-named candidate must be counted as blocked"
        );
        assert!(
            !in_tree(&reopened, 1, 1),
            "no-cycle: a live-clause candidate is never reopened"
        );
        // The pool's next-best still reopened (b=2: the two tied 0.10s at
        // depth 0).
        assert!(in_tree(&reopened, 0, 1));
        assert!(in_tree(&reopened, 0, 2));
        assert_eq!(stats.reopens_applied, 2);
    }

    #[test]
    fn round_cap_terminates_under_persistent_conflict() {
        // Depth 1 is 8-wide with 7 eliminations (0.875 > 0.58); b=1 can
        // clear one per round, so the conflict persists past round 1 — the
        // cap must be what stops the loop (the FSM
        // no-unbounded-back-and-forth law, structurally).
        let mut marginals = workhorse_marginals();
        let mut d1 = vec![0.5_f32, 0.14_f32];
        d1.extend(vec![0.05_f32; 6]);
        marginals[1] = d1;
        let slices = slices_of(&marginals);
        let screener = MassScreener {
            marginals: marginals.clone(),
        };
        let mut config = test_config(0.15, 256);
        config.vocab_size = 8;

        let mut db = ConflictClauseDB::default();
        let mut stats = ReopenStats::default();
        let cfg = ReopenConfig {
            budget_b: 1,
            max_rounds: 1,
            ..ReopenConfig::default()
        };
        let reopened = build_dd_tree_screened_reopen(
            &slices, &config, &screener, true, &cfg, &mut db, &mut stats,
        );

        assert_eq!(stats.rounds, 1, "round cap holds");
        assert_eq!(stats.reopens_applied, 1);
        assert!(
            stats.conflicts_fired >= 2,
            "depths 0 and 1 both fired (the cap is what stopped the loop)"
        );
        // The top-scored eliminated candidate at depth 1 (0.14) revived.
        assert!(in_tree(&reopened, 1, 1));
    }

    #[test]
    fn determinism_replay_is_bit_identical() {
        let marginals = workhorse_marginals();
        let slices = slices_of(&marginals);
        let screener = MassScreener {
            marginals: marginals.clone(),
        };
        let config = test_config(0.15, 64);
        let cfg = ReopenConfig {
            budget_b: 2,
            max_rounds: 3,
            ..ReopenConfig::default()
        };

        let run = || {
            let mut db = ConflictClauseDB::default();
            let mut stats = ReopenStats::default();
            let tree = build_dd_tree_screened_reopen(
                &slices, &config, &screener, true, &cfg, &mut db, &mut stats,
            );
            (tree, stats, db.len())
        };
        let (t1, s1, c1) = run();
        let (t2, s2, c2) = run();
        assert!(trees_identical(&t1, &t2));
        assert_eq!(s1, s2);
        assert_eq!(c1, c2);
    }

    #[test]
    fn alpha_composition_conflict_reopens() {
        // The α-lattice composition: `AlphaScreeningPruner` as the inner
        // screener (relevance 1.0 allowed / 0.0 not — mass-blind). The
        // solutions exclude 3 of 4 tokens at depth 0 → prune 0.75 > 0.60
        // fires; α eliminations revive exactly like mass eliminations.
        let marginals = workhorse_marginals();
        let slices = slices_of(&marginals);
        let alpha = AlphaScreeningPruner::with_solutions(
            DEPTHS,
            vec![vec![0, 1, 1], vec![0, 1, 1], vec![0, 1, 2]],
        );
        let config = test_config(0.5, 64);

        let plain = build_dd_tree_screened(&slices, &config, &alpha, true);
        assert!(
            !in_tree(&plain, 1, 0),
            "precondition: (1,0) is not in any solution → α-eliminated"
        );

        let mut db = ConflictClauseDB::default();
        let mut stats = ReopenStats::default();
        let cfg = ReopenConfig {
            budget_b: 4,
            ..ReopenConfig::default()
        };
        let reopened = build_dd_tree_screened_reopen(
            &slices, &config, &alpha, true, &cfg, &mut db, &mut stats,
        );

        assert!(stats.conflicts_fired >= 1, "α eliminations fire conflicts");
        assert_eq!(stats.reopens_applied, 4, "b=4 applied");
        assert!(
            in_tree(&reopened, 1, 0),
            "the α-excluded candidate revived and landed"
        );
        // The revival extends the chain: the depth-1 argmax token was
        // α-eliminated in the plain build, so the plain chain died at
        // depth 0 (tree[1] is a heap pop, not the chain); after revival
        // Phase A walks through it.
        assert_eq!(reopened[1].depth, 1, "chain extended through the revival");
        assert_ne!(plain[1].depth, 0, "sanity: the plain chain is shorter");
    }

    // ── T2b: expected-reopen-count preservation (the anti-thrash budget) ──

    /// The bisection helper directly: distinct scores, sharp γ. The helper
    /// takes the PRE-SCALED slope (the call site computes `γ·n/spread`);
    /// these cases scale by hand the same way. The cut must isolate exactly
    /// `round(target)` top scores when the shape allows, and the count
    /// above the cut is what the caller takes.
    #[test]
    fn bisect_cut_isolates_top_target() {
        let rank: Vec<(f32, usize, usize)> = [0.9_f32, 0.5, 0.3, 0.1]
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, 0, i))
            .collect();
        // γ·n/spread = 50·4/0.8 = 250 (the call site's scaling).
        let gamma_eff = 50.0 * 4.0 / 0.8;
        // target 2 → the cut lands at the 0.5 plateau edge (E's root on a
        // plateau slides to the boundary; `>=` keeps both 0.9 and 0.5).
        let cut = bisect_reopen_cut(&rank, gamma_eff, 2.0);
        let count = rank.iter().take_while(|r| r.0 >= cut).count();
        assert_eq!(count, 2, "the top-2 group survives the plateau-edge cut");
        // target 1 → cut just under the 0.9 standout.
        let cut = bisect_reopen_cut(&rank, gamma_eff, 1.0);
        let count = rank.iter().take_while(|r| r.0 >= cut).count();
        assert_eq!(count, 1);
        // target ≈ pool → cut below the minimum → all four.
        let cut = bisect_reopen_cut(&rank, gamma_eff, 3.9);
        let count = rank.iter().take_while(|r| r.0 >= cut).count();
        assert_eq!(count, 4);
        // target → 0 → even the TOP score's propensity is below 0.5, so the
        // cut lands ABOVE it: an expected count under half-a-candidate reads
        // deterministically as none (the honest σ-cut semantics).
        let cut = bisect_reopen_cut(&rank, gamma_eff, 0.1);
        let count = rank.iter().take_while(|r| r.0 >= cut).count();
        assert_eq!(
            count, 0,
            "a 0.1-expected target puts the cut above every score"
        );
    }

    /// NaN η must fall back to the fixed-b posture, not panic and not
    /// select a degenerate count. (eta = 0 — the T2 fixed posture — is
    /// already pinned by every test above: Default::default() carries it.)
    #[test]
    fn eta_nan_falls_back_to_fixed_b() {
        let marginals = workhorse_marginals();
        let slices = slices_of(&marginals);
        let screener = MassScreener {
            marginals: marginals.clone(),
        };
        let config = test_config(0.15, 64);

        let run = |eta: f32| {
            let mut db = ConflictClauseDB::default();
            let mut stats = ReopenStats::default();
            let cfg = ReopenConfig {
                budget_b: 2,
                max_rounds: 2,
                eta,
                ..ReopenConfig::default()
            };
            let tree = build_dd_tree_screened_reopen(
                &slices, &config, &screener, true, &cfg, &mut db, &mut stats,
            );
            (tree, stats)
        };
        let (fixed_tree, fixed_stats) = run(0.0);
        let (nan_tree, nan_stats) = run(f32::NAN);
        assert!(trees_identical(&fixed_tree, &nan_tree));
        assert_eq!(fixed_stats, nan_stats);
    }

    /// Roomy headroom (budget 64, space 12 → ratio 5.33 → headroom ≈ 0.79
    /// at the shipped calibration): η = 4 targets ≈ 3.2, and the
    /// workhorse's score shape (one 0.12, three 0.10, …) cannot split the
    /// 0.10 tie, so the count lands on the cap-adjacent 4 — the SAME set
    /// fixed-b=4 takes. Pins the shipped headroom constants: retuning them
    /// breaks this test loudly (the calibration-pin convention).
    #[test]
    fn eta_mode_roomy_recovers_the_full_budget() {
        let marginals = workhorse_marginals();
        let slices = slices_of(&marginals);
        let screener = MassScreener {
            marginals: marginals.clone(),
        };
        let config = test_config(0.15, 64);

        let mut db = ConflictClauseDB::default();
        let mut stats = ReopenStats::default();
        let cfg = ReopenConfig {
            budget_b: 4,
            max_rounds: 2,
            eta: 4.0,
            ..ReopenConfig::default()
        };
        let reopened = build_dd_tree_screened_reopen(
            &slices, &config, &screener, true, &cfg, &mut db, &mut stats,
        );

        assert_eq!(stats.reopens_applied, 4, "roomy headroom spends the budget");
        assert_eq!(
            stats.rounds, 1,
            "the round-1 revival clears both fired depths"
        );
        assert!(in_tree(&reopened, 1, 1));
        assert!(in_tree(&reopened, 0, 1));
        assert!(in_tree(&reopened, 0, 2));
        assert!(in_tree(&reopened, 0, 3));
    }

    /// THE anti-displacement pin (the T2 regression's fix): tight headroom
    /// (budget 12 = the distinct space, ratio 1.0 → headroom ≈ 0.047 at
    /// the shipped calibration) bisects the target to ≈ 0.19, which the
    /// workhorse's score gaps cannot admit — zero reopens, the round breaks
    /// before re-propagating, and the returned tree IS the plain build.
    /// Detection still fired (the signal is recorded, the spend is not).
    #[test]
    fn eta_mode_tight_headroom_makes_no_reopens() {
        let marginals = workhorse_marginals();
        let slices = slices_of(&marginals);
        let screener = MassScreener {
            marginals: marginals.clone(),
        };
        let config = test_config(0.15, 12);

        let plain = build_dd_tree_screened(&slices, &config, &screener, true);
        let mut db = ConflictClauseDB::default();
        let mut stats = ReopenStats::default();
        let cfg = ReopenConfig {
            budget_b: 4,
            max_rounds: 2,
            eta: 4.0,
            ..ReopenConfig::default()
        };
        let reopened = build_dd_tree_screened_reopen(
            &slices, &config, &screener, true, &cfg, &mut db, &mut stats,
        );

        assert!(
            trees_identical(&plain, &reopened),
            "tight headroom must collapse to the plain build (the T2 budget-150 regression is the measured reason)"
        );
        assert_eq!(stats.reopens_applied, 0);
        assert_eq!(stats.rounds, 0, "a zero-count round never re-propagates");
        assert_eq!(
            stats.conflicts_fired, 2,
            "detection still ran and recorded the signal"
        );
    }

    /// A huge η saturates at the ABSOLUTE cap: the bisection target is
    /// clamped to `budget_b`, and the count is clamped AGAIN after the cut.
    /// With a standout elimination (0.14) above two 0.002 tails, target 2
    /// roots at the tail cluster's edge — the cut admits all three scores,
    /// and the cap takes exactly `budget_b`.
    #[test]
    fn eta_mode_never_exceeds_the_absolute_cap() {
        // Depth 0: one standout elimination (0.14) + two 0.002 tails; depth
        // 1 stays silent (2/4 = 0.5 < 0.58), depth 2 silent (1/4).
        let marginals = vec![
            vec![0.6, 0.14, 0.002, 0.002],
            vec![0.5, 0.30, 0.10, 0.10],
            vec![0.4, 0.30, 0.20, 0.10],
        ];
        let slices = slices_of(&marginals);
        let screener = MassScreener {
            marginals: marginals.clone(),
        };
        let config = test_config(0.15, 64);

        let mut db = ConflictClauseDB::default();
        let mut stats = ReopenStats::default();
        let cfg = ReopenConfig {
            budget_b: 2,
            max_rounds: 2,
            eta: 100.0,
            ..ReopenConfig::default()
        };
        let reopened = build_dd_tree_screened_reopen(
            &slices, &config, &screener, true, &cfg, &mut db, &mut stats,
        );

        assert_eq!(
            stats.reopens_applied, 2,
            "η = 100 saturates: the count is clamped at budget_b = 2"
        );
        assert_eq!(stats.rounds, 1, "two restores clear depth 0's prune rate");
        assert!(in_tree(&reopened, 0, 1), "the standout candidate revived");
        assert!(in_tree(&reopened, 0, 2), "the first tail fills the cap");
        assert!(!in_tree(&reopened, 0, 3), "the second tail is capped out");
    }

    /// η mode is deterministic end to end (the bisection is pure
    /// arithmetic; no RNG anywhere in the count path).
    #[test]
    fn eta_mode_replay_is_bit_identical() {
        let marginals = workhorse_marginals();
        let slices = slices_of(&marginals);
        let screener = MassScreener {
            marginals: marginals.clone(),
        };
        let config = test_config(0.15, 64);
        let cfg = ReopenConfig {
            budget_b: 3,
            max_rounds: 3,
            eta: 2.0,
            ..ReopenConfig::default()
        };

        let run = || {
            let mut db = ConflictClauseDB::default();
            let mut stats = ReopenStats::default();
            let tree = build_dd_tree_screened_reopen(
                &slices, &config, &screener, true, &cfg, &mut db, &mut stats,
            );
            (tree, stats, db.len())
        };
        let (t1, s1, c1) = run();
        let (t2, s2, c2) = run();
        assert!(trees_identical(&t1, &t2));
        assert_eq!(s1, s2);
        assert_eq!(c1, c2);
    }
}
