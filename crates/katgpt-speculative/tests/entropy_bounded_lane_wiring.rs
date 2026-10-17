#![cfg(feature = "entropy_bounded_commit")]
//! Issue 917 T3 — G1-style gates for the EB lane wiring.
//!
//! The T1 policy itself is gated upstream (`katgpt-core`); these tests pin
//! the two lane wirings on top of it:
//!
//! - **DDTree EB expansion** ([`build_dd_tree_eb`]): no-stall on flat and
//!   degenerate distributions (≥ 1 child committed per expansion whenever a
//!   finite-surprisal candidate exists), cap behavior, determinism across
//!   builds and across input permutations.
//! - **DFlash EB block commit** ([`dflash_block_commit_eb_with`]): no-stall,
//!   brute-force `Σ H − max H` agreement over seeded random cases (mirrors
//!   the T1 test style), determinism across depth permutations, cap
//!   behavior, NaN-row handling, argmax emission, and the end-to-end
//!   [`dflash_predict_eb_with`] composition.
//!
//! ```sh
//! cargo test -p katgpt-speculative --features entropy_bounded_commit \
//!   --test entropy_bounded_lane_wiring
//! ```

use katgpt_core::entropy_bounded_commit::{ErrorProxy, position_stats};
use katgpt_speculative::NoPruner;
use katgpt_speculative::dflash::{DflashCache, DflashCtx};
use katgpt_speculative::entropy_bounded::{
    EbBuildScratch, EbCommitConfig, EbCommitScratch, build_dd_tree_eb, build_dd_tree_eb_into,
    dflash_block_commit_eb_with, dflash_predict_eb_with,
};
use katgpt_types::Config;

fn test_config(vocab: usize, budget: usize) -> Config {
    let mut c = Config::draft();
    c.vocab_size = vocab;
    c.tree_budget = budget;
    c.early_exit_patience = 0;
    c.early_exit_gap = 0.0;
    c
}

fn eb(gamma: f32, max_commit: usize, proxy: ErrorProxy) -> EbCommitConfig {
    EbCommitConfig {
        gamma,
        max_commit,
        proxy,
    }
}

fn uniform_row(vocab: usize) -> Vec<f32> {
    vec![1.0 / vocab as f32; vocab]
}

/// Degenerate probability row (a DDTree marginal): all mass on `hot`. The
/// singleton's surprisal is exactly −ln 1 = 0, so it commits at any γ ≥ 0.
fn one_hot_probs(vocab: usize, hot: usize) -> Vec<f32> {
    let mut r = vec![0.0f32; vocab];
    r[hot] = 1.0;
    r
}

/// Degenerate logits row (a DFlash depth row; the T1 peaked fixture's
/// shape): softmax is one-hot and the entropy is exactly 0.0 in f32 (the
/// tail terms underflow), so a whole zero-entropy block commits even at γ = 0.
fn one_hot_logits(vocab: usize, hot: usize) -> Vec<f32> {
    let mut r = vec![-1e4f32; vocab];
    r[hot] = 0.0;
    r
}

// ── DDTree EB expansion ─────────────────────────────────────────

#[test]
fn ddtree_eb_no_stall_on_flat_marginals() {
    // Flat rows: every surprisal is ln(V) — a second child would blow the γ
    // budget, so each expansion commits exactly the EB singleton. No
    // expansion ever commits zero: children_committed == expansions and no
    // histogram bucket at 0.
    let (depths, vocab) = (4usize, 16usize);
    let rows: Vec<Vec<f32>> = (0..depths).map(|_| uniform_row(vocab)).collect();
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let config = test_config(vocab, 64);

    let mut scratch = EbBuildScratch::default();
    let tree = build_dd_tree_eb_into(
        &mut scratch,
        &refs,
        &config,
        &NoPruner,
        &eb(0.1, usize::MAX, ErrorProxy::Entropy),
    )
    .to_vec();
    assert_eq!(scratch.expansions, 4, "root + one expansion per chain pop");
    assert_eq!(
        scratch.children_committed, scratch.expansions,
        "no-stall: every expansion committed ≥ 1 child"
    );
    assert_eq!(tree.len(), depths, "the width-1 chain fills depth by depth");
    assert_eq!(scratch.commit_histogram[0], 0, "no stalled expansion");
    assert_eq!(scratch.commit_histogram[1], 4);
}

#[test]
fn ddtree_eb_no_stall_on_degenerate_one_hot() {
    // Degenerate zero-entropy rows: exactly one candidate per expansion —
    // the singleton floor commits it at any γ ≥ 0.
    let (depths, vocab) = (5usize, 8usize);
    let rows: Vec<Vec<f32>> = (0..depths)
        .map(|d| one_hot_probs(vocab, d % vocab))
        .collect();
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let config = test_config(vocab, 64);

    for gamma in [0.0f32, 0.1] {
        let mut scratch = EbBuildScratch::default();
        let tree = build_dd_tree_eb_into(
            &mut scratch,
            &refs,
            &config,
            &NoPruner,
            &eb(gamma, usize::MAX, ErrorProxy::Entropy),
        )
        .to_vec();
        assert_eq!(
            tree.len(),
            depths,
            "γ={gamma}: the chain is fully committed"
        );
        assert_eq!(
            scratch.children_committed, depths,
            "γ={gamma}: no stall on a degenerate distribution"
        );
        assert_eq!(scratch.commit_histogram[0], 0);
    }
}

#[test]
fn ddtree_eb_commits_wide_when_probs_are_close() {
    // Geometric row [0.5, 0.25, 0.125, 0.125]: surprisals
    // [0.693, 1.386, 2.079, 2.079]. γ = 1.5 admits the first two
    // (residual 0 then min(1.386, 0.693) = 0.693) and rejects the third
    // (0.693 + min(2.079, 1.386) = 2.079 > 1.5) — the adaptive width the
    // fixed count never had.
    let (depths, vocab) = (2usize, 4usize);
    let row = [0.5f32, 0.25, 0.125, 0.125];
    let rows: Vec<Vec<f32>> = (0..depths).map(|_| row.to_vec()).collect();
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let config = test_config(vocab, 64);

    let mut scratch = EbBuildScratch::default();
    let tree = build_dd_tree_eb_into(
        &mut scratch,
        &refs,
        &config,
        &NoPruner,
        &eb(1.5, usize::MAX, ErrorProxy::Entropy),
    )
    .to_vec();
    assert_eq!(
        scratch.commit_histogram[2], scratch.expansions as u64,
        "root + both depth-0 pops"
    );
    assert_eq!(tree.len(), 6, "2 roots × 2 children");
    assert!(tree.iter().all(|n| n.depth <= 1));
}

#[test]
fn ddtree_eb_cap_bounds_per_expansion_children() {
    // γ = ∞ accepts everything the cap allows: with max_commit = 2 and ≥ 2
    // valid candidates per row, no expansion can commit more than 2 children
    // — the fixed-width behavior expressed through the same policy.
    let (depths, vocab) = (3usize, 8usize);
    let mut row = vec![0.025f32; vocab];
    row[0] = 0.4;
    row[1] = 0.3;
    row[2] = 0.1;
    let rows: Vec<Vec<f32>> = (0..depths).map(|_| row.clone()).collect();
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let config = test_config(vocab, 64);

    let mut scratch = EbBuildScratch::default();
    build_dd_tree_eb_into(
        &mut scratch,
        &refs,
        &config,
        &NoPruner,
        &eb(f32::INFINITY, 2, ErrorProxy::Entropy),
    );
    assert!(scratch.expansions > 0);
    assert!(
        scratch
            .commit_histogram
            .iter()
            .enumerate()
            .all(|(k, &n)| k <= 2 || n == 0),
        "cap bounds every expansion's commit count"
    );
    assert_eq!(
        scratch.commit_histogram[2], scratch.expansions as u64,
        "every expansion had ≥ 2 candidates and hit the cap"
    );
    assert_eq!(
        scratch.children_committed,
        2 * scratch.expansions,
        "γ=∞ + cap=2 is exactly fixed width-k=2 here"
    );
}

#[test]
fn ddtree_eb_deterministic_and_permutation_invariant() {
    let mut rng = fastrand::Rng::with_seed(917);
    let (depths, vocab) = (4usize, 12usize);
    let rows: Vec<Vec<f32>> = (0..depths)
        .map(|_| {
            let raw: Vec<f32> = (0..vocab).map(|_| 0.01 + rng.f32()).collect();
            let sum: f32 = raw.iter().sum();
            raw.iter().map(|&p| p / sum).collect()
        })
        .collect();
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let config = test_config(vocab, 32);
    let cfg = eb(0.5, usize::MAX, ErrorProxy::Entropy);

    let t1 = build_dd_tree_eb(&refs, &config, &NoPruner, &cfg);
    let t2 = build_dd_tree_eb(&refs, &config, &NoPruner, &cfg);
    assert_eq!(t1.len(), t2.len());
    assert!(
        t1.iter().zip(&t2).all(|(a, b)| a.depth == b.depth
            && a.token_idx == b.token_idx
            && a.score.to_bits() == b.score.to_bits()),
        "two builds on the same input are identical"
    );

    // Permute the vocabulary: the (key, index) order may pick different
    // tied tokens, but the commit COUNTS are a function of the entropy
    // multiset, which the permutation preserves.
    let perm: Vec<usize> = {
        let mut p: Vec<usize> = (0..vocab).collect();
        for i in (1..vocab).rev() {
            p.swap(i, rng.usize(..=i));
        }
        p
    };
    let permuted: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| perm.iter().map(|&src| row[src]).collect())
        .collect();
    let pref: Vec<&[f32]> = permuted.iter().map(|r| r.as_slice()).collect();
    let mut s1 = EbBuildScratch::default();
    let mut s2 = EbBuildScratch::default();
    let t1 = build_dd_tree_eb_into(&mut s1, &refs, &config, &NoPruner, &cfg);
    let t2 = build_dd_tree_eb_into(&mut s2, &pref, &config, &NoPruner, &cfg);
    assert_eq!(t1.len(), t2.len(), "tree size is permutation-invariant");
    assert_eq!(
        s1.children_committed, s2.children_committed,
        "commit counts are permutation-invariant"
    );
    assert_eq!(s1.commit_histogram, s2.commit_histogram);
}

#[test]
fn ddtree_eb_skips_nan_and_infinite_probabilities() {
    // NaN and +inf probabilities are malformed marginals: never candidates
    // (an +inf surprisal would terminate the EB prefix at the head and
    // stall the expansion). The finite remainder still commits ≥ 1. Both
    // depths carry the poison so no token id 2/5 can enter at any level.
    let vocab = 8usize;
    let mut poisoned = uniform_row(vocab);
    poisoned[2] = f32::NAN;
    poisoned[5] = f32::INFINITY;
    let rows = [poisoned, uniform_row(vocab)]
        .into_iter()
        .map(|mut r| {
            r[2] = f32::NAN;
            r[5] = f32::INFINITY;
            r
        })
        .collect::<Vec<_>>();
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let config = test_config(vocab, 64);

    let mut scratch = EbBuildScratch::default();
    let tree = build_dd_tree_eb_into(
        &mut scratch,
        &refs,
        &config,
        &NoPruner,
        &eb(f32::INFINITY, usize::MAX, ErrorProxy::Entropy),
    )
    .to_vec();
    assert!(scratch.children_committed >= scratch.expansions);
    assert!(tree.iter().all(|n| n.token_idx != 2 && n.token_idx != 5));
}

// ── DFlash EB block commit ──────────────────────────────────────

#[test]
fn dflash_block_commit_no_stall_flat_and_degenerate() {
    let (steps, vocab) = (8usize, 8usize);

    // Flat: entropy ln 8 everywhere — the singleton floor commits exactly 1.
    let mut flat = Vec::with_capacity(steps * vocab);
    for _ in 0..steps {
        flat.extend(uniform_row(vocab));
    }
    let mut scratch = EbCommitScratch::default();
    let mut tokens = vec![0usize; steps];
    let k = dflash_block_commit_eb_with(
        &flat,
        steps,
        vocab,
        &mut scratch,
        &mut tokens,
        &eb(0.1, usize::MAX, ErrorProxy::Entropy),
    );
    assert_eq!(k, 1, "flat block: the EB singleton, never a stall");

    // Degenerate one-hot rows: all entropies are 0 — the residual never
    // leaves 0, so the whole block commits even at γ = 0.
    let mut one_hot = Vec::with_capacity(steps * vocab);
    for d in 0..steps {
        one_hot.extend(one_hot_logits(vocab, d % vocab));
    }
    let k = dflash_block_commit_eb_with(
        &one_hot,
        steps,
        vocab,
        &mut scratch,
        &mut tokens,
        &eb(0.0, usize::MAX, ErrorProxy::Entropy),
    );
    assert_eq!(
        k, steps,
        "degenerate block: zero entropy commits everything"
    );
    for (d, &t) in tokens.iter().enumerate() {
        assert_eq!(t, d % vocab, "argmax token of depth {d}");
    }
}

/// The largest prefix under the strict (NaN-last, key, index) order whose
/// brute-force `Σ H − max H` stays ≤ γ — the T1 test's oracle, rebuilt here
/// over the lane's own [`position_stats`] so the comparison isolates the
/// lane plumbing (batching, candidate order, depth-ordered emit).
fn bruteforce_prefix(
    stats: &[katgpt_core::entropy_bounded_commit::PositionStats],
    proxy: ErrorProxy,
    gamma: f32,
) -> Vec<u32> {
    let n = stats.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        let (ka, kb) = (stats[a as usize].key(proxy), stats[b as usize].key(proxy));
        ka.is_nan()
            .cmp(&kb.is_nan())
            .then(ka.total_cmp(&kb))
            .then(a.cmp(&b))
    });
    let mut residual = 0.0f32;
    let mut max_h = 0.0f32;
    let mut k = 0usize;
    for &p in &order {
        let h = stats[p as usize].entropy;
        if !h.is_finite() {
            break;
        }
        let h = h.max(0.0);
        let next = if k == 0 { 0.0 } else { residual + h.min(max_h) };
        if next > gamma {
            break;
        }
        residual = next;
        max_h = max_h.max(h);
        k += 1;
    }
    order[..k].to_vec()
}

#[test]
fn dflash_commit_matches_bruteforce_sum_minus_max() {
    let vocab = 8usize;
    let mut rng = fastrand::Rng::with_seed(917);
    let mut flat: Vec<f32> = Vec::new();
    let mut scratch = EbCommitScratch::default();
    let mut tokens = vec![0usize; 32];

    for case in 0..500 {
        let steps = rng.usize(1..24);
        flat.clear();
        for _ in 0..steps * vocab {
            flat.push(rng.f32() * 8.0 - 4.0);
        }
        let gamma = rng.f32() * 3.0;
        let proxy = match case % 3 {
            0 => ErrorProxy::Entropy,
            1 => ErrorProxy::Confidence,
            _ => ErrorProxy::Margin,
        };

        let stats: Vec<_> = (0..steps)
            .map(|d| position_stats(&flat[d * vocab..(d + 1) * vocab]))
            .collect();
        let expected = bruteforce_prefix(&stats, proxy, gamma);
        let n_expected = expected.len();

        let k = dflash_block_commit_eb_with(
            &flat,
            steps,
            vocab,
            &mut scratch,
            &mut tokens,
            &eb(gamma, usize::MAX, proxy),
        );
        let mut got: Vec<u32> = scratch.committed_ids(k).to_vec();
        got.sort_unstable();
        let mut want = expected.clone();
        want.sort_unstable();
        assert_eq!(
            got, want,
            "case {case}: committed set (proxy {proxy:?}, γ {gamma})"
        );
        assert_eq!(k, n_expected, "case {case}: committed count");

        // Every emitted token is the argmax of its committed depth, in
        // ascending depth order: walk the committed depths (already sorted
        // by the emit) and compare slot-wise against the emitted tokens.
        let mut sorted: Vec<usize> = expected.iter().map(|&d| d as usize).collect();
        sorted.sort_unstable();
        assert!(
            tokens[..k].iter().enumerate().all(|(s, &t)| {
                let row = &flat[sorted[s] * vocab..(sorted[s] + 1) * vocab];
                let argmax = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                t == argmax
            }),
            "case {case}: emitted tokens are the committed depths' argmaxes"
        );
    }
}

#[test]
fn dflash_commit_deterministic_across_depth_permutations() {
    let vocab = 8usize;
    // Two sharp rows + two flat rows: a mixed entropy profile.
    let rows = vec![
        {
            let mut r = uniform_row(vocab);
            r[3] += 0.5;
            r
        },
        uniform_row(vocab),
        {
            let mut r = uniform_row(vocab);
            r[1] += 0.9;
            r
        },
        uniform_row(vocab),
    ];
    let perm = [3usize, 1, 0, 2];

    let run = |rows: &[Vec<f32>], order: [usize; 4]| {
        let mut flat = Vec::new();
        for &d in &order {
            flat.extend(rows[d].iter());
        }
        let mut scratch = EbCommitScratch::default();
        let mut tokens = vec![0usize; 4];
        let k = dflash_block_commit_eb_with(
            &flat,
            4,
            vocab,
            &mut scratch,
            &mut tokens,
            &eb(0.2, usize::MAX, ErrorProxy::Entropy),
        );
        scratch.committed_ids(k).to_vec()
    };

    // Depth d in the permuted block holds row perm[d]; map run 2's
    // committed depths back to original row ids and compare the SETS. The
    // count is a function of the entropy multiset; the set, of which rows
    // carry the committed entropies.
    let committed1 = run(&rows, [0, 1, 2, 3]);
    let committed2 = run(&rows, perm);
    assert_eq!(
        committed1.len(),
        committed2.len(),
        "commit count is permutation-invariant"
    );
    let mut set1: Vec<usize> = committed1.iter().map(|&d| d as usize).collect();
    set1.sort_unstable();
    let mut set2: Vec<usize> = committed2.iter().map(|&d| perm[d as usize]).collect();
    set2.sort_unstable();
    assert_eq!(set1, set2, "the same rows commit under both orders");
}

#[test]
fn dflash_commit_cap_and_nan_handling() {
    let (steps, vocab) = (8usize, 6usize);

    // Cap: with γ huge every depth is admissible; the cap binds at 3.
    let mut flat = Vec::with_capacity(steps * vocab);
    for d in 0..steps {
        let mut r = uniform_row(vocab);
        r[d % vocab] += 0.01 * (d + 1) as f32; // tiny spread, nonzero margin
        flat.extend(r);
    }
    let mut scratch = EbCommitScratch::default();
    let mut tokens = vec![0usize; steps];
    let k = dflash_block_commit_eb_with(
        &flat,
        steps,
        vocab,
        &mut scratch,
        &mut tokens,
        &eb(100.0, 3, ErrorProxy::Entropy),
    );
    assert_eq!(k, 3, "the cap binds");
    let uncapped = dflash_block_commit_eb_with(
        &flat,
        steps,
        vocab,
        &mut scratch,
        &mut tokens,
        &eb(100.0, usize::MAX, ErrorProxy::Entropy),
    );
    assert_eq!(uncapped, steps, "without the cap every depth commits");
    // Negative or NaN γ is a garbage config: never commits.
    let k = dflash_block_commit_eb_with(
        &flat,
        steps,
        vocab,
        &mut scratch,
        &mut tokens,
        &eb(f32::NAN, usize::MAX, ErrorProxy::Entropy),
    );
    assert_eq!(k, 0, "NaN γ commits nothing");

    // NaN row: a non-distribution, never committed; the finite depths are.
    let mut nan_block = flat.clone();
    for v in &mut nan_block[2 * vocab..3 * vocab] {
        *v = f32::NAN;
    }
    let k = dflash_block_commit_eb_with(
        &nan_block,
        steps,
        vocab,
        &mut scratch,
        &mut tokens,
        &eb(100.0, usize::MAX, ErrorProxy::Entropy),
    );
    assert_eq!(k, steps - 1, "the NaN depth is skipped");
    assert!(
        !scratch.committed_ids(k).contains(&2),
        "depth 2 never commits"
    );
}

// ── End-to-end: draft + EB commit ───────────────────────────────

struct FakeWeights;

struct FakeCtx {
    logits: Vec<f32>,
}

impl DflashCtx<FakeWeights> for FakeCtx {
    fn logits_slice(&self) -> &[f32] {
        &self.logits
    }

    fn hidden_state_slice(&self) -> &[f32] {
        &[]
    }

    fn apply_mtp_conditioning(
        &mut self,
        _weights: &FakeWeights,
        _mtp_ctx: &[f32],
        _n_embd: usize,
        _vocab_size: usize,
    ) {
    }
}

struct FakeCache;

impl DflashCache for FakeCache {
    fn reset(&mut self) {}
    fn invalidate_position(&mut self, _pos: usize, _kv_dim: usize) {}
    fn seed_layers(&mut self, _target_hidden: &[f32], _draft_kv_dim: usize) {}
}

#[test]
fn dflash_predict_eb_with_end_to_end() {
    let mut config = Config::draft();
    config.vocab_size = 16;
    config.draft_lookahead = 5;
    config.block_size = 64;
    config.temperature = 1.0;

    let mut ctx = FakeCtx {
        logits: vec![0.0; config.vocab_size],
    };
    let mut cache = FakeCache;
    let mut probs = vec![0.0; config.vocab_size];
    let mut marginals = vec![0.0; config.draft_lookahead * config.vocab_size];
    let mut eb_scratch = EbCommitScratch::default();
    let mut tokens = vec![0usize; config.draft_lookahead];

    // Smooth step-varying logits: every row is a genuine distribution, the
    // entropies differ per depth, and the commit prefix is deterministic.
    let forward = |ctx: &mut FakeCtx,
                   _w: &FakeWeights,
                   _c: &mut FakeCache,
                   _token: usize,
                   p: usize,
                   cfg: &Config| {
        for (i, l) in ctx.logits.iter_mut().enumerate() {
            *l = ((i as f32) * 0.37 + (p as f32) * 0.21).sin();
        }
        let _ = cfg;
    };

    let (steps, committed) = dflash_predict_eb_with(
        &mut ctx,
        &mut cache,
        &FakeWeights,
        forward,
        &mut probs,
        &mut marginals,
        &mut eb_scratch,
        &mut tokens,
        &eb(0.5, usize::MAX, ErrorProxy::Entropy),
        &config,
        3,
        0,
    );
    assert_eq!(steps, config.draft_lookahead);
    assert!(committed >= 1, "no-stall end to end");
    assert!(committed <= steps);
    // The committed depths (ascending after the emit) and their argmaxes:
    // tokens[slot] must be the argmax token of the slot-th committed depth.
    let depths: Vec<usize> = eb_scratch
        .committed_ids(committed)
        .iter()
        .map(|&d| d as usize)
        .collect();
    assert!(
        depths.iter().enumerate().all(|(s, &d)| s <= d),
        "ascending depth order"
    );
    for (slot, &d) in depths.iter().enumerate() {
        let row = &marginals[d * config.vocab_size..(d + 1) * config.vocab_size];
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row {d} is a normalized marginal");
        let argmax = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(tokens[slot], argmax, "depth {d} emits its argmax");
    }
}

#[test]
fn ddtree_eb_convenience_matches_into_form() {
    let (depths, vocab) = (3usize, 10usize);
    let mut rng = fastrand::Rng::with_seed(77);
    let rows: Vec<Vec<f32>> = (0..depths)
        .map(|_| {
            let raw: Vec<f32> = (0..vocab).map(|_| rng.f32()).collect();
            let sum: f32 = raw.iter().sum();
            raw.iter().map(|&p| p / sum).collect()
        })
        .collect();
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let config = test_config(vocab, 16);
    let cfg = eb(0.4, usize::MAX, ErrorProxy::Entropy);

    let tree = build_dd_tree_eb(&refs, &config, &NoPruner, &cfg);
    let mut scratch = EbBuildScratch::default();
    let borrowed = build_dd_tree_eb_into(&mut scratch, &refs, &config, &NoPruner, &cfg);
    assert_eq!(tree.len(), borrowed.len());
    assert!(
        tree.iter()
            .zip(borrowed)
            .all(|(a, b)| a.depth == b.depth && a.token_idx == b.token_idx)
    );
}
