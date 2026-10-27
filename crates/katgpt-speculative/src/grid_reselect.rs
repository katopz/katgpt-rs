//! Grid re-selection — prefix-first re-selection over the factorized block
//! drafter's OWN top-K candidate grid (Issue 966; the DCOCO-class corrector,
//! zero training).
//!
//! # The mechanism (Issue 966, from DCOCO's evidence chain minus the head)
//!
//! 1. **Coverage premise** — a parallel drafter's top-K pool contains the
//!    verifier-committed token far more often than its argmax does; the grid
//!    has the right tokens, per-slot argmax fails to select them. (The
//!    premise READ is [`crate::correlation_budget::TopKPoolCoverage`]'s on
//!    the lane posture — not executed here; see the issue.)
//! 2. **Selection law** — under prefix acceptance, prefix-first greedy beats
//!    aggregate-path selection at equal scores.
//! 3. **Weak-signal robustness** — a candidate-blind statistical transition
//!    prior still beats the uncorrected drafter; the sparse bigram table is
//!    our modelless stand-in for the learned corrector.
//!
//! # What is NEW here vs the sibling lanes (substrate-first record)
//!
//! The per-position selection LAW is not new: it is
//! [`crate::pair_select::pick_pair_scored`] verbatim (Issue 671 / Research
//! 490 — `ln U + λ·ln P`, first strict max, λ = 0 bit-identical to
//! argmax-of-marginals, missing pairs floored at
//! [`crate::pair_select::MISSING_PAIR_LOGP`]). The NEW part is the LATTICE
//! SOURCE: `pair_select` propagates its own marginals from the bigram table
//! alone (the drafter-substitute lane); THIS module restricts the FACTORIZED
//! BLOCK DRAFTER's own marginal output
//! ([`crate::dflash::dflash_predict_with`]'s step-major `marginals_flat`) to
//! per-slot top-K pools and re-selects WITHIN them — the drafter proposes,
//! the correction only changes WHICH pool tokens are proposed. Re-selection
//! is lossless by construction: the block still passes standard speculative
//! verification.
//!
//! # Two documented deviations from the issue text (both recorded in-issue)
//!
//! 1. **Missing pairs use the `MISSING_PAIR_LOGP` floor, not a zero
//!    contribution.** The issue prose says absent pairs "get zero bigram
//!    contribution", but a zero contribution is provably inverted at λ > 0:
//!    a present pair contributes `λ·ln P < 0` (a penalty) while an absent
//!    pair would contribute 0 — the walk would AVOID table successors. The
//!    floor (shared with `pair_select`) makes hits relatively favored: the
//!    coherent sparse-table discipline. The issue's own hit-rate column
//!    adjudicates the sparsity hazard either way.
//! 2. **`U` is the drafter's marginal PROBABILITY, not the raw logit.** The
//!    shared cores do not persist pre-softmax logits (re-shaping
//!    `dflash_predict_with` for a scale choice is churn the calibration
//!    absorbs), and the selection law consumes `ln U` — so λ blends two
//!    log-probability terms at a common scale. The issue's "λ calibrates the
//!    SCALE between a logit and a log-probability" concern is subsumed; the
//!    sweep still calibrates λ itself.
//!
//! # Zero-alloc + determinism
//!
//! Both functions take caller-owned buffers only — no allocation in steady
//! state (G4, pinned by the `grid_reselect_alloc_check` test binary). Pools
//! are sorted `(score desc, token asc)`, the walk takes the first strict max
//! via the shared core, and `λ = 0` reduces bit-for-bit to per-slot argmax
//! of `marginals_flat` (G1 property, pinned by `lambda_zero_equals_argmax`).

use crate::bigram_markov::BigramMarkovTable;
use crate::pair_select::pick_pair_scored;
use katgpt_types::simd::{
    BLOCK_MAX_SUB_WIDTH, BLOCK_MAX_WIDTH, simd_block_max_256_sub64, simd_block_max_256_sub64_into,
    simd_first_ge_f32, simd_max_f32,
};

/// Block width for the full-pool skip in [`topk_pools_into`]: once the pool
/// is full, a block is admitted to the per-element path only when its SIMD
/// max beats the pool tail (Issue 966 G2 — the per-element tail compare alone
/// measured 2.7–3.0× the stack's SIMD argmax). 256 f32 = 1 KiB: long enough
/// to amortise the reduction call, short enough that one hot token does not
/// drag a large span through the scalar path.
const SKIP_BLOCK: usize = BLOCK_MAX_WIDTH;

/// Second-level width inside an admitted [`SKIP_BLOCK`] (8 NEON / 4 AVX2
/// vectors): the scalar insert path only ever sees the 32-wide spans whose
/// max beats the tail.
const SKIP_SUB: usize = 32;

/// Default per-slot pool width — DCOCO's K scale, and the bigram table's
/// own default `top_m` (Issue 659); keeping the two at the same scale makes
/// the K×K adjacent-pair hit rate meaningful.
pub const DEFAULT_TOP_K: usize = 16;

/// Extract per-slot top-K pools from step-major drafter marginals.
///
/// `marginals_flat` is `[steps * vocab_size]`, step `p` occupying
/// `[p * vocab_size .. (p + 1) * vocab_size]` — exactly the layout
/// [`crate::dflash::dflash_predict_with`] writes. `pools` is a flat
/// `[steps * slot_len]` `(token, score)` buffer with
/// `slot_len = min(k, vocab_size)`; slot `p` occupies
/// `pools[p * slot_len .. (p + 1) * slot_len]`, written sorted
/// `(score desc, token asc)` — stable: equal scores keep the smaller token
/// first, so the pool's head IS per-slot argmax and the λ = 0 walk reduces
/// to per-slot argmax bit-for-bit.
///
/// Returns the number of pool slots populated (`steps`, clamped to what
/// `marginals_flat` and `pools` can hold). Scores are copied bit-exact —
/// no transform — so downstream comparisons are comparisons of the
/// drafter's own numbers.
///
/// Zero-alloc: bounded insertion over a caller-owned buffer. Whole
/// [`SKIP_BLOCK`]-wide blocks that cannot reach the pool are skipped — exact,
/// not approximate (the pool is bit-identical to the per-element scan; see
/// [`scan_row_floor`] and [`scan_row_tail`] for the tie + NaN arguments).
pub fn topk_pools_into(
    marginals_flat: &[f32],
    steps: usize,
    vocab_size: usize,
    k: usize,
    pools: &mut [(u32, f32)],
) -> usize {
    if steps == 0 || vocab_size == 0 || k == 0 {
        return 0;
    }
    let slot_len = k.min(vocab_size);
    let max_steps = steps
        .min(marginals_flat.len() / vocab_size)
        .min(pools.len() / slot_len);
    // The floor path needs a NaN-PROPAGATING block max to route NaN rows to
    // the tail path (the NEON kernel is one; x86's `max_ps` is not).
    let floor_path = cfg!(target_arch = "aarch64") && slot_len <= FLOOR_MAX_POOL;
    for p in 0..max_steps {
        let margs = &marginals_flat[p * vocab_size..p * vocab_size + vocab_size];
        let pool = &mut pools[p * slot_len..p * slot_len + slot_len];
        if !(floor_path && scan_row_floor(margs, pool)) {
            scan_row_tail(margs, pool);
        }
    }
    max_steps
}

/// Floor-path capacity: 1024 blocks × [`SKIP_BLOCK`] covers vocabs to
/// 262 144 (20 KiB of stack for block + sub-block maxima); wider rows take
/// the tail path.
const FLOOR_MAX_BLOCKS: usize = 1024;

/// Sub-blocks per [`SKIP_BLOCK`] (64 floats each).
const SUBS: usize = SKIP_BLOCK / BLOCK_MAX_SUB_WIDTH;

/// Widest pool [`scan_row_floor`] ranks (a stack top-list).
const FLOOR_MAX_POOL: usize = 64;

/// Block-floor row scan (Issue 967 follow-up). Pass 1 computes every block's
/// max + its 64-wide sub-block maxima and, fused into the same loop, ranks
/// the block maxima: τ = the `slot_len`-th largest. τ lower-bounds the final
/// pool — each block max is a distinct element, so at least `slot_len`
/// elements are `>= τ` — so an element `< τ` is never in the final pool, and
/// the pool over the offered SUPERSET of the top-`slot_len` is the same pool
/// (and fills). Once full, an element `<= tail` never enters (every pooled
/// token has a smaller index, so a tie loses). Pass 2 rescans only the
/// sub-blocks clearing both bounds.
///
/// Returns `false` without a usable pool when the row is out of range (more
/// than [`FLOOR_MAX_BLOCKS`] blocks) or a block max is NaN — a NaN appended
/// while the pool fills is never evicted, so dropping its block would change
/// the pool; the caller re-runs the row on the tail path. A whole-row floor
/// measured ~25% faster than a per-32-block running floor at a 32 000 vocab:
/// the looser prefix floor admits more candidate blocks than L1 residency
/// saves.
fn scan_row_floor(margs: &[f32], pool: &mut [(u32, f32)]) -> bool {
    let slot_len = pool.len();
    let nblk = margs.len().div_ceil(SKIP_BLOCK);
    if nblk > FLOOR_MAX_BLOCKS {
        return false;
    }
    let mut bmax = [0f32; FLOOR_MAX_BLOCKS];
    let mut smax = [0f32; FLOOR_MAX_BLOCKS * SUBS];
    // top[..slot_len] descending; the last entry is the running τ.
    let mut top = [f32::NEG_INFINITY; FLOOR_MAX_POOL];
    let (full, rest) = margs.as_chunks::<SKIP_BLOCK>();
    for (b, block) in full.iter().enumerate() {
        let (m, s4) = simd_block_max_256_sub64(block);
        bmax[b] = m;
        smax[b * SUBS..b * SUBS + SUBS].copy_from_slice(&s4);
        if m > top[slot_len - 1] {
            let mut i = slot_len - 1;
            while i > 0 && top[i - 1] < m {
                top[i] = top[i - 1];
                i -= 1;
            }
            top[i] = m;
        } else if m.is_nan() {
            return false;
        }
    }
    if !rest.is_empty() {
        let b = full.len();
        simd_block_max_256_sub64_into(
            rest,
            &mut bmax[b..b + 1],
            &mut smax[b * SUBS..b * SUBS + SUBS],
        );
        let m = bmax[b];
        if m.is_nan() {
            return false;
        }
        if m > top[slot_len - 1] {
            let mut i = slot_len - 1;
            while i > 0 && top[i - 1] < m {
                top[i] = top[i - 1];
                i -= 1;
            }
            top[i] = m;
        }
    }
    let tau = top[slot_len - 1];
    let mut len = 0usize;
    for (b, &m) in bmax[..nblk].iter().enumerate() {
        if m < tau || (len == slot_len && m <= pool[slot_len - 1].1) {
            continue;
        }
        let base = b * SKIP_BLOCK;
        let block = &margs[base..(base + SKIP_BLOCK).min(margs.len())];
        for (q, sub) in block.chunks(BLOCK_MAX_SUB_WIDTH).enumerate() {
            let mut thr = match len == slot_len {
                true => pool[slot_len - 1].1.max(tau),
                false => tau,
            };
            if smax[b * SUBS + q] < thr {
                continue;
            }
            let sub_base = base + q * BLOCK_MAX_SUB_WIDTH;
            let mut i = 0usize;
            while let Some(off) = simd_first_ge_f32(&sub[i..], thr) {
                let j = i + off;
                len = insert_candidate(pool, len, slot_len, (sub_base + j) as u32, sub[j]);
                if len == slot_len {
                    thr = pool[slot_len - 1].1.max(tau);
                }
                i = j + 1;
            }
        }
    }
    debug_assert_eq!(len, slot_len, "floor path under-filled the pool");
    true
}

/// Tail-skip row scan (the Issue-966 G2 path; non-aarch64, NaN rows, wide
/// pools). Once the pool is full, a later index can never win a tie against
/// the tail (every pooled token has a SMALLER index than `j`), so the entry
/// rule reduces to `s > tail` and a whole block whose max is `<= tail` cannot
/// change the pool. `<=` (not `!(>)`) keeps a NaN-propagating max (NEON
/// `vmaxq`) on the exact per-element path; a NaN-ignoring max is safe too
/// (NaN never enters a full pool).
fn scan_row_tail(margs: &[f32], pool: &mut [(u32, f32)]) {
    let slot_len = pool.len();
    let mut len = 0usize;
    for (bi, block) in margs.chunks(SKIP_BLOCK).enumerate() {
        if len == slot_len && simd_max_f32(block) <= pool[slot_len - 1].1 {
            continue;
        }
        // Admitted: re-check each SKIP_SUB-wide sub-block against the
        // CURRENT tail (it rises as the block inserts), so one hot token
        // pulls 32 elements through the scalar path, not 256.
        for (si, sub) in block.chunks(SKIP_SUB).enumerate() {
            if len == slot_len && simd_max_f32(sub) <= pool[slot_len - 1].1 {
                continue;
            }
            let sub_base = bi * SKIP_BLOCK + si * SKIP_SUB;
            for (off, &s) in sub.iter().enumerate() {
                len = insert_candidate(pool, len, slot_len, (sub_base + off) as u32, s);
            }
        }
    }
    debug_assert_eq!(len, slot_len, "tail path under-filled the pool");
}

/// One candidate into a `(score desc, token asc)`-sorted pool holding `len`
/// of `slot_len` entries; returns the new length.
#[inline(always)]
fn insert_candidate(
    pool: &mut [(u32, f32)],
    mut len: usize,
    slot_len: usize,
    j: u32,
    s: f32,
) -> usize {
    if len == slot_len {
        // Full: the candidate enters only by beating the tail
        // (score desc, then token asc). Otherwise skip.
        let tail = pool[slot_len - 1];
        let better = s > tail.1 || (s == tail.1 && j < tail.0);
        if !better {
            return len;
        }
        len -= 1; // evict the tail, fall through to the insert
    }
    // len < slot_len: every candidate belongs in the pool (append at the
    // tail when it is worse-or-equal to everything held). Insert position:
    // first entry strictly worse than (s, j) under (score desc, token asc).
    let at = pool[..len]
        .iter()
        .position(|&(ti, ts)| s > ts || (s == ts && j < ti))
        .unwrap_or(len);
    // Make room: rotate the tail right by one (the old tail falls off when
    // full), then write the candidate at `at`.
    pool[at..=len].rotate_right(1);
    pool[at] = (j, s);
    len + 1
}

/// Prefix-first scored walk over the extracted pools (Issue 966's
/// selection law, via the shared [`crate::pair_select::pick_pair_scored`]).
///
/// From `a = prev`, slot `p` picks
/// `argmax_b [ ln U_p(b) + λ · ln P(b|a) ]` over slot `p`'s pool and `a`
/// becomes the pick. `λ = 0` is bit-identical to per-slot argmax of the
/// source marginals (pools are `(score desc, token asc)`-sorted; the shared
/// core's first-strict-max keeps the head). Missing pairs score at
/// [`crate::pair_select::MISSING_PAIR_LOGP`] — the documented floor
/// deviation, module header §deviations.
///
/// `pools` is `steps * slot_len` entries as [`topk_pools_into`] wrote them
/// (pass the SAME `slot_len`). Returns the number of tokens written to
/// `out` (clamped to `steps` and `out.len()`). Zero-alloc.
pub fn reselect_block_into(
    table: &BigramMarkovTable,
    prev: u32,
    steps: usize,
    slot_len: usize,
    pools: &[(u32, f32)],
    lambda: f32,
    out: &mut [u32],
) -> usize {
    if steps == 0 || slot_len == 0 {
        return 0;
    }
    let written = steps.min(out.len()).min(pools.len() / slot_len);
    let mut a = prev;
    for p in 0..written {
        let pool = &pools[p * slot_len..p * slot_len + slot_len];
        a = pick_pair_scored(pool, a, table, lambda);
        if a == u32::MAX {
            break;
        }
        out[p] = a;
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bigram_markov::BigramMarkovBuilder;

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

    fn argmax_per_slot(m: &[f32], steps: usize, vocab: usize) -> Vec<u32> {
        (0..steps)
            .map(|p| {
                let row = &m[p * vocab..p * vocab + vocab];
                let mut best = 0usize;
                for j in 1..vocab {
                    if row[j] > row[best] {
                        best = j;
                    }
                }
                best as u32
            })
            .collect()
    }

    /// The pre-G2 per-element scan, kept as the oracle the block skip must
    /// reproduce bit-for-bit (ties, NaN, ragged tail block included).
    fn pools_per_element(m: &[f32], steps: usize, vocab: usize, k: usize) -> Vec<(u32, f32)> {
        let slot_len = k.min(vocab);
        let mut pools = vec![(0u32, 0f32); steps * slot_len];
        for p in 0..steps {
            let pool = &mut pools[p * slot_len..(p + 1) * slot_len];
            let mut len = 0;
            for (j, &s) in m[p * vocab..(p + 1) * vocab].iter().enumerate() {
                len = insert_candidate(pool, len, slot_len, j as u32, s);
            }
        }
        pools
    }

    #[test]
    fn block_skip_is_bit_identical_to_the_per_element_scan() {
        let mut rng = fastrand::Rng::with_seed(9662);
        // vocab not a multiple of SKIP_BLOCK; k > vocab; k == vocab; then
        // rows with many more blocks than k (the block floor τ binds — a
        // strong one at k=4), and one past FLOOR_MAX_BLOCKS (tail fallback).
        for &(steps, vocab, k) in &[
            (4usize, 1000usize, 16usize),
            (3, 777, 8),
            (2, 10, 16),
            (2, 16, 16),
            (3, 32_077, 16),
            (2, 8192, 4),
            (1, FLOOR_MAX_BLOCKS * SKIP_BLOCK + 65, 16),
        ] {
            let shapes: [fn(&mut fastrand::Rng, usize) -> f32; 4] = [
                |r, _| r.f32(),
                // heavy ties: 4 distinct levels
                |r, _| f32::from(r.u8(0..4)) * 0.25,
                // peaked (softmax-like): one hot token per 300
                |r, j| if j % 300 == 7 { 0.9 } else { r.f32() * 1e-4 },
                // NaN sprinkled
                |r, _| if r.u8(0..20) == 0 { f32::NAN } else { r.f32() },
            ];
            for make in &shapes {
                let m: Vec<f32> = (0..steps * vocab)
                    .map(|i| make(&mut rng, i % vocab))
                    .collect();
                let slot_len = k.min(vocab);
                let mut got = vec![(0u32, 0f32); steps * slot_len];
                topk_pools_into(&m, steps, vocab, k, &mut got);
                let want = pools_per_element(&m, steps, vocab, k);
                let bits =
                    |v: &[(u32, f32)]| v.iter().map(|&(t, s)| (t, s.to_bits())).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&want), "shape ({steps},{vocab},{k})");
            }
        }
    }

    #[test]
    fn pools_are_exact_topk_stable() {
        // Brute-force reference: sort by (score desc, token asc), take k.
        let mut rng = fastrand::Rng::with_seed(966);
        let (steps, vocab, k) = (5usize, 40usize, 6usize);
        let m: Vec<f32> = (0..steps * vocab).map(|_| rng.f32()).collect();
        let mut pools = vec![(0u32, 0f32); steps * k];
        let n = topk_pools_into(&m, steps, vocab, k, &mut pools);
        assert_eq!(n, steps);
        for p in 0..steps {
            let row = &m[p * vocab..p * vocab + vocab];
            let mut ref_sorted: Vec<(u32, f32)> = row
                .iter()
                .enumerate()
                .map(|(j, &s)| (j as u32, s))
                .collect();
            ref_sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            assert_eq!(
                &pools[p * k..p * k + k],
                &ref_sorted[..k],
                "pool mismatch at slot {p}"
            );
        }
    }

    #[test]
    fn k_one_pool_is_the_argmax() {
        let mut rng = fastrand::Rng::with_seed(967);
        let (steps, vocab) = (3usize, 25usize);
        let m: Vec<f32> = (0..steps * vocab).map(|_| rng.f32()).collect();
        let mut pools = vec![(0u32, 0f32); steps];
        topk_pools_into(&m, steps, vocab, 1, &mut pools);
        let argmax = argmax_per_slot(&m, steps, vocab);
        for p in 0..steps {
            assert_eq!(pools[p].0, argmax[p], "k=1 pool head must be the argmax");
        }
    }

    #[test]
    fn k_over_vocab_clamps_and_ties_keep_smaller_token() {
        // All-equal scores: (score desc, token asc) → the pool is tokens
        // 0..slot_len in ascending order, and k > vocab clamps slot_len.
        let vocab = 7usize;
        let m = vec![0.5f32; 2 * vocab];
        let mut pools = vec![(0u32, 0f32); 2 * 10];
        let n = topk_pools_into(&m, 2, vocab, 10, &mut pools);
        assert_eq!(n, 2);
        for p in 0..2 {
            for (i, cell) in pools[p * 7..p * 7 + 7].iter().enumerate() {
                assert_eq!(cell.0, i as u32, "tie must keep smaller token first");
                assert_eq!(cell.1, 0.5);
            }
        }
        // The walk over the clamped pools (slot_len = 7) is still the argmax.
        let table = table_from_pairs(&[], 7, 4);
        let mut out = vec![0u32; 2];
        reselect_block_into(&table, 0, 2, 7, &pools, 0.0, &mut out);
        assert_eq!(out, argmax_per_slot(&m, 2, vocab));
    }

    #[test]
    fn lambda_zero_equals_per_slot_argmax() {
        // G1 property: the walk at λ = 0 is per-slot argmax of the source
        // marginals, bit-for-bit, across shapes and seeded draws.
        let table = table_from_pairs(&[], 16, 8); // no evidence — floor everywhere
        let mut rng = fastrand::Rng::with_seed(968);
        for &(steps, vocab, k) in &[(4usize, 32usize, 8usize), (2, 9, 16), (6, 64, 3)] {
            let m: Vec<f32> = (0..steps * vocab).map(|_| rng.f32()).collect();
            let slot_len = k.min(vocab);
            let mut pools = vec![(0u32, 0f32); steps * slot_len];
            topk_pools_into(&m, steps, vocab, k, &mut pools);
            let mut out = vec![0u32; steps];
            reselect_block_into(&table, 0, steps, slot_len, &pools, 0.0, &mut out);
            assert_eq!(
                out,
                argmax_per_slot(&m, steps, vocab),
                "steps={steps} vocab={vocab} k={k}"
            );
        }
    }

    /// The pair_select divergent fixture, re-seated on the DRAFTER side:
    /// slot-0 argmax is `2`; slot 1's marginal argmax is `3`, but the table
    /// says `P(5|2) = 0.75 > P(3|2) = 0.25` — and `5` IS in the slot-1 pool
    /// (mid-rank). λ = 0 walks `[2, 3]`; λ = 1 flips to `[2, 5]`.
    /// (Marginals hand-built: no forward pass needed — the pools are what
    /// the module consumes.)
    #[test]
    fn pair_term_flips_within_drafter_grid() {
        let table = table_from_pairs(&[(2, 5, 3), (2, 3, 1)], 8, 8);
        let mut m = vec![0.0f32; 2 * 8];
        m[2] = 0.9;
        m[0] = 0.1;
        m[8 + 3] = 0.55;
        m[8 + 5] = 0.30;
        m[8 + 1] = 0.15;
        let (steps, vocab, k) = (2usize, 8usize, 3usize);
        let mut pools = vec![(0u32, 0f32); steps * k];
        topk_pools_into(&m, steps, vocab, k, &mut pools);
        // Slot 1's pool: 3 (0.55), 5 (0.30), 1 (0.15) — 5 IS in the grid.
        assert_eq!(pools[k].0, 3);
        assert_eq!(pools[k + 1].0, 5);

        let mut out = vec![0u32; steps];
        reselect_block_into(&table, 0, steps, k, &pools, 0.0, &mut out);
        assert_eq!(out, vec![2, 3], "λ=0 must be the argmax path");

        reselect_block_into(&table, 0, steps, k, &pools, 1.0, &mut out);
        assert_eq!(
            out,
            vec![2, 5],
            "λ=1 must flip slot 1 to the table successor"
        );
    }

    #[test]
    fn hit_beats_bigger_u_miss_at_lambda_one() {
        // The floor's purpose, both directions: with prev = 1 (whose row
        // has evidence 1→0 at P = 1.0), slot 1's marginal ARGMAX is the
        // miss `4` — λ = 0 walks to it; λ = 1 flips to the HIT `0` despite
        // its smaller U. And a dominant-U pick with NO evidence still holds
        // (slot 0 keeps `1` at both λ).
        let table = table_from_pairs(&[(1, 0, 5)], 8, 8);
        let mut m = vec![0.0f32; 2 * 8];
        m[1] = 0.98; // slot 0 argmax = 1 (no 0→x evidence, floor everywhere)
        m[8 + 4] = 0.6; // slot 1 argmax = 4 (a miss)
        m[8] = 0.4; // slot 1's 0 IS a hit (P(0|1) = 1.0)
        let (steps, vocab, k) = (2usize, 8usize, 2usize);
        let mut pools = vec![(0u32, 0f32); steps * k];
        topk_pools_into(&m, steps, vocab, k, &mut pools);
        let mut out = vec![0u32; steps];

        reselect_block_into(&table, 0, steps, k, &pools, 0.0, &mut out);
        assert_eq!(out, vec![1, 4], "λ=0 must be the argmax path");

        reselect_block_into(&table, 0, steps, k, &pools, 1.0, &mut out);
        assert_eq!(
            out,
            vec![1, 0],
            "λ=1 must flip to the hit over the bigger-U miss"
        );
    }

    #[test]
    fn edges_zero_steps_and_degenerate_widths() {
        let table = table_from_pairs(&[], 4, 4);
        let mut pools = vec![(0u32, 0f32); 8];
        assert_eq!(topk_pools_into(&[], 0, 4, 2, &mut pools), 0);
        assert_eq!(topk_pools_into(&[0.0; 4], 1, 4, 0, &mut pools), 0);
        assert_eq!(topk_pools_into(&[0.0; 4], 1, 0, 2, &mut pools), 0);
        assert_eq!(
            reselect_block_into(&table, 0, 0, 2, &pools, 1.0, &mut []),
            0
        );
        assert_eq!(
            reselect_block_into(&table, 0, 2, 0, &pools, 1.0, &mut []),
            0
        );
        // steps clamp: marginals hold 2 steps, caller asks 4 → 2 populated.
        let m = [0.1f32, 0.9, 0.3, 0.7, 0.5, 0.5, 0.2, 0.8];
        let n = topk_pools_into(&m, 4, 4, 2, &mut pools);
        assert_eq!(n, 2, "steps must clamp to marginals_flat capacity");
        let mut out = vec![0u32; 8];
        let w = reselect_block_into(&table, 0, 2, 2, &pools, 0.0, &mut out);
        assert_eq!(w, 2);
        assert_eq!(&out[..2], &[1, 3]);
    }
}
