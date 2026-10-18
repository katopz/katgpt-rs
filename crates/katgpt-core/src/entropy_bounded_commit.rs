//! EB-Sampler-class entropy-bounded commitment (Issue 917 T1).
//!
//! Given the still-masked positions of a diffusion / drafter canvas, decide
//! HOW MANY to commit this pass. The rule is the EB-Sampler's (Ben-Hamu,
//! Gat, Severo, Nolte, Karrer — "Accelerated Sampling from Masked Diffusion
//! Models via Entropy Bounded Unmasking", arXiv:2505.24857, NeurIPS 2025,
//! Algorithm 1 / Eq. 8): sort the candidates by an error proxy (lowest
//! error first) and commit the LARGEST prefix `U` with
//!
//! ```text
//! Σ_{l∈U} H(l) − max_{l∈U} H(l) ≤ γ
//! ```
//!
//! — a bound on the joint-dependence error of committing `U` in parallel
//! under a factorized (per-position) proposal. Zero novelty is claimed: the
//! class is EB-Sampler / PC-Sampler / APD / the DiffusionGemma report's
//! `entropy budget b`. What this module adds is the zero-alloc, deterministic
//! transcription the workspace's commit paths (D2F τ_conf, DDTree width-k,
//! DFlash block commit) can share — Issue 917 T2/T3 wire it.
//!
//! **Shipped reference.** HF transformers carries the entropy-proxy form as
//! DiffusionGemma's `EntropyBoundSampler.accept_canvas` (default
//! `entropy_bound = 0.1`), written `cumsum(H_sorted) − H_sorted ≤ bound`;
//! `hf_entropy_bound_sampler_parity` pins exact agreement with that formula.
//! What this transcription adds over it: the confidence/margin proxies, a
//! strict tie order (torch's default sort is not stable), a per-pass cap with
//! partial selection, NaN handling, and zero allocation.
//!
//! **The EB form, not the essay's prefix form.** The Srivastava appendix
//! variant `Σ_{j<i} H_j ≤ τ` (riir-train Research 465) is a weaker
//! heuristic; the `− max` term is what makes the bound about JOINT error —
//! the single largest-entropy member is paid for by sampling it alone, which
//! a singleton commit does anyway.
//!
//! **No-stall by construction (G1).** A singleton prefix has residual
//! `H − H = 0 ≤ γ` for every `γ ≥ 0`, so any pass with at least one
//! finite-entropy candidate commits ≥ 1 position. The incumbent
//! per-position τ threshold ([`crate::set_diffusion_schedule`]'s
//! `confidence_threshold_eligible`) has no such floor — a flat
//! distribution everywhere commits nothing and burns the pass. This
//! property is EB-Sampler's, not ours; the tests assert the transcription
//! preserves it.
//!
//! **Monotone residual ⇒ one linear scan.** Appending `h ≥ 0` to a prefix
//! with running max `m` raises the residual by `min(h, m)` (if `h ≤ m`, `h`
//! joins the sum; if `h > m`, the old max leaves the max slot and joins the
//! sum). So the residual is non-decreasing along the order and the largest
//! admissible prefix ends at the first violation. The update is carried in
//! that incremental form, never as `sum − max`, so there is no cancellation
//! and the residual is exact for the summands it adds.
//!
//! **Determinism.** Candidates are ordered by `(key is NaN, key, index)`
//! under `f32::total_cmp` — a strict total order, so the unstable sort's
//! output is unique and no RNG is involved. NaN keys sort last. A
//! non-finite ENTROPY terminates the prefix (an incomparable distribution
//! is never committed, mirroring the incumbent's `NaN ≥ τ` = false), and
//! tiny negative entropies from the `ln_z − mean_shift` kernel are clamped
//! to 0 so the monotonicity argument holds.
//!
//! Zero allocation (G4): the caller owns the candidate buffer; the function
//! sorts it in place and returns the committed prefix length.
//!
//! Issue 917 T1 (2026-10-03): EB-Sampler-class entropy-bounded commitment
//! (arXiv:2505.24857) — the largest error-proxy-ordered prefix with
//! Σ H − max H ≤ γ; zero-alloc, deterministic, no-stall by construction.
//! Opt-in pending the T2/T3 lane A/B (G2 NFE at matched quality).

use crate::simd::logsumexp_parts;
use std::cmp::Ordering;

/// Which error proxy orders the candidates (EB-Sampler §4: entropy,
/// confidence, margin — all three are "lower key = commit first").
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ErrorProxy {
    /// Shannon entropy of the position's categorical (nats).
    #[default]
    Entropy,
    /// Top-1 probability (higher is safer; keyed as `−p₁`).
    Confidence,
    /// Top-1 minus top-2 probability (higher is safer; keyed as `−(p₁−p₂)`).
    Margin,
}

/// Per-position statistics one commit decision needs, from one logits row.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PositionStats {
    /// Entropy in nats (≥ 0 up to kernel rounding; NaN for a NaN row).
    pub entropy: f32,
    /// Top-1 probability.
    pub top1: f32,
    /// Top-1 minus top-2 probability (`top1` for a single-outcome row).
    pub margin: f32,
}

impl PositionStats {
    /// The ordering key under `proxy` — ascending key = commit first.
    #[inline]
    pub fn key(&self, proxy: ErrorProxy) -> f32 {
        match proxy {
            ErrorProxy::Entropy => self.entropy,
            ErrorProxy::Confidence => -self.top1,
            ErrorProxy::Margin => -self.margin,
        }
    }
}

/// Entropy / top-1 / margin of one position's softmax over `logits`.
///
/// Two passes: the shared [`logsumexp_parts`] kernel (the same one
/// `regime_probe::conditional_entropy_nats` and `breakeven` cross-entropy
/// use — `H = ln_z − mean_shift`, `p₁ = e^{−ln_z}` after the max shift) plus
/// one top-2 scan for the margin. Zero alloc. An empty row returns all-NaN
/// stats (no distribution — never committed).
#[inline]
pub fn position_stats(logits: &[f32]) -> PositionStats {
    if logits.is_empty() {
        return PositionStats {
            entropy: f32::NAN,
            top1: f32::NAN,
            margin: f32::NAN,
        };
    }
    let (max, ln_z, mean_shift) = logsumexp_parts(logits);
    // Second-largest logit: skip exactly one occurrence of the max so a tied
    // top pair yields margin 0.
    let mut seen_max = false;
    let mut second = f32::NEG_INFINITY;
    for &x in logits {
        if !seen_max && x == max {
            seen_max = true;
            continue;
        }
        if x > second {
            second = x;
        }
    }
    let top1 = (-ln_z).exp();
    let top2 = (second - max - ln_z).exp();
    PositionStats {
        entropy: ln_z - mean_shift,
        top1,
        margin: top1 - top2,
    }
}

/// Batch [`position_stats`] over a row-major `positions × vocab` logits
/// block into caller-owned `out` (len = `positions`). Zero alloc; rows in
/// index order (bit-identical to per-row calls).
pub fn position_stats_into(
    logits: &[f32],
    positions: usize,
    vocab: usize,
    out: &mut [PositionStats],
) {
    debug_assert_eq!(logits.len(), positions * vocab, "flat logits shape");
    debug_assert_eq!(out.len(), positions, "one stats slot per position");
    for (p, slot) in out.iter_mut().enumerate() {
        *slot = position_stats(&logits[p * vocab..(p + 1) * vocab]);
    }
}

/// Commit-policy parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EbCommitConfig {
    /// Joint-error budget γ (nats). `0` commits exactly the singleton
    /// unless further candidates have zero entropy; larger γ commits more
    /// per pass. The DiffusionGemma report's default budget is `0.1`.
    pub gamma: f32,
    /// Hard per-pass cap (`usize::MAX` = uncapped). A cap below the
    /// candidate count also lets the ordering use a partial selection.
    pub max_commit: usize,
    /// Which proxy orders the candidates.
    pub proxy: ErrorProxy,
}

impl Default for EbCommitConfig {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            max_commit: usize::MAX,
            proxy: ErrorProxy::Entropy,
        }
    }
}

#[inline]
fn order_cmp(key: &[f32], a: u32, b: u32) -> Ordering {
    let (ka, kb) = (key[a as usize], key[b as usize]);
    ka.is_nan()
        .cmp(&kb.is_nan())
        .then(ka.total_cmp(&kb))
        .then(a.cmp(&b))
}

/// Order `candidates` by `(NaN-last, key, index)` and return the length of
/// the largest prefix whose EB residual `Σ H − max H` stays `≤ gamma`.
///
/// - `candidates`: the still-masked position indices (caller-owned; sorted
///   in place — on return `candidates[..k]` is the commit set in commit
///   order and the tail is the rest in key order, or, when `max_commit`
///   triggered a partial selection, in unspecified order).
/// - `entropy[p]` / `key[p]`: indexed by POSITION (not by candidate slot);
///   pass the same slice twice for the entropy proxy.
/// - For any `gamma ≥ 0` the singleton floor holds (a singleton's residual
///   is exactly `0`). A NaN or negative `gamma` commits nothing — a garbage
///   config never commits.
///
/// Zero alloc, deterministic, O(n log n) (O(n + m log m) under a cap `m`).
pub fn entropy_bounded_commit(
    candidates: &mut [u32],
    entropy: &[f32],
    key: &[f32],
    gamma: f32,
    max_commit: usize,
) -> usize {
    let n = candidates.len();
    let cap = max_commit.min(n);
    // NaN `gamma` is rejected too (`is_nan() || < 0`).
    if cap == 0 || gamma.is_nan() || gamma < 0.0 {
        return 0;
    }
    if cap < n {
        // Partial selection: the `cap` smallest under the total order go to
        // the front (`select_nth_unstable_by` partitions around index cap−1),
        // then only that head is sorted.
        candidates.select_nth_unstable_by(cap - 1, |&a, &b| order_cmp(key, a, b));
        candidates[..cap].sort_unstable_by(|&a, &b| order_cmp(key, a, b));
    } else {
        candidates.sort_unstable_by(|&a, &b| order_cmp(key, a, b));
    }

    let mut residual = 0.0f32;
    let mut max_h = 0.0f32;
    let mut k = 0usize;
    for &p in &candidates[..cap] {
        let h = entropy[p as usize];
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
    k
}

/// [`entropy_bounded_commit`] driven by precomputed [`PositionStats`]
/// (indexed by position) and an [`EbCommitConfig`]. `entropy_scratch` and
/// `key_scratch` must each be at least as long as `stats` — they receive the
/// entropies and proxy keys once, so the sort compares plain `f32` slices
/// rather than re-deriving keys per comparison. Zero alloc.
pub fn entropy_bounded_commit_stats(
    candidates: &mut [u32],
    stats: &[PositionStats],
    entropy_scratch: &mut [f32],
    key_scratch: &mut [f32],
    cfg: &EbCommitConfig,
) -> usize {
    debug_assert!(entropy_scratch.len() >= stats.len());
    debug_assert!(key_scratch.len() >= stats.len());
    for (i, s) in stats.iter().enumerate() {
        entropy_scratch[i] = s.entropy;
        key_scratch[i] = s.key(cfg.proxy);
    }
    let n = stats.len();
    entropy_bounded_commit(
        candidates,
        &entropy_scratch[..n],
        &key_scratch[..n],
        cfg.gamma,
        cfg.max_commit,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn commit(h: &[f32], gamma: f32) -> Vec<u32> {
        let mut c: Vec<u32> = (0..h.len() as u32).collect();
        let k = entropy_bounded_commit(&mut c, h, h, gamma, usize::MAX);
        c.truncate(k);
        c
    }

    #[test]
    fn no_stall_all_equal() {
        // Flat high entropy everywhere: the incumbent τ rule commits nothing;
        // EB commits the singleton (lowest index on the tie).
        let h = [2.0f32; 8];
        assert_eq!(commit(&h, 0.1), vec![0]);
        assert_eq!(commit(&h, 0.0), vec![0]);
    }

    #[test]
    fn no_stall_one_dominant_and_degenerate_zero() {
        let h = [3.0, 0.0, 3.0, 3.0];
        // Zero-entropy position first, then one 3.0 (residual = min(3,0)=0).
        assert_eq!(commit(&h, 0.0), vec![1, 0]);
        let z = [0.0f32; 5];
        assert_eq!(commit(&z, 0.0), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn residual_bound_is_sum_minus_max() {
        let h = [0.05, 0.02, 0.4, 0.03];
        // Ascending: 0.02(1) 0.03(3) 0.05(0) 0.4(2).
        // Prefix residuals: 0, 0.02, 0.05, 0.10 (sum 0.5 − max 0.4).
        assert_eq!(commit(&h, 0.04), vec![1, 3]);
        assert_eq!(commit(&h, 0.05), vec![1, 3, 0]);
        assert_eq!(commit(&h, 0.10), vec![1, 3, 0, 2]);
        // Brute-force cross-check of the incremental residual.
        let sum: f32 = h.iter().sum();
        let max = h.iter().cloned().fold(0.0f32, f32::max);
        assert!(((sum - max) - 0.10).abs() < 1e-6);
    }

    #[test]
    fn residual_monotone_matches_bruteforce_random() {
        let mut rng = fastrand::Rng::with_seed(917);
        let mut c: Vec<u32> = Vec::new();
        for _ in 0..500 {
            let n = rng.usize(1..40);
            let h: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0).collect();
            let gamma = rng.f32() * 3.0;
            c.clear();
            c.extend(0..n as u32);
            let k = entropy_bounded_commit(&mut c, &h, &h, gamma, usize::MAX);
            assert!(k >= 1, "no-stall");
            // Largest admissible prefix under brute-force sum − max.
            let resid = |m: usize| {
                let s: f32 = c[..m].iter().map(|&p| h[p as usize]).sum();
                let mx = c[..m].iter().map(|&p| h[p as usize]).fold(0.0f32, f32::max);
                s - mx
            };
            assert!(resid(k) <= gamma + 1e-5);
            if k < n {
                assert!(resid(k + 1) > gamma - 1e-5);
            }
        }
    }

    #[test]
    fn cap_and_partial_selection_agree_with_full_sort() {
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..200 {
            let n = rng.usize(2..64);
            let h: Vec<f32> = (0..n).map(|_| (rng.u8(..) % 8) as f32 * 0.01).collect();
            let cap = rng.usize(1..n + 1);
            let mut full: Vec<u32> = (0..n as u32).collect();
            let kf = entropy_bounded_commit(&mut full, &h, &h, 0.2, usize::MAX);
            let mut part: Vec<u32> = (0..n as u32).collect();
            let kp = entropy_bounded_commit(&mut part, &h, &h, 0.2, cap);
            assert_eq!(kp, kf.min(cap));
            assert_eq!(part[..kp], full[..kp]);
        }
    }

    #[test]
    fn nan_handling_never_commits_garbage() {
        let h = [f32::NAN, 0.1, 0.2];
        // NaN key sorts last; NaN entropy stops the prefix before it.
        assert_eq!(commit(&h, 10.0), vec![1, 2]);
        assert_eq!(commit(&[f32::NAN], 10.0), Vec::<u32>::new());
        assert_eq!(commit(&[0.1, 0.2], f32::NAN), Vec::<u32>::new());
        assert_eq!(commit(&[0.1, 0.2], -1.0), Vec::<u32>::new());
        assert_eq!(commit(&[], 1.0), Vec::<u32>::new());
    }

    #[test]
    fn candidate_subset_only() {
        // Positions 0 and 2 are already committed — only 1, 3, 4 compete.
        let h = [0.0, 0.5, 0.0, 0.1, 0.2];
        let mut c = vec![1u32, 3, 4];
        let k = entropy_bounded_commit(&mut c, &h, &h, 0.1, usize::MAX);
        assert_eq!(&c[..k], &[3, 4]);
    }

    #[test]
    fn position_stats_known_answers() {
        let v = 16usize;
        let uniform = vec![0.0f32; v];
        let s = position_stats(&uniform);
        assert!((s.entropy - (v as f32).ln()).abs() < 1e-4);
        assert!((s.top1 - 1.0 / v as f32).abs() < 1e-5);
        assert!(s.margin.abs() < 1e-6, "tied top pair → margin 0");

        let mut peaked = vec![-1e4f32; v];
        peaked[3] = 0.0;
        let s = position_stats(&peaked);
        assert!(s.entropy.abs() < 1e-4);
        assert!((s.top1 - 1.0).abs() < 1e-5);
        assert!((s.margin - 1.0).abs() < 1e-5);

        let two = [0.0f32, (3.0f32).ln()]; // p = [0.25, 0.75]
        let s = position_stats(&two);
        assert!((s.top1 - 0.75).abs() < 1e-5);
        assert!((s.margin - 0.5).abs() < 1e-5);
        let h = -(0.25f32 * 0.25f32.ln() + 0.75 * 0.75f32.ln());
        assert!((s.entropy - h).abs() < 1e-4);

        assert!(position_stats(&[]).entropy.is_nan());
    }

    #[test]
    fn proxies_reorder_but_floor_holds() {
        // Two positions: A low entropy but low top-1 (spread tail),
        // B higher entropy but higher top-1. Proxies disagree on order.
        let a = PositionStats {
            entropy: 0.5,
            top1: 0.6,
            margin: 0.3,
        };
        let b = PositionStats {
            entropy: 0.7,
            top1: 0.8,
            margin: 0.7,
        };
        let stats = [a, b];
        let (mut hs, mut ks) = ([0.0f32; 2], [0.0f32; 2]);
        for (proxy, first) in [
            (ErrorProxy::Entropy, 0u32),
            (ErrorProxy::Confidence, 1),
            (ErrorProxy::Margin, 1),
        ] {
            let cfg = EbCommitConfig {
                gamma: 0.0,
                max_commit: usize::MAX,
                proxy,
            };
            let mut c = vec![0u32, 1];
            let k = entropy_bounded_commit_stats(&mut c, &stats, &mut hs, &mut ks, &cfg);
            assert_eq!(k, 1, "γ=0 with positive entropies → singleton");
            assert_eq!(c[0], first, "{proxy:?}");
        }
    }

    /// Reference parity with HF transformers' DiffusionGemma
    /// `EntropyBoundSampler.accept_canvas` (entropy proxy): sort ascending,
    /// accept where `cumsum(H) − H ≤ bound` — the same EB form, written with
    /// the subtraction. Under ascending order the mask is a prefix, so its
    /// popcount must equal our `k`.
    #[test]
    fn hf_entropy_bound_sampler_parity() {
        let mut rng = fastrand::Rng::with_seed(2505);
        for _ in 0..500 {
            let n = rng.usize(1..64);
            let h: Vec<f32> = (0..n).map(|_| rng.f32() * 0.3).collect();
            let bound = rng.f32() * 0.5 + 1e-3;
            let mut sorted = h.clone();
            sorted.sort_by(f32::total_cmp);
            let mut cum = 0.0f64;
            let hf_k = sorted
                .iter()
                .filter(|&&x| {
                    cum += x as f64;
                    cum - x as f64 <= bound as f64
                })
                .count();
            let ours = commit(&h, bound).len();
            // Exact: measured 500/500 equal (seed 2505) — the incremental
            // f32 residual and HF's f64 `cumsum − H` agree on this range; a
            // tolerance here would only hide a future divergence.
            assert_eq!(ours, hf_k, "n={n} bound={bound}");
        }
    }

    #[test]
    fn deterministic_across_input_permutations() {
        let h = [0.1f32, 0.1, 0.05, 0.1, 0.3, 0.05];
        let mut a: Vec<u32> = (0..6).collect();
        let mut b: Vec<u32> = (0..6).rev().collect();
        let ka = entropy_bounded_commit(&mut a, &h, &h, 0.2, usize::MAX);
        let kb = entropy_bounded_commit(&mut b, &h, &h, 0.2, usize::MAX);
        assert_eq!(ka, kb);
        assert_eq!(
            a, b,
            "total order ⇒ identical output regardless of input order"
        );
    }
}
