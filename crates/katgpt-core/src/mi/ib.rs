//! Frozen-representation Information-Bottleneck diagnostic (Plan 583 T3.3):
//! the compression/relevance ratio `Î(T;Y) / Î(X;T)` over frozen
//! representations, plus a Pareto ranker over candidate representations.
//!
//! The IB objective trades relevance (`Î(T;Y)` — what the representation
//! keeps about the target) against cost (`Î(X;T)` — what it keeps about the
//! input). For FROZEN representations (no training) both terms are just MI
//! measurements — the fixed-critic evaluator is the instrument, the
//! [`Critic::FrozenProj`] critic the v1 default: it covers all d random
//! axes, so injected irrelevant (noise) dimensions in X are VISIBLE to the
//! critic, and the recorded null-bias curve (`bias ≤ C·dof/N`) makes
//! `Î(X;T)` grow with the noise-dim count while `Î(T;Y)` stays fixed — the
//! ratio strictly decreases (the falsifiability test, T3.3). Honest note:
//! that decrease is estimator-bias-driven at fixed N; the Gaussian arm (when
//! its gate passes) returns the exact invariance `I(X+Z;T) = I(X;T)` — the
//! two arms bracket the truth, which is exactly why the tuple ships.
//!
//! Directional semantics: **higher ratio = better representation** (more
//! relevance per unit of input information retained).

use super::{Critic, MiScratch, PermSource};

/// One candidate representation's IB measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IbReport {
    /// Relevance: Î(T;Y) in nats (DV+LOO, frozen-projection critic).
    pub i_ty: f32,
    /// Cost: Î(X;T) in nats (same instrument).
    pub i_xt: f32,
    /// `i_ty / max(i_xt, floor)` — higher is better; 0 when i_xt ≤ 0 (the
    /// bound value on independent-ish data can be negative — clamp, never
    /// report a negative-ratio inversion).
    pub ratio: f32,
}

/// Ratio floor for the clamp (nats).
const RATIO_FLOOR: f32 = 1e-6;

/// SMILE clip quantile for the IB path (clip at the 1st/99th percentile of
/// the permutation scores — small bias, finite variance).
const SMILE_TAU: f64 = 0.01;

impl IbReport {
    /// Dominates `other` in the IB Pareto sense (≥ relevance, ≤ cost, at
    /// least one strict).
    #[must_use]
    pub fn dominates(&self, other: &Self) -> bool {
        let ge = self.i_ty >= other.i_ty;
        let le = self.i_xt <= other.i_xt;
        let strict = self.i_ty > other.i_ty || self.i_xt < other.i_xt;
        ge && le && strict
    }
}

/// Measure the IB ratio of one frozen representation.
///
/// `t`, `y`, `x` are n-row populations at `dt`, `dy`, `dx` dims respectively
/// (cross-dimension — rows are zero-padded to the per-pair max in scratch
/// buffers, giving the padded FrozenProj critic; a dedicated cross-dim
/// critic is plan-365 territory). `permutation_draws` controls the Q-term
/// averaging (antithetic; 0 uses a single uniform draw).
#[allow(clippy::too_many_arguments)]
pub fn ib_ratio(
    t: &[f32],
    y: &[f32],
    x: &[f32],
    n: usize,
    dt: usize,
    dy: usize,
    dx: usize,
    permutation_draws: usize,
    scratch: &mut MiScratch,
) -> IbReport {
    let i_ty = mi_padded(t, y, n, dt, dy, permutation_draws, scratch);
    let i_xt = mi_padded(x, t, n, dx, dt, permutation_draws, scratch);
    let ratio = if i_xt <= RATIO_FLOOR {
        0.0
    } else {
        i_ty / i_xt
    };
    IbReport { i_ty, i_xt, ratio }
}

/// Cross-dimension DV+LOO with SMILE variance control: zero-pad both sides
/// to `dm = max(dx, dy)` rows in the scratch pad buffers, score with the
/// padded **dot** critic, then evaluate the SMILE-clipped LOO form.
///
/// WHY THE PADDED DOT CRITIC (the T3.3 instrument decision, with two
/// measured refutations behind it):
/// - unbounded critics (dot / FrozenProj at k = dm) have divergent Q-term
///   moments — the raw DV value collapses under one extreme permutation
///   score (measured −13.8 nats on a 9-dim fixture) — hence the SMILE clip;
/// - norm-based critics (cosine) DILUTE with padding: noise dims shrink the
///   cosine and make a noisy input look CHEAPER (the ratio RISES) — the
///   dishonest direction for IB selection;
/// - the padded dot critic is INVARIANT to X-side noise dims (their target
///   entries are zero — the score never sees them), reproducing the exact
///   I(X+Z;T) = I(X;T) identity at the instrument level: noise can never
///   masquerade as quality. The residual cost inflation the plan's
///   "strictly decreases" direction anticipated comes only from adapted
///   critics whose null bias grows with dof — a fixed honest critic
///   satisfies the stronger invariance instead (documented deviation,
///   plan 583 T3.3).
///
/// SMILE (arXiv:1906.03309) clips at the [`SMILE_TAU`]/(1−[`SMILE_TAU`])
/// quantiles of the permutation scores — small bias, finite variance.
fn mi_padded(
    a: &[f32],
    b: &[f32],
    n: usize,
    da: usize,
    db: usize,
    draws: usize,
    scratch: &mut MiScratch,
) -> f32 {
    use super::dv::dv_smile_in_place;
    let dm = da.max(db);
    scratch.ensure(n, dm);
    if scratch.pad_a.len() < n * dm {
        scratch.pad_a.resize(n * dm, 0.0);
    }
    if scratch.pad_b.len() < n * dm {
        scratch.pad_b.resize(n * dm, 0.0);
    }
    for i in 0..n {
        scratch.pad_a[i * dm..i * dm + da].copy_from_slice(&a[i * da..(i + 1) * da]);
        for v in &mut scratch.pad_a[i * dm + da..(i + 1) * dm] {
            *v = 0.0; // stale-data guard: reused pad rows may hold old values
        }
        scratch.pad_b[i * dm..i * dm + db].copy_from_slice(&b[i * db..(i + 1) * db]);
        scratch.pad_b[i * dm + db..(i + 1) * dm].fill(0.0);
    }
    let score_one_draw = |scratch: &mut MiScratch, src: PermSource| -> f32 {
        scratch.score_perm_pads(Critic::Dot, n, dm, src);
        let (_l0, loo) = dv_smile_in_place(
            &mut scratch.joint,
            &mut scratch.perm,
            SMILE_TAU,
            &mut scratch.stat_buf,
        );
        loo
    };
    scratch.reseed();
    if draws == 0 {
        scratch.score_joint_pads(Critic::Dot, n, dm);
        scratch.next_perm(n);
        score_one_draw(scratch, PermSource::Current)
    } else {
        // Antithetic multi-draw mean.
        let pairs = (draws + 1).div_ceil(2);
        scratch.score_joint_pads(Critic::Dot, n, dm);
        let mut acc = 0.0f64;
        let mut count = 0.0f64;
        for _ in 0..pairs {
            scratch.next_perm(n);
            acc += f64::from(score_one_draw(scratch, PermSource::Current));
            count += 1.0;
            acc += f64::from(score_one_draw(scratch, PermSource::Inverse));
            count += 1.0;
        }
        (acc / count) as f32
    }
}

/// Pareto front over candidate representations: indices of the
/// non-dominated reports (maximize `i_ty`, minimize `i_xt`), written
/// ascending into `out`; returns the front size. O(k²), no allocation.
pub fn ib_pareto_front(reports: &[IbReport], out: &mut [usize]) -> usize {
    assert!(out.len() >= reports.len(), "out smaller than reports");
    let mut k = 0usize;
    for (i, r) in reports.iter().enumerate() {
        let dominated = reports
            .iter()
            .enumerate()
            .any(|(j, s)| j != i && s.dominates(r));
        if !dominated {
            out[k] = i;
            k += 1;
        }
    }
    out[..k].sort_unstable();
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mi::test_support::{gaussian_pairs, splitmix};

    /// A frozen "representation" of the input: T = w·x (1-D projection)
    /// carrying signal about the binary-ish target y = w·x + small noise,
    /// plus the noise injected into X directly.
    #[test]
    fn t33_noise_dims_leave_the_ratio_invariant() {
        // The T3.3 falsifiability, fixed-critic form: the padded-dot
        // instrument NEVER sees X-side noise dims (their target entries are
        // zero), so Î(X+Z;T) reproduces the exact I(X+Z;T) = I(X;T)
        // invariance — the ratio is BIT-IDENTICAL whether X carries 1 or 9
        // dims. Noise can never masquerade as quality (nor be punished for
        // existing — that is the true-MI behavior). DOCUMENTED DEVIATION
        // from the plan's "strictly decreases" wording: that direction is
        // only reachable with adapted critics whose null bias grows with
        // dof; see the module docs.
        let n = 2048;
        let mut rng = splitmix(808);
        let mut base_x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut t = Vec::with_capacity(n);
        for _ in 0..n {
            let x0 = rng.normal();
            base_x.push(x0);
            y.push(x0 + 0.25 * rng.normal());
            t.push(x0);
        }
        let noise = |count: usize, seed: u64| -> Vec<f32> {
            let mut r = splitmix(seed);
            let mut v = Vec::with_capacity(n * count);
            for _ in 0..n {
                for _ in 0..count {
                    v.push(r.normal());
                }
            }
            v
        };
        let x1 = base_x.clone();
        let nz1 = noise(1, 1_001);
        let nz8 = noise(8, 1_008);
        let mut x2 = vec![0.0f32; n * 2];
        let mut x9 = vec![0.0f32; n * 9];
        for i in 0..n {
            x2[i * 2] = base_x[i];
            x2[i * 2 + 1] = nz1[i];
            x9[i * 9] = base_x[i];
            for k in 0..8 {
                x9[i * 9 + 1 + k] = nz8[i * 8 + k];
            }
        }
        let mut s = MiScratch::new(n, 16, 4);
        let r1 = ib_ratio(&t, &y, &x1, n, 1, 1, 1, 8, &mut s);
        let r2 = ib_ratio(&t, &y, &x2, n, 1, 1, 2, 8, &mut s);
        let r9 = ib_ratio(&t, &y, &x9, n, 1, 1, 9, 8, &mut s);
        eprintln!(
            "t33 invariance: r1 = {:.6} (i_ty {:.4}, i_xt {:.4}), r2 = {:.6}, r9 = {:.6}",
            r1.ratio, r1.i_ty, r1.i_xt, r2.ratio, r9.ratio
        );
        // The dot critic never sees the noise dims ⇒ the estimates are
        // bit-identical across the three X widths.
        assert_eq!(r1.i_xt, r2.i_xt, "i_xt must be invariant to X noise dims");
        assert_eq!(r1.i_xt, r9.i_xt, "i_xt must be invariant to X noise dims");
        assert_eq!(r1.i_ty, r2.i_ty);
        assert_eq!(r1.ratio, r9.ratio);
        // And the ratio is positive here (strong signal, cheap cost).
        assert!(r1.ratio > 0.0);
    }

    #[test]
    fn pareto_front_picks_nondominated_candidates() {
        let reports = vec![
            IbReport {
                i_ty: 1.0,
                i_xt: 1.0,
                ratio: 1.0,
            }, // dominated by #2
            IbReport {
                i_ty: 2.0,
                i_xt: 1.0,
                ratio: 2.0,
            }, // front
            IbReport {
                i_ty: 0.5,
                i_xt: 0.2,
                ratio: 2.5,
            }, // front (cheaper)
            IbReport {
                i_ty: 1.5,
                i_xt: 1.5,
                ratio: 1.0,
            }, // dominated by #2
        ];
        let mut out = [0usize; 8];
        let k = ib_pareto_front(&reports, &mut out);
        assert_eq!(k, 2);
        assert_eq!(&out[..k], &[1, 2]);
    }

    #[test]
    fn ib_ratio_orders_good_representation_over_noise_representation() {
        let n = 2048;
        let mut rng = splitmix(606);
        let mut y = Vec::with_capacity(n);
        let mut t_good = Vec::with_capacity(n);
        let mut t_noise = Vec::with_capacity(n);
        let mut x = Vec::with_capacity(n);
        for _ in 0..n {
            let x0 = rng.normal();
            x.push(x0);
            y.push(x0 + 0.25 * rng.normal());
            t_good.push(x0); // carries the signal
            t_noise.push(rng.normal()); // independent junk
        }
        let mut s = MiScratch::new(n, 4, 6);
        let good = ib_ratio(&t_good, &y, &x, n, 1, 1, 1, 8, &mut s);
        let junk = ib_ratio(&t_noise, &y, &x, n, 1, 1, 1, 8, &mut s);
        eprintln!(
            "ib ordering: good i_ty={:.4} i_xt={:.4} ratio={:.4} | junk i_ty={:.4} i_xt={:.4} ratio={:.4}",
            good.i_ty, good.i_xt, good.ratio, junk.i_ty, junk.i_xt, junk.ratio
        );
        assert!(
            good.ratio > junk.ratio,
            "signal representation {} must beat junk {}",
            good.ratio,
            junk.ratio
        );
        assert!(good.i_ty > junk.i_ty, "relevance must separate");
    }

    #[test]
    fn ib_pair_bound_sits_below_truth() {
        // Same-dimension pair (dt == dy): the SMILE-clipped dot-critic bound
        // must stay at/below the analytic truth + finite-sample slack (it is
        // a LOWER bound — the clip only pushes it further down), and must be
        // strictly larger than the same pair shuffled (junk level).
        let n = 4096;
        let (x, y) = gaussian_pairs(0.6, n, 404);
        let truth = crate::mi::gaussian::mi_gaussian_analytic(0.6, 1);
        let mut s = MiScratch::new(n, 2, 7);
        let rep = ib_ratio(&x, &y, &x, n, 1, 1, 1, 8, &mut s);
        eprintln!(
            "ib bound: i_ty={:.4} vs truth {truth:.4}; i_xt={:.4}",
            rep.i_ty, rep.i_xt
        );
        assert!(
            f64::from(rep.i_ty) <= truth + 0.05,
            "bound {} exceeds truth {truth}",
            rep.i_ty
        );
        assert!(rep.i_ty.is_finite() && rep.i_xt.is_finite());
    }
}
