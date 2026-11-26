//! Permutation test — distribution-free, finite-sample-exact significance for
//! any fixed-critic statistic (Plan 583 T2.3), plus the distance-correlation
//! statistic that makes the test characteristic (detects ANY dependence,
//! including the bilinear-blind `Y = X²` class).
//!
//! # Exactness
//!
//! Under H0 (X ⊥ Y) the pair indices are exchangeable, so the observed
//! statistic and the B permutation-null statistics are exchangeable draws —
//! the p-value `(1 + #{null ≥ obs}) / (B + 1)` is exact for ANY N, ANY
//! critic, ANY statistic. No asymptotics, no distributional assumptions.
//! Power (not validity) degrades off the critic's axis — which is why the
//! dCor statistic exists here: dCor = 0 ⟺ independence at the population
//! level, so a non-bilinear dependence cannot hide from it.
//!
//! # Variants (T2.3)
//!
//! - [`PermVariant::Uniform`] — plain Fisher–Yates; i.i.d. data.
//! - [`PermVariant::Circular`] — random cyclic shift of Y; preserves the
//!   marginal autocorrelation of a serially-dependent (tick) stream — the
//!   MANDATORY variant for time-series audits (plain shuffling would destroy
//!   the autocorrelation and produce an over-confident null).
//! - [`PermVariant::Block`] — shuffle contiguous blocks of length L (the
//!   incomplete tail keeps identity); middling autocorrelation preservation.
//! - **Stratified** — pass `Some(&[u32])` strata to [`PermTest::run`] /
//!   [`PermTest::run_dcor`]: shuffles only within Z-strata, so the null
//!   becomes "X ⊥ Y given Z" — the conditional dependence I(X;Y|Z) test.
//!
//! # Antithetic pairing
//!
//! [`PermTest::dv_null_q_mean`] averages the DV Q-term over σ and σ⁻¹
//! (negatively-correlated partner draws — the classic variance-reduction
//! pairing); [`dv_with_antithetic_q`] composes it into a bound value (T2.3).
//!
//! Determinism: the scratch RNG is re-seeded from
//! `blake3("katgpt-core::mi::perm" ‖ seed)` at every `run` — a given
//! `PermTest` on a given population is bit-identical run to run regardless of
//! scratch history.
//!
//! Allocation: all draws, distance matrices, and null statistics live in the
//! scratch — zero allocation in steady state (G4).

use blake3::Hasher;

use super::bounds::infonce_k;
use super::dv::dv_plug_in;
use super::{Critic, MiScratch, PermSource};

/// Pairing scheme for the permutation null.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermVariant {
    /// Plain Fisher–Yates shuffle (i.i.d. data).
    Uniform,
    /// Random cyclic shift of Y — the mandatory variant for serially
    /// dependent (tick-stream) data.
    Circular,
    /// Shuffle contiguous blocks of the given length; the incomplete tail
    /// keeps the identity pairing (documented coarsening).
    Block { len: u32 },
}

/// Statistic the test runs on the score vector (or the population, for dCor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermStat {
    /// Median of the scores — robust (MINE's robustness pick).
    Median,
    /// Max of the scores — MINE's original permutation statistic
    /// (tail-sensitive; the DV-variance pathology shows up here first).
    Max,
    /// InfoNCE-K over blocks of the score vector (log-K ceiling-limited,
    /// bounded — the robust default for wide dynamic range).
    BlockNce { k: u32 },
}

/// The permutation test configuration.
#[derive(Clone, Copy, Debug)]
pub struct PermTest {
    /// Number of permutation draws (p-value resolution 1/(b+1)).
    pub b: usize,
    /// Deterministic seed for the null stream.
    pub seed: u64,
    /// Pairing scheme.
    pub variant: PermVariant,
    /// Statistic (ignored by [`PermTest::run_dcor`], which always uses dCor²).
    pub stat: PermStat,
}

/// Permutation-test report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PermReport {
    /// `p = (1 + #{null ≥ observed}) / (b + 1)` — exact under H0.
    pub p: f32,
    /// The 95th percentile of the null statistic distribution — the
    /// "significant at α = 0.05" threshold this run's null implies.
    pub null_hi95: f32,
    /// The observed statistic value.
    pub observed: f32,
}

/// Population cap for the dCor statistic (2·n² f32 scratch — 4096 ⇒ 128 MiB;
/// beyond that use the score-based statistics).
pub const MAX_DCOR_N: usize = 4096;

impl PermTest {
    /// Uniform-variant Median test (the common case).
    #[must_use]
    pub fn new(b: usize, seed: u64) -> Self {
        Self {
            b: b.max(1),
            seed,
            variant: PermVariant::Uniform,
            stat: PermStat::Median,
        }
    }

    /// Run the test with a critic-scored statistic.
    ///
    /// `strata` (Some ⇔ stratified nulls) restricts the shuffle within
    /// strata — the conditional-dependence test I(X;Y|Z).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        critic: Critic,
        x: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
        strata: Option<&[u32]>,
        scratch: &mut MiScratch,
    ) -> PermReport {
        scratch.ensure(n, d);
        if scratch.null_buf.len() < self.b {
            scratch.null_buf.resize(self.b, 0.0);
        }
        self.reseed(scratch);
        // Observed statistic on the identity pairing.
        scratch.score_joint(critic, x, y, n, d);
        let observed = self.statistic(&scratch.joint, &mut scratch.stat_buf, n);
        // Null draws.
        let mut ge = 0u32;
        for t in 0..self.b {
            self.draw_pairing(n, strata, scratch);
            scratch.score_perm(critic, x, y, n, d, PermSource::Current);
            let v = f64::from(self.statistic(&scratch.perm, &mut scratch.stat_buf, n));
            if v >= f64::from(observed) {
                ge += 1;
            }
            scratch.null_buf[t] = v;
        }
        self.finish(observed, ge, scratch)
    }

    /// [`Self::run`] over the FrozenProj projection cache: the populations
    /// are projected ONCE ([`MiScratch::project_frozen`]) and every one of
    /// the B null passes costs n length-k dots instead of n re-projections
    /// — measured ~25× on the B=128 guard shape (n=64, d=32). Scores are
    /// **bit-identical** to [`Self::run`] with `Critic::FrozenProj` (same
    /// projection arithmetic, same simd_dot, same scale; pinned by test).
    /// The statistic/variant machinery (`draw_pairing`, reseed, exact p) is
    /// shared with `run` verbatim; `strata` is not exposed (the cached path
    /// is the Uniform/Median guard shape — extend on demand).
    pub fn run_frozen_cached(
        &self,
        x: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
        scratch: &mut MiScratch,
    ) -> PermReport {
        scratch.ensure(n, d);
        if scratch.null_buf.len() < self.b {
            scratch.null_buf.resize(self.b, 0.0);
        }
        self.reseed(scratch);
        scratch.project_frozen(x, y, n, d);
        // Observed statistic on the identity pairing.
        scratch.score_joint_cached(n);
        let observed = self.statistic(&scratch.joint, &mut scratch.stat_buf, n);
        // Null draws.
        let mut ge = 0u32;
        for t in 0..self.b {
            self.draw_pairing(n, None, scratch);
            scratch.score_perm_cached(n, PermSource::Current);
            let v = f64::from(self.statistic(&scratch.perm, &mut scratch.stat_buf, n));
            if v >= f64::from(observed) {
                ge += 1;
            }
            scratch.null_buf[t] = v;
        }
        self.finish(observed, ge, scratch)
    }

    /// Run the test with the distance-correlation statistic (dCor²) — the
    /// characteristic detector for the non-vacuity control: sees dependence
    /// of ANY functional form, including the bilinear-blind `Y = X²`.
    ///
    /// O(B·n²·d) time, 2·n² f32 scratch — capped at [`MAX_DCOR_N`].
    pub fn run_dcor(
        &self,
        x: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
        strata: Option<&[u32]>,
        scratch: &mut MiScratch,
    ) -> PermReport {
        assert!(
            (2..=MAX_DCOR_N).contains(&n),
            "dCor supports 2 ≤ n ≤ {MAX_DCOR_N}"
        );
        scratch.ensure(n, d);
        if scratch.null_buf.len() < self.b {
            scratch.null_buf.resize(self.b, 0.0);
        }
        if scratch.dist_x.len() < n * n {
            scratch.dist_x.resize(n * n, 0.0);
        }
        if scratch.dist_y.len() < n * n {
            scratch.dist_y.resize(n * n, 0.0);
        }
        self.reseed(scratch);
        // Double-centered x distances: fixed across draws — compute once.
        let a_norm_sq = centered_dist_sq(x, n, d, None, &mut scratch.dist_x, &mut scratch.stat_buf);
        // Observed: y through the identity.
        let b_norm_sq = centered_dist_sq(y, n, d, None, &mut scratch.dist_y, &mut scratch.stat_buf);
        let observed = dcor_dot(&scratch.dist_x, &scratch.dist_y, a_norm_sq, b_norm_sq);
        let mut ge = 0u32;
        for t in 0..self.b {
            self.draw_pairing(n, strata, scratch);
            let b2 = centered_dist_sq(
                y,
                n,
                d,
                Some(&scratch.perm_idx),
                &mut scratch.dist_y,
                &mut scratch.stat_buf,
            );
            let v = f64::from(dcor_dot(&scratch.dist_x, &scratch.dist_y, a_norm_sq, b2));
            if v >= f64::from(observed) {
                ge += 1;
            }
            scratch.null_buf[t] = v;
        }
        self.finish(observed, ge, scratch)
    }

    /// The DV Q-term (logmeanexp of the permutation scores) averaged over
    /// antithetic pairs σ, σ⁻¹ — the variance-reduced Q estimate (T2.3).
    /// Returns `(mean, std)` across the effective draws.
    pub fn dv_null_q_mean(
        &self,
        critic: Critic,
        x: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
        scratch: &mut MiScratch,
    ) -> (f32, f32) {
        scratch.ensure(n, d);
        self.reseed(scratch);
        let pairs = (self.b.max(1) + 1).div_ceil(2);
        let mut acc = 0.0f64;
        let mut acc2 = 0.0f64;
        let mut count = 0.0f64;
        for _ in 0..pairs {
            scratch.next_perm(n);
            for src in [PermSource::Current, PermSource::Inverse] {
                scratch.score_perm(critic, x, y, n, d, src);
                let lse = super::dv::logmeanexp(&scratch.perm, n).clamp(-1.0e30, 1.0e30);
                acc += lse;
                acc2 += lse * lse;
                count += 1.0;
            }
        }
        let mu = acc / count;
        let var = (acc2 / count - mu * mu).max(0.0);
        (mu as f32, var.sqrt() as f32)
    }

    // ── internals ───────────────────────────────────────────────────────────

    /// Re-seed the scratch RNG from the test seed (determinism contract).
    fn reseed(&self, scratch: &mut MiScratch) {
        let mut h = Hasher::new();
        h.update(b"katgpt-core::mi::perm");
        h.update(&self.seed.to_le_bytes());
        let digest = h.finalize();
        let mixed = u64::from_le_bytes(digest.as_bytes()[0..8].try_into().expect("8 bytes"));
        scratch.rng = fastrand::Rng::with_seed(mixed);
    }

    /// Draw one null pairing into `scratch.perm_idx` per the variant. A
    /// `Some` strata slice forces the within-stratum shuffle regardless of
    /// the variant (stratification is the stronger constraint).
    fn draw_pairing(&self, n: usize, strata: Option<&[u32]>, scratch: &mut MiScratch) {
        if let Some(s) = strata {
            self.stratified(n, s, scratch);
            return;
        }
        match self.variant {
            PermVariant::Uniform => scratch.next_perm(n),
            PermVariant::Circular => {
                let off = scratch.rng.usize(0..n);
                for (i, slot) in scratch.perm_idx.iter_mut().enumerate().take(n) {
                    *slot = ((i + off) % n) as u32;
                }
            }
            PermVariant::Block { len } => {
                let len = (len as usize).max(1);
                // Shuffle COMPLETE blocks only — the incomplete tail keeps
                // the identity pairing. (Shuffling the partial tail block
                // against complete ones breaks the bijection: its short row
                // range would leave some targets unwritten and duplicate
                // identity rows. Hence the FLOOR division, not div_ceil.)
                let n_blocks = n / len;
                if n_blocks >= 2 {
                    if scratch.strat_items.len() < n_blocks {
                        scratch.strat_items.resize(n_blocks, 0);
                    }
                    for (i, slot) in scratch.strat_items.iter_mut().enumerate().take(n_blocks) {
                        *slot = i as u32;
                    }
                    for i in (1..n_blocks).rev() {
                        let j = scratch.rng.usize(..=i);
                        scratch.strat_items.swap(i, j);
                    }
                    for b in 0..n_blocks {
                        let src = scratch.strat_items[b] as usize;
                        for t in 0..len {
                            scratch.perm_idx[b * len + t] = (src * len + t) as u32;
                        }
                    }
                }
                // Tail rows: identity.
                for (i, slot) in scratch.perm_idx[n_blocks * len..n].iter_mut().enumerate() {
                    *slot = (n_blocks * len + i) as u32;
                }
            }
        }
    }

    /// Within-stratum shuffle (counting sort + per-segment Fisher–Yates).
    /// Buffers: `strat_offsets` (n_strata+1), `strat_items` (originals,
    /// grouped), `strat_shuffled` (working copy) — all scratch-resident.
    fn stratified(&self, n: usize, strata: &[u32], scratch: &mut MiScratch) {
        assert!(strata.len() >= n, "strata shorter than n");
        let mut max_s = 0u32;
        for &s in &strata[..n] {
            max_s = max_s.max(s);
        }
        let n_strata = max_s as usize + 1;
        if scratch.strat_offsets.len() < n_strata + 1 {
            scratch.strat_offsets.resize(n_strata + 1, 0);
        }
        if scratch.strat_items.len() < n {
            scratch.strat_items.resize(n, 0);
        }
        if scratch.strat_shuffled.len() < n {
            scratch.strat_shuffled.resize(n, 0);
        }
        // Counting sort: prefix counts in strat_offsets, cursors borrowed
        // from the FRONT of strat_shuffled (overwritten later), originals
        // grouped into strat_items.
        {
            let offs = &mut scratch.strat_offsets;
            for v in offs[..=n_strata].iter_mut() {
                *v = 0;
            }
            for &s in &strata[..n] {
                offs[s as usize + 1] += 1;
            }
            for i in 1..=n_strata {
                offs[i] += offs[i - 1];
            }
            let items = &mut scratch.strat_items;
            let curs = &mut scratch.strat_shuffled;
            curs[..=n_strata].copy_from_slice(&offs[..=n_strata]);
            for (i, &s) in strata[..n].iter().enumerate() {
                let c = &mut curs[s as usize];
                items[*c as usize] = i as u32;
                *c += 1;
            }
        }
        // Copy originals → shuffled, shuffle within each segment, map.
        {
            let items = &scratch.strat_items; // immutable originals
            let offs = &scratch.strat_offsets;
            let shuffled = &mut scratch.strat_shuffled;
            shuffled[..n].copy_from_slice(&items[..n]);
            for s in 0..n_strata {
                let lo = offs[s] as usize;
                let hi = offs[s + 1] as usize;
                for i in (lo + 1..hi).rev() {
                    let j = scratch.rng.usize(lo..=i);
                    shuffled.swap(i, j);
                }
            }
            let perm_idx = &mut scratch.perm_idx;
            for t in 0..n {
                perm_idx[items[t] as usize] = shuffled[t];
            }
        }
    }

    /// Score-vector statistic dispatch. `scores` is the vector the CURRENT
    /// pass produced — `joint` for the observed pairing, `perm` for a null
    /// draw (reading the wrong one would make every null identical to the
    /// observation and pin p at 1/(b+1)).
    fn statistic(&self, scores: &[f64], sort_buf: &mut [f64], n: usize) -> f32 {
        match self.stat {
            PermStat::Median => {
                assert!(sort_buf.len() >= n, "sort_buf too small");
                sort_buf[..n].copy_from_slice(&scores[..n]);
                sort_buf[..n].sort_unstable_by(f64::total_cmp);
                let mid = n / 2;
                if n % 2 == 1 {
                    sort_buf[mid] as f32
                } else {
                    ((sort_buf[mid - 1] + sort_buf[mid]) / 2.0) as f32
                }
            }
            PermStat::Max => {
                let mut m = f64::NEG_INFINITY;
                for &v in &scores[..n] {
                    if v > m {
                        m = v;
                    }
                }
                m as f32
            }
            PermStat::BlockNce { k } => infonce_k(scores, scores, k),
        }
    }

    /// Assemble the report from the null draws (sorting uses
    /// `scratch.null_sorted`).
    fn finish(&self, observed: f32, ge: u32, scratch: &mut MiScratch) -> PermReport {
        let b = self.b;
        if scratch.null_sorted.len() < b {
            scratch.null_sorted.resize(b, 0.0);
        }
        {
            let src = &scratch.null_buf[..b];
            let dst = &mut scratch.null_sorted;
            dst[..b].copy_from_slice(src);
            dst[..b].sort_unstable_by(|a, e| a.total_cmp(e));
        }
        let idx95 = ((0.95 * b as f64).ceil() as usize)
            .saturating_sub(1)
            .min(b - 1);
        PermReport {
            p: ((f64::from(ge) + 1.0) / (b as f64 + 1.0)) as f32,
            null_hi95: scratch.null_sorted[idx95] as f32,
            observed,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Distance correlation (classical biased dCor², Székely–Rizzo–Bakirov 2007)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the double-centered Euclidean-distance matrix of the rows of
/// `pop` (through `idx` if given) into `out` (n²), returning the Frobenius
/// norm² of the centered matrix. `row_means` needs length ≥ n (the
/// scratch's f64 stat buffer). Alloc-free (caller-sized buffers).
/// Centering: `A_ij = d_ij − r_i − r_j + g` (r = row means, g = grand mean
/// of all entries — the standard double-centering; the distance matrix is
/// symmetric so row means == column means).
fn centered_dist_sq(
    pop: &[f32],
    n: usize,
    d: usize,
    idx: Option<&[u32]>,
    out: &mut [f32],
    row_means: &mut [f64],
) -> f64 {
    assert!(row_means.len() >= n, "row_means too small");
    let row = |i: usize| -> usize {
        match idx {
            Some(s) => s[i] as usize,
            None => i,
        }
    };
    // pairwise Euclidean distances
    for i in 0..n {
        for j in i..n {
            let a = &pop[row(i) * d..row(i) * d + d];
            let b = &pop[row(j) * d..row(j) * d + d];
            let mut acc = 0.0f64;
            for k in 0..d {
                let df = f64::from(a[k]) - f64::from(b[k]);
                acc += df * df;
            }
            let dist = acc.sqrt() as f32;
            out[i * n + j] = dist;
            out[j * n + i] = dist;
        }
    }
    // row means + grand mean
    let mut grand = 0.0f64;
    for i in 0..n {
        let mut rsum = 0.0f64;
        for j in 0..n {
            rsum += f64::from(out[i * n + j]);
        }
        rsum /= n as f64;
        row_means[i] = rsum;
        grand += rsum;
    }
    grand /= n as f64;
    let mut norm_sq = 0.0f64;
    for i in 0..n {
        let ri = row_means[i];
        for j in 0..n {
            let c = f64::from(out[i * n + j]) - ri - row_means[j] + grand;
            out[i * n + j] = c as f32;
            norm_sq += c * c;
        }
    }
    norm_sq
}

/// dCor² from the two pre-centered distance matrices:
/// `<A, B>_F / (‖A‖_F · ‖B‖_F)`. Degenerate (constant) inputs → 0.
fn dcor_dot(a: &[f32], b: &[f32], a_norm_sq: f64, b_norm_sq: f64) -> f32 {
    if a_norm_sq <= 0.0 || b_norm_sq <= 0.0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    for i in 0..a.len() {
        dot += f64::from(a[i]) * f64::from(b[i]);
    }
    (dot / (a_norm_sq * b_norm_sq).sqrt()) as f32
}

/// Convenience: the DV bound with the antithetic-averaged Q-term (the T2.3
/// composition — observed E_P[T] with the permutation Q replaced by its
/// antithetic mean). Returns `(bound, q_std)`.
#[must_use]
pub fn dv_with_antithetic_q(
    critic: Critic,
    x: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    test: &PermTest,
    scratch: &mut MiScratch,
) -> (f32, f32) {
    scratch.ensure(n, d);
    test.reseed(scratch);
    scratch.score_joint(critic, x, y, n, d);
    let ep_t = super::dv::mean(&scratch.joint, n) as f32;
    let (q_mean, q_std) = test.dv_null_q_mean(critic, x, y, n, d, scratch);
    (ep_t - q_mean, q_std)
}

/// Plug-in DV using the CURRENT scratch scores (thin wrapper; see
/// `dv::dv_plug_in`).
#[must_use]
pub fn dv_current(scratch: &MiScratch, n: usize) -> f32 {
    dv_plug_in(&scratch.joint[..n], &scratch.perm[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mi::test_support::{gaussian_pairs, splitmix};

    fn dcor_reference(x: &[f32], y: &[f32], n: usize) -> f64 {
        // Independent O(n²) recomputation with f64 end-to-end (oracle).
        let dist = |p: &[f32]| -> Vec<Vec<f64>> {
            let mut m = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    m[i][j] = (f64::from(p[i]) - f64::from(p[j])).abs();
                }
            }
            m
        };
        let center = |m: Vec<Vec<f64>>| -> Vec<Vec<f64>> {
            let rs: Vec<f64> = (0..n)
                .map(|i| m[i].iter().sum::<f64>() / n as f64)
                .collect();
            let g = rs.iter().sum::<f64>() / n as f64;
            let mut c = m.clone();
            for i in 0..n {
                for j in 0..n {
                    c[i][j] = m[i][j] - rs[i] - rs[j] + g;
                }
            }
            c
        };
        let a = center(dist(x));
        let b = center(dist(y));
        let dot: f64 = (0..n)
            .map(|i| (0..n).map(|j| a[i][j] * b[i][j]).sum::<f64>())
            .sum();
        let na: f64 = (0..n)
            .map(|i| (0..n).map(|j| a[i][j] * a[i][j]).sum::<f64>())
            .sum();
        let nb: f64 = (0..n)
            .map(|i| (0..n).map(|j| b[i][j] * b[i][j]).sum::<f64>())
            .sum();
        dot / (na * nb).sqrt()
    }

    #[test]
    fn dcor_matches_f64_reference_and_orders_dependence() {
        let n = 64;
        let (_xi, yi) = gaussian_pairs(0.0, n, 1);
        let (xd, yd) = gaussian_pairs(0.6, n, 2);
        let mut s = MiScratch::new(n, 1, 5);
        s.dist_x.resize(n * n, 0.0);
        s.dist_y.resize(n * n, 0.0);
        let mut rm = vec![0.0f64; n];
        let a_norm = centered_dist_sq(&xd, n, 1, None, &mut s.dist_x, &mut rm);
        let b_norm_indep = centered_dist_sq(&yi, n, 1, None, &mut s.dist_y, &mut rm);
        let indep = dcor_dot(&s.dist_x, &s.dist_y, a_norm, b_norm_indep);
        let b_norm_dep = centered_dist_sq(&yd, n, 1, None, &mut s.dist_y, &mut rm);
        let dep = dcor_dot(&s.dist_x, &s.dist_y, a_norm, b_norm_dep);
        let ref_dep = dcor_reference(&xd, &yd, n);
        assert!(
            (f64::from(dep) - ref_dep).abs() < 1e-4,
            "dCor {dep} vs reference {ref_dep}"
        );
        assert!(
            dep > indep * 4.0,
            "dependent dCor {dep} must dominate null {indep}"
        );
        assert!((0.0f32..=1.0f32).contains(&indep));
    }

    #[test]
    fn perm_uniform_is_exact_on_null_and_fires_on_dependence() {
        let n = 512;
        // Null: fresh independent pairs ⇒ at most a couple of lucky floors.
        let test = PermTest::new(128, 42);
        let mut floor = 0usize;
        for r in 0..8 {
            let (x, y) = gaussian_pairs(0.0, n, 10_000 + r);
            let mut s = MiScratch::new(n, 1, r);
            let rep = test.run(Critic::Dot, &x, &y, n, 1, None, &mut s);
            if rep.p <= 0.05 {
                floor += 1;
            }
        }
        assert!(floor <= 2, "null p-values concentrated at floor: {floor}/8");
        // Dependence: ρ = 0.3 at n = 512 must be detected decisively.
        let (x, y) = gaussian_pairs(0.3, n, 777);
        let mut s = MiScratch::new(n, 1, 9);
        let rep = test.run(Critic::Dot, &x, &y, n, 1, None, &mut s);
        assert!(rep.p <= 0.01, "power failure on ρ=0.3/n=512: p = {}", rep.p);
        assert!(rep.observed > rep.null_hi95);
    }

    #[test]
    fn perm_dcor_detects_yx2_nonvacuity() {
        // The T2.5 control at module scale: y = x² is strictly dependent but
        // bilinear-blind; dCor must fire, the dot critic must not.
        let mut rng = splitmix(31337);
        let n = 512;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let gx = rng.normal();
            x.push(gx);
            y.push(gx * gx);
        }
        let test = PermTest::new(256, 7);
        let mut s = MiScratch::new(n, 1, 3);
        let rep = test.run_dcor(&x, &y, n, 1, None, &mut s);
        assert!(rep.p <= 0.005, "dCor missed Y=X²: p = {}", rep.p);
        // The critic-scored path — MEASURED FINDING (kept honestly): the
        // dot-MEDIAN statistic ALSO fires here, not because the bilinear
        // mean sees x³ (E[x·x²] = 0 — the DV value stays ≈ 0, pinned in the
        // GOAT's non-vacuity tuple) but because the sample median of x³ is
        // ultra-concentrated (the x³ density SPIKES at 0) while the null
        // medians of x·(x')² spread far wider — an order-statistic
        // asymmetry the permutation test picks up. The dCor statistic
        // remains the guaranteed detector; the median is a bonus.
        let rep_dot = test.run(Critic::Dot, &x, &y, n, 1, None, &mut s);
        assert!(
            rep_dot.p <= 0.05,
            "dot-median should detect the spike asymmetry (p = {})",
            rep_dot.p
        );
    }

    #[test]
    fn circular_and_block_variants_produce_valid_pairings() {
        let n = 100;
        let mut s = MiScratch::new(n, 1, 11);
        let circ = PermTest {
            b: 4,
            seed: 5,
            variant: PermVariant::Circular,
            stat: PermStat::Median,
        };
        let block = PermTest {
            b: 4,
            seed: 5,
            variant: PermVariant::Block { len: 8 },
            stat: PermStat::Median,
        };
        for t in [&circ, &block] {
            for _ in 0..4 {
                t.draw_pairing(n, None, &mut s);
                let mut seen = [false; 100];
                for &v in &s.perm_idx[..n] {
                    assert!((v as usize) < n, "out of range");
                    assert!(!seen[v as usize], "duplicate in pairing");
                    seen[v as usize] = true;
                }
            }
        }
        // Circular pairing is a rotation: perm_idx[i] = (i + off) % n.
        circ.draw_pairing(n, None, &mut s);
        let step = s.perm_idx[0] as usize;
        for (i, &pi) in s.perm_idx[1..n].iter().enumerate() {
            assert_eq!(pi as usize, (i + 1 + step) % n);
        }
        // Block pairing preserves within-block offsets: idx[i] ≡ i (mod L).
        block.draw_pairing(n, None, &mut s);
        for (i, &pi) in s.perm_idx[..n].iter().enumerate() {
            assert_eq!((pi as usize) % 8, i % 8, "block offset broken at {i}");
        }
    }

    #[test]
    fn stratified_null_stays_within_strata() {
        let n = 60;
        let strata: Vec<u32> = (0..n).map(|i| (i / 10) as u32).collect(); // 6 strata × 10
        let mut s = MiScratch::new(n, 1, 13);
        let test = PermTest::new(4, 17);
        for _ in 0..4 {
            test.draw_pairing(n, Some(&strata), &mut s);
            for i in 0..n {
                assert_eq!(
                    strata[s.perm_idx[i] as usize], strata[i],
                    "cross-stratum pairing at {i}"
                );
            }
            let mut seen = [false; 60];
            for &v in &s.perm_idx[..n] {
                assert!(!seen[v as usize]);
                seen[v as usize] = true;
            }
        }
    }

    #[test]
    fn antithetic_q_averaging_is_deterministic_and_finite() {
        let n = 256;
        let (x, y) = gaussian_pairs(0.4, n, 21);
        let test = PermTest::new(8, 99);
        let mut s1 = MiScratch::new(n, 1, 1);
        let mut s2 = MiScratch::new(n, 1, 1);
        let a = test.dv_null_q_mean(Critic::Dot, &x, &y, n, 1, &mut s1);
        let b = test.dv_null_q_mean(Critic::Dot, &x, &y, n, 1, &mut s2);
        assert_eq!(a, b, "re-seeded run must be bit-identical");
        assert!(a.0.is_finite() && a.1.is_finite());
        let (dv, dv_std) = dv_with_antithetic_q(Critic::Dot, &x, &y, n, 1, &test, &mut s1);
        assert!(dv.is_finite() && dv_std.is_finite());
    }

    #[test]
    fn perm_reseeding_gives_run_to_run_bit_determinism() {
        let n = 256;
        let (x, y) = gaussian_pairs(0.35, n, 33);
        let test = PermTest::new(32, 1234);
        let mut s = MiScratch::new(n, 1, 8);
        let r1 = test.run(Critic::Dot, &x, &y, n, 1, None, &mut s);
        // Contaminate the scratch RNG stream between runs.
        s.rng = fastrand::Rng::new();
        let _ = s.rng.u64(0..1_000_000);
        let r2 = test.run(Critic::Dot, &x, &y, n, 1, None, &mut s);
        assert_eq!(r1, r2, "PermTest must be scratch-history independent");
    }

    #[test]
    fn dv_current_matches_dv_plug_in() {
        let n = 128;
        let (x, y) = gaussian_pairs(0.2, n, 55);
        let mut s = MiScratch::new(n, 1, 2);
        s.score_joint(Critic::Dot, &x, &y, n, 1);
        s.next_perm(n);
        s.score_perm(Critic::Dot, &x, &y, n, 1, PermSource::Current);
        assert_eq!(dv_current(&s, n), dv_plug_in(&s.joint, &s.perm));
    }
}
