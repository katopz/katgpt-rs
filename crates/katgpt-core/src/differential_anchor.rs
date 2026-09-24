//! Differential anchor scoring — Issue 882 P0 (Research 586, Diff
//! Transformer arXiv:2410.05258 §3: the subtract arm of the
//! attention-noise-control family — `SSMax` rescales UP, `ASEntmax` sparsifies
//! DOWN, this one SUBTRACTS a correlated reference).
//!
//! # The primitive
//!
//! `score(q, d) = sim(q, d) − λ·sim(ā, d) = (q − λ·ā)·d` — one axpy
//! produces the corrected query `q̂ = q − λā`; the existing KNN/rerank
//! pass runs unchanged. Candidates similar-to-everything (hubs) lose
//! their generic mass; query-specific candidates keep theirs. Prior art
//! (honest, Research 586 §4): CSLS subtracts per-candidate neighborhood
//! hubness (the single-shared-anchor form is the O(d)-per-query
//! simplification); contrastive decoding owns the subtract principle at
//! logits level. Ours is the integration on the score surfaces we own
//! (retrieval/routing/abstention — no trained weights), not a new
//! principle.
//!
//! # The pieces (P0 scope)
//!
//! - [`correct_query`] / [`correct_query_into`]: the one-axpy correction,
//!   λ=0 bit-identical (the kill switch — G3).
//! - [`differential_score`]: the fused two-dot form for one-off scoring
//!   (no corrected-query materialization).
//! - [`AnchorBuilder`] + [`AnchorSource`]: fits the anchor from observed
//!   vectors via the shared [`crate::fitted_anchor_table::StreamingMeanTable`]
//!   substrate (Issue 883's table builder — one substrate, two
//!   consumers). Two sources, A/B'd per domain on fixtures, **never
//!   assumed** (trap 1: hubness ≠ illegitimacy — mechanical lints are
//!   CORRECTLY high-frequency): `MeanQuery` (the generic-query direction
//!   — the panel's safer default) and `MeanCorpus` (the corpus centroid).
//! - [`lambda_init`] + [`LAMBDA_INIT_TABLE`]: the frozen per-layer prior
//!   `λinit(l) = 0.8 − 0.6·exp(−0.3(l−1))` (0.2 → 0.8, residual halves
//!   ≈ every 2.31 layers) — the paper's own schedule, pasted as a const
//!   table and pinned by test against the closed form. **One training
//!   run's hyperparameters, not a law** (trap 5): override-able prior,
//!   never a hard default.
//! - [`reparam_lambda`]: the neutral-at-zero reparameterization
//!   `λ = e^u − e^v + λinit(l)` — latents at 0 ⇒ the prior exactly (the
//!   paper's own λ reparam shape; the envelope for any runtime-tunable
//!   seeded by a frozen prior).
//! - [`pick_lambda_by_eval`]: λ* by **direct evaluation on oracle
//!   fixtures** (grid + argmax, deterministic smallest-λ tie-break —
//!   ties must not drift toward more correction), never GD.
//! - [`hubness_skewness`]: the G1 gate's statistic — skewness of the
//!   per-candidate mean-similarity distribution (`S_N` strictly decreases
//!   ⇒ generic mass was removed; the *correctness* read stays with the
//!   oracle top-1, since concentration ≠ relevance).
//!
//! # Gates (P0)
//!
//! - **G1**: oracle-fixture top-1 at λ* not worse than λ=0 AND hubness
//!   skewness strictly decreases (the synthetic hub-world test).
//! - **G2**: O(d) axpy ≤ 1 µs at d=512, invisible against the ~123 µs
//!   latent-KNN p50; corrected-vs-plain full scoring pass ≤ 2% (`bench_886`,
//!   the `ab_median_ratio` paired protocol — never two sequential arms).
//! - **G3**: λ=0 bit-identical, pinned below.
//! - **G4**: stack-buffer query, zero steady-state allocs (`bench_886`
//!   counting allocator).
//!
//! Opt-in feature `differential_anchor` (implies `fitted_anchor_tables`)
//! per the no-default-consumer rule; promotion rides a live consumer's
//! GOAT (882 P0 rider: the healer rerank lane).

use crate::fitted_anchor_table::StreamingMeanTable;

/// Frozen per-layer λ prior, `λinit(l) = 0.8 − 0.6·exp(−0.3·(l−1))` for
/// 1-based layer `l` (l=1 ⇒ 0.2; l→∞ ⇒ 0.8). Override-able prior, never
/// a hard default (trap 5). Layer 0 clamps to layer 1 (the 1-based
/// contract).
#[inline]
#[must_use]
pub fn lambda_init(layer_1_based: usize) -> f32 {
    let l = if layer_1_based == 0 { 1 } else { layer_1_based };
    0.8 - 0.6 * (-0.3 * (l - 1) as f32).exp()
}

/// The frozen `λinit` table for layers 1..=64 — the schedule as a const
/// lookup (the `static_cal_tables` pattern: `exp` is not const-callable,
/// so the values ship pre-evaluated and `lambda_init_table_pins_the_
/// closed_form` pins every entry to the closed form so the paste cannot
/// drift from its formula).
#[allow(clippy::excessive_precision)] // pasted schedule values, pinned by test (the kinematics/perception.rs A&S-coefficients precedent)
pub const LAMBDA_INIT_TABLE: [f32; 64] = [
    0.2000000, 0.355_509_1, 0.4707130, 0.556_058_2, 0.619_283_5, 0.666_121_9, 0.700_820_7, 0.726_526_1,
    0.745_569_2, 0.759_676_7, 0.770_127_8, 0.777_870_1, 0.783_605_8, 0.787_854_9, 0.791_002_7, 0.793_334_6,
    0.795_062_2, 0.7963420, 0.797_290_1, 0.797_992_4, 0.798_512_7, 0.798_898_2, 0.799_183_8, 0.799_395_3,
    0.7995520, 0.799_668_1, 0.799_754_2, 0.799_817_9, 0.799_865_1, 0.7999000, 0.7999260, 0.799_945_1,
    0.799_959_4, 0.799_969_9, 0.799_977_7, 0.799_983_5, 0.799_987_8, 0.799_990_9, 0.799_993_3, 0.7999950,
    0.799_996_3, 0.799_997_3, 0.7999980, 0.799_998_5, 0.799_998_9, 0.799_999_2, 0.799_999_4, 0.799_999_5,
    0.799_999_7, 0.799_999_8, 0.799_999_8, 0.799_999_9, 0.799_999_9, 0.799_999_9, 0.799_999_9, 0.8000000,
    0.8000000, 0.8000000, 0.8000000, 0.8000000, 0.8000000, 0.8000000, 0.8000000, 0.8000000,
];

/// The one-axpy query correction, in place: `q̂ = q − λ·ā`.
///
/// **λ == 0.0 returns with `q` bit-identical** (G3 kill switch — no
/// arithmetic at all, so no −0.0/rounding surprises). Panics in debug on
/// width mismatch (a wiring bug must be loud).
#[inline]
pub fn correct_query(q: &mut [f32], anchor: &[f32], lambda: f32) {
    debug_assert_eq!(
        q.len(),
        anchor.len(),
        "differential_anchor: query/anchor width mismatch"
    );
    if lambda == 0.0 {
        return;
    }
    for (qi, &ai) in q.iter_mut().zip(anchor.iter()) {
        *qi -= lambda * ai;
    }
}

/// Out-of-place correction into caller scratch (`out = q − λ·ā`; `q`
/// untouched). λ == 0.0 is a plain copy — bit-identical to the input.
#[inline]
pub fn correct_query_into(q: &[f32], anchor: &[f32], lambda: f32, out: &mut [f32]) {
    debug_assert_eq!(q.len(), anchor.len());
    debug_assert_eq!(q.len(), out.len());
    if lambda == 0.0 {
        out.copy_from_slice(q);
        return;
    }
    for ((&qi, &ai), oi) in q.iter().zip(anchor.iter()).zip(out.iter_mut()) {
        *oi = qi - lambda * ai;
    }
}

/// The fused one-off form: `(q − λā)·d = q·d − λ·(ā·d)` — two dots, no
/// corrected-query materialization (for scoring surfaces that score each
/// query once rather than re-running a KNN pass).
#[inline]
#[must_use]
pub fn differential_score(q: &[f32], anchor: &[f32], lambda: f32, d: &[f32]) -> f32 {
    debug_assert_eq!(q.len(), d.len());
    debug_assert_eq!(anchor.len(), d.len());
    let mut qd = 0.0f32;
    let mut ad = 0.0f32;
    for i in 0..d.len() {
        qd += q[i] * d[i];
        ad += anchor[i] * d[i];
    }
    qd - lambda * ad
}

/// Which observed vectors fit the anchor — the P0 A/B pair (trap 1:
/// adjudicated per domain on fixtures, never assumed; `MeanQuery` is the
/// panel's safer default — the generic-QUERY direction, so hubs lose
/// exactly the mass a generic query gives them).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorSource {
    /// ā = mean of observed QUERY vectors (the generic query).
    MeanQuery,
    /// ā = mean of observed CORPUS/candidate vectors (the corpus centroid).
    MeanCorpus,
}

/// Fits one anchor row from observed vectors on the shared
/// [`StreamingMeanTable`] substrate (883's builder — one substrate, two
/// consumers). The finished anchor is **L2-normalized** (λ then reads in
/// query-scale units on unit-normalized embeddings; a zero observation
/// set finishes as the zero vector — no direction, so `correct_query` at
/// any λ is rank-preserving, never NaN).
pub struct AnchorBuilder {
    table: StreamingMeanTable,
}

impl AnchorBuilder {
    /// One anchor row of `width` dims (the source choice — which vectors
    /// you `observe` — is the caller's A/B; the builder itself is
    /// source-agnostic).
    #[must_use]
    pub fn new(width: usize) -> Self {
        Self {
            table: StreamingMeanTable::new(1, width),
        }
    }

    /// Observe one vector (query or corpus row per [`AnchorSource`]).
    /// Alloc-free (the substrate's observe).
    pub fn observe(&mut self, x: &[f32]) {
        self.table.observe(0, x);
    }

    /// Observations fitted so far.
    #[must_use]
    pub fn n(&self) -> u64 {
        self.table.count(0)
    }

    /// The fitted anchor: mean of observations, L2-normalized, into
    /// caller scratch. Empty builder ⇒ zero row.
    pub fn finish_into(&self, out: &mut [f32]) {
        self.table.mean_into(0, out);
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for v in out.iter_mut() {
                *v *= inv;
            }
        }
    }

    /// Convenience: fit + finish in one pass over the observations
    /// (allocating — offline calibration only).
    #[must_use]
    pub fn fit(source: AnchorSource, queries: &[&[f32]], corpus: &[&[f32]]) -> Vec<f32> {
        let rows: &[&[f32]] = match source {
            AnchorSource::MeanQuery => queries,
            AnchorSource::MeanCorpus => corpus,
        };
        let width = rows.first().map_or(0, |r| r.len());
        let mut b = Self::new(width);
        for r in rows {
            b.observe(r);
        }
        let mut out = vec![0.0; width];
        b.finish_into(&mut out);
        out
    }
}

/// λ* by direct evaluation: run `eval(λ)` per grid point, argmax, with
/// the **deterministic smallest-λ tie-break** (ties must not drift toward
/// more correction — the conservative-default law). Never GD (the Plan
/// 340 conformal-calibration precedent).
#[must_use]
pub fn pick_lambda_by_eval<F: Fn(f32) -> f32>(grid: &[f32], eval: F) -> f32 {
    let mut best = grid[0];
    let mut best_v = eval(grid[0]);
    for &lam in &grid[1..] {
        let v = eval(lam);
        if v > best_v {
            best = lam;
            best_v = v;
        }
    }
    best
}

/// Skewness (third standardized moment, population form, f64
/// accumulation) of a sample — the G1 hubness statistic when fed the
/// per-candidate **top-1 win counts** (the CSLS k-occurrence form: a hub
/// = wins many queries' nearest-neighbor slot). Strictly decreasing
/// under anchor correction ⇒ generic mass was removed. Fewer than 3
/// samples or zero-variance input ⇒ 0 (no discrimination, not NaN).
///
/// ⚠ **Do NOT feed this the mean-similarity distribution when the anchor
/// is the mean query** — measured invariant BY CONSTRUCTION (the G1
/// session's finding): `mean_q[(q−λā)·d] = (1−λ/|m|)·mean_q[q·d]` — a pure
/// scalar per candidate — and standardized skewness is scale-invariant,
/// so that statistic cannot move at any λ. The correction's effect lives
/// in PER-QUERY rankings (the per-query penalty λ·(ā·d) varies across
/// candidates), which the win-count distribution captures exactly.
#[must_use]
pub fn hubness_skewness(xs: &[f32]) -> f32 {
    let n = xs.len();
    if n < 3 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = xs.iter().map(|&x| f64::from(x)).sum::<f64>() / nf;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &x in xs {
        let d = f64::from(x) - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= nf;
    m3 /= nf;
    if m2 <= f64::EPSILON {
        return 0.0;
    }
    (m3 / (m2 * m2.sqrt())) as f32
}

/// Neutral-at-zero reparameterization: `λ = e^u − e^v + λinit(l)` —
/// latents (0, 0) ⇒ λ = λinit exactly (the frozen prior; the paper's own
/// gradient-flow reparam shape). The envelope ANY runtime-tunable seeded
/// by a frozen prior should use: neutral latents never move the default.
#[inline]
#[must_use]
pub fn reparam_lambda(u: f32, v: f32, layer_1_based: usize) -> f32 {
    u.exp() - v.exp() + lambda_init(layer_1_based)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// G3: λ=0 is bit-identical — the kill switch.
    #[test]
    fn lambda_zero_is_bit_identical() {
        let q0 = [0.25f32, -1.5, 3.0, -0.0, 1e-20, 1e20];
        let anchor = [1.0f32, 2.0, -3.0, 0.5, 1.0, -1.0];
        let mut q = q0;
        correct_query(&mut q, &anchor, 0.0);
        assert!(
            q.iter().zip(q0.iter()).all(|(a, b)| a.to_bits() == b.to_bits()),
            "in-place λ=0 must not touch bits"
        );

        let mut out = [0.0f32; 6];
        correct_query_into(&q0, &anchor, 0.0, &mut out);
        assert!(out
            .iter()
            .zip(q0.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// The correction is one axpy: known-value check, fused form agrees
    /// with the materialized form.
    #[test]
    fn correction_is_one_axpy() {
        let q = [1.0f32, 2.0, 3.0];
        let a = [0.5f32, -0.5, 1.0];
        let mut out = [0.0f32; 3];
        correct_query_into(&q, &a, 2.0, &mut out);
        for i in 0..3 {
            let want = q[i] - 2.0 * a[i];
            assert!((out[i] - want).abs() < 1e-6);
        }
        let d = [1.0f32, 1.0, -1.0];
        let fused = differential_score(&q, &a, 2.0, &d);
        let mat: f32 = out.iter().zip(d.iter()).map(|(x, y)| x * y).sum();
        assert!((fused - mat).abs() < 1e-4);
    }

    /// λinit closed form: l=1 ⇒ 0.2 exactly; non-decreasing toward 0.8
    /// (strict while the f32 increments exceed 1 ULP — past l≈50 the
    /// residual is sub-ULP and the f32 curve flattens by construction);
    /// the frozen table pins the closed form everywhere.
    #[test]
    fn lambda_init_law_and_table_pin() {
        assert!((lambda_init(1) - 0.2).abs() < 1e-6);
        assert!((lambda_init(2) - (0.8 - 0.6 * 0.740_818_2)).abs() < 1e-6);
        let mut prev = lambda_init(1);
        for l in 2..=50 {
            let v = lambda_init(l);
            assert!(v > prev, "λinit must be strictly ↑ through l=50 (l={l})");
            assert!(v <= 0.8 + 1e-6);
            prev = v;
        }
        for l in 51..=80 {
            assert!(lambda_init(l) >= prev - 1e-7);
            prev = lambda_init(l);
        }
        assert!((lambda_init(64) - 0.8).abs() < 1e-4);
        // Layer 0 clamps to layer 1 (1-based contract).
        assert_eq!(lambda_init(0), lambda_init(1));
        // The pasted const table pins the closed form (max |Δ| < 1e-6).
        for (i, &t) in LAMBDA_INIT_TABLE.iter().enumerate() {
            let want = lambda_init(i + 1);
            assert!(
                (t - want).abs() < 1e-6,
                "LAMBDA_INIT_TABLE[{i}] = {t} vs closed form {want}"
            );
        }
        assert_eq!(LAMBDA_INIT_TABLE[0], 0.2);
    }

    /// Neutral-at-zero: (0,0) latents ⇒ the prior, bit-exactly; latents
    /// move λ off the prior in both directions.
    #[test]
    fn reparam_neutral_at_zero() {
        for l in [1usize, 2, 8, 33, 64] {
            let prior = lambda_init(l);
            assert_eq!(reparam_lambda(0.0, 0.0, l), prior);
        }
        assert!(reparam_lambda(0.5, 0.0, 4) > lambda_init(4));
        assert!(reparam_lambda(0.0, 0.5, 4) < lambda_init(4));
    }

    /// λ* grid: argmax with the smallest-λ tie-break.
    #[test]
    fn pick_lambda_tie_breaks_to_smallest() {
        let grid = [0.0f32, 0.25, 0.5, 1.0];
        assert_eq!(pick_lambda_by_eval(&grid, |_| 0.5), 0.0);
        assert_eq!(pick_lambda_by_eval(&grid, |l| 1.0 - (l - 0.25).abs()), 0.25);
        assert_eq!(
            pick_lambda_by_eval(&grid, |l| if l >= 0.5 { 1.0 } else { 0.0 }),
            0.5
        );
    }

    /// Skewness: symmetric ⇒ 0; right-tailed ⇒ > 0; degenerate ⇒ 0.
    #[test]
    fn skewness_shapes() {
        assert!(hubness_skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]).abs() < 1e-6);
        assert!(hubness_skewness(&[0.0, 0.1, 0.1, 0.1, 5.0]) > 0.5);
        assert_eq!(hubness_skewness(&[2.0; 8]), 0.0);
        assert_eq!(hubness_skewness(&[]), 0.0);
    }

    /// G1 on a synthetic hub world (deterministic, no RNG): every
    /// query = private direction + shared generic direction; hubs carry
    /// only the generic direction; true matches carry the private one.
    /// At λ=0 hubs win every query; at λ* (grid-evaluated) the true
    /// matches recover; the hubness skewness strictly decreases.
    #[test]
    fn g1_synthetic_hub_world() {
        const D: usize = 48;
        const Q: usize = 24;
        const N_HUBS: usize = 8;

        // q_i = normalize(e_{2i} + 2·h), h = the uniform direction —
        // cos(q_i, h) ≈ 0.91 (generic mass dominates), cos(q_i, true_i)
        // ≈ 0.48, so hubs win every query at λ=0.
        let h: Vec<f32> = vec![1.0 / (D as f32).sqrt(); D];
        let mut queries = Vec::with_capacity(Q);
        for i in 0..Q {
            let mut q = vec![0.0f32; D];
            q[i * 2] = 1.0;
            for (qi, hv) in q.iter_mut().zip(h.iter()) {
                *qi += 2.0 * hv;
            }
            let n = q.iter().map(|v| v * v).sum::<f32>().sqrt();
            for v in q.iter_mut() {
                *v /= n;
            }
            queries.push(q);
        }
        // True matches: the query's private block + parity-signed twin
        // coordinate (deterministic private noise).
        let mut trues = Vec::with_capacity(Q);
        for i in 0..Q {
            let mut t = vec![0.0f32; D];
            t[i * 2] = 1.0;
            t[i * 2 + 1] = 0.5 * if i % 2 == 0 { 1.0 } else { -1.0 };
            let n = t.iter().map(|v| v * v).sum::<f32>().sqrt();
            for v in t.iter_mut() {
                *v /= n;
            }
            trues.push(t);
        }
        // Hubs: the generic direction with tiny per-hub jitter.
        let mut hubs = Vec::with_capacity(N_HUBS);
        for j in 0..N_HUBS {
            let mut hb = h.clone();
            hb[j] += 0.05;
            let n = hb.iter().map(|v| v * v).sum::<f32>().sqrt();
            for v in hb.iter_mut() {
                *v /= n;
            }
            hubs.push(hb);
        }
        let mut cands: Vec<&[f32]> = Vec::with_capacity(Q + N_HUBS);
        for t in &trues {
            cands.push(t.as_slice());
        }
        for hb in &hubs {
            cands.push(hb.as_slice());
        }

        // Anchors: mean-query (the safer default) and mean-corpus.
        let qrefs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
        let a_query = AnchorBuilder::fit(AnchorSource::MeanQuery, &qrefs, &cands);
        let a_corpus = AnchorBuilder::fit(AnchorSource::MeanCorpus, &qrefs, &cands);

        let top1 = |anchor: &[f32], lambda: f32| -> usize {
            let mut hits = 0usize;
            let mut qh = vec![0.0f32; D];
            for (qi, q) in queries.iter().enumerate() {
                correct_query_into(q, anchor, lambda, &mut qh);
                let mut best = 0usize;
                let mut best_s = f32::NEG_INFINITY;
                for (c, d) in cands.iter().enumerate() {
                    let s: f32 = qh.iter().zip(d.iter()).map(|(x, y)| x * y).sum();
                    if s > best_s {
                        best_s = s;
                        best = c;
                    }
                }
                if best == qi {
                    hits += 1;
                }
            }
            hits
        };
        let mean_sims = |anchor: &[f32], lambda: f32| -> Vec<f32> {
            let mut out = vec![0.0f32; cands.len()];
            let mut qh = vec![0.0f32; D];
            for q in &queries {
                correct_query_into(q, anchor, lambda, &mut qh);
                for (o, d) in out.iter_mut().zip(cands.iter()) {
                    *o += qh.iter().zip(d.iter()).map(|(x, y)| x * y).sum::<f32>();
                }
            }
            let inv = 1.0 / queries.len() as f32;
            for o in out.iter_mut() {
                *o *= inv;
            }
            out
        };
        // The mean-query anchor scales the mean-sim distribution UNIFORMLY
        // (the module doc's measured invariant) — pinned after λ* below.

        let base = top1(&a_query, 0.0);
        assert_eq!(base, 0, "fixture must have hubs winning at λ=0");

        let grid: Vec<f32> = (0..=40).map(|i| i as f32 * 0.05).collect();
        let lam_star = pick_lambda_by_eval(&grid, |lam| top1(&a_query, lam) as f32);
        let corrected = top1(&a_query, lam_star);
        assert!(
            corrected >= base,
            "G1: top-1 at λ* must not be worse ({corrected} vs {base})"
        );
        assert!(
            corrected >= Q / 2,
            "the fixture's headroom must show (λ*={lam_star}): got {corrected}/{Q}"
        );

        // Pin the measured invariant itself: mean-sims scale uniformly and
        // their skewness cannot move — so nobody re-derives the "statistic
        // that cannot move" as a future G1 gate.
        {
            let s0 = mean_sims(&a_query, 0.0);
            let s1 = mean_sims(&a_query, lam_star);
            let ratio: Vec<f32> = s0.iter().zip(&s1).map(|(a, b)| a / b).collect();
            let spread = ratio.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                - ratio.iter().cloned().fold(f32::INFINITY, f32::min);
            assert!(spread < 1e-3, "mean-sim must scale uniformly (spread {spread})");
            let sk0 = hubness_skewness(&s0);
            let sk1 = hubness_skewness(&s1);
            assert!((sk0 - sk1).abs() < 1e-4, "and its skewness is invariant");
        }

        // G1's hubness statistic: per-candidate TOP-1 WIN COUNTS (the CSLS
        // k-occurrence form — the module doc's invariant makes the
        // mean-sim skewness unusable under a mean-query anchor). Hubs
        // concentrate all 24 wins at λ=0 (right-skewed); after correction
        // the wins spread one-per-true-candidate (flat/left-skewed) —
        // skewness strictly decreases.
        let win_counts = |anchor: &[f32], lambda: f32| -> Vec<f32> {
            let mut wins = vec![0.0f32; cands.len()];
            let mut qh = vec![0.0f32; D];
            for q in &queries {
                correct_query_into(q, anchor, lambda, &mut qh);
                let mut best = 0usize;
                let mut best_s = f32::NEG_INFINITY;
                for (c, d) in cands.iter().enumerate() {
                    let s: f32 = qh.iter().zip(d.iter()).map(|(x, y)| x * y).sum();
                    if s > best_s {
                        best_s = s;
                        best = c;
                    }
                }
                wins[best] += 1.0;
            }
            wins
        };
        let s_before = hubness_skewness(&win_counts(&a_query, 0.0));
        let s_after = hubness_skewness(&win_counts(&a_query, lam_star));
        assert!(
            s_after < s_before,
            "hubness skewness must strictly decrease: {s_after} vs {s_before}"
        );

        // A/B: on THIS fixture mean-query beats mean-corpus (recorded as
        // a fixture-specific read; per-domain adjudication stays the law).
        let corpus_lam = pick_lambda_by_eval(&grid, |lam| top1(&a_corpus, lam) as f32);
        let corpus_hits = top1(&a_corpus, corpus_lam);
        assert!(
            corrected >= corpus_hits,
            "mean-query is the safer default on this fixture ({corrected} vs {corpus_hits})"
        );
    }
}
