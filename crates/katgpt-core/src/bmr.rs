//! bmr — Bayesian Model Reduction + EFE-over-models: modelless
//! structure-selection primitives (Plan 597 / Research 551, distilled from
//! Friston et al., "Active inference and artificial reasoning", Nat Commun
//! 2026, DOI 10.1038/s41467-026-77209-5 / arXiv:2512.21129).
//!
//! # What ships here (equation numbers = the paper)
//!
//! | Primitive | Eq | What it computes |
//! |---|---|---|
//! | [`bmr_log_evidence`] | 7/9 | per-column BMR free energy of a reduced Dirichlet model vs the full model |
//! | [`posterior_over_models`] | 9 | posterior over rival models (uniform model prior, log-sum-exp) |
//! | [`ModelSpace::predictive_posterior_into`] | 11 | sparse-Δ predictive posterior: ONE count increment, O(#models), no lgamma on the hot path |
//! | [`ModelSpace::efe_model_gain`] | 10 | expected information gain over models: `Σ_o P(o|u) · KL(Q(m|·,o,u) ‖ Q(m|·,u))` |
//! | [`occam_log_bayes_factor`] | 12 | `ln[p*/(1−p*)]` commit statistic |
//! | [`enumerate_isomorphic_rules`] | 14 | generic model-space enumerator over choice/criterion/context factors |
//!
//! # Sign convention (load-bearing)
//!
//! [`bmr_log_evidence`] computes the BMR free energy
//! `F(m) = Σ_cols [ln B(a_c) − ln B(ã_c) + ln B(ã_m,c) − ln B(ã_m,c + a_c − ã_c)]`
//! — the exact log Bayes factor `ln p(D|full) − ln p(D|m)` from the BMR
//! identity (Friston et al. 2018: `ln p(y|m) = ln B(ã_m) + ln B(ã_m + a − ã)
//! − ln B(a) − ln B(ã)`). **Lower F = more evidence for model m**; every
//! posterior in this module is `softmax(−F)`, under which the model-
//! independent terms cancel exactly. The magnitude is pinned against an
//! exact marginal-likelihood oracle (sequential posterior-predictive
//! chaining, no lgamma) in `bmr_matches_brute_force_evidence`; the sign is
//! pinned in `evidence_favors_data_consistent_model`. NOTE: Research 551's
//! Eq-7/9 transcription carries the `ln B(a_c)` term's sign flipped (a
//! "+ln B(ã) + ln B(a) − …" form), which contradicts the T2.2 oracle by the
//! model-dependent amount `2·Σ_c ln B(ã_m,c)` — the oracle (the plan's
//! designated G1 anchor) wins, and the corrected form is implemented here.
//!
//! # Conventions
//!
//! - **Counts contract**: `post` = `prior` + accumulated non-negative
//!   observations; the BMR delta is `n = post − prior`, entrywise per column.
//! - **Shrinkage**: zeroed concentration parameters in a REDUCED prior are
//!   read at [`SHRINKAGE`] (= 1/32, the paper's convention) — an all-zero
//!   reduced column scores exactly as an all-(1/32) column. Full priors and
//!   posteriors are expected strictly positive on used columns.
//! - **×512 novelty suppression** (canonical reading, consumer concern):
//!   after an Occam commit ([`occam_log_bayes_factor`] > 16 nats) the
//!   consumer replaces its posterior with the selected model and multiplies
//!   that model's Dirichlet counts by 512, entering exploitation; planning
//!   reverts to the committed model. This module DOCUMENTS the convention —
//!   the commit policy itself belongs to the consumer.
//!
//! # Scope guard (anti-FEP, Research 551 §2.4)
//!
//! This primitive serves the **discovery axis only**: epistemic foraging
//! under unknown likelihood structure. It must NOT be wired into policy
//! extraction over known/frozen models — MOP (Research 478) and exact HMM
//! control (Research 543) own the exploitation axis, per the standing
//! anti-FEP ledger. After the Occam commit, planning reverts to the committed
//! model, where those incumbents remain the policy extractors.
//!
//! # Substrate
//!
//! All lgamma arithmetic goes through the shared Lanczos kernel
//! `crate::special_fn::ln_gamma` (Plan 597 T1.1 — the same kernel
//! `best_belief` consumes; one ln_gamma per crate). Pure math + std +
//! arrayvec; cross-compiles to wasm32; hot paths are zero-allocation with
//! caller-supplied scratch. Column values live in fixed-size [`Counts`]
//! stores (bounded by [`MAX_OUT`] rows / [`MAX_COLS`] columns) so nothing on
//! the hot path touches the heap.

use arrayvec::ArrayVec;
use crate::special_fn::ln_gamma;

/// Maximum outcomes (rows) per column. The paper's modalities are ≤ 4.
pub const MAX_OUT: usize = 64;

/// Maximum columns per [`Counts`] store (the three-ball layout needs 81).
pub const MAX_COLS: usize = 128;

/// Maximum models in one [`ModelSpace`] (the three-ball space has 81 rules).
pub const MAX_MODELS: usize = 128;

/// Maximum context factors in a [`FactorLayout`].
pub const MAX_FACTORS: usize = 8;

/// The paper's shrinkage convention: zeroed concentration parameters are
/// read at 1/32. Keeps `ln B` finite (ln Γ(0) = +∞) and makes reduced
/// models with zeroed columns comparable.
pub const SHRINKAGE: f64 = 1.0 / 32.0;

// ──────────────────────────────────────────────────────────────────────────
// ln_beta + incremental column cache (T1.2)
// ──────────────────────────────────────────────────────────────────────────

/// Multi-arg (Dirichlet-normalizer) log Beta function:
/// `ln B(x) = Σ_i ln Γ(x_i) − ln Γ(Σ_i x_i)`.
///
/// All entries must be > 0 (a zero entry gives ln Γ(0) = +∞; an all-zero
/// slice is ∞ − ∞ = NaN). Reduced-prior callers should read zeros at
/// [`SHRINKAGE`] first — the bmr paths below do this internally.
pub fn ln_beta(x: &[f64]) -> f64 {
    debug_assert!(!x.is_empty(), "ln_beta of empty slice");
    let mut sum_ln_gamma = 0.0;
    let mut sum = 0.0;
    for &v in x {
        sum_ln_gamma += ln_gamma(v);
        sum += v;
    }
    sum_ln_gamma - ln_gamma(sum)
}

/// Per-column cached sums of a [`Counts`] store: `Σ_i x_i` and `Σ_i ln Γ(x_i)`
/// per column, maintained incrementally by the caller between uses.
///
/// Raw reads (no shrinkage): build from strictly-positive counts, or apply
/// [`SHRINKAGE`] yourself before caching.
#[derive(Clone, Debug)]
pub struct ColumnSums {
    sum: ArrayVec<f64, MAX_COLS>,
    ln_gamma_sum: ArrayVec<f64, MAX_COLS>,
}

impl ColumnSums {
    /// O(entries) full build.
    pub fn from_counts(counts: &Counts) -> Self {
        let mut sum = ArrayVec::new();
        let mut ln_gamma_sum = ArrayVec::new();
        for c in 0..counts.cols() {
            let col = counts.col(c);
            sum.push(col.iter().sum());
            ln_gamma_sum.push(col.iter().map(|&v| ln_gamma(v)).sum());
        }
        Self { sum, ln_gamma_sum }
    }

    /// `Σ_i x_i` of column `col` (cached).
    pub fn sum(&self, col: usize) -> f64 {
        self.sum[col]
    }

    /// Cached `ln B` of column `col` (without re-deriving the sums).
    pub fn ln_beta(&self, col: usize) -> f64 {
        self.ln_gamma_sum[col] - ln_gamma(self.sum[col])
    }

    /// Fold one more observation `(row, w)` into the cache for `col`.
    /// Uses `ln Γ(x+w) − ln Γ(x)` corrections — exact for any `w ≥ 0`.
    pub fn accumulate(&mut self, counts: &Counts, col: usize, row: usize, w: f64) {
        let x = counts.col(col)[row];
        self.ln_gamma_sum[col] += ln_gamma(x + w) - ln_gamma(x);
        self.sum[col] += w;
    }
}

/// `ln B` of column `col` with `delta` added, from a [`ColumnSums`] cache:
/// `Σ_i ln Γ(x_i + δ_i) − ln Γ(Σ x_i + Σ δ_i)`. Only nonzero delta entries
/// are corrected, so sparse fractional deltas cost `(#nonzero + 1)` lgamma
/// evaluations instead of a full column recompute.
pub fn ln_beta_delta(counts: &Counts, base: &ColumnSums, col: usize, delta: &[f64]) -> f64 {
    let x = counts.col(col);
    let mut correction = 0.0;
    let mut delta_sum = 0.0;
    for (i, &d) in delta.iter().enumerate().take(x.len()) {
        if d != 0.0 {
            correction += ln_gamma(x[i] + d) - ln_gamma(x[i]);
            delta_sum += d;
        }
    }
    base.ln_gamma_sum[col] + correction - ln_gamma(base.sum[col] + delta_sum)
}

/// Read a zeroed concentration parameter at [`SHRINKAGE`].
#[inline]
fn shrunk(v: f64) -> f64 {
    if v > 0.0 {
        v
    } else {
        SHRINKAGE
    }
}

/// `ln B` of one column with every zero entry read at [`SHRINKAGE`].
fn ln_beta_shrunk_col(col: &[f64]) -> f64 {
    let mut sum_ln_gamma = 0.0;
    let mut sum = 0.0;
    for &v in col {
        let v = shrunk(v);
        sum_ln_gamma += ln_gamma(v);
        sum += v;
    }
    sum_ln_gamma - ln_gamma(sum)
}

// ──────────────────────────────────────────────────────────────────────────
// Counts — fixed-size column store (T1.3)
// ──────────────────────────────────────────────────────────────────────────

/// Fixed-capacity column store: `cols` columns × `rows` outcomes, flat
/// row-major-per-column layout inside one `ArrayVec` — no heap per store,
/// bounded by [`MAX_OUT`] × [`MAX_COLS`]. Column `c` occupies
/// `[c*rows, (c+1)*rows)`.
#[derive(Clone)]
pub struct Counts {
    rows: usize,
    data: ArrayVec<f64, { MAX_OUT * MAX_COLS }>,
}

impl Counts {
    /// Zero-filled store. Panics if `rows`/`cols` exceed the bounds.
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self::filled(rows, cols, 0.0)
    }

    /// Store with every entry set to `value`.
    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        assert!((1..=MAX_OUT).contains(&rows), "rows {rows} out of 1..={MAX_OUT}");
        assert!((1..=MAX_COLS).contains(&cols), "cols {cols} out of 1..={MAX_COLS}");
        let mut data = ArrayVec::new();
        for _ in 0..rows * cols {
            data.push(value);
        }
        Self { rows, data }
    }

    /// Outcomes (rows) per column.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.data.len() / self.rows
    }

    /// Column `c` as a slice of length `rows`.
    pub fn col(&self, c: usize) -> &[f64] {
        let r = self.rows;
        &self.data[c * r..(c + 1) * r]
    }

    /// Mutable column `c`.
    pub fn col_mut(&mut self, c: usize) -> &mut [f64] {
        let r = self.rows;
        &mut self.data[c * r..(c + 1) * r]
    }

    /// Add `w` to entry `(col, row)`.
    pub fn add(&mut self, col: usize, row: usize, w: f64) {
        let idx = col * self.rows + row;
        self.data[idx] += w;
    }

    /// Sum of all entries.
    pub fn total(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Build the raw [`ColumnSums`] cache (see its shrinkage note).
    pub fn column_sums(&self) -> ColumnSums {
        ColumnSums::from_counts(self)
    }
}

impl PartialEq for Counts {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.data.as_slice() == other.data.as_slice()
    }
}

impl std::fmt::Debug for Counts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Counts")
            .field("rows", &self.rows)
            .field("cols", &self.cols())
            .field("total", &self.total())
            .finish()
    }
}

// ──────────────────────────────────────────────────────────────────────────
// BMR core (Eq 7/9 — T2.1)
// ──────────────────────────────────────────────────────────────────────────

/// Per-column BMR free energy of a reduced model against the full model
/// (Eq 7/9 of the paper, in its exact-Bayes-factor form — see the module
/// header's sign-convention note):
///
/// ```text
/// F(m) = Σ_cols [ ln B(a_c) − ln B(ã_c) + ln B(ã_m,c) − ln B(ã_m,c + a_c − ã_c) ]
///      = ln p(D|full) − ln p(D|m)
/// ```
///
/// where `prior` = ã (full prior), `post` = a (prior + accumulated counts),
/// `reduced` = ã_m (the reduced prior defining model m). Free-energy-shaped:
/// **lower is better**; the module posterior is `softmax(−F)`.
///
/// - Symmetric edge: `ã_m = ã` yields `0.0` (exactly, for exactly-
///   representable count deltas; ~1e-16 otherwise).
/// - Zero entries in `reduced` are read at [`SHRINKAGE`] (the paper's
///   convention); `prior`/`post` must be strictly positive on used columns.
pub fn bmr_log_evidence(prior: &Counts, post: &Counts, reduced: &Counts) -> f64 {
    let rows = prior.rows();
    let cols = prior.cols();
    assert_eq!(post.rows(), rows, "post rows mismatch");
    assert_eq!(post.cols(), cols, "post cols mismatch");
    assert_eq!(reduced.rows(), rows, "reduced rows mismatch");
    assert_eq!(reduced.cols(), cols, "reduced cols mismatch");

    let mut f = 0.0;
    for c in 0..cols {
        let a_col = prior.col(c);
        let p_col = post.col(c);
        let m_col = reduced.col(c);
        // t4 = ln B(shrunk(ã_m,c) + a_c − ã_c) over the reduced posterior.
        let mut sum_ln_gamma = 0.0;
        let mut sum = 0.0;
        for (&m, (&p, &a)) in m_col.iter().zip(p_col.iter().zip(a_col)) {
            let x = shrunk(m) + (p - a);
            sum_ln_gamma += ln_gamma(x);
            sum += x;
        }
        // (t2 − t1) + (t3 − t4), grouped so the symmetric edge cancels
        // exactly: fl(t2−t1) and fl(t1−t2) are exact negations.
        let t4 = sum_ln_gamma - ln_gamma(sum);
        f += (ln_beta_shrunk_col(p_col) - ln_beta_shrunk_col(a_col))
            + (ln_beta_shrunk_col(m_col) - t4);
    }
    f
}

// ──────────────────────────────────────────────────────────────────────────
// Posterior over models (Eq 9 — T2.3) + predictive oracle
// ──────────────────────────────────────────────────────────────────────────

/// In-place log-softmax over `v`.
fn log_softmax_into(v: &mut [f64]) {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return;
    }
    let sum: f64 = v.iter().map(|x| (x - max).exp()).sum();
    let lse = max + sum.ln();
    for x in v.iter_mut() {
        *x -= lse;
    }
}

/// Posterior over the model space (Eq 9): uniform model prior, normalized in
/// log space via log-sum-exp over the NEGATED free energies. `out[m]` gets
/// `Q(m) = softmax(−F(m))[m]`; zero-mass models stay at ~0 without NaN.
///
/// Full O(#cols × #models) recompute — the naive baseline. The cached
/// [`ModelSpace`] answers the same question incrementally.
pub fn posterior_over_models(
    prior: &Counts,
    models: &[Counts],
    post: &Counts,
    out: &mut [f64],
) {
    assert!(out.len() >= models.len(), "out too small");
    for (m, reduced) in models.iter().enumerate() {
        out[m] = -bmr_log_evidence(prior, post, reduced);
    }
    log_softmax_into(&mut out[..models.len()]);
    for o in out[..models.len()].iter_mut() {
        *o = o.exp();
    }
}

/// Predictive model posterior (Eq 11), FULL-RECOMPUTE oracle: the posterior
/// over models after one more unit observation at `(col, row)`, i.e. with
/// `a + Δa` where `Δa` is the one-hot increment. O(#cols × #models) — used
/// by tests to validate [`ModelSpace::predictive_posterior_into`] (the
/// sparse-Δ path) and by cold-path callers.
pub fn predictive_model_posterior(
    prior: &Counts,
    models: &[Counts],
    post: &Counts,
    col: usize,
    row: usize,
    out: &mut [f64],
) {
    let mut bumped = post.clone();
    bumped.add(col, row, 1.0);
    posterior_over_models(prior, models, &bumped, out);
}

// ──────────────────────────────────────────────────────────────────────────
// ModelSpace — the cached sparse-Δ engine (Eq 11 hot path)
// ──────────────────────────────────────────────────────────────────────────

/// Cached model space: per-model reduced posteriors `b_m = shrunk(ã_m) + n`
/// with per-column sums, and the maintained per-model free energies.
///
/// - [`new`](ModelSpace::new) is the O(#models × #cols × #rows) init.
/// - [`accumulate`](ModelSpace::accumulate) folds one observation (any `w ≥ 0`)
///   into every cache in O(#models) lgamma evaluations.
/// - [`predictive_posterior_into`](ModelSpace::predictive_posterior_into)
///   (Eq 11) and [`efe_model_gain`](ModelSpace::efe_model_gain) (Eq 10) are
///   O(#models) per call with **zero lgamma evaluations** — a unit increment
///   satisfies `ln Γ(x+1) − ln Γ(x) = ln x` exactly.
pub struct ModelSpace {
    prior: Counts,
    post: Counts,
    reduced: Vec<Counts>,
    /// Reduced posteriors `b_m` (one per model).
    b: Vec<Counts>,
    /// Per-model per-column `Σ b_m[c]`.
    b_sum: Vec<ArrayVec<f64, MAX_COLS>>,
    /// Per-column `Σ a[c]`.
    post_sum: ArrayVec<f64, MAX_COLS>,
    /// Maintained full-plan-formula free energies `F_m`.
    free_energy: ArrayVec<f64, MAX_MODELS>,
}

impl ModelSpace {
    /// Build all caches from scratch. `models` may have at most [`MAX_MODELS`]
    /// entries; all stores must share dimensions. `post` must be
    /// entrywise ≥ `prior` (the counts contract).
    pub fn new(prior: Counts, models: Vec<Counts>, post: Counts) -> Self {
        assert!(!models.is_empty(), "empty model space");
        assert!(models.len() <= MAX_MODELS, "> {MAX_MODELS} models");
        let rows = prior.rows();
        let cols = prior.cols();
        for reduced in &models {
            assert_eq!(reduced.rows(), rows, "model rows mismatch");
            assert_eq!(reduced.cols(), cols, "model cols mismatch");
        }
        for c in 0..cols {
            for r in 0..rows {
                debug_assert!(
                    post.col(c)[r] >= prior.col(c)[r],
                    "counts contract: post >= prior entrywise (col {c}, row {r})"
                );
            }
        }

        // Model-independent constant: Σ_c [ln B(a_c) − ln B(ã_c)] = ln p(D|full).
        let mut const_part = 0.0;
        let mut post_sum = ArrayVec::new();
        for c in 0..cols {
            const_part += ln_beta_shrunk_col(post.col(c)) - ln_beta_shrunk_col(prior.col(c));
            post_sum.push(post.col(c).iter().sum());
        }

        let mut b = Vec::with_capacity(models.len());
        let mut b_sum = Vec::with_capacity(models.len());
        let mut free_energy = ArrayVec::new();
        for reduced in &models {
            let mut bm = Counts::zero(rows, cols);
            let mut sums = ArrayVec::new();
            let mut f_tilde = 0.0;
            for c in 0..cols {
                let m_col = reduced.col(c);
                let p_col = post.col(c);
                let a_col = prior.col(c);
                let b_col = bm.col_mut(c);
                let mut sum_ln_gamma = 0.0;
                let mut sum = 0.0;
                for ((o, &m), (&p, &a)) in b_col.iter_mut().zip(m_col).zip(p_col.iter().zip(a_col))
                {
                    *o = shrunk(m) + (p - a);
                    sum_ln_gamma += ln_gamma(*o);
                    sum += *o;
                }
                // F̃_c = t3 − t4 = ln B(ã_m,c) − ln B(b_m,c)
                f_tilde += ln_beta_shrunk_col(m_col) - (sum_ln_gamma - ln_gamma(sum));
                sums.push(sum);
            }
            b.push(bm);
            b_sum.push(sums);
            free_energy.push(const_part + f_tilde);
        }

        Self {
            prior,
            post,
            reduced: models,
            b,
            b_sum,
            post_sum,
            free_energy,
        }
    }

    /// Number of models.
    pub fn n_models(&self) -> usize {
        self.free_energy.len()
    }

    /// Maintained free energies `F_m` (the full plan formula; matches
    /// [`bmr_log_evidence`] per model — pinned by test).
    pub fn free_energy(&self) -> &[f64] {
        self.free_energy.as_slice()
    }

    /// The accumulated posterior counts `a`.
    pub fn post(&self) -> &Counts {
        &self.post
    }

    /// The full prior `ã`.
    pub fn prior(&self) -> &Counts {
        &self.prior
    }

    /// The reduced prior of model `m` (ã_m).
    pub fn model(&self, m: usize) -> &Counts {
        &self.reduced[m]
    }

    /// `Q(m)` — posterior over models, `softmax(−F)`. O(#models).
    pub fn posterior_into(&self, out: &mut [f64]) {
        assert!(out.len() >= self.n_models(), "out too small");
        let n = self.n_models();
        for (o, &f) in out[..n].iter_mut().zip(self.free_energy.iter()) {
            *o = -f;
        }
        log_softmax_into(&mut out[..n]);
        for o in out[..n].iter_mut() {
            *o = o.exp();
        }
    }

    /// Fold one observation of weight `w ≥ 0` at `(col, row)` into the
    /// accumulated counts and every cache. O(#models) lgamma evaluations —
    /// the per-trial cold path.
    pub fn accumulate(&mut self, col: usize, row: usize, w: f64) {
        assert!(w >= 0.0, "negative observation weight");
        let a_r = self.post.col(col)[row];
        let s_a = self.post_sum[col];
        let d_ln_post =
            (ln_gamma(a_r + w) - ln_gamma(a_r)) - (ln_gamma(s_a + w) - ln_gamma(s_a));
        self.post.add(col, row, w);
        self.post_sum[col] += w;
        for m in 0..self.n_models() {
            let x = self.b[m].col(col)[row];
            let s = self.b_sum[m][col];
            let d_ln_b = (ln_gamma(x + w) - ln_gamma(x)) - (ln_gamma(s + w) - ln_gamma(s));
            self.b[m].add(col, row, w);
            self.b_sum[m][col] += w;
            // F_m = Σ_c [ln B(ã_c) + ln B(a_c) − ln B(ã_m,c) − ln B(b_m,c)]
            self.free_energy[m] += d_ln_post - d_ln_b;
        }
    }

    /// Predictive model posterior (Eq 11), sparse-Δ path: the posterior over
    /// models after one more unit observation at `(col, row)`. Only the
    /// touched column's per-model Beta terms change, and a unit increment
    /// reduces to `ln(x_row) − ln(Σx)` exactly — O(#models), zero lgamma,
    /// zero allocation. Identical to the full recompute
    /// ([`predictive_model_posterior`]) — pinned by test.
    pub fn predictive_posterior_into(&self, col: usize, row: usize, out: &mut [f64]) {
        assert!(out.len() >= self.n_models(), "out too small");
        let d_post = self.post.col(col)[row].ln() - self.post_sum[col].ln();
        let n = self.n_models();
        for (o, (&f_e, (b_m, b_s))) in out[..n]
            .iter_mut()
            .zip(self.free_energy.iter().zip(self.b.iter().zip(self.b_sum.iter())))
        {
            let x = b_m.col(col)[row];
            *o = -(f_e + d_post - (x.ln() - b_s[col].ln()));
        }
        log_softmax_into(&mut out[..n]);
        for o in out[..n].iter_mut() {
            *o = o.exp();
        }
    }

    /// Expected information gain over models for a candidate action (Eq 10):
    ///
    /// ```text
    /// G(u) = Σ_o P(o|u) · KL( Q(m|·,o,u) ‖ Q(m|·,u) )
    /// ```
    ///
    /// `action` is the caller's anticipated-outcome list: each triple says
    /// "a unit count at column `state`, row `outcome`, with probability
    /// `prob`" (the caller derives these from its generative model — the
    /// primitive is agnostic). Each triple is evaluated as a unit-increment
    /// branch through the sparse path, weighted by `prob`. Zero-allocation:
    /// all scratch is caller-supplied.
    pub fn efe_model_gain(&self, action: &[(usize, usize, f64)], scratch: &mut EfeScratch) -> f64 {
        let n = self.n_models();
        for (s, &f) in scratch.log_base[..n].iter_mut().zip(self.free_energy.iter()) {
            *s = -f;
        }
        log_softmax_into(&mut scratch.log_base[..n]);

        let mut gain = 0.0;
        for &(col, row, p) in action {
            if p.is_nan() || p <= 0.0 {
                continue;
            }
            let d_post = self.post.col(col)[row].ln() - self.post_sum[col].ln();
            for m in 0..n {
                let x = self.b[m].col(col)[row];
                let f = self.free_energy[m] + d_post - (x.ln() - self.b_sum[m][col].ln());
                scratch.branch_log[m] = -f;
            }
            log_softmax_into(&mut scratch.branch_log[..n]);
            let mut kl = 0.0;
            for ((&log_pred, &log_base), lin) in scratch.branch_log[..n]
                .iter()
                .zip(scratch.log_base[..n].iter())
                .zip(scratch.branch_lin[..n].iter_mut())
            {
                *lin = log_pred.exp();
                kl += *lin * (log_pred - log_base);
            }
            gain += p * kl;
        }
        gain
    }
}

/// Caller-supplied scratch for [`ModelSpace::efe_model_gain`] — fixed-size
/// arrays, zero allocation per call.
pub struct EfeScratch {
    /// `ln Q(m|·,u)` — base log posterior.
    pub log_base: [f64; MAX_MODELS],
    /// Branch log posterior `ln Q(m|·,o,u)` workspace.
    pub branch_log: [f64; MAX_MODELS],
    /// Branch linear posterior workspace.
    pub branch_lin: [f64; MAX_MODELS],
}

impl EfeScratch {
    pub const fn new() -> Self {
        Self {
            log_base: [0.0; MAX_MODELS],
            branch_log: [0.0; MAX_MODELS],
            branch_lin: [0.0; MAX_MODELS],
        }
    }
}

impl Default for EfeScratch {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Occam commit statistic (Eq 12 — T2.5)
// ──────────────────────────────────────────────────────────────────────────

/// Occam's-razor commit statistic (Eq 12): `ln [ p* / (1 − p*) ]` with
/// `p*` the posterior mass of the best model against the rest.
///
/// - Uniform posterior over 2 models ⇒ exactly `0.0` (p* = 1/2).
/// - `p* → 1` saturates gracefully: the denominator is floored at 1e-16,
///   so the statistic caps at ~36.8 nats instead of diverging.
/// - A degenerate (all-zero / empty) posterior returns `−∞`.
///
/// The paper's commit threshold is > 16 nats.
pub fn occam_log_bayes_factor(model_posterior: &[f64]) -> f64 {
    let total: f64 = model_posterior.iter().sum();
    if model_posterior.is_empty() || !total.is_finite() || total <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let p_star = model_posterior.iter().copied().fold(f64::NEG_INFINITY, f64::max) / total;
    let denom = (1.0 - p_star).max(1e-16);
    (p_star / denom).ln()
}

// ──────────────────────────────────────────────────────────────────────────
// Isomorphic model-space enumerator (Eq 14 — T3.2)
// ──────────────────────────────────────────────────────────────────────────

/// Factorial layout for [`enumerate_isomorphic_rules`]: the generic Eq-14
/// model-space generator. A controllable **choice** factor (the criterion's
/// annotated mirror) pairs with non-controlled **context** factors; each
/// candidate rule is "if context state X then choice Y is the correct
/// criterion state".
///
/// - `n_context_states` = product of `context_levels`; columns =
///   `n_context_states × choice_levels`.
/// - `match_row` / `mismatch_row` select which outcome row carries the
///   rule-consistent / rule-inconsistent claim (e.g. reward vs penalty).
/// - `rule_weight` (V) is the concentration placed on a claimed outcome;
///   everything else in a constrained column sits at [`SHRINKAGE`].
/// - Columns the rule does NOT constrain are filled with
///   `unconstrained_fill` — set it to your full prior's pseudocount so
///   unconstrained columns contribute zero evidence delta.
#[derive(Clone, Debug)]
pub struct FactorLayout {
    /// Levels per context factor (at least one).
    pub context_levels: ArrayVec<usize, MAX_FACTORS>,
    /// Levels of the choice factor (the criterion's annotated mirror).
    pub choice_levels: usize,
    /// Outcome rows per column (≥ 2).
    pub outcome_rows: usize,
    /// Row carrying the rule-consistent (match) claim.
    pub match_row: usize,
    /// Row carrying the rule-inconsistent (mismatch) claim.
    pub mismatch_row: usize,
    /// Concentration V on a claimed outcome (default 8.0).
    pub rule_weight: f64,
    /// Fill for unconstrained columns (default 1.0).
    pub unconstrained_fill: f64,
}

impl FactorLayout {
    /// Default weights (V = 8.0, fill = 1.0).
    pub fn new(
        context_levels: &[usize],
        choice_levels: usize,
        outcome_rows: usize,
        match_row: usize,
        mismatch_row: usize,
    ) -> Self {
        assert!(
            !context_levels.is_empty() && context_levels.len() <= MAX_FACTORS,
            "1..={MAX_FACTORS} context factors"
        );
        assert!(context_levels.iter().all(|&l| l >= 1), "context levels >= 1");
        assert!(choice_levels >= 1, "choice_levels >= 1");
        assert!((2..=MAX_OUT).contains(&outcome_rows), "outcome_rows in 2..={MAX_OUT}");
        assert!(match_row < outcome_rows && mismatch_row < outcome_rows, "rows in range");
        assert_ne!(match_row, mismatch_row, "match and mismatch rows must differ");
        let context_levels = context_levels.iter().copied().collect();
        Self {
            context_levels,
            choice_levels,
            outcome_rows,
            match_row,
            mismatch_row,
            rule_weight: 8.0,
            unconstrained_fill: 1.0,
        }
    }

    /// Product of context levels — the number of context states.
    pub fn n_context_states(&self) -> usize {
        self.context_levels.iter().product()
    }

    /// `n_context_states × choice_levels` — columns per rule tensor.
    pub fn n_columns(&self) -> usize {
        self.n_context_states() * self.choice_levels
    }
}

/// Enumerate the isomorphic rule space (Eq 14): one reduced-prior
/// [`Counts`] per candidate rule "if context X then choice Y is correct".
///
/// **Rule index = `x * choice_levels + y`** (x-major, y-minor — a documented
/// ordering consumers rely on for marginals). Housekeeping: every column of
/// every rule keeps ≥ [`SHRINKAGE`] (or the fill) on every row, so no
/// reduced column is ever all-zero.
///
/// The paper's three-ball layout (3 context factors of 3 levels, 3 choices)
/// yields 3 × 3³ = 81 rules. Note the paper reports 79 *unique* hypotheses
/// under its own table encoding; under this tensor encoding all 81 are
/// distinct (pinned by test) — the plan's qualitative-reproduction rule
/// covers the difference.
pub fn enumerate_isomorphic_rules(layout: &FactorLayout) -> Vec<Counts> {
    let n_ctx = layout.n_context_states();
    let n_col = layout.n_columns();
    assert!(n_col <= MAX_COLS, "{n_col} columns > MAX_COLS={MAX_COLS}");
    let rows = layout.outcome_rows;
    let choice = layout.choice_levels;

    let mut rules = Vec::with_capacity(n_ctx * choice);
    for x in 0..n_ctx {
        for y in 0..choice {
            let mut rule = Counts::zero(rows, n_col);
            for ctx in 0..n_ctx {
                for j in 0..choice {
                    let col = rule.col_mut(ctx * choice + j);
                    if ctx == x {
                        for v in col.iter_mut() {
                            *v = SHRINKAGE;
                        }
                        let (m, mm) = if j == y {
                            (layout.rule_weight, SHRINKAGE)
                        } else {
                            (SHRINKAGE, layout.rule_weight)
                        };
                        col[layout.match_row] = m;
                        col[layout.mismatch_row] = mm;
                    } else {
                        for v in col.iter_mut() {
                            *v = layout.unconstrained_fill;
                        }
                    }
                }
            }
            rules.push(rule);
        }
    }
    rules
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact marginal-likelihood oracle for ONE column: sequential
    /// posterior-predictive chaining, integer counts, NO lgamma anywhere.
    /// ln p(n|α) = Σ_t ln[(α + n_<t)_{y_t} / Σ(α + n_<t)]
    fn chained_ln_evidence_col(alpha: &[f64], units: &[usize]) -> f64 {
        let mut a = alpha.to_vec();
        let mut ln_p = 0.0;
        for (i, &k) in units.iter().enumerate() {
            for _ in 0..k {
                let s: f64 = a.iter().sum();
                ln_p += (a[i] / s).ln();
                a[i] += 1.0;
            }
        }
        ln_p
    }

    /// Build a small random store with integer entries drawn from `choices`.
    fn random_store(rng: &mut fastrand::Rng, rows: usize, cols: usize, choices: &[f64]) -> Counts {
        let mut c = Counts::zero(rows, cols);
        for col in 0..cols {
            let col_slice = c.col_mut(col);
            for v in col_slice.iter_mut() {
                *v = choices[rng.usize(..choices.len())];
            }
        }
        c
    }

    // ── T1.2: ln_beta + ColumnSums ──────────────────────────────────────────

    /// ln B against exact small-integer Beta values (factorial identities —
    /// the reference uses no floating lgamma at all).
    #[test]
    fn ln_beta_exact_small_integers() {
        let cases: &[(&[f64], f64)] = &[
            (&[1.0, 1.0], 0.0),                       // B(1,1)=1
            (&[2.0, 2.0], (1.0f64 / 6.0).ln()),       // B(2,2)=1/6
            (&[2.0, 3.0], (1.0f64 / 12.0).ln()),      // B(2,3)=1/12
            (&[1.0, 1.0, 1.0], -(2.0_f64).ln()),       // B=1/2! =1/2
            (&[2.0, 2.0, 2.0], -(120.0_f64).ln()),     // B=(1!³)/5! =1/120
            (&[3.0, 2.0, 1.0], (2.0f64 / 120.0).ln()), // (2!·1!·0!)/5!
        ];
        for (x, want) in cases {
            let got = ln_beta(x);
            assert!(
                (got - want).abs() < 1e-12,
                "ln_beta({x:?}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn ln_beta_matches_direct_lgamma_sum() {
        let mut rng = fastrand::Rng::with_seed(5971);
        for _ in 0..50 {
            let rows = 2 + rng.usize(..3);
            let cols = 1 + rng.usize(..3);
            let c = random_store(&mut rng, rows, cols, &[1.0, 2.0, 3.0, 5.0]);
            for col in 0..cols {
                let x = c.col(col);
                let direct: f64 = x.iter().map(|&v| ln_gamma(v)).sum::<f64>()
                    - ln_gamma(x.iter().sum::<f64>());
                assert!((ln_beta(x) - direct).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn column_sums_match_direct_and_incremental() {
        let mut rng = fastrand::Rng::with_seed(5972);
        let rows = 3;
        let cols = 4;
        let mut c = random_store(&mut rng, rows, cols, &[1.0, 2.0, 4.0]);
        let mut sums = ColumnSums::from_counts(&c);
        for col in 0..cols {
            let x = c.col(col);
            assert!((sums.sum(col) - x.iter().sum::<f64>()).abs() < 1e-12);
            assert!((sums.ln_beta(col) - ln_beta(x)).abs() < 1e-12);
        }
        // Incremental fold must match a full rebuild.
        let (col, row) = (rng.usize(..cols), rng.usize(..rows));
        sums.accumulate(&c, col, row, 2.5);
        c.add(col, row, 2.5);
        let rebuilt = ColumnSums::from_counts(&c);
        for col in 0..cols {
            assert!((sums.ln_beta(col) - rebuilt.ln_beta(col)).abs() < 1e-12);
        }
    }

    #[test]
    fn ln_beta_delta_fractional_matches_direct() {
        let mut rng = fastrand::Rng::with_seed(5973);
        let rows = 3;
        let cols = 2;
        let c = random_store(&mut rng, rows, cols, &[1.0, 2.5, 4.0]);
        let sums = ColumnSums::from_counts(&c);
        for col in 0..cols {
            let mut delta = vec![0.0f64; rows];
            delta[rng.usize(..rows)] = rng.f64() * 2.0; // one sparse fractional entry
            delta[rng.usize(..rows)] += 0.5;
            let mut direct_col = c.col(col).to_vec();
            for (d, v) in delta.iter().zip(direct_col.iter_mut()) {
                *v += d;
            }
            let got = ln_beta_delta(&c, &sums, col, &delta);
            let want = ln_beta(&direct_col);
            assert!((got - want).abs() < 1e-12, "col {col}: {got} vs {want}");
        }
        // Zero delta ⇒ unchanged.
        let zeros = vec![0.0f64; rows];
        for col in 0..cols {
            assert!((ln_beta_delta(&c, &sums, col, &zeros) - sums.ln_beta(col)).abs() < 1e-14);
        }
    }

    // ── T2.1: bmr_log_evidence edges ────────────────────────────────────────

    #[test]
    fn bmr_symmetric_edge_is_exactly_zero() {
        let mut rng = fastrand::Rng::with_seed(5974);
        for _ in 0..25 {
            let rows = 2 + rng.usize(..3);
            let cols = 1 + rng.usize(..3);
            let prior = random_store(&mut rng, rows, cols, &[1.0, 2.0]);
            let mut post = prior.clone();
            for _ in 0..(1 + rng.usize(..6)) {
                post.add(rng.usize(..cols), rng.usize(..rows), 1.0);
            }
            let f = bmr_log_evidence(&prior, &post, &prior.clone());
            assert!(
                f == 0.0,
                "symmetric edge (ã_m = ã) must be exactly 0, got {f}"
            );
        }
    }

    #[test]
    fn bmr_all_zero_reduced_column_matches_shrinkage_fill() {
        let mut rng = fastrand::Rng::with_seed(5975);
        let rows = 3;
        let cols = 2;
        let prior = random_store(&mut rng, rows, cols, &[1.0, 2.0]);
        let mut post = prior.clone();
        post.add(0, 1, 3.0);
        let zeros = Counts::zero(rows, cols);
        let filled = Counts::filled(rows, cols, SHRINKAGE);
        let f_zero = bmr_log_evidence(&prior, &post, &zeros);
        let f_fill = bmr_log_evidence(&prior, &post, &filled);
        assert!(
            f_zero == f_fill,
            "all-zero reduced column must equal all-(1/32) column: {f_zero} vs {f_fill}"
        );
    }

    /// The SIGN pin: with data accumulated on row 0, the row-0-heavy reduced
    /// model must have MORE evidence (LOWER F). Guards the softmax(−F)
    /// convention end to end.
    #[test]
    fn evidence_favors_data_consistent_model() {
        let rows = 2;
        let cols = 1;
        let prior = Counts::filled(rows, cols, 1.0);
        let mut post = prior.clone();
        for _ in 0..3 {
            post.add(0, 0, 1.0); // three row-0 observations
        }
        let mut consistent = Counts::zero(rows, cols); // row-0-heavy
        consistent.col_mut(0)[0] = 4.0;
        consistent.col_mut(0)[1] = 0.5;
        let mut inconsistent = consistent.clone(); // row-1-heavy
        inconsistent.col_mut(0).swap(0, 1);

        let f_c = bmr_log_evidence(&prior, &post, &consistent);
        let f_i = bmr_log_evidence(&prior, &post, &inconsistent);
        assert!(f_c < f_i, "consistent model must have lower F ({f_c} vs {f_i})");

        let mut out = [0.0f64; 2];
        posterior_over_models(&prior, &[consistent, inconsistent], &post, &mut out);
        assert!(out[0] > out[1], "posterior must favor the consistent model");
        assert!((out[0] + out[1] - 1.0).abs() < 1e-12, "normalizes");
    }

    // ── T2.2: brute-force cross-check (PERMANENT G1 anchor) ────────────────

    /// BMR log-evidence must match the exact marginal likelihood obtained by
    /// sequential predictive chaining (no lgamma). Small tensors, ≤5 models,
    /// zeros in reduced priors exercising the shrinkage convention.
    #[test]
    fn bmr_matches_brute_force_evidence() {
        let mut rng = fastrand::Rng::with_seed(5976);
        for _ in 0..50 {
            let rows = 2 + rng.usize(..3); // ≤ 4 outcomes (≤ 3×4 tensors)
            let cols = 1 + rng.usize(..3);
            let prior = random_store(&mut rng, rows, cols, &[1.0, 2.0]);
            // Observed counts: integers 0..=5.
            let mut units = vec![vec![0usize; rows]; cols];
            let mut post = prior.clone();
            for (col, u) in units.iter_mut().enumerate() {
                for (r, v) in u.iter_mut().enumerate() {
                    *v = rng.usize(..6);
                    post.add(col, r, *v as f64);
                }
            }
            // ln p(D|full) via chaining.
            let mut ln_full = 0.0;
            for (c, u) in units.iter().enumerate() {
                ln_full += chained_ln_evidence_col(prior.col(c), u);
            }
            for _m in 0..(1 + rng.usize(..5)) {
                // ≤ 5 models
                let reduced = random_store(&mut rng, rows, cols, &[0.0, 1.0, 4.0]);
                let f = bmr_log_evidence(&prior, &post, &reduced);
                // ln p(D|m) via chaining over the SHRUNK reduced prior.
                let mut ln_m = 0.0;
                for (c, u) in units.iter().enumerate() {
                    let alpha: Vec<f64> = reduced.col(c).iter().map(|&v| shrunk(v)).collect();
                    ln_m += chained_ln_evidence_col(&alpha, u);
                }
                // F = ln p(D|full) − ln p(D|m)  ⇔  ln p(D|m) = ln_full − F.
                let got = ln_full - f;
                assert!(
                    (got - ln_m).abs() < 1e-9,
                    "BMR evidence {got} vs brute force {ln_m} (rows {rows}, cols {cols})"
                );
            }
        }
    }

    // ── T2.3: posterior over models ─────────────────────────────────────────

    #[test]
    fn posterior_matches_softmax_of_brute_force_evidence() {
        let mut rng = fastrand::Rng::with_seed(5977);
        let rows = 2;
        let cols = 2;
        let prior = random_store(&mut rng, rows, cols, &[1.0]);
        let mut units = vec![vec![0usize; rows]; cols];
        let mut post = prior.clone();
        for (col, u) in units.iter_mut().enumerate() {
            for (r, v) in u.iter_mut().enumerate() {
                *v = rng.usize(..4);
                post.add(col, r, *v as f64);
            }
        }
        let models = vec![
            random_store(&mut rng, rows, cols, &[0.0, 4.0]),
            random_store(&mut rng, rows, cols, &[0.0, 4.0]),
            random_store(&mut rng, rows, cols, &[1.0, 2.0]),
        ];
        let mut out = [0.0f64; 3];
        posterior_over_models(&prior, &models, &post, &mut out);
        // Oracle: softmax over chained evidences.
        let mut ln_ev = [0.0f64; 3];
        for (m, red) in models.iter().enumerate() {
            for (c, u) in units.iter().enumerate() {
                let alpha: Vec<f64> = red.col(c).iter().map(|&v| shrunk(v)).collect();
                ln_ev[m] += chained_ln_evidence_col(&alpha, u);
            }
        }
        let max = ln_ev.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = ln_ev.iter().map(|v| (v - max).exp()).sum();
        for m in 0..3 {
            let want = (ln_ev[m] - max - z.ln()).exp();
            assert!((out[m] - want).abs() < 1e-9, "{m}: {} vs {want}", out[m]);
        }
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    // ── ModelSpace engine ───────────────────────────────────────────────────

    fn random_engine_case(rng: &mut fastrand::Rng) -> (Counts, Vec<Counts>, Counts) {
        let rows = 2 + rng.usize(..2);
        let cols = 2 + rng.usize(..2);
        let prior = random_store(rng, rows, cols, &[1.0, 2.0]);
        let mut post = prior.clone();
        for _ in 0..rng.usize(..8) {
            post.add(rng.usize(..cols), rng.usize(..rows), 1.0);
        }
        let n_models = 2 + rng.usize(..3);
        let models: Vec<Counts> = (0..n_models)
            .map(|_| random_store(rng, rows, cols, &[0.0, 1.0, 8.0]))
            .collect();
        (prior, models, post)
    }

    #[test]
    fn modelspace_free_energy_matches_free_fn() {
        let mut rng = fastrand::Rng::with_seed(5978);
        for _ in 0..25 {
            let (prior, models, post) = random_engine_case(&mut rng);
            let engine = ModelSpace::new(prior.clone(), models.clone(), post.clone());
            for (m, red) in models.iter().enumerate() {
                let want = bmr_log_evidence(&prior, &post, red);
                let got = engine.free_energy()[m];
                assert!(
                    (got - want).abs() < 1e-9,
                    "model {m}: incremental {got} vs from-scratch {want}"
                );
            }
        }
    }

    #[test]
    fn modelspace_accumulate_keeps_free_energy_parity() {
        let mut rng = fastrand::Rng::with_seed(5979);
        for _ in 0..15 {
            let (prior, models, post) = random_engine_case(&mut rng);
            let mut engine = ModelSpace::new(prior.clone(), models.clone(), post);
            for _ in 0..(1 + rng.usize(..10)) {
                let col = rng.usize(..prior.cols());
                let row = rng.usize(..prior.rows());
                let w = if rng.bool() { 1.0 } else { rng.f64() };
                engine.accumulate(col, row, w);
            }
            let post = engine.post().clone();
            for (m, red) in models.iter().enumerate() {
                let want = bmr_log_evidence(&prior, &post, red);
                let got = engine.free_energy()[m];
                assert!(
                    (got - want).abs() < 1e-8,
                    "model {m} after accumulates: {got} vs {want}"
                );
            }
        }
    }

    /// T2.4 pin: the sparse-Δ predictive posterior is identical to the full
    /// recompute with `a + Δa`.
    #[test]
    fn predictive_sparse_matches_full_recompute() {
        let mut rng = fastrand::Rng::with_seed(5980);
        for _ in 0..25 {
            let (prior, models, post) = random_engine_case(&mut rng);
            let mut engine = ModelSpace::new(prior.clone(), models.clone(), post.clone());
            for _ in 0..rng.usize(..6) {
                engine.accumulate(rng.usize(..prior.cols()), rng.usize(..prior.rows()), 1.0);
            }
            let col = rng.usize(..prior.cols());
            let row = rng.usize(..prior.rows());
            let mut sparse = vec![0.0f64; models.len()];
            let mut full = vec![0.0f64; models.len()];
            engine.predictive_posterior_into(col, row, &mut sparse);
            predictive_model_posterior(
                engine.prior(),
                &models,
                engine.post(),
                col,
                row,
                &mut full,
            );
            for m in 0..models.len() {
                assert!(
                    (sparse[m] - full[m]).abs() < 1e-9,
                    "model {m}: sparse {} vs full {}",
                    sparse[m],
                    full[m]
                );
            }
        }
    }

    /// A single model equal to the full prior never shifts — the symmetric
    /// edge through the predictive path.
    #[test]
    fn predictive_symmetric_model_is_unchanged() {
        let rows = 2;
        let cols = 2;
        let prior = Counts::filled(rows, cols, 1.0);
        let mut post = prior.clone();
        post.add(1, 0, 2.0);
        let engine = ModelSpace::new(prior.clone(), vec![prior.clone()], post);
        let mut out = [0.0f64; 1];
        engine.predictive_posterior_into(0, 1, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-12);
    }

    // ── T3.1: efe_model_gain ────────────────────────────────────────────────

    /// Cross-check `efe_model_gain` against its definition assembled from the
    /// full-recompute primitives: Σ_o p · KL(Q(m|o,u) ‖ Q(m|u)).
    #[test]
    fn efe_matches_full_definition() {
        let mut rng = fastrand::Rng::with_seed(5981);
        for _ in 0..20 {
            let (prior, models, post) = random_engine_case(&mut rng);
            let mut engine = ModelSpace::new(prior.clone(), models.clone(), post);
            for _ in 0..rng.usize(..5) {
                engine.accumulate(rng.usize(..prior.cols()), rng.usize(..prior.rows()), 1.0);
            }
            let n = models.len();
            let mut base = vec![0.0f64; n];
            engine.posterior_into(&mut base);
            let action: Vec<(usize, usize, f64)> = (0..2)
                .map(|_| {
                    (
                        rng.usize(..prior.cols()),
                        rng.usize(..prior.rows()),
                        rng.f64(),
                    )
                })
                .collect();
            // Oracle.
            let mut want = 0.0;
            for &(col, row, p) in &action {
                let mut pred = vec![0.0f64; n];
                predictive_model_posterior(engine.prior(), &models, engine.post(), col, row, &mut pred);
                let mut kl = 0.0;
                for m in 0..n {
                    if pred[m] > 0.0 {
                        kl += pred[m] * ((pred[m] / base[m]).ln());
                    }
                }
                want += p * kl;
            }
            let mut scratch = EfeScratch::new();
            let got = engine.efe_model_gain(&action, &mut scratch);
            assert!(
                (got - want).abs() < 1e-9,
                "efe {got} vs definition {want}"
            );
            assert!(got >= -1e-12, "KL-weighted gain must be non-negative, got {got}");
        }
    }

    #[test]
    fn efe_zero_for_empty_or_degenerate_action() {
        let rows = 2;
        let cols = 2;
        let prior = Counts::filled(rows, cols, 1.0);
        let mut post = prior.clone();
        post.add(0, 0, 1.0);
        let models: Vec<Counts> = (0..3)
            .map(|m| {
                let mut r = Counts::zero(rows, cols);
                r.col_mut(0)[m % 2] = 4.0;
                r
            })
            .collect();
        let engine = ModelSpace::new(prior, models, post);
        let mut scratch = EfeScratch::new();
        assert_eq!(engine.efe_model_gain(&[], &mut scratch), 0.0);
        let zero_prob = [(0usize, 0usize, 0.0f64)];
        assert_eq!(engine.efe_model_gain(&zero_prob, &mut scratch), 0.0);
        // Single-model space: the posterior cannot shift ⇒ KL = 0.
        let single = ModelSpace::new(
            Counts::filled(rows, cols, 1.0),
            vec![Counts::filled(rows, cols, 1.0)],
            Counts::filled(rows, cols, 2.0),
        );
        let action = [(0usize, 1usize, 1.0f64)];
        assert!(single.efe_model_gain(&action, &mut scratch).abs() < 1e-12);
    }

    // ── T2.5: occam ──────────────────────────────────────────────────────────

    #[test]
    fn occam_uniform_pair_is_exactly_zero() {
        let got = occam_log_bayes_factor(&[0.5, 0.5]);
        assert!(got == 0.0, "uniform 2-model posterior ⇒ 0, got {got}");
    }

    #[test]
    fn occam_monotone_in_best_mass() {
        let cases: [(&[f64], f64); 3] = [
            (&[0.5, 0.5], 0.0),
            (&[0.9, 0.1], 9.0f64.ln()),
            (&[0.99, 0.01], 99.0f64.ln()),
        ];
        for (post, want) in cases {
            let got = occam_log_bayes_factor(post);
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        assert!(occam_log_bayes_factor(&[0.9, 0.1]) < occam_log_bayes_factor(&[0.99, 0.01]));
    }

    #[test]
    fn occam_saturates_gracefully_at_one() {
        let got = occam_log_bayes_factor(&[1.0, 0.0]);
        assert!(got.is_finite(), "p*→1 must stay finite, got {got}");
        assert!(got > 30.0 && got < 40.0, "saturated near ln(1e16)≈36.8, got {got}");
        assert_eq!(occam_log_bayes_factor(&[]), f64::NEG_INFINITY);
        assert_eq!(occam_log_bayes_factor(&[0.0, 0.0]), f64::NEG_INFINITY);
    }

    // ── T3.2: Eq 14 enumerator ──────────────────────────────────────────────

    /// The paper's three-ball layout: 3 context factors × 3 levels, 3
    /// choices, 3 feedback outcomes (none/reward/penalty).
    fn three_ball_layout() -> FactorLayout {
        FactorLayout::new(&[3, 3, 3], 3, 3, 1, 2)
    }

    /// Three-ball: exactly 3 × 3³ = 81 rules; all distinct under this tensor
    /// encoding (the paper's 79-unique count arises from its own table
    /// encoding); every column housekept (≥ SHRINKAGE or fill on every row).
    #[test]
    fn enumerate_three_ball_yields_81_distinct_housekept_rules() {
        let layout = three_ball_layout();
        let rules = enumerate_isomorphic_rules(&layout);
        assert_eq!(rules.len(), 81, "3 × 3³ = 81 candidate rules");
        let mut unique: Vec<&Counts> = Vec::new();
        for rule in &rules {
            if !unique.contains(&rule) {
                unique.push(rule);
            }
        }
        assert_eq!(unique.len(), 81, "all 81 distinct under tensor equality");
        for rule in &rules {
            for c in 0..rule.cols() {
                for &v in rule.col(c) {
                    assert!(v > 0.0, "housekeeping: every entry ≥ SHRINKAGE");
                }
            }
        }
    }

    /// Spot-check rule (x=5, y=2): its constrained context's columns carry
    /// the claim (match-heavy at choice 2, mismatch-heavy elsewhere);
    /// unconstrained columns stay at the fill.
    #[test]
    fn enumerate_rule_shape_spot_check() {
        let layout = three_ball_layout();
        let rules = enumerate_isomorphic_rules(&layout);
        let (x, y) = (5usize, 2usize);
        let rule = &rules[x * 3 + y];
        assert_eq!(rule.cols(), 81);
        // Constrained context 5: choice 2 → match-heavy (row 1).
        let hit = rule.col(5 * 3 + y);
        assert_eq!(hit[layout.match_row], layout.rule_weight);
        assert_eq!(hit[layout.mismatch_row], SHRINKAGE);
        assert_eq!(hit[0], SHRINKAGE, "none-row sits at shrinkage");
        // Choice 0 in context 5 → mismatch-heavy (row 2).
        let miss = rule.col(5 * 3);
        assert_eq!(miss[layout.mismatch_row], layout.rule_weight);
        assert_eq!(miss[layout.match_row], SHRINKAGE);
        // Unconstrained context 4 → fill.
        let other = rule.col(4 * 3 + 1);
        for &v in other {
            assert_eq!(v, layout.unconstrained_fill);
        }
    }

    // ── bounds ──────────────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "out of 1..=64")]
    fn counts_rows_bound_panics() {
        let _ = Counts::zero(MAX_OUT + 1, 1);
    }

    #[test]
    #[should_panic(expected = "out of 1..=128")]
    fn counts_cols_bound_panics() {
        let _ = Counts::zero(2, MAX_COLS + 1);
    }

    #[test]
    #[should_panic(expected = "models")]
    fn modelspace_too_many_models_panics() {
        let prior = Counts::filled(2, 1, 1.0);
        let models: Vec<Counts> = (0..=MAX_MODELS).map(|_| prior.clone()).collect();
        let _ = ModelSpace::new(prior.clone(), models, prior);
    }
}
