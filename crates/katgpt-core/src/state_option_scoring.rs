//! Per-option centroid-cosine scoring over a fixed option table —
//! katgpt-rs Plan 607 T1, the modelless game-decision lane's scoring
//! primitive (opt-in `state_option_scoring`).
//!
//! # The shape being upstreamed
//!
//! riir-reflex `engine.rs` `route_terms` (Issue 004 T7): each decision
//! option carries a corpus centroid; the state is embedded; per-option
//! `sigmoid(dot(state, centroid) · ROUTE_SCALE)` ranks the options and the
//! argmax decides. Reflex measured WHY the byte-level drafter cannot do
//! this job (a short option encoding never moves a long shared context's
//! compressed length → every option ties → constant pick), and this module
//! carries the fix upstream as a generic primitive: **centroid-cosine
//! scoring only** — the compression drafter stays out of the per-decision
//! loop (Plan 607 R3, decided at the plan, never a number relaxed later).
//!
//! # Generic by law (R4)
//!
//! The surface takes `(state vector, option matrix)` — nothing
//! arena-specific in katgpt-core (a public upstream surface is not shaped
//! by one consumer). The first consumer embeds its domain's sentences into
//! vectors elsewhere and hands the vectors here; embedding is deliberately
//! NOT this module's job.
//!
//! # Contracts
//!
//! - **Sigmoid, never softmax** — per-option scores are independent
//!   [`exact_sigmoid`] projections (the Issue-870 exact form), never a
//!   normalized group distribution.
//! - **Pinned tie-break** — argmax ties go to the LOWEST index
//!   ([`cmp_for_max`] + reverse-index comparator, the same shape
//!   `pick_domain` uses). This is the exact tie-break the Plan 607 oracle
//!   fixture pins (lowest-index on equal p_clean); a scorer that broke ties
//!   differently would disagree with the oracle on honest equivalences.
//! - **Zero-alloc** — table build, scoring and the argmax are all
//!   stack-local folds. The G4 gate is the separate
//!   `state_option_scoring_alloc_check` binary (the `*_alloc_check`
//!   convention: a counting allocator would pick up sibling tests in a
//!   shared binary).
//! - **Determinism** — the unit-normalized table IS the scoring state:
//!   same input rows → bit-identical table (sequential folds, no SIMD
//!   reduction reordering, no RNG). The GOAT prints a BLAKE3 over the
//!   table bytes; two runs / two boxes must agree.
//!
//! # Unit normalization
//!
//! Both the state and every row are unit-normalized (the
//! `distance_abstain` idiom — this module CONSUMES that implementation via
//! the feature implication rather than forking a bit-parity-critical
//! numeric helper), so the dot product is a true cosine in [−1, 1] and the
//! score is `exact_sigmoid(scale · cos)` in (0, 1). A zero vector passes
//! through normalization: "no direction" reads as cosine 0 against
//! everything, never NaN.

use crate::float_order::cmp_for_max;

/// The option table: `K` unit-normalized rows built once per decision
/// corpus. This is the determinism-committed artifact — the GOAT digest is
/// taken over [`Self::rows`] bytes.
///
/// `K` is const-generic (the `pick_domain` shape): the option count is
/// part of the type, the whole hot path stays stack-local, and the
/// monomorphized folds stay branch-free. The first consumer's decision
/// sets are 9/17/34 options wide.
pub struct CentroidTable<const D: usize, const K: usize> {
    rows: [[f32; D]; K],
}

impl<const D: usize, const K: usize> CentroidTable<D, K> {
    /// Build from raw option vectors; every row is unit-normalized once at
    /// build. Bit-identical for identical inputs.
    pub fn new(options: &[[f32; D]; K]) -> Self {
        assert!(K > 0, "option table needs at least one option");
        Self {
            rows: core::array::from_fn(|i| crate::distance_abstain::unit(options[i])),
        }
    }

    /// Option count (the const `K`).
    pub const fn len(&self) -> usize {
        K
    }

    /// True only for the impossible `K = 0` instantiation (rejected at
    /// [`Self::new`]); present so `K` reads as a length, not a magic bound.
    pub const fn is_empty(&self) -> bool {
        K == 0
    }

    /// Unit-normalized row `i`.
    pub fn row(&self, i: usize) -> &[f32; D] {
        &self.rows[i]
    }

    /// All rows (the determinism digest is taken over these bytes).
    pub fn rows(&self) -> &[[f32; D]; K] {
        &self.rows
    }

    /// Score every option: `out[i] = exact_sigmoid(scale · cos(state,
    /// row_i))`. Returns the argmax index (`cmp_for_max`, ties → lowest
    /// index). Zero-alloc; deterministic fold order throughout.
    ///
    /// `scale` shapes the score SPREAD for callers that blend scores with
    /// other terms; for every `scale > 0` the returned argmax equals
    /// [`Self::pick`] (the exact sigmoid is strictly monotone) — the scale
    /// never moves the decision on its own.
    pub fn score_into(&self, state: &[f32; D], scale: f32, out: &mut [f32; K]) -> usize {
        let q = crate::distance_abstain::unit(*state);
        for (o, row) in out.iter_mut().zip(self.rows.iter()) {
            let mut dot = 0.0f32;
            for (a, b) in q.iter().zip(row.iter()) {
                dot += a * b;
            }
            *o = crate::exact_sigmoid(scale * dot);
        }
        argmax_lowest_index_tie(out)
    }

    /// Argmax cosine only — no scores materialized, no sigmoid. Ties → the
    /// lowest index. Equals [`Self::score_into`]'s return for `scale > 0`.
    pub fn pick(&self, state: &[f32; D]) -> usize {
        let q = crate::distance_abstain::unit(*state);
        let mut dots = [0.0f32; K];
        for (s, row) in dots.iter_mut().zip(self.rows.iter()) {
            let mut dot = 0.0f32;
            for (a, b) in q.iter().zip(row.iter()) {
                dot += a * b;
            }
            *s = dot;
        }
        argmax_lowest_index_tie(&dots)
    }
}

/// Argmax over `scores` with the deterministic tie-break: [`cmp_for_max`]
/// (NaN-safe total order) composed with reverse-index so the LOWEST index
/// wins a tie — `pick_domain`'s comparator shape and the Plan 607 oracle's
/// pinned tie-break. (std `max_by` folds left-to-right and keeps the
/// accumulator unless the next element compares Greater, so
/// `then(i2.cmp(i1))` makes an equal later element lose to the earlier
/// one.)
fn argmax_lowest_index_tie<const K: usize>(scores: &[f32; K]) -> usize {
    let (best, _) = (0..K)
        .map(|i| (i, scores[i]))
        .max_by(|(i1, s1), (i2, s2)| cmp_for_max(*s1, *s2).then(i2.cmp(i1)))
        .unwrap_or((0, scores[0]));
    best
}

/// Plan 607 T3 — the corpus-fitted head, determinism-constrained.
///
/// Bench 876's first reading showed the untuned sentence-cosine scorer
/// discriminates but is not accurate (10.8%, tying the constant-pick
/// baseline); this module is the plan's designated lever: a **linear head
/// fitted over frozen per-option features** by closed-form ridge least
/// squares, imitating the oracle corpus (p_clean per option) — the plan's
/// "80–90% corpus-viable" candidate.
///
/// # The determinism line (Plan 607 T3 — held, not argued)
///
/// Fixed recipe, no RNG, no iterations, no gradient descent on base
/// weights: Gram `XᵀX + λI` and covariance `Xᵀy` accumulate in f64 over
/// the rows in corpus order, then ONE [`ridge_solve_direct_f64`] solve
/// closes the fit. The f64 path is scalar `mul_add` + IEEE `sqrt` — both
/// exactly rounded, so the head is reconstructible from the corpus alone
/// and bit-identical across boxes (the GOAT digests the weights; two
/// runs / two boxes must agree).
///
/// # Substrate
///
/// The solve CONSUMES [`crate::linalg::ridge_solve`]'s f64 path — KARC
/// Plan 308's fit math (the T0a substrate-first finding: the fitters are
/// precedent to consume, never re-implement; attn-match's beta_fitter is
/// the pattern, and this repo's own `linalg` is the dependency-legal
/// copy). `λ > 0` is that module's hard precondition and is what makes
/// the Gram positive-definite by construction.
///
/// # Scale contract
///
/// Ridge is scale-sensitive: callers should standardize feature columns
/// corpus-side (deterministic: corpus mean/std, fixed order) and put the
/// intercept in the design matrix (a constant-1 column) — the first
/// consumer does exactly that. The head itself stays raw: one math, no
/// embedded policy.
pub mod head {
    use crate::float_order::cmp_for_max_f64;
    use crate::linalg::ridge_solve::ridge_solve_direct_f64;

    /// A fitted linear head: `score(x) = w·x` over `D` design columns
    /// (standardized features + intercept, by caller convention). The
    /// weights ARE the determinism-committed artifact — digest them for
    /// the two-box claim. The decision path ([`Self::score`] /
    /// [`Self::pick`]) is a stack-local f64 fold: zero-alloc.
    #[derive(Debug, Clone)]
    pub struct FittedHead<const D: usize> {
        w: [f64; D],
    }

    /// Scratch owner for (repeated) fits — the `D×D` Gram and Cholesky
    /// buffers cannot be stack arrays under a plain const-generic `D`
    /// (stable Rust: `[0.0; D * D]` is rejected), so they live here,
    /// allocated ONCE and reused across every [`Self::fit_into`] call.
    /// The LOO protocol (fit per held-out state) reuses one fitter for
    /// all refits; the hot decision path never touches this.
    pub struct HeadFitter<const D: usize> {
        gram: Vec<f64>,
        l: Vec<f64>,
        cov: Vec<f64>,
        z: Vec<f64>,
        w: Vec<f64>,
    }

    impl<const D: usize> Default for HeadFitter<D> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<const D: usize> HeadFitter<D> {
        /// Allocate the scratch (the only allocation in the fit path;
        /// cold — once per corpus/protocol, not per decision).
        pub fn new() -> Self {
            Self {
                gram: vec![0.0; D * D],
                l: vec![0.0; D * D],
                cov: vec![0.0; D],
                z: vec![0.0; D],
                w: vec![0.0; D],
            }
        }

        /// Closed-form ridge least squares into a fresh [`FittedHead`]:
        /// `argmin_w ‖Xw − y‖² + λ‖w‖²`. Panics on empty input, a length
        /// mismatch, a non-finite `λ`, or `λ ≤ 0` (the substrate's PD
        /// precondition). Reuses this fitter's scratch — no allocation.
        pub fn fit_into(&mut self, rows: &[[f64; D]], target: &[f64], ridge: f64) -> FittedHead<D> {
            assert!(!rows.is_empty(), "head fit needs at least one row");
            assert_eq!(rows.len(), target.len(), "design/target length mismatch");
            assert!(
                ridge > 0.0 && ridge.is_finite(),
                "ridge λ must be finite and > 0"
            );
            self.gram.fill(0.0);
            self.cov.fill(0.0);
            for (x, &y) in rows.iter().zip(target.iter()) {
                for i in 0..D {
                    self.cov[i] = x[i].mul_add(y, self.cov[i]);
                    let g_row = i * D;
                    for j in 0..D {
                        self.gram[g_row + j] = x[i].mul_add(x[j], self.gram[g_row + j]);
                    }
                }
            }
            for i in 0..D {
                self.gram[i * D + i] += ridge;
            }
            ridge_solve_direct_f64(
                &mut self.w,
                &mut self.l,
                &mut self.z,
                &self.gram,
                &self.cov,
                D,
                1,
            );
            let mut w = [0.0f64; D];
            w.copy_from_slice(&self.w);
            FittedHead { w }
        }
    }

    impl<const D: usize> FittedHead<D> {
        /// One-shot convenience: [`HeadFitter::new`] + [`HeadFitter::fit_into`].
        /// Cold path (the fitter's scratch is allocated and dropped here);
        /// loop callers should own a [`HeadFitter`] instead.
        pub fn fit(rows: &[[f64; D]], target: &[f64], ridge: f64) -> Self {
            HeadFitter::<D>::new().fit_into(rows, target, ridge)
        }

        /// The fitted weights (digest these bytes for the determinism
        /// row — LE f64, fixed order).
        pub fn weights(&self) -> &[f64; D] {
            &self.w
        }

        /// `w·x` — sequential f64 fold, no SIMD reordering.
        pub fn score(&self, x: &[f64; D]) -> f64 {
            let mut s = 0.0f64;
            for (wi, &xi) in self.w.iter().zip(x.iter()) {
                s = wi.mul_add(xi, s);
            }
            s
        }

        /// Argmax over the FIRST `k` rows of a decision set (ties → the
        /// lowest index, [`cmp_for_max_f64`] + strict-greater fold — the
        /// pinned oracle tie-break). Zero-alloc: no score buffer, the
        /// running max is enough. The exact option set needs no padding
        /// (unlike the padded const-K table path).
        pub fn pick(&self, rows: &[[f64; D]], k: usize) -> usize {
            assert!(
                k > 0 && k <= rows.len(),
                "pick needs 1..={} rows, got {k}",
                rows.len()
            );
            let mut best = 0usize;
            let mut best_s = f64::NEG_INFINITY;
            for (i, row) in rows.iter().take(k).enumerate() {
                let s = self.score(row);
                if cmp_for_max_f64(s, best_s) == core::cmp::Ordering::Greater {
                    best = i;
                    best_s = s;
                }
            }
            best
        }
    }

    /// One λ's grouped-LOO reading (katgpt-rs Plan 609 T1.5).
    pub struct GroupLamRow {
        pub lam: f64,
        pub mse: f64,
        pub agree: usize,
    }

    /// The grouped leave-one-group-out result at the selected λ.
    pub struct GroupLooOut {
        /// The selected λ (lowest MSE; strict `<`, so the FIRST λ of the
        /// grid wins an exact tie — the state-level selection law).
        pub lam: f64,
        /// Per-decision-set LOO picks at the chosen λ (lowest-index
        /// tie-break).
        pub picks: Vec<usize>,
        /// Per-row LOO predictions at the chosen λ (aligned with `rows`).
        pub preds: Vec<f64>,
        /// Per-λ readings for printing.
        pub rows: Vec<GroupLamRow>,
    }

    /// Group-level leave-one-group-out λ selection over a corpus of
    /// decision sets (katgpt-rs Plan 609 T1.5).
    ///
    /// `state_offsets` bounds each decision set in `rows`/`targets`
    /// (`state_offsets[s]..state_offsets[s+1]`); `group_offsets` bounds
    /// GROUPS of consecutive decision sets and is the HOLD-OUT unit — a
    /// group's every option is excluded from the fit that predicts it, so
    /// correlated states (the v4 paired corpus: all 7 preview states of one
    /// board) never leak across the fold. Per λ: refit on the complement,
    /// predict each held-out decision set's options, per-set argmax
    /// (strict-greater, lowest-index ties) and squared error; MSE over ALL
    /// rows selects λ. This is the generalization of the example-side
    /// state-level recipe (groups of size 1); that path is UNCHANGED and
    /// its published head digests are the G3 pins — this fn adds the grouped
    /// unit, never re-spells the fit math ([`HeadFitter::fit_into`] is the
    /// one arithmetic).
    pub fn loo_group_select<const D: usize>(
        fitter: &mut HeadFitter<D>,
        rows: &[[f64; D]],
        targets: &[f64],
        state_offsets: &[usize],
        group_offsets: &[usize],
        argmaxes: &[usize],
        ridge_grid: &[f64],
    ) -> GroupLooOut {
        assert!(!rows.is_empty(), "grouped LOO needs rows");
        assert_eq!(rows.len(), targets.len(), "design/target length mismatch");
        assert_eq!(
            state_offsets.first(),
            Some(&0),
            "state_offsets must start at 0"
        );
        assert_eq!(
            state_offsets.last(),
            Some(&rows.len()),
            "state_offsets must end at rows.len()"
        );
        let n_states = argmaxes.len();
        assert_eq!(
            state_offsets.len(),
            n_states + 1,
            "one state bound per argmax + 1"
        );
        assert_eq!(group_offsets.first(), Some(&0), "groups must start at 0");
        assert_eq!(
            group_offsets.last(),
            Some(&n_states),
            "groups must end at the state count"
        );
        for w in group_offsets.windows(2) {
            assert!(w[0] < w[1], "groups are contiguous and non-empty");
        }
        for w in state_offsets.windows(2) {
            assert!(w[0] < w[1], "decision sets are non-empty");
        }
        assert!(!ridge_grid.is_empty(), "the λ grid must not be empty");

        let mut chosen: Option<(f64, Vec<usize>, Vec<f64>)> = None;
        let mut chosen_mse = f64::INFINITY;
        let mut lam_rows = Vec::with_capacity(ridge_grid.len());
        for &lam in ridge_grid {
            let mut sq = 0.0f64;
            let mut agree = 0usize;
            let mut picks = vec![0usize; n_states];
            let mut preds = vec![0.0f64; rows.len()];
            for group in group_offsets.windows(2) {
                let (gs, ge) = (group[0], group[1]);
                let (ra, rb) = (state_offsets[gs], state_offsets[ge]);
                let mut train: Vec<[f64; D]> = Vec::with_capacity(rows.len() - (rb - ra));
                train.extend_from_slice(&rows[..ra]);
                train.extend_from_slice(&rows[rb..]);
                let mut ty: Vec<f64> = Vec::with_capacity(targets.len() - (rb - ra));
                ty.extend_from_slice(&targets[..ra]);
                ty.extend_from_slice(&targets[rb..]);
                let head = fitter.fit_into(&train, &ty, lam);
                for s in gs..ge {
                    let (a, b) = (state_offsets[s], state_offsets[s + 1]);
                    let mut best_pred = f64::NEG_INFINITY;
                    let mut bi = 0usize;
                    for (j, row) in rows[a..b].iter().enumerate() {
                        let p = head.score(row);
                        preds[a + j] = p;
                        let e = p - targets[a + j];
                        sq += e * e;
                        if p > best_pred {
                            best_pred = p;
                            bi = j;
                        }
                    }
                    picks[s] = bi;
                    if bi == argmaxes[s] {
                        agree += 1;
                    }
                }
            }
            let mse = sq / targets.len() as f64;
            lam_rows.push(GroupLamRow { lam, mse, agree });
            if mse < chosen_mse {
                chosen_mse = mse;
                chosen = Some((lam, picks, preds));
            }
        }
        let (lam, picks, preds) = chosen.expect("ridge_grid is non-empty");
        GroupLooOut {
            lam,
            picks,
            preds,
            rows: lam_rows,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const RIDGE: f64 = 1e-2;

        /// Deterministic row factory (seeded LCG — no global RNG).
        fn lcg_row<const D: usize>(seed: u64) -> [f64; D] {
            let mut s = seed;
            let mut v = [0.0f64; D];
            for x in v.iter_mut() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *x = ((s >> 40) as f64) / (1u64 << 24) as f64 * 2.0 - 1.0;
            }
            v
        }

        #[test]
        fn fit_recovers_a_planted_linear_head() {
            // Planted w (with intercept); y = Xw. The fit must rank the
            // same option best on held queries — closed-form LS on
            // well-conditioned data recovers the direction.
            const D: usize = 4;
            let planted = [0.5, -1.0, 2.0, 0.25];
            let mut rows = Vec::new();
            let mut y = Vec::new();
            for i in 0..64 {
                let x = lcg_row(0x0670_0001 + i);
                let t: f64 = planted.iter().zip(x.iter()).map(|(w, v)| w * v).sum();
                rows.push(x);
                y.push(t);
            }
            let head = FittedHead::<D>::fit(&rows, &y, RIDGE);
            for q in 0..16 {
                let a = lcg_row(0x0670_a000 + q * 2);
                let b = lcg_row(0x0670_a000 + q * 2 + 1);
                let oracle = if planted
                    .iter()
                    .zip(a.iter())
                    .map(|(w, v)| w * v)
                    .sum::<f64>()
                    >= planted
                        .iter()
                        .zip(b.iter())
                        .map(|(w, v)| w * v)
                        .sum::<f64>()
                {
                    0
                } else {
                    1
                };
                assert_eq!(
                    head.pick(&[a, b], 2),
                    oracle,
                    "planted ranking lost at q={q}"
                );
            }
        }

        #[test]
        fn fit_is_bit_deterministic() {
            const D: usize = 6;
            let rows: Vec<[f64; D]> = (0..40).map(|i| lcg_row(0x0670_b000 + i)).collect();
            let y: Vec<f64> = rows.iter().map(|x| x.iter().sum::<f64>()).collect();
            let a = FittedHead::<D>::fit(&rows, &y, RIDGE);
            let b = FittedHead::<D>::fit(&rows, &y, RIDGE);
            let ba: Vec<u8> = a.weights().iter().flat_map(|f| f.to_le_bytes()).collect();
            let bb: Vec<u8> = b.weights().iter().flat_map(|f| f.to_le_bytes()).collect();
            assert_eq!(ba, bb, "same corpus → bit-identical head");
        }

        #[test]
        fn pick_ties_break_to_lowest_index() {
            // Two identical rows (plus intercept) → equal scores → pick 0.
            const D: usize = 2;
            let rows = [[1.0, 3.0], [1.0, 3.0], [1.0, -3.0]];
            let y = [1.0, 1.0, -1.0];
            let head = FittedHead::<D>::fit(&rows, &y, RIDGE);
            assert_eq!(head.pick(&rows, 3), 0);
            // Restricted to the third row only — picks it.
            assert_eq!(head.pick(&rows[2..], 1), 0);
        }

        #[test]
        fn ridge_shrinks_but_keeps_the_direction() {
            // A huge λ shrinks w toward 0 but the RANKING (argmax) must
            // survive — the decision, not the magnitude, is the contract.
            const D: usize = 3;
            let planted = [1.0, -2.0, 0.5];
            let mut rows = Vec::new();
            let mut y = Vec::new();
            for i in 0..64 {
                let x = lcg_row(0x0670_c000 + i);
                rows.push(x);
                y.push(
                    planted
                        .iter()
                        .zip(x.iter())
                        .map(|(w, v)| w * v)
                        .sum::<f64>(),
                );
            }
            let head = FittedHead::<D>::fit(&rows, &y, 1e6);
            let norm: f64 = head.weights().iter().map(|w| w * w).sum::<f64>().sqrt();
            assert!(norm < 1.0, "λ=1e6 must shrink the head (norm {norm})");
            let a = lcg_row(0x0670_d001);
            let b = lcg_row(0x0670_d002);
            let oracle = if planted
                .iter()
                .zip(a.iter())
                .map(|(w, v)| w * v)
                .sum::<f64>()
                >= planted
                    .iter()
                    .zip(b.iter())
                    .map(|(w, v)| w * v)
                    .sum::<f64>()
            {
                0
            } else {
                1
            };
            assert_eq!(
                head.pick(&[a, b], 2),
                oracle,
                "shrunk head keeps the ranking"
            );
        }

        #[test]
        #[should_panic(expected = "ridge λ must be finite and > 0")]
        fn zero_ridge_is_refused() {
            let rows = [[1.0, 2.0]];
            let _ = FittedHead::<2>::fit(&rows, &[1.0], 0.0);
        }

        #[test]
        #[should_panic(expected = "design/target length mismatch")]
        fn length_mismatch_is_refused() {
            let rows = [[1.0, 2.0], [3.0, 4.0]];
            let _ = FittedHead::<2>::fit(&rows, &[1.0], RIDGE);
        }

        /// A tiny grouped corpus: `groups` decision sets each with `per`
        /// options, target = planted linear head + a group-specific offset
        /// (the correlated-within-group structure the hold-out must guard).
        fn grouped_corpus(
            groups: usize,
            per: usize,
        ) -> (Vec<[f64; 3]>, Vec<f64>, Vec<usize>, Vec<usize>, Vec<usize>) {
            const PLANTED: [f64; 2] = [1.0, -2.0];
            let mut rows = Vec::new();
            let mut targets = Vec::new();
            let mut state_offsets = vec![0usize];
            let mut argmaxes = Vec::new();
            for g in 0..groups {
                for s in 0..per {
                    for j in 0..4 {
                        let x = lcg_row::<3>(0x0670_e000 + (g * 97 + s * 4 + j) as u64);
                        let base: f64 = PLANTED.iter().zip(x.iter()).map(|(w, v)| w * v).sum();
                        // group offset shifts the LEVEL (calibration), the
                        // planted direction drives the within-set ranking —
                        // exactly the paired-corpus shape.
                        let t = base + 0.25 * g as f64;
                        rows.push(x);
                        targets.push(t);
                    }
                    state_offsets.push(rows.len());
                    argmaxes.push(0); // placeholder; the test reads preds/picks
                }
            }
            let group_offsets = (0..=groups).map(|g| g * per).collect();
            (rows, targets, state_offsets, group_offsets, argmaxes)
        }

        #[test]
        fn grouped_loo_predictions_equal_explicit_complement_refits() {
            // The load-bearing hold-out property: the prediction for every
            // row of group g comes from a head fitted on the complement of
            // group g's rows — verified against explicit refits.
            let (rows, targets, state_offsets, group_offsets, argmaxes) = grouped_corpus(4, 3);
            let mut fitter = HeadFitter::<3>::new();
            let grid = [1e-2];
            let out = loo_group_select(
                &mut fitter,
                &rows,
                &targets,
                &state_offsets,
                &group_offsets,
                &argmaxes,
                &grid,
            );
            for g in 0..4 {
                let (gs, ge) = (group_offsets[g], group_offsets[g + 1]);
                let (ra, rb) = (state_offsets[gs], state_offsets[ge]);
                let mut train: Vec<[f64; 3]> = Vec::new();
                train.extend_from_slice(&rows[..ra]);
                train.extend_from_slice(&rows[rb..]);
                let mut ty: Vec<f64> = Vec::new();
                ty.extend_from_slice(&targets[..ra]);
                ty.extend_from_slice(&targets[rb..]);
                let head = FittedHead::<3>::fit(&train, &ty, grid[0]);
                for s in gs..ge {
                    let (a, b) = (state_offsets[s], state_offsets[s + 1]);
                    for (j, row) in rows[a..b].iter().enumerate() {
                        let expect = head.score(row);
                        assert_eq!(
                            out.preds[a + j], expect,
                            "group {g} state {s} option {j}: not the complement refit"
                        );
                    }
                }
            }
        }

        #[test]
        fn grouped_loo_lambda_tie_takes_the_first_of_the_grid() {
            // Two identical grid entries → identical MSE → the FIRST is
            // chosen (the state-level selection law, strict <). The grid
            // carries ONLY the duplicated λ so a different-λ winner cannot
            // mask the tie law.
            let (rows, targets, state_offsets, group_offsets, argmaxes) = grouped_corpus(3, 2);
            let mut fitter = HeadFitter::<3>::new();
            let out = loo_group_select(
                &mut fitter,
                &rows,
                &targets,
                &state_offsets,
                &group_offsets,
                &argmaxes,
                &[0.5, 0.5],
            );
            assert_eq!(out.lam, 0.5, "the first λ of a tie wins");
            assert_eq!(out.rows.len(), 2);
            assert_eq!(out.rows[0].mse, out.rows[1].mse, "the tie is exact");
        }

        #[test]
        #[should_panic(expected = "groups must end at the state count")]
        fn grouped_loo_refuses_a_group_bound_that_drops_states() {
            let (rows, targets, state_offsets, _group_offsets, argmaxes) = grouped_corpus(3, 2);
            let mut fitter = HeadFitter::<3>::new();
            let _ = loo_group_select(
                &mut fitter,
                &rows,
                &targets,
                &state_offsets,
                &[0, 2], // drops the last state
                &argmaxes,
                &[1e-2],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: f32 = 8.0; // the substrate's route scale (reflex ROUTE_SCALE)

    #[test]
    fn unit_idiom_zero_vector_passes_through() {
        let v = crate::distance_abstain::unit([0.0f32; 4]);
        assert_eq!(
            v, [0.0; 4],
            "zero vector must not NaN — cosine 0 pass-through"
        );
        let u = crate::distance_abstain::unit([3.0, 4.0, 0.0, 0.0]);
        assert_eq!(u, [0.6, 0.8, 0.0, 0.0]);
    }

    #[test]
    fn pick_is_cosine_argmax_with_lowest_index_ties() {
        // state along +x; options 0 and 2 identical (cos 1), option 1 off-axis.
        let options: [[f32; 2]; 3] = [[4.0, 0.0], [0.6, 1.0], [9.0, 0.0]];
        let table = CentroidTable::<2, 3>::new(&options);
        assert_eq!(
            table.pick(&[2.0, 0.0]),
            0,
            "tie between 0 and 2 → lowest index"
        );
        assert_eq!(
            table.pick(&[0.0, 5.0]),
            1,
            "aligned with option 1's direction"
        );
    }

    #[test]
    fn score_into_argmax_equals_pick_for_positive_scale() {
        let mut options = [[0.0f32; 8]; 7];
        for (i, o) in options.iter_mut().enumerate() {
            for (d, x) in o.iter_mut().enumerate() {
                *x = ((d * 7 + i * 3) % 13) as f32 - 6.0;
            }
        }
        let table = CentroidTable::<8, 7>::new(&options);
        let state = [1.0, -2.0, 3.0, -1.0, 0.5, 2.0, -0.5, 1.5];
        let mut scores = [0.0f32; 7];
        let best_scored = table.score_into(&state, S, &mut scores);
        assert_eq!(best_scored, table.pick(&state));
        for s in scores {
            assert!(
                s > 0.0 && s < 1.0,
                "exact sigmoid is bounded (0, 1), got {s}"
            );
        }
    }

    #[test]
    fn scores_are_monotone_in_cosine() {
        let options: [[f32; 2]; 2] = [[1.0, 0.0], [2.0, 0.0]]; // same direction
        let table = CentroidTable::<2, 2>::new(&options);
        let state = [3.0, 4.0]; // cos 0.6 to both
        let mut scores = [0.0f32; 2];
        table.score_into(&state, S, &mut scores);
        assert_eq!(scores[0], scores[1], "equal cosine → equal score");
    }

    #[test]
    fn table_build_is_bit_deterministic() {
        let mut options = [[0.0f32; 16]; 5];
        for (i, o) in options.iter_mut().enumerate() {
            for (d, x) in o.iter_mut().enumerate() {
                *x = ((i * 31 + d * 7) % 11) as f32 - 5.0;
            }
        }
        let a = CentroidTable::<16, 5>::new(&options);
        let b = CentroidTable::<16, 5>::new(&options);
        for (ra, rb) in a.rows().iter().zip(b.rows().iter()) {
            let ba: Vec<u8> = ra.iter().flat_map(|f| f.to_le_bytes()).collect();
            let bb: Vec<u8> = rb.iter().flat_map(|f| f.to_le_bytes()).collect();
            assert_eq!(ba, bb, "same input rows → bit-identical table");
        }
    }

    #[test]
    fn single_option_picks_itself() {
        let options: [[f32; 4]; 1] = [[1.0, 2.0, 3.0, 4.0]];
        let table = CentroidTable::<4, 1>::new(&options);
        assert_eq!(table.pick(&[1.0, 1.0, 1.0, 1.0]), 0);
        let mut scores = [0.0f32; 1];
        table.score_into(&[1.0, 2.0, 3.0, 4.0], S, &mut scores);
        assert!(scores[0] > 0.999, "state == option → cos 1 → sigmoid ≈ 1");
    }

    #[test]
    fn planted_discrimination_refuses_constant_pick() {
        // Eight states, each aligned with a DIFFERENT basis option — the
        // reflex discrimination floor: distinct picks ≥ 2 over distinct
        // state vectors (a constant picker scores 1 distinct pick here).
        const K: usize = 8;
        let mut options = [[0.0f32; K]; K];
        for (i, row) in options.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let table = CentroidTable::<K, K>::new(&options);
        let mut picks = std::collections::HashSet::new();
        for i in 0..K {
            let mut state = [0.05f32; K];
            state[i] = 1.0;
            picks.insert(table.pick(&state));
        }
        assert_eq!(
            picks.len(),
            K,
            "rotated planted index → every pick distinct"
        );
    }

    #[test]
    fn zero_state_reads_as_no_direction_not_nan() {
        let options: [[f32; 4]; 3] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let table = CentroidTable::<4, 3>::new(&options);
        assert_eq!(
            table.pick(&[0.0, 0.0, 0.0, 0.0]),
            0,
            "all cosines 0 → tie → lowest index"
        );
    }
}
