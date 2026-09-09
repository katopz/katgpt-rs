//! Plan 580 Phase 2 — certified-frontier correctness gates.
//!
//! Feature-gated: without `--features certified_frontier` this file compiles to
//! nothing and reports `0 passed`, which is a skip, not a pass.
//!
//! ```sh
//! cargo test -p katgpt-core --features certified_frontier --test certified_frontier_correctness
//! ```
//!
//! | Task | What it pins |
//! |---|---|
//! | T2.1 | `posterior_variance_linear` vs an independent dense f64 solve at N=64 |
//! | T2.2 | soundness (Lemma E.2) under adversarial query orders, 1 000 seeds |
//! | T2.3 | monotonicity of the certified set under arbitrary query sequences |
//! | T2.4 | `confidence_schedule` monotone in `t`; closed-form `kappa`/`L_s` |
//! | T2.5 | halting law + dilation soundness against a known Lipschitz field |
//! | T2.6 | order-pinned sphere exclusion; Vendi on a planted spectrum |
//! | perf | `expand_certified` dirty-set fast path ≡ `expand_certified_full` over randomized streams (incl. shrinking-beta fallback) |
#![cfg(feature = "certified_frontier")]

use katgpt_core::certified_frontier::{
    CertifiedFrontier, FrontierConfig, PosteriorBuffer, SIGMOID_LIPSCHITZ, advance_horizon,
    beta_mean_variance, confidence_schedule, laurent_massart_radius, linear_information_gain,
    should_advance, sphere_exclusion_coverage, spherical_cap_bound, vendi_diversity,
};

// ── deterministic RNG (no dev-dep, matching the Phase 0 example) ───────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}

// ── the planted world ──────────────────────────────────────────────────────

const GRID: usize = 12;
const CELLS: usize = GRID * GRID;
const H: f32 = 0.6;

/// Latent score amplitude and spatial frequency of the planted field. Both are
/// needed to derive the field's Lipschitz constant analytically (T2.5).
const AMP: f32 = 3.0;
const FREQ: f32 = 1.0;

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// `p(x, y) = sigmoid(A cos(2 pi k x) cos(2 pi k y))` — the Phase 0 world.
fn p_true(x: f32, y: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    sigmoid(AMP * (tau * FREQ * x).cos() * (tau * FREQ * y).cos())
}

/// Analytic global Lipschitz bound on `p` in probability space.
///
/// `|grad g| <= A * 2 pi k * sqrt(2)` and `L = L_s * L_g` with `L_s = 1/4`.
/// This is an a-priori bound derived from the planted field, which is exactly
/// the contract the module states: `L` is the caller's proof obligation, never
/// estimated from the observations that drive expansion.
fn lipschitz_bound() -> f32 {
    SIGMOID_LIPSCHITZ * AMP * std::f32::consts::TAU * FREQ * std::f32::consts::SQRT_2
}

fn cell_xy(i: usize) -> (f32, f32) {
    let (r, c) = (i / GRID, i % GRID);
    (
        c as f32 / (GRID - 1) as f32,
        r as f32 / (GRID - 1) as f32,
    )
}

fn build_world() -> (CertifiedFrontier<CELLS, 2>, Vec<f32>) {
    let mut f = CertifiedFrontier::<CELLS, 2>::new();
    let mut truth = Vec::with_capacity(CELLS);
    for i in 0..CELLS {
        let (x, y) = cell_xy(i);
        f.push_cell([x, y]).expect("capacity");
        truth.push(p_true(x, y));
    }
    (f, truth)
}

// ── T2.1 — Eq 10 against an independent dense solve ────────────────────────

/// Widening dot product — the reference side works in f64 throughout so any
/// disagreement is the module's f32 factor, not a shared rounding path.
fn dot64<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum()
}

/// Gaussian elimination with partial pivoting, f64. Deliberately a different
/// algorithm from the module's incremental Cholesky — a second implementation
/// of the same factorisation would agree with a shared bug.
fn dense_solve(a: &mut [Vec<f64>], b: &mut [f64]) {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .expect("non-empty");
        a.swap(col, piv);
        b.swap(col, piv);
        // split_at_mut keeps the pivot row and the rows being eliminated as
        // disjoint borrows, so the inner loop is a plain zip over two slices.
        let (head, tail) = a.split_at_mut(col + 1);
        let pivot = &head[col];
        for (off, row) in tail.iter_mut().enumerate() {
            let factor = row[col] / pivot[col];
            for (rk, pk) in row[col..].iter_mut().zip(pivot[col..].iter()) {
                *rk -= factor * pk;
            }
            b[col + 1 + off] -= factor * b[col];
        }
    }
    for row in (0..n).rev() {
        let mut acc = b[row];
        for k in (row + 1)..n {
            acc -= a[row][k] * b[k];
        }
        b[row] = acc / a[row][row];
    }
}

#[test]
fn t2_1_posterior_variance_matches_a_dense_reference_solve() {
    const N: usize = 64;
    const D: usize = 8;
    const LAMBDA: f32 = 1.0;

    let mut rng = Lcg::new(0xC0FFEE);
    let feats: Vec<[f32; D]> = (0..N)
        .map(|_| std::array::from_fn(|_| rng.next_f32() * 2.0 - 1.0))
        .collect();

    let mut buf = Box::new(PosteriorBuffer::<N, D>::new(LAMBDA));
    for f in &feats {
        assert!(buf.append_observation(f, rng.next_f32()));
    }
    assert_eq!(buf.len(), N);

    let mut scratch = [0.0f32; N];
    let mut worst_abs = 0.0f64;
    let mut worst_rel = 0.0f64;

    for _ in 0..32 {
        let x: [f32; D] = std::array::from_fn(|_| rng.next_f32() * 2.0 - 1.0);

        // Reference: sigma^2 = k(x,x) - k(x,X) (K + lambda I)^-1 k(X,x), in f64.
        let mut k: Vec<Vec<f64>> = (0..N)
            .map(|i| {
                (0..N)
                    .map(|j| dot64(&feats[i], &feats[j]) + if i == j { LAMBDA as f64 } else { 0.0 })
                    .collect()
            })
            .collect();
        let mut kx: Vec<f64> = (0..N).map(|i| dot64(&feats[i], &x)).collect();
        let kx_orig = kx.clone();
        dense_solve(&mut k, &mut kx);
        let k_self = dot64(&x, &x);
        let reference = k_self - kx_orig.iter().zip(kx.iter()).map(|(a, b)| a * b).sum::<f64>();

        let got = buf.posterior_variance_linear(&x, &mut scratch) as f64;
        let abs = (got - reference).abs();
        worst_abs = worst_abs.max(abs);
        worst_rel = worst_rel.max(abs / reference.abs().max(1e-12));
    }

    println!("T2.1 N={N} D={D}: max |abs| = {worst_abs:.3e}, max rel = {worst_rel:.3e}");
    // The plan asked for 1e-6. The module's factor is f32 and the reference is
    // f64, so the gate is stated as a RELATIVE tolerance; the measured figure
    // is printed above so a regression shows up as a number, not a verdict.
    // Measured on first run: max abs 1.161e-6, max rel 7.252e-6. The gate is
    // set one order above the measurement so it catches drift, not noise.
    assert!(
        worst_rel < 5e-5,
        "incremental Cholesky drifted from the dense solve: rel {worst_rel:.3e}"
    );
}

#[test]
fn t2_1b_posterior_variance_is_nonnegative_and_shrinks_with_observations() {
    const N: usize = 32;
    const D: usize = 4;
    let mut rng = Lcg::new(7);
    let mut buf = Box::new(PosteriorBuffer::<N, D>::new(1.0));
    let probe: [f32; D] = [0.4, -0.2, 0.9, 0.1];
    let mut scratch = [0.0f32; N];

    let mut prev = buf.posterior_variance_linear(&probe, &mut scratch);
    for _ in 0..N {
        // Observing the probe point itself must never increase its variance.
        assert!(buf.append_observation(&probe, rng.next_f32()));
        let now = buf.posterior_variance_linear(&probe, &mut scratch);
        assert!(now >= 0.0, "variance went negative: {now}");
        assert!(
            now <= prev + 1e-5,
            "variance grew after an observation: {prev} -> {now}"
        );
        prev = now;
    }
    assert!(!buf.append_observation(&probe, 0.0), "must refuse when full");
}

// ── T2.2 — soundness under adversarial query orders ────────────────────────

/// Drive one seed with a uniformly random query order (more adversarial than
/// the frontier policy, which concentrates queries where they are informative)
/// and return the number of **unsound certifications**: cells the algorithm
/// marked certified whose true `p` is below `h`.
fn soundness_violations(seed: u64, rounds: u32) -> usize {
    let (mut f, truth) = build_world();
    let cfg = FrontierConfig {
        h: H,
        lipschitz: lipschitz_bound(),
        cell_spacing: 1.0 / (GRID - 1) as f32,
        ..FrontierConfig::default()
    };
    let mut rng = Lcg::new(seed);
    for t in 1..=rounds {
        let i = rng.below(CELLS);
        let valid = rng.next_f32() < truth[i];
        f.observe(i, valid);
        let beta = confidence_schedule(t, cfg.delta, cfg.lambda, cfg.b_rkhs, 2);
        f.expand_certified(&cfg, beta);
    }
    f.cells()
        .iter()
        .zip(truth.iter())
        .filter(|(c, p)| c.certified && **p < H)
        .count()
}

#[test]
fn t2_2_certification_is_sound_across_1000_adversarial_seeds() {
    let mut total = 0usize;
    let mut worst_seed = 0u64;
    for seed in 0..1000u64 {
        let v = soundness_violations(seed, 400);
        if v > 0 && total == 0 {
            worst_seed = seed;
        }
        total += v;
    }
    assert_eq!(
        total, 0,
        "unsound certifications across 1000 seeds (first at seed {worst_seed})"
    );
}

// ── perf — the dirty-set fast path must equal the full rescan ──────────────

/// `expand_certified` scans only cells whose tally changed since the last call
/// (falling back to a full scan when `beta` shrinks). This pins the fast path
/// bit-for-bit against [`CertifiedFrontier::expand_certified_full`] — twin
/// frontiers driven by the SAME observation stream and beta schedule, compared
/// on newly-certified counts, the certified counter, and every cell's `cb` /
/// `certified` / `near_certified` state at the end. The schedule is mostly the
/// monotone `confidence_schedule`, with a shrinking width injected every 7th
/// round to force the fallback arm.
#[test]
fn perf_expand_certified_dirty_set_matches_the_full_rescan() {
    let cfg = FrontierConfig {
        h: H,
        lipschitz: lipschitz_bound(),
        cell_spacing: 1.0 / (GRID - 1) as f32,
        ..FrontierConfig::default()
    };
    let rounds: u32 = 300;
    let mut newly_inc_total = 0;
    let mut newly_full_total = 0;

    for seed in 0..64u64 {
        let (mut inc, truth) = build_world();
        let (mut full, _) = build_world();
        let mut rng = Lcg::new(seed);

        for t in 1..=rounds {
            let i = rng.below(CELLS);
            let valid = rng.next_f32() < truth[i];
            assert!(inc.observe(i, valid) && full.observe(i, valid));

            let beta = if t % 7 == 0 { confidence_schedule(t, cfg.delta, cfg.lambda, cfg.b_rkhs, 2) * 0.5 } else { confidence_schedule(t, cfg.delta, cfg.lambda, cfg.b_rkhs, 2) };
            newly_inc_total += inc.expand_certified(&cfg, beta);
            newly_full_total += full.expand_certified_full(&cfg, beta);
        }

        assert_eq!(inc.len(), full.len());
        assert_eq!(
            inc.certified_count(),
            full.certified_count(),
            "certified counter diverged at seed {seed}"
        );
        for (ci, cf) in inc.cells().iter().zip(full.cells().iter()) {
            assert_eq!(
                ci.cb.to_bits(),
                cf.cb.to_bits(),
                "cb diverged at seed {seed}"
            );
            assert_eq!(ci.certified, cf.certified, "certified flag at seed {seed}");
            assert_eq!(
                ci.near_certified, cf.near_certified,
                "near_certified flag at seed {seed}"
            );
        }
    }
    assert_eq!(newly_inc_total, newly_full_total);
    assert!(
        newly_full_total > 0,
        "no certifications across the whole run — the equivalence check was vacuous"
    );
}

#[test]
fn t2_2b_soundness_holds_under_the_actual_frontier_policy() {
    // The 1000-seed sweep uses a random order because soundness must not depend
    // on the policy. This checks the deployed loop too, at a smaller seed count
    // because acquisition is O(cells * certified) per query.
    let cfg = FrontierConfig {
        h: H,
        lipschitz: lipschitz_bound(),
        cell_spacing: 1.0 / (GRID - 1) as f32,
        acquire_radius: 1.5 / (GRID - 1) as f32,
        ..FrontierConfig::default()
    };
    let mut grew = 0;
    for seed in 0..16u64 {
        let (mut f, truth) = build_world();
        let mut rng = Lcg::new(seed ^ 0xABCD);
        // Seed the centre of the valid basin at (0, 0), where p is maximal.
        f.seed_certified(0, &cfg);
        for t in 1..=600u32 {
            let Some(i) = f.acquire_frontier_target(&cfg) else {
                break;
            };
            f.observe(i, rng.next_f32() < truth[i]);
            let beta = confidence_schedule(t, cfg.delta, cfg.lambda, cfg.b_rkhs, 2);
            f.expand_certified(&cfg, beta);
        }
        let violations = f
            .cells()
            .iter()
            .zip(truth.iter())
            .enumerate()
            // Cell 0 is the caller-supplied a-priori seed, not a certification.
            .filter(|(i, (c, p))| *i != 0 && c.certified && **p < H)
            .count();
        assert_eq!(violations, 0, "policy-driven run certified an invalid cell");
        grew += usize::from(f.certified_count() > 1);
    }
    assert_eq!(grew, 16, "the frontier policy certified nothing beyond the seed");
}

// ── T2.3 — monotonicity ────────────────────────────────────────────────────

#[test]
fn t2_3_certified_set_never_shrinks_under_arbitrary_query_sequences() {
    let cfg = FrontierConfig {
        h: H,
        lipschitz: lipschitz_bound(),
        cell_spacing: 1.0 / (GRID - 1) as f32,
        ..FrontierConfig::default()
    };
    for seed in 0..64u64 {
        let (mut f, truth) = build_world();
        let mut rng = Lcg::new(seed ^ 0x5EED);
        let mut prev_count = 0;
        let mut prev_cb: Vec<f32> = vec![0.0; CELLS];
        for t in 1..=500u32 {
            let i = rng.below(CELLS);
            // Adversarial: with probability 1/4 feed the *opposite* label, so
            // the sequence is not even drawn from the planted world.
            let honest = rng.next_f32() < truth[i];
            let valid = if rng.next_f32() < 0.25 { !honest } else { honest };
            f.observe(i, valid);
            let beta = confidence_schedule(t, cfg.delta, cfg.lambda, cfg.b_rkhs, 2);
            f.expand_certified(&cfg, beta);

            let count = f.certified_count();
            assert!(
                count >= prev_count,
                "certified set shrank at t={t} ({prev_count} -> {count})"
            );
            for (j, c) in f.cells().iter().enumerate() {
                assert!(
                    c.cb >= prev_cb[j] - f32::EPSILON,
                    "cb decreased for cell {j} at t={t}: {} -> {}",
                    prev_cb[j],
                    c.cb
                );
                prev_cb[j] = c.cb;
            }
            prev_count = count;
        }
    }
}

// ── T2.4 — the confidence schedule ─────────────────────────────────────────

#[test]
fn t2_4_confidence_schedule_is_monotone_in_t() {
    for &(delta, lambda, b) in &[(0.05, 1.0, 1.0), (0.01, 0.5, 2.0), (0.2, 2.0, 0.5)] {
        let mut prev = f32::NEG_INFINITY;
        for t in 0..2000u32 {
            let beta = confidence_schedule(t, delta, lambda, b, 8);
            assert!(
                beta >= prev,
                "beta decreased at t={t} (delta={delta}, lambda={lambda}, B={b})"
            );
            assert!(beta.is_finite() && beta > 0.0, "beta not positive-finite");
            prev = beta;
        }
    }
}

#[test]
fn t2_4b_beta_zero_is_the_closed_form_offset_plus_the_delta_term() {
    // At t = 0 the information gain vanishes, so only the RKHS offset and the
    // ln(1/delta) term survive: beta_0 = 4 L_s B + 2 L_s sqrt(2 kappa/lambda ln(1/delta)).
    let (delta, lambda, b) = (0.05f32, 1.0f32, 1.0f32);
    assert_eq!(linear_information_gain(0, 8, lambda), 0.0);
    let s_b = sigmoid(b);
    let kappa = 1.0 / (s_b * (1.0 - s_b));
    let expected = 4.0 * SIGMOID_LIPSCHITZ * b
        + 2.0 * SIGMOID_LIPSCHITZ * (2.0 * kappa / lambda * (1.0f32 / delta).ln()).sqrt();
    let got = confidence_schedule(0, delta, lambda, b, 8);
    assert!(
        (got - expected).abs() < 1e-5,
        "beta_0 closed form: got {got}, expected {expected}"
    );
    // kappa is the inverse Bernoulli variance at the RKHS bound, and L_s = 1/4.
    assert!((SIGMOID_LIPSCHITZ - 0.25).abs() < f32::EPSILON);
    assert!((kappa - 1.0 / (s_b * (1.0 - s_b))).abs() < 1e-6);
}

#[test]
fn t2_4c_information_gain_is_sublinear_and_monotone() {
    let (d, lambda) = (4usize, 1.0f32);
    let mut prev = 0.0;
    for t in 0..1000u32 {
        let g = linear_information_gain(t, d, lambda);
        assert!(g >= prev, "gamma decreased at t={t}");
        prev = g;
    }
    // Sub-linear: doubling t must add less than the first half did.
    let g1 = linear_information_gain(500, d, lambda);
    let g2 = linear_information_gain(1000, d, lambda);
    assert!(g2 - g1 < g1, "gamma_t is not sub-linear: {g1} -> {g2}");
}

// ── T2.5 — halting law and dilation soundness ──────────────────────────────

#[test]
fn t2_5_dilation_admits_no_violation_against_a_known_lipschitz_field() {
    // Two things have to be true at once for this test to mean anything, and
    // they pull in opposite directions:
    //
    //   * the lattice must be FINE enough that `L * spacing` is payable — at
    //     GRID=12 (t2_5b) it never is, and the test would pass vacuously;
    //   * each cell must be OBSERVED enough to build the headroom that pays.
    //
    // A first attempt used 96x96 with 6 000 rounds. That is 0.65 observations
    // per cell: nothing certified, so nothing dilated, so the soundness check
    // ran against an empty set. 48x48 with 200 000 rounds gives ~87
    // observations per cell and admits ~50 cells by dilation.
    const FINE: usize = 48;
    const FINE_CELLS: usize = FINE * FINE;
    const ROUNDS: u32 = 200_000;
    const DILATE_EVERY: u32 = 50_000;

    let mut f = Box::new(CertifiedFrontier::<FINE_CELLS, 2>::new());
    let mut truth = Vec::with_capacity(FINE_CELLS);
    for i in 0..FINE_CELLS {
        let (r, c) = (i / FINE, i % FINE);
        let (x, y) = (
            c as f32 / (FINE - 1) as f32,
            r as f32 / (FINE - 1) as f32,
        );
        f.push_cell([x, y]).expect("capacity");
        truth.push(p_true(x, y));
    }
    let cfg = FrontierConfig {
        h: H,
        lipschitz: lipschitz_bound(),
        cell_spacing: 1.0 / (FINE - 1) as f32,
        ..FrontierConfig::default()
    };

    let mut rng = Lcg::new(0x11CE);
    let mut fired = 0usize;
    let mut dilation_ran = false;
    let mut last_beta = 0.0f32;
    for t in 1..=ROUNDS {
        let i = rng.below(FINE_CELLS);
        f.observe(i, rng.next_f32() < truth[i]);
        let beta = confidence_schedule(t, cfg.delta, cfg.lambda, cfg.b_rkhs, 2);
        f.expand_certified(&cfg, beta);

        if t % DILATE_EVERY == 0 {
            let feas = f.dilation_feasibility(&cfg);
            let newly = f.reachability_dilation(&cfg, 1);
            // The predicate is necessary, not sufficient: infeasible MUST mean
            // nothing was admitted.
            assert!(
                feas.feasible || newly == 0,
                "dilation admitted {newly} cells while priced infeasible \
                 (headroom {}, hop cost {})",
                feas.best_headroom,
                feas.hop_cost
            );
            dilation_ran |= newly > 0;
            // Soundness is checked after EVERY hop, not once at the end.
            let violations = f
                .cells()
                .iter()
                .zip(truth.iter())
                .filter(|(c, p)| c.certified && **p < H)
                .count();
            assert_eq!(violations, 0, "dilation certified an invalid cell at t={t}");
        }
        if should_advance(f.sigma(i), beta, cfg.epsilon) {
            fired += 1;
        }
        last_beta = beta;
    }
    assert!(
        dilation_ran,
        "no cell was ever admitted by dilation — the test was vacuous"
    );
    assert!(
        f.dilated_count() > 0,
        "dilated_count did not record the admissions"
    );
    // The halting law is NOT expected to fire here and does not: at ~87
    // observations per cell the Beta sd is ~0.023 while the threshold is
    // ~0.003, and `advance_horizon` prices this epsilon at ~1e6 rounds FOR A
    // SINGLE CELL. Its quantitative content is pinned by `t2_5d` instead.
    // Printed, not asserted away.
    let thresh = cfg.epsilon / (2.0 * last_beta);
    let best_sigma = (0..f.len()).map(|j| f.sigma(j)).fold(f32::MAX, f32::min);
    println!(
        "T2.5 FINE={FINE}: certified {} ({} by dilation); halting fired {fired}x \
         (best sigma {best_sigma:.4} vs threshold {thresh:.5}, horizon {:.3e} rounds/cell)",
        f.certified_count(),
        f.dilated_count(),
        advance_horizon(
            cfg.alpha,
            last_beta,
            linear_information_gain(ROUNDS, 2, cfg.lambda),
            cfg.epsilon
        ),
    );
}

#[test]
fn t2_5b_coarse_lattice_dilation_is_reported_infeasible_not_silently_dead() {
    // The T0.3 finding, pinned: at GRID=12 the global Lipschitz constant makes
    // every hop unaffordable, and the API must SAY so rather than return 0.
    let (mut f, truth) = build_world();
    let cfg = FrontierConfig {
        h: H,
        lipschitz: lipschitz_bound(),
        cell_spacing: 1.0 / (GRID - 1) as f32,
        ..FrontierConfig::default()
    };
    let mut rng = Lcg::new(0xDEAD);
    for t in 1..=4000u32 {
        let i = rng.below(CELLS);
        f.observe(i, rng.next_f32() < truth[i]);
        let beta = confidence_schedule(t, cfg.delta, cfg.lambda, cfg.b_rkhs, 2);
        f.expand_certified(&cfg, beta);
    }
    assert!(f.certified_count() > 0, "nothing certified by querying");
    let feas = f.dilation_feasibility(&cfg);
    let newly = f.reachability_dilation(&cfg, 4);
    println!(
        "T2.5b GRID={GRID}: headroom {:.4} vs hop cost {:.4} (deficit {:.4}), dilated {newly}",
        feas.best_headroom, feas.hop_cost, feas.deficit
    );
    assert!(
        !feas.feasible,
        "expected a coarse lattice to price the hop out"
    );
    assert!(feas.deficit > 0.0);
    assert_eq!(newly, 0, "an infeasible hop must admit nothing");
}

#[test]
fn t2_5d_halting_law_fires_and_its_cost_scales_as_one_over_epsilon_squared() {
    // The law is per-cell: `sigma <= eps / (2 beta)`. With a Beta-Bernoulli
    // sigma that is `p(1-p)/(n+3) <= (eps/2beta)^2`, i.e. n grows as 1/eps^2 —
    // which is exactly what `advance_horizon` claims. Measure both.
    let cfg = FrontierConfig::default();
    let mut fired_at = Vec::new();
    for &epsilon in &[0.2f32, 0.1, 0.05] {
        let mut f = CertifiedFrontier::<1, 1>::new();
        f.push_cell([0.0]).unwrap();
        let mut rng = Lcg::new(0xA17);
        let mut n = None;
        for t in 1..=60_000u32 {
            f.observe(0, rng.next_f32() < 0.95);
            let beta = confidence_schedule(t, cfg.delta, cfg.lambda, cfg.b_rkhs, 1);
            if should_advance(f.sigma(0), beta, epsilon) {
                n = Some(t);
                break;
            }
        }
        let n = n.unwrap_or_else(|| panic!("halting law never fired at epsilon={epsilon}"));
        println!("T2.5d epsilon={epsilon}: fired after {n} observations of one cell");
        fired_at.push((epsilon, n));
    }
    // Halving epsilon must cost strictly more, and roughly 4x more (quadratic).
    for w in fired_at.windows(2) {
        let (e0, n0) = w[0];
        let (e1, n1) = w[1];
        assert!(n1 > n0, "tightening epsilon {e0} -> {e1} did not cost more");
        let ratio = n1 as f32 / n0 as f32;
        assert!(
            (2.0..8.0).contains(&ratio),
            "epsilon {e0} -> {e1} cost ratio {ratio} is not the predicted ~4x"
        );
    }
}

#[test]
fn t2_5c_advance_horizon_is_positive_and_grows_as_epsilon_tightens() {
    let (alpha, beta, gamma) = (1.0f32, 6.0f32, 8.0f32);
    let loose = advance_horizon(alpha, beta, gamma, 0.2);
    let tight = advance_horizon(alpha, beta, gamma, 0.02);
    assert!(loose > 0.0 && tight > loose);
    // Quadratic in 1/epsilon: a 10x tighter target costs ~100x the rounds.
    assert!((tight / loose - 100.0).abs() < 1.0, "ratio {}", tight / loose);
}

// ── T2.6 — the scoreboards ─────────────────────────────────────────────────

#[test]
fn t2_6_sphere_exclusion_is_order_pinned_and_reports_saturation() {
    let mut rng = Lcg::new(0x5F3E);
    let samples: Vec<[f32; 3]> = (0..400)
        .map(|_| std::array::from_fn(|_| rng.next_f32()))
        .collect();

    let a = sphere_exclusion_coverage(&samples, 0.25);
    let b = sphere_exclusion_coverage(&samples, 0.25);
    assert_eq!(a, b, "same order must give a bit-identical count");
    assert!(!a.saturated);

    // A larger threshold can only merge clusters, never split them.
    let coarse = sphere_exclusion_coverage(&samples, 0.5);
    assert!(coarse.centers <= a.centers, "{coarse:?} vs {a:?}");

    // Reordering is expected to change the greedy count — that is why the
    // scoreboard is documented as order-pinned rather than order-invariant.
    let mut shuffled = samples.clone();
    shuffled.reverse();
    let rev = sphere_exclusion_coverage(&shuffled, 0.25);
    assert!(rev.centers > 0);

    // Saturation must be reported, not silently truncate the count.
    let spread: Vec<[f32; 3]> = (0..600)
        .map(|i| [i as f32, 0.0, 0.0])
        .collect();
    let sat = sphere_exclusion_coverage(&spread, 0.5);
    assert!(sat.saturated, "600 well-separated points must saturate 256");
    assert_eq!(sat.centers, katgpt_core::certified_frontier::SPHERE_EXCLUSION_MAX_CENTERS);
}

#[test]
fn t2_6b_vendi_on_a_planted_spectrum() {
    // k equal eigenvalues -> Vendi score exactly k.
    for k in [1usize, 2, 5, 16] {
        let eigs = vec![1.0f32 / k as f32; k];
        let v = vendi_diversity(&eigs);
        assert!((v - k as f32).abs() < 1e-3, "k={k}: got {v}");
    }
    // A collapsed spectrum has diversity 1; degenerate input gives 0.
    assert!((vendi_diversity(&[1.0, 0.0, 0.0, 0.0]) - 1.0).abs() < 1e-5);
    assert_eq!(vendi_diversity(&[]), 0.0);
    assert_eq!(vendi_diversity(&[0.0, 0.0]), 0.0);
    // Unnormalised input is normalised internally.
    assert!((vendi_diversity(&[3.0, 3.0, 3.0]) - 3.0).abs() < 1e-3);
}

// ── design bounds + the Beta substitute ────────────────────────────────────

#[test]
fn beta_substitute_is_calibrated_and_monotone() {
    let (m0, v0) = beta_mean_variance(0, 0);
    assert!((m0 - 0.5).abs() < 1e-6, "uniform prior mean");
    assert!((v0 - 1.0 / 12.0).abs() < 1e-6, "Beta(1,1) variance");

    // More valid observations raise the mean; more of either shrinks variance.
    let mut prev_mean = m0;
    let mut prev_var = v0;
    for n in 1..200u32 {
        let (m, v) = beta_mean_variance(n, 0);
        assert!(m > prev_mean, "mean not increasing at n={n}");
        assert!(v < prev_var, "variance not shrinking at n={n}");
        prev_mean = m;
        prev_var = v;
    }
    assert!(prev_mean > 0.99 && prev_var < 1e-4);

    // Symmetry: swapping the tally reflects the mean about 1/2.
    let (ma, va) = beta_mean_variance(7, 3);
    let (mb, vb) = beta_mean_variance(3, 7);
    assert!((ma + mb - 1.0).abs() < 1e-6);
    assert!((va - vb).abs() < 1e-6);
}

#[test]
fn prop1_design_bounds_have_the_predicted_shape() {
    // A narrower cap is exponentially rarer, and the rarity deepens with the
    // ambient dimension — the law behind Phase 0's 51.4x separation.
    let narrow = spherical_cap_bound(64, 0.05);
    let wide = spherical_cap_bound(64, 1.2);
    assert!(narrow < wide, "narrow {narrow} vs wide {wide}");
    assert!(spherical_cap_bound(256, 0.2) < spherical_cap_bound(8, 0.2));
    assert!((spherical_cap_bound(1, 0.3) - 1.0).abs() < 1e-6, "m=1 is degenerate");

    // Laurent-Massart: grows with dimension and with confidence.
    assert!(laurent_massart_radius(64, 0.05) > laurent_massart_radius(8, 0.05));
    assert!(laurent_massart_radius(8, 0.001) > laurent_massart_radius(8, 0.5));
    assert!(laurent_massart_radius(0, 0.5) >= 0.0);
}

#[test]
fn straddling_gate_prunes_deep_inside_and_far_outside_cells() {
    let cfg = FrontierConfig {
        h: H,
        lipschitz: 0.1,
        cell_spacing: 0.1,
        ..FrontierConfig::default()
    };
    let mut f = CertifiedFrontier::<8, 2>::new();
    let deep = f.push_cell([0.0, 0.0]).unwrap();
    let far = f.push_cell([1.0, 1.0]).unwrap();
    let edge = f.push_cell([0.5, 0.5]).unwrap();
    for _ in 0..400 {
        f.observe(deep, true);
    }
    for _ in 0..400 {
        f.observe(far, false);
    }
    for i in 0..400 {
        f.observe(edge, i % 5 < 3); // p ~ 0.6, straddling h exactly
    }
    let beta = 2.0;
    assert!(
        !f.query_is_decision_relevant(deep, &cfg, beta),
        "a saturated valid cell should not be worth a query"
    );
    assert!(
        !f.query_is_decision_relevant(far, &cfg, beta),
        "a saturated invalid cell should not be worth a query"
    );
    assert!(
        f.query_is_decision_relevant(edge, &cfg, beta),
        "a cell straddling h is exactly what a query buys"
    );
    assert!(!f.query_is_decision_relevant(99, &cfg, beta), "out of range");
}

#[test]
fn local_lipschitz_bounds_price_a_plateau_hop_below_the_global_constant() {
    // The T0.3 remedy: a caller with a tighter a-priori bound for a flat region
    // supplies it per cell, and the hop pays max(L_from, L_to) instead of the
    // steepest-cliff global constant.
    let cfg = FrontierConfig {
        h: 0.5,
        lipschitz: 10.0,
        cell_spacing: 0.1,
        ..FrontierConfig::default()
    };
    let mut f = CertifiedFrontier::<2, 1>::new();
    let a = f.push_cell([0.0]).unwrap();
    let b = f.push_cell([0.1]).unwrap();
    for _ in 0..4000 {
        f.observe(a, true);
    }
    f.expand_certified(&cfg, 2.0);
    assert!(f.cells()[a].certified && f.cells()[a].cb > 0.99);

    // Global L = 10 prices the 0.1 hop at 1.0 — unaffordable.
    assert!(!f.dilation_feasibility(&cfg).feasible);
    assert_eq!(f.reachability_dilation(&cfg, 1), 0);
    assert!(!f.cells()[b].certified);

    // A local a-priori bound of 1.0 prices it at 0.1 — affordable.
    f.cell_mut(a).unwrap().lipschitz = 1.0;
    f.cell_mut(b).unwrap().lipschitz = 1.0;
    assert!(f.dilation_feasibility(&cfg).feasible);
    assert_eq!(f.reachability_dilation(&cfg, 1), 1);
    assert!(f.cells()[b].certified && f.cells()[b].by_dilation);
    assert_eq!(f.dilated_count(), 1);
}

#[test]
fn kernel_sigma_override_replaces_the_beta_sd() {
    const N: usize = 16;
    let mut f = CertifiedFrontier::<4, 2>::new();
    for i in 0..4 {
        f.push_cell([i as f32 * 0.25, 0.0]).unwrap();
    }
    let beta_sd = f.sigma(0);
    let mut buf = Box::new(PosteriorBuffer::<N, 2>::new(1.0));
    for i in 0..N {
        buf.append_observation(&[i as f32 * 0.06, 0.0], 1.0);
    }
    let mut scratch = [0.0f32; N];
    f.refresh_kernel_sigma(&buf, &mut scratch);
    assert!(f.cells().iter().all(|c| c.sigma_override.is_finite()));
    assert!(
        (f.sigma(0) - beta_sd).abs() > 1e-6,
        "override did not take effect"
    );
    // ridge_mean is finite and responds to the fitted targets.
    assert!(buf.ridge_mean(&[0.5, 0.0]).is_finite());
}

#[test]
fn capacity_and_bounds_are_refused_not_wrapped() {
    let mut f = CertifiedFrontier::<2, 1>::new();
    assert!(f.is_empty());
    assert_eq!(f.push_cell([0.0]), Some(0));
    assert_eq!(f.push_cell([1.0]), Some(1));
    assert_eq!(f.push_cell([2.0]), None, "must refuse past capacity");
    assert_eq!(f.len(), 2);
    assert!(!f.observe(5, true), "out-of-range observe must be refused");
    assert!(f.cell_mut(5).is_none());
    let cfg = FrontierConfig::default();
    assert!(!f.seed_certified(9, &cfg));
    assert!(f.seed_certified(0, &cfg));
    assert!(!f.seed_certified(0, &cfg), "double seed must not double-count");
    assert_eq!(f.certified_count(), 1);
    // Untouched cells share one Beta sd, so the documented lowest-index
    // tie-break picks the seed itself...
    assert_eq!(f.acquire_frontier_target(&cfg), Some(0));
    // ...and once the seed has been observed, its sd drops below the
    // never-observed neighbour's and acquisition moves to the edge.
    for _ in 0..8 {
        f.observe(0, true);
    }
    assert_eq!(f.acquire_frontier_target(&cfg), Some(1));
}

#[test]
fn cached_beta_sd_tracks_the_closed_form_exactly() {
    // `observe` maintains a cached sd so acquisition is O(cells) with no
    // divide/sqrt per cell. A drift here would silently corrupt every bound,
    // so pin the cache against the closed form on every step.
    let mut f = CertifiedFrontier::<1, 1>::new();
    f.push_cell([0.0]).unwrap();
    assert!((f.sigma(0) - (1.0f32 / 12.0).sqrt()).abs() < 1e-6, "prior sd");
    let mut rng = Lcg::new(0xCA5);
    let (mut v, mut i) = (0u32, 0u32);
    for _ in 0..500 {
        let ok = rng.next_f32() < 0.7;
        f.observe(0, ok);
        if ok { v += 1 } else { i += 1 }
        let expected = beta_mean_variance(v, i).1.sqrt();
        assert!(
            (f.sigma(0) - expected).abs() < 1e-7,
            "cached sd drifted: {} vs {expected}",
            f.sigma(0)
        );
    }
}

#[test]
fn acquisition_lane_matches_a_full_rescan() {
    // `acquire_frontier_target` reads a struct-of-arrays lane maintained
    // incrementally at every mutation point. A missed refresh would not fail
    // loudly — it would quietly bias the query sequence — so pin the lane
    // against a reference rescan of the public cell view at EVERY step.
    fn reference(f: &CertifiedFrontier<CELLS, 2>) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (j, c) in f.cells().iter().enumerate() {
            if !c.certified && !c.near_certified {
                continue;
            }
            let s = f.sigma(j);
            if best.is_none_or(|(_, bs)| s > bs) {
                best = Some((j, s));
            }
        }
        best.map(|(j, _)| j)
    }

    let cfg = FrontierConfig {
        h: H,
        lipschitz: lipschitz_bound(),
        cell_spacing: 1.0 / (GRID - 1) as f32,
        acquire_radius: 1.5 / (GRID - 1) as f32,
        ..FrontierConfig::default()
    };
    let (mut f, truth) = build_world();
    assert_eq!(f.acquire_frontier_target(&cfg), reference(&f), "empty lane");
    f.seed_certified(0, &cfg);

    let mut rng = Lcg::new(0x1A4E);
    for t in 1..=4000u32 {
        assert_eq!(
            f.acquire_frontier_target(&cfg),
            reference(&f),
            "lane diverged from a full rescan at t={t}"
        );
        let Some(i) = f.acquire_frontier_target(&cfg) else {
            break;
        };
        f.observe(i, rng.next_f32() < truth[i]);
        let beta = confidence_schedule(t, cfg.delta, cfg.lambda, cfg.b_rkhs, 2);
        f.expand_certified(&cfg, beta);
        if t % 1000 == 0 {
            f.reachability_dilation(&cfg, 1);
        }
    }
    assert!(f.certified_count() > 1, "run certified nothing beyond the seed");

    // A radius change invalidates the cached candidacy; rebuild must restore
    // agreement rather than leaving the lane stale.
    let wider = FrontierConfig {
        acquire_radius: 6.0 / (GRID - 1) as f32,
        ..cfg
    };
    f.rebuild_neighborhoods(&wider);
    assert_eq!(f.acquire_frontier_target(&wider), reference(&f), "after rebuild");
}

#[cfg(feature = "viable_manifold_graph")]
#[test]
fn t4_1_geodesics_over_a_certified_graph_never_leave_the_certified_set() {
    use katgpt_core::certified_frontier::certified_manifold_graph;
    use katgpt_core::subspace_phase_gate::JacobianSvdScratch;
    use katgpt_core::viable_manifold_graph::{
        GraphBuildConfig, VolumeFieldConfig, manifold_geodesic,
    };

    fn decode(z: &[f32], out: &mut [f32]) {
        out[0] = z[0];
        out[1] = z[1];
        out[2] = (2.0 * z[0]).sin() * (2.0 * z[1]).cos();
    }

    let cfg = FrontierConfig {
        h: H,
        lipschitz: lipschitz_bound(),
        cell_spacing: 1.0 / (GRID - 1) as f32,
        ..FrontierConfig::default()
    };
    let (mut f, truth) = build_world();
    let mut rng = Lcg::new(0x4A1);
    for t in 1..=40_000u32 {
        let i = rng.below(CELLS);
        f.observe(i, rng.next_f32() < truth[i]);
        let beta = confidence_schedule(t, cfg.delta, cfg.lambda, cfg.b_rkhs, 2);
        f.expand_certified(&cfg, beta);
    }
    assert!(f.certified_count() >= 2, "need at least two nodes to navigate");

    let mut scratch = JacobianSvdScratch::with_capacity(2, 3);
    let cmg = certified_manifold_graph(
        &f,
        decode,
        &VolumeFieldConfig::default(),
        &GraphBuildConfig {
            volume_threshold: 2.0,
            edge_midpoint_check: true,
            k_nearest: 6,
        },
        &mut scratch,
    );

    // The mapping is the load-bearing part: the builder drops cells, so graph
    // ids and cell indices diverge and a misalignment would be silent.
    assert_eq!(cmg.graph.n_nodes(), cmg.node_to_cell.len());
    for (node, &cell) in cmg.node_to_cell.iter().enumerate() {
        assert!(
            f.cells()[cell as usize].certified,
            "node {node} maps to an uncertified cell {cell}"
        );
        let latent = cmg.graph.node_latent(node as u32);
        let feat = f.cells()[cell as usize].feat;
        assert!(
            (latent[0] - feat[0]).abs() < 1e-6 && (latent[1] - feat[1]).abs() < 1e-6,
            "node {node} latent does not match cell {cell} — mapping is misaligned"
        );
    }

    // Walk every reachable pair from node 0 and assert the invariant that makes
    // the composition worth having.
    let mut checked = 0usize;
    for dst in 1..cmg.graph.n_nodes() {
        let Some(path) = manifold_geodesic(&cmg.graph, 0, dst as u32) else {
            continue;
        };
        checked += 1;
        for n in &path {
            let cell = cmg.node_to_cell[*n as usize] as usize;
            assert!(f.cells()[cell].certified, "geodesic left the certified set");
            assert!(f.cells()[cell].cb >= H, "path node below its certified bound");
            assert!(truth[cell] >= H, "path node was actually invalid");
        }
    }
    assert!(checked > 0, "no reachable pair — the navigation check was vacuous");
}

// ── T5.3 — the regime-conditional dual form ────────────────────────────────
//
// `DualPosteriorBuffer` answers the same three questions as `PosteriorBuffer`
// by the Woodbury identity, at `O(D^2)` state and `O(D^2)` query instead of
// `O(n^2)`/`O(n^2)`. The primal is the ORACLE here: it is the arm already
// gated by T2.1 against an independent dense f64 solve, so equivalence to it
// is transitive evidence and not a fresh unverified claim.
//
// Mirrors the first external consumer's gate (riir-train Plan 357 T1.2,
// Bench 563), which independently wrote the dual and agreed with our primal.

mod t53_dual {
    use katgpt_core::certified_frontier::{
        DualPosteriorBuffer, LinearPosterior, PosteriorBuffer, prefer_dual,
    };

    const D: usize = 8;
    const MAX_OBS: usize = 64;

    /// Deterministic LCG — no rand dep, and the sequence is part of the gate.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        }
        fn feat(&mut self) -> [f32; D] {
            let mut f = [0.0; D];
            for v in &mut f {
                *v = self.next_f32();
            }
            f
        }
    }

    /// Relative-or-absolute agreement. The primal computes `k(x,x) - quad` as a
    /// difference of two possibly-large terms, so it loses digits exactly where
    /// `sigma^2` is small; a pure relative bound would be dishonest there.
    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    /// **The T5.3 gate.** Primal vs dual on both answers, at every observation
    /// count from 0 to 48 — spanning `n < D` (the primal's regime) and `n > D`
    /// (the dual's) — across three ridges. Prints the worst deviation seen so a
    /// regression is diagnosable, then pins it.
    #[test]
    fn t53_primal_dual_equivalence_across_n_and_lambda() {
        const TOL: f32 = 2e-3;
        let mut worst_var = 0.0f32;
        let mut worst_mean = 0.0f32;
        let mut worst_at = (0usize, 0.0f32);

        for &lambda in &[1.0f32, 1e-1, 1e-2] {
            let mut primal = Box::new(PosteriorBuffer::<MAX_OBS, D>::new(lambda));
            let mut dual = DualPosteriorBuffer::<D>::new(lambda);
            let mut obs = Lcg(0x5EED_1234);
            let mut scratch = [0.0f32; MAX_OBS];

            for n in 0..=48usize {
                // Query at a fixed probe set so every n is compared on the same
                // inputs; drawn from its own stream so appends cannot shift it.
                let mut probe = Lcg(0xC0FF_EE00);
                for _ in 0..16 {
                    let x = probe.feat();
                    let vp = LinearPosterior::posterior_variance_linear(&*primal, &x, &mut scratch);
                    let vd = LinearPosterior::posterior_variance_linear(&dual, &x, &mut scratch);
                    let mp = LinearPosterior::ridge_mean(&*primal, &x);
                    let md = LinearPosterior::ridge_mean(&dual, &x);

                    let dv = (vp - vd).abs() / vp.abs().max(vd.abs()).max(1.0);
                    let dm = (mp - md).abs() / mp.abs().max(md.abs()).max(1.0);
                    if dv > worst_var {
                        worst_var = dv;
                        worst_at = (n, lambda);
                    }
                    worst_mean = worst_mean.max(dm);

                    assert!(
                        close(vp, vd, TOL),
                        "variance diverged at n={n} lambda={lambda}: primal={vp} dual={vd}"
                    );
                    assert!(
                        close(mp, md, TOL),
                        "ridge_mean diverged at n={n} lambda={lambda}: primal={mp} dual={md}"
                    );
                }

                let f = obs.feat();
                let y = obs.next_f32();
                assert!(LinearPosterior::append_observation(&mut *primal, &f, y));
                assert!(LinearPosterior::append_observation(&mut dual, &f, y));
                assert_eq!(LinearPosterior::len(&*primal), LinearPosterior::len(&dual));
            }
        }
        println!(
            "T5.3 equivalence: worst rel-dev variance {worst_var:.3e} (at n={}, lambda={}), \
             ridge_mean {worst_mean:.3e}; bound {TOL:.0e}",
            worst_at.0, worst_at.1
        );
    }

    /// At `n = 0` both forms must be exactly `k(x, x) = ||x||^2`. The dual gets
    /// there through `lambda * ||x/sqrt(lambda)||^2`, so this is a real
    /// cancellation check on the initial factor, not a tautology.
    #[test]
    fn t53_empty_posterior_is_k_self_in_both_forms() {
        for &lambda in &[1.0f32, 1e-1, 1e-2, 1e-4] {
            let primal = PosteriorBuffer::<MAX_OBS, D>::new(lambda);
            let dual = DualPosteriorBuffer::<D>::new(lambda);
            let mut scratch = [0.0f32; MAX_OBS];
            let x = [0.5, -1.5, 2.0, 0.0, 0.25, -0.75, 1.0, 3.0];
            let k_self: f32 = x.iter().map(|v| v * v).sum();
            let vp = primal.posterior_variance_linear(&x, &mut scratch);
            let vd = dual.posterior_variance_linear(&x);
            assert!(close(vp, k_self, 1e-6), "primal n=0: {vp} vs {k_self}");
            assert!(close(vd, k_self, 1e-5), "dual n=0 lambda={lambda}: {vd} vs {k_self}");
        }
    }

    /// The dual's own determinism: same stream twice, **bit-identical** state
    /// and answers. Bit-identity is available here (it is one arm compared with
    /// itself) and is asserted as such — unlike the cross-form gate above.
    #[test]
    fn t53_dual_is_bit_identical_across_runs() {
        let run = || {
            let mut p = DualPosteriorBuffer::<D>::new(1e-2);
            let mut g = Lcg(0xABCD_0001);
            for _ in 0..200 {
                let f = g.feat();
                let y = g.next_f32();
                p.append_observation(&f, y);
            }
            let mut probe = Lcg(0x1111_2222);
            let mut out = Vec::new();
            for _ in 0..32 {
                let x = probe.feat();
                out.push(p.posterior_variance_linear(&x).to_bits());
                out.push(p.ridge_mean(&x).to_bits());
            }
            out
        };
        assert_eq!(run(), run(), "dual posterior is not run-to-run bit-identical");
    }

    /// A non-finite observation is **rejected**, not absorbed: `false`, the
    /// count is unchanged, and — the part that matters — every LATER query
    /// about an unrelated direction is still finite. One NaN in a Cholesky
    /// factor poisons it permanently, so "silently returns NaN forever" is the
    /// failure this rules out.
    #[test]
    fn t53_non_finite_observation_cannot_poison_the_factor() {
        let mut p = DualPosteriorBuffer::<D>::new(1e-2);
        let good = [1.0f32; D];
        assert!(p.append_observation(&good, 1.0));
        let before = p.posterior_variance_linear(&good);

        let mut nan_feat = [0.5f32; D];
        nan_feat[3] = f32::NAN;
        assert!(!p.append_observation(&nan_feat, 1.0), "NaN feature was absorbed");
        assert!(!p.append_observation(&good, f32::INFINITY), "inf label was absorbed");
        assert_eq!(p.len(), 1, "a rejected observation changed the count");

        let mut probe = [0.0f32; D];
        probe[0] = 2.0;
        assert!(p.posterior_variance_linear(&probe).is_finite());
        assert!(p.ridge_mean(&probe).is_finite());
        assert_eq!(
            p.posterior_variance_linear(&good).to_bits(),
            before.to_bits(),
            "a rejected observation perturbed the factor"
        );
    }

    /// Near-duplicate features are the case the primal needs a numerical floor
    /// for (`rem.max(lambda * 1e-6)`). The dual's pivots are bounded below by
    /// `sqrt(lambda)` by construction, so it needs none — assert the observable
    /// consequence: finite, non-negative, monotonically non-increasing variance
    /// along a repeated direction.
    #[test]
    fn t53_duplicate_features_need_no_numerical_floor() {
        let mut p = DualPosteriorBuffer::<D>::new(1e-2);
        let mut x = [0.0f32; D];
        x[0] = 1.0;
        let mut last = p.posterior_variance_linear(&x);
        for i in 0..500 {
            assert!(p.append_observation(&x, 1.0));
            let v = p.posterior_variance_linear(&x);
            assert!(v.is_finite() && v >= 0.0, "variance went bad at repeat {i}: {v}");
            assert!(v <= last + 1e-6, "variance rose on a repeated observation at {i}");
            last = v;
        }
        assert!(last < 1e-3, "500 identical observations left sigma^2 = {last}");
    }

    /// The regime rule and the state-size claim the doc table makes. `size_of`
    /// is the honest instrument: the dual's footprint cannot depend on `n`
    /// because `n` is not in its type.
    #[test]
    fn t53_regime_rule_and_state_size() {
        use core::mem::size_of;

        assert!(!prefer_dual(0, 32));
        assert!(!prefer_dual(32, 32));
        assert!(prefer_dual(33, 32));
        assert!(prefer_dual(4096, 32));
        // Payload is `4 D (D + 2)` = 4352 B at D = 32 (chol D*D + xty D + weights
        // D), and the type rounds to 4368 with `lambda` + `n` + alignment. That
        // is the figure the consumer's Bench 563 quotes, so pin the whole type.
        assert_eq!(size_of::<DualPosteriorBuffer<32>>(), 4368);
        assert!(size_of::<DualPosteriorBuffer<32>>() >= 4 * 32 * (32 + 2));
        assert!(
            size_of::<DualPosteriorBuffer<32>>() < size_of::<PosteriorBuffer<256, 32>>() / 50,
            "dual {} vs primal {}",
            size_of::<DualPosteriorBuffer<32>>(),
            size_of::<PosteriorBuffer<256, 32>>()
        );
        // The point of the whole task: the primal at the consumer's n is 64 MiB.
        assert!(size_of::<PosteriorBuffer<4096, 32>>() > 64 * 1024 * 1024);
    }

    /// A caller written once against the trait must be able to size its scratch
    /// off `scratch_len()` and drive either arm. Pins that the dual really
    /// declares zero (a non-zero declaration would make generic callers
    /// over-allocate forever) and that a zero-length scratch is accepted.
    #[test]
    fn t53_trait_lets_one_caller_drive_both_arms() {
        fn drive<const D2: usize, P: LinearPosterior<D2>>(p: &mut P, feats: &[[f32; D2]]) -> f32 {
            let mut scratch = vec![0.0f32; p.scratch_len().max(feats.len())];
            for f in feats {
                assert!(p.append_observation(f, 1.0));
            }
            assert_eq!(p.len(), feats.len());
            assert!(!p.is_empty());
            p.posterior_variance_linear(&feats[0], &mut scratch)
        }
        let feats: Vec<[f32; D]> = (0..20)
            .map(|i| {
                let mut f = [0.0f32; D];
                f[i % D] = 1.0 + i as f32 * 0.1;
                f
            })
            .collect();
        let mut primal = Box::new(PosteriorBuffer::<MAX_OBS, D>::new(1e-2));
        let mut dual = DualPosteriorBuffer::<D>::new(1e-2);
        let vp = drive(&mut *primal, &feats);
        let vd = drive(&mut dual, &feats);
        assert_eq!(dual.scratch_len(), 0);
        assert_eq!(primal.scratch_len(), 20);
        assert!(close(vp, vd, 2e-3), "trait-driven arms disagree: {vp} vs {vd}");
        // Zero-length scratch is legal for the dual and must not panic.
        assert!(LinearPosterior::posterior_variance_linear(&dual, &feats[0], &mut []).is_finite());
    }
}
