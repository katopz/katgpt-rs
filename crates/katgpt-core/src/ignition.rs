//! Closed-form logistic ignition — the modelless half of NQF staged learning.
//!
//! **Provenance:** arXiv:2608.13335 "Neural Quadratic Forms" (Liu Ziyin et al.,
//! Aug 2026) Thms 5–8 — riir-train Research 422 §3.5 / Issue 459 T5. Under gradient descent on a factorized (LoRA-shaped) model,
//! each singular mode of the task correlation follows generalized
//! Lotka–Volterra dynamics whose solution is **logistic ignition in time**:
//!
//! ```text
//! z(t) = K / (1 + ((K − z₀)/z₀) · e^{−ζt})  ≡  K · σ(ζt − ln((K−z₀)/z₀))
//! ```
//!
//! with `z₀` the small initial amplitude, `K` the capacity (asymptote) and `ζ`
//! the growth rate — the mode's alignment with the task correlation. Modes
//! ignite sequentially in descending `ζ` order (saddle-to-saddle); the wait
//! before a mode becomes readable is the patience law
//! `t* = ln(1/ε)/ζ` (Thm 8). Small init changes how *sharp* the acquisition
//! is, not *what* is learned.
//!
//! Two design anchors this formalizes for the house:
//! 1. **Sigmoid-in-time is the adoption shape GD itself produces** — the
//!    second theoretical grounding for the sigmoid-not-softmax rule (first:
//!    R315's scale-exponent universality).
//! 2. **Patience should scale as 1/ζ** — pre-ignition signal is ε-small, so
//!    selection should key on *ignited* modes, not raw rates. This predicts
//!    riir-clippy Issue 026's measured starved-pool negative (amplifying
//!    pre-ignition evidence amplifies noise).
//!
//! All evaluation is closed-form (one `exp`, no iteration) and allocation-free.
//!
//! **Feature status:** opt-in (`ignition_schedule`). GOAT G1–G4 PASS
//! (Bench 666): G1 monotone ranking preservation (higher ζ ignites strictly
//! earlier; curve ordering matches ζ ordering), G2 ns-latency closed form,
//! G3 no-regression (feature-off build untouched — module compiles away),
//! G4 alloc-free. Promotion to default requires the consumer pilot win
//! (riir-clippy selection patience scaled by `ignition_time` vs fixed
//! patience on the heal-loop fixture); decision stays with the owner.
//!
//! ignition — closed-form logistic ignition (Issue 459 T5, Research 422 §3.5
//! via arXiv:2608.13335): z(t) = K·σ(ζt − ln((K−z₀)/z₀)) per singular mode,
//! the patience law t* = ln(1/ε)/ζ, and a ζ-descending ordering helper
//! (modes ignite in ζ order). Sigmoid-in-time is the adoption shape GD itself
//! produces — the second grounding for sigmoid-not-softmax (after R315).
//! Opt-in; GOAT G1–G4 PASS (Bench 666). Promotion needs the consumer pilot
//! win (selection patience ∝ 1/ζ vs fixed patience).

/// Closed-form logistic ignition curve `z(t) = K·σ(ζt − ln((K−z₀)/z₀))`.
///
/// `Copy` (three `f32`s) — pass by value; evaluation allocates nothing.
///
/// # Contract (asserted)
/// - `0 < z0 < k` — a mode starts small and grows toward capacity.
/// - `zeta > 0` — a growth rate, not a decay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IgnitionSchedule {
    z0: f32,
    k: f32,
    zeta: f32,
}

impl IgnitionSchedule {
    /// Construct from initial amplitude `z0`, capacity `k`, growth rate `zeta`.
    ///
    /// # Panics
    /// If `0 < z0 < k` or `zeta > 0` is violated (constructor contract, not
    /// hot path).
    pub fn new(z0: f32, k: f32, zeta: f32) -> Self {
        assert!(z0 > 0.0, "z0 must be positive, got {z0}");
        assert!(k > z0, "capacity k must exceed z0 (got k={k}, z0={z0})");
        assert!(zeta > 0.0, "zeta must be positive, got {zeta}");
        Self { z0, k, zeta }
    }

    /// Initial amplitude `z₀`.
    #[inline]
    pub fn initial(&self) -> f32 {
        self.z0
    }

    /// Capacity (asymptote) `K`.
    #[inline]
    pub fn capacity(&self) -> f32 {
        self.k
    }

    /// Growth rate `ζ` — the mode's alignment with the task correlation.
    #[inline]
    pub fn zeta(&self) -> f32 {
        self.zeta
    }

    /// Closed-form evaluation `z(t) = K / (1 + ((K−z₀)/z₀)·e^{−ζt})`.
    ///
    /// One `exp`, three multiplies — no iteration, no allocation. Saturates
    /// gracefully at both ends: `z(∞) = K` exactly (the exponential
    /// underflows), and for large negative `t` the ratio overflows to `inf`
    /// giving `z → 0` — the correct pre-ignition limit.
    #[inline]
    pub fn at(&self, t: f32) -> f32 {
        let r = (self.k - self.z0) / self.z0;
        self.k / (1.0 + r * (-self.zeta * t).exp())
    }

    /// Time at which the curve reaches `target` (inverse of [`at`](Self::at)).
    ///
    /// `t = ln((K−z₀)·target / ((K−target)·z₀)) / ζ`. This is the per-curve
    /// ignition time — the ε-relative form [`ignition_time`] is its
    /// capacity-free limit.
    ///
    /// # Panics
    /// Unless `z0 < target < k`.
    pub fn time_to_reach(&self, target: f32) -> f32 {
        let (z0, k) = (self.z0, self.k);
        assert!(
            target > z0,
            "target must exceed z0 (got target={target}, z0={z0})"
        );
        assert!(
            target < k,
            "target must stay below k (got target={target}, k={k})"
        );
        (((k - z0) * target) / ((k - target) * z0)).ln() / self.zeta
    }
}

/// The patience law `t* = ln(1/ε)/ζ` — the wait before a mode of alignment
/// `ζ` becomes readable at threshold `ε`.
///
/// Capacity-free: the paper's Thm 8 saddle-to-saddle ignition time. Modes
/// with small `ζ` have `ln(1/ε)`-amplified delays — the formal anchor for
/// scaling exploration patience as `1/ζ` instead of a fixed budget.
///
/// # Panics
/// Unless `zeta > 0` and `0 < eps < 1`.
#[inline]
pub fn ignition_time(zeta: f32, eps: f32) -> f32 {
    assert!(zeta > 0.0, "zeta must be positive, got {zeta}");
    assert!(
        (0.0..1.0).contains(&eps),
        "eps must be in (0, 1), got {eps}"
    );
    (-eps.ln()) / zeta
}

/// The REM commitment-time law `t* = 1 − (1 + σ·√(2 ln V))^(−1/a)` (DBTM,
/// arXiv:2609.15903 Thm 4.1 — Issue 811 / Research 563).
///
/// Same family as [`ignition_time`] (a closed-form "when is commitment
/// safe?" boundary) but derived from Random-Energy-Model statistics instead
/// of GLV patience: the posterior over the `V` simplex vertices is Gibbs at
/// inverse temperature β(t); the argmax is correct iff the target's energy
/// ρ² beats the max of `V−1` competing Gaussians ≈ ρ·√(2 ln V). Anchoring
/// (committing tokens / applying the anchor loss) **before** t* collapses
/// modes; **after** t* slows convergence (DBTM Table 12).
///
/// Reference points from the paper: LM1B (V=30 522) → t*≈0.82;
/// Sudoku (V=12) → t*≈0.69. Nearly flat in V — commitment safety scales
/// √(log V).
///
/// `sigma` is the noise scale σ (> 0); `a` the noise-growth exponent (> 0);
/// `vocab` the choice-set size V (≥ 2 so the log is defined). Closed form:
/// one `ln`, one `sqrt`, one `powf` — no iteration, no allocation.
///
/// # Panics
/// Unless `vocab >= 2`, `sigma > 0`, `a > 0` (constructor contract, not hot
/// path).
#[inline]
pub fn commit_time_star(vocab: usize, sigma: f32, a: f32) -> f32 {
    assert!(vocab >= 2, "vocab must be >= 2 so ln(V) > 0, got {vocab}");
    assert!(sigma > 0.0, "sigma must be positive, got {sigma}");
    assert!(a > 0.0, "a must be positive, got {a}");
    let v = vocab as f64;
    let spread = sigma as f64 * (2.0 * v.ln()).sqrt();
    1.0 - (1.0 + spread).powf(-1.0 / a as f64) as f32
}

/// Order mode indices by ignition order: **ζ-descending** (highest alignment
/// ignites first), ties broken by ascending index (deterministic).
///
/// Allocation-free selection sort over a caller-owned buffer — the pools this
/// serves (candidate selection among a handful of modes) are small; O(n²) is
/// the right shape here, matching the `&mut [T]`-scratch house pattern.
///
/// # Panics
/// Unless `out.len() == zetas.len()`.
pub fn order_by_ignition_into(zetas: &[f32], out: &mut [usize]) {
    assert_eq!(zetas.len(), out.len(), "out buffer must match zetas length");
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = i;
    }
    // Selection sort: repeatedly select the max-ζ remaining index.
    for i in 0..out.len() {
        let mut best = i;
        for j in (i + 1)..out.len() {
            let bj = out[j];
            let bb = out[best];
            // Strictly greater ζ wins; ties keep the earlier-found index
            // (index-ascending), which selection sort gives for free.
            if zetas[bj] > zetas[bb] {
                best = j;
            }
        }
        out.swap(i, best);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn std_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn closed_form_anchors() {
        let s = IgnitionSchedule::new(0.01, 1.0, 0.7);
        // Starts exactly at z0.
        assert!((s.at(0.0) - 0.01).abs() < 1e-7);
        // Saturates exactly at K for huge t (exp underflows to 0).
        assert_eq!(s.at(1e6), 1.0);
        // Monotone increasing over a grid.
        let mut prev = s.at(0.0);
        for i in 1..200 {
            let t = i as f32 * 0.1;
            let z = s.at(t);
            assert!(z > prev, "z must increase: t={t}, z={z}, prev={prev}");
            prev = z;
        }
    }

    #[test]
    fn sigmoid_form_identity() {
        // z(t) ≡ K·σ(ζt − ln((K−z₀)/z₀)) with the exact (non-approx) sigmoid.
        let s = IgnitionSchedule::new(0.02, 2.5, 1.3);
        let c = ((s.k - s.z0) / s.z0).ln();
        for i in 0..50 {
            let t = i as f32 * 0.25;
            let lhs = s.at(t);
            let rhs = s.k * std_sigmoid(s.zeta * t - c);
            assert!(
                ((lhs - rhs) / rhs).abs() < 1e-5,
                "sigmoid identity broken at t={t}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn ode_rk4_anchor() {
        // The closed form must be the solution of the GLV ODE
        // ż = ζ·z·(1 − z/K). RK4-integrate and compare — this pins the
        // formula to the dynamics, not just to a shape.
        let (z0, k, zeta) = (0.01f32, 1.0f32, 0.8f32);
        let s = IgnitionSchedule::new(z0, k, zeta);
        let dt = 0.01f32;
        let mut z = z0;
        for step in 1..=1000u32 {
            let t = (step - 1) as f32 * dt;
            let f = |z: f32| zeta * z * (1.0 - z / k);
            let k1 = f(z);
            let k2 = f(z + 0.5 * dt * k1);
            let k3 = f(z + 0.5 * dt * k2);
            let k4 = f(z + dt * k3);
            z += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            if step % 100 == 0 {
                let t_check = step as f32 * dt;
                let closed = s.at(t_check);
                assert!(
                    ((z - closed) / closed).abs() < 5e-4,
                    "ODE anchor drifted at t={t_check}: rk4={z} closed={closed} (anchor t={t})"
                );
            }
        }
    }

    #[test]
    fn time_to_reach_roundtrip() {
        let s = IgnitionSchedule::new(0.01, 1.0, 0.9);
        for &target in &[0.05f32, 0.2, 0.5, 0.8, 0.95] {
            let t = s.time_to_reach(target);
            assert!(t > 0.0, "future crossing expected for target={target}");
            let recovered = s.at(t);
            assert!(
                ((recovered - target) / target).abs() < 1e-5,
                "roundtrip failed: target={target}, at(t)={recovered}, t={t}"
            );
        }
    }

    #[test]
    fn g1_ignition_time_strictly_decreasing_in_zeta() {
        // Higher ζ ⟹ strictly earlier ignition, at every ε.
        let zetas = [0.1f32, 0.25, 0.5, 1.0, 2.0, 4.0];
        for &eps in &[1e-2f32, 1e-3, 1e-4] {
            let mut prev = f32::INFINITY;
            for &z in &zetas {
                let t = ignition_time(z, eps);
                assert!(
                    t < prev,
                    "t* must strictly decrease as ζ rises (z={z}, eps={eps})"
                );
                prev = t;
            }
            // The ln(1/ε) amplification: smaller ε scales every wait by the
            // same factor — patience grows logarithmically, not linearly.
            let t_e2 = ignition_time(1.0, 1e-2);
            let t_e4 = ignition_time(1.0, 1e-4);
            let ratio = t_e4 / t_e2;
            let expected = 1e-4f32.ln() / 1e-2f32.ln();
            assert!((ratio - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn g1_curve_ordering_matches_zeta_ordering() {
        // Same (z0, K), distinct ζ: at every t > 0 the z-ordering is exactly
        // the ζ-ordering (the latent-ops ranking contract).
        let zetas = [0.3f32, 0.8, 1.4, 2.6];
        let scheds: Vec<_> = zetas
            .iter()
            .map(|&z| IgnitionSchedule::new(0.01, 1.0, z))
            .collect();
        for i in 0..120 {
            let t = i as f32 * 0.25;
            let mut prev = f32::NEG_INFINITY;
            // Walk ζ ascending ⟹ z ascending at every t > 0, STRICTLY while
            // below capacity — at saturation all curves converge to exactly K
            // (f32) and strict ordering degenerates by design.
            for s in &scheds {
                let z = s.at(t);
                if t > 0.0 {
                    if z < 1.0 {
                        assert!(
                            z > prev,
                            "curve ordering broken at t={t}: z={z} prev={prev}"
                        );
                    } else {
                        assert_eq!(z, 1.0, "saturated curves must sit exactly at K");
                    }
                }
                prev = z;
            }
        }
    }

    #[test]
    fn g1_ordering_helper_matches_threshold_crossing_order() {
        // The behavioral contract: modes ignite (cross a threshold) in
        // exactly the order the helper reports.
        let zetas = [0.4f32, 1.9, 0.7, 3.1, 1.2];
        let mut order = [0usize; 5];
        order_by_ignition_into(&zetas, &mut order);
        assert_eq!(order, [3, 1, 4, 2, 0]);

        // Crossing times along the helper's order are strictly increasing.
        let threshold = 0.5f32;
        let scheds: Vec<_> = zetas
            .iter()
            .map(|&z| IgnitionSchedule::new(0.01, 1.0, z))
            .collect();
        let mut prev_t = f32::NEG_INFINITY;
        for &idx in &order {
            let t = scheds[idx].time_to_reach(threshold);
            assert!(t > prev_t, "crossing order must follow helper order");
            prev_t = t;
        }
    }

    #[test]
    fn ordering_helper_tie_breaks_by_index() {
        // Equal ζ ⟹ deterministic index-ascending order.
        let zetas = [1.0f32, 2.0, 1.0, 0.5];
        let mut order = [0usize; 4];
        order_by_ignition_into(&zetas, &mut order);
        assert_eq!(order, [1, 0, 2, 3]);
    }

    #[test]
    fn g2_latency_closed_form() {
        // Closed form: one exp + a few flops. Generous debug bound (the house
        // `g2_*` pattern — debug pays ~10× on transcendental paths).
        let s = IgnitionSchedule::new(0.01, 1.0, 0.9);
        let n = 100_000u32;
        // Warm up caches/branch predictors.
        let mut warm = 0.0f32;
        for i in 0..1000 {
            warm += s.at(i as f32 * 0.001);
        }
        assert!(warm > 0.0);
        let t0 = Instant::now();
        let mut acc = 0.0f32;
        for i in 0..n {
            acc += s.at(i as f32 * 0.001);
        }
        let ns_per = t0.elapsed().as_nanos() as f64 / f64::from(n);
        assert!(acc > 0.0, "accumulator must stay live");
        println!("g2: at() = {ns_per:.2} ns/call (n={n})");
        let bound = if cfg!(debug_assertions) { 500.0 } else { 50.0 };
        assert!(
            ns_per < bound,
            "at() too slow: {ns_per:.2} ns/call (bound {bound})"
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    fn g4_alloc_free_evaluation() {
        // at() + the ordering helper allocate nothing (the lib test binary
        // installs TrackingAllocator via TEST_GLOBAL_ALLOC).
        crate::alloc::reset_alloc_stats();
        let s = IgnitionSchedule::new(0.01, 1.0, 0.9);
        let mut acc = 0.0f32;
        for i in 0..1000u32 {
            acc += s.at(i as f32 * 0.01);
            acc += ignition_time(0.5, 1e-3);
        }
        let zetas = [0.3f32, 1.7, 0.9, 2.2];
        let mut order = [0usize; 4];
        order_by_ignition_into(&zetas, &mut order);
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(
            count, 0,
            "ignition evaluation must be alloc-free (acc={acc})"
        );
        assert_eq!(order[0], 3);
    }

    #[test]
    #[should_panic(expected = "z0 must be positive")]
    fn constructor_rejects_zero_z0() {
        let _ = IgnitionSchedule::new(0.0, 1.0, 0.5);
    }

    #[test]
    #[should_panic(expected = "capacity k must exceed z0")]
    fn constructor_rejects_inverted_range() {
        let _ = IgnitionSchedule::new(2.0, 1.0, 0.5);
    }

    #[test]
    #[should_panic(expected = "zeta must be positive")]
    fn patience_law_rejects_zero_zeta() {
        let _ = ignition_time(0.0, 1e-2);
    }

    #[test]
    #[should_panic(expected = "eps must be in (0, 1)")]
    fn patience_law_rejects_unit_eps() {
        let _ = ignition_time(1.0, 1.0);
    }

    // ── DBTM commitment-time law (Issue 811 / Research 563) ──

    #[test]
    fn commit_time_star_matches_paper_reference_points() {
        // The paper's own task constants: LM1B (V=30 522) t*≈0.82 and
        // Sudoku (V=12) t*≈0.69. The paper does not pin (σ, a) per task in
        // the text we distilled, so this pins the FORMULA at the paper's
        // σ=1-style operating point and asserts the shape (larger V → later
        // t*, both in (0,1), near-flat in V) rather than exact decimals.
        let lm1b = commit_time_star(30_522, 1.0, 1.0);
        let sudoku = commit_time_star(12, 1.0, 1.0);
        assert!((0.0..1.0).contains(&lm1b));
        assert!((0.0..1.0).contains(&sudoku));
        assert!(lm1b > sudoku, "larger V must push t* later");
        // Near-flat in V: a 2500× vocab increase moves t* by well under 0.2
        // (the √(log V) scaling the paper emphasizes).
        assert!(
            lm1b - sudoku < 0.2,
            "t* must be near-flat in V ({lm1b} vs {sudoku})"
        );
        // Sanity anchor at the closed form, hand-computed: V=12, σ=1, a=1 →
        // t* = 1 − 1/(1 + √(2 ln 12)) = 1 − 1/(1 + 2.3830…).
        let expected = 1.0 - 1.0 / (1.0 + (2.0f64 * 12f64.ln()).sqrt());
        assert!((sudoku as f64 - expected).abs() < 1e-6);
    }

    #[test]
    fn commit_time_star_monotone_in_sigma_and_vocab() {
        // More noise or more competitors ⟹ later safe commitment.
        for &v in &[12usize, 100, 30_522] {
            let mut prev = 0.0f32;
            for &s in &[0.25f32, 0.5, 1.0, 2.0] {
                let t = commit_time_star(v, s, 1.0);
                assert!(t > prev, "t* must increase in σ (v={v}, σ={s})");
                prev = t;
            }
        }
        let mut prev = 0.0f32;
        for &v in &[2usize, 12, 100, 1000, 30_522] {
            let t = commit_time_star(v, 1.0, 1.0);
            assert!(t > prev, "t* must increase in V (v={v})");
            prev = t;
        }
    }

    #[test]
    fn commit_time_star_degenerate_limits() {
        // σ → 0⁺: no competing energy, commitment safe immediately (t* → 0).
        assert!(commit_time_star(1000, 1e-6, 1.0) < 1e-4);
        // a → ∞: noise grows infinitely slowly — same immediate-safe limit.
        assert!(commit_time_star(1000, 1.0, 1e6) < 1e-4);
    }

    #[test]
    #[should_panic(expected = "vocab must be >= 2")]
    fn commit_time_star_rejects_unit_vocab() {
        let _ = commit_time_star(1, 1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn commit_time_star_rejects_zero_sigma() {
        let _ = commit_time_star(12, 0.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "a must be positive")]
    fn commit_time_star_rejects_zero_a() {
        let _ = commit_time_star(12, 1.0, 0.0);
    }
}
