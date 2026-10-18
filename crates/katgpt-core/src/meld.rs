//! meld — bounded non-associative composition law (Issue 801 T3).
//!
//! SOURCES: Research 560 (`.research/560_NAP_Meld_Bounded_NonAssociative_Composition.md`),
//! distilling arXiv:2609.14384 (Murphy — NAP + Meld) §4.9.1, PDF-verified.
//!
//! The law: `meld(u,v) = sat(κ·W·[λ⋆u + (1−λ⋆)v])`, κ=2, where
//! - λ⋆ is the **pointwise soft-min mixture** — the argmin of the
//!   offset-corrected Rényi-2 gate
//!   `g(λ) = λu + (1−λ)v + (1/β)·ln(λ² + (1−λ)²)` solved per coordinate
//!   (quadratic in λ, clamped to [0,1]; β→0 degrades to the mean, β→∞ to
//!   the hard min),
//! - W is a **tied orthogonal non-permutation** transform (normalized
//!   Hadamard — the second-order bracketing separator; tying IS the
//!   commutativity requirement, untied weights leak daughter order),
//! - sat is tanh(κ·) by default, with divisive-normalization and the
//!   unsaturated law-8 (`W·[λ⋆u+(1−λ⋆)v]`, boundedness dropped) arms.
//!
//! Properties (each carries a self-test): commutative EXACTLY (canonical
//! daughter ordering makes the argmin path swap-symmetric bit-for-bit),
//! non-associative at nesting (the point — grouping survives), bounded
//! through arbitrary recursion depth under tanh (Pod f32 safety),
//! disagreement-coding (λ⋆ → min-like as β·|u−v| grows; agreement → mean).
//!
//! Zero-alloc fixed-size `[f32; D]` → `[f32; D]`, const-generic D (power of
//! two, for the fast Walsh–Hadamard transform). OPT-IN behind the `meld`
//! feature pending the Issue 801 T4 riir-poc PoC verdict (Super-GOAT Q3 is
//! quality-contingent — this primitive does NOT claim the quality axis yet).
//!
//! # λ⋆ closed form (the derivation behind [`soft_min_lambda_star`])
//!
//! Per coordinate, with the daughters canonicalized to `hi ≥ lo` and
//! `d = hi − lo ≥ 0`, the gate objective is
//!
//! ```text
//! g(λ) = λ·hi + (1−λ)·lo + (1/β)·ln s(λ),   s(λ) = λ² + (1−λ)² = 2λ² − 2λ + 1
//! ```
//!
//! with exact boundary shortcuts `g(0) = lo` and `g(1) = hi` (s(0)=s(1)=1
//! zeroes the entropy term). Stationary points solve
//! `g'(λ) = d + (1/β)·(4λ−2)/s(λ) = 0`; multiplying through by `β·s(λ)`
//! gives the quadratic (with `t ≡ βd`):
//!
//! ```text
//! 2t·λ² + (4 − 2t)·λ + (t − 2) = 0,   disc = (4−2t)² − 8t(t−2) = 4(4 − t²)
//! ```
//!
//! - `t ≥ 2`: no real root and `g' ≥ 0` on [0,1] → argmin at λ = 0 (hard
//!   min — the disagreement-coding limit: large separation takes `lo`).
//! - `0 < t < 2`: g is strictly convex on [0,1] (`g'' = 8λ(1−λ)/(β·s²) ≥ 0`),
//!   so the (clamped) real root is the unique global argmin.
//! - `t == 0` (d = 0 exactly, or β·d underflow): the quadratic degenerates
//!   to the linear `4λ − 2 = 0` → λ⋆ = ½ (the mean).
//!
//! Numerically stable root (avoids the 0/0 of the textbook form as t → 0):
//! with `b = 4 − 2t`, `c = t − 2`, `q = −(b + √disc)/2` (b > 0 whenever
//! disc > 0, so q < 0 strictly), the load-bearing root is `λ₊ = c/q ∈
//! [0, ½]`; the conjugate `q/a` is provably ≤ 0 in this regime but is still
//! evaluated (after clamping) for robustness against regime drift. All
//! candidates `{0, clamped roots, 1}` are scored in a FIXED order with a
//! strict-< argmin — deterministic, hence commutativity-safe.
//!
//! # Bit-exact commutativity (canonical daughter ordering)
//!
//! `meld(u,v)` and `meld(v,u)` both canonicalize every coordinate to the
//! same `(hi, lo)` via `f32::max`/`f32::min` (selection operations: the
//! result is the same f32 regardless of operand order), solve the SAME λ⋆
//! from the same `(hi, lo, β)`, and form the same mixture
//! `λ̃·hi + (1−λ̃)·lo` (IEEE-754 addition is commutative bit-for-bit), then
//! apply the same tied W and the same pointwise saturation → identical
//! output bits. Asserted over thousands of pairs in the self-tests.
//!
//! # Saturation laws ([`MeldLaw`])
//!
//! - [`MeldLaw::Tanh`] — `tanh(κ·y)`, κ = [`MELD_KAPPA`] = 2 (the paper
//!   default; bounded |out| < 1 at any recursion depth).
//! - [`MeldLaw::DivisiveNorm`] — `y_i / √(ε + Σ_j y_j²)` with
//!   `ε = 1e-6`: population divisive normalization (Carandini & Heeger
//!   2012, ½-exponent L2 pool), hard `|out| ≤ 1` by construction since
//!   `Σ_j y_j² ≥ y_i²`. ⚠ Deliberate divergence from the Issue 801 T3 task
//!   text, flagged in the T3 report: the task sketched
//!   `y_i/√(ε + Σ_j y_j²/D)` ("unit-RMS") — that variant is bounded by √D,
//!   NOT by 1 (any coordinate above RMS maps above 1), so it cannot satisfy
//!   the depth-64 boundedness gate the same task mandates. The shipped form
//!   is the same law family scaled by 1/√D (identical up to `ε ↔ ε·D`) and
//!   is the variant that is bounded by construction.
//! - [`MeldLaw::Linear`] — the paper's unsaturated law-8
//!   `W·[λ⋆u + (1−λ⋆)v]` (Research 560 §1.6: matches meld on every quality
//!   axis across the depth ladder). **UNBOUNDED by design** — excluded from
//!   boundedness self-tests; consumers needing Pod-safe magnitude must use
//!   Tanh or DivisiveNorm.
//!
//! # Self-test map (inline `#[cfg(test)]`; run with `--features meld`)
//!
//! 1. `commutativity_is_bit_exact` — 100 seeded random pairs × β ∈
//!    {0.03, 0.3, 3.0, 30} × all three laws × D ∈ {4, 16}, ± the no-W arm:
//!    bitwise `(u,v) == (v,u)`.
//! 2. `lambda_star_matches_brute_force_argmin` — 128 random (hi, lo) pairs
//!    × 4 β, 10_001-point λ grid: `g(λ⋆ours) ≤ min_g_grid + 1e-6`.
//! 3. `non_associativity_exists` — seeded random triples where
//!    `meld(meld(a,b),c) ≠ meld(a,meld(b,c))`; hit rate printed.
//! 4. `bounded_through_depth_64` — 64-step chains `x_{k+1} = meld(x_k,
//!    fresh)` with `max|coord| ≤ 1.0` asserted at every step (Tanh +
//!    DivisiveNorm; Linear excluded — unbounded by design).
//! 5. `disagreement_coding_limits` — λ̃ → 0 (< 0.01) for β·|hi−lo| ≥ 2;
//!    → ½ for near-zero separation and for moderate separation at β = 0.03.
//! 6. `walsh_hadamard_is_orthogonal_non_permutation` — ‖Wx‖ = ‖x‖ within
//!    1e-5 (D = 16, 32); D = 4 exhaustive basis orthonormality + every
//!    |entry| = 1/√D (hence non-permutation).
//! 7. `beta_limits_mean_and_hard_min` — β = 0.003 ⇒ `meld ≈
//!    tanh(κ·W[(u+v)/2])` within 1e-3 at moderate separation; β = 1e6 ⇒
//!    exact hard-min mixture for well-separated daughters.
//!
//! meld — bounded non-associative composition law (Issue 801 T3, Research 560,
//! arXiv:2609.14384 §4.9.1): meld(u,v) = sat(κ·W·[λ⋆u+(1−λ⋆)v]) — pointwise
//! soft-min mixture λ⋆ (offset-corrected Rényi-2 gate argmin; β→0 mean,
//! β→∞ hard min), tied orthogonal non-permutation W (normalized Hadamard —
//! the second-order bracketing separator), tanh saturation κ=2 (+ divisive-
//! normalization and unsaturated law-8 arms). Commutative EXACTLY (swap-
//! symmetric argmin), non-associative by construction, disagreement-coding.
//! Zero-alloc fixed-size [f32; D] (D a power of two). OPT-IN pending the
//! Issue 801 T4 riir-poc PoC verdict.

/// κ — the saturation gain applied inside the tanh transfer (paper §4.9.1
/// fixes κ = 2; Research 560 §1.6 "tanh(κW[λ⋆u + (1−λ⋆)v]), κ=2").
pub const MELD_KAPPA: f32 = 2.0;

/// Divisive-normalization ε — keeps the denominator away from zero when the
/// transformed mixture vanishes (an all-zero input stays all-zero, not NaN).
const DIVISIVE_EPS: f32 = 1e-6;

/// Saturation transfer `sat` of the meld law (paper §4.9.1); the paper's
/// ablation hierarchy (Research 560 §1.6) says saturation contributes
/// ONLY boundedness — any bounded nonlinearity works, and the unsaturated
/// law-8 arm ([`MeldLaw::Linear`]) matches meld on every quality axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MeldLaw {
    /// `tanh(κ·y)` with κ = [`MELD_KAPPA`] — the paper default; bounded
    /// |out| < 1 under arbitrary composition depth (Pod f32 safety).
    Tanh = 0,
    /// `y_i / √(ε + Σ_j y_j²)` — divisive normalization; bounded
    /// |out| ≤ 1 by construction (see the module doc for the divergence
    /// note vs the task text's unit-RMS sketch).
    DivisiveNorm = 1,
    /// Law-8, the unsaturated arm: `out = y` unchanged — UNBOUNDED, kept
    /// for the paper's ablation ladder and quality-only comparisons.
    Linear = 2,
}

impl MeldLaw {
    /// Stable name for benches/logs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tanh => "tanh",
            Self::DivisiveNorm => "divisive_norm",
            Self::Linear => "linear",
        }
    }
}

/// The meld composer: `meld(u,v) = sat(κ·W·[λ⋆u + (1−λ⋆)v])` with κ = 2
/// (Issue 801 T3). Fixed-size, zero-alloc, `Copy`; D must be a power of
/// two ≥ 2 (fast Walsh–Hadamard transform). See the module doc for the
/// law, the λ⋆ closed form, and the bit-exact commutativity argument.
#[derive(Clone, Copy, Debug)]
pub struct MeldCompose<const D: usize> {
    beta: f32,
    law: MeldLaw,
    apply_w: bool,
}

impl<const D: usize> MeldCompose<D> {
    /// Standard composer: tied orthogonal W applied, saturation per `law`.
    ///
    /// debug_asserts: `beta` finite > 0; `D` a power of two ≥ 2.
    #[must_use]
    pub fn new(beta: f32, law: MeldLaw) -> Self {
        debug_assert!(
            beta.is_finite() && beta > 0.0,
            "meld: beta must be finite and > 0"
        );
        debug_assert!(
            D.is_power_of_two() && D >= 2,
            "meld: D must be a power of two >= 2"
        );
        Self {
            beta,
            law,
            apply_w: true,
        }
    }

    /// The paper's no-W ablation arm (Research 560 §1.6: removing the tied
    /// orthogonal transform drops bracketing recovery 0.843 → 0.684 at
    /// depth — the W is the load-bearing second-order separator).
    #[must_use]
    pub fn no_w(beta: f32, law: MeldLaw) -> Self {
        debug_assert!(
            beta.is_finite() && beta > 0.0,
            "meld: beta must be finite and > 0"
        );
        debug_assert!(
            D.is_power_of_two() && D >= 2,
            "meld: D must be a power of two >= 2"
        );
        Self {
            beta,
            law,
            apply_w: false,
        }
    }

    /// The soft-min inverse temperature.
    #[must_use]
    #[inline]
    pub const fn beta(&self) -> f32 {
        self.beta
    }

    /// The saturation law.
    #[must_use]
    #[inline]
    pub const fn law(&self) -> MeldLaw {
        self.law
    }

    /// Whether the tied orthogonal W is applied (false = the no-W ablation).
    #[must_use]
    #[inline]
    pub const fn applies_w(&self) -> bool {
        self.apply_w
    }

    /// Compose `u` and `v`: `sat(κ·W·[λ⋆u + (1−λ⋆)v])`. Commutative
    /// bit-for-bit; zero-alloc.
    #[must_use]
    pub fn meld(&self, u: &[f32; D], v: &[f32; D]) -> [f32; D] {
        let mut out = [0.0_f32; D];
        self.meld_into(u, v, &mut out);
        out
    }

    /// [`Self::meld`] writing into `out` (no return-value copy on the
    /// caller side). `out` may not alias `u`/`v`.
    pub fn meld_into(&self, u: &[f32; D], v: &[f32; D], out: &mut [f32; D]) {
        // 1) Canonical soft-min mixture — the bit-exact-commutativity core:
        //    (hi, lo) and λ⋆ depend only on the unordered daughter pair.
        for ((o, &a), &b) in out.iter_mut().zip(u.iter()).zip(v.iter()) {
            let hi = a.max(b);
            let lo = a.min(b);
            let lam = soft_min_lambda_star(hi, lo, self.beta);
            *o = lam * hi + (1.0 - lam) * lo;
        }
        // 2) Tied orthogonal non-permutation W (in place; ‖·‖ preserved).
        if self.apply_w {
            walsh_hadamard_in_place_normalized(out);
        }
        // 3) Saturation — pointwise, in place.
        match self.law {
            MeldLaw::Tanh => {
                for x in out.iter_mut() {
                    *x = (MELD_KAPPA * *x).tanh();
                }
            }
            MeldLaw::DivisiveNorm => {
                let mut ss = 0.0_f32;
                for x in out.iter_mut() {
                    ss += *x * *x;
                }
                let inv = 1.0 / (DIVISIVE_EPS + ss).sqrt();
                for x in out.iter_mut() {
                    *x *= inv;
                }
            }
            // Law-8: no saturation — deliberately unbounded (see module doc).
            MeldLaw::Linear => {}
        }
    }

    /// The canonical λ⋆ trace (the weight each coordinate puts on the
    /// LARGER daughter after `hi ≥ lo` ordering). This is the
    /// disagreement signal consumed by Issue 801 T4(c): λ̃ → 0 as
    /// β·(hi−lo) grows (min-taking), λ̃ ≈ ½ under agreement or small β.
    /// Commutative bit-for-bit (same canonicalization as `meld`).
    pub fn lambda_star_into(&self, u: &[f32; D], v: &[f32; D], out: &mut [f32; D]) {
        for ((o, &a), &b) in out.iter_mut().zip(u.iter()).zip(v.iter()) {
            *o = soft_min_lambda_star(a.max(b), a.min(b), self.beta);
        }
    }

    /// Apply the composer's tied orthogonal transform `W = H/√D` (the
    /// normalized Walsh–Hadamard map). Honors [`Self::applies_w`]: the
    /// no-W arm copies through unchanged, exactly as `meld` composes.
    #[must_use]
    pub fn apply_w(&self, x: &[f32; D]) -> [f32; D] {
        let mut out = *x;
        if self.apply_w {
            walsh_hadamard_in_place_normalized(&mut out);
        }
        out
    }

    /// [`Self::apply_w`] into `out` (aliasing is impossible through the
    /// borrow checker).
    pub fn apply_w_into(&self, x: &[f32; D], out: &mut [f32; D]) {
        *out = *x;
        if self.apply_w {
            walsh_hadamard_in_place_normalized(out);
        }
    }
}

/// The tied orthogonal non-permutation transform: the normalized
/// Walsh–Hadamard map `W = H/√D` (H² = D·I ⇒ W orthogonal and involutive;
/// every entry ±1/√D ⇒ non-permutation for all D ≥ 2). In-place iterative
/// butterfly, zero-alloc, O(D log D).
pub fn walsh_hadamard_in_place_normalized<const D: usize>(x: &mut [f32; D]) {
    debug_assert!(
        D.is_power_of_two(),
        "Walsh–Hadamard requires D a power of two"
    );
    let inv = 1.0 / (D as f32).sqrt();
    let mut h = 1;
    while h < D {
        for chunk in x.chunks_exact_mut(2 * h) {
            let (lo, hi) = chunk.split_at_mut(h);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let s = *a + *b;
                let d = *a - *b;
                *a = s;
                *b = d;
            }
        }
        h <<= 1;
    }
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// The per-coordinate pointwise soft-min argmin λ⋆ for the CANONICAL
/// daughter ordering `hi ≥ lo` (callers must canonicalize — all methods on
/// [`MeldCompose`] do). λ⋆ weights `hi`; it tends to 0 as `β·(hi−lo)`
/// grows (min-taking) and to ½ as `β·(hi−lo) → 0` (mean). See the module
/// doc for the full derivation and the candidate-scoring determinism.
#[inline]
#[must_use]
pub fn soft_min_lambda_star(hi: f32, lo: f32, beta: f32) -> f32 {
    let d = hi - lo;
    // Exactly-equal daughters: the linear term is degenerate → λ⋆ = ½.
    if d == 0.0 {
        return 0.5;
    }
    let t = beta * d;
    // β·d underflowed (or a degenerate β in release builds): mean limit.
    if t <= 0.0 {
        return 0.5;
    }
    // Candidate λ = 0 is the exact boundary value g(0) = lo (s(0) = 1).
    let mut best_lam = 0.0_f32;
    let mut best_g = lo;
    // Interior stationary points exist only for disc > 0 (t < 2; t > 0 here).
    // Full discriminant of 2t·λ² + (4−2t)·λ + (t−2): 16 − 4t² = 4(4 − t²).
    let disc = 16.0 - 4.0 * t * t;
    if disc > 0.0 {
        let b = 4.0 - 2.0 * t;
        let c = t - 2.0;
        let q = -0.5 * (b + disc.sqrt());
        let a = 2.0 * t;
        // Stable pair {q/a, c/q}; both clamped, both scored (see module doc).
        for root in [q / a, c / q] {
            let lam = root.clamp(0.0, 1.0);
            let g = g_of(lam, d, lo, beta);
            if g < best_g {
                best_g = g;
                best_lam = lam;
            }
        }
    }
    best_lam
}

/// The gate objective in canonical (d = hi − lo) form:
/// `g(λ) = λ·d + lo + ln(2λ² − 2λ + 1)/β`. Exact at the boundaries
/// (g(0) = lo, g(1) = hi); used only for interior candidates.
#[inline]
fn g_of(lam: f32, d: f32, lo: f32, beta: f32) -> f32 {
    let s = 2.0 * lam * lam - 2.0 * lam + 1.0;
    lam * d + lo + s.ln() / beta
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic xorshift64* — enough for self-tests; no dep.
    struct XorShift(u64);

    impl XorShift {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// Uniform [0, 1).
        fn next_unit(&mut self) -> f32 {
            f32::from_bits(((self.next_u64() >> 40) as u32) | 0x3F80_0000) - 1.0
        }
        /// Uniform [−1, 1).
        fn next_pm1(&mut self) -> f32 {
            self.next_unit() * 2.0 - 1.0
        }
        fn fill<const N: usize>(&mut self, out: &mut [f32; N]) {
            for v in out.iter_mut() {
                *v = self.next_pm1();
            }
        }
    }

    /// Independent transcription of the gate objective (the brute-force
    /// referee for test 2 — deliberately NOT the d-form used internally,
    /// and evaluated in f64: at small β the objective reaches |g| ≈ 23,
    /// where one f32 ulp ≈ 2e-6 already exceeds the 1e-6 tolerance, so an
    /// f32 referee would false-fail on rounding alone. A real algebra slip
    /// moves g by 0.1–20; the f64 referee keeps the 1e-6 bar meaningful
    /// while the implementation under test stays f32.)
    fn g_ref(lam: f64, hi: f64, lo: f64, beta: f64) -> f64 {
        let mix = lam * hi + (1.0 - lam) * lo;
        let s = lam * lam + (1.0 - lam) * (1.0 - lam);
        mix + s.ln() / beta
    }

    /// Self-test 1 — commutativity EXACT over the full law × β × D matrix.
    #[test]
    fn commutativity_is_bit_exact() {
        fn run<const N: usize>(rng: &mut XorShift) -> usize {
            let betas = [0.03_f32, 0.3, 3.0, 30.0];
            let laws = [MeldLaw::Tanh, MeldLaw::DivisiveNorm, MeldLaw::Linear];
            let mut checked = 0_usize;
            for &beta in &betas {
                for &law in &laws {
                    let mc = MeldCompose::<N>::new(beta, law);
                    let mcnw = MeldCompose::<N>::no_w(beta, law);
                    for _ in 0..100 {
                        let mut u = [0.0_f32; N];
                        let mut v = [0.0_f32; N];
                        rng.fill(&mut u);
                        rng.fill(&mut v);
                        for m in [&mc, &mcnw] {
                            let ab = m.meld(&u, &v);
                            let ba = m.meld(&v, &u);
                            for k in 0..N {
                                assert_eq!(
                                    ab[k].to_bits(),
                                    ba[k].to_bits(),
                                    "commutativity broke: law {law:?} beta {beta} coord {k}"
                                );
                            }
                            checked += 1;
                        }
                    }
                }
            }
            checked
        }
        let mut rng = XorShift::new(801);
        let n = run::<4>(&mut rng) + run::<16>(&mut rng);
        println!("commutativity: {n} (composer, pair) cases bit-identical under (u,v)↔(v,u)");
        assert!(n >= 4800);
    }

    /// Self-test 2 — λ⋆ against a 10_001-point brute-force argmin.
    #[test]
    fn lambda_star_matches_brute_force_argmin() {
        const GRID: u32 = 10_000;
        let betas = [0.03_f32, 0.3, 3.0, 30.0];
        let mut rng = XorShift::new(802);
        let mut worst_margin = f64::NEG_INFINITY;
        let mut cases = 0_usize;
        for _ in 0..128 {
            let a = rng.next_pm1();
            let b = rng.next_pm1();
            let (hi, lo) = (a.max(b), a.min(b));
            for &beta in &betas {
                let lam = soft_min_lambda_star(hi, lo, beta);
                let g_ours = g_ref(
                    f64::from(lam),
                    f64::from(hi),
                    f64::from(lo),
                    f64::from(beta),
                );
                let mut g_min = f64::INFINITY;
                for k in 0..=GRID {
                    let l = f64::from(k) / f64::from(GRID);
                    g_min = g_min.min(g_ref(l, f64::from(hi), f64::from(lo), f64::from(beta)));
                }
                assert!(
                    g_ours <= g_min + 1e-6,
                    "λ⋆ missed the grid argmin: hi {hi} lo {lo} β {beta} g(λ⋆) {g_ours} g_min {g_min}"
                );
                worst_margin = worst_margin.max(g_min - g_ours);
                cases += 1;
            }
        }
        println!(
            "λ⋆ brute force: {cases} cases, worst margin g_min − g(λ⋆) = {worst_margin:.3e} (≥ 0, ≤ 1e-6)"
        );
        assert!(worst_margin >= -1e-6);
    }

    /// Self-test 3 — non-associativity EXISTS (the point of the primitive).
    #[test]
    fn non_associativity_exists() {
        let mut rng = XorShift::new(803);
        let mc = MeldCompose::<16>::new(3.0, MeldLaw::Tanh);
        let tries = 512;
        let mut hits = 0_usize;
        for _ in 0..tries {
            let (mut a, mut b, mut c) = ([0.0_f32; 16], [0.0_f32; 16], [0.0_f32; 16]);
            rng.fill(&mut a);
            rng.fill(&mut b);
            rng.fill(&mut c);
            let ab_c = mc.meld(&mc.meld(&a, &b), &c);
            let a_bc = mc.meld(&a, &mc.meld(&b, &c));
            if ab_c != a_bc {
                hits += 1;
            }
        }
        println!("non-associativity hit rate: {hits}/{tries}");
        assert!(
            hits >= 1,
            "no non-associative triple found in {tries} tries"
        );
    }

    /// Self-test 4 — bounded through depth-64 recursion (Tanh + DN only;
    /// Linear is unbounded by design).
    #[test]
    fn bounded_through_depth_64() {
        fn run<const N: usize>(rng: &mut XorShift, law: MeldLaw) {
            let mc = MeldCompose::<N>::new(3.0, law);
            let mut x = [0.0_f32; N];
            rng.fill(&mut x);
            let n0 = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for v in x.iter_mut() {
                *v /= n0;
            }
            for step in 0..64 {
                let mut fresh = [0.0_f32; N];
                rng.fill(&mut fresh);
                let nf = fresh.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                for v in fresh.iter_mut() {
                    *v /= nf;
                }
                x = mc.meld(&x, &fresh);
                for (k, &val) in x.iter().enumerate() {
                    assert!(
                        val.abs() <= 1.0,
                        "law {:?} step {step}: |coord {k}| = {} > 1",
                        law,
                        val.abs()
                    );
                }
            }
        }
        let mut rng = XorShift::new(804);
        run::<16>(&mut rng, MeldLaw::Tanh);
        run::<16>(&mut rng, MeldLaw::DivisiveNorm);
    }

    /// Self-test 5 — disagreement coding: min-taking vs mean-like limits.
    #[test]
    fn disagreement_coding_limits() {
        // Large separation at β = 30 (t = 30 ≥ 2): hard min — hi's weight 0.
        let lam = soft_min_lambda_star(0.5, -0.5, 30.0);
        assert!(
            lam < 0.01,
            "large separation should take the min, got {lam}"
        );
        // Near-zero separation: mean.
        let lam = soft_min_lambda_star(1e-6, -1e-6, 3.0);
        assert!(
            (lam - 0.5).abs() < 0.01,
            "agreement should be mean-like, got {lam}"
        );
        // Wide-window low end: moderate separation at β = 0.03 (t = 0.015).
        let lam = soft_min_lambda_star(0.25, -0.25, 0.03);
        assert!(
            (lam - 0.5).abs() < 0.01,
            "small β should be mean-like, got {lam}"
        );
        // Vector level: the λ⋆ trace reads ~0 across a well-separated pair.
        let mc = MeldCompose::<16>::new(30.0, MeldLaw::Tanh);
        let u = [0.5_f32; 16];
        let v = [-0.5_f32; 16];
        let mut trace = [0.0_f32; 16];
        mc.lambda_star_into(&u, &v, &mut trace);
        for (k, &l) in trace.iter().enumerate() {
            assert!(l < 0.01, "λ⋆ trace coord {k} = {l} not min-like");
        }
    }

    /// Self-test 6 — W is orthogonal and non-permutation.
    #[test]
    fn walsh_hadamard_is_orthogonal_non_permutation() {
        fn norm_preserved<const N: usize>(rng: &mut XorShift) {
            let mc = MeldCompose::<N>::new(1.0, MeldLaw::Tanh);
            for _ in 0..32 {
                let mut x = [0.0_f32; N];
                rng.fill(&mut x);
                let y = mc.apply_w(&x);
                let nx = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                let ny = y.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                assert!(
                    (nx - ny).abs() / nx < 1e-5,
                    "‖Wx‖ {ny} != ‖x‖ {nx} at D = {N}"
                );
            }
        }
        let mut rng = XorShift::new(805);
        norm_preserved::<16>(&mut rng);
        norm_preserved::<32>(&mut rng);
        // D = 4 exhaustive: W's columns (W·e_i) form an orthonormal set,
        // every entry has magnitude 1/√D (hence W is NOT a permutation).
        let mc = MeldCompose::<4>::new(1.0, MeldLaw::Tanh);
        let cols: [[f32; 4]; 4] = std::array::from_fn(|i| {
            let mut e = [0.0_f32; 4];
            e[i] = 1.0;
            mc.apply_w(&e)
        });
        for (i, ci) in cols.iter().enumerate() {
            for (j, cj) in cols.iter().enumerate() {
                let dot: f32 = ci.iter().zip(cj.iter()).map(|(a, b)| a * b).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - want).abs() < 1e-6,
                    "column {i}·column {j} = {dot}, want {want}"
                );
            }
        }
        let inv_sqrt = 1.0 / (4.0_f32).sqrt();
        for col in &cols {
            for &val in col {
                assert!(
                    (val.abs() - inv_sqrt).abs() < 1e-6,
                    "entry magnitude {} != 1/√D — W is not the normalized Hadamard",
                    val.abs()
                );
            }
        }
    }

    /// Self-test 7 — the two β limits (mean at β→0, hard min at β→∞).
    #[test]
    fn beta_limits_mean_and_hard_min() {
        fn run<const N: usize>(rng: &mut XorShift) {
            // Reference composer used ONLY for its apply_w (W + flag).
            let w_ref = MeldCompose::<N>::new(3.0, MeldLaw::Tanh);
            // β → 0 (0.003): λ⋆ ≈ ½ ⇒ meld ≈ tanh(κ·W[(u+v)/2]) within 1e-3
            // at MODERATE separation (|hi−lo| ≤ 0.1 keeps t ≤ 3e-4).
            let low = MeldCompose::<N>::new(0.003, MeldLaw::Tanh);
            for _ in 0..64 {
                let mut u = [0.0_f32; N];
                rng.fill(&mut u);
                let mut v = [0.0_f32; N];
                for (vi, &ui) in v.iter_mut().zip(u.iter()) {
                    let step = 0.01 + 0.09 * rng.next_unit();
                    *vi = ui + if rng.next_unit() < 0.5 { step } else { -step };
                }
                let got = low.meld(&u, &v);
                let mut m = [0.0_f32; N];
                for (mi, (&ui, &vi)) in m.iter_mut().zip(u.iter().zip(v.iter())) {
                    *mi = (ui + vi) * 0.5;
                }
                let want = w_ref.apply_w(&m).map(|x| (MELD_KAPPA * x).tanh());
                for (k, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-3,
                        "β→0 limit: coord {k} got {g} want {w} (|Δ| = {})",
                        (g - w).abs()
                    );
                }
            }
            // β → ∞ (1e6) with |u−v| ≥ 1 ⇒ t ≥ 1e3 ≥ 2 ⇒ exact hard-min.
            let high = MeldCompose::<N>::new(1.0e6, MeldLaw::Tanh);
            for _ in 0..64 {
                let mut u = [0.0_f32; N];
                let mut v = [0.0_f32; N];
                rng.fill(&mut u);
                rng.fill(&mut v);
                for (ui, vi) in u.iter_mut().zip(v.iter_mut()) {
                    if (*ui - *vi).abs() < 1.0 {
                        *vi -= 1.0;
                    }
                }
                let got = high.meld(&u, &v);
                let mut mn = [0.0_f32; N];
                for (mi, (&ui, &vi)) in mn.iter_mut().zip(u.iter().zip(v.iter())) {
                    *mi = ui.min(vi);
                }
                let want = w_ref.apply_w(&mn).map(|x| (MELD_KAPPA * x).tanh());
                for (k, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-6,
                        "β→∞ limit: coord {k} got {g} want {w} (|Δ| = {})",
                        (g - w).abs()
                    );
                }
            }
        }
        let mut rng = XorShift::new(806);
        run::<16>(&mut rng);
    }
}
