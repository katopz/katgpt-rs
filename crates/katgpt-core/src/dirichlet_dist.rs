//! Dirichlet-distribution primitives — the exact explore dial, the thinning
//! transition, and Dirichlet-EMA belief memory (Issue 912 T2+T3, Research 596).
//!
//! Source: [Research 596](../../.research/596_Simplex_Diffusion_Models.md) —
//! arXiv:2609.35553 "Simplex Diffusion Models" (the thinning/EMA propositions
//! are classical; the paper disclaims originality). Opt-in
//! (`dirichlet_dist`, default-off). All hot paths are allocation-free and
//! log-space — see "Why log space" below.
//!
//! # The three operators
//!
//! 1. **[`sample_conc_into`] — the exact explore dial.** Given a belief
//!    vector `p`, draw `Y ~ Dir(c·p)`. `E[Y] = p` **exactly by
//!    construction** (`c` = Σα is the concentration and drops out of the
//!    mean), `Var[Y_i] = p_i(1−p_i)/(c+1)`, `Cov[Y_i,Y_j] =
//!    −p_i·p_j/(c+1)`. This is the operator for "K noisy copies of a
//!    categorical belief"; the designer-facing knob is `c = 1/ν − 1` with
//!    ν ∈ (0,1) the variance fraction.
//! 2. **[`thinning_into`] — the Dirichlet thinning transition** (Prop A.2):
//!    `X ~ Dir(α)`, independent `B_i ~ Beta(ρα_i, (1−ρ)α_i)` ⇒
//!    `Y_i = B_iX_i/ΣB_jX_j ~ Dir(ρα)`. **Exact ONLY under Dirichlet input
//!    with the same `α`** — NOT mean-preserving on a fixed vector `p`
//!    (Jensen bias toward uniform: `p=(0.9,0.07,0.03)`, `c=2`, `ρ=0.5` ⇒
//!    `E[Y]=(0.847,0.103,0.051)`; pinned by
//!    `g1_thinning_fixed_vector_not_mean_preserving`). The per-coordinate
//!    variance ratio under a Dirichlet input is `(c+1)/(ρc+1)`, NOT ρ.
//! 3. **[`DirichletEma`] — belief memory** (Prop E.1 + Cor E.1): memory
//!    weights `L ~ Dir(ε·shares)` over the last `M` predictions under a
//!    geometric schedule. `E[L_j] = share_j` — deterministic and
//!    ε-independent; `Var[L_j] = ε·share_j(1−share_j)/(ε+1) → 0` as
//!    ε→∞ (deterministic weights) and the draw concentrates on vertices as
//!    ε→0 (resample-from-history). A finite ring TRUNCATES the infinite EMA
//!    tail — the `L_{i,M}` residual mass is named, not hidden: the drawn
//!    path's expectation is the RING closed form (normalized truncated
//!    geometric shares), deliberately NOT the recursive `h`, which retains
//!    the infinite tail (`E[drawn] − h = the truncated mass by
//!    construction`, so a consistency test against the recursive path would
//!    fail by design).
//!
//! # Why log space
//!
//! The Issue 912 T1 repair in `data_probe/markov.rs` draws the small-α boost
//! `U^{1/α}` in linear f32 — for α ≲ 0.02 that exponent underflows toward
//! the uniform fallback (the OPPOSITE of peaked). This module samples every
//! Gamma variate's LOGARITHM: the small-α boost becomes `ln(U)/α` (exact, no
//! underflow anywhere in the draw), and normalization is log-sum-exp.
//! Coordinates whose true probability is below f32's minimum still round to
//! 0.0 in the OUTPUT — correctly — but the draw itself never loses the
//! distribution's shape. G1 grid case C exercises α_min = 0.01, exactly
//! where the linear-space limit bites.
//!
//! # Consumers (T4 notes — documented candidates, NO wiring in this issue)
//!
//! - `perturbation_ensemble` (Bernoulli bucket dropout, heuristic law) — an
//!   interface-swap arm to the exact categorical noiser; G3 = `U_pair`
//!   stats no-regress on the existing rig. NOTE: reflex Bench 092 measured
//!   the perturbation-ensemble *confidence-ranking* use NEGATIVE — this
//!   lane claims only exploration/diversity uses, not UQ-for-confidence.
//! - `katgpt-sense` `evolve_belief_additive` sibling — G3 = mean-path
//!   bit-identical at ε=∞ (the recursive `h` routing is exactly that path).
//! - riir-neuron-db Raven/δ-Mem consolidation randomized merge weights
//!   ("uniform averaging has no attention-shift mechanism by design") —
//!   a mean-exact attention shift; their gate, their repo.
//! - `bom_arena` hypothesis sampler alternative (`sample_conc_into` vs the
//!   Gaussian/QMC hypothesis sampling) — their gate, their repo.
//!
//! # Explicit non-claims
//!
//! No coverage/prediction-interval claim → the conformal-naive floor does
//! NOT bind these primitives; it binds any future consumer that claims
//! calibrated uncertainty. No default-on promotion. Thinning is NOT claimed
//! as a general mean-preserving operator.
//!
//! Dirichlet-distribution primitives (Issue 912 T2+T3 / Research 596) — the
//! exact explore dial `Dir(c·p)` (mean exact by construction, log-space
//! sampling so α ≲ 0.02 stays exact), the thinning transition (exact ONLY
//! under Dirichlet input — NOT mean-preserving on fixed vectors), and the
//! Dirichlet-EMA belief memory (recursive mean path + randomized drawn path
//! with the ring-truncation named). Opt-in (`dirichlet_dist`).

/// Sample one Gamma(α, 1) variate's natural log — Marsaglia–Tsang
/// squeeze-rejection with the small-α boost taken IN LOG SPACE.
///
/// `α < 1`: `Gamma(α) = Gamma(α+1) · U^{1/α}` (Stuart's theorem), so
/// `ln Gamma(α) = ln Gamma(α+1) + ln(U)/α` — the division replaces the
/// linear-space `U^{1/α}` exponentiation that underflows for α ≲ 0.02
/// (the markov.rs limit this module exists to avoid). `α ≥ 1` runs the
/// standard `d·v` construction and returns `ln(d·v)` (`v > 0` is enforced
/// by rejection, so the log is finite).
///
/// `α == 0` (a zero shape — e.g. a zero-probability category) is the
/// degenerate point mass at 0: returns `-inf`.
///
/// Deterministic given the rng stream. Diagnostic/decision-event grade —
/// clarity over speed; allocation-free.
fn sample_gamma_log(alpha: f32, rng: &mut fastrand::Rng) -> f32 {
    debug_assert!(alpha >= 0.0);
    if alpha == 0.0 {
        return f32::NEG_INFINITY;
    }
    if alpha < 1.0 {
        let u = rng.f32().max(1e-10);
        return sample_gamma_log(alpha + 1.0, rng) + u.ln() / alpha;
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Box–Muller standard normal (one pair per draw).
        let u1 = rng.f32().max(1e-10);
        let u2 = rng.f32();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        let v = 1.0 + c * z;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.f32();
        // Squeeze test (fast acceptance).
        if u < 1.0 - 0.0331 * z * z * z * z {
            return (d * v).ln();
        }
        // Log-likelihood test (exact acceptance).
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return (d * v).ln();
        }
    }
}

/// Normalize a buffer of log-weights in place via log-sum-exp
/// (subtract max, exp, divide by the sum — the max coordinate contributes
/// exp(0) = 1, so the sum is ≥ 1 and the division cannot degenerate).
///
/// Returns `false` when every log-weight is non-finite (all `-inf`), in
/// which case the buffer is left for the caller's uniform fallback.
#[inline]
fn lse_normalize_into(buf: &mut [f32]) -> bool {
    let mut max = f32::NEG_INFINITY;
    for &v in buf.iter() {
        if v > max {
            max = v;
        }
    }
    if !max.is_finite() {
        return false;
    }
    let mut sum = 0.0f32;
    for v in buf.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in buf.iter_mut() {
        *v *= inv;
    }
    true
}

/// The exact explore dial: draw `Y ~ Dir(c·p)` into `out`.
///
/// `E[Y] = p / Σp` **exactly** (the Dirichlet mean is its normalized shape
/// vector; `p` need not be normalized by the caller — the draw treats `p`
/// as the shape). `Var[Y_i] = p̃_i(1−p̃_i)/(c+1)` with `p̃ = p/Σp`.
///
/// Each coordinate draws `Gamma(c·p_i, 1)` IN LOG SPACE
/// ([`sample_gamma_log`]) and the row is normalized by log-sum-exp — the
/// underflow regime (α ≲ 0.02, where the linear-space sampler collapses to
/// uniform) is exact here (grid case C pins it at α_min = 0.01).
///
/// Zero-probability categories (`p_i == 0`) get exactly `0.0`. If every
/// `p_i` is zero the draw falls back to uniform (documented degenerate
/// case, mirroring `markov.rs`).
///
/// `out.len()` must equal `p.len()`. Reuses `out` as scratch — zero
/// allocation. Deterministic in `(p, c, seed)`.
pub fn sample_conc_into(p: &[f32], c: f32, seed: u64, out: &mut [f32]) {
    debug_assert_eq!(p.len(), out.len());
    debug_assert!(c.is_finite() && c > 0.0);
    let k = out.len();
    if k == 0 {
        return;
    }
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut total = 0.0f32;
    for (i, slot) in out.iter_mut().enumerate() {
        let a = c * p[i];
        total += p[i];
        *slot = if a > 0.0 && a.is_finite() {
            sample_gamma_log(a, &mut rng)
        } else {
            f32::NEG_INFINITY
        };
    }
    if total <= 0.0 || !lse_normalize_into(out) {
        let v = 1.0 / k as f32;
        out.fill(v);
    }
}

/// The Dirichlet thinning transition (Prop A.2): draw
/// `B_i ~ Beta(ρα_i, (1−ρ)α_i)` per coordinate and return
/// `Y_i = B_i·x_i / ΣB_j·x_j`.
///
/// **Exact only when `x ~ Dir(α)` with the SAME `α` passed here** — the
/// `B_i` shapes depend on `α_i`, which is why the signature carries
/// `alpha` (the issue's one-line signature omitted it; exactness requires
/// it: `B_i·G_i ~ Gamma(ρα_i)` for `G_i ~ Gamma(α_i)` is the Beta–Gamma
/// identity the transition is built on, and it needs both factors'
/// shapes). On fixed (non-Dirichlet) vectors the map is NOT
/// mean-preserving — see the module docs and the pinned Jensen-bias demo.
///
/// `ρ == 1.0` short-circuits to the identity (`B ≡ 1` point mass): `out`
/// is a bitwise copy of `x` and NO rng stream is consumed — the ρ=1
/// bit-identity pin. `ρ` must be in `(0, 1]`.
///
/// Log-space throughout: `ln w_i = ln x_i + ln B_i`, log-sum-exp
/// normalize. A zero `x_i` contributes weight 0; if every `x_i` is zero
/// the output falls back to uniform (documented degenerate case).
///
/// `x.len() == alpha.len() == out.len()`. Reuses `out` as scratch — zero
/// allocation. Deterministic in `(x, alpha, rho, seed)`.
pub fn thinning_into(x: &[f32], alpha: &[f32], rho: f32, seed: u64, out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    debug_assert_eq!(x.len(), alpha.len());
    debug_assert!(rho > 0.0 && rho <= 1.0);
    let k = out.len();
    if k == 0 {
        return;
    }
    if rho == 1.0 {
        out.copy_from_slice(x);
        return;
    }
    let mut rng = fastrand::Rng::with_seed(seed);
    for (i, slot) in out.iter_mut().enumerate() {
        debug_assert!(alpha[i] >= 0.0);
        // ln B_i = ln G_a − ln(G_a + G_b) for the two Gamma draws.
        let lg_a = sample_gamma_log(rho * alpha[i], &mut rng);
        let lg_b = sample_gamma_log((1.0 - rho) * alpha[i], &mut rng);
        // A degenerate shape (α_i == 0 → both logs -inf) would NaN the
        // two-term LSE; in a valid x ~ Dir(α) pairing that coordinate's
        // x_i is 0, so any finite β weight yields the same (zero) mass.
        let ln_b = if lg_a.is_finite() && lg_b.is_finite() {
            // ln(G_a + G_b) = mx + ln(1 + exp(mn − mx)) — the OTHER term
            // is exactly 1 after the max shift, so the ln_1p argument is
            // the MIN's shift (using the max's own shift here would give
            // ln 2 whenever the two draws order one way — a bimodal,
            // Beta-incorrect B; the closed-form pins caught exactly that).
            let mx = lg_a.max(lg_b);
            let mn = lg_a.min(lg_b);
            lg_a - (mx + (mn - mx).exp().ln_1p())
        } else {
            0.0
        };
        *slot = if x[i] > 0.0 {
            x[i].ln() + ln_b
        } else {
            f32::NEG_INFINITY
        };
    }
    if !lse_normalize_into(out) {
        let v = 1.0 / k as f32;
        out.fill(v);
    }
}

/// Dirichlet-EMA belief memory over a fixed ring of the last `M` one-hot
/// predictions (Prop E.1 + Cor E.1).
///
/// Two views of the same memory, by design:
///
/// - **Mean path (deterministic, ε=∞ routing)** — `h`, a plain recursive
///   EMA `h ← β·h + (1−β)·onehot(new)` at a FIXED op order (scale-all,
///   then add). [`DirichletEma::mean`] returns it; it is bit-identical to
///   any EMA written at the same op order (pinned). Over a fresh start it
///   equals the raw truncated geometric weighted sum of the ring.
/// - **Drawn path (randomized)** — [`DirichletEma::drawn_weights`] draws
///   position weights `L_pos ~ Dir(ε·w)` over the ring (M Gammas, log
///   space, on decision events only) and aggregates to classes. By the
///   Dirichlet aggregation identity the class weights are exactly
///   `Dir(ε·s)` with `s` the ring's geometric shares, so `E[L_j] = share_j`
///   — deterministic and ε-independent — and
///   `Var[L_j] = ε·share_j(1−share_j)/(ε+1)`.
///
/// **The truncation is named, not hidden:** the ring covers the last `M`
/// pushes; `h` retains the infinite-EMA tail beyond the ring (the
/// `L_{i,M}` residual mass). [`DirichletEma::shares`] returns the ring
/// closed form (normalized truncated geometric shares) — the drawn path's
/// expectation — while `h` carries the tail, so `E[drawn] ≠ h` once the
/// history exceeds the ring (pinned as a divergence, not papered over).
///
/// Marginal drawn variance (single coordinate, `S = 1−β^filled` the raw
/// weight mass, `C = ε·S` the class-wise concentration):
/// `Var[L_j] = share_j(1−share_j)/(C+1)` → 0 as ε→∞ (deterministic
/// weights) with the draw concentrating on vertices as ε→0
/// (resample-from-history). The mean is `share_j` either way —
/// ε-independent.
///
/// `M` = ring length, `K` = class count (both compile-time: the whole
/// struct is fixed-size stack — zero heap).
pub struct DirichletEma<const M: usize, const K: usize> {
    /// Class index per ring slot. Slot `(pos + M − 1) % M` is the newest.
    ring: [u32; M],
    /// Next write slot (oldest slot once the ring is full).
    pos: usize,
    /// Number of observed pushes, capped at `M`.
    filled: usize,
    /// Geometric schedule parameter `β ∈ (0, 1)` — weight of the
    /// `i`-th-freshest observation is `(1−β)·β^i`.
    beta: f32,
    /// Recursive EMA state (the ε=∞ deterministic path).
    h: [f32; K],
}

impl<const M: usize, const K: usize> DirichletEma<M, K> {
    /// Fresh memory. `β ∈ (0,1)` is the geometric schedule (debug-asserted;
    /// the schedule degenerates at the endpoints).
    pub fn new(beta: f32) -> Self {
        debug_assert!(beta > 0.0 && beta < 1.0);
        Self {
            ring: [0; M],
            pos: 0,
            filled: 0,
            beta,
            h: [0.0; K],
        }
    }

    /// Record one one-hot prediction. O(1): one ring slot + the fixed EMA
    /// op (`h ← β·h` scale-all, then `h[class] += 1−β`) — the op order the
    /// bit-identity pin freezes.
    pub fn push(&mut self, class: usize) {
        debug_assert!(class < K);
        self.ring[self.pos] = class as u32;
        self.pos = (self.pos + 1) % M;
        if self.filled < M {
            self.filled += 1;
        }
        for v in self.h.iter_mut() {
            *v *= self.beta;
        }
        self.h[class] += 1.0 - self.beta;
    }

    /// The deterministic (ε=∞) path: the recursive EMA state. Sums to
    /// `1−β^N` after `N` fresh pushes (the infinite tail starts at zero).
    pub fn mean(&self) -> &[f32; K] {
        &self.h
    }

    /// The RING closed form: normalized truncated geometric shares
    /// `share_j = Σ_i w_i·[ring_i = j] / Σ_i w_i` over the `filled` ring
    /// positions, `w_i = (1−β)·β^i` newest-first. Sums to 1 when
    /// `filled > 0`; all zeros before the first push. This is the drawn
    /// path's expectation — NOT `h` once history exceeds the ring.
    pub fn shares(&self) -> [f32; K] {
        let mut s = [0.0f32; K];
        if self.filled == 0 {
            return s;
        }
        let mut norm = 0.0f32;
        let mut w = 1.0 - self.beta;
        for j in 0..self.filled {
            let slot = (self.pos + M - 1 - j) % M;
            s[self.ring[slot] as usize] += w;
            norm += w;
            w *= self.beta;
        }
        let inv = 1.0 / norm;
        for v in s.iter_mut() {
            *v *= inv;
        }
        s
    }

    /// Draw class memory weights `L ~ Dir(ε·shares)` (aggregated from the
    /// position-space draw — M Gammas in log space, exactly
    /// `Dir(ε·s)` class-wise by the Dirichlet aggregation identity).
    ///
    /// `ε > 0` finite. Deterministic in `(state, eps, seed)`. Before any
    /// push the ring is empty and `out` is returned all-zero (no memory to
    /// weight — documented degenerate case).
    pub fn drawn_weights(&self, eps: f32, seed: u64, out: &mut [f32; K]) {
        debug_assert!(eps.is_finite() && eps > 0.0);
        out.fill(0.0);
        if self.filled == 0 {
            return;
        }
        let mut lpos = [0.0f32; M];
        let mut w = [0.0f32; M];
        let mut wi = 1.0 - self.beta;
        for wj in w.iter_mut().take(self.filled) {
            *wj = wi;
            wi *= self.beta;
        }
        sample_conc_into(&w[..self.filled], eps, seed, &mut lpos[..self.filled]);
        for (j, &l) in lpos.iter().take(self.filled).enumerate() {
            let slot = (self.pos + M - 1 - j) % M;
            out[self.ring[slot] as usize] += l;
        }
    }
}

#[cfg(all(test, feature = "dirichlet_dist"))]
mod tests {
    use super::*;

    /// Bitwise equality for f32 slices (NaN-safe, ±0.0 distinguished) —
    /// the bit-identity pins compare bit PATTERNS, not float equality.
    fn bits_eq(a: &[f32], b: &[f32]) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    // ── G1: sample_conc_into — closed-form mean / variance / covariance ──

    /// Grid case A (moderate α): mean + variance + covariance vs the
    /// closed forms `E=p`, `Var=p(1−p)/(c+1)`, `Cov=−p_i p_j/(c+1)`.
    #[test]
    fn g1_conc_mean_var_cov_case_a() {
        let p = [0.5f32, 0.3, 0.2];
        let c = 20.0f32;
        let n = 40_000usize;
        let mut acc = [0.0f32; 3];
        let mut acc_sq = [0.0f32; 3];
        let mut acc_cross = 0.0f32;
        for t in 0..n {
            let mut out = [0.0f32; 3];
            sample_conc_into(&p, c, 1000 + t as u64, &mut out);
            for i in 0..3 {
                acc[i] += out[i];
                acc_sq[i] += out[i] * out[i];
            }
            acc_cross += out[0] * out[1];
        }
        let nf = n as f32;
        let mean = [acc[0] / nf, acc[1] / nf, acc[2] / nf];
        for i in 0..3 {
            assert!(
                (mean[i] - p[i]).abs() < 0.004,
                "mean[{i}] = {} vs p = {}",
                mean[i],
                p[i]
            );
        }
        // Var_0 = 0.25/21 ≈ 0.0119 (sample-var rel σ ≈ 0.7% at n = 40k).
        let var0 = acc_sq[0] / nf - mean[0] * mean[0];
        let var0_expected = 0.5 * 0.5 / (c + 1.0);
        assert!(
            (var0 - var0_expected).abs() < 0.0015,
            "var[0] = {var0} vs {var0_expected}"
        );
        // Cov_01 = −0.15/21 ≈ −0.00714 (SE ≈ 3e-4).
        let cov01 = acc_cross / nf - mean[0] * mean[1];
        let cov01_expected = -0.5 * 0.3 / (c + 1.0);
        assert!(
            (cov01 - cov01_expected).abs() < 0.0015,
            "cov[0,1] = {cov01} vs {cov01_expected}"
        );
    }

    /// Grid case B (small α): mean exactness holds at α_min = 0.1.
    #[test]
    fn g1_conc_mean_case_b() {
        let p = [0.6f32, 0.3, 0.1];
        let c = 1.0f32;
        let n = 40_000usize;
        let mut acc = [0.0f32; 3];
        for t in 0..n {
            let mut out = [0.0f32; 3];
            sample_conc_into(&p, c, 2000 + t as u64, &mut out);
            for i in 0..3 {
                acc[i] += out[i];
            }
        }
        let nf = n as f32;
        for i in 0..3 {
            assert!(
                (acc[i] / nf - p[i]).abs() < 0.01,
                "mean[{i}] = {} vs p = {}",
                acc[i] / nf,
                p[i]
            );
        }
    }

    /// Grid case C — the UNDERFLOW REGIME the module exists for:
    /// α_min = 0.01, where the linear-space sampler (markov.rs's
    /// `U^{1/α}`) collapses to uniform. The log-space draw keeps both the
    /// exact mean AND the peaked shape (near-vertex draws, mean row-max
    /// ≈ 0.9 vs uniform ≈ 0.52).
    #[test]
    fn g1_conc_underflow_regime_case_c() {
        let p = [0.5f32, 0.3, 0.2];
        let c = 0.05f32; // α = (0.025, 0.015, 0.01)
        let n = 40_000usize;
        let mut acc = [0.0f32; 3];
        let mut mean_max = 0.0f32;
        for t in 0..n {
            let mut out = [0.0f32; 3];
            sample_conc_into(&p, c, 3000 + t as u64, &mut out);
            for i in 0..3 {
                acc[i] += out[i];
            }
            mean_max += out.iter().copied().fold(0.0f32, f32::max);
        }
        let nf = n as f32;
        for i in 0..3 {
            assert!(
                (acc[i] / nf - p[i]).abs() < 0.015,
                "mean[{i}] = {} vs p = {}",
                acc[i] / nf,
                p[i]
            );
        }
        assert!(
            mean_max / nf > 0.8,
            "draws should be near-vertex peaked, mean max = {} (uniform ≈ 0.52)",
            mean_max / nf
        );
    }

    /// Determinism: same seed → bitwise-identical draw.
    #[test]
    fn g1_conc_deterministic_same_seed() {
        let p = [0.42f32, 0.33, 0.25];
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        sample_conc_into(&p, 3.0, 424242, &mut a);
        sample_conc_into(&p, 3.0, 424242, &mut b);
        assert!(bits_eq(&a, &b));
    }

    // ── G1: thinning_into ──

    /// Applied to seeded Dir(c·p) draws with the same α: the variance
    /// ratio across the transition is `(c+1)/(ρc+1)` — NOT ρ — and the
    /// post-transition distribution matches Dir(ρα) (mean + covariance
    /// closed forms).
    #[test]
    fn g1_thinning_var_ratio_and_dir_rho_alpha_match() {
        let p = [0.6f32, 0.3, 0.1];
        let c = 8.0f32;
        let rho = 0.5f32;
        let alpha = [c * p[0], c * p[1], c * p[2]];
        let n = 20_000usize;
        let mut var_before = [0.0f32; 3];
        let mut mean_before = [0.0f32; 3];
        let mut sq_before = [0.0f32; 3];
        let mut mean_after = [0.0f32; 3];
        let mut sq_after = [0.0f32; 3];
        let mut cross_after = 0.0f32;
        for t in 0..n {
            let mut x = [0.0f32; 3];
            let mut y = [0.0f32; 3];
            sample_conc_into(&p, c, 5000 + t as u64, &mut x);
            thinning_into(&x, &alpha, rho, 6000 + t as u64, &mut y);
            for i in 0..3 {
                mean_before[i] += x[i];
                sq_before[i] += x[i] * x[i];
                mean_after[i] += y[i];
                sq_after[i] += y[i] * y[i];
            }
            cross_after += y[0] * y[1];
        }
        let nf = n as f32;
        for i in 0..3 {
            mean_before[i] /= nf;
            var_before[i] = sq_before[i] / nf - mean_before[i] * mean_before[i];
            mean_after[i] /= nf;
        }
        let var_after = [
            sq_after[0] / nf - mean_after[0] * mean_after[0],
            sq_after[1] / nf - mean_after[1] * mean_after[1],
            sq_after[2] / nf - mean_after[2] * mean_after[2],
        ];
        // Ratio pin: (c+1)/(ρc+1) = 9/5 = 1.8 (rel σ of a variance-ratio
        // estimate ≈ 1% at n = 20k; 6% tolerance).
        for i in 0..3 {
            let ratio = var_after[i] / var_before[i];
            let expected = (c + 1.0) / (rho * c + 1.0);
            assert!(
                (ratio - expected).abs() / expected < 0.06,
                "var ratio[{i}] = {ratio} vs {expected}"
            );
            // Mean is p for BOTH Dir(α) and Dir(ρα).
            assert!(
                (mean_after[i] - p[i]).abs() < 0.012,
                "post mean[{i}] = {} vs p = {}",
                mean_after[i],
                p[i]
            );
        }
        // Cov matches Dir(ρα): −p_i p_j/(ρc+1) = −0.18/5 = −0.036.
        // Tolerance 3e-3 ≈ 7σ of the cov estimator (whose SE carries
        // higher-moment inflation beyond the gaussian 4e-4 estimate) —
        // still an order below the mis-drawn-B defect this pin exists for.
        let cov01 = cross_after / nf - mean_after[0] * mean_after[1];
        let cov01_expected = -p[0] * p[1] / (rho * c + 1.0);
        assert!(
            (cov01 - cov01_expected).abs() < 0.003,
            "post cov[0,1] = {cov01} vs {cov01_expected}"
        );
    }

    /// ρ = 1 is the identity: bitwise copy, no rng consumption.
    #[test]
    fn g1_thinning_rho1_bit_identity() {
        let x = [0.4f32, 0.35, 0.25];
        let alpha = [0.8f32, 0.7, 0.5];
        let mut out = [0.0f32; 3];
        thinning_into(&x, &alpha, 1.0, 7, &mut out);
        assert!(bits_eq(&x, &out));
    }

    /// The documented non-claim, pinned: on a FIXED vector the transition
    /// is NOT mean-preserving (Jensen bias toward uniform). The paper's
    /// simulated case: `p=(0.9,0.07,0.03)`, `c=2`, `ρ=0.5` ⇒
    /// `E[Y]=(0.847,0.103,0.051)`.
    #[test]
    fn g1_thinning_fixed_vector_not_mean_preserving() {
        let p = [0.9f32, 0.07, 0.03];
        let alpha = [2.0 * p[0], 2.0 * p[1], 2.0 * p[2]];
        let rho = 0.5f32;
        let n = 40_000usize;
        let mut acc = [0.0f32; 3];
        for t in 0..n {
            let mut y = [0.0f32; 3];
            thinning_into(&p, &alpha, rho, 7000 + t as u64, &mut y);
            for i in 0..3 {
                acc[i] += y[i];
            }
        }
        let nf = n as f32;
        let e0 = acc[0] / nf;
        println!(
            "fixed-vector thinning: E[Y] = ({:.4}, {:.4}, {:.4}) vs p = (0.9, 0.07, 0.03)",
            acc[0] / nf,
            acc[1] / nf,
            acc[2] / nf
        );
        // Matches the paper's simulated mean (±2e-2) AND is biased away
        // from p by > 3e-2 — the bias exists and is material.
        assert!(
            (e0 - 0.847).abs() < 0.02,
            "E[Y_0] on fixed p = {e0}, paper's simulated value 0.847"
        );
        assert!(
            p[0] - e0 > 0.03,
            "expected Jensen bias toward uniform, got E[Y_0] = {e0}"
        );
    }

    // ── G2: latency (release posture per the house rule — the bar is
    //    asserted only where the profile can honor it; debug runs the
    //    call for correctness) ──

    #[test]
    fn goat_g2_sample_conc_latency_n8() {
        let p = [0.26f32, 0.22, 0.18, 0.14, 0.10, 0.055, 0.030, 0.015];
        let mut out = [0.0f32; 8];
        // Warmup + liveness (a black_boxed result the optimizer cannot
        // delete — the loud-zero defence).
        sample_conc_into(std::hint::black_box(&p), 4.0, 1, &mut out);
        std::hint::black_box(&out);
        let iters = 20_000usize;
        let mut best_ns = f64::INFINITY;
        for round in 0..5u64 {
            let t = std::time::Instant::now();
            for i in 0..iters {
                sample_conc_into(
                    std::hint::black_box(&p),
                    4.0,
                    (round * iters as u64 + i as u64) | 1,
                    &mut out,
                );
                std::hint::black_box(&out);
            }
            let ns = t.elapsed().as_nanos() as f64 / iters as f64;
            best_ns = best_ns.min(ns);
        }
        #[cfg(not(debug_assertions))]
        {
            println!("G2 sample_conc_into N=8: best-of-5 rounds = {best_ns:.0} ns/call (bar 1000)");
            assert!(
                best_ns < 1000.0,
                "G2 FAIL: sample_conc_into N=8 best-of-5 rounds = {best_ns:.0} ns/call (bar 1000)"
            );
        }
        #[cfg(debug_assertions)]
        let _ = best_ns; // debug profile: correctness only, no latency bar
    }

    // ── T3: DirichletEma ──

    /// The mean path IS a plain recursive EMA at the frozen op order
    /// (scale-all then add) — bit-identical to a naive EMA written the
    /// same way. And pre-wrap, the ring shares equal `h` (the same raw
    /// weighted sum; ≤1e-6 f32 accumulation drift from the different op
    /// order) — pin (a).
    #[test]
    fn t3_mean_path_bit_identity_and_shares_pre_wrap() {
        let beta = 0.9f32;
        let history = [0usize, 1, 2, 0, 0, 1, 2, 2, 0, 1, 1, 0];
        let mut ema: DirichletEma<8, 3> = DirichletEma::new(beta);
        let mut naive = [0.0f32; 3];
        for &cls in &history {
            ema.push(cls);
            for v in naive.iter_mut() {
                *v *= beta;
            }
            naive[cls] += 1.0 - beta;
            assert!(
                bits_eq(ema.mean(), &naive),
                "h must be bit-identical to the plain EMA at the same op order"
            );
        }
        // Pre-wrap (N = 12 ≤ M = 8? no — 12 > 8, so wrap; use a shorter
        // history for the shares == h half of the pin).
        let mut ema2: DirichletEma<8, 3> = DirichletEma::new(beta);
        let mut h2 = [0.0f32; 3];
        for &cls in &history[..6] {
            ema2.push(cls);
            for v in h2.iter_mut() {
                *v *= beta;
            }
            h2[cls] += 1.0 - beta;
        }
        let shares = ema2.shares();
        // shares are NORMALIZED (sum 1); h sums to 1−β^6 — normalize h the
        // same way and compare.
        let h_sum: f32 = h2.iter().sum();
        for i in 0..3 {
            assert!(
                (shares[i] - h2[i] / h_sum).abs() < 1e-6,
                "pre-wrap shares[{i}] = {} vs h = {}",
                shares[i],
                h2[i] / h_sum
            );
        }
    }

    /// Pin (d) — the truncation is REAL: post-wrap the drawn path's
    /// expectation is the RING closed form (normalized shares), NOT `h`,
    /// which retains the infinite-EMA tail beyond the ring. The two views
    /// diverge by the truncated mass, so a consistency test of drawn
    /// against the recursive path fails by design.
    #[test]
    fn t3_drawn_expectation_is_ring_closed_form_not_recursive_h() {
        let beta = 0.9f32;
        let m = 8usize;
        // Seeded wrapped history: N = 24 pushes, class = t % 3 (biased by
        // phase so the ring and the tail differ materially).
        let n_pushes = 3 * m;
        let mut ema: DirichletEma<8, 3> = DirichletEma::new(beta);
        for t in 0..n_pushes {
            ema.push(t % 3);
        }
        let shares = ema.shares();
        let h = *ema.mean();
        // E[drawn] == shares (ε-independent; 8000 seeded draws, tolerance
        // ~5σ of the Dir(4·s) sampling noise).
        let eps = 4.0f32;
        let n_draws = 8_000usize;
        let mut acc = [0.0f32; 3];
        for t in 0..n_draws {
            let mut w = [0.0f32; 3];
            ema.drawn_weights(eps, 9000 + t as u64, &mut w);
            for i in 0..3 {
                acc[i] += w[i];
            }
        }
        let nf = n_draws as f32;
        for i in 0..3 {
            assert!(
                (acc[i] / nf - shares[i]).abs() < 0.01,
                "E[drawn][{i}] = {} vs ring shares = {}",
                acc[i] / nf,
                shares[i]
            );
        }
        // …and E[drawn] is NOT h (the L_{i,M} residual tail). h also sums
        // to 1−β^24 ≈ 0.92 while shares sum to 1.
        let divergence: f32 = (0..3).map(|i| (shares[i] - h[i]).abs()).sum();
        assert!(
            divergence > 0.05,
            "drawn path should follow the ring closed form, not the recursive h (divergence {divergence})"
        );
        // h equals the RAW infinite-EMA weighted sum of the FULL history
        // to within f32 accumulation drift (1e-5).
        let mut raw = [0.0f32; 3];
        let mut w = 1.0 - beta;
        for t in (0..n_pushes).rev() {
            raw[t % 3] += w;
            w *= beta;
        }
        for i in 0..3 {
            assert!(
                (h[i] - raw[i]).abs() < 1e-5,
                "h[{i}] = {} vs raw weighted sum = {}",
                h[i],
                raw[i]
            );
        }
    }

    /// Pin (b): `Var[L_j] = ε·share_j(1−share_j)/(ε+1)` on a wrapped ring.
    #[test]
    fn t3_drawn_variance_formula() {
        let beta = 0.9f32;
        let mut ema: DirichletEma<8, 3> = DirichletEma::new(beta);
        for t in 0..24 {
            ema.push(t % 3);
        }
        let shares = ema.shares();
        let eps = 4.0f32;
        // Class-wise concentration C = ε·Σw_raw = ε(1−β^filled) — the raw
        // geometric weights do NOT sum to 1 (the truncation), so C < ε.
        let raw_mass: f32 = (0..8_i32).map(|j| (1.0 - beta) * beta.powi(j)).sum();
        let conc = eps * raw_mass;
        let n_draws = 4_000usize;
        let mut acc = [0.0f32; 3];
        let mut acc_sq = [0.0f32; 3];
        for t in 0..n_draws {
            let mut w = [0.0f32; 3];
            ema.drawn_weights(eps, 11_000 + t as u64, &mut w);
            for i in 0..3 {
                acc[i] += w[i];
                acc_sq[i] += w[i] * w[i];
            }
        }
        let nf = n_draws as f32;
        for i in 0..3 {
            let mean = acc[i] / nf;
            let var = acc_sq[i] / nf - mean * mean;
            let expected = shares[i] * (1.0 - shares[i]) / (conc + 1.0);
            assert!(
                (var - expected).abs() < 0.08 * expected.max(1e-3),
                "var[{i}] = {var} vs closed form {expected}"
            );
        }
    }

    /// Pins (c) + the ε→∞ law: ε→0 concentrates on vertices (commit
    /// frequency → 1); ε→∞ makes every draw ≈ the deterministic shares.
    #[test]
    fn t3_drawn_epsilon_extremes() {
        let beta = 0.9f32;
        let mut ema: DirichletEma<8, 3> = DirichletEma::new(beta);
        for t in 0..16 {
            ema.push(match t % 4 {
                0 | 1 => 0,
                2 => 1,
                _ => 2,
            });
        }
        let shares = ema.shares();
        // ε→0: resample-from-history — vertices.
        let n_draws = 2_000usize;
        let mut commits = 0usize;
        for t in 0..n_draws {
            let mut w = [0.0f32; 3];
            ema.drawn_weights(1e-3, 12_000 + t as u64, &mut w);
            if w.iter().copied().fold(0.0f32, f32::max) >= 0.999 {
                commits += 1;
            }
        }
        assert!(
            commits as f32 / n_draws as f32 > 0.99,
            "ε→0 commit frequency = {} (bar 0.99)",
            commits as f32 / n_draws as f32
        );
        // ε→∞: deterministic weights — every draw ≈ shares.
        for t in 0..200 {
            let mut w = [0.0f32; 3];
            ema.drawn_weights(1e6, 13_000 + t as u64, &mut w);
            for i in 0..3 {
                assert!(
                    (w[i] - shares[i]).abs() < 0.01,
                    "ε→∞ draw[{i}] = {} vs shares = {}",
                    w[i],
                    shares[i]
                );
            }
        }
    }

    // ── G4: allocation-free hot paths ──

    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn goat_g4_alloc_free() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};
        reset_alloc_stats();
        let p = [0.5f32, 0.3, 0.2];
        let mut out = [0.0f32; 3];
        sample_conc_into(&p, 2.0, 21, &mut out);
        let x = [0.4f32, 0.35, 0.25];
        let alpha = [0.8f32, 0.6, 0.4];
        let mut out2 = [0.0f32; 3];
        thinning_into(&x, &alpha, 0.5, 22, &mut out2);
        let mut ema: DirichletEma<8, 3> = DirichletEma::new(0.9);
        for cls in [0usize, 1, 2, 0, 0, 1, 2, 2, 0, 1] {
            ema.push(cls);
        }
        let _ = ema.mean();
        let _ = ema.shares();
        let mut w = [0.0f32; 3];
        ema.drawn_weights(4.0, 23, &mut w);
        let (count, _bytes) = get_alloc_stats();
        assert_eq!(
            count, 0,
            "hot paths must be allocation-free, got {count} allocs"
        );
    }
}
