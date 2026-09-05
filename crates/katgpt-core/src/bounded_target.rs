//! Issue 695 — bounded-target correction primitives (Research 432 /
//! riir-train, arXiv:2608.24646 "DiffusionOPSD", Zhou et al. 2026).
//!
//! The OPSD recipe's defensible modelless half, extracted as substrate:
//! (1) a **one-measurement SPSA direction** whose *normalized* form is unit
//! by construction, (2) **bounded ± pairs / corrections** whose step norm is
//! ε by a type-level contract, (3) an ε **ladder** line-search with an
//! honest monotone (no-interior-optimum) flag, and (4) a **scorer-vitality
//! canary** that detects a dead/noisy scorer before its output is trusted
//! by a self-evolve loop.
//!
//! # The primitive (all closed-form, modelless)
//!
//! ```text
//! Δ ∈ {−1,+1}^D          Rademacher, BLAKE3(seed)-derived (XOF bits)
//! dq = Q(x+δΔ) − Q(x−δΔ) one SPSA measurement pair
//! ĝ  = [dq/(2δ)]·Δ       SPSA gradient estimate — UNIFORM component magnitude
//! d̂  = ĝ/‖ĝ‖ = sign(dq)·Δ/√D   exact unit norm, no magnitude estimation
//! t± = x ± ε·d̂           bounded pair, ‖t±−x‖ = ε (d̂ unit by type)
//! Δstep = ε·d̂            BoundedCorrection — ‖Δstep‖ = ε ≤ ε, always
//! ```
//!
//! The load-bearing identity: with Rademacher Δ every component of ĝ shares
//! the same magnitude |dq|/(2δ), so the normalized estimate collapses to
//! `sign(dq)·Δ/√D` — **unit by construction** (‖·‖² = D·(1/D) = 1 exactly in
//! the math; float rounding only). No "pure-sign fallback" is needed as a
//! separate mode: the pure-sign form *is* the normalized SPSA estimate.
//! In the linearized model `d̂·∇Q = |Δ·∇Q| ≥ 0` always — a (weak) guaranteed
//! **ascent** direction. Negate for descent.
//!
//! # Design notes (honest deviations from the issue sketch)
//!
//! - **`Option` = indeterminate, floor-parameterized.** The issue's
//!   `spsa_direction(q, x, delta, seed)` signature carries no noise scale,
//!   so the default floor is `0.0` (only an exactly-flat or non-finite
//!   response is indeterminate — the flat-Q negative control). σ-aware
//!   callers use `spsa_direction_with_floor(.., noise_floor)` with the
//!   paper's 2σ guard.
//! - **Ladder minimizes Q.** The OPSD target is a loss; the ladder picks
//!   the scale ∈ {ε/4, ε/2, ε, 2ε, 4ε} with the *lowest* Q along +d̂.
//!   `monotone` fires only when the largest scale is *strictly* best (no
//!   interior optimum inside the probed range — tie goes to the smaller
//!   scale, so the flag is conservative). `any_improvement == false` means
//!   the direction/ε pair does not descend at any probed scale (caller
//!   sign or scale choice is wrong).
//! - **`[f32; 64]` cap.** `MAX_D = 64` keeps every path on the stack
//!   (zero-alloc by construction — no `Vec`/`Box`/`String` in this
//!   module); the Rademacher byte buffer is a fixed 8 bytes.
//! - **c in `rho_hat` is landscape-dependent** (see `realization_gap`).
//!
//! # Domain classification
//!
//! Latent, local, never synced: directions/bounds are per-caller belief
//! state; the outputs are scalars + fixed arrays. No sync dependency, no
//! replay coupling, no chain surface.
//!
//! Feature: `bounded_target` (opt-in POC). Named consumers (riir-train
//! Plan 360 T3.1, riir-clippy score-bench promised-vs-realized axis,
//! riir-ai self-adaptive-loop ρ) file consumer-side at adoption time.

use blake3::Hasher;

/// Maximum supported dimension. The fixed-cap G4 contract: every path in
/// this module is stack-only; the Rademacher buffer is sized from this.
pub const MAX_D: usize = 64;

const RAD_BYTES: usize = 8; // ceil(MAX_D / 8)

/// A unit-norm direction vector. Unit-ness is guaranteed by construction:
/// the SPSA path builds `sign(dq)·Δ/√D` (exact unit in the math); the
/// adopt path validates ‖v‖₂ ≈ 1 within a caller tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitDir<const D: usize>([f32; D]);

impl<const D: usize> UnitDir<D> {
    /// Adopt `v` as a unit direction, validating ‖v‖₂ ≈ 1 within `tol`.
    pub fn from_normalized(v: [f32; D], tol: f32) -> Option<Self> {
        if D == 0 || D > MAX_D {
            return None;
        }
        let n = l2_norm(&v);
        if n.is_finite() && (n - 1.0).abs() <= tol {
            Some(Self(v))
        } else {
            None
        }
    }

    pub fn as_array(&self) -> &[f32; D] {
        &self.0
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

/// Fill `out` with Rademacher ±1 draws from a BLAKE3 XOF keyed by
/// `(seed, width)` — deterministic, seed-reproducible, width-domain-separated.
fn rademacher_into(seed: u64, out: &mut [f32]) {
    debug_assert!(out.len() <= MAX_D, "MAX_D cap exceeded");
    let mut hasher = Hasher::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(&(out.len() as u32).to_le_bytes());
    let mut buf = [0u8; RAD_BYTES];
    let need = out.len().div_ceil(8);
    hasher.finalize_xof().fill(&mut buf[..need]);
    for (i, v) in out.iter_mut().enumerate() {
        let bit = (buf[i / 8] >> (i % 8)) & 1;
        *v = if bit == 1 { 1.0 } else { -1.0 };
    }
}

/// One-measurement SPSA direction with the default indistinguishability
/// floor (`0.0`): only an exactly-flat or non-finite scorer response is
/// indeterminate. σ-aware callers want [`spsa_direction_with_floor`].
///
/// Returns `None` (indeterminate) when the measurement pair shows no
/// distinguishable signal: |ΔQ| ≤ `noise_floor`, a non-finite ΔQ, or a
/// non-finite/non-positive `delta`.
pub fn spsa_direction<const D: usize>(
    q: impl Fn(&[f32]) -> f32,
    x: &[f32; D],
    delta: f32,
    seed: u64,
) -> Option<UnitDir<D>> {
    spsa_direction_with_floor(q, x, delta, seed, 0.0)
}

/// σ-aware SPSA direction: the paper's 2σ guard as the indeterminacy floor.
/// A scorer noisier than its own signal cannot license a direction.
pub fn spsa_direction_with_floor<const D: usize>(
    q: impl Fn(&[f32]) -> f32,
    x: &[f32; D],
    delta: f32,
    seed: u64,
    noise_floor: f32,
) -> Option<UnitDir<D>> {
    if !delta.is_finite() || delta <= 0.0 || !noise_floor.is_finite() || noise_floor < 0.0 {
        return None;
    }
    let mut delta_vec = [0.0f32; D];
    rademacher_into(seed, &mut delta_vec);

    let mut xp = [0.0f32; D];
    let mut xm = [0.0f32; D];
    for i in 0..D {
        let d = delta * delta_vec[i];
        xp[i] = x[i] + d;
        xm[i] = x[i] - d;
    }
    let dq = q(&xp) - q(&xm);
    if !dq.is_finite() || dq.abs() <= noise_floor {
        return None;
    }
    // Normalized SPSA estimate: all components share magnitude |dq|/(2δ)
    // (Rademacher Δ ⇒ |Δᵢ| = 1), so ĝ/‖ĝ‖ = sign(dq)·Δ/√D — unit by
    // construction. Ascending on Q in the linearized model; negate for
    // descent.
    let s = dq.signum() / (D as f32).sqrt();
    let mut dir = [0.0f32; D];
    for i in 0..D {
        dir[i] = s * delta_vec[i];
    }
    Some(UnitDir(dir))
}

/// The bounded pair `t± = x ± ε·d̂` with `‖t± − x‖₂ = ε` exact in the math
/// (d̂ is unit by type; float rounding only).
pub fn bounded_pair<const D: usize>(
    x: &[f32; D],
    dir: &UnitDir<D>,
    eps: f32,
) -> ([f32; D], [f32; D]) {
    let mut t_plus = *x;
    let mut t_minus = *x;
    for i in 0..D {
        let step = eps * dir.as_array()[i];
        t_plus[i] += step;
        t_minus[i] -= step;
    }
    (t_plus, t_minus)
}

/// Ladder scales, in probe order: {ε/4, ε/2, ε, 2ε, 4ε}.
pub const LADDER_SCALES: [f32; 5] = [0.25, 0.5, 1.0, 2.0, 4.0];

/// ε-ladder outcome: the best (lowest-Q) probed scale along `+d̂`.
#[derive(Clone, Copy, Debug)]
pub struct LadderOutcome {
    /// Index into [`LADDER_SCALES`] of the best probe (first strict minimum
    /// wins ties — smaller scales are preferred, keeping `monotone`
    /// conservative).
    pub best_idx: usize,
    /// Absolute best step size (`eps * LADDER_SCALES[best_idx]`).
    pub best_scale: f32,
    /// Q at the best probe (f32::INFINITY when no probe was finite).
    pub best_q: f32,
    /// Q(x) reference.
    pub q0: f32,
    /// Best probe is the LARGEST scale, strictly — no interior optimum
    /// inside the probed range; the ladder is exhausted in this direction.
    pub monotone: bool,
    /// Some probe beat Q(x). `false` ⇒ the direction/ε pair does not
    /// descend at any probed scale (check the sign / shrink ε).
    pub any_improvement: bool,
}

/// 5-eval line search along `+d̂`: Q at `x + s·ε·d̂` for s ∈
/// [`LADDER_SCALES`]. **Minimizes Q** (the OPSD target is a loss).
pub fn eps_ladder<const D: usize>(
    q: impl Fn(&[f32]) -> f32,
    x: &[f32; D],
    dir: &UnitDir<D>,
    eps: f32,
) -> LadderOutcome {
    let q0 = q(x.as_slice());
    let mut probe = [0.0f32; D];
    let mut best_idx = 0usize;
    let mut best_q = f32::INFINITY;
    let mut any_finite = false;
    for (idx, &s) in LADDER_SCALES.iter().enumerate() {
        for i in 0..D {
            probe[i] = x[i] + (eps * s) * dir.as_array()[i];
        }
        let v = q(probe.as_slice());
        if v.is_finite() && (!any_finite || v < best_q) {
            best_q = v;
            best_idx = idx;
            any_finite = true;
        }
    }
    LadderOutcome {
        best_idx,
        best_scale: eps * LADDER_SCALES[best_idx],
        best_q,
        q0,
        monotone: any_finite && best_idx == LADDER_SCALES.len() - 1,
        any_improvement: any_finite && best_q < q0,
    }
}

/// A bounded correction `Δ = ε·d̂`. The type-level contract: `dir` is unit
/// by construction and `eps` is clamped ≥ 0, so **no method on this type
/// can produce a step with norm > `eps`** — the bounded-target discipline
/// (the latent→raw clamp pattern, applied to step size).
#[derive(Clone, Copy, Debug)]
pub struct BoundedCorrection<const D: usize> {
    dir: UnitDir<D>,
    eps: f32,
}

impl<const D: usize> BoundedCorrection<D> {
    pub fn new(dir: UnitDir<D>, eps: f32) -> Self {
        Self {
            dir,
            eps: eps.max(0.0),
        }
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn dir(&self) -> &UnitDir<D> {
        &self.dir
    }

    /// The bounded step `ε·d̂` — `‖Δ‖₂ = ε` (d̂ unit by type).
    pub fn delta(&self) -> [f32; D] {
        let mut d = [0.0f32; D];
        for (di, &dir_i) in d.iter_mut().zip(self.dir.as_array()) {
            *di = self.eps * dir_i;
        }
        d
    }

    /// `x + Δ` — apply the bounded correction.
    pub fn apply(&self, x: &[f32; D]) -> [f32; D] {
        let mut out = *x;
        for (o, di) in out.iter_mut().zip(self.delta()) {
            *o += di;
        }
        out
    }
}

/// Canary tolerance: the vitality floor is `2ε·g_min·(1 − CANARY_TOL)`.
pub const CANARY_TOL: f32 = 0.10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Canary {
    /// Measured contrast meets the floor implied by `g_min`.
    Alive,
    /// Measured contrast below the floor: the scorer is dead or noisier
    /// than the signal it must carry. Do NOT trust its directions.
    Dead,
}

/// Scorer-vitality canary. On a fixture whose true directional derivative
/// satisfies `|∇Q·d̂| ≥ g_min`, a healthy scorer shows
/// `|Q(x+εd̂) − Q(x−εd̂)| ≥ 2ε·g_min·(1 − CANARY_TOL)`. Below that, the
/// direction estimate from this scorer is noise — flag it before a
/// self-evolve loop consumes it.
pub fn contrast<const D: usize>(
    q: impl Fn(&[f32]) -> f32,
    x: &[f32; D],
    dir: &UnitDir<D>,
    eps: f32,
    g_min: f32,
) -> Canary {
    let (t_plus, t_minus) = bounded_pair(x, dir, eps);
    let dq = q(t_plus.as_slice()) - q(t_minus.as_slice());
    let floor = 2.0 * eps * g_min * (1.0 - CANARY_TOL);
    if dq.is_finite() && dq.abs() >= floor {
        Canary::Alive
    } else {
        Canary::Dead
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn g1_determinism_same_seed_bit_identical() {
        let x = [0.25f32; 16];
        let q = |v: &[f32]| v.iter().map(|t| t * t).sum::<f32>();
        for seed in [0u64, 1, 42, u64::MAX] {
            let a = spsa_direction(q, &x, 1e-2, seed).unwrap();
            let b = spsa_direction(q, &x, 1e-2, seed).unwrap();
            assert_eq!(a, b, "seed {seed} must be bit-identical across calls");
        }
    }

    #[test]
    fn g1_rademacher_population_varies_by_seed() {
        // Different seeds must produce genuinely different directions
        // (an XOF stuck at one output would silently collapse exploration).
        let x = [0.0f32; 8];
        let q = |v: &[f32]| v.iter().sum::<f32>();
        let a = spsa_direction(q, &x, 1e-2, 1).unwrap();
        let b = spsa_direction(q, &x, 1e-2, 2).unwrap();
        assert_ne!(a, b, "seeds 1 and 2 collided");
    }

    #[test]
    fn g1_ascent_guarantee_on_quadratic_bowl() {
        // Q(x) = ½‖x‖² ⇒ ∇Q = x. The normalized SPSA direction satisfies
        // d̂·∇Q = |Δ·∇Q| > 0 whenever Δ is not orthogonal to ∇Q — a
        // guaranteed (weak) ASCENT direction in the linearized model.
        let x = [0.5f32, -0.3, 0.8, -0.1, 0.2, -0.7, 0.4, 0.6];
        let q = |v: &[f32]| 0.5 * v.iter().map(|t| t * t).sum::<f32>();
        let mut ascending = 0usize;
        let mut informative = 0usize;
        for seed in 0..32u64 {
            // A draw exactly orthogonal to ∇Q in f32 carries no information
            // (dq = 0) — legitimately indeterminate, not an ascent violation.
            let Some(d) = spsa_direction(q, &x, 1e-3, seed) else {
                continue;
            };
            informative += 1;
            let ip = dot(d.as_slice(), &x);
            assert!(
                ip >= 0.0,
                "seed {seed}: d̂·∇Q = {ip} < 0 — ascent guarantee violated"
            );
            if ip > 0.0 {
                ascending += 1;
            }
        }
        assert!(informative >= 24, "only {informative}/32 seeds informative");
        assert!(
            ascending >= 24,
            "only {ascending}/{informative} non-orthogonal"
        );
    }

    #[test]
    fn g1_flat_scorer_is_indeterminate() {
        let x = [0.5f32; 16];
        let q = |_v: &[f32]| 7.0f32; // exactly flat: identical bits both sides
        assert!(spsa_direction(q, &x, 1e-2, 0).is_none());
        // σ-aware floor: sub-floor signal is indeterminate even when nonzero.
        let q2 = |v: &[f32]| if v[0] > 0.5 { 1.0 } else { 0.0 };
        assert!(spsa_direction_with_floor(q2, &x, 1e-2, 0, 10.0).is_none());
        // Non-finite / non-positive δ: refuse.
        assert!(spsa_direction(q, &x, 0.0, 0).is_none());
        assert!(spsa_direction(q, &x, f32::NAN, 0).is_none());
    }

    #[test]
    fn g1_unit_norm_by_construction() {
        let x = [0.1f32; 64];
        let q = |v: &[f32]| v.iter().sum::<f32>();
        let d = spsa_direction(q, &x, 1e-2, 7).unwrap();
        let n = l2_norm(d.as_slice());
        assert!((n - 1.0).abs() <= 1e-5, "‖d̂‖ = {n} at D=64 (rounding only)");
        // Every component is ±1/√D — the pure-sign shape.
        let inv = 1.0f32 / (64.0f32).sqrt();
        for &c in d.as_slice() {
            assert!(
                (c.abs() - inv).abs() <= 1e-6 * inv,
                "component {c} not ±1/√D"
            );
        }
    }

    #[test]
    fn g1_bounded_correction_norm_bound_bit_exact() {
        let x = [0.25f32; 32];
        let q = |v: &[f32]| v.iter().map(|t| t * t).sum::<f32>();
        let d = spsa_direction(q, &x, 1e-2, 3).unwrap();
        for &eps in &[0.001f32, 0.25, 1.0, 8.0] {
            let c = BoundedCorrection::new(d, eps);
            let step = c.delta();
            let n = l2_norm(&step);
            assert!(
                (n - eps).abs() <= 1e-5 * eps.max(1.0),
                "‖Δ‖ = {n} vs ε = {eps}"
            );
            // eps clamped ≥ 0 — a negative ask still cannot escape the bound.
            let neg = BoundedCorrection::new(d, -1.0);
            assert!(neg.eps() >= 0.0);
            assert_eq!(neg.delta(), [0.0f32; 32]);
        }
        let applied = BoundedCorrection::new(d, 0.5).apply(&x);
        let diff: Vec<f32> = applied.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        assert!((l2_norm(&diff) - 0.5).abs() <= 1e-5);
    }

    #[test]
    fn g1_eps_ladder_interior_optimum() {
        // 1-D quadratic Q(s) = (s − 1.5)² probed from x=0 along +d̂=+1 with
        // ε=1: probes at {0.25, 0.5, 1, 2, 4} → Q {1.5625, 1.0, 0.25, 0.25, 6.25}.
        // First strict min at idx 2 (scale ε): interior optimum, not monotone.
        let x = [0.0f32];
        let dir = UnitDir::from_normalized([1.0f32], 1e-3).unwrap();
        let q = |v: &[f32]| (v[0] - 1.5f32).powi(2);
        let out = eps_ladder(q, &x, &dir, 1.0);
        assert_eq!(out.best_idx, 2, "first strict min at scale ε");
        assert!((out.best_scale - 1.0).abs() < 1e-6);
        assert!((out.best_q - 0.25).abs() < 1e-6);
        assert!(!out.monotone, "interior optimum must not flag monotone");
        assert!(out.any_improvement, "0.25 < Q(0) = 2.25");
    }

    #[test]
    fn g1_eps_ladder_monotone_negative_control() {
        // Monotone decreasing Q inside the probed range: best is STRICTLY
        // the largest scale → the ladder is exhausted, no interior optimum.
        let x = [0.0f32];
        let dir = UnitDir::from_normalized([1.0f32], 1e-3).unwrap();
        let q = |v: &[f32]| -v[0]; // Q strictly decreasing along +d̂
        let out = eps_ladder(q, &x, &dir, 1.0);
        assert_eq!(out.best_idx, 4);
        assert!(out.monotone, "strictly-decreasing Q must flag monotone");
        assert!(out.any_improvement);
        // And a direction that never improves: ascending Q along +d̂.
        let q_up = |v: &[f32]| v[0];
        let out2 = eps_ladder(q_up, &x, &dir, 1.0);
        assert!(!out2.any_improvement, "no probe beats Q(x)");
        assert!(!out2.monotone);
    }

    #[test]
    fn g1_canary_alive_and_dead() {
        let x = [0.0f32];
        let dir = UnitDir::from_normalized([1.0f32], 1e-3).unwrap();
        // Linear scorer, gradient 1 along d̂: |ΔQ| = 2ε exactly.
        let alive = |v: &[f32]| v[0];
        assert_eq!(contrast(alive, &x, &dir, 0.1, 1.0), Canary::Alive);
        // g_min above the true gradient: the floor exceeds the measurable
        // contrast → dead.
        assert_eq!(contrast(alive, &x, &dir, 0.1, 2.0), Canary::Dead);
        // Flat scorer: zero contrast → dead.
        let flat = |_v: &[f32]| 3.0f32;
        assert_eq!(contrast(flat, &x, &dir, 0.1, 0.5), Canary::Dead);
    }

    #[test]
    fn g4_max_d_cap_is_64_and_stack_only() {
        // Compile-time pin: MAX_D == 64 is load-bearing for the fixed buffers
        // below; a drift must fail the build, not a runtime assert.
        const _: () = assert!(MAX_D == 64);
        const { assert!(RAD_BYTES * 8 >= MAX_D); }
        // MAX_D-dim path exercises the fixed buffers end-to-end.
        let x = [0.2f32; MAX_D];
        let q = |v: &[f32]| v.iter().sum::<f32>();
        assert!(spsa_direction(q, &x, 1e-2, 11).is_some());
        assert!(UnitDir::from_normalized([1.0f32, 0.0], 1e-3).is_some());
        // D=0 is refused by the adopt path.
        assert!(UnitDir::<0>::from_normalized([], 1e-3).is_none());
    }

    #[cfg_attr(debug_assertions, ignore = "timing gate — release-only")]
    #[test]
    fn g2_spsa_direction_under_budget_at_d16() {
        // Budget: the issue's ~100 ns for the direction math INCLUDING the
        // one BLAKE3 XOF call (measured 97.6 ns/call @ d=16 on M3 Max under
        // ambient box load; 2× margin for load variance). The scorer is
        // caller-owned; a trivial linear one bounds its share.
        const BUDGET_NS: f64 = 200.0;
        const N: u64 = 10_000;

        let x = [0.1f32; 16];
        let q = |v: &[f32]| v.iter().sum::<f32>();
        let mut acc = 0.0f32;
        for seed in 0..256u64 {
            if let Some(d) = spsa_direction(q, &x, 1e-2, seed) {
                acc += d.as_slice()[0];
            }
        }
        assert!(acc != 0.0, "warmup sanity");
        let t0 = std::time::Instant::now();
        acc = 0.0;
        for seed in 0..N {
            if let Some(d) = spsa_direction(q, &x, 1e-2, seed) {
                acc += d.as_slice()[seed as usize & 15];
            }
        }
        let dt = t0.elapsed();
        let per = dt.as_nanos() as f64 / N as f64;
        std::eprintln!("g2 spsa_direction d=16: {per:.1} ns/call (acc {acc})");
        assert!(acc.is_finite());
        assert!(
            per <= BUDGET_NS,
            "{per:.1} ns/call > {BUDGET_NS} ns budget @ d=16"
        );
    }
}
