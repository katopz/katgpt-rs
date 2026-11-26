//! DEC Wave Kernel — the ballistic (hyperbolic) twin of the heat kernel
//! (Issue 775, Research 554 — PC-ALM, arXiv:2605.31022).
//!
//! The shipped propagation family is entirely **parabolic**: `heat_kernel`
//! (Plan 359), the `sheaf_admm` z-diffusion (Plan 407), leaky
//! `evolve_belief` — credit/anomaly signals through a chain of L cells
//! reach usable amplitude in O(L²) ticks. This module ships the
//! **hyperbolic member**: a 1:1-interleaved primal-dual step
//! (`h ← h − η·δᵀ(λ + ρ·δh)` then `λ ← λ + α·δh` on the post-primal
//! residual) whose per-mode dynamics are a **damped wave** — dispersion
//! `μ±(k) ≈ exp(−½ρη_h k² ± i√(αη_h)k)`, group velocity `√(αη_h)`, reach
//! `T√(αη_h)` in T steps (paper Eq 23), with per-mode Jury stability
//! `ησ²(2ρ+α) < 4` and the α-independent annulus `|μ±| = √(1−ηρσ²)`
//! (paper C.6 — closed-form setters live in `katgpt_core::dual` when the
//! core re-export is available; this crate is dep-free so the constants
//! travel as `WaveParams` values the caller derives from those laws).
//!
//! # The step
//!
//! State: a primal rank-k cochain `h` and a dual rank-(k+1) cochain `λ`
//! (the dual lives on the constraint surface — edges for a rank-0 primal;
//! the constraint operator is the coboundary `δ`, literally a
//! first-difference operator with `σ(k) = 2 sin(k/2)`, `λ_max ≈ 4`).
//!
//! ```text
//! 1. r  = δh                       (pre-step residual)
//! 2. e  = λ + ρ·r                  (composite credit)
//! 3. h ← h − η·δᵀe                 (primal; δᵀe = δᵀλ + ρ·Δh)
//! 4. r' = δh                       (POST-primal residual — the interleave)
//! 5. λ ← λ + α·r'                  (dual accumulator)
//! ```
//!
//! # α = 0 is the incumbent diffusion step, bit-identical
//!
//! With `α = 0` the dual stays exactly zero and the primal reduces to
//! `h ← h − η·(ρ·δᵀδh)` — the heat/diffusion step at effective rate `ηρ`,
//! computed through the same operators in the same order (unit-pinned on
//! generic inputs; the only theoretical divergence is the ±0.0 corner
//! where `0.0 + (−0.0)` normalizes a negative zero).
//!
//! # Zero-alloc
//!
//! All intermediate storage lives in caller-owned [`WaveScratch`],
//! allocated once per complex and reused across steps
//! ([`wave_step_into`]). Uses the shipped `_into` DEC operators
//! ([`exterior_derivative_into`] / [`codifferential_into`]) verbatim.
//!
//! # Modelless
//!
//! Closed-form algebra over the shipped DEC substrate — no training, no
//! backprop. The dual `λ` is latent accumulator state (it never replaces
//! the raw plane: at convergence `h` returns to feasible values and all
//! correction lives in `λ` — the raw/latent sync discipline).
//!
//! # References
//!
//! - Issue 775 (this extraction), Research 554 (distillation + prior art).
//! - Plan 407 / R438 — `sheaf_admm` (the diffusion-mode sibling whose
//!   z-update this can replace).
//! - Plan 359 — `heat_kernel` (the parabolic family this is the twin of).
//!
//! [`exterior_derivative_into`]: crate::operators::exterior_derivative_into
//! [`codifferential_into`]: crate::operators::codifferential_into

use crate::hodge::{HodgeComponents, hodge_decompose};
use crate::operators::{codifferential_into, exterior_derivative_into};
use crate::types::{CellComplex, CochainField};

/// Rates for [`wave_step_into`]. Derive these from the closed-form laws
/// (`katgpt_core::dual::jury_eta_max` / `alpha_reach` / `budget_ticks`)
/// rather than hand-tuning: for path/difference operators
/// `λ_max ≈ 4`, so the paper's self-calibrated defaults are
/// `η = 1/4`, `ρ = 1`, `α = 1` at budget `T = 2L`.
#[derive(Clone, Copy, Debug)]
pub struct WaveParams {
    /// Primal rate `η_h` (Jury: `η ρ σ² < 2` per mode).
    pub eta: f32,
    /// Penalty `ρ` (sets the annulus radius `√(1 − ηρσ²)`).
    pub rho: f32,
    /// Dual rate `α` — the **phase knob**: group velocity `√(αη)`. `α = 0`
    /// is the incumbent diffusion step (bit-identical).
    pub alpha: f32,
}

impl WaveParams {
    /// The paper's self-calibrated constants for path/difference operators
    /// (`λ_max ≈ 4`): `η = 1/λ_max`, `ρ = 1`, `α = 1` — arrival at
    /// `t_infl = L/√(αη) = 2L` ticks.
    pub const fn self_calibrated() -> Self {
        Self {
            eta: 0.25,
            rho: 1.0,
            alpha: 1.0,
        }
    }
}

/// Caller-owned scratch for [`wave_step_into`] — allocate once per complex
/// (per rank/dim pair) and reuse.
#[derive(Clone, Debug)]
pub struct WaveScratch {
    /// Residual / composite-credit buffer at rank `k+1` (edge rank for a
    /// rank-0 primal). Reused for both the pre- and post-primal residual.
    pub residual: CochainField,
    /// Primal gradient buffer at rank `k`.
    pub grad: CochainField,
}

impl WaveScratch {
    /// Size the scratch for a primal cochain of rank `rank`, `dim`
    /// channels over `cx`. The dual must have rank `rank + 1` and the same
    /// `dim`.
    pub fn new(cx: &CellComplex, rank: u8, dim: usize) -> Self {
        Self {
            residual: CochainField::zeros(rank + 1, cx.n_cells(rank + 1), dim),
            grad: CochainField::zeros(rank, cx.n_cells(rank), dim),
        }
    }
}

/// One interleaved primal-dual wave step, in place (the real
/// implementation — shapes are `debug_assert`ed).
///
/// `h` is the primal rank-k cochain; `lambda` the dual rank-(k+1) cochain;
/// both mutated. See the module docs for the update order (the dual sees
/// the POST-primal residual — that interleave is what makes the
/// propagation ballistic instead of diffusive).
#[allow(clippy::too_many_arguments)]
pub fn wave_step_into(
    cx: &CellComplex,
    h: &mut CochainField,
    lambda: &mut CochainField,
    params: &WaveParams,
    scratch: &mut WaveScratch,
) {
    let k = h.rank;
    debug_assert!(
        k < crate::types::MAX_RANK,
        "wave_step_into: rank {k} has no dual surface"
    );
    debug_assert_eq!(lambda.rank, k + 1, "dual rank must be primal rank + 1");
    debug_assert_eq!(lambda.dim, h.dim, "dual dim must match primal dim");
    debug_assert_eq!(lambda.n_cells(), cx.n_cells(k + 1), "dual n_cells mismatch");
    debug_assert_eq!(h.n_cells(), cx.n_cells(k), "primal n_cells mismatch");
    debug_assert_eq!(scratch.residual.rank, k + 1);
    debug_assert_eq!(scratch.grad.rank, k);

    // 1–2. r = δh; e = λ + ρ·r (composite credit, in the residual buffer).
    exterior_derivative_into(cx, h, &mut scratch.residual);
    let total = lambda.data.len();
    for i in 0..total {
        scratch.residual.data[i] = lambda.data[i] + params.rho * scratch.residual.data[i];
    }
    // 3. h ← h − η·δᵀe.
    codifferential_into(cx, &scratch.residual, &mut scratch.grad);
    for i in 0..h.data.len() {
        h.data[i] -= params.eta * scratch.grad.data[i];
    }
    // 4–5. r' = δh (post-primal); λ ← λ + α·r'.
    exterior_derivative_into(cx, h, &mut scratch.residual);
    for i in 0..total {
        lambda.data[i] += params.alpha * scratch.residual.data[i];
    }
}

// ---------------------------------------------------------------------------
// T9 — Hodge triage of a residual flow (the paper's discussion, load-bearing)
// ---------------------------------------------------------------------------

/// Energy-dominant class of a residual flow, from its Hodge decomposition.
///
/// Dual ascent absorbs the **exact** component of a constraint violation
/// (it has a local cause — some node's primal can fix it). The **harmonic**
/// class is the part NO local dual can ever fix (no local cause — the
/// residual is topological: a cycle obligation, a global inconsistency).
/// The **coexact** class circulates — credit that loops without ever
/// reducing disagreement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResidualClass {
    /// Dominant energy in `im(d)` — a gradient/conservative residual.
    /// Dual ascent absorbs it: keep iterating.
    Exact = 0,
    /// Dominant energy in `ker(Δ)` — topological. No `α` absorbs it (the
    /// unit-pinned test `harmonic_residual_not_absorbed_by_any_alpha`):
    /// escalate/redesign, don't iterate.
    Harmonic = 1,
    /// Dominant energy in `im(δ)` — circulating. Detect and break the
    /// loop (a constraint cycle feeding itself).
    Coexact = 2,
}

/// Verdict for a repair-path consumer: what to DO with the residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TriageVerdict {
    /// Exact-dominant — the dual is making progress; keep iterating.
    KeepIterating = 0,
    /// Harmonic-dominant — no local fix exists; escalate (redesign the
    /// constraint set, or accept the topological residual).
    Escalate = 1,
    /// Coexact-dominant — credit is circulating; break the loop.
    BreakCirculation = 2,
}

/// Result of [`hodge_triage`]: the decomposition energies + the verdict.
#[derive(Clone, Copy, Debug)]
pub struct HodgeTriage {
    /// `‖exact‖²` — the locally-fixable share.
    pub exact_energy: f32,
    /// `‖harmonic‖²` — the topological share (no local fix).
    pub harmonic_energy: f32,
    /// `‖coexact‖²` — the circulating share.
    pub coexact_energy: f32,
    /// The energy-dominant class.
    pub class: ResidualClass,
    /// The repair-path verdict derived from `class`.
    pub verdict: TriageVerdict,
}

/// Classify a residual flow by its Hodge decomposition — the repair-path
/// triage helper (Issue 775 T9). Consumes [`hodge_decompose`] verbatim;
/// the verdict is the energy-dominant component. d≤3 caveat respected:
/// this is for zone graphs / belief regions, not high-dim shards (the
/// boundary is larger than the interior there — the curse-of-dimensionality
/// note in the workspace manifold rules).
///
/// Ties (two components within 1e-9 relative energy) resolve to the more
/// actionable class: Exact < Harmonic < Coexact ordering means the
/// "cheapest to act on first" wins ties — keep-iterating is preferred over
/// escalating, escalating over breaking.
pub fn hodge_triage(cx: &CellComplex, residual: &CochainField) -> HodgeTriage {
    let HodgeComponents {
        exact,
        harmonic,
        coexact,
    } = hodge_decompose(cx, residual);
    let energy = |f: &CochainField| -> f32 { f.data.iter().map(|x| x * x).sum() };
    let exact_energy = energy(&exact);
    let harmonic_energy = energy(&harmonic);
    let coexact_energy = energy(&coexact);

    // Energy-dominant class; ties resolve to the more actionable (lower
    // enum value) — Exact before Harmonic before Coexact.
    let (class, verdict) = if exact_energy >= harmonic_energy && exact_energy >= coexact_energy {
        (ResidualClass::Exact, TriageVerdict::KeepIterating)
    } else if harmonic_energy >= coexact_energy {
        (ResidualClass::Harmonic, TriageVerdict::Escalate)
    } else {
        (ResidualClass::Coexact, TriageVerdict::BreakCirculation)
    };

    HodgeTriage {
        exact_energy,
        harmonic_energy,
        coexact_energy,
        class,
        verdict,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{codifferential, exterior_derivative, graph_laplacian};

    /// Path graph with L vertices: edges (i, i+1).
    fn path_complex(l: usize) -> CellComplex {
        let edges: Vec<(usize, usize)> = (0..l - 1).map(|i| (i, i + 1)).collect();
        CellComplex::from_edges(l, &edges)
    }

    /// Cycle graph with N vertices (for the harmonic-residual test).
    fn cycle_complex(n: usize) -> CellComplex {
        let mut edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        edges.push((n - 1, 0));
        CellComplex::from_edges(n, &edges)
    }

    #[test]
    fn alpha_zero_is_bit_identical_to_diffusion_step() {
        // wave(α=0) ≡ h − η·(ρ·δᵀδh), same operators, same order. Generic
        // input (no equal adjacent values ⇒ no ±0.0 corners — module docs).
        let cx = path_complex(24);
        let mut h_wave = CochainField::zeros(0, cx.n_vertices(), 1);
        let mut h_ref = CochainField::zeros(0, cx.n_vertices(), 1);
        for (i, v) in h_wave.data.iter_mut().enumerate() {
            *v = (i as f32) * 0.37 + 0.11;
            h_ref.data[i] = *v;
        }
        let mut lambda = CochainField::zeros(1, cx.n_edges(), 1);
        let mut scratch = WaveScratch::new(&cx, 0, 1);
        let params = WaveParams {
            eta: 0.25,
            rho: 1.3,
            alpha: 0.0,
        };
        for _ in 0..10 {
            wave_step_into(&cx, &mut h_wave, &mut lambda, &params, &mut scratch);
            // Manual incumbent: h ← h − η·(ρ·δᵀδh), same op order.
            let r = exterior_derivative(&cx, &h_ref);
            let mut scaled = r.clone();
            for k in 0..scaled.data.len() {
                scaled.data[k] = params.rho * r.data[k];
            }
            let g = codifferential(&cx, &scaled);
            for i in 0..h_ref.data.len() {
                h_ref.data[i] -= params.eta * g.data[i];
            }
        }
        assert_eq!(h_wave.data, h_ref.data, "α=0 must be bit-identical");
        // And the dual never left zero:
        assert!(lambda.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn wave_step_reduces_to_graph_laplacian_semantically() {
        // δᵀδh == L h for rank-0 (the DEC identity) — tolerance check that
        // the gradient is the Laplacian (the α=0 path IS diffusion).
        let cx = path_complex(16);
        let mut h = CochainField::zeros(0, cx.n_vertices(), 1);
        for (i, v) in h.data.iter_mut().enumerate() {
            *v = (i as f32) * 0.5 - 3.0;
        }
        let r = exterior_derivative(&cx, &h);
        let g = codifferential(&cx, &r);
        let lap = graph_laplacian(&cx, &h);
        for i in 0..h.data.len() {
            assert!(
                (g.data[i] - lap.data[i]).abs() < 1e-5,
                "idx {i}: δᵀδ={} L={}",
                g.data[i],
                lap.data[i]
            );
        }
    }

    #[test]
    fn annulus_law_measured_on_path_modes() {
        // Seed one cosine mode; the (h, λ) state energy in that mode decays
        // at the annulus radius² per step — α-INDEPENDENT. Amplitude ratios
        // oscillate under a complex pair (magnitude r^t times an
        // oscillating factor), so the measurement uses windowed energy
        // SUMS: Σ‖state‖² over a window ∝ r^{2·t_mid} — the oscillating
        // factor averages out over windows ≫ the period.
        let n = 64usize;
        let cx = path_complex(n);
        let (eta, rho) = (0.25f32, 1.0f32);
        let k_mode = 3usize;
        // Path-graph Laplacian eigenpair: v_j = cos(kπ(j+½)/n),
        // σ² = 2(1 − cos(kπ/n)).
        let sigma_sq = 2.0 * (1.0 - (std::f32::consts::PI * k_mode as f32 / n as f32).cos());
        let radius_sq = 1.0 - eta * rho * sigma_sq; // δ = r²

        // Vertex mode v and its edge image δv (normalized) — the dual
        // projects onto that image.
        let mut v_mode = CochainField::zeros(0, n, 1);
        for (j, slot) in v_mode.data[..n].iter_mut().enumerate() {
            *slot = (std::f32::consts::PI * k_mode as f32 * (j as f32 + 0.5) / n as f32).cos();
        }
        let v_norm = v_mode.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in v_mode.data.iter_mut() {
            *v /= v_norm;
        }
        let mut w_mode = exterior_derivative(&cx, &v_mode);
        let w_norm = w_mode.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in w_mode.data.iter_mut() {
            *v /= w_norm;
        }

        for &alpha in &[0.7f32, 1.4] {
            let params = WaveParams { eta, rho, alpha };

            let mode_energy = |h: &CochainField, lambda: &CochainField| -> f32 {
                let x: f32 = h.data.iter().zip(&v_mode.data).map(|(&a, &b)| a * b).sum();
                let l: f32 = lambda
                    .data
                    .iter()
                    .zip(&w_mode.data)
                    .map(|(&a, &b)| a * b)
                    .sum();
                x * x + l * l
            };

            // Two adjacent 240-step windows after a 120-step transient;
            // midpoint separation 240 ⇒ energy ratio = radius_sq^240.
            let (burn, win) = (120usize, 240usize);
            let mut h = v_mode.clone();
            let mut lambda = CochainField::zeros(1, cx.n_edges(), 1);
            let mut scratch = WaveScratch::new(&cx, 0, 1);
            for _ in 0..burn {
                wave_step_into(&cx, &mut h, &mut lambda, &params, &mut scratch);
            }
            let mut e_win1 = 0.0f64;
            for _ in 0..win {
                wave_step_into(&cx, &mut h, &mut lambda, &params, &mut scratch);
                e_win1 += mode_energy(&h, &lambda) as f64;
            }
            let mut e_win2 = 0.0f64;
            for _ in 0..win {
                wave_step_into(&cx, &mut h, &mut lambda, &params, &mut scratch);
                e_win2 += mode_energy(&h, &lambda) as f64;
            }
            let measured_ratio = e_win2 / e_win1.max(1e-300);
            let predicted = (radius_sq as f64).powi(win as i32);
            let rel = ((measured_ratio - predicted) / predicted).abs();
            assert!(
                rel < 0.08,
                "α={alpha}: measured energy ratio {measured_ratio:.6} vs annulus {predicted:.6} (rel {rel:.4})"
            );
        }
    }

    #[test]
    fn jury_instability_measured_dynamically() {
        // An unstable configuration (η σ²(2ρ+α) > 4) must blow up — the
        // classifier's Unstable verdict validated on the real kernel.
        // 30 steps: |μ−| ≈ 5.6 ⇒ growth ≈ 5.6³⁰ ≈ 6e21 — unambiguous, and
        // still inside f32 range (no inf → NaN contamination).
        let n = 32usize;
        let cx = path_complex(n);
        let params = WaveParams {
            eta: 0.9, // σ²_max ≈ 3.99 ⇒ ησ²(2ρ+α) ≈ 10.8 > 4
            rho: 1.0,
            alpha: 1.0,
        };
        let mut h = CochainField::zeros(0, n, 1);
        for (i, v) in h.data.iter_mut().enumerate() {
            // Ramp + alternating seed (the top mode is alternating-shaped).
            *v = (i as f32) * 0.01 - 0.15 + if i % 2 == 0 { 1e-3 } else { -1e-3 };
        }
        let mut lambda = CochainField::zeros(1, cx.n_edges(), 1);
        let mut scratch = WaveScratch::new(&cx, 0, 1);

        let norm0: f32 = h.data.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        for _ in 0..30 {
            wave_step_into(&cx, &mut h, &mut lambda, &params, &mut scratch);
        }
        // max-abs, NOT squared norm — the amplitudes themselves stay
        // inside f32 here, but their squares overflow (1e19² > f32 max).
        let norm1: f32 = h.data.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!(
            norm1 > 100.0 * norm0,
            "unstable config must grow: {norm0} → {norm1}"
        );
        assert!(norm1.is_finite());
    }

    #[test]
    fn ballistic_reach_beats_diffusion_on_path() {
        // The T6 GOAT gate in miniature (Bench 763 runs the full sweep):
        // inject at vertex 0, detect at the far end. Wave must arrive
        // strictly earlier than the α=0 diffusion twin at the same (η, ρ).
        let l = 64usize;
        let cx = path_complex(l);
        let threshold = 1e-3f32;
        let cap = 40 * l;

        let run = |alpha: f32| -> Option<usize> {
            let mut h = CochainField::zeros(0, l, 1);
            h.data[0] = 1.0;
            let mut lambda = CochainField::zeros(1, cx.n_edges(), 1);
            let mut scratch = WaveScratch::new(&cx, 0, 1);
            let params = WaveParams {
                eta: 0.25,
                rho: 1.0,
                alpha,
            };
            for t in 0..cap {
                wave_step_into(&cx, &mut h, &mut lambda, &params, &mut scratch);
                if h.data[l - 1].abs() >= threshold {
                    return Some(t + 1);
                }
            }
            None
        };

        let wave = run(1.0).expect("wave must arrive within cap");
        let heat = run(0.0).expect("heat must arrive within cap (L=64)");
        assert!(
            wave < heat,
            "wave ({wave} ticks) must beat heat ({heat} ticks) at L={l}"
        );
    }

    // -------------------------------------------------------------------
    // T9 — hodge triage
    // -------------------------------------------------------------------

    #[test]
    fn harmonic_residual_not_absorbed_by_any_alpha() {
        // The all-ones edge field on a cycle is harmonic (ker Δ₁). With a
        // CONSTANT primal (δh = 0), wave steps seeded with it as the dual
        // never move anything: δᵀ(harmonic) = 0 ⇒ zero primal gradient,
        // and δh contributes no harmonic component back. No local fix
        // exists — for ANY α. The triage verdict: Escalate.
        let n = 12usize;
        let cx = cycle_complex(n);
        let mut h = CochainField::zeros(0, n, 1);
        h.data.fill(0.5);
        let h_before = h.data.clone();
        let mut lambda = CochainField::zeros(1, cx.n_edges(), 1);
        lambda.data.fill(1.0); // the harmonic seed
        let lambda_before = lambda.data.clone();
        let mut scratch = WaveScratch::new(&cx, 0, 1);

        for &alpha in &[0.5f32, 1.0, 2.0] {
            let params = WaveParams {
                eta: 0.25,
                rho: 1.0,
                alpha,
            };
            for _ in 0..50 {
                wave_step_into(&cx, &mut h, &mut lambda, &params, &mut scratch);
            }
            assert_eq!(h.data, h_before, "α={alpha}: harmonic dual must not move h");
        }
        // Bit-identical: nothing was absorbed, nothing grew.
        assert_eq!(lambda.data, lambda_before);
        // Triage says escalate:
        let triage = hodge_triage(&cx, &lambda);
        assert_eq!(triage.class, ResidualClass::Harmonic);
        assert_eq!(triage.verdict, TriageVerdict::Escalate);
    }

    #[test]
    fn triage_exact_and_coexact_classes() {
        // A gradient (exact) residual: δ of a scalar potential on a path.
        let n = 16usize;
        let cx = path_complex(n);
        let mut pot = CochainField::zeros(0, n, 1);
        for (i, v) in pot.data.iter_mut().enumerate() {
            *v = 0.3 * i as f32;
        }
        let grad_res = exterior_derivative(&cx, &pot);
        let t = hodge_triage(&cx, &grad_res);
        assert_eq!(t.class, ResidualClass::Exact);
        assert_eq!(t.verdict, TriageVerdict::KeepIterating);

        // A coexact residual: δ of a rank-2 field on a grid (im(δ₂)) —
        // use a 6×6 grid so rank-2 cells exist.
        let gx = CellComplex::grid_2d(6, 6);
        let mut face_field = CochainField::zeros(2, gx.n_faces(), 1);
        for (i, v) in face_field.data.iter_mut().enumerate() {
            *v = ((i * 7 + 3) % 11) as f32 * 0.1 - 0.5;
        }
        let coexact_res = codifferential(&gx, &face_field);
        let t2 = hodge_triage(&gx, &coexact_res);
        assert_eq!(t2.class, ResidualClass::Coexact);
        assert_eq!(t2.verdict, TriageVerdict::BreakCirculation);
    }
}
