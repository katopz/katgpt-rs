//! LP-authored overlay arm — deterministic, no-LLM authoring for the
//! weight-edit lane (Issue 993; Research 664 — the weight-space-lp
//! distill).
//!
//! # The formulation (Research 664 §1, blog-quoted shape)
//!
//! Given a correction direction `u` (the layer's output width) and paired
//! labeled activation traces — push class `R` (the states the behavior
//! must move, by at least `δ`) and keep class `P` (the states the
//! behavior must barely move, within `ε`) — solve for per-column
//! coefficients `v`:
//!
//! ```text
//! min_v   Σ_j |v_j|  +  C · Σ_t slack_t
//! s.t.    v · h_t ≤ −δ    t ∈ R   (push the looping states)
//!         |v · h_t| ≤ ε   t ∈ P   (barely move the productive states)
//! ```
//!
//! L1 minimization makes `v` sparse — the LP picks the columns AND the
//! amounts in one solve. The composed edit is the rank-1 overlay
//! `ΔW = −u vᵀ`, emitted as one `EditForm::DirectionSubtraction` spec
//! per nonzero coefficient (single-column, ascending column order),
//! riding the EXISTING carrier ([`super`]) unchanged: at each step the
//! edit adds `−(v·h_t)·u` — a push along `u` that is LINEAR in `v`,
//! exactly the quantity the constraints pin.
//!
//! # The solver (T1)
//!
//! The slacks are eliminated (`slack_t* = relu((A v)_t − b_t)`), giving
//! the penalized form
//!
//! ```text
//! min_v  ‖v‖₁ + C · Σ_t relu((A v)_t − b_t)
//! ```
//!
//! with rows `A` (in [`TraceSet`] order — the caller pins the trace
//! order): push rows `(h_t, b = −δ)`, keep-upper rows `(h_t, b = +ε)`,
//! keep-lower rows `(−h_t, b = +ε)`. Solved by PDHG (Chambolle–Pock,
//! over-relaxation θ = 1 — the same first-order primal-dual family as
//! `cce/primal_dual.rs` (Plan 295), whose pinned schedule this module
//! reuses as the PATTERN, never the CCE-bound code):
//!
//! ```text
//! y⁺ = clamp(y + σ(A v̄ − b), 0, C)      # prox_{σg*} — a box projection
//! v⁺ = soft_threshold(v − τ Aᵀ y⁺, τ)   # prox_{τf} — L1 shrinkage
//! v̄⁺ = 2 v⁺ − v                          # θ = 1 reflection
//! ```
//!
//! with `τσ = 0.995²/L²` and `L` the closed-form bound
//! `sqrt(‖A‖₁·‖A‖_∞)` — a GUARANTEED upper bound on `‖A‖₂`, so the
//! convergence condition `τσ‖A‖² < 1` holds unconditionally (a
//! power-iteration estimate could UNDER-estimate the norm and break the
//! guarantee; the looser bound costs iterations, not correctness, and
//! authoring is offline). Iteration runs to step stability
//! (`‖Δv‖∞, ‖Δy‖∞ ≤ tol`) or `max_iter` — both deterministic functions
//! of the inputs. Non-convergence at the cap is NOT an error: the
//! solution is returned with its diagnostics in [`LpAuthoringSolution`]
//! and the CALLER's property gates judge it (authoring-time-only —
//! latency is not load-bearing).
//!
//! # Determinism law (Issue 940)
//!
//! **Cross-platform bit-identity holds iff the sequential-fold
//! discipline is followed; per-platform otherwise.** Every reduction in
//! this module is a sequential `acc += …` fold in a fixed index order —
//! matrix-vector products, the transpose accumulation, the norm bound.
//! No platform-lane SIMD kernel, no reassociation (Rust contracts no FMA
//! by default), no HashMap-order dependence anywhere. The returned `v`
//! is a pure function of (traces, config): two solves on the same pinned
//! input are bit-identical, so the composed artifact digest is a
//! committed, hash-comparable result. Pinned iteration counts alone
//! would NOT deliver this — the fold discipline does.
//!
//! # The loud refusals
//!
//! A `v = 0` return (the unconstrained / degenerate case — e.g. a trace
//! set with no push rows) is a LOUD
//! [`LpAuthoringError::DegenerateZeroSolution`], never a silent empty
//! overlay. A zero constraint matrix (all traces zero) refuses
//! [`LpAuthoringError::Malformed`]. Non-finite or shape-mismatched input
//! refuses loud.
//!
//! # Modelless + budget compliance
//!
//! No training, no gradient descent, no LLM anywhere: one convex
//! first-order solve over recorded activations. Zero new dependencies.
//! **Authoring-time-only** — no serving wiring, no real-model
//! evaluation, no promotion claim (the Bench-976 serving veto stands;
//! the 976 no-default-consumer verdict is unchanged). Allocations and
//! latency are not load-bearing at mint time.
//!
//! # The decision law (T3, recorded at the fixture)
//!
//! **Probe steering first; bake only on selectivity need or
//! injection-cost refusal** — the constant-push (steering) arm is probed
//! at {0.5, 1, 2}× the LP edit's average push BEFORE any weight edit is
//! baked; a steering recovery means the behavior does not need per-token
//! selectivity and the frozen-model quadrant (inference-time injection)
//! is the cheaper arm. Connects `latent_field_steering` (Plan 309) and
//! the `weight_edit_overlay` lane (Issue 980); corroborates 309's
//! α ≤ 0.3 narrow-window caveat — narrow viable strength bands are a
//! family trait.

use super::{EditForm, WeightEditOverlay, WeightEditSpec};

// ─────────────────────────────────────────────────────────────────────
// Public surface
// ─────────────────────────────────────────────────────────────────────

/// The authoring LP's pinned knobs. The defaults are the module's neutral
/// posture; a fixture/consumer re-pins `(delta_push, epsilon_keep,
/// c_slack)` from its own geometry and documents the derivation (the
/// fixture gate carries the worked example).
#[derive(Clone, Debug, PartialEq)]
pub struct LpAuthoringConfig {
    /// Push target `δ`: every push-class state must move by at least `δ`
    /// (the edit's push at state `t` is `−(v·h_t)·u`).
    pub delta_push: f32,
    /// Keep band `ε`: every keep-class state's movement is bounded by
    /// `|v·h_t| ≤ ε`. `0.0` is legal — the retention law as a constraint
    /// (the fixture's productive class is exactly carrier-silent; a
    /// real-model authoring pins `ε > 0` to the measured productive noise
    /// floor).
    pub epsilon_keep: f32,
    /// Slack penalty `C > 0` — the price of one unit of constraint
    /// violation (the push/keep constraints are soft; `C` scales how hard
    /// the solve refuses to violate).
    pub c_slack: f32,
    /// Iteration cap. Non-convergence at the cap is NOT an error: the
    /// solution is returned with its diagnostics and the caller's property
    /// gates judge it (authoring-time-only).
    pub max_iter: u32,
    /// Step-stability early-stop threshold on (`‖Δv‖∞`, `‖Δy‖∞`). `0.0`
    /// disables early stopping (runs to `max_iter`).
    pub tol: f32,
}

impl Default for LpAuthoringConfig {
    fn default() -> Self {
        Self {
            delta_push: 1.0,
            epsilon_keep: 0.1,
            c_slack: 10.0,
            max_iter: 20_000,
            tol: 1e-5,
        }
    }
}

/// One labeled activation-trace set: `n_rows` row-major `dim`-wide f32
/// rows (the recorded activations `h_t`). The ROW ORDER is load-bearing —
/// the caller pins it (seed order = step order), and the solve is a pure
/// function of the pinned bytes.
#[derive(Clone, Copy, Debug)]
pub struct TraceSet<'a> {
    /// Row-major trace rows (`n_rows × dim`).
    pub rows: &'a [f32],
    /// Number of trace rows.
    pub n_rows: usize,
    /// Trace width (the layer's input width = the coefficient dim).
    pub dim: usize,
}

impl<'a> TraceSet<'a> {
    /// Validate + wrap. Refuses loud on a shape mismatch, a zero dim, or
    /// any non-finite entry (tamper class — a NaN trace row would poison
    /// every fold after it).
    pub fn new(rows: &'a [f32], n_rows: usize, dim: usize) -> Result<Self, LpAuthoringError> {
        if dim == 0 {
            return Err(LpAuthoringError::Malformed("trace dim must be >= 1"));
        }
        let expect = n_rows
            .checked_mul(dim)
            .ok_or(LpAuthoringError::Malformed("trace shape overflows usize"))?;
        if rows.len() != expect {
            return Err(LpAuthoringError::Malformed(
                "trace rows length != n_rows × dim",
            ));
        }
        if rows.iter().any(|x| !x.is_finite()) {
            return Err(LpAuthoringError::Malformed("non-finite trace entry"));
        }
        Ok(Self { rows, n_rows, dim })
    }
}

/// Why authoring refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LpAuthoringError {
    /// Malformed input (shape, non-finite entries, invalid config, or a
    /// zero constraint matrix — no signal to solve against).
    Malformed(&'static str),
    /// The solve returned `v = 0` — the unconstrained/degenerate case. A
    /// LOUD refusal, never a silent empty overlay.
    DegenerateZeroSolution,
}

/// The authored coefficients + the solve's diagnostics. `v` is the raw
/// first-order iterate — exact zeros where shrinkage landed them, but
/// sub-floor dust may remain off the support; apply [`truncate_small`]
/// before composing.
#[derive(Clone, Debug)]
pub struct LpAuthoringSolution {
    /// The per-column coefficients (length = trace dim).
    pub v: Vec<f32>,
    /// Count of exact-nonzero entries.
    pub nonzeros: usize,
    /// Iterations run.
    pub iterations: u32,
    /// Final objective `‖v‖₁ + C·Σ relu((Av)_t − b_t)` (sequential fold).
    pub objective: f32,
    /// Final primal step size `‖Δv‖∞` (convergence diagnostic).
    pub final_primal_step: f32,
    /// Final dual step size `‖Δy‖∞` (convergence diagnostic).
    pub final_dual_step: f32,
}

/// Solve the min-L1 authoring LP for the paired trace sets. Deterministic:
/// same pinned input → bit-identical [`LpAuthoringSolution`].
pub fn solve_min_l1_authoring_lp(
    push: &TraceSet<'_>,
    keep: &TraceSet<'_>,
    cfg: &LpAuthoringConfig,
) -> Result<LpAuthoringSolution, LpAuthoringError> {
    if push.dim != keep.dim {
        return Err(LpAuthoringError::Malformed("push/keep trace dims differ"));
    }
    if !cfg.delta_push.is_finite() || cfg.delta_push < 0.0 {
        return Err(LpAuthoringError::Malformed(
            "delta_push must be finite >= 0",
        ));
    }
    if !cfg.epsilon_keep.is_finite() || cfg.epsilon_keep < 0.0 {
        return Err(LpAuthoringError::Malformed(
            "epsilon_keep must be finite >= 0",
        ));
    }
    if !cfg.c_slack.is_finite() || cfg.c_slack <= 0.0 {
        return Err(LpAuthoringError::Malformed("c_slack must be finite > 0"));
    }
    if cfg.max_iter == 0 {
        return Err(LpAuthoringError::Malformed("max_iter must be >= 1"));
    }
    if !cfg.tol.is_finite() || cfg.tol < 0.0 {
        return Err(LpAuthoringError::Malformed("tol must be finite >= 0"));
    }

    let dim = push.dim;
    let delta = cfg.delta_push;
    let eps = cfg.epsilon_keep;
    let c = cfg.c_slack;
    let m = push.n_rows + 2 * keep.n_rows;

    let l = operator_norm_bound(push, keep)?;
    let step = 0.995 / l;
    let (sigma, tau) = (step, step);

    let mut v = vec![0.0f32; dim];
    let mut v_bar = vec![0.0f32; dim];
    let mut v_new = vec![0.0f32; dim];
    let mut grad = vec![0.0f32; dim];
    let mut y = vec![0.0f32; m];
    let mut y_new = vec![0.0f32; m];

    let mut iterations = 0u32;
    let mut primal_step = 0.0f32;
    let mut dual_step = 0.0f32;

    for k in 1..=cfg.max_iter {
        iterations = k;

        // ── Dual: y⁺ = clamp(y + σ(A v̄ − b), 0, C) ──
        dual_step = 0.0;
        for_each_row(push, keep, delta, eps, |t, h, sign, b| {
            // Sequential fold (Issue-940 order pin): h_t · v̄.
            let mut acc = 0.0f32;
            for (w, x) in h.iter().zip(v_bar.iter()) {
                acc += w * x;
            }
            let y_up = y[t] + sigma * (sign * acc - b);
            let y_next = y_up.clamp(0.0, c);
            let d = (y_next - y[t]).abs();
            if d > dual_step {
                dual_step = d;
            }
            y_new[t] = y_next;
        });
        std::mem::swap(&mut y, &mut y_new);

        // ── Primal: v⁺ = soft(v − τ Aᵀ y⁺, τ) ──
        grad.fill(0.0);
        for_each_row(push, keep, delta, eps, |t, h, sign, _b| {
            let s = sign * y[t];
            for (g, w) in grad.iter_mut().zip(h.iter()) {
                *g += s * w;
            }
        });

        primal_step = 0.0;
        for j in 0..dim {
            let vj = v[j];
            let vn = soft_threshold(vj - tau * grad[j], tau);
            let d = (vn - vj).abs();
            if d > primal_step {
                primal_step = d;
            }
            v_new[j] = vn;
            v_bar[j] = 2.0 * vn - vj;
        }
        std::mem::swap(&mut v, &mut v_new);

        if cfg.tol > 0.0 && primal_step <= cfg.tol && dual_step <= cfg.tol {
            break;
        }
    }

    let nonzeros = v.iter().filter(|&&x| x != 0.0).count();
    if nonzeros == 0 {
        return Err(LpAuthoringError::DegenerateZeroSolution);
    }

    // Objective report — sequential folds (the committed diagnostic).
    let mut l1 = 0.0f32;
    for &x in v.iter() {
        l1 += x.abs();
    }
    let mut pen = 0.0f32;
    for_each_row(push, keep, delta, eps, |_t, h, sign, b| {
        let mut acc = 0.0f32;
        for (w, x) in h.iter().zip(v.iter()) {
            acc += w * x;
        }
        let viol = sign * acc - b;
        if viol > 0.0 {
            pen += viol;
        }
    });

    Ok(LpAuthoringSolution {
        v,
        nonzeros,
        iterations,
        objective: l1 + c * pen,
        final_primal_step: primal_step,
        final_dual_step: dual_step,
    })
}

// ─────────────────────────────────────────────────────────────────────
// Internals
// ─────────────────────────────────────────────────────────────────────

/// Enumerate the constraint rows in the pinned order, handing each row's
/// slice, sign and bias to `f`: push rows `(h, +1, −δ)`, keep-upper rows
/// `(h, +1, +ε)`, keep-lower rows `(h, −1, +ε)`.
fn for_each_row<F>(push: &TraceSet<'_>, keep: &TraceSet<'_>, delta: f32, eps: f32, mut f: F)
where
    F: FnMut(usize, &[f32], f32, f32),
{
    let dim = push.dim;
    for t in 0..push.n_rows {
        f(t, &push.rows[t * dim..(t + 1) * dim], 1.0, -delta);
    }
    for i in 0..keep.n_rows {
        f(
            push.n_rows + i,
            &keep.rows[i * dim..(i + 1) * dim],
            1.0,
            eps,
        );
    }
    for i in 0..keep.n_rows {
        f(
            push.n_rows + keep.n_rows + i,
            &keep.rows[i * dim..(i + 1) * dim],
            -1.0,
            eps,
        );
    }
}

/// The closed-form spectral-norm bound `sqrt(‖A‖₁·‖A‖_∞)` — a GUARANTEED
/// upper bound on `‖A‖₂`, so `σ = τ = 0.995/L` satisfies `τσ‖A‖² < 1`
/// unconditionally. Sequential folds throughout. An all-zero constraint
/// matrix (no signal) refuses loud.
fn operator_norm_bound(push: &TraceSet<'_>, keep: &TraceSet<'_>) -> Result<f32, LpAuthoringError> {
    let dim = push.dim;
    let mut col_abs = vec![0.0f32; dim];
    let mut max_row = 0.0f32;
    for_each_row(push, keep, 0.0, 0.0, |_t, h, _sign, _b| {
        let mut row_abs = 0.0f32;
        for (c, &x) in col_abs.iter_mut().zip(h.iter()) {
            let a = x.abs();
            *c += a;
            row_abs += a;
        }
        if row_abs > max_row {
            max_row = row_abs;
        }
    });
    let max_col = col_abs.iter().copied().fold(0.0f32, f32::max);
    // NaN-safe "strictly positive" guards (a NaN max means a poisoned row —
    // refused, never silently solved).
    if max_col <= 0.0 || max_col.is_nan() || max_row <= 0.0 || max_row.is_nan() {
        return Err(LpAuthoringError::Malformed(
            "degenerate zero constraint matrix",
        ));
    }
    Ok((max_col * max_row).sqrt())
}

/// L1 shrinkage prox (`prox_{τ‖·‖₁}`): exact zeros below the threshold —
/// this is what makes the solution's support well-defined.
fn soft_threshold(x: f32, t: f32) -> f32 {
    if x > t {
        x - t
    } else if x < -t {
        x + t
    } else {
        0.0
    }
}

// ─────────────────────────────────────────────────────────────────────
// Authoring helpers
// ─────────────────────────────────────────────────────────────────────

/// The authoring instrument's sparsity floor: entries with `|v_j| <
/// floor` become exact `0.0`. First-order shrinkage leaves sub-floor dust
/// off the support; the floor turns the nonzero count into a pinned,
/// reproducible number (PDLP's solution-polishing posture).
/// Deterministic; apply before composing, never inside the solver.
pub fn truncate_small(v: Vec<f32>, floor: f32) -> Result<Vec<f32>, LpAuthoringError> {
    if !floor.is_finite() || floor < 0.0 {
        return Err(LpAuthoringError::Malformed("floor must be finite >= 0"));
    }
    let mut v = v;
    for slot in v.iter_mut() {
        if slot.abs() < floor {
            *slot = 0.0;
        }
    }
    Ok(v)
}

/// Compose the rank-1 overlay from a direction `u` (the layer's output
/// width) and per-column coefficients `v` (the layer's input width): one
/// single-column `EditForm::DirectionSubtraction` spec per nonzero
/// coefficient, in ASCENDING column order (the determinism law — the
/// overlay's application folds edits in spec order, so the order is
/// load-bearing). Exact-`0.0` coefficients are skipped. An all-zero `v`
/// is the loud [`LpAuthoringError::DegenerateZeroSolution`].
pub fn compose_direction_overlay(
    direction: &[f32],
    coefficients: &[f32],
    layer: u8,
    in_dim: usize,
) -> Result<WeightEditOverlay, LpAuthoringError> {
    if direction.is_empty() {
        return Err(LpAuthoringError::Malformed("empty edit direction"));
    }
    if direction.iter().any(|x| !x.is_finite()) {
        return Err(LpAuthoringError::Malformed(
            "non-finite edit direction entry",
        ));
    }
    if in_dim == 0 || coefficients.len() != in_dim {
        return Err(LpAuthoringError::Malformed(
            "coefficients length must equal in_dim >= 1",
        ));
    }
    if in_dim > u32::MAX as usize {
        return Err(LpAuthoringError::Malformed(
            "in_dim exceeds the u32 column index",
        ));
    }
    let mut edits = Vec::new();
    for (j, &coef) in coefficients.iter().enumerate() {
        if coef == 0.0 {
            continue;
        }
        if !coef.is_finite() {
            return Err(LpAuthoringError::Malformed("non-finite coefficient"));
        }
        edits.push(WeightEditSpec {
            layer,
            form: EditForm::DirectionSubtraction {
                columns: vec![j as u32],
                direction: direction.to_vec(),
                strength: coef,
            },
        });
    }
    if edits.is_empty() {
        return Err(LpAuthoringError::DegenerateZeroSolution);
    }
    WeightEditOverlay::new(edits).ok_or(LpAuthoringError::Malformed(
        "overlay refused an LP-authored edit",
    ))
}

// ─────────────────────────────────────────────────────────────────────
// Unit tests — the synthetic planted fixture (T2's solver half)
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SYNTH_DIM: usize = 256;
    const SYNTH_SPOTS: [usize; 8] = [3, 17, 64, 100, 129, 200, 230, 255];
    const BLOCK_LO: usize = 40;
    const BLOCK_HI: usize = 60;

    /// The planted unit direction: 8 nonzeros ±1/√8 at pinned indices.
    fn planted_u() -> Vec<f32> {
        let mut u = vec![0.0f32; SYNTH_DIM];
        let amp = 1.0 / (SYNTH_SPOTS.len() as f32).sqrt();
        for (k, &j) in SYNTH_SPOTS.iter().enumerate() {
            u[j] = if k % 2 == 0 { amp } else { -amp };
        }
        u
    }

    /// Deterministic LCG value in [−1, 1) (test-only fixture randomness).
    fn seeded_unit_lcg(seed: u64) -> f32 {
        let s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 33) as f32) / ((1u64 << 31) as f32)) * 2.0 - 1.0
    }

    /// Push-class rows: `a·u` with `a` ramping from `a_base`.
    fn synth_push_rows(a_base: f32, a_step: f32, n: usize) -> Vec<f32> {
        let u = planted_u();
        let mut rows = Vec::with_capacity(n * SYNTH_DIM);
        for t in 0..n {
            let a = a_base + a_step * (t % 64) as f32;
            for &x in u.iter() {
                rows.push(a * x);
            }
        }
        rows
    }

    /// Keep-class rows: deterministic pseudo-values on a block disjoint
    /// from the direction's support (exactly carrier-silent off it).
    fn synth_keep_rows(seed_base: u64, n: usize) -> Vec<f32> {
        let mut rows = Vec::with_capacity(n * SYNTH_DIM);
        for t in 0..n {
            for j in 0..SYNTH_DIM {
                let val = if (BLOCK_LO..BLOCK_HI).contains(&j) {
                    seeded_unit_lcg(seed_base + (t * SYNTH_DIM + j) as u64) * 0.8
                } else {
                    0.0
                };
                rows.push(val);
            }
        }
        rows
    }

    fn synth_cfg() -> LpAuthoringConfig {
        LpAuthoringConfig {
            delta_push: 1.0,
            epsilon_keep: 0.1,
            c_slack: 10.0,
            ..Default::default()
        }
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            acc += x * y;
        }
        acc
    }

    #[test]
    fn planted_direction_solve_is_sparse_and_separating() {
        let push_rows = synth_push_rows(1.0, 0.05, 64);
        let keep_rows = synth_keep_rows(1_000, 64);
        let push = TraceSet::new(&push_rows, 64, SYNTH_DIM).unwrap();
        let keep = TraceSet::new(&keep_rows, 64, SYNTH_DIM).unwrap();
        let sol = solve_min_l1_authoring_lp(&push, &keep, &synth_cfg()).expect("solve");

        let max_abs = sol.v.iter().copied().fold(0.0f32, f32::max).abs();
        assert!(max_abs > 0.0, "solve must not return an empty direction");
        let v = truncate_small(sol.v.clone(), 1e-3 * max_abs).expect("floor");

        // (a) sparsity ≤ K, and the support is ⊆ the planted direction's
        // support: off-support entries carry no constraint mass and cost
        // L1, so ANY optimum is supported on the spots.
        let nonzeros = v.iter().filter(|&&x| x != 0.0).count();
        assert!(
            nonzeros <= SYNTH_SPOTS.len(),
            "support {nonzeros} exceeds the planted K"
        );
        for (j, &x) in v.iter().enumerate() {
            if !SYNTH_SPOTS.contains(&j) {
                assert_eq!(x, 0.0, "off-support dust at {j}");
            }
        }

        // (b) separation: the fit binding constraint saturates …
        let u = planted_u();
        let vu = dot(&v, &u);
        assert!(vu <= -1.0 + 1e-2, "v·u = {vu} must saturate the push");

        // … and HELD-OUT rows the solve never saw separate too.
        let held_push = synth_push_rows(1.5, 0.07, 32);
        let held_keep = synth_keep_rows(2_000, 16);
        for t in 0..32usize {
            let row = &held_push[t * SYNTH_DIM..(t + 1) * SYNTH_DIM];
            let acc = dot(&v, row);
            assert!(acc <= -1.0 + 0.05, "held-out push violated: {acc}");
        }
        for t in 0..16usize {
            let row = &held_keep[t * SYNTH_DIM..(t + 1) * SYNTH_DIM];
            let acc = dot(&v, row);
            assert!(acc.abs() <= 1e-3, "held-out keep violated: {acc}");
        }
    }

    #[test]
    fn solve_is_bit_identical_across_runs() {
        let push_rows = synth_push_rows(1.0, 0.05, 64);
        let keep_rows = synth_keep_rows(1_000, 64);
        let cfg = synth_cfg();
        let run = || {
            let push = TraceSet::new(&push_rows, 64, SYNTH_DIM).unwrap();
            let keep = TraceSet::new(&keep_rows, 64, SYNTH_DIM).unwrap();
            solve_min_l1_authoring_lp(&push, &keep, &cfg).expect("solve")
        };
        let a = run();
        let b = run();
        assert!(
            a.v.iter()
                .zip(b.v.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "two solves on the same pinned input must be bit-identical"
        );
        assert_eq!(a.iterations, b.iterations);
        assert_eq!(a.objective.to_bits(), b.objective.to_bits());
        assert_eq!(a.nonzeros, b.nonzeros);
    }

    #[test]
    fn unconstrained_problem_refuses_loud() {
        // No push rows → the only optimum is v = 0 → the LOUD refusal.
        let keep_rows = synth_keep_rows(1_000, 64);
        let push = TraceSet::new(&[], 0, SYNTH_DIM).unwrap();
        let keep = TraceSet::new(&keep_rows, 64, SYNTH_DIM).unwrap();
        let err =
            solve_min_l1_authoring_lp(&push, &keep, &LpAuthoringConfig::default()).unwrap_err();
        assert_eq!(err, LpAuthoringError::DegenerateZeroSolution);
    }

    #[test]
    fn binding_at_weakest_state() {
        // Push rows [1,0] and [0.5,0]; keep [0,1]: the WEAKEST push row
        // pins v₀ = −δ/0.5 = −2δ exactly; v₁ = 0 (L1).
        let rows = [1.0f32, 0.0, 0.5, 0.0];
        let keep_rows = [0.0f32, 1.0];
        let push = TraceSet::new(&rows, 2, 2).unwrap();
        let keep = TraceSet::new(&keep_rows, 1, 2).unwrap();
        let sol = solve_min_l1_authoring_lp(&push, &keep, &synth_cfg()).unwrap();
        assert!(
            (sol.v[0] + 2.0).abs() <= 0.02,
            "v₀ = {} (want −2: the weakest row binds)",
            sol.v[0]
        );
        assert_eq!(sol.v[1], 0.0);
    }

    #[test]
    fn malformed_inputs_refuse() {
        // Shape mismatch.
        assert!(TraceSet::new(&[1.0], 2, 1).is_err());
        // Non-finite entry.
        assert!(TraceSet::new(&[f32::NAN], 1, 1).is_err());
        // Zero dim.
        assert!(TraceSet::new(&[], 0, 0).is_err());
        // Dim mismatch between the sets.
        let p = TraceSet::new(&[1.0, 0.0], 1, 2).unwrap();
        let k1 = TraceSet::new(&[1.0], 1, 1).unwrap();
        assert!(matches!(
            solve_min_l1_authoring_lp(&p, &k1, &LpAuthoringConfig::default()),
            Err(LpAuthoringError::Malformed(_))
        ));
        // Bad config.
        let k2 = TraceSet::new(&[1.0, 0.0], 1, 2).unwrap();
        let bad = LpAuthoringConfig {
            c_slack: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            solve_min_l1_authoring_lp(&p, &k2, &bad),
            Err(LpAuthoringError::Malformed(_))
        ));
        // Zero constraint matrix — no signal.
        let z = TraceSet::new(&[0.0, 0.0], 1, 2).unwrap();
        assert!(matches!(
            solve_min_l1_authoring_lp(&z, &z, &LpAuthoringConfig::default()),
            Err(LpAuthoringError::Malformed(_))
        ));
    }

    #[test]
    fn compose_direction_overlay_order_and_refusals() {
        let u = vec![0.0f32, 1.0];
        let mut v = vec![0.0f32; 4];
        v[2] = 1.5;
        v[0] = -0.5;
        let ov = compose_direction_overlay(&u, &v, 3, 4).unwrap();
        let mut cols = Vec::new();
        for e in ov.edits() {
            if let EditForm::DirectionSubtraction { columns, .. } = &e.form {
                cols.extend_from_slice(columns);
            }
        }
        assert_eq!(cols, vec![0, 2], "ascending column order is load-bearing");

        // All-zero coefficients refuse loud.
        let zero = vec![0.0f32; 4];
        assert_eq!(
            compose_direction_overlay(&u, &zero, 3, 4).unwrap_err(),
            LpAuthoringError::DegenerateZeroSolution
        );
        // Non-finite direction refuses.
        let mut bad_u = u.clone();
        bad_u[0] = f32::INFINITY;
        assert!(compose_direction_overlay(&bad_u, &v, 3, 4).is_err());
        // Shape mismatch refuses.
        assert!(compose_direction_overlay(&u, &v, 3, 5).is_err());
    }

    #[test]
    fn truncate_small_refuses_bad_floor_and_zeroes_dust() {
        let v = vec![1.0f32, 1e-5, -1e-5, 0.0];
        let t = truncate_small(v.clone(), 1e-3).unwrap();
        assert_eq!(t, vec![1.0, 0.0, 0.0, 0.0]);
        assert!(truncate_small(v, f32::NAN).is_err());
    }
}
