//! TPR (Tensor Product Representation) binding algebra — the modelless
//! rank-`m` generalization of the single-direction-vector latent ops
//! (Issue 707, Research 527; arXiv:2608.29530 McCoy/Soulos/Linzen/Smolensky).
//!
//! A structured latent state is `e = W·c + b`, where the **core**
//! `c ∈ R^{m·d}` is role-major (block `i` = coords `[i·d, (i+1)·d)`) and a
//! binding `(role p, filler v)` contributes `r_p[i]·f_v` to block `i`. The
//! one-hot ([`TprScheme::Orthogonal`]) case degenerates to "block `p` holds
//! `f_v`", which is the exact-recovery regime; general role vectors
//! ([`TprScheme::RoleVectors`]) trade exactness for a computable crosstalk
//! envelope ([`unbind_error_bound`]).
//!
//! Four runtime ops, all zero-alloc and caller-scratch (Issue 707 T1–T4):
//!
//! | op | form | cost |
//! |---|---|---|
//! | [`bind_into`] | `W_r·f` | one skinny `D×d` GEMV |
//! | [`unbind_into`] | `M·B_p` | one `m×d` contraction |
//! | [`surgery_delta_into`] | `e + W_r(f_new − f_old)` | one GEMV + axpy |
//! | [`project_into`] | `W·C⁻¹Wᵀ(e−b)+b` | two GEMVs + a cached Cholesky solve |
//!
//! The fit ([`als`]) is closed-form ridge-ALS — **no gradient descent
//! anywhere**; it emits a frozen BLAKE3-committed [`TprArtifact`] that thaws
//! at runtime. The validation harness ([`validate`]) ships the GOAT-gate
//! instruments (planted control, withheld-pair OOD vs the atomic-dictionary
//! null, BoW structure router, BIC scheme selection).
//!
//! **Kill switch:** `RIIR_TPR=0` in the environment makes every op return
//! [`TprError::Disabled`] (the G3 gate) — read once and cached, so the check
//! costs a relaxed atomic load in steady state.
//!
//! # Every control reports whether it could have failed
//!
//! A control whose permutation (or null, or candidate pool) is a provable
//! identity returns the same numbers as a real negative result, so its pass is
//! silent. [`AtomicNull::coverage`] was designed against this; the role control
//! was not, and shipped a within-state shuffle that is an **empty loop** on
//! single-binding corpora — the shape every retrieval consumer produces. It
//! condemned riir-clippy's Issue 062 corpus (325 states, all `len() == 1`) at
//! `ratio == 1.0` while [`bow_router`] called the same artifact structured.
//!
//! Fixed by making vacuity a reported quantity rather than something a
//! consumer must infer from the source (Issue 710, closed + removed here):
//!
//! | instrument | vacuity signal | vacuous when |
//! |---|---|---|
//! | [`AtomicNull`] | [`AtomicNull::coverage`] | the dictionary has no entry for the candidates |
//! | [`shuffled_role_control`] | [`ShuffledRoleReport::vacuous`] + `moved` | the drawn permutation is the identity ([`role_shuffle_is_vacuous`]) |
//! | [`bow_router`] | [`BowRouterReport::vacuous`] | the `m = 1` null IS the caller's fit |
//! | [`withheld_pair_top1`] | [`candidate_pool_coverage`] | a truth is absent from the shared pool |
//!
//! [`shuffled_role_control`] additionally *resolves* the arm
//! ([`role_shuffle_mode_for`]): single-binding corpora get the cross-state
//! permutation, which can fail (measured `ratio` 1.85e8 vs the within-state
//! arm's 1.0 on the same planted corpus). Multi-binding corpora are
//! bit-identical to the pre-710 behaviour.
//!
//! ## The mode no vacuity flag can catch (Issue 711)
//!
//! A control can be perfectly capable of failing and still measure the wrong
//! question. When every filler is seen with exactly one role — `role =
//! f(filler)` — the probes still *move* (the `m`-role model has more capacity
//! than the 1-role null and fits differently) and neither `vacuous` flag
//! fires, but there are **no unseen `(role, filler)` pairs**, so systematicity
//! is not posed on that corpus. Measured: 445 GPU-optimization rules, one
//! declared category each, reported `structured = false` — read as "the
//! largest corpus we have carries no binding structure" when it was a
//! statement about the scheme's applicability (riir-clippy Bench 063 §12.4).
//!
//! The covariate is therefore reported, not just its threshold:
//! [`FillerRoleSpread`] rides on both [`BowRouterReport::spread`] and
//! [`ShuffledRoleReport::spread`], and `verdict()` on either report returns
//! `None` rather than an uninterpretable bool. `max`/`mean` are both carried
//! so a *near*-degenerate corpus (1.02 roles per filler) is visible too.
//!
//! [`withheld_pair_top1`] is hit hardest — withholding a pair withholds the
//! whole filler there, so its OOD arm becomes a different question rather than
//! a harder one. [`withheld_pair_top1_report`] carries the covariate, the
//! [`candidate_pool_coverage`] ceiling and the raw number together;
//! [`WithheldPairReport::verdict`] withholds. The raw `f32` function is
//! **unchanged**, so the Issue 707 G8 gate keeps its number and the refusal is
//! available to any consumer that wants it — which is what made "should it
//! REFUSE?" a non-question rather than an owner call.

pub mod als;
pub mod types;
pub mod validate;

#[cfg(test)]
mod tests;

pub use als::als_fit;
pub use types::{
    AlsConfig, AlsInput, AlsReport, L21, SCHEMA_VERSION, TPR_MAX_PROJECTION_K, TprArtifact,
    TprBindings, TprError, TprScheme,
};
pub use validate::{
    AtomicNull, BicSelection, BindingReport, BowRouterReport, FillerRoleSpread, MAX_SHUFFLE_DRAWS,
    ObservedPairs, RoleShuffleMode, ShuffledRoleReport, WithheldPairReport, bic_select, bow_router,
    candidate_pool_coverage, filler_role_spread, role_determined_by_filler,
    role_shuffle_is_vacuous, role_shuffle_mode_for, shuffled_role_control,
    shuffled_role_control_with, validate_bindings, withheld_pair_top1, withheld_pair_top1_report,
};

use crate::simd::{simd_dot_f32, simd_matvec};
use std::sync::OnceLock;

/// Logistic slope of [`unbind_confidence`]. Fixed so the gate is a pure
/// function of the artifact + binding count (no tunable that could be fitted
/// after the fact).
const CONFIDENCE_SLOPE: f32 = 6.0;

/// Pure parse of the kill-switch variable — `RIIR_TPR=0` disables, anything
/// else (including unset and empty) leaves the ops live. Split out so the
/// contract is testable without mutating process env, which the `OnceLock`
/// cache below would make order-dependent anyway.
fn parse_kill(v: Option<&str>) -> bool {
    matches!(v, Some("0"))
}

/// `RIIR_TPR=0` kill switch, read once.
fn kill_switch() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| parse_kill(std::env::var("RIIR_TPR").ok().as_deref()))
}

/// Whether the TPR ops are live (`RIIR_TPR` unset or ≠ `0`).
#[inline]
pub fn tpr_enabled() -> bool {
    !kill_switch()
}

#[inline]
fn enabled() -> Result<(), TprError> {
    if kill_switch() { Err(TprError::Disabled) } else { Ok(()) }
}

#[inline]
fn check_len(what: &'static str, got: usize, expected: usize) -> Result<(), TprError> {
    if got == expected { Ok(()) } else { Err(TprError::DimMismatch {
            what,
            expected,
            got,
        }) }
}

/// `out += alpha · src` over a contiguous run. No cross-iteration
/// dependency, so LLVM vectorizes this directly — the house SIMD module has
/// no axpy kernel and does not need one here.
#[inline]
fn axpy(out: &mut [f32], src: &[f32], alpha: f32) {
    for (o, &s) in out.iter_mut().zip(src.iter()) {
        *o = alpha.mul_add(s, *o);
    }
}

/// `out += alpha · (W_block · f)` expressed as `d` column axpys — the form
/// that made bind 10× faster than the row-reduction form (see
/// [`TprArtifact::w`] for the measured layout table).
#[inline]
fn block_axpy(art: &TprArtifact, base: usize, filler: &[f32], alpha: f32, out: &mut [f32]) {
    let dim = art.dim;
    for (j, &fv) in filler.iter().enumerate() {
        let a = alpha * fv;
        if a == 0.0 { continue } else {
                let s = base + j * dim;
                axpy(out, &art.w[s..s + dim], a);
            }
    }
}

/// Caller-owned scratch for the ops that need working space
/// ([`project_into`], [`encode_into`], [`unbind_state_into`]). Allocate once
/// per consumer, reuse across calls — the ops themselves never allocate.
#[derive(Debug, Clone)]
pub struct TprScratch {
    /// Core-space working vector, length `K = m·d`.
    pub(crate) core: Vec<f32>,
    /// `Wᵀ(e−b)` accumulator, length `K`.
    pub(crate) t: Vec<f32>,
    /// Solution vector, length `K`.
    pub(crate) x: Vec<f32>,
    /// Filler-space working vector, length `d`.
    pub(crate) f: Vec<f32>,
    /// State-space centering buffer (`e − b`), length `D`.
    pub(crate) e: Vec<f32>,
}

impl TprScratch {
    /// Size the scratch for `art`. One allocation set, then zero-alloc ops.
    pub fn new(art: &TprArtifact) -> Self {
        let k = art.core_len();
        Self {
            core: vec![0.0; k],
            t: vec![0.0; k],
            x: vec![0.0; k],
            f: vec![0.0; art.d],
            e: vec![0.0; art.dim],
        }
    }

    /// Read-only view of the core buffer (populated by [`encode_into`] /
    /// [`state_to_core_into`]).
    pub fn core(&self) -> &[f32] {
        &self.core
    }
}

// ---------------------------------------------------------------------------
// T1 — bind
// ---------------------------------------------------------------------------

/// Per-role role weights over the `m` core blocks. Orthogonal schemes are the
/// implicit one-hot; role-vector schemes read the fitted row.
#[inline]
fn role_weights<'a>(art: &'a TprArtifact, role: u16) -> Result<RoleWeights<'a>, TprError> {
    match &art.scheme {
        TprScheme::Orthogonal { arity } => if (role as usize) < *arity { Ok(RoleWeights::OneHot(role as usize)) } else { Err(TprError::BadId {
                what: "role",
                max: arity.saturating_sub(1),
                got: role as usize,
            }) },
        TprScheme::RoleVectors { .. } => match art.scheme.role_vec(role) {
            Some(r) => Ok(RoleWeights::Dense(r)),
            None => Err(TprError::BadId {
                what: "role",
                max: art.scheme.n_bind_slots().saturating_sub(1),
                got: role as usize,
            }),
        },
    }
}

enum RoleWeights<'a> {
    OneHot(usize),
    Dense(&'a [f32]),
}

/// `out += alpha · (W_r · f)` — the T1 skinny GEMV, accumulating.
///
/// One pass over the `D` rows of `W`; the one-hot case touches exactly one
/// contiguous `d`-run per row.
fn bind_accum(
    art: &TprArtifact,
    role: u16,
    filler: &[f32],
    alpha: f32,
    out: &mut [f32],
) -> Result<(), TprError> {
    check_len("filler", filler.len(), art.d)?;
    check_len("state", out.len(), art.dim)?;
    match role_weights(art, role)? {
        RoleWeights::OneHot(block) => {
            block_axpy(art, art.block_offset(block), filler, alpha, out);
        }
        RoleWeights::Dense(r) => {
            for (blk, &rw) in r.iter().enumerate().take(art.m) {
                if rw == 0.0 { continue } else { block_axpy(art, art.block_offset(blk), filler, alpha * rw, out) }
            }
        }
    }
    Ok(())
}

/// **T1** `bind(f, r) = W_r·f` — writes the state-space contribution of one
/// role-filler binding into `out` (overwriting).
pub fn bind_into(
    art: &TprArtifact,
    role: u16,
    filler: &[f32],
    out: &mut [f32],
) -> Result<(), TprError> {
    enabled()?;
    check_len("state", out.len(), art.dim)?;
    out.fill(0.0);
    bind_accum(art, role, filler, 1.0, out)
}

/// `out += W_r·f` — the accumulating form (compose a multi-binding state
/// without a core buffer).
pub fn bind_add_into(
    art: &TprArtifact,
    role: u16,
    filler: &[f32],
    out: &mut [f32],
) -> Result<(), TprError> {
    enabled()?;
    bind_accum(art, role, filler, 1.0, out)
}

/// Accumulate the core `c = Σ_j r_{p_j} ⊗ f_{v_j}` for `bindings`, reading
/// filler rows from the artifact's shared table. `core` is overwritten.
pub fn core_encode_into(
    art: &TprArtifact,
    bindings: &TprBindings,
    core: &mut [f32],
) -> Result<(), TprError> {
    enabled()?;
    check_len("core", core.len(), art.core_len())?;
    core.fill(0.0);
    for (&p, &v) in bindings.roles.iter().zip(bindings.fillers.iter()) {
        let vi = v as usize;
        if vi >= art.n_fillers {
            return Err(TprError::BadId {
                what: "filler",
                max: art.n_fillers.saturating_sub(1),
                got: vi,
            });
        }
        let f = &art.fillers[vi * art.d..(vi + 1) * art.d];
        match role_weights(art, p)? {
            RoleWeights::OneHot(block) => {
                // Core-space offset (`p·d`), NOT `block_offset` — that one
                // addresses the encoder's contiguous D×d block.
                let off = block * art.d;
                for (j, &fv) in f.iter().enumerate() {
                    core[off + j] += fv;
                }
            }
            RoleWeights::Dense(r) => {
                for (blk, &rw) in r.iter().enumerate().take(art.m) {
                    if rw == 0.0 { continue } else {
                            let off = blk * art.d;
                            for (j, &fv) in f.iter().enumerate() {
                                core[off + j] = rw.mul_add(fv, core[off + j]);
                            }
                        }
                }
            }
        }
    }
    Ok(())
}

/// `out = W·core + b` — the core → state decoder.
pub fn state_from_core_into(
    art: &TprArtifact,
    core: &[f32],
    out: &mut [f32],
) -> Result<(), TprError> {
    enabled()?;
    check_len("core", core.len(), art.core_len())?;
    check_len("state", out.len(), art.dim)?;
    let d = art.d;
    let dim = art.dim;
    out.copy_from_slice(&art.bias);
    for p in 0..art.m {
        let base = art.block_offset(p);
        for j in 0..d {
            let c = core[p * d + j];
            if c == 0.0 { continue } else {
                    let s = base + j * dim;
                    axpy(out, &art.w[s..s + dim], c);
                }
        }
    }
    Ok(())
}

/// Encode a full binding set to a state: `e = W·(Σ r⊗f) + b`.
pub fn encode_into(
    art: &TprArtifact,
    bindings: &TprBindings,
    scratch: &mut TprScratch,
    out: &mut [f32],
) -> Result<(), TprError> {
    core_encode_into(art, bindings, &mut scratch.core)?;
    state_from_core_into(art, &scratch.core, out)
}

// ---------------------------------------------------------------------------
// T2 — unbind
// ---------------------------------------------------------------------------

/// **T2** `f̂_p = M·B_p` — recover the filler bound to role `p` from a core.
///
/// The core is read as the `d×m` matrix `M` (column `i` = block `i`), so this
/// is the contraction `Σ_i B_p[i]·c[i·d + j]`. Orthogonal schemes copy block
/// `p` exactly; role-vector schemes contract against the artifact's signed,
/// permuted orthonormal unbind basis and carry the crosstalk envelope of
/// [`unbind_error_bound`].
pub fn unbind_into(
    art: &TprArtifact,
    core: &[f32],
    role: u16,
    out: &mut [f32],
) -> Result<(), TprError> {
    enabled()?;
    check_len("core", core.len(), art.core_len())?;
    check_len("filler", out.len(), art.d)?;
    match (&art.scheme, &art.unbind_basis) {
        (TprScheme::Orthogonal { arity }, _) => {
            let p = role as usize;
            if !(p < *arity) { Err(TprError::BadId {
                    what: "role",
                    max: arity.saturating_sub(1),
                    got: p,
                }) } else {
                    let off = p * art.d;
                    out.copy_from_slice(&core[off..off + art.d]);
                    Ok(())
                }
        }
        (TprScheme::RoleVectors { .. }, Some(basis)) => {
            let p = role as usize;
            let m = art.m;
            if !((p + 1) * m <= basis.len()) { Err(TprError::BadId {
                    what: "role",
                    max: basis.len() / m.max(1) - 1,
                    got: p,
                }) } else {
                    let bp = &basis[p * m..(p + 1) * m];
                    out.fill(0.0);
                    for (blk, &w) in bp.iter().enumerate() {
                        if w == 0.0 { continue } else {
                                let off = blk * art.d;
                                for (j, o) in out.iter_mut().enumerate() {
                                    *o = w.mul_add(core[off + j], *o);
                                }
                            }
                    }
                    Ok(())
                }
        }
        (TprScheme::RoleVectors { .. }, None) => Err(TprError::BadEncoding(
            "role-vector artifact without an unbind basis",
        )),
    }
}

/// Recover a filler straight from a **state** — projects to the core with the
/// cached Cholesky, then unbinds. Requires the projection artifact.
pub fn unbind_state_into(
    art: &TprArtifact,
    state: &[f32],
    role: u16,
    scratch: &mut TprScratch,
    out: &mut [f32],
) -> Result<(), TprError> {
    state_to_core_into(art, state, scratch)?;
    let core = std::mem::take(&mut scratch.x);
    let r = unbind_into(art, &core, role, out);
    scratch.x = core;
    r
}

/// The T2 crosstalk envelope: `‖Δf‖ ≤ μ·(n_bindings − 1)·max‖f‖`.
///
/// `μ` is the artifact's fitted coherence (0 for orthogonal schemes, where
/// recovery is exact). Reported as an absolute state-independent bound — the
/// worst case over filler tables of the fitted scale.
pub fn unbind_error_bound(art: &TprArtifact, n_bindings: usize) -> f32 {
    art.crosstalk_mu * (n_bindings.saturating_sub(1) as f32) * art.max_filler_norm
}

/// Sigmoid gate over the crosstalk envelope: `σ(k·(1 − μ·(n−1)))`.
///
/// ≥ 0.5 exactly when the relative envelope stays under the filler scale —
/// i.e. when the recovered filler is closer to its own row than to the
/// crosstalk floor. Sigmoid, never softmax (house rule).
pub fn unbind_confidence(art: &TprArtifact, n_bindings: usize) -> f32 {
    let rel = art.crosstalk_mu * (n_bindings.saturating_sub(1) as f32);
    1.0 / (1.0 + (-CONFIDENCE_SLOPE * (1.0 - rel)).exp())
}

// ---------------------------------------------------------------------------
// T3 — surgery
// ---------------------------------------------------------------------------

/// **T3** in-place constituent replacement: `e ← e + W_r(f_new − f_old)`.
///
/// Exactly additive by construction — the edit never re-encodes the untouched
/// bindings, so their contribution is bit-preserved. `scratch` holds the
/// filler-space difference (length `d`) so this costs ONE GEMV.
pub fn surgery_delta_into(
    art: &TprArtifact,
    state: &mut [f32],
    role: u16,
    f_old: &[f32],
    f_new: &[f32],
    scratch: &mut TprScratch,
) -> Result<(), TprError> {
    enabled()?;
    check_len("filler", f_old.len(), art.d)?;
    check_len("filler", f_new.len(), art.d)?;
    for j in 0..art.d {
        scratch.f[j] = f_new[j] - f_old[j];
    }
    let diff = std::mem::take(&mut scratch.f);
    let r = bind_accum(art, role, &diff, 1.0, state);
    scratch.f = diff;
    r
}

/// **T3** role-crossing surgery: `e ← e − W_{r_from}·f + W_{r_to}·f`.
///
/// One pass over the `D` rows of `W` — both role blocks are read per row, so
/// this costs the same memory traffic as a single GEMV over `2d` columns.
pub fn surgery_move_into(
    art: &TprArtifact,
    state: &mut [f32],
    role_from: u16,
    role_to: u16,
    filler: &[f32],
) -> Result<(), TprError> {
    enabled()?;
    check_len("filler", filler.len(), art.d)?;
    check_len("state", state.len(), art.dim)?;
    // Two contiguous block passes. Fusing them into one row loop would stride
    // both blocks and cost more than the second sequential pass.
    bind_accum(art, role_from, filler, -1.0, state)?;
    bind_accum(art, role_to, filler, 1.0, state)
}

// ---------------------------------------------------------------------------
// T4 — structural projection (denoise)
// ---------------------------------------------------------------------------

/// Solve `C·x = Wᵀ(e−b)` into `scratch.x` with the artifact's cached
/// Cholesky — the shared half of [`project_into`] and [`unbind_state_into`].
pub fn state_to_core_into(
    art: &TprArtifact,
    state: &[f32],
    scratch: &mut TprScratch,
) -> Result<(), TprError> {
    enabled()?;
    check_len("state", state.len(), art.dim)?;
    let Some(inv) = &art.projection_inv else { return Err(TprError::ProjectionUnavailable) };
    let k = art.core_len();
    let d = art.d;
    let dim = art.dim;
    // Centre once into scratch, then read each core coordinate as ONE
    // contiguous length-`D` dot against its column — the column-slice layout
    // turns `Wᵀ(e−b)` into `K` independent SIMD reductions with no strided
    // access and no accumulator aliasing.
    for (i, e) in scratch.e.iter_mut().enumerate() {
        *e = state[i] - art.bias[i];
    }
    let centered = std::mem::take(&mut scratch.e);
    for p in 0..art.m {
        let base = art.block_offset(p);
        for j in 0..d {
            let c = base + j * dim;
            scratch.t[p * d + j] = simd_dot_f32(&art.w[c..c + dim], &centered, dim);
        }
    }
    scratch.e = centered;
    // `x = C⁻¹·t` as ONE K×K matvec. The fit factors `C` by Cholesky (that
    // is where positive-definiteness is checked) but ships the explicit
    // inverse, because a triangular solve at K=32 is a 32-step SERIAL
    // dependency chain: it measured the whole projection at 2.14× its
    // two-GEMV floor against a 2× bar, while the matvec form is latency-free.
    let t = std::mem::take(&mut scratch.t);
    let x = std::mem::take(&mut scratch.x);
    let mut x = x;
    simd_matvec(&mut x, inv, &t, k, k);
    scratch.t = t;
    scratch.x = x;
    Ok(())
}

/// **T4** structural denoise: `ê = W·C⁻¹Wᵀ(e−b) + b`, with
/// `C = WᵀW + λI` factored once at fit time.
///
/// The a-priori certificate is [`TprArtifact::residual_energy_fraction`] —
/// the fraction of centered state energy the fitted manifold does NOT
/// capture. The a-posteriori one is the returned residual `‖e − ê‖`.
pub fn project_into(
    art: &TprArtifact,
    state: &[f32],
    scratch: &mut TprScratch,
    out: &mut [f32],
) -> Result<f32, TprError> {
    state_to_core_into(art, state, scratch)?;
    let x = std::mem::take(&mut scratch.x);
    let r = state_from_core_into(art, &x, out);
    scratch.x = x;
    r?;
    let mut ss = 0.0f32;
    for (o, s) in out.iter().zip(state.iter()) {
        let dv = o - s;
        ss = dv.mul_add(dv, ss);
    }
    Ok(ss.sqrt())
}
