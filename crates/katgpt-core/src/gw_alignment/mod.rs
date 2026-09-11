// Gromov–Wasserstein quotient alignment (Issue 743 / riir-ai Issue 912 T4
// BUILD decision, Research 371) — structure-only cross-space alignment of two
// precomputed distance matrices `D_A` (n×n) and `D_B` (m×m), n,m ≤
// [`GW_MAX`], uniform weights. Unlike `mag::transfer::Wasserstein1d`
// (pointwise ground cost — shared coordinates required) and RSA (index-aligned
// rank correlation — correspondence required), GW aligns using ONLY
// intra-space structure: "do these two probe sets see the same shape of
// distances" with no shared coordinates and no correspondence prior.
//
// Method: first-order stationary point via the product-graph power iteration
// (Peyré/Cuturi/Solomon 2016, "Gromov-Wasserstein Averaging of Kernel and
// Distance Matrices" — conditional-gradient formulation; Mémoli 2011 for the
// distance). The multiplicative reweighting `T ← T ⊙ (D_A·T·D_B)` is the
// power method on the product graph; between power steps the coupling is
// projected back into the uniform-weight transportation polytope by
// alternating row/column normalization (Sinkhorn without entropy). Fixed init
// + fixed iteration count + no RNG ⇒ bit-deterministic (replay-safe by
// construction).
//
// The closed-form loss uses the uniform-coupling identity (row sums 1/n, col
// sums 1/m): `loss = froA2/n² + froB2/m² − 2·⟨T, D_A·T·D_B⟩`. Verified
// against the direct O(n²m²) quadratic form by a unit test.
//
// Latent-only boundary (house rule): the SCORE may cross into a social KG
// triple (encounters/relationships from quotient-space proximity — sanctioned
// by the domain rules); the distance matrices and the coupling plan are
// think-brain-local and never cross any sync surface.
//
// Scope (Issue 743 "explicitly out of scope"): fused GW (needs shared-feature
// costs — none exist across NPCs), entropic/large-n variants (probe sets are
// ≤ 64), TDA/persistent homology (the katgpt-dec TDA lane owns stage-1
// orbits). Uniform weights only.

mod solve;

use solve::SolveCore;

/// Hard size cap for both probe sets (issue scope: quotient probe sets).
pub const GW_MAX: usize = 64;
/// Fixed outer-iteration count. Part of the determinism contract: same inputs
/// ⇒ same iteration schedule ⇒ bit-identical outputs. Planted losses plateau
/// well inside this budget at n ≤ 16 (G2 sweep); 128 is headroom at the cap.
pub const GW_ITERS: usize = 128;
/// Fixed tail projections (row/column normalization passes) after the last
/// power step — drives polytope drift (row sums 1/n, col sums 1/m) to f32
/// noise so the closed-form loss identity holds tightly.
pub const GW_TAIL_PASSES: usize = 4;
/// Score calibration constant for [`gw_score`]: `sigmoid(−β·loss)`. A
/// calibration CHOICE, not learned — consumers wanting a different operating
/// point re-derive from the raw [`gw_loss`] value.
pub const GW_SCORE_BETA: f32 = 4.0;
/// exp clamp for the score logistic: past `β·loss ≥ 88` the score is 0 for
/// any practical purpose (e^88 ≈ 1.7e38) and f32 would lose it anyway.
const SCORE_EXP_CLAMP: f64 = 88.0;
/// Scores below this collapse to exactly 0.0 (f32 underflow region).
const SCORE_ZERO_FLOOR: f64 = 1e-30;

/// Why a solve refused to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GwError {
    /// A probe set exceeded [`GW_MAX`].
    TooLarge {
        len: usize,
    },
    /// A distance matrix was empty, or its first row did not have `rows`
    /// entries (not square).
    NotSquare {
        rows: usize,
    },
    /// A row length did not match the first row's.
    Ragged {
        row: usize,
        expected: usize,
        got: usize,
    },
}

impl core::fmt::Display for GwError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GwError::TooLarge { len } => {
                write!(f, "probe set has {len} points, GW_MAX is {GW_MAX}")
            }
            GwError::NotSquare { rows } => write!(
                f,
                "distance matrix has {rows} rows; must be square and non-empty"
            ),
            GwError::Ragged {
                row,
                expected,
                got,
            } => write!(f, "row {row} has {got} entries, expected {expected}"),
        }
    }
}

impl std::error::Error for GwError {}

/// Preallocated scratch for the solver loop. Size-capped at [`GW_MAX`]² per
/// buffer; construct once, reuse across solves of any (n, m) ≤ [`GW_MAX`].
/// The solve loop touches nothing but this scratch and stack locals — zero
/// steady-state allocation (G4).
#[derive(Clone)]
pub struct GwScratch {
    pub(crate) t: Vec<[f32; GW_MAX]>,
    pub(crate) prod: Vec<[f32; GW_MAX]>,
    pub(crate) buf: Vec<[f32; GW_MAX]>,
    /// Best-coupling snapshot across the multi-start schedule (and run()’s
    /// per-start checkpoint target). Owned by the checkpoint logic — never
    /// used as solver staging.
    pub(crate) best: Vec<[f32; GW_MAX]>,
    /// Cross-start winner store for `solve()`. run() never touches this.
    pub(crate) winner: Vec<[f32; GW_MAX]>,
    /// f64 staging for `D_A·T` — keeps the loss evaluation (and the loop's
    /// products) free of f32 intermediate rounding.
    pub(crate) buf64: Vec<[f64; GW_MAX]>,
    pub(crate) row_a2: Vec<f64>,
    pub(crate) col_b2: Vec<f64>,
    pub(crate) row_sum: Vec<f64>,
    pub(crate) col_sum: Vec<f64>,
}

impl Default for GwScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl GwScratch {
    /// Allocate the scratch buffers (setup-time allocation; the solve path
    /// allocates nothing).
    #[must_use]
    pub fn new() -> Self {
        Self {
            t: vec![[0.0; GW_MAX]; GW_MAX],
            prod: vec![[0.0; GW_MAX]; GW_MAX],
            buf: vec![[0.0; GW_MAX]; GW_MAX],
            best: vec![[0.0; GW_MAX]; GW_MAX],
            winner: vec![[0.0; GW_MAX]; GW_MAX],
            buf64: vec![[0.0; GW_MAX]; GW_MAX],
            row_a2: vec![0.0; GW_MAX],
            col_b2: vec![0.0; GW_MAX],
            row_sum: vec![0.0; GW_MAX],
            col_sum: vec![0.0; GW_MAX],
        }
    }
}

/// GW self-similarity loss of the alignment between two distance matrices.
///
/// The GW quadratic form is evaluated SUM-EXACT for the solver's coupling
/// `T` — `Σ a_ij²·RS_i·RS_j + Σ b_kl²·CS_k·CS_l − 2·⟨T, D_A·T·D_B⟩` with the
/// coupling's actual row/col sums (`RS`, `CS`) — which reduces to the classic
/// `froA2/n² + froB2/m² − 2·⟨T, D_A·T·D_B⟩` on the polytope interior and is
/// algebraically identical to the direct O(n²m²) quadratic form for ANY `T`
/// (pinned by a unit test against that form). `0.0` = structurally isometric;
/// larger = more misaligned. Lower is better. Deterministic: same inputs ⇒
/// bit-identical output.
///
/// Errors: [`GwError::TooLarge`], [`GwError::NotSquare`], [`GwError::Ragged`].
pub fn gw_loss(a: &[&[f32]], b: &[&[f32]], scratch: &mut GwScratch) -> Result<f32, GwError> {
    let core = SolveCore::validate(a, b)?;
    Ok(core.solve(a, b, scratch))
}

/// Convenience wrapper: sigmoid-bounded alignment score
/// `sigmoid(−[`GW_SCORE_BETA`]·loss)` per the house bridge rule (sigmoid,
/// never softmax). Monotone decreasing in [`gw_loss`]; `0.5` at loss 0
/// (perfect structural alignment), → 0 as geometries diverge. Deterministic.
///
/// Errors: same as [`gw_loss`].
pub fn gw_score(a: &[&[f32]], b: &[&[f32]], scratch: &mut GwScratch) -> Result<f32, GwError> {
    let loss = gw_loss(a, b, scratch)?;
    Ok(score_from_loss(loss))
}

/// Sigmoid-bounded score from a precomputed loss (bridge-rule projection;
/// exposed so consumers can project losses they already hold without
/// re-solving). Non-finite loss ⇒ 0.0.
#[must_use]
pub fn score_from_loss(loss: f32) -> f32 {
    if !loss.is_finite() {
        return 0.0;
    }
    let loss = loss.max(0.0);
    let z = (f64::from(GW_SCORE_BETA) * f64::from(loss)).min(SCORE_EXP_CLAMP);
    let s = 1.0 / (1.0 + f64::exp(z));
    if s < SCORE_ZERO_FLOOR {
        0.0
    } else {
        s as f32
    }
}

#[cfg(test)]
mod tests;
