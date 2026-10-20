//! Vocabulary types for the `kron_memory` cell — the pair-addressed
//! scalar-target associative memory (Issue 954, Research 630).
//!
//! Kept in `types.rs` per the house convention (decoupled structs/impls);
//! the cell's operations live in [`crate::kron_memory`] (mod.rs).

/// The write paths' outcome. A write can be applied, elided (exact-zero
/// residual — a true bit-level no-op, the identity-fixed-point property), or
/// refused (zero-norm address vector: the Newton step is undefined there).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    /// The rank-1 correction was applied and the pair entered the registry.
    Applied,
    /// `residual == 0.0` exactly — the write is a no-op by construction and
    /// the state is left bit-identical (G1 hash pin).
    Elided,
    /// A zero-norm address vector (`‖k‖ = 0` or `‖v‖ = 0`): the Newton
    /// denominator vanishes. Nothing was written.
    ZeroAddress,
}

impl WriteOutcome {
    /// True when the state changed (Applied; Elided and ZeroAddress do not
    /// touch the state).
    #[inline]
    pub fn applied(self) -> bool {
        matches!(self, WriteOutcome::Applied)
    }
}

/// Why an explicit-`beta` write was refused by the contraction governor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteRefusal {
    /// `β·‖k‖²·‖v‖² ≤ 0` — a zero-norm address vector, or a non-positive β:
    /// the update cannot contract.
    NoContraction,
    /// `β·‖k‖²·‖v‖² ≥ 2` — the update would overshoot the target (the
    /// residual grows in magnitude instead of shrinking). Refused before any
    /// state change.
    Diverges,
}

impl core::fmt::Display for WriteRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WriteRefusal::NoContraction => {
                write!(f, "no contraction: β·‖k‖²·‖v‖² ≤ 0 (zero address or β ≤ 0)")
            }
            WriteRefusal::Diverges => {
                write!(
                    f,
                    "β·‖k‖²·‖v‖² ≥ 2 — the step diverges (governor interval is (0, 2))"
                )
            }
        }
    }
}

/// Why a snapshot write refused: the write touches more rows than the
/// caller's [`TouchedSnap`] capacity. No state change; re-call with a larger
/// `CAP` (or `CAP = DK` for the always-sufficient bound). Never truncates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapCapacityExceeded {
    pub touched_rows: usize,
    pub capacity: usize,
}

/// Policy for [`crate::kron_memory::KronCell::unlearn_audited`] when the
/// pre-write audit measures collateral above `tau`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollateralPolicy {
    /// Apply nothing: the erase is refused and the state is untouched.
    Refuse,
    /// Apply the erase, then write back every collided pair's old readout
    /// (one closed-form Newton write per collided pair).
    Repair,
}

/// The pre-registered verdict data of one audited unlearn.
///
/// `repair_residual_max` is the MEASURED |residual| each repair write saw at
/// write time — the empirical check on the audit kernel's prediction (the
/// repair's residual is exactly the collateral the audit predicted, within
/// fp). Zero when policy was [`CollateralPolicy::Refuse`] or nothing
/// collided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollateralReport {
    /// The erase was applied (true also for a repaired erase).
    pub applied: bool,
    /// max |Δᵢ| the audit predicted over the registry (0 on an empty registry).
    pub predicted_max: f32,
    /// Registry pairs whose |Δᵢ| exceeded `tau`.
    pub collided: usize,
    /// Pairs actually repaired (0 under `Refuse`).
    pub repaired: usize,
    /// max |residual| measured during repair writes — validates the audit.
    pub repair_residual_max: f32,
}

impl CollateralReport {
    /// The bound actually in force after the operation: predicted max when
    /// nothing was repaired (exact), the repair-measured max otherwise
    /// (empirical — the repair's own residual is the measured collateral the
    /// audit predicted).
    #[inline]
    pub fn bound_in_force(&self) -> f32 {
        if self.repaired > 0 {
            self.repair_residual_max
        } else {
            self.predicted_max
        }
    }
}

/// The streaming marginal-moment tracker behind [`CellHealth`] and
/// [`crate::kron_memory::KronCell::consolidate_ls`].
///
/// Maintains, per applied write `(k, v, c)` with EMA rate `alpha`:
/// - `k_gram = EMA(kkᵀ)` (DK×DK) and `v_gram = EMA(vvᵀ)` (DV×DV) — the
///   marginal second moments whose λ̂_min feed the retention gauge;
/// - `b_moment = EMA(c·k⊗v)` (DK×DV) — the LS cross-moment B of the normal
///   equations `K̄ S V̄ = B` (one rank-1 EMA per write, S-sized — the
///   disclosed cost);
/// - `cross = EMA(k⊗v)` (DK×DV) — the measured independence-violation gauge
///   (`≈ 0` for zero-mean independent marginals);
/// - `c_sq = EMA(c²)` — the target second moment in the norm budget.
///
/// Opt-in at construction (`KronCell::with_health`): tracking roughly doubles
/// the write cost (the cross moment alone is S-sized) and must not tax cells
/// nobody reads gauges from.
pub struct HealthTracker<const DK: usize, const DV: usize> {
    pub(crate) k_gram: [[f32; DK]; DK],
    pub(crate) v_gram: [[f32; DV]; DV],
    pub(crate) b_moment: [[f32; DV]; DK],
    pub(crate) cross: [[f32; DV]; DK],
    pub(crate) c_sq: f32,
    pub(crate) alpha: f32,
    pub(crate) observations: u64,
}

impl<const DK: usize, const DV: usize> HealthTracker<DK, DV> {
    /// `alpha` is the EMA rate in (0, 1]; 1.0 = last-write-wins (no smoothing).
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        assert!(
            alpha > 0.0 && alpha <= 1.0,
            "HealthTracker alpha must be in (0, 1], got {alpha}"
        );
        Self {
            k_gram: [[0.0; DK]; DK],
            v_gram: [[0.0; DV]; DV],
            b_moment: [[0.0; DV]; DK],
            cross: [[0.0; DV]; DK],
            c_sq: 0.0,
            alpha,
            observations: 0,
        }
    }

    /// Fold one applied write into the moments. Called from the applied write
    /// paths only (elided/zero-address writes change no state and observe no
    /// moments — the gauge reads the APPLIED stream).
    #[inline]
    pub(crate) fn observe(&mut self, k: &[f32; DK], v: &[f32; DV], c: f32) {
        let a = self.alpha;
        let keep = 1.0 - a;
        for i in 0..DK {
            let ki = k[i];
            // k_gram row: (1-a)·row + a·ki·k — fused in one pass.
            let kg = &mut self.k_gram[i];
            for j in 0..DK {
                kg[j] = keep * kg[j] + a * ki * k[j];
            }
            // b_moment row + cross row share the (i, j) walk over v.
            let bm = &mut self.b_moment[i];
            let cr = &mut self.cross[i];
            for j in 0..DV {
                let kv = ki * v[j];
                bm[j] = keep * bm[j] + a * c * kv;
                cr[j] = keep * cr[j] + a * kv;
            }
        }
        for i in 0..DV {
            let vi = v[i];
            let vg = &mut self.v_gram[i];
            for j in 0..DV {
                vg[j] = keep * vg[j] + a * vi * v[j];
            }
        }
        self.c_sq = keep * self.c_sq + a * c * c;
        self.observations = self.observations.wrapping_add(1);
    }
}

/// The analytic health gauges (lower-bound heuristics — every field here is a
/// SIGNAL for callers, never a guarantee and never a forced action).
///
/// Caveats carried on the face (Issue 954 verdict round):
/// - `lambda_min_a` factorizes Ā = K̄ ⊗ V̄, exact only under key/value
///   independence; associative streams are correlated by construction —
///   [`CellHealth::independence_violation`] measures the violation.
/// - `lambda_min_*` are tolerance-thresholded on the OBSERVED subspace:
///   directions the stream never excited are dropped, so a rank-deficient
///   stream keeps the gauges finite instead of sending the half-life to ∞.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellHealth {
    /// ‖S‖²_F of the current state.
    pub frobenius_sq: f32,
    /// λ̂_min(K̄) above the tolerance floor; `None` = the observed key
    /// subspace is (numerically) rank-0 — no data or every direction below
    /// tolerance.
    pub lambda_min_k: Option<f32>,
    /// λ̂_min(V̄), same rule.
    pub lambda_min_v: Option<f32>,
    /// λ̂_min(K̄)·λ̂_min(V̄) — the Kronecker eigenstructure product; `Some`
    /// only when both factors are.
    pub lambda_min_a: Option<f32>,
    /// Retention half-life in WRITES: the first-order `ln2 / (β·λ̂_min(Ā))`
    /// — the number of i.i.d. writes over which a steady-state perturbation
    /// contracts by half. `None` when λ̂_min(Ā) is unavailable (rank-deficient
    /// observed subspace) — finite-by-absence, never ∞.
    pub half_life_writes: Option<f32>,
    /// `‖S‖²_F > θ·4·Ê[c²]/λ̂_min(Ā)` — the norm-budget alarm (regime-shift
    /// detector). `None` when the budget denominator is unavailable.
    pub norm_budget_exceeded: Option<bool>,
    /// `‖EMA(k⊗v)‖_F / √(tr K̄ · tr V̄)` — measured dependence between the
    /// key and value marginals (0 for zero-mean independent streams). The
    /// gauge that tells you how much the factorized envelope should be
    /// distrusted on THIS stream.
    pub independence_violation: f32,
}

/// A caller-owned bit-exact pre-write snapshot of the state rows the next
/// write will touch (rows with any nonzero key component).
///
/// `undo_last` (tolerance-pinned) needs no snapshot; `undo_snap` restores
/// from this and is BIT-exact. Capacity is the caller's choice — `CAP = DK`
/// is the always-sufficient bound; a write touching more rows than `CAP`
/// refuses loudly rather than truncating.
#[derive(Debug, Clone)]
pub struct TouchedSnap<const DV: usize, const CAP: usize> {
    row_idx: [u16; CAP],
    rows: [[f32; DV]; CAP],
    len: usize,
}

impl<const DV: usize, const CAP: usize> TouchedSnap<DV, CAP> {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            row_idx: [0; CAP],
            rows: [[0.0; DV]; CAP],
            len: 0,
        }
    }

    /// How many rows the snapshot holds.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when nothing was captured.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The captured row indices (ascending), for caller-side assertions.
    pub fn row_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.row_idx[..self.len].iter().map(|&r| r as usize)
    }

    #[inline]
    pub(crate) fn capture(&mut self, rows: &[[f32; DV]], touched: impl Iterator<Item = usize>) {
        self.len = 0;
        for (slot, i) in self.row_idx.iter_mut().zip(touched) {
            *slot = i as u16;
            self.rows[self.len].copy_from_slice(&rows[i]);
            self.len += 1;
        }
    }

    #[inline]
    pub(crate) fn restore_into(&self, rows: &mut [[f32; DV]]) {
        for s in 0..self.len {
            rows[self.row_idx[s] as usize].copy_from_slice(&self.rows[s]);
        }
    }
}

/// The state-size disclosure (the honest bytes law): the sizing doc states
/// Pod + registry + undo-record + tracker bytes, never "16 KB" alone.
///
/// - state Pod: `DK·DV·4` B (e.g. 16 KB @ 64×64);
/// - pair registry: `N·(DK+DV)·4` B (+ `N·6` B index/seq overhead);
/// - undo record: `(DK+DV)·4` B + 4 B scalar;
/// - health tracker (opt-in): `(DK² + DV² + 2·DK·DV)·4` B + 16 B.
pub const fn kron_state_bytes(dk: usize, dv: usize) -> usize {
    dk * dv * 4
}

pub const fn kron_registry_bytes(n: usize, dk: usize, dv: usize) -> usize {
    n * (dk + dv) * 4
}

pub const fn kron_health_bytes(dk: usize, dv: usize) -> usize {
    (dk * dk + dv * dv + 2 * dk * dv) * 4
}
