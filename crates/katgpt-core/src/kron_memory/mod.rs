//! `kron_memory` — the pair-addressed scalar-target delta-rule cell
//! (Kronecker Delta Rule, Issue 954 / Research 630, arXiv:2610.26834).
//!
//! A new cell type on the associative-memory ladder beside the shipped
//! vector-target rungs (`delta_mem`, LatticeMemory): a **pair-addressed
//! scalar table** `S ∈ R^{DK×DV}` whose readout
//! `read(k, v) = Σ_i k[i]·(Σ_j S[i][j]·v[j])` regresses a SCALAR target `c`
//! on the rank-1 feature `a = k ⊗ v`.
//!
//! # The update law
//!
//! ```text
//! residual r = read(k, v) − c
//! S ← S − β·r·(v kᵀ)          [one rank-1 axpy]
//! β* = 1 / (‖k‖²·‖v‖²)        [the closed-form Newton step — r → 0 in one step]
//! ```
//!
//! The contraction governor refuses any explicit-β write outside
//! `0 < β·‖k‖²·‖v‖² < 2` (inside the interval the residual contracts
//! geometrically; at ≥ 2 it diverges).
//!
//! # Determinism (Issue 940 law)
//!
//! Every dot goes through the sequential fold [`crate::simd::dot_f32_ordered`]
//! and the row fold is a fixed sequential `i`-ascending accumulation
//! (documented as [`KRON_READ_ORDER`]). Identical inputs therefore produce
//! identical residuals, which is what makes the exact-zero **write elision**
//! a true bit-level no-op (the identity-fixed-point property) and the
//! `undo_last` tolerance pin stable.
//!
//! # Interference, audit, and surgical unlearning
//!
//! Interference on a stored pair `(kᵢ, vᵢ)` from a write `(k, v, β, c)` is
//! exactly `Δᵢ = −β·r·(vᵢᵀv)·(kᵀkᵢ)` — the PRODUCT of value overlap and key
//! overlap; either factor ≈ 0 protects the pair, and the damage is computable
//! BEFORE the write ([`KronCell::collateral_audit`], into caller scratch,
//! zero alloc). What ONLY the scalar-target member adds is the **double
//! gating** — erase-with-audit alone is not Kron-exclusive (a vector-target
//! Newton erase admits the same audit pattern); G0's baselines exist so the
//! headline claim stays falsifiable (GOAT bench `kron_memory_goat`).
//!
//! # Size disclosure (never "16 KB" alone)
//!
//! State Pod `DK·DV·4` B — 16 KB @ 64×64 — plus the pair registry
//! `N·(DK+DV)·4` B, the undo record `(DK+DV)·4` B, and (opt-in) the health
//! tracker `(DK²+DV²+2·DK·DV)·4` B. See [`types::kron_state_bytes`] and
//! siblings; `.docs/03_memory/kron_memory.md` carries the full table.

#[cfg(test)]
mod tests;
mod types;

pub use types::{
    CellHealth, CollateralPolicy, CollateralReport, HealthTracker, SnapCapacityExceeded,
    TouchedSnap, WriteOutcome, WriteRefusal, kron_health_bytes, kron_registry_bytes,
    kron_state_bytes,
};

use crate::simd::dot_f32_ordered;

#[cfg(feature = "kron_chunk_wy")]
mod chunk_wy;
#[cfg(feature = "kron_chunk_wy")]
pub use chunk_wy::{ChunkItem, ChunkRefusal, write_chunk_wy};

/// The pinned read reduction order (Issue 940 determinism law): per-row
/// `dot_f32_ordered` over the value index `j`, then a fixed sequential
/// accumulation over the key index `i` (ascending). Changing either fold
/// order changes committed readouts — bump this name and re-pin the
/// tolerance-sensitive tests if the order ever moves.
pub const KRON_READ_ORDER: &str = "v1: per-row ordered dot (j ascending), sequential i fold";

/// The undo tolerance pin: max per-element drift of a write + `undo_last`
/// round trip. Two f32 roundings per element on O(1)-magnitude states —
/// pinned by `undo_last_within_pinned_epsilon` (G1); raise only with a new
/// measurement.
pub const EPS_UNDO: f32 = 1e-5;

/// One stored pair in the registry: the ADDRESS only (key + value). Targets
/// are not stored — a pair's "value" is whatever the state currently reads.
pub(crate) struct RegistryPair<const DK: usize, const DV: usize> {
    k: [f32; DK],
    v: [f32; DV],
}

/// Fixed-capacity ring of written `(kᵢ, vᵢ)` pairs — the collateral audit's
/// enumeration source. Eviction is oldest-first (the ring's write head); the
/// audit cost is `O(N·(d_k+d_v))` per call.
pub struct PairRegistry<const N: usize, const DK: usize, const DV: usize> {
    pairs: [RegistryPair<DK, DV>; N],
    head: usize,
    len: usize,
}

impl<const N: usize, const DK: usize, const DV: usize> PairRegistry<N, DK, DV> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            // `RegistryPair` is plain data; `from_fn` builds the array
            // without any `Default`-bound dance.
            pairs: core::array::from_fn(|_| RegistryPair {
                k: [0.0; DK],
                v: [0.0; DV],
            }),
            head: 0,
            len: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Push a pair, evicting the oldest when full. Returns the slot it
    /// landed in (the audit's `u16` handle).
    #[inline]
    pub(crate) fn push(&mut self, k: &[f32; DK], v: &[f32; DV]) -> u16 {
        let slot = self.head;
        self.pairs[slot].k.copy_from_slice(k);
        self.pairs[slot].v.copy_from_slice(v);
        self.head = (self.head + 1) % N;
        if self.len < N {
            self.len += 1;
        }
        slot as u16
    }

    /// The stored pair at `slot` (valid for `slot < len()` until the next
    /// eviction wraps it).
    #[inline]
    pub fn pair(&self, slot: u16) -> Option<(&[f32; DK], &[f32; DV])> {
        if (slot as usize) < self.len {
            let p = &self.pairs[slot as usize];
            Some((&p.k, &p.v))
        } else {
            None
        }
    }

    /// The live slots in slot order (NOT write order — slot order is the
    /// fixed enumeration order the audit and its tests pin).
    pub fn slots(&self) -> impl Iterator<Item = u16> + '_ {
        0..self.len as u16
    }

    /// Clear (used by consolidation, which refits the state wholesale).
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    #[inline]
    pub(crate) fn pairs_iter(&self) -> impl Iterator<Item = &RegistryPair<DK, DV>> + '_ {
        self.pairs[..self.len].iter()
    }
}

impl<const N: usize, const DK: usize, const DV: usize> Default for PairRegistry<N, DK, DV> {
    fn default() -> Self {
        Self::new()
    }
}

/// The rank-1 inverse record behind [`KronCell::undo_last`]: the last
/// APPLIED write's `(β, r, k, v)` — applying `S += β·r·v kᵀ` undoes it to
/// within [`EPS_UNDO`] (NOT bit-exact; [`KronCell::write_snap`] +
/// [`KronCell::undo_snap`] is the bit-exact path).
struct UndoRecord<const DK: usize, const DV: usize> {
    beta: f32,
    residual: f32,
    k: [f32; DK],
    v: [f32; DV],
}

/// The scalar-target delta-rule cell.
///
/// `S: [[f32; DV]; DK]` — `S[i][j]`, `i` indexes the key axis, `j` the value
/// axis. All hot paths are allocation-free (G4): the audit writes into
/// caller scratch, the undo snapshot lives in the caller's [`TouchedSnap`].
pub struct KronCell<const DK: usize, const DV: usize, const N: usize> {
    s: [[f32; DV]; DK],
    registry: PairRegistry<N, DK, DV>,
    undo: Option<UndoRecord<DK, DV>>,
    /// EMA of |residual| over applied writes — the signal-only uncertainty
    /// heuristic (the analytic CI was dropped at G5: it did not beat the
    /// conformal-naive floor; see the Bench 951 record).
    residual_ema: f32,
    health: Option<HealthTracker<DK, DV>>,
    writes_applied: u64,
}

impl<const DK: usize, const DV: usize, const N: usize> KronCell<DK, DV, N> {
    /// An empty cell (zero state, empty registry, no health tracking).
    #[must_use]
    pub fn new() -> Self {
        Self {
            s: [[0.0; DV]; DK],
            registry: PairRegistry::new(),
            undo: None,
            residual_ema: 0.0,
            health: None,
            writes_applied: 0,
        }
    }

    /// An empty cell with health tracking at EMA rate `alpha` — enables
    /// [`Self::health`] and [`Self::consolidate_ls`] at the disclosed
    /// per-write cost (roughly 2×: the S-sized cross moment dominates).
    #[must_use]
    pub fn with_health(alpha: f32) -> Self {
        let mut cell = Self::new();
        cell.health = Some(HealthTracker::new(alpha));
        cell
    }

    /// The pinned read: `Σ_i k[i]·dot_f32_ordered(S[i], v)` — see
    /// [`KRON_READ_ORDER`].
    #[must_use]
    pub fn read(&self, k: &[f32; DK], v: &[f32; DV]) -> f32 {
        let mut acc = 0.0f32;
        for (row, &ki) in self.s.iter().zip(k.iter()) {
            acc += ki * dot_f32_ordered(row, v);
        }
        acc
    }

    /// ‖S‖²_F (Frobenius, squared) — sequential fold in the pinned order.
    #[must_use]
    pub fn frobenius_sq(&self) -> f32 {
        let mut acc = 0.0f32;
        for row in &self.s {
            for &x in row {
                acc += x * x;
            }
        }
        acc
    }

    /// The Newton write: `β* = 1/(‖k‖²·‖v‖²)` zeroes the residual in one
    /// step (exact in real arithmetic). Elides on an exact-zero residual;
    /// reports [`WriteOutcome::ZeroAddress`] for a zero-norm address vector
    /// (the Newton denominator vanishes — nothing is written).
    pub fn write_newton(&mut self, k: &[f32; DK], v: &[f32; DV], target: f32) -> WriteOutcome {
        let sk = dot_f32_ordered(k, k);
        let sv = dot_f32_ordered(v, v);
        let denom = sk * sv;
        if denom <= 0.0 || denom.is_nan() {
            return WriteOutcome::ZeroAddress;
        }
        let residual = self.read(k, v) - target;
        if residual == 0.0 {
            return WriteOutcome::Elided;
        }
        let beta = 1.0 / denom;
        self.apply_rank1(k, v, beta, residual, target);
        WriteOutcome::Applied
    }

    /// Explicit-`beta` write under the contraction governor
    /// `0 < β·‖k‖²·‖v‖² < 2`. Same elision and zero-address posture as
    /// [`Self::write_newton`].
    ///
    /// # Errors
    /// [`WriteRefusal::NoContraction`] (zero address / non-positive β) or
    /// [`WriteRefusal::Diverges`] (governor upper edge). The state is never
    /// touched on refusal.
    pub fn write(
        &mut self,
        k: &[f32; DK],
        v: &[f32; DV],
        target: f32,
        beta: f32,
    ) -> Result<WriteOutcome, WriteRefusal> {
        let sk = dot_f32_ordered(k, k);
        let sv = dot_f32_ordered(v, v);
        let contraction = beta * sk * sv;
        if contraction <= 0.0 || contraction.is_nan() {
            return Err(WriteRefusal::NoContraction);
        }
        if contraction >= 2.0 {
            return Err(WriteRefusal::Diverges);
        }
        let residual = self.read(k, v) - target;
        if residual == 0.0 {
            return Ok(WriteOutcome::Elided);
        }
        self.apply_rank1(k, v, beta, residual, target);
        Ok(WriteOutcome::Applied)
    }

    /// `write` (governor + elision) that first captures the touched rows
    /// into `snap` for a later bit-exact [`Self::undo_snap`]. A refused or
    /// elided write burns no snapshot; an insufficient snapshot capacity
    /// refuses loudly (never truncates) with the state untouched. `CAP` is
    /// the caller's memory/coverage trade: dense keys need `CAP = DK`, a
    /// sparse designed-address stream can run `CAP ≪ DK`.
    ///
    /// # Errors
    /// [`WriteSnapError::Governor`] — the contraction governor refused;
    /// [`WriteSnapError::SnapTooSmall`] — the write touches more rows than
    /// `CAP`. The state is untouched in both arms.
    pub fn write_snap<const CAP: usize>(
        &mut self,
        k: &[f32; DK],
        v: &[f32; DV],
        target: f32,
        beta: f32,
        snap: &mut TouchedSnap<DV, CAP>,
    ) -> Result<WriteOutcome, WriteSnapError> {
        let sk = dot_f32_ordered(k, k);
        let sv = dot_f32_ordered(v, v);
        let contraction = beta * sk * sv;
        if contraction <= 0.0 || contraction.is_nan() {
            return Err(WriteSnapError::Governor(WriteRefusal::NoContraction));
        }
        if contraction >= 2.0 {
            return Err(WriteSnapError::Governor(WriteRefusal::Diverges));
        }
        let residual = self.read(k, v) - target;
        if residual == 0.0 {
            return Ok(WriteOutcome::Elided);
        }
        let count = k.iter().filter(|&&ki| ki != 0.0).count();
        if count > CAP {
            return Err(WriteSnapError::SnapTooSmall(SnapCapacityExceeded {
                touched_rows: count,
                capacity: CAP,
            }));
        }
        let touched = self
            .s
            .iter()
            .zip(k.iter())
            .enumerate()
            .filter(|&(_, (_, &ki))| ki != 0.0)
            .map(|(i, _)| i);
        snap.capture(&self.s, touched);
        self.apply_rank1(k, v, beta, residual, target);
        Ok(WriteOutcome::Applied)
    }

    /// Bit-exact undo: restore the touched rows from a snapshot captured by
    /// [`Self::write_snap`].
    pub fn undo_snap<const CAP: usize>(&mut self, snap: &TouchedSnap<DV, CAP>) {
        snap.restore_into(&mut self.s);
    }

    /// Tolerance-pinned undo of the last APPLIED write: `S += β·r·v kᵀ`.
    /// The fp round-trip is NOT bit-exact — element drift is pinned by
    /// [`EPS_UNDO`]; returns false when there is nothing to undo.
    pub fn undo_last(&mut self) -> bool {
        match self.undo.take() {
            Some(rec) => {
                let step_scale = rec.beta * rec.residual;
                for (row, &ki) in self.s.iter_mut().zip(rec.k.iter()) {
                    if ki == 0.0 {
                        continue;
                    }
                    let step = step_scale * ki;
                    for (s, &vj) in row.iter_mut().zip(rec.v.iter()) {
                        *s += step * vj;
                    }
                }
                true
            }
            None => false,
        }
    }

    /// The shared applied tail: rank-1 axpy + registry push + undo record +
    /// residual EMA + health observation. All hot paths funnel through here.
    fn apply_rank1(&mut self, k: &[f32; DK], v: &[f32; DV], beta: f32, residual: f32, target: f32) {
        debug_assert!(residual.is_finite() && beta.is_finite());
        let step_scale = -beta * residual;
        for (row, &ki) in self.s.iter_mut().zip(k.iter()) {
            if ki == 0.0 {
                continue; // the row cannot move — skip the full row axpy
            }
            let step = step_scale * ki;
            for (s, &vj) in row.iter_mut().zip(v.iter()) {
                *s += step * vj;
            }
        }
        self.registry.push(k, v);
        self.undo = Some(UndoRecord {
            beta,
            residual,
            k: *k,
            v: *v,
        });
        // EMA(0.25) of |residual| — a slow scale tracker, not an interval.
        self.residual_ema = 0.75 * self.residual_ema + 0.25 * residual.abs();
        if let Some(h) = self.health.as_mut() {
            h.observe(k, v, target);
        }
        self.writes_applied = self.writes_applied.wrapping_add(1);
    }

    /// Pre-write interference audit over the registry (E2 kernel):
    /// `Δᵢ = −β·r·(vᵢᵀv)·(kᵀkᵢ)` per stored pair, where
    /// `r = read(k, v) − target`.
    ///
    /// Writes `(slot, Δᵢ)` rows into `scratch` and returns the used subslice
    /// PLUS the registry length — when `written < registry_len` the caller's
    /// scratch was too small and the audit is TRUNCATED (size scratch
    /// `≥ registry.capacity()`; the count is the loud tell, never a silent
    /// cap). Zero allocation; cell-local enumeration only (no global audit).
    #[must_use]
    pub fn collateral_audit<'s>(
        &self,
        k: &[f32; DK],
        v: &[f32; DV],
        beta: f32,
        target: f32,
        scratch: &'s mut [(u16, f32)],
    ) -> (&'s [(u16, f32)], usize) {
        let residual = self.read(k, v) - target;
        let registry_len = self.registry.len();
        let mut written = 0usize;
        if residual != 0.0 {
            let scale = -beta * residual;
            for (slot, pair) in self.registry.slots().zip(self.registry.pairs_iter()) {
                if written >= scratch.len() {
                    break;
                }
                let kv = dot_f32_ordered(&pair.v, v);
                let kk = dot_f32_ordered(k, &pair.k);
                scratch[written] = (slot, scale * kv * kk);
                written += 1;
            }
        }
        (&scratch[..written], registry_len)
    }

    /// max |Δᵢ| the audit would predict (the scan without the scratch walk —
    /// convenience for callers that only need the bound).
    #[must_use]
    pub fn collateral_max(&self, k: &[f32; DK], v: &[f32; DV], beta: f32, target: f32) -> f32 {
        let residual = self.read(k, v) - target;
        if residual == 0.0 {
            return 0.0;
        }
        let scale = -(beta * residual).abs();
        let mut max = 0.0f32;
        for pair in self.registry.pairs_iter() {
            let d = scale * dot_f32_ordered(&pair.v, v).abs() * dot_f32_ordered(k, &pair.k).abs();
            if d > max {
                max = d;
            }
        }
        max
    }

    /// Audited surgical unlearning (E3 — the headline): a `target = 0`
    /// Newton write that first audits its own collateral. At or below `tau`
    /// it applies plainly. Above `tau`: [`CollateralPolicy::Refuse`] leaves
    /// the state untouched; [`CollateralPolicy::Repair`] applies the erase
    /// and writes back every collided pair's old readout with one
    /// closed-form Newton write each — the repair's measured |residual| is
    /// the empirical validation of the audit kernel (it lands in
    /// [`CollateralReport::repair_residual_max`]).
    pub fn unlearn_audited(
        &mut self,
        k: &[f32; DK],
        v: &[f32; DV],
        tau: f32,
        policy: CollateralPolicy,
        scratch: &mut [(u16, f32)],
    ) -> CollateralReport {
        let beta_star = newton_beta(k, v);
        let (rows, _registry_len) = self.collateral_audit(k, v, beta_star, 0.0, scratch);
        let mut predicted_max = 0.0f32;
        let mut collided: Vec<u16> = Vec::new();
        for &(slot, d) in rows {
            let ad = d.abs();
            if ad > predicted_max {
                predicted_max = ad;
            }
            if ad > tau {
                collided.push(slot);
            }
        }

        if predicted_max > tau && policy == CollateralPolicy::Refuse {
            return CollateralReport {
                applied: false,
                predicted_max,
                collided: collided.len(),
                repaired: 0,
                repair_residual_max: 0.0,
            };
        }

        // Capture the collided pairs' readouts BEFORE the erase (the repair
        // targets). The addresses are copied out first because the erase's
        // own registry push can evict slot 0 on a full ring.
        let pre: Vec<([f32; DK], [f32; DV], f32)> = collided
            .iter()
            .filter_map(|&s| self.registry.pair(s))
            .map(|(pk, pv)| (*pk, *pv, self.read(pk, pv)))
            .collect();

        // The erase itself (c = 0 Newton write).
        let applied = self.write_newton(k, v, 0.0);
        debug_assert!(
            applied.applied() || predicted_max == 0.0,
            "audit predicted collateral but the elision path fired"
        );

        // Repairs: one Newton write per collided pair, target = its
        // pre-erase readout. The repair's residual IS the measured
        // collateral — the audit kernel's empirical validation.
        let mut repair_residual_max = 0.0f32;
        let mut repaired = 0usize;
        for (pk, pv, pre_read) in &pre {
            let measured = (self.read(pk, pv) - *pre_read).abs();
            if measured > repair_residual_max {
                repair_residual_max = measured;
            }
            self.write_newton(pk, pv, *pre_read);
            repaired += 1;
        }

        CollateralReport {
            applied: applied.applied(),
            predicted_max,
            collided: collided.len(),
            repaired,
            repair_residual_max,
        }
    }

    /// The signal-only uncertainty heuristic: EMA of |residual| over applied
    /// writes (the analytic CI was dropped at G5 — it did not beat the
    /// conformal-naive floor; Bench 951). A SIGNAL for callers, never an
    /// interval.
    #[must_use]
    pub fn residual_signal(&self) -> f32 {
        self.residual_ema
    }

    /// Applied-write count (elided and refused writes excluded).
    #[must_use]
    pub fn writes_applied(&self) -> u64 {
        self.writes_applied
    }

    /// The pair registry (the audit's enumeration source).
    #[must_use]
    pub fn registry(&self) -> &PairRegistry<N, DK, DV> {
        &self.registry
    }

    /// Mutable state rows — the chunk kernel's fused-application walk
    /// (pub(crate): the chunk lane is the only external writer).
    #[inline]
    #[cfg(feature = "kron_chunk_wy")]
    pub(crate) fn state_rows_mut(&mut self) -> &mut [[f32; DV]; DK] {
        &mut self.s
    }

    /// Immutable state rows (tests and callers that fingerprint the state).
    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn state_rows(&self) -> &[[f32; DV]; DK] {
        &self.s
    }

    /// Chunk-path bookkeeping for one applied write: registry push, residual
    /// EMA, health moments, applied count. The composite chunk is NOT
    /// rank-1-invertible, so the undo record is cleared (undo_last returns
    /// false until the next single write) — the documented posture.
    #[cfg(feature = "kron_chunk_wy")]
    pub(crate) fn note_chunk_applied(
        &mut self,
        k: &[f32; DK],
        v: &[f32; DV],
        residual: f32,
        target: f32,
    ) {
        self.registry.push(k, v);
        self.undo = None;
        self.residual_ema = 0.75 * self.residual_ema + 0.25 * residual.abs();
        if let Some(h) = self.health.as_mut() {
            h.observe(k, v, target);
        }
        self.writes_applied = self.writes_applied.wrapping_add(1);
    }

    /// The analytic health gauges. `beta` is the β the caller's write path
    /// is running (the half-life is β-dependent); `theta` scales the
    /// norm-budget alarm. Both gauges are documented lower-bound heuristics
    /// — see [`CellHealth`]. `None` without `with_health`.
    ///
    /// The eigensolve reuses `eig_scratch` across calls; allocation happens
    /// in the solver's scratch growth — health is a read-side gauge, never a
    /// hot path.
    #[must_use]
    pub fn health(&self, beta: f32, theta: f32) -> Option<CellHealth> {
        let h = self.health.as_ref()?;
        let fro = self.frobenius_sq();
        let lam_k = min_eig_thresholded(&h.k_gram);
        let lam_v = min_eig_thresholded(&h.v_gram);
        let lam_a = match (lam_k, lam_v) {
            (Some(a), Some(b)) => Some(a * b),
            _ => None,
        };
        let half_life = lam_a
            .filter(|&l| l > 0.0 && beta > 0.0)
            .map(|l| core::f32::consts::LN_2 / (beta * l));
        let budget = lam_a.map(|l| theta * 4.0 * h.c_sq / l);
        let norm_budget_exceeded = budget.map(|b| fro > b);

        // Measured independence violation: ‖EMA(k⊗v)‖_F / √(tr K̄ · tr V̄).
        let mut cross_f = 0.0f32;
        for row in &h.cross {
            for &x in row {
                cross_f += x * x;
            }
        }
        let tr_k: f32 = (0..DK).map(|i| h.k_gram[i][i]).sum();
        let tr_v: f32 = (0..DV).map(|i| h.v_gram[i][i]).sum();
        let denom = (tr_k * tr_v).sqrt();
        let violation = if denom > 0.0 {
            cross_f.sqrt() / denom
        } else {
            0.0
        };

        Some(CellHealth {
            frobenius_sq: fro,
            lambda_min_k: lam_k,
            lambda_min_v: lam_v,
            lambda_min_a: lam_a,
            half_life_writes: half_life,
            norm_budget_exceeded,
            independence_violation: violation,
        })
    }

    /// LS-snap consolidation (E6): refit `S` to the exact recency-weighted
    /// least-squares solution of the tracked stream — `S := K̄⁻¹ B V̄⁻¹` over
    /// the marginal Grams and the cross moment B — resetting the accumulated
    /// O(β) write jitter.
    ///
    /// Both marginal solves are regularized by `ridge ≥ 0` (Tikhonov, lifted
    /// eigenvalues); rank-deficient streams NEED a positive ridge to stay
    /// finite. The solve is exact LS regardless of independence; the
    /// RETENTION envelope the gauges report stays independence-conditional.
    ///
    /// Invalidates the undo record (a refit is not a rank-1 inverse) and
    /// clears the registry (the addresses stay valid vocabulary, but their
    /// readouts are refit — the audit re-enumerates from future writes).
    ///
    /// # Errors
    /// [`ConsolidateError::NoMoments`] without `with_health`;
    /// [`ConsolidateError::Singular`] if the regularized refit still produces
    /// a non-finite entry (pathological input — the state is untouched).
    pub fn consolidate_ls(&mut self, ridge: f32) -> Result<(), ConsolidateError> {
        let Some(h) = self.health.as_ref() else {
            return Err(ConsolidateError::NoMoments);
        };
        let ridge = if ridge.is_finite() && ridge > 0.0 {
            ridge
        } else {
            0.0
        };

        // Eigendecompose both marginals in f64 (the shipped solver's domain).
        // K̄ = Σ_a λₐ wₐ wₐᵀ (eigenvector a is COLUMN a of the returned
        // vecs — component i of wₐ is vecs[i·D + a]; pinned by the solver's
        // own eigenpair test).
        let (qk, lk) = eig_decompose_thresholded(&h.k_gram, ridge);
        let (qv, lv) = eig_decompose_thresholded(&h.v_gram, ridge);

        // X = K̄⁻¹ B = Σ_a (1/λₐ)·wₐ·(wₐᵀ B)   (DK×DV)
        let mut x = [[0.0f64; DV]; DK];
        for a in 0..DK {
            // wb[j] = (wₐᵀ B)[j] = Σ_b wₐ[b]·B[b][j]
            let mut wb = [0.0f64; DV];
            for (b, row) in h.b_moment.iter().enumerate() {
                let wab = qk[b * DK + a];
                if wab != 0.0 {
                    for (j, &bv) in row.iter().enumerate() {
                        wb[j] += wab * bv as f64;
                    }
                }
            }
            let inv = 1.0 / lk[a];
            for i in 0..DK {
                let wai = qk[i * DK + a] * inv;
                if wai != 0.0 {
                    for j in 0..DV {
                        x[i][j] += wai * wb[j];
                    }
                }
            }
        }

        // S = X·V̄⁻¹ = Σ_a (1/μₐ)·(X·uₐ)·uₐᵀ   (DK×DV)
        let mut new_s = [[0.0f32; DV]; DK];
        for a in 0..DV {
            // xu[i] = (X·uₐ)[i] = Σ_m X[i][m]·uₐ[m]
            let mut xu = [0.0f64; DK];
            for i in 0..DK {
                let mut acc = 0.0f64;
                for m in 0..DV {
                    acc += x[i][m] * qv[m * DV + a];
                }
                xu[i] = acc;
            }
            let inv = 1.0 / lv[a];
            for i in 0..DK {
                let coef = xu[i] * inv;
                if coef == 0.0 {
                    continue;
                }
                for j in 0..DV {
                    new_s[i][j] += (coef * qv[j * DV + a]) as f32;
                }
            }
        }
        for row in &new_s {
            for &val in row {
                if !val.is_finite() {
                    return Err(ConsolidateError::Singular);
                }
            }
        }

        self.s = new_s;
        self.undo = None;
        self.registry.clear();
        Ok(())
    }
}

impl<const DK: usize, const DV: usize, const N: usize> Default for KronCell<DK, DV, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Why a snapshot write refused: the contraction governor, or an
/// insufficient snapshot capacity (unreachable at `CAP = DK`, kept for the
/// type's generality — the state is untouched either way).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteSnapError {
    Governor(WriteRefusal),
    SnapTooSmall(SnapCapacityExceeded),
}

impl core::fmt::Display for WriteSnapError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WriteSnapError::Governor(r) => core::fmt::Display::fmt(r, f),
            WriteSnapError::SnapTooSmall(e) => write!(
                f,
                "snapshot capacity {} < {} touched rows",
                e.capacity, e.touched_rows
            ),
        }
    }
}

/// Consolidation refusals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsolidateError {
    /// Health tracking was never enabled — the marginal moments do not exist.
    NoMoments,
    /// The regularized refit produced a non-finite entry; the state is
    /// untouched. Raise the ridge or investigate the stream.
    Singular,
}

impl core::fmt::Display for ConsolidateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ConsolidateError::NoMoments => write!(
                f,
                "consolidate_ls requires with_health (no tracked moments)"
            ),
            ConsolidateError::Singular => write!(
                f,
                "consolidate_ls refit non-finite even with ridge — state untouched"
            ),
        }
    }
}

/// The Newton step size for an address pair (0.0 on a zero-norm address).
#[inline]
#[must_use]
pub fn newton_beta<const DK: usize, const DV: usize>(k: &[f32; DK], v: &[f32; DV]) -> f32 {
    let sk = dot_f32_ordered(k, k);
    let sv = dot_f32_ordered(v, v);
    if sk * sv > 0.0 { 1.0 / (sk * sv) } else { 0.0 }
}

/// λ̂_min of a symmetric Gram above the tolerance floor (the observed
/// subspace rule): eigenvalues ≤ 1e-6·λ_max are dropped as numerically
/// unexcited directions; `None` when nothing survives (no data, or every
/// direction below tolerance) — the gauge stays finite by absence, never ∞.
fn min_eig_thresholded<const D: usize>(gram: &[[f32; D]; D]) -> Option<f32> {
    let (vals, _) = eig_raw(gram);
    let max = vals.iter().cloned().fold(0.0f64, f64::max);
    if max <= 0.0 {
        return None;
    }
    let floor = max * 1e-6;
    vals.into_iter()
        .filter(|&l| l > floor)
        .map(|l| l as f32)
        .min_by(|a, b| a.total_cmp(b))
}

/// Eigendecomposition for the consolidation solve: f64 symmetric eig + ridge
/// on the spectrum (eigenvalues below `ridge` lift to `ridge`, keeping the
/// solve finite on rank-deficient streams). With `ridge == 0` and a singular
/// Gram the floor clamps to a tiny positive scale (documented: pass
/// `ridge > 0` for the general correlated case). Returns
/// `(eigvecs column-major, eigenvalues)`.
fn eig_decompose_thresholded<const D: usize>(
    gram: &[[f32; D]; D],
    ridge: f32,
) -> (Vec<f64>, Vec<f64>) {
    let (mut vals, vecs) = eig_raw(gram);
    let mut floor = ridge as f64;
    if floor <= 0.0 {
        // Smallest positive eigenvalue scale, so a singular Gram with
        // ridge == 0 cannot divide by zero (and cannot pass `Singular`
        // unnoticed — the refit is then garbage-but-finite, which the G1
        // rank-deficient arm catches by comparing against the ridge path).
        floor = f64::MIN_POSITIVE * 16.0;
    }
    for v in vals.iter_mut() {
        if *v < floor {
            *v = floor;
        }
    }
    (vecs, vals)
}

/// The shared f32→f64 conversion + `symmetric_eig` call. Returns
/// `(eigenvalues, eigvecs column-major)`.
fn eig_raw<const D: usize>(gram: &[[f32; D]; D]) -> (Vec<f64>, Vec<f64>) {
    use crate::linalg::symmetric_eig::{SymmetricEigScratch, symmetric_eig};
    let n = D;
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = gram[i][j] as f64;
        }
    }
    let mut vals = vec![0.0f64; n];
    let mut vecs = vec![0.0f64; n * n];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut vals, &mut vecs, &a, &mut scratch, n, 200);
    (vals, vecs)
}
