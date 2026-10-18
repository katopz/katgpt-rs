//! Calibration staleness at the freeze/thaw seam (Issue 841 §B-3).
//!
//! # The defect this refuses
//!
//! A calibration head — a Platt temperature/bias pair, a per-head attention
//! scale table, a conformal residual pool — is fitted against ONE set of
//! weights. [`AGENTS.md`'s modelless mandate](../../../AGENTS.md) permits
//! exactly three runtime weight mutations, and the first of them is
//! **freeze/thaw**: swapping a frozen snapshot. The moment that swap lands,
//! every attached calibration is describing weights that are no longer there.
//!
//! Nothing in the stack noticed. `SigmoidGateCalibrator::apply` returns an
//! `f32`; `StaticCalTable::get_scale` returns an `f32`; a conformal interval
//! comes back with its `alpha` intact. A stale calibration does not produce a
//! NaN, an error, or an implausible number — it produces a **confident wrong
//! one**, which is the silent direction and the reason this is a refusal
//! rather than a warning.
//!
//! The doctrine is already written in this repo, one domain over:
//! `katgpt_dec::types::invalidate_coboundary_cache` — *"every mutation path
//! invalidates every topology-derived invariant"*. A calibration is a
//! weights-derived invariant. The seam is the same.
//!
//! # The rule
//!
//! [`SnapshotBound<T>`] binds a calibration to the [`SnapshotId`] it was
//! fitted under and will not hand it back once that identity moves:
//! [`SnapshotBound::get`] returns `None`, and warns **once per binding**.
//!
//! ⛔ `None` is the whole point. An API that returned a *default* calibration
//! on staleness (the `StreamingTauCalibrator` cold-start shape — fall back to
//! `DEFAULT_TAU_LO`) is right for a cold start and wrong here: a cold start
//! has no fitted claim to be wrong about, and a stale head does. The caller
//! must decide to refit, abstain, or run uncalibrated. This primitive refuses
//! to decide that for them, which is the `CalibratedPolicy::fell_back`
//! precedent — *"the caller is told, not protected by accident"*.
//!
//! # The predicate
//!
//! Generalised from `DefaultMpiRouterSnapshotHook::cache_valid`
//! (`katgpt-spectral`), the one shipped version-keyed invalidation at a swap
//! boundary, whose two rules are both load-bearing and are kept verbatim:
//!
//! 1. **`version == 0` is never fresh.** A caller that has not wired a
//!    generation counter gets [`Staleness::Unversioned`] and `None`, not a
//!    plausible score. The unwired caller is exactly the one this exists for,
//!    so the default must be refusal — a zero that meant "fine" would make
//!    adoption look like a no-op.
//! 2. **The commitment is re-checked even when the version matches.** That
//!    catches the caller bug the version alone cannot see: weights swapped,
//!    generation not bumped. It is reported as its own
//!    [`Staleness::CommitmentMoved`] rather than pooled into `VersionMoved`,
//!    because the two need opposite repairs — one is a stale calibration, the
//!    other is a broken bump site.
//!
//! ⚠ **What this does NOT claim.** It does not detect a calibration that has
//! gone stale for a reason that is not a snapshot swap (distribution drift,
//! a changed prompt mix, a tokenizer change). Those are a different class and
//! this returning `Some` is not evidence of freshness — only that the
//! *snapshot identity* is unchanged. It also cannot invalidate a calibration
//! somebody read out of the wrapper with [`SnapshotBound::peek_unchecked`];
//! that method is named to be greppable for exactly that reason.
//!
//! Zero allocation on every path. Opt-in (`calibration_staleness`).
//!
//! Calibration staleness at the freeze/thaw seam (Issue 841 §B-3) — a
//! snapshot swap invalidates every attached calibration head; `SnapshotBound`
//! returns `None` + one loud warning instead of a stale plausible score.
//! Opt-in (`calibration_staleness`).

use std::sync::atomic::{AtomicBool, Ordering};

/// Identity of the frozen snapshot a calibration was fitted against.
///
/// Two independent fields, because they fail differently — see the module
/// docs. `version` is the **generation ordinal** (the
/// `FuncAttnWeightsSnapshot::version` / `MicroRecurrentKernelSnapshot::version`
/// convention: the contents ordinal, deliberately NOT part of the hash), and
/// `commitment` is the BLAKE3 of the weights themselves.
///
/// ⛔ Do **not** populate `version` from a freeze envelope's `version` field.
/// `SWTF_VERSION` / `SDBF_VERSION` are **format** versions — they are `1` for
/// every snapshot ever written, so a `SnapshotId` built from one is constant
/// and this whole guard silently degrades to the commitment check alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotId {
    /// Snapshot generation ordinal. `0` means *unversioned* and is never fresh.
    pub version: u64,
    /// BLAKE3 commitment over the snapshot's weights.
    pub commitment: [u8; 32],
}

impl SnapshotId {
    /// The identity a caller that has wired nothing has. Never fresh.
    pub const UNVERSIONED: Self = Self {
        version: 0,
        commitment: [0u8; 32],
    };

    /// Bind a generation ordinal to a weight commitment.
    #[inline]
    #[must_use]
    pub const fn new(version: u64, commitment: [u8; 32]) -> Self {
        Self {
            version,
            commitment,
        }
    }

    /// `true` when this identity carries no generation ordinal.
    #[inline]
    #[must_use]
    pub const fn is_unversioned(&self) -> bool {
        self.version == 0
    }
}

impl Default for SnapshotId {
    #[inline]
    fn default() -> Self {
        Self::UNVERSIONED
    }
}

/// A frozen snapshot that can NAME its own identity.
///
/// The supply side of the freeze/thaw seam, and it has to exist before any
/// consumer can be wired: [`SnapshotBound`] refuses by default
/// ([`SnapshotId::UNVERSIONED`] is never fresh), so a caller with no way to
/// obtain a real [`SnapshotId`] gets `None` on every call and reads the guard
/// as a no-op rather than as a rule.
///
/// ⛔ **Implement this from the snapshot's GENERATION ordinal, never from a
/// format version.** `SWTF_VERSION` / `SDBF_VERSION` are `1` for every
/// snapshot ever written, so an identity built from one holds `version`
/// constant and the guard degrades **silently** to the commitment check
/// alone — the failure this module's `t14` arm exists to pin. The shipped
/// generation counters are `FuncAttnWeightsSnapshot::version` and
/// `MicroRecurrentKernelSnapshot::version`, both of which are deliberately
/// NOT part of their own hash input, which is exactly what makes them
/// ordinals rather than content.
///
/// ⚠ Implementing this says the type knows WHICH weights it is. It does not
/// say a calibration bound to it is fresh — see the module docs on what
/// `Some` does and does not claim.
pub trait SnapshotIdentity {
    /// The identity a calibration fitted against this snapshot must carry.
    fn snapshot_id(&self) -> SnapshotId;
}

/// Why a [`SnapshotBound`] refused — or did not.
///
/// The three refusal arms are never pooled: `VersionMoved` is an ordinary
/// stale calibration (refit it), `Unversioned` is an unwired call site (wire
/// the generation counter), and `CommitmentMoved` is a **caller bug** (a swap
/// that did not bump). Reporting one count for all three would send every
/// reader to the wrong repair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Staleness {
    /// Same generation, same weights — the calibration still describes them.
    Fresh,
    /// The generation ordinal moved: a snapshot was swapped.
    VersionMoved,
    /// Either side carries generation `0`. Never fresh, by rule 1.
    Unversioned,
    /// Generation matched and the weights did NOT — a bump site is broken.
    CommitmentMoved,
}

impl Staleness {
    /// `true` only for [`Staleness::Fresh`].
    #[inline]
    #[must_use]
    pub const fn is_fresh(&self) -> bool {
        matches!(self, Self::Fresh)
    }

    /// A short, stable reason string for the one-time warning.
    #[inline]
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::Fresh => "fresh",
            Self::VersionMoved => "snapshot generation moved (a freeze/thaw swap landed)",
            Self::Unversioned => {
                "snapshot generation is 0 (the call site wired no generation counter)"
            }
            Self::CommitmentMoved => {
                "weights changed while the generation held (a swap did not bump its counter)"
            }
        }
    }
}

/// A calibration head bound to the snapshot identity it was fitted under.
///
/// See the module docs. The type parameter is deliberately unconstrained: a
/// `SigmoidGateCalibrator`, a `StaticCalTable`, a conformal residual pool and
/// a bare `(f32, f32)` Platt pair are all things that go stale at this seam,
/// and none of them share a trait.
#[derive(Debug)]
pub struct SnapshotBound<T> {
    inner: T,
    fitted: SnapshotId,
    /// One-time-warning latch. Per BINDING, not global — a global latch means
    /// the second stale head in a process is silent, and it is the second one
    /// nobody is looking for.
    warned: AtomicBool,
}

impl<T> SnapshotBound<T> {
    /// Bind `inner` to the snapshot identity it was fitted against.
    #[inline]
    #[must_use]
    pub const fn new(inner: T, fitted: SnapshotId) -> Self {
        Self {
            inner,
            fitted,
            warned: AtomicBool::new(false),
        }
    }

    /// Classify freshness against `current`. **Pure** — no warning, no latch.
    ///
    /// Separate from [`Self::get`] so the rule is testable without observing
    /// a side effect, and so a caller can report staleness without consuming
    /// its one warning.
    #[inline]
    #[must_use]
    pub fn staleness(&self, current: SnapshotId) -> Staleness {
        // Rule 1 first: an unversioned side is never fresh, whatever else
        // matches. Ordering matters — checking equality first would report a
        // 0-vs-0 pair as Fresh, which is the unwired call site reading clean.
        match () {
            () if self.fitted.is_unversioned() || current.is_unversioned() => {
                Staleness::Unversioned
            }
            () if self.fitted.version != current.version => Staleness::VersionMoved,
            () if self.fitted.commitment != current.commitment => Staleness::CommitmentMoved,
            () => Staleness::Fresh,
        }
    }

    /// The calibration, or `None` if the snapshot identity has moved.
    ///
    /// Warns **once per binding** on the first refusal. The latch is reset by
    /// [`Self::rebind`], so a head that goes stale, is refitted, and goes
    /// stale again warns both times.
    #[inline]
    pub fn get(&self, current: SnapshotId) -> Option<&T> {
        let verdict = self.staleness(current);
        match verdict.is_fresh() {
            true => Some(&self.inner),
            false => {
                self.warn_once(verdict, current);
                None
            }
        }
    }

    /// The calibration **mutably**, or `None` if the snapshot identity has
    /// moved. Same warning semantics as [`Self::get`].
    ///
    /// ⛔ The mutable half is not a convenience — it is the half that guards
    /// the **supply** side. A read-only refusal protects one call; refusing a
    /// WRITE protects the pool itself. A calibration that keeps ACCUMULATING
    /// across a snapshot swap is not stale in a way any later check can see:
    /// it becomes a mixture of two models' statistics and heals into something
    /// that looks fitted and describes nothing. See
    /// [`crate::conformal::staleness`] for the worked case — exchangeability,
    /// the property a conformal interval's coverage guarantee rests on, is
    /// destroyed by exactly that mixture.
    ///
    /// ⚠ It is **not** [`Self::refit_with`]: this does not rebind, so it is
    /// for continuing to use a calibration that is still fresh, never for
    /// re-fitting one that is not.
    #[inline]
    pub fn get_mut(&mut self, current: SnapshotId) -> Option<&mut T> {
        let verdict = self.staleness(current);
        match verdict.is_fresh() {
            true => Some(&mut self.inner),
            false => {
                self.warn_once(verdict, current);
                None
            }
        }
    }

    /// Re-bind to a new snapshot identity after a refit, and re-arm the
    /// warning latch.
    ///
    /// ⚠ This asserts the caller has actually REFITTED against `fitted`.
    /// Calling it to silence a warning re-attaches a stale calibration under
    /// a fresh-looking identity, which is strictly worse than the defect this
    /// module exists for — the guard then certifies it.
    #[inline]
    pub fn rebind(&mut self, fitted: SnapshotId) {
        self.fitted = fitted;
        *self.warned.get_mut() = false;
    }

    /// Mutable access for a refit, paired with the identity to rebind to.
    #[inline]
    pub fn refit_with<R>(&mut self, fitted: SnapshotId, f: impl FnOnce(&mut T) -> R) -> R {
        let out = f(&mut self.inner);
        self.rebind(fitted);
        out
    }

    /// The identity this calibration was fitted under.
    #[inline]
    #[must_use]
    pub const fn fitted(&self) -> SnapshotId {
        self.fitted
    }

    /// The calibration, **unchecked**.
    ///
    /// The escape hatch, named so it is greppable. Legitimate uses are
    /// serialization, metrics and tests — anything that reads the fitted
    /// parameters *as parameters* rather than applying them to live inputs.
    /// Applying what this returns to a live score is the defect.
    #[inline]
    #[must_use]
    pub const fn peek_unchecked(&self) -> &T {
        &self.inner
    }

    /// Consume the binding, yielding the calibration.
    #[inline]
    #[must_use]
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// `true` once this binding has spent its warning.
    #[inline]
    #[must_use]
    pub fn has_warned(&self) -> bool {
        self.warned.load(Ordering::Relaxed)
    }

    #[inline]
    fn warn_once(&self, verdict: Staleness, current: SnapshotId) {
        // `swap` rather than load+store: two threads hitting a stale head at
        // once must produce ONE warning, not two or none.
        match self.warned.swap(true, Ordering::Relaxed) {
            true => {}
            false => {
                log::warn!(
                    "calibration REFUSED at the freeze/thaw seam — {} (fitted gen {}, current gen {}); \
                     returning None. Refit against the current snapshot, or run uncalibrated \
                     deliberately. This warning fires once per binding.",
                    verdict.reason(),
                    self.fitted.version,
                    current.version,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests;
