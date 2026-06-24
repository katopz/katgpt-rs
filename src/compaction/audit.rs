//! The `CompactionAuditRecord` — deterministic, `#[repr(C)]` audit record
//! that crosses the sync boundary as raw.
//!
//! Per AGENTS.md: "If data crosses the boundary, use a **bridge function**
//! (raw → latent projection, latent → raw scalar clamp)." The audit record
//! IS the raw side — it is constructed by projecting the latent predicate
//! confidences through fixed `Yes`/`No` thresholds and recording the
//! trajectory span that grounded each `Yes`. Two honest nodes observing the
//! same trajectory MUST produce bit-identical records (anti-cheat /
//! explainability contract, Research 300 §7).
//!
//! The per-predicate rich result ([`super::PredicateResult`]) is *latent /
//! local* — it never crosses the boundary. Only the compact
//! [`PredicateAudit`] POD does.

use super::rubric::PredicateReason;

/// POD snapshot of one predicate's verdict for the audit record.
///
/// `#[repr(C)]` so the layout is fixed across platforms — required for the
/// sync-boundary bit-identical contract. The `kind` byte discriminates
/// `Yes`/`No`; `reason` carries the [`PredicateReason`] byte (only meaningful
/// when `kind == NO`); `quote_start`/`quote_len` cite the trajectory span
/// (only meaningful when `kind == YES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct PredicateAudit {
    /// `1` = Yes, `0` = No. Stored as a raw byte (not the enum) so the layout
    /// is unambiguous across compilers.
    pub kind: u8,
    /// `PredicateReason` discriminant byte. Meaningful iff `kind == 0` (No).
    /// For `Yes`, set to `Unset` (0).
    pub reason: u8,
    /// Padding to align the `u32 quote_start` to a 4-byte boundary. Keeps the
    /// struct a clean `#[repr(C)]` POD without compiler-dependent holes.
    pub _pad: u16,
    /// Start of the trajectory span grounding the `Yes`. Meaningful iff
    /// `kind == 1`. For `No`, set to `0`.
    pub quote_start: u32,
    /// Length of the trajectory span grounding the `Yes`. Meaningful iff
    /// `kind == 1`. For `No`, set to `0`.
    pub quote_len: u16,
    /// Explicit padding to round the struct to a multiple of 4 bytes
    /// (`quote_start` u32 + `quote_len` u16 = 6 bytes; this pad makes it 8).
    pub _pad2: u16,
}

// SAFETY: `PredicateAudit` is all `u8`/`u16`/`u32` fields — plain old data.
// `#[repr(C)]` fixes the layout. Required for `bytemuck` / zero-copy sync
// serialization (the audit record is committed raw to the Cold tier).
unsafe impl bytemuck::Pod for PredicateAudit {}
unsafe impl bytemuck::Zeroable for PredicateAudit {}

impl PredicateAudit {
    /// `kind == 1` (Yes).
    pub const YES: u8 = 1;
    /// `kind == 0` (No).
    pub const NO: u8 = 0;

    /// Construct a `Yes` audit with the cited trajectory span.
    #[inline]
    #[must_use]
    pub const fn yes(quote_start: u32, quote_len: u16) -> Self {
        Self {
            kind: Self::YES,
            reason: 0,
            _pad: 0,
            quote_start,
            quote_len,
            _pad2: 0,
        }
    }

    /// Construct a `No` audit with the reason byte.
    #[inline]
    #[must_use]
    pub const fn no(reason: PredicateReason) -> Self {
        Self {
            kind: Self::NO,
            reason: PredicateReason::discriminant_byte(reason),
            _pad: 0,
            quote_start: 0,
            quote_len: 0,
            _pad2: 0,
        }
    }

    /// Returns `true` iff this is a `Yes` audit.
    #[inline]
    #[must_use]
    pub const fn is_yes(&self) -> bool {
        self.kind == Self::YES
    }
}

impl Default for PredicateAudit {
    /// Default is `No { reason: Unset }` — mirrors
    /// [`super::PredicateResult`]'s default so a freshly-initialized record
    /// is consistent with an all-`No` rubric verdict.
    #[inline]
    fn default() -> Self {
        Self::no(PredicateReason::Unset)
    }
}

/// Discriminant byte for the compaction decision (Research 300 §2.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DecisionKind {
    /// `0` — Continue from `(x, y_{1:t})` unchanged.
    #[default]
    Continue = 0,
    /// `1` — Compaction is structurally safe; caller runs the summarizer.
    Compress = 1,
    /// `2` — Token-pct backstop forced the decision (rubric may disagree).
    Forced = 2,
}

impl DecisionKind {
    /// Returns the discriminant byte.
    #[inline]
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        self as u8
    }

    /// Construct from a discriminant byte. Returns `None` for invalid bytes.
    #[inline]
    #[must_use]
    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Continue),
            1 => Some(Self::Compress),
            2 => Some(Self::Forced),
            _ => None,
        }
    }
}

/// Snapshot of the fire-rule evaluation for the audit record. Records the
/// packed `yes_mask` and the resulting fire decision, so the audit can
/// reconstruct *why* the rule fired (or didn't) without re-evaluating.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct FireRuleEval {
    /// Packed `Yes`/`No` bitmask the rubric produced (bit `i` set iff
    /// predicate `i` was `Yes`).
    pub yes_mask: u8,
    /// `1` iff the fire rule evaluated to `true` (COMPRESS) given
    /// `yes_mask`. `0` otherwise.
    pub fired: u8,
    /// Padding to 4 bytes.
    pub _pad: u16,
}

// SAFETY: all-integer POD, `#[repr(C)]`.
unsafe impl bytemuck::Pod for FireRuleEval {}
unsafe impl bytemuck::Zeroable for FireRuleEval {}

/// Deterministic, bit-identical audit record. Crosses the sync boundary as
/// raw (per AGENTS.md + Research 300 §7).
///
/// Constructed by the gate's [`super::ClosedUnitCompactionGate::evaluate`]
/// from the latent [`super::RubricVerdict`] via a fixed-threshold
/// projection. Two honest nodes observing the same trajectory MUST produce
/// equal records — this is the anti-cheat / explainability contract.
///
/// `N` is the rubric arity (const generic), matching
/// [`super::RubricVerdict<N>`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct CompactionAuditRecord<const N: usize> {
    /// Length of the trajectory prefix at the probe point (`|y_{1:t}|`).
    pub trajectory_len: u32,
    /// Per-predicate audit PODs. Fixed-size stack array — zero alloc.
    pub predicates: [PredicateAudit; N],
    /// Snapshot of the fire-rule evaluation.
    pub fire_rule_eval: FireRuleEval,
    /// `1` iff the token-pct backstop forced the decision.
    pub backstop_triggered: u8,
    /// `1` iff the skip-if-reliable fuse suppressed a would-be `Compress`.
    pub skip_if_reliable_triggered: u8,
    /// Decision discriminant (see [`DecisionKind`]).
    pub decision: u8,
    /// Padding to a 4-byte boundary.
    pub _pad: u8,
}

// SAFETY: `CompactionAuditRecord` is all POD fields (`u32`, `[PredicateAudit;
// N]`, `FireRuleEval`, `u8`s). `#[repr(C)]` fixes the layout.
unsafe impl<const N: usize> bytemuck::Pod for CompactionAuditRecord<N> {}
unsafe impl<const N: usize> bytemuck::Zeroable for CompactionAuditRecord<N> {}

impl<const N: usize> Default for CompactionAuditRecord<N> {
    /// Default is an all-`No`, `Continue` record.
    #[inline]
    fn default() -> Self {
        Self {
            trajectory_len: 0,
            predicates: [PredicateAudit::default(); N],
            fire_rule_eval: FireRuleEval::default(),
            backstop_triggered: 0,
            skip_if_reliable_triggered: 0,
            decision: DecisionKind::Continue.to_byte(),
            _pad: 0,
        }
    }
}

impl<const N: usize> CompactionAuditRecord<N> {
    /// Returns the decision kind, or `None` if the byte is invalid (should
    /// never happen for a record produced by the gate).
    #[inline]
    #[must_use]
    pub const fn decision_kind(&self) -> Option<DecisionKind> {
        DecisionKind::from_byte(self.decision)
    }

    /// Returns `true` iff the record's decision is `Compress`.
    #[inline]
    #[must_use]
    pub fn is_compress(&self) -> bool {
        self.decision_kind() == Some(DecisionKind::Compress)
    }

    /// Returns `true` iff the record's decision is `Forced`.
    #[inline]
    #[must_use]
    pub fn is_forced(&self) -> bool {
        self.decision_kind() == Some(DecisionKind::Forced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicate_audit_yes_round_trips() {
        let a = PredicateAudit::yes(42, 7);
        assert!(a.is_yes());
        assert_eq!(a.kind, PredicateAudit::YES);
        assert_eq!(a.quote_start, 42);
        assert_eq!(a.quote_len, 7);
    }

    #[test]
    fn predicate_audit_no_carries_reason_byte() {
        let a = PredicateAudit::no(PredicateReason::NotClosedUnit);
        assert!(!a.is_yes());
        assert_eq!(a.kind, PredicateAudit::NO);
        assert_eq!(a.reason, PredicateReason::NotClosedUnit.discriminant_byte());
        assert_eq!(a.quote_start, 0);
        assert_eq!(a.quote_len, 0);
    }

    #[test]
    fn predicate_audit_default_is_no_unset() {
        let a = PredicateAudit::default();
        assert!(!a.is_yes());
        assert_eq!(a.reason, PredicateReason::Unset.discriminant_byte());
    }

    #[test]
    fn predicate_audit_is_pod_and_zeroable() {
        // bytemuck traits are implemented; constructing zeroed is valid.
        let zero: PredicateAudit = bytemuck::Zeroable::zeroed();
        assert_eq!(zero.kind, 0);
        assert_eq!(zero.reason, 0);
        assert_eq!(zero.quote_start, 0);
        assert_eq!(zero.quote_len, 0);

        // bytes_of round-trip.
        let a = PredicateAudit::yes(100, 3);
        let bytes: &[u8] = bytemuck::bytes_of(&a);
        let back: &PredicateAudit = bytemuck::from_bytes(bytes);
        assert_eq!(back, &a);
    }

    #[test]
    fn decision_kind_round_trips() {
        for k in [
            DecisionKind::Continue,
            DecisionKind::Compress,
            DecisionKind::Forced,
        ] {
            let b = k.to_byte();
            assert_eq!(DecisionKind::from_byte(b), Some(k));
        }
        assert_eq!(DecisionKind::from_byte(3), None);
        assert_eq!(DecisionKind::from_byte(255), None);
    }

    #[test]
    fn decision_kind_default_is_continue() {
        assert_eq!(DecisionKind::default(), DecisionKind::Continue);
    }

    #[test]
    fn fire_rule_eval_default_is_no_fire() {
        let e = FireRuleEval::default();
        assert_eq!(e.yes_mask, 0);
        assert_eq!(e.fired, 0);
    }

    #[test]
    fn compaction_audit_record_default_is_continue_no_predicates() {
        let r: CompactionAuditRecord<4> = CompactionAuditRecord::default();
        assert_eq!(r.trajectory_len, 0);
        assert!(r.predicates.iter().all(|p| !p.is_yes()));
        assert_eq!(r.fire_rule_eval.fired, 0);
        assert_eq!(r.backstop_triggered, 0);
        assert_eq!(r.skip_if_reliable_triggered, 0);
        assert_eq!(r.decision_kind(), Some(DecisionKind::Continue));
        assert!(!r.is_compress());
        assert!(!r.is_forced());
    }

    #[test]
    fn compaction_audit_record_is_pod_and_zeroable() {
        let zero: CompactionAuditRecord<3> = bytemuck::Zeroable::zeroed();
        assert_eq!(zero.trajectory_len, 0);
        assert_eq!(zero.decision, DecisionKind::Continue.to_byte());

        // Round-trip through bytes.
        let mut r: CompactionAuditRecord<3> = CompactionAuditRecord::default();
        r.trajectory_len = 1234;
        r.decision = DecisionKind::Compress.to_byte();
        r.predicates[0] = PredicateAudit::yes(10, 2);
        let bytes: &[u8] = bytemuck::bytes_of(&r);
        let back: &CompactionAuditRecord<3> = bytemuck::from_bytes(bytes);
        assert_eq!(back, &r);
        assert_eq!(back.trajectory_len, 1234);
        assert!(back.is_compress());
        assert!(back.predicates[0].is_yes());
    }

    #[test]
    fn compaction_audit_record_predicate_array_layout_is_repr_c() {
        // The `[PredicateAudit; N]` field must be `N * size_of::<PredicateAudit>()`
        // with no padding between elements — this is the repr(C) invariant.
        assert_eq!(std::mem::size_of::<PredicateAudit>(), 12);
        let n = 4;
        let offset_predicates = std::mem::size_of::<u32>(); // trajectory_len
        let size_predicates = n * std::mem::size_of::<PredicateAudit>();
        assert_eq!(
            offset_predicates + size_predicates,
            std::mem::size_of::<CompactionAuditRecord<4>>()
                // minus the trailing fields (fire_rule_eval 4 + 3 bytes + pad)
                - std::mem::size_of::<FireRuleEval>()
                - 4 // backstop/skip/decision/pad
        );
    }
}
