//! The `CompactionDecision` enum — the gate's verdict.
//!
//! Returned by [`super::ClosedUnitCompactionGate::evaluate`]. Each variant
//! carries its own [`super::CompactionAuditRecord`] so the caller always has
//! a deterministic record of *why* the decision was made, regardless of
//! which branch fired (rubric fire → `Compress`, rubric decline → `Continue`,
//! backstop override → `Forced`).

use super::audit::CompactionAuditRecord;

/// The gate's decision. Each variant carries a full audit record so the
/// caller can log / sync the reason for the decision.
///
/// Decision order in [`super::ClosedUnitCompactionGate::evaluate`]:
/// 1. Backstop forces → [`CompactionDecision::Forced`].
/// 2. Else rubric evaluates; fire rule checks the verdict.
/// 3. Fire + no skip-suppression → [`CompactionDecision::Compress`].
/// 4. Fire + skip-suppression (CLR reliable) → [`CompactionDecision::Continue`].
/// 5. No fire → [`CompactionDecision::Continue`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompactionDecision<const N: usize> {
    /// Compaction is structurally safe. Caller should run the summarizer,
    /// then hard-reset to `(x, ỹ)`.
    Compress {
        /// Audit record for this decision.
        audit: CompactionAuditRecord<N>,
    },
    /// Continue from `(x, y_{1:t})` unchanged. Probe verdict is reverted
    /// from the rolling cache — does not pollute subsequent generation.
    Continue {
        /// Audit record for this decision.
        audit: CompactionAuditRecord<N>,
    },
    /// Token-pct backstop forced the decision. Rubric verdict may disagree —
    /// the audit records both so the caller can detect "I was forced even
    /// though the rubric said no".
    Forced {
        /// Audit record for this decision.
        audit: CompactionAuditRecord<N>,
    },
}

impl<const N: usize> CompactionDecision<N> {
    /// Returns a reference to the audit record, regardless of variant.
    #[inline]
    #[must_use]
    pub const fn audit(&self) -> &CompactionAuditRecord<N> {
        match self {
            Self::Compress { audit } | Self::Continue { audit } | Self::Forced { audit } => audit,
        }
    }

    /// Returns `true` iff this is [`CompactionDecision::Compress`].
    #[inline]
    #[must_use]
    pub const fn is_compress(&self) -> bool {
        matches!(self, Self::Compress { .. })
    }

    /// Returns `true` iff this is [`CompactionDecision::Continue`].
    #[inline]
    #[must_use]
    pub const fn is_continue(&self) -> bool {
        matches!(self, Self::Continue { .. })
    }

    /// Returns `true` iff this is [`CompactionDecision::Forced`].
    #[inline]
    #[must_use]
    pub const fn is_forced(&self) -> bool {
        matches!(self, Self::Forced { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::audit::DecisionKind;

    #[test]
    fn audit_accessor_works_for_all_variants() {
        let a: CompactionAuditRecord<2> = CompactionAuditRecord {
            decision: DecisionKind::Compress.to_byte(),
            ..Default::default()
        };
        let d = CompactionDecision::<2>::Compress { audit: a };
        assert_eq!(d.audit().decision, DecisionKind::Compress.to_byte());

        let b: CompactionAuditRecord<2> = CompactionAuditRecord {
            decision: DecisionKind::Continue.to_byte(),
            ..Default::default()
        };
        let d = CompactionDecision::<2>::Continue { audit: b };
        assert_eq!(d.audit().decision, DecisionKind::Continue.to_byte());

        let c: CompactionAuditRecord<2> = CompactionAuditRecord {
            decision: DecisionKind::Forced.to_byte(),
            ..Default::default()
        };
        let d = CompactionDecision::<2>::Forced { audit: c };
        assert_eq!(d.audit().decision, DecisionKind::Forced.to_byte());
    }

    #[test]
    fn is_predicates_distinguish_variants() {
        let a: CompactionAuditRecord<1> = CompactionAuditRecord::default();
        assert!(CompactionDecision::<1>::Compress { audit: a }.is_compress());
        assert!(CompactionDecision::<1>::Continue { audit: a }.is_continue());
        assert!(CompactionDecision::<1>::Forced { audit: a }.is_forced());

        assert!(!CompactionDecision::<1>::Compress { audit: a }.is_continue());
        assert!(!CompactionDecision::<1>::Continue { audit: a }.is_forced());
        assert!(!CompactionDecision::<1>::Forced { audit: a }.is_compress());
    }
}
