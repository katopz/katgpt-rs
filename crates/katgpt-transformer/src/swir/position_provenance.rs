//! Token/Latent position-provenance ledger — the swir decode-history contract
//! (Issue 941).
//!
//! Distilled from HybridAgent's `kv_ledger` semantics (arXiv:2610.18861
//! §"continuous working memory", Research 622) — a confirmed law, lifted into
//! the host contract under this repo's vocabulary. No novelty claim.
//!
//! # The law
//!
//! A decode position fed to a
//! [`DecodeBackend`](super::bench::DecodeBackend) is either:
//!
//! - **`Tok(id)`** — a concrete token id (Explicit mode, or a control-token
//!   injection). A REAL position: it persists into the materialized token
//!   history **M(L)** and is the only thing samplers (repeat/frequency
//!   penalties, n-gram bans, ...) may read.
//! - **`Lat { span, step }`** — a soft-embedding position (`decode_step` was
//!   called with `token_id = None`). Think-brain scratch: it attended and was
//!   attended, but it NEVER serializes into M(L). Its per-step argmax is not
//!   a token decision and must never enter a penalty set.
//!
//! # Purity invariant
//!
//! The ledger is **metadata, never an attention mask**: latent positions
//! attend and are attended exactly like token positions (HybridAgent's own
//! design). Nothing in this module may be consumed as a mask. This is the
//! two-brain law at cache level: `Tok` positions persist (info brain,
//! syncable); `Lat` positions are think-brain scratch that never serializes.
//!
//! # Reference implementation
//!
//! riir-ai `riir-engine/src/swir_validation/gemma2_backend.rs` (the Plan 313
//! real-model host) — it pushes to `generated_tokens` ONLY on
//! `token_id = Some(id)`. Consume, never fork: hosts wire
//! [`ProvenanceLedger`] instead of re-deriving the convention.
//!
//! Cross-ref: the two LIVE defects found in the same subsystem during verdict
//! review are filed riir-ai-side (riir-ai Issue 1075: penalty compounding over
//! duplicate history entries + latent pseudo-argmaxes leaking into the bench
//! scoring path) — this module owns only the katgpt-rs contract.

/// Provenance of one decode position fed to a [`DecodeBackend`](super::bench::DecodeBackend).
///
/// `Tok` positions are real and persist; `Lat` positions are think-brain
/// scratch and never serialize (see the module docs for the law).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PositionProvenance {
    /// A concrete token id (Explicit mode / control-token injection). MUST be
    /// included in the materialized history M(L).
    Tok(u32) = 0,
    /// A soft-embedding position (Latent mode: `decode_step` called with
    /// `token_id = None`). MUST NOT enter M(L). `span` identifies the latent
    /// span (host-assigned, monotonically increasing per entry), `step` the
    /// position within the span.
    Lat { span: u32, step: u32 } = 1,
}

/// The provenance ledger and its materialized view M(L).
///
/// Hosts record every decode position; the ledger retains the Tok ids (the
/// materialized history) plus Lat-position COUNTS ONLY — latent scratch is
/// never retained, mirroring "never serializes" structurally: there is no
/// field a serializer could accidentally persist for `Lat`.
///
/// # Allocation
///
/// `record` pushes to the Tok buffer (amortized); Lat records touch only two
/// scalars. No per-step allocation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProvenanceLedger {
    /// The materialized token history M(L) — Tok ids in decode order. THE
    /// single history surface samplers read.
    tok: Vec<u32>,
    /// Total latent positions recorded (metadata only).
    latent_positions: u32,
    /// Number of distinct latent spans entered (a span change = a `Lat` record
    /// whose `span` differs from the previous `Lat` record's span, or the
    /// first `Lat`).
    latent_spans: u32,
    /// Span id of the last `Lat` record (for span-change detection).
    last_latent_span: Option<u32>,
}

impl ProvenanceLedger {
    /// Empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one decode position.
    ///
    /// `Tok(id)` appends to M(L); `Lat { .. }` bumps the latent counters only
    /// (the position's content is deliberately not retained).
    pub fn record(&mut self, p: PositionProvenance) {
        match p {
            PositionProvenance::Tok(id) => self.tok.push(id),
            PositionProvenance::Lat { span, .. } => {
                if self.last_latent_span != Some(span) {
                    self.latent_spans += 1;
                    self.last_latent_span = Some(span);
                }
                self.latent_positions += 1;
            }
        }
    }

    /// The materialized token history **M(L)** — Tok ids only, in decode
    /// order. The single history surface samplers (repeat/frequency penalty,
    /// n-gram bans) may read.
    #[inline]
    pub fn materialized(&self) -> &[u32] {
        &self.tok
    }

    /// Total decode positions recorded (Tok + Lat).
    #[inline]
    pub fn position_count(&self) -> u64 {
        self.tok.len() as u64 + u64::from(self.latent_positions)
    }

    /// Latent positions recorded (metadata only — no content is retained).
    #[inline]
    pub fn latent_positions(&self) -> u32 {
        self.latent_positions
    }

    /// Distinct latent spans entered.
    #[inline]
    pub fn latent_spans(&self) -> u32 {
        self.latent_spans
    }

    /// Reset between problems/requests (fresh history, fresh counters).
    pub fn clear(&mut self) {
        self.tok.clear();
        self.latent_positions = 0;
        self.latent_spans = 0;
        self.last_latent_span = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal penalty-count model: occurrences per token id over a history
    /// surface — the class of consumers M(L) exists for.
    fn penalty_counts(history: &[u32]) -> std::collections::BTreeMap<u32, usize> {
        let mut m = std::collections::BTreeMap::new();
        for &id in history {
            *m.entry(id).or_insert(0) += 1;
        }
        m
    }

    /// Contract 1: with NO latent spans, provenance-on is bit-identical to a
    /// provenance-off host that pushed every token directly.
    #[test]
    fn no_latent_spans_bit_identical_to_plain_history() {
        let stream = [3_u32, 7, 3, 9, 7, 128];
        let mut ledger = ProvenanceLedger::new();
        for &id in &stream {
            ledger.record(PositionProvenance::Tok(id));
        }
        assert_eq!(ledger.materialized(), &stream[..]);
        assert_eq!(
            penalty_counts(ledger.materialized()),
            penalty_counts(&stream)
        );
        assert_eq!(ledger.latent_positions(), 0);
        assert_eq!(ledger.latent_spans(), 0);
    }

    /// Contract 2: a latent span's argmaxes NEVER enter the penalty set — the
    /// ledger has no path for them, and M(L) compresses them out.
    #[test]
    fn latent_argmaxes_never_enter_penalty_set() {
        let mut ledger = ProvenanceLedger::new();
        // Reference-host discipline (riir-ai gemma2_backend): only Some(id)
        // positions are recorded. The latent steps' argmaxes (5, 5) are simply
        // never fed to the ledger.
        ledger.record(PositionProvenance::Tok(1));
        ledger.record(PositionProvenance::Lat { span: 0, step: 0 });
        ledger.record(PositionProvenance::Lat { span: 0, step: 1 });
        ledger.record(PositionProvenance::Tok(1));

        let m = ledger.materialized();
        assert_eq!(m, &[1_u32, 1][..]); // latent positions compressed out
        let counts = penalty_counts(m);
        assert_eq!(counts.get(&1), Some(&2)); // real repeat detected
        assert!(!counts.contains_key(&5)); // the latent argmax never lands
        assert_eq!(ledger.latent_positions(), 2);
        assert_eq!(ledger.latent_spans(), 1);
        assert_eq!(ledger.position_count(), 4); // 2 tok + 2 latent
    }

    /// Contract 3: a real repeated token ALWAYS enters the penalty set.
    #[test]
    fn real_repeated_token_always_penalized() {
        let mut ledger = ProvenanceLedger::new();
        for id in [7_u32, 3, 7, 11, 7] {
            ledger.record(PositionProvenance::Tok(id));
        }
        let counts = penalty_counts(ledger.materialized());
        assert_eq!(counts.get(&7), Some(&3));
        assert_eq!(counts.get(&3), Some(&1));
    }

    /// Span accounting: span changes count; steps within a span do not.
    #[test]
    fn span_counting_tracks_span_changes_only() {
        let mut ledger = ProvenanceLedger::new();
        ledger.record(PositionProvenance::Lat { span: 0, step: 0 });
        ledger.record(PositionProvenance::Lat { span: 0, step: 1 });
        ledger.record(PositionProvenance::Lat { span: 0, step: 2 });
        ledger.record(PositionProvenance::Tok(4)); // span 0 ends
        ledger.record(PositionProvenance::Lat { span: 3, step: 0 }); // new span
        assert_eq!(ledger.latent_spans(), 2);
        assert_eq!(ledger.latent_positions(), 4);
    }

    /// Lat content is structurally unretainable: interleaved latent noise
    /// leaves M(L) bit-identical to the Tok-only stream.
    #[test]
    fn materialized_ignores_latent_interleaving() {
        let toks = [2_u32, 2, 5, 9];
        let mut a = ProvenanceLedger::new();
        let mut b = ProvenanceLedger::new();
        for (i, &t) in toks.iter().enumerate() {
            a.record(PositionProvenance::Tok(t));
            b.record(PositionProvenance::Tok(t));
            if i % 2 == 0 {
                b.record(PositionProvenance::Lat {
                    span: 0,
                    step: i as u32,
                });
            }
        }
        assert_eq!(a.materialized(), b.materialized());
    }

    /// clear() resets history AND counters (fresh-problem posture).
    #[test]
    fn clear_resets_everything() {
        let mut ledger = ProvenanceLedger::new();
        ledger.record(PositionProvenance::Tok(1));
        ledger.record(PositionProvenance::Lat { span: 2, step: 0 });
        ledger.clear();
        assert!(ledger.materialized().is_empty());
        assert_eq!(ledger.position_count(), 0);
        assert_eq!(ledger.latent_positions(), 0);
        assert_eq!(ledger.latent_spans(), 0);
        // A post-clear latent records as a NEW span (counters truly reset).
        ledger.record(PositionProvenance::Lat { span: 0, step: 0 });
        assert_eq!(ledger.latent_spans(), 1);
    }
}
