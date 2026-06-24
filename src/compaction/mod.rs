//! Closed-Unit Compaction Gate (CUCG) — generic, rubric-gated, training-free
//! trajectory compaction primitive.
//!
//! # Source
//!
//! SelfCompact (Li et al., JHU + Apple, Jun 2026,
//! [arXiv:2606.23525](https://arxiv.org/abs/2606.23525)). Plan 320, Research
//! 300.
//!
//! # What it does
//!
//! Fires context compaction at **structurally-safe moments** (closed-unit ∧
//! summarizable ∧ progress ∧ ¬stuck) rather than at fixed token thresholds.
//! The paper shows this matches fixed-interval accuracy at 30–70% lower token
//! cost on agents, with a skip-if-correct oracle variant having +11.5pp
//! further headroom.
//!
//! # Why it's a Super-GOAT
//!
//! The C1/C2/C3/N1 rubric is structurally isomorphic to our already-shipped
//! [`can_freeze`](../../riir_neuron_db) shard gate (riir-neuron-db Plan 002).
//! Unifying them as instances of one primitive (CUCG) is the cross-domain
//! force multiplier — this is G7 of the GOAT gate.
//!
//! # Architecture
//!
//! - [`Rubric<N>`] — evaluates a trajectory prefix into a fixed-size
//!   [`RubricVerdict<N>`] of `N` predicate results (latent, local).
//! - [`FireRule`] — Boolean combination of the verdict's `Yes`/`No` pattern
//!   (paper's `C1∧C2∧C3∧¬N1` search rule, `Q1∨(Q2∧Q3)` math rule, or the
//!   `P0∧P1` shard-freeze rule for G7).
//! - [`Backstop`] — token-pct force-compaction override (the safety net).
//! - [`ClosedUnitCompactionGate<R, N>`] — composes the above + an optional
//!   skip-if-reliable fuse (paper §4.1) into a single [`evaluate`](ClosedUnitCompactionGate::evaluate)
//!   entry point producing a [`CompactionDecision<N>`].
//! - [`CompactionAuditRecord<N>`] — `#[repr(C)]` POD that crosses the sync
//!   boundary as raw (the bridge: latent verdict → raw deterministic record).
//!
//! # Latent vs raw (per AGENTS.md)
//!
//! - **Latent / local** (never synced): the trajectory `y_{1:t}`, the
//!   rubric's predicate confidences, the rich [`PredicateResult`] enum.
//! - **Raw / deterministic** (crosses sync boundary): [`CompactionAuditRecord`]
//!   — the bit-identical decision record. Two honest nodes MUST agree on
//!   whether compaction fired and why.
//! - **Bridge**: the audit record is constructed by projecting the latent
//!   predicate confidences through fixed `Yes`/`No` thresholds and recording
//!   the trajectory span that grounded each `Yes`. Zero-allocation, gateable
//!   by feature flag, no sync dependency introduced.
//!
//! # Zero-allocation
//!
//! `evaluate()` performs no heap allocation for `N ≤ 8`. The audit record is
//! a stack `#[repr(C)]` POD; the scratch is caller-owned and reused. The
//! fire-rule tree's `Box` storage is allocated once at construction and
//! never touched on the hot path.
//!
//! # Sigmoid, never softmax
//!
//! Per AGENTS.md: each predicate's confidence is a scalar in `[0,1]` derived
//! from a sigmoid projection (Research 300 §2.4); the fire rule is a Boolean
//! combination, not a 3-way softmax over {COMPRESS, CONTINUE, DEFER}. The
//! latent-reframed rubrics (Phase 3+) compute their predicates from scalar
//! features the caller supplies via sigmoid gates.
//!
//! # Example
//!
//! ```no_run
//! use katgpt_rs::compaction::*;
//!
//! // A rubric with arity 4 (paper's C1/C2/C3/N1). The caller supplies the
//! // latent features; here we stub with a trivial marker-based rubric.
//! struct MyRubric;
//! impl Rubric<4> for MyRubric {
//!     fn evaluate(&self, traj: &[u8], _scratch: &mut RubricScratch) -> RubricVerdict<4> {
//!         // ... compute C1/C2/C3/N1 from latent features ...
//!         RubricVerdict::all_no()
//!     }
//! }
//!
//! let gate = ClosedUnitCompactionGate::builder(MyRubric)
//!     .fire_rule(FireRule::search_rule_4())
//!     .backstop(Backstop::token_pct(0.30))
//!     .skip_if_reliable(0.8)
//!     .probe_interval_tokens(1024)
//!     .build();
//!
//! let mut scratch = RubricScratch::new();
//! let decision = gate.evaluate(b"trajectory bytes", 100, 4096, Some(0.9), &mut scratch);
//! match decision {
//!     CompactionDecision::Compress { audit } => { /* run summarizer */ }
//!     CompactionDecision::Continue { audit } => { /* keep going */ }
//!     CompactionDecision::Forced   { audit } => { /* backstop fired */ }
//! }
//! ```
//!
//! # GOAT gate
//!
//! - **G1** — rubric beats fixed-interval on structural safety (≥ 80% recall
//!   at safe points, ≤ 20% FDR at mid-derivation). Phase 3.
//! - **G2** — skip-if-reliable suppression (≥ 50% suppression on reliable
//!   prefixes). Phase 6.
//! - **G3** — cache-reuse probe overhead independent of `L`. Phase 4.
//! - **G4** — zero-alloc hot path. Phase 2.
//! - **G5** — feature isolation (compiles ± the feature). Phase 6.
//! - **G6** — sigmoid, never softmax (static check). Phase 6.
//! - **G7** — cross-domain isomorphism with `can_freeze` (bit-identical
//!   decisions on a shard-freeze rubric). Phase 5.
//!
//! G1–G7 are the open-primitive gate. G8 (per-NPC runtime fusion at 20Hz ×
//! 1000 NPCs) is riir-ai's responsibility.
//!
//! # References
//!
//! - Plan 320: `.plans/320_closed_unit_compaction_gate.md`
//! - Research 300: `.research/300_Closed_Unit_Compaction_Gate_Rubric_Gated.md`
//! - Cross-ref: `riir-neuron-db/.research/007_Can_Freeze_As_Cucg_Instance_Crossref.md`

pub mod audit;
pub mod backstop;
pub mod decision;
pub mod fire_rule;
pub mod gate;
pub mod rubric;
pub mod rubrics;

pub use audit::{CompactionAuditRecord, DecisionKind, FireRuleEval, PredicateAudit};
pub use backstop::Backstop;
pub use decision::CompactionDecision;
pub use fire_rule::{CombineOp, FireRule};
pub use gate::{ClosedUnitCompactionGate, ClosedUnitCompactionGateBuilder};
pub use rubric::{PredicateReason, PredicateResult, Rubric, RubricScratch, RubricVerdict};

#[cfg(test)]
mod integration_tests {
    //! End-to-end integration tests exercising the full gate pipeline.
    //! Per-module unit tests live alongside each type.

    use super::*;

    /// A rubric whose predicates the test directly controls via a verdict
    /// seed. This lets us drive the gate through every decision branch
    /// without needing a real latent-feature pipeline.
    struct SeededRubric<const N: usize> {
        seed: RubricVerdict<N>,
    }
    impl<const N: usize> Rubric<N> for SeededRubric<N> {
        fn evaluate(&self, _traj: &[u8], _scratch: &mut RubricScratch) -> RubricVerdict<N> {
            self.seed.clone()
        }
    }

    fn yes(i: u32) -> PredicateResult {
        PredicateResult::Yes {
            quote_start: i,
            quote_len: 1,
        }
    }
    fn no(r: PredicateReason) -> PredicateResult {
        PredicateResult::No { reason: r }
    }

    #[test]
    fn full_pipeline_search_rule_compress() {
        let rubric = SeededRubric::<4> {
            seed: RubricVerdict::new([yes(0), yes(1), yes(2), no(PredicateReason::StillNovel)]),
        };
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::search_rule_4())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();
        let d = gate.evaluate(b"ignored", 0, 10_000, None, &mut scratch);
        assert!(d.is_compress());
        assert_eq!(d.audit().fire_rule_eval.yes_mask, 0b0111);
        assert_eq!(d.audit().fire_rule_eval.fired, 1);
        assert_eq!(d.audit().decision_kind(), Some(DecisionKind::Compress));
        // Bridge: the latent Yes predicates project to PredicateAudit::yes.
        assert!(d.audit().predicates[0].is_yes());
        assert!(d.audit().predicates[1].is_yes());
        assert!(d.audit().predicates[2].is_yes());
        assert!(!d.audit().predicates[3].is_yes());
        assert_eq!(
            d.audit().predicates[3].reason,
            PredicateReason::StillNovel.discriminant_byte()
        );
    }

    #[test]
    fn full_pipeline_shard_freeze_isomorphism_shape() {
        // G7 shape check (the actual bit-identical can_freeze comparison is
        // Phase 5 — requires the riir-neuron-db dependency wired in). Here we
        // verify the 2-predicate shard-freeze gate produces the right
        // decision shape.
        let both_yes = SeededRubric::<2> {
            seed: RubricVerdict::new([yes(0), yes(1)]),
        };
        let gate = ClosedUnitCompactionGate::builder(both_yes)
            .fire_rule(FireRule::shard_freeze_rule_2())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();
        let d = gate.evaluate(b"shard", 0, 10_000, None, &mut scratch);
        assert!(d.is_compress(), "P0 ∧ P1 → Compress (mirrors can_freeze)");

        let only_p0 = SeededRubric::<2> {
            seed: RubricVerdict::new([yes(0), no(PredicateReason::Custom(1))]),
        };
        let gate2 = ClosedUnitCompactionGate::builder(only_p0)
            .fire_rule(FireRule::shard_freeze_rule_2())
            .backstop(Backstop::None)
            .build();
        let d2 = gate2.evaluate(b"shard", 0, 10_000, None, &mut scratch);
        assert!(
            d2.is_continue(),
            "P1 missing → no freeze (mirrors can_freeze)"
        );
    }

    #[test]
    fn audit_record_crosses_sync_boundary_as_pod_bytes() {
        // The sync-boundary contract: the audit record must round-trip
        // through raw bytes bit-identically. This is what makes two honest
        // nodes agree on the compaction decision.
        let rubric = SeededRubric::<4> {
            seed: RubricVerdict::new([
                yes(10),
                yes(20),
                no(PredicateReason::NoProgress),
                no(PredicateReason::StillNovel),
            ]),
        };
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::search_rule_4())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();
        let d = gate.evaluate(b"traj", 0, 10_000, None, &mut scratch);
        let audit = d.audit();

        // Serialize to bytes (the "sync" side).
        let bytes: &[u8] = bytemuck::bytes_of(audit);
        // Deserialize (the "other node" side).
        let back: &CompactionAuditRecord<4> = bytemuck::from_bytes(bytes);
        assert_eq!(back, audit, "audit must round-trip bit-identically");
        assert_eq!(back.trajectory_len, 4);
        assert!(back.predicates[0].is_yes());
        assert_eq!(back.predicates[0].quote_start, 10);
    }
}
