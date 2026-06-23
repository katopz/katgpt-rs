# Plan 307: Claim Rubric Runtime — L1/L2/L3 Evidence Ladder as Code

**Date:** 2026-06-22
**Research:** [katgpt-rs/.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md](../.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md) (Gain-tier meta-discipline)
**Source paper:** [arxiv 2606.07612](https://arxiv.org/abs/2606.07612) — Gupta et al., ICML 2026 position paper (Pass as mechanism; rubric is the fusion value).
**Target:** `katgpt-rs/src/claim_rubric/` (new module) + Cargo feature `claim_rubric`
**Status:** Active — Phase 1 <state: in-progress>

---

## Goal

Materialize Research 287's L1/L2/L3 evidence ladder as a **generic, modelless,
zero-dependency Rust runtime** that any probe/steering primitive (or research
note / GOAT gate) can use to:

1. Declare a claim shape (`Claim { text, feature_class, declared_level }`).
2. Track which S1–S4 checklist items it satisfies (`EvidenceItem`).
3. Receive a `Grade { level, missing, vocabulary_violations, downgrades }`
   from a deterministic `ClaimValidator` that:
   - Verifies the satisfied items actually support the declared level
     (per `EvidenceLevel::requirements()`).
   - Scans the claim text for vocabulary forbidden at that level
     (e.g., "causally controls" at L1 → overclaim → downgrade to L0).
4. Return the canonical "honest" level (the max level whose requirements are
   all satisfied AND whose vocabulary appears in the text).

The output IS the rubric — but executable. Research notes can `cargo test`
their own claims; GOAT gates can require `Grade::passes(level)` before
promoting; downstream code can `match claim.grade().level` to pick which
API is licensed (read-only monitor vs intervention).

**Why now, despite R287 saying "no code":** the rubric's teeth come from
enforcement (§2.3 — vocabulary must match level). A code artifact enforces
this at CI time, not at author-discretion time. The user explicitly asked
for code.

**GOAT gate (per AGENTS.md):** no perf claim — this is a meta-discipline
primitive, not a hot-path kernel. Gate is *correctness*: the seven §4
primitive scores must round-trip through the validator to the levels R287
records. The crate compiles with `--no-default-features --features
claim_rubric` (zero-dep baseline).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Add `claim_rubric = []` feature to `katgpt-rs/Cargo.toml`
      with comment referencing Plan 307 + Research 287.
- [x] **T1.2** Add module wiring in `katgpt-rs/src/lib.rs`:
      `#[cfg(feature = "claim_rubric")] pub mod claim_rubric;` + re-exports
      of public surface (`EvidenceLevel`, `FeatureClass` re-use, `Claim`,
      `EvidenceItem`, `ChecklistSection`, `Grade`, `ClaimValidator`).
      **Deviation:** `Claim` is NOT re-exported at the crate root (it clashes
      with `clr::Claim<T>` under `--all-features`). Access it as
      `katgpt_rs::claim_rubric::Claim`.
- [x] **T1.3** Create `katgpt-rs/src/claim_rubric/` with:
      - `mod.rs` — module doc (links to R287 §2.2 rubric + §5 checklist),
        `pub use` of public surface.
      - `types.rs` — decoupled structs/enums per AGENTS.md rule.
      - `vocabulary.rs` — verb allow-lists per `EvidenceLevel` + scanner.
      - `checklist.rs` — S1–S4 items as `&'static [(EvidenceItem)]` tables.
      - `validator.rs` — `ClaimValidator::grade(&Claim) -> Grade` impl.
- [x] **T1.4** Implement `EvidenceLevel` enum (`#[repr(u8)]`, `L0 < L1 < L2
      < L3`, where `L0 = "no evidence"` is the auto-downgrade target).
      Derive `Ord, PartialOrd` so `Grade::max_satisfied()` is a one-liner.
- [x] **T1.5** Implement `FeatureClass` re-export from `katgpt_core::traits`
      (do not duplicate — same shim pattern as `src/pruners/feature_class.rs`).
- [x] **T1.6** Encode the R287 §2.2 rubric as data:
      `EvidenceLevel::L1.requirements() -> &'static [EvidenceItemId]`,
      similarly L2 (inherits L1 + adds), L3 (inherits L2 + adds).
      `EvidenceItemId` is a `#[non_exhaustive]` `#[repr(u16)]` enum so new
      items can be added without breaking ABI.
- [x] **T1.7** Encode the R287 §5 S1–S4 checklist as four static tables
      tagged by minimum level. Each row is `(EvidenceItemId, &'static str
      description, EvidenceLevel)` so callers can build a UI / CI report.
- [x] **T1.8** Encode the R287 §2.3 vocabulary rule: a `&'static [(verb,
      max_allowed_level: EvidenceLevel)]` table. Scanner is
      case-insensitive substring + word-boundary match. Forbid list
      (must not appear below the indicated level): "causally controls" (L3),
      "mechanistically mediates" (L3), "is the direction for" (L3),
      "induces" (L2), "reliably produces" (L2), "functionally steers" (L2).
      Allow list (safe at L1): "reads", "detects", "projects to", "emits",
      "correlates with".
- [x] **T1.9** Implement `ClaimValidator::grade()`:
      1. Compute `evidence_level = max L such that all L's `requirements()`
         are in `claim.satisfied`.
      2. Compute `vocab_level = min L such that no forbidden-for-<L verb
         appears in claim.text`. (Verb forbidden at L3 but allowed at L2 →
         if it appears, vocab_level is capped at L2.)
      3. `honest_level = min(evidence_level, vocab_level)`.
      4. If `honest_level < claim.declared_level`, record each missing
         item + each violating verb in `Grade.downgrades`.
      **Semantic deviation (documented in validator.rs module doc):** step 3
      was changed to `honest_level = evidence_level`. Vocabulary does NOT
      silently force `honest_level` down — that would hide the violation.
      Instead, verbs above `honest_level` are recorded in
      `Grade::vocabulary_violations` so the author can fix them. This matches
      R287 §2.3 "vocabulary must match evidence level" and makes the
      overclaim visible rather than silent.
- [x] **T1.10** Add `ClaimValidator::promote_advice(&Grade) -> Vec<String>`
       that returns the next-level missing items as human-readable
       "to upgrade to L2, add: ..." strings. (Return type is `Vec<String>`,
       not `Vec<&'static str>`, because the messages are formatted at runtime
       with the target level label.)

### Phase 1 acceptance

- `cargo check --no-default-features --features claim_rubric` clean.
- `cargo test --no-default-features --features claim_rubric` runs
  `tests/claim_rubric_test.rs` (next phase).

---

## Phase 2 — Round-trip Tests on R287 §4 Scores

### Tasks

- [x] **T2.1** `tests/claim_rubric_test.rs` — gated on `claim_rubric`.
- [x] **T2.2** Encode all seven R287 §4 primitive rows as test fixtures:
      | Primitive | Feature class | Declared | Expected honest level |
      |-----------|---------------|----------|------------------------|
      | `EmotionDirections::project` | Detection | L1 | L1 |
      | CNA contrastive | Detection | L1+ | L1 (modulation evidence informal) |
      | `FaithfulnessProbe::behavior_delta` | Detection (intervention) | L2 candidate | L1+ (specificity control TBD) |
      | `FutureBehaviorProbe` (FPCG) | Prediction | L1 (planned) | L1 |
      | `PosteriorGuidedPruner` | Detection | L1–L2 | L1–L2 |
      | HLA `evolve_hla` | Detection | L1 | L1 |
      | Spectral probes (EGA/SpectralQuant/irrep) | Detection | L1 | L1 |
      **Reconciliations** (documented in each test's doc-comment):
      - **FaithfulnessProbe** (`L2 candidate` → honest L1): R287 §2.2 L2 row
        requires generalization across ≥3 variations as a hard gate. That
        evidence is not yet shipped (Plan 278 Phase 4), so the honest grade
        is L1. The rubric flags "L2 candidate" as aspirational.
      - **PosteriorGuidedPruner** (`L1–L2` → honest L2): R287 §4 gives a
        range; we model the upper bound L2 because Plan 239's GOAT gate
        measured the gain across regime shifts (Generalization3Variations).
        Drop the L2 items from `satisfied` if a future audit downgrades to L1.
- [x] **T2.3** For each fixture, assert `validator.grade(&claim).level`
      equals the expected honest level from R287 §4. These tests ARE the
      R287 §4 table — if R287 revises a score, the test is the
      single-source-of-truth update site.
- [x] **T2.4** Vocabulary overclaim tests: a claim that says "causally
      controls" with only L1 evidence grades at **L1** (not L0) with a
      `VocabularyViolation` listing the verb. Honest level is evidence-bound;
      the violation is visible, not silently absorbed. (Plan T2.4 originally
      said "grade at L0" — that was the pre-revision `min(evidence, vocab)`
      semantic, superseded by the T1.9 deviation above.)
- [x] **T2.5** Vocabulary-allowed-at-level tests: "reads" at L1 passes
      cleanly; "induces" at L1 is a violation (honest stays L1); "induces"
      at L2 with L2 evidence passes.
- [x] **T2.6** Feature-class interaction tests: a `Prediction` claim that
      fails the predict-control-parity item must NOT auto-promote to L3
      even if every other L3 item is satisfied (R287 §3 row 2).

---

## Phase 3 — Example + Docs Hook

### Tasks

- [x] **T3.1** `examples/claim_rubric_minimal.rs` — construct a claim for
      a fictional probe, run the validator, print the grade. Single-file,
      runs with `cargo run --example claim_rubric_minimal --features
      claim_rubric --no-default-features`.
- [x] **T3.2** Cross-link from R287 §4 table: add a footer line
      "Executable form: `katgpt_rs::claim_rubric` (Plan 307)" so future
      readers know the table has a code mirror.
      **Status:** Done 2026-06-23. Footer added at end of
      `.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md`,
      pointing to `src/claim_rubric/` + `tests/claim_rubric_test.rs` as the
      single source of truth for §4 score revisions.
- [x] **T3.3** Add `claim_rubric` to the `default` feature list only AFTER
      Phase 2 passes (the meta-discipline should be on for every probe/
      steering primitive's CI). Until then, opt-in.
      **Status:** ✅ **DONE 2026-06-23 (parent).** Phase 2 re-verified green
      (17/17 round-trip + 1/1 GOAT gate, run by parent agent). `claim_rubric`
      promoted to `default` in `katgpt-rs/Cargo.toml` (now 133 default
      features). Promotion is the deliberate parent decision the plan called
      for; both `cargo check --no-default-features --features claim_rubric`
      and the default build compile clean.

---

## Phase 4 — GOAT Gate (correctness, not perf)

### Tasks

- [x] **T4.1** `tests/bench_307_claim_rubric_goat.rs` — runs the seven §4
      fixtures + the overclaim fixtures + the feature-class parity fixture.
      All must PASS. Gates promotion to default.
- [x] **T4.2** Audit: every probe/steering research note that invokes L1/
      L2/L3 vocabulary must now link to a `Claim` fixture in its
      corresponding primitive's test file. (Documentation task, not code.)
      **Status:** Done 2026-06-23. Audit doc at
      `.docs/claim_rubric_audit.md`. Honest finding: the rubric vocabulary
      is currently confined to R287 itself; the six cousin notes (R053,
      R144, R211, R244, R267, R286) do NOT yet self-invoke L1/L2/L3 — they
      pre-date the rubric. All seven §4 primitives have fixtures in the
      shared `claim_rubric_test.rs`; the gap is *adoption* (cousin notes
      should declare their level in-header), not fixture count. See the
      audit's "Recommended follow-ups" for the four backfill actions.

---

## Latent vs Raw boundary

This module operates on **claim text + metadata**, not on latent embeddings
or raw physical values. There is no sigmoid, no projection, no sync
crossing. The `FeatureClass` re-use is a *tag*, not a math operation. R287
§6 anti-patterns #3 (sync-boundary leakage) is encoded as a checklist item
(S2 "checked synced form matches probed form") rather than as a code path.

---

## Out of scope (explicitly)

- No LLM-judge integration. The validator is rule-based over text +
  metadata. R287 §5 S3 "if LLM judge" checklist items are present as
  `EvidenceItem`s but the validator does not call a judge.
- No markdown linter that scans `.research/*.md` files. That is a future
  CLI tool layered on top of `ClaimValidator`; the validator itself is
  library-only.
- No runtime invocation by hot-path kernels. The validator is a
  development-time / CI-time / GOAT-gate-time tool, not a 20Hz-tick tool.

---

## TL;DR

Research 287 says "no code, the note IS the output". The user disagrees —
the rubric's teeth come from enforcement, so we materialize it as a
zero-dep `claim_rubric` module: `EvidenceLevel` enum + `Claim` struct +
`ClaimValidator::grade()` + S1–S4 checklist as data + vocabulary scanner.
Phase 1 ships the skeleton, Phase 2 round-trips the seven §4 primitive
scores as tests (the tests ARE the R287 §4 table), Phase 3 adds an example
+ doc hook, Phase 4 GOAT gate is correctness-only (no perf claim).
Opt-in until Phase 2 passes; then promote to default so every probe/
steering primitive's CI runs the validator on its claims.
