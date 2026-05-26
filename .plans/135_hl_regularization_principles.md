# Plan 135: HL Regularization Principles — Document Patch Acceptance Guidelines

> **Source:** Research 096 — HL-ImageNet Symbolic Pipeline Overfitting
> **Status:** ✅ Complete (T1–T3 ✓)
> **Priority:** Low — infrastructure quality assurance
> **GOAT Pillar:** ❌ Not a pillar — see [MMO GOAT Pillars Decision Matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md). General HL design principles, not game-specific. Stays in `katgpt-rs` domain.
> **Domain:** `katgpt-rs` — no game IP, no secret, no selling point. The regularization principles are public knowledge from the HL-ImageNet experiment.

## Context

HL-ImageNet Phase 2 (Research 096) demonstrated that **code-as-model overfits** in exactly the same way neural networks do — by accumulating narrow, example-specific patches that improve train accuracy but not validation. Our `AbsorbCompress` + `BanditPruner` HL loop (Plan 032) currently lacks explicit regularization for patch acceptance. This plan documents the design principles and adds lightweight support instrumentation — no new feature gate, no hot-path changes.

The key insight from the paper: **reranking generalizes better than verify rules**. Our architecture already follows this pattern (`ScreeningPruner` > `AbsorbCompress` patches), but we don't enforce it structurally.

## Tasks

### T1: Add Regularization Section to HL Docs ✏️
- [x] Add "Patch Regularization Principles" section to `.docs/09_heuristic-learning.md`
- [x] Document the generalization hierarchy: ScreeningPruner > AbsorbCompress patches
- [x] Document the 6 regularization criteria from Research 096 D1:
  - **Support:** min episode count for arm acceptance
  - **Precision:** Q-value threshold (already have)
  - **Transfer:** held-out split (future — not implementing now)
  - **Complexity:** branch/threshold budget (future)
  - **Locality:** HotSwapPruner isolation (already have)
  - **Cascade risk:** compress phase handles stale (already have)
- [x] Cross-reference Research 014 (Learning Beyond Gradients) and Research 096
- **Files:** `.docs/09_heuristic-learning.md`
- **No new code** — documentation only

### T2: Add Support Count Tracking to AbsorbCompress ✏️
- [x] Add `support_count: usize` field to the absorb log entry (already tracked implicitly via win count)
- [x] Document that `support_count` is available for future regularization gate
- [x] Do NOT add acceptance gate — just instrument for observability
- **Files:** `src/pruners/absorb_compress.rs` (doc comment + field documentation)
- **No behavioral change** — documentation + observability only

### T3: GOAT Proof — Verify HL Loop Doesn't Overfit on Bomber 🐐
- [x] Run Bomber arena 1000 rounds with current `AbsorbCompress` config
- [x] Record win rate over time — verify no "train-only" degradation pattern
- [x] If degradation found → escalate to separate plan with held-out split
- **Files:** `katgpt-rs/.benchmarks/035_hl_regularization_goat.md` (new benchmark record)
- **Dependency:** `--features bandit,bomber`

## Not In Scope

- ❌ No new feature gate
- ❌ No held-out train/dev split (future work if T3 shows degradation)
- ❌ No complexity budget enforcement (future work)
- ❌ No changes to `forward()` or hot-path code
- ❌ No riir-ai changes (not game-specific)

## Why This Is Not a Plan in riir-ai

Per Decision Matrix 27, this is not a GOAT pillar:
- ❌ MMO-product: HL regularization is infrastructure, not player-facing
- ❌ Game-specific: Principles apply to any HL domain
- ❌ Defensible: Public knowledge from the paper
- ❌ Secret: No game IP involved

If future game-specific HL regularization becomes a competitive advantage (e.g., game-tuned acceptance thresholds that produce measurably better NPC behavior), that would go in riir-ai with a feature gate. This is the foundation, not the moat.

## References

- Research 096: HL-ImageNet Symbolic Pipeline Overfitting
- Research 014: Learning Beyond Gradients (Heuristic Learning)
- Plan 032: Heuristic Learning Infrastructure
- Plan 033: Bomberman HL Arena
- `.docs/27_mmo_goat_pillars_decision_matrix.md`: GOAT pillar decision matrix
