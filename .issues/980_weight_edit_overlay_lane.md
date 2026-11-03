# Issue 980: WeightEditOverlay — reasoning-authored, gate-gated weight-edit lane (Research 648 modelless half)

**Status:** DONE 2026-11-04 — T1–T6 LANDED, GOAT G1–G4 PASS ([Bench 976](../.benchmarks/976_issue980_weight_edit_overlay_goat.md)); opt-in per the no-default-consumer rule (metabolic_gate precedent), promotion consumer-gated (serving rung is veto-gated, Research 648 §4)
**Lease closed:** zed-agent@m3, 2026-11-03, expired 2026-11-05T00:00Z (48 h) — landed `b0f1ed615` well inside the window

> **Research:** [`.research/648_Weight_Space_Programming.md`](../.research/648_Weight_Space_Programming.md) — "Sparks of Weight-Space Programming" (Takehi/Teoh/Zhang/Khattab, MIT CSAIL blog, 2026-11-02; blog-only, no arXiv). Verdict: GOAT (fusion). Cousins: Plan 121 RandOpt (control arm), Issue 565 QuantErrorLora (sparse carrier + G5-negative precedent), `gain_cost_halt`/`saddle_escape` (obstacle telemetry), Plan 025 LoraPair (overlay dispatch shape).

## The primitive

A **deterministic weight-edit overlay**: an edit spec authored once (LLM session or hand-written from diagnostics), applied by a fixed compiler, gated by a pre-registered control battery, committed as a BLAKE3-pinned frozen artifact. Two edit forms:

1. **Direction subtraction** — `ΔW[:, S] = −k·d` (rank-1, column-masked; the blog's Exp-8 loop fix — the one edit class unpublished before the blog).
2. **Bypass rows** — small sigmoid-gated pathway through existing units (the blog's Exp-4 ramp fix; the gate is a SIGMOID DRIVE on a measured quantity per the FSM doctrine — never the blog's hard switch-off, which is the measured oscillation shape).

Carriers (existing substrate, consume-don't-fork): `LoraPair { reader, writer }` dispatch shape; `QuantErrorLora::SparseErrorBypass` top-K COO for the sparse form; `MerkleFrozenEnvelope` + provenance trailer for the committed artifact (mint-once-then-freeze — recipes are not durable, artifacts are). **PQ/ternary posture: apply deltas in the dequant domain at forward (QuantErrorLora posture) — never requantize a packed base.**

## The law the lane obeys (obstacle-vs-ability)

Weight edits remove obstacles that hide abilities; they do not add abilities. The **arming condition is obstacle telemetry**: `GainCostLoopHalter` (`HaltReason::Oscillation`) / `structural_cot_halt` counters above threshold arm the lane; below threshold the lane REFUSES "no obstacle found". This is the falsifier built into the arming logic.

## Pre-registered proof path (POC + GOAT gate)

Fixture: small dense MLP with an INJECTED degenerate direction (seeded: pick unit `d`, add `k₀·d` to a pinned column set so a two-action oscillation attractor forms in a pinned input region) — no LLM needed; the POC proves the overlay + gate machinery. Plus a no-degeneracy control fixture.

- [x] **T1** Fixture: `InjectedDegeneracyMlp` (128→64→64→4, seeded degenerate direction, pinned oscillation region; deterministic rollout) + no-degeneracy twin. Measure base obstacle rate (oscillation counter ≥ N consecutive opposite actions). — rates 1.0/0.0, halter cross-agreement 16/16 seeds (`tests/issue980_weight_edit_overlay_gates.rs`)
- [x] **T2** Overlay carrier: edit spec type (direction form + bypass form) → deterministic applier; apply-in-dequant-domain arm for a quantized twin of the fixture; untouched columns bit-identical assert. — int8 twin repaired with packed bytes BLAKE3-unchanged
- [x] **T3** Gate battery (G1): (a) dose-response monotone non-increasing over pinned k ∈ {1,2,4,8} — {1,1,1,0} + strictly-decreasing vertical-pull probe; (b) neighbor-layer isolation null ✓; (c) norm-matched random control, 16 BLAKE3-seeded arms, paired LCB95 > 0 vs the once-picked best arm ✓; (d) pre-screen refusal arm ✓; (e) retention: bit-identical out-of-region + k=k₀ trajectory == twin ✓
- [x] **T4** G2/G3/G4: compose-vs-bake recorded at 3 shapes (fixture 1.67× compose; LLM-scale strided-column penalty → forward-delta posture 281 ns, the designed serving path); flag-off byte-identical; alloc-free apply 0 allocs (canary'd; fixed a real alloc — touched_count's Vec inside the apply path); rejected-edit revert θ bit-identical (BLAKE3, backup-based)
- [x] **T5** Mint-once freeze: `WeightEditArtifact` envelope + provenance trailer (all optional except artifact BLAKE3); tamper refusal at thaw + payload verification
- [x] **T6** Feature gate `weight_edit_overlay` (opt-in, implies gain_cost_halt); GOAT verdict recorded ([Bench 976](../.benchmarks/976_issue980_weight_edit_overlay_goat.md)); battery passed so no demotion — but promotion stays consumer-gated

## Non-goals

- No LLM authoring lane here (hosted-author + weight-egress policy is `riir-train/.issues/665`'s scope; local Bonsai is not the capability class).
- No serving-rung wiring (rethink LAYA has no documented loop class — veto-gated consumer, see Research 648 §4).
- No training, no gradient descent anywhere; the runtime primitive stays modelless.
