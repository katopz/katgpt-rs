# Issue 980: WeightEditOverlay — reasoning-authored, gate-gated weight-edit lane (Research 648 modelless half)

**Status:** OPEN — claimed (in progress)
**Claimed:** zed-agent@m3, 2026-11-03, expires 2026-11-05T00:00Z (48 h)

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

- [ ] **T1** Fixture: `InjectedDegeneracyMlp` (128→64→64→4, seeded degenerate direction, pinned oscillation region; deterministic rollout) + no-degeneracy twin. Measure base obstacle rate (oscillation counter ≥ N consecutive opposite actions).
- [ ] **T2** Overlay carrier: edit spec type (direction form + bypass form) → deterministic applier; apply-in-dequant-domain arm for a quantized twin of the fixture; untouched columns bit-identical assert.
- [ ] **T3** Gate battery (G1): (a) dose-response monotone non-increasing over pinned k ∈ {1,2,4,8}; (b) neighbor-layer isolation null (same edit at an uninvolved column set ≈ no change); (c) norm-matched random control, ≥16 BLAKE3-seeded arms (RandOpt posture); the statistic is the **paired LCB95 on (targeted − best random)** — the best random arm is selected ONCE on its mean over the eval protocol, then paired per-seed deltas are computed against that single arm (reading (i), not per-seed max-over-arms — the stricter reading is likely unreachable at 16 arms on a small fixture; the once-picked form still cannot produce a false win since the null is biased conservative); beating only the ensemble MEAN fails; (d) **pre-screen refusal arm**: no-degeneracy fixture ⇒ lane refuses, zero fake gain; (e) retention: clean-region accuracy held within ε (paired).
- [ ] **T4** G2/G3/G4: apply-at-load compose cost vs full bake; flag-off byte-identical boot; alloc-free apply (bounded scratch); rejected-edit revert leaves θ bit-identical (BLAKE3 before/after).
- [ ] **T5** Mint-once freeze: committed artifact envelope with provenance trailer (prompt digest / seed / author id, all optional except artifact BLAKE3); tamper refusal at load.
- [ ] **T6** Feature gate `weight_edit_overlay` (opt-in); GOAT verdict recorded in `.benchmarks/`; on gate failure: record negative (Issue-565 precedent) and demote — the carrier stays only as the SparseErrorBypass representation.

## Non-goals

- No LLM authoring lane here (hosted-author + weight-egress policy is `riir-train/.issues/665`'s scope; local Bonsai is not the capability class).
- No serving-rung wiring (rethink LAYA has no documented loop class — veto-gated consumer, see Research 648 §4).
- No training, no gradient descent anywhere; the runtime primitive stays modelless.
