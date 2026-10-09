# Issue 929: Loop Alignment Probe + DecodeReinject PoC (DiscoLoop, arXiv:2607.00341)

**Status:** Active
**Research:** [katgpt-rs/.research/614](../.research/614_DiscoLoop_Decode_Reinject_Between_Loops.md)
**Paper:** [arXiv:2607.00341](https://arxiv.org/abs/2607.00341) — DiscoLoop (NeurIPS 2026)
**Type:** PoC / proof — measurement-first, injection second, feature-flagged, default-OFF
**Created:** 2026-10-09

## Problem

DiscoLoop shows looped-transformer carry states are decodable-but-misaligned (`P(bridge|H)≈1.0`, `cos(H, W[bridge])≈0.27–0.33`) and that a training-free decode-reinject (`H ← (1−α)H + α·RMSNorm(W[argmax(W·H)])`, α≈0.5) lifts two-hop OOD 8.3%→~100% **on loop-trained weights**. Our looped path (`lt2_looped`, default-on) loops standard checkpoints that never saw looping in training — whether the misalignment signal even exists there is unmeasured (loop-intervention record on this stack: 0-for-3, Benches 847/850/906 + Issue 568).

## Plan of attack (probe-first — measurement before intervention)

### Phase 1 — Alignment probe (pure measurement, zero behavioral risk)

- [ ] **T1.1** Probe fn on the `forward_looped` path: per loop k, per probe position, compute `cos(H_t⁽ᵏ⁾, W[v̂])` with `v̂=argmax(W·H_t)` + top1−top2 margin. Reuse the logits scratch (G4: zero alloc). G2: one fused head-GEMV+dot per probe position.
- [ ] **T1.2** Fixture: two-hop compositional prompts with verifiable bridge entities (paper §2 generator shape: 500 entities, |R|=50, out-degree 10), run through a real looped checkpoint (Bonsai-27B PQ2_0 first per owner priority; MiniCPM5-1B for the cheap leg). **Precondition: verify tied embeddings per checkpoint** — the probe/re-anchor read the LM-head row; on an untied checkpoint, head-row vs input-embedding-row is a materially different operator. Record which before running.
- [ ] **T1.3** G1 gate: AUROC of cos(k)+margin(k) vs final-answer correctness. Record cos trajectories ID vs OOD-style splits.
- [ ] **T1.4** Kill criterion: AUROC bootstrap 95% CI lower bound < 0.6 ⇒ the signal is absent/weak on non-loop-trained weights ⇒ record negative, close this issue (**reopen trigger = riir-train Plan 451's trained artifact landing** — a loop-trained checkpoint re-arms the probe question immediately), route the question to Plan 451's trained-to-loop lane.

### Phase 2 — `LoopStabilityMode::DecodeReinject` (gated injection, opt-in feature)

- [ ] **T2.1** 5th enum variant (4th active sibling beside `InterLoopNorm`/`FixedAnchor`/`StateNoise`) in katgpt-types `enums.rs` + wiring in `variants.rs::forward_looped` at τ>0: `H ← (1−α)H + α·RMSNorm(W[v̂])`. **α=0 byte-identical to `None` — pin it in the same commit** (strongest G3 form).
- [ ] **T2.2** Health-gated α: `α_t = σ(a·cos_t + b·margin_t + c)` (sigmoid — house law + the paper's own Eq. 6 shape; the vocab softmax inside Φ is a decode readout, not a decision weight). Low confidence ⇒ α→0 ⇒ no-op.
- [ ] **T2.3** Soft arm: tempered top-k mixture Φ (k∈{16,64,128,256} sweep — re-measure the k=128 sufficiency on OUR vocab/dim geometry, don't import it).
- [ ] **T2.4** 3-arm paired eval (none / hard / soft) on the T1.2 fixture + PPL ladders flat on the riir-infer rigs (G3). Honest claim shape: paired delta sign on the high-margin subset — NOT the paper's 8.3→100.
- [ ] **T2.5** GOAT gate: G1 paired deltas, G2 bounded overhead, G3 bit-identity + flat PPL, G4 pre-allocated Φ/top-k scratch. Negative ⇒ record with the full posture sweep (Bench-906 discipline), feature stays dead-or-narrow.

### Phase 3 — Halt-on-readiness (into the SaddleEscapeGate / gain_cost_halt family)

- [ ] **T3.1** Halt-on-readiness as a NEW `TrapObservables` field consumed by `SaddleEscapeGate` (which already wraps `halt_decision` and already carries `decoded_key` decode evidence). ⛔ Do NOT overload `halt_decision`'s `cos_theta` — its semantics are fixed and INVERTED (`< 0` trips `HaltReason::Oscillation`, bench-pinned in `gain_cost_halt_bench.rs` G3/G4) vs this signal's high-alignment = composition-ready. Signal family: flip-stability (`decoded_key`, ships) ≠ decode-margin (T1.1) ≠ state-vs-manifold alignment (this) — three distinct signals from the same decode event. G2: equal-accuracy avg-loop cut on the default-on path. Thresholds: calibrate-once frozen table (freeze/thaw doctrine).

## Constraints

- Every arm behind a feature flag, default-OFF; α=0 bit-identity is the load-bearing G3 pin.
- Eval-side Δcos generalization predictor (riir-infer column) may pre-screen quant-retention rows but can never replace the lossy-surface per-family gate — validate before any proxy use.
- GPU exclusivity rule for any correctness gate run.

## Success criteria

Probe AUROC 95% CI LB ≥ 0.6 on ≥1 real checkpoint (signal exists) AND/OR paired injection delta > 0 at some posture with flat PPL elsewhere ⇒ GOAT adjudication + promotion decision. All-negative ⇒ permanent negative record beside Benches 847/850/906 (the class ledger), close with the Plan-451 reopen trigger.
