# Research 614: DiscoLoop — Decode-Reinject Between Loops (Self-Anchored Discrete Channel)

> **Source:** "DiscoLoop: Looping Discrete Embeddings and Continuous Hidden States for Multi-hop Reasoning" — [arXiv:2607.00341](https://arxiv.org/abs/2607.00341), Fu, Guo, Wang, Zhu, Lee, Jiao, Russell, Mei (UC Berkeley + Princeton, NeurIPS 2026)
> **Date:** 2026-10-09
> **Status:** Active — modelless track filed as Issue 929 (probe-first PoC); training track filed as riir-train Plan 451 (toy composition lab)
> **Related Research:** 602 (LoopCD — the 0-for-3 guidance negative), 073 (LT2 looped inference), 097 (training-free loop wrapper), 414 (loop stability / readout blind spot)
> **Related Plans:** 108 (lt2_looped), 136 (tf_loop), 428 (loop_stability_fix), 304 (gain_cost_halt), 431 (cross_stage_relocation), 617 (loop_guidance, NEGATIVE); riir-train 451 (this note's track-c plan)
> **Classification:** Public

---

## TL;DR

Looped transformers fail multi-hop composition not because the intermediate answer is missing but because it is **decodable yet misaligned**: after loop 1 the bridge entity reads out through the LM head with P≈1.0 while `cos(H, W[bridge])` sits at 0.27–0.33 — the next loop consumes a noisy continuous mixture instead of a clean embedding. A **training-free** intervention (decode → RMSNorm → convex-mix, α≈0.5) lifts two-hop OOD accuracy 8.3% → ~100% on their loop-trained weights. For us: (a) a **loop alignment probe** (cosine of the carry state against the top-decoded token embedding) as a new runtime/eval observable — a different signal than `gain_cost_halt`'s update-direction cosine; (b) a **`LoopStabilityMode::DecodeReinject`** sibling — self-anchored re-injection toward the model's OWN decoded embedding, health-gated by sigmoid, α=0 bit-identical; (c) a training-track toy lab + trained-to-loop CPT go/no-go in riir-train.

**Distilled for katgpt-rs (modelless, inference-time):** between iterations of a looped state machine (evidence class: loop-trained transformers on synthetic KG composition — transfer to non-loop-trained checkpoints is exactly what Issue 929's probe measures), the carry state drifts off the discrete manifold the next stage was designed to consume. Re-anchoring toward `Norm(W[argmax(W·h)])` — the model's own decode of its own state — is a zero-parameter, zero-training correction operator, and the alignment cosine `cos(h, W[v̂])` is a cheap health signal that predicts composition success. The gate is sigmoid (paper's own Eq. 6); the vocab softmax inside Φ is a decode readout (a legitimate softmax use — same class as an LM head), not a decision weight.

**Δ 2026-10-10 (Bonsai-2 leg):** the probe question's capable-checkpoint surface LANDED code-only in the sibling `riir-infer` (owner directive; the kimi legs' label axis was dead — Benches 927+930 UNDECIDABLE): the looped probe runner over the REAL qwen35 hybrid forward, PACKED ternary load (≈7.3 GB, never f32-dequantized), K weight-shared stack re-entries at the answer position, packed-head probe variant (`probe_alignment_with_row`), K=1 bit-identity to the stock forward pinned by test. Run pending box availability — see Issue 929 §T1.2b.

---

## 1. Paper Core Findings

1. **Task**: implicit two-hop composition in ONE forward pass (facts in weights, no CoT): `(a,r1,b) ∧ (b,r2,c) ⇒ (a,r1,r2,c)`. Two entity-disjoint KGs (500 entities, |R|=50, out-degree 10): train compositions on G_A only, atomic facts of both — `test_ood` = compositions over G_B (never seen composed).
2. **Vanilla looped transformer** (K=2, weight-shared): fits train perfectly, but plateaus at 71.1% ID / **8.3% OOD**.
3. **Mechanistic diagnosis** (logit lens at the bridge position): Stage-1 `P(b | H⁽¹⁾) = 1.000` on BOTH splits — the bridge is decodable and the second-hop memory exists (same block recalls all atomic facts in one loop). But `cos(H⁽¹⁾, W[b])` = 0.327 ID / 0.266 OOD. **Decodable ≠ aligned.** The second loop is asked to consume a distribution (noisy hidden states) it was never trained on (clean embeddings) — a representation mismatch, not a storage gap.
4. **Training-free intervention (Eq. 3)**: between loops, at the bridge position only: `H ← (1−α)·H + α·RMSNorm(W[argmax(W·H)])`. α=0.1 lifts OOD 8.3%→25.9%; **α≈0.5 → ID and OOD both ≈100%** (10× OOD gain, zero training).
5. **DiscoLoop architecture (Eq. 4–6)**: lift the intervention to every position/loop, differentiable: `H̃⁽ᵏ⁺¹⁾ = H⁽ᵏ⁺¹⁾ + α⁽ᵏ⁾ ⊙ RMSNorm(Φ(H⁽ᵏ⁺¹⁾))` with `Φ(h) = Σᵥ pᵥ(h)·W[v]` (tempered softmax, τ=1; top-k=128 suffices at 440M), token-wise sigmoid gate `α_t = σ(w_α·Φ(h_t) + b_α)` (d+1 params), `α⁽ᴷ⁻¹⁾ ≡ 0` (final readout untouched).
6. **Results**: near-perfect 2-hop ID/OOD; 3-hop OOD ~65% where vanilla loop ≈ 0% (misalignment compounds per hop); synthetic natural language ~95% OOD; 440M/20B-token pretraining beats vanilla loop + PonderLM (50.5 vs 49.3/49.8 avg over 7 zero-shot benchmarks; lowest loss after ~13B tokens).
7. **Training dynamics**: the discrete channel accelerates the Stage-1-collapse phase transition (single-loop memorization suppressed at ~epoch 400 vs >2000) — composition is offloaded to the recurrence earlier and sharper. This half is irreducibly training-track.

## 2. Distillation

### 2.1 Path 0 inventory (component table)

| # | Paper component | Extraction (closed-form w/o GD?) | Coverage in our stack (signal-diff'd) | Disposition |
|---|---|---|---|---|
| 1 | Alignment cosine `cos(H_t⁽ᵏ⁾, W[v̂])`, `v̂=argmax(W·H)` — health metric (ID 0.327 vs OOD 0.266) | **YES** — pure probe, reuses the LM-head GEMV + one dot | **NO ship.** Nearest: `GainCostLoopHalter::halt_decision(…, cos_theta)` (Plan 304) — but its cosine is **update-direction alignment across loops** (oscillation detection, `cos_theta < 0` trips); the paper's is **state-vs-decode-manifold alignment** (representation quality). Different signal, same socket. | Issue 929 (probe-first beachhead) |
| 2 | Decode-margin confidence (top1−top2 at the probe position) | **YES** | **PARTIAL at katgpt-core level.** `SaddleEscapeGate`/`TrapObservables` (`saddle_escape.rs`) already consumes `decoded_key: Option<u64>` (hash of the loop's decoded answer) via `FlipDetector`'s flip-rate EMA — decode EVIDENCE ships one layer up from the halter. Signal-diff: flip-STABILITY of the decode (same answer across loops?) vs decode-CONFIDENCE (margin) vs state-vs-manifold ALIGNMENT (cosine) — three distinct signals from the same decode event; margin + cosine do not ship. | Issue 929 |
| 3 | Hard re-anchor Eq. 3 (`H ← (1−α)H + α·RMSNorm(W[v̂])`) | **YES** — zero params | **NO ship, one near-cousin.** `LoopStabilityMode::FixedAnchor` (Issue 698 T2) re-injects a **frozen h⁽⁰⁾** (static content hoisted once; GRT Table 11: +0.70 nats vs drifting carry). Signal-diff: frozen-hoist vs **per-loop model-decoded** anchor + RMS norm-match. `cross_stage_relocation` (Plan 431) moves RAW residual state between stages — no decode, no norm-match, no mix. Convergent evidence: both GRT and DiscoLoop say carry-state quality dominates; the delta is the anchor SOURCE. | Issue 929 (`LoopStabilityMode::DecodeReinject`, 5th variant) |
| 4 | Soft Φ decode-then-encode (tempered top-k mixture) | **YES** as a fixed operator | NO ship on the transformer path. Shape-precedent only: the healer's retrieval fan-out (query → top-k → weighted mix) is the same shape on a different surface (see §2.3 fusion idea). | Issue 929 (soft arm of 3) |
| 5 | Health-gated α — sigmoid over diagnostics (modelless reformulation of the trained Eq. 6 gate) | **YES** — `α_t = σ(a·cos_t + b·margin_t + c)` | PARTIAL: sigmoid gates are the house pattern, but none consumes decode-alignment signals. | Issue 929 |
| 5b | *(sibling, from the same enum)* `StateNoise { scale }` (Issue 698 T6, GRT) — BLAKE3-seeded Gaussian state perturbation per loop | n/a (shipped sibling, not a paper component) | SHIPS — the OPPOSITE-polarity sibling in `LoopStabilityMode`: injects noise to escape traps vs DecodeReinject removing misalignment toward the decode manifold. Listed for inventory completeness over the socket family. | — |
| 6 | Halt-on-readiness — "decodable-then-aligned precedes composition" ordering law | **YES** (ordering law) | PARTIAL: the socket family ships (`GainCostLoopHalter` + `SaddleEscapeGate` wrapping it with decode-flip evidence); the criterion is new (see rows 1–2). ⛔ Socket law: `halt_decision`'s `cos_theta` has FIXED INVERTED semantics (`< 0` trips `Oscillation`, bench-pinned) — the alignment signal (high = ready) must be a NEW `TrapObservables` field consumed by `SaddleEscapeGate`, never an overload of `cos_theta`. | Issue 929 |
| 7 | Δcos generalization predictor (checkpoint-level, offline) | **YES** (eval column) | NO ship (no alignment column in the PPL/argmax ladders). Must be VALIDATED before use as a proxy — it can pre-screen, never replace, the lossy-surface per-family gate. | Issue 929 (eval column arm) |
| 8 | DiscoLoop recurrence (trained Φ + learned gate, from-scratch / CPT) | **NO** at inference — the paper's central training-dynamics effect (Stage-1 collapse acceleration) does not exist without training | N/A | riir-train Plan 451 |
| 9 | Stage-1 collapse timing as a training health signal | **NO** — training-dynamics claim | N/A | riir-train Plan 451 |

### 2.2 The primitive (why this is DRY-consistent, not greenfield)

The sockets already exist and are GOAT-gated; this paper fills the one missing sibling:

- `LoopStabilityMode::{None, InterLoopNorm, FixedAnchor, StateNoise}` — norm control (428), frozen-anchor control (698 T2), seeded noise injection (698 T6). **`DecodeReinject` = the 5th variant / 4th active sibling, the content-adaptive one**: anchor = the model's own decode, refreshed per loop, RMS-norm-matched to the stream (vs FixedAnchor's frozen hoist and StateNoise's injected noise — opposite polarity). α=0 must be **byte-identical** to `None` (the kill-switch-bit-restores pattern; strongest G3 form).
- `GainCostLoopHalter::halt_decision(τ, gain, cost, cos_theta)` + `SaddleEscapeGate`/`TrapObservables` one layer up (already carries `decoded_key` decode evidence) — the halt-on-readiness criterion lands as a NEW `TrapObservables` field consumed by the gate (which wraps the halter). ⛔ Never overload `halt_decision`'s `cos_theta`: its semantics are inverted (`< 0` = oscillation trip, bench-pinned) vs high-alignment = ready.
- `ForwardContext` (`prev_h`, `loop_anchor` pre-allocated buffers) — the probe/reinject scratch rides existing allocation discipline (G4).

### 2.3 Fusion

Closest cousins: **A** = `loop_stability_fix`/`FixedAnchor` (norm + static anchor), **B** = `gain_cost_halt` (update-direction cosine halting), **C** = `cross_stage_relocation` (raw state relocation), **D** = Research 602 LoopCD (external-target contrast guidance, NEGATIVE 0-for-3).

**Novel combination none has alone**: a *self-anchored* inter-loop operator — the anchor is neither external (D's hand-constructed target: the exact class that failed), nor frozen (A: stale by construction), nor raw (C: passes the noise through) — it is the model's **own argmax decode**, norm-matched, sigmoid-health-gated so that low decode confidence ⇒ α→0 ⇒ no-op. The probe (row 1) additionally converts "does any of this transfer to non-loop-trained checkpoints?" from an argument into a one-dot-product measurement, and its cosine feeds BOTH the gate (row 5) and the halter (row 6).

**Fusion idea, novelty TBD (recorded, not filed)**: Φ's shape (query → top-k candidates → weighted embedding mix → re-inject) as an **iterative-retrieval reformulation lane** for the healer's fan-out (riir-refine/riir-rag) — today single-shot; the paper's principle would make pass 2's query a sigmoid-gated mix of pass 1's top-k rule embeddings. Novelty unverified (no iterative-retrieval grep run); revisit if the transformer-surface PoC lands.

**Belief-substrate convergence (no new files — already ships)**: riir-ai's two-brain `SpatialBelief` (zone-level KG triple = discrete channel + stale `last_known_pos` = continuous channel) IS the paper's core architectural principle — "the recurrence should carry both a discrete embedding channel and a continuous hidden-state channel" — arrived at independently. The paper is third-party validation of the mixed-channel carry design, at transformer scale.

### 2.4 Game-context + consumer-context reframing

- **Game**: per-NPC cognition loops (perception → belief → drives) are recurrences whose carry states face the same mismatch class — the belief produced by stage N is not the representation stage N+1 consumes cleanly. The alignment-cosine as a **consolidation/salience signal** (how close is this NPC's belief to a nameable discrete state — "fleeing", "hunting") is the game-side read; gated re-anchoring toward the discrete state is the runtime self-adaptive half. (Routing: the probe operator is generic math → katgpt-rs public; game wiring would be riir-ai's call, not filed here.)
- **Healer (priority #2)**: single-shot retrieval today; the iterative-reformulation idea above is the honest one-liner — a bare "no" would be the canonical-failure-#5 shape, and the shape-precedent (fan-out top-k weighted mix) is real but is NOT coverage of an iterative loop.

## 3. Verdict

**Pinned claim (novelty, §4-checked):** *Self-anchored decode-reinjection between loop iterations — re-anchoring the carry hidden state toward the model's OWN top-decoded token embedding (hard argmax or tempered top-k mixture), RMS-norm-matched, sigmoid-health-gated, α=0-bit-identical — for weight-shared looped inference, plus the state-vs-decode-manifold alignment cosine as a runtime/eval observable; distinguished from PonderLM (embedding-only recurrence, trained from scratch), LoopCD (external contrast target — our 0-for-3 negative), FixedAnchor (frozen h⁽⁰⁾, not live decode), cross_stage_relocation (raw state, no decode/norm/mix), logit lens (read-only), and the Jacobian-lens injection swap (causal probe, not an intervention).*

**Per-track verdicts (one per track, never pooled):**

| Track | Verdict | Tier | Files |
|---|---|---|---|
| (a) modelless inference | **GAIN** | GOAT-candidate (pending PoC) | katgpt-rs Issue 929 (probe → gated DecodeReinject → halt-on-readiness), feature-flagged, default-OFF |
| (b) self-adaptive runtime | GAIN (rides (a)) | the health-gated α IS the self-adaptive instance | folded into Issue 929 |
| (c) model-based training | **GAIN** (secondary by serving-envelope fit) | recipe + falsifier | riir-train Plan 451 (toy lab 10–25 GPU-hrs; CPT go/no-go 5–15 GPU-hrs) |

**Not Super-GOAT**: Q3 (product selling point) fails — no "our NPCs do X" claim; Q2 is conditional on transfer to non-loop-trained checkpoints. Q1 holds (mechanism verified unshipped via signal-diffs above); Q4 partial (loop line + halting + eval rigs).

**MOAT gate:** katgpt-rs = public generic math (probe + re-anchor operator, no game/chain/shard semantics) — correct public home; riir-train plan = private recipe — correct private home. Nothing private-class in this note.

**Weakest point (named by us):** the paper's evidence is on **loop-trained** weights; our default-on looped path (`lt2_looped`) loops **standard checkpoints that never saw looping in training** — the exact regime where decode confidence may be too low for the paper's dynamics to exist. Our own record agrees: loop_guidance 0-for-3 (Benches 847/850/906) and Issue 568's fixed-embedding injection showed no transfer on untrained substrates. This is why Issue 929 is **probe-first**: measure whether the bridge is even decodable + misaligned on our path before any injection ships. Honest claim shape for the injection arm = paired delta sign on the high-margin subset, NOT the paper's 8.3→100.

## 4. Prior art (§4 searches run)

- DiscoLoop itself is the origin of the decode-reinject intervention + mixed-channel recurrence (searched verbatim; alphaxiv/HF/Awesome-Loop-Models all index it as such).
- **PonderLM** (Zeng et al., arXiv:2505.20674): embedding-space-ONLY recurrence (discards continuous state) — closest published baseline; DiscoLoop beats it (50.5 vs 49.8) keeping both channels.
- **Logit lens** (nostalgebraist): read-only intermediate decode — no reinjection.
- **Jacobian lens on looped transformers** (arXiv:2609.01924): injection-swap is a causal ANALYSIS probe on trained looped models (Ouro/Huginn), not a training-free accuracy intervention.
- Adjacent families, one-liners so no future reviewer "discovers" them: **ACT/PonderNet halting** (row 6's halt-on-readiness ancestry — halting machinery, no decode-alignment criterion); **ITI / inference-time steering** (intervenes on activations with EXTERNAL learned directions — different anchor source, not loop-scoped, not self-decoded); **Self-Refine / iterative reprompting** (text-space round-trip through the full tokenizer — full re-encode, not a hidden-state convex mix).
- Kohli et al. 2026 (arXiv:2604.07822): vanilla looped transformers on implicit multi-hop — the baseline DiscoLoop fixes.
- Internal: Research 602 (LoopCD negative), Issue 568 (injection regime-split no-transfer), Plan 428/698 (norm/frozen-anchor), Plan 431 (relocation) — all signal-diffed in §2.1.

## 5. PoC protocol (Issue 929's gate design)

1. **Probe (E1/E2)**: real checkpoint looped via `forward_looped` (Bonsai-27B PQ2_0 first per owner priority), T∈{2..12}, two-hop compositional prompts with verifiable bridge entities. **Precondition: check tied embeddings per checkpoint** (Φ and the re-anchor read the LM head row; on an untied checkpoint, head-row vs input-embedding-row is a materially different operator — record which before running). G1: AUROC of cos(k) + margin(k) vs final-answer correctness, with a bootstrap 95% CI — kill bar is a CI-informed margin (LB ≥ 0.6), not a bare > 0.5 (noise-adjacent bars invite zombie issues). G2: one fused head-GEMV+dot per probe position. G4: zero alloc (reuse logits scratch). **No transfer assumption needed — it's a correlation measurement.**
2. **Injection (E3/E4/E5)**: 3-arm paired (none / hard α=0.5 / soft top-k), health-gated α. G1: paired per-prompt deltas, gated subset. G3: **α=0 byte-identity pin** + PPL ladders flat on the riir-infer rigs. G2: one head-GEMV per probe position per loop ≪ one block forward.
3. **Halt (E7)**: halt-on-readiness as a NEW `TrapObservables` field consumed by `SaddleEscapeGate` (which wraps the halter and already carries `decoded_key` decode evidence) — never an overload of `halt_decision`'s inverted `cos_theta`; G2 = equal-accuracy avg-loop cut on the default-on path. Thresholds = calibrate-once frozen table (freeze/thaw doctrine — a frozen calibration is not training).

Kill criteria: probe AUROC 95% CI LB < 0.6 (signal absent/weak on non-loop-trained weights) → record negative, close 929 **with the reopen trigger = riir-train Plan 451's trained artifact landing** (a loop-trained checkpoint re-arms the probe question immediately — the operator is wanted again the moment 451 succeeds); injection paired delta ≤ 0 at every posture → record negative with the posture sweep (the Bench-906 discipline).

## 6. References

- [arXiv:2607.00341](https://arxiv.org/abs/2607.00341) + blog [hengyuf.github.io/discoloop](https://hengyuf.github.io/discoloop/)
- Our loop line: Plans 108/136/428/304/431/617, Issues 568/698, Benches 847/850/906, Research 073/097/414/602
- riir-train Plan 451 (track-c), Plan 449 (CPT precedent)
