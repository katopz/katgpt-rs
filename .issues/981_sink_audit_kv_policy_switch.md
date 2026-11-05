# Issue 981: Sink-Audit KV Policy Switch — one-time per-position census selects SinkWindowPolicy posture

**Status:** Filed 2026-11-05 — POC, pre-registered gates below; from katgpt-rs Research 651 §3.3 (arXiv:2505.06708 "Gated Attention for LLMs", NeurIPS 2025 Best Paper). Revised same day after verdict round 1: the served flagship's full-attention layers are ALREADY gated (riir-infer Issue 594), so the audit is runnable NOW — Bonsai is the leading SinkFree candidate, not a contingency.
**Claimed:** zed-m3-i981sinkaudit, 1793849627, expires 2026-11-07T03:33:47+00:00

## Problem

`katgpt_core::kv_sink_window` (`SinkWindowPolicy`, `sink_pin_mask_into`; `n_sink` default 4, `policy.rs:77`) is consumed unconditionally by three lanes:

1. `riir-infer/src/deltanet/kv_evict.rs` (import at :49, pin at :299, `SinkWindowPolicy::new(4, …)` at :436) — eviction sink pinning.
2. `riir-infer/src/bin/needle_eviction_gate.rs` (:113 import, :1000 `SinkWindowPolicy::new(args.sinks, …)`) — the eviction-gate harness.
3. `riir-infer/src/tokengate.rs:48` — the token-gate reserve (`SinkWindowPolicy::default().n_sink` PLUS the final position; a correctness reserve for delimiter-class sinks, not an eviction budget).

Two failure modes, quantified by the paper:

1. **Sink-free models** (trained with post-SDPA output gates) waste the pin budget on positions the model does not attend to: gated F-Attn 0.048–0.073 vs ungated 0.297–0.467. `q8kv_sink_guard` — the f32 sink-rows sidecar in `riir-infer/crates/riir-infer-gpu/src/attention_q8kv_cubecl.rs` — carries rows for them.
2. **Sink-heavy checkpoints under position extension** degrade hardest (paper Table 5: ungated −41.6 RULER at the 32k→ext step vs gated −6.9; YaRN 128k: 31.65 vs 58.82). No measured flag routes a checkpoint to "pin + long-context risk" posture BEFORE serving.

**Why this is runnable NOW (round-1 correction):** Bonsai-2's full-attention layers already ship the paper's G1 gate (riir-infer Issue 594 — `attn_wq` → per-head interleaved `[q, gate]`, `attn_out *= fast_sigmoid(gate)` post-SDPA, qwen35-hybrid). The flagship is the primary SinkFree candidate; whether it actually reads sink-free is exactly what T1 measures.

## Mechanism (all offline; zero per-call cost — the Plan 287 G3 lesson)

ONE-TIME census pass per checkpoint (boot-time or CLI):

- **Scope: full-attention layers ONLY** (the qwen35-hybrid's `forward_attention_layer` sites). GDN layers have no softmax sink; a blended F-Attn dilutes the signal.
- **Per-position mass profile over the first N positions** (not token-0 alone — secondary/delimiter sinks at positions 1–3 must not hide behind a low token-0 share), aggregated from `sink_classify::SinkDiagnostic.strength` (already measures attention mass on sink positions, per head) plus `fitted_v_gate`-style norms for **M-Act** (mean per-layer max activation).
- Emit `{f_attn_token0, first_n_profile, m_act, verdict}`; verdict ∈ {SinkFree, SinkHeavy, **Undecided**}. The paper gives measured CLUSTERS (sink-free 0.048–0.073, sunk 0.297–0.467) but no decision boundary — **the cut points are CHOSEN, stated as chosen**: F-Attn ≤ 0.10 → SinkFree; ≥ 0.25 → SinkHeavy; 0.10–0.25 → Undecided (keep incumbent posture; disclose).

`kv_sink_window` gains `SinkWindowPolicy::from_audit(verdict, profile)`: SinkFree ⇒ `n_sink` set from the per-position profile with a **NON-ZERO floor (≥ 1)** until T4's end-to-end cell proves 0 safe, plus skip q8kv `sink_rows` growth; SinkHeavy/Undecided ⇒ incumbent pin, plus (SinkHeavy) one long-context-risk line. Flag-off byte-identical default.

## Tasks

- [ ] **T1 — census arm.** One-pass per-position F-Attn profile (first-N) + M-Act over FULL-ATTENTION layers only (reuse `sink_classify` geometry + `fitted_v_gate` norms); deterministic bytes, blake3 sidecar. Included sub-checks: verify the GPU flash path's gate epilogue applies the shipped gate (the CPU path is verified — forward.rs:854-857); read the dense qwen3.8 lane's gating status (fixture-pair input for G2).
- [ ] **T2 — policy selector.** `SinkWindowPolicy::from_audit` + wire-through in the two EVICTION lanes (`kv_evict.rs`, `needle_eviction_gate.rs`). `tokengate.rs` EXCLUDED from this scope with reason: its sink window PLUS final position is a correctness margin for logits/delimiter sinks, not an eviction budget — revisit only if the audit shows near-zero delimiter-sink mass there.
- [ ] **T3 — bench_841 extension.** Policy-switch arm proving verdict ⇒ correct pin posture; planted-sink canary must trip; population floor on scanned layers×heads×positions.
- [ ] **T4 — end-to-end cell.** On a SinkFree-read checkpoint: reduced pin ⇒ no long-context quality delta vs incumbent (the GOAT half). On a SinkHeavy checkpoint: pin ⇒ protected. Undecided cells disclosed, never folded.

## GOAT gates (pre-registered)

- **G1** determinism: same checkpoint → same census bytes.
- **G2** discrimination on a REAL gated-vs-ungated pair: primary fixture = the paper's released 1.7B baseline/gated pair (HF `QwQZh/gated_attention` — the id as linked from the paper's own abstract page; T1 resolves the exact current model ids/files before pinning the fixture — the caution from verdict round 2), the calibrated pair the threshold clusters come from; secondary = a held pair (Bonsai gated-hybrid vs a verified-ungated dense checkpoint; qwen3.8's status read in T1). Must reproduce the paper's class ordering.
- **G3** policy correctness: T3's switch arm.
- **G4** alloc-free census pass.
- Promotion to default only with T4's end-to-end pass.

## Provenance

- Research 651 §3.3; paper Table 4 (F-Attn 0.048–0.467 clusters; M-Act 94–1053), Table 5 (YaRN), Fig 4 (gating substitutes the sink's absorption function).
- Signal-diff vs Plan 287 (sink_aware_attn): 287 classifies NOP/Broadcast sinks PER CALL inside the engine's sigmoid attention and gates NOP sinks in the FORWARD (G3 latency FAIL ⇒ opt-in diagnostic). This issue is ONCE-PER-CHECKPOINT, offline, acting on the KV-EVICTION policy. Different consumer, different cadence — 287's G3 failure does not transfer; the audit can feed 287's classifier as a prior (composition, not replacement).
- Round-1 verdict correction (2026-11-05): the "may never fire on today's fleet" premise was WRONG — riir-infer Issue 594 shipped gated full-attention layers in the qwen35-hybrid (Bonsai); the reviewer's grep was coordinator-verified at `forward.rs:728-857` before this revision.
