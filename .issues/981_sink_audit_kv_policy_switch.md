# Issue 981: Sink-Audit KV Policy Switch — one-time per-position census selects SinkWindowPolicy posture

**Status:** T1 (decision half + census lane) + T2 + T3 LANDED 2026-11-05 (zed-m3-i981sinkaudit, session epoch 1793849627); T4 reduced-scale cell DONE — full-scale + the two real-pair arms are the named follow-ups. katgpt-rs `923aef6a4` (the audit vocabulary: `SinkAuditProfile`/`SinkAuditVerdict::classify` with the chosen 0.10/0.25 cuts stated as chosen, `SinkWindowPolicy::from_audit` — leading-mass pin, floor ≥1, never grows the incumbent, window carried — `accumulate_head_mass_into` + `finalize_profile`, the root `kv_sink_window` forward, 14 unit tests) + bench_981_sink_audit_switch_goat (T3: G2 discrimination planted 0.4716 vs free 0.0236 — both inside the paper's clusters; G3 switch grid; both planted-sink canaries TRIP; population floor 8192; G1 bit-identical; G4 alloc-free at alloc_tracking — ALL GATES PASS). riir-infer (this date, .benchmarks/947): the `sink_audit` census lane (cfg-gated Q capture riding `HybridForwardScratch`, full-attention layers only, bounded cache per Issue 929) + the `sink_audit` bin + `needle_eviction_gate --sink-audit N` wire-through; **the real gated checkpoint (Qwen3.5-0.8B-Base, Issue-594 gated shape) reads f_attn_token0 0.0764 → SinkFree → audited pin 4→1 — the paper's gated-cluster signature on a real model; G1 bit-identical (blake3 pinned)**; T4 reduced cell: 25% budget audited-vs-incumbent BIT-IDENTICAL (quality-neutral; the pin never bound), 2% budget DIFFERS (wire-through consumption proven on the real gate). T1's first sub-check verified at source: the GPU hybrid attention path applies the shipped gate epilogue (`forward_attention_layer_gpu` step 7 via `QwenAttentionDecodeGatedCubeCL` — the decode kernel applies sigmoid at its final write); qwen3.8 dense read: NO attention output gate (only SwiGLU MLP gates) — the natural ungated G2 arm, but GPU-only (no CPU forward).
**Lease closed:** zed-m3-i981sinkaudit, 1793849627 — T1 decision-half + T2 + T3 landed in the lease window; the census lane + real-model cell landed the same session (riir-infer bench 947)
**FOLLOW-UPS (the honest remainder):** (a) the REAL ungated G2 arm — qwen3.8 dense has no CPU forward and the paper's HF gated/ungated pair needs its own loader; the synthetic arms carry G2 on the arithmetic meanwhile. (b) the Bonsai-2-27B census — the flagship's serving path is the TERNARY loader + ternary forward body; the capture hook rides the dense qwen35 body only (the dense loader mis-shapes the PQ2_0 ternary file: in_proj_qkv row-overrun panic at katgpt-types dot.rs:879). (c) the full-scale T4 cell (64k-class long-context quality) — owner-adjacent runtime. (d) `q8kv_sink_rows` skip: the q8kv `sink_kv` sidecar is caller-sized and today no caller populates it (grep: only the gemma2 kv_cache `Vec::new()`), so the audited n_sink sizes it WHEN a caller arms it — no code now.
**Arming condition for the follow-up census runs:** same pre-registered gates; the ternary-body capture hook is the (b) precondition.
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

- [x] **T1 — census arm.** One-pass per-position F-Attn profile (first-N) + M-Act over FULL-ATTENTION layers only (reuse `sink_classify` geometry + `fitted_v_gate` norms); deterministic bytes, blake3 sidecar. Included sub-checks: verify the GPU flash path's gate epilogue applies the shipped gate (the CPU path is verified — forward.rs:854-857); read the dense qwen3.8 lane's gating status (fixture-pair input for G2). — LANDED: the decision half + accumulator/finalize in katgpt-core (`923aef6a4`), the capture + harvest + bin in riir-infer (bench 947); GPU epilogue verified at source; qwen3.8 dense = ungated (GPU-only). M-Act defined as the mean per-layer max V-row L2 norm (fitted_v_gate norm class, disclosed never verdict-bearing).
- [x] **T2 — policy selector.** `SinkWindowPolicy::from_audit` + wire-through in the two EVICTION lanes (`kv_evict.rs`, `needle_eviction_gate.rs`). `tokengate.rs` EXCLUDED from this scope with reason: its sink window PLUS final position is a correctness margin for logits/delimiter sinks, not an eviction budget — revisit only if the audit shows near-zero delimiter-sink mass there. — LANDED: from_audit + the needle-gate `--sink-audit N` inline census (consumption proven by the differing 2%-budget cell); kv_evict's lane consumes audited policies through its EXISTING SinkWindowPolicy parameter (zero-diff wire-through — LayerEvict/EvictorState take any policy; the kv_evict w11 pin law already asserts the honored-pin semantics). tokengate exclusion stands as written.
- [x] **T3 — bench_841 extension.** Policy-switch arm proving verdict ⇒ correct pin posture; planted-sink canary must trip; population floor on scanned layers×heads×positions. — LANDED as `bench_981_sink_audit_switch_goat` (the name bench_841 is TAKEN by the legal-token-set bench — a DIFFERENT Issue 841): ALL GATES PASS incl. alloc_tracking G4.
- [ ] **T4 — end-to-end cell.** On a SinkFree-read checkpoint: reduced pin ⇒ no long-context quality delta vs incumbent (the GOAT half). On a SinkHeavy checkpoint: pin ⇒ protected. Undecided cells disclosed, never folded. — REDUCED CELL DONE (8k/4-needle, bench 947): audited-vs-incumbent bit-identical at 25% budget (neutral); the full-scale cell + the SinkHeavy-checkpoint protection arm are the follow-ups (no local SinkHeavy checkpoint census yet — needs (b)).

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
