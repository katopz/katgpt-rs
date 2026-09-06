# Research 539: SparDA — Decoupled Lookahead Selection & the Prefill Offload Schedule

> **Source:** [SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2606.04511) — Yaosheng Fu, Guangxuan Xiao, Xin Dong, Song Han, Oreste Villa (NVIDIA / Thinking Machines Lab / ByteDance Seed / MIT), arXiv:2606.04511, 2026-06-03. Code: https://github.com/NVlabs/SparDA
> **Date:** 2026-09-06
> **Status:** Gain — per-track outputs filed this session: (a) modelless/league: katgpt-rs Issue 730 (deterministic prefill KV-offload double-buffer); (b) game runtime: riir-ai Issue 880 (belief-Forecast Warm-tier prefetch, fusion idea); (c) training: riir-train Issue 524 (Plan 337 recipe deltas, ~0 GPU-h).
> **Classification:** Public note (katgpt-rs); mechanisms route per track — the trained Forecast does NOT ship modellessly (no sparse-pretrained backbone in the stack).
> **Related Research:** 436 (FlashMemory — the SHIPPED corridor this paper is the per-layer sibling of; carries an earlier PASS-Redirect on this paper, re-evaluated below), 176 (Vortex), 225 (MSA), 399 (hierarchical landmark), 523 (H2O — the dense-trained-selector collapse warning), 483 (KEEP)
> **Related Plans:** riir-train Plan 337 (FlashMemory indexer training — the recipe deltas' home)
> **Related Issues:** katgpt-rs 730 · riir-train 524 · riir-ai 880
> **Panel:** two-advocate adversarial round executed 2026-09-06 (No-GD + model-based); this note is the coordinator merge. Discards carry auditable reasons inline.

---

## TL;DR

SparDA adds a **fourth per-layer projection, the Forecast** (`F_l`), that predicts the top-k KV blocks layer l+1 will need, decoupling sparse selection from the attention query. Two payoffs: (1) one-layer-ahead selection exposes the next layer's memory-access pattern early enough for **CPU→GPU prefetch to overlap layer execution** (persistent UVA kernel, batch-adaptive CTA count); (2) selection itself gets **~G× cheaper** (one Forecast head per GQA group instead of G query heads, softmax skipped). Trained via KL against the original selector's block-attention distribution — <0.5% params, backbone frozen. 1.25× prefill / 1.7× decode over sparse-offload; 5.3× vs non-offload via feasible batch.

**The finding the paper doesn't state and our corridor needs:** at **prefill**, lookahead needs **no learner**. Causal attention at layer l attends every prior chunk, so the next layer's KV demand is a *known deterministic schedule* — a double-buffer that streams layer l+1's KV slice from CPU while computing layer l of the current chunk gets the overlap with **zero prediction and zero training**. The trained Forecast only earns its keep at decode sparsity — which is the half FlashMemory's τ=64 periodic scoring (R436, **shipped**, Bench 671: 1.8× decode at 64K on 4090) already largely covers. On our stack the actionable lever is therefore the **prefill offload schedule** (Issue 730), not the Forecast.

---

## 1. Paper mechanism (what to steal vs what to skip)

### 1.1 Architecture

- `(Q_l, K_l, V_l, F_l) = φ_l(X_l)` — Forecast rides the same linear projection.
- `B_{l+1} = B_init ∪ B_local ∪ f_top(F_l K̃_{l+1}^⊤, k)` — layer l's Forecast scores layer l+1's compressed keys. Attention at l+1 still uses the real `Q_{l+1}` (Eq. 5) — selection and attention are decoupled roles.
- Layer 0 has no prior Forecast → a separate same-layer projection `F_0^cur` (cold-start fallback, always declared). Final layer's F is unused.
- Compressed keys `K̃^{cache}` stay GPU-resident and update incrementally; `B_{l+1}` doesn't depend on layer l+1's newly appended keys (selection excludes init/local) — that is what makes the prediction well-posed before the next layer exists.

### 1.2 Compact indexer

Once selection is decoupled from Q, it no longer needs the query-head layout: **one Forecast head per GQA group** (= one per KV head), no softmax (no cross-head sum needed — scores are summed anyway). Measured: block-selection cost −2.5× at 128K prefill, >2× at decode; decode selection stays near-flat with context.

### 1.3 Training recipe (riir-train material — full analysis in Issue 524)

- Only the Forecast projections train (33.5M = 0.41% of 8B); backbone frozen; **KL against the selector's shared importance score** on a **top-k partitioned distribution**: k selected blocks individually + all remaining mass in one **rest bucket**, renormalized. The rest bucket keeps out-of-set logits in the gradient.
- **Fine-grained supervision:** target computed at compression window (2,1), max-pooled to the (32,16) inference grid before the loss. Ablation: RULER +3.0, reasoning +2.2 — *supervise finer than you infer, then pool.*
- Schedule: AdamW 5e-4 constant, 2000 steps, batch 32, BF16, clip 0.5, ProLong-64K, 65K seq; 48h on 32×H100 (live forwards at 65K — this is the unaffordable half at our scale; Plan 337's offline-label pipeline exists precisely to avoid it).

### 1.4 Prefetch runtime

Persistent UVA Triton kernel; fixed CTA set continuously drains block-transfer tasks in one launch. **Batch-adaptive CTA allocation** (Table 7): 16 CTAs at B<32, 32 otherwise on H100; 16/32 with threshold 64 on A100 — device-quantized swept launch table. **Regime finding (Table 10):** overlap pays only above a crossover batch (B16 where prefetch and layer time balance; at B64 full-SparDA ≈40% faster than no-prefetch; at B4 the prefetch pipeline is net overhead).

### 1.5 Results

- Accuracy: matches or improves the sparse baseline on both 8B models (NOSA-8B avg +2.3); **beats Sparse on RULER at every length**, gap widening to +4.3 at 128K.
- InfiniGen (the training-free hidden-state-proxy prior art) **collapses** (−6 to −13 avg) — adjacent-layer hidden similarity is not a reliable cross-layer proxy. This is the paper's own evidence that a *deterministic* proxy alone is insufficient (relevant to our fusion gate below: any proxy-based forecaster must be gated against exactly this failure).
- Limitations (paper §6): bounded by base sparse quality; extends naturally to token-level DSA (future work).

---

## 2. Relationship to the shipped corridor (R436) — redirect re-evaluated

R436 carried a PASS-Redirect on this paper (written 2026-06-17, pre-shipping) with two dismissal grounds. Both are superseded:

| R436 redirect claim | 2026-09-06 reality |
|---|---|
| "the UVA persistent kernel is H100/A100 PCIe serving infrastructure, **out of scope for the CPU/modelless stack**" | The league lane IS a PCIe machine (4090, 24GB) with a measured KV wall (R436's own motivation: ≈67GB at 256K vs 24GB VRAM). PCIe offload is in-scope for the 4090 lane; only the M3 (unified memory) is exempt. |
| "neither applies at **micro-transformer scale** (n_layer=1)" | FlashMemory **shipped and validated** (Issue 584 closed; Benches 021–026 + 671; 1.8× decode at 64K on 4090). The corridor is multi-layer and live; the dismissal's premise no longer describes the stack. |

**What survives the redirect:** per-layer lookahead (SparDA) and periodic τ=64 refresh (FlashMemory) remain two solutions to the same amortize/overlap problem on **different axes** — layer vs time. They compose: τ=64 rescoring is the periodic correctness anchor; a per-layer (or per-window) forecast feeds prefetch. The genuinely new extraction is §3.1 below — neither the paper nor R436 states the prefill case.

---

## 3. Three-track verdicts

### 3.1 Modelless / league (track a) — the prefill decomposition → Issue 730

The No-GD advocate's structural finding, adopted:

- **At 256K, prefill KV write volume alone exceeds VRAM** — offload is unavoidable at prefill *regardless of compression* (compression governs what you READ; there is nothing compressed yet while prefilling).
- **Order-of-magnitude (derived from the ≈67GB wall):** ~256KB/token total KV → ~8MB/layer per 2048-token chunk → fetch:compute ≈ 1:1 at the tail chunks → **un-overlapped offload ≈1.5×'s the prefill wall; a double-buffer hides it.** No learned predictor, no selector — causal attention's next-layer demand is deterministic.
- **Decode is NOT the wall here:** a 2048-token chunk is ~8MB ≈ 0.3ms of PCIe vs ~11ms/token compute — bandwidth is not the decode bottleneck; the decode lever is bounded by cold-miss rate and must be measured before arming. FlashMemory's τ=64 batched scoring already collapsed decode selection cost ~64×, so SparDA's decode selection headline has little room left on our stack.
- **⚠ Hybrid-layer confound (verify-first, T0 of Issue 730):** Bonsai is a DeltaNet hybrid with a minority of full-attention layers; if the dspark checkpoint's attention-layer count is small, the 67GB wall shrinks proportionally and this issue closes as N/A. Recompute the wall from the GGUF header before any engineering.
- **MRCR composition flip:** FlashMemory's measured dense-memory failure mode (R436 §1.6) is where offload changes category — compression collapses on scattered-needle retrieval; offload keeps **full KV addressable** at CPU cost, and prefetch makes the lossless lane affordable. G8 of Issue 730 is compression-only vs compression+offload-fallback.
- **DRY constraint:** prefetch must compose INTO the FlashMemory cold/hot pool promotion path — a second offload system beside it is a violation.

**Refused (auditable discards):**
- *Compact per-group scorer applied to SP-KV GateBias* — REFUSED on semantics: GateBias is a per-(position, head) attention-VALUE modifier, not a block selector; nothing in the paper argues gates are group-correlated, and collapsing per group changes every head's semantics. (The +7–12% gate overhead from Issue 727 may still motivate a group-scan experiment, but on OUR overhead budget, not licensed by SparDA.)
- *Trained Forecast projections on Bonsai/qwen3.8* — kill (model-based advocate, §3.3).
- *CTA constants / crossover numbers* — hardware facts; pattern yes, numbers no (re-sweep on 4090/M3 before writing any down; the B104 no-harm rule applies to any bucket table we ship).

**Fusion ideas recorded (GOAT-gated, unfiled — pursue on demand):**
- **F1 — Ridge-overlay Forecast (the "Forecast without GD"):** fit `s_j = h_l^⊤ M_l K_j` (low-rank M_l) by closed-form ridge least squares on a frozen calibration snapshot; ship M_l frozen + BLAKE3-committed; updates only via freeze/thaw. **Gate:** held-out next-layer top-k recall must beat BOTH full QK scoring's cost point AND the unfitted identity-metric proxy (InfiniGen's statistic) — the paper measured InfiniGen collapsing, so beating the proxy is the bar that proves the fit earns its bytes. Honest cost: per-snapshot refit on every freeze.
- **F2 — Deterministic forecaster stack:** layer l+1 reuses layer l's selected set (IndexCache, arXiv:2603.12201 — published, training-free, eliminates up to 75% of indexer compute in DSA) ∪ hidden-state-proxy top ∪ sigmoid-decay hot-set (`σ(−λ·(t − last_hit))` — literally `GenericSpatialBelief::decay_confidence`), with τ=64 full rescoring as the correctness anchor and the 3-layer OR-mode union providing graceful degradation for a superset selector. **Gate:** recall vs true selection WITH the failure tail reported (topic-switch regime); G8 MRCR A/B.
- **F4 — Compact group-sum scorer** for `PerGroupTopKRouter`/FlashMemory (the portable piece): rank blocks by `Σ_{heads∈group} raw_logit`, sigmoid-threshold readout (our convention) — no softmax, ~G× scoring-pass reduction. **Gate:** G1 = argmax-identity + max_abs band (NOT bit-identity — group-sum-of-exp is not a monotone-invariant of group-sum; scores legitimately move; the certified form is the Bench-773/Issue-775 shape). Softmax-skip validity is entropy-profile-dependent — measure on Bonsai, never inherit from the paper's models.

### 3.2 Self-adaptive / game runtime (track b) → riir-ai Issue 880 (fusion idea, novelty TBD)

Pinned claim (§4 precondition): *"Belief-driven Warm-tier prefetch for MMORPG NPCs: a per-NPC think-brain projection (dot+sigmoid over zone embeddings + decay confidence) predicts which zone/KG/shard rows the entity's next cognition stage needs, fetching them from the Warm tier during the current tick so the fetch overlaps hot-path compute."*

- Fusion map: SparDA's Forecast principle × the two-brain model (prediction is think-brain; prefetch is local consumption, one-way gated — no sync-boundary violation) × Warm tier (ndb `LocalKvStore` / workerd DO rows) × `decay_confidence` (gate prefetch on belief confidence — never prefetch what the NPC is about to forget) × `sleep_time_reload` (the OFFLINE half already ships; this is the online per-tick half).
- Prior-art sweep: open-world streaming prefetch = static world geometry, not per-entity cognition memory; TierKV (arXiv-class KV tiering with PCIe/NVMe prefetch overlap) = serving-side analog, confirms the overlap mechanism; MemoryRepository-for-AI-NPC = memory STRUCTURE without prefetch-overlap. No direct published prior for per-entity predicted cognition-memory prefetch on a fixed tick — but the combination is unmeasured, so per §1.5 this files as a fusion idea (Issue 880), not a Super-GOAT.
- The honest kill-switch is T0 of Issue 880: in-process `LocalKvStore` reads are ~826ns — if production ticks show no Warm-tier stalls, the issue closes as no-consumer and the real surface is the workerd DO deployment (network-class reads).

### 3.3 Model-based training (track c) → riir-train Issue 524 (Plan 337 deltas, ~0 GPU-h)

The model-based advocate read Plan 337 + the trainer + the precompute and produced 7 deltas targeting the **measured** Bench-458 defects. Headlines (full table in Issue 524):

- **Δ-A1 soft-target asym-BCE:** `dz = (1+(w−1)m)p − wm` (one line; recovers the current gradient exactly at m∈{0,1}); supervises row ranking while keeping the Bernoulli readout — and because m is in probability units, the gate threshold becomes context-invariant (the direct fix for D3's per-context threshold drift).
- **Δ-B1 max-pool vs sum-pool block labels:** SparDA's fine-grained-supervision insight, translated — our pipeline is already finer than the paper's fine arm (per-token scores are computed then discarded); the transferable axis is the POOLING OPERATOR. Sum-pooling under-ranks needle blocks as context grows; max-pool ranks the must-fetch needle block first.
- **Δ-B2 (highest leverage):** FMID v2 stores m_sum + m_max (+ optional token top-k) → every label/loss/threshold experiment becomes a post-hoc CPU-minute instead of a 3–13h M3 precompute re-run.
- **Kill list (auditable):** rest-bucket KL as the PRIMARY loss — structurally inapplicable to a threshold gate (no partition exists over independent Bernoulli readouts); its two functions ship via Δ-A1/A2 instead. SparDA-style per-layer Forecast heads on Bonsai (hybrid: too few attention layers for the lookahead chain to pay) and qwen3.8 (≈165 GPU-h at 8K — the weak-signal regime, golden base rate 40%). Kimi-K3 full replication (arch-test only per AGENTS.md).
- **Pre-conceded risk (stated before the skeptic):** every delta trains a selector to imitate DENSE attention, then runs SPARSE inference on a dense-trained backbone — Research 523 (H2O-norm) measured exactly this regime collapsing for *eviction* policies. Mitigations: FlashMemory selects coverage over a complete cache (recoverable per-step), not deletion (compounding); and D1 (real sparse-forward A/B) + the R/p128 runaway canary remain the standing preconditions for ANY serving claim.
- Cost: **0 GPU-hours** for deltas 1–7; ~9–15 M3-hours one-time label regeneration.

---

## 4. External confirmations for the kernel_opt corpus (rider material — next batch touching these rules)

| SparDA evidence | Corpus rule(s) confirmed | Note |
|---|---|---|
| Batch-adaptive CTA allocation (Table 7: device-quantized swept table, 16/32 @ H100, 16/32 @ A100-threshold-64) | `swept-shape-bucket-launch-tables` (B104) + `launch-heuristic-pinned-to-calibration-arch` (B110) + spin-barrier co-residency capping (B110) | The pattern (sweep → bucket table → arch-pinned heuristic matching best-fixed within 4%) is verbatim B104/B110; constants are hardware facts. |
| Prefetch overlap pays only above a crossover batch (Table 10: B4 = net overhead, B16 = breakeven, B64 = +40%) | `fusion-payback-gated-on-per-dispatch-overhead` (B80) + `wavefront-overlap-pair-ceiling` (B103) | Third external instance of the payback-crossover doctrine, on the transfer/compute axis instead of dispatch. |
| InfiniGen accuracy collapse (−6 to −13 avg) | Confirmation for any F1/F2 forecaster gate that must beat the deterministic proxy | The paper's own ablation is the evidence that hidden-state proxies are insufficient unaided. |
| Layer-0 KV kept GPU-resident + `F_0^cur` same-layer fallback | cold-start fallback discipline (prefill-chunk / warm-reachable family) | Pattern only: every lookahead path declares its cold-start fallback. |

---

## 5. What does NOT transfer (complete list, with reasons)

1. **The trained Forecast projection** — requires a sparse-pretrained backbone; none exists in the stack; qwen3.8-scale replication ≈165 GPU-h at 8K in the weak-signal regime (kill).
2. **The 2.5×/1.7× headline numbers** — trained-vs-trained comparisons on different selectors; our mechanism wins through a different denominator (τ=64 batched scoring already collapsed decode selection). Never cite as ours.
3. **Rest-bucket KL as a primary loss** — softmax-partition device; our threshold gate has no partition. Functions ship via soft-target BCE (Issue 524 Δ-A1/A2).
4. **UVA/Triton kernel specifics** — CUDA-only; our CubeCL/Metal/cudarc twin split needs per-backend work; propose only after Issue 730's G2 proves the stall is real on our box.
5. **CTA constants and the batch<32 threshold** — hardware facts, re-sweep or don't write them down.
6. **Prefill Forecast benefit as trained** — at prefill SparDA's gain is selection-cost only (keys already on GPU); our prefill lever is the deterministic schedule (Issue 730), which needs no training at all.

---

## 6. Verdict

**Gain.** GOAT-class levers filed per track (Issue 730 prefill offload; Issue 524 recipe deltas; Issue 880 game fusion idea). **Not Super-GOAT:** no new capability class — the lookahead/overlap mechanism is the FlashMemory corridor's own selling point extended along the layer axis, and the strongest new lever (prefill double-buffer) is deterministic engineering, not a primitive. The corridor owns this mechanism; this note arms it with the piece the corridor was missing (the prefill wall) and the recipe deltas its training plan was missing.

---

> **PASS-Redirects (synthesis):** Bai et al. [arXiv:2603.12201 "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse"] — the training-free prior art for F2's stickiness arm (reuses top-k indices across adjacent layers; eliminates up to 75% of indexer compute in DSA; 1.82× prefill / 1.48× decode). Confirms the modelless selection-reuse space is published; any fusion claim must differentiate on the decay-gated union + prefetch-composition, not on reuse itself. Follow-on "You Only Index Once" (arXiv:2606.06467) shares indices across layers even more aggressively — same disposition.
> **PASS-Redirects (synthesis):** TierKV ["Prefetch-Aware Memory Tiering for KV Cache in LLM Serving"] — serving-side confirmation that KV tiering with PCIe/NVMe prefetch fully overlapped with GPU compute is an established shape; relevant to Issue 730's design space and Issue 880's game-side analogy.
