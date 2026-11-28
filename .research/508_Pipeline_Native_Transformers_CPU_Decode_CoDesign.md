# Research 508: Pipeline-Native Transformers — CPU Decode Co-Design (cflow)

> **Source:** [arXiv:2608.23841](https://arxiv.org/abs/2608.23841) — "Pipeline-Native Transformers: Co-Designing Model Architecture and CPU Inference for Bandwidth-Efficient Autoregressive Decode" — Tom Poperszsky, 24 Aug 2026 (independent research report, 77pp)
> **Date:** 2026-08-26
> **Status:** DISTILLED — §6.3 Approach A+B **MEASURED NEGATIVE** ([Bench 683](../.benchmarks/683_stale_residual_poc.md), Issue 691 closed 2026-08-26: residual-dominance fails on Bonsai-27B AND Gemma-2-2B AND K3-0.40B — ratio ≈ k/√L, k≈1.5–3 → the 0.05 bar needs ≥~1000 layers); extractions #1–#3 landed in `katgpt-core/stale_residual`; riir-train Issue 482 (delay-arch screening) unaffected and still queued
> **Related Research:** 110 (Ciot ternary CPU — our Plasma tier), 066 (TileRT persistent pipeline), 161 (dMoE block routing), 059 (MoE spec-decode co-design), 447 (Kimi K3 latent MoE), 456 (Gigatoken cache hierarchy); riir-ai 328 (deepseek v3 noaux_tc router — the exact router family in `moe.rs`)
> **Related Issues:** katgpt-rs 691 (§6.3 Approach A/B POC — closed 2026-08-26, file removed; record = Bench 683 + git history), riir-train 482 (../../riir-train/.issues/482_delay_architecture_screening_dense_delay_expert_delay.md) (delay-arch screening)
> **Classification:** Public (katgpt-rs — transformer-stack inference mechanics)

---

## TL;DR

Single-token CPU decode is bandwidth-bound (Q4 matvec = 4 FLOP/byte vs machine balance ~20), so the paper co-designs **architecture + runtime**: rewrite the layer dependency DAG (`dense_delay` δd — FFN reads a residual from δd layers back; `expert_delay` δe — router fires at layer ℓ, expert output injects at ℓ+δe) so a vertical stage-major schedule becomes valid, cutting critical-path weight bandwidth **2.00×** (9.00→4.50 MB/token) at quality within 0.24 ppl of the best arm. For us the two load-bearing extractions are (a) the **critical-path bandwidth law** `Σ max(B_attn, B_dense/(δd+1), B_expert/(δe+1))` — a closed-form decode-latency calculator we lack, and (b) the paper's own **untested §6.3 hypothesis** (stale-residual speculative layer pipelining on *standard* checkpoints) which our shipped rollback machinery can falsify and neither the paper nor any prior art has measured.

**Distilled for katgpt-rs (modelless, inference-time):** the bandwidth law as a pure calculator (encoding-parameterized — it tells us at which bits/weight the regime flips), the `(C+IO)/max(C,IO_eff)` I/O-overlap predictor, and the ‖δℓ‖/‖x‖ residual-dominance gate for speculative layer execution.

---

## 1. Paper Core Findings

### 1.1 The regime (why CPU decode is bandwidth-bound)

- Machine balance ≈ 20 FLOP/byte (1 TFLOP/s vs ~50 GB/s). Single-token matvec at Q4 (0.5 B/param, 2 FLOP/param) → **4 FLOP/byte, 5× below balance**. Token latency ≈ bytes(all weights)/B_max.
- GPU-first runtimes (llama.cpp et al.) inherit row-major layout + all-experts MoE loads (E/k waste = 16× at Gemma-4 E=128,k=8).

### 1.2 cflow runtime (inference-side, all modelless)

- **Tile-native format**: 128×256 Q4 tiles ≈ 18 KB (L2-sized), stored in **compute-consumption order**; zero-copy mmap (`MappedTile`); `.cflow` per-layer + `.vflow` stage-major vertical-group format.
- **Fused projections**: QKV (and gate+up) tiles interleaved by output stripe — one activation load services 3 projections.
- **Conditional expert loading**: expert offset table `(offset,len)` per expert → only top-k experts' tiles read after the router fires (structural E/k reduction).
- **Staged direct-I/O expert fetch**: async read issued at the routing layer, consumed at the injection layer δe later — the delay window realized as I/O overlap. Measured net up to **1.68×** on NVMe; the model `net = (C+IO)/max(C, IO_eff)` predicts all four thread-count points within 1%.
- **Negative results (honest, instructive)**: PREFETCHT0 **refuted** — at storage-bound scales it is useless (±0.1%) to actively harmful (**48% of decode time** wasted walking expert regions at 30.9B; disabling it = 1.92× speedup). Stage-major disk layout **inconclusive** without async streaming. Mechanistic explanations given for both.

### 1.3 The architecture transforms (training-side — the paper's core novelty)

Standard pre-norm: layer ℓ+1 needs layer ℓ's *complete* output (Attn→FFN intra-layer chain) → strict sequential weight-read schedule. Two relaxations:

- **dense_delay δd**: `f_ℓ = FFN_ℓ(Norm(x_out^{ℓ-δd}))` — breaks the intra-layer dependency; layer ℓ+1 attention can start once layer ℓ attention finishes.
- **expert_delay δe**: route at ℓ off `Norm(x_in^ℓ)` (**pre-dense routing anchor**), inject `x_out^{ℓ+δe} += e_ℓ` — creates the δe-layer I/O overlap window.
- **Critical-path law**: `B_critical = Σ_ℓ max(B_attn, B_dense/(δd+1), B_expert/(δe+1))`. Delay helps **only while the amortized stream remains binding**; attention (never delayed) is the floor.

Five architectures + 1 ablation, TinyStories 10K steps, d=512/L=6 (ppl):

| Arch | δd/δe | ppl | B_crit |
|---|---|---|---|
| arch1 decoupled streams | 0/0 | 7.21 | 9.00 MB |
| **arch2_4_combined** | **1/2** | **6.50** | **4.50 MB (2.00×)** |
| arch2_4_sync (ablation) | 1/0 | 6.52 | 4.50 MB |
| arch3 pipeline registers | 0/0 | 7.24 | 9.00 MB |
| **arch4 async experts** (pre-dense routing) | 0/2 | **6.26 (best)** | 9.00 MB |
| arch5 weight-shared 2×3 | 0/0 | 6.77 | 14.06 MB (4.69 MB unique) |

Key quality findings: **expert delay is quality-free** (6.52 vs 6.50 = noise); **pre-dense routing is quality-positive** (+0.24 — the router sees a cleaner signal before the dense FFN perturbs the residual). Scaled validation: 8.34B (d=8192/L=4, ppl 4.52) + 30.9B (L=16) trained FSDP 8×A100/H100.

### 1.4 End-to-end

5.94 tok/s on 30.9B MoE (Q4, 32-vCPU Ice Lake) vs llama.cpp 4.75 (dense Qwen2.5-32B) vs vLLM CPU 1.65 — **not quality-matched** (the paper says so explicitly; the clean row is llama.cpp-vs-vLLM). 7.29× fewer L1-d misses (PMU) from tile layout.

### 1.5 §6.3 Speculative pipeline recovery — **MEASURED NEGATIVE (Bench 683, 2026-08-26)**

For **standard** (non-rewritten) transformers, exploit residual dominance (‖δℓ‖/‖x_in‖ ≪ 1):

- **Approach A**: run layer ℓ+1 speculatively on stale `x_in^ℓ` while ℓ computes; accept + post-hoc correction if ‖δℓ‖ below threshold, else rollback + recompute. Paper's own success criterion: >50% of layers with ratio < 0.05 and top-1 preserved.
- **Approach B**: closed-form-fit linear predictor router-logits→FFN-delta for corrected speculative input (R² > 0.7 target).

**First measured verdict (ours — the paper never ran it):** the premise fails on every architecture class we hold — 0 of 8/64/26 layers under 0.05 on K3-0.40B / Bonsai-27B / Gemma-2-2B respectively (medians 0.15–54). The measured law is `ratio ≈ k/√L`, k≈1.5–3 (per-layer ‖δ‖ stays O(1) while the stream grows ~√L) → passing 0.05 needs ≥~1000 layers. Approach B's router predictor reaches held-out R² 0.445 at best (< 0.7). Record: [Bench 683](../.benchmarks/683_stale_residual_poc.md), Issue 691 (closed, file removed — see git history).

---

## 2. Distillation

### 2.0 The honest regime caveat for THIS stack (read before citing the 2×)

Our CPU path is **ternary, not Q4**. At 1.58 bits/weight the arithmetic intensity is 2/(0.1975) ≈ **10.1 FLOP/byte — only 2× below machine balance**, vs the paper's 5×. The bandwidth wall binding the paper's entire thesis is **half as binding** for the Ciot/PlasmaPath path (Research 110): ternary encoding already delivers a ~2.5× byte reduction vs Q4 before any scheduling trick. Consequence: delay-architecture gains on our ternary stack should be projected from the **law** (§2.1) at our per-layer stream ratios, not lifted from the paper's 2.00× headline. The law itself survives intact — any bandwidth-bound regime benefits from critical-path amortization.

### 2.1 Modelless extractions (No-GD inventory, coverage-diffed)

| # | Primitive | Ships? / closest cousin (signal-diff) | Home |
|---|---|---|---|
| 1 | **Critical-path bandwidth calculator** `Σ max(B_attn, B_dense/(δd+1), B_exp/(δe+1))` + attention-floor limit law | **No analog** — nothing in the stack computes per-layer decode byte floors. Not in any `.research/` note. | katgpt-transformer (pure fn) |
| 2 | Arithmetic-intensity classifier over encodings (Q4/Q8/ternary/f16) | Implicit in Research 110 prose; **no code** | katgpt-core |
| 3 | `(C+IO)/max(C,IO_eff)` I/O-overlap predictor | riir-clippy B58 rules cover async HtoD *mechanics*, not this closed-form tier model | katgpt-core fn / kernel_opt rule |
| 4 | `.cflow` tile-format converter (Q4, compute-order, L2-sized) | GPU tile layouts ship (riir-gpu, kernel_opt corpus); **CPU weight-file compute-order format does not**. Honest scope: our CPU inference is ternary — value is adopt-if-we-ever-run-Q4-CPU | katgpt-rs (conditional) |
| 5 | Expert offset table + conditional expert **I/O** | Signal-diff: `moe_forward_token` (moe.rs) already computes ONLY selected experts — but from **resident** weights. The paper's offset-table win applies to **disk-resident** experts (model > RAM). Our deployments (K3-0.4B, Bonsai-27B@Q2≈7GB) are RAM/GPU-resident → **near-term value LOW**, rises if we ever serve >RAM MoE | katgpt-transformer (conditional) |
| 6 | Fused QKV/gate+up single-activation-load | Ships on GPU (fused dispatch, B15/B26 family); CPU scalar path re-loads — micro | katgpt-core (micro) |
| 7 | Staged async expert fetch under delay window | No analog (no delay window exists in stack) | rides #8's issue |
| 8 | **§6.3 Approach A: stale-residual speculative layer execution** (‖δℓ‖/‖x‖ gate + correction/rollback) | Signal-diff vs shipped cousins: `HydraSkipPlan`/`should_skip_layer` **skips** layers on cumulative-DE thresholds (different mechanism — drop vs stale-run); GDN tree-verify rolls back *recurrent state*, not layer inputs; token-level spec decode is orthogonal. **Nothing runs a layer on a stale residual.** | **Issue 691** (POC) |
| 9 | §6.3 Approach B: router-logit→FFN-delta predictor via **closed-form least squares** | No analog | folds into Issue 691 |
| 10 | PREFETCHT0-harmful-when-storage-bound (negative law) | riir-clippy kernel_opt takes measured negatives (B70 pattern) — candidate corpus rule | kernel_opt backlog |

### 2.2 What already ships (coverage — do not re-claim)

- Conditional expert **computation** (top-k only): `moe.rs` `moe_forward_token` — sigmoid top-k, noaux_tc bias, renormalize — the exact Kimi-K3 router family the paper's MoE analysis assumes. Selected-expert loop touches only k experts' weights.
- Layer-level conditional compute: `HydraSkipPlan` (bitmask skip + cumulative-DE early termination).
- Speculative decode + rollback machinery: `LeviathanVerifier`, GDN tree-verify (rollback-free S₀), `checkpoint/rollback_speculative_gpu` (riir-train cudarc), KV page ensure/free fast path (bench_414).
- GPU tile layouts + fused dispatch: riir-gpu (block-contiguous, row-tile amortize, LUT plane tiles) + ~15 kernel_opt corpus rules.
- Ternary CPU SIMD path: Research 110 Ciot bit-planes (the stronger-encoding alternative to the paper's whole Q4 apparatus).

### 2.3 Fusion (paper × stack — what neither has alone)

1. **Approach A × shipped rollback = the first measured verdict on the paper's own open hypothesis.** The paper proves the schedule math but leaves speculative recovery untested; we hold real checkpoints (Gemma-2/Bonsai/K3-class), per-layer trace tooling, and token-level rollback machinery. The offline analyzer (per-layer ‖δℓ‖/‖x‖ distributions) is zero-GPU and answers viability in hours. → **Issue 691**.
2. **Delay-arch × `moe.rs`**: the pre-dense routing anchor + expert_delay is a ~100-line diff on a forward/backward that already exists (advocate's estimate), screening ~6 GPU-h at TinyStories scale — and pre-dense routing is the arm that is quality-**positive**, so it sidesteps the usual G1-vs-G2 tension. → **riir-train Issue 482**.
3. **Bandwidth law × ternary PlasmaPath**: parameterize the calculator by bits/weight → it *predicts* at which encoding/d_ff ratio the dense stream stops binding for us (§2.0's caveat made quantitative). Cheap rider on extraction #1.
4. **Delay window × GPU H2D**: on GPU, expert weights are resident — but the delay-window *overlap* pattern maps onto layer-ahead **H2D prefetch for Cold-tier shard thaw** (neuron-db) rather than disk reads. Speculative; recorded, not filed.

### 2.4 Training-plan candidates (riir-train; model-based track)

Recorded in riir-train Issue 482 (../../riir-train/.issues/482_delay_architecture_screening_dense_delay_expert_delay.md) with GPU-h estimates: (a) K3-class MoE pre-dense + δe screening (~6 GPU-h) → optional 0.4B distill re-run (300–600 GPU-h); (b) Bonsai-27B δd re-distill (1.5–2K GPU-h — **owner call, not filed as a plan**); (c) arch5 weight-share for the 0.4B edge class (<100 GPU-h rider).

---

## 3. Verdict

**Gain.** Not Super-GOAT (no game-pillar selling point; Q4-CPU runtime apparatus is off our ternary path), well above Pass (two unshipped closed-form tools + one testable open hypothesis + a quality-positive training lever we can screen for 6 GPU-h).

**MOAT gate (katgpt-rs):** ✓ Transformer stack / decode / spec-decode is this repo's MOAT row; the calculator + Approach-A gate are generic inference mechanics (no game semantics) → katgpt-core/katgpt-transformer. Delay-architecture training → riir-train (correct home). Tile-format converter stays opt-in/conditional (honest low near-term value).

**Why not GOAT now:** extractions #1–#3 are small pure fns (a session each, behind feature flags + benches); the POC (#8/#9) is a hypothesis test, not a proven gain — promote only if the analyzer + simulator pass. The 2.00× headline does **not** transfer to our ternary regime without re-derivation (§2.0).

---

## References

- [arXiv:2608.23841](https://arxiv.org/abs/2608.23841) — full report (cflow runtime, five architectures, PMU + wall-clock evaluation, two negative results).
- Internal: Research 110 (Ciot ternary — §2.0 caveat source), Research 066 (TileRT P2 overlap analog), Research 161 (dMoE), Research 447 (K3 KDA latent MoE); `katgpt-transformer/src/moe.rs`; `katgpt-pruners/src/hydra_budget.rs`; riir-train speculative seam (`forward_speculative_verify`, `rollback_speculative_gpu`).
- Paper's own precedents: GPT-J/PaLM parallel Attn+FFN (the dense-delay generalization), FlexGen/DejaVu/LLM-in-a-Flash/PowerInfer (streaming/offload — no dependency rewrite).
