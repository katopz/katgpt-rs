# Research 489: oMLX Dual-ANE/GPU Hybrid Prompt Processing — Verdict

> **Source:** [jundot/omlx PR #2756](https://github.com/jundot/omlx/pull/2756) — onthehub97, "feat(qwen3.5): add experimental dual-ANE/GPU prompt processing", merged 2026-08-17 (commit `fbb98dc`, 19.3k-star MLX inference server). Follow-ups: #2760 (oQ4e + runtime hardening, merged), #2781 (M3 Max: 16K +13%, 32K +57% prefill, open), #2803 (DeepSeek, open).
> **Date:** 2026-08-18
> **Status:** RESOLVED — external Gain stands as prior-art record; OUR POC failed G2 **but the 7-8× gap was later DIAGNOSED as our integration, not the fabric** (same day, riir-ai Bench 775 — renumbered from 774 in the 2026-08-28 collision resolution: ANE-driver ~4-8-program hot-set → 96-program round-robin pays 45.6 ms/switch ≈ the whole branch-D residual, pair-alternation free — omlx's 2-program/112-procedure bank is the dodge; + 56% of the hybrid arm was per-op queue-drain readbacks, 14% scalar host unpack; pure ANE compute ≈ P10's 4.7 s/block). riir-ai Issue 726 stays closed as-built; **Phase C (segment-split overlap) LANDED 2026-08-28 per the delegated perf/sec decision — riir-ai Plan 549 / Bench 776: overlap thesis VALIDATED (+40% over serial @2048, G1 stretch-pass 2048+4096) but G2 MISS vs the GPU baseline (0.864×/0.941× — the GPU arm itself lifted to 61.8 tok/s by the Issue-768 32×32 GEMM); the host-pointer readback is the isolated structural 60-75% — the omlx +27-57% class requires the IOSurface zero-copy bridge (armed follow-up); toggles default-off.** Partially fires Research 427's revisit trigger #1: ANE fused-kernel dispatch decisions now matter at LLM-prefill scale — but on the **private-runtime** path, not CoreML. **Companion note:** the Weschera/drowzeys deployment-recipe evidence (fraction=0.5 sweet spot, 0.75 worse; GDN-off winning config; decode +10% second-order via shared-engine interference) is distilled in riir-ai Research 343, not here.
> **T0 update (2026-08-19, riir-ai `5b9fdb73c`):** binding spike DONE — private runtime reachable on M3 Max/macOS 26.5 via an ObjC bridge (pure-Rust msgSend rejected by ANECCompile with byte-identical artifacts — negative result recorded); **M3 Max is ONE 16-core ANE fabric, not discrete pinnable dies** (concurrent programs overlap 2.4-2.9× regardless of instance hint — the omlx dual-instance constraint is M3-Ultra-only and unnecessary here); handle budget ≥256; multi-procedure banks rejected on macOS 26.5 (single-procedure programs viable); fp16 conv numerics cosine 1.000016; int8 dual-residency budget fits 64 GB at any fraction. POC proceeds (T1+).
> **Related Research:** 155 (ANE Backend Verdict — Path B `rane` private-API precedent, Path A CoreML decision), 377 (ANE architecture: 0.23ms dispatch floor, 2MB working set, int8 formats), 223/224 (ANE distillation + coremltools), 427 (tile-graph overlap — dependency-chain analysis kin), 419 (PackInfer, loose)
> **Related Plans/Benches:** Plan 255 + Bench 053 (`ane_npc` GOAT FAIL — dispatch-floor economics), Plan 379 (`ane_roofline.rs`), Bench 438/439 (ANE fused-chain cost model: fusion saves 95.3% dispatch-bound; CoreML routes large F32 to CPU); riir-clippy **Bench 010** (the live-gap anchor — see §5)
> **Classification:** Public

---

## TL;DR

oMLX PR #2756 adds a hybrid prompt-processing path for dense Qwen 3.5/3.6/3.8: eligible fixed-size (2048-token) prefill blocks split the **output channels** of MLP gate/up projections (and GDN `z+qkv`) across **both physical ANE instances + the GPU concurrently**, with a Metal merge kernel that fuses SwiGLU. Measured on M3 Ultra / Qwen3.8-27B-AWQ: **+35.6% prompt throughput** (334.9 → 454.3 tok/s), growing with context (+14.4% @4K, +26.1% @16K; M3 Max #2781: +57% @32K), at cosine 0.9992 hidden-state fidelity vs GPU-only.

For us this is the **first production-grade proof of the regime Bench 053 predicted ANE would win**: our CoreML ANE path failed economically because per-dispatch compute (~11µs of NPC projection) was dwarfed by the ~0.28ms ANE dispatch floor. A 2048-token × hidden GEMM is ~25-60ms of compute per layer — the floor amortizes ~100×. Crucially, the winning path is **NOT CoreML** (our Path A): it is the private `AppleNeuralEngine.framework` runtime with **procedure banks** (112 procedures in 2 resident programs under a ~121-handle budget), **instance pinning** (two pinned submissions beat one unpinned: 41.51ms vs 57.90ms), and INT8 per-output-channel requant with fail-open GPU fallback. That is Research 155's Path B (`rane`-style private access), which we shelved as "experimental fallback" — now validated at production scale by an external codebase.

**Verdict: Gain (POC-gated).** Not Super-GOAT (external prior art caps Q1; perf not a new behavior class). Actionable output: riir-ai Issue 726 — dual-ANE/GPU hybrid prefill POC on **Ternary-Bonsai-27B** (`TernaryDeltanetGpuForward`, the only model in focus per Issue 629's 2026-08-15 directive; retargeted from gemma2, which we don't run — Bonsai's deltanet layers are omlx's own GDN target): default-off, fail-open, GOAT-gated. katgpt-rs gets **no** new primitive now — `ane_roofline.rs` suffices; a hybrid-split cost term is a follow-up only if the POC lands.

---

## 1. The eight load-bearing design decisions

| # | Decision | Measured/argued evidence |
|---|---|---|
| 1 | **Two instance-pinned ANE submissions per op** (not one unpinned) | Private runtime does not stripe one call across both ANEs: 55% MLP slice 57.90ms unpinned vs 41.51ms as two pinned calls |
| 2 | **Procedure banks** — 112 procedures (64 MLP + 48 GDN) packed into 2 resident programs (one per ANE) | Private runtime caps resident handles at ~121; one-program-per-layer would exhaust the budget before covering MLP+GDN |
| 3 | **Combined gate/up (and z/qkv) in one procedure** | Splitting into more calls raises launch count + sync overhead without adding useful ANE parallelism |
| 4 | **Merge kernel fuses SwiGLU** | Avoids full gate/up materialization + extra concatenations (our B15 fused-dual-GEMV+GeGLU shape) |
| 5 | **Blocking input-pack wait is intentional** | Async alternatives all measured worse; Metal-callback submission made a layer 47.5 → 71ms and the full body slower than GPU-only. Blocking lets ANE submit while the GPU suffix is still queued |
| 6 | **Eager compile at model load** (config ∈ runtime signature) | 112-procedure bank = 37-44s compile; moved out of first request |
| 7 | **Exact fixed-shape eligibility only, fail-open** — no synthetic padding | Qwen padding token is learned: padding perturbs logits, KV positions, RoPE, GDN recurrent state. Tails/decode/verify/unsupported → GPU |
| 8 | **INT8 per-output-channel ANE weights from resident q4; down-proj stays GPU** | Requant is not bit-exact (cosine 0.9992 / 0.9985, top-1 unchanged, HumanEval 28/30 both). An experimental split-down path had unacceptable errors and was **removed, not flagged** |

Residual inefficiency honestly reported: each ANE runs only ~38.8% duty cycle; downtime is dominated by waiting on upstream GPU dependency chains, not submission delay (tens of µs) — more submissions cannot fix it without restructuring the layer boundary.

## 2. Their measured results (THEIR stack — not a claim about ours)

| Measurement | GPU | Hybrid | Δ |
|---|---|---|---|
| Layer-0 MLP, 2048 tok (M3 Ultra, 27B AWQ) | 61.12 ms | 48.45 ms | 1.26× |
| Full language body (corrected MLP+GDN) | 6.115 s | 4.508 s | 1.356× |
| Prompt throughput | 334.9 tok/s | 454.3 tok/s | +35.6% |
| E2E 4K / 8K / 16K | 331.4 / 326.0 / 318.9 | 379.1 / 397.7 / 402.1 | +14.4 / +22.0 / +26.1% |
| M3 Max (#2781) 16K / 32K | — | — | +13% / +57% |
| Numerics | — | — | hidden cos 0.999200, logit cos 0.998522, top-1 unchanged |

Gains grow with prompt length (more complete fixed-shape blocks amortize eager compile + per-request overheads). Costs: >2× resident weights (INT8 ANE copies + originals + IO surfaces), 37-44s startup compile, macOS-private-API fragility, prompt-processing only (decode stays GPU).

## 3. Prior art — the Aug-2026 hybrid-ANE wave (caps novelty)

| Work | Contribution |
|---|---|
| SqueezeBits, "Disaggregated Inference on Apple Silicon: NPU prefill and GPU decode" | The concept split: CoreML/ANE better for prefill, MLX/GPU for decode — use both |
| AtomGradient `hybrid-ane-mlx-bench` (+ paper) | "CoreML and ANE **Private API** Prefill + MLX GPU Decode" — the private-API angle predates omlx |
| `thebasedcapital/ane-infer` | From-scratch engine running Qwen3.5-2B on three accelerators simultaneously |
| **omlx #2756 line** (#2760/#2781/#2803) | Dual-**instance pinning**, **procedure banking** under handle budget, GDN `z+qkv`, fused merge+SwiGLU, production server integration + UI |

Q1 (no prior art?) = **NO** — the hybrid ANE-prefill/GPU-decode concept is published multiple times. omlx's specific novelties are the pinning + banking mechanics, not the concept. This caps the verdict at GOAT-tier Gain; Super-GOAT (all-4-YES) is not available.

## 4. Why their regime pays where our Bench 053 failed

| Axis | Ours (Bench 053, CoreML) | omlx (private runtime) |
|---|---|---|
| Compute per dispatch | ~11µs (1000 NPC projections) | 2048-token × hidden GEMM, 25-60ms/layer |
| ~0.23-0.28ms dispatch floor | **Dominates** → 26× slowdown, GOAT FAIL | Amortized ~100× |
| API path | CoreML prediction (opaque placement; Bench 439: large F32 → CPU) | Private `AppleNeuralEngine.framework` procedures (explicit placement) |
| ANE instances | 1 (CoreML decides) | 2, explicitly pinned per-op |
| Shapes | fixed batch 1024 (padded) | exact 2048-token blocks, **no padding** (positional state) |
| Numerics | FP16 CoreML | INT8 per-channel requant from q4, originals resident for fallback |
| Fusion | Bench 438: fusion saves 95.3% (dispatch-bound) | gate/up + SwiGLU fused into merge kernel |

Bench 053's "When ANE Would Win" section predicted exactly this: heavier workloads amortize the floor. Bench 438's fusion result is the same mechanism at smaller scale. Research 155 Path B (`rane`: dlopen private frameworks, ~0.24ms dispatch, IOSurface zero-copy, zero deps) is the Rust precedent for the access path and was explicitly shelved as "experimental fallback … when CoreML refuses ANE placement" — that trigger has now fired externally, at production scale.

## 5. Novelty gate + routing verdict

- **Q1 no** (§3). **Q2 no** (perf, not a new behavior class). **Q3 partial** ("all-three-engines inference on Apple Silicon" is an engine-tooling selling point, not a game-NPC one). **Q4 yes** (connects ANE substrate 155/377/427/379 × riir-gpu batched ternary prefill × our deltanet substrate × Bonsai-27B, the only model in focus per Issue 629's directive).
- → **Gain, not Super-GOAT.** No guide/plan mandated by the gate.
- **riir-ai Issue 726** (filed; retargeted 2026-08-18): dual-ANE/GPU hybrid prefill POC on **Ternary-Bonsai-27B deltanet** — `in_proj_concat` (qkv|z|a|b, already fused — mirrors omlx's combined-procedure decision) + `gate_up_proj` channel split, ANE prefix + GPU batched-GEMM suffix (the Issue 637/641 kernels — **riir-clippy Bench 010's "no batched prefill at all" finding is stale**, they landed after it and `prefill_project` is wired), feature `ane_prefill` (default-off, macOS-only), fail-open, GOAT G1 numerics (ternary→int8 requant is **lossless** — weights ∈ {−1,0,1}; better than omlx's 0.9992) / G2 ≥+15% @≥4K / G3 no-regression. Live-gap anchor: Bench 010 (prefill was 8.38× behind llama.cpp when unbatched; 17.89 vs 149.95 tok/s). **Honest scope:** short clippy prompts (93-154 tok) never engage the ANE path — the lever is long-context prefill (4K-32K; omlx M3 Max +13%→+57%), complementing Issue 721 (tree-verify) which owns the decode half of the latency budget.
- **katgpt-rs: nothing now.** `ane_roofline.rs` (Plan 379) already models dispatch floor + working set; if the POC lands, extend it with a hybrid-split cost term (pick `fraction` by predicted ANE/GPU overlap, replacing a manual sweep). Research 427 tile-graph stays deferred — the PR's 38.8% duty-cycle analysis is dependency-chain reasoning, but we don't need a simulator to pick one scalar split fraction; a sweep suffices.

## 6. Defend-wrong scope (what we do NOT claim)

No quality or performance parity claim for OUR stack. Their cosine 0.9992 is their requant path on their model; any adoption claim requires our own PoC (GPU-only vs hybrid, identical prompts, riir-gpu bench) per the §3.6 rule. Architecturally the analog exists (Bench 438 measured our ANE fusion mechanism); latency and quality on the private runtime from Rust are **unproven** until Issue 726's gates run. The ≥+15% gate is a hypothesis derived from their measurements on a different model/quant/runtime — not a transferable number.

## 7. riir-clippy distill verdict — NO (recorded; conversation-only per distill skill)

- **Source language:** Swift/MLX + private-framework C++ — corpus fixtures must be original minimal Rust; no direct mining.
- **Dedup gate (both vocabularies grepped):** fused gate/up+SwiGLU = B15 `fused-dual-gemv-geglu`; fixed-shape fast path + GPU fallback = B23 `hardcoded-fast-path-jit-specialization`; eager compile-at-load = B26/B33 prebuilt-at-init family; hybrid device split with sync points = B35 `hybrid-cpu-elementwise-gpu-throughput-sync-points`; combined sibling ops in one call = B25 fan-out family.
- **New candidates, all failing the GOAT benchable test:** resident-handle-budget procedure banking (no capped runtime in our validators); instance-pinned dual submission (no validator); blocking-submission-beats-completion-callback (B36-adjacent; measurement is ANE-private-runtime-specific).
- **Revisit trigger:** if we land Rust ANE substrate (Issue 726 T0) and fix things on it → measured-GOAT distillation of OUR fixes (the B49 path).

## 8. Revisit triggers

1. **riir-ai 726 GOAT pass — RESOLVED NEGATIVE 2026-08-28**: G2 FAIL at every length (0.792-0.965×, Bench 760); the `ane_roofline` hybrid-split cost term was never needed. 726's reopen is owner-gated (its Status block): M3 prefill becomes a production surface, OR a diagnosis probe attributes the 7-8× in-pipeline ANE cost gap to a Phase C-removable cause. Any reopen seeds `fraction` from Research 343's 0.5 sweet spot + the GDN-off recipe shape.
2. **Apple ships a public multi-device/split ANE API** or CoreML exposes procedure banks → re-evaluate Path A vs B (stability win).
3. **Bench 053 economics reopen** — NPC-side workloads growing to batched-heavy ops (≥ ~25k-NPC-equivalent FLOPs per dispatch).
4. **Release-notes delta (2026-08-28):** oMLX v0.5.4rc1 → v0.6.3 final refined two §1 decisions — #2966 tail-padding-to-fixed-tile refines decision #7 (no synthetic padding → no STATE padding: intermediates only, synthetic rows stripped before any stateful consumer; +29% on a 2,047-token tail) and #2935/#2892 intra-op hidden-dim split with per-branch down-projection refines decision #8 (down-proj no longer GPU-only; +50.9% @27B M3 Ultra; CPU sharing +4–6% with its own memory cost). Plus #3133's approximate-scope rule (recurrent rows checkpoint-precision — the 32K-quality guardrail). Full delta: riir-ai Research 353 §2.5.1; execution: riir-ai Issue 769 T8/T9.

> **PASS-Redirects (synthesis):** Imprint [github.com/ashhart/Imprint](https://github.com/ashhart/Imprint) v0.1 (2026-09-26 distill, user list item 4) — hot-swappable PERSISTED prefill state for local MLX: compute repeated context once → save KV state as immutable checksummed profiles → restore after restart, process only the suffix (claimed 19 s → 0.3 s TTFT, author-reported unreproduced). PASS: prefix KV reuse already ships twice in the stack (riir-gpu `Qwen38PrefixCache` flat whole-prefix cache, the bench_762 flat control; katgpt-kv `radix_prefix` Issue 771 radix-tree index); Imprint's only delta is CROSS-RESTART disk persistence + multi-model profile hot-swap, for which no documented consumer or gap exists in the stack (serving surfaces are single-shot league benches + modelless game runtime + encoder-lane reflex); technique class incumbent = llama.cpp state save/restore. Reopen triggers: a long-lived repeated-prefix serving consumer appears (agentic harness on our engine), or a league cell defines a memoized-prefill axis.
