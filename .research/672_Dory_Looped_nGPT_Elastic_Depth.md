# Research 672: Dory1.0-4B — Looped nGPT at Production Scale (Elastic Depth + the First Real Checkpoint)

> **Source:** [Dory1.0-4B: A Looped Normalized-GPT](https://arxiv.org/abs/2611.03998) — Fishman, Sun, Acharya, Ladhak, Chmiel, Akle Serano, Banner, Ginsburg, Puvvada (NVIDIA Nemotron Labs), arXiv:2611.03998v1, 2 Nov 2026. Checkpoints open (OpenMDW-1.1): `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-{BF16,MXFP8,NVFP4,GGUF}` on HF.
> **Date:** 2026-11-15
> **Status:** Done — per-track verdicts: **Track A (modelless inference/serving) Gain → riir-infer Issue 1221** (the lane, not a primitive) · **Track B (self-adaptive runtime) record-only** · **Track C (training recipe) Gain-lite → recorded as Plan-481-family rider items** (cross-ref added to `riir-train/.plans/481`). Not Super-GOAT — the capability class and the serving mechanics are already adjudicated in Research 644; Dory's delta is scale, the curriculum recipe, and the open checkpoints.
> **Screening:** new ID this session — not present in any prior arxiv-walk record grepped (R258 walk #14 predates it by a month; the ID does not appear in any `.research/` before this note).
> **Related Research:** 644 (ELR — the methods paper, PRIMARY cousin; Dory is its scaled instantiation), 073 (LT2), 282 (LoopCoder-v2 halting), 602 (LoopCD negative), 519 (GRT negative), 589 (ATC — the latent-curriculum cousin), 592 (KL effective depth), 414 (loop stability)
> **Related Plans:** katgpt-rs 108 (T-pass default-on), 136 (tf_loop), 304 (GainCostLoopHalter GOAT), 428 (loop stability), 617 (LoopCD negative); riir-train **481** (mixed-depth looping, OPEN — cite by NUMBER, never "ELR", which collides with Effective Learning Rate in Plans 359/362), **324** (Ouro looped pretraining on C13, ACTIVE)
> **Classification:** Public

> **PASS-Redirects (synthesis):** Cho et al. [arXiv:2609.17184 "LoopSpec: Pipelined Self-Speculative Decoding for Looped Language Models"] — recurrent depth as token-level pipeline parallelism; no looped-serving surface in this stack to adopt it on; rides Issue 1221's lane if it lands.
> **PASS-Redirects (synthesis):** Gu et al. [arXiv:2610.07659 "DLoop: Looped Speculative Decoding"] — adaptive multi-stage drafting before verification, same class.
> **PASS-Redirects (synthesis):** Ha & Kim [arXiv:2609.23033 "WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models"] — training-free depth-draft (2.42× Ouro-2.6B, 3.54× Huginn-3.5B); intermediate recurrence outputs as draft predictions — the cheapest variant if the lane lands.
> **PASS-Redirects (synthesis):** Schwethelm, Rueckert, Kaissis [arXiv:2608.09444 "Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching"] — published adaptive-depth SERVING class (continuous depth batching); our adaptive-depth incumbent is the in-stack `GainCostLoopHalter` (Plan 304 GOAT) — different mechanism (per-dispatch gain/cost vs continuous batching), no novelty claim either way; fusion = halter calibrated on a REAL looped checkpoint (§5).
> **PASS-Redirects (synthesis):** [arXiv:2610.21594 "LoopAhead: Speculating Ahead for Faster Decoding in Looped LMs"] + RIPE caching (prefix-K/V for looped self-spec) — same serving class; recorded for the lane's related-work surface.

---

## TL;DR

Dory1.0-4B is the **scaled, post-trained, open-checkpoint instantiation** of the elastic-latent-recurrence recipe this stack already adjudicated (Research 644 → riir-train Plan 481). Every serving mechanic the paper uses is **validation, not gaps**: last-loop KV caching = our `CacheStrategy::Last` (structural default) + `forward_looped`'s overwrite slot; elastic depth = our `effective_loop_count` override; per-loop early exit = our `GainCostLoopHalter` (which goes FURTHER than the paper — Dory serves fixed k only, no inference-time adaptive depth). The Dory-specific deltas worth keeping: (1) the **difficulty-aware loop curriculum** — an RLVR rollout-routing recipe item that is NOT looped-specific (general prompt→compute-budget allocator), new to the corpus; (2) the **depth-axis self-speculative decoding instance** — a thoroughly published class (LoopSpec/DLoop/WaveFront/LoopAhead), adoption not invention; (3) the **nGPT re-injection recurrence** `s_t = N(F(N(s_{t-1}+x)))` — a third loop-update family beside our convex-gate blends (Plan 108) and ELR's adaptive-delta; (4) **the first REAL looped-trained checkpoint this stack can consume** (GGUF Q5_K_M) — which unblocks every contingent looped surface recorded in R644 §4, Research 374 P2's parked DoS-guard half, and Plan 588's deferred half, and gives the synthetic-only instruments (`kl_depth_probe`, halter calibration, tf_loop damping) their first real-data bench.

**Verdict shape:** Gain (validation-heavy), filed as ONE riir-infer lane issue + recipe riders on the OPEN looped-training plan. No katgpt-rs primitive — nothing new is needed in the public engine.

## 1. Paper core — the Dory deltas vs ELR (2610.12582 / R644)

| # | Component | Dory form | Numbers | New vs R644's corpus? |
|---|---|---|---|---|
| 0 | Scale (context for every row) | 6T training tokens total: ~4.5T NVFP4 at k=1 + ~1.465T BF16 loop-sampled continuation + ~300B 256K long-context extension; 256K context; 4B params | — | Scale instantiation of the R644 recipe |
| 1 | Difficulty-aware loop curriculum | RLVR rollout routing: profile each prompt (8 samples, k=1, SFT checkpoint), keep `0 < mean_reward < 1`; within-batch midpoint quantiles `q_i`; routing score `s_i = α·q_i + (1−α)·u_i`, **α=0.3**; sort, lower score fills deeper-loop slots; rollout workers at FIXED k with quotas 4:1:1:1 | +98 Elo GDPval-AA v2, +7.4 pts Terminal-Bench v2.1, +4.9 τ³-Banking at k=4 vs no curriculum | **YES** — training-time prompt→compute routing is new to the corpus (ATC R589 is latent-pass curriculum advancement; different layer, different signal) |
| 2 | Depth-axis self-spec decode | same checkpoint: draft k=1 (γ=4), verify k=4 batched, exact rejection sampling; per-loop caches for spec mode (last-loop for plain serving); KV + activation reuse between draft/verify | 1.37–1.71× decode throughput vs k=4 AR; 1.11–1.53× request duration | Class published (§4); instance details (per-loop-vs-last-loop cache interplay, activation reuse) are lane-grade |
| 3 | nGPT re-injection recurrence | `s_0=x; u_t=N(s_{t-1}+x); s_t=N(F(u_t))` — input block output re-added EVERY loop, L2 norm both boundaries, NO learned recurrence adapter; hypersphere-resident state | 72 physical layers → 32+40k effective (72→192) | Third loop-update family (vs our ResidualGate convex blends, ELR adaptive-delta); the normalization REPLACES the learned gate |
| 4 | QAD on a looped checkpoint | student = PTQ + quantizer in forward; teacher = BF16; full-vocab KL @ temp 1 at the SAME loop count both sides | MXFP8 PTQ parity; NVFP4 + Q5_K_M GGUF via QAD; ≤2.44× inference speedup | The matched-k law is the looped-specific part (quant error compounds per depth; ppl is k=1-flavored and blind to it) |
| 5 | Open checkpoints | BF16 / MXFP8 / NVFP4 / GGUF Q5_K_M, OpenMDW-1.1 | — | **THE unblocker** — R644 recorded "no looped consumer checkpoint exists to bench against" twice as a discard reason |
| 6 | Token-efficiency law | k=4 generates 1.3–2.5× fewer tokens than k=1 (agentic); k=2 = best gain/compute (1.03–1.25× cost, 42–86% of k=4 gains) | latent depth substitutes for CoT tokens | Corroborates our latent-first doctrine; selling-point ammo, no code |

Also: the GGUF is reported (community run-through; verify at T0) to **default to k=4** — the most expensive posture — silently; llama.cpp support is an UNMERGED PR (mainline has open eval bug #31780; "the GGUF format could still change before it lands") — a native Rust lane would be first-support, with format-drift risk.

## 2. What we ship — validation, not gaps (R644 Track A PASS stands)

R644's §2 table adjudicated the ELR serving story against shipped code; Dory changes none of it:

| Dory serving component | Shipped equivalent | Status |
|---|---|---|
| Last-loop KV caching (k-independent KV footprint) | `CacheStrategy::{First,Last,Mean}` (Last = structural default, Plan 136) + `forward_looped` overwrite slot (last write wins) | SHIPPED — paper's −2.12..+0.42 pts @ k=4 validates the Last default |
| Elastic depth dial | `Config::effective_loop_count(elastic_override)` clamp + static override | SHIPPED |
| Per-loop early exit | `GainCostLoopHalter` (Plan 304 GOAT, gain/cost scissors + oscillation cosθ) | SHIPPED — **strictly ahead of the paper** (Dory has NO inference-time adaptive depth) |
| One-loop parity / loop stability | Plan 481 recipe (mixing), Plan 428 stability, Bench 699 damping toolbox | recipe lane OPEN/ACTIVE |
| Truncated backprop through loops | ATC's truncated backprop row (R589: b=5 → 27% mem / 63% time); Plan 481 fixtures | recipe lane |

The one thing the stack does NOT ship: a **looped-serving surface with real weights**. Two recorded pending items sat exactly on this blocker — riir-ai Research 374 P2's looped-serving DoS-guard half and Plan 588's deferred half ("needs a looped-serving surface that does not exist yet", recorded 2026-09-12) — and R644 §4's two contingent fusions (ESC × loop dial; NPC limelight × loops) were gated on "a looped game-model checkpoint". Dory GGUF is that surface, one GGUF loader away.

## 3. Path 0 inventory (adversarial panel, 2026-11-15; R644-panel-inherited + the 4 Dory rows)

The three-track panel ran this session (No-GD advocate + Model-based advocate; R644's panel covers the shared recipe substrate — its inventory transfers verbatim, extended below with the Dory-specific rows). Coordinator spot-checked every advocate code fact cited here (`loss_grpo.rs` path; Plan 324 G-OURO-1 + ~105h full-BPTT estimate; the Plan-359 "ELR" name collision).

| Component | Analog ships? | Extraction? | Verdict |
|---|---|---|---|
| Elastic-k forward + last-loop cache + halter on REAL weights | Yes (all three, default-on/GOAT) | — | Track A: **the lane** → riir-infer Issue 1221 |
| Depth self-spec (k1 draft / k4 verify) | Class published; stack has self-spec family (Tri-Mode D2F drafter Plan 089, Bench 634 exact rejection sampling, belief_drafter MLP, DFlash) but NOT depth-axis | Feasible (No-GD row 7: exactness is provable, assert the distribution identity) | Track A rider in Issue 1221 (phase 2 of the lane) |
| Loop-alignment probe bin (dense-checkpoint loopability predictor) | Partial — riir-infer `loop_probe` (Issue 929, DiscoLoop) probes loop-alignment on a NON-looped checkpoint; No-GD advocate's variant chunks the middle block, iterates `N(F(N(s+x)))`, reads contraction λ + KL per k → predicts loopability of ANY GGUF | Feasible, report-only | Track A candidate — folded into Issue 1221 as **task T3(d)**, not a separate filing |
| Difficulty-aware loop curriculum | **No analog at training time** (ATC is curriculum-advancement; stack's difficulty-aware routers — Plan 212/587 — are inference-time) | Recipe is fully specified (the s_i formula + profiling protocol); NOT looped-specific — general RLVR rollout routing for any multi-compute rollout axis | **Track C rider → Plan 481 family** (recorded in §6 + cross-ref added to Plan 481); consumer = any GRPO run with a tunable per-rollout compute axis (`loss_grpo.rs`); the (0,1)-mean-reward band filter alone is free prompt curation |
| TBPTL depth-1 + CE-after-final-invocation | Partial (ATC truncated backprop; xHC depth-1 Bench 383) | The paper's claim is a ~k× looped-ACTIVATION-MEMORY cut (memory law, not wall-clock); unmeasured estimate ≈53 h at k=4 unchanged batch (½-of-full-BPTT floor), lower only if the memory cut converts into batch/checkpointing changes | **Track C rider** — pointer added to Plan 324; riders recorded in Plan 481 |
| Loop sampling must continue through ALL post-training | No (recorded law; Plan 481's mixing is the pre-training half) | Fixed-k fine-tune breaks depth generalization (k>1 competence is trained, not architectural) | **Track C rider** — governs any future Dory-QLoRA/SFT arm |
| QAD matched-k | Partial (laya Q8 fake-quant lane is single-depth) | Teacher-logit caching + LoRA-on-fake-quant ≈ ≤2 GPU-days/format | Track C record — mandatory before any looped GGUF ships from this stack |
| QLoRA SFT of open Dory-4B as compute-dialable teacher | No lane | 10–30 h on the 4090 (NF4 + LoRA r16–64, item-3 sampling) | **Owner-gated** (GPU training act) — recorded, not filed |
| nGPT recurrence family | No (convex-gate + adaptive-delta families ship) | Closed-form init extractable; parity property needs GD | Track C record — Plan 481 arm candidate (hyperparameter axis: recurrence transform), never a modelless claim |
| ESC × loop dial (k as intra-checkpoint escalation rung) | Contingent (R644 §4) — NOW HAS a checkpoint, still no serving consumer; rethink charter, owner-gated | — | Track B: record-only |
| Flip-rate EMA → per-NPC loop budget (Plan 587's caps over k) | Contingent (needs a looped game-model consumer) | — | Track B: record-only |

**Discards:** none — every panel finding merged above. The No-GD advocate's serving-side reuse of the routing blend (`0.3·q + 0.7·u` as a serving tier allocator) is folded into the ESC fusion row, not filed separately (no consumer).

## 4. Prior-art search (§4, executed this session)

1. **Headline** (looped LM serving / elastic depth): Dory + its own methods paper (2610.12582, distilled as R644) + Geiping 2502.05171 + Zhu 2510.25741 + Saunshi 2502.17416 — all corpus-known.
2. **Depth-axis self-spec**: **thoroughly published class** — LoopSpec 2609.17184, DLoop 2610.07659, WaveFront 2609.23033 (training-free, 2.42×/3.54×), LoopAhead 2610.21594, RIPE caching. Kills any novelty claim for the mechanism; the lane ADOPTS.
3. **Adaptive depth serving**: Schwethelm 2608.09444 (continuous depth batching) — the published class; our `GainCostLoopHalter` is the in-stack alternative (per-dispatch gain/cost). No claim either way; fusion is calibration-on-real-weights.
4. **Ecosystem state**: llama.cpp Dory support = unmerged PR; NVIDIA's model card pins a branch; open mainline eval bug #31780; community GGUF repos note format drift risk; community run-throughs report the GGUF defaults to k=4 silently (unverified — Issue 1221 T0 verifies). A native Rust lane is first-support, not catch-up.
5. **Workspace greps** (internal-first): `2611.03998`/`Dory` zero hits before this note; loop-family corpus enumerated in §Related — LT2/TF-loop/halter/LoopCD/GRT/Ouro/ATC/ELR-family all present; `nGPT` zero hits stack-wide.

## 5. Fusion (real-checkpoint-enabled — what NVIDIA did not do)

1. **Halter on a REAL looped checkpoint** (priority 1, rides Issue 1221): Plan 304's cost_floor calibration question (0.6 crowd vs 0.01 important) was answered on synthetic traces; Dory k∈{1..4} gives real per-loop step-size/angular traces. The 2608.09444 class makes adaptive-depth serving a live comparison axis.
2. **`kl_depth_probe` first real-data run** (rides Issue 1221): Issue 898's instrument was GOAT-proven on synthetic weights only; Dory gives per-loop KL/argmax-flip spectra on trained looped weights — directly tests the paper's "accuracy saturates after 4 loops" claim with our instrument.
3. **Token-efficiency law as doctrine ammo**: "extra computation is spent in latent space rather than on longer outputs" is the stack's latent-first thesis demonstrated at production scale by NVIDIA — quote it in selling-point docs; k=2-best-gain-per-compute maps onto every budget ladder in the decision-serving family.
4. **ESC × loop dial** (unchanged from R644 §4, now checkpoint-ful): still gated on a serving consumer + rethink charter.

## 6. Verdict

- **Track A (modelless inference/serving): Gain → `riir-infer/.issues/1221_dory_gguf_looped_lane.md`.** Not a katgpt-rs primitive (nothing new needed in the public engine — R644 Track A PASS stands). The issue carries pre-registered proof paths; the unmerged-PR GGUF format-drift risk and the parity-oracle gap are named in the issue.
- **Track B (self-adaptive runtime): record-only.** ESC × loop dial and per-NPC loop budgets stay contingent (§5 item 4); no files beyond this note.
- **Track C (model-based training recipe): Gain-lite → rider items on the OPEN looped-training lane.** Three one-liners added to `riir-train/.plans/481_elr_mixed_depth_recipe.md` (curriculum; TBPTL-depth-1 cost law; loop-sampling-through-post-training law). QLoRA-Dory teacher = owner-gated, recorded only. No new riir-train plan — Plans 481 + 324 own this lane; a new plan would duplicate.

PoC: none required — no quality-parity claim is granted anywhere in this note (the Track A Gain's proof paths live in Issue 1221's task list, pre-registered, unexecuted).
