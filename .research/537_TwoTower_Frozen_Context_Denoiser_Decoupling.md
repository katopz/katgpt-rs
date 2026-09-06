# Research 537: TwoTower — Frozen AR Context + Trainable Denoiser Decoupling

> **Paper:** [Nemotron-Labs-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context](https://arxiv.org/abs/2606.26493) — Reda, Kamalu, Waleffe, Patwary, Shoeybi, Catanzaro (NVIDIA), June 2026 (v2). Code + weights released (Nemotron-Labs-TwoTower collection, built on Nemotron-3-Nano-30B-A3B).
> **Date:** distilled 2026-09-06
> **Status:** RECORD
> **Verdict:** GAIN, split per track (TTPO rule — one verdict per track, not per paper):
> - **Model-based track → Gain:** `riir-train/.plans/389_twotower_decoupled_denoiser_adaptation.md`. The paper's Table 2 decoupling ablation **contradicts our shipped dllm D2F trainer configuration** (single weight set + adapter toggle = the "tied" family), and the recipe (frozen context tower, block-size curriculum, confidence unmasking) is directly actionable.
> - **Modelless track → PASS-with-redirects:** the decode orchestration (block-wise denoise loop, confidence unmasking, diffusion-draft/AR-verify composition) already ships. No new modelless files. Two calibration contracts pinned in Plan 389 (§G4). Mandatory redirects below.
> **Related Research:** 055 (Nemotron TriMode — the same lab's predecessor; TwoTower's ablation refutes its tied/joint-training premise at scale), 034 (D2F), 154 (DFlare), 177 (Domino), 430 (DiffusionBlocks), 480 (FLARE), 485 (UGC certified unmasking), 490 (DFlash2)
> **Related Plans:** 066 (D2F), 089 (tri-mode / D2fDrafterVerifier), 399 (verifier relocation), riir-train Plan 389 (this paper's training plan)

---

## TL;DR

TwoTower converts a pretrained AR model into a **block-wise AR diffusion** generator by splitting the entangled roles of single-network diffusion LMs into **two full copies of the backbone**: a **frozen AR context tower** (causal over clean tokens, owns the single persistent KV + recurrent-state cache) and a **trainable diffusion denoiser tower** (bidirectional attention within the noisy block, **layer-aligned cross-attention** to the context tower's per-layer KV, recurrent states **seeded from the context tower's block-boundary states**, adaLN time conditioning). Blocks are committed autoregressively; the context tower then absorbs the committed block to update its caches.

On Nemotron-3-Nano-30B-A3B (hybrid Mamba-2/attention/MoE): **98.7% of AR baseline quality at 2.42× wall-clock throughput** (γ=0.8, S=16), trained on ~2.1T tokens — a fraction of the backbone's 25T pretraining.

**Why we care (three things, in priority order per the fusion ladder):**

1. **The Table 2 decoupling ablation is measured intel against our own trainer config.** At equal budget (~167B phase-1 tokens, rel. accuracy vs AR baseline): frozen-context decoupled (−6.2/−10.5/−11.3) **beats** continued-AR-training (−10.5/−8.3/−17.8), and **tied towers under joint AR+diffusion loss are catastrophic** (−21 to −28 both decode modes). Our `riir-gpu/src/dllm/` D2F trainers (Plan 068) run **one weight set with a LoRA adapter toggled** (teacher B=0 ↔ student) — the tied/continued family TwoTower's data says loses. This is "paper data contradicts a current config default" → riir-train Plan 389.
2. **League lane (priority #3):** a quality-tolerant unverified block-diffusion decode mode (2.42× at −1.3%) is a *different curve point* than our exact spec-decode lanes; our DFlash2 loss (Bench 746/747: DFlash2+lookup 5.386 < lookup-only 6.799) refuted a *tiny-head* drafter, not a *full-capacity denoiser tower*. Downstream of Plan 389.
3. **Modelless corroboration + two contracts:** the confidence-unmasking sampler's measured dynamics (front-loaded commitments, left-to-right bias inherited from the causal backbone) corroborate our drafter findings, and the **sampling-block-size asymmetry** (sampling S > training S collapses generation: HumanEval 76.4 → 19.85 at S=64 vs trained S=16) is a calibration contract any trained block-denoiser on our stack must pin.

---

## Paper Core

### 1. Two-tower architecture

- **Context tower:** frozen copy of the pretrained backbone, causal over prompt + committed tokens. Produces per-layer KV + recurrent (Mamba-2) states at every clean block boundary. Its LM head is optional in the diffusion path (kept for AR scoring/verification). At inference: 2× fixed weight memory, ONE prefix cache — sequence-length-dependent cache scales like AR.
- **Denoiser tower:** trainable copy. Within the block under refinement: bidirectional self-attention among noisy tokens (no new params, no FLOP change); **layer-aligned cross-attention** — denoiser layer *i* attends the concatenated KV `[context<Kb ; own block b]` at the *same index i* (multi-scale, not last-hidden-state broadcast); **context-seeded Mamba states** — denoiser Mamba layer *i* starts from context-tower layer-*i* state after block b−1 (requires Mamba chunk size == block size S).
- **adaLN-single time conditioning:** global MLP → shared scale/shift/gate + per-layer embeddings; **1.5M params**; measured gain +1.18 gen / +0.97 code / +0.73 math. On MoE: tokens are noise-aware via adaLN modulation; no routing changes.
- **Bidirectional Mamba (LR+RL averaged) REFUTED by ablation:** +0.02 gen, −0.6 code, −0.8 math, ~2× SSM FLOPs → keep recurrent layers causal. The L2R bias of the backbone is a feature, not a bug (see §5 dynamics).

### 2. Block AR diffusion

`log p(x) = Σ_b log p(x_b | x_<b)` — AR over blocks, masked diffusion within a block (linear schedule α_t = 1−t). Loss = **unweighted mean NLL over masked positions** — the ELBO's 1/t importance weight is **deliberately omitted for stability** (recipe intel; cf. our UGC schedules for the certified-unmasking side).

**Sampler (Algorithm 2):** confidence unmasking with threshold γ — each step predicts ALL masked positions in parallel, commits predictions with confidence ≥ γ, keeps the rest masked; τ (the diffusion time fed to the denoiser) is **derived from the remaining masked fraction** |{masked}|/S, not a step counter. Adaptive tokens-per-step; block always completes within T steps.

### 3. Training recipe

Denoiser only; context tower frozen. Two-stage curriculum mirroring the backbone's own (phase-1 broad → phase-2 high-quality/STEM); WSD LR 1e-4 → 1e-6, BF16, AdamW, reset at phase boundaries. Block-size curriculum: S=32 (phase 1) → S=32 (phase 2) → S=16 (final). Frozen context tower runs once no-grad per step; denoiser processes **all noisy blocks in one forward** (blocks folded into batch; block 0 from zero recurrent state, block b from context state after b−1).

### 4. The load-bearing ablation (Table 2, ~167B phase-1 tokens, rel. accuracy vs AR baseline)

| Config | Gen | Code | Math |
|---|---|---|---|
| AR baseline | 0.0 | 0.0 | 0.0 |
| Continued AR training | **−10.5** | −8.3 | −17.8 |
| **TwoTower (frozen ctx, separate denoiser)** | **−6.2** | −10.5 | −11.3 |
| Joint loss, tied towers → AR decode | −26.2 | −21.0 | −26.4 |
| Joint loss, tied towers → diffusion decode | −27.9 | −27.8 | −27.0 |

Ordering: **frozen decoupled > continued > tied** — with a wide margin. This is the intel that corrects our D2F trainer configuration (Plan 389 G3).

### 5. Block sizes + sampling dynamics

- **Training block size:** smaller = better quality, worse throughput (S=128 → 2.23×, S=16 → 2.02×, S=8 → 1.71×). S=16 default.
- **Sampling block size (asymmetry contract):** sampling S **larger** than training S collapses generation tasks (S=64 vs trained S=16: HumanEval 76.40 → **19.85**, GSM8K 89.84 → **2.20**); sampling smaller is robust (S=8 slightly better quality, lower throughput). **A trained block-denoiser must sample at S ≤ S_train.**
- **Dynamics:** commitments are front-loaded (first diffusion step commits the most tokens; most blocks finish in 1–3 steps) and left-to-right within the block (upper-left triangular pattern) — an inductive bias inherited from the causal context tower + causal recurrent layers (23 of 52 layers are Mamba).

---

## Cross-Reference: What We Already Ship

| TwoTower component | Our code | Status |
|---|---|---|
| Block-wise denoise decode loop | `katgpt-core/src/dllm_solver.rs` (`denoise_loop_rcd`, RCD Plan 258, 3SR warm-start Plan 291) + `DecodeStrategy::DiscreteDiffusion` (`dllm` feature) | ✅ Production |
| D2F block decode + context | `katgpt-forward/src/d2f.rs` + `d2f_context.rs` (block-causal draft caches separate from verify cache) | ✅ Production |
| Diffusion-drafts → AR-verifies | `katgpt-forward/src/d2f_verifier.rs` — `D2fDrafterVerifier` (Plan 089; Issue 587 FLARE Eq 8/21/22 acceptance taxonomy; streaming verify) | ✅ Production — Research 055's P0 gap CLOSED |
| Certified unmasking schedules | `katgpt-core/src/ugc_schedule.rs` (arXiv:2608.13520, Issue 664) — certified iteration counts, Bernoulli-unmask grids | ✅ Production |
| KV prefix cache reuse across blocks | `MultiLayerKVCache` snapshot/restore + GDN+KV prefix cache (Issue 742, 441× TTFT) | ✅ Production |
| Recurrent-state exposure at block boundaries | GDN `conv_state`/SSM state carry in `prefill_tokens_chunk`; spec checkpoint/rollback (`rollback_speculative_gpu`) | ✅ Production |
| Layer-selected cross-attention to a frozen target | DSpark dflash: "encoder fusion of target layers [1,16,31,46,61] → per-layer KV injection" (Issue 717 G2-full) | ✅ Production (subset of layers; drafter is a tiny head, not a tower) |
| Trained block-diffusion head | DFlash2 head (Bench 746/747) — **lost to lookup drafting** (5.386 < 6.799 doc-repro); Issue 742 closed | ✅ Measured negative (tiny-head class) |
| D2F **training** with two roles | `riir-gpu/src/dllm/` (Plan 068) — **ONE weight set + LoRA adapter toggle** (teacher B=0 ↔ student) | ⚠️ The tied/continued family Table 2 says loses → Plan 389 |
| Frozen-context + separate full-capacity denoiser tower | — | ❌ Not shipped (Plan 389) |
| adaLN time conditioning | — | ❌ Not shipped (training-side; Plan 389 recipe item) |

---

## Per-Track Verdicts

### Track A — modelless inference: PASS-with-redirects

The decode orchestration ships end-to-end (table above): block-wise denoise loop, confidence-based adaptive commit, diffusion-draft/AR-verify with exact acceptance policies, certified unmasking schedules. The paper's *quality mechanism* (98.7% retention) is a **trained-denoiser property** — no modelless path can claim it (Path 0 below). The two modelless-actionable residues are calibration contracts, pinned in Plan 389 rather than new code:

- **G4 contract (sampling-block-size asymmetry):** any trained block-denoiser on our stack must sample at S ≤ S_train; DFlash2's K=16 knee (Issue 742) is the same law on the drafter side.
- **Dynamics corroboration:** front-loaded + L2R commitment matches our measured drafter acceptance patterns (Bench 693–696, Issue 721/742) — no action, recorded.

**Reverse-grep discharged:** documented gaps checked — Research 055's "D2F Drafter Verifier MISSING" has since **landed** (`d2f_verifier.rs`, Plan 089/399); `.docs/09_feature_catalog/negative_results.md` #32 (DFlare modelless trio, 3× GOAT-FAILED) is NOT un-failed by this paper — DFlare's three mechanisms and TwoTower's quality lever are both training-based. Game-context reframe: no new per-NPC behavior class (the "commit-confident / refine-rest" shape already manifests as our deliberation cadence + speculation cadence); the game benefit is indirect — cheaper quality-tolerant serving for LLM seams. Healer-consumer reframe: no actionable surface — the fixer lane's local-model arm is UNARMED (measured negative, ~191 s/fix), and the two-tower *pattern* (frozen representation provider + trained task head) is already the healer's shape (frozen corpus + corpus-trained heads).

> **PASS-Redirects (synthesis):** Reda et al. [arXiv:2606.26493 "Nemotron-Labs-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context"] — block-wise denoise-decode orchestration and the diffusion-draft/AR-verify composition already ship (`d2f`, `dllm_solver`, `D2fDrafterVerifier`); the quality mechanism is training-based → riir-train Plan 389; sampling-block-size + commitment-dynamics contracts pinned there (§G4).

### Track B — model-based training: Gain → riir-train Plan 389

Filed: `riir-train/.plans/389_twotower_decoupled_denoiser_adaptation.md`. Path 0.5 discharge (full table in the plan):

| Component | Modelless analog? | Verdict |
|---|---|---|
| Masked-diffusion training of the denoiser (the quality mechanism) | none — learned masked-token prediction at 98.7% AR quality is a trained capability | **Requires GD** |
| adaLN time conditioning | none (1.5M trained params; measured +~1 pt) | Requires GD (recipe item) |
| Layer-aligned cross-attn + context-seeded recurrent states | mechanics ship (GDN state carry, KV injection) but their *usefulness* is trained | Requires GD (architecture item) |
| Confidence unmasking sampler | ✅ ships (`dllm_solver`, UGC, FLARE policies) | Covered |
| Frozen-context + single reusable cache | ✅ ships (KV/GDN prefix caches) | Covered |
| Sampling-block-size contract | ✅ modelless calibration law | Pinned (Plan 389 G4) |

Paths 1–3 (freeze/thaw correction, deterministic LoRA, latent-space correction) all fail for the core component: none can *create* denoising capability — they can only correct systematic bias in an existing predictor. Affordability: full 2.1T-token replication is out of scope; the plan validates (a) the **decoupling ordering** at micro scale (3-arm ablation, cheap, preregistered) and (b) an affordable **LoRA-budget denoiser twin** (frozen context tower + separately-LoRA'd copy = weight-distinct towers, the affordable TwoTower) at Kimi-K3-0.4B scale, with a Bonsai-27B ternary Phase 2 behind owner gating.

---

## Fusion

1. **TwoTower-as-drafter + frozen-context-tower-as-verifier** (full-capacity exactness upgrade over our D2fDrafterVerifier): **PUBLISHED PRIOR ART** — DEER "Draft with Diffusion, Verify with Autoregressive Models" (arXiv:2512.15176), DiffuSpec (ACL 2026 Findings), Speculative Diffusion Decoding (arXiv:2408.05636), Trajectory-Level Speculative Decoding for dLLMs (ICML 2026). Recorded, **not novel as a class**. Our unmeasured angle, if Plan 389 produces a denoiser twin: a *full-capacity* denoiser drafter with GDN-state seeding (vs the tiny heads those papers and our DFlash2 use) — Issue 742's negative does not cover it, but the 2× weight memory and the league's exactness-lane economics do not favor it either. Downstream of Plan 389; no separate issue.
2. **League lane (priority #3):** a quality-tolerant unverified decode cell (2.42× @ −1.3% on the paper's point) — a NEW curve point for the perf league matrix, not comparable to the bit-exact greedy anchors. Downstream of Plan 389's Phase 2; recommendation recorded in the plan.
3. **Correction to Research 055:** TwoTower's Table 2 refutes the tied/joint-training configuration at 30B scale. 055's D5 (LoRA drafter alignment on shared weights) and the dllm trainer's single-weight-set design sit in the refuted family. Honest caveat: 055's self-speculation keeps the AR mode exact and its LoRA is small — the refutation is strongest against *joint AR+diffusion training*, directional against adapter-on-shared-weights. Plan 389's G3 measures the ordering at our scale rather than assuming it transfers.

---

## References

- TwoTower: arXiv:2606.26493 (code + weights: Nemotron-Labs-TwoTower collection)
- Predecessor: Nemotron-Labs-Diffusion (Research 055) — tri-mode joint training; TwoTower is the decoupled successor from the same lab
- Block diffusion: arXiv:2503.09573 (BD3-LM); encoder-decoder diffusion (tied weights): arXiv:2510.22852
- DFlash: arXiv:2602.06036; DFlare: arXiv:2606.02091 (Research 154); DFlash2: Research 490 / Bench 746/747
- Diffusion-draft/AR-verify prior art: DEER arXiv:2512.15176; DiffuSpec (ACL 2026); arXiv:2408.05636
- Our substrate: Plan 066 (D2F), Plan 089 (tri-mode), Issue 587 (FLARE acceptance), Issue 664 (UGC), Issue 742 (league DFlash/lookup verdict), riir-gpu `dllm/` (Plan 068 trainers)
