# Research 595: PISA — Pyramid Sparse Attention with Log-Linear Complexity
**Status:** CLOSED 2026-09-30 — Plan 612 Phase 2 EXECUTED with a G2 iso-quality NEGATIVE (Bench 612: pyramid_lse −0.133 Recall@8 vs the single-level exact-LSE scan at 32K+; the latency-complexity claim CONFIRMED at slope 1.07 vs 1.99; the ladder's quality gain over mean-scoring +4.1 pt is real). `pyramid_topk` stays opt-in. The gate caught two Phase-1 defects pre-measurement (the ci-as-node-id walk bug + the ln_z leaf-LSE arm), both fixed with regression pins. Phase 1 of Plan 612 landed 2026-09-29 (commit `371b0ad64`: `pyramid_topk` primitive, 8 tests, forced-expansion bound refinement); Phase 2 fixtures landed 2026-09-29 (Issue 908)

> **Source:** [Block Sparse Attention with Log-Linear Complexity](https://arxiv.org/abs/2609.31093) — Bohao Tang, Zhen Qin, Yuqi Pan, Zheng Li, Pengfei Liu (SJTU + ByteDance Seed), Sep 2026.
> **Code:** none found (web search 2026-09-29 returned only the paper).
> **Date:** 2026-09-29
> **Related Research:** 071 (DashAttention — closest shipped cousin, single-level), 379 (HGA — two-level refinement, G2-proxy FAIL), 022 (Lighthouse — training-only multi-resolution pyramid; recorded the pyramid as an unshipped opportunity), 044 (PFlash — single-resolution pooled-K prefill), 225 (MSA — GOAT FAILED), 233 (Attention Matching), 378 (HOLA)
> **Related Plans:** 106 (DashAttention, default-on), 044 (PFlash, default-on), 126 (RTPurbo, opt-in), 256 (MSA GOAT-FAILED + argtopk kernels), 271 (AM compaction), 397 (HGA — negative result), **612 (this note's plan)**
> **Feature Gate:** `pyramid_topk` (opt-in)
> **Classification:** Public

---

## TL;DR

PISA kills the quadratic Top-K block-selection stage of block-sparse attention with a coarse-to-fine key pyramid: mean-pooled summaries at O(log N) levels, selection seeded from the root, LogSumExp (LSE) scoring over child summaries at each level, bounded candidate expansion (≤ gK per level). Selection drops from O(N²/C) to O(N log N) prefill and from O(N) to O(log N) decode. The modelless content is complete — the hierarchy is a deterministic pure function of the frozen KV cache and the scorer is closed-form — and the paper's own decisive validation is *training-free*: replaying selectors on full-attention checkpoint tensors, the LSE scorer reached Recall@8 = 90.95% vs 85.91% for mean scoring (K=8, C=64, 100 real FDA prompts, 24 layers).

**Distilled for katgpt-rs (modelless, inference-time):** three stacked upgrades to the shipped sparse-attention routing slot, all consuming kernels that already ship:

1. **LSE block scoring** (the paper's central delta). Jensen chain (paper Appendix A): normalized child-mean score ≤ normalized LSE-over-child-means ≤ normalized raw-key LSE. LSE provably preserves more true block attention mass than the mean-summary dot-product every shipped selector scores with. The shipped `routing.rs` HiLS entropy bias (Issue 044, Research 399) is *the order-1 Taylor rung of this exact ladder* ("first-order Taylor rectification of the LogSumExp chunk mass", bias `ln(chunk_size)` at zero-init); `katgpt-core::simd::logsumexp_parts` is the order-∞ kernel. PISA completes the ladder the stack already stands on.
2. **Root-seeded bounded-expansion pyramid.** No shipped selector avoids a full-scan level: HGA (the one shipped two-level refinement, `hga_forward.rs`) still scores ALL chunks at its coarse level and scores by mean-summary dot-product. PISA's loop scores ≤ gK candidates per level — the existing `argtopk_with_scratch` IS the per-level Select-K primitive (K=8 fits the NEON k≤16 lane).
3. **Decode pyramid cache.** Rank-1 leaf update `(c·k̄+k)/(c+1)` + ancestor-path recompute per token; amortized O(log N)/step. Extends single-level `ChunkSummaryCache`.

---

## 1. Paper Core Findings

- **Mechanism.** Level 0 = original keys (singleton blocks); level 1 = leaf blocks of C=64 (summary = mean of its keys); level ℓ≥2 pools g=2 adjacent child summaries → ⌈log₂(N/C)⌉+1 levels. Selection starts at the coarsest level with candidate set {1}; per level, score each candidate `s = log Σ_r exp(q·k̄_child/√d)` (leaf level: exact LSE over the C original keys), retain Top-K, expand retained blocks' children. GQA: heads sharing a KV head select one shared set (scores summed over the group). Forced blocks: first/previous/current leaf + ancestors always retained (NSA convention).
- **Scoring ablation (the load-bearing result for us).** Recall@8 on full-attention checkpoint tensors: PISA 90.95 > PISA-2 88.24 > BSA 85.91 > PISA-1 84.75. PISA-1 (mean-logit only) is the WEAKEST row — below even plain mean-scoring BSA. Derived: `ln C` is rank-invariant at uniform block size (a constant added to all candidates does not move Top-K), so at uniform C the **variance/exactness terms carry the entire gain**.
- **Complexity/latency.** Selection 2.86×/5.31×/9.95× faster than single-level BSA at 64K/128K/256K; **BSA wins at ≤16K** (Table 4) — the crossover T\* is a dispatch calibration constant, not a law. Two-stage prefill kernel reuses each loaded key block across a query tile (Qtile=4, profitable when GQ + C/Qtile < C); decode uses a single fused kernel. Pyramid storage ≈ 2N/C summaries per KV head.
- **Training track (for the record).** Native sparse training from scratch: 100B tokens @ 4K at 418M/1.47B/2.67B + 10B-token CPT @ 16K (K=32, RoPE base 10K→80K). Selection indices held fixed in backprop (NSA convention) → deterministic, LoRA-composable. Best sparse containment-retrieval average at all 3 scales; RULER NIAH 62.80 vs NSA 61.07 post-CPT (2.67B).
- **Honest limits.** The paper never evaluates on random keys — its diagnostic uses real prompts. On random keys even PISA's intermediate levels dilute a single needle (needle logit/C vs `ln 2` noise floor at g=2; survival needs needle_logit ≳ 44·σ); its leaf-level exact LSE + forced blocks are what carry selection there. The paper's own limitation section notes compute-constrained scale (≤2.67B, 100B tokens).

---

## 2. Distillation

### 2.1 Why the two in-house negatives do NOT kill this (and what they do kill)

The sparse-attention routing slot carries two GOAT failures: MSA (R225/Plan 256) and HGA (R379/Plan 397, G2-proxy FAIL 2/12). HGA's recorded root cause: *"group summaries of random keys dilute the single-needle signal below the dot-product detection threshold — same failure mode as MSA."* That is precisely the mean-scoring failure PISA's ablation isolates: PISA-1 (mean scoring) is the paper's worst selector, losing ~6 Recall points to full LSE. The paper both supplies the mechanism our failures lacked (exact LSE at the leaf recovers the within-block variance that mean destroys) and supplies the training-free diagnostic our harness lacked (replay on REAL checkpoint tensors, not random keys).

What the negatives DO kill: any pyramid claim validated on the random-key NIAH harness. **The Plan 612 gate is real-tensor replay or nothing** — the HGA lesson is binding.

### 2.2 Path 0 inventory (merged No-GD + Model-based panel)

| Paper component | Analog ships? | Verdict |
|---|---|---|
| Multi-level mean-pool hierarchy (g=2) | Partial — `chunk_summary.rs` single-level; PFlash single-resolution mean-K | Extract: multi-level build, ≤2N/C·d storage; derived: every level-j node = exact subtree mean (integral-image identity) |
| LSE block score | Partial — HiLS order-1 bias (Issue 044) + `logsumexp_parts` order-∞ kernel | Extract: exact LSE over child logits at every level; Taylor-order knob (mean / +½Var / exact) |
| Root-seeded bounded expansion | **No** — HGA full-scans its chunk level | Extract: coarse-to-fine loop consuming `argtopk_with_scratch` per level; candidate bound assertable per call: ≤ 1 + gK·⌈log₂(N/C)⌉ |
| Decode pyramid cache | Partial — `ChunkSummaryCache` single-level | Extract: rank-1 leaf update + ancestor path |
| GQA group-sum selection | Yes — GQA conventions ship | Consume |
| Forced-block policy | Partial — local windows/sink ship (SpKv window, PFlash sink) | Mirror first/prev/current + ancestors; degenerate-sequence canary |
| Crossover dispatch T\* | Yes — `meta_router` dispatch pattern | Calibrate locally; a constant, never a model |
| O(N log N) / O(log N) complexity | Derived | Log-log latency-slope pin (≈1 vs ≈2) |
| Native sparse pretraining / CPT | No (riir-train has `sparse_subnetwork_training/` + `sparse_lora_export/`; recipe affordable at 418M) | → riir-train Issue 586 (Path 0.5) |

**Advocate-finding discards (auditable):**
- **27B CPT @16K (Item C)** — discarded on three grounds: ≥2,800 GPU-h on the 4090 (QLoRA-class 0.5–1K tok/s × 10B tokens = months); zero league-pin value (benefit lands 16K+; the league pins pp2048/pp4096 where the paper's own Table 4 has single-level WINNING); serving-stack re-pin cost (the FA-promotion + argmax-stability + fork-re-pin discipline would all be invalidated). Re-arm: 16K+ serving requirement AND a measured trained-scorer margin AND multi-GPU availability.
- ~~**Bonsai-27B / GDN lane** — out of scope~~ **RETRACTED at verdict review:** Bonsai-27B is a GDN+FA *hybrid*, not pure linear attention — `riir-train/crates/riir-train-engine/src/bin/bonsai_clippy_l4_sft_train.rs:242` ("mirrors Bonsai's hybrid pattern", `synthetic_attn_period` = full-attention layer every N layers) and the qwen3.8 twin's 48 GDN + 16 FA interval-4 layout (`riir-infer-gpu/src/qwen38_dense_cudarc.rs:9`, same kernel family). Bonsai's full-attention layers carry K/V blocks PISA selects; only the GDN layers have no K/V structure. The lane is SCOPED to the FA layers (fixture + eval on FA layers only), not discarded — riir-train Issue 586 updated.
- **Var(z) regime dispatch as separate work** — folded into the scorer's Taylor-order knob (sweep decides), not a separate primitive.
- **riir-neuron-db follow-ups** (DenseEmbedIndex LSE-over-subcluster law; AnyRAG low-selected-mass abstention certificate; ShardIndex rank-1 centroid update) — real per the Jensen transfer, but no current consumer pain and retrieval pools are small; recorded here, not filed.

### 2.3 Game-context reframe (fusion priority #1)

`limelight` (riir-engine, default-on, Bench 960) ranks per-entity observational salience; its law L6 says the module only RANKS — per-zone budget redistribution is the (unbuilt) Wave-2 consumer's job. The PISA transfer: zone salience summary = **LSE over entity salience scores** (the Jensen chain transfers: LSE zone scoring provably preserves more salience mass than mean zone scoring — a zone with one highly salient entity ranks correctly instead of averaging into noise), then coarse-to-fine budget allocation: rank zones by pooled summaries → refine only top zones' entity salience → allocate the fixed cognition/aliveness budget. Forced set = current interact target + last observers + combat-flagged. Filed as riir-ai Issue 1017 (opt-in Wave-2 arm; unarmed bit-identical, the Bench 950 consumer-arming pattern).

### 2.4 Consumer-context reframe (fusion priority #2 — the healer)

Retrieval pools are 28–112 rules and the demonstrability gate (Bench 101) measured pool depth as a dead allocation axis — top-1 is decided within head-28. No healer surface is selection-bound today; the corpus would need ~10× growth before a hierarchical selector pays. Recorded, not filed.

### 2.5 Serving-envelope honesty

No live consumer operates at PISA's win regime: the league pins prefill at 2–4K (single-level wins there per the paper's own Table 4), the laya encoder serves seq ~100–317, NPC inference is short-context. The primitive lands opt-in and bench-gated; its consumers are future long-context qwen3.8 serving, `flashmemory_sparse`, and `meta_router` dispatch at a measured T\*.

---

## 3. Verdict

| Q | Answer |
|---|---|
| Q1 prior art? | **NO** — mechanism published (PISA itself; HiP; Double-P; HISA; LLSA; in-house R022 recorded the pyramid as an opportunity, HGA shipped a weaker two-level variant) |
| Q2 new behavior class? | NO vs the world (class shipped in literature); the codebase delta is real but that is not this gate's axis |
| Q3 selling point? | NO — perf/quality point on a slot with no long-context consumer today; game selling points are cognition, not block selection |
| Q4 force multiplier? | Partial — connects attention + perception + retrieval as a *consumed* mechanism |

**Verdict: GAIN, GOAT-tier on the primary extraction** (LSE pyramid block selection). Provable-complexity gain over the shipped single-level selectors, with a theorem-backed mass-fidelity direction (Jensen chain — mass ordering is proven; the recall ordering is empirical and must be pinned by replay) and a per-call assertable candidate bound. Not Super-GOAT. The gating discipline is inherited from the slot's history: MSA + HGA are the standing negatives; Plan 612's G2 is load-bearing and real-tensor-only.

**MOAT gate:** `katgpt-rs` owns the Transformer stack including sparse attention — primary home (note + plan). Game reframe → riir-ai Issue 1017. Training track → riir-train Issue 586 (Path 0.5: applicable, affordable at 418M scale, so filed rather than redirected; the 27B CPT leg is redirected with three-number justification above).

**Per-stack ledger:** slot = attention/KV (sparse-attention routing slot — the most contested slot in the repo: DashAttention + PFlash default-on, RTPurbo opt-in, MSA + HGA negative). `pyramid_topk` stays opt-in pending a real long-context consumer; promote/demote only on the Plan 612 G2 head-to-head at iso-quality on real tensors.

**PoC standing:** no quality-parity claim with the paper is made here. The paper's 90.95% is on their checkpoints; Plan 612's MGATE floor is re-measured on our captured tensors, never quoted.

> **PASS-Redirects (synthesis):** Lufkin et al. (Zyphra) [arXiv:2610.42112 "Subquadratic Attention with a Hierarchical Tree-Structured Indexer" — ZHA] — cites and extends this note's source paper (PISA [39], dismissed by ZHA for retrieval degradation of the UNTRAINED variant): the Bench-612 negative's two untested remainders are exactly ZHA's deltas — per-level **top-p mass retention** (vs our Top-K) and **KL-distilled reference-query summaries** (vs mean-pool+LSE); deep-tree robustness claimed at ≤256K. Record + design rows + riir-train PoC issue 666: riir-infer `.research/021_ZHA_Hierarchical_Tree_Indexer.md`.
