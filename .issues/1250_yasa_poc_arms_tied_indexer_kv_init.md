# Issue 1250: YASA distill PoC arms — tied-indexer scorer + KV-sharing init loss-delta

**Status:** Open — unclaimed (claimable; claim-lease per katgpt-rs Issue 943: append `**Claimed:** <session>, <epoch>, expires <ISO-UTC>` on its OWN line before implementing)
**Date:** 2026-11-28
**Source:** [Research 698](../.research/698_YASA_Tied_Sparse_Attention_KV_Sharing.md) — YASA (Kolibri 1.1, Aleph Alpha blog 2026-11-25)
**Lane:** modelless inference (katgpt-attn / katgpt-spectral); league-serving consumer is riir-infer's (cite-only — its sparse-prefill lane is closed+promoted; challenger rides slot discipline)

Two bounded PoC arms distilled from the YASA blog. Neither claims a quality win up front — both are falsifier-first.

## Arm A — tied-indexer scorer: recall vs single-level exact-LSE

The zero-parameter indexer from YASA: score candidate key j with the main attention's OWN group-0 (or any one) query-head group, `s_j = mean_h ReLU(q_h·k_j)`, coarse blocks scored by mean-pooled compressed keys, fine top-k within the pool. No trained indexer weights (contrast `flashmemory_sparse::DualEncoderIndexer`).

- [ ] **T1.** Implement the scorer behind a feature flag in `katgpt-attn` (name: `tied_indexer`), consuming existing attention tensors — no new weights, zero-alloc selection path per the house discipline.
- [ ] **T2.** Recall harness on the Bench-612 full-set tensors (`PYRAMID_612_FULL_DIR` corpus, SDXC), run under the **Bench-612 block contract** (C=64 blocks, top-8, same harness + same `single_lse` reference) so the numbers are comparable to the recorded bars. A YASA-contract cell (16-token blocks, 16,384 pool → 2,048 fine) is a SEPARATE column needing its own reference — never mixed into the −0.133 comparison. Two recall columns, never pooled: (a) same-group recall, (b) **cross-group transfer** — recall on groups 1–3 when scored by GROUP-0's selection (that is YASA's actual mechanism: one group's selection shared by all groups; the 612 harness never measured it).
- [ ] **T3.** Indexer-cost accounting: measure the scorer's FLOPs/cache-read fraction vs main attention on league shapes (YASA claims 1/8 — one head-group of four × no value sum; verify on OUR geometry, never inherit).
- [ ] **T4.** **Pre-registered bars (two-sided, CI mandatory):**
  - **PASS** = paired diff vs `single_lse` ≥ **−0.01** at no more scoring cost than `single_lse` (Bench 612's original iso-quality bar — NOT "beats −0.133": that bar only means "beat a failed arm").
  - **Plan-filing** additionally requires the 95% lower bound of (tied − `single_mean`) > 0, **per length bin, never pooled** (the mean-key coarse arm is essentially BSA: `single_mean` 0.6475 vs `single_lse` 0.7793 — beating −0.133 alone would re-measure BSA). At L ≥ 32K, n = 64 and sd ≈ 0.27 → SE ≈ 0.034: "strictly better" without a CI is a coin flip.
  - Everything else **closes the arm negative** (record in Research 698 + Bench file, demote, delete the issue). A PASS+plan-filing result files a plan with a real-model gate before any league-facing claim (the sparse-prefill lane consumer is riir-infer's; slot discipline applies).

## Arm B — cross-layer KV-sharing init: closed-form loss-delta measurement

YASA's appendix recipe as a load-time weight transform: keys via `katgpt_spectral::procrustes` (alternate per-member rotations + average, logit-invariant), values via Eckart–Young rank-n over stacked `O_m·V_m` products, `C_m` absorbed into output matrices, rank-one reference correction. YASA measured (post-init, pre-CPT): naive copy +2.31 bits/byte → aligned 1.15 → +correction 0.62.

- [ ] **T5.** Harness: pick **TWO CONSECUTIVE FULL-ATTENTION layers** of a league checkpoint (qwen3.8-27B is a hybrid — FA layers are not adjacent; say which pair), restructure the consumer to read the producer's shared K/V through the closed-form init, and measure perplexity/loss delta on a fixed eval slice: naive-copy init vs aligned init vs unshared baseline. **Measured arm = the f16/f32 DEQUANTIZED weights** (the checkpoint is Q4_K_M — applying the transform then re-quantizing adds a noise confound; the quantized arm is reported separately, never the headline). The key rotation R must be **RoPE-commuting**: per-frequency 2×2 block-diagonal rotations (QK-RMSNorm gain uniform within each pair, or folded first), or applied to the non-RoPE part of a partial-rotary head — a free orthogonal R does not commute with QK-RMSNorm+RoPE and the measured delta would be that corruption, not the recipe. Reproduce the bits-per-byte delta METHODOLOGY (YASA appendix), never a hand-wavy "quality looks fine".
- [ ] **T6.** Substrate check first: confirm `orthogonal_procrustes` + the SVD tail machinery cover the two halves — **including the constrained (RoPE-commuting) variant, which `procrustes.rs` does NOT ship today**; anything missing is a katgpt-spectral issue (constrained Procrustes for RoPE-commuting rotations), never a local copy.
- [ ] **T7.** **Pre-registered bar (RELATIVE):** aligned init ≤ **0.5× naive-copy** loss delta, with a confidence interval over eval slices — YASA's 2.31 → 0.62 bits/byte (~3.7×) is THEIR architecture's reference point, not our gate. Arm B is DIAGNOSTIC — its consumer is a continued-pre-training lane that does not exist in this workspace (CPT at YASA's 805B-token class is budget-infeasible; see Research 698 §3 track c). A positive result (≤ 0.5×) files a riir-train-facing note row; whether to open a CPT window is decided by a recorded Claude verdict (budget vs a single-digit-B-token CPT probe) — it does NOT open a plan on its own. A null/negative result closes the arm in this issue's closure record.

## Non-goals

- No reindexer/pool-reuse arm — already dispositioned in Research 539 (IndexCache/LatentIndex prior art).
- No NVFP4-cache or QAT arm — training posture, redirected (Research 698 §3 track b/c).
- No league claim without T4's bar passing AND a real-model gate.

**Ungoat discipline:** both arms carry their proof paths above; either arm failing its bar is DEMOTED in the same commit that records the result (grade struck in Research 698, issue closed).
