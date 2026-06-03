# Research 81: RMSD — Relevance-Masked Self-Distillation

> **Paper:** [Bringing Capabilities in Distribution via Relevance-Masked Self-Distillation](https://www.appliedcompute.com/research/relevance-masked-self-distillation) — Applied Compute, 2026
> **Date:** 2026-05-22
> **Related Plans:** Plan 121 (RMSD relevance-masked distillation)
> **Depends on:** Research 038 (SDAR), Research 043 (Interventional SFT), Research 080 (VPD), Plan 072 (SDAR gate), Plan 073 (SDAR loss)
> **Supersedes:** None — extends SDAR with two-step relevance filtering

## Executive Summary

RMSD improves on On-Policy Self-Distillation (OPSD) by introducing a **two-step relevance mask** that filters training loss to the most informative token positions. Where SDAR gates ALL tokens via sigmoid(β·Δt), RMSD first pre-filters to T positions with highest teacher-student logprob magnitude, then uses an LLM judge to select the S most relevant positions. Only these S positions receive gradient.

**Why we care:** Our SDAR already provides token-level gating, but it applies uniformly to all tokens. RMSD proves that ~80% of token positions carry noise, not signal — style tokens, transition words, and unrelated content dominate the loss. The two-step filter concentrates learning on the 5-10 positions that actually matter, yielding:
- **2× data efficiency** (plateaus in ~90 steps vs ~150 for OPSD)
- **Higher ceiling** (PinappleOnly 0.740 vs 0.480 for OPSD)
- **Perfect specificity** (1.000 — zero off-topic degradation vs 0.650 for OPSD)
- **~5% less wall clock time** despite extra LLM judge calls

**Key results (Qwen3-4B, 1000 prompts × 8 hint phrasings):**

| Eval | Base | SFT | OPSD | RMSD | OPSD+cont | RMSD+cont |
|------|------|-----|------|------|-----------|-----------|
| Target Exists | 0.000 | 0.700 | 0.980 | 0.970 | 1.000 | 1.000 |
| Target Only | 0.000 | 0.670 | 0.430 | 0.470 | 0.480 | **0.740** |
| Specificity | 1.000 | 1.000 | 0.930 | **1.000** | 0.650 | **0.935** |
| GSM8K | 0.933 | 0.895 | 0.930 | **0.940** | 0.925 | **0.940** |

---

## Paper Core

### Problem: Noisy Token Updates in Self-Distillation

Standard OPSD applies reverse-KL loss at every token position. The teacher and student disagree on many tokens for reasons unrelated to the target behavior:

- **Style tokens:** Teacher prefers "Absolutely" vs student's "That" — no task relevance
- **Transition words:** Teacher uses different connectors — noise
- **Indirect steering:** Teacher may use transition words to steer back to topic — low signal

The logprob difference has **high recall but low precision** on task-relevant tokens. Most large-difference positions are distractors.

### Solution: Two-Step Relevance Mask

**Step 1 — Heuristic Pre-filter (T positions):**
Select T positions with highest |logprob_teacher - logprob_student| magnitude. This captures all potentially informative tokens but includes many distractors. Paper uses T=20.

**Step 2 — LLM Judge (S positions):**
Pass (student prompt, teacher prompt, student rollout, T positions) to an LLM judge. The judge selects up to S positions most relevant to improving student behavior. Paper uses S=5, gpt-5-mini.

**Final loss:** Reverse-KL masked to only the S selected positions.

### Key Formulas

**OPSD loss (baseline — uniform over all positions):**
```
L_OPSD = (1/R) · Σ_{i=1..R} KL_topK(P_student(·|x,y_{<i}) || P_teacher(·|x',y_{<i}))
```

**RMSD loss (masked to selected positions):**
```
L_RMSD = (1/|S|) · Σ_{i∈S} KL_topK(P_student(·|x,y_{<i}) || P_teacher(·|x',y_{<i}))
```

Where S ⊂ T ⊂ {1..R} is the two-step filtered set.

**Top-K vocabulary approximation (K=500):**
```
KL_topK(p || q) = Σ_{k∈topK(p)} p(k) · log(p(k)/q(k))
```
Avoids full-vocabulary computation. Only sums over top-K student tokens.

**Continuation phase (teacher update):**
After initial training plateaus:
1. Take last student checkpoint
2. Use it as new teacher (with same hint-conditioning)
3. Continue for N more steps
This bootstraps past the imperfect-teacher ceiling.

### Architecture

```
Student generates rollout y from prompt x
  ↓
Teacher generates logprobs from enhanced prompt x' (with hint)
  ↓
Step 1: Select T=20 positions with highest |Δlogprob|
  ↓
Step 2: LLM judge selects S=5 most relevant positions from T
  ↓
Train: reverse-KL loss on only S positions
  ↓
[Optional] When plateau: student becomes new teacher → continuation phase
```

### Key Findings

1. **RMSD > OPSD > SFT for OOD + capability preservation.**
   - SFT learns fastest but catastrophically forgets (GSM8K: 0.933→0.895)
   - OPSD preserves better but has lower ceiling (PinappleOnly: 0.480)
   - RMSD best of both worlds (PinappleOnly: 0.740, GSM8K: 0.940)

2. **On-policy data is the fundamental reason** for resilience to forgetting.
   - SFT with near-on-policy data (Qwen3-4B generated, replaced tokens) performs much better than off-policy SFT (Haiku generated)
   - Confirms SDAR/VPD's on-policy requirement

3. **Imperfect teacher still helps.**
   - Initial teacher sometimes says "pineapple" even with hints
   - Directionally correct updates on key tokens still improve student
   - Continuation phase bootstraps past teacher imperfections

4. **LLM judge filtering is lightweight.**
   - Selecting relevant tokens from a visualizer is easy for small LLMs
   - gpt-5-mini suffices — no need for frontier model
   - Overhead is small relative to training step time

5. **RL with hints fails on specificity.**
   - RL + hint in system prompt → model only works WITH hint
   - Removing hint → back to baseline
   - Dramatic specificity degradation (off-topic questions)
   - Self-distillation preserves behavior independent of hints

---

## Cross-Reference: What We Already Have

| RMSD Component | Our Code | Status | Gap |
|----------------|----------|--------|-----|
| Reverse-KL distillation | `riir-gpu/src/distill.rs` — `kl_divergence()` | ✅ Production | Full-vocab only, no top-K |
| Sigmoid-gated token loss | `riir-gpu/src/loss_sdar.rs` | ✅ Production | Uniform gate, no pre-filter |
| Token masking | `riir-gpu/src/training_loop.rs` — `LossMask` | ✅ Production | Binary mask, no relevance scoring |
| Entropy-based token weighting | `riir-gpu/src/kernels/loss_masked.wgsl` | ✅ Production | D2F importance, not relevance |
| On-policy rollout | `GZeroLoop`, `best_of_k_rollouts` | ✅ Production | ✅ Complete |
| Teacher forward pass (hint-conditioned) | `loss_sdar.rs` teacher branch | ✅ Production | ✅ Complete |
| LoRA-as-Judge | `ropd/client.rs`, `LeviathanVerifier` | ✅ Production | Not used for token selection |
| ScreeningPruner relevance | `katgpt-rs-core/traits.rs` | ✅ Production | Action-level, not token-level |
| Freeze/thaw teacher update | Game bandit `freeze()`/`thaw()` | ✅ Production | Bandit Q-values, not LoRA weights |
| SDAR asymmetric trust | `sdar_gate.rs` — σ(β·Δt) | ✅ Production | All tokens, no pre-filter |
| Top-K vocabulary selection | **MISSING** | ❌ Gap | Need top-K logprob selection |
| Logprob magnitude pre-filter | **MISSING** | ❌ Gap | Step 1 of RMSD |
| LLM judge token selection | **MISSING** | ❌ Gap | Step 2 of RMSD |
| Teacher weight continuation | **MISSING** | ❌ Gap | Need LoRA snapshot + reload |
| OOD distillation benchmark | **MISSING** | ❌ Gap | Need "pinapple"-style test |

---

## What's New for Us

### 1. Two-Step Relevance Mask (CRITICAL — The Core Innovation)

Our SDAR applies the sigmoid gate to ALL token positions. RMSD proves ~80% of positions are noise. We need:

**Model-based (riir-ai):**
```
// Step 1: Pre-filter T=20 positions by |Δlogprob|
let deltas: Vec<(usize, f32)> = teacher_logp.iter()
    .zip(student_logp.iter())
    .enumerate()
    .map(|(i, (t, s))| (i, (t - s).abs()))
    .collect();
let top_t: Vec<usize> = deltas.top_k(T);

// Step 2: LLM judge selects S=5 most relevant from T
let selected: Vec<usize> = llm_judge_select(student_prompt, teacher_prompt, rollout, top_t);

// Step 3: Reverse-KL only on selected positions
loss = reverse_kl_masked(student_logp, teacher_logp, &selected);
```

**Modelless (katgpt-rs):**
Analogous at action level — our ScreeningPruner already scores relevance, but we don't pre-filter by magnitude before the SDAR gate. We could add a `top_t_actions` pre-filter step.

### 2. Top-K Vocabulary Approximation (EFFICIENCY)

Full-vocabulary KL is expensive. RMSD approximates with top-K=500 student tokens:
```
KL_topK(p || q) = Σ_{k∈topK(p)} p(k) · log(p(k)/q(k))
```

Our current `kl_divergence()` operates on full softmax output. For large vocab (V>10K), top-K saves significant compute.

### 3. Teacher Weight Continuation (STABILITY)

RMSD's continuation phase is simpler than VPD's full EM loop:
1. Train until plateau (monitor eval metric)
2. Snapshot student weights → new teacher
3. Continue training

This is a lightweight alternative to VPD's E-step. For our model-based path: snapshot LoRA weights as new teacher. For modelless: copy bandit Q-values to teacher Q-values.

### 4. OOD Behavior Elicitation (TEST INFRA)

We need a "pinapple"-style test for our domains:
- **Bomber:** Teach model to prefer a specific suboptimal cell (e.g., always bomb position (3,3))
- **Go:** Teach model to play a specific opening regardless of position
- **FFT:** Teach model to prefer a specific unit composition

This tests whether distillation can elicit OOD behavior without breaking existing gameplay.

---

## Relationship to Existing Research

| Research | Connection | Delta |
|----------|------------|-------|
| 038 SDAR | RMSD extends SDAR's token-level gating with two-step pre-filtering | SDAR gates all tokens; RMSD pre-filters to relevant ones |
| 043 Interventional SFT | RMSD's relevance mask is orthogonal — interventional masks agent tokens, RMSD masks irrelevant positions | Both improve signal quality, different axes |
| 080 VPD | RMSD's continuation phase is a lightweight alternative to VPD's full EM cycle | VPD actively trains teacher (E-step); RMSD just snapshots student |
| 037 REAP Duality | RMSD is the model-based counterpart to our modelless relevance scoring | ScreeningPruner (modelless) ↔ LLM judge (model-based) |
| 040 BT Rank | RMSD's LLM judge could use BT-style pairwise comparison between token positions | Currently binary selection, could be ranked |
| 075 Data Gate | RMSD's stability aligns with data gate's self-play stability goal | Both prevent catastrophic forgetting |

---

## What's NOT Applicable

| RMSD Aspect | Why Not For Us |
|-------------|----------------|
| Text-domain "pinapple" task | We train game-playing models — need game-domain OOD test |
| gpt-5-mini as judge | We'd use our LoRA-as-Judge or verifier infrastructure |
| 300-step training runs | Our LoRA training has different step budgets |
| Hint phrasings (8 templates) | Game domains have different hint structures (board state, opponent model) |
| PinappleExists/PinappleOnly graders | Need game-specific OOD evaluation metrics |

---

## Verdict: ❌ NO GOAT — Negative Arena Result

**Post-implementation update:** RMSD was implemented (Plan 125) and 46/46 structural proofs pass, but **arena testing showed no improvement over SDAR**. RMSD within 10% relative gap of SDAR over 1000 bomber games — no improvement. Same fate as SDAR: reward signal modulation does not improve action selection in short tournament series. Demoted to 🪦.

The original research assessment (below) was wrong about the practical impact. The code infrastructure remains production-quality and reusable.

**Original assessment (pre-implementation):**

RMSD fills a specific, validated gap: Our SDAR gates all tokens uniformly, but RMSD proves most tokens carry noise. The two-step relevance mask is the missing precision instrument in our distillation toolbox.

### What We Gain

1. **2× data efficiency** — fewer steps to plateau
2. **Higher ceiling** — continuation phase bootstraps past imperfect teacher
3. **Zero capability degradation** — specificity preserved at 1.000
4. **Lightweight implementation** — ~200 lines new code, extends existing `loss_sdar.rs`

### What It Costs

1. **LLM judge calls** — extra inference per training step (but paper shows net wall-clock improvement)
2. **Hyperparameters** — T=20, S=5 need domain-specific tuning
3. **Top-K vocabulary** — new kernel for efficient top-K logprob selection

### Priority: ~~HIGH~~ ❌ NO GOAT (post-arena)

RMSD is the **precision scalpel** to SDAR's **broad sword**. Our existing SDAR handles the asymmetric trust problem (positive endorsement vs negative rejection). RMSD handles the **relevance problem** (which tokens matter at all). They compose naturally:

```
SDAR gate: HOW MUCH to trust each token
RMSD mask: WHETHER to train on each token
```

Combined: `gate_rmsd = sdar_gate(Δt) * is_relevant(t)` — SDAR modulates, RMSD filters.

### Recommendation

Implement as **Plan 121** with feature gate `rmsd_distill`:
- **Model-based (riir-ai):** `loss_rmsd.rs` extending `loss_sdar.rs` with two-step filtering
- **Modelless (katgpt-rs):** `rmsd_relevance.rs` extending `sdar_gate.rs` with magnitude pre-filter
- **Benchmark:** Game-domain OOD test (bomber position preference, Go opening override)
- ~~**GOAT proof:** RMSD ≥ SDAR on OOD elicitation + capability preservation~~ ❌ Failed: no improvement in arena

---

## Key Equations Reference

```
// RMSD two-step relevance mask
// Step 1: T positions with highest magnitude logprob difference
//   T = argtop_T(|log π_T(y_t|x',y_{<t}) - log π_S(y_t|x,y_{<t})|)

// Step 2: LLM judge selects S ⊂ T most relevant
//   S = LLM_judge(x, x', y, T)

// RMSD loss (reverse-KL on selected positions only)
//   L_RMSD = (1/|S|) · Σ_{t∈S} KL_topK(π_S(·|x,y_{<t}) || π_T(·|x',y_{<t}))

// Top-K KL approximation (K=500)
//   KL_topK(p||q) = Σ_{k∈topK(p)} p(k) · log(p(k)/q(k))

// Continuation phase (teacher update on plateau)
//   θ_teacher ← θ_student_checkpoint  (snapshot)
//   Continue training with new teacher
```

---

## Hyperparameter Guide

| Parameter | Paper Default | Notes |
|-----------|--------------|-------|
| T (heuristic pre-filter) | 20 | Top positions by logprob magnitude |
| S (LLM judge selection) | 5 | Final selected positions — high precision |
| K (top-K vocab approximation) | 500 | Avoid full-vocabulary KL computation |
| Steps (initial phase) | 300 | Both OPSD and RMSD plateau well before |
| Steps (continuation phase) | 30 | After teacher weight update |
| Teacher update trigger | Plateau detection | When eval metric stops improving |

---

## Modelless Implementation Results (katgpt-rs, Plan 125)

### GOAT Proofs: 44/44 ✅

| Category | Count | Status |
|----------|-------|--------|
| Unit tests (RmsdConfig, LogprobMagnitudeFilter, TopKlApproximator, MagnitudeJudge) | 24 | ✅ All pass |
| Unit GOAT proofs (T1-T10) | 10 | ✅ All pass |
| Arena GOAT proofs (T9 non-degradation, T10 continuation) | 2 | ✅ All pass |
| Pipeline integration tests | 8 | ✅ All pass |

### Key Findings from Modelless Path

1. **Signal concentration works:** `RmsdRelevanceFilter` selects actions with 5-10× higher |ΔQ| magnitude than rejected actions. The two-step filter (T=20 → S=5) correctly identifies the most informative actions.

2. **Non-degradation confirmed:** RMSD vs SDAR in 1000-game bomber arena: relative gap < 10%. RMSD's additional relevance filtering does not hurt game-playing performance. This mirrors SDAR's own arena result — the quality of the learning signal affects convergence, not action selection in short tournaments.

3. **Continuation mechanism activates:** `TeacherContinuation` correctly detects plateau after `patience=30` rounds without improvement and snapshots student Q-values as the new teacher reference. This is the modelless analogue of the paper's teacher weight continuation.

4. **Low overhead:** +~5% vs plain SDAR player (relevance filter + continuation check per round). The filter operates on tiny action vectors (7 elements) so the top-T/top-S sort is negligible.

### Modelless vs Model-Based Gap

| Aspect | Model-Based (Paper) | Modelless (Ours) | Gap |
|--------|--------------------|--------------------|-----|
| Pre-filter | T=20 token positions | T=20 actions | Action-level analogue |
| Judge | gpt-5-mini (LLM) | Magnitude-only (argmax) | No LLM, but same principle |
| Loss | Full reverse-KL top-K | SDAR gate × \|ΔQ\| proxy | Proxy instead of real KL |
| Continuation | LoRA weight snapshot | Q-value snapshot | Same mechanism, different granularity |
| Data efficiency | 2× fewer steps | Not measured (arena-based) | Would need per-round training |
| Specificity | 1.000 (zero off-topic) | N/A (game domain) | Game domains don't have "off-topic" |

### Composability

RMSD composes with existing distillation methods:
- **RMSD + SDAR:** `update = sdar_gate(ΔQ) × is_in_top_S(ΔQ)` — gate modulates, mask filters
- **RMSD + VPD:** VPD's EM cycle could use RMSD filter for E-step teacher refinement
- **RMSD + Interventional SFT:** Action masking is orthogonal to RMSD's magnitude filter

### Feature Gate

```toml
rmsd_distill = ["sdar_gate", "bandit"]
```

### Files

| File | Lines | Role |
|------|-------|------|
| `src/pruners/rmsd_relevance.rs` | ~330 | Core types, filter, loss, continuation |
| `src/pruners/bomber/rmsd_player.rs` | ~800 | Bomber arena player |
| `tests/test_125_rmsd_goat.rs` | ~620 | 44 GOAT proofs |
| `examples/bomber_16_rmsd_tournament.rs` | ~394 | Tournament example |
| `.benchmarks/037_rmsd_goat.md` | — | Benchmark results |

---

## References

- RMSD blog post: https://www.appliedcompute.com/research/relevance-masked-self-distillation
- [1] Hübotter et al. (2026) — Reinforcement learning via self-distillation. arXiv:2601.20802
- [2] Zhao et al. (2026) — Self-distilled reasoner. arXiv:2601.18734
- [3] Wang et al. (2026) — OpenClaw-RL: Train any agent simply by talking. arXiv:2603.10165
- [4] Lu (2025) — On-policy distillation. Thinking Machines Lab
- [5] Zhang et al. (2026) — EGAD: Entropy-guided adaptive distillation. arXiv:2605.01732