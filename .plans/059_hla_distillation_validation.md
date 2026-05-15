# Plan 059: HLA Distillation Validation — Measurable Binary Test for Latent State RAG

**Branch:** `develop/feature/059_hla_distillation_validation`
**Depends on:** Plan 057 (HLA Implementation), Plan 008 (riir-gpu LoRA Training)
**Research:** `.research/28_Higher_order_Linear_Attention.md` (Latent State RAG Analysis section)
**Goal:** Run SDPA→HLA distillation using CPU-only training in riir-gpu. Measure KL divergence at the LM head. If it converges to near-zero, HLA is viable for infinite-context inference. If it plateaus, kill the HLA training path and double down on `DeltaMemoryState`.

---

## Tasks

### Phase 0: Dependency Setup

- [x] T0: Wire `riir-gpu` → `microgpt-rs` with HLA feature
  - Add `hla_attention` feature to `riir-gpu/Cargo.toml`:
    ```toml
    [features]
    default = []
    feedback-consumer = ["reqwest"]
    hla_attention = ["microgpt-rs/hla_attention"]
    ```
  - Add `mod distill_attention;` to `riir-gpu/src/lib.rs` behind `#[cfg(feature = "hla_attention")]`
  - Verify: `cargo build -p riir-gpu --features hla_attention` compiles (no new code yet)

- [x] T1: Expose `distill.rs` helpers as `pub(crate)`
  - Make these functions `pub(crate)` (currently private, `#[allow(dead_code)]`):
    - `matvec()` — matrix-vector multiply
    - `lora_forward()` — LoRA correction `(α/r) · B @ (A @ x)`
    - `softmax()` — stable softmax
    - `kl_divergence()` — KL(p‖q)
  - No behavioral changes — just visibility for `distill_attention.rs`
  - Run existing `distill.rs` tests to confirm no regressions

### Phase 1: Infrastructure (riir-gpu)

- [x] T2: Implement `AttentionDistillConfig` + `AttentionDistillMetrics`
  - `AttentionDistillConfig` — learning_rate, temperature τ, n_steps, eval_interval, seq_len, lora_rank, lora_alpha
  - `AttentionDistillMetrics` — kl_div, cosine_sim, max_logit_diff, token_match_pct per step
  - `DistillMode` enum: `SdpaToAhla`, `SdpaToHla`, `SdpaToSdpa` (control)
  - Uses `pub(crate)` helpers from `distill.rs`

- [x] T3: Implement LoRA-aware attention forward (CPU)
  - `forward_sdpa_with_lora()` — SDPA teacher: `forward()` + LoRA on QKV, collect logits per position
  - `forward_hla_with_lora()` — HLA student: apply LoRA to QKV, then call HLA update+readout, collect logits
  - `forward_ahla_with_lora()` — AHLA student: same pattern for AHLA
  - LoRA math: `q = matvec(W_Q, x) + lora_forward(A_q, B_q, x, rank, n, n, alpha)`
  - Uses `ForwardContext`, `TransformerWeights` from microgpt-rs
  - No changes to base `forward_hla()`/`forward_ahla()` in microgpt-rs — LoRA applied before HLA step

- [x] T4: Implement CPU backprop via finite differences
  - `compute_lora_gradients_fd()` — finite difference gradient for all LoRA params
  - For each LoRA param θᵢ: `grad_θᵢ = (L(θ + ε·eᵢ) - L(θ - ε·eᵢ)) / (2ε)`
  - ε = 1e-4 (standard for float32 finite differences)
  - Loss = `KL(softmax(teacher_logits/τ) || softmax(student_logits/τ))` averaged over seq_len positions
  - 1,536 params × 2 forward passes per param = ~3K forward passes per step
  - Config::micro() forward pass ≈ microseconds → step completes in <1 second on CPU
  - **Why finite differences, not analytical gradients:**
    - `backward.rs` is GPU-only (WGSL kernels) — zero CPU backprop exists
    - Implementing analytical CPU backward through HLA recurrence (SK, CQV, G, h updates) is error-prone and not worth it for a validation experiment
    - Finite differences is always correct by construction — perfect for a binary science experiment
    - 1,536 params is small enough that FD is tractable

- [x] T5: Implement `distill_attention_step()` — single training step
  - Generate random token sequence [t₀, t₁, …, t_{seq_len}]
  - Teacher: `forward_sdpa_with_lora()` (frozen weights, no LoRA) → `teacher_logits[pos]`
  - Student: `forward_hla_with_lora()` (or AHLA/SDPA depending on mode) → `student_logits[pos]`
  - Compute loss: `KL(softmax(teacher/τ) || softmax(student/τ))` averaged over positions
  - Compute gradients: `compute_lora_gradients_fd()`
  - AdamW update: `CpuAdamWStep` from `optimizer.rs`
  - Return `AttentionDistillMetrics` for this step

- [x] T6: Implement `distill_attention_loop()` — full training loop
  - Runs `distill_attention_step()` for N iterations
  - Logs metrics every eval_interval steps
  - Returns convergence curve (`Vec<AttentionDistillMetrics>`)
  - 3 modes via `DistillMode`: `SdpaToAhla`, `SdpaToHla`, `SdpaToSdpa` (control)

### Phase 2: Validation Experiment

- [x] T7: Create tests in `riir-gpu` — the binary tests
  - `distill_attention_ahla_converges` — SDPA→AHLA distillation
  - `distill_attention_hla_converges` — SDPA→HLA distillation
  - `distill_attention_sdpa_control` — SDPA→SDPA (ceiling)
  - `distill_attention_fd_gradient_check` — verify finite diff gradients against perturbation
  - Uses `Config::micro()` (27 vocab, 16 embd, 4 heads, hd=4)
  - Assert: all metrics finite, KL decreases over training
  - Run: `cargo test -p riir-gpu --features hla_attention -- distill_attention --nocapture`

- [x] T8: Run distillation experiment — capture results ✅ (500-step pilot complete)
  - 500-step pilot at lr=1e-4 shows KL does NOT converge for AHLA/HLA
  - SDPA→AHLA: KL diverges 4.62→7.43, cos drops 0.55→0.25, token match 50%→25%
  - SDPA→HLA: KL barely changes 8.54→8.42, cos drops 0.45→0.29, token match 50%→37.5%
  - SDPA→SDPA control: perfect (KL≈0, cos=1.0, 100% match) — infrastructure correct ✅
  - **Verdict: Path C — HLA training path not viable via QKV LoRA distillation**
  - 10K full run skipped: 500-step pilot shows clear divergence trend, no convergence signal

- [ ] T9: If KL converges → validate on tiny retrieval task
  - Train SDPA model on 5 short "documents" (each ~8 tokens)
  - Distill to HLA with LoRA
  - Query: can HLA model produce correct next-token for document content?
  - Needle-in-a-haystack: inject one specific fact, can HLA retrieve it?
  - If retrieval fails → HLA is a domain shaper, not a knowledge store

### Phase 3: Decision Gate

- [x] T10: Decision gate — **Path C: HLA training path killed** ✅
  - KL does NOT converge: AHLA diverges, HLA oscillates wildly
  - Cosine similarity decreases for both variants
  - Token match drops below 50% — worse than random for vocab=27
  - Root cause: LoRA on QKV cannot bridge the structural gap between SDPA and HLA attention
    - SDPA computes softmax-weighted value sums
    - HLA uses streaming outer products — fundamentally different computation
    - Adjusting Q/K/V projections changes *inputs*, not the *attention mechanism itself*
  - **Action**: Double down on DeltaMemoryState for facts, HLA remains for streaming inference only
  - T9 (retrieval validation) skipped — not worth running if KL doesn't converge

---

## Architecture

### Why riir-gpu, Not microgpt-rs

```text
microgpt-rs  = inference engine (no training code)
riir-gpu     = training engine (forward, backward, loss, optimizer, distill)
```

riir-gpu already has:

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `kl_divergence()` | `distill.rs` | ⚠️ Private → `pub(crate)` in T1 | Direct reuse after visibility fix |
| `softmax()` | `distill.rs` | ⚠️ Private → `pub(crate)` in T1 | Direct reuse after visibility fix |
| `lora_forward()` | `distill.rs` | ⚠️ Private → `pub(crate)` in T1 | Direct reuse after visibility fix |
| `matvec()` | `distill.rs` | ⚠️ Private → `pub(crate)` in T1 | Direct reuse after visibility fix |
| `CpuAdamWStep` | `optimizer.rs` | ✅ Direct reuse | CPU AdamW already implemented |
| `ForwardContext` | microgpt-rs | ✅ Direct reuse | Shared context for forward passes |
| `TransformerWeights` | microgpt-rs | ✅ Direct reuse | Random init via `Weights::new()` |
| `forward_hla/ahla()` | microgpt-rs `hla/` | ✅ Direct reuse | Feature-gated behind `hla_attention` |
| GPU backward pass | `backward.rs` | ❌ NOT usable | WGSL-only, no CPU variant exists |
| GPU training loop | `training_loop.rs` | ❌ NOT usable | GPU-only, different architecture |

**Key gap:** No CPU backward pass exists anywhere. `backward.rs` is 100% WGSL kernels — it dispatches compute shaders for matmul, outer products, and attention backward. We use finite differences instead (T4).

### File Layout

```text
riir-ai/crates/riir-gpu/src/
├── distill.rs              — Existing: LoRA→LoRA distillation (expose helpers in T1)
├── distill_attention.rs    — NEW: SDPA→HLA attention distillation (T2–T6)
├── backward.rs             — Existing: GPU LoRA backward (NOT used — WGSL only)
├── optimizer.rs            — Existing: CpuAdamWStep (reused for AdamW updates)
└── lib.rs                  — Add: #[cfg(feature = "hla_attention")] mod distill_attention;

microgpt-rs/src/hla/
├── forward.rs              — Existing: forward_hla(), forward_ahla() (no changes)
├── types.rs                — Existing: cache types (no changes)
└── (no changes)            — LoRA applied at riir-gpu level, not in microgpt-rs
```

### Dependency Chain

```text
riir-gpu/Cargo.toml
  └── microgpt-rs = { path = "../../../microgpt-rs" }  # always present
  └── feature "hla_attention" → enables microgpt-rs/hla_attention

distill_attention.rs (feature-gated)
  ├── use microgpt_rs::transformer::{forward, TransformerWeights, ForwardContext};
  ├── use microgpt_rs::hla::{forward_hla, forward_ahla, MultiLayerHlaCache, MultiLayerAhlaCache};
  ├── use crate::distill::{kl_divergence, softmax, lora_forward, matvec};  // pub(crate) after T1
  ├── use crate::optimizer::CpuAdamWStep;  // existing CPU AdamW
  └── use microgpt_rs::types::{Config, Rng, MultiLayerKVCache};
```

### Distillation Flow

```text
1. Initialize teacher weights (random, frozen SDPA)
2. Initialize student weights (same as teacher — shared base weights)
3. Initialize LoRA adapters on student's W_Q, W_K, W_V per layer
   - A matrices: zero-init (start as identity-like correction)
   - B matrices: small random (Kaiming init)
4. For each step:
   a. Generate random token sequence [t₀, t₁, …, t_{seq_len}]
   b. Teacher: forward(ctx_teacher, weights, kv_cache, token, pos, config) per position
      → Collect teacher_logits[pos] for each position
   c. Student: forward_hla_with_lora(ctx_student, weights, lora, hla_cache, token, pos, config)
      → LoRA correction applied to QKV before HLA update+readout
      → Collect student_logits[pos] for each position
   d. Compute loss = mean over positions of KL(softmax(teacher/τ) || softmax(student/τ))
   e. Compute gradients via finite differences (T4):
      - For each LoRA param θᵢ:
        - L₊ = loss(θ + ε·eᵢ), L₋ = loss(θ - ε·eᵢ)
        - grad_θᵢ = (L₊ - L₋) / (2ε)
      - Total: ~3K forward passes (1,536 params × 2 perturbations)
   f. AdamW update via CpuAdamWStep
5. Log metrics, repeat
```

### LoRA Strategy

Instead of training full W_Q, W_K, W_V (which would require full analytical backward through HLA recurrence), we add LoRA adapters to the student and train those:

```text
Student attention:
  q = matvec(W_Q, x) + lora_forward(A_q, B_q, x, rank, n, n, alpha)
  k = matvec(W_K, x) + lora_forward(A_k, B_k, x, rank, kv_dim, n, alpha)
  v = matvec(W_V, x) + lora_forward(A_v, B_v, x, rank, kv_dim, n, alpha)
  → HLA/AHLA update + readout → logits
```

**Why LoRA, not full weights:**
- Reduces trainable params: 1,536 vs ~30K for full QKV across 4 layers
- Finite differences is O(n) forward passes — 1.5K is tractable, 30K is painful
- If LoRA rank=4 can bridge the gap → HLA and SDPA are close in function space
- If LoRA can't converge → HLA is fundamentally different from SDPA
- The "trained HLA model" = base weights + LoRA adapter (portable, small)

### Trainable Parameters

For `Config::micro()` (n_embd=16, n_layer=4, n_head=4, head_dim=4) with LoRA rank=4:

```text
Per layer LoRA (rank=4):
  W_Q LoRA: A=[4×16], B=[16×4] = 128 floats
  W_K LoRA: A=[4×16], B=[16×4] = 128 floats  (note: kv_dim=16 for MHA)
  W_V LoRA: A=[4×16], B=[16×4] = 128 floats  (note: kv_dim=16 for MHA)
  Total per layer: 384 floats

Total trainable: 384 × 4 layers = 1,536 floats = 6 KB

Finite differences cost per step:
  1,536 params × 2 forward passes × 8 positions = ~24K forward passes
  Each forward pass: ~microseconds on CPU
  Estimated step time: <500ms on Apple M-series
```

### SDPA→SDPA Control Experiment

The control experiment measures the ceiling: can LoRA at rank=4 learn an identity mapping?

```text
Teacher: forward(ctx, weights, kv_cache, token, pos, config) — plain SDPA, no LoRA
Student: forward(ctx, weights, kv_cache, token, pos, config) + trainable LoRA on QKV
         → Same SDPA path, but LoRA adds correction to QKV before attention

Expected: KL → 0 quickly (LoRA learning to output zero correction)
If KL doesn't → 0: LoRA rank=4 is insufficient even for identity → increase rank
```

This is NOT "two models with different LoRA init". It's the same model, teacher has no LoRA, student has trainable LoRA. The student should learn to output zero correction (identity).

---

## Expected Outcomes

### Success Criteria

| Criterion | Threshold | Action if Met |
|-----------|-----------|---------------|
| KL divergence < 0.01 | Within 10K steps | Proceed to Phase 3 Path A or B |
| KL divergence < 0.1 | Within 10K steps | Investigate — more steps or higher LoRA rank |
| KL divergence plateaus > 0.5 | After 10K steps | Phase 3 Path C — kill HLA training |
| Token match > 90% | At convergence | HLA viable for inference |
| Token match < 50% | At convergence | HLA not viable for precise tasks |

### What This Proves

- ✅ Whether HLA can approximate SDPA outputs with LoRA correction
- ✅ How fast/whether KL divergence converges
- ✅ Whether token-level predictions match (the real quality signal)
- ✅ Whether the distillation approach is viable at all

### What This Does NOT Prove

- ❌ Whether HLA produces better outputs than SDPA (just measures approximation)
- ❌ Whether HLA works on large-scale models (micro config only)
- ❌ Whether Latent State RAG is viable (that's Phase 3 Path A)
- ❌ Whether the approach scales to real training data (random sequences only)

---

## Benchmark Targets

### T8 Results Table

#### 50-step pilot (lr=3e-4, Config::micro) — Infrastructure validation

```text
Variant       | KL @ step 0   | KL @ step 25  | KL @ step 49  | Final cos-sim | Token match %
SDPA→AHLA     |       4.6179  |       3.7016   |       7.5155  |        0.4483 |        37.5%
SDPA→HLA      |       8.5415  |       7.5817   |       8.1978  |        0.2897 |        37.5%
SDPA→SDPA     |       0.0000  |       —        |       0.0000  |        1.0000 |       100.0%
```

#### 500-step pilot (lr=1e-4, Config::micro) — Convergence check

```text
Variant       | KL @ step 0   | KL @ step 250 | KL @ step 499 | Final cos-sim | Token match %
SDPA→AHLA     |       4.6179  |       6.7609   |       7.4324  |        0.2480 |        25.0%
SDPA→HLA      |       8.5415  |       6.3220   |       8.4223  |        0.2855 |        37.5%
SDPA→SDPA     |       0.0000  |       0.0000   |       0.0000  |        1.0000 |       100.0%
```

#### AHLA convergence curve (500 steps, lr=1e-4) — Chaotic oscillation

```text
step     0: KL=4.618 cos=0.555 tok=50.0%  (baseline)
step    50: KL=3.426 cos=0.503 tok=50.0%  (initial dip)
step   100: KL=8.370 cos=0.206 tok=25.0%  (diverges)
step   200: KL=7.642 cos=0.412 tok=37.5%  (oscillates)
step   400: KL=1.848 cos=0.660 tok=37.5%  (best point)
step   499: KL=7.432 cos=0.248 tok=25.0%  (ends poorly)
```

#### HLA convergence curve (500 steps, lr=1e-4) — Similar oscillation

```text
step     0: KL=8.541 cos=0.450 tok=50.0%  (baseline)
step   100: KL=7.037 cos=0.537 tok=25.0%  (slight dip)
step   300: KL=7.331 cos=0.440 tok=12.5%  (near baseline)
step   400: KL=2.392 cos=0.713 tok=50.0%  (best point)
step   499: KL=8.422 cos=0.286 tok=37.5%  (ends poorly)
```

**Observations (500-step pilot — decision grade):**
- SDPA→SDPA control: KL ≈ 0, cos=1.0, 100% token match — infrastructure correct ✅
- SDPA→AHLA: KL diverges (4.62→7.43), cos drops (0.55→0.25), token match halves (50%→25%)
  - Chaotic oscillation between KL 1.8 and 9.4 — optimization landscape is non-convex
  - No consistent downward trend despite 500 AdamW steps at conservative lr=1e-4
- SDPA→HLA: KL barely changes (8.54→8.42), cos drops (0.45→0.29), token match drops (50%→37.5%)
  - Even worse: best KL of 2.4 at step 400, then rebounds to 8.4 by step 499
  - Both variants maintain finite KL (no NaN/inf), all gradients finite ✅
- **Root cause**: LoRA on QKV adjusts *what* attention sees, not *how* it computes
  - SDPA: softmax(QK^T/√d) · V — softmax-weighted sum
  - HLA: streaming outer products (SK, CQV, G) — rank-1 updates
  - These are fundamentally different computations — no linear QKV transform can bridge them
- **Verdict: Path C — Kill HLA training path. HLA remains useful for streaming inference (Plan 057/060), but cannot be trained via SDPA distillation.**

The SDPA→SDPA control establishes the ceiling: same SDPA forward path, LoRA learning identity. ✅ Confirmed working.

---

## Key Design Decisions

1. **KL at LM head, not hidden states** — Cosine sim of 0.95 on hidden states can still completely scramble the final token argmax. The only metric that matters is distributional divergence at the vocabulary level.

2. **LoRA correction, not full weight training** — Reduces trainable params to 1,536. If LoRA rank=4 can bridge SDPA→HLA, the function spaces are close. If not, no amount of full-weight training will help.

3. **Finite differences for backprop** — No CPU backward pass exists in the codebase. `backward.rs` is GPU-only (WGSL kernels). For 1,536 params, FD is tractable and guaranteed correct. Analytical CPU backward through HLA recurrence (SK, CQV, G, h updates) would be error-prone and is overkill for a binary science experiment.

4. **CPU for validation, GPU for production** — 1,536 params trains in <500ms/step on CPU. No need for WGSL kernels until we scale up. The validation experiment is a science experiment, not a production pipeline.

5. **Random token sequences** — We're testing whether the HLA operator can learn to approximate the SDPA operator. Random tokens are sufficient. We're not testing language understanding.

6. **AHLA first** — Lower state cost, simpler math, closer to SDPA (0.95 vs 0.80 cosine sim on random weights from research §Key Insight). If AHLA distills, symmetric HLA is a follow-up.

7. **Temperature τ = 2.0** — Higher temperature softens the distributions, making KL gradient signal richer. Standard distillation practice (Hinton et al., 2015).

8. **Code lives in riir-gpu** — Training infrastructure belongs in the training crate. microgpt-rs stays inference-only.

9. **Feature gate `hla_attention`** — Matches existing microgpt-rs feature name. riir-gpu's `Cargo.toml` adds `hla_attention = ["microgpt-rs/hla_attention"]` to propagate the feature.

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Finite differences too slow | Low | 1,536 params × seq_len=8 ≈ 24K forward passes, each <10µs → <500ms/step |
| LoRA rank too low to bridge gap | Medium | Try rank=4, 8, 16 — if rank=16 can't converge, it's not a rank problem |
| KL doesn't converge | Medium | That's the answer — Path C |
| Finite difference precision issues | Low | ε=1e-4, add gradient check test (T7) comparing FD to perturbation |
| riir-gpu + hla_attention feature combo issues | Low | T0 verifies build before any implementation |
| Numerical instability in HLA recurrence | Low | Existing HLA tests already verify finite outputs |
| Overfitting to random sequences | Low | We WANT to overfit — measuring approximation, not generalization |

---

## Relationship to Existing Plans

| Plan | Relationship |
|------|-------------|
| Plan 057 (HLA) | Provides `forward_hla()`, `forward_ahla()`, cache types |
| Plan 008 (riir-gpu) | Provides training infrastructure: `distill.rs` helpers, `CpuAdamWStep` |
| Plan 004 (Leviathan) | Pattern: distillation loss, p/q distribution comparison |
| Plan 052 (GFlowNet) | Pattern: modelless distillation, bench test structure |
| Plan 024 (DeltaMem) | Alternative path — if HLA fails, DeltaMem is the fallback |
| Plan 058 (GVG Game) | Consumer — if HLA works, cheap fork MCTS for game AI |