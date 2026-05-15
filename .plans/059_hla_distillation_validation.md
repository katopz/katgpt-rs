# Plan 059: HLA Distillation Validation — Measurable Binary Test for Latent State RAG

**Branch:** `develop/feature/059_hla_distillation_validation`
**Depends on:** Plan 057 (HLA Implementation — `forward_hla()`, `forward_ahla()`), Plan 004 (Leviathan distillation pattern)
**Research:** `.research/28_Higher_order_Linear_Attention.md` (Latent State RAG Analysis section)
**Goal:** Run SDPA→HLA distillation on micro config. Measure KL divergence at the LM head. If it converges to near-zero, HLA is viable for infinite-context inference. If it plateaus, kill the HLA training path and double down on `DeltaMemoryState`.

---

## Tasks

### Phase 1: Distillation Infrastructure

- [ ] T1: Create `src/hla/distill.rs` — feature-gated behind `hla_attention`
  - `struct DistillConfig` — learning_rate, temperature τ, n_steps, eval_interval, seq_len
  - `struct DistillMetrics` — kl_div, cosine_sim, max_logit_diff, token_match_pct per step
  - `fn kl_divergence(p: &[f32], q: &[f32]) -> f32` — KL(p || q) with numerical stability
  - `fn cosine_similarity(a: &[f32], b: &[f32]) -> f32`

- [ ] T2: Implement `distill_step()` — single training step
  - Forward pass with SDPA teacher (frozen) → teacher_logits
  - Forward pass with HLA student (trainable) → student_logits
  - Compute KL(softmax(teacher/τ) || softmax(student/τ))
  - Backprop through student W_Q, W_K, W_V only (FFN/embeddings frozen)
  - SGD update with gradient clipping

- [ ] T3: Implement `distill_loop()` — full training loop
  - Generates random token sequences (seq_len tokens)
  - Runs `distill_step()` for N iterations
  - Logs metrics every eval_interval steps
  - Returns convergence curve (Vec<DistillMetrics>)
  - Zero external deps — pure Rust, no autograd framework

- [ ] T4: Implement manual backprop for attention projections
  - Forward: x → W_Q·x=q, W_K·x=k, W_V·x=v → HLA readout → logits
  - Backward: ∂L/∂logits → ∂L/∂attn_out → ∂L/∂W_Q,W_K,W_V (chain rule through HLA readout)
  - Only `attn_wq`, `attn_wk`, `attn_wv` per layer are trainable
  - `attn_wo`, `mlp_w1`, `mlp_w2`, `wte`, `wpe`, `lm_head` frozen

### Phase 2: Validation Experiment

- [ ] T5: Create `tests/bench_hla_distill.rs` — the binary test
  - Run distill_loop with `Config::micro()` (27 vocab, 16 embd, 4 heads, hd=4)
  - 3 variants: SDPA→HLA (symmetric), SDPA→AHLA (asymmetric), SDPA→SDPA (control)
  - Report: KL divergence curve, final cosine sim, token match %, convergence speed
  - Assert: all metrics finite, KL decreases monotonically (or plateaus)
  - Run: `cargo test --features hla_attention --test bench_hla_distill -- --nocapture`

- [ ] T6: Run distillation experiment — capture results
  - Fill in the results table below
  - The binary question: does KL drop below 0.01 within 10K steps?

- [ ] T7: If KL converges → validate on tiny retrieval task
  - Train SDPA model on 5 short "documents" (each ~8 tokens)
  - Distill to HLA
  - Query: can HLA model produce correct next-token for document content?
  - Needle-in-a-haystack: inject one specific fact, can HLA retrieve it?
  - If retrieval fails → HLA is a domain shaper, not a knowledge store

### Phase 3: Decision Gate

- [ ] T8: Write decision document based on T6/T7 results
  - Path A: KL ≈ 0, retrieval works → Proceed to `forward_hybrid()` (Plan 060)
  - Path B: KL ≈ 0, retrieval fails → HLA is domain shaper only, DeltaMem for facts
  - Path C: KL plateaus → Kill HLA training path, double down on DeltaMemoryState

---

## Architecture

```text
src/hla/
├── mod.rs              — Add: pub mod distill; (behind #[cfg(test)] or feature gate)
├── distill.rs          — NEW: distillation loop + metrics
├── types.rs            — Existing: HLA/AHLA cache types
├── kernel.rs           — Existing: HLA/AHLA kernels
└── forward.rs          — Existing: forward_hla(), forward_ahla()

tests/
└── bench_hla_distill.rs — NEW: the binary validation test
```

### Distillation Flow

```text
1. Initialize teacher weights (random, frozen SDPA)
2. Copy teacher weights to student (trainable HLA)
3. For each step:
   a. Generate random token sequence [t0, t1, ..., t_{seq_len}]
   b. Teacher: forward(ctx_t, weights_teacher, cache_kv, ...) for each position
      → Collect teacher_logits[pos] for each position
   c. Student: forward_hla(ctx_s, weights_student, cache_hla, ...) for each position
      → Collect student_logits[pos] for each position
   d. For each position:
      - p = softmax(teacher_logits[pos] / τ)
      - q = softmax(student_logits[pos] / τ)
      - KL += p · (log(p) - log(q))
   e. Backprop KL through student W_Q, W_K, W_V
   f. SGD update: w -= lr · ∂KL/∂w
4. Log metrics, repeat
```

### Trainable Parameters

For `Config::micro()` (n_embd=16, n_layer=4, n_head=4, head_dim=4):

```text
Per layer trainable:
  attn_wq: [16 × 16] = 256 floats
  attn_wk: [16 × 16] = 256 floats  (n_embd × n_embd since n_kv_head == n_head)
  attn_wv: [16 × 16] = 256 floats
  Total per layer: 768 floats

Frozen per layer:
  attn_wo: [16 × 16] = 256 floats
  mlp_w1:  [64 × 16] = 1024 floats
  mlp_w2:  [16 × 64] = 1024 floats

Frozen global:
  wte:     [27 × 16] = 432 floats
  wpe:     [16 × 16] = 256 floats
  lm_head: [27 × 16] = 432 floats

Total trainable: 768 × 4 layers = 3,072 floats = 12 KB
Total frozen:     2,560 × 4 + 1,120 = 11,360 floats = 45 KB
```

Tiny model, fast iteration. The entire training loop should run in seconds, not minutes.

---

## Expected Outcomes

### Success Criteria

| Criterion | Threshold | Action if Met |
|-----------|-----------|---------------|
| KL divergence < 0.01 | Within 10K steps | Proceed to Phase 3 Path A or B |
| KL divergence < 0.1 | Within 10K steps | Investigate — maybe more steps or lower LR |
| KL divergence plateaus > 0.5 | After 10K steps | Phase 3 Path C — kill HLA training |
| Token match > 90% | At convergence | HLA viable for inference |
| Token match < 50% | At convergence | HLA not viable for precise tasks |

### What This Proves

- ✅ Whether HLA can approximate SDPA outputs with distillation
- ✅ How fast/whether KL divergences converges
- ✅ Whether token-level predictions match (the real quality signal)
- ✅ Whether the distillation approach is viable at all

### What This Does NOT Prove

- ❌ Whether HLA produces better outputs than SDPA (just measures approximation)
- ❌ Whether HLA works on large-scale models (micro config only)
- ❌ Whether Latent State RAG is viable (that's Phase 3 Path A)
- ❌ Whether the approach scales to real training data (random sequences only)

---

## Benchmark Targets

### T6 Results Table (to be filled)

```text
Variant       | KL @ step 100 | KL @ step 1K | KL @ step 10K | Final cos-sim | Token match %
SDPA→AHLA     |           ??? |          ??? |            ??? |           ??? |          ???
SDPA→HLA      |           ??? |          ??? |            ??? |           ??? |          ???
SDPA→SDPA     |           ??? |          ??? |            ??? |           ??? |          ???
```

The SDPA→SDPA control (training one SDPA model to match another with different init) establishes the ceiling.

---

## Key Design Decisions

1. **KL at LM head, not hidden states** — Cosine sim of 0.95 on hidden states can still completely scramble the final token argmax. The only metric that matters is distributional divergence at the vocabulary level.

2. **Manual backprop, no autograd** — We don't have a tensor framework. The backward pass through HLA readout is tractable (chain rule through matmuls). For micro config (3K trainable params), this is ~200 lines of code.

3. **Random token sequences** — We're not testing language understanding. We're testing whether the HLA operator can learn to approximate the SDPA operator. Random tokens are sufficient for this.

4. **AHLA first** — Lower state cost, simpler math, closer to SDPA (0.95 vs 0.80 cosine sim). If AHLA distills, symmetric HLA is a follow-up.

5. **Temperature τ = 2.0** — Higher temperature softens the distributions, making KL gradient signal richer. Standard distillation practice (Hinton et al., 2015).

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Manual backprop bugs | High | Finite difference gradient check in T4 |
| KL doesn't converge | Medium | That's the answer — Path C |
| HLA readout gradient is ill-conditioned | Low | Gradient clipping + lower LR |
| Overfitting to random sequences | Low | We WANT to overfit — measuring approximation, not generalization |

---

## Relationship to Existing Plans

| Plan | Relationship |
|------|-------------|
| Plan 057 (HLA) | Provides `forward_hla()`, `forward_ahla()`, cache types |
| Plan 004 (Leviathan) | Pattern: distillation loss, p/q distribution comparison |
| Plan 052 (GFlowNet) | Pattern: modelless distillation, bench test structure |
| Plan 024 (DeltaMem) | Alternative path — if HLA fails, DeltaMem is the fallback |
| Plan 058 (GVG Game) | Consumer — if HLA works, cheap fork MCTS for game AI |