# Plan 048: Research Audit Fixes — Close Critical Gaps in Training Pipeline

**Branch:** `develop/feature/048_research_audit_fixes`
**Depends on:** Plan 040 (Cross-Training), Plan 041 (E2E Game Training), Plan 043 (TurboQuant), Plan 044 (PFlash)
**Goal:** Fix known bugs and wire deferred GPU paths identified in research-to-implementation cross-reference audit.

---

## Problem Statement

The research audit (comparing 21 `.research/` papers against 4 repos) revealed **90% implementation coverage** but uncovered 5 critical gaps that prevent the full training→inference→feedback loop from being production-reliable:

1. **Attention backward is incomplete** — Q perturbations don't propagate through attention to logits (`backward.rs:L690-694`). LoRA gradients for attention layers may be incorrect.
2. **KL divergence = 0.0 placeholder** — Distillation quality is unmeasurable (`distill.rs:L531`).
3. **PFlash GPU shaders orphaned** — 4 WGSL kernels written, compiled, but never dispatched from Rust. CPU fallback works but GPU acceleration is dead code.
4. **Game replay parser is stub** — `parse_replay()` returns empty `Vec::new()`, blocking real game replay training.
5. **TTT feedback loop incomplete** — `feedback.rs` sends inference results to cache endpoint, but nothing consumes them for retraining.

These gaps mean: training may produce incorrect gradients, distillation quality is invisible, GPU-accelerated prefill is unused, game training can't consume real replays, and the self-improving loop sends data into a void.

## Audit Summary (Research ↔ Implementation)

### Fully Implemented (17/21 papers)

| # | Paper | Where | Status |
|---|-------|-------|--------|
| 00 | Neuro-Symbolic Architecture | `katgpt-rs/src/speculative/` | ✅ DFlash, DDTree, Percepta |
| 01 | Advanced Neuro-Symbolic | `katgpt-rs/src/transformer.rs` | ✅ PagedKV, GQA, SIMD hints |
| 02 | Speculative Decoding (Leviathan) | `katgpt-rs/src/speculative/verifier.rs` | ✅ Full rejection sampling |
| 03 | Commercial Strategy | 4-repo architecture split | ✅ Engine/Fuel separation |
| 04 | LoRA Architecture | `riir-gpu/src/lora.rs` | ✅ 6 targets/layer, BLAKE3 |
| 05 | Artifact Definition | `riir-validator-sdk/` | ✅ 10 WASM validators |
| 06 | Raven RSM | `katgpt-rs/src/transformer.rs` | ✅ O(1) KV cache |
| 07 | Screening Absolute Relevance | `katgpt-rs/src/speculative/types.rs` | ✅ Continuous [0,1] |
| 08 | TwELL Sparse MLP | `katgpt-rs/src/types.rs` | ✅ Feature-gated sparse GEMV |
| 09 | EMO Emergent Modularity | `riir-ai/crates/riir-router/` | ✅ ExpertRegistry + routing |
| 11 | PPoT | `katgpt-rs/src/speculative/ppot/` | ✅ CPU logit resampling |
| 12 | TRT (rejection knowledge) | `katgpt-rs/src/speculative/ppot/knowledge.rs` | ✅ Adaptive patterns |
| 14 | Learning Beyond Gradients | `katgpt-rs/src/pruners/absorb_compress.rs` | ✅ Absorb+Compress |
| 15 | Reinforced Agent (reviewer) | `katgpt-rs/src/pruners/review_metrics.rs` | ✅ Helpfulness/Harmfulness |
| 16 | AutoTTS (β parameterization) | `riir-gpu/src/training_config.rs` | ✅ BetaConfig |
| 18 | Free Transformer Latent Injection | `katgpt-rs/src/types.rs` (DomainLatent), `riir-gpu/src/domain_latent.rs` | 🟡 Full VAE ❌, mid-layer K/V domain embedding ✅ (Plan 038) |
| 19 | TTT Test-Time Training | `katgpt-rs/src/feedback.rs`, `riir-burner/` | 🟡 Feedback sends, not consumed |
| 20 | TurboQuant | `katgpt-rs/src/turboquant/` | ✅ CPU path, GPU kernel exists |

### Correctly Rejected (1/21 papers)

| # | Paper | Reason |
|---|-------|--------|
| 10 | ColaDLM Latent Diffusion | Architecturally incompatible — DDTree branches on discrete tokens, ColaDLM branches on continuous latent vectors; multi-step denoising incompatible with single-pass speculative decoding |

### Partially Distilled (2/21 papers)

| # | Paper | Rejected Mechanism | Distilled Concept |
|---|-------|--------------------|-------------------|
| 17 | Fast BLT Byte-Level | ❌ Language path — BPE tokens, monolithic architecture, speculative decoding already exists via LeviathanVerifier | ✅ Game domain via Plan 039 — action-level = byte-level; 6-action vocab maps to BLT's byte concept, 13×13 grid cells as "byte sequence", no BPE needed |
| 18 | Free Transformer Latent Injection | ❌ Full VAE + binary mapper (65536-dim one-hot) — requires training from scratch, no pretrained weights exist | ✅ DomainLatent via Plan 038 — mid-layer K/V injection of learned domain embedding [kv_dim], LoRA-compatible, feature-gated, GPU training in riir-gpu |

### Correctly Deferred (0/21 papers)

### Partially Implemented (1/21 papers)

| # | Paper | Reason |
|---|-------|--------|
| 13 | NVIDIA Dynamo Agentic Lessons | Catalog shaping ✅, general agentic streaming ❌ |

---

## Design

### Phase 1: Fix Training Pipeline Bugs (riir-gpu)

#### Task 1: Fix Attention Backward Propagation

**File:** `riir-ai/crates/riir-gpu/src/backward.rs`
**Problem:** Q perturbations don't propagate through attention to logits (L690-694, test disabled).
**Root Cause:** The backward pass through attention assumes a simplified gradient path. The softmax attention gradient requires computing:
```
dL/dQ = (dL/d_scores × K^T) where dL/d_scores = softmax(scores) × (dL/d_output - Σ(dL/d_output × softmax(scores)))
```
But the current code doesn't properly accumulate gradients through the multi-head attention score computation.

**Fix:**
1. Implement `backward_attention()` that correctly computes dQ, dK, dV from d_attn_out
2. The gradient through scaled dot-product attention is:
   - `dscores = softmax(scores) * (d_out - sum(d_out * softmax(scores), dim=-1)) / sqrt(d_k)`
   - `dQ = dscores @ K^T`
   - `dK = dscores^T @ Q`
   - `dV = d_out` (simple)
3. Then propagate dQ through the QKV projection weights to get LoRA gradients for q_proj, k_proj, v_proj
4. Re-enable `test_analytical_gradients_reasonable` with numerical gradient check
5. CPU fallback (`compute_backward_cpu`) already exists for verification — use it to validate

**Estimated changes:** ~80 lines new gradient computation, ~30 lines test fixes

#### Task 2: Implement Real KL Divergence in Distillation

**File:** `riir-ai/crates/riir-gpu/src/distill.rs`
**Problem:** `kl_divergence: 0.0` placeholder at L531 — can't measure distillation quality.
**Fix:**
1. After SVD truncation, compute actual KL divergence:
   - Forward pass with target LoRA → get logits_target
   - Forward pass with draft LoRA → get logits_draft
   - `KL = Σ softmax(target) * (log_softmax(target) - log_softmax(draft))`
2. Use existing `loss_per_sample.wgsl` kernel pattern for GPU-accelerated KL
3. Or CPU-only for simplicity (distillation is offline): implement `kl_divergence_cpu()`
4. Write actual value to `DistillResult.kl_divergence`
5. Add test: known distributions → correct KL value

**Estimated changes:** ~40 lines KL computation, ~20 lines test

#### Task 3: Implement Game Replay Parser

**File:** `riir-ai/crates/riir-gpu/src/game/replay.rs`
**Problem:** `parse_replay()` at L281-287 returns empty `Vec::new()` — stub.
**Context:** The replay format is defined by `bomber_04_replay_gen` example (Plan 039/041), which outputs JSONL with fields: `round`, `tick`, `board`, `action`, `quality`, `player_type`.
**Fix:**
1. Implement `parse_replay(jsonl: &str) -> Vec<GameSample>` using existing `parse_jsonl()` helper
2. Map board state (13×13 cells) to token sequence
3. Map action enum to action token
4. Filter by quality > 0.5 and player_type in ["Validator", "HL"]
5. Return `Vec<GameSample>` ready for `encode_game_samples()`
6. Test: parse known replay JSONL → correct samples

**Estimated changes:** ~30 lines parser logic, ~15 lines test

### Phase 2: Wire Orphaned GPU Paths

#### Task 4: Wire PFlash GPU Dispatch

**Files:** `riir-ai/crates/riir-gpu/src/kernels/mod.rs`, `riir-ai/crates/riir-gpu/src/forward.rs`
**Problem:** 4 WGSL shaders exist (`flashprefill_mean_k.wgsl`, `flashprefill_block_score.wgsl`, `flashprefill_block_select.wgsl`, `flashprefill_sparse_forward.wgsl`) but are never compiled into `GpuPipelines` or dispatched.

**Current state (kernels/mod.rs GpuPipelines):**
```rust
pub struct GpuPipelines {
    pub matmul: PipelineBundle,
    pub add: PipelineBundle,
    // ... 16 pipelines, NO flashprefill entries
}
```

**Fix:**
1. Add 4 new pipeline fields to `GpuPipelines`:
   ```rust
   pub flashprefill_mean_k: PipelineBundle,
   pub flashprefill_block_score: PipelineBundle,
   pub flashprefill_block_select: PipelineBundle,
   pub flashprefill_sparse_forward: PipelineBundle,
   ```
2. Add shader source constants and entry points in `mod.rs`
3. Create `GpuFlashPrefillBuffers` struct (buffers for mean_k, block_scores, selected_indices, sparse_output)
4. Create `GpuFlashPrefillPass` struct with `score_and_select()` method that dispatches 4 kernels in sequence
5. Wire into `forward.rs` as optional pre-pass before standard attention (feature-gated)
6. Benchmark: GPU PFlash vs CPU PFlash on 1024+ token prompts
7. Note: `attention_score_tq.wgsl` (TurboQuant scoring) is also orphaned — wire separately in Task 5

**Estimated changes:** ~60 lines pipeline setup, ~100 lines buffer/pass structs, ~40 lines dispatch, ~30 lines test

#### Task 5: Wire TurboQuant GPU Attention Scoring

**Files:** `riir-ai/crates/riir-gpu/src/kernels/mod.rs`, `riir-ai/crates/riir-gpu/src/forward.rs`
**Problem:** `attention_score_tq.wgsl` exists but is not in `GpuPipelines` or dispatched.
**Context:** CPU `forward_turboquant()` in `katgpt-rs` works. GPU path would accelerate the dequantize→score→attention step.
**Fix:**
1. Add `attention_score_tq: PipelineBundle` to `GpuPipelines`
2. Create uniform buffer for TQ params (centroids, boundaries, bits, scale)
3. Wire into `forward.rs` as alternative to `attention_score` when turboquant config is active
4. Feature-gate behind existing `turboquant` feature
5. Test: GPU TQ scores match CPU TQ scores within tolerance

**Estimated changes:** ~30 lines pipeline, ~40 lines buffer/dispatch, ~20 lines test

### Phase 3: Close TTT Feedback Loop

#### Task 6: Feedback Consumer Service

**File:** `riir-ai/crates/riir-gpu/src/feedback.rs` (new) or extend `riir-ai/crates/riir-rest/`
**Problem:** `katgpt-rs/src/feedback.rs` POSTs `InferenceResult` to cache endpoint (Plan 042 Task 6 ✅), but nothing reads from that endpoint to trigger retraining. Feedback goes into a void.
**Context:** Plan 042 implemented the send side. This task implements the receive side.

**Architecture:**
```
katgpt-rs (inference)
    │ POST InferenceResult
    ▼
anyrag /cache/ingest (Plan 042 ✅)
    │ accumulate high-reward results
    │ POST /cache/export → JSONL
    ▼
riir-gpu feedback consumer (THIS TASK)
    │ 1. Poll anyrag for new high-reward JSONL
    │ 2. If enough new samples (> N):
    │ 3. Trigger Trainer::train_from_jsonl() with BetaConfig
    │ 4. Export new lora.bin
    │ 5. Signal hot-swap (write to watched path)
    ▼
katgpt-rs HotSwapPruner (Plan 032 ✅)
    │ BLAKE3 change detected → reload lora.bin
    ▼
Next inference uses updated LoRA
```

**Fix:**
1. Add `feedback_consumer.rs` to `riir-gpu` with `FeedbackConsumer` struct:
   - `config: FeedbackConfig` (endpoint URL, min_samples, poll_interval, beta)
   - `last_export_hash: u64` (avoid re-processing same data)
   - Methods: `poll()`, `should_retrain()`, `retrain()`, `signal_hotswap()`
2. `FeedbackConfig` with sensible defaults:
   - `min_new_samples: usize` (default: 100)
   - `poll_interval_secs: u64` (default: 300 = 5 min)
   - `beta: f32` (default: 0.5)
   - `domain: String` (which domain to consume)
   - `output_path: PathBuf` (where to write new lora.bin)
3. `retrain()` calls existing `Trainer::train_from_jsonl()` — no new training code
4. `signal_hotswap()` writes new lora.bin to output_path (HotSwapPruner watches via BLAKE3)
5. Add CLI example: `feedback_consumer --endpoint http://localhost:8080 --domain py2rs --beta 0.5`
6. Feature-gate behind `feedback-consumer` feature

**Estimated changes:** ~120 lines consumer logic, ~30 lines config, ~40 lines CLI example, ~20 lines test

### Phase 4: Validation & Documentation

#### Task 7: E2E Validation Suite

Run all fixes together to prove the full loop works:

1. **Attention backward correctness:**
   - Train LoRA with 100 steps on toy data
   - Compare GPU loss curve with CPU reference loss curve
   - Assert: loss difference < 5% at every step

2. **KL divergence measurement:**
   - Train target LoRA (rank=16), distill draft LoRA (rank=4)
   - Assert: `kl_divergence > 0.0` (no longer placeholder)
   - Assert: `kl_divergence < 2.0` (reasonable distillation)

3. **Game replay training:**
   - Run `bomber_04_replay_gen` → JSONL
   - Run `train_bomber --replay-dir output/replays`
   - Assert: samples loaded > 0 (parse_replay no longer returns empty)
   - Assert: loss decreases over epochs

4. **PFlash GPU dispatch:**
   - Score 1024-token prompt with GPU PFlash
   - Compare block selection with CPU PFlash
   - Assert: same blocks selected (within tolerance)
   - Benchmark: GPU time < CPU time

5. **TurboQuant GPU scoring:**
   - Run `forward_turboquant` on CPU
   - Run GPU TQ scoring path
   - Assert: attention scores match within 1%

6. **TTT feedback loop (manual):**
   - Start anyrag with `solution-cache` feature
   - Start `feedback_consumer` watching domain "py2rs"
   - Run 50 inferences in py2rs domain
   - Assert: feedback_consumer triggers retraining
   - Assert: new lora.bin written to output path
   - Assert: BLAKE3 checksum differs from original

#### Task 8: Update Documentation

1. Update `riir-ai/README.md`:
   - Mark wgpu training as "✅ Production-ready" (remove experimental caveat)
   - Add PFlash GPU section
   - Add TTT feedback consumer section
2. Update `riir-ai/.docs/06_gpu_training.md`:
   - Mark attention backward as fixed
   - Add GpuFlashPrefillPass to module layout
   - Add FeedbackConsumer to module layout
   - Update known issues (all resolved)
3. Update `katgpt-rs/README.md`:
   - Add "Self-Improving Loop" section referencing Plan 048
4. Create `riir-ai/.docs/13_research_audit_results.md`:
   - Full research↔implementation cross-reference table
   - Audit findings and resolutions

#### Task 9: Commit with conventional messages

Separate commits per logical unit:
1. `fix(riir-gpu): correct attention backward gradient propagation (Plan 048 T1)`
2. `fix(riir-gpu): implement real KL divergence in distillation (Plan 048 T2)`
3. `fix(riir-gpu): implement game replay parser (Plan 048 T3)`
4. `feat(riir-gpu): wire PFlash GPU dispatch — 4 WGSL kernels connected (Plan 048 T4)`
5. `feat(riir-gpu): wire TurboQuant GPU attention scoring (Plan 048 T5)`
6. `feat(riir-gpu): add feedback consumer for TTT retraining loop (Plan 048 T6)`
7. `docs: update research audit and training pipeline docs (Plan 048 T8)`

---

## Tasks

- [x] **Task 1:** Fix attention backward propagation in `backward.rs` (~110 lines)
- [x] **Task 2:** Implement real KL divergence in `distill.rs` (~60 lines)
- [x] **Task 3:** Implement game replay parser in `game/replay.rs` (~45 lines)
- [x] **Task 4:** Wire PFlash GPU dispatch in `kernels/mod.rs` + `forward.rs` (~230 lines)
- [x] **Task 5:** Wire TurboQuant GPU attention scoring in `kernels/mod.rs` + `forward.rs` (~90 lines)
- [x] **Task 6:** Add feedback consumer for TTT retraining loop (~210 lines)
- [x] **Task 7:** E2E validation suite — all 6 fixes proven working
- [x] **Task 8:** Update README, docs, create research audit doc
- [x] **Task 9:** Commit with conventional messages per task

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │      RESEARCH AUDIT          │
                        │   21 papers → 4 repos        │
                        │   90% implemented, 5 gaps    │
                        └──────────┬──────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────────┐
          │                        │                            │
    ┌─────▼─────┐          ┌──────▼──────┐            ┌────────▼───────┐
    │  PHASE 1  │          │  PHASE 2    │            │   PHASE 3      │
    │  Fix Bugs │          │  Wire GPU   │            │  Close Loop    │
    └─────┬─────┘          └──────┬──────┘            └────────┬───────┘
          │                       │                            │
   ┌──────┼──────┐        ┌──────┼──────┐            ┌────────▼───────┐
   │      │      │        │      │      │            │  feedback_     │
   │ attn │ KL   │        │PFlash│ TQ   │            │  consumer.rs   │
   │ bwd  │ div  │        │ GPU  │ GPU  │            │                │
   │      │      │        │      │      │            │  poll → train  │
   │ replay│     │        │      │      │            │  → hot-swap    │
   │ parse │     │        │      │      │            │                │
   └──┬───┴──┬───┘        └──┬───┴──┬───┘            └───┬────────────┘
      │      │               │      │                    │
      ▼      ▼               ▼      ▼                    ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    riir-gpu (training pipeline)               │
  │                                                               │
  │  forward.rs ──▶ [PFlash GPU] ──▶ attention ──▶ [TQ GPU]     │
  │  backward.rs ──▶ [FIXED attn grad] ──▶ LoRA grads           │
  │  distill.rs ──▶ [REAL KL divergence] ──▶ quality metric     │
  │  game/replay.rs ──▶ [REAL parse_replay] ──▶ training data   │
  │  feedback_consumer.rs ──▶ poll → retrain → hot-swap signal   │
  └───────────────────────────────────────────────────────────────┘
```

## File Change Summary

### Modified files (riir-ai)

| File | Change | Lines |
|------|--------|-------|
| `riir-gpu/src/backward.rs` | Fix attention gradient computation, re-enable test | ~110 |
| `riir-gpu/src/distill.rs` | Replace KL placeholder with real computation | ~60 |
| `riir-gpu/src/game/replay.rs` | Implement `parse_replay()` from stub | ~45 |
| `riir-gpu/src/kernels/mod.rs` | Add 5 new pipelines (4 PFlash + 1 TQ) | ~30 |
| `riir-gpu/src/forward.rs` | Add PFlash dispatch path, TQ scoring path | ~180 |
| `riir-gpu/Cargo.toml` | Add `feedback-consumer` feature | ~5 |
| `riir-ai/README.md` | Update training status, add new sections | ~30 |

### New files (riir-ai)

| File | Purpose | Lines |
|------|---------|-------|
| `riir-gpu/src/feedback_consumer.rs` | TTT retraining consumer | ~150 |
| `riir-gpu/examples/feedback_consumer.rs` | CLI for running consumer | ~40 |
| `riir-ai/.docs/13_research_audit_results.md` | Full audit report | ~120 |

### Modified files (katgpt-rs)

| File | Change | Lines |
|------|--------|-------|
| `README.md` | Add self-improving loop section | ~10 |

---

## Design Decisions

### 1. CPU KL divergence (not GPU)

Distillation is an offline operation run once per training cycle. CPU KL divergence is simpler and avoids adding another WGSL kernel. The bottleneck is forward passes, not KL computation. If profiling shows KL is slow, we can GPU-accelerate later.

### 2. Feature-gate feedback consumer

`feedback-consumer` is a new subsystem. Feature-gate it to avoid adding dependencies (HTTP client, polling) to the core training pipeline. Default: off.

### 3. PFlash GPU dispatch alongside CPU

Both paths coexist. `GpuFlashPrefillPass` is optional — if GPU context is available, use GPU; otherwise fall back to existing CPU path in `katgpt-rs/src/speculative/prefill.rs`. No behavior change without explicit opt-in.

### 4. Fix backward, don't rewrite

The backward pass architecture is sound. Only the attention gradient path needs fixing. We don't rewrite the entire backward pass — we correct the specific gradient computation for the attention score → Q/K/V path.

### 5. Replay parser uses existing JSONL infrastructure

`parse_jsonl()` and `parse_jsonl_filtered()` already exist in `game/replay.rs`. `parse_replay()` just needs to call them and map to `GameSample`. No new parsing code.

---

## Priority Order

| Priority | Task | Why | Effort | Repo |
|----------|------|-----|--------|------|
| P0 | Task 1: Fix attention backward | Correctness — wrong gradients = wrong training | Medium | riir-ai |
| P0 | Task 3: Implement replay parser | Unblocks real game training data | Small | riir-ai |
| P1 | Task 2: Real KL divergence | Quality visibility — can't measure distillation | Small | riir-ai |
| P1 | Task 4: Wire PFlash GPU | 90% done — shaders exist, just needs dispatch | Medium | riir-ai |
| P2 | Task 5: Wire TQ GPU | Enhancement — CPU path works, GPU is faster | Small | riir-ai |
| P2 | Task 6: Feedback consumer | Closes the self-improving loop | Medium | riir-ai |
| P3 | Task 7: E2E validation | Prove everything works together | Small | both |
| P3 | Task 8: Documentation | Record audit results | Small | both |
| P3 | Task 9: Commit | Clean git history | Trivial | both |

---

## Connection to Existing Plans & Research

| Item | Relationship |
|------|-------------|
| **Research 19 (TTT-Discover)** | Task 6 closes the feedback loop — observe → reward → retrain → deploy |
| **Research 04 (LoRA Architecture)** | Tasks 1-2 ensure training produces correct, measurable LoRA adapters |
| **Research 20 (TurboQuant)** | Task 5 wires GPU kernel for production KV cache compression |
| **Research 00 (PFlash)** | Task 4 wires GPU shaders for block-sparse prefill acceleration |
| **Plan 040 (Cross-Training)** | Provides BetaConfig, ReviewMetrics, CompressReport used in Tasks 1-2 |
| **Plan 041 (E2E Game Training)** | Task 3 unblocks real game training data from replay files |
| **Plan 042 (TTT Feedback)** | Task 6 implements the receive side — Plan 042 was send only |
| **Plan 043 (TurboQuant)** | Task 5 wires GPU path — Plan 043 was CPU only |
| **Plan 044 (PFlash)** | Task 4 wires GPU dispatch — Plan 044 wrote shaders but didn't connect |

---

## Expected Outcomes

1. **Training correctness verified** — GPU LoRA gradients match CPU reference within tolerance
2. **Distillation quality measured** — KL divergence is a real number, not 0.0 placeholder
3. **Game training on real data** — `parse_replay()` produces real samples from bomber replays
4. **GPU-accelerated prefill** — PFlash runs on GPU for long-context prompts (target: 2-5× faster than CPU)
5. **GPU-accelerated KV compression** — TurboQuant scoring on GPU for production inference
6. **Self-improving loop operational** — inference → feedback → retrain → hot-swap cycle works E2E
7. **Research audit documented** — 21 papers mapped to implementations, gaps recorded, fixes proven

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Attention backward fix is complex (softmax gradient through multi-head) | CPU reference exists for numerical verification; fix incrementally with per-layer tests |
| PFlash GPU dispatch may not match CPU exactly (float precision) | Tolerance-based comparison (1e-4 relative); same approach as existing GPU loss tests |
| Feedback consumer adds HTTP dependency to riir-gpu | Feature-gated; no dependency without explicit opt-in |
| Replay format may have changed since Plan 039 | Verify with actual `bomber_04_replay_gen` output before implementing parser |
| All changes in one plan is risky | Each task is independent — can commit and test individually |

---

## Research Citations

```bibtex
@article{yuksekgonul2026tttdiscover,
  title   = {Learning to Discover at Test Time},
  author  = {Yuksekgonul, Mert and others},
  journal = {arXiv preprint arXiv:2601.16175},
  year    = {2026}
}

@article{hu2022lora,
  title   = {LoRA: Low-Rank Adaptation of Large Language Models},
  author  = {Hu, Edward J and others},
  journal = {ICLR},
  year    = {2022}
}

@article{zandieh2025turboquant,
  title   = {TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate},
  author  = {Zandieh, Amir and others},
  year    = {2025}
}

@article{leviathan2022fast,
  title   = {Fast Inference from Transformers via Speculative Decoding},
  author  = {Leviathan, Yaniv and others},
  year    = {2022}
}