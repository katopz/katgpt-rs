# Plan 176: Runtime GPU/ANE Offload with Trigger Gate

**Source:** [Research 155 — ANE Compute Backend Verdict](../.research/155_ANE_Compute_Backend_Verdict.md)
**Related:** [Plan 197 — ANE Inference Backend (riir-ai)](../../riir-ai/.plans/197_ane_inference_backend.md)
**Status:** Active — rewriting from .mlmodelc model-loading to runtime weight compilation
**Goal:** Survive 30K CCU by offloading transformer forward to GPU/ANE when load demands it. CPU is NOT enough — it also runs WASM, DDTree, bandit, MCTS. GPU/ANE sit idle is a crime.

---

## Why This Plan Exists

### The 30K CCU Math

```
30K CCU × 20Hz frame sampling = 600K inferences/sec
```

CPU single-thread: 1.65 µs/token → ~600K tokens/sec. Barely one thread's worth.
But CPU also runs: WASM validation, DDTree tree search, bandit pruning, MCTS rollout, ConstraintPruner.
CPU cannot do forward + validation + tree search + prune simultaneously at 30K CCU.

**GPU and ANE sit idle while CPU chokes. This is the problem.**

### Why the Old Plan Was Wrong

Old Plan 176 assumed `.mlmodelc` file loading + `coreml-native`. That's the architecture for LLM inference (load big model, run many tokens). katgpt-rs is **modelless** — weights are in-memory `TransformerWeights`, generated or LoRA-adapted at runtime. There is no file to load. The model IS the weight struct.

The correct approach: **runtime weight compilation** — take `TransformerWeights`, build Metal/GPU/ANE compute pipelines on-the-fly, hot-swap when LoRA updates.

---

## Architecture: Trigger Gate + Three-Way Compute

```
                    ┌─────────────────────────┐
                    │     TriggerGate          │
                    │  monitors: qps, latency  │
                    │  queue depth, load avg   │
                    └──────┬──────┬──────┬─────┘
                           │      │      │
               ┌───────────┘      │      └───────────┐
               ▼                  ▼                  ▼
        ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
        │  CPU Tier   │   │  GPU Tier   │   │  ANE Tier   │
        │  <1K CCU    │   │  1K-10K CCU │   │  >10K CCU   │
        │  always on  │   │  trigger on │   │  trigger on │
        └─────────────┘   └─────────────┘   └─────────────┘
        CPU forward()     Metal compute     CoreML runtime
        1.65µs/tok        pipeline from     compile from
        SIMD kernels      Transformer-      Transformer-
                          Weights           Weights
```

### Trigger Gate Logic

```
load = queue_depth / target_latency

if load < LOW_THRESHOLD:
    tier = CPU_ONLY          # idle, dev mode, <1K CCU
elif load < HIGH_THRESHOLD:
    tier = CPU + GPU         # medium load, GPU handles forward, CPU handles discrete
else:
    tier = CPU + GPU + ANE   # 30K CCU — ALL hardware engaged

# Hysteresis: tier-down requires load < threshold * 0.7 (avoid thrashing)
# Compilation: on tier-up, compile weights to target device (~2ms for microGPT)
# Hot-swap: LoRA update triggers recompilation of GPU/ANE pipelines
```

### Why Not CPU-Only

| Metric | CPU-Only | CPU+GPU | CPU+GPU+ANE |
|--------|----------|---------|-------------|
| Forward throughput | 600K tok/s | 5M tok/s | 15M tok/s |
| CPU free for DDTree+WASM | 0% (contended) | 80% | 95% |
| 30K CCU survivable | ❌ | ✅ | ✅✅ |
| Power draw | 30W CPU | 30W CPU + 40W GPU | 30W CPU + 40W GPU + 5W ANE |

---

## Prerequisites

- macOS 15+ (Sequoia) for Metal 3 + ANE
- Apple Silicon (M1+)
- `metal` crate for GPU compute pipelines
- `coreml-native` for ANE path (optional, behind feature gate)

---

## Task List

### Part 1: InferenceBackend Trait (Runtime Weight-Based)

- [x] Create `src/inference_backend.rs` with `InferenceBackend` trait
- [x] `CpuBackend` wrapping existing `transformer::forward`
- [x] `auto_backend()` for CPU/ANE auto-route (legacy — will be replaced by TriggerGate)
- [x] Unit tests: CpuBackend matches direct forward
- [ ] Refactor `InferenceBackend` trait to accept `&TransformerWeights` + token + pos directly (remove indirection through `ForwardContext`)
- [x] Add `fn compile(&mut self, weights: &TransformerWeights, config: &Config) -> Result<()>` for runtime weight compilation
- [x] Add `fn is_compiled(&self) -> bool` to check if backend has valid compiled weights
- [x] Add `fn recompile_hint(&mut self)` — called when LoRA weights change

### Part 2: GPU Backend via Metal Compute

- [ ] Add `metal = { version = "0.31", optional = true }` to macOS dependencies
- [ ] Add `gpu_inference = ["dep:metal"]` feature flag
- [ ] Create `src/gpu_inference_backend.rs` behind `#[cfg(all(target_os = "macos", feature = "gpu_inference"))]`
- [ ] Implement `GpuInferenceBackend`:
  - [ ] `compile()`: take `TransformerWeights`, build Metal compute pipeline for matmul + attention + FFN
  - [ ] `forward()`: dispatch to GPU, wait for completion, return logits
  - [ ] Use Metal command buffer + compute encoder for kernel dispatch
  - [ ] Map `TransformerWeights` fields → Metal buffers (zero-copy via `new_with_bytes`)
  - [ ] Write Metal shaders for: RMSNorm, QK matmul, softmax, V attention, FFN, output projection
- [ ] Handle batch inference: multiple tokens in single GPU dispatch (amortize kernel launch overhead)
- [ ] Benchmark: CPU forward vs GPU forward for microGPT (1-layer, 16-dim)
- [ ] Benchmark: CPU forward vs GPU forward for game LoRA scale (4-layer, 64-dim)
- [ ] Test: GPU forward produces numerically equivalent logits (cosine sim ≥ 0.999)

### Part 3: ANE Backend via Runtime CoreML Compilation

- [x] Keep `coreml-native = { version: "0.2", optional = true }` dependency
- [x] Refactor `AneBackend` from `.mlmodelc` loader → runtime weight compiler:
  - [ ] `compile()`: build `MLModel` from `TransformerWeights` programmatically (blocked on coreml-native API)
  - [ ] Use `coreml_native::Model::from_spec()` or equivalent runtime API
  - [ ] Map transformer layers → CoreML neural network operations
  - [ ] Conv2d(1×1) trick for linear layers (ANE-friendly)
  - [ ] Set compute units to `.All` and verify ANE placement via residency check
- [ ] `forward()`: predict with compiled model, extract logits from output (blocked on compile)
- [x] Hot-swap: `recompile_hint()` rebuilds CoreML model when LoRA weights change
- [ ] Residency validation: time micro-prediction, verify <1ms (ANE) vs >5ms (CPU fallback)
- [x] Test: residency error messages validated
- [ ] Test: ANE forward produces numerically equivalent logits (cosine sim ≥ 0.997)

### Part 4: Trigger Gate

- [x] Create `src/trigger_gate.rs`
- [x] `TriggerGateConfig` with thresholds:
  ```rust
  pub struct TriggerGateConfig {
      /// Activate GPU when QPS exceeds this (default: 10K inferences/sec)
      pub gpu_activate_qps: f64,
      /// Activate ANE when QPS exceeds this (default: 100K inferences/sec)
      pub ane_activate_qps: f64,
      /// Deactivate tier at threshold * this factor (hysteresis, default: 0.7)
      pub hysteresis_factor: f64,
      /// Queue depth that triggers tier-up (default: 100 pending)
      pub queue_depth_trigger: usize,
      /// Latency P99 that triggers tier-up (default: 5ms)
      pub latency_p99_trigger_us: u64,
      /// Minimum time between tier changes (default: 500ms, avoid thrashing)
      pub min_tier_change_interval_ms: u64,
  }
  ```
- [ ] `TriggerGate` struct:
  - [x] `AtomicU64` counters for QPS, queue depth, latency samples
  - [x] `current_tier(): ComputeTier` — returns active tier
  - [x] `record_inference(duration_us: u64)` — called after each forward pass
  - [x] `record_queue_depth(depth: usize)` — called when submitting to queue
  - [x] `should_promote() -> Option<ComputeTier>` — check if load exceeds next tier threshold
  - [x] `should_demote() -> Option<ComputeTier>` — check if load dropped below threshold × hysteresis
  - [ ] Background thread (or interval check) that evaluates tier changes
- [x] `ComputeTier` enum: `CpuOnly`, `CpuGpu`, `CpuGpuAne`
- [ ] On tier-up: compile weights to new device (~2ms), start routing
- [ ] On tier-down: stop routing to device, release Metal/CoreML resources
- [x] Thread-safe: all counters atomic, tier change behind Mutex
- [x] Test: trigger activates GPU at threshold
- [x] Test: trigger activates ANE at higher threshold
- [x] Test: hysteresis prevents tier thrashing under oscillating load
- [x] Test: tier-down requires load < threshold × hysteresis_factor

### Part 5: InferenceRouter (The Glue)

- [x] Create `src/inference_router.rs`
- [x] `InferenceRouter` struct:
  ```rust
  pub struct InferenceRouter {
      cpu: CpuBackend,
      gpu: Option<GpuInferenceBackend>,
      ane: Option<AneBackend>,
      gate: TriggerGate,
      weights: TransformerWeights,
      config: Config,
  }
  ```
- [x] `fn forward(&mut self, token: usize, pos: usize) -> &[f32]`:
  - Read `gate.current_tier()`
  - Route to highest available tier for this inference
  - Fallback to CPU if GPU/ANE compilation fails
- [x] `fn update_weights(&mut self, weights: TransformerWeights)`:
  - Update CPU weights immediately
  - Set `recompile_hint` on GPU/ANE backends
  - Background recompile on next idle cycle
- [x] `fn stats(&self) -> RouterStats` — QPS per tier, latency histograms, tier transitions
- [x] Batch mode: `fn forward_batch(&mut self, tokens: &[(usize, usize)]) -> Vec<Vec<f32>>` — GPU/ANE shine here
- [x] Test: router starts in CPU-only mode
- [x] Test: router promotes to GPU under simulated load
- [x] Test: router falls back to CPU on GPU compilation failure
- [x] Test: weight update propagates to all active backends

### Part 6: Wire Into Existing Pipeline

- [ ] Add `InferenceRouter` to main inference loop (behind feature gate)
- [x] `--device cpu|gpu|ane|auto|gate` CLI flag (new: `gate` = trigger gate mode)
- [ ] When `--device gate`: use `TriggerGate` + `InferenceRouter`
- [x] When `--device auto/cpu/gpu/ane`: direct backend selection (existing behavior)
- [ ] Log tier transitions: `"TriggerGate: CPU → CPU+GPU (QPS: 12K, queue: 150)"`
- [ ] Expose `TriggerGateConfig` for tuning thresholds per deployment
- [ ] Update bomber/Go arena to support `--device gate` mode

### Part 7: Benchmarks + GOAT Proof

- [x] Bench: single-token CPU latency (1.65 µs — baseline)
- [x] Bench: 50-token CPU generation (2.50 µs/token)
- [x] Bench: backend selection overhead (0.20 µs)
- [ ] Bench: GPU forward latency vs CPU (expect 4-10× faster for batch)
- [ ] Bench: ANE forward latency vs CPU (expect 2-4× faster for single-token)
- [x] Bench: trigger gate overhead (<1µs per inference call)
- [ ] Bench: compilation time from TransformerWeights → Metal/CoreML pipeline
- [ ] Bench: tier-up latency (compilation + first forward)
- [ ] GOAT: GPU forward == CPU forward (cosine ≥ 0.999)
- [ ] GOAT: ANE forward == CPU forward (cosine ≥ 0.997)
- [x] GOAT: trigger gate correctly tier-up at simulated 10K QPS
- [x] GOAT: trigger gate correctly tier-down when load drops
- [ ] GOAT: 30K CCU simulation survives with GPU+ANE, dies with CPU-only

### Part 8: Feature Gates + Cleanup

- [x] `ane = ["dep:coreml-native"]` feature flag
- [x] `gpu_inference = []` feature flag (placeholder — pending metal crate)
- [x] `inference_router = ["gpu_inference", "ane"]` — pulls in everything
- [x] Remove `.mlmodelc` file-loading code from `AneBackend`
- [x] Remove `scripts/convert_to_coreml.py` (no longer needed — runtime compilation)
- [ ] Default: all features off (CPU-only), opt-in GPU/ANE
- [ ] Document trigger gate + three-way compute in README.md

---

## Architecture

```mermaid
graph TD
    subgraph Per Inference Request
        A[Token + Pos] --> B{TriggerGate Tier?}
        B -->|CPU_ONLY| C[CPU: SIMD Forward:::accent1]
        B -->|CPU+GPU| D[GPU: Metal Forward:::accent2]
        B -->|CPU+GPU+ANE| E[ANE: CoreML Forward:::accent0]
    end
    subgraph Background
        F[TriggerGate Monitor] -->|load > 10K QPS| G[Compile Weights → GPU Pipeline]
        F -->|load > 100K QPS| H[Compile Weights → ANE Model]
        F -->|load < threshold| I[Release GPU/ANE Resources]
    end
    subgraph LoRA Update
        J[New LoRA Weights] --> K[recompile_hint]
        K --> G
        K --> H
    end
```

## Trigger Gate Decision Flow

```mermaid
graph TD
    A[Record Inference] --> B{QPS > ane_activate_qps?}
    B -->|yes| C{ANE available?}
    C -->|yes| D[Tier: CPU+GPU+ANE:::accent0]
    C -->|no| E{QPS > gpu_activate_qps?}
    B -->|no| E
    E -->|yes| F{GPU available?}
    F -->|yes| G[Tier: CPU+GPU:::accent2]
    F -->|no| H[Tier: CPU_ONLY:::accent1]
    E -->|no| H
```

## Expected Performance at 30K CCU

| Tier | Throughput | CPU Free | Power | CCU Capacity |
|------|-----------|----------|-------|-------------|
| CPU_ONLY | 600K tok/s | 0% | 30W | ~1K CCU |
| CPU+GPU | 5M tok/s | 80% | 70W | ~10K CCU |
| CPU+GPU+ANE | 15M tok/s | 95% | 75W | **30K+ CCU** |

## Key Crate Dependencies

```toml
[target.'cfg(target_os = "macos")'.dependencies]
metal = { version = "0.31", optional = true }
coreml-native = { version = "0.2", optional = true }

[features]
gpu_inference = ["dep:metal"]
ane = ["dep:coreml-native"]
inference_router = ["gpu_inference", "ane"]
```

## Risks

| Risk | Mitigation |
|------|-----------|
| Metal shader compilation slow | Compile once per weight set, cache pipeline state |
| CoreML runtime compilation API limited | Fall back to GPU tier if ANE compile fails |
| Trigger gate adds latency | All counters atomic, tier check is a single compare-and-swap |
| Tier thrashing under bursty load | Hysteresis + min change interval (500ms) |
| GPU/ANE not available | Always have CPU fallback, gate skips unavailable tiers |
| Small models don't benefit from GPU | Batch multiple inferences into single GPU dispatch |

## Migration from Old Plan 176

| Old | New | Why |
|-----|-----|-----|
| `.mlmodelc` file loading | Runtime weight compilation | katgpt-rs is modelless — no files exist |
| `coreml_native::Model::load(path)` | `coreml_native::Model::from_spec()` | Build model programmatically from TransformerWeights |
| `scripts/convert_to_coreml.py` | DELETE | No conversion pipeline needed |
| Always-on backend selection | Trigger gate with thresholds | Don't waste GPU/ANE at low load, engage at scale |
| Single-backend routing | InferenceRouter with tier promotion | Survive 30K CCU by using ALL available hardware |
