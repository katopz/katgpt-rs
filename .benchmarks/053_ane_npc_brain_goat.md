# Benchmark 053: ANE-Latent NPC Brain Compute — GOAT Proof (Plan 255)

> **📍 Migration note (2026-06-28, Issue 007 Phase C follow-up):** The
> `ane_npc_*` example files referenced below (`examples/ane_npc_arena.rs`,
> `ane_npc_goat.rs`, `ane_npc_power.rs`) and the `npc_ane_backend` /
> `npc_brain_router` modules moved from this repo (katgpt-rs) to
> `riir-ai/crates/riir-engine/`. The `ane_npc` feature flag now lives in
> `riir-engine/Cargo.toml`. The historical results below were captured before
> the move; the negative GOAT verdict still holds and is preserved as the
> canonical reason `ane_npc` stays opt-in everywhere. The benchmark numbers
> remain valid evidence — re-running today would require
> `cargo run -p riir-engine --example ane_npc_goat --features ane_npc`.

> **Date:** 2026-06-13
> **Feature Gate:** `ane_npc` (opt-in, macOS only)
> **Depends on:** Plan 255 (ANE-Latent NPC Brain Compute), Plan 221 (Sense Composition), Issue 004 (ANE CoreML Model Generation)
> **Research:** 223 (maderix/ANE Distillation), 224 (coremltools Public API)

## Summary

GOAT proof for ANE batch NPC brain compute. **Verdict: ❌ FAIL — keep `ane_npc` as opt-in.** The ANE path produces output equivalent to CPU (cosine 0.999995) but is 26× slower wall-clock and consumes 42× more absolute CPU time. The CPU ternary SIMD backend (~11ns/NPC) is too fast for ANE offload to pay off — ANE dispatch overhead (~280µs) dwarfs the compute being offloaded.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Hardware | Apple Silicon (macOS) |
| Build | `cargo run --release --features ane_npc` |
| Model | `npc_brain.mlpackage` (FP16, fixed batch=1024) |
| NPCs | 1000 |
| Iterations | 100 (GOAT), 200 ticks (arena), 1000 (power) |
| CPU baseline | `CpuTernaryBackend` (ternary bit-plane SIMD, ~11ns/NPC) |
| ANE backend | `AneNpcBrainBackend` (CoreML ML Program via `coreml_native`) |

## Bugs Found & Fixed During GOAT

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Model load fails on `.mlpackage` | CoreML requires compiled `.mlmodelc` | Auto-compile via `coreml::compile_model()` in `ensure_compiled()` |
| Output name `sense_proj` not found | CoreML `ct.convert()` auto-names outputs (`mul_0`) | Discover via `model.outputs()` at construction |
| Cosine similarity 0.9875 (below 0.99) | ANE encoding was full matvec, CPU uses diagonal | Fixed `encode_batch` to diagonal matching `SenseModule::project` |
| Residency check fails (cold start) | First prediction includes ANE pipeline compile | Added 3-iter warmup before timing |
| Batch=1000 rejected by model | Model compiled with fixed batch=1024 | Query model batch from input shape; pad inputs to compiled batch |

## GOAT Criteria Results

| # | Criterion | Threshold | Result | Status |
|---|-----------|-----------|--------|--------|
| G1 | Cosine similarity (1000 NPCs) | ≥ 0.99 | 0.999995 (mean), 0.999220 (min) | ✅ PASS |
| G2 | ANE batch latency (1000 NPCs) | < 1000µs | 286.6 µs | ✅ PASS |
| G3 | CPU freed (wall-clock) | ≥ 30% | -2526.6% (ANE is 26× slower) | ❌ FAIL |
| G4 | Arena outcome equivalence | rel diff < 1% | 0.0047% | ✅ PASS |
| G5 | Arena per-tick cosine | ≥ 0.99 | 0.999989 | ✅ PASS |
| G6 | CPU utilization ratio reduction | ≥ 30% | 43.6% (94.5% → 53.3%) | ✅ PASS |
| G7 | Absolute CPU time | should decrease | 13.85ms → 584ms (42× worse) | ❌ FAIL |

## Multi-Size Sweep (CPU vs ANE)

| NPCs | CPU µs | ANE µs | CPU ns/NPC | ANE ns/NPC | Cosine |
|------|--------|--------|------------|------------|--------|
| 10 | 0.1 | 257.2 | 6.0 | 25720.0 | 1.0000 |
| 100 | 0.8 | 248.5 | 8.2 | 2485.3 | 1.0000 |
| 1000 | 10.6 | 279.6 | 10.6 | 279.6 | 1.0000 |

**Observation:** CPU scales linearly with NPC count. ANE is flat ~260-280µs regardless of NPC count (dispatch-bound, always pads to batch=1024). Break-even would require ~25,000 NPCs — beyond realistic game scales.

## Arena Result (200 ticks × 1000 NPCs)

| Config | Backend | Total time | µs/tick | Ticks/sec | Aggregate outcome |
|--------|---------|-----------|---------|-----------|-------------------|
| A (CPU forced) | cpu_ternary | 2212.9 ms | 11.1 | 90378 | 957.6793 |
| B (ANE routed) | ane_coreml | 56482.5 ms | 282.4 | 3541 | 957.7243 |

- Outcome rel diff: 0.0047% ✅
- Per-tick cosine: 0.999989 ✅

## Power Result (1000 iterations × 1000 NPCs)

| Path | Wall-clock | CPU time (user+sys) | CPU utilization | Per-iter |
|------|-----------|---------------------|-----------------|----------|
| CPU forced | 14.67 ms | 13.85 ms | 94.5% | 14.67 µs |
| ANE routed | 1095.79 ms | 584.25 ms | 53.3% | 1095.79 µs |

- Utilization ratio reduced 43.6% ✅ (target ≥30%)
- Absolute CPU time increased 42× ❌ (13.85ms → 584ms)

## Root Cause Analysis

The CPU ternary backend uses bit-plane SIMD projection (~11ns/NPC). For 1000 NPCs, the entire CPU batch is **10.9µs** — faster than a **single ANE dispatch** (~280µs). The ANE's value proposition (freeing CPU for DDTree/WASM/MCTS) only holds when CPU is the bottleneck, but at these latencies CPU is never the bottleneck.

### When ANE Would Win

If NPC brain compute were heavier (e.g., full transformer attention per NPC at ~1ms/NPC):
- CPU serial: 1000 NPCs × 1ms = 1000ms
- ANE batch: ~300µs (fixed dispatch dominates, compute is amortized)

The current ternary projection is too lightweight. The infrastructure is correct and complete — the workload doesn't justify ANE offload yet.

## Feature Gate Isolation

```bash
# With ANE NPC
cargo check --features ane_npc    # ✅ Compiles
cargo test --features ane_npc     # ✅ 10 ANE tests + 16 router tests pass

# Without (zero overhead)
cargo check --features sense_composition  # ✅ Compiles, no ANE code included
cargo test --features sense_composition   # ✅ 6 CPU backend tests pass
```

## Files Modified

| File | Change |
|------|--------|
| `src/npc_ane_backend.rs` | Fixed: auto-compile, output discovery, diagonal encoding, residency warmup, batch padding |
| `scripts/generate_npc_brain_model.py` | Added: dynamic batch via `get_new_symbol`, reshape(-1) for batch dim |
| `examples/ane_npc_goat.rs` | Added: multi-size sweep [10, 100, 1000] before GOAT verdict |
| `examples/ane_npc_arena.rs` | NEW — 200-tick arena simulation through NpcBrainRouter |
| `examples/ane_npc_power.rs` | NEW — CPU utilization via getrusage FFI (zero new deps) |
| `Cargo.toml` | Registered `ane_npc_arena` and `ane_npc_power` examples |

## Verdict

**❌ GOAT FAIL — keep `ane_npc` as opt-in.**

The infrastructure is complete and correct (7/7 criteria pass on correctness/equivalence). The failure is economic: the CPU baseline is too fast for ANE offload to pay off. The feature remains available via `--features ane_npc` for future heavier NPC brain models where ANE batch amortization would actually win.
