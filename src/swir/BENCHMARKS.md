# SwiR Switch-Thinking — Phase 3 GOAT Gate Results

**Plan:** 275 (Phase 3, tasks T3.2–T3.11)
**Test file:** `tests/bench_275_swir_goat.rs` (10 tests)
**Benchmark report:** `.benchmarks/275_swir_switch_thinking.md`
**Profile:** release (G3 enforced) + debug (G7 allocation audit)
**Hardware:** Apple Silicon arm64 (NEON SIMD), Rust 1.93.0
**Model dependency:** None — all gates run on synthetic entropy streams + synthetic embedding matrices. Real-model gates (G1 accuracy, G2 efficiency, T3.9 ablations) are deferred to riir-ai Plan 299 (NPC Curiosity Self-Play Runtime), which has the model loader + MATH500 harness.

## Reproduce

```bash
# All synthetic gates (debug — fast, includes allocation audit)
cargo test --features swir_switch_thinking --test bench_275_swir_goat -- --nocapture --test-threads=1

# G3 perf gate (release — the actual 200ns budget is release-mode)
cargo test --release --features swir_switch_thinking --test bench_275_swir_goat g3_step_perf -- --nocapture

# G5 isolation: swir code must NOT compile-link when feature is off
cargo check --no-default-features --features thinking_cot
```

## Gate verdicts

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1c** controller correctness | Latent→Explicit switches, convergence at ½c_max, termination above c_max | 6 switches, 3 CloseThink, 1 ForceAnswerPrefix, terminated at step 21 | ✅ PASS |
| **G2p** efficiency proxy | SwiR terminates < ½ fixed budget on switching schedule | 33 steps vs 1024 = **31.03× fewer steps** (97% reduction) | ✅ PASS |
| **G3** step() perf | < 200 ns/call (release) | **3.1 ns/call** (release), 28.0 ns (debug) | ✅ PASS (64× margin) |
| **G4** convex hull | 1000 random probs all in vocab hull | 1000/1000 in hull (100.00%) | ✅ PASS |
| **G5** feature isolation | swir code absent without feature | `cargo check --no-default-features --features thinking_cot` clean | ✅ PASS |
| **G6** kurtosis auto-fallback | High kurtosis forces Explicit mode | kurtosis=5.0 > threshold=3.0 → forced Explicit | ✅ PASS |
| **G7** zero-alloc step() | 0 allocations in `step()` (debug) | 0 allocs, 0 bytes over 1023 steps | ✅ PASS |
| **G8** signal-mix schedule | α_t/β_t monotonic non-decreasing in step_index | [0.70, 0.72, 0.74, 0.78, 0.85, 0.93, 1.0] — monotonic ✓ | ✅ PASS |

**All 8 synthetic-data gates PASS.**

### Deferred to riir-ai Plan 299 (needs real model)

| Gate | Target | Why deferred |
|------|--------|--------------|
| **G1** accuracy | +1.5pp on MATH500 vs `thinking_cot` baseline | Needs a real model (Gemma 2 / Qwen3) + MATH500 dataset + inference loop. katgpt-rs is the public MIT engine — no model loader. riir-ai has the runtime. |
| **G2** efficiency | 1.3× token efficiency at fixed accuracy | Same — needs real decoding to measure actual token counts at matched accuracy. |
| **T3.9** ablations | W_E→L, α_0, C_max, signal mix sweeps | Same — accuracy ablations need a real task. |

## Algorithmic bug fixed during Phase 3

**Switch-count guard livelock** (fixed in `src/swir/controller.rs`):

The original code enqueued `CloseThink`/`ForceAnswerPrefix` on *every step* while in Explicit mode with switch_count in the convergence window. This caused a livelock: the enqueued token was drained at the start of the next step (skipping mode-switch logic), which prevented switch_count from advancing, which re-enqueued the same token forever.

**Fix:** only enqueue when `switched_to == Some(ThinkMode::Explicit)` — i.e., on the step where the Latent→Explicit switch *just happened*. This matches the paper's intent (§3.4 describes switch-count thresholds, not continuous conditions) and fires each guard exactly once per switch event.

Before fix: G2p ran 1024 steps without terminating (0% reduction).
After fix: G2p terminates at step 33 (97% reduction, 31.03× speedup) with the REAL `c_convergence_fraction=0.5` (no workaround).

## G7 adapter allocation note

The `SwiRStrategyAdapter::on_step` path allocates ~2× per step in debug builds:
- 1 clone of the soft-embedding result (embedding_dim × 4 bytes = 128 bytes for dim=32)
- 1 allocation tracked internally by the allocator harness

This is documented in Plan 275 T2.2 ("soft-embedding clone + InjectTokens Vec"). The `step()` path itself (G7 step variant) is zero-allocation. The adapter allocations are bounded and acceptable for the thinking-cot integration; a future optimization could pass a scratch buffer if the adapter becomes a measured hot path.

## Decision

**Keep `swir_switch_thinking` OPT-IN** (default-off) until riir-ai Plan 299 proves G1/G2 on a real model. The algorithmic invariants (G3–G8, G1c, G2p) all pass on synthetic data — the controller is correct by construction. The missing piece is empirical proof on a real reasoning task, which is riir-ai's mandate.
