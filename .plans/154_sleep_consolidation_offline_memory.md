# Plan 154: Sleep Consolidation — Offline Recursive Memory Consolidation at Eviction

> **Research:** [116 — LLM Sleep: Offline Recursive Memory Consolidation](../.research/116_LLM_Sleep_Offline_Recursive_Memory_Consolidation.md)
> **Paper:** [arXiv:2605.26099](https://arxiv.org/abs/2605.26099) — Lee et al., May 2026
> **Feature Gate:** `sleep_consolidation` (depends on `lt2_looped`, `gdn2_attention`)
> **Priority:** MEDIUM — Infrastructure improvement, not blocking any GOAT pillar
> **Status:** ✅ Infrastructure complete — pending GOAT proof (T10–T13)

## Summary

Implement sleep-time consolidation: when KV cache fills, perform N offline recurrent passes to consolidate context into GDN2 fast weights, then evict. Preserves single-pass wake-time latency for real-time game constraints (20Hz frame sampling).

Key insight: Sleep moves LT2's wake-time looping to eviction-time consolidation. This is the model-based analog of our modelless AutoDreamer (Plan 107), applied to GDN2 fast weights.

---

## Tasks

- [x] T1: Add `sleep_consolidation` feature gate to `katgpt-rs/Cargo.toml` (depends on `lt2_looped`, `gdn2_attention`)
- [x] T2: Create `src/sleep/` module scaffold (`mod.rs`, `types.rs`)
- [x] T3: Implement `SleepConfig` { sleep_passes: usize, eviction: EvictionStrategy, window_size: usize }
- [x] T4: Implement `EvictionStrategy` enum { HardEvict, SlidingWindow }
- [x] T5: Implement `consolidation_pass()` — single recurrent forward pass through all layers, carrying GDN2 fast-weight state
- [x] T6: Implement `sleep()` — N calls to `consolidation_pass()` at eviction boundary
- [x] T7: Implement `eviction::HardEvict` — clear entire KV cache after sleep
- [x] T8: Implement `eviction::SlidingWindow` — retain last L-1 tokens, evict older
- [x] T9: Integrate sleep hook into LT2 forward pass (Plan 108) at eviction boundary
- [ ] T10: GOAT proof — sleep (N=2,4) vs no-sleep on multi-hop reasoning (synthetic graph task)
- [ ] T11: GOAT proof — sleep + TurboQuant hybrid vs TurboQuant-only on long-context task
- [ ] T12: GOAT proof — sleep on game context (long Bomber session >2000 tokens, long NPC dialog)
- [ ] T13: Benchmark — sleep overhead (N=2,4,6) vs no-sleep vs LT2 wake-time (tok/s, µs/step)
- [x] T14: Update README + .docs with sleep consolidation section

---

## Context

### Why Sleep?

Our LT2 (Plan 108) loops at wake time — good for quality, bad for latency. Our real-time game loop (Pillar 4) needs ≤50ms per tick at 20Hz. Sleep moves loops to eviction time:
- Wake time: single-pass (≤50ms budget preserved)
- Sleep time: N recurrent passes (offline, no latency constraint)

### Architecture Fit

```
Existing LT2 Pipeline:
  Input → [SDPA → GDN2 → SDPA → GDN2 → ...]×T (wake-time loops) → Output
  
With Sleep:
  Input → Context fills → [SDPA → GDN2 → ...]×N (sleep-time consolidation) → Evict KV → Continue
         ↑ Single-pass at wake time (T=1)                    ↑ N-pass at eviction boundary
```

### Integration Points

| Component | Change | Scope |
|-----------|--------|-------|
| `transformer.rs` | Add sleep hook at eviction boundary | `lt2_looped` + `sleep_consolidation` |
| `gdn2_recurrent_step` | Fast-weight state carries across sleep passes | Already supported |
| `kv_cache` | Eviction after sleep | New `eviction.rs` |
| `Config` | Add `SleepConfig` field | Behind feature gate |

---

## Feature Gate

```toml
[features]
sleep_consolidation = ["lt2_looped", "gdn2_attention"]
```

- Requires LT2 loop infrastructure (weight sharing, residual gates)
- Requires GDN2 attention (fast-weight memory blocks)
- NOT default-on until GOAT proof passes

---

## GOAT Proof Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Multi-hop accuracy | ≥15% improvement over no-sleep at 8-hop | Paper shows 30-47% on hardest tasks |
| Long-context quality | ≥5% improvement at 4× window length | Paper shows 9-10% on GSM-Infinite 6-op |
| Wake-time latency | ≤5% increase over single-pass | Sleep is offline; wake stays single-pass |
| Game context | ≥10% improvement on >2000-token game session | Game-specific validation |

---

## Module Structure

```text
src/sleep/
├── mod.rs              # Index, re-exports
├── types.rs            # SleepConfig, EvictionStrategy
├── consolidation.rs    # N-pass recurrent consolidation loop
├── eviction.rs         # Hard/sliding-window eviction after sleep
└── training.rs         # BPTT through sleep (future, requires riir-ai training)
```

---

## Dependencies

- Plan 108 (LT2) — ✅ Complete (11/11 GOAT)
- Plan 105 (GDN2) — ✅ Complete (14/14 GOAT)
- Plan 107 (AutoDreamer) — ✅ Complete (8/8 GOAT) — modelless consolidation complement
- Plan 092 (Freeze/Thaw) — ✅ Complete — context→weights pipeline

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| GOAT proof fails (no gain over compression) | Medium | Fallback to TurboQuant/SpectralQuant. Sleep was free to try. |
| Training infrastructure not ready | High | Implement inference-only sleep first. Training later in riir-ai. |
| GDN2 channel-wise gating interacts poorly | Low | Paper confirms GDN is most stable mixer for sleep. |
| Feature gate explosion | Low | Single `sleep_consolidation` gate composes with existing `lt2_looped`. |

---

## References

- Research 116: LLM Sleep — detailed distillation and analysis
- Paper: https://arxiv.org/abs/2605.26099
- Related: Research 070 (GDN2), Research 073 (LT2), Research 069 (AutoDreamer)
