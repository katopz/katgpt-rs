# Plan 182: Trust-Region Adaptive Speculation (TRAS)

> **Research:** 162 (Trust-Region Adaptive Speculation)
> **Status:** ✅ Complete — GOAT T6 acceptance ≥15% — Default ON
> **Feature gate:** `trust_region_spec` — depends on `speculative`, optionally `bandit`, `inference_router`, `thinking_cot`
> **Default-on:** ✅ Yes — GOAT proof passed (≥15% acceptance improvement), promoted to default
> **Commercial alignment:** Per Verdict 003 — modelless inference trust in MIT engine (katgpt-rs), LoRA training trust in private SaaS (riir-ai)

---

## Summary

Extend speculative decoding with TrOPD's trust region and TRB's behavior blending at inference time. The trust signal (P_accept = min(πT/πS, 1)) — already computed by `LeviathanVerifier` — drives:

1. **Adaptive speculation window** (high trust → batch accept, low trust → verify every token)
2. **TRB-style blend on rejection** (sample from μ_β = πS^(1-β)·πT^β instead of pure reject)
3. **Bandit-driven CPU/GPU routing** (low trust → CoT/GPU, high trust → direct/CPU)
4. **Freeze/thaw persistence** of learned trust patterns

Zero cost when disabled. After GOAT proof, must be on by default.

---

## Architecture

```mermaid
graph TD
    DRAFT[Draft from student] --> TRUST[Compute P_trust]
    TRUST -->|P >= tau| ACCEPT[Accept — fast path]
    TRUST -->|P < tau| BLEND[Blend: mu_beta sample]
    ACCEPT --> TRACK[Track trust per query]
    BLEND --> TRACK
    TRACK --> WIN[Adaptive window: resize based on trust]
    TRACK --> ROUTE[Route: CPU fast vs GPU think]
    TRACK --> FREEZE[Freeze/thaw: persist trust patterns]
    WIN --> BANDIT[Bandit learns per-domain]
    ROUTE --> BANDIT
    FREEZE --> BANDIT
```

---

## Tasks

- [x] T1: `TrustRegionVerifier` Trait — Extension Point
- [x] Add `trust_metric(&self) -> f32` — running average of P_accept
- [x] Add `adaptive_window(&self, base: usize) -> usize` — expand/shrink based on trust
- [x] Add `blend_sample(&mut self, beta: f32, rng: &mut Rng) -> usize` — TRB μ_β sampling
- [x] Feature-gate behind `trust_region_spec`

- [x] T2: `TrustRegionLeviathanVerifier` — Implementation
- [x] Track running acceptance rate per decode call (sliding window of 16 tokens)
- [x] Adaptive window: trust > 0.85 → base_window × 1.5, trust < 0.5 → window = 1
- [x] Blend on rejection: compute β via binary search (10 iterations max), sample from μ_β
- [x] Zero additional allocation: reuse existing `SpeculativeContext` buffers

### T3: Trust Signal → InferenceRouter Integration

- [x] Add `trust_signal: f32` to router state, updated from verifier
- [x] Low trust (< 0.4) triggers tier-up: CPU → GPU (if available)
- [x] High trust (> 0.8) allows tier-down: GPU → CPU (if load permits)
- [x] Wire through `forward()` method
- [x] Log trust-triggered tier transitions

### T4: Trust Signal → ThinkingController Integration

- [x] Trust metric as additional signal for think/direct decision
- [x] Low trust → prefer thinking mode (PPoT resample or RiM buffer)
- [x] High trust → prefer direct mode (skip thinking)
- [x] Combine with existing entropy and bandit signals

- [x] T5: Bandit Learning for Trust Patterns
- [x] Reward: successful decode (tokens accepted without quality regression)
- [x] Freeze/thaw: persist trust-bandit knowledge per domain
- [x] Self-improving: bandit adapts trust thresholds per query type

- [x] T6: Test — Before/After Trust-Region Speculation
- [x] Test 2: TRAS adaptive window — measure acceptance rate + output quality
- [x] Assert: TRAS acceptance rate ≥ 15% higher than baseline
- [x] Assert: Output quality (valid sequences) not regressed
- [x] Print before/after comparison table

### T7: Bench — Micro-benchmark Blend Cost

- [x] Benchmark blend computation: πS^(1-β)·πT^β for vocab_size tokens
- [x] Benchmark binary search for β: 10 iterations over KL computation
- [x] Assert: blend cost < 2μs (acceptable in speculative decode hot path)
- [x] Compare: total speculative decode time with and without TRAS

---

## Expected Performance

| Metric | Before TRAS | After TRAS | Notes |
|--------|------------|------------|-------|
| Acceptance rate (converged) | ~70% | ~85% | Adaptive window + blend on rejection |
| Verification cost per query | 100% | 70-80% | Window expansion on high-trust tokens |
| Quality on hard queries | Baseline | +3-6 pts | Blend ensures teacher guidance in outlier regions |
| CPU/GPU routing | Load-based | Trust + load | Principled routing metric |
| Overhead when disabled | 0 | 0 | Feature-gated, same binary |

---

## Feature Gate

```toml
[features]
trust_region_spec = ["speculative", "bandit"]
```

GOAT proof passed (T6 acceptance ≥15%). Promoted to default features in Cargo.toml.

---

## Dependencies

- Plan 194 (Adaptive CoT) — trust signal integrates with ThinkingController
- Plan 131 (SpecHop) — TRAS complements SpecHop: SpecHop does continuous speculation, TRAS adapts the window
- Plan 176 (TriggerGate) — trust signal feeds tier routing
- Research 162 — this plan's research basis
