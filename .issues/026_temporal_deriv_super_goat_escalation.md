# Issue 026: Temporal Derivative Kernel — Super-GOAT Escalation (Unified Surprise Bus)

**Date:** 2026-06-16
**Status:** Open — escalation pending separate validation
**Plan:** [277_temporal_derivative_kernel.md](../.plans/277_temporal_derivative_kernel.md)
**Research:** [243_Temporal_Derivative_Kernel_Neocortical_Learning.md](../.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md) §2.5
**Benchmark:** [277_temporal_deriv_goat.md](../.benchmarks/277_temporal_deriv_goat.md)

---

## Context

Plan 277 passed all 4 fusion GOAT gates (G2–G5). The `temporal_deriv` feature is promoted to DEFAULT-ON. Per Plan 277 T6.5, this issue tracks the Super-GOAT escalation for the "unified surprise bus" pattern.

## The Super-GOAT Claim (NOT yet validated)

> A single `TemporalDerivativeKernel` instance with one paper-default α-pair (0.3, 0.03) drives all four consumers (HLA companion, δ-Mem gate, collapse detector, derivative curiosity) without per-consumer tuning.

## Evidence So Far

All four consumers independently chose the same paper-default α-pair:

| Consumer | α_fast | α_slow | N | Decision Basis |
|----------|--------|--------|---|----------------|
| HLA companion | 0.3 | 0.03 | 8 | ReconstructionConfig default |
| δ-Mem gate | 0.3 | 0.03 | 8 | enable_surprise_gate default |
| Collapse detector | 0.3 | 0.03 | 1 | paper-default alphas |
| Derivative curiosity | 0.3 | 0.03 | 64 | DerivativeCuriosity default |

No consumer required per-instance α tuning to pass its gate. The paper's ~10× ratio (0.3 / 0.03) worked universally.

## What's Needed to Claim Super-GOAT

Per AGENTS.md and Plan 277 T6.5, Super-GOAT requires a **separate validation note** (not this issue). The validation must demonstrate:

1. **Single-α universality**: A controlled experiment sweeping α_fast ∈ {0.1, 0.2, 0.3, 0.5, 0.8} × α_slow ∈ {0.01, 0.03, 0.05, 0.1} across all four consumers, showing that the paper-default (0.3, 0.03) is within the Pareto-optimal region for ALL four simultaneously.

2. **Cross-consumer interference test**: When a single NPC runs all four fusions concurrently (HLA + δ-Mem + collapse + curiosity), does the shared α-pair still produce correct behavior? Or does the combined load require per-consumer α adjustment?

3. **Honest failure mode**: Identify at least one scenario where the unified α-pair fails and per-consumer tuning is needed. If no failure exists, document why.

## Why This Matters

If the unified surprise bus validates, it means the neocortical prediction-error signal (O'Reilly 2026) is a **universal primitive** — one α-schedule, four consumers, zero per-consumer tuning. This would be the strongest possible evidence that the distillation captured the essential structure of the biological mechanism.

If it fails (per-consumer tuning is needed), the primitive is still GOAT (4/4 individual gates passed) but not Super-GOAT. Each consumer would document its own recommended α-pair.

## Next Steps

- [ ] Create `.research/244_unified_surprise_bus_validation.md` with the controlled sweep design
- [ ] Run the sweep across all four consumers
- [ ] Document the Pareto-optimal α-region per consumer
- [ ] Run the cross-consumer interference test (single NPC, all 4 fusions)
- [ ] Honest assessment: is the paper-default in the intersection of all four Pareto regions?
- [ ] If yes: claim Super-GOAT in a separate validation note
- [ ] If no: document per-consumer recommended α-pairs, close this issue as "GOAT but not Super-GOAT"

---

**TL;DR:** Plan 277's 4/4 fusion gates passed with the same paper-default α-pair (0.3, 0.03). This issue tracks the Super-GOAT escalation: is the unified surprise bus (one α-pair for all consumers) a real universal property, or did each consumer just happen to work with the paper default? Requires a separate controlled sweep to validate. NOT claimed here.
