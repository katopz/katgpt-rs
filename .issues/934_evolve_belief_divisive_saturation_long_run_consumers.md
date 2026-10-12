# Issue 934: Divisive `evolve_belief` saturates ≥ 6 of 8 dims for long-run per-tick consumers

**Status:** Open — detection only (2026-10-13). No substrate change proposed here: `evolve_belief` is byte-identity-pinned (`evolve_belief_is_byte_identical_to_inline_reference`, the shipped belief benches). The question is a CONSUMER-side one, owned by riir-ai.

**Found by:** Issue 932 T3 (closed negative), probe `crates/katgpt-sense/tests/belief_views_viability_probe.rs` (`56ad37c4c`).

## Finding

`ReconstructionState::evolve_belief` → `leaky_core::leaky_step`: once accumulated evidence `total ≥ 1`, each dim's step is

```text
δ_i = lr · (p_i − 0.5),   p_i = cumulative share of dim i's SenseKind,  Σ_kinds p = 1
```

The step reads no belief state (no leak, no self-dynamics — measured interior drift 1.6e-7 for two inits on the same evidence), and `TripleEvidence` is cumulative (never reset after construction). So for ANY evidence stream, at most one kind (≤ 2 dims via `KIND_MAP`) can drift non-negative; every other dim ramps monotonically into the `−1` clamp and stays there.

Measured (real `route → accumulate → evolve_belief`, 16 seeds × 2000 ticks, dominant-kind regime switch every 250 ticks):

| kernel | clamped channel-ticks (post-warm-up) | constant channels | ACF half-life (live channels) |
|---|---|---|---|
| divisive `evolve_belief` | **96.9%** | **96 / 128** (6 of 8 per run, pinned at −1) | ~208 ticks |
| `evolve_belief_additive` (default-on feature, no consumer) | 0% | 0 / 128 | ~108 ticks |

## Why it matters

Fine for the design shape the kernel was built for: a bounded multi-step reconstruction (`step: u8`), or a K-step branch loop that reloads the belief (riir-games `multi_hypothesis`). Degenerate for per-tick long-run consumers, where 6 of 8 belief dims carry a constant:

- riir-engine `habituation_wiring::HabituatedPerceptor::sense_tick` (`gate → route → accumulate → evolve_belief`, per tick);
- riir-engine `sync_relation_gate` (accumulate + evolve every tick; the SyncBank observes this channel — co-activation among constant dims is zero by construction);
- `ict_runtime::maybe_evolve_belief` (feature-gated).

## Tasks

- [ ] T1 (riir-ai): per consumer, decide whether the long-run state is intended. Candidates, all modelless: (a) migrate to `evolve_belief_additive` (convex, never clamps, per-kind retention); (b) a windowed/decayed `TripleEvidence` at the consumer (re-construct or decay evidence per episode); (c) record "intended" with the reason. File the riir-ai issue citing this one.
- [ ] T2: if (b) is chosen by ≥ 2 consumers, the decay belongs in the substrate as an opt-in `TripleEvidence` retention knob (default = current cumulative behaviour, byte-identical) — not a parallel accumulator.

## Notes

- The probe pins the premise (≥ 6/8 saturation, zero additive clamps); a substrate change that breaks it reds the probe and re-opens this record.
