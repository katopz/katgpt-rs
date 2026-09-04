# Plan 570: CLR-Amplified Set Attention (`clr_weighted_set_attention`)

**Date:** 2026-08-06
**Prior PoC:** `Issue 575` — G8 CLOSED (CLR path)
**Research:** [469 §PoC Addendum](../.research/469_collective_intelligence_payoff_schemes.md) — Wang/Plotkin PNAS 2025 feedback-payoff distillation
**Closes:** [Bench 354](../.benchmarks/354_set_attention_goat.md) L71 G8 documented limitation
**Feature flag:** `clr_weighted_set_attention` (opt-in → default-on if GOAT passes)

## Context

Issue 575 proved CLR's `^M` nonlinear reliability gate (Plan 284) closes the
Set Attention G8 collective-inference failure (Bench 354 L71: "averaging
cannot amplify detection"). On a synthetic N=64 crowd threat-detection task:

- **Plain SA (G8 baseline):** 9.4% top-1 (averaging dilutes signal)
- **Individual cosine (floor):** 12.0%
- **CLR sigmoid^M (M=5):** **17.6%** (+5.6pp over individual — G8 closed)
- **Aggregate amplification:** CLR-weighted aggregate carries **6.23×** more
  threat signal than the plain mean

The paper's feedback payoff (Wang/Plotkin PNAS 2025) amplifies the aggregate
(5.02×) but fails per-entity identification (0.9%) — the paper's mechanism
works at the learning-dynamics level (replicator equation), not single-shot
inference. CLR's `^M` exponent is the production mechanism for modelless
crowd-level amplification.

This plan ships the CLR-weighted variant as a sibling of
`set_sigmoid_attention_into`.

## Design

### New function: `clr_weighted_set_attention_into`

A sibling of `set_sigmoid_attention_into` that accepts per-entity reliability
weights `r_j` and uses them to modulate the attention contribution:

```text
output_i = h_i + (γ / Σ_j r_j) · Σ_j α_ij · r_j · (v_j − h_i)
```

The `r_j` weighting concentrates the aggregate toward high-reliability
entities, converting plain averaging into amplification. The caller computes
`r_j` externally (CLR-style sigmoid^M, or any other reliability score) and
passes it as a slice.

```rust
pub fn clr_weighted_set_attention_into(
    states: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: Option<&[f32]>,
    reliability: &[f32],  // NEW: per-entity reliability weights r_j
    output: &mut [f32],
    cfg: &SetAttentionConfig,
    n: usize,
    d: usize,
    k: usize,
    scratch_q: &mut [f32],
    scratch_k: &mut [f32],
    scratch_alpha: &mut [f32],
) -> Result<(), SetAttentionError>
```

When `reliability` is all-ones (uniform), this reduces to plain
`set_sigmoid_attention_into` — the existing primitive is a special case.

### Reliability computation helper: `clr_reliability_scores`

A convenience function that computes CLR-style reliability scores from belief
states + M direction vectors:

```rust
pub fn clr_reliability_scores(
    states: &[f32],
    directions: &[f32],  // M * d, row-major
    m: usize,            // number of directions (CLR exponent)
    n: usize,
    d: usize,
    output: &mut [f32],  // n reliability scores
)
```

Computes `r_j = (mean_m sigmoid(dot(h_j, dir_m)))^M` — the CLR headline
formula (Plan 284). The `^M` unrolled path for M=5 matches the CLR hot path.

### Feature gate: `clr_weighted_set_attention` (opt-in)

Ships behind a new opt-in feature (implies `set_attention`). Promotion to
default-on requires the GOAT gate to pass:
- G1: CLR-weighted = plain SA when reliability is uniform (special case).
- G2: latency ≤ 2× plain SA (one extra multiply per peer).
- G3: no-regression on existing set_attention tests.
- G4: zero-alloc (same scratch discipline as plain SA).
- G8 (the quality gate this primitive targets): CLR-weighted identification
  accuracy > plain SA by ≥5pp on the Issue 575 PoC fixture, reproduced in
  the katgpt-core test suite.

## Phases

### Phase 1 — Implement `clr_weighted_set_attention_into`
- [x] Add the function to `katgpt-core/src/set_attention.rs`
- [x] Add `clr_reliability_scores` helper
- [x] G1 test: uniform reliability ≡ plain SA (bit-identical) — PASS (dense + topk)
- [x] G4 test: zero-alloc (counting allocator) — PASS (0 allocs/100 calls)
- [x] G2 test: latency ≤ 2× plain SA at N=64 — PASS (1.00× ratio in unit test)

### Phase 2 — GOAT gate (G8 closure reproduction)
- [x] Port the Issue 575 PoC fixture as a katgpt-core integration test
- [x] G8 test: CLR reliability identification accuracy > plain SA +5pp — PASS (+8.7pp)
- [x] G8 test: CLR-weighted SA aggregate amplification ≥ 2× — PASS (3.88×)

### Phase 3 — Promotion
- [x] If all gates pass: promote `clr_weighted_set_attention` to default-on
- [x] Update Bench 354: G8 status → "CLOSED by clr_weighted_set_attention"
- [x] Update Plan 354 docs: add CLR-weighted variant to the API surface

## Non-goals

- **Multi-direction CLR inside the primitive.** The reliability scores are
  computed externally — the primitive is a generic weighted-attention kernel,
  not CLR-specific. This keeps it composable with other reliability sources.
- **Feedback payoff variant.** Issue 575 showed feedback payoff amplifies the
  aggregate (5.02×) but fails identification (0.9%). Not shipped as a Set
  Attention variant — the learning-dynamics convergence is out of scope for
  modelless inference.
- **SIMD optimization.** The extra multiply per peer is cheap; deferred until
  a real consumer (N>100 crowd zone) demands it.

## Cross-references

- `Issue 575` — the PoC (G8 CLOSED)
- [Research 469](../.research/469_collective_intelligence_payoff_schemes.md) — Wang/Plotkin distillation + PoC Addendum
- [Bench 354](../.benchmarks/354_set_attention_goat.md) — the G8 documented limitation
- [Research 255](../.research/255_VibeThinker_CLR_Test_Time_Reliability.md) — CLR (the ^M source)
- `Plan 284` — CLR voter (the `reliability_gate` function)
