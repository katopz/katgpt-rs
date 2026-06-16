# FaithfulnessProbe — Causal Intervention Diagnostic for Injected Memory

**Plan:** [278](../.plans/278_faithfulness_probe_modelless.md)
**Research:** [244 — Self-Evolver Faithfulness / Cognitive Integrity Layer](../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md)
**Private guide (riir-ai):** [129 — Cognitive Integrity Layer Architectural Guide](../../riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md)
**Runtime integration (riir-ai):** Plan 308 (private, unblocked by this primitive)
**Source paper:** [Zhao et al. 2026 — Large Language Model Agents Are Not Always Faithful Self-Evolvers](https://arxiv.org/pdf/2601.22436) (ICML 2026)
**Benchmark:** [278_faithfulness_probe_goat.md](../.benchmarks/278_faithfulness_probe_goat.md)

---

## TL;DR

Three modelless primitives that verify a consumer's behavior is **causally bound** to injected memory. All zero-training, zero-backprop, zero-allocation on hot paths. Based on Zhao et al. 2026's finding that LLM agents silently ignore 60%+ of their condensed experience.

| Primitive | Purpose | Feature | Cost |
|---|---|---|---|
| `FaithfulnessProbe` | Detect dead injections (consumer ignores memory) | `faithfulness_probe` (opt-in) | <1ms/segment, audit cadence |
| `AttributionProbe` | Rank memory segments by causal influence (IG surrogate) | `faithfulness_probe` (opt-in) | <100µs/segment, audit cadence |
| `TriggeredInjectionGate` | Skip injection when consumer is saturated | `triggered_injection` (**default-ON**) | 0.132ns/call, hot path |

---

## Feature Flags

```toml
[dependencies]
katgpt-rs = { version = "...", features = ["triggered_injection"] }  # default: gate only
# or for the full diagnostic suite:
katgpt-rs = { version = "...", features = ["faithfulness_probe", "triggered_injection"] }
```

- **`triggered_injection`** (default-ON after GOAT G3): enables `TriggeredInjectionGate` + `EntropyThresholdGate` + `UncertaintySignal`. Hot-path inject/skip decision.
- **`faithfulness_probe`** (opt-in, diagnostic): additionally enables `FaithfulnessProbe` + `AttributionProbe` + perturbation strategies. Runs at audit cadence (every N ticks), not per-tick.

**Why separate?** See [ADR-2](../.plans/278_faithfulness_probe_modelless.md#adr-2-why-separate-faithfulness_probe-and-triggered_injection-features). The diagnostic is expensive (full intervention suite); the gate is cheap (one compare). Coupling them would either make the diagnostic too cheap or the hot-path too expensive.

---

## API Reference

### `ConsumerContext` trait (implement this for your consumer)

```rust
use katgpt_rs::faithfulness::types::ConsumerContext;

impl ConsumerContext for MyConsumer {
    type Behavior = f32;           // or Vec<f32>, action enum, etc.
    type Delta = f32;              // must be PartialOrd + Copy + Default
    type Memory = Vec<f32>;        // must implement MemorySlice

    fn baseline_behavior(&self) -> Self::Behavior { /* prior / fallback */ }
    fn behavior_with_memory(&self, memory: &Self::Memory) -> Self::Behavior { /* forward pass */ }
    fn behavior_delta(&self, a: &Self::Behavior, b: &Self::Behavior) -> Self::Delta { /* distance */ }
}
```

### `FaithfulnessProbe` — detect dead injections

```rust
use fastrand::Rng;
use katgpt_rs::faithfulness::probe::{DefaultFaithfulnessProbe, FaithfulnessProbe};

let consumer = MyConsumer { /* ... */ };
let irrelevant_pool = vec![/* tokens from a different context */];
let filler = 0.0_f32; // or <pad> token id
let mut probe = DefaultFaithfulnessProbe::new(consumer, irrelevant_pool, filler);

let memory = vec![/* the injected segment under audit */];
let mut rng = Rng::with_seed(42);
let profile = probe.faithfulness_profile(&memory, &mut rng);

if profile.is_faithfully_used(0.5) {
    // memory is causally driving behavior — keep it
} else {
    // dead injection — consumer ignores this memory, demote in retrieval priority
}
```

The `FaithfulnessProfile` has four delta fields:
- `empty_delta` — content zeroed. Faithful consumer falls back to baseline (small delta).
- `shuffle_or_corrupt_delta` — structure destroyed. Faithful consumer reacts (large delta).
- `irrelevant_delta` — unrelated content substituted. Faithful consumer reacts (large delta).
- `filler_delta` — placeholder constant. Faithful consumer reacts (large delta).

`is_faithfully_used(threshold)` returns `true` iff all four conditions hold.

### `AttributionProbe` — rank segments by causal influence

```rust
use katgpt_rs::faithfulness::attribution::{AttributionProbe, FiniteDifferenceAttributionProbe};

let mut probe = FiniteDifferenceAttributionProbe::new(consumer);
let norm = probe.attribution_norm(&memory, 1e-3); // epsilon = 1e-3
// Higher norm = memory has more causal influence on behavior.
// Rank segments by this to prioritize retrieval.
```

Validated against exact Integrated Gradients on a non-linear consumer: **Spearman ρ = 1.0000** across 64 segments (G2 GOAT gate).

### `TriggeredInjectionGate` — skip injection when saturated

```rust
use katgpt_rs::faithfulness::gate::{EntropyThresholdGate, TriggeredInjectionGate};

let gate = EntropyThresholdGate::default(); // tau=0.5, lambda=8.0

let uncertainty = consumer.uncertainty(); // ∈ [0, 1]
if gate.should_inject(uncertainty) {
    // inject memory — consumer is uncertain, memory will help
} else {
    // skip — consumer is saturated, memory would be redundant
}
```

The gate uses **sigmoid** (never softmax — AGENTS.md constraint): `should_inject(u) := sigmoid(λ·(u−τ)) > 0.5`. Since `sigmoid(x) > 0.5 ⟺ x > 0` and `λ > 0`, this collapses to `u > τ` for the boolean case — one compare, no `exp()` (0.132ns/call). The full sigmoid value is available via `EntropyThresholdGate::sigmoid_value(u)` for opt-in soft-gating.

### `UncertaintySignal` — unify entropy / collapse / curiosity

```rust
use katgpt_rs::faithfulness::gate::UncertaintySignal;

impl UncertaintySignal for MyConsumer {
    fn uncertainty(&self) -> f32 {
        // collapse signal (Plan 212), curiosity pulse (Research 041),
        // or action entropy — all collapse to [0, 1]
    }
}
```

---

## Canonical Example (generic — no game semantics)

The katgpt-rs primitive ships **generic math only**. The canonical game wiring (HLA `evolve_hla`, NeuronShard, KG triples, emotion channels) is private → riir-ai Plan 308.

```rust
use fastrand::Rng;
use katgpt_rs::faithfulness::{
    ConsumerContext, DefaultFaithfulnessProbe, FaithfulnessProbe,
    EntropyThresholdGate, TriggeredInjectionGate,
};

// 1. Implement ConsumerContext for your consumer.
struct DotProductConsumer { weights: Vec<f32> }
impl ConsumerContext for DotProductConsumer {
    type Behavior = f32;
    type Delta = f32;
    type Memory = Vec<f32>;
    fn baseline_behavior(&self) -> f32 { 0.0 }
    fn behavior_with_memory(&self, m: &Vec<f32>) -> f32 {
        m.iter().zip(self.weights.iter()).map(|(&v, &w)| v * w).sum()
    }
    fn behavior_delta(&self, a: &f32, b: &f32) -> f32 { (a - b).abs() }
}

// 2. Gate: skip injection when saturated.
let gate = EntropyThresholdGate::default();
let uncertainty = 0.3; // low — consumer is confident
if !gate.should_inject(uncertainty) {
    // skip — no need to inject memory
}

// 3. Probe: audit whether injected memory is actually used.
let consumer = DotProductConsumer { weights: vec![1.0, 2.0, 3.0] };
let mut probe = DefaultFaithfulnessProbe::new(consumer, vec![0.5, 0.6], 1.0);
let memory = vec![1.0, 2.0, 3.0];
let profile = probe.faithfulness_profile(&memory, &mut Rng::with_seed(42));
assert!(profile.is_faithfully_used(0.5)); // memory drives behavior
```

---

## Latent vs Raw Boundary

| Quantity | Space | Synced? |
|---|---|---|
| `FaithfulnessProfile` per segment | Latent (behavioral deltas) | NO — per-entity diagnostic |
| `AttributionProbe` norm | Latent (sensitivity scalar) | NO — per-entity, local |
| Gate decision (inject/skip) | Latent (bool) | NO — local consumer state |
| `dead_injection` event | Raw (event) | YES — audit trail (segment ID + deltas as f64) |

Probes NEVER substitute latent for raw in anti-cheat validation. The "raw signature co-emission" rule emits raw alongside latent — raw is the anti-cheat anchor.

---

## GOAT Gate Results (Phase 3)

| Gate | Metric | Threshold | Measured | Verdict |
|---|---|---|---|---|
| **G1** | Faithful detection rate | ≥99% | **100.0%** (200/200) | ✅ PASS |
| **G1b** | Unfaithful detection rate | ≥99% | **100.0%** (200/200) | ✅ PASS |
| **G2** | IG surrogate Spearman ρ | ≥0.8 | **1.0000** (64 segments, non-linear consumer) | ✅ PASS |
| **G3a** | Triggered injection skip rate | ≥50% | **50.0%** (1000/2000 in saturated regime) | ✅ PASS |
| **G3b** | Quality parity (cosine delta) | ≤2% | **0.63%** | ✅ PASS |
| **G8** | Zero-overhead off | 0% regression | **0 symbols** in default-off build | ✅ PASS |

**Decision:** `triggered_injection` promoted to **default-ON** (G3 passed — saves compute, matches quality). `faithfulness_probe` kept **opt-in** (diagnostic, audit cadence).

---

## Cross-References

- **Plan:** [278_faithfulness_probe_modelless.md](../.plans/278_faithfulness_probe_modelless.md)
- **Research:** [244_Self_Evolver_Faithfulness_Cognitive_Integrity.md](../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md)
- **Private guide (riir-ai):** [129_Cognitive_Integrity_Layer_Guide.md](../../riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md)
- **Runtime integration (riir-ai Plan 308):** unblocked by this primitive
- **Benchmark:** [278_faithfulness_probe_goat.md](../.benchmarks/278_faithfulness_probe_goat.md)
- **Source paper:** [arxiv 2601.22436](https://arxiv.org/pdf/2601.22436)

## TL;DR

Generic, modelless, zero-alloc causal intervention diagnostic for injected memory. Three primitives: `FaithfulnessProbe` (detect dead injections), `AttributionProbe` (IG surrogate ranking), `TriggeredInjectionGate` (saturated-regime skip). All GOAT gates pass. `triggered_injection` default-on; `faithfulness_probe` opt-in. Unblocks riir-ai Plan 308.
