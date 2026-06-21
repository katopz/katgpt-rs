# FaithfulnessProbe — Causal Intervention Diagnostic for Injected Memory

**Plan:** [278](../.plans/278_faithfulness_probe_modelless.md) | **Plan 298 (SmearClassifier extension):** [298](../.plans/298_smear_aware_faithfulness_probe.md)
**Research:** [244 — Self-Evolver Faithfulness / Cognitive Integrity Layer](../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md) | **Research 277 (SmearClassifier):** [277 — DiffusionGemma Transparency / Smearing / Faithfulness](../.research/277_DiffusionGemma_Transparency_Smearing_Faithfulness.md)
**Private guide (riir-ai):** [129 — Cognitive Integrity Layer Architectural Guide](../../riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md)
**Runtime integration (riir-ai):** Plan 308 (private, unblocked by this primitive)
**Source paper:** [Zhao et al. 2026 — Large Language Model Agents Are Not Always Faithful Self-Evolvers](https://arxiv.org/pdf/2601.22436) (ICML 2026) | **SmearClassifier paper:** [Engels et al. 2026 — How Transparent is DiffusionGemma?](https://arxiv.org/abs/2606.20560) §5.2 (arXiv:2606.20560)
**Benchmark:** [278_faithfulness_probe_goat.md](../.benchmarks/278_faithfulness_probe_goat.md) | **298_smear_classifier_goat.md**](../.benchmarks/298_smear_classifier_goat.md)

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
use katgpt_core::faithfulness::types::ConsumerContext;

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
use katgpt_core::faithfulness::probe::{DefaultFaithfulnessProbe, FaithfulnessProbe};

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
use katgpt_core::faithfulness::attribution::{AttributionProbe, FiniteDifferenceAttributionProbe};

let mut probe = FiniteDifferenceAttributionProbe::new(consumer);
let norm = probe.attribution_norm(&memory, 1e-3); // epsilon = 1e-3
// Higher norm = memory has more causal influence on behavior.
// Rank segments by this to prioritize retrieval.
```

Validated against exact Integrated Gradients on a non-linear consumer: **Spearman ρ = 1.0000** across 64 segments (G2 GOAT gate).

### `TriggeredInjectionGate` — skip injection when saturated

```rust
use katgpt_core::faithfulness::gate::{EntropyThresholdGate, TriggeredInjectionGate};

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
use katgpt_core::faithfulness::gate::UncertaintySignal;

impl UncertaintySignal for MyConsumer {
    fn uncertainty(&self) -> f32 {
        // collapse signal (Plan 212), curiosity pulse (Research 041),
        // or action entropy — all collapse to [0, 1]
    }
}
```

---

## SmearClassifier — Ternary Latent-Mass Vocabulary (Plan 298)

**Feature:** `smear_classifier` (opt-in, implies `faithfulness_probe`).
**Paper:** Engels et al. 2026, [arXiv:2606.20560](https://arxiv.org/abs/2606.20560) §5.2.1 / §5.2.2.
**Research:** [277 — DiffusionGemma Transparency / Smearing / Faithfulness](../.research/277_DiffusionGemma_Transparency_Smearing_Faithfulness.md).

The binary `FaithfulnessProbe` (Plan 278) answers "is this memory segment
causally driving behavior?". It does NOT answer "HOW is the consumer's latent
mass distributed?". The `SmearClassifier` (Plan 298) adds that vocabulary:

extending the binary faithful/unfaithful signal with three classes:

| Class | Paper § | Mass distribution | Verdict |
|---|---|---|---|
| `CoherentSingle` | — | One dominant hypothesis at one site. | Faithful single-hypothesis. |
| `TokenSmear { span }` | §5.2.1 | One direction spread across `span` adjacent sites (cosine ≈ 1.0 between rows). | **Benign** positional uncertainty. Faithful. |
| `SequenceSmear { n_hypotheses, semantic_distance }` | §5.2.2 | ≥2 semantically distinct directions at one site (cosine ≈ 0.0 between rows). | **Potentially unfaithful** multi-hypothesis superposition requiring disambiguation. |

`#[repr(u8)]` enum — 1-byte sync-friendly output (AGENTS.md rule). Zero-allocation
hot path: caller passes a scratch `&mut [f32]` of length `k + k*(k-1)/2`.
`simd_dot_f32` for the pairwise cosines. **k=8, d=32 at 107.6 ns** on Apple
Silicon arm64 (G3 GOAT gate — plasma-tier).

### Wiring `SmearClassifier` into `DefaultFaithfulnessProbe`

```rust
use katgpt_core::faithfulness::{
    CosineSmearClassifier, DefaultFaithfulnessProbe, SmearSource,
};

// 1. (Optional) Implement `SmearSource` on your consumer IF it carries
//    a multi-hypothesis superposition (MUX Plan 178, BoM Plan 281).
//    Do NOT implement this on plain-autoregressive consumers — they are
//    always `CoherentSingle` by construction.
impl SmearSource for MyMuxConsumer {
    fn latent_mass_distribution(&self) -> (&[f32], usize, usize) {
        (&self.k_hypotheses_flat, self.k, self.d)
    }
}

// 2. Construct the probe with the classifier attached.
let probe = DefaultFaithfulnessProbe::new(consumer, irrelevant_pool, filler)
    .with_smear_classifier(Box::new(CosineSmearClassifier::default()));

// 3. Run the smear-aware audit.
let outcome = probe.faithfulness_profile_full(
    &memory, &mut rng,
    Some(&consumer as &dyn SmearSource),
    &mut scratch,
);
// outcome.profile.is_faithfully_used(0.5)  — binary verdict (Plan 278)
// outcome.smear                               — Option<SmearReport> (Plan 298)
```

The existing `probe_intervention` / `faithfulness_profile` methods are
**unaffected** — they continue to return only the binary `Delta` /
`FaithfulnessProfile`. The smear-aware surface is additive.

### Diagnostic-only contract

The `SmearReport` is a **diagnostic** — it does NOT:
- Add a sync dependency (the `#[repr(u8)]` class byte CAN be synced but the
  report itself is not committed to the chain).
- Emit a chain commitment.
- Change the `TriggeredInjectionGate` decision (Plan 278 default-on gate
  remains the source of truth for inject/skip).

Downstream consumers (riir-ai Cognitive Integrity Layer, `.research/129`)
consume the report to react differently to benign positional uncertainty
vs potentially-unfaithful multi-hypothesis computation.

### When to escalate to Cognitive Integrity Layer attention

A `SequenceSmear` report with high `semantic_distance` (default threshold:
`tau_same = 0.1`, so `semantic_distance > 0.1` already classifies as
SequenceSmear; values near `1.0` indicate near-orthogonal hypotheses) is the
signal that warrants Cognitive Integrity Layer attention (riir-ai
`.research/129`). The consumer is holding multiple semantically distinct
hypotheses in superposition without having committed — this is the
paper's "sequence smearing" failure mode (§5.2.2) where the model's output
is ambiguous because it hasn't disambiguated between competing candidate
sequences.

---

## Canonical Example (generic — no game semantics)

The katgpt-rs primitive ships **generic math only**. The canonical game wiring (HLA `evolve_hla`, NeuronShard, KG triples, emotion channels) is private → riir-ai Plan 308.

```rust
use fastrand::Rng;
use katgpt_core::faithfulness::{
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
| `SmearReport` per audit (Plan 298) | Latent (mass distribution) | NO — per-entity diagnostic; the `#[repr(u8)]` class byte CAN be synced if downstream wants to, but the report itself is not committed |
| Gate decision (inject/skip) | Latent (bool) | NO — local consumer state |
| `dead_injection` event | Raw (event) | YES — audit trail (segment ID + deltas as f64) |

Probes NEVER substitute latent for raw in anti-cheat validation. The "raw signature co-emission" rule emits raw alongside latent — raw is the anti-cheat anchor.

---

## GOAT Gate Results

### Plan 278 (binary probe)

| Gate | Metric | Threshold | Measured | Verdict |
|---|---|---|---|---|
| **G1** | Faithful detection rate | ≥99% | **100.0%** (200/200) | ✅ PASS |
| **G1b** | Unfaithful detection rate | ≥99% | **100.0%** (200/200) | ✅ PASS |
| **G2** | IG surrogate Spearman ρ | ≥0.8 | **1.0000** (64 segments, non-linear consumer) | ✅ PASS |
| **G3a** | Triggered injection skip rate | ≥50% | **50.0%** (1000/2000 in saturated regime) | ✅ PASS |
| **G3b** | Quality parity (cosine delta) | ≤2% | **0.63%** | ✅ PASS |
| **G8** | Zero-overhead off | 0% regression | **0 symbols** in default-off build | ✅ PASS |

**Decision:** `triggered_injection` promoted to **default-ON** (G3 passed — saves compute, matches quality). `faithfulness_probe` kept **opt-in** (diagnostic, audit cadence).

### Plan 298 (SmearClassifier — ternary extension)

| Gate | Metric | Threshold | Measured | Verdict |
|---|---|---|---|---|
| **G1** | Correctness + determinism | 6 unit tests pass | **6/6 PASS** | ✅ PASS |
| **G2** | Useful discrimination (SequenceSmear / TokenSmear unfaithfulness ratio) | ≥2.0× | **2.11×** (1000 trials/class, k=8, d=16) | ✅ PASS |
| **G3** | Latency (k=8, d=32) | ≤200 ns | **107.6 ns** (Apple Silicon arm64 SIMD) | ✅ PASS |

**Decision:** `smear_classifier` stays **opt-in** — correct, useful, fast, but default-on promotion requires real-workload evidence from riir-ai Plan 308 integration (T4.3, deferred). See [298_smear_classifier_goat.md](../.benchmarks/298_smear_classifier_goat.md) for the full evidence.

---

## Cross-References

- **Plan 278 (FaithfulnessProbe):** [278_faithfulness_probe_modelless.md](../.plans/278_faithfulness_probe_modelless.md)
- **Plan 298 (SmearClassifier):** [298_smear_aware_faithfulness_probe.md](../.plans/298_smear_aware_faithfulness_probe.md)
- **Plan 178 (MUX superposition generator):** consumed by `SmearClassifier` — the K parallel token streams it produces are the classifier's primary input
- **Plan 281 (BoMSampler):** consumed by `SmearClassifier` — the K belief states it samples are the classifier's secondary input
- **Research 244:** [244_Self_Evolver_Faithfulness_Cognitive_Integrity.md](../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md)
- **Research 277 (SmearClassifier):** [277_DiffusionGemma_Transparency_Smearing_Faithfulness.md](../.research/277_DiffusionGemma_Transparency_Smearing_Faithfulness.md)
- **Private guide (riir-ai):** [129_Cognitive_Integrity_Layer_Guide.md](../../riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md)
- **Runtime integration (riir-ai Plan 308):** unblocked by this primitive
- **Benchmark 278:** [278_faithfulness_probe_goat.md](../.benchmarks/278_faithfulness_probe_goat.md)
- **Benchmark 298:** [298_smear_classifier_goat.md](../.benchmarks/298_smear_classifier_goat.md)
- **Source paper 278:** [arxiv 2601.22436](https://arxiv.org/pdf/2601.22436)
- **Source paper 298:** [arxiv 2606.20560](https://arxiv.org/abs/2606.20560)

## TL;DR

Generic, modelless, zero-alloc causal intervention diagnostic for injected memory. Three primitives: `FaithfulnessProbe` (detect dead injections), `AttributionProbe` (IG surrogate ranking), `TriggeredInjectionGate` (saturated-regime skip). All Plan 278 GOAT gates pass. `triggered_injection` default-on; `faithfulness_probe` opt-in. **Plan 298 adds `SmearClassifier`** — a ternary (CoherentSingle / TokenSmear / SequenceSmear) latent-mass vocabulary extending the binary probe, distilled from arXiv:2606.20560. All Plan 298 GOAT gates pass (G1 6/6, G2 ratio 2.11×, G3 107.6 ns). `smear_classifier` stays opt-in pending real-workload evidence. Unblocks riir-ai Plan 308.
