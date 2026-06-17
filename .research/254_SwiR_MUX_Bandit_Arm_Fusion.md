# Research 254: Fusion B — SwiR × Three-Mode Router (MUX Bandit Arm)

> **Date:** 2026-06-17
> **Status:** Exploratory — Super-GOAT candidate (per Research 241 §2.3). Novelty gate ⚠️ NOT FULLY CHECKED. No implementation yet.
> **Fuses:** [Research 241](241_SwiReasoning_Explicit_Latent_Switch.md) (SwiR) × Plan 211 Three-Mode Router × [Research 158](../.research/158_MUX_Pruner_Modelless.md) (MUX Pruner)
> **Related Plans:** [Plan 275](../.plans/275_swir_switch_thinking.md) (SwiR), Plan 211 (Three-Mode Router)
> **Classification:** Public — generic inference mechanics (WHAT, not HOW)

---

## TL;DR

Add a **Latent arm** to Plan 211's Three-Mode Router, creating a Four-Mode Router where the bandit decides per-query whether to use:

1. **Direct** (no reasoning, output immediately)
2. **CoT** (explicit chain-of-thought, full thinking_cot)
3. **Early-Exit** (truncated CoT, confidence-based stop)
4. **Latent** (NEW — SwiR's soft-embedding mode, no discrete tokens during reasoning)

The bandit learns the per-query optimal mode via reward feedback (accuracy / latency tradeoff). SwiR's entropy-driven controller becomes one arm of the MAB rather than a standalone module — the bandit decides WHEN to invoke SwiR, and SwiR decides HOW to switch within Latent↔Explicit once invoked.

**Why Super-GOAT candidate:** the bandit learns the meta-decision "is this query suited to latent reasoning?" from data, rather than applying SwiR uniformly. Some queries (rigid-constraint, arithmetic) may always want Explicit; others (creative, open-ended) may always want Latent. The bandit routes accordingly.

---

## Source paper grounding

### SwiReasoning (Research 241, arXiv:2510.05069)

SwiR is applied uniformly — every query goes through the Explicit↔Latent switch logic. The G6 kurtosis escape hatch (Plan 275 T3.8) is a crude per-query filter (rigid-constraint tasks bypass Latent), but it's a hard threshold, not a learned router.

### Three-Mode Router (Plan 211)

Plan 211's Three-Mode Router uses a bandit to select between Direct / CoT / Early-Exit per query. The bandit learns from reward = accuracy − λ · latency. Each "mode" is a different decode strategy.

### MUX Pruner (Research 158)

MUX Pruner distills the multi-arm bandit pattern into a modelless primitive — the bandit doesn't need to understand the arms, it just tracks reward per arm and samples via Thompson sampling (sigmoid-bounded, never softmax per AGENTS.md).

---

## The fusion: Four-Mode Router with Latent arm

### Current Three-Mode Router (Plan 211)

```rust
enum ThreeMode {
    Direct,    // no reasoning
    Cot,       // full chain-of-thought
    EarlyExit, // truncated CoT
}

// Bandit selects mode per query, then the decode loop uses that mode.
let mode = bandit.select_arm();  // Thompson sample
let result = match mode {
    ThreeMode::Direct => decode_direct(query),
    ThreeMode::Cot => decode_cot(query),
    ThreeMode::EarlyExit => decode_early_exit(query),
};
bandit.update(mode, reward(result));
```

### Proposed Four-Mode Router

```rust
enum FourMode {
    Direct,    // no reasoning
    Cot,       // full chain-of-thought (Explicit only)
    EarlyExit, // truncated CoT
    Latent,    // NEW — SwiR's Explicit↔Latent switching
}

let mode = bandit.select_arm();
let result = match mode {
    FourMode::Direct => decode_direct(query),
    FourMode::Cot => decode_cot(query),
    FourMode::EarlyExit => decode_early_exit(query),
    FourMode::Latent => decode_swir(query, &swir_controller),  // NEW arm
};
bandit.update(mode, reward(result));
```

### What the Latent arm provides

When the bandit selects `Latent`, the decode loop uses `SwiRStrategyAdapter` (Plan 275 Phase 2) — the full Explicit↔Latent switching with entropy-driven mode selection, convergence guards, and overthinking suppression. This is SwiR as a sub-component of the router, not a standalone strategy.

### Reward signal

The bandit learns:
- **Direct** wins on easy queries (low latency, acceptable accuracy)
- **CoT** wins on reasoning-heavy queries (high accuracy, high latency)
- **EarlyExit** wins on medium queries (good accuracy, lower latency than CoT)
- **Latent** wins on queries where SwiR's mode switching provides efficiency gains (accuracy comparable to CoT, latency comparable to EarlyExit)

The bandit's Thompson sampling automatically routes each query to the optimal strategy.

---

## Expected gains (hypothesis)

### Why this might beat uniform SwiR

Uniform SwiR applies mode switching to every query. But some queries are pure-arithmetic (no benefit from Latent exploration) or pure-retrieval (no reasoning needed). The bandit learns to route those to Direct/CoT instead, saving the SwiR overhead.

Conversely, some queries are open-ended reasoning (math proofs, code generation) where SwiR's Latent mode shines. The bandit learns to route those to Latent.

The net effect: SwiR is used only where it helps, not uniformly.

### Why this might NOT beat uniform SwiR

SwiR's G6 kurtosis escape hatch already filters rigid-constraint tasks. If G6 works well, the bandit's per-query routing might be redundant — SwiR already adapts via G6.

The bandit adds cold-start cost (exploration phase) and a reward-signal dependency (needs accuracy feedback, which may not be available at inference time).

---

## Novelty gate (Q1–Q4)

### Q1: Is this already in the literature?

⚠️ **NOT FULLY CHECKED.** Partial findings:

- **"Adaptive computation time"** (Graves, 2016) uses a learned halting distribution, but doesn't switch between discrete/continuous modes.
- **"Mixture of Depths"** (Raposo et al., 2024) routes tokens through different transformer depths, but stays in token-space.
- **"Self-speculative decoding"** (Plan 089, internal) uses a bandit to select draft length, but doesn't switch reasoning modes.
- **"Three-Mode Router"** (Plan 211, internal) is the direct predecessor — adding a Latent arm is the novel extension.

**No direct prior art found** for "bandit-selected latent reasoning mode in a multi-mode decode router". But the search was not exhaustive.

### Q2: Is this a derivative of existing katgpt-rs primitives?

Yes — it's a composition of Plan 211 (Three-Mode Router) and Plan 275 (SwiR). The novelty is in the composition, not in either component alone.

### Q3: Does the paper claim this?

No. SwiReasoning doesn't discuss per-query routing — it applies SwiR uniformly.

### Q4: Is this super-GOAT-shaped?

**Moderately.** Adding a learned arm to an existing bandit router is incremental, not a new capability class. But if the Latent arm consistently wins on a query subclass, it validates that SwiR has a "sweet spot" that uniform application misses.

---

## Implementation plan (if pursued)

### Phase 1: Extend Three-Mode Router to Four-Mode

1. Add `FourMode::Latent` variant.
2. Wire `SwiRStrategyAdapter` as the decode strategy for the Latent arm.
3. Unit test: bandit can select all 4 arms over a synthetic reward schedule.

### Phase 2: Synthetic reward validation

1. Construct a synthetic task suite where each mode is optimal for a subset:
   - Easy queries → Direct wins
   - Medium queries → EarlyExit wins
   - Hard queries → CoT wins
   - "Explorable" queries → Latent wins (SwiR finds the answer faster)
2. Verify the bandit converges to per-query optimal arm selection.

### Phase 3: Real-model GOAT gate (riir-ai)

1. G1-bandit: accuracy ≥ uniform SwiR on MATH500.
2. G2-bandit: token efficiency ≥ uniform SwiR.
3. G-routing: bandit's arm distribution matches expected query difficulty.

### Phase 4: Promotion

If G1-bandit ≥ uniform SwiR AND G2-bandit ≥ uniform SwiR, extend Plan 211 with the Latent arm.

---

## Verdict

**Moderate Super-GOAT candidate — pursue after Plan 211 validates.** The fusion is clean (add an arm to an existing bandit) and the hypothesis is reasonable (per-query routing beats uniform SwiR). But:

1. The gain depends on SwiR having a clear "sweet spot" query subclass — if SwiR is uniformly better (or worse), the bandit adds nothing.
2. Novelty is not fully verified.
3. The reward signal needs accuracy feedback, which may not be available at inference time (need a proxy).

**Recommendation:** extend Plan 211 (no new plan needed) after:
- Plan 275's real-model G1/G2 proves SwiR works (riir-ai Plan 299)
- A quick synthetic experiment on the existing harness showing the Latent arm wins on a distinct query subclass
