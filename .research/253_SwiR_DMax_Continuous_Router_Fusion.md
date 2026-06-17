# Research 253: Fusion A — SwiR × DMax SPD Continuous-Mode Router

> **Date:** 2026-06-17
> **Status:** Exploratory — Super-GOAT candidate (per Research 241 §2.3). Novelty gate ⚠️ NOT FULLY CHECKED. No implementation yet.
> **Fuses:** [Research 241](241_SwiReasoning_Explicit_Latent_Switch.md) (SwiR) × [Research 072](../.research/072_DMax_Soft_Parallel_Decode.md) (DMax SPD)
> **Related Plans:** [Plan 275](../.plans/275_swir_switch_thinking.md) (SwiR), [Plan 109](../.plans/109_dmax_spd.md) (DMax SPD)
> **Classification:** Public — generic inference mechanics (WHAT, not HOW)

---

## TL;DR

Replace SwiR's binary Explicit↔Latent mode switch with a **sigmoid-weighted continuous blend** of the soft embedding and the argmax token embedding:

```
ẽ_t = σ(λ · (H̄ − H_t)) · ẽ_latent + (1 − σ(λ · (H̄ − H_t))) · e_argmax_token
```

Where:
- `H̄` = reference entropy (SwiR's block-relative baseline)
- `H_t` = current step's entropy
- `λ` = steepness (controls how sharp the mode transition is)
- `σ` = sigmoid (never softmax per AGENTS.md constraint)
- `ẽ_latent = Σ_v p_t[v] · e(v)` (SwiR's existing soft embedding)
- `e_argmax_token` = the concrete token embedding (Explicit mode)

This eliminates the discrete mode switch entirely — the residual stream continuously blends between "explore" (high entropy, soft embedding) and "commit" (low entropy, concrete token). DMax SPD's hybrid embedding pattern provides the infrastructure for feeding either a discrete token or a continuous embedding into the same decode step.

**Why Super-GOAT candidate:** this would be a new capability class — runtime-adaptive mode blending without weight updates. SwiR binary switches are already +1.8–3.1pp over baseline; continuous blending could capture the in-between regime (e.g., 70% committed + 30% exploring) that binary SwiR cannot represent.

---

## Source paper grounding

### SwiReasoning (Research 241, arXiv:2510.05069)

SwiR uses a **discrete** mode switch: each step is either Explicit (emit token) or Latent (emit soft embedding). The switch is driven by the sign of `entropy − reference_entropy`:

- `entropy < reference` → Latent→Explicit (converged, commit)
- `entropy > reference` (after dwell) → Explicit→Latent (explore)

The paper does NOT explore continuous blending — it's strictly binary.

### DMax SPD (Research 072, Plan 109)

DMax Soft Parallel Decode uses a **hybrid embedding** pattern: the decode step can consume either a discrete token id or a continuous embedding vector. This is the infrastructure that makes the continuous-mode router feasible — the residual stream already supports "feed a continuous vector that isn't exactly a token embedding".

DMax's use case is different (diffusion-based parallel decode), but the `decode_step(token_id: Option<u32>, soft_embedding: Option<&[f32]>)` interface is exactly what the continuous-mode router needs.

---

## The fusion: sigmoid-weighted continuous blend

### Current SwiR (binary)

```rust
match ctrl.step(entropy, step_index) {
    StepAction::EmitToken(id) => {
        // Feed concrete token id to the model.
        backend.decode_step(Some(id), None, &mut probs);
    }
    StepAction::EmitSoftEmbedding => {
        // Compute ẽ_latent = Σ_v p_t[v] · e(v), feed as continuous embedding.
        soft_embedding(&probs, &emb, dim, &mut soft_buf);
        backend.decode_step(None, Some(&soft_buf), &mut probs);
    }
    // ...
}
```

### Proposed continuous router

```rust
// Replace the binary step() with a continuous blend:
let h_bar = ctrl.reference_entropy();  // expose the current reference
let blend = sigmoid(lambda * (h_bar - entropy));  // ∈ (0, 1)
// blend → 1 when entropy << h_bar (converged → commit)
// blend → 0 when entropy >> h_bar (exploring → soft)

// Always compute both:
let argmax_id = probs.argmax();
let e_argmax = &emb[argmax_id * dim..(argmax_id + 1) * dim];
soft_embedding(&probs, &emb, dim, &mut soft_buf);  // ẽ_latent

// Blend:
for d in 0..dim {
    blended[d] = blend * soft_buf[d] + (1.0 - blend) * e_argmax[d];
}

// Feed the blend:
backend.decode_step(None, Some(&blended), &mut probs);
```

### The blend is still in the vocab convex hull? ⚠️ NOT NECESSARILY

The SwiR soft embedding `ẽ_latent = Σ_v p_t[v] · e(v)` is a convex combination of vocabulary embeddings → guaranteed in the per-dim [min, max] range (G4 invariant).

But `e_argmax_token` is a single vertex of the convex hull. The blend `blend · ẽ_latent + (1 − blend) · e_argmax` is:

- When `blend ∈ [0, 1]`: the blend is a convex combination of (a point inside the hull) and (a vertex) → still inside the hull. ✓
- When `blend < 0` or `blend > 1`: could exit the hull. But sigmoid bounds `blend ∈ (0, 1)`, so this is safe. ✓

**Conclusion: G4 convex-hull invariant still holds.** The continuous blend is always inside the vocab convex hull because sigmoid bounds the blend ratio to (0, 1) and both endpoints (soft embedding and argmax token) are inside/on the hull.

---

## Expected gains (hypothesis)

### Why this might beat binary SwiR

Binary SwiR wastes the transition regime. When entropy is near the reference (`H_t ≈ H̄`), the controller must pick a mode — but the "right" answer might be "60% committed, 40% exploring". Binary switching forces a cliff:

- Step N: `H_t = H̄ + ε` → Explicit (100% committed)
- Step N+1: `H_t = H̄ − ε` → Latent (100% exploring)

The continuous router would instead produce:
- Step N: `blend ≈ 0.5` → 50% committed, 50% exploring
- Step N+1: `blend ≈ 0.5` → same

This avoids the "switch shock" where the residual stream jumps discontinuously between token-space and embedding-space. The paper's signal-mixing at switch instants (paper Eq. 4) is a partial fix, but it only fires once per switch — the continuous router applies it every step.

### Why this might NOT beat binary SwiR

The paper's binary switch is simple and interpretable. The continuous blend adds a hyperparameter (`λ`, the steepness) that needs tuning. If `λ` is too small, the blend is always ≈0.5 (no mode differentiation); if too large, it approximates binary switching (no gain over SwiR).

The paper's +1.8–3.1pp gain might already capture most of the benefit — the mode switch itself, not its sharpness, might be what matters.

---

## Novelty gate (Q1–Q4 per research skill)

### Q1: Is this already in the literature?

⚠️ **NOT FULLY CHECKED.** Partial findings:

- **"Soft prompting"** (Lester et al., 2021) blends learned continuous vectors with token embeddings, but at training time, not as a runtime entropy-driven router.
- **"Input-dependent mixture of experts"** (Blanche et al., 2020) routes tokens to experts via a learned gate, but doesn't blend token-space and embedding-space.
- **"Diffusion-LM"** (Li et al., 2022) uses continuous embeddings throughout, but doesn't switch between discrete and continuous modes.
- **"SwitchHead"** (Daras et al., 2024) uses a soft router for attention head selection, but stays in token-space.

**No direct prior art found** for "entropy-driven continuous blend of discrete token and soft embedding at inference time". But the search was not exhaustive — a deeper arxiv sweep (`continuous mode switching LLM`, `entropy-driven soft routing`, `hybrid token embedding decode`) is needed before claiming novelty.

### Q2: Is this a derivative of existing katgpt-rs primitives?

Partially. It combines SwiR (Plan 275) and DMax SPD (Plan 109) in a new way. The sigmoid blend is similar to `mix_thinking_signal` (paper Eq. 4) but applied continuously rather than at switch instants.

### Q3: Does the paper claim this?

No. SwiReasoning (Research 241) is strictly binary. The continuous extension is our distillation idea.

### Q4: Is this super-GOAT-shaped?

**Yes, conditionally.** A runtime-adaptive mode blend without weight updates would be a new capability class. But the actual gain needs empirical proof — the hypothesis could be wrong (binary might be sufficient).

---

## Implementation plan (if pursued)

### Phase 1: Continuous router skeleton

1. Add `ContinuousSwiRConfig` with `lambda: f32` steepness parameter.
2. Implement `continuous_blend(entropy, h_bar, probs, emb, dim, lambda, out)`.
3. Unit test: G4 convex-hull invariant holds for 1000 random blends.
4. Unit test: blend ratio is sigmoid-shaped in `(H̄ − H_t)`.

### Phase 2: Integration with DMax SPD backend

1. Implement `ContinuousSwiRStrategyAdapter` that blends instead of switching.
2. Integration test: drive through mock decode loop, verify no mode-switch shocks.

### Phase 3: GOAT gate

1. G1-continuous: accuracy ≥ binary SwiR accuracy on MATH500 (riir-ai).
2. G2-continuous: token efficiency ≥ binary SwiR efficiency (riir-ai).
3. G4-continuous: convex-hull invariant holds (katgpt-rs).
4. G-lambda: sweep λ ∈ {0.5, 1.0, 2.0, 5.0, 10.0}, find sweet spot.

### Phase 4: Promotion

If G1-continuous ≥ binary SwiR + 0.5pp AND G2-continuous ≥ binary SwiR × 1.1, promote. Otherwise shelve.

---

## Verdict

**Super-GOAT candidate — pursue only if Q1 (novelty) checks clean.** The continuous blend is theoretically elegant and the G4 invariant still holds, but:

1. Novelty is not fully verified (Q1 needs a deeper arxiv sweep).
2. The gain hypothesis (continuous > binary) is unproven — binary SwiR might already be sufficient.
3. The hyperparameter `λ` adds tuning burden.

**Recommendation:** create this as a plan (Plan NNN) only after:
- A proper arxiv novelty search (`input-adaptive MoE routing`, `entropy-driven soft routing`, `hybrid token embedding decode`)
- A quick synthetic experiment comparing binary vs continuous on the existing `bench_275` harness (cheap, no real model needed)

If the synthetic experiment shows promise AND novelty checks clean, proceed to Plan NNN with a real-model GOAT gate in riir-ai.
