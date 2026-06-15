# Plan 273: Latent Functor — Analogy as Vector Arithmetic (SPEC)

**Date:** 2026-06-15 (rewritten 2026-06-15 — verdict + fix after audit)
**Status:** 📐 **SPEC — audited, corrected.** Awaiting implementation via [riir-ai Plan 303](../../riir-ai/.plans/303_latent_functor_runtime_npc_relational_learning.md).
**Research:** [111 (Emergent Analogical Reasoning, arXiv:2602.01992)](../.research/111_Emergent_Analogical_Reasoning_Transformers.md), [riir-ai Research 123 (Latent Functor Runtime Guide)](../../riir-ai/.research/123_Latent_Functor_Runtime_Guide.md)
**Related:** Plan 149 (Dirichlet Energy diagnostic — sibling primitive, ✅ SHIPPED to katgpt-rs), Research 231 / Plan 264 (`sparse_task_vector` — *weight*-space deltas, distinct), Research 144 / Plan 162 (linear emotion direction vectors)
**Implemented by:** [riir-ai Plan 303](../../riir-ai/.plans/303_latent_functor_runtime_npc_relational_learning.md)

---

## Audit Verdict (why this plan was rewritten)

This plan came from the pre-SKILL-bug-fix research era. Critical audit found:

| Issue | Verdict |
|-------|---------|
| Target repo `katgpt-core/src/dirichlet.rs` (public MIT) | ❌ **Wrong per `003`.** Functor arithmetic is a latent operation (projection direction + sigmoid gate). `003` explicitly: "Latent-operation internals → riir-ai internal." Must ship to riir-ai. |
| Tasks marked `[x]` COMPLETE | ❌ **Dishonest.** Code does not exist in katgpt-rs (verified: `dirichlet.rs` is 104 lines, only Plan 149 diagnostics). Reverted to `[ ]`. |
| API: `extract_functor(sources, targets, dim)` — pair alignment ambiguous | ❌ **Bug.** How are source/target paired? Position-aligned? The riir-train version uses explicit `pairs: &[(usize,usize)]`. Fixed below. |
| No zero-alloc `extract_functor_into` | ❌ **Gap.** `functor_parallelism` has `_into` but `extract_functor` doesn't. Inconsistent. Added. |
| Coherence formula not in API section | ❌ **Gap.** Defined in prose but not next to the signature. Added. |
| G5 (ranking preserved) under-specified | ⚠️ **Vague.** "Held-out source" from what distribution? Made concrete. |
| Missing: `FunctorTable`, re-estimation, NPC integration | ⚠️ **Scope creep risk if added here.** Those are runtime/game layers → Plan 303. This plan is the arithmetic spec only. |
| Constraint analysis (modelless/latent-to-latent/freeze-thaw) | ✅ **Correct and well-argued.** Kept verbatim. |
| API signatures (modulo fixes) | ✅ **Salvageable.** Kept with corrections. |
| GOAT proofs G1–G4 | ✅ **Correct.** Kept. |

**Net:** spec is 70% salvageable. Target repo, completeness claims, and API ambiguity were wrong. Fixed below. Plan 303 implements the corrected spec in riir-ai.

---

## Goal

Lift the paper's residual-stream analogy mechanism (`e_target ≈ e_source + f`) out of the Transformer into a **modelless, latent-to-latent arithmetic spec**. No weights, no training — just direction vectors over latents:

- **Estimate** a functor `f = mean_k(target_k − source_k)` from observed analog pairs.
- **Apply** it by addition: `ĥ_target = source + f` (predict a novel source's analog).
- **Verify** a candidate pair with parallelism `cos(target − source, f)` (paper §4.2 eq. 3).
- **Gate** application by a sigmoid of the functor's coherence (dot + sigmoid, never softmax).

**Target repo (corrected):** `riir-ai/crates/riir-engine/src/latent_functor/` (private, per `003`). NOT `katgpt-core/src/dirichlet.rs` (public) — the original target was wrong.

**What stays in katgpt-rs (public, unchanged):** The Dirichlet Energy diagnostic from Plan 149 — `dirichlet_energy`, `functor_adjacency`, `consecutive_adjacency`, `kv_cache_dirichlet_energy`. This is a pure measurement (graph signal smoothness), not a latent operation. Already shipped.

## Why this is modelless (constraint analysis — kept from original)

| Constraint | How this satisfies it |
|---|---|
| Modelless / inference-time | `f` is estimated from latents at runtime; zero backprop, zero weight mutation. |
| Latent-to-latent preferred | Operates entirely in embedding space; decode/scalar projection only at the boundary. Uses dot-product + **sigmoid** for the trust gate. |
| Freeze/thaw over fine-tuning | `f` is a frozen snapshot. When `coherence` decays (relation drifts), re-estimate — a direction-vector swap, not an in-place weight update. |
| Self-learn welcome | Re-estimation from fresh latent observations is runtime self-improvement of a routing/direction table, not of base weights. |
| Plasma/Hot tiering | Extraction is O(N·dim) adds; apply is one vector add; parallelism is two dots — all SIMD, L1-resident, batchable across thousands of NPCs. |

## API (corrected — target: `riir-engine/src/latent_functor/arithmetic.rs`, feature `latent_functor`)

```rust
/// Estimate functor direction + coherence from N position-aligned analog pairs.
///
/// `sources` and `targets` are flat `[f32]` of length `n_pairs * dim`, position-aligned:
/// pair k is (sources[k*dim..(k+1)*dim], targets[k*dim..(k+1)*dim]).
///
/// Returns (functor_direction, coherence) where:
///   f          = (1/N) Σ_k (target_k − source_k)      // mean displacement
///   coherence  = mean_k cos(target_k − source_k, f)   // parallelism quality (paper §4.2)
///
/// Edge cases: empty (N=0) → (zero vec, 0.0); single pair (N=1) → (displacement, 1.0).
pub fn extract_functor(sources: &[f32], targets: &[f32], dim: usize) -> (Vec<f32>, f32);

/// Zero-alloc variant: writes functor to `f_out` (len dim), returns coherence.
pub fn extract_functor_into(sources: &[f32], targets: &[f32], dim: usize, f_out: &mut [f32]) -> f32;

/// Apply functor by vector addition: out = source + gate · functor.
/// Gate defaults to 1.0 (ungated); use functor_gate() to compute trust from coherence.
pub fn apply_functor(source: &[f32], functor: &[f32], dim: usize, out: &mut [f32]);

/// Paper §4.2 eq. 3: cos(target − source, functor). Quality metric for a candidate pair.
pub fn functor_parallelism(source: &[f32], target: &[f32], functor: &[f32], dim: usize) -> f32;

/// Zero-alloc variant: reuses `disp` (len dim) as scratch for displacement vector.
pub fn functor_parallelism_into(
    source: &[f32], target: &[f32], functor: &[f32], dim: usize, disp: &mut [f32]
) -> f32;

/// Sigmoid-gated trust from coherence. dot-product + sigmoid, NEVER softmax (per AGENTS.md).
/// gate = sigmoid(beta · (coherence − tau)). Default: beta=8.0, tau=0.6.
pub fn functor_gate(coherence: f32, beta: f32, tau: f32) -> f32;
```

**Key corrections vs original 273 API:**
1. **Pair alignment made explicit:** position-aligned flat slices, documented in the doc comment. (Original was ambiguous.)
2. **Added `extract_functor_into`:** zero-alloc variant matching `functor_parallelism_into`. (Original was missing this.)
3. **Coherence formula in the API doc:** `mean_k cos(target_k − source_k, f)`. (Original only had it in prose.)
4. **`apply_functor` documents gate parameter:** the caller composes `gate · functor` or passes ungated `functor`. Clarified that `functor_gate()` is the companion.

## GOAT Proofs (corrected — all must pass before promoting feature)

| # | Proof | Threshold | Concrete setup |
|---|-------|-----------|----------------|
| G1 | Constant-offset pairs → `f ≈ offset`, coherence ≈ 1 | err < 1e-4, coh > 0.999 | `sources = [[1,2,3,4]]`, `targets = [[2,3,4,5]]` (offset `[1,1,1,1]`), dim=4, N=1 |
| G2 | `apply_functor` then parallelism vs true target | `|p − 1|` < 1e-4 | Extract from G1 pairs, apply to held-out source, check parallelism with true target |
| G3 | Unrelated random pairs → low coherence | coh < 0.5 | 20 random source/target pairs, dim=128, no shared offset |
| G4 | Gate monotone in coherence, crosses 0.5 at `tau` | monotone + centered | Sweep coherence 0.0→1.0 at 0.01 steps, beta=8.0, tau=0.6; verify monotone non-decreasing, gate(0.6) ≈ 0.5 |
| G5 | **Ranking preserved:** held-out source maps nearer its true analog than any distractor | `d_true < d_wrong` ∀ distractors | 10 analog pairs with shared offset f. Hold out source s*. Apply f → ĥ. True target t* = s* + f. 3 distractors = random vectors. Verify `‖ĥ − t*‖ < ‖ĥ − d_i‖` for all i. |

Run: `cargo test -p riir-engine --features latent_functor functor_tests`

## What's in scope (this spec) vs out of scope (Plan 303)

**In scope (this spec — the arithmetic layer):**
- The 6 functions above (`extract_functor`, `extract_functor_into`, `apply_functor`, `functor_parallelism`, `functor_parallelism_into`, `functor_gate`)
- GOAT proofs G1–G5

**Out of scope (Plan 303 — runtime/game layers):**
- `FunctorTable` data structure (per-NPC, per-relation, versioned, BLAKE3, papaya lock-free)
- `predict_stance` / `rank_relational_candidates` wrappers
- KG triple emission bridge
- Emotion bridge wiring
- Coherence-decay re-estimation scheduler
- Curiosity integration (Fusion F2)
- Cross-game functor transfer (Super-GOAT confirmation)

## What stays out entirely

- **Full synthetic analogy task** — toy, not valuable (Research 111 "What NOT to Do" #1).
- **Analogy as a "reasoning mode"** bolted onto the transformer (Research 111 #2).
- **LoRA weight-decay / training-dynamics work** → riir-train.
- **Open primitive shipping to katgpt-rs** — per `003`, functor arithmetic is riir-ai private. The Dirichlet Energy diagnostic (Plan 149) is the only public primitive from this paper, and it's already shipped.
