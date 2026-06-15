# Plan 273: Latent Functor — Analogy as Vector Arithmetic

**Date:** 2026-06-15
**Status:** ⚠️ **SUPERSEDED — spec valid, target repo wrong per `003`.** See [riir-ai/.plans/303_latent_functor_runtime_npc_relational_learning.md](../../riir-ai/.plans/303_latent_functor_runtime_npc_relational_learning.md) for the corrected plan.

> **Why superseded:** This plan targeted `katgpt-core/src/dirichlet.rs` (public, MIT). Per [`003_Commercial_Open_Source_Strategy_Verdict.md`](../.research/003_Commercial_Open_Source_Strategy_Verdict.md), latent-operation internals (projection directions + sigmoid gate tuning) are explicitly **riir-ai private** — "the efficiency multiplier, keep private." The functor arithmetic (`extract_functor`, `apply_functor`, `functor_parallelism`, `functor_gate`) IS a projection direction + sigmoid gate. It was (correctly) never shipped to katgpt-rs. Plan 303 (riir-ai) implements the spec in the correct repo.
>
> **What stays valid from this plan:** The API signatures, GOAT proofs G1-G5, and the modelless constraint analysis. These are reused in Plan 303.
>
> **What already shipped (unchanged):** The Dirichlet Energy diagnostic (`dirichlet_energy`, `functor_adjacency`, `consecutive_adjacency`, `kv_cache_dirichlet_energy`) from Plan 149 — this IS public-safe (pure measurement, not a latent operation) and stays in katgpt-rs.

**Research:** 111 (Emergent Analogical Reasoning in Transformers, arXiv:2602.01992)
**Related:** Plan 149 (Dirichlet Energy diagnostic — sibling primitive from same paper, SHIPPED), Research 231 / Plan 264 (`sparse_task_vector` — *weight*-space deltas, distinct), Research 144 / Plan 162 (linear emotion direction vectors)
**Feature Gate:** `dirichlet_energy` (opt-in, katgpt-rs open) — reused, no new gate
**Superseded by:** [riir-ai Plan 303](../../riir-ai/.plans/303_latent_functor_runtime_npc_relational_learning.md)

---

## Task Index

- [x] T1: `extract_functor()` — estimate functor direction + coherence from analog pairs
- [x] T2: `apply_functor()` — analogical map `out = source + f`
- [x] T3: `functor_parallelism()` / `_into()` — paper's `cos(target − source, f)` quality metric (alloc + zero-alloc)
- [x] T4: `functor_gate()` — sigmoid-gated trust from coherence
- [x] T5: GOAT proofs G1–G5 (inline `functor_tests`)

## Goal

Research 111 distilled the paper into **two** open-engine primitives:
1. **Dirichlet Energy** — structural-alignment diagnostic → built in Plan 149. ✅
2. **Functor direction probe** — the mechanism `e_target ≈ e_source + f` → flagged
   open in Research 111 ("Functor direction probe | katgpt-rs (open)") but **deferred**
   by Plan 149, which only shipped the diagnostic.

This plan closes that gap: lift the paper's residual-stream analogy mechanism out of
the network into a **modelless, latent-to-latent primitive**. No weights, no training —
just direction vectors over latents:

- **Estimate** a functor `f = mean_k(target_k − source_k)` from observed analog pairs.
- **Apply** it by addition: `ĥ_target = source + f` (predict a novel source's analog).
- **Verify** a candidate pair with parallelism `cos(target − source, f)` (Research 111 §3).
- **Gate** application by a sigmoid of the functor's coherence (dot + sigmoid, never softmax).

It sits in `crates/katgpt-core/src/dirichlet.rs` beside its sibling diagnostic, behind the
same `dirichlet_energy` feature.

## Why this is in scope (and modelless)

| Constraint | How this satisfies it |
|---|---|
| Modelless / inference-time | `f` is estimated from latents at runtime; zero backprop, zero weight mutation. |
| Latent-to-latent preferred | Operates entirely in embedding space; decode/scalar projection only at the boundary. Uses dot-product + **sigmoid** for the trust gate. |
| Freeze/thaw over fine-tuning | `f` is a frozen snapshot. When `coherence` decays (relation drifts), re-estimate — a direction-vector swap, not an in-place weight update. |
| Self-learn welcome | Re-estimation from fresh latent observations is runtime self-improvement of a routing/direction table, not of base weights. |
| Plasma/Hot tiering | Extraction is O(N·dim) adds; apply is one vector add; parallelism is two dots — all SIMD, L1-resident, batchable across thousands of NPCs. |

## Game / MMORPG application (generic here, game-specific stays in riir-ai)

The open primitive is just "latent analogy via direction vectors." The runtime use —
an NPC that learned a *relational* displacement (`f_betrayal = h(enemy) − h(former_ally)`)
applying it to predict its stance toward a **new** entity (`ĥ = h(newcomer) + f_betrayal`)
without decoding or retraining — is a riir-ai concern. `coherence` doubles as a
**collapse-detection** signal: if an NPC population's relational displacements stop being
parallel (coherence → 0), the functor is stale → trigger re-estimation. Cross-game functor
transfer (Bomber↔FFT) remains Super-GOAT in riir-ai.

## API (crates/katgpt-core/src/dirichlet.rs, gated `dirichlet_energy`)

```rust
pub fn extract_functor(sources: &[f32], targets: &[f32], dim: usize) -> (Vec<f32>, f32); // (f, coherence)
pub fn apply_functor(source: &[f32], functor: &[f32], dim: usize, out: &mut [f32]);
pub fn functor_parallelism(source: &[f32], target: &[f32], functor: &[f32], dim: usize) -> f32;
pub fn functor_parallelism_into(source: &[f32], target: &[f32], functor: &[f32], dim: usize, disp: &mut [f32]) -> f32;
pub fn functor_gate(coherence: f32, beta: f32, tau: f32) -> f32; // sigmoid(beta·(coherence − tau))
```

## GOAT Proofs (inline `functor_tests`, all PASS)

| # | Proof | Threshold | Result |
|---|-------|-----------|--------|
| G1 | Constant-offset pairs → `f ≈ offset`, coherence ≈ 1 | err < 1e-4, coh > 0.999 | ✅ |
| G2 | `apply_functor` then parallelism vs true target | `\|p − 1\|` < 1e-4 | ✅ |
| G3 | Unrelated random pairs → low coherence | coh < 0.5 | ✅ |
| G4 | Gate monotone in coherence, crosses 0.5 at `tau` | monotone + centered | ✅ |
| G5 | **Ranking preserved**: functor maps held-out source nearer its true analog than any distractor | `d_true < d_wrong` ∀ | ✅ |

Run: `cargo test -p katgpt-core --lib --features dirichlet_energy functor_tests`

## What stays out

- **Full synthetic analogy task** — it's a toy (Research 111 "What NOT to Do" #1); only the
  primitive is valuable.
- **Analogy as a "reasoning mode"** bolted onto the transformer (Research 111 #2).
- **LoRA weight-decay / training-dynamics work** → riir-ai Plan 146 (private) / riir-train.
- **Cross-game functor extraction** → riir-ai (Super-GOAT, private).
