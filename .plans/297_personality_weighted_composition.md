# Plan 297: Personality-Weighted Latent Layer Composition (Open Primitive)

**Date:** 2026-06-21
**Research:** [katgpt-rs/.research/276_Personality_Weighted_Latent_Layer_Composition.md](../.research/276_Personality_Weighted_Latent_Layer_Composition.md)
**Cross-ref (riir-ai):** [Research 146](../../../riir-ai/.research/146_Entity_Cognition_Stack_Guide.md), [Plan 327](../../../riir-ai/.plans/327_entity_cognition_stack_runtime.md) (runtime wiring)
**Target:** `katgpt-rs/crates/katgpt-core/src/personality_composition/` (new module) + Cargo feature `personality_composition`
**Status:** Complete — Phases 1-5 done, GOAT G4/G5 PASS, promoted to default-on

---

## Goal

Ship the generic, modelless, MIT-licensed open half of the Entity Cognition Stack Super-GOAT (R146/R276). A `PersonalityWeightedComposition<N, D>` kernel that composes `N` latent direction vectors via `Σᵢ sigmoid(wᵢ/τ) · belief_confidence_i · dᵢ`, plus a drift rule `Δwᵢ = α(R_obs − R_exp)·d_recent` with clamp, plus the `LayerDirectionSource` trait. Zero-allocation, sigmoid-gated (per AGENTS.md — never softmax), belief-gated, snapshot-integrated. No game semantics — applies to NPC, player, predator, prey, robot, recommender user.

**GOAT gate (per AGENTS.md):** feature flag `personality_composition` + G4 (<1µs/entity compose) + G5 (zero heap alloc) must pass before promoting to default features.

---

## Phase 1 — Unblocking Skeleton (CORE)

Goal: a compiling, feature-gated module with the type surface frozen. No drift yet — just the composition kernel.

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/personality_composition/` directory with empty `mod.rs`
- [x] **T1.2** Add feature flag `personality_composition = []` to `katgpt-rs/crates/katgpt-core/Cargo.toml` features section (alphabetical)
- [x] **T1.3** Add `#[cfg(feature = "personality_composition")] pub mod personality_composition;` to `katgpt-rs/crates/katgpt-core/src/lib.rs` (alphabetical, after `peira` or similar)
- [x] **T1.4** Implement `personality_composition/types.rs`:
  - [x] `PersonalityConfig` struct (tau, alpha, w_max, ema_decay)
  - [x] `ArchetypeLabel` newtype (opaque label that seeds initial `w`; not interpreted by the kernel)
  - [x] Default impl: `tau = 1.0`, `alpha = 0.01`, `w_max = 5.0`, `ema_decay = 0.95`
- [x] **T1.5** Implement `personality_composition/sigmoid.rs` — numerically stable sigmoid helper:
  - [x] `pub fn sigmoid(x: f32) -> f32` — branching impl per AGENTS.md (positive vs negative branch, no overflow)
  - [x] Vectorized variant `sigmoid_into(x: &[f32], out: &mut [f32])` for batch
- [x] **T1.6** Implement `personality_composition/trait.rs`:
  - [x] `pub trait LayerDirectionSource` with `direction(&self, scratch: &mut [f32]) -> &[f32]`
  - [x] `fn recent_direction(&self) -> &[f32]` (default: returns current direction)
  - [x] `fn belief_confidence(&self) -> f32 { 1.0 }` (default: plasma-tier layers)
- [x] **T1.7** Implement `personality_composition/kernel.rs` — the composition struct:
  - [x] `pub struct PersonalityWeightedComposition<const N: usize, const D: usize>` with `w: [f32; N]`, config, `r_expected: [f32; N]`
  - [x] `pub fn new(config: PersonalityConfig, initial_w: [f32; N]) -> Self`
  - [x] `pub fn compose_into(&self, layers: &[&dyn LayerDirectionSource; N], scratch: &mut [f32], out: &mut [f32]) -> &mut [f32]` — the kernel: `out[j] += sigmoid(w[i]/tau) * belief_confidence[i] * d[i][j]`
  - [x] `debug_assert_eq!(out.len(), D)`, `debug_assert!(scratch.len() >= D)`

### Validation

- [x] **T1.V1** `cargo build --features personality_composition` compiles cleanly
- [x] **T1.V2** `cargo test --features personality_composition` — write `personality_composition/tests.rs` with: `compose_zero_weights_uniform`, `compose_extreme_positive_weight_selects_layer`, `compose_extreme_negative_weight_zeros_layer`, `compose_belief_confidence_decay_shrinks_contribution`

---

## Phase 2 — Drift Rule + Clamp + EMA

Goal: the modelless adaptation — weights move with reward surprise.

### Tasks

- [x] **T2.1** Add `pub fn drift(&mut self, layers: &[&dyn LayerDirectionSource; N], r_observed: f32)` to `PersonalityWeightedComposition`:
  - [x] For each layer i: `surprise = r_observed - r_expected[i]`
  - [x] For each dim j: `w[i] += alpha * surprise * d_recent[i][j]`
  - [x] `w[i] = w[i].clamp(-w_max, w_max)`
  - [x] `r_expected[i] = ema_decay * r_expected[i] + (1 - ema_decay) * r_observed`
- [x] **T2.2** Add `pub fn w_snapshot(&self) -> &[f32; N]` — read-only access for snapshot integration
- [x] **T2.3** Add `pub fn restore_w(&mut self, w: [f32; N])` — for snapshot thaw

### Validation

- [x] **T2.V1** Unit tests in `tests.rs`: `drift_positive_surprise_reinforces`, `drift_negative_surprise_penalizes`, `drift_clamps_to_w_max`, `drift_ema_tracks_recent_reward`

---

## Phase 3 — Snapshot Integration

Goal: extend `MicroRecurrentKernelSnapshot` (R242) to carry the `w` vector + archetype label.

### Tasks

- [x] **T3.1** Add `personality_composition/snapshot.rs`:
  - [x] `pub struct PersonalitySnapshot<const N: usize>` with `w: [f32; N]`, `archetype: ArchetypeLabel`, `version: u64`, `blake3: [u8; 32]`
  - [x] `pub fn from_composition(composition: &PersonalityWeightedComposition<N, D>, archetype: ArchetypeLabel, version: u64) -> Self` — hashes `w` + archetype into BLAKE3
  - [x] `pub fn verify_blake3(&self) -> bool`
- [x] **T3.2** G6 unit test: build snapshot from composition → mutate `w` → verify BLAKE3 mismatch → restore from snapshot → verify BLAKE3 matches
- [x] **T3.3** Version test: two snapshots with same `w` but different `version` → BLAKE3 identical (version is metadata, not personality contents, per R242 §snapshot.rs precedent)

### Validation

- [x] **T3.V1** All snapshot tests pass

---

## Phase 4 — Performance Hardening (Plasma Tier)

Goal: prove G4 (<1µs/entity) and G5 (zero heap allocation).

### Tasks

- [x] **T4.1** Audit `compose_into` for zero-allocation:
  - [x] No `Vec` allocations in the hot path
  - [x] All scratch buffers caller-provided
  - [x] `#[inline]` on inner sigmoid+mul loop
- [x] **T4.2** SIMD auto-vectorization:
  - [x] Verify LLVM auto-vectorizes the inner `for j in 0..D { out[j] += weight * d[j] }` loop (check `cargo asm`)
  - [x] If not auto-vectorized, use `katgpt-core::simd::simd_fused_scale_acc` or similar
- [x] **T4.3** Add `benches/personality_composition_bench.rs` (criterion):
  - [x] `compose_n9_d32` — N=9, D=32 (the production case)
  - [x] `compose_n9_d32_batch_10k` — 10K entities per tick (the crowd-scale case)
  - [x] `drift_n9_d32` — drift update cost

### GOAT Gate (must pass before promote to default)

- [x] **G4-GOAT** `compose_n9_d32` < 1µs per entity (plasma tier)
- [x] **G5-GOAT** Heap profiler confirms 0 allocations in `compose_into` hot path (use `dhat` or `valgrind --tool=massif`)
- [x] **G1** `compose_tau_infinity_uniform` — when `tau → ∞`, all weights contribute 0.5, output = `0.5 * Σ dᵢ` (no personality); divergence only with finite `tau`
- [x] **G4.V1** If GOAT fails, demote: keep feature flag, do NOT add to default features, create `.issues/NNN_personality_composition_perf.md` with the bottleneck

---

## Phase 5 — Promotion & Documentation

Goal: feature flag → default; README updated.

### Tasks

- [x] **T5.1** If G4/G5 pass: add `personality_composition` to default features in `katgpt-rs/Cargo.toml`
- [x] **T5.2** Add Feature Showcase section to `katgpt-rs/README.md` (1 paragraph + the kernel formula + the trait surface)
- [x] **T5.3** Add `examples/personality_composition_01_basic.rs` — minimal demo: 3 layers, drift over 100 ticks, show weights converge
- [x] **T5.4** Add `examples/personality_composition_02_taming.rs` — minimal demo: wildlife entity, food reward, show `w_COMPANIONS(player)` rises above `τ_tame` (the open-primitive half of the taming story; the species-swap happens in the host/riir-ai)
- [x] **T5.5** Cross-link `katgpt-rs/.research/276_*.md` to this plan in the "Status" line

---

## Risk Register

| Risk | Mitigation |
|---|---|
| N as `const generic` blows up compile times for many N values | Pin `N` to a small enum-like set (1, 4, 7, 9) via `PersonalityWeightedComposition` aliases in `types.rs`; host picks the right one |
| BLAKE3 of `w` is too cheap (collisions at scale) | Per R242 precedent, `w` is just contents — version + archetype disambiguate; collisions within one entity's lifetime are astronomically unlikely |
| Drift is too slow per-tick at 10K entities | ICT branching gate (R142) lives in riir-ai host, not here — kernel stays O(N·D) |
| Hosts abuse `belief_confidence = 0` to disable a layer | That's the point — sigmoid(0) gives 0.5 contribution anyway; pure 0 requires `w → -∞`, which the clamp prevents |

---

## Cross-References

- **Research:** [276 (this plan's parent)](../.research/276_Personality_Weighted_Latent_Layer_Composition.md)
- **Companion (riir-ai):** [Research 146](../../../riir-ai/.research/146_Entity_Cognition_Stack_Guide.md), [Plan 327](../../../riir-ai/.plans/327_entity_cognition_stack_runtime.md) (runtime wiring)
- **Depends on:** R242 (`MicroRecurrentKernelSnapshot` in `micro_belief/snapshot.rs`) — Phase 3 extends this
- **Does NOT depend on:** game systems (entity-agnostic), chain (host responsibility), LatCal (host responsibility)
