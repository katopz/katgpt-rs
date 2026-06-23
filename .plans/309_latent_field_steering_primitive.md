# Plan 309: Latent Field Steering — Open Primitive

**Date:** 2026-06-23
**Research:** [katgpt-rs/.research/290_latent_field_steering_open_primitive.md](../.research/290_latent_field_steering_open_primitive.md)
**Source:** Synthesized from CAA + Anthropic Functional Emotions + Gemini "wave interference" reframing
**Target:** `katgpt-rs/crates/katgpt-core/src/latent_steering.rs` (new module) + Cargo feature `latent_field_steering`
**Status:** Phase 0–2 COMPLETE (2026-06-23). All 5 GOAT gates PASS — primitive proven, ready for Phase 4 promotion decision. Phase 3 T3.1 DONE (AVX2 SAXPY backend landed, bit-identity verified); T3.2 INCONCLUSIVE — dev host is aarch64 so the AVX2 path is compiled out and the speedup gate cannot be measured here (requires x86_64+AVX2 host). G4 carry-over still PASS (7.1µs with dispatcher). Phase 5 (game integration) deferred to riir-ai Plan 330.

---

## Goal

Ship the minimal concrete prototype of Latent Field Steering: a zero-allocation,
SIMD-accelerated primitive for injecting a frozen direction vector into a mutable
latent state slice, with optional localized support (radius, zone, graph
neighborhood). Prove or kill the Super-GOAT candidate (Research 290) via 5 GOAT
gates measuring steering strength, behavior rank preservation, localization,
crowd-scale performance, and zero-allocation steady state.

**Proves the idea:** G1 ≥30% affect shift · G2 cos(action_rank) ≥0.95 ·
G3 zero leakage at d > b+ε · G4 5000-NPC crowd <1ms · G5 0 allocs after warmup.

**Kills the idea:** G2 <0.95 (steering corrupts decisions) **OR** G3 >0.01
leakage (uncontrolled propagation) **OR** G4 >1ms (too slow for 20Hz tick).

---

## Phase 0 — Design (COMPLETE)

- [x] T0.1 Research note created ([Research 290](../.research/290_latent_field_steering_open_primitive.md))
- [x] T0.2 Private guide created ([riir-ai/.research/153](../../../riir-ai/.research/153_latent_field_steering_game_runtime_guide.md))
- [x] T0.3 Plan created (this file)
- [x] T0.4 Fusion grep complete: zero codebase hits for residual-stream steering on hot path; closest cousins are CNA (neuron-level), EmotionDirections (read-only), FPCG (explicit non-mutation)

---

## Phase 1 — Unblocking Skeleton (CORE)

**Target:** minimal compilable primitive behind feature flag. No perf optimization, no GOAT gate. Ships the trait + scalar impl + smoke tests.

**STATUS: COMPLETE — all 6 smoke tests pass, `cargo check -p katgpt-core --features latent_field_steering` clean.**

### Tasks

- [x] T1.1 Created `katgpt-rs/crates/katgpt-core/src/latent_steering.rs` (437 lines):
  `LatentSteeringVector` (BLAKE3-committed via per-element LE f32, matches
  `engram/commitment.rs` + `cross_resolution.rs` conventions), `LatentSteeringError::{NotUnitNorm,
  AlphaOutOfRange}`, `FieldSupport::{Global, Radius, Zone}`, `LatentField`,
  `apply_latent_steering`, `apply_latent_steering_weighted`, `kernel_weight`,
  `apply_field_to_crowd`, HLA axis index constants (`HLA_VALENCE`..`HLA_FEAR`,
  `HLA_DIM=8`).
- [x] T1.2 Design decision: `Vec<f32>` for dynamically-sized direction (matches
  `EmotionDirections` storage). Documented in module doc. Game-side typed alias
  (`HLAField([f32; 8])`) deferred to riir-ai.
- [x] T1.3 Feature gates added. In `katgpt-core/Cargo.toml`:
  `latent_field_steering = []`. In root `Cargo.toml`:
  `latent_field_steering = ["katgpt-core/latent_field_steering"]`.
- [x] T1.4 Wired module into `katgpt-core/src/lib.rs` with `pub mod` + `pub use`.
- [x] T1.5 Smoke tests (6 in-module tests, all PASS):
  - `smoke_global_field_shifts_state` — verifies `state[i] += alpha * dir[i]` exactly.
  - `smoke_radius_field_localizes` — inside shifted, outside skipped.
  - `smoke_constructor_rejects_non_unit_norm` — norm=2.0 → `NotUnitNorm`.
  - `smoke_constructor_rejects_alpha_out_of_range` — α=1.5 → `AlphaOutOfRange`.
  - `smoke_commitment_roundtrip` — `verify(tol)` returns true.
  - `smoke_zone_field_matches_only_matching_zone` — only matching zone shifts.
- [x] T1.6 `cargo check -p katgpt-core --features latent_field_steering` clean.

---

## Phase 2 — GOAT Gate (PROVES OR KILLS)

Each gate is a standalone file. All must pass to promote from opt-in.

**STATUS: ALL 5 GATES PASS — primitive proven. Ready for Phase 4 promotion decision.**

### Results summary (2026-06-23)

| Gate | Result | Threshold | Verdict |
|---|---|---|---|
| G1 fear-axis shift | ratio **1.50×** (post=1.5, pre=1.0) | ≥ 1.30 | **PASS** |
| G2 mean cos (α=0.3) | **0.9958** | ≥ 0.95 | **PASS** |
| G2 min cos (α=0.3) | **0.9667** | ≥ 0.90 | **PASS** |
| G3 leakage ratio | **0.000045** | < 0.01 | **PASS** |
| G4 crowd p50 | **19.2µs** (5000 NPCs × 8d) | < 1000µs | **PASS** (52× headroom) |
| G5 zero-alloc | **0** allocs / 1000 applies | 0 | **PASS** |

### Key findings

1. **G2 is the headline pass**: at α=0.3, action rankings are preserved with
   mean cos = 0.9958 (gate ≥ 0.95). The primitive does NOT corrupt NPC
   decision-making at moderate steering strength.
2. **G2 argmax flip caveat**: the α-sweep reveals that 8% of NPCs change their
   top-1 action at α=0.3 (12% at α=0.5, 18% at α=0.9). The cosine gate passes
   cleanly, but deployment should be aware that ~1 in 12 NPCs may select a
   different action under steering. This is expected for a 5-action system with
   close scores — not a failure, but a deployment caveat.
3. **G2 α-sweep characterization**:
   - α=0.1: mean 0.9995, 1% flips — very safe
   - α=0.3: mean 0.9958, 8% flips — **gate passes**
   - α=0.5: mean 0.9883, 12% flips — borderline
   - α=0.9: mean 0.9634, min 0.59, 18% flips — dangerous
   **Recommended max α for hot-path deployment: 0.3.**
4. **G3 confirms zero leakage**: sigmoid kernel at distance 15 with bandwidth 10
   produces kernel weight ≈ 4.5e-5, giving leakage ratio 4.5e-5 — far below the
   0.01 gate.
5. **G4 confirms sub-millisecond crowd perf**: 5000 NPCs × 8d in 19.2µs p50
   (release). This is 52× under the 1ms budget — the element-wise SAXPY
   auto-vectorizes well at d=8. No manual SIMD needed (Phase 3 is a no-op).
6. **G5 confirms zero-alloc hot path**: 0 allocations over 1000 crowd-applies.

### Tasks

- [x] T2.1 **G1 — Steering strength.** File:
      `tests/latent_steering_g1_strength.rs`. One-hot fear-axis direction at
      α=0.5, baseline fear=1.0. **Result: post=1.5, ratio=1.50 ≥ 1.30. PASS.**
      Also verifies non-target axes unchanged (|delta| < 1e-5).

- [x] T2.2 **G2 — Behavior rank preservation.** File:
      `tests/latent_steering_g2_rank_preservation.rs`. 100 random 8-dim states,
      8×5 action weights, random unit direction. α-sweep {0.1, 0.3, 0.5, 0.9}.
      **Result at α=0.3: mean cos 0.9958 ≥ 0.95, min cos 0.9667 ≥ 0.90. PASS.**
      Argmax flip rate 8% documented as deployment caveat.

- [x] T2.3 **G3 — Localization.** File: `tests/latent_steering_g3_localization.rs`.
      Radius field (0,0) b=10 s=2.0, 500 inside at d=5, 500 outside at d=15.
      **Result: leakage ratio 0.000045 < 0.01. PASS.**

- [x] T2.4 **G4 — Crowd-scale perf.** File: `tests/latent_steering_g4_crowd_perf.rs`.
      5000 NPCs × 8d, global field. **Result: p50=19.2µs < 1000µs. PASS (52× headroom).**

- [x] T2.5 **G5 — Zero-alloc steady state.** File:
      `tests/latent_steering_g5_zero_alloc.rs`. Debug-only via
      `katgpt_rs::alloc::TrackingAllocator`. **Result: 0 allocations over 1000
      crowd-applies. PASS.**

---

## Phase 3 — SIMD Acceleration (if G1–G5 pass)

- [x] T3.1 Replace the scalar loop in `apply_latent_steering` with manual SIMD
      (8× f32 for d=8 HLA via `std::arch::x86_64::_mm256_add_ps` and
      `_mm256_mul_ps`, with fallback scalar path for non-AVX2 targets).
      **DONE (2026-06-23).** Extracted a shared `saxpy_inplace` dispatcher in
      `katgpt-core/src/latent_steering.rs` (3 call sites now share it:
      `apply_latent_steering`, `apply_latent_steering_weighted`, the
      `apply_field_to_crowd` inner loop). AVX2 backend uses `_mm256_mul_ps` +
      `_mm256_add_ps` (NOT FMA — bit-identical to scalar mul-then-add rounding).
      Runtime dispatch via `is_x86_feature_detected!("avx2")`. Scalar tail handles
      `len % 8` remainder. Unit test `saxpy_simd_matches_scalar` asserts
      bit-equality at d=8 and d=16 across multiple seeds/alphas — PASSES.
      Clean `cargo clippy` and `cargo check` on katgpt-core with the feature.
- [-] T3.2 Benchmark SIMD vs scalar on d=8 and d=16. **Gate:** ≥2× speedup at
      d=8, ≥1.5× at d=16. Re-run G4 with SIMD path; gate still <1ms.
      **DEFERRED (2026-06-23) — GATE CANNOT BE EVALUATED ON DEV HOST.**
      The dev machine is **aarch64 (Apple Silicon)**, so the `#[cfg(target_arch = "x86_64")]`
      AVX2 backend is compiled out and `apply_latent_steering` routes to the
      scalar fallback. The bench harness `tests/latent_steering_t3_simd_vs_scalar.rs`
      runs cleanly but reports scalar-vs-scalar (0.0ns/call at d=8/16 — below timer
      resolution, NaN speedup) — **these numbers are meaningless and must NOT be
      used to satisfy the gate.** To get a real verdict, re-run on an x86_64 host
      with AVX2 (e.g. CI runner, x86 Linux box, or Rosetta-free x86 Mac).
      **Carry-over gates that DID pass on this host:**
      - G4 crowd re-run (SIMD dispatcher, 5000×8): p50=7.1µs < 1ms — PASS
        (dispatcher overhead is invisible at crowd scale).
      - G1–G5 all still PASS with the dispatcher in place (G4=21.9µs, G5=6.8µs/call).
      **Honest expectation when measured on x86_64:** the plan author flagged
      Phase 3 as "likely a no-op" because LLVM already auto-vectorizes the scalar
      SAXPY at `-O3` (Phase 2 G4 was 19.2µs). A 2× gate at d=8 is unlikely to
      pass against an auto-vectorizing scalar baseline — manual AVX2 typically
      only wins when it unlocks instructions the optimizer won't emit (FMA, gather,
      etc.), and the user explicitly required non-FMA mul+add for bit-identity.
      Recommendation: treat the SIMD path as a correctness/ portability asset
      (explicit, no reliance on auto-vec) rather than a perf gate. If the x86_64
      measurement comes back <2×, do NOT promote T3.2 but keep T3.1 (the code is
      correct and may help on targets where auto-vec is disabled, e.g.
      `RUSTFLAGS=-C target-cpu=x86-64` baseline builds).

---

## Phase 4 — Promotion Decision

**STATUS: COMPLETE (2026-06-23) — promoted to DEFAULT-ON per AGENTS.md rule 'GOAT pass → promote to default'.**

G1–G5 all pass with significant headroom. The argmax flip caveat (8% at α=0.3)
is documented in the feature comment and the README — deployment should use
α ≤ 0.3 for hot-path steering.

Changes:
- `katgpt-rs/crates/katgpt-core/Cargo.toml`: added `latent_field_steering` to `default = [...]`.
- `katgpt-rs/Cargo.toml`: added `latent_field_steering` to `default = [...]`.
- `katgpt-rs/README.md`: added showcase section under "Feature Showcase" with GOAT table + argmax caveat.

- [x] T4.1 G1–G5 all pass → promoted to opt-in default in katgpt-rs feature showcase; documented in README.
- [x] T4.2 G2 mean cos ≥ 0.95 — confirmed (0.9958). No demotion.
- [x] T4.3 G2 min cos ≥ 0.90 — confirmed (0.9667). No demotion.
- [x] T4.4 Argmax flip rate (8% at α=0.3) documented as deployment caveat — not a gate failure.

---

## Phase 5 — Game Integration (DEFERRED to riir-ai)

**Status (2026-06-23):** Phase 2 G1–G2 PASS — primitive proven and promoted to default in katgpt-rs. Game-side wiring deferred to riir-ai Plan 330. See
[riir-ai/.research/153](../../../riir-ai/.research/153_latent_field_steering_game_runtime_guide.md)
for the integration guide.

- [x] T5.1 HLA post-evolve wiring. **DONE (riir-ai):** `ReconstructionState::hla_mut()` accessor added to katgpt-core (commit `094854e9`); `FieldRegistry` + post-`evolve_hla` steering pass landed in `riir-engine/src/latent_field_wiring.rs` behind `latent_field_wiring` feature (papaya-backed zone→field registry, `apply_to_reconstruction` runs the additive overlay after evolve). 10/10 tests pass.
- [-] T5.2 CWM soft-rule → field mapping. **BLOCKED on missing primitive.** No soft/hard rule distinction exists in `induced_cwm`/`cwm_runtime` — `InducedCwmKernel` is a monolithic forward-model trait with no `soft: bool` field. Research 153's "soft rules become steering fields" is aspirational; doing this honestly requires first designing + shipping a soft-rule taxonomy (separate plan), not a wiring task. Deferred — file as a design issue, not silently skipped.
- [x] T5.3 Faction "battle stance" frozen field. **DONE (riir-ai):** `FactionStanceRegistry` in `riir-engine/src/latent_field_wiring.rs` — atomic `Arc` swap via papaya (`set_stance`/`stance`), readers hold consistent snapshots. **Partial vs Research 153:** the hot-path atomic-swap runtime IP ships now; cold-tier `MerkleFrozenEnvelope<LatentSteeringVector>` persistence is a documented follow-up (riir-neuron-db is not currently a riir-engine dep — adding it just for the envelope is scope creep; the BLAKE3 `commitment` on `LatentSteeringVector` is already syncable raw if a game needs to commit the stance hash).
- [x] T5.4 `GrudgeMemory` (Plan 317) integration. **DONE (riir-ai, commit `fdd24182`):** `grudge_to_field` + `apply_grudge_steering` in `riir-games/src/game_traits/grudge_field.rs` — emits a fear-axis `LatentField` (Radius support, centered on target, intensity×anger-scaled α) when a grudged target is within `visible_radius`; None for unknown/far/decayed grudges. 10/10 tests pass.

---

## Cross-Refs

- [katgpt-rs/.research/290_latent_field_steering_open_primitive.md](../.research/290_latent_field_steering_open_primitive.md) — research note
- [riir-ai/.research/153_latent_field_steering_game_runtime_guide.md](../../../riir-ai/.research/153_latent_field_steering_game_runtime_guide.md) — private guide
- [katgpt-rs/.plans/162_emotion_vector_inference_control.md](162_emotion_vector_inference_control.md) — read-only counterpart
- [katgpt-rs/.plans/087_cna_contrastive_neuron_attribution.md](087_cna_contrastive_neuron_attribution.md) — neuron-level mutation counterpart
- [katgpt-rs/.plans/286_functional_attention_spectral_transport.md](286_functional_attention_spectral_transport.md) — F2 fusion target (cross-domain steering)
- [katgpt-rs/.plans/297_personality_weighted_composition.md](297_personality_weighted_composition.md) — F3 cousin (composition vs overlay)
