# Plan 276: MicroRecurrentBeliefState — Implicit Per-Entity State Tracking Kernel

**Date:** 2026-06-15
**Research:** [katgpt-rs/.research/242_Topological_State_Tracking_Recurrent_Belief.md](../.research/242_Topological_State_Tracking_Recurrent_Belief.md)
**Private guide (Super-GOAT):** [riir-ai/.research/127_Implicit_Microcognition_Crowd_NPC_Guide.md](../../../riir-ai/.research/127_Implicit_Microcognition_Crowd_NPC_Guide.md)
**Source paper:** [arXiv:2604.17121](https://arxiv.org/abs/2604.17121) — Mozer, Siddiqui, Liu (DeepMind, Jun 2026), "The Topological Trouble With Transformers"
**Target:** `katgpt-rs/src/micro_belief/` (new module) + Cargo feature `micro_belief`
**Status:** Active — Phase 0 (planning)

---

## Goal

Ship a generic, modelless, freeze/thaw-compatible **per-entity recurrent belief-state kernel** in katgpt-rs. The kernel implements one step of `s_t = f(s_{t-1}, x_t)` in a fixed-size latent vector, in three recurrence families drawn from Mozer et al.'s taxonomy (attractor loop, latent-thought loop, delta-rule SSM). This is the open primitive for the Super-GOAT fusion documented in `riir-ai/.research/127` — the generic math with no game semantics.

**GOAT gate (G1, must pass to promote to default-on):**
- G1.1 Determinism (bit-identical `s_T` for fixed input sequence)
- G1.2 Boundedness (`‖s_t‖` stays bounded over 10k ticks; Family A doesn't diverge)
- G1.3 Bridge ranking preservation (scalar projection preserves belief ranking)
- G1.4 Latency (Family A ≤ 100ns/NPC/tick CPU SIMD; ≤ 50ns ANE batch)
- G1.5 Freeze/thaw atomicity (readers never see torn kernel swap)

If G1 passes → promote `micro_belief` to default-on. If G1.2 (stability) fails for Family A, fall back to Family C (always-stable linear) as default and gate Family A behind a sub-flag.

**Out of scope for this plan (stays in riir-ai/.plans/304):** NPC tick integration, sense-composition wiring, the 5 emotion channels, ANE batch dispatch, CGSP fusion, collapse detector. This plan ships *only* the generic kernel + snapshot + bridge math.

---

## Phase 0 — Pre-flight (this plan)

### Tasks

- [x] **T0.1** Research note `katgpt-rs/.research/242_*.md` created.
- [x] **T0.2** Private guide `riir-ai/.research/127_*.md` created (Super-GOAT mandatory output).
- [x] **T0.3** This plan created.
- [ ] **T0.4** Audit existing freeze/thaw snapshot infra: locate `LoRAWeightVersion`, `LoRAHotSwap`, BLAKE3 commit path. Confirm `MicroRecurrentKernelSnapshot` can reuse the same atomic-swap plumbing without forking it. (Output: a 1-paragraph note in this plan's §Notes identifying the exact trait/struct to extend.)
- [ ] **T0.5** Audit existing `latent_to_raw_scalar` / `sigmoid(dot())` bridge (per Plan 262 `curator_bridge.rs`). Confirm the bridge function signature matches what `MicroRecurrentBeliefState::project_to_scalars()` needs.

---

## Phase 1 — Core Skeleton + Family A (Attractor Loop)

**Unblocks:** G1.1, G1.2, G1.3, G1.4 (partial), G1.5 (partial). This is the GOAT-gate phase.

### Architecture

```text
katgpt-rs/src/micro_belief/
├── mod.rs                  // Module root, re-exports
├── types.rs                // MicroRecurrentBeliefState trait, Family enum, KernelConfig
├── attractor.rs            // Family A: s_t = σ(W_s·s_{t-1} + W_x·x_t + b)
├── delta_rule.rs           // Family C: s_t = diag(1-α)·s_{t-1} + β·x_t   [Phase 2]
├── latent_thought.rs       // Family B: K iters of Family A                 [Phase 3]
├── snapshot.rs             // MicroRecurrentKernelSnapshot (BLAKE3, versioned)
├── bridge.rs               // project_to_scalars(): sigmoid(dot(s, d)) for k channels
└── tests.rs                // G1.1–G1.5 GOAT tests + property tests
```

### Tasks

- [ ] **T1.1** `types.rs`: define `MicroRecurrentBeliefState` trait
  ```rust
  pub trait MicroRecurrentBeliefState: Send + Sync {
      /// Belief vector dimension (fixed at construction).
      fn dim(&self) -> usize;

      /// Advance one tick: s_t = f(s_{t-1}, x_t). In-place update of `state`.
      /// Zero-allocation: no Vec creation; operates on the &mut [f32] slice.
      fn step(&self, state: &mut [f32], input: &[f32]);

      /// Bridge: project belief vector to K bounded scalars via sigmoid(dot).
      /// `directions` is `[K][dim]`, `out` is `&mut [f32; K]`.
      fn project_to_scalars(&self, state: &[f32], directions: &[[f32; /*dim*/]], out: &mut [f32]);

      /// Family identifier (for routing, snapshot versioning).
      fn family(&self) -> RecurrenceFamily;
  }

  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  #[repr(u8)]
  pub enum RecurrenceFamily { Attractor = 0, LatentThought = 1, DeltaRule = 2 }
  ```
- [ ] **T1.2** `types.rs`: `KernelConfig { dim: usize, family: RecurrenceFamily, ... }` with builder. Default `dim = 32` (fits L1, matches Plan 255 budget).
- [ ] **T1.3** `attractor.rs`: `AttractorKernel { ws: [[f32; D]; D], wx: [[f32; D]; D], b: [f32; D] }` (use `#![feature(generic_const_exprs)]` if stable, else `const D: usize = 32` default + macro for other dims).
  - `step()`: compute `σ(W_s·s + W_x·x + b)` elementwise, write back to `state`.
  - SIMD via existing `wide` crate or std::simd; chunked 4 or 8 lanes for auto-vec.
  - Clamp `state[i]` to `[-CLAMP, CLAMP]` after update (CLAMP=6.0 default — sigmoid saturates by then anyway).
- [ ] **T1.4** `bridge.rs`: `project_to_scalars(state, directions, out)` — for each k, `out[k] = fast_sigmoid(dot(state, &directions[k]))`. Reuse existing `fast_sigmoid` and dot-product helpers (grep for them first — Plan 262 / `curator_bridge.rs` likely has both).
- [ ] **T1.5** `snapshot.rs`: `MicroRecurrentKernelSnapshot { family, dim, weights_blob: Vec<u8>, blake3: [u8; 32], version: u64 }`.
  - `commit(&self) -> [u8; 32]` — BLAKE3 over `(family, dim, weights_blob)`.
  - `verify(&self) -> bool` — recompute and compare.
  - Serialization via existing `serde` + `bincode` pattern (match whatever `LoRAWeightVersion` uses).
- [ ] **T1.6** `mod.rs`: re-export public API, register module behind `micro_belief` feature flag in `lib.rs`.
- [ ] **T1.7** `Cargo.toml`: add `micro_belief` feature, default-off until G1 passes. Dependencies: `blake3` (already in tree), `serde` (already), no new deps.
- [ ] **T1.8** `tests.rs` — **G1.1 Determinism**:
  ```rust
  #[test] fn g1_1_determinism() {
      let kernel = AttractorKernel::from_seed(42, 32);
      let mut s_a = vec![0.0f32; 32];
      let mut s_b = vec![0.0f32; 32];
      let xs: Vec<Vec<f32>> = (0..1000).map(|i| deterministic_input(i)).collect();
      for x in &xs { kernel.step(&mut s_a, x); }
      for x in &xs { kernel.step(&mut s_b, x); }
      assert_eq!(s_a, s_b); // bit-identical
  }
  ```
- [ ] **T1.9** `tests.rs` — **G1.2 Boundedness**:
  ```rust
  #[test] fn g1_2_boundedness() {
      let kernel = AttractorKernel::from_seed(42, 32);
      let mut s = vec![0.0f32; 32];
      let mut rng = ChaCha8Rng::seed_from_u64(7);
      for _ in 0..10_000 {
          let x: Vec<f32> = (0..32).map(|_| rng.gen_range(-1.0..1.0)).collect();
          kernel.step(&mut s, &x);
          for v in &s { assert!(*v >= -6.0 && *v <= 6.0, "attractor diverged"); }
      }
  }
  ```
- [ ] **T1.10** `tests.rs` — **G1.3 Bridge ranking preservation** (property test):
  ```rust
  #[quickcheck] fn g1_3_ranking(sa: Vec<f32>, sb: Vec<f32>, d: Vec<f32>) -> bool {
      let (sa, sb, d) = pad_to_dim(sa, sb, d, 32);
      let dot_a = dot(&sa, &d); let dot_b = dot(&sb, &d);
      let sig_a = sigmoid(dot_a); let sig_b = sigmoid(dot_b);
      (dot_a.partial_cmp(&dot_b) == sig_a.partial_cmp(&sig_b))
  }
  ```
- [ ] **T1.11** `tests.rs` — **G1.4 Latency** (criterion benchmark, gated):
  ```rust
  #[cfg(feature = "bench")] #[bench] fn g1_4_attractor_step_32(b: &mut Bencher) {
      let kernel = AttractorKernel::from_seed(42, 32);
      let mut s = vec![0.0f32; 32]; let x = vec![0.5f32; 32];
      b.iter(|| kernel.step(black_box(&mut s), black_box(&x)));
      // Assert ns < 100 in the GOAT-gate CI job.
  }
  ```
- [ ] **T1.12** `tests.rs` — **G1.5 Freeze/thaw atomicity** (stress test, reuses existing `LoRAHotSwap` test harness if it has one; else write minimal):
  ```rust
  #[test] fn g1_5_snapshot_atomicity() {
      // 1000 reader threads call step() in a tight loop;
      // 1 swapper thread hot-swaps the kernel snapshot every 100ms;
      // assert no reader ever sees a torn read (panic / NaN / dimension mismatch).
  }
  ```
- [ ] **T1.13** Run `cargo test --features micro_belief` — all G1 tests green.
- [ ] **T1.14** Run `cargo bench --features micro_belief,bench` — capture G1.4 numbers, paste into `katgpt-rs/.benchmarks/276_micro_belief_goat.md`.
- [ ] **T1.15** Write `katgpt-rs/.benchmarks/276_micro_belief_goat.md` with the GOAT proof (G1.1–G1.5 pass/fail table + latency numbers).

### GOAT Gate Decision (end of Phase 1)

- [ ] **T1.16** If G1.1–G1.5 all pass → flip `micro_belief` to default-on in `Cargo.toml`. Update `.docs/01_overview.md` Feature Flags table.
- [ ] **T1.17** If G1.2 (stability) fails for Family A but Family C (Phase 2) passes → keep Family A behind `micro_belief_attractor` sub-flag, default to Family C. Document in `types.rs` doc-comment.
- [ ] **T1.18** If G1.4 (latency) fails (>100ns) → profile with `perf record` / `Instruments`, identify bottleneck (likely SIMD lane width or memory layout), file as issue in `katgpt-rs/.issues/`.

---

## Phase 2 — Family C (Delta-Rule SSM) — Always-Stable Fallback

**Why:** Family A (attractor) can diverge if `W_s` has unstable eigenvalues. Family C is linear with per-channel gates `α, β ∈ [0,1]` — always stable, always bounded. This is the safety net and the ANE-batch-friendly variant (pure elementwise + axpy).

### Tasks

- [ ] **T2.1** `delta_rule.rs`: `DeltaRuleKernel { alpha: [f32; D], beta: [f32; D] }`.
  - `step()`: for each i, `state[i] = (1.0 - alpha[i]) * state[i] + beta[i] * input[i]`.
  - Pure elementwise — trivially SIMD, trivially ANE-batchable.
  - `α=0, β=1` = pure integrator (accumulates input — good for "memory of past encounters").
  - `α=1, β=0` = no update (frozen state).
  - `α=λ, β=0` = static decay (matches today's `sigmoid(-λΔt)` when composed with sigmoid bridge — backward-compatible fallback).
- [ ] **T2.2** Extend `MicroRecurrentBeliefState` impl for `DeltaRuleKernel`.
- [ ] **T2.3** Tests: G1.1, G1.2 (trivially passes — linear + bounded gates), G1.3, G1.4 (should be faster than Family A), G1.5.
- [ ] **T2.4** Backward-compat test: `DeltaRuleKernel { alpha: [λ; D], beta: [0; D] }` composed with sigmoid bridge produces output within ε of today's `SpatialBelief::decay_confidence()` for the same Δt. (Validates the "upgrade target" claim in Plan 262.)

---

## Phase 3 — Family B (Latent-Thought Loop) + Composability

**Why:** Family B (K iterations of Family A before advancing) is for "deliberation ticks" — negotiation, planning, multi-step social reasoning. Opt-in; not on the critical path for G1.

### Tasks

- [ ] **T3.1** `latent_thought.rs`: `LatentThoughtKernel { inner: AttractorKernel, k_iters: u8 }`.
  - `step()`: apply `inner.step()` K times with the same input `x_t`. K=1 reduces to Family A.
- [ ] **T3.2** Tests: same G1 suite. Add G1.6: K=1 case bit-identical to Family A with same weights.
- [ ] **T3.3** Composability test: a `TrainingFreeLoop` (Plan 136) wrapping a model that contains a `MicroRecurrentBeliefState` stage works end-to-end. (Validates the "composable, not redundant" claim in Research 242 §2.3.)

---

## Phase 4 — Docs + Examples

### Tasks

- [ ] **T4.1** `katgpt-rs/.docs/NN_micro_belief.md` — API reference (trait, families, snapshot, bridge).
- [ ] **T4.2** `katgpt-rs/examples/micro_belief_demo.rs` — minimal example: construct a kernel, run 1000 steps, project to 3 scalars, print. Shows the full lifecycle.
- [ ] **T4.3** Update `.docs/01_overview.md` Feature Flags table with `micro_belief` row.
- [ ] **T4.4** Update `.docs/02_architecture.md` with the new `micro_belief/` module entry.

---

## Phase 5 — GOAT Promotion + Commit

### Tasks

- [ ] **T5.1** If G1 gate passes → flip `micro_belief` to default-on (T1.16). Run full `cargo test` (all features) to confirm no regressions.
- [ ] **T5.2** Update `.docs/01_overview.md` to mark `micro_belief` as default-on with GOAT proof reference (`.benchmarks/276_*.md`).
- [ ] **T5.3** Commit with `feat:` prefix on `develop` branch (per AGENTS.md).
- [ ] **T5.4** Mark all `- [ ]` tasks in this plan as `- [x]` when complete.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **R1: Family A diverges** (G1.2 fails) | Clamp after update (T1.3); fall back to Family C (Phase 2) as default; gate Family A behind sub-flag. |
| **R2: G1.4 latency > 100ns** | Profile; likely fix is memory layout (SoA vs AoS) or wider SIMD lanes. File issue if not fixable in 1-2 attempts. |
| **R3: Freeze/thaw atomicity hard to extend** (T0.4 reveals `LoRAHotSwap` is LoRA-specific) | Either generalize the trait in `LoRAHotSwap`, or write a parallel `KernelHotSwap` reusing the same primitives. Decide in T0.4. |
| **R4: Bridge function signature mismatch** (T0.5) | Adapt `project_to_scalars` to match existing `latent_to_raw_scalar`; or extract a shared trait. |
| **R5: Generic const expr (`[f32; D]`) not stable** | Use `Vec<f32>` internally with `dim` checked at construction; or macro-generate for D=32/64/128. Performance impact negligible at D=32. |

---

## Cross-references

- **Research:** [`katgpt-rs/.research/242_*.md`](../.research/242_Topological_State_Tracking_Recurrent_Belief.md) (open primitive)
- **Private guide:** [`riir-ai/.research/127_*.md`](../../../riir-ai/.research/127_Implicit_Microcognition_Crowd_NPC_Guide.md) (Super-GOAT selling point)
- **Source paper:** [arXiv:2604.17121](https://arxiv.org/abs/2604.17121) — Mozer et al., DeepMind, Jun 2026
- **Closest cousins:** Research 097 (training-free loop), 192 (NextLat belief dynamics), 070 (Gated DeltaNet-2); Plans 108 (LT2), 136 (Training-Free Loop), 217 (NextLat drafter), 255 (ANE-Latent NPC Brain), 262 (Latent Physics — upgrade target), 275 (SwiR switch-thinking)
- **Commercial strategy:** [`katgpt-rs/.research/003_*.md`](../.research/003_Commercial_Open_Source_Strategy_Verdict.md) §Super-GOAT Capture Protocol

---

## TL;DR

Ship a generic `MicroRecurrentBeliefState` kernel in `katgpt-rs/src/micro_belief/` — three recurrence families (attractor loop, delta-rule SSM, latent-thought loop) drawn from Mozer et al. 2026's taxonomy of recurrent transformers. Each family implements `s_t = f(s_{t-1}, x_t)` in a fixed-size latent vector (default dim=32), with a freeze/thaw-snapshotable kernel and a sigmoid-dot bridge to bounded raw scalars. GOAT gate G1 (determinism, boundedness, ranking preservation, ≤100ns/tick, atomic swap) must pass before promoting to default-on. Family C (delta-rule) is the always-stable fallback; Family A (attractor) is the default if stable. This is the open primitive for the Super-GOAT fusion in `riir-ai/.research/127`.
