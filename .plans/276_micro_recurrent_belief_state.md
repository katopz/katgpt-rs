# Plan 276: MicroRecurrentBeliefState — Implicit Per-Entity State Tracking Kernel

**Date:** 2026-06-15
**Research:** [katgpt-rs/.research/242_Topological_State_Tracking_Recurrent_Belief.md](../.research/242_Topological_State_Tracking_Recurrent_Belief.md)
- **Private guide:** [`riir-ai/.research/127_*.md`](../../../riir-ai/.research/127_Implicit_Microcognition_Crowd_NPC_Guide.md) — **reframed as GOAT design context** (verdict revised from Super-GOAT after `evolve_hla` prior-art check)
**Source paper:** [arXiv:2604.17121](https://arxiv.org/abs/2604.17121) — Mozer, Siddiqui, Liu (DeepMind, Jun 2026), "The Topological Trouble With Transformers"
**Target:** Extend `katgpt-rs/crates/katgpt-core/src/sense/` (refactor `evolve_hla` into a trait + add attractor family) + new `micro_belief/` submodule for the trait + snapshot + bridge + Cargo feature `micro_belief`
**Status:** Active — Phase 0 (planning). **Verdict revised same day: Super-GOAT → GOAT** after prior-art check found `evolve_hla` already implements Family C.

---

## Goal

**Revised (verdict downgrade):** the prior-art check found `ReconstructionState::evolve_hla()` already implements Family C (delta-rule SSM) — per-NPC recurrent latent state tracking is *shipped*. This plan is no longer "ship a new primitive"; it is **"extend `evolve_hla` into a trait, add an attractor-family variant (Family A) for beliefs-with-hysteresis, and optionally add kernel-as-versioned-snapshot for per-NPC divergence."**

The GOAT-gate question becomes: **does attractor update (`σ(W_s·s + W_x·x + b)`) reduce long-horizon flip-flops vs HLA's leaky integrator on a coherence benchmark?** If yes → promote attractor family as an opt-in variant (probably NOT default — HLA's leaky integrator is battle-tested). If no → demote to Gain, keep only the trait unification.

**GOAT gate (G1 for the trait/snapshot mechanics; G2 for the attractor quality claim):**
- G1.1 Determinism (bit-identical `s_T` for fixed input sequence) — applies to all families
- G1.2 Boundedness (`‖s_t‖` stays bounded over 10k ticks; Family A doesn't diverge)
- G1.3 Bridge ranking preservation (scalar projection preserves belief ranking) — already true for existing `SenseModule::project`, just re-verify
- G1.4 Latency (Family A ≤ 100ns/NPC/tick CPU SIMD; ≤ 50ns ANE batch) — HLA's `evolve_hla_simd` is the baseline to match
- G1.5 Freeze/thaw atomicity for `MicroRecurrentKernelSnapshot` (readers never see torn kernel swap)
- **G2.1 (the actual GOAT gate for this plan):** attractor-family coherence ≥ HLA-leaky-integrator coherence on a long-horizon (1000-turn) dialogue/interaction benchmark, with strictly less flip-flopping. If G2.1 fails, attractor stays opt-in behind a sub-flag and the trait unification is the only shippable output.

**Out of scope (stays in riir-ai/.plans/304):** NPC tick integration changes, ANE batch dispatch for the attractor variant, CGSP fusion, collapse detector. This plan ships *only* the trait refactor + attractor family + snapshot + the G2.1 benchmark.

---

## Phase 0 — Pre-flight (this plan)

### Tasks

- [x] **T0.1** Research note `katgpt-rs/.research/242_*.md` created.
- [x] **T0.2** Private guide `riir-ai/.research/127_*.md` created (Super-GOAT mandatory output).
- [x] **T0.3** This plan created.
- [x] **T0.4** Audit existing freeze/thaw snapshot infra: locate `LoRAWeightVersion`, `LoRAHotSwap`, BLAKE3 commit path. Confirm `MicroRecurrentKernelSnapshot` can reuse the same atomic-swap plumbing without forking it. (Output: a 1-paragraph note in this plan's §Notes identifying the exact trait/struct to extend.)
- [x] **T0.5** Audit existing `SenseModule::project` (the bridge) — confirm it already does dot-product + sigmoid (it does, per the grep). The new trait's `project_to_scalars()` should *delegate* to it, not duplicate it.
- [x] **T0.6** **NEW (post-verdict-revision):** Confirm `evolve_hla` + `evolve_hla_simd` call sites — anywhere else in the codebase that calls them directly needs to either (a) keep working unchanged via the trait impl, or (b) be updated to call through the trait. Grep for `evolve_hla` callers before the refactor.

### Phase 0 Audit Results (T0.4–T0.6)

**T0.4 — Snapshot infra (katgpt-rs side, not riir-ai):** The plan text mentioned `LoRAWeightVersion`/`LoRAHotSwap`, but those live in **riir-ai** (`riir-ai/crates/riir-engine/src/episode_buffer.rs`), not katgpt-rs. katgpt-rs (public engine) uses a *different* atomic-swap idiom:
- **`SenseHotSwap`** (`katgpt-rs/crates/katgpt-core/src/sense/hotswap.rs`): `AtomicPtr<Box<SenseModule>>` + `AtomicBool` lock flag, fixed-size array indexed by `SenseKind`. This is the lock-free hot-swap primitive in the public repo.
- **`SenseModule::commit()` / `verify()`** (`types.rs` L4864–4907): BLAKE3 commitment over struct bytes with `TernaryDir` padding zeroed first for determinism. Same pattern reused by `JlProjectionMatrix::commit()/verify()` (`shard_embedding.rs`) and `GpartAdapter::commitment()/verify()` (`types.rs`).
- **`MerkleOctree`** (`merkle.rs`): hierarchical BLAKE3 for KG latent octree nodes (Plan 221-M).

**Decision (R3 resolved):** Write a parallel `KernelHotSwap` reusing the `SenseHotSwap` `AtomicPtr` primitive (NOT `arc_swap` crate — not a current katgpt-rs dep; NOT `LoRAWeightVersion` — that's riir-ai-private game IP). `MicroRecurrentKernelSnapshot` reuses the `SenseModule::commit()` BLAKE3-over-struct-bytes pattern directly. No forking needed; the BLAKE3 commitment + AtomicPtr swap primitives are already public-engine idioms.

**T0.5 — Bridge confirmed:** `SenseModule::project(hla_state: &[f32; 8]) -> f32` (`types.rs` L4825) does exactly dot-product (ternary sign × hla_val × row_scale, FMA-fused) + `crate::simd::fast_sigmoid(dot)` scaled by `confidence`. This IS the dot-product + sigmoid bridge. The new trait's `project_to_scalars()` will **delegate** to this pattern (operating on the belief vector the same way `project` operates on `hla_state`), reusing `crate::simd::fast_sigmoid` and the dot-product helper. No duplication.

**T0.6 — Call sites confirmed (safe to refactor):** `evolve_hla()` is called ONLY inside `ReconstructionState` methods: `reconstruct()` (L704), `reconstruct_matvec()` (L728), `reconstruct_with_weights()` (L753), and the shared `reconstruct_inner()` (L785, dispatches to scalar or SIMD). `evolve_hla_simd()` is called only in `reconstruct_inner()` behind `sense_composition` feature. **No external direct callers** — benchmarks (`reconstruction_bench.rs`) go through `ReconstructionState::reconstruct*`. The refactor is safe: move `evolve_hla` body into `LeakyIntegrator::step()` (Phase 2), make `ReconstructionState::evolve_hla()` a thin delegate. `reconstruct_inner` is the only dispatch site to update for the trait.

---

## Phase 1 — Core Skeleton + Family A (Attractor Loop)

**Unblocks:** G1.1, G1.2, G1.3, G1.4 (partial), G1.5 (partial). This is the GOAT-gate phase.

### Architecture (revised — extend existing sense/, not greenfield)

```text
katgpt-rs/crates/katgpt-core/src/
├── sense/
│   ├── reconstruction.rs       // EXISTING — evolve_hla() becomes an impl of the trait
│   ├── brain.rs                // EXISTING — NpcBrain::hla_state is the state vector
│   └── ...                     // EXISTING — SenseModule::project() is the bridge (unchanged)
└── micro_belief/               // NEW submodule (small)
    ├── mod.rs                  // Module root, re-exports
    ├── types.rs                // BeliefKernel trait, RecurrenceFamily enum, KernelConfig
    ├── attractor.rs            // Family A: s_t = σ(W_s·s_{t-1} + W_x·x_t + b)  [the GOAT candidate]
    ├── leaky.rs                // Family C wrapper: delegates to existing evolve_hla logic (zero-behavior-change refactor)
    ├── snapshot.rs             // MicroRecurrentKernelSnapshot (BLAKE3, versioned) — optional, for per-NPC divergence
    └── tests.rs                // G1.1–G1.5 + G2.1 (the coherence benchmark)
```

**Key refactor principle:** `ReconstructionState::evolve_hla()` logic moves into `leaky.rs` as `impl BeliefKernel for LeakyIntegrator`. The existing call site in the `expand → route → accumulate → evolve_hla` loop calls through the trait. **Zero behavior change for the default path** — this is critical to avoid regressing the shipped HLA benchmarks.

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

## Phase 2 — Family C wrapper (zero-behavior-change refactor of evolve_hla)

**Why (revised):** This is no longer "the fallback" — it's the **default that already ships**. The task is to wrap the existing `evolve_hla` logic in the trait without changing behavior, so the call site can dispatch to either LeakyIntegrator (today's default) or AttractorKernel (the GOAT candidate) transparently.

### Tasks

- [ ] **T2.1** `leaky.rs`: `LeakyIntegrator { lr, max_delta }` — move the body of `evolve_hla` into `impl BeliefKernel for LeakyIntegrator`. The existing `ReconstructionState::evolve_hla()` becomes a thin delegate.
- [ ] **T2.2** SIMD path: `evolve_hla_simd` logic moves into `LeakyIntegrator::step_simd()`; the existing method delegates.
- [ ] **T2.3** **Zero-behavior-change test:** the existing HLA benchmarks (`reconstruction_bench.rs`) produce identical numbers before and after the refactor. This is the regression gate.
- [ ] **T2.4** Backward-compat: `DeltaRuleKernel { alpha: [λ; D], beta: [0; D] }` (from the original Phase 2 plan) composed with sigmoid bridge matches `SpatialBelief::decay_confidence()` — only relevant if Plan 262's static-decay fallback needs a path; otherwise skip.

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

## Phase 5 — GOAT Gate G2.1 + Decision + Commit

### Tasks

- [ ] **T5.0** **NEW (the actual GOAT gate for this plan):** Build the G2.1 coherence benchmark — a synthetic long-horizon (1000-step) input sequence with injected ambiguity/flip-flop triggers (analog of the paper's "bank" polysemy). Run LeakyIntegrator (HLA default) vs AttractorKernel (Family A). Measure flip-flop rate + belief stability.
- [ ] **T5.1** Run G2.1. **If attractor wins (less flip-flopping, ≥X% coherence gain)** → promote `micro_belief_attractor` as an opt-in variant (NOT default — HLA leaky is battle-tested). Document in `.docs/01_overview.md`.
- [ ] **T5.2** **If attractor loses or ties** → demote to Gain. Keep the trait unification + LeakyIntegrator wrapper as the only shippable output. Attractor family stays behind `micro_belief_attractor` sub-flag for experimentation.
- [ ] **T5.3** Commit with `feat:` (if attractor promoted) or `refactor:` (if only trait unification shipped) prefix on `develop`.
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

**Verdict revised Super-GOAT → GOAT** after a prior-art check found `ReconstructionState::evolve_hla()` already implements the core (per-NPC recurrent latent state + sigmoid bridge — it's literally Family C of the proposed primitive, already shipped and benchmarked). This plan is now **"extend `evolve_hla` into a `BeliefKernel` trait, add an attractor-family variant (Family A) for beliefs-with-hysteresis, and benchmark attractor vs leaky on a long-horizon coherence task (G2.1)."** The trait unification + LeakyIntegrator wrapper is a zero-behavior-change refactor; the attractor family is the GOAT candidate that must beat HLA's leaky integrator on flip-flop rate to justify its existence. If G2.1 fails, attractor stays an opt-in experiment and only the trait refactor ships. The bridge (`SenseModule::project`) and sync-boundary discipline are reused unchanged — HLA already got those right.
