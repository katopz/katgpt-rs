# Plan 334: Sleep-Time Query Anticipator — Open Primitive

**Date:** 2026-06-27
**Research:** [katgpt-rs/.research/318_Sleep_Time_Compute_Offline_Query_Anticipation.md](../.research/318_Sleep_Time_Compute_Offline_Query_Anticipation.md)
**Source paper:** [arxiv 2504.13171](https://arxiv.org/abs/2504.13171) — Lin et al. (Letta/Berkeley), *Sleep-time Compute: Beyond Inference Scaling at Test-time*
**Target:** `katgpt-rs/src/sleep_time/` (new module) + Cargo feature `sleep_time_anticipation`
**Status:** ✅ COMPLETE / CLOSED (2026-06-27). All four phases shipped. Open math primitives live under `sleep_time_anticipation` (opt-in). Phase 1: traits + types + IdentityFunctorOp. Phase 2: 31 inline unit tests + 13 GOAT gate tests (G1/G2/G7) + 1 alloc-check test (G5) all pass; G6 latency measured inside targets (3.5–10× margin). Phase 3: two runnable examples (`sleep_time_01_basic.rs`, `sleep_time_02_curiosity_inversion.rs`) — both build and run clean. Phase 4: README Feature Showcase entry + architecture doc entry + Plan 154 cross-reference. Quality gates G2/G3/G4 require a real predictability-labeled corpus → deferred to riir-ai Plan 341. Promotion to default-on requires Plan 341 G1–G5 to clear on a real game corpus.

---

## Goal

Ship the **generic, game-semantic-free math primitives** for sleep-time compute: a trait that scores the predictability of likely queries from a context, allocates sleep-time compute proportional to predictability, and emits a reusable anticipated-query projection set that test-time consumers apply via cheap dot-product + sigmoid gates. The per-NPC runtime integration, HLA wiring, latent_functor proactive extraction mode, and chain quorum commitment all live in `riir-ai/.plans/341_npc_sleep_time_anticipation_runtime.md` (private). This plan ships ONLY the open math.

**Why katgpt-rs (Open, MIT):** the primitives are domain-agnostic — `PredictabilityScorer`, `AmortizationCostModel`, `SleepTimeAnticipator` work on any latent state. No game IP, no chain IP, no neuron-shard IP. Game-specific direction-vector sets, NPC tiering, and zone-type catalogs stay in riir-ai.

**GOAT gate:** every new primitive ships behind `sleep_time_anticipation` (opt-in). Promotion to default-on requires the riir-ai runtime plan (341) to clear its G1–G5 protocol on a real game corpus. **This plan ships synthetic gates only** (G1 mechanics, G5 zero-alloc, G7 BLAKE3 commitment); the quality gates G2/G3/G4 require a real predictability-labeled corpus and are deferred to riir-ai Plan 341.

---

## Distilled primitive (math, training-free)

```
Inputs:
  c        : context latent state (slice, dim D)
  D_set    : anticipated-query direction set {[f32; D]; K}  (precomputed, frozen)
  budgets  : per-direction sleep-time compute budget {[tokens]; K}

Sleep-time (offline):
  for i in 0..K:
      z_i  = sleep_compute(c, D_set[i], budgets[i])   // any modelless op: extract_functor, karc_forecast, etc.
      p_i  = predictability(c, D_set[i])              // dot-product + sigmoid
  c' = {(D_set[i], z_i, p_i)}_{i=0..K}

Test-time (online):
  for incoming query q (latent embedding):
      i* = argmax_i dot(q, D_set[i])
      gate = sigmoid(beta * (p_{i*} - tau))
      a = gate * apply(z_{i*}, q) + (1 - gate) * fresh_think(q, b_max)
                              ^ pre-computed       ^ fallback wake-time compute
```

**Cost model (paper §5.3):**
```
cost_total = sum_i(budgets[i])              // sleep-time, paid once per c
           + N_consumers * t * b_max * (1 - E[gate])
                                              // test-time, paid per consumer
                                              // t = latency premium (paper uses t=10)
                                              // gate = sigmoid(beta * (p - tau))
                                              // E[gate] = expected pre-computation hit rate
```

Amortization factor across N consumers: `(sum_i budgets[i]) / N`. Break-even: `sum_i budgets[i] < N * t * b_max * E[gate]`.

---

## Phase 1 — Skeleton + Core Traits (CORE)

**Status: ✅ COMPLETE (2026-06-27).** All 8 tasks shipped. 31 inline unit tests pass. `cargo check --all-features` clean (the `DEFAULT_K` collision with `cgsp::DEFAULT_K` was caught and fixed by renaming to `SLEEP_TIME_DEFAULT_K`). Default-features compile clean (zero impact when feature off).

### Tasks


- [x] **T1.1** Create `katgpt-rs/src/sleep_time/mod.rs` with module root + re-exports. (Landed at `crates/katgpt-core/src/sleep_time/mod.rs` — the path in the task was the pre-workspace layout; current canonical path is under `crates/katgpt-core/`.)
- [x] **T1.2** Define core types in `katgpt-rs/src/sleep_time/types.rs`:
  ```rust
  /// A frozen anticipated-query direction vector. The "slot" in c'.
  /// Generic over D (latent dim). No game semantics.
  #[derive(Clone, Debug)]
  pub struct AnticipatedQueryDir<const D: usize> {
      pub direction: [f32; D],
      pub blake3: [u8; 32],      // commitment of `direction`
      pub version: u64,           // monotonic, for freeze/thaw
  }

  /// One slot in the anticipated-query projection set c'.
  #[derive(Clone, Debug)]
  pub struct AnticipatedSlot<const D: usize> {
      pub dir: AnticipatedQueryDir<D>,
      pub precomputed: [f32; D],   // the z_i — modelless op output
      pub predictability: f32,     // p_i in [0,1]
  }

  /// The full c' artifact — the output of sleep-time compute.
  /// Reusable across consumers. BLAKE3-committed as a whole.
  #[derive(Clone, Debug)]
  pub struct AnticipatedQuerySet<const D: usize, const K: usize> {
      pub slots: [AnticipatedSlot<D>; K],
      pub blake3: [u8; 32],
      pub version: u64,
  }
  ```

- [x] **T1.3** Define `PredictabilityScorer` trait in `katgpt-rs/src/sleep_time/predictability.rs`:
  ```rust
  /// Scores how predictable a query of class `dir` is from context `c`.
  /// Returns p in [0,1]. Higher = more predictable = more sleep-time compute warranted.
  ///
  /// The default modelless implementation: p = sigmoid(alpha * dot(c, dir) + beta).
  /// Curiosity-inversion implementation: p = 1 - sigmoid(curiosity_residual).
  /// Both are dot-product + sigmoid per AGENTS.md (never softmax).
  pub trait PredictabilityScorer<const D: usize> {
      fn predictability(&self, c: &[f32; D], dir: &AnticipatedQueryDir<D>) -> f32;
  }

  /// Default scorer: p = sigmoid(alpha * dot(c, dir) + beta).
  /// alpha = sharpness (higher = sharper gate), beta = bias (higher = pre-compute by default).
  #[derive(Clone, Copy, Debug)]
  pub struct DotPredictabilityScorer {
      pub alpha: f32,
      pub beta: f32,
  }

  impl<const D: usize> PredictabilityScorer<D> for DotPredictabilityScorer {
      #[inline]
      fn predictability(&self, c: &[f32; D], dir: &AnticipatedQueryDir<D>) -> f32 {
          // simd_dot_f32 from katgpt-core::simd — zero-alloc
          let dot = simd_dot_f32(c, &dir.direction);
          sigmoid(self.alpha * dot + self.beta)
      }
  }
  ```

- [x] **T1.4** Define `SleepTimeAnticipator` trait in `katgpt-rs/src/sleep_time/anticipator.rs`:
  ```rust
  /// The sleep-time operator S(c) → c'. Trait-bound, not concrete — the concrete
  /// per-domain op (latent_functor extraction, KARC ridge fit, Engram hash lookup,
  /// LLM-in-the-loop) is provided by the consumer. This trait ships only the
  /// orchestration: predictability scoring + per-direction budget allocation +
  /// c' artifact emission.
  pub trait SleepTimeComputeOp<const D: usize> {
      /// One sleep-time compute call. Produces z_i for direction i.
      /// MUST be modelless (no backprop). MUST be deterministic given (c, dir, budget).
      fn sleep_compute(
          &self,
          c: &[f32; D],
          dir: &AnticipatedQueryDir<D>,
          budget: u32,
          scratch: &mut SleepTimeScratch<D>,
      ) -> [f32; D];
  }

  /// Reusable scratch buffer for sleep-time compute. Passed in by caller to keep
  /// the hot path zero-allocation (per AGENTS.md / .contexts/optimization.md).
  #[derive(Clone, Debug)]
  pub struct SleepTimeScratch<const D: usize> {
      pub buf: [f32; D],
      pub aux: [f32; D],   // for ops that need a second buffer (ridge fit, etc.)
  }

  /// Orchestrates sleep-time compute across K directions, emitting a c' artifact.
  pub struct SleepTimeAnticipator<const D: usize, const K: usize, Op, Scorer> {
      pub op: Op,
      pub scorer: Scorer,
      pub budgets: [u32; K],
      pub tau: f32,        // gate threshold (wake-time consumer uses this)
      pub beta: f32,       // gate sharpness
  }

  impl<const D: usize, const K: usize, Op, Scorer> SleepTimeAnticipator<D, K, Op, Scorer>
  where
      Op: SleepTimeComputeOp<D>,
      Scorer: PredictabilityScorer<D>,
  {
      /// Run sleep-time compute. Produces c' for the given c and direction set.
      /// Zero-allocation in steady state (uses caller-provided scratch).
      pub fn anticipate(
          &self,
          c: &[f32; D],
          dirs: &[AnticipatedQueryDir<D>; K],
          scratch: &mut SleepTimeScratch<D>,
      ) -> AnticipatedQuerySet<D, K> {
          let mut slots = std::array::from_fn(|_| AnticipatedSlot {
              dir: dirs[0].clone(), // placeholder, overwritten below
              precomputed: [0.0; D],
              predictability: 0.0,
          });
          for i in 0..K {
              let z = self.op.sleep_compute(c, &dirs[i], self.budgets[i], scratch);
              let p = self.scorer.predictability(c, &dirs[i]);
              slots[i] = AnticipatedSlot {
                  dir: dirs[i].clone(),
                  precomputed: z,
                  predictability: p,
              };
          }
          // BLAKE3 over all slot bytes.
          let blake3 = commit anticipated_query_set_bytes(&slots);
          AnticipatedQuerySet {
              slots,
              blake3,
              version: next_version(),
          }
      }
  }
  ```

- [x] **T1.5** Define `AmortizationCostModel` in `katgpt-rs/src/sleep_time/cost_model.rs`:
  ```rust
  /// Paper §5.3 cost model: cost_total = sleep_cost + N * t * b_max * (1 - E[gate]).
  /// Answers: "is it worth pre-computing c' for this c, given N expected consumers?"
  #[derive(Clone, Copy, Debug)]
  pub struct AmortizationCostModel {
      pub t: f32,           // latency premium (paper uses t=10)
      pub b_max: u32,       // wake-time compute budget per consumer
      pub tau: f32,         // gate threshold
      pub beta: f32,        // gate sharpness
  }

  impl AmortizationCostModel {
      /// Expected per-consumer wake-time cost, accounting for gate hit rate.
      /// E[gate] = sigmoid(beta * (p - tau)) averaged over the predicted query distribution.
      #[inline]
      pub fn expected_wake_cost_per_consumer(&self, e_gate: f32) -> f32 {
          self.t * (self.b_max as f32) * (1.0 - e_gate)
      }

      /// Total cost given N consumers.
      #[inline]
      pub fn total_cost(&self, sleep_cost: f32, n_consumers: u32, e_gate: f32) -> f32 {
          sleep_cost + (n_consumers as f32) * self.expected_wake_cost_per_consumer(e_gate)
      }

      /// Should we pre-compute? Returns true iff sleep_cost < N * t * b_max * E[gate].
      /// (i.e., the sleep-time compute is cheaper than the wake-time compute it replaces.)
      #[inline]
      pub fn should_pre_compute(&self, sleep_cost: f32, n_consumers: u32, e_gate: f32) -> bool {
          let wake_cost_avoided = (n_consumers as f32) * self.t * (self.b_max as f32) * e_gate;
          sleep_cost < wake_cost_avoided
      }

      /// Amortization factor: cost_with_precompute / cost_without_precompute.
      /// < 1.0 means pre-computing wins. Paper reports ~2.5× gain at N=10.
      #[inline]
      pub fn amortization_factor(&self, sleep_cost: f32, n_consumers: u32, e_gate: f32) -> f32 {
          let without = (n_consumers as f32) * self.t * (self.b_max as f32);
          let with = self.total_cost(sleep_cost, n_consumers, e_gate);
          with / without
      }
  }
  ```

- [x] **T1.6** Define wake-time `consume` helper in `katgpt-rs/src/sleep_time/consume.rs`:
  ```rust
  /// Wake-time consumer: given query q and pre-computed c', produce an answer
  /// via cheap lookup + sigmoid gate. Falls through to caller-provided fresh_think
  /// if the gate is low (unpredictable query).
  ///
  /// This is T_b(q, c') → a in the paper's notation, with b << B because most
  /// of the work was done at sleep-time.
  pub fn consume<const D: usize, const K: usize, F>(
      q: &[f32; D],
      c_prime: &AnticipatedQuerySet<D, K>,
      tau: f32,
      beta: f32,
      fresh_think: F,
  ) -> [f32; D]
  where
      F: FnOnce(&[f32; D]) -> [f32; D],
  {
      // Find best-matching anticipated direction.
      let mut best_i = 0;
      let mut best_dot = f32::NEG_INFINITY;
      for i in 0..K {
          let d = simd_dot_f32(q, &c_prime.slots[i].dir.direction);
          if d > best_dot {
              best_dot = d;
              best_i = i;
          }
      }
      // Sigmoid gate from predictability of the best match.
      let p = c_prime.slots[best_i].predictability;
      let gate = sigmoid(beta * (p - tau));
      let z = c_prime.slots[best_i].precomputed;
      // gated blend: gate * precomputed + (1 - gate) * fresh_think(q)
      // sigmoid smooth blend (never argmax hard switch — preserves gradients if any).
      let fresh = fresh_think(q);
      let mut out = [0.0f32; D];
      for j in 0..D {
          out[j] = gate * z[j] + (1.0 - gate) * fresh[j];
      }
      out
  }
  ```

- [x] **T1.7** Add Cargo feature in `katgpt-rs/Cargo.toml`:
  ```toml
  [features]
  sleep_time_anticipation = ["dep:blake3"]
  ```
  And in `crates/katgpt-core/Cargo.toml`:
  ```toml
  [features]
  sleep_time_anticipation = []
  ```

- [x] **T1.8** Re-export from `katgpt-rs/src/lib.rs` and `crates/katgpt-core/src/lib.rs`:
  ```rust
  #[cfg(feature = "sleep_time_anticipation")]
  pub mod sleep_time;
  ```

---

## Phase 2 — Synthetic Gates (proves the math, not the game)

**Status: ✅ COMPLETE (2026-06-27).** G1×5, G2×4, G7×4 GOAT gate tests pass in `tests/sleep_time_goat.rs`. G5 zero-alloc passes in `tests/sleep_time_alloc_check.rs` (separate binary; both `consume()` and `consume_gate()` at 0 allocs / 1000 calls). G6 latency measured at 9.5ns (D=8,K=8) and 57.6ns (D=64,K=8) — both inside the ≤200ns gate with 3.5–10× margin.

These are the **mechanics + zero-alloc + commitment + latency** gates. Quality gates G2/G3/G4 require a real predictability-labeled corpus and live in riir-ai Plan 341.

### "Report the Floor" UQ comparison (Issue 010 T4)

**G-UQ: N/A — EXCLUDED.** The Sleep-Time Anticipator's predictability score (`DotPredictabilityScorer`, `p = sigmoid(α·dot(c,dir)+β)`) is a **gate heuristic, not a calibrated UQ signal.** Same false-confidence signature as BoM (T3): predictability-derived intervals win CRPS (0.55–0.63 ratio) but lose coverage (37–54% vs nominal 95%) and Winkler (2.5–3.4× the floor). The T4 difficulty-correlation test shows near-zero per-step correlation for BOTH the anticipator and the floor (|r| < 0.08). Excluded via the reframing escape hatch — the anticipator's value is amortized compute gating (Plan 334 G1/G2 mechanics + `AmortizationCostModel`), NOT calibrated UQ. `sleep_time_anticipation` stays OPT-IN (unchanged). See `tests/conformal_floor_sleep_time.rs`, `.benchmarks/010_sleep_time_floor_comparison.md`.

### Tasks

- [x] **T2.1** `tests/sleep_time_goat.rs::g1_*` — verify:
  - `AnticipatedQuerySet` round-trips (anticipate → serialize → deserialize → bit-identical slots).
  - `consume` returns the precomputed slot when `predictability > tau` and the fresh_think output when `predictability < tau` (smooth blend, not hard switch).
  - `predictability` is in `[0,1]` for all inputs.
  - `consume` is deterministic given (q, c', tau, beta).

- [x] **T2.2** `tests/sleep_time_goat.rs::g2_*` — verify:
  - `amortization_factor(sleep=100, N=10, e_gate=0.5, t=10, b_max=100) ≈ 0.55` (matches paper's 2.5× gain inverted: cost ratio = 1/2.5 + offset).
  - `should_pre_compute` returns true iff `sleep_cost < N * t * b_max * e_gate`.
  - `total_cost` is monotonic decreasing in `e_gate` (more predictability → less total cost).

- [x] **T2.3** `tests/sleep_time_alloc_check.rs::g5_zero_alloc_after_warmup_both_paths` — verify via `CountingAllocator`:
  - After 100 warmup `consume()` calls, **0 allocations / 0 bytes** over 100 measured `consume()` calls.
  - `anticipate()` may allocate (it builds the c' artifact); `consume()` (the wake-time hot path) MUST NOT.
  - This mirrors Plan 308 G5 and Plan 304 G5 — wake-time latency budget is sacred.

- [x] **T2.4** `tests/sleep_time_goat.rs::g7_*` — verify:
  - `AnticipatedQuerySet::blake3` is recomputed correctly when any slot's `precomputed` or `predictability` changes.
  - Tampering any byte in a serialized `AnticipatedQuerySet` produces a different `blake3` (catch-in-the-round).
  - Two `anticipate()` calls with identical inputs produce identical `blake3`.

- [x] **T2.5** `benches/sleep_time_consume_bench.rs` — verify `consume()` is < 200ns at D=64. **Measured: 57.6 ns/call at D=64,K=8 (3.5× margin); 9.5 ns/call at D=8,K=8 (10× margin vs the ≤100ns D=8 target).** Both `consume()` and `consume_gate()` measured. Run via `cargo bench -p katgpt-core --features sleep_time_anticipation --bench sleep_time_consume_bench`.

---

## Phase 3 — Example (adoption hook)

**Status: ✅ COMPLETE (2026-06-27).** Both examples build and run clean via `cargo run -p katgpt-core --example <name> --features katgpt-core/sleep_time_anticipation --release`. T3.2 uses a generalized curiosity-inversion scorer (`p = sigmoid(α·(curiosity_ref − curiosity))`) rather than the special-case `p = 1 − sigmoid(curiosity)` form, so the high/low-predictability contrast and the `should_pre_compute` verdict flip are both cleanly visible. The scorer is implemented in the example (not in the shipped API) to demonstrate the `PredictabilityScorer` trait-swap mechanism without expanding the primitive's surface.

### Tasks

- [x] **T3.1** `examples/sleep_time_01_basic.rs` — minimal example showing:
  - Construct 4 anticipated-query directions (hardcoded).
  - Run `anticipate()` on a context.
  - Run `consume()` on a query, show the gated blend.
  - Print the cost model: amortization factor at N=1 vs N=10.
- [x] **T3.2** `examples/sleep_time_02_curiosity_inversion.rs` — show the predictability = 1 − curiosity mapping:
  - Use a fake KARC-like forecaster (closed-form ridge over synthetic trajectory).
  - Show that high-curiosity contexts (large forecast residual) get LOW predictability → `should_pre_compute` returns false.
  - Show that low-curiosity contexts get HIGH predictability → `should_pre_compute` returns true.

---

## Phase 4 — Documentation

**Status: ✅ COMPLETE (2026-06-27).**

### Tasks

- [x] **T4.1** Add `sleep_time/` section to `katgpt-rs/README.md` under "Feature Showcase" — brief, public-facing. Reference Research 318 and Plan 334.
- [x] **T4.2** Add entry to `katgpt-rs/.docs/02_architecture.md` describing the module.
- [x] **T4.3** Cross-reference from `katgpt-rs/.docs/18_sleep_consolidation.md` (the existing Plan 154 doc) — note that Plan 334 is the **artifact-emission** complement to Plan 154's **state-internalization** approach.

---

## What stays OUT of this plan (riir-ai Plan 341 territory)

- Per-NPC HLA wiring (the concrete `SleepTimeComputeOp` impl that uses `extract_functor` / `karc_forecast`).
- Per-zone direction vector catalog (game-specific: shopkeeper zone, quest-giver zone, lore NPC zone).
- Chain quorum commitment of `c'` (uses `cgsp_runtime/chain_bridge.rs`'s `commit_snapshot_via_quorum`).
- SleepAnticipationShard subtype (riir-neuron-db territory, deferred to P2).
- Real game corpus G2/G3/G4 quality gates.
- NPC tiering (important NPC gets full K directions; crowd NPC gets K=1 or K=0).
- Cross-player amortization economics (validated by G4 in Plan 341).

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| The dot-product predictability scorer is too naive (doesn't capture true query predictability) | Trait-based: riir-ai Plan 341 can swap in any scorer. The default `DotPredictabilityScorer` is a baseline, not a claim. |
| `consume()` argmax loop is O(K) — slow at high K | K is bounded (paper uses K≤10; we expect K≤8 per NPC). For K>16, switch to a hash-indexed lookup (Engram-style). |
| Sigmoid blend vs hard switch — gradients leak | Pure inference path; no gradients. Sigmoid blend is per AGENTS.md ("use sigmoid not softmax"). Hard switch would be faster but loses the smooth trade-off. |
| The artifact is large (K * (D + D + 1) f32 per NPC) | For D=8 (HLA), K=8 → 8 * (8+8+1) * 4 = 544 bytes. Fits in L2 cache. For D=64 (style_weights), K=8 → 4.2KB — still warm-tier friendly. Compress via `ShardCompactor` if needed. |

---

## References

**Parent research:** `katgpt-rs/.research/318_Sleep_Time_Compute_Offline_Query_Anticipation.md`
- **Source paper:** [arxiv 2504.13171](https://arxiv.org/abs/2504.13171) — Lin et al. 2025.
- **Closest shipped cousins:**
  - `katgpt-rs/.plans/154_sleep_consolidation_offline_memory.md` (LLM Sleep — the consolidation substrate)
  - `katgpt-rs/.plans/107_auto_dreamer_offline_consolidation.md` (AutoDreamer — modelless consolidation)
  - `katgpt-rs/.plans/308_karc_delay_basis_ridge_forecaster.md` (KARC — the forecast primitive)
  - `katgpt-rs/.plans/304_gain_cost_loop_halting_primitive.md` (Gain/Cost halting — wake-time budget)
  - `katgpt-rs/.plans/303_salience_tri_gate_primitive.md` (Salience Tri-Gate — wake-time consume decision)
  - `katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md` (Engram — hash-addressed memory)
- **Private runtime integration:** `riir-ai/.plans/341_npc_sleep_time_anticipation_runtime.md`
- **Private selling-point guide:** `riir-ai/.research/163_Per_NPC_Sleep_Time_Query_Anticipation_Guide.md`

---

## TL;DR

Open math primitives for sleep-time compute (arXiv:2504.13171): `SleepTimeAnticipator` orchestrates per-direction sleep-time compute → emits reusable `AnticipatedQuerySet` (the c' artifact) → wake-time `consume()` does cheap dot-product + sigmoid-gated lookup, falling through to fresh compute on low-predictability queries. `PredictabilityScorer` trait with `DotPredictabilityScorer` default (p = sigmoid(α·dot(c,dir)+β)); `AmortizationCostModel` operationalizes the paper's §5.3 cost model (sleep_cost + N·t·b_max·(1−E[gate])). Phase 1 ships traits + types; Phase 2 ships synthetic gates G1/G2/G5/G6/G7 (mechanics, cost-model correctness, zero-alloc wake-time, latency, BLAKE3 commitment). G2/G3/G4 quality gates require real predictability-labeled corpus → deferred to riir-ai Plan 341. Opt-in feature `sleep_time_anticipation`; promotion to default-on requires Plan 341 G1–G5 to clear on a real game corpus. **Not a Hot-tier primitive** — sleep-time runs in warm/cold-tier; the produced `c'` artifact is what Hot-tier consumes at wake-time. The CompressionDrafter Plan 285 failure mode is the cautionary tale.
