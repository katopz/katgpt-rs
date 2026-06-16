# Plan 282: Reachable Dual-Pool Memory Router (Modelless)

**Date:** 2026-06-16
**Research:** [katgpt-rs/.research/249_DecentMem_DualPool_Reachable_Router.md](../.research/249_DecentMem_DualPool_Reachable_Router.md)
**Source paper:** [arXiv:2605.22721](https://arxiv.org/pdf/2605.22721) — Hao, Long, Zhao 2026, "Self-Evolving MAS via Decentralized Memory"
**Target:** `crates/katgpt-core/src/cgsp/dual_pool.rs` (new module) + Cargo feature `cgsp_dual_pool`
**Status:** Active — Phase 1 (unblocking skeleton)

---

## Goal

Ship a generic **dual-pool memory router** that splits a bandit's candidate pool into an exploitation pool (consolidated past successes, grows over time) and an exploration pool (fresh candidates, regenerated per cycle). The router uses sigmoid-based routing with provable **global reachability** (X-pool always has nonzero probability → Markov chain irreducible + aperiodic, DecentMem Theorem 1) and **O(log T) cumulative regret** (DecentMem Theorem 2). This extends the existing single-pool CGSP `HintDeltaBandit` — single-pool is the degenerate case `α = 1`.

**GOAT gate:** `cgsp_dual_pool` is opt-in. Promote to consideration for CGSP default only after benchmarks show (G1) proactive non-trapping beats CGSP's reactive collapse recovery, (G2) O(log T) regret verified on synthetic bandit, (G3) E-pool growth produces strategies the static pool misses, (G4) FaithfulnessProbe-consolidated items are not dead weight.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Define `PoolId` enum (`Exploitation = 0`, `Exploration = 1`) with `#[repr(u8)]` in `crates/katgpt-core/src/cgsp/dual_pool.rs`. Zero-cost tag.
- [x] **T1.2** Define `ReachableDualPoolRouter` trait (associated types `Item`, `Reward: Copy`; methods `route_select`, `route_update`, `consolidate`, `exploitation_probability`, `is_reachable`). Doc-comment cites DecentMem Theorems 1 + 2.
- [x] **T1.3** Implement `DualPoolBandit<B: HintDeltaBandit>` struct:
  - Fields: `e_pool: B` (exploitation — wraps existing HintDeltaBandit), `x_pool: B` (exploration), `w_e: f32` (exploitation weight, init 1.0), `w_x: f32` (exploration weight, fixed 1.0 per paper Eq. 6/7), `alpha_update_gain: f32` (paper's `α = 0.5`), `decay: f32` (paper's `β = 0.5`).
  - `exploitation_probability()` → `sigmoid(self.w_e - self.w_x).clamp(ε, 1−ε)` (NOT ratio — per AGENTS.md sigmoid rule; regret proof transfers per Research 249 §2.3). **Note:** f32 sigmoid saturates at `x ≳ 18` (1+exp(−18) rounds to 1.0 in f32), so raw sigmoid gives α=1.0 exactly at extreme weights — breaking `is_reachable()`. Added `min_exploration_prob` clamp (default `1e-4`) as the numerical reachability guarantee. The paper's continuous-math theorem holds; the clamp makes it hold in f32.
  - `route_select()` → sample pool by `exploitation_probability()`, select item from chosen pool's bandit (pure `sample_arm_from` fn avoids borrow conflict).
  - `route_update(pool, reward)` → DecentMem Eq. 6/7 (4-case match, only `w_e` updates; `w_x` fixed at 1.0).
  - `consolidate()` → Phase 1 priority-blend (same-size pools): `e[i] = blend·e[i] + (1−blend)·x[i]`, X-pool reset to uniform. True arm growth deferred to Phase 4.
  - `is_reachable()` → `exploitation_probability() < 1.0` (always true via clamp — reachability by construction in f32).
  - Implements `HintDeltaBandit` by delegating to the **active** pool (one pool per cycle, selected in `begin_cycle()`). Drops into `CgspLoop` as the `B` type parameter with zero loop changes.
- [x] **T1.4** Unit tests (10 tests, all pass):
  - `t14_sigmoid_routing_in_unit_interval`: α ∈ (0, 1) for default, extreme-high, and extreme-low w_e.
  - `t14_x_pool_always_reachable`: extreme w_e → `is_reachable()` true + α < 1.0; moderate w_e → X-pool selected ~12% of trials.
  - `t14_weight_update_e_pool_success`: E + success → w_e += gain.
  - `t14_weight_update_e_pool_fail`: E + fail → w_e decays, floors at 1.0.
  - `t14_weight_update_x_pool_success`: X + success → w_e decays.
  - `t14_consolidate_merges_x_into_e`: E-pool blended, size unchanged, X-pool reset to uniform.
  - Bonus: `route_select_returns_valid_arm_and_pool`, `hintdeltabandit_delegates_to_active_pool`, `begin_end_cycle_drives_routing`, `single_pool_degenerate_case_alpha_one`.
- [x] **T1.5** CgspLoop integration (minimal — Phase 1 skeleton): `DualPoolBandit<B>` implements `HintDeltaBandit`, so it drops into `CgspLoop` as `B` with zero changes to `cycle()`. Caller wraps `begin_cycle()` / `end_cycle()` around the existing cycle call. No `DualPoolMode` config variant needed for Phase 1 — the router is self-contained. Full automated `cycle_dual_pool` method deferred to Phase 5 (CGSP Integration Benchmark).
- [x] **T1.6** Register module + feature flag:
  - `#[cfg(feature = "cgsp_dual_pool")] pub mod dual_pool;` in `crates/katgpt-core/src/cgsp/mod.rs` ✓
  - Re-exports: `DualPoolBandit, DualPoolConfig, PoolId, ReachableDualPoolRouter` in `mod.rs` + `lib.rs` ✓
  - `cgsp_dual_pool = ["cgsp"]` in `crates/katgpt-core/Cargo.toml` ✓
  - `cgsp_dual_pool = ["katgpt-core/cgsp_dual_pool", "cgsp"]` passthrough in root `katgpt-rs/Cargo.toml` ✓
- [x] **T1.7** Validation: `cargo test -p katgpt-core --features cgsp_dual_pool --lib cgsp::dual_pool --release` → **10 passed, 0 failed**. `cargo check -p katgpt-core --lib --release` (default) → **clean**. `cargo check --features cgsp_dual_pool --release` (root) → **clean**.

**Phase 1 exit:** `ReachableDualPoolRouter` trait + `DualPoolBandit` impl compile and pass unit tests. Existing CGSP single-pool behavior unchanged.

---

## Phase 2 — Reachability Guarantee Proof (G1)

### Tasks

- [ ] **T2.1** `g1_proactive_non_trapping` test:
  - Build `DualPoolBandit` with 8-arm E-pool + 8-arm X-pool.
  - Force E-pool to one-hot (arm 0 only) by feeding reward only to arm 0 for 100 cycles.
  - **Without** any collapse detector (no `EntropyCollapse::inject_exploration`), verify that over the next 100 cycles, the router selects the X-pool at least once (sigmoid `1 - α > 0` guarantees this).
  - Compare: single-pool CGSP without collapse detector stays trapped indefinitely (this is the baseline failure mode).
- [ ] **T2.2** `g1_reachability_vs_collapse_recovery` benchmark:
  - Same one-hot trap setup.
  - Measure cycles-to-escape: dual-pool (proactive, no detector) vs single-pool + collapse detector (reactive).
  - Target: dual-pool escapes within the sigmoid-driven exploration schedule; single-pool escapes in 1 cycle once detector trips (Plan 274 G2). Both escape — dual-pool is proactive (no detector needed), single-pool is reactive (needs detector). Document the tradeoff: dual-pool has constant nonzero exploration overhead; single-pool has zero overhead until collapse.
- [ ] **T2.3** `g1_markov_chain_irreducibility` property test:
  - Build transition matrix `M = α·T + (1-α)·h·1ᵀ` from the dual-pool's effective transition probabilities.
  - Assert all entries of `M` are strictly positive (Theorem 1).
  - Assert `M` is irreducible (standard graph reachability check).

**Phase 2 exit:** G1 passes — dual-pool provably never traps, by construction (sigmoid), without needing a reactive collapse detector.

---

## Phase 3 — Regret Bound Proof (G2)

### Tasks

- [ ] **T3.1** `g2_log_regret_synthetic` test:
  - Synthetic 2-pool bandit: E-pool has expected reward 0.7, X-pool has expected reward 0.5 (E-pool is better).
  - Run 10,000 cycles. Track cumulative regret vs the always-E-pool oracle.
  - Assert: cumulative regret ≤ C · log(T) for a reasonable constant C (curve-fit the regret curve; verify it's logarithmic, not linear).
- [ ] **T3.2** `g2_fixed_routing_suboptimal` test (Corollary 1):
  - Same setup. Compare online router vs fixed `α = 0.5` vs fixed `α = 0.8`.
  - Assert: online router regret is O(log T); fixed routing regret is Θ(T) (linear).
  - Plot: regret curves diverge — online flattens, fixed grows linearly.
- [ ] **T3.3** `g2_sigmoid_vs_ratio_routing` test:
  - Run both `α = sigmoid(w_e - w_x)` and `α = w_e / (w_e + w_x)` on the same bandit.
  - Assert: both achieve O(log T) regret (validates Research 249 §2.3 — sigmoid preserves the regret bound).

**Phase 3 exit:** G2 passes — O(log T) regret verified empirically on synthetic bandit, matching DecentMem Theorem 2.

---

## Phase 4 — E-Pool Growth + Faithfulness Gate (G3, G4)

### Tasks

- [ ] **T4.1** `g3_epool_grows` test:
  - Start with empty E-pool, 16-arm X-pool.
  - Run 100 cycles. After each cycle, consolidate (rewarded X-pool items → E-pool).
  - Assert: E-pool size monotonically increases (or stays same if no rewards); E-pool ≥ 1 item after 100 cycles on a bandit with any positive-reward arm.
- [ ] **T4.2** `g3_growing_pool_discovers_new_strategies` test:
  - Scenario: E-pool initialized with 4 "known" directions. X-pool generates from a 16-direction superset.
  - The optimal direction is NOT in the initial E-pool (only in X-pool's superset).
  - Run 500 cycles. Assert: the optimal direction gets consolidated into E-pool (the NPC discovers a strategy beyond its initial template — the capability gap identified in Research 249 §2.1).
  - Compare: single-pool CGSP (static 4-direction pool) can never select the optimal direction (it's not in the pool). This is the GOAT gain.
- [ ] **T4.3** Wire `FaithfulnessProbe` (Plan 278) as consolidation gate:
  - Before consolidating an X-pool item into E-pool, run a causal intervention probe.
  - Only items with behavioral delta > `τ_faith` (configurable) enter E-pool.
  - This prevents Research 244's "dead condensed memory" failure — items the consumer structurally ignores don't clog the E-pool.
- [ ] **T4.4** `g4_faithfulness_gate_rejects_dead_items` test:
  - Construct an X-pool item that the consumer (Solver) structurally ignores (perturbation produces no behavioral delta).
  - Run consolidation with faithfulness gate ON.
  - Assert: dead item is rejected (not in E-pool after consolidate).
  - Run consolidation with gate OFF.
  - Assert: dead item enters E-pool (baseline failure mode — E-pool fills with dead weight).

**Phase 4 exit:** G3 + G4 pass — E-pool grows, discovers strategies beyond initial pool, and faithfulness gate keeps it clean.

---

## Phase 5 — CGSP Integration Benchmark (G5)

### Tasks

- [ ] **T5.1** Integrate `DualPoolBandit` into `NpcCgspRuntime` (riir-ai, behind `cgsp_dual_pool` feature):
  - Each NPC's `PriorityTableBandit` wraps in `DualPoolBandit`.
  - E-pool = faction-template directions (frozen at spawn, as today).
  - X-pool = dynamically conjectured directions (the `CuriosityConjecturer` trait already supports this — the shipped impl uses a fixed pool, but the trait can generate novel directions).
  - Consolidation: rewarded X-pool directions → E-pool, with FaithfulnessProbe gate.
- [ ] **T5.2** `g5_personality_divergence_widens` benchmark:
  - Two same-faction NPCs, same RNG seed, 1000 cycles.
  - Measure priority-table cosine similarity over time.
  - Dual-pool: NPCs diverge MORE than single-pool (X-pool conjectures different novel directions per NPC → E-pools diverge).
  - Target: dual-pool cosine similarity < single-pool cosine similarity at cycle 1000.
- [ ] **T5.3** `g5_latency_budget` benchmark:
  - Dual-pool adds: 1 sigmoid + 1 branch + consolidation scan per cycle.
  - Assert: per-cycle overhead < 0.5µs over single-pool CGSP baseline (plasma tier).
  - No allocation in hot path (reuse pre-allocated pools).
- [ ] **T5.4** `g5_epool_persistence` test:
  - After 1000 cycles, snapshot the grown E-pool via existing `CuriosityPrioritySnapshot` + BLAKE3 + chain quorum (Plan 299 T4.6 infrastructure).
  - Reload snapshot. Assert: E-pool items are bit-identical (deterministic replay preserved).

**Phase 5 exit:** G5 passes — dual-pool CGSP shows wider personality divergence, stays in plasma latency budget, snapshots persist correctly.

---

## Phase 6 — Documentation + Promotion Decision

### Tasks

- [ ] **T6.1** Add `dual_pool.rs` module docs citing DecentMem Theorems 1 + 2, sigmoid routing rationale, and the CGSP single-pool-as-degenerate-case relationship.
- [ ] **T6.2** Update `katgpt-rs/.docs/07_adaptation.md` with dual-pool as CGSP extension.
- [ ] **T6.3** Update `katgpt-rs/README.md` Feature Showcase with dual-pool entry (after GOAT gate passes).
- [ ] **T6.4** Add example: `examples/cgsp_dual_pool_demo.rs` showing growing E-pool + X-pool exploration on a synthetic 8-direction pool.
- [ ] **T6.5** GOAT gate decision:
  - If G1–G5 all pass AND dual-pool shows measurably wider personality divergence (G5.2) → recommend `cgsp_dual_pool` for promotion to CGSP default in riir-ai (separate riir-ai plan).
  - If G1–G4 pass but G5.2 shows no divergence improvement → keep opt-in, document as "reachability guarantee without personality benefit at this scale."
  - If any gate fails → demote to experimental, create issue.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Sigmoid routing changes regret bound vs paper's ratio | G2.3 explicitly benchmarks sigmoid vs ratio — both must show O(log T). Research 249 §2.3 proves concavity transfers. |
| E-pool grows unbounded → memory + latency | Cap E-pool size (e.g., 64 items). Evict lowest-priority items on consolidation. Pre-allocate fixed-size ring buffer. |
| FaithfulnessProbe is too expensive for hot path | Run probe at consolidation cadence (every N cycles), not every cycle. Probe is O(1) finite-difference per item. |
| X-pool conjecture generation is slow (LLM call) | X-pool items can be pre-generated at spawn (from faction template superset) or generated offline. Hot path only selects, doesn't generate. |
| Dual-pool overhead exceeds plasma budget | G5.3 gates on < 0.5µs overhead. Sigmoid + branch is ~10ns. Consolidation scan is O(E-pool size), done every N cycles not every cycle. |
| Single-pool CGSP already good enough (G5.2 flat) | Acceptable — means the reachability guarantee is the value, not the growth. Keep opt-in for the guarantee, document as such. |

---

## Cross-References

- **Research:** [249_DecentMem_DualPool_Reachable_Router.md](../.research/249_DecentMem_DualPool_Reachable_Router.md)
- **Closest cousin (shipped):** [riir-ai Plan 299](../../riir-ai/.plans/299_npc_curiosity_self_play_runtime.md) — CGSP runtime (single-pool, this plan extends it to dual-pool)
- **Faithfulness gate:** [Plan 278](278_faithfulness_probe_modelless.md) — `FaithfulnessProbe` primitive (consolidation gate in Phase 4)
- **Collapse detector (reactive baseline):** [Plan 212](212_collapse_aware_adaptive_thinking.md) — `EntropyCollapse::inject_exploration` (dual-pool makes this proactive)
- **Same author lineage:** [Research 244](../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md) — Zhao et al. ICML 2026 faithfulness paper (G-Memory is DecentMem's baseline AND the system that silently ignores 60%+ of its memory)
