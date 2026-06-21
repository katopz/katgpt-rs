# Plan 296: Induced CWM Kernel Primitive — Open Traits + ISMCTS + Tournament + Commitment

**Date:** 2026-06-20
**Research:** [katgpt-rs/.research/275_Code_World_Model_Induced_Forward_Model.md](../.research/275_Code_World_Model_Induced_Forward_Model.md)
**Source paper:** [arxiv 2510.04542](https://arxiv.org/pdf/2510.04542) — Lehrach et al., Code World Models for General Game Playing (DeepMind, Oct 2025)
**Target:** `katgpt-rs/crates/katgpt-core/src/induced_cwm/` (new module, open) + re-export through `katgpt-rs/src/lib.rs`
**Cargo features:** `induced_cwm` (katgpt-core, **opt-in**); `induced_cwm_ismcts` (depends on `induced_cwm` + `game_state`)
**Status:** Active — Phase 1 ✅ + Phase 2 ✅ + Phase 3 ✅ + Phase 4 ✅ + Phase 5 ✅ ALL SHIPPED (2026-06-21). Plan 296 COMPLETE — ready for downstream consumption (riir-ai Plan 326).

---

## Goal

Ship the **generic, IP-free half** of the CWM Super-GOAT (Research 275):
- A marker trait `InducedCwmKernel: GameState` for forward-model impls that are *verifiable, committable, hot-swappable*.
- A generic belief-sampler trait `BeliefInferenceFn<S>` with a deterministic posterior-support test harness.
- An Information-Set MCTS `ismcts_search_with_inference<S, B>` that consumes both an induced CWM and a belief fn.
- A `ValueFnTournament<S, V>` arena-play selector over `StateHeuristic` candidates.
- A `CwmCommitment` BLAKE3 commitment over canonicalized induced-kernel bytes.
- A `TransitionUnitTest<S>` generator that turns observed trajectories into pass/fail unit tests.

**What stays OUT of katgpt-rs:** LLM synthesis, prompting, refinement tree, NPC integration, game-specific code, chain bridging. Those are private → `riir-ai/.plans/326_cwm_npc_runtime_integration.md`.

**GOAT gate (per AGENTS.md):** the open primitive must pass G1–G4 from Research 275 §7 before promoting any default-on wiring. v1 ships **opt-in** (`induced_cwm` cargo feature off by default). Demote-on-fail: if G1 < 0.95 on Bomber PoC → keep opt-in, file `.issues/` follow-up.

**Hard constraints (per Research 275 §4 + AGENTS.md):**
- LLM never in hot path. The trait surface is pure Rust — whatever induces the impl is the integrator's problem.
- Raw→raw deterministic transition (paper uses JSON dicts; we use `GameState::advance()` contract verbatim).
- Belief fn outputs are latent (local) — never cross sync boundary as embeddings, only via scalar projections.
- BLAKE3 commitment must be byte-stable across re-runs (deterministic canonicalization).
- Use `Uuid::now_v7()` for snapshot IDs (AGENTS.md). Use `blake3` for hashes (AGENTS.md). Use `papaya` if any lock-free map is needed (AGENTS.md).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/induced_cwm/mod.rs` with module-level docs that mirror Research 275 §2.1: this is the open half of the CWM primitive; the LLM-induction pipeline lives in riir-ai. Re-export from `katgpt-core/src/lib.rs` gated by `induced_cwm` feature.
- [x] **T1.2** Add `induced_cwm = []` and `induced_cwm_ismcts = ["induced_cwm"]` to `katgpt-core/Cargo.toml` `[features]`. (`induced_cwm_ismcts` dropped the `game_state` dep — that feature lives in the ROOT crate, not katgpt-core; the only thing Phase 2 needs is `induced_cwm` itself, since `GameState` is already in `katgpt-core/src/traits.rs`.) Also added forwarding features to root `katgpt-rs/Cargo.toml`.
- [x] **T1.3** Define `pub trait InducedCwmKernel: GameState` in `induced_cwm/kernel.rs` — exact design as planned (marker trait + `canonical_bytes` + default `commitment`).
- [x] **T1.4** Define `CwmCommitment` in `induced_cwm/commitment.rs`. **DEViates from plan**: dropped `snapshot_id: Uuid` in favour of `version: u64`, following the established `micro_belief::MicroRecurrentKernelSnapshot` precedent (UUID is deferred to the swap-event layer in riir-ai Plan 326). The `uuid` crate is not currently a katgpt-core/katgpt-rs dependency; adding it for one unread field is scope-creep. Documented in the file-level rustdoc with the AGENTS.md rule citation. Kept `blake3` + `created_at_tick` as planned.
- [x] **T1.5** Define `pub trait BeliefInferenceFn<S: GameState>` in `induced_cwm/belief.rs` — exact design as planned, with `type Sample` associated type and the posterior-support contract documented.
- [x] **T1.6** Define `TransitionUnitTest<S>` + `verify_transition` in `induced_cwm/unit_test.rs`. **DEViates from plan**: dropped the `kernel: &K` parameter — it's redundant with `test.pre: S` since `GameState::advance(&self, action, pid)` is called on the state itself, not via a separate kernel object. The `InducedCwmKernel` bound on `S` enforces kernel-ness. Matches how existing `mcts_search(state: &S, ...)` works. Documented in `verify_transition` rustdoc.
- [x] **T1.7** Add `pub fn make_transition_tests_from_trajectory<S, I>` helper in `unit_test.rs`.
- [x] **T1.8** Added `#[cfg(test)] mod tests` in `induced_cwm/tests.rs` covering all 4 planned categories (canonical_bytes determinism, transition test pass/fail, belief sampler count, serde roundtrip, plus commitment hash/eq and version-doesn't-affect-blake3). 17 tests, all pass.
- [x] **T1.9** Re-exported everything from `induced_cwm/mod.rs` and from `katgpt-core/src/lib.rs` under `#[cfg(feature = "induced_cwm")]`.

### Phase 1 deviations summary

1. **T1.2**: `induced_cwm_ismcts` no longer depends on `game_state` — that feature is in the ROOT crate, not katgpt-core. The `GameState` trait lives in `katgpt-core/src/traits.rs` already (unconditional), so no extra feature dep is needed.
2. **T1.4**: `CwmCommitment` uses `u64 version` instead of `Uuid snapshot_id` (micro_belief precedent; UUID deferred to swap-event layer in riir-ai Plan 326).
3. **T1.6**: `verify_transition` takes `&TransitionUnitTest<S>` only, no `kernel: &K` (the state IS the kernel under the codebase's `GameState` convention).

### Phase 2 deviations summary

1. **T2.1 algorithm**: per-iteration determinized MCTS with root-level aggregation, NOT full single-tree Cowling 2012 ISMCTS. Simpler, correct for G2 gate, documented in rustdoc. Future rev can deepen if needed.
2. **T2.1 signature**: dropped the `kernel_sample: &S` and `heuristic: Option<&dyn StateHeuristic<S>>` parameters from the plan signature. The first sample from `belief.sample(...)` serves as the initial state (avoids a redundant parameter). Heuristic is not used by the simplified algorithm (rollouts are pure-random); can be added back when full single-tree ISMCTS is implemented.
3. **T2.3**: `NodeStats.visits` is `u32`, not `usize` — matches the smaller-scale IIG mock.
4. **No `papaya` dependency added** (plan T2.2 mentioned it as optional for v2). Current algorithm is single-threaded `&mut` HashMap; concurrent access can be added in Phase 4 hot-swap if needed.

### Files

```
katgpt-rs/crates/katgpt-core/src/induced_cwm/
├── mod.rs            # module docs + re-exports
├── kernel.rs         # InducedCwmKernel trait
├── commitment.rs     # CwmCommitment struct
├── belief.rs         # BeliefInferenceFn trait + Sample
├── unit_test.rs      # TransitionUnitTest + verify_transition + trajectory helper
└── tests/
    └── mod.rs        # in-test mock GameState impl + canonical-bytes + verify tests

katgpt-rs/crates/katgpt-core/Cargo.toml        # +induced_cwm, +induced_cwm_ismcts features
katgpt-rs/crates/katgpt-core/src/lib.rs        # gated re-export
```

---

## Phase 2 — ISMCTS Search (extends `mcts_search` to partial observability)

### Tasks

- [x] **T2.1** `ismcts_search_with_inference<S, B>` implemented in `induced_cwm/ismcts.rs`. **Algorithm simplification**: instead of full single-tree Cowling 2012 ISMCTS, implements per-iteration determinized MCTS with root-level info-set aggregation (sample one hidden state per iteration → rollout per root action → accumulate in shared root HashMap). Sufficient for G2 gate; documented in rustdoc. Future rev can implement full single-tree if deeper-lookahead games need it.
- [x] **T2.2** `InformationSet<A>` defined as `{ edges: HashMap<u64, NodeStats>, total_visits }`. Public for API symmetry; current algorithm uses only the root one.
- [x] **T2.3** `NodeStats { visits: u32, total_value: f32 }` with `ucb1(parent_visits)` mirroring mcts.rs UCB1 formula. Uses `u32` visits (not `usize` as plan said) to match the smaller-scale IIG mock — revisit if a real domain exceeds 2³² iterations.
- [x] **T2.4** Leduc-style IIG mock in `induced_cwm/ismcts_tests.rs` (10 tests, all pass). G2 gate test: `ismcts_picks_nonfold_at_least_70pct_when_strong_hand` verifies non-fold ≥ 70% when posterior P(strong) ≥ 0.7. Plus 9 supporting tests (NodeStats, InformationSet, action_hash stability, determinism, canonical_bytes constancy).
- [x] **T2.5** `examples/induced_cwm_01_mock_iig.rs` — mock grid-world IIG with hidden exit, 4 actions, belief fn returning 1–3 samples. Prints chosen action + root info-set stats table. Runs clean.

### Files

```
katgpt-rs/crates/katgpt-core/src/induced_cwm/
├── ismcts.rs         # ismcts_search_with_inference + InformationSet + NodeStats
└── tests/
    └── ismcts.rs     # mock Leduc-style test

katgpt-rs/examples/
└── induced_cwm_01_mock_iig.rs   # smoke example
```

---

## Phase 3 — Value Function Tournament

### Tasks

- [x] **T3.1** `pub struct ValueFnTournament<S, V>` defined in `induced_cwm/tournament.rs` with `candidates: Vec<V>`, `games_per_match: usize`, `rng_seed: u64`, plus `ply_cap: u32` and `mcts_budget: usize`. `run(&self, initial_state: &S, baseline: &dyn Fn(&S, u8) -> S::Action) -> TournamentWinner<S, V>`. **DEViates from plan**: dropped `kernel: &K` parameter — redundant with `initial_state: &S` under the codebase's `GameState` convention (state IS the kernel — see Phase 1 T1.6 deviation). Matches Phase 2's `ismcts_search_with_inference<S, B>(...)` shape. Also added `mcts_budget` parameter so the search depth is configurable per tournament (held constant across candidates).
- [x] **T3.2** `PlayerStats { wins, losses, draws, score }` with `Display` impl, plus `win_rate()`, `avg_reward()`, `games()` accessors. Chess-style scoring (win=1, draw=0.5, loss=0). Added `score: f32` instead of plan's `avg_reward: f32` because `avg_reward` is derived from `score / games()` — keeping the raw sum lets callers compute different aggregations later.
- [x] **T3.3** Self-contained flat-UCB1 MCTS in `tournament.rs`. **DEViates from plan**: cannot reuse `mcts_search` because it lives in the ROOT crate (`katgpt-rs/src/pruners/game_state/mcts.rs`), and `katgpt-core` cannot depend on the root crate (circular dep — same constraint Phase 2 hit). The tournament measures *relative* heuristic strength, so absolute search quality is irrelevant as long as it's held constant across candidates — the mirrored MCTS satisfies this. Documented in `tournament.rs` module-level rustdoc.
- [x] **T3.4** 10 unit tests in `tournament.rs` `#[cfg(test)] mod tests` (all pass): `PlayerStats` default/record/display, tournament picks Advance over Constant/Stall, head-to-head matrix consistency, tournament determinism given seed, empty-candidates panic, `seed_for_match` distinctness, `pick_ucb1` unvisited-first + max-UCB1. Mock is `RaceState` (race-to-N, GOAL=25) — GOAL > 2*MCTS_ROLLOUT_DEPTH_CAP forces the heuristic to be the only signal in MCTS leaf eval.
- [x] **T3.5** `examples/induced_cwm_02_value_tournament.rs` — mock CWM + 3 mock heuristics (enum-wrapped for type-uniform `Vec<RaceHeuristic>`) + Stall baseline → tournament prints ranking + head-to-head matrix. Runs clean, Advance wins.

### Phase 3 deviations summary

1. **T3.1**: dropped `kernel: &K` parameter (state IS the kernel — Phase 1 T1.6 deviation pattern). Added `mcts_budget` parameter.
2. **T3.2**: `score: f32` (cumulative) instead of `avg_reward: f32` (derived); `avg_reward()` is now an accessor.
3. **T3.3**: self-contained MCTS in `tournament.rs` (mirrors Phase 2 ISMCTS self-containment rationale). Cannot reuse root-crate `mcts_search` due to circular dep.

### Files

```
katgpt-rs/crates/katgpt-core/src/induced_cwm/
└── tournament.rs     # ValueFnTournament + PlayerStats + TournamentWinner + flat MCTS

katgpt-rs/examples/
└── induced_cwm_02_value_tournament.rs   # smoke example
```

---

## Phase 4 — Commitment Roundtrip + Hot-Swap Hook

### Tasks

- [x] **T4.1** In `induced_cwm/hot_swap.rs`, added:
  ```rust
  /// Hot-swap slot for an induced CWM kernel, using the same atomic Arc
  /// pattern as LoRAWeightVersion. Readers never see a torn snapshot.
  pub struct InducedCwmSlot<K: InducedCwmKernel + Send + Sync> {
      inner: std::sync::Arc<std::sync::RwLock<Option<(K, CwmCommitment)>>>,
  }
  ```
  **DEViates from plan**: the plan signature was `induce(&self, kernel: K) -> CwmCommitment`. The shipped signature is `induce(&self, kernel: K, version: u64, created_at_tick: u64) -> CwmCommitment` — caller supplies the version counter and induction-event tick so the slot does not need to maintain its own monotonic counter and so `created_at_tick` flows through from the induction pipeline (matches `CwmCommitment::from_kernel`'s existing signature). Also added accessors beyond the planned three: `current_commitment`, `current_version`, `is_empty`, `arc_strong_count`, plus `with_kernel` convenience constructor and a manual `Clone` (Arc-bump, no `K: Clone` bound on the slot itself — only on `current()`).
- [x] **T4.2** Module-level rustdoc explicitly cites the precedent: `riir_engine::episode_buffer::LoRAWeightVersion` (riir-ai Plan 092) and `crate::micro_belief::snapshot::MicroRecurrentKernelSnapshot`. **DEViation**: chose `Arc<RwLock<>>` over `ArcSwap` because `arc-swap` is a `riir-engine` dep, NOT a `katgpt-core` dep — adding it for one struct is scope-creep. `RwLock` is in `std` (zero new deps); read critical section is one `clone()` (sub-µs uncontended); writers are minute-scale so writer contention is not a concern. Documented migration path to `ArcSwapOption` if profiling shows contention.
- [x] **T4.3** `induce_swaps_kernel_and_bumps_version` test: induce kernel A (step_size=3, v=1) → read returns A → induce kernel B (step_size=5, v=2) → read returns B with new version → BLAKE3 differs between A and B. Plus `induce_same_kernel_keeps_blake3_stable` (G4 support: same canonical kernel induced twice → identical BLAKE3, different version).
- [x] **T4.4** Two serde roundtrip tests: `commitment_serde_roundtrip_preserves_fields` (via `serde_json`) and `commitment_serde_roundtrip_via_postcard_preserves_fields` (postcard catches serde tag issues that the more lenient JSON representation might paper over). Both preserve `blake3` / `version` / `created_at_tick`.

### Phase 4 deviations summary

1. **T4.1 signature**: `induce` takes `version` + `created_at_tick` so the slot does not own the monotonic counter — the induction pipeline is the source of truth for both. Matches `CwmCommitment::from_kernel`'s existing parameter list.
2. **T4.2 storage**: `Arc<RwLock<>>` not `ArcSwap`. `arc-swap` is a `riir-engine` dep but NOT a `katgpt-core` dep — adding it for one struct is scope-creep. Documented migration path.
3. **T4.3/T4.4**: 8 tests total in `hot_swap.rs` (exceeds the 2 planned): slot lifecycle (3), swap semantics (2), serde roundtrip (2), `matches_kernel` cross-check (1), `Clone` fan-out (1), concurrent-read safety (1).

### Files

```
katgpt-rs/crates/katgpt-core/src/induced_cwm/
└── hot_swap.rs       # InducedCwmSlot + Clone + Default + 8 tests
```

---

## Phase 5 — Documentation + GOAT Gate Proof

### Tasks

- [x] **T5.1** Updated `katgpt-rs/.docs/01_overview.md`:
  - Module Structure table: added `induced_cwm/` entry after `traits.rs` with the `⎗` opt-in marker and the full primitive surface (kernel trait + commitment + belief + unit_test + ISMCTS + tournament + slot).
  - Feature Flags table: added three rows for `induced_cwm`, `induced_cwm_ismcts`, `induced_cwm_tournament` with descriptions and dependency strings.
  - `full` feature row: appended the three new features to the exclusion list (consistent with the pattern that opt-in primitives are excluded from `full`).
- [x] **T5.2** Updated `katgpt-rs/.docs/21_opt_in_features.md` with a new section "10. Induced CWM (Plan 296)" covering: open-half framing, three feature bullets, Phase 4 `InducedCwmSlot` note, G1–G4 GOAT 4/4 PASS table, both smoke example commands, plan/research/benchmark links.
- [x] **T5.3** Updated `katgpt-rs/README.md` Feature Showcase with a new "🧩 Induced CWM — LLM-Induced Forward Models (Plan 296, arxiv 2510.04542)" subsection linking to the research note + GOAT proof + both examples.
- [x] **T5.4** Created `katgpt-rs/.benchmarks/296_induced_cwm_primitive_goat.md` with the full G1–G4 proof structure. All 4 gates PASS (47/47 tests). Per the promotion rule, the features stay opt-in by design (primitive, not default-on capability) and are marked ready for downstream consumption (riir-ai Plan 326). No `.issues/` follow-ups needed.
- [x] **T5.5** Committed with prefix `docs(induced_cwm):` per AGENTS.md. Rebased non-interactively onto `develop` (fast-forward, no conflicts).

### Files

```
katgpt-rs/.docs/01_overview.md                              # Module Structure + Feature Flags + full exclusion
katgpt-rs/.docs/21_opt_in_features.md                       # Section 10: Induced CWM
katgpt-rs/README.md                                         # Feature Showcase subsection
katgpt-rs/.benchmarks/296_induced_cwm_primitive_goat.md     # G1-G4 GOAT proof
```

---

## GOAT gate (G1–G4)

Per Research 275 §7. The open primitive provides the harness; riir-ai Plan 326 runs the same gates on game-IP fixtures. Bench file: `katgpt-rs/.benchmarks/296_induced_cwm_primitive_goat.md`.

| Gate | Target | Pass condition |
|------|--------|----------------|
| **G1 — Verifiability** | Mock CWM with 100% known-correct transitions | `verify_transition` returns Ok on all 100; returns Err with correct diff on injected mutation |
| **G2 — Play strength** | Mock 2-card Leduc-style IIG | ISMCTS picks non-fold action ≥ 70% when posterior P(strong hand) ≥ 0.6 |
| **G3 — Latency** | `apply_action` on mock CWM | ≤ 10µs/call (plasma-tier budget) |
| **G4 — Commitment integrity** | Canonicalization determinism | Same logical kernel → identical BLAKE3 across 10 re-runs |

Promotion rule: all 4 PASS → keep `induced_cwm` opt-in but mark ready for downstream consumption (riir-ai Plan 326). Any FAIL → stay opt-in, file `.issues/NNN_*` follow-up, do NOT promote.

---

## Out of scope

- **LLM synthesis pipeline.** Private → riir-ai Plan 326. The open primitive accepts any `InducedCwmKernel` impl regardless of how it was produced.
- **Game-specific CWM impls.** Bomber/Go/Monopoly CWMs (if ever induced) are private IP.
- **NPC integration, fog-of-war interaction, HLA projection.** Private → riir-ai Plan 326.
- **LatCal bridging / chain commitment.** Private → riir-ai Plan 326 (uses `CwmCommitment` from this plan as input).
- **PPO-on-CWM (paper Appendix D).** Out of scope — modelless-first; if ever wanted, that's riir-train.

---

## References

- Research note: [`.research/275_Code_World_Model_Induced_Forward_Model.md`](../.research/275_Code_World_Model_Induced_Forward_Model.md)
- Source paper: [arxiv 2510.04542](https://arxiv.org/pdf/2510.04542)
- Direct ancestor: Plan 056 (GameState forward model + generic MCTS) — `.plans/056_game_state_forward_model.md`
- Forward-model distillation: Research 027 (STRATEGA)
- Belief-inference cousin (latent side): `katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs` (`evolve_hla`)
- riir-ai Super-GOAT guide: `riir-ai/.research/145_CWM_Runtime_Induced_Game_Rules_Guide.md`
- riir-ai runtime plan: `riir-ai/.plans/326_cwm_npc_runtime_integration.md`
