# Plan 351: Density-Aware Zone Routing — Mobility Scheduling + LRU + Outer-First

**Date:** 2026-06-29
**Research:** [katgpt-rs/.research/350_density_aware_compute_scheduling.md](../.research/350_density_aware_compute_scheduling.md)
**Source papers:**
- [Treuille, Cooper, Popović (2006) — "Continuum Crowds"](https://www.cs.ubc.ca/~van/papers/2006-continuum-crowds/continuum-crowds-siggraph-2006.pdf) (SIGGRAPH 2006) — density-dependent speed field
- van Toll, Pettré et al. — density-aware navigation meshes (line of work)
- Fokker-Planck / continuity equation on cochains — internal mapping via [Plan 314](314_stokes_calculus_wrappers.md)
**Target:** `katgpt-rs/crates/katgpt-core/src/zone_density.rs` (new module, ~500 LOC + tests) + Cargo feature `zone_density_routing` (opt-in, NOT default until G5a+G5b+G5c pass).
**Status:** Active — Phase 1 starting.

---

## Goal

Ship three modelless primitives that turn a per-zone population count into a compute-scheduling decision, then benchmark the combined primitive against the mean-aggregation baseline from Plan 001 to settle the deferred G5 routing-quality gate.

**Three primitives:**

1. **`zone_density_classify()`** — per-zone `(mobility_weight, tier_class, cache_key)` from raw population count. The fundamental classifier. Sigmoid-based (per AGENTS.md), deterministic, zero-alloc.
2. **`schedule_outer_first()`** — stable ascending-density sort of zone indices. Outer (sparse, high-mobility, high-entropy) zones compute first.
3. **`ZoneDensityCache<V>`** — `papaya::HashMap`-backed LRU with three invalidation rules (tier transition, density drift > δ, TTL expiry). Optional mass-divergence-driven stampede invalidation hook.

**GOAT gate:** three sub-gates (G5a routing quality, G5b compute saved, G5c stampede correctness). Promote `zone_density_routing` to default only if **all three pass**. If G5a passes alone, promote as opt-in-friendly (primitive ships, stays opt-in). If G5a fails, demote — keep the code for cache/scheduling reuse but document the routing-quality miss.

**Why this matters:** the current G5 verdict on Plan 001 is **DEFERRED** — cluster-and-distribute PCA passed G1–G4+G6 but routing quality needs riir-ai integration. This plan lands the missing integration layer (the *physical* compute scheduler) that the cognitive gating (Plan 305) does not provide. The two compose as orthogonal siblings.

## Non-Goals

- ❌ **NO replacement of `ZoneGatingProfile` (Plan 305).** That module is cognitive-tiered gating (`tau, beta, reest_budget` per zone for functor coherence). This plan is physical-tiered scheduling (`mobility, tier, cache_key` per zone for compute budget). They compose; they do not overlap.
- ❌ **NO training.** Pure modelless inference. Per Research 350 §2.2, all parameters are closed-form derivable from Plan 305's existing tier thresholds.
- ❌ **NO Super-GOAT guide.** Verdict is GOAT (Research 350 §4). Per skill §1.5, no private riir-ai guide is created in this plan. Super-GOAT-promotion path is documented but conditional on G5 revealing emergent crowd behavior.
- ❌ **NO high-dim shard compression via density commitment.** Curse of dimensionality (per AGENTS.md and Research 296 §3.5). This primitive operates on per-zone scalar density, not on `NeuronShard` embeddings.
- ❌ **NO UQ claim.** Mobility is a deterministic weight in `[0, 1]`, not a probability distribution, predictive interval, or calibrated uncertainty. The "Report the Floor" rule (Plan 340) does NOT apply — no conformal-naive floor benchmark needed. Documented explicitly to prevent a future reviewer from re-introducing this requirement by mistake.

## Constraint Checklist (per AGENTS.md + skill)

All constraints verifiable by construction during Phase 1.

- [ ] Modelless (inference-time only, no backprop) — ✓ by construction (pure functions + papaya cache)
- [ ] Latent-to-latent preferred (sigmoid not softmax) — ✓ (mobility = `fast_sigmoid(-β·(ρ−ρ₀))`)
- [ ] Freeze/thaw over fine-tuning — ✓ (no weight mutation; cache is data, not weights)
- [ ] 5-repo discipline (open primitive in katgpt-rs) — ✓
- [ ] SOLID, DRY, modular, generic, decouple — ✓ (generic over `V` for cache value; no game-IP semantics)
- [ ] Zero-alloc hot path — ✓ (caller-owned output slices; papaya is lock-free)
- [ ] CPU/SIMD auto-route — ✓ (reuses `katgpt_core::simd::fast_sigmoid`)
- [ ] File < 2048 lines — ✓ (target ~500 LOC + tests)
- [ ] `papaya` for lock-free HashMap — ✓ (`ZoneDensityCache` backend)
- [ ] `Uuid::now_v7()` if any UUIDs — N/A (none needed; cache key is composite u64, not UUID)
- [ ] `blake3` for commitment — N/A (no commitment in this primitive; density bucket is decodable composite, not a hash)
- [ ] `match` over `if` — ✓ (tier classification, cache bypass)

---

## Phase 1 — Unblocking Skeleton (CORE)

Three pure functions + one cache struct. No game semantics. No allocations in the hot path (caller owns output slices; papaya is the only allocator, and it's lock-free).

### Tasks

- [x] **T1.1** Add `pub mod zone_density;` to `katgpt-rs/crates/katgpt-core/src/lib.rs` behind `#[cfg(feature = "zone_density_routing")]`. Add Cargo feature `zone_density_routing = []` (opt-in, NO default features) to `katgpt-rs/Cargo.toml` AND `katgpt-rs/crates/katgpt-core/Cargo.toml`. Add `papaya` to `katgpt-rs/crates/katgpt-core/Cargo.toml` dependencies if not already present (check — it's used pervasively in sibling repos but may not be in katgpt-core yet; if absent, add `papaya = "0.x"` matching sibling-repo version).
  - `papaya = { version = "0.2", optional = true }` already present in katgpt-core/Cargo.toml (L25) — reused via `dep:papaya`.
  - Feature added to BOTH manifests: `zone_density_routing = ["dep:papaya"]` (core) + `zone_density_routing = ["katgpt-core/zone_density_routing"]` (root re-export).
- [x] **T1.2** Implement `DensityTier` enum in `zone_density.rs` (+ `from_cache_key_high` decoder helper):
  ```rust
  /// Per-zone physical compute tier. **Distinct from `ZoneGatingTier`** (Plan 305)
  /// which is cognitive. This is physical: dense = cached (NPCs can't move),
  /// sparse = full compute (high movement freedom).
  #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
  #[repr(u8)]
  pub enum DensityTier {
      /// Sparse periphery — full compute every tick. High mobility, high entropy.
      Sparse = 0,
      /// Transitional — moderate compute, cached with short TTL.
      Transitional = 1,
      /// Dense core — LRU-cached, low compute. NPCs physically constrained.
      Dense = 2,
  }
  ```
  Field-less + `#[repr(u8)]` per AGENTS.md (1-byte size).
- [x] **T1.3** Implement `DensityClassifyConfig` with `Default`:
  ```rust
  #[derive(Clone, Copy, Debug)]
  pub struct DensityClassifyConfig {
      /// Sigmoid midpoint density. Default 5.0 (Plan 305 midpoint between
      /// transitional=1.0 and dense=10.0).
      pub rho0: f32,
      /// Sigmoid slope. Default 0.5 — puts the 0.1→0.9 mobility transition
      /// across roughly one Plan-305 tier step.
      pub beta: f32,
      /// Mobility threshold above which a zone is `Sparse`. Default 0.7.
      pub tier_high: f32,
      /// Mobility threshold below which a zone is `Dense`. Default 0.3.
      /// (Between `tier_low` and `tier_high` = `Transitional`.)
      pub tier_low: f32,
      /// Density drift beyond which a cached entry is invalidated even if
      /// the tier hasn't changed. Default 2.0 (one Plan-305 tier step).
      pub cache_invalidation_delta: f32,
  }
  
  impl Default for DensityClassifyConfig {
      fn default() -> Self {
          Self { rho0: 5.0, beta: 0.5, tier_high: 0.7, tier_low: 0.3, cache_invalidation_delta: 2.0 }
      }
  }
  ```
- [x] **T1.4** Implement `DensityClassifyReport` and `zone_density_classify()` (+ `decode_cache_key` inverse helper):
  ```rust
  #[derive(Debug, Default)]
  pub struct DensityClassifyReport {
      pub n_sparse: usize,
      pub n_transitional: usize,
      pub n_dense: usize,
      pub mean_mobility: f32,
  }
  
  /// Per-zone density → (mobility, tier, cache_key). Deterministic, zero-alloc.
  ///
  /// # Panics
  /// Debug-asserts all output slices are `>= population.len()`.
  pub fn zone_density_classify(
      population: &[f32],
      config: &DensityClassifyConfig,
      out_mobility: &mut [f32],
      out_tier: &mut [DensityTier],
      out_cache_key: &mut [u64],
  ) -> DensityClassifyReport
  ```
  Algorithm: single-pass over `population`. Per zone: `m = fast_sigmoid(-beta * (rho - rho0))`; tier via `match` on `m` thresholds; cache_key = `(tier as u64) << 32 | density_bucket` where `density_bucket = (rho * 0.5).floor() as u64` (buckets of size 2.0). Use `crate::simd::fast_sigmoid` (already shipped, pervasively used in riir-ai).
- [x] **T1.5** Implement `schedule_outer_first()`:
  ```rust
  /// Sort zone indices ascending by density. Outer (sparse) zones come first.
  /// O(Z log Z), stable. Uses caller-provided scratch to avoid allocation.
  pub fn schedule_outer_first(
      population: &[f32],
      out_order: &mut [u32],
      scratch: &mut Vec<(u32, f32)>,
  )
  ```
  Implementation: `scratch.clear(); scratch.reserve(population.len()); for (i,&rho) in population.iter().enumerate() { scratch.push((i as u32, rho)); } scratch.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal)); for (i,&(z,_)) in scratch.iter().enumerate() { out_order[i] = z; }`. Stable sort preserves within-tier ordering for determinism.
- [x] **T1.6** Implement `ZoneDensityCache<V>` (+ `ttl_ticks()` / `is_empty()` accessors):
  ```rust
  use papaya::HashMap;
  
  /// Per-zone LRU cache for dense-tier values. Lock-free via papaya.
  ///
  /// **Sparse-tier zones are NEVER cached** — always full recompute.
  /// **Transitional/Dense tiers** are cached with TTL + tier-stability +
  /// density-drift invalidation rules.
  pub struct ZoneDensityCache<V: Clone> {
      map: HashMap<u32, CacheEntry<V>>,
      ttl_ticks: u64,
  }
  
  #[derive(Clone)]
  struct CacheEntry<V> {
      value: V,
      cached_density: f32,
      cached_tier: DensityTier,
      cached_at_tick: u64,
  }
  
  impl<V: Clone> ZoneDensityCache<V> {
      pub fn new(ttl_ticks: u64) -> Self { Self { map: HashMap::new(), ttl_ticks } }
      
      /// Returns cached value iff (1) tier is Transitional/Dense, (2) tier
      /// hasn't transitioned, (3) density drift < delta, (4) entry within TTL.
      /// Sparse-tier zones always return None.
      pub fn get_or_invalidate(
          &self, zone_id: u32, current_density: f32, current_tier: DensityTier,
          current_tick: u64, invalidation_delta: f32,
      ) -> Option<V>;
      
      /// Insert. Sparse-tier zones are silently skipped (never cached).
      pub fn insert(&self, zone_id: u32, density: f32, tier: DensityTier, tick: u64, value: V);
      
      /// Bulk-invalidate all entries. Called on stampede detection
      /// (caller decides trigger — typically `belief_mass_divergence > τ`).
      pub fn invalidate_all(&self);
      
      /// Current cache size. For diagnostics / G5b benchmark.
      pub fn len(&self) -> usize;
  }
  ```
  Get path uses `match current_tier { DensityTier::Sparse => return None, _ => {} }` early-return (per AGENTS.md "early returns").
- [x] **T1.7** Run `cargo check --features zone_density_routing` — clean, no warnings. Also verified: `--all-features` clean, default-features clean, root crate re-export compiles.
- [x] **T1.8** Run `cargo test -p katgpt-core --features zone_density_routing --lib zone_density::` — 16/16 Phase 1 smoke tests pass (0.00s). Full lib suite: 689/689 pass (673 pre-existing + 16 new), zero regressions.

**Exit:** all three primitives compile, type-check, basic smoke tests green.

---

## Phase 2 — Unit Tests (Correctness, not perf)

Each primitive gets ≥4 unit tests: identity case, scaling, edge case, determinism. ≥18 tests total.

### Tasks

- [ ] **T2.1** `zone_density_classify` tests:
  - [ ] **T2.1.1** **Monotonicity**: as `ρ` increases from 0 to 50, `mobility` monotonically decreases (or stays flat) — verified at 20 evenly-spaced sample points.
  - [ ] **T2.1.2** **Midpoint**: at `ρ = rho0 = 5.0`, `mobility ≈ 0.5` (sigmoid symmetry) — tolerance 1e-5.
  - [ ] **T2.1.3** **Tier boundaries**: at `mobility = tier_high = 0.7` (resolve `ρ` from sigmoid inverse), zone classifies as `Sparse` (strict `>`); at `mobility = tier_low = 0.3`, classifies as `Transitional`; at `mobility = 0.0` (saturated dense), classifies as `Dense`.
  - [ ] **T2.1.4** **Cache key decode**: given a `cache_key`, decode `(tier, density_bucket)` by bit-shift; verify round-trip matches input tier and bucket.
  - [ ] **T2.1.5** **Determinism**: same input + same config → bit-identical output across two calls (G3-style).
  - [ ] **T2.1.6** **Empty input**: `population = &[]` returns `DensityClassifyReport::default()` without panic, writes nothing.
- [ ] **T2.2** `schedule_outer_first` tests:
  - [ ] **T2.2.1** **Ascending order**: for `population = [10.0, 1.0, 5.0, 0.5]`, `out_order = [3, 1, 2, 0]` (indices sorted by ascending density).
  - [ ] **T2.2.2** **Stable within-tier ties**: for `population = [5.0, 5.0, 5.0]`, `out_order = [0, 1, 2]` (original order preserved — stable sort).
  - [ ] **T2.2.3** **Single zone**: `population = [3.0]` → `out_order = [0]`.
  - [ ] **T2.2.4** **Empty input**: writes nothing, doesn't panic.
  - [ ] **T2.2.5** **Scratch reuse**: calling twice with the same `scratch` produces identical results (the `clear()` contract holds).
- [ ] **T2.3** `ZoneDensityCache` tests:
  - [ ] **T2.3.1** **Sparse bypass**: insert with `tier=Sparse` → `len() == 0` (silently dropped).
  - [ ] **T2.3.2** **Transitional hit**: insert `(zone=1, density=4.0, tier=Transitional, tick=0, value="v0")`; `get(zone=1, density=4.0, tier=Transitional, tick=0, delta=2.0)` returns `Some("v0")`.
  - [ ] **T2.3.3** **Dense hit**: same as T2.3.2 with `tier=Dense`.
  - [ ] **T2.3.4** **Tier transition invalidates**: insert `tier=Dense`; `get` with `tier=Transitional` returns `None`.
  - [ ] **T2.3.5** **Density drift invalidates**: insert `density=10.0`; `get` with `density=15.0, delta=2.0` returns `None` (drift=5.0 > 2.0).
  - [ ] **T2.3.6** **TTL expiry**: insert at tick=0; `get` at tick=`ttl+1` returns `None`.
  - [ ] **T2.3.7** **`invalidate_all`**: insert 5 entries; `invalidate_all()`; all subsequent `get`s return `None`; `len() == 0`.
  - [ ] **T2.3.8** **Concurrent access** (if `papaya` exposes a sync test harness): two threads, one inserting, one getting, no deadlock, no panic. If papaya's test harness isn't easily reusable, document as "concurrency trusted to papaya; no extra test".
- [ ] **T2.4** Run full katgpt-core test suite: `cargo test -p katgpt-core --features zone_density_routing --lib` — all pre-existing tests still pass + new tests pass. Zero warnings.

**Exit:** three primitives verified correct on identities, boundaries, determinism, cache invalidation rules.

---

## Phase 3 — GOAT Gate (G5a + G5b + G5c)

Three sub-gates. **Promotion requires all three pass.** Each sub-gate has a numeric target and an honest failure mode documented.

### Tasks

- [ ] **T3.1** **G5a — Routing quality (Shannon entropy of events).** Target: **≥ +15% Shannon entropy** of event types in a 60s sim vs the mean-aggregation baseline from Plan 001 G5.
  - **Benchmark file**: `katgpt-rs/benches/bench_002_density_routing_goat.rs`.
  - **Setup**: synthetic crowd of N=10,000 NPCs across Z=64 zones with a Gaussian spatial density profile (dense core at center, sparse periphery). Each NPC emits one event per tick, event type = `{move, idle, interact, queue}` chosen stochastically weighted by local mobility.
  - **Baseline A (mean-aggregation)**: events drawn from the zone's mean-aggregated mobility (Plan 001's per-zone mean).
  - **Candidate B (density-aware)**: events drawn from the per-zone sigmoid-weighted mobility + outer-first scheduling (this plan).
  - **Metric**: Shannon entropy `H = -Σ p_e log p_e` over event types, averaged over 60s × 20Hz = 1200 ticks.
  - **Target**: `(H_B - H_A) / H_A ≥ 0.15`.
  - **Honest failure mode**: if sparse zones don't produce more event diversity than dense zones (counter-intuitively), G5a fails. The Treuille theory predicts they should, but a synthetic benchmark may not capture real game dynamics. **If G5a fails**, the primitive is retained for G5b (compute saving) but NOT promoted for routing quality; document the miss and move on.
- [ ] **T3.2** **G5b — Compute saved via dense-tier cache.** Target: **≥ 50% reduction in per-tick compute on dense-dominated workloads** vs always-recompute baseline.
  - **Setup**: same synthetic crowd. Measure wall-clock per-tick cost of (a) baseline: every zone's projections recomputed every tick; (b) candidate: dense-tier zones served from `ZoneDensityCache`, sparse-tier zones recomputed.
  - **Workload mix**: 70% of zones Dense, 20% Transitional, 10% Sparse (urban-core-dominated).
  - **Metric**: `(time_baseline - time_candidate) / time_baseline` per tick, averaged over 1200 ticks.
  - **Target**: `≥ 0.50`.
  - **Stampede stress test**: at tick 600, inject a stampede (10× density spike in a Dense zone, persisting 50 ticks). Measure cache hit rate during stampede (should drop to ~0) and recovery time after stampede ends (cache rebuilds within ~64 ticks = `ttl_ticks_dense`).
  - **Honest failure mode**: if papaya's lock overhead exceeds the compute saving on small cached values (e.g., 8-byte projections), G5b fails. Mitigation: only cache values ≥ 32 bytes (document a `min_cacheable_size` config knob if needed).
- [ ] **T3.3** **G5c — Stampede invalidation correctness.** Target: **zero stale reads during a density-class transition**; **≤ 1 tick invalidation latency**.
  - **Setup**: same synthetic crowd. At tick 300, force a tier transition in zone 5 (Dense → Sparse via density drop). At tick 600, force stampede (Sparse → Dense via 10× spike).
  - **Metric**: for each transition tick, count reads from `get_or_invalidate` that return `Some` after the tier has already changed. **Must be 0.**
  - **Mass-divergence hook (optional, bonus)**: integrate `belief_mass_divergence` (Plan 314) as a stampede detector. If `belief_mass_divergence(ρ·v_flow) > τ` at tick T, call `cache.invalidate_all()` before T+1's reads. Measure: does this catch transitions earlier than the per-zone tier check alone?
  - **Target**: 0 stale reads in the core gate. Mass-divergence hook is a bonus (not gating).
  - **Honest failure mode**: if papaya's eventual consistency allows a stale read between insert and invalidate on a different thread, G5c fails. Mitigation: pin the cache access to a single thread (acceptable for the hot path), OR use `pin()` explicitly per papaya docs.
- [ ] **T3.4** Write benchmark summary at `katgpt-rs/.benchmarks/351_density_routing_goat.md`. Honest results — pass or fail, document the root cause if any sub-gate fails.
- [ ] **T3.5** **Promotion decision**:
  - All three pass → promote `zone_density_routing` to default in `katgpt-rs/Cargo.toml`. Update `katgpt-rs/AGENTS.md` feature list.
  - G5a fails, G5b+G5c pass → keep opt-in. Document as "compute optimization (caching), not routing improvement". File `.issues/NNN_density_routing_g5a_miss.md` with the root-cause analysis.
  - G5b fails → keep opt-in. The primitive is technically correct but papaya overhead exceeds the gain. File `.issues/NNN_density_cache_overhead.md`.
  - G5c fails → **do not promote.** Stale reads during stampede are a correctness bug, not a perf miss. Fix before any promotion.
  - **Demote the loser**: if `zone_density_routing` wins (≥2 of 3 gates pass), the mean-aggregation path in Plan 001 G5 stays as the documented baseline. No code removal — just documentation.

**Exit:** all three sub-gates have verdicts. Promotion decided. Honest docs written.

---

## Phase 4 — riir-ai Wiring (to be filed as separate riir-ai plan)

**Not started.** Filed as a riir-ai plan once Phase 1–3 land. Phase 4 is riir-engine work, not katgpt-rs work.

### Tasks (forward-references for the riir-ai plan)

- [-] **T4.1** Create `riir-ai/crates/riir-engine/src/latent_functor/zone_scheduling.rs` as a **sibling** module to `zone_gating.rs`. Imports `katgpt_core::zone_density::{zone_density_classify, schedule_outer_first, ZoneDensityCache, DensityTier, DensityClassifyConfig}`.
- [-] **T4.2** Add `NpcFunctorRuntime::with_physical_scheduling()` constructor that composes the cognitive tier (existing `ZoneGatingProfile`) with the physical tier (new `DensityClassifyConfig` + `ZoneDensityCache`). The two run in sequence: cognitive gating picks `tau/beta/budget`, physical scheduling picks `compute-this-tick: bool` + cache hit/miss.
- [-] **T4.3** Wire `belief_mass_divergence` (Plan 314) as the optional stampede detector. When it exceeds τ, call `cache.invalidate_all()`.
- [-] **T4.4** Run the actual 60s sim with real game zones (not synthetic). Compare to the mean-aggregation baseline. This is the *real* G5 verdict (T3.1 is the synthetic proxy).
- [-] **T4.5** If T4.4 reveals emergent crowd behavior (lanes, queues, stampedes from local density gradients), file the Super-GOAT-promotion note + riir-ai guide (per Research 350 §4 conditional path).

---

## Architecture

```
katgpt-rs/crates/katgpt-core/src/
└── zone_density.rs               NEW (~500 LOC + ~250 LOC tests)
    // Types
    pub enum DensityTier { Sparse, Transitional, Dense }       #[repr(u8)]
    pub struct DensityClassifyConfig { rho0, beta, tier_high, tier_low, cache_invalidation_delta }
    pub struct DensityClassifyReport { n_sparse, n_transitional, n_dense, mean_mobility }
    pub struct ZoneDensityCache<V: Clone> { ... }              papaya-backed
    
    // Functions
    pub fn zone_density_classify(population, config, out_mobility, out_tier, out_cache_key) -> DensityClassifyReport
    pub fn schedule_outer_first(population, out_order, scratch)
    
    // Composes with:
    //   zone_manifold.rs (Plan 001)      — per-cluster density classification
    //   dec/stokes_calculus.rs (Plan 314) — mass-divergence stampede detector
    //   (riir-ai) zone_gating.rs (Plan 305) — cognitive tier (sibling, NOT replaced)

katgpt-rs/benches/
└── bench_002_density_routing_goat.rs   G5a/G5b/G5c benchmarks

katgpt-rs/.benchmarks/
└── 351_density_routing_goat.md         benchmark summary (T3.4)
```

**Layering diagram** (composes, does not replace):

```
                  ┌─────────────────────────────────────────────┐
                  │  Per-NPC per-tick compute decision          │
                  └──────────────────┬──────────────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              │                                             │
   ┌──────────▼──────────┐                    ┌─────────────▼─────────────┐
   │  Plan 305 (existing) │                    │  Plan 351 (this plan)     │
   │  COGNITIVE gating    │                    │  PHYSICAL scheduling      │
   │                      │                    │                           │
   │  ZoneGatingProfile   │   ORTHOGONAL       │  DensityClassifyConfig    │
   │  {tau,beta,budget}   │   ← siblings →     │  {mobility,tier,cache}    │
   │  per zone            │                    │  per zone                 │
   │                      │                    │                           │
   │  Q: "how hard to     │                    │  Q: "how much movement    │
   │  THINK here?"        │                    │  COMPUTE here?"           │
   └──────────┬──────────┘                    └─────────────┬─────────────┘
              │                                             │
              └──────────────────────┬──────────────────────┘
                                     │
                  ┌──────────────────▼──────────────────────────┐
                  │  Plan 001 (just shipped) — cluster-and-     │
                  │  distribute crowd PCA. Per-cluster mood     │
                  │  axes feed BOTH cognitive and physical      │
                  │  gating as the shared substrate.            │
                  └─────────────────────────────────────────────┘
```

## Feature Gate

```toml
# katgpt-rs/crates/katgpt-core/Cargo.toml
[features]
zone_density_routing = ["dep:papaya"]   # opt-in, NOT default until G5a+G5b+G5c pass

[dependencies]
papaya = { version = "0.x", optional = true }   # match sibling-repo version
```

```toml
# katgpt-rs/Cargo.toml (root crate, re-exposes for downstream)
[features]
zone_density_routing = ["katgpt-core/zone_density_routing"]
```

Promotion to default requires G5a AND G5b AND G5c passing (T3.5). G5c is a hard correctness gate (no stale reads during stampede); G5a and G5b are perf/quality gates.

## Validation

- [ ] All Phase-2 unit tests pass (≥18 tests).
- [ ] G5a synthetic benchmark: ≥ +15% Shannon entropy vs mean-aggregation baseline.
- [ ] G5b compute benchmark: ≥ 50% per-tick reduction on dense-dominated workload.
- [ ] G5c stampede correctness: zero stale reads during tier transitions.
- [ ] Determinism: same input + same config → bit-identical output across two calls.
- [ ] Zero-alloc hot path confirmed (no allocations after warmup; papaya is the only allocator).
- [ ] File < 2048 lines (target ~750 LOC + tests).
- [ ] `cargo check --features zone_density_routing` clean, no warnings.
- [ ] `cargo test -p katgpt-core --features zone_density_routing --lib` clean.

## Honest Risk Notes

- **G5a may fail on synthetic data.** The Treuille theory predicts sparse zones have higher event entropy, but a synthetic benchmark with hand-picked event distributions may not capture real game dynamics. **Mitigation**: the real test is Phase 4 T4.4 (60s sim in riir-ai). If T3.1 synthetic fails but T4.4 sim passes, the synthetic benchmark was the wrong proxy, not the primitive.
- **G5b may fail on small cached values.** `papaya` lock-free access has overhead. If the cached value is small (e.g., 8-byte projection), the lock overhead may exceed the recomputation cost. **Mitigation**: gate caching behind a `min_cacheable_size` knob; document the break-even point.
- **G5c may fail under high concurrency.** Papaya is eventually-consistent; a stale read between `insert` on thread A and `invalidate_all` on thread B is possible if the operations aren't properly pinned. **Mitigation**: explicitly `pin()` per papaya docs; if necessary, restrict cache access to the simulation's main thread (game hot path is typically single-threaded per zone anyway).
- **Stampede invalidation may be too aggressive.** `invalidate_all()` clears the entire cache, even for zones unaffected by the stampede. This is wasteful. **Mitigation**: future work could add `invalidate_zones(zone_ids: &[u32])` for finer granularity. Tracked as a potential follow-up issue if G5c passes but G5b regresses during stampede recovery.
- **The Super-GOAT path is conditional and not committed.** Per Research 350 §4, the formal verdict is GOAT. The "emergent crowd behavior" angle (lanes, queues, stampedes) is a *possible* Phase-4 finding, not a guaranteed one. Do NOT pre-claim Super-GOAT in any commit message or doc until T4.4 empirically observes emergence. This is the #1 false-positive class the skill §1.5 warns about.
- **The "bell-shape" framing is partially metaphorical.** The user's "bell-shape over density" intuition refers to the **spatial density profile** of a crowd (dense core, sparse periphery — Gaussian when sliced through the center). The mobility function `m(ρ) = sigmoid(-β·(ρ−ρ₀))` is monotonically decreasing in `ρ`, not bell-shaped. The bell shape appears in the spatial cross-section, not in the per-zone mobility curve. This is documented in Research 350 §1.1 to prevent confusion. The math is correct; the metaphor is just two views of the same physics.
- **Plan 314 G-A precedent.** Plan 314's `belief_mass_divergence` failed its G-A gate (9.5× slower than JS-divergence baseline as an ICT branching detector). That's a cautionary precedent: the mass-divergence hook in T4.3 is **optional and bonus**, not gating. Do not let T4.3 block promotion — it's a "nice-to-have" for stampede detection, not a correctness requirement.
- **`ZoneGatingProfile` (Plan 305) remains authoritative for cognitive gating.** This plan does NOT touch `zone_gating.rs`. The two modules compose in `NpcFunctorRuntime` (Phase 4 T4.2). If a future caller needs only cognitive gating, they use Plan 305 directly. If they need only physical scheduling, they use this plan directly. If they need both, they compose. Clean separation per SOLID.

## TL;DR

Three modelless primitives (`zone_density_classify`, `schedule_outer_first`, `ZoneDensityCache`) ship in `katgpt-rs/crates/katgpt-core/src/zone_density.rs` behind opt-in feature `zone_density_routing`. The classifier turns raw per-zone population into `(mobility, tier, cache_key)` via `fast_sigmoid(-β·(ρ−ρ₀))` + tier match + composite bucket key. The scheduler stable-sorts zones ascending by density. The cache is `papaya`-backed with three invalidation rules (tier transition, density drift, TTL). GOAT gate is three sub-gates: **G5a** (≥ +15% Shannon entropy vs mean-aggregation), **G5b** (≥ 50% compute saved on dense-dominated workloads), **G5c** (zero stale reads during stampede transitions). Promotion requires all three. Composes orthogonally with Plan 305's cognitive gating as a sibling layer (NOT replacement). Super-GOAT-promotion path documented but conditional on Phase 4 T4.4 observing emergent crowd behavior — explicitly NOT pre-claimed per skill §1.5.
