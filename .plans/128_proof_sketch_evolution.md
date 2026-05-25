# Plan 128: Proof Sketch Evolution — Elo-Rated Population + Global Goal Cache

**Branch:** `develop/feature/128_proof_sketch_evolution`
**Depends on:** Plan 030 (Multi-armed bandit), Plan 080 (MaxSim late-interaction), Plan 040 (Bradley-Terry ranking)
**Research:** 088 (AlphaProof Nexus — AI-Driven Formal Proof Search)
**Paper:** [Advancing Mathematics Research with AI-Driven Formal Proof Search](https://arxiv.org/abs/2605.22763) (Tsoukalas et al., Google DeepMind, May 2026)
**Status:** ✅ Complete (T1–T9, 46 GOAT unit tests pass) · Benchmark convergence criteria deferred (needs real arena runs)

---

## Tasks

- [x] T1: `ProofGoalCache` — blake3-keyed global goal deduplication (`proof/goal_cache.rs`)
- [x] T2: `SketchEntry` + `SketchId` types — proof state + pending goals + lessons (`proof/sketch_types.rs`)
- [x] T3: `SketchPopulation` — top-64 Elo-rated sketch database (`proof/sketch_population.rs`)
- [x] T4: Plackett-Luce rating — pairwise comparison → Elo via Gibbs sampling (`proof/plackett_luce.rs`)
- [x] T5: P-UCB sketch sampling + diversity injection — bridge from `BanditPruner` to sketch selection (`proof/sketch_sampler.rs`)
- [x] T6: DDTree integration — goal cache shared across draft branches (`proof/dtree_goal_cache.rs`) ✅
- [x] T7: Feature gate `proof_sketch_evolution` + parallelism guard + module glue ✅
- [x] T8: GOAT proof — 46/46 tests (cache dedup, population CRUD, Plackett-Luce rating, P-UCB sampling, diversity injection, parallelism guard, integration) ✅
- [x] T9: Benchmark — `.benchmarks/039_proof_sketch_evolution_goat.md` ✅

---

## Context

AlphaProof Nexus demonstrates that an **evolutionary agent with Elo-rated sketches outperforms independent parallel agents by 2-5× on the hardest formal verification problems**. The key insight: binary fitness (pass/fail verification) can be bridged to graduated fitness via LLM-based pairwise ranking aggregated through Plackett-Luce → Elo.

This maps directly to our model-based/modelless architecture:

| AlphaProof Nexus | Our System | Type |
|------------------|------------|------|
| Agent (A): N independent subagents | `NoScreeningPruner` — brute force parallel | Modelless |
| Agent (D): Evo + Elo + goal cache | `BanditPruner` + `ProofGoalCache` | Model-based |
| Gemini Flash raters (cheap) | `ScreeningPruner::relevance()` | Modelless signal |
| Gemini Pro provers (expensive) | LLM generation / MCTS expansion | Model-based signal |
| Global goal cache (blake3 hash) | `ProofGoalCache` (blake3 hash) | Shared infra |

**Core finding from paper:** "The basic agent solved all 9 problems, though at a higher cost on the harder problems." → Our `NoScreeningPruner` matches quality but `BanditPruner` wins on efficiency for hard domains.

---

## Architecture

### T1: `ProofGoalCache` — Global Goal Deduplication

The paper's global goal cache uses deep hashes of exact formal state. We use `blake3` per convention:

```text
ProofGoalCache
├── cache: HashMap<GoalHash, GoalResult>   // blake3 keyed
├── hits: AtomicU64                        // GOAT metric
├── misses: AtomicU64                      // GOAT metric
└── get_or_verify(goal, verifier) → GoalResult
    1. hash = GoalHash(blake3::hash(goal.canonical_bytes()))
    2. cache.entry(hash).or_insert_with(|| verifier.verify(goal))
    3. Increment hits/misses atomically
```

`GoalHash` wraps `blake3::Hash`. `GoalResult` is `Proved` | `Disproved(Counterexample)` | `Unknown`.

### T2: `SketchEntry` Types

```text
SketchEntry {
    id: SketchId(Uuid::now_v7()),
    proof_state: ProofState,       // serialized game/proof state
    pending_goals: Vec<Goal>,       // unresolved subgoals
    lessons: Vec<String>,           // episode summaries (Ralph loop)
    elo_rating: f64,                // Plackett-Luce aggregated
    visits: usize,                  // P-UCB visit count
    created_at: Instant,
}
```

### T3: `SketchPopulation` — Top-64 Elo Database

```text
SketchPopulation {
    sketches: HashMap<SketchId, SketchEntry>,
    top_k: usize,                   // 64 per paper
    max_population: usize,          // configurable
}
```

Eviction policy: keep top-64 by Elo, LRU for ties. New entries enter at Elo 1200 (paper default).

### T4: Plackett-Luce Rating

The paper uses P=7 sketches per rating match, aggregated via Gibbs sampling:

```text
PlackettLuceRater {
    match_size: usize,              // P=7 per paper
    gibbs_samples: usize,           // I=1000
    burn_in: usize,                 // B=200
    elo_offset: f64,                // 1200
    elo_scale: f64,                 // 400
}

rate(sketches: &[SketchEntry], rankings: &[Vec<usize>]) -> HashMap<SketchId, f64>
    1. Initialize λ_s ~ Gamma(1, r_s), r_s ~ Gamma(1, 1)  // hierarchical prior
    2. For each Gibbs sample:
       a. Sample λ_s | rankings, λ_{-s} ~ conjugate posterior
       b. Discard first B=200 burn-in
    3. Elo_s = 1200 + 400 * log10(mean(λ_s))
```

**Bridge to existing:** Our `BradleyTerry` (Research 040, Plan 080) handles pairwise. Plackett-Luce is the multi-item generalization. Implement as extension, not replacement.

### T5: P-UCB Sketch Sampling + Diversity Injection

Already partially in `BanditPruner`. The sketch-specific variant:

```text
score = q + c * sqrt(total_visits / (visits + 1))

where:
  q = normalize_to_01(sketch.elo_rating, top_64_min, top_64_max)
  c = 0.2  // paper's empirical value
```

**Diversity injection** (Supplementary Insight 7 from Research 088): The paper's controller stochastically injects structured exploration hints. We apply this by randomly selecting from `DiversityStrategy` during explore:

```text
enum DiversityStrategy {
    Decompose,    // "Split complex goals into sub-goals" (Go: split territory fight)
    Combine,      // "Merge ideas from prior attempts" (Bomber: team tactic merge)
    NovelApproach, // "Try completely new strategy" (switch opening/heuristic)
}

// During explore arm (ε-greedy or UCB exploration):
fn inject_diversity(sketch: &SketchEntry, rng: &mut impl Rng) -> DiversityHint {
    match rng.gen_range(0..3) {
        0 => DiversityHint::Decompose,   // 33% chance
        1 => DiversityHint::Combine,     // 33% chance
        _ => DiversityHint::NovelApproach, // 34% chance
    }
}
```

This prevents population collapse into a single lineage — a failure mode the paper observed when diversity injection was disabled.

### T6: DDTree Integration

The goal cache is shared across DDTree draft branches within a single decode step:

```text
DDTree with ProofGoalCache:
  1. Create empty ProofGoalCache per decode step
  2. For each draft branch (up to tree_budget):
     a. For each constraint check in branch:
        - Check cache first (blake3 hash of constraint + context)
        - Cache miss → verify, store result
     b. Branch result cached for future branches
  3. Report cache hit rate as GOAT metric
```

### T7: Feature Gate + Parallelism Guard

```toml
[features]
proof_sketch_evolution = ["bandit_pruner"]
```

Default: off (opt-in). Requires `bandit_pruner` feature.

**Parallelism guard** (Supplementary Insight 6 from Research 088): The paper found that population search with only 1 generator **underperforms** the basic setup. The database only helps when multiple agents contribute asynchronously. Runtime guard:

```text
fn should_use_population() -> bool {
    rayon::current_num_threads() > 1  // need ≥2 threads for population to help
}

// In sketch selection:
if should_use_population() {
    population.sample_p_ucb()  // evolutionary path
} else {
    bandit.select()            // fallback to basic UCB (single-threaded)
}
```

Single-threaded decode (e.g., `NoScreeningPruner`) should skip population lookup entirely and use basic Q-value bandit. The population adds overhead without benefit in serial execution.

Module structure:
```text
katgpt-core/src/
├── proof/
│   ├── mod.rs                  # pub mod goal_cache, sketch_types, ...
│   ├── goal_cache.rs           # T1: ProofGoalCache
│   ├── sketch_types.rs         # T2: SketchEntry, SketchId, GoalResult
│   ├── sketch_population.rs    # T3: SketchPopulation (top-64 Elo DB)
│   ├── plackett_luce.rs        # T4: Plackett-Luce → Elo rating
│   ├── sketch_sampler.rs       # T5: P-UCB sampling for sketches
│   └── dtree_goal_cache.rs     # T6: DDTree integration
```

### T8: GOAT Proof — Convergence Speedup

**Hypothesis:** Evolutionary sketch search converges 2× faster than independent search on formal verification tasks.

**Test protocol:**
1. Run Bomber arena (1000 rounds) with:
   - (A) Independent: N parallel branches, no shared state → our baseline
   - (D) Evolutionary: `SketchPopulation` + `ProofGoalCache` + P-UCB sampling
2. Measure:
   - Rounds to reach 90% win rate (convergence speed)
   - Total constraint verification calls (cache efficiency)
   - Wall-clock time per round
3. Target: (D) reaches 90% win rate in ≤50% of rounds vs (A)
4. Target: cache hit rate ≥60% on structured domains (Bomber, Go)

**GOAT proof checklist:**
- [x] 46/46 unit + integration tests pass (cache dedup, population CRUD, Plackett-Luce, P-UCB, diversity, parallelism guard) ✅
- [ ] **Benchmark**: Evolutionary converges ≥2× faster (rounds to 90% win rate) — needs real arena runs
- [ ] **Benchmark**: Goal cache hit rate ≥60% (reduces verification calls 3×) — needs real arena runs
- [ ] **Benchmark**: No regression on win rate ceiling (both reach same final quality) — needs real arena runs
- [ ] **Benchmark**: Wall-clock overhead <10% (cache lookup is cheap) — needs real arena runs

### T9: Benchmark — Constraint Verification Reduction

**Domains:** Bomber, Go (9×9), Monopoly FSM

**Metrics:**
- Verification calls per game (with/without cache)
- Cache hit rate by game phase (opening/midgame/endgame)
- Memory overhead of `SketchPopulation` (target: <1MB for 64 entries)
- Elo rating convergence (does it stabilize within 1000 rounds?)

**Expected results (from paper's pattern):**
- Easy domains (Monopoly): Cache hit rate ~40%, marginal speedup
- Medium domains (Bomber): Cache hit rate ~60%, 2× speedup
- Hard domains (Go 9×9): Cache hit rate ~70%, 3× speedup

---

## Dependency Graph

```
T1 (GoalCache) ─────────────────────┐
T2 (SketchTypes) ───────────────────┼── T3 (Population) ── T4 (PL) ── T5 (Sampler)
                                    │                                    │
                                    └── T6 (DDTree integration) ─────────┘
                                                                         │
T7 (Feature gate) ──────────────────────────────────────────────────────┘
                                                                         │
T8 (GOAT proof) ────────────────────────────────────────────────────────┘
T9 (Benchmark) ─────────────────────────────────────────────────────────┘
```

T1 + T2 can be parallelized. T3-T5 are sequential. T6 depends on T1 + T5. T7 can land anytime. T8/T9 depend on all.

---

## Key Design Decisions

1. **blake3 for goal hashing** — Per project convention (Research 063, OCTOPUS). Faster than SHA256, adequate for cache keys.
2. **Top-64 population cap** — Paper's empirical value. Prevents unbounded memory growth. Configurable.
3. **Plackett-Luce over Bradley-Terry** — Multi-item ranking (P=7) is more information-efficient than pairwise BT. Our BT implementation (Plan 080) handles pairwise; PL extends it for >2 items.
4. **Gamma(1, Gamma(1,1)) hierarchical prior** — Paper's choice. Heavy tails prevent premature convergence. We replicate rather than innovate on the statistical model.
5. **Feature gate opt-in** — Not all users need sketch evolution. `bandit_pruner` users get the basic path; `proof_sketch_evolution` adds the population + rating layer.
6. **Per-decode-step cache scope** — Goal cache is created fresh per decode step, not persisted across steps. This avoids stale entries and keeps memory bounded. Transposition tables (GoState) handle cross-step caching separately.
7. **Parallelism guard required** — Paper's ablation shows population search underperforms basic with single generator. Runtime check `rayon::current_num_threads() > 1` gates population usage. Single-threaded fallback to basic UCB.
8. **Diversity injection via enum** — `DiversityStrategy { Decompose, Combine, NovelApproach }` prevents population collapse. Cheap (no extra compute), applied during explore arm only.

---

## What We're NOT Doing

- **Lean compiler integration** — Out of scope for Rust inference runtime
- **AlphaProof RL training** — Requires massive TPU budget, not applicable
- **LLM-as-rater** — Our `ScreeningPruner::relevance()` provides the cheap signal; we don't call external LLMs during inference
- **Full Ralph loop** — Our `GZeroLoop` already covers the episode structure
- **EVOLVE-BLOCK/EVOLVE-VALUE markers** — Lean-specific, not applicable to game/proof domains

---

## Honest Assessment

**Confidence:** Medium-High

The concepts are well-validated in the paper (9 Erdős problems solved, clear cost-efficiency data). The mapping to our architecture is clean. The main risk is that **game domains may not benefit as much as formal proof domains** from sketch evolution — game states are more structured than proof sketches, so the "graduated fitness from Elo rating" may provide less marginal signal over our existing `BanditPruner` Q-values.

**Mitigation:** GOAT proof (T8) measures actual speedup. If <1.5×, we document as negative result and keep the `ProofGoalCache` (which is almost certainly useful regardless).

**Expected outcome:** `ProofGoalCache` is the real win (3× verification reduction). `SketchPopulation` + Plackett-Luce is nice-to-have for formal verification tasks but may not justify complexity for game arenas alone.