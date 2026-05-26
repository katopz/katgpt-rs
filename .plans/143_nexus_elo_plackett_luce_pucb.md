# Plan 143: Nexus Elo — Plackett-Luce + P-UCB for DDTree Search Enhancement

> **Research:** [104 — AlphaProof Nexus Formal Proof Search](../.research/104_AlphaProof_Nexus_Formal_Proof_Search.md)
> **Paper:** [arXiv:2605.22763](https://arxiv.org/abs/2605.22763) — AI-Driven Formal Proof Search (Google DeepMind, 2026)
> **Feature Gate:** `nexus_elo` (opt-in, NOT default-on, super-GOAT candidate)
> **Status:** ✅ COMPLETE
> **GOAT Pillar:** ❌ Not a pillar — search infrastructure enhancement. See [MMO GOAT Pillars Decision Matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md).
> **Domain:** `katgpt-rs` — generic Plackett-Luce ranking + P-UCB sampling + goal cache. No game-specific code.
> **Blocks:** None. Enhances existing DDTree + SR²AM + Bandit.

---

## Summary

Extract three search primitives from AlphaProof Nexus: (1) **Plackett-Luce Elo ranking** for multi-candidate relative ranking of DDTree partial solutions, (2) **P-UCB evolutionary sampling** with top-K pre-filtering for SR²AM configurator bandit, and (3) **Global goal cache** with deep hash memoization for cross-thread DDTree deduplication. All modelless, no training, no model changes.

---

## Why

1. **Plackett-Luce > pairwise BT:** Our Bradley-Terry ranking (Plan 040) handles pairwise comparisons. Plackett-Luce handles N-way rankings — more information per LLM call when rating DDTree nodes (P=7 per match vs P=2 pairwise).
2. **P-UCB improves SR²AM:** Our UCB1 configurator (Plan 112) doesn't pre-filter the population or normalize scores. Top-64 filter + Elo normalization prevents search collapse and improves convergence on hard problems.
3. **Goal cache reduces redundant computation:** Multiple DDTree threads exploring the same game state waste MCTS/Bandit budget. Deep hash memoization catches this.
4. **Paper validates our architecture:** Their basic agent (modelless loop + compiler feedback) solved all 9 Erdős problems. This confirms our G-Zero modelless-first thesis (Plan 049).
5. **Super-GOAT potential for Percepta:** If we apply proof sketch evolution (EVOLVE-BLOCK pattern) to our transformer-VM compiler stack, that's a novel capability. Feature-gate it.

---

## Architecture

### Phase 1: Plackett-Luce Ranking Module (T1–T4)

- [x] **T1: `PlackettLuceRating` struct**

```rust
/// Plackett-Luce model for ranking N candidates from relative orderings.
/// Each candidate has latent strength λ ~ Gamma(1, r) with hierarchical prior.
pub struct PlackettLuceRating {
    strengths: HashMap<u64, f64>,  // item_id → λ_mean
    visit_counts: HashMap<u64, usize>,
    gibbs_samples: usize,  // default 1000
    burn_in: usize,        // default 200
}
```

Key methods:
- `rate_match(ranking: &[u64])`: Update strengths given an ordering
- `elo_score(item: u64) -> f64`: `1200 + 400 * log10(λ_mean)`
- `thompson_sample(candidates: &[u64]) -> u64`: Sample from posterior for exploration
- `top_k(k: usize) -> Vec<u64>`: Filter to top-K by Elo

- [x] **T2: Gibbs sampler for posterior inference**

Hierarchical prior: `λ_s | r_s ~ Gamma(1, r_s)`, `r_s ~ Gamma(1, 1)`

```rust
fn gibbs_update(
    strengths: &mut HashMap<u64, f64>,
    ranking: &[u64],  // ordered, best first
    n_samples: usize,
    burn_in: usize,
) {
    // For each ranking observation, update posterior via conjugate Gamma
    // Retain every 25th sample to mitigate in-chain correlation
    // Return posterior mean as final strength
}
```

- [x] **T3: Integration with DDTree**

Add optional Elo scores to DDTree frontier nodes:

```rust
pub struct DDTreeNode {
    // ... existing fields ...
    #[cfg(feature = "nexus_elo")]
    elo_score: Option<f64>,
    #[cfg(feature = "nexus_elo")]
    visit_count: usize,
}
```

- [x] **T4: Multi-candidate rating for DDTree**

When DDTree explores multiple partial solutions, rate them jointly:

```rust
#[cfg(feature = "nexus_elo")]
fn rate_ddtree_frontier(
    nodes: &[DDTreeNode],
    rater: &dyn ScreeningPruner,
) -> Vec<usize> {
    // Sample P=7 nodes, rank by ScreeningPruner.relevance()
    // Feed ranking into PlackettLuceRating
    // Return indices sorted by Elo
}
```

---

### Phase 2: P-UCB Sampling for SR²AM (T5–T7)

- [x] **T5: P-UCB selector**

```rust
pub struct PUCBSelector {
    exploration_constant: f64,  // c = 0.2
    top_k_filter: usize,        // K = 64
}

impl PUCBSelector {
    pub fn select(&self, population: &[(u64, f64, usize)]) -> u64 {
        // 1. Filter to top-K by Elo
        // 2. Normalize Elo to [0, 1] → base score q
        // 3. P-UCB: q + c * sqrt(ln(total_visits) / (visits + 1))
        // 4. Return highest P-UCB score
    }
}
```

- [x] **T6: Integration with SR²AM ConfiguratorBandit**

```rust
#[cfg(feature = "nexus_elo")]
impl ConfiguratorBandit {
    fn select_configuration_pucb(&self) -> BanditArm {
        // Replace UCB1 arm selection with P-UCB
        // Requires maintaining Elo scores per arm
    }
}
```

- [x] **T7: Adaptive exploration constant**

```rust
// Start with c=0.2, anneal based on solve rate:
// - If solve rate < 10%: increase c (more exploration)
// - If solve rate > 50%: decrease c (more exploitation)
fn adaptive_c(solve_rate: f64, base_c: f64) -> f64 {
    base_c * (1.0 + (0.5 - solve_rate).max(-0.3).min(0.3))
}
```

---

### Phase 3: Global Goal Cache for DDTree (T8–T10)

- [x] **T8: `GoalCache` struct with deep hashing**

```rust
pub struct GoalCache {
    cache: HashMap<u64, CacheEntry>,
    hasher: Blake3Hasher,
}

struct CacheEntry {
    result: GoalResult,
    visit_count: usize,
    last_access: Instant,
}

pub enum GoalResult {
    Proven(Vec<u8>),       // serialized proof/tactic
    Disproven,             // goal is false
    Unresolved { feedback: String },
}
```

- [x] **T9: Deep hash computation**

```rust
fn goal_hash(state: &[u8], target: &[u8]) -> u64 {
    // blake3(state || target) → u64
    // Same hash = same subproblem, skip re-computation
}
```

- [x] **T10: Integration with DDTree exploration**

```rust
#[cfg(feature = "nexus_elo")]
impl DDTree {
    fn explore_with_cache(&mut self, cache: &mut GoalCache) -> DDTreeResult {
        let hash = goal_hash(&self.state, &self.target);
        if let Some(entry) = cache.get(hash) {
            return entry.result.clone();
        }
        let result = self.explore_uncached();
        cache.insert(hash, result.clone());
        result
    }
}
```

---

### Phase 4: Percepta Proof Sketch Evolution (T11–T13) — Super-GOAT

> ⚠️ This phase is the selling point. Feature-gate it separately.
> If Percepta (Plan 064) is not active, this phase is a no-op.

- [x] **T11: `EvolveBlock` markers for Percepta**

```rust
#[cfg(all(feature = "nexus_elo", feature = "percepta"))]
pub struct EvolveBlock {
    pub start_marker: usize,  // line number
    pub end_marker: usize,
    pub allowed_mutations: MutationKind,
}

pub enum MutationKind {
    Definitions,    // can modify helper definitions
    Lemmas,         // can modify proof steps
    Values,         // can change parameter values
}
```

- [x] **T12: Population database for proof sketches**

```rust
#[cfg(all(feature = "nexus_elo", feature = "percepta"))]
pub struct ProofSketchPopulation {
    sketches: Vec<ProofSketch>,
    rater: PlackettLuceRating,
    selector: PUCBSelector,
    goal_cache: GoalCache,
}

impl ProofSketchPopulation {
    /// EVOLVE loop: sample parent → mutate → validate → register
    pub fn evolve_step(&mut self) -> Option<ProofSketch> {
        // 1. P-UCB sample parent sketch
        // 2. Apply constrained mutation within EVOLVE-BLOCK markers
        // 3. Validate via ConstraintPruner
        // 4. Check goal cache for known subgoals
        // 5. Register in population with Elo update
        // 6. Return sorry-free proof if found
    }
}
```

- [x] **T13: Constrained mutation engine**

```rust
#[cfg(all(feature = "nexus_elo", feature = "percepta"))]
pub fn constrained_mutate(
    sketch: &ProofSketch,
    blocks: &[EvolveBlock],
    rng: &mut impl Rng,
) -> ProofSketch {
    // Only modify within EVOLVE-BLOCK/EVOLVE-VALUE markers
    // Strategies: decompose goals, combine ideas, try new approach
    // Preserve theorem statement integrity
}
```

---

## Feature Gates

```toml
[features]
default = []
nexus_elo = []  # Plackett-Luce + P-UCB + goal cache (Phases 1-3)
# Super-GOAT: requires both nexus_elo and percepta
# Proof sketch evolution (Phase 4) activates only when both are enabled
```

**Why feature-gated:**
- Plackett-Luce ranking is novel but unproven in our game domain
- P-UCB may not improve over UCB1 for our problem sizes
- Goal cache adds memory overhead
- Phase 4 (Percepta evolution) is the selling point — keep it opt-in until GOAT-proven

---

## GOAT Proof Targets

| Target | Metric | Threshold |
|--------|--------|-----------|
| T4: DDTree Elo ranking | Convergence speed on Bomber 9×9 | ≥ 1.2× faster than UCB1 baseline |
| T7: P-UCB SR²AM | Config selection accuracy | ≥ 5% improvement over UCB1 |
| T10: Goal cache | Redundant computation reduction | ≥ 20% fewer duplicate MCTS nodes |
| T13: Percepta evolution | Proof sketch quality | Elo convergence within 100 generations |

---

## What This Is NOT

- ❌ Not a new game feature
- ❌ Not a GOAT pillar (per [decision matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md))
- ❌ Not model-based — entirely modelless search infrastructure
- ❌ Not Lean/Math-specific — generic ranking + caching primitives

---

## What This Enables

- ✅ Better DDTree search for all games (Bomber, Go, Monopoly, FFT)
- ✅ More efficient SR²AM configuration selection
- ✅ Reduced redundant computation in parallel DDTree exploration
- 🔒 Super-GOAT: Percepta proof sketch evolution (selling point, feature-gated)

---

## Module Structure

```
katgpt-rs-core/src/
├── search/
│   ├── dd_tree.rs          # existing, +#[cfg(feature = "nexus_elo")] Elo fields
│   ├── bandit.rs           # existing, +PUCBSelector
│   └── plackett_luce.rs    # NEW: PlackettLuceRating + Gibbs sampler
├── cache/
│   └── goal_cache.rs       # NEW: Global goal cache with deep hash
└── percepta/               # existing, +evolution modules
    └── sketch_evolution.rs # NEW: #[cfg(all(feature = "nexus_elo", feature = "percepta"))]
```

---

## References

- Research: [104 — AlphaProof Nexus](../.research/104_AlphaProof_Nexus_Formal_Proof_Search.md)
- Related: Plan 040 (Bradley-Terry), Plan 049 (G-Zero modelless), Plan 061 (Fourier MCTS transposition), Plan 064 (Percepta), Plan 112 (SR²AM), Research 088 (AlphaProof Nexus — existing)
- [MMO GOAT Pillars Decision Matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md)
