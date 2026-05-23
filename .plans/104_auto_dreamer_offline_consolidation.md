# Plan 104: Auto-Dreamer Offline Memory Consolidation

**Research**: `.research/069_AutoDreamer_Offline_Memory_Consolidation.md`
**Paper**: arXiv:2605.20616 — Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents
**Status**: 🟡 Planning

## Tasks

- [ ] T1: Create `dreamer` feature gate + module scaffold (`src/pruners/dreamer/`)
- [ ] T2: Implement `DreamerConfig` + `WorkingRegion` + `ReplacementSet` types
- [ ] T3: Implement consolidation scheduler (cadence-based region selection)
- [ ] T4: Implement region rewriting consolidator (deterministic modelless merge)
- [ ] T5: Implement memory decay/forgetting policy
- [ ] T6: Implement counterfactual dropout utility estimation
- [ ] T7: Integrate Dreamer into existing bandit/δ-mem pipeline
- [ ] T8: Add Freeze/Thaw support for consolidated banks
- [ ] T9: Bomber GOAT proof — dreamer vs no-dreamer (1000 rounds)
- [ ] T10: Go GOAT proof — dreamer vs no-dreamer (20 games)
- [ ] T11: Update README + .docs + .research

---

## Context

Auto-Dreamer decouples fast per-session memory acquisition from slow cross-session consolidation. Key insight: **region rewriting** (treat a working region as read-only evidence, synthesize a compact replacement) forces compactness structurally rather than as an afterthought.

Our existing `AbsorbCompress` already merges similar bandit arms, but lacks:
1. **Scheduled consolidation** — no periodic "dreaming" trigger
2. **Memory decay** — entries grow without bound (no forgetting)
3. **Counterfactual utility** — no signal for load-bearing vs redundant entries
4. **Working ↔ long-term hierarchy** — single-tier only

## Architecture

### Module Structure

```text
src/pruners/dreamer/
├── mod.rs              # Index only, re-exports
├── types.rs            # DreamerConfig, WorkingRegion, ReplacementSet, DecayPolicy
├── scheduler.rs        # Consolidation cadence, region selection
├── consolidator.rs     # Region rewriting (deterministic merge)
├── counterfactual.rs   # Dropout-based utility estimation
└── decay.rs            # Exponential decay + omission-based forgetting
```

### Feature Gate

```toml
[features]
dreamer = ["bandit"]  # Offline memory consolidation scheduler
```

### Data Flow

```text
Episode k completes
  → Writer appends to TrialLog (existing)
  → Scheduler checks: k % cadence == 0?
    → YES: Select WorkingRegion R (recent writes + recent retrievals)
    → Consolidator: treat R as read-only, synthesize ReplacementSet S
      → Merge similar arms (weighted avg Q-values)
      → Apply decay to stale entries
      → Counterfactual: random dropout to estimate utility
    → Bank update: B* = (B \ R) ∪ S
    → Freeze if configured
    → NO: continue
```

---

## T1: Feature Gate + Module Scaffold

**File**: `Cargo.toml`, `src/pruners/dreamer/mod.rs`

- Add `dreamer = ["bandit"]` to `[features]`
- Create module with `pub mod types; pub mod scheduler;` etc.
- Add `#[cfg(feature = "dreamer")]` gate in `src/pruners/mod.rs`
- Zero implementation, just types and structure

## T2: Types (`types.rs`)

```rust
/// Configuration for the Dreamer consolidation scheduler.
#[derive(Debug, Clone)]
pub struct DreamerConfig {
    /// Episodes between consolidation events (paper: k=5-10)
    pub cadence: usize,
    /// Fraction of bank to include in working region (paper: recent writes + retrievals)
    pub region_fraction: f32,
    /// Similarity threshold for merging arms during consolidation
    pub merge_threshold: f32,
    /// Decay factor per consolidation event (0.0 = full decay, 1.0 = no decay)
    pub decay_factor: f32,
    /// Counterfactual dropout fraction (paper: ρ=0.25-0.5)
    pub dropout_fraction: f32,
    /// Number of Monte Carlo samples for counterfactual estimation (paper: M≥1)
    pub mc_samples: usize,
    /// Minimum visits before an arm is eligible for consolidation
    pub min_visits: usize,
}

/// A working region selected from the memory bank for consolidation.
pub struct WorkingRegion {
    /// Indices of arms in the working region
    pub arm_indices: Vec<usize>,
    /// Q-values at time of selection (read-only snapshot)
    pub q_snapshot: Vec<f32>,
    /// Visit counts at time of selection
    pub visit_snapshot: Vec<usize>,
    /// Timestamp of selection
    pub selected_at_episode: usize,
}

/// A compact replacement set synthesized from a working region.
pub struct ReplacementSet {
    /// Merged arms (index → new Q-value)
    pub merged: Vec<(usize, f32)>,
    /// Arms to forget (omitted from replacement)
    pub forgotten: Vec<usize>,
    /// Counterfactual utility scores per merged arm
    pub utility: Vec<f32>,
}

/// Policy for memory decay during consolidation.
#[derive(Debug, Clone, Copy)]
pub enum DecayPolicy {
    /// No decay (baseline)
    None,
    /// Exponential: q *= decay_factor each consolidation
    Exponential { factor: f32 },
    /// Access-based: decay proportional to episodes since last access
    AccessBased { half_life: usize },
}
```

## T3: Consolidation Scheduler (`scheduler.rs`)

Select working region R from bandit arms:
- R = {recently written arms} ∪ {recently retrieved arms}
- "Recently" = within last `cadence` episodes
- Sort by recency, take top `region_fraction` of bank

```rust
impl DreamerScheduler {
    /// Check if consolidation should trigger at this episode.
    pub fn should_consolidate(&self, episode: usize) -> bool {
        episode > 0 && episode % self.config.cadence == 0
    }

    /// Select working region from bandit state.
    pub fn select_region(&self, bandit: &BanditPruner, episode: usize) -> WorkingRegion {
        // 1. Collect arms written in last `cadence` episodes
        // 2. Collect arms retrieved in last `cadence` episodes
        // 3. Union → working region R
        // 4. Snapshot Q-values and visits (read-only evidence)
    }
}
```

Requires `last_write_episode: usize` and `last_retrieve_episode: usize` fields on bandit arms. These need to be added to `BanditStats` or similar.

## T4: Region Rewriting Consolidator (`consolidator.rs`)

Deterministic modelless merge (NOT LLM-based like the paper):

```rust
impl DreamerConsolidator {
    /// Consolidate working region into replacement set.
    /// This is the core "region rewriting" operation.
    pub fn consolidate(&self, region: &WorkingRegion) -> ReplacementSet {
        // 1. Cluster arms by similarity (feature hash or Q-value proximity)
        // 2. For each cluster:
        //    a. Compute weighted average Q-value (weight = visits)
        //    b. Create single merged arm
        // 3. Arms below min_visits or below Q-threshold → forgotten
        // 4. Return ReplacementSet
    }
}
```

Key difference from paper: Our consolidation is **O(n log n) deterministic** (clustering + merge), not iterative LLM tool-use. This preserves the modelless, CPU-only constraint.

## T5: Memory Decay (`decay.rs`)

```rust
impl MemoryDecay {
    /// Apply decay to arms NOT in the working region.
    /// Arms in working region are handled by consolidator.
    pub fn apply(&self, bandit: &mut BanditPruner, region: &WorkingRegion) {
        match self.policy {
            DecayPolicy::None => {},
            DecayPolicy::Exponential { factor } => {
                // For each arm NOT in region: q *= factor
            }
            DecayPolicy::AccessBased { half_life } => {
                // For each arm NOT in region:
                //   let age = current_episode - last_access_episode;
                //   q *= 0.5f32.powi(age as i32 / half_life as i32)
            }
        }
    }
}
```

This addresses the paper's insight: **omission-based forgetting** — entries not rewritten are forgotten.

## T6: Counterfactual Dropout (`counterfactual.rs`)

```rust
impl CounterfactualEstimator {
    /// Estimate utility of each arm in replacement set.
    /// rcf(S) = U(S) - E[U(S\{e})] for random e
    pub fn estimate_utility(
        &self,
        replacement: &ReplacementSet,
        evaluator: &dyn Fn(&[usize]) -> f32,
    ) -> Vec<f32> {
        // 1. Evaluate full replacement set: U(S)
        // 2. For each arm, randomly drop `dropout_fraction` others
        // 3. Evaluate masked set: U(S\e)
        // 4. Utility of arm = U(S) - avg(U(S\e)) where e is dropped
        // 5. High utility → load-bearing, low → redundant, negative → harmful
    }
}
```

The evaluator function uses existing `TrialLog` metrics or game win-rate as utility signal.

## T7: Integration

Wire into existing pipeline:

1. **`BanditPruner`** — add `last_write_episode`, `last_retrieve_episode` tracking
2. **`AbsorbCompress`** — call Dreamer scheduler after absorb cycle
3. **`DeltaGatedAbsorbCompress`** — gate consolidation by hint-δ quality
4. **`MultiDomainMemory`** — per-domain consolidation with shared cadence
5. **`g_zero/`** — use counterfactual utility as additional reward signal

## T8: Freeze/Thaw for Consolidated Banks

Extend `freeze.rs`:
- `DreamerFrozenBank` struct with consolidated arms + decay state
- `save_frozen_dreamer()` / `load_frozen_dreamer()`
- Integrate with existing Freeze/Thaw pipeline (Plan 092)

## T9: Bomber GOAT Proof

```bash
cargo test -p microgpt-rs --test bomber_dreamer_goat --features dreamer
```

Design:
- 1000 rounds, seed=42
- A: Baseline (bandit only, no consolidation)
- B: Dreamer (cadence=10, region_fraction=0.3, exponential decay=0.9)
- Metric: win rate, arm count at end, memory tokens (estimated)

Expected:
- B wins ≥ A (compact memory improves retrieval quality)
- B arm count ≤ 50% of A arm count

## T10: Go GOAT Proof

```bash
cargo test -p microgpt-rs --test go_dreamer_goat --features dreamer
```

Design:
- 20 games, 9×9, vs Random
- A: Baseline (bandit only)
- B: Dreamer (cadence=5, region_fraction=0.4, access-based decay half_life=50)

Expected:
- B win rate ≥ A win rate
- B arm count significantly smaller

## T11: Documentation Updates

- Update `README.md` Dreamer section
- Update `.research/069_AutoDreamer_Offline_Memory_Consolidation.md` with results
- Update feature flags table in README
- Add `.docs/` for Dreamer API

---

## Estimated Effort

| Task | Effort | Dependencies |
|------|--------|-------------|
| T1: Scaffold | 0.5h | None |
| T2: Types | 1h | T1 |
| T3: Scheduler | 2h | T2 |
| T4: Consolidator | 3h | T2 |
| T5: Decay | 1h | T2 |
| T6: Counterfactual | 2h | T2 |
| T7: Integration | 3h | T3-T6 |
| T8: Freeze/Thaw | 1h | T7 |
| T9: Bomber GOAT | 2h | T7 |
| T10: Go GOAT | 1h | T7 |
| T11: Docs | 1h | T9-T10 |
| **Total** | **~17.5h** | |

---

## Key Design Decisions

1. **Deterministic modelless merge** over LLM-based synthesis — fits our CPU-only constraint
2. **Single `dreamer` feature gate** — composes with `bandit`, not a separate dependency chain
3. **Decay as omission** — don't rewrite = forget (same as paper's region rewriting semantics)
4. **Counterfactual via existing metrics** — reuse TrialLog/game win-rate, not a separate eval loop
5. **Per-domain cadence** — different games may need different consolidation frequencies

## Risk Mitigation

| Risk | Mitigation | Rollback |
|------|-----------|----------|
| Over-abstraction loses useful details | Preserve top-k concrete arms alongside merged rules | Disable decay, keep all arms |
| Consolidation too aggressive | Tune `region_fraction` and `merge_threshold` per domain | Set `cadence=usize::MAX` (effectively disabled) |
| Counterfactual too expensive | Default `mc_samples=1`, increase only for research | Set `dropout_fraction=0.0` |
| Freeze/Thaw incompatibility | `DreamerFrozenBank` is additive, doesn't change existing format | Don't freeze dreamer state |

## References

- Auto-Dreamer paper: arXiv:2605.20616
- Existing `AbsorbCompress`: `src/pruners/absorb_compress.rs`
- Existing `DeltaMemoryState`: `src/pruners/delta_mem/state.rs`
- Existing Freeze/Thaw: `src/pruners/freeze.rs`
- G-Zero self-play: `src/pruners/g_zero/`
- MeMo Reflections: `src/pruners/reflection.rs`
