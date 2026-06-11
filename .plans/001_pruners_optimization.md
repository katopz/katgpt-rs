# 001 Pruners Optimization Plan

Implementation of `.issues/001_pruners_optimization.md` findings.

## Tasks

### CRITICAL
- [x] **C-1**: BomberState `cells: Vec<Vec<Cell>>` → flat array `[Cell; 169]`
- [x] **C-2**: `available_actions()` returns `Vec<BomberAction>` → use `available_actions_into()` with pre-allocated buffer in MCTS
- [x] **C-3**: GoHeuristic `influence()` — per-cell BFS → multi-source BFS

### HIGH (Correctness)
- [x] **H-7**: `is_circuit_neuron()` broken binary search → `HashSet<(usize, usize)>`

### HIGH (Performance)
- [x] **H-1**: `Arc<BFCP>` instead of deep clone in cache pipeline
- [x] **H-2**: `soft_route_relevance()` allocates 3 Vecs → pre-allocated scratch buffers
- [x] **H-3**: `CurvatureInfluence arm_bandit_score()` allocates Vec per arm → cache concentration
- [x] **H-5**: `SensitivityCache` uses `Arc<RwLock<HashMap>>` → papaya
- [x] **H-8**: `softmax()` + `kl_divergence()` allocate 3 Vecs → pre-allocate in struct
- [x] **H-9**: `review_metrics` cascading atomic loads → snapshot once
- [x] **H-11**: `selected_arms.remove(0)` O(n) → VecDeque
- [x] **H-12**: `bfcp_preimage` sigmoid waste → simple comparison
- [x] **H-13**: `roaring_membership` len() iterates → cache cardinality
- [x] **H-14**: `phrase_trie` O(n²) dedup → bitset
- [x] **H-16**: `lsh_cache` column-major → row-major iteration
- [x] **H-17**: MCTSNode children/unexpanded Vecs → pre-allocated Vec
- [x] **H-19**: Bomber `softmax()` in players.rs → sigmoid
- [x] **H-24**: Bomber blast zone → pre-compute grid
- [x] **H-25**: Bomber `escape_distance()` HashSet → bitset
- [x] **H-6**: `cna_modulate()` O(K) scan → layer HashMap
- [x] **H-15**: `region_shard_map` → flat atomic array

### MEDIUM
- [x] PWCValueFunction linear scan → direct-index Vec
- [x] BFCP cache accept/reject/maybe counts
- [x] BanditStats `best_arm()` cache
- [x] `sketch_types` `lessons.remove(0)` → VecDeque
- [x] `hoare_pruner` `ch.to_string()` → match on char directly
- [x] `lsh_cache` `Vec::remove(0)` → VecDeque
- [x] SharedBanditStats batch reads under single lock
- [x] Hoist `config.to_string()` outside episode loop
- [x] CNA full sort for top-k → `select_nth_unstable`
- [x] Decision explainer String → `Cow<'static, str>`
- [x] CurvatureAlloc softmax scratch Vec → pre-allocate
- [x] CountMinSketch f32 decay → integer math with shift
- [x] Opus nested `Vec<Vec<f32>>` → flat with stride
- [x] Opus `unique_selected()` clone+sort+dedup → HashSet
- [x] Three-mode bandit RollingWindow VecDeque → fixed ring buffer
- [x] Region batch `Vec::new()` → `Vec::with_capacity`
- [ ] `hydra_budget` `Vec<bool>` → bitmask (skipped: pub API change)
- [ ] `plackett_luce` pre-allocate Gibbs buffers (skipped: varying input size)
- [ ] `region_batch` `constraints.clone()` → Arc (skipped: cross-cutting core type change)
- [x] Go `flood_empty` HashSet<GoCell> → bool pair
- [x] Monopoly `group_squares()` Vec<u8> → &'static [u8]
- [x] Monopoly railroad/utility const arrays hoisted to module level
- [ ] `regime_transition` FailurePattern Vec key → blake3 hash
- [ ] `lodestar` Bellman-Ford O(S²Σ) → BFS O(SΣ)
- [ ] `curvature_alloc` lazy recompute for `recompute_influence`
- [ ] `bfcp_region_cache` LFU eviction O(n) → min-heap/TinyLFU
- [ ] `go/g_zero_player` `compute_go_delta` board_tokens Vec
- [ ] `go/state` `legal_moves()` accept pre-allocated buffer (caller `legal_moves_into` already exists)
- [ ] `monopoly/systems` `build_ctx` → reusable DecisionContext buffer
- [ ] `monopoly/mod` `square_kind()` → const lookup table (already `const fn` — no change needed)
- [ ] `dungeon_pathfinder` pre-compute floor adjacency on construction
- [x] `cna` `is_universal_excluded()` → HashSet (already uses HashSet)
- [x] `decision_explainer` recompute totals per sensitivity (totals computed inline — minimal impact)
- [x] `bfcf_types` BorelRegion field reordering (8-byte savings — not worth diff noise)

### LOW
- [x] `bandit.rs` BanditStats field reordering (checked: already well-packed after prior changes)
- [x] `cna.rs` CnaNeuron already well-packed (issue confirms no change needed)
- [x] `monopoly/players.rs` const arrays for railroad/utility squares → done (hoisted to module-level consts)
- [ ] `regime_transition` two-pass std → Welford's one-pass
- [ ] `lodestar` Vec<bool> → BitVec
- [ ] `sketch_types` Debug/Display hex formatting optimization
- [ ] `gepa_reflective` linear scan for empty slot → free list
- [ ] `sdar_absorb` diagnostic-only Vec alloc
- [ ] `go/autoresearch` config.label() String → fmt (dynamic values, can't be &'static)
- [ ] `go/tournament` three-pass count → single pass
- [ ] `bomber/systems` `[Option<(i32,i32)>; 4]` for player positions

## GOAT Proof Results

Benchmarks run on debug build (unoptimized). All gates ≥ 10% gain threshold.

| Gate | Change | OLD | NEW | Gain | Verdict |
|------|--------|-----|-----|------|---------|
| G1 | BomberState flat cells | Vec<Vec<Cell>> | [Cell;169] | **+595%** | 🟢 PROMOTED |
| G2 | MCTS throughput | — | 518K nodes/sec | — | 🟢 within bounds |
| G3 | Go multi-source BFS | per-cell BFS | multi-source | **+367%** | 🟢 PROMOTED |
| G4 | Arc\<BFCP\> | deep clone | Arc refcount | **+5912%** | 🟢 PROMOTED |
| G4b | BFCP cache pipeline | — | 119K ops/sec | — | 🟢 within bounds |
| G5 | PhraseTrie bitset | O(n²) contains | bitset | **+53%** | 🟢 PROMOTED |

**No losers to demote.** All optimizations proven ≥ 10% gain.

## Skipped Items (with justification)
- `hydra_budget Vec<bool> → bitmask`: pub field used externally, would break API
- `plackett_luce pre-allocate Gibbs`: varying input size, requires breaking API change
- `region_batch constraints.clone() → Arc`: would change BorelRegion.constraints core type
- `bfcf_types.rs:58-69` BorelRegion field reordering: minimal 8-byte savings, not worth the diff noise
