# 001 Pruners Optimization Plan

Comprehensive audit of `src/pruners/` (~100 files, ~65K lines) against the optimization guide.

## Methodology

5 parallel sub-agents analyzed all files, categorized findings by priority (HIGH/MEDIUM/LOW),
and identified cross-cutting themes.

---

## CRITICAL (Immediate — Eliminates 10K+ allocations per hot-path cycle)

### C-1: BomberState `cells: Vec<Vec<Cell>>` → flat array `[Cell; 169]`
- **File**: `game_state/bomber_state.rs:65`
- **Impact**: Every MCTS `advance()` and `select_inline()` does 13 heap allocations per clone. With 500-2000 tree nodes per search, this eliminates **~10K-25K heap allocs/search**.
- **Fix**: `pub cells: [Cell; ARENA_W * ARENA_H]` — clone becomes single `memcpy` of 169 bytes.

### C-2: `available_actions()` returns `Vec<BomberAction>` → `ArrayVec<BomberAction, 7>`
- **File**: `game_state/bomber_state.rs:415`
- **Impact**: Called in `select_inline()`, `expand_and_rollout()`, `rollout()` — **~1000-3000 allocs/search**.
- **Fix**: Return `ArrayVec<BomberAction, 7>` or `SmallVec<[BomberAction; 7]>` (max 7 actions).

### C-3: GoHeuristic `influence()` — per-cell BFS → multi-source BFS
- **File**: `go/state.rs:642-691`
- **Impact**: For 200 empty cells on 19×19, runs 200 BFS passes → **72K+ allocations per `evaluate()`**. Called per legal move.
- **Fix**: Single multi-source BFS from all stones simultaneously — O(area) instead of O(empty × area).

---

## HIGH (Hot-path allocations / O(n) scans / correctness bugs)

### H-1: `Arc<BFCP>` instead of deep clone
- **Files**: `bfcp_region_cache.rs:94-118`, `bfcp_lfu_shard.rs:217-227`, `bfcp_lsh_cms.rs:71-89`
- **Impact**: Every cache hit/insert does full deep clone of BFCP partition (Vec of BorelRegion × Vec of HalfSpace).
- **Fix**: `Arc<BFCP>` — clones become atomic refcount bumps. **Single highest-impact cross-cutting change.**

### H-2: `soft_route_relevance()` allocates 3 Vecs per call
- **File**: `bandit.rs:899-908`
- **Impact**: Called per-node during DDTree construction.
- **Fix**: Pre-allocate scratch buffers in `BanditPruner`, reuse with `clear()`.

### H-3: `CurvatureInfluence arm_bandit_score()` allocates Vec per arm
- **File**: `bandit.rs:829-831`
- **Impact**: N×N allocations across all arm score computations.
- **Fix**: Cache concentration in struct, compute once in `prepare_episode()`.

### H-4: `AdversarialBreaker::is_valid()` allocates Vec per failure
- **File**: `regime_transition.rs:470-479`
- **Impact**: `is_valid()` is called per-candidate per-node. Failures are common.
- **Fix**: Pre-allocate scratch buffer with `RefCell<Vec<usize>>`.

### H-5: `SensitivityCache` uses `Arc<RwLock<HashMap>>` → papaya
- **File**: `decision_explainer.rs:41-42`
- **Impact**: Lock contention on every cache access. User rules mandate papaya.
- **Fix**: `papaya::HashMap<[u8; 32], Vec<f32>>`.

### H-6: `cna_modulate()` O(K) scan per layer
- **File**: `cna.rs:275-284`
- **Impact**: Iterates all circuit neurons to find matching layer. Most iterations wasted.
- **Fix**: Pre-compute `HashMap<usize, Vec<usize>>` layer → neuron indices.

### H-7: `is_circuit_neuron()` broken binary search
- **File**: `cna.rs:309-317`
- **Impact**: **BUG** — neurons sorted by delta, but binary search uses (layer, index) comparator. Gives incorrect results.
- **Fix**: Use `HashSet<(usize, usize)>` for O(1) lookup, or sort secondary index.

### H-8: `softmax()` + `kl_divergence()` allocate 3 Vecs per `m_step()`
- **File**: `vpd_em.rs:66-90, 395-422`
- **Impact**: Per-decode-step hot path. 3 allocations per EM iteration.
- **Fix**: Pre-allocate `student_log_p`, `teacher_log_p` in struct. Rewrite `softmax` in-place.

### H-9: `review_metrics` cascading atomic loads
- **File**: `review_metrics.rs:180-234`
- **Impact**: `summary()` causes 14 redundant `AtomicU64::load()` calls.
- **Fix**: Snapshot all 4 counters once, compute all ratios from snapshot.

### H-10: `sorted_by_elo()` allocates + sorts per sample
- **Files**: `proof/sketch_population.rs:313-317`, `proof/sketch_sampler.rs:248-332`
- **Impact**: Every sampling path (per decode step) allocates + sorts. `sample_random()` full-sorts just to pick random.
- **Fix**: Cache sorted order in population, invalidate on mutation. `sample_random()` → `HashMap::keys()` + random index. `sample_best_elo()` → `max_by_key`.

### H-11: `selected_arms.remove(0)` O(n) per eviction
- **File**: `opus/types.rs:236`
- **Impact**: Per-selection hot path.
- **Fix**: Replace `Vec<usize>` with `VecDeque<usize>` for O(1) `pop_front`.

### H-12: `bfcp_preimage` sigmoid waste
- **File**: `bfcp_preimage.rs:110-118`
- **Impact**: `sigmoid(x - 0.5) > 0.5` mathematically equals `x > 0.5`. ~50K wasted `exp()` + divisions per maybe region.
- **Fix**: `if relevance > 0.5 { accept } else { reject }`.

### H-13: `roaring_membership` — `len()` iterates 1024 words, `iter()` heap-allocates `Box<dyn Iterator>`
- **File**: `roaring_membership.rs:37, 69-81`
- **Fix**: Cache cardinality in `Bits` variant. Use enum-based iterator instead of `Box<dyn>`.

### H-14: `phrase_trie` O(n²) dedup
- **File**: `phrase_trie.rs:92, 109, 115`
- **Impact**: `result.contains()` in `get_boosted_tokens` is quadratic in boosted token count. Called every decode step.
- **Fix**: `HashSet<usize>` or `Vec<bool>` bitset for dedup.

### H-15: `region_shard_map` — papaya HashMap for 9 fixed entries
- **File**: `region_shard_map.rs:19`
- **Fix**: `[AtomicUsize; 9]` indexed by `(label as usize) * 3 + (tier as usize)`.

### H-16: `lsh_cache::SimHashFingerprint` column-major iteration
- **File**: `lsh_cache.rs:27-42`
- **Impact**: Strides by 64 f32s per inner iteration → terrible cache locality.
- **Fix**: Transpose loop — iterate logits outer, accumulate into `[f64; 64]`.

### H-17: Bomber MCTSNode children/unexpanded Vecs
- **File**: `game_state/mcts.rs:56-58, 69`
- **Impact**: 2 Vec allocations per tree node. 1000 nodes = 2000 allocs.
- **Fix**: `SmallVec<[usize; 7]>` or `ArrayVec<usize, 7>` (max 7 actions).

### H-18: Bomber per-tick `HashSet` in `score_action()`
- **File**: `bomber/players.rs:608-609`
- **Impact**: `HashSet` from bombs allocated per-action per-tick.
- **Fix**: Pre-compute once in `select_action()`, pass as `&[(i32, i32)]`.

### H-19: Bomber `softmax()` in players.rs — violates sigmoid rule
- **File**: `bomber/players.rs:724-732`
- **Impact**: Per project rules: "Use sigmoid not softmax".
- **Fix**: Replace with per-element sigmoid scoring.

### H-20: Go duplicated `board_neighbors`/`flood_group` (3 copies)
- **Files**: `go/players.rs:51-108`, `go/g_zero_player.rs:81-136`, `go/autoresearch.rs:360-414`
- **Fix**: Extract to shared `go/utils.rs`, use `GoState::neighbors()` + scratch buffers.

### H-21: Go `greedy_score`/`compute_move_score` clones GoState per candidate
- **Files**: `go/players.rs:260-299`, `go/g_zero_player.rs:229-270`
- **Impact**: ~200 full state clones per turn per player.
- **Fix**: Analytical delta computation or in-place try_move.

### H-22: Pathfinder `HashMap`/`HashSet` → flat arrays
- **File**: `pathfinder.rs:109-110, 180, 231`
- **Fix**: `Vec<Option<...>>` indexed by `row * cols + col` for came_from, `Vec<bool>` for visited.

### H-23: `bfcp_lsh_cms` rebuilds all bitmaps from scratch per `process()`
- **File**: `bfcp_lsh_cms.rs:152-167`
- **Fix**: Incremental diff-based update. Pre-allocate `Vec::with_capacity`.

### H-24: Bomber blast zone — `is_in_blast_zone()` O(bombs × range) per BFS step
- **File**: `game_state/bomber_state.rs:184-188`
- **Fix**: Pre-compute blast zone grid `[u8; 169]` once per `advance()`.

### H-25: Bomber `escape_distance()` `HashSet` + `VecDeque` every call
- **File**: `game_state/bomber_state.rs:196-218`
- **Fix**: `[bool; 169]` bitset for visited, pre-allocated `VecDeque`.

### H-26: `template_proposer` clones QueryTemplate per proposal
- **File**: `g_zero/template_proposer.rs:380-382`
- **Fix**: Extract needed data before mutable borrow, avoid clone.

### H-27: Bomber `validator_agent` clones `ArenaGrid` per player per tick
- **File**: `bomber/validator_agent.rs:586-597`
- **Fix**: Pass `&ArenaGrid` reference — trait already accepts it.

### H-28: `blake3_logit_hash` computed redundantly across pipeline
- **Files**: `bfcp_lfu_shard.rs:159-185`, `bfcp_lsh_cms.rs:71-78`
- **Fix**: Thread hash through: `fn process(logits, hash) -> ...`

---

## MEDIUM (Notable but bounded impact)

- [ ] `bfcf_types.rs:247-264` — `PWCValueFunction::value/update` linear scan → direct-index Vec
- [ ] `bfcf_types.rs:58-69` — `BorelRegion` field reordering (save 8 bytes/region)
- [ ] `bfcf_types.rs:187-208` — Cache accept/reject/maybe counts on BFCP
- [ ] `bandit.rs:274-281` — `best_arm()` cache in BanditStats
- [ ] `bandit.rs:1658-1725` — `SharedBanditStats` batch reads under single lock
- [ ] `bandit.rs:1549` — Hoist `config.to_string()` outside episode loop
- [ ] `regime_transition.rs:338` — `FailurePattern` Vec key → blake3 hash
- [ ] `cna.rs:320-325` — `is_universal_excluded()` → HashSet
- [ ] `cna.rs:233-249` — Full sort for top-k → `select_nth_unstable`
- [ ] `decision_explainer.rs:372-398` — String alloc per attribution → `&str` / `Cow`
- [ ] `decision_explainer.rs:511-536` — Recomputed totals per sensitivity call
- [ ] `lodestar.rs:262-296` — Bellman-Ford O(S²Σ) → BFS O(SΣ)
- [ ] `curvature_alloc.rs:129` — Softmax scratch Vec alloc → pre-allocate
- [ ] `curvature_alloc.rs:83-95` — Lazy recompute for `recompute_influence`
- [ ] `count_min_sketch.rs:84-90` — f32 decay → integer math with shift
- [ ] `opus/types.rs:134` — Nested `Vec<Vec<f32>>` → flat with stride
- [ ] `opus/types.rs:357` — `unique_selected()` clone+sort+dedup → HashSet/bitmap
- [ ] `hydra_budget.rs:22-34` — `Vec<bool>` → bitmask, `skipped: Vec<usize>` → `&[bool]`
- [ ] `three_mode_bandit.rs:410-436` — `RollingWindow` VecDeque → fixed ring buffer
- [ ] `plackett_luce.rs:230-276` — Pre-allocate Gibbs sampler buffers in struct
- [ ] `sketch_types.rs:493-498` — `lessons.remove(0)` → VecDeque
- [ ] `hoare_pruner.rs:167` — `ch.to_string()` → match on char directly
- [ ] `lsh_cache.rs:85-89` — `Vec::remove(0)` → VecDeque
- [ ] `bfcp_region_cache.rs:146-157` — LFU eviction O(n) → min-heap or TinyLFU
- [ ] `go/g_zero_player.rs:285-321` — `compute_go_delta` board_tokens Vec → pre-compute or defer
- [ ] `go/state.rs:246-256` — `legal_moves()` → accept pre-allocated buffer
- [ ] `go/state.rs:405-432` — `flood_empty` HashSet for 2 values → bool pair
- [ ] `monopoly/systems.rs:70-151` — `build_ctx` → reusable DecisionContext buffer
- [ ] `monopoly/mod.rs:532-576` — `square_kind()` → const lookup table
- [ ] `monopoly/group_squares` → return `&'static [u8]`
- [ ] `dungeon_pathfinder.rs:225-231` — Pre-compute floor adjacency on construction
- [ ] `region_batch.rs:108` — `Vec::new()` → `Vec::with_capacity`
- [ ] `region_batch.rs:138,146` — `constraints.clone()` → `Arc<Vec<HalfSpace>>`

---

## LOW (Infrequent or minor)

- [ ] `bandit.rs:199-208` — `BanditStats` field reordering
- [ ] `regime_transition.rs:87-103` — Two-pass std → Welford's one-pass
- [ ] `cna.rs:33-41` — `CnaNeuron` already well-packed
- [ ] `lodestar.rs:58` — `Vec<bool>` → BitVec (only for large state spaces)
- [ ] `sketch_types.rs:104,111` — Debug/Display hex formatting optimization
- [ ] `gepa_reflective.rs:298` — Linear scan for empty slot → free list
- [ ] `sdar_absorb.rs:381` — Diagnostic-only Vec alloc
- [ ] `go/autoresearch.rs:130-139` — `config.label()` String → `&'static str`
- [ ] `go/tournament.rs:491-503` — Three-pass count → single pass
- [ ] `monopoly/players.rs:280,303` — Const arrays for railroad/utility squares
- [ ] `bomber/systems.rs:462-464` — `[Option<(i32,i32)>; 4]` for player positions

---

## Cross-Cutting Themes

1. **Scratch buffer pattern**: Most hot-path allocation issues solved by: allocate once in struct, pass `&mut`, `clear()` before use.
2. **`Arc<T>` for shared immutable data**: BFCP, constraints, arena grids — `Arc` turns deep clones into refcount bumps.
3. **Fixed-size arrays for bounded domains**: `Vec<T>` where domain is 4-7 elements → `[T; N]` or `ArrayVec<T, N>`.
4. **Flat arrays over `Vec<Vec<T>>`**: 2D grids → `[T; W*H]` with `row * W + col` indexing.
5. **Code deduplication**: `move_target`, `update_bombs`, `update_powerups`, `update_opponents` copied across 8+ bomber files. `board_neighbors`/`flood_group` copied across 3 go files.
6. **Hashing redundancy**: `blake3_logit_hash` computed 2-3× per pipeline call.
7. **`Vec::remove(0)` anti-pattern**: Found in 3 files (opus, lsh_cache, sketch_types). Use `VecDeque`.

---

## GOAT Gate Recommendations

Feature-gate the biggest changes to measure impact:

| Gate | Change | Metric |
|------|--------|--------|
| `goat_flat_cells` | `BomberState cells: [Cell; 169]` | MCTS nodes/sec |
| `goat_arrayvec_actions` | `available_actions() → ArrayVec` | MCTS nodes/sec |
| `goat_arc_bfcp` | `Arc<BFCP>` in cache pipeline | allocations/tick |
| `goat_multisource_bfs` | Go `influence()` multi-source BFS | evaluate() μs |
| `goat_scratch_bandit` | Pre-allocated bandit scratch buffers | DDTree nodes/sec |

Promote to default when benchmark shows ≥10% improvement.

---

## Implementation Order (Suggested)

1. **C-1 + C-2** (BomberState flat cells + ArrayVec) — biggest MCTS win, isolated change
2. **H-1** (Arc<BFCP>) — cross-cutting, high impact
3. **C-3** (Go multi-source BFS) — algorithmic improvement
4. **H-2 + H-3** (Bandit scratch buffers) — per-node improvement
5. **H-7** (CNA binary search bug) — correctness fix
6. **H-8** (VPD-EM pre-allocate) — per-decode-step win
7. **H-20 + H-21** (Go dedup + avoid advance() clones) — maintenance + perf
8. **Remaining HIGH items** — in order of estimated impact

---

TL;DR: Found **~100 optimization opportunities** across 100+ files. The 3 CRITICAL items (flat cells, ArrayVec actions, multi-source BFS) alone eliminate tens of thousands of allocations per MCTS search/evaluate cycle. The `Arc<BFCP>` change eliminates deep clones across the entire cache pipeline. A correctness bug was found in CNA binary search. Implementation should be gated behind GOAT feature flags and benchmarked before promotion.
