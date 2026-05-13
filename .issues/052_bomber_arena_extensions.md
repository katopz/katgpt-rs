# Issue 052: Bomber Arena Extensions — Complex Bombs, Custom Maps, Agent Validators

**Source:** Plan 033 (Bomberman Arena) — Out of Scope items now actionable
**Status:** Partial — Task B (B1-B3, B6) complete, Tasks A & C open
**Feature gate:** `bomber`, `bomber-wasm`

---

## Context

Plan 033 proved HL works (+177 score over Random). The bomber arena is feature-complete with 4 player architectures, TUI replay, and WASM validator infrastructure (78ns/check, batch API, zero-copy). Three extensions are now actionable:

1. **Complex bomb types** — ECS is clean, `Bomb` is a ZST marker, blast propagation uses a two-phase compute→apply pattern
2. **Custom maps** — Only `ArenaGrid::generate(seed)` exists; fixed maps would improve benchmark reproducibility
3. **Coding agent validators** — WASM loader is mature, replay capture exists, needs the outer agent loop

---

## Task A: Complex Bomb Types

### Current State

- `Bomb` is a ZST marker component — no bomb type field
- `BombRange { cells: u32 }` — single range for all bombs
- `BombFuse { owner: Entity, ticks_remaining: u32 }` — always `BOMB_FUSE_TICKS = 4`
- `BomberAction` has 6 variants: `Up, Down, Left, Right, Bomb, Wait`
- Blast propagation (`process_explosions`) uses 4 cardinal directions, stops at first wall, supports chain reactions

### Extension

Add `BombType` enum and modify blast behavior per type:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BombType {
    Timed,      // default: fuse-based, stops at walls
    Piercing,   // blast continues through destructible walls
    Remote,     // detonates on player action (new BomberAction::Detonate)
    Landmine,   // invisible until stepped on, 1-range instant blast
}
```

### Subtasks

- [ ] A1: Add `BombType` enum to `src/pruners/bomber/mod.rs`
- [ ] A2: Add `bomb_type: BombType` field to `Bomb` component (change from ZST to struct)
- [ ] A3: Add `BomberAction::Detonate` variant for remote bombs
- [ ] A4: Modify `place_bomb()` in `systems.rs` to accept `BombType` (default `Timed`)
- [ ] A5: Modify `propagate_blast()` — `Piercing` continues through `DestructibleWall` (destroys + continues)
- [ ] A6: Add remote detonation system — `Detonate` action triggers all `Remote` bombs owned by player
- [ ] A7: Add landmine trigger system — step on `Landmine` → instant 1-range explosion
- [ ] A8: Add `BombType` to replay/action serialization
- [ ] A9: Update `BomberWasmPruner` WASM state format — add bomb_type per bomb token
- [ ] A10: Update player heuristics — `HLPlayer` / `GreedyPlayer` consider bomb types
- [ ] A11: Add tests for each bomb type behavior
- [ ] A12: Add bomber example demonstrating all bomb types

### Design Notes

- `BombType` is per-bomb, not per-player — a player can carry and place different types
- PowerUp could grant bomb types: `PowerUpKind::BombPiercing`, `PowerUpKind::BombRemote`
- Remote bombs should have a max count (prevent spamming)
- Piercing blast should still stop at `FixedWall` — only passes through `DestructibleWall`
- Landmine is invisible to other players' grid view (fog of war consideration for future)

---

## Task B: Custom Maps (Fixed Arena)

### Current State

- Only `ArenaGrid::generate(seed: u64)` exists — fully procedural
- 13×13 grid with fixed pillar positions (even x, even y) and random destructible fill (~40%)
- Spawn safe zones: 3×3 around 4 corners
- No `from_cells()`, `load()`, or preset map support

### Extension

Add fixed arena support for reproducible benchmarks and competitive play:

```rust
impl ArenaGrid {
    pub fn fixed(template: &str) -> Self { ... }
    pub fn from_cells(cells: &[Vec<Cell>]) -> Self { ... }
}
```

### Subtasks

- [x] B1: Add `ArenaGrid::from_cells(cells: &[Vec<Cell>])` constructor — validates dimensions (13×13), border walls, spawn zones
- [x] B2: Add `ArenaGrid::fixed(template: &str)` — parse compact string format (e.g., `"##....##\n#......#\n..."`)
- [x] B3: Add 2-3 preset constants: `EMPTY_ARENA`, `STANDARD_ARENA`, `PILLAR_HEAVY_ARENA`
- [ ] B4: Update bomber examples to accept `--map <preset>` or `--seed <u64>` CLI arg
- [ ] B5: Add benchmark comparison: fixed map vs procedural (score variance across runs)
- [x] B6: Add tests: from_cells validation (bad dimensions, missing borders), fixed parsing roundtrip

### Design Notes

- Compact string format: `#` = FixedWall, `.` = Floor, `D` = DestructibleWall, `P` = PowerUpHidden
- Fixed maps must still satisfy invariants: border walls, spawn zones clear, pillar positions valid
- This is **not** file I/O — presets are `const` or `fn` returns, keeping the zero-dep approach
- Benchmark variance: fixed map should have σ < 5% across 1000 runs; procedural has σ ≈ 15-20%

---

## Task C: Coding Agent Validator Loop

### Current State

- `BomberWasmPruner`: loads WASM from bytes/file, fuel-limited sandbox, batch API, per-thread papaya pool
- ABI: `is_valid`, `relevance`, `name`, `version` (required) + `batch_is_valid`, `batch_relevance` (optional)
- WASM state: 169-cell grid + player position + bomb info, zero-copy serialization
- Replay capture: `bomber_05_replay_gen.rs` (P3/P4 winners), `bomber_06_replay_gen_v2.rs` (all players, enriched)
- HL proof: +177 score over Random, pattern proven
- **No agent loop exists** — the generation → evaluation → iteration pipeline

### Extension

Build an outer orchestration loop that:
1. Generates a WASM validator from rules/templates
2. Loads it via `BomberWasmPruner`
3. Evaluates in arena (survival rate, kill rate, score)
4. Iterates on the validator rules based on failure traces

### Subtasks

- [ ] C1: Define `ValidatorCandidate` struct — rules as serializable AST (not raw code string)
- [ ] C2: Define `ArenaEvaluation` struct — survival_rate, kill_rate, score, failure_traces
- [ ] C3: Implement `evaluate_validator(candidate, rounds) -> ArenaEvaluation` — runs bomber arena with candidate as NNPlayer's WASM
- [ ] C4: Implement `failure_traces()` — extract rounds where validator approved fatal moves
- [ ] C5: Implement `TemplateProposer` — rule templates with configurable thresholds (no neural model)
- [ ] C6: Implement `propose_from_trace(failures) -> Vec<ValidatorCandidate>` — generate fix candidates from failure patterns
- [ ] C7: Implement `AgentLoop` — propose → evaluate → filter → iterate, with max rounds and convergence check
- [ ] C8: Add bomber example: `bomber_07_agent_loop.rs` — runs agent loop, outputs best validator as `.wasm`
- [ ] C9: Add feature gate `bomber-agent` (depends on `bomber-wasm`)
- [ ] C10: Benchmarks: agent-discovered validator vs hand-written `ValidatorPlayer` rules

### Design Notes

- **Rule-based first, LLM later** — `TemplateProposer` uses pattern matching on failure traces, not neural generation
- Candidate rules are a serializable AST (thresholds, spatial predicates), not freeform Rust — this keeps the search space bounded and the output deterministic
- `evaluate_validator` reuses existing arena infrastructure — just swaps the player's WASM module
- Convergence: stop when `ArenaEvaluation.score` plateaus for N iterations or exceeds hand-written baseline
- WASM compilation happens externally (via `riir-validator-sdk`) — the loop only loads the resulting bytes
- The agent loop is **CPU-only** — no GPU, no training weights, just rule search + arena evaluation

---

## Priority

| Task | Scope | Impact | Effort |
|------|-------|--------|--------|
| **B: Custom Maps** | Low | High (benchmark reproducibility) | Low — trivial `from_cells` + presets |
| **A: Complex Bombs** | Medium | Medium (gameplay depth) | Medium — ECS extension + blast logic |
| **C: Agent Loop** | High | High (self-improving validators) | High — new orchestration layer |

**Recommended order:** B → A → C (low-hanging fruit first, then gameplay, then research)

---

## Dependencies

- Task A: only `bomber` feature, no external deps
- Task B: only `bomber` feature, no external deps
- Task C: requires `bomber-wasm` + `riir-validator-sdk` for WASM compilation

## References

- Plan 033: `.plans/033_bomberman_arena.md`
- Plan 034: `.plans/034_bomber_wasm_validator.md` (WASM infrastructure)
- Plan 037: `.plans/037_wasm_batch_zero_copy.md` (batch API, 6.5× speedup)
- Plan 032: `.plans/032_heuristic_learning_infrastructure.md` (AbsorbCompress, TrialLog)
- Source: `src/pruners/bomber/` (arena, systems, players, wasm_pruner, wasm_state)
- Examples: `examples/bomber_0{1,2,3,4,5,6}_*.rs`
