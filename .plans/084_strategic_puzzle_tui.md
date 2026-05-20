# 084 — Strategic Puzzle TUI: Boss Chase, Traps, Keys, Levers

## Overview

Extend the tactical AI framework with a richer strategic puzzle game featuring:
- **Traps** — deadly tiles the player must avoid (or lure boss into)
- **Boss chase** — unkillable boss pursues player each step; must be lured into trap
- **Random keys** — 3 keys for 3 boxes, unknown mapping (permutation search)
- **3 levers** — must be pulled in correct order to open bridge
- **Bridge** — blocks path to goal until levers solved

This creates a **multi-layered constraint puzzle** where the DDTree must reason about:
- Path avoidance (traps)
- Time pressure (boss chase)
- Hidden information (key-box mapping)
- Sequence constraints (lever order)

## Architecture

### New File
- `examples/tactical_07_strategic.rs` — self-contained strategic puzzle with TUI

### Game State Extensions
```rust
pub struct StrategicState {
    pub r: usize,
    pub c: usize,
    pub keys_held: u8,          // bitmask of keys collected
    pub boxes_opened: u8,       // bitmask of boxes opened
    pub levers_pulled: u8,      // bitmask of levers activated
    pub lever_sequence: Vec<u8>,// order pulled (for validation)
    pub bridge_open: bool,      // all levers in correct order?
    pub boss_r: usize,
    pub boss_c: usize,
    pub boss_alive: bool,       // killed by trap?
    pub total_cost: u32,
    pub dead: bool,             // player hit trap or boss caught
}
```

### Map Symbols
| Symbol | Meaning |
|--------|---------|
| `B` | Player start |
| `O` | Boss start |
| `!` | Trap |
| `k`/`K`/`L` | Keys (index 0,1,2) |
| `b`/`B`/`N` | Boxes (index 0,1,2) |
| `1`/`2`/`3` | Levers (index 0,1,2) |
| `=` | Bridge (impassable until levers solved) |
| `G` | Goal |
| `#` | Wall |
| `.` | Floor |

### Target Types (for DDTree)
```
Key(0), Key(1), Key(2)
Box(0), Box(1), Box(2)
Lever(0), Lever(1), Lever(2)
Goal
```
= 10 strategic targets

### Boss AI
- After each player move, boss moves 1 step toward player (BFS shortest path)
- Boss treats traps as passable (walks onto them and dies)
- Boss treats walls as impassable
- If boss reaches player → game over
- If boss steps on trap → boss dies

### Key-Box Mechanic
- Random permutation σ: key i opens box σ(i)
- Player auto-tries all held keys when standing on box
- First matching key opens box (reveals treasure/item needed for goal)
- DDTree doesn't know σ — must explore permutations

### Lever Mechanic
- One correct order (e.g., 2→0→1) randomly assigned
- Pulling wrong lever resets all levers to unpulled
- DDTree explores 3! = 6 orderings

### Solver Strategy
1. DDTree explores target visit permutations (10 targets)
2. For each permutation, A* computes paths accounting for:
   - Trap avoidance (blocked set)
   - Boss position tracking (simulated per step)
   - Bridge state (impassable until levers solved)
3. Boss chase simulation happens during A* path execution
4. Key-box mapping is unknown — solver tries all mappings until one works

## Results

- ✅ **Solved**: 63-81 steps depending on boss rng, 7 targets, ~35ms (release), 383 DDTree nodes
- ✅ **Boss chase**: Boss walks into trap at (7,6) and dies — "lure boss into trap" mechanic works!
- ✅ **Rayon speedup**: Search phase 22x faster (28ms → 1.2ms), overall 1.86x (65ms → 35ms)
- ✅ **Bridge chokepoint**: Solid wall at row 11, only cols 7-9 are bridge — mandatory crossing
- ✅ **Key-box mapping**: seed=42 → key_mapping=[0,1] (identity), lever_order=[0,1,2]
- ✅ **Boss can't cross bridge**: Once player crosses, boss is stuck — safety zone

### DDTree + Rayon Analysis

| Phase | Sequential | Parallel | Speedup |
|-------|-----------|----------|---------|
| DDTree build | 34ms | 34ms | 1.0x (inherent sequential) |
| Search (try_sequence) | 28ms | 1.2ms | **22x** |
| **Total** | **65ms** | **35ms** | **1.86x** |

**Why rayon doesn't help DDTree build**: The tree expansion is a sequential heap-based best-first search. Each pop depends on prior state. With vocab_size=8, per-iteration parallelism has too much overhead. The pruner calls (A* pathfinding) are the bottleneck but too few per iteration.

**Where rayon shines**: Testing DDTree candidate orderings in parallel — each `try_sequence` call is independent (boss simulation, A* pathfinding, key-box matching).

## Tasks

- [x] T1: Create `examples/tactical_07_strategic.rs` with map, state types, constants
- [x] T2: Implement `StrategicGame` struct (parse map, initial state, apply_action)
- [x] T3: Implement trap mechanic (player dies on trap, boss dies on trap)
- [x] T4: Implement boss chase (BFS move toward player every 3 steps)
- [x] T5: Implement key-box mechanic (seed-based permutation, try keys on boxes)
- [x] T6: Implement lever mechanic (correct order validation, bridge toggle)
- [x] T7: Implement bridge mechanic (solid wall chokepoint until levers solved)
- [x] T8: Implement `StrategicPruner` (ConstraintPruner for DDTree)
- [x] T9: Implement solver (hierarchical: DDTree strategic + A* tactical with boss sim)
- [x] T10: Implement TUI rendering (map, boss, traps, keys, boxes, levers, bridge, status panel)
- [x] T11: Implement animation and controls (step-through, auto-play, ← → Space R Q)
- [x] T12: Add puzzle configuration (seed=42, LCG Fisher-Yates shuffle for mapping + order)
- [x] T13: Test and verify solvability — ✅ 63 steps, boss dies on trap, all boxes opened
- [x] T14: Add rayon parallel search benchmark (1.86x overall, 22x search speedup)
- [ ] T15: Update README.md with new example entry

## Map Design (16×16 — 8 targets)

```
# # # # # # # # # # # # # # # #
# B . . . . . . . . . . k . . #   B(1,1)  O(7,7)
# . # # . # . . # . # . . # . #   k(1,12) j(7,3)      — 2 keys
# . . . . . . . . . . . . . . #   a(12,3) b(12,10)     — 2 boxes
# . # . ! # . # # . # ! . # . #   1(9,6) 2(9,9) 3(10,8) — 3 levers
# . . . . . . . . . . . . . . #   !(4,4) !(4,11) !(7,6) — 3 traps
# . # # . . . . . . . # # # . #   =(11,7-9)             — bridge
# . . j . . ! O . . . . . . . #   G(14,13)              — goal
# . # . # . # . . # . # . # . #
# . . . . . 1 . . 2 . . . . . #   8 targets (fits u128 DDTree):
# . # # . . . . 3 . . . # # . #   K0 K1 B0 B1 L0 L1 L2 Goal
# # # # # # # = = = # # # # # #
# . . a . . . . . . b . . . . #   Row 11 = solid wall + bridge chokepoint
# . . . . . . . . . . . . . . #   Boss can NEVER cross bridge (safety zone)
# . . . . . . . . . . . G . . #
# # # # # # # # # # # # # # # #
```

## Key Design Decisions

1. **8 targets max** — DDTree uses u128 with 16-bit tokens (128/16 = 8 max)
2. **2 keys + 2 boxes** — simpler mapping (2! = 2 permutations), keeps under limit
3. **3 levers** — still has 3! = 6 orderings to explore
4. **Boss speed = 3** — moves every 3 player steps, gives time to maneuver
5. **Boss can't cross bridge** — once player crosses, boss is trapped; creates safety zone
6. **Seed = 42** — deterministic config: key_mapping=[0,1], lever_order=[0,1,2]
7. **No lever reset** — wrong order = bridge stays closed; DDTree tries different orderings

## Constraints

- Boss starts far from player (middle area)
- Traps between player and objectives (force detours)
- Bridge blocks direct path to goal
- Keys scattered across map
- Boxes near bridge/lever area
- Levers in separate rooms (force exploration)

## Success Criteria

- [x] Solver finds solution with boss avoidance
- [x] Boss can be killed by luring into trap (dies at trap (7,6))
- [x] All 2 boxes opened with correct keys
- [x] All 3 levers pulled in correct order
- [x] Bridge opens after levers (solid wall chokepoint)
- [x] Player reaches goal
- [x] TUI shows all mechanics clearly (map, strategy panel, status panel, nav bar)
- [x] Solution verifiable end-to-end (63-81 steps, ~35ms release)
- [x] Rayon parallel search benchmark (1.86x overall speedup)