# Plan 047: FFT Tactics Arena — 4v4 Turn-Based Battle

**Branch:** `develop/feature/047_fft_tactics_arena`
**Depends on:** Plan 033 (Bomber Arena pattern), Plan 035 (Monopoly FSM pattern)
**Goal:** Final Fantasy Tactics-inspired headless battle arena with 4v4 PVP, speed-based turn queue, classes, and AI strategies.

---

## Problem Statement

Create a self-contained FFT-style tactical RPG battle example that demonstrates:
- Speed-based turn queue (CT system)
- 4 classes: Knight, Archer, Black Mage, White Mage
- 8 units (4 players vs 4 enemies) with different AI strategies
- Attack, Defense, Heal, Potion mechanics with HP/MP
- Height-aware grid with basic movement (A*-like range calculation)
- Headless tournament output similar to bomber_01_arena.rs

## Design

### Game Mechanics (simplified FFT)
- **Grid:** 8x8 flat grid (height=0 for simplicity, future: add Z-axis)
- **Turn Order:** Units sorted by Speed stat, cyclic queue
- **Turn Phases:** Move → Action → Wait (simplified, no facing)
- **Actions:** Attack (melee/ranged), Defend, Magic (black/white), Potion
- **Stats:** HP, MP, Speed, Attack, Defense, Magic, Range
- **Damage Formula:** `dmg = (atk * skill_power) - (def * 0.5)` with min 1
- **Hit Chance:** Base 90% (no facing modifiers for v1)

### Classes
| Class | HP | MP | Spd | Atk | Def | Mag | Range | Special |
|-------|----|----|-----|-----|-----|-----|-------|---------|
| Knight | 120 | 20 | 3 | 14 | 12 | 4 | 1 | High defense |
| Archer | 80 | 30 | 5 | 10 | 6 | 6 | 4 | Ranged physical |
| BlackMage | 70 | 60 | 4 | 4 | 4 | 16 | 3 | AoE magic (Fire) |
| WhiteMage | 80 | 70 | 4 | 4 | 6 | 14 | 3 | Heal allies |

### Teams
**Party (Players 0-3):** Knight-Random, Archer-Greedy, BlackMage-Validator, WhiteMage-HL
**Enemy (Players 4-7):** Knight-HL, Archer-Validator, BlackMage-Greedy, WhiteMage-Random

### AI Strategies (adapted from bomber pattern)
- **Random:** Pick random valid action/target
- **Greedy:** Attack weakest enemy in range, heal if HP low, move toward nearest
- **Validator:** Safety-first, prioritize healing allies, avoid overextension
- **HL:** Bandit Q-learning on action types, adapts across rounds

### Module Structure
All in one example file `examples/fft_01_arena.rs` (no new pruner module needed for v1).
The example is self-contained with inline types, similar to a game jam prototype.

---

## Tasks

- [x] Task 1: Create `examples/fft_01_arena.rs` with core types (Stats, Class, Action, Team)
- [x] Task 2: Implement `BattleGrid` (8x8) with positions and movement range calculation
- [x] Task 3: Implement `Unit` struct with HP/MP/stats and `Class` enum with stat templates
- [x] Task 4: Implement speed-based `TurnQueue` (sorted by speed, cyclic)
- [x] Task 5: Implement `Action` enum and damage/heal resolution
- [x] Task 6: Implement AI trait `FftPlayer` with Random/Greedy/Validator/HL strategies
- [x] Task 7: Implement `run_battle()` tick loop with move→action phases
- [x] Task 8: Implement `main()` with tournament loop, scoreboard, and final standings
- [x] Task 9: Add `[[example]]` entry in Cargo.toml (no extra features needed)
- [x] Task 10: Test run and verify output — clean compile, ~59 ticks/round avg, HL Q-values converge
- [x] Task 11: Update README.md with FFT example entry
- [x] Task 12: Commit with conventional message (`feat: FFT Tactics Arena 4v4 turn-based battle example`)

---

## Architecture

```
examples/fft_01_arena.rs
├── Constants (GRID_W, GRID_H, ROUNDS, etc.)
├── Enums: Class, Team, ActionType, GameEvent
├── Structs: Stats, Unit, BattleGrid, TurnQueue, BattleResult
├── Trait: FftPlayer { select_action, name, reset, as_any_mut }
├── Players: RandomPlayer, GreedyPlayer, ValidatorPlayer, HLPlayer
├── Game Loop: run_battle() → move phase → action phase → next turn
└── main() → tournament → standings
```

## Output Format

```
╔═══ FFT Tactics Arena ═════════════════════════════════════════╗
║  Party: ⚔️Knight-Random 🏹Archer-Greedy 🔮BMage-Validator ✨WMage-HL ║
║  Enemy: ⚔️Knight-HL 🏹Archer-Validator 🔮BMage-Greedy ✨WMage-Random ║
╚═══════════════════════════════════════════════════════════════╝

Round   1: Winner=Party  Ticks=24  Kills=[P0:2, P1:1, E5:1]
Round   2: Winner=Enemy  Ticks=31  Kills=[E4:3, P3:1]
...

═══ Final Standings (100 rounds) ═══
  #1 Party   Wins=62  Losses=38  AvgTicks=27
  #2 Enemy   Wins=38  Losses=62  AvgTicks=27

  Unit MVP: ⚔️Knight-HL (57 kills, 12 deaths)
```

## Dependencies
- `fastrand` (already in Cargo.toml)
- `bandit` feature for HL bandit Q-values (already available)

## Scope Notes
- v1: Flat 8x8 grid, no height, no facing, no charge time, no AoE
- Future v2: Height layers, facing bonuses, charge abilities, AoE patterns
- Keep it under 800 lines, clean and readable
- No ECS dependency (pure data-driven, unlike bomber which uses bevy_ecs)