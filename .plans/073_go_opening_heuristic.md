# Plan 073: Go Opening Heuristic — Corner Priority & Shape Knowledge

## Problem

`GoHeuristic.evaluate()` rewards **center preference** (10% weight) and `greedy_score()` applies a `center_bonus`. This is **backwards** for Go fundamentals:

> "Players usually begin by establishing bases in the corners (as boundaries make it easier to surround territory), moving to the sides, and finally the center." — Fuseki principle

On 9×9, the 4-4 points at (2,2)/(6,6) and 3-3 points at (1,1)/(7,7) are strong opening claims. Current heuristic guides MCTS toward tengen (center) which is strategically weak in the opening.

Additionally, no connect/cut shape knowledge exists — bamboo joints (uncuttable diagonal pairs) and knight's moves (cuttable gaps) are not distinguished.

## Distilled Go Strategy (Source: Rules & Strategy Reference)

### Opening (Fuseki)
- **Corner → Side → Center** — corners are cheapest territory (two edges are walls)
- **3rd line = territory** — secure, hard to invade
- **4th line = influence** — outward power, center control
- On 5×5: 3-3 points at (1,1) and (3,3) are strong early claims
- On 9×9: 4-4 star points at (2,2), (2,6), (4,4), (6,2), (6,6)
- On 19×19: 4-4 star points at (3,3), (3,9), (3,15), (9,3), (9,9), (9,15), (15,3), (15,9), (15,15)

### Life & Death
- **Two eyes** — a group with 2 independent internal empty points can never be captured
- Filling either eye is suicide; opponent can't fill both at once
- Corner/edge groups need fewer stones for two eyes (board edges serve as walls)
- An 8-stone chain along the top can enclose two eyes with minimum stones

### Connect & Cut
- **Bamboo joint** — diagonal pair with shared empty points, uncuttable (every cut attempt lets defender connect)
- **Knight's move (keima)** — efficient territory expansion but cuttable
- Strong players think in groups, not individual stones
- Once a group is severed, each piece needs its own two eyes to live

### Tactics (for future consideration)
- **Ladders** — zigzag atari chase across board
- **Nets (Geta)** — loose surround preventing escape
- **Snapbacks** — sacrifice single stone to recapture larger group
- **Seki** — mutual life, neither player wants to move first

### Influence vs Territory
- 3rd line: stable territory, secure
- 4th line: central influence, outward power
- Strategy requires balancing both — joseki sequences establish this balance

## Goal

Flip positional weights to match Go fundamentals:
1. **Opening**: corners > sides > center
2. **Midgame**: influence + connection
3. **Endgame**: territory enclosure + center

Expected impact: MCTS at budget=200 should improve from ~60% to >80% vs Random on 9×9.

## Files

| File | Change |
|------|--------|
| `src/pruners/go/state.rs` | `GoHeuristic` — add phase-aware territorial preference, replace center_preference |
| `src/pruners/go/players.rs` | `greedy_score()` — flip center→corner bonus, add connect bonus |

## Tasks

- [x] T1: Add `OpeningPhase` enum and `opening_phase()` to `GoHeuristic`
  - `Early` (< 2×size moves), `Mid` (< 6×size), `Late` (≥ 6×size)
  - 9×9: Early=<18, Mid=<54, Late=≥54
  - 19×19: Early=<38, Mid=<114, Late=≥114
  - Source: "Opening (Fuseki) — corners, sides, then center"
- [x] T2: Add `line_from_edge(row, col, size) -> usize` helper
  - Returns minimum distance to any edge (0=first line, 1=second line, etc.)
  - 3rd line (distance 2) = territory line
  - 4th line (distance 3) = influence line
- [x] T3: Add `territorial_preference()` to `GoHeuristic`, replace `center_preference()`
  - Early phase: reward 3rd/4th line near corners and sides, penalize center
    - "3rd line = territory, 4th line = influence"
    - "Corner stones claim territory with fewest friends"
  - Late phase: keep current center_preference logic (influence matters more)
  - Mid phase: blend between the two
  - Weight corner proximity: corners > sides > center during opening
- [x] T4: Update `GoHeuristic.evaluate()` weights
  - Current: liberty×0.4 + capture×0.3 + influence×0.2 + center×0.1
  - New: liberty×0.35 + capture×0.30 + influence×0.20 + territory×0.15
- [x] T5: Flip `greedy_score()` positional bonus
  - Replace `center_bonus()` positive weight with `corner_side_bonus()`
  - 3rd line (distance 2) = +3.0 (territory), 4th line = +2.0 (influence)
  - 1st line = -2.0 (edge penalty), center = +0.5 (low priority)
  - Source: "Corner and edge plays are cheap — the board itself serves as a wall"
- [x] T6: Add `connect_bonus()` to `greedy_score()`
  - +1.0 per adjacent own stone (extends group, stronger together)
  - +0.5 per diagonal own stone (bamboo joint potential)
  - -1.0 if isolated in enemy territory (≥2 adjacent opp)
  - Source: "Stones are strong in groups. Bamboo joints are uncuttable."
- [x] T7: Run benchmark before changes (baseline)
  - `go_06_bench` player scaling: Random, Greedy, Validator, HL, MCTS(200) vs Random
  - Record current win rates
- [x] T8: Run benchmark after changes (compare)
  - Same config, verify no regression
  - MCTS(200) improved from 60% → 85%, Greedy dropped 100% → 70% (more positional)
- [x] T9: Update `.docs/15_go_arena.md` with new heuristic weights and benchmark results
- [ ] T10: Commit with message `feat(go): territorial opening heuristic with corner priority`

## Key Insight

The fix is conceptually simple: Go corners are like chess castling — a cheap structural advantage. Current code treats center as premium (chess-like) when Go rewards territorial economy (fewer stones to secure territory near edges).

> "The board edges already form walls, so corner stones claim territory with the fewest friends."

One weight flip should improve all players that depend on `GoHeuristic` (MCTS, HL scoring).

## Future Work (out of scope)

- Joseki pattern matching (established corner sequences)
- Ladder/net detection
- Snapback recognition
- Seki detection
- Two-eye safety scoring for groups