# Issue 060: Go Self-Play Komi Imbalance — 98.6% Black Wins

## Severity

Medium — self-play learning signal is broken for GZero (partially resolved)

## Summary

GZero self-play (`go_04_gzero`) produces 98.6% Black wins across 500 episodes on 9×9 with komi=7.5. The first-move advantage overwhelms template-based learning, making outcome-based delta-gating meaningless.

## Current Behavior

```
GZero Self-Play Results (500 episodes, 9×9, komi=7.5):
  Black Wins:  493 (98.6%)
  White Wins:    7 (1.4%)
  Draws:         0

Template δ Ranking:
  Capture:    +0.0000 (only neutral template)
  CornerStar: -7.25
  Tenuku:     -9.50
  Defend:     -50.22
```

## Root Cause Analysis

Komi IS applied correctly — `GoState::new()` uses `DEFAULT_KOMI = 7.5`, and `score()` adds komi to white_score before computing `black_score - white_score`. The problem is structural:

1. **Weak players amplify first-move advantage** — Both sides use identical templates (Capture, CornerStar, Tenuki, Defend). Black moves first, securing territory before White can respond. With no strategic depth, this compounds every turn.

2. **Delta-gating sees no signal** — Template deltas are computed from move quality, but the game outcome is predetermined by color assignment. All templates get negative δ because "playing any template as White = losing."

3. **7.5 komi is calibrated for strong players** — In professional Go, 7.5 komi produces ~50% Black wins. But pros extract maximum value from every move. Weak players leave so much on the board that the 7.5 point compensation is negligible.

## Proposed Solutions

### Option A: Adaptive Komi (Recommended)

Dynamically adjust komi based on observed win rates during self-play.

```text
if black_win_rate > 0.7:
    komi += delta_komi  (e.g., +2.0)
if white_win_rate > 0.7:
    komi -= delta_komi  (e.g., -2.0)
clamp komi to [0.0, 20.0]
re-evaluate every 100 episodes
```

### Option B: Swap Colors Mid-Training ✅ Implemented (T8)

Each agent plays both colors equally. Episode N: A=Black, B=White. Episode N+1: A=White, B=Black. Aggregate deltas across both colors per agent.

Implementation: `proposer_a`/`proposer_b` replace `black_proposer`/`white_proposer`. On odd episodes, color assignment is swapped via `match (swapped, state.to_play)` pattern. Each agent experiences equal Black/White assignments → per-agent win rates converge toward ~50%.

### Option C: Score-Based Rewards Instead of Win/Loss

Use margin of victory as reward signal instead of binary win/loss.

```text
reward = score / abs(score).max(1.0)  # normalized to [-1, 1]
```

This gives partial credit even to the losing side if they played well.

### Option D: Handicap Stone Placement

Instead of komi, give White 1-2 handicap stones (placed before game starts). More directly compensates for move-order advantage.

## Implemented Approach (Plan 091)

**Option A** (adaptive komi) + **Option C** (score-based rewards) + **Option B** (swap colors):

1. Adaptive komi: score-margin-guided, damped (0.5×), clamped ±10, converges 7.5→42 in ~300 eps
2. Score-based rewards: `reward = score / |score|.max(1)` → normalized [-1, 1]
3. Swap colors: each agent plays both sides equally (even eps normal, odd eps swapped)
4. Per-agent win rates now balanced (each agent wins ~50% across both colors)

## Files to Modify

| File | Change |
|------|--------|
| `src/pruners/go/g_zero_player.rs` | Add adaptive komi logic to `run_gzero_selfplay()`, change reward from binary to score-based |
| `src/pruners/go/state.rs` | Add `GoState::with_komi()` already exists, may need `set_komi()` |
| `examples/go_04_gzero.rs` | Report komi adjustments during training |
| `.docs/15_go_arena.md` | Update results after fix |

## Success Criteria

- [~] Black win rate: cumulative 98.6% (low-komi convergence phase), ~81% at komi=42 with 14.7% draws. Swap-colors gives per-agent balance (each agent plays both sides)
- [x] Adaptive komi algorithm converges correctly: 7.5 → 42 in ~300 episodes, score margin +30 → ~0
- [x] Swap-colors (Option B) implemented — `swapped_episodes == num_episodes / 2`, per-agent balance achieved
- [ ] Template deltas still reflect color assignment (templates too weak for komi alone)
- [ ] No templates promoted via absorb-compress (all δ below threshold)
- [x] Zero regressions — 760 existing tests pass, 6 new komi tests pass (4 komi + 2 swap-colors)
- [x] Updated docs with new results (`14_go_arena.md`)

### Production Run Results (500 episodes, initial_komi=7.5)

```
Komi convergence: 7.5→17.5→27.5→34.3→38.2→40.1→41.0→41.5→41.8→41.9
Score margin:     +30.2→+22.8→+13.6→+7.7→+3.8→+1.9→+1.0→+0.5→+0.2→~0
```

At pre-converged komi=42 (150 eps): B=121(80.7%) W=7(4.7%) D=22(14.7%)
Recommendation: use `initial_komi=42` for 9×9 production runs.

## Remaining Work

To achieve **true 50/50 color-based win rates** (not just per-agent balance):
- **Option D (handicap stones)** — more direct compensation than komi alone
- Stronger templates — current templates are barely above random play
- The color imbalance (Black ~91% at komi=7.5) persists structurally; swap-colors only ensures each agent plays both sides equally

## References

- `go_04_gzero` example output
- `.docs/14_go_arena.md` — full Go arena documentation (updated with komi results)
- Plan 065 — Go arena implementation
- Plan 091 — Adaptive komi + swap-colors fix