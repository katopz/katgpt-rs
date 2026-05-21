# Issue 065: Freeze/Thaw Bandit Too Coarse for Meaningful Knowledge Transfer

> Frozen GoHLPlayer knowledge hurts performance (-3pp) because 8 category arms compress all Q-values to ~0.25 when losing 86% of games.

## Summary

Plan 092 implemented freeze/thaw pipeline correctly — `repr(C)` structs serialize/deserialize, disk I/O works, round-trip tests pass. However, the **knowledge transfer is negative**: frozen HL performs worse than naive HL against Validator.

## Experiment Results

### Setup (3 phases, 100 rounds each)

| Phase | Player | Opponent | Win% | Avg Score |
|-------|--------|----------|------|-----------|
| 1 LEARN | naive GoHL | Validator | 14% | -20.9 |
| 2 FROZEN | frozen GoHL | Validator | 11% | -22.7 |
| 3 BASELINE | naive GoHL | Validator | 14% | -16.8 |

### Comparison

| Metric | Frozen | Baseline | Δ |
|--------|--------|----------|---|
| Win Rate | 11% | 14% | **-3pp ❌** |
| Avg Score | -22.7 | -16.8 | **-6.0 ❌** |

### More rounds = worse

| Rounds | Frozen Win% | Baseline Win% | Δ |
|--------|-------------|---------------|---|
| 100 | 11% | 14% | -3pp |
| 200 | 12% | 13% | -1pp |
| 300 | 8% | 14.3% | -6.3pp |
| 500 | 9.4% | ~14% | ~-5pp |

## Root Cause

**8 category arms is too coarse for Go.** When losing 86% of games against Validator:

1. All Q-values converge to ~0.25 (nearly identical)
2. Bandit component (weight 0.2) adds uniform negative bias
3. Heuristic (weight 0.8) still dominates but the bandit nudge is counterproductive
4. More rounds amplify the compression, making it worse

Q-values after learning vs Validator:
```
Corner:0.26 Side:0.26 Center:0.26 Cap:0.26 Def:0.25 Ext:0.25 Inf:0.25 Pass:0.00
```

No meaningful differentiation between categories.

## Go Player Rankings (from tournament)

| Rank | Player | vs Random Win% |
|------|--------|---------------|
| #1 | Validator | 100% |
| #2 | HL | 100% |
| #3 | Greedy | 70% |
| #4 | MCTS | 70% |
| #5 | Random | 30% |

Validator dominates HL head-to-head (~86% win rate).

## What Works

- ✅ `repr(C)` struct serialization/deserialization
- ✅ Disk I/O (`save_frozen`/`load_frozen`) — 92 bytes
- ✅ Magic/version validation
- ✅ Round-trip tests pass
- ✅ 3-phase experiment design (learn → frozen test → baseline)
- ✅ Alternating colors for fairness

## What Doesn't Work

- ❌ Frozen knowledge transfers negatively against Validator
- ❌ More learning rounds makes it worse (deeper convergence to "everything loses")
- ❌ Learning vs Random also doesn't help (Q-values all ~0.85, no differentiation)

## Potential Fixes

### Option A: Finer Bandit Granularity
- Per-position bandit (81 arms for 9×9) — too sparse
- Per-template bandit like GoGZeroPlayer (4 arms) — still coarse
- Per-quadrant + category hybrid (8×4 = 32 arms) — may work

### Option B: Curriculum Learning
- Learn vs Random → then vs Greedy → then vs Validator
- Gradual difficulty increase avoids "everything loses" compression

### Option C: Opponent-Specific Freeze
- Learn against Validator → freeze → replay against Validator
- Already tried, doesn't help with 8 arms

### Option D: Per-Move Reward Only (no game-end reward)
- Current blend: `α=0.3 * per_move + 0.7 * game_end`
- Per-move heuristic delta has more signal than binary win/loss
- Try `α=1.0` (pure per-move reward)

### Option E: Asymmetric Weight on Loss
- Only update Q-values for categories that had positive per-move reward during losses
- Avoids penalizing good categories that happened to be played in lost games

## Files Changed

| File | Change |
|------|--------|
| `src/pruners/go/g_zero_player.rs` | Fix missing `swapped_episodes` field |
| `examples/go_08_self_play_freeze.rs` | 3-phase experiment: learn vs Validator, frozen vs Validator, baseline |

## Related

- Plan 092: `.plans/092_self_play_freeze_thaw.md`
- Bomber freeze/thaw: `examples/bomber_12_self_play_freeze.rs` (also shows marginal/no improvement)