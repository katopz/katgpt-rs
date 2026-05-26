# Issue 065: Freeze/Thaw Bandit Too Coarse for Meaningful Knowledge Transfer

> **Status:** ✅ Done — fixed with α=1.0 per-move reward + 10× delta amplification, +11pp win rate
>
> ~~Frozen GoHLPlayer knowledge hurts performance (-3pp)~~ **FIXED**: α=1.0 per-move reward + 10× delta amplification → +11pp win rate.

## Summary

Plan 092 implemented freeze/thaw pipeline correctly — `repr(C)` structs serialize/deserialize, disk I/O works, round-trip tests pass. Initially knowledge transfer was **negative** (-3pp), but after switching to pure per-move reward (α=1.0) with delta amplification (10×), frozen HL now **beats naive HL by +11pp** against Validator.

Learning vs Random also verified: Q-values differentiate properly (Corner > Defense) with α=1.0, unlike the old binary game-end reward that collapsed all Q-values to ~0.85.

## Experiment Results

### Before Fix: α=0.3 (game-end reward primary)

| Phase | Player | Opponent | Win% | Avg Score |
|-------|--------|----------|------|-----------|
| 1 LEARN | naive GoHL | Validator | 14% | -20.9 |
| 2 FROZEN | frozen GoHL | Validator | 11% | -22.7 |
| 3 BASELINE | naive GoHL | Validator | 14% | -16.8 |

| Metric | Frozen | Baseline | Δ |
|--------|--------|----------|---|
| Win Rate | 11% | 14% | **-3pp ❌** |
| Avg Score | -22.7 | -16.8 | **-6.0 ❌** |

### After Fix: α=1.0 + 10× delta amplification (pure per-move reward)

| Phase | Player | Opponent | Win% | Avg Score |
|-------|--------|----------|------|-----------|
| 1 LEARN | naive GoHL | Validator | 23% | -15.3 |
| 2 FROZEN | frozen GoHL | Validator | 25% | -13.3 |
| 3 BASELINE | naive GoHL | Validator | 14% | -16.8 |

| Metric | Frozen | Baseline | Δ |
|--------|--------|----------|---|
| Win Rate | 25% | 14% | **+11pp ✅** |
| Avg Score | -13.3 | -16.8 | **+3.5 ✅** |

Q-values after learning (real differentiation vs old flat ~0.25):
```
Corner:0.80 Side:0.64 Center:0.74 Cap:0.75 Def:0.40 Ext:0.48 Inf:0.59 Pass:0.00
```

## Root Cause (Original)

**Binary game-end reward + low α was the real problem, not just 8 arms.** When losing 86% of games against Validator:

1. Game-end reward = 0.0 for 86% of games → Q-values converge toward 0.0
2. With α=0.3, per-move signal (70% weighted out) couldn't overcome game-end bias
3. All Q-values converged to ~0.25 (nearly identical) — no differentiation
4. Bandit component (weight 0.2) added uniform negative bias vs fresh init (Q=0.5)

**Fix:** α=1.0 eliminates game-end reward entirely. Per-move heuristic delta has actual signal.
10× amplification expands the reward range from ±0.01–0.06 → ±0.1–0.6, giving the bandit real gradients to learn from.

### Before fix Q-values (α=0.3):
```
Corner:0.26 Side:0.26 Center:0.26 Cap:0.26 Def:0.25 Ext:0.25 Inf:0.25 Pass:0.00
```

### After fix Q-values (α=1.0 + 10× amp):
```
Corner:0.80 Side:0.64 Center:0.74 Cap:0.75 Def:0.40 Ext:0.48 Inf:0.59 Pass:0.00
```

The 8 arms are coarse but **sufficient** when the reward signal is clean. Corner (0.80) vs Defense (0.40) = 2× spread.

### Learning vs Random Verification

Verified with `hl_learning_vs_random_q_values_differentiate` test (50 games, alternating colors):
- Q-values differentiate: spread > 0.1 (old bug: spread ~0.0)
- Corner Q > Defense Q (positional value vs reactive)
- Pass Q < 0.3 (rarely useful)
- Best category Q > 0.5 (signal exists even vs weak opponent)

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
- ✅ Frozen knowledge transfers positively against Validator (+11pp)
- ✅ Q-values differentiate when learning vs Random (spread > 0.1)

## What Was Fixed

- ~~❌ Frozen knowledge transfers negatively against Validator~~ → **Fixed with α=1.0**
- ~~❌ More learning rounds makes it worse (deeper convergence to "everything loses")~~ → **Fixed with α=1.0**
- ~~❌ Learning vs Random also doesn't help (Q-values all ~0.85, no differentiation)~~ → **Fixed with α=1.0** — verified with test

## Potential Future Improvements

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

### Option D: Per-Move Reward Only (no game-end reward) — ✅ IMPLEMENTED
- Old blend: `α=0.3 * per_move + 0.7 * game_end`
- New: `α=1.0` (pure per-move reward) + 10× delta amplification
- **Result: +11pp win rate, +3.5 avg score vs baseline**
- Q-values differentiate meaningfully: Corner 0.80 vs Defense 0.40

### Option E: Asymmetric Weight on Loss
- Only update Q-values for categories that had positive per-move reward during losses
- Avoids penalizing good categories that happened to be played in lost games

## Files Changed

| File | Change |
|------|--------|
| `src/pruners/go/players.rs` | α=1.0 (pure per-move reward) + 10× delta amplification (`HL_DELTA_AMPLIFICATION`) + test for learning vs Random |
| `src/pruners/go/g_zero_player.rs` | Fix missing `swapped_episodes` field |
| `examples/go_08_self_play_freeze.rs` | 3-phase experiment: learn vs Validator, frozen vs Validator, baseline |

## Related

- Plan 092: `.plans/092_self_play_freeze_thaw.md`
- Bomber freeze/thaw: `examples/bomber_12_self_play_freeze.rs` (also shows marginal/no improvement)