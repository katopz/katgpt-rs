# Plan 091: Go Self-Play Komi Imbalance Fix

> Issue: [#060 Go Self-Play Komi Imbalance — 98.6% Black Wins](../.issues/060_go_selfplay_komi.md)

## Tasks

- [x] T1: Add `GoState::set_komi()` setter in `state.rs`
- [x] T2: Add adaptive komi config fields to `GoGZeroSelfPlayConfig`
- [x] T3: Add komi history + score reward fields to `GoGZeroSelfPlayResults`
- [x] T4: Implement adaptive komi + score-based rewards in `run_gzero_selfplay()`
- [x] T5: Update `go_04_gzero.rs` example to report komi adjustments
- [x] T6: Add unit tests for adaptive komi logic
- [x] T7: Run clippy + existing tests, fix diagnostics

## Context

GZero self-play produces **98.6% Black wins** on 9×9 with `komi=7.5`. Template-based bots
amplify first-move advantage — all template deltas are negative for White. 7.5 komi is
calibrated for strong players, not template-based bots.

## Approach: Adaptive Komi + Score-Based Rewards

### Adaptive Komi (Option A)
- Start with `komi=7.5`
- Every 100 episodes, compute win rate over last window
- If `black_win_rate > 0.7`: `komi += 2.0`
- If `white_win_rate > 0.7`: `komi -= 2.0`
- Clamp to `[0.0, 20.0]`
- Log adjustments at absorb-compress checkpoints

### Score-Based Rewards (Option C)
- Use `state.score()` margin instead of binary win/loss
- `reward = score / abs(score).max(1.0)` → normalized to [-1, 1]
- Gives partial credit even to losing side if they played well
- Feed score-based reward into template delta observation

## Files to Modify

| File | Change |
|------|--------|
| `src/pruners/go/state.rs` | T1: Add `set_komi()` setter |
| `src/pruners/go/g_zero_player.rs` | T2-T4: Config, results, adaptive komi loop |
| `examples/go_04_gzero.rs` | T5: Report komi in output |
| `tests/go_komi_test.rs` | T6: Unit tests |

## Success Criteria

- [ ] Black win rate between **40-60%** over 500 episodes
- [ ] Template deltas reflect move quality, not color assignment
- [ ] At least 2 templates promoted via delta-gating absorb-compress
- [ ] Zero regressions in existing tests

## Design Notes

### `GoGZeroSelfPlayConfig` additions:
```rust
pub initial_komi: f32,          // default: 7.5
pub adaptive_komi: bool,        // default: true
pub komi_adjustment_step: f32,  // default: 2.0
pub komi_min: f32,              // default: 0.0
pub komi_max: f32,              // default: 20.0
pub komi_window: usize,         // default: 100 (episodes between adjustments)
pub score_based_rewards: bool,  // default: true
```

### `GoGZeroSelfPlayResults` additions:
```rust
pub komi_history: Vec<(usize, f32)>,           // (episode, komi)
pub final_komi: f32,
pub avg_score_margin: f32,
```

### Adaptive komi adjustment logic (inside episode loop):
```rust
if config.adaptive_komi && episode_num % config.komi_window == 0 && episode_num > 0 {
    let window_start = episode_num.saturating_sub(config.komi_window);
    let window_episodes = &episodes[window_start..episode_num];
    let window_black_wins = window_episodes.iter()
        .filter(|e| e.winner == Some(GoCell::Black)).count();
    let window_total = window_episodes.len().max(1);
    let black_wr = window_black_wins as f32 / window_total as f32;

    if black_wr > 0.7 {
        current_komi = (current_komi + config.komi_adjustment_step).min(config.komi_max);
    } else if black_wr < 0.3 {
        current_komi = (current_komi - config.komi_adjustment_step).max(config.komi_min);
    }

    komi_history.push((episode_num, current_komi));
}
```

### Score-based reward (replaces binary win counting):
```rust
let score = state.score(); // positive = Black advantage
let reward = score / score.abs().max(1.0); // normalized [-1, 1]
```

## Dependencies

- Plan 073 (Go Opening Heuristic) ✅ complete
- Plan 074 (Go HL Credit Assignment) ✅ complete
- Plan 065 (AutoGo Distillation) ✅ complete