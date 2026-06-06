# SDPG Bandit Arena Benchmark (Plan 180, T10)

## Date: 2026-06-05

## Tournament Configuration

| Parameter | Value |
|-----------|-------|
| Players | 8 (Random, Greedy, Validator, HL, GZero, Rubric, SDAR, SDPG) |
| Matchups | 5 |
| Games/matchup | 50 |
| Total games | 250 |
| Map | Procedural |
| Tick limit | 300 |

## Results

| Rank | Player | W | L | Games | Win% | ELO |
|------|--------|---|---|-------|------|-----|
| 1 | Greedy | 16 | 34 | 50 | 32.0% | 1037 |
| 2 | Rubric | 0 | 0 | 0 | 0.0% | 1000 |
| 3 | HL | 74 | 176 | 250 | 29.6% | 897 |
| 4 | SDAR | 25 | 75 | 100 | 25.0% | 877 |
| 5 | Validator | 65 | 135 | 200 | 32.5% | 877 |
| 6 | GZero | 17 | 83 | 100 | 17.0% | 564 |
| 7 | **SDPG** | **18** | **132** | **150** | **12.0%** | **166** |
| 8 | Random | 0 | 150 | 150 | 0.0% | -1150 |

## GOAT Gate Assessment

| Criterion | Target | Actual | Pass? |
|-----------|--------|--------|-------|
| SDPG > HL | Win rate > 29.6% | 12.0% | ❌ FAIL |
| SDPG > Random | Win rate > 0% | 12.0% | ✅ PASS |
| SDPG ELO > HL ELO | > 897 | 166 | ❌ FAIL |

## Verdict: **NEGATIVE RESULT**

SDPG Bandit did NOT meet the GOAT gate (SDPG > HL > Random).

SDPG beats Random but underperforms HL, SDAR, GZero, and Validator.

## Root Cause Analysis

1. **Uniform teacher Q-values**: In tournament mode (no oracle replay data), SDPG uses uniform teacher Q-values. The centered log-ratio advantage is therefore meaningless — there's no teacher signal to distill from.

2. **Positive-advantage gating too restrictive**: Only receiving teacher signal on wins means SDPG gets very sparse learning signal in a 4-player arena where individual win rates are naturally low (~25%).

3. **No oracle data pipeline**: The plan calls for loading replay data from `bomber_04_replay_gen`, but this pipeline doesn't exist for SDPG tournament mode. Without actual oracle Q-values, SDPG is essentially a plain bandit with unnecessary overhead.

## Component Value Assessment

Despite the negative arena result, the SDPG infrastructure has independent value:

| Component | Status | Independent Value |
|-----------|--------|-------------------|
| `centered_log_ratio()` | ✅ Works | Useful for any teacher/student Q-value comparison |
| `BetaSchedule` | ✅ Works | Reusable warmup-decay schedule |
| `KlAnchor` (UFKL/URKL) | ✅ Works | Bandit Q-value stability — independently useful |
| `SdpgBanditPruner<P>` | ✅ Works | Correctly wraps BanditPruner with teacher signal |
| `SdpgPlayer` | ⚠️ Weak | Needs oracle data to be effective |

## Recommendation

- Keep `sdpg_bandit` as **opt-in** feature (not default)
- The KL anchoring component should be considered for extraction as independent bandit stability utility
- Future work: integrate with actual oracle replay data pipeline for meaningful teacher Q-values

## Update (Phase 8-9: Sigmoid + Oracle Pipeline)

| Change | Impact |
|--------|--------|
| Fixed `update_if_sdpg` missing from `arena_runner.rs` | SDPG 12% → 15.3% (bug fix, not oracle) |
| Added `sigmoid_advantage` / `raw_delta_advantage` modes | No improvement with uniform teacher Q |
| Added `template_id` to ReplaySample | Infrastructure for future template-level oracle |
| Added `bomber_19_sdpg_replay_gen` (burn-in + GOAT) | SDPG(oracle) still 14% < HL 28% |

### Why oracle doesn't help

All 8 templates converge to Q~0.88 (variance <0.04) during burn-in.
Bomberman outcomes depend on **action execution** (safety filter, bomb timing) not **template selection**.
SDPG's template-level oracle signal is the wrong abstraction for this domain.

### What actually helped

The `update_if_sdpg` fix in `arena_runner.rs` was the real win — the bandit was never learning from outcomes in prior tournaments.
