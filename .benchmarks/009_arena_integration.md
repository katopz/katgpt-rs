# Benchmark 009: Arena Integration — RubricPlayer Tournaments

**Date:** 2026-05-19
**Plan:** 077 (Arena Integration)
**Features:** `ropd_rubric,g_zero,bomber,fft`
**Command (Bomber):** `cargo run --example bomber_09_rubric_tournament --features ropd_rubric,g_zero,bomber --release`
**Command (FFT):** `cargo run --example fft_02_rubric_tournament --features ropd_rubric,g_zero,fft --release`

## Purpose

Pit `RubricPlayer` (Bomber) and `RubricFFTPlayer` (FFT) against the full player
hierarchy to validate Plan 071's hypothesis:

> **Bomber (single-axis):** Rubric ≈ GZero — rubric adds little when survival is dominant.
> **FFT (multi-axis):** RubricFFT > GZeroFFT — class-dependent rubrics help when quality has multiple axes.

## Players

| ID | Player | Type | Bomber | FFT |
|----|--------|------|--------|-----|
| P1 | Random | Baseline | ✅ | ✅ |
| P2 | Greedy | Heuristic | ✅ | ✅ |
| P3 | Validator | Safety rules | ✅ | ✅ |
| P4 | HL | Bandit Q-learning | ✅ | ✅ |
| P5 | GZero | Hint-δ gated absorb | ✅ | ✅ |
| P6 | Rubric | ROPD rubric-vector | ✅ | ✅ |

---

## Bomber Arena Results

**Configuration:** 4 matchups × 50 games, procedural maps, 300 tick limit

### Matchup 1: Baseline Hierarchy (Random, Greedy, Validator, HL)

| Player | Wins | Losses | Draws | Win% |
|--------|------|--------|-------|------|
| Random | 6 | 44 | 40 | 12.0% |
| Greedy | 2 | 48 | 40 | 4.0% |
| HL | 2 | 48 | 40 | 4.0% |
| Validator | 0 | 50 | 40 | 0.0% |

**Note:** Bomber 4-player FFA produces ~80% draws (multiple survivors at tick limit).
Wins indicate last-player-standing events.

### Matchup 2: GZero Challenge (Random, HL, GZero, Validator)

| Player | Wins | Losses | Draws | Win% |
|--------|------|--------|-------|------|
| Random | 6 | 44 | 38 | 12.0% |
| GZero | 4 | 46 | 38 | 8.0% |
| HL | 2 | 48 | 38 | 4.0% |
| Validator | 0 | 50 | 38 | 0.0% |

### Matchup 3: Rubric Challenge (Random, HL, Rubric, Validator)

| Player | Wins | Losses | Draws | Win% |
|--------|------|--------|-------|------|
| Random | 6 | 44 | 38 | 12.0% |
| Rubric | 4 | 46 | 38 | 8.0% |
| HL | 2 | 48 | 38 | 4.0% |
| Validator | 0 | 50 | 38 | 0.0% |

### Matchup 4: Championship (GZero, Rubric, HL, Validator)

| Player | Wins | Losses | Draws | Win% |
|--------|------|--------|-------|------|
| GZero | 4 | 46 | 42 | 8.0% |
| Rubric | 4 | 46 | 42 | 8.0% |
| HL | 0 | 50 | 42 | 0.0% |
| Validator | 0 | 50 | 42 | 0.0% |

### Bomber Aggregated Leaderboard

| Rank | Player | Total W | Total L | Games | Win% | ELO |
|------|--------|---------|---------|-------|------|-----|
| 1 | Random | 18 | 132 | 150 | 12.0% | 1042 |
| 2 | Greedy | 2 | 48 | 50 | 4.0% | 994 |
| 3 | Rubric | 8 | 92 | 100 | 8.0% | 985 |
| 4 | GZero | 8 | 92 | 100 | 8.0% | 974 |
| 5 | HL | 6 | 194 | 200 | 3.0% | 957 |
| 6 | Validator | 0 | 200 | 200 | 0.0% | 957 |

### Bomber Analysis

- **Rubric ≈ GZero** (8W vs 8W) — confirms single-axis hypothesis
- High draw rate (~80%) makes Bomber an unreliable tournament domain
- Random's high win rate reflects FFA chaos, not strategy quality
- GZero and Rubric are indistinguishable in 4-player Bomberman

---

## FFT Tactics Arena Results

**Configuration:** 30 round-robin matchups × 20 games = 600 total battles

### Win Rate Matrix (Party row vs Enemy column)

| Party \ Enemy | Random | Greedy | Validator | HL | GZero | Rubric |
|---------------|--------|--------|-----------|-----|-------|--------|
| **Random** | — | 0% | 0% | 0% | 0% | 0% |
| **Greedy** | 100% | — | 100% | 95% | 0% | 0% |
| **Validator** | 0% | 10% | — | 20% | 0% | 0% |
| **HL** | 35% | 40% | 50% | — | 0% | 0% |
| **GZero** | 70% | 100% | 50% | 85% | — | 0% |
| **Rubric** | 70% | 100% | 50% | 85% | 0% | — |

### FFT Aggregated Stats

| Strategy | Wins | Losses | Draws | Games | Win% |
|----------|------|--------|-------|-------|------|
| GZero | 120 | 0 | 80 | 200 | 60.0% |
| Rubric | 120 | 0 | 80 | 200 | 60.0% |
| Greedy | 101 | 91 | 8 | 200 | 50.5% |
| HL | 47 | 104 | 49 | 200 | 23.5% |
| Validator | 10 | 88 | 102 | 200 | 5.0% |
| Random | 0 | 115 | 85 | 200 | 0.0% |

### GZero vs Rubric Head-to-Head

| Direction | W | L | D | Games |
|-----------|---|---|---|-------|
| GZero(Party) vs Rubric(Enemy) | 0 | 0 | 20 | 20 |
| Rubric(Party) vs GZero(Enemy) | 0 | 0 | 20 | 20 |
| **Combined** | **0** | **0** | **40** | **40** |

**Result: Tie — 100% draws. Rubric and GZero are strategically equivalent in FFT.**

---

## Hypothesis Evaluation

### Plan 071 Hypothesis: "Rubric helps more in multi-axis (FFT) than single-axis (Bomber)"

| Domain | Axes | GZero Win% | Rubric Win% | Δ | Conclusion |
|--------|------|-----------|-------------|---|------------|
| Bomber | 1 (survival) | 8.0% | 8.0% | 0% | ✅ Confirmed: no rubric advantage |
| FFT | 3 (attack/heal/debuff) | 60.0% | 60.0% | 0% | ❌ Rejected: rubric ≠ GZeroFFT improvement |

### Finding

**Rubric and GZero are identical** in both domains. The 3-criterion rubric vector
(TaskFulfillment, Completeness, ConstraintSatisfaction) produces the same
effective signal as scalar Hint-δ. Possible explanations:

1. **Rubric criteria collapse to a single axis** — when all 3 criteria are
   positively correlated with "winning," the rubric degenerates to a scalar.
2. **Template proposer dominates** — UCB1 template selection is the same
   in both GZero and Rubric; the rubric/δ only affects absorb bandit gating,
   which may not influence template choice enough.
3. **Insufficient training episodes** — 200 games may not be enough for
   rubric-differentiated learning to emerge from bandit exploration.

### Recommendation

The rubric approach needs either:
- **Decorrelated criteria** — criteria that trade off against each other
  (e.g., "aggression" vs "safety" vs "efficiency")
- **Class-specific rubric weights** — Knight rubric ≠ WhiteMage rubric
- **More training episodes** — 1000+ games to let bandit learning differentiate

---

## Infrastructure Summary

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| `arena/types.rs` | Shared tournament types | 369 | ✅ 14 tests |
| `arena/scheduler.rs` | Round-robin scheduling | 126 | ✅ 10 tests |
| `arena/mod.rs` | Module index | 9 | ✅ |
| `bomber/arena_runner.rs` | Bomber match runner | ~320 | ✅ 8 tests |
| `fft/arena_runner.rs` | FFT battle runner | ~200 | ✅ 6 tests |
| `bomber_09_rubric_tournament.rs` | Bomber tournament example | 353 | ✅ Runs clean |
| `fft_02_rubric_tournament.rs` | FFT tournament example | 463 | ✅ Runs clean |

### Total Tests: 962 pass (features `ropd_rubric,g_zero,bomber,fft`)

### Tournament Runtime

| Tournament | Matchups | Games | Duration |
|------------|----------|-------|----------|
| Bomber | 4 × 50 | 200 | 0.3s |
| FFT | 30 × 20 | 600 | ~2s |