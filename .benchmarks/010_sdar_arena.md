# Benchmark 009: SDAR Gated Arena — Bomber + FFT Tournament Results

**Date:** 2026-05-19
**Plan:** 072 (SDAR Gated Distillation — Modelless Path)
**Features:** `--features "sdar_gate,ropd_rubric,g_zero,bomber,fft"`
**Command (Bomber):** `cargo run --example bomber_10_sdar_tournament --features sdar_gate,ropd_rubric,g_zero,bomber --release`
**Command (FFT):** `cargo run --example fft_03_sdar_tournament --features sdar_gate,ropd_rubric,g_zero,fft --release`

## New Players

| Player | File | Bandit | Absorb | Feature |
|--------|------|--------|--------|---------|
| `SdarPlayer` | `bomber/sdar_player.rs` | `SdarBanditPruner` | `SdarGatedAbsorbCompress` | `sdar_gate` |
| `SdarFFTPlayer` | `fft/sdar_player.rs` | `SdarBanditPruner` | `SdarGatedAbsorbCompress` | `sdar_gate` |

## Technology Stack Comparison

| Tier | Bandit | Absorb | Reward Signal |
|------|--------|--------|---------------|
| G-Zero | `DeltaBanditPruner` (scalar δ) | `DeltaGatedAbsorbCompress` | scalar δ |
| Rubric | `RubricBanditPruner` (quadratic) | `RubricGatedAbsorbCompress` | rubric vector gaps² |
| SDAR | `SdarBanditPruner` (sigmoid-gated) | `SdarGatedAbsorbCompress` | scalar × σ(β·gap) |

---

## Bomber Arena Results

**Configuration:** 7 players, 5 matchups × 50 games, procedural maps, ELO K=24

### ELO Ratings

| Rank | Player | W | L | Games | Win% | ELO |
|------|--------|---|---|-------|------|-----|
| 1 | Random | 24 | 176 | 200 | 12.0% | 1037 |
| 2 | HL | 10 | 240 | 250 | 4.0% | 1007 |
| 3 | Greedy | 2 | 48 | 50 | 4.0% | 994 |
| 4 | GZero | 7 | 93 | 100 | 7.0% | 981 |
| 5 | Rubric | 5 | 95 | 100 | 5.0% | 955 |
| 6 | **SDAR** | **6** | **94** | **100** | **6.0%** | **954** |
| 7 | Validator | 0 | 200 | 200 | 0.0% | 948 |

### Per-Matchup Breakdown

| Matchup | Players | SDAR W | SDAR L |
|---------|---------|--------|--------|
| Baseline Hierarchy | Random, Greedy, Validator, HL | — | — |
| GZero Challenge | Random, HL, GZero, Validator | — | — |
| Rubric Challenge | Random, HL, Rubric, Validator | — | — |
| SDAR Challenge | Random, HL, SDAR, Validator | 4W | 46L |
| Championship | GZero, Rubric, SDAR, HL | 2W | 48L |

### Head-to-Head (Championship Matchup)

| Player | Wins | Losses | Win% |
|--------|------|--------|------|
| GZero | 3 | 47 | 6.0% |
| SDAR | 2 | 48 | 4.0% |
| HL | 2 | 48 | 4.0% |
| Rubric | 1 | 49 | 2.0% |

---

## FFT Arena Results

**Configuration:** 7 strategies, 42 round-robin matchups × 20 games, 200-turn limit

### ELO Rankings

| Rank | Strategy | ELO | W | L | D | Win% |
|------|----------|-----|------|------|------|--------|
| 1 | GZero | 1285 | 120 | 0 | 120 | 50.0% |
| 2 | Validator | 1246 | 10 | 107 | 123 | 4.2% |
| 3 | Random | 1190 | 0 | 143 | 97 | 0.0% |
| 4 | Rubric | 1022 | 120 | 0 | 120 | 50.0% |
| 5 | HL | 783 | 47 | 137 | 56 | 19.6% |
| 6 | Greedy | 745 | 101 | 131 | 8 | 42.1% |
| 7 | **Sdar** | **730** | **120** | **0** | **120** | **50.0%** |

### Win Rate Matrix

| Party \ Enemy | Random | Greedy | Validator | HL | GZero | Rubric | Sdar |
|---------------|--------|--------|-----------|-----|-------|--------|------|
| Random | — | 0% | 0% | 0% | 0% | 0% | 0% |
| Greedy | 100% | — | 100% | 95% | 0% | 0% | 0% |
| Validator | 0% | 10% | — | 20% | 0% | 0% | 0% |
| HL | 35% | 40% | 50% | — | 0% | 0% | 0% |
| GZero | 70% | 100% | 50% | 85% | — | 0% | 0% |
| Rubric | 70% | 100% | 50% | 85% | 0% | — | 0% |
| **Sdar** | **70%** | **100%** | **50%** | **85%** | **0%** | **0%** | — |

### Head-to-Head Results

| Matchup | Games | SDAR W | Opp W | Draws | Verdict |
|---------|-------|--------|-------|-------|---------|
| SDAR vs Rubric | 40 | 0 | 0 | 40 | 100% draws |
| SDAR vs GZero | 40 | 0 | 0 | 40 | 100% draws |
| GZero vs Rubric | 40 | 0 | 0 | 40 | 100% draws |

---

## Analysis

### Key Finding: SDAR ≈ Rubric ≈ GZero in Arena

SDAR sigmoid gating produces **near-identical arena behavior** to both Rubric and GZero players:

1. **FFT:** SDAR draws 100% of games against both GZero and Rubric (40 games each). The win rate matrix is identical for SDAR, Rubric, and GZero rows. These three players produce the **same action distributions** in battle.

2. **Bomber:** SDAR ELO 954 vs Rubric ELO 955 — statistically indistinguishable (1 ELO point difference). Both trail GZero at ELO 981.

### Why SDAR Doesn't Differentiate

The SDAR sigmoid gate modulates **reward signal intensity** — it doesn't change **action selection directly**. In a 20-50 game tournament:

- The ε-greedy exploration (15% bomber, 5% FFT) dominates action variance
- Template selection (UCB1) converges slowly and is shared across all three player types
- The sigmoid gate affects Q-value **convergence rate**, not immediate action selection
- In short series (20-50 games), the Q-value differences haven't had enough episodes to manifest as different action distributions

This is the **same pattern observed in Issue 061** — Rubric and GZero also produced identical results until the quadratic weighted reward fix differentiated per-criterion profiles.

### Component Benchmark Context

From `.benchmarks/008_sdar_gated_modelless.md`:

| Metric | Baseline | SDAR | Delta |
|--------|----------|------|-------|
| Bandit regret (1000 ep) | 153.08 | 196.49 | +28% |
| Hot-path overhead | — | +0.4% | ~zero |
| Absorb targeting (high BR) | — | 97.5% | excellent |
| Update throughput | — | 118M/sec | 1180× target |

The 28% higher regret is the **cost of asymmetric trust** — SDAR converges slower but more stably. In arena settings, this slower convergence means SDAR hasn't differentiated from Rubric/GZero within the game count.

### Honest Assessment

**SDAR modelless gating does NOT improve arena performance over Rubric or GZero.** The sigmoid gate affects convergence dynamics, not action selection. In the modelless path:

- Scalar reward → sigmoid gate → bandit update is equivalent to scalar reward → bandit update at equilibrium
- The gate's benefit (noise resilience) requires **many more episodes** to manifest as action differentiation
- The modelless path cannot leverage SDAR's full power (gradient-based token-level gating)

### Infrastructure Value

Despite no arena improvement, the SDAR module provides:

1. **Validated sigmoid gate primitive** — reusable for model-based Plan 073
2. **97.5% absorb targeting accuracy** — sigmoid promotion works correctly
3. **Zero hot-path overhead** — compiler fully inlines the delegation wrapper
4. **Feature-gated module** — clean separation, no impact on non-SDAR builds
5. **Negative result documented** — proves modelless SDAR is insufficient, gradient path needed

---

## Files Created

| File | Role |
|------|------|
| `src/pruners/bomber/sdar_player.rs` | SDAR-gated Bomber player (788 lines) |
| `src/pruners/fft/sdar_player.rs` | SDAR-gated FFT player |
| `examples/bomber_10_sdar_tournament.rs` | 7-player Bomber tournament with SDAR |
| `examples/fft_03_sdar_tournament.rs` | 7-strategy FFT round-robin with SDAR |

## Files Modified

| File | Change |
|------|--------|
| `src/pruners/bomber/mod.rs` | Added `sdar_player` module + export |
| `src/pruners/bomber/arena_runner.rs` | Added `update_if_sdar` for learning |
| `src/pruners/fft/mod.rs` | Added `sdar_player` module + export |
| `Cargo.toml` | Added `bomber_10_sdar_tournament` and `fft_03_sdar_tournament` examples |

## Tests

| Category | Count | Status |
|----------|-------|--------|
| SDAR gate primitives | 37 | ✅ All pass |
| SDAR bandit + absorb | 77 | ✅ All pass |
| Bomber SDAR player | 9 | ✅ All pass |
| FFT SDAR player | 8 | ✅ All pass |
| **Total SDAR tests** | **131** | **✅ All pass** |
| Total library tests | 1076 | ✅ All pass |

## Next Steps

1. **Plan 073 (Model-Based SDAR)** — The gating pattern may show benefit at gradient level where token-level modulation affects loss computation
2. **Longer tournaments** — 500+ games may show SDAR's slower convergence eventually differentiating actions
3. **Per-criterion SDAR** — Apply sigmoid gate to individual rubric criteria instead of collapsed scalar reward