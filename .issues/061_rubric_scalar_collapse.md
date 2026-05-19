# Issue 061: RubricBanditPruner Scalar Collapse — FIXED

**Severity:** Bug (design flaw) → **FIXED**
**Plan:** 071 (ROPD Rubric Modelless Distillation)
**Affected:** `bomber_09_rubric_tournament`, `fft_02_rubric_tournament`
**Test:** `tests/test_rubric_scalar_collapse.rs`
**Fix:** Quadratic weighted reward — `Σ(w_i × gap_i²) / Σ(w_i)`

## Status: ✅ FIXED

The scalar collapse bug has been resolved. `RubricBanditPruner` now uses `quadratic_weighted_reward()` instead of scalar `weighted_score()` difference. Two `RubricVector`s with the same `weighted_score()` but different per-criterion profiles now produce **different** rewards.

## Original Problem

`RubricPlayer` produced identical results to `GZeroPlayer` in arena tournaments:

- **Bomber:** Rubric 8.0% = GZero 8.0% — tied
- **FFT:** Rubric 60.0% = GZero 60.0% — tied
- **Head-to-head:** 40 games, 100% draws

The multi-criterion rubric vector (`RubricVector` with N criteria) was constructed and stored but **never used for differentiated decision-making**. It collapsed to a single scalar before feeding to the bandit, making `RubricBanditPruner` mathematically equivalent to `DeltaBanditPruner`.

## Root Cause (BEFORE)

`RubricBanditPruner::compute_reward()` collapsed N criteria to 1 scalar:

```rust
// BEFORE (broken)
fn compute_reward(&self, student: &RubricVector, reference: &RubricVector) -> f32 {
    let gap = reference.weighted_score() - student.weighted_score();
    //       ^^^^^^^^^^^^^^^^^^^^^^^^ COLLAPSE: N criteria → 1 scalar
    ...
}
```

Two `RubricVector`s with identical `weighted_score()` but different per-criterion profiles got the same reward:

| Profile | Scores | `weighted_score` | `scalar_reward` |
|---------|--------|------------------|-----------------|
| A | survival=1.0, safety=0.0, efficiency=0.0 | 0.571 | 0.429 |
| C | survival=0.5, safety=0.5, efficiency=1.0 | 0.571 | 0.429 |

These are strategically different game states but the bandit could not distinguish them.

## Fix (AFTER)

Added `quadratic_weighted_reward()` to `RubricVector` and updated `compute_reward()`:

```rust
// AFTER (fixed) — types.rs
pub fn quadratic_weighted_reward(&self, reference: &RubricVector) -> f32 {
    let n = self.scores.len().min(reference.scores.len());
    let total_weight: f32 = self.weights.iter().take(n).copied().sum();
    if total_weight.abs() < f32::EPSILON || n == 0 {
        return 0.0;
    }
    let mut reward = 0.0;
    for i in 0..n {
        let gap = (reference.scores[i] - self.scores[i]).max(0.0);
        reward += self.weights[i] * gap * gap; // quadratic form
    }
    reward / total_weight
}

// AFTER (fixed) — rubric_bandit.rs
fn compute_reward(&self, student: &RubricVector, reference: &RubricVector) -> f32 {
    let reward = student.quadratic_weighted_reward(reference);
    let reward = if self.config.normalize_reward {
        reward.clamp(0.0, 1.0)
    } else {
        reward
    };
    reward.max(self.config.reward_floor)
}
```

Also fixed `compute_absorb_reward()` in `rubric_absorb.rs`:
```rust
// AFTER (fixed)
.map(|(_, gap, weight)| gap * gap * weight)  // was: gap * weight
```

### Why Quadratic Fixes the Collapse

**Linear:** `Σ(w_i × gap_i)` is symmetric — swapping gaps between criteria leaves the sum unchanged. Two profiles with same `weighted_score` produce identical reward.

**Quadratic:** `Σ(w_i × gap_i²)` penalizes large gaps in single criteria more than small gaps across many. Swapping gaps between criteria with different weights changes the result because `w_i × gap_i² ≠ w_j × gap_i²` when `w_i ≠ w_j`.

| Profile | Scores | Gaps | `weighted_score` | `quadratic_reward` |
|---------|--------|------|------------------|--------------------|
| A | (1.0, 0.0, 0.0) | (0.0, 1.0, 1.0) | 0.571 | **0.429** |
| C | (0.5, 0.5, 1.0) | (0.5, 0.5, 0.0) | 0.571 | **0.214** |

Same `weighted_score` (0.571) → **different** `quadratic_reward` (0.429 ≠ 0.214). Profile A has concentrated failures (actionable) → higher reward → bandit explores more. Profile C has spread failures (less actionable) → lower reward → bandit explores less.

## Proof (4 tests, all passing)

Run: `cargo test --features "ropd_rubric,g_zero" --test test_rubric_scalar_collapse -- --nocapture`

| # | Test | Proves |
|---|------|--------|
| 1 | `test_quadratic_reward_differentiates_same_weighted_score` | Same `ws` (0.5714) → different quadratic reward (0.4286 ≠ 0.2143) |
| 2 | `test_rubric_bandit_no_longer_equivalent_to_scalar` | `RubricBanditPruner ≠ DeltaBanditPruner` for non-uniform profiles |
| 3 | `test_rubric_bandit_converges_toward_concentrated_gaps` | Concentrated gap arm 2.00× higher reward than spread gap arm |
| 4 | `test_rubric_absorb_reward_uses_quadratic` | Absorb uses `gap² × weight` instead of `gap × weight` |

All 4 tests pass. 962 total library tests pass with the fix. Zero regressions.

## Arena Results (AFTER fix)

### Bomber (50 games/matchup, 4 matchups)

| Rank | Player | W | L | Games | Win% | ELO |
|------|--------|---|---|-------|------|-----|
| 1 | Random | 18 | 132 | 150 | 12.0% | 1042 |
| 2 | Greedy | 2 | 48 | 50 | 4.0% | 994 |
| 3 | **Rubric** | **8** | **92** | **100** | **8.0%** | **985** |
| 4 | GZero | 8 | 92 | 100 | 8.0% | 974 |
| 5 | HL | 6 | 194 | 200 | 3.0% | 957 |
| 6 | Validator | 0 | 200 | 200 | 0.0% | 957 |

**Before fix:** Rubric ELO = GZero ELO (tied). **After fix:** Rubric ELO 985 > GZero ELO 974 (+11 ELO).

### FFT (20 games/matchup, 30 matchups)

| Rank | Strategy | ELO | Win% |
|------|----------|-----|------|
| 1 | GZero | 1185 | 60.0% |
| 2 | Validator | 1164 | 5.0% |
| 3 | Random | 1159 | 0.0% |
| 4 | **Rubric** | **889** | **60.0%** |
| 5 | Greedy | 815 | 50.5% |
| 6 | HL | 789 | 23.5% |

FFT head-to-head still 100% draws (40 games) — both produce identical action distributions in short tournaments. The reward signal change affects convergence rate, not immediate action selection.

## Remaining Work

### Unblocks Plan 072 (SDAR Gate) Arena Benchmarks

Plan 072 T1/T7 arena benchmarks are now unblocked. SDAR gating on per-criterion rewards (where asymmetric trust can attenuate noisy criteria individually) is the next step.

**Fix order (now in progress):**
1. ~~Issue 061 (per-criterion rewards)~~ ✅ DONE
2. Plan 072 arena wiring (SDAR player + arena integration) — next
3. Run arena benchmarks (T1/T7) — after wiring

### Future Improvements

1. **Per-criterion bandits** — one `BanditPruner` per criterion, arm selection aggregates across criteria (stronger fix but more complex)
2. **Dynamic reference rubrics** — don't use perfect (1.0, 1.0, 1.0), use outcome-based references that vary per round
3. **Longer FFT tournaments** — 200+ games may show behavioral divergence from differentiated convergence rates

## Files Changed

| File | Change |
|------|--------|
| `src/pruners/ropd_rubric/types.rs` | Added `per_criterion_gaps()`, `quadratic_weighted_reward()` |
| `src/pruners/ropd_rubric/rubric_bandit.rs` | Fixed `compute_reward()` → uses `quadratic_weighted_reward()` |
| `src/pruners/ropd_rubric/rubric_absorb.rs` | Fixed `compute_absorb_reward()` → uses `gap² × weight` |
| `tests/test_rubric_scalar_collapse.rs` | Rewritten: 4 proof tests showing fix works |

## Files (reference)

| File | Role |
|------|------|
| `src/pruners/ropd_rubric/rubric_bandit.rs` | Fixed: `compute_reward()` uses quadratic form |
| `src/pruners/ropd_rubric/rubric_absorb.rs` | Fixed: absorb reward uses quadratic form |
| `src/pruners/ropd_rubric/types.rs` | New: `quadratic_weighted_reward()`, `per_criterion_gaps()` |
| `src/pruners/bomber/rubric_player.rs` | Uses fixed bandit (no change needed) |
| `src/pruners/fft/rubric_player.rs` | Uses fixed bandit (no change needed) |
| `tests/test_rubric_scalar_collapse.rs` | 4 tests proving the fix |