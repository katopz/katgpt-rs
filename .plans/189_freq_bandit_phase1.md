# Plan 189: Oscillatory State-Space Modelless Distillation — Phase 1 (FreqBandit)

## Status: ✅ Phase 1 Complete (GOAT, Default On)

## Summary

FreqBandit uses spectral analysis of recent token streams to adaptively select speculative decode parameters. A simple DFT dot-product classifies token windows into three frequency bands, and a UCB1 bandit learns which band configuration yields the best `acceptance_rate × latency_improvement`.

## Implementation

- **File**: `src/freq_bandit.rs` (402 lines, under 500-line limit)
- **Feature gate**: `freq_bandit = ["bandit"]` in `Cargo.toml`
- **Default on**: Added to `default` and `full` feature sets
- **Module**: `#[cfg(feature = "freq_bandit")] pub mod freq_bandit;` in `src/lib.rs`

## Tasks

- [x] Create `FrequencyProfile` struct with band energies and spectral entropy
- [x] Create `FrequencyBand` enum (Low/Mid/High) with `#[repr(u8)]`
- [x] Implement `token_stream_spectrum()` — DFT dot-product for small windows
- [x] Create `FrequencyBandit` struct with UCB1 selection
- [x] Create `SpecBandConfig` — maps bands to speculative decode parameters
- [x] Feature gate `freq_bandit = ["bandit"]`
- [x] Module declaration in `src/lib.rs`
- [x] Add to default and full features (GOAT, Default On)
- [x] Sigmoid activation (NOT softmax) — verified via `test_sigmoid_not_softmax`

## Tests (17 pass)

| Test | Validates |
|------|-----------|
| `test_frequency_profile_cyclic` | Pattern 0,1,0,1 → High band |
| `test_frequency_profile_flat` | Constant → Low band (DC) |
| `test_frequency_profile_random` | Random → spread across bands |
| `test_frequency_profile_short_window` | <4 tokens → fallback |
| `test_frequency_profile_mid_period` | Period 8 → Mid band |
| `test_frequency_profile_low_period` | Period 32+ → Low band |
| `test_sigmoid_bounds` | σ(0)=0.5, σ(-∞)≈0, σ(+∞)≈1 |
| `test_sigmoid_not_softmax` | Weights don't sum to 1 |
| `test_bandit_cold_start_explores_all` | UCB1 visits all arms first |
| `test_bandit_selection_convergence` | Converges to best arm after 200 eps |
| `test_bandit_update_incremental_mean` | Q-value = incremental mean |
| `test_spec_config_mapping_distinct` | Each band → distinct config |
| `test_spec_config_low_deep_tree` | Low → deeper tree |
| `test_spec_config_high_shallow_tree` | High → more verify iterations |
| `test_bandit_map_to_spec_config` | Config mapping works |
| `test_full_pipeline` | End-to-end spectral→bandit→reward |
| `test_frequency_band_roundtrip` | Index roundtrip |

## Band Mappings

| Band | Period | draft_tree_width | draft_tree_depth | verify_iterations |
|------|--------|-----------------|-----------------|-------------------|
| Low  | >16    | 5               | 8               | 1                 |
| Mid  | 4–16   | 4               | 5               | 2                 |
| High | <4     | 3               | 3               | 3                 |

## TL;DR

FreqBandit Phase 1: DFT spectral analysis → 3-arm UCB1 bandit → speculative decode config. 17/17 tests pass, default-on, sigmoid (not softmax).
