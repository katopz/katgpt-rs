//! Cross-sectional estimate head (Plan 628 T0.4).
//!
//! PLACEHOLDER — implemented by the T0.4 lane. Contract: kNN over a
//! comparables corpus (tabular content parsed FROM the state, and the
//! per-deployment outcomes ledger as it accumulates — the ledger
//! interface stubs here, Phase 2 fills it) → point predictor →
//! `ConformalIntervalCalibrator` overlay → median/p10/p90 on the caller's
//! grid (linear/log), p_below/p_above outside it. The Report-the-Floor
//! gate: beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (m=1)
//! via `run_floor_comparison` or the floor serves.
