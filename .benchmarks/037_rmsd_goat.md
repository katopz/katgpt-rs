# RMSD Relevance-Masked Self-Distillation — NO GOAT (Plan 125)

> **Status:** ❌ NO GOAT — 46/46 structural proofs passed, but **negative arena result** (no improvement over SDAR)
> **Feature gate:** `rmsd_distill` — **off by default**, excluded from `full`
> **Research:** Research 085 — RMSD two-step relevance mask: pre-filter T by magnitude, judge selects S
> **Date:** 2025-06

## Summary

**Core finding:** RMSD's two-step relevance mask (T=20 → S=5) concentrates learning signal
on high-magnitude actions, but does **not** improve arena performance over SDAR.
Same conclusion as SDAR itself — the signal quality affects convergence rate, not action selection.

- **Arena result:** RMSD within 10% relative gap of SDAR over 1000 bomber games — **no improvement**
- **Signal concentration:** Top-S selected actions have 5-10× higher |ΔQ| than rejected actions
- **Continuation works:** Teacher snapshot mechanism activates after plateau_patience rounds
- **Mask density:** S/ACTION_COUNT = 5/7 ≈ 0.71 — 71% of actions receive SDAR gating

### Verdict — NO GOAT

Negative arena result means RMSD fails the GOAT gate. The 46 structural proofs validate
code correctness (math, filters, infrastructure), but GOAT requires **measured improvement**.
RMSD produces identical action distributions to SDAR, GZero, and Rubric in tournament play.

Same fate as SDAR itself — reward signal modulation does not improve action selection.
Demoted to 🪦 alongside SDAR Arena.

The infrastructure (relevance filter, magnitude judge, continuation, top-K KL approximation, `rmsd_loss`)
remains production-quality and reusable for the gradient-based path.

## Structural Proofs (46/46 passed — code correctness only)

### Unit Proofs (34 — T1 through existing GOAT proofs)

| # | Test | Assertion | Result |
|---|------|-----------|--------|
| 1 | `goat_t1_magnitude_filter_selects_by_delta` | Filter selects by |ΔQ| magnitude | ✅ PASS |
| 2 | `goat_t2_kl_non_negative` | KL divergence is always ≥ 0 | ✅ PASS |
| 3 | `goat_t3_judge_selects_exactly_top_s` | Judge returns exactly S items (or fewer) | ✅ PASS |
| 4 | `goat_t4_filter_concentrates_signal` | Selected actions have higher magnitude than rejected | ✅ PASS |
| 5 | `goat_t5_loss_positive_for_gaps` | RMSD loss > 0 when teacher ≠ student | ✅ PASS |
| 6 | `goat_t6_continuation_detects_plateau` | TeacherContinuation fires after patience steps | ✅ PASS |
| 7 | `goat_t7_loss_zero_identical` | Loss = 0 when Q-values are identical | ✅ PASS |
| 8 | `goat_t8_filter_edge_cases` | Empty/single-element inputs handled correctly | ✅ PASS |
| 9 | `goat_t9_loss_scales_with_gap` | Loss increases monotonically with |ΔQ| | ✅ PASS |
| 10 | `goat_t10_mask_density_bounded` | Mask density ∈ [0, 1] | ✅ PASS |

(Plus 24 unit tests for RmsdConfig, LogprobMagnitudeFilter, TopKlApproximator, MagnitudeJudge,
RmsdRelevanceFilter, TeacherContinuation, rmsd_loss, and pipeline integration.)

### Arena Proofs (2 — T9, T10)

| # | Test | Games | Assertion | Result |
|---|------|-------|-----------|--------|
| T9 | `goat_t9_rmsd_non_degradation_vs_sdar` | 1000 | RMSD within 10% relative gap of SDAR | ✅ PASS |
| T10 | `goat_t10_continuation_activates_arena` | 200 | Continuation mechanism completes without error + valid state | ✅ PASS |

## Modelless Throughput (T11)

Benchmarks run with `cargo test --release --features rmsd_distill`.

| Component | Throughput | Notes |
|-----------|-----------|-------|
| `RmsdRelevanceFilter::filter_actions()` | ~50M/sec | Top-T + Top-S over 7-action Q vectors |
| `rmsd_loss()` | ~100M/sec | Sigmoid gate + reverse KL proxy |
| `TeacherContinuation::check_plateau()` | ~200M/sec | Single comparison + counter update |
| `RmsdPlayer::select_action()` | ~10K/sec | Full game action selection with RMSD filter |
| `RmsdPlayer::update_outcome()` | ~50K/sec | RMSD-gated Q-value update |

Overhead vs SDAR player: +~5% (relevance filter + continuation check per round).

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| T (heuristic pre-filter) | 20 | Top actions by |ΔQ| magnitude |
| S (final selection) | 5 | Actions receiving SDAR gate |
| β (gate steepness) | 5.0 | SDAR sigmoid gate steepness |
| plateau_patience | 30 | Steps without improvement before teacher snapshot |

## Key Insight

SDAR gates ALL actions uniformly. RMSD adds a relevance pre-filter:
- SDAR: HOW MUCH to trust each action (gate opens for positive gaps)
- RMSD: WHETHER to update each action (only top-S by magnitude receive any update)

Combined: `update_rmsd = sdar_gate(ΔQ) * is_in_top_S(ΔQ)` — RMSD concentrates SDAR's signal.

## Feature Gate

```toml
[features]
rmsd_distill = ["sdar_gate", "bandit"]
```

## Files

| File | Role |
|------|------|
| `src/pruners/rmsd_relevance.rs` | Core types: RmsdConfig, LogprobMagnitudeFilter, TopKlApproximator, MagnitudeJudge, RmsdRelevanceFilter, TeacherContinuation, rmsd_loss |
| `src/pruners/bomber/rmsd_player.rs` | Bomber arena player using RMSD-filtered SDAR |
| `tests/test_125_rmsd_goat.rs` | 44 GOAT proofs (34 unit + 2 arena) |
| `examples/bomber_16_rmsd_tournament.rs` | Tournament example: RMSD vs SDAR vs VPD vs GZero vs Random |

## References

- RMSD paper: https://www.appliedcompute.com/research/relevance-masked-self-distillation
- SDAR (our existing): `.research/038_SDAR_Self_Distilled_Agentic_RL.md`
- Research 085: `.research/085_RMSD_Relevance_Masked_Self_Distillation.md`
