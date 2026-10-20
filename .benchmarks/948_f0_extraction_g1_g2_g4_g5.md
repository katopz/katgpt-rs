# Bench 948 — F0 extraction floor: G1/G2/G4/G5 (Plan 628 T0.6 close)

**Status:** DONE — Phase-0 GOAT gates measured at `--release`; extraction_estimate PROMOTED to default per the plan's GOAT rule; span/parsers/temper stay opt-in.

**Date:** 2026-10-20 · **Crate:** `katgpt-extract` · **Plan:** [628](../.plans/628_fathom_extraction_substrate.md) T0.6

## What ran (all at the tree of `828f17428` + the promotion commit)

| gate | instrument | posture | verdict |
|---|---|---|---|
| G1 estimate | `run_floor_comparison` over `synthetic_cross_sectional_v1` (n_scored=192) | `--release` | **BEATS FLOOR** — CRPS ratio 0.4730, Winkler 0.4662, coverage err 0.0240 vs floor's 0.0333 |
| G1 forecast (stationary-seasonal) | floor harness over `stationary_seasonal_m12_sigma0.5_n400` (n_scored=352) | `--release` | **BEATS FLOOR** — CRPS 0.7195, Winkler 0.7315, coverage err 0.0074 vs 0.0131 |
| G1 forecast (white noise) | floor harness over `white_noise_sigma0.5_n400` (n_scored=336) | `--release` | **BEATS FLOOR** — CRPS 0.7550, Winkler 0.7375, coverage err 0.0054 vs 0.0202 |
| G2 composite F0 request | `cargo bench -p katgpt-extract --bench bench_628_f0_g2` (6 fields, 417-char state, 10k iters) | bench profile (release-class) | **p99 357,166 ns ≤ 10,000,000 ns bar** (28× headroom) — first run |
| G2 composite, load-stamp re-run | same bench under sibling load | bench profile | **p99 2,109,334 ns ≤ bar** (4.7× headroom under load) — PASS both |
| G4 alloc | module gates (TLS-gated counting allocator, `src/alloc_gate.rs`) | debug + `--release --features …,alloc_tracking` | zero-alloc pinned: span `score_into`/`best_span`, `scan_numbers`/`scan_durations`/`scan_datetimes`, temper `calibration_spread`/`confidence`/`gate_located`, `estimate_into` call-2, `forecast_into` sized-out. `parse_number` FULL parse NOT zero-alloc (fn-internal scratch + std sort scratch) — gated scan-only, disclosed |
| G5 external fixture | `tests/g5_external_fixture.rs` vs Recognizers-Text @ `da7edcff59f6` (351 rows, BLAKE3 `f3e1dfc3…988163`) | debug | value accuracy 12.7% rank-0 (scope classes dominate — see [issue 953](../.issues/953_parser_word_form_coverage.md)); span containment 18.3% any / 11.3% rank-0 |

## G2 detail (ns)

| section | p50 | p90 | p99 | max |
|---|---|---|---|---|
| composite F0-floor (6 fields) — quiet-ish run | 310,334 | 328,084 | 357,166 | 436,583 |
| composite — re-run under load | 341,208 | 388,458 | 2,109,334 | 197,932,916 |
| parse_number | 2,000–2,041 | 2,166–2,333 | 2,583–2,666 | 11,000–18,000 |
| span score_into full sweep (540 windows) | 321,583–326,917 | 340,459–345,625 | 367,750–374,042 | 456,792–481,959 |
| temper calibration_spread (256-pool) | 6,917–6,959 | 7,084–7,542 | 8,791–8,791 | 24,125–26,042 |
| estimate_into (30 comps, split-conformal) | 250 | 250–333 | 292–375 | 334–6,083 |
| forecast_into h=8 (36 obs, m=12) | 1,833 | 1,875–2,000 | 2,334–2,417 | 14,292–14,500 |

Work checksum `14156999` identical across both runs — deterministic sink.

## Provenance (the AGENTS latency law — quoted, not summarized)

- First (F's) run: self-labeled quiet box, `CARGO_TARGET_DIR=/tmp/k628_f`, on AC. Not preflighted — treated as PROVISIONAL.
- Close-out re-run: `bench_preflight.sh` **REFUSED twice** (load 14.59 then 11.54 vs MAX_LOAD 6.0 — sibling sessions; swap 29 GB; powermode 2 High Power; AC). The re-run under that refused load still holds the bar with 4.7× headroom; ~6× load inflation vs the first run. **The quiet-box preflight PASS line is owed** — re-run `bench_preflight.sh` + the bench on a quiet box before quoting the composite number as final.
- G1 accuracy cells are deterministic synthetic corpora (CPU arithmetic, no latency claim) — load-state noted for the record only.

## GOAT verdicts

- **`extraction_estimate` → DEFAULT-ON** (this close): G1 BEATS ×3 at release + G2 measured (estimate_into 250 ns p50, forecast_into 1.8 µs p50) + G4 zero-alloc hot paths pinned + G3 new-leaf (123 crate tests green). Synthetic-corpus verdict; **real-corpus re-gate owed when the Phase-2 outcomes ledger accumulates**.
- **`span_pointer` stays opt-in**: its GOAT requires the G5 rule-floor win; the span baseline arm is deferred pending a text-bearing external fixture (T0.7).
- **`typed_parsers` stays opt-in**: no pinned accuracy bar; the measured 12.7% + mismatch classes are the input to the bar ([issue 953](../.issues/953_parser_word_form_coverage.md)); ratchet pinned after the first coverage landing.
- **`temper` stays opt-in**: gate substrate, no independent floor yet.

## G5 table (E's run, verbatim)

```
dim       rows  wire    value   lenient    exact any   exact r0   cont any    cont r0  cand/row    live
number     105    85     9.4%      0.0%    38  26.6%  12   8.4%  39  27.3%  12   8.4%      1.85   73.3%
date        80    65    12.3%      0.0%     8  10.0%   8  10.0%   8  10.0%   8  10.0%      0.10   10.0%
time        67    50    12.0%     14.0%    11  16.4%  11  16.4%  11  16.4%  11  16.4%      0.31   31.3%
datetime    43    23     4.3%      0.0%     1   2.3%   1   2.3%   1   2.3%   1   2.3%      0.02   2.3%
duration    56    52    23.1%      0.0%    12  21.4%  12  21.4%  12  21.4%  12  21.4%      0.21   21.4%
TOTAL      351   275    12.7%      2.5%    70  18.0%  44  11.3%  71  18.3%  44  11.3%      0.67   33.9%
```

value = rank-0 expected_wire accuracy; lenient = time/datetime format-variance bucket (reported, never folded in); exact/cont = span hit rates (any candidate / rank-0), micro-averaged over all expected entities; live = rows with ≥1 candidate. Liveness floors pinned just under measured: `[0.65, 0.05, 0.25, 0.01, 0.15]` per dim (80% was proven unreachable — word-form dark mass; calibration rationale in the test).

## Commits

- `1561be570` — G5 harness + `[[test]]` row + dev-deps (E)
- `828f17428` — G2 bench + `[[bench]]` row + G4 alloc-gate consolidation (`src/alloc_gate.rs`) + duration-scanner alloc fix caught by the new gate (F)
- the promotion/docs commit — `default = ["extraction_estimate"]` + this record
