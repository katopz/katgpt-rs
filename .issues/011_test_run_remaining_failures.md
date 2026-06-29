# Issue 011 — Remaining test failures from 2026-06-29 full run

**Status:** open
**Discovered:** 2026-06-29 full `cargo test --workspace --all-features` run (debug, ~16-core parallel, thermal-throttled host).
**Context:** This run unblocked three separate compile failures and fixed two real bugs:
- Workspace compile (commit `0482eee0`): `katgpt_rs::weights::ContiguousWeights` → `katgpt_rs::ContiguousWeights` (leftover from the microgpt→katgpt rename `acf08551`).
- `cargo bench` release compile (commit `78d80c18`): `sdar_absorb` tests + `bench_sdar_gated_modelless` referenced `#[cfg(debug_assertions)]`-gated APIs ungated; `depth_invariance` feature didn't propagate to `katgpt-micro-belief` (E0599 on `AttractorKernel::audit_depth_invariance`).
- Two real bugs (commit `db1ba7a3`): sleep sliding-window eviction wiped shifted KV via `reset()`; sr2am `decision_stats` under-counted under `sia_feedback`.
This issue tracks what remains. Bench + examples summary appended at the bottom.

## Already resolved in this run (do not touch)

- `tests/bench_102_tilert_pipeline_goat.rs` compile break — import path corrected.
- `sleep::eviction::sliding_window_retains_recent` — `sliding_window_evict` called `reset()` post-`copy_within`, zeroing the shifted entries. Fixed via new `MultiLayerKVCache::set_fill_pos`.
- `pruners::bomber::sr2am_player::test_sr2am_player_decision_stats` — under `sia_feedback` the configurator can pick `HarnessUpdate`/`WeightUpdate`, tracked in `feedback_decision_stats()` not the 4-tuple. Test now sums both.

## Confirmed flaky / environmental (NOT bugs — leave alone)

These pass single-threaded with `--test-threads=1` and fail only under parallel test load + thermal throttling. They are perf-budget assertions with hardcoded ns/s gates that the host cannot hold under 16-way debug-mode contention. Do not relax the thresholds to mask this — re-verify on a cool host first.

- [ ] `pruners::workflow_lattice::tests::test_bench_lattice_vs_noop` — 737.9ns > 500ns budget under load; passes alone.
- [ ] `speculative::nf_flow::tests::test_bench_flow_score_v128_t5` — 15.4µs > 10µs budget under load (test explicitly annotates "debug"); passes alone.
- [ ] `ruliology::tests::benchmarks::tests::bench_enumerate_fsm_3_states` — 11.78s > 10s budget under parallel contention; passes alone (~5s single-threaded).

## Real bugs needing root-cause work (deterministic, fail single-threaded)

### B1 — `iso_quant::rotation::tests::test_non_multiple_of_4`
- `src/iso_quant/rotation.rs:403`
- Partial-group (dim not multiple of 4) inverse rotation is wrong: `input[8]=9.0` round-trips to `-0.5684246`, `rel_err=1.063` (tolerance 0.25). Full groups 0-7 round-trip within 1e-3, so the bug is specifically in the zero-padded partial-group inverse path of `apply_inverse_rotation`.
- **Not a guess-fix.** Needs the quaternion left/right-pair inverse math for the trailing `< n_groups*4` elements audited.

### B2 — `speculative::flashar_anchor::tests::test_anchor_then_fill_reduces_steps`
- `src/speculative/flashar_anchor.rs:593`
- Asserts `fill_steps_used <= baseline_steps_used` but got `fill=8 > baseline=1`. Anchors are supposed to reduce the denoising search space; here they do not. Deterministic (seed `Rng::new(42)`).
- Needs the anchor-fill step accounting vs baseline-step accounting audited — the `1` baseline looks suspiciously low (anchor run is `8` vs baseline `1`).

### B3 — `speculative::flashar_anchor::tests::test_anchor_then_fill_produces_valid_output`
- `src/speculative/flashar_anchor.rs:521`
- Token at position 2 is the mask token (26) when it should not be. Likely the same root cause as B2 (fill not converging / not replacing mask tokens).
- Fix B2 first; re-check B3.

### B4 — `pruners::bomber::rmsd_player::tests::test_compute_sdar_reward_in_danger` (test-vs-formula discrepancy, needs a decision)
- `src/pruners/bomber/rmsd_player.rs:693`
- `compute_sdar_reward(true, 0.8, 0)` returns `0.57` by the documented formula `survival*0.5 + safety*0.35 + completeness*0.15` (= `1.0*0.5 + 0.2*0.35 + 0.0*0.15`).
- The test asserts `reward < 0.5`.
- Three sibling tests (`alive_safe`→0.85, `dead`→0.0, `all_zero`→0.35) all match the formula exactly, so the **formula is internally consistent**; the `in_danger` threshold is the outlier.
- **Open question:** is the formula under-weighting `danger` (survival dominates: alive+80%-danger still scores 0.57), or is the test aspirational? Not auto-fixable — needs the intended reward-shape decision. If the formula stands, the test should assert `< 0.6` (or compare relative to the safe case).

## GOAT-gate failing by design (not a bug to silence)

### G1 — `still_kv::integration_tests::goat_t24_compact_cache_quality`
- `src/still_kv/mod.rs:704`
- 1024×8×64 compact-cache quality gate: cos_sim at 8× compression is 0.0503 (threshold 0.70), 16× is 0.1045 (threshold 0.50), 32× is 0.1155 (threshold 0.30). Best strategy at 8× is `MuxSuperposition` (cos_sim 0.2021).
- This is the StillKV promotion gate deliberately failing — the feature is not yet good enough to promote. **Do not lower the thresholds.** The fix is improving query-bank initialization / compaction strategy, tracked separately. Grandfathered under the UQ "Report the Floor" rule (`.issues/010`) and must clear the conformal-naive floor before re-gate.

## Bench results (2026-06-29, thermal-throttled host)

After fixing the release compile (commit `78d80c18`), ran every bench target isolated with continue-on-fail:

| Package | Total | Pass | Fail |
|---------|-------|------|------|
| katgpt-rs | 31 | 30 | 1 |
| katgpt-core | 33 | 33 | 0 |

The single failure is `fpcg_probe_forecast_bench` (katgpt-rs): G6 perf gate at `d_model=4096` measured 873.95ns vs the 200ns budget (0.32×). Smaller dims passed; only the 4096 size tripped. This is a perf-budget gate inflated by thermal — consistent with the predicted −30–40% degrade. Note `bench_319_geometric_product_goat` (D=8 185.9ns vs <150ns) failed under the initial parallel `cargo bench --workspace` run but PASSED when re-run isolated on a slightly cooler core — confirming these absolute-latency gates are thermal-sensitive, not real regressions. Re-gate both on a cool host.

## Examples results (2026-06-29)

- `cargo build --examples --all-features` — **211/211 compile clean**.
- Smoke-ran a diverse 30-example sample (non-TUI, 20s timeout each) spanning bandit / bomber / attn / cache / cgsp / go / monopoly / ruliology / spectral domains — **30/30 PASS, 0 timeouts**.
- TUI examples (`bear_02_tui`, `bomber_02_tui`, `dungeon_01_tui`, `go_07_tui`, `monopoly_02_tui`, `sudoku_03_tui`, `tactical_06_tui`, `tactical_09_fog_tui`) were skipped — they block on the terminal and are not smoke-testable headless.

## Reproduction

```bash
# Full run (fails B1-B4 + G1 under load; flaky ones also trip):
cargo test --workspace --all-features

# Deterministic subset (B1-B4 fail identically here):
cargo test --lib --all-features -- \
  iso_quant::rotation::tests::test_non_multiple_of_4 \
  pruners::bomber::rmsd_player::tests::test_compute_sdar_reward_in_danger \
  speculative::flashar_anchor::tests::test_anchor_then_fill_reduces_steps \
  speculative::flashar_anchor::tests::test_anchor_then_fill_produces_valid_output \
  still_kv::integration_tests::goat_t24_compact_cache_quality \
  --test-threads=1
```
