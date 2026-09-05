# Issue 727 — SP-KV misses BOTH T16 bars once the gate is measured at a realistic sequence length

**Status:** OPEN — filed 2026-09-05 by Issue 723 T7. `sp_kv` is opt-in
(not in `default`), so nothing ships on these numbers. The gate
(`tests/bench_sp_kv.rs::bench_gate_bias_overhead`) is `#[ignore]`d with the
provenance below — the T8 `goat_169_g1` precedent — so the assertions stay
executable via `--ignored` and are not laundered into a re-pinned bar.

## What the gate could never have measured

`bench_gate_bias_overhead` took its sequence length from `Config::micro()`,
whose `block_size` is **16**. Two consequences, neither visible in the output:

1. **The "50% pruned" arm pruned nothing.** Its guard is
   `if t < t_n - 16 && t.is_multiple_of(2)`, which at `t_n = 16` is `t < 0` —
   vacuously false. Measured: **0 of 16** entries set to `-inf`. So
   `assert!(prune_skip_speedup > 1.05)` was asserting that identical work is 5%
   faster than itself. It could not have passed for the reason it names.
2. **The workload was small enough that code layout decided the answer.**
   4 heads x 16 positions x head_dim 4 is ~256 MACs/iter. Adding a single
   unrelated `eprintln!` elsewhere in the same function moved the gated-zero
   ratio from **1.036 to 0.707** and the mixed ratio from **1.489 to 1.027**.
   A 3% bar cannot be read off an instrument an unrelated edit moves by 30%.

Both are now fixed in the test: the sequence length is decoupled from
`block_size` (`T_N = 512`, giving a real 48% prune), the two asserted
comparisons are interleaved chunk-by-chunk against the same baseline with a
median across chunks, and a `pruned` assertion makes a vacuous mixed arm a
loud FAIL instead of a silent pass.

## What the repaired instrument measures

M3 Max, release, three runs, `t_n = 512`, 48% of positions pruned:

| quantity | bar | measured | per-round spread |
|---|---|---|---|
| gate-bias dispatch overhead | < 3% (paper target < 1%) | **+8.0 / +8.1 / +8.4%** | 1.026 – 1.126 |
| prune-skip speedup | > 1.05x | **1.046 / 1.042 / 1.015x** | 0.929 – 1.000 |

Stable across runs; neither is the box.

## Mechanism, as far as it is understood

- The +8% is **not dispatch**. `GateBias` monomorphizes, and the legacy
  Option-dispatch wrapper measures the same **+8.2%** — so the two dispatch
  strategies are equal and the cost is elsewhere. It is the bias **load**: one
  extra `get_unchecked(t)` per position per head, 512 extra loads against a
  512x4 attention. At `t_n = 16` that was under the layout noise, which is
  exactly how a "<1% overhead" claim survived unexecuted.
- Prune-skip elides the value accumulation for pruned positions but **not** the
  score dot, so 48% pruned buys at most ~24% of the work; 4% is what survives
  the branch.

## Tasks

- [ ] T1 — Decide whether "zero-overhead gate bias" is a claim SP-KV still
      makes. If the bias load is inherent, the paper target is wrong and should
      be restated as a measured budget with the load counted.
- [ ] T2 — Try hoisting the bias probe out of the position loop (a
      precomputed active-position list, so pruned positions cost neither a load
      nor a score dot). That is the change that would move both numbers at once.
- [ ] T3 — Re-measure at more than one `t_n` (128 / 512 / 2048): the overhead
      is a per-position load against a per-position dot, so the ratio should be
      roughly scale-invariant and a deviation is itself a finding.
- [ ] T4 — Only after T1–T3: either un-`#[ignore]` at bars the primitive meets
      (with provenance), or record SP-KV's gate-bias path as a known cost and
      say so in the feature's Cargo.toml comment.

## Gates

| Gate | Criterion |
|---|---|
| G1 | The mixed arm prunes 33–66% of positions (already asserted in-test) |
| G2 | Gate-bias overhead and prune-skip speedup measured at three `t_n` values, three runs each, per-round spread reported |
| G3 | Any re-pin carries the measurement, not a nudge — the same rule Issue 723 T7 applied to `bench_257` and `bench_270` |

## References

- `.issues/723` T7 — the wall-clock-gate treatment that produced these numbers
- `tests/bench_sp_kv.rs::bench_gate_bias_overhead` — the repaired instrument
  and its `#[ignore]` provenance
- `.issues/723` T8 `goat_169_g1` — the precedent for a deliberate,
  provenance-carrying `#[ignore]` over a re-pin
