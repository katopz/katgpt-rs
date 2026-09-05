# Issue 727 — SP-KV misses BOTH T16 bars once the gate is measured at a realistic sequence length

**Status:** RESOLVED 2026-09-05 (same-day T1–T4; worktree-only, uncommitted).
Verdict: **T2 hoist LANDED and is bit-identical (pinned)**; **prune-skip now PASSES its bar
(1.12–1.58x vs >1.05x)**; **gate-bias overhead does NOT pass and CANNOT** — it is an inherent
per-position gate read, restated as a measured budget (~+7–12% vs NoBias at hd=4, roughly
scale-invariant in t_n); the paper's <1% was a t_n=16 artifact. Per T4's second branch the gate
stays `#[ignore]`d with an updated reason and the known cost is recorded in the `sp_kv` feature
comment (crates/katgpt-kv/Cargo.toml + the root forward). `sp_kv` remains opt-in — nothing ships
on these numbers.

Filed 2026-09-05 by Issue 723 T7. The gate
(`tests/bench_sp_kv.rs::bench_gate_bias_overhead`) was `#[ignore]`d with the
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

**Follow-up, recorded-not-built (T2):** the ~+3pp the hoist adds to the
all-active case is the per-call 64-chunk scan. A provider-level active list
(`GateBiasBuffer` building `(position, bias)` pairs inside `build_gate_biases`,
which already walks every position) would amortize the scan out of the attention
call entirely — an API addition to the `BiasProvider` surface, worth trying only
if a production consumer needs the all-active case to approach `NoBias`.

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

- [x] T1 — Decide whether "zero-overhead gate bias" is a claim SP-KV still
      makes. If the bias load is inherent, the paper target is wrong and should
      be restated as a measured budget with the load counted.
      **DONE (2026-09-05): the claim is FALSE.** Any gated attention must read
      the gate once per position per head; the measured budget is ~+7–12% vs
      `NoBias` at hd=4 (old interleaved-load structure +7.1–9.3%; hoisted
      structure +11.1–11.8% at the load floor — the scan adds ~3pp to the
      all-active case). Restated in the bench provenance, the primitive's doc
      comment (`attention_head_core`), and both `sp_kv` feature comments.
      `NoBias` / the `None` wrapper arm remains exactly 0%.
- [x] T2 — Try hoisting the bias probe out of the position loop (a
      precomputed active-position list, so pruned positions cost neither a load
      nor a score dot). That is the change that would move both numbers at once.
      **DONE (2026-09-05): LANDED.** `attention_head_core` now dispatches to two
      `#[inline(never)]` impls: the NoBias path verbatim (pre-hoist machine
      code), and a hoisted GateBias path that scans the bias slice in
      64-position chunks (stack `(u32, f32)` pairs — zero alloc, no
      thread-local), runs the dot loop over active pairs only, and skips
      pruned/underflowed exact-zero entries in the exp and value-accumulate
      passes. Bit-identity: removing exact `+0.0`/`±0.0` contributions from a
      fixed ascending-t accumulation is a bit-identical no-op (sum/output
      can never be `-0.0`); `max_score` sees the same scores in the same order.
      Two DOCUMENTED divergences, both unreachable through `build_gate_biases`:
      an all-`-inf` bias (old: NaN output; new: zeros) and non-finite cache
      values at zero-weight positions. Pinned by the new
      `gate_bias_hoist_bit_identity` test (to_bits level, 6 bias cases x 2
      head-offset arms + underflow + the documented all-pruned divergence) —
      plus the primitive's own tests (3/3), the full bench_sp_kv suite (6/0/1i),
      and bench_sp_kv_quant (8/0).
      First draft kept both bodies in one `#[inline(always)]` fn and the bench
      caught it: the NoBias baseline absorbed a 1.66x layout penalty (1.79 →
      2.98 µs/iter in the SAME load window — load moves both arms of a ratio,
      a layout edit moves one), reading as a bogus −25% gated-zero overhead.
      The split into `#[inline(never)]` impls restored the baseline (2.03 vs
      2.06 µs/iter matched-window) — the fix is recorded at the dispatcher.
- [x] T3 — Re-measure at more than one `t_n` (128 / 512 / 2048): the overhead
      is a per-position load against a per-position dot, so the ratio should be
      roughly scale-invariant and a deviation is itself a finding.
      **DONE (2026-09-05): roughly scale-invariant, as predicted.** Full table
      in the bench doc comment (3 t_n x 3 runs, per-round spread printed per
      cell by the gate itself). Before: +7.1–9.3% across all three t_n; the
      512-cell 0.594x prune outlier is a scheduling spike the per-round spread
      exposes. After: +11.1–11.8% at the load floor; two 2048 cells read
      +24/+27% in a load-climb window (1x → 13x during the sweep — sibling
      benchmark session) and are flagged as such, not averaged in.
- [x] T4 — Only after T1–T3: either un-`#[ignore]` at bars the primitive meets
      (with provenance), or record SP-KV's gate-bias path as a known cost and
      say so in the feature's Cargo.toml comment.
      **DONE (2026-09-05): second branch — known cost recorded, `#[ignore]`
      KEPT with an updated reason.** The primitive meets the prune-skip bar
      (1.12–1.58x) but not the gate-bias bar (<3%); the gate asserts BOTH, so
      un-ignoring would ship a permanently-red gate. The `#[ignore]` reason now
      carries the measured budget; the cost is recorded in
      `crates/katgpt-kv/Cargo.toml` (where `sp_kv` is defined) + the root
      forward comment. No bar was re-pinned (G3).

## Gates

| Gate | Criterion | Result |
|---|---|---|
| G1 | The mixed arm prunes 33–66% of positions (already asserted in-test) | PASS — 56/128 = 43.8%, 248/512 = 48.4%, 1016/2048 = 49.6% at the three t_n |
| G2 | Gate-bias overhead and prune-skip speedup measured at three `t_n` values, three runs each, per-round spread reported | DONE — tables in the bench doc comment; the gate prints every cell with its per-round range before asserting |
| G3 | Any re-pin carries the measurement, not a nudge — the same rule Issue 723 T7 applied to `bench_257` and `bench_270` | HELD — no bar re-pinned; the <3%/1.05x asserts stand unchanged; the measured budget is recorded as provenance instead |

## Files touched (worktree-only, uncommitted)

- `src/sp_kv_forward_mod.rs` — the T2 hoist (dispatcher + two `#[inline(never)]` impls) + numerics contract docs
- `tests/bench_sp_kv.rs` — T3 three-t_n sweep, measured-budget provenance, updated `#[ignore]` reason, new `gate_bias_hoist_bit_identity` test
- `crates/katgpt-kv/Cargo.toml` — `sp_kv` feature comment (measured cost, T4)
- `Cargo.toml` — root `sp_kv` forward comment (T4)
- this file

## References

- `.issues/723` T7 — the wall-clock-gate treatment that produced these numbers
- `tests/bench_sp_kv.rs::bench_gate_bias_overhead` — the repaired instrument
  and its `#[ignore]` provenance
- `.issues/723` T8 `goat_169_g1` — the precedent for a deliberate,
  provenance-carrying `#[ignore]` over a re-pin
