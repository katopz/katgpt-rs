# Issue 001: Sink-Aware Attention G3 Latency — bring 1671% → ≤5%

**Filed:** 2026-06-18
**Source:** `.benchmarks/059_sink_aware_goat.md` G3 row
**Plan:** [287_sink_aware_attention](../.plans/287_sink_aware_attention.md)
**Status:** RESOLVED (partial). Per-call path structurally infeasible; cached variant ships as the production answer and meets the 5% target in steady state.

---

## Problem

`SinkAwarePolicy::DualPolicy` adds 1671% latency overhead at n=128, d_h=64
versus `SinkAwarePolicy::Uniform` (single n·d copy). Target: ≤5%. Missing by
~3 orders of magnitude.

Raw numbers from `sink_aware_latency_bench`:

| n    | d_h | uniform_us | dual_us | overhead% | kind       |
|------|-----|-----------:|--------:|----------:|------------|
| 128  | 64  | 0.71       | 12.54   | 1671%     | Broadcast  |
| 512  | 64  | 2.96       | 158.75  | 5266%     | Broadcast  |

---

## Root causes (in order of impact)

1. **`apply_dual_policy_gate` always computes `stable_rank_update_into`** (line
   555 `Some(o)` is unconditionally passed) even when `value_norm_ratio` alone
   would classify the head as NOP. NOP heads do not need stable rank.

2. **Two per-call allocations**:
   - `stable_rank_update_into` line 245: `let mut ov_buf = vec![0.0f32; n];`
   - `apply_dual_policy_gate` line 535: `let mut col_sums = vec![0.0f32; n];`

3. **`Vec<Vec<f32>>` layout** for `o`, `values`, `attn` defeats cross-row SIMD.
   Each `simd_dot_f32(row, v, d)` follows a pointer to a heap-allocated row.

4. **`classify_sink_at` always scans `values` row-by-row** for `mean_norm`
   even when the sink position's own `‖v_s‖` already proves NOP. Cheap but
   visible on micro-bench scale.

---

## Tasks

- [x] T1: Extend `StableRankScratch` to carry 4 buffers (`v`, `w`, `ov_buf`,
      `col_sums`). Existing `v`/`w` semantics preserved. New buffers `pub`.
      `ensure_capacity(d)` retained for back-compat; `ensure_capacity_dn(d, n)` added.
- [x] T2: NOP fast-path in `classify_sink_at` — `stable_rank_reachable` gate
      skips power iteration unless `value_norm_ratio` is in Broadcast window.
- [x] T3: `col_sums` allocation eliminated in `apply_dual_policy_gate` and
      `classify_all_sinks` — both reuse `scratch.col_sums`.
- [x] T4: `ov_buf` allocation eliminated in `stable_rank_update_into` —
      reuses `scratch.ov_buf`.
- [x] T5 (revised): cheap rank-1 cosine probe in `stable_rank_update_into`.
      Compares `O[0]` vs `O[n-1]` (3 SIMD dots, O(d) work); if cosine > 0.95,
      returns 1.0 immediately, skipping the O(n·d) power iteration. Drops
      `classify_sink_at` rank-1 case from 3.125µs → 0.625µs at n=128, d=64.
- [x] T5b (added): `apply_dual_policy_gate_cached` + `CachedSinkClassification`.
      Production-realistic path that amortizes the classifier over
      `audit_every_n` calls (default 16). Steady-state overhead measured at
      ≤5% (often beats Uniform due to simpler code path).
- [x] T6: Re-ran `sink_aware_latency_bench`; numbers recorded in
      `.benchmarks/059_sink_aware_goat.md` G3 row.
- [x] T7: **Verdict — per-call path structurally misses 5% target** (memory
      bandwidth bound). Cached cadence-16 variant hits the target in steady
      state. Per AGENTS.md "demote loser" rule: `sink_aware_attn` stays
      opt-in; `apply_dual_policy_gate_cached` is the documented production path.
      No new plan filed — the latency issue is closed with the cached variant
      as the answer. A forward-path wiring plan would be a separate concern.

## Acceptance criteria

- `cargo test --features sink_aware_attn --test sink_aware_g2_synthetic` → 2/2 PASS ✅
- `cargo test --features data_probe -p katgpt-rs --lib data_probe::` → 54/54 PASS (was 52; +2 cached-variant tests) ✅
- `cargo bench --features sink_aware_attn --bench sink_aware_latency_bench` →
  - per-call (`dual_us`): **STRUCTURAL MISS** — 1000–3000% overhead.
    Memory-bandwidth bound (classifier reads attn (n²) + values (n·d); Uniform
    is a single n·d copy). Cannot be made ≤5% without skipping classification.
  - cached cadence-16 (`cached_us`): **PASS** — steady-state overhead ≤5%
    (often negative, i.e. cached is faster than Uniform due to simpler code path).

## GOAT gate

**Outcome: partial promote.** Per-call `DualPolicy` stays opt-in (demoted).
`apply_dual_policy_gate_cached` is the production path and meets the 5% target.
The `sink_aware_attn` feature remains opt-in (not added to default features)
until a real-ViT G2 gate passes — that's the remaining blocker, not latency.

---

## Related

- Plan 287: `.plans/287_sink_aware_attention.md` (original)
- Research 258: `.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md`
- Bench 059: `.benchmarks/059_sink_aware_goat.md`
- Source: `crates/katgpt-core/src/data_probe.rs`
