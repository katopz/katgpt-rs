# Issue 894 — KVarN 4-bit value dequant is a scalar loop (a zip rewrite measured 3.2× faster)

**Status:** OPEN — found 2026-09-25 while re-attempting Issue 883 P1 G2 ([Bench 895 addendum](../.benchmarks/895_fitted_value_table_primitives_goat.md#addendum--2026-09-25-the-dequant-fold-lever-p1-g2-re-attempt)). This is a working-tree probe only; nothing has landed.

## Finding

`KVarNKVCache::dequantize_value_into` at 4 bits (`crates/katgpt-kv/src/kvarn/kv_cache.rs`, the `4 =>` arm) indexes `out[2*i]`, `out[2*i+1]`, `s_col[2*i]`, … inside `for i in 0..full_pairs`. None of those slice lengths is pinned, so every access keeps a bounds check and the loop runs scalar, at about 0.75 ns per element on an M3 Max.

The probe rewrote the full-pair loop as a zip, keeping the per-element ops and their order the same:

```rust
for ((o, &b), sc) in out[..2 * full_pairs].chunks_exact_mut(2)
    .zip(&packed_row[..full_pairs])
    .zip(s_col[..2 * full_pairs].chunks_exact(2))
{
    o[0] = ((b & 0x0F) as f32).mul_add(rtn_scale, rtn_zp_val) * sc[0] * var_row;
    o[1] = ((b >> 4) as f32).mul_add(rtn_scale, rtn_zp_val) * sc[1] * var_row;
}
```

The workload was Bench 895's G2 plain arm: T=4096, kv_dim 128, 4-bit, dequant + axpy, `--release`. Box state: M3 Max, AC, powermode 2, load 35–38, 0.9–1.2 GiB free, swap 1078/2048 MiB. The plain pass took **156–162 µs, against 494–531 µs** for the shipped loop at similar load: **3.2×**, in 3 runs.

## Tasks

- [ ] **T1 — bit-identity old vs new.** Dump the dequantized rows from the shipped loop and the rewrite across bits × var-norm × partial tiles, and require 0 differing bits. The rewrite is bit-identical by construction (no reassociation, `mul_add` in both), but only a same-kernel comparison has been run so far (Bench 895 G3b).
- [ ] **T2 — the same rewrite for the other arms.** That covers the 2-bit (grouped and var-norm), 8-bit, and key-dequant (`dequantize_key_into`) loops, which use the same indexed shape.
- [ ] **T3 — GOAT gate.** A paired `ab_median_ratio` over the plain dequant+axpy pass, in `--release`, with box state recorded, for 3 runs. G3 is T1. G4 is 0 allocs. Promote on a pass: it is a pure kernel change with no feature flag of its own, so it rides `kvarn`.
- [ ] **T4 — note the interaction with Issue 883 P1 G2.** A faster denominator makes every per-position cost of the mean restore a larger share of the ratio. In the probe, the per-element fold cost +28% and the deferred restore +3.1–3.3%. Re-measure P1 G2 on the new kernel once this lands; the bar is not loosened.
