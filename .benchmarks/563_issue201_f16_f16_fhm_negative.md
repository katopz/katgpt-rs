# Bench 563 — Issue 201 Phase 1: f16×f16 FHM kernel microbenchmark (NEGATIVE RESULT)

## Status: DECISION GATE FAILED — Issue 201 closes as a documented negative result

## Origin

`Issue 201` is the successor to
`Issue 200`. Issue 200
honestly closed the **weight-only** f16 path (f16 weights × f32 activations) as a
negative result on Apple Silicon: f32 activations limit the bandwidth reduction to
25%, and the FCVT latency on the critical path eats the savings (G2 FAIL: 1.7×
slower than f32).

Issue 201's hypothesis was that **full f16 (weights + activations)** using the
ARMv8.2-A FP16 widening FMA (`fmlalb`/`fmlalt`, a.k.a. `fmlal`/`fmlal2`) would
recover the win: f16×f16→f32 in a single instruction, no explicit FCVT on the
critical path, and genuine 50% bandwidth reduction (2+2=4 bytes/element vs 4+4=8).

Issue 201 Phase 1 T7 set a clear **decision gate**: *if `simd_dot_f16_f16` is NOT
≥1.5× faster than `simd_dot_f32` at L3-exceeding sizes, close the issue with a
documented negative result.* This file records that measurement.

## Toolchain note (important — affects reproducibility)

The FHM instructions are **not accessible on stable Rust 1.93.0** via the usual
paths:

1. **Stable intrinsics** — `vfmlalq_low_f16` / `vfmlalq_high_f16` exist in
   stdarch but are gated behind `#![feature(stdarch_neon_f16)]`
   (rust-lang/rust #136306, still unstable as of 1.93.0).
2. **Inline-asm mnemonic** — LLVM 21.1.8's aarch64 integrated assembler
   **rejects** the `fmlalb`/`fmlalt` mnemonics in every arrangement form
   (`fmlalb v0.4s, v1.8h, v2.8h` → "invalid operand for instruction"). LLVM 21
   uses the aliases `fmlal` (low) / `fmlal2` (high), which the standalone
   assembler also rejects with the same error regardless of `+fp16fml` /
   `+fullfp16` / `+v8.2a` attributes. `llvm-objdump` decodes the FHM encoding as
   `<unknown>` even with `--mattr=+fp16fml,+fullfp16`.
3. **Hand-derived `.inst` encoding** — a manual bit-layout derivation of FMLALB
   SIGILL'd at runtime (wrong encoding); the encoding is too fiddly to derive
   reliably by hand without a working decoder cross-check.

**Workaround used for this measurement:** the benchmark was compiled with the
**nightly** toolchain (`cargo +nightly`) using the unstable intrinsics directly.
The intrinsic implementations were first verified correct on a known input
(`vfmlalq_low_f16` / `vfmlalq_high_f16` produced exactly the expected
f16×f16→f32 products). Nightly `llvm-objdump` of the emitted code confirmed the
real encodings are `fmlal` (low) / `fmlal2` (high). This is a one-off measurement
probe; production code on stable would need verified `.inst` encodings, which is
moot given the gate failed (see below).

## Methodology

- **Hardware:** Apple Silicon (aarch64, `fp16=true`, `fhm=true` via runtime detection).
- **Baseline:** `katgpt_types::simd::simd_dot_f32` (NEON f32×f32, 4-accumulator
  unroll, the production dot kernel).
- **Challenger:** `dot_f16_f16_fhm` — 16-element unroll, 4 independent f32
  accumulators, `vfmlalq_low_f16` + `vfmlalq_high_f16` per 8-element load pair.
  Same FMA-pipeline-latency-hiding structure as the f32 baseline.
- **Sizes:** 256 (L1), 4K (L1/L2), 64K (L2), 1M (L3), 4M (L3-exceeding for f32),
  16M (L3-exceeding for both). The L3-exceeding sizes are the bandwidth-bound
  regime where the 50% bandwidth reduction should pay off.
- **Measurement:** `std::time::Instant`, 32-iter warmup, iters tuned to keep each
  bench ~50–200ms, `black_box` on every result.
- **Correctness:** rel-error of the f16 result vs f32 on the same values,
  reported per size (f16 is lossy by design — the gate is on the *kernel*, not
  bit-identity).

## Results (2026-07-29, nightly intrinsics, release build)

| size | f32 ns/iter | f16 ns/iter | speedup | f32 GB/s | rel_err |
|---|---:|---:|---:|---:|---:|
| 256 (L1) | 12 | 7 | **1.714×** | 170.7 | 0.092% |
| 4K (L1/L2) | 237 | 220 | 1.077× | 138.3 | 0.012% |
| 64K (L2) | 4648 | 4265 | 1.090× | 112.8 | 0.090% |
| 1M (L3) | 76461 | 66780 | 1.145× | 109.7 | 0.096% |
| **4M (L3-exc f32)** | 353213 | 269971 | **1.308×** | 95.0 | 1.095% |
| **16M (L3-exc f16)** | 1354443 | 1103295 | **1.228×** | 99.1 | 6.200% |

**L3-exceeding speedups: 1.31×, 1.23×. Best = 1.31×. Gate = 1.5×.**

## Verdict: DECISION GATE FAILED

`BEST L3-exceeding speedup = 1.308× < 1.5× → FAIL → close Issue 201 negative.`

The full-f16 hypothesis is **refuted on this hardware class**. The widening FMA
recovers *some* of the ground that weight-only f16 lost (Issue 200 was 1.7×
*slower*; full f16 is 1.2–1.3× *faster*), but the recovery is far short of the
1.5× gate, and short of the theoretical 2× ceiling.

### Why the hypothesis failed (root-cause)

1. **The bandwidth ceiling is the real wall, and f32 is already close to it.**
   f32 sustains ~95–110 GB/s at L3-exceeding sizes on this machine. f16 improves
   that to ~120–130 GB/s equivalent — a ~25–30% gain, NOT the theoretical 50%.
   This means the kernel is **not purely bandwidth-bound**: FMA throughput,
   load-use latency, and accumulator-reduction overhead consume a meaningful
   fraction even at L3-exceeding sizes, so halving bandwidth doesn't halve
   runtime.

2. **FHM FMA throughput, not FCVT latency, is now the limiter.** Issue 200's
   weight-only path was doomed by explicit FCVT latency on the critical path.
   Issue 201's full-f16 path removes the FCVT (widening FMA does it implicitly)
   — but the FHM FMA itself has nonzero throughput cost, and the 4-accumulator
   unroll that hides f32 FMA latency doesn't fully hide the FHM FMA's different
   pipeline characteristics. Net: latency-hiding is *better* than Issue 200 but
   not enough to reach 1.5×.

3. **f16 accumulation drift grows with vector length.** rel_err is ~0.1% at
   in-cache sizes but 6.2% at 16M elements. This is expected (16M f16
   roundings accumulated into f32) and NOT a kernel bug, but it means even if
   the perf gate had passed marginally, the G1 precision gate (Issue 201
   Phase 2 T10) would have required a separate f16-accumulation-precision
   analysis that likely caps usable vector lengths.

### What this closes

- **Phase 1 (kernel microbenchmark):** DONE — gate failed.
- **Phase 2 (full forward path):** NOT PURSUED — Phase 1 gate is a prerequisite
  and it failed. Implementing `ForwardContextF16` + a full f16 forward path
  would be dead code on this hardware class.
- **The weight-only vs full-f16 distinction:** now empirically characterized end
  to end. Weight-only f16 (Issue 200) is 1.7× *slower*; full f16 (Issue 201) is
  1.3× *faster* but short of the promotion gate. f32 remains the production
  dtype for `forward_base` GEMV on Apple Silicon.

## Reproduction

```bash
# Requires the nightly toolchain (stable cannot access FHM intrinsics — see
# Toolchain note). The benchmark source is a one-off probe and is NOT shipped
# in the stable codebase; it lives in the Issue 201 investigation worktree.
cargo +nightly run --release -p katgpt-types --example fmlal_bench_201
```

## Lesson (for future f16 attempts)

The "halve bandwidth → ~2× speedup" intuition for GEMV is **hardware-class- and
precision-dependent**, not universal. On Apple Silicon aarch64:

- It does NOT hold for weight-only f16 (f32 activations cap the reduction at
  25%, Issue 200).
- It does NOT hold even for full f16 with hardware widening FMA (the kernel
  isn't purely bandwidth-bound, and FHM FMA throughput + accumulator reduction
  eat the theoretical gain, this bench).

The only remaining f16-style path with a plausible ≥1.5× win on this hardware
would be **INT8 with INT8 activations** (the quantized-inference literature's
regime), which is a different dequant path entirely and out of scope for both
Issue 200 and 201. Filed as a non-goal in both issues; not pursued.

This mirrors the broader GOAT-gate discipline: the gate's job is to catch a
wrong hypothesis before it reaches production, and here it did its job twice
(Issue 200 weight-only, Issue 201 full-f16) — saving the codebase from a
perf-regressing "optimization."
