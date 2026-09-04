# Bench 568 — Plan 568 Phase 3: fused multi-stage LUT dequant+dot (NEGATIVE RESULT)

## Status: G2 PERF GATE FAILED — Phase 3 closes as a documented negative result on the perf axis (G1 correctness PASS; the kernel ships as correct substrate, opt-in, not promoted)

## Origin

[Plan 568](../.plans/568_recurrent_residual_quantization.md) — Recurrent Residual
Quantization (RRQ). Phases 1 + 2 ship the modelless, calibration-free,
single-checkpoint multi-precision weight primitive (`RrqWeights` =
`Ŵ0 + Σ R̂k`, each stage 2-bit; prefix-t is a usable model at 2/4/6/8-bit
for the default 2+2+2+2 config) + the load-time PMR/KS quant-strategy
router. **Both phases ship GOAT-positive.** Phase 1 + 2 G1/G3/G4 all PASS;
the arithmetic `prefix_dot_into` (sum of per-stage GEMVs, exploiting matmul
linearity) is the recommended hot path.

Phase 3's hypothesis was that a **fused multi-stage LUT dequant+dot kernel**
(summing N stage contributions in registers before the single FMA with `x`,
amortizing the gather across stages) could reach parity with the
**single-stage 8-bit LUT path** at the 8-bit prefix — i.e. that 4×
(4-entry 2-bit LUT gather) could match 1× (256-entry 8-bit LUT gather)
because the tiny 4-entry LUTs stay hot in L1 while the 1KB 8-bit LUT spills.

Phase 3 T3.2 set a clear **decision gate**: *if the 4-stage 2-bit LUT path
is NOT within 1.05× of the single-8-bit LUT path at the 8-bit prefix, the
fused kernel is a documented negative result; the arithmetic
`prefix_dot_into` remains the recommended hot path.* This file records
that measurement.

## Setup

- **Hardware:** Apple Silicon (M3), NEON backend.
- **Build:** `--release`, `--features rrq_quant,simd_lut_dequant`.
- **Shape:** 256×256 weight matrix, `group_size = 128`, 4 RRQ stages (the
  default 2+2+2+2 config → 8-bit effective at the full prefix). Single-vector
  dot (x is `cols`-long), 200 iterations after a warmup.
- **Comparison paths (both at the 8-bit prefix):**
  - **4-stage 2-bit LUT** — `RrqWeights::prefix_dot_lut_into(4, ...)`: per
    group, builds `[f32;4]` LUTs per stage (`zp + code·scale`), pre-unpacks
    2-bit codes to 1/byte once, then `dequant_dot_via_lut_multi_stage_slice`
    sums 4 stage LUT values per element in registers before one FMA with `x`.
  - **single-8-bit LUT** — `dequant_dot_via_lut::<Int8Lut>`: one 256-entry
    LUT gather per element (the Plan 452 single-stage kernel on an
    independently-constructed 8-bit representation of the same weights).
- **Both paths produce a numerically equivalent result** (the LUT path is
  verified against the arithmetic `prefix_dot_into` in
  `g1_prefix_dot_lut_matches_arithmetic`). This bench measures latency only.

## Results

| Run | 4-stage 2-bit LUT (ns) | single-8-bit LUT (ns) | ratio | gate |
|---|---:|---:|---:|---:|
| Previous session (2026-08-07) | 84 560 | 13 266 | **6.374×** | ≤ 1.05× |
| Re-verification (this session) | 69 750.6 | 12 402.1 | **5.624×** | ≤ 1.05× |

The two runs bracket the microbenchmark noise (5.6×–6.4×); the verdict is
identical. The 4-stage LUT path is **~5–6× slower** than the single-8-bit
LUT path at parity. Gate = 1.05×. **FAIL by a wide margin.**

## Verdict: G2 DECISION GATE FAILED

`ratio ≈ 5.6×–6.4× ≫ 1.05× → FAIL → Phase 3 fused kernel is a documented negative on the perf axis.`

The fused multi-stage LUT hypothesis is **refuted at realistic LLM-tile
scale**. The kernel is correct substrate (G1 PASS — see below) but not a
perf win. The arithmetic `prefix_dot_into` from Phase 1 remains the
recommended hot path.

### Why the hypothesis failed (root-cause)

1. **4× code-read + 4× gather overhead dominates the L1-residency benefit.**
   The multi-stage kernel reads 4 separate code streams and performs 4 LUT
   gathers per element (summed in registers), then one FMA. The single-8-bit
   path reads 1 code stream and performs 1 gather per element, then one FMA.
   At any tile scale where both LUTs fit in L1, the 4× gather cost is pure
   overhead with nothing to amortize it against.

2. **The 8-bit LUTs do NOT spill L1 at realistic tile scales.** The
   hypothesis depended on the 1KB 8-bit LUT spilling to L2 (where cold-gather
   latency would hurt the single-stage path) while the 4×16-byte 2-bit LUTs
   stayed in L1. At `gs = 128`, a 256×256 matrix needs only 2 LUTs per row
   (2 KB total for the 8-bit path) — well within the 128 KB L1. There is no
   spilling to amortize. The premise of the hypothesis doesn't hold at the
   scale where LLM weight tiles actually live.

3. **Structural, not parametric.** The only regime where the hypothesis
   could hold is when the 8-bit LUTs exceed L1 (>32 KB of LUT data → tens of
   thousands of distinct groups). That requires matrices far larger than any
   realistic single-layer tile. And at that scale, cold-LUT gather latency
   hurts *both* paths equally (the 2-bit LUTs also leave L1), so the
   relative advantage still doesn't materialize. There is no parametric knob
   (group size, stage count, tile size) that flips the verdict within the
   realistic operating range.

### What this closes

- **Phase 3 (fused multi-stage LUT kernel):** DONE — G2 gate failed; kernel
  ships as correct opt-in substrate, not promoted.
- **Phase 4 (prefix-t as tier dispatch):** REMAINS DEFERRED (P3 stretch, no
  concrete consumer). The G2 negative makes the tier-dispatch story rely on
  the arithmetic path (`prefix_dot_into`), not the fused kernel. No change
  to the deferral status.

### Why the kernel ships anyway (correct substrate, not dead code)

The G2 negative is a **perf-axis** negative, not a correctness negative.
G1 is unambiguously PASS:

- `g1_multi_stage_matches_scalar_reference` — multi-stage kernel matches an
  independent scalar reference.
- `g1_multi_stage_single_stage_matches_single_stage_kernel` — at N=1 stage,
  the multi-stage kernel is bit-identical to the Plan 452 single-stage
  kernel.
- `g1_multi_stage_alignment_boundaries` — correct at non-multiple-of-16 tails.
- `g1_group_lut_matches_rrq_affine` — `group_lut_at(g)` matches the RRQ
  affine (`zp + code·scale`) including the degenerate `scale=0` case.
- `g1_codes_unpacked_roundtrip` — 2-bit → 1/byte unpack round-trips.
- `g1_prefix_dot_lut_matches_arithmetic` — the full LUT prefix-dot matches
  the arithmetic `prefix_dot_into` within FP tolerance.

The kernel + RRQ integration (`codes_unpacked_into`, `group_lut_at`,
`prefix_dot_lut_into`) stay as opt-in substrate (`rrq_quant +
simd_lut_dequant`) for consumers whose LUT construction is already amortized
— e.g. a future hardware StreamDQ analog where the gather is near-memory and
the 4× gather overhead vanishes. On current general-purpose CPUs, the
arithmetic path wins.

## GOAT summary (Plan 568, all phases)

| Gate | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| G1 (correctness) | PASS (7 tests) | PASS (7 tests) | PASS (5 kernel + 3 RRQ tests) |
| G2 (perf) | n/a (arithmetic, no kernel gate) | n/a (load-time, not hot path) | **FAIL (5.6×–6.4×; gate ≤1.05×)** |
| G3 (no-regression) | PASS (1840→1847) | PASS (1840→1854) | PASS (1845 / 1862 w/ features) |
| G4 (alloc-free) | PASS (0 allocs after warmup) | PASS (load-time, zero-alloc by construction) | PASS (per-group `[f32;4]` on stack) |

**Promotion:** NOT promoted. Default-off (`rrq_quant`), pending a concrete
consumer per the GOAT gate. The G2 negative makes promotion less likely
(the headline fusion is not a perf win); the primitive ships as opt-in
substrate.

## Reproducing

```bash
cargo test -p katgpt-core --release --lib \
  --features rrq_quant,simd_lut_dequant \
  rrq_quant::tests::g2 -- --nocapture
```

The test is `#[cfg_attr(debug_assertions, ignore)]` (release-only perf
gate, same convention as `g2_monster_ai_under_load` and the other tight
perf gates). It passes (asserts nothing — documents the ratio in the log)
so the negative is visible in CI logs rather than hidden.

## See also

- [Plan 568](../.plans/568_recurrent_residual_quantization.md) — the parent plan
- [Research 467](../.research/467_Recurrent_Residual_Quantization.md) — the parent research note
- `Plan 452` — SIMD LUT fused dequant+dot (Phase 3 substrate)
- [Research 418](../.research/418_StreamDQ_SIMD_LUT_DeQuant.md) — StreamDQ → SIMD LUT DeQuant
- [Bench 563](563_issue201_f16_f16_fhm_negative.md) — the prior negative-result benchmark this file mirrors in format
