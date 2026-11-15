# Issue 1220 — `gegelu_tanh` tail-lane divergence: SIMD sigmoid-form vs scalar `1+tanh` cancellation form

**Status:** OPEN — poc/audit task (arXiv:2610.31642 companion arc, Research 673 §4.1; band/cap re-pinned per the verdict-round EXECUTED measurement, 2026-11-15)

**Measured lane divergence** (verdict round, length 65, up = 1 — these numbers are the test's oracle, do not re-derive by hand):

| g | SIMD body | scalar tail | diff |
|---|---|---|---|
| −5.0 | −2.29e−7 | −2.98e−7 | 6.9e−8 (peak region) |
| −5.2 | −5.69e−8 | −0.0 | 5.7e−8 (onset: scalar reads −0.0 from here) |
| −5.5 | −5.93e−9 | −0.0 | 5.9e−9 |
| −6.0 | −8.44e−11 | −0.0 | 8.4e−11 (matches the blog's −8.4e−11) |
| +5.0 | 5.0 | 4.9999995 | 4.8e−7 (one ulp of 5.0 — `1+tanh` rounding, not cancellation) |

- [ ] Property test reproducing the divergence across chunk-boundary lengths (63/64/65/128/127): assert SIMD vs scalar answers over g ∈ [−8, +8], pinning BOTH sides — the negative tail (scalar reads −0.0 from g ≈ −5.2; lane-gap cap ≈ **1e−7·|up|**, peak ≈ 7e−8 near the onset) AND the positive side (g = +5 lane gap 4.8e−7 = one ulp; include g > 0 in the band — it is a `1+tanh` rounding effect, not cancellation, but it is lane parity all the same). The mechanism note: scalar `1+tanh` loses digits when |1−tanh| < half-ulp (cancellation, g ≲ −5.2) and rounds `1+tanh ≈ 2` to the ulp grid near +2 (positive side).
- [ ] Site census of the same shape (`0.5·x·(1+tanh z)` and fast_tanh spellings): `crates/katgpt-types/src/math.rs::gegelu_tanh` (scalar tail only — SIMD body already `x·σ(2z)`), `crates/katgpt-core/src/coda.rs::GateActivation::GegeluTanh` (fast_tanh), `crates/katgpt-speculative/src/belief_drafter.rs::gelu` (fast_tanh), `riir-engine` `motor_gated_rehearsal/policy_head.rs::gelu` (sibling repo — note-only there), bench/example copies (`bench_561_inversion_goat.rs`, `inversion/tests.rs`, `transformer_inversion_01_forensics.rs`).
- [ ] Alignment decision, gated on the training contract: the scalar tail's in-code comment pins it to standard `tanh` because `gemma2_train::gegelu_backward_into` (analytic backward) must match for FD gradient checks — re-derive the backward's tail form first; if backward moves to the σ-form derivative in the same change (`d/dz[x·σ(2z)]` composition), switch forward+backward together and keep FD parity green; otherwise document the tail as a disclosed inference-only divergence (severity: ≈1e−7 absolute on a hidden activation, the Gemma-2 hot path is the SIMD lane).
- [ ] Upstream cross-check at fix time: TorchLean's tanh-GELU row (PyTorch/JAX/Transformers all return −0 at x=−6 f32; rewrite `x·σ(2z)` measured, bound pending upstream) — cite Research 673.
- [ ] Close-out: note + ref the commit hash in Research 673 §4.1, remove this file.

Severity: LOW (≈1e−7 absolute on a hidden activation; parity/audit class — not a serving defect). The falsification arm is the property test in task 1 (it must FAIL on the current scalar tail and PASS post-alignment WITHIN THE PINNED CAP, not bit-identical — the SIMD lane uses its own `exp` and a scalar σ-form would use libm, so a one-ulp gap can survive alignment).
