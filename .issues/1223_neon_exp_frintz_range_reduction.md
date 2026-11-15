# Issue 1223 — `simd_exp_inplace` NEON range reduction uses FRINTZ (toward-zero), not nearest — ~2.4e−5 systematic relative error on aarch64

**Status:** OPEN — kernel accuracy bug (found by the katgpt-rs Issue 1220 lane's `exp_kernel_census` probe, 2026-11-16); load-bearing: this exp backs every softmax/attention/activation on native aarch64 (M3/M5 lanes), and the module doc claims ~1 ULP.

## The defect

`crates/katgpt-types/src/simd/activations.rs` reduces the exp argument with `n = round(x·log2e)` — and the three NEON sites compute "round" with **`vrndq_f32`, which is FRINTZ (round toward ZERO), not round-to-nearest** (`vrndnq_f32` is the nearest instruction). The scalar `cephes_exp_scalar` uses `.round()` (correct); all four AVX2 sites use `_mm256_round_ps(…, ROUND_NEAREST)` (correct); **only the NEON path truncates**.

Whenever `frac(x/ln2) ∈ [0.5, 1)` (≈ half of all inputs), truncation puts the reduced argument at `|g|` up to `ln2` (0.693) instead of `ln2/2` (0.347) — outside the 6th-order polynomial's designed range. Taylor truncation error there is `≈ g⁷/7! ≈ 1.9e−5` relative, phase-locked to `k·ln2` (the census spike spacing is exactly 3·ln2 = 2.079).

## Measured (M3, aarch64, `exp_kernel_census` probe in `crates/katgpt-types/tests/gegelu_lane_parity.rs`, 2026-11-16)

- Max relative diff vs libm `f32::exp` over [−10, +10] (4001 points): **2.81e−5 = 236 ulps** (worst points: x = −5.545, −3.465, −9.010, −6.930, −1.385 — all within one grid step of `k·ln2`).
- The doc claims "~1 ULP for |x| < 88" — false on aarch64 by ~235×.
- Bias is SYSTEMATIC (not random): every softmax weight computed through the NEON path on half of all inputs skews the same direction per phase. For softmax the renormalization cancels part of it, but the residual is input-dependent — attention weights and gated activations (gegelu/silu/swiglu bodies all route through `simd_exp_inplace`) carry a ~1e−5-class systematic distortion on every native aarch64 run.
- Why it survived: the historic x86_64 clamp bug (test_gate.sh header, riir-train Issue 549) got the AVX2 path the ROUND_NEAREST scrutiny; the NEON path "always clamped" and never got an accuracy audit; CI's executing lane is ubuntu x86_64, where the correct AVX2 path runs. **The M3/M5 native lanes have been computing on the truncated kernel since it landed.**

## The fix + the minefield it must plan around

- One-word fix per site: `vrndq_f32 → vrndnq_f32` at lines ~1086 (`neon_exp_inplace`), ~1180 (`neon_exp_sum_inplace`), ~1253 (third NEON kernel — same reduction family; verify which). FRINTN and FRINTZ cost the same (one instruction) — G2 should be a wash.
- ⛔ **Every aarch64 value-pinned fixture was computed WITH the bug.** Fixing the kernel moves outputs by up to ~5.6e−5 relative on affected phases, so the change must sweep and re-pin every value-pinned fixture that routes through exp on aarch64 (BLAKE3-pinned walks, golden decision fixtures, G5-style parity captures computed natively). A grep for pinned-digest fixtures touching softmax/attention/activation outputs is the first task; the fix and the re-pin land in ONE commit family.
- ⛔ Adjacent finding from the same census: the `gegelu_tanh` SIMD body computes `E/(E+1)` with `E = exp(+2·inner)` — **NaN for `inner > 44.5`** (`inf/inf`) where the old tanh-form tail saturated gracefully. The Issue-1220 tail + the `gemma2_train::gegelu_backward_into` backward now use the overflow-free reciprocal spelling `1/(1+exp(−2·inner))` (finite for every finite gate; training fixtures exercise the range — 9 `gemma2_lora` test failures proved it pre-fix). The BODY still carries the `E/(E+1)` overflow hole on all platforms — same reciprocal spelling fixes it, same re-pin sweep, same campaign.
- Gate shape: G1 = post-fix census ≤ ~2 ulp vs libm across the range (the scalar `.round()` Cephes proves the polynomial itself is 1-ulp-class once fed the right `g`); G2 = unchanged (same instruction count; pre-registered A/B bar per the build-hygiene skill); G3 = the fixture re-pin sweep is the no-regression evidence; G4 n/a (no alloc change). Demote posture: keep `vrndq` behind a kill-env for one cycle if a fixture disagrees loud.

## Related

- Found by: katgpt-rs Issue 1220 (gegelu_tanh tail-lane divergence) — the 4-ulp band hypothesis failed at ~150 ulps mid-band, which is what forced the kernel census; the 1220 lane-parity band pins this measured accuracy class until this issue lands.
- Research 673 §4.1 (the rounding-bug atlas) — this is a new atlas row: the range-reduction rounding regime, discovered in-stack.
- Module doc fix rides the same commit ("~1 ULP" becomes true post-fix).

- [ ] Sweep: enumerate every value-pinned fixture computed through `simd_exp_inplace`/`simd_exp_sum_inplace`/`cephes_exp_scalar` consumers on aarch64 (softmax, attention, gegelu/silu/swiglu bodies, sigmoid kernels) — BLAKE3 pins, golden decision fixtures, G5-style captures.
- [ ] Fix: `vrndnq_f32` at the three NEON sites + the reciprocal spelling for the gegelu body's `E/(E+1)` + the doc correction, one commit family.
- [ ] Re-pin: every fixture the sweep named, with a per-fixture note of the phase-class delta (≤ ~5.6e−5 relative) — a fixture that moves MORE than that names a second defect, not a re-pin.
- [ ] Gates: post-fix census probe (≤ ~2 ulp claim becomes TRUE — keep `exp_kernel_census` as the standing witness); G2 A/B (wash expected); any fixture that cannot re-pin (frozen external capture) gets the demote-env cycle instead.
- [ ] Close-out: Research 673 §4.1 atlas row + HISTORY line + this file removed.
