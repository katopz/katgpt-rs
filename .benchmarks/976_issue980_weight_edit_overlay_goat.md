# Bench 976: WeightEditOverlay GOAT — reasoning-authored, gate-gated weight-edit lane (Issue 980)

**Date:** 2026-11-04
**Feature:** `weight_edit_overlay` (katgpt-core, implies `gain_cost_halt`), root forwarder in the root manifest
**Issue:** [`.issues/980_weight_edit_overlay_lane.md`](../.issues/980_weight_edit_overlay_lane.md)
**Research:** [`.research/648_Weight_Space_Programming.md`](../.research/648_Weight_Space_Programming.md) — "Sparks of Weight-Space Programming" (Takehi/Teoh/Zhang/Khattab, MIT CSAIL blog, 2026-11-02)
**Verdict: GOAT G1–G4 PASS — stays OPT-IN** (no-default-consumer rule, metabolic_gate / Bench-716 precedent; promotion to default requires a real consumer lane)

## Substrate check (substrate-first, pre-implementation)

| Concept | Searched variants | Found (consume, not fork) |
|---|---|---|
| Obstacle telemetry (arming condition) | oscillation detector, loop halter, halt counters | `GainCostLoopHalter` + `HaltReason::Oscillation` + `step_size`/`angular_change` (`gain_cost_halt.rs`) — CONSUMED |
| Overlay dispatch shape | LoRA pair, adapter swap | `LoraPair { reader, writer }` (katgpt-types, Plan 025) — shape precedent |
| Sparse carrier | COO bypass, top-K sparse | `QuantErrorLora::SparseErrorBypass` (Issue 565) — carrier precedent + dequant-domain posture |
| Freeze envelope | merkle freeze, frozen bank, artifact commitment | `FrozenDirectionBank` freeze/verify/thaw (`set_admission_freeze.rs`) — pattern consumed |
| Gate drive | sigmoid gate, soft gate | `katgpt_core::exact_sigmoid` (the bit-stable two-branch form — never the ±40-clamped Cephes `sigmoid` for gates) |
| G4 alloc gate | counting allocator | `tests/common/mod.rs` `counting_allocator!()` + Issue-714 liveness canary |

Decision: build the overlay in katgpt-core (it IS the substrate repo); zero new deps; zero `riir-*` deps.

## The fixture (T1 — no LLM needed; proves the overlay + gate machinery)

`InjectedDegeneracyMlp` (128→64→64→4, tanh) policy over a 2-D grid world
(actions up/down/left/right; step ±1; success = x ≥ 6 within 64 steps) with
a seeded degenerate direction: the carrier column of W2
(`ΔW2[:, carrier] = −k₀·d`, `d = −e₁`, k₀ = 8) with a region-gated carrier
input feature `bump·(y/8 + 0.05)·4` (bump = 1 iff |y| ≤ 4). Decision gaps:
down−up = `1.6·h2[1]`, right−left = `0.5·(h2[2] − h2[0])`.

Analytic shape, verified by the test: inside the region the injected slope
flips the vertical decision (marches to the boundary); at the boundary the
region-gated carrier turns off/on each step → a sustained two-action
oscillation at y ∈ {4,5} / {−5,−4} with both gap magnitudes (≥ 0.887)
exceeding the max horizontal pull (0.67) → the episode never advances on x
→ FAILS. The no-degeneracy twin (same seed, no injection) centers on y = 0
then marches right → succeeds. **The obstacle hides the ability; the repair
un-hides it.**

- Base obstacle rate (degenerate): **1.0** (32/32 in-region starts, 16 eval seeds)
- Base obstacle rate (twin): **0.0**
- Arming telemetry (the shipped `GainCostLoopHalter` consumed, patience 2):
  degenerate → `Armed`, twin → `Refused(NoObstacleFound)` — and the
  halter-based statistic agrees with the fixture's reversal-streak statistic
  on **every seeded episode** (cross-agreement test, 16/16 seeds)

## G1 — specificity battery (T3, all arms PASS)

| Arm | Result |
|---|---|
| (a) dose-response, k ∈ {1,2,4,8} | obstacle rates **{1.0, 1.0, 1.0, 0.0}** — monotone non-increasing ✓; continuous secondary probe (mean \|h2[1]\| at y ∈ {−1,0,1}) strictly decreasing **0.947 → 0.918 → 0.808 → 0.083** ✓ |
| (b) neighbor-column isolation null (same edit at the tiny-constant column) | rate ≈ base, \|Δrate\| ≤ 0.05 ✓ |
| (c) norm-matched random control, 16 BLAKE3-seeded arms (RandOpt posture) | targeted mean 0.0 beats the **once-picked** best random arm's mean; paired **LCB95 > 0** ✓ (beating only the ensemble mean would have failed) |
| (d) obstacle pre-screen refusal | twin telemetry → `Refused(NoObstacleFound)`; zero fake gain ✓ |
| (e) retention | out-of-region forwards **bit-identical** (carrier mass exactly 0 → delta exactly 0); clean-start rollout prefixes identical; full-dose repair restores the twin's **whole trajectory bit-exactly** ✓ |

## G2 — apply-at-load compose vs full bake (--release, interleaved A/B, 31 pairs, 8× inner)

| shape | matrix | \|S\| | compose cycle (ns) | full bake (ns) | bake/compose | forward-delta (ns) |
|---|---|---|---|---|---|---|
| fixture | 64×64 | 1 | 93.8 | 156.2 | **1.67×** (compose wins) | 9.8 |
| llm-layer | 4096×1024 | 64 | 2,508,677 | 238,255 | 0.09× (bake wins) | **280.6** |
| wide-in | 4096×16384 | 64 | 5,211,281 | 4,230,818 | 0.81× (bake wins) | **282.5** |

**Honest finding (recorded, not buried):** the compose form writes
row-major COLUMNS — every write strides `in_dim·4` bytes, so at LLM scale
each column walk re-fetches a cache line per row and compose becomes
memory-bandwidth-bound by the stride pattern while the bake streams
sequentially. Compose wins where the tensor is cache-resident (the fixture
posture, 1.67×). **The designed serving posture for large/packed bases is
`forward_delta_into`** — O(|S| + out), ZERO matrix traffic: 281 ns at
LLM scale, ~9,000× under the compose cycle there, and it never touches the
packed bytes at all (the QuantErrorLora posture the module preaches).

**Box state (Issue-021 law):** macOS arm64, 30 logical CPUs, AC power,
lowpowermode 0 (Automatic), load 3.17/3.98/3.73 at bench start (no cargo
compute consumers at snapshot; Zed + fleet-web resident). Isolated target
dir `/tmp/issue980_tgt`. Compose number = the compose+revert lifecycle
(the apply-at-load + rejected-edit-revert cycle), reported as the cycle.

## G3 — untouched-surface discipline

- Untouched W2 columns **bit-identical** after compose (asserted elementwise)
- Rejected-edit revert leaves θ **bit-identical** (BLAKE3 before/after — backup-based, since `W−=k·d; W+=k·d` is not bit-exact in f32)
- Flag-off (no overlay) == empty overlay: forwards and rollouts **byte-identical**
- Default-feature list unchanged (the new feature is opt-in only); default `cargo test -p katgpt-core --lib` 2167 passed, 0 failed
- Packed-base law: int8 dequant twin's bytes **BLAKE3-unchanged** through overlay application

## G4 — alloc-free apply (CountingAllocator + Issue-714 liveness canary)

Apply path (compose + forward-delta + bypass-forward + revert), warmup
excluded, **0 allocs** at every shape (fixture + both LLM scales). The
G4 arm initially caught a real allocation — `touched_count`'s seen-column
`Vec` inside the apply path — fixed with the allocation-free nested-scan
dedupe shared by compose/revert/count (`column_already_touched`). The canary
proves the counter is live.

## Dequant-domain arm (the PQ/ternary posture)

The int8 (symmetric per-row) quantized twin of the DEGENERATE base is
repaired by `forward_delta_into` alone: packed base bytes BLAKE3-unchanged,
out-of-region forwards bit-identical, in-region obstacle rate 1.0 → **0.0**
(the quantized −127/s row entry dequantizes to exactly −8.0 at s₁ = 127/8,
so the +8 delta cancels exactly; the residual 1→16-rounding error on the
y-channel is absorbed by the decision margins).

## Envelope (T5)

Mint-once-then-freeze: `WeightEditArtifact::mint` → bytes → `thaw` (header
BLAKE3 verified at load) → `verify_overlay` (payload digest + length) →
apply. Tamper refusal: a flipped envelope byte refuses thaw; a flipped
payload byte fails artifact verification. Provenance trailer (prompt digest
/ seed / author) all optional — the artifact BLAKE3 is the only mandatory
commitment.

## Promotion verdict

**GOAT G1–G4 PASS; STAYS OPT-IN.** No consumer lane exists (the serving
rung is veto-gated per Research 648 §4: a consumer needs a documented loop
class + a WSP-capable author; hosted-author + weight-egress policy is
riir-train `.issues/665`'s scope). Demotion law pre-registered (Issue-565
precedent): if the battery fails on a real consumer, record the negative
and demote the carrier to the SparseErrorBypass representation.

## Artifacts

- Module: `crates/katgpt-core/src/weight_edit_overlay.rs` (11 unit tests)
- Gate battery: `crates/katgpt-core/tests/issue980_weight_edit_overlay_gates.rs` (12 tests, green-zero paired with `required-features`)
- Bench: `crates/katgpt-core/benches/issue980_weight_edit_overlay_goat.rs`
- Feature catalog: `.docs/09_feature_catalog/opt_in_features.md` + README row

## Reproduce

```bash
cargo test -p katgpt-core --features weight_edit_overlay --test issue980_weight_edit_overlay_gates
cargo test -p katgpt-core --features weight_edit_overlay --lib weight_edit_overlay
cargo bench -p katgpt-core --features weight_edit_overlay --bench issue980_weight_edit_overlay_goat
```
