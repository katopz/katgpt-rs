# Bench 899 — KL effective depth probe GOAT (Issue 898 / Research 592)

**Status:** COMPLETE. **G1 PASS** (STRUCTURED branch on all three fixtures), lens premise + kill-switch + G1b + G4 PASS, G2 recorded with provenance. **No Config default changes**: the calibrated exit depends on the checkpoint, so no single default holds. `kl_depth_probe` stays **OPT-IN**. It is a measurement instrument, not a primitive that could be promoted.

- **Gate:** `tests/issue_898_kl_depth_goat.rs` (6 tests; `--release` and debug both 6/6)
- **Primitive:** `crates/katgpt-core/src/loop_depth_probe.rs` (13 unit tests)
- **Wiring:** `LoopDeepRun::capture_logits` (`src/transformer/loop_deep.rs`), which shares the Issue-717 tripwire's `lm_head` matmul
- **Source:** arXiv:2609.19107, distilled in [Research 592](../.research/592_Loop_Growth_Scaling_Exponents.md)

```bash
cargo test -p katgpt-rs --release --features kl_depth_probe,alloc_tracking \
  --test issue_898_kl_depth_goat -- --nocapture --test-threads=1
```

**PROVENANCE** (the G2 line, M3 Max on AC power): loadavg `6.99 7.55 7.46`, release. A sibling session was running heavy cargo builds the whole time. Every gate except G2 is deterministic (bit comparisons and allocation counters), so the load does not affect them.

## Fixtures

These are the looped checkpoints the stack already serves: the micro config (vocab 27, n_embd 16, 1 layer, Uniform, Ahla) with seeded weights and deterministically built residual-gate schedules. Nothing is trained. Each fixture is measured at T=8 loops over 4 sequences × 16 positions = 64 samples, split into half A (even samples) and half B (odd).

| checkpoint | weights | residual gate |
|---|---|---|
| seed42-zero-gate | `Rng::new(42)` | `ResidualGate::new` (zero) |
| seed7-stable-gate0.2 | `Rng::new(7)` | `new_loop_stable(decay 0.2)` |
| seed1234-stable-gate0.2 | `Rng::new(1234)` | `new_loop_stable(decay 0.2)` |

## Structural gates

| gate | result |
|---|---|
| Kill-switch | `capture_logits` on vs `run: None`: final logits **bit-identical** (3 fixtures × 64 samples). The last snapshot of the lens is bit-identical to the readout. |
| Lens premise | The snapshot at loop τ of a T=8 run is **bit-identical** to the FINAL readout of a (τ+1)-loop run, for k=1..8 at pos 0. This makes "effective depth" the output of actually exiting at that loop. It holds exactly only at pos 0, because at later positions the two runs' KV caches were written by different loop counts. |
| G1b determinism | KL vectors are byte-identical across two runs, and the last-loop KL is exactly 0. |
| G1b canary | On a planted identity loop the write fraction is exactly `[0, 0]` and `stall_onset(ε=0)` fires at loop 1. |
| G4 | 8 warm lens runs (capture + KL profile + write fractions): **0 allocations**. See the finding below. |
| G2 | KL profile over T=8 snapshots × vocab 27: **1748 ns/profile** in release. The ceiling is 50 µs, so this passes. |

**Monotone-decay signature** (G1b, measured rather than assumed; tolerance 1e-3 relative):

| checkpoint | monotone samples |
|---|---|
| seed42-zero-gate | 60/64 |
| seed7-stable-gate0.2 | 49/64 |
| seed1234-stable-gate0.2 | **26/64** |

The paper's monotone KL decay holds for only some of our fixtures. seed1234 is mostly non-monotone: its readout moves away from the final readout before it comes back. This is why `effective_depth` is defined as the first loop at or after the KL peak, and not as the first loop under the threshold.

## G1: holdout calibration

The branch rule was stated before the first run: FLAT if every oracle exit is loop 1, otherwise STRUCTURED. The bar is holdout hit rate ≥ 0.80 within ±1 loop.

The oracle is `agreement_exit`: the smallest loop count at which the readout argmax equals the final argmax and never changes again. It needs no labels.

| checkpoint | branch | fit threshold (half A) | MAE(A) | holdout(B, ±1) | oracle mean exit | predicted mean exit |
|---|---|---|---|---|---|---|
| seed42-zero-gate | STRUCTURED | 1.0 nat | 0.344 | **1.000** | 1.70 | 1.59 |
| seed7-stable-gate0.2 | STRUCTURED | 1.0 nat | 0.188 | **0.969** | 2.44 | 2.28 |
| seed1234-stable-gate0.2 | STRUCTURED | 1.0 nat | 0.281 | **0.844** | 1.81 | 1.94 |

⛔ **The first grid was too narrow.** The first run's candidate grid stopped at 0.3 nat, and all three fixtures fit exactly at that edge (holdout 0.969/0.969/0.875). A fit at the edge of the grid does not show where the optimum is. I widened the grid to 10 nats, and the fit moved inside it to **1.0 nat** on all three fixtures. The test file records both grids. Don't cite the first run's numbers.

### Matched compute: a fixed exit at the calibrated depth vs all T loops

The KL rule needs the final readout, so it cannot be used as a runtime exit. What it produces is an offline calibration: a fixed loop count at the p95 effective depth measured on half A. That fixed exit is scored on half B against running all T loops, which is the hand-tuned default.

| checkpoint | calibrated fixed exit | compute saved | argmax agreement on B | per-k agreement k=1..8 |
|---|---|---|---|---|
| seed42-zero-gate | 2/8 | 75% | 0.969 | .531 .969 1 1 1 1 1 1 |
| seed7-stable-gate0.2 | 6/8 | 25% | 1.000 | .313 .719 .969 .906 1 1 1 1 |
| seed1234-stable-gate0.2 | 4/8 | 50% | 0.969 | .688 .813 .938 .969 1 1 1 1 |

**Verdict on defaults:** the calibrated exit ranges from 2 to 6 of 8 loops across three fixtures that share one architecture. No single `loop_min`/`loop_max` default comes out of this. The design the stack already has, `loop_max = 0` meaning "use the loop count from `loop_mode`", is correct. The probe's product is a **per-checkpoint** calibration, and none of these fixtures is a production looped checkpoint. **No default changed.**

**The paper's K\*=4 constant:** the loop count at which argmax agreement first reaches ≥ 0.969 is 2, 3 and 4 on the three fixtures. We record our own elbows; the paper's K\*=4 is not promoted.

### R9: retuned vs transferred threshold

| target | retuned (own half A) | transferred (from another checkpoint) |
|---|---|---|
| seed42-zero-gate | 1.000 | 1.000 (from seed7) |
| seed7-stable-gate0.2 | 0.969 | 0.969 (from seed1234) |
| seed1234-stable-gate0.2 | 0.844 | 0.844 (from seed42) |

⚠ **The negative control could not tell the arms apart.** All three fixtures fit the same 1.0-nat threshold, so the transferred arm is the same threshold as the retuned one. R9 expected the transferred threshold to do worse, and on these fixtures it did not. The KL threshold carries across checkpoints; the exit depth does not (see the table above). So the negative control does not prove the tuned arm isn't luck. What the holdout split proves is that the fit generalises within a checkpoint.

## Findings about write fractions (the `gain_cost_halt` ε)

| checkpoint | mean write fraction, first step | mean, last step | stalled at ε=1e-3 | write fraction of the step INTO the oracle exit (median) |
|---|---|---|---|---|
| seed42-zero-gate | 0.424 | 2.5e-3 | 59/64 | 0.419 (n=37) |
| seed7-stable-gate0.2 | 0.601 | 4.2e-2 | 0/64 | 0.516 (n=48) |
| seed1234-stable-gate0.2 | 0.488 | 4.6e-2 | 0/64 | 0.253 (n=25) |

⛔ **On these fixtures a write-fraction halt cannot find the readout's exit.** The step that settles the argmax still writes 25–52% of the state norm, and on both fixtures with a stable gate the state never stalls at ε=1e-3. A halt that fires when ‖Δh‖/‖h‖ falls below ε would run every loop on those fixtures, even though the argmax settled at loop 2–4. Here the residual keeps moving along directions the readout ignores. So the "principled ε" the issue wanted from the write-fraction spectrum **does not exist on these fixtures**. A halt that tracks the readout has to measure in logit space: the Plan-283 advantage gate, or a KL-to-previous-loop rule. This finding matters for any calibration of `gain_cost_halt` that assumes the state norm tracks output convergence.

## G4: an allocation in the warm-up

The first G4 run found **1 allocation of 192 B**. The spare pool that recycles snapshot buffers grows on the first `clear()` after the first capture: 8 `Vec` headers × 24 B. This happens once, not on every call, so the gate now warms up twice. The steady state is 0.

The spare pool itself fixes a latent defect in Issue 717's `LoopDeepStats::clear()`. It cleared the inner buffers in place but kept the outer length, so a cleared run with `capture_states` appended the new snapshots after stale empty entries. No gate hit it, because Issue 717's G4 runs with `capture_states = false`. G4 now asserts `state_snapshots.len() == T` after a cleared warm run.

## What this does NOT cover

- **The contingent boundary-operator arm** (`BO(h,e) = Norm(h) + α·e` at the loop boundaries and the readout) is **still blocked** on a checkpoint trained with BO (riir-train `.plans/421` Phase 4). Issue 568 measured no transfer for injection on a model not trained for it.
- **No trained looped checkpoint** exists in this repo. Every number here comes from seeded micro fixtures, and the verdicts are scoped to them.
- The lens premise is exact only at pos 0.
