# Issue 898: KL Effective Depth — Measured Exit Calibration for the Looped Runtime

**Status:** Open — instrument half LANDED 2026-09-26 ([Bench 899](../.benchmarks/899_kl_effective_depth_goat.md), commit in the Tasks list): G1 PASS on 3 micro fixtures (holdout ±1 loop 1.000/0.969/0.844, fit 1.0 nat interior after the first grid hit its edge), lens premise + kill-switch + G1b + G4 PASS, no Config default justified (calibrated p95 exit 2/6/4 of 8 — checkpoint-dependent). Remaining: the contingent BO arm only, BLOCKED on a BO-trained checkpoint (riir-train `.plans/421` Phase 4).
**Source:** [arXiv:2609.19107](https://arxiv.org/abs/2609.19107) "How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents" — distillation: [`.research/592`](../.research/592_Loop_Growth_Scaling_Exponents.md)
**Class:** poc + optimization (measurement instrument + config calibration)
**Cousins:** Plan 428 (`loop_stability_fix` — the Norm(h) half of the boundary operator, shipped), Plan 304 (`gain_cost_halt`), Research 273 / the ELT any-time-exit implementation in `forward_looped` (the Issue-035 work landed as code + Research 273; the issue file was removed per the noise-reduction rule), Issue 568 (injection NO TRANSFER — hazards the contingent arm below, record: `negative_results.md` §22)

## Problem

The looped runtime's depth knobs are hand-tuned: `Config::loop_min` / `loop_max` (ELT any-time exit), the `gain_cost_halt` ε threshold, and the ELT 2× over-iteration cap all carry defaults with no per-checkpoint measurement behind them. arXiv:2609.19107 ships the measurement that fills this: **KL effective depth** — decode the residual stream after each block with the logit lens; the first block after the KL peak whose decode is within a KL threshold of the final output is the effective depth — the natural exit point. (The diagnostic itself is established prior art — HRM-Text runs it; our claim is the calibration instrument, not the diagnostic.)

## What ships

Four offline measurements over checkpoints the stack already serves, plus the calibration they feed:

1. **KL effective depth probe** — per-checkpoint scalar + per-layer KL vector (BLAKE3-pinnable, deterministic).
2. **Loop-flatness score** — spread of L(k) over k ∈ {2..8} per checkpoint; checkpoint-selection criterion for loop-count-elastic serving (paper reference band: fixed-trained +0.008..+0.042, random-recurrence-trained +0.0003..+0.0028 — context, never our bar).
3. **Write-fraction spectrum** — ‖Δh_k‖/‖h_k‖ decay across loop iterations; principled ε calibration for `gain_cost_halt` (early-depth waste is the training mirror of runtime dead compute).
4. **ELT depth-execute distribution** — histogram of per-token executed depth under any-time exit.

**Output:** depth-calibrated defaults for `loop_min` / `loop_max` / halt ε, replacing hand-tuned values where the measurement justifies it.

## GOAT gate

- **G1 (headline): holdout prediction** — fit the threshold map on fixture half A, predict optimal exits on half B within ±1 loop; loss parity vs hand-tuned at matched compute. The paper's own 8×-extrapolation protocol is adopted as the gate arm: a gate without holdout validation is unvalidated extrapolation.
- **G1-flat (passing-negative):** if the measured KL-depth profiles are FLAT across our checkpoints (the Plan-428-PoC / Bench-699 precedent: both fixtures measured flat — no explosion, no depth structure), the calibration degenerates to CONFIRMING the hand-tuned defaults. That outcome CLOSES THIS ISSUE CLEAN as a passing negative — the instrument measured, the defaults stood — never as a miss. The gate asserts only the structural floors (determinism, canary, alloc) in that branch.
- **G1b:** monotone-decay signature asserted on OUR checkpoints (tested, not assumed) + planted identity-loop canary (write-fraction must read exactly 0 → detector must fire).
- **G2:** probe cost ceiling with box-state PROVENANCE line.
- **G4:** zero-alloc probe path.
- **Kill-switch:** calibration off = bit-identical to hand-tuned defaults.
- **R9 law (from the paper):** any mode comparison carries the retuned-config arm + the transferred-config negative control (expected to underperform — proving the tuned arm is not luck).

Feature-gated (opt-in until the GOAT passes; demote hand-tuned defaults only if the gate wins).

## Contingent arm — boundary operator at inference (DO NOT implement first)

`BO(h,e) = Norm(h) + α·e` between loop passes and **before the readout** extends Plan 428's inter-loop RMSNorm (which ships the Norm half only — no input re-injection, no readout-boundary application; the paper's Table 4 puts coda injection at ~2×10⁻³ of loss). **This arm is blocked on a BO-trained checkpoint** (riir-train `.plans/421` Phase 4 output): Issue 568 measured NO TRANSFER for input injection on a substrate not trained for it — injection phenomena are trained-model phenomena, and our frozen checkpoints were not trained with BO. When the BO-trained checkpoint exists, this arm becomes: apply BO at the loop boundaries + readout, α retuned per mode, G1 allowed to REFUTE (an out-of-distribution transform on a checkpoint trained with BO is still a distribution shift across the growth boundary); the Issue-568 PoC source (`riir-poc/loop_injection_poc.rs`) is the permanent regression check.

Also recorded (testable prior, not a default): the paper's K*=4 growth target — if our own K-sweep elbow differs, OUR constant is recorded here and the paper's is not promoted.

## Validation

Probe determinism on frozen fixtures (byte-identical per-layer KL vectors); calibration holdout arm above; every existing looped gate (`goat_108_lt2_looped`, `issue_035_any_time_lt2_dispatch`, `goat_428`, `issue_717_*`) green with the feature off.

## Tasks

- [x] T1 KL effective depth probe — `katgpt_core::loop_depth_probe` (`kl_from_logits`, `kl_profile`, `effective_depth`), fed by `LoopDeepRun::capture_logits` (shares the Issue-717 tripwire matmul). Lens premise measured bit-exact: loop-τ snapshot == exit-at-(τ+1) readout at pos 0.
- [x] T2 Loop-flatness score — `spread`; loss-vs-k spread 0.45 / 0.88 / 0.80 nats on the three fixtures (Bench 899).
- [x] T3 Write-fraction spectrum — `write_fractions` + `stall_onset`. ⛔ Finding: on these fixtures the state keeps writing 25–52 % of its norm on the step that settles the argmax, and the stable-gate fixtures never stall at ε=1e-3, so no write-fraction ε finds the readout's exit (a halt that tracks the readout has to measure in logit space).
- [x] T4 Executed-depth histogram — `DepthHistogram<N>` (fixed buckets, allocation-free).
- [x] G1 holdout calibration — `agreement_exit` oracle + `fit_threshold` (half A) + `holdout_hit_rate` (half B). STRUCTURED branch, PASS.
- [x] R9 retuned vs transferred — reported; the negative control could not tell the arms apart (all three fits land on 1.0 nat — the threshold transfers, the exit depth does not).
- [x] G1b / G2 / G4 / kill-switch — see Bench 899. G4 surfaced and fixed a latent Issue-717 `LoopDeepStats::clear()` defect (stale empty entries under `capture_states`).
- [x] Validation — `goat_108_lt2_looped` 11/11, `issue_035_any_time_lt2_dispatch` 13/13, `goat_428_loop_stability` 1/1, `issue_717_*` 3/3 + 5/5 green with the feature off.
- [x] Output — calibrated defaults: **none changed** (checkpoint-dependent; `loop_max = 0` → "use loop_mode's count" stands). Feature stays opt-in (a measurement instrument, not a promotion candidate).
- [ ] Contingent BO arm — BLOCKED on a BO-trained checkpoint (riir-train `.plans/421` Phase 4); Issue 568's `riir-poc/loop_injection_poc.rs` stays the regression check.
