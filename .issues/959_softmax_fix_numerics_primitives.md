# Issue 959: Fixed-form softmax numerics primitives (arXiv:2610.31642 adoption, modelless)

**Status:** Open — 2026-10-23 (T1–T4 landed; T5 landed 2026-10-24; T6 claimed)
**Claimed (T6):** zed-m3-idle-loop, 1792777800, expires 2026-10-26T18:16:40Z
**Lease released:** zed-m3-idle-loop (epoch 1792776198) — T5 landed this session; T6 unclaimed, its own lane (riir-infer consumer)

**T1–T4 LANDED (zed-m3-idle-loop, 2026-10-23):** `crates/katgpt-core/src/softmax_fix.rs` —
`FloatFormat` + `PRECISION_CLIFF_NATS` + `precision_cliff(_bits)` (ungated; goldens vs the paper
±0.1 nat + the p·ln2 scaling law + byte-sized enum + full-table coverage, 5 tests),
`margin_equilibrium` (ungated; the Slingshot golden 18.42 > 16.64 HAZARD + the ε=1e-7 clear +
bf16-hazard pin), the `−σ(−m)` doctrine citation beside the sigmoid law (lib.rs) + the cliff-band
property test for `fast_sigmoid` (m ∈ [5.5, 40], 140-point ladder, 1e-6 rel budget — caught the
issue's garbled expm1 reference formula, which is −tanh(m/2), not −σ(−m); the exact f64 form
pins instead), and feature-gated `logspace_diff(_f64)` (opt-in `logspace_diff`; anchored expm1
form, exact +0.0 at equality, overflow-branch-safe; 7 gated tests incl. the no-false-zero
fixture at probabilities near 1.0 — where ULP(1.0)=1.19e-7 zeroes the incumbent at δ=1e-8/5e-8
— and the measured ~19% one-ULP incumbent residual at δ=1e-7). T3's CLI:
`scripts/margin_equilibrium_audit.py` (`--self-test` PASS; `--sweep` prints the paper's regime
table; named in AGENTS.md). Feature count: 708 → 709 at all 5 claim sites; catalog §156;
validation: 12/12 gated + 5/5 ungated module tests, full lib 2167 pass 0 fail (default), clippy
-D clean at default / logspace_diff / no-default postures, all-features check clean, rustfmt
clean, docs_gate 35/35.

**Research:** [katgpt-rs/.research/632_Softmax_Fix_Collapse_Cancellation.md](../.research/632_Softmax_Fix_Collapse_Cancellation.md) (allocated 631, renumbered 632 — dual allocation vs `a4433631d`, citation_weight awarded 631 to the SST note)
**Source:** [arXiv:2610.31642](https://arxiv.org/abs/2610.31642) — Barron et al., "A Fix for Softmax and Cross-Entropy" (Google DeepMind, 2026-10-22)
**Target:** `crates/katgpt-core/src/` (numerics surface; new stable-form family) + consumers in riir-infer/reflex measurement lanes
**Cross-ref:** riir-train Issue 645 (the training-loss half — same paper, different repo)

## Problem

Standard max-subtraction log-softmax saturates at exactly −0.0 once the logit margin passes the precision cliff c_d = p·ln2 (5.5 bf16 / 16.6 fp32 / 36.7 fp64, evaluated in the ACCUMULATOR format), and probability-space differences of near-equal softmax vectors lose their difference (error ~eps·p vs true difference ~δ·p). Neither fixed form ships anywhere in the workspace (grep-verified, 8 repos): ~10 hand-rolled max-subtract variants, softplus never used as log-softmax, expm1/log1p never in a CE/KL path. Scoping: **instrument-grade, not product-grade** — the forward absolute win is ≤ machine-epsilon per value; every benefit is a differencing/measurement benefit. Never a serving-forward replacement (clef expf-fidelity contract and attention O(1) mixtures stay untouched).

## Tasks

- [x] **T1 — cliff constants table.** `PRECISION_CLIFF_NATS` const table + `precision_cliff(format) -> f64`, c_d = p·ln2 per accumulator format (fp8e4m3 ≈ 2.8, bf16 5.545, fp16/tf32 ≈ 7.6, fp32 16.64, fp64 36.73; derivation ours — the paper states numbers, not the law). Ungated const. GOAT G1: goldens vs paper ±0.1 nat + scaling property c_d ∝ p·ln2. *(LANDED 2026-10-23: `softmax_fix.rs`, 5 ungated tests green.)*
- [x] **T2 — sigmoid doctrine citation + property test.** Doc line beside the sigmoid law: the numerically-correct CE target gradient IS −σ(−m) (arXiv:2610.31642 Eq. 3). Property test pinning `fast_sigmoid` relative error vs `−exp_m1(−m)/(1+exp(−m))` over m ∈ [5.5, 40] — the saturation band where the Cephes polynomial / early-exit envelopes live. *(LANDED 2026-10-23, with a correction: the issue's quoted reference formula equals −tanh(m/2), not −σ(−m) — expm1(−m) = e^{−m}−1, not e^{−m}. The test pins the exact f64 form −1/(1+e^m) instead; 140-point ladder green.)*
- [x] **T3 — margin-equilibrium config audit.** Offline instrument: given (ε, format) → (m* = ln(1/ε), c_d, verdict). Golden: (1e-8, fp32) → (18.42, 16.64, HAZARD — the paper's Slingshot prediction). Home question (katgpt-rs scripts vs riir-refine dev-tools) flagged for boundary-guard; default katgpt-rs scripts family. *(LANDED 2026-10-23: `scripts/margin_equilibrium_audit.py` — --self-test PASS, --sweep prints the paper's table; + the in-crate `margin_equilibrium(eps, format)` half with the same goldens.)*
- [x] **T4 — `logspace_diff`.** `p − q` from δ logits via expm1/log1p (paper Eq. 9 family): faithful EXACT 0.0 at δ = 0 (incumbent probability-space subtraction carries residual ~1e-3 bf16 / ~1e-8 fp32 class). Feature-gated `logspace_diff`. G1: at δ ∈ {0, 1e-6, 1e-3} vs f64 oracle — incumbent residual matches the paper's figures, fixed is exact-0/ulp-scale. Consumers: comparison-lane agreement/divergence metrics (reflex), capture-verification deltas (RIDT), KL checks. Rides the already-shipped `noisy_or_stable` resolution law (`katgpt-core/src/lib.rs` L94-98) — extension, not duplication. *(LANDED 2026-10-23: opt-in `logspace_diff`, f32+f64 arms, anchored-expm1 overflow-safe form; 7 gated tests — the no-false-zero fixture lives at probabilities near 1.0, where ULP(1.0)=1.19e-7 zeroes the incumbent at δ=1e-8/5e-8 exactly as the paper's collapse class predicts.)*
- [x] **T5 — `fixed_log_softmax`.** Argmax-excluded LSE + softplus form (paper Eq. 5): log-prob resolution extends from the cliff c_d to exponent underflow (~87 nats fp32) instead of exact −0.0. Feature-gated `fixed_log_softmax` (f32 + f64 arms). G1: golden vs f64 oracle over margins 0..~103 — incumbent saturates at −0.0 past c_d, fixed ≤ 1 ulp of truth. G2 (--release, box-state provenance): ns/call parity (log1p vs log same cost class). G3: opt-in, default path byte-identical. G4: alloc-free. **Determinism-law seam:** the excluded-argmax LSE changes fold structure → pinned fold order documented at the site (determinism law, katgpt-rs Issue 940 class); any synced crossing goes through the certified precedent (`closure/bridge_certified.rs` shape), opt-in until GOAT'd. *(LANDED 2026-10-24: `fixed_log_softmax(_f64)` + `fixed_log_softmax_target(_f64)` (the NLL-oracle shape, bit-identical to the full form's slot), opt-in `fixed_log_softmax`, feature count 709→710 at all claim sites. G1: 207-rung margin ladder 0..103 in 0.5-nat steps, BOTH slots ≤ 2 ulp of the f64 oracle incl. the subnormal tail — the budget is the f32 libm composition's (expf ≤ 1 ulp + log1p ≤ 1 ulp), disclosed; the aspirational 1-ulp read holds at most rungs (measured worst 3 at m=21.5 against the CORRECT oracle). The incumbent's exact-0.0 saturation witnessed at the cliff in both f32 and f64. TEST-SUITE FINDING worth keeping: the oracle's naive `ln(1.0 + e^{−m})` close reproduced the paper's own Softmax Collapse INSIDE this test — on this box at m = 28.5, `ln(1.0 + 4.1938e-13)` returns 4.19442e-13, a 1.5e-4 relative corruption of ε (the `1 +` add round's half-ulp of 1.0 is 2.65e-4 of ε) — the oracle reads `ln_1p` and the incident is documented at the test. G2: `benches/softmax_fix_log_softmax_goat.rs` — interleaved A/B pairs ×9 rounds, median ns/call, black_box'd; measured 2026-10-24 on the M3 Max (16 cores, AC power, 94 % mem free, load 4.4–6.3, epoch 1792777601): n=2 ratio 0.562 (fixed FASTER at tiny n), n=64 1.212, n=1024 1.315, gate ≤ 1.5 PASS ×3. G4: `tests/softmax_fix_g4_alloc.rs` — 0 allocs / 1000 mixed calls, both arms, both forms. G3: default build byte-identical (the gated module compiles to nothing; full default lib suite 2167 pass / 0 fail unchanged). Overflow-free by construction (every exp argument ≤ +0.0, R ∈ [0, n−1]); lone-`+inf` max prices exactly; single-element row → exactly 0.0. Edges test-covered.)*
- [ ] **T6 — f64 golden-NLL lossy-quant oracle (consumer, deferred until T5).** riir-infer harness arm: per-token NLL at confident positions via fixed log-softmax in f64, as the tail-faithful readout for lossy-quant gates (si_zero_shot_ppl golden arm; PQ2_0 / RIIR_GDN_STATE_HALF lossy arms — the lossy-surface promotion law distrusts flat aggregate perplexity; quant deltas of 1e-6..1e-3 nats on confident tokens are invisible to an incumbent fp32 log-softmax). Files land in riir-infer when wired; this issue owns the primitive only.

Sequencing (no-GD advocate order): T1 → T2 → T3 → T4 → T5 → T6. Each GOAT-gated before the next consumes it.

## Non-goals

- No serving-forward replacement (G5 parity capital is not spent here; §scoping above).
- No training-loss edits in this repo (riir-train Issue 645 owns those).
- No claim of novelty — adoption of published fixes; the diagnosis itself is published prior art (Prieto 2501.04697; Liu 2605.06152; TF #126525/#126748).
