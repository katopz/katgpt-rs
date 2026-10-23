# Issue 959: Fixed-form softmax numerics primitives (arXiv:2610.31642 adoption, modelless)

**Status:** Open — 2026-10-23 (unclaimed)
**Claimed:** zed-m3-idle-loop, 1792774280, expires 2026-10-25T16:51:20Z — scope T1–T4 (T5/T6 follow-up if the session allows)

**Research:** [katgpt-rs/.research/632_Softmax_Fix_Collapse_Cancellation.md](../.research/632_Softmax_Fix_Collapse_Cancellation.md) (allocated 631, renumbered 632 — dual allocation vs `a4433631d`, citation_weight awarded 631 to the SST note)
**Source:** [arXiv:2610.31642](https://arxiv.org/abs/2610.31642) — Barron et al., "A Fix for Softmax and Cross-Entropy" (Google DeepMind, 2026-10-22)
**Target:** `crates/katgpt-core/src/` (numerics surface; new stable-form family) + consumers in riir-infer/reflex measurement lanes
**Cross-ref:** riir-train Issue 645 (the training-loss half — same paper, different repo)

## Problem

Standard max-subtraction log-softmax saturates at exactly −0.0 once the logit margin passes the precision cliff c_d = p·ln2 (5.5 bf16 / 16.6 fp32 / 36.7 fp64, evaluated in the ACCUMULATOR format), and probability-space differences of near-equal softmax vectors lose their difference (error ~eps·p vs true difference ~δ·p). Neither fixed form ships anywhere in the workspace (grep-verified, 8 repos): ~10 hand-rolled max-subtract variants, softplus never used as log-softmax, expm1/log1p never in a CE/KL path. Scoping: **instrument-grade, not product-grade** — the forward absolute win is ≤ machine-epsilon per value; every benefit is a differencing/measurement benefit. Never a serving-forward replacement (clef expf-fidelity contract and attention O(1) mixtures stay untouched).

## Tasks

- [ ] **T1 — cliff constants table.** `PRECISION_CLIFF_NATS` const table + `precision_cliff(format) -> f64`, c_d = p·ln2 per accumulator format (fp8e4m3 ≈ 2.8, bf16 5.545, fp16/tf32 ≈ 7.6, fp32 16.64, fp64 36.73; derivation ours — the paper states numbers, not the law). Ungated const. GOAT G1: goldens vs paper ±0.1 nat + scaling property c_d ∝ p·ln2.
- [ ] **T2 — sigmoid doctrine citation + property test.** Doc line beside the sigmoid law: the numerically-correct CE target gradient IS −σ(−m) (arXiv:2610.31642 Eq. 3). Property test pinning `fast_sigmoid` relative error vs `−exp_m1(−m)/(1+exp(−m))` over m ∈ [5.5, 40] — the saturation band where the Cephes polynomial / early-exit envelopes live.
- [ ] **T3 — margin-equilibrium config audit.** Offline instrument: given (ε, format) → (m* = ln(1/ε), c_d, verdict). Golden: (1e-8, fp32) → (18.42, 16.64, HAZARD — the paper's Slingshot prediction). Home question (katgpt-rs scripts vs riir-refine dev-tools) flagged for boundary-guard; default katgpt-rs scripts family.
- [ ] **T4 — `logspace_diff`.** `p − q` from δ logits via expm1/log1p (paper Eq. 9 family): faithful EXACT 0.0 at δ = 0 (incumbent probability-space subtraction carries residual ~1e-3 bf16 / ~1e-8 fp32 class). Feature-gated `logspace_diff`. G1: at δ ∈ {0, 1e-6, 1e-3} vs f64 oracle — incumbent residual matches the paper's figures, fixed is exact-0/ulp-scale. Consumers: comparison-lane agreement/divergence metrics (reflex), capture-verification deltas (RIDT), KL checks. Rides the already-shipped `noisy_or_stable` resolution law (`katgpt-core/src/lib.rs` L94-98) — extension, not duplication.
- [ ] **T5 — `fixed_log_softmax`.** Argmax-excluded LSE + softplus form (paper Eq. 5): log-prob resolution extends from the cliff c_d to exponent underflow (~87 nats fp32) instead of exact −0.0. Feature-gated `fixed_log_softmax` (f32 + f64 arms). G1: golden vs f64 oracle over margins 0..~103 — incumbent saturates at −0.0 past c_d, fixed ≤ 1 ulp of truth. G2 (--release, box-state provenance): ns/call parity (log1p vs log same cost class). G3: opt-in, default path byte-identical. G4: alloc-free. **Determinism-law seam:** the excluded-argmax LSE changes fold structure → pinned fold order documented at the site (determinism law, katgpt-rs Issue 940 class); any synced crossing goes through the certified precedent (`closure/bridge_certified.rs` shape), opt-in until GOAT'd.
- [ ] **T6 — f64 golden-NLL lossy-quant oracle (consumer, deferred until T5).** riir-infer harness arm: per-token NLL at confident positions via fixed log-softmax in f64, as the tail-faithful readout for lossy-quant gates (si_zero_shot_ppl golden arm; PQ2_0 / RIIR_GDN_STATE_HALF lossy arms — the lossy-surface promotion law distrusts flat aggregate perplexity; quant deltas of 1e-6..1e-3 nats on confident tokens are invisible to an incumbent fp32 log-softmax). Files land in riir-infer when wired; this issue owns the primitive only.

Sequencing (no-GD advocate order): T1 → T2 → T3 → T4 → T5 → T6. Each GOAT-gated before the next consumes it.

## Non-goals

- No serving-forward replacement (G5 parity capital is not spent here; §scoping above).
- No training-loss edits in this repo (riir-train Issue 645 owns those).
- No claim of novelty — adoption of published fixes; the diagnosis itself is published prior art (Prieto 2501.04697; Liu 2605.06152; TF #126525/#126748).
