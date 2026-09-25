# Issue 886 — activation-diagonal weight-quant fitting (AWQ/imatrix-class substrate for the authoring paths)

**Status:** OPEN — research-note-filed gap ([Research 588](../.research/588_AWQ_Activation_Aware_Weight_Quantization.md), arXiv:2306.00978 MLSys 2024 Best Paper). **No plan yet** — no production weight-quant authoring consumer exists; this tracks the substrate so a consumer opens cheap. The 2-bit negative prior is part of the record (§"Honest prior" below) — P1 is a measurement, not a promised win.

## Problem

Three lanes need activation-aware weight-quant fitting; the stack ships none (grep-confirmed — the signal-diff table is Research 588 §2.3):

1. **`katgpt-types::quantize_from_f32` / `quantize_from_f32_pot`** (`crates/katgpt-types/src/ternary_group.rs`, + the ternary/binary/bitcos siblings): error-compensated (carry loop, GPTQ-flavored) but the group scale/threshold fit is **activation-blind mean-abs** — exactly the fit AWQ's Eq-5 rescale and llama.cpp's imatrix diagonal upgrade.
2. **riir-train LoTA ternary merge** (`lota_ternary.rs::ternary_merge_on_grid`, `MergeStats::saturation_ratio`): merging trained dense-float LoRA deltas into the ternary base is activation-blind today; the saturation rig already exists (`plan333_merge_saturation_real.rs`).
3. **The future GGUF writer** (named in riir-ai `.benchmarks/rematch/2026-09-18_watch_4090.md`): llama.cpp *warns/refuses* on ≤2-bit quantization without its imatrix (PR #4897) — activation-aware fitting is the ecosystem's floor requirement for low-bit authoring, and the AWQ-rescale stage composes on top (orthogonality: the paper's Table 9).

**Nearest shipped weight-side cousin (added at verdict round 1):** riir-train `zero_qat.rs` `ZeroQatCalibrator` (Plan 255 Ph4) already sits at the same insertion point — the quantizer's group-scale fit — but is **GD-family search**: central finite differences on an injected loss (`scale -= lr·gradient`, 100 steps × 2 forwards, lr/convergence knobs). This issue's primitive is its **closed-form modelless counterpart** (one streaming pass of channel moments; deterministic; BLAKE3-committable; ~100× cheaper) — the §3.5 Path-0 distinction (search-the-loss vs compute-the-statistic). The closed-form seat is genuinely open; P1 names ZeroQAT as a comparator.

## The primitive

One offline calibration pass (the Issue 883 tapped-forward pattern; the `fitted_anchor_table.rs` streaming-accumulator substrate) collects, per linear layer, the **per-input-channel activation diagonal** `{mean|x_j|, E[x_j²]}`. Two composable, backprop-free consumers:

- **(a) Weighted-fit** (imatrix spelling): weight the scale/threshold/clip regression by the channel diagonal inside `quantize_with_scale_rule` — zero runtime artifact, kernel-identical payloads.
- **(b) Equivalent-rescale** (AWQ spelling): fold `s = s_X^α` (single α, 20-point grid search, per-model) into weights pre-quant; `diag(s)` absorbed by the previous op. Runtime-identical for in-memory repacks; changes payloads.

Deterministic, BLAKE3-committable calibration table (joins the `StaticCalTable` commit family). Feature-gated; `diagonal = uniform` must be bit-identical to the mean-abs baseline (G3).

## Honest prior (carried from Research 588 §2.6)

At 2-bit/ternary, published evidence (QuIP# 2402.04396; AQLM 2401.06118) says per-channel scaling alone is **not** the winning lever — rotations + codebook structure are — and our Bonsai-2 lane already ships the rotation lever (`riir-infer` `bonsai2_hadamard`). **Expect a small-or-negative measured delta at ternary; expect the real payoff at the 4-bit k-quant authoring tier.** P1 exists to make that measurement cheap, not to promise a win; a clean negative is a valid close.

## Tasks

- [ ] **P0 — moment-collector substrate** (`katgpt-core`, Issue-883 pattern): streaming per-input-channel `{mean|x|, E[x²]}` accumulator per linear layer, alloc-free observe (G4), synthetic-diagonal unit tests; optional 883-pass co-collection hook.
- [ ] **P1 — weighted-fit on the ternary authoring path** (`katgpt-types`, feature `act_aware_fit`): (a) spelling on `quantize_with_scale_rule`; G3 bit-identity at uniform diagonal (pinned); G1 quality by **per-family conditional retention walk** (the riir-ai Bench 948 pattern — the lossy-surface law; aggregate PPL alone disqualified); record the measured delta either sign. **Comparators: (i) the activation-blind mean-abs baseline (G3 anchor) and (ii) the shipped `ZeroQatCalibrator` class (GD-family finite-difference search — the same-insertion-point incumbent).** The modelless preference is explicit: closed-form diagonal fit over lr-knobbed loss perturbation — determinism and one-pass cost are the claimed axes, not just accuracy.
- [ ] **P2 (deferred until a consumer opens) — (b) spelling + lanes**: α-rescale for the GGUF-writer recipe section; LoTA merge-weighting experiment in riir-train (saturation A/B); the outlier-collapse defensive read (Research 200/085 — diagonal-vs-weight contradiction as an injected-outlier tripwire, riir-train Issue 503 P3's complement).
