# Issue 886 — activation-diagonal weight-quant fitting (AWQ/imatrix-class substrate for the authoring paths)

**Status:** IN PROGRESS — P0 DONE + P1 modelless half DONE (`0fb2254d9`, [Bench 896](../.benchmarks/896_act_diagonal_quant_fit_goat.md): G1 synthetic recorded, G2/G3/G4 PASS; both features OPT-IN). P1's model-bound G1 (real-checkpoint per-family conditional retention walk) = riir-infer Issue 014. P2 deferred. Filed from [Research 588](../.research/588_AWQ_Activation_Aware_Weight_Quantization.md) (arXiv:2306.00978); the 2-bit negative prior (§"Honest prior") is NOT settled by the synthetic fixture — see Bench 896 finding 3.

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

- [x] **P0 — moment-collector substrate** (`katgpt-core`, feature `act_channel_moments`) — **DONE `0fb2254d9`**: `ActChannelMoments` (per-layer widths, f64 `{Σ|x|, Σx²}` + count, `observe`/`observe_batch` alloc-free — **0.28–0.32 ns/element**, G4 **0 allocs**) → `freeze()` → `ActChannelDiagonal {mean|x|, E[x²]}`, BLAKE3-committed over a canonical LE image (`to_bytes`/`from_bytes` verify, every byte flip refused, digest pinned); synthetic-diagonal known-answer tests. 883 co-collection = same tapped forward, different tap points (linear INPUTS vs K/V outputs — 883's grand Σx² is not a substitute); documented, no extra hook needed.
- [~] **P1 — weighted-fit on the ternary authoring path** (`katgpt-types`, feature `act_aware_fit`) — **modelless half landed `0fb2254d9`; model-bound G1 retention walk = riir-infer Issue 014.** Shipped `TernaryGroupWeights::quantize_from_f32_act_aware(w, rows, cols, diag: &[f32], fit)` (diagonal as a plain slice — no katgpt-core dep); quantizer core refactored to a per-group scale closure (refactored baseline within ±0.5% of the pre-refactor loop, payload identical). `WeightedMeanAbs` (G3: uniform diagonal ⇒ payload **bit-identical**, pinned by bytes) + `WeightedSearch` (imatrix-class 21-pt grid + weighted LS refit through the carry loop; +12.5% / ≈22× baseline fit cost). **Bench 896 synthetic G1** (held-out `E‖(W−Ŵ)x‖²`, PTQ-of-dense fixtures): blind search alone −20…−27%; the diagonal adds **−54%** over it on planted 1%×20 heavy channels and −14% on a log-normal spread at ternary (INT4 bench-local reference −60% / −26.5%) — mechanism-verified, NOT gain-verified on the born-ternary rotated Bonsai lane. Comparators for the model-bound walk: (i) mean-abs baseline (G3 anchor) + the blind-search control, (ii) the `ZeroQatCalibrator` class (GD-family finite-difference search — cost note in Bench 896; no riir-train dep by boundary). Original spec: G1 by **per-family conditional retention walk** (riir-ai Bench 948 pattern; aggregate PPL alone disqualified); record the measured delta either sign.
- [-] **P2 (deferred until a consumer opens) — (b) spelling + lanes**: α-rescale for the GGUF-writer recipe section; LoTA merge-weighting experiment in riir-train (saturation A/B); the outlier-collapse defensive read (Research 200/085 — diagonal-vs-weight contradiction as an injected-outlier tripwire, riir-train Issue 503 P3's complement).
