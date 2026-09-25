# Research 588: AWQ — Activation-aware Weight Quantization (Equivalent Per-Channel Scaling, No Backprop)

> **Source:** "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration" — Lin*, Tang*, Tang†, Yang†, Chen, Wang, Xiao, Dang, Gan, Han (MIT / SJTU / NVIDIA / Tsinghua; MIT-IBM). [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) (v6). **MLSys 2024 Best Paper Award.** Code: [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq).
> **Date:** 2026-09-25
> **Status:** DONE — verdict **Gain** (this note + [Issue 886](../.issues/886_activation_diagonal_weight_quant_fit.md) — renumbered 885→886 at commit time: concurrent allocation with the tetris lane `1a05a976`; **no plan** — the stack has no live weight-quant *authoring* consumer; the note discharges riir-clippy Research 192 walk-10's pending quant-lane citation).
> **Related Research:** 467 (RRQ — our own modelless weight-only PTQ; AWQ sits in its "don't take" table), 502 (AVQ2 — the quality-vs-bits doctrine + PTQ-ternary negative controls), 463 (moka quant-error lever audit — "GPTQ/AWQ all use learned corrections" novelty denial), 586 (SmoothQuant/outlier lineage prior-art mention), 083 (asymmetric KV quant — K-side × weight-quant error coupling).
> **Cross-ref:** riir-train Research 086 + Plan 378 (outlier collapse — the only AWQ-*mechanism* row in the corpus pre-this-note) · riir-ai Research 085 (outlier-collapse twin) · riir-clippy Research 192 (walk-10 row 19, arXiv 2609.21450 — **citation obligation discharged below**) · katgpt-rs Issue 883 (the calibration-pass substrate family this rides) · riir-ai Plan 100 / riir-infer `quant/` (the k-quant consume lane).
> **Classification:** Public.

---

## TL;DR

AWQ protects the ~1% of weight channels that matter — selected by **activation** magnitude, not weight magnitude — not by keeping them in FP16 (hardware-unfriendly mixed precision) but by a **mathematically equivalent per-channel rescale**: `y = W·x = (W·diag(s)⁻¹)·(diag(s)·x)`, with `s = s_X^α` (s_X = per-input-channel mean activation magnitude) and a **single α grid-searched over [0,1] in 20 steps — no backprop, no reconstruction, 16 calibration sequences**. It is a PTQ paper that is itself **modelless-tractable** in our taxonomy.

For this stack the mechanism does **not** ship anywhere (confirmed: zero per-channel equivalent-rescale / activation-diagonal-weighted weight-quant fits across all 14 repos) — but neither does a production weight-quant *authoring* path (we consume pre-quantized GGUFs; the Bonsai packs come from the external PrismML fork). The value is therefore **substrate + recipe**: the activation-diagonal fitting primitive for the in-memory ternary authoring path (`katgpt-types::quantize_from_f32` — error-compensated but activation-blind today), the quality recipe for any future GGUF writer of ours (llama.cpp *guards* 2-bit quantization without its activation-aware imatrix), and the weight-side twin of the already-shipped per-head KV sensitivity bridge.

**Distilled for katgpt-rs (modelless, inference-time):** one offline calibration pass collecting per-input-channel activation moments `{mean|x|, E[x²]}` per linear layer feeds two composable, backprop-free fitting upgrades: (a) **weighted-fit** (llama.cpp imatrix spelling — weight the scale/threshold regression by the channel diagonal), (b) **equivalent-rescale** (AWQ spelling — fold `s = s_X^α` into the weights pre-quant, absorb `diag(s)` into the previous op). Both are deterministic, calibration-table-shaped, and BLAKE3-committable — they join the `StaticCalTable`/`fitted_anchor_table` commit family, not the training track.

---

## 1. Paper core findings

**The observation (Table 1).** Under INT3-g128 on OPT-6.7B, keeping 0.1–1% of weight channels in FP16 cuts WikiText PPL 43.2 → 13.0 — but **only when the channels are selected by activation magnitude**. Selection by weight magnitude/L2-norm ≈ random selection. Weight-only quantization error should be measured in *activation-weighted* units: the salient channels are where the input features are large, not where the weights are.

**The error analysis (Eq 1–3).** With `Q(w) = Δ·Round(w/Δ)`, `Δ = max(|w|)/2^{N-1}` (absmax group scale — the exact scale rule our `quantize_from_f32` and every k-quant use), scaling one channel by `s>1`:

- `Err(Q(w·s)·(x/s)) / Err(Q(w)·x) = (Δ′/Δ)·(1/s)`.
- `RoundErr(·) ≈ 0.25` constant (rounding error ~uniform on [0, 0.5]).
- `Δ′ ≈ Δ` — scaling a single channel rarely moves the group max (<5% of groups change Δ for s<2).
- ⇒ the salient channel's relative quantization error shrinks by ~1/s **for free** (mathematically equivalent transform).

**The two-sided catch (Table 2).** Push `s` too far and `Δ′/Δ > 1` for the *non-salient* channels (21.2% of channels at s=4) — their error amplifies. Optimum at s=2 on OPT-6.7B (PPL 23.54 → 11.92). So the scale must balance salient protection against non-salient amplification → search, not closed form.

**The search (Eq 4–5).** Minimize `L(s) = ‖Q(W·diag(s))(diag(s)⁻¹X) − WX‖` over a **1-D family**: `s = s_X^α`, α ∈ [0,1], 20-point grid, one α per model. `diag(s)·X` folds into the previous operator (norm/linear — the SmoothQuant/Outlier-Suppression+ folding). Plus weight clipping to minimize MSE. **No gradient, no reconstruction** — the reason AWQ generalizes where GPTQ overfits: 10× smaller calibration set suffices (16 vs 192 sequences), and cross-domain calibration costs only +0.5–0.6 PPL vs GPTQ's +2.3–4.9.

**Results.** Beats GPTQ (± reorder) on LLaMA/Llama-2 7B–70B, Mistral, Mixtral; first VLM low-bit quant (OpenFlamingo, LLaVA, VILA — near-lossless at INT4-g128); instruction-tuned models (Vicuna GPT-4 eval); coding/math (CodeLlama MBPP, GSM8K). **Orthogonal to GPTQ** (Table 9: AWQ+GPTQ composes at INT2-g64 — the rescale changes *what* the quantizer sees, GPTQ compensates *after*). Industrial adoption: HF Transformers, TensorRT-LLM, vLLM (Marlin kernels), DirectML, Vertex AI, Intel NC, SageMaker, AMD.

**TinyChat (§4).** The systems half: on-the-fly dequant fused into MM/MV kernels, SIMD-aware interleave packing (`w0,w16,w1,w17,…` so a 128-bit NEON register = 32 nibbles, 3 SIMD instructions to unpack), fused QKV/LN, 3.2–3.3× over HF FP16. (This half is *not* novel to us — our fused dequant+GEMV kernels, `simd_lut_dequant`, the Q4_K WGSL shader, and the SIMD packing tables in katgpt-types already occupy this ground. The league-relevant content is the algorithm half.)

## 2. Distillation

### 2.1 The transferable primitive: activation-diagonal weight-quant fitting

Strip the LLM framing; what remains is: **when fitting a quantizer's free parameters (group scales, thresholds, clip ranges), weight the fit by the per-input-channel activation diagonal instead of treating all channels equally.** Two spellings, same signal, composable (AWQ Table 9):

| Spelling | Insertion point | Artifact | Ecosystem exemplar |
|---|---|---|---|
| **Equivalent-rescale** (AWQ) | modify weights pre-quant; fold `diag(s)` into previous op | requantized weights | AWQ `s = s_X^α` |
| **Weighted-fit** (imatrix) | modify the quantizer's *loss* for the same weights | nothing at runtime | llama.cpp imatrix (diagonal of GPTQ's Hessian, weighted-RMSE fit — PR #4861) |

Both consume one offline artifact: per-input-channel `{mean|x|, E[x²]}` per linear layer. That artifact is exactly the shape our calibration substrate already produces in other domains (§2.3).

### 2.2 Vocabulary translation (paper → codebase)

| paper term → codebase spelling | where |
|---|---|
| activation-aware / salient channel → "sensitivity = activation-magnitude variance", Welford streaming saliency | riir-engine `targeted_precision_bridge.rs` (per-**head KV** domain — cites the GPTQ/AWQ family explicitly) |
| calibration statistics → `calibrate_from_stats` (StaticCalTable), `calibrate_eigenbasis` (SpectralQuant), V/K taps (`gemma2_calibration.rs`, Issue 883) — **and `ZeroQatCalibrator`** (the weight-side "calibrator" spelling — see §2.3; found by vocabulary translation, not by AWQ-word greps) | katgpt-attn / katgpt-spectral / katgpt-core / riir-train `zero_qat.rs` |
| per-channel scale `s = s_X^α` → nothing ships (AWQ-vocabulary greps return zero: no `scale_search`, no equivalent-rescale) | the gap = Issue 886 |
| weight clipping search → `quantize_with_scale_rule`'s mean-abs + carry loop (activation-**blind**) | katgpt-types `ternary_group.rs` |
| α grid search → config sweep (e.g. `cal-select-cap` sweeps in riir-reflex) | pattern precedent only |

### 2.3 Signal-diff vs shipped cousins (§3.6 discipline — one read each)

| Shipped cousin | Signal it consumes | Diff vs AWQ |
|---|---|---|
| `targeted_precision_bridge.rs` (riir-engine, `targeted_precision`) | streaming Welford per-head activation variance | **Different domain** (KV cache per-head bits, not weight channels) + **online** vs offline; the AWQ-family *citation* is already in its doc comment — the weight-side twin never got built |
| `StaticCalTable` (katgpt-attn `static_cal`) | per-(layer,head) activation stats → sigmoid scale table, EMA updates | per-head KV gating scales, not weight-quant fitting |
| `SpectralQuant` (katgpt-spectral) | activation eigenbasis → water-fill per-dim bits | KV compression; eigenbasis is a *richer* signal than AWQ's diagonal — but again KV-side |
| riir-train `quant_robust_collimation.rs` `QuantizerKind::Awq4Bit` | nothing (RTN group-128-from-max **simulation**) | explicitly "Not a bit-exact GPTQ/AWQ implementation" — a robustness *test* of collimated LoRA, not a quantizer |
| `katgpt-types::quantize_from_f32` (+ `_pot`, + ternary siblings) | none — mean-abs group scale + carry compensation | **the actual gap**: the error-compensation is GPTQ-flavored but the scale/threshold fit is activation-blind |
| riir-train `zero_qat.rs` `ZeroQatCalibrator` (Plan 255 Ph4, feature `zero_qat_calibrate`) | a **loss** via injected `loss_fn` — perturbative, not channel statistics: `gradient ≈ (loss(s+ε) − loss(s−ε))/2ε`, then `scale -= lr·gradient`, 100 steps × 2 forwards | **the nearest weight-side shipped cousin and the same insertion point Issue 886 P1 targets** (the quantizer's group-scale fit) — but it is gradient descent wearing forward-only clothes: iterative, lr/convergence-knobbed, loss-dependent. The AWQ/imatrix class is its closed-form modelless counterpart: one streaming pass of channel moments, deterministic, BLAKE3-committable, ~100× cheaper (1 pass vs 100 steps × 2 forwards). The diff is the §3.5 Path-0 class (search-the-loss vs compute-the-statistic) — and it *strengthens* the modelless case: the stack's one shipped weight-side scale calibrator is GD-family, so the closed-form seat is genuinely open |
| riir-clippy batch-44 `single-sweep-minmax-for-quant-scale` | — | a *mined kernel rule about* AWQ-class scale prep (the joint (lo,hi) sweep), not a quantizer — the healer already sees AWQ as an ecosystem idiom |

**Verdict of the diff:** the stack ships activation-aware *saliency* (KV-side, three implementations), error-compensated *quantization* (weight-side, activation-blind), and one weight-side scale *calibrator* (ZeroQAT — loss-driven GD-family search). The join — closed-form activation-diagonal-driven weight-quant fitting — ships nowhere. This is not a documented-gap miss; it was simply never in scope while we were a pure GGUF consumer. (Process note, recorded honestly: the first draft's cousin sweep ran AWQ-vocabulary greps and missed ZeroQAT — found under the codebase spelling "calibrator"; the vocabulary-translation rule is load-bearing both directions.)

### 2.4 Fusion (paper × shipped substrate)

1. **AWQ × LoTA ternary merge** (riir-train `lota_ternary.rs`): `ternary_merge_on_grid` merges a trained dense-float LoRA delta into the ternary base and measures `MergeStats::saturation_ratio` — the merge is activation-blind today. An activation-diagonal-weighted merge (error budget preferentially spent on high-`E[x²]` channels) is the AWQ insight in the merge lane; the saturation measurement rig already exists (`plan333_merge_saturation_real.rs`).
2. **AWQ/imatrix × the future GGUF writer**: riir-ai `.benchmarks/rematch/2026-09-18_watch_4090.md` already names "a future GGUF writer of ours". This note is its quality recipe: imatrix-weighted fit as the floor (llama.cpp *refuses/warns* on 2-bit quants without imatrix — PR #4897), AWQ-rescale as the composable second stage (orthogonality, Table 9).
3. **AWQ × Issue 883's calibration pass**: one tapped forward over a corpus can collect V/K token tables (883) **and** the weight-channel diagonal (886) — the same offline pass, two product families; the `fitted_anchor_table.rs` streaming-accumulator pattern is the shared substrate.
4. **AWQ × outlier-collapse security** (Research 200/085, riir-train 086+378): the attack injects outliers that collapse *group-scale-from-max* quantizers (>90% ASR across GPTQ/AWQ/GGUF/NF4/FP4/HQQ/SINQ/AutoRound). Activation-diagonal fitting is the *defensive* mirror of that attack class — the fit notices when a channel's activation statistics contradict its weight statistics. riir-train Issue 503 P3 already proposes "probe-derived channel-outlier detection as an a-priori quant guard"; this is its fitting-side complement.

### 2.5 Reframes, honestly (steps 3–4 of the fusion protocol)

- **Latent-space reframe: weak.** The calibration stats are per-channel first/second moments — statistics over raw activations, not latent ops on belief/functor/shard state. Nothing here is a sigmoid/dot-product latent primitive. Recorded honestly as a reason the tier is Gain, not Super-GOAT.
- **Game-context reframe: none.** No per-NPC behavior signal, crowd pattern, or selling point; quantization quality-at-bits is not a game surface. (Its indirect product surface — quality at lower bits → smaller on-device models — is serving, not game AI.)
- **Consumer/healer reframe: real but already mined.** riir-clippy's kernel_opt corpus carries AWQ-adjacent rules (batch 44), and walk-10 row 19 (arXiv 2609.21450 — exact backprop-free W4A4 error decomposition into an activation-guided weight-compensation term + orthogonal residual, competitive with SpinQuant) is mechanistically the closest *published* cousin to AWQ in our orbit: **cited here as that note instructed** ("cite at the next lossy-surface / quant-lane doc touch").

### 2.6 The 2-bit caveat (read before promising anything on the Bonsai lane)

Published evidence is consistent: at ≤3 bits, per-channel scaling alone is **not** the winning lever. QuIP# ("without fine-tuning or lattice codebooks significantly outperforms OmniQuant and AWQ, which both rely on heuristics"), AQLM (Pareto-optimal <3bpw with learned additive codebooks), QTIP (trellis codecs). The 2-bit winners use **rotations (incoherence processing) + codebook structure + error compensation** — and scaling subsumes poorly. Notably, the stack already ships the 2-bit-winning lever: the Bonsai-2 Hadamard rotation lane (`riir-infer/src/deltanet/rotation.rs`, `prism.hadamard.*` GGUF keys, feature `bonsai2_hadamard`) **is** the incoherence-processing class the literature says dominates AWQ-style scaling at ternary bits. Expect AWQ-class fitting to matter at the Q4_K-class authoring tier (4-bit k-quants), and expect a small-or-negative delta at 2-bit ternary — Issue 886 carries that prior explicitly.

## 3. Verdict

**Gain.**

- **Not Super-GOAT:** Q1 fails (3-year-old Best Paper, 3000+ citations, industrial default; the *mechanism* is maximally published prior art). Q2/Q3 fail (no new capability class for our product; no "our NPCs/systems do X" sentence). Q4 partial (connects calibration substrate + quant lane + security lane — a multiplier, but of substrate, not pillars).
- **Not GOAT:** no provable gain over an *incumbent in our stack* — there is no live weight-quant authoring path to beat; the incumbent (`quantize_from_f32`'s mean-abs fit) is substrate, not a deployed surface. A GOAT gate needs a deployed A/B.
- **Gain, because:** (a) the mechanism does not ship (AWQ-vocabulary greps return zero across all 14 repos; the three nearest shipped cousins each signal-diffed per §2.3 — including the nearest weight-side cousin `ZeroQatCalibrator`, a loss-driven GD-family search rather than a closed-form channel-statistics fit); (b) reverse-grep hits documented gaps it fills: the named future GGUF writer + riir-train Issue 503 P3's proposed quant guard; (c) it is itself modelless (no §3.5 deferral needed — the grid search *is* the modelless search); (d) the corpus gap is real (≈20 incidental AWQ mentions across 11 files workspace-wide, recounted 2026-09-25 — model names, attack-target lists, don't-take tables, batch thresholds, one mined rule — zero mechanism notes); (e) it discharges the walk-10 citation obligation.
- **MOAT gate (katgpt-rs):** "Transformer stack (… quant-aware inference …)" — **in scope**; the fitting primitive is a katgpt-core/katgpt-types-class substrate (public), consumers span riir-infer/riir-train/riir-ai. Routing: note + substrate issue here; no private architectural guide (nothing private to guide — the mechanism is public knowledge).

## 4. Prior art (the §4 sweep, from the session's web research)

- **Successors, search-only class:** GPTQ (2210.17323, Hessian error compensation — the RTN-beating baseline), QuaRot (2404.00456, fixed Hadamard rotations end-to-end), QuIP# (2402.04396, Hadamard incoherence + E8 lattice; beats AWQ/OmniQuant *without* its own fine-tuning), QTIP (2406.11235, trellis codecs), GPTAQ (2025). **Backprop class:** OmniQuant (2308.13137, learnable clipping + equivalent transform — AWQ's transform *learned*), SpinQuant (2405.16406, learned rotations), AQLM (2401.06118, codebook SGD), PV-Tuning/QAT family, OneBit (2402.11295, PTQ-to-ternary needs distillation).
- **llama.cpp/GGUF:** AWQ runtime added PR #4593 (Dec 2023, with an `awq-py` quantizer folding scales into GGUF q4/q2_k), **removed PR #5768 (Sep 2024)** — the ecosystem's activation-aware mechanism is the **imatrix** (PR #4861: store the diagonal `⟨a_i²⟩`, weight the k-quant scale/min fit; ≈ GPTQ's Hessian diagonal per the PR discussion), mandatory-with-warning for ≤2-bit (PR #4897), extended to all k-quants (#4930) and legacy quants (#4969). The Qwen-documented AWQ→GGUF workflow is pre-scaling only (`export_compatible=True` applies scales without quantizing, then convert → llama-quantize). vLLM's llm-compressor is adding an imatrix weighted-MSE observer (RFC #2456, 2026-03) — the idea leaving llama.cpp.
- **Ternary:** BitNet b1.58 (2402.17764) is *trained* ternary, not PTQ; PTQ-to-ternary needs learned transforms/distillation (OneBit) or heavy search (BiLLM 2402.04291). **Our Bonsai is born-ternary** — the AWQ question there is only the per-channel *group scale* fit (Issue 886's P1), with the §2.6 negative prior.

## 5. What we do about it

[Issue 886](../.issues/886_activation_diagonal_weight_quant_fit.md): the activation-diagonal fitting substrate (moment collector + weighted-fit + optional α-rescale, feature-gated, G3 bit-identity when the diagonal is uniform, G1 by per-family retention walk per the lossy-surface law) — filed against the authoring paths that exist, with the honest 2-bit negative prior and the consumer lanes (LoTA merge, future GGUF writer) named as P2. No plan until a consumer opens.

> **Related (added at verdict round 1):** riir-train `zero_qat.rs` — the nearest weight-side shipped cousin (see §2.3).

> **PASS-Redirects (n/a):** Gain verdict — cousins cross-linked in the header instead (467, 502, 463, 586; riir-train 086; riir-clippy 192).
