# Research 538: GDN W4A4 — Why the Recurrent Half of a Hybrid LLM Survives 4-Bit Quantization

> **Source:** [arXiv:2609.04098](https://arxiv.org/abs/2609.04098) "Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM" — Sergii Kozyrev, Davyd Maiboroda (mnma.ai), 2026-09-03. Checkpoint: `minima-ai/mnma_qwen3.8_27b_nvfp4` (HF). 14 pp.
> **Date:** 2026-09-06
> **Status:** Done — **GAIN** (mechanism + modelless certification diagnostic; actionable items → riir-ai `.issues/879`) **+ PASS repeating on the NVFP4 format track** (Research 439/534 verdict stands: fleet is M3 Max Metal + Ada 4090, no FP4 tensor cores; NVFP4 native needs SM120).
> **Related Research:** 439 (NVFP4 RL — format-class PASS), 534 (FP4 FA4 — same format-class verdict), 200 (quantization outlier collapse), 487 (massive activations → Issue 716 sink guard), 159/049 (KVarN variance-normalized KV), 070 (GDN-2 erase/write — the §5.3 overwrite mechanism's architectural source; records "Ternary-Bonsai is scalar-decay GDN"), 502 (behavior-before-perplexity lossy-surface rule), 516 (TTT-KVB secretly-linear-attention); **riir-clippy walk-#6 row 9 = DAMP [arXiv:2608.27513] — adjacent published prior art: recurrent-STATE PTQ (decay-aware channel allocation; INT8-uniform degrades reasoning)** — different surface (state storage vs projection weights/activations), cited for honesty.
> **Related Issues:** riir-ai `.issues/879_gdn_quant_certification_and_recipe_audit.md` (certification bench + recipe audit + KV-interaction check + league note)
> **Classification:** Public

---

## TL;DR

The paper builds **Minima**: NVFP4 W4A4 on **all 496 linear layers** of Qwen3.8-27B — 48 GDN + 16 attention, gates included — the first recipe to quantize the recurrent half (every public build and the model authors' FP8 release protect GDN in FP8/BF16 on the intuition that recurrence errors compound). Result: matches BF16 within seed noise on six suites (5-task avg −0.52; RULER 100 @ 32K/64K), 2.9× smaller (17.5 GiB), fastest prefill (TTFT 6.90→4.03 s @ 32K; +14–19% @ 8K vs community recipes whose GDN stays FP8). The value for us is not the format (we have no FP4 silicon) but the **four-part mechanism study** — the first mechanistic account of why recurrent-mixer *weights/activations* tolerate aggressive PTQ — plus two serving-stack findings that map directly onto our stack:

1. **The delta rule bounds and erases quantization noise** (state error plateaus at 12.6% flat over 32K; a 1% state impulse dies to 1/e in 80–1,382 steps vs decay-implied horizons of 44K–62K, because each write overwrites the state along the current key — Research 070's erase/write duality doing noise control). **This validates our own Bonsai recipe post-hoc**: Bonsai-27B is a scalar-decay GDN DeltaNet hybrid served at Q2_0/PTQ1_0 ternary — *more* aggressive than this paper's 4-bit — and the mechanism chain is why it works. It also hands us a **modelless certification diagnostic** (lockstep plateau + impulse-half-life vs horizon) we do not ship.
2. **The α-horizon hazard**: direct relative noise on the decay parameter is catastrophic (0.1% multiplicative noise on α → 22% state error, because α≈1 makes δα an enormous horizon change), while the *pre-activation* gate weights are the safest tensors in the model (softplus/exp + sigmoid squash 11% GEMM error → 2.1–2.6% output error). Quantize gate weights; never quantize post-activation gate scalars or `A_log`/`dt_bias`.
3. **Per-module calibration vs fused-GEMM global scales** (§6 finding 1): vLLM/ModelOpt fuse `in_proj_qkv+z` and `b+a` into one GEMM taking the **max of constituent global scales without rescaling the local ones** — paired scales differ 1.82×/2.75× in all 48 layers → silently mis-scaled gates, and the failure is *deceptively plausible* (long-context PPL reads BETTER than BF16 because a broken forget gate holds everything). A kernel_opt distill candidate (D-class loader/contract rule; no live in-stack exposure today — GGML-family formats are per-block-scale-only).
4. **KV × weight-quant interaction**: FP8-KV costs the W4A4 model **3×** the perplexity it costs the BF16 model (+0.41 vs +0.13 @ 32K) unless calibrated per-tensor scales ship — which are performance-free (≤0.4% throughput delta) and recover 83% of the penalty. Maps to our KVarN/sink-guard lineage and warns that Bench 756-style KV-quant must be re-checked *jointly* with weight quant at long context.

**Verdict:** GAIN. The NVFP4 format/kernel track PASSes exactly as Research 439/534 (no FP4 silicon; Plan 568 already excludes NVFP4 stages). The mechanism + diagnostics are new to the corpus (Q1 clean vs DAMP, which covers the state side and concludes calibration is *needed* there — complementary surfaces), actionable, and consumed by our own serving lane → riir-ai Issue 879. Not Super-GOAT (Q3 fails: no customer-facing selling point; the certification is internal quality assurance).

---

## 1. Paper core findings (verified against fetched full text)

### 1.1 The recipe and the headline

- Qwen3.8-27B: hidden 5120, 64 layers = 48 GDN + 16 full attention; per GDN layer 5 quantizable projections (`in_proj_qkv`, `z` (output gate), `a` (decay), `b` (write strength), `out_proj`). Minima quantizes all 496 linears to NVFP4 W4A4 (E2M1 + E4M3 scale per 16-elt block + FP32 per-tensor scale); keeps embeddings, lm_head, GDN conv1d, norms, `A_log`/`dt_bias` in BF16. Calibrated on 128×32K samples.
- Table 1: Minima 5-task avg 85.10 vs BF16 85.62 (−0.52, within seed noise; no pair CI-separated on any task); AIME'25 exactly 26/30 on all four seeds; RULER 100 everywhere. PPL is the honest residual: +0.72 @ 4K, **+0.49 @ 32K — the gap SHRINKS with context**.
- Efficiency: 17.53 GiB (BF16 50.13), decode 1,154 tok/s @32 (BF16 621 — all quantized models land within 4%, decode is weight-bandwidth-bound), TTFT @32K 4.03 s (BF16 6.90; community FP8-GDN recipes 4.39–4.49). **The GDN block is ~23% of decode weight bytes** — quantizing it is where the 7–13% size win over community NVFP4 builds comes from.

### 1.2 The mechanism chain (the transferable content)

Captured real activations of all 48 GDN layers on 8×32K-token documents; single-layer standalone re-implementation verified to 6e-3 (BF16 rounding); NVFP4 **fake quantization** (quantize→dequantize→continue in FP32) injects exactly the 4-bit rounding error.

| # | Finding | Numbers |
|---|---|---|
| (i) | **Inputs are NOT the reason** — GDN reads the same outlier-heavy residual stream (median-layer max/RMS 63.5, out_proj 298.1; kurtosis ~1,560; 10–32% of 16-elt blocks one-hot). 16-elt block scaling **localizes each outlier to its 15 neighbors** (and bounds max/RMS ≤ 4 within a block), so per-token A4 error is *uniform across layer roles* (7.5–9.2%) despite the outliers. Weight error (10.5–11.9%) exceeds activation error everywhere; both flat in position. | Table 2 |
| (ii) | **The protected projections are the safest.** Single-projection W4A4 replays (96 layer×sequence runs, 8K tokens): gates `a`/`b` move layer output only **2.1% / 2.6%** — the two smallest effects — despite 11.0% / 8.5% GEMM error; softplus+exp squash the error to 7.5% on 1−α and sigmoid to 5.2% on β. The error actually carried comes from the plain GEMMs: out 12.7%, qkv 10.4%, z 9.9%. Projection errors are statistically independent — single-projection errors combine **in quadrature** (19.4% predicted vs 19.2% measured all-at-once). Nothing grows along the sequence (19.5% first quarter vs 19.7% last). | Table 3 |
| (iii) | **The recurrence bounds and erases the noise.** Lockstep FP32 (one clean + 11 perturbed trajectories, identical inputs, 32K tokens, 5 layers across depth): full-Minima state error **flat at relS plateau 12.6%** (12.96% @ 256 → 12.31% @ 32,768; max 14.9%). A 1% state impulse at t=1024 falls to 1/e in **80–1,382 steps** and 1/10 in ~2,200–2,900 — vs decay-implied horizons 1/(1−α) of **44K–62K tokens** (mean α = 0.862). The extra erasure is the delta rule itself: every write overwrites the state along the current key direction — old errors are deleted key-by-key, not merely decayed. | Fig 1, Table 6 |
| (iii-H) | **THE HAZARD — parameterization is load-bearing.** Relative noise applied directly to α: **0.1% → 22% state error** (α≈1 ⇒ tiny δα is an enormous change in the horizon 1/(1−α)); 1% → 46% plateau. Noise on β is harmless (1% → 0.4% state error — the delta write is self-correcting). Quantizing `a` produces only 3.6% state error *because the noise lands on the pre-activation* where softplus+exp compress it. "The log-space gate parameterization — chosen for training stability — is precisely what makes the gates quantization-proof at serving time." Caveat (their §9): recurrent mixers with **linearly-parameterized decay** may not enjoy the shielding. | Table 6 |
| (iv) | **End-to-end, weight-quant cost washes out with context; KV cost is the opposite.** 32K NLL by 2K-position bins: weight gap +0.081 nats (first half) → +0.011 (second half) → **negative in the last bins** (Minima beats BF16 at the window end — a filled state absorbs the per-token cost). FP8-KV gap: small, **rises with position**, and **3× larger for the W4A4 model** (+0.41 vs +0.13 @ 32K) — an attention-path interaction, not a weight effect. | Table 4, Fig 2 |

### 1.3 Serving-stack findings (§6 — each silently corrupted a result first)

1. **Per-module calibration vs fused-GEMM scaling** — the load-bearing one for us. `llm-compressor` calibrates one FP32 global scale per linear *module*; vLLM serves the GDN projections **fused** (`in_proj_qkv`+`z` as one NVFP4 GEMM, `b`+`a` as another), taking the **max of the constituent global scales without rescaling the local ones** (ModelOpt path identical). Paired scales differed **1.82× (qkv/z)** and **2.75× (b/a)** in every one of the 48 layers → the served model computed decay/write gates with mis-scaled weights. Failure is deceptively plausible: reasoning degrades moderately (AIME 80.8 vs 86.7) while long-context PPL reads **better than BF16** (flat 6.86 vs true 10.84) — a broken forget gate makes the state hold everything. Repair is checkpoint-side: rewrite each fused group to the shared global scale and **fold the ratio into the per-block E4M3 scales** (94 scale sets, worst ratio 2.81×, re-rounding error ≤6.2%); kernel-vs-reference error 0.35/0.57 → 0.002. Invisible on checkpoints whose fused-adjacent modules carry equal scales (both community checkpoints audit uniform) — which is why it survived: it only bites recipes that quantize the GDN block.
2. Multimodal-composite vs text-only serving path scores long context differently (10.04 vs 10.22 PPL @32K) — serve text-only extractions.
3. Raw-completion harnesses are invalid for thinking models (few-shot prompts without chat template → `<think>` never disabled → ±40–60 pt per-subject swings).
4. **Context inversion**: BF16 scores the same tokens worse inside a 32K request than in 4K windows (6.95 → 10.35; deterministic; reproduced in vLLM and the reference implementation) — a model property; PPL@N comparisons only meaningful within one serving path + window protocol.

### 1.4 KV recipe (§7)

FP8 KV (scale 1.0) on the 16 attention layers moves no task score; capacity +1.8–1.9×; the one systematic cost is PPL @32K (+0.13 BF16, **+0.41 W4A4**). Minima+scales adds static per-tensor calibrated FP8 scales (32 tensors): PPL 10.84 → 10.50 (**83% recovered**, residual below BF16's own uncalibrated cost), RULER unchanged, throughput within 0.4%. Recipe: **"quantize everything, ship KV scales."**

### 1.5 Related work landscape

- **QUASAR** [arXiv:2608.13966] — concurrent QAT checkpoint of the same model (distillation-trained 4-bit); this paper shows **PTQ alone suffices** and explains why.
- **DAMP** [arXiv:2608.27513, riir-clippy walk-#6 row 9] — adjacent prior art: first PTQ study of GDN/KDA recurrent **states** (not weights); INT8-uniform state quant degrades reasoning; decay-aware channel calibration picks high-risk channels. **Complementary surface** — the paper's own "recurrent-mixer quantization has not been studied" claim holds for the *weights/activations* projection side; the state side was already (concurrently) studied and NEEDS calibration, which dovetails with the §7 finding that the state-adjacent surface (KV) needs calibrated scales while the weight side needs nothing.

---

## 2. What transfers to our stack (and what doesn't)

| Paper item | Our surface | Disposition |
|---|---|---|
| NVFP4 W4A4 format, FP4 kernels, SM120 tensor cores | Fleet = M3 Max Metal + Ada 4090 (SM89) | **PASS — format track** (Research 439/534 verdict repeats verbatim; Plan 568 already excludes NVFP4 stages) |
| Mechanism (i)–(iv): block-scale outlier localization, gate squashing, delta-rule plateau/erasure, context-washout | Generic quant-aware-inference knowledge; directly describes OUR serving stack | **Recorded here**; validates Bonsai Q2_0 GDN post-hoc |
| Lockstep plateau + impulse-half-life vs decay horizon | **No shipped instrument** (`impulse\|lockstep\|relS` grep clean; `numeric_stability` lab (Issue 776/802) certifies attention numerics, not recurrent-state quant; riir-train Issue 492 `numeric_drift` probes *training* drift — different axis, signal-diff: training-time weight/grad drift vs inference-time quant-noise propagation through a recurrence) | **GAIN → Issue 879 T1**: certification bench on our GDN kernels + Bonsai GGUF |
| α-horizon hazard (post-activation gate noise catastrophic; pre-activation shielded; `A_log`/`dt_bias` kept BF16) | Our ternary recipe + GGUF loader | **GAIN → Issue 879 T2**: audit that gate weights are quantized as weight matrices only and `A_log`/`dt_bias`/conv1d/norms stay high precision |
| Per-module vs fused-GEMM scale harmonization | kernel_opt corpus (B63 concat-at-upload, B26 merge-dispatches) | **kernel_opt distill candidate (D-class)** — no live in-stack exposure (GGML Q4_K/Q2_0/PQ2_0/PTQ1_0 are per-block-scale formats, no global tensor scale to max-fold; B63 carries per-group scales through the concat), but externally validated in the most popular serving stack; becomes live the day we adopt a globally-calibrated format (NVFP4-ish, per-tensor calibrated KV scales, UE8M0-tensor-scale variants) |
| KV × weight-quant interaction (3× NLL penalty; calibrated scales free, 83% recovered) | Bench 756 (f16 KV: decode −3.4%, verify C-axis −5.7~−6.2%), Bench 802 (f16-KV srel deviation **dilutes** with context — not contradictory: f16 = 10 mantissa bits vs FP8 e4m3 = 3, and deviation-vs-NLL are different metrics), KVarN (variance-normalized = calibrated per-channel), q8kv sink guard (Issue 716/731: f32 sidecar for sink rows) | **GAIN → Issue 879 T3**: one NLL-by-position decomposition on Bonsai (ternary weights) × {f16 KV, calibrated} to close the interaction cell; calibrated-KV-scales posture validated |
| Quadrature composition of independent projection errors | Selective-precision concept (Research 202 TPB, `quant_expert_goat` per-expert precision) | **Design law recorded**: mixed-precision budgets compose in quadrature — allocate per-projection precision by *output* sensitivity, not GEMM error |
| League: qwen3.8-27B W4A4 checkpoint (SM120-only); "the recurrent half is the easy half — quantize it" | perf league (qwen3.8-27B is a tracked opponent; GGUF n_embd=5120/64 layers = same architecture, Bench 833) | **Issue 879 T4**: no manifest change (quant-class is pinned per fairness manifest; an arm change is an owner re-pin); llama.cpp GGUF GDN quantization = opponent-watch item |
| 1/√K query-scale reproducibility trap (Appendix A: omitting → spurious 91.2% = 1−1/√128) | Our deltanet kernels (G1 bit-identical vs references) | Covered — noted for any future GDN reimplementation |

**Game-context reframe (skill §1 step 4):** the (iii) mechanism — a decay+overwrite recurrence bounds injected noise at a plateau and erases impulses faster than its decay horizon — is the same structural property our per-NPC latent state relies on (`evolve_belief` decay+overwrite, HLA moments, shard `style_weights`): overwritten-style state is structurally safe to store coarse. Context sentence; no new action.

**Consumer/healer reframe:** the fused-GEMM scale-harmonization rule is the kernel_opt candidate above; no retrieval/memory/eval surface manifests the mechanism.

---

## 3. Novelty gate (Q1–Q4)

- **Q1 (no prior art):** External — the paper's own claim ("quantization of recurrent-state mixers in large hybrids has, to our knowledge, not been studied") verified by search for the *projection weights/activations* side; **DAMP [2608.27513] is the adjacent published study for the recurrent-STATE side** (recorded in walk #6 before this paper surfaced) — cited, complementary (state: calibration NEEDED; weights: PTQ sufficient), and the *why* mechanism is unclaimed elsewhere. Internal — zero notes cover GDN quantization (439/534 are format-class; grep `2609.04098|GDN.*quant|quantiz.*GDN` clean).
- **Q2 (new behavior class):** Partial — the modelless quant-headroom certification (plateau + impulse-vs-horizon) is a capability the stack lacks; the mechanism itself is validation knowledge.
- **Q3 (product selling point):** No — internal quality/perf assurance, not customer-facing.
- **Q4 (force multiplier):** Yes — connects quant codecs (ternary/Q4_K/KVarN/q8kv), GDN kernels (riir-gpu deltanet/qwen38), serving (Bench 734/742/756/802 lineage), and the league.
- **All 4 YES? No (Q3 fails) → not Super-GOAT. GAIN** (not a bare Pass: four actionable, falsifiable deltas do not ship — T1–T4 in Issue 879).

## 4. Fusion

**B55 sink-sidecar (state/outlier side) × KVarN (calibrated per-channel scales) × this paper (weight side needs nothing; gate parameterization is the shield; fused-scale harmonization at the boundary)** = a complete doctrine for hybrid serving: *quantize every weight matrix; spend calibration effort only where quantization touches state or crosses a fused-scale boundary.* The quadrature law (§1.2 ii) is the budget rule that makes per-projection selective precision (202/439's TPB concept) composable instead of ad-hoc.

## 5. Verdict

**GAIN** (mechanism/diagnostic track) + **PASS repeating** (NVFP4 format track, Research 439/534).

- Filed: riir-ai `.issues/879_gdn_quant_certification_and_recipe_audit.md` (T1 certification bench · T2 recipe/loader audit · T3 KV-interaction cell · T4 league note).
- Not filed: no plan (no feature to build; the diagnostic is bench-ware inside the issue), no format work (no FP4 silicon), no league-doc edit (fairness manifests pin quant-class; re-pins are measurement events, not paper events).
- Honest caveats carried from the paper: one model family/size, one format, 32K measured (128K+ is extrapolation); decode trails RadixArk 2–4% (small-batch activation-quant overhead); the gate-shielding argument depends on the log-space softplus/exp parameterization — recurrent mixers with **linearly-parameterized decay** do not inherit it (our GDN-1-family kernels do; any future GDN-2/KDA gate quantization must re-check per-gate).
- kernel_opt distill candidate (fused-quantized-GEMM scale harmonization) recorded in §2 for the mining loop; not appended to the queue snapshot this session (file under active concurrent edit — the Issue-665 caution).
