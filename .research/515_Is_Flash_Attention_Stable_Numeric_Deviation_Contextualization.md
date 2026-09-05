# Research 515: Is Flash Attention Stable? — Numeric-Deviation Contextualization for Fast-Kernel Gates

> **Source:** "Is Flash Attention Stable?" — Golden, Hsia, Sun, Acun, Hosmer, Lee, DeVito, Johnson, Wei, Brooks, Wu. Meta FAIR + Harvard. [arXiv:2405.02803](https://arxiv.org/abs/2405.02803) (2024-05-05, 6 pages)
> **Date:** 2026-08-28
> **Status:** Active — Gain verdict; probe issue filed (`Issue 697`), riir-train monitor issue filed (riir-train Issue 492)
> **Related Research:** 487 (Massive Activations — sink-aware KV quant, fixed tolerances), 159 (KVarN), 270 (Wasserstein ground-metric caveat for categorical vocab), 355 (LieFlow — Wasserstein-1 as distribution distance)
> **Related Plans:** 418 (MAG — `TransferMetric::Wasserstein1d` ships in `katgpt-core::mag::transfer`)
> **Cross-ref (riir-ai / riir-train):** riir-ai Bench 773 (certifiable-metric lesson) + Issue 753 (f16 KV, 80K ctx, shape-independent tolerances) + Issues 731/716 (misattribution case studies); riir-train Issue 492 (drift probe + divergence ledger + dist-guard axis)
> **Classification:** Public

---

## TL;DR

Meta's framework answers "is a fast kernel optimization numerically safe?" in two modelless steps: **(1)** a numerical microbenchmark (high-level re-implementation of the kernel so precision/tiling internals can be perturbed — the shipped CUDA kernel cannot be) quantifies deviation vs a baseline via max-diff (elementwise bound) + Wasserstein Distance (distribution-aware); **(2)** the deviation is *contextualized* — bounded against known-acceptable references (different-random-init divergence, precision-change divergence) instead of hand-picked absolute tolerances. Case study: FlashAttention shows ~10× more forward-pass deviation than Baseline Attention at BF16, but its downstream weight divergence is **2–5× LESS significant than simply training at FP16 vs FP32**.

**Distilled for katgpt-rs (modelless, measurement-time):**
The transferable primitive is the **contextualization acceptance rule**: accept a kernel/precision/quantization variant's numeric deviation iff it is dominated by references the system demonstrably tolerates — R1 = divergence between two random inits, R2 = divergence caused by a precision change (modelless proxy: quantize→dequant round-trip, honestly labeled a single-step lower bound). This replaces the hand-pinned absolute/relative tolerance bands our GOAT gates currently use — the exact class Bench 773 documented as uncertifiable (`max_rel` with a `1e-6` floor cannot certify any f16-activation design on this logit distribution; the recorded certifiable form, argmax-identity + max_abs band, is still an *absolute* band with no external reference).

---

## 1. Paper Core Findings

1. **Microbenchmark methodology.** FA's CUDA kernel only exposes FP16/BF16 and cannot be perturbed internally → the authors numerically re-implement the tiling + online-softmax in high-level code with knobs: number format (BF16..FP64), sequence length, tile shape (Bc/Br), tile dimension order.
2. **~10× forward deviation at BF16.** Against the FP64-baseline golden, FA at BF16 deviates roughly an order of magnitude more than Baseline Attention at BF16. Deviation shrinks monotonically as mantissa bits grow (an *ordering law*, not a constant).
3. **Sequence-length scaling.** At fixed tile size, deviation grows with sequence length — more tiles → more online-softmax rescaling operations → more accumulated rounding error. Closed-form predictor: rescale count `R = ⌈S/T⌉ − 1`.
4. **Tile-geometry ordering triple.** Larger tile area → LESS deviation (fewer rescales). Swapping tile dimension order → MORE deviation (accumulation-order effect at fixed R). Constraining tiles to be square → no significant change (a free negative control).
5. **Weight-divergence contextualization (the headline).** FA-vs-baseline weight divergence grows over training but is 2–5× less significant than FP16-vs-FP32 training divergence and comparable-or-less than different-random-init divergence. Method: max-diff + Wasserstein Distance between checkpoints of otherwise-identical runs, compared against the two reference bands.
6. **Explicit anti-claim.** The paper does NOT link numeric deviation to loss spikes — "ultimately linking this numeric deviation back to training instability requires further investigation."

**Published prior art / lineage (§4 search, via arXiv export API — web-search quota exhausted this session):**
- **arXiv:2510.04212** (ICLR 2026) "Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention" — the mechanistic successor: low-precision FA training CAN catastrophically fail via low-rank representation emergence × biased rounding errors creating a vicious cycle; ships a minimal rounding-bias fix that stabilizes training. Answers the question this paper only bounds.
- **arXiv:2503.01873** (PASA) — low-precision-safe online-softmax variant (pseudo-average shifting + global recovering); documents f16 overflow (65504) driven by sequence-dimension bias and Q-K "resonance" amplification. Inference-side numerics.
- **arXiv:2606.28116** — mechanism-driven preemptive instability monitors (spectral entropy of QK decomposition under low-precision FA fires thousands of steps before loss collapse). The "monitor" descendant of this paper's proxy methodology.
- **arXiv:2604.12798** (VFA) — reduces rescale chains for SPEED; incidentally also reduces the deviation axis this paper measures.

## 2. Distillation

### 2.1 What ships vs what's new (signal-diff per component)

| Paper component | Coverage in stack | Signal-diff |
|---|---|---|
| Max-diff metric | **COVERED** — G1 gates everywhere (FNV anchors, max_abs bands, bit-identity gates) | Same signal (elementwise bound) |
| Wasserstein-1d metric | **COVERED** — `katgpt-core::mag::transfer::TransferMetric::Wasserstein1d` (Plan 418) | MAG consumes it to score transfer between activation *datasets*; the paper consumes it to measure *weight divergence between two training states*. Metric covered, protocol not. |
| Numerical microbenchmark | **PARTIAL** — every kernel ships a CPU reference twin for parity, but no *perturbable* tiled online-softmax numerics lab (tile shape × dim order × mantissa × seq-len) exists as a standalone artifact | Parity twins answer "does the kernel match the reference"; the lab answers "how does deviation move with design knobs" |
| **Contextualization acceptance rule** | **NOT COVERED — the open primitive.** Every numeric gate we ship pins a hand-picked band: q8kv `5e-3` budget at a fixed shape (riir-ai Issue 716/Bench 691), `1e-2`/`2e-2` parity tolerances, Bench 773's argmax+max_abs certifiable form (absolute) | Bench 773's form has **no external reference**; the acceptance rule replaces "hand-picked band" with "dominated by what the system demonstrably tolerates" — a third, stronger certifiable form |
| Ordering laws (mantissa↑, R = ⌈S/T⌉−1, tile-area↑, dim-order-swap, square-neutral) | **NOT COVERED** as pinned laws; we hold isolated one-off measurements (Bench 773 f16 class ~2.4e-3 max_abs; Issue 753 f16 KV −3.4% decode) | No pinned scaling law or rescale-count predictor anywhere |
| Seq-len-scaled tolerance schedule | **NOT COVERED** — our tolerance gates are shape/length-independent (Issue 716 gates ran n_positions=512; Issue 753 validated f16 KV at fixed shapes yet ships an 80K-ctx path where the paper's law predicts ~5× the rescale count of 16K) | Real gap, cheap to close offline |
| Loss-spike linkage | **ANTI-CLAIM** — paper explicitly declines; 2510.04212 later supplied the mechanism + fix | Ship as API-doc scope-limit + tripwire test |
| Training recipe | **NONE** — zero optimizer/loss/schedule/init content (advocate-confirmed) | Monitors only, → riir-train |

### 2.2 Three-track adversarial panel (Path 0 + advocates)

Run per the mandatory protocol (classification touched training framing). Coordinator merge:

- **No-GD advocate** (14-item ledger): strongest extraction = the acceptance rule (item 1), enabled by the perturbable reference attention (item 2) + FP64 golden protocol (item 3); plus the two-surface metric protocol (max-diff on forwards + W1d on weights — "an elementwise bound alone under-counts where mass moved; a distributional measure alone has no hard bound"), the mantissa-truncation emulator (`truncate_mantissa(f64, bits)` — arbitrary widths, zero deps), the R = ⌈S/T⌉−1 predictor with an ordinal (Spearman) gate, self-calibrating R1/R2 reference builders, and three honest weak/anti extractions (the paper's constants are context-specific footnotes; loss-spike prediction is refused by the paper itself; a faithful trained R2 reference needs training runs — ship the slot + round-trip proxy, labeled lower-bound).
- **Model-based advocate** (honest verdict): **"this paper's value at our scale is measurement methodology, not recipe."** Zero recipe items. Landable monitors: (1) attention numeric-drift probe vs f64 golden with seq×tile sweep — would have caught **Issue 731** (the Q4_K `get_window` KV-layout bug read as "8.72e0 FMA divergence" and survived a multi-day misattribution) and **Issue 716** (outlier channels inflating attention error 584×–18,844×) on day one; (2) a contextualized divergence ledger for mid-campaign kernel/dtype/quant swaps on lossy surfaces (cheap pre-gate complementing the Issue 750 T3 lossy-surface promotion rule); (3) a drift-correlate audit axis on the existing edge_lora dist guard (Bench 494 erank/gaussianity) — causally motivated by 2510.04212's low-rank-emergence mechanism; (4) **the triage demotion rule**: suspect ranking `quantization scheme (ternary/Q4_K) > activation precision (bf16↔f16) > attention kernel numerics` — the paper's own exculpatory result is permission to stop A/B-ing attention kernels for stability reasons at single-GPU scale; kernel work stays throughput-gated with bit-identity correctness gates. Filed as riir-train Issue 492.

### 2.3 Latent-space / game-context reframe (both mandatory steps — honestly thin)

No latent-space angle exists: this is a measurement protocol, not a state operation — there is nothing to project, gate, or freeze. Game-context: the drift probe protects the 20 Hz NPC cognition hot path (attention decode kernels under f16-scale / f16-KV toggles) from silent numeric regressions — a gate-quality concern, not a per-NPC behavior class. No Super-GOAT path; recorded as such rather than forced.

### 2.4 Fusion

- **Bench 773 (riir-ai) × this paper:** the acceptance rule is the *third* certifiable form for f16-activation kernels — after absolute bands (failed: max_rel floor artifact) and argmax-identity + max_abs band (works, but arbitrary) — reference-band dominance is the first form whose threshold is *derived from the system's own demonstrated tolerance* instead of picked.
- **Issue 716 q8kv sink guard × tolerance schedule:** the `5e-3` budget was validated at fixed n_positions; `tol(S) = tol(S₀)·f(S/S₀)` fitted offline from the microbench makes the budget length-aware — directly relevant to the f16 KV 80K-ctx path (Issue 753).
- **Issue 750 T3 lossy-surface promotion rule × divergence ledger:** per-family conditional retention gains a cheap pre-gate: a swap whose checkpoint divergence sits inside both reference bands proceeds; outside both → escalate before burning a full re-gate.
- **Bench 494 dist guard × drift axis:** one more audit column (sampled attention deviation vs f64 golden), recorded alongside erank/gaussianity so a future low-rank-emergence failure is *attributable* on arrival — avoiding another Bench-640-class attribution war.

## 3. Verdict

**Gain.** One-line: a measurement-methodology upgrade for every numeric gate we ship — no new capability class (not Super-GOAT: prior art exists, it's infra, no product selling point), but documented gaps map (Bench 773's tolerance-class failure, Issue 731/716 misattribution cost, Issue 753's shape-independent tolerances at 80K ctx), so not Pass.

**MOAT gate:** katgpt-rs = base measurement primitive (acceptance-rule probe fused over the shipped `mag::Wasserstein1d` + Bench 773's lesson) → public note + Issue 697. riir-train = monitor recipes → Issue 492. riir-ai = gate-layer consumer, referenced, no separate file. Feature-flag discipline applies at implementation time (`numeric_stability`, opt-in; GOAT = the panel's falsifiability gates — planted deviations at 0.1×/1×/10× of reference must land Accept/margin/Reject).

### 4. Numbers to keep (bench-record footnotes, never product thresholds)

~10× (FA-vs-baseline BF16 forward deviation), 2–5× (weight-divergence significance vs precision change) — both context-specific to Meta's model/seq-len/hardware; the *ordering laws* they instantiate are the durable extractions, the constants are not.

## 5. Tile findings vs our kernel fleet (scope note)

Our GEMM family (8×32 / 32×32 / 128×64 ternary simdgroup) holds **bit-identity by construction** across tile shapes (same k-tile accumulation order — Bench 768/790 G1 gates), so the rescale-count law does not apply there. It applies to the **online-softmax attention kernels** (decode, gated prefill, tree-verify) where running-max rescaling is the mechanism. The paper's square-tile-neutral row ships free as a negative control against any future "require square tiling for stability" proposal.

> **PASS-Redirects (synthesis):** Hu [arXiv:2609.04105 "Hardware-Aware FP4 FlashAttention-4"] — Blackwell-FP4 attention distill (katgpt-rs `.research/534`); its four-arm factorial (both MXFP4-P/V training arms diverge ~0.1B tokens while both FP8-P/V arms descend 55.5B, projection-format-independent) is the format-isolating instance of the 2510.04212 mechanism class this note records — a measured prior any future P/V-quantization proposal must cite, and E5M2-for-dO (E4M3 zero-flushes ~97% of dO) is the zero-flush-rate calibration method for any future gradient-format table.
