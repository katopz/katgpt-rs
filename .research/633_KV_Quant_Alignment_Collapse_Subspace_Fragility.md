# Research 633: Alignment Collapse Under KV Cache Quantization — the Behavioral-Subspace Fragility Law

**Status:** RECORD — distilled 2026-10-24 (alphaxiv interpretability trends digest, user addition request). Verdict: **GAIN** (modelless track; katgpt-rs Issue 960 filed). Training track: **NONE** — the paper's protocol is explicitly training-free; no riir-train plan. Source: Bruce Changlong Xu, Adarsh Kumarappan, Mu Zhou, [arXiv:2606.09864](https://arxiv.org/abs/2606.09864) "Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation" (v2 2026-08-08; submitted 2026-06-01; 61 pp, 9 figs).

## TL;DR

KV cache quantization — the single most-deployed serving memory lever — can **silently destroy a model's trained behavior** while every metric the field ships stays flat: Mistral-7B loses **15.2% of its refusals at only 1.03× perplexity**, and across 11 instruction-tuned models (3.8B–72B) × 5 benchmarks (1,894 prompts) there is **no universal safe bit-width** — sharp, model-specific phase transitions that standard PPL/accuracy metrics cannot see. The root cause is **geometric**: the features carrying the trained behavior occupy a **low-dimensional activation subspace 10²–10³× more vulnerable to quantization noise** than the full representation space that perplexity averages over. The fix is a training-free diagnostic (**Per-Channel Reduction**, PCR) that classifies each model into one of three mechanistic failure modes from ~20 calibration prompts (~35 GPU-min) and thereby predicts the correct mitigation — recovering up to **97% of lost behavior** (97.2% with KIVI), validated in production vLLM serving with FP8 KV cache on NVIDIA GPUs.

This is **external validation #4** of the stack's lossy-surface gate (Issue-750-T3) — and the first one carrying the **mechanism**: not just "aggregates blind, behavior flips" (observed four times now) but *why* — the behavior lives in a subspace the aggregate metric averages away.

## Paper core (read from the v2 abstract + the alphaxiv guide's section — the full 61-pp PDF was NOT read this session; the note is a RECORD, not a replication, and every number below is abstract- or guide-sourced)

| Finding | Number |
|---|---|
| Refusal loss at near-flat PPL | Mistral-7B: −15.2% refusals @ 1.03× PPL (4-bit KV) |
| Per-layer localization | Qwen: compressing **layer 0's KV alone** flips most refusals; keeping layers 0+1 at full precision cuts lost refusals 90.3% → 1.9% (guide §"Diagnosing failures by degrading the computation") |
| Universal safe bit-width | **none** — model-specific phase transitions, invisible to PPL/accuracy |
| Mechanism | safety features live in a low-dim subspace **10²–10³× more quantization-fragile** than the space PPL averages over |
| Diagnostic (PCR) | per-channel reduction ranking; classifies 3 failure modes; correct mitigation predicted on 9/9 primary + 1/1 held-out model from an **independent family**, using **20 calibration prompts** |
| Failure modes | (a) **outlier-crushes-safety** — behavior sits in non-outlier channels collaterally damaged by outlier-driven scale factors; (b) **outlier-as-safety** — behavior overlaps outlier channels, finer granularity cannot rescue; (c) **multi-layer dilution** — behavior spread across many layers, per-layer fixes fail |
| Mitigation | training-free, ~35 GPU-min; up to 97% recovery; generalizes across production quantizers (KIVI 97.2%); beats attention-based bit-allocation methods |
| Production confirmation | vLLM + FP8 KV cache, NVIDIA GPUs |

## Why this matters here — the lossy gate's external-validation ledger, #4

The stack's **Lossy-surface promotion rule** (Issue-750-T3, katgpt-rs/riir-ai AGENTS): *a lossy surface (quantization, compression, any bit-changing transform) gates on deployed-path behavior — per-family, conditional retention — not on bit-identity or aggregate perplexity alone; aggregate metrics can be flat while family-conditional behavior flips.* External confirmations so far:

| # | Source | What it added |
|---|---|---|
| 1 | the founding datum (AGENTS lossy rule text) | 43–45% exact-match flip under BF16 with **flat downstream eval aggregates** |
| 2 | walk #15 pair: 2610.15139 (certification-horizon law) + 2610.14475 (EdgeLLM-Nano LCDF) | *temporal* axis — "report the decode horizon at which a bit-width was certified"; PPL gives the SAME score to methods whose downstream damage differs by orders |
| 3 | mlx-serve distill (riir-refine Batch 242, rule `lossy-route-physical-error-bar-from-noise-floor`) | *magnitude* axis — the physical bar is relative error against the output itself, not a ratio against a near-zero baseline |
| 4 | **this paper (2606.09864)** | **the mechanism** — behavior concentrates in a low-dim subspace 10²–10³× more fragile than the PPL-averaged space; plus a *diagnostic* that predicts the failure mode from 20 prompts, and a *mitigation* that spends precision where the subspace lives |

**Signal-diffs (each prior validation is a different axis, not a duplicate):**

- vs **#2 (2610.15139 horizon law)**: that law says *when* a bit-width stops being certified (decode horizon, feedback instability); this paper says *where* the damage concentrates (channel/layer/subspace geometry) and *which* mitigation direction that geometry implies. A bit-width claim without its horizon is one blind spot; a bit-width claim without its subspace-sensitivity map is the other. Complementary, both needed.
- vs **#3 (noise-floor bar)**: that rule fixes how to *measure* the error (relative to the output's own scale); this paper fixes how to *weight* the error across the representation (a nats-scale aggregate error can hide a 10²–10³× -amplified error inside one behavioral subspace). The noise floor is necessary but subspace-blind.
- vs **#2 (2610.14475 LCDF)**: LCDF is the *metric* that separates methods PPL cannot; PCR is the *diagnostic* that explains and repairs what LCDF detects. Detect → diagnose → mitigate is the complete chain now.

## Three-track verdict

- **Track (a) modelless inference — GAIN.** The entire contribution is serving-time and training-free: a per-channel sensitivity probe (quantize-per-channel vs f16 reference on ~20 prompts, rank by behavioral damage not MSE), a failure-mode classification, and sensitivity-guided mixed-precision KV allocation. All zero-gradient, zero-alloc-able, calibration-only. Filed: **`katgpt-rs/.issues/960_kv_quant_behavioral_subspace_diagnostic.md`**.
- **Track (b) self-adaptive runtime — nothing.** No latent-update mechanism; the paper mutates no state at runtime beyond cache bit-widths chosen at load.
- **Track (c) model-based training — NONE, honestly.** The protocol is explicitly training-free (that is its selling point). No riir-train plan; no recipe worth recording beyond the eval-discipline rows.

**Panel note:** the three-track adversarial panel was **skipped per the skill's own carve-out** — this is a clearly inference-side paper (serving-time KV quantization); the §0 technique-stripping pass is vacuous (the value IS the serving-time diagnostic, nothing hardware- or training-framed to strip), and the abstract carries no optimizer/loss/RL/backprop framing. The training-advocate's only possible finding (fine-tune the behavior back) is named and refused by the paper itself: the training-free protocol recovers up to 97% in 35 GPU-min, and track (c) re-opens only if a replication on our stack shows the diagnostic missing a mode the paper's three don't cover.

## Stack mapping — what this touches, concretely

The KV-quant family **ships here, already crate-ized** (Proposal 001 landed), all over the `QuantizedKVCache` trait in `crates/katgpt-types/src/kv_cache.rs` — 10 in-tree impls, in scope / out of scope per Issue 960's census: **six bit-width quantizers IN SCOPE** — TurboQuant (Research 020, demoted), PlanarQuant/IsoQuant (Research 065 / Plan 100), HybridOctPq + OCTOPUS (**default**, 12.2× compression, MSE −22% to −49% vs SQ) in `crates/katgpt-quant/src/*/kv_cache.rs`, KVarN (Research 159 / Plan 179 / Bench 053 — Hadamard + Sinkhorn variance normalization, eval'd on AIME24/MATH500/HumanEval/IFEval) in `crates/katgpt-kv/src/kvarn/kv_cache.rs`; **SpectralQuant IN SCOPE as the allocation prior art** (`crates/katgpt-spectral/src/spectral_kv_cache.rs` — per-dim variable-bit packing by structural rank, the in-tree precedent T2 extends); **ShardKV in scope as compressor** (`crates/katgpt-kv/src/shard_kv/kv_cache.rs` — asymmetric K/V compression); **OUT OF SCOPE: `osc_kv`** (`crates/katgpt-kv/src/osc_kv.rs` — IMEX oscillatory states, lossy but not a bit-width quantizer) **and `fitted_value_table`** (`crates/katgpt-core/src/fitted_value_table.rs` — Bench 895 value table, not a quantizer).

- **Already covered by our evals:** KVarN's GOAT bench ran *task suites* (IFEval included) — behavioral axes, not PPL alone. The family was never eval'd on PPL alone; the lossy gate's family-conditional-retention law has been binding since Issue 750.
- **The actual gaps (what issue 960 prices):**
  1. **No diagnostic.** If a backend fails a family-conditional retention read, nothing tells us *why* or *which mitigation direction* — we would brute-force bit-width sweeps. PCR's failure-mode classification (outlier-crushes / behavior-in-outliers / multi-layer dilution) is the missing decision layer.
  2. **No behavioral weighting on the allocation.** The shipped allocator is `katgpt-spectral`'s water-fill (`BitAllocator` + `waterfill_bits`, manifest: "Shared by katgpt-kv (ShardKV, KVarN)") weighting bits by STRUCTURAL rank; the paper's layer-0/1 result and per-channel ranking say the bit budget should concentrate where the BEHAVIORAL subspace lives. The natural shape on our stack: T0's per-channel behavioral-damage ranking → a damage weight INPUT to the existing water-fill (homed in `katgpt-spectral`, the shared substrate — no new allocator, no new dependency edge) → GOAT vs uniform **and vs the shipped structural-rank weighting at equal average bits** (the honest controls).
  3. **Eval-discipline intel for the league lanes.** Our league models (Bonsai-2-27B PQ2_0 ternary, Qwen3.8-27B) are base/decision models — the paper's *safety-refusal* axis is their evidence, not ours. Our behavioral axes are decision-suite pick retention (the reflex harness suites), greedy-walk forking (the GDN state-half demotion's kill axis), instruction-following. The law transfers: **any lossy-route promotion measured only on NLL/PPL/logit-aggregates is unaudited** — the suite pick-flip read belongs beside the golden-NLL band (this is the riir-infer Issue-959-T6 oracle's natural extension; recorded here, wired in the issue's T3).

## Game-context reframe (one honest paragraph)

Per-NPC affect/persona state on this stack is **low-dimensional by construction** — the 5 synced affect scalars (valence/arousal/desperation/calm/fear) are sigmoid projections onto learned direction vectors, and the 64-dim HLA/shard latents stay local. The paper's law reads directly: a lossy route (weight quant, KV quant, state quant) that touches the forward these projections read from can move a 10²–10³× -fragile direction subspace while every aggregate loss metric stays flat — and the projected scalars inherit the flip. The standing rules already fence the dangerous half (scalars-only sync, raw-vs-latent boundary, latent features opt-in while raw sync is always-on); what this paper adds is the *retention-read discipline*: when a lossy route touches a forward feeding affect/direction projections, the gate is **direction-retention** (dot-product stability of the projection directions), not the loss curve. Recorded as the game-side reading of the law; no separate filing — no current game path runs affect projections through a lossy-quantized forward (the cached-emotion KV paths carry weight-epoch staleness gates, Issue 938, and refuse stale reads loudly).

## Trends-page landscape (the rest of the digest — one line each, routed)

The user's request targeted the whole alphaxiv interpretability trends page; the rest grades WATCH or NO-FILING:

| Item | Grade | One-line read |
|---|---|---|
| **"Neural Networks Are Already Clean"** (Cammarata/Goodfire, Oct 2026) + LessWrong rebuttal (Oct 17) + Hilton's 432-param example | WATCH | The field's live debate: neurons may be far more monosemantic than the superposition era assumed, but "a plausible pattern in activating examples" is not understanding — the rebuttal's *simulation test* (replace activations with predicted ones; behavior change = the explanation missed something) is structurally the same honesty bar as our G5-parity / bit-identity discipline. No filing: no primitive to extract. |
| Persona vectors (Anthropic, Aug 2025); refusal mediated by a single direction (Arditi, Jul 2024); AxBench (steering baselines beat SAEs, Jan 2025) | WATCH | Direction-space steering works (contrastive mean-difference directions), but AxBench's verdict — simple baselines outperform SAE steering — tempers the SAE route. Nearest our machinery: the sigmoid direction-vector gates (riir-ai affect directions). No consumer today; re-open only if a game-side persona-steering need appears. |
| SAE saturation findings (A is for Absorption; Scaling SAEs; SAEBench) | NO-FILING | Feature-absorption breaks SAE detectors as safety monitors; relevant to labs serving chat models, not to our serving surface. |
| AuditBench (hidden behaviors), Monitoring Monitorability, eval-awareness (Sonnet 4.5 §7.6.4), unfaithful CoT (Turpin) | NO-FILING | Alignment-auditing surfaces for agentic chat models; we serve decision/base models — no refusal/persona surface to audit. |
| Induction heads / circuits / ROME / TRAK / SHAP lineage (2016–2023 canon) | NO-FILING | The canon this guide teaches from; already embedded in how the workspace reasons about residual-stream direction structure. |

## References

- Xu, Kumarappan, Zhou — [arXiv:2606.09864](https://arxiv.org/abs/2606.09864) "Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation" (the primary; all numbers above from the v2 abstract + the guide's per-layer ablation section).
- alphaxiv Interpretability trends digest — <https://www.alphaxiv.org/trends/interpretability> (the input; its guide section independently reproduces the layer-0/layer-1 keep-precision mitigation numbers).
- Internal cousins: Research 159 (KVarN — the family's strongest KV backend, task-suite evals), Research 083 (Asymmetric K/V compression — compound-quantization attention-pattern flips, the behavior-flip precedent inside our own corpus), Research 065 / Plan 100 (rotation family), Bench 053 (KVarN GOAT), Proposal 001 (the `katgpt-quant` crate promotion), riir-refine Research 265 walk-15 rows 2610.15139/2610.14475 (validation #2), riir-refine Batch 242 rule `lossy-route-physical-error-bar-from-noise-floor` (validation #3), riir-infer Issue 959 T6 (the golden-NLL oracle — the aggregate metric this law says is insufficient alone).
- Filed: `katgpt-rs/.issues/960_kv_quant_behavioral_subspace_diagnostic.md` (the modelless PoC).
