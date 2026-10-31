# LoopUno — Looped-Host Lossless Drafter (Uno × Nanbeige4.2-3B) Validation Distill

**Status:** GAIN — follow-up validation of the already-distilled Uno mechanism (Research 533 / riir-train Plan 376). Plan 376 amended in the same family (effective-step grid, budget ladder, self-generated-data arm, vehicle re-point per the owner model-scope law). No new moat claim: the looped×drafting cell is doubly published (LoopUno trained-drafter; LoopSpec training-free). Update 2026-10-31: the target-conditioned pipelining cell (this table's empty row) is now occupied + distilled — Research 641 (JetSpec++, arXiv:2610.19792: hidden-state-prediction foresight adapter + top-B speculation cache + conformal threshold; recipe amendments landed in Plan 376).

- **Sources:** page `fengsxy.github.io/LoopUno` (Yu, UC Riverside; Xiong, Georgia Tech; Oct 2026 — **no arXiv of its own**); HF `liddlefish/LoopUno-Nanbeige4.2-3B` (step60 + step240 adapter folders); repo `github.com/fengsxy/LoopUno` @ `bfdbf456000520d67861c7e4b9eee18a8ce66bb8` (Apache-2.0, shallow-cloned to `.raw/LoopUno`, removed at landing — page/card claims re-verified at this pin: `training/losses.py` TV loss, `loopuno/deepspeed_zero0.json`, `loopuno/curriculum_b8_60.yaml`, `nanbeige/build_uno_model.py` two-loop builder; `modeling.py` read deferred to the Phase-1 porting-oracle task in Plan 376).
- **Base paper:** arXiv:2609.04010 (Research 533, pin `46fbdb66`). **Base model:** Nanbeige4.2-3B (arXiv:2607.22083) — 22 layers × 2 loops.

## TL;DR

LoopUno replicates Uno's recipe on a **looped** host and measures what our plan left as estimates: (a) the looped host makes draft rows nearly free (8-row draft = 1.15× an AR step with CUDA graphs, 1.29× measured e2e, 1.66× without graphs) because small-batch decode is weight-read bound; (b) the training-cost curve is steep then flat — TPF 1.62 at 8M tokens/14 min → 2.04 at 31M/52 min → 2.18 at 126M/3.5 h, plateau ≈ 2.2; (c) **agentic outputs draft better than free-form math** (BFCL TPF 2.34 > gsm8k 2.04 > SWE replay 2.06 e2e 1.98×/1.47×/1.58×) from math-only training data; (d) lossless holds (accuracy unchanged within noise). The plateau is diagnosed: foreign model-written data scored against the base's own continuations caps per-position acceptance at ≈ 0.54 (vs Qwen3-8B Uno's 0.64 at 14.7B tokens); the named fix is training on the model's **own self-generated responses**. For us this is validation evidence + cost-calibration for Plan 376, not a new mechanism.

## The deltas (distilled — what 533 does not carry)

1. **Looped-host economics.** Nanbeige4.2-3B reads its 22 shared layers twice per token (≈ 6.3B weight reads/step, Qwen3-8B-class step cost at 3B params). Drafting 8 rows costs 1.15× one row (graphs) — so the model's own weight-read floor, not compute, sets the draft price. Loop count is a serving knob (their named future work: draft with one loop, halving draft cost at some acceptance loss — lossless verify untouched).
2. **The one-hour cost curve.** 60 steps × 0.52M tokens = 31M tokens → TPF 2.04 (e2e 1.47× gsm8k). Most of the 60-step gain arrives by step 16 (8M tokens: TPF 1.62, e2e already 1.13×). lr 1e-4 constant (3 warmup steps) reaches in 52 min what the official lr 1e-5 needs ~4 h for — an ~4.6× token-efficiency ratio. α/r = 2048/128 = **16 at ~0.13 epochs** — the aggressive effective step is the fast arm, not a marginal-resume lever.
3. **Agentic acceptance ranking.** Trained on OpenThoughts3 only: BFCL TPF 2.34 (1.98× e2e) **>** gsm8k 2.04 (1.47×) — tool calls are highly structured, so the drafter guesses them more often. SWE-agent trajectory replay 2.06 (1.58×). Accuracy unchanged everywhere (lossless verify: AR 96.0% gsm8k / 87.2% BFCL; Uno within sampling noise).
4. **Plateau + named fix.** Per-position acceptance ≈ 0.54 ceiling on foreign data ("the training data was written by other models, while the drafter is scored against Nanbeige's own continuations") vs 0.64 for Qwen3-8B Uno at 14.7B tokens. Fix to test: train on the base model's self-generated responses.
5. **ZeRO-0 through shared loops** — LoRA grads accumulate across both loop iterations (accumulate, never overwrite — the classic weight-tied-training port bug their fork patches).

## §4 prior-art landscape (the looped × drafting cell)

| Cell | Occupant | Verdict for us |
|---|---|---|
| Looped host, **training-free** drafter | **LoopSpec** (arXiv:2609.17184, Cho/Cui/Kim/Han, 2026-09-15): drafts from early recurrent states, pipelined, selective second proposal from deeper depth, closed-form optimal proposal depths, lossless greedy+sampling, up to 6.83× | Published prior art for the training-free looped cell; a Shape-B-class proposal source if a looped host ever joins the fleet |
| Looped host, **trained** drafter | **LoopUno** (this note) — diffusion-LoRA drafter per Uno | The recipe's second-host validation + cost constants |
| Looped drafting stages (adaptive) | DLoop (HF papers listing): adaptively performs multiple drafting stages before verification | Landscape only |
| Base cell (non-looped, adapter drafter) | Uno 2609.04010 (Research 533) + landscape therein (DFlash, EAGLE-3, DVI, LK losses) | Already distilled |

**Empty after these:** nothing claims our fusion — the league-model drafter *evaluation instrument* (probe-TV speedup predictor + header cost model + bit-identity protocol) and quantized-teacher drafter training on our own served weights. Our claims stay engineering/consumption (GOAT-shaped), never novelty.

## Path 0 / track deltas (each row ends in exactly one disposition)

| LoopUno delta | Track | Disposition |
|---|---|---|
| 1−TV acceptance law confirmed at second host (predicted E[A] = 1 + Σ(1−p)^j = 2.16 at p=0.54, b=8 vs measured TPF 2.04–2.2) | modelless | **Extract** — probe-TV pre-training speedup predictor; gate rides Plan 376 G0 harness (`G0.5` amendment) |
| Weight-read-bound host cost model (draft-row price from GGUF header + context; KV-share crossover earlier on quantized hosts) | modelless | **Extract** — header-derived `c_iter(b, ctx)` scorer; rides 376 Phase-3 wall table |
| Bit-identity lossless protocol (drafter-on ≡ drafter-off transcripts at same seed) | modelless | Already in 376 (G0/G2 lossless gates); wording sharpened with graphs-ON requirement |
| Acceptance telemetry → per-cluster routing/collapse detector | modelless (self-adaptive) | **Record** — serving-side follow-up gated on Phase-3 promotion; not filed now |
| Workload-class acceptance rank (structured > free-form) | modelless | **Extract** as prior — league row schema: acceptance per workload class; Spearman check in G0.5 |
| CUDA-graph block-draft sensitivity (1.66×→1.15×) | modelless | **Extract** — requirement statement + G2(ii) wording in 376; our `RIIR_PREFILL_CUDA_GRAPHS` lane is the consumer |
| 1-loop draft toggle | modelless | **Record** — no loop-counted host in our fleet; future-work line in 376 Shape B |
| Corpus-naturalness probe (teacher-forced top-1 agreement as drafter-TV-ceiling proxy) | modelless | **Extract** — cheap pre-fit data gate, rides 376 self-data arm |
| lr×α/r effective-step, budget ladder, rung cap, self-data arm, chat-mix demotion, ZeRO-0 grad-accum oracle | model-based | **Plan** — 376 amendments (dated section, this family) |
| Nanbeige4.2 adapter consumption as second G0 checkpoint | either | **Discard** — auditable: needs a non-fleet arch loader (multi-day port) to reproduce a 3B curve already published; free calibration numbers banked from the page. Re-open trigger: a qwen-family external adapter appears (+1–2 GPU-h replication cell) |
| LoopSpec training-free looped drafting | modelless | **Record** — no looped host in fleet; cited in 376 Shape B notes |

Caveats pinned (advocate honesty): the 1−TV law is exact for the **sampled** lossless rule — greedy verification's acceptance is argmax-agreement, a different exact functional; gates must name which rule. `p` is prefix-conditional; teacher-forced probe-TV approximates it — the ±10% identity gate validates the approximation. The e2e ≈ TPF/c_iter decomposition fits their numbers at 1.18–1.39× c_iter (band explained by context-length KV share); fused-vs-two-pass iteration accounting is our inference — gates measure the identity, not the mechanism. All LoopUno numbers: one 3B host, 200 problems/point (±10%), simplified AST checker, replay-not-closed-loop agentic.

## Verdicts (§1.5 per track; four-question gate)

**Training track: GAIN → Plan 376 amended.** Nine deltas examined; seven adopted (effective-step grid; Phase-2 8M→31M ladder; Phase-3 rung cap with measured continue-trigger; self-generated-data arm gated 2B→27B; chat-mix demoted to gate-conditional; LoopUno repo as Phase-1 porting oracle; graph-captured fixed-shape draft requirement + G2 wording), one rejected (Nanbeige adapter consumption), one weak note (InfiLoop cross-ref — different loop axis: 626/465's loop is a test-time aggregation recurrence, not depth-looped decode). **Compliance fix rode along:** 376's Phase-2 vehicle `gemma-2-2b-it-f16.gguf` predated the owner's 2026-10-26 model-scope law (no new gemma-2-2b execution cells) — re-pointed to a small-qwen vehicle at Phase-2 start.

**Modelless track: GAIN → recorded here; load-bearing items gated inside 376.** The drafter-evaluation instrument (probe-TV predictor + host cost model + bit-identity protocol) is the lever: it converts drafter choice into a **pure perf axis** — quality is fixed by lossless verify, so gates are speed-only. Not Super-GOAT: (1) prior art exists — the mechanism cell is doubly published (Uno/LoopUno/LoopSpec) and internally distilled (533); (2) no new behavior class for the stack — consumption of a published protocol; (3) selling point is league wall re-arithmetic, an engineering claim; (4) force multiplier yes, but the tier ceiling here is GOAT-engineering via 376, which is where it lives.

## Game-context reframe (mandatory step 4)

The serving path IS the game cognition path: riir-router/riir-gpu serves NPC cognition at 20 Hz; lossless verify means zero cognition-distribution risk — a drafter can only change speed, never behavior (the strongest possible posture for a 20 Hz consumer). The agentic-acceptance finding maps to our workload: tool-call-shaped cognition rollouts (function-calling-class outputs) draft better than free-form — our serving mix is exactly the favored class. RL-freeze finding unchanged from 533 (applies to game-behavior GRPO rollouts).

## Consumer-context reframe (mandatory step 4b — priority ladder)

- **#3 inference-perf league — the primary consumer.** The wall arithmetic re-priced with measured constants: our Shape-A self-draft pays ~2 full weight passes/cycle ⇒ break-even ≈ TPF 2.0 — *at* LoopUno's foreign-data plateau, zero margin; foreign-data linear mode ⇒ mean-len ≈ 2.0–2.2 ⇒ ~2× decode ⇒ **below the ≥4.0 mean-len / ≥100 tok/s wall by construction**. Two consequences: the self-generated-data arm is the named route to the wall (not garnish), and Shape B (cheap proposal source, cycle ≈ 40 ms, break-even ≈ 1.3×) is the quantitatively favored first funding target — the 1.29×/1.15×/1.66× constants are the argument.
- **#1 game runtime:** transitive via serving (above).
- **#2 healer:** no hot local decode loop — unchanged from 533's honest "no healer surface".
- **#5 training:** hosts the amendments (376).

## Fusion (paper deltas × our stack)

1. **Drafter compiler (modelless).** Header math picks the host (`c_iter(b, ctx)` from GGUF weight/KV bytes), probe-TV picks the drafter and block size (predicted E[A] before any training), bit-identity guarantees correctness, acceptance telemetry closes the loop at runtime. The trained LoRA becomes an *optional* upgrade evaluated by the same modelless gates — exactly 376's G0-first structure, now with a predictor gate.
2. **Wall table with measured constants.** Bench-824's 40 ms cycle floor × LoopUno's step-cost band ⇒ the promote/demote tree in 376 Phase-3 G2(vi) gets measured break-evens instead of estimates.
3. **Quantized-teacher nuance.** Our teacher would be the served Q2_0-dequantized forward — the correct target (draft what deployment runs); LoopUno is silent on quantized teachers, so bf16 acceptance priors transfer with a discount — priced by the rung gates, asserted by nobody.

## GOAT / validation protocol

Owned by Plan 376 as amended (G0 + new G0.5 predictor gates → Phase-1 port gates → Phase-2 rung gates with the lr×α/r grid → Phase-3 rung-capped league gates with graphs-ON wording + self-data arm gate). The 376 G-gates carry: τ monotone, τ_chat ≥ 2× modelless, bit-identical greedy stream, determinism, grad-check, wall table with measured break-evens, Δτ(self vs foreign) ≥ +0.1 TPF-class as the 27B self-data funding trigger.

## Priority routing + files

- `katgpt-rs/.research/638_LoopUno_Looped_Host_Lossless_Drafter.md` — this note.
- `riir-train/.plans/376_uno_tv_self_drafter_distillation.md` — amended (LoopUno validation section + in-place vehicle/risks fixes).
- `katgpt-rs/.research/533_Uno_Psi_Spec_Diffusion_Augmented_LLM.md` — one-line follow-up pointer.
- Serving integration stays riir-ai territory (filed at Phase-3 start per 376's own rule); no new issue files this session — the amendments live in the plan they amend.
