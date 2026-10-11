# Research 602: LoopCD — Recurrent-Depth Contrast Guidance for Looped Inference

> **Source:** "Decoding Looped Transformers Better for (Almost) Free" — Liu, Zheng, Chen, Dilip, Bai, Jiao, Wang, Zhang (Apple), arXiv:2610.02185, 2026-10-01. **Twin prior art:** "LoopCD: Loop-wise Contrastive Decoding for Improving Reasoning in Looped Language Models" — Yu, So, Park, arXiv:2609.24196 (EMNLP 2026 Main, 2026-09-21) — same name, same class, published 10 days earlier.
> **Date:** 2026-10-02
> **Status:** Active
> **Related Plans:** katgpt-rs 136 (loop modes `WeightShared`/`TrainingFree`), 283 (`AdvantageMarginGate` self-advantage exit), 304 (`GainCostLoopHalter`), 324 (Ouro loop KV rewind), 731 T5 (cadence-config for `LoopResidualExit`); riir-ai 587 (`deliberation_budget`), Issue 1008 (`multi_hypothesis_deliberation`)
> **Related Research:** katgpt-rs 578 (Probe Guidance for Language Flows, arXiv:2609.19356 — the in-stack strong−weak extrapolation prior; Issue 865 landed it as `probe_guidance`)
> **Cross-ref (riir-ai):** Issue 1025 (game-side deliberation contrast analog — fusion idea, novelty TBD)
> **Classification:** Public

---

## TL;DR

Looped transformers get a free weak-to-strong prediction pair: loop *k* and loop *R* of the same shared block decode the same next token, so `z′ = z_R + ω(z_R − z_k)` (logit space) or `h′ = h_R + ω(h_R − h_k)` (hidden space, zero extra head passes) re-ranks close decisions by extrapolating along the direction of recurrent refinement — training-free, no auxiliary model. Headline: guidance at full depth raises Ouro-2.6B-Thinking AIME24 pass@1 61.88→73.33; guidance at **half** the loop count matches or beats the unguided full-depth model, cutting forward FLOPs 22.5–48.2%. For us: the strong−weak combine arithmetic **already ships** (`probe_guidance`, Issue 865 — measured NEGATIVE twice on the saturated mini dLLM lane, Benches 847+850), and the unsampled delta is threefold: the weak side **re-sourced to a completed loop iteration** of the weight-shared block (modelless, aligned by construction — vs a trained connector probe or dropout tap), the **adaptive margin gate** `ω = ω_max[1 − (p₁ − p₂)]`, and a **decision-level runtime settle-exit** with contrast recovery.

**Distilled for katgpt-rs (modelless, inference-time):**

1. **Recurrent-depth contrast guidance** — the negative-weight member of the shipped `product_policy_log` blend family: `z′ = z_R + ω(z_R − z_k)`, probability form `p′(v) ∝ p_s(v)·(p_s(v)/p_w(v))^ω`.
2. **Adaptive margin gating** — `ω = ω_max·[1 − (p_(1) − p_(2))]` (top-two probability margin): full strength on contested tokens, withheld on settled ones. Stabilizes generation where fixed ω overshoots.
3. **Hidden-space variant** — `h′ = h_R + ω(h_R − h_k)` before coda/head: one head pass total; for coda-free stacks it is provably ≈ the *orthogonal re-ranking component* of the logit update (cos 0.96 measured in the paper), which is the component that carries the gain.
4. **Settle-exit certificate** — early-vs-late loop *decision* agreement as a runtime exit signal complementing the shipped representation-residual exits (`LoopResidualExit`, `GainCostLoopHalter`, `ConditionalGate`).

**Lineage warning (read before building):** the combine arithmetic is not new here — Issue 865 / Research 578 landed it as `probe_guidance` on the D2F lane and the GOAT gate measured **NEGATIVE** twice on the mini lane (Benches 847 + 850; the negative is pinned as the inverted-bar gate `tests/probe_guidance_goat.rs`). §2.35 carries the signal-diff and why the negatives constrain rather than close the loop-index claim.

---

## 1. Paper Core Findings

- **Setup.** Looped transformer: `h_0 →^f h_1 →^f … →^f h_R`, decode `z_R = lm_head(coda(h_R))`. Standard decoding discards `h_1 … h_{R−1}`, each of which is a valid, fully-projected prediction in the same vocabulary/feature space.
- **LoopCD-Logits (Eq. 3):** `z_1 = lm_head(coda(h_1))`, `z′ = z_R + ω(z_R − z_1)`. Costs one extra readout pass (1.01–1.31× forward FLOPs; 1.02× for Ouro/Huginn).
- **LoopCD-Hidden (Eq. 2):** `h′ = h_R + ω(h_R − h_1)` fed through coda+head **once** — 1.000× FLOPs. Retains most of the logit gain behind a shallow coda (Huginn: +0.83 vs +0.74 seven-benchmark mean) and *exceeds* it on code generation (Huginn HumanEval 22.56→31.71).
- **Full-depth gains:** Ouro-2.6B-Thinking AIME24 pass@1 61.88→73.33 (adaptive); AIME25 49.58→56.88; OlympiadBench 64.05→67.29. Every evaluated model×suite mean improves.
- **Half-depth parity (the compute story):** halving loops costs 0.17–1.29 points unguided; LoopCD at half depth recovers it and beats full-depth unguided in 4/6 settings (Huginn 32→16: **+1.02** logits, +0.51 hidden). FLOPs at half depth incl. guidance: 0.52–0.78× → **22.5–48.2% removed**.
- **Adaptive strength (Eq. 4):** `ω = ω_max[1 − (p_(1) − p_(2))]` — margin gating keeps gains positive across `ω_max ∈ [0.5, 1.0]` where fixed ω overshoots (least-confident-fifth gains recede at ω = 1.0: +6.4→+3.9 for Ouro-1.4B).
- **Where the gain comes from (§5):**
  - Gain **tracks reference disagreement, not reference accuracy**: Huginn k=1 (51% of answers disagree with final) → +3.07 points; k=16 (7% disagree) → +0.17. Parcae's k=6 is *more accurate* standalone than final but nearly converged → only +1.02.
  - The contrast decomposes into a **parallel component** (pure temperature rescale, order-preserving) and an **orthogonal component** (pure re-ranking). The orthogonal part alone matches or beats the full update (HellaSwag +1.72 vs +0.76 for Ouro-1.4B). LoopCD-Hidden ≈ orthogonal re-ranking (F.2: verdict divergence 205/10,042 vs the re-ranking-only update; cosine 0.96).
  - Gains concentrate on **close decisions**: least-confident fifth +6.4 to +13.3 points, most-confident fifth ≤ +0.4; only 3.9–14.9% of answers flip.
  - **First iteration is the best logit reference** across all swept models. But **intra-pass (mid-iteration) states are invalid** — prediction JSD spikes 0.57–0.84 bits *inside* passes; sweeping internal layers as DoLa-style references *drops* ARC-C by up to 24.9 points, while completed-iteration `h_1` gains +2.1. Spatial-layer contrast (DoLa's axis) structurally fails on looped archs; only completed loop states are trained exit points.
  - Huginn's hidden form needs a post-burn-in reference (h6/h7 at R=32/16): its Gaussian-init noise dominates the `h_1` hidden contrast (path length 6× the net displacement), while the *logit* form's `h_1` stays best because the coda projects the noise away.
  - Strength windows: scoring tolerates ω ≈ 0.5; **generation needs ω ∈ [0.2, 0.3]** (early token shifts compound); negative ω interpolates toward the weak reference and universally degrades.
- **Class prior art (their own §Related Work + our searches):** contrastive decoding needs an aligned weak reference — auxiliary model (CD, Li et al. 2023), perturbed context (context-aware decoding, Shi et al. 2023; "thinking by subtraction" 2602.18232), or intermediate *spatial layers* (DoLa, Chuang et al. 2024; ALW, Zhou et al. 2025). Diffusion analog: classifier-free guidance / autoguidance (Ho & Salimans; Karras et al. 2024). The **loop-index axis** is the novelty both concurrent papers claim; adaptive loop *counts* exist separately (LoopCoder-v2 arXiv:2606.18023 "only loop once"; RecurTrace arXiv:2609.03379 adaptive loop-time, 2.0 loops avg; Yang et al. 2026 parallel refinement) but without the contrast-recovery mechanism.

---

## 2. Distillation

### 2.1 The primitive (modelless, decode-time)

```
Guided decode over a looped forward (weight-shared or training-free mode):

  logit mode:   z_k = head(h_k)            # one extra head pass (scratch)
                z′  = z_R + ω·(z_R − z_k)  # subtractive contrast — NEGATIVE weight on the weak pass
                ω   = fixed  |  ω_max·(1 − (p_(1) − p_(2)))   # adaptive margin gate
  hidden mode:  h′  = h_R + ω·(h_R − h_k)  # blend BEFORE head; single head pass

  decode argmax/sample from softmax(z′).
```

Zero training, deterministic, O(V) or O(d) extra work per decode step, zero-alloc with scratch buffers. `ω = 0` bit-recovers the unguided path.

### 2.2 Substrate mapping — what ships vs the delta

Internal sweep (7 repos, vocabulary translation; full hit table in the session record):

| Paper component | Shipped analog | Signal-diff (what's missing) |
|---|---|---|
| Looped transformer forward | `forward_looped` (`katgpt-rs/src/transformer/variants.rs` L132-250, 580-620) + `LoopMode::WeightShared{loop_count}` / `TrainingFree` (`katgpt-types/src/enums.rs` L414) + ELT elastic loop-count override (arXiv:2604.09168) + Ouro-style KV rewind (`katgpt-attn/src/mla.rs` L328, Plan 324) | structure complete; nothing consumes the intermediate readouts at *decode* time |
| Extra readout pass on early state | `LoopDeepRun::capture_logits` **field** (`src/transformer/loop_deep.rs` L146) — one `lm_head` matmul per loop snapshot into `stats.logit_snapshot_buf`; plus the `AdvantageMarginGate` scratch readout | exists as **instrumentation**, not wired into token selection |
| Contrast/blend of two depth readouts | **`probe_guidance` (Issue 865, feature-gated)**: affine strong−weak combine `x̂_s + (λ−1)(x̂_s − probe(h_early))` at the D2F decode step (`d2f_context.rs::apply_probe_guidance`, zero-alloc, λ=1 bit-identical); weak sides = `MlpWeakProbe` (trained connector, riir-train artifact) or `DropoutHeadProbe` (modelless dropout-masked tap) | **the contrast arithmetic is sampled and shipped** — but on the dLLM denoise lane with a tap/connector weak side, NOT on the looped forward with a completed-iteration weak side. Measured **NEGATIVE twice on the mini lane**: Bench 847 (gain temperature-reachable, dominated by unguided T=1.0 at matched diversity; pinned as inverted-bar gate `tests/probe_guidance_goat.rs`) and Bench 850 ("the mini lane cannot produce a guidance win with any modelless weak side, in any trunk or decode regime tested"; Bonsai-scale re-open the only path) |
| The same contrast *signal*, different use | `AdvantageMarginGate::should_recurse` (same file L269-397; wired `weight_shared_advantage_gate`, variants.rs L174/L580): `margin = A(y*) − E[A]`, `A = log_softmax(post) − log_softmax(pre)`, exit when margin < 0.01 (reported 5×+ forward reduction at 100% argmax quality, Plan 283) | exits and **accepts the intermediate answer** when further loops are dead compute. LoopCD instead **sharpens the final answer** and recovers full-depth quality where loops are *not* settled (close decisions). Skip-settled-compute ≠ settle-close-decisions. |
| Early-vs-late decision agreement | `agreement_exit` (`katgpt-core/src/loop_depth_probe.rs` L237), `kl_profile`/`effective_depth` (same file, Issue 898) | **offline calibration oracles only** — every *runtime* exit consumes representation residuals (`LoopResidualExit` ‖Δh‖, `GainCostLoopHalter` step-norm/cosine, `ConditionalGate` hidden cosine, riir-ai CCE `‖Δρ‖₁`) |
| Adaptive loop-count policy | game-side only: `deliberation_budget`/`budgeted_horizon` (riir-ai Plan 587, flip-EMA-driven); DDTree `early_exit_patience/gap` (tree axis) | decode-side quality-preserving depth reduction missing |
| Margin-gated guidance strength | top-two margin machinery exists inside `AdvantageMarginGate`; `SigmoidGateCalibrator`-style gates elsewhere | margin→strength map (Eq. 4) unsampled |

Other repos: riir-infer GDN is token-recurrent (no depth iteration); riir-reflex is single-pass (its weak/strong shape is cascade *routing*, never logit combination); riir-refine `heal_fixpoint` exits on edit-set emptiness (no decision signal); riir-neuron-db/riir-chain: no hits.

### 2.3 Novelty pin (§1.5 precondition form)

> **Completed-loop-iteration contrast guidance** — the `probe_guidance` combine (`z_R + ω(z_R − z_weak)`) with its weak side **re-sourced to a completed loop iteration `k` of the same weight-shared block, decoded by the same head (modelless, zero training, aligned by construction — vs the 865 lane's trained connector probe or dropout tap on the dLLM denoise path)** — plus the **adaptive margin gate** `ω = ω_max·[1 − (p_(1) − p_(2))]` (unsampled anywhere) and a **decision-level runtime settle-exit** with contrast recovery on the looped AR lane (`forward_looped` `WeightShared`/`TrainingFree`), distinguished from `AdvantageMarginGate` by what it does with the disagreement: it re-ranks the *final* prediction and lets loop budget drop with quality *recovered*, rather than exiting early and accepting the intermediate answer when loops are provably dead.

In-stack: the combine arithmetic ships (`probe_guidance`); the loop-index weak side, the margin gate, and the runtime decision-level exit do not (sweep + verdict-review correction). Published: the class is **concurrently published prior art** (2609.24196 + 2610.02185). We claim the in-stack delta, never class novelty.

### 2.35 The 865/847/850 lineage — why the negatives do not kill (and do not prove) the loop-index claim

The two in-house negatives are **consistent with LoopCD's own mechanism account**, which sharpens rather than refutes:
- Bench 850's root cause — "the mini lane cannot produce a guidance win… the strict-config decode uncertainty is unstructured" — is the paper's §5.1/§F.1 prediction: gain tracks **reference disagreement** and lands on **close decisions**; a saturated mini trunk has neither, so any contrast is temperature-shaped noise (exactly 847's finding).
- What the negatives DID kill is *"any modelless weak side on a saturated lane"*. They did not test: (a) a weak side aligned by construction (same block, same head, same prefix — only compute differs, vs a connector approximation or dropout perturbation), (b) a lane with structured depth-computation gaps (the loop lane's depth→accuracy slope is asserted before any guidance leg, Plan 617 T3.1), (c) margin-gated strength.
- The Bonsai-scale re-open named by Bench 850 and the loop-lane gate in Plan 617 are the same open question at different lanes. T3.3(c) adjudicates against `probe_guidance` as a third incumbent — if the loop-index weak side beats the dropout-tap weak side on the close-decision stratum, that is the first in-stack evidence that weak-side *alignment source* matters; if not, the negative extends to a third lane and the class stays closed here until a non-saturated trunk exists.

Tier consequence: two class negatives on our substrate cap this at **Gain** — the plan is falsifiable and cheap, the mechanism account predicts a specific fixture where it should win, but the gain is unproven and the priors are negative (§3).

**UPDATE 2026-10-03 — Bench 906 (Plan 617 T3/T4.1): the negative extends to a third lane, and this one closes the class here.** The adjudication the lineage above called for ran on a fixture that PROVABLY has everything the mechanism account demands (hand-constructed looped fixture, non-degeneracy gate committed before any guided arm: depth→accuracy slope 0.50, close-decision stratum 19, disagreement 50%): guided LOST on every G1 leg (guided-full 0.8028 vs unguided 1.0000; the cheap-weak control matched unguided at 1.0000 while the depth arm destroyed — the alignment-source story refuted in the harmful direction). The posture sweep (24 postures) isolates the mechanism: at ref_loop ≥ 6 the guidance is EXACTLY neutral at every strength — neutrality by vanishing contrast (z_k already agrees at the argmax), never superiority. The k→R contrast amplifies its DOMINANT difference component; under any biased head that component is bias/drift, not decision signal — and when post-burn-in removes it, a monotone-refinement fixture needs no correction. The mechanism is squeezed between "amplifies noise when it has signal" and "has no signal when it could tie". Reopen trigger (the only regime left): a real looped checkpoint with measured non-monotone close-decision churn — post-burn-in contrasts carrying corrective signal. Full record: `.benchmarks/906_loop_guidance_goat.md`; verdict pinned as the inverted-bar gate `tests/loop_guidance_goat.rs`. Tier consequence: three negatives — the class is closed on this stack; the F2 deliberation analog (riir-ai Issue 1025) inherits this negative as prior art AGAINST, not just as lineage.

### 2.4 Fusion

**F1 — Guided settle-exit (primary, this repo).** Fuse the contrast term with the shipped exit family: exit when the early-vs-current readout argmax has been stable for a window after `d_min` (decision-level certificate — the runtime shape `agreement_exit` computes offline), then apply the contrast to the exit-iteration prediction. Expected behavior: exits *earlier* than residual-based halters on settled tokens, and unlike truncation, the half-depth answer keeps full-depth quality because close decisions were re-ranked. Composes with `GainCostLoopHalter` (keep it as the backstop; precedence documented in the plan).

**F2 — Game-side deliberation analog (riir-ai Issue 1025).** The principle maps onto NPC deliberation: the first deliberation iteration's hypothesis is a weak prediction aligned with the final; (a) first-vs-final contrast as a tie-break weight on close decisions, (b) disagreement magnitude as a think-budget certificate complementing the flip EMA (Plan 587), (c) halve deliberation depth at quality parity. The deliberation loop is not a transformer loop, so this is an analog, not a port — novelty TBD, filed as an issue per the no-candidate-escape rule.

**F3 — Not a fit (recorded honestly).** riir-reflex serves point decisions (choice/score), not open-ended looped generation; the cascade abstain→escalate already implements a different weak/strong contract. riir-infer GDN recurrence is per-token state, not depth. Revisit if a looped-arch serving surface appears.

### 2.5 Track classification (Path 0 note)

Inference-side paper: no optimizer/loss/RL/backprop content; every component (contrast arithmetic, margin gate, exit criterion) is closed-form and modelless. No riir-train deferral question arises; the three-track panel is skipped per the skill's clearly-inference-side carve-out. No quant/kernel/loader surface → no riir-infer routing.

---

## 3. Verdict

**Tier: Gain** (revised down from the first-draft GOAT after the verdict review surfaced the Issue-865 lineage) — a real in-stack delta with a falsifiable plan, capped by two measured class negatives on the same combine arithmetic (Benches 847 + 850): the gain is unproven and the priors are negative. The plan (617) is the Gain-tier vehicle — feature flag, no promotion claim — and its T3 gate doubles as the loop-lane adjudication: **tier upgrades to GOAT only at the Bench-617 record if the loop-index weak side beats all three incumbents on a non-saturated fixture.**

Gate scoring (all four required for Super-GOAT):

1. **No prior art in-stack?** NO at the arithmetic level — `probe_guidance` (Issue 865 / Research 578 / Research 68 §7.2) ships the strong−weak affine combine; YES at the delta level (loop-index weak side, margin gate, runtime decision-level exit).
2. **New behavior class vs shipped incumbents?** YES — quality-*recovering* depth reduction. Shipped exits (`AdvantageMarginGate`, `LoopResidualExit`, `GainCostLoopHalter`) save compute only when extra loops provably change nothing; the guided settle-exit extends the frontier into the *unsettled* region. (Class-level prior art exists — the two LoopCD papers — and in-class negatives exist — 847/850.)
3. **Product selling point?** NO — needs a production looped consumer (looped mode is a Plan-136 research surface today) AND a passing gate; both fail today.
4. **Force multiplier?** YES — looped forward + probe_guidance seam vocabulary + exit family + margin machinery; ≥2 pillars.

**MOAT gate (§1.6):** `katgpt-rs` — in scope (modelless inference primitive, decode slot, looped-forward surface it already owns). Routing: open primitive + plan here; game analog → riir-ai Issue 1025; no chain/shard/train/infer/reflex surface.

**GOAT gate + Report-the-Floor disposition:** the primitive shapes a point decision (argmax/sample over contrasted logits); it claims no interval, coverage, or calibrated-uncertainty quantity, so the conformal-naive floor (Plan 340) is **not applicable** — the G1 axis is **greedy argmax accuracy** vs baselines. Metric pin (the Bench-847 lesson): if ANY sampled or pass@k metric is used, the unguided temperature front at matched diversity is a mandatory baseline — 847 killed 865's +0.21 exactly this way. Gate design in Plan 617 (three incumbents, cheap-weak control, stratum floor); promotion stays opt-in `loop_guidance` until a real looped consumer exists AND the gate passes on a non-saturated fixture.

### §4 prior-art record (searches executed)

- Internal-first: workspace grep for `2610.02185` + `LoopCD|Decoding Looped Transformers` → **0 hits** (no prior distillation of these papers). **Correction (verdict review):** the first sweep's class vocabulary missed the closest in-stack prior — **Research 578 (arXiv:2609.19356 Probe Guidance for Language Flows) and its landed feature `probe_guidance` (Issue 865)**, plus Research 68 §7.2 (the formula documented, unlanded, even earlier). §2.2/§2.35 now carry the signal-diff and the Bench-847/850 negative lineage.
- Headline verbatim + landscape: found the **twin paper 2609.24196** (same name, EMNLP 2026 Main, 10 days earlier — contrast earlier-iteration logits vs last refined iteration, + early-exit combination); RecurTrace 2609.03379 (adaptive loop-time memory); "Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning" 2605.26733 (loop instability); LoopCoder-v2 2606.18023 (loop-once); recurrent-depth latent-reasoning landscape (Huginn/Ouro/Astra coverage).
- Component: DoLa (Chuang 2024, spatial-axis — the paper itself shows this axis fails on looped archs, §E.3), ALW (Zhou 2025), CD (Li 2023), context-aware (Shi 2023), "Thinking by Subtraction" (2602.18232, confidence-driven CD), autoguidance (Karras 2024).
- Conclusion: the loop-index contrast class is published (twice, concurrently); the strong−weak combine arithmetic **already ships** (`probe_guidance`, measured negative ×2 on the mini dLLM lane); our claim is scoped to the in-stack delta (§2.3) — loop-index weak side on the looped AR lane, margin gate, decision-level settle-exit — and §2.35 records why the negatives constrain but do not close it. Nothing found that implements decision-agreement-guided settle-exit with contrast recovery in any form — that fusion is ours to build (or refute).

### Validation protocol (summary; full detail in Plan 617)

Toy looped fixture with **hand-constructed modelless weights** carrying known iterative semantics (no in-repo training; a BLAKE3-committed riir-train artifact is the documented fallback only if hand-construction cannot produce the depth slope) → assert the fixture is non-degenerate BEFORE any guidance leg: depth→accuracy slope exists AND the close-decision stratum is non-empty with a minimum count (a saturated toy has no close decisions — the 847 failure mode; T3.3(c) would pass vacuously over zero rows). G1 = **greedy argmax** ladder: guided-full ≥ unguided-full; guided-half ≥ unguided-full; on the close-decision stratum: guided > **all three incumbents** (`probe_guidance`-style combine with a dropout-tap weak side, `product_policy_log` positive blend, `AdvantageMarginGate` exit-only) plus the **cheap-weak control** (a DropoutHeadProbe-style weak side on `z_R` — separates "any contrast helps" from "depth contrast helps", the paper's own disagreement-tracking claim). G2 includes the **per-loop readout cost leg** (the settle-exit needs a head readout per loop; at real vocabularies that rivals the block — bounded candidate-set readout or larger snapshot stride is part of the design, and net-FLOP accounting must precede any half-depth claim). G3 flag-off bit-identity + incumbent exits unchanged, G4 zero-alloc. Caveat recorded up front: no real looped checkpoint exists on this stack; toy-domain evidence only — same evidence posture as Plan 136's loop modes, now with the 847/850 priors making the null hypothesis live rather than formal.

### P0–P3

- **P0** — primitive: `loop_guidance` module + config + contrast/margin-gate + unit tests (ω=0 bit-identity, adaptive concentration, head-pass count).
- **P1** — guided settle-exit wired into `forward_looped` next to `AdvantageMarginGate` with documented precedence + loop-reduction accounting.
- **P2** — GOAT bench + stratum analysis (close vs settled decisions) + three-incumbent adjudication + README/docs row.
- **P3** — tier upgrade decision at the Bench-617 record (GOAT only if all three incumbents beaten on a non-saturated fixture; otherwise the negative extends to a third lane and the class stays closed here) + production-consumer gate unchanged + revisit F2 if the deliberation analog is wanted.
