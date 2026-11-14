# Research 492: Recirculation × Full-Bandwidth Transformer — Vertical Feedback Channel Widening

> **Sources:**
> - [Recirculation](https://arxiv.org/abs/2608.17981) — Mozer, Siddiqui, Sawyer, Sanyal, Liu (Google DeepMind / UT Austin), 2026-08-18
> - [Full-bandwidth transformer](https://arxiv.org/abs/2608.08888) — Wang, Cai, Zhan, Dong, Fan, de Rosa, Pearce, Langford (JHU / Princeton / Microsoft AI Frontiers), 2026-08-09
> **Date:** 2026-08-19
> **Status:** DISTILLED — pending owner decision (two filings: katgpt-rs Issue 673 primitive PoC; riir-train Plan 344 training recipe)
> **Verdict: GAIN (high) — two papers, one principle, two ends of the training-free↔trained spectrum. Recirculation is the immediately-actionable modelless half (works on off-the-shelf Gemma2 — the paper's own App. C.1 shows Gemma2 gains as pronounced as Gemma3, and we hold gemma-2-2b-it-f16.gguf). Full-bandwidth is the trained half (multi-pass scheduled objective → riir-train Plan 344 per Path 0.5); its GLU-fusion decode + reachability math are extractable now. Neither mechanism ships: signal-diff vs the closest cousins (`cross_stage_relocation` = same-pass OVERWRITE; `LoopMode::TrainingFree` = depth-only sub-stepping) below.**
> **Related Research:** 417 (Cross-Stage Residual Relocation — SHIPPED, closest cousin, PoC-refuted fixed pair), 097 (Training-Free Looped Transformers — depth-only), 073 (LT2), 018 (Free Transformer latent injection), 343 (System 1.5 depth-step shortcuts), 325 (latent reasoning taxonomy), 035/344/266 (fixed-point/contraction family — the 3%-three-pass contraction result joins this family)
> **Related Plans:** 431 (cross_stage_relocation — shipped opt-in), 108 (LT2), 136 (training-free loop wrapper)
> **Classification:** Public (the generic operator mechanism; tuned LLM application is private-side)

---

## TL;DR

Both papers attack the same structural limitation from opposite ends. In a decoder-only transformer, the **vertical feedback channel between decoding steps is narrow**: only the sampled token (≤ log₂|V| bits) returns to the bottom of the stack, while the top-layer hidden state (D dims) is discarded. Intermediate activations persist in the KV cache but are **depth-frozen** — a state produced at layer ℓ is readable only by layers above ℓ; the deepest state never returns to the bottom.

The reachability formalization (Full-bandwidth §2, the transferable math):

```
R_std(t, ℓ) = {(t′, ℓ′) : t′ < t, ℓ′ < ℓ}   |R_std| = Θ(T·ℓ)   — shallow layers see a partially-processed past
R_lf (t, ℓ) = {(t′, ℓ′) : t′ < t, 0 ≤ ℓ′ ≤ L} |R_lf|  = Θ(T·L)   — every layer reads the full-stack-processed past
```

- **Recirculation** (training-free): run the model step-by-step; at each step, **leak a small mixture** `z_{t+1,d} = α·f(z_{t,s}) + β·z_{t,d}` of a deep-layer (source s) activation down into a shallow layer (destination d) at the **next input step**. Convex mixture (α ≈ 0.07–0.15), source L2-renormalized to destination norm. Off-the-shelf Gemma3: 4.7–16% ppl reduction (up to 35% at 12B); **adaptive variant (tiny MLP emitting per-token vector α/β, base frozen) hits 23% mean ppl reduction and 8.8%/20.9% GSM8k error-rate reduction (pass@1/@128)**.
- **Full-bandwidth** (trained): fuse the previous top-layer hidden state with the sampled token embedding through a GLU — `e ⊗ h = W_U h ⊙ σ(W_G e)` (state on the value path, token as gate — the asymmetry closes the shortcut of ignoring the state) — and feed back as the next input. Trained via scheduled multi-pass teacher forcing (bulk single-pass; 22% two-pass; **3% three-pass batches turn the learned map into a contraction stable at 1000 feedback steps**). 1B/400B-token model matches standard transformers trained on ~1.5–2× the tokens; **shorter reasoning traces at equal or better accuracy** (state rides the latent instead of being verbalized).

**The one-line synthesis:** deep conclusions die at the top of the stack; a small norm-matched leak of them back into the shallow input of the next step buys state tracking that neither CoT (token-bottled) nor looping (depth-only, same input) provides.

---

## 1. Paper Core Findings

### 1.1 Recirculation — the training-free half

**Mechanism (Eq. 1–2).** `z_{t+1,t,d} = α·f(z_{t,t,s}|d,t) + β·z_{t,t,d}` with `β = 1−α` (1B) or `β = 1` non-convex (4B/12B), and `f(z) = (‖z_{t,t,d}‖₂ / ‖z‖₂)·z` — rescale the source to the destination's L2 norm. Recurrence is in **both depth and step** (the paper's Fig. 3/4 unrolling): state z(t) and z(t+1) can live in the *same layer* across steps, which looping (depth-only recurrence) cannot express.

**Layer-pair landscape (Gemma3, sweep-validated):** destination shallow — {11,4} for 26-layer 1B (source 0.42L → dest 0.15L), {18,9} for 34-layer 4B (0.53L → 0.26L), {35,16} for 48-layer 12B (0.73L → 0.33L). Smooth heatmaps; mid-architecture destination band 0.15–0.33L, source band 0.42–0.73L. Robust across Ministral3, Pythia, Qwen3, Phi2 (qualitatively) — **and Gemma2/Gemma4 gains "as pronounced as Gemma3"** (App. C.1). Gemma's Peri-LN (output as well as input norm) is the hypothesized compatibility factor.

**Why it works without training:** the residual stream is a shared blackboard — features are layer-aligned because a feature added at any depth has the same direct effect on the output (commutativity of addition). Recirculating a disambiguated deep representation forward amplifies the already-meaningful feature at the shallow layer. (Same substrate argument as training-free looping, R097.)

**Token analysis (the salience map):** benefit is a power function of lag k (large at short lags, residual tail at 256); positions ~20–200 have the most persistent effects; **content classes dominate — adverbs/adjectives/verbs ≫ numerals/determiners/pronouns; plural nouns benefit, singular don't.** Effects are additive in log-likelihood. Early positions (t<10) can be *harmed* at 1B → ramping schedule `α_t = min(t/10, 1)·α`.

**Controls:** not temperature tuning (temp-1.2 gives −8.5% ppl alone, recirculation −14.2%, combined −19.6% ≈ additive); not looping (looped heatmaps qualitatively different; looping shows no robust benefit on Gemma3 at ≤4B); adaptive-recirculation MLP (frozen base) ≈ full fine-tune quality (23.0% vs 21.6% ppl reduction).

**The racing-thoughts motivation (§1):** contextualization errors — a polysemous token disambiguated at deep layers is re-read *ambiguously* by the shallow layers of the next step; response generation outpaces semantic convergence. Activation-patching the resolved interpretation down (Patchscopes, Lepori et al.) cut contextualization errors 60%; recirculation is the undifferentiated, continuous, training-free version of that patch.

**Costs (honest read):** generation-time = **two stack instances in parallel per step** (near-free on throughput hardware, ~2× FLOPs serial; **2× KV-cache footprint** — each stack keeps its own cache). **Prefill becomes serial** (token-by-token) — the paper flags this as the real cost and defers blockwise recirculation (K tokens per recirculation step) as future work.

### 1.2 Full-bandwidth — the trained half

**GLU fusion (Eq. 3–4).** `e_t ⊗ h_{t-1} = W_U h_{t-1} ⊙ σ(W_G e_t)` — hidden state on the value pathway, token embedding only as a multiplicative gate. Discarding the state discards the input itself; reading the state is mandatory. An additive fusion `e + W h` leaves a shortcut (suppress the state path, recover plain pretraining loss).

**Three decode regimes:** STANDARD (no feedback — the trained model at step 0 is within a small margin of baseline), SOFT (single prefill; fused inputs during decode; **<1% per-token overhead**, 2 D×D matmuls), FUSED (one extra fused prefill pass; 2× prefill, best for coding). Math tasks favor SOFT (state carried through generation); coding favors FUSED (deeper prompt representation).

**Training (temporal parallelism).** k passes; pass k shifts pass k−1's states one position right, fuses with token embeddings, re-runs the stack **parallel over all positions**. Sequentiality paid across passes, not positions (~k× compute). Loss on every pass; gradients flow through (later passes backprop into earlier states — an auxiliary "make states reusable as inputs" objective that improves the model even under STANDARD decode).

**The 3% contraction stabilizer (the crown-jewel training finding).** 75% single / 25% two-pass batches: the learned map **diverges past its trained depth** (ppl rises; hidden-state change oscillates). Adding just **3% three-pass batches** makes the iterates a contraction toward a fixed point — flat ppl and decaying ‖h^(k)−h^(k−1)‖ through 30 feedback passes, stable at **k=1000**. Train the transition until it is stable under further self-composition, not to the inference horizon.

**Stability recipes:** depth scaling to keep ‖h_L‖ ~ O(1) not O(L); RMSNorm on the fused input; embedding↔readout weight tying (shared input basis for plain vs fused inputs); jitter noise σ=0.02 on the carried state; **prefix mixin** (random plain-embedding prefix per pass, matching the prompt-then-generate boundary of inference).

**Results:** 1B/400B-token runs; 100B full-bandwidth ≈ 200B standard; 200B ≈ 400B (≈2× data efficiency at 2 feedback prefill passes); GSM8k/Math500/HumanEval/MBPP gains on every task; carries through long-context extension + instruction tuning; **median reasoning length shrinks at equal-or-better accuracy** (the conciseness the widened channel predicts — disappears after instruction tuning, attributed to off-policy verbosity of the tuning data).

**Layer-0 probe validation (§4.4):** under one-step recurrent prefill, layer-0 probe accuracy on completion-tracking/delayed-memory state hits 99.6–100% (vs near-chance under standard prefill) — the recirculated input literally delivers the fully-processed prefix summary to the bottom of the stack.

---

## 2. Distillation

### 2.1 Path 0 decomposition (both papers)

| Component | Ships? | Modelless extraction |
|---|---|---|
| Recirculation mixture `α·normmatch(z_s) + β·z_d` cross-step | **NO** — `RelocateOp` is same-pass overwrite | YES — closed form, O(D) |
| Layer-pair selection heuristic (dest 0.15–0.33L ← src 0.42–0.73L) | Partial — R417 pairs target 0.45L mid; R097 window 0.45–0.60L. **Different bands.** | YES — offline sweep |
| Ramping schedule `min(t/10,1)·α` | NO | YES — closed form |
| Token-class benefit structure (content ≫ function; plural ≫ singular) | NO — but per-tick salience gating (R148) is the same shape of idea | YES — a salience-masked recirculation |
| Adaptive α/β MLP | NO | Modelless replacement: sigmoid(dot(salience_dir, z)) gate per the house pattern |
| GLU fusion `W_U h ⊙ σ(W_G e)` | NO | The *op* is deterministic — but only meaningful on a model TRAINED with fused inputs |
| Multi-pass scheduled training + 3% contraction stabilizer | NO | Training → riir-train Plan 344 |
| Reachability bounds Θ(T·ℓ) vs Θ(T·L) | NO | Pure math, public primitive |
| Conciseness effect (latent carries state) | NO | Observable consequence, not constructible modellessly |

**Recirculation is ~fully modelless** (inference-time, frozen weights, deterministic ops). **Full-bandwidth's value is the training loop** → Path 0.5 Plan; its math + GLU design principle extract now.

### 2.2 Prior-art surface (signal-diff, §3.6 — what already ships, do not duplicate)

- **`cross_stage_relocation` (R417 / Plan 431, katgpt-core, SHIPPED opt-in)** — the closest name-match cousin. Signal diff: `RelocateOp` **overwrites** the anchor's state at `dst_stage` from `src_stage` **within one forward pass**, one-shot, diagnostic-guided (`RelocatePair::Custom`). Recirculation **mixes** (convex, norm-matched, small α) at **every** position and carries the result to the **next input step**. **Plan 431's defend-wrong PoC refuted its fixed-pair default for exactly the failure mode recirculation's semantics prevent: the overwrite CLOBBERS in 2/4 clean configs.** Recirculation is the missing safe-mixing variant of the same source→destination topology — a direct follow-up to a known-open PoC failure, not a duplicate.
- **`LoopMode::TrainingFree` (R097 / Plan 136, katgpt-core)** — damped-Euler sub-stepping of a mid-stack window **within one position** (depth recurrence). No cross-step state transport; explicitly contrasted by the Recirculation paper (their Fig. 8: looping shows no robust benefit on Gemma3 ≤4B while recirculation does).
- **LT2 (R073 / Plan 108)** — training-time weight-shared looping. Different axis.
- **The Free Transformer (R018)** — random latent Z injected mid-layer; requires trained-with-Z weights (the note's own NEVER verdict: random injection on untrained models degrades). Recirculation is the finding that a *small norm-matched leak of the model's own deep state* DOES work untrained — the delta that flips R018's verdict for the self-state case.
- **Belief/persistence substrate (riir-games-shared)** — `GenericSpatialBelief::decay_confidence = sigmoid(−λ·Δt)` is the *scalar collapse* of recirculation (keep a value, fade it). The *vector* leak of a resolved deep state into next-tick perception input does not ship; the two-brain model's think→info direction is deliberately one-way (Issue 047 — recirculation would target the **perception→perception** next-tick path, not the sync boundary; no raw/latent violation).
- **Fixed-point/contraction family (R035/R344/R266)** — the 3%-three-pass contraction result joins this corpus as a *training-schedule* instance; our shipped instances are inference-side halting/damping.

**Web prior-art check:** the technique space (latent feedback decoding / inference-time cross-layer state reinjection on frozen checkpoints) returns only the papers' own cousin tree — Feedback Transformer (Fan '20, sequential training, attention-level), T2MLR (middle-layer injection, extra MLPs), Latent Recurrent Transformer (layerwise projections), PonderLM-2 (doubles input length), Coconut/Soft Thinking (latent CoT). No published training-free cross-step mixture variant prior to Recirculation; no scheduled multi-pass + 3% stabilizer prior to Full-bandwidth.

### 2.3 Latent-space reframing (the mandatory check)

The mechanism is a **latent-to-latent op on the residual stream between forward steps** — squarely the house pattern family:

- `α·normmatch(z_s) + β·z_d` is a per-dimension convex blend of two latent states — the same shape as `CommittedFieldBlend` (R158, per-NPC personality blend) with the roles played by (previous deep self-state, current shallow self-state).
- The adaptive variant's per-token vector α/β is a **sigmoid-gated elementwise leak** — literally `σ(MLP(z)) ⊙ Δ` direction-gated state update, the house `dot + sigmoid` projection with the direction vector derived from the state itself.
- The GLU fusion `W_U h ⊙ σ(W_G e)` is a **gate-on-value** composition — token *gates* the state rather than mixing into it; the anti-shortcut asymmetry is a design principle for any fused latent channel where a frozen path could otherwise dominate.

On the DEC side (d ≤ 3 caveat respected — this is a per-stage, not per-dimension, operator): recirculation across steps on a cochain is a discrete dynamical system whose fixed point the 3%-stabilizer makes contractive — adjacent to `hodge_laplacian` damping semantics, not overlapping.

### 2.4 Fusion (Recirculation × Full-bandwidth × R417 × two-brain NPC belief)

**The pattern both papers validate:** *a small, norm-matched leak of a layered processor's own deep-stage output into its shallow-stage input at the next step buys state tracking, interpretation persistence, and shorter verbalization — without training, if the leak is convex and small.*

Applied to the per-NPC cognition stack (riir-ai, L0 reactive → L1 perception → L2 deliberation, Issue 054):

1. **The racing-thoughts failure ships in our game.** L0 commits to flee on an ambiguous percept before L1/L2 resolves "ally". Today the only remedy is the next think-cycle; the resolved interpretation dies with the tick. This is *exactly* the depth-frozen past + narrow feedback channel, at game scale.
2. **Belief recirculation:** each tick, leak `α·normmatch(L2_resolved_affect)` into next tick's L1 perception input (the perception→perception path — NOT the think→info sync direction; the sync boundary stays untouched). Convex mixture + norm matching + small α is precisely the semantics Plan 431's PoC showed the overwrite lacks.
3. **α as personality knob:** leak rate = interpretation trust. Skittish NPC (low α) re-derives every tick — fast to alarm, prone to flip-flop. Steady NPC (high α) keeps its conclusion — calm, slow to re-alarm. `decay_confidence = sigmoid(−λΔt)` is the scalar limit; the vector leak preserves more of the resolved state. The PoS finding (recirculate content, not function) maps to recirculating **salience-gated** percepts only (R148's per-tick emit gate).
4. **Full-bandwidth's contribution to the fusion:** the conciseness law — when state rides the latent, verbalization shrinks. Game analog: an NPC with recirculated belief needs fewer deliberation cycles (cheaper L2) to hold the same interpretation — a per-NPC compute knob, same shape as R149's gain-cost reasoning depth.

Novelty: the vector cross-tick leak does not ship (belief updates are fog-of-war-gated observations only). All four §1.5 axes except quality-on-our-substrate are satisfied; per the no-candidate rule the Super-GOAT guide is deferred until the PoC lands (Issue 673 covers both the LLM half and this half).

---

## 3. Verdict

**GAIN (high).** Two filings:

1. **katgpt-rs Issue 673** — `recirculation` open primitive (katgpt-core, public): `RecircOp { src_stage, dst_stage, alpha, beta, norm_match, ramp }` cross-step mixture operator, sibling of `RelocateOp` under a shared stage abstraction; + defend-wrong PoC on a real pretrained model (gemma-2-2b GGUF in riir-train/data; paper App. C.1 shows Gemma2 family gains) measuring ppl delta vs the no-recirculation baseline and vs the R417 overwrite semantics on the same layer pairs. **Honest cost accounting required in the PoC: 2× decode FLOPs serial + 2× KV-cache footprint + serial prefill** — the paper's "negligible" is a throughput-hardware claim; on our decode stacks the KV doubling is the binding cost (interacts with KVarN, R159).
2. **riir-train Plan 344** — Full-bandwidth training recipe per Path 0.5 (below).

**Novelty gate scoring (why Gain, not Super-GOAT, this session):** Q1 prior art — clear (signal-diff above). Q2 new behavior class — yes on paper's substrate, unproven on ours. Q3 selling point — the NPC interpretation-hysteresis story is plausible and gameplay-visible, but quality-unproven (§3.6: a quality claim needs the head-to-head). Q4 force multiplier — yes (417, 148, two-brain, blend family). **3.5/4 → Gain + issue, guide on PoC success.**

**Not covered / honestly out of scope:** Full-bandwidth's pretraining-scale runs (≥200B token-equivalents) are not affordable on the 4090; only micro-scale replication is (§4). Instruction-tuned conciseness loss (off-policy tuning data) — noted for Plan 344's eval design.

---

## 4. Plan sketches

### 4.1 katgpt-rs (Issue 673): the recirculation primitive

- **Phase 1 — operator (katgpt-core, `recirculation` feature, opt-in):** `RecircOp`, mixture + L2-norm-matching math, ramping schedule, layer-pair heuristic constants (dest 0.15–0.33L ← src 0.42–0.73L). Unit tests: mixture boundedness, norm-matching idempotence on equal norms, ramp schedule, β=1 non-convex variant. Zero-alloc (fixed D scratch).
- **Phase 2 — PoC (riir-poc, defend-wrong):** gemma-2-2b forward, arXiv/PG19-style windows; arms: (a) baseline, (b) recirculation fixed α sweep {0.07, 0.10, 0.15}, (c) R417 overwrite on the same pairs (expected: clobbering — reproduces the Plan 431 failure on a real model, the strongest possible contrast), (d) recirculation + temperature-1.2 (additivity control). Gate: ppl reduction > 0 on ≥2 datasets with the mixture, and strictly safer than (c) at equal pairs.
- **Phase 3 — decode-stack integration decision (riir-ai, only on Phase 2 PASS):** the 2×-KV cost call — if KVarN absorbs it, wire into the Gemma2 decode loop behind a feature flag; else keep the operator substrate-only.
- **Phase 4 — GOAT:** G1 determinism (fixed α, bit-identical repeat), G2 overhead (O(D) mixture ≤ 1µs), G3 no-regression default-off, G4 alloc-free.

### 4.2 riir-train (Plan 344): full-bandwidth recipe — Path 0.5

- **Recipe:** start from a pretrained checkpoint (the paper's schedule requires it); mid-training introduction of feedback passes; mixture 75/22/3 (single/two/three-pass); prefix mixin; jitter σ=0.02; depth scaling to O(1) top-state norm; RMSNorm on fused input; embedding↔readout tying; λ=1 multi-pass loss, no gradient detach.
- **GPU-hours (honest):** micro replication at ~124M params / 1B tokens / 3-pass-equivalent ≈ 2.2e18 FLOPs ≈ **~15–24 h on the 4090** — affordable. The paper's 1B/400B config (~1.8e20+ FLOPs) is **not** affordable; do not plan it.
- **GOAT gate:** trained-with-feedback SOFT decode vs (a) same checkpoint STANDARD decode, (b) equal-compute baseline trained longer without feedback. The 3%-stabilizer claim (contraction at ≥30 self-compositions) is a cheap, load-bearing assertion to replicate first — it needs no benchmark, only the ‖h^(k)−h^(k−1)‖ decay curve.
- **Dual-track contribution:** modelless side = the GLU fusion op + reachability bounds (public); trained side = the checkpoint.

### 4.3 The fusion follow-up (deferred to Issue 673 Phase 5, on PoC success)

Belief-recirculation guide (riir-ai `.research/`): the L2→L1 next-tick leak, α as personality parameter, salience-gated recirculation (R148 bridge), decay_confidence as the scalar limit. Super-GOAT guide obligations trigger only if the PoC quality axis confirms.

---

## 5. Cross-references

- R417 / Plan 431 — cross_stage_relocation (the clobbering PoC this paper's semantics answers)
- R097 / Plan 136 — training-free looped transformers (depth-only; the contrast arm)
- R073 / Plan 108 — LT2 weight-shared looping
- R018 — Free Transformer (random-Z injection; recirculation flips its NEVER verdict for self-state)
- R325 — latent reasoning taxonomy (this pair slots as the "feedback/width" axis)
- R148 — per-tick emit salience (the PoS→salience bridge)
- R149 — per-NPC gain-cost reasoning depth (conciseness → cheaper deliberation)
- R158 — CommittedFieldBlend (convex latent blend precedent)
- R159 / KVarN — KV compression (the 2×-KV interaction)
- R035 / R344 / R266 — fixed-point/contraction family (3%-stabilizer joins here)
- riir-ai Issue 047/054 — two-brain model, L0/L1/L2 flee cognition (the racing-thoughts game analog)

## Citation

## PoC Addendum (Issue 673 Phase 2, 2026-08-19 — the defend-wrong verdict)

**Result: ppl axis FAILED on our substrate; operator ships opt-in; Phases 3/5 do not proceed.**

Harness: `riir-ai/crates/riir-poc/tests/recirculation_poc.rs` — gemma-2-2b-**it** f16 GGUF, serial token-by-token windows, paper layer pair for 26 layers (src=11/dst=4), ramp 10, two local text registers (katgpt-rs `.research` = arXiv-style; riir-ai `.docs` = long-form technical; PG19 not on disk).

| run | dataset | baseline ppl | α=0.07 | α=0.10 | α=0.15 | overwrite | temp-1.2 |
|---|---|---|---|---|---|---|---|
| 2×96, post-dst | A | 91.32 | 99.66 | 104.80 | 114.28 | 2.2e8 | 68.38 |
| 2×96, post-dst | B | 311.20 | 329.88 | 342.66 | 374.97 | 5.4e8 | 197.16 |
| 3×160, pre-dst | A | 339.03 | 361.97 | 380.10 | 432.03 | 4.4e9 | 233.87 |
| 3×160, pre-dst | B | 342.44 | 370.27 | 389.42 | 439.43 | 8.5e9 | 233.11 |

- **12/12 recirc cells harmful** (−6…−28% ppl, dose-dependent in α, both injection points, both registers). The paper's Gemma3-family training-free gains do NOT transfer to gemma-2-2b-**it** under this harness.
- **Safety axis PASSES**: mixture ≫ overwrite at equal pairs (the R417 clobbering catastrophe reproduced on a real model — Plan 431's failure mode, the strongest contrast arm; the mixture is the safe member of the relocation family by construction and by measurement).
- **temp-1.2 improves ppl** (+25…+37%) — same direction as the paper's control, so the harness measures real distribution effects.

Honest caveats (binding for any reopen): session-scaled counts (192–480 tokens/arm vs paper ~500×1024 — direction consistent + monotone, magnitudes small-sample); IT-tuned model (paper validated base models); markdown corpora (not wikitext/PG19 prose). Reopen = paper-scale re-run + base-model check on the 4090. Full record: [`.benchmarks/668_recirculation_goat_and_poc.md`](../.benchmarks/668_recirculation_goat_and_poc.md).

**Status update:** Phase 1 operator + Phase 2 PoC SHIPPED (bench 668); Phase 3/5 closed by the gate; `recirculation` stays opt-in with the negative recorded.

```bibtex
@article{mozer2026recirculation,
  title   = {Recirculation},
  author  = {Mozer, Michael C. and Siddiqui, Shoaib Ahmed and Sawyer, Danny and Sanyal, Sunny and Liu, Rosanne},
  journal = {arXiv preprint arXiv:2608.17981},
  year    = {2026}
}
@article{wang2026fullbandwidth,
  title   = {Full-bandwidth transformer},
  author  = {Wang, Xi and Cai, Ziyang and Zhan, Zheng and Dong, Harry and Fan, Ying and de Rosa, Gustavo and Pearce, Tim and Langford, John},
  journal = {arXiv preprint arXiv:2608.08888},
  year    = {2026}
}
```

> Update 2026-11-14 (riir-train Research 497, riir-train Issue 690): the budget-fair redo of this family landed — latent feedback (additive top-layer state via learned W) priced against a Transformer spending the same FLOPs on MORE EPOCHS over repeated data, closing the exact gap this note and the 073/241 redirects flagged ("none repeats data"): kappa <= 1.91x compute, mu <= 1.69x data, one coupled law (v = 1.14 capacity factor, held-out 778M <= 0.29% err), UTPP break-even 75/33/20/12. The trained cell routes to riir-train Plan 344 (Issue 690 T3); a constraint pointer line was appended to that plan.
