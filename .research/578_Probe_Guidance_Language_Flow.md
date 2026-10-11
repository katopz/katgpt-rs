# Research 578: Probe Guidance for Language Flows (frozen-state weak models)

> **Source:** [How to Guide Your Language Flow](https://arxiv.org/abs/2609.19356) — Dilip, Chen, Wang, Van Valen, Susskind, Bautista (Apple + Caltech), arXiv:2609.19356v1, 2026-09-16.
> **Date:** 2026-09-21
> **Status:** DISTILLED — GOAT-tier Gain verdict (inference track) + subordinate Gain (training track); Issue 865 filed. Verdict gate: ran via the generic sub-agent reviewer path (`request_verdict` timed out 2×180s); #Verdict: AGREE with three nits applied in the landing commit (2026-09-21).
> **Related Research:** 068 (RAEv2 — the shelved `self_guidance` proposal this paper unblocks), 044 (ELF — CFG via self-conditioning + progressive distillation), 058 (GRAM — learned guidance direction, unlanded sketch), 376 riir-train (SW-SetDLM), 575 (arness)
> **Related Plans:** 066 (D2F pipeline), 078 (RePlaid → `dllm_solver`), 104 (`mls_aggregate` — Research 68's landed half)
> **Classification:** Public

---

## TL;DR

Probe guidance steers a **frozen** flow-matching/diffusion LM with a tiny MLP probe (~2% of trunk FLOPs, ~1% of training compute, 20k steps) trained on **detached hidden states** at an **early** layer. The probe's output is a *weak* prediction; the guided output is the affine extrapolation `x̂_guided = x̂_s + (λ−1)(x̂_s − x̂_w)` — autoguidance's principle with **no second forward pass** (1.4–2× cheaper than checkpoint/dropout autoguidance). SOTA GenPPL on continuous dLMs (ELF-L 38→23 at H=5.40); +16.8 BoolQ on a 1.7B CFM.

**Distilled for katgpt-rs (inference-time):** the strong−weak extrapolation is a zero-alloc affine latent op on predictions, and the "weak model" can be the probe class we already ship for drafting (NextLat/EAGLE-3 `LatentDynamicsMLP` on frozen hidden states) — repurposed from *draft-and-verify* to *extrapolate-and-steer*. This **unblocks Research 68's shelved `self_guidance` feature**: its blocker ("needs weight training") has eroded twice over — (a) riir-train now trains this probe architecture class (`nextlat_*` lane — same MLP-on-frozen-states shape; the guidance variant retrains against the denoise target), and (b) the paper shows the weak side needs far less than a full intermediate LM head (Research 68 sketched a mid-layer LM head; the paper measures that an early-layer MLP at 3–7% compute suffices, and that **mid/late layers are the wrong position** — the weak model collapses toward the strong one and the guidance direction vanishes).

---

## 1. Paper Core Findings

1. **Mechanism.** Train `x̂_w = MLP[sg(h_ℓ)]` against the flow target (any Bregman divergence) on a frozen trunk; at inference `x̂_guided = x̂_s + (λ−1)(x̂_s − x̂_w)`, converted to velocity via Eq 2.1. λ=1 recovers unguided. Stop-gradient means probes never corrupt trunk dynamics and multiple probes train simultaneously on one trunk.
2. **Position law.** Probes at **early layers** (L0–L4) dominate; late layers make weak≈strong and the guidance direction vanish. This *refines* Research 68's mid-layer (n/2) sketch — and inverts the DoLa intuition (DoLa subtracts early from *late* through the shared head; the probe finding says the weak signal should come from where the trunk has *not yet* committed).
3. **Cheap-weak law.** Flat optimum at 3–7% of trunk compute (MLP, no cross-token attention); ~10⁹ tokens (~10⁴ steps) training suffices vs 10¹¹⁺ for the trunk; longer training does not hurt. A separately-trained small transformer of depth ℓ **cannot** substitute for a probe at layer ℓ — the probe inherits trunk dynamics through the shared frozen prefix, which is the entire point.
4. **Autoguidance mechanism finding.** Traditional autoguidance (weak = early checkpoint) works **iff the checkpoint precedes the entropy climbout** (the low-entropy, low-genPPL degenerate region at the start of training). Every post-climbout checkpoint collapses (overguidance). Holds identically for latent (ELF) and data-space (FLM) models. This is the first mechanism account of *why* weak-model guidance needs correlated dynamics.
5. **Amortization loop.** The guidance effect distills into the model in <10⁴ steps (FLM distills cleanly; ELF collapses), and you can **re-probe the distilled model and distill again** for continued gains — an iterative probe→distill cycle.
6. **Guided scoring estimator** (secondary): for MCQA likelihood proxies, reintroduce the terminal reconstruction term (Eq 5.3) that the unweighted KL bound drops — guidance shifts off-manifold and amplifies decode-step errors otherwise.

## 2. Distillation

### 2.1 What transfers (inference-side, katgpt-rs)

| Paper concept | Our substrate | Delta |
|---|---|---|
| Probe = tiny MLP on frozen detached states | `katgpt-speculative/src/belief_drafter.rs` `LatentDynamicsMLP` (NextLat, EAGLE-3 class); riir-train `nextlat_*` trains it | Exists as **drafter** (predict h_{t+1}); new use = predict the denoise/clean target → guidance weak side |
| `x̂_s + (λ−1)(x̂_s − x̂_w)` affine combine | `mls_aggregate` (weak-layer signal, landed), `sdar_gate` (sigmoid strong/weak mix), TILR (gated direction injection) | None ship **extrapolation** (`(1+w)·strong − w·weak`, w>0); all are gated mixes. Research 68 §7.2 documented the exact formula and shelved it |
| Guidance on a dLLM decode path | `katgpt-forward/src/d2f/` (D2F denoise loop: logits→sample→remask, τ_conf), `katgpt-core/dllm_solver.rs` (DPM-Solver++ 2M), `DecodeStrategy::SetDiffusion` | No guidance knob exists anywhere in the lane; λ would wire into `d2f_decode_block_prompt_q_core` at the logits step |
| Weak-from-low-entropy checkpoint | riir-train loss-curve telemetry (c13-daily-check), DASD entropy-routed distillation (plan318) | No entropy-phase/checkpoint-selection law; the climbout law is new |
| Amortize (distill guidance into weights, re-probe) | riir-train `distill_attention.rs` (frozen-teacher KL), ELF+PD progressive distillation (Research 44 addendum) | The **iterative re-probe loop** is new; progressive-distillation precedent exists |

### 2.2 Latent-space reframe

The guidance vector `D = x̂_s − x̂_w` is a per-token latent direction: "what the committed trunk believes beyond its uncommitted prefix". Extrapolating along D at λ>1 is a latent-to-latent op on predictions — affine, zero-alloc, gateable — exactly the constraint-#2 shape (no softmax renorm needed; the paper operates in x-space and converts once). For the think-brain/game surface: a frozen NPC brain (`npc_brain` freeze/thaw) + a tiny probe reading its early belief state could expose λ as a per-NPC **decisiveness knob** (low λ = exploratory crowd diversity, high λ = sharpened/stereotyped behavior) — plausible but speculative (unproven on tiny brains; recorded as a fusion idea, not a claim).

### 2.3 Consumer reframe (healer, priority #2)

The healer's rerank already combines strong/weak signals under a sigmoid gate (`sdar_gate`). The affine **extrapolation** form `score' = (1+w)·score_structural − w·score_bm25_baseline` would amplify "what the structural rerank sees beyond the cheap baseline" — a different knob shape (extrapolation vs gated mix). Marginal: `sdar_gate` covers the combination need; recorded as a one-line idea, not filed.

### 2.4 What does NOT transfer

- **CFG/self-conditioning specifics** — ELF's conditioning wire is not our decode path.
- **The 1.7B MCQA scaling results** — we do not train dLMs at that scale; quality claims do not carry to our mini-dLLM test models without measurement (the §3.6 discipline: no quality-parity claim without a PoC).

## 3. Path 0 decomposition (three-track check)

| Component | Coverage | Extraction |
|---|---|---|
| Affine strong−weak extrapolation at inference | Partial — formula documented (Research 68 §7.2, unlanded), `mls_aggregate`/`sdar_gate` adjacent | Modelless: closed-form op, ships behind `probe_guidance` feature |
| Tiny probe on frozen hidden states | Exists — `LatentDynamicsMLP` + riir-train `nextlat_*` lane (currently h_{t+1} target) | Repurpose: train against denoise target; rides existing lane (tiny training task, not a new pipeline) |
| Guidance on dLLM decode | Partial — D2F/dllm_solver lane ships, no guidance knob | Modelless combine at the logits step |
| Amortization (guidance→weights, <10k steps, re-probe) | Partial — distill lanes exist (`distill_attention`, ELF+PD note) | riir-train recipe; the iterative loop is new |
| Entropy-climbout checkpoint law | None | Recorded law; actionable only when weak checkpoints are selected for distill/guidance |

No component requires a new training pipeline; none is genuinely training-only. The modelless combine is the primary deliverable; probe training is a ~20k-step extension of the riir-train `nextlat_*` lane *pattern* — the training TARGET differs (denoise prediction, not h_{t+1} drafting), so it is a new training-task definition on existing infrastructure, not a free ride.

## 4. Prior art (published)

- **O'Brien & Lewis 2023** (arXiv:2309.09117): `(1+β)·Y_expert − β·Y_amateur` for AR LMs — **identical algebra** (β = λ−1), but the weak side is a full second LM (two forward passes).
- **DoLa** (arXiv:2309.03883): late−early layer logits through the shared LM head — weak source inside one model, no trained probe.
- **Tuned Lens** (arXiv:2303.08112): trained affine per-layer probes on a frozen model decoding hidden states to logits — **the exact probe substrate, used for interpretability only, never wired into decoding**.
- **SCD** (arXiv:2311.08981, ACL 2024): speculative drafting + contrastive verification target — proves the families compose, inside verify-reject. Its accepted=low-entropy finding is the AR echo of the paper's low-entropy weak-model law.
- **Karras et al. 2024** (arXiv:2406.02531): autoguidance base.
- AR-side **composition** gap: no published work uses a trained probe on the model's own hidden states as the guidance weak side at inference (Tuned Lens has the substrate, O'Brien-Lewis/DoLa have the consumer, nobody has the join) — integration-novel, not primitive-novel.

## 5. Verdict

**Tiers (per track):**

- **Modelless/self-adaptive inference track (katgpt-rs): GOAT-tier Gain** — not Super-GOAT: Q1 fails (formula is in-workspace prior art via Research 68's proposal and published AR prior art above); Q2 yes for the dLLM lane (its first guidance knob); Q3 moderate; Q4 yes (speculative + forward + core + riir-train lane). Actionable: unblocks a shelved feature and adds a missing capability class to the D2F lane → **Issue 865**.
- **Model-based track (riir-train): Gain (subordinate)** — probe training rides the `nextlat_*` lane *pattern* (~20k steps, frozen trunk; new target = denoise prediction, a new task on existing infra); the iterative probe→distill amortization loop + the entropy-climbout teacher-selection law recorded as recipe rows in the note; no standalone riir-train plan (Path 0 decomposition above).

**MOAT gate (katgpt-rs):** in scope — sampling/decode-adjacent base primitive (affine guidance combine) with feature flag + benchmark; the dLLM lane gains a capability class (quality knob) it lacks. Fusion ladder position: #3 (inference-perf league is speed; this is inference *quality* on the dLLM lane) — below prefill perf work in priority, cheap to land on shipped substrate.

**Weakest point (named):** the GOAT claim rests on the dLLM lane actually being exercised and the probe being trained — both unproven until Issue 865's PoC runs; guidance effects on our tiny dLLM test models may be negligible, and the AR-side composition is prior-art-adjacent (integration claim only). λ-sweep quality evidence must come from a measured Pareto (genPPL-class proxy vs entropy), not from the paper's numbers. Two sub-risks under that: **(a) probe-target mismatch** — `nextlat_*` trains h_{t+1} prediction for AR drafting; the guidance probe needs the denoise target, so the training task must be redefined (data construction + loss), not merely rerun; **(b) domain gap** — the paper's mechanism lives in continuous flow-matching x-space while D2F is discrete logits→sample→remask; logit-space extrapolation on a remask lane is a domain adaptation, and T3's λ-sweep is the instrument that catches it if it breaks.

**Fusion:** paper × Research 68 `self_guidance` × nextlat probe lane × D2F decode = guidance knob for the dLLM lane at ~2% FLOP overhead, with an AR-side experiment arm (probe-guidance contrastive decode, the unpublished join) as the stretch goal. None of the four alone produces a steering signal on our shipped decode paths.

## 6. Issue 865 task sketch (summary)

1. `probe_guidance` feature: affine combine `x̂_s + (λ−1)(x̂_s − probe(h_early))` at the D2F logits step (`d2f_decode_block_prompt_q_core`), λ config knob, zero-alloc.
2. Weak-side probe: `LatentDynamicsMLP` class repurposed to the denoise target; early-layer tap (position law above); trained via the riir-train `nextlat_*` lane pattern (frozen trunk, stop-grad, ~20k steps).
3. GOAT gate: λ-sweep Pareto (quality proxy vs diversity) vs unguided D2F and vs a dropout-autoguidance arm; G1 correctness (λ=1 bit-identical), G2 quality, G3 no-regression at λ=1, G4 alloc-free combine.
4. AR experiment arm (stretch): tuned-lens-style probe at an early layer steering AR decode logits (the unpublished join); compare vs DoLa-style shared-head contrast as baseline.
5. Record the entropy-climbout law + amortization loop as riir-train recipe rows (checkpoint selection for weak-model teachers).
