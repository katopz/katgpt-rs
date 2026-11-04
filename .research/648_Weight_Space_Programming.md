# Research 648: Weight-Space Programming — LLMs natively read & write neural network weights

> **Source:** "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights" — Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL), blog post 2026-11-02. https://rikiyat.github.io/blog/weight-space-programming/ — **blog-only, no arXiv/peer-reviewed version as of 2026-11-03** (verified: arXiv API, OpenReview, author profile all 0 hits). Complete experiment logs: github.com/RikiyaT/lm_understand-logs
> **Date:** 2026-11-03
> **Status:** Active
> **Related Research:** 080 RandOpt (the closest cousin — same harness, random proposals)
> **Related Plans:** 121 RandOpt weight-space perturbation (katgpt-rs); riir-train Plan 413 (civ re-open test); riir-ai Plan 596 (npc_brain freeze wire)
> **Files created this session:** `.issues/980_weight_edit_overlay_lane.md` (katgpt-rs, the modelless half); `riir-train/.issues/665_wsp_weight_production_lane.md` (the weight-production half)
> **Classification:** Public

---

## TL;DR

A general-purpose frontier LLM reads and writes neural-network weights natively — no optimizer, no gradient descent, sometimes no environment feedback: it identifies a policy's game from raw weights, repairs a PPO local minimum with a one-shot bypass edit (3.5% of weights, 0%→100% success), and repairs a looping 0.8B reasoning model by subtracting a loop direction from MLP output columns (0.016% of weights; MATH-500 7.9→48.1). **The mechanism layer is fully modelless** — closed-form weight arithmetic + deterministic accept/reject over pinned evals — and maps onto our existing substrate almost component-for-component: the harness IS Plan-121 RandOpt's (propose → N measured evals → keep-if-wins), the control arms are RandOpt/BanditPruner arms, the edit carriers are `LoraPair`/`SparseErrorBypass` COO shapes, the refusal precondition is our halter telemetry, and the artifact discipline is the freeze/thaw envelope. Filed as **GOAT (fusion)**: the composition is ours, the components are published.

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable primitive is the **reasoning-authored, deterministic-applied, gate-gated weight-edit overlay**: an edit spec (rank-1 column subtraction / sparse bypass rows) authored once (by an LLM session or by hand from diagnostics), applied by a fixed compiler at load, gated by a pre-registered control battery (dose-response monotonicity, neighbor-layer isolation, norm-matched random control, obstacle pre-screen, retention), and committed as a BLAKE3-pinned frozen artifact. Authoring may be stochastic; application and verdict are deterministic. The one genuinely new-in-literature edit class the blog demonstrates: **weight-level direction subtraction for repetition-loop repair** (abliteration literature covers refusal-direction weight orthogonalization; no published weight-level loop repair existed before this blog).

---

## 1. Core findings (blog inventory)

| Exp | Task | Scale | Result |
|---|---|---|---|
| 1 | Identify PPO policy's game from weights (text, no tools) | 17,253-param MLP | Correct on all 4 games, unnamed option set |
| 2 | Identify diffusion model's output shape | 23,824-param DDPM | Correct; draws the shape's outline from output-layer weights |
| 3 | Identify char-LM corpus from MIDDLE layers only + find induction heads | 7.2M params, I/O layers removed | Corpus correct 3/3; induction heads exactly match measured ground truth (via weight-product composition, the Elhage method) |
| 4 | Fix PPO local minimum (car never reverses before ramp) | 171,664 params | One-shot bypass pathway through existing units + position-conditioned switch-off; 3.5% of weights; 0%→100% over 1024 episodes; **no env feedback while authoring** |
| 5 | Write MinAtar policies from scratch (weights as text) | 600→32→32→6 | Beats PPO-100M on all 3 games (Breakout 517.4 vs 15.8) |
| 6 | Write 9×9 Go policy (sandbox + forward-pass budget 0→50) | 1.5M params | 0 passes: beats smaller PPO; 50 passes: beats equal-size PPO trained 20M games |
| 7 | Identify sycophantic fine-tune | Gemma-3-270M ×2 | **Weights alone FAILED** (4 attempts); attention-patterns-only succeeded 3/3 — heads 10.2/11.2 attend the user's suggestion |
| 8 | Repair looping Qwen3.5-0.8B (math) | 0.8B params | Loop direction localized to layers 15/19 MLP output columns; subtract at strength k (dose-response 1→6); top-64-neuron restriction kept (same gain); norm-matched random-direction control did nothing; final 131,072 weights (0.016%); MATH-500 7.9→48.1, GSM8K 13.8→67.1, IFEval 21.0→62.3, token-limit-hit 93→33% |
| Addl | Improve Gemma-3-270M (rarely loops) | 270M | **Could NOT be improved** — no single failing component |

**The obstacle-vs-ability law (the blog's most durable claim):** weight edits remove obstacles that hide abilities; they do not add abilities. A model with no single failing component is out of scope — this converts directly into a refusal precondition.

**Harness shape:** LLM proposes edit → harness runs N measured forward passes → keep if it wins, revert otherwise; edits interleaved between passes.

---

## 2. Prior art (searched 2026-11-03, ~30 fetches)

| Class | Prior art | Delta vs this blog |
|---|---|---|
| LLM edits weights, no GD | **csinva/weight-space-programming-autoresearch** (same-day companion; Claude hand-writes 959 weight programs, 10/10 algorithmic tasks at 100%) | Closest existing; algorithmic tasks only, no repair-at-scale |
| Training-free weight-edit behavior repair | Resisting Harmful Instructions (arXiv:2610.21189), FORGE (2610.05190), GROM unlearning (2608.05783), TRNE rank-one toxicity edits (2605.27997), QK-product steering (2606.20419), LLM-VA (2601.19487) | Not LLM-authored — closed-form/contrast-based edits, same "permanent edit, no gradients" class |
| Weight-level direction ablation | **Abliteration line**: Labonne HF blog 2024 (weight orthogonalization on the refusal direction, from Arditi 2406.11717), Heretic (~22k stars, automated weight surgery + "Arbitrary-Rank Ablation"), steering-compiled-to-weights (2604.12359), abiteration survey 2512.13655, "Abliteration Is Not a Scalpel" (2607.17427) | Well-published for REFUSAL removal. **No published weight-level MLP-column direction-subtraction fix for repetition/looping before this blog** (checked: `"loop direction"`, `looping AND weights` → 0 on-point; the loops paper 2512.12895 proposes only training-time interventions) |
| Audit without execution | **Weight Oracles (arXiv:2610.07334, NeurIPS 2026 NeuralArtifacts workshop spotlight)** — trained readers diagnose weights, backdoor AUROC 0.93; Detecting Backdoored LoRAs from Weights Alone (2602.15195) | Strong published field; the blog's READ half is anticipated here |
| Surveys | A Survey of Weight Space Learning (2603.10090); MLE-bench line | **No survey covers LLM-native weight programming** — the blog sits in a gap |
| Reception | HN front page 2,515 pts/780 comments; VentureBeat; XenoSpectrum deep critique; **Willison replication + critique** (2026-11-03: Qwen 3.8 27B local, 3/4 games; Honnibal: misleading to conflate the no-tools read with the with-code edit; Goedecke: Exp-8 = human-researcher steps; Willison: "notepad and pencil" attainable, not declaring the source-code era over) | Critiques recorded below |

---

## 3. Honest caveats (must ride any filing)

1. **Gains ≈ loop suppression, not new ability** — the blog's own numbers: 576 of 579 newly-correct MATH-500 answers were token-limit truncations before the edit; accuracy on previously-finishing answers unchanged. The edit un-hides an ability (the obstacle law); it does not create one.
2. **Not purely "native"** — HN critique (jsenn): the Exp-8 fix used tool calls (Python kernel) + what is recognizably standard steering-vector technique; the tool-free read/write demos are the small-scale experiments only.
3. **Single proposer; read half independently replicated 2026-11-03** — one model (gpt-6-astra), reasoning effort varied between experiments, best-of-3 seeds in Exp 1 (XenoSpectrum: "not peer-reviewed, no DOI"; missing failure history, reproducibility gaps). **Update:** Simon Willison replicated Exp 1 (which-game-from-weights) with a different model class — Qwen 3.8 27B (17 GB, open-weights, local laptop, LM Studio): **3/4 games** (Pac-Man 72% / Breakout 88% / Pong 75% confidence; Freeway failed), plus Kimi K3.1 Pro correct on Breakout — at 19K–44K reasoning tokens and 9.7–65 minutes per classification (simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights; details + the lane relevance in Research 649 §2.1). The write/repair half (Exp 8) remains unreplicated.
4. **Missing baselines** — no rule-based-controller, ES, or PEFT cost comparison at the measured budgets.
5. **Scale unproven** — demonstrated at 17K–1.5M (read/write) and 0.8B (repair). Nothing at 27B; our Bonsai-27B is PQ2 ternary-packed, where column edits need dequant→edit→requant (breaking packed-quant perf pins) or dequant-domain delta application.
6. **Egress** — weight-as-text means transmitting weights to the authoring model. Private weights (`civ_leo_frozen.json`, Bonsai GGUF, LAYA checkpoints) never cross the wire (Research 003); WRITE-from-scratch crosses only task spec + architecture.

---

## 4. Distillation — the two-layer split

**Mechanism layer (fully modelless):** every fix/read/harness technique reduces to closed-form weight arithmetic + deterministic accept/reject + pinned eval. Extracted inventory (No-GD advocate pass, 17 items, condensed to the load-bearing):

1. **Rank-1 column-masked direction subtraction** (`ΔW[:, S] = −k·d`) — structurally a `LoraPair { reader, writer }` edit; sparse-restricted variant = `QuantErrorLora::SparseErrorBypass` top-K COO shape. On PQ-quantized bases: apply the delta in the **dequant domain at forward** (the QuantErrorLora posture) — never requantize.
2. **Norm-matched random-direction control** — this is **Plan-121 RandOpt** (`θ' = θ + σ·ε(seed)`) with the seed pinned and σ norm-matched; `BanditPruner` ensembles make the null arm a one-liner. Gate on ≥16 seeded arms, not one.
3. **Dose-response + layer isolation** — k-grid monotonicity + neighbor-layer nullity are mandatory gate arms, not diagnostics.
4. **Obstacle-vs-ability → refusal precondition** — our halters are the obstacle detector: `GainCostLoopHalter` (`HaltReason::Oscillation`) + `structural_cot_halt` telemetry measure loop rate in production; low obstacle telemetry ⇒ the edit lane REFUSES ("no obstacle found"). The Gemma control becomes the falsifier arm.
5. **Deterministic bypass insertion** (the ramp fix) — construction = compose existing modules with a gate. ⛔ The blog's position-hard switch-off violates our FSM doctrine (measured precedent: plan-317-class oscillation); the stack port is a **sigmoid drive on measured position**, never a hard threshold beside the FSM. Ternary {-1,0,+1} weights make bypass routing a sign-routing problem — arguably where this technique is strongest in our stack.
6. **Accept/reject harness** — `cargo refine`'s `fix_verify` is the shipped precedent: baseline → apply → re-check → auto-REVERT. Revert correctness gate: rejected edit leaves θ bit-identical (BLAKE3 before/after).
7. **Mint-once-then-freeze** — any LLM-sampled artifact is minted once offline, BLAKE3-committed into a frozen envelope (`MerkleFrozenEnvelope` + `AnchorSidecar` provenance trailer; the laya Q8 sidecar posture at load). Nondeterminism quarantined at mint time; serving stays bit-deterministic. **Recipes are not durable, artifacts are** — API replay is not guaranteed across provider drift, so the pinned bytes are the only source of truth, with prompt/seed/proposer digest as provenance metadata.
8. **Static ⊕ runtime composition** — the edit leaves 33% loop residue; the halters catch it. The 2×2 {edit on/off} × {halter on/off} grid is the experiment; re-calibrate halter thresholds on the edited model (the edit shifts hidden-state statistics).
9. **Proposer-utility gate** — the decisive question: at equal touched-weight budget and equal eval budget, does targeted authoring beat RandOpt search? T2 strict-superiority (paired LCB95 > 0, frozen read). If not, the honest verdict is "a RandOpt rebrand with extra egress" — die with the negative recorded (the Issue-565 precedent).

**Proposer layer (the honest fault line):** authoring quality was purchased with someone else's training run. Local authoring (Bonsai-27B PQ2) is nowhere near the demonstrated capability class — no local path today. Fix/read lanes that need no frontier model (diff-of-means direction extraction is pure arithmetic; the overlay machinery is hand-authorable) are fully local-shippable. This is a provenance disclosure obligation, not a mandate violation: the modelless mandate bans gradient descent in the shipped primitive, and consuming a frozen hosted model is the same class as every shipped hosted-expert lane — but it must never be papered over as "no training involved".

**Fusion (what paper × A × B produces that none alone can):** blog's loop-repair protocol (A) × our `GainCostLoopHalter` telemetry (B) × Plan-121 RandOpt control machinery (C) = a repair lane whose targeting, null-control, and refusal precondition are ALL already-shipped instruments — the only new code is the overlay carrier + the gate battery. Nothing in the literature combines a weight-edit lane with production obstacle telemetry as its arming condition.

**Path 0 inventory (per-track outcomes, no "recorded" tier):**
- Overlay primitive + gate battery (Track A) → **filed**: `.issues/980_weight_edit_overlay_lane.md` (GOAT-tier fusion; full pre-registered proof path inside).
- Weight-production lanes: civ-LEO repair arm keyed to Plan 413 + authored-brain lane (Track B) → **filed**: `riir-train/.issues/665_wsp_weight_production_lane.md`.
- Weights-only audit of frozen artifacts (Track C) → **audited discard**: published prior art (Weight Oracles, trained-reader class) covers the capability; our frozen ternary brains are 8-dim-scale where direct arithmetic reads suffice — no LLM reader needed, no documented in-stack failure to fill. Re-open trigger: a distributed-brain audit need with artifact scale ≥ 10⁵ params, or a documented backdoor/tamper concern on shipped brains.

**Two-sided check (fusion priority #1):** *Healer* — the accept/reject harness is fix_verify-shaped (consumed as precedent, no new lane); TernaryDraftModel `.bits` repair has no documented failure class → not filed. *Decision-serving family* — rethink's LAYA rung is single-question encoder serving with **no autoregressive surface and no documented loop class** (verified this session) → the loop-repair consumer there is **veto-gated**: (documented loop failure) ∧ (open-weights WSP-capable author). The ESC lane is EXACT/stationary — no dynamics-signal wiring here (§1 step 4a respected: the halter telemetry counter is the only mutated state in this lane, and it already ships).

---

## 5. Verdict

**Pinned novelty claim (one sentence):** *a reasoning-authored, deterministic-applied, gate-gated weight-edit overlay lane (propose → dose-response → norm-matched control → keep-if-wins → freeze) whose arming condition, null arm, and refusal precondition are the stack's own shipped instruments (halter telemetry, RandOpt/BanditPruner, obstacle law), for repairing systematic named biases in served models without retraining.*

**Novelty gate (the four questions):**
1. No prior art? **No (qualified)** — components are published (abliteration weight-level line; training-free weight-edit repairs; Weight Oracles; Heretic; the blog itself). The composition + the ternary/PQ dequant-domain adaptation are ours. Fails the Super-GOAT bar on Q1.
2. New behavior class? **Yes** — repair-without-retrain is a class no stack lane has.
3. Product selling point? **Yes** — "systematic named bias repaired by a reasoning-authored deterministic edit, gate-gated, artifact-committed — no retraining" (with the author-capability caveat stated).
4. Force multiplier? **Yes** — freeze/thaw + halter telemetry + RandOpt + QuantErrorLora carriers (≥2 pillars).

**3 solid + Q1 qualified ⇒ GOAT (fusion), not Super-GOAT.**

**Per-track verdicts:**
- **Track A (modelless overlay primitive + gate battery): GOAT (fusion)** — files: katgpt-rs `.issues/980`. Proof path (pre-registered, inside the issue): seeded-degeneracy MLP fixture; falsifier arms = norm-matched random control (≥16 seeds) + low-obstacle control refusal; gate = dose-response monotonicity ∧ control-null ∧ neighbor-layer nullity ∧ retention within ε ∧ flag-off byte-identical boot.
- **Track B (weight-production lanes): Gain** — files: `riir-train/.issues/665` (civ-LEO arm-3 keying to Plan 413's partial band + authored-brain lane; egress law; artifact-durability law).
- **Track C (weights-only audit): audited discard** — reason above; re-open triggers recorded.
- **Track D (weights-as-text provenance read): folded into 980's fixture census** (cheap, local, no frontier dependency at 8-dim brain scale) — not a separate filing.

**MOAT gate:** note in katgpt-rs (base primitive via fusion — correct public home, zero game/chain/shard IP in the primitive); consumer issues routed to the owning repos (riir-train for weight production; katgpt-rs for the generic carrier). No rethink-class content in any public file — the serving-posture application is described as a veto-gated consumer only.

**Demotion law (pre-registered):** if the 980 gate battery fails its control arms, the lane records the negative (Issue-565 precedent) and the overlay carrier demotes to the SparseErrorBypass representation it already is — no zombie lane.

## 6. GOAT gate sketch (for 980 — full form lives in the issue)

- **G1 specificity battery:** (a) dose-response monotone non-increasing over pinned k grid {1,2,4,8}; (b) neighbor-layer isolation null; (c) norm-matched RandOpt control, paired LCB95 > 0 for the targeted edit; (d) **obstacle pre-screen refusal** on a no-degeneracy control fixture (the falsifier); (e) retention: clean-region accuracy held within ε (paired, seeded).
- **G2:** apply-at-load compose cost vs full-tensor bake; flag-off byte-identical boot.
- **G3:** untouched columns bit-identical; regression sweep on untouched behavior.
- **G4:** overlay apply alloc-free (bounded scratch).
- **Quant posture:** dequant-domain apply for PQ bases (never requantize); FP delta sidecar for ternary bases; bit-exactness pinned at load.
