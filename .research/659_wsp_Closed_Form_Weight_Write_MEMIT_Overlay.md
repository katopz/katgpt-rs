# Research 659: wsp — Closed-Form Weight-Write Recipe (MEMIT ΔW as a Deterministic Forward-Time Overlay) + Measured Failure Envelope

> **Source:** areu01or00/qwen0.8B-wsp — "wsp: teach a language model facts by writing its weights" — github.com/areu01or00/qwen0.8B-wsp @ `39265dbccfa3122e96dcacde5a7a0e50d63d022c` — ~375 lines Python (`wsp.py` + `run.py`), README + `results/` raw outputs. **NO LICENSE FILE** (all-rights-reserved default → **decision-content-only posture**: mechanisms and measured facts distill; NO code import; outbound-PR fail-closed).
> **Class anchor:** Meng et al. 2022, "Mass-Editing Memory in a Transformer" (MEMIT) — arXiv:2210.07229. The update rule IS MEMIT's; the repo states so itself.
> **Date:** 2026-11-09
> **Status:** Active — **Gain** (GOAT-tier fusion idea; Q1 FAILS — published class; not Super-GOAT). PoC filed: katgpt-rs Issue 990 (proof path + falsification arms + pre-registered floor).
> **Related Research:** 658 (Circuit Grafts — the ADJACENT quadrant: sampling-time logit-bias overlays on a frozen LM; signal-diff §4), 455 (Hebbian construct-side sibling — its PASS-Redirects parked "fact-editing pretrained LLMs in-place" as unactionable future work on 2026-07-24; wsp is the first external evidence that lane works, with a measured fragility envelope), 463 (the quant-error reader-LoRA corrector lane — Issue 565's substrate; the constructor cousin, signal-diff §4), 238 (closed-form-no-backprop grafting — recorded PASS, **gauge-incoherent**: that verdict is about CROSS-model graft transport, where a fitted circuit's basis does not survive moving to another checkpoint; a wsp-style write targets ONE model's own matrices and is evaluated by that same model's forward — no cross-model transport, so gauge coherence is not a validity condition here; the scramble control plays the falsifier role instead), 653 (the carrier's fold law — the carrier's ORIGIN is Research 648), 648 (Weight-Space Programming — **the `weight_edit_overlay` carrier's origin note** + the obstacle-vs-ability law. Missed by this note's first-draft internal grep (keyword miss: "weight-space" vocabulary) — caught by owner challenge 2026-11-09. Its §2 prior-art table surveyed the ADJACENT training-free closed-form edit class (Resisting Harmful Instructions/FORGE/GROM/TRNE/QK-steering/LLM-VA + the abliteration line) WITHOUT naming the MEMIT locate-then-edit family — hence "first MEMIT-family distillation" below stays true qualified. **Exp 8 is the same-surface complementary edit class**: rank-1 loop-repair column subtraction on Qwen3.5-0.8B MLP output columns (repair-an-obstacle) vs wsp's rank-r supervised fact-write (write-a-fact). **Caveat 5 recorded the packed-PQ2 edit problem 2026-11-03** — this note's ternary negative (below) corroborates it; §6 carries the dequant-domain apply + FP delta sidecar posture this note adopts), 655 (Weight-Space Learning/model zoos — population frame, no mechanism-novelty claims; symmetry canonicalization is a DEFERRED conditional dependency (riir-train Plan 487 T1.2, no live consumer). Relevant here as the ZOO-HYGIENE lens: a growing per-fact overlay population is a small model zoo, and 990's envelope metadata row (base digest, hyperparams, per-fact provenance) is exactly 655's Phase-0 hygiene).
> **Related substrate (shipped, verified this session):** katgpt-core `weight_edit_overlay` (Issue 980: `WeightEditSpec`/`EditForm`/`ProvenanceTrailer`/`WeightEditArtifact` + gate battery + GOAT bench; **`forward_delta_into` :505 applies deltas in the DEQUANT domain at forward — never requantizing a packed base; `compose_layer_into`/`revert_layer` :408/:466 is the dense pinned-exact path with BIT-IDENTICAL f32 revert**; arming = `gain_cost_halt` `Halt{Oscillation}` obstacle telemetry, refuses "no obstacle found" otherwise), **`quant_error_lora`** (Issue 565 / Research 463: the deterministically-constructed reader-LoRA CORRECTOR — Strategy A weight-space SVD, **Strategy B output-space data-aware reduced-rank regression `min ‖E·X − B·A·X‖_F` over calibration activations** (Izenman 1975), Strategy D top-K sparse; struct + `thin_svd_into` (sequential one-sided Jacobi, `subspace_phase_gate`)), `sst_certified::factor_tail` (quantization-safe `ΔW = U_⊥ C V_⊥ᵀ`, dropped energy REPORTED), katgpt-types `LoraPair { reader, writer }` + `LoraAdapter` (Plan 025 transport; blake3-checksummed `.bin`), riir-engine `adapters/hot_swap.rs` (`LoRAHotSwap`).
> **Cross-ref (riir-neuron-db):** `.research/303` names MEMIT explicitly — "no retraining, no GD, no MEMIT null-space projection" — the shard-level construct-vs-edit contrast.
> **Classification:** Public — open-primitive layer (generic closed-form math; wsp decision content only, never code).

---

## TL;DR

wsp is a 375-line demonstration that a **fact** can be written into a pretrained decoder's weights with **no gradient descent** — every step a forward pass or a linear solve — and its README honestly measures the failure envelope (2/5 unseen wordings exact on the shipped example; key-template leakage; subject competition; bf16 save rounding; teacher-forced scores misleading vs free generation). Three findings survive the gates:

1. **The class is published** (MEMIT/ROME/AlphaEdit/AnyEdit/WISE + surveys) — Q1 NO, not Super-GOAT. wsp's deltas are engineering increments: per-token key/target pairs for a WHOLE free-form answer; closed-form targets from output embeddings × final-norm scale (no optimized target vector); an on-policy refinement loop over the model's OWN free generation (IRLS-family reweighting + exact re-solve, ≤6 rounds, capped boosts).
2. **The ΔW factors EXACTLY as the sanctioned overlay pair.** `dW = V·Kpᵀ·M⁻¹` with `M = λ·Kn·Knᵀ + Kp·Kpᵀ + μI` ([in×in] SPD) gives `dW [out×in] = V [out×P] · (KpᵀM⁻¹) [P×in]`, rank ≤ P — the Plan-025 `LoraPair { reader, writer }` shape verbatim; truncation to rank r rides the `factor_tail` discipline (dropped energy REPORTED). Shipped as an overlay, the base artifact never moves — which **structurally DELETES wsp's worst measured failure mode** (bf16 save rounding flips borderline tokens) and makes undo **structural** (drop the fact row, recompose `Σ_f B_f(A_f h)`) instead of float-subtractive.
3. **In-lattice writes on quantized artifacts are ARGUED to fail (coordinator-confirmed advocate argument — never measured; wording corrected at the owner's challenge) — but the quantized posture is NOT a future seam: its primitive half ships.** The argument: typical ΔW entries (10⁻³–10⁻² of weight scale) round to zero in a ternary {-s,0,+s} lattice; block scales are frozen; undo is non-invertible in-lattice. Q8_0 (~8 bits) is plausibly viable in-lattice — riir-infer Issue 1189's Cell A can measure the pair honestly. The chosen quantized posture is **overlay-at-dequant** (solve against the dequantized view, apply at forward without requantizing) — chosen on STRUCTURAL grounds (base bytes never move, undo exact, provenance clean) — and that apply path ALREADY EXISTS in katgpt-core: `weight_edit_overlay::forward_delta_into` (:505, "the QuantErrorLora posture — never requantize a packed base", per the Cargo feature row) and `QuantErrorLora`'s own `y = W_q·x + α·correction(x)` inference law. Only SERVING wiring into the riir-infer kernel lanes is future.

Verdict: **Gain**. The note is the workspace's first MEMIT-family **distillation** (R648 §2 surveyed the adjacent training-free closed-form edit class without naming the locate-then-edit family; R303 named MEMIT in a contrast clause only — qualifier added after the owner challenge surfaced the missed 648/655 cousins); the fusion (a new TARGET + protect-weighting on the existing closed-form reader-LoRA constructor family → mint-once-then-freeze envelope → structural undo → shipped dequant-domain apply) composes four shipped pillars; wsp's failure envelope supplies pre-registered gates the PoC must beat.

---

## 1. The mechanism (verified at the pinned sha)

From `wsp.py` @ `39265dbc` (all line refs verified this session):

1. **Read** (`read()`, :154): one teacher-forced pass of question + answer + end token per phrasing; forward hooks capture the `mlp.down_proj` input (the **key**) at every answer position, per edited layer. Default edited layers: 5 ending at `min(n-2, round(0.875·n))` depth (:56).
2. **Targets** (:193–202): per answer token, `d = g · head[token]` (output-embedding row × final-norm per-dim scale `g`, measured by `norm(ones)` — handles plain and zero-centred RMSNorm, :63), sized `Dj = max(0, −logprob) + 3` nats (:201), normalized to unit dot then scaled by residual RMS `r[j]` (:202). **Closed form from the model's own weights — no optimized target vector.**
3. **Protect** (:173–210): per-layer covariance `C` of keys from ~40 general Q&As answered by the model itself (cached under `~/.cache/wsp`, sha1-keyed by model+layers+question-set, :174); look-alike `contrast` questions at weight ×10 (`contrast_weight=10.0`); the question's own tokens; every earlier fact's keys. `Cn[l] = λ·C[l] + λ·(Qk·Qkᵀ + Σ earlier-fact keys) + λ·cw·(contrast keys)` (:207–210).
4. **Solve, spread bottom-up** (`write_all()`, :212–227): MEMIT schedule — each layer delivers `share = missing / (layers_left)`, recomputed after lower layers are written via a fresh read pass with the edit on. `M = Cn[l] + Kp·Kpᵀ; M += μ·mean(diag(M))·I`; `dW = solve(M, Kp·shareᵀ)ᵀ`; **capped at 25% of matrix norm** (`max_rel=0.25`, :224).
5. **Refine, still closed-form** (:229–251): one calibration pass scales all targets by `1/gain` (clamped [0.5, 8]); then ≤6 rounds: free-generate with the edit on, find the first divergent token per phrasing, reweight (`p<0.5 → ×1.5`, `p<0.2 → ×1.4`, `p>0.98 → ×0.85`, first-miss ×2.0, per-token boost capped at 10× calibrated start), re-solve. **IRLS-family: multiplicative data-reweighting + exact re-solve. No gradient of any loss w.r.t. any parameter.**
6. **Save** (:270–303): copy model folder, patch ONLY the edited `mlp.down_proj` safetensors tensors (dtype kept), write `wsp_edit.json` (base model, patched tensor names, layers, λ, per-fact ask/answer/contrast/gain/rel/fit). Per-fact `dW` stored in memory; `undo()` subtracts it (:264–268).

---

## 2. Measured results + failure envelope (their claims, their numbers)

Qwen3.5-0.8B bf16, 4 GB laptop GPU, ~10 min per write, greedy decoding, judged on **free generation for wordings never used in the write** (`results/v9_saved_model_fresh_load.txt`):

| Question class | Result |
|---|---|
| 2 of 5 unseen wordings | answer + GitHub link **word-perfect** |
| 1 of 5 | starts right, then loops |
| 1 of 5 | first clause only |
| 1 of 5 | miss (answers in Chinese about an image model) |
| Protected look-alikes (@karpathy, @sama) | exactly stock (10× weighting held) |
| Unprotected neighbors (@naval, @paulg) | drifted |

Their ablation ladder (v0–v10, README table) is the most valuable part — every row is a gate we get for free:

- **v0 (LoRA SFT)** planted the answer but broke a neighbor — dropped: "the point is no training".
- **v1 (single layer, ×3 scale)**: answer repeats ~20× (no stop learned), 46% of matrix norm — why the end-token target and the 25% cap exist.
- **v2 (ten facts, one layer)**: teacher-forced all held, free generation only 6/11 AND "capital of Japan → Quellin" — **keys at answer positions encode the sentence TEMPLATE**, so a write fired on every capital question. The central failure mode; contrast protection (v3) is the mitigation.
- **v4 (raise-scale loop)**: writes grew to 80–92% of the matrix, text garbled — why the caps bind.
- **v5 (five facts, one subject, one layer)**: 1/5 landed — **subject competition**; "one rich answer per subject works better than many small ones".
- **v7 (per-token reweighting, one layer)**: WORSE (19%) — under a fixed budget, boosting failing tokens starves working ones; **v8 (spread over 5 layers)**: 89% of answer tokens — the multi-layer spread is load-bearing, not decoration.
- **v9 (shipped)**: boost cap 10× + more look-alikes — leaks contained at the cost of unprotected-neighbor drift.
- **v10 (reword the answer to avoid fighting the model)**: WORSE on unseen wordings (loops) despite better teacher-forced scores — **teacher-forced metrics mislead; free-generation is the only honest judge**.
- **Save rounding**: "sonnet 3.1" before save, "3.5" after — `t[key].float() + delta → bf16` moves borderline tokens both ways (:294). Numbers and dates are fragile.

---

## 3. Published prior art (§4 searches — the class is NOT novel)

Pinned searches (run this session, before any novelty claim):

- **MEMIT** — arXiv:2210.07229 (1,534 citations) + memit.baulab.info + kmeng01/memit. The update rule, the multi-layer spread, the protected covariance — all published. wsp cites it.
- **ROME** (rank-one model editing) — the single-fact predecessor, same lab.
- **AlphaEdit** — ICLR 2025 (~417 citations): null-space-constrained editing — projects perturbations onto the null space of preserved-knowledge keys before applying. The published protect-then-solve cousin (stronger preservation machinery than wsp's weighted covariance).
- **AnyEdit** — arXiv:2502.05628 (Feb 2025): decomposes long-form knowledge into sequential chunks and iteratively edits the key token in each chunk — **the published multi-token/whole-answer cousin**.
- **WISE** — lifelong editing with separate side-memory weights + inference-time gating — the published "edit rides as side weights" overlay concept.
- **"A Unified Framework for Model Editing"** (Gupta et al. 2024): formalizes MEMIT's least-squares closed form.
- **Surveys**: Wang et al. 2024 knowledge-editing survey (474 citations); lifelong-editing limits benchmarks (WikiBigEdit); over-attention in editing (arXiv:2502.14838 — edited knowledge over-generalizing, the published face of wsp's template-leakage failure).

**Q1 verdict: NO prior-art novelty.** wsp's increments (whole-answer per-token closed-form targets; on-policy free-generation refinement; the measured failure envelope) are engineering contributions on a saturated published class.

---

## 4. Signal-diff vs shipped substrate (coverage dismissals, mechanism-level)

| Surface | What ships | wsp's shape | Verdict |
|---|---|---|---|
| katgpt-core `quant_error_lora` (Issue 565 / Research 463) | **The closed-form reader-LoRA constructor**: Strategy B solves data-aware reduced-rank regression `min ‖E·X − B·A·X‖_F` over calibration activations `X`, returns the rank-r pair (`a [rank×in]`, `b [out×rank]`, trimmed effective rank); `thin_svd_into` does the deterministic SVD; apply law `y = W_q·x + α·correction(x)` | Solves `min ‖V·Kpᵀ·shareᵀ − dW·Kp‖`-shaped ridge with a PROTECTED covariance in the penalty and targets derived from output embeddings; spread over layers | **Same constructor family, different TARGET + weighting.** wsp's novelty for this stack reduces to: a new regression target (fact-write residuals, not quantization error `E`), a protected/weighted penalty (contrast ×10, question tokens, earlier facts), and the multi-layer schedule. Verdict revised from this note's first draft (which wrongly said "no protected-ridge-delta constructor ships" — caught by verdict round 1). |
| katgpt-core `weight_edit_overlay` (Issue 980) | `EditForm` vocabulary — DISCRETE probe-authored repair forms (rank-1 column-masked `DirectionSubtraction`, exact_sigmoid-gated `BypassRows`) over linear specialist heads; strength via dose sweep + `paired_lcb95`; `ProvenanceTrailer` + BLAKE3 `WeightEditArtifact` (mint-once-then-freeze, FrozenDirectionBank pattern); **apply machinery**: `forward_delta_into` (dequant-domain apply at forward) + `compose_layer_into`/`revert_layer` (dense pinned-exact, bit-identical revert); **arming law**: `gain_cost_halt` `Halt{Oscillation}` obstacle telemetry, refuses "no obstacle found" | CONTINUOUS closed-form ridge solve over transformer MLP matrices; strength by IRLS reweighting; telemetry = teacher-forced hidden states, not behavior pools; **no oscillation obstacle exists for a fact-write** | **Composition on the apply/envelope half; NOT a host for the construction arm.** The arming law is the obstacle-vs-ability falsifier for the REPAIR lane — a fact-write has no obstacle telemetry, so riding the carrier feature would either refuse to arm or need a bypass (verdict-round-1 finding). The PoC therefore ships under a NEW opt-in feature implying `quant_error_lora`, with its own minimal mint-once-then-freeze envelope following the same pattern, CONSTRUCTED-WITHOUT-ARMING (arming stays the apply side's decision, stated on the artifact). |
| Plan 559 Hebbian constructor (DEFAULT-ON, R455) | CONSTRUCTS a new fact-storing bilinear MLP `B(Ax⊙Gx)` from a fact set — construct-then-swap into our own slots; information-optimal capacity | EDITS an existing pretrained matrix in place — write-then-overlay | **Different operation.** R455's PASS-Redirects explicitly parked "fact-editing pretrained LLMs in-place" as unactionable future work (2026-07-24) — wsp is the first external demonstration the lane works at all, with a fragility envelope. That parked row upgrades to "actionable with known failure modes". |
| R658 Circuit Grafts (filed yesterday, Issue 988) | Fitted overlays on a frozen LM intervening at the **SAMPLING distribution** (logit bias at the last layer): probe reads → deterministic rule table → write; procedural SKILL (arithmetic); fitted switches arm per input class; scramble control | Intervenes INSIDE the **forward pass** (residual-stream ΔW at 5 MLP layers): teacher-forced keys → ridge solve → weight write; DECLARATIVE FACT | **Adjacent quadrants, neither covers the other.** Same doctrine ancestor (Plan 025 deterministic overlay); same target-model family appears in both (Qwen3.5-0.8B — and R658 §7 records that neuron-translation into Qwen3.5-0.8B FAILED so far; wsp's weight-write on the same model SUCCEEDED — the weight-space lane is the one that landed on this model). |
| katgpt-types `LoraPair`/`LoraAdapter` (Plan 025) | Transport only: load/apply blake3-checksummed adapters. Plan 025's caveat — "actual trained `.bin` files come from training runs" — is PARTLY OUTDATED since Issue 565: a deterministic constructor ships for the quant-error target | A RECIPE for **deterministically constructing** the pair from a handful of exemplar phrasings — a new TARGET for the same constructor family, no training run | **The gap this note fills is narrower than first drafted: the capture/target/spread half** (teacher-forced key capture, out-embedding-derived targets, protected-covariance ridge penalty, bottom-up schedule, IRLS refinement) — not the whole pipeline. |
| riir-neuron-db R303 | Shard-level construct-vs-edit contrast, names MEMIT as the rejected alternative | The LM-level in-place variant | Consistent; no conflict. |

---

## 5. Path 0 inventory (three-track advocate merge, coordinator-adjudicated)

Advocates ran this session (No-GD + model-based, one spawn round; briefs mechanism-only, no classification leak).

| Component | Ships in stack? | GD-free extractable? | Disposition |
|---|---|---|---|
| Closed-form targets: out-embedding × final-norm scale × RMS (C2) | Formula new; the tensor pieces (wte, final norm) exist in katgpt-core's toy GPT | YES — pure weight reads | **Extract** (PoC) |
| Protected covariance + weighted ridge solve (C3+C4) | The rank-r pair CONSTRUCTOR ships (`quant_error_lora` Strategy B) but its regression is UNWEIGHTED over calibration activations — no protected/weighted penalty, no Cholesky ridge solve | YES — offline SPD Cholesky (fixed row-major loop order, sequential f64 accumulation, no BLAS — Cholesky on an SPD matrix does not pivot), seconds; recompute-not-incremental for byte determinism; stale keyset refuses loud | **Extract the solve; compose the pair struct + SVD** (PoC) |
| Rank-r overlay factorization (C5) | `quant_error_lora` Strategy A/B + `thin_svd_into` ship the constructor + deterministic SVD; `sst_certified::factor_tail` ships the dropped-energy-REPORTED discipline | YES — the pair falls out of the solve exactly; truncate by `thin_svd_into` | **Compose** (fusion core — reuse, never fork) |
| Per-fact delta ledger + manifest + undo (C9) | Carrier ships `ProvenanceTrailer`/`WeightEditArtifact` TYPES but under the carrier feature (drags `gain_cost_halt` + the arming law); LoraPair ships the transport | YES — and **undo becomes structural** (drop the row, recompose) — strictly better than wsp's float subtraction | **Compose the pattern, not the feature**: the PoC's new module carries its own minimal mint-once-then-freeze envelope (same pattern family), constructed-without-arming |
| Bottom-up layer-spread schedule + re-propagation (C6) | Not shipped (multi-layer toys exist) | YES — bounded Gauss-Seidel over ≤5 layers | **Extract** (PoC) |
| On-policy IRLS refinement (C7+C8) | Not shipped (dose sweep is the behavior analog, not a solver reweight) | YES — with determinism gates: byte-identical overlay across runs; monotone-or-decline | **Extract** (PoC, hardest gate load) |
| `down_proj`-input tap seam (C1) | No hidden-state tap ships in the toy forward | YES — one hook seam | **Extract** (PoC; riir-infer seam is a future, separate lane) |
| Ternary/quantized **in-lattice** write | n/a | **NO for ternary** (argued — never measured): rounding shatters the delta; block scales frozen; undo non-invertible. **Q8_0: plausibly VIABLE** (~8 bits) — measurable in riir-infer Issue 1189's Cell A | **Recorded negative for ternary** (coordinator-confirmed advocate finding; NOTE: argued, never measured). Consistent with R648 caveat 5 (recorded 2026-11-03: packed-PQ2 edits need dequant→edit→requant — breaking perf pins — or dequant-domain delta application). **Generalized with care 2026-11-09 (owner challenge "too low bit"): Q4_K_M EXPECTED fragile (argued, unmeasured)** — the overlay-at-dequant posture is chosen on structural grounds (base bytes never move, undo exact) for every quantized target regardless |

**Track routing (coordinator ruling, overruling the model-based advocate's routing note):** the construction math routes **katgpt-rs** — the boundary domain test for the training repo is "does this TRAIN something?", and nothing here does (no GD, no training loop, no optimizer state); deterministic reader-LoRA construction is the katgpt-rs-sanctioned runtime mutation class #2 (AGENTS.md modelless mandate; canonical precedent Plan 313's Path-2 resolution). The serving apply-at-dequant seam is a **riir-infer** future (zero-riir-dep upstream boundary respected — the PoC never touches riir-infer). Corrected-checkpoint PRODUCTION (a real bias fix on a real serving artifact) would later be a riir-train-shaped *plan* if the PoC passes — not filed now; the PoC decides.

---

## 6. The fusion (what none of the pieces alone has)

**Closed-form exemplar solve → rank-r pair (reuse the `quant_error_lora` constructor family + `thin_svd_into`) → mint-once-then-freeze envelope (the `WeightEditArtifact`/FrozenDirectionBank pattern, own minimal instance) → structural undo → apply-at-dequant (the primitive half SHIPS: `forward_delta_into` / the `QuantErrorLora` inference law; serving wiring is the only future).**

The workspace can swap overlays (transport), construct fact-stores from scratch (Hebbian), graft skills at the logit layer (R658), edit linear heads with discrete forms (carrier), and CONSTRUCT rank-r correction pairs in closed form (Issue 565). It cannot yet **write a fact into a pretrained weight matrix** — the capture/target/spread half (teacher-forced keys, out-embedding targets, protected ridge penalty, bottom-up schedule, IRLS refinement) is the missing piece, and wsp supplies both the recipe and the measured gates.
- Shipping the write AS AN OVERLAY (never patching the base artifact) deletes wsp's worst failure mode (bf16 save rounding) *structurally*, makes undo exact, keeps base bytes BLAKE3-stable, and fits the freeze-envelope law.
- The refinement loop is deterministic by construction (fixed schedule `(base, corpus, constants) → overlay bytes`) — byte-identity across runs is gateable per the Issue-940 determinism law; monotone-or-decline stops on non-improving rounds and reports MISS honestly.

**Gate mapping (their failure mode → our gate):**

| Their failure | Our gate |
|---|---|
| Template leakage (capital-of write fired on all capitals) | G1b: zero regression on look-alike contrast classes; falsification arm: corrupt the protected covariance → leakage MUST fire |
| Subject competition | Per-fact ledger; gate per-fact specificity; record "one rich answer per subject" as authoring doctrine |
| bf16 save rounding | Deleted structurally (overlay law; base bytes never move) |
| Teacher-forced scores mislead | Accept/reject judged on free generation only |
| Repeats / no stop (v1) | End-of-turn target token + 25% norm cap from step one |
| Boost starvation (v7) | Multi-layer spread + per-token boost caps (v8/v9 constants), monotone-or-decline |

**Pre-registered floor:** wsp's shipped cell is 2/5 unseen wordings exact on one rich fact — the PoC must beat that rate at fixture scale (scaled honestly by fixture size) or record NEGATIVE.

---

## 7. Verdict + next step

**Gain** (GOAT-tier fusion idea; Q1 fails → not Super-GOAT; no plan filed — PoC first per the ungoat discipline).

Filed: **katgpt-rs Issue 990** — PoC on a ≥2-layer toy decoder: closed-form write → rank-r pair under a NEW opt-in feature `closed_form_weight_write` (implying `quant_error_lora`; deliberately NOT under `weight_edit_overlay` — the carrier's arming law has no fact-write arm, verdict-round-1 finding), gates G1/G1b/G2(byte-identity, platform-recorded)/G3(empty-overlay demote byte-identical)/G4(monotone-or-decline), falsification arms (corrupt covariance; scramble pair), structural-undo exactness, the pinned ≥5/8 floor on 8 held-out trials. A PASS promotes the construction arm; a FAIL records the negative and closes.

### §7.1 — PoC EXECUTED: NEGATIVE at fixture scale (2026-11-09, zed-amo-i990cfw@amo)

T1–T6 all executed on amo (`closed_form_weight_write = ["quant_error_lora"]` + `subspace_phase_gate`; gates `tests/issue990_closed_form_weight_write_gates.rs`, module `crates/katgpt-core/src/closed_form_weight_write.rs`). **Verdict: G1 FAILED — measured 0/8 against the pre-registered ≥5/8 floor; the PoC closes NEGATIVE per T6's FAIL branch.** The module is RETAINED as correct opt-in substrate (issue-565 precedent); default promotion is REFUSED by this verdict.

What the run ESTABLISHED (the machinery is correct):

- **G2 byte-identity PASS** — two constructions → BLAKE3-identical envelope bytes (f64 fixed-order Cholesky, sequential folds, no RNG, no BLAS; the Issue-940 determinism seams hold).
- **G3 PASS** — empty-overlay demote bit-identical to base; base weights digest unchanged across the write (the base artifact never moves).
- **G4 PASS** — refine rounds monotone-or-decline (declines reverted before recording).
- **T5b VACUOUS at close** — scrambled ≤ baseline read 0 ≤ 0, which cannot fail while the real overlay also scores 0/8; the arm only discriminates once G1 > baseline. The gate now carries that precondition explicitly (the assert fires only when the real write beats the baseline) and is recorded non-discriminating here.
- **T5a NON-DISCRIMINATING at close** — G1b already fires WITH protection on at the close posture, so corruption-leaking proves nothing about the protection being load-bearing. Recorded; the gate stays for re-armed runs where G1b holds.
- **T5c pre-registered claim FAILED** (unsatisfiable by construction, not a near-miss): whole-model "recompose-without-row == never-had-the-row, bit-exact logits" does not hold for a multi-layer multi-fact overlay because pairs RE-PROPAGATE — fact B's layer-1 addend changes the x that fact A's layer-2 pair reads, and downstream layers re-amplify the difference into every logit. The scoped substitute asserted instead (test renamed `t5c_structural_undo_rows_byte_identical`): remaining facts' pairs returned byte-identical in canonical order (no float-subtract, no re-derivation), deterministic recomposition, no residue either way. wsp's float-subtract undo remains strictly worse. **Sibling re-anchor: riir-infer Issue 1189's status updated + committed there `3fdaea0`** (990 NEGATIVE → 1189 is the deciding feasibility lane; T5 arms conditional on G1 > baseline; Cell A floor = wsp's own 2/5, never a toy-inherited number). Disclosure: `3fdaea0` also carries an incidental 100%-similarity file rename of a sibling lane's `flashmemory_decode_gate.rs` (content-neutral, swept from the shared index — content verified identical, the sibling's Cargo.toml wiring stays uncommitted on amo).
- **Delivery debug (temporary instrument, removed at close)**: at training keys the solved dW delivers EXACTLY the intended per-position direction (`dW·k_p · u_p ≈ |dW·k_p|`, positive, every position) at uniformly cap-trimmed magnitude (~⅓ of desired at the 25% cap) — the MEMIT ridge solve, the nats-form targets and the overlay apply are all correct.

**The uncapped diagnostic (the verdict's requested run — measured, not hypothesised):** re-running the full construct at `max_rel_norm = 1.0` (the physical ceiling, direction machinery untouched): **G1 still 0/8 while contrast-drift goes 4/4** — removing the cap neither reaches the floor nor preserves locality. The negative is therefore NOT caused by the 25% cap: at this dimensionality, AUTHORITY AND LOCALITY ARE COUPLED — more push moves every salient token's logit (the write cannot push 13/14 up without equally moving the rival tokens whose random base logits dominate). That is the measured failure mechanism; "toy scale" is its shorthand, and the coupling is a property of a 32-dim random-embedding vocabulary, not of the recipe.

The other recorded failure classes (each fixed by one of the ~20 principled fixture regimes, each revealing the next): backward targets (the r[j]-scaled magnitude under-shoots the residual's √D-scaled existing component — replaced with the self-consistent nats form, ours) → cap-bound authority (once cap-bound, refine boosts cannot move the write) → key-overlap blur (near-parallel keys + ridge deliver the AVERAGE share direction, random ± per token) → cross-fact fire (shared/overlapping key manifolds; the earlier-fact protection needs key separation to hold) → self-copy attractors (greedy chains re-capture on the just-generated salient token) → refine escalation (every position boosts toward the cap when fits are equal — the integer-granular fit hides sub-token progress).

So the decisive feasibility test is NOT the toy: **riir-infer Issue 1189's overlay-at-dequant real-scale cells (Qwen3.5-0.8B-Q8_0, then qwen3.8-27B-Q4_K_M)** are now the DECIDING lane for the recipe (990 no longer gates them — see 1189's re-anchored status). Real vocabularies have thousands of near-orthogonal keys where the authority/locality coupling is orders of magnitude weaker; only a real checkpoint adjudicates the recipe. The toy PoC's value is the verified substrate + the failure-mode taxonomy above — every class is a gate a real-scale cell inherits for free.

Application targets (owner-directed 2026-11-09 — "ternary too low bit, try qwen3.8 instead"): the real-scale ladder runs on the DENSE qwen lane, never ternary — **Cell A: `Qwen3.5-0.8B-Base-Q8_0.gguf`** (VERIFIED in riir-train/data; wsp's own model class, so the ~10 min/write cost datum transfers; gated-DeltaNet hybrid — load check is riir-infer 1189's T0) then **Cell B: `qwen3.8-27b-Q4_K_M.gguf`** at the bench-946 pinned path `/Users/katopz/uzu27b/models/` (M5-host path; NOT in riir-train/data on this box — the `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` there is the speculative-decode build, a different artifact). Both OVERLAY-AT-DEQUANT — chosen on structural grounds (base bytes never move, undo exact, provenance clean); in-lattice fragility is ARGUED for ternary and EXPECTED (argued) for Q4_K_M, while Q8_0 is plausibly viable — so 1189's Cell A measures the in-lattice-vs-overlay pair honestly. Filed: **riir-infer Issue 1189** (tap seam + overlay apply + the two real-scale cells; league posture untouched, bench-946 demote checks must stay green).

Cleanup note: the `.raw/qwen0.8B-wsp` clone is deleted at commit time per §0.5; provenance = the pinned sha above (all quotes verified at that sha this session).
