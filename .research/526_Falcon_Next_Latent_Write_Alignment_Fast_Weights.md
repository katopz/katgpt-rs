# Research 526: Falcon — Next-Latent Write Alignment for Fast-Weight Attention (Delta-Rule Online-Ridge Recast)

**Date:** 2026-09-01
**Source:** arXiv:2608.27763 — "Fast Weight Attention for Continual Learning" (Y. Zhang, Ta, J. Zhang, Feng, Li, Y. Zhang, Liu, Yuan, Wang, Gu, Yao — ByteDance Seed / Princeton / Tsinghua). v1, 2026-08-27. Project: github.com/yifanzhang-pro/fast-weight-attention.
**Status:** RECORD — note + trigger-gated riir-train Plan 369 filed (digit-OOD harness open; ablation arms gated). Modelless track: PASS (redirects below). **2026-09-07:** Path-0 rows 6–7 corrected — the chunked delta-rule prefill SHIPPED default-on (riir-ai Issue 734, closed 2026-08-26) after the original grep missed it under different vocabulary (Gram / forward substitution, not WY / TriSolve).
**Verdict:** Per-track split (TTPO lesson). **Tracks a+b (modelless): PASS** — no modelless-adoptable primitive with a consumer. **Track c (model-based): Gain, trigger-gated** — recipe rows → `riir-train/.plans/369_falcon_write_alignment_recipe_backlog.md`.

---

## TL;DR

Falcon recasts the delta-rule / linear-attention family as **online ridge regression under read-after-write (RAW) autoregressive semantics** and derives a 6-variant family. Three deltas vs the recurrence our stack serves (Gated DeltaNet — Bonsai-27B / qwen3.8 path, `riir-infer-core::deltanet::forward::gated_deltanet_step`):

1. **Next-latent write alignment (the headline).** The causal fast-memory training pair under AR prefix prediction is **(φ(k_{t−1}), v_t)** — the newly revealed value written under the *prefix* feature available at prediction time — not the universal same-step (φ(k_t), v_t). Same-step "remains causal, but optimizes a different internal objective." A 2×2 design (RAW/RBW × shifted/same-step, paper Fig. 11) separates timing convention from index shift.
2. **NLMS-normalized ridge writes + ridge-as-decay.** Regression family Falcon-1/2/3: `S_t = (1−η_tλ_t)S_{t−1} + η_t·x_t·r_tᵀ`, residual `r_t = v_t − S_{t−1}ᵀx_t`, normalized step `η_t = β_t/(‖x_t‖²+λ_t+ε)`, β∈(0,2) (per-step descent lemma; β>1 = stable sign-flip for state tracking). **Ridge-as-decay: the carry γ_t = 1−η_tλ_t is *derived* from ridge × step** — forgetting coupled to plasticity, vs GDN's independent learned decay exp(g). Inner-product variants (-A suffix): additive writes with energy normalization; these are the benchmarked ones.
3. **Kernel numerics.** Chunk-parallel WY/Gram + TriSolve with a single-solve residual merge (`B = L⁻¹(V−P)`, both paths share L — one batched TriSolve per chunk instead of two) and log1p-clamped positive-decay renormalization (`α = ηλ clamped < 1−ε_γ`, chunk-local log-prefixes, values rescaled by exp(−u_{i−1}), step sizes by 1/γ, outputs/state rescaled back at chunk exit).

**Empirics (124–130M, 50B-token FineWeb-Edu):** ppl parity-to-marginal — Falcon-1.3 (regression+shift) **17.10** vs Gated DeltaNet **17.32** (best baseline), Mamba-2 17.70, Transformer 17.38; Falcon-1A.3 17.40. Downstream 8-task avg ≈ parity. **The real separation is variable-digit addition OOD length extrapolation** (train widths 1–32, eval teacher-forced 33–48 digits): Falcon-3A.3 **87.2** / Falcon-1A.3 **85.9** vs Mamba-2 **75.2**, RetNet 78–83, Transformer **65.8** — it discriminates *within* the GDN family, where 130M ppl does not. Per-column (Falcon-2) and sliding-regression (Falcon-3) are defined but **not benchmarked**. QK-RMSNorm beats QK-ℓ2 (coordinate magnitudes Θ(1) vs unit-norm; mixed-precision stability).

---

## The alignment claim — novelty check (§4 searches, 3 rounds)

- Test-time regression (Wang/Shi/Fox, arXiv:2501.12352, 60 cites) is the **framing predecessor** — the paper cites it and sits inside its design space ("state update as one step of online regression"). Longhorn (2407.14207), Titans, ATLAS, TTT, MesaNet cited as the internal-objective lineage — all write same-step.
- Searches for the shifted write ("next-latent", "k_{t−1}", "prefix-aligned", "delayed key", "off-by-one" pairing) across delta-rule/fast-weight literature found **no published prior doing the shifted write**. The RAW-vs-RBW observation (the shift is just consistency with read-before-write prefix prediction) appears in the paper as its own contribution ("we identify the fast-memory training pair…").
- **Honest size of the novelty:** it is a one-line index shift in the write stream — an architecture/training-time choice with a clean information argument (bind the revealed target to the feature that predicted it), not a new operator. Its value is the falsifiable claim that this changes length-extrapolation behavior.

---

## Path 0 inventory (three-track adversarial panel merged)

Panel: No-GD advocate (9-row extraction) + model-based advocate (6-row recipe table), coordinator-merged with auditable dispositions.

| # | Component | Advocate finding | Coverage / disposition |
|---|---|---|---|
| 1 | **Next-latent alignment (φ(k_{t−1}), v_t)** | Belief-update semantic; feature-gated inference override conceivable | **Training-bound for served weights** — every K/Q projection in Bonsai/qwen3.8 co-adapted to same-step writes; an inference-time re-pairing shifts a trained model's semantics (never default). Modelless consumer: per-NPC belief kernels are O(d_k·d_v)/tick — at 1000-NPC swarm budget (~25 ns/NPC) a matrix memory is unaffordable at any useful d; the *semantic* (bind revealed target to prediction-time feature) already ships under other substrates: keyed episodic overwrite (swarm memory), closed-form ridge forecasters (KARC R288 — batch, predict-then-correct). → **Plan 369 Arm A (trigger-gated)** |
| 2 | **NLMS step η = β/(‖x‖²+λ+ε)** | O(d) statistic fused into the existing dot product | **Covered in the GDN regime**: keys are L2-normalized ⇒ ‖k̃‖² = 1 ⇒ denominator ≈ constant (the paper itself flags "common ℓ2-normalized DeltaNet variants" as the differing default). Remaining delta is the λ coupling → Arm B |
| 3 | Per-column step sizes (Falcon-2) | per-value-channel plasticity | Paper defines, does NOT benchmark. Gate-structure axis (channel-wise erase/write) shipped via R070 `Kda` variant; R447's DPLR `a=b=k` binding = GOAT FAIL on this stack |
| 4 | Sliding-window mini-batch (Falcon-3) | window Gram λ_max normalizer (≈5 power iters on B×B Gram) | Benchmarks only 3A. Memory-horizon axis covered by R482/R133 cousins; no sliding-window memory consumer in the stack |
| 5 | **Ridge-as-decay γ = 1−ηλ** | principled forgetting where no learned decay exists; robustness economy (γ∈(0,1) by construction, deletes the learned-decay clamp/saturation path, fewer params) | We have *felt* the decay-clamp bug class (Issue 594: double-exp decay in the GGUF converter → flat logits). Requires retrain to adopt. → **Plan 369 Arm B** |
| 6 | log1p clamp chunk-local renorm | pure fp recipe, G2 no-regression | **Status 2026-09-07:** a shipped consumer EXISTS — riir-gpu chunked delta-rule prefill: `deltanet_delta_rule_chunked.rs` (Issue 734 T4, 2026-08-20) + `prefill_cuda_gdn_chunked.rs` (Arm 12, owner sign-off 2026-08-23), chunk driver default-on since `3ef764bfa` (2026-08-21). Its log-space cumulative decay (`Σ log α`, floor −60) covers the same underflow axis; the 0.97×/0.98× negative was the CubeCL e2e arm ONLY (`deltanet_chunked_cubecl.rs` — conv1d shipped, recurrence arm lost). Falcon's exact log1p-clamp + η/γ + value-rescale triple remains unshipped as such |
| 7 | Single-TriSolve merge (B = L⁻¹(V−P)) | −1 batched solve per chunk, exact | **Status 2026-09-07:** the shape ships under other names (vocabulary miss — the original grep for WY/TriSolve returned zero because the shipped kernels say Gram / forward substitution / unit-lower-triangular): `deltanet_delta_rule_chunked.rs` solves (I+X)·U = RHS with the history + value paths MERGED in one RHS (`RHS = β·v − β·γ·(S₀·k)`) — the same single-solve merge — and Arm 12 forms the explicit inverse T = (I+X)⁻¹. Falcon's per-chunk shared-Gram batching (Falcon-2: one Gram, dᵥ channel solves) remains unshipped |
| 8 | β∈(0,2) descent bound | provable non-divergence interval for adaptive β | One-line debug-assert guard; folded into any implementation of #1/#5 |
| 9 | **Variable-digit-addition OOD harness** | eval-only, minutes; discriminates within the GDN family where ppl doesn't | **The immediately actionable modelless artifact — NOT covered.** The stack has no standing length-extrapolation gate for the GDN family (our GDN gates are ppl + pinned logits; R070's RULER numbers are paper-reported). → **Plan 369 T1 (open)** |

---

## Fusion assessment

Closest cousins: **R070** (GDN2 — channel-wise erase/write gates, the delta-rule family note), **R447** (KDA — channel gating + DPLR binding, GOAT FAIL), **R516** (TTT-KVB — "state transition = online learning rule", equivalence kit + associativity predicate), **R288** (KARC — closed-form ridge readout, different layer), **R192** (NextLat belief states — same words, different sense: world-model latent prediction, not write alignment).

Fusion candidate examined: alignment-as-belief-semantic — a per-NPC online NLMS associative memory (`evolve_belief`-adjacent). **Discarded on budget**: the delta step is O(d_k·d_v) per NPC per tick; at the swarm cognition budget (~25 ns/NPC at N=1000) that exceeds headroom by orders of magnitude for any d that carries content; and the *semantic* already exists at game scale under different substrates (keyed episodic overwrite, KARC ridge readout). No paper × cousin combination produced a capability none has alone — alignment × GDN2-gates is precisely Plan 369's arm set, a recipe, not a new primitive.

---

## PoC note (§3.6)

No quality-parity claim is made that requires a PoC. The "covered" claims are **architectural** (algebra: L2-normalized keys ⇒ constant NLMS denominator; grep: no shifted write, no TriSolve, no sliding-window memory in-tree). The paper's own claim (shift helps length extrapolation) is accepted as its evidence at 130M and becomes *our* claim only through Plan 369's gates.

---

## Verdict

### Track a+b (modelless): PASS

No modelless-adoptable primitive with a live consumer. Alignment + normalization are semantics of trained weights (retraining-bound for every served GDN checkpoint); the kernel numerics have no consumer (chunked-recurrence negative stands, riir-ai Issue 734); the game reframe is budget-refuted at matrix dims and semantically covered at game scale. The one modelless artifact worth building is the **digit-OOD harness** (eval-only) — filed as Plan 369 T1, not deferred.

### Track c (model-based): Gain — trigger-gated

`riir-train/.plans/369_falcon_write_alignment_recipe_backlog.md`:
- **T1 (open, immediately executable):** variable-digit-addition OOD harness over existing GDN-hybrid checkpoints (Bonsai ternary q2_0 / qwen3.8) — baselines the production models on the paper's discriminating axis.
- **Arms A/B/C (trigger-gated):** write-alignment shift / ridge-as-decay coupling / QK-RMSNorm-vs-ℓ2 — each gated on a from-scratch GDN-family pretrain lane existing (the stack today trains LoRA/distill on existing checkpoints; no recurrence pretrain lane; the 0.4B Kimi test-arch door recorded retired — riir-train Plan 352, Issue 407). Gate per arm: ppl ≤ baseline −0.10 at matched tokens (paper evidence −0.22 at 130M) AND digit-OOD ≥ baseline +3 (Arm A); parity + OOD-win (Arm B); ppl ≤ ℓ2 arm (Arm C). Losers demoted + recorded.

Not GOAT: no measured gain on the stack. Becomes a GOAT candidate only through Plan 369's gates on a future arch template.

### MOAT gate

- `katgpt-rs`: no open primitive lands (nothing survives the consumer test). No note-side artifact beyond this record.
- `riir-train`: the harness + recipe rows strengthen the recurrence-arch-decision moat **conditional on the pretrain lane trigger** — that condition is the plan's own gate, per R070's "When NOT to Implement" #1 ("If we never train recurrent models from scratch").

### Forbidden-by-default (recorded)

Inference-time re-pairing (writing v_t under k_{t−1}) on a **served, same-step-trained** checkpoint is a semantics change to trained weights — opt-in-experiment-only, never default. Same discipline as R516's inner-steps train/test-mismatch predictor: changing the operator under a frozen checkpoint = serving a different model.

---

## PASS-Redirects (recorded here; one-line blocks added to the cousin notes)

> Closest shipped cousins updated with this paper: **R070** (GDN2 — the delta-rule family note) and **R516** (TTT-KVB — the online-learning-rule framing).

---

## References

- Paper: arXiv:2608.27763 "Fast Weight Attention for Continual Learning" (Zhang et al., 2026)
- Test-time regression framework: arXiv:2501.12352 (Wang, Shi, Fox) — the framing predecessor
- DeltaNet: Schlag et al., "Linear Transformers Are Secretly Fast Weight Programmers" (ICML 2021)
- Gated DeltaNet: arXiv:2412.06464 (Yang, Kautz, Hatamizadeh) — the served recurrence
- Longhorn: arXiv:2407.14207; Titans: arXiv:2501.00663; ATLAS: arXiv:2505.23735; MesaNet: arXiv:2506.05233
- Stack: R070 (GDN2), R447 (Kimi K3/KDA), R516 (TTT-KVB), R288 (KARC ridge), R192 (NextLat)
- Plan: `riir-train/.plans/369_falcon_write_alignment_recipe_backlog.md`
- External repo: github.com/yifanzhang-pro/fast-weight-attention @ e4f04b302ee0e669311d59f39093df0d88ba5fb4 (Apache-2.0; project page only — README + paper PDF, no kernel source)
