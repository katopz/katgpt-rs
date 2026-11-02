# Research 644: ELR — Elastic Latent Recurrence (One-Loop-Parity Looped LMs)

> **Source:** [One Model to Loop Them All: Elastic Latent Recurrence for Language Models](https://arxiv.org/abs/2610.12582) — Faisal Ladhak, Shantanu Acharya, Santiago Akle Serano, Simeng Sun, Boris Ginsburg, Krishna C Puvvada (NVIDIA), arXiv:2610.12582v1, 8 Oct 2026
> **Date:** 2026-11-02
> **Status:** Done — per-track verdicts: **Track A (modelless inference) PASS** (validated shipped defaults, no new primitive) · **Track B (training) Gain → riir-train Plan 481** · serving fusion recorded below. Not Super-GOAT (Q2 fails — the capability class exists in RINS).
> **Screening:** Research 258 (arxiv walk #14) listed this ID as an ungraded C-record — this note is the full distillation.
> **Related Research:** 073 (LT2), 097 (Training-Free Loop), 273 (ELT — the visual-domain elastic-loop prior), 519 (GRT), 414 (loop stability), 602 (LoopCD), 35 (Attractor/Parcae), 343 (System-1.5)
> **Related Plans:** 108 (LT2, default-on), 136 (tf_loop), 304 (GainCostLoopHalter), 428 (loop stability), 617 (LoopCD, negative) — katgpt-rs; 364 (GRT, CLOSED DONE-NEGATIVE), 324 (Ouro looped pretraining, ACTIVE), 395 (looped flows, OPEN owner-gated) — riir-train
> **Classification:** Public

> **PASS-Redirects (synthesis):** Jeddi, Ciccone, Taati [arXiv:2602.11451 "LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation"] (ICLR 2026) — pre-empts (progress, step-size) loop conditioning (from scratch, shortcut-consistency loss aligning variable-length trajectories); no one-loop-parity claim, no gated state update; ELR's delta is the zero-init graft AT the switch + the mixing geometry. Serves Research 602's class context.
> **PASS-Redirects (synthesis):** Wenkai Chen et al. [arXiv:2606.04438 "LoopMoE: Unifying Iterative Computation with Mixture-of-Experts for Language Modeling"] — IterAdaLN conditioning on iteration index + per-token hidden state (from scratch); fixed depth; reconciles with our tf-loop layer-mode pinning exactly as "Done Right" did (Research 097 redirect).
> **PASS-Redirects (synthesis):** You et al. [arXiv:2609.35748 "Improving Test-Time Scaling with Adaptive Looped Transformers (TaH2)"] (Sep 2026) — independent documentation of the one-loop tax ten days before ELR; evidence the parity property was open landscape-wide.
> **PASS-Redirects (synthesis):** Alabdulmohsin & Zhai [arXiv:2502.07503 "Recursive Inference Scaling" (RINS)] (NeurIPS 2025) — the closest published non-inferiority result: stochastic recursion dropout + depth-specific linear adapters (<1% params) gives "no-regret" (recursion-dropped inference still beats baseline) at 600M. Kills the Super-GOAT Q2 (capability class exists); ELR's delta is parity WITHOUT depth-specific parameters + latent-state loops + budget dial.
> **PASS-Redirects (synthesis):** Yang et al. [arXiv:2605.26733 "STARS: Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models"] (ICML 2026) — Jacobian-spectral-radius regularization + random loop sampling; the stability-constraint sibling of our Bench-699 damping toolbox (Issue 717).

---

## TL;DR

ELR is a training recipe that removes the **one-loop tax** — looped LMs trained with it match a non-recurrent baseline at k=1 (−0.52 pt at 4B, +1.04 at 30B-A3B) and gain +7.25/+4.75 pts at k=4 from the SAME checkpoint. Three load-bearing choices: (1) an **adaptive-delta gated state update** `sᵢ = sᵢ₋₁ + [σ(Wβ·s̃ᵢ+bβ)⊙softplus(γ)]⊙(rᵢ−sᵢ₋₁)` (the only update rule matching baseline at k=1; zero-init → step ≈ 0.35); (2) **within-update mixing of loop counts** — a rotating fraction f≈0.10–0.15 of data-parallel ranks runs k=1 every microbatch while the rest sample k uniformly, so one-loop and k-loop gradients enter ONE optimizer step (the discriminator: synchronized per-step sampling at the same marginal p₁ FAILS parity — loss 1.6871 vs 1.6774); (3) **(progress τ=i/k, budget κ=k/K) conditioning** with zero-initialized AdaLN-style heads grafted at a mid-training switch (late schedule beats from-scratch mixed-depth: 1.16× vs 1.62× baseline compute). Plus: cross-loop MoE route replay (−11.7% update time), last-loop KV caching (2.5×→1× KV memory at k=4, 0–3% score cost), Koi-4B post-trained 18→29 AA-Index by loop count alone.

**For this stack:** the inference surface already ships everything the paper's serving story needs (validation, not gaps); the training recipe is the actionable half — and its one genuinely new mechanism (within-update mixing) is precisely the "mechanism change" Plan 364's DONE-NEGATIVE record names as its reopen trigger. → riir-train Plan 481.

## 1. Paper Core Findings

1. **One-loop tax is real and open** — Huginn (mean recurrence 32) trails its fixed-depth baseline on nearly every task (34.9 vs 46.4 ARC-Easy after 800B tokens); Parcae below baseline at one loop; TaH2 documents it independently. Naive fixes fail: uniform sampling over {1..5} still leaves the one-loop endpoint behind baseline (their §6.4).
2. **Within-update mixing is the discriminator** (Table 3): same marginal distribution over k, three groupings — synchronized (one depth/update): k=1 loss 1.6871; +15% one-loop overrides (≤2 depths/update): 1.6830; independent-rank (≤5 depths/update): 1.6774 — all beat/match baseline (1.6852) only as depths-per-update grows. The gradient-averaging geometry, not the marginal, is load-bearing.
3. **Adaptive delta is the only surviving state update** (Fig 4a): replace/residual/parameter-free rules fail one-loop parity; delta-static (learned per-channel rate, input-independent) ranks second (+0.006–0.007) — input-conditioning of the step size matters.
4. **Delayed mixed-depth + zero-init graft** (Fig 5a, §6.6): switching to mixed depth at 75% of tokens costs ~20% added compute (vs 79% from scratch) while retaining 84% of the five-loop gain; conditioning grafted zero-init AT the switch starts as EXACTLY the one-loop checkpoint (π₁=1, πᵢ>₁=0, identity AdaLN) and BEATS from-scratch mixed-depth at every depth (1.6147 vs 1.6273 at k=5).
5. **Diminishing returns**: loop 2 captures 79–87% of the k=1→4 gain; 3 loops capture 91–96% at 1.87× TTFT.
6. **Serving findings**: k=1 latency ≈ baseline (every unique layer executes once); higher k → SHORTER outputs (−21.8% to −32.7% tokens per rollout — latent depth substitutes for CoT tokens, so per-rollout decode time at k=4 is only ~1.4×); last-loop KV keeps memory at 1× with 0–3% score cost.
7. **Honest caveats in-paper**: conditioning closes most but not all of the gap to fixed-depth specialists at intermediate depths; extrapolation beyond trained depth not evaluated (Huginn reports step-index conditioning hurt extrapolation); per-request adaptive depth is future work.

## 2. What we already ship — validation, not gaps

| ELR component | Shipped equivalent | Signal-diff / status |
|---|---|---|
| I/R/O partition, ~middle half recurrent (14/24/14 of 52) | `LoopMode::WeightShared` + `HybridPattern` (Plan 108) + tf_loop depth-fraction 0.45–0.60 mid-stack window (Plan 136; cross-validated by Skip-a-Layer-or-Loop-It) | Partition matches SMELT's middle-50% and our window default — shipped |
| **Last-loop KV caching** | `CacheStrategy::{First,Last,Mean}` in `src/transformer/tf_loop.rs` (Last = structural default; Mean = Issue 698 T5) **and** `forward_looped`'s full-SDPA path structurally overwrites the same `pos_off` slot every loop (`src/transformer/variants.rs` ~L446–460 — last write wins); linear layers carry AHLA recurrent state (no KV) | **SHIPPED on both loop hosts.** Paper Table 2 validates Last: 0–3% score cost at k=4, KV 2.5×→1×. No work |
| Elastic serving (loop dial) | `Config::effective_loop_count(elastic_override)` clamped `[max(loop_min,1), 2×loop_max]` + `GainCostLoopHalter` (`Halt`/`RefusedFloor`, Plan 304) | Shipped. Paper implication: an ELR-trained checkpoint could relax `loop_min` floors toward 1 (today's floors exist because sub-floor loops collapse — ELT §1.4); config guidance for a future trained checkpoint, not code now |
| Adaptive-delta gated update | `ResidualGate` (zero-init ρ_τ, index-conditioned; convex + conditional variants, Issue 698) + GRT's input-conditioned convex gate (Research 519) + GDN delta rule (`riir-infer` `deltanet/` — time-axis) | Same convex-blend family: `s = s + β⊙(r−s)` ≡ `(1−β)s + βr` = GRT's blend. ELR deltas: softplus(γ) per-channel base rate (unbounded amplitude — breaks our convex-copy bound contract, a Plan-481-arm detail) + zero-init at 0.35 (vs GRT's saturated-OPEN +4 init) + per-token-per-channel. Bench 560 T5.4 transfer rule applies: ELR's gate reads a RECYCLED state (s̃ᵢ = sᵢ₋₁+h_in) → predicts LN-dependence; the Plan-481 gate arm carries the LN rung |
| (τ, κ) budget conditioning (trained) | `ResidualGate` ρ_τ is index-only; `copy_late_schedule(loop_count, g0, gR)` IS budget-parameterized — a modelless schedule-level emulation of budget awareness (external gating vs ELR's internal trained conditioning — different authority point, weaker) | Trained heads → Plan 481 Phase 3 rider |
| Delayed mixed-depth schedule | — (training-side) | Plan 481 |
| Within-update loop mixing | NONE anywhere: `xhc_grt_lm.rs` samples per-step shared (synchronized — the failing control); `nextlat_depth.rs` per-window draw keyed `window_stream(seed, window_idx)` = per-window-FROZEN assignment (batch co-mixes but assignments never rotate) | **The one genuinely new mechanism** → Plan 481 |
| Cross-loop MoE route replay | tf-loop layer-mode routing pin (frozen-MoE thrash prevention, Research 097); SMELT observed the route-recurrence ELR exploits | No looped-MoE consumer in this stack — record only |
| Inference-time loop modulation on FROZEN checkpoints | tf_loop damped-Euler 1/K + Bench 699 damping toolbox | CAVEAT: the paper changes nothing here — its gains require trained-for-looping weights; the 0-for-3 negative class on this stack (Benches 847/850/906, Issue 568) stands; do not re-try |

## 3. Path 0 inventory (merged from the adversarial panel)

| Component | Analog ships? | Extraction? | Verdict |
|---|---|---|---|
| Adaptive-delta update | Partial (convex-gate family) | Constant init γ₀ = ln2/2 ≈ 0.347 extractable; parity property needs GD | Track c |
| Within-update mixing | No | No — the value lives in the schedule↔gradient interaction of the trained minimum (panel §3.5 statement) | Track c — the plan's bet |
| (τ,κ) conditioning | Partial (index gates; schedule emulation) | Signals closed-form; heads learned | Track c rider |
| Delayed schedule | No | Eq-5 cost formula closed-form | Track c schedule |
| Last-loop KV | **Yes** (both hosts) | — | Track a: SHIPPED, validated |
| Route replay | Partial (layer-mode pin) | — | No consumer — record |
| Gain-at-k=2 / shorter-outputs laws | Halter economics exist | Offline spectra on a future checkpoint | Record |
| Measured-signal β gate surrogate (No-GD candidate) | — | Feasible in principle | **Discarded from filing** — auditable reason: rides the 0-for-3 inference-time loop-modulation negative class (Benches 847/850/906, Issue 568); quality claim would need a §3.6 head-to-head PoC; no looped consumer checkpoint exists to bench against. Recorded here as the re-open shape if a looped checkpoint lands |
| Residual-adaptive sub-stepping (Richardson-style, No-GD candidate) | tf_loop fixed 1/K + Bench 699 toolbox | Feasible | Not filed — no consumer demand; issue-at-most if one asks |
| π₁=1 k=1-exactness invariant | Issue-698 T5 k=1-degenerate gate pattern | Test-shaped | Absorbed as a Plan-481 determinism pin |

## 4. Fusion (recorded, contingent on a trained looped checkpoint)

- **Decision-serving (priority 1): ESC × loop dial.** The rethink ESC ladder escalates cheap→think legs ACROSS lanes; an ELR-trained encoder checkpoint makes loop-count an INTRA-checkpoint escalation axis — one vessel, k=1 cheap leg (baseline latency), k=N think leg (+5–7 pts class gain). Composes with the existing `EscalateSpec` grammar; needs Plan 324/C14-class pretraining + a serving re-train (rethink charter, owner-gated). No rethink file filed (smallest-artifact discipline).
- **Game runtime (priority 2): NPC limelight × loops.** COLD-ring NPCs at k=1 (usable only once the tax is gone — today's floors exist because sub-floor loops collapse), HOT-ring at k=N. Precedent: `latent_functor::ReestimationScheduler::set_active_budget` (riir-ai Research 128/136). Contingent on a looped game-model checkpoint.
- **Healer (the two-sided check):** no transformer forward on the healer hot path — nothing. The recipe's decision-serving half is the only priority-1 consumer, and it is contingent.

## 5. Prior-art search (§4, executed)

Per-component verdicts (web advocate, 13 searches; caveat: abstract-level, competitor codebases not cloned): within-update DP-rank mixing — **no published instance found** (adjacent: RINS/STARS/LoopFormer/MoR all at per-example/per-token granularity); adaptive-delta update rule — the delta-rule-with-gating itself is Gated DeltaNet (arXiv:2412.06464, same org) but its use as loop-carried state across weight-shared iterations is un-published elsewhere; zero-init (progress, budget) graft at a mid-pretraining switch — no combination prior (AdaLN-Zero/LLaMA-Adapter/LLaMA-Pro are the generic zero-init-graft priors). One-loop parity itself: open as of Sep 2026 (TaH2). Full combination NOT pre-empted; the novelty is the recipe composition. Q1 partial / Q2 no (RINS) / Q3 contingent / Q4 yes-but-contingent → **not Super-GOAT; Gain.**

## 6. Verdict

- **Track A (modelless): PASS.** Every inference-side component ships (`CacheStrategy` family, elastic override + halter, structural last-loop KV) or is validated by the paper's measurements. No new primitive; no plan. No new issue (the two feasible modelless candidates are recorded above with auditable discard reasons).
- **Track B (training): Gain → `riir-train/.plans/481_elr_mixed_depth_recipe.md`** (Path 0.5). The plan's rank-1 arm re-tests the recorded Bench-562 negative with ELR's mixing fix — Plan 364's own reopen trigger ("reopen = mechanism change") fired by exactly this paper. Cheapest decisive experiment in the stack (≤2 M3 CPU-h).
- **Serving fusion:** recorded in §4, no product files (owner-gated).

PoC: none required for the PASS track (no quality claim granted); the training-track proof path IS Plan 481's pre-registered gates + falsification arms.
