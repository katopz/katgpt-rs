# Research 273: ELT — Elastic Looped Transformers for Visual Generation

> **Source:** [ELT: Elastic Looped Transformers for Visual Generation](https://arxiv.org/pdf/2604.09168) — Sahil Goyal, Swayam Agrawal, Gautham Govind, Anil Prateek, Jain Sujoy Paul, Aditya Kusupati (Google), arxiv 2604.09168v2, 13 Apr 2026
> **Date:** 2026-06-20
> **Status:** Done
> **Validation:** Any-Time property **empirically validated** 2026-07-16 (`tests/issue_156_anytime_lt2_poc.rs` — permanent regression guard). All 4 gate regimes exhibit monotonic KL decrease as R → R_max with random untrained weights — structural Any-Time holds WITHOUT the ILSD training ELT requires. (Issue 156 resolved and removed per noise-reduction rule; full PoC results table preserved in this note's git history + the test file.)
> **Related Research:** 073 (LT2 — architecture we already ship), 097 (Training-Free Loop), 114 (AMUSE Anytime Muon), 148 (HydraBudget early_exit_layer), 051 (Mosaic of Small Elastic Models)
> **Related Plans:** 108 (LT2 Looped Pipeline — ✅ shipped, GOAT 8/8, default-on), 136 (Training-Free Loop Wrapper — ✅ shipped), 212 (Collapse-Aware early exit), 231 (PathwayTracker stability early exit), 283 (Self-Advantage Gate)
> **PASS-Redirects (synthesis):** Loopie [arXiv:2607.16051 "Loop the Loopies!"] — training-only layer-loop ordering variant, no inference-quality evidence; we already ship elastic any-time loop counts via LT2 + ReestimationScheduler.
> **PASS-Redirects (synthesis):** MoR — Mixture-of-Recursions [arXiv:2507.10524 "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation"] (Bae et al., KAIST/Google, Jul 2025) — extends LT2's weight-shared looping with per-TOKEN adaptive recursion depth via a trained router (vs our per-DISPATCH elastic loop_count via ELT/Issue 035/GainCostHalter). The test-time scaling finding (§5.3: more recursion steps at inference improves log-likelihood) is the per-token analog of our per-dispatch Any-Time property — but requires a MoR-pretrained model with a co-trained router. §3.5 Path 0 fails (router θ is trained, not closed-form). Router training + MoR pretraining → **riir-train**; inference-time Any-Time is already shipped via LT2 + Issue 035 elastic-loop-override + GainCostLoopHalter (Plan 304).
> **PASS-Redirects (synthesis):** Li, Li & Zhou [arXiv:2606.06574 "Skip a Layer or Loop It? Learning Program-of-Layers in LLMs"] (ICML 2026) — Pass. Per-INPUT layer-program prediction (segment-level skip/keep/repeat) is the segment sibling of MoR's per-token router: same disposition — trained predictor → riir-train, inference-time budget adaptivity already ships (ELT/Issue 035 elastic override + GainCostLoopHalter + HydraSkipPlan). Their F3 (monotone accuracy vs latent recurrence budget, Fig 5a) is the program-level restatement of our Any-Time property (Issue 156 PoC). Recipe crumbs for the riir-train backstop: down-weight full-depth labels when shorter valid programs exist (search-distillation label preference) + contiguous-segment action-space compression for MCTS branching.
> **PASS-Redirects (synthesis):** GRT — Gated Recurrent Transformers [arXiv:2608.15062 "Gated Recurrent Transformers: Expressive Depth through Recurrent Modulation"] (Hegazy/Alanwar/Elhoushi, Aug 2026) — 5th paper in this family; beats MoR/RRT/Ouro/Poisson via an input-conditioned convex gate σ(f([LN h, LN h_pre])) + per-step noise + fixed-anchor re-injection + uniform depth sampling (emergent early exit, NO aux loss — strictly simpler than ELT's ILSD). Gain both tracks: modelless corollaries (fixed anchor, convex copy-late schedule, l_min=2/concavity halter floors, KV-mean across steps) → katgpt-rs `.issues/698`; training recipe (depth sampling, gate-init, GRT-micro arms) → `riir-train/.plans/364`. Full distillation + the static-vs-conditioned `ResidualGate` signal-diff: katgpt-rs `.research/519`.
> **Successor paper (2026-10-07):** LiFT [arXiv:2610.05538 "Loop Flow Transformers"] — stages each loop's target on the straight line from the DETACHED ZERO-DEPTH estimate `sg(u0)` to the flow target (`ū_s = (1−s)·sg(u0) + s·u*`; vs ELT's ILSD distilling intermediates toward the max-depth teacher) + endpoint-pinned random interior depth grids + λ=0.01 aux prelude loss: MEASURED monotone depth extrapolation to 8–10.7× training loops (FID 17.07→9.31) where ELT-class full-output targets regress beyond training depth. Distilled → riir-train `.research/468` + Plan 395 arm A; modelless residue (min-action floor, η straightness, chord stalemate monitor) → katgpt-rs `.issues/922`.
> **Cross-ref (riir-ai):** Research 128 (Zone-Density Dynamic Functor Gating — elastic budget per zone), Research 136 (Per-NPC Runtime Test-Time Scaling Guide), riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs (`set_active_budget`, `set_zone_gating`)
> **PASS-Redirects (synthesis):** Vakde [blog "Looped mdlARC", Nov 2026, https://mvakde.github.io/blog/looped-mdlarc/] — the plain-loop family member on discrete autoregressive decoding: weight-tied middle-block loop (×6 from 8 blocks, 28 passes), trained-at-6-runs-3–8 robustness (the Any-Time property's trained member, saturating at ×6), deep supervision + input injection within seed noise (±4.25 pts), iteration embeddings/gates/per-pass norms neutral; per-task embeddings halve the loop gain — recipe laws + constraints in riir-train `.research/494`. No new primitive — the family ships (`forward_looped`, Plan 108).
> **Classification:** Public

---

## TL;DR

ELT trains a weight-shared transformer loop so that **any prefix of loops L_int ∈ [L_min, L_max] yields a useful output** — "Any-Time inference" from a single artifact. The mechanism is Intra-Loop Self Distillation (ILSD): a stochastic student path (exits at random L_int) is supervised by the teacher path (full L_max), both sharing parameters Θ, with `λ` curriculum decaying from GT-anchored to distillation-anchored over training. Result: 4× parameter reduction at iso-inference-compute, FID 2.0 on ImageNet 256², FVD 72.8 on UCF-101; one artifact serves every compute tier without retraining.

**Distilled for katgpt-rs (modelless, inference-time):** The architecture is already shipped as **LT2 (Plan 108, `LoopMode::WeightShared`)**. The ILSD training algorithm → **riir-train** (backprop through shared Θ, training-time only — out of scope per 4-repo strategy). The transferable inference primitive — *elastic loop count L per dispatch, with intermediate states being valid belief states* — is partially shipped via `latent_functor::ReestimationScheduler::set_active_budget` + `set_zone_gating` (per-zone elastic budget) and Per-NPC Runtime Test-Time Scaling (riir-ai Research 136). The only genuinely missing piece is **per-dispatch elastic `loop_count` on the LT2 forward path** driven by compute tier / NPC importance. That's a small coordination layer on top of LT2, not a new capability class.

---

## 1. Paper Core Findings

### 1.1 Architecture — weight-shared looping (we already ship this)

A composite block `g_Θ` of N unique transformer layers is applied L times:

```
F_(N,L)(x) = g_Θ^L(x)   // L loops, N unique layers, effective depth N×L
```

Parameter count is bounded by N; depth D = N×L scales with L. This is **exactly `LoopMode::WeightShared { loop_count }`** in katgpt-rs (`crates/katgpt-types/src/lib.rs`), shipped in Plan 108, default-on, GOAT 8/8. ELT adds nothing new architecturally over our LT2.

### 1.2 Intra-Loop Self Distillation (ILSD) — the training contribution → riir-train

ELT's training loss per step (sampled student loop count `L_int ∼ U(L_min, L_max)`):

```
L_ILSD = L_GT(F_(N,L_max)(x), y)                        // teacher ground-truth
       + λ · L_GT(F_(N,L_int)(x), y)                     // student ground-truth
       + (1-λ) · L_dist(F_(N,L_int)(x), sg(F_(N,L_max))) // intra-loop distillation
```

with `λ` linearly decayed 1→0 over training (anchor to GT early, switch to teacher distillation late), and `sg` = stop-grad on teacher. The student trajectory `L_int` is a **strict prefix** of the teacher trajectory `L_max`, so distillation adds ~zero training overhead.

**This is a training algorithm with backprop through shared Θ.** Per the 4-repo strategy (`003_Commercial_Open_Source_Strategy_Verdict.md`) and the modelless-first constraint, this is **→ riir-train**. Not distilled here.

### 1.3 Any-Time inference — the transferable inference primitive

The payoff of ILSD: at inference, the same artifact serves any compute budget `L ∈ [L_min, L_max]` by simply exiting the loop early. Vanilla looped transformers collapse when L ≠ L_train; ELT-trained models remain coherent across the whole spectrum. Pareto front FID vs GFLOPs is traversable at test time without retraining.

### 1.4 The L_min floor

`1N × 32L` (a single unique layer looped 32×) fails — FID 10.30 vs 2.83 for `16N × 2L`. A minimum block size N is required for representational capacity. The paper analogously requires L_min ≥ some floor before elastic exit is permitted.

### 1.5 Modest extrapolation beyond L_max

On UCF-101 (N=6, L_max=4), peak FVD 69.20 occurs at L=6 (not L_max=4). ILSD regularizes the iterative process enough that modest over-looping works. Quality eventually deteriorates for L >> L_max.

---

## 2. Distillation

### 2.1 What we already ship — do NOT reimplement

| ELT component | Our shipped equivalent | Evidence |
|---|---|---|
| Weight-shared looping `g_Θ^L` | `LoopMode::WeightShared { loop_count }` (Plan 108 / Research 073) | katgpt-rs/.docs/02_architecture.md, default-on, GOAT 8/8 |
| Hybrid dispatch (full + linear interleave) | `HybridPattern::Interleave { full_ratio }` + AHLA | Plan 108, `forward_looped()` in transformer.rs |
| Per-loop residual gate ρ_τ | `ResidualGate` (zero-init) | Plan 108 |
| SDPA output gate | `SdpaOutputGate` (zero-init sigmoid) | Plan 108 |
| Elastic budget per zone | `ReestimationScheduler::set_active_budget`, `set_zone_gating` | riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs (lines 567–594) |
| Zone-density → budget mapping | `ZoneGatingProfile { tiers }` — `I_d → (τ, β, reest_budget)` | riir-ai/.research/128 (default-on for `latent_functor`) |
| Per-NPC compute dispatch | Per-NPC saCLR cycle, freeze/thaw per NPC | riir-ai/.research/136 |
| Stability-based early exit | `PathwayTracker` (Plan 231, GOAT 7/7, default-on) | katgpt-rs/.benchmarks/231_pathway_tracker_goat.md |
| Dead-compute detector | `Self-Advantage Gate` (Plan 283, arxiv 2511.16886) | katgpt-rs/.benchmarks/056_self_advantage_gate.md |
| Collapse-driven early exit | `Collapse-Aware Adaptive Thinking` (Plan 212, GOAT 6/6) | katgpt-rs/.benchmarks/212_collapse_aware_goat.md |
| `early_exit_layer` field | `HydraBudgetResult::early_exit_layer` | katgpt-rs/.docs/02_architecture.md L1597–1601 |
| Training-free loop sub-stepping | `LoopMode::TrainingFree` (Plan 136, Research 097) | katgpt-rs/.docs/01_overview.md L588–592 |

The ELT architecture is **subsumed by LT2**. The Any-Time inference property is **partially shipped** via the latent_functor budget mechanics (elastic per zone) and per-NPC test-time scaling (elastic per NPC). The ILSD training method → riir-train.

### 2.2 What's genuinely new — intermediate-state validity as a *runtime* property

ELT's strongest transferable claim is conceptual: **"intermediate loop states are themselves valid belief states, not just intermediate computations to be discarded on early exit."** In our stack, this reframes two existing kernels:

- **HLA evolution** (`katgpt-rs/crates/katgpt-sense/src/reconstruction.rs` — `evolve_hla`): the recurrent belief-state kernel. ELT framing: any prefix of `evolve_hla` applications is a valid belief state, not just a step toward one. This justifies **elastic HLA depth** — exit early when belief stabilizes (which PathwayTracker already detects, but the framing as "belief-valid prefix" is new vocabulary).
- **latent_functor applications** (`riir-ai/crates/riir-engine/src/latent_functor/`): functor application is by definition weight-shared block reuse. ELT framing: any prefix of functor applications is a valid operator-valued latent state. This aligns with the `set_active_budget` mechanism but adds the claim that **the prefix-state itself is a valid functor**, not just a budget-truncated approximation.

This is a **reframing**, not a new mechanism — the kernels already produce intermediate states; we just don't currently market them as "valid belief states in their own right."

### 2.3 The only missing piece — per-dispatch elastic `loop_count`

LT2's `loop_count` is static per Config. ELT's contribution at inference is making it **per-dispatch dynamic**: the same artifact serves requests with different `L` based on compute tier. Our existing per-NPC and per-zone budget gating operates on the **re-estimation scheduler** and **CLR cycles**, not on the LT2 forward-pass loop count itself.

A minimal Gain-tier fusion would be: **per-dispatch elastic LT2 `loop_count` driven by the same signals that already drive `set_active_budget`** (zone density, NPC importance, compute tier). Plasma-tier crowd NPCs run L_min; hot-tier important NPCs run L_max; same frozen artifact. No new training, no new weights.

### 2.4 Fusion

The closest cousins and the proposed combination:

| Source | What it contributes |
|---|---|
| **ELT (this paper)** | Any-Time framing: intermediate loop states are valid belief states; L_min floor; λ curriculum concept (maps to runtime: heuristic-gate early → distillation-gate late) |
| **LT2 (Plan 108 / Research 073)** | The architecture we already ship — `LoopMode::WeightShared`, `HybridPattern`, `ResidualGate`, `SdpaOutputGate` |
| **latent_functor / reestimation (Plan 303, riir-ai Research 128)** | Per-zone elastic budget (`set_active_budget`, `set_zone_gating`, `ZoneGatingProfile`); archetype hibernation below `I_d<1.0` |
| **Per-NPC Test-Time Scaling (riir-ai Research 136)** | Per-NPC dispatch with freeze/thaw cycle |
| **PathwayTracker (Plan 231)** | Stability-based exit signal for the elastic L |

**Synthesized combination:** *Any-Time LT2 Dispatch* — expose `loop_count` as a per-dispatch override on `forward_looped()`, gated by a `ElasticLoopBudget` config sourced from the same signals that feed `set_active_budget` today (zone density, NPC tier, compute budget). PathwayTracker signals when stability has been reached, allowing early L_exit < L_max. Below some L_min (analogous to ELT's L_min floor), elastic exit is refused and the full L_max runs.

**This fusion does NOT produce a new capability class** — it's a coordination layer that connects LT2 (forward path) to the existing per-NPC / per-zone budget signals (which currently only affect the re-estimation scheduler and CLR cycles). Honest assessment: Gain-tier. The pieces are all shipped; the synthesis is small.

---

## 3. Verdict

| Question | Answer |
|---|---|
| Q1. No prior art? | **NO** — architecture shipped (LT2), elastic budget shipped (latent_functor), per-NPC dispatch shipped (Research 136), early-exit detectors shipped (Plan 212/231/283). |
| Q2. New class of behavior? | **NO** — elastic depth per dispatch is a coordination layer on LT2, not a new capability. |
| Q3. Product selling point? | Weak — "one artifact serves multiple compute tiers" is already implicit in our freeze/thaw + per-NPC scaling story. |
| Q4. Force multiplier? | Limited — connects LT2 forward path to existing budget signals, but doesn't open new pillars. |

**Verdict: Gain.**

**One-line reasoning:** ELT's architecture is shipped as LT2 (Plan 108, GOAT 8/8, default-on); ELT's training algorithm (ILSD) → riir-train (out of scope); the only transferable inference primitive — per-dispatch elastic `loop_count` driven by existing budget signals — is a small coordination layer, not a new capability class. Document for the Any-Time vocabulary and the L_min floor concept; no plan, no riir-ai guide.

### Tiers

| Tier | Criteria | Routing |
|---|---|---|
| **Super-GOAT** | Novel mechanism + new capability class + selling point + force multiplier | (not met) |
| **GOAT** | Provable gain over existing approach, promotes to default if it wins | (not met — mechanism shipped) |
| **Gain** | Incremental improvement, useful framing | **← this** — note recorded, no plan, ILSD → riir-train |
| **Pass** | Not relevant / training-only | (close call — ILSD itself is training-only, but the Any-Time framing is useful) |

### Routing decisions

- **ILSD training algorithm (§1.2)** → **riir-train** (backprop through shared Θ, training-time). Out of scope for this workflow — one-line note recorded here, no files created in riir-train this session.
- **LT2 architecture (§1.1)** → already shipped (Plan 108, Research 073). No action.
- **Per-dispatch elastic `loop_count` (§2.3)** → optional small plan in katgpt-rs if pursued. **Not created** — defer to user. If user wants it, create `katgpt-rs/.issues/NNN_any_time_lt2_dispatch.md` as an optimization task per AGENTS.md ("Create issue at ./issues for optimization task, do not create plan").

---

## TL;DR

ELT = weight-shared transformer loops (architecture we already ship as LT2/Plan 108) + Intra-Loop Self Distillation (training algorithm → riir-train) → Any-Time inference (one artifact serves multiple compute budgets). Three-layer prior-art check confirms our stack already covers the architecture (LT2 GOAT 8/8 default-on), the elastic budget per zone (`riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs` + Research 128), and per-NPC dispatch (Research 136). The only genuinely missing piece — per-dispatch elastic `loop_count` on the LT2 forward path driven by existing budget signals — is a small coordination layer, not a new capability class. **Verdict: Gain.** Research note recorded for the Any-Time vocabulary and L_min floor concept. ILSD training method routed to riir-train (not distilled here). No plan, no riir-ai guide, no Super-GOAT claim — Q1 (no prior art) decisively fails.
