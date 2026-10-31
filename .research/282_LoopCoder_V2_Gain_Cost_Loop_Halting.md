# Research 282: LoopCoder-v2 — Gain/Cost Loop-Count Halting for Looped Transformers

> **Source:** [LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling](https://arxiv.org/abs/2606.18023) — Yang et al. (Beihang/IQuest/Langboat/RUC), arxiv 2606.18023v1, 16 Jun 2026
> **Date:** 2026-06-22
> **Status:** Done
> **Related Research:** 073 (LT2 — architecture we ship), 097 (Training-Free Loop), 266 (FPRM damped fixed-point halting), 273 (ELT elastic any-time)
> **Related Plans:** 108 (LT2 — shipped, GOAT 8/8), 136 (TF-Loop — shipped), 152 (River-Valley Diagnostics — effective rank, shipped), 231 (PathwayTracker stability exit), 283 (Self-Advantage Gate residual halt), 304 (Gain/Cost Loop Halting primitive — this note)
> **Cross-ref (riir-ai):** Research 149 (Per-NPC Gain/Cost Reasoning Depth Guide — the private selling-point doc), riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs (coherence-decay signals = cost curve), riir-ai/crates/riir-engine/src/latent_functor/k_selector.rs (KSelectionBandit — functor-rank selector, complementary granularity)
> **PASS-Redirects (synthesis):** Ouro [arXiv:2510.25741 "Scaling Latent Reasoning via Looped Language Models"] (Zhu et al., Bengio/Eshraghian, v5 Jul 2026) — pre-trained Looped LMs with entropy-regularized LEARNED depth allocation, scaled to 7.7T tokens. This GainCostLoopHalter (Plan 304) is the **runtime modelless analog** of Ouro's trained depth allocator: Ouro bakes the halt policy into base weights via entropy-regularized pre-training; we compute the halt decision per-loop at runtime from gain/cost economics (effective-rank gain vs coherence-decay cost). Same functional goal (per-dispatch adaptive depth), different mechanism (trained vs runtime). §3.5 Path 0 fails for Ouro (the allocator IS the trained objective, not closed-form math). Ouro pre-training recipe → riir-train; the runtime halt capability ships here as the Super-GOAT GainCostLoopHalter.
> **Classification:** Public

---

> **PASS-Redirects (synthesis):** Zhang et al. [arXiv:2510.07880 "Do LLMs Really Need 10+ Thoughts for 'Find the Time 1000 Days Later'? Towards Structural Understanding of LLM Overthinking"] — TRACE's utility curves empirically validate the gain/cost convergence point at the explicit-CoT granularity; its two structural halting heuristics (self-loop-K, backtrack-revisit) distilled in Research 525 → Issue 699.
> **PASS-Redirects (synthesis):** Looped-DiT [arXiv:2609.40305 "Looped Diffusion Transformer"] — halter signal-diff: its adaptive-exit head is TRAINED on realized max-future-gain per loop (y_r = max(0, max_{k>r}(ℓ_r−ℓ_k)/(k−r)), zero if no later exit improves; λ threshold = the compute dial without retraining) — the supervised-target sibling of this note's runtime gain/cost scissors + erank halt signal (Plan 304 ships the runtime half; complementary classes, not a replacement; its few-loop penalty halving at matched mean loop count is the gate the two would share). Table 3 decomposes the loop×CoT complementarity (this note §1.5's super-additivity) per subtask: looping owns constraint resolution (spatial/procedural/multi-relation +6.0..9.4), CoT owns implied-content inference (+3.5..22.2), both compound — the T2I confirmation of §1.5's division of labor.
> **PASS-Redirects (synthesis):** Ekbote, Pagnoni, Sachan, Bhosale, Lewis, Fazel-Zarandi, Rajput & Gonugondla [arXiv:2610.40325 "How to Loop Your Transformer: Loop Scaling, Specialization, and Parallel Inference"] (FAIR/Meta, 29 Oct 2026) — dense ≤1B from-scratch looping survey whose PLT half extends THIS note's line: raw shifted CLP degrades with loop count and "frequently fails even to match Vanilla" (§G.1 — confirming this note's CLP positional-mismatch cost law), and the repair is PARAMETER-FREE: preserve the current-token embedding + separately RMSNorm both streams + combine (fixed 50/50 ≈ rank-8 learned gate within small matched deltas §G.3; shifted-without-e_n diverges §G.4); normalized shifted PLT at L=2 matches Full-Stack Looping while decoding concurrently and — unlike this note's R≥3 regress — shows no sharp decline through L=8 (the scale mismatch, not the loop count, was the tax). Runtime posture unchanged: no CLP-trained checkpoint exists in the stack, the halt-at-crossover law (GainCostLoopHalter, Plan 304) still owns untrained/frozen looping, and the training recipes (per-loop LoRA scaling, placement table, L∈{2,4} working range) ride the riir-train backstop (Plan 421/422 lineage). Sibling redirect lines this date: 414 (stability taxonomy) + 592 (scaling program).

## TL;DR

LoopCoder-v2 trains 7B Parallel Loop Transformer (PLT) coders with R ∈ {1,2,3,4} loops from scratch on 18T tokens and discovers a **strongly non-monotonic loop-count effect**: R=2 improves SWE-bench Verified from 43.0→64.4, but R=3 *regresses* to 27.6 and R=4 to 22.4. The paper explains this via a **gain–cost scissors**: each additional loop provides marginal representational refinement (gain, measured by output-distribution shift Δp(r), attention re-routing D_KL(r), and effective-rank trajectory) that **shrinks monotonically**, while the CLP-induced positional-mismatch cost Ω(r) stays **roughly flat** across loops. Beyond R=2, the flat cost increasingly dominates the shrinking gain — at every extra loop the cost exceeds the gain by 30–45×. The crossover is the principled halt point.

**Distilled for katgpt-rs (modelless, inference-time):** The novel transferable primitive is a **per-loop gain/cost halting criterion** — evaluate marginal refinement (gain) and marginal drift/mismatch (cost) *each loop*, halt when gain < cost. This is strictly more principled than our existing halting mechanisms (FPRM = gain-only fixed-point residual; ELT = static budget; PathwayTracker = stability-only; Self-Advantage Gate = residual-only) because it tracks **two crossing curves** instead of one threshold. The gain signals (effective-rank trajectory, output shift, step size) are partially shipped as River-Valley Diagnostics (Plan 152); the cost signal (coherence decay, drift) is partially shipped as latent_functor re-estimation triggers. The **fusion** — combining these into a per-loop halt decision — is the Super-GOAT.

**Super-GOAT selling point (private guide at riir-ai/.research/149):** *"Per-NPC reasoning depth auto-selected by gain/cost ratio — NPCs still refining productively keep thinking; NPCs whose refinement has stalled or gone oscillatory stop. No fixed loop count, no wasted cycles. One frozen artifact serves a crowd of 10,000 NPCs each running at their individually-optimal depth."*

---

## 1. Paper Core Findings

### 1.1 PLT architecture — parallel loops via CLP + G-SWA

PLT (Parallel Loop Transformer) breaks the sequential dependency between loops via two mechanisms:

- **Cross-Loop Position offset (CLP):** before loop r≥2, the previous loop's hidden states are right-shifted by one token and added back: `B(r) = Embed(x) + shift(h(r-1))`. Token x_i at loop r receives the loop-(r-1) state of token x_{i-1} (its neighbor), not its own. This enables parallel execution across loops (near-single-pass latency).
- **Gated Sliding-Window Attention (G-SWA):** first-loop KV cache is frozen and shared with all subsequent loops (near-constant memory regardless of R). Non-first loops fuse `y_global` (full attention on frozen loop-1 KV) and `y_local` (sliding-window w=64 on current-loop KV) via a per-head sigmoid gate.

### 1.2 The headline empirical result — non-monotonic loop count

| Variant | SWE-bench Verified | Multi-SWE | Avg (10 benchmarks) |
|---------|-------------------|-----------|---------------------|
| Baseline (R=1) | 43.0 | 14.0 | 38.0 |
| **LoopCoder-v2 (R=2)** | **64.4** | **31.0** | **46.5** |
| LoopCoder-v2 (R=3) | 27.6 | 11.0 | 36.9 |
| LoopCoder-v2 (R=4) | 22.4 | 9.3 | 34.3 |

R=2 is competitive with 30B–72B open models on agentic tasks. R=3 regresses *below the non-looped baseline*. The pattern holds across code generation, code reasoning, agentic software engineering, and tool use.

### 1.3 The gain–cost scissors (the core mechanism)

The paper decomposes each loop's net effect into two opposing forces:

**Gain side** (marginal refinement — *shrinks monotonically with r*):

- **Output-distribution shift** Δp(r) = KL(p(r) || p(r-1)) — collapses after loop 2, never recovers
- **Inter-loop attention KL** D_KL(r) — drops sharply after loop 2; attention routing "freezes"
- **Effective rank** erank(h(r)) — *peaks at loop 2*, declines for every deeper loop (representations narrow, not enrich)
- **Step size** δ(r) = ||h(r) - h(r-1)||₂ — shrinks to a mid-depth minimum
- **Angular change** cos θ(r) — alignment of successive updates; **<0 beyond loop 2 means oscillatory (non-convergent) refinement**

**Cost side** (CLP offset mismatch — *stays roughly flat across r*):

- **Intrinsic offset cost** Ω(r) = mean_i ||h(r-1)_i - h(r-1)_{i-1}||₂ — the positional tax CLP imposes at every loop boundary. Empirically **nearly constant** across loops because adjacent token representations remain comparably heterogeneous at every boundary.

**The scissors:** gain collapses exponentially (Δp drops ~30–45× from loop 2 to loop 3+), cost stays flat. Beyond R=2, cost dominates gain by 30–45× at every extra loop. The crossover is the principled halt point.

### 1.4 Practical diagnostic — effective-rank trajectory as lightweight halt signal

> "If effective rank is still rising at the candidate loop (representational diversity is not yet saturated), an additional loop may yield genuine refinement, whereas a rank that has begun to fall signals the onset of narrowing, after which further loops mostly add the fixed CLP offset cost without compensating gain." (§5)

This is the cheapest halt criterion: track erank(h(r)) per loop; halt when it starts declining. No output-head evaluation, no KL computation, no exhaustive sweep — just the singular-value spectrum of the hidden-state matrix.

### 1.5 Latent loops × explicit CoT are complementary (super-additive)

At R=2, the "thinking" variant (explicit CoT + latent loop) gains **+26.9 points** on LiveCodeBench over the instruction-tuned variant (latent loop only) — far exceeding either ingredient in isolation. The two mechanisms operate at different granularities: explicit CoT decomposes a problem into textual steps; latent loop refines the representation underlying each step. They compound.

### 1.6 What this paper is NOT

- It is NOT about standard sequential looped transformers (LT2/Universal Transformer). The Ω(r) offset cost is **PLT-specific**. Standard sequential loops don't have CLP.
- It is NOT a training method paper. The 18T-token from-scratch training is the *experimental vehicle*, not the contribution.
- It does NOT propose a new architecture. PLT was introduced in [Wu et al. 2025, arxiv 2510.24824]. This paper studies *why PLT saturates at R=2*.

---

## 2. Distillation

### 2.1 What we already ship (the prior-art surface)

| LoopCoder-v2 component | Shipped cousin | Evidence |
|---|---|---|
| Weight-shared looped block | `LoopMode::WeightShared { loop_count }` — default-on, GOAT 8/8 | Plan 108, `crates/katgpt-types/src/lib.rs`, `forward_looped` |
| ODE-motivated damped sub-stepping | `LoopMode::TrainingFree` + K-stage RK β=0.5 — default-on | Plan 136, `tf_loop` feature |
| Per-dispatch elastic loop count | `elastic_loop_override: Option<usize>` on `forward_looped()`, clamped to `[loop_min, 2×loop_max]` | Issue 035, Research 273, `Config::effective_loop_count()` |
| L_min floor (refuse exit below) | `Config::loop_min` (default 0→1) | Issue 035, `.docs/02_architecture.md` L670-676 |
| L_max ceiling | `Config::loop_max` (default 0→derive from loop_mode) | Issue 035 |
| **Effective rank computation** | `crates/katgpt-core/src/data_probe/geometry.rs::effective_rank` — default-on, GOAT 25/25 | Plan 152 (River-Valley Diagnostics) |
| Update cosine similarity | River-Valley Diagnostics (subspace ratios, update cos) — default-on | Plan 152 |
| Fixed-point residual halt (gain-only) | FPRM → `fpopt_halt` (planned, Plan 267); Self-Advantage Gate on HLA (Plan 283) | Research 266, Plan 283 Bench 057 |
| Stability-based early exit | `PathwayTracker` — default-on, GOAT 7/7 | Plan 231 |
| Collapse-driven early exit | Collapse-Aware Adaptive Thinking — GOAT 6/6 | Plan 212 |
| Coherence-decay re-estimation trigger | `ReestimationScheduler::tick` — coherence < tau_reest triggers re-derivation | riir-ai riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs, Plan 303 |
| Gain/cost bandit (functor-rank level) | `KSelectionBandit` — UCB1 over K_OPTIONS=[1,2,4,8,16], reward = coherence − α·latency | riir-ai riir-ai/crates/riir-engine/src/latent_functor/k_selector.rs, Plan 318 |
| Per-zone elastic budget | `ReestimationScheduler::set_active_budget`, `set_zone_gating` | riir-ai Research 128, Plan 305 |

### 2.2 The LoopCoder-v2 delta vs shipped prior art

The honest read: **~70% of the ingredients are shipped**, but the specific fusion — a per-loop, per-dispatch, runtime gain/cost halting criterion — is genuinely missing. Our existing primitives each track ONE curve:

| Existing primitive | What it tracks | What it IGNORES |
|---|---|---|
| FPRM (Research 266) | Fixed-point residual (gain/convergence) | Cost/drift |
| ELT override (Issue 035) | Static budget L chosen by caller | Gain AND cost (caller decides blindly) |
| PathwayTracker (Plan 231) | Stability signal | Cost/drift |
| Self-Advantage Gate (Plan 283) | Residual collapse (gain) | Cost/drift |
| ReestimationScheduler (Plan 303) | Coherence decay (cost) | Gain (always re-estimates when cost high) |
| KSelectionBandit (Plan 318) | coherence − α·latency (gain/cost!) | But at functor-rank granularity, learned over ~30 episodes, NOT per-loop incremental |

**KSelectionBandit is the closest cousin** — it does compute a gain/cost ratio (coherence vs latency). But:
1. It selects **functor extraction rank K** (1,2,4,8,16), not **loop count R**.
2. It's a **learned UCB1 policy** (explores arms, converges over ~30 pulls per NPC-relation pair), not a **runtime per-loop halt decision**.
3. It doesn't use the **effective-rank trajectory** or **oscillation detection** (cos θ < 0).

LoopCoder-v2's contribution at inference: a **per-loop incremental** halt that evaluates gain and cost EACH LOOP and stops the moment gain < cost. No learning, no exploration, no episodes — just two crossing curves evaluated in O(1) per loop using signals we mostly already compute.

### 2.3 The gap this fills

We ship looped transformers (LT2), elastic loop overrides (Issue 035), effective-rank diagnostics (Plan 152), coherence-decay triggers (Plan 303), and gain-only halting (FPRM/Self-Advantage). **What we don't ship:** a principled criterion that says "this NPC/inference-path should stop looping now because its marginal refinement (gain) has dropped below its marginal drift cost." Without this:

- A crowd NPC with budget L=4 runs all 4 loops even if loop 2 onward is pure oscillation (cos θ < 0). Wasted compute.
- An important NPC with budget L=2 stops at 2 even if its effective rank is still rising (gain > cost at loop 3 would help). Lost quality.
- The caller of `elastic_loop_override` has no principled way to choose L — they guess or use a static tier mapping.

### 2.4 Latent-space reframing (mandatory per workflow §1 step 3)

Re-casting the gain/cost halting criterion as a latent-to-latent operation on the codebase's latent-state kernels:

| Substrate | Gain signal (refinement) | Cost signal (drift) | Halt rule |
|---|---|---|---|
| **HLA evolution** (`evolve_hla`) | Belief-state step δ = ||h^(r) − h^(r-1)||₂; effective-rank delta | Staleness: sigmoid(−λ·(tick − last_observed)) — the two-brain confidence decay | Halt when δ < staleness |
| **latent_functor application** | Coherence improvement Δc per application | Coherence decay (already tracked in reestimation.rs as the re-estimation trigger) | Halt when Δc < decay_rate |
| **LT2 forward path** | Output shift Δp(r); effective-rank trajectory; attention D_KL(r) | Attention-sink compounding (LT2 SDPA output gate mitigates); drift from trained regime | Halt when erank declines OR Δp(r) < τ |
| **cgsp_runtime curiosity cycle** | Curiosity reduction per cycle | Compute budget + staleness | Halt when curiosity_reduction < staleness |
| **NeuronShard consolidation** (riir-neuron-db) | Memory quality improvement per pass | Compute + staleness | Halt when quality_delta < staleness |

**The strongest landing zone is latent_functor + LT2 forward path** — we already track coherence (cost) and can compute effective rank (gain) from the shipped River-Valley Diagnostics. The fusion is: each functor application / LT2 loop computes both signals; halt when gain < cost.

### 2.5 Fusion (per workflow §1 step 5)

The closest cousins and the proposed Super-GOAT combination:

| Source | What it contributes |
|---|---|
| **LoopCoder-v2 (this paper)** | The gain/cost halting criterion — two crossing curves determine the halt point; effective-rank decline as lightweight diagnostic; oscillation detection (cos θ < 0) |
| **FPRM (Research 266)** | Damped fixed-point iteration — the *stability* mechanism for the loops (without damping, oscillation dominates) |
| **ELT (Research 273) / Issue 035** | Elastic budget [L_min, L_max] — the *range* within which halting operates; `elastic_loop_override` is the wiring point |
| **River-Valley Diagnostics (Plan 152)** | `effective_rank` computation — already shipped, default-on, the *gain signal* |
| **latent_functor re-estimation (Plan 303)** | Coherence-decay tracking — already shipped, the *cost signal* |
| **KSelectionBandit (Plan 318)** | Gain/cost reward shaping (coherence − α·latency) — the *vocabulary* for combining the two curves; complementary granularity (functor-rank vs loop-count) |

**Synthesized Super-GOAT combination:** *Gain/Cost Elastic Loop Halting* — a looped computation (LT2 forward, latent_functor application, HLA evolution) that:
1. Operates within an elastic budget [L_min, L_max] (ELT / Issue 035).
2. Uses damped sub-stepping for loop stability (FPRM / TF-Loop).
3. Computes per-loop **gain** via effective-rank delta + output shift (River-Valley Diagnostics).
4. Computes per-loop **cost** via coherence decay + drift (latent_functor re-estimation signals).
5. **Halts when gain < cost** (LoopCoder-v2 scissors) — automatically, per-dispatch, per-loop.
6. Detects oscillation (cos θ < 0 for P consecutive loops) as an early-halt signal.
7. Falls back to L_max if gain never drops below cost (safety; rare for well-trained models).

This is a **new capability class**: principled, per-dispatch, per-loop adaptive depth based on gain/cost economics — not stability/convergence/budget alone.

---

## 3. Verdict

| Question | Answer |
|---|---|
| Q1. No prior art? | **YES (mechanism).** We ship elastic override (caller chooses L statically), effective rank, coherence-decay re-estimation, KSelectionBandit (functor-rank not loop-count, learned not per-loop), FPRM (gain-only), Self-Advantage Gate (residual-only). None implement per-loop gain/cost halting. |
| Q2. New class of behavior? | **YES.** Current: caller statically chooses L. New: L auto-determined per-loop by gain/cost economics. Oscillatory loops (cos θ<0) halt even with budget remaining — current primitives would keep looping. |
| Q3. Product selling point? | **YES.** "Per-NPC reasoning depth auto-selected by gain/cost ratio — no fixed loop count, no wasted cycles." Crowd NPCs halt at L=1; important NPCs run to L_max; mid-tier halt at individual crossover. |
| Q4. Force multiplier? | **YES.** Connects LT2 (108), TF-Loop (136), ELT override (Issue 035), River-Valley effective rank (152), latent_functor re-estimation (303), KSelectionBandit (318), Self-Advantage Gate (283), PathwayTracker (231), Per-NPC CLR (316). ≥6 pillars. |

**Verdict: Super-GOAT.**

**One-line reasoning:** LoopCoder-v2's gain/cost halting criterion ("scissors" — gain shrinks monotonically, cost stays flat, halt at crossover) is a new decision modality we don't ship; it fuses shipped ingredients (effective rank from Plan 152, coherence decay from Plan 303, elastic override from Issue 035, damping from TF-Loop) into a per-loop, per-dispatch, runtime halt decision that no existing primitive provides; the selling point is per-NPC reasoning depth auto-selected by gain/cost economics, applicable to LT2 loops, latent_functor applications, and HLA evolution.

### Tiers

| Tier | Criteria | Routing |
|---|---|---|
| **Super-GOAT** | Novel mechanism + new capability class + selling point + force multiplier (≥2 pillars) | **← this**. Open primitive → `katgpt-rs/.plans/304`. Private guide → `riir-ai/.research/149`. |
| GOAT | Provable gain, not new class | (not met — new class confirmed) |
| Gain | Incremental | (not met) |
| Pass | Not relevant / training-only | (not met) |

### Mandatory outputs (created in this session)

1. **Open primitive** → `katgpt-rs/.plans/304_gain_cost_loop_halting_primitive.md` — generic `GainCostLoopHalter` kernel, substrate-agnostic.
2. **Architectural guide** → `riir-ai/.research/149_Per_NPC_Gain_Cost_Reasoning_Depth_Guide.md` — private selling-point doc, per-NPC reasoning depth, connection map, latent-vs-raw boundary, validation protocol.
3. **Cross-ref** — this note references the guide; the guide references this note.

### Latent vs raw boundary (per AGENTS.md)

- **Gain/cost computation:** LATENT — operates on hidden-state dynamics (effective rank, step size, coherence). Never decoded to tokens.
- **Halt decision (loop count L):** RAW SCALAR — deterministic given the same hidden states. Safe to sync, safe to commit, safe for deterministic replay.
- **Bridge:** latent gain/cost curves → raw L via `if gain(r) < cost(r) × τ { halt } else { continue }`. Zero-allocation, gateable by feature flag, no sync dependency introduced.
- **Anti-cheat note:** the halt count L is a function of the input (deterministic), so two nodes processing the same NPC state produce the same L. No anti-cheat concern. The gain/cost curves themselves are local (latent), never synced — only the resulting L (raw scalar) crosses the boundary if needed.

---

## TL;DR

LoopCoder-v2 (arxiv 2606.18023) studies PLT (Parallel Loop Transformer) loop-count selection via a gain–cost lens: each loop provides marginal refinement (gain, measured by output shift / effective rank / attention KL) that shrinks monotonically, while the CLP-induced positional-mismatch cost Ω(r) stays flat — the "scissors." R=2 wins (+21 pp SWE-bench), R=3+ regresses because cost dominates gain by 30–45×. **Verdict: Super-GOAT.** The novel transferable primitive is a per-loop gain/cost halting criterion (halt when marginal refinement < marginal drift) that fuses shipped ingredients — effective rank (Plan 152), coherence decay (Plan 303), elastic loop override (Issue 035), damped sub-stepping (TF-Loop Plan 136) — into a new capability class we don't ship: runtime, per-dispatch, per-loop adaptive depth based on gain/cost economics. All four novelty-gate questions pass (no prior art for the mechanism; new class of behavior; product selling point = per-NPC reasoning depth auto-selected; force multiplier across ≥6 pillars). Mandatory outputs: open primitive plan `katgpt-rs/.plans/304`, private guide `riir-ai/.research/149`. Latent-vs-raw boundary: gain/cost curves are latent (local), halt count L is a raw deterministic scalar (safe to sync/replay). The KSelectionBandit (Plan 318) is the closest cousin but operates at functor-rank granularity (learned UCB1, not per-loop incremental) — complementary, not competing.
