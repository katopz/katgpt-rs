# Research 319: Paired Token-Level Loss Gap & Discourse-State Diagnostic

> **Source:** [Comparing Transformers and Hybrid Models at the Token Level](https://arxiv.org/pdf/2606.20936) — Yanhong Li, William Merrill (Allen Institute for AI), arXiv:2606.20936v1, Jun 2026
> **Date:** 2026-06-27
> **Status:** Active — GOAT (paper alone) + GOAT (fusion)
> **Related Research:** 242 (Topological State Tracking Recurrent Belief — the *theoretical predecessor*; this paper is its empirical validation), 192 (NextLat belief-state dynamics), 070 (Gated DeltaNet-2), 036 (Luce Megakernel Hybrid DeltaNet/Attention), 097 (Training-Free Looped Transformers), 230 (SSD duality)
> **Related Plans:** 335 (this doc's plan — paired loss gap diagnostic primitive)
> **Cross-ref (riir-ai):** Research 127 (Implicit Microcognition Crowd-NPC Guide — design context for HLA-as-discourse-state); the theoretical validation of raw-vs-latent split + HLA lands here as private design context.
> **Classification:** Public — generic modelless evaluation primitive, no game/chain/shard semantics

---

## TL;DR

Li & Merrill (AI2) take two matched 7B models from the same recipe family — **Olmo 3** (pure transformer) and **Olmo Hybrid** (attention + GDN recurrent) — and compare them at the **token level** using the paired per-position loss gap `Δ_i = ℓ_Tr(x_i | x_<i) − ℓ_Hyb(x_i | x_<i)`. Three patterns recur across prose, code, and markup: (i) the hybrid edge concentrates on **open-class content words** (state-conditioned readout); (ii) **opening delimiters are hybrid-favored while closing delimiters are transformer-favored** (state-update vs state-closure asymmetry); (iii) the hybrid advantage **vanishes on visible-prefix copying** (repeated n-grams). The paper formalizes this with **Proposition 1**: `DKL(p⋆_τ ‖ p_ϕ,τ) ≤ log|V_τ|` — the reducible loss from any richer feature map is bounded by the log-vocabulary-size of the target token class. The missing feature for non-copy open-class tokens is **discourse state**: `δ_j = Update(δ_{j-1}, sent_j)`. They close with **filtered evaluations** (TOP-10∩NO-COPY vs COPY-5-ONLY) as higher-signal architecture diagnostics during pretraining.

**Distilled for katgpt-rs (modelless, inference-time):**

This is the **empirical validation** of Research 242's theoretical framing (Mozer et al. 2026, arXiv:2604.17121 — the topological state-tracking diagnosis). R242 already established that HLA `evolve_hla` is the discourse-state Update operator and that OLMo Hybrid (Merrill 2026) is the empirical realization. This paper adds three transferable, inference-time-only artifacts our codebase does NOT have:

1. **A reusable diagnostic primitive: paired token-level loss gap.** Given two forward passes (or two log-prob sequences) over the same prefixes, compute `Δ_i` per token, then stratify by tag class (open-class content, closed-class function, copy/n-gram, bracket open/close). This is a measurement tool, not an inference mechanism — but it makes our GOAT gates (every new primitive ships with a benchmark proving the gain) **sharper**: instead of aggregate loss hiding a 0.04-nat gap, the filtered view amplifies it.
2. **Proposition 1 as a theoretical tool**: `DKL ≤ log|V_τ|`. This is a *volume-of-support* bound that formally justifies our raw-vs-latent sync boundary — physical state (small `V_τ`, e.g., boolean/u8) has near-zero reducible loss, so LatCal raw commitment is information-theoretically sufficient; semantic state (large `V_τ`) is where latent encoding earns its keep.
3. **Filtered evaluation modes**: `ALL_TOKENS`, `TOP-K∩NO-COPY`, `COPY-N-ONLY`. Computed from the same per-token NLL as standard validation — negligible overhead, capability-resolved view.

---

## 1. Paper Core Findings

### 1.1 The paired token-level gap (§3)

For target position `i` with prefix `x_<i` and observed token `x_i`, both models are scored on the same prefix and same target:

```
ℓ^Tr_i = −log p^Tr(x_i | x_<i),    ℓ^Hyb_i = −log p^Hyb(x_i | x_<i)
Δ_i = ℓ^Tr_i − ℓ^Hyb_i    (Δ_i > 0 ⇔ hybrid assigns higher probability)
```

This moves the question from "which model has lower average loss?" to "which *prediction events* produce the gain?". All analyses are forward-pass-only on released checkpoints (Olmo 3 7B, Olmo Hybrid 7B — same tokenizer, data mixture, training recipe; the gap primarily reflects the architectural difference = the sequence mixer). **No training.** ~100 GPU-hours for the full suite.

### 1.2 Three recurring empirical patterns (§4)

**Pattern (i): hybrid advantage concentrates on open-class content words.** In prose, content words have a raw gap of 0.0384 nats vs function words at 0.0238 nats (61% larger). The content–function contrast survives regression controls (difficulty, frequency, position, subword status, local reuse, previous-token distance, token frequency). Tag vocabulary size correlates with hybrid favor: open-class tags (large `|V_τ|`) occupy the more hybrid-favored region (slope +0.0019 ± 0.0010, raw).

**Pattern (ii): opening vs closing delimiter asymmetry.** Across all seven domains (prose, Python, HTML, LaTeX), opening brackets are consistently more hybrid-favored than the corresponding closing brackets. Openers initiate a new region/scope (a state update); closers satisfy an already-established structural obligation (a closure determined by the visible opener). Same surface form, different predictive role.

**Pattern (iii): hybrid advantage nearly vanishes on repeated n-grams.** Raw gap shrinks rapidly with n-gram length and approaches zero for long repeated spans. The repeated continuation in the visible prefix provides a strong prediction, so the hybrid no longer has a measurable absolute advantage.

### 1.3 Controlled synthetic probes (§4.2)

Three probe families, varying antecedent distance `d ∈ {32, 64, 128, 256, 512, 1024}`:

- **Pronoun memory**: introduce two people with roles, later refer to a role and force a pronoun choice. **Favors hybrid.**
- **Entity tracking**: bind entities to attributes, later query the attribute. **Favors hybrid** (hybrid stays above chance; transformer dips below chance at intermediate distances).
- **Structural closure**: open a bracket/tag, insert filler, score the closing token. **Favors transformer** at every distance.

**Key insight:** delay alone is not the relevant variable. All three probes require information from earlier context, but only the *state-readout* probes favor the hybrid. The closure probe involves delayed information too, but the answer is determined by a visible opener.

### 1.4 Theoretical interpretation (§5) — Proposition 1

For a target class `τ` with vocabulary `V_τ`, and any richer feature map `ϕ` (the prefix features an architecture makes easy to express):

```
DKL(p⋆_τ ‖ p_ϕ,τ) ≤ log|V_τ|
```

**Proof sketch:** `p_ϕ,τ` can always ignore `ϕ(x_<t)` and emulate the best class-only predictor `p_class,τ`. So `DKL(p⋆_τ ‖ p_ϕ,τ) ≤ DKL(p⋆_τ ‖ p_class,τ) ≤ DKL(p⋆_τ ‖ U_τ) = log|V_τ| − H(p⋆_τ) ≤ log|V_τ|`.

**Three consequences:**
1. **Visible-prefix sufficiency** (copy/closure): when `ϕ` deterministically predicts the class-conditional next token, `p_ϕ,τ = p⋆_τ` and `DKL = 0`. Both architectures converge.
2. **Local class size bound**: for small closed classes, knowing the local class leaves limited room for any richer feature to help — even when that feature is useful. For open classes, the bound is loose.
3. **Discourse state hypothesis**: when recall, closure, and local class are not enough, the missing feature is *semantic/discourse state* `δ_j = Update(δ_{j-1}, sent_j)` — a recurrent state updated per sentence/unit. The largest hybrid gains appear on open-class, non-copy, state-conditioned targets whose fillers depend on accumulated semantic/program/document context.

### 1.5 Filtered evaluations as architecture diagnostics (§6)

Three filters computed from the same per-token NLL:
- `ALL_TOKENS` — standard aggregate.
- `TOP-10∩NO-COPY` — ten most hybrid-favored open-class POS families, excluding positions completing a repeated n-gram (n ≤ 4). Targets state-conditioned readout; removes visible-prefix retrieval.
- `COPY-5-ONLY` — positions completing a repeated 5-gram. Isolates retrieval.

On 1B development runs (Transformer / Hybrid / Pure RNN), aggregate loss compresses distinct regimes into one number. Filtered losses surface them: under `TOP-10∩NO-COPY`, the Transformer–Hybrid separation roughly doubles; under `COPY-5-ONLY`, the Pure RNN is consistently 0.10–0.20 nats worse than attention-based models (a weakness aggregate loss hides).

### 1.6 What the paper is NOT

- Not a new architecture. Not a new training method (→ not a riir-train redirect).
- Not a mechanism paper. It is an **empirical diagnostic + theoretical interpretation** paper.
- The authors explicitly frame §5 as a *description* of where the gap concentrates, "not a mechanistic claim" (§E, limitation 3). They do not localize the advantage to specific layers or heads.

---

## 2. Distillation

### 2.1 The transferable primitive: `PairedLossGap` + filtered evaluations

The distilled inference-time primitive is a **measurement tool** (not an inference mechanism): given two log-probability sequences over the same prefixes, compute per-token `Δ_i`, stratify by token class, and report filtered aggregates.

| Component | What it does | Cost | When to use |
|---|---|---|---|
| `PairedLossGap` | Compute `Δ_i = ℓ_A − ℓ_B` per token from two log-prob traces | O(L) subtract | Any A/B comparison of two inference paths |
| `TokenTagStratifier` | Assign each token a class τ (content/function/other, bracket open/close, copy/n-gram) | O(L) per tagger | When you want class-resolved diagnostics, not just aggregate |
| `FilteredEval` | Aggregate `Δ_i` over a filter mask (ALL, TOP-K∩NO-COPY, COPY-N-ONLY) | O(L) sum | GOAT-gate amplification; architecture search |
| `ClassSizeBound` | Compute `log|V_τ|` for each class | O(1) per class (precompute vocab sizes) | Theoretical bound on how much any richer feature can help |

**Properties:**
- Pure forward-pass analysis — no training, no backprop, no gradient descent. Modelless.
- Generic: works on any pair of log-prob traces (two adapters, two snapshots, HLA-on vs HLA-off, two router configs). No game/chain/shard semantics.
- Zero-allocation hot path: the per-token subtract is one f32 op; stratification is a precomputed lookup.
- The paper's Proposition 1 bound is a *theoretical* tool — it tells you when a diagnostic gap is structurally bounded (small `V_τ`) vs when it has room to grow (large `V_τ`).

### 2.2 Proposition 1 as a raw-vs-latent justification (the theoretical validation)

**Proposition 1 is an information-theoretic justification of our raw-vs-latent sync boundary** (AGENTS.md):

| Domain | Example | `V_τ` | `log|V_τ|` | Where state lives |
|---|---|---|---|---|
| **Physical** (raw, synced) | position `{x, y}`, HP, wallet balance | small (e.g., 2^16 grid cells, u32 HP) | ≤ 16–32 bits | Raw, LatCal-committed — the bound proves latent encoding adds nothing |
| **Semantic** (latent, local) | emotion, mood, curiosity, style | large (open-class) | unbounded in principle | Latent HLA state — this is where recurrence earns its keep |
| **Social** (KG triples) | "entity met entity", "entity fears entity" | medium (typed relations) | moderate | Latent similarity → KG triple emission |

The bound says: for physical state, `DKL(p⋆_τ ‖ p_ϕ,τ) ≤ log|V_τ| ≈ 0`, so no feature map (including a learned latent representation) can beat the class-only predictor by more than the log-vocabulary-size. **Raw commitment is information-theoretically sufficient for small `V_τ`.** For semantic state, `log|V_τ|` is large, so the discourse state `δ_j` (the HLA vector) is the only way to capture the gain.

**This is a moat-relevant theoretical validation** — but it validates an existing design, not a new capability. The capability (raw physical sync + latent semantic HLA) already ships. Proposition 1 is the proof that the split is *optimal*, not just convenient.

### 2.3 Open-bracket vs close-bracket asymmetry = two-brain model

The paper's finding (openers = state-update → hybrid-favored; closers = state-closure → transformer-favored) maps exactly to our **two-brain model** (AGENTS.md):

| Paper concept | Our codebase concept |
|---|---|
| Opener (state update) | Think brain: recurrent HLA `evolve_hla` updates belief state |
| Closer (state closure, visible opener) | Info brain: raw retrieval from synced ground-truth state |
| Copy/n-gram (visible-prefix reuse) | Info brain: deterministic replay from raw sync log |
| State-conditioned readout | Latent functor application: project state → prediction |

The paper provides *empirical evidence* that the two-brain split is the right split: state updates belong to recurrence (think brain), closures/copies belong to retrieval (info brain).

### 2.4 What's NOT here (stays in riir-train / not needed)

- The Olmo 3 / Olmo Hybrid checkpoints are trained models — we don't train them. We use the diagnostic on our own inference paths.
- The regression controls (§4 Analysis II) are a research-grade statistical tool; the modelless primitive ships the raw tag-stratified means + the filtered aggregates, not the full OLS. (The regression is reproducibility context for the paper's claims, not a runtime primitive.)
- The "discourse state tracking benchmark" call (§5.1) is a research agenda item, not a primitive.

### 2.5 Relationship to existing katgpt-rs primitives

| Existing primitive | Relationship |
|---|---|
| **Research 242 / Plan 276** (Topological State Tracking Recurrent Belief) | **Theoretical predecessor.** R242 (Mozer 2026) diagnosed the topological gap; this paper (Li & Merrill 2026) empirically validates it on Olmo 3 vs Olmo Hybrid and adds the diagnostic primitive + Proposition 1. R242 already established HLA `evolve_hla` = discourse-state Update operator. This paper's contribution is the *measurement framework* (paired gap + filtered evals) + the *theoretical bound* (Prop 1). |
| **`evolve_hla`** (`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs`) | IS the discourse-state `Update(δ_{j-1}, sent_j)` operator. This paper validates it is the right operator for state-conditioned readout. No change needed — the validation is theoretical. |
| **`latent_functor/reestimation.rs`** (riir-engine) | Coherence-driven re-estimation IS the "state-conditioned readout triggers re-derivation" pattern. The paper's finding (state-conditioned tokens are hybrid-favored) validates this is where runtime self-learn effort should concentrate. |
| **SalienceTriGate** (Plan 303) | Three-way per-tick emit gate (Speak/Silent/Delegate). Fusion candidate: route curiosity budget by token class — high on open-class state-conditioned tokens, low on copy/closure tokens (see §4 Fusion). |
| **DEC Stokes operators** (`dec/operators.rs`) | Proposition 1 is a volume-of-support bound. For a cochain on a cell complex, `V_τ` = the set of values the cochain can take on a cell. Connects to boundary-vs-volume perf (Plan 314): boundary-only mass computation wins when the boundary is smaller than the interior — the *same* curse-of-dimensionality intuition as `log|V_τ|` bound. |

---

## 3. Verdict

**Paper-alone: GOAT.** An empirical diagnostic + theoretical interpretation paper — no novel mechanism of its own, but a high-leverage measurement framework (paired token-level gap + filtered evals) and a clean theoretical bound (Proposition 1). Its value is *organizational and diagnostic*: it tells us *where* the hybrid state-tracking gain comes from and gives us a tool to measure it.

**Fusion: GOAT (not Super-GOAT).** Honest novelty-gate scoring:

| Gate | Question | Honest answer |
|---|---|---|
| **Q1 Novelty** | Any existing code cover this? | **PARTIAL.** The *theoretical claim* (recurrence helps state tracking; attention helps copy/closure) is already in R242. The *diagnostic primitive* (paired token-level loss gap + filtered evals) is novel to our codebase (grep for `paired_loss\|nll_gap\|filtered_eval\|class_size` returns zero semantic hits). Proposition 1 is novel as a bound. |
| **Q2 New capability class** | New behavior, not just better numbers? | **FAILS.** The diagnostic primitive is a *measurement* tool, not an inference mechanism. It makes our GOAT gates sharper; it doesn't enable a new class of inference. The theoretical validation (Prop 1 + two-brain mapping) validates existing design. |
| **Q3 Selling point** | "Our NPCs/systems do X that no competitor can"? | **WEAKENS.** "Our raw-vs-latent split is provably optimal (Prop 1)" is a nice moat footnote but validates existing design, not a new capability. The sharpened claim is incremental. |
| **Q4 Force multiplier (≥2)** | Connects to ≥2 existing pillars? | Passes (6 systems: HLA, latent_functor, cgsp_runtime, LatCal, DEC, NeuronShard consolidation). But Q4 alone ≠ Super-GOAT — needs Q1+Q2+Q3 too. |

**Why not Super-GOAT:** this is the empirical follow-up to R242's theoretical paper. R242 already downgraded from Super-GOAT → GOAT after the `evolve_hla` prior-art check. This paper adds the measurement tool and the bound, both of which are GOAT-tier (provable gain in diagnostic resolution + theoretical validation of existing design), not Super-GOAT-tier (new capability class). The paper itself explicitly does not claim a new mechanism — it is a diagnostic/analysis paper.

**Outputs:**
1. **Open primitive** — this doc + `katgpt-rs/.plans/335_*.md` (the paired loss gap diagnostic primitive).
2. **No private guide** — not Super-GOAT. The theoretical validation of raw-vs-latent split + HLA-as-discourse-state lands as a cross-ref in this note (§2.2, §2.3), not a separate riir-ai guide.
3. **Plan** — `katgpt-rs/.plans/335_*.md`.

---

## 4. Fusion (the GOAT combination)

**The combination:** Li & Merrill 2026 (paired token-level gap + Proposition 1 + filtered evals) × **R242 / `evolve_hla`** (discourse-state Update operator, already shipped) × **raw-vs-latent sync boundary** (AGENTS.md) × **SalienceTriGate** (Plan 303, per-tick emit gate) × **DEC boundary-vs-volume** (Plan 314).

**What this combination produces that none alone can:**

| Component alone | What it can't do | What the fusion adds |
|---|---|---|
| Li & Merrill 2026 | Measures the gap on Olmo; doesn't ship a reusable primitive | Gives us the `PairedLossGap` + `FilteredEval` tool to measure gaps on *our* A/B comparisons (HLA-on/off, adapter hot-swap, router configs) |
| R242 / `evolve_hla` | Ships the Update operator; no measurement of whether it's earning its keep | The paired gap diagnostic lets us *prove* HLA's state-tracking advantage on our own NPC runtime (validate the R242 GOAT gate retroactively) |
| Raw-vs-latent sync boundary (AGENTS.md) | Asserted by rule, not proven | Proposition 1 *proves* raw commitment is information-theoretically sufficient for small `V_τ` (physical domain); latent encoding earns its keep only for large `V_τ` (semantic domain) |
| SalienceTriGate (Plan 303) | Three-way emit gate, curiosity budget uniform across token classes | **Curiosity budget routed by token class** — high on open-class state-conditioned tokens (where recurrence helps), low on copy/closure tokens (where the answer is determined by visible structure). New routing signal. |
| DEC boundary-vs-volume (Plan 314) | Boundary flux `O(n^{(d-1)/d})` vs interior `O(n)` perf heuristic | Proposition 1 is the same curse-of-dimensionality intuition as a volume-of-support bound: for cochains, `V_τ` = values a cochain can take on a cell; `log|V_τ|` bounds the reducible loss |

**Capability increment (over existing corpus):**
- (a) A token-level paired diagnostic primitive we don't have today — makes GOAT gates sharper.
- (b) A theoretical bound (Prop 1) that justifies raw-vs-latent split — validates existing design.
- (c) A new curiosity routing signal (by token class) — incremental improvement to cgsp_runtime.

All three are GOAT-tier (provable gain), not Super-GOAT-tier (new capability class).

**Closest cousins across all repos (for the fusion protocol):**
- `katgpt-rs/.research/242_Topological_State_Tracking_Recurrent_Belief.md` — the theoretical predecessor. This paper is its empirical validation.
- `katgpt-rs/.research/192_NextLat_Belief_State_Latent_Dynamics.md` — belief-state residual MLP as token drafter; the discourse-state hypothesis generalizes it.
- `katgpt-rs/.research/036_Luce_Megakernel_Hybrid_DeltaNet_Attention.md` — hybrid DeltaNet/attention GPU kernel (the architecture this paper analyzes).
- `katgpt-rs/.research/070_Gated_DeltaNet_2_Decoupled_Erase_Write_Linear_Attention.md` — GDN (the recurrent layer in Olmo Hybrid).
- `riir-ai/.research/127_Implicit_Microcognition_Crowd_NPC_Guide.md` — design context for HLA-as-discourse-state (private).
- `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` — coherence-driven re-estimation = state-conditioned readout triggers re-derivation.

---

## 5. Open Questions / Risks

- **R1 — Regression controls scope.** The paper ships a full OLS regression (Eq 1) with controls for difficulty, frequency, position, subword status, local reuse, previous-token distance, token frequency. The modelless primitive ships the raw tag-stratified means + filtered aggregates, not the full regression. If we ever need the controlled view on our own data, the regression is reproducibility context, not a runtime primitive. Risk: low — raw means + filtered evals are the high-signal subset.
- **R2 — Token tagger availability.** The diagnostic needs a tagger (POS for prose, source-level categories for code/markup). For our NPC runtime, the analog is "token class" in the game-dialog sense (open-class content vs closed-class function vs copy/n-gram). We'd need a lightweight tagger — or reuse the existing game-state-derived class labels. Risk: medium — tagger quality determines diagnostic resolution.
- **R3 — Two forward passes required.** The paired gap needs two log-prob traces over the same prefixes. For HLA-on/off comparisons this is 2× the forward cost. Mitigation: run on a held-out eval set, not the hot path. The diagnostic is a warm/cold-tier tool, not a plasma-tier tool.
- **R4 — Proposition 1 is a bound, not an equality.** `DKL ≤ log|V_τ|` is a worst-case bound; the actual reducible loss can be much smaller. Don't overclaim that raw commitment is *optimal* — only that the *room for latent encoding to help* is bounded by `log|V_τ|`.

---

## 6. References

- Paper: [arXiv:2606.20936](https://arxiv.org/abs/2606.20936) — Li & Merrill, AI2, Jun 2026.
- Theoretical predecessor (our Research 242): [arXiv:2604.17121](https://arxiv.org/abs/2604.17121) — Mozer, Siddiqui, Liu, DeepMind, Jun 2026.
- Models analyzed: Olmo 3 7B ([arXiv:2512.13961](https://arxiv.org/abs/2512.13961)), Olmo Hybrid 7B ([arXiv:2604.03444](https://arxiv.org/abs/2604.03444) — Merrill et al. 2026).
- Cited in paper, in our corpus: NextLat (Teoh 2025b — our Research 192), Gated DeltaNet-2 (our Research 070), Luce Megakernel hybrid (our Research 036).
- Our related: 242 (topological state tracking — the predecessor), 192 (NextLat belief dynamics), 070 (Gated DeltaNet-2), 036 (Luce hybrid), 097 (training-free loop), 230 (SSD duality), 303 (SalienceTriGate — fusion candidate), 314 (DEC Stokes wrappers — Proposition 1 connection).
- riir-ai: 127 (Implicit Microcognition Crowd-NPC Guide — HLA-as-discourse-state design context).

---

## TL;DR

Li & Merrill (AI2, arXiv:2606.20936) empirically validate R242's theoretical state-tracking diagnosis by comparing Olmo 3 (transformer) vs Olmo Hybrid (attention+GDN) at the **per-token paired loss gap** level. Three patterns: hybrid wins on open-class content words (state-conditioned readout); openers are hybrid-favored, closers are transformer-favored (state-update vs state-closure); hybrid advantage vanishes on copy/n-grams (visible-prefix retrieval). **Proposition 1** (`DKL ≤ log|V_τ|`) formally justifies our raw-vs-latent sync boundary: physical state (small `V_τ`) has near-zero reducible loss → raw LatCal commitment is information-theoretically sufficient; semantic state (large `V_τ`) is where latent HLA earns its keep. **Discourse state hypothesis** (`δ_j = Update(δ_{j-1}, sent_j)`) is exactly `evolve_hla` — already shipped (R242 prior art). **Verdict: GOAT (paper alone) + GOAT (fusion).** Not Super-GOAT: this is the empirical follow-up to R242's theoretical paper; the diagnostic primitive (paired gap + filtered evals) is a measurement tool, not a new inference capability class; Proposition 1 validates existing design rather than creating a new capability. Outputs: open primitive note (this doc) + open plan (`katgpt-rs/.plans/335_*.md`). No private guide — not Super-GOAT.
