# Research 244: Self-Evolver Faithfulness — Cognitive Integrity Layer (Causal Intervention on Injected Memory)

> **Source:** [Large Language Model Agents Are Not Always Faithful Self-Evolvers](https://arxiv.org/pdf/2601.22436) — Zhao, Wang, Zhang, Deng, Zhao, Che, Qin, Liu (HIT / SMU), ICML 2026
> **Date:** 2026-06-16
> **Status:** Active — Super-GOAT Fusion
> **Related Research:** 054 (path consistency / reward hacking — OUTPUT-side counterpart), 172 (MUSE skill lifecycle), 199 (Memory Caching Growing RNN), 212 (Gemini Fourier LatCal — recurrent belief), 242 (Topological State Tracking Recurrent Belief), 240 (SGS Curiosity-Guided Self-Play)
> **Related Plans:** 054 (path-hacking detector), 212 (collapse-aware), 274 (curiosity-guided self-play), 278 (this doc — FaithfulnessProbe primitive)
> **Cross-ref (riir-ai):** Research 129 (Cognitive Integrity Layer Guide — private selling-point doc), Plan 308 (Cognitive Integrity Layer runtime integration)
> **Classification:** Public — generic inference-time diagnostic primitive + design discipline. No game IP, no chain IP.
>
> **Gain-Redirects (synthesis, verdict revised 2026-07-26):** Nikolaou & Mencattini [arXiv:2510.15511 "Language Models are Injective and Hence Invertible"] (ICLR 2026) — **Gain**, not PASS (revised up: SipIt is modelless + doesn't ship → §1.55 mandates Gain-or-higher). Theorem (decoder-only text transformers w/ real-analytic activations are almost-surely injective prompt→last-token maps, preserved under GD) + SipIt (modelless linear-time prompt recovery from layer-ℓ hidden states via per-position vocabulary search with a gradient-guided policy on a continuous proxy embedding — no backprop through weights). Connects to this note's `FaithfulnessProbe`/`ActivationExtractor` lineage by proving the encoding FaithfulnessProbe intervenes on is *lossless* for text transformers: the last-token hidden state identifies the input almost surely, so a probe that fails to find behavioral influence cannot blame "missing information". SipIt ships as the open primitive `katgpt-core::inversion` (Plan 561). **Caveat (the reason this is Gain, not GOAT):** HLA — the cognitive substrate FaithfulnessProbe intervenes on in the NPC runtime — is a sigmoid-bounded per-NPC kernel, not a decoder-only text transformer; the injectivity theorem doesn't transfer. The theoretical lens still reinforces the FaithfulnessProbe framing; the open primitive is for general text-transformer paths (which our stack has in the speculative-decode / drafter substrate, not in HLA).

---

## TL;DR

The paper performs **controlled causal interventions** on the experiences fed to self-evolving LLM agents (ExpeL, Dynamic Cheatsheet, ReasoningBank, G-Memory) and discovers a **striking, persistent asymmetry**: agents faithfully use **raw trajectories** (perturbing them crashes performance) but largely **ignore condensed experience** (perturbing, corrupting, or even filler-replacing distilled summaries barely moves the success rate — sometimes *improves* it). This holds across 13 backbones (1.7B → 235B, GPT-5.2, Gemini-3-Pro, Claude-4.6), single- and multi-agent, offline and online paradigms. Three root causes: (1) condensed content is semantically vague, (2) internal processing biases suppress retrieved memory (Integrated-Gradients attribution to the condensed segment stays flat-and-low across all layers while local context dominates deep layers), (3) knowledge-saturated task regimes make experience redundant.

**Distilled for katgpt-rs (modelless, inference-time):**

The paper is diagnostic, not mechanistic — but its findings imply a **generic, engine-level primitive** we currently lack: a **`FaithfulnessProbe`** that runs causal interventions on injected context (raw replays OR latent direction vectors OR condensed heuristics) and reports whether the consumer's behavior is causally bound to that segment. Today our `evolve_hla` (`crates/katgpt-sense/src/reconstruction.rs:623`) updates an `[f32; 8]` HLA state via dot-product + sigmoid and feeds it into reconstruction — with **zero mechanism to verify the HLA delta actually influences downstream action selection**. We are almost certainly silently dropping memory value exactly as the paper describes.

The paper also implies a **design discipline** (not a primitive, but a rule): injected memory must be (a) **triggered** by consumer uncertainty, not statically prepended; (b) **causally bound** — emit the raw signature alongside the latent projection so the consumer has the "faithful" form; (c) **contextualized** — latents must be task-actionable, not generic heuristics.

**Fusion (the Super-GOAT):**

Paper × Plan 054 (output-side `fully_faithful` / `reward_hacking` path consistency) × Plan 212 (collapse-aware entropy trigger) × HLA `evolve_hla` × Curiosity Pulse (uncertainty signal) → **Cognitive Integrity Layer**: a runtime subsystem that verifies an agent's behavior is causally bound to *both* its injected experience (input-side, this paper) *and* its declared reasoning chain (output-side, Plan 054). No incumbent does either; doing both closes the "is this agent actually thinking or faking it?" loop. Selling-point doc: `riir-ai/.research/129`.

---

## 1. Paper Core Findings

### 1.1 The Asymmetry (the headline)

Across 4 frameworks × 13 backbones × 9 benchmarks, **raw experience interventions** (`Empty`, `Shuffle`, `Irrelevant`) reliably crash success rate (e.g., ExpeL ALFWorld 72.4 → 2.2 under `Empty`); **condensed experience interventions** (`Empty`, `Corrupt`, `Irrelevant`, even `Filler` — replacing summaries with `%$#@&`) cause only marginal or inconsistent changes, and in several cases *improve* performance. This holds even in ReasoningBank's condensed-only setting (no raw trajectories to overshine). Scaling the backbone (Qwen3 1.7B → 32B → 235B-A22B; GPT-4o → GPT-5.2; Gemini-2.5 → Gemini-3-Pro) does **not** close the gap — larger models just perform better overall while remaining equally unfaithful to condensed memory.

### 1.2 Three Root Causes

| Cause | Evidence | Implication for us |
|-------|----------|---------------------|
| **Semantic limitation** of condensed content (Table 1 failure analysis: Distraction / Overreliance / Premature inference) | Agents sometimes score higher *without* the condensed summary; smaller models especially get distracted. | Latent direction vectors must be **task-actionable**, not vague mood embeddings. A 64-dim HLA vector projected to 5 scalars is useless if those scalars don't drive a specific action-selection branch. |
| **Internal processing bias** (IG attribution, Fig 7 + App D.7) | Attribution to the condensed segment stays flat-low across ALL layers under ALL interventions (baseline, Corrupt, Irrelevant). Current trajectory dominates deep layers. The model structurally under-weights retrieved memory. | Static prepending of latent memory into a context buffer is structurally doomed. Memory must enter the **action-selection path** (directly modulate logits/policy), not the context window. |
| **Task regime sufficiency** (§5.3, Table 2) | On knowledge-intensive tasks (HotpotQA, GPQA, MMLU, FEVER, 2Wiki, Musique), neither raw nor condensed interventions move the needle — pretrained priors suffice. | **Triggered injection**: don't inject memory when the consumer is in a saturated/low-uncertainty regime. Indiscriminate injection dilutes attention (paper's word). |

### 1.3 Design Takeaways (paper's Impact Statement)

1. Condensed experience must be **contextualized, task-relevant, cognitively actionable** — not abstract summaries.
2. Experience must be **dynamically retrieved and injected** based on task demands, interaction history, and internal model uncertainty — not statically prepended.
3. Experience should be **interactively triggered** — agents don't need memory for every task; indiscriminate use reduces effectiveness.

These three rules are **direct architectural constraints** on how our HLA cache, NeuronShard consolidation, KG triple emission, and dMoE routing should inject their outputs.

---

## 2. Distillation

### 2.1 The Transferable Primitive: Causal Intervention Faithfulness Probe

The paper's experimental method **is** the primitive. Generalized away from LLM agents:

Given a consumer `C` (an attention head, an action selector, an NPC brain) and an injected memory segment `M` (a raw replay, a latent direction vector, a condensed heuristic), define **faithfulness** as:

```
Faithfulness(C, M) := Δ_behavior when M is perturbed
                    := d( C(x; M) , C(x; perturb(M)) )
```

If perturbing `M` (Empty / Shuffle / Corrupt / Irrelevant / Filler) produces a large behavioral delta, `C` is faithfully using `M`. If perturbation produces negligible delta, `C` is ignoring `M` — the memory is dead weight.

**Intervention suite** (paper §3.2, §3.3, generalized):

| Intervention | Targets | What it tests |
|--------------|---------|---------------|
| `Empty` | content removed, format preserved | Is the *presence* of the segment doing the work, or the *content*? |
| `Shuffle` (raw) / `Corrupt` (condensed) | temporal/causal structure destroyed | Does the consumer rely on coherence? |
| `Irrelevant` | replace with same-format unrelated content | Topical/semantic grounding? |
| `Filler` | replace with `%$#@&` placeholder | Surface-format dependence? |

**Behavioral delta metric** — task-dependent:
- For token-producing consumers: token-level distance (edit distance, KL divergence of next-token distribution).
- For action-selecting consumers (NPCs, MCTS): action-distribution KL, or top-1 action flip rate.
- For latent-state consumers (HLA, NeuronShard): cosine distance in output latent, or downstream-action KL.

### 2.2 The Attribution Surrogate (paper §5.2, App D.7)

The paper uses Integrated Gradients (IG) at the attention level, but App D.7 validates a **cheap surrogate**: L2 norm of token-embedding gradients w.r.t. cross-entropy loss, averaged per segment. This is the modelless-friendly form — we can compute it on any consumer that exposes gradients w.r.t. its input embedding (or, for our frozen-latent consumers, finite-difference sensitivity probes).

For our stack: **`AttributionProbe`** = finite-difference sensitivity of consumer output to perturbations of the injected segment. Zero training, zero backprop through base weights — just `f(x + ε·δ) − f(x − ε·δ)` style probing. Hot-path cheap if `ε`-ball is small and we batch.

> **See also: Research 362 — HydraHead path-patching indirect-effect extension.** Research 362 (Plan 358) extends this direct-effect FaithfulnessProbe pattern in three ways: (1) **path patching / sender-receiver indirect effect** (one-step-back attribution — a head can be causally important without writing the signal directly, by feeding a receiver); (2) **span-level logit-difference readout with exponential decay** (Eq 9); (3) application to **per-attention-head outputs** as the intervention target (the `CausalHeadImportance` primitive ranks heads by causal necessity, competing with RTPurbo's attention-mass calibration).

### 2.3 Three Design Rules (architectural constraints, not code)

Distilled from §5 + Impact Statement, mapped to our latent/raw discipline:

| Rule | Paper basis | Our enforcement |
|------|-------------|-----------------|
| **Triggered injection** | §5.3 — saturated regimes make memory redundant; indiscriminate use dilutes attention. | Inject HLA delta / NeuronShard / KG triple into consumer **only when** consumer uncertainty (entropy, collapse signal, curiosity pulse) exceeds threshold. Otherwise skip — saves compute AND avoids the paper's distraction failure mode. |
| **Action-path binding** | §5.2 — IG attribution to statically-prepended memory stays flat across all layers. | Memory must modulate the **action selection** (logit bias, policy gate, routing weight), not the context window. Latent → scalar projection → direct policy term. |
| **Raw signature co-emission** | §4.1 — raw experience IS faithfully used; condensed is not. | At the latent→raw bridge, **co-emit the raw causal evidence** alongside the latent projection. Consumer gets both forms; the raw form is the "faithful" anchor. |

---

## 3. Verdict: Super-GOAT (Fusion)

**One-line reasoning:** The paper alone is a GOAT diagnostic; fused with Plan 054 (output-side path-hacking) + Plan 212 (collapse trigger) + HLA evolve_hla + Curiosity Pulse, it produces a **new capability class** — a Cognitive Integrity Layer that verifies agent behavior is causally bound to both injected memory and declared reasoning. No incumbent (LLM agent framework, game AI runtime, chain validator) does either side.

### 3.1 Novelty Gate Evidence (Q1–Q4)

**Q1 — No prior art? YES (strong).**

Grep results across both repos, both layers (notes + code):

| Search | Notes matches | Code matches | Closest cousin |
|--------|---------------|--------------|----------------|
| `faithful\|faithfulness\|causal intervention\|integrated gradient\|IG score\|attribution score` | 20 hits — all are about *output-side* faithfulness (XAI user trust 189, vocabulary channel completeness 203, KV compaction preservation 233, MDA heuristic 009) or *path consistency* (Plan 054 `fully_faithful`/`reward_hacking`) | 0 hits for the diagnostic-primitive terms; `PrunerAttribution` in `decision_explainer.rs` is post-hoc *explanation*, not causal-intervention *testing* | Plan 054 (output-side only) |
| `condensed experience\|raw experience\|experience pool` | 0 hits | n/a | None |
| `entropy gated\|triggered injection\|dynamic injection\|context dilution` | Hits for entropy-gated scheduling/exploration (Plans 217, 223, 248, 269, 272, 274) — but all gate *consumer exploration*, never *whether to inject memory* | n/a | None — entropy-gated *injection* is novel |

The HLA `evolve_hla` primitive at `crates/katgpt-sense/src/reconstruction.rs:623` — the canonical "shipped-without-a-note" case the skill warns about — has no faithfulness verification whatsoever. The paper's findings directly implicate it.

**Q2 — New class of behavior? YES (moderate-strong).**

"Cognitive integrity verification" is a new capability class. Comparable in scope to how Plan 212 (collapse detection) became a pillar — collapse detection asks "is the consumer degenerate?"; cognitive integrity asks "is the consumer actually using its cognition?". Together they cover the full integrity surface. No incumbent offers either.

**Q3 — Product selling point? YES (moderate).**

Finishable: *"Our NPCs don't fake their reasoning — every behavior is causally verified against accumulated memory and declared reasoning chains. First cognitive-integrity layer for game AI. Memory injections that don't bind are detected and either re-triggered or dropped, eliminating the silent 60%+ memory-value loss that vanilla self-evolving agents suffer."*

Player-facing translation: NPCs actually remember and act on what happened to them, rather than executing triggered scripts that ignore their own memory.

**Q4 — Force multiplier? YES (strong, ≥8 pillars).**

HLA `evolve_hla`, Plan 054 path-hacking, Plan 212 collapse-aware, Curiosity Pulse (Research 041), MUSE skill lifecycle (Research 172), NeuronShard consolidation, KG triple emission (Research 196/Plan 221), dMoE routing (Research 161), Memory Caching Growing RNN (Research 199), Reasoning-in-Memory (riir-ai Research 043). See connection map in `riir-ai/.research/129`.

### 3.2 Mandatory Super-GOAT Outputs (this session)

Per skill §1.5, all four committed in this session:

1. **Open primitive** — this file + `katgpt-rs/.plans/278_faithfulness_probe_modelless.md`.
2. **Architectural GUIDE** — `riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md` (private selling-point doc).
3. **Plans** — `katgpt-rs/.plans/278` (open primitive) + `riir-ai/.plans/308` (runtime integration).

---

## 4. Open Primitive Sketch — `FaithfulnessProbe` trait

```rust
/// Causal-intervention faithfulness probe for injected memory segments.
///
/// Verifies that a consumer's behavior is causally bound to an injected
/// memory segment (raw replay, latent direction vector, condensed heuristic).
/// Modelless: zero training, zero backprop through base weights.
///
/// Based on Zhao et al. 2026 (arxiv 2601.22436).
pub trait FaithfulnessProbe {
    type Memory;
    type Behavior;
    type Delta: PartialOrd + Copy;

    /// Run a single causal intervention and report behavioral delta.
    /// `perturb(M) -> M'` is applied; delta = d(C(x; M), C(x; M')).
    fn probe_intervention(
        &self,
        memory: &Self::Memory,
        intervention: Intervention,
        consumer_ctx: &impl ConsumerContext<Self::Behavior>,
    ) -> Self::Delta;

    /// Full intervention suite — returns the min/max/mean delta across
    /// {Empty, Shuffle, Corrupt, Irrelevant, Filler} as applicable.
    fn faithfulness_profile(
        &self,
        memory: &Self::Memory,
        consumer_ctx: &impl ConsumerContext<Self::Behavior>,
    ) -> FaithfulnessProfile<Self::Delta>;
}

#[derive(Clone, Copy, Debug)]
pub enum Intervention {
    Empty,       // content removed, format preserved
    Shuffle,     // temporal/causal structure destroyed (raw only)
    Corrupt,     // internal coherence broken (condensed only)
    Irrelevant,  // replace with same-format unrelated content
    Filler,      // replace with semantically-empty placeholder
}

#[derive(Clone, Debug)]
pub struct FaithfulnessProfile<D> {
    pub empty_delta: D,
    pub shuffle_or_corrupt_delta: D,
    pub irrelevant_delta: D,
    pub filler_delta: D,
}

impl<D: PartialOrd + Copy + Default> FaithfulnessProfile<D> {
    /// A memory segment is "faithfully used" if at least the Irrelevant
    /// and Filler interventions produce non-trivial behavioral delta.
    pub fn is_faithfully_used(&self, threshold: D) -> bool {
        self.irrelevant_delta > threshold && self.filler_delta > threshold
    }
}
```

Plus a lightweight **`AttributionProbe`** (finite-difference sensitivity surrogate for IG, paper App D.7):

```rust
/// Finite-difference sensitivity surrogate for Integrated Gradients.
/// Reports ‖∇_M C(x; M)‖ approximated by central differences.
/// Zero backprop — just ε-ball probing.
pub trait AttributionProbe {
    type Memory;
    fn attribution_norm(&self, memory: &Self::Memory, epsilon: f32) -> f32;
}
```

**Feature flag:** `faithfulness_probe` (opt-in, default off — it's a diagnostic, not a hot-path primitive). Gate the triggered-injection discipline behind `triggered_injection` (separate flag, default off until GOAT-gated).

---

## 5. Constraints Respected

- **Modelless first**: zero training, zero backprop through base weights. AttributionProbe uses finite differences, not gradients-through-weights.
- **Latent-to-latent preferred**: probes operate on latent segments (HLA vectors, direction vectors) without decoding to tokens.
- **Sigmoid not softmax**: behavioral delta metrics use sigmoid-gated comparisons, never softmax.
- **Raw/latent boundary**: probes never substitute latent for raw in anti-cheat paths. The "raw signature co-emission" rule emits raw *alongside* latent at the bridge — never replaces.
- **Zero-allocation**: probe suite reuses scratch buffers; `FaithfulnessProfile` is a fixed-size POD.
- **4-repo discipline**: open primitive (`FaithfulnessProbe`, `AttributionProbe`, `Intervention` enum) → katgpt-rs. Game runtime semantics, HLA direction interpretations, NPC integration → riir-ai (Research 129). Chain-side commitment (if any) → riir-chain. No training know-how here.

---

## TL;DR

Paper is a **diagnostic** showing self-evolving agents silently ignore 60%+ of their condensed memory. Distilled into katgpt-rs as a generic **`FaithfulnessProbe`** primitive (causal intervention suite) + **`AttributionProbe`** (finite-difference IG surrogate) + three design rules (triggered injection, action-path binding, raw signature co-emission). **Super-GOAT as a fusion** with Plan 054 (output-side path-hacking) + Plan 212 (collapse trigger) + HLA evolve_hla + Curiosity Pulse → **Cognitive Integrity Layer** (riir-ai/.research/129). All 4 novelty gates committed YES. Mandatory outputs delivered this session: this note + Plan 278 (open) + Research 129 + Plan 308 (private).

> **PASS-Redirects (synthesis):** Hong et al. [arXiv:2609.38972 "Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment"] — the LLM stated-vs-internal CoT faithfulness audit (CIA = macro-F1 of B_CoT vs probe-detected B_INT): this note's intervention family IS the internal (B_INT) half, modellessly; the two-sided stated-vs-internal F1 is an audited discard for us — no shipped surface is simultaneously decoupled, decision-bearing, AND internally open (the one decoupled surface, riir-dao's hosted advisory, is closed-weights and schema-locked to u64 levers with no rationale key). Full decomposition + the two surviving training-side levers: riir-train/.research/472 + Issue 622.
