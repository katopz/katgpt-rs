# Research 295: AC-GPT — Arbitrary Conditionals via Position-Aware Conditioning Prefix (Modelless Distillation)

> **Source:** Yinhan Lu, Eric Elmoznino, Léo Gagnon, Sarthak Mittal, Tejas Kasetty, Guillaume Lajoie. *Simplifying the Modeling of Arbitrary Conditionals in Natural Language* (AC-GPT). [arXiv:2606.14943](https://arxiv.org/abs/2606.14943). Mila / McGill / Université de Montréal. 12 Jun 2026.
> **Date:** 2026-06-23
> **Status:** Done — **GOAT** (plan-only, feature-flagged, no Super-GOAT promotion).
> **Related Research:** 269 (Variable-Width `> <former` — same week, same downgrade pattern: latent reframing available but prior art in `BlockCausal` + `LoraPair`), 278 (Engram Conditional Memory — closest "conditional retrieval" cousin), 290 / 153 (Latent Field Steering — closest "top-down latent injection" cousin), 067 (Domino Decoupled Causal Spec Decoding), 192 (NextLat Belief-State Drafter), 243 (Temporal Derivative Kernel — surprise signal cousin), 091 (SpecHop — continuous multi-hop speculation cousin).
> **Related Plans:** 295 (this note → plan), 025 (Bidirectional Prefill + LoraPair — closest shipped cousin), 066 (D2F BlockCausal attention — closest shipped attention-mask cousin), 238 (MUX-Latent prefix compression — closest shipped position-aware-prefix cousin), 299 (Engram — closest shipped conditional-memory cousin), 309 (Latent Field Steering — closest shipped latent-injection cousin).
> **Classification:** Public (katgpt-rs engine note). The fine-tuning recipe itself → **riir-train**.

---

## TL;DR

AC-GPT augments a standard causal Transformer with the ability to evaluate and sample from arbitrary conditionals `p(xe | xc)` — including conditioning on **future** tokens — in a **single forward pass**. The mechanism is minimal: copy the conditioning tokens `xc` to the front of the sequence with their **original position encodings**, allow **bidirectional self-attention among the copies** (no causal mask inside the prefix), and apply causal attention everywhere else. The paper's load-bearing insight is that you cannot just let later evaluation tokens attend to the originals `xc` in place — that leaks information from `x_{t+1}` to `x_t` through `xc` over multiple layers; the **front-of-sequence copies with bidirectional self-attention** are what prevents the leakage. Authors show this recipe lets you **fine-tune existing LLMs (Qwen3-8B/14B/32B, LLaMA-3.1-8B) with LoRA** to gain the capability — +13–17% perplexity on training-distribution conditional queries, while paying −1.3% to −3.9% on standard left-to-right.

**Distilled for katgpt-rs (modelless, inference-time):** the training recipe redirects to riir-train (LoRA fine-tuning of pretrained LLMs). What stays here is the **mechanism**: a position-aware conditioning-prefix primitive that (a) reuses `AttentionMode::BlockCausal` (already shipped in Plan 066), (b) reuses the `LoraPair { reader, writer }` switch (already shipped in Plan 025), (c) reuses position-aware prefix entries (already shipped as `MixedPrefillSequence::Raw { token_id, original_pos }` in Plan 238 MUX-Latent), and (d) adds the **single novel primitive**: a zero-allocation attention-mask builder that emits the AC-GPT mask shape `[xc-bidirectional | causal-everywhere-else]` over an augmented sequence, with `original_pos` propagated to the copy. This is a GOAT-tier inference primitive (provable single-pass vs iterative-MLM speedup at iso-quality), not a Super-GOAT — the architectural pieces ship, the latent-space reframing overlaps with Latent Field Steering + Engram conditional memory, and the only genuinely new bit is the mask builder + leakage-prevention discipline.

**Verdict:** GOAT. Plan-only in katgpt-rs behind a `ac_prefix` feature flag. No private guide (riir-ai / riir-chain / riir-neuron-db) — this is not Super-GOAT, the selling point is a speedup not a new capability class.

---

## 1. Paper Core Findings (verified by reading)

| Finding | Mechanism | Relevance here |
|---|---|---|
| Standard causal Transformers can't tractably evaluate arbitrary conditionals `p(xe | xc)` | `p(xe | xc)` for non-causal `xc` is an intractable integral; no single-pass factorization exists for vanilla GPT | Motivates the work; our stack has the same gap |
| AC-GPT solves this with a position-aware conditioning-prefix copy | Copy `xc` to the front of the augmented sequence; copies carry their **original position encoding** (RoPE rotation at original position); use **bidirectional self-attention** within `xc` copies, causal everywhere else | The mechanism — minimal architectural change |
| The copy is necessary; you cannot let later tokens attend to the originals in place | Worked example: `xe = {x1, x2, x4}, xc = {x3}`. Without copy, `x2 → x3 → x1` over two layers leaks future info from `x2` to `x1`. With copy at front, `x1` attends only to `x3_copy`, never to `x2` via `x3` | **The load-bearing insight** — the leakage argument is the genuine novelty |
| Single-pass conditional evaluation | Loss is computed only on tokens in `xe`; conditioning tokens `xc` are masked out of the loss | Efficient training; efficient inference |
| Fine-tunes pretrained LLMs at billion-parameter scale | Qwen3-8B/14B/32B + LLaMA-3.1-8B with LoRA r=8 or r=64, or full FT | **→ riir-train** (training recipe) |
| +13–17% perplexity gain on training-distribution conditional queries | Compared to matched-budget causal LoRA baseline; gain holds across model scales, families, corpora, adapter capacities | Training-side result → riir-train |
| Small left-to-right cost (−1.3% to −3.9% on Unconditional) | Fine-tuning for arbitrary conditioning costs a little standard-LM performance | Training-side tradeoff → riir-train |
| Conditioning range `rmax` controls capability concentration | `rmax=0.2` → +1.2% TD gain; `rmax=1.0` → +11.6%; `rmax=0.6` is the balance | Hyperparameter → riir-train |
| σ-GPT (any-order AR) fails catastrophically without curriculum; AC-GPT preserves L→R | The paper's central hypothesis: decoupling arbitrary conditioning from arbitrary ordering is the win | Validates that BlockCausal (which we already ship) is the right attention shape, not full bidirectional-everywhere |
| Block-based conditioning set sampler | Sample `|xc|`, number of contiguous blocks `B`, block sizes, gap locations — interpolates fully-arbitrary and structured | Conditioning-set construction; orthogonal to the mechanism |
| Modular deployment via mode-specific adapters | "Future work": one adapter per conditional mode (infilling vs training-dist), switch at inference | **Already ships as `LoraPair { reader, writer }` + hot-swap** — this is the inference-time cousin |

---

## 2. Distillation (modelless, inference-time)

### 2.1 What's already shipped (the prior-art surface — five granularities)

| AC-GPT feature | Shipped cousin | File / Plan | Granularity |
|---|---|---|---|
| Bidirectional attention within conditioning set, causal elsewhere | **`AttentionMode::BlockCausal`** | `crates/katgpt-types/src/enums.rs:74`, P066 (D2F) | Attention mask |
| Reader/writer LoRA split (bidirectional prefill vs causal decode) | **`LoraPair { reader, writer }`** | `crates/katgpt-types/src/lora.rs:392`, P025 (Bidirectional Prefill + Modality LoRA) | Adapter switch |
| Position-aware prefix entries (token + original_pos) | **`MixedPrefillSequence::Raw { token_id, original_pos }`** | `crates/katgpt-core/src/mux_latent/inject.rs:34`, P238 (MUX-Latent) | Prefix construction |
| Conditional retrieval / fuse into hidden state | **Engram** `fuse_into_hidden_state`, hash-addressed conditional pattern memory | `crates/katgpt-core/src/engram/`, R278 / P299 | Conditional memory |
| Top-down direction-vector injection (additive overlay) | **Latent Field Steering** `apply_latent_steering` | `crates/katgpt-core/src/latent_steering.rs`, R290 / P309 / riir-ai R153 | Latent injection |
| Target-conditioned draft seeding | **`speculative_step_conditioned` / `dflash_predict_conditioned`** | `crates/katgpt-speculative/src/dflash.rs:179`, P012 Phase 5 | Speculative conditioning |
| Conditional q(x|x_{<i}) drafting | **`dflash_predict_ar`** — "Produces conditional q(x|x_{<i}) distributions" | `riir-ai/crates/riir-engine/src/dflash.rs:248` | Conditional drafting |

**The gap:** no shipped primitive composes (a) `BlockCausal`-shape attention with (b) original-position-aware copies of conditioning tokens placed at the front of an augmented sequence, where (c) the conditioning tokens form a *bidirectional self-attention cluster* that prevents multi-layer information leakage from later evaluation tokens to earlier ones. Each piece ships; the **composition** with the explicit leakage-prevention discipline does not.

### 2.2 The novel modelless primitive — `AcPrefix` mask builder + sequence augmenter

```text
                augmented sequence layout
┌────────────────────────────┬──────────────────────────────────────┐
│  xc copies (front)         │  full sequence x = xc ∪ xe           │
│  bidirectional self-attn   │  causal attention everywhere         │
│  original_pos propagated   │  loss only on xe                     │
└────────────────────────────┴──────────────────────────────────────┘
```

The distilled primitive is a **zero-allocation attention-mask builder + sequence augmenter**:

```rust
/// AC-GPT-style arbitrary-conditional prefix.
/// Copies `conditioning_positions` to the front of the augmented sequence with
/// their original positions, builds a `[xc-bidirectional | causal-elsewhere]`
/// attention mask, and exposes single-pass conditional likelihood / sampling.
///
/// Modelless: no training. The base model is whatever causal Transformer
/// already ships (GPT-2 small, micro_dllm, game configs). The prefix is a
/// runtime construction over the existing token sequence.
pub struct AcPrefix<'a> {
    /// Borrowed base token sequence (the original x = xc ∪ xe, in original order).
    base_tokens: &'a [u32],
    /// Sorted indices into `base_tokens` marking which positions are in xc.
    conditioning_positions: &'a [usize],
}

impl<'a> AcPrefix<'a> {
    /// Augmented sequence length: |x| + |xc| (the copy doubles the conditioning set).
    pub fn augmented_len(&self) -> usize { /* ... */ }

    /// Original-position lookup for each augmented position.
    /// Used by RoPE to apply the rotation at the *original* token position,
    /// not the augmented position. Zero-alloc — writes into caller's buffer.
    pub fn original_positions_into(&self, out: &mut [usize]) { /* ... */ }

    /// Attention mask builder. Returns a function `attends(i, j) -> bool`:
    ///   - For i, j both in [0, |xc|): always true (bidirectional self-attn in prefix).
    ///   - For i in [|xc|, |x|+|xc|), j in [0, |xc|): always true (eval attends to all copies).
    ///   - For i, j both in [|xc|, |x|+|xc|): causal — `original_pos(i) >= original_pos(j)`.
    /// Branch-free inner loop; SIMD-friendly via bit-packed mask.
    pub fn attends(&self, i: usize, j: usize) -> bool { /* ... */ }

    /// Loss mask: 1.0 for positions in xe, 0.0 for positions in xc (and 0.0 for the copy).
    pub fn loss_mask_into(&self, out: &mut [f32]) { /* ... */ }

    /// Single-pass conditional log-likelihood: forward(augmented) → sum loss over xe.
    pub fn conditional_logprob(/* forward fn */) -> f32 { /* ... */ }

    /// Single-pass conditional sampling: forward(augmented) → sample xe one token at a time
    /// left-to-right, conditioning set fixed.
    pub fn conditional_sample(/* forward fn, rng */) -> Vec<u32> { /* ... */ }
}
```

**Why this is modelless:** the primitive operates on whatever causal Transformer already ships. No new weights, no training, no backprop. It's a **mask builder + sequence augmenter + RoPE-position remapper** that turns a standard causal forward pass into an arbitrary-conditional forward pass. The "fine-tune Qwen3 with LoRA" recipe in the paper redirects to riir-train; what stays here is the runtime construction.

**Why the leakage-prevention matters even modellessly:** the paper's worked example is a *training-time* argument (loss on `x1` shouldn't see `x2`'s gradient path through `x3`). But the same argument applies at inference: if you let later evaluation tokens attend to in-place conditioning tokens, the conditioning signal at layer L leaks future-position information into earlier-position predictions at layer L+1. The copy-at-front discipline is a **correctness invariant**, not just a training trick — single-pass conditional likelihood computed without it is biased.

### 2.3 Latent-space reframing (mandatory per workflow step 3)

Re-cast AC-GPT's mechanism on the six Super-GOAT factory modules:

**(a) HLA per-NPC belief state** (`riir-engine/src/hla/`, `MultiLayerHlaCache`): HLA is a recurrent belief state evolving tick-by-tick. AC-GPT's mechanism reframes as **goal-conditioned belief-state sampling**: "given partial known future latents (e.g., the NPC's emotional state at tick T+5 is observed), sample the latent trajectory between now and T+5 conditioned on that future." This is **counterfactual curiosity** — "how surprising was the actual trajectory given that we know where it ended up?" Closest shipped cousin: Latent Field Steering (additive overlay) — but AC-GPT's mechanism is **attention-mask-disciplined**, not additive. The HLA reframing is real but lands at GOAT-tier because Latent Field Steering already covers ~80% of the use case at lower complexity.

**(b) `latent_functor/` operations** (`reestimation.rs`, `zone_gating.rs`): the functor table predicts `ĥ = source + gate·f`. AC-GPT's prefix-copy mechanism reframes as **hindsight-conditioned functor re-estimation**: when `reestimation.rs` triggers a coherence-driven re-derive cycle, instead of re-deriving from scratch, inject copies of known future latents at the front of the observation buffer with bidirectional self-attention among them, then re-derive the functor with leakage-free conditioning. This is a *better re-estimator*, not a new capability — GOAT-tier.

**(c) `cgsp_runtime/` curiosity signals**: curiosity is the gap between predicted and observed. AC-GPT enables **counterfactual curiosity queries**: "what would the NPC have done given a known future outcome?" Single-pass evaluation of this query (vs iterative MCTS-style rollouts) is a provable speedup. Closest shipped cousin: BoMSampler K-hypothesis planning (samples K diverse belief evolutions). AC-GPT's mechanism is complementary — BoM samples *alternative* futures, AC-GPT conditions *on* a known future.

**(d) LatCal fixed-point commitment** (`riir-chain/src/encoding/latcal*.rs`): the conditioning set `xc` is a latent-space construct (which tokens / which latents are observed). What crosses the sync boundary is the **decision** (a bit-mask or sparse index list of which positions are in `xc`), committed as raw LatCal-fixed values for deterministic replay. The belief scalar projections (valence/arousal/desperation/calm/fear) cross the wire as raw scalars per AGENTS.md; the full augmented sequence stays local. This is a straightforward application of the existing sync-boundary discipline — no new chain primitive needed.

**(e) `NeuronShard` style_weights / dendritic branch** (`riir-neuron-db/src/shard/mod.rs`): shards are fixed Pod blobs. AC-GPT's "position-aware copies at the front" pattern reframes as **shard-level conditioning** — when retrieving a shard for a query, the shard's `style_weights[64]` are a position-aware conditioning vector. But shards already do this (BLAKE3-committed retrieval by zone hash). The neuron-shard reframing is the weakest — no new shard primitive needed.

**Summary of the reframing:** the latent-to-latent reframing is **available** (HLA counterfactual curiosity, functor hindsight re-estimation, cgsp counterfactual queries) but each lands at GOAT-tier because the closest shipped cousin (Latent Field Steering for HLA, Engram for functor re-estimation, BoM for cgsp) already covers the use case at coarser granularity. The genuine novelty is at the **inference primitive** level: the mask builder + sequence augmenter + leakage-prevention discipline. That's a GOAT-tier speedup (single-pass conditional evaluation vs iterative), not a Super-GOAT new-capability-class.

### 2.4 Fusion — what novel combination does this enable?

The 2–3 closest cousins across all five repos, and what fusing them produces:

| Fusion | Cousins | What it produces that none alone can |
|---|---|---|
| **AC-Prefix × Engram × Latent Field Steering** | R295 (this) + R278/P299 + R290/P309 | **Hindsight-conditioned pattern retrieval**: Engram retrieves a pattern by hash; AC-Prefix injects it as a position-aware conditioning set (not additive overlay); Latent Field Steering provides the direction vector. Produces "NPC samples behavior conditioned on a known future outcome AND a retrieved similar past pattern AND a designer-authored steering direction" — three conditioning signals, one forward pass, no leakage. None of the three alone composes all three signals. |
| **AC-Prefix × D2F BlockCausal × RCD Residual** | R295 (this) + P066 + P258/R228 | **Conditional discrete diffusion**: D2F's BlockCausal already does bidirectional-within-block; AC-Prefix adds the leakage-free copy-at-front discipline for the conditioning set; RCD injects residuals from discarded token distributions. Produces a single-pass conditional diffusion sampler (vs MDLM's iterative unmasking). |
| **AC-Prefix × BoMSampler × SpecHop** | R295 (this) + R248/P281 + R091/P131 | **Counterfactual multi-hypothesis speculation**: BoM samples K diverse futures; AC-Prefix conditions the draft on each known future in parallel; SpecHop accepts/rejects per hypothesis. Produces "K hindsight-conditioned draft branches in one forward pass" — strictly more coverage than BoM alone. |

The first fusion (AC-Prefix × Engram × Latent Field Steering) is the strongest — it composes three different conditioning modalities (retrieved past, known future, designer steering) into one leakage-free forward pass. This is a **plausible Super-GOAT candidate** for the riir-ai game runtime (per-NPC counterfactual curiosity at crowd scale), but per the skill rule, I am NOT writing "Super-GOAT candidate" without committing all 4 novelty-gate YES answers in this session. The novelty gate is **borderline**:

- **Q1 (no prior art):** each of the three primitives ships; the **composition** is novel as a combination. Partial.
- **Q2 (new class of behavior):** "three-modality single-pass conditional sampling" is arguably a new capability class, but Latent Field Steering × Engram already gets you ~70% there additively. Borderline.
- **Q3 (product selling point):** "Our NPCs do counterfactual curiosity — sample behavior conditioned on known future outcomes, retrieved past patterns, and designer steering, in a single 20Hz tick, no leakage" — finishable, but Engram × Latent Steering already gives a weaker version of this sentence.
- **Q4 (force multiplier):** ≥5 pillars touched (HLA, functor, cgsp, freeze/thaw, Engram, Latent Steering, speculative decode). YES.

Not confident enough on Q2 + Q3 simultaneously. **Downgrade to GOAT for this session.** If the fusion delivers a measurable quality win over Latent Field Steering × Engram at iso-compute in a future riir-ai integration, file an issue in `katgpt-rs/.issues/` to re-open the Super-GOAT gate with that evidence.

---

## 3. Verdict

**Tier: GOAT.** Plan-only in katgpt-rs behind a `ac_prefix` feature flag. No private guide (not Super-GOAT). The training recipe → riir-train.

**One-line reasoning:** the architectural pieces ship (`BlockCausal`, `LoraPair`, `MixedPrefillSequence::Raw { original_pos }`), the latent-space reframing overlaps with Latent Field Steering + Engram conditional memory at coarser granularity, and the only genuinely new modelless bit — the zero-allocation AC-GPT mask builder + sequence augmenter + leakage-prevention discipline — is a provable **speedup** (single-pass conditional likelihood/sampling vs iterative MLM unmasking), not a new capability class.

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism + new capability class + selling point + force multiplier | (not this) |
| **GOAT** ✓ | Provable gain over existing approach, not a new class. Promotes to default if it wins. | Plan + implement → katgpt-rs. Feature flag `ac_prefix` + benchmark. |
| **Gain** | Incremental. | (below this) |
| **Pass** | Not relevant. | (not this) |

**GOAT gate (must pass before promoting `ac_prefix` to default):**

- **G1 (correctness):** AC-GPT conditional likelihood matches iterative-MLM conditional likelihood on a micro-GPT config, to within float tolerance. (The leakage-prevention discipline is the load-bearing correctness invariant — without it, the conditional likelihood is biased.)
- **G2 (speedup):** single-pass AC-Prefix conditional evaluation ≥ 3× faster than iterative-MLM unmasking at iso-quality on a 128-token sequence with |xc|=64.
- **G3 (no regression):** standard causal forward pass with `AcPrefix::empty()` (empty conditioning set) is bit-identical to vanilla causal forward. Zero-cost when feature is off.
- **G4 (mask-builder alloc-free):** `attends(i, j)` is branch-free and the mask can be materialized into a caller-provided bit-packed buffer with zero heap allocations on the hot path.

If G1–G4 pass → promote `ac_prefix` to default. If G2 fails (no speedup over iterative MLM) → demote to opt-in only, document the negative result.

---

## 4. What This Is NOT

- **Not a training method.** The LoRA fine-tuning of Qwen3 / LLaMA to gain arbitrary-conditional capability is **→ riir-train**. We do not train here.
- **Not a new attention mechanism.** `AttentionMode::BlockCausal` already ships (P066). The novelty is the *mask shape* + *sequence augmentation* + *leakage-prevention discipline*, not the attention kernel.
- **Not a replacement for Engram or Latent Field Steering.** Both ship and cover the conditional-retrieval and top-down-injection use cases. AC-Prefix is a *complementary* conditioning modality (attention-mask-disciplined vs additive-overlay vs hash-addressed-retrieval).
- **Not Super-GOAT.** The latent reframing is available but does not clear Q2 (new capability class) cleanly given existing shipped primitives. The fusion (AC-Prefix × Engram × Latent Field Steering) is a plausible Super-GOAT candidate for riir-ai game runtime but the evidence isn't in yet — file an issue if the GOAT-gate benchmark shows a quality win over the additive baseline.

---

## 5. Cross-references

- **Plan:** `295_AC_GPT_Prefix_Primitive.md` (next free plan slot)
- **Closest shipped cousins:**
  - P025 — Bidirectional Prefill + Modality LoRA Switching (`LoraPair { reader, writer }`)
  - P066 — D2F Discrete Diffusion Forcing (`AttentionMode::BlockCausal`)
  - P238 — MUX-Latent (`MixedPrefillSequence::Raw { token_id, original_pos }`)
  - P299 — Engram Hash-Addressed Pattern Memory (`fuse_into_hidden_state`)
  - P309 — Latent Field Steering (`apply_latent_steering`)
  - P012 Phase 5 — Target-Conditioned Draft (`speculative_step_conditioned`)
- **Closest research cousins:**
  - R269 — Variable-Width `> <former` (same downgrade pattern; latent reframing available, prior art in shipped primitives)
  - R278 — Engram Conditional Memory
  - R290 / riir-ai R153 — Latent Field Steering
  - R248 — BoM K-Hypothesis Sampling (counterfactual cousin)
  - R192 — NextLat Belief-State Drafter
- **Training recipe redirect:** → riir-train (LoRA fine-tuning of pretrained LLMs for arbitrary conditioning)
- **Source paper:** [arXiv:2606.14943](https://arxiv.org/abs/2606.14943) — Lu, Elmoznino, Gagnon, Mittal, Kasetty, Lajoie. Mila, 12 Jun 2026.

## TL;DR

AC-GPT lets a causal Transformer evaluate arbitrary conditionals `p(xe | xc)` in a single forward pass by copying `xc` to the front with original positions + bidirectional self-attention among the copies (to prevent multi-layer leakage from later eval tokens to earlier ones). Training recipe → riir-train. Modelless distillation: a zero-alloc mask builder + sequence augmenter that composes `BlockCausal` + `LoraPair` + `MixedPrefillSequence` into the AC-GPT mask shape. **GOAT** (plan-only, `ac_prefix` feature flag), not Super-GOAT — the architectural pieces ship, the latent reframing overlaps with Latent Field Steering + Engram, and the genuine novelty (leakage-prevention discipline + mask builder) is a provable speedup over iterative MLM unmasking, not a new capability class. Fusion candidate (AC-Prefix × Engram × Latent Field Steering) is a plausible future Super-GOAT for riir-ai game runtime if the GOAT-gate benchmark shows a quality win over the additive baseline.
