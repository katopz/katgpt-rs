# Research 258: Attention Sinks — Two Algorithms (NOP vs Broadcast), Two Solutions

> **Source:** [A Unifying View of Attention Sinks: Two Algorithms, Two Solutions](https://arxiv.org/pdf/2606.08105) — Fesser, Jacobs, Fel, Keller, Kakade (Kempner Institute, Harvard) — arXiv:2606.08105v1, 2026-06-09
> **Date:** 2026-06-17
> **Status:** Active
> **Related Research:** 100 (EGA — spectral salience gate), 070 (GDN2 — decoupled erase/write), 018 (Free Transformer — latent injection), 113 (NITP), 125 (Weight Norm Kolmogorov), 140 (sigmoid parallax), 213 (Still Perceiver), 233 (Attention Matching)
> **Related Plans:** 271 (attention matching compaction), 269 (chiaroscuro), 279 (manifold power iter MoE router)
> **Classification:** Public

---

## TL;DR

The same "vertical stripe" attention pattern (a sink) can implement two **distinct algorithms**: *Adaptive NOP* (sink has `||v_s|| ≈ 0`, suppresses residual update) vs *Broadcast* (sink has meaningful `v_s`, writes a rank-1 update `O ≈ a_s v_s^T` to many positions). The paper gives two clean diagnostics — **value-norm ratio** and **stable rank of attention update** — that separate the regimes in pretrained ViTs, and shows that *gating* (kills NOP) and *register tokens* (support Broadcast) are **complementary**, not redundant.

**Distilled for katgpt-rs (modelless, inference-time):**

We already ship half of the paper's conclusion by default — sigmoid attention (`parallax_attn.rs`, `funcattn.rs`) eliminates sinks by replacing softmax. But the paper's value is the **discrimination**: some sinks are useful broadcasters doing real computation. A blanket "kill all sinks" policy over-suppresses. The transferable primitive is a **per-head/per-sink classifier** that decides whether a sink should be killed (NOP) or preserved (Broadcast), plus a **dual-policy attention**: sigmoid gate for NOP heads, regular attention for Broadcast heads. This is a clean extension of our `data_probe/geometry.rs` diagnostics from whole-layer to per-sink scope, fused with our existing sigmoid-attention intervention.

---

## 1. Paper Core Findings

### 1.1 The two algorithms

The paper's central claim: visually identical "attention sink" patterns (a column in the attention map receiving disproportionate mass from many queries) can implement two distinct computations.

| Property | Adaptive NOP | Broadcast |
|---|---|---|
| Role | "trash can" — suppress residual update | "coffee cup" — communication hub |
| Sink value norm `‖v_s‖` | ≈ 0 | ≈ content-scale (≈ 5.5 in toy model) |
| Residual update `x_i^{ℓ+1} − x_i^ℓ` | ≈ 0 | ≈ `A_is · v_s` (shared component) |
| Output matrix `O = AV` rank | (suppressed — no rank signature) | rank-1: `O ≈ a_s v_s^T` |
| Token similarity effect | none | increases (Lemma 4 — variance reduces) |
| Adaptivity mechanism | spectral spike in `W_Q W_K^T` *or* massive activation `‖x_s‖` | rank-1 query subspace (queries collapse to universal query) |
| Right intervention | **gating** (`O ⊙ σ(X W_θ)`) | **register tokens** (dedicated workspace slots) |

### 1.2 The diagnostics (the transferable primitive)

For a sink position `s` and query set `I`:

```
sink(s; I) = (1/|I|) Σ_i A_is                              # sink strength (Definition 1)
value_norm_ratio(s) = ‖v_s‖ / mean_i(‖v_i‖)                # NOP if < 0.2, Broadcast if ≈ 1
stable_rank(O) = (Σ σ_k)^2 / Σ σ_k^2                        # on the attention update O = AV
                                                              # Broadcast → ≈ 1
                                                              # NOP → output is zero so rank is irrelevant
```

Where `σ_k` are singular values of the per-head attention output `O ∈ ℝ^{n × d_h}`.

### 1.3 The lemmas (proofs in paper §B)

- **Lemma 1 (NOP uniqueness):** At a perfect sink (`ε = 0`), `‖v_s‖ = 0` is the *unique* solution. Cancellation solutions only exist at `ε > 0`.
- **Lemma 2 (Adaptivity):** Hard gate (`A_is = 1`) requires `σ · γ_s ≫ M √d_h / λ_i` where `σ = ‖Θ‖_2`, `γ_s = ‖x_s‖`, `Θ = W_Q W_K^T`. Two regimes: **spectral gating** (large `σ`) or **massive activation** (large `γ_s`).
- **Lemma 3 (Broadcast rank-1):** If `s` is an `ε`-broadcast sink, `O − a_s v_s^T` has `‖·‖_∞ ≤ ε · max_j ‖v_j‖`. So `O` is approximately rank-1 with dominant singular value `σ_1 ≥ √n (1−ε) ‖v_s‖`.
- **Lemma 4 (Token similarity):** Broadcast reduces variance: `Var(x^{ℓ+1}) ≤ Var(x^ℓ) + O(‖v_s‖^2 · V(a_s)) + O(ε)`. When attention is uniform across queries (`V(a_s) = 0`), variance is preserved but all tokens shift collectively along `v_s`.

### 1.4 Empirical phenomenology (DINOv2-G, OpenCLIP-L, EVA-G)

1. **Phase transition:** `[CLS]` is the sink in early layers (NOP — model is protecting CLS from saturation), patches become sinks in deeper layers (Broadcast — distributing semantic content).
2. **Head specialization:** sink behavior is sparse — some heads sink for ~80% of inputs, adjacent heads never do.
3. **Dual phenomenology in same model:** projecting all detected sinks onto (value-norm-ratio, stable-rank-of-update) reveals two distinct clusters — both mechanisms co-exist.
4. **Registers absorb both:** register tokens (designed for Broadcast) get repurposed for NOP too. They move where the sink lives but don't eliminate either mechanism.
5. **Combination beats either alone:** Gated Attention + Registers → ADE20K 0.200 (vs 0.166 baseline, 0.187 registers-only, 0.186 gating-only). ImageNet classification unchanged (~69%). The gain is in **dense spatial representations**, not global classification.

---

## 2. Distillation

### 2.1 Transferable primitive — `AttentionSinkClassifier`

The distilled primitive is a **classifier**, not an attention modification. Given an attention map `A ∈ ℝ^{n×n}` and value matrix `V ∈ ℝ^{n×d_h}` for a single head, classify each high-mass column as one of `{None, NOP, Broadcast}`:

```rust
pub enum SinkKind { None, Nop, Broadcast }

pub struct SinkDiagnostic {
    pub position: usize,
    pub strength: f32,           // sink(s; I) — mean attention mass
    pub value_norm_ratio: f32,   // ‖v_s‖ / mean(‖v_i‖)
    pub update_stable_rank: f32, // stable rank of O = AV (per-head)
    pub kind: SinkKind,
}

// Decision rule (paper §4):
//   strength > τ_sink (e.g. 0.5)        → candidate sink
//   value_norm_ratio < 0.2              → Nop
//   value_norm_ratio ∈ [0.5, 1.5] AND
//     update_stable_rank < 1.5          → Broadcast
//   otherwise                           → None / ambiguous
```

This is a clean extension of our existing `data_probe/geometry.rs` (`effective_rank`, `avg_cosine_similarity`) from whole-layer scope to **per-sink-position scope**. The existing whole-layer metrics are the *aggregate symptom* (broadcast sinks reduce effective rank across all tokens); the per-sink metrics are the *mechanism locator*.

### 2.2 Intervention fusion — Dual-Policy Attention

Our default sigmoid attention (`parallax_attn.rs`, `funcattn.rs`) already eliminates sinks as a side effect of replacing softmax. But the paper shows this is **over-suppression**: Broadcast sinks are doing useful work (rank-1 global information distribution). The fusion is a *per-head* dual policy:

```
For each head h:
    compute SinkDiagnostic for the dominant sink
    if kind == Nop:
        apply sigmoid gate  g_h = σ(X W_θ)         # kill the suppression — but it was already useless
    elif kind == Broadcast:
        preserve softmax attention                 # keep the rank-1 write — it's load-bearing
    else:
        default (sigmoid per AGENTS.md)
```

This is a **diagnostic-driven attention policy**. It generalizes our existing `ega_attn` (Research 100 — spectral salience gate) from a *uniform gate* to a *categorically conditioned gate*. EGA gates all keys by spectral energy; dual-policy gates *only the NOP heads* and leaves Broadcast heads untouched.

### 2.3 Fusion

The three closest cousins across both repos, and the novel combination:

| Cousin | Repo | What it ships | Relation to this paper |
|---|---|---|---|
| **Research 100 (EGA)** + `ega_attn` feature | katgpt-rs | Spectral salience gate on attention output | Same intervention family (gating), but EGA is uniform — doesn't distinguish NOP from Broadcast. This paper provides the *categorization* EGA lacks. |
| **Research 070 (GDN2)** + `deltanet_inference` | katgpt-rs | Decoupled **erase** (NOP) and **write** (Broadcast) gates in *linear* attention | Exact dual-mechanism analog, but in linear-attention form. This paper provides the same duality for *softmax* attention with cleaner diagnostics. |
| **`data_probe/geometry.rs`** | katgpt-rs | Whole-layer `effective_rank` + `avg_cosine_similarity` | Aggregate symptom detector. This paper's per-sink diagnostics are the mechanism locator. |
| **Research 018 (Free Transformer)** | katgpt-rs | Mid-layer latent injection | Broadcast-like mechanism. The Free Transformer's Z injection is essentially an explicit broadcast sink — confirming the paper's "broadcast is useful computation" thesis from a different angle. |
| **riir-ai Guide 134 (SwiR Think/Info Brain)** | riir-ai | Two-brain spatial cognition: info brain (raw, synced) vs think brain (latent belief, fog-of-war) | The Think brain's `SpatialBelief` (zone-level KG triple read by many NPCs) is a *cross-NPC broadcast sink* — many NPC latent states acquire the same zone direction vector → rank-1 update across the crowd. |

**Novel combination (fusion idea — novelty TBD, needs Q1–Q4 check before verdict):**

*Sink-classifier × Two-Brain × HLA latent functor* → **Crowd-scale broadcast detection for MMORPG AI**. At 20Hz with thousands of NPCs, compute per-zone: (a) which zone-attention heads are NOP (NPCs suppressing zone signal — busy with their own LEO goal) vs Broadcast (NPCs aggregating zone signal — e.g., reacting to mayor's tax policy broadcast). The stable-rank-of-update across the crowd's latent states is a real-time *zone-coherence signal*: low rank = crowd is synchronized (broadcasting), high rank = crowd is dispersed (each NPC doing their own thing). This is a **crowd-level curiosity signal** that the existing per-NPC curiosity (`cgsp/derivative_curiosity.rs`) cannot see — it's an emergent property of the crowd, not any single NPC.

This fusion is interesting but does not pass the Super-GOAT novelty gate (Q1 fails — too much prior art in our own codebase; Q3 weak — selling point is incremental over existing curiosity signals). It's a **GOAT-tier plan candidate**, not a Super-GOAT.

### 2.4 What does NOT transfer

| Paper element | Why it stays out |
|---|---|
| Toy-model training experiments (§3.1, §3.2, §A) | Training-only. → riir-train if anyone cares. |
| LeJEPA ViT-L training ablations (§5, §D) | Training-only. |
| Register-token training recipe | Requires re-training the base model. Our frozen-base modelless constraint (AGENTS.md) rules this out. We can *simulate* register tokens at inference time (reserved KV slots), but cannot train them. |
| Spectral spike in `W_Q W_K^T` as a *training* signal | Same — base weights are frozen. We can *measure* the spike as a diagnostic but not enforce it. |

---

## 3. Verdict

**🟢 GOAT — Plan + feature flag + benchmark.**

### One-line reasoning

The diagnostics (value-norm ratio + stable-rank-of-update) are a clean, novel-per-scope extension of our existing `data_probe/geometry.rs`, and the dual-policy attention is a categorically-conditioned generalization of our existing `ega_attn` — provable gain over uniform gating, but not a new capability class.

### Why NOT Super-GOAT

| Novelty gate question | Answer |
|---|---|
| Q1: No prior art? | **NO.** Sigmoid attention (kills sinks), EGA (spectral gate), GDN2 (decoupled erase/write), `data_probe/geometry.rs` (effective rank + cosine sim) all ship in our codebase. The paper's per-sink scope is novel *relative to our diagnostics*, but the mechanism family is well-covered. |
| Q2: New class of behavior? | **NO.** It's a refinement of existing diagnostic + intervention families, not a new capability. |
| Q3: Product selling point? | **WEAK.** "Our attention distinguishes suppression from broadcast" — true but niche. Doesn't finish the "NPCs do X no competitor can" sentence strongly. |
| Q4: Force multiplier? | **YES** (connects to Two-Brain, HLA, EGA, GDN2, curiosity signals) — but Q1–Q3 fail, so not enough for Super-GOAT. |

### GOAT gate (must beat baseline to promote)

Implement `AttentionSinkClassifier` + dual-policy attention behind `sink_aware_attn` feature flag. Benchmark vs default sigmoid attention on:

- **G1 (correctness):** classifier labels match hand-built ground truth on synthetic NOP-only, Broadcast-only, and mixed heads.
- **G2 (quality):** on a frozen ViT-style model (or our `percepta` test bed), dual-policy attention preserves or improves `effective_rank` vs uniform sigmoid (because Broadcast sinks are no longer over-suppressed).
- **G3 (latency):** overhead of computing value-norm-ratio + stable rank per head ≤ 5% of attention forward time. Stable rank is the expensive part — `O(n · d_h)` SVD per head; mitigate via power iteration (we already ship `manifold_power_iter_router`).
- **G4 (game integration):** crowd-level zone-coherence signal in `riir-games` correlates with hand-labeled "synchronized crowd" events (e.g., mayor broadcast → tax payment spike).

If G2 passes → promote `sink_aware_attn` to default in `parallax_attn` and `funcattn`. If G2 fails → demote, keep as opt-in diagnostic only (still useful for debugging via `data_probe`).

### Routing

| Artifact | Repo | Path |
|---|---|---|
| `AttentionSinkClassifier` primitive + `stable_rank_update` math | katgpt-rs (public, MIT) | `crates/katgpt-core/src/data_probe/sink_classify.rs` (new) + extend `geometry.rs` |
| Dual-policy attention (gating conditioned on sink kind) | katgpt-rs (public) | extend `parallax_attn.rs` + `funcattn.rs` behind `sink_aware_attn` feature |
| Plan | katgpt-rs | `.plans/283_sink_aware_attention.md` |
| Crowd-level coherence signal (fusion) | riir-ai (private) | not planned yet — file as an issue if G4 becomes interesting |
| Game-domain τ thresholds per head class | riir-ai (private, if it materializes) | TBD |

---

## References

- Fesser, L. et al. (2026). *A Unifying View of Attention Sinks: Two Algorithms, Two Solutions*. arXiv:2606.08105.
- Darcet, T. et al. (2023). *Vision Transformers Need Registers*. arXiv:2309.16588. (Broadcast intervention)
- Qiu, Z. et al. (2025). *Gated Attention for Large Language Models*. (NOP intervention)
- Roy, V. & Vetterli, M. (2007). *The effective rank: a measure of effective dimensionality*. (Our existing `effective_rank` in `data_probe/geometry.rs`.)
- Sandoval-Segura, P. et al. (2025). *Using attention sinks to identify and evaluate dormant heads in pretrained LLMs*. (NOP = "dormant head" — directly maps.)

---

## Cross-link disambiguation (Plan 306 T8.3)

**Do not confuse this paper (arXiv:2606.08105, target-side sink classification)
with arXiv:2605.09992 (Eldenk et al., *Attention Drift* — drafter-side
magnitude accumulation in the recursive residual, Plan 306, Research 286).**
Different paper, different mechanism. The two are frequently confused because
both diagnose "attention drift" phenomenologically but operate on different
sides of the speculative-decoding loop:

| Aspect | This paper (Plan 287 / Research 258) | arXiv:2605.09992 (Plan 306 / Research 286) |
|---|---|---|
| Side | Target model (verifier) | Drafter (speculator) |
| Mechanism | Sink classification (NOP vs Broadcast) | Recursive residual magnitude accumulation |
| Diagnostic | `value_norm_ratio` + `stable_rank_of_update` per head | `magnitude_slope` on the hidden-state chain |
| Fix | Dual-policy attention | Post-norm on the recursive residual (retrain for frozen MLPs) |

See `.research/286_Attention_Drift_Depth_Invariance_Diagnostic.md` and
`.plans/306_depth_invariance_diagnostic.md` for the drafter-side counterpart.
