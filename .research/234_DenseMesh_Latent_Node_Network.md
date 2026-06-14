# Research: DenseMesh — Latent Node Network for Modelless Inference

**Date:** 2026-06-14
**Status:** Implemented (all 8 phases). **DEMOTED TO EXPERIMENTAL** — Gate 2 FAILED empirically (0/1000 wins with real trained Bomber LoRAs, improvement -0.00%). Gates 1 + 3 + 5 PASS. Gate 4 measured (9.27× single-thread vs 2.5× paper bound). Framework is sound plumbing, but composition gain requires riir-ai R122 trained communication edges.
**Context:** katgpt-rs inference engine — modelless (inference-time only, no LLM training)
**Source Paper:** "Language Model Networks: Supervision-Efficient Learning through Dense Communication" (arXiv:2505.12741, ICML 2026) — Wu, Wang, Yao (Tsinghua / BIMSA)
**Commercial Bound:** Public (katgpt-rs/MIT). Generic trait + topology framework. The trained-edge LoRA recipes stay in riir-ai (R122).

---

## TL;DR

LMNet organises pre-trained LLMs as **nodes** in a directed graph, communicating via **dense hidden-state vectors** instead of natural-language tokens. Intermediate nodes have their embedding/de-embedding layers *stripped* — they never round-trip through the vocabulary. Trainable seq2seq modules sit on every edge and are learned **end-to-end from final-task supervision only** (no intermediate labels). With shared-vertex parameter sharing (1/4/4/4/1 topology, Qwen2.5-0.5B vertex), LMNet beats Prompt by **+30.5%**, beats Self-Consistency ×16 by **+27.0 pp**, and beats LoRA on the E2E benchmark under scarce data.

Our distillation: we **cannot** train the edge modules in katgpt-rs (modelless). The genuinely novel angle is to fuse three pieces we *already have*:

1. **LoRA adapters as edges** (we ship many game LoRAs)
2. **Stripped forward passes** (skip token round-trip between speculative branches)
3. **Topology-aware adaptive width** (Collapse-Aware + Breakeven already detect difficulty)

…into a single generic trait `DenseMesh` + `DenseEdge` that lives in the public engine. The dense hidden-state path between nodes is a *latent* channel (per AGENTS.md latent/raw rules) — only the input and output boundary nodes touch tokens.

---

## Paper Core Insights

### 1. The Paradigm Shift

> "Pre-trained language models can be reused as computational nodes, and intelligence can be improved by learning how information flows among them."

Three orthogonal axes define a language model network:
- **Node function** (vertex): a stripped transformer (embed removed, de-embed removed)
- **Communication medium** (edge): dense vectors vs discrete tokens
- **Topology**: chain / tree / fully-connected / dynamic

Existing test-time-scaling (CoT, Self-Consistency, Self-Refine) is the trivial case: one node talking to itself through tokens. LMNet generalises this to *N nodes* through *dense vectors*.

### 2. Stripped Vertex Construction

Given LLM `f = D ∘ T ∘ E` (de-embed ∘ transformer ∘ embed):

- Naive LLM-to-LLM: `f₂ ∘ f₁ = D₂ ∘ T₂ ∘ E₂ ∘ D₁ ∘ T₁ ∘ E₁` — note `D₁` includes `argmax`, which **cuts gradients** and **loses information**.
- LMNet: `D₂ ∘ T₂ ∘ T₁ ∘ E₁` — only the **first** `E` and **last** `D` remain. Intermediate communication is `X_out → X_in` directly.

### 3. Trainable Seq2Seq Edges

Each edge `e_{l,i→l+1,j}` is a small attention block (single transformer layer, ~5M params). The stripped vertex `T` has never seen raw dense inputs from another vertex, so the edge also acts as an **alignment/translation** module. Multiple distinct edges fan a single predecessor output to multiple successors differently — this is what gives the topology expressive power beyond a chain.

### 4. Layer-wise Fully-Connected Topology

Default: `1 / 4 / 4 / 4 / 1` (input embedding, three hidden layers of width 4, output de-embedding). Aggregation at each vertex is simple **summation** over incoming edge outputs (preserves causal mask). Concatenation is noted as a richer alternative.

### 5. Vertex Parameter Sharing (the cheap trick)

All vertexes share **one** pre-trained LLM `θ_v`. Only edges `ω` differ. Parameter budget: `|θ_v| + L·W²·|ω|` where `|ω| ≪ |θ_v|`. Within a layer, vertexes are batch-parallel (same weights, different inputs). Latency: `L·t_θ + L·W²·t_ω`.

### 6. End-to-End Differentiability & Supervision Efficiency

Gradients flow through dense vectors from final loss `∂L/∂p_o` all the way back to every `θ` and `ω`. **No intermediate message annotations are needed.** The communication protocol is *learned*, not hand-designed.

### 7. Empirical Highlights

| Setting | Baseline | LMNet | Δ |
|---|---|---|---|
| Qwen2.5-0.5B, 11-benchmark avg | Prompt 30.5 | — | **+30.5%** |
| vs Self-Consistency ×16 | +3.4% | +30.5% | +27.0 pp |
| vs Self-Refine ×16 | +0.5% | +30.5% | +30.0 pp |
| vs SFT (same data) | −5.7% | +30.5% | +36.2 pp |
| E2E GPT2-M vs LoRA | rank 2.3 | rank 1.6 | wins |
| GSM8K Qwen2.5-1.5B | 68.5 (Prompt) | 72.7 | +4.2 pp |
| MMLU Qwen2.5-0.5B | 44.3 (Pred) | 53.9 | +9.6 pp |
| Cost | — | <1% of vertex pre-training | — |

---

## What's Training (Paper) vs Inference (Ours)

| Paper Mechanism | Training-Time | Modelless (katgpt-rs) |
|---|---|---|
| Edge seq2seq modules `ω` | ✅ trained end-to-end | ❌ N/A — but **existing LoRA adapters stand in as frozen edges** |
| Vertex parameter sharing | ✅ all `θ` share one LLM | ✅ **we already reuse one model across passes** |
| Stripped intermediate nodes | ✅ remove `E`/`D` | ✅ **hidden-state handoff between speculative branches** |
| Layer-wise fully-connected topology | ✅ fixed 1/4/4/4/1 | ✅ **configurable topology, adaptive width** |
| End-to-end gradient optimisation | ✅ backprop through dense path | ❌ — replaced by **bandit-selected edge routing** |
| Supervision-efficient learning | ✅ final-loss-only | ✅ **reward signal = speculative verifier acceptance** |
| Inner-auto-regressive decoding | future work in paper | ✅ **each "node" is a full forward pass** — natural fit |

---

## Fusion Ideas — Modelless (katgpt-rs)

### F1: DenseMesh — LoRA-as-Edge Network (PRIMARY)

**Core idea.** Treat our existing library of game LoRA adapters as the **edges** of an LMNet-style graph. The frozen base LLM is the **shared vertex**. Multiple inference passes through the same model, each pass conditioned by a different LoRA edge, exchange **hidden states** (not tokens) at topology boundaries.

```
                ┌── LoRA_Bomber ──┐
 input ──► T₀ ──┼── LoRA_Go ──────┼──► T₁ --► T_out --► de-embed --► tokens
                └── LoRA_FFT ─────┘
```

Each `T_i` is a forward pass through the **same** base model (vertex parameter sharing), but with a different LoRA edge active (Bomber / Go / FFT). The hidden state output of `T₀` is routed through 1..N edge-LoRA projections and **summed** at `T₁`'s input (paper's aggregation rule).

**Why it's modelless.** No training. The LoRA edges already exist (trained in riir-ai). The composition topology and routing weights are chosen at inference time by a bandit.

**Why it's novel vs our existing work.**
- *vs ThoughtFold (R175):* ThoughtFold folds **token** chains; DenseMesh routes **hidden states** between passes. Orthogonal — can compose.
- *vs MoA (R126):* MoA mixes activations **within** a single FFN; DenseMesh composes **between** passes.
- *vs MUX-Latent (R158/P238):* MUX compresses context to a latent slot; DenseMesh is a **multi-node topology** with plottable edges.
- *vs MLS (P104):* MLS sums layers **within** one model; DenseMesh sums edges **between** virtual nodes.
- *vs SubstrateGate (P216):* SubstrateGate picks **one** substrate per query; DenseMesh **composes multiple** substrates through a topology.

**Expected gain.** The paper's frozen-vertex + trained-edge setting is precisely this configuration. Their reported gains (+30.5% over Prompt, beats LoRA on E2E) are the upper bound when edges are well-trained. Even with imperfect edges (our game LoRAs were trained for single-game, not for inter-game communication), we expect:
- Multi-game queries (e.g., "play Go then Bomber") → meaningful composition gain
- Hard single-game queries → adaptive depth (route through more nodes) → +5–15% quality at ≤2× cost
- Easy queries → collapse to 1/1/1 chain → zero overhead

### F2: Stripped Hidden-State Handoff

**Core idea.** In our existing speculative pipeline (drafter → verifier), the drafter currently **decodes tokens** and the verifier **re-embeds** them. This is exactly the wasteful `D₁ ∘ E₂` round-trip LMNet removes. Replace with: drafter emits its final hidden state, verifier consumes it directly.

**Landing.** New `HiddenHandoff` variant on the speculative generator trait — drafter returns `Vec<f32>` (last hidden) instead of `Vec<u32>` (tokens). Verifier's forward pass seeds its input embedding from the handoff buffer.

**Why it matters.** Eliminates one `argmax` information bottleneck per drafter→verifier hop. For long speculation chains (DDTree depth > 4), this compounds. Also eliminates a tokenisation pass — pure win on latency.

**Latent/Raw compliance.** The handoff buffer is a **latent** channel (semantic domain). It never crosses `SyncBlock` / chain quorum. The verifier's output tokens are **raw** (physical domain) and commit normally. Bridge: `HiddenHandoff → token` happens inside the verifier's de-embed only.

### F3: EdgeBandit — Topology & Edge Selection

**Core idea.** We can't train edges, but we can **select** them. `EdgeBandit` is a Thompson-sampling bandit over `(topology_shape, active_edge_set)` pairs. Reward = speculative acceptance rate × quality proxy.

- Arm space: `{chain(1/1/1), diamond(1/2/1), wide(1/4/4/1)}` × subsets of available LoRA edges
- Reward: verifier acceptance + downstream task signal (win/lose for games, BLEU for dialogue)
- Update: existing `ThinkingBandit` infrastructure, one new arm family

**Why it's self-learning (constraint 4).** No gradient, but the bandit adapts the communication topology online. Over a session, the mesh "learns" which LoRA composition works for which query class — purely from reward.

### F4: Adaptive Width via Collapse-Aware + Breakeven

**Core idea.** The paper fixes topology at 1/4/4/4/1. We can do better: use our existing `CollapseAwareThinking` (P212) and `BreakevenRouter` (P250) to pick width per-query.

- Easy query (low entropy, high verifier confidence) → `chain 1/1/1` (single pass, CPU)
- Medium query → `diamond 1/2/1` (CPU + 1 GPU branch)
- Hard query (entropy spike, low confidence) → `wide 1/4/4/1` (full GPU parallel)

This gives the **benefit** of LMNet's depth on hard queries and **zero cost** on easy ones — something the paper doesn't do because it trains a fixed topology.

### F5: CPU / GPU / ANE Auto-Route by Topology Width

**Core idea.** The paper notes that within-layer vertexes are data-parallel (same weights, different inputs). Map this to our heterogeneous compute:

| Layer | Width | Compute Target | Rationale |
|---|---|---|---|
| L0 (input embed) | 1 | CPU | single small op, latency-bound |
| L1 (hidden) | 1–4 | GPU | parallel branches, throughput-bound |
| L2 (hidden) | 1–4 | GPU | parallel branches |
| L3 (hidden) | 1–4 | GPU | parallel branches |
| L4 (output de-embed) | 1 | ANE | final decode, latency-sensitive, ANE wins on fixed-shape decode |

Threshold-governed: if `width == 1`, stay on CPU (no GPU launch overhead, per optimisation.md ~50μs launch cost). If `width ≥ 4`, go GPU. Final decode always ANE on Apple Silicon (per R155/R223 verdicts).

This satisfies **constraint 7** (CPU/GPU/ANE auto-route) and **constraint 9** (threshold-based switching).

---

## Plasma / Hot / Warm / Cold / Freeze Mapping (constraint 8)

| Tier | Artifact | Lifetime |
|---|---|---|
| **Plasma** | `HiddenHandoff` scratch buffer (reused `Vec<f32>` per thread) | per-call, `clear()` + reuse |
| **Hot** | `DenseMesh::forward_dense()` inner loop — zero-alloc, SIMD-chunked | per-token |
| **Warm** | `EdgeBandit` arm selection — microsecond Thompson sample | per-query |
| **Cold** | Topology config + LoRA edge registry — loaded once at startup | per-session |
| **Freeze** | Base LLM weights + LoRA edge weights — never mutated at inference | immutable |

**Chain security alignment.** Dense hidden states flowing between nodes are **latent-encoded** (semantic domain). Raw values (token outputs, balances, positions) appear **only** at input/output boundary nodes. This matches the paper's design (only first `E` and last `D` exist) and our AGENTS.md sync-boundary rule (latent inside, raw at boundary, bridge functions at the seam).

For chain validators-as-nodes: each validator is a DenseMesh node. Inter-validator dense state diffs are latent (dot-product + sigmoid projection onto learned directions, **never softmax**). Commit-to-chain happens only at the output boundary node — raw values, BLAKE3-committed, quorum-gated. **Raw sync correctness is never feature-flagged** (AGENTS.md anti-pattern).

---

## Latent / Raw Space Compliance (AGENTS.md)

| Data | Domain | Treatment |
|---|---|---|
| Drafter hidden state → verifier | **latent** (semantic) | dense vector, never tokenised, never synced |
| LoRA edge projection output | **latent** | per-edge transformation, local to mesh |
| Final de-embedded tokens | **raw** (physical) | committed, replayable, anti-cheat-validatable |
| EdgeBandit reward signal | **raw** scalar | verifier acceptance ∈ [0,1], deterministic |
| Inter-validator state diff (chain) | **latent** | sigmoid projection of dense state, **not** the dense state itself |
| Committed chain transaction | **raw** | exact values, BLAKE3-hashed, quorum commit |

**Anti-patterns avoided:**
- ❌ Never encode `MapPos` as a DenseMesh hidden state then decode for sync — lossy round-trip breaks quorum.
- ❌ Never validate a movement claim by latent similarity — deterministic replay needs exact (x, y).
- ❌ Never send the full hidden vector over the chain network when a scalar projection suffices.

---

## SOLID / DRY Compliance

- **SRP:** `DenseNode` does forward-pass; `DenseEdge` does routing; `DenseMesh` does topology; `EdgeBandit` does selection. Four traits, four responsibilities.
- **OCP:** New edge types (identity, LoRA, projection, attention-head-select) plug in via `DenseEdge` impl — no mesh changes.
- **LSP:** Any `DenseNode` impl works in any topology slot. Identity edge is a valid no-op edge.
- **ISP:** `DenseNode` and `DenseEdge` are separate small traits, not one fat trait.
- **DIP:** `DenseMesh` depends on `DenseNode` / `DenseEdge` traits, not concrete LoRA or transformer.
- **DRY:** The stripped-forward primitive is defined once and reused by both speculative (F2) and mesh (F1) paths.

---

## Verdict

### ❌ DEMOTED TO EXPERIMENTAL — gate 2 gain NOT achievable without trained edges

**Empirical proof (2026-06-14).** Gate 2 REAL test (`test_dense_mesh_gate2_real_lora_composition_gain`) loaded 3 trained Bomber LoRAs from `riir-train/output/`:
- A = `game_lora_baseline.bin`, B = `game_lora_echo.bin`, T = `game_lora_v2_moa.bin`
- 1000 random inputs, measured L2 distance to target T

Result: **diamond (A+B)/2 never beats best single** — 0/1000 wins, improvement -0.00%.

```
LoRA_A (baseline): 0.964656
LoRA_B (echo):     0.964604
Diamond (A+B)/2:   0.964630  ← between A and B
Diamond beats BOTH: 0/1000 (0.0%)
```

This is the mathematical property of averaging independently-trained predictors: `|(A+B)/2 - T|` can only beat `min(|A-T|, |B-T|)` when A and B are on opposite sides of T. With correlated outputs (same domain, similar training), this never happens.

**The modelless hypothesis was wrong.** Reusing game LoRAs as edges does not produce composition gain. The paper's +30.5% required **trained communication edges** — modules specifically trained end-to-end to make inter-node composition beneficial. Without that training, DenseMesh is a no-op ensemble.

### DenseMesh framework status

The **framework itself is sound** (proven by gates 1, 3, 5):
- Gate 1 ✅ correctness (`[1,1]`+IdentityEdge == vanilla)
- Gate 3 ✅ easy overhead (0.997× at production scale — framework cost is negligible)
- Gate 5 ✅ bandit convergence
- Gate 2 ❌ composition gain (untrained edges produce no improvement — this test)
- Gate 4 ⚠️ hard bound (9.27× single-thread vs 2.5× bound — needs vertex parallelism)

The value proposition collapses to: **if riir-ai R122 trains dedicated communication edges, this framework can route them.** Without trained edges, DenseMesh is plumbing without water.

### Action

- Keep `dense_mesh` feature as **opt-in, experimental** (not default, not promoted)
- **Pivot riir-ai R122** from "reuse game LoRAs" to "train dedicated communication-edge LoRAs"
- Gate 2 cannot be revisited until R122 produces trained comm edges
- DenseMesh's `adaptive_width` controller (CollapseAware + BreakevenRouter) remains useful as a generic compute-gating primitive independent of edge composition

### Commercial Bound

| Artifact | Repo | Why |
|---|---|---|
| `DenseNode`, `DenseEdge`, `DenseMesh` traits | katgpt-rs (MIT) | generic framework hooks — adoption value, like `ConstraintPruner` |
| Topology config types, `EdgeBandit` | katgpt-rs (MIT) | generic bandit infra |
| Stripped-forward / hidden-handoff primitive | katgpt-rs (MIT) | generic speculative improvement |
| **Which specific LoRA edges to compose for Go/Bomber/FFT** | riir-ai (private) | the fuel — composition recipe is IP |
| **Trained communication-edge LoRAs** (riir-ai R122) | riir-ai (private) | the actual gain-driver |

---

## Tests / Examples (constraint 6)

### Before vs After — Thinking vs Non-Thinking

```
test_dense_mesh_correctness:
  - vanilla:   forward(input) → output_v
  - chain:     DenseMesh[topology=1/1/1, edge=Identity].forward(input) → output_c
  - assert output_v == output_c  (gate 1)

test_dense_mesh_multi_game_composition:
  - single:    SubstrateGate.pick(Go-LoRA).forward(go_position) → win_rate_single
  - mesh:      DenseMesh[topology=1/2/1, edges={Go-LoRA, FFT-LoRA}].forward(go_position) → win_rate_mesh
  - assert win_rate_mesh >= win_rate_single + 0.03  (gate 2)

test_dense_mesh_easy_query_zero_overhead:
  - easy input (low entropy)
  - measure: vanilla_latency, mesh_latency
  - assert mesh_latency <= vanilla_latency * 1.05  (gate 3)

test_dense_mesh_hard_query_bound:
  - hard input (entropy spike)
  - measure: vanilla_latency, mesh_latency_at_width_4
  - assert mesh_latency_at_width_4 <= vanilla_latency * 2.5  (gate 4)

test_edge_bandit_convergence:
  - 3 query classes, optimal topology per class
  - run 200 queries, track regret
  - assert cumulative_regret < O(log T * sqrt(N_arms))  (gate 5)
```

### Benchmark Format

```
[DenseMesh] topology=1/4/4/1 edges={Go,Bomber,FFT,Civ} latency=12.4ms quality=+8.2pp
[DenseMesh] topology=1/2/1   edges={Go,FFT}             latency=4.1ms  quality=+3.1pp
[DenseMesh] topology=1/1/1   edges={Identity}           latency=1.2ms  quality=+0.0pp
[Vanilla]   SubstrateGate                              latency=1.1ms  quality=baseline
```

---

## CPU / GPU / ANE Auto-Route (constraint 7)

```rust
fn pick_compute(width: usize, layer_role: LayerRole) -> ComputeTarget {
    match (width, layer_role) {
        (1, LayerRole::Input)  => ComputeTarget::Cpu,   // single embed, latency-bound
        (1, LayerRole::Hidden) => ComputeTarget::Cpu,   // no parallelism to exploit
        (w, LayerRole::Hidden) if w >= 4 => ComputeTarget::Gpu,  // data-parallel branches
        (_, LayerRole::Output) => ComputeTarget::Ane,   // final decode, ANE-optimal
        _ => ComputeTarget::Cpu,
    }
}
```

Threshold: GPU dispatch kicks in only when `width ≥ 4` (per optimisation.md ~50μs launch overhead — must amortise across ≥ 4 parallel branches). ANE takes the final decode layer unconditionally on Apple Silicon (per R155).

---

## Relationship to Existing Work

| Existing | Overlap | DenseMesh's Novelty |
|---|---|---|
| ThoughtFold (R175, P195) | Both are multi-pass inference | ThoughtFold folds **tokens**; DenseMesh routes **hidden states** between nodes. Composable. |
| MoA (R126, P158) | Both mix representations | MoA mixes within one FFN; DenseMesh composes **between** passes. |
| MUX-Latent (R158, P238) | Both use latent representations | MUX compresses context; DenseMesh is a multi-node **topology** with edges. |
| MLS (P104) | Both aggregate across layers | MLS sums within one model; DenseMesh sums **edges between virtual nodes**. |
| SubstrateGate (P216) | Both route to capability substrates | SubstrateGate picks **one**; DenseMesh **composes many** through topology. |
| NextLat (R192, P217) | Both use latent belief states | NextLat is single-drafter belief; DenseMesh is multi-node. NextLat can be **one node** inside a DenseMesh. |
| BreakevenRouter (P250) | Both adapt compute by difficulty | Breakeven picks tier pair; DenseMesh picks **topology width**. Compose: Breakeven decides *if* to expand, DenseMesh decides *how*. |
| SHINE Hypernetwork (R062) | Both generate adapter weights | SHINE generates one LoRA from context; DenseMesh composes many LoRAs. Orthogonal. |

---

## References

- Paper: arXiv:2505.12741 (LMNet, ICML 2026)
- Companion (model-based, trained edges): riir-ai R122 EdgeLoRA Dense Network Training
- Related katgpt-rs research: R126 MoA, R158 MUX-Latent, R175 ThoughtFold, R192 NextLat
- Related katgpt-rs plans: P104 MLS, P212 Collapse-Aware, P216 SubstrateGate, P250 Breakeven
- AGENTS.md: latent/raw domain rules, plasma/hot/warm/cold/freeze, bridge functions

---

## TL;DR (final)

**Empirically DEMOTED to experimental.** The modelless hypothesis (reuse trained game LoRAs as DenseMesh edges) was tested with real Bomber LoRAs (`game_lora_baseline.bin` + `game_lora_echo.bin` vs `game_lora_v2_moa.bin` target). Diamond (A+B)/2 composition never beat the best single LoRA — 0/1000 samples, improvement -0.00%. The paper's +30.5% gain requires trained communication edges, not repurposed game LoRAs.

The framework itself is sound: gates 1 (correctness), 3 (0.997× overhead), and 5 (bandit convergence) all pass. The value proposition collapses to **plumbing**: the topology engine, edge traits, and adaptive width controller are correct and reusable, but **without trained edges (riir-ai R122), DenseMesh produces no composition gain**.

Keep `dense_mesh` opt-in and experimental. Pivot riir-ai R122 from "reuse game LoRAs" to "train dedicated communication-edge LoRAs". Revisit gate 2 only after R122 delivers trained edges. The `adaptive_width` controller remains independently useful as a generic compute-gating primitive.
